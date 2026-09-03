//! The keyboard: one reader, and the line editor it feeds.
//!
//! rustyline used to own both, and it owned them by blocking: nothing else could
//! happen while a line was being typed, and nothing could be typed while anything
//! else was happening. Reading the keyboard on its own thread and handing the events
//! to an editor we drive ourselves is what makes the second half possible, and it is
//! also the only way to get an input box that grows over more than one line.

use crate::modes::Mode;
use crate::out;
use crate::tools::TodoItem;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind};
use crossterm::style::{Color, ResetColor, SetBackgroundColor, SetForegroundColor};
use crossterm::tty::IsTty;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc::{error::TryRecvError, unbounded_channel, UnboundedReceiver, UnboundedSender};

// The one thing that reads the keyboard.
//
// Raw mode is on for the whole session now, so a second reader anywhere would race
// this one for keystrokes and each would get an arbitrary half of what was typed.
// Everything that needs a key — the permission prompt, the plan review, the model
// picker — takes events from here instead of calling `event::read` itself.
pub(crate) struct Keys {
    events: UnboundedReceiver<Event>,
}

impl Keys {
    // `vt_input` says the console is delivering everything as escape sequences, which on
    // Windows is the price of the mouse reporting anything at all. Crossterm's Windows
    // reader has no escape parser — it maps one input record to one event — so in that
    // mode the byte stream has to be read and parsed here instead.
    pub(crate) fn start(vt_input: bool) -> Self {
        let (sender, events) = unbounded_channel();

        #[cfg(windows)]
        if vt_input {
            vt::start(sender);
            return Keys { events };
        }
        #[cfg(not(windows))]
        let _ = vt_input;

        // A plain thread rather than a task: `event::read` blocks, and blocking a
        // tokio worker would stall every other future on that thread. It ends when
        // the receiver goes, which is when the process is on its way out anyway.
        std::thread::spawn(move || loop {
            match event::read() {
                Ok(event) => {
                    if sender.send(event).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        });
        Keys { events }
    }

    pub(crate) async fn next(&mut self) -> Option<Event> {
        self.events.recv().await
    }

    // Throw away anything typed ahead. A prompt that has just been answered should not
    // also consume the keystrokes that arrived while the user was reading it.
    pub(crate) fn drain(&mut self) {
        while !matches!(self.events.try_recv(), Err(TryRecvError::Empty | TryRecvError::Disconnected)) {}
    }
}

pub(crate) fn stdin_is_a_terminal() -> bool {
    io::stdin().is_tty()
}

// Piped input has no keyboard: no raw mode, no region, no editing. Kept because
// scripting haico — feeding it a file of commands — is worth not breaking, and
// because the alternative is a program that only works when a human is watching.
pub(crate) fn read_piped_line() -> Option<String> {
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line.trim_end_matches(['\r', '\n']).to_string()),
    }
}

// What is typed after. The mode used to be spelled out here as well, which was one
// place too many once the status bar carries it.
pub(crate) const PROMPT: &str = "> ";

// What came out of the editor.
pub(crate) enum Input {
    Text(String),
    // Ctrl+C: the half-typed line is abandoned, the session is not.
    Interrupted,
    // Ctrl+D on an empty line, or the keyboard going away.
    Eof,
    // Shift+Tab. The editor does not own the mode, so it says so and lets the caller
    // decide whether cycling means anything where it is being used.
    CycleMode,
    // Ctrl+T, and the same arrangement: the editor does not own the plan panel either,
    // and the prompts that draw no panel simply ignore this.
    TogglePlan,
    // Ctrl+O: show what the transcript folded away, or fold it back. The state lives in the
    // writer rather than here, so unlike the two above this carries nothing — the caller
    // just tells `out` to flip it.
    ToggleExpand,
}

// How much of the plan the panel shows, cycled with Ctrl+T.
//
// The windowed default is a compromise, and both ends of it are sometimes what is wanted:
// the whole list while checking what the model set out to do, one line while reading the
// transcript underneath it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum PlanView {
    // One row: how far along the plan is, and what is being worked on.
    Collapsed,
    #[default]
    Windowed,
    // As much of it as the screen can spare, which is still not unbounded.
    Full,
}

impl PlanView {
    // Smallest, middle, largest, and round again. Starting from the middle, the first
    // press opens the panel up and the second folds it away.
    pub(crate) fn next(self) -> Self {
        match self {
            PlanView::Collapsed => PlanView::Windowed,
            PlanView::Windowed => PlanView::Full,
            PlanView::Full => PlanView::Collapsed,
        }
    }
}

pub(crate) struct Editor {
    // One entry per logical line: Shift+Enter adds one, Enter submits them all.
    lines: Vec<String>,
    row: usize,
    // In characters, not bytes.
    column: usize,
    history: Vec<String>,
    // Where in the history we are while browsing it, and what was being typed before
    // browsing started, so walking back down returns it rather than losing it.
    browsing: Option<usize>,
    stash: Vec<String>,
}

impl Editor {
    pub(crate) fn new(history: Vec<String>) -> Self {
        Editor {
            lines: vec![String::new()],
            row: 0,
            column: 0,
            history,
            browsing: None,
            stash: Vec::new(),
        }
    }

    pub(crate) fn history(&self) -> &[String] {
        &self.history
    }

    fn current(&self) -> &String {
        &self.lines[self.row]
    }

    fn line_length(&self) -> usize {
        self.current().chars().count()
    }

    fn is_empty(&self) -> bool {
        self.lines.iter().all(|line| line.is_empty())
    }

    fn reset(&mut self) {
        self.lines = vec![String::new()];
        self.row = 0;
        self.column = 0;
        self.browsing = None;
        self.stash.clear();
    }

    // Byte offset of the caret within the current line.
    fn offset(&self) -> usize {
        self.current()
            .char_indices()
            .nth(self.column)
            .map(|(index, _)| index)
            .unwrap_or_else(|| self.current().len())
    }

    fn insert(&mut self, ch: char) {
        let at = self.offset();
        self.lines[self.row].insert(at, ch);
        self.column += 1;
    }

    pub(crate) fn paste(&mut self, text: &str) {
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        for (index, piece) in normalized.split('\n').enumerate() {
            if index > 0 {
                self.split_line();
            }
            for ch in piece.chars() {
                self.insert(ch);
            }
        }
    }

    // What Ctrl+V and Shift+Insert both do. Goes through `paste` rather than inserting
    // the text directly, so a clipboard holding several lines becomes several lines here
    // instead of one long one with the breaks swallowed.
    //
    // Nothing happens where there is no clipboard to read — see the `clipboard` module for
    // why that is only Windows.
    fn paste_from_clipboard(&mut self) {
        if let Some(text) = clipboard::read() {
            self.paste(&text);
        }
    }

    // Put a message that was typed but never sent back where it came from.
    //
    // Interrupting a turn is the user taking it back, and what they had queued behind it
    // is part of what they are taking back: handing it to the model as the next prompt
    // instead would send the one thing they just cancelled. It goes in front of whatever
    // is half-typed rather than at the caret, which is wherever it happened to be left —
    // splicing an unsent message through the middle of a new one helps nobody. The caret
    // ends up between the two, so typing carries on where it stopped.
    pub(crate) fn restore(&mut self, text: &str) {
        let separated = if self.is_empty() {
            text.to_string()
        } else {
            format!("{}\n", text)
        };
        self.row = 0;
        self.column = 0;
        self.browsing = None;
        self.paste(&separated);
    }

    fn split_line(&mut self) {
        let at = self.offset();
        let tail = self.lines[self.row].split_off(at);
        self.lines.insert(self.row + 1, tail);
        self.row += 1;
        self.column = 0;
    }

    fn backspace(&mut self) {
        if self.column > 0 {
            let at = self.offset();
            let previous = self.current()[..at]
                .char_indices()
                .next_back()
                .map(|(index, _)| index)
                .unwrap_or(0);
            self.lines[self.row].replace_range(previous..at, "");
            self.column -= 1;
        } else if self.row > 0 {
            let tail = self.lines.remove(self.row);
            self.row -= 1;
            self.column = self.line_length();
            self.lines[self.row].push_str(&tail);
        }
    }

    fn delete_forward(&mut self) {
        if self.column < self.line_length() {
            let at = self.offset();
            let next = self.current()[at..]
                .char_indices()
                .nth(1)
                .map(|(index, _)| at + index)
                .unwrap_or_else(|| self.current().len());
            self.lines[self.row].replace_range(at..next, "");
        } else if self.row + 1 < self.lines.len() {
            let tail = self.lines.remove(self.row + 1);
            self.lines[self.row].push_str(&tail);
        }
    }

    fn kill_to_start(&mut self) {
        let at = self.offset();
        self.lines[self.row].replace_range(..at, "");
        self.column = 0;
    }

    fn kill_to_end(&mut self) {
        let at = self.offset();
        self.lines[self.row].truncate(at);
    }

    // Where the previous word starts, counted in characters from the line start.
    fn word_start(&self) -> usize {
        let chars: Vec<char> = self.current().chars().collect();
        let mut at = self.column;
        while at > 0 && chars[at - 1].is_whitespace() {
            at -= 1;
        }
        while at > 0 && !chars[at - 1].is_whitespace() {
            at -= 1;
        }
        at
    }

    fn word_end(&self) -> usize {
        let chars: Vec<char> = self.current().chars().collect();
        let mut at = self.column;
        while at < chars.len() && chars[at].is_whitespace() {
            at += 1;
        }
        while at < chars.len() && !chars[at].is_whitespace() {
            at += 1;
        }
        at
    }

    fn delete_word_back(&mut self) {
        let start = self.word_start();
        if start == self.column {
            return;
        }
        let from = self
            .current()
            .char_indices()
            .nth(start)
            .map(|(index, _)| index)
            .unwrap_or(0);
        let to = self.offset();
        self.lines[self.row].replace_range(from..to, "");
        self.column = start;
    }

    fn left(&mut self) {
        if self.column > 0 {
            self.column -= 1;
        } else if self.row > 0 {
            self.row -= 1;
            self.column = self.line_length();
        }
    }

    fn right(&mut self) {
        if self.column < self.line_length() {
            self.column += 1;
        } else if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.column = 0;
        }
    }

    // Up and down move within a multi-line entry first and only reach for the history
    // once there is nowhere left to go, so arrow keys never eat what is being typed.
    fn up(&mut self) {
        if self.row > 0 {
            self.row -= 1;
            self.column = self.column.min(self.line_length());
            return;
        }
        if self.history.is_empty() {
            return;
        }
        let next = match self.browsing {
            None => {
                self.stash = self.lines.clone();
                self.history.len() - 1
            }
            Some(0) => return,
            Some(index) => index - 1,
        };
        self.browsing = Some(next);
        self.load(self.history[next].clone());
    }

    fn down(&mut self) {
        if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.column = self.column.min(self.line_length());
            return;
        }
        let Some(index) = self.browsing else {
            return;
        };
        if index + 1 < self.history.len() {
            self.browsing = Some(index + 1);
            self.load(self.history[index + 1].clone());
        } else {
            self.browsing = None;
            let stashed = std::mem::take(&mut self.stash);
            self.lines = if stashed.is_empty() { vec![String::new()] } else { stashed };
            self.row = self.lines.len() - 1;
            self.column = self.line_length();
        }
    }

    fn load(&mut self, entry: String) {
        self.lines = entry.split('\n').map(str::to_string).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.row = self.lines.len() - 1;
        self.column = self.line_length();
    }

    fn remember(&mut self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        if self.history.last().map(String::as_str) == Some(text) {
            return;
        }
        self.history.push(text.to_string());
        const MAX_ENTRIES: usize = 1000;
        if self.history.len() > MAX_ENTRIES {
            let excess = self.history.len() - MAX_ENTRIES;
            self.history.drain(..excess);
        }
    }

    pub(crate) fn key(&mut self, key: KeyEvent) -> Option<Input> {
        // Windows reports releases as well as presses; acting on both types every
        // character twice.
        if key.kind != KeyEventKind::Press {
            return None;
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        match key.code {
            // Shift+Enter is what most people reach for; Alt+Enter is accepted too, for
            // terminals that cannot tell Shift+Enter from Enter — though not on Windows,
            // where the console keeps Alt+Enter for toggling its own full-screen mode and
            // it never reaches here.
            //
            // The Windows console in VT input mode is one of those terminals, and that is
            // the mode the full-screen renderer needs to get the wheel at all. `vt` puts
            // the modifier back by asking Windows whether Shift is down at the moment the
            // byte arrives; see `restore_shift`.
            KeyCode::Enter if shift || alt => self.split_line(),
            KeyCode::Enter => {
                let text = self.lines.join("\n");
                self.remember(&text);
                self.reset();
                return Some(Input::Text(text));
            }
            KeyCode::BackTab => return Some(Input::CycleMode),
            KeyCode::Char('t') if control => return Some(Input::TogglePlan),
            KeyCode::Char('c') if control => {
                self.reset();
                return Some(Input::Interrupted);
            }
            KeyCode::Char('d') if control => {
                if self.is_empty() {
                    return Some(Input::Eof);
                }
                self.delete_forward();
            }
            KeyCode::Char('a') if control => self.column = 0,
            KeyCode::Char('e') if control => self.column = self.line_length(),
            KeyCode::Char('u') if control => self.kill_to_start(),
            KeyCode::Char('k') if control => self.kill_to_end(),
            KeyCode::Char('w') if control => self.delete_word_back(),
            KeyCode::Char('v') if control => self.paste_from_clipboard(),
            KeyCode::Char('o') if control => return Some(Input::ToggleExpand),
            KeyCode::Char(ch) if !control => self.insert(ch),
            // Shift+Insert is how a terminal has always pasted, and in the console it
            // used to be the console's own doing. VT input mode ends that: the key is
            // forwarded here like any other, so pasting is now haico's job.
            //
            // The modifier is not required. Insert means nothing else here, and accepting
            // it either way means the paste still works whatever the terminal decides to
            // do with the Shift — which, on the one key where the encoding drops modifiers,
            // is not something to be relying on.
            KeyCode::Insert => self.paste_from_clipboard(),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete_forward(),
            KeyCode::Left if control || alt => self.column = self.word_start(),
            KeyCode::Right if control || alt => self.column = self.word_end(),
            KeyCode::Left => self.left(),
            KeyCode::Right => self.right(),
            KeyCode::Home => self.column = 0,
            KeyCode::End => self.column = self.line_length(),
            KeyCode::Up => self.up(),
            KeyCode::Down => self.down(),
            _ => {}
        }
        None
    }

    // The input box, wrapped to the terminal, and where the cursor belongs in it.
    //
    // Wrapping here rather than letting the terminal do it is not a preference: the
    // writer erases the region by counting rows back from the cursor, and a row the
    // terminal wrapped on its own is two rows it would not know about.
    pub(crate) fn rows(&self, prompt: &str, width: usize) -> (Vec<String>, usize, usize) {
        let indent = " ".repeat(prompt.chars().count());
        // A column short of the edge, for the reason `bar` gives: a row that fills the
        // last cell leaves the terminal with a wrap pending, and a wrap on the bottom row
        // scrolls the screen out from under a writer that is counting rows. Every other
        // row of the region already stopped here; this one did not.
        let width = width.saturating_sub(1).max(prompt.chars().count() + 8);

        let mut rows: Vec<String> = Vec::new();
        let mut caret = (0usize, 0usize);

        for (index, line) in self.lines.iter().enumerate() {
            let prefix = if index == 0 { prompt } else { indent.as_str() };
            let text: Vec<char> = prefix.chars().chain(line.chars()).collect();
            let first = rows.len();

            if text.is_empty() {
                rows.push(String::new());
            } else {
                let mut start = 0;
                while start < text.len() {
                    let end = (start + width).min(text.len());
                    rows.push(text[start..end].iter().collect());
                    start = end;
                }
            }

            if index == self.row {
                let at = prefix.chars().count() + self.column;
                caret = (first + at / width, at % width);
            }
        }

        // A caret exactly on a wrap boundary belongs at the start of the row after the
        // last one, which does not exist yet.
        while caret.0 >= rows.len() {
            rows.push(String::new());
        }

        (rows, caret.0, caret.1)
    }

}

fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(columns, _)| columns as usize)
        .unwrap_or(80)
        .max(20)
}

fn terminal_height() -> usize {
    crossterm::terminal::size()
        .map(|(_, rows)| rows as usize)
        .unwrap_or(24)
        .max(1)
}

// Everything the bottom of the screen shows besides what is being typed: what the
// model is busy with above the box, and what the session is set to below it.
pub(crate) struct Chrome<'a> {
    pub(crate) mode: Mode,
    pub(crate) busy: Option<&'a str>,
    // What the top bar says when nothing is running. Ignored while `busy` is set, since
    // what is happening now always beats what happened last.
    pub(crate) last: Option<&'a str>,
    // The right-hand end of the status bar: which model, and how full its context is.
    pub(crate) status: &'a str,
    // The plan as todo_write last left it, drawn above everything else. Empty until the
    // model writes one, and then the panel takes no rows at all.
    pub(crate) todos: &'a [TodoItem],
    // Messages typed while this was running that have not reached the model yet. Held
    // above the bar rather than written into the transcript, because the transcript is
    // what was said and these have not been said yet.
    pub(crate) queued: &'a [String],
    // How much of it to show. Owned by the caller, since it outlives any one draw.
    pub(crate) plan: PlanView,
}

// The chrome's colors. The bars are the only thing on screen that is painted rather than
// merely tinted, which is what makes them read as furniture instead of as output.
//
// A 256-color light blue rather than an RGB one: it lands the same on any terminal from
// the last twenty years, and the exact shade is not worth a truecolor requirement.
// Shared rather than private because the transcript borrows the shade for the line that
// reports a compressed chain of thought — that line is haico talking about the reply
// rather than part of it, and wearing the chrome's color is what says so. Kept as one
// constant so the two cannot drift apart into two nearly-identical blues.
pub(crate) const BAR_BACKGROUND: Color = Color::AnsiValue(153);
const BAR_TEXT: Color = Color::Black;

// A message waiting its turn. Painted rather than tinted, like the bars — but only behind
// the text itself, and in a gray that says furniture rather than message: it is on screen
// because it has *not* happened yet, and the moment it does it moves into the transcript
// as an ordinary line.
const QUEUED_BACKGROUND: Color = Color::AnsiValue(252);
const QUEUED_TEXT: Color = Color::Black;

// Paint one row of chrome the whole way across.
//
// Two things this owes the writer. The row stops a column short of the width, because a
// row the terminal wrapped on its own is a row `out` cannot count and the region's height
// is wrong from then on — with a background that last cell simply stays unpainted. And
// line breaks and tabs become spaces, because a summary of a multi-line shell command
// would otherwise put line feeds inside a row that is supposed to be exactly one.
//
// Runs of spaces are left alone, and that is not an oversight: the gap that right-aligns
// the status bar is a run of spaces, and collapsing it flattens the bar to the left.
fn bar(content: &str, width: usize) -> String {
    let room = width.saturating_sub(1);
    if room == 0 {
        return String::new();
    }

    let content: String = content
        .chars()
        .map(|ch| if ch.is_whitespace() { ' ' } else { ch })
        .collect();

    // Cut by characters: what reaches here is plain text, painted here rather than
    // carrying escapes of its own, so there is no sequence to cut in half.
    let content = fit(&content, room);

    let pad = room.saturating_sub(content.chars().count());
    format!(
        "{}{}{}{}{}",
        SetBackgroundColor(BAR_BACKGROUND),
        SetForegroundColor(BAR_TEXT),
        content,
        " ".repeat(pad),
        ResetColor
    )
}

// What the activity bar says while something is running: the label, plus how much reply
// has arrived so far once that is anything at all.
//
// Four characters to the token, the same rule of thumb `estimate_tokens` uses, so the two
// figures on screen cannot disagree about what a token is. Nothing is bracketed until the
// first token lands — through the silence before it, "(↓ 0)" says less than no bracket at
// all, and the silence before it is exactly what this exists for.
pub(crate) fn busy_label(busy: &str, streamed: Option<&AtomicUsize>) -> String {
    let characters = match streamed {
        Some(counter) => counter.load(Ordering::Relaxed),
        None => 0,
    };

    match characters / 4 {
        0 => busy.to_string(),
        tokens if tokens < 1000 => format!("{} (↓ {})", busy, tokens),
        tokens => format!("{} (↓ {:.1}k)", busy, tokens as f64 / 1000.0),
    }
}

// The bar above the box: what is running, or what ran last.
//
// It used to exist only while something was running, which meant the input box jumped a
// row every time work started or stopped. Always drawn, so it does not.
fn activity_bar(chrome: &Chrome, width: usize) -> String {
    let text = chrome.busy.or(chrome.last).unwrap_or("ready");
    bar(&format!("○ {}", text), width)
}

// One row: the mode on the left, the model and context on the right. Kept a column
// short of the terminal width so it can never be the thing that wraps — a row the
// terminal wrapped on its own is a row the writer cannot count.
fn status_bar(mode: Mode, status: &str, width: usize) -> String {
    let left = mode.label();
    let room = width.saturating_sub(1);

    // On a narrow terminal something has to give, and it is the front of the status:
    // a truncated model name still tells you which one it is, while a truncated
    // context figure is worse than none.
    let room_for_status = room.saturating_sub(left.chars().count() + 1);
    let length = status.chars().count();
    let status: String = if length > room_for_status {
        status.chars().skip(length - room_for_status).collect()
    } else {
        status.to_string()
    };

    let used = left.chars().count() + status.chars().count();
    let gap = room.saturating_sub(used).max(1);
    bar(&format!("{}{}{}", left, " ".repeat(gap), status), width)
}

// What was typed while something was running, one row each, above the bar.
//
// These used to go straight into the transcript, which read as though they had already
// been sent — a message queued behind a long tool call would sit in the scrollback for a
// minute or more looking exactly like one the model had already been given. Held here
// instead, in gray, until the round ends and they are actually pushed into the
// conversation.
//
// Flattened and cut to one row apiece for the same reason `bar` does it: a region row
// that turns out to be two is a row the writer cannot count.
fn queued_rows(queued: &[String], width: usize) -> Vec<String> {
    let room = width.saturating_sub(1);
    if room == 0 {
        return Vec::new();
    }

    queued
        .iter()
        .map(|text| {
            let flattened: String = text
                .chars()
                .map(|ch| if ch.is_whitespace() { ' ' } else { ch })
                .collect();
            let content = fit(&format!("{}{}", PROMPT, flattened.trim()), room);
            format!(
                "{}{}{}{}",
                SetBackgroundColor(QUEUED_BACKGROUND),
                SetForegroundColor(QUEUED_TEXT),
                content,
                ResetColor
            )
        })
        .collect()
}

// The plan, pinned above the activity bar.
//
// todo_write used to print the list as ordinary output, which meant it scrolled away
// behind the next few tool calls — exactly when it is most use. Pinned, it is on screen
// for as long as the plan exists, and that is what the windowing pays for: a long plan
// would otherwise take the transcript's rows away for the rest of the turn.
//
// The ceiling on the windowed view, whatever the height of the terminal, heading and "…"
// lines included.
const PLAN_MAX_ROWS: usize = 10;

// Cut a plain-text row down to the room it has, for the same reason `bar` does it: a row
// the terminal wraps on its own is a row the writer cannot count, and the region's height
// is wrong from then on.
pub(crate) fn fit(text: &str, columns: usize) -> String {
    let length = text.chars().count();
    if length <= columns || columns == 0 {
        return text.to_string();
    }
    text.chars().take(columns - 1).chain(['…']).collect()
}

fn dim(text: String) -> String {
    format!("{}{}{}", SetForegroundColor(Color::DarkGrey), text, ResetColor)
}

// `room` is how many rows the panel may take at most — what the box and the bars leave.
// Even the full view is bounded by it: the writer erases the region by counting rows back
// from the caret, so a region taller than the screen would have it walking back over rows
// that scrolled away.
pub(crate) fn plan_rows(
    todos: &[TodoItem],
    width: usize,
    room: usize,
    view: PlanView,
) -> Vec<String> {
    if todos.is_empty() || room == 0 {
        return Vec::new();
    }

    let budget = match view {
        PlanView::Collapsed => 1,
        // A third of what is free, because the transcript above is what the panel borrows
        // from, and never more than the ceiling however tall the terminal is.
        PlanView::Windowed => (room / 3).clamp(4, PLAN_MAX_ROWS),
        PlanView::Full => room,
    }
    .min(room);

    let columns = width.saturating_sub(1);
    let done = todos.iter().filter(|item| item.status == "done").count();
    let mark = |item: &TodoItem| match item.status.as_str() {
        "done" => "x",
        "in_progress" => "»",
        _ => " ",
    };

    // What the window is kept around: whatever is being worked on, or failing that the
    // first thing still to do. A finished list shows its end, which is the part that says
    // it is finished.
    let focus = todos
        .iter()
        .position(|item| item.status == "in_progress")
        .or_else(|| todos.iter().position(|item| item.status != "done"))
        .unwrap_or(todos.len() - 1);

    let heading = format!("  Plan  {}/{}", done, todos.len());

    // The one-row form: asked for, or all there is room for. Below four rows a windowed
    // list is mostly ellipses, and this says the same thing in less space.
    if budget < 4 && todos.len() + 1 > budget {
        let item = &todos[focus];
        let line = format!("{}  [{}] {}", heading, mark(item), item.task);
        return vec![dim(fit(&line, columns))];
    }

    // One row of the budget goes to the heading, and one to each "…" that turns out to be
    // needed. Both are assumed and a row handed back when the window sits against an end,
    // which is shorter than solving it exactly and lands on the same answer.
    let body = budget - 1;
    let (mut start, mut end) = (0, todos.len());
    if todos.len() > body {
        let shown = body.saturating_sub(2).max(1);
        start = focus.saturating_sub(shown / 2).min(todos.len() - shown);
        end = start + shown;
        if start == 0 {
            end = (end + 1).min(todos.len());
        }
        if end == todos.len() {
            start = start.saturating_sub(1);
        }
    }

    let mut rows = vec![dim(fit(&heading, columns))];
    if start > 0 {
        rows.push(dim(fit(&format!("   … {} above", start), columns)));
    }
    for item in &todos[start..end] {
        let line = fit(&format!("   [{}] {}", mark(item), item.task), columns);
        // Only what is left to do is worth full brightness: the finished items are there
        // to say the plan is being followed, not to be read again.
        rows.push(if item.status == "done" { dim(line) } else { line });
    }
    if end < todos.len() {
        rows.push(dim(fit(
            &format!("   … +{} more", todos.len() - end),
            columns,
        )));
    }
    rows
}

// How far an event asks the transcript to move, if it asks at all. Negative is backward,
// into what has already gone past.
//
// The wheel only ever arrives because the full-screen renderer asked the terminal for
// button reporting; under the inline renderer the terminal keeps the wheel to itself and
// `out::scroll` does nothing anyway. Home and End are already the editor's, so paging is
// on PageUp and PageDown, which nothing else wanted.
fn scroll_rows(event: &Event) -> Option<isize> {
    // A page is a screenful less two rows, so a couple of lines of context carry over and
    // the reader can see where they were.
    let page = || out::viewport_rows().saturating_sub(2).max(1) as isize;
    match event {
        Event::Mouse(mouse) => match mouse.kind {
            MouseEventKind::ScrollUp => Some(-3),
            MouseEventKind::ScrollDown => Some(3),
            _ => None,
        },
        Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
            KeyCode::PageUp => Some(-page()),
            KeyCode::PageDown => Some(page()),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn draw(editor: &Editor, chrome: &Chrome) {
    let width = terminal_width();
    let (mut rows, caret_row, caret_column) = editor.rows(PROMPT, width);

    rows.insert(0, activity_bar(chrome, width));
    rows.push(status_bar(chrome.mode, chrome.status, width));

    // Between the plan and the bar: near enough to the box to read as something the user
    // typed, above the bar because what is waiting to be sent is not what is running.
    let waiting = queued_rows(chrome.queued, width);
    let waiting_rows = waiting.len();
    rows.splice(0..0, waiting);

    // What is left for the plan once the box, the bars and the blank row above them have
    // had their rows, less two: one so the transcript never disappears entirely, and one
    // for a streamed line that has not ended yet, which is drawn above the region and
    // counts against the screen just the same.
    let room = terminal_height().saturating_sub(rows.len() + 3);
    let plan = plan_rows(chrome.todos, width, room, chrome.plan);

    // The plan sits above the activity bar, and one blank row sits above the lot: without
    // it the pinned block butts straight up against the transcript and the last line of
    // output reads as another row of chrome. Everything below them — the box included —
    // is that many rows further down than it used to be.
    let above = plan.len() + waiting_rows + 2;
    rows.splice(0..0, plan);
    rows.insert(0, String::new());

    out::set_region(rows, caret_row + above, caret_column);
}

// What a submitted entry looks like once it is part of the scrollback rather than
// something being edited.
pub(crate) fn echo(prompt: &str, text: &str) -> String {
    let indent = " ".repeat(prompt.chars().count());
    text.split('\n')
        .enumerate()
        .map(|(index, line)| {
            if index == 0 {
                format!("{}{}", prompt, line)
            } else {
                format!("{}{}", indent, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// Read one entry with nothing else going on. The editor is redrawn after every
// keystroke, which is cheap: it is a handful of rows, and the writer only touches the
// ones below the transcript.
pub(crate) async fn read_text(keys: &mut Keys, editor: &mut Editor, prompt: &str) -> Input {
    loop {
        let (rows, caret_row, caret_column) = editor.rows(prompt, terminal_width());
        out::set_region(rows, caret_row, caret_column);
        let Some(event) = keys.next().await else {
            return Input::Eof;
        };
        if let Some(delta) = scroll_rows(&event) {
            out::scroll(delta);
            continue;
        }
        match event {
            Event::Key(key) => match editor.key(key) {
                // This prompt draws no plan panel, so there is nothing to fold.
                Some(Input::TogglePlan) => {}
                Some(Input::ToggleExpand) => out::toggle_expand(),
                Some(input) => {
                    out::follow();
                    out::clear_region();
                    return input;
                }
                None => {}
            },
            Event::Paste(text) => editor.paste(&text),
            // Redrawn at the top of the loop with the new width.
            Event::Resize(..) => {}
            _ => {}
        }
    }
}

// The main prompt: the full bottom of the screen, and the one key that means something
// only here.
pub(crate) async fn read_command(
    keys: &mut Keys,
    editor: &mut Editor,
    mode: &mut Mode,
    status: &str,
    last: Option<&str>,
    todos: &[TodoItem],
    plan: &mut PlanView,
) -> Input {
    loop {
        draw(
            editor,
            &Chrome {
                mode: *mode,
                busy: None,
                last,
                status,
                todos,
                // Nothing can be waiting here: a queued message becomes the next prompt
                // rather than reaching this one.
                queued: &[],
                plan: *plan,
            },
        );
        let Some(event) = keys.next().await else {
            return Input::Eof;
        };
        if let Some(delta) = scroll_rows(&event) {
            out::scroll(delta);
            continue;
        }
        match event {
            Event::Key(key) => match editor.key(key) {
                Some(Input::CycleMode) => *mode = mode.next(),
                Some(Input::TogglePlan) => *plan = plan.next(),
                Some(Input::ToggleExpand) => out::toggle_expand(),
                Some(input) => {
                    // Whatever was just typed is about to be answered, and the answer goes
                    // on the end: a view still held back would miss its own reply.
                    out::follow();
                    out::clear_region();
                    return input;
                }
                None => {}
            },
            Event::Paste(text) => editor.paste(&text),
            Event::Resize(..) => {}
            _ => {}
        }
    }
}

// What to do with a key that arrives while the model is working.
pub(crate) enum WhileWorking {
    // Give up on this turn.
    Interrupt,
    // Typed and submitted: it goes in the queue for when the turn is over.
    Queued(String),
    // Anything else — still editing, or nothing that means something here.
    Nothing,
}

// One keystroke's worth of the live input box, for use in a select! arm alongside
// whatever the model is doing.
pub(crate) fn while_working(
    editor: &mut Editor,
    mode: &mut Mode,
    plan: &mut PlanView,
    event: Event,
) -> WhileWorking {
    // Reading back through the transcript while the model works is the whole point of
    // being able to scroll at all, so this comes before anything that might consume the
    // key. What arrives while the view is held back leaves it where it is.
    if let Some(delta) = scroll_rows(&event) {
        out::scroll(delta);
        return WhileWorking::Nothing;
    }

    match event {
        Event::Key(key) => {
            if key.kind == KeyEventKind::Press && key.code == KeyCode::Esc {
                return WhileWorking::Interrupt;
            }
            match editor.key(key) {
                Some(Input::Text(text)) => {
                    out::follow();
                    WhileWorking::Queued(text)
                }
                // Ctrl+C means the same thing here as Esc does.
                Some(Input::Interrupted) => WhileWorking::Interrupt,
                Some(Input::CycleMode) => {
                    *mode = mode.next();
                    WhileWorking::Nothing
                }
                // Nothing to report: the caller redraws after every key it is handed.
                Some(Input::TogglePlan) => {
                    *plan = plan.next();
                    WhileWorking::Nothing
                }
                Some(Input::ToggleExpand) => {
                    out::toggle_expand();
                    WhileWorking::Nothing
                }
                // Ctrl+D is for leaving, and leaving mid-turn is what Esc is for.
                Some(Input::Eof) | None => WhileWorking::Nothing,
            }
        }
        Event::Paste(text) => {
            editor.paste(&text);
            WhileWorking::Nothing
        }
        _ => WhileWorking::Nothing,
    }
}

// Reading the Windows console as a byte stream.
//
// Only reached when the full-screen renderer has put the console into VT input mode, which
// it does because that is the only configuration where the wheel reports at all *and*
// Shift+drag still selects text. The cost is that every key now arrives as an escape
// sequence, and crossterm reports those a character at a time: an arrow key comes through
// as "[", "A", then Up, and a mouse report comes through as its characters and nothing
// else. Filtering that after the fact would mean guessing whether a "[" starts a sequence
// or is someone typing `arr[i]`, and guessing wrong eats pasted code. Parsing the bytes
// removes the guess.
//
// `terminput` is crossterm's own escape parser published separately, so this is the same
// parsing that happens on Unix, just applied where crossterm declines to.
#[cfg(windows)]
mod vt {
    use super::{Event, UnboundedSender};
    use std::io::Read;

    pub(super) fn start(sender: UnboundedSender<Event>) {
        let sizes = sender.clone();
        std::thread::spawn(move || read_stream(sender));
        std::thread::spawn(move || watch_size(sizes));
    }

    fn read_stream(sender: UnboundedSender<Event>) {
        let mut stdin = std::io::stdin();
        let mut chunk = [0u8; 1024];
        let mut buffer: Vec<u8> = Vec::new();

        loop {
            let read = match stdin.read(&mut chunk) {
                Ok(0) | Err(_) => return,
                Ok(read) => read,
            };
            buffer.extend_from_slice(&chunk[..read]);

            loop {
                if buffer.is_empty() {
                    break;
                }
                // An Escape all on its own is an Escape; an Escape with anything behind it
                // is the start of a sequence. The console writes a sequence in one piece,
                // so by the time the buffer holds nothing but the Escape, that is what it
                // was — which is what lets Esc interrupt a turn without waiting on a timer.
                if buffer.len() == 1 && buffer[0] == 0x1B {
                    buffer.clear();
                    if sender.send(Event::Key(super::KeyCode::Esc.into())).is_err() {
                        return;
                    }
                    break;
                }
                match take_event(&mut buffer) {
                    Some(Some(event)) => {
                        if sender.send(event).is_err() {
                            return;
                        }
                    }
                    // Consumed, but nothing haico has a use for.
                    Some(None) => {}
                    // Incomplete: the rest of it is still on its way.
                    None => break,
                }
            }
        }
    }

    // One event off the front of the buffer, or `None` while it is still incomplete.
    //
    // The shortest prefix that parses is the event, and is also how many bytes it took.
    // Asking about the whole buffer instead would be wrong in both directions: `parse_from`
    // ignores trailing bytes, so it would swallow the keystroke behind this one.
    fn take_event(buffer: &mut Vec<u8>) -> Option<Option<Event>> {
        // A leading Escape is never the whole event here — the caller has already dealt
        // with the case where it is — so the search starts past it.
        let first = if buffer[0] == 0x1B { 2 } else { 1 };
        for end in first..=buffer.len() {
            match terminput::Event::parse_from(&buffer[..end]) {
                Ok(Some(parsed)) => {
                    buffer.drain(..end);
                    return Some(
                        terminput_crossterm::to_crossterm(parsed)
                            .ok()
                            .map(restore_shift),
                    );
                }
                Ok(None) => continue,
                Err(_) => {
                    // Nothing beginning here can ever parse, so it goes rather than
                    // blocking every keystroke that follows it.
                    buffer.drain(..1);
                    return Some(None);
                }
            }
        }
        None
    }

    // The one thing the byte stream cannot say, asked of Windows directly.
    //
    // Enter is the single byte 0x0D whether or not Shift was held. The legacy encoding has
    // a modifier field only for keys sent as CSI sequences — that is how `ESC[1;5D` manages
    // to mean Ctrl+Left — and a key sent as a bare control character has nowhere to put
    // one. So the state is gone before any parser could see it, and no amount of parsing
    // brings it back.
    //
    // But it is only gone from the *encoding*. The key is still physically down: this runs
    // within microseconds of the console writing the byte, and nobody releases Shift that
    // fast. `GetAsyncKeyState` reports the live physical state and, unlike `GetKeyState`,
    // needs no message queue — which a console program does not have.
    //
    // Deliberately only Enter. Every other key carries its own modifiers correctly, and
    // second-guessing state we already have would be strictly worse than trusting it.
    fn restore_shift(event: Event) -> Event {
        let Event::Key(mut key) = event else {
            return event;
        };
        if key.code == super::KeyCode::Enter
            && key.modifiers.is_empty()
            && shift_is_down()
        {
            key.modifiers |= super::KeyModifiers::SHIFT;
        }
        Event::Key(key)
    }

    fn shift_is_down() -> bool {
        extern "system" {
            fn GetAsyncKeyState(key: i32) -> i16;
        }
        const VK_SHIFT: i32 = 0x10;
        // The high bit is "down right now"; the low bit is "was pressed since last asked"
        // and is not what this wants.
        unsafe { GetAsyncKeyState(VK_SHIFT) as u16 & 0x8000 != 0 }
    }

    // VT input reports keys and the mouse and says nothing about the window changing size,
    // and the console records that used to carry that are no longer being read. Polling is
    // the whole of the fix: dragging a window edge is a human-speed thing, and four times a
    // second is faster than anyone can do it.
    fn watch_size(sender: UnboundedSender<Event>) {
        let mut last = crossterm::terminal::size().unwrap_or((80, 24));
        loop {
            std::thread::sleep(std::time::Duration::from_millis(250));
            let Ok(size) = crossterm::terminal::size() else {
                continue;
            };
            if size != last {
                last = size;
                if sender.send(Event::Resize(size.0, size.1)).is_err() {
                    return;
                }
            }
        }
    }
    #[cfg(test)]
    mod tests {
        use super::take_event;
        use crossterm::event::{Event, KeyCode, MouseEventKind};

        fn drain(bytes: &[u8]) -> (Vec<Event>, Vec<u8>) {
            let mut buffer = bytes.to_vec();
            let mut events = Vec::new();
            while !buffer.is_empty() {
                match take_event(&mut buffer) {
                    Some(Some(event)) => events.push(event),
                    Some(None) => {}
                    None => break,
                }
            }
            (events, buffer)
        }

        // The whole reason the bytes are parsed rather than crossterm's per-record events
        // being filtered: an arrow key is one event, not "[" then "A" then Up.
        #[test]
        fn an_arrow_key_is_one_event_and_not_its_characters() {
            let (events, rest) = drain(b"\x1b[A");

            assert_eq!(events.len(), 1, "{:?}", events);
            assert!(matches!(events[0], Event::Key(key) if key.code == KeyCode::Up));
            assert!(rest.is_empty());
        }

        // `parse_from` is happy to ignore trailing bytes, so asking it about the whole
        // buffer would take the next keystroke along with this one. Only the shortest
        // prefix that parses may be consumed.
        #[test]
        fn two_sequences_running_together_stay_two_events() {
            let (events, rest) = drain(b"\x1b[A\x1b[B");

            assert_eq!(events.len(), 2, "{:?}", events);
            assert!(matches!(events[0], Event::Key(key) if key.code == KeyCode::Up));
            assert!(matches!(events[1], Event::Key(key) if key.code == KeyCode::Down));
            assert!(rest.is_empty());
        }

        // The point of the exercise.
        #[test]
        fn a_wheel_report_becomes_a_scroll_event() {
            let (events, _) = drain(b"\x1b[<64;10;5M");

            assert_eq!(events.len(), 1, "{:?}", events);
            let Event::Mouse(mouse) = events[0] else {
                panic!("not a mouse event: {:?}", events[0]);
            };
            assert!(
                matches!(mouse.kind, MouseEventKind::ScrollUp | MouseEventKind::ScrollDown),
                "{:?}",
                mouse.kind
            );
        }

        // A filter working on crossterm's per-character events could not tell this from the
        // start of a sequence, and eating it would corrupt pasted code.
        #[test]
        fn ordinary_text_with_a_bracket_in_it_survives() {
            let (events, rest) = drain(b"arr[i]");

            let typed: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::Key(key) => match key.code {
                        KeyCode::Char(ch) => Some(ch),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            assert_eq!(typed, "arr[i]");
            assert!(rest.is_empty());
        }

        // Half a sequence is not an event yet, and must not be mistaken for one.
        #[test]
        fn an_unfinished_sequence_waits_for_the_rest_of_itself() {
            let mut buffer = b"\x1b[".to_vec();

            assert!(take_event(&mut buffer).is_none());
            assert_eq!(buffer, b"\x1b[", "an incomplete sequence was consumed");

            buffer.push(b'C');
            let event = take_event(&mut buffer).expect("now complete").expect("an event");
            assert!(matches!(event, Event::Key(key) if key.code == KeyCode::Right));
        }
    }
}

// Reading the clipboard, for the paste the terminal is no longer doing for us.
//
// Windows only, and deliberately so: this exists because VT input mode stops the console
// acting on Shift+Insert itself. Every other terminal still handles that key on its own
// and never forwards it, so there is nothing here to do and no dependency worth taking on
// to do it.
mod clipboard {
    #[cfg(windows)]
    pub(super) fn read() -> Option<String> {
        // Empty is not worth a paste, and a clipboard holding an image rather than text
        // reports an error rather than something to insert.
        clipboard_win::get_clipboard_string()
            .ok()
            .filter(|text| !text.is_empty())
    }

    #[cfg(not(windows))]
    pub(super) fn read() -> Option<String> {
        None
    }
}

// History lives in one file, one entry per line, with the newlines of a multi-line
// entry escaped so an entry stays an entry.

pub(crate) fn load_history(path: &Path) -> Vec<String> {
    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.replace("\\n", "\n").replace("\\\\", "\\"))
        .collect()
}

pub(crate) fn save_history(path: &Path, history: &[String]) {
    let body: String = history
        .iter()
        .map(|entry| format!("{}\n", entry.replace('\\', "\\\\").replace('\n', "\\n")))
        .collect();
    let _ = fs::write(path, body);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyEventKind, KeyModifiers};

    fn press(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    fn type_text(editor: &mut Editor, text: &str) {
        for ch in text.chars() {
            editor.key(press(KeyCode::Char(ch), KeyModifiers::NONE));
        }
    }

    #[test]
    fn shift_enter_grows_the_entry_and_enter_submits_all_of_it() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "first");
        editor.key(press(KeyCode::Enter, KeyModifiers::SHIFT));
        type_text(&mut editor, "second");

        let (rows, caret_row, caret_column) = editor.rows("> ", 80);
        assert_eq!(rows, vec!["> first".to_string(), "  second".to_string()]);
        assert_eq!((caret_row, caret_column), (1, 8));

        match editor.key(press(KeyCode::Enter, KeyModifiers::NONE)) {
            Some(Input::Text(text)) => assert_eq!(text, "first\nsecond"),
            _ => panic!("Enter should submit"),
        }
        // And the editor is empty again afterwards.
        assert_eq!(editor.rows("> ", 80).0, vec!["> ".to_string()]);
    }

    #[test]
    fn a_long_line_wraps_and_the_caret_follows_it() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, &"x".repeat(12));

        // Width 10, prompt 2 wide: "> " plus 8 x's, then the remaining 4.
        let (rows, caret_row, caret_column) = editor.rows("> ", 10);
        assert_eq!(rows, vec!["> xxxxxxxx".to_string(), "xxxx".to_string()]);
        assert_eq!((caret_row, caret_column), (1, 4));
    }

    #[test]
    fn editing_keys_work_on_characters_not_bytes() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "привет мир");
        editor.key(press(KeyCode::Char('w'), KeyModifiers::CONTROL));
        assert_eq!(editor.rows("> ", 80).0, vec!["> привет ".to_string()]);

        editor.key(press(KeyCode::Backspace, KeyModifiers::NONE));
        editor.key(press(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0, vec!["> приве".to_string()]);

        editor.key(press(KeyCode::Home, KeyModifiers::NONE));
        editor.key(press(KeyCode::Delete, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0, vec!["> риве".to_string()]);
    }

    #[test]
    fn the_arrows_reach_the_history_only_past_the_edge_of_the_entry() {
        let mut editor = Editor::new(vec!["older".to_string(), "newer".to_string()]);
        type_text(&mut editor, "draft");
        editor.key(press(KeyCode::Enter, KeyModifiers::SHIFT));
        type_text(&mut editor, "second");

        // From the second row, Up is a cursor move.
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0.len(), 2);

        // From the first, it is the history — and what was being typed comes back.
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0, vec!["> newer".to_string()]);
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0, vec!["> older".to_string()]);
        editor.key(press(KeyCode::Down, KeyModifiers::NONE));
        editor.key(press(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(
            editor.rows("> ", 80).0,
            vec!["> draft".to_string(), "  second".to_string()]
        );
    }

    #[test]
    fn a_pasted_block_becomes_one_entry_rather_than_several() {
        let mut editor = Editor::new(Vec::new());
        editor.paste("one\ntwo\r\nthree");
        match editor.key(press(KeyCode::Enter, KeyModifiers::NONE)) {
            Some(Input::Text(text)) => assert_eq!(text, "one\ntwo\nthree"),
            _ => panic!("Enter should submit"),
        }
    }

    #[test]
    fn the_status_bar_never_reaches_the_edge_of_the_screen() {
        // A row exactly as wide as the terminal wraps, and a wrapped row is one the
        // writer cannot count — so the bar has to stop a column short whatever is in it.
        for width in [20usize, 40, 80] {
            for status in ["", "m", "a-model-with-a-long-name  ctx 97%"] {
                let bar = status_bar(Mode::Auto, status, width);
                assert!(
                    crate::out::visible_width(&bar) <= width - 1,
                    "width {} overflowed with {:?}: {:?}",
                    width,
                    status,
                    bar
                );
            }
        }
    }

    #[test]
    fn the_activity_bar_counts_up_only_once_there_is_something_to_count() {
        let streamed = AtomicUsize::new(0);

        // Work that produces no stream of its own says only what it is doing.
        assert_eq!(busy_label("Thinking…", None), "Thinking…");

        // And so does a stream that has not delivered a token yet: the whole point is the
        // silence before the first one, and "(↓ 0)" would fill it with nothing.
        assert_eq!(busy_label("Thinking…", Some(&streamed)), "Thinking…");
        streamed.store(3, Ordering::Relaxed);
        assert_eq!(busy_label("Thinking…", Some(&streamed)), "Thinking…");

        // Four characters to the token, the same rule `estimate_tokens` uses.
        streamed.store(4, Ordering::Relaxed);
        assert_eq!(busy_label("Thinking…", Some(&streamed)), "Thinking… (↓ 1)");
        streamed.store(2596, Ordering::Relaxed);
        assert_eq!(busy_label("Thinking…", Some(&streamed)), "Thinking… (↓ 649)");

        // Thousands become the short form rather than a five-digit number nobody reads.
        streamed.store(4000, Ordering::Relaxed);
        assert_eq!(busy_label("Thinking…", Some(&streamed)), "Thinking… (↓ 1.0k)");
        streamed.store(94_000, Ordering::Relaxed);
        assert_eq!(busy_label("Thinking…", Some(&streamed)), "Thinking… (↓ 23.5k)");
    }

    #[test]
    fn the_box_sits_between_the_line_above_it_and_the_bar_below() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "hello");

        let drawn = |chrome: &Chrome| {
            let _guard = crate::out::capture_guard();
            crate::out::start_capture();
            draw(&editor, chrome);
            crate::out::take_capture()
        };

        let written = drawn(&Chrome {
            mode: Mode::Plan,
            busy: Some("thinking"),
            last: Some("ran cargo build"),
            status: "a-model  ctx 3%",
            todos: &[],
            queued: &[],
            plan: PlanView::Windowed,
        });

        let busy = written.find("○ thinking").expect("the busy line");
        let box_row = written.find("> hello").expect("the input box");
        // The two spaces before "ctx" are the separator status_line puts there. Only line
        // breaks are flattened, so a run of spaces has to survive the bar intact — the gap
        // that right-aligns this one is made of them.
        let bar = written.rfind("a-model  ctx 3%").expect("the status bar");
        assert!(busy < box_row && box_row < bar, "wrong order in {:?}", written);

        // Both bars are painted, and black on light blue is what says "chrome" rather
        // than "something the model said".
        let painted = format!("{}{}", SetBackgroundColor(BAR_BACKGROUND), SetForegroundColor(BAR_TEXT));
        assert_eq!(written.matches(&painted).count(), 2, "both bars paint: {:?}", written);

        // Nothing running: the top bar stays on screen showing what last happened, so the
        // box never jumps a row between turns.
        let idle = drawn(&Chrome {
            mode: Mode::Plan,
            busy: None,
            last: Some("ran cargo build"),
            status: "a-model  ctx 3%",
            todos: &[],
            queued: &[],
            plan: PlanView::Windowed,
        });
        assert!(idle.contains("○ ran cargo build"), "got {:?}", idle);

        // ...and before anything has happened at all, it still says something.
        let fresh = drawn(&Chrome {
            mode: Mode::Plan,
            busy: None,
            last: None,
            status: "a-model  ctx 3%",
            todos: &[],
            queued: &[],
            plan: PlanView::Windowed,
        });
        assert!(fresh.contains("○ ready"), "got {:?}", fresh);
    }


    #[test]
    fn an_unsent_message_comes_back_to_the_box_ahead_of_what_is_half_typed() {
        // Nothing being typed: it simply lands, ready to send or edit.
        let mut editor = Editor::new(Vec::new());
        editor.restore("what about the tests");
        let (rows, caret_row, caret_column) = editor.rows(PROMPT, 80);
        assert_eq!(rows, vec!["> what about the tests".to_string()]);
        assert_eq!((caret_row, caret_column), (0, 22));

        // Something being typed: the unsent message goes in front of it rather than
        // through the middle of it, and the caret stays with the new text.
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "half typed");
        editor.restore("first message");
        let (rows, caret_row, caret_column) = editor.rows(PROMPT, 80);
        assert_eq!(
            rows,
            vec!["> first message".to_string(), "  half typed".to_string()]
        );
        assert_eq!((caret_row, caret_column), (1, 2));

        // Two of them keep their own lines, so Enter still sends what was typed.
        let mut editor = Editor::new(Vec::new());
        editor.restore("one\ntwo");
        let (rows, ..) = editor.rows(PROMPT, 80);
        assert_eq!(rows, vec!["> one".to_string(), "  two".to_string()]);
    }
    #[test]
    fn a_queued_message_waits_above_the_bar_rather_than_in_the_transcript() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "hello");

        let _guard = crate::out::capture_guard();
        crate::out::start_capture();
        draw(
            &editor,
            &Chrome {
                mode: Mode::Normal,
                busy: Some("thinking"),
                last: None,
                status: "a-model  ctx 3%",
                todos: &[],
                queued: &["ask about the tests".to_string()],
                plan: PlanView::Windowed,
            },
        );
        let written = crate::out::take_capture();

        let waiting = written
            .find("> ask about the tests")
            .expect("the queued message");
        let bar = written.find("○ thinking").expect("the activity bar");
        let box_row = written.rfind("> hello").expect("the input box");
        assert!(
            waiting < bar && bar < box_row,
            "wrong order in {:?}",
            written
        );
    }

    #[test]
    fn a_queued_message_takes_exactly_one_row_however_it_was_typed() {
        let rows = queued_rows(&["one\ntwo\nthree".to_string(), "x".repeat(200)], 40);
        assert_eq!(rows.len(), 2, "one row each: {:?}", rows);
        for row in &rows {
            // A region row that turns out to be two is a row the writer cannot count.
            assert!(!row.contains('\n'), "more than one row: {:?}", row);
            assert!(crate::out::visible_width(row) < 40, "too wide: {:?}", row);
        }
        assert!(rows[0].contains("one two three"), "got {:?}", rows[0]);
    }

    fn todo(task: &str, status: &str) -> TodoItem {
        TodoItem {
            task: task.to_string(),
            status: status.to_string(),
        }
    }

    // A plan long enough that no view can show all of it, with the work part way down.
    fn long_plan() -> Vec<TodoItem> {
        let mut plan: Vec<TodoItem> = (0..20)
            .map(|index| todo(&format!("step {}", index), "done"))
            .collect();
        plan[12].status = "in_progress".to_string();
        for item in plan.iter_mut().skip(13) {
            item.status = "pending".to_string();
        }
        plan
    }

    #[test]
    fn the_plan_panel_stays_short_and_keeps_the_current_item_in_view() {
        // Nothing pinned until there is a plan: the panel costs no rows at all.
        assert!(plan_rows(&[], 80, 20, PlanView::Windowed).is_empty());

        let short = vec![
            todo("write the config", "done"),
            todo("write the server", "in_progress"),
            todo("write main", "pending"),
        ];
        let rows = plan_rows(&short, 80, 20, PlanView::Windowed);
        assert_eq!(rows.len(), 4, "a heading and one row each: {:?}", rows);
        assert!(rows[0].contains("Plan  1/3"), "got {:?}", rows[0]);
        assert!(rows[2].contains("[»] write the server"), "got {:?}", rows[2]);

        // Longer than the panel: windowed around whatever is in progress, with both ends
        // saying how much is out of sight rather than silently dropping it.
        let long = long_plan();
        for room in [6usize, 20, 60] {
            let rows = plan_rows(&long, 80, room, PlanView::Windowed);
            let budget = (room / 3).clamp(4, PLAN_MAX_ROWS).min(room);
            assert!(rows.len() <= budget, "{} rows for room {}: {:?}", rows.len(), room, rows);
            assert!(
                rows.iter().any(|row| row.contains("step 12")),
                "the item being worked on has to be in the window: {:?}",
                rows
            );
            assert!(rows.iter().any(|row| row.contains("above")), "{:?}", rows);
            assert!(rows.iter().any(|row| row.contains("more")), "{:?}", rows);
        }

        // And a row that wrapped would be a row the writer cannot count.
        let wide = vec![todo(&"a very long task ".repeat(20), "pending")];
        for row in plan_rows(&wide, 40, 20, PlanView::Windowed) {
            assert!(crate::out::visible_width(&row) < 40, "too wide: {:?}", row);
        }
    }

    #[test]
    fn ctrl_t_cycles_the_panel_between_its_three_sizes() {
        let mut editor = Editor::new(Vec::new());
        let mut view = PlanView::default();
        let mut cycle = || {
            match editor.key(press(KeyCode::Char('t'), KeyModifiers::CONTROL)) {
                Some(Input::TogglePlan) => view = view.next(),
                other => panic!("Ctrl+T should ask for the panel, got {:?}", other.is_some()),
            }
            view
        };
        // The default is the middle size; the first press opens it up, the second folds
        // it away, and the third is back where it started.
        assert_eq!(PlanView::default(), PlanView::Windowed);
        assert_eq!(cycle(), PlanView::Full);
        assert_eq!(cycle(), PlanView::Collapsed);
        assert_eq!(cycle(), PlanView::Windowed);

        let long = long_plan();

        // Folded away: one row, whatever the plan and whatever the room, and it still
        // names what is being worked on.
        let rows = plan_rows(&long, 80, 40, PlanView::Collapsed);
        assert_eq!(rows.len(), 1, "got {:?}", rows);
        assert!(rows[0].contains("Plan  12/20"), "got {:?}", rows[0]);
        assert!(rows[0].contains("[»] step 12"), "got {:?}", rows[0]);

        // Opened up: all twenty and the heading, since there is room for them.
        let rows = plan_rows(&long, 80, 40, PlanView::Full);
        assert_eq!(rows.len(), 21, "got {:?}", rows);
        assert!(rows[1].contains("[x] step 0"), "got {:?}", rows[1]);
        assert!(rows[20].contains("[ ] step 19"), "got {:?}", rows[20]);

        // ...but never past what the screen can spare. The full view is bounded by the
        // room it is given, because a region taller than the terminal scrolls its own top
        // away and the writer then erases rows that are no longer there.
        for room in [1usize, 2, 3, 7, 12] {
            for view in [PlanView::Collapsed, PlanView::Windowed, PlanView::Full] {
                let rows = plan_rows(&long, 80, room, view);
                assert!(
                    rows.len() <= room,
                    "{:?} took {} rows out of {}: {:?}",
                    view,
                    rows.len(),
                    room,
                    rows
                );
            }
        }
    }

    #[test]
    fn the_plan_sits_above_the_activity_bar_and_the_caret_moves_down_with_it() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "hello");
        let todos = vec![todo("write the server", "in_progress")];

        let _guard = crate::out::capture_guard();
        crate::out::start_capture();
        draw(
            &editor,
            &Chrome {
                mode: Mode::Normal,
                busy: Some("thinking"),
                last: None,
                status: "a-model  ctx 3%",
                todos: &todos,
                queued: &[],
                plan: PlanView::Windowed,
            },
        );
        let written = crate::out::take_capture();

        let plan = written.find("Plan  0/1").expect("the plan");
        let busy = written.find("○ thinking").expect("the busy line");
        let box_row = written.find("> hello").expect("the input box");
        assert!(plan < busy && busy < box_row, "wrong order in {:?}", written);

        // Two plan rows, the bar, the box, the status bar: the caret belongs on the fourth
        // row of five, so the writer walks back exactly one from the last and along to
        // column 7. An offset that forgot the plan is invisible in the text above and
        // obvious on screen.
        use crossterm::cursor::{MoveRight, MoveToColumn, MoveUp};
        let landing = format!("{}{}{}", MoveUp(1), MoveToColumn(0), MoveRight(7));
        assert!(written.contains(&landing), "the caret lands in the wrong place: {:?}", written);
    }

    #[test]
    fn a_bar_stays_one_row_whatever_it_is_handed() {
        // A multi-line shell command is a plausible summary, and a line feed inside a row
        // the writer counts as one is how the region's height goes wrong.
        for width in [20usize, 40, 80] {
            for content in [
                "",
                "○ ran cargo build",
                "○ ran for f in *.rs\ndo\n  echo $f\ndone",
                &"○ ran ".to_string().repeat(40),
            ] {
                let row = bar(content, width);
                assert!(!row.contains('\n'), "line feed in {:?}", row);
                // Painted the whole way across, and never wider than the writer allows.
                assert_eq!(
                    crate::out::visible_width(&row),
                    width - 1,
                    "width {} with {:?}: {:?}",
                    width,
                    content,
                    row
                );
            }
        }
    }

    #[test]
    fn history_survives_a_multi_line_entry() {
        let directory = std::env::temp_dir().join("haico-history-test");
        let _ = fs::create_dir_all(&directory);
        let path = directory.join("history");

        let entries = vec!["plain".to_string(), "two\nlines".to_string(), "back\\slash".to_string()];
        save_history(&path, &entries);
        assert_eq!(load_history(&path), entries);

        let _ = fs::remove_file(&path);
    }
}
