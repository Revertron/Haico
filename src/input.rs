//! The keyboard: one reader, and the line editor it feeds.
//!
//! rustyline used to own both, and it owned them by blocking: nothing else could
//! happen while a line was being typed, and nothing could be typed while anything
//! else was happening. Reading the keyboard on its own thread and handing the events
//! to an editor we drive ourselves is what makes the second half possible, and it is
//! also the only way to get an input box that grows over more than one line.

use crate::modes::Mode;
use crate::out;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::style::{Color, ResetColor, SetForegroundColor};
use crossterm::tty::IsTty;
use std::fs;
use std::io;
use std::path::Path;
use tokio::sync::mpsc::{error::TryRecvError, unbounded_channel, UnboundedReceiver};

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
    pub(crate) fn start() -> Self {
        let (sender, events) = unbounded_channel();
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
            // Shift+Enter is what most people reach for; Alt+Enter is the one that
            // survives terminals which cannot tell Shift+Enter from Enter.
            KeyCode::Enter if shift || alt => self.split_line(),
            KeyCode::Enter => {
                let text = self.lines.join("\n");
                self.remember(&text);
                self.reset();
                return Some(Input::Text(text));
            }
            KeyCode::BackTab => return Some(Input::CycleMode),
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
            KeyCode::Char(ch) if !control => self.insert(ch),
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
        let width = width.max(prompt.chars().count() + 8);

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

// Everything the bottom of the screen shows besides what is being typed: what the
// model is busy with above the box, and what the session is set to below it.
pub(crate) struct Chrome<'a> {
    pub(crate) mode: Mode,
    pub(crate) busy: Option<&'a str>,
    // The right-hand end of the status bar: which model, and how full its context is.
    pub(crate) status: &'a str,
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
    format!(
        "{}{}{}{}{}",
        SetForegroundColor(Color::DarkGrey),
        left,
        " ".repeat(gap),
        status,
        ResetColor
    )
}

pub(crate) fn draw(editor: &Editor, chrome: &Chrome) {
    let width = terminal_width();
    let (mut rows, caret_row, caret_column) = editor.rows(PROMPT, width);

    let offset = match chrome.busy {
        Some(busy) => {
            rows.insert(
                0,
                format!("{}○ {}{}", SetForegroundColor(Color::DarkGrey), busy, ResetColor),
            );
            1
        }
        None => 0,
    };
    rows.push(status_bar(chrome.mode, chrome.status, width));

    out::set_region(rows, caret_row + offset, caret_column);
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
        match keys.next().await {
            None => return Input::Eof,
            Some(Event::Key(key)) => {
                if let Some(input) = editor.key(key) {
                    out::clear_region();
                    return input;
                }
            }
            Some(Event::Paste(text)) => editor.paste(&text),
            // Redrawn at the top of the loop with the new width.
            Some(Event::Resize(..)) => {}
            Some(_) => {}
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
) -> Input {
    loop {
        draw(
            editor,
            &Chrome {
                mode: *mode,
                busy: None,
                status,
            },
        );
        match keys.next().await {
            None => return Input::Eof,
            Some(Event::Key(key)) => match editor.key(key) {
                Some(Input::CycleMode) => *mode = mode.next(),
                Some(input) => {
                    out::clear_region();
                    return input;
                }
                None => {}
            },
            Some(Event::Paste(text)) => editor.paste(&text),
            Some(Event::Resize(..)) => {}
            Some(_) => {}
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
pub(crate) fn while_working(editor: &mut Editor, mode: &mut Mode, event: Event) -> WhileWorking {
    match event {
        Event::Key(key) => {
            if key.kind == KeyEventKind::Press && key.code == KeyCode::Esc {
                return WhileWorking::Interrupt;
            }
            match editor.key(key) {
                Some(Input::Text(text)) => WhileWorking::Queued(text),
                // Ctrl+C means the same thing here as Esc does.
                Some(Input::Interrupted) => WhileWorking::Interrupt,
                Some(Input::CycleMode) => {
                    *mode = mode.next();
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
    fn the_box_sits_between_the_line_above_it_and_the_bar_below() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "hello");

        let _guard = crate::out::capture_guard();
        crate::out::start_capture();
        draw(
            &editor,
            &Chrome {
                mode: Mode::Plan,
                busy: Some("thinking"),
                status: "a-model  ctx 3%",
            },
        );
        let written = crate::out::take_capture();

        let busy = written.find("○ thinking").expect("the busy line");
        let box_row = written.find("> hello").expect("the input box");
        let bar = written.rfind("a-model  ctx 3%").expect("the status bar");
        assert!(busy < box_row && box_row < bar, "wrong order in {:?}", written);
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
