//! The line editor: what is being typed, the history behind it, and the slash-command
//! suggestions above it.

use super::*;

pub(crate) struct Editor {
    // One entry per logical line: Shift+Enter adds one, Enter submits them all.
    pub(super) lines: Vec<String>,
    row: usize,
    // In characters, not bytes.
    column: usize,
    history: Vec<String>,
    // Where in the history we are while browsing it, and what was being typed before
    // browsing started, so walking back down returns it rather than losing it.
    browsing: Option<usize>,
    stash: Vec<String>,
    // What a `/` at the start of the box offers. Empty for the editors the dialogs make,
    // which is what keeps the list out of them: a reply to a permission prompt that starts
    // with a slash is a path, not a command.
    commands: Vec<Command>,
    // Which row of the list is picked. Back to the top whenever the query changes, since
    // the row it pointed at is then a different command or gone.
    picked: usize,
    // The entry as it stood when Esc closed the list. It stays closed until the entry
    // changes, rather than reopening on the very next redraw.
    dismissed: Option<String>,
}

// How many suggestions show at once. The list scrolls to keep the picked one in view.
pub(super) const SUGGESTION_ROWS: usize = 5;

// The widest a name column gets; a longer skill name is cut rather than pushing every
// description over.
const SUGGESTION_NAME_COLUMNS: usize = 24;

// The picked row's color, and the marker beside it — the marker being what still shows
// which row it is under NO_COLOR.
const SUGGESTION_PICKED: Color = Color::AnsiValue(68);
const SUGGESTION_MARKER: &str = "› ";

impl Editor {
    pub(crate) fn new(history: Vec<String>) -> Self {
        Editor {
            lines: vec![String::new()],
            row: 0,
            column: 0,
            history,
            browsing: None,
            stash: Vec::new(),
            commands: Vec::new(),
            picked: 0,
            dismissed: None,
        }
    }

    // Re-read where the skills could have changed: at startup, after /update-skills, and
    // after /cd, since the project's own skills moved with it.
    pub(crate) fn set_commands(&mut self, commands: Vec<Command>) {
        self.commands = commands;
        self.picked = 0;
    }

    // What follows the `/`, while the entry is still nothing but the start of a command.
    // Not while walking the history: there Up and Down belong to the history, and an old
    // `/compact` coming round must not trap them in a list.
    fn query(&self) -> Option<&str> {
        if self.commands.is_empty() || self.lines.len() != 1 || self.browsing.is_some() {
            return None;
        }
        let rest = self.lines[0].strip_prefix('/')?;
        if rest.chars().any(char::is_whitespace) {
            return None;
        }
        if self.dismissed.as_deref() == Some(self.lines[0].as_str()) {
            return None;
        }
        Some(rest)
    }

    fn suggestions(&self) -> Vec<Suggestion> {
        match self.query() {
            Some(query) => commands::suggest(&self.commands, query),
            None => Vec::new(),
        }
    }

    // Put a picked command in the box, with a space after it when it takes arguments, so
    // the next thing typed is the argument.
    fn fill(&mut self, command: &Command) {
        let space = if command.args == Args::None { "" } else { " " };
        self.lines = vec![format!("/{}{}", command.name, space)];
        self.row = 0;
        self.column = self.line_length();
    }

    // The keys the list takes while it is open. `None` lets the key through to the editor.
    fn suggestion_key(&mut self, key: &KeyEvent) -> Option<Option<Input>> {
        let suggestions = self.suggestions();
        if suggestions.is_empty() {
            return None;
        }
        let count = suggestions.len();
        let picked = self.picked.min(count - 1);
        let command = self.commands[suggestions[picked].index].clone();
        let modified = key.modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT | KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Up => self.picked = (picked + count - 1) % count,
            KeyCode::Down => self.picked = (picked + 1) % count,
            KeyCode::Tab => self.fill(&command),
            KeyCode::Esc => self.dismissed = Some(self.lines[0].clone()),
            // A command that cannot run bare goes into the box to be finished; any other
            // runs, as though its whole name had been typed.
            KeyCode::Enter if !modified => {
                if command.args == Args::Required {
                    self.fill(&command);
                } else {
                    let text = format!("/{}", command.name);
                    self.remember(&text);
                    self.reset();
                    return Some(Some(Input::Text(text)));
                }
            }
            _ => return None,
        }
        Some(None)
    }

    // The list, one row per suggestion, topmost first. Empty while it is closed.
    pub(crate) fn suggestion_rows(&self, width: usize) -> Vec<String> {
        let suggestions = self.suggestions();
        let room = width.saturating_sub(1);
        if suggestions.is_empty() || room == 0 {
            return Vec::new();
        }
        let picked = self.picked.min(suggestions.len() - 1);
        let first = (picked + 1).saturating_sub(SUGGESTION_ROWS);

        // Wide enough for every name in the list, not only the ones in view, so the
        // descriptions do not shift sideways as it scrolls.
        let column = suggestions
            .iter()
            .map(|suggestion| self.commands[suggestion.index].name.chars().count() + 1)
            .max()
            .unwrap_or(0)
            .min(SUGGESTION_NAME_COLUMNS);

        suggestions
            .iter()
            .enumerate()
            .skip(first)
            .take(SUGGESTION_ROWS)
            .map(|(position, suggestion)| {
                let command = &self.commands[suggestion.index];
                let is_picked = position == picked;
                let marker = if is_picked { SUGGESTION_MARKER } else { "  " };
                let name = fit(&format!("/{}", command.name), column);
                let pad = column.saturating_sub(name.chars().count());
                let about_room = room.saturating_sub(marker.chars().count() + column + 2);
                let about = fit(&command.about, about_room);

                // The hit is counted in the name without its slash.
                let hit = suggestion.hit.map(|(start, end)| (start + 1, end + 1));
                let mut painted = String::new();
                for (at, ch) in name.chars().enumerate() {
                    if hit.map(|(start, _)| start == at).unwrap_or(false) {
                        painted.push_str(&SetAttribute(Attribute::Bold).to_string());
                    }
                    if hit.map(|(_, end)| end == at).unwrap_or(false) {
                        painted.push_str(&SetAttribute(Attribute::NormalIntensity).to_string());
                    }
                    painted.push(ch);
                }
                painted.push_str(&SetAttribute(Attribute::NormalIntensity).to_string());

                let gap = " ".repeat(pad + 2);
                if is_picked {
                    format!(
                        "{}{}{}{}{}{}",
                        SetForegroundColor(SUGGESTION_PICKED),
                        marker,
                        painted,
                        gap,
                        about,
                        ResetColor
                    )
                } else {
                    format!("{}{}{}{}", marker, painted, gap, dim(about))
                }
            })
            .collect()
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
        if let Some(taken) = self.suggestion_key(&key) {
            return taken;
        }
        let before = self.lines.clone();
        let input = self.edit(key);
        if self.lines != before {
            self.picked = 0;
        }
        input
    }

    fn edit(&mut self, key: KeyEvent) -> Option<Input> {
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

            let points = wrap_points(&text, width, prefix.chars().count());
            if points.is_empty() {
                rows.push(String::new());
            } else {
                rows.extend(
                    points
                        .iter()
                        .map(|&(start, end, _)| text[start..end].iter().collect::<String>()),
                );
            }

            if index == self.row {
                let at = prefix.chars().count() + self.column;
                let (row, column) = caret_in(&points, at, width);
                caret = (first + row, column);
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

// Where each row of one wrapped line begins and ends, broken at a blank rather than
// wherever the right edge happens to fall. What this used to do — and what the terminal
// does when left to it — cuts a word through the middle, which is what a pasted paragraph
// looked like in the box while the same paragraph in the transcript above wrapped between
// words. `out::Wrap` already does this for output, but not in a form that can be reused
// here: it works on a string and reports only the text, and the box needs to know where
// the breaks landed in order to put the caret on the right one.
//
// A row is `start..end`, and `next` is where the row after it begins. The two differ by
// the one blank a break swallows: the space at the edge *is* the break and is not drawn
// again at the start of the row below, but the caret can still be sitting on it, which is
// the only reason `next` is reported apart from `end`.
//
// `floor` is the first index a break may fall on. Without it the prompt's own trailing
// space is a break like any other, and a first word too long for the row would wrap to a
// row holding nothing but ">".
fn wrap_points(text: &[char], width: usize, floor: usize) -> Vec<(usize, usize, usize)> {
    let mut points = Vec::new();
    let mut start = 0;

    while start < text.len() {
        if text.len() - start <= width {
            points.push((start, text.len(), text.len()));
            break;
        }
        // The row holds `width` characters, and a blank in the column just past it is a
        // break that costs nothing — the row in front of it is full either way.
        let blank = (start.max(floor)..=start + width)
            .rev()
            .find(|&index| matches!(text[index], ' ' | '\t'))
            // A blank the row starts with is no break at all: honoring it would emit an
            // empty row and leave the word exactly where it was.
            .filter(|&index| index > start);

        match blank {
            Some(index) => {
                points.push((start, index, index + 1));
                start = index + 1;
            }
            // A word too long for a row of its own, broken wherever the edge falls.
            // There is no better place for it, and refusing would push it off the screen
            // instead of moving the break one column.
            None => {
                points.push((start, start + width, start + width));
                start += width;
            }
        }
    }

    points
}

// Which row of a wrapped line the caret is on, and where along it.
//
// Not arithmetic any more, now that the rows are not all the same length: the caret is on
// the first row whose text still reaches it. A caret on a swallowed blank has no column
// of its own and sits at the end of the row that break ended.
fn caret_in(points: &[(usize, usize, usize)], at: usize, width: usize) -> (usize, usize) {
    for (row, &(start, end, next)) in points.iter().enumerate() {
        if at < next {
            return (row, (at - start).min(end - start));
        }
    }

    // One past the last character, which is where a caret at the end of the line is: it
    // belongs immediately after that character, unless the row is full and has no column
    // left to stand in — then it takes the row below, the one the caller is about to add.
    match points.last() {
        Some(&(start, end, _)) if at == end && end - start < width => {
            (points.len() - 1, end - start)
        }
        _ => (points.len(), 0),
    }
}

// The input box needs room for a word and its caret, so it never lays out narrower than this.
pub(super) fn terminal_width() -> usize {
    out::terminal_width().max(20)
}

pub(super) fn terminal_height() -> usize {
    out::terminal_height()
}
