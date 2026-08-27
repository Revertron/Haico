//! The one place that writes to the terminal.
//!
//! Everything the REPL shows goes through here rather than reaching `println!`
//! directly. Two things need that. Output arriving while the model works has to be
//! able to step around a live input line, which only works if one writer knows what
//! is currently on screen; and raw mode drops the terminal's own newline
//! translation, so "\n" has to become "\r\n" in one place rather than at two
//! hundred call sites.
//!
//! The bottom of the screen is a *region*: a few rows holding the input box, owned
//! by this module and redrawn around anything else that gets printed. Finished
//! output still scrolls into the terminal's own scrollback above it, so selecting
//! and copying it works the way it does in any other program.

use crossterm::cursor::{Hide, MoveRight, MoveToColumn, MoveUp, Show};
use crossterm::terminal::{Clear, ClearType};
use std::fmt;
use std::io::{self, Write};
use std::sync::{Mutex, MutexGuard, OnceLock};

struct Out {
    raw: bool,
    // Whether the cursor is sitting at the start of a row. The region has to begin on
    // one of its own, so a half-finished line gets a newline before it is drawn.
    at_line_start: bool,
    // The rows of the input region, already wrapped to the terminal width, and where
    // the cursor belongs inside them. Empty when there is no region on screen.
    region: Vec<String>,
    caret: (usize, usize),
    region_drawn: bool,
    // How far below the top of the drawn block the cursor was left, so erasing it
    // walks back exactly as far as drawing walked forward.
    drawn_caret_row: usize,
    // A line of output that has not ended yet. The model streams a few characters at
    // a time, and a half-finished line cannot go into the scrollback above the input
    // box — the next fragment would have nowhere to continue from. It waits here and
    // is drawn just above the box until its newline arrives.
    pending: String,
    // Set by the tests so they can assert on what would have been written.
    capture: Option<String>,
}

impl Out {
    // Show something, stepping around whatever is currently pinned to the bottom.
    fn emit(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.erase_region();

        if self.region.is_empty() {
            // Nothing is pinned, so there is nothing to hold a half-finished line back
            // for: write it straight out, the way every other program does.
            self.put(text);
            return;
        }

        self.pending.push_str(text);
        if let Some(end) = self.pending.rfind('\n') {
            let completed: String = self.pending.drain(..=end).collect();
            self.put(&completed);
        }
        self.draw_region();
    }

    fn put(&mut self, text: &str) {
        // Raw mode turns off the terminal's newline translation, so a bare line feed
        // moves down a row without returning to column 0 and the output walks
        // diagonally off the right of the screen. A line feed that already has its
        // carriage return is left alone.
        if self.raw && text.contains('\n') {
            let mut translated = String::with_capacity(text.len() + 8);
            let mut previous = '\0';
            for ch in text.chars() {
                if ch == '\n' && previous != '\r' {
                    translated.push('\r');
                }
                translated.push(ch);
                previous = ch;
            }
            self.write_out(&translated);
        } else {
            self.write_out(text);
        }

        if let Some(last) = text.chars().last() {
            self.at_line_start = last == '\n';
        }
    }

    fn write_out(&mut self, text: &str) {
        if let Some(buffer) = &mut self.capture {
            buffer.push_str(text);
        } else {
            let _ = io::stdout().write_all(text.as_bytes());
        }
    }

    // Take the region off the screen, leaving the cursor where the region started.
    // Everything below that point is the region's own, so clearing to the end of the
    // screen cannot eat anything else.
    fn erase_region(&mut self) {
        if !self.region_drawn {
            return;
        }
        self.region_drawn = false;

        let mut escapes = format!("{}{}", Hide, MoveToColumn(0));
        if self.drawn_caret_row > 0 {
            escapes.push_str(&MoveUp(self.drawn_caret_row as u16).to_string());
        }
        escapes.push_str(&Clear(ClearType::FromCursorDown).to_string());
        escapes.push_str(&Show.to_string());
        self.write_out(&escapes);
        self.at_line_start = true;
    }

    // How many screen rows a line of text takes up, given that the terminal wraps it
    // itself. Escape sequences move the cursor nowhere, so they do not count.
    fn rows_used(text: &str) -> usize {
        let width = terminal_width();
        let visible = visible_width(text);
        if visible == 0 {
            1
        } else {
            (visible - 1) / width + 1
        }
    }

    fn draw_region(&mut self) {
        if self.region.is_empty() || self.region_drawn {
            return;
        }
        if !self.at_line_start {
            self.put("\n");
        }

        self.write_out(&Hide.to_string());

        // The unfinished line goes above the box, and its own wrapping is left to the
        // terminal: it carries colors, and re-wrapping it here would mean cutting
        // escape sequences in half.
        let pending_rows = if self.pending.is_empty() {
            0
        } else {
            let pending = self.pending.clone();
            self.put(&pending);
            self.write_out(&Clear(ClearType::UntilNewLine).to_string());
            self.put("\n");
            Self::rows_used(&pending)
        };

        for (index, row) in self.region.clone().iter().enumerate() {
            if index > 0 {
                self.put("\n");
            }
            self.put(row);
            // The row that was here before may have been longer.
            self.write_out(&Clear(ClearType::UntilNewLine).to_string());
        }

        // The cursor ends up after the last row; walk it back to where the caret is.
        let last = pending_rows + self.region.len() - 1;
        let caret_row = pending_rows + self.caret.0;
        let mut escapes = String::new();
        if last > caret_row {
            escapes.push_str(&MoveUp((last - caret_row) as u16).to_string());
        }
        escapes.push_str(&MoveToColumn(0).to_string());
        if self.caret.1 > 0 {
            escapes.push_str(&MoveRight(self.caret.1 as u16).to_string());
        }
        escapes.push_str(&Show.to_string());
        self.write_out(&escapes);

        self.drawn_caret_row = caret_row;
        self.region_drawn = true;
        if self.capture.is_none() {
            let _ = io::stdout().flush();
        }
    }
}

fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(columns, _)| columns as usize)
        .unwrap_or(80)
        .max(1)
}

// How wide a string prints. Escape sequences take no space, and counting them would
// make every colored line look several times longer than it is.
pub(crate) fn visible_width(text: &str) -> usize {
    let mut width = 0;
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            width += 1;
            continue;
        }
        // CSI sequences run until a byte in @..~; anything else is a two-character
        // escape. Either way none of it lands on the screen.
        match chars.next() {
            Some('[') => {
                for ch in chars.by_ref() {
                    if ('@'..='~').contains(&ch) {
                        break;
                    }
                }
            }
            Some(_) | None => {}
        }
    }
    width
}

// A panic while the lock is held would otherwise poison it and take the whole
// interface down with it, which is a bad trade for a line of output.
fn lock() -> MutexGuard<'static, Out> {
    static OUT: OnceLock<Mutex<Out>> = OnceLock::new();
    OUT.get_or_init(|| {
        Mutex::new(Out {
            raw: false,
            at_line_start: true,
            region: Vec::new(),
            caret: (0, 0),
            region_drawn: false,
            drawn_caret_row: 0,
            pending: String::new(),
            capture: None,
        })
    })
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn write_fmt(args: fmt::Arguments) {
    lock().emit(&args.to_string());
}

pub(crate) fn write_line(args: fmt::Arguments) {
    let mut text = args.to_string();
    text.push('\n');
    lock().emit(&text);
}

// Start a fresh row, unless the last thing written already ended one.
//
// The caller that follows a stream cannot know whether it stopped mid-line: a model that
// ended its message on a newline needs nothing, and one that did not needs exactly one.
// Emitting a newline either way is what put a blank row after every reply. `pending`
// counts as unfinished even though nothing reached the screen — it is a line waiting for
// its ending, and the next thing written would otherwise land on the same row.
pub(crate) fn ensure_line_start() {
    let mut out = lock();
    if !out.at_line_start || !out.pending.is_empty() {
        out.emit("\n");
    }
}

pub(crate) fn flush() {
    if lock().capture.is_none() {
        let _ = io::stdout().flush();
    }
}

// Whether line feeds need a carriage return in front of them. On for as long as the
// input loop holds raw mode.
pub(crate) fn set_raw(raw: bool) {
    lock().raw = raw;
}

// Put the input box on the bottom of the screen, or move it. The rows are physical
// ones: whatever draws them is responsible for wrapping them to the terminal width,
// because it is also the only thing that knows where the cursor belongs afterwards.
pub(crate) fn set_region(rows: Vec<String>, caret_row: usize, caret_col: usize) {
    let mut state = lock();
    state.erase_region();
    state.region = rows;
    state.caret = (caret_row, caret_col);
    state.draw_region();
}

// Whether anything is pinned to the bottom of the screen at all. What has a permanent
// place down there does not also need to be written into the transcript — and a piped
// session, which has no region, still needs it written.
pub(crate) fn has_region() -> bool {
    !lock().region.is_empty()
}

// Take the input box off the screen and leave it off. Whatever line was waiting for
// its newline is written out as it stands: with nothing pinned below it, the next
// thing printed can simply continue it.
pub(crate) fn clear_region() {
    let mut state = lock();
    state.erase_region();
    state.region.clear();
    state.caret = (0, 0);
    if !state.pending.is_empty() {
        let pending = std::mem::take(&mut state.pending);
        state.put(&pending);
    }
}

// The writer is a global, so two tests capturing at once would read each other's
// output. Everything that captures takes this first.
#[cfg(test)]
pub(crate) fn capture_guard() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
pub(crate) fn start_capture() {
    let mut state = lock();
    state.raw = false;
    state.at_line_start = true;
    state.region.clear();
    state.caret = (0, 0);
    state.region_drawn = false;
    state.drawn_caret_row = 0;
    state.pending.clear();
    state.capture = Some(String::new());
}

#[cfg(test)]
pub(crate) fn take_capture() -> String {
    let mut state = lock();
    state.raw = false;
    state.capture.take().unwrap_or_default()
}

macro_rules! out {
    ($($arg:tt)*) => ($crate::out::write_fmt(format_args!($($arg)*)));
}

macro_rules! outln {
    () => ($crate::out::write_line(format_args!("")));
    ($($arg:tt)*) => ($crate::out::write_line(format_args!($($arg)*)));
}

#[cfg(test)]
mod tests {
    #[test]
    fn raw_mode_returns_the_carriage_on_every_line_feed() {
        let _guard = super::capture_guard();

        super::start_capture();
        outln!("one");
        out!("two\nthree");
        assert_eq!(super::take_capture(), "one\ntwo\nthree");

        super::start_capture();
        super::set_raw(true);
        outln!("one");
        out!("two\nthree");
        // And text that already has its carriage returns is passed through as it is.
        out!("\r\nfour");
        assert_eq!(super::take_capture(), "one\r\ntwo\r\nthree\r\nfour");
    }

    #[test]
    fn output_steps_over_the_region_and_puts_it_back() {
        let _guard = super::capture_guard();

        // A line printed while the region is up: the region comes off first, the line
        // lands in the scrollback, and the region goes back on below it.
        super::start_capture();
        super::set_region(vec!["> hello".to_string()], 0, 7);
        outln!("a result");
        let written = super::take_capture();
        let erase = written.find("[J").expect("the region should be cleared first");
        let line = written.find("a result").expect("the line should be written");
        let redraw = written.rfind("> hello").expect("the region should be redrawn");
        assert!(erase < line && line < redraw, "wrong order in {:?}", written);

        // And once it is cleared it stays off.
        super::start_capture();
        super::set_region(vec!["> hello".to_string()], 0, 7);
        super::clear_region();
        let written = super::take_capture();
        let gone = written.rfind("[J").expect("the region should be cleared");
        assert!(
            written[gone..].find("> hello").is_none(),
            "the region came back in {:?}",
            written
        );
    }

    #[test]
    fn a_half_finished_line_waits_above_the_region_until_its_newline() {
        let _guard = super::capture_guard();

        super::start_capture();
        super::set_region(vec!["> ".to_string()], 0, 2);

        // Streamed in pieces, none of which ends the line. Each redraw carries the whole
        // line so far, so the fragments never land on top of each other — and the
        // newline is what finally commits it to the scrollback, exactly once.
        out!("one ");
        out!("two");
        out!(" three\n");
        let written = super::take_capture();
        assert!(written.contains("one two three\n"), "got {:?}", written);
        assert_eq!(written.matches("one two three").count(), 1, "got {:?}", written);

        // Taking the region away flushes whatever is still waiting for its newline.
        super::start_capture();
        super::set_region(vec!["> ".to_string()], 0, 2);
        out!("tail");
        super::clear_region();
        let written = super::take_capture();
        assert!(written.ends_with("tail"), "got {:?}", written);
    }
}
