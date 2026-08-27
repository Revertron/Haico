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
    // Where the last thing committed to the scrollback left the cursor, and what the
    // wrapped rows of the line being written line up under. Only committed output moves
    // it: `pending` is re-wrapped from a copy on every redraw, because a word arriving a
    // fragment at a time has to stay free to move to the next row until its line ends.
    wrap: Wrap,
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

        // Read again rather than cached: a terminal can be resized between two lines of
        // the same reply.
        self.wrap.width = wrap_width();

        if self.region.is_empty() {
            // Nothing is pinned, so there is nothing to hold a half-finished line back
            // for: write it straight out, the way every other program does.
            let wrapped = self.wrap.feed(text);
            self.put(&wrapped);
            return;
        }

        self.pending.push_str(text);
        if let Some(end) = self.pending.rfind('\n') {
            let completed: String = self.pending.drain(..=end).collect();
            let wrapped = self.wrap.feed(&completed);
            self.put(&wrapped);
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

    fn draw_region(&mut self) {
        if self.region.is_empty() || self.region_drawn {
            return;
        }
        if !self.at_line_start {
            self.put("\n");
            // The box starts on a row of its own, which ends the line that was being
            // written just as surely as a newline of its own would have.
            self.wrap.reset_line();
        }

        self.write_out(&Hide.to_string());

        // The unfinished line goes above the box, wrapped from a *copy* of the state so
        // that nothing about it is settled yet: the word at the end of it may still turn
        // out to belong on the next row once the rest of it arrives.
        let pending_rows = if self.pending.is_empty() {
            0
        } else {
            let mut wrap = self.wrap.clone();
            wrap.width = wrap_width();
            let pending = wrap.feed(&self.pending);
            // Each row it was broken into is exactly one screen row, which is the whole
            // point of breaking it here instead of letting the terminal do it.
            let rows = pending.matches('\n').count() + 1;
            self.put(&pending);
            self.write_out(&Clear(ClearType::UntilNewLine).to_string());
            self.put("\n");
            rows
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

// The width text is broken at. A row exactly as wide as the terminal wraps in some
// terminals and not others, so everything stops a column short — the same margin the
// input box keeps, for the same reason.
fn wrap_width() -> usize {
    terminal_width().saturating_sub(1).max(1)
}

// Consume one escape sequence, the ESC itself already taken, and give it back whole. A
// sequence cut in half paints everything after it, so nothing here may ever split one.
fn take_escape(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut escape = String::from('\u{1b}');
    match chars.next() {
        Some('[') => {
            escape.push('[');
            for ch in chars.by_ref() {
                escape.push(ch);
                if ('@'..='~').contains(&ch) {
                    break;
                }
            }
        }
        Some(ch) => escape.push(ch),
        None => {}
    }
    escape
}

// Where the wrapped rows of a line start, so they line up under the text they continue
// rather than under whatever introduced it: past the line's own indentation, and past a
// bullet or a numbered item if it has one. Without this a wrapped list item runs back to
// column zero and reads as the start of a new paragraph.
fn hanging_indent(line: &str, width: usize) -> usize {
    let mut chars = line.chars().peekable();
    let mut indent = 0usize;
    loop {
        match chars.peek() {
            Some('\u{1b}') => {
                chars.next();
                take_escape(&mut chars);
            }
            Some(' ') => {
                chars.next();
                indent += 1;
            }
            _ => break,
        }
    }

    // A marker can have colors around it — the reply's own "● " arrives that way — so it
    // is matched against what actually prints rather than against the raw characters.
    let mut visible = String::new();
    let mut rest = chars.clone();
    while visible.chars().count() < 6 {
        match rest.next() {
            Some('\u{1b}') => {
                take_escape(&mut rest);
            }
            Some(ch) => visible.push(ch),
            None => break,
        }
    }

    let marker = ["- ", "* ", "+ ", "• ", "● ", "○ ", "→ "]
        .into_iter()
        .find(|marker| visible.starts_with(marker))
        .map(|marker| marker.chars().count())
        .or_else(|| numbered_marker(&visible))
        .unwrap_or(0);

    // Half the row is as far as a hanging indent can usefully go; a deeply indented code
    // line past that would leave a column too narrow to read.
    (indent + marker).min(width / 2)
}

// "1. " or "12) ": the width of the number together with the punctuation after it.
fn numbered_marker(visible: &str) -> Option<usize> {
    let digits = visible.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let mut after = visible.chars().skip(digits);
    match (after.next(), after.next()) {
        (Some('.') | Some(')'), Some(' ')) => Some(digits + 2),
        _ => None,
    }
}

// Breaking output into rows at spaces rather than wherever the right edge happens to fall.
//
// Left to itself the terminal cuts words in half, which is what a streamed reply of any
// length used to look like. Doing it here costs two things the terminal never has to
// think about. Escape sequences carry the colors and move the cursor nowhere: they take
// no width and must survive whole. And a run with no space in it that is wider than a
// whole row — a URL, a hash, a base64 blob — has to be broken anyway, because refusing
// would push it off the edge, which is worse than the break that was going to happen
// regardless.
//
// The state is carried across calls because a line does not arrive in one piece: the
// reply's "● " and the words after it are separate writes, and only the first of them
// says where the wrapped rows should line up.
#[derive(Clone)]
struct Wrap {
    width: usize,
    // Columns already used on the current screen row.
    column: usize,
    // What a wrapped row of the current line is indented by, worked out once from the
    // start of the line and kept for the rest of it.
    indent: String,
    // Nothing has been written on this line yet, so its indent is still to be found.
    fresh: bool,
    // Nothing but the indent on the current row, so breaking again would only leave an
    // empty row behind.
    row_empty: bool,
}

impl Wrap {
    fn new(width: usize) -> Self {
        Wrap {
            width: width.max(1),
            column: 0,
            indent: String::new(),
            fresh: true,
            row_empty: true,
        }
    }

    // A line ended for real, by a newline of its own or by the input box taking the row.
    fn reset_line(&mut self) {
        self.column = 0;
        self.indent.clear();
        self.fresh = true;
        self.row_empty = true;
    }

    // The same text with the breaks put in. Newlines that were already there keep their
    // meaning: they end a line, and the next one works out its own indent.
    fn feed(&mut self, text: &str) -> String {
        let mut wrapped = String::with_capacity(text.len() + 16);
        for (index, line) in text.split('\n').enumerate() {
            if index > 0 {
                wrapped.push('\n');
                self.reset_line();
            }
            self.line(line, &mut wrapped);
        }
        wrapped
    }

    fn line(&mut self, line: &str, out: &mut String) {
        if line.is_empty() {
            return;
        }
        if self.fresh {
            self.indent = " ".repeat(hanging_indent(line, self.width));
            self.fresh = false;
        }

        // Almost everything printed is short. Passing it through untouched keeps the
        // common case byte-for-byte what it was before any of this existed.
        let visible = visible_width(line);
        if self.column + visible <= self.width {
            out.push_str(line);
            self.column += visible;
            self.row_empty &= visible == 0;
            return;
        }

        // A word is a run of non-blanks together with any escape sequence touching it:
        // the color has to travel to whichever row the word lands on.
        let mut word = String::new();
        let mut word_width = 0usize;
        let mut gap = String::new();
        let mut gap_width = 0usize;
        let mut chars = line.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '\u{1b}' => word.push_str(&take_escape(&mut chars)),
                // A stray carriage return prints nothing and belongs where it was written.
                '\r' => word.push('\r'),
                ' ' | '\t' => {
                    if !word.is_empty() {
                        self.place(out, &word, word_width, &gap, gap_width);
                        word.clear();
                        word_width = 0;
                        gap.clear();
                        gap_width = 0;
                    }
                    gap.push(ch);
                    gap_width += 1;
                }
                _ => {
                    word.push(ch);
                    word_width += 1;
                }
            }
        }
        self.place(out, &word, word_width, &gap, gap_width);
    }

    // One word, with the blanks that stood in front of it. They are held back until here
    // so that a break can drop them instead of leaving them dangling past the edge.
    fn place(
        &mut self,
        out: &mut String,
        word: &str,
        word_width: usize,
        gap: &str,
        gap_width: usize,
    ) {
        if word.is_empty() {
            return;
        }
        // Escape sequences on their own — the reset closing a colored run — take no width
        // and belong exactly where they were written.
        if word_width == 0 {
            out.push_str(gap);
            self.column += gap_width;
            out.push_str(word);
            return;
        }
        if self.column + gap_width + word_width <= self.width {
            out.push_str(gap);
            self.column += gap_width;
        } else {
            if !self.row_empty {
                self.break_row(out);
            }
            if self.column + word_width > self.width {
                return self.hard(out, word);
            }
        }
        out.push_str(word);
        self.column += word_width;
        self.row_empty = false;
    }

    fn break_row(&mut self, out: &mut String) {
        out.push('\n');
        out.push_str(&self.indent);
        self.column = self.indent.chars().count();
        self.row_empty = true;
    }

    // A word too long for a row of its own, broken wherever the edge falls. There is no
    // better place for it, and letting it run would only move the same break one column
    // to the right.
    fn hard(&mut self, out: &mut String, word: &str) {
        let mut chars = word.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\u{1b}' {
                let escape = take_escape(&mut chars);
                out.push_str(&escape);
                continue;
            }
            if ch == '\r' {
                out.push('\r');
                continue;
            }
            if self.column >= self.width && !self.row_empty {
                self.break_row(out);
            }
            out.push(ch);
            self.column += 1;
            self.row_empty = false;
        }
    }
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
            wrap: Wrap::new(80),
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
        state.wrap.width = wrap_width();
        let wrapped = state.wrap.feed(&pending);
        state.put(&wrapped);
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
    state.wrap = Wrap::new(80);
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

    // Wrapping is tested at a width given here rather than the terminal's, which under a
    // test runner is whatever happens to be attached.
    fn wrapped(text: &str, width: usize) -> String {
        super::Wrap::new(width).feed(text)
    }

    fn no_row_overflows(wrapped: &str, width: usize) {
        for row in wrapped.split('\n') {
            assert!(
                super::visible_width(row) <= width,
                "row wider than {}: {:?}",
                width,
                row
            );
        }
    }

    #[test]
    fn text_breaks_between_words_rather_than_through_them() {
        const TEXT: &str = "the quick brown fox jumps over the lazy dog and keeps going";

        let rows = wrapped(TEXT, 20);
        assert!(rows.contains('\n'), "nothing was wrapped: {:?}", rows);
        no_row_overflows(&rows, 20);

        // Every break landed on a space, so putting the rows back together with one space
        // each returns exactly what went in.
        let joined = rows.split('\n').collect::<Vec<_>>().join(" ");
        assert_eq!(joined, TEXT);
    }

    #[test]
    fn a_word_wider_than_a_row_is_broken_anyway() {
        const TEXT: &str = "see https://example.com/a/very/long/path/that/never/breaks now";

        let rows = wrapped(TEXT, 20);
        no_row_overflows(&rows, 20);

        // Broken, but nothing lost or repeated on the way.
        let printed: String = rows.chars().filter(|ch| !ch.is_whitespace()).collect();
        let want: String = TEXT.chars().filter(|ch| !ch.is_whitespace()).collect();
        assert_eq!(printed, want);
    }

    #[test]
    fn wrapped_rows_line_up_under_the_text_they_continue() {
        let rows = wrapped("  - a list item long enough to need a second row", 24);
        let rows: Vec<&str> = rows.split('\n').collect();
        assert!(rows.len() > 1, "nothing was wrapped: {:?}", rows);
        for row in &rows[1..] {
            assert!(
                row.starts_with("    ") && !row.starts_with("     "),
                "not under the item's text: {:?}",
                row
            );
        }

        // And the reply's own bullet counts, even though it is written separately from
        // the words that follow it.
        let mut wrap = super::Wrap::new(24);
        let mut rows = wrap.feed("● ");
        rows.push_str(&wrap.feed("an answer long enough to wrap"));
        let rows: Vec<&str> = rows.split('\n').collect();
        assert!(rows.len() > 1, "nothing was wrapped: {:?}", rows);
        for row in &rows[1..] {
            assert!(
                row.starts_with("  ") && !row.starts_with("   "),
                "not under the answer: {:?}",
                row
            );
        }
    }

    #[test]
    fn colors_survive_the_break_and_never_count_toward_the_width() {
        const TEXT: &str =
            "plain \u{1b}[1mbold words in here\u{1b}[0m and more plain text after that";

        let rows = wrapped(TEXT, 20);
        no_row_overflows(&rows, 20);
        // Cut in half, either sequence would paint everything after it.
        assert_eq!(rows.matches("\u{1b}[1m").count(), 1, "got {:?}", rows);
        assert_eq!(rows.matches("\u{1b}[0m").count(), 1, "got {:?}", rows);
        assert_eq!(rows.split('\n').collect::<Vec<_>>().join(" "), TEXT);
    }


    #[test]
    fn a_line_that_fits_is_left_exactly_as_it_was() {
        assert_eq!(wrapped("short enough\n", 40), "short enough\n");
        assert_eq!(wrapped("", 40), "");
        // Blank lines keep their rows.
        assert_eq!(wrapped("one\n\ntwo\n", 40), "one\n\ntwo\n");
    }
}
