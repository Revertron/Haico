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
//!
//! That is the *inline* renderer, and it is still here. The other one — `Screen`, below —
//! takes the alternate screen and owns the whole viewport instead. It costs the
//! terminal's own scrollback and buys the three things that are impossible without it:
//! a transcript that can be scrolled with the input box staying put, one that re-wraps
//! when the window is resized, and one whose lines can still be changed after they were
//! written. Which one runs is decided once at startup by `config.tui`.

use crossterm::cursor::{Hide, MoveDown, MoveRight, MoveTo, MoveToColumn, MoveUp, Show};
use crossterm::terminal::{Clear, ClearType};
use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Write};
use std::sync::{Mutex, MutexGuard, OnceLock};
use unicode_width::UnicodeWidthChar;

struct Out {
    raw: bool,
    // Write text only, every escape dropped. For output that is going to a file or a
    // pipe rather than a screen — see `set_plain`.
    plain: bool,
    // Whether the cursor is sitting at the start of a row. The region has to begin on
    // one of its own, so a half-finished line gets a newline before it is drawn.
    at_line_start: bool,
    // The rows of the input region, already wrapped to the terminal width, and where
    // the cursor belongs inside them. Empty when there is no region on screen.
    region: Vec<String>,
    caret: (usize, usize),
    region_drawn: bool,
    // The rows of the block exactly as they are on the screen, in the order they were
    // written: the unfinished line first, then the region. The next frame is compared
    // against this, so only the rows that actually say something different are painted
    // again — the whole reason a token counter ticking away in the bar no longer makes
    // the plan panel and the input box blink along with it.
    drawn: Vec<String>,
    // Where in that block the cursor was left, so moving about inside it — and erasing
    // it — walks back exactly as far as drawing walked forward.
    drawn_caret: (usize, usize),
    // How wide the screen was when it was drawn. Every row was laid out for that width,
    // so a different one means none of them can be patched.
    drawn_width: usize,
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
    // The full-screen renderer, when it is the one running. While this is `Some` none of
    // the inline machinery above is used: nothing is committed to the terminal's
    // scrollback, so there is no half-finished line to hold back and no region to step
    // around — every row on the screen is redrawn from the transcript.
    screen: Option<Screen>,
}

impl Out {
    // Show something, stepping around whatever is currently pinned to the bottom.
    fn emit(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        if self.screen.is_some() {
            self.screen_write(text);
            return;
        }

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
            // The line is finished, so it goes to the scrollback — and that scrolls the
            // screen, which the block cannot be on while it happens. Nothing else moves
            // the block: text that only makes the unfinished line longer is painted into
            // the rows that line already has.
            self.erase_region();
            let completed: String = self.pending.drain(..=end).collect();
            let wrapped = self.wrap.feed(&completed);
            self.put(&wrapped);
        }
        self.paint();
    }

    // The rows the block should be showing: the unfinished line, wrapped, and then the
    // region itself.
    //
    // The line is wrapped from a *copy* of the state so that nothing about it is settled
    // yet — the word at the end of it may still turn out to belong on the next row once
    // the rest of it arrives. Each row it is broken into is exactly one screen row, which
    // is the whole point of breaking it here instead of letting the terminal do it.
    fn frame(&self, width: usize) -> (Vec<String>, usize) {
        let mut rows = Vec::with_capacity(self.region.len() + 2);
        if !self.pending.is_empty() {
            let mut wrap = self.wrap.clone();
            wrap.width = width;
            let pending = wrap.feed(&self.pending);
            // Every row carries the colors it is written in, including the ones switched
            // on before the break that put it here. `patch` rewrites only the rows that
            // changed, so a row painted on its own starts in whatever color the terminal
            // was last left in — and a word carried onto a new row by the wrap arrives
            // with its escape left behind on the row above. That is what used to leave
            // the first word of a wrapped line of thinking in the terminal's own text
            // color until the line ended and the whole of it went to the scrollback in
            // one piece.
            let mut open = String::new();
            for row in pending.split('\n') {
                rows.push(format!("{}{}", open, row));
                open = open_colors(&open, row);
            }
        }
        let pending_rows = rows.len();
        rows.extend(self.region.iter().cloned());
        (rows, pending_rows)
    }

    // Bring the screen up to date with what the block should say.
    fn paint(&mut self) {
        if self.region.is_empty() {
            // Nothing to pin any more: whatever was there comes off the screen.
            self.erase_region();
            return;
        }

        let width = wrap_width();
        let (rows, pending_rows) = self.frame(width);
        let caret = (pending_rows + self.caret.0, self.caret.1);

        // A block of a different height, or one laid out for a different screen, cannot be
        // patched row by row: everything below the first difference has moved.
        if self.region_drawn && width == self.drawn_width && rows.len() == self.drawn.len() {
            self.patch(&rows, caret);
        } else {
            self.erase_region();
            self.draw(&rows, caret);
        }

        self.drawn = rows;
        self.drawn_caret = caret;
        self.drawn_width = width;
        self.region_drawn = true;
        if self.capture.is_none() {
            let _ = io::stdout().flush();
        }
    }

    // Rewrite the rows that differ and nothing else, stepping over the ones that already
    // say the right thing rather than clearing them: a row that is never touched cannot
    // blink. While a reply streams this is one row — the activity bar, whose token counter
    // moves several times a second — out of a block a dozen rows tall.
    //
    // Not one line feed is written, which is also what keeps the update to a single write:
    // stdout flushes on newlines, and a frame split across flushes is exactly what a
    // half-drawn block looks like.
    fn patch(&mut self, rows: &[String], caret: (usize, usize)) {
        let changed: Vec<usize> = (0..rows.len())
            .filter(|&index| rows[index] != self.drawn[index])
            .collect();
        if changed.is_empty() && caret == self.drawn_caret {
            return;
        }

        self.write_out(&Hide.to_string());
        let mut at = self.drawn_caret.0;
        for index in changed {
            self.move_to_row(at, index);
            self.put(&rows[index]);
            // The row that was here before may have been longer.
            self.write_out(&Clear(ClearType::UntilNewLine).to_string());
            at = index;
        }
        self.move_to_row(at, caret.0);
        self.place_caret(caret.1);
    }

    // Draw the whole block from the top down. Only ever reached with the screen already
    // clear from here on: either nothing has been drawn yet, or the block just came off.
    fn draw(&mut self, rows: &[String], caret: (usize, usize)) {
        if !self.at_line_start {
            self.put("\n");
            // The box starts on a row of its own, which ends the line that was being
            // written just as surely as a newline of its own would have.
            self.wrap.reset_line();
        }

        self.write_out(&Hide.to_string());
        for (index, row) in rows.iter().enumerate() {
            if index > 0 {
                self.put("\n");
            }
            self.put(row);
            // The row that was here before may have been longer.
            self.write_out(&Clear(ClearType::UntilNewLine).to_string());
        }

        // The cursor ends up after the last row; walk it back to where the caret is.
        self.move_to_row(rows.len() - 1, caret.0);
        self.place_caret(caret.1);
    }

    // Put the cursor at the start of another row of the block. Both moves are guarded
    // rather than let through as zero: `MoveUp(0)` is CSI 0 A, which a terminal reads as
    // one row, not none.
    fn move_to_row(&mut self, from: usize, to: usize) {
        let mut escapes = String::new();
        if to > from {
            escapes.push_str(&MoveDown((to - from) as u16).to_string());
        } else if from > to {
            escapes.push_str(&MoveUp((from - to) as u16).to_string());
        }
        escapes.push_str(&MoveToColumn(0).to_string());
        self.write_out(&escapes);
    }

    // Along the row the cursor is already on, and visible again.
    fn place_caret(&mut self, column: usize) {
        let mut escapes = String::new();
        if column > 0 {
            escapes.push_str(&MoveRight(column as u16).to_string());
        }
        escapes.push_str(&Show.to_string());
        self.write_out(&escapes);
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
        } else if self.plain {
            let _ = io::stdout().write_all(without_escapes(text).as_bytes());
        } else {
            let _ = io::stdout().write_all(for_the_terminal(text).as_bytes());
        }
    }

    // The two ways into the full-screen renderer. Both destructure `self` rather than
    // reaching through it, because the screen needs `&mut` while the region it is drawing
    // needs `&` — borrowing the fields apart is what lets one call do both.
    fn screen_write(&mut self, text: &str) {
        let Out { screen, region, caret, capture, .. } = self;
        if let Some(screen) = screen {
            screen.write(text);
            screen.paint(region, *caret, capture, (terminal_width(), terminal_height()));
        }
    }

    fn screen_paint(&mut self) {
        let Out { screen, region, caret, capture, .. } = self;
        if let Some(screen) = screen {
            screen.paint(region, *caret, capture, (terminal_width(), terminal_height()));
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
        self.drawn.clear();

        let mut escapes = format!("{}{}", Hide, MoveToColumn(0));
        if self.drawn_caret.0 > 0 {
            escapes.push_str(&MoveUp(self.drawn_caret.0 as u16).to_string());
        }
        escapes.push_str(&Clear(ClearType::FromCursorDown).to_string());
        escapes.push_str(&Show.to_string());
        self.write_out(&escapes);
        self.at_line_start = true;
    }
}

// ---- The full-screen renderer ------------------------------------------------------

// Taking and giving back the alternate screen.
// The gray of everything that is not the answer: tool calls and results, the chain of
// thought, the turn summary. A fixed 256-color gray rather than `DarkGrey`, which is palette
// slot 8 ("bright black") and so the theme's to choose: several dark themes set it within a
// shade of their own background, and every dim line on screen vanished — tool arguments,
// results, the thinking — leaving only the cyan tool names. The grayscale ramp (232-255) is
// not part of any theme, so 245 is a mid gray on a black screen and on a white one alike.
pub(crate) const DIM: crossterm::style::Color = crossterm::style::Color::AnsiValue(245);

const ENTER_ALT: &str = "\x1b[?1049h";
const LEAVE_ALT: &str = "\x1b[?1049l";

// A frame is written between these two so the terminal shows the finished frame rather
// than the middle of one. This is the whole of "flicker-free": a frame that repaints many
// rows takes several milliseconds to write, and without the pair around it the screen can
// be caught halfway through. Terminals that do not know the sequence ignore it.
const BEGIN_SYNC: &str = "\x1b[?2026h";
const END_SYNC: &str = "\x1b[?2026l";

// Auto-wrap is off while a frame is painted and on again afterwards.
//
// Every row is written at an absolute position and is supposed to be exactly one row. A
// row that turns out to be wider than the screen — a tab counted as one column and drawn
// as eight, a double-width character counted as one and drawn as two, an off-by-one in
// whatever built it — would otherwise wrap, and a wrap on the bottom row scrolls the
// whole screen. That does not merely look wrong: everything shifts up by a row while the
// diff still believes it knows what is on the screen, so nothing is ever repainted to fix
// it. With auto-wrap off the overspill is simply cut off at the edge, which is wrong in a
// way that repaints away by itself.
const WRAP_OFF: &str = "\x1b[?7l";
const WRAP_ON: &str = "\x1b[?7h";

// Wheel and button reporting with SGR coordinates, and deliberately nothing else.
//
// 1002 (drag) and 1003 (any motion) are the two that make a terminal hand every mouse
// movement to the application, and a terminal that is reporting motion has stopped doing
// its own drag-selection. Asking for 1000 alone leaves selecting and copying where they
// belong — with the terminal — while the wheel still reaches the transcript here. Where a
// terminal does it differently, holding Shift while dragging is the usual way back to
// native selection.
const MOUSE_ON: &str = "\x1b[?1000h\x1b[?1006h";
const MOUSE_OFF: &str = "\x1b[?1006l\x1b[?1000l";

// Ask the terminal to say which modifiers were held, for the keys that otherwise cannot
// carry them.
//
// The legacy encoding has room for a modifier only on keys that are sent as CSI sequences:
// that is how `ESC[1;5D` manages to mean Ctrl+Left. A key sent as a bare control character
// has nowhere to put one — Enter is the single byte 0x0D, and Shift+Enter is the same
// single byte. Shift+Tab survives only because xterm gave it a sequence of its own,
// `ESC[Z`, not because anything is encoded in it.
//
// The Kitty keyboard protocol's first flag — "disambiguate escape codes" — replaces that
// with a form that has a modifier field for every key. `terminput` already parses it, so
// if the terminal honors this there is nothing else to do; a terminal that does not
// understand the sequence ignores it and the only loss is Shift+Enter, which Alt+Enter
// covers. Pushed rather than set, so leaving pops it and whatever ran before gets its own
// setting back.
const KEYS_ON: &str = "\x1b[>1u";
const KEYS_OFF: &str = "\x1b[<u";

// Turning the Windows console's VT input mode on and off.
//
// Everything measured in a real cmd window: leaving quick edit on means the console keeps
// every mouse event for itself; turning quick edit off gets the wheel but destroys
// selection outright, with no Shift override to bring it back; and asking for VT mouse
// mode without VT input gets nothing at all. Only VT input delivers the wheel, and with it
// Shift+drag still selects — so this is the one configuration that gives both.
//
// It is not free: VT input makes the console deliver every key as an escape sequence too,
// which is why `Keys` reads and parses the byte stream itself when this is on. The
// original mode is put back on the way out, because a console left in VT input mode
// confuses whatever runs next.
#[cfg(windows)]
mod console;

// Make sure the console is still in the input mode the keyboard reader needs.
//
// Called from `Keys` rather than from here, because the moment that matters is immediately
// before a read: the mode in force when a read *starts* is the mode that read obeys, and a
// read already waiting when the mode changed is unaffected by it.
#[cfg(windows)]
pub(crate) fn reassert_input_mode() {
    console::reassert_input_mode();
}

mod screen;
mod wrap;
pub(crate) use screen::*;
pub(crate) use wrap::*;


// The terminal's size in columns and rows, when there is a terminal to ask. Everything that
// lays something out asks here, rather than each asking crossterm with a fallback of its own;
// the minimum each one can work with is still its own, since an input box needs more room
// than a line of transcript does.
pub(crate) fn terminal_size() -> Option<(usize, usize)> {
    crossterm::terminal::size().ok().map(|(columns, rows)| (columns as usize, rows as usize))
}

// The size with the fallback everything uses when there is no terminal: 80 by 24.
pub(crate) fn terminal_width() -> usize {
    terminal_size().map_or(80, |(columns, _)| columns).max(1)
}

pub(crate) fn terminal_height() -> usize {
    terminal_size().map_or(24, |(_, rows)| rows).max(1)
}


// A panic while the lock is held would otherwise poison it and take the whole
// interface down with it, which is a bad trade for a line of output.
fn lock() -> MutexGuard<'static, Out> {
    static OUT: OnceLock<Mutex<Out>> = OnceLock::new();
    OUT.get_or_init(|| {
        Mutex::new(Out {
            raw: false,
            plain: false,
            at_line_start: true,
            region: Vec::new(),
            caret: (0, 0),
            region_drawn: false,
            drawn: Vec::new(),
            drawn_caret: (0, 0),
            drawn_width: 0,
            pending: String::new(),
            wrap: Wrap::new(80),
            capture: None,
            screen: None,
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
    // The full-screen renderer keeps its unfinished line in the transcript rather than in
    // `pending`, and never touches `at_line_start` at all.
    let unfinished = match &out.screen {
        Some(screen) => !screen.open.is_empty(),
        None => !out.at_line_start || !out.pending.is_empty(),
    };
    if unfinished {
        out.emit("\n");
    }
}

pub(crate) fn flush() {
    if lock().capture.is_none() {
        let _ = io::stdout().flush();
    }
}

// Drop every escape from what is written, colors included. For a `--prompt` run whose
// output is going to a log rather than a screen: there a color code is not a color, it is
// `[38;5;8m` in the middle of the line someone is grepping. Applied where the bytes leave
// rather than where they are made, because they are made in a dozen places that all
// rightly assume a terminal.
pub(crate) fn set_plain(plain: bool) {
    lock().plain = plain;
}

// NO_COLOR (no-color.org), honored here rather than left to crossterm.
//
// crossterm honors it too, but only by writing nothing between `ESC[` and `m` — and an empty
// SGR is not "no color", it is a full reset. Every color change in the output became a reset
// that also switched off bold, italics and reverse video in the middle of a line, and the
// rendered text itself changed with the environment: the render tests failed whenever the
// suite was run from a shell that set NO_COLOR. So crossterm is told to always write its
// colors (`own_colors`), everything is rendered the same everywhere, and the colors are taken
// out here, on the way to the terminal, with every other attribute left alone. Captured
// output is what was rendered, never filtered, so no test depends on the environment.
fn no_color() -> bool {
    static NO_COLOR: OnceLock<bool> = OnceLock::new();
    // Set and not empty, as the convention says.
    *NO_COLOR.get_or_init(|| std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()))
}

// Take crossterm's NO_COLOR handling away from it; see `no_color`. Called before anything is
// rendered, and by the tests that compare colors, since the switch is crossterm's global.
pub(crate) fn own_colors() {
    crossterm::style::force_color_output(true);
}

fn for_the_terminal(text: &str) -> std::borrow::Cow<'_, str> {
    if no_color() { without_color(text) } else { std::borrow::Cow::Borrowed(text) }
}

// Every SGR sequence with its color parameters removed: foreground, background and underline
// color, in the basic, bright, 256-color and truecolor forms. What remains — bold, dim,
// italic, underline, reverse and the resets — is kept, and a sequence left with nothing to
// say is dropped rather than written as `ESC[m`, which would be a reset of its own.
fn without_color(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains('\u{1b}') {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut kept = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            kept.push(ch);
            continue;
        }
        let escape = take_escape(&mut chars);
        let Some(params) = escape.strip_prefix("\u{1b}[").and_then(|rest| rest.strip_suffix('m'))
        else {
            kept.push_str(&escape);
            continue;
        };
        // A reset stays a reset: it is what turns bold back off.
        if params.split(';').all(|param| param.is_empty() || param == "0") {
            kept.push_str(&escape);
            continue;
        }
        let mut left = Vec::new();
        let mut params = params.split(';');
        while let Some(param) = params.next() {
            match param.parse::<u16>().unwrap_or(0) {
                // Extended colors carry their value in the parameters after them:
                // `5;n` for the 256-color palette, `2;r;g;b` for truecolor.
                38 | 48 | 58 => match params.next() {
                    Some("5") => {
                        params.next();
                    }
                    Some("2") => {
                        params.nth(2);
                    }
                    _ => {}
                },
                30..=37 | 39 | 40..=47 | 49 | 59 | 90..=97 | 100..=107 => {}
                _ => left.push(param),
            }
        }
        if !left.is_empty() {
            kept.push_str("\u{1b}[");
            kept.push_str(&left.join(";"));
            kept.push('m');
        }
    }
    std::borrow::Cow::Owned(kept)
}

fn without_escapes(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains('\u{1b}') {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut kept = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            take_escape(&mut chars);
        } else {
            kept.push(ch);
        }
    }
    std::borrow::Cow::Owned(kept)
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
    state.region = rows;
    state.caret = (caret_row, caret_col);
    if state.screen.is_some() {
        // Every row on the screen comes from one frame, so there is nothing to erase and
        // nothing to step around: the box is simply the bottom of the next one.
        state.screen_paint();
        return;
    }
    // Nothing is erased first. The block is redrawn on a timer as well as on every
    // keystroke, and most of those redraws ask for what is already on the screen, or for
    // one row of it changed: `paint` writes the difference, and writes nothing at all when
    // there is none.
    state.paint();
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
    if state.screen.is_some() {
        state.region.clear();
        state.caret = (0, 0);
        state.screen_paint();
        return;
    }
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

// Throw away everything that has been shown, along with the conversation it was showing.
//
// `/clear` starts over, and the transcript of what was just discarded is not something to
// scroll back through — the window frontend has always dropped it with the history, and
// this is the same thing for the two terminal renderers. Under the full-screen one the
// transcript is haico's own, so it is simply forgotten. Under the inline one the lines
// belong to the terminal, so the terminal is asked to purge them, its scrollback included:
// clearing only the viewport would leave the whole cleared conversation one wheel-click
// above it, which is not clearing it at all.
//
// Nothing happens when the output is a pipe rather than a terminal — there is no screen to
// clear and the escapes would land in whatever is reading it. `raw` is what tells the two
// apart, being switched on for exactly as long as there is a terminal to hold in raw mode.
pub(crate) fn clear_screen() {
    let mut state = lock();
    if state.screen.is_some() {
        if let Some(screen) = &mut state.screen {
            screen.forget();
        }
        state.screen_paint();
        return;
    }
    if !state.raw {
        return;
    }
    // The region comes off first, so what is drawn next is drawn from the top of an empty
    // screen rather than patched into rows the last frame still believes it knows the
    // position of. The half-finished line waiting above it goes with the rest.
    state.erase_region();
    state.pending.clear();
    state.wrap.reset_line();
    let escapes = format!(
        "{}{}{}",
        Clear(ClearType::Purge),
        Clear(ClearType::All),
        MoveTo(0, 0)
    );
    state.write_out(&escapes);
    state.at_line_start = true;
    state.paint();
}

// Take the alternate screen and start owning the viewport.
//
// Everything printed from here on belongs to haico rather than to the terminal, which is
// what pays for scrolling, for re-wrapping on a resize, and for being able to change a
// line after it was written. Refused while capturing: the tests assert on the inline
// renderer's output and a frame of absolute cursor moves is not that.
// Returns whether the console was put into VT input mode, which is a Windows-only thing
// and the one piece of this the keyboard reader has to know about: from then on it has to
// read and parse the byte stream itself rather than take crossterm's word for each key.
pub(crate) fn enter_screen(mouse: bool) -> bool {
    let mut state = lock();
    if state.screen.is_some() || state.capture.is_some() {
        return false;
    }

    let mut screen = Screen::new();
    screen.mouse = mouse;

    let mut vt_input = false;
    if mouse {
        #[cfg(windows)]
        {
            // Without this the escape sequence below is written into a void: the console
            // never sees it, which is exactly why the wheel used to do nothing.
            screen.console_mode = console::enable_vt_input();
            vt_input = screen.console_mode.is_some();
        }
        #[cfg(not(windows))]
        {
            // A VT terminal needs no such arrangement; the sequence is the whole of it.
            vt_input = false;
        }
    }

    let start = format!(
        "{}{}{}{}",
        ENTER_ALT,
        if mouse { MOUSE_ON } else { "" },
        // Worth asking for whether or not the mouse is wanted: it is the keyboard this
        // affects, and it costs nothing where it is not understood.
        KEYS_ON,
        Clear(ClearType::All)
    );
    state.write_out(&start);
    state.screen = Some(screen);
    let _ = io::stdout().flush();
    vt_input
}

// Give the terminal its screen back, and write the session into the scrollback behind it.
//
// Restoring the alternate screen on its own would take the whole conversation with it.
// The transcript goes out through `put` while raw mode is still on, so its line feeds
// still get their carriage returns; the caller drops raw mode afterwards.
pub(crate) fn leave_screen() {
    let mut state = lock();
    let Some(screen) = state.screen.take() else {
        return;
    };
    let transcript = screen.dump(wrap_width());
    state.write_out(&format!(
        "{}{}{}",
        KEYS_OFF,
        if screen.mouse { MOUSE_OFF } else { "" },
        LEAVE_ALT
    ));
    #[cfg(windows)]
    if let Some(original) = screen.console_mode {
        console::restore(original);
    }
    state.at_line_start = true;
    state.wrap = Wrap::new(wrap_width());
    state.put(&transcript);
    let _ = io::stdout().flush();
}

// Start writing detail: everything until `end_fold` is kept but shown only while the
// transcript is expanded, with `summary` standing in for it until then.
//
// A separate begin and end rather than one call taking both, because the thing this most
// needs to fold is a reply's chain of thought, and that arrives a fragment at a time over
// several seconds. Nothing in between has to know it is being folded.
//
// A no-op under the inline renderer, where a printed line belongs to the terminal and
// cannot be taken back.
pub(crate) fn begin_fold(summary: &str) {
    let mut state = lock();
    if state.screen.is_none() {
        // Nothing can be folded away later, so the summary would be a promise that is never
        // kept. The detail is simply printed, which is what happened before any of this.
        return;
    }
    // The summary is one row of its own; a line still being written would otherwise have
    // the summary land in the middle of it.
    ensure_line_start_locked(&mut state);
    if let Some(screen) = &mut state.screen {
        screen.writing = Kind::Summary;
    }
    state.screen_write(summary);
    state.screen_write("\n");
    if let Some(screen) = &mut state.screen {
        screen.writing = Kind::Detail;
    }
}

// Safe to call whether or not anything is open, which is what lets every path out of a
// streamed reply — finished, interrupted, failed — simply close the fold without first
// working out whether there is one. Closing nothing must cost nothing: `ensure_line_start`
// in the middle of a line that is not detail would break the line in two.
pub(crate) fn end_fold() {
    let mut state = lock();
    let open = matches!(
        state.screen.as_ref().map(|screen| screen.writing),
        Some(Kind::Detail)
    );
    if !open {
        return;
    }
    ensure_line_start_locked(&mut state);
    if let Some(screen) = &mut state.screen {
        screen.writing = Kind::Normal;
    }
    state.screen_paint();
}

// The common case in one call: a summary that is already the line haico would have printed
// anyway, and the full text behind it.
pub(crate) fn foldable(summary: &str, detail: &str) {
    if !lock().screen.is_some() {
        // Same as before folding existed: the summary is all there is room for.
        write_line(format_args!("{}", summary.trim_end_matches('\n')));
        return;
    }
    begin_fold(summary.trim_end_matches('\n'));
    // Indented so an opened block reads as belonging to the row above it rather than as
    // more transcript.
    for line in detail.trim_end_matches('\n').split('\n') {
        write_line(format_args!("      {}", line));
    }
    end_fold();
}

// A line that stands in for an outcome not known yet — "Reviewing tool call…" — and the
// handle `settle` replaces it through once it is.
pub(crate) struct Mark(u64);

// Write a line that will be replaced, and say where it went.
//
// `None`, and nothing written, under the inline renderer: a line handed to the terminal
// cannot be taken back, so the outcome is simply written when it arrives. That is also
// what a `--prompt` log wants — the verdict, not a line saying one was coming.
pub(crate) fn placeholder(text: &str) -> Option<Mark> {
    let mut state = lock();
    state.screen.as_ref()?;
    ensure_line_start_locked(&mut state);
    state.emit(&format!("{}\n", text));
    state.screen.as_ref().and_then(Screen::last_seq).map(Mark)
}

// Replace the placeholder with what it was waiting for — or, when there is none to
// replace (the inline renderer, a `/clear` in between, a placeholder that has scrolled out
// of the buffer), write the line after whatever is there now.
pub(crate) fn settle(mark: Option<Mark>, text: &str) {
    let mut state = lock();
    let replaced = match (&mut state.screen, mark) {
        (Some(screen), Some(Mark(seq))) => screen.replace(seq, text),
        _ => false,
    };
    if replaced {
        state.screen_paint();
    } else {
        ensure_line_start_locked(&mut state);
        state.emit(&format!("{}\n", text));
    }
}

// Show the detail instead of the summaries, or the other way about.
//
// The view goes back to following: what is on the screen has just changed height by a great
// deal, and an anchor into the middle of it would land somewhere the reader did not choose.
pub(crate) fn toggle_expand() {
    let mut state = lock();
    let changed = match &mut state.screen {
        Some(screen) => {
            screen.expanded = !screen.expanded;
            screen.scroll = None;
            true
        }
        None => false,
    };
    if changed {
        state.screen_paint();
    }
}

// `ensure_line_start` with the lock already held.
fn ensure_line_start_locked(state: &mut MutexGuard<'static, Out>) {
    let unfinished = match &state.screen {
        Some(screen) => !screen.open.is_empty(),
        None => !state.at_line_start || !state.pending.is_empty(),
    };
    if unfinished {
        state.emit("\n");
    }
}

// Move the view back through what has already gone past, or forward again. Negative is
// backward. A no-op under the inline renderer, where scrolling belongs to the terminal.
pub(crate) fn scroll(delta: isize) {
    let mut state = lock();
    if state.screen.is_none() {
        return;
    }
    let (width, height) = (terminal_width(), terminal_height());
    let body = height.saturating_sub(state.region.len().min(height)).max(1);
    let wrap = width.saturating_sub(1).max(1);
    if let Some(screen) = &mut state.screen {
        screen.scroll_by(delta, wrap, body);
    }
    state.screen_paint();
}

// Go back to the end and stay there. What the user has just sent is about to be answered,
// and the answer is at the bottom.
pub(crate) fn follow() {
    let mut state = lock();
    let moved = match &mut state.screen {
        Some(screen) => screen.scroll.take().is_some(),
        None => false,
    };
    if moved {
        state.screen_paint();
    }
}

// Whether `scroll` does anything, for a prompt deciding whether to offer it. Under the inline
// renderer the terminal's own scrollback is the way back, and a hint naming keys that do
// nothing there would be a lie.
pub(crate) fn scrollable() -> bool {
    lock().screen.is_some()
}

// How tall the transcript area is, for a caller working out what a page of it is.
pub(crate) fn viewport_rows() -> usize {
    let state = lock();
    let height = terminal_height();
    height.saturating_sub(state.region.len().min(height)).max(1)
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
    state.drawn.clear();
    state.drawn_caret = (0, 0);
    state.drawn_width = 0;
    state.pending.clear();
    state.wrap = Wrap::new(80);
    state.screen = None;
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
    fn plain_output_keeps_the_text_and_nothing_else() {
        let colored = "\u{1b}[38;5;8m✻ Whirred for 2s\u{1b}[0m\n\u{1b}[2K\u{1b}[1Gnext";
        assert_eq!(super::without_escapes(colored), "✻ Whirred for 2s\nnext");
        assert!(matches!(super::without_escapes("as is"), std::borrow::Cow::Borrowed(_)));
    }

    // NO_COLOR takes the colors and nothing else. In particular a color change must not turn
    // into `ESC[m`, which is a reset and would switch bold off in the middle of a heading.
    #[test]
    fn no_color_keeps_every_attribute_but_the_colors() {
        use super::without_color;
        // A color alone disappears whole; a reset stays.
        assert_eq!(without_color("\u{1b}[38;5;12mcode\u{1b}[0m"), "code\u{1b}[0m");
        assert_eq!(without_color("\u{1b}[31mred\u{1b}[m"), "red\u{1b}[m");
        // Bold survives the color it was combined with, in every color form.
        assert_eq!(without_color("\u{1b}[1;38;2;10;20;30mhead"), "\u{1b}[1mhead");
        assert_eq!(without_color("\u{1b}[48;5;22;1;97mx"), "\u{1b}[1mx");
        assert_eq!(without_color("\u{1b}[7;42;4mx"), "\u{1b}[7;4mx");
        // Escapes that are not SGR are none of this function's business.
        assert_eq!(without_color("\u{1b}[2K\u{1b}[1Gx"), "\u{1b}[2K\u{1b}[1Gx");
    }

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

    #[test]
    fn a_fragment_that_does_not_end_its_line_leaves_the_box_where_it_is() {
        let _guard = super::capture_guard();

        super::start_capture();
        super::set_region(vec!["> hello".to_string()], 0, 7);
        // The first fragment puts a line above the box, which the box has to move for.
        out!("some ");
        // The second only makes that line longer: it is written where the line already is,
        // and nothing pinned below it is touched.
        out!("words");
        let written = super::take_capture();

        assert_eq!(
            written.matches("> hello").count(),
            2,
            "the box was repainted for a fragment: {:?}",
            written
        );
        let box_row = written.rfind("> hello").expect("the box");
        let line = written.rfind("some words").expect("the line above it");
        assert!(box_row < line, "wrong order in {:?}", written);
    }

    #[test]
    fn a_color_carried_across_a_wrap_is_switched_on_again_on_the_next_row() {
        let _guard = super::capture_guard();
        const RED: &str = "\u{1b}[31m";

        super::start_capture();
        super::set_region(vec!["> ".to_string()], 0, 2);
        // Long enough to wrap however wide the terminal running the test happens to be,
        // and written in one piece so the color is switched on exactly once.
        let line = "word ".repeat(400);
        out!("{}{}\u{1b}[0m", RED, line.trim_end());
        let written = super::take_capture();

        let rows: Vec<&str> = written.split('\n').collect();
        let carrying = rows.iter().filter(|row| row.contains("word")).count();
        assert!(carrying > 1, "nothing wrapped: {} rows", rows.len());
        // Written once and painted once per row: a row is repainted on its own, so the
        // one the wrap carried a word onto has to say the color again rather than take
        // whatever the terminal was left in.
        assert_eq!(written.matches(RED).count(), carrying, "got {:?}", written);
    }

    #[test]
    fn only_the_colors_still_open_are_carried_to_the_next_row() {
        let carried = super::open_colors;

        // Switched on and left that way.
        assert_eq!(carried("", "\u{1b}[31mred"), "\u{1b}[31m");
        // Two of them, in the order they have to be written again.
        assert_eq!(carried("", "\u{1b}[31m\u{1b}[1mboth"), "\u{1b}[31m\u{1b}[1m");
        // Closed again, by either spelling of the reset.
        assert_eq!(carried("\u{1b}[31m", "red\u{1b}[0m plain"), "");
        assert_eq!(carried("\u{1b}[31m", "red\u{1b}[m plain"), "");
        // A cursor move says nothing about color, so it carries nothing either way.
        assert_eq!(carried("\u{1b}[31m", "\u{1b}[2Cred"), "\u{1b}[31m");
    }

    #[test]
    fn a_counter_ticking_in_the_bar_repaints_its_own_row_and_nothing_else() {
        let _guard = super::capture_guard();

        // A block shaped like the real one: a blank row, the activity bar, the input box.
        let block = |bar: &str| {
            vec![
                String::new(),
                format!("* {}", bar),
                "> hello".to_string(),
            ]
        };

        super::start_capture();
        super::set_region(block("Thinking (1)"), 2, 7);
        super::set_region(block("Thinking (2)"), 2, 7);
        let written = super::take_capture();

        // The bar was written once for each thing it said; everything around it was
        // written when the block went up and never touched again.
        assert_eq!(written.matches("* Thinking").count(), 2, "got {:?}", written);
        assert_eq!(
            written.matches("> hello").count(),
            1,
            "the box was repainted for a tick: {:?}",
            written
        );

        // And the tick is the row on its own: nothing cleared below it, and no line feed,
        // which is also what keeps it to a single write to the terminal.
        let tick = &written[written.rfind("* Thinking").expect("the second tick")..];
        assert!(!tick.contains('\n'), "a line feed in {:?}", tick);
        assert!(!tick.contains("[J"), "the block was cleared in {:?}", tick);
    }

    #[test]
    fn drawing_the_same_box_again_writes_nothing() {
        let _guard = super::capture_guard();

        super::start_capture();
        super::set_region(vec!["> hello".to_string()], 0, 7);
        super::set_region(vec!["> hello".to_string()], 0, 7);
        super::set_region(vec!["> hello".to_string()], 0, 7);
        let written = super::take_capture();
        assert_eq!(
            written.matches("> hello").count(),
            1,
            "repainted what was already there: {:?}",
            written
        );

        // Anything that actually differs is drawn, of course — here one more character
        // and the caret that moved with it.
        super::start_capture();
        super::set_region(vec!["> hello".to_string()], 0, 7);
        super::set_region(vec!["> hello!".to_string()], 0, 8);
        let written = super::take_capture();
        assert_eq!(written.matches("> hello").count(), 2, "got {:?}", written);
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

    // What broke a table: the screen is measured in cells, and a character is not one of
    // them. U+2713 and U+2714 sit next to each other in the Unicode tables and are drawn
    // differently — the second carries the emoji property and gets a color glyph two
    // cells wide — which is why this cannot be a rule about blocks of characters.
    #[test]
    fn width_is_counted_in_cells_rather_than_characters() {
        assert_eq!(super::visible_width("abc"), 3);
        // Two cells: CJK by the standard, the emoji by the font every terminal falls back
        // to for them.
        assert_eq!(super::visible_width("\u{4e2d}\u{6587}"), 4);
        assert_eq!(super::visible_width("\u{2611} Pass"), 7);
        assert_eq!(super::visible_width("\u{1f680}"), 2);
        // One cell, and it must stay that way: the frame of every table is drawn out of
        // these, and a check mark that is not an emoji is drawn as text.
        assert_eq!(super::visible_width("\u{2500}\u{2502}\u{250c}"), 3);
        assert_eq!(super::visible_width("\u{2713}"), 1);
        assert_eq!(super::visible_width("\u{2714}"), 2);
        // None at all: a variation selector and a combining mark are drawn into the
        // character in front of them, and an escape sequence is not drawn.
        assert_eq!(super::visible_width("\u{2611}\u{fe0f}"), 2);
        assert_eq!(super::visible_width("e\u{301}"), 1);
        assert_eq!(super::visible_width("\u{1b}[31mred\u{1b}[0m"), 3);
    }

    // The wrapper counts the same way, so a row of wide characters stops at the edge
    // rather than one character past it.
    #[test]
    fn a_run_of_wide_characters_wraps_on_the_cell_it_runs_out_on() {
        let text = "\u{4e2d}".repeat(12) + "\n";
        let wrapped = wrapped(&text, 10);
        no_row_overflows(&wrapped, 10);
        // Five characters to a ten-cell row, not ten.
        assert_eq!(wrapped.lines().next().map(super::visible_width), Some(10));
        assert_eq!(wrapped.lines().next().map(|row| row.chars().count()), Some(5));
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

    // /clear takes the screen with the conversation. Under the inline renderer the lines
    // are the terminal's, so the scrollback has to go too — a cleared conversation still
    // sitting one wheel-click above the input box was never cleared.
    #[test]
    fn clearing_takes_the_scrollback_with_the_screen_and_puts_the_box_back() {
        let _guard = super::capture_guard();

        super::start_capture();
        super::set_raw(true);
        super::set_region(vec!["> hello".to_string()], 0, 7);
        outln!("a result");
        super::clear_screen();
        let written = super::take_capture();

        let purge = written.rfind("[3J").expect("the scrollback should be purged");
        assert!(
            written[purge..].find("a result").is_none(),
            "the transcript survived the clear in {:?}",
            written
        );
        let redraw = written.rfind("> hello").expect("the box should be drawn again");
        assert!(purge < redraw, "wrong order in {:?}", written);
    }

    // Nothing is written at all when there is no screen to clear: the escapes would just
    // be junk in whatever is reading the output.
    #[test]
    fn clearing_a_piped_session_writes_no_escapes_into_the_pipe() {
        let _guard = super::capture_guard();

        // `start_capture` leaves raw mode off, which is what a piped session leaves it.
        super::start_capture();
        outln!("a result");
        super::clear_screen();
        assert_eq!(super::take_capture(), "a result\n");
    }
    // ---- the full-screen renderer ----

    fn screen_with(lines: &[&str]) -> super::Screen {
        let mut screen = super::Screen::new();
        for line in lines {
            screen.write(line);
            screen.write("\n");
        }
        screen
    }

    // Paint into a buffer instead of the terminal and give back what would have been
    // written.
    fn frame(screen: &mut super::Screen, region: &[&str], size: (usize, usize)) -> String {
        let rows: Vec<String> = region.iter().map(|row| row.to_string()).collect();
        let mut capture = Some(String::new());
        screen.paint(&rows, (0, 0), &mut capture, size);
        capture.unwrap_or_default()
    }

    // The point of keeping lines logical rather than as the rows they were last shown as:
    // the same line gives different rows on a narrower screen. The inline renderer cannot
    // do this — once a line is in the terminal's scrollback it keeps the wrap it was
    // written with forever.
    #[test]
    fn a_resize_rewraps_the_transcript_instead_of_leaving_it_as_it_was() {
        let screen = screen_with(&["the quick brown fox jumps over the lazy dog"]);

        let wide = screen.rows(40, 20);
        let narrow = screen.rows(20, 20);

        assert!(
            narrow.len() > wide.len(),
            "the narrower screen should have needed more rows: {:?} vs {:?}",
            wide,
            narrow
        );
        for row in &narrow {
            assert!(
                super::visible_width(row) <= 20,
                "row ran past the edge: {:?}",
                row
            );
        }
        // Nothing was consumed working that out: the line is still there to be wrapped
        // again, at either width.
        assert_eq!(screen.rows(40, 20), wide);
    }

    // A session shorter than the screen leaves the gap at the top, so the input box stays
    // on the bottom row rather than floating in the middle of an empty screen.
    #[test]
    fn a_short_transcript_sits_against_the_bottom_with_the_region_below_it() {
        let mut screen = screen_with(&["first", "second"]);
        let painted = frame(&mut screen, &["> ", "status"], (40, 8));

        let rows: Vec<&str> = painted.match_indices("first").map(|_| "first").collect();
        assert_eq!(rows.len(), 1, "the transcript should be painted once");

        // Six rows for the transcript, two for the region, and the two lines of transcript
        // land on the last two of the six.
        assert!(
            painted.contains(&format!("{}first", super::MoveTo(0, 4))),
            "transcript should be pushed to the bottom of its area: {:?}",
            painted
        );
        assert!(
            painted.contains(&format!("{}> ", super::MoveTo(0, 6))),
            "the box should be the second row from the bottom: {:?}",
            painted
        );
    }

    // The reason a token counter ticking in the bar does not make the whole screen blink.
    #[test]
    fn only_the_rows_that_changed_are_painted_again() {
        let mut screen = screen_with(&["hello"]);
        frame(&mut screen, &["> ", "Thinking (1)"], (40, 6));

        let second = frame(&mut screen, &["> ", "Thinking (2)"], (40, 6));

        assert!(second.contains("Thinking (2)"));
        assert!(
            !second.contains("hello"),
            "the transcript was repainted for a change in the bar: {:?}",
            second
        );
        assert_eq!(
            second.matches(&super::MoveTo(0, 5).to_string()).count(),
            1,
            "only the bar's own row should have been moved to: {:?}",
            second
        );
    }

    #[test]
    fn a_frame_that_says_the_same_thing_writes_nothing() {
        let mut screen = screen_with(&["hello"]);
        frame(&mut screen, &["> "], (40, 6));

        assert_eq!(frame(&mut screen, &["> "], (40, 6)), "");
    }

    // Every row is painted on its own, so a row can never rely on the terminal still being
    // in the color the row above left it in.
    #[test]
    fn a_color_left_open_on_one_line_is_switched_on_again_on_the_next() {
        let screen = screen_with(&["\u{1b}[31mred", "still red"]);

        let rows = screen.rows(40, 20);

        assert!(rows[1].starts_with("\u{1b}[31m"), "{:?}", rows);
    }

    // The buffer is bounded, unlike the terminal's own scrollback.
    #[test]
    fn the_oldest_lines_fall_off_once_the_buffer_is_full() {
        let mut screen = super::Screen::new();
        for index in 0..super::SCROLLBACK_LINES + 50 {
            screen.write(&format!("line {}\n", index));
        }

        assert_eq!(screen.lines.len(), super::SCROLLBACK_LINES);
        assert!(screen.lines.front().unwrap().text.contains("line 50"));
        assert!(screen.lines.back().unwrap().text.contains(&format!(
            "line {}",
            super::SCROLLBACK_LINES + 49
        )));
    }

    // A row is painted again and again, so an escape that clears the screen is not
    // something that happened once — it is something that happens on every repaint. This
    // is what used to wipe the input box and the status bar and leave them wiped, because
    // the diff had already decided those two rows were correct.
    #[test]
    fn cursor_moves_and_screen_clears_never_reach_the_transcript() {
        let mut screen = super::Screen::new();
        screen.write(&format!(
            "{}{}kept\n",
            super::MoveTo(0, 0),
            super::Clear(super::ClearType::All)
        ));

        let rows = screen.rows(40, 20);

        assert_eq!(rows, vec!["kept".to_string()]);
    }

    #[test]
    fn the_colors_a_line_is_written_in_are_kept() {
        let mut screen = super::Screen::new();
        screen.write("\u{1b}[31mred\u{1b}[0m\n");

        assert_eq!(screen.rows(40, 20), vec!["\u{1b}[31mred\u{1b}[0m".to_string()]);
    }

    // A row too wide for the screen must not be allowed to wrap: a wrap on the bottom row
    // scrolls everything up and the diff never finds out.
    #[test]
    fn a_frame_paints_with_auto_wrap_switched_off() {
        let mut screen = screen_with(&["hello"]);
        let painted = frame(&mut screen, &["> "], (40, 6));

        assert!(painted.starts_with(super::BEGIN_SYNC));
        assert!(painted.contains(super::WRAP_OFF));
        assert!(painted.ends_with(super::END_SYNC));
        assert!(painted.contains(super::WRAP_ON));
    }

    // ---- folding ----

    // Writes a summary and its detail the way `foldable` does, without going near the
    // global writer.
    fn with_fold(before: &[&str], summary: &str, detail: &[&str], after: &[&str]) -> super::Screen {
        let mut screen = super::Screen::new();
        for line in before {
            screen.write(line);
            screen.write("\n");
        }
        screen.writing = super::Kind::Summary;
        screen.write(summary);
        screen.write("\n");
        screen.writing = super::Kind::Detail;
        for line in detail {
            screen.write(line);
            screen.write("\n");
        }
        screen.writing = super::Kind::Normal;
        for line in after {
            screen.write(line);
            screen.write("\n");
        }
        screen
    }

    // The whole point: both are written when it happens, and which one is on screen is
    // decided later — long after the inline renderer would have handed the line to the
    // terminal and lost the choice.
    #[test]
    fn the_summary_shows_until_it_is_expanded_and_then_the_detail_does() {
        let mut screen = with_fold(&["before"], "* ran a tool", &["line one", "line two"], &["after"]);

        let folded = screen.rows(40, 20);
        assert_eq!(folded, vec!["before", "* ran a tool", "after"], "{:?}", folded);

        screen.expanded = true;
        let opened = screen.rows(40, 20);
        assert_eq!(
            opened,
            vec!["before", "line one", "line two", "after"],
            "{:?}",
            opened
        );
    }

    // Folded lines are skipped, not stopped at: everything written after a fold has to stay
    // reachable when scrolled to.
    #[test]
    fn scrolling_reads_straight_through_a_folded_run() {
        let screen = with_fold(&["before"], "* summary", &["hidden one", "hidden two"], &["after"]);

        let anchor = super::Anchor { seq: 0, row: 0 };
        let rows = screen.rows_from(anchor, 40, 20);

        assert_eq!(rows, vec!["before", "* summary", "after"], "{:?}", rows);
    }

    // A fold changes the height of the transcript by a great deal, so the anchor it was
    // holding means somewhere else entirely afterwards.
    #[test]
    fn expanding_puts_the_view_back_to_following() {
        let mut screen = with_fold(&["a", "b", "c"], "* summary", &["x", "y"], &["d"]);
        screen.scroll_by(-2, 40, 3);
        assert!(screen.scroll.is_some(), "the view should be held back");

        let mut capture = Some(String::new());
        screen.expanded = true;
        screen.scroll = None;
        screen.paint(&["> ".to_string()], (0, 0), &mut capture, (40, 8));

        assert_eq!(screen.scroll, None);
    }

    // What was on the screen is what goes into the scrollback on the way out.
    #[test]
    fn the_exit_dump_follows_whichever_way_the_fold_is_set() {
        let mut screen = with_fold(&["before"], "* summary", &["secret detail"], &[]);

        let folded = screen.dump(40);
        assert!(folded.contains("* summary"), "{:?}", folded);
        assert!(!folded.contains("secret detail"), "{:?}", folded);

        screen.expanded = true;
        let opened = screen.dump(40);
        assert!(opened.contains("secret detail"), "{:?}", opened);
        assert!(!opened.contains("* summary"), "{:?}", opened);
    }

    // ---- scrolling ----

    fn numbered(count: usize) -> super::Screen {
        let mut screen = super::Screen::new();
        for index in 0..count {
            screen.write(&format!("line {}\n", index));
        }
        screen
    }

    // What arrives while the view is held back must not drag it along: the whole point of
    // scrolling back during a turn is to read something while the turn carries on.
    #[test]
    fn output_arriving_does_not_move_a_view_that_is_held_back() {
        let mut screen = numbered(50);
        screen.scroll_by(-10, 40, 10);
        let held = screen.scroll.expect("the view should be held back");

        for index in 50..70 {
            screen.write(&format!("line {}\n", index));
        }

        assert_eq!(screen.scroll, Some(held), "new output moved the view");
        assert!(screen.rows_from(held, 40, 3)[0].contains("line 30"));
    }

    // The reason the anchor is a line and a row within it rather than a count of rows from
    // the end. Rows are a different size at a different width, so an offset counted in them
    // means somewhere else entirely once the screen changes; a line still means that line.
    #[test]
    fn a_resize_leaves_the_view_looking_at_the_same_line() {
        let mut screen = super::Screen::new();
        for name in ["aaa", "bbb", "ccc", "ddd", "eee"] {
            screen.write(&format!("{} {}\n", name, name.repeat(30)));
        }

        screen.scroll_by(-6, 40, 6);
        let held = screen.scroll.expect("the view should be held back");
        let before = screen.rows_from(held, 40, 1)[0].clone();

        // The same anchor, a screen half the width.
        let after = screen.rows_from(held, 20, 1)[0].clone();

        let name = |row: &str| row.chars().take(3).collect::<String>();
        assert_eq!(
            name(&before),
            name(&after),
            "the resize moved the view: {:?} then {:?}",
            before,
            after
        );
    }

    #[test]
    fn scrolling_back_down_to_the_end_starts_following_again() {
        let mut screen = numbered(50);

        screen.scroll_by(-5, 40, 10);
        assert!(screen.scroll.is_some());

        screen.scroll_by(5, 40, 10);
        assert_eq!(screen.scroll, None, "reaching the end should follow again");
    }

    #[test]
    fn the_view_stops_at_the_start_rather_than_running_off_it() {
        let mut screen = numbered(20);

        screen.scroll_by(-1000, 40, 5);

        let rows = screen.rows_from(screen.scroll.unwrap(), 40, 1);
        assert!(rows[0].contains("line 0"), "{:?}", rows);
    }

    // Output carries on arriving while the view stays put, which without something saying
    // so looks exactly like haico having stopped.
    #[test]
    fn a_held_back_view_says_so_and_pays_a_row_for_it() {
        let mut screen = numbered(50);
        screen.scroll_by(-10, 40, 8);

        let painted = frame(&mut screen, &["> ", "status"], (40, 10));

        assert!(painted.contains("held back"), "{:?}", painted);
        // Eight rows of transcript area: seven of content and the row that says so.
        assert!(
            painted.contains(&format!("{}\u{1b}[38;5;245m── held back", super::MoveTo(0, 7))),
            "the notice should be the last row above the region: {:?}",
            painted
        );
    }

    // Without this the whole session goes with the alternate screen when it is restored.
    #[test]
    fn the_transcript_is_written_back_out_when_the_screen_is_given_up() {
        let mut screen = screen_with(&["first", "second"]);
        screen.write("a line that never ended");

        let dumped = screen.dump(40);

        assert!(dumped.contains("first"), "{:?}", dumped);
        assert!(dumped.contains("second"), "{:?}", dumped);
        assert!(
            dumped.contains("a line that never ended"),
            "the unfinished line was dropped: {:?}",
            dumped
        );
    }

    // /clear throws the conversation away, and the transcript is a transcript of that
    // conversation: it goes too, off the screen and out of the dump both, so giving up the
    // alternate screen later does not hand the cleared session back.
    #[test]
    fn a_cleared_transcript_leaves_the_screen_and_the_dump_empty() {
        let mut screen = screen_with(&["first", "second"]);
        screen.write("a line that never ended");
        frame(&mut screen, &["> "], (40, 6));

        screen.forget();
        let painted = frame(&mut screen, &["> "], (40, 6));

        for gone in ["first", "second", "a line that never ended"] {
            assert!(!painted.contains(gone), "{:?} survived in {:?}", gone, painted);
        }
        assert!(screen.rows(40, 6).is_empty(), "{:?}", screen.rows(40, 6));
        assert_eq!(screen.dump(40), "");

        // And the screen is a working one again rather than a dead end.
        screen.write("after\n");
        assert_eq!(screen.dump(40), "after\n");
    }

    // ---- replacing a line ----

    // By sequence number, not "the last line": what landed below the placeholder while it
    // waited stays where it is.
    #[test]
    fn a_placeholder_is_replaced_where_it_is_with_later_lines_left_alone() {
        let mut screen = screen_with(&["→ run_command  cargo build"]);
        screen.write("⁇ Reviewing tool call…\n");
        let seq = screen.last_seq().expect("a line was written");
        screen.write("  Searching: something\n");

        assert!(screen.replace(seq, "✓ Allowed: builds the project"));

        assert_eq!(
            screen.rows(40, 20),
            vec!["→ run_command  cargo build", "✓ Allowed: builds the project", "  Searching: something"]
        );
    }

    // A placeholder that is no longer there is reported rather than standing in for some
    // other line: `/clear` in between, or the buffer having dropped it off the front.
    #[test]
    fn a_line_that_is_gone_is_not_replaced() {
        let mut screen = screen_with(&["waiting"]);
        let seq = screen.last_seq().unwrap();
        screen.forget();
        screen.write("after the clear\n");
        assert!(!screen.replace(seq, "verdict"));
        assert_eq!(screen.rows(40, 20), vec!["after the clear"]);

        let mut screen = super::Screen::new();
        screen.write("first\n");
        let first = screen.last_seq().unwrap();
        for index in 0..super::SCROLLBACK_LINES {
            screen.write(&format!("line {}\n", index));
        }
        assert!(!screen.replace(first, "verdict"));
    }

    // The colors the line inherited from the one above survive the rewrite, and the colors
    // the next line inherits come from the new text, not the old.
    #[test]
    fn a_replaced_line_keeps_what_it_inherited_and_passes_on_what_it_leaves_open() {
        let mut screen = super::Screen::new();
        screen.write("\u{1b}[31mred\n");
        screen.write("waiting\n");
        let seq = screen.last_seq().unwrap();

        assert!(screen.replace(seq, "done\u{1b}[0m"));
        screen.write("plain\n");

        let rows = screen.rows(40, 20);
        assert_eq!(rows[1], "\u{1b}[31mdone\u{1b}[0m", "{:?}", rows);
        assert_eq!(rows[2], "plain", "{:?}", rows);
    }

    // The inline renderer cannot take a row back, so there is no placeholder to replace and
    // the outcome is simply written when it arrives.
    #[test]
    fn inline_a_placeholder_writes_nothing_and_the_outcome_is_written_on_its_own() {
        let _guard = super::capture_guard();
        super::start_capture();

        let mark = super::placeholder("Reviewing tool call…");
        assert!(mark.is_none());
        super::settle(mark, "Allowed: fine");

        assert_eq!(super::take_capture(), "Allowed: fine\n");
    }
}
