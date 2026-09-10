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
        } else {
            let _ = io::stdout().write_all(text.as_bytes());
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
mod console {
    extern "system" {
        fn GetStdHandle(which: u32) -> isize;
        fn GetConsoleMode(handle: isize, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: isize, mode: u32) -> i32;
    }

    const STD_INPUT_HANDLE: u32 = -10i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;
    // The three raw mode exists to have off. Named together because they are only ever
    // cleared together: echo is documented as having no effect without line input, and
    // processed input is what turns Ctrl+C into a signal instead of a byte.
    const COOKED: u32 = 0x0001 | 0x0002 | 0x0004;

    fn mode() -> Option<u32> {
        unsafe {
            let mut mode = 0;
            if GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mut mode) == 0 {
                return None;
            }
            Some(mode)
        }
    }

    // Gives back the mode as it was, for putting back later.
    pub(super) fn enable_vt_input() -> Option<u32> {
        let original = mode()?;
        unsafe {
            if SetConsoleMode(
                GetStdHandle(STD_INPUT_HANDLE),
                original | ENABLE_VIRTUAL_TERMINAL_INPUT,
            ) == 0
            {
                return None;
            }
        }
        Some(original)
    }

    pub(super) fn restore(original: u32) {
        unsafe {
            SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), original);
        }
    }

    // Put the mode back if something else has changed it, and say nothing when it has not.
    //
    // The console input mode belongs to the console, not to the process that set it: anything
    // sharing this window can rewrite it, and cmd.exe does exactly that — which is why
    // `run_command` no longer hands a child the console's own stdin. This is the other half of
    // it. Prevention only covers the children haico spawns, and the mode being wrong is not a
    // degraded session but an unusable one: cooked input means the console echoes every escape
    // sequence into the input box and hands nothing over until Enter, with the transcript on
    // the alternate screen where the scrollback cannot be reached either.
    //
    // Only the flags that matter are forced. Whatever else the mode carries — quick edit
    // above all, which is what keeps drag-selection working — is left exactly as found,
    // because this runs on every read and is not the place to be making policy.
    pub(super) fn reassert_input_mode() {
        let Some(mode) = mode() else {
            return;
        };
        let wanted = (mode & !COOKED) | ENABLE_VIRTUAL_TERMINAL_INPUT;
        if wanted == mode {
            return;
        }
        unsafe {
            SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), wanted);
        }
    }
}

// Make sure the console is still in the input mode the keyboard reader needs.
//
// Called from `Keys` rather than from here, because the moment that matters is immediately
// before a read: the mode in force when a read *starts* is the mode that read obeys, and a
// read already waiting when the mode changed is unaffected by it.
#[cfg(windows)]
pub(crate) fn reassert_input_mode() {
    console::reassert_input_mode();
}

// The row that says the view is being held back, and by how much.
//
// It costs a row of transcript, and it is worth it: output carries on arriving while the
// view stays where it was put, which without something saying so looks exactly like haico
// having stopped.
fn held_back_row(below: usize, width: usize) -> String {
    let text = format!(
        "── held back · {} line{} below · PageDown or the wheel to follow ──",
        below,
        if below == 1 { "" } else { "s" }
    );
    let text: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("\x1b[90m{}\x1b[0m", text)
}

// How many logical lines of transcript are kept. The buffer has to be bounded — the
// inline renderer needed no limit because a line became the terminal's problem the moment
// it was written, and these never do. A session long enough to reach this has scrolled
// well past anything anyone is going back for.
const SCROLLBACK_LINES: usize = 10_000;

// Wrap one logical line into the screen rows it occupies, each row carrying the colors
// that were open where it starts.
//
// The rows have to stand alone for the same reason the inline renderer's do: `paint`
// rewrites individual rows and leaves the rest untouched, so a row repainted on its own
// cannot rely on the terminal still being in the color the row above left it in.
pub(crate) fn wrap_rows(line: &str, width: usize) -> Vec<String> {
    let wrapped = Wrap::new(width).feed(line);
    let mut rows = Vec::new();
    let mut open = String::new();
    for row in wrapped.split('\n') {
        rows.push(format!("{}{}", open, row));
        open = open_colors(&open, row);
    }
    rows
}

// Everything in a stored line except the parts that move the cursor or clear the screen.
//
// The transcript holds text and the colors it is written in, and nothing else. A line is
// painted wherever the frame puts it, over and over, long after whatever wrote it has
// stopped caring — so an escape that moves the cursor or clears the screen is not a thing
// that happened once, it is a thing that happens on every repaint from now on. The model
// picker used to clear the screen by printing the escape for it through `out!`, which the
// inline renderer passes to the terminal and forgets: here it landed in a transcript row
// and wiped the screen from the middle of a frame every time that row was drawn again.
// Everything painted before it in that frame died with it, and the rows the diff had
// decided were already correct — the input box, the status bar — were never drawn again.
//
// Only SGR sequences survive, which is exactly what `open_colors` already tracks.
fn text_and_color(text: &str) -> std::borrow::Cow<'_, str> {
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
        if escape.starts_with("\u{1b}[") && escape.ends_with('m') {
            kept.push_str(&escape);
        }
    }
    std::borrow::Cow::Owned(kept)
}

// What a stored line is, and therefore when it is shown.
//
// The summary and the detail of the same thing are both written into the transcript when
// it happens; which of the two is on screen is decided at paint time. That is the whole of
// the folding, and it is why it needs no block identity and no per-block state: one flag
// says which kind is currently being shown, and a line knows which kind it is.
//
// It is also the only reason any of this is possible. The inline renderer hands a line to
// the terminal and forgets it, so what was printed is what there is forever; here the text
// is still ours and can be shown or not long after it was written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    // Always shown.
    Normal,
    // The one row standing in for something folded away.
    Summary,
    // What that row stands in for.
    Detail,
}

struct Line {
    text: String,
    kind: Kind,
}

// Where the top of the viewport sits when the view is being held back.
//
// The line is named by a sequence number rather than by its index, because the buffer
// drops its oldest lines once it is full and an index would quietly come to mean a
// different line. The row says which of that line's wrapped rows is the first one shown:
// a single line can be taller than the screen, so the top of the viewport is not always
// the top of a line.
//
// This is the whole reason a resize does not throw the view somewhere arbitrary. An offset
// counted in rows from the end means nothing once the rows are a different width; a line
// and a row within it still means the same place.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Anchor {
    seq: u64,
    row: usize,
}

// The transcript, and the screen it is painted onto.
//
// Lines are kept *logical* — as they were written, not as they were last shown — and
// wrapped only when a frame is built. That is the whole trick: a resize changes nothing
// here, it just produces different rows next time, and a line rewritten later takes its
// new width from wherever the screen happens to be by then.
struct Screen {
    // Finished lines, oldest first. Each is self-contained: it begins with whatever colors
    // were still open when the line before it ended, so it can be wrapped and painted
    // without reading anything that came earlier.
    lines: VecDeque<Line>,
    // The line being written that has not found its newline yet. It is part of the frame
    // like any other, which is why a streamed reply appears as it arrives rather than a
    // line at a time — but it stays out of `lines` until it ends, because until then its
    // last word is still free to move to another row.
    open: String,
    // The colors open at the end of the last finished line, prefixed onto the next one.
    carry: String,
    // What kind the lines being written now are. Set by `begin_fold` and put back by
    // `end_fold`, so a whole streamed run of output can be marked as detail without every
    // writer along the way having to know about it.
    writing: Kind,
    // Whether the detail is being shown instead of the summaries. One flag for the whole
    // transcript, which is what makes this a toggle rather than a thing to navigate.
    expanded: bool,
    // How many lines have fallen off the front, which is what turns a sequence number into
    // an index and back.
    dropped: u64,
    // Where the top of the viewport is, or `None` to stay at the end and follow whatever
    // arrives. Scrolling back down to the end puts it back to `None` rather than leaving
    // an anchor pinned there, so new output starts moving the view again by itself.
    scroll: Option<Anchor>,
    // The rows exactly as they are on the screen, and where the cursor was left. The next
    // frame is compared against these and only the rows that differ are painted again.
    drawn: Vec<String>,
    drawn_caret: (usize, usize),
    // The size the frame was laid out for. A different one means every row has moved and
    // none of them can be patched.
    drawn_size: (usize, usize),
    // Whether mouse reporting was asked for, so leaving can turn off exactly what
    // entering turned on.
    mouse: bool,
    // The Windows console input mode as it was before VT input was switched on.
    #[cfg(windows)]
    console_mode: Option<u32>,
}

impl Screen {
    fn new() -> Self {
        Screen {
            lines: VecDeque::new(),
            open: String::new(),
            carry: String::new(),
            writing: Kind::Normal,
            expanded: false,
            dropped: 0,
            scroll: None,
            drawn: Vec::new(),
            drawn_caret: (0, 0),
            drawn_size: (0, 0),
            mouse: false,
            #[cfg(windows)]
            console_mode: None,
        }
    }

    fn write(&mut self, text: &str) {
        let text = text_and_color(text);
        for (index, part) in text.split('\n').enumerate() {
            if index > 0 {
                self.end_line();
            }
            self.open.push_str(part);
        }
    }

    fn end_line(&mut self) {
        let line = std::mem::take(&mut self.open);
        // Worked out before the line is stored with its prefix, so the colors of the
        // prefix count toward what the next line inherits.
        let carried = std::mem::take(&mut self.carry);
        self.carry = open_colors(&carried, &line);
        self.lines.push_back(Line {
            text: carried + &line,
            kind: self.writing,
        });
        while self.lines.len() > SCROLLBACK_LINES {
            self.lines.pop_front();
            self.dropped += 1;
        }
    }

    // Throw the transcript away.
    //
    // The lines are gone rather than scrolled past, because this is what `/clear` does to
    // the conversation and a transcript of a conversation that no longer exists is not
    // something to scroll back through. `dump` writes what is left on the way out, so a
    // cleared session does not reappear in the terminal's own scrollback either.
    //
    // The sequence numbers carry on from where they were rather than starting again.
    // `dropped` means "how many lines this buffer no longer has", and these are exactly
    // that; handing their numbers to lines written after the clear is how an anchor left
    // over from before it would come to point at one of them.
    fn forget(&mut self) {
        self.dropped += self.lines.len() as u64;
        self.lines.clear();
        self.open.clear();
        self.carry.clear();
        // A fold that was still open belonged to output that has just gone. Left on
        // Detail, the next thing written would be hidden behind a summary that is no
        // longer on the screen to expand.
        self.writing = Kind::Normal;
        self.scroll = None;
    }

    fn shows(&self, kind: Kind) -> bool {
        match kind {
            Kind::Normal => true,
            Kind::Summary => !self.expanded,
            Kind::Detail => self.expanded,
        }
    }

    // Every logical line in order, the unfinished one included: it is as much a part of the
    // transcript as the rest, it just has not stopped growing yet. `None` where the index is
    // past the end *or* the line is one the fold is currently hiding — the walkers below
    // step over both the same way, which is what keeps folding out of their arithmetic.
    fn line_at(&self, index: usize) -> Option<std::borrow::Cow<'_, str>> {
        if let Some(line) = self.lines.get(index) {
            if !self.shows(line.kind) {
                return None;
            }
            return Some(std::borrow::Cow::Borrowed(&line.text));
        }
        if index == self.lines.len() && !self.open.is_empty() && self.shows(self.writing) {
            return Some(std::borrow::Cow::Owned(self.carry.clone() + &self.open));
        }
        None
    }

    // Whether there is anything at this index at all, shown or not. `line_at` cannot answer
    // that: it says `None` for a hidden line too, on purpose.
    fn exists(&self, index: usize) -> bool {
        index < self.line_count()
    }

    fn line_count(&self) -> usize {
        self.lines.len() + usize::from(!self.open.is_empty())
    }

    fn rows_in(&self, index: usize, width: usize) -> usize {
        self.line_at(index)
            .map(|line| Wrap::new(width).feed(&line).split('\n').count())
            .unwrap_or(0)
    }

    // The anchor that shows the last `height` rows — where the view sits when it is
    // following. Walked backward from the end, which is the only direction that can find
    // it without wrapping the whole transcript.
    fn bottom_anchor(&self, width: usize, height: usize) -> Anchor {
        let mut seen = 0;
        let mut index = self.line_count();
        while index > 0 {
            let rows = self.rows_in(index - 1, width);
            // A folded-away line takes no rows and cannot be the top of the viewport.
            if rows > 0 && seen + rows >= height {
                return Anchor {
                    seq: self.dropped + (index - 1) as u64,
                    row: rows - (height - seen),
                };
            }
            seen += rows;
            index -= 1;
        }
        Anchor { seq: self.dropped, row: 0 }
    }

    // Rows from an anchor forward, for a view that is being held back.
    fn rows_from(&self, anchor: Anchor, width: usize, height: usize) -> Vec<String> {
        let mut index = anchor.seq.saturating_sub(self.dropped) as usize;
        let mut skip = anchor.row;
        let mut rows = Vec::with_capacity(height);
        while rows.len() < height {
            if !self.exists(index) {
                break;
            }
            let Some(line) = self.line_at(index) else {
                // Hidden by the fold: step over it rather than stopping, or everything
                // below a folded run would be unreachable.
                index += 1;
                skip = 0;
                continue;
            };
            for row in wrap_rows(&line, width).into_iter().skip(skip) {
                if rows.len() == height {
                    break;
                }
                rows.push(row);
            }
            skip = 0;
            index += 1;
        }
        rows
    }

    // A row at a time in both directions. A wheel notch is three of them and a page is a
    // screenful, so the walk is short even though each step may have to wrap a line to
    // find out how tall it is.
    fn step_back(&self, mut anchor: Anchor, count: usize, width: usize) -> Anchor {
        for _ in 0..count {
            if anchor.row > 0 {
                anchor.row -= 1;
                continue;
            }
            // Past the start of this line, so back through the lines before it until one of
            // them has a row to land on. A folded-away line has none.
            loop {
                if anchor.seq <= self.dropped {
                    return anchor;
                }
                anchor.seq -= 1;
                let index = (anchor.seq - self.dropped) as usize;
                let rows = self.rows_in(index, width);
                if rows > 0 {
                    anchor.row = rows - 1;
                    break;
                }
            }
        }
        anchor
    }

    fn step_forward(&self, mut anchor: Anchor, count: usize, width: usize, bottom: Anchor) -> Anchor {
        for _ in 0..count {
            if anchor >= bottom {
                return bottom;
            }
            let index = (anchor.seq - self.dropped) as usize;
            if anchor.row + 1 < self.rows_in(index, width) {
                anchor.row += 1;
                continue;
            }
            // On to the next line with anything to show, stopping at the end either way.
            loop {
                anchor.seq += 1;
                anchor.row = 0;
                if anchor >= bottom {
                    return bottom;
                }
                let index = (anchor.seq - self.dropped) as usize;
                if !self.exists(index) || self.rows_in(index, width) > 0 {
                    break;
                }
            }
        }
        anchor.min(bottom)
    }

    fn scroll_by(&mut self, delta: isize, width: usize, height: usize) {
        let bottom = self.bottom_anchor(width, height);
        let from = self.scroll.unwrap_or(bottom);
        let anchor = if delta < 0 {
            self.step_back(from, delta.unsigned_abs(), width)
        } else {
            self.step_forward(from, delta as usize, width, bottom)
        };
        // Back at the end means following again, rather than an anchor pinned to the
        // bottom that new output would have to keep pushing along.
        self.scroll = if anchor >= bottom { None } else { Some(anchor) };
    }

    // The last `height` rows of the transcript.
    //
    // Walked backward from the newest line and stopped as soon as the viewport is full.
    // Wrapping the whole transcript to find the last twenty rows of it would cost the
    // length of the session on every frame, and frames are painted several times a second
    // while a reply streams.
    fn rows(&self, width: usize, height: usize) -> Vec<String> {
        if height == 0 {
            return Vec::new();
        }

        let mut blocks: Vec<Vec<String>> = Vec::new();
        let mut total = 0;
        if !self.open.is_empty() && self.shows(self.writing) {
            let rows = wrap_rows(&(self.carry.clone() + &self.open), width);
            total += rows.len();
            blocks.push(rows);
        }
        for line in self.lines.iter().rev() {
            if total >= height {
                break;
            }
            if !self.shows(line.kind) {
                continue;
            }
            let rows = wrap_rows(&line.text, width);
            total += rows.len();
            blocks.push(rows);
        }
        blocks.reverse();

        let rows: Vec<String> = blocks.into_iter().flatten().collect();
        let start = rows.len().saturating_sub(height);
        rows[start..].to_vec()
    }

    // Bring the screen up to date. The transcript takes whatever the region leaves.
    //
    // The size is passed in rather than read here so that a frame can be built for a
    // screen that is not the one haico happens to be running on, which is the only way the
    // tests can say anything about what gets painted.
    fn paint(
        &mut self,
        region: &[String],
        caret: (usize, usize),
        capture: &mut Option<String>,
        (width, height): (usize, usize),
    ) {
        let region_rows = region.len().min(height);
        let body = height - region_rows;

        // A column short of the edge, for the same reason everything else stops there: a
        // row exactly as wide as the screen wraps in some terminals and not others.
        let wrap = width.saturating_sub(1).max(1);

        // An anchor that has ended up at or past the end is not holding anything back any
        // more — the region may have shrunk, or lines may have fallen off the front — so it
        // goes back to following rather than sitting one row short of the bottom forever.
        if let Some(anchor) = self.scroll {
            if anchor >= self.bottom_anchor(wrap, body) {
                self.scroll = None;
            }
        }

        // The row that says the view is held back costs a row of transcript, and there has
        // to be one to spare for it.
        let held = self.scroll.filter(|_| body >= 2);
        let content = body - usize::from(held.is_some());
        let transcript = match held {
            Some(anchor) => self.rows_from(anchor, wrap, content),
            None => self.rows(wrap, content),
        };

        let mut rows = Vec::with_capacity(height);
        // A transcript shorter than the viewport sits against the bottom, the way a shell
        // leaves one, rather than at the top with the input box stranded in the middle.
        rows.extend(std::iter::repeat(String::new()).take(content - transcript.len()));
        rows.extend(transcript);
        if let Some(anchor) = held {
            let index = anchor.seq.saturating_sub(self.dropped) as usize;
            rows.push(held_back_row(
                self.line_count().saturating_sub(index + 1),
                width,
            ));
        }
        // A region taller than the screen loses its top rows rather than its bottom ones.
        // `input::draw` budgets the plan panel so this cannot normally happen, but if it
        // ever does the thing to keep is the box and the bar under it — dropping those
        // would leave the user typing into a screen with no visible input.
        let dropped = region.len() - region_rows;
        rows.extend(region.iter().skip(dropped).cloned());

        let caret = (
            (body + caret.0.saturating_sub(dropped)).min(height.saturating_sub(1)),
            caret.1.min(width.saturating_sub(1)),
        );

        // A resize invalidates the whole frame rather than any part of it: every row was
        // laid out for the old width and the rows below the first change have all moved.
        // Cheaper and far more reliable than working out what survived.
        let full = self.drawn_size != (width, height) || self.drawn.len() != rows.len();
        if !full && rows == self.drawn && caret == self.drawn_caret {
            return;
        }

        let mut frame = String::with_capacity(1024);
        frame.push_str(BEGIN_SYNC);
        frame.push_str(WRAP_OFF);
        frame.push_str(&Hide.to_string());
        if full {
            frame.push_str(&Clear(ClearType::All).to_string());
        }
        for (row, text) in rows.iter().enumerate() {
            if !full && self.drawn[row] == *text {
                continue;
            }
            frame.push_str(&MoveTo(0, row as u16).to_string());
            frame.push_str(text);
            // A row that ends with a color still open would paint it across everything the
            // clear touches, all the way to the right edge.
            frame.push_str("\x1b[0m");
            frame.push_str(&Clear(ClearType::UntilNewLine).to_string());
        }
        frame.push_str(&MoveTo(caret.1 as u16, caret.0 as u16).to_string());
        frame.push_str(&Show.to_string());
        frame.push_str(WRAP_ON);
        frame.push_str(END_SYNC);

        if let Some(buffer) = capture {
            buffer.push_str(&frame);
        } else {
            let mut stdout = io::stdout();
            let _ = stdout.write_all(frame.as_bytes());
            let _ = stdout.flush();
        }

        self.drawn = rows;
        self.drawn_caret = caret;
        self.drawn_size = (width, height);
    }

    // The transcript as it would have looked had it been printed to the terminal all
    // along, for writing back to the main screen on the way out.
    //
    // Without this the whole session goes with the alternate screen when it is restored,
    // which is the one thing the inline renderer got for free. Wrapped at haico's width
    // rather than left to the terminal, so what lands in the scrollback is what was on
    // the screen.
    fn dump(&self, width: usize) -> String {
        let mut out = String::new();
        for line in &self.lines {
            // What was on the screen is what goes into the scrollback: a detail that was
            // folded away was not on the screen.
            if !self.shows(line.kind) {
                continue;
            }
            out.push_str(&wrap_rows(&line.text, width).join("\n"));
            out.push('\n');
        }
        if !self.open.is_empty() && self.shows(self.writing) {
            out.push_str(&wrap_rows(&(self.carry.clone() + &self.open), width).join("\n"));
            out.push('\n');
        }
        out
    }
}

fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(columns, _)| columns as usize)
        .unwrap_or(80)
        .max(1)
}

fn terminal_height() -> usize {
    crossterm::terminal::size()
        .map(|(_, rows)| rows as usize)
        .unwrap_or(24)
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
pub(crate) fn wrap_width() -> usize {
    terminal_width().saturating_sub(1).max(1)
}

// Which colors are still switched on at the end of a row, given the ones that were on at
// the start of it. Only SGR sequences — the ones ending in "m" — say anything about color;
// a cursor move carries nothing over. A reset closes everything, so it clears what has
// been collected; anything else joins it in the order it was written, which is the order
// it has to be written again in to put the terminal back the way the row left it.
fn open_colors(open: &str, row: &str) -> String {
    let mut colors = open.to_string();
    let mut chars = row.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            continue;
        }
        let escape = take_escape(&mut chars);
        let Some(params) = escape
            .strip_prefix("\u{1b}[")
            .and_then(|rest| rest.strip_suffix('m'))
        else {
            continue;
        };
        // "\x1b[m" and "\x1b[0m" are the same reset; a list of them still resets.
        if params.split(';').all(|param| param.is_empty() || param == "0") {
            colors.clear();
        } else {
            colors.push_str(&escape);
        }
    }
    colors
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
            painted.contains(&format!("{}\u{1b}[90m── held back", super::MoveTo(0, 7))),
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
}
