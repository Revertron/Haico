//! The full-screen renderer: the transcript as logical lines, wrapped when a frame is
//! built, scrolled by anchor, folded by line kind.

use super::*;

// The row that says the view is being held back, and by how much.
//
// It costs a row of transcript, and it is worth it: output carries on arriving while the
// view stays where it was put, which without something saying so looks exactly like haico
// having stopped.
pub(super) fn held_back_row(below: usize, width: usize) -> String {
    let text = format!(
        "── held back · {} line{} below · PageDown or the wheel to follow ──",
        below,
        if below == 1 { "" } else { "s" }
    );
    let text: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("\x1b[38;5;245m{}\x1b[0m", text)
}

// How many logical lines of transcript are kept. The buffer has to be bounded — the
// inline renderer needed no limit because a line became the terminal's problem the moment
// it was written, and these never do. A session long enough to reach this has scrolled
// well past anything anyone is going back for.
pub(super) const SCROLLBACK_LINES: usize = 10_000;

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
pub(super) fn text_and_color(text: &str) -> std::borrow::Cow<'_, str> {
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
pub(super) enum Kind {
    // Always shown.
    Normal,
    // The one row standing in for something folded away.
    Summary,
    // What that row stands in for.
    Detail,
}

pub(super) struct Line {
    pub(super) text: String,
    pub(super) kind: Kind,
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
pub(super) struct Anchor {
    pub(super) seq: u64,
    pub(super) row: usize,
}

// The transcript, and the screen it is painted onto.
//
// Lines are kept *logical* — as they were written, not as they were last shown — and
// wrapped only when a frame is built. That is the whole trick: a resize changes nothing
// here, it just produces different rows next time, and a line rewritten later takes its
// new width from wherever the screen happens to be by then.
pub(super) struct Screen {
    // Finished lines, oldest first. Each is self-contained: it begins with whatever colors
    // were still open when the line before it ended, so it can be wrapped and painted
    // without reading anything that came earlier.
    pub(super) lines: VecDeque<Line>,
    // The line being written that has not found its newline yet. It is part of the frame
    // like any other, which is why a streamed reply appears as it arrives rather than a
    // line at a time — but it stays out of `lines` until it ends, because until then its
    // last word is still free to move to another row.
    pub(super) open: String,
    // The colors open at the end of the last finished line, prefixed onto the next one.
    pub(super) carry: String,
    // What kind the lines being written now are. Set by `begin_fold` and put back by
    // `end_fold`, so a whole streamed run of output can be marked as detail without every
    // writer along the way having to know about it.
    pub(super) writing: Kind,
    // Whether the detail is being shown instead of the summaries. One flag for the whole
    // transcript, which is what makes this a toggle rather than a thing to navigate.
    pub(super) expanded: bool,
    // How many lines have fallen off the front, which is what turns a sequence number into
    // an index and back.
    pub(super) dropped: u64,
    // Where the top of the viewport is, or `None` to stay at the end and follow whatever
    // arrives. Scrolling back down to the end puts it back to `None` rather than leaving
    // an anchor pinned there, so new output starts moving the view again by itself.
    pub(super) scroll: Option<Anchor>,
    // The rows exactly as they are on the screen, and where the cursor was left. The next
    // frame is compared against these and only the rows that differ are painted again.
    pub(super) drawn: Vec<String>,
    pub(super) drawn_caret: (usize, usize),
    // The size the frame was laid out for. A different one means every row has moved and
    // none of them can be patched.
    pub(super) drawn_size: (usize, usize),
    // Whether mouse reporting was asked for, so leaving can turn off exactly what
    // entering turned on.
    pub(super) mouse: bool,
    // The Windows console input mode as it was before VT input was switched on.
    #[cfg(windows)]
    pub(super) console_mode: Option<u32>,
}

impl Screen {
    pub(super) fn new() -> Self {
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

    pub(super) fn write(&mut self, text: &str) {
        let text = text_and_color(text);
        for (index, part) in text.split('\n').enumerate() {
            if index > 0 {
                self.end_line();
            }
            self.open.push_str(part);
        }
    }

    pub(super) fn end_line(&mut self) {
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
    pub(super) fn forget(&mut self) {
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

    pub(super) fn shows(&self, kind: Kind) -> bool {
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
    pub(super) fn line_at(&self, index: usize) -> Option<std::borrow::Cow<'_, str>> {
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
    pub(super) fn exists(&self, index: usize) -> bool {
        index < self.line_count()
    }

    pub(super) fn line_count(&self) -> usize {
        self.lines.len() + usize::from(!self.open.is_empty())
    }

    pub(super) fn rows_in(&self, index: usize, width: usize) -> usize {
        self.line_at(index)
            .map(|line| Wrap::new(width).feed(&line).split('\n').count())
            .unwrap_or(0)
    }

    // The anchor that shows the last `height` rows — where the view sits when it is
    // following. Walked backward from the end, which is the only direction that can find
    // it without wrapping the whole transcript.
    pub(super) fn bottom_anchor(&self, width: usize, height: usize) -> Anchor {
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
    pub(super) fn rows_from(&self, anchor: Anchor, width: usize, height: usize) -> Vec<String> {
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
    pub(super) fn step_back(&self, mut anchor: Anchor, count: usize, width: usize) -> Anchor {
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

    pub(super) fn step_forward(&self, mut anchor: Anchor, count: usize, width: usize, bottom: Anchor) -> Anchor {
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

    pub(super) fn scroll_by(&mut self, delta: isize, width: usize, height: usize) {
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
    pub(super) fn rows(&self, width: usize, height: usize) -> Vec<String> {
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
    pub(super) fn paint(
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
        rows.extend(std::iter::repeat_n(String::new(), content - transcript.len()));
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
            let _ = stdout.write_all(for_the_terminal(&frame).as_bytes());
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
    pub(super) fn dump(&self, width: usize) -> String {
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
