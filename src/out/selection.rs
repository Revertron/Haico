//! Selecting transcript text with the mouse, and copying it.
//!
//! Asking the terminal for the wheel means asking it for the buttons too — there is no
//! mode that reports one without the other — and from then on a plain drag is reported to
//! haico instead of selecting anything. The terminal's own selection was still there
//! behind Shift, but nothing but another Shift+click could take it away again: a plain
//! click went to haico, so the highlight stayed on screen. Doing the selection here gives
//! the plain drag back, and a plain click clears it.
//!
//! It also copies better than the terminal could. Every row is painted at a fixed
//! position with auto-wrap off, so a terminal copying a wrapped paragraph gets a hard line
//! break at every place haico wrapped it, the hanging indent at the start of each row, and
//! the input box and status bar if the drag strayed into them. Here a selection is a range
//! of the logical lines, and what is copied is their text.

use super::*;

// A place in the transcript: a logical line, and a character of its text with every escape
// taken out.
//
// Not a screen cell. The view scrolls, output keeps streaming in under a drag, and a
// resize re-wraps every line; a cell means a different character after each of those,
// while a line and a character in it still mean the same one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) struct Point {
    pub(super) seq: u64,
    pub(super) offset: usize,
}

pub(super) struct Selection {
    // Where the button went down, and where it is now. Either may be the earlier one.
    pub(super) anchor: Point,
    pub(super) head: Point,
    // A press and release with no drag between them is a click, and a click clears the
    // selection rather than making a one-character one.
    pub(super) moved: bool,
    // Still between the press and the release.
    pub(super) dragging: bool,
}

impl Selection {
    // The first and last character selected, both included.
    pub(super) fn bounds(&self) -> (Point, Point) {
        (self.anchor.min(self.head), self.anchor.max(self.head))
    }

    // Which characters of one line are selected, as a half-open range, or `None` for a
    // line the selection does not reach.
    pub(super) fn columns_of(&self, seq: u64, length: usize) -> Option<(usize, usize)> {
        let (first, last) = self.bounds();
        if seq < first.seq || seq > last.seq {
            return None;
        }
        let from = if seq == first.seq { first.offset.min(length) } else { 0 };
        let to = if seq == last.seq { (last.offset + 1).min(length) } else { length };
        (from < to).then_some((from, to))
    }
}

// Which wrapped row of which logical line a screen row is showing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct RowId {
    pub(super) seq: u64,
    pub(super) row: usize,
}

// Where one wrapped row's text sits in its line: the characters `start..start + length` of
// the line's plain text, drawn after `indent` cells of hanging indent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Span {
    pub(super) start: usize,
    pub(super) length: usize,
    pub(super) indent: usize,
}

// A line's text as it prints, a character at a time.
pub(super) fn plain_chars(line: &str) -> Vec<char> {
    without_escapes(line).chars().collect()
}

// Where each of a line's wrapped rows comes from.
//
// Worked out from what `Wrap` produced rather than by wrapping a second way, so the two
// cannot disagree about where a row starts. The wrapper only ever does three things to a
// line: it puts in breaks, it starts every row after the first with the hanging indent,
// and it drops the blanks a break lands on. So the first row is the start of the line, and
// each later one, less its indent, is the text after the blanks that follow the row above.
pub(super) fn spans(line: &str, width: usize) -> Vec<Span> {
    let chars = plain_chars(line);
    let indent = hanging_indent(line, width.max(1));
    let mut spans = Vec::new();
    let mut position = 0;
    for (index, row) in wrap_rows(line, width).iter().enumerate() {
        let row: Vec<char> = without_escapes(row).chars().collect();
        let skip = if index == 0 { 0 } else { indent.min(row.len()) };
        let text = &row[skip..];
        if index > 0 {
            while chars.get(position).is_some_and(|ch| *ch == ' ' || *ch == '\t') {
                position += 1;
            }
        }
        // A row that is not where it was expected is a wrapper rule this does not know
        // about. Found by searching on rather than assumed, so the worst it can do is
        // highlight a row that is not quite the one under the pointer.
        if !chars[position.min(chars.len())..].starts_with(text) {
            if let Some(found) = (position..chars.len()).find(|&at| chars[at..].starts_with(text)) {
                position = found;
            }
        }
        let length = text.len().min(chars.len().saturating_sub(position));
        spans.push(Span { start: position, length, indent: skip });
        position += length;
    }
    spans
}

// The character a column of a row lands on.
//
// Left of the text — in the hanging indent — is its first character, and right of it is
// its last, so dragging out past either end of a row takes the row whole rather than
// stopping wherever the text happened to.
pub(super) fn offset_at(chars: &[char], span: Span, column: usize) -> usize {
    let mut cells = span.indent;
    for (index, ch) in chars.iter().enumerate().skip(span.start).take(span.length) {
        cells += cell_width(*ch);
        if column < cells {
            return index;
        }
    }
    span.start + span.length.saturating_sub(1)
}

// The cells a range of characters covers on one row, or `None` where the range misses the
// row entirely.
pub(super) fn cells_of(chars: &[char], span: Span, (from, to): (usize, usize)) -> Option<(usize, usize)> {
    let start = from.max(span.start);
    let end = to.min(span.start + span.length);
    if start >= end {
        return None;
    }
    let before: usize = chars[span.start..start].iter().map(|ch| cell_width(*ch)).sum();
    let inside: usize = chars[start..end].iter().map(|ch| cell_width(*ch)).sum();
    Some((span.indent + before, span.indent + before + inside))
}

const REVERSE_ON: &str = "\x1b[7m";
const REVERSE_OFF: &str = "\x1b[27m";

// The row with the cells `from..to` in reverse video.
//
// Reverse rather than a color of its own, because it is the one highlight that reads on
// a light theme and a dark one alike and on whatever colors the text already has.
//
// A reset inside the range would switch reverse off with everything else, so it is put
// back before the next character that needs it. It is switched off straight after the last
// one rather than at the end of the row: the user's band is filled to the edge by an erase
// written after the row, and an erase made while reverse is still on paints the rest of
// the row in it.
pub(super) fn highlight(row: &str, from: usize, to: usize) -> String {
    let mut out = String::with_capacity(row.len() + 16);
    let mut column = 0;
    let mut on = false;
    let mut chars = row.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            let escape = take_escape(&mut chars);
            if on && ends_reverse(&escape) {
                on = false;
            }
            out.push_str(&escape);
            continue;
        }
        let width = cell_width(ch);
        if width > 0 && !on && column >= from && column < to {
            out.push_str(REVERSE_ON);
            on = true;
        }
        out.push(ch);
        column += width;
        if on && column >= to {
            out.push_str(REVERSE_OFF);
            on = false;
        }
    }
    if on {
        out.push_str(REVERSE_OFF);
    }
    out
}

// Whether an SGR sequence switches reverse video off: a reset, or 27 itself.
fn ends_reverse(escape: &str) -> bool {
    let Some(params) = escape.strip_prefix("\x1b[").and_then(|rest| rest.strip_suffix('m')) else {
        return false;
    };
    params.is_empty() || params.split(';').any(|param| param.is_empty() || param == "0" || param == "27")
}

impl Screen {
    // Which line and row each of the `height` rows from `anchor` shows. The same walk as
    // `rows_from`, counting rows instead of building them.
    pub(super) fn row_ids_from(&self, anchor: Anchor, width: usize, height: usize) -> Vec<RowId> {
        let mut index = anchor.seq.saturating_sub(self.dropped) as usize;
        let mut skip = anchor.row;
        let mut ids = Vec::with_capacity(height);
        while ids.len() < height && self.exists(index) {
            let rows = self.rows_in(index, width);
            let seq = self.dropped + index as u64;
            for row in skip..rows {
                if ids.len() == height {
                    break;
                }
                ids.push(RowId { seq, row });
            }
            skip = 0;
            index += 1;
        }
        ids
    }

    // The plain text of a line by sequence number, if it is still here and shown.
    fn plain_line(&self, seq: u64) -> Option<(String, Vec<char>)> {
        let index = seq.checked_sub(self.dropped)? as usize;
        let line = self.line_at(index)?.into_owned();
        let chars = plain_chars(&line);
        Some((line, chars))
    }

    // The transcript rows of a frame with the selection painted over them.
    pub(super) fn highlighted(&self, rows: &mut [String], ids: &[RowId], width: usize) {
        let Some(selection) = &self.selection else {
            return;
        };
        // A line runs over several rows, and its spans are the same for all of them.
        let mut cached: Option<(u64, Vec<char>, Vec<Span>)> = None;
        for (row, id) in rows.iter_mut().zip(ids) {
            if cached.as_ref().is_none_or(|(seq, ..)| *seq != id.seq) {
                cached = self
                    .plain_line(id.seq)
                    .map(|(line, chars)| (id.seq, chars, spans(&line, width)));
            }
            let Some((_, chars, spans)) = &cached else {
                continue;
            };
            let Some(columns) = selection.columns_of(id.seq, chars.len()) else {
                continue;
            };
            let Some(span) = spans.get(id.row) else {
                continue;
            };
            if let Some((from, to)) = cells_of(chars, *span, columns) {
                *row = highlight(row, from, to);
            }
        }
    }

    // The transcript position under a screen cell of the frame last painted, and whether
    // the cell was on the transcript at all.
    //
    // Off it, the nearest end of what is shown: the start of the top row for the blank
    // space above a short transcript, the end of the bottom row for the held-back notice
    // and everything pinned under it. That is what lets a drag carry on past either edge.
    pub(super) fn point_at(&self, column: usize, row: usize) -> Option<(Point, bool)> {
        let shown: Vec<(usize, RowId)> = self
            .drawn_ids
            .iter()
            .enumerate()
            .filter_map(|(at, id)| id.map(|id| (at, id)))
            .collect();
        let (first_row, first) = *shown.first()?;
        let (last_row, last) = *shown.last()?;
        let (id, column, inside) = match self.drawn_ids.get(row).copied().flatten() {
            Some(id) => (id, column, true),
            None if row < first_row => (first, 0, false),
            None if row > last_row => (last, usize::MAX, false),
            None => return None,
        };
        let (line, chars) = self.plain_line(id.seq)?;
        let span = *spans(&line, self.drawn_wrap).get(id.row)?;
        Some((Point { seq: id.seq, offset: offset_at(&chars, span, column) }, inside))
    }

    // What the selection covers, as text: lines joined with line feeds, each without the
    // blanks at its end. A line folded away or fallen off the front is not on the screen,
    // so it is not copied either.
    pub(super) fn selected_text(&self) -> Option<String> {
        let selection = self.selection.as_ref()?;
        let (first, last) = selection.bounds();
        let mut lines = Vec::new();
        for seq in first.seq..=last.seq {
            let Some((_, chars)) = self.plain_line(seq) else {
                continue;
            };
            let (from, to) = selection.columns_of(seq, chars.len()).unwrap_or((0, 0));
            let text: String = chars[from..to].iter().filter(|ch| **ch != '\r').collect();
            lines.push(text.trim_end().to_string());
        }
        let text = lines.join("\n");
        (!text.trim().is_empty()).then_some(text)
    }

    pub(super) fn press(&mut self, column: usize, row: usize) {
        self.selection = match self.point_at(column, row) {
            Some((point, true)) => Some(Selection {
                anchor: point,
                head: point,
                moved: false,
                dragging: true,
            }),
            // A press on the input box or the bars clears what was selected and starts
            // nothing: those are not transcript.
            _ => None,
        };
    }

    pub(super) fn drag(&mut self, column: usize, row: usize) {
        let point = match &self.selection {
            Some(selection) if selection.dragging => self.point_at(column, row),
            _ => return,
        };
        if let (Some(selection), Some((point, _))) = (&mut self.selection, point) {
            selection.moved |= point != selection.anchor;
            selection.head = point;
        }
    }

    // The button came up: what to copy, if anything was selected.
    pub(super) fn release(&mut self) -> Option<String> {
        let selection = self.selection.as_mut().filter(|selection| selection.dragging)?;
        selection.dragging = false;
        if !selection.moved {
            self.selection = None;
            return None;
        }
        self.selected_text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every row of a wrapped line found where it really is, over the shapes the wrapper
    // treats differently: a bullet's hanging indent, a word too long for any row, blanks
    // that a break lands on, wide characters, colors.
    #[test]
    fn spans_put_every_wrapped_row_back_where_it_came_from() {
        let lines = [
            "the quick brown fox jumps over the lazy dog".to_string(),
            "- a bullet whose text runs on long enough to wrap under itself".to_string(),
            "x".repeat(50),
            "words    with    wide    gaps    between    them    all    along".to_string(),
            "表示 する 文字 が 二つ の 幅 を 取る 行 です ね ほら".to_string(),
            "\x1b[31mred \x1b[1mbold\x1b[0m and plain words after the reset carry on".to_string(),
        ];
        for line in &lines {
            for width in [7, 12, 20] {
                let chars = plain_chars(line);
                let rows = wrap_rows(line, width);
                let spans = spans(line, width);
                assert_eq!(spans.len(), rows.len(), "{line:?} at {width}");
                for (row, span) in rows.iter().zip(&spans) {
                    let expected: String = without_escapes(row).chars().skip(span.indent).collect();
                    let found: String = chars[span.start..span.start + span.length].iter().collect();
                    assert_eq!(found, expected, "{line:?} at {width}");
                }
            }
        }
    }

    // Wide characters take two cells each, and the indent of a continued row is not text.
    #[test]
    fn a_column_lands_on_the_character_drawn_there() {
        let chars: Vec<char> = "ab表c".chars().collect();
        let span = Span { start: 0, length: 4, indent: 2 };
        let at = |column| offset_at(&chars, span, column);
        assert_eq!([at(0), at(2), at(3), at(4), at(5), at(6), at(40)], [0, 0, 1, 2, 2, 3, 3]);
        assert_eq!(cells_of(&chars, span, (1, 3)), Some((3, 6)));
        assert_eq!(cells_of(&chars, span, (4, 9)), None);
    }

    #[test]
    fn the_highlight_survives_a_reset_and_stops_before_the_fill() {
        let row = "\x1b[44m> hi\x1b[0m there";
        assert_eq!(
            highlight(row, 2, 6),
            "\x1b[44m> \x1b[7mhi\x1b[0m\x1b[7m t\x1b[27mhere"
        );
        // Selected to the end of the text: the closing reset still comes last, so the
        // band's fill is drawn without reverse on.
        assert_eq!(highlight("\x1b[44mhi\x1b[0m", 0, 2), "\x1b[44m\x1b[7mhi\x1b[27m\x1b[0m");
    }

    // Painted with a selection, a frame says which rows are which; a press and a drag over
    // it select across lines and copy their text without the wrap.
    #[test]
    fn a_drag_selects_logical_text_and_a_click_clears_it() {
        let mut screen = Screen::new();
        for line in ["first line", "- second line that is long enough to wrap", "third"] {
            screen.write(line);
            screen.write("\n");
        }
        let mut capture = Some(String::new());
        screen.paint(&["> ".to_string()], (0, 0), &mut capture, (21, 8));
        let shown: Vec<_> = screen.drawn_ids.iter().flatten().collect();
        assert!(shown.len() >= 4, "{:?}", screen.drawn_ids);

        // From the "l" of "first line" to the end of the second line's last row.
        let top = screen.drawn_ids.iter().position(Option::is_some).unwrap();
        let second_last = screen
            .drawn_ids
            .iter()
            .rposition(|id| id.is_some_and(|id| id.seq == 1))
            .unwrap();
        screen.press(6, top);
        screen.drag(80, second_last);
        let copied = screen.release().expect("something was selected");
        assert_eq!(copied, "line\n- second line that is long enough to wrap");

        let mut capture = Some(String::new());
        screen.paint(&["> ".to_string()], (0, 0), &mut capture, (21, 8));
        assert!(capture.unwrap().contains(REVERSE_ON));

        // A click with no drag clears it, and the next frame repaints without it.
        screen.press(0, top);
        assert_eq!(screen.release(), None);
        assert!(screen.selection.is_none());
        let mut capture = Some(String::new());
        screen.paint(&["> ".to_string()], (0, 0), &mut capture, (21, 8));
        assert!(!capture.unwrap().contains(REVERSE_ON));

        // A press on the input box selects nothing.
        screen.press(1, 7);
        screen.drag(5, top);
        assert_eq!(screen.release(), None);
    }
}
