//! Terminal rendering: streamed markdown, and the colored lines of the tool log and diffs.

use crate::diff::{Diff, DiffRow, LineKind};
use crate::summary::{strip_project_prefix, tool_call_summary};
use crate::util::truncate_at_boundary;
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor};
use serde_json::Value;
use std::fmt::Write as _;

// Markdown as it arrives.
//
// A stream cannot be rendered a fragment at a time: "**bold**" turns up as several
// pieces and means nothing until its closing pair lands. A whole line can be, and a
// line is the unit markdown is written in — so text waits here until its newline and
// is rendered then, with the code-fence state carried from one line to the next.
pub(crate) struct MarkdownStream {
    buffer: String,
    in_code_block: bool,
    // Blank lines seen but not yet printed.
    //
    // A blank line only earns its row once something follows it. At the end of a reply
    // nothing does: models routinely end a message on one or two newlines, and those used
    // to print as a gap before whatever came next — the tool call line, the next prompt.
    // Held here instead, flushed when a real line arrives, and dropped by `finish`.
    held_blanks: usize,
    // How many bytes of the line still in `buffer` are already on the screen.
    //
    // A reply used to appear a whole line at a time, because a line cannot be painted
    // until it is known what it is: a leading "# " colors all of it, "```" turns the next
    // rows into a code block, and `**bold**` is only bold once its closing pair arrives.
    // But that is a statement about the *undecided* part of a line, and most of a line of
    // prose is decided long before it ends. So as much of it as is settled is written as
    // it arrives, and this is how much of it has gone — the rest of the line, when it
    // finally ends, is all that is left to write.
    streamed: usize,
    // The rows of a table that has started but not yet ended.
    //
    // The one thing here that cannot be rendered a line at a time. A column is as wide as
    // its widest cell, so the shape of the first row is not known until the last one has
    // arrived — the rows wait here, and the first line that is not one draws the table and
    // is then rendered after it. A single row is only a candidate: it becomes a table when
    // a delimiter row lands under it, and goes out as prose when one never does.
    table: Vec<String>,
}

impl MarkdownStream {
    pub(crate) fn new() -> Self {
        MarkdownStream {
            buffer: String::new(),
            in_code_block: false,
            held_blanks: 0,
            streamed: 0,
            table: Vec::new(),
        }
    }

    // Whatever is now settled, rendered: every line that has ended, and as much of the
    // one still arriving as cannot be changed by what follows it.
    pub(crate) fn push(&mut self, text: &str) -> String {
        self.buffer.push_str(text);
        let mut rendered = String::new();
        while let Some(end) = self.buffer.find('\n') {
            let line: String = self.buffer.drain(..=end).collect();
            let line = line.trim_end_matches('\n');

            // Inside a fence a blank line is content, painted like every other line of
            // the block, so only prose blanks are held back. A line that has already had
            // part of itself written cannot be blank, so this never fires under one.
            if !self.in_code_block && line.trim().is_empty() {
                // A blank line ends a table, and the rows belong above the gap rather than
                // below it. The blanks are held back, so without this they would be paid
                // out in front of a table that had not been drawn yet.
                rendered.push_str(&self.flush_table(true));
                self.held_blanks += 1;
                continue;
            }

            // Something followed after all, so they were separators rather than trailing
            // whitespace: pay them out before the line that earned them.
            rendered.push_str(&"\n".repeat(std::mem::take(&mut self.held_blanks)));
            rendered.push_str(&self.rest_of_line(line, true));
        }
        rendered.push_str(&self.stream_settled());
        rendered
    }

    // As much of the unfinished line as can be shown now.
    //
    // Only prose: a fence, a heading or a line inside a code block is painted as a whole
    // and would have to be unpainted to correct it. And only up to the last word boundary
    // that no inline markup crosses — an opener with no closer yet is exactly the case
    // where the next few characters decide what the ones already here mean.
    fn stream_settled(&mut self) -> String {
        // Rows are waiting to be drawn, and anything written now would land in front of
        // them. The line being streamed here is the one that will end the table.
        if !self.table.is_empty() {
            return String::new();
        }
        if self.buffer.is_empty() || looks_like_prose(self.in_code_block, &self.buffer) != Some(true) {
            return String::new();
        }
        let cut = settled_prefix(&self.buffer);
        if cut <= self.streamed {
            return String::new();
        }

        let mut rendered = render_inline_markdown(&self.buffer[self.streamed..cut]);
        // Each piece is rendered on its own and they are simply written one after another,
        // so none of them may carry the newline that ends the line.
        rendered.truncate(rendered.trim_end_matches('\n').len());
        // The blank rows this line earned are paid before the first of it reaches the
        // screen, which is where they would have gone had the line arrived all at once.
        if self.streamed == 0 {
            rendered.insert_str(0, &"\n".repeat(std::mem::take(&mut self.held_blanks)));
        }
        self.streamed = cut;
        rendered
    }

    // What is left of a line to write. A line nothing has been shown of is rendered whole,
    // exactly as it always was; one that has been streamed renders only its tail, which
    // comes to the same bytes because the cut never falls inside anything markup spans.
    fn rest_of_line(&mut self, line: &str, newline: bool) -> String {
        let streamed = std::mem::take(&mut self.streamed);
        if streamed == 0 {
            return self.line(line, newline);
        }
        let mut rendered = render_inline_markdown(&line[streamed..]);
        if !newline {
            rendered.truncate(rendered.trim_end_matches('\n').len());
        }
        rendered
    }

    // The last line, which the model usually ends without a newline of its own. Blank
    // lines still held at this point were trailing after all, and are dropped.
    pub(crate) fn finish(&mut self, newline: bool) -> String {
        self.held_blanks = 0;
        if self.buffer.is_empty() {
            self.streamed = 0;
            return self.flush_table(newline);
        }
        let line = std::mem::take(&mut self.buffer);
        let rendered = self.rest_of_line(&line, newline);
        // A table that runs to the end of a reply never meets the line that would have
        // ended it, so this is the only place left to draw it.
        rendered + &self.flush_table(newline)
    }

    // The table drawn, and the rows forgotten. A candidate that never found a delimiter
    // row was not a table at all, and goes out as the prose it always was.
    fn flush_table(&mut self, newline: bool) -> String {
        let rows = std::mem::take(&mut self.table);
        if rows.is_empty() {
            return String::new();
        }
        let mut rendered = if rows.len() < 2 {
            self.plain_line(&rows[0], true)
        } else {
            render_table(&rows)
        };
        if !newline {
            rendered.truncate(rendered.trim_end_matches('\n').len());
        }
        rendered
    }

    fn line(&mut self, line: &str, newline: bool) -> String {
        // Inside a fence a pipe is code, and none of this applies.
        if self.in_code_block {
            return self.plain_line(line, newline);
        }
        let held = self.table.len();
        // A row joins the table it belongs to; the first line that is not one draws what
        // has collected and is then rendered after it, in the place it would have had.
        if held == 1 && is_delimiter_row(line) || held > 1 && is_table_row(line) {
            self.table.push(line.to_string());
            return String::new();
        }
        if held > 0 {
            return self.flush_table(true) + &self.line(line, newline);
        }
        if is_table_row(line) {
            self.table.push(line.to_string());
            return String::new();
        }
        self.plain_line(line, newline)
    }

    fn plain_line(&mut self, line: &str, newline: bool) -> String {
        let ending = if newline { "\n" } else { "" };
        let trimmed = line.trim();

        // Handle code blocks
        if trimmed.starts_with("```") {
            self.in_code_block = !self.in_code_block;
            return String::new();
        }

        if self.in_code_block {
            // Same look as an inline span: blue on the terminal's own background, so a
            // dark theme is not banded with a light strip per row.
            return format!(
                "{}  {}{}{}",
                SetForegroundColor(Color::Blue),
                line,
                ResetColor,
                ending
            );
        }

        // Handle headers
        for (marker, color) in [
            ("# ", Color::Cyan),
            ("## ", Color::Cyan),
            ("### ", Color::Blue),
        ] {
            if trimmed.starts_with(marker) {
                return format!(
                    "{}\n{}{}{}",
                    SetForegroundColor(color),
                    trimmed.trim_start_matches(marker),
                    ending,
                    ResetColor
                );
            }
        }

        // Process inline formatting
        let mut rendered = render_inline_markdown(line);
        if !newline {
            rendered.truncate(rendered.trim_end_matches('\n').len());
        }
        rendered
    }
}

// Whether the line so far is ordinary prose, which can be written a word at a time, or
// something whose look depends on the whole line and so cannot be written until it ends.
// `None` while too little of it has arrived to tell the two apart.
fn looks_like_prose(in_code_block: bool, line: &str) -> Option<bool> {
    // A row of a code block is painted whole — it takes no inline formatting, and the
    // fence that ends the block is never printed at all.
    if in_code_block {
        return Some(false);
    }
    let trimmed = line.trim_start();
    // Still nothing but indentation: this may yet turn out to be a blank line, which is
    // held back rather than printed.
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with("```") {
        return Some(false);
    }
    // One backtick, or two, could still become the third.
    if trimmed.chars().all(|ch| ch == '`') {
        return None;
    }
    if ["# ", "## ", "### "].iter().any(|marker| trimmed.starts_with(marker)) {
        return Some(false);
    }
    // A table row waits for longer than that: not only what it is but how wide each of
    // its columns prints depends on rows that have not arrived.
    if trimmed.starts_with('|') {
        return Some(false);
    }
    // The same for the hashes: "##" is a heading if a space comes next and prose if a
    // fourth hash does.
    if trimmed.len() <= 3 && trimmed.chars().all(|ch| ch == '#') {
        return None;
    }
    Some(true)
}

// Tables.
//
// The one construct here that cannot be drawn a line at a time, which is why the stream
// holds rows instead: a column is as wide as its widest cell, so the header cannot be laid
// out until the last row has arrived. Before this every row went out as prose — pipes,
// dashes and all — and that only lines up if the cells happen to be the same length, which
// they never are.

// A row is any line starting with a pipe. That is deliberately looser than the delimiter
// test below: a row on its own means nothing until the line under it says whether this is
// a table, and it is that line which has to be matched exactly.
fn is_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.len() > 1
}

fn is_delimiter_row(line: &str) -> bool {
    if !is_table_row(line) {
        return false;
    }
    let cells = split_row(line);
    !cells.is_empty()
        && cells.iter().all(|cell| {
            let dashes = cell.trim_start_matches(':').trim_end_matches(':');
            !dashes.is_empty() && dashes.chars().all(|ch| ch == '-')
        })
}

// The cells of one row. A pipe the writer escaped is content rather than a divider, and is
// the only way a cell can hold one at all — so the trailing pipe of a row ending in an
// escaped one is not the row's own.
fn split_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let inner = match inner.strip_suffix('|') {
        Some(rest) if !rest.ends_with('\\') => rest,
        _ => inner,
    };

    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' if chars.peek() == Some(&'|') => {
                chars.next();
                cell.push('|');
            }
            '|' => cells.push(std::mem::take(&mut cell)),
            _ => cell.push(ch),
        }
    }
    cells.push(cell);
    cells.iter().map(|cell| cell.trim().to_string()).collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Align {
    Left,
    Center,
    Right,
}

fn alignment(delimiter: &str) -> Align {
    match (delimiter.starts_with(':'), delimiter.ends_with(':')) {
        (true, true) => Align::Center,
        (false, true) => Align::Right,
        _ => Align::Left,
    }
}

// The narrowest a column may be squeezed to. Under this a wrapped cell is more line breaks
// than text, and nothing is gained by making room for it.
const MIN_COLUMN: usize = 4;

fn render_table(rows: &[String]) -> String {
    layout_table(rows, crate::out::wrap_width())
}

// The width is passed in rather than read here so that a test lays a table out at a width
// it chose, instead of at whatever the terminal running the test happens to be.
fn layout_table(rows: &[String], available: usize) -> String {
    let mut parsed: Vec<Vec<String>> = rows.iter().map(|row| split_row(row)).collect();
    if parsed.len() < 2 {
        return String::new();
    }
    // The delimiter row is not content: it carries the alignments and nothing else.
    let delimiter = parsed.remove(1);
    let columns = parsed[0].len().max(1);
    let aligns: Vec<Align> = (0..columns)
        .map(|column| {
            delimiter
                .get(column)
                .map(|cell| alignment(cell))
                .unwrap_or(Align::Left)
        })
        .collect();

    // Rendered once, here, and measured as rendered: the markers come out of the text on
    // the way through, so `**done**` prints four columns narrower than it was written. The
    // header row decides how many columns there are — a short row is padded out and a long
    // one is cut, rather than knocking the rest of the table out of line.
    let cells: Vec<Vec<String>> = parsed
        .iter()
        .map(|row| {
            (0..columns)
                .map(|column| {
                    // `render_inline_markdown` ends the line for a caller that is
                    // printing one. A cell is not a line: left on, that newline counts a
                    // column toward the width and wraps every row to two rows tall.
                    let cell =
                        render_inline_markdown(row.get(column).map(String::as_str).unwrap_or(""));
                    cell.trim_end().to_string()
                })
                .collect()
        })
        .collect();

    let mut widths: Vec<usize> = (0..columns)
        .map(|column| {
            cells
                .iter()
                .map(|row| crate::out::visible_width(&row[column]))
                .max()
                .unwrap_or(1)
                .max(1)
        })
        .collect();

    // Every column costs a space either side of itself, and there is one border more than
    // there are columns. What gives when that does not fit is the widest column, one step
    // at a time: a version or a yes/no keeps its natural width, and the prose column —
    // which is what a too-wide table always turns out to have — is the one that wraps.
    let frame = 3 * columns + 1;
    while frame + widths.iter().sum::<usize>() > available {
        let widest = widths
            .iter()
            .enumerate()
            .max_by_key(|(_, width)| **width)
            .map(|(column, _)| column);
        match widest {
            Some(column) if widths[column] > MIN_COLUMN => widths[column] -= 1,
            // All of them are at the floor already. A table with more columns than the
            // terminal has room for is drawn too wide rather than not drawn at all.
            _ => break,
        }
    }

    let mut table = String::new();
    table.push_str(&table_rule("┌", "┬", "┐", &widths));
    for (index, row) in cells.iter().enumerate() {
        if index == 1 {
            table.push_str(&table_rule("├", "┼", "┤", &widths));
        }
        table.push_str(&table_row(row, &widths, &aligns, index == 0));
    }
    table.push_str(&table_rule("└", "┴", "┘", &widths));
    table
}

fn table_rule(left: &str, joint: &str, right: &str, widths: &[usize]) -> String {
    let mut rule = String::new();
    let _ = write!(rule, "{}{}", SetForegroundColor(crate::out::DIM), left);
    for (index, width) in widths.iter().enumerate() {
        if index > 0 {
            rule.push_str(joint);
        }
        rule.push_str(&"─".repeat(width + 2));
    }
    let _ = write!(rule, "{}{}\n", right, ResetColor);
    rule
}

// One row of cells, as many screen rows tall as its tallest cell wraps to. The wrapping is
// `out`'s own, so a cell whose color runs past a break has it reopened on the row below
// exactly as a wrapped line of prose does — and `out` will not wrap these lines a second
// time, because the layout above already fitted them.
fn table_row(cells: &[String], widths: &[usize], aligns: &[Align], header: bool) -> String {
    let wrapped: Vec<Vec<String>> = cells
        .iter()
        .zip(widths)
        .map(|(cell, width)| crate::out::wrap_rows(cell, *width))
        .collect();
    let height = wrapped.iter().map(|cell| cell.len()).max().unwrap_or(1).max(1);

    let mut rows = String::new();
    for row in 0..height {
        let _ = write!(rows, "{}│", SetForegroundColor(crate::out::DIM));
        for (column, cell) in wrapped.iter().enumerate() {
            let piece = cell.get(row).map(String::as_str).unwrap_or("");
            let pad = widths[column].saturating_sub(crate::out::visible_width(piece));
            let (before, after) = match aligns[column] {
                Align::Left => (0, pad),
                Align::Right => (pad, 0),
                Align::Center => (pad / 2, pad - pad / 2),
            };
            let _ = write!(
                rows,
                "{} {}{}{}{}{} {}│",
                ResetColor,
                " ".repeat(before),
                if header {
                    SetAttribute(Attribute::Bold).to_string()
                } else {
                    String::new()
                },
                piece,
                ResetColor,
                " ".repeat(after),
                SetForegroundColor(crate::out::DIM)
            );
        }
        let _ = write!(rows, "{}\n", ResetColor);
    }
    rows
}

// Inline markup starting at a position, as far as the text that has arrived can say.
enum Inline {
    // Not the start of any: an ordinary character.
    No,
    // One that has found its closing marker; it ends just before this index.
    Whole(usize),
    // One that has opened and could still be closed by text that has not arrived. Nothing
    // from its opener onwards can be written yet: `**bold` is two literal stars until the
    // closing pair turns the whole of it bold.
    Open,
}

fn inline_at(chars: &[char], at: usize) -> Inline {
    match chars[at] {
        '`' => match find_closing('`', chars, at + 1) {
            Some(end) => Inline::Whole(end + 1),
            None => Inline::Open,
        },
        '*' if chars.get(at + 1) == Some(&'*') => match find_double_closing("**", chars, at + 2) {
            Some(end) => Inline::Whole(end + 2),
            None => Inline::Open,
        },
        // A lone marker is only an opener under the same test the renderer applies, so the
        // underscores in an identifier are ordinary characters here too.
        marker @ ('*' | '_') if opens_emphasis(marker, chars, at) => {
            match closing_emphasis(marker, chars, at + 1) {
                Some(end) => Inline::Whole(end + 1),
                None => Inline::Open,
            }
        }
        '[' => {
            let Some(close) = find_closing(']', chars, at + 1) else {
                return Inline::Open;
            };
            match chars.get(close + 1) {
                // Whether this is a link at all is the next character's to say.
                None => Inline::Open,
                Some('(') => match find_closing(')', chars, close + 2) {
                    Some(end) => Inline::Whole(end + 1),
                    None => Inline::Open,
                },
                Some(_) => Inline::No,
            }
        }
        _ => Inline::No,
    }
}

// How much of an unfinished line is settled: the longest prefix that ends on a word
// boundary and that no character still to arrive can change the meaning of.
//
// Rendering a line in pieces has to come to the same bytes as rendering it whole, which
// it does as long as no piece ends inside something markup spans — every construct is
// rendered from its own characters alone, and the one place a piece boundary is visible
// from the outside is the "not inside a word" test on an underscore, which reads a space
// before the cut and the start of the piece as the same thing.
fn settled_prefix(line: &str) -> usize {
    let indices: Vec<(usize, char)> = line.char_indices().collect();
    let chars: Vec<char> = indices.iter().map(|(_, ch)| *ch).collect();

    let mut settled = 0;
    let mut at = 0;
    while at < chars.len() {
        match inline_at(&chars, at) {
            Inline::Open => return settled,
            Inline::Whole(end) => at = end,
            Inline::No => {
                // Words are never cut in half: partly because half a word on the screen
                // reads as a typo, and partly because the writer wraps on spaces and a
                // word split here would be free to wrap in the middle of itself.
                if chars[at].is_whitespace() {
                    settled = indices[at].0 + chars[at].len_utf8();
                }
                at += 1;
            }
        }
    }
    settled
}

pub(crate) fn render_markdown(text: &str) -> String {
    let mut stream = MarkdownStream::new();
    let mut rendered = stream.push(text);
    rendered.push_str(&stream.finish(true));
    rendered
}

// Render a line with inline Markdown formatting (bold, italic, code, links)


pub(crate) fn render_inline_markdown(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut rendered = String::new();
    let mut i = 0;

    while i < chars.len() {
        // Check for inline code (`code`)
        if chars[i] == '`' {
            if let Some(end) = find_closing('`', &chars, i + 1) {
                let _ = write!(rendered, "{}", SetForegroundColor(Color::Blue));
                for j in (i + 1)..end {
                    rendered.push(chars[j]);
                }
                let _ = write!(rendered, "{}", ResetColor);
                i = end + 1;
                continue;
            }
        }

        // Check for bold (**text**)
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_closing("**", &chars, i + 2) {
                let _ = write!(rendered, "{}", SetAttribute(Attribute::Bold));
                for j in (i + 2)..end {
                    rendered.push(chars[j]);
                }
                let _ = write!(rendered, "{}", SetAttribute(Attribute::Reset));
                i = end + 2;
                continue;
            }
        }

        // Check for italic (*text* or _text_)
        if chars[i] == '*' || chars[i] == '_' {
            let marker = chars[i];
            // Make sure it's not part of **
            let part_of_bold = marker == '*' && i + 1 < chars.len() && chars[i + 1] == '*';
            if !part_of_bold && opens_emphasis(marker, &chars, i) {
                if let Some(end) = closing_emphasis(marker, &chars, i + 1) {
                    let _ = write!(rendered, "{}", SetForegroundColor(Color::Green));
                    for j in (i + 1)..end {
                        rendered.push(chars[j]);
                    }
                    let _ = write!(rendered, "{}", ResetColor);
                    i = end + 1;
                    continue;
                }
            }
        }

        // Check for links ([text](url))
        if chars[i] == '[' {
            if let Some(close_bracket) = find_closing(']', &chars, i + 1) {
                if close_bracket + 1 < chars.len() && chars[close_bracket + 1] == '(' {
                    if let Some(close_paren) = find_closing(')', &chars, close_bracket + 2) {
                        // Render link text in blue
                        let _ = write!(rendered, "{}", SetForegroundColor(Color::Blue));
                        for j in (i + 1)..close_bracket {
                            rendered.push(chars[j]);
                        }
                        let _ = write!(rendered, "{}", ResetColor);

                        // Render URL in dark grey
                        let _ = write!(rendered, "{} (", SetForegroundColor(crate::out::DIM));
                        for j in (close_bracket + 2)..close_paren {
                            rendered.push(chars[j]);
                        }
                        let _ = write!(rendered, "){}", ResetColor);

                        i = close_paren + 1;
                        continue;
                    }
                }
            }
        }

        // Regular character
        rendered.push(chars[i]);
        i += 1;
    }

    rendered.push('\n');
    rendered
}

// Find closing character from a starting position

// Whether a '*' or '_' at this position is opening emphasis rather than just sitting
// in the middle of something.
//
// Without this, rendering every line as markdown mangles the text this tool spends
// most of its time on: `load_project_context and build_system_prompt` has four
// underscores in it, and the naive rule turns the middle of two identifiers green.
// The test is the one CommonMark uses, cut down: an opener is followed by something
// other than a space, and an underscore additionally may not sit inside a word.
fn opens_emphasis(marker: char, chars: &[char], at: usize) -> bool {
    let next = chars.get(at + 1);
    if !next.is_some_and(|ch| !ch.is_whitespace()) {
        return false;
    }
    if marker == '_' {
        if at > 0 && chars[at - 1].is_alphanumeric() {
            return false;
        }
    }
    true
}

// The matching close: not preceded by a space, and for an underscore not glued to the
// front of a word either.
fn closing_emphasis(marker: char, chars: &[char], start: usize) -> Option<usize> {
    for i in start..chars.len() {
        if chars[i] != marker {
            continue;
        }
        if i == start || chars[i - 1].is_whitespace() {
            continue;
        }
        if marker == '_' && chars.get(i + 1).is_some_and(|ch| ch.is_alphanumeric()) {
            continue;
        }
        return Some(i);
    }
    None
}

pub(crate) fn find_closing(marker: char, chars: &[char], start: usize) -> Option<usize> {
    for i in start..chars.len() {
        if chars[i] == marker {
            return Some(i);
        }
    }
    None
}

// Find closing double-character marker (like **)

pub(crate) fn find_double_closing(marker: &str, chars: &[char], start: usize) -> Option<usize> {
    let marker_chars: Vec<char> = marker.chars().collect();
    if marker_chars.len() != 2 {
        return None;
    }

    for i in start..(chars.len() - 1) {
        if chars[i] == marker_chars[0] && chars[i + 1] == marker_chars[1] {
            return Some(i);
        }
    }
    None
}

// Announce a tool call before it runs.
//
// Read-only tools are auto-allowed and never prompt, and once "allow all" is answered for
// run_command nothing prompts either — so without this the only visible sign of a model
// going in circles was a repeated "Fancying around..." and a long wait. Seeing the same
// call scroll past ten times is what makes a loop obvious while it is still cheap to stop.
// `note` explains why a call that would normally have stopped to ask is running anyway.
// It rides on the call line rather than a line of its own, so a run that was never
// interrupted still reads as one line per call — and so the reason is impossible to miss
// while scrolling back through what was allowed on the user's behalf.

pub(crate) fn tool_call_line(tool_name: &str, arguments: &Value, note: Option<&str>) -> String {
    let summary = tool_call_summary(tool_name, arguments);
    let mut line = String::new();

    // The name is the only part painted: the arrow, the summary and the result line
    // underneath stay dim, so the call reads as the heading of the step and what it did
    // stays subordinate to it. Cyan because that is already what a heading is here.
    let _ = write!(
        line,
        "{}  → {}{}{}{}{}",
        SetForegroundColor(crate::out::DIM),
        SetAttribute(Attribute::Bold),
        SetForegroundColor(Color::Cyan),
        tool_name,
        SetAttribute(Attribute::Reset),
        SetForegroundColor(crate::out::DIM)
    );
    if !summary.is_empty() {
        // Not clipped. Cutting at 110 characters is harmless for a path, whose front says
        // most of what it is, and wrong for everything whose meaning is in the tail: a
        // command read as a build when the `&& rm -rf` fell off the end, and a long regex
        // lost exactly the part that narrowed it. out.rs wraps the line, so the whole of
        // it costs rows rather than legibility. Whitespace is still collapsed, so a
        // heredoc arrives as one wrapped paragraph rather than ten rows of its own.
        let summary: String = summary.split_whitespace().collect::<Vec<_>>().join(" ");
        if matches!(tool_name, "run_command" | "run_background") {
            // A command in the plain text color, though, unlike a path or a pattern: it
            // is the one summary that has to be read rather than recognized — whether
            // the analyzer was right to let it through is decided from it — and in dim
            // grey a long one was hard going. The analyzer's note after it stays dim.
            let _ = write!(
                line,
                "  {}{}{}",
                ResetColor,
                summary,
                SetForegroundColor(crate::out::DIM)
            );
        } else {
            let _ = write!(line, "  {}", summary);
        }
    }
    if let Some(note) = note.map(str::trim).filter(|n| !n.is_empty()) {
        // Not clipped either, and for a sharper reason than the command: this is the whole
        // account the user gets of why consent was spent on their behalf, on a call that
        // would otherwise have stopped to ask them. A verdict cut off mid-sentence keeps
        // the part that restates the call and loses the part that justifies it.
        let note: String = note.split_whitespace().collect::<Vec<_>>().join(" ");
        let _ = write!(line, "  (allowed by auto analyzer: {})", note);
    }
    let _ = write!(line, "{}", ResetColor);
    line
}

// A diff as the terminal shows it: dim line numbers, and the changed lines in black on a
// red or a green band. The text is `Diff::plain`'s, line for line, with color added.
pub(crate) fn diff_text(diff: &Diff) -> String {
    let mut out = format!("File: {}\n", diff.file);
    if diff.rows.is_empty() {
        out.push_str("No changes\n");
    }
    // Changed lines are written in black on the red or the green behind them
    let black = SetForegroundColor(Color::Black);
    let number = |n: usize| format!("{}{:6}{} ", SetForegroundColor(crate::out::DIM), n, ResetColor);
    for row in &diff.rows {
        let _ = match row {
            DiffRow::Gap => writeln!(out, "       ..."),
            DiffRow::Line { kind: LineKind::Context, number: n, text } => writeln!(out, "{}  {}", number(*n), text),
            DiffRow::Line { kind, number: n, text } => {
                let band = if *kind == LineKind::Removed { Color::Red } else { Color::Green };
                writeln!(
                    out,
                    "{}{}{}{} {}{}",
                    number(*n),
                    SetBackgroundColor(band),
                    black,
                    kind.marker(),
                    text,
                    ResetColor
                )
            }
        };
    }
    out
}

// Show a diff inline in the log, indented under the call that produced it.
//
// Capped, because a newly written file renders as one "+" line per line of content and a
// large one would bury everything around it. The head is what tells you whether the file is
// what you expected; the tail rarely changes that.

pub(crate) fn change_preview(diff: &str) -> Option<String> {
    const MAX_LINES: usize = 24;

    // The first line is "File: <path>", already shown by the call line above.
    let body: Vec<&str> = diff.lines().skip(1).collect();
    if body.is_empty() {
        return None;
    }

    let mut preview = String::new();
    for line in body.iter().take(MAX_LINES) {
        let _ = write!(preview, "    {}\n", line);
    }

    if body.len() > MAX_LINES {
        let _ = write!(
            preview,
            "{}    … {} more line(s){}\n",
            SetForegroundColor(crate::out::DIM),
            body.len() - MAX_LINES,
            ResetColor
        );
    }
    Some(preview)
}

// One line of what a tool gave back. The repeated identical result is usually the clearest
// evidence that a model is stuck, so it belongs in the log next to the call.

pub(crate) fn tool_result_line(text: &str) -> Option<String> {
    let first = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    if first.is_empty() {
        return None;
    }

    // Whole, not cut at 110 characters. The fold behind this row holds the rest of the
    // result, but the first line is not the first 110 characters of one — "error[E0308]:
    // mismatched types" is the half that names the error, and the file and line it wants
    // are in the half that used to be dropped.
    let first = strip_project_prefix(first);
    Some(format!(
        "{}    {}{}",
        SetForegroundColor(crate::out::DIM),
        first,
        ResetColor
    ))
}

#[cfg(test)]
mod diff_tests {
    use super::*;

    #[test]
    fn only_the_lines_with_a_background_are_written_in_black() {
        crate::out::own_colors();
        let black = SetForegroundColor(Color::Black).to_string();
        let diff = diff_text(&crate::diff::generate_diff("a.txt", "keep\none\n", "keep\ntwo\n"));

        // Two lines carry it, and they are exactly the two haico paints a background
        // behind: the one removed and the one added. A third would mean the context line
        // got it too, where nothing is painted and black would be black on a black screen.
        assert_eq!(diff.matches(&black).count(), 2, "got {:?}", diff);
        for line in diff.lines().filter(|line| line.contains(&black)) {
            let colored = line.contains(&SetBackgroundColor(Color::Red).to_string())
                || line.contains(&SetBackgroundColor(Color::Green).to_string());
            assert!(colored, "black on an unpainted line: {:?}", line);
        }

        // The color is all it adds: the lines, the numbers and the markers are exactly the
        // plain text the reviewer reads.
        let source = crate::diff::generate_diff("a.txt", "keep\none\n", "keep\ntwo\n");
        assert_eq!(crate::util::strip_ansi(&diff), source.plain());
    }
}

// Lay a command out so it can actually be read before being approved.
//
// It used to be cut at 50 characters, which for a generated one-liner meant approving a
// script sight unseen. Wrapped to the terminal width and indented, with a generous ceiling
// so a pathological command cannot flood the screen either.
pub(crate) fn indented_command(command: &str) -> String {
    const MAX: usize = 4000;
    let shown = truncate_at_boundary(command.trim(), MAX);
    let width = crate::out::terminal_size().map_or(100, |(columns, _)| columns.saturating_sub(6).max(40));

    let mut out = String::new();
    for line in shown.lines() {
        let mut rest = line;
        while !rest.is_empty() {
            let mut take = truncate_at_boundary(rest, width);
            if take.is_empty() {
                take = rest;
            }
            // Break at a space when there is one near the end of the line. Splitting through
            // the middle of a token is legible but jarring, and the whole point here is that
            // the command can be read. Never break early enough to hide anything.
            if take.len() < rest.len() {
                if let Some(space) = take.rfind(' ').filter(|at| *at > width * 3 / 4) {
                    take = &take[..space + 1];
                }
            }
            out.push_str("    ");
            out.push_str(take.trim_end());
            out.push('\n');
            rest = rest[take.len()..].trim_start_matches(' ');
        }
    }
    if shown.len() < command.trim().len() {
        out.push_str("    ... [truncated]\n");
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod markdown_tests {
    use super::*;

    #[test]
    fn a_stream_renders_the_same_as_the_whole_text_at_once() {
        const TEXT: &str = "# Title\nSome **bold** and `code`.\n\nplain line\ndone\n";
        let whole = render_markdown(TEXT);

        // Fed three characters at a time, the way a model sends it — the line breaks
        // fall in the middle of chunks, which is the whole point.
        let mut stream = MarkdownStream::new();
        let mut piecemeal = String::new();
        let chars: Vec<char> = TEXT.chars().collect();
        for chunk in chars.chunks(3) {
            piecemeal.push_str(&stream.push(&chunk.iter().collect::<String>()));
        }
        piecemeal.push_str(&stream.finish(true));

        assert_eq!(piecemeal, whole);
    }

    #[test]
    fn prose_reaches_the_screen_a_word_at_a_time() {
        let mut stream = MarkdownStream::new();

        // The words that have ended do not wait for the end of the line; the one still
        // being spelled does.
        assert_eq!(stream.push("The quick bro"), "The quick ");
        assert_eq!(stream.push("wn fox "), "brown fox ");

        // Markup that has opened and not closed holds everything from its opener: what
        // arrives next is what decides whether those are two stars or the start of bold.
        assert_eq!(stream.push("is **very "), "is ");
        let rest = stream.push("bold** now\n");
        assert!(rest.contains("very bold"), "got {:?}", rest);
        assert!(!rest.contains("**"), "got {:?}", rest);

        // A heading is painted as a whole, so none of it shows until it ends.
        let mut stream = MarkdownStream::new();
        assert_eq!(stream.push("# A title stil"), "");
        assert!(stream.push("l coming\n").contains("A title still coming"));

        // And so is every row of a code block.
        let mut stream = MarkdownStream::new();
        assert_eq!(stream.push("```rust\nlet x = 1"), "");
        assert!(stream.push(";\n").contains("let x = 1;"));
    }

    #[test]
    fn a_word_at_a_time_still_comes_to_the_same_bytes() {
        // Everything the renderer treats specially, in one text: a heading, a link, a code
        // span, bold, an indented continuation, a fence, a table, and a blank line between
        // paragraphs.
        const TEXT: &str = "# Title\nA [link](http://x/y), `code`, **bold** here.\n\n- a list item\n  indented continuation\n```rust\nlet x = 1;\n```\n| a | bb |\n|---|--:|\n| 1 | 2 |\ndone\n";
        let whole = render_markdown(TEXT);
        let chars: Vec<char> = TEXT.chars().collect();

        // Every chunk size from one character up, because a chunk boundary falling in the
        // middle of a marker — between the two stars of a bold, before a fence's third
        // backtick — is exactly what this has to survive.
        for size in 1..=7 {
            let mut stream = MarkdownStream::new();
            let mut piecemeal = String::new();
            for chunk in chars.chunks(size) {
                piecemeal.push_str(&stream.push(&chunk.iter().collect::<String>()));
            }
            piecemeal.push_str(&stream.finish(true));
            assert_eq!(piecemeal, whole, "chunks of {}", size);
        }
    }

    #[test]
    fn trailing_blank_lines_never_reach_the_screen() {
        // What a local model actually sends when it is about to call a tool: a sentence
        // and then one or two newlines to end its turn. Those used to print as a gap
        // between the reply and the tool call line below it.
        let mut stream = MarkdownStream::new();
        let mut rendered = stream.push("Let me look at the diff.\n\n");
        rendered.push_str(&stream.finish(false));
        assert_eq!(rendered.matches('\n').count(), 1, "got {:?}", rendered);
        assert!(rendered.ends_with('\n'));

        // A blank line between two paragraphs is a separator, not trailing whitespace,
        // and still gets its row.
        let mut stream = MarkdownStream::new();
        let mut rendered = stream.push("first\n\nsecond\n\n\n");
        rendered.push_str(&stream.finish(false));
        assert_eq!(rendered.matches('\n').count(), 3, "got {:?}", rendered);

        // Held across chunk boundaries too, since a newline and what follows it rarely
        // arrive in the same piece — and a blank held in one chunk and paid out in the
        // next must come to exactly what it would have unsplit.
        let mut stream = MarkdownStream::new();
        let mut split = stream.push("first\n");
        split.push_str(&stream.push("\n"));
        split.push_str(&stream.push("second\n"));
        split.push_str(&stream.finish(false));
        assert_eq!(split, rendered.trim_end_matches('\n').to_string() + "\n");

        // Inside a fence a blank line is content: dropping it would rewrite the code.
        let mut stream = MarkdownStream::new();
        let rendered = stream.push("```\nlet x = 1;\n\nlet y = 2;\n```\n");
        assert_eq!(rendered.matches('\n').count(), 3, "got {:?}", rendered);
    }

    #[test]
    fn a_fence_opened_in_one_chunk_still_closes_in_another() {
        crate::out::own_colors();
        let mut stream = MarkdownStream::new();
        let mut rendered = stream.push("```rust\nlet x = 1;\n");
        rendered.push_str(&stream.push("```\nafter\n"));

        // The fenced line got the code color; the line after it did not.
        assert!(rendered.contains("let x = 1;"));
        let code = rendered.find("let x = 1;").unwrap();
        let after = rendered.find("after").unwrap();
        assert!(rendered[..code].ends_with("38;5;12m  "), "no code color: {:?}", rendered);
        assert!(!rendered[code..after].contains("38;5;12m"));
        // And nothing paints a background behind it.
        assert!(!rendered.contains("48;5;"), "code is banded: {:?}", rendered);
    }

    // What a table actually looks like on the screen, with the colors taken back out. The
    // frame is the assertion here, so it is compared as text rather than as bytes.
    fn plain(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(ch) = chars.next() {
            if ch != '\u{1b}' {
                out.push(ch);
                continue;
            }
            if chars.next() == Some('[') {
                for ch in chars.by_ref() {
                    if ('@'..='~').contains(&ch) {
                        break;
                    }
                }
            }
        }
        out
    }

    fn rows(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|line| line.to_string()).collect()
    }

    #[test]
    fn a_table_is_laid_out_from_its_widest_cell() {
        let table = rows(&[
            "| Dependency | Version |",
            "|---|---:|",
            "| actix-files | 0.7.0 |",
            "| toml | 1.1.5 |",
        ]);
        let drawn = plain(&layout_table(&table, 60));
        let lines: Vec<&str> = drawn.lines().collect();

        // "Dependency" is the header, but "actix-files" is a row lower down and one
        // column wider — which is the whole reason none of this can be drawn until the
        // last row has arrived.
        assert_eq!(
            lines,
            vec![
                "┌─────────────┬─────────┐",
                "│ Dependency  │ Version │",
                "├─────────────┼─────────┤",
                "│ actix-files │   0.7.0 │",
                "│ toml        │   1.1.5 │",
                "└─────────────┴─────────┘",
            ],
            "got {:?}",
            drawn
        );
    }

    // A cell holding a character the terminal draws two cells wide. Measured in
    // characters the row is a character shorter than the frame; measured in cells, which
    // is what the screen actually is, it has to be exactly as wide. Getting that wrong is
    // what put the right-hand border of every row one column past the corner above it.
    #[test]
    fn a_wide_character_in_a_cell_does_not_push_the_border_past_the_frame() {
        let table = rows(&[
            "| Test | Status |",
            "|---|---|",
            "| Host-based routing | \u{2611} Pass |",
            "| Missing file | \u{2611} Pass |",
        ]);
        let drawn = plain(&layout_table(&table, 60));

        let widths: Vec<usize> = drawn.lines().map(crate::out::visible_width).collect();
        assert!(
            widths.windows(2).all(|pair| pair[0] == pair[1]),
            "the rows are not all the same width on screen: {:?} in {:?}",
            widths,
            drawn
        );
        // And it really is the wide character doing it, rather than every row happening
        // to hold the same number of characters.
        let characters: Vec<usize> = drawn.lines().map(|line| line.chars().count()).collect();
        assert!(
            characters.iter().any(|count| *count != characters[0]),
            "the test is not exercising a wide character: {:?}",
            drawn
        );
    }

    #[test]
    fn a_table_too_wide_for_the_terminal_wraps_its_widest_column() {
        let table = rows(&[
            "| Dependency | Version | Role |",
            "|---|---|---|",
            "| toml | 1.1.5 | TOML config parsing built with only parse and serde features |",
        ]);
        let drawn = plain(&layout_table(&table, 40));

        // Every row of the frame is the same width, and it is the width asked for rather
        // than whatever the widest cell wanted: nothing here is left for `out` to wrap a
        // second time, which would have broken the frame apart.
        for line in drawn.lines() {
            assert_eq!(line.chars().count(), 40, "got {:?}", drawn);
        }

        // The columns that were already narrow kept their width; what gave is the prose
        // column, which is now several rows tall.
        assert!(drawn.contains("Dependency"), "got {:?}", drawn);
        assert!(drawn.lines().count() > 6, "got {:?}", drawn);

        // And nothing was dropped on the way: every word of the long cell is still there.
        let words: String = drawn
            .lines()
            .flat_map(|line| line.split('│').skip(3).next())
            .collect::<Vec<&str>>()
            .join(" ");
        for word in "TOML config parsing built with only parse and serde features".split(' ') {
            assert!(words.contains(word), "{} missing from {:?}", word, drawn);
        }
    }

    #[test]
    fn a_table_waits_for_the_line_that_ends_it() {
        let mut stream = MarkdownStream::new();
        // Not one of these can be painted yet, so not one of them prints.
        assert_eq!(stream.push("| Dependency | Version |\n"), "");
        assert_eq!(stream.push("|---|---|\n"), "");
        assert_eq!(stream.push("| actix-files | 0.7.0 |\n"), "");

        // The first line that is not a row draws the table, and then follows it.
        let rendered = plain(&stream.push("after\n"));
        assert!(rendered.starts_with('┌'), "got {:?}", rendered);
        assert!(rendered.contains("actix-files"), "got {:?}", rendered);
        assert!(rendered.ends_with("┘\nafter\n"), "got {:?}", rendered);
        // The pipes and dashes it was written with are gone.
        assert!(!rendered.contains('|'), "got {:?}", rendered);

        // A blank line ends one too, and the rows go above the gap rather than below it.
        let mut stream = MarkdownStream::new();
        stream.push("| a | b |\n|---|---|\n| 1 | 2 |\n");
        let rendered = plain(&stream.push("\nafter\n"));
        assert!(rendered.starts_with('┌'), "got {:?}", rendered);
        assert!(rendered.ends_with("┘\n\nafter\n"), "got {:?}", rendered);

        // And a table that runs to the end of the reply is drawn by `finish`, which is
        // the only thing left that can: no line ever arrives to end it.
        let mut stream = MarkdownStream::new();
        assert_eq!(stream.push("| a | b |\n|---|---|\n| 1 | 2 |\n"), "");
        let rendered = plain(&stream.finish(true));
        assert!(rendered.starts_with('┌'), "got {:?}", rendered);
        assert!(rendered.contains('│'), "got {:?}", rendered);
    }

    #[test]
    fn a_row_with_nothing_under_it_is_prose() {
        // A line can start with a pipe and mean nothing by it. It is the delimiter row
        // that makes a table, so without one the line is printed as it was written.
        let mut stream = MarkdownStream::new();
        assert_eq!(stream.push("| just | text |\n"), "");
        assert_eq!(plain(&stream.push("more\n")), "| just | text |\nmore\n");

        // Including when it is the last thing in the reply.
        let mut stream = MarkdownStream::new();
        assert_eq!(stream.push("| just | text |\n"), "");
        assert_eq!(plain(&stream.finish(true)), "| just | text |\n");

        // Inside a fence a pipe is code, not the start of anything.
        let mut stream = MarkdownStream::new();
        let rendered = plain(&stream.push("```\n| a | b |\n```\ndone\n"));
        assert!(rendered.contains("| a | b |"), "got {:?}", rendered);
    }

    #[test]
    fn a_cell_keeps_its_own_markup_and_its_escaped_pipes() {
        assert_eq!(split_row("| a | b |"), vec!["a", "b"]);
        // A pipe the writer escaped divides nothing — it is the only way a cell can
        // hold one at all.
        assert_eq!(split_row(r"| a \| b | c |"), vec!["a | b", "c"]);
        assert_eq!(split_row(r"| a | b \|"), vec!["a", "b |"]);

        assert!(is_delimiter_row("|---|:---:|---:|"));
        assert!(is_delimiter_row("| - | - |"));
        assert!(!is_delimiter_row("| -- | text |"));
        assert!(!is_delimiter_row("| :: | :: |"));

        // The markers come out of the text, so the cell is measured as it prints: this
        // fits a column that "**done**" would have overflowed by four.
        let table = rows(&["| Status |", "|---|", "| **done** |"]);
        let drawn = plain(&layout_table(&table, 60));
        assert!(drawn.contains("│ done   │"), "got {:?}", drawn);
        assert!(!drawn.contains('*'), "got {:?}", drawn);
    }

    #[test]
    fn identifiers_are_not_mistaken_for_emphasis() {
        // The text this tool spends its life on: underscores inside words, and a lone
        // asterisk between numbers. Both used to come out half green.
        for plain in [
            "call load_project_context and build_system_prompt",
            "2 * 3 * 4",
            "a_b_c_d",
        ] {
            assert_eq!(
                render_inline_markdown(plain),
                format!("{}\n", plain),
                "{}",
                plain
            );
        }
    }

    #[test]
    fn emphasis_still_renders_where_it_is_meant() {
        let rendered = render_inline_markdown("an *italic* and _another_ one");
        assert!(!rendered.contains("*italic*"), "got {:?}", rendered);
        assert!(!rendered.contains("_another_"), "got {:?}", rendered);
        assert!(rendered.contains("italic") && rendered.contains("another"));
    }
}

