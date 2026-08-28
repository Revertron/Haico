//! Terminal rendering: markdown, diffs, and the model picker.

use crate::input::Keys;
use anyhow::Result;
use crossterm::{
    cursor,
    event::{Event, KeyCode, KeyEventKind},
    execute,
    style::{Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor},
    terminal::{Clear, ClearType},
};
use serde_json::Value;
use similar::{ChangeTag, TextDiff};
use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

// Which way round the terminal's own colors run.
//
// Everything haico paints was picked against a light background, and all of it holds up
// against a dark one except the diff: its lines are left in the terminal's own text color,
// which is exactly what makes them read as a quotation of the file rather than as
// something haico wrote, and against the red and the green of a change that color is too
// dim to sit on. Nothing here detects it — the query that would (OSC 11) needs the
// keyboard that `Keys` owns, and answers only on a timeout that not every terminal keeps —
// so `/dark` and `/light` say which it is, for the session.
static DARK: AtomicBool = AtomicBool::new(false);

pub(crate) fn set_dark(dark: bool) {
    DARK.store(dark, Ordering::Relaxed);
}

pub(crate) fn is_dark() -> bool {
    DARK.load(Ordering::Relaxed)
}

pub(crate) fn generate_diff(file_path: &str, old_content: &str, new_content: &str) -> Result<String> {
    let diff = TextDiff::from_lines(old_content, new_content);
    let mut output = Vec::new();

    writeln!(&mut output, "File: {}", file_path)?;

    const CONTEXT_LINES: usize = 5;

    // First pass: collect all changes with their positions
    #[derive(Clone)]
    struct Change {
        tag: ChangeTag,
        text: String,
        old_line: Option<usize>,
        new_line: Option<usize>,
    }

    let mut changes = Vec::new();
    let mut old_line_num = 1;
    let mut new_line_num = 1;

    for change in diff.iter_all_changes() {
        let text = change.as_str().unwrap_or("");
        let text_trimmed = text.trim_end_matches('\n').trim_end_matches('\r').to_string();

        match change.tag() {
            ChangeTag::Delete => {
                changes.push(Change {
                    tag: ChangeTag::Delete,
                    text: text_trimmed,
                    old_line: Some(old_line_num),
                    new_line: None,
                });
                old_line_num += 1;
            }
            ChangeTag::Insert => {
                changes.push(Change {
                    tag: ChangeTag::Insert,
                    text: text_trimmed,
                    old_line: None,
                    new_line: Some(new_line_num),
                });
                new_line_num += 1;
            }
            ChangeTag::Equal => {
                changes.push(Change {
                    tag: ChangeTag::Equal,
                    text: text_trimmed,
                    old_line: Some(old_line_num),
                    new_line: Some(new_line_num),
                });
                old_line_num += 1;
                new_line_num += 1;
            }
        }
    }

    // Second pass: identify change regions
    let mut change_indices = Vec::new();
    for (i, change) in changes.iter().enumerate() {
        if !matches!(change.tag, ChangeTag::Equal) {
            change_indices.push(i);
        }
    }

    if change_indices.is_empty() {
        writeln!(&mut output, "No changes")?;
        return Ok(String::from_utf8(output)?);
    }

    // Group consecutive changes into regions with context
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut current_start = change_indices[0].saturating_sub(CONTEXT_LINES);
    let mut current_end = change_indices[0] + CONTEXT_LINES;

    for &idx in &change_indices[1..] {
        let region_start = idx.saturating_sub(CONTEXT_LINES);
        let region_end = idx + CONTEXT_LINES;

        // If regions overlap or are adjacent, merge them
        if region_start <= current_end + 1 {
            current_end = region_end;
        } else {
            // Save current region and start a new one
            regions.push((current_start, current_end.min(changes.len() - 1)));
            current_start = region_start;
            current_end = region_end;
        }
    }
    regions.push((current_start, current_end.min(changes.len() - 1)));

    // Both empty on a light background, where the terminal's own text color is the right
    // one and the output stays byte for byte what it always was.
    let (text_color, text_reset) = if is_dark() {
        (
            SetForegroundColor(Color::White).to_string(),
            ResetColor.to_string(),
        )
    } else {
        (String::new(), String::new())
    };

    // Render regions
    for (region_idx, (start, end)) in regions.iter().enumerate() {
        if region_idx > 0 {
            writeln!(&mut output, "       ...")?;
        }

        for i in *start..=*end {
            let change = &changes[i];
            match change.tag {
                ChangeTag::Delete => {
                    if let Some(line_num) = change.old_line {
                        execute!(&mut output, SetForegroundColor(Color::DarkGrey))?;
                        write!(&mut output, "{:6}", line_num)?;
                        execute!(&mut output, ResetColor)?;
                        write!(&mut output, " ")?;
                        execute!(&mut output, SetBackgroundColor(Color::Red))?;
                        write!(&mut output, "{}- {}", text_color, change.text)?;
                        execute!(&mut output, ResetColor)?;
                        writeln!(&mut output)?;
                    }
                }
                ChangeTag::Insert => {
                    if let Some(line_num) = change.new_line {
                        execute!(&mut output, SetForegroundColor(Color::DarkGrey))?;
                        write!(&mut output, "{:6}", line_num)?;
                        execute!(&mut output, ResetColor)?;
                        write!(&mut output, " ")?;
                        execute!(&mut output, SetBackgroundColor(Color::Green))?;
                        write!(&mut output, "{}+ {}", text_color, change.text)?;
                        execute!(&mut output, ResetColor)?;
                        writeln!(&mut output)?;
                    }
                }
                ChangeTag::Equal => {
                    if let Some(line_num) = change.old_line {
                        execute!(&mut output, SetForegroundColor(Color::DarkGrey))?;
                        write!(&mut output, "{:6}", line_num)?;
                        execute!(&mut output, ResetColor)?;
                        write!(&mut output, " ")?;
                        writeln!(&mut output, "{}  {}{}", text_color, change.text, text_reset)?;
                    }
                }
            }
        }
    }

    Ok(String::from_utf8(output)?)
}

// Ask user for permission to execute a tool


// Render Markdown text with colors and formatting for terminal output



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
}

impl MarkdownStream {
    pub(crate) fn new() -> Self {
        MarkdownStream {
            buffer: String::new(),
            in_code_block: false,
            held_blanks: 0,
            streamed: 0,
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
            return String::new();
        }
        let line = std::mem::take(&mut self.buffer);
        self.rest_of_line(&line, newline)
    }

    fn line(&mut self, line: &str, newline: bool) -> String {
        let ending = if newline { "\n" } else { "" };
        let trimmed = line.trim();

        // Handle code blocks
        if trimmed.starts_with("```") {
            self.in_code_block = !self.in_code_block;
            return String::new();
        }

        if self.in_code_block {
            // Render code with dark grey background
            return format!(
                "{}{}  {}{}{}",
                SetBackgroundColor(Color::Grey),
                SetForegroundColor(Color::Black),
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
    // Every row of a code block carries the block's background, and the fence itself is
    // never printed at all.
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
    // The same for the hashes: "##" is a heading if a space comes next and prose if a
    // fourth hash does.
    if trimmed.len() <= 3 && trimmed.chars().all(|ch| ch == '#') {
        return None;
    }
    Some(true)
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
                let _ = write!(
                    rendered,
                    "{}{} ",
                    SetBackgroundColor(Color::Grey),
                    SetForegroundColor(Color::Black)
                );
                for j in (i + 1)..end {
                    rendered.push(chars[j]);
                }
                let _ = write!(rendered, " {}", ResetColor);
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
                        let _ = write!(rendered, "{} (", SetForegroundColor(Color::DarkGrey));
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

// Render text (auto-detect Markdown and render accordingly)

// Case-insensitive substring search over ASCII, working in bytes so a non-ASCII path cannot
// desync indices the way lowercasing the whole string would.
fn find_ascii_ci(haystack: &str, needle: &str, from: usize) -> Option<usize> {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    (from..=h.len() - n.len())
        .find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n) && haystack.is_char_boundary(i))
}

// Drop the working-directory prefix wherever it appears, so a log line is readable at a
// glance. Applied to tool results as well as call summaries: a result that echoes the full
// path back prints the long form directly under the short one, which is just noise.
//
// Only the display changes. What the model receives keeps its absolute paths, which are
// unambiguous and are what it passed in.
fn strip_project_prefix(text: &str) -> String {
    let Ok(cwd) = std::env::current_dir() else {
        return text.to_string();
    };
    let cwd = cwd.display().to_string();
    if cwd.is_empty() {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut at = 0usize;

    while let Some(start) = find_ascii_ci(text, &cwd, at) {
        let end = start + cwd.len();
        let next = text[end..].chars().next();

        // Only a real boundary counts, or a sibling directory whose name merely starts with
        // the same characters ("nginx-rs1" vs "nginx-rs12") would be mangled.
        let replacement = match next {
            Some('\\') | Some('/') => Some(""),
            None => Some("."),
            Some(c) if !c.is_alphanumeric() && c != '-' && c != '_' && c != '.' => Some("."),
            _ => None,
        };

        let Some(replacement) = replacement else {
            out.push_str(&text[at..end]);
            at = end;
            continue;
        };

        out.push_str(&text[at..start]);
        out.push_str(replacement);
        at = if matches!(next, Some('\\') | Some('/')) {
            end + 1
        } else {
            end
        };
    }

    out.push_str(&text[at..]);
    out
}

fn short_path(path: &str) -> String {
    // A missing argument stays absent rather than becoming ".", which would read as the
    // project root and be actively misleading.
    if path.is_empty() {
        return String::new();
    }
    strip_project_prefix(path)
}

// The interesting argument of a tool call, in a few words.
pub(crate) fn tool_call_summary(tool_name: &str, arguments: &Value) -> String {
    let text = |key: &str| arguments[key].as_str().unwrap_or("").to_string();

    match tool_name {
        "read" | "write_file" | "edit" | "multi_edit" => short_path(&text("file_path")),
        "ls" => short_path(&text("path")),
        "glob_search" => {
            let base = text("path");
            if base.is_empty() {
                text("pattern")
            } else {
                format!("{} in {}", text("pattern"), short_path(&base))
            }
        }
        "grep" => {
            let glob = text("glob");
            if glob.is_empty() {
                text("pattern")
            } else {
                format!("{}  ({})", text("pattern"), glob)
            }
        }
        "run_command" | "run_background" => strip_project_prefix(&text("command")),
        "web_fetch" | "web_get" => text("url"),
        "web_search" | "crate_search" => text("query"),
        "crate_info" | "crate_api" => text("name"),
        "background_output" | "background_stop" => match arguments["id"].as_u64() {
            Some(id) => format!("job {}", id),
            None => "all".to_string(),
        },
        "todo_write" => match arguments["todos"].as_array() {
            Some(todos) => {
                let done = todos
                    .iter()
                    .filter(|t| t["status"] == "done")
                    .count();
                format!("{}/{} done", done, todos.len())
            }
            None => String::new(),
        },
        "calculator" => format!(
            "{} {} {}",
            arguments["a"], text("operation"), arguments["b"]
        ),
        _ => String::new(),
    }
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

    let _ = write!(line, "{}  → {}", SetForegroundColor(Color::DarkGrey), tool_name);
    if !summary.is_empty() {
        let summary: String = summary.split_whitespace().collect::<Vec<_>>().join(" ");
        let clipped = crate::util::truncate_at_boundary(&summary, 110);
        let _ = write!(
            line,
            "  {}{}",
            clipped,
            if clipped.len() < summary.len() { "…" } else { "" }
        );
    }
    if let Some(note) = note.map(str::trim).filter(|n| !n.is_empty()) {
        // Clipped harder than the summary: this is a sentence of prose from the model, and
        // an unbounded one would push the call itself off the line it belongs on.
        let note: String = note.split_whitespace().collect::<Vec<_>>().join(" ");
        let clipped = crate::util::truncate_at_boundary(&note, 90);
        let _ = write!(
            line,
            "  (allowed by auto analyzer: {}{})",
            clipped,
            if clipped.len() < note.len() { "…" } else { "" }
        );
    }
    let _ = write!(line, "{}", ResetColor);
    line
}

// What a write is about to do, rendered as a diff.
//
// Lifted out of the permission prompt so a pre-approved write can show it as well. With
// "allow all" answered for write_file, every later file used to arrive as nothing but a
// path and a byte count — no way to see what had been put in it.
pub(crate) fn file_change_diff(tool_name: &str, arguments: &Value) -> Option<String> {
    match tool_name {
            "write_file" => {
                let file_path = arguments["file_path"].as_str().unwrap_or("");
                let new_content = arguments["content"].as_str().unwrap_or("");

                // Check if file exists and read old content
                if PathBuf::from(file_path).exists() {
                    if let Ok(old_content) = fs::read_to_string(file_path) {
                        generate_diff(file_path, &old_content, new_content).ok()
                    } else {
                        None
                    }
                } else {
                    // New file - show all as additions
                    generate_diff(file_path, "", new_content).ok()
                }
            }
            "edit" => {
                let file_path = arguments["file_path"].as_str().unwrap_or("");
                let old_string = arguments["old_string"].as_str().unwrap_or("");
                let new_string = arguments["new_string"].as_str().unwrap_or("");
                let replace_all = arguments["replace_all"].as_bool().unwrap_or(false);

                // Read current content and generate preview
                if let Ok(content) = fs::read_to_string(file_path) {
                    let new_content = if replace_all {
                        content.replace(old_string, new_string)
                    } else {
                        if let Some(pos) = content.find(old_string) {
                            let mut result = String::with_capacity(content.len());
                            result.push_str(&content[..pos]);
                            result.push_str(new_string);
                            result.push_str(&content[pos + old_string.len()..]);
                            result
                        } else {
                            content.clone()
                        }
                    };
                    generate_diff(file_path, &content, &new_content).ok()
                } else {
                    None
                }
            }
            "multi_edit" => {
                let file_path = arguments["file_path"].as_str().unwrap_or("");

                // Same order the tool applies them in, so the preview shows
                // what will actually land rather than each edit in isolation.
                if let Ok(content) = fs::read_to_string(file_path) {
                    let mut working = content.clone();
                    if let Some(edits) = arguments["edits"].as_array() {
                        for edit in edits {
                            let old_string = edit["old_string"].as_str().unwrap_or("");
                            let new_string = edit["new_string"].as_str().unwrap_or("");
                            if old_string.is_empty() {
                                continue;
                            }
                            working = if edit["replace_all"].as_bool().unwrap_or(false) {
                                working.replace(old_string, new_string)
                            } else {
                                working.replacen(old_string, new_string, 1)
                            };
                        }
                    }
                    generate_diff(file_path, &content, &working).ok()
                } else {
                    None
                }
            }
            _ => None
    }
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
            SetForegroundColor(Color::DarkGrey),
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

    let first = strip_project_prefix(first);
    let first = first.as_str();
    let clipped = crate::util::truncate_at_boundary(first, 110);
    Some(format!(
        "{}    {}{}{}",
        SetForegroundColor(Color::DarkGrey),
        clipped,
        if clipped.len() < first.len() { "…" } else { "" },
        ResetColor
    ))
}

#[cfg(test)]
mod diff_tests {
    use super::*;

    #[test]
    fn a_dark_terminal_gets_the_diff_in_white() {
        // The flag is global, and this is the only test that reads or writes it. It is put
        // back afterwards so the rest of the suite sees what it always did.
        let white = SetForegroundColor(Color::White).to_string();

        set_dark(false);
        let light = generate_diff("a.txt", "keep\none\n", "keep\ntwo\n").unwrap();
        set_dark(true);
        let dark = generate_diff("a.txt", "keep\none\n", "keep\ntwo\n").unwrap();
        set_dark(false);

        // Light leaves the text in the terminal's own color, exactly as it always has.
        assert!(!light.contains(&white), "got {:?}", light);
        // Dark colors all three kinds of line: the one removed, the one added, and the
        // context around them.
        assert_eq!(dark.matches(&white).count(), 3, "got {:?}", dark);
        // And the color is the only difference: same lines, same numbers, same markers.
        let plain = |diff: &str| {
            diff.replace(&white, "")
                .replace(&ResetColor.to_string(), "")
        };
        assert_eq!(plain(&dark), plain(&light));
    }
}

#[cfg(test)]
mod log_tests {
    use super::{short_path, tool_call_summary};
    use serde_json::json;

    #[test]
    fn tool_summaries_name_the_thing_that_matters() {
        let s = |name: &str, args: serde_json::Value| tool_call_summary(name, &args);

        // The point is spotting a repeat, so the identifying argument has to be there.
        assert_eq!(
            s("run_command", json!({"command": "cargo build"})),
            "cargo build"
        );
        assert_eq!(
            s("grep", json!({"pattern": "fn main", "glob": "*.rs"})),
            "fn main  (*.rs)"
        );
        assert_eq!(s("grep", json!({"pattern": "fn main"})), "fn main");
        assert_eq!(s("web_search", json!({"query": "axum routing"})), "axum routing");
        assert_eq!(s("background_stop", json!({"id": 3})), "job 3");
        assert_eq!(s("background_stop", json!({})), "all");
        assert_eq!(
            s("todo_write", json!({"todos": [
                {"task": "a", "status": "done"}, {"task": "b", "status": "pending"}]})),
            "1/2 done"
        );

        // A tool with nothing worth showing degrades to just its name.
        assert_eq!(s("get_date", json!({})), "");
        // Missing arguments must not panic.
        assert_eq!(s("run_command", json!({})), "");
        assert_eq!(s("read", json!({})), "");
    }

    #[test]
    fn paths_are_shown_relative_to_the_project() {
        let cwd = std::env::current_dir().unwrap();

        // A long absolute path is what makes a log line unreadable.
        let inside = cwd.join("src").join("main.rs");
        let shown = short_path(&inside.display().to_string());
        assert!(!shown.contains(&cwd.display().to_string()), "got: {}", shown);
        assert!(shown.ends_with("main.rs"), "got: {}", shown);
        assert!(shown.starts_with("src"), "got: {}", shown);

        // Outside the project it stays absolute, because that is the notable part.
        let outside = std::env::temp_dir().join("elsewhere.rs");
        assert_eq!(short_path(&outside.display().to_string()), outside.display().to_string());

        assert_eq!(short_path(&cwd.display().to_string()), ".");
        // A missing argument must not turn into the project root.
        assert_eq!(short_path(""), "");
    }

    #[test]
    fn a_write_can_be_previewed_without_a_permission_prompt() {
        use super::file_change_diff;

        let dir = std::env::temp_dir().join(format!("haico-diff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("test3.ps1");
        let path = file.display().to_string();

        // A brand-new file: every line is an addition, which is exactly what you need to
        // see when "allow all" means no prompt will show it to you.
        let _ = std::fs::remove_file(&file);
        let diff = file_change_diff("write_file", &json!({
            "file_path": path, "content": "line one\nline two\n"
        }))
        .expect("a new file still has a diff");
        assert!(diff.contains("line one"), "got: {}", diff);
        assert!(diff.contains("line two"), "got: {}", diff);

        // An overwrite shows what actually changed, not the whole file.
        std::fs::write(&file, "keep\nold value\nkeep too\n").unwrap();
        let diff = file_change_diff("write_file", &json!({
            "file_path": path, "content": "keep\nnew value\nkeep too\n"
        }))
        .unwrap();
        assert!(diff.contains("old value"), "got: {}", diff);
        assert!(diff.contains("new value"), "got: {}", diff);

        // edit and multi_edit preview through the same path.
        let diff = file_change_diff("edit", &json!({
            "file_path": path, "old_string": "old value", "new_string": "replaced"
        }))
        .unwrap();
        assert!(diff.contains("replaced"), "got: {}", diff);

        let diff = file_change_diff("multi_edit", &json!({
            "file_path": path,
            "edits": [{"old_string": "keep", "new_string": "kept", "replace_all": true}]
        }))
        .unwrap();
        assert!(diff.contains("kept"), "got: {}", diff);

        // Tools that do not touch files have nothing to preview.
        assert!(file_change_diff("run_command", &json!({"command": "ls"})).is_none());
        assert!(file_change_diff("read", &json!({"file_path": path})).is_none());
        // A path that cannot be read must not panic.
        assert!(file_change_diff("edit", &json!({"file_path": "no/such/file"})).is_none());

        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn result_lines_lose_the_project_prefix_too() {
        use super::strip_project_prefix;

        let cwd = std::env::current_dir().unwrap().display().to_string();

        // The reported case: the call line said "write_file test.ps1" and the result line
        // underneath repeated the whole absolute path.
        let echoed = format!("Successfully wrote 430 bytes to {}\\test.ps1", cwd);
        assert_eq!(
            strip_project_prefix(&echoed),
            "Successfully wrote 430 bytes to test.ps1"
        );

        // More than one occurrence in a line, and forward slashes too.
        let two = format!("copied {}/a.rs to {}/b.rs", cwd, cwd);
        assert_eq!(strip_project_prefix(&two), "copied a.rs to b.rs");

        // Windows casing varies between what the model passes and the process cwd.
        if cfg!(windows) {
            let lowered = format!("wrote {}\\x.rs", cwd.to_lowercase());
            assert_eq!(strip_project_prefix(&lowered), "wrote x.rs");
        }

        // A sibling directory that merely starts with the same characters must survive
        // intact, or the log would point at the wrong project.
        let sibling = format!("{}2\\other.rs", cwd);
        assert_eq!(strip_project_prefix(&sibling), sibling);

        // The bare directory becomes ".", and unrelated text is untouched.
        assert_eq!(strip_project_prefix(&cwd), ".");
        assert_eq!(strip_project_prefix("nothing to see"), "nothing to see");
    }
}


// Interactive model selection using arrow keys

pub(crate) async fn select_model_interactive(keys: &mut Keys, models: Vec<String>) -> Result<String> {
    if models.is_empty() {
        return Err(anyhow::anyhow!("No models available"));
    }

    let mut selected_index = 0;

    loop {
        // Clear screen and move cursor to top
        out!("{}{}", cursor::MoveTo(0, 0), Clear(ClearType::All));

        outln!("Select a model (use Up/Down arrows, Enter to confirm):\n");

        // Display models
        for (index, model) in models.iter().enumerate() {
            if index == selected_index {
                outln!("> {} <", model);
            } else {
                outln!("  {}  ", model);
            }
        }

        crate::out::flush();

        // Read key event
        let Some(Event::Key(key_event)) = keys.next().await else {
            return Err(anyhow::anyhow!("no interactive terminal to pick a model"));
        };

        // Only process key press events, ignore key release
        if key_event.kind != KeyEventKind::Press {
            continue;
        }

        match key_event.code {
            KeyCode::Up => {
                if selected_index > 0 {
                    selected_index -= 1;
                }
            }
            KeyCode::Down => {
                if selected_index < models.len() - 1 {
                    selected_index += 1;
                }
            }
            KeyCode::Enter => {
                // Selection confirmed
                break;
            }
            KeyCode::Esc => {
                return Err(anyhow::anyhow!("Model selection cancelled"));
            }
            _ => {
                // Ignore other keys
            }
        }
    }

    // Clear screen one final time and show selection
    out!("{}{}", cursor::MoveTo(0, 0), Clear(ClearType::All));
    outln!("Selected model: {}\n", models[selected_index]);

    Ok(models[selected_index].clone())
}

// Configuration structure


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
        // span, bold, an indented continuation, a fence, and a blank line between
        // paragraphs.
        const TEXT: &str = "# Title\nA [link](http://x/y), `code`, **bold** here.\n\n- a list item\n  indented continuation\n```rust\nlet x = 1;\n```\ndone\n";
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
        let mut stream = MarkdownStream::new();
        let mut rendered = stream.push("```rust\nlet x = 1;\n");
        rendered.push_str(&stream.push("```\nafter\n"));

        // The fenced line got the code background; the line after it did not.
        assert!(rendered.contains("let x = 1;"));
        let code = rendered.find("let x = 1;").unwrap();
        let after = rendered.find("after").unwrap();
        assert!(rendered[..code].contains("48;5;7"), "no code background: {:?}", rendered);
        assert!(!rendered[code..after].ends_with("48;5;7"));
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
