//! Terminal rendering: markdown, diffs, and the model picker.

use anyhow::Result;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    style::{Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor},
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
};
use serde_json::Value;
use similar::{ChangeTag, TextDiff};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

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
                        write!(&mut output, "- {}", change.text)?;
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
                        write!(&mut output, "+ {}", change.text)?;
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
                        writeln!(&mut output, "  {}", change.text)?;
                    }
                }
            }
        }
    }

    Ok(String::from_utf8(output)?)
}

// Ask user for permission to execute a tool

pub(crate) fn is_markdown(text: &str) -> bool {
    let mut markdown_score = 0;
    let lines: Vec<&str> = text.lines().collect();

    if lines.is_empty() {
        return false;
    }

    for line in &lines {
        let trimmed = line.trim();

        // Check for headers (lines starting with #)
        if trimmed.starts_with('#') && trimmed.len() > 1 {
            markdown_score += 2;
        }

        // Check for unordered list items (lines starting with -)
        if trimmed.starts_with('-') && trimmed.len() > 1 {
            markdown_score += 1;
        }

        // Check for unordered list items (lines starting with *)
        if trimmed.starts_with('*') && trimmed.len() > 1 && !trimmed.starts_with("**") {
            markdown_score += 1;
        }

        // Check for bold text markers (**text**)
        if line.contains("**") {
            // Count pairs of **
            let count = line.matches("**").count();
            if count >= 2 {
                markdown_score += 1;
            }
        }

        // Check for italic markers (*text* or _text_) - only if they appear in pairs and not part of snake_case
        // Count underscores only if surrounded by non-word chars to avoid snake_case false positives
        let underscore_count = line.matches('_').count();
        if underscore_count >= 2 {
            // Check if they look like markdown (not snake_case)
            // Simple heuristic: if there's whitespace around them, they're likely markdown
            if line.contains(" _") || line.contains("_ ") {
                markdown_score += 1;
            }
        }

        // Check for code blocks (```)
        if trimmed.starts_with("```") {
            markdown_score += 3;
        }

        // Check for inline code (`text`)
        if line.contains('`') && !trimmed.starts_with("```") {
            markdown_score += 1;
        }

        // Check for links ([text](url))
        if line.contains("[") && line.contains("](") {
            markdown_score += 2;
        }

        // Check for ordered list (1. text)
        if trimmed.len() > 2 {
            let chars: Vec<char> = trimmed.chars().collect();
            if chars[0].is_numeric() && chars.len() > 1 && chars[1] == '.' {
                markdown_score += 1;
            }
        }
    }

    // Determine threshold: if we have enough Markdown indicators relative to text length
    // Consider it Markdown if score is at least 3, or if score/lines ratio is > 0.2
    let ratio = markdown_score as f64 / lines.len() as f64;
    markdown_score >= 3 || ratio > 0.2
}

// Render Markdown text with colors and formatting for terminal output

pub(crate) fn render_markdown(text: &str) -> Result<()> {
    let mut in_code_block = false;
    let mut code_lang = String::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Handle code blocks
        if trimmed.starts_with("```") {
            if in_code_block {
                // End of code block
                in_code_block = false;
                code_lang.clear();
            } else {
                // Start of code block
                in_code_block = true;
                code_lang = trimmed.trim_start_matches("```").to_string();
            }
            continue;
        }

        if in_code_block {
            // Render code with dark grey background
            execute!(io::stdout(), SetBackgroundColor(Color::Grey), SetForegroundColor(Color::Black))?;
            print!("  {}", line);
            execute!(io::stdout(), ResetColor)?;
            println!();
            continue;
        }

        // Handle headers
        if trimmed.starts_with("# ") {
            execute!(io::stdout(), SetForegroundColor(Color::Cyan))?;
            println!("\n{}", trimmed.trim_start_matches("# "));
            execute!(io::stdout(), ResetColor)?;
            continue;
        } else if trimmed.starts_with("## ") {
            execute!(io::stdout(), SetForegroundColor(Color::Cyan))?;
            println!("\n{}", trimmed.trim_start_matches("## "));
            execute!(io::stdout(), ResetColor)?;
            continue;
        } else if trimmed.starts_with("### ") {
            execute!(io::stdout(), SetForegroundColor(Color::Blue))?;
            println!("\n{}", trimmed.trim_start_matches("### "));
            execute!(io::stdout(), ResetColor)?;
            continue;
        }

        // Process inline formatting
        render_inline_markdown(line)?;
    }

    Ok(())
}

// Render a line with inline Markdown formatting (bold, italic, code, links)

pub(crate) fn render_inline_markdown(line: &str) -> Result<()> {
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check for inline code (`code`)
        if chars[i] == '`' {
            if let Some(end) = find_closing('`', &chars, i + 1) {
                execute!(io::stdout(), SetBackgroundColor(Color::Grey), SetForegroundColor(Color::Black))?;
                print!(" ");
                for j in (i + 1)..end {
                    print!("{}", chars[j]);
                }
                print!(" ");
                execute!(io::stdout(), ResetColor)?;
                i = end + 1;
                continue;
            }
        }

        // Check for bold (**text**)
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_closing("**", &chars, i + 2) {
                execute!(io::stdout(), SetAttribute(Attribute::Bold))?;
                for j in (i + 2)..end {
                    print!("{}", chars[j]);
                }
                execute!(io::stdout(), SetAttribute(Attribute::Reset))?;
                i = end + 2;
                continue;
            }
        }

        // Check for italic (*text* or _text_)
        if chars[i] == '*' || chars[i] == '_' {
            let marker = chars[i];
            if let Some(end) = find_closing(marker, &chars, i + 1) {
                // Make sure it's not part of **
                if marker == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
                    print!("{}", chars[i]);
                    i += 1;
                    continue;
                }
                execute!(io::stdout(), SetForegroundColor(Color::Green))?;
                for j in (i + 1)..end {
                    print!("{}", chars[j]);
                }
                execute!(io::stdout(), ResetColor)?;
                i = end + 1;
                continue;
            }
        }

        // Check for links ([text](url))
        if chars[i] == '[' {
            if let Some(close_bracket) = find_closing(']', &chars, i + 1) {
                if close_bracket + 1 < chars.len() && chars[close_bracket + 1] == '(' {
                    if let Some(close_paren) = find_closing(')', &chars, close_bracket + 2) {
                        // Render link text in blue
                        execute!(io::stdout(), SetForegroundColor(Color::Blue))?;
                        for j in (i + 1)..close_bracket {
                            print!("{}", chars[j]);
                        }
                        execute!(io::stdout(), ResetColor)?;

                        // Render URL in dark grey
                        execute!(io::stdout(), SetForegroundColor(Color::DarkGrey))?;
                        print!(" (");
                        for j in (close_bracket + 2)..close_paren {
                            print!("{}", chars[j]);
                        }
                        print!(")");
                        execute!(io::stdout(), ResetColor)?;

                        i = close_paren + 1;
                        continue;
                    }
                }
            }
        }

        // Regular character
        print!("{}", chars[i]);
        i += 1;
    }

    println!();
    Ok(())
}

// Find closing character from a starting position

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
fn tool_call_summary(tool_name: &str, arguments: &Value) -> String {
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
pub(crate) fn log_tool_call(tool_name: &str, arguments: &Value, note: Option<&str>) {
    let summary = tool_call_summary(tool_name, arguments);
    let mut out = io::stdout();

    let _ = execute!(out, SetForegroundColor(Color::DarkGrey));
    print!("  → {}", tool_name);
    if !summary.is_empty() {
        let summary: String = summary.split_whitespace().collect::<Vec<_>>().join(" ");
        let clipped = crate::util::truncate_at_boundary(&summary, 110);
        print!(
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
        print!(
            "  (allowed by auto analyzer: {}{})",
            clipped,
            if clipped.len() < note.len() { "…" } else { "" }
        );
    }
    let _ = execute!(out, ResetColor);
    println!();
    let _ = out.flush();
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
pub(crate) fn print_change_preview(diff: &str) {
    const MAX_LINES: usize = 24;

    // The first line is "File: <path>", already shown by the call line above.
    let body: Vec<&str> = diff.lines().skip(1).collect();
    if body.is_empty() {
        return;
    }

    for line in body.iter().take(MAX_LINES) {
        println!("    {}", line);
    }

    if body.len() > MAX_LINES {
        let mut out = io::stdout();
        let _ = execute!(out, SetForegroundColor(Color::DarkGrey));
        print!("    … {} more line(s)", body.len() - MAX_LINES);
        let _ = execute!(out, ResetColor);
        println!();
    }
    let _ = io::stdout().flush();
}

// One line of what a tool gave back. The repeated identical result is usually the clearest
// evidence that a model is stuck, so it belongs in the log next to the call.
pub(crate) fn log_tool_result(text: &str) {
    let first = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    if first.is_empty() {
        return;
    }

    let first = strip_project_prefix(first);
    let first = first.as_str();
    let clipped = crate::util::truncate_at_boundary(first, 110);
    let mut out = io::stdout();
    let _ = execute!(out, SetForegroundColor(Color::DarkGrey));
    print!(
        "    {}{}",
        clipped,
        if clipped.len() < first.len() { "…" } else { "" }
    );
    let _ = execute!(out, ResetColor);
    println!();
    let _ = out.flush();
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

pub(crate) fn render_text(text: &str) -> Result<()> {
    if is_markdown(text) {
        render_markdown(text)
    } else {
        println!("{}", text);
        Ok(())
    }
}

// Interactive model selection using arrow keys

pub(crate) fn select_model_interactive(models: Vec<String>) -> Result<String> {
    if models.is_empty() {
        return Err(anyhow::anyhow!("No models available"));
    }

    let mut selected_index = 0;

    println!("Select a model (use Up/Down arrows, Enter to confirm):\n");

    // Enable raw mode for key events
    enable_raw_mode()?;

    loop {
        // Clear screen and move cursor to top
        execute!(io::stdout(), cursor::MoveTo(0, 0), Clear(ClearType::All))?;

        println!("Select a model (use Up/Down arrows, Enter to confirm):\n");

        // Display models
        for (index, model) in models.iter().enumerate() {
            if index == selected_index {
                println!("> {} <", model);
            } else {
                println!("  {}  ", model);
            }
        }

        io::stdout().flush()?;

        // Read key event
        if let Event::Key(key_event) = event::read()? {
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
                    disable_raw_mode()?;
                    return Err(anyhow::anyhow!("Model selection cancelled"));
                }
                _ => {
                    // Ignore other keys
                }
            }
        }
    }

    disable_raw_mode()?;

    // Clear screen one final time and show selection
    execute!(io::stdout(), cursor::MoveTo(0, 0), Clear(ClearType::All))?;
    println!("Selected model: {}\n", models[selected_index]);

    Ok(models[selected_index].clone())
}

// Configuration structure
