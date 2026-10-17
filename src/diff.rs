//! What a file-writing call is about to change, as structure rather than as text.
//!
//! The core needs a diff before a call runs — under the call line, in the permission
//! prompt, in the evidence the auto-mode reviewer reads — and each of those shows it
//! differently. So what crosses the seam is this: lines that know whether they were
//! removed, added or kept, and the number they have in the file. The terminal colors
//! them (`render::diff_text`), the window draws them, the reviewer reads `plain`.
//!
//! It used to be the terminal's text. `generate_diff` lived in `render.rs` and wrote SGR
//! escapes into a `String`, the core imported it, and the window then stripped the
//! escapes again and worked out from the eighth character of each line which ones had
//! been added — the coupling the seam exists to prevent, run in both directions.

use crate::util::{dominant_line_ending, with_line_endings};
use serde_json::Value;
use similar::{ChangeTag, TextDiff};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Diff {
    pub(crate) file: String,
    /// Empty when the change changes nothing.
    pub(crate) rows: Vec<DiffRow>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LineKind {
    Removed,
    Added,
    Context,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DiffRow {
    /// One line, numbered as it is in the old file for a removal or a kept line and in
    /// the new one for an addition.
    Line { kind: LineKind, number: usize, text: String },
    /// Unchanged lines left out between two regions of change.
    Gap,
}

impl LineKind {
    pub(crate) fn marker(self) -> char {
        match self {
            LineKind::Removed => '-',
            LineKind::Added => '+',
            LineKind::Context => ' ',
        }
    }
}

impl Diff {
    /// The diff as text with no styling, for a reader that is not a screen: the
    /// auto-mode reviewer, which is a model.
    pub(crate) fn plain(&self) -> String {
        let mut out = format!("File: {}\n", self.file);
        if self.rows.is_empty() {
            out.push_str("No changes\n");
        }
        for row in &self.rows {
            match row {
                DiffRow::Gap => out.push_str("       ...\n"),
                DiffRow::Line { kind, number, text } => {
                    out.push_str(&format!("{:6} {} {}\n", number, kind.marker(), text));
                }
            }
        }
        out
    }
}

// How many unchanged lines are kept around each change.
const CONTEXT_LINES: usize = 5;

pub(crate) fn generate_diff(file_path: &str, old_content: &str, new_content: &str) -> Diff {
    let diff = TextDiff::from_lines(old_content, new_content);

    // First pass: every line, numbered.
    let mut lines: Vec<(LineKind, usize, String)> = Vec::new();
    let mut old_line = 1;
    let mut new_line = 1;
    for change in diff.iter_all_changes() {
        let text = change.as_str().unwrap_or("");
        let text = text.trim_end_matches('\n').trim_end_matches('\r').to_string();
        match change.tag() {
            ChangeTag::Delete => {
                lines.push((LineKind::Removed, old_line, text));
                old_line += 1;
            }
            ChangeTag::Insert => {
                lines.push((LineKind::Added, new_line, text));
                new_line += 1;
            }
            ChangeTag::Equal => {
                lines.push((LineKind::Context, old_line, text));
                old_line += 1;
                new_line += 1;
            }
        }
    }

    let changed: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, (kind, _, _))| *kind != LineKind::Context)
        .map(|(index, _)| index)
        .collect();
    let Some(&first) = changed.first() else {
        return Diff { file: file_path.to_string(), rows: Vec::new() };
    };

    // Second pass: group the changes into regions with context, merging regions that
    // overlap or touch.
    let last = lines.len() - 1;
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut start = first.saturating_sub(CONTEXT_LINES);
    let mut end = first + CONTEXT_LINES;
    for &index in &changed[1..] {
        let region_start = index.saturating_sub(CONTEXT_LINES);
        if region_start <= end + 1 {
            end = index + CONTEXT_LINES;
        } else {
            regions.push((start, end.min(last)));
            start = region_start;
            end = index + CONTEXT_LINES;
        }
    }
    regions.push((start, end.min(last)));

    let mut rows = Vec::new();
    for (region, &(start, end)) in regions.iter().enumerate() {
        if region > 0 {
            rows.push(DiffRow::Gap);
        }
        for (kind, number, text) in &lines[start..=end] {
            rows.push(DiffRow::Line { kind: *kind, number: *number, text: text.clone() });
        }
    }
    Diff { file: file_path.to_string(), rows }
}

// What a write is about to do, as a diff.
//
// Lifted out of the permission prompt so a pre-approved write can show it as well. With
// "allow all" answered for write_file, every later file used to arrive as nothing but a
// path and a byte count — no way to see what had been put in it.
pub(crate) fn file_change_diff(tool_name: &str, arguments: &Value) -> Option<Diff> {
    match tool_name {
        "write_file" => {
            let file_path = arguments["file_path"].as_str().unwrap_or("");
            let new_content = arguments["content"].as_str().unwrap_or("");

            if !PathBuf::from(file_path).exists() {
                // A new file: every line an addition.
                return Some(generate_diff(file_path, "", new_content));
            }
            let old_content = fs::read_to_string(file_path).ok()?;
            // As write_file will: the model's content is LF and the file may not be, and
            // a preview that did not follow it would report every line as changed on a
            // CRLF file.
            let ending = dominant_line_ending(&old_content);
            Some(generate_diff(
                file_path,
                &with_line_endings(&old_content, ending),
                &with_line_endings(new_content, ending),
            ))
        }
        "edit" => {
            let file_path = arguments["file_path"].as_str().unwrap_or("");
            let old_string = arguments["old_string"].as_str().unwrap_or("");
            let new_string = arguments["new_string"].as_str().unwrap_or("");
            let replace_all = arguments["replace_all"].as_bool().unwrap_or(false);

            let on_disk = fs::read_to_string(file_path).ok()?;
            // The same translation `edit` does, for the same reason and so that the two
            // agree. Without it the preview searched for an LF old_string in a CRLF file,
            // found nothing, and drew "No changes" over an edit that was about to fail —
            // the user approved a no-op and got an error.
            //
            // Both sides of the diff are normalized, so a mixed file's repaired endings are
            // not shown as a wall of changed lines around the one change the user is
            // actually being asked about.
            let ending = dominant_line_ending(&on_disk);
            let content = with_line_endings(&on_disk, ending);
            let old_string = with_line_endings(old_string, ending);
            let new_string = with_line_endings(new_string, ending);

            let new_content = if replace_all {
                content.replace(old_string.as_ref(), &new_string)
            } else {
                content.replacen(old_string.as_ref(), &new_string, 1)
            };
            Some(generate_diff(file_path, &content, &new_content))
        }
        "multi_edit" => {
            let file_path = arguments["file_path"].as_str().unwrap_or("");

            // Same order the tool applies them in, so the preview shows what will actually
            // land rather than each edit in isolation.
            let on_disk = fs::read_to_string(file_path).ok()?;
            let ending = dominant_line_ending(&on_disk);
            let content = with_line_endings(&on_disk, ending);
            let mut working = content.to_string();
            for edit in arguments["edits"].as_array().into_iter().flatten() {
                let old_string = edit["old_string"].as_str().unwrap_or("");
                let new_string = edit["new_string"].as_str().unwrap_or("");
                if old_string.is_empty() {
                    continue;
                }
                let old_string = with_line_endings(old_string, ending);
                let new_string = with_line_endings(new_string, ending);
                working = if edit["replace_all"].as_bool().unwrap_or(false) {
                    working.replace(old_string.as_ref(), &new_string)
                } else {
                    working.replacen(old_string.as_ref(), &new_string, 1)
                };
            }
            Some(generate_diff(file_path, &content, &working))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_diff_knows_which_lines_it_removed_and_added() {
        let diff = generate_diff("a.txt", "keep\none\n", "keep\ntwo\n");
        assert_eq!(diff.file, "a.txt");
        assert_eq!(
            diff.rows,
            vec![
                DiffRow::Line { kind: LineKind::Context, number: 1, text: "keep".into() },
                DiffRow::Line { kind: LineKind::Removed, number: 2, text: "one".into() },
                DiffRow::Line { kind: LineKind::Added, number: 2, text: "two".into() },
            ]
        );
        // The text the reviewer reads, laid out as the terminal lays it out.
        assert_eq!(diff.plain(), "File: a.txt\n     1   keep\n     2 - one\n     2 + two\n");
    }

    #[test]
    fn distant_changes_are_two_regions_with_a_gap_between() {
        let old: String = (1..=30).map(|n| format!("{n}\n")).collect();
        let new = old.replacen("2\n", "two\n", 1).replacen("28\n", "twenty-eight\n", 1);
        let diff = generate_diff("n.txt", &old, &new);
        assert_eq!(diff.rows.iter().filter(|row| **row == DiffRow::Gap).count(), 1);
        // The context stops five lines either side of each change.
        assert!(matches!(diff.rows.first(), Some(DiffRow::Line { number: 1, .. })));
        assert!(matches!(diff.rows.last(), Some(DiffRow::Line { number: 30, .. })));
    }

    #[test]
    fn a_change_that_changes_nothing_says_so() {
        let diff = generate_diff("same.txt", "a\n", "a\n");
        assert!(diff.rows.is_empty());
        assert_eq!(diff.plain(), "File: same.txt\nNo changes\n");
    }

    // A call's preview as the text the reviewer reads, which is what these assert on.
    fn preview(tool_name: &str, arguments: &Value) -> Option<String> {
        file_change_diff(tool_name, arguments).map(|diff| diff.plain())
    }
    use serde_json::json;

    #[test]
    fn a_write_can_be_previewed_without_a_permission_prompt() {

        let dir = std::env::temp_dir().join(format!("haico-diff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("test3.ps1");
        let path = file.display().to_string();

        // A brand-new file: every line is an addition, which is exactly what you need to
        // see when "allow all" means no prompt will show it to you.
        let _ = std::fs::remove_file(&file);
        let diff = preview("write_file", &json!({
            "file_path": path, "content": "line one\nline two\n"
        }))
        .expect("a new file still has a diff");
        assert!(diff.contains("line one"), "got: {}", diff);
        assert!(diff.contains("line two"), "got: {}", diff);

        // An overwrite shows what actually changed, not the whole file.
        std::fs::write(&file, "keep\nold value\nkeep too\n").unwrap();
        let diff = preview("write_file", &json!({
            "file_path": path, "content": "keep\nnew value\nkeep too\n"
        }))
        .unwrap();
        assert!(diff.contains("old value"), "got: {}", diff);
        assert!(diff.contains("new value"), "got: {}", diff);

        // edit and multi_edit preview through the same path.
        let diff = preview("edit", &json!({
            "file_path": path, "old_string": "old value", "new_string": "replaced"
        }))
        .unwrap();
        assert!(diff.contains("replaced"), "got: {}", diff);

        let diff = preview("multi_edit", &json!({
            "file_path": path,
            "edits": [{"old_string": "keep", "new_string": "kept", "replace_all": true}]
        }))
        .unwrap();
        assert!(diff.contains("kept"), "got: {}", diff);

        // Tools that do not touch files have nothing to preview.
        assert!(preview("run_command", &json!({"command": "ls"})).is_none());
        assert!(preview("read", &json!({"file_path": path})).is_none());
        // A path that cannot be read must not panic.
        assert!(preview("edit", &json!({"file_path": "no/such/file"})).is_none());

        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn a_crlf_file_previews_the_change_rather_than_no_changes() {

        let dir = std::env::temp_dir().join(format!("haico-crlf-diff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("crlf.rs");
        let path = file.display().to_string();
        std::fs::write(&file, "fn a() {\r\n    let x = 1;\r\n}\r\n").unwrap();

        // The preview used to search for the model's LF old_string in these CRLF bytes,
        // find nothing, and fall back to the unchanged file — so it drew "No changes" over
        // an edit that was about to fail, and the user approved a no-op.
        let diff = preview("edit", &json!({
            "file_path": path,
            "old_string": "fn a() {\n    let x = 1;\n}",
            "new_string": "fn a() {\n    let x = 2;\n}",
        }))
        .unwrap();
        assert!(!diff.contains("No changes"), "got: {}", diff);
        assert!(diff.contains("let x = 2;"), "got: {}", diff);

        // A whole-file rewrite is shown as the one line it changes, not as every line
        // changing because the model's content is LF and the file is not.
        let diff = preview("write_file", &json!({
            "file_path": path, "content": "fn a() {\n    let x = 3;\n}\n"
        }))
        .unwrap();
        assert!(diff.contains("let x = 3;"), "got: {}", diff);
        // One line out, one line in — the rest of the file is context. Without the
        // normalization every line came back changed, because the whole file was being
        // converted to LF underneath the one edit.
        let changed = diff
            .lines()
            .filter(|line| matches!(line.as_bytes().get(7), Some(b'-') | Some(b'+')))
            .count();
        assert_eq!(changed, 2, "whole file redrawn: {}", diff);

        let diff = preview("multi_edit", &json!({
            "file_path": path,
            "edits": [{"old_string": "fn a() {\n    let x = 1;", "new_string": "fn a() {\n    let y = 1;"}]
        }))
        .unwrap();
        assert!(diff.contains("let y = 1;"), "got: {}", diff);

        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }

}
