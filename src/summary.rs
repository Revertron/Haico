//! What a tool call is about, in a few words: the argument worth showing and the paths
//! inside it made short. Wording without styling, so both frontends use it — the
//! terminal's call line colors it, the window sets it beside the tool's name.

use serde_json::Value;

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
pub(crate) fn strip_project_prefix(text: &str) -> String {
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

pub(crate) fn short_path(path: &str) -> String {
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
        "library" | "research" => text("question"),
        // The first line only: a task is written to be complete, so it runs long.
        "sub_agent" => text("task").lines().next().unwrap_or_default().to_string(),
        "crate_info" | "crate_api" => text("name"),
        "background_output" | "background_stop" => match arguments["id"].as_u64() {
            Some(id) => format!("job {}", id),
            None => "all".to_string(),
        },
        "todo_write" => match arguments["todos"].as_array() {
            Some(todos) => {
                // Counted through the normalizer because these are the raw arguments: the
                // tool has not run yet, so a model that wrote "done" or "complete" would
                // otherwise have its finished list summarized here as 0/4.
                let done = todos
                    .iter()
                    .filter(|t| {
                        t["status"]
                            .as_str()
                            .and_then(crate::tools::normalize_status)
                            == Some("completed")
                    })
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
        assert_eq!(s("library", json!({"question": "mio: register a listener?"})), "mio: register a listener?");
        assert_eq!(s("background_stop", json!({"id": 3})), "job 3");
        assert_eq!(s("background_stop", json!({})), "all");
        // Either word counts: the arguments have not been through the tool's normalizer.
        assert_eq!(
            s("todo_write", json!({"todos": [
                {"task": "a", "status": "done"}, {"task": "b", "status": "pending"}]})),
            "1/2 done"
        );
        assert_eq!(
            s("todo_write", json!({"todos": [
                {"task": "a", "status": "completed"}, {"task": "b", "status": "pending"}]})),
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


