//! Saving, resuming and compacting a conversation.

use crate::config::compact_hint_threshold;
use crate::llm::LMStudioClient;
use crate::tools::TodoItem;
use crate::util::truncate_at_boundary;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
pub(crate) struct SavedSession {
    pub(crate) version: u32,
    pub(crate) cwd: String,
    pub(crate) model: String,
    pub(crate) saved_at: String,
    // Everything after the system prompt. The prompt itself is rebuilt on load so the date,
    // branch and git status are current rather than whatever they were last time.
    pub(crate) messages: Vec<Value>,
    #[serde(default)]
    pub(crate) todos: Vec<TodoItem>,
}


pub(crate) const SESSION_FORMAT_VERSION: u32 = 1;

// Where this directory's session is kept.
//
// Keyed by working directory, so each project resumes its own conversation. The directory
// name is included for legibility when poking around, and a hash of the full path
// disambiguates two projects that happen to share a final component.

pub(crate) fn session_file_path() -> Option<PathBuf> {
    session_file_path_for(&env::current_dir().ok()?)
}

// Split out from session_file_path so it can be exercised without changing the process's
// working directory, which every other test that walks "." would race against.

pub(crate) fn session_file_path_for(cwd: &std::path::Path) -> Option<PathBuf> {
    use std::hash::{Hash, Hasher};

    let home = env::var_os("USERPROFILE").or_else(|| env::var_os("HOME"))?;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cwd.hash(&mut hasher);
    let digest = hasher.finish();

    let label: String = cwd
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "root".to_string())
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .take(40)
        .collect();

    Some(
        PathBuf::from(home)
            .join(".haico")
            .join("sessions")
            .join(format!("{}-{:016x}.json", label, digest)),
    )
}

// Persist the conversation so closing the terminal does not lose it.

pub(crate) fn save_session(path: &std::path::Path, messages: &[Value], model: &str, todos: &[TodoItem]) {
    // Skip the system prompt: it is rebuilt on load.
    let Some(rest) = messages.get(1..) else { return };

    let session = SavedSession {
        version: SESSION_FORMAT_VERSION,
        cwd: env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        model: model.to_string(),
        saved_at: chrono::Local::now().to_rfc3339(),
        messages: rest.to_vec(),
        todos: todos.to_vec(),
    };

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string(&session) {
        let _ = fs::write(path, json);
    }
}


pub(crate) fn load_session(path: &std::path::Path) -> Result<SavedSession> {
    let raw = fs::read_to_string(path).context("no saved session for this directory")?;
    let session: SavedSession =
        serde_json::from_str(&raw).context("saved session could not be read")?;

    if session.version != SESSION_FORMAT_VERSION {
        bail!(
            "saved session is version {}, this build writes version {}",
            session.version,
            SESSION_FORMAT_VERSION
        );
    }

    Ok(session)
}

// Where the REPL keeps its input history between sessions.

pub(crate) fn history_file_path() -> Option<PathBuf> {
    let home = env::var_os("USERPROFILE").or_else(|| env::var_os("HOME"))?;
    Some(PathBuf::from(home).join(".haico_history"))
}


pub(crate) fn estimate_tokens(messages: &[Value]) -> usize {
    messages
        .iter()
        .map(|message| message.to_string().chars().count())
        .sum::<usize>()
        / 4
}

// Assumed context window when none is configured. Override with --context / HAICO_CONTEXT;
// a hard-coded guess would either nag constantly on a large window or warn too late on a
// small one.

pub(crate) fn report_context(messages: &[Value], context_tokens: usize) {
    let turns = messages.iter().filter(|m| m["role"] == "user").count();
    let used = estimate_tokens(messages);
    let percent = (used as f64 / context_tokens as f64) * 100.0;

    outln!(
        "{} messages, {} user turn(s), ~{} of {} tokens ({:.1}%, estimated).",
        messages.len(),
        turns,
        used,
        context_tokens,
        percent
    );

    if used > compact_hint_threshold(context_tokens) {
        outln!("Run /compact to replace the history with a summary.");
    }
    outln!();
}

// Flatten the conversation for a summarizer, dropping the system prompt (it is re-attached
// unchanged) and trimming tool output, which is the bulk of a long session and the part
// least worth preserving verbatim.

pub(crate) fn render_transcript(messages: &[Value]) -> String {
    let mut out = String::new();

    for message in messages.iter().skip(1) {
        let role = message["role"].as_str().unwrap_or("?");

        if let Some(calls) = message.get("tool_calls").and_then(|c| c.as_array()) {
            for call in calls {
                out.push_str(&format!(
                    "[called {}]\n",
                    call["function"]["name"].as_str().unwrap_or("?")
                ));
            }
        }

        let content = message["content"].as_str().unwrap_or("");
        if content.is_empty() {
            continue;
        }
        let content = if role == "tool" {
            truncate_at_boundary(content, 1500)
        } else {
            content
        };
        out.push_str(&format!("{}: {}\n\n", role, content));
    }

    truncate_at_boundary(&out, 60_000).to_string()
}

// Replace the conversation with a summary of itself, keeping the system prompt.
//
// Everything after the system message goes, which also guarantees no orphaned tool_call_id
// survives into the compacted history — the summary is plain prose, so the pairing the
// server checks cannot be broken by this.

pub(crate) async fn compact_history(client: &LMStudioClient, messages: &mut Vec<Value>) -> Result<String> {
    if messages.len() <= 2 {
        bail!("nothing to compact yet");
    }

    let before = estimate_tokens(messages);

    let summary = client
        .simple_completion(
            "You compress a coding session's transcript so work can continue without it. \
             Preserve: what the user asked for, decisions made and why, files and symbols \
             touched, what has been done, what is still outstanding, and any constraint or \
             correction the user gave. Drop tool output that no longer matters. Write it as \
             notes for whoever picks the task up next, not as a narrative summary.",
            &render_transcript(messages),
        )
        .await?;

    let summary = summary.trim();
    if summary.is_empty() {
        bail!("the model returned an empty summary; history left untouched");
    }

    let system = messages[0].clone();
    messages.clear();
    messages.push(system);
    messages.push(json!({
        "role": "user",
        "content": format!("Notes from the conversation so far:\n\n{}", summary)
    }));
    messages.push(json!({
        "role": "assistant",
        "content": "Understood — I have the context above and will continue from there."
    }));

    Ok(format!(
        "Compacted: ~{} -> ~{} estimated tokens.",
        before,
        estimate_tokens(messages)
    ))
}


#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use crate::config::*;
    use crate::llm::*;
    use crate::permissions::*;
    use crate::render::*;
    use crate::session::*;
    use crate::testutil::*;
    use crate::tools::*;
    use crate::util::*;
    use serde_json::json;

    #[test]
    fn transcript_keeps_the_shape_of_the_session_and_trims_tool_output() {
        use crate::render_transcript;

        let huge = "x".repeat(50_000);
        let messages = vec![
            json!({"role": "system", "content": "SYSTEM PROMPT SHOULD NOT APPEAR"}),
            json!({"role": "user", "content": "find the parser"}),
            json!({
                "role": "assistant",
                "content": "Looking.",
                "tool_calls": [{"id": "c1", "function": {"name": "grep", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "content": huge}),
            json!({"role": "assistant", "content": "It is in src/readable.rs."}),
        ];

        let transcript = render_transcript(&messages);

        // The system prompt is re-attached unchanged, so it must not be summarized too.
        assert!(!transcript.contains("SYSTEM PROMPT SHOULD NOT APPEAR"));
        assert!(transcript.contains("find the parser"));
        assert!(transcript.contains("[called grep]"));
        assert!(transcript.contains("src/readable.rs"));
        // The 50k tool result is the bulk of the session and gets cut down.
        assert!(transcript.len() < 10_000, "got {} bytes", transcript.len());
    }


    #[tokio::test]
    async fn compaction_replaces_history_without_orphaning_tool_calls() {
        use crate::{compact_history, estimate_tokens};

        let body = concat!(
            "{\"choices\":[{\"message\":{\"role\":\"assistant\",",
            "\"content\":\"Fixed the parser; tests still to run.\"}}]}"
        );
        let url = serve_once(body, "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        let mut messages = vec![
            json!({"role": "system", "content": "system prompt"}),
            json!({"role": "user", "content": "fix the parser"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "edit", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "content": "y".repeat(20_000)}),
            json!({"role": "assistant", "content": "Done."}),
        ];
        let before = estimate_tokens(&messages);

        let report = compact_history(&client, &mut messages).await.unwrap();
        assert!(report.contains("Compacted"), "got: {}", report);

        // System prompt survives verbatim; the summary carries the rest.
        assert_eq!(messages[0]["content"], "system prompt");
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("Fixed the parser"));

        // Nothing referencing a tool call is left, so no tool_call_id can be orphaned —
        // the failure the server would reject the next request for.
        assert!(messages.iter().all(|m| m.get("tool_calls").is_none()));
        assert!(messages.iter().all(|m| m["role"] != "tool"));
        assert!(estimate_tokens(&messages) < before / 2);
    }


    #[tokio::test]
    async fn compaction_refuses_rather_than_destroying_history() {
        use crate::compact_history;

        // Nothing worth compacting yet: the history must be left alone.
        let url = serve_once("{}", "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        let mut messages = vec![json!({"role": "system", "content": "s"})];
        assert!(compact_history(&client, &mut messages).await.is_err());
        assert_eq!(messages.len(), 1);

        // An empty summary must not wipe a real conversation either.
        let empty = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"  \"}}]}";
        let url = serve_once(empty, "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        let mut messages = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "important context"}),
            json!({"role": "assistant", "content": "noted"}),
        ];
        assert!(compact_history(&client, &mut messages).await.is_err());
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["content"], "important context");
    }


    #[test]
    fn sessions_round_trip_without_the_stale_system_prompt() {
        use crate::{load_session, save_session, TodoItem};

        let dir = scratch("anchor4", "").parent().unwrap().to_path_buf();
        let path = dir.join("session.json");
        let _ = std::fs::remove_file(&path);

        let messages = vec![
            json!({"role": "system", "content": "system prompt from LAST time"}),
            json!({"role": "user", "content": "fix the parser"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "edit", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "content": "done"}),
            json!({"role": "assistant", "content": "Fixed."}),
        ];
        let todos = vec![TodoItem {
            task: "run tests".to_string(),
            status: "pending".to_string(),
        }];

        save_session(&path, &messages, "some-model", &todos);
        let restored = load_session(&path).unwrap();

        // The system prompt is rebuilt on load, so the saved one must not come back — it
        // carries the date, branch and git status from whenever the session was saved.
        assert_eq!(restored.messages.len(), messages.len() - 1);
        assert!(!restored
            .messages
            .iter()
            .any(|m| m["role"] == "system"));
        assert_eq!(restored.messages[0]["content"], "fix the parser");
        assert_eq!(restored.model, "some-model");
        assert_eq!(restored.todos.len(), 1);
        assert_eq!(restored.todos[0].task, "run tests");

        // Tool calls survive paired with their replies; an orphaned tool_call_id would make
        // the first request after resuming fail server-side.
        let call_ids: Vec<&str> = restored
            .messages
            .iter()
            .filter_map(|m| m.get("tool_calls").and_then(|c| c.as_array()))
            .flatten()
            .filter_map(|c| c["id"].as_str())
            .collect();
        for id in call_ids {
            assert!(
                restored
                    .messages
                    .iter()
                    .any(|m| m["role"] == "tool" && m["tool_call_id"] == id),
                "tool_call {} came back with no reply",
                id
            );
        }

        // A future format is declined rather than misread as this one.
        let mut raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        raw["version"] = json!(crate::SESSION_FORMAT_VERSION + 1);
        std::fs::write(&path, raw.to_string()).unwrap();
        assert!(load_session(&path).is_err());

        // So is a file that is not a session at all.
        std::fs::write(&path, "not json").unwrap();
        assert!(load_session(&path).is_err());

        let _ = std::fs::remove_file(&path);
    }


    #[test]
    fn session_paths_are_per_directory() {
        use crate::session_file_path_for;
        use std::path::Path;

        // Two different working directories must not share one saved conversation.
        let a = session_file_path_for(Path::new("/projects/alpha"))
            .expect("a home directory is set in tests");
        let b = session_file_path_for(Path::new("/projects/beta")).unwrap();
        assert_ne!(a, b);

        // Same directory, same file, every time — otherwise --resume would never find it.
        assert_eq!(a, session_file_path_for(Path::new("/projects/alpha")).unwrap());

        // Projects sharing a final component still get separate files: the name is only for
        // legibility, the hash is what distinguishes them.
        let one = session_file_path_for(Path::new("/work/one/src")).unwrap();
        let two = session_file_path_for(Path::new("/work/two/src")).unwrap();
        assert_ne!(one, two);
        assert!(one.file_name().unwrap().to_string_lossy().starts_with("src-"));

        assert!(a.extension().is_some_and(|e| e == "json"));
    }

}
