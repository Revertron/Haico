//! Saving, resuming and compacting a conversation.

use crate::config::{compact_hint_threshold, Reasoning};
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

// Where the conversation proper starts. The system message is optional — a model can be
// configured without haico's prompt, and in a project with no instruction file that leaves
// none at all — so index 0 cannot be assumed to be it.

pub(crate) fn history_start(messages: &[Value]) -> usize {
    usize::from(messages.first().is_some_and(|m| m["role"] == "system"))
}

// Put the system message in place, replacing or removing whatever is there now. Used when
// /model switches to a model whose config asks for a different prompt.

pub(crate) fn set_system_message(messages: &mut Vec<Value>, content: Option<&str>) {
    let present = history_start(messages) == 1;
    match (present, content) {
        (true, Some(content)) => messages[0] = json!({"role": "system", "content": content}),
        (true, None) => {
            messages.remove(0);
        }
        (false, Some(content)) => {
            messages.insert(0, json!({"role": "system", "content": content}))
        }
        (false, None) => {}
    }
}

// Persist the conversation so closing the terminal does not lose it.

pub(crate) fn save_session(path: &std::path::Path, messages: &[Value], model: &str, todos: &[TodoItem]) {
    // Skip the system prompt: it is rebuilt on load.
    let Some(rest) = messages.get(history_start(messages)..) else { return };

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

pub(crate) fn report_context(
    messages: &[Value],
    context_tokens: usize,
    reasoning: &ReasoningStats,
) {
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

    // Only once there is something to report. A line of zeroes under every /context would
    // be noise for the default policy, which never compresses anything.
    if reasoning.turns > 0 {
        outln!(
            "Reasoning compressed on {} repl{}: ~{} -> ~{} tokens, saving ~{}.",
            reasoning.turns,
            if reasoning.turns == 1 { "y" } else { "ies" },
            reasoning.before,
            reasoning.after,
            reasoning.saved()
        );
    }

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

    for message in messages.iter().skip(history_start(messages)) {
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
    let start = history_start(messages);
    if messages.len() <= start + 1 {
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

    let system = messages.first().filter(|_| start == 1).cloned();
    messages.clear();
    if let Some(system) = system {
        messages.push(system);
    }
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

// The key `into_message` writes a chain of thought under, and the one the thinking-model
// chat templates read back. Only the one spelling is written here even though the stream
// may have arrived under either, because the assembler has already normalized it.

const REASONING_KEY: &str = "reasoning_content";

// What the compressor is told to produce.
//
// The voice matters more than it looks. With reasoning preserved, the template feeds this
// field back to the model as its own thinking, so a summary written as minutes about a
// thought ("The model considered X, then rejected it") is further out of distribution than
// the long thought it replaces. Asking for first-person compressed thinking keeps the
// replacement the same kind of text as the original.
//
// What it keeps is chosen for what dropping the trace outright would actually lose. Tool
// results already carry the evidence for anything the model went and checked; what nothing
// else records is the reasoning that never became a tool call — the approach weighed and
// abandoned, and the reason it was abandoned. Without that the next round is free to
// propose it again.

const COMPRESS_PROMPT: &str = "You compress your own chain of thought so you can carry on \
     without re-reading it. Keep only: approaches you tried or considered and then \
     rejected, and why; facts you established along the way; and the intent you ended on. \
     Drop restatements of the question, anything already visible in the conversation, and \
     any working you would reach the same way again. Write it in the first person as \
     compressed thinking, the way the trace itself is written — not as a report about it. \
     No preamble, no headings, no more than a short paragraph.";

// Generous for what should be a short answer, for the same reason the auto-mode reviewer's
// cap is: the compressor is the same model, so it may well think before it summarizes, and
// a cap that lands mid-thought yields a reply with no summary in it at all.

const COMPRESS_MAX_TOKENS: u64 = 1000;

// The chain of thought on an assistant message, if it has one worth acting on.

pub(crate) fn reasoning_text(message: &Value) -> Option<&str> {
    message[REASONING_KEY]
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

// A reasoning model asked to summarize thinks first, and its scratchpad arrives ahead of
// the answer in the same field. Kept out of the stored summary for the obvious reason —
// storing the thinking about the compression instead of the compression would defeat the
// whole exercise.

fn without_think_block(text: &str) -> &str {
    match text.find("</think>") {
        Some(end) => text[end + "</think>".len()..].trim(),
        None => text.trim(),
    }
}

// What the policy did to one reply, in numbers rather than in prose.
//
// Returned instead of a formatted line because two callers want different things from it:
// the transcript wants a sentence, and the session totals want the figures. Formatting it
// here would mean the totals had to parse their own output back.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReasoningOutcome {
    Dropped { tokens: usize },
    Compressed { before: usize, after: usize },
}

// What compression has bought on the conversation so far, for /context.
//
// Cleared by /clear with the history it describes, and only by that: /compact and /model
// both keep the conversation, so the savings made on it still stand. Kept out here rather
// than derived from `messages` because it cannot be derived — once a trace is replaced, the
// history holds the compressed version and no record of what it cost before.

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReasoningStats {
    pub(crate) turns: usize,
    pub(crate) before: usize,
    pub(crate) after: usize,
}

impl ReasoningStats {
    pub(crate) fn record(&mut self, outcome: ReasoningOutcome) {
        if let ReasoningOutcome::Compressed { before, after } = outcome {
            self.turns += 1;
            self.before += before;
            self.after += after;
        }
    }

    pub(crate) fn saved(&self) -> usize {
        self.before.saturating_sub(self.after)
    }
}

// Whether applying the policy will cost a request to the server.
//
// The activity bar is drawn before the work starts, so the caller has to know this without
// doing the work first. Dropping a field and declining a trace too short to be worth
// compressing are both over before a frame could be painted; announcing either as work in
// progress would report something that is not happening.

pub(crate) fn reasoning_needs_a_request(
    message: &Value,
    policy: Reasoning,
    threshold: usize,
) -> bool {
    policy == Reasoning::Compress
        && reasoning_text(message).is_some_and(|text| estimate_reasoning_tokens(text) >= threshold)
}

// Apply the session's reasoning policy to the reply that just finished.
//
// Called on the tail of the conversation and only there, which is what makes this safe to
// do every turn. The KV cache on the server is a prefix cache: rewriting the message that
// was just generated invalidates from that message forward, and nothing is in front of it
// yet, so the re-prefill is this reply and no more. Rewriting anything older would
// invalidate every turn after it and cost far more than the tokens it saved.
//
// That same property is what stops a trace being compressed twice. Once this returns, the
// message moves into the history and is never revisited, so there is no state to track and
// no marker to write into the message — a compressed trace is simply one that is already
// behind the write point.
//
// A failure here keeps the original. Compression is an optimization, and losing a chain of
// thought to a summarizer that timed out would be a worse outcome than the tokens it was
// trying to save.

pub(crate) async fn apply_reasoning(
    client: &LMStudioClient,
    message: &mut Value,
    policy: Reasoning,
    threshold: usize,
) -> Option<ReasoningOutcome> {
    let Some(original) = reasoning_text(message) else {
        return None;
    };

    match policy {
        Reasoning::Keep => None,

        Reasoning::Drop => {
            let tokens = estimate_reasoning_tokens(original);
            if let Some(object) = message.as_object_mut() {
                object.remove(REASONING_KEY);
            }
            Some(ReasoningOutcome::Dropped { tokens })
        }

        Reasoning::Compress => {
            let before = estimate_reasoning_tokens(original);
            if before < threshold {
                return None;
            }

            let original = original.to_string();
            let summary = match client
                .chat_completion_plain(
                    vec![
                        json!({"role": "system", "content": COMPRESS_PROMPT}),
                        json!({"role": "user", "content": original.clone()}),
                    ],
                    COMPRESS_MAX_TOKENS,
                )
                .await
            {
                Ok(reply) => without_think_block(&reply).to_string(),
                Err(e) => {
                    crate::debug::log("reasoning compression failed", &format!("{:#}", e));
                    return None;
                }
            };

            // A summary that came back empty, or that is no shorter than what it replaces,
            // is not worth the substitution: the original is at least the model's own words
            // in its own voice, and a compressor that expanded the trace has misunderstood
            // the job. Both happen with small models often enough to be worth the guard.
            let after = estimate_reasoning_tokens(&summary);
            if summary.is_empty() || after >= before {
                return None;
            }

            message[REASONING_KEY] = json!(summary);
            Some(ReasoningOutcome::Compressed { before, after })
        }
    }
}

// The same chars/4 estimate `estimate_tokens` uses, applied to one string rather than to a
// whole conversation. Rough, and deliberately the same roughness: a threshold tuned against
// one number and reported against another would be confusing to reason about.

pub(crate) fn estimate_reasoning_tokens(text: &str) -> usize {
    text.chars().count() / 4
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


    // A completion whose content is built at runtime. `serve_once` wants a `&'static str`
    // because it answers from a spawned task, and a test body long enough to be worth
    // measuring is not worth writing out as a literal.
    fn reply(content: &str) -> &'static str {
        let body = json!({"choices": [{"message": {"role": "assistant", "content": content}}]});
        Box::leak(body.to_string().into_boxed_str())
    }

    // A reply carrying `long` tokens of chain of thought, as the assembler would build it.
    fn thought(chars: usize) -> serde_json::Value {
        json!({
            "role": "assistant",
            "content": "Done.",
            "reasoning_content": "z".repeat(chars),
        })
    }



    #[tokio::test]
    async fn keep_leaves_the_chain_of_thought_where_it_is() {
        let url = serve_once("{}", "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        let mut message = thought(8_000);
        let note = apply_reasoning(&client, &mut message, Reasoning::Keep, 800).await;

        // The default must not touch the message at all — no request, no rewrite.
        assert!(note.is_none());
        assert_eq!(message["reasoning_content"].as_str().unwrap().len(), 8_000);
    }


    #[tokio::test]
    async fn drop_removes_the_field_rather_than_emptying_it() {
        let url = serve_once("{}", "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        let mut message = thought(8_000);
        let note = apply_reasoning(&client, &mut message, Reasoning::Drop, 800).await;

        // Removed outright: an empty string is still a key the chat template will read and
        // render as an empty thought, which is not the same as never having sent one.
        assert!(message.get("reasoning_content").is_none());
        assert_eq!(message["content"], "Done.");
        // 8000 chars at four to the token, reported so /context can total it up.
        assert_eq!(note, Some(ReasoningOutcome::Dropped { tokens: 2_000 }));
    }


    #[tokio::test]
    async fn compress_ignores_a_thought_too_short_to_be_worth_a_round_trip() {
        let body = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"summary\"}}]}";
        let url = serve_once(body, "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        // 400 chars is ~100 estimated tokens, well under the threshold.
        let mut message = thought(400);
        let note = apply_reasoning(&client, &mut message, Reasoning::Compress, 800).await;

        assert!(note.is_none());
        assert_eq!(message["reasoning_content"].as_str().unwrap().len(), 400);
    }


    #[tokio::test]
    async fn compress_replaces_a_long_thought_and_strips_the_compressors_own_thinking() {
        // The compressor is the same model, so it may think about the summary before
        // writing it. Storing that instead of the summary would defeat the whole exercise.
        let body = concat!(
            "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":",
            "\"<think>What matters here?</think>Tried the regex path, too slow.\"}}]}"
        );
        // Capturing rather than `serve_once`: a request this size is not drained by the
        // one-shot server, and a socket closed over an unread body is an RST on Windows —
        // which the client reports as a failed compression rather than as a bad fixture.
        let (url, sent) = serve_once_capturing(body, "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        let mut message = thought(8_000);
        let note = apply_reasoning(&client, &mut message, Reasoning::Compress, 800).await;

        // The trace itself is what the compressor was asked about, not a summary of the
        // conversation around it: this request has no history in it at all.
        let sent = sent.lock().unwrap().clone();
        assert!(sent.contains(&"z".repeat(100)), "the trace was not sent");

        let stored = message["reasoning_content"].as_str().unwrap();
        assert_eq!(stored, "Tried the regex path, too slow.");
        assert!(!stored.contains("What matters here"));
        // The content of the reply is untouched — only the thinking is rewritten.
        assert_eq!(message["content"], "Done.");
        // Both figures come back, not a sentence: the transcript line and the session
        // totals want different things out of the same result.
        let before = crate::estimate_reasoning_tokens(&"z".repeat(8_000));
        let after = crate::estimate_reasoning_tokens("Tried the regex path, too slow.");
        assert_eq!(note, Some(ReasoningOutcome::Compressed { before, after }));
    }


    #[tokio::test]
    async fn a_failed_compression_keeps_the_original_thought() {
        // Nothing listening: the request cannot succeed. Losing a chain of thought to a
        // summarizer that fell over would be a worse outcome than the tokens it saves.
        let client = crate::LMStudioClient::new(
            "http://127.0.0.1:1".to_string(),
            "m".to_string(),
            None,
        );

        let mut message = thought(8_000);
        let note = apply_reasoning(&client, &mut message, Reasoning::Compress, 800).await;

        assert!(note.is_none());
        assert_eq!(message["reasoning_content"].as_str().unwrap().len(), 8_000);
    }


    #[tokio::test]
    async fn a_compression_that_grew_the_thought_is_rejected() {
        // Small models do expand rather than compress when the trace is terse. The
        // substitution is only worth making when it actually shrinks something, and the
        // model's own words are the better fallback when it does not.
        let (url, _sent) = serve_once_capturing(
            reply(&"much longer than the original ".repeat(200)),
            "application/json",
        )
        .await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        let mut message = thought(4_000);
        let note = apply_reasoning(&client, &mut message, Reasoning::Compress, 800).await;

        assert!(note.is_none());
        assert_eq!(message["reasoning_content"].as_str().unwrap().len(), 4_000);
    }


    #[tokio::test]
    async fn a_reply_with_no_thinking_is_left_alone_under_every_policy() {
        let url = serve_once("{}", "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        // A non-thinking model, and the empty-string case a server can send instead.
        for mut message in [
            json!({"role": "assistant", "content": "Done."}),
            json!({"role": "assistant", "content": "Done.", "reasoning_content": "   "}),
        ] {
            for policy in [Reasoning::Keep, Reasoning::Compress, Reasoning::Drop] {
                let note = apply_reasoning(&client, &mut message, policy, 800).await;
                assert!(note.is_none(), "{:?} acted on a reply with no thinking", policy);
            }
            assert_eq!(message["content"], "Done.");
        }
    }


    #[test]
    fn the_session_total_counts_compressions_and_ignores_drops() {
        let mut stats = ReasoningStats::default();
        assert_eq!(stats.turns, 0);
        assert_eq!(stats.saved(), 0);

        stats.record(ReasoningOutcome::Compressed { before: 1_200, after: 300 });
        stats.record(ReasoningOutcome::Compressed { before: 800, after: 100 });

        // Dropping is not compression and must not inflate the figure /context reports —
        // the question that line answers is whether the extra request is paying for
        // itself, and a drop costs no request.
        stats.record(ReasoningOutcome::Dropped { tokens: 5_000 });

        assert_eq!(stats.turns, 2);
        assert_eq!(stats.before, 2_000);
        assert_eq!(stats.after, 400);
        assert_eq!(stats.saved(), 1_600);
    }


    #[test]
    fn only_a_compression_over_the_threshold_counts_as_work() {
        // What the activity bar is gated on. It has to agree with what `apply_reasoning`
        // then actually does, or the bar announces a request that never goes out — or
        // worse, a compression runs with no bar and a frozen input box.
        let long = thought(8_000);
        let short = thought(400);
        let none = json!({"role": "assistant", "content": "Done."});

        assert!(reasoning_needs_a_request(&long, Reasoning::Compress, 800));
        assert!(!reasoning_needs_a_request(&short, Reasoning::Compress, 800));
        assert!(!reasoning_needs_a_request(&none, Reasoning::Compress, 800));

        // Neither of the other two ever reaches the server.
        for policy in [Reasoning::Keep, Reasoning::Drop] {
            assert!(!reasoning_needs_a_request(&long, policy, 800));
        }

        // Exactly at the threshold is over it, the same comparison the compressor makes.
        assert!(reasoning_needs_a_request(&thought(3_200), Reasoning::Compress, 800));
    }


    #[test]
    fn reasoning_names_parse_the_way_the_config_file_spells_them() {
        assert_eq!(Reasoning::parse("keep"), Some(Reasoning::Keep));
        assert_eq!(Reasoning::parse(" Compress "), Some(Reasoning::Compress));
        assert_eq!(Reasoning::parse("DROP"), Some(Reasoning::Drop));
        assert_eq!(Reasoning::parse("compact"), None);

        // /reasoning prints these back and the config file reads them, so the two
        // spellings have to stay the same string.
        assert_eq!(Reasoning::Compress.to_string(), "compress");
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
