//! Compaction: the history replaced with notes on it, and the transcript those notes are
//! written from.

use crate::trace::Caller;
use crate::llm::LMStudioClient;
use crate::util::truncate_at_boundary;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::HashSet;
use super::*;
// How much of the compactor's window the transcript may take, and the bytes a token is
// counted at. Half the window, so the compactor has the other half for its prompt, its own
// thinking (it is the same model, and it thinks before it writes) and the notes. Three bytes
// to the token rather than `estimate_tokens`' four: on the p2p_auth run that four ran 20% low
// on exactly this kind of text, code and tool output.
const TRANSCRIPT_SHARE: f64 = 0.5;
const BYTES_PER_TOKEN: usize = 3;

pub(crate) fn transcript_budget(context_tokens: usize) -> usize {
    (context_tokens as f64 * TRANSCRIPT_SHARE) as usize * BYTES_PER_TOKEN
}

// The newest turns are rendered in full and fill up to this share of the budget; everything
// older is rendered briefly. The newest turns are the state the work continues from — the
// last build's errors, the edit half made — and the old renderer, which kept the *first*
// 60 KB, threw exactly those away: on the p2p_auth run the compactor never saw the last four
// minutes, and its notes sent the model back into a rename it had already half done.
const RECENT_SHARE: f64 = 0.5;

// How much of one tool result each rendering keeps, cut in the middle (a build says what went
// wrong last).
const FULL_RESULT: usize = 6000;
const BRIEF_RESULT: usize = 400;

// The arguments a reader needs to know what a call did, shown whole; any other long string
// argument (a file's contents, an edit's text) is shown by its size, plus its start when the
// call is recent.
const NAMING_ARGUMENTS: &[&str] =
    &["file_path", "path", "command", "cwd", "pattern", "glob", "query", "question", "url", "name", "action"];

fn describe_call(call: &Value, full: bool) -> String {
    let name = call["function"]["name"].as_str().unwrap_or("?");
    let arguments = crate::tools::parse_tool_arguments(&call["function"]).unwrap_or_default();
    let Some(object) = arguments.as_object() else {
        return format!("[called {}]", name);
    };
    let mut parts: Vec<String> = Vec::new();
    for (key, value) in object {
        match value.as_str() {
            Some(text) if NAMING_ARGUMENTS.contains(&key.as_str()) => {
                parts.push(format!("{}: {}", key, truncate_at_boundary(text, 300)))
            }
            Some(text) if full && text.len() > 400 => parts.push(format!(
                "{}: {}… ({} lines)",
                key,
                truncate_at_boundary(text, 400),
                text.lines().count()
            )),
            Some(text) if full => parts.push(format!("{}: {}", key, text)),
            Some(text) => parts.push(format!("{}: ({} lines)", key, text.lines().count().max(1))),
            None => parts.push(format!("{}: {}", key, truncate_at_boundary(&value.to_string(), 100))),
        }
    }
    if parts.is_empty() { format!("[called {}]", name) } else { format!("[called {} — {}]", name, parts.join("; ")) }
}

// Whether a file was written after a given message: a `read` of it before that point shows a
// version that no longer exists, and the brief rendering says so instead of quoting it.
fn written_after(messages: &[Value], index: usize, path: &str) -> bool {
    messages[index + 1..].iter().any(|m| {
        m["tool_calls"].as_array().is_some_and(|calls| {
            calls.iter().any(|call| {
                matches!(call["function"]["name"].as_str(), Some("write_file" | "edit" | "multi_edit"))
                    && crate::tools::parse_tool_arguments(&call["function"])
                        .ok()
                        .and_then(|a| a["file_path"].as_str().map(|p| p == path))
                        .unwrap_or(false)
            })
        })
    })
}

fn render_message(messages: &[Value], index: usize, full: bool) -> String {
    let message = &messages[index];
    let role = message["role"].as_str().unwrap_or("?");
    let mut out = String::new();

    // Thinking goes in: it is where the reasons and the dead ends are, which the notes are asked
    // to keep and which the session's other text does not say. The old renderer left it out.
    if let Some(thinking) = reasoning_text(message) {
        out.push_str(&format!("{} (thinking): {}\n\n", role, thinking));
    }
    let content = message_text(message);
    if !content.is_empty() {
        let content = if role == "tool" {
            let call = messages[..index]
                .iter()
                .rev()
                .filter_map(|m| m["tool_calls"].as_array())
                .flatten()
                .find(|c| c["id"] == message["tool_call_id"]);
            let read_of = call
                .filter(|c| c["function"]["name"] == "read")
                .and_then(|c| crate::tools::parse_tool_arguments(&c["function"]).ok())
                .and_then(|a| a["file_path"].as_str().map(str::to_string));
            match read_of {
                Some(path) if !full && written_after(messages, index, &path) => {
                    format!("[the contents of {} as read then; it was changed afterwards]", path)
                }
                _ => crate::util::truncate_middle(&content, if full { FULL_RESULT } else { BRIEF_RESULT }),
            }
        } else {
            content
        };
        out.push_str(&format!("{}: {}\n\n", role, content));
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            out.push_str(&describe_call(call, full));
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// The session as the compactor reads it, aiming at `budget` bytes.
///
/// The task statement is always whole; the newest turns are whole up to `RECENT_SHARE` of the
/// budget; older turns are brief. `compact_history` compresses the largest chains of thought
/// when the result is still over budget, and cuts the middle only after that.
pub(crate) fn render_transcript(messages: &[Value], budget: usize) -> String {
    let start = history_start(messages);
    let count = messages.len() - start;
    if count == 0 {
        return String::new();
    }
    let first_user = messages[start..].iter().position(|m| m["role"] == "user");

    // Walk back from the end while the full renderings still fit the recent share. The newest
    // message is always full, however large.
    let recent_budget = (budget as f64 * RECENT_SHARE) as usize;
    let mut recent_from = count;
    let mut used = 0usize;
    for offset in (0..count).rev() {
        let size = render_message(messages, start + offset, true).len();
        if used + size > recent_budget && recent_from < count {
            break;
        }
        used += size;
        recent_from = offset;
    }

    let mut out = String::new();
    for offset in 0..count {
        let full = offset >= recent_from || Some(offset) == first_user;
        out.push_str(&render_message(messages, start + offset, full));
    }
    out
}

// What the compactor is told to produce.
//
// A record, under fixed headings, and explicitly not a continuation. The old one-line prompt
// asked for "notes for whoever picks the task up next", and on the p2p_auth run the compactor
// wrote the next message instead ("Continuing from where the session left off… Let me read
// the current state of files"); the model it was handed to said one sentence of intent and
// stopped, and an unattended run with 19 compile errors exited as finished. The headings name
// what that run's notes were missing: the current failures verbatim, and work half done.
const COMPACT_PROMPT: &str = "The transcript in the next message is a coding session that is \
     about to be deleted. The notes you write are all that will remain of it, and the work will \
     continue from them. Write them as a record of the session, not as its next message: do not \
     start the work, and do not write \"Let me\", \"I will now\" or anything else addressed to \
     anyone. Use these headings:\n\
     Task: what was asked, with every requirement and constraint, word for word where the \
     wording matters.\n\
     Done: what has been built or changed so far, file by file.\n\
     Current state: the result of the last build or test run, with the errors still failing \
     quoted exactly, and anything left half done (a change applied in some places but not \
     others, a file only partly rewritten).\n\
     Established: facts found along the way, such as API details, versions, causes of earlier \
     failures, and approaches that did not work and why.\n\
     Next step: the one thing to do next.\n\
     Be specific: name files, functions, types, crates, versions and error codes. Leave out tool \
     output that no longer matters.";

// After the transcript, so the instruction is the last thing the compactor reads.
const WRITE_THE_NOTES: &str = "Now write the notes for the session in the transcript above, under \
     the headings Task, Done, Current state, Established and Next step. A record, not the next \
     message: nothing in the transcript's own format, no tool calls.";

// Added on the retry, when the first answer was not notes.
const NOT_NOTES: &str = "Your previous answer continued the transcript instead of recording it. \
     Write the notes themselves, starting with the line \"Task:\".";

// How the compacted history's first message is laid out. `original_task` reads the first one
// back, so a second compaction keeps the task as it was given rather than the notes of the first.
const TASK_HEADER: &str = "The task, as it was given:\n\n";
const NOTES_HEADER: &str = "\n\nNotes from the conversation so far:\n\n";

// The longest a task is carried over word for word; one pasted with a whole log is cut in the
// middle rather than left to crowd out the notes.
const MAX_TASK_BYTES: usize = 20_000;

/// The first thing the user said, as it was said: from inside an earlier compaction's message
/// when there was one.
fn original_task(messages: &[Value]) -> Option<String> {
    let first = messages[history_start(messages)..].iter().find(|m| m["role"] == "user")?;
    let text = message_text(first);
    let task = match text.strip_prefix(TASK_HEADER) {
        Some(rest) => rest.split(NOTES_HEADER).next().unwrap_or(rest).to_string(),
        None if text.starts_with(NOTES_HEADER.trim_start()) => return None,
        None => text,
    };
    let task = task.trim();
    (!task.is_empty()).then(|| crate::util::truncate_middle(task, MAX_TASK_BYTES))
}

/// Whether the compactor wrote notes, not more transcript: the two headings the work cannot do
/// without, and none of the transcript's own call lines.
fn notes_look_like_notes(notes: &str) -> bool {
    let has = |heading: &str| {
        notes.lines().any(|line| {
            let line = line.trim_start_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            line.starts_with(heading)
        })
    };
    has("task") && has("current state") && !notes.contains("[called ")
}

// At most this many chains of thought are compressed to make a transcript fit: each is a model
// call of its own, and a compaction is something an unattended run waits on.
const MAX_TRACES_COMPRESSED: usize = 12;

// Replace the conversation with a summary of itself, keeping the system prompt.
//
// Everything after the system message goes, which also guarantees no orphaned tool_call_id
// survives into the compacted history — the summary is plain prose, so the pairing the
// server checks cannot be broken by this.
//
// `context_tokens` sizes the transcript to the compactor's window. When the transcript is over
// that budget, the largest chains of thought are compressed first — one model call each, on a
// copy of the history — because for a model that thinks 20–30k tokens a turn that is where the
// room is; for one whose thinking is short, the budget is usually met without a single call.
// `before` is the size to report, which the caller knows better than an estimate does.
pub(crate) async fn compact_history(
    client: &LMStudioClient,
    messages: &mut Vec<Value>,
    context_tokens: usize,
    before: usize,
) -> Result<String> {
    let start = history_start(messages);
    if messages.len() <= start + 1 {
        bail!("nothing to compact yet");
    }

    let budget = transcript_budget(context_tokens);
    let mut copy = messages.clone();
    let mut transcript = render_transcript(&copy, budget);
    let mut tried: HashSet<usize> = HashSet::new();
    let mut compressed = 0usize;
    while transcript.len() > budget && tried.len() < MAX_TRACES_COMPRESSED {
        // Largest first, the oldest of equals: the most room per call.
        let largest = (start..copy.len())
            .filter(|i| !tried.contains(i))
            .filter_map(|i| reasoning_text(&copy[i]).map(|text| (i, text.len())))
            .max_by_key(|&(i, len)| (len, std::cmp::Reverse(i)));
        let Some((index, _)) = largest else { break };
        tried.insert(index);
        let original = reasoning_text(&copy[index]).unwrap_or_default().to_string();
        if let Some(summary) = compress_trace(client, &original).await {
            copy[index][REASONING_KEY] = json!(summary);
            compressed += 1;
            transcript = render_transcript(&copy, budget);
        }
    }
    let transcript = crate::util::truncate_middle(&transcript, budget);

    // Fenced, with the instruction repeated after it. Unfenced, a transcript rendered this
    // regularly was *continued*: on a p2p_auth run the "notes" were an assistant line, a
    // `[called read]` line and the file it read, in the transcript's own format, with no
    // heading and no word about the tests the task required — the model resumed from them,
    // wrote no tests, and finished on a clean build. The compressor's `<trace>` tags are the
    // same lesson, learned earlier.
    let request = format!("<transcript>\n{}\n</transcript>\n\n{}", transcript, WRITE_THE_NOTES);
    let mut summary = String::new();
    for attempt in 0..2 {
        let request = if attempt == 0 { request.clone() } else { format!("{}\n\n{}", request, NOT_NOTES) };
        let reply = match client.simple_completion(Caller::Compactor, COMPACT_PROMPT, &request).await {
            Ok(reply) => reply,
            // A retry that fails keeps the first answer: poor notes beside the task verbatim
            // still beat a compaction that did not happen at the edge of the window.
            Err(_) if attempt > 0 && !summary.is_empty() => break,
            Err(e) => return Err(e),
        };
        summary = without_think_block(&reply).to_string();
        if notes_look_like_notes(&summary) {
            break;
        }
        crate::debug::log("compaction notes rejected, not a record of the session", &summary);
    }
    if summary.is_empty() {
        bail!("the model returned an empty summary; history left untouched");
    }

    // The task goes in word for word whatever the notes say, because it is the one thing the
    // work cannot be finished without, and the one thing a bad set of notes is worst at losing.
    let task = original_task(messages);
    let system = messages.first().filter(|_| start == 1).cloned();
    messages.clear();
    if let Some(system) = system {
        messages.push(system);
    }
    let content = match task {
        Some(task) => format!("{}{}{}{}", TASK_HEADER, task, NOTES_HEADER, summary),
        None => format!("{}{}", NOTES_HEADER.trim_start(), summary),
    };
    messages.push(json!({ "role": "user", "content": content }));
    messages.push(json!({
        "role": "assistant",
        "content": "Understood — I have the context above and will continue from there."
    }));

    let traces = match compressed {
        0 => String::new(),
        1 => ", 1 chain of thought compressed to fit".to_string(),
        n => format!(", {} chains of thought compressed to fit", n),
    };
    Ok(format!("Compacted: ~{} -> ~{} tokens{}.", before, estimate_tokens(messages), traces))
}

