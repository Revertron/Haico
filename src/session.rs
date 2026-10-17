//! Saving, resuming and compacting a conversation.

use crate::trace::Caller;
use crate::config::{compact_hint_threshold, Reasoning};
use crate::llm::LMStudioClient;
use crate::tools::TodoItem;
use crate::util::truncate_at_boundary;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
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
    let cwd = env::current_dir().ok()?;
    let path = session_file_path_for(&cwd)?;
    adopt_legacy_session(&cwd, &path);
    Some(path)
}

// Split out from session_file_path so it can be exercised without changing the process's
// working directory, which every other test that walks "." would race against.

pub(crate) fn session_file_path_for(cwd: &std::path::Path) -> Option<PathBuf> {
    let store = crate::store::dir()?;
    Some(store.join("sessions").join(format!("{}-{:016x}.json", session_label(cwd), path_digest(cwd))))
}

fn session_label(cwd: &std::path::Path) -> String {
    cwd.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "root".to_string())
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .take(40)
        .collect()
}

// The hash in a session's file name, which has to come out the same on every build.
//
// It was `DefaultHasher`, whose output the standard library explicitly does not promise to
// keep between Rust releases: a toolchain upgrade could have moved every project's session
// to a new name, and `--resume` would have found nothing, with the old file still sitting
// beside it. FNV-1a is a dozen lines and is defined by its constants, not by a release.
//
// Windows paths are compared without regard to case, so they are hashed that way too —
// `E:\proj` and `e:\proj` are one directory, and were two sessions.
fn path_digest(cwd: &std::path::Path) -> u64 {
    let text = cwd.to_string_lossy();
    let text = if cfg!(windows) { text.to_lowercase() } else { text.into_owned() };
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3))
}

// Moves a session saved under the old `DefaultHasher` name to the stable one, once.
//
// Without it the switch to a stable hash would itself be the upgrade that loses every
// session. The old name can only be computed by the toolchain that wrote it, which is why
// this has to run now, while builds still share that hasher; it can go once there are no
// sessions left that predate the change. Best-effort: a rename that fails leaves the old
// file where it was, which is no worse than not trying.
fn adopt_legacy_session(cwd: &std::path::Path, path: &std::path::Path) {
    use std::hash::{Hash, Hasher};

    if path.exists() {
        return;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cwd.hash(&mut hasher);
    let legacy = path.with_file_name(format!("{}-{:016x}.json", session_label(cwd), hasher.finish()));
    if legacy != path && legacy.exists() {
        let _ = fs::rename(&legacy, path);
    }
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
//
// Written beside the file and renamed over it, because this runs after every turn and the
// file is the only copy: a plain write interrupted by a kill (a second Ctrl+C, a scheduled
// `-p --resume` stopped by its scheduler) left it truncated, and the next `--resume`
// answered "could not be read" for the whole conversation. A rename replaces the file
// whole or not at all. The error is returned rather than dropped, so a session that is
// not being saved is something the user hears about before they need it.

pub(crate) fn save_session(path: &std::path::Path, messages: &[Value], model: &str, todos: &[TodoItem]) -> Result<()> {
    // Skip the system prompt: it is rebuilt on load.
    let Some(rest) = messages.get(history_start(messages)..) else { return Ok(()) };

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

    let json = serde_json::to_string(&session).context("could not serialize the session")?;
    crate::util::write_atomically(path, json.as_bytes())
}


pub(crate) fn load_session(path: &std::path::Path) -> Result<SavedSession> {
    let raw = fs::read_to_string(path).context("no saved session for this directory")?;
    let mut session: SavedSession =
        serde_json::from_str(&raw).context("saved session could not be read")?;

    if session.version != SESSION_FORMAT_VERSION {
        bail!(
            "saved session is version {}, this build writes version {}",
            session.version,
            SESSION_FORMAT_VERSION
        );
    }

    // A session saved before 'done' became 'completed' is the same format and loads fine,
    // so it is not worth a version bump — but left alone its finished items would come
    // back looking outstanding, and the model would be handed a plan it has to do twice.
    for item in &mut session.todos {
        if let Some(status) = crate::tools::normalize_status(&item.status) {
            item.status = status.to_string();
        }
    }

    Ok(session)
}

// Where the REPL keeps its input history between sessions.

// Named `history` inside the store rather than staying a second dotfile beside it: it was
// `~/.haico_history` only because the home directory was the whole of haico's storage, and
// there is no reason for the input history to be the one thing that lives outside.
pub(crate) fn history_file_path() -> Option<PathBuf> {
    crate::store::path("history")
}


// What one image is counted as. A round figure, and honestly so: a vision model
// tiles an image and charges by the tile, so the true number depends on
// dimensions this never decodes, and it lands somewhere in the high hundreds to
// low thousands for anything screenshot-sized. Being roughly right here is the
// entire requirement — what it feeds is a percentage on a status bar and the
// threshold auto mode compacts at.
//
// What it must not be is the alternative. A data URL is four base64 characters
// for every three bytes of image, so counting the message literally makes a
// two-megabyte screenshot read as ~700k tokens: twenty times a 32k window, an
// instant autocompact in auto mode, and compaction cannot shrink an image, so
// the turn would go round that loop rather than answering.
const IMAGE_TOKENS: usize = 1000;

pub(crate) fn estimate_tokens(messages: &[Value]) -> usize {
    messages.iter().map(estimate_message).sum()
}

fn estimate_message(message: &Value) -> usize {
    match message.get("content").and_then(Value::as_array) {
        // The ordinary shape, and every message that carries no image:
        // characters over four across the whole thing, envelope included.
        None => message.to_string().chars().count() / 4,
        // A content-parts array, costed a part at a time so an image's base64 —
        // the bulk of the message and none of its meaning — is replaced rather
        // than counted.
        Some(parts) => parts
            .iter()
            .map(|part| {
                if part["type"] == "image_url" {
                    IMAGE_TOKENS
                } else {
                    part.to_string().chars().count() / 4
                }
            })
            .sum(),
    }
}

// Assumed context window when none is configured. Override with --context / HAICO_CONTEXT;
// a hard-coded guess would either nag constantly on a large window or warn too late on a
// small one.

/// `used` is `Agent::context_used`, and `measured` whether that rests on the server's own
/// count of the last request rather than on the estimate alone.
pub(crate) fn report_context(
    messages: &[Value],
    used: usize,
    measured: bool,
    context_tokens: usize,
    reasoning: &ReasoningStats,
) {
    let turns = messages.iter().filter(|m| m["role"] == "user").count();
    let percent = (used as f64 / context_tokens as f64) * 100.0;

    outln!(
        "{} messages, {} user turn(s), ~{} of {} tokens ({:.1}%, {}).",
        messages.len(),
        turns,
        used,
        context_tokens,
        percent,
        if measured { "from the server's count of the last request" } else { "estimated" }
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

/// What a message says, whatever shape its content is in.
///
/// A content-parts array reads as an empty string through `as_str`, and the
/// caller below drops an empty one — so without this a turn that started with an
/// image summarized as though the user had said nothing at all. The image itself
/// cannot go to a text summarizer, so it leaves a mark saying it was there;
/// "the user asked about this" is the part the summary needs to keep.
pub(crate) fn message_text(message: &Value) -> String {
    match &message["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("text") => part["text"].as_str().unwrap_or("").to_string(),
                Some("image_url") => "[an image]".to_string(),
                _ => String::new(),
            })
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

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
// Asking for the first person was not enough on its own, which is what the rest of the
// wording is for. A compressor handed a trace as a user turn reads it as a question and
// answers it, and one came back in the first person of an *advisor*: "My intent: steer them
// to capture just owned locals", then a "Concrete fix:" and a fenced Rust block. Stored as
// the reply's own chain of thought, that left the history carrying a thought in which "I"
// was somebody else and the model itself was "them". The round after it did not continue
// the work — it continued the voice, and its thinking became commentary on whether there
// had been any previous thinking at all to quote. So the trace is named as the model's own
// before anything is asked of it, answering and advising are refused outright, and the
// pronouns that carried the other voice are named rather than left to be inferred from
// "first person".

//
// The trace goes inside `<trace>` tags, and the prompt names them, because a bare trace as the
// whole user turn was read as part of the instruction instead. At temperature 0 and on a trace
// no different from the ones before it, the compressor concluded "there's no actual chain of
// thought provided ... the entire content IS the instruction itself" and refused: "There's no
// prior chain of thought here to condense". Twice in 53 compressions, in two separate runs, and
// the refusal was stored as the model's thinking both times. The tags leave nothing to decide
// about where the instruction ends and the text begins.

const COMPRESS_PROMPT: &str = "The text inside <trace> tags in the next message is your own \
     chain of thought from a moment ago — not a question, not a request, and nobody is waiting \
     on an answer to it. \
     Rewrite it shorter so you can carry on from it without re-reading it. Do not solve the \
     problem it is about, do not propose a fix, do not address anyone. Keep only: approaches \
     you tried or considered and then rejected, and why; facts you established along the \
     way; and the intent you ended on. Drop restatements of the question, anything already \
     visible in the conversation, and any working you would reach the same way again. Write \
     it as thinking rather than as a report about thinking: first person, the way the trace \
     itself is written — \"I\" is you, and there is no \"they\", no \"the model\" and no \
     \"the user\". No preamble, no headings, no fenced code, no more than a short paragraph.";

// Uncapped: the compressor is the same model, so it thinks before it summarizes, and a cap
// that lands mid-thought would leave nothing but an unfinished scratchpad to store. Letting
// it run to its own stop is what `chat_completion_plain`'s finish_reason check is for — a
// reply that still hit "length" (the model's own context limit, not an artificial one here)
// is treated as a failure rather than trusted as a summary.

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

pub(crate) fn without_think_block(text: &str) -> &str {
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

            let Some(summary) = compress_trace(client, original).await else {
                return None;
            };
            let after = estimate_reasoning_tokens(&summary);
            message[REASONING_KEY] = json!(summary);
            Some(ReasoningOutcome::Compressed { before, after })
        }
    }
}

// One chain of thought shortened by the compressor, or `None` when the compressor failed or
// gave back something that is not a shorter summary of it. Shared by the reasoning policy
// and by compaction, which uses it on the largest traces when a transcript is over budget.
pub(crate) async fn compress_trace(client: &LMStudioClient, original: &str) -> Option<String> {
    let before = estimate_reasoning_tokens(original);
    // A sub-agent copy, like the reviewer and research: the main conversation's
    // thinking level is no more relevant to summarizing a trace than its temperature
    // is. It is not asked to skip thinking, though — that was tried, and a compressor
    // answering cold reads the trace it was handed as a question and answers it,
    // which is the adviser voice the prompt above now argues against.
    let summary = match client
        .for_sub_agent()
        .chat_completion_plain(
            Caller::Compressor,
            vec![
                json!({"role": "system", "content": COMPRESS_PROMPT}),
                json!({"role": "user", "content": format!(
                    "The chain of thought to shorten:\n\n<trace>\n{}\n</trace>",
                    original
                )}),
            ],
            None,
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
    if !reads_as_a_summary(original, &summary) {
        crate::debug::log("reasoning compression rejected, not a summary of the trace", &summary);
        return None;
    }

    Some(summary)
}

// Whether a compressor's reply is a summary of the trace rather than something else — a
// refusal, an answer to a question the trace was mistaken for, a reply to somebody.
//
// Judged by vocabulary: a summary is made of the trace's own words, so nearly all of its
// longer words appear in the trace, while a reply about the request is made of words about
// requests. Measured over every compression recorded so far (53, several runs): real summaries
// share at least 63% of their distinct words of four or more characters with their trace, the
// two refusals 34% and 42%. Identifiers need no special treatment — `bound_addr` or `E0308`
// in a summary are exactly the words that come from the trace. A reply too short to carry
// evidence either way is let through; the length guard above has already seen it.

const MIN_SHARED_WORDS: f64 = 0.5;
const MIN_WORDS_TO_JUDGE: usize = 8;

fn reads_as_a_summary(original: &str, summary: &str) -> bool {
    fn words(text: &str) -> HashSet<String> {
        text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|w| w.chars().count() >= 4)
            .map(str::to_lowercase)
            .collect()
    }
    let summary_words = words(summary);
    if summary_words.len() < MIN_WORDS_TO_JUDGE {
        return true;
    }
    let original_words = words(original);
    let shared = summary_words.iter().filter(|w| original_words.contains(*w)).count();
    shared as f64 / summary_words.len() as f64 >= MIN_SHARED_WORDS
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

        let transcript = render_transcript(&messages, 60_000);

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

        let report = compact_history(&client, &mut messages, 131_072, 0).await.unwrap();
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


    // A session shaped like the p2p_auth one: a task, an early read of a file that was then
    // rewritten, a long middle, and a latest build whose errors are the state to continue from.
    fn long_session(middle_turns: usize, thinking: &str) -> Vec<serde_json::Value> {
        let call = |id: &str, name: &str, arguments: serde_json::Value| {
            json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments.to_string()}})
        };
        let mut messages = vec![
            json!({"role": "system", "content": "system prompt"}),
            json!({"role": "user", "content": "THE TASK: build a p2p node"}),
            json!({"role": "assistant", "content": "", "reasoning_content": thinking,
                   "tool_calls": [call("r1", "read", json!({"file_path": "src/lib.rs"}))]}),
            json!({"role": "tool", "tool_call_id": "r1", "content": "OLD LIB CONTENTS ".repeat(200)}),
            json!({"role": "assistant", "content": "", "tool_calls": [call("w1", "write_file",
                   json!({"file_path": "src/lib.rs", "content": "fn new() {}\n".repeat(300)}))]}),
            json!({"role": "tool", "tool_call_id": "w1", "content": "Wrote src/lib.rs"}),
        ];
        for turn in 0..middle_turns {
            let id = format!("m{turn}");
            messages.push(json!({"role": "assistant", "content": "", "reasoning_content": thinking,
                                 "tool_calls": [call(&id, "run_command", json!({"command": "cargo check"}))]}));
            messages.push(json!({"role": "tool", "tool_call_id": id, "content": format!("OLD BUILD {turn} ").repeat(300)}));
        }
        messages.push(json!({"role": "assistant", "content": "", "reasoning_content": "LATEST THOUGHT",
                             "tool_calls": [call("last", "run_command", json!({"command": "cargo test"}))]}));
        messages.push(json!({"role": "tool", "tool_call_id": "last",
                             "content": "error[E0425]: cannot find value `bob_pk` in this scope"}));
        messages
    }

    #[test]
    fn the_compactor_sees_the_task_the_latest_state_and_the_reasons() {
        use crate::render_transcript;
        let messages = long_session(40, "I chose X because Y.");
        let transcript = render_transcript(&messages, 20_000);

        assert!(!transcript.contains("system prompt"));
        assert!(transcript.contains("THE TASK: build a p2p node"));
        // The end, which the old renderer cut off first.
        assert!(transcript.contains("cannot find value `bob_pk`"), "{transcript}");
        assert!(transcript.contains("assistant (thinking): LATEST THOUGHT"));
        assert!(transcript.contains("I chose X because Y."));
        // An old read of a file written since is a marker, not the old contents.
        assert!(transcript.contains("[the contents of src/lib.rs as read then; it was changed afterwards]"));
        assert!(!transcript.contains("OLD LIB CONTENTS"));
        // An old write is its path and size, not its contents.
        assert!(transcript.contains("[called write_file — content: (300 lines); file_path: src/lib.rs]"), "{transcript}");
        // Old builds are cut short; the calls themselves still say what was run.
        assert!(transcript.contains("command: cargo check"));
        assert!(transcript.len() < 40_000, "{} bytes", transcript.len());
    }

    #[tokio::test]
    async fn an_over_budget_transcript_has_its_largest_thinking_compressed_first() {
        use crate::compact_history;
        // A chatty model: every turn thinks at length. 131k-token window, so a 196 KB budget,
        // and twelve 30 KB traces are well over it.
        let chatty = "Let me think about the handshake ordering again. ".repeat(600);
        let mut messages = long_session(12, &chatty);
        let summary = "I settled the handshake ordering: the dialer's keys first.";
        let notes = "Task: build a p2p node\nCurrent state: error[E0425] bob_pk\nNext step: rename";
        let bodies: &'static [&'static str] = Box::leak(Box::new([
            &*Box::leak(reply(summary).to_string().into_boxed_str()),
            &*Box::leak(reply(summary).to_string().into_boxed_str()),
            &*Box::leak(reply(summary).to_string().into_boxed_str()),
            &*Box::leak(reply(summary).to_string().into_boxed_str()),
            &*Box::leak(reply(summary).to_string().into_boxed_str()),
            &*Box::leak(reply(summary).to_string().into_boxed_str()),
            &*Box::leak(reply(summary).to_string().into_boxed_str()),
            &*Box::leak(reply(notes).to_string().into_boxed_str()),
        ]));
        let (url, served) = serve_sequence(bodies).await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        let report = compact_history(&client, &mut messages, 131_072, 118_000).await.unwrap();
        let requests = *served.lock().unwrap();
        // Some traces were compressed — not all thirteen, only until it fit — then the notes.
        assert!(requests > 1 && requests < 14, "{requests} requests");
        assert!(report.contains("chains of thought compressed to fit"), "{report}");
        assert!(report.starts_with("Compacted: ~118000 ->"), "{report}");
        assert!(messages[1]["content"].as_str().unwrap().contains("Current state"));
    }

    #[tokio::test]
    async fn notes_that_continue_the_transcript_are_asked_for_again_and_the_task_is_kept() {
        use crate::compact_history;
        // The p2p_auth run's answer: more transcript, in the transcript's own format.
        let continued = "The remaining errors are: `_inner` needs renaming. Let me fix them.\n\n\
                         [called read — file_path: src/lib.rs; limit: 15]\ntool: use chacha20poly1305::aead;";
        let notes = "Task: a p2p node\nDone: lib.rs\nCurrent state: 2 errors\nNext step: fix them";
        let bodies: &'static [&'static str] = Box::leak(Box::new([
            &*Box::leak(reply(continued).to_string().into_boxed_str()),
            &*Box::leak(reply(notes).to_string().into_boxed_str()),
            &*Box::leak(reply("Task: same\nCurrent state: 1 error").to_string().into_boxed_str()),
        ]));
        let (url, served) = serve_sequence(bodies).await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        let mut messages = long_session(3, "thinking");

        compact_history(&client, &mut messages, 131_072, 0).await.unwrap();
        assert_eq!(*served.lock().unwrap(), 2, "the continuation was not asked for again");
        let first = messages[1]["content"].as_str().unwrap().to_string();
        assert!(first.starts_with("The task, as it was given:\n\nTHE TASK: build a p2p node"), "{first}");
        assert!(first.contains("Current state: 2 errors") && !first.contains("[called read"), "{first}");

        // A second compaction carries the task as it was given, not the first one's notes.
        messages.push(json!({"role": "assistant", "content": "Fixed one."}));
        compact_history(&client, &mut messages, 131_072, 0).await.unwrap();
        let first = messages[1]["content"].as_str().unwrap();
        assert!(first.starts_with("The task, as it was given:\n\nTHE TASK: build a p2p node\n\nNotes"), "{first}");
        assert!(first.contains("Current state: 1 error") && !first.contains("2 errors"), "{first}");
    }

    #[tokio::test]
    async fn compaction_refuses_rather_than_destroying_history() {
        use crate::compact_history;

        // Nothing worth compacting yet: the history must be left alone.
        let url = serve_once("{}", "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        let mut messages = vec![json!({"role": "system", "content": "s"})];
        assert!(compact_history(&client, &mut messages, 131_072, 0).await.is_err());
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
        assert!(compact_history(&client, &mut messages, 131_072, 0).await.is_err());
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

    // A reply carrying `chars` characters of chain of thought, as the assembler would build it.
    // Words rather than filler: a compression is checked against the trace's vocabulary, and a
    // trace of one repeated letter has none for a summary to share.
    fn thought(chars: usize) -> serde_json::Value {
        json!({
            "role": "assistant",
            "content": "Done.",
            "reasoning_content": thinking(chars),
        })
    }

    fn thinking(chars: usize) -> String {
        "Tried the regex path first, but it was too slow on long inputs, so the parser has to be          written by hand. "
            .chars()
            .cycle()
            .take(chars)
            .collect()
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
        assert!(sent.contains(&thinking(300)), "the trace was not sent");
        // Inside the tags the prompt names, so it cannot be taken for part of the instruction.
        // (The body is JSON, so the newline arrives escaped.)
        assert!(sent.contains(r"<trace>\nTried the regex path"), "the trace was not in its tags");

        let stored = message["reasoning_content"].as_str().unwrap();
        assert_eq!(stored, "Tried the regex path, too slow.");
        assert!(!stored.contains("What matters here"));
        // The content of the reply is untouched — only the thinking is rewritten.
        assert_eq!(message["content"], "Done.");
        // Both figures come back, not a sentence: the transcript line and the session
        // totals want different things out of the same result.
        let before = crate::estimate_reasoning_tokens(&thinking(8_000));
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


    // The trace and the reply of the compression that went wrong, cut down: the compressor took
    // the trace for part of its instruction and refused, and the refusal was stored as thinking.
    const REFUSED_TRACE: &str = "Now I have all the API details. Let me also verify HKDF 0.13 API         quickly, and confirm getrandom 0.4 fill API.

HKDF 0.13: `Hkdf::<Sha256>::new(Some(salt),         ikm)`, `.expand(info, &mut okm)`. Also `.prk()`.

getrandom 0.4: `getrandom::fill(&mut buf)`         returns Result. Let me verify quickly but I'm fairly confident. Actually getrandom 0.4 API —         `getrandom::fill` exists. Let me just write the code and let cargo check tell me.

Now,         design decisions to finalize:

**Keygen**: Ed25519 identity key. Secret stored as raw 32         bytes (seed). Public key = verifying_key. The binary writes secret key file — I'll store the         32-byte seed in the file. Then `--key <file>` reads 32 bytes and constructs         SigningKey::from_bytes. That's clean.

Wait — \"prints the public key in hex\". So keygen         prints pubkey hex. Good.

**Node**:
- `new(identity: SigningKey, trusted:         HashSet<VerifyingKey>, listen_addr: SocketAddr) -> Node`
- The task says a node is created         from those two plus a listen address. Should Node own the SigningKey? Yes.";
    const REFUSAL: &str = "There's no prior chain of thought here to condense — this is the first         turn in the conversation, so nothing preceded it for me to shorten. I'm not going to invent         a reasoning trace (e.g., about an Ed25519/X25519 P2P handshake) and rewrite that as if it         were mine; there's no such text supplied. If you paste the actual trace you want         shortened, I'll compress it per your rules.";

    #[test]
    fn a_refusal_is_not_a_summary_and_a_summary_is() {
        assert!(!reads_as_a_summary(REFUSED_TRACE, REFUSAL));
        let summary = "I'll store the identity as the raw 32-byte seed and rebuild it with             SigningKey::from_bytes; keygen prints the verifying key in hex. HKDF 0.13 is             Hkdf::<Sha256>::new(Some(salt), ikm) then expand, and getrandom 0.4 is getrandom::fill,             which cargo check will confirm. Node owns the SigningKey, the trusted HashSet and the             listen address.";
        assert!(reads_as_a_summary(REFUSED_TRACE, summary));
        // Too short to judge either way: let through, the length guard has already seen it.
        assert!(reads_as_a_summary(REFUSED_TRACE, "Going with raw seeds."));
    }

    #[tokio::test]
    async fn a_compression_that_is_not_a_summary_keeps_the_original_thought() {
        let (url, _sent) = serve_once_capturing(reply(REFUSAL), "application/json").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);

        let mut message = json!({"role": "assistant", "content": "Done.", "reasoning_content": REFUSED_TRACE});
        let note = apply_reasoning(&client, &mut message, Reasoning::Compress, 100).await;

        assert!(note.is_none());
        assert_eq!(message["reasoning_content"], REFUSED_TRACE);
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

        save_session(&path, &messages, "some-model", &todos).unwrap();
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

    #[test]
    fn the_session_hash_is_fixed_rather_than_whatever_the_toolchain_hashes_to() {
        use super::path_digest;
        use std::path::Path;

        // Pinned, because the point is that no release may change it: a different value
        // here is every user's `--resume` finding nothing. FNV-1a's own published value
        // for the empty input is the offset basis.
        assert_eq!(path_digest(Path::new("")), 0xcbf2_9ce4_8422_2325);
        assert_eq!(path_digest(Path::new("a")), 0xaf63_dc4c_8601_ec8c);

        // One directory spelled in two cases is one directory on Windows.
        if cfg!(windows) {
            assert_eq!(path_digest(Path::new(r"E:\Proj")), path_digest(Path::new(r"e:\proj")));
        }
    }

    #[test]
    fn a_session_under_the_old_hash_is_moved_to_the_new_name() {
        use super::{adopt_legacy_session, session_label};
        use std::hash::{Hash, Hasher};

        let dir = scratch("legacy-session", "").parent().unwrap().to_path_buf();
        let cwd = dir.join("project");
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        cwd.hash(&mut hasher);
        let legacy = dir.join(format!("{}-{:016x}.json", session_label(&cwd), hasher.finish()));
        let current = dir.join("project-current.json");
        let _ = std::fs::remove_file(&current);
        std::fs::write(&legacy, "{}").unwrap();

        adopt_legacy_session(&cwd, &current);
        assert!(current.exists());
        assert!(!legacy.exists());

        // A session already under the new name is never replaced by an old one.
        std::fs::write(&legacy, "old").unwrap();
        std::fs::write(&current, "new").unwrap();
        adopt_legacy_session(&cwd, &current);
        assert_eq!(std::fs::read_to_string(&current).unwrap(), "new");
        let _ = std::fs::remove_file(&legacy);
        let _ = std::fs::remove_file(&current);
    }

    #[test]
    fn a_session_that_cannot_be_written_says_so() {
        use crate::save_session;

        // A directory where the file should be: the rename cannot replace it.
        let dir = scratch("unsaveable", "").parent().unwrap().join("unsaveable-session.json");
        std::fs::create_dir_all(&dir).unwrap();
        let messages = vec![json!({"role": "user", "content": "hi"})];
        assert!(save_session(&dir, &messages, "m", &[]).is_err());
        // …and leaves no temporary behind.
        let leftovers = std::fs::read_dir(dir.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("unsaveable-session.json."))
            .count();
        assert_eq!(leftovers, 0);
        let _ = std::fs::remove_dir(&dir);
    }
}
