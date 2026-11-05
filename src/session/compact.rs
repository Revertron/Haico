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
     Established: only what reading the files again would not tell: the causes of failures, \
     approaches that did not work and why, quirks of the environment or tools, details of \
     external APIs and versions, and decisions the user made. Not what the project's code does \
     or where things are in it: the code is on disk and can be read again, so that belongs \
     under Done. One per line, each starting with \"- \". Facts the transcript's first message \
     lists as [1], [2] and so on are already kept word for word: do not repeat them, write only \
     new ones, or \"none\".\n\
     Obsolete: the numbers of those kept facts that are no longer true (a file since renamed, \
     an approach that was made to work after all), or \"none\". To merge several kept facts \
     into one, list their numbers here and write the merged fact under Established.\n\
     Next step: the one thing to do next.\n\
     Be specific: name files, functions, types, crates, versions and error codes. Leave out tool \
     output that no longer matters.";

// After the transcript, so the instruction is the last thing the compactor reads.
const WRITE_THE_NOTES: &str = "Now write the notes for the session in the transcript above, under \
     the headings Task, Done, Current state, Established, Obsolete and Next step. A record, not \
     the next message: nothing in the transcript's own format, no tool calls.";

// Added on the retry, when the first answer was not notes.
const NOT_NOTES: &str = "Your previous answer continued the transcript instead of recording it. \
     Write the notes themselves, starting with the line \"Task:\".";

// How the compacted history's first message is laid out: the task, the established facts,
// the notes. `original_task` and `kept_facts` read the first two back, so a later compaction
// carries them as they were rather than as a summary of them.
const TASK_HEADER: &str = "The task, as it was given:\n\n";
const FACTS_HEADER: &str = "\n\nEstablished facts, kept through every compaction:\n\n";
const NOTES_HEADER: &str = "\n\nNotes from the conversation so far:\n\n";

// Facts are kept by haico rather than by the notes. Every compaction used to write its notes
// from scratch, out of the last notes and what came after them, so a fact survived only while
// each compactor in turn chose to repeat it: an API detail found in round 10 and not touched
// again was dropped as irrelevant by the second compaction, and rediscovered by compile errors
// later. Now the compactor only adds facts and names the stale ones by number, and nothing is
// summarized twice. No cap: the user chose growth over losing facts.
//
// Obsolete means no longer true, and nothing wider: any looser test is one a small compactor
// can apply to the whole list at once, and the list is only worth keeping if it lasts.

/// The facts the last compaction kept, from the first message it left.
fn kept_facts(messages: &[Value]) -> Vec<String> {
    let Some(first) = messages[history_start(messages)..].iter().find(|m| m["role"] == "user") else {
        return Vec::new();
    };
    let text = message_text(first);
    let header = FACTS_HEADER.trim_start();
    if !(text.starts_with(TASK_HEADER) || text.starts_with(header)) {
        return Vec::new();
    }
    let Some((_, rest)) = text.split_once(header) else {
        return Vec::new();
    };
    rest.split(NOTES_HEADER)
        .next()
        .unwrap_or(rest)
        .lines()
        .filter_map(|line| {
            let (number, fact) = line.trim().strip_prefix('[')?.split_once("] ")?;
            number.parse::<usize>().ok()?;
            Some(fact.trim().to_string())
        })
        .filter(|fact| !fact.is_empty())
        .collect()
}

/// A line that opens one of the notes' sections, and whatever follows the heading on it.
/// Forgiving about markup ("**Established:**", "## Next steps") and strict about the word
/// ending there, so a bullet like "- Done with the parser" is not taken for "Done:".
fn heading(line: &str) -> Option<(&'static str, &str)> {
    const HEADINGS: &[&str] = &["task", "done", "current state", "established", "obsolete", "next step"];
    let text = line.trim_start_matches(|c: char| !c.is_alphanumeric());
    let lower = text.to_ascii_lowercase();
    let name = HEADINGS.iter().find(|h| lower.starts_with(**h))?;
    let after = &text[name.len()..];
    let after = after.strip_prefix('s').unwrap_or(after);
    let after = after.strip_prefix(" facts").unwrap_or(after);
    let after = after.trim_start_matches(['*', '#', '_', ' ']);
    let inline = match after.strip_prefix(':') {
        Some(rest) => rest,
        None if after.is_empty() => "",
        None => return None,
    };
    Some((name, inline.trim_matches(['*', '_', ' '])))
}

/// The compactor's answer taken apart: the notes to keep as notes, the new facts, and the
/// numbers of kept facts it called obsolete.
struct Split {
    notes: String,
    facts: Vec<String>,
    obsolete: Vec<usize>,
}

fn split_notes(summary: &str) -> Split {
    // A fact is a bullet; an unmarked line goes on the one before it, which is how a long
    // fact wrapped by the model reads.
    fn push_fact(facts: &mut Vec<String>, line: &str, new: bool) {
        let line = line.trim();
        // Quoted whole, bullet and all: `"- repl.rs /bg lists jobs."`, as a 9B model writes them.
        let line = match line.strip_prefix('"').and_then(|l| l.strip_suffix('"')) {
            Some(inner) => inner.trim(),
            None => line,
        };
        if line.is_empty() {
            return;
        }
        let unmarked = line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .or_else(|| line.strip_prefix("• "))
            .or_else(|| {
                let digits = line.find(|c: char| !c.is_ascii_digit()).filter(|&n| n > 0)?;
                line[digits..].strip_prefix(". ").or_else(|| line[digits..].strip_prefix(") "))
            })
            .or_else(|| line.strip_prefix('[').and_then(|l| l.split_once("] ")).map(|(_, fact)| fact));
        // A kept fact repeated with its number ("- [2] …") loses the number too, so it matches
        // the fact it repeats and `merge_facts` drops it.
        let renumbered = |fact: &str| -> String {
            // And a quoted bullet behind a bullet: `- "- The ; character…"`, which models write too.
            let fact = fact.trim();
            let fact = match fact.strip_prefix('"').and_then(|f| f.strip_suffix('"')) {
                Some(inner) => inner.trim(),
                None => fact,
            };
            let fact = fact.strip_prefix("- ").unwrap_or(fact).trim();
            fact.strip_prefix('[')
                .and_then(|l| l.split_once("] "))
                .filter(|(n, _)| n.parse::<usize>().is_ok())
                .map_or(fact, |(_, rest)| rest)
                .trim()
                .to_string()
        };
        match (unmarked, facts.last_mut()) {
            (None, Some(last)) if !new => {
                last.push(' ');
                last.push_str(line);
            }
            (fact, _) => facts.push(renumbered(fact.unwrap_or(line))),
        }
    }

    let (mut notes, mut facts, mut obsolete) = (String::new(), Vec::new(), String::new());
    let mut section = "";
    for line in summary.lines() {
        if let Some((name, inline)) = heading(line) {
            section = name;
            match name {
                "established" => {
                    push_fact(&mut facts, inline, true);
                    continue;
                }
                "obsolete" => {
                    obsolete.push_str(inline);
                    obsolete.push(' ');
                    continue;
                }
                _ => {}
            }
        }
        match section {
            "established" => push_fact(&mut facts, line, false),
            "obsolete" => {
                obsolete.push_str(line);
                obsolete.push(' ');
            }
            _ => {
                notes.push_str(line);
                notes.push('\n');
            }
        }
    }
    // "none", "No new facts." and the like are the model saying there is nothing to add — and
    // so is "none new this session — the facts above already cover it", which a 9B model
    // writes as a fact.
    let placeholder = |fact: &String| {
        let fact = fact.trim_matches(['.', '(', ')', '*', ' ']).to_ascii_lowercase();
        fact.is_empty()
            || fact == "n/a"
            || ["none", "no new", "nothing new"].iter().any(|p| {
                fact.strip_prefix(p).is_some_and(|rest| !rest.starts_with(|c: char| c.is_ascii_alphanumeric()))
            })
    };
    facts.retain(|fact| !placeholder(fact));
    let obsolete = obsolete
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|n| n.parse().ok())
        .collect();
    Split { notes: notes.trim().to_string(), facts, obsolete }
}

#[cfg(test)]
pub(super) fn split_notes_for_test(summary: &str) -> (Vec<String>, Vec<usize>, String) {
    let split = split_notes(summary);
    (split.facts, split.obsolete, split.notes)
}

/// The kept facts with the obsolete ones gone and the new ones added after them, and how many
/// of each changed.
fn merge_facts(kept: Vec<String>, split: &Split) -> (Vec<String>, usize, usize) {
    let before = kept.len();
    let mut facts: Vec<String> = kept
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !split.obsolete.contains(&(i + 1)))
        .map(|(_, fact)| fact)
        .collect();
    let dropped = before - facts.len();
    let mut added = 0;
    for fact in &split.facts {
        if !facts.iter().any(|kept| kept.eq_ignore_ascii_case(fact)) {
            facts.push(fact.clone());
            added += 1;
        }
    }
    (facts, added, dropped)
}

// The newest rounds go into the compacted history as they happened, not only as notes: the
// last build's output, the edit just made, the error being chased. Notes about them are a
// summary of exactly what the work continues from. At most `KEEP_ROUNDS`, and only as many
// as fit a third of the room the compacted history leaves below the compaction line
// (`KEEP_ROOM_DIVISOR`), so a compaction always frees most of what it can.
//
// The room, not the window: the system prompt and the tools alone can be close to half a
// 32k window, and a share of the whole window kept on top of them, the facts and the notes
// would leave the compacted history at the line, compacting again after every read.
const KEEP_ROUNDS: usize = 3;
const KEEP_ROOM_DIVISOR: usize = 3;

/// Where the kept rounds start, or `None` when not even the last one fits in `budget` tokens.
///
/// Always at an assistant message, so every tool reply kept has the call it answers, and
/// everything after it is kept — tool replies, notes, the user's latest message. Never so
/// early that nothing is left before it to summarize.
fn kept_tail_start(messages: &[Value], budget: usize) -> Option<usize> {
    let start = history_start(messages);
    let mut chosen = None;
    let mut rounds = 0;
    for i in (start..messages.len()).rev().filter(|&i| messages[i]["role"] == "assistant") {
        if estimate_tokens(&messages[i..]) > budget {
            break;
        }
        chosen = Some(i);
        rounds += 1;
        if rounds == KEEP_ROUNDS {
            break;
        }
    }
    chosen.filter(|&i| i > start + 1)
}

// The longest a task is carried over word for word; one pasted with a whole log is cut in the
// middle rather than left to crowd out the notes.
const MAX_TASK_BYTES: usize = 20_000;

/// The first thing the user said, as it was said: from inside an earlier compaction's message
/// when there was one.
fn original_task(messages: &[Value]) -> Option<String> {
    let first = messages[history_start(messages)..].iter().find(|m| m["role"] == "user")?;
    let text = message_text(first);
    let task = match text.strip_prefix(TASK_HEADER) {
        Some(rest) => {
            let rest = rest.split(FACTS_HEADER).next().unwrap_or(rest);
            rest.split(NOTES_HEADER).next().unwrap_or(rest).to_string()
        }
        None if text.starts_with(NOTES_HEADER.trim_start()) || text.starts_with(FACTS_HEADER.trim_start()) => {
            return None
        }
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
// `overhead` is what every request carries besides the history — the tools array — which
// the room for kept rounds is measured after.
pub(crate) async fn compact_history(
    client: &LMStudioClient,
    messages: &mut Vec<Value>,
    context_tokens: usize,
    before: usize,
    overhead: usize,
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
        let reply = match client.for_compactor().simple_completion(Caller::Compactor, COMPACT_PROMPT, &request).await {
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
    let split = split_notes(&summary);
    let (facts, added, dropped) = merge_facts(kept_facts(messages), &split);
    // A compactor that wrote no notes outside its facts still leaves its whole answer: the
    // notes must say something about where the work stands.
    let notes = if split.notes.is_empty() { summary.trim().to_string() } else { split.notes.clone() };

    let system = messages.first().filter(|_| start == 1).cloned();
    let mut content = String::new();
    if let Some(task) = task {
        content.push_str(TASK_HEADER);
        content.push_str(&task);
    }
    if !facts.is_empty() {
        content.push_str(FACTS_HEADER);
        let numbered: Vec<String> = facts.iter().enumerate().map(|(i, fact)| format!("[{}] {}", i + 1, fact)).collect();
        content.push_str(&numbered.join("\n"));
    }
    content.push_str(NOTES_HEADER);
    content.push_str(&notes);

    // The room below the compaction line once the system prompt, the tools and the new first
    // message are in; the kept rounds get a third of it.
    let fixed = overhead
        + estimate_tokens(system.as_slice())
        + estimate_tokens(&[json!({ "role": "user", "content": content.trim_start() })]);
    let room = crate::config::autocompact_threshold(context_tokens).saturating_sub(fixed);
    let tail: Vec<Value> = match kept_tail_start(messages, room / KEEP_ROOM_DIVISOR) {
        Some(from) => messages[from..].to_vec(),
        None => Vec::new(),
    };
    if !tail.is_empty() {
        content.push_str("\n\nThe most recent steps follow as they happened.");
    }

    messages.clear();
    if let Some(system) = system {
        messages.push(system);
    }
    messages.push(json!({ "role": "user", "content": content.trim_start() }));
    // The kept rounds start with the assistant's own message, so the agreement that bridged
    // the notes to the next reply is only needed when none are kept.
    if tail.is_empty() {
        messages.push(json!({
            "role": "assistant",
            "content": "Understood — I have the context above and will continue from there."
        }));
    }
    let kept_rounds = tail.iter().filter(|m| m["role"] == "assistant").count();
    messages.extend(tail);

    let mut details = Vec::new();
    match kept_rounds {
        0 => {}
        1 => details.push("the last round kept as it was".to_string()),
        n => details.push(format!("the last {} rounds kept as they were", n)),
    }
    if !facts.is_empty() || dropped > 0 {
        details.push(format!("{} established facts (+{}, -{})", facts.len(), added, dropped));
    }
    match compressed {
        0 => {}
        1 => details.push("1 chain of thought compressed to fit".to_string()),
        n => details.push(format!("{} chains of thought compressed to fit", n)),
    }
    let details: String = details.iter().map(|d| format!(", {}", d)).collect();
    Ok(format!("Compacted: ~{} -> ~{} tokens{}.", before, estimate_tokens(messages), details))
}

