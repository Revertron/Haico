//! `--trace <file>`: every model call, as it happens, one JSON object per line.
//!
//! For collecting training data out of real runs. Not the session file, and on purpose:
//! the session is the history *as it stands*, and in auto mode that history is replaced by
//! a summary part-way through a long job — everything before it would be gone. It also
//! carries neither the system prompt nor the tool definitions, and a tool-calling model
//! trained on examples without the tools it was shown learns to call tools out of thin
//! air. Here each line is the exact request that went to the server and the whole reply
//! that came back, written the moment it lands and only ever appended to, so nothing a
//! later compaction or reasoning policy does can reach back and change it.
//!
//! Every conversation is recorded, not only the main one: the auto-mode reviewer, the
//! research sub-agent, the reasoning compressor, the compactor and `web_fetch`'s
//! summarizer each have their own `caller`, so a trainer keeps what it wants and a
//! reviewer's verdicts are there to train a reviewer on.
//!
//! Lines carry a `type`:
//! - `start` — once per process: the run id, haico's version, the directory, the task
//!   when there is one (`--prompt`).
//! - `call` — one model call: `caller`, `model`, `request` (the body as sent, token
//!   excluded — it is a header, never in the body), then `response` (the assistant message,
//!   reasoning included) or `error`, and `elapsed_ms`. A call cut off by Esc or Ctrl+C
//!   never finished and is not recorded: there is no reply to learn from.
//! - `end` — once, on the way out, with `exit_code` for a `--prompt` run (0 is the run
//!   that finished — the label a trainer filters on) and none for the REPL.
//!
//! Global rather than a field of the client, for `debug.rs`'s reason and one more: the
//! client is cloned into the tool server, copied for every sub-agent and rebuilt by
//! `/model`, and a trace that one of those copies forgot would be a trace with a hole in
//! it that nothing reports.
//!
//! `--trace-lib <file>` is the other recording, and a much narrower one: each library
//! lookup as the question and the librarian's final answer, two messages and nothing else,
//! already in the shape a fine-tuning run reads. It exists for teaching a model what the
//! library says so it stops needing to ask, which the whole trace can also give — through
//! `scripts/trace_to_dataset.py --qa`, after keeping every request of every run. The
//! filters are the same ones that script applies, and the two sets of patterns must agree.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::llm::Ending;

/// A call that came back: the assistant message as the server gave it — for a stream, as
/// reassembled — and how it ended.
pub(crate) struct Reply<'a> {
    pub(crate) message: &'a Value,
    pub(crate) ending: &'a Ending,
}

/// Which conversation a call belongs to. An argument at every call site rather than a
/// label on the client, so a new call cannot be recorded as the main conversation by
/// forgetting to say otherwise — the one mislabeling that would quietly poison a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Caller {
    /// The conversation itself — what the user, or a `--prompt` task, is talking to.
    Main,
    /// The `research` tool's own loop.
    Research,
    /// The `library` tool's own loop.
    Library,
    /// The `sub_agent` tool's own loop.
    SubAgent,
    /// Auto mode judging a gated call.
    Reviewer,
    /// The reasoning policy compressing a chain of thought.
    Compressor,
    /// `/compact` and auto-compaction summarizing the history.
    Compactor,
    /// `web_fetch` extracting what was asked from a page.
    Summarizer,
}

impl Caller {
    fn name(self) -> &'static str {
        match self {
            Caller::Main => "main",
            Caller::Research => "research",
            Caller::Library => "library",
            Caller::SubAgent => "sub_agent",
            Caller::Reviewer => "reviewer",
            Caller::Compressor => "compressor",
            Caller::Compactor => "compactor",
            Caller::Summarizer => "summarizer",
        }
    }
}

/// An open trace file and the id every line of it carries.
///
/// The id is what tells runs apart once several have been appended to one file — the
/// usual way to use this is one file for a whole night of scheduled runs.
pub(crate) struct Trace {
    file: Mutex<File>,
    run: String,
}

impl Trace {
    pub(crate) fn open(path: &Path) -> Result<Trace> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("could not create {}", dir.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("could not open the trace file {}", path.display()))?;
        let run = format!(
            "{}-{}",
            chrono::Local::now().format("%Y%m%dT%H%M%S%.3f"),
            std::process::id()
        );
        Ok(Trace { file: Mutex::new(file), run })
    }

    /// One line, written whole under the lock so two calls finishing together cannot
    /// interleave. Failures are swallowed for `debug::log`'s reason: a run should not
    /// die, or scribble into a raw-mode terminal, over a line of its own recording.
    fn write(&self, mut line: Value) {
        line["run"] = json!(self.run);
        line["time"] = json!(chrono::Local::now().to_rfc3339());
        let Ok(mut text) = serde_json::to_string(&line) else { return };
        text.push('\n');
        let mut file = self.file.lock().unwrap_or_else(|p| p.into_inner());
        let _ = file.write_all(text.as_bytes());
        let _ = file.flush();
    }

    fn start(&self, cwd: &Path, task: Option<&str>) {
        self.write(json!({
            "type": "start",
            "haico": env!("CARGO_PKG_VERSION"),
            "cwd": cwd.display().to_string(),
            "task": task,
        }));
    }

    fn call(
        &self,
        caller: Caller,
        request: &Value,
        outcome: Result<Reply<'_>, String>,
        elapsed: Duration,
    ) {
        let mut line = json!({
            "type": "call",
            "caller": caller.name(),
            "model": request.get("model").cloned().unwrap_or(Value::Null),
            "request": request,
            "elapsed_ms": elapsed.as_millis() as u64,
        });
        match outcome {
            Ok(reply) => {
                line["response"] = reply.message.clone();
                line["finish_reason"] = json!(reply.ending.finish_reason);
                line["usage"] = reply.ending.usage.clone().unwrap_or(Value::Null);
            }
            Err(error) => line["error"] = json!(error),
        }
        self.write(line);
    }

    fn end(&self, exit_code: Option<i32>) {
        self.write(json!({ "type": "end", "exit_code": exit_code }));
    }
}

static TRACE: OnceLock<Trace> = OnceLock::new();

/// Start recording into `path`, appending. Once per process; called before the first
/// request so the `start` line is the first line of the run.
pub(crate) fn init(path: &Path, cwd: &Path, task: Option<&str>) -> Result<()> {
    let trace = Trace::open(path)?;
    trace.start(cwd, task);
    let _ = TRACE.set(trace);
    Ok(())
}

/// Whether anything is being recorded, so callers can skip cloning a request body that
/// nobody will read.
pub(crate) fn enabled() -> bool {
    TRACE.get().is_some()
}

/// Record one finished call, or the error it failed with.
pub(crate) fn call(caller: Caller, request: &Value, outcome: Result<Reply<'_>, String>, elapsed: Duration) {
    if let Some(trace) = TRACE.get() {
        trace.call(caller, request, outcome, elapsed);
    }
}

/// The last line of the run.
pub(crate) fn end(exit_code: Option<i32>) {
    if let Some(trace) = TRACE.get() {
        trace.end(exit_code);
    }
}

/// One of the question-and-answer files (`--trace-lib`, `--trace-research`,
/// `--trace-gaps`), and the pairs already written to it by this process.
struct QaFile {
    inner: Mutex<(File, HashSet<(String, String)>)>,
}

impl QaFile {
    fn open(path: &Path) -> Result<QaFile> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("could not create {}", dir.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("could not open the trace file {}", path.display()))?;
        Ok(QaFile { inner: Mutex::new((file, HashSet::new())) })
    }

    /// One line for this question and answer, unless the exact pair was written already — a
    /// model stuck on one error asks the same thing again and gets the same answer, and a
    /// dataset of forty copies of it teaches that one answer forty times over. Failures
    /// swallowed, as for `write`.
    fn record(&self, question: &str, answer: &str, line: impl FnOnce(&str, &str) -> Value) {
        let (question, answer) = (question.trim(), answer.trim());
        if question.is_empty() || answer.is_empty() {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if !inner.1.insert((question.to_string(), answer.to_string())) {
            return;
        }
        let Ok(mut text) = serde_json::to_string(&line(question, answer)) else { return };
        text.push('\n');
        let _ = inner.0.write_all(text.as_bytes());
        let _ = inner.0.flush();
    }

    /// As an SFT example: the question from the user, the answer from the assistant.
    fn record_example(&self, question: &str, answer: &str) {
        self.record(question, answer, |question, answer| {
            json!({"messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]})
        });
    }
}

/// Why a librarian's answer is no use as something to say from memory, if it is not.
///
/// "The library does not cover this" is right with the library behind it and exactly wrong
/// learned on its own. An answer pointing at its source — "the article", "I read", a section
/// id like `tokio/select#2` — teaches citing something the model never had. "the library" is
/// in there although a crate is a library too: the librarian says it of its own shelves far
/// more often than of a crate. Both patterns are kept in step with `scripts/trace_to_dataset.py`;
/// the first lives with the librarian, which uses it too.
fn qa_rejection(answer: &str) -> Option<&'static str> {
    static REFERS_TO_SOURCE: OnceLock<regex::Regex> = OnceLock::new();
    let refers_to_source = REFERS_TO_SOURCE.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\bthe articles?\b|\bthis article\b|\bthe library\b|\b(?:I|we) (?:have )?read\b|\bsections? (?:I|you) read\b|\b[\w.-]+/[\w.-]+#\d+\b",
        )
        .unwrap()
    });
    if answer.is_empty() {
        Some("empty")
    } else if crate::tools::library::says_not_covered(answer) {
        Some("the library did not cover it")
    } else if refers_to_source.is_match(answer) {
        Some("it refers to the articles")
    } else {
        None
    }
}

static LIBRARY_QA: OnceLock<QaFile> = OnceLock::new();
static RESEARCH_QA: OnceLock<QaFile> = OnceLock::new();
static LIBRARY_GAPS: OnceLock<QaFile> = OnceLock::new();

/// Start recording library answers into `path`, appending. Once per process.
pub(crate) fn init_library(path: &Path) -> Result<()> {
    let _ = LIBRARY_QA.set(QaFile::open(path)?);
    Ok(())
}

/// Start recording research answers into `path`, appending. Once per process.
pub(crate) fn init_research(path: &Path) -> Result<()> {
    let _ = RESEARCH_QA.set(QaFile::open(path)?);
    Ok(())
}

/// Start recording the questions the library could not answer into `path`. Once per process.
pub(crate) fn init_gaps(path: &Path) -> Result<()> {
    let _ = LIBRARY_GAPS.set(QaFile::open(path)?);
    Ok(())
}

/// One finished library lookup: the question as asked and the librarian's whole answer.
///
/// An answer saying the library had nothing is not an example, but it is the most useful
/// thing the librarian says for whoever writes the articles: a question a model really had
/// in the middle of real work, that the shelves could not answer. That goes to the gaps
/// file, with the answer, since "covers X but not Y" says which half is missing.
pub(crate) fn library_answer(question: &str, answer: &str) {
    record_library_answer(LIBRARY_QA.get(), LIBRARY_GAPS.get(), question, answer);
}

// The files as arguments, so a test can hand in its own: the statics are set once per
// process, and the tests share one.
fn record_library_answer(examples: Option<&QaFile>, gaps: Option<&QaFile>, question: &str, answer: &str) {
    if crate::tools::library::says_not_covered(answer) {
        if let Some(file) = gaps {
            file.record(question, answer, |question, answer| {
                json!({
                    "time": chrono::Local::now().to_rfc3339(),
                    "cwd": std::env::current_dir().map(|d| d.display().to_string()).unwrap_or_default(),
                    "question": question,
                    "answer": answer,
                })
            });
        }
    }
    if let Some(file) = examples {
        match qa_rejection(answer.trim()) {
            Some(reason) => crate::debug::log("library answer not traced", reason),
            None => file.record_example(question, answer),
        }
    }
}

/// One finished research call: the question and the answer the sub-agent gave.
///
/// Unfiltered, unlike the library's: what a page on the web said is not checked the way an
/// article is, so this file is raw material for someone to curate — which also means there
/// is no pattern here that could know what to leave out. The answer is the sub-agent's own
/// words only; the pages it read go beside the messages as `sources`, for whoever checks the
/// answer against them — a key trainers ignore, and one to strip once it is checked.
pub(crate) fn research_answer(question: &str, answer: &str, sources: &[String]) {
    if let Some(file) = RESEARCH_QA.get() {
        file.record(question, answer, |question, answer| {
            json!({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "sources": sources,
            })
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory in this thread's scratch area — `scratch` hands out files.
    fn temp_dir(name: &str) -> std::path::PathBuf {
        let marker = crate::testutil::scratch(&format!("{name}.marker"), "");
        let dir = marker.parent().unwrap().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn lines(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).expect("every line is one JSON object"))
            .collect()
    }

    #[test]
    fn a_run_is_recorded_as_start_calls_and_end() {
        let dir = temp_dir("trace_run");
        let path = dir.join("nested").join("trace.jsonl");
        let trace = Trace::open(&path).unwrap();
        trace.start(&dir, Some("update deps"));

        let request = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": []});
        let reply = json!({"role": "assistant", "content": "hello", "reasoning_content": "greet"});
        let ending = Ending { finish_reason: Some("stop".into()), usage: Some(json!({"completion_tokens": 7})) };
        trace.call(Caller::Main, &request, Ok(Reply { message: &reply, ending: &ending }), Duration::from_millis(1500));
        trace.call(Caller::Reviewer, &request, Err("server down".into()), Duration::ZERO);
        trace.end(Some(0));

        let seen = lines(&path);
        assert_eq!(seen.len(), 4);
        assert!(seen.iter().all(|l| l["run"] == json!(trace.run)), "every line names its run");
        assert_eq!(seen[0]["task"], "update deps");
        assert_eq!(seen[1]["caller"], "main");
        assert_eq!(seen[1]["request"], request, "the request is kept exactly as sent");
        assert_eq!(seen[1]["response"]["reasoning_content"], "greet");
        assert_eq!(seen[1]["elapsed_ms"], 1500);
        assert_eq!(seen[1]["finish_reason"], "stop");
        assert_eq!(seen[1]["usage"]["completion_tokens"], 7);
        assert_eq!(seen[2]["caller"], "reviewer");
        assert_eq!(seen[2]["error"], "server down");
        assert!(seen[2].get("response").is_none());
        assert_eq!(seen[3], json!({"type": "end", "exit_code": 0, "run": seen[3]["run"], "time": seen[3]["time"]}));
    }

    #[test]
    fn a_second_run_appends_rather_than_replacing() {
        let dir = temp_dir("trace_append");
        let path = dir.join("trace.jsonl");
        Trace::open(&path).unwrap().end(None);
        Trace::open(&path).unwrap().end(None);
        assert_eq!(lines(&path).len(), 2);
    }

    #[test]
    fn library_answers_are_written_as_sft_pairs_once_each() {
        let dir = temp_dir("trace_lib");
        let (path, gaps_path) = (dir.join("lib.jsonl"), dir.join("gaps.jsonl"));
        let (file, gaps) = (QaFile::open(&path).unwrap(), QaFile::open(&gaps_path).unwrap());
        let answer = |q: &str, a: &str| record_library_answer(Some(&file), Some(&gaps), q, a);
        answer("  tokio: select in a loop?  ", "Pin it with `tokio::pin!` first.");
        answer("tokio: select in a loop?", "Pin it with `tokio::pin!` first.");
        answer("mio: Registry?", "The library does not cover mio's Registry.");
        answer("mio: Registry?", "The library does not cover mio's Registry.");
        answer("mio: Registry?", "The article shows `registry.register`.");
        answer("mio: Registry?", "See mio/net#1 for `register`.");
        answer("mio: Registry?", "");
        answer("", "An answer to nothing.");

        assert_eq!(lines(&path), vec![json!({"messages": [
            {"role": "user", "content": "tokio: select in a loop?"},
            {"role": "assistant", "content": "Pin it with `tokio::pin!` first."},
        ]})]);

        // The one question the library had nothing on, once, and nothing else.
        let gaps = lines(&gaps_path);
        assert_eq!(gaps.len(), 1, "{:?}", gaps);
        assert_eq!(gaps[0]["question"], "mio: Registry?");
        assert_eq!(gaps[0]["answer"], "The library does not cover mio's Registry.");
    }
}
