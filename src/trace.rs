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

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
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
}
