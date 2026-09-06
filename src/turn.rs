//! What a turn says, as values rather than as text.
//!
//! The turn loop used to print. Forty-five `out!` sites, interleaved with the
//! history maintenance and the tool dispatch, each one deciding for itself what
//! a thing should look like on a terminal. That is fine while there is exactly
//! one frontend and fatal the moment there are two: a GUI does not want a line
//! of SGR escapes, it wants to know that a tool started.
//!
//! So the loop emits these and something else renders them. Two rules keep the
//! split honest:
//!
//! - **Structure survives, formatting does not.** [`AgentEvent::ToolDiff`]
//!   carries the diff, not a preview of the first few lines of it — the terminal
//!   makes a preview because a terminal has a height, and a GUI would fold the
//!   whole thing. Nothing here has been through `render.rs`.
//! - **A [`Notice`] is a thing that happened, not a sentence.** They are the
//!   long tail — a round cap reached, a plan sent back, a call refused — and the
//!   temptation is to make them one `Line(String)` variant and be done. That is
//!   a sink with extra steps: it puts the wording in the core, where the second
//!   frontend cannot change it and the core has to know a terminal's width to
//!   choose it.
//!
//! The one thing deliberately *not* here is anything the frontend needs to
//! answer. Prompts still go through the frontend directly (`ask_about_call` and
//! friends); turning those into events carrying a reply channel is the next step
//! and needs the pump inverted first.

use serde_json::Value;
use std::time::Duration;

/// Something that happened during a turn, worth one line and carrying no
/// structure beyond its own fields.
///
/// Split out from [`AgentEvent`] rather than flattened into it because these are
/// the cases where the frontend is choosing *wording*, and there are two dozen of
/// them; the variants above are the ones where it is choosing a *layout*.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Notice {
    /// Esc during the turn. Every path that reports this has already left the
    /// history consistent.
    Interrupted,
    /// The request failed and the turn is over, but the conversation is intact.
    RequestFailed(String),
    /// The round cap stopped a turn that never reached a final answer. Carries
    /// the cap, since the number is the whole point of saying so.
    RoundCapReached(usize),
    /// Auto-compaction ran; the string is the summarizer's own report.
    Compacted(String),
    /// Compaction ran and the result is still over the threshold, so it will not
    /// be tried again this turn. Carries the percentage of the window.
    CompactionStillLarge(usize),
    CompactionFailed(String),
    ReasoningCompressed { before: usize, after: usize },
    ReasoningDropped { tokens: usize },
    /// `tool_calls` was present but not an array.
    ToolCallsMalformed(String),
    /// A tool call with no function name. Carries the detail already assembled
    /// for the model, since the same words serve both.
    ToolCallUnnamed(String),
    ToolArgumentsInvalid { tool: String, detail: String },
    ToolFailed(String),
    /// The model put itself into plan mode.
    PlanModeEntered,
    /// The user approved a plan. `auto` distinguishes the two approvals, which
    /// say different things because auto mode needs spelling out.
    PlanApproved { auto: bool },
    PlanRejected,
    /// A prompt could not be shown at all — no interactive terminal. `what` names
    /// which prompt, since all three failure paths read alike otherwise.
    AskFailed { what: &'static str, detail: String },
    /// The gate refused the call outright, with no prompt: outside the workspace,
    /// or a write in plan mode.
    Refused(String),
    PermissionAllowed,
    /// Allowed for the rest of the session, under the narrow key it resolved to.
    /// Carries the label so the frontend says what was actually granted.
    PermissionAllowedAll(String),
    PermissionDenied { alternative: String },
    /// The same call has been made this many times in a row; the model has been
    /// told to stop.
    LoopNudge { tool: String, count: usize },
    /// …and this is where the turn gave up on it.
    LoopBroken { tool: String, count: usize },
    /// Text typed while the model was working, now joining the conversation.
    QueuedSent(String),
    /// `/init` wrote the project's instructions and they were re-read.
    ProjectContextReloaded(String),
}

/// One thing the turn has to say. Everything the loop used to print.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AgentEvent {
    /// A piece of the model's chain of thought, as it arrives.
    ReasoningChunk(String),
    /// A piece of the model's answer, as it arrives. Raw markdown: rendering it
    /// is the frontend's business, and a streaming renderer is the only thing
    /// that can decide when a line has settled enough to wrap.
    ReplyChunk(String),
    /// The reply is over, however it ended — finished, interrupted or failed.
    /// Sent on every one of those paths so a frontend holding open state (a
    /// markdown stream mid-line, a fold) has exactly one place to close it.
    ReplyEnd,
    /// A tool is about to run. The arguments are the arguments, not a rendering
    /// of them; `auto_note` is auto mode's reason, when it approved this call.
    ToolStarted {
        name: String,
        arguments: Value,
        auto_note: Option<String>,
    },
    /// The change a file-writing call is about to make, as a whole diff. Whether
    /// to show a few lines of it or all of it is a question about the display.
    ToolDiff(String),
    /// A finished tool's output, whole.
    ToolResult(String),
    Notice(Notice),
    /// The turn is over. `verb` is the flavor word; the duration is what the
    /// summary is actually for.
    Finished { verb: &'static str, elapsed: Duration },
}

impl From<Notice> for AgentEvent {
    fn from(notice: Notice) -> Self {
        AgentEvent::Notice(notice)
    }
}
