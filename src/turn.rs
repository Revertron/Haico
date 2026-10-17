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
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};

use crate::modes::{Mode, PlanDecision};
use crate::permissions::PermissionResponse;

/// Where a turn sends what it has to say. Unbounded because the alternative is a
/// turn that blocks on its own narration: the frontend is busy drawing and
/// answering prompts, and a full channel would stall the model mid-stream.
pub(crate) type EventSink = mpsc::UnboundedSender<AgentEvent>;

/// The state a running turn and its frontend genuinely share — and the only such
/// state there is. Three things, each shared because *both* sides move it:
///
/// - **the mode**, which the user cycles with Shift+Tab while the model works,
///   and the model changes by getting a plan approved;
/// - **the queue** of input typed during the turn, which the frontend collects
///   and shows above the input box and the turn drains at a round boundary —
///   never sooner, because a user message between an assistant turn and the tool
///   results it is waiting for is an order servers reject outright;
/// - **cancellation**, which is Esc.
///
/// Everything else flows one way: events out through [`EventSink`], answers back
/// through their own reply channel. This is deliberately tiny. It is the part of
/// the old design where the loop reached into `main`'s locals, and it is the part
/// most likely to grow back if it is not kept obviously minimal.
#[derive(Clone)]
pub(crate) struct TurnLink {
    shared: Arc<Mutex<Shared>>,
    /// Cancellation has its own channel rather than a flag in `shared`, because
    /// the turn has to *await* it — several places select against it — and a
    /// bool cannot be awaited.
    cancel: watch::Sender<bool>,
}

struct Shared {
    mode: Mode,
    queued: Vec<String>,
}

impl TurnLink {
    pub(crate) fn new(mode: Mode) -> Self {
        TurnLink {
            shared: Arc::new(Mutex::new(Shared { mode, queued: Vec::new() })),
            cancel: watch::channel(false).0,
        }
    }

    pub(crate) fn mode(&self) -> Mode {
        self.shared.lock().unwrap().mode
    }

    pub(crate) fn set_mode(&self, mode: Mode) {
        self.shared.lock().unwrap().mode = mode;
    }

    /// Text typed while the turn was running, still unsent.
    pub(crate) fn queue(&self, text: String) {
        self.shared.lock().unwrap().queued.push(text);
    }

    /// What is waiting, for drawing it above the input box.
    pub(crate) fn queued(&self) -> Vec<String> {
        self.shared.lock().unwrap().queued.clone()
    }

    /// Take what is waiting, leaving the queue empty.
    pub(crate) fn take_queued(&self) -> Vec<String> {
        std::mem::take(&mut self.shared.lock().unwrap().queued)
    }

    pub(crate) fn cancel(&self) {
        // `send_replace`, not `send`. `send` *fails and does not store the value*
        // when there are no receivers, and there usually are none: the turn only
        // subscribes once it reaches an interruptible point, so an Esc that
        // arrives before then — or between two of them — would set nothing and be
        // silently lost. This always stores, so a later `subscribe` still sees it.
        self.cancel.send_replace(true);
    }

    /// Resolves once the turn has been cancelled — the thing the turn selects
    /// against instead of being dropped.
    ///
    /// **Cancelling by dropping the turn's future would be wrong**, which is why
    /// this exists. The future can be suspended between pushing an assistant
    /// message and pushing the `role: "tool"` replies its tool calls are owed,
    /// and a drop there would leave a `tool_call_id` unanswered — the one thing
    /// that makes the *next* request fail server-side. Selecting against this
    /// instead means every cancellation lands somewhere the turn chose, with the
    /// history already balanced.
    pub(crate) async fn until_cancelled(&self) {
        let mut rx = self.cancel.subscribe();
        loop {
            // Checked before waiting: `changed()` only reports transitions after
            // the subscribe, so a turn cancelled before it reached its first
            // select must still see it.
            if *rx.borrow_and_update() {
                return;
            }
            if rx.changed().await.is_err() {
                // Every sender is gone, so nothing can cancel this any more.
                // Pending rather than resolving: a dropped frontend must not be
                // mistaken for the user pressing Esc.
                std::future::pending::<()>().await;
            }
        }
    }
}

/// Run `work`, giving it up if the user cancels. `None` means cancelled.
///
/// This is what the five `work_with_input` calls became. Each of those wrapped
/// the frontend's whole draw-and-read loop around one piece of the turn, which is
/// why there were five of them and why the turn had to hold the keyboard. The
/// pump now sits outside the entire turn, so all that is left here is the
/// choice — finish, or stop — and the label to say which kind of waiting this is.
pub(crate) async fn interruptible<T>(
    tx: &EventSink,
    link: &TurnLink,
    waiting: Waiting,
    work: impl std::future::Future<Output = T>,
) -> Option<T> {
    let _ = tx.send(AgentEvent::Busy(waiting));
    tokio::select! {
        output = work => Some(output),
        _ = link.until_cancelled() => None,
    }
}

/// Ask the frontend something and wait for the answer, or `None` if it cannot
/// answer at all.
///
/// `None` covers both "there is no interactive terminal" and "the frontend is
/// gone", because the turn must treat them the same: as a refusal. Every caller
/// still owes its `tool_call_id` a reply, so none of them may use `?` here.
pub(crate) async fn ask_frontend<T>(
    tx: &EventSink,
    make: impl FnOnce(oneshot::Sender<T>) -> Ask,
) -> Option<T> {
    let (reply_tx, reply_rx) = oneshot::channel();
    if tx.send(AgentEvent::Ask(make(reply_tx))).is_err() {
        return None;
    }
    reply_rx.await.ok()
}

/// A question the turn cannot answer, with the channel to answer it on.
///
/// The reply channel is the whole design. A turn awaits it, so the frontend can
/// take as long as it likes — and if the sender is **dropped** the turn hears
/// that as a refusal rather than hanging. That matters more than it looks: a
/// window closed over an open permission prompt, or a frontend that panicked,
/// must deny rather than leave a `tool_call_id` unanswered.
pub(crate) enum Ask {
    /// A plan is ready for approval.
    Plan { plan: String, reply: oneshot::Sender<PlanDecision> },
    /// The model asked the user a question through the `ask` tool.
    User {
        question: String,
        options: Vec<String>,
        reply: oneshot::Sender<crate::ask::Answer>,
    },
    /// A gated call needs consent.
    ///
    /// The call itself, not a sentence about it. Describing a pending write as
    /// "Write 412 bytes to file: src/main.rs" is a rendering decision — how much
    /// of the command to show, how to lay it out, whether to put the diff above
    /// or below — and it was the last piece of that decision left in the core.
    /// `allow_all_label` stays a string because it is the *gate's* own name for
    /// the scope being granted, which is policy rather than presentation.
    Permission {
        tool_name: String,
        arguments: Value,
        /// What the auto-mode reviewer said, when it declined to wave this
        /// through. `None` when there was no reviewer or it approved.
        review: Option<ReviewNote>,
        allow_all_label: String,
        diff: Option<crate::diff::Diff>,
        reply: oneshot::Sender<PermissionResponse>,
    },
}

/// Why a gated call reached the user despite auto mode being on.
///
/// A reviewer's no never gets here — it refuses the call on its own (see
/// [`Notice::ReviewDenied`]) — and neither does a review that failed to produce a
/// verdict, which is reported to the model as a temporary error to retry (see
/// [`Notice::ReviewFailed`]): auto mode is for walking away, and a prompt nobody is
/// there to answer parks the whole run. What is left is the user pressing Esc
/// during a review, which is the one case that proves somebody is there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReviewNote {
    /// Esc arrived while the review was running.
    Interrupted,
}

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
    /// A request failed while the history carried a picture, so the pictures
    /// were taken out and the request tried again. Carries the server's own
    /// complaint: it is the evidence for the guess that the image was the
    /// problem, and if the retry fails too it is the only clue left.
    ImagesStripped(String),
    /// The round cap stopped a turn that never reached a final answer. Carries
    /// the cap, since the number is the whole point of saying so.
    RoundCapReached(usize),
    /// Auto-compaction ran; the string is the summarizer's own report.
    Compacted(String),
    /// Compaction ran and the result is still over the threshold, so it will not
    /// be tried again this turn. Carries the percentage of the window.
    CompactionStillLarge(usize),
    CompactionFailed(String),
    /// The conversation could not be saved at the end of the turn, so `--resume`
    /// would bring back an older one. Carries the reason.
    SessionNotSaved(String),
    /// The policy replaced a chain of thought with a summary of it. The figures
    /// say a compression happened; `summary` is what came back, and it travels
    /// with them because by the time this is read the original is gone from the
    /// history — there is nowhere else left to look at what the trace became.
    ReasoningCompressed { before: usize, after: usize, summary: String },
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
    /// The auto-mode reviewer judged the call unsafe, and it was refused without
    /// asking. The call travels with the reason, because the user reading back
    /// through an unattended run needs to see what was stopped, not only why.
    ReviewDenied { tool: String, arguments: Value, reason: String },
    /// The auto-mode reviewer gave no verdict (cut off at its token cap, an
    /// unreadable reply, a server error). The call was not run, and the model was
    /// told to make it again rather than the user being asked.
    ReviewFailed { tool: String, arguments: Value, detail: String },
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
    /// The sub-agent is about to run one of its own calls, which has cleared every
    /// gate. Its own kind rather than `ToolStarted`, because it is one step inside
    /// a call that is already on screen, not a call of the conversation's.
    SubAgentStep { tool: String, arguments: Value },
    /// A reply was cut off mid-stream for going round in a loop, and dropped.
    /// `retrying` says whether the model has been told and asked again, or the
    /// turn has given up; `passage` is a line of what kept coming back.
    Repeating { passage: String, retrying: bool },
    /// The request failed on the server's side (a 5xx, a rate limit, an error in the
    /// middle of the stream, a dropped connection) and will be sent again after `delay`.
    /// Any part of the reply already shown belonged to the failed attempt and was dropped.
    /// `retry` counts from 1, of `of` at most; once they run out the turn ends with
    /// `RequestFailed` as before.
    Retrying { error: String, retry: usize, of: usize, delay: std::time::Duration },
    /// A file the message named with `@` was read into it before the request
    /// went out. `outside` marks the ones that were beyond the workspace and
    /// widened it, which is the half worth reading twice.
    FileAttached { path: String, lines: usize, outside: bool },
    /// The same, for an image, which goes into the message as a data URL rather
    /// than as text. Bytes and not lines: an image has none, and its size is the
    /// figure worth showing because it is what the request will carry.
    ImageAttached { path: String, bytes: usize, outside: bool },
    /// …and one that could not be read, so the message carries the name alone.
    /// Worth saying: silence here looks exactly like a file that was attached.
    FileNotAttached { path: String, reason: String },
    /// Text typed while the model was working, now joining the conversation.
    QueuedSent(String),
    /// Timers the model set came due and their prompts were sent to it — mid-turn at a
    /// round boundary, or at the idle prompt as a turn of their own.
    TimersFired(Vec<crate::timers::Fired>),
    /// `/init` wrote the project's instructions and they were re-read.
    ProjectContextReloaded(String),
}

/// One thing the turn has to say. Everything the loop used to print, plus the
/// three things it needs the frontend to *do*.
///
/// One channel rather than two — events on one and questions on the other —
/// because order matters and a single queue gives it for free: a refusal has to
/// appear before the next call's prompt, and two channels have no relative
/// ordering. The cost is that [`Ask`] carries a reply channel, which is neither
/// `Clone` nor `Debug`, so this enum can derive nothing. [`Notice`] keeps those
/// derives, which is where tests want them anyway.
pub(crate) enum AgentEvent {
    /// A new reply is starting. A turn is many rounds and each streams one, so
    /// this is what tells a frontend holding per-reply state — a half-rendered
    /// markdown line, a fold — to start again. Sent by the turn rather than
    /// inferred from the first chunk, because a reply can produce no chunks at
    /// all and still need the reset.
    ReplyStart,
    /// A piece of the model's chain of thought, as it arrives.
    ReasoningChunk(String),
    /// A piece of the model's answer, as it arrives. Raw markdown: rendering it
    /// is the frontend's business, and a streaming renderer is the only thing
    /// that can decide when a line has settled enough to wrap.
    ReplyChunk(String),
    /// How many characters of a tool call have just arrived. A count and not the
    /// text, for the reason given on `llm::Chunk::ToolFragment`: the call itself
    /// is shown whole by [`Self::ToolStarted`] once it is about to run. This is
    /// only so a frontend with a progress counter can keep it moving through the
    /// stretch of a reply that puts nothing else on the screen.
    ToolCallChunk(usize),
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
    ToolDiff(crate::diff::Diff),
    /// A finished tool's output, whole.
    ToolResult(String),
    /// A tool ran, whatever it returned. Separate from [`Self::ToolResult`]
    /// because it is sent even for the tools whose result is not shown, and
    /// because it feeds different chrome: the row above the input box, which says
    /// what happened last once the turn is over. A failed build is still the last
    /// thing that happened, and the bar saying so is the point.
    DidTool { name: String, arguments: Value },
    Notice(Notice),
    /// What kind of waiting this is. Sent whenever the turn moves to a different
    /// one; replaces the `busy` argument the five `work_with_input` calls used to
    /// carry, which is what made them five.
    Busy(Waiting),
    /// The numbers behind the status bar, sent when the history changes. The
    /// numbers rather than the line: the turn can count its own messages and
    /// tokens, but how wide the bar is and what it puts where is the frontend's.
    Status {
        /// The configured provider the model is served by, when the config names one.
        provider: Option<String>,
        model: String,
        used_tokens: usize,
        context_tokens: usize,
        /// The thinking level, when one is set. Carried with the numbers rather than read
        /// off the agent by the frontend, for the same reason the model name is: while a
        /// turn is running the frontend cannot look at the agent at all.
        effort: Option<String>,
    },
    /// A question, and the channel to answer it on. Not a `Notice`: this is the
    /// one kind of event the frontend must *do* something about.
    Ask(Ask),
    /// The turn is over, however it ended. The duration is what the summary is
    /// for; the wording around it is the frontend's.
    Finished { elapsed: Duration },
}

/// What a turn is waiting on. A kind rather than a label, because the label is
/// flavor text — haico picks a different one each time from a long list — and
/// choosing words is not something the core should be doing. It also means a GUI
/// can show a spinner and an icon per kind instead of parsing a sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Waiting {
    /// The model is generating. The only kind with a stream behind it, so the
    /// only one the character counter means anything for.
    Model,
    Compacting,
    CompressingReasoning,
    ReviewingCall,
    /// A named tool is running.
    Tool(String),
}

/// How a turn ended, handed back by `run_turn` itself.
///
/// The REPL has no use for it — whatever happened, the prompt comes back and the
/// user reads the transcript. It exists for the one caller with nobody reading:
/// a `--prompt` run has to decide on its own whether to exit, carry on, or report
/// a failure, and inferring that from which notices happened to go past would be
/// guessing at the loop's control flow from its narration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnEnd {
    /// The model stopped calling tools and replied.
    Answered,
    /// Cancelled — Esc in the REPL, Ctrl+C in a `--prompt` run.
    Interrupted,
    /// Ended without an answer: the request failed, the reply was malformed, the
    /// round cap was hit, or the loop guard gave up on a repeated call.
    Failed,
}

impl From<Notice> for AgentEvent {
    fn from(notice: Notice) -> Self {
        AgentEvent::Notice(notice)
    }
}
