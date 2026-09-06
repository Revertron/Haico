//! The second consumer of the event stream: a Lumio window.
//!
//! This exists to prove the seam more than to be finished. Everything below
//! reads the same `AgentEvent`s the terminal reads and answers the same `Ask`s,
//! and it shares no rendering code with `present.rs` at all — which is the only
//! real test of whether the vocabulary in `turn.rs` describes *what happened*
//! rather than what a terminal wanted to print. Where a variant turns out to be
//! terminal-shaped, that is a bug in the vocabulary, and this is where it shows.
//!
//! ## The threading, which is forced rather than chosen
//!
//! winit owns the main thread and `lumio::run` blocks on it until the last
//! window closes, so the conversation cannot live there. It runs on a worker
//! thread with its own Tokio runtime — `Agent` is `Send`, asserted at compile
//! time in `agent.rs` — and reaches the UI through Lumio's `UiHandle`, whose
//! queued closures run on the UI thread at the next tick.
//!
//! Input goes the other way through a plain channel. So the two threads share no
//! lock, only two queues, and neither can block the other: a dialog waiting for
//! an answer does not stop the model streaming, and a model streaming does not
//! stop the window redrawing.
//!
//! ## What is deliberately unfinished
//!
//! The transcript is a `Frame` of `RichText` children inside a `ScrollView`
//! rather than a virtualized `RecyclerView`, which is fine for a session and
//! wrong for a long one. Code blocks are not syntax-highlighted; that wants
//! `syntect` emitting spans. And the permission dialog offers two answers where
//! the terminal offers three: "allow for the rest of the session" is a scope
//! decision a yes/no box cannot express, so it is not offered rather than being
//! quietly folded into "yes".

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use lumio::prelude::*;
use lumio::types::rect;

use crate::agent::{run_turn, Agent};
use crate::ask::Answer;
use crate::modes::PlanDecision;
use crate::permissions::PermissionResponse;
use crate::turn::{AgentEvent, Ask, Notice, TurnLink, Waiting};

const TRANSCRIPT: &str = "transcript";
const SCROLL: &str = "scroll";
const INPUT: &str = "input";
const STATUS: &str = "status";
const BUSY: &str = "busy";
const SEND: &str = "send";

/// The id the reply being streamed lives under, so later chunks can find it.
/// Cleared when the reply ends, so a chunk can never land in a closed one.
const OPEN_REPLY: &str = "reply-open";
const OPEN_REASONING: &str = "reasoning-open";

/// Text sizes in dip. Prose a little larger than the code beside it, because a
/// monospace face at the same nominal size reads bigger than a proportional one.
const PROSE: f32 = 15.0;
const CODE: f32 = 13.0;

const LAYOUT: &str = r#"
<Frame id="root" width="max" height="max" direction="vertical" padding="0">
    <ScrollView id="scroll" width="max" height="max" direction="vertical">
        <Frame id="transcript" width="max" height="min" direction="vertical" padding="12"/>
    </ScrollView>
    <Label id="busy" width="max" height="min" text="" margin_left="12" margin_top="2"/>
    <Frame width="max" height="min" direction="horizontal" padding="8">
        <Memo id="input" width="max" height="70"/>
        <Button id="send" text="Send" width="90" height="70" margin_left="8"/>
    </Frame>
    <Label id="status" width="max" height="min" text="starting…" margin_left="12" margin_bottom="4"/>
</Frame>
"#;

/// Open a window and run `agent` behind it until the window closes.
pub(crate) fn run(agent: Agent) -> Result<()> {
    let mut ui = UI::from_xml(LAYOUT, 1040, 760, default_typeface(), 1.0)
        .ok_or_else(|| anyhow::anyhow!("the GUI layout failed to parse"))?;

    // Submitted input, UI thread to worker. A plain std channel: the worker
    // blocks on it between turns, which is exactly what is wanted, and nothing
    // on the UI side ever waits for it.
    let (input_tx, input_rx) = mpsc::channel::<String>();

    // A Memo is multi-line, so Enter belongs to the text. Submitting is the
    // button, and Ctrl+Enter for anyone who would rather not reach for it.
    if let Some(button) = ui.get_view(SEND) {
        let tx = input_tx.clone();
        button.borrow_mut().on_event(
            EventType::Click,
            Box::new(move |ui, _view, _data| submit(ui, &tx)),
        );
    }
    let tx = input_tx.clone();
    ui.add_shortcut("Ctrl+Enter", Box::new(move |ui| submit(ui, &tx)));

    // The link of the turn in flight, if any, so closing the window can stop it.
    // Without this, shutting down waits for whatever the model is in the middle
    // of — a long build, a slow request — and the process outlives the window it
    // came from, which looks exactly like a window that will not close.
    let running: Arc<Mutex<Option<TurnLink>>> = Arc::new(Mutex::new(None));

    let handle = ui.handle();
    let worker_running = Arc::clone(&running);
    let worker = std::thread::Builder::new()
        .name("haico-agent".to_string())
        .spawn(move || {
            // `block_on` rather than `spawn`, so nothing here has to be
            // `'static + Send` beyond the agent itself.
            match tokio::runtime::Runtime::new() {
                Ok(runtime) => runtime.block_on(converse(agent, input_rx, handle, worker_running)),
                Err(e) => eprintln!("could not start the agent runtime: {e}"),
            }
        })?;

    // Dropped before the loop starts, and this is load-bearing. The handlers hold
    // their own clones, so the channel closes when the window drops them — but
    // only if this one is gone too. While it lives, the worker's `recv` can never
    // return and the join below never finishes.
    drop(input_tx);

    lumio::run(ui, WindowConfig::new("haico", 1040, 760).center().skin("dark"));

    // The window is gone. Stop the turn in flight rather than waiting it out;
    // cancellation lands somewhere the turn chose, with its history balanced.
    if let Some(link) = running.lock().unwrap().take() {
        link.cancel();
    }
    let _ = worker.join();
    Ok(())
}

/// Take whatever is in the input box and send it to the worker. Returns whether
/// a redraw is needed, which is Lumio's convention for an event handler.
fn submit(ui: &mut UI, tx: &mpsc::Sender<String>) -> bool {
    let Some(element) = ui.get_view(INPUT) else { return false };
    let text = {
        let view = element.borrow();
        let Some(memo) = view.as_any().downcast_ref::<Memo>() else { return false };
        let text = memo.get_text().trim().to_string();
        if text.is_empty() {
            return false;
        }
        memo.set_text("");
        text
    };
    let _ = tx.send(text);
    true
}

/// One turn per submitted line, until the window closes and the channel with it.
async fn converse(
    mut agent: Agent,
    input_rx: mpsc::Receiver<String>,
    handle: UiHandle,
    running: Arc<Mutex<Option<TurnLink>>>,
) {
    show(&handle, |ui| line(ui, "Ready. Type below, then Send or Ctrl+Enter.", Tone::Muted));

    while let Ok(input) = input_rx.recv() {
        let echoed = input.clone();
        show(&handle, move |ui| user(ui, &echoed));

        let link = TurnLink::new(agent.mode);
        // Published so a closing window can cancel it, and cleared below so a
        // finished turn is never cancelled after the fact.
        *running.lock().unwrap() = Some(link.clone());
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        {
            let turn = run_turn(&mut agent, &input, false, &tx, &link);
            tokio::pin!(turn);
            // The same shape as the terminal's pump, minus the keyboard: race the
            // turn against its own narration, then drain what is left.
            loop {
                tokio::select! {
                    _ = &mut turn => break,
                    event = events.recv() => match event {
                        Some(event) => dispatch(&handle, event),
                        None => break,
                    },
                }
            }
        }
        while let Ok(event) = events.try_recv() {
            dispatch(&handle, event);
        }
        agent.mode = link.mode();
        // Unpublished, so a window closing after this turn ended does not cancel
        // a link nothing is waiting on.
        running.lock().unwrap().take();
        show(&handle, |ui| set_label(ui, BUSY, ""));
    }
}

/// Hand one event to the window, or answer it if it is a question.
///
/// Not async, and it does not wait for a dialog: the *turn* is already suspended
/// on its reply channel, so nothing else can arrive on top of the question.
fn dispatch(handle: &UiHandle, event: AgentEvent) {
    match event {
        AgentEvent::Ask(ask) => ask_in_window(handle, ask),
        AgentEvent::Busy(waiting) => {
            let text = match waiting {
                Waiting::Model => "thinking…".to_string(),
                Waiting::Compacting => "compacting context…".to_string(),
                Waiting::CompressingReasoning => "compressing reasoning…".to_string(),
                Waiting::ReviewingCall => "reviewing the call…".to_string(),
                Waiting::Tool(name) => format!("{name}…"),
            };
            show(handle, move |ui| set_label(ui, BUSY, &text));
        }
        AgentEvent::Status { model, used_tokens, context_tokens } => {
            let percent = (used_tokens as f64 / context_tokens.max(1) as f64) * 100.0;
            let text = format!("{model}    ctx {percent:.0}%");
            show(handle, move |ui| set_label(ui, STATUS, &text));
        }
        AgentEvent::ReplyStart => show(handle, |ui| close_open(ui)),
        // Settled before the id is taken away, so the reply's last block is laid
        // out as finished markdown rather than left provisional.
        AgentEvent::ReplyEnd => show(handle, |ui| {
            finish_reply(ui);
            close_open(ui);
        }),
        AgentEvent::ReplyChunk(text) => show(handle, move |ui| stream_reply(ui, &text)),
        AgentEvent::ReasoningChunk(text) => {
            show(handle, move |ui| stream(ui, OPEN_REASONING, &text, true))
        }
        AgentEvent::ToolStarted { name, arguments, auto_note } => {
            let summary = crate::render::tool_call_summary(&name, &arguments);
            let text = match auto_note {
                Some(note) => format!("{name} {summary}   ({note})"),
                None => format!("{name} {summary}"),
            };
            show(handle, move |ui| code_line(ui, text.trim()));
        }
        AgentEvent::ToolDiff(diff) => show(handle, move |ui| block(ui, &diff)),
        AgentEvent::ToolResult(result) => show(handle, move |ui| block(ui, &result)),
        // The row above the terminal's input box has no counterpart here; the
        // busy label already says what is running.
        AgentEvent::DidTool { .. } => {}
        AgentEvent::Finished { elapsed } => {
            let text = format!("done in {}", crate::format_elapsed(elapsed));
            show(handle, move |ui| {
                set_label(ui, BUSY, "");
                line(ui, &text, Tone::Muted);
            });
        }
        AgentEvent::Notice(notice) => {
            let (text, tone) = describe(notice);
            show(handle, move |ui| line(ui, &text, tone));
        }
    }
}

/// Put a question in front of the user. The reply channel travels into the
/// dialog's callback, so whichever button is pressed answers the turn directly.
///
/// If the window closes with a dialog open, the callback is dropped, the sender
/// with it, and the turn hears a refusal. That is the one behaviour that must
/// never be a hang or a silent yes.
fn ask_in_window(handle: &UiHandle, ask: Ask) {
    match ask {
        Ask::Plan { plan, reply } => {
            let message = format!("{plan}\n\nStart work on this plan?");
            confirm(handle, "Plan ready", message, move |approved| match approved {
                true => PlanDecision::Approve,
                // No feedback box yet, so the model is told plainly that there is
                // none rather than handed an empty string it would read as assent.
                false => PlanDecision::Revise(
                    "The user declined the plan in the GUI without written feedback. \
                     Ask what to change."
                        .to_string(),
                ),
            }, reply);
        }
        Ask::User { question, options, reply } => {
            let prompt = match options.is_empty() {
                true => question,
                false => format!("{question}\n\nSuggested: {}", options.join("  /  ")),
            };
            let mut reply = Some(reply);
            handle.run_on_ui_thread(move |ui| {
                ui.show_input("haico asks", &prompt, "", move |_ui, answer| {
                    let Some(reply) = reply.take() else { return };
                    // Escaping the box is not an answer, and must not become one:
                    // the turn is told it was dismissed and decides for itself.
                    let _ = reply.send(match answer {
                        Some(text) if !text.trim().is_empty() => Answer::Typed(text),
                        _ => Answer::Dismissed,
                    });
                });
            });
        }
        Ask::Permission { tool_name, arguments, review, allow_all_label, diff, reply } => {
            // The description is built by `present.rs`, which is the terminal's
            // renderer — reused deliberately, because it is the one piece of
            // wording both frontends genuinely want the same. If a GUI ever wants
            // it laid out differently, this is the line that changes.
            let mut message =
                crate::present::permission_description(&tool_name, &arguments, review.as_ref());
            if let Some(diff) = &diff {
                message.push_str("\n\n");
                message.push_str(diff.trim_end());
            }
            // Not offered: `allow_all_label` names a session-wide scope, and a
            // yes/no box cannot ask for it without overstating what was granted.
            let _ = allow_all_label;
            confirm(handle, "Permission", message, move |allowed| match allowed {
                true => PermissionResponse::Allow,
                false => PermissionResponse::Reject(
                    "The user declined this call in the GUI.".to_string(),
                ),
            }, reply);
        }
    }
}

/// A yes/no dialog whose answer is mapped by `decide` and sent on `reply` once.
fn confirm<T: Send + 'static>(
    handle: &UiHandle,
    title: &'static str,
    message: String,
    decide: impl FnOnce(bool) -> T + Send + 'static,
    reply: tokio::sync::oneshot::Sender<T>,
) {
    let mut once = Some((decide, reply));
    handle.run_on_ui_thread(move |ui| {
        ui.show_confirm(title, &message, move |_ui, yes| {
            if let Some((decide, reply)) = once.take() {
                let _ = reply.send(decide(yes));
            }
        });
    });
}

/// Queue a closure to run on the UI thread. Every write to the window goes
/// through here, which is what keeps the worker from ever touching a Lumio type.
fn show(handle: &UiHandle, f: impl FnOnce(&mut UI) + Send + 'static) {
    handle.run_on_ui_thread(f);
}

/// How a line is coloured — the GUI's own vocabulary, not the turn's. The turn
/// says a plan was rejected; this decides that reads as a warning.
#[derive(Clone, Copy)]
enum Tone {
    Normal,
    Muted,
    Warning,
    Error,
}

impl Tone {
    fn color(self) -> Option<u32> {
        match self {
            Tone::Normal => None,
            Tone::Muted => Some(0xFF8A8A8A),
            Tone::Warning => Some(0xFFD8A03A),
            Tone::Error => Some(0xFFD86A6A),
        }
    }
}

/// The words for a notice.
///
/// Deliberately not the terminal's wording. This is the check that a `Notice`
/// carries a fact rather than a sentence: every one of these is phrased
/// differently here, and the turn did not change to allow it.
fn describe(notice: Notice) -> (String, Tone) {
    match notice {
        Notice::Interrupted => ("interrupted".into(), Tone::Warning),
        Notice::RequestFailed(e) => (format!("request failed: {e}"), Tone::Error),
        Notice::RoundCapReached(cap) => {
            (format!("stopped after {cap} rounds with no answer"), Tone::Warning)
        }
        Notice::Compacted(report) => (report, Tone::Muted),
        Notice::CompactionStillLarge(percent) => {
            (format!("still over {percent}% of the window; not compacting again"), Tone::Warning)
        }
        Notice::CompactionFailed(e) => (format!("could not compact: {e}"), Tone::Error),
        Notice::ReasoningCompressed { before, after } => {
            (format!("compressed reasoning {before} → {after} tokens"), Tone::Muted)
        }
        Notice::ReasoningDropped { tokens } => {
            (format!("dropped ~{tokens} tokens of reasoning"), Tone::Muted)
        }
        Notice::ToolCallsMalformed(shown) => {
            (format!("malformed tool_calls: {shown}"), Tone::Error)
        }
        Notice::ToolCallUnnamed(detail) => (detail, Tone::Error),
        Notice::ToolArgumentsInvalid { tool, detail } => (format!("{tool}: {detail}"), Tone::Error),
        Notice::ToolFailed(e) => (format!("tool failed: {e}"), Tone::Error),
        Notice::PlanModeEntered => ("plan mode — investigating first".into(), Tone::Normal),
        Notice::PlanApproved { auto: false } => ("plan approved".into(), Tone::Normal),
        Notice::PlanApproved { auto: true } => {
            ("plan approved — auto mode; each call is reviewed first".into(), Tone::Normal)
        }
        Notice::PlanRejected => ("plan sent back for revision".into(), Tone::Warning),
        Notice::AskFailed { what, detail } => {
            (format!("could not ask {what}: {detail}"), Tone::Error)
        }
        Notice::Refused(reason) => (reason, Tone::Error),
        Notice::PermissionAllowed => ("allowed".into(), Tone::Muted),
        Notice::PermissionAllowedAll(label) => (label, Tone::Muted),
        Notice::PermissionDenied { alternative } => {
            (format!("denied — {alternative}"), Tone::Warning)
        }
        Notice::LoopNudge { tool, count } => {
            (format!("'{tool}' called {count} times in a row — telling the model"), Tone::Warning)
        }
        Notice::LoopBroken { tool, count } => {
            (format!("stopped: '{tool}' called {count} times in a row"), Tone::Warning)
        }
        Notice::QueuedSent(text) => (text, Tone::Normal),
        Notice::ProjectContextReloaded(name) => {
            (format!("reloaded project context from {name}"), Tone::Muted)
        }
    }
}

// ---------------------------------------------------------------------------
// Writing to the window. Everything below runs on the UI thread.
// ---------------------------------------------------------------------------

/// A fresh selectable `RichText`, carrying `id` when it has to be found again.
fn rich(id: &str, size: f32) -> Element {
    let text = RichText::new(rect((0, 0), (0, 0)), size);
    text.set_selectable(true);
    let element: Element = Rc::new(RefCell::new(text));
    // A gap above every entry, so two banded blocks in a row read as two blocks.
    // Without it a tool's output and the diff after it share one continuous
    // background and look like a single one.
    element.borrow_mut().set_margin(4, 0, 0, 0);
    if !id.is_empty() {
        element.borrow_mut().set_id(id);
    }
    element
}

/// Add a view to the end of the transcript and follow the end.
fn append(ui: &mut UI, view: Element) {
    let Some(transcript) = ui.get_view(TRANSCRIPT) else { return };
    if let Some(container) = transcript.borrow_mut().as_container_mut() {
        container.add_view(view);
    }
    ui.relayout();
    follow_end(ui);
}

/// Keep the newest line in view while the transcript grows.
fn follow_end(ui: &mut UI) {
    if let Some(element) = ui.get_view(SCROLL) {
        let view = element.borrow();
        if let Some(scroll) = view.as_any().downcast_ref::<ScrollView>() {
            scroll.scroll_to_end();
        }
    }
}

/// Run `f` against the `RichText` under `id`, if there is one.
fn with_rich<R>(ui: &mut UI, id: &str, f: impl FnOnce(&RichText) -> R) -> Option<R> {
    let element = ui.get_view(id)?;
    let view = element.borrow();
    let rich = view.as_any().downcast_ref::<RichText>()?;
    Some(f(rich))
}

fn line(ui: &mut UI, text: &str, tone: Tone) {
    let element = rich("", PROSE);
    if let Some(view) = element.borrow().as_any().downcast_ref::<RichText>() {
        let mut style = SpanStyle::default();
        style.color = tone.color();
        view.push(text, style);
    }
    append(ui, element);
}

fn user(ui: &mut UI, text: &str) {
    let element = rich("", PROSE);
    if let Some(view) = element.borrow().as_any().downcast_ref::<RichText>() {
        view.push("you   ", SpanStyle::default().bold().color(0xFF6FA8DC));
        view.push(text, SpanStyle::default());
    }
    append(ui, element);
}

/// A tool call: one row, but code, so it gets the monospace family (Lumio L1).
fn code_line(ui: &mut UI, text: &str) {
    let element = rich("", CODE);
    if let Some(view) = element.borrow().as_any().downcast_ref::<RichText>() {
        view.push(text, SpanStyle::default().code().color(0xFF8FBC8F));
    }
    append(ui, element);
}

/// Preformatted text in its own band — a diff, a tool's output.
///
/// This is Lumio's `BlockStyle::pre` path, and the reason it had to exist:
/// whitespace is content here, so a diff keeps its leading `+`/`-` alignment and
/// a code block keeps its indentation, and the band makes the whole thing read
/// as one box rather than a stack of tinted strips.
fn block(ui: &mut UI, text: &str) {
    let element = rich("", CODE);
    if let Some(view) = element.borrow().as_any().downcast_ref::<RichText>() {
        view.start_block(BlockStyle::pre().background(0xFF232323).padding(8.0).hanging(20.0));
        view.push(text.trim_end(), SpanStyle::default().code());
    }
    append(ui, element);
}

// The markdown being streamed into the open reply.
//
// A thread-local rather than a field, because every write to the window happens
// in a one-shot closure on the UI thread: there is no long-lived object here to
// hang it on, and the view itself is found by id. It is only ever touched from
// the UI thread, which is what makes this safe rather than merely convenient.
#[cfg(feature = "gui")]
thread_local! {
    static REPLY_MARKDOWN: RefCell<Option<crate::markdown::MarkdownView>> =
        const { RefCell::new(None) };
}

/// Append a chunk of the model's answer, rendering it as markdown.
///
/// The reply is one `RichText` for the whole message, so it can be selected in
/// one drag; the markdown renderer replaces only the block still being written.
fn stream_reply(ui: &mut UI, text: &str) {
    let existed = with_rich(ui, OPEN_REPLY, |view| {
        REPLY_MARKDOWN.with(|md| {
            let mut md = md.borrow_mut();
            md.get_or_insert_with(|| crate::markdown::MarkdownView::new(PROSE)).push(view, text);
        });
    })
    .is_some();
    if existed {
        ui.relayout();
        follow_end(ui);
        return;
    }
    let element = rich(OPEN_REPLY, PROSE);
    if let Some(view) = element.borrow().as_any().downcast_ref::<RichText>() {
        REPLY_MARKDOWN.with(|md| {
            let mut md = md.borrow_mut();
            *md = Some(crate::markdown::MarkdownView::new(PROSE));
            if let Some(md) = md.as_mut() {
                md.push(view, text);
            }
        });
    }
    append(ui, element);
}

/// The reply is over: settle the last block and forget the renderer, so the next
/// reply starts from nothing rather than continuing this one's markdown.
fn finish_reply(ui: &mut UI) {
    with_rich(ui, OPEN_REPLY, |view| {
        REPLY_MARKDOWN.with(|md| {
            if let Some(md) = md.borrow_mut().as_mut() {
                md.finish(view);
            }
        });
    });
    REPLY_MARKDOWN.with(|md| *md.borrow_mut() = None);
    ui.relayout();
}

/// Append to the open view, creating it on the first chunk. Used for the chain
/// of thought, which is not markdown — it is the model talking to itself, and
/// running it through a renderer would put headings and lists in a stream of
/// half-finished sentences.
///
/// Uses `RichText::push`, so only the block being written into is reshaped —
/// which is what makes streaming into one long message affordable.
fn stream(ui: &mut UI, id: &str, text: &str, dim: bool) {
    let style = match dim {
        true => SpanStyle::default().italic().color(0xFF767676),
        false => SpanStyle::default(),
    };
    if with_rich(ui, id, |view| view.push(text, style.clone())).is_some() {
        ui.relayout();
        follow_end(ui);
        return;
    }
    let element = rich(id, if dim { CODE } else { PROSE });
    if let Some(view) = element.borrow().as_any().downcast_ref::<RichText>() {
        view.push(text, style);
    }
    append(ui, element);
}

/// Close whatever reply is open by taking its id away, so a later chunk starts a
/// new one instead of landing in a finished message.
fn close_open(ui: &mut UI) {
    for id in [OPEN_REPLY, OPEN_REASONING] {
        if let Some(element) = ui.get_view(id) {
            element.borrow_mut().set_id("");
        }
    }
}

fn set_label(ui: &mut UI, id: &str, text: &str) {
    if let Some(element) = ui.get_view(id) {
        if let Some(label) = element.borrow_mut().as_any_mut().downcast_mut::<Label>() {
            label.set_text(text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumio::drawing::{set_current_palette, DrawableRegistry, Palette};
    use lumio::render::render_to_pixmap;
    use serde_json::json;

    /// Drive the window with synthetic events and render it with no window at
    /// all, which is the only way this frontend can be tested at all: there is
    /// no terminal to capture and no display to open.
    ///
    /// The assertion is weak on purpose — that something drew, and that the
    /// transcript grew by one view per event. What it really guards is that the
    /// layout parses, every id the code looks up exists, and none of the write
    /// paths panic; those are exactly the mistakes that are invisible until a
    /// window opens.
    fn render_events(events: Vec<AgentEvent>, name: &str) -> (usize, bool) {
        // Both Lumio backends are compiled in (GL, with software as the fallback
        // for GPU-less machines), and in that build the text is shaped by
        // speedy2d while `RendererSoftware::draw_text` can only draw what fontdue
        // shaped. Without this the pixmap comes back with every widget in place
        // and not one glyph on it — see Lumio's own
        // `dual_build_software_text_renders`. The real window uses GL and needs
        // nothing here; this is the harness, not the frontend.
        lumio::backend::set_render_backend(RenderBackend::Software);

        let (w, h) = (1040u32, 760u32);
        let palette = Palette::dark();
        set_current_palette(palette.clone());
        let registry = DrawableRegistry::new();

        let mut ui = UI::from_xml(LAYOUT, w, h, default_typeface(), 1.0)
            .expect("the GUI layout must parse");
        ui.layout(w, h, 1.0);

        for event in events {
            // The same mapping `dispatch` does, minus the channel: `dispatch`
            // only queues these closures onto the UI thread, and here we *are*
            // the UI thread.
            apply(&mut ui, event);
            ui.layout(w, h, 1.0);
        }

        let children = ui
            .get_view(TRANSCRIPT)
            .and_then(|t| t.borrow().as_container().map(|c| c.get_view_count()))
            .unwrap_or(0);

        let pixmap = render_to_pixmap(&ui, w, h, 1.0, &palette, &registry).expect("pixmap");
        let data = pixmap.data();
        let first = [data[0], data[1], data[2], data[3]];
        let drew = data.chunks_exact(4).any(|px| px != first);

        // Written where a person can look at it, because "it rendered" and "it
        // rendered correctly" are different questions and only one of them can
        // be asserted.
        let out = std::env::temp_dir().join(format!("haico-gui-{name}.png"));
        pixmap.save_png(&out).expect("write png");
        (children, drew)
    }

    /// The subset of `dispatch` that writes to the window. Kept here rather than
    /// exported so the production path stays a single match.
    fn apply(ui: &mut UI, event: AgentEvent) {
        match event {
            AgentEvent::ReplyStart => close_open(ui),
            AgentEvent::ReplyEnd => {
                finish_reply(ui);
                close_open(ui);
            }
            AgentEvent::ReplyChunk(text) => stream_reply(ui, &text),
            AgentEvent::ReasoningChunk(text) => stream(ui, OPEN_REASONING, &text, true),
            AgentEvent::ToolStarted { name, arguments, .. } => {
                let summary = crate::render::tool_call_summary(&name, &arguments);
                code_line(ui, format!("{name} {summary}").trim());
            }
            AgentEvent::ToolDiff(diff) => block(ui, &diff),
            AgentEvent::ToolResult(result) => block(ui, &result),
            AgentEvent::Notice(notice) => {
                let (text, tone) = describe(notice);
                line(ui, &text, tone);
            }
            AgentEvent::Busy(_) => set_label(ui, BUSY, "thinking…"),
            AgentEvent::Status { .. } => set_label(ui, STATUS, "test-model    ctx 7%"),
            AgentEvent::Finished { .. } => line(ui, "done in 2s", Tone::Muted),
            AgentEvent::DidTool { .. } | AgentEvent::Ask(_) => {}
        }
    }

    /// A markdown reply arriving in chunks, which is how it actually arrives.
    /// Rendered small enough to read, and written out so the formatting can be
    /// looked at rather than only asserted about.
    #[test]
    fn a_markdown_reply_renders() {
        const REPLY: &str = "## What I found\n\nThe loop counts **after** the parse, so a call \
             with bad `arguments` never advances it. Two things follow:\n\n\
             - normal mode still stops at the round cap\n\
             - auto mode skips that cap, so nothing stopped it\n\n\
             Here is the fix:\n\n\
             ```rust\nlet parsed = parse_tool_arguments(function);\n\
             // Counted before the branches below.\nlet key = match &parsed {\n\
             \x20   Ok(a) => a.to_string(),\n\x20   Err(_) => raw(function),\n};\n```\n\n\
             > Measured: fifty rounds before, five after.\n\n\
             See [the plan](plan.md) for the rest.";

        let mut events = vec![AgentEvent::ReplyStart];
        // Chunked the way a model streams: a few characters at a time, so every
        // partial construct — an unclosed fence, a half-typed list marker — is
        // seen and replaced.
        let mut at = 0;
        while at < REPLY.len() {
            let mut next = (at + 7).min(REPLY.len());
            while next < REPLY.len() && !REPLY.is_char_boundary(next) {
                next += 1;
            }
            events.push(AgentEvent::ReplyChunk(REPLY[at..next].to_string()));
            at = next;
        }
        events.push(AgentEvent::ReplyEnd);

        let (children, drew) = render_events(events, "markdown");
        assert!(drew, "the window rendered blank");
        // One view for the whole reply, whatever its internal structure: that is
        // what lets a person select the answer in a single drag.
        assert_eq!(children, 1, "the reply should be one selectable view");
    }

    #[test]
    fn a_whole_exchange_renders() {
        let events = vec![
            AgentEvent::Status { model: "m".into(), used_tokens: 900, context_tokens: 32768 },
            AgentEvent::Busy(Waiting::Model),
            AgentEvent::ReplyStart,
            AgentEvent::ReasoningChunk("Let me look at the file first. ".into()),
            AgentEvent::ReplyChunk("I'll read it and then patch the loop.".into()),
            AgentEvent::ReplyEnd,
            AgentEvent::ToolStarted {
                name: "read".into(),
                arguments: json!({ "file_path": "src/main.rs" }),
                auto_note: None,
            },
            AgentEvent::ToolResult("fn main() {\n    let xs = vec![1, 2, 3];\n}".into()),
            AgentEvent::ToolDiff(
                "--- a/src/main.rs\n+++ b/src/main.rs\n@@\n-    let xs = vec![1];\n+    let xs = vec![1, 2, 3];"
                    .into(),
            ),
            AgentEvent::Notice(Notice::PermissionAllowed),
            AgentEvent::Notice(Notice::LoopNudge { tool: "read".into(), count: 3 }),
            AgentEvent::Notice(Notice::RequestFailed("connection refused".into())),
            AgentEvent::Finished { elapsed: std::time::Duration::from_secs(2) },
        ];
        let (children, drew) = render_events(events, "exchange");
        assert!(drew, "the window rendered blank");
        // Nine: the reasoning and the reply are separate views because they are
        // separately styled and separately closed, then the tool line, the two
        // pre blocks, three notices and the closing line. Asserted exactly, so
        // an event that silently stops appearing is a failure rather than a
        // slightly shorter transcript nobody notices.
        assert_eq!(children, 9, "unexpected number of transcript views");
    }

    /// Every `Notice` has to survive being phrased by this frontend. A variant
    /// added to the vocabulary without a GUI wording fails to compile in
    /// `describe`; this checks the wordings are non-empty and do not panic.
    /// The regression that made the window look unclosable: `run` held the
    /// original input sender for its whole body, so when the window closed and
    /// dropped the handlers' clones the channel still had a live sender, the
    /// worker stayed blocked in `recv`, and `join` never returned. The window
    /// went away and the process did not.
    ///
    /// With every sender gone `converse` has to fall out at its first read. The
    /// test would hang rather than fail if that broke, which is why it is worth
    /// having: the failure mode is a hang, and a hang is invisible in review.
    #[tokio::test]
    async fn a_closed_input_channel_ends_the_conversation() {
        let (w, h) = (400u32, 300u32);
        let ui = UI::from_xml(LAYOUT, w, h, default_typeface(), 1.0).expect("layout");
        let handle = ui.handle();

        let cwd = std::env::current_dir().unwrap();
        let agent = Agent::new(crate::testutil::test_config("http://127.0.0.1:1"), "m".into(), &cwd);

        let (tx, rx) = mpsc::channel::<String>();
        drop(tx);
        let running = Arc::new(Mutex::new(None));
        // Returns, rather than blocking forever on a channel nobody can send to.
        converse(agent, rx, handle, running).await;
    }

    #[test]
    fn every_notice_has_gui_wording() {
        let all = vec![
            Notice::Interrupted,
            Notice::RequestFailed("e".into()),
            Notice::RoundCapReached(50),
            Notice::Compacted("r".into()),
            Notice::CompactionStillLarge(90),
            Notice::CompactionFailed("e".into()),
            Notice::ReasoningCompressed { before: 10, after: 2 },
            Notice::ReasoningDropped { tokens: 10 },
            Notice::ToolCallsMalformed("x".into()),
            Notice::ToolCallUnnamed("x".into()),
            Notice::ToolArgumentsInvalid { tool: "t".into(), detail: "d".into() },
            Notice::ToolFailed("e".into()),
            Notice::PlanModeEntered,
            Notice::PlanApproved { auto: false },
            Notice::PlanApproved { auto: true },
            Notice::PlanRejected,
            Notice::AskFailed { what: "w", detail: "d".into() },
            Notice::Refused("r".into()),
            Notice::PermissionAllowed,
            Notice::PermissionAllowedAll("l".into()),
            Notice::PermissionDenied { alternative: "a".into() },
            Notice::LoopNudge { tool: "t".into(), count: 3 },
            Notice::LoopBroken { tool: "t".into(), count: 5 },
            Notice::QueuedSent("q".into()),
            Notice::ProjectContextReloaded("CLAUDE.md".into()),
        ];
        for notice in all {
            let (text, _) = describe(notice.clone());
            assert!(!text.is_empty(), "{notice:?} has no wording in the GUI");
        }
    }
}
