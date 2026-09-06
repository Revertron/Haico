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
//! `syntect` emitting spans.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use lumio::prelude::*;
use lumio::types::rect;

use crate::agent::{run_turn, Agent, Knob};
use crate::config::Reasoning;
use crate::llm::{fetch_available_models, fetch_loaded_models};
use crate::ask::Answer;
use crate::modes::{Mode, PlanDecision};
use crate::permissions::PermissionResponse;
use crate::turn::{AgentEvent, Ask, Notice, TurnLink, Waiting};
use tokio::sync::oneshot;

const TRANSCRIPT: &str = "transcript";
const SCROLL: &str = "scroll";
const INPUT: &str = "input";
const COMPOSER: &str = "composer";
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

/// The permission panel and its parts. It takes the composer's place while a
/// gated call waits for an answer, so the two are never on screen together.
const GATE: &str = "gate";
const GATE_ALLOW: &str = "gate-allow";
const GATE_ALL: &str = "gate-all";
const GATE_REJECT: &str = "gate-reject";
const GATE_REPLY: &str = "gate-reply";
const GATE_TEXT: &str = "gate-text";
const GATE_SEND: &str = "gate-send";

const MENUBAR: &str = "menubar";

/// Menu item ids. Matched against `MenuBar::clicked_item`, so they are the one
/// place the layout and the code have to agree twice.
const MENU_SETTINGS: &str = "menu-settings";
const MENU_CLEAR: &str = "menu-clear";
const MENU_QUIT: &str = "menu-quit";
const MENU_MODE_NORMAL: &str = "menu-mode-normal";
const MENU_MODE_PLAN: &str = "menu-mode-plan";
const MENU_MODE_AUTO: &str = "menu-mode-auto";
/// The Model menu is built from the server's list once it arrives, so its item
/// ids are this prefix plus the model name, and its title is how the menu is
/// found again when the check has to move.
const MENU_MODEL_PREFIX: &str = "menu-model:";
const MENU_MODEL: &str = "Model";

/// What the server said it had, when it was asked.
///
/// Held for as long as the window is open because the Model menu is rebuilt
/// from it whenever the current model changes. Not re-fetched at that point:
/// switching is not a good moment to wait on the server, and a list that grew
/// stale in one direction (a model loaded since) is a better menu than no menu
/// while a request is in flight.
struct Models {
    all: Vec<String>,
    loaded: Vec<String>,
}

/// Menu icons, by the path [`Icons`] serves them under.
const ICON_CURRENT: &str = "icons/check.svg";
const ICON_LOADED: &str = "icons/memory.svg";

/// The handful of icons the window needs, compiled in.
///
/// haico is one binary and stays one: an icon that has to be found on disk is an
/// icon that is missing on somebody's machine. SVG rather than PNG because Lumio
/// rasterizes SVG at the size it draws, so the same file is right on a HiDPI
/// screen — see `assets/icons/README.md` for where they came from and why they
/// carry an explicit fill.
struct Icons;

impl AssetsProvider for Icons {
    fn get_file(&self, path: &str) -> Option<&[u8]> {
        match path {
            ICON_CURRENT => Some(include_bytes!("../assets/icons/check.svg")),
            ICON_LOADED => Some(include_bytes!("../assets/icons/memory.svg")),
            _ => None,
        }
    }
}

const LAYOUT: &str = r#"
<Frame id="root" width="max" height="max" direction="vertical" padding="0">
    <MenuBar id="menubar" width="max" height="min">
        <Menu title="haico">
            <MenuItem id="menu-settings" text="Model settings…"/>
            <Separator/>
            <MenuItem id="menu-clear" text="Clear conversation"/>
            <Separator/>
            <MenuItem id="menu-quit" text="Quit"/>
        </Menu>
        <Menu title="Mode">
            <MenuItem id="menu-mode-normal" text="Normal — ask about every gated call"/>
            <MenuItem id="menu-mode-plan" text="Plan — investigate, change nothing"/>
            <MenuItem id="menu-mode-auto" text="Auto — review each call first"/>
        </Menu>
    </MenuBar>
    <ScrollView id="scroll" width="max" height="max" direction="vertical">
        <Frame id="transcript" width="max" height="min" direction="vertical" padding="12"/>
    </ScrollView>
    <Label id="busy" width="max" height="min" text="" margin_left="12" margin_top="2"/>
    <Frame id="gate" width="max" height="min" direction="vertical" padding="8" visibility="gone">
        <Button id="gate-allow" width="max" height="32" text="1. Allow once"/>
        <Button id="gate-all" width="max" height="32" text="2. Allow all" margin_top="4"/>
        <Button id="gate-reject" width="max" height="32" text="3. Reject and provide alternative" margin_top="4"/>
        <Frame id="gate-reply" width="max" height="min" direction="horizontal" margin_top="4" visibility="gone">
            <Edit id="gate-text" width="max" height="32"/>
            <Button id="gate-send" text="Send" width="90" height="32" margin_left="8"/>
        </Frame>
    </Frame>
    <Frame id="composer" width="max" height="min" direction="horizontal" padding="8">
        <Memo id="input" width="max" height="70"/>
        <Button id="send" text="Send" width="90" height="70" margin_left="8"/>
    </Frame>
    <Label id="status" width="max" height="min" text="starting…" margin_left="12" margin_bottom="4"/>
</Frame>
"#;

/// Open a window and run `agent` behind it until the window closes.
pub(crate) fn run(agent: Agent) -> Result<()> {
    // Before anything is built, and on this thread: Lumio keeps the provider in
    // thread-local storage and looks it up while drawing, so it has to be set on
    // the thread that owns the window.
    set_provider(Box::new(Icons));

    let mut ui = UI::from_xml(LAYOUT, 1040, 760, default_typeface(), 1.0)
        .ok_or_else(|| anyhow::anyhow!("the GUI layout failed to parse"))?;

    // Commands, UI thread to worker. A plain std channel: the worker blocks on
    // it between turns, which is exactly what is wanted, and nothing on the UI
    // side ever waits for it.
    let (input_tx, input_rx) = mpsc::channel::<Command>();

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

    wire_gate(&mut ui);

    // The menubar fires one Click for whichever leaf was chosen; which one it
    // was comes from `clicked_item`. Quit is handled here because closing the
    // window is the frontend's own business; everything else needs the agent.
    if let Some(bar) = ui.get_view(MENUBAR) {
        let tx = input_tx.clone();
        bar.borrow_mut().on_event(
            EventType::Click,
            Box::new(move |ui, view, _data| {
                let Some(item) =
                    view.as_any().downcast_ref::<MenuBar>().and_then(|bar| bar.clicked_item())
                else {
                    return false;
                };
                if item == MENU_QUIT {
                    ui.request_quit();
                    return true;
                }
                match command_for(&item) {
                    Some(command) => {
                        let _ = tx.send(command);
                        true
                    }
                    None => false,
                }
            }),
        );
    }

    // The link of the turn in flight, if any, so closing the window can stop it.
    // Without this, shutting down waits for whatever the model is in the middle
    // of — a long build, a slow request — and the process outlives the window it
    // came from, which looks exactly like a window that will not close.
    let running: Arc<Mutex<Option<TurnLink>>> = Arc::new(Mutex::new(None));

    let handle = ui.handle();
    let worker_running = Arc::clone(&running);
    // The worker's own sender, so it can hand one to a dialog: the settings form
    // answers on the same channel every other command arrives on.
    let worker_tx = input_tx.clone();
    let worker = std::thread::Builder::new()
        .name("haico-agent".to_string())
        .spawn(move || {
            // `block_on` rather than `spawn`, so nothing here has to be
            // `'static + Send` beyond the agent itself.
            match tokio::runtime::Runtime::new() {
                Ok(runtime) => {
                    runtime.block_on(converse(agent, input_rx, worker_tx, handle, worker_running))
                }
                Err(e) => eprintln!("could not start the agent runtime: {e}"),
            }
        })?;

    // Shutdown is signalled from a drop guard rather than a line after the loop,
    // and that is the whole point of it. The worker holds a sender of its own now
    // (to hand to dialogs), so the channel closing can no longer end its loop —
    // and if `lumio::run` unwinds, a plain statement after it never runs. Either
    // way the process would outlive the window, which is exactly what made it
    // look unclosable the first time.
    struct Shutdown(mpsc::Sender<Command>, Arc<Mutex<Option<TurnLink>>>);
    impl Drop for Shutdown {
        fn drop(&mut self) {
            // Stop the turn in flight rather than waiting it out; cancellation
            // lands somewhere the turn chose, with its history balanced.
            if let Some(link) = self.1.lock().unwrap().take() {
                link.cancel();
            }
            let _ = self.0.send(Command::Shutdown);
        }
    }
    let signal = Shutdown(input_tx, Arc::clone(&running));

    lumio::run(ui, WindowConfig::new("haico", 1040, 760).center().skin("dark"));

    drop(signal);
    let _ = worker.join();
    Ok(())
}

/// Take whatever is in the input box and send it to the worker. Returns whether
/// a redraw is needed, which is Lumio's convention for an event handler.
fn submit(ui: &mut UI, tx: &mpsc::Sender<Command>) -> bool {
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
    let _ = tx.send(Command::Send(text));
    true
}

/// One turn per submitted line, until the window closes and the channel with it.
async fn converse(
    mut agent: Agent,
    input_rx: mpsc::Receiver<Command>,
    tx: mpsc::Sender<Command>,
    handle: UiHandle,
    running: Arc<Mutex<Option<TurnLink>>>,
) {
    show(&handle, |ui| line(ui, "Ready. Type below, then Send or Ctrl+Enter.", Tone::Muted));

    // The model list, fetched once so the Model menu can offer it.
    //
    // Every model the server knows, not only the loaded ones. The REPL's picker
    // prefers loaded models, but that is a different question — "which one do we
    // start with" — and this is switching mid-session, where an unloaded model is
    // a perfectly good answer that the server will load on demand. Narrowing the
    // menu to what is already in memory would take that away, so the loaded ones
    // are *marked* instead and the list stays whole however long it is.
    //
    // Best-effort: a server that cannot be reached leaves the menu absent, which
    // says more than an empty menu would.
    let all = fetch_available_models(&agent.config.llm_url, agent.config.token.as_deref())
        .await
        .unwrap_or_default();
    let models = Arc::new(match all.is_empty() {
        true => Models { all, loaded: Vec::new() },
        false => Models {
            all,
            loaded: fetch_loaded_models(&agent.config.llm_url, agent.config.token.as_deref())
                .await,
        },
    });
    if !models.all.is_empty() {
        let current = agent.model().to_string();
        let models = Arc::clone(&models);
        show(&handle, move |ui| add_model_menu(ui, &models, &current));
    }

    while let Ok(command) = input_rx.recv() {
        let input = match command {
            Command::Send(input) => input,
            other => {
                if !obey(&mut agent, &handle, &tx, &models, other) {
                    break;
                }
                continue;
            }
        };
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
            let description =
                crate::present::permission_description(&tool_name, &arguments, review.as_ref());
            show(handle, move |ui| {
                // What is being asked goes in the transcript, where a diff gets
                // the preformatted block the window already renders well and can
                // be scrolled and selected like anything else. The panel below
                // carries only the answers.
                line(ui, "Allow this tool usage?", Tone::Warning);
                line(ui, &description, Tone::Normal);
                if let Some(diff) = &diff {
                    block(ui, diff);
                }
                open_gate(ui, &allow_all_label, reply);
            });
        }
    }
}

thread_local! {
    /// The permission question the panel is putting, and the channel it answers
    /// on. `None` whenever the panel is closed, which is what makes the number
    /// keys inert the rest of the time.
    ///
    /// UI-thread state, like the asset provider, because the panel's controls
    /// are wired once at startup and have to find the current question from
    /// there. Wiring them per question would be the obvious alternative and does
    /// not work for the accelerators: `fire_shortcut` puts a handler back into
    /// the registry after running it, so one that unregistered itself while
    /// answering would survive anyway and go on eating the key afterwards.
    static PENDING: RefCell<Option<oneshot::Sender<PermissionResponse>>> =
        const { RefCell::new(None) };
}

/// Whether a gated call is waiting on the panel right now.
fn gate_open() -> bool {
    PENDING.with(|pending| pending.borrow().is_some())
}

/// Wire the permission panel's controls. Once, at startup — see [`PENDING`].
fn wire_gate(ui: &mut UI) {
    on_click(ui, GATE_ALLOW, |ui| answer_gate(ui, PermissionResponse::Allow));
    on_click(ui, GATE_ALL, |ui| answer_gate(ui, PermissionResponse::AllowAll));
    on_click(ui, GATE_REJECT, open_reject);
    on_click(ui, GATE_SEND, send_reject);

    // Enter sends the alternative. A KeyDown listener on the focused view runs
    // before the view's own handling, so this needs no accelerator — which is
    // just as well, since a global one for Enter would have to be kept out of
    // the composer's way.
    if let Some(element) = ui.get_view(GATE_TEXT) {
        element.borrow_mut().on_event(
            EventType::KeyDown,
            Box::new(|ui, _view, data| match data {
                EventData::Key { code: Some(VirtualKeyCode::Return), .. } => {
                    send_reject(ui);
                    true
                }
                _ => false,
            }),
        );
    }

    // The number keys, as the terminal's prompt has them. Left registered for
    // the whole session because they do nothing unless a question is pending,
    // and because Lumio fires an accelerator only when the focused view did not
    // take the key first — so typing "1" into the composer, or into the
    // alternative field, types a 1 rather than answering.
    ui.add_shortcut("1", Box::new(|ui| gate_answer(ui, PermissionResponse::Allow)));
    ui.add_shortcut("2", Box::new(|ui| gate_answer(ui, PermissionResponse::AllowAll)));
    ui.add_shortcut(
        "3",
        Box::new(|ui| {
            if !gate_open() {
                return false;
            }
            open_reject(ui);
            true
        }),
    );
}

/// An accelerator that answers the panel, or declines the key if nothing asked.
fn gate_answer(ui: &mut UI, response: PermissionResponse) -> bool {
    if !gate_open() {
        return false;
    }
    answer_gate(ui, response);
    true
}

/// Put a gated call to the user, in the composer's place.
fn open_gate(ui: &mut UI, allow_all_label: &str, reply: oneshot::Sender<PermissionResponse>) {
    // Replacing a pending question drops its sender, which the turn hears as a
    // refusal. It cannot happen — a turn waits for its answer before going on —
    // but if that assumption ever breaks, denying is the direction to break in.
    PENDING.with(|pending| *pending.borrow_mut() = Some(reply));
    // The label names the scope the approval would have, which is the whole
    // reason this answer can be offered at all: a yes/no box had nowhere to
    // put it, so it used to be dropped.
    set_button(ui, GATE_ALL, &format!("2. {allow_all_label}"));
    show_view(ui, GATE, true);
    show_view(ui, COMPOSER, false);
    ui.relayout();
}

/// Answer the pending question, once, and put the composer back.
fn answer_gate(ui: &mut UI, response: PermissionResponse) {
    let Some(reply) = PENDING.with(|pending| pending.borrow_mut().take()) else { return };
    let _ = reply.send(response);
    later(ui, close_gate);
}

/// Swap the third option for the field that says what to do instead.
fn open_reject(ui: &mut UI) {
    later(ui, |ui| {
        show_view(ui, GATE_REJECT, false);
        show_view(ui, GATE_REPLY, true);
        ui.relayout();
        // Focusing the field is also what disarms the number keys: an
        // accelerator fires only if the focused view did not take the key, and
        // an Edit takes them all.
        if let Some(element) = ui.get_view(GATE_TEXT) {
            ui.set_focus_to(&element);
        }
    });
}

/// Reject with whatever the field says, empty included — the terminal's third
/// answer allows an empty alternative too, and the model is told either way.
fn send_reject(ui: &mut UI) {
    let alternative = edit_text(ui, GATE_TEXT).trim().to_string();
    answer_gate(ui, PermissionResponse::Reject(alternative));
}

/// Hide the panel, restore the composer, and leave the panel ready for the next
/// question rather than mid-answer.
fn close_gate(ui: &mut UI) {
    show_view(ui, GATE, false);
    show_view(ui, GATE_REPLY, false);
    show_view(ui, GATE_REJECT, true);
    set_edit(ui, GATE_TEXT, "");
    show_view(ui, COMPOSER, true);
    if let Some(element) = ui.get_view(INPUT) {
        ui.set_focus_to(&element);
    }
    ui.relayout();
}

/// Run `f` on the next UI tick instead of now.
///
/// Mouse dispatch walks the tree holding a `borrow()` on every view on the path
/// down to the one that was clicked, so a handler that hides its own panel —
/// or anything containing it — would `borrow_mut()` a view that is still
/// borrowed, and panic. Queued work runs at the top of the next update, outside
/// any walk, which is the only safe place to change what is on screen from
/// inside a click.
fn later(ui: &UI, f: impl FnOnce(&mut UI) + Send + 'static) {
    ui.handle().run_on_ui_thread(f);
}

/// Show or hide a view, taking its space back when it goes.
fn show_view(ui: &mut UI, id: &str, visible: bool) {
    if let Some(element) = ui.get_view(id) {
        element.borrow_mut().set_visibility(match visible {
            true => Visibility::Visible,
            false => Visibility::Gone,
        });
    }
}

fn on_click(ui: &mut UI, id: &str, f: impl Fn(&mut UI) + 'static) {
    if let Some(element) = ui.get_view(id) {
        element.borrow_mut().on_event(
            EventType::Click,
            Box::new(move |ui, _view, _data| {
                f(ui);
                true
            }),
        );
    }
}

fn set_button(ui: &mut UI, id: &str, text: &str) {
    if let Some(element) = ui.get_view(id) {
        if let Some(button) = element.borrow().as_any().downcast_ref::<Button>() {
            button.set_text(text);
        }
    }
}

fn set_edit(ui: &mut UI, id: &str, text: &str) {
    if let Some(element) = ui.get_view(id) {
        if let Some(edit) = element.borrow().as_any().downcast_ref::<Edit>() {
            edit.set_text(text);
        }
    }
}

fn edit_text(ui: &mut UI, id: &str) -> String {
    ui.get_view(id)
        .and_then(|element| {
            let view = element.borrow();
            view.as_any().downcast_ref::<Edit>().map(|edit| edit.get_text())
        })
        .unwrap_or_default()
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
    // Reasoning is flowed into one paragraph rather than kept as written. Its
    // line breaks carry no meaning — it is the model talking to itself, not
    // structured text — and models habitually end a trace with a blank line or
    // two, which showed up as an empty gap between the thinking and the answer.
    let flowed = if dim { flow(text) } else { text.to_string() };
    let text = flowed.as_str();
    if text.is_empty() {
        return;
    }
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

/// Collapse every run of whitespace containing a newline down to one space.
///
/// A chunk boundary can fall inside such a run, so this cannot see the whole run
/// at once — hence the leading-space case: a chunk that is nothing but a break
/// becomes a single space, and one that begins with a break contributes one.
/// Two spaces where a run was split is the worst it can do, and that is
/// invisible.
fn flow(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut just_broke = false;
    for ch in text.chars() {
        match ch {
            '\n' | '\r' => {
                if !just_broke && !out.ends_with(' ') {
                    out.push(' ');
                }
                just_broke = true;
            }
            ch => {
                out.push(ch);
                just_broke = false;
            }
        }
    }
    out
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


// ---------------------------------------------------------------------------
// Menus and the settings dialog
// ---------------------------------------------------------------------------

/// What the window asks the conversation to do.
///
/// The channel between the two threads used to carry a submitted line and
/// nothing else. Menus make it a command channel, because everything a menu does
/// needs the agent and the agent lives on the worker — including *reading* it:
/// the settings dialog is opened by the worker with a snapshot, rather than by
/// the UI reaching across for the current values.
pub(crate) enum Command {
    Send(String),
    /// The window has closed. An explicit signal rather than relying on every
    /// sender being dropped: the worker holds one itself now, to hand to
    /// dialogs, so closure alone can no longer end the loop.
    Shutdown,
    /// Read the settings and open the dialog on them.
    OpenSettings,
    /// Field values from the dialog, by canonical knob name. An empty string
    /// clears that knob, which is what an emptied field should mean.
    ApplySettings { knobs: Vec<(String, String)>, reasoning: String },
    SetMode(Mode),
    SwitchModel(String),
    Clear,
}

/// The settings as they stand, for filling the form.
struct Snapshot {
    model: String,
    /// Canonical name, label, and current value — empty when the knob is unset.
    knobs: Vec<(&'static str, String)>,
    reasoning: Reasoning,
}

/// Field id for a knob in the settings form.
fn field_id(name: &str) -> String {
    format!("set-{name}")
}

const REASONING_FIELD: &str = "set-reasoning";
const SETTINGS_APPLY: &str = "settings-apply";
const SETTINGS_CANCEL: &str = "settings-cancel";

/// Build the settings form and show it.
///
/// The form is its own `UI` handed to `Dialog::content`, which means the buttons
/// have to live in this layout too and the size has to be given: custom content
/// is not auto-measured. Reading it back happens in `on_result`, where the
/// dialog's own ids resolve.
fn show_settings(ui: &mut UI, snapshot: Snapshot, tx: mpsc::Sender<Command>) {
    let names: Vec<&'static str> = snapshot.knobs.iter().map(|(name, _)| *name).collect();
    let Some(form) = settings_form(&snapshot) else {
        line(ui, "the settings form failed to build", Tone::Error);
        return;
    };

    Dialog::new("Model settings")
        .content(form)
        .size(FORM_WIDTH, FORM_HEIGHT)
        .button_id(SETTINGS_APPLY, "Apply", ButtonSide::Right)
        .button_id(SETTINGS_CANCEL, "Cancel", ButtonSide::Right)
        .default_button(SETTINGS_APPLY)
        .cancel_button(SETTINGS_CANCEL)
        .on_result(move |ui, pressed| {
            if pressed != SETTINGS_APPLY {
                return;
            }
            // The dialog's own ids resolve here, which is how the form is read
            // back — the same way `UI::show_input` reads its one field.
            let knobs = names
                .iter()
                .map(|name| {
                    let text = ui
                        .get_view(&field_id(name))
                        .and_then(|e| {
                            let view = e.borrow();
                            view.as_any().downcast_ref::<Edit>().map(|edit| edit.get_text())
                        })
                        .unwrap_or_default();
                    (name.to_string(), text)
                })
                .collect();
            let reasoning = ui
                .get_view(REASONING_FIELD)
                .and_then(|e| {
                    let view = e.borrow();
                    view.as_any().downcast_ref::<ComboBox>().and_then(|c| c.get_selected_text())
                })
                .unwrap_or_default();
            let _ = tx.send(Command::ApplySettings { knobs, reasoning });
        })
        .show(ui);
}

const FORM_WIDTH: u32 = 560;
const FORM_HEIGHT: u32 = 470;

/// Build the form's own `UI`, filled from `snapshot`.
///
/// Separate from showing it because a Lumio dialog is an app-modal *window*
/// rather than an overlay, so the only way to render the form for inspection is
/// to build it and render it on its own. Which is also the only way to test it.
fn settings_form(snapshot: &Snapshot) -> Option<UI> {
    let mut rows = String::new();
    for (name, value) in &snapshot.knobs {
        rows.push_str(&format!(
            r#"<Frame width="max" height="min" direction="horizontal" margin_bottom="6">
                 <Label width="170" text="{name}"/>
                 <Edit id="{id}" width="max" text="{value}"/>
               </Frame>"#,
            name = name,
            id = field_id(name),
            value = value,
        ));
    }

    let layout = format!(
        r#"
<Frame id="settings-root" width="max" height="max" direction="vertical" padding="14">
    <Label width="max" text="Model: {model}" margin_bottom="10"/>
    <Label width="max" text="Empty means unset — the server decides." margin_bottom="10"/>
    {rows}
    <Frame width="max" height="min" direction="horizontal" margin_bottom="6">
        <Label width="170" text="reasoning"/>
        <ComboBox id="{reasoning_field}" width="max"/>
    </Frame>
    <Frame width="max" height="min" direction="horizontal" gravity="right" margin_top="10">
        <Button id="{apply}" text="Apply" width="110"/>
        <Button id="{cancel}" text="Cancel" width="110" margin_left="8"/>
    </Frame>
</Frame>
"#,
        model = snapshot.model,
        rows = rows,
        reasoning_field = REASONING_FIELD,
        apply = SETTINGS_APPLY,
        cancel = SETTINGS_CANCEL,
    );

    let form = UI::from_xml(&layout, FORM_WIDTH, FORM_HEIGHT, default_typeface(), 1.0)?;
    // Filled after parsing, because a ComboBox's items are added from code
    // rather than from the layout.
    if let Some(element) = form.get_view(REASONING_FIELD) {
        let view = element.borrow();
        if let Some(combo) = view.as_any().downcast_ref::<ComboBox>() {
            for (index, policy) in REASONING_CHOICES.iter().enumerate() {
                combo.add_item(&policy.to_string());
                if *policy == snapshot.reasoning {
                    combo.set_selected(index);
                }
            }
        }
    }
    Some(form)
}


/// The reasoning policies, in the order the dropdown offers them.
const REASONING_CHOICES: &[Reasoning] = &[Reasoning::Keep, Reasoning::Compress, Reasoning::Drop];

/// Turn a menu click into a command. `None` for an item that needs no worker.
fn command_for(item: &str) -> Option<Command> {
    match item {
        MENU_SETTINGS => Some(Command::OpenSettings),
        MENU_CLEAR => Some(Command::Clear),
        MENU_MODE_NORMAL => Some(Command::SetMode(Mode::Normal)),
        MENU_MODE_PLAN => Some(Command::SetMode(Mode::Plan)),
        MENU_MODE_AUTO => Some(Command::SetMode(Mode::Auto)),
        model if model.starts_with(MENU_MODEL_PREFIX) => {
            Some(Command::SwitchModel(model[MENU_MODEL_PREFIX.len()..].to_string()))
        }
        _ => None,
    }
}

/// Add the Model menu once the server's list has arrived.
///
/// Appended rather than filled in place, because `MenuBar` can add a menu but
/// not replace one — so the menu simply is not there until the list is, which is
/// also the honest thing to show when the server cannot be reached.
fn add_model_menu(ui: &mut UI, models: &Models, current: &str) {
    let Some(element) = ui.get_view(MENUBAR) else { return };
    let items = model_items(models, current);
    if let Some(bar) = element.borrow_mut().as_any_mut().downcast_mut::<MenuBar>() {
        bar.add_menu(MENU_MODEL, items);
    }
    ui.relayout();
}

/// Move the check to `current`, by rebuilding the Model menu's items.
///
/// The whole menu rather than the two rows that changed, because "which row
/// holds the check" is not state this keeps — it is derived from the current
/// model each time, so there is nothing to get out of step. The dropdown is
/// built from these items as it opens, so the change is there the next time it
/// is opened, which is the first time anyone can see it.
///
/// False when there is no Model menu to remark, which is the ordinary state of
/// a session whose server never answered with a list.
fn remark_model_menu(ui: &mut UI, models: &Models, current: &str) -> bool {
    let Some(element) = ui.get_view(MENUBAR) else { return false };
    let items = model_items(models, current);
    // Bound rather than chained: the borrow guard would otherwise outlive
    // `element` itself, there being no statement after it to end it.
    let mut view = element.borrow_mut();
    match view.as_any_mut().downcast_mut::<MenuBar>() {
        Some(bar) => bar.set_menu_items(MENU_MODEL, items),
        None => false,
    }
}

/// One row per model the server knows, marked with what haico knows about it.
fn model_items(models: &Models, current: &str) -> Vec<MenuItem> {
    models
        .all
        .iter()
        .map(|model| {
            // Which one is in use, and which are already in memory — the second
            // matters because picking an unloaded one means waiting for it to
            // load. Marked rather than filtered: everything stays reachable.
            //
            // In the icon column a menu already has, rather than as a character
            // in front of the text: a bullet glyph has to be paid for out of the
            // label, so unmarked rows need matching padding to stay aligned and
            // the model name starts in a different place depending on its state.
            let icon = match (model == current, models.loaded.contains(model)) {
                (true, _) => ICON_CURRENT,
                (false, true) => ICON_LOADED,
                (false, false) => "",
            };
            MenuItem {
                id: format!("{MENU_MODEL_PREFIX}{model}"),
                icon_path: icon.to_string(),
                text: model.clone(),
                separator: false,
                children: Vec::new(),
            }
        })
        .collect()
}

/// Empty the transcript, for `Clear`.
fn clear_transcript(ui: &mut UI) {
    let Some(element) = ui.get_view(TRANSCRIPT) else { return };
    let ids: Vec<String> = {
        let view = element.borrow();
        match view.as_container() {
            Some(container) => container.get_views().iter().map(|v| v.borrow().get_id()).collect(),
            None => Vec::new(),
        }
    };
    if let Some(container) = element.borrow_mut().as_container_mut() {
        for id in ids {
            container.remove_view(&id);
        }
    }
    // Anything mid-stream is gone with the views it was writing into.
    REPLY_MARKDOWN.with(|md| *md.borrow_mut() = None);
    ui.relayout();
}

/// Carry out everything that is not a message. Returns false when the window is
/// gone and the loop should end.
///
/// On the worker, because all of it needs the agent — including *reading* it:
/// the settings dialog is opened from here with a snapshot, rather than by the
/// UI reaching across for the current values.
fn obey(
    agent: &mut Agent,
    handle: &UiHandle,
    tx: &mpsc::Sender<Command>,
    models: &Arc<Models>,
    command: Command,
) -> bool {
    match command {
        // Handled by the caller.
        Command::Send(_) => {}
        Command::Shutdown => return false,
        Command::OpenSettings => {
            let snapshot = Snapshot {
                model: agent.model().to_string(),
                knobs: Knob::ALL
                    .iter()
                    .map(|knob| (knob.name(), agent.knob_value(*knob).unwrap_or_default()))
                    .collect(),
                reasoning: agent.reasoning,
            };
            // The dialog answers on the same channel every other command comes
            // in on, so Apply is just another command and needs no second path.
            let tx = tx.clone();
            show(handle, move |ui| show_settings(ui, snapshot, tx));
        }
        Command::ApplySettings { knobs, reasoning } => {
            let mut complaints = Vec::new();
            for (name, value) in knobs {
                let Some(knob) = Knob::parse(&name) else { continue };
                // An emptied field means "stop overriding it", which is exactly
                // what `set_knob` does with an empty string.
                if let Err(complaint) = agent.set_knob(knob, Some(&value)) {
                    complaints.push(format!("{complaint}, got '{value}'"));
                }
            }
            if let Some(policy) = REASONING_CHOICES.iter().find(|p| p.to_string() == reasoning) {
                agent.set_reasoning(*policy);
            }
            // Every field is attempted, and the ones that failed are named. A
            // rejected field changes nothing, so the rest still apply — better
            // than refusing the whole form over one typo.
            let summary = match complaints.is_empty() {
                true => "settings applied".to_string(),
                false => complaints.join("; "),
            };
            let tone = if complaints.is_empty() { Tone::Muted } else { Tone::Error };
            show(handle, move |ui| line(ui, &summary, tone));
        }
        Command::SetMode(mode) => {
            agent.mode = mode;
            let said = format!("mode: {}", mode.describe());
            show(handle, move |ui| line(ui, &said, Tone::Normal));
        }
        Command::SwitchModel(model) => {
            agent.switch_model(model.clone());
            let said = format!("now using {model}");
            let models = Arc::clone(models);
            show(handle, move |ui| {
                remark_model_menu(ui, &models, &model);
                line(ui, &said, Tone::Normal);
            });
        }
        Command::Clear => {
            // What survives a clear and what does not is the agent's business,
            // documented on `Agent::clear`. The transcript going with it is this
            // frontend's, because a window has no scrollback to leave it in.
            let reload = agent.clear();
            show(handle, move |ui| {
                clear_transcript(ui);
                match reload {
                    crate::agent::ContextReload::Reloaded(name) => {
                        line(ui, &format!("reloaded project context from {name}"), Tone::Muted)
                    }
                    crate::agent::ContextReload::Gone => {
                        line(ui, "project context file is gone", Tone::Warning)
                    }
                    crate::agent::ContextReload::Unchanged => {}
                }
                line(ui, "conversation cleared", Tone::Muted);
            });
        }
    }
    true
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
    /// The regression that made the window look unclosable: the worker stayed
    /// blocked in `recv` after the window closed, so `join` never returned —
    /// the window went away and the process did not.
    ///
    /// With every sender gone `converse` has to fall out at its first read. The
    /// test would hang rather than fail if that broke, which is why it is worth
    /// having: the failure mode is a hang, and a hang is invisible in review.
    #[tokio::test]
    async fn a_shutdown_message_ends_the_conversation() {
        let (w, h) = (400u32, 300u32);
        let ui = UI::from_xml(LAYOUT, w, h, default_typeface(), 1.0).expect("layout");
        let handle = ui.handle();

        let cwd = std::env::current_dir().unwrap();
        let agent = Agent::new(crate::testutil::test_config("http://127.0.0.1:1"), "m".into(), &cwd);

        let (tx, rx) = mpsc::channel::<Command>();
        let running = Arc::new(Mutex::new(None));
        // Sent before the loop starts, so it is waiting: `converse` has to see
        // it and return rather than blocking on the next read.
        //
        // Shutdown is a message now rather than the channel closing, because the
        // worker holds a sender of its own to hand to dialogs. Closure alone can
        // therefore no longer end the loop, which is why `run` signals from a
        // drop guard — a plain statement after `lumio::run` would be skipped if
        // it unwound, and the process would once again outlive its window.
        tx.send(Command::Shutdown).unwrap();
        converse(agent, rx, tx, handle, running).await;
    }

    /// Reasoning is flowed, and the reason is the case in the middle: a model
    /// habitually ends a trace with a blank line or two, and those showed as an
    /// empty gap between the thinking and the answer. Chunk boundaries can fall
    /// inside a run of breaks, so the rule has to work per chunk.
    #[test]
    fn reasoning_breaks_collapse_to_single_spaces() {
        assert_eq!(flow("one\ntwo"), "one two");
        assert_eq!(flow("trailing thought.\n\n"), "trailing thought. ");
        assert_eq!(flow("a\r\n\r\nb"), "a b");
        // A chunk that is nothing but the break contributes one space, and a
        // chunk starting with one does not add a second.
        assert_eq!(flow("\n"), " ");
        assert_eq!(flow("\n\n\n"), " ");
        assert_eq!(flow("already \n"), "already ");
        assert_eq!(flow(""), "");
    }


    /// The settings form has to build, fill from the snapshot, and carry every
    /// knob — a field that silently goes missing is a setting that silently
    /// cannot be changed. Rendered too, because a form that parses can still lay
    /// out to nothing.
    #[test]
    fn the_settings_form_builds_and_fills() {
        use lumio::drawing::{set_current_palette, DrawableRegistry, Palette};
        use lumio::render::render_to_pixmap;

        lumio::backend::set_render_backend(RenderBackend::Software);
        let palette = Palette::dark();
        set_current_palette(palette.clone());
        let registry = DrawableRegistry::new();

        let snapshot = Snapshot {
            model: "test-model".into(),
            knobs: Knob::ALL
                .iter()
                .map(|knob| {
                    let value = match knob {
                        Knob::Temperature => "0.7".to_string(),
                        Knob::ReasoningThreshold => "800".to_string(),
                        // The rest unset, which a field shows as empty.
                        _ => String::new(),
                    };
                    (knob.name(), value)
                })
                .collect(),
            reasoning: Reasoning::Compress,
        };

        // A Lumio dialog is an app-modal window rather than an overlay, so the
        // form is built and rendered on its own; rendering the main UI would
        // never show it.
        let mut ui = settings_form(&snapshot).expect("the settings form must build");
        ui.layout(FORM_WIDTH, FORM_HEIGHT, 1.0);

        // Every knob has a field, found by the id the read-back uses.
        for knob in Knob::ALL {
            assert!(
                ui.get_view(&field_id(knob.name())).is_some(),
                "no field for {}",
                knob.name()
            );
        }
        // Filled from the snapshot, not left blank.
        let temperature = ui
            .get_view(&field_id("temperature"))
            .and_then(|e| {
                let view = e.borrow();
                view.as_any().downcast_ref::<Edit>().map(|edit| edit.get_text())
            })
            .unwrap_or_default();
        assert_eq!(temperature, "0.7");

        // The dropdown offers every policy and starts on the current one.
        let selected = ui
            .get_view(REASONING_FIELD)
            .and_then(|e| {
                let view = e.borrow();
                view.as_any()
                    .downcast_ref::<ComboBox>()
                    .map(|c| (c.item_count(), c.get_selected_text()))
            })
            .expect("no reasoning dropdown");
        assert_eq!(selected.0, REASONING_CHOICES.len());
        assert_eq!(selected.1.as_deref(), Some("compress"));

        let pixmap =
            render_to_pixmap(&ui, FORM_WIDTH, FORM_HEIGHT, 1.0, &palette, &registry).unwrap();
        let data = pixmap.data();
        let first = [data[0], data[1], data[2], data[3]];
        assert!(
            data.chunks_exact(4).any(|px| px != first),
            "the settings form rendered blank"
        );
        pixmap.save_png(std::env::temp_dir().join("haico-gui-settings.png")).unwrap();
    }

    /// Menu ids resolve to the commands they name, including the model items
    /// whose ids carry the model name in them.

    /// The icons have to be served, parse as SVG, and be *visible* on the dark
    /// skin the window opens with.
    ///
    /// The last part is the one worth a test. An MDI icon ships with no `fill`,
    /// so it rasterizes black and vanishes against a dark menu — the file looks
    /// fine, the provider finds it, and nothing appears. Rendering one on the
    /// window's own background and demanding a pixel that is neither the
    /// background nor black is what catches that.
    #[test]
    fn the_menu_icons_are_served_and_visible_on_the_dark_skin() {
        use lumio::drawing::{set_current_palette, DrawableRegistry, Palette};
        use lumio::render::render_to_pixmap;

        lumio::backend::set_render_backend(RenderBackend::Software);
        // Thread-local, so a test thread needs its own.
        set_provider(Box::new(Icons));
        for path in [ICON_CURRENT, ICON_LOADED] {
            let bytes = Icons.get_file(path).unwrap_or_else(|| panic!("{path} is not served"));
            assert!(bytes.starts_with(b"<svg"), "{path} is not an SVG");
            assert!(
                String::from_utf8_lossy(bytes).contains("fill="),
                "{path} has no explicit fill, so it will rasterize black and vanish"
            );
        }

        let palette = Palette::dark();
        set_current_palette(palette.clone());
        let registry = DrawableRegistry::new();
        let (w, h) = (64u32, 64u32);

        for path in [ICON_CURRENT, ICON_LOADED] {
            let xml = format!(
                r#"<Frame width="max" height="max" padding="8">
                     <ImageView id="icon" width="max" height="max" image="{path}"/>
                   </Frame>"#
            );
            let mut ui = UI::from_xml(&xml, w, h, default_typeface(), 1.0).expect("layout");
            ui.layout(w, h, 1.0);
            let pixmap = render_to_pixmap(&ui, w, h, 1.0, &palette, &registry).expect("pixmap");

            // The corner is background; anything the icon drew differs from it.
            let px = pixmap.pixels();
            let background = px[0];
            let drew = px.iter().any(|p| {
                (p.red(), p.green(), p.blue()) != (background.red(), background.green(), background.blue())
            });
            assert!(drew, "{path} rendered nothing at all");


            // And what it drew is lighter than the dark ground, rather than the
            // black an unfilled MDI path would give.
            let brightest = px
                .iter()
                .map(|p| p.red() as u32 + p.green() as u32 + p.blue() as u32)
                .max()
                .unwrap_or(0);
            let ground = background.red() as u32 + background.green() as u32 + background.blue() as u32;
            assert!(
                brightest > ground + 60,
                "{path} is not lighter than the dark background it sits on \
                 (brightest {brightest}, ground {ground}) — it would be invisible"
            );
        }
    }
    /// Whether a view is on screen, as the panel's swap is expressed.
    fn visible(ui: &UI, id: &str) -> bool {
        ui.get_view(id)
            .map(|element| element.borrow().get_visibility() == Visibility::Visible)
            .unwrap_or(false)
    }

    fn press(ui: &mut UI, code: VirtualKeyCode) {
        ui.on_key_down(Some(code), 0, ModifiersState::default());
    }

    fn gate_ui() -> UI {
        lumio::backend::set_render_backend(RenderBackend::Software);
        let mut ui = UI::from_xml(LAYOUT, 800, 600, default_typeface(), 1.0).expect("layout");
        wire_gate(&mut ui);
        ui
    }

    /// The panel takes the composer's place for exactly as long as a gated call
    /// is waiting, and the number keys answer it the way the terminal's prompt
    /// does. The scope of "allow all" is on the button, which is the thing a
    /// yes/no box could not say and so used to drop.
    #[test]
    fn the_panel_replaces_the_composer_while_a_call_waits() {
        let mut ui = gate_ui();
        assert!(visible(&ui, COMPOSER) && !visible(&ui, GATE), "closed until asked");

        let (tx, mut rx) = oneshot::channel();
        open_gate(&mut ui, "Allow all 'edit' calls under src", tx);
        assert!(visible(&ui, GATE) && !visible(&ui, COMPOSER));
        assert!(gate_open());

        let label = ui
            .get_view(GATE_ALL)
            .map(|element| {
                let view = element.borrow();
                view.as_any().downcast_ref::<Button>().expect("a Button").get_text()
            })
            .unwrap_or_default();
        assert_eq!(label, "2. Allow all 'edit' calls under src");

        press(&mut ui, VirtualKeyCode::Key1);
        assert!(matches!(rx.try_recv(), Ok(PermissionResponse::Allow)));
        assert!(!gate_open(), "answered once, and only once");

        // Putting the composer back is deferred, because the same call can come
        // from inside a click on the panel's own button.
        assert!(visible(&ui, GATE), "not until the UI ticks");
        ui.update();
        assert!(visible(&ui, COMPOSER) && !visible(&ui, GATE));
    }

    /// The third answer is the one a yes/no box could not carry at all: it needs
    /// the user's own words. Choosing it turns the option into the field that
    /// takes them, and Enter sends what was typed.
    #[test]
    fn rejecting_swaps_the_third_option_for_a_field() {
        let mut ui = gate_ui();
        let (tx, mut rx) = oneshot::channel();
        open_gate(&mut ui, "Allow all", tx);

        press(&mut ui, VirtualKeyCode::Key3);
        ui.update();
        assert!(!visible(&ui, GATE_REJECT), "the option gave up its place");
        assert!(visible(&ui, GATE_REPLY));
        // The first two answers stay reachable, so choosing to say something is
        // not a one-way door.
        assert!(visible(&ui, GATE_ALLOW) && visible(&ui, GATE_ALL));

        set_edit(&mut ui, GATE_TEXT, "  read it first  ");
        // Enter reaches the field because it has the focus, which the swap gave
        // it — the same focus that stops 1/2/3 answering while typing.
        press(&mut ui, VirtualKeyCode::Return);
        match rx.try_recv() {
            Ok(PermissionResponse::Reject(alternative)) => {
                assert_eq!(alternative, "read it first", "trimmed, and the user's own words")
            }
            _ => panic!("Enter did not send the alternative"),
        }

        ui.update();
        assert!(visible(&ui, COMPOSER) && !visible(&ui, GATE));
        // And the panel is ready for the next question rather than mid-answer.
        assert!(visible(&ui, GATE_REJECT) && !visible(&ui, GATE_REPLY));
        assert_eq!(edit_text(&mut ui, GATE_TEXT), "");
    }

    /// Clicking an option must work, and the click is the dangerous path: mouse
    /// dispatch walks the tree holding a `borrow()` on every view above the one
    /// hit, so a handler that hid the panel there and then would `borrow_mut()`
    /// a view still borrowed and panic. This test is the reason the swap is
    /// deferred to the next tick — without that it is not a wrong pixel, it is
    /// a crash on the most ordinary thing a user can do.
    #[test]
    fn clicking_an_option_answers_without_panicking() {
        let mut ui = gate_ui();
        let (tx, mut rx) = oneshot::channel();
        open_gate(&mut ui, "Allow all", tx);
        ui.force_layout();

        // In window coordinates, which is what dispatch is given: a view's own
        // rect is relative to its parent, because each container subtracts its
        // origin before offering the point to its children.
        let rect_of = |id: &str| ui.get_view(id).expect("laid out").borrow().get_rect();
        let panel = rect_of(GATE);
        let option = rect_of(GATE_ALLOW);
        let middle = Point::new(
            panel.min.x + (option.min.x + option.max.x) / 2,
            panel.min.y + (option.min.y + option.max.y) / 2,
        );
        ui.on_mouse_button_down(middle, MouseButton::Left);
        ui.on_mouse_button_up(middle, MouseButton::Left);

        assert!(matches!(rx.try_recv(), Ok(PermissionResponse::Allow)));
        ui.update();
        assert!(visible(&ui, COMPOSER) && !visible(&ui, GATE));
    }

    /// The accelerators are registered for the whole session, so they have to be
    /// inert when nothing is being asked — otherwise "1" would be spent on a
    /// panel that is not there.
    #[test]
    fn the_number_keys_do_nothing_when_no_call_is_waiting() {
        let mut ui = gate_ui();
        for code in [VirtualKeyCode::Key1, VirtualKeyCode::Key2, VirtualKeyCode::Key3] {
            press(&mut ui, code);
        }
        ui.update();
        assert!(visible(&ui, COMPOSER) && !visible(&ui, GATE));
        assert!(visible(&ui, GATE_REJECT) && !visible(&ui, GATE_REPLY));
    }

    /// The check is derived from the current model, never stored — which is
    /// the whole fix. It used to be written once when the list arrived, so
    /// switching left it sitting on the model that was current at startup.
    #[test]
    fn the_check_follows_the_current_model() {
        let models = Models {
            all: vec!["alpha".into(), "beta".into(), "gamma".into()],
            loaded: vec!["alpha".into(), "beta".into()],
        };
        let icons = |current: &str| -> Vec<String> {
            model_items(&models, current).into_iter().map(|item| item.icon_path).collect()
        };

        // Loaded and in use, loaded, and neither.
        assert_eq!(icons("alpha"), [ICON_CURRENT, ICON_LOADED, ""]);

        // Switching moves the check, and hands the model it left behind back
        // the marker its loaded state earns it.
        assert_eq!(icons("beta"), [ICON_LOADED, ICON_CURRENT, ""]);

        // Including onto one the server has not loaded, which is a legitimate
        // choice: the server loads it on demand.
        assert_eq!(icons("gamma"), [ICON_LOADED, ICON_LOADED, ICON_CURRENT]);

        // A model the list never mentioned marks nothing rather than guessing.
        assert_eq!(icons("delta"), [ICON_LOADED, ICON_LOADED, ""]);
    }

    /// The bug this is for: picking a model moved nothing. The icon logic was
    /// right and the menu could be rebuilt — what was missing was the call
    /// between them, so the check has to be followed the whole way, from the
    /// command a click produces to the item the menu ends up holding.
    #[test]
    fn switching_moves_the_check_all_the_way_to_the_menu() {
        lumio::backend::set_render_backend(RenderBackend::Software);
        let mut ui = UI::from_xml(LAYOUT, 800, 600, default_typeface(), 1.0).expect("layout");
        let handle = ui.handle();

        let models = Arc::new(Models {
            all: vec!["alpha".into(), "beta".into()],
            loaded: vec!["alpha".into()],
        });
        add_model_menu(&mut ui, &models, "alpha");

        let icons = |ui: &UI| -> Vec<String> {
            let bar = ui.get_view(MENUBAR).expect("menubar");
            let bar = bar.borrow();
            bar.as_any()
                .downcast_ref::<MenuBar>()
                .expect("a MenuBar")
                .menu_items(MENU_MODEL)
                .expect("the Model menu")
                .into_iter()
                .map(|item| item.icon_path)
                .collect()
        };
        assert_eq!(icons(&ui), [ICON_CURRENT, ""]);

        // Exactly what a click on the second row produces.
        let command = command_for(&format!("{MENU_MODEL_PREFIX}beta")).expect("a command");
        let cwd = std::env::current_dir().unwrap();
        let mut agent =
            Agent::new(crate::testutil::test_config("http://127.0.0.1:1"), "alpha".into(), &cwd);
        let (tx, _rx) = mpsc::channel::<Command>();
        assert!(obey(&mut agent, &handle, &tx, &models, command));
        assert_eq!(agent.model(), "beta");

        // Queued, not run: it takes a tick on the UI thread, which is what the
        // window does for itself and a test has to do for it.
        assert_eq!(icons(&ui), [ICON_CURRENT, ""], "not until the UI ticks");
        ui.update();
        assert_eq!(icons(&ui), [ICON_LOADED, ICON_CURRENT]);
    }

    /// The two halves have to name the same menu: `set_menu_items` finds it by
    /// title, and a title that did not match would fail silently, leaving
    /// exactly the stale check this replaced.
    #[test]
    fn remarking_finds_the_menu_that_was_added() {
        lumio::backend::set_render_backend(RenderBackend::Software);
        let mut ui = UI::from_xml(LAYOUT, 800, 600, default_typeface(), 1.0).expect("layout");
        let models = Models { all: vec!["alpha".into()], loaded: Vec::new() };

        assert!(!remark_model_menu(&mut ui, &models, "alpha"), "no Model menu yet");
        add_model_menu(&mut ui, &models, "alpha");
        assert!(remark_model_menu(&mut ui, &models, "alpha"));

        // And it replaced the menu rather than adding a second one beside it.
        let bar = ui.get_view(MENUBAR).expect("menubar");
        let bar = bar.borrow();
        let titles = bar.as_any().downcast_ref::<MenuBar>().expect("a MenuBar").menu_titles();
        assert_eq!(titles.iter().filter(|t| *t == MENU_MODEL).count(), 1);
    }

    #[test]
    fn menu_items_map_to_commands() {
        assert!(matches!(command_for(MENU_SETTINGS), Some(Command::OpenSettings)));
        assert!(matches!(command_for(MENU_CLEAR), Some(Command::Clear)));
        assert!(matches!(command_for(MENU_MODE_PLAN), Some(Command::SetMode(Mode::Plan))));
        assert!(matches!(command_for(MENU_MODE_AUTO), Some(Command::SetMode(Mode::Auto))));
        // The model name survives the round trip through the id, colons and all.
        let id = format!("{MENU_MODEL_PREFIX}vendor/model:q4_k_m");
        match command_for(&id) {
            Some(Command::SwitchModel(model)) => assert_eq!(model, "vendor/model:q4_k_m"),
            other => panic!("expected a model switch, got {}", other.is_some()),
        }
        // Quit is the frontend's own and never becomes a command.
        assert!(command_for(MENU_QUIT).is_none());
        assert!(command_for("nonsense").is_none());
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
