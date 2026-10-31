//! The terminal frontend: the REPL's loop, its slash commands, and the pump that runs a
//! turn with the input box still live underneath it.
//!
//! `main.rs` only decides which entry a run takes — this, the MCP server, `--prompt`, the
//! window. Everything below is the terminal's: the keyboard, the editor, the screen, and
//! every line printed. The conversation itself is `Agent`'s and has no terminal in it.

use crate::agent::*;
use crate::ask::*;
use crate::config::*;
use crate::input::*;
use crate::llm::*;
use crate::modes::*;
use crate::permissions::*;
use crate::present::*;
use crate::prompt::*;
use crate::picker::*;
use crate::render::*;
use crate::session::*;
use crate::tools::*;
use crate::turn::*;
use crate::util::*;
use crate::{commands, out, skills, store, timers, trace};
use anyhow::Result;
use rmcp::handler::server::wrapper::Parameters;
use std::env;
use std::path::PathBuf;



// Raw mode and the alternate screen both have to come off however the program leaves,
// including the error paths that return early and a panic on its way out. A terminal left
// in raw mode stops echoing what is typed into it, which looks exactly like a hung shell;
// one left on the alternate screen loses the session entirely.
//
// The order matters. `leave_screen` writes the transcript back to the main screen, and it
// does that while raw mode is still on so its line feeds still get their carriage returns.
struct TerminalGuard(bool);

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.0 {
            out::clear_region();
            out::leave_screen();
            let _ = crossterm::terminal::disable_raw_mode();
            out::set_raw(false);
        }
    }
}

// The four questions that need a keyboard. Each one refuses when there is none rather
// than picking an answer: a terminal that cannot be read from is not consent, and that
// rule is the same whether what is being asked about is a tool call, a plan, which model
// to load, or a fork the model wants the user to settle.

async fn pick_model(keys: Option<&mut Keys>, models: Vec<String>) -> Result<String> {
    match keys {
        Some(keys) => select_model_interactive(keys, models).await,
        None => Err(anyhow::anyhow!(
            "choosing a model needs an interactive terminal; pass --model"
        )),
    }
}

async fn ask_about_plan(keys: Option<&mut Keys>, plan: &str) -> Result<PlanDecision> {
    match keys {
        Some(keys) => review_plan(keys, plan).await,
        None => Err(anyhow::anyhow!("no interactive terminal")),
    }
}

async fn ask_the_user(
    keys: Option<&mut Keys>,
    question: &str,
    options: &[String],
) -> Result<Answer> {
    match keys {
        Some(keys) => ask_question(keys, question, options).await,
        None => Err(anyhow::anyhow!("no interactive terminal")),
    }
}

async fn ask_about_call(
    keys: Option<&mut Keys>,
    allow_all_label: &str,
    description: &str,
    diff: Option<&str>,
) -> Result<PermissionResponse> {
    match keys {
        Some(keys) => ask_permission(keys, allow_all_label, description, diff).await,
        None => Err(anyhow::anyhow!("no interactive terminal")),
    }
}

// The right-hand end of the status bar.
// Takes the count rather than the history: during a turn the frontend draws this
// while the turn owns the messages, so the number has to arrive as a number.
fn status_line(
    provider: Option<&str>,
    model: &str,
    effort: Option<&str>,
    used: usize,
    context_tokens: usize,
) -> String {
    let percent = (used as f64 / context_tokens.max(1) as f64) * 100.0;
    // Before the model, as where it is served from: two providers can serve a model of
    // the same name, and the bar is where the user looks to see which one is answering.
    let named;
    let model = match provider {
        Some(provider) => {
            named = format!("{provider} • {model}");
            named.as_str()
        }
        None => model,
    };
    // Only when one is set. An unset effort means the model thinks as much as it thinks,
    // and saying so on every model that has no such setting would be a word of chrome that
    // never changes — the bar is worth its row for what varies.
    match effort {
        // "off effort" would read as a level called off; it is thinking that is off.
        Some(effort) if effort.eq_ignore_ascii_case("off") => {
            format!("{} • thinking off • ctx {:.0}%", model, percent)
        }
        Some(effort) => format!("{} • {} effort • ctx {:.0}%", model, effort, percent),
        None => format!("{} • ctx {:.0}%", model, percent),
    }
}
// Run one turn with the input box still live underneath it.
//
// This is what the whole rework is for: while the model is thinking, or a tool is
// running, the user can still type. What they submit is queued for the end of the
// round rather than interrupting it — a message injected between an assistant turn
// that called tools and the results of those calls is a message order the server
// rejects. Esc, on the other hand, does interrupt.
//
// There is exactly one of these per turn. There used to be five, one wrapped
// around each piece of the turn that could block, and that is what made the loop
// unmovable: each wrap handed the turn the keyboard, the editor and the queue for
// the duration. Now the turn narrates itself through a channel and shares three
// things through a `TurnLink`, and everything below the seam is this function's.
//
// A `/compact` runs through here too, for the same reason a turn does: it is minutes on
// a local model, and run directly from the loop it had no busy bar and no Esc.
async fn drive(agent: &mut Agent, io: &mut TerminalIo, job: Job) {
    let link = TurnLink::new(agent.mode);
    let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
    let mut presenter = Presenter::new();

    // Chrome the pump draws around the transcript. Seeded from the agent before
    // the turn borrows it, and kept current afterwards by the turn's own Status
    // and Busy events — the pump cannot look at the agent again until the turn has
    // given it back. The mode needs no event: the link is shared, so both sides
    // already see the same one.
    let mut status = status_line(
        agent.provider(),
        agent.model(),
        agent.reasoning_effort(),
        agent.context_used(),
        agent.context_tokens,
    );
    let mut busy = get_thinking_message().to_string();
    // Whether the label is one a character count belongs on — see `Waiting::Model`.
    let mut counting = true;
    // The plan panel is read fresh on every redraw rather than snapshotted:
    // todo_write runs inside the very work this is waiting on, and a panel that
    // only caught up at the end of the turn would be worth nothing. An Arc clone,
    // so it can be read while the turn holds the agent.
    let todos = agent.server.todos.clone();
    // Characters streamed so far, thinking included, for the activity bar. A big
    // model can reason for many minutes before its first content token, and a bar
    // that only says "Thinking…" for all of it is indistinguishable from one that
    // has hung. Counted here now: every chunk goes past this function anyway.
    let mut streamed = 0usize;

    {
        let turn = async {
            match job {
                Job::Turn { input, initializing } => {
                    run_turn(agent, &input, initializing, &tx, &link).await;
                }
                Job::Compact => compact_on_request(agent, &tx, &link).await,
            }
        };
        tokio::pin!(turn);

        let mut tick = tokio::time::interval(std::time::Duration::from_millis(250));
        // What the bar last said, and whether anything else needs redrawing. A tick
        // that would draw the same thing again is skipped: work with no counter to
        // move — a tool running, a call being judged — never changes it, and
        // repainting the region four times a second for the length of a long build
        // is traffic and flicker bought for nothing.
        let mut shown: Option<String> = None;
        let mut shown_plan: Vec<TodoItem> = Vec::new();
        let mut dirty = true;
        // The turn's last act is to narrate its own ending, so every exit drains
        // what is still queued before leaving. The channel cannot close while the
        // pump holds a sender, which is why emptiness rather than closure is the
        // signal.
        macro_rules! drain {
            () => {
                while let Ok(event) = events.try_recv() {
                    answer_or_show(&mut presenter, &mut io.keys, event).await;
                }
            };
        }

        loop {
            // Piped input has nothing to type with, so there is no box to draw and
            // the only way to stop is the signal raw mode would have swallowed.
            let Some(keys) = io.keys.as_mut() else {
                outln!("\n○ {}", busy);
                tokio::select! {
                    _ = &mut turn => {}
                    _ = tokio::signal::ctrl_c() => {
                        link.cancel();
                        (&mut turn).await;
                    }
                }
                drain!();
                break;
            };

            let label = busy_label(&busy, if counting { streamed } else { 0 });
            let plan = todos.lock().unwrap().clone();
            if dirty || shown.as_deref() != Some(label.as_str()) || shown_plan != plan {
                let queued = link.queued();
                draw(
                    &io.editor,
                    &Chrome {
                        mode: link.mode(),
                        busy: Some(label.as_str()),
                        // Something is running, so the bar has better than history
                        // to show. The last thing that happened goes there at the
                        // next prompt instead, which is drawn once this returns.
                        last: None,
                        status: &status,
                        todos: &plan,
                        queued: &queued,
                        plan: io.plan_view,
                    },
                );
                shown = Some(label);
                shown_plan = plan;
                dirty = false;
            }

            tokio::select! {
                // Nothing to do with the tick itself: the loop redraws at the top
                // if the counter has moved since it last did.
                _ = tick.tick() => {}
                _ = &mut turn => {
                    drain!();
                    break;
                }
                event = events.recv() => {
                    match event {
                        Some(event) => {
                            match &event {
                                AgentEvent::Busy(waiting) => {
                                    busy = describe_waiting(waiting);
                                    // Only the model's own generation has a
                                    // stream behind it, so only that label means
                                    // anything by a character count.
                                    counting = matches!(waiting, Waiting::Model);
                                }
                                AgentEvent::Status {
                                    provider,
                                    model,
                                    used_tokens,
                                    context_tokens,
                                    effort,
                                } => {
                                    status = status_line(
                                        provider.as_deref(),
                                        model,
                                        effort.as_deref(),
                                        *used_tokens,
                                        *context_tokens,
                                    );
                                }
                                AgentEvent::ReplyStart => streamed = 0,
                                AgentEvent::ReasoningChunk(text)
                                | AgentEvent::ReplyChunk(text) => {
                                    streamed += text.chars().count();
                                }
                                // The arguments of a call being written out are
                                // generation like any other, and the longest
                                // stretch of a turn that shows nothing at all.
                                AgentEvent::ToolCallChunk(characters) => {
                                    streamed += characters;
                                }
                                _ => {}
                            }
                            answer_or_show(&mut presenter, &mut io.keys, event).await;
                            dirty = true;
                        }
                        // Every sender is gone, which only happens once the turn has
                        // been dropped: nothing more is coming.
                        None => break,
                    }
                }
                event = keys.next() => {
                    // Whatever it was — a character typed, a mode cycled, a resize —
                    // the box below the bar has to be drawn again even when the
                    // label has not moved.
                    dirty = true;
                    let Some(event) = event else {
                        // The keyboard is gone. Finishing the turn is still better
                        // than dropping it on the floor.
                        (&mut turn).await;
                        drain!();
                        break;
                    };
                    let mut mode = link.mode();
                    let action =
                        crate::input::while_working(&mut io.editor, &mut mode, &mut io.plan_view, event);
                    if mode != link.mode() {
                        link.set_mode(mode);
                    }
                    match action {
                        crate::input::WhileWorking::Interrupt => {
                            // What was waiting above the bar was never sent, and the
                            // user has just said they did not want this turn. It goes
                            // back into the box rather than on to the model as the
                            // next prompt.
                            let held = link.take_queued();
                            if !held.is_empty() {
                                io.editor.restore(&held.join("\n"));
                            }
                            // Not a drop: the turn is told, and stops somewhere it
                            // chose, with every tool_call_id already answered. See
                            // `TurnLink::until_cancelled`.
                            link.cancel();
                        }
                        // Not echoed into the transcript here: it has not been sent
                        // yet, and the row above the bar is where it waits until it has.
                        crate::input::WhileWorking::Queued(text) => link.queue(text),
                        crate::input::WhileWorking::Nothing => {}
                    }
                }
            }
        }
    }

    out::clear_region();
    // The turn owned the mode while it ran; hand it back.
    agent.mode = link.mode();
    // And the row above the box keeps whatever the turn last did, across the
    // prompt and into the next turn. Only overwritten when this turn ran a tool:
    // a turn that answered without touching anything leaves the previous answer
    // standing rather than blanking the row.
    if let Some(activity) = presenter.last_activity() {
        io.last_activity = Some(activity.to_string());
    }
    // Anything still queued when the turn ended becomes the next prompt, which is
    // where the user takes over.
    io.queued = link.take_queued();
    if let Some(path) = &io.history_path {
        save_history(path, io.editor.history());
    }
}

// The words for a kind of waiting. Here rather than in the turn because the
// model-thinking one is flavor text — one of a long list, picked fresh each
// time — and choosing words is the display's job.
fn describe_waiting(waiting: &Waiting) -> String {
    match waiting {
        Waiting::Model => get_thinking_message().to_string(),
        Waiting::Compacting => "Compacting context…".to_string(),
        Waiting::CompressingReasoning => "Compressing reasoning…".to_string(),
        Waiting::ReviewingCall => "⁇ Reviewing tool call…".to_string(),
        Waiting::Tool(name) => format!("{}…", name),
    }
}
// Either answer the turn's question or put its output on screen.
//
// Answering happens here rather than in the pump's select arm because a prompt
// runs its own read loop on the keyboard: it has to be awaited outside the
// select that is also borrowing it. The turn is waiting on a reply channel
// meanwhile, so nothing is racing.
async fn answer_or_show(
    presenter: &mut Presenter,
    keys: &mut Option<Keys>,
    event: AgentEvent,
) {
    let AgentEvent::Ask(ask) = event else {
        presenter.show(event);
        return;
    };
    // A dropped reply channel is a refusal, which is what a terminal that cannot
    // be read from has to mean: never consent. Every `let _ =` below is that —
    // the turn hears the drop and treats it as "could not ask".
    match ask {
        Ask::Plan { plan, reply } => {
            if let Ok(decision) = ask_about_plan(keys.as_mut(), &plan).await {
                let _ = reply.send(decision);
            }
        }
        Ask::User { question, options, reply } => {
            if let Ok(answer) = ask_the_user(keys.as_mut(), &question, &options).await {
                let _ = reply.send(answer);
            }
        }
        Ask::Permission { tool_name, arguments, allow_all_label, diff, reply } => {
            // The turn sent the call; the sentence above the buttons is made here.
            let description = permission_description(&tool_name, &arguments);
            let response =
                ask_about_call(keys.as_mut(), &allow_all_label, &description, diff.as_ref().map(diff_text).as_deref())
                    .await;
            if let Ok(response) = response {
                let _ = reply.send(response);
            }
        }
    }
}

// Which model the session starts with: the one configured, else the only one loaded, else
// the user's pick. With no keyboard (`keys` is `None` — piped input, or a `--prompt` run)
// the last step is an error naming the fix rather than a guess.
pub(crate) async fn choose_model(config: &Config, keys: Option<&mut Keys>) -> Result<String> {
    if let Some(model_name) = config.model.clone() {
        // Model specified via command-line, use it directly
        outln!("Using specified model: {}\n", model_name);
        return Ok(model_name);
    }
    find_model(&config.llm_url, config.token.as_deref(), keys).await
}

// The model to use on a server nobody named one for: the only one loaded, else the user's
// pick. Shared by startup and by /provider switching to a provider with no `model`.
async fn find_model(url: &str, token: Option<&str>, keys: Option<&mut Keys>) -> Result<String> {
    outln!("Looking for a model...");

    // Whatever is already in memory is almost always what was meant: the server was
    // started with it, and asking which model to use when there is only one loaded is
    // a question with a single answer. More than one and it is worth asking, but only
    // among those — the rest are downloads, not something that is running. The full
    // list is what the picker falls back to when nothing is loaded at all.
    let loaded = fetch_loaded_models(url, token).await;
    if let [only] = loaded.as_slice() {
        outln!("Using the loaded model: {}\n", only);
        return Ok(only.clone());
    }
    let models = if loaded.is_empty() {
        fetch_available_models(url, token)
            .await
            .map_err(|e| anyhow::anyhow!("Error connecting to LLM: {e}"))?
    } else {
        loaded
    };

    if models.is_empty() {
        return Err(anyhow::anyhow!("No models available from server"));
    }

    // Let user select a model interactively
    pick_model(keys, models).await
}

/// Read both mcp.json files, ask about each project server nobody has approved yet, and
/// start the servers in the background. Whatever happens after this arrives as the hub's
/// notices, printed by `print_mcp_notices` while the prompt waits.
///
/// Asked here, before the first prompt, because a server is a process: by the time a turn
/// wanted its tools it would be too late to have asked. With no keyboard (piped input) an
/// unapproved server is skipped, and the hub says so.
async fn start_mcp_servers(agent: &Agent, keys: Option<&mut Keys>, cwd: &std::path::Path) {
    let mut startup = crate::mcp::Startup::load(cwd);
    if let Some(keys) = keys {
        for server in startup.unapproved() {
            outln!();
            for line in crate::mcp::approval_lines(&server) {
                outln!("{}", line);
            }
            let choices = vec!["Approve and start".to_string(), "Skip for now".to_string()];
            let answer = crate::picker::select_interactive(keys, "answer", choices, 0).await;
            if answer.is_ok_and(|a| a.starts_with("Approve")) {
                if let Err(e) = startup.approve(&server) {
                    outln!("Could not save the approval ({:#}); it holds for this session only.", e);
                }
            }
        }
    }
    startup.start(&agent.server.mcp);
    print_mcp_notices(agent);
}

/// Print what the MCP servers have to say since the last time.
fn print_mcp_notices(agent: &Agent) {
    for notice in agent.server.mcp.take_notices() {
        if notice.is_problem() {
            outln!("⚠ {}", notice.text());
        } else {
            outln!("{}", notice.text());
        }
    }
}

/// The REPL, from the banner to the last background job stopped.
pub(crate) async fn run(config: Config) -> Result<()> {
    // From here on the keyboard is read a key at a time rather than a line at a time,
    // which is what lets the input box exist while the model is working. Piped input
    // has no keyboard to read, so it keeps the plain line-at-a-time path; scripting
    // haico still works.
    let interactive = stdin_is_a_terminal();
    let _terminal = TerminalGuard(interactive);
    // Which renderer draws the session, and then the keyboard — in that order, because
    // taking the screen is what decides how the keyboard has to be read. On Windows the
    // mouse reports nothing at all until the console is in VT input mode, and from then on
    // every key arrives as an escape sequence that has to be parsed rather than read off a
    // record. Only ever the inline renderer when there is no terminal to own: a piped
    // session has no viewport, and taking the alternate screen of something that is not a
    // screen would put the escape sequences into whatever is reading the output.
    let mut keys = if interactive {
        crossterm::terminal::enable_raw_mode()?;
        out::set_raw(true);
        let vt_input = config.tui == Tui::Fullscreen
            && out::enter_screen(config.mouse == Mouse::Scroll);
        Some(Keys::start(vt_input))
    } else {
        None
    };

    outln!("{:^37}", concat!("Welcome to Haico v", env!("CARGO_PKG_VERSION"), "!"));
    outln!("=====================================\n");
    if let Some(notice) = starter_config_notice(&config) {
        outln!("{}", notice);
    } else if let Some(notice) = config_error_notice(&config) {
        outln!("{}
", notice);
    } else if let Some(path) = config.config_path.as_deref().filter(|p| p.exists()) {
        outln!("Config: {}", path.display());
    }
    match &config.provider {
        Some(provider) => outln!("Connecting to {} at: {}", provider, config.llm_url),
        None => outln!("Connecting to LLM server at: {}", config.llm_url),
    }
    match &config.search {
        Some(SearchBackend::Searxng(url)) => outln!("Searching through SearxNG at: {}", url),
        Some(SearchBackend::Tinyfish(_)) => outln!("Searching through Tinyfish"),
        None => {}
    }

    let selected_model = match choose_model(&config, keys.as_mut()).await {
        Ok(model) => model,
        Err(e) => {
            outln!("{:#}", e);
            return Ok(());
        }
    };

    // Everything the conversation itself needs — client, tools, history,
    // permissions, workspace, mode — now lives in one place with no terminal
    // attached (`src/agent.rs`). What stays out here is this frontend's own:
    // the keyboard, the editor, and every line printed.
    let mut cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let resume_requested = config.resume;
    let mut agent = Agent::new(config, selected_model, &cwd);

    // The core reports what it loaded rather than announcing it, so the saying
    // of it stays here with the rest of the output.
    if let Some((name, _)) = &agent.project_context {
        outln!("Loaded project context from {}", name);
    }
    if agent.user_context.is_some() {
        outln!("Loaded user context from {}", user_context_file());
    }
    if resume_requested {
        match agent.resume() {
            Ok(restored) => outln!(
                "Resumed {} message(s) from {} (model then: {}).",
                restored.messages, restored.saved_at, restored.model
            ),
            Err(e) => outln!("Could not resume: {:#}. Starting fresh.", e),
        }
    } else if agent.has_saved_session() {
        outln!("A previous conversation exists here — /resume to continue it.");
    }
    if !agent.workspace.readable_roots().is_empty() {
        outln!(
            "Readable outside the project: {}",
            agent
                .workspace
                .readable_roots()
                .iter()
                .map(|r| display_path(r))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    start_mcp_servers(&agent, keys.as_mut(), &cwd).await;

    // Line editing and history. The editor holds what is being typed; the writer keeps
    // it pinned to the bottom of the screen and prints everything else above it.
    let history_path = history_file_path();
    let history = history_path
        .as_deref()
        .map(load_history)
        .unwrap_or_default();
    // The keyboard joins the rest of the frontend's state here, once the model
    // picker (the one thing that needs it before a session exists) is done with it.
    let mut io = TerminalIo {
        keys,
        editor: Editor::new(history),
        queued: Vec::new(),
        plan_view: PlanView::default(),
        last_activity: None,
        history_path,
    };
    io.editor.set_commands(commands::for_cwd());

    outln!("Type /help for commands.\n");

    // Main loop
    loop {
        // Read user input — unless there is already something waiting that was typed
        // while the last turn was running.
        let input = if !io.queued.is_empty() {
            // It waited above the bar for as long as it was unsent. Now that it is the
            // prompt, it becomes a plain line of the transcript like anything else typed.
            let text = io.queued.remove(0);
            outln!("{}", echo_message(PROMPT, &text));
            text
        } else {
            // Whatever the MCP servers said while the last turn ran, or since the prompt
            // last woke for them.
            print_mcp_notices(&agent);
            match io.keys.as_mut() {
                Some(keys) => {
                    let mut status = status_line(
                        agent.provider(),
                        agent.model(),
                        agent.reasoning_effort(),
                        agent.context_used(),
                        agent.context_tokens,
                    );
                    if let Some((loading, total)) = agent.server.mcp.loading() {
                        status.push_str(&format!(" • MCP {}/{}", total - loading, total));
                    }
                    let plan = agent.plan();
                    match read_command(
                        keys,
                        &mut io.editor,
                        &mut agent.mode,
                        &status,
                        io.last_activity.as_deref(),
                        &plan,
                        &mut io.plan_view,
                        agent.server.next_timer_wake(),
                        agent.server.mcp.changed(),
                    )
                    .await
                    {
                        // A timer's prompt runs as a turn of its own, straight through: it
                        // is the model's message to itself, not a line the user typed, so
                        // it is neither a command nor a skill nor history.
                        Input::Wake => {
                            let fired = agent.server.take_due_timers();
                            if !fired.is_empty() {
                                outln!("{}", timers_fired(&fired));
                                let message = timers::message_for(&fired);
                                drive(&mut agent, &mut io, Job::turn(message)).await;
                            }
                            continue;
                        }
                        // The finished entry becomes a plain line of the transcript, so
                        // the conversation reads as one when scrolled back.
                        Input::Text(text) => {
                            outln!("{}", echo_message(PROMPT, &text));
                            text
                        }
                        // Ctrl+C at the prompt abandons the half-typed line, not the
                        // session.
                        Input::Interrupted => continue,
                        // Ctrl+D, or the keyboard going away.
                        Input::Eof => {
                            outln!("Goodbye!");
                            break;
                        }
                        // read_command handles these itself; none of them ends an entry.
                        Input::CycleMode | Input::TogglePlan | Input::ToggleExpand => continue,
                    }
                }
                None => match read_piped_line() {
                    Some(line) => line,
                    None => {
                        outln!("Goodbye!");
                        break;
                    }
                },
            }
        };
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        let job = match slash_command(&mut agent, &mut io, &mut cwd, input).await {
            Next::Exit => break,
            Next::Prompt => continue,
            Next::Run(job) => job,
        };
        drive(&mut agent, &mut io, job).await;
    }

    if let Some(path) = &io.history_path {
        save_history(path, io.editor.history());
    }

    // Every way out of the loop lands here, so a dev server started mid-session does not
    // outlive it holding a port. Silent when there is nothing to do.
    let stopped = agent.server.stop_all_background().await;
    if stopped > 0 {
        outln!("Stopped {} background job(s).", stopped);
    }
    // The MCP servers' processes go too. The Job Object would take them with haico's
    // exit anyway on Windows; this is the clean close, and the only one elsewhere.
    agent.server.mcp.stop_all();

    trace::end(None);
    Ok(())
}


// ---------------------------------------------------------------------------
// Slash commands
// ---------------------------------------------------------------------------

/// What the loop does once a line has been dealt with.
enum Next {
    Exit,
    /// Back to the prompt: the command was answered here.
    Prompt,
    Run(Job),
}

/// Work that runs through the pump, with the input box live and Esc listened for.
enum Job {
    Turn { input: String, initializing: bool },
    Compact,
}

impl Job {
    fn turn(input: impl Into<String>) -> Self {
        Job::Turn { input: input.into(), initializing: false }
    }
}

/// Every built-in, one per row of `commands::BUILT_IN`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Cmd {
    Help,
    Plan,
    Auto,
    Normal,
    Mode,
    Add,
    Workspace,
    Cd,
    Init,
    Tools,
    Mcp,
    Model,
    Provider,
    Config,
    Set,
    Temp,
    Effort,
    Bg,
    Timers,
    Undo,
    Todo,
    Context,
    Reasoning,
    Compact,
    Resume,
    Clear,
    Skills,
    UpdateSkills,
    Exit,
}

/// The command a typed name runs, aliases included. The table is what knows the names;
/// this is what knows the code.
fn command_for(name: &str) -> Option<Cmd> {
    Some(match commands::built_in_named(name)?.name {
        "help" => Cmd::Help,
        "plan" => Cmd::Plan,
        "auto" => Cmd::Auto,
        "normal" => Cmd::Normal,
        "mode" => Cmd::Mode,
        "add" => Cmd::Add,
        "workspace" => Cmd::Workspace,
        "cd" => Cmd::Cd,
        "init" => Cmd::Init,
        "tools" => Cmd::Tools,
        "mcp" => Cmd::Mcp,
        "model" => Cmd::Model,
        "provider" => Cmd::Provider,
        "config" => Cmd::Config,
        "set" => Cmd::Set,
        "temp" => Cmd::Temp,
        "effort" => Cmd::Effort,
        "bg" => Cmd::Bg,
        "timers" => Cmd::Timers,
        "undo" => Cmd::Undo,
        "todo" => Cmd::Todo,
        "context" => Cmd::Context,
        "reasoning" => Cmd::Reasoning,
        "compact" => Cmd::Compact,
        "resume" => Cmd::Resume,
        "clear" => Cmd::Clear,
        "skills" => Cmd::Skills,
        "update-skills" => Cmd::UpdateSkills,
        "exit" => Cmd::Exit,
        _ => return None,
    })
}

/// Whether `/name` reads as an attempt at a command rather than the start of a path:
/// `/etc/hosts` and `/.config` are not, `/hepl` is.
fn is_command_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
}

/// One line typed at the prompt: a built-in, a skill, or a message for the model.
async fn slash_command(agent: &mut Agent, io: &mut TerminalIo, cwd: &mut PathBuf, input: &str) -> Next {
    let Some(command) = input.strip_prefix('/') else {
        return Next::Run(Job::turn(input));
    };
    let (name, rest) = command.split_once(char::is_whitespace).unwrap_or((command, ""));
    let rest = rest.trim();

    let Some(cmd) = command_for(name) else {
        // A skill's instructions become the message and run as an ordinary turn — `/init`'s
        // trick. Built-ins were matched first, so a skill sharing a name with one is still
        // there for the model, just not by typing it.
        if let Some(expanded) = skills::expand_command(&skills::roots_for_cwd(), input) {
            return Next::Run(Job::turn(expanded));
        }
        // A mistyped command sent on to the model only gets "I have no such command" back,
        // for a round trip. Something that cannot be a command name (a path, say) is a
        // message like any other.
        if is_command_name(name) {
            outln!("Unknown command: /{} — /help lists them, /skills the skills.\n", name);
            return Next::Prompt;
        }
        return Next::Run(Job::turn(input));
    };

    match cmd {
        Cmd::Exit => {
            outln!("Goodbye!");
            return Next::Exit;
        }
        Cmd::Help => print_repl_help(),
        Cmd::Plan => show_mode(agent, Some(Mode::Plan)),
        Cmd::Auto => show_mode(agent, Some(Mode::Auto)),
        Cmd::Normal => show_mode(agent, Some(Mode::Normal)),
        Cmd::Mode => show_mode(agent, None),
        Cmd::Add => add_to_workspace(agent, rest),
        Cmd::Workspace => show_workspace(agent),
        Cmd::Cd => change_directory(agent, io, cwd, rest).await,
        Cmd::Init => return init(agent, cwd),
        Cmd::Tools => list_tools(agent),
        Cmd::Mcp => mcp_command(agent, io, cwd, rest).await,
        Cmd::Model => switch_model(agent, io, rest).await,
        Cmd::Provider => switch_provider(agent, io, rest).await,
        Cmd::Config => show_config(agent),
        // One command for every sampling knob rather than one command each. `/temp 0.7` is
        // the shorthand that predates `/set`, kept because it is what fingers already know.
        // `/effort high` earns the same: it is a knob that gets changed with the task in
        // hand rather than tuned once.
        Cmd::Set => set_knob(agent, rest),
        Cmd::Temp => set_knob(agent, &format!("temperature {}", rest)),
        Cmd::Effort => set_knob(agent, &format!("reasoning_effort {}", rest)),
        Cmd::Bg => background_jobs(agent, rest).await,
        Cmd::Timers => timers_command(agent, rest),
        Cmd::Undo => match agent.server.undo_last() {
            Ok(what) => outln!("Reverted: {}\n", what),
            Err(why) => outln!("{}\n", why),
        },
        Cmd::Todo => show_todo(agent),
        Cmd::Context => report_context(
            &agent.messages,
            agent.context_used(),
            agent.context_is_measured(),
            agent.context_tokens,
            &agent.reasoning_stats,
        ),
        Cmd::Reasoning => reasoning_command(agent, rest),
        Cmd::Compact => return Next::Run(Job::Compact),
        Cmd::Resume => resume(agent),
        Cmd::Clear => clear(agent),
        Cmd::Skills => list_skills(),
        Cmd::UpdateSkills => update_skills(agent, io),
    }
    Next::Prompt
}

// The command half from `commands::BUILT_IN`, so a command added there is in the help
// without anyone remembering to; the keys and the rest by hand, since they are not
// commands.
fn print_repl_help() {
    outln!("Commands:");
    for command in commands::BUILT_IN {
        let mut left = format!("/{}", command.name);
        for alias in command.aliases {
            left.push_str(&format!(" /{}", alias));
        }
        if !command.usage.is_empty() {
            left.push(' ');
            left.push_str(command.usage);
        }
        outln!("  {:<21} {}", left, command.about);
        for line in command.more {
            outln!("  {:<21} {}", "", line);
        }
    }
    outln!("  {:<21} {}", "/<skill> [args]", "Run a skill by name; the model is shown the list and loads");
    outln!("  {:<21} {}", "", "one on its own when a request matches it");
    outln!();
    outln!("In a message:  write @path to send the file with it — relative or absolute,");
    outln!("               and a file outside the workspace comes into reach with it.");
    outln!("               A PNG, JPEG, GIF or WEBP goes as a picture, for a model that");
    outln!("               can see one; a model that cannot will say so.");
    outln!("At the prompt: up/down for history, Shift+Enter for another line,");
    outln!("               Ctrl+V or Shift+Insert to paste,");
    outln!("               Shift+Tab to change mode, Ctrl+C to abandon the line.");
    outln!("While working: keep typing — Enter queues the message for when the turn");
    outln!("               ends. Esc interrupts and returns to the prompt.");
    outln!("Either way:    Ctrl+T cycles the plan panel — windowed, all of it, one line.");
    outln!("               Ctrl+O opens what the transcript folded away — the full tool");
    outln!("               output, whole diffs, the reasoning — and folds it back.");
    outln!("Fullscreen:    the wheel and PageUp/PageDown scroll the transcript; the view");
    outln!("               stays put while output arrives and follows again at the end.");
    outln!("               Set tui = \"inline\" in the config, or HAICO_TUI=inline, to go");
    outln!("               back to the terminal's own scrollback.");
    outln!("               On Windows the wheel needs Shift held to select text; set");
    outln!("               mouse = \"off\" to keep plain drag-selection instead.");
    outln!("Project rules:  HAICO.md, AGENTS.md or CLAUDE.md is appended to the system prompt.");
    outln!("Skills:         <name>/SKILL.md under .haico/skills, .agents/skills or .claude/skills");
    outln!("                in the project, or under {}.", crate::store::display("skills"));
    outln!();
}

fn show_mode(agent: &mut Agent, requested: Option<Mode>) {
    if let Some(requested) = requested {
        agent.mode = requested;
    }
    outln!("Mode: {}\n", agent.mode.describe());
    if agent.mode == Mode::Auto {
        // Spelled out, because this is the mode that acts without asking and the user
        // should know exactly how far that goes before leaving it on.
        outln!(
            "{} reviews each call that would have asked you, with no conversation \
             history. Calls it judges routine run on their own, tagged with the \
             reason. Anything it refuses — or cannot judge — still comes to you.",
            agent.client.model
        );
        outln!(
            "It is the same model, so treat it as a check against drift and \
             mistakes, not as a security boundary.\n"
        );
    }
}

fn add_to_workspace(agent: &mut Agent, requested: &str) {
    if requested.is_empty() {
        outln!("Usage: /add <path>   (a directory to read and write for this session)\n");
        return;
    }
    match agent.workspace.add(std::path::Path::new(requested)) {
        Ok(added) => outln!("Added to the workspace: {}\n", display_path(&added)),
        Err(reason) => outln!("{}\n", reason),
    }
}

fn show_workspace(agent: &Agent) {
    outln!("Readable and writable:");
    for root in agent.workspace.writable_roots() {
        // The scratch directory is in this list and looks like a stray temp path without
        // the label — it is haico's own, not something the user added.
        let scratch = agent.workspace.scratch() == Some(root.as_path());
        let suffix = if scratch { "   (this session's scratch directory)" } else { "" };
        outln!("  {}{}", display_path(root), suffix);
    }
    if agent.workspace.readable_roots().is_empty() {
        outln!("Readable only: none configured (see 'read_paths' in the config file).");
    } else {
        outln!("Readable only:");
        for root in agent.workspace.readable_roots() {
            outln!("  {}", display_path(root));
        }
    }
    // Listed apart from the directories above, because that is what they are not: the
    // file is in reach and nothing else in its directory is.
    if !agent.workspace.attached_files().is_empty() {
        outln!("Single files, named with @ in a message:");
        for file in agent.workspace.attached_files() {
            outln!("  {}", display_path(file));
        }
    }
    outln!("\nAnything else is refused outright, for reads as well as writes.\n");
}

async fn change_directory(agent: &mut Agent, io: &mut TerminalIo, cwd: &mut PathBuf, requested: &str) {
    if requested.is_empty() {
        outln!("Working directory: {}\n", display_path(cwd));
        return;
    }
    match agent.change_directory(std::path::Path::new(requested)) {
        Ok(moved) => {
            // The project is now the new one — its instructions, its memory, its branch,
            // its writable root. What has not moved is the conversation, and that is the
            // whole of the reminder: the model is still carrying a history about somewhere
            // else.
            *cwd = moved;
            io.editor.set_commands(commands::for_cwd());
            outln!("Working directory: {}", display_path(cwd));
            if let Some((name, _)) = &agent.project_context {
                outln!("Loaded project context from {}", name);
            }
            outln!("The conversation came with you — /clear (or /new) to start fresh here.");
            // Said after the reminder rather than before it, because it is the alternative
            // to it: one throws the history away, the other swaps it for the one this
            // directory was left in.
            if agent.has_saved_session() {
                outln!("A conversation was saved here before — /resume picks it up instead.");
            }
            // `Agent::change_directory` stopped the last project's servers; these are this
            // one's, asked about and started the way they are at startup.
            start_mcp_servers(agent, io.keys.as_mut(), cwd).await;
            outln!();
        }
        Err(why) => outln!("{}\n", why),
    }
}

// The one "command" that is not one: /init does not do anything itself, it expands into a
// long instruction and runs as an ordinary turn. Written that way so it gets the tools, the
// permission prompts and /undo for free — there is nothing about writing an instruction
// file that needs its own machinery.
fn init(agent: &Agent, cwd: &std::path::Path) -> Next {
    let target = init_target_file(cwd);
    // It ends in write_file, which plan mode strips from the request. Better to say so now
    // than to have the model research the whole project and then find it cannot save the
    // result.
    if agent.mode == Mode::Plan {
        outln!("/init writes {}, which plan mode does not allow. Switch with /normal first.\n", target);
        return Next::Prompt;
    }
    outln!("Analyzing the project to write {}...\n", target);
    Next::Run(Job::Turn { input: build_init_prompt(cwd), initializing: true })
}

fn list_tools(agent: &Agent) {
    outln!("{} tools available:", agent.server.tool_router.list_all().len());
    for tool in agent.server.tool_router.list_all() {
        let gate = if AUTO_ALLOWED_TOOLS.contains(&tool.name.as_ref()) { "auto" } else { "asks" };
        outln!("  {:<14} [{}]", tool.name, gate);
    }
    outln!();
}

/// `/mcp`: every configured server, what state it is in, and each tool with how it is
/// gated. `/mcp reconnect <name>` restarts one, and asks first about a project server that
/// was never approved — this is how a server skipped at startup gets a second chance.
async fn mcp_command(agent: &Agent, io: &mut TerminalIo, cwd: &std::path::Path, argument: &str) {
    use crate::mcp::{ReadOnly, State};

    let hub = &agent.server.mcp;
    if let Some(name) = argument.strip_prefix("reconnect").map(str::trim) {
        if name.is_empty() {
            outln!("Usage: /mcp reconnect <name>\n");
            return;
        }
        let unapproved = hub.status().into_iter().any(|s| s.name == name && s.state == State::NotApproved);
        if unapproved {
            let (Some(server), Some(keys)) = (hub.config(name), io.keys.as_mut()) else {
                outln!("\"{}\" needs approval, and there is no keyboard to ask on.\n", name);
                return;
            };
            for line in crate::mcp::approval_lines(&server) {
                outln!("{}", line);
            }
            let choices = vec!["Approve and start".to_string(), "Skip for now".to_string()];
            let answer = crate::picker::select_interactive(keys, "answer", choices, 0).await;
            if !answer.is_ok_and(|a| a.starts_with("Approve")) {
                return;
            }
            if let Err(e) = crate::mcp::trust::Trust::from_store().approve(cwd, &server.name, &server.fingerprint) {
                outln!("Could not save the approval ({:#}); it holds for this session only.", e);
            }
        }
        match hub.reconnect(name) {
            Ok(()) => print_mcp_notices(agent),
            Err(e) => outln!("{}\n", e),
        }
        return;
    }
    if !argument.is_empty() {
        outln!("Usage: /mcp [reconnect <name>]\n");
        return;
    }

    let servers = hub.status();
    if servers.is_empty() {
        outln!("No MCP servers. Add them to {} or", store::display("mcp.json"));
        outln!("{} in Claude Code's format.\n", display_path(&crate::mcp::config::project_path(cwd)));
        return;
    }
    for server in &servers {
        let state = match &server.state {
            State::Loading => "loading".to_string(),
            State::Ready => format!("{} tool(s)", server.tools.len()),
            State::Failed(error) => format!("failed: {}", error),
            State::Disabled => "disabled".to_string(),
            State::NotApproved => "not approved — /mcp reconnect to ask".to_string(),
        };
        outln!("{} ({}) — {}", server.name, server.scope.label(), state);
        outln!("    {}", server.target);
        if server.state == State::Ready {
            for tool in &server.tools {
                let gate = if tool.auto_allow { "auto" } else { "asks" };
                let read_only = match tool.read_only {
                    Some(ReadOnly::Hint) => ", read-only (server)",
                    Some(ReadOnly::Config) => ", read-only (mcp.json)",
                    None => "",
                };
                outln!("    {} [{}{}]", tool.name, gate, read_only);
            }
        }
        if matches!(server.state, State::Failed(_)) && !server.stderr.is_empty() {
            outln!("    stderr:");
            for line in &server.stderr {
                outln!("      {}", line);
            }
        }
    }
    outln!("Tools join the conversation at the start of the next turn; read-only ones are offered in plan mode.\n");
}

fn list_skills() {
    let found = skills::discover(&skills::roots_for_cwd());
    if found.is_empty() {
        outln!("No skills found. Put <name>/SKILL.md under .haico/skills, .agents/skills");
        outln!("or .claude/skills in the project, or under {}.\n", store::display("skills"));
        return;
    }
    outln!("{} skill(s):", found.len());
    for skill in &found {
        outln!("  /{} — {}", skill.name, skill.description);
        outln!("      {}", display_path(&skill.dir));
    }
    outln!();
}

fn update_skills(agent: &mut Agent, io: &mut TerminalIo) {
    let changed = agent.reload_skills();
    io.editor.set_commands(commands::for_cwd());
    let count = skills::discover(&skills::roots_for_cwd()).len();
    if changed {
        outln!("Skills reloaded: {} available. The system prompt was updated.\n", count);
    } else {
        outln!("Skills unchanged: {} available.\n", count);
    }
}

async fn switch_model(agent: &mut Agent, io: &mut TerminalIo, requested: &str) {
    let chosen = if requested.is_empty() {
        // Copied out so the borrow ends before agent.client is reassigned below.
        let llm_url = agent.client.base_url.clone();
        let token = agent.client.token.clone();
        match fetch_available_models(&llm_url, token.as_deref()).await {
            Ok(models) if !models.is_empty() => match pick_model(io.keys.as_mut(), models).await {
                Ok(model) => model,
                Err(e) => return outln!("Model selection cancelled: {}\n", e),
            },
            Ok(_) => return outln!("The server reports no available models.\n"),
            Err(e) => return outln!("Could not list models: {:#}\n", e),
        }
    } else {
        requested.to_string()
    };

    // Everything that is configured per model is re-resolved inside `switch_model` — the
    // context budget, the sampling, the reasoning policy, both copies of the client, and
    // the system message. The conversation itself is kept. It also records the choice in
    // the config file, which is what comes back here.
    let kept = agent.switch_model(chosen.clone());
    outln!(
        "Now using model: {}{}",
        chosen,
        match kept {
            Kept::Config => "  (saved as the default)",
            _ => "",
        }
    );
    if let Kept::Failed(why) = &kept {
        outln!("  [warning] not saved as the default: {}", why);
    }
    outln!(
        "  context {} ({}){}\n",
        agent.context_tokens,
        agent.settings.context_source,
        match agent.settings.temperature.value {
            Some(t) => format!(", temperature {}", t),
            None => String::new(),
        }
    );
}

async fn switch_provider(agent: &mut Agent, io: &mut TerminalIo, requested: &str) {
    let providers = agent.providers().to_vec();
    if providers.is_empty() {
        // Said with the shape to write, since the file is the only way to add one.
        let path = agent
            .config
            .config_path
            .as_deref()
            .map(display_path)
            .unwrap_or_else(|| "the config file".to_string());
        outln!("No providers are configured. List them in {} like this:\n", path);
        outln!("  version = 1");
        outln!("  providers = [\"local\", \"lmstudio\"]\n");
        outln!("  [local]");
        outln!("  url = \"http://127.0.0.1:11434\"\n");
        outln!("  [lmstudio]");
        outln!("  url = \"http://127.0.0.1:1234\"");
        outln!("  token = \"sk-lm-...\"");
        outln!("  model = \"qwen2.5:14b\"\n");
        return;
    }

    let name = if requested.is_empty() {
        // The caret starts on the provider in force, so Enter alone changes nothing.
        let current = agent
            .provider()
            .and_then(|p| providers.iter().position(|name| name == p))
            .unwrap_or(0);
        match io.keys.as_mut() {
            Some(keys) => match select_interactive(keys, "provider", providers, current).await {
                Ok(name) => name,
                Err(e) => return outln!("{}\n", e),
            },
            None => {
                return outln!("Choosing a provider needs an interactive terminal; give its name: /provider <name>\n")
            }
        }
    } else if providers.iter().any(|name| name == requested) {
        requested.to_string()
    } else {
        return outln!("There is no provider named '{}'; configured: {}\n", requested, providers.join(", "));
    };
    if agent.provider() == Some(name.as_str()) {
        return outln!("Already using provider: {}\n", name);
    }

    let Some(section) = agent.provider_section(&name).cloned() else { return };
    let url = section.url.clone().unwrap_or_default();
    let model = match section.model.clone() {
        Some(model) => model,
        None => {
            let token = section.token.clone().filter(|t| !t.trim().is_empty());
            match find_model(&url, token.as_deref(), io.keys.as_mut()).await {
                Ok(model) => model,
                // Nothing has changed yet: the switch happens only with a model to switch
                // to, so a server that is down leaves the session on the provider that
                // works.
                Err(e) => {
                    return outln!("{:#}\nStill using provider: {}\n", e, agent.provider().unwrap_or("-"))
                }
            }
        }
    };

    // The conversation is kept, as with /model; the url, token, model and every per-model
    // setting move with the switch.
    let kept = agent.switch_provider(&name, model.clone());
    outln!(
        "Now using provider: {} at {}, model {}{}",
        name,
        url,
        model,
        match kept {
            Kept::Config => "  (saved as the default)",
            _ => "",
        }
    );
    if let Kept::Failed(why) = &kept {
        outln!("  [warning] not saved as the default: {}", why);
    }
    outln!("  context {} ({})\n", agent.context_tokens, agent.settings.context_source);
}

fn show_config(agent: &Agent) {
    match agent.config.config_path.as_deref() {
        Some(path) if path.exists() => outln!("Config file: {}", path.display()),
        Some(path) => outln!("Config file: {} (not created yet)", path.display()),
        None => outln!("Config file: unavailable (no home directory)"),
    }
    outln!("\nIn force for '{}':", agent.client.model);
    outln!("  provider          {}", agent.provider().unwrap_or("-"));
    outln!("  url               {}", agent.client.base_url);
    outln!("  token             {}", if agent.client.token.is_some() { "set" } else { "not set" });
    outln!("  context           {}  [{}]", agent.context_tokens, agent.settings.context_source);
    outln!("  temperature       {}", agent.settings.temperature.describe());
    outln!("  top_p             {}", agent.settings.top_p.describe());
    outln!("  top_k             {}", agent.settings.top_k.describe());
    outln!("  min_p             {}", agent.settings.min_p.describe());
    outln!("  repeat_penalty    {}", agent.settings.repeat_penalty.describe());
    outln!("  max_tokens        {}", agent.settings.max_tokens.describe());
    outln!("  reasoning_effort  {}", agent.settings.reasoning_effort.describe());
    // Unset still says more than the bare "-" the request fields share, but for the
    // opposite reason it used to: it no longer sends anything at all, so what needs saying
    // is not what it does but what it does not do — inherit the level above.
    outln!(
        "  sub_agent_effort  {}",
        match &agent.settings.sub_agent_effort {
            Some(effort) => format!("{}  [config [models]]", effort),
            None => "-  [unset (the model's own default, never the level above)]".to_string(),
        }
    );
    // Shown as what is actually in force rather than as what the file said, since
    // /reasoning can have moved it since. The source string is only honest for the former
    // case, so it is replaced rather than kept when they differ.
    outln!(
        "  reasoning         {}{}",
        agent.reasoning,
        if agent.reasoning == agent.settings.reasoning() {
            format!("  [{}]", agent.settings.reasoning_source())
        } else {
            "  [/reasoning, this session]".to_string()
        }
    );
    outln!(
        "  repetition_retries {}  [{}]",
        agent.settings.repetition_retries(),
        match agent.settings.repetition_retries_setting.value {
            Some(_) => agent.settings.repetition_retries_setting.source,
            None => "built-in default",
        }
    );
    outln!(
        "  system prompt     {}",
        match (agent.settings.use_system_prompt, &agent.system_prompt) {
            (true, _) => "haico's own",
            (false, Some(_)) => "off — project instructions only  [config [models]]",
            (false, None) => "off — nothing sent  [config [models]]",
        }
    );
    outln!(
        "  extra_body        {}",
        if agent.settings.extra_body.is_empty() {
            "-".to_string()
        } else {
            serde_json::Value::Object(agent.settings.extra_body.clone()).to_string()
        }
    );
    // Names only: a header here is as likely as not a key of some kind.
    let headers: Vec<&str> = agent.settings.extra_headers.keys().map(|name| name.as_str()).collect();
    outln!("  extra_headers     {}", if headers.is_empty() { "-".to_string() } else { headers.join(", ") });
    outln!(
        "\nA '-' means the field is left out of the request, so the server's own \
         setting for that model applies."
    );
    outln!("Change one for this session with /set <name> <value>.\n");
}

fn set_knob(agent: &mut Agent, rest: &str) {
    let mut parts = rest.split_whitespace();
    let name = parts.next().unwrap_or("");
    let value = parts.next().unwrap_or("");

    // No name: show everything. What each one *is* comes from the agent; the layout of the
    // listing is this frontend's.
    if name.is_empty() {
        for knob in Knob::ALL {
            outln!("  {:<20}{}", knob.name(), agent.knob(*knob));
        }
        outln!("\nUsage: /set <name> <value>, or /set <name> unset\n");
        return;
    }

    let Some(knob) = Knob::parse(name) else {
        outln!(
            "Unknown setting '{}'. Try: {}.\n",
            name,
            Knob::ALL.iter().map(|k| k.name()).collect::<Vec<_>>().join(", ")
        );
        return;
    };

    // A name with no value asks rather than sets.
    if value.is_empty() {
        outln!("  {:<20}{}\n", knob.name(), agent.knob(knob));
        return;
    }

    match agent.set_knob(knob, Some(value)) {
        Ok(kept) => {
            match agent.knob_value(knob) {
                // "the config file is updated" rather than calling the value shown the
                // saved one: clearing `context` writes too, and what it writes is the
                // key's absence, not the number now in force.
                Some(now) => outln!(
                    "{} = {} for this session{}.",
                    knob.name(),
                    now,
                    match kept {
                        Kept::Config => ", and the config file is updated",
                        _ => "",
                    }
                ),
                None => outln!("{} unset — the server decides again.", knob.name()),
            }
            if let Kept::Failed(why) = kept {
                outln!("  [warning] the config file was not updated: {}", why);
            }
            outln!();
        }
        Err(complaint) => outln!("{}, got '{}'.\n", complaint, value),
    }
}

async fn background_jobs(agent: &Agent, argument: &str) {
    if argument.is_empty() {
        let listing: Vec<String> = {
            let registry = agent.server.background.lock().unwrap();
            registry
                .jobs
                .iter()
                .map(|(id, job)| format!("  {}  [{}]  {}", id, job.status(), job.command))
                .collect()
        };
        if listing.is_empty() {
            outln!("No background jobs.\n");
        } else {
            outln!("Background jobs:\n{}\n", listing.join("\n"));
        }
    } else if argument == "stop" {
        let stopped = agent.server.stop_all_background().await;
        outln!("Stopped {} job(s).\n", stopped);
    } else if let Some(id) = argument.strip_prefix("stop ").and_then(|n| n.trim().parse::<u32>().ok()) {
        let result = agent.server.background_stop(Parameters(BackgroundStopParams { id: Some(id) })).await;
        match result {
            Ok(result) => outln!(
                "{}\n",
                result
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            Err(e) => outln!("Could not stop job {}: {}\n", id, e.message),
        }
    } else {
        outln!("Usage: /bg, /bg stop <id>, /bg stop\n");
    }
}

fn timers_command(agent: &Agent, argument: &str) {
    if argument.is_empty() {
        outln!("{}\n", agent.server.describe_timers());
    } else if argument == "cancel" {
        let cancelled = agent.server.timers.lock().unwrap().clear();
        outln!("Cancelled {} timer(s).\n", cancelled);
    } else if let Some(id) = argument.strip_prefix("cancel ").and_then(|n| n.trim().parse::<u32>().ok()) {
        // The model is not told: it finds out from the reminder never arriving, or from
        // `timer list` if it asks — the same as a background job stopped by /bg.
        match agent.server.timers.lock().unwrap().cancel(id) {
            Some(timer) => outln!("Cancelled timer {} ({}).\n", id, timer.schedule.describe()),
            None => outln!("There is no timer {}.\n", id),
        }
    } else {
        outln!("Usage: /timers, /timers cancel <id>, /timers cancel\n");
    }
}

fn show_todo(agent: &Agent) {
    let todos = agent.plan();
    if todos.is_empty() {
        outln!("No plan yet.\n");
        return;
    }
    outln!("Plan:");
    for item in &todos {
        let mark = match item.status.as_str() {
            "completed" => "x",
            "in_progress" => "»",
            _ => " ",
        };
        outln!("  [{}] {}", mark, item.task);
    }
    outln!();
}

// Session-only, like /set: the config file is where a lasting choice belongs, and the whole
// point of this one is trying the three against each other on the same task without
// editing a file between the runs.
fn reasoning_command(agent: &mut Agent, rest: &str) {
    if rest.is_empty() {
        let source = if agent.reasoning == agent.settings.reasoning() {
            agent.settings.reasoning_source()
        } else {
            "/reasoning, this session"
        };
        outln!("Chain of thought: {}  [{}]", agent.reasoning, source);
        outln!("  keep      send it back to the model with the rest of the history");
        outln!("  compress  replace it with its hard points once the turn is over");
        outln!("  drop      strip it the moment the turn is over");
        outln!(
            "\nCompressing costs one extra request per reply and only runs on a \
             trace over ~{} tokens (/set reasoning_threshold).\n",
            agent.reasoning_threshold
        );
    } else if let Some(chosen) = Reasoning::parse(rest) {
        agent.set_reasoning(chosen);
        outln!("Chain of thought: {} (this session).\n", agent.reasoning);
    } else {
        outln!("[error] unknown setting '{}'. Use keep, compress or drop.\n", rest);
    }
}

fn resume(agent: &mut Agent) {
    if !agent.has_saved_session() {
        outln!("No saved conversation for this directory.\n");
        return;
    }
    // Dropped through `clear` rather than by emptying the history here: what goes with a
    // conversation and what survives it is the core's business, and a second answer to that
    // question in the frontend is the one that would drift. The screen goes with it for
    // `/clear`'s reason — the transcript on it belongs to a conversation that is no longer
    // in the history.
    out::clear_screen();
    let _ = agent.clear();
    match agent.resume() {
        Ok(restored) => outln!(
            "Resumed {} message(s) from {} (model then: {}).\n",
            restored.messages, restored.saved_at, restored.model
        ),
        Err(e) => outln!("Could not resume: {:#}. The conversation is empty.\n", e),
    }
}

fn clear(agent: &mut Agent) {
    // What is discarded, what survives and why is `Agent::clear`'s business. The screen is
    // this frontend's: a transcript of a conversation that no longer exists goes with it,
    // the way it always has in the window. What is left here is saying what happened.
    out::clear_screen();
    match agent.clear() {
        ContextReload::Unchanged => {}
        ContextReload::Reloaded(name) => outln!("Reloaded project context from {}", name),
        ContextReload::Gone => outln!("Project context file is gone; carrying on without it"),
    }
    outln!("Conversation cleared.\n");
}

/// Everything the terminal frontend owns for the length of a session, bundled so
/// the turn loop can be a function rather than nine hundred lines of loop body.
///
/// The split is the point: `Agent` is the conversation with no terminal attached,
/// and this is the terminal with no conversation in it.
struct TerminalIo {
    /// `None` for a piped session, which has no keyboard to read.
    keys: Option<Keys>,
    editor: Editor,
    /// Typed while the model was working. Drained at the next round boundary, or
    /// at the next prompt if the turn finished first.
    queued: Vec<String>,
    /// How much of the plan the panel shows. Lives here rather than in the panel
    /// because Ctrl+T is pressed in one turn and has to still hold in the next.
    plan_view: PlanView,
    /// What the bar above the box says once a run is over: the last thing that
    /// actually happened, rather than an empty row.
    last_activity: Option<String>,
    history_path: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_the_list_offers_runs_something() {
        // A suggestion that runs nothing is worse than none: picked, it would go to the
        // model as a message. Aliases too, since they are what fingers type.
        for command in commands::BUILT_IN {
            for spelling in std::iter::once(&command.name).chain(command.aliases) {
                assert!(command_for(spelling).is_some(), "/{spelling} is offered and runs nothing");
            }
        }
        // An alias runs what its command runs.
        assert_eq!(command_for("new"), command_for("clear"));
        assert_eq!(command_for("quit"), Some(Cmd::Exit));
        // And a name that is not one is left for the skills and the model.
        assert_eq!(command_for("temperature"), None);
    }

    #[test]
    fn only_a_word_after_the_slash_is_taken_for_a_command() {
        assert!(is_command_name("hepl"));
        assert!(is_command_name("update-skills"));
        assert!(is_command_name("my_skill2"));
        assert!(!is_command_name(""));
        assert!(!is_command_name("etc/hosts"));
        assert!(!is_command_name(".config"));
        assert!(!is_command_name("C:\\temp"));
    }

    #[test]
    fn the_help_lists_every_command_from_the_table() {
        let _capture = out::capture_guard();
        out::start_capture();
        print_repl_help();
        let help = out::take_capture();
        for command in commands::BUILT_IN {
            assert!(help.contains(&format!("/{}", command.name)), "/help leaves out /{}", command.name);
            assert!(help.contains(command.about), "/help words /{} differently", command.name);
        }
        assert!(help.contains("/clear /new /reset"), "aliases are shown with their command");
    }

    /// The bar earns its row for what varies, so an unset effort is absent rather than
    /// spelled out: a model with no such setting would otherwise carry a word that never
    /// changes. The two-space gaps are the separator the bar is built from.
    #[test]
    fn the_status_bar_names_the_effort_only_when_one_is_set() {
        assert_eq!(status_line(None, "qwen3", None, 900, 32768), "qwen3 • ctx 3%");
        // The provider, when the config names one, goes before the model it serves.
        assert_eq!(
            status_line(Some("lmstudio"), "qwen3", None, 900, 32768),
            "lmstudio • qwen3 • ctx 3%"
        );
        assert_eq!(
            status_line(None, "qwen3", Some("high"), 900, 32768),
            "qwen3 • high effort • ctx 3%"
        );
        // Shown as given. A level haico does not know is a model it has not met, and the
        // bar is the one place the user can see which one was actually sent.
        assert_eq!(
            status_line(None, "m", Some("xhigh"), 0, 32768),
            "m • xhigh effort • ctx 0%"
        );
    }
}
