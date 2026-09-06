use anyhow::{Context, Result};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::transport::stdio;
use rmcp::ServiceExt;
use std::env;
use std::path::PathBuf;

// Declared first, and with #[macro_use], so out!/outln! are in scope for every
// module below it.
#[macro_use]
mod out;

mod agent;
mod ask;
mod config;
mod debug;
#[cfg(feature = "gui")]
mod gui;
mod input;
mod llm;
#[cfg(feature = "gui")]
mod markdown;
mod modes;
mod permissions;
mod present;
mod prompt;
mod readable;
mod render;
mod session;
#[cfg(test)]
mod testutil;
mod tools;
mod turn;
mod util;
mod workspace;

use agent::*;
use ask::*;
use config::*;
use input::*;
use llm::*;
use modes::*;
use permissions::*;
use present::*;
use prompt::*;
use render::*;
use session::*;
use tools::*;
use turn::*;
use util::*;



fn print_repl_help() {
    outln!("Commands:");
    outln!("  /help              Show this list");
    outln!("  /plan              Research and propose only; nothing changes until approved");
    outln!("  /auto              Review each gated call; routine ones run without asking");
    outln!("  /normal            Back to answering every prompt yourself");
    outln!("  /add <path>        Let the file tools read and write another directory too");
    outln!("  /workspace         Show which paths the file tools may touch");
    outln!("  /mode              Show which of the three is in force");
    outln!("  /init              Study the project and write its instruction file");
    outln!("  /tools             List available tools and whether they prompt");
    outln!("  /model [name]      Switch model; with no name, pick from the server's list");
    outln!("  /config            Show the settings in force and where each came from");
    outln!("  /set [name] [val]  Show or set temperature, top_p, top_k, min_p,");
    outln!("                     repeat_penalty, max_tokens or reasoning_threshold ('/set");
    outln!("                     <name> unset' to undo). '/temp' is short for '/set");
    outln!("                     temperature'.");
    outln!("  /bg [stop [id]]    List background jobs, or stop one or all of them");
    outln!("  /undo              Revert the last file change made by a tool");
    outln!("  /todo              Show the current plan, if the model made one");
    outln!("  /context           Show how much context the conversation is using");
    outln!("  /reasoning [what]  Keep, compress or drop the model's chain of thought");
    outln!("  /compact           Replace the history with a summary, freeing context");
    outln!("  /clear /new /reset Start over, keeping the system prompt");
    outln!("  /exit /quit        Leave (Ctrl+D also works)");
    outln!();
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
    outln!();
}

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
fn status_line(model: &str, used: usize, context_tokens: usize) -> String {
    let percent = (used as f64 / context_tokens.max(1) as f64) * 100.0;
    format!("{}  ctx {:.0}%", model, percent)
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
async fn drive_turn(agent: &mut Agent, io: &mut TerminalIo, input: &str, initializing: bool) {
    let link = TurnLink::new(agent.mode);
    let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
    let mut presenter = Presenter::new();

    // Chrome the pump draws around the transcript. Seeded from the agent before
    // the turn borrows it, and kept current afterwards by the turn's own Status
    // and Busy events — the pump cannot look at the agent again until the turn has
    // given it back. The mode needs no event: the link is shared, so both sides
    // already see the same one.
    let mut status =
        status_line(agent.model(), estimate_tokens(&agent.messages), agent.context_tokens);
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
        let turn = run_turn(agent, input, initializing, &tx, &link);
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
                    answer_or_show(&mut presenter, &mut io.keys, event, &link).await;
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

            let label = crate::input::busy_label(&busy, if counting { streamed } else { 0 });
            let plan = todos.lock().unwrap().clone();
            if dirty || shown.as_deref() != Some(label.as_str()) || shown_plan != plan {
                let queued = link.queued();
                crate::input::draw(
                    &mut io.editor,
                    &crate::input::Chrome {
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
                                AgentEvent::Status { model, used_tokens, context_tokens } => {
                                    status = status_line(model, *used_tokens, *context_tokens);
                                }
                                AgentEvent::ReplyStart => streamed = 0,
                                AgentEvent::ReasoningChunk(text)
                                | AgentEvent::ReplyChunk(text) => {
                                    streamed += text.chars().count();
                                }
                                _ => {}
                            }
                            answer_or_show(&mut presenter, &mut io.keys, event, &link).await;
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
        crate::input::save_history(path, io.editor.history());
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
        Waiting::ReviewingCall => "🤖 Reviewing tool call…".to_string(),
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
    link: &TurnLink,
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
        Ask::Permission { tool_name, arguments, review, allow_all_label, diff, reply } => {
            // The turn sent the call; the sentence above the buttons is made here.
            let description = permission_description(&tool_name, &arguments, review.as_ref());
            let response =
                ask_about_call(keys.as_mut(), &allow_all_label, &description, diff.as_deref()).await;
            if let Ok(response) = response {
                let _ = reply.send(response);
            }
        }
    }
    // A prompt draws over the input region and leaves the cursor wherever it
    // finished; the pump redraws on the next pass, and this keeps the scroll
    // following the end while it does.
    let _ = link;
}

// Rough size of the conversation, in tokens.
//
// Deliberately a heuristic: the server never tells us which tokenizer it uses, and an
// approximation that is always available is worth more here than an exact number that would
// cost another round-trip. Four characters per token is the usual rule of thumb for English
// and code; treat it as an order of magnitude, not a budget.
async fn run_mcp_server(config: Config) -> Result<()> {
    eprintln!("[haico] starting MCP server (stdio)...");

    // web_fetch summarizes pages via the LLM, so it is only exposed when --url (or LLM_URL)
    // says explicitly where that LLM lives. Otherwise it is hidden and only web_get (raw
    // cleaned page text, no LLM round-trip) is exposed. Every other tool works regardless.
    if !config.url_supplied {
        eprintln!("[haico] note: no LLM URL configured (pass --url or set LLM_URL); \
exposing web_get (raw page text) and hiding web_fetch (summarization).");
    }

    // --model is optional here: with none given we omit the "model" field from requests,
    // so the LLM server uses whatever model it already has loaded.
    let model = config.model.unwrap_or_default();
    if config.url_supplied && model.is_empty() {
        eprintln!("[haico] no --model given; using the model already loaded on the LLM server.");
    }

    if let Some(searxng) = &config.searxng {
        eprintln!("[haico] web_search will query SearxNG at {}.", searxng);
    }

    let lm_client = LMStudioClient::new(config.llm_url, model, config.token);
    let server = MyMcpServer::new(lm_client, config.url_supplied, config.searxng, false);

    eprintln!("[haico] MCP server ready; serving over stdio.");
    let running = server
        .serve(stdio())
        .await
        .context("failed to start MCP stdio server")?;
    running
        .waiting()
        .await
        .context("MCP server terminated with error")?;
    eprintln!("[haico] MCP server stopped.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse configuration first so we know the mode before any stdout output.
    let config = parse_config()?;

    // Server mode: hand off to the MCP stdio server and return (no banner, no REPL).
    if config.server {
        return run_mcp_server(config).await;
    }

    // The window, as a third entry beside the REPL and the MCP server. It needs
    // the model chosen first — picking one is a conversation, and the GUI has no
    // picker yet — so `--gui` without `--model` says so rather than opening an
    // empty window.
    if config.gui {
        return run_gui(config);
    }

    // From here on the keyboard is read a key at a time rather than a line at a time,
    // which is what lets the input box exist while the model is working. Piped input
    // has no keyboard to read, so it keeps the plain line-at-a-time path; scripting
    // haico still works.
    let interactive = crate::input::stdin_is_a_terminal();
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

    outln!("          Welcome to Haico!");
    outln!("=====================================\n");
    if let Some(path) = config.config_path.as_deref().filter(|p| p.exists()) {
        outln!("Config: {}", path.display());
    }
    outln!("Connecting to LLM server at: {}", config.llm_url);
    if let Some(searxng) = &config.searxng {
        outln!("Searching through SearxNG at: {}", searxng);
    }

    // Determine which model to use
    let selected_model = if let Some(model_name) = config.model.clone() {
        // Model specified via command-line, use it directly
        outln!("Using specified model: {}\n", model_name);
        model_name
    } else {
        outln!("Looking for a model...");

        // Whatever is already in memory is almost always what was meant: the server was
        // started with it, and asking which model to use when there is only one loaded is
        // a question with a single answer. More than one and it is worth asking, but only
        // among those — the rest are downloads, not something that is running. The full
        // list is what the picker falls back to when nothing is loaded at all.
        let loaded = fetch_loaded_models(&config.llm_url, config.token.as_deref()).await;
        if let [only] = loaded.as_slice() {
            outln!("Using the loaded model: {}\n", only);
            only.clone()
        } else {
            let models = if loaded.is_empty() {
                match fetch_available_models(&config.llm_url, config.token.as_deref()).await {
                    Ok(models) => models,
                    Err(e) => {
                        outln!("Error connecting to LLM: {e}");
                        return Ok(());
                    }
                }
            } else {
                loaded
            };

            if models.is_empty() {
                return Err(anyhow::anyhow!("No models available from server"));
            }

            // Let user select a model interactively
            match keys.as_mut() {
                Some(keys) => select_model_interactive(keys, models).await?,
                None => {
                    return Err(anyhow::anyhow!(
                        "choosing a model needs an interactive terminal; pass --model"
                    ))
                }
            }
        }
    };

    // Everything the conversation itself needs — client, tools, history,
    // permissions, workspace, mode — now lives in one place with no terminal
    // attached (`src/agent.rs`). What stays out here is this frontend's own:
    // the keyboard, the editor, and every line printed.
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let resume_requested = config.resume;
    let mut agent = Agent::new(config, selected_model, &cwd);

    // The core reports what it loaded rather than announcing it, so the saying
    // of it stays here with the rest of the output.
    if let Some((name, _)) = &agent.project_context {
        outln!("Loaded project context from {}", name);
    }
    if agent.user_context.is_some() {
        outln!("Loaded user context from {}", USER_CONTEXT_FILE);
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
        outln!("A previous conversation exists here — start with --resume to continue it.");
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

    // Line editing and history. The editor holds what is being typed; the writer keeps
    // it pinned to the bottom of the screen and prints everything else above it.
    let history_path = history_file_path();
    let history = history_path
        .as_deref()
        .map(crate::input::load_history)
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

    outln!("Type /help for commands.\n");

    // Main loop
    loop {
        // Read user input — unless there is already something waiting that was typed
        // while the last turn was running.
        let input = if !io.queued.is_empty() {
            // It waited above the bar for as long as it was unsent. Now that it is the
            // prompt, it becomes a plain line of the transcript like anything else typed.
            let text = io.queued.remove(0);
            outln!("{}", echo(PROMPT, &text));
            text
        } else {
            match io.keys.as_mut() {
                Some(keys) => {
                    let status = status_line(agent.model(), estimate_tokens(&agent.messages), agent.context_tokens);
                    let plan = agent.plan();
                    match read_command(
                        keys,
                        &mut io.editor,
                        &mut agent.mode,
                        &status,
                        io.last_activity.as_deref(),
                        &plan,
                        &mut io.plan_view,
                    )
                    .await
                    {
                        // The finished entry becomes a plain line of the transcript, so
                        // the conversation reads as one when scrolled back.
                        Input::Text(text) => {
                            outln!("{}", echo(PROMPT, &text));
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
                None => match crate::input::read_piped_line() {
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

        if input == "/exit" || input == "/quit" {
            outln!("Goodbye!");
            break;
        }

        if input == "/help" {
            print_repl_help();
            continue;
        }

        if input == "/add" || input.starts_with("/add ") {
            let requested = input.strip_prefix("/add").unwrap_or("").trim();
            if requested.is_empty() {
                outln!("Usage: /add <path>   (a directory to read and write for this session)\n");
                continue;
            }
            match agent.workspace.add(std::path::Path::new(requested)) {
                Ok(added) => outln!("Added to the workspace: {}\n", display_path(&added)),
                Err(reason) => outln!("{}\n", reason),
            }
            continue;
        }

        if input == "/workspace" {
            outln!("Readable and writable:");
            for root in agent.workspace.writable_roots() {
                outln!("  {}", display_path(root));
            }
            if agent.workspace.readable_roots().is_empty() {
                outln!("Readable only: none configured (see 'read_paths' in the config file).");
            } else {
                outln!("Readable only:");
                for root in agent.workspace.readable_roots() {
                    outln!("  {}", display_path(root));
                }
            }
            outln!("\nAnything else is refused outright, for reads as well as writes.\n");
            continue;
        }

        if matches!(input, "/plan" | "/auto" | "/normal" | "/mode") {
            let requested = match input {
                "/plan" => Some(Mode::Plan),
                "/auto" => Some(Mode::Auto),
                "/normal" => Some(Mode::Normal),
                _ => None,
            };
            if let Some(requested) = requested {
                agent.mode = requested;
            }
            outln!("Mode: {}\n", agent.mode.describe());
            if agent.mode == Mode::Auto {
                // Spelled out, because this is the mode that acts without asking and the
                // user should know exactly how far that goes before leaving it on.
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
            continue;
        }

        if input == "/clear" || input == "/new" || input == "/reset" {
            // What is discarded, what survives and why is `Agent::clear`'s
            // business; all that is left here is saying what happened.
            match agent.clear() {
                ContextReload::Unchanged => {}
                ContextReload::Reloaded(name) => outln!("Reloaded project context from {}", name),
                ContextReload::Gone => {
                    outln!("Project context file is gone; carrying on without it")
                }
            }
            outln!("Conversation cleared.\n");
            continue;
        }

        if input == "/tools" {
            outln!("{} tools available:", agent.server.tool_router.list_all().len());
            for tool in agent.server.tool_router.list_all() {
                let gate = if AUTO_ALLOWED_TOOLS.contains(&tool.name.as_ref()) {
                    "auto"
                } else {
                    "asks"
                };
                outln!("  {:<14} [{}]", tool.name, gate);
            }
            outln!();
            continue;
        }

        if input == "/model" || input.starts_with("/model ") {
            let requested = input.strip_prefix("/model").unwrap_or("").trim().to_string();
            // Copied out so the borrow ends before agent.client is reassigned below.
            let llm_url = agent.client.base_url.clone();
            let token = agent.client.token.clone();

            let chosen = if requested.is_empty() {
                match fetch_available_models(&llm_url, token.as_deref()).await {
                    Ok(models) if !models.is_empty() => match pick_model(io.keys.as_mut(), models).await {
                        Ok(model) => Some(model),
                        Err(e) => {
                            outln!("Model selection cancelled: {}\n", e);
                            None
                        }
                    },
                    Ok(_) => {
                        outln!("The server reports no available models.\n");
                        None
                    }
                    Err(e) => {
                        outln!("Could not list models: {:#}\n", e);
                        None
                    }
                }
            } else {
                Some(requested)
            };

            if let Some(model) = chosen {
                // Everything that is configured per model is re-resolved inside
                // `switch_model` — the context budget, the sampling, the
                // reasoning policy, both copies of the client, and the system
                // message. The conversation itself is kept.
                agent.switch_model(model.clone());

                outln!("Now using model: {}", model);
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
            continue;
        }

        if input == "/config" {
            match agent.config.config_path.as_deref() {
                Some(path) if path.exists() => outln!("Config file: {}", path.display()),
                Some(path) => outln!("Config file: {} (not created yet)", path.display()),
                None => outln!("Config file: unavailable (no home directory)"),
            }
            outln!("
In force for '{}':", agent.client.model);
            outln!("  url             {}", agent.client.base_url);
            outln!(
                "  token           {}",
                if agent.client.token.is_some() { "set" } else { "not set" }
            );
            outln!(
                "  context         {}  [{}]",
                agent.context_tokens, agent.settings.context_source
            );
            outln!("  temperature     {}", agent.settings.temperature.describe());
            outln!("  top_p           {}", agent.settings.top_p.describe());
            outln!("  top_k           {}", agent.settings.top_k.describe());
            outln!("  min_p           {}", agent.settings.min_p.describe());
            outln!("  repeat_penalty  {}", agent.settings.repeat_penalty.describe());
            outln!("  max_tokens      {}", agent.settings.max_tokens.describe());
            // Shown as what is actually in force rather than as what the file said, since
            // /reasoning can have moved it since. The source string is only honest for the
            // former case, so it is replaced rather than kept when they differ.
            outln!(
                "  reasoning       {}{}",
                agent.reasoning,
                if agent.reasoning == agent.settings.reasoning() {
                    format!("  [{}]", agent.settings.reasoning_source())
                } else {
                    "  [/reasoning, this session]".to_string()
                }
            );
            outln!(
                "  system prompt   {}",
                match (agent.settings.use_system_prompt, &agent.system_prompt) {
                    (true, _) => "haico's own",
                    (false, Some(_)) => "off — project instructions only  [config [models]]",
                    (false, None) => "off — nothing sent  [config [models]]",
                }
            );
            outln!(
                "\nA '-' means the field is left out of the request, so the server's own \
                 setting for that model applies."
            );
            outln!("Change one for this session with /set <name> <value>.\n");
            continue;
        }

        // One command for every sampling knob rather than one command each. "/temp" is kept
        // as a shorthand for the one that gets touched most.
        if input == "/set" || input.starts_with("/set ") || input.starts_with("/temp") {
            // `/temp 0.7` is the shorthand that predates `/set`, kept because it
            // is what fingers already know.
            let rest = if let Some(rest) = input.strip_prefix("/temp") {
                format!("temperature {}", rest.trim())
            } else {
                input.strip_prefix("/set").unwrap_or("").trim().to_string()
            };
            let mut parts = rest.split_whitespace();
            let name = parts.next().unwrap_or("");
            let value = parts.next().unwrap_or("");

            // No name: show everything. What each one *is* comes from the agent;
            // the layout of the listing is this frontend's.
            if name.is_empty() {
                for knob in Knob::ALL {
                    outln!("  {:<20}{}", knob.name(), agent.knob(*knob));
                }
                outln!("\nUsage: /set <name> <value>, or /set <name> unset\n");
                continue;
            }

            let Some(knob) = Knob::parse(name) else {
                outln!(
                    "Unknown setting '{}'. Try: {}.\n",
                    name,
                    Knob::ALL.iter().map(|k| k.name()).collect::<Vec<_>>().join(", ")
                );
                continue;
            };

            // A name with no value asks rather than sets.
            if value.is_empty() {
                outln!("  {:<20}{}\n", knob.name(), agent.knob(knob));
                continue;
            }

            match agent.set_knob(knob, Some(value)) {
                Ok(()) => match agent.knob_value(knob) {
                    Some(now) => outln!("{} = {} for this session.\n", knob.name(), now),
                    None => outln!("{} unset — the server decides again.\n", knob.name()),
                },
                Err(complaint) => outln!("{}, got '{}'.\n", complaint, value),
            }
            continue;
        }

        if input == "/bg" || input.starts_with("/bg ") {
            let argument = input.strip_prefix("/bg").unwrap_or("").trim().to_string();

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
                let result = agent.server
                    .background_stop(Parameters(BackgroundStopParams { id: Some(id) }))
                    .await;
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
                    Err(e) => outln!("Could not stop job {}: {:?}\n", id, e),
                }
            } else {
                outln!("Usage: /bg, /bg stop <id>, /bg stop\n");
            }
            continue;
        }

        if input == "/undo" {
            match agent.server.undo_last() {
                Ok(what) => outln!("Reverted: {}\n", what),
                Err(why) => outln!("{}\n", why),
            }
            continue;
        }

        if input == "/todo" {
            let todos = agent.plan();
            if todos.is_empty() {
                outln!("No plan yet.\n");
            } else {
                outln!("Plan:");
                for item in &todos {
                    let mark = match item.status.as_str() {
                        "done" => "x",
                        "in_progress" => "»",
                        _ => " ",
                    };
                    outln!("  [{}] {}", mark, item.task);
                }
                outln!();
            }
            continue;
        }

        if input == "/context" {
            report_context(&agent.messages, agent.context_tokens, &agent.reasoning_stats);
            continue;
        }

        // Session-only, like /set: the config file is where a lasting choice belongs, and
        // the whole point of this one is trying the three against each other on the same
        // task without editing a file between the runs.
        if input == "/reasoning" || input.starts_with("/reasoning ") {
            let rest = input.strip_prefix("/reasoning").unwrap_or("").trim().to_string();
            if rest.is_empty() {
                let source = if agent.reasoning == agent.settings.reasoning() {
                    agent.settings.reasoning_source()
                } else {
                    "/reasoning, this session"
                };
                outln!("Chain of thought: {}  [{}]", agent.reasoning, source);
                outln!(
                    "  keep      send it back to the model with the rest of the history"
                );
                outln!(
                    "  compress  replace it with its hard points once the turn is over"
                );
                outln!("  drop      strip it the moment the turn is over");
                outln!(
                    "\nCompressing costs one extra request per reply and only runs on a \
                     trace over ~{} tokens (/set reasoning_threshold).\n",
                    agent.reasoning_threshold
                );
            } else if let Some(chosen) = Reasoning::parse(&rest) {
                agent.reasoning = chosen;
                outln!("Chain of thought: {} (this session).\n", agent.reasoning);
            } else {
                outln!("[error] unknown setting '{}'. Use keep, compress or drop.\n", rest);
            }
            continue;
        }

        if input == "/compact" {
            match compact_history(&agent.client, &mut agent.messages).await {
                // The same reclaim as /clear, on the one path that actually shrank the
                // session. A compaction that failed freed nothing to go with it.
                Ok(summary) => {
                    agent.review_cache.clear();
                    outln!("{}\n", summary)
                }
                Err(e) => outln!("[error] could not compact: {:#}\n", e),
            }
            continue;
        }

        // The one "command" that is not one: /init does not do anything itself, it expands
        // into a long instruction and runs as an ordinary turn. Written that way so it gets
        // the tools, the permission prompts and /undo for free — there is nothing about
        // writing an instruction file that needs its own machinery.
        let init_instruction;
        let mut initializing = false;
        let input = if input == "/init" {
            let target = init_target_file(&cwd);
            // It ends in write_file, which plan mode strips from the request. Better to say
            // so now than to have the model research the whole project and then find it
            // cannot save the result.
            if agent.mode == Mode::Plan {
                outln!(
                    "/init writes {}, which plan mode does not allow. Switch with /normal first.\n",
                    target
                );
                continue;
            }
            outln!("Analyzing the project to write {}...\n", target);
            initializing = true;
            init_instruction = build_init_prompt(&cwd);
            init_instruction.as_str()
        } else {
            input
        };

        drive_turn(&mut agent, &mut io, input, initializing).await;
    }

    if let Some(path) = &io.history_path {
        crate::input::save_history(path, io.editor.history());
    }

    // Every way out of the loop lands here, so a dev server started mid-session does not
    // outlive it holding a port. Silent when there is nothing to do.
    let stopped = agent.server.stop_all_background().await;
    if stopped > 0 {
        outln!("Stopped {} background job(s).", stopped);
    }

    Ok(())
}

// ROUNDTRIP

fn get_thinking_message() -> &'static str {
    use std::time::{SystemTime, UNIX_EPOCH};
    const MESSAGES: &[&str] = &[
        // First step
        "Fancying around…",
        "Thinking deeply…",
        "Connecting the dots…",
        "Pondering possibilities…",
        "Weighing options…",
        "Untangling logic…",
        "Simulating neurons…",
        "Brewing ideas…",
        "Scanning the horizon…",
        "Assembling the puzzle…",

        // Technical / Agent-y
        "Parsing intent…",
        "Traversing the graph…",
        "Running inference…",
        "Compiling thoughts…",
        "Optimizing the plan…",
        "Executing mental threads…",
        "Querying the knowledge base…",
        "Running a quick sanity check…",
        "Bootstrapping context…",
        "Refining the approach…",

        // Whimsical / Personable
        "Staring into the void (productively)…",
        "Consulting the oracle…",
        "Summoning the code spirits…",
        "Bribing the compiler…",
        "Rubber-ducking internally…",
        "Channeling Ada Lovelace…",
        "Polishing the crystal ball…",
        "Negotiating with entropy…",
        "Warming up the neurons…",
        "Reticulating splines…",

        // Concise / Dry
        "Processing…",
        "Analyzing…",
        "Evaluating…",
        "Computing…",
        "Reasoning…",
        "Synthesizing…",
        "Deliberating…",
        "Calculating…",
        "Inspecting…",
        "Contemplating…",

        // Creative / Poetic
        "Following the thread…",
        "Tracing constellations…",
        "Weaving the narrative…",
        "Tuning the frequency…",
        "Listening to the silence…",
        "Reading between the lines…",
        "Chasing the spark…",
        "Stirring the cauldron…",
        "Sharpening the quill…",
        "Lighting the lantern…",
    ];

    let idx = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as usize % MESSAGES.len();

    MESSAGES[idx]
}

// The verb in the line printed when a turn ends. A companion to the list above: the
// spinner says what it is doing in the present tense, this says what it did.
//
// Every one of these has to read as an intransitive past tense — "Baked for 3m 34s" —
// so verbs that want an object ("Untangled", "Assembled") stay out however well they
// fit the spinner.
fn get_done_verb() -> &'static str {
    use std::time::{SystemTime, UNIX_EPOCH};
    const VERBS: &[&str] = &[
        // Kitchen
        "Baked",
        "Brewed",
        "Simmered",
        "Stewed",
        "Marinated",
        "Percolated",
        "Steeped",
        "Kneaded",

        // Thought
        "Pondered",
        "Ruminated",
        "Cogitated",
        "Deliberated",
        "Mused",
        "Reasoned",
        "Reflected",
        "Deduced",

        // Puttering
        "Noodled",
        "Tinkered",
        "Puttered",
        "Pottered",
        "Fiddled",
        "Meandered",
        "Rummaged",
        "Foraged",

        // Machinery
        "Churned",
        "Whirred",
        "Hummed",
        "Ticked",
        "Crunched",
        "Toiled",
        "Labored",
        "Computed",
    ];

    let idx = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as usize % VERBS.len();

    VERBS[idx]
}

// How long the turn took, in the largest two units that carry information.
//
// Seconds are dropped past the hour on purpose: a turn that ran for over an hour is not
// one anybody is counting seconds on, and "1h 4m 12s" is three numbers to read where two
// say the same thing.
fn format_elapsed(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

// The line that closes a turn: what it did, how long for, and the time it finished at.
//
// Both halves earn their place. A turn on a local model can run for minutes with its own
// start scrolled far off the top of the terminal, so the duration is not recoverable from
// what is on screen; and a turn that finished while nobody was watching wants a clock
// time, because a duration alone does not say how stale the answer is.
fn turn_summary(verb: &str, elapsed: std::time::Duration, finished: chrono::DateTime<chrono::Local>) -> String {
    format!(
        "✻ {} for {} · done {}",
        verb,
        format_elapsed(elapsed),
        finished.format("%-I:%M %p")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use std::time::Duration;

    #[test]
    fn elapsed_uses_two_units_at_most() {
        assert_eq!(format_elapsed(Duration::from_millis(400)), "0s");
        assert_eq!(format_elapsed(Duration::from_secs(12)), "12s");
        assert_eq!(format_elapsed(Duration::from_secs(59)), "59s");
        // The minute boundary keeps the seconds, even when there are none of them: a bare
        // "1m" reads as a rounded figure where "1m 0s" reads as the measurement it is.
        assert_eq!(format_elapsed(Duration::from_secs(60)), "1m 0s");
        assert_eq!(format_elapsed(Duration::from_secs(214)), "3m 34s");
        assert_eq!(format_elapsed(Duration::from_secs(3599)), "59m 59s");
        // Past the hour the seconds go, rather than becoming a third number.
        assert_eq!(format_elapsed(Duration::from_secs(3600)), "1h 0m");
        assert_eq!(format_elapsed(Duration::from_secs(3852)), "1h 4m");
    }

    #[test]
    fn summary_reads_as_one_sentence() {
        let finished = chrono::Local.with_ymd_and_hms(2026, 8, 28, 2, 13, 0).unwrap();
        assert_eq!(
            turn_summary("Baked", Duration::from_secs(214), finished),
            "✻ Baked for 3m 34s · done 2:13 AM"
        );

        // The hour is unpadded — "2:13 AM", never "02:13 AM" — and the afternoon still
        // reads on a twelve-hour clock.
        let finished = chrono::Local.with_ymd_and_hms(2026, 8, 28, 14, 5, 0).unwrap();
        assert_eq!(
            turn_summary("Pottered", Duration::from_secs(9), finished),
            "✻ Pottered for 9s · done 2:05 PM"
        );
    }

    #[test]
    fn every_done_verb_reads_intransitively() {
        // The list is picked from by clock, so nothing here can check the choice — what it
        // can check is that whatever comes out slots into the sentence without a space or
        // a stray capital in the middle of it.
        let verb = get_done_verb();
        assert!(!verb.is_empty());
        assert!(verb.chars().all(|c| c.is_ascii_alphabetic()));
        assert!(verb.starts_with(|c: char| c.is_uppercase()));
    }
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

/// One turn: the user's message in, and everything that follows from it — rounds
/// of tool calls, the prompts they need answering, the reply as it streams.
///
/// Pulled out of `main` so that it is a callable unit instead of a loop body.
/// That is the precondition for the next step rather than the end of one: it
/// still prints, and still owns the keyboard, but there is now a single seam to
/// invert when the core starts emitting events instead of drawing (11.2).

/// Tests that drive a whole turn against a fake LLM server. Separate from the
/// module above because they need a Tokio runtime and the output capture, and
/// because what they are pinning is different in kind: not a helper's output but
/// the loop's invariants — the ones whose failure mode is the *next* request
/// being rejected rather than anything visible in this one.
#[cfg(test)]
mod turn_tests {
    use super::*;
    use serde_json::Value;
    use crate::testutil::{serve_sequence, test_config};

    /// One turn against `bodies`, returning the agent so the history can be
    /// inspected, plus how many requests the server actually served.
    async fn run_one(bodies: &'static [&'static str], input: &str) -> (Agent, usize) {
        run_one_in(bodies, input, Mode::Normal).await
    }

    /// As `run_one`, in a chosen mode. Auto mode matters here because it is the
    /// only one that skips `MAX_TOOL_ROUNDS`, so it is the only one where a
    /// runaway turn has nothing but the repetition guard to stop it.
    async fn run_one_in(
        bodies: &'static [&'static str],
        input: &str,
        mode: Mode,
    ) -> (Agent, usize) {
        run_and_watch(bodies, input, mode).await.0
    }

    /// As `run_one_in`, but also every event the turn emitted.
    ///
    /// This is what the event stream bought the tests: the turn is driven with no
    /// frontend at all — no terminal, no keyboard, no presenter — and what it
    /// *said* can be asserted on directly instead of inferred from output. An
    /// `Ask` arriving here is answered by nobody, so its reply channel drops and
    /// the turn hears a refusal, which is exactly what a test wants: no prompt
    /// can hang the suite.
    async fn run_and_watch(
        bodies: &'static [&'static str],
        input: &str,
        mode: Mode,
    ) -> ((Agent, usize), Vec<AgentEvent>) {
        let (url, served) = serve_sequence(bodies).await;
        let cwd = std::env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        agent.mode = mode;

        let link = TurnLink::new(agent.mode);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, input, false, &tx, &link).await;
        agent.mode = link.mode();

        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        let count = *served.lock().unwrap();
        ((agent, count), seen)
    }

    /// Every `Notice` the turn emitted, in order. The notices are the part of the
    /// vocabulary that derives `PartialEq`, which is why assertions use them.
    fn notices(events: &[AgentEvent]) -> Vec<Notice> {
        events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Notice(notice) => Some(notice.clone()),
                _ => None,
            })
            .collect()
    }

    /// Every `role: "tool"` reply in the history, as `(tool_call_id, content)`.
    fn tool_replies(messages: &[Value]) -> Vec<(String, String)> {
        messages
            .iter()
            .filter(|m| m.get("role").and_then(Value::as_str) == Some("tool"))
            .map(|m| {
                (
                    m.get("tool_call_id").and_then(Value::as_str).unwrap_or_default().to_string(),
                    m.get("content").and_then(Value::as_str).unwrap_or_default().to_string(),
                )
            })
            .collect()
    }

    /// The invariant that costs the most when it breaks, and shows nothing when
    /// it does: **every `tool_call_id` must get a `role: "tool"` reply**, on
    /// every path. An unanswered id is accepted by this request and rejected by
    /// the *next* one, server-side, which is why it took a release to find and
    /// why it is worth a test that asserts on the history rather than the screen.
    ///
    /// Both malformed shapes are covered at once because both used to `continue`
    /// past the reply: a call with no function name, and one whose arguments are
    /// not JSON. Local models emit the second often enough that it was the
    /// original bug report.
    #[tokio::test]
    async fn every_tool_call_id_is_answered_even_when_malformed() {
        let _capture = crate::out::capture_guard();
        const MALFORMED: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[
                {"id":"call_no_name","type":"function","function":{"arguments":"{}"}},
                {"id":"call_bad_args","type":"function","function":{"name":"read","arguments":"not json at all"}}
            ]}}]}"#;
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"all done"}}]}"#;

        let (agent, served) = run_one(&[MALFORMED, FINAL], "do something").await;
        let replies = tool_replies(&agent.messages);
        let ids: Vec<&str> = replies.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.contains(&"call_no_name") && ids.contains(&"call_bad_args"),
            "an id went unanswered — the next request would be rejected. Replies: {replies:?}"
        );

        // Answered *with something the model can act on*, not merely answered:
        // the point of replying to a malformed call is to say what was wrong.
        let bad_args = replies.iter().find(|(id, _)| id == "call_bad_args").unwrap();
        assert!(
            bad_args.1.contains("read"),
            "the reply does not name the tool that failed: {:?}",
            bad_args.1
        );

        // And the turn went on to a second request and finished on the answer,
        // rather than stopping at the malformed round.
        assert_eq!(served, 2, "the turn did not come back for a final answer");
        assert_eq!(
            agent.messages.last().and_then(|m| m.get("content")).and_then(Value::as_str),
            Some("all done")
        );
    }

    /// A reply with no tool calls ends the turn on the answer, and the user's
    /// message and the assistant's both land in the history — the ordinary case,
    /// asserted so that the interesting ones above have a baseline.
    #[tokio::test]
    async fn a_plain_answer_ends_the_turn() {
        let _capture = crate::out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"hello back"}}]}"#;

        let (agent, served) = run_one(&[ANSWER], "hello").await;
        assert_eq!(served, 1, "a turn with no tool calls should take one request");
        let roles: Vec<&str> =
            agent.messages.iter().filter_map(|m| m.get("role").and_then(Value::as_str)).collect();
        assert_eq!(roles.last(), Some(&"assistant"));
        assert!(roles.contains(&"user"));
        assert!(tool_replies(&agent.messages).is_empty());
    }

    /// `MAX_TOOL_ROUNDS` is not the only thing stopping a runaway turn, and in
    /// auto mode it is not applied at all — the guard there is the identical call
    /// repeated with nothing in between. A server that answers with the same
    /// tool call forever must therefore still terminate.
    #[tokio::test]
    async fn an_identical_call_repeated_forever_still_terminates() {
        let _capture = crate::out::capture_guard();
        // `serve_sequence` repeats its last body forever, so this is a model that
        // never stops asking for the same thing.
        const SAME: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"same","type":"function","function":{"name":"read","arguments":"also not json"}}]}}]}"#;

        let (agent, served) = run_one(&[SAME], "loop forever").await;
        // The bound is what matters, not its exact value: without one this test
        // would hang rather than fail.
        assert!(served > 1, "the loop never came back for a second round");
        // The claim is specifically that the *repetition* guard stopped it and
        // not the round cap: `LOOP_BREAK` is 5 against a cap of 50, so a count
        // in single figures is the identical-call run firing. A range rather
        // than an exact number, because the nudge rounds are free to move.
        assert!(
            served <= 8,
            "a repeated identical call ran for {served} rounds — that is the round cap \
             stopping it, not the loop guard"
        );
        // However it stopped, every id it made is still answered.
        let unanswered = agent
            .messages
            .iter()
            .filter(|m| m.get("tool_calls").is_some_and(|c| !c.is_null()))
            .count()
            - tool_replies(&agent.messages).len();
        assert_eq!(unanswered, 0, "a round left its tool call unanswered");
    }

    /// The same runaway turn in **auto mode**, which is where it actually
    /// mattered. Auto mode skips `MAX_TOOL_ROUNDS` deliberately — a fixed number
    /// of rounds is a poor stand-in for "this is going nowhere", and a long job
    /// that hits it unattended has nobody to say "continue" — so the repetition
    /// guard is the *only* thing bounding it.
    ///
    /// This is the test that found the bug it now guards: the run counter used to
    /// be incremented after `parse_tool_arguments`, whose error arm `continue`s,
    /// so a model repeating a call whose arguments are not JSON never advanced
    /// the run. In normal mode the cap still caught it. In auto mode nothing did,
    /// and the turn span forever with no user present to stop it. Written second,
    /// because writing it first would have hung the suite rather than failed.
    #[tokio::test]
    async fn auto_mode_still_bounds_a_repeated_malformed_call() {
        let _capture = crate::out::capture_guard();
        const SAME: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"same","type":"function","function":{"name":"read","arguments":"still not json"}}]}}]}"#;

        let (agent, served) = run_one_in(&[SAME], "loop forever", Mode::Auto).await;
        assert!(
            served <= 8,
            "auto mode ran {served} rounds of an identical malformed call: with no round 
             cap there, the repetition guard is all there is and it did not fire"
        );
        assert_eq!(
            agent.messages.iter().filter(|m| m.get("tool_calls").is_some_and(|c| !c.is_null())).count()
                - tool_replies(&agent.messages).len(),
            0,
            "a round left its tool call unanswered"
        );
    }

    /// The turn narrates itself, and this is the assertion that could not be
    /// written before: driven with no frontend at all, what it *said* is checked
    /// directly rather than inferred from terminal output.
    #[tokio::test]
    async fn a_malformed_call_is_narrated_as_well_as_answered() {
        let _capture = crate::out::capture_guard();
        const MALFORMED: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"bad","type":"function","function":{"name":"read","arguments":"nope"}}]}}]}"#;
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"done"}}]}"#;

        let (_, events) = run_and_watch(&[MALFORMED, FINAL], "go", Mode::Normal).await;
        let said = notices(&events);
        assert!(
            said.iter().any(|n| matches!(
                n,
                Notice::ToolArgumentsInvalid { tool, .. } if tool == "read"
            )),
            "the turn did not say which tool had bad arguments: {said:?}"
        );

        // A reply is opened and closed exactly once per round, on every path.
        // The presenter's fold and half-rendered markdown line depend on that
        // pairing, and a missing ReplyEnd leaks a fold into the next round.
        let starts = events.iter().filter(|e| matches!(e, AgentEvent::ReplyStart)).count();
        let ends = events.iter().filter(|e| matches!(e, AgentEvent::ReplyEnd)).count();
        assert_eq!(starts, 2, "two rounds should open two replies");
        assert_eq!(starts, ends, "every reply must be closed: {starts} opened, {ends} closed");

        // And the turn reported its own end, which is the line the summary is
        // drawn from.
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Finished { .. })));
    }

    /// Cancellation is a message, not a dropped future — see
    /// `TurnLink::until_cancelled`. Cancelling before the turn starts means it
    /// stops at its first interruptible point, and the invariant still holds: the
    /// user message is in the history and nothing is left half-written.
    #[tokio::test]
    async fn a_cancelled_turn_stops_without_orphaning_anything() {
        let _capture = crate::out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"hello"}}]}"#;

        let (url, served) = serve_sequence(&[ANSWER]).await;
        let cwd = std::env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        let link = TurnLink::new(agent.mode);
        // Cancelled up front: `until_cancelled` has to see a flag that was already
        // set, not only a transition after it subscribed.
        link.cancel();

        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, "hello", false, &tx, &link).await;

        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(
            notices(&seen).contains(&Notice::Interrupted),
            "a cancelled turn should say so: {:?}",
            notices(&seen)
        );
        assert_eq!(*served.lock().unwrap(), 0, "cancelled before the first request, so none went");
        // The user's message is there and nothing else is: no assistant message
        // was pushed, so no tool call can be unanswered.
        let roles: Vec<&str> =
            agent.messages.iter().filter_map(|m| m.get("role").and_then(Value::as_str)).collect();
        assert_eq!(roles.last(), Some(&"user"));
        assert!(tool_replies(&agent.messages).is_empty());
    }
}

/// The window frontend, when this binary was built with it.
///
/// Not `async`: winit owns the main thread and `lumio::run` blocks on it, so the
/// Tokio runtime the conversation needs is created on the worker inside
/// `gui::run` instead. That is also why this is reached before `main` takes the
/// terminal — there is no terminal to take.
#[cfg(feature = "gui")]
fn run_gui(config: Config) -> Result<()> {
    let Some(model) = config.model.clone().filter(|m| !m.is_empty()) else {
        eprintln!(
            "--gui needs --model for now: choosing a model is a conversation, and the window \
             has no picker yet."
        );
        return Ok(());
    };
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut agent = Agent::new(config, model, &cwd);
    // Same as the REPL: resume only when asked, and report nothing if there is
    // nothing to report — the window shows its own greeting.
    if agent.config.resume {
        if let Err(e) = agent.resume() {
            eprintln!("could not resume: {e:#}");
        }
    }
    gui::run(agent)
}

/// The same flag in a build without the feature. Says so and exits, rather than
/// falling back to the REPL — a silent fallback looks like the flag worked.
#[cfg(not(feature = "gui"))]
fn run_gui(_config: Config) -> Result<()> {
    eprintln!(
        "this haico was built without the GUI. Rebuild with `cargo build --release \
         --features gui` to use --gui."
    );
    Ok(())
}
