use anyhow::{Context, Result};
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::transport::stdio;
use rmcp::ServiceExt;
use serde_json::{json, Value};
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

// Declared first, and with #[macro_use], so out!/outln! are in scope for every
// module below it.
#[macro_use]
mod out;

mod agent;
mod ask;
mod config;
mod debug;
mod input;
mod llm;
mod modes;
mod permissions;
mod prompt;
mod readable;
mod render;
mod session;
#[cfg(test)]
mod testutil;
mod tools;
mod util;
mod workspace;

use agent::*;
use ask::*;
use config::*;
use input::*;
use llm::*;
use modes::*;
use permissions::*;
use prompt::*;
use render::*;
use session::*;
use tools::*;
use util::*;


// MCP Server with tools
// Lay a command out so it can actually be read before being approved.
//
// It used to be cut at 50 characters, which for a generated one-liner meant approving a
// script sight unseen. Wrapped to the terminal width and indented, with a generous ceiling
// so a pathological command cannot flood the screen either.
fn indented_command(command: &str) -> String {
    const MAX: usize = 4000;
    let shown = truncate_at_boundary(command.trim(), MAX);
    let width = crossterm::terminal::size()
        .map(|(cols, _)| (cols as usize).saturating_sub(6).max(40))
        .unwrap_or(100);

    let mut out = String::new();
    for line in shown.lines() {
        let mut rest = line;
        while !rest.is_empty() {
            let mut take = truncate_at_boundary(rest, width);
            if take.is_empty() {
                take = rest;
            }
            // Break at a space when there is one near the end of the line. Splitting through
            // the middle of a token is legible but jarring, and the whole point here is that
            // the command can be read. Never break early enough to hide anything.
            if take.len() < rest.len() {
                if let Some(space) = take.rfind(' ').filter(|at| *at > width * 3 / 4) {
                    take = &take[..space + 1];
                }
            }
            out.push_str("    ");
            out.push_str(take.trim_end());
            out.push('\n');
            rest = rest[take.len()..].trim_start_matches(' ');
        }
    }
    if shown.len() < command.trim().len() {
        out.push_str("    ... [truncated]\n");
    }
    out.trim_end().to_string()
}

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
fn status_line(model: &str, messages: &[Value], context_tokens: usize) -> String {
    let used = estimate_tokens(messages);
    let percent = (used as f64 / context_tokens.max(1) as f64) * 100.0;
    format!("{}  ctx {:.0}%", model, percent)
}

// Run something with the input box still live underneath it.
//
// This is what the whole rework is for: while the model is thinking, or a tool is
// running, the user can still type. What they submit is queued for the end of the
// round rather than interrupting it — a message injected between an assistant turn
// that called tools and the results of those calls is a message order the server
// rejects. Esc, on the other hand, does interrupt: None comes back and the turn ends.
async fn work_with_input<T>(
    keys: &mut Option<Keys>,
    editor: &mut Editor,
    mode: &mut Mode,
    queued: &mut Vec<String>,
    status: &str,
    // The plan, read fresh on every redraw rather than snapshotted once: todo_write runs
    // inside the very work this is waiting on, and a panel that only caught up at the end
    // of the turn would be worth nothing.
    todos: &std::sync::Mutex<Vec<TodoItem>>,
    plan_view: &mut PlanView,
    busy: &str,
    // Characters of reply seen so far, for the work that produces one. Shared rather than
    // returned because it has to be readable while the future filling it is still running.
    streamed: Option<&AtomicUsize>,
    work: impl std::future::Future<Output = T>,
) -> Option<T> {
    let Some(keys) = keys.as_mut() else {
        // Piped input has nothing to type with, so the only way to stop is the signal
        // that raw mode would otherwise have swallowed.
        outln!("
○ {}", busy);
        return tokio::select! {
            output = work => Some(output),
            _ = tokio::signal::ctrl_c() => None,
        };
    };

    tokio::pin!(work);

    // The bar is redrawn on a timer as well as on keystrokes. Without a tick it only
    // changes when a key is pressed, which would leave the counter frozen through exactly
    // the long silence it exists to fill.
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(250));

    // What the bar last said, and whether anything else needs redrawing. A tick that would
    // draw the same label again is skipped: work with no counter to move — a tool running,
    // a call being judged — never changes it, and repainting the region four times a second
    // for the length of a long build is traffic and flicker bought for nothing.
    let mut shown: Option<String> = None;
    let mut shown_plan: Vec<TodoItem> = Vec::new();
    let mut dirty = true;

    loop {
        let label = crate::input::busy_label(busy, streamed);
        let plan = todos.lock().unwrap().clone();
        if dirty || shown.as_deref() != Some(label.as_str()) || shown_plan != plan {
            // Ctrl+T comes through `while_working` below, which sets `dirty` like any
            // other key, so the view does not need watching separately here.
            crate::input::draw(
                editor,
                &crate::input::Chrome {
                    mode: *mode,
                    busy: Some(&label),
                    // Something is running, so the bar has better than history to show.
                    last: None,
                    status,
                    todos: &plan,
                    queued,
                    plan: *plan_view,
                },
            );
            shown = Some(label);
            shown_plan = plan;
            dirty = false;
        }

        tokio::select! {
            // Nothing to do with the tick itself: the loop redraws at the top if the
            // counter has moved since it last did.
            _ = tick.tick() => {}
            output = &mut work => {
                out::clear_region();
                return Some(output);
            }
            event = keys.next() => {
                // Whatever it was — a character typed, a mode cycled, a resize — the box
                // below the bar has to be drawn again even when the label has not moved.
                dirty = true;
                let Some(event) = event else {
                    // The keyboard is gone. Finishing the work is still better than
                    // dropping it on the floor.
                    out::clear_region();
                    return Some(work.await);
                };
                match crate::input::while_working(editor, mode, plan_view, event) {
                    crate::input::WhileWorking::Interrupt => {
                        out::clear_region();
                        // What was waiting above the bar was never sent, and the user has
                        // just said they did not want this turn. It goes back into the box
                        // rather than on to the model as the next prompt.
                        if !queued.is_empty() {
                            let held: Vec<String> = queued.drain(..).collect();
                            editor.restore(&held.join("\n"));
                        }
                        return None;
                    }
                    // Not echoed into the transcript here: it has not been sent yet,
                    // and the row above the bar is where it waits until it has.
                    crate::input::WhileWorking::Queued(text) => queued.push(text),
                    crate::input::WhileWorking::Nothing => {}
                }
            }
        }
    }
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
                    let status = status_line(agent.model(), &agent.messages, agent.context_tokens);
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
                "\nA '-' means the field is left out of the request, so the agent.server's own \
                 setting for that model applies."
            );
            outln!("Change one for this session with /set <name> <value>.\n");
            continue;
        }

        // One command for every sampling knob rather than one command each. "/temp" is kept
        // as a shorthand for the one that gets touched most.
        if input == "/set" || input.starts_with("/set ") || input.starts_with("/temp") {
            let rest = if let Some(rest) = input.strip_prefix("/temp") {
                format!("temperature {}", rest.trim())
            } else {
                input.strip_prefix("/set").unwrap_or("").trim().to_string()
            };
            let mut parts = rest.split_whitespace();
            let name = parts.next().unwrap_or("").to_lowercase();
            let value = parts.next().unwrap_or("").to_string();

            if name.is_empty() {
                outln!("  temperature     {}", agent.settings.temperature.describe());
                outln!("  top_p           {}", agent.settings.top_p.describe());
                outln!("  top_k           {}", agent.settings.top_k.describe());
                outln!("  min_p           {}", agent.settings.min_p.describe());
                outln!("  repeat_penalty  {}", agent.settings.repeat_penalty.describe());
                outln!("  max_tokens      {}", agent.settings.max_tokens.describe());
                outln!("  reasoning_threshold  {} tokens", agent.reasoning_threshold);
                outln!("
Usage: /set <name> <value>, or /set <name> unset
");
                continue;
            }

            // Canonical wire names, plus the abbreviations people actually type.
            let canonical = match name.as_str() {
                "temp" | "temperature" => "temperature",
                "top_p" | "top-p" | "topp" => "top_p",
                "top_k" | "top-k" | "topk" => "top_k",
                "min_p" | "min-p" | "minp" => "min_p",
                "repeat_penalty" | "repeat-penalty" | "repeat" | "penalty" => "repeat_penalty",
                "max_tokens" | "max-tokens" | "max" => "max_tokens",
                "reasoning_threshold" | "reasoning-threshold" | "threshold" => {
                    "reasoning_threshold"
                }
                other => {
                    outln!(
                        "Unknown setting '{}'. Try: temperature, top_p, top_k, min_p, \
                         repeat_penalty, max_tokens, agent.reasoning_threshold.\n",
                        other
                    );
                    continue;
                }
            };

            // Handled here rather than through the machinery below, because this one is
            // haico's own and not the server's: it never goes into a request, so there is
            // nothing to apply to the two clients afterwards, and "unset" has to mean
            // falling back to the config file rather than leaving a field out for the
            // server to decide. Everything below this line is about the sampling knobs.
            if canonical == "reasoning_threshold" {
                if value.is_empty() {
                    outln!("  agent.reasoning_threshold  {} tokens
", agent.reasoning_threshold);
                } else if value == "unset" || value == "default" || value == "-" {
                    agent.reasoning_threshold = agent.settings.reasoning_threshold();
                    outln!(
                        "agent.reasoning_threshold back to {} tokens, as configured.
",
                        agent.reasoning_threshold
                    );
                } else if let Ok(parsed) = value.parse::<usize>() {
                    agent.reasoning_threshold = parsed;
                    outln!("agent.reasoning_threshold = {} tokens for this session.
", parsed);
                } else {
                    outln!(
                        "agent.reasoning_threshold needs a whole number of tokens, got '{}'.
",
                        value
                    );
                }
                continue;
            }

            if value.is_empty() {
                let shown = match canonical {
                    "temperature" => agent.settings.temperature.describe(),
                    "top_p" => agent.settings.top_p.describe(),
                    "top_k" => agent.settings.top_k.describe(),
                    "min_p" => agent.settings.min_p.describe(),
                    "repeat_penalty" => agent.settings.repeat_penalty.describe(),
                    _ => agent.settings.max_tokens.describe(),
                };
                outln!("  {}  {}
", canonical, shown);
                continue;
            }

            let clearing = value == "unset" || value == "default" || value == "-";
            const SET_HERE: &str = "set with /set";

            let applied = if clearing {
                match canonical {
                    "temperature" => agent.settings.temperature = Setting::unset(FROM_UNSET),
                    "top_p" => agent.settings.top_p = Setting::unset(FROM_UNSET),
                    "top_k" => agent.settings.top_k = Setting::unset(FROM_UNSET),
                    "min_p" => agent.settings.min_p = Setting::unset(FROM_UNSET),
                    "repeat_penalty" => agent.settings.repeat_penalty = Setting::unset(FROM_UNSET),
                    _ => agent.settings.max_tokens = Setting::unset(FROM_UNSET),
                }
                true
            } else {
                // top_k and max_tokens are counts; sending 40.0 where a server wants an
                // integer is a needless way to get a 400 back.
                match canonical {
                    "top_k" | "max_tokens" => match value.parse::<u64>() {
                        Ok(parsed) => {
                            let setting = Setting { value: Some(parsed), source: SET_HERE };
                            if canonical == "top_k" {
                                agent.settings.top_k = setting;
                            } else {
                                agent.settings.max_tokens = setting;
                            }
                            true
                        }
                        Err(_) => {
                            outln!("{} needs a whole number, got '{}'.
", canonical, value);
                            false
                        }
                    },
                    _ => match value.parse::<f64>() {
                        Ok(parsed) => {
                            let setting = Setting { value: Some(parsed), source: SET_HERE };
                            match canonical {
                                "temperature" => agent.settings.temperature = setting,
                                "top_p" => agent.settings.top_p = setting,
                                "min_p" => agent.settings.min_p = setting,
                                _ => agent.settings.repeat_penalty = setting,
                            }
                            true
                        }
                        Err(_) => {
                            outln!("{} needs a number, got '{}'.
", canonical, value);
                            false
                        }
                    },
                }
            };

            if applied {
                // Both clients, same as /model: web_fetch summarizes through the server's.
                agent.client.apply_settings(&agent.settings);
                agent.server.llm_client.apply_settings(&agent.settings);
                if clearing {
                    outln!("{} unset — the agent.server decides again.
", canonical);
                } else {
                    outln!("{} = {} for this session.
", canonical, value);
                }
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
                     trace over ~{} tokens (/set agent.reasoning_threshold).\n",
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

        run_turn(&mut agent, &mut io, input, initializing).await;
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
async fn run_turn(
    agent: &mut Agent,
    io: &mut TerminalIo,
    input: &str,
    // True when this turn is `/init` writing the project instruction file, so the
    // file it produces can be picked up at the end of it.
    initializing: bool,
) {
        // Wall clock, started here rather than at the first request: what the summary at
        // the end of the turn reports is how long the user waited, which includes the tool
        // runs, the permission prompts they answered and every round of the loop below —
        // not just the time the model spent generating.
        let turn_started = std::time::Instant::now();

        // Add user message
        agent.messages.push(json!({
            "role": "user",
            "content": input
        }));

        // Get LLM response and handle tool call chains. The cap stops a model that keeps
        // calling tools forever from spinning until the user kills the process. It does
        // not apply in auto mode: a fixed number of rounds is a poor stand-in for "this is
        // going nowhere", and a long job that hits it unattended has nobody to say
        // "continue". Repetition is the better signal, and it is what guards auto mode.
        const MAX_TOOL_ROUNDS: usize = 50;
        // A stuck model gives itself away by calling the same tool with the same arguments
        // over and over with nothing in between, which can happen on the tenth round or
        // the fiftieth. The first threshold says so in the conversation; the second gives
        // up on the turn.
        const LOOP_NUDGE: usize = 3;
        const LOOP_BREAK: usize = 5;
        let mut round = 0usize;

        // The (tool, arguments) pair called last, and how long the unbroken run of it is.
        // Consecutive rather than counted across the turn: a model working through a list
        // of compiler errors legitimately runs the same build command a dozen times in one
        // turn, and every one of those is progress because there were edits in between. A
        // loop is the same call with nothing else happening between the tries. Any
        // different call resets the run, `nudged_at` with it, so a later loop still gets
        // its warning.
        let mut last_call: Option<(String, String)> = None;
        let mut run = (0usize, String::new());
        let mut nudged_at = 0usize;

        // Cleared when compacting fails to get the history back under the line, which
        // means the window is too small to hold even a summary of it. Trying again the
        // next round would summarize the summary and throw the round away with it, so
        // the turn runs on instead and lets the server error be the honest end of it.
        let mut autocompact = true;

        loop {
            round += 1;
            // Re-read every round rather than captured once: approving a plan can switch
            // into auto mode mid-turn, and the cap should stop applying from that point.
            if agent.mode != Mode::Auto && round > MAX_TOOL_ROUNDS {
                outln!("\n[stopped after {} tool-call rounds without a final answer]", MAX_TOOL_ROUNDS);
                break;
            }

            // Auto mode is meant to run unattended, so the one thing it must not do at the
            // edge of the window is stop and ask for /compact: there is nobody there to
            // type it, and a job left running overnight would be found dead at the prompt.
            // Checked here rather than at the end of a turn, where the old hint printed,
            // because a long job lives inside this loop for hours and may never reach the
            // end of a turn at all. The history is balanced at this point — every tool call
            // from the round before already has its reply — so replacing it wholesale
            // cannot orphan a tool_call_id, which is not true further down.
            if agent.mode == Mode::Auto
                && autocompact
                && estimate_tokens(&agent.messages) > autocompact_threshold(agent.context_tokens)
            {
                let status = status_line(agent.model(), &agent.messages, agent.context_tokens);
                let compacted = work_with_input(
                    &mut io.keys,
                    &mut io.editor,
                    &mut agent.mode,
                    &mut io.queued,
                    &status,
                    &agent.server.todos,
                    &mut io.plan_view,
                    "Compacting context…",
                    None,
                    compact_history(&agent.client, &mut agent.messages),
                )
                .await;

                match compacted {
                    Some(Ok(report)) => {
                        // The same reclaim /compact does, for the same reason: the cache is
                        // keyed to a history that no longer exists.
                        agent.review_cache.clear();
                        outln!(
                            "{}{}{}",
                            SetForegroundColor(crate::input::BAR_BACKGROUND),
                            report,
                            ResetColor
                        );
                        // The compacted history ends on the assistant agreeing that it has
                        // the notes, and asking for a reply straight after that is asking
                        // the model to talk past itself. This says the one thing the notes
                        // cannot: the task was mid-flight when the summarizer ran, and it
                        // is the same task that continues.
                        agent.messages.push(json!({
                            "role": "user",
                            "content": "Your context was compacted mid-task. Continue the \
                                        work described in the notes above from where it left \
                                        off; do not start over."
                        }));

                        if estimate_tokens(&agent.messages) > autocompact_threshold(agent.context_tokens) {
                            autocompact = false;
                            outln!(
                                "[the summary is still over {}% of the window; carrying on without compacting again]",
                                (AUTOCOMPACT_FRACTION * 100.0) as usize
                            );
                        }
                    }
                    // Not fatal by itself: the request below may still fit, and if it does
                    // not, the server's own error ends the turn — exactly what would have
                    // happened with none of this here.
                    Some(Err(e)) => outln!("\n[error] could not compact: {:#}\n", e),
                    // Esc during the summarization is Esc during the turn, as everywhere
                    // else in this loop.
                    None => {
                        outln!("\n[interrupted]\n");
                        break;
                    }
                }
            }

            let thinking = get_thinking_message().to_string();

            // Assistant text is printed the moment it arrives, so none of the branches
            // below re-print it.
            let mut streamed_any = false;
            // The same for the chain of thought, which is shown dimmed as it lands. Tracked
            // separately because it is not markdown and does not go through the stream
            // above, and because the branches below have to know that something was already
            // put on screen even when no content ever followed.
            let mut streamed_thinking = false;
            // Characters of reply seen so far, thinking included, for the activity bar. A
            // big model can reason for many minutes before its first content token, and a
            // bar that only says "Thinking…" for all of it is indistinguishable from one
            // that has hung. An atomic rather than a Cell so the future stays Send.
            let streamed_chars = AtomicUsize::new(0);
            // Rendered as it arrives rather than at the end, so the answer still
            // appears a piece at a time and still comes out as markdown.
            let mut markdown = MarkdownStream::new();
            // Rebuilt every round rather than once per turn: approving a plan switches the
            // mode mid-turn, and the very next request has to carry the full tool list.
            let request_messages = with_mode_note(agent.mode, &agent.workspace, &agent.messages);
            let request_tools = tools_for_mode(agent.mode, &agent.tools);
            let status = status_line(agent.model(), &agent.messages, agent.context_tokens);

            // Esc abandons the turn instead of killing the process. It arrives here as a
            // keystroke rather than as a signal because raw mode is on for the whole
            // session — which is also what lets the next message be typed while this one
            // is still running.
            let result = work_with_input(
                &mut io.keys,
                &mut io.editor,
                &mut agent.mode,
                &mut io.queued,
                &status,
                &agent.server.todos,
                &mut io.plan_view,
                &thinking,
                Some(&streamed_chars),
                agent.client.chat_completion_streaming(
                    request_messages,
                    request_tools,
                    |chunk| {
                        match chunk {
                            // Dimmed, and painted per piece rather than left switched on
                            // across the stream: an unfinished line is redrawn above the
                            // input box every time the box moves, and a color left open at
                            // the end of it would bleed into the box itself.
                            Chunk::Reasoning(text) => {
                                streamed_chars.fetch_add(text.chars().count(), Ordering::Relaxed);
                                if !streamed_thinking {
                                    streamed_thinking = true;
                                    // Folded from the start rather than after the fact: the
                                    // reasoning can run for pages, and the activity bar is
                                    // already counting the characters as they arrive, so
                                    // nothing about the wait goes unreported while it is
                                    // closed. Ctrl+O opens it, and it stays open.
                                    crate::out::begin_fold(&format!(
                                        "{}{}○ (thinking){}",
                                        SetAttribute(Attribute::Italic),
                                        SetForegroundColor(Color::DarkGrey),
                                        ResetColor
                                    ));
                                    out!(
                                        "{}{}○ {}",
                                        SetAttribute(Attribute::Italic),
                                        SetForegroundColor(Color::DarkGrey),
                                        ResetColor
                                    );
                                }
                                // Italic as well as dim: a tool call and its result are
                                // dim too, and until this the three were the same grey.
                                // A terminal that cannot do italic is no worse off than
                                // before — conhost drops the attribute and prints the
                                // dim text it printed already.
                                out!(
                                    "{}{}{}{}",
                                    SetAttribute(Attribute::Italic),
                                    SetForegroundColor(Color::DarkGrey),
                                    text,
                                    ResetColor
                                );
                            }
                            Chunk::Content(text) => {
                                streamed_chars.fetch_add(text.chars().count(), Ordering::Relaxed);
                                if !streamed_any {
                                    streamed_any = true;
                                    // The answer starts on a row of its own, so it is not
                                    // mistaken for the tail of the thinking above it. The
                                    // reasoning is over the moment the answer begins, so
                                    // its fold closes here and the answer is not swept into
                                    // it.
                                    if streamed_thinking {
                                        crate::out::end_fold();
                                        crate::out::ensure_line_start();
                                        outln!();
                                    }
                                    out!("● ");
                                }
                                out!("{}", markdown.push(&text));
                            }
                        }
                        crate::out::flush();
                    },
                ),
            )
            .await;

            let Some(result) = result else {
                // The partial reply is dropped rather than stored: half a tool call in the
                // history is worse than no reply at all. Everything before this turn stays
                // consistent, so the conversation is still usable.
                crate::out::end_fold();
                if streamed_any {
                    out!("{}", markdown.finish(false));
                }
                if streamed_any || streamed_thinking {
                    crate::out::ensure_line_start();
                }
                outln!("\n[interrupted]\n");
                break;
            };

            let mut message = match result {
                Ok(message) => message,
                Err(e) => {
                    // A dropped connection, a server-side error, or an unparseable body
                    // must not end the session. Every tool call made so far already has its
                    // reply, so the history is consistent: drop back to the prompt.
                    crate::out::end_fold();
                    if streamed_any || streamed_thinking {
                        outln!();
                    }
                    outln!("\n[error] {:#}", e);
                    outln!("Conversation kept. Type a message to retry.");
                    break;
                }
            };

            // Both kinds of text were printed as they arrived — including the case of a
            // model whose entire reply is chain of thought with no content at all, which
            // is why nothing is re-printed from `message` here. All that is left is to
            // close whichever of the two was last on screen.
            // A reply that was nothing but reasoning ends with its fold still open.
            crate::out::end_fold();
            if streamed_any {
                out!("{}", markdown.finish(false));
            }
            if streamed_any || streamed_thinking {
                crate::out::ensure_line_start();
            }

            // The one place the reasoning policy is applied, deliberately ahead of every
            // `agent.messages.push(message.clone())` below rather than at each of them: all three
            // push this same value, and a policy enforced at two of the three sites would
            // leave whole traces in the history depending on how the turn happened to end.
            //
            // Only the compression of a long enough trace costs a request, and only that
            // goes through `work_with_input` — several seconds with a frozen input box
            // every turn would be worse than the tokens are worth. Esc during it means
            // `None`, which lands in the same place a failure does: the original chain of
            // thought is kept and the turn carries on. Dropping a field, and declining a
            // trace under the threshold, are both over before a frame could be painted, so
            // they are awaited directly rather than announced as work in progress.
            if agent.reasoning != Reasoning::Keep {
                let note = if reasoning_needs_a_request(&message, agent.reasoning, agent.reasoning_threshold) {
                    work_with_input(
                        &mut io.keys,
                        &mut io.editor,
                        &mut agent.mode,
                        &mut io.queued,
                        &status,
                        &agent.server.todos,
                        &mut io.plan_view,
                        "Compressing reasoning…",
                        None,
                        apply_reasoning(&agent.client, &mut message, agent.reasoning, agent.reasoning_threshold),
                    )
                    .await
                    .flatten()
                } else {
                    apply_reasoning(&agent.client, &mut message, agent.reasoning, agent.reasoning_threshold).await
                };

                match note {
                    // Shown rather than folded, and wearing the chrome's light blue: the
                    // whole point of compressing is to find out whether it is paying for
                    // itself, which cannot be judged from a line that is hidden by
                    // default. Tinted, not painted — the bars are the only thing on
                    // screen that gets a background, and that is what makes them read as
                    // furniture rather than as output.
                    Some(ReasoningOutcome::Compressed { before, after }) => {
                        agent.reasoning_stats.record(ReasoningOutcome::Compressed { before, after });
                        outln!(
                            "{}Compressed reasoning from {} to {} tokens.{}",
                            SetForegroundColor(crate::input::BAR_BACKGROUND),
                            before,
                            after,
                            ResetColor
                        );
                    }
                    // Dropping says nothing the policy did not already say: it removes
                    // every trace, every turn, so a line per reply would be a line per
                    // reply saying the same thing.
                    Some(ReasoningOutcome::Dropped { tokens }) => {
                        let line = format!("[reasoning dropped (~{} tokens)]", tokens);
                        crate::out::foldable(&line, &line);
                    }
                    None => {}
                }
            }

            // Check if model wants to call a tool
            if let Some(tool_calls) = message.get("tool_calls").filter(|c| !c.is_null()) {
                let tool_calls_array = match tool_calls.as_array() {
                    Some(arr) => arr,
                    None => {
                        outln!("[ERROR] tool_calls is not an array: {:?}", tool_calls);
                        break;
                    }
                };

                // Check if array is empty
                if tool_calls_array.is_empty() {
                    // No actual tool calls, this is the final response — already streamed.
                    agent.messages.push(message.clone());
                    break;
                }

                // Add assistant message with tool call first
                agent.messages.push(message.clone());

                // Set when Ctrl+C lands during a tool call, so the turn ends rather than
                // only the tool loop.
                let mut interrupted = false;

                for tool_call in tool_calls_array {
                    //outln!("[DEBUG] Processing tool call: {}", serde_json::to_string_pretty(&tool_call).unwrap_or_default());

                    // Captured up front: every path out of this iteration owes the model a
                    // `role:"tool"` message under this id. Leaving one unanswered makes the
                    // next request fail server-side, which is far harder to recover from
                    // than telling the model its call was malformed.
                    let tool_call_id = tool_call.get("id").cloned().unwrap_or(Value::Null);
                    let function = &tool_call["function"];

                    let tool_name = match function["name"].as_str() {
                        Some(name) => name,
                        None => {
                            let detail = format!(
                                "Malformed tool call: no function name in {}",
                                truncate_at_boundary(&function.to_string(), 500)
                            );
                            outln!("[ERROR] {}", detail);
                            agent.messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": detail
                            }));
                            continue;
                        }
                    };

                    let arguments = match parse_tool_arguments(function) {
                        Ok(arguments) => arguments,
                        Err(detail) => {
                            outln!("[ERROR] {}: {}", tool_name, detail);
                            agent.messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": format!(
                                    "Could not run '{}': {}. Call it again with valid JSON arguments.",
                                    tool_name, detail
                                )
                            }));
                            continue;
                        }
                    };

                    // Counted here, before any of the branches below, so a call that is
                    // refused or denied counts too — retrying a call the workspace or the
                    // user has already turned down, unchanged and immediately, is a loop
                    // like any other. What it returned is deliberately not part of the key:
                    // a build that fails three times in a row with nothing done between the
                    // tries is stuck whatever the errors say. serde_json sorts object keys,
                    // so the same call always renders to the same string.
                    let call = (tool_name.to_string(), arguments.to_string());
                    if last_call.as_ref() == Some(&call) {
                        run.0 += 1;
                    } else {
                        last_call = Some(call);
                        run = (1, tool_name.to_string());
                        nudged_at = 0;
                    }

                    // The model can put itself into plan mode, for when the user asked for
                    // a plan rather than for the work. Nothing to approve: this only ever
                    // takes capability away, and present_plan is where consent is needed.
                    if tool_name == START_PLAN {
                        let content = if agent.mode == Mode::Plan {
                            "Already in plan agent.mode. Investigate with the reading tools, then \
                             call present_plan."
                        } else {
                            agent.mode = Mode::Plan;
                            outln!("\n📋 Plan mode — investigating before changing anything.\n");
                            "Plan agent.mode is active. The tools that change this machine are no \
                             longer available to you. Investigate the real code with read, \
                             grep, ls and glob_search, then call present_plan with what you \
                             propose. Do not start work until the user approves it."
                        };
                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content
                        }));
                        continue;
                    }

                    // A finished plan is not a tool call so much as the end of plan mode,
                    // and the mode it ends lives out here rather than inside the tool.
                    if tool_name == PRESENT_PLAN {
                        let plan = arguments["plan"].as_str().unwrap_or("").trim().to_string();
                        if plan.is_empty() {
                            agent.messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": "The plan was empty. Put the plan itself in the \
                                            'plan' field and call present_plan again."
                            }));
                            continue;
                        }

                        // A terminal that cannot be read from must not be taken as
                        // approval — the same rule the permission prompt follows.
                        let content = match ask_about_plan(io.keys.as_mut(), &plan).await {
                            Ok(PlanDecision::Approve) => {
                                agent.mode = Mode::Normal;
                                outln!("\n✓ Plan approved — starting work.\n");
                                "The user approved the plan. Plan agent.mode is over and the full \
                                 tool set is available again. Begin work now, following the \
                                 plan you just had approved."
                                    .to_string()
                            }
                            Ok(PlanDecision::ApproveWithAuto) => {
                                agent.mode = Mode::Auto;
                                outln!("\n✓ Plan approved — starting work in auto mode.");
                                // Said plainly, because "auto" sounds like it will stop
                                // asking and right now it does not.
                                outln!(
                                    "  Each gated call is reviewed first; you still answer \
                                     the prompt.\n"
                                );
                                // The model is told nothing about the review. It should
                                // propose the same call either way, and knowing it is being
                                // checked is an invitation to write for the checker.
                                "The user approved the plan. Plan agent.mode is over and the full \
                                 tool set is available again. Begin work now, following the \
                                 plan you just had approved."
                                    .to_string()
                            }
                            Ok(PlanDecision::Revise(feedback)) => {
                                outln!("\n✗ Plan sent back for revision.\n");
                                format!(
                                    "The user did not approve the plan. Their corrections: {}\n\n\
                                     You are still in plan agent.mode and must not change anything. \
                                     Fold this in — investigating further if you need to — and \
                                     call present_plan again.",
                                    feedback
                                )
                            }
                            Err(e) => {
                                outln!("\n[error] could not ask about the plan: {}", e);
                                "The plan could not be shown for approval (no interactive \
                                 terminal), so it is not approved. Do not start work."
                                    .to_string()
                            }
                        };

                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content
                        }));
                        continue;
                    }

                    // A question for the user is answered out here for the same reason a
                    // plan is: the keyboard belongs to the REPL, and the tool body runs in
                    // the MCP server too, where there is nobody to ask.
                    if tool_name == ASK_USER {
                        let question = arguments["question"].as_str().unwrap_or_default();
                        // Anything that is not a string is dropped rather than rendered as
                        // JSON: a model that sent objects here is better told the list was
                        // too short than shown its own braces back on screen.
                        let options: Vec<String> = arguments["options"]
                            .as_array()
                            .map(|items| {
                                items
                                    .iter()
                                    .filter_map(|item| item.as_str().map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default();

                        let content = match ask::normalize(question, &options) {
                            Err(reason) => reason,
                            Ok((question, options)) => {
                                match ask_the_user(io.keys.as_mut(), &question, &options).await {
                                    Ok(answer) => ask::reply_for(&answer, &options),
                                    // Nobody to ask is not the same as an answer, and it
                                    // must not stall the turn either: the model is told to
                                    // carry the decision itself.
                                    Err(e) => {
                                        outln!("\n[error] could not ask the user: {}", e);
                                        "The question could not be put to the user (no \
                                         interactive terminal). Decide it yourself, say in \
                                         one line which way you went and why, and carry on."
                                            .to_string()
                                    }
                                }
                            }
                        };

                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content
                        }));
                        continue;
                    }

                    // Decide what this specific call needs: nothing (read-only), the user's
                    // consent under a narrowly scoped key, or an outright refusal. The mode
                    // has the first word — plan mode refuses anything that changes the
                    // machine, whatever the call would normally be entitled to.
                    let gate = gate_for_mode(agent.mode, &agent.workspace, tool_name, &arguments);

                    if let Gate::Deny(reason) = &gate {
                        outln!("\n⛔ {}\n", reason);
                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": reason
                        }));
                        continue;
                    }

                    let mut ask = match &gate {
                        Gate::Ask { keys, label } if !agent.permissions.is_allowed(keys) => {
                            Some((keys.clone(), label.clone()))
                        }
                        _ => None,
                    };

                    // Auto mode: the reviewer answers the routine yeses so the user does
                    // not have to. Only a call that was going to interrupt is reviewed —
                    // an already-approved one costs nothing and should stay free.
                    //
                    // An allow ends it here. Anything else — a refusal, an unreadable
                    // reply, a review the user interrupted — falls through to the prompt
                    // with the reason attached, so the reviewer can decline to act but
                    // never decides against the user on its own.
                    // Why the call is running unasked, and what to add to the prompt when
                    // it still has to be asked. Only one is ever set.
                    let mut auto_note: Option<String> = None;
                    let mut review_line: Option<String> = None;
                    if agent.mode == Mode::Auto && ask.is_some() {
                        // Transient: a review takes seconds, and a silent pause reads as a
                        // hang. Erased once it lands, because the outcome belongs on the
                        // call line below — left standing, every single call would cost two
                        // lines of log to say one thing.
                        let status = status_line(agent.model(), &agent.messages, agent.context_tokens);
                        let verdict = work_with_input(
                            &mut io.keys,
                            &mut io.editor,
                            &mut agent.mode,
                            &mut io.queued,
                            &status,
                            &agent.server.todos,
                            &mut io.plan_view,
                            "🤖 Reviewing tool call…",
                            // The reviewer's own reply is not streamed, so there is
                            // nothing to count.
                            None,
                            judge_tool_call(&agent.client, &mut agent.review_cache, tool_name, &arguments),
                        )
                        .await;

                        match verdict {
                            Some(Ok(Verdict { allow: true, reason, cached })) => {
                                // Marked, so reading back through a run still tells apart
                                // a call the reviewer weighed from one it merely
                                // recognized.
                                auto_note = Some(match cached {
                                    true => format!("cached — {}", reason),
                                    false => reason,
                                });
                                ask = None;
                            }
                            Some(Ok(Verdict { allow: false, reason, .. })) => {
                                review_line = Some(format!("🤖 Review: DENY — {}", reason));
                            }
                            Some(Err(e)) => {
                                review_line = Some(format!("🤖 Review failed: {:#}", e));
                            }
                            None => {
                                review_line = Some("🤖 Review interrupted.".to_string());
                            }
                        }
                    }

                    // Captured before the prompt consumes it: the diff below is skipped only
                    // when the prompt has already shown it.
                    let prompted = ask.is_some();

                    if let Some((gate_keys, allow_all_label)) = ask {
                        // Generate diff if needed
                        let diff_content: Option<String> = file_change_diff(tool_name, &arguments);

                        // Create human-readable description
                        let mut description = match tool_name {
                            "web_search" => {
                                let query = arguments["query"].as_str().unwrap_or("unknown");
                                format!("Search the web for: '{}'", query)
                            }
                            "web_fetch" | "web_get" => {
                                let url = arguments["url"].as_str().unwrap_or("unknown");
                                format!("Fetch content from URL: {}", url)
                            }
                            "write_file" => {
                                let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
                                let size = arguments["content"].as_str().map(|s| s.len()).unwrap_or(0);
                                format!("Write {} bytes to file: {}", size, file_path)
                            }
                            "edit" => {
                                let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
                                let from = arguments["old_string"].as_str().unwrap_or("unknown");
                                let to = arguments["new_string"].as_str().unwrap_or("unknown");
                                let replace_all = arguments["replace_all"].as_bool().unwrap_or(false);
                                if replace_all {
                                    format!("Edit file (replace all '{from}'->'{to}') in {}", file_path)
                                } else {
                                    format!("Edit file (replace once '{from}'->'{to}') in {}", file_path)
                                }
                            }
                            "multi_edit" => {
                                let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
                                let count = arguments["edits"].as_array().map(|e| e.len()).unwrap_or(0);
                                format!("Apply {} edit(s) to file: {}", count, file_path)
                            }
                            "run_background" => {
                                let command = arguments["command"].as_str().unwrap_or("unknown");
                                format!(
                                    "Start in the background:\n{}",
                                    indented_command(command)
                                )
                            }
                            "run_command" => {
                                let command = arguments["command"].as_str().unwrap_or("unknown");
                                format!("Execute command:\n{}", indented_command(command))
                            }
                            _ => format!("Execute '{}' tool", tool_name)
                        };

                        // The reviewer declined to wave this through. Its reason goes in the
                        // prompt rather than on its own line: shown twice it reads as two
                        // separate verdicts, and a refusal here is advice, not a decision.
                        if let Some(line) = &review_line {
                            description.push_str(&format!("\n\n{}", line));
                        }

                        // Ask for permission with diff if available. A terminal we cannot
                        // read from (piped stdin, no TTY) must not end the session, and must
                        // never be taken as consent: refuse the call and tell the model.
                        let response = match ask_about_call(
                            io.keys.as_mut(),
                            &allow_all_label,
                            &description,
                            diff_content.as_deref(),
                        )
                        .await
                        {
                            Ok(response) => response,
                            Err(e) => {
                                outln!("\n[error] could not ask for permission: {}", e);
                                agent.messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": format!(
                                        "Permission for '{}' could not be requested (no \
                                         interactive terminal), so it was not run.",
                                        tool_name
                                    )
                                }));
                                continue;
                            }
                        };

                        match response {
                            PermissionResponse::Allow => {
                                // Allow once, proceed with execution
                                outln!("✓ Permission granted for this request\n");
                            }
                            PermissionResponse::AllowAll => {
                                // Remembered under the narrow keys this call resolved to,
                                // not under the tool name.
                                agent.permissions.allow(&gate_keys);
                                outln!("✓ {}\n", allow_all_label);
                            }
                            PermissionResponse::Reject(alternative) => {
                                // User rejected, send alternative to LLM
                                outln!("✗ Permission denied. User suggestion: {}\n", alternative);

                                agent.messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": format!("Permission denied by user. User suggests: {}", alternative)
                                }));
                                continue;
                            }
                        }
                    }

                    // Execute the tool
                    // A slow tool (a big fetch, a long build) is interruptible too. The
                    // tool_call_id is still answered before breaking out — an unanswered one
                    // is what the next request gets rejected for.
                    outln!("{}", tool_call_line(tool_name, &arguments, auto_note.as_deref()));

                    // The diff belongs under the call line, not before it, and it has to
                    // appear on every path that skipped the prompt — a pre-approved write
                    // and now an auto-approved one both used to arrive as a path and a byte
                    // count, with no way to see what went into the file. Convenience should
                    // cost the prompt, not the oversight. Suppressed only when the prompt
                    // itself already showed the same diff a moment ago.
                    if !prompted {
                        if let Some(diff) = file_change_diff(tool_name, &arguments) {
                            if let Some(preview) = change_preview(&diff) {
                                // The preview is a few lines of the change; the whole diff
                                // waits behind it.
                                crate::out::foldable(preview.trim_end_matches('\n'), &diff);
                            }
                        }
                    }

                    let status = status_line(agent.model(), &agent.messages, agent.context_tokens);
                    let busy = format!("{}…", tool_name);

                    // Captured before the arguments move into the call. For a shell tool
                    // the command is the thing that ran, so naming the tool as well would
                    // only say "run_command" twice.
                    let did = match crate::render::tool_call_summary(tool_name, &arguments) {
                        summary if summary.is_empty() => format!("ran {}", tool_name),
                        summary if matches!(tool_name, "run_command" | "run_background") => {
                            format!("ran {}", summary)
                        }
                        summary => format!("ran {} {}", tool_name, summary),
                    };
                    let executed = work_with_input(
                        &mut io.keys,
                        &mut io.editor,
                        &mut agent.mode,
                        &mut io.queued,
                        &status,
                        &agent.server.todos,
                        &mut io.plan_view,
                        &busy,
                        // A tool produces its output all at once, so there is no partial
                        // size to report while it runs.
                        None,
                        agent.server.execute_tool(tool_name, arguments),
                    )
                    .await;

                    let Some(result) = executed else {
                        outln!("\n[interrupted]\n");
                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "Interrupted by the user before this tool finished."
                        }));
                        interrupted = true;
                        break;
                    };

                    // It ran, whatever it returned: a failed build is still the last
                    // thing that happened, and the bar saying so is the point.
                    io.last_activity = Some(did);

                    let tool_result = result.unwrap_or_else(|e| {
                        outln!("[ERROR] Tool execution failed: {:?}", e);
                        format!("Error: {}", e)
                    });

                    // The summary is the row that used to be all there was: the first
                    // line of the result, cut to fit. The result itself is kept behind it
                    // now rather than thrown away, and Ctrl+O opens it.
                    //
                    // research is the exception: it has already printed its progress and
                    // its whole answer while it ran, so a summary here would repeat the
                    // answer's first line immediately underneath the answer.
                    if tool_name != "research" {
                        if let Some(line) = tool_result_line(&tool_result) {
                            crate::out::foldable(&line, &tool_result);
                        }
                    }

                    // Add tool result
                    agent.messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result
                    }));
                }

                // The break above only leaves the tool loop; this leaves the turn.
                if interrupted {
                    break;
                }

                // The tool log makes a loop obvious to whoever is watching it scroll past,
                // but the model reads none of that — every round looks to it like a fresh
                // decision, which is why it can repeat one call fifty times. Saying the
                // repetition out loud in the conversation is what breaks it, and it breaks
                // it far earlier than the round cap. That matters most in auto mode, where
                // the cap is the only backstop and nobody is watching it climb.
                if run.0 >= LOOP_BREAK {
                    outln!(
                        "\n[stopped: '{}' was called {} times in a row with the same arguments]\n",
                        run.1, run.0
                    );
                    // Whatever was typed while this ran is deliberately left queued: it
                    // becomes the next prompt, which is where the user takes over.
                    break;
                }
                if run.0 >= LOOP_NUDGE && run.0 > nudged_at {
                    nudged_at = run.0;
                    outln!(
                        "\n[looping: '{}' called {} times in a row — telling the model to stop]\n",
                        run.1, run.0
                    );
                    // A user message rather than a system one: mid-conversation system
                    // messages are ignored by some local models and rejected outright by
                    // some servers, and this one has to land. The order is legal here for
                    // the same reason the queued text below is — every tool call in this
                    // round already has its result.
                    agent.messages.push(json!({
                        "role": "user",
                        "content": format!(
                            "You have called '{}' with the same arguments {} times in a \
                             row. Get out of the loop — calling it again will not tell you \
                             anything you have not already been told. Take a different \
                             approach, or stop and say what you are stuck on.",
                            run.1, run.0
                        )
                    }));
                }

                // Anything typed while this round ran joins the conversation here, where
                // every tool call already has its result. Any sooner would put a user
                // message between an assistant turn and the tool results it is waiting
                // for, which is a message order servers reject outright.
                for text in io.queued.drain(..) {
                    outln!("{}", echo(PROMPT, &text));
                    agent.messages.push(json!({"role": "user", "content": text}));
                }

                // Continue the loop to get next response (might be more tool calls or final answer)
                continue;
            } else {
                // No tool call, this is the final response — already streamed above.
                agent.messages.push(message.clone());
                break;
            }
        }

        // Every way out of the loop above lands here, an interrupt and a server error
        // included: the line says how long the turn ran for, not whether it got anywhere,
        // and a turn that ended badly after four minutes is exactly the one worth timing.
        // Dimmed for the same reason the chain of thought is — it is not the answer.
        outln!(
            "\n{}{}{}\n",
            SetForegroundColor(Color::DarkGrey),
            turn_summary(get_done_verb(), turn_started.elapsed(), chrono::Local::now()),
            ResetColor
        );

        // The file /init just wrote is the one that belongs in the system message, and what
        // was loaded at startup — usually nothing, since /init is what you run when the
        // project has no instructions yet — is now out of date. Re-read it here rather than
        // leaving it to the next startup, so the new rules are in force for the rest of this
        // session and survive /clear, which re-pushes this same content.
        if initializing {
            let reloaded = load_project_context();
            if reloaded != agent.project_context {
                agent.project_context = reloaded;
                if let Some((name, _)) = &agent.project_context {
                    outln!("\nLoaded project context from {}", name);
                }
                agent.system_prompt = compose_system_prompt(
                    agent.config.system_prompt.as_deref(),
                    &agent.client.model,
                    agent.settings.use_system_prompt,
                    agent.user_context.as_deref(),
                    agent.project_context.as_ref(),
                );
                set_system_message(&mut agent.messages, agent.system_prompt.as_deref());
            }
        }

        // Persisted every turn, so a crash does not cost the session's history.
        if let Some(path) = &io.history_path {
            crate::input::save_history(path, io.editor.history());
        }
        if let Some(path) = &agent.session_path {
            let todos = agent.plan();
            save_session(path, &agent.messages, &agent.client.model, &todos);
        }
}
