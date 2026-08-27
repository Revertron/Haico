use anyhow::{Context, Result};
use crossterm::style::{Color, ResetColor, SetForegroundColor};
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

mod ask;
mod config;
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
use workspace::*;


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
    outln!("  /set [name] [val]  Show or set temperature, top_p, top_k, repeat_penalty or");
    outln!("                     max_tokens ('/set <name> unset' to undo). '/temp' is short");
    outln!("                     for '/set temperature'.");
    outln!("  /bg [stop [id]]    List background jobs, or stop one or all of them");
    outln!("  /undo              Revert the last file change made by a tool");
    outln!("  /todo              Show the current plan, if the model made one");
    outln!("  /context           Show how much context the conversation is using");
    outln!("  /compact           Replace the history with a summary, freeing context");
    outln!("  /clear /new /reset Start over, keeping the system prompt");
    outln!("  /exit /quit        Leave (Ctrl+D also works)");
    outln!();
    outln!("At the prompt: up/down for history, Shift+Enter for another line,");
    outln!("               Shift+Tab to change mode, Ctrl+C to abandon the line.");
    outln!("While working: keep typing — Enter queues the message for when the turn");
    outln!("               ends. Esc interrupts and returns to the prompt.");
    outln!("Project rules:  HAICO.md, AGENTS.md or CLAUDE.md is appended to the system prompt.");
    outln!();
}

// Raw mode has to come off however the program leaves, including the error paths that
// return early. A terminal left in raw mode stops echoing what is typed into it, which
// looks exactly like a hung shell.
struct TerminalGuard(bool);

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.0 {
            out::clear_region();
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
    let mut dirty = true;

    loop {
        let label = crate::input::busy_label(busy, streamed);
        if dirty || shown.as_deref() != Some(label.as_str()) {
            crate::input::draw(
                editor,
                &crate::input::Chrome {
                    mode: *mode,
                    busy: Some(&label),
                    // Something is running, so the bar has better than history to show.
                    last: None,
                    status,
                },
            );
            shown = Some(label);
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
                match crate::input::while_working(editor, mode, event) {
                    crate::input::WhileWorking::Interrupt => {
                        out::clear_region();
                        return None;
                    }
                    crate::input::WhileWorking::Queued(text) => {
                        outln!("{}", crate::input::echo(PROMPT, &text));
                        queued.push(text);
                    }
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
    let mut keys = if interactive {
        crossterm::terminal::enable_raw_mode()?;
        out::set_raw(true);
        Some(Keys::start())
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
        // No model specified, fetch available models and let user select
        outln!("Fetching available models...");
        let models = match fetch_available_models(&config.llm_url, config.token.as_deref()).await {
            Ok(models) => models,
            Err(e) => {
                outln!("Error connecting to LLM: {e}");
                return Ok(());
            }
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
    };

    // Initialize LM Studio client with selected model
    let mut lm_client = LMStudioClient::new(
        config.llm_url.clone(),
        selected_model,
        config.token.clone(),
    );

    // Initialize MCP server with tools
    // Now that the model is known, its own config section can be applied. Held in a
    // variable because /model re-resolves it for whatever is switched to.
    let mut settings = resolve_settings(&config, &lm_client.model);
    lm_client.apply_settings(&settings);
    let mut context_tokens = settings.context_tokens;

    let mut server = MyMcpServer::new(lm_client.clone(), true, config.searxng.clone(), true);
    let openai_tools = mcp_tools_to_openai_format(&server);

    // outln!("Available tools:");
    // for tool in server.tool_router.list_all() {
    //     outln!("  - {}: {}", tool.name, tool.description.as_deref().unwrap_or(""));
    // }
    // outln!();

    // Read once and kept: /model rebuilds the system message, and re-reading the file on
    // every switch would announce it again each time.
    let mut project_context = load_project_context();
    if let Some((name, _)) = &project_context {
        outln!("Loaded project context from {}", name);
    }

    // None when this model is configured without our prompt and the project has no
    // instructions of its own — then the conversation carries no system message at all.
    let mut system_prompt_content = compose_system_prompt(
        config.system_prompt.as_deref(),
        &lm_client.model,
        settings.use_system_prompt,
        project_context.as_ref(),
    );

    // Conversation history
    let mut messages: Vec<Value> = vec![];
    if let Some(content) = &system_prompt_content {
        messages.push(json!({
            "role": "system",
            "content": content
        }));
    }

    // Restoring only the messages, never the stored system prompt: the date, branch and git
    // status in it would otherwise be whatever they were when the session was saved.
    let session_path = session_file_path();
    if config.resume {
        match session_path.as_deref().map(load_session) {
            Some(Ok(session)) => {
                let count = session.messages.len();
                messages.extend(session.messages);
                *server.todos.lock().unwrap() = session.todos;
                outln!(
                    "Resumed {} message(s) from {} (model then: {}).",
                    count, session.saved_at, session.model
                );
            }
            Some(Err(e)) => outln!("Could not resume: {:#}. Starting fresh.", e),
            None => outln!("Could not resume: no home directory. Starting fresh."),
        }
    } else if session_path.as_deref().is_some_and(|p| p.exists()) {
        outln!("A previous conversation exists here — start with --resume to continue it.");
    }

    // Permission manager. Deliberately not persisted: approvals last for one session, so
    // resuming never silently carries over permission to run something.
    let mut perm_manager = PermissionManager::new();
    let mut review_cache = ReviewCache::new();

    // Not persisted either, and for the same reason: resuming a session that happened to
    // end in auto mode should not quietly resume deciding things on the user's behalf.
    let mut mode = Mode::Normal;

    // Which paths the file tools may touch. The working directory to start with; `/add`
    // widens it for this session only, and `read_paths` in the config file names
    // directories that may be read but never written.
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut workspace = Workspace::new(&cwd, &config.file.read_paths);
    if !workspace.readable_roots().is_empty() {
        outln!(
            "Readable outside the project: {}",
            workspace
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
    let mut editor = Editor::new(history);

    // Typed while the model was working. Drained at the next round boundary, or at the
    // next prompt if the turn finished first.
    let mut queued: Vec<String> = Vec::new();

    // What the bar above the box says once a run is over: the last thing that actually
    // happened, rather than an empty row. None until the first tool has run.
    let mut last_activity: Option<String> = None;

    outln!("Type /help for commands.\n");

    // Main loop
    loop {
        // Read user input — unless there is already something waiting that was typed
        // while the last turn was running.
        let input = if !queued.is_empty() {
            queued.remove(0)
        } else {
            match keys.as_mut() {
                Some(keys) => {
                    let status = status_line(&lm_client.model, &messages, context_tokens);
                    match read_command(keys, &mut editor, &mut mode, &status, last_activity.as_deref()).await {
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
                        // read_command handles this one itself.
                        Input::CycleMode => continue,
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
            match workspace.add(std::path::Path::new(requested)) {
                Ok(added) => outln!("Added to the workspace: {}\n", display_path(&added)),
                Err(reason) => outln!("{}\n", reason),
            }
            continue;
        }

        if input == "/workspace" {
            outln!("Readable and writable:");
            for root in workspace.writable_roots() {
                outln!("  {}", display_path(root));
            }
            if workspace.readable_roots().is_empty() {
                outln!("Readable only: none configured (see 'read_paths' in the config file).");
            } else {
                outln!("Readable only:");
                for root in workspace.readable_roots() {
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
                mode = requested;
            }
            outln!("Mode: {}\n", mode.describe());
            if mode == Mode::Auto {
                // Spelled out, because this is the mode that acts without asking and the
                // user should know exactly how far that goes before leaving it on.
                outln!(
                    "{} reviews each call that would have asked you, with no conversation \
                     history. Calls it judges routine run on their own, tagged with the \
                     reason. Anything it refuses — or cannot judge — still comes to you.",
                    lm_client.model
                );
                outln!(
                    "It is the same model, so treat it as a check against drift and \
                     mistakes, not as a security boundary.\n"
                );
            }
            continue;
        }

        if input == "/clear" || input == "/new" || input == "/reset" {
            // The boundary where a stale copy of the project's instructions actually costs
            // something: whatever the model knew about the file from having read it is about
            // to be thrown out with the rest of the history, leaving the system message as
            // the only copy. One file read at a pause the user asked for, and it picks up an
            // edit made in an editor just as well as one a tool made.
            let reloaded = load_project_context();
            if reloaded != project_context {
                project_context = reloaded;
                system_prompt_content = compose_system_prompt(
                    config.system_prompt.as_deref(),
                    &lm_client.model,
                    settings.use_system_prompt,
                    project_context.as_ref(),
                );
                match &project_context {
                    Some((name, _)) => outln!("Reloaded project context from {}", name),
                    None => outln!("Project context file is gone; carrying on without it"),
                }
            }

            messages.clear();
            if let Some(content) = &system_prompt_content {
                messages.push(json!({
                    "role": "system",
                    "content": content
                }));
            }
            // To hand the memory back, not to withdraw anything: an entry is keyed by the
            // whole evidence string, so it holds a whole script or a whole diff, and a long
            // session accumulates them. No verdict is invalidated by this — the reviewer
            // never saw the conversation, so dropping the conversation tells it nothing.
            // Approvals in `perm_manager` deliberately survive, same as they always have.
            review_cache.clear();
            outln!("Conversation cleared.\n");
            continue;
        }

        if input == "/tools" {
            outln!("{} tools available:", server.tool_router.list_all().len());
            for tool in server.tool_router.list_all() {
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
            // Copied out so the borrow ends before lm_client is reassigned below.
            let llm_url = lm_client.base_url.clone();
            let token = lm_client.token.clone();

            let chosen = if requested.is_empty() {
                match fetch_available_models(&llm_url, token.as_deref()).await {
                    Ok(models) if !models.is_empty() => match pick_model(keys.as_mut(), models).await {
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
                // The new model's own config section applies from here on. Carrying the old
                // model's context budget over would be worse than having no config at all:
                // a 131072 budget against a 32k model warns far too late.
                settings = resolve_settings(&config, &model);
                context_tokens = settings.context_tokens;

                // Both copies: the REPL drives one, and web_fetch summarizes through the
                // server's. Leaving one behind would silently keep using the old model.
                lm_client.model = model.clone();
                lm_client.apply_settings(&settings);
                server.llm_client.model = model.clone();
                server.llm_client.apply_settings(&settings);

                // The reviewer is this model too, so its verdicts leave with it.
                review_cache.clear();

                // The system message belongs to the new model too: it names the model, and
                // the model's own config section decides whether ours is sent at all.
                // Replaced in place, so the conversation itself carries on unchanged.
                let rebuilt = compose_system_prompt(
                    config.system_prompt.as_deref(),
                    &model,
                    settings.use_system_prompt,
                    project_context.as_ref(),
                );
                if rebuilt != system_prompt_content {
                    system_prompt_content = rebuilt;
                    set_system_message(&mut messages, system_prompt_content.as_deref());
                }

                outln!("Now using model: {}", model);
                outln!(
                    "  context {} ({}){}\n",
                    context_tokens,
                    settings.context_source,
                    match settings.temperature.value {
                        Some(t) => format!(", temperature {}", t),
                        None => String::new(),
                    }
                );
            }
            continue;
        }

        if input == "/config" {
            match config.config_path.as_deref() {
                Some(path) if path.exists() => outln!("Config file: {}", path.display()),
                Some(path) => outln!("Config file: {} (not created yet)", path.display()),
                None => outln!("Config file: unavailable (no home directory)"),
            }
            outln!("
In force for '{}':", lm_client.model);
            outln!("  url             {}", lm_client.base_url);
            outln!(
                "  token           {}",
                if lm_client.token.is_some() { "set" } else { "not set" }
            );
            outln!(
                "  context         {}  [{}]",
                context_tokens, settings.context_source
            );
            outln!("  temperature     {}", settings.temperature.describe());
            outln!("  top_p           {}", settings.top_p.describe());
            outln!("  top_k           {}", settings.top_k.describe());
            outln!("  repeat_penalty  {}", settings.repeat_penalty.describe());
            outln!("  max_tokens      {}", settings.max_tokens.describe());
            outln!(
                "  system prompt   {}",
                match (settings.use_system_prompt, &system_prompt_content) {
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
            let rest = if let Some(rest) = input.strip_prefix("/temp") {
                format!("temperature {}", rest.trim())
            } else {
                input.strip_prefix("/set").unwrap_or("").trim().to_string()
            };
            let mut parts = rest.split_whitespace();
            let name = parts.next().unwrap_or("").to_lowercase();
            let value = parts.next().unwrap_or("").to_string();

            if name.is_empty() {
                outln!("  temperature     {}", settings.temperature.describe());
                outln!("  top_p           {}", settings.top_p.describe());
                outln!("  top_k           {}", settings.top_k.describe());
                outln!("  repeat_penalty  {}", settings.repeat_penalty.describe());
                outln!("  max_tokens      {}", settings.max_tokens.describe());
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
                "repeat_penalty" | "repeat-penalty" | "repeat" | "penalty" => "repeat_penalty",
                "max_tokens" | "max-tokens" | "max" => "max_tokens",
                other => {
                    outln!(
                        "Unknown setting '{}'. Try: temperature, top_p, top_k,                          repeat_penalty, max_tokens.
",
                        other
                    );
                    continue;
                }
            };

            if value.is_empty() {
                let shown = match canonical {
                    "temperature" => settings.temperature.describe(),
                    "top_p" => settings.top_p.describe(),
                    "top_k" => settings.top_k.describe(),
                    "repeat_penalty" => settings.repeat_penalty.describe(),
                    _ => settings.max_tokens.describe(),
                };
                outln!("  {}  {}
", canonical, shown);
                continue;
            }

            let clearing = value == "unset" || value == "default" || value == "-";
            const SET_HERE: &str = "set with /set";

            let applied = if clearing {
                match canonical {
                    "temperature" => settings.temperature = Setting::unset(FROM_UNSET),
                    "top_p" => settings.top_p = Setting::unset(FROM_UNSET),
                    "top_k" => settings.top_k = Setting::unset(FROM_UNSET),
                    "repeat_penalty" => settings.repeat_penalty = Setting::unset(FROM_UNSET),
                    _ => settings.max_tokens = Setting::unset(FROM_UNSET),
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
                                settings.top_k = setting;
                            } else {
                                settings.max_tokens = setting;
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
                                "temperature" => settings.temperature = setting,
                                "top_p" => settings.top_p = setting,
                                _ => settings.repeat_penalty = setting,
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
                lm_client.apply_settings(&settings);
                server.llm_client.apply_settings(&settings);
                if clearing {
                    outln!("{} unset — the server decides again.
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
                    let registry = server.background.lock().unwrap();
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
                let stopped = server.stop_all_background().await;
                outln!("Stopped {} job(s).\n", stopped);
            } else if let Some(id) = argument.strip_prefix("stop ").and_then(|n| n.trim().parse::<u32>().ok()) {
                let result = server
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
            match server.undo_last() {
                Ok(what) => outln!("Reverted: {}\n", what),
                Err(why) => outln!("{}\n", why),
            }
            continue;
        }

        if input == "/todo" {
            let todos = server.todos.lock().unwrap().clone();
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
            report_context(&messages, context_tokens);
            continue;
        }

        if input == "/compact" {
            match compact_history(&lm_client, &mut messages).await {
                // The same reclaim as /clear, on the one path that actually shrank the
                // session. A compaction that failed freed nothing to go with it.
                Ok(summary) => {
                    review_cache.clear();
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
            if mode == Mode::Plan {
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

        // Add user message
        messages.push(json!({
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

        loop {
            round += 1;
            // Re-read every round rather than captured once: approving a plan can switch
            // into auto mode mid-turn, and the cap should stop applying from that point.
            if mode != Mode::Auto && round > MAX_TOOL_ROUNDS {
                outln!("\n[stopped after {} tool-call rounds without a final answer]", MAX_TOOL_ROUNDS);
                break;
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
            let request_messages = with_mode_note(mode, &workspace, &messages);
            let request_tools = tools_for_mode(mode, &openai_tools);
            let status = status_line(&lm_client.model, &messages, context_tokens);

            // Esc abandons the turn instead of killing the process. It arrives here as a
            // keystroke rather than as a signal because raw mode is on for the whole
            // session — which is also what lets the next message be typed while this one
            // is still running.
            let result = work_with_input(
                &mut keys,
                &mut editor,
                &mut mode,
                &mut queued,
                &status,
                &thinking,
                Some(&streamed_chars),
                lm_client.chat_completion_streaming(
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
                                    out!(
                                        "{}○ {}",
                                        SetForegroundColor(Color::DarkGrey),
                                        ResetColor
                                    );
                                }
                                out!(
                                    "{}{}{}",
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
                                    // mistaken for the tail of the thinking above it.
                                    if streamed_thinking {
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
                if streamed_any {
                    out!("{}", markdown.finish(false));
                }
                if streamed_any || streamed_thinking {
                    crate::out::ensure_line_start();
                }
                outln!("\n[interrupted]\n");
                break;
            };

            let message = match result {
                Ok(message) => message,
                Err(e) => {
                    // A dropped connection, a server-side error, or an unparseable body
                    // must not end the session. Every tool call made so far already has its
                    // reply, so the history is consistent: drop back to the prompt.
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
            if streamed_any {
                out!("{}", markdown.finish(false));
            }
            if streamed_any || streamed_thinking {
                crate::out::ensure_line_start();
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
                    messages.push(message.clone());
                    break;
                }

                // Add assistant message with tool call first
                messages.push(message.clone());

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
                            messages.push(json!({
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
                            messages.push(json!({
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
                        let content = if mode == Mode::Plan {
                            "Already in plan mode. Investigate with the reading tools, then \
                             call present_plan."
                        } else {
                            mode = Mode::Plan;
                            outln!("\n📋 Plan mode — investigating before changing anything.\n");
                            "Plan mode is active. The tools that change this machine are no \
                             longer available to you. Investigate the real code with read, \
                             grep, ls and glob_search, then call present_plan with what you \
                             propose. Do not start work until the user approves it."
                        };
                        messages.push(json!({
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
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": "The plan was empty. Put the plan itself in the \
                                            'plan' field and call present_plan again."
                            }));
                            continue;
                        }

                        // A terminal that cannot be read from must not be taken as
                        // approval — the same rule the permission prompt follows.
                        let content = match ask_about_plan(keys.as_mut(), &plan).await {
                            Ok(PlanDecision::Approve) => {
                                mode = Mode::Normal;
                                outln!("\n✓ Plan approved — starting work.\n");
                                "The user approved the plan. Plan mode is over and the full \
                                 tool set is available again. Begin work now, following the \
                                 plan you just had approved."
                                    .to_string()
                            }
                            Ok(PlanDecision::ApproveWithAuto) => {
                                mode = Mode::Auto;
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
                                "The user approved the plan. Plan mode is over and the full \
                                 tool set is available again. Begin work now, following the \
                                 plan you just had approved."
                                    .to_string()
                            }
                            Ok(PlanDecision::Revise(feedback)) => {
                                outln!("\n✗ Plan sent back for revision.\n");
                                format!(
                                    "The user did not approve the plan. Their corrections: {}\n\n\
                                     You are still in plan mode and must not change anything. \
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

                        messages.push(json!({
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
                                match ask_the_user(keys.as_mut(), &question, &options).await {
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

                        messages.push(json!({
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
                    let gate = gate_for_mode(mode, &workspace, tool_name, &arguments);

                    if let Gate::Deny(reason) = &gate {
                        outln!("\n⛔ {}\n", reason);
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": reason
                        }));
                        continue;
                    }

                    let mut ask = match &gate {
                        Gate::Ask { keys, label } if !perm_manager.is_allowed(keys) => {
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
                    if mode == Mode::Auto && ask.is_some() {
                        // Transient: a review takes seconds, and a silent pause reads as a
                        // hang. Erased once it lands, because the outcome belongs on the
                        // call line below — left standing, every single call would cost two
                        // lines of log to say one thing.
                        let status = status_line(&lm_client.model, &messages, context_tokens);
                        let verdict = work_with_input(
                            &mut keys,
                            &mut editor,
                            &mut mode,
                            &mut queued,
                            &status,
                            "🤖 Reviewing this call…",
                            // The reviewer's own reply is not streamed, so there is
                            // nothing to count.
                            None,
                            judge_tool_call(&lm_client, &mut review_cache, tool_name, &arguments),
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
                            keys.as_mut(),
                            &allow_all_label,
                            &description,
                            diff_content.as_deref(),
                        )
                        .await
                        {
                            Ok(response) => response,
                            Err(e) => {
                                outln!("\n[error] could not ask for permission: {}", e);
                                messages.push(json!({
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
                                perm_manager.allow(&gate_keys);
                                outln!("✓ {}\n", allow_all_label);
                            }
                            PermissionResponse::Reject(alternative) => {
                                // User rejected, send alternative to LLM
                                outln!("✗ Permission denied. User suggestion: {}\n", alternative);

                                messages.push(json!({
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
                                out!("{}", preview);
                            }
                        }
                    }

                    let status = status_line(&lm_client.model, &messages, context_tokens);
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
                        &mut keys,
                        &mut editor,
                        &mut mode,
                        &mut queued,
                        &status,
                        &busy,
                        // A tool produces its output all at once, so there is no partial
                        // size to report while it runs.
                        None,
                        server.execute_tool(tool_name, arguments),
                    )
                    .await;

                    let Some(result) = executed else {
                        outln!("\n[interrupted]\n");
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "Interrupted by the user before this tool finished."
                        }));
                        interrupted = true;
                        break;
                    };

                    // It ran, whatever it returned: a failed build is still the last
                    // thing that happened, and the bar saying so is the point.
                    last_activity = Some(did);

                    let tool_result = result.unwrap_or_else(|e| {
                        outln!("[ERROR] Tool execution failed: {:?}", e);
                        format!("Error: {}", e)
                    });

                    if let Some(line) = tool_result_line(&tool_result) {
                        outln!("{}", line);
                    }

                    // Add tool result
                    messages.push(json!({
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
                    messages.push(json!({
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
                for text in queued.drain(..) {
                    messages.push(json!({"role": "user", "content": text}));
                }

                // Continue the loop to get next response (might be more tool calls or final answer)
                continue;
            } else {
                // No tool call, this is the final response — already streamed above.
                messages.push(message.clone());
                break;
            }
        }

        // The file /init just wrote is the one that belongs in the system message, and what
        // was loaded at startup — usually nothing, since /init is what you run when the
        // project has no instructions yet — is now out of date. Re-read it here rather than
        // leaving it to the next startup, so the new rules are in force for the rest of this
        // session and survive /clear, which re-pushes this same content.
        if initializing {
            let reloaded = load_project_context();
            if reloaded != project_context {
                project_context = reloaded;
                if let Some((name, _)) = &project_context {
                    outln!("\nLoaded project context from {}", name);
                }
                system_prompt_content = compose_system_prompt(
                    config.system_prompt.as_deref(),
                    &lm_client.model,
                    settings.use_system_prompt,
                    project_context.as_ref(),
                );
                set_system_message(&mut messages, system_prompt_content.as_deref());
            }
        }

        // Say something before the context runs out rather than after: a local model that
        // silently drops the front of the conversation just starts behaving oddly, with no
        // indication why. Compaction stays manual — dropping detail is the user's call.
        let used = estimate_tokens(&messages);
        if used > compact_hint_threshold(context_tokens) {
            outln!(
                "\n[context: ~{} of {} tokens. Run /compact to summarize and free space.]",
                used, context_tokens
            );
        }

        // Persisted every turn, so a crash does not cost the session's history.
        if let Some(path) = &history_path {
            crate::input::save_history(path, editor.history());
        }
        if let Some(path) = &session_path {
            let todos = server.todos.lock().unwrap().clone();
            save_session(path, &messages, &lm_client.model, &todos);
        }
    }

    if let Some(path) = &history_path {
        crate::input::save_history(path, editor.history());
    }

    // Every way out of the loop lands here, so a dev server started mid-session does not
    // outlive it holding a port. Silent when there is nothing to do.
    let stopped = server.stop_all_background().await;
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