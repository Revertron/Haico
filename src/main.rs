use anyhow::{Context, Result};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::transport::stdio;
use rmcp::ServiceExt;
use serde_json::{json, Value};
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;

mod config;
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

use config::*;
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
    println!("Commands:");
    println!("  /help              Show this list");
    println!("  /plan              Research and propose only; nothing changes until approved");
    println!("  /auto              Review each gated call; routine ones run without asking");
    println!("  /normal            Back to answering every prompt yourself");
    println!("  /add <path>        Let the file tools read and write another directory too");
    println!("  /workspace         Show which paths the file tools may touch");
    println!("  /mode              Show which of the three is in force");
    println!("  /tools             List available tools and whether they prompt");
    println!("  /model [name]      Switch model; with no name, pick from the server's list");
    println!("  /config            Show the settings in force and where each came from");
    println!("  /set [name] [val]  Show or set temperature, top_p, top_k, repeat_penalty or");
    println!("                     max_tokens ('/set <name> unset' to undo). '/temp' is short");
    println!("                     for '/set temperature'.");
    println!("  /bg [stop [id]]    List background jobs, or stop one or all of them");
    println!("  /undo              Revert the last file change made by a tool");
    println!("  /todo              Show the current plan, if the model made one");
    println!("  /context           Show how much context the conversation is using");
    println!("  /compact           Replace the history with a summary, freeing context");
    println!("  /clear /new /reset Start over, keeping the system prompt");
    println!("  /exit /quit        Leave (Ctrl+D also works)");
    println!();
    println!("At the prompt: up/down for history, Ctrl+C to abandon the current line.");
    println!("While working:  Ctrl+C interrupts the run and returns to the prompt.");
    println!("Project rules:  HAICO.md, AGENTS.md or CLAUDE.md is appended to the system prompt.");
    println!();
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

    println!("          Welcome to Haico!");
    println!("=====================================\n");
    if let Some(path) = config.config_path.as_deref().filter(|p| p.exists()) {
        println!("Config: {}", path.display());
    }
    println!("Connecting to LLM server at: {}", config.llm_url);
    if let Some(searxng) = &config.searxng {
        println!("Searching through SearxNG at: {}", searxng);
    }

    // Determine which model to use
    let selected_model = if let Some(model_name) = config.model.clone() {
        // Model specified via command-line, use it directly
        println!("Using specified model: {}\n", model_name);
        model_name
    } else {
        // No model specified, fetch available models and let user select
        println!("Fetching available models...");
        let models = match fetch_available_models(&config.llm_url, config.token.as_deref()).await {
            Ok(models) => models,
            Err(e) => {
                println!("Error connecting to LLM: {e}");
                return Ok(());
            }
        };

        if models.is_empty() {
            return Err(anyhow::anyhow!("No models available from server"));
        }

        // Let user select a model interactively
        select_model_interactive(models)?
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

    // println!("Available tools:");
    // for tool in server.tool_router.list_all() {
    //     println!("  - {}: {}", tool.name, tool.description.as_deref().unwrap_or(""));
    // }
    // println!();

    // Determine system prompt to use
    let mut system_prompt_content = if let Some(custom_prompt) = config.system_prompt.clone() {
        // Use custom prompt if provided
        custom_prompt
    } else {
        // Use default prompt with environment variables
        build_system_prompt(&lm_client.model)
    };

    // A project's own instructions are appended to whichever prompt is in use, so they
    // apply to a custom --system-prompt too.
    if let Some((name, project_context)) = load_project_context() {
        println!("Loaded project context from {}", name);
        system_prompt_content.push_str(&format!(
            "\n\n# Project instructions ({})\n\n\
             The following comes from this project and takes precedence over the general \
             guidance above wherever the two disagree.\n\n{}",
            name, project_context
        ));
    }

    // Conversation history
    let mut messages: Vec<Value> = vec![];
    messages.push(json!({
        "role": "system",
        "content": system_prompt_content
    }));

    // Restoring only the messages, never the stored system prompt: the date, branch and git
    // status in it would otherwise be whatever they were when the session was saved.
    let session_path = session_file_path();
    if config.resume {
        match session_path.as_deref().map(load_session) {
            Some(Ok(session)) => {
                let count = session.messages.len();
                messages.extend(session.messages);
                *server.todos.lock().unwrap() = session.todos;
                println!(
                    "Resumed {} message(s) from {} (model then: {}).",
                    count, session.saved_at, session.model
                );
            }
            Some(Err(e)) => println!("Could not resume: {:#}. Starting fresh.", e),
            None => println!("Could not resume: no home directory. Starting fresh."),
        }
    } else if session_path.as_deref().is_some_and(|p| p.exists()) {
        println!("A previous conversation exists here — start with --resume to continue it.");
    }

    // Permission manager. Deliberately not persisted: approvals last for one session, so
    // resuming never silently carries over permission to run something.
    let mut perm_manager = PermissionManager::new();

    // Not persisted either, and for the same reason: resuming a session that happened to
    // end in auto mode should not quietly resume deciding things on the user's behalf.
    let mut mode = Mode::Normal;

    // Which paths the file tools may touch. The working directory to start with; `/add`
    // widens it for this session only, and `read_paths` in the config file names
    // directories that may be read but never written.
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut workspace = Workspace::new(&cwd, &config.file.read_paths);
    if !workspace.readable_roots().is_empty() {
        println!(
            "Readable outside the project: {}",
            workspace
                .readable_roots()
                .iter()
                .map(|r| display_path(r))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Line editing, history and bracketed paste. Plain read_line gave no arrow-key
    // history, no editing beyond backspace, and turned a pasted multi-line snippet into one
    // user turn per line.
    let rl_config = rustyline::Config::builder()
        .auto_add_history(true)
        .bracketed_paste(true)
        .build();
    let mut editor: rustyline::Editor<(), rustyline::history::DefaultHistory> =
        rustyline::Editor::with_config(rl_config)?;
    let history_path = history_file_path();
    if let Some(path) = &history_path {
        let _ = editor.load_history(path);
    }

    println!("Type /help for commands.\n");

    // Main loop
    loop {
        // Read user input
        let input = match editor.readline(mode.prompt()) {
            Ok(line) => line,
            // Ctrl+C at the prompt abandons the half-typed line, it does not quit.
            Err(rustyline::error::ReadlineError::Interrupted) => continue,
            // Ctrl+D, or stdin closing when input is piped in.
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(e) => {
                eprintln!("[error] could not read input: {}", e);
                break;
            }
        };
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "/exit" || input == "/quit" {
            println!("Goodbye!");
            break;
        }

        if input == "/help" {
            print_repl_help();
            continue;
        }

        if input == "/add" || input.starts_with("/add ") {
            let requested = input.strip_prefix("/add").unwrap_or("").trim();
            if requested.is_empty() {
                println!("Usage: /add <path>   (a directory to read and write for this session)\n");
                continue;
            }
            match workspace.add(std::path::Path::new(requested)) {
                Ok(added) => println!("Added to the workspace: {}\n", display_path(&added)),
                Err(reason) => println!("{}\n", reason),
            }
            continue;
        }

        if input == "/workspace" {
            println!("Readable and writable:");
            for root in workspace.writable_roots() {
                println!("  {}", display_path(root));
            }
            if workspace.readable_roots().is_empty() {
                println!("Readable only: none configured (see 'read_paths' in the config file).");
            } else {
                println!("Readable only:");
                for root in workspace.readable_roots() {
                    println!("  {}", display_path(root));
                }
            }
            println!("\nAnything else is refused outright, for reads as well as writes.\n");
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
            println!("Mode: {}\n", mode.describe());
            if mode == Mode::Auto {
                // Spelled out, because this is the mode that acts without asking and the
                // user should know exactly how far that goes before leaving it on.
                println!(
                    "{} reviews each call that would have asked you, with no conversation \
                     history. Calls it judges routine run on their own, tagged with the \
                     reason. Anything it refuses — or cannot judge — still comes to you.",
                    lm_client.model
                );
                println!(
                    "It is the same model, so treat it as a check against drift and \
                     mistakes, not as a security boundary.\n"
                );
            }
            continue;
        }

        if input == "/clear" || input == "/new" || input == "/reset" {
            messages.clear();
            messages.push(json!({
                "role": "system",
                "content": &system_prompt_content
            }));
            println!("Conversation cleared.\n");
            continue;
        }

        if input == "/tools" {
            println!("{} tools available:", server.tool_router.list_all().len());
            for tool in server.tool_router.list_all() {
                let gate = if AUTO_ALLOWED_TOOLS.contains(&tool.name.as_ref()) {
                    "auto"
                } else {
                    "asks"
                };
                println!("  {:<14} [{}]", tool.name, gate);
            }
            println!();
            continue;
        }

        if input == "/model" || input.starts_with("/model ") {
            let requested = input.strip_prefix("/model").unwrap_or("").trim().to_string();
            // Copied out so the borrow ends before lm_client is reassigned below.
            let llm_url = lm_client.base_url.clone();
            let token = lm_client.token.clone();

            let chosen = if requested.is_empty() {
                match fetch_available_models(&llm_url, token.as_deref()).await {
                    Ok(models) if !models.is_empty() => match select_model_interactive(models) {
                        Ok(model) => Some(model),
                        Err(e) => {
                            println!("Model selection cancelled: {}\n", e);
                            None
                        }
                    },
                    Ok(_) => {
                        println!("The server reports no available models.\n");
                        None
                    }
                    Err(e) => {
                        println!("Could not list models: {:#}\n", e);
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

                println!("Now using model: {}", model);
                println!(
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
                Some(path) if path.exists() => println!("Config file: {}", path.display()),
                Some(path) => println!("Config file: {} (not created yet)", path.display()),
                None => println!("Config file: unavailable (no home directory)"),
            }
            println!("
In force for '{}':", lm_client.model);
            println!("  url             {}", lm_client.base_url);
            println!(
                "  token           {}",
                if lm_client.token.is_some() { "set" } else { "not set" }
            );
            println!(
                "  context         {}  [{}]",
                context_tokens, settings.context_source
            );
            println!("  temperature     {}", settings.temperature.describe());
            println!("  top_p           {}", settings.top_p.describe());
            println!("  top_k           {}", settings.top_k.describe());
            println!("  repeat_penalty  {}", settings.repeat_penalty.describe());
            println!("  max_tokens      {}", settings.max_tokens.describe());
            println!(
                "\nA '-' means the field is left out of the request, so the server's own \
                 setting for that model applies."
            );
            println!("Change one for this session with /set <name> <value>.\n");
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
                println!("  temperature     {}", settings.temperature.describe());
                println!("  top_p           {}", settings.top_p.describe());
                println!("  top_k           {}", settings.top_k.describe());
                println!("  repeat_penalty  {}", settings.repeat_penalty.describe());
                println!("  max_tokens      {}", settings.max_tokens.describe());
                println!("
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
                    println!(
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
                println!("  {}  {}
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
                            println!("{} needs a whole number, got '{}'.
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
                            println!("{} needs a number, got '{}'.
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
                    println!("{} unset — the server decides again.
", canonical);
                } else {
                    println!("{} = {} for this session.
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
                    println!("No background jobs.\n");
                } else {
                    println!("Background jobs:\n{}\n", listing.join("\n"));
                }
            } else if argument == "stop" {
                let stopped = server.stop_all_background().await;
                println!("Stopped {} job(s).\n", stopped);
            } else if let Some(id) = argument.strip_prefix("stop ").and_then(|n| n.trim().parse::<u32>().ok()) {
                let result = server
                    .background_stop(Parameters(BackgroundStopParams { id: Some(id) }))
                    .await;
                match result {
                    Ok(result) => println!(
                        "{}\n",
                        result
                            .content
                            .iter()
                            .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n")
                    ),
                    Err(e) => println!("Could not stop job {}: {:?}\n", id, e),
                }
            } else {
                println!("Usage: /bg, /bg stop <id>, /bg stop\n");
            }
            continue;
        }

        if input == "/undo" {
            match server.undo_last() {
                Ok(what) => println!("Reverted: {}\n", what),
                Err(why) => println!("{}\n", why),
            }
            continue;
        }

        if input == "/todo" {
            let todos = server.todos.lock().unwrap().clone();
            if todos.is_empty() {
                println!("No plan yet.\n");
            } else {
                println!("Plan:");
                for item in &todos {
                    let mark = match item.status.as_str() {
                        "done" => "x",
                        "in_progress" => "»",
                        _ => " ",
                    };
                    println!("  [{}] {}", mark, item.task);
                }
                println!();
            }
            continue;
        }

        if input == "/context" {
            report_context(&messages, context_tokens);
            continue;
        }

        if input == "/compact" {
            match compact_history(&lm_client, &mut messages).await {
                Ok(summary) => println!("{}\n", summary),
                Err(e) => eprintln!("[error] could not compact: {:#}\n", e),
            }
            continue;
        }

        // Add user message
        messages.push(json!({
            "role": "user",
            "content": input
        }));

        // Get LLM response and handle tool call chains. The cap stops a model that keeps
        // calling tools forever from spinning until the user kills the process.
        const MAX_TOOL_ROUNDS: usize = 50;
        let mut round = 0usize;

        loop {
            round += 1;
            if round > MAX_TOOL_ROUNDS {
                println!("\n[stopped after {} tool-call rounds without a final answer]", MAX_TOOL_ROUNDS);
                break;
            }

            println!("\n○ {}", get_thinking_message());

            // Assistant text is printed the moment it arrives, so none of the branches
            // below re-print it.
            let mut streamed_any = false;
            // Ctrl+C abandons the turn instead of killing the process. rustyline puts the
            // terminal in raw mode at the prompt, so Ctrl+C there is read as a keypress and
            // never reaches this handler — it only fires while a request is in flight,
            // which is exactly when a model heading down the wrong path needs stopping.
            // Rebuilt every round rather than once per turn: approving a plan switches the
            // mode mid-turn, and the very next request has to carry the full tool list.
            let request_messages = with_mode_note(mode, &workspace, &messages);
            let request_tools = tools_for_mode(mode, &openai_tools);

            let result = tokio::select! {
                result = lm_client.chat_completion_streaming(
                    request_messages,
                    request_tools,
                    |text| {
                        if !streamed_any {
                            streamed_any = true;
                            print!("● ");
                        }
                        print!("{}", text);
                        let _ = io::stdout().flush();
                    },
                ) => Some(result),
                _ = tokio::signal::ctrl_c() => None,
            };

            let Some(result) = result else {
                // The partial reply is dropped rather than stored: half a tool call in the
                // history is worse than no reply at all. Everything before this turn stays
                // consistent, so the conversation is still usable.
                if streamed_any {
                    println!();
                }
                println!("\n[interrupted]\n");
                break;
            };

            let message = match result {
                Ok(message) => message,
                Err(e) => {
                    // A dropped connection, a server-side error, or an unparseable body
                    // must not end the session. Every tool call made so far already has its
                    // reply, so the history is consistent: drop back to the prompt.
                    if streamed_any {
                        println!();
                    }
                    eprintln!("\n[error] {:#}", e);
                    println!("Conversation kept. Type a message to retry.");
                    break;
                }
            };

            if streamed_any {
                println!();
            } else if let Some(reasoning) = message
                .get("reasoning")
                .and_then(|r| r.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                // Some models put their whole preamble in `reasoning` and send no content.
                print!("● ");
                let _ = render_text(reasoning);
                println!();
            }

            // Check if model wants to call a tool
            if let Some(tool_calls) = message.get("tool_calls").filter(|c| !c.is_null()) {
                let tool_calls_array = match tool_calls.as_array() {
                    Some(arr) => arr,
                    None => {
                        eprintln!("[ERROR] tool_calls is not an array: {:?}", tool_calls);
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
                    //println!("[DEBUG] Processing tool call: {}", serde_json::to_string_pretty(&tool_call).unwrap_or_default());

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
                            eprintln!("[ERROR] {}", detail);
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
                            eprintln!("[ERROR] {}: {}", tool_name, detail);
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

                    // The model can put itself into plan mode, for when the user asked for
                    // a plan rather than for the work. Nothing to approve: this only ever
                    // takes capability away, and present_plan is where consent is needed.
                    if tool_name == START_PLAN {
                        let content = if mode == Mode::Plan {
                            "Already in plan mode. Investigate with the reading tools, then \
                             call present_plan."
                        } else {
                            mode = Mode::Plan;
                            println!("\n📋 Plan mode — investigating before changing anything.\n");
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
                        let content = match review_plan(&plan) {
                            Ok(PlanDecision::Approve) => {
                                mode = Mode::Normal;
                                println!("\n✓ Plan approved — starting work.\n");
                                "The user approved the plan. Plan mode is over and the full \
                                 tool set is available again. Begin work now, following the \
                                 plan you just had approved."
                                    .to_string()
                            }
                            Ok(PlanDecision::ApproveWithAuto) => {
                                mode = Mode::Auto;
                                println!("\n✓ Plan approved — starting work in auto mode.");
                                // Said plainly, because "auto" sounds like it will stop
                                // asking and right now it does not.
                                println!(
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
                                println!("\n✗ Plan sent back for revision.\n");
                                format!(
                                    "The user did not approve the plan. Their corrections: {}\n\n\
                                     You are still in plan mode and must not change anything. \
                                     Fold this in — investigating further if you need to — and \
                                     call present_plan again.",
                                    feedback
                                )
                            }
                            Err(e) => {
                                eprintln!("\n[error] could not ask about the plan: {}", e);
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

                    // Decide what this specific call needs: nothing (read-only), the user's
                    // consent under a narrowly scoped key, or an outright refusal. The mode
                    // has the first word — plan mode refuses anything that changes the
                    // machine, whatever the call would normally be entitled to.
                    let gate = gate_for_mode(mode, &workspace, tool_name, &arguments);

                    if let Gate::Deny(reason) = &gate {
                        println!("\n⛔ {}\n", reason);
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
                        print!("  🤖 Reviewing this call…");
                        let _ = io::stdout().flush();
                        let verdict = tokio::select! {
                            verdict = judge_tool_call(&lm_client, tool_name, &arguments) => Some(verdict),
                            _ = tokio::signal::ctrl_c() => None,
                        };
                        print!("\r");
                        let _ = crossterm::execute!(
                            io::stdout(),
                            crossterm::terminal::Clear(crossterm::terminal::ClearType::CurrentLine)
                        );

                        match verdict {
                            Some(Ok(Verdict { allow: true, reason })) => {
                                auto_note = Some(reason);
                                ask = None;
                            }
                            Some(Ok(Verdict { allow: false, reason })) => {
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
                        let response = match ask_permission(
                            &allow_all_label,
                            &description,
                            diff_content.as_deref(),
                        ) {
                            Ok(response) => response,
                            Err(e) => {
                                eprintln!("\n[error] could not ask for permission: {}", e);
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
                                println!("✓ Permission granted for this request\n");
                            }
                            PermissionResponse::AllowAll => {
                                // Remembered under the narrow keys this call resolved to,
                                // not under the tool name.
                                perm_manager.allow(&gate_keys);
                                println!("✓ {}\n", allow_all_label);
                            }
                            PermissionResponse::Reject(alternative) => {
                                // User rejected, send alternative to LLM
                                println!("✗ Permission denied. User suggestion: {}\n", alternative);

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
                    log_tool_call(tool_name, &arguments, auto_note.as_deref());

                    // The diff belongs under the call line, not before it, and it has to
                    // appear on every path that skipped the prompt — a pre-approved write
                    // and now an auto-approved one both used to arrive as a path and a byte
                    // count, with no way to see what went into the file. Convenience should
                    // cost the prompt, not the oversight. Suppressed only when the prompt
                    // itself already showed the same diff a moment ago.
                    if !prompted {
                        if let Some(diff) = file_change_diff(tool_name, &arguments) {
                            print_change_preview(&diff);
                        }
                    }

                    let executed = tokio::select! {
                        result = server.execute_tool(tool_name, arguments) => Some(result),
                        _ = tokio::signal::ctrl_c() => None,
                    };

                    let Some(result) = executed else {
                        println!("\n[interrupted]\n");
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "Interrupted by the user before this tool finished."
                        }));
                        interrupted = true;
                        break;
                    };

                    let tool_result = result.unwrap_or_else(|e| {
                        eprintln!("[ERROR] Tool execution failed: {:?}", e);
                        format!("Error: {}", e)
                    });

                    log_tool_result(&tool_result);

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

                // Continue the loop to get next response (might be more tool calls or final answer)
                continue;
            } else {
                // No tool call, this is the final response — already streamed above.
                messages.push(message.clone());
                break;
            }
        }

        // Say something before the context runs out rather than after: a local model that
        // silently drops the front of the conversation just starts behaving oddly, with no
        // indication why. Compaction stays manual — dropping detail is the user's call.
        let used = estimate_tokens(&messages);
        if used > compact_hint_threshold(context_tokens) {
            println!(
                "\n[context: ~{} of {} tokens. Run /compact to summarize and free space.]",
                used, context_tokens
            );
        }

        // Persisted every turn, so a crash does not cost the session's history.
        if let Some(path) = &history_path {
            let _ = editor.save_history(path);
        }
        if let Some(path) = &session_path {
            let todos = server.todos.lock().unwrap().clone();
            save_session(path, &messages, &lm_client.model, &todos);
        }
    }

    if let Some(path) = &history_path {
        let _ = editor.save_history(path);
    }

    // Every way out of the loop lands here, so a dev server started mid-session does not
    // outlive it holding a port. Silent when there is nothing to do.
    let stopped = server.stop_all_background().await;
    if stopped > 0 {
        println!("Stopped {} background job(s).", stopped);
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