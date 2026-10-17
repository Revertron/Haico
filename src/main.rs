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
mod commands;
mod config;
mod debug;
#[cfg(feature = "gui")]
mod gui;
mod headless;
mod input;
mod library;
mod llm;
#[cfg(feature = "gui")]
mod markdown;
mod memory;
mod mention;
mod modes;
mod permissions;
mod present;
mod prompt;
mod readable;
mod render;
mod repetition;
mod scratch;
mod session;
mod skills;
mod store;
mod subagent;
#[cfg(test)]
mod testutil;
mod timers;
mod tools;
mod trace;
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
    outln!("  /cd [path]         Move to another project; with no path, show where you are");
    outln!("  /mode              Show which of the three is in force");
    outln!("  /init              Study the project and write its instruction file");
    outln!("  /tools             List available tools and whether they prompt");
    outln!("  /model [name]      Switch model and save it as the default; with no name,");
    outln!("                     pick from the server's list");
    outln!("  /provider [name]   Switch to another server from the config's providers and");
    outln!("                     save it as the default; with no name, pick from the list");
    outln!("  /config            Show the settings in force and where each came from");
    outln!("  /set [name] [val]  Show or set context, temperature, top_p, top_k, min_p,");
    outln!("                     repeat_penalty, max_tokens, reasoning_effort or");
    outln!("                     reasoning_threshold ('/set <name> unset' to undo).");
    outln!("                     '/temp' and '/effort' are short for '/set temperature'");
    outln!("                     and '/set reasoning_effort'; '/effort on|off' switch");
    outln!("                     thinking itself. All last for the session except");
    outln!("                     'context', which is saved for this model.");
    outln!("  /bg [stop [id]]    List background jobs, or stop one or all of them");
    outln!("  /timers [cancel]   List the timers the model set, or cancel one (by id) or all");
    outln!("  /undo              Revert the last file change made by a tool");
    outln!("  /todo              Show the current plan, if the model made one");
    outln!("  /context           Show how much context the conversation is using");
    outln!("  /reasoning [what]  Keep, compress or drop the model's chain of thought");
    outln!("  /compact           Replace the history with a summary, freeing context");
    outln!("  /resume            Load this directory's saved conversation, dropping the current one");
    outln!("  /clear /new /reset Start over, keeping the system prompt");
    outln!("  /exit /quit        Leave (Ctrl+D also works)");
    outln!("  /skills            List the skills the model can load, and where each lives");
    outln!("  /update-skills     Re-read the skills directories into the system prompt");
    outln!("  /<skill> [args]    Run a skill by name; the model is shown the list and loads");
    outln!("                     one on its own when a request matches it");
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
async fn drive_turn(agent: &mut Agent, io: &mut TerminalIo, input: &str, initializing: bool) {
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

            let label = busy_label(&busy, if counting { streamed } else { 0 });
            let plan = todos.lock().unwrap().clone();
            if dirty || shown.as_deref() != Some(label.as_str()) || shown_plan != plan {
                let queued = link.queued();
                draw(
                    &mut io.editor,
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
    // stderr, like every diagnostic here: stdout carries the protocol.
    if let Some(notice) = starter_config_notice(&config) {
        eprintln!("[haico] {}", notice);
    }
    if let Some(notice) = config_error_notice(&config) {
        eprintln!("[haico] {}", notice);
    }

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

// Which model the session starts with: the one configured, else the only one loaded, else
// the user's pick. With no keyboard (`keys` is `None` — piped input, or a `--prompt` run)
// the last step is an error naming the fix rather than a guess.
async fn choose_model(config: &Config, keys: Option<&mut Keys>) -> Result<String> {
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

#[tokio::main]
async fn main() -> Result<()> {
    // Parse configuration first so we know the mode before any stdout output.
    let config = parse_config()?;
    // Before anything is rendered: NO_COLOR is haico's to apply, not crossterm's.
    out::own_colors();

    // The process moves, rather than the directory being handed to whatever needs it:
    // the project's instructions, memory, skills, the workspace, the session path, the
    // trace's `cwd` and every shell command's starting point each ask for the current
    // directory on their own, and a path passed down would have to reach all of them —
    // and the next one added. A directory that is not there stops the run rather than
    // leaving it to work wherever it was started, which for `-p "... and commit"` is
    // the wrong repository.
    if let Some(dir) = &config.dir {
        env::set_current_dir(dir)
            .with_context(|| format!("cannot work in {}", dir.display()))?;
    }

    // Before anything talks to the model, so the `start` line opens the run, and for
    // every entry path — the MCP server's `web_fetch` summarizer is a model call too. A
    // trace that cannot be opened stops the run: asked to record it and not recording it
    // would be found out only after the run was gone.
    if let Some(path) = &config.trace {
        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        trace::init(path, &cwd, config.prompt.as_deref())?;
    }
    if let Some(path) = &config.trace_lib {
        trace::init_library(path)?;
    }
    if let Some(path) = &config.trace_research {
        trace::init_research(path)?;
    }
    if let Some(path) = &config.trace_gaps {
        trace::init_gaps(path)?;
    }

    // Server mode: hand off to the MCP stdio server and return (no banner, no REPL).
    if config.server {
        let result = run_mcp_server(config).await;
        trace::end(None);
        return result;
    }

    // The window, as a third entry beside the REPL and the MCP server. It needs
    // the model chosen first — picking one is a conversation, and the GUI has no
    // picker yet — so `--gui` without `--model` says so rather than opening an
    // empty window.
    if config.gui {
        let result = run_gui(config);
        trace::end(None);
        return result;
    }

    // A task given on the command line runs unattended and exits, so it takes neither
    // the keyboard nor the screen: its output is a log, and it has to read as one in a
    // pipe or a scheduled task as much as in a terminal. `exit` rather than a return,
    // because the code is the result — and only once `run` has returned, so the agent's
    // scratch directory has already been dropped and cleaned up.
    if let Some(task) = config.prompt.clone() {
        let code = headless::run(config, task).await?;
        trace::end(Some(code));
        std::process::exit(code);
    }

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

    outln!("          Welcome to Haico!");
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
    if let Some(searxng) = &config.searxng {
        outln!("Searching through SearxNG at: {}", searxng);
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
            outln!("{}", echo(PROMPT, &text));
            text
        } else {
            match io.keys.as_mut() {
                Some(keys) => {
                    let status = status_line(
                        agent.provider(),
                        agent.model(),
                        agent.reasoning_effort(),
                        agent.context_used(),
                        agent.context_tokens,
                    );
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
                                drive_turn(&mut agent, &mut io, &message, false).await;
                            }
                            continue;
                        }
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
                // The scratch directory is in this list and looks like a stray temp path
                // without the label — it is haico's own, not something the user added.
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
            // Listed apart from the directories above, because that is what they
            // are not: the file is in reach and nothing else in its directory is.
            if !agent.workspace.attached_files().is_empty() {
                outln!("Single files, named with @ in a message:");
                for file in agent.workspace.attached_files() {
                    outln!("  {}", display_path(file));
                }
            }
            outln!("\nAnything else is refused outright, for reads as well as writes.\n");
            continue;
        }

        if input == "/cd" || input.starts_with("/cd ") {
            let requested = input.strip_prefix("/cd").unwrap_or("").trim();
            if requested.is_empty() {
                outln!("Working directory: {}\n", display_path(&cwd));
                continue;
            }
            match agent.change_directory(std::path::Path::new(requested)) {
                Ok(moved) => {
                    // The project is now the new one — its instructions, its memory, its
                    // branch, its writable root. What has not moved is the conversation,
                    // and that is the whole of the reminder: the model is still carrying
                    // a history about somewhere else.
                    cwd = moved;
                    io.editor.set_commands(commands::for_cwd());
                    outln!("Working directory: {}", display_path(&cwd));
                    if let Some((name, _)) = &agent.project_context {
                        outln!("Loaded project context from {}", name);
                    }
                    outln!("The conversation came with you — /clear (or /new) to start fresh here.");
                    // Said after the reminder rather than before it, because it is the
                    // alternative to it: one throws the history away, the other swaps it
                    // for the one this directory was left in.
                    if agent.has_saved_session() {
                        outln!("A conversation was saved here before — /resume picks it up instead.");
                    }
                    outln!();
                }
                Err(why) => outln!("{}\n", why),
            }
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

        if input == "/resume" {
            if !agent.has_saved_session() {
                outln!("No saved conversation for this directory.\n");
                continue;
            }
            // Dropped through `clear` rather than by emptying the history here: what goes
            // with a conversation and what survives it is the core's business, and a
            // second answer to that question in the frontend is the one that would drift.
            // The screen goes with it for `/clear`'s reason — the transcript on it belongs
            // to a conversation that is no longer in the history.
            out::clear_screen();
            let _ = agent.clear();
            match agent.resume() {
                Ok(restored) => outln!(
                    "Resumed {} message(s) from {} (model then: {}).\n",
                    restored.messages, restored.saved_at, restored.model
                ),
                Err(e) => outln!("Could not resume: {:#}. The conversation is empty.\n", e),
            }
            continue;
        }

        if input == "/clear" || input == "/new" || input == "/reset" {
            // What is discarded, what survives and why is `Agent::clear`'s
            // business. The screen is this frontend's: a transcript of a
            // conversation that no longer exists goes with it, the way it always
            // has in the window. What is left here is saying what happened.
            out::clear_screen();
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

        if input == "/skills" {
            let found = skills::discover(&skills::roots_for_cwd());
            if found.is_empty() {
                outln!("No skills found. Put <name>/SKILL.md under .haico/skills, .agents/skills");
                outln!("or .claude/skills in the project, or under {}.
", store::display("skills"));
            } else {
                outln!("{} skill(s):", found.len());
                for skill in &found {
                    outln!("  /{} — {}", skill.name, skill.description);
                    outln!("      {}", display_path(&skill.dir));
                }
                outln!();
            }
            continue;
        }

        if input == "/update-skills" {
            let changed = agent.reload_skills();
            io.editor.set_commands(commands::for_cwd());
            let count = skills::discover(&skills::roots_for_cwd()).len();
            if changed {
                outln!("Skills reloaded: {} available. The system prompt was updated.
", count);
            } else {
                outln!("Skills unchanged: {} available.
", count);
            }
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
                // message. The conversation itself is kept. It also records the
                // choice in the config file, which is what comes back here.
                let kept = agent.switch_model(model.clone());

                outln!(
                    "Now using model: {}{}",
                    model,
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
            continue;
        }

        if input == "/provider" || input.starts_with("/provider ") {
            let requested = input.strip_prefix("/provider").unwrap_or("").trim().to_string();
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
                continue;
            }

            let chosen = if requested.is_empty() {
                // The caret starts on the provider in force, so Enter alone changes nothing.
                let current = agent
                    .provider()
                    .and_then(|p| providers.iter().position(|name| name == p))
                    .unwrap_or(0);
                match io.keys.as_mut() {
                    Some(keys) => match select_interactive(keys, "provider", providers, current).await {
                        Ok(name) => Some(name),
                        Err(e) => {
                            outln!("{}\n", e);
                            None
                        }
                    },
                    None => {
                        outln!("Choosing a provider needs an interactive terminal; give its name: /provider <name>\n");
                        None
                    }
                }
            } else if providers.contains(&requested) {
                Some(requested)
            } else {
                outln!(
                    "There is no provider named '{}'; configured: {}\n",
                    requested,
                    providers.join(", ")
                );
                None
            };
            let Some(name) = chosen else { continue };
            if agent.provider() == Some(name.as_str()) {
                outln!("Already using provider: {}\n", name);
                continue;
            }

            let Some(section) = agent.provider_section(&name).cloned() else { continue };
            let url = section.url.clone().unwrap_or_default();
            let model = match section.model.clone() {
                Some(model) => model,
                None => {
                    let token = section.token.clone().filter(|t| !t.trim().is_empty());
                    match find_model(&url, token.as_deref(), io.keys.as_mut()).await {
                        Ok(model) => model,
                        Err(e) => {
                            // Nothing has changed yet: the switch happens only with a model
                            // to switch to, so a server that is down leaves the session on
                            // the provider that works.
                            outln!("{:#}\nStill using provider: {}\n", e, agent.provider().unwrap_or("-"));
                            continue;
                        }
                    }
                }
            };

            // The conversation is kept, as with /model; the url, token, model and every
            // per-model setting move with the switch.
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
            outln!(
                "  context {} ({})\n",
                agent.context_tokens,
                agent.settings.context_source
            );
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
            outln!("  provider          {}", agent.provider().unwrap_or("-"));
            outln!("  url               {}", agent.client.base_url);
            outln!(
                "  token             {}",
                if agent.client.token.is_some() { "set" } else { "not set" }
            );
            outln!(
                "  context           {}  [{}]",
                agent.context_tokens, agent.settings.context_source
            );
            outln!("  temperature       {}", agent.settings.temperature.describe());
            outln!("  top_p             {}", agent.settings.top_p.describe());
            outln!("  top_k             {}", agent.settings.top_k.describe());
            outln!("  min_p             {}", agent.settings.min_p.describe());
            outln!("  repeat_penalty    {}", agent.settings.repeat_penalty.describe());
            outln!("  max_tokens        {}", agent.settings.max_tokens.describe());
            outln!("  reasoning_effort  {}", agent.settings.reasoning_effort.describe());
            // Unset still says more than the bare "-" the request fields share, but for the
            // opposite reason it used to: it no longer sends anything at all, so what needs
            // saying is not what it does but what it does not do — inherit the level above.
            outln!(
                "  sub_agent_effort  {}",
                match &agent.settings.sub_agent_effort {
                    Some(effort) => format!("{}  [config [models]]", effort),
                    None => "-  [unset (the model's own default, never the level above)]"
                        .to_string(),
                }
            );
            // Shown as what is actually in force rather than as what the file said, since
            // /reasoning can have moved it since. The source string is only honest for the
            // former case, so it is replaced rather than kept when they differ.
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
                "  system prompt     {}",
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
        if input == "/set"
            || input.starts_with("/set ")
            || input.starts_with("/temp")
            || input.starts_with("/effort")
        {
            // `/temp 0.7` is the shorthand that predates `/set`, kept because it
            // is what fingers already know. `/effort high` earns the same: it is a
            // knob that gets changed with the task in hand rather than tuned once.
            let rest = if let Some(rest) = input.strip_prefix("/temp") {
                format!("temperature {}", rest.trim())
            } else if let Some(rest) = input.strip_prefix("/effort") {
                format!("reasoning_effort {}", rest.trim())
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
                Ok(kept) => {
                    match agent.knob_value(knob) {
                        // "the config file is updated" rather than calling the value
                        // shown the saved one: clearing `context` writes too, and what
                        // it writes is the key's absence, not the number now in force.
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

        if input == "/timers" || input.starts_with("/timers ") {
            let argument = input.strip_prefix("/timers").unwrap_or("").trim();
            if argument.is_empty() {
                outln!("{}\n", agent.server.describe_timers());
            } else if argument == "cancel" {
                let cancelled = agent.server.timers.lock().unwrap().clear();
                outln!("Cancelled {} timer(s).\n", cancelled);
            } else if let Some(id) = argument.strip_prefix("cancel ").and_then(|n| n.trim().parse::<u32>().ok()) {
                // The model is not told: it finds out from the reminder never arriving, or
                // from `timer list` if it asks — the same as a background job stopped by /bg.
                match agent.server.timers.lock().unwrap().cancel(id) {
                    Some(timer) => outln!("Cancelled timer {} ({}).\n", id, timer.schedule.describe()),
                    None => outln!("There is no timer {}.\n", id),
                }
            } else {
                outln!("Usage: /timers, /timers cancel <id>, /timers cancel\n");
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
                        "completed" => "x",
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
            report_context(
                &agent.messages,
                agent.context_used(),
                agent.context_is_measured(),
                agent.context_tokens,
                &agent.reasoning_stats,
            );
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
                agent.set_reasoning(chosen);
                outln!("Chain of thought: {} (this session).\n", agent.reasoning);
            } else {
                outln!("[error] unknown setting '{}'. Use keep, compress or drop.\n", rest);
            }
            continue;
        }

        if input == "/compact" {
            let (context_tokens, before) = (agent.context_tokens, agent.context_used());
            match compact_history(&agent.client, &mut agent.messages, context_tokens, before).await {
                // The same reclaim as /clear, on the one path that actually shrank the
                // session. A compaction that failed freed nothing to go with it.
                Ok(summary) => {
                    agent.review_cache.clear();
                    // The history that held anything remembered this session has just
                    // become a paragraph of notes; the index is now the only thing that
                    // still knows, so it goes into the prompt here.
                    agent.refresh_memory();
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
        let skill_instruction;
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
        } else if let Some(expanded) = skills::expand_command(&skills::roots_for_cwd(), input) {
            // `/init`'s trick again: the skill's instructions become the message and run as
            // an ordinary turn. Built-in commands were all matched above, so a skill that
            // shares a name with one is still there for the model, just not by typing it.
            skill_instruction = expanded;
            skill_instruction.as_str()
        } else {
            input
        };

        drive_turn(&mut agent, &mut io, input, initializing).await;
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

    trace::end(None);
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
    use crate::testutil::{looping_stream, serve_sequence, test_config};

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
        run_and_watch(bodies, input, mode, Reasoning::Keep).await.0
    }

    /// As `run_one_in`, but also every event the turn emitted.
    ///
    /// This is what the event stream bought the tests: the turn is driven with no
    /// frontend at all — no terminal, no keyboard, no presenter — and what it
    /// *said* can be asserted on directly instead of inferred from output. An
    /// `Ask` arriving here is answered by nobody, so its reply channel drops and
    /// the turn hears a refusal, which is exactly what a test wants: no prompt
    /// can hang the suite.
    ///
    /// The reasoning policy is a parameter for the same reason the mode is: it
    /// changes what the turn does after a reply rather than what it says to the
    /// server, so a fixture cannot express it.
    async fn run_and_watch(
        bodies: &'static [&'static str],
        input: &str,
        mode: Mode,
        reasoning: Reasoning,
    ) -> ((Agent, usize), Vec<AgentEvent>) {
        let (url, served) = serve_sequence(bodies).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        agent.mode = mode;
        agent.reasoning = reasoning;

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

    /// A `--prompt` run decides whether to exit, follow up or report a failure from
    /// what `run_turn` hands back, so an answer and a failed request must not look
    /// alike. A server that answers garbage is the failure that needs no timing.
    #[tokio::test]
    async fn a_turn_says_whether_it_ended_in_an_answer() {
        let _capture = out::capture_guard();
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"all done"}}]}"#;
        let cases: [(&'static [&'static str], TurnEnd); 2] =
            [(&[FINAL], TurnEnd::Answered), (&["not json"], TurnEnd::Failed)];
        for (bodies, expected) in cases {
            let (url, _) = serve_sequence(bodies).await;
            let cwd = env::current_dir().unwrap();
            let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
            agent.mode = Mode::Auto;
            let link = TurnLink::new(agent.mode);
            let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
            assert_eq!(run_turn(&mut agent, "go", false, &tx, &link).await, expected, "for {bodies:?}");
        }
    }

    /// A timer that comes due while a turn is working joins at the round boundary,
    /// after the tool replies — the only place a user message may go mid-turn — and
    /// says it came from a timer. One that is not due yet stays armed.
    #[tokio::test]
    async fn a_due_timer_joins_the_turn_after_the_tool_replies() {
        let _capture = out::capture_guard();
        let (url, served) = serve_sequence(&[
            r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"get_date","arguments":"{}"}}]}}]}"#,
            r#"{"choices":[{"message":{"role":"assistant","content":"checked"}}]}"#,
        ])
        .await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
        {
            use crate::timers::Schedule;
            use std::time::{Duration, Instant};
            let mut timers = agent.server.timers.lock().unwrap();
            // Set two minutes ago for one minute: due now.
            let past = Instant::now() - Duration::from_secs(120);
            timers.add(Schedule::Once(Duration::from_secs(60)), "check the tests", past, chrono::Local::now()).unwrap();
            timers.add(Schedule::Once(Duration::from_secs(3600)), "much later", Instant::now(), chrono::Local::now()).unwrap();
        }

        let link = TurnLink::new(Mode::Normal);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(run_turn(&mut agent, "go", false, &tx, &link).await, TurnEnd::Answered);
        assert_eq!(*served.lock().unwrap(), 2);

        let roles: Vec<&str> = agent.messages.iter().filter_map(|m| m["role"].as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "user", "assistant"]);
        let note = agent.messages[4]["content"].as_str().unwrap();
        assert!(note.starts_with("[Timer 1 fired"), "{note}");
        assert!(note.ends_with("check the tests"), "{note}");

        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(notices(&seen).iter().any(|n| matches!(n, Notice::TimersFired(f) if f.len() == 1 && f[0].id == 1)));
        let left: Vec<u32> = agent.server.timers.lock().unwrap().list().iter().map(|t| t.id).collect();
        assert_eq!(left, vec![2]);

        // And they go with the conversation.
        agent.clear();
        assert!(agent.server.timers.lock().unwrap().is_empty());
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
        let _capture = out::capture_guard();
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

    /// `@path` in a message puts the file in the message, which is the whole
    /// point: the turn must reach the model with the contents already there,
    /// rather than spending a round on a `read` the user has already asked for.
    #[tokio::test]
    async fn a_mentioned_file_arrives_in_the_message() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"seen"}}]}"#;

        let dir = testutil::TempDir::new("mention-turn");
        let file = dir.join("notes.txt");
        std::fs::write(&file, "the quick brown fox").unwrap();

        let input = format!("what is in @{}?", file.display());
        let (agent, served) = run_one(&[ANSWER], &input).await;

        // One request, not two: nothing was fetched by a tool call.
        assert_eq!(served, 1);
        let sent = agent
            .messages
            .iter()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap();
        assert!(sent.contains("the quick brown fox"), "contents missing from: {sent}");
        // The instruction is still the last thing the model reads, and the
        // reference it was written as is still in it.
        assert!(sent.trim_end().ends_with(&input), "the request is not last in: {sent}");

        // Read through the `read` tool, so the file counts as seen and an edit
        // is not refused for a file the model was just handed.
        assert!(
            agent.server.check_may_modify(&file).is_none(),
            "the mentioned file was not marked read — the saved round is spent re-reading it"
        );
    }

    /// A mentioned image cannot go in the text, so the message stops being
    /// text: `content` becomes the parts array, with the image as a data URL.
    #[tokio::test]
    async fn a_mentioned_image_becomes_a_content_part() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"seen"}}]}"#;

        let dir = testutil::TempDir::new("mention-image-turn");
        let png = dir.join("shot.png");
        // A signature and some bytes after it: nothing decodes this, and what
        // is asserted is the shape of the message, not the picture.
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03").unwrap();
        let notes = dir.join("notes.txt");
        std::fs::write(&notes, "some prose").unwrap();

        let input = format!("what is in @{} — see also @{}", png.display(), notes.display());
        let (agent, served) = run_one(&[ANSWER], &input).await;
        assert_eq!(served, 1);

        let message = agent
            .messages
            .iter()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .cloned()
            .unwrap();
        let parts = message["content"].as_array().expect("content should be a parts array");

        // The image first, the text last — the request is what the model should
        // read at the end.
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "image_url");
        assert!(
            parts[0]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,"),
            "not a png data URL: {}",
            parts[0]["image_url"]["url"]
        );
        assert_eq!(parts[1]["type"], "text");

        // The text part still carries the text file and the request itself, so
        // mixing the two kinds in one message loses neither.
        let text = parts[1]["text"].as_str().unwrap();
        assert!(text.contains("some prose"), "the text mention is missing: {text}");
        assert!(text.trim_end().ends_with(&input), "the request is not last in: {text}");

        // And the base64 is costed as an image rather than as prose. Left alone,
        // a screenshot reads as hundreds of thousands of tokens and auto mode
        // starts compacting a history that compaction cannot shrink. Measured
        // against the same message with a realistic payload in it, since the
        // twelve-byte one above would pass either way.
        let mut heavy = message.clone();
        heavy["content"][0]["image_url"]["url"] =
            Value::String(format!("data:image/png;base64,{}", "A".repeat(400_000)));
        let estimate = estimate_tokens(&[heavy]);
        assert!(
            estimate < 2000,
            "a 300 KB image is being counted as {estimate} tokens of text"
        );
    }

    /// The first reply after an auto-compaction may not end the turn by calling nothing: on a
    /// p2p_auth run it was one sentence of intent, and the unattended run exited as finished.
    #[tokio::test]
    async fn a_reply_that_calls_nothing_right_after_a_compaction_does_not_end_the_turn() {
        let _capture = out::capture_guard();
        let bodies: &'static [&'static str] = &[
            // The compactor's notes.
            r#"{"choices":[{"message":{"role":"assistant","content":"Task: x\nCurrent state: tests not run\nNext step: run the tests"}}]}"#,
            // Intent, and nothing called.
            r#"{"choices":[{"message":{"role":"assistant","content":"Let me check the current state."}}]}"#,
            // Sent back, it answers properly.
            r#"{"choices":[{"message":{"role":"assistant","content":"All tests pass."}}]}"#,
        ];
        let (url, served) = serve_sequence(bodies).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
        agent.mode = Mode::Auto;
        // Tiny, so the history is over the compaction line on the first round; and a history
        // with something in it, since one message is nothing to compact.
        agent.context_tokens = 100;
        agent.messages.push(serde_json::json!({"role": "user", "content": "build it"}));
        agent.messages.push(serde_json::json!({"role": "assistant", "content": "Built half of it."}));

        let link = TurnLink::new(Mode::Auto);
        let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(run_turn(&mut agent, "go", false, &tx, &link).await, TurnEnd::Answered);
        assert_eq!(*served.lock().unwrap(), 3);

        let last = agent.messages.last().unwrap();
        assert_eq!(last["content"], "All tests pass.");
        assert!(
            agent.messages.iter().any(|m| m["role"] == "user"
                && m["content"].as_str().is_some_and(|c| c.contains("You replied without calling a tool"))),
            "the note is missing"
        );
    }

    /// The size of the history comes from the server's count once one has arrived, plus an
    /// estimate of only what was added after it; a history that shrank since falls back to
    /// the estimate with the tools array counted. A p2p_auth run reached 130,820 of 131,072
    /// tokens on an estimate that still read 107k, so compaction never started.
    #[tokio::test]
    async fn the_context_is_sized_by_the_servers_count() {
        let _capture = out::capture_guard();
        let (mut agent, _) = run_one(
            &[r#"{"choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":50000,"completion_tokens":3,"total_tokens":50003}}"#],
            "hi",
        )
        .await;
        let measured = agent.measured.expect("the reply's usage was not recorded");
        assert_eq!(measured.prompt_tokens, 50000);
        assert!(agent.context_is_measured());

        // Everything after the measured request is estimated on top of the count.
        let grown = estimate_tokens(&agent.messages) - measured.estimate;
        assert_eq!(agent.context_used(), 50000 + grown);
        agent.messages.push(serde_json::json!({"role": "user", "content": "x".repeat(4000)}));
        assert_eq!(agent.context_used(), 50000 + grown + estimate_tokens(&agent.messages[agent.messages.len() - 1..]));

        // A history that shrank is not the one that was counted.
        agent.messages.truncate(1);
        assert!(!agent.context_is_measured());
        assert_eq!(agent.context_used(), estimate_tokens(&agent.messages) + estimate_tokens(&agent.tools));

        // A server that reports no usage leaves the estimate in charge.
        let (agent, _) = run_one(&[r#"{"choices":[{"message":{"role":"assistant","content":"hello"}}]}"#], "hi").await;
        assert!(agent.measured.is_none());
        assert_eq!(agent.context_used(), estimate_tokens(&agent.messages) + estimate_tokens(&agent.tools));
    }

    /// The edit from the p2p_auth trace: `new_string` carrying a second copy of `old_string`
    /// after an escaped quote. The file must get only the intended text, and the reply must
    /// say what was cut, or the model goes on making the same slip.
    #[tokio::test]
    async fn an_edit_with_its_old_string_leaked_into_new_string_is_repaired_and_said_to_be() {
        let _capture = out::capture_guard();

        let dir = testutil::TempDir::new("leaked-edit");
        let file = dir.join("lib.rs");
        std::fs::write(&file, "let (tx, mut rx) = channel(64);\nlet node = 1;\n").unwrap();
        let path = file.display().to_string();

        let call = |id: &str, name: &str, arguments: serde_json::Value| -> &'static str {
            Box::leak(
                format!(
                    r#"{{"choices":[{{"message":{{"role":"assistant","content":null,
                    "tool_calls":[{{"id":"{id}","type":"function","function":{{"name":"{name}","arguments":{}}}}}]}}}}]}}"#,
                    serde_json::to_string(&arguments.to_string()).unwrap()
                )
                .into_boxed_str(),
            )
        };
        let old = "let (tx, mut rx) = channel(64);";
        let leaked = format!("let (tx, rx) = channel(64);\", \"old_string\": \"{}\"}}", old);
        let edit = serde_json::json!({"file_path": path, "new_string": leaked, "old_string": old});
        let bodies: &'static [&'static str] = Box::leak(Box::new([
            call("c1", "read", serde_json::json!({"file_path": path})),
            call("c2", "edit", edit.clone()),
            r#"{"choices":[{"message":{"role":"assistant","content":"done"}}]}"#,
        ]));
        let (url, requests) = serve_sequence(bodies).await;

        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
        agent.workspace.add(dir.path()).unwrap();
        // Approved up front: in a test nobody answers the prompt, which reads as a refusal.
        if let permissions::Gate::Ask { keys, .. } = permissions::gate_for(&agent.workspace, "edit", &edit) {
            agent.permissions.allow(&keys);
        }

        let link = TurnLink::new(Mode::Normal);
        let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, "fix it", false, &tx, &link).await;
        assert_eq!(*requests.lock().unwrap(), 3);

        assert_eq!(std::fs::read_to_string(&file).unwrap(), "let (tx, rx) = channel(64);\nlet node = 1;\n");
        let reply = agent
            .messages
            .iter()
            .find(|m| m["tool_call_id"] == "c2")
            .and_then(|m| m["content"].as_str())
            .unwrap()
            .to_string();
        assert!(reply.contains("Your `new_string` argument ended with a second copy"), "{reply}");
    }

    /// `read` on an image returns the picture, and the picture cannot ride in
    /// the `role: "tool"` reply — that message may carry text and nothing else.
    /// So it arrives as a user message, and only once every tool call in the
    /// round has its reply.
    #[tokio::test]
    async fn reading_an_image_puts_it_in_a_user_message_after_the_tool_reply() {
        let _capture = out::capture_guard();

        let dir = testutil::TempDir::new("read-image");
        let png = dir.join("shot.png");
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03").unwrap();

        // The model asks to read it, then answers. The path is escaped for JSON
        // because a Windows path is full of backslashes.
        let call = format!(
            r#"{{"choices":[{{"message":{{"role":"assistant","content":null,
            "tool_calls":[{{"id":"c1","type":"function","function":{{"name":"read","arguments":{}}}}}]}}}}]}}"#,
            serde_json::to_string(
                &serde_json::json!({"file_path": png.display().to_string()}).to_string()
            )
            .unwrap()
        );
        let (url, requests) = serve_sequence(Box::leak(Box::new([
            &*Box::leak(call.into_boxed_str()),
            r#"{"choices":[{"message":{"role":"assistant","content":"a tiny png"}}]}"#,
        ])))
        .await;

        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
        // The temp directory is outside the project, so without this the read is
        // refused by the workspace and never reaches the image path at all.
        agent.workspace.attach(&png);

        // `read` is on the auto-allowed list, so no prompt stands in the way.
        let link = TurnLink::new(Mode::Normal);
        let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, "look at the screenshot", false, &tx, &link).await;
        assert_eq!(*requests.lock().unwrap(), 2);

        let roles: Vec<&str> =
            agent.messages.iter().filter_map(|m| m["role"].as_str()).collect();
        // system, user, assistant(tool_calls), tool, user(image), assistant
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "user", "assistant"]);

        // The tool reply is text and names the file. It must not have become an
        // array: a server rejects an image part in a tool message outright.
        let reply = &agent.messages[3];
        let text = reply["content"].as_str().expect("a tool reply must be a string");
        assert!(text.contains("PNG"), "the reply does not say what the file is: {text}");

        // And the picture is in the user message that follows it.
        let parts = agent.messages[4]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "image_url");
        assert!(parts[0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    /// A model with no vision head rejects the request carrying a picture, and
    /// every request after it, so the conversation is dead until the picture
    /// leaves. The turn takes it out once and tries again rather than handing
    /// back a history that can only fail.
    #[tokio::test]
    async fn a_refused_image_is_stripped_and_the_request_retried() {
        let _capture = out::capture_guard();

        let dir = testutil::TempDir::new("strip-image");
        let png = dir.join("shot.png");
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03").unwrap();

        // The server refuses the first request outright, then answers. That is
        // the shape of a text-only model being handed an image.
        let (url, served) = serve_sequence(&[
            "not json, and not a 200 either",
            r#"{"choices":[{"message":{"role":"assistant","content":"no picture here"}}]}"#,
        ])
        .await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);

        let link = TurnLink::new(Mode::Normal);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, &format!("look at @{}", png.display()), false, &tx, &link).await;

        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(
            notices(&seen).iter().any(|n| matches!(n, Notice::ImagesStripped(_))),
            "the retry never happened: {:?}",
            notices(&seen)
        );
        assert_eq!(*served.lock().unwrap(), 2, "the request was not tried a second time");

        // No image left anywhere, and the message that carried one still says
        // something rather than being emptied.
        for message in &agent.messages {
            if let Some(parts) = message["content"].as_array() {
                assert!(
                    !parts.iter().any(|p| p["type"] == "image_url"),
                    "an image survived the strip"
                );
                assert!(!parts.is_empty(), "a message was left with no content at all");
            }
        }
        let user = message_text(&agent.messages[1]);
        assert!(user.contains("removed"), "the model was not told why: {user}");
    }

    /// A message carrying an image must still reach a text summarizer as
    /// something, or the turn it began summarizes as though nobody spoke.
    #[test]
    fn an_image_message_still_has_text_for_the_summarizer() {
        let message = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "why is this button grey?"}
            ]
        });
        let text = message_text(&message);
        assert!(text.contains("why is this button grey?"));
        assert!(text.contains("[an image]"), "the image left no mark: {text}");
        assert!(!text.contains("AAAA"), "the base64 reached the summarizer: {text}");
    }

    /// A mentioned file outside every root comes into reach, and only that file
    /// does. `/add` widens to a directory because that is what it is asked for;
    /// a mention is asked about one file, and `@~/.ssh/config` must not hand
    /// over the keys beside it.
    #[tokio::test]
    async fn a_mention_outside_the_workspace_reaches_that_file_alone() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"seen"}}]}"#;

        // Outside the working directory the test agent is built in, which is the
        // crate root.
        let dir = testutil::TempDir::new("mention-outside");
        let named = dir.join("named.txt");
        let sibling = dir.join("sibling.txt");
        std::fs::write(&named, "in reach").unwrap();
        std::fs::write(&sibling, "not in reach").unwrap();

        let (agent, _) = run_one(&[ANSWER], &format!("read @{}", named.display())).await;

        assert!(agent.workspace.may_write(&named), "the named file is still refused");
        assert!(
            !agent.workspace.may_read(&sibling),
            "the mention opened the whole directory, not the one file"
        );
        // And the model is told, or it finds out through a refused call instead.
        // The note is built per request from the workspace, so this is what the
        // next request would carry rather than what the system prompt holds.
        let note = agent.workspace.note();
        assert!(
            note.contains(&display_path(&absolute_path(&named))),
            "the attached file is not in the workspace note: {note}"
        );
        assert!(
            !note.contains(&display_path(&absolute_path(&sibling))),
            "the note names a file that was never attached: {note}"
        );
    }

    /// A reply with no tool calls ends the turn on the answer, and the user's
    /// message and the assistant's both land in the history — the ordinary case,
    /// asserted so that the interesting ones above have a baseline.
    #[tokio::test]
    async fn a_plain_answer_ends_the_turn() {
        let _capture = out::capture_guard();
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
        let _capture = out::capture_guard();
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
        let _capture = out::capture_guard();
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

    /// The failure that motivated `repetition`: a model that never calls a tool
    /// and never finishes, only restates itself, which the tool-call loop guard
    /// cannot see. The reply is cut off and dropped, the model is shown what it
    /// was repeating, and the next reply is taken normally. Auto mode, because
    /// unattended is where it cost the most.
    #[tokio::test]
    async fn a_reply_that_repeats_itself_is_dropped_and_the_model_told() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"fixed it"}}]}"#;
        let bodies: &'static [&'static str] = Box::leak(Box::new([looping_stream(), ANSWER]));

        let ((agent, served), events) = run_and_watch(bodies, "debug it", Mode::Auto, Reasoning::Keep).await;

        assert_eq!(served, 2);
        assert!(
            notices(&events).iter().any(|n| matches!(n, Notice::Repeating { retrying: true, .. })),
            "notices: {:?}",
            notices(&events)
        );
        // The looping reply never reached the history: neither its text nor the
        // content it would have ended on.
        let history = serde_json::to_string(&agent.messages).unwrap();
        assert!(!history.contains("finished anyway"), "the looping reply was stored");
        // Folded into the user's own message, since the loop was the turn's first reply.
        let user: Vec<&Value> = agent.messages.iter().filter(|m| m["role"] == "user").collect();
        assert_eq!(user.len(), 1, "two user messages in a row");
        let asked = user[0]["content"].as_str().unwrap();
        assert!(asked.starts_with("debug it") && asked.contains("repeating the same passage"), "{asked}");
        assert_eq!(agent.messages.last().unwrap()["content"], "fixed it");
    }

    /// And when telling it does not help, the turn gives up instead of paying
    /// for the same loop forever — which in auto mode, with no round cap, is
    /// the only thing that would stop it.
    #[tokio::test]
    async fn a_model_that_keeps_repeating_itself_ends_the_turn() {
        let _capture = out::capture_guard();
        let bodies: &'static [&'static str] = Box::leak(Box::new([looping_stream()]));
        let (url, served) = serve_sequence(bodies).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        agent.mode = Mode::Auto;
        let link = TurnLink::new(agent.mode);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();

        assert_eq!(run_turn(&mut agent, "debug it", false, &tx, &link).await, TurnEnd::Failed);
        assert_eq!(*served.lock().unwrap(), 2);
        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(notices(&seen).iter().any(|n| matches!(n, Notice::Repeating { retrying: false, .. })));
    }

    /// A review that ends without a verdict is reported to the model as a
    /// temporary failure to retry, and the user is not asked: auto mode is for
    /// walking away. Worded as a refusal it was taken as final — an unattended
    /// model whose `write_file` met a review cut off at its token cap wrote its
    /// files through `echo` for the rest of the run — and sent to the prompt, it
    /// parked the run on a dialog nobody was there to answer.
    #[tokio::test]
    async fn a_review_without_a_verdict_is_a_temporary_error_to_retry() {
        let _capture = out::capture_guard();
        const CALL: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"run_command","arguments":"{\"command\":\"cargo --version\"}"}}]}}]}"#;
        // What a review cut off at its cap comes back as: all reasoning, no answer.
        const CUT_OFF: &str = r#"{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"length"}]}"#;
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"done"}}]}"#;
        let (url, served) = serve_sequence(&[CALL, CUT_OFF, FINAL]).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        agent.mode = Mode::Auto;
        let link = TurnLink::new(agent.mode);
        // A frontend that is there but never answers, like a REPL whose user went
        // for coffee: a prompt sent to it would wait forever, so the timeout is
        // what turns "it asked" into a failure rather than a hung suite.
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();

        let turn = run_turn(&mut agent, "check the toolchain", false, &tx, &link);
        tokio::time::timeout(std::time::Duration::from_secs(10), turn)
            .await
            .expect("the turn waited on a prompt nobody answered");

        assert_eq!(*served.lock().unwrap(), 3, "call, review, then the reply to the retry note");
        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(!seen.iter().any(|e| matches!(e, AgentEvent::Ask(_))), "the user was asked");
        assert!(
            notices(&seen).iter().any(|n| matches!(n, Notice::ReviewFailed { tool, .. } if tool == "run_command")),
            "notices: {:?}",
            notices(&seen)
        );
        let replies = tool_replies(&agent.messages);
        assert_eq!(replies.len(), 1);
        let (id, content) = &replies[0];
        assert_eq!(id, "c1");
        assert!(
            content.starts_with("Temporary error while evaluating the 'run_command' call")
                && content.contains("make the same call again"),
            "{content}"
        );
    }

    /// The turn narrates itself, and this is the assertion that could not be
    /// written before: driven with no frontend at all, what it *said* is checked
    /// directly rather than inferred from terminal output.
    #[tokio::test]
    async fn a_malformed_call_is_narrated_as_well_as_answered() {
        let _capture = out::capture_guard();
        const MALFORMED: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"bad","type":"function","function":{"name":"read","arguments":"nope"}}]}}]}"#;
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"done"}}]}"#;

        let (_, events) =
            run_and_watch(&[MALFORMED, FINAL], "go", Mode::Normal, Reasoning::Keep).await;
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

    /// Compression replaces the chain of thought in the history, so the
    /// transcript is the only place the summary can still be read — which is why
    /// the notice carries the text as well as the two figures. Reporting a saving
    /// and nothing else leaves the reader no way to judge whether what survived
    /// was worth keeping, and the original is gone by then.
    #[tokio::test]
    async fn a_compressed_trace_is_reported_with_the_text_that_replaced_it() {
        let _capture = out::capture_guard();
        // Well over the 800-token threshold, so the turn spends a request on
        // compressing it and the second body is the compressor's answer.
        // `serve_sequence` wants `'static` bodies and a trace this long has to be
        // built rather than written out, so it is leaked — a test process is the
        // one place where that costs nothing.
        let reply: &'static str = Box::leak(
            format!(
                "{{\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"Done.\",\
                 \"reasoning_content\":\"{}\"}}}}]}}",
                "z".repeat(8_000)
            )
            .into_boxed_str(),
        );
        const SUMMARY: &str = r#"{"choices":[{"message":{"role":"assistant",
            "content":"Tried the regex path, too slow."}}]}"#;
        let bodies: &'static [&'static str] = Box::leak(Box::new([reply, SUMMARY]));

        let ((agent, _), events) =
            run_and_watch(bodies, "go", Mode::Normal, Reasoning::Compress).await;
        let said = notices(&events);
        let reported = said.iter().find_map(|n| match n {
            Notice::ReasoningCompressed { summary, .. } => Some(summary.clone()),
            _ => None,
        });
        assert_eq!(
            reported.as_deref(),
            Some("Tried the regex path, too slow."),
            "the compression was reported without the text it produced: {said:?}"
        );
        // And what was reported is what the history actually carries — the point
        // of reading it back out of the message rather than passing it alongside
        // the figures.
        assert_eq!(
            agent.messages.last().unwrap()["reasoning_content"].as_str(),
            reported.as_deref()
        );
    }

    /// Cancellation is a message, not a dropped future — see
    /// `TurnLink::until_cancelled`. Cancelling before the turn starts means it
    /// stops at its first interruptible point, and the invariant still holds: the
    /// user message is in the history and nothing is left half-written.
    #[tokio::test]
    async fn a_cancelled_turn_stops_without_orphaning_anything() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"hello"}}]}"#;

        let (url, served) = serve_sequence(&[ANSWER]).await;
        let cwd = env::current_dir().unwrap();
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

    /// Esc during the first of two calls in one reply. The turn used to stop at the
    /// call it interrupted and leave the second id unanswered, and the next request
    /// was rejected for it.
    #[tokio::test]
    async fn an_interrupt_mid_round_still_answers_every_call_of_the_reply() {
        let _capture = out::capture_guard();
        let sleeper = if cfg!(windows) { "ping -n 30 127.0.0.1 > NUL" } else { "sleep 30" };
        let reply = Box::leak(
            serde_json::json!({"choices": [{"message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "slow", "type": "function", "function": {
                        "name": "run_command",
                        "arguments": serde_json::json!({"command": sleeper}).to_string()}},
                    {"id": "never", "type": "function", "function": {
                        "name": "get_date", "arguments": "{}"}}
                ]
            }}]})
            .to_string()
            .into_boxed_str(),
        );
        let bodies: &'static [&'static str] = Box::leak(Box::new([&*reply]));

        let (url, served) = serve_sequence(bodies).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        let link = TurnLink::new(agent.mode);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();

        // The frontend: approve the command, and press Esc once it is running.
        let frontend = async {
            while let Some(event) = events.recv().await {
                match event {
                    AgentEvent::Ask(Ask::Permission { reply, .. }) => {
                        let _ = reply.send(PermissionResponse::Allow);
                    }
                    AgentEvent::ToolStarted { name, .. } if name == "run_command" => link.cancel(),
                    _ => {}
                }
            }
        };
        let turn = async {
            let end = run_turn(&mut agent, "go", false, &tx, &link).await;
            drop(tx);
            end
        };
        let (end, ()) = tokio::join!(turn, frontend);

        assert_eq!(end, TurnEnd::Interrupted);
        assert_eq!(*served.lock().unwrap(), 1);
        let replies = tool_replies(&agent.messages);
        let ids: Vec<&str> = replies.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["slow", "never"], "every call of the reply needs its reply");
        assert!(replies[1].1.contains("Not run"), "got: {}", replies[1].1);
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
    if let Some(notice) = starter_config_notice(&config) {
        eprintln!("{}", notice);
    }
    if let Some(notice) = config_error_notice(&config) {
        eprintln!("{}", notice);
    }
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
