use anyhow::{Context, Result};
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
mod diff;
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
mod progress;
mod prompt;
mod readable;
mod render;
mod repl;
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

use config::*;
use llm::*;
use tools::*;

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

    repl::run(config).await
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
    let mut agent = agent::Agent::new(config, model, &cwd);
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
