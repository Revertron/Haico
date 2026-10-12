//! Command line, environment and the config file in haico's own directory, layered.

use anyhow::{Context, Result};
use getopts::Options;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Item, Table};

// Which renderer draws the REPL.
//
// `Inline` is the original: finished output goes into the terminal's own scrollback and
// only the input box is redrawn. `Fullscreen` takes the alternate screen and owns the
// whole viewport, which is what lets the transcript be scrolled, re-wrapped on a resize
// and edited after the fact — none of which is possible once a line belongs to the
// terminal. It is the default because those are what a streaming REPL needs: the input
// box stays where it is while a long turn scrolls past above it, and scrolling back to
// read something no longer means losing the box off the top of the screen. Inline stays
// reachable because it is the fallback whenever the alternate screen is the wrong
// answer: a dumb terminal, a session being piped somewhere, or a user who would rather
// keep native scrollback. The piped case needs no setting of its own — `enter_screen`
// is only reached when stdin is a terminal.
#[derive(Deserialize, Default, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Tui {
    Inline,
    #[default]
    Fullscreen,
}

// Whether the full-screen renderer asks the terminal for the mouse wheel.
//
// `Scroll` asks for wheel and button reporting and nothing else — never motion reporting,
// which is what would take drag-selection away from the terminal. On Windows this also
// means turning on the console's VT input mode, and there selecting text needs Shift held
// while dragging. `Off` leaves the mouse entirely to the terminal; PageUp and PageDown
// still scroll.
#[derive(Deserialize, Default, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Mouse {
    #[default]
    Scroll,
    Off,
}

// What happens to a thinking model's chain of thought once its turn is over.
//
// `Keep` is the default because it is what haico has always done, and because it is not
// the obviously wrong answer it looks like: llama.cpp's own `--reasoning-preserve` exists
// to keep reasoning across the whole history rather than only the last assistant message,
// and the measurements attached to it report multi-turn recall several times better with
// it on. Reasoning is load-bearing context on templates built for it, not noise.
//
// `Drop` is the other end — strip it the moment the turn is over, which is what most
// clients do and what the classic Qwen3 and DeepSeek-R1 templates do for themselves.
// It is here as the baseline to measure the other two against.
//
// `Compress` is the middle that neither the server nor the template offers: replace the
// trace with the hard points of it — what was tried and rejected, and the intent it ended
// on — so the useful part survives at a fraction of the tokens. It costs one extra
// generation per turn, which is why it is off unless asked for.
#[derive(Deserialize, Default, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Reasoning {
    #[default]
    Keep,
    Compress,
    Drop,
}

impl std::fmt::Display for Reasoning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Reasoning::Keep => "keep",
            Reasoning::Compress => "compress",
            Reasoning::Drop => "drop",
        })
    }
}

impl Reasoning {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "keep" => Some(Reasoning::Keep),
            "compress" => Some(Reasoning::Compress),
            "drop" => Some(Reasoning::Drop),
            _ => None,
        }
    }
}

pub(crate) struct Config {
    pub(crate) llm_url: String,
    pub(crate) token: Option<String>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) server: bool,
    // Open a window rather than driving the terminal. Only meaningful in a build
    // with the `gui` feature; without it `main` says so and exits rather than
    // silently falling back to the REPL, which would look like the flag worked.
    pub(crate) gui: bool,
    pub(crate) url_supplied: bool,
    pub(crate) searxng: Option<String>,
    // The model's context window, in tokens. Only used to tell the user how close the
    // conversation is to filling it; haico never truncates on its own.
    // Model-dependent settings as given on the command line or in the environment. Kept
    // separate from the file's values because they outrank both config layers, and the
    // model is not known until after interactive selection.
    pub(crate) context_arg: Option<usize>,
    pub(crate) temperature_arg: Option<f64>,
    pub(crate) max_tokens_arg: Option<u64>,
    pub(crate) top_p_arg: Option<f64>,
    pub(crate) top_k_arg: Option<u64>,
    pub(crate) min_p_arg: Option<f64>,
    pub(crate) repeat_penalty_arg: Option<f64>,
    // The parsed config file, consulted again whenever the model changes.
    pub(crate) file: FileConfig,
    pub(crate) config_path: Option<PathBuf>,
    // The starter config was written by this run, because there was none — said once at
    // startup with its path, since a file nobody knows about is a file nobody edits.
    pub(crate) config_created: bool,
    // Continue this directory's previous conversation instead of starting fresh.
    pub(crate) resume: bool,
    // A task to run unattended in auto mode, exiting when it is done. `None` is the REPL.
    pub(crate) prompt: Option<String>,
    // Append every model call to this file as JSON lines — see `src/trace.rs`.
    pub(crate) trace: Option<PathBuf>,
    // Append each library lookup to this file as a question-and-answer SFT example.
    pub(crate) trace_lib: Option<PathBuf>,
    // The same for each research call, left for someone to curate.
    pub(crate) trace_research: Option<PathBuf>,
    // Append each question the library could not answer to this file.
    pub(crate) trace_gaps: Option<PathBuf>,
    // The directory to work in, entered by `main` before anything else happens.
    pub(crate) dir: Option<PathBuf>,
    // Which renderer to use. Not model-dependent, so it never goes through
    // `resolve_settings` — it is decided once, before the first thing is drawn.
    pub(crate) tui: Tui,
    pub(crate) mouse: Mouse,
}

// Parse configuration from command-line arguments and environment variables

pub(crate) fn parse_config() -> Result<Config> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optopt("u", "url", "LLM server URL (default: http://127.0.0.1:11434 for Ollama)", "URL");
    opts.optopt("s", "system-prompt", "System prompt (string or path to text file)", "PROMPT");
    opts.optopt("m", "model", "Model name (skip model selection)", "MODEL");
    opts.optopt("t", "token", "API token for servers with authentication enabled", "TOKEN");
    opts.optopt("", "searxng", "Base URL of a SearxNG instance to search through instead of DuckDuckGo", "URL");
    opts.optopt("", "context", "Model context window in tokens, for the /context report (default: 131072)", "TOKENS");
    opts.optopt("", "temperature", "Sampling temperature (default: leave it to the server)", "VALUE");
    opts.optopt("", "max-tokens", "Cap on tokens generated per reply (default: leave it to the server)", "N");
    opts.optopt("", "top-p", "Nucleus sampling cutoff (default: leave it to the server)", "VALUE");
    opts.optopt("", "top-k", "Top-k sampling cutoff (default: leave it to the server)", "N");
    opts.optopt("", "min-p", "Minimum-probability sampling cutoff (default: leave it to the server)", "VALUE");
    opts.optopt("", "repeat-penalty", "Repetition penalty (default: leave it to the server)", "VALUE");
    opts.optflag("r", "resume", "Continue this directory's previous conversation");
    opts.optopt("p", "prompt", "Run this task unattended in auto mode, then exit", "TASK");
    opts.optopt("", "trace", "Append every model call to FILE as JSON lines, for training data", "FILE");
    opts.optopt("", "trace-lib", "Append each library question and answer to FILE, ready for fine-tuning", "FILE");
    opts.optopt("", "trace-research", "Append each research question and answer to FILE, for curating", "FILE");
    opts.optopt("", "trace-gaps", "Append each question the library could not answer to FILE", "FILE");
    opts.optopt("d", "dir", "Work in DIR instead of the current directory", "DIR");
    opts.optflag("", "server", "Run as an stdio MCP server (no REPL, no prompts)");
    opts.optflag("", "gui", "Open a window instead of using the terminal");
    opts.optflag("h", "help", "Print this help message");

    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing arguments: {}", e);
            print_usage(&program, opts);
            std::process::exit(1);
        }
    };

    if matches.opt_present("h") {
        print_usage(&program, opts);
        std::process::exit(0);
    }

    // The config file supplies defaults for everything below, so the usual invocation
    // is a bare `haico`. Read before anything else resolves, since it sits under the command
    // line and the environment but above the built-in defaults.
    let config_path = config_file_path();
    let config_created = config_path.as_deref().is_some_and(write_starter_config);
    let file_config = load_file_config(config_path.as_deref());

    // Parse LLM URL
    // Priority: command-line arg > environment variable > config file > default
    let llm_url = if let Some(url) = matches.opt_str("u") {
        url
    } else if let Ok(url) = env::var("LLM_URL") {
        url
    } else if let Some(url) = file_config.url.clone() {
        url
    } else {
        // Default to Ollama's port (previously was LM Studio's 15200)
        "http://127.0.0.1:11434".to_string()
    };

    // Whether the URL was set explicitly (flag, env var or config) vs. falling back to the
    // default. web_fetch's LLM summarization requires the URL to be supplied, not defaulted.
    let url_supplied =
        matches.opt_present("u") || env::var("LLM_URL").is_ok() || file_config.url.is_some();

    // Parse API token, for servers that have authentication enabled.
    // Priority: command-line arg > environment variable > config file > none.
    let token = matches
        .opt_str("t")
        .or_else(|| env::var("LM_API_TOKEN").ok())
        .or_else(|| file_config.token.clone())
        .filter(|t| !t.trim().is_empty());

    // Parse system prompt
    // Can be a string or a path to a text file
    let system_prompt = if let Some(prompt_arg) = matches.opt_str("s") {
        // Check if it's a file path
        let path = PathBuf::from(&prompt_arg);
        if path.exists() && path.is_file() {
            // Read from file
            Some(fs::read_to_string(&path)
                .context(format!("Failed to read system prompt from file: {}", prompt_arg))?)
        } else {
            // Treat as a string
            Some(prompt_arg)
        }
    } else {
        file_config.system_prompt.clone().map(|prompt| {
            let path = PathBuf::from(&prompt);
            if path.is_file() {
                fs::read_to_string(&path).unwrap_or(prompt)
            } else {
                prompt
            }
        })
    };

    // Parse model
    let model = matches.opt_str("m").or_else(|| file_config.model.clone());

    // Run as an stdio MCP server instead of the interactive REPL
    let server = matches.opt_present("server");
    let gui = matches.opt_present("gui");

    // SearxNG instance for web_search, replacing DuckDuckGo when set.
    // Priority: command-line arg > environment variable > none (DuckDuckGo).
    let searxng = matches
        .opt_str("searxng")
        .or_else(|| env::var("SEARXNG_URL").ok())
        .or_else(|| file_config.searxng.clone())
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty());

    // The model's context window, used only to report how full the conversation is.
    // Priority: command-line arg > environment variable > default.
    // Only what the command line or environment actually supplied. The rest is resolved
    // once the model is known, because a model's own section outranks the file's top level.
    let context_arg = matches
        .opt_str("context")
        .or_else(|| env::var("HAICO_CONTEXT").ok())
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|tokens| *tokens > 0);

    let temperature_arg = matches
        .opt_str("temperature")
        .and_then(|value| value.trim().parse::<f64>().ok());

    let max_tokens_arg = matches
        .opt_str("max-tokens")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|n| *n > 0);

    let top_p_arg = matches
        .opt_str("top-p")
        .and_then(|value| value.trim().parse::<f64>().ok());
    let top_k_arg = matches
        .opt_str("top-k")
        .and_then(|value| value.trim().parse::<u64>().ok());
    let min_p_arg = matches
        .opt_str("min-p")
        .and_then(|value| value.trim().parse::<f64>().ok());
    let repeat_penalty_arg = matches
        .opt_str("repeat-penalty")
        .and_then(|value| value.trim().parse::<f64>().ok());

    let file_tui = file_config.tui;
    let file_mouse = file_config.mouse;

    Ok(Config {
        llm_url,
        token,
        system_prompt,
        model,
        server,
        gui,
        url_supplied,
        searxng,
        context_arg,
        temperature_arg,
        max_tokens_arg,
        top_p_arg,
        top_k_arg,
        min_p_arg,
        repeat_penalty_arg,
        file: file_config,
        config_path,
        config_created,
        resume: matches.opt_present("r"),
        // A blank task is treated as none, rather than starting a run with nothing to do
        // that the model would fill by inventing a task of its own.
        prompt: matches.opt_str("p").filter(|task| !task.trim().is_empty()),
        // Made absolute here, while the current directory is still the one the command was
        // typed in: `--dir` moves the process before the trace is opened, and a script that
        // says `--trace runs.jsonl` means the runs.jsonl beside it, not one inside every
        // project it visits. `--system-prompt` gets this for free by being read above.
        trace: matches.opt_str("trace").map(|path| {
            env::current_dir().map(|cwd| cwd.join(&path)).unwrap_or_else(|_| PathBuf::from(&path))
        }),
        trace_lib: matches.opt_str("trace-lib").map(|path| {
            env::current_dir().map(|cwd| cwd.join(&path)).unwrap_or_else(|_| PathBuf::from(&path))
        }),
        trace_research: matches.opt_str("trace-research").map(|path| {
            env::current_dir().map(|cwd| cwd.join(&path)).unwrap_or_else(|_| PathBuf::from(&path))
        }),
        trace_gaps: matches.opt_str("trace-gaps").map(|path| {
            env::current_dir().map(|cwd| cwd.join(&path)).unwrap_or_else(|_| PathBuf::from(&path))
        }),
        dir: matches.opt_str("d").map(PathBuf::from),
        // The environment wins so a terminal that cannot cope with the alternate screen
        // can be worked around without editing the config file.
        tui: match env::var("HAICO_TUI").ok().as_deref().map(str::trim) {
            Some("fullscreen") => Tui::Fullscreen,
            Some("inline") => Tui::Inline,
            _ => file_tui.unwrap_or_default(),
        },
        mouse: match env::var("HAICO_MOUSE").ok().as_deref().map(str::trim) {
            Some("off") => Mouse::Off,
            Some("scroll") => Mouse::Scroll,
            _ => file_mouse.unwrap_or_default(),
        },
    })
}

// Project instruction files, in the order they are looked for. The first one found wins;
// AGENTS.md is the cross-tool convention, CLAUDE.md and HAICO.md are accepted because a
// repo that already has one should not need a second copy.

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelSettingsFile {
    pub(crate) context: Option<usize>,
    // Whether haico's own system prompt is sent for this model. Per-model only and true
    // unless said otherwise: a model fine-tuned with its own baked-in instructions can be
    // made worse by a second set layered on top, and that is a property of the model, not
    // of the machine or the project.
    pub(crate) use_system_prompt: Option<bool>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<u64>,
    pub(crate) min_p: Option<f64>,
    pub(crate) repeat_penalty: Option<f64>,
    // How hard the model should think, handed straight to the server as `reasoning_effort`.
    // A string rather than an enum on purpose: the levels belong to the model's own chat
    // template, not to the API — three on one model, `minimal` or `none` on the next — so a
    // fixed set here would reject the next model that ships a vocabulary of its own.
    pub(crate) reasoning_effort: Option<String>,
    // What the sub-agents ask for, which is deliberately not what the main conversation
    // asks for. They summarize, judge and look things up, and asking them to think as hard
    // as the coding model spends the user's waiting time on a chain of thought that is
    // thrown away with their message list.
    //
    // Unset says nothing at all, so the model's own default applies. It asked for none at
    // first, through the template, on the grounds that what a sub-agent thinks is discarded
    // either way — true, and beside the point: the compressor, answering cold, read the
    // trace it was handed as a question and answered it, storing a fix with a fenced code
    // block as that reply's chain of thought. The brief deliberation was what the summary
    // was being made out of. Naming a level here sends that instead, for a model that takes
    // one and where less of it is worth the wait.
    //
    // Per model only, like `use_system_prompt`: whether a template takes a level at all,
    // and what it calls it, is a property of the model rather than of the machine.
    pub(crate) sub_agent_effort: Option<String>,
    // "keep" (default), "compress" or "drop". Layered like the sampling settings rather
    // than per-model only, but it belongs here first: whether a model even produces a
    // chain of thought is a property of the model, so the useful setting is usually the
    // per-model one.
    pub(crate) reasoning: Option<Reasoning>,
    pub(crate) reasoning_threshold: Option<usize>,
}

// the config file (see `crate::store`), so the usual invocation is just `haico`.
//
// Unknown fields are rejected rather than ignored: a typo'd key that silently does nothing
// is the worst outcome for a config file, since it looks like the setting is not working.

#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileConfig {
    pub(crate) url: Option<String>,
    pub(crate) token: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) searxng: Option<String>,
    pub(crate) system_prompt: Option<String>,

    // Directories the file tools may read but never write, on top of the working
    // directory. Empty by default and deliberately so: the value of the boundary is that
    // nothing outside it is reachable unless it was named on purpose.
    #[serde(default)]
    pub(crate) read_paths: Vec<String>,

    // "fullscreen" (default) or "inline".
    pub(crate) tui: Option<Tui>,
    // "scroll" (default) or "off".
    pub(crate) mouse: Option<Mouse>,

    // Top-level fallbacks, used when the selected model has no entry of its own.
    pub(crate) context: Option<usize>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<u64>,
    pub(crate) min_p: Option<f64>,
    pub(crate) repeat_penalty: Option<f64>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) reasoning: Option<Reasoning>,
    pub(crate) reasoning_threshold: Option<usize>,

    #[serde(default)]
    pub(crate) models: std::collections::HashMap<String, ModelSettingsFile>,
}


pub(crate) fn config_file_path() -> Option<PathBuf> {
    crate::store::path("config.toml")
}

// The config written on first start.
//
// Every top-level key is commented out, so a fresh file changes nothing: `url` in particular
// must stay unset, because a supplied URL is what turns on `web_fetch`'s summarizer, and a
// starter that set it would switch that on for a server the user never named. The one live
// section is a model's own, which only applies when that model is the one in use. What the
// file is for is being read — a user who never opens it has no way to learn that `[models]`
// exists, or which keys it takes, short of reading this source.
//
// Examples are written `# key = value`, one space after the hash, and prose never starts that
// way: the test that uncomments every example and parses the result depends on it, and that
// test is what keeps a key renamed here from being handed to users as a typo that
// `deny_unknown_fields` then refuses.
pub(crate) const STARTER_CONFIG: &str = r#"# haico's config. Everything here is optional; the command line and the environment
# outrank it, and a model's own [models."<name>"] section outranks the top level.
# /model and /set context write their choices back into this file, leaving the rest alone.

# The OpenAI-compatible server: Ollama (11434), LM Studio (1234), llama-server (8080).
# url = "http://127.0.0.1:11434"
# token = "..."

# The model to start with, skipping the picker.
# model = "qwen2.5:14b"

# Search through a SearxNG instance instead of DuckDuckGo.
# searxng = "http://127.0.0.1:8888"

# Replaces haico's own system prompt: the text itself, or a path to a file holding it.
# system_prompt = "path/to/prompt.txt"

# Directories the file tools may read but never write, besides the working directory.
# read_paths = ["C:/src/shared-lib"]

# "fullscreen" or "inline", and whether the mouse wheel scrolls ("scroll") or not ("off").
# tui = "fullscreen"
# mouse = "scroll"

# Fallbacks for every model without a section of its own. The sampling settings are left
# to the server unless given.
# context = 131072
# temperature = 0.7
# max_tokens = 8192
# top_p = 0.95
# top_k = 40
# min_p = 0.05
# repeat_penalty = 1.1
# reasoning_effort = "medium"
# What happens to a thinking model's chain of thought after its turn: "keep", "compress"
# (summarized, once longer than reasoning_threshold tokens) or "drop".
# reasoning = "keep"
# reasoning_threshold = 800

# Per-model settings, keyed by the name the server reports. Takes every setting above
# except url, token, model, searxng, system_prompt, read_paths, tui and mouse, plus:
#   use_system_prompt = false     for a model trained with instructions of its own
#   sub_agent_effort = "low"      reasoning_effort for research, review and summaries
[models."tiel-coder-35b-a3b-mtp"]
context = 131072
temperature = 0.3
reasoning = "keep"
"#;

// What each entry path prints when this run wrote the starter config. One wording for all
// of them; where it goes (the REPL's screen, a log, stderr under MCP) is theirs to decide.
pub(crate) fn starter_config_notice(config: &Config) -> Option<String> {
    let path = config.config_path.as_deref().filter(|_| config.config_created)?;
    Some(format!(
        "Created a starter config at {} — every setting in it is described there.",
        crate::util::display_path(path)
    ))
}

// Write the starter config if there is no config at all. `true` only when it was written
// just now, so the caller can say where it went. Anything else — a file already there, no
// home directory, a directory that cannot be created — is `false` and silent: the config is
// optional, and failing to write an example is no reason to stop haico starting.
fn write_starter_config(path: &Path) -> bool {
    if path.exists() {
        return false;
    }
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_err() {
            return false;
        }
    }
    // `create_new`, so a second haico starting at the same moment cannot have its file
    // replaced by this one's.
    let Ok(mut file) = fs::OpenOptions::new().write(true).create_new(true).open(path) else {
        return false;
    };
    use std::io::Write;
    file.write_all(STARTER_CONFIG.as_bytes()).is_ok()
}

// Read the config file. A missing file is normal; a malformed one is reported and then
// ignored, so a stray character cannot stop haico starting.

pub(crate) fn load_file_config(path: Option<&std::path::Path>) -> FileConfig {
    let Some(path) = path else {
        return FileConfig::default();
    };
    let Ok(raw) = fs::read_to_string(path) else {
        return FileConfig::default();
    };

    match toml::from_str(&raw) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("[haico] ignoring {}: {}", path.display(), e);
            FileConfig::default()
        }
    }
}

// Writing one setting back into the config file.
//
// Two of them outlive the session: the model chosen with /model, and the context window
// set with /set context. Both are answers to "which model, and how big is its window" —
// asked once per machine and then true until the machine changes — where a sampling knob
// is something tried against the task in hand and rightly forgotten with the session.
//
// Edited with toml_edit rather than serialized back out of `FileConfig`, because the file
// belongs to the user: it is written by hand, commented, and ordered the way they wanted
// it. Rewriting it from the struct would answer "set one key" by deleting every comment
// in the file, and the loss would only be noticed later.

fn edit_config_file(path: &Path, edit: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<()> {
    // A missing file is the ordinary first case — the config is optional, and saving is
    // most use to the user who never wrote one. A *malformed* one is refused rather than
    // treated as empty: parsing failure here would replace whatever the user has with a
    // one-line file, which is the worst possible way to answer a typo.
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e).context(format!("could not read {}", path.display())),
    };
    let mut doc: DocumentMut = raw
        .parse()
        .with_context(|| format!("{} is not valid TOML", path.display()))?;

    edit(&mut doc)?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    fs::write(path, doc.to_string()).with_context(|| format!("could not write {}", path.display()))
}

// Record the model to start with next time.

pub(crate) fn save_default_model(path: &Path, model: &str) -> Result<()> {
    edit_config_file(path, |doc| {
        uncomment_example(doc, "model");
        doc["model"] = toml_edit::value(model);
        Ok(())
    })
}

// Turn a commented `# key = ...` example at the top of the file into the key itself, so a
// value saved there lands under the comment describing it.
//
// Without this, saving into the starter config put the key above the file's header: its
// top level holds only comments, which toml_edit keeps as the prefix of the first section's
// header, and a new top-level key is rendered before all of it. Splitting that prefix at the
// example gives the key everything above the example and the header everything below.
// Only when the key is absent and the top level holds no keys at all — otherwise a new key
// goes after the existing ones, which is already somewhere sensible.
fn uncomment_example(doc: &mut DocumentMut, key: &str) {
    if doc.contains_key(key) || doc.iter().any(|(_, item)| !item.is_table()) {
        return;
    }
    // The first header in the file, which holds the comments above it. An implicit
    // `[models]` has no header of its own; the text sits on its first real subsection.
    fn first_header(table: &mut Table) -> Option<&mut Table> {
        let mut first: Option<&mut Table> = None;
        for (_, item) in table.iter_mut() {
            let Some(sub) = item.as_table_mut() else { continue };
            let sub = if sub.is_implicit() {
                match first_header(sub) {
                    Some(sub) => sub,
                    None => continue,
                }
            } else {
                sub
            };
            if first.as_ref().is_none_or(|f| sub.position() < f.position()) {
                first = Some(sub);
            }
        }
        first
    }
    let Some(header) = first_header(doc.as_table_mut()) else { return };
    let Some(prefix) = header.decor().prefix().and_then(|p| p.as_str()).map(str::to_string) else {
        return;
    };

    let example = format!("# {key} = ");
    let Some(start) = prefix.lines().scan(0, |at, line| {
        let here = *at;
        *at += line.len() + 1;
        Some((here, line))
    })
    .find(|(_, line)| line.starts_with(&example))
    .map(|(at, _)| at) else {
        return;
    };
    let end = prefix[start..].find('\n').map_or(prefix.len(), |n| start + n + 1);

    header.decor_mut().set_prefix(prefix[end..].to_string());
    doc[key] = toml_edit::value("");
    if let Some(mut k) = doc.as_table_mut().key_mut(key) {
        k.leaf_decor_mut().set_prefix(prefix[..start].to_string());
    }
}

// Record a model's context window in that model's own section, or take it back out
// again when `None`.
//
// Per model rather than at the top level because that is what a context window is: a
// property of the model, not of the machine. The top-level `context` stays what it was —
// the user's own fallback for every model they never set one for.

pub(crate) fn save_model_context(path: &Path, model: &str, context: Option<usize>) -> Result<()> {
    edit_config_file(path, |doc| {
        let models = doc.as_table_mut().entry("models").or_insert_with(|| {
            let mut table = Table::new();
            // Implicit so a new section is written as `[models."name"]` alone, rather
            // than as an empty `[models]` header with the real one underneath it.
            table.set_implicit(true);
            Item::Table(table)
        });
        // `models` written as an inline table parses and resolves fine, and rewriting it
        // into sections behind the user's back is not this function's business. Said
        // rather than silently skipped: a save that changed nothing must not report that
        // it saved.
        let models = models
            .as_table_mut()
            .context("[models] is an inline table in the config file; edit it by hand")?;

        let section = models
            .entry(model)
            .or_insert_with(|| Item::Table(Table::new()))
            .as_table_mut()
            .with_context(|| format!("[models.\"{}\"] is an inline table; edit it by hand", model))?;

        match context {
            Some(tokens) => section["context"] = toml_edit::value(tokens as i64),
            // Only the key goes. An emptied section stays where it is: it may carry the
            // user's own comments, and throwing those away to tidy up after one removed
            // setting is not a trade anybody asked for.
            None => {
                section.remove("context");
            }
        }
        Ok(())
    })
}

// Where a resolved setting came from, for /config.

pub(crate) const FROM_ARGS: &str = "command line / environment";
pub(crate) const FROM_MODEL: &str = "config [models]";
pub(crate) const FROM_GLOBAL: &str = "config top level";
pub(crate) const FROM_UNSET: &str = "unset (server decides)";

// One resolved setting: its value, if any, and which layer supplied it.

#[derive(Debug, Clone, Copy)]
pub(crate) struct Setting<T> {
    pub(crate) value: Option<T>,
    pub(crate) source: &'static str,
}


impl<T: std::fmt::Display> Setting<T> {
    pub(crate) fn describe(&self) -> String {
        match &self.value {
            Some(value) => format!("{}  [{}]", value, self.source),
            None => format!("-  [{}]", self.source),
        }
    }
}


impl<T> Setting<T> {
    pub(crate) fn unset(source: &'static str) -> Self {
        Setting {
            value: None,
            source,
        }
    }
}

// Resolve one setting through the layers, keeping track of which one supplied it.

pub(crate) fn pick<T>(from_args: Option<T>, per_model: Option<T>, global: Option<T>) -> Setting<T> {
    if let Some(value) = from_args {
        Setting { value: Some(value), source: FROM_ARGS }
    } else if let Some(value) = per_model {
        Setting { value: Some(value), source: FROM_MODEL }
    } else if let Some(value) = global {
        Setting { value: Some(value), source: FROM_GLOBAL }
    } else {
        Setting { value: None, source: FROM_UNSET }
    }
}

// The settings actually in force for the model in use, with their provenance.

#[derive(Debug, Clone)]
pub(crate) struct ResolvedSettings {
    pub(crate) context_tokens: usize,
    pub(crate) context_source: &'static str,
    pub(crate) temperature: Setting<f64>,
    pub(crate) max_tokens: Setting<u64>,
    pub(crate) top_p: Setting<f64>,
    pub(crate) top_k: Setting<u64>,
    pub(crate) min_p: Setting<f64>,
    pub(crate) repeat_penalty: Setting<f64>,
    // A request field like the six above rather than one of haico's own, so it travels with
    // them and reaches the server the same way — unset means the field is left out, and the
    // template's own default thinking budget applies.
    pub(crate) reasoning_effort: Setting<String>,
    // No provenance and no layers: per model or not at all, so there is only ever one place
    // it can have come from. `None` means the sub-agents say nothing about thinking, and get
    // whatever the model does by default.
    pub(crate) sub_agent_effort: Option<String>,
    // Not layered like the rest: it only exists per model, so there is nothing to fall
    // through to. Absent means true.
    pub(crate) use_system_prompt: bool,
    // What to do with a reply's chain of thought once the turn moves on, and how long a
    // trace has to be before compressing it is worth a round trip. Carried with the
    // provenance of the rest so /config can say where it came from, but read through
    // `reasoning`/`reasoning_threshold` below rather than unwrapped at each use — /reasoning
    // overrides it for the session, and the default has to be the same in both places.
    pub(crate) reasoning_setting: Setting<Reasoning>,
    pub(crate) reasoning_threshold_setting: Setting<usize>,
}

impl ResolvedSettings {
    pub(crate) fn reasoning(&self) -> Reasoning {
        self.reasoning_setting.value.unwrap_or_default()
    }

    // Where the policy in force came from. Not `Setting::describe`'s wording: the shared
    // FROM_UNSET reads "server decides", which is true of a sampling field left out of the
    // request and false of this one — an unset reasoning policy is haico keeping the chain
    // of thought, and nothing is asked of the server either way.
    pub(crate) fn reasoning_source(&self) -> &'static str {
        if self.reasoning_setting.value.is_some() {
            self.reasoning_setting.source
        } else {
            "built-in default"
        }
    }

    pub(crate) fn reasoning_threshold(&self) -> usize {
        self.reasoning_threshold_setting
            .value
            .unwrap_or(DEFAULT_REASONING_THRESHOLD)
    }
}

// Apply the layers for a specific model: command line / environment first, then that
// model's own section, then the file's top level, then the built-in default.
//
// Resolved per model rather than once at startup, so switching with /model picks up the new
// model's context window and sampling instead of carrying the old model's over — a 131072
// budget against a 32k model would be worse than having no config at all.

pub(crate) fn resolve_settings(config: &Config, model: &str) -> ResolvedSettings {
    let per_model = config.file.models.get(model).cloned().unwrap_or_default();

    let context = pick(config.context_arg, per_model.context, config.file.context);

    ResolvedSettings {
        context_tokens: context.value.unwrap_or(DEFAULT_CONTEXT_TOKENS),
        context_source: if context.value.is_some() {
            context.source
        } else {
            "built-in default"
        },
        temperature: pick(
            config.temperature_arg,
            per_model.temperature,
            config.file.temperature,
        ),
        max_tokens: pick(
            config.max_tokens_arg,
            per_model.max_tokens,
            config.file.max_tokens,
        ),
        top_p: pick(config.top_p_arg, per_model.top_p, config.file.top_p),
        top_k: pick(config.top_k_arg, per_model.top_k, config.file.top_k),
        min_p: pick(config.min_p_arg, per_model.min_p, config.file.min_p),
        repeat_penalty: pick(
            config.repeat_penalty_arg,
            per_model.repeat_penalty,
            config.file.repeat_penalty,
        ),
        // No command-line layer: which levels exist at all is a property of the model's
        // chat template, so the per-model section is where the answer belongs, and /effort
        // covers wanting a different one for a single session.
        reasoning_effort: pick(
            None,
            per_model.reasoning_effort,
            config.file.reasoning_effort.clone(),
        ),
        sub_agent_effort: per_model.sub_agent_effort,
        use_system_prompt: per_model.use_system_prompt.unwrap_or(true),
        // No command-line layer for either: both are properties of the model and of what
        // the user wants done with its output, neither of which is worth a flag when the
        // per-model section is right there and /reasoning covers the one-off case.
        reasoning_setting: pick(None, per_model.reasoning, config.file.reasoning),
        reasoning_threshold_setting: pick(
            None,
            per_model.reasoning_threshold,
            config.file.reasoning_threshold,
        ),
    }
}

// A saved conversation. Versioned so a later format change can be recognized and declined
// rather than deserializing into something subtly wrong.

pub(crate) const DEFAULT_CONTEXT_TOKENS: usize = 131_072;

// Fraction of the window at which compaction is worth suggesting. Late enough not to nag,
// early enough to leave room for the summarization round-trip itself.

pub(crate) const COMPACT_HINT_FRACTION: f64 = 0.75;

// Estimated tokens of chain of thought below which compressing it is not worth the round
// trip. Set by the arithmetic rather than by taste: compression pays off when the tokens
// saved, multiplied by the turns that will carry them, beat one extra generation. A few
// hundred tokens saved on a trace this short does not, and firing the compressor on every
// brief thought would add seconds per turn to save almost nothing. Raise it on a slow
// model, lower it on a fast one.

pub(crate) const DEFAULT_REASONING_THRESHOLD: usize = 800;


pub(crate) fn compact_hint_threshold(context_tokens: usize) -> usize {
    (context_tokens as f64 * COMPACT_HINT_FRACTION) as usize
}


// Fraction of the window at which auto mode compacts itself instead of asking. Higher than
// the hint above because this one throws detail away with nobody consulted: everything
// under it is still a conversation the user could have chosen to keep whole. It does not
// need to leave room for the summarization request itself — that goes out as a fresh two
// message conversation whose transcript `render_transcript` caps at 60k characters, not as
// the history it is about to replace.

pub(crate) const AUTOCOMPACT_FRACTION: f64 = 0.90;

pub(crate) fn autocompact_threshold(context_tokens: usize) -> usize {
    (context_tokens as f64 * AUTOCOMPACT_FRACTION) as usize
}


pub(crate) fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
    println!("\nEnvironment variables:");
    println!("  LLM_URL       LLM server URL (overridden by --url)");
    println!("  LM_API_TOKEN  API token for authenticated servers (overridden by --token)");
    println!("  SEARXNG_URL   SearxNG instance for web_search (overridden by --searxng)");
    println!("  HAICO_CONTEXT Model context window in tokens (overridden by --context)");
    println!("  HAICO_DEBUG   Set to 1 to append diagnostics to {}", crate::store::display("debug.log"));
    println!("\nConfig file: {} — supplies defaults for all of the above,", crate::store::display("config.toml"));
    println!("plus per-model sections. Command line and environment take precedence over it.");
    println!("\n  url   = \"http://127.0.0.1:1234\"");
    println!("  token = \"sk-lm-...\"");
    println!("  model = \"qwen2.5:14b\"");
    println!("\n  [models.\"qwen2.5:14b\"]");
    println!("  context        = 32768");
    println!("  temperature    = 0.2");
    println!("  top_p          = 0.95");
    println!("  top_k          = 40");
    println!("  min_p          = 0.05");
    println!("  repeat_penalty = 1.1");
    println!("  reasoning_effort = \"high\"  # how hard to think, if the model takes it");
    println!("  sub_agent_effort = \"low\"   # haico's own sub-agents; unset follows the model");
    println!("  use_system_prompt = false  # send only the project's own instructions");
    println!("  reasoning      = \"keep\"    # keep | compress | drop the chain of thought");
    println!("  reasoning_threshold = 800  # tokens of thinking before compressing is worth it");
    println!("\nreasoning decides what happens to a thinking model's chain of thought once its");
    println!("turn is over. 'keep' sends it back with the rest of the history, which is what");
    println!("llama.cpp's --reasoning-preserve is for; 'drop' strips it, as most clients do;");
    println!("'compress' replaces it with its hard points, at the cost of one extra request");
    println!("per reply. /reasoning switches it for one session.");
    println!("\nreasoning_effort is sent to the server verbatim, so the levels are whatever that");
    println!("model's chat template accepts — commonly low, medium and high. Left out, the");
    println!("field is omitted and the model thinks as much as it thinks. It is also handed to");
    println!("the chat template in chat_template_kwargs, \"on\"/\"off\" as enable_thinking and any");
    println!("other level as reasoning_effort, since a server may read the top-level field");
    println!("itself (LM Studio checks it against the model's metadata) or not at all");
    println!("(llama.cpp). /effort switches it for one session.");
    println!("\nsub_agent_effort is the same setting for haico's own sub-agents — research, the");
    println!("auto-mode reviewer and the reasoning compressor. Left out, nothing is sent and");
    println!("they think as much as the model thinks; the main conversation's level is never");
    println!("carried over to them.");
    println!("\nInstructions: {} holds preferences that apply everywhere; a", crate::store::display("HAICO.md"));
    println!("HAICO.md, AGENTS.md or CLAUDE.md in the working directory holds the project's");
    println!("own, and takes precedence over both. Both are sent even to a model configured");
    println!("with use_system_prompt = false.");
    println!("\n--prompt runs one task in the current directory in auto mode, with no keyboard:");
    println!("questions are left for the model to decide and anything the reviewer refuses is");
    println!("not run. A turn that stops with its plan unfinished, on a question, or with an");
    println!("empty reply is sent back to work up to three times. Exit codes: 0 done, 1 a turn");
    println!("failed, 2 still unfinished after the follow-ups, 130 interrupted with Ctrl+C.");
    println!("\n--trace appends one JSON object per line: a 'start' line, then every model call —");
    println!("the request exactly as sent (messages, tools, sampling) and the whole reply,");
    println!("reasoning included — tagged with its caller (main, research, reviewer, compressor,");
    println!("compactor, summarizer), then an 'end' line carrying the exit code of a --prompt");
    println!("run. Nothing is ever rewritten, so a compaction mid-run loses nothing. Use");
    println!("reasoning = \"keep\" if the earlier replies' thinking should stay in later requests.");
    println!("
--trace-lib appends each library lookup as one fine-tuning example: the question");
    println!("and the librarian's answer, {{\"messages\": [user, assistant]}}, nothing else. Answers");
    println!("that lean on the articles (\"the library does not cover\", \"the article shows\") are");
    println!("left out, and so is a pair already written by the same run.");
    println!("--trace-research writes research calls the same way plus the URLs read, unfiltered:");
    println!("web answers are unchecked, so curate them first. --trace-gaps records each question the");
    println!("library said it does not cover, with the librarian's answer: the articles to write.");
    println!("\nAnything left out is omitted from the request, so the server's own setting");
    println!("for that model still applies. top_k, min_p and repeat_penalty are extensions");
    println!("that llama.cpp, Ollama, LM Studio and vLLM understand, not OpenAI API fields.");
    println!("\nExamples:");
    println!("  {} --url http://127.0.0.1:15200  # Connect to LM Studio", program);
    println!("  {} --url http://127.0.0.1:11434  # Connect to Ollama (default)", program);
    println!("  {} --model \"qwen2.5:14b\"          # Use specific model", program);
    println!("  {} --system-prompt \"You are a helpful assistant\"  # Custom prompt", program);
    println!("  {} --system-prompt ./prompt.txt  # Load prompt from file", program);
    println!("  {} --server                       # Run as a stdio MCP server (for LM Studio)", program);
    println!("  LLM_URL=http://127.0.0.1:15200 {}  # Use environment variable", program);
    println!("  {} --token sk-lm-...              # Server with authentication enabled", program);
    println!("  LM_API_TOKEN=sk-lm-... {}         # Token via environment variable", program);
    println!("  {} --searxng http://127.0.0.1:8080  # Search via SearxNG instead of DuckDuckGo", program);
    println!("  {} --context 32768                # Warn about /compact sooner on a small window", program);
    println!("  {} --resume                       # Continue this directory's last conversation", program);
    println!("  {} -p \"Update dependencies and commit\"  # Run one task unattended, then exit", program);
    println!("  {} -p \"Fix the build\" --trace runs.jsonl  # ...and record every model call", program);
    println!("  {} --trace-lib library-qa.jsonl  # Collect library answers for fine-tuning", program);
    println!("  {} -d ../other-project -p \"Update dependencies\"  # ...in another directory", program);
}

// Run haico as an stdio MCP server. stdout/stdin carry the JSON-RPC protocol, so this
// function must NOT print to stdout — all diagnostics go to stderr via eprintln!.

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use crate::config::*;
    use crate::llm::*;
    use crate::permissions::*;
    use crate::render::*;
    use crate::session::*;
    use crate::testutil::*;
    use crate::tools::*;
    use crate::util::*;
    use serde_json::json;

    fn config_with(toml_text: &str) -> crate::Config {
        crate::Config {
            llm_url: "http://127.0.0.1:1234".to_string(),
            token: None,
            system_prompt: None,
            model: None,
            server: false,
            gui: false,
            url_supplied: false,
            searxng: None,
            context_arg: None,
            temperature_arg: None,
            max_tokens_arg: None,
            top_p_arg: None,
            top_k_arg: None,
            min_p_arg: None,
            repeat_penalty_arg: None,
            file: toml::from_str(toml_text).expect("test config should parse"),
            config_path: None,
            config_created: false,
            resume: false,
            prompt: None,
            trace: None,
            trace_lib: None,
            trace_research: None,
            trace_gaps: None,
            dir: None,
            tui: crate::config::Tui::Inline,
            mouse: crate::config::Mouse::Scroll,
        }
    }

    #[test]
    fn reasoning_layers_per_model_over_the_file_and_defaults_to_keep() {
        use crate::{resolve_settings, Reasoning, DEFAULT_REASONING_THRESHOLD};
        use crate::{FROM_GLOBAL, FROM_MODEL, FROM_UNSET};

        let text = r#"
            reasoning = "drop"

            [models."thinker"]
            reasoning = "compress"
            reasoning_threshold = 2000
        "#;
        let config = config_with(text);

        // The model's own section outranks the top level.
        let thinker = resolve_settings(&config, "thinker");
        assert_eq!(thinker.reasoning(), Reasoning::Compress);
        assert_eq!(thinker.reasoning_threshold(), 2_000);
        assert_eq!(thinker.reasoning_setting.source, FROM_MODEL);

        // A model with no section of its own still gets the file's top level, and the
        // threshold falls through to the built-in rather than to the other model's.
        let other = resolve_settings(&config, "other");
        assert_eq!(other.reasoning(), Reasoning::Drop);
        assert_eq!(other.reasoning_threshold(), DEFAULT_REASONING_THRESHOLD);
        assert_eq!(other.reasoning_setting.source, FROM_GLOBAL);

        // Unset anywhere is `keep` — what haico did before the setting existed. A default
        // that silently started discarding chains of thought would change every existing
        // user's conversation without them asking for it.
        let bare = config_with("");
        let settings = resolve_settings(&bare, "any");
        assert_eq!(settings.reasoning(), Reasoning::Keep);
        assert_eq!(settings.reasoning_setting.source, FROM_UNSET);
    }

    #[test]
    fn config_layers_resolve_in_the_documented_order() {
        use crate::{resolve_settings, DEFAULT_CONTEXT_TOKENS, FROM_ARGS, FROM_GLOBAL, FROM_MODEL};

        let text = r#"
            context = 8192
            temperature = 0.9

            [models."big-model"]
            context = 131072
            temperature = 0.2

            [models."small-model"]
            context = 32768
            top_k = 40
            min_p = 0.05
            repeat_penalty = 1.15
            reasoning_effort = "high"
            sub_agent_effort = "low"
        "#;

        // A model with its own section beats the file's top level.
        let config = config_with(text);
        let big = resolve_settings(&config, "big-model");
        assert_eq!(big.context_tokens, 131072);
        assert_eq!(big.context_source, FROM_MODEL);
        assert_eq!(big.temperature.value, Some(0.2));

        // A model section only overrides what it actually sets; the rest falls through.
        let small = resolve_settings(&config, "small-model");
        assert_eq!(small.context_tokens, 32768);
        assert_eq!(small.temperature.value, Some(0.9));
        assert_eq!(small.temperature.source, FROM_GLOBAL);
        // The extensions layer the same way as everything else.
        assert_eq!(small.top_k.value, Some(40));
        assert_eq!(small.top_k.source, FROM_MODEL);
        assert_eq!(small.min_p.value, Some(0.05));
        assert_eq!(small.repeat_penalty.value, Some(1.15));
        // A string setting layers by the same rule as the numbers, and is kept verbatim:
        // the level names are the model's chat template's, so there is nothing to normalize
        // it to and nothing to reject it against.
        assert_eq!(small.reasoning_effort.value.as_deref(), Some("high"));
        assert_eq!(small.reasoning_effort.source, FROM_MODEL);
        assert_eq!(big.reasoning_effort.value, None);
        // The sub-agents' own level is per model and nothing else, so a model without one
        // gets None — which is not "inherit", it is the setting that means no thinking.
        assert_eq!(small.sub_agent_effort.as_deref(), Some("low"));
        assert_eq!(big.sub_agent_effort, None);
        // ...and a model that does not name them leaves them unset, not inherited from a
        // sibling model's section.
        assert_eq!(big.top_k.value, None);
        assert_eq!(big.min_p.value, None);
        assert_eq!(big.repeat_penalty.value, None);

        // A model with no section gets the top level.
        let unknown = resolve_settings(&config, "never-heard-of-it");
        assert_eq!(unknown.context_tokens, 8192);
        assert_eq!(unknown.context_source, FROM_GLOBAL);

        // The command line outranks both.
        let mut config = config_with(text);
        config.context_arg = Some(4096);
        config.temperature_arg = Some(0.0);
        let big = resolve_settings(&config, "big-model");
        assert_eq!(big.context_tokens, 4096);
        assert_eq!(big.context_source, FROM_ARGS);
        assert_eq!(big.temperature.value, Some(0.0));
        assert_eq!(big.temperature.source, FROM_ARGS);

        // With nothing configured at all, context falls back and sampling stays unset.
        let empty = config_with("");
        let bare = resolve_settings(&empty, "any");
        assert_eq!(bare.context_tokens, DEFAULT_CONTEXT_TOKENS);
        assert_eq!(bare.temperature.value, None);
        assert_eq!(bare.max_tokens.value, None);
        assert_eq!(bare.top_p.value, None);
        assert_eq!(bare.top_k.value, None);
        assert_eq!(bare.min_p.value, None);
        assert_eq!(bare.repeat_penalty.value, None);
    }


    #[test]
    fn a_model_can_be_configured_without_haico_s_own_prompt() {
        use crate::{compose_system_prompt, resolve_settings};
        use crate::workspace::Workspace;

        let workspace = Workspace::new(&std::env::current_dir().unwrap(), &[]);

        let config = config_with(
            r#"
            [models."raw-model"]
            use_system_prompt = false

            [models."guided-model"]
            temperature = 0.2
        "#,
        );

        // Only the model that asked for it loses the prompt; absent means true, so no
        // existing config changes behavior.
        assert!(!resolve_settings(&config, "raw-model").use_system_prompt);
        assert!(resolve_settings(&config, "guided-model").use_system_prompt);
        assert!(resolve_settings(&config, "never-heard-of-it").use_system_prompt);

        let project = ("CLAUDE.md".to_string(), "PROJECT RULES".to_string());

        // Off: the user's and the project's own instructions still go through, ours does
        // not — including a custom one supplied with --system-prompt.
        let bare = compose_system_prompt(
            Some("CUSTOM"),
            "raw-model",
            false,
            Some("USER RULES"),
            Some(&project),
            None,            None,            None,
            &workspace,
        )
        .expect("project instructions still make a system message");
        assert!(bare.contains("USER RULES"));
        assert!(bare.contains("PROJECT RULES"));
        assert!(!bare.contains("CUSTOM"));
        assert!(!bare.contains("# Tone and style"), "got: {}", bare);

        // ...and with nothing to say at all there is no system message to send.
        assert!(
            compose_system_prompt(Some("CUSTOM"), "raw-model", false, None, None, None, None, None, &workspace)
                .is_none()
        );

        // On: unchanged from before — custom prompt if given, ours otherwise, project
        // instructions appended either way.
        let custom =
            compose_system_prompt(
                Some("CUSTOM"),
                "guided-model",
                true,
                None,
                Some(&project),
                None,                None,                None,
                &workspace,
            )
            .expect("a prompt is in force");
        assert!(custom.starts_with("CUSTOM"));
        assert!(custom.contains("PROJECT RULES"));
        let ours =
            compose_system_prompt(None, "guided-model", true, None, None, None, None, None, &workspace)
                .expect("built-in");
        assert!(ours.contains("guided-model"));
    }


    #[test]
    fn compaction_hint_scales_with_the_configured_window() {
        use crate::{compact_hint_threshold, DEFAULT_CONTEXT_TOKENS};

        // A 131k window must not nag at the point a 32k one would.
        let small = compact_hint_threshold(32_768);
        let large = compact_hint_threshold(DEFAULT_CONTEXT_TOKENS);
        assert!(small < large);
        assert!(large > 90_000, "got {}", large);
        // And it has to leave room for the summarization round-trip itself.
        assert!(large < DEFAULT_CONTEXT_TOKENS);
    }


    #[test]
    fn auto_compaction_fires_after_the_hint_and_before_the_window_ends() {
        use crate::{autocompact_threshold, compact_hint_threshold, DEFAULT_CONTEXT_TOKENS};

        for window in [8_192usize, 32_768, DEFAULT_CONTEXT_TOKENS] {
            // Auto mode summarizes itself only once the point where a user sitting there
            // would have been told to has gone by unheeded...
            assert!(compact_hint_threshold(window) < autocompact_threshold(window));
            // ...and while there is still window left for the turn to carry on into.
            assert!(autocompact_threshold(window) < window);
        }
    }


    #[test]
    fn a_typo_in_the_config_is_reported_not_ignored() {
        // A key that silently does nothing is the worst outcome for a config file: it looks
        // like the setting is broken rather than misspelled.
        assert!(toml::from_str::<crate::FileConfig>("temprature = 0.5").is_err());
        // Easy one to get wrong: the OpenAI spelling is not what these servers want.
        assert!(toml::from_str::<crate::FileConfig>("repetition_penalty = 1.1").is_err());
        assert!(toml::from_str::<crate::FileConfig>("topk = 40").is_err());
        assert!(toml::from_str::<crate::FileConfig>(
            "[models.\"m\"]\ncontex = 100"
        )
        .is_err());

        // A well-formed file still parses, including an empty one.
        assert!(toml::from_str::<crate::FileConfig>("").is_ok());
        let parsed: crate::FileConfig =
            toml::from_str("url = \"http://h:1\"\n[models.\"m\"]\ncontext = 8").unwrap();
        assert_eq!(parsed.url.as_deref(), Some("http://h:1"));
        assert_eq!(parsed.models["m"].context, Some(8));
    }


    #[test]
    fn a_broken_config_file_does_not_stop_startup() {
        use crate::load_file_config;

        let path = scratch("broken-config.toml", "this is not valid toml {{{");
        // Reported on stderr, then ignored: a stray character must not make haico refuse
        // to start.
        let config = load_file_config(Some(&path));
        assert!(config.url.is_none());
        assert!(config.models.is_empty());

        // A missing file is simply the no-config case.
        let absent = path.parent().unwrap().join("definitely-not-here.toml");
        let _ = std::fs::remove_file(&absent);
        assert!(load_file_config(Some(&absent)).url.is_none());
        assert!(load_file_config(None).url.is_none());

        let _ = std::fs::remove_file(&path);
    }

    // The file is the user's: hand-written, commented, ordered the way they wanted it.
    // Saving one setting has to leave all of that alone, which is the whole reason the
    // writing goes through toml_edit rather than through `FileConfig`.
    #[test]
    fn saving_the_model_leaves_the_rest_of_the_file_as_it_was() {
        let path = scratch(
            "save-default-model.toml",
            "# the machine's own notes\nurl = \"http://127.0.0.1:1234\"\n\n\
             [models.\"other\"]\n# small on purpose\ncontext = 8192\n",
        );

        save_default_model(&path, "qwen2.5:14b").unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("# the machine's own notes"), "comment lost:\n{raw}");
        assert!(raw.contains("# small on purpose"), "comment lost:\n{raw}");

        let parsed = load_file_config(Some(&path));
        assert_eq!(parsed.model.as_deref(), Some("qwen2.5:14b"));
        assert_eq!(parsed.url.as_deref(), Some("http://127.0.0.1:1234"));
        assert_eq!(parsed.models["other"].context, Some(8192));

        // Saving again replaces the key rather than adding a second one.
        save_default_model(&path, "llama3.1:8b").unwrap();
        assert_eq!(load_file_config(Some(&path)).model.as_deref(), Some("llama3.1:8b"));

        let _ = std::fs::remove_file(&path);
    }

    // Per model, because that is what a context window is. The top level stays the
    // user's own fallback for every model they never set one for.
    #[test]
    fn a_context_window_goes_into_the_model_s_own_section() {
        let path = scratch("save-model-context.toml", "context = 4096\n");

        save_model_context(&path, "qwen2.5:14b", Some(32768)).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("[models.\"qwen2.5:14b\"]"), "wrong shape:\n{raw}");
        let parsed = load_file_config(Some(&path));
        assert_eq!(parsed.context, Some(4096), "the top level is not the model's");
        assert_eq!(parsed.models["qwen2.5:14b"].context, Some(32768));

        save_model_context(&path, "qwen2.5:14b", Some(16384)).unwrap();
        let parsed = load_file_config(Some(&path));
        assert_eq!(parsed.models["qwen2.5:14b"].context, Some(16384));

        // Clearing takes the key out and nothing else with it.
        save_model_context(&path, "qwen2.5:14b", None).unwrap();
        let parsed = load_file_config(Some(&path));
        assert_eq!(parsed.models["qwen2.5:14b"].context, None);
        assert_eq!(parsed.context, Some(4096));

        let _ = std::fs::remove_file(&path);
    }

    // Saving is most use to the user who never wrote a config, so a missing file is
    // the ordinary case and not an error.
    #[test]
    fn saving_writes_a_config_that_was_never_there() {
        let path = scratch("save-into-nothing.toml", "");
        std::fs::remove_file(&path).unwrap();

        save_model_context(&path, "m", Some(8192)).unwrap();
        save_default_model(&path, "m").unwrap();

        let parsed = load_file_config(Some(&path));
        assert_eq!(parsed.model.as_deref(), Some("m"));
        assert_eq!(parsed.models["m"].context, Some(8192));

        let _ = std::fs::remove_file(&path);
    }

    // A file that does not parse is left exactly where it is. Starting from an empty
    // document instead would answer one stray character by replacing everything the
    // user has with a one-line file.
    #[test]
    fn saving_refuses_a_malformed_file_rather_than_replacing_it() {
        let path = scratch("save-over-broken.toml", "this is not valid toml {{{");
        let before = std::fs::read_to_string(&path).unwrap();

        assert!(save_default_model(&path, "m").is_err());
        assert!(save_model_context(&path, "m", Some(8192)).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);

        let _ = std::fs::remove_file(&path);
    }

    // As written, the starter changes nothing but its one model section. `url` above all:
    // a supplied URL turns on web_fetch's summarizer.
    #[test]
    fn the_starter_config_sets_nothing_but_its_example_model() {
        let parsed: FileConfig = toml::from_str(STARTER_CONFIG).expect("starter must parse");
        assert!(parsed.url.is_none() && parsed.model.is_none() && parsed.token.is_none());
        assert!(parsed.context.is_none() && parsed.temperature.is_none());
        assert!(parsed.read_paths.is_empty() && parsed.tui.is_none());
        assert_eq!(parsed.models.len(), 1);

        let tiel = &parsed.models["tiel-coder-35b-a3b-mtp"];
        assert_eq!(tiel.context, Some(131072));
        assert_eq!(tiel.temperature, Some(0.3));
        assert_eq!(tiel.reasoning, Some(Reasoning::Keep));
    }

    // Every commented example, uncommented, must still be a config haico accepts — a key
    // renamed in `FileConfig` and not here would be handed to users as a typo that
    // `deny_unknown_fields` then refuses.
    #[test]
    fn every_example_in_the_starter_config_is_a_real_setting() {
        let example = regex::Regex::new(r"^# ([a-z_]+ = .*)$").unwrap();
        let mut uncommented = 0;
        let text: String = STARTER_CONFIG
            .lines()
            .map(|line| match example.captures(line) {
                Some(caps) => {
                    uncommented += 1;
                    format!("{}\n", &caps[1])
                }
                None => format!("{line}\n"),
            })
            .collect();
        assert!(uncommented >= 15, "the pattern stopped matching the examples: {uncommented}");

        let parsed: FileConfig = toml::from_str(&text)
            .unwrap_or_else(|e| panic!("uncommented starter does not parse: {e}\n{text}"));
        assert!(parsed.url.is_some() && parsed.reasoning_threshold.is_some());
    }

    #[test]
    fn the_starter_config_is_written_once_and_never_over_a_file() {
        let path = scratch("starter-config.toml", "");
        std::fs::remove_file(&path).unwrap();

        assert!(write_starter_config(&path));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), STARTER_CONFIG);

        std::fs::write(&path, "url = \"http://mine\"\n").unwrap();
        assert!(!write_starter_config(&path));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "url = \"http://mine\"\n");

        let _ = std::fs::remove_file(&path);
    }

    // /model writes into the starter like into any other file: its comments stay, and the
    // model lands at the top level rather than inside the example's section.
    #[test]
    fn saving_into_the_starter_config_keeps_its_comments() {
        let path = scratch("starter-then-save.toml", STARTER_CONFIG);

        save_default_model(&path, "qwen2.5:14b").unwrap();
        save_model_context(&path, "qwen2.5:14b", Some(32768)).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("# Per-model settings, keyed by"), "comment lost:\n{raw}");
        // In place of its own example, under the comment describing it — not above the
        // file's header, where toml_edit puts a new key when the top level has none.
        assert!(
            raw.contains("# The model to start with, skipping the picker.\nmodel = \"qwen2.5:14b\"\n"),
            "model landed elsewhere:\n{raw}"
        );
        assert!(!raw.contains("# model ="), "the example was left beside the value:\n{raw}");
        let parsed = load_file_config(Some(&path));
        assert_eq!(parsed.model.as_deref(), Some("qwen2.5:14b"));
        assert_eq!(parsed.models["qwen2.5:14b"].context, Some(32768));
        assert_eq!(parsed.models["tiel-coder-35b-a3b-mtp"].context, Some(131072));

        let _ = std::fs::remove_file(&path);
    }

}
