//! Command line, environment and ~/.haico/config.toml, layered.

use anyhow::{Context, Result};
use getopts::Options;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::PathBuf;

// Which renderer draws the REPL.
//
// `Inline` is the original: finished output goes into the terminal's own scrollback and
// only the input box is redrawn. `Fullscreen` takes the alternate screen and owns the
// whole viewport, which is what lets the transcript be scrolled, re-wrapped on a resize
// and edited after the fact — none of which is possible once a line belongs to the
// terminal. Inline stays reachable because it is the fallback whenever the alternate
// screen is the wrong answer: a dumb terminal, a session being piped somewhere, or a
// user who would rather keep native scrollback.
#[derive(Deserialize, Default, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Tui {
    #[default]
    Inline,
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
    pub(crate) repeat_penalty_arg: Option<f64>,
    // The parsed config file, consulted again whenever the model changes.
    pub(crate) file: FileConfig,
    pub(crate) config_path: Option<PathBuf>,
    // Continue this directory's previous conversation instead of starting fresh.
    pub(crate) resume: bool,
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
    opts.optopt("", "repeat-penalty", "Repetition penalty (default: leave it to the server)", "VALUE");
    opts.optflag("r", "resume", "Continue this directory's previous conversation");
    opts.optflag("", "server", "Run as an stdio MCP server (no REPL, no prompts)");
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

    // ~/.haico/config.toml supplies defaults for everything below, so the usual invocation
    // is a bare `haico`. Read before anything else resolves, since it sits under the command
    // line and the environment but above the built-in defaults.
    let config_path = config_file_path();
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
        url_supplied,
        searxng,
        context_arg,
        temperature_arg,
        max_tokens_arg,
        top_p_arg,
        top_k_arg,
        repeat_penalty_arg,
        file: file_config,
        config_path,
        resume: matches.opt_present("r"),
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
    pub(crate) repeat_penalty: Option<f64>,
    // "keep" (default), "compress" or "drop". Layered like the sampling settings rather
    // than per-model only, but it belongs here first: whether a model even produces a
    // chain of thought is a property of the model, so the useful setting is usually the
    // per-model one.
    pub(crate) reasoning: Option<Reasoning>,
    pub(crate) reasoning_threshold: Option<usize>,
}

// ~/.haico/config.toml, so the usual invocation is just `haico`.
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

    // "inline" (default) or "fullscreen".
    pub(crate) tui: Option<Tui>,
    // "scroll" (default) or "off".
    pub(crate) mouse: Option<Mouse>,

    // Top-level fallbacks, used when the selected model has no entry of its own.
    pub(crate) context: Option<usize>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<u64>,
    pub(crate) repeat_penalty: Option<f64>,
    pub(crate) reasoning: Option<Reasoning>,
    pub(crate) reasoning_threshold: Option<usize>,

    #[serde(default)]
    pub(crate) models: std::collections::HashMap<String, ModelSettingsFile>,
}


pub(crate) fn config_file_path() -> Option<PathBuf> {
    let home = env::var_os("USERPROFILE").or_else(|| env::var_os("HOME"))?;
    Some(PathBuf::from(home).join(".haico").join("config.toml"))
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
    pub(crate) repeat_penalty: Setting<f64>,
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
        repeat_penalty: pick(
            config.repeat_penalty_arg,
            per_model.repeat_penalty,
            config.file.repeat_penalty,
        ),
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


pub(crate) fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
    println!("\nEnvironment variables:");
    println!("  LLM_URL       LLM server URL (overridden by --url)");
    println!("  LM_API_TOKEN  API token for authenticated servers (overridden by --token)");
    println!("  SEARXNG_URL   SearxNG instance for web_search (overridden by --searxng)");
    println!("  HAICO_CONTEXT Model context window in tokens (overridden by --context)");
    println!("  HAICO_DEBUG   Set to 1 to append diagnostics to ~/.haico/debug.log");
    println!("\nConfig file: ~/.haico/config.toml — supplies defaults for all of the above,");
    println!("plus per-model sections. Command line and environment take precedence over it.");
    println!("\n  url   = \"http://127.0.0.1:1234\"");
    println!("  token = \"sk-lm-...\"");
    println!("  model = \"qwen2.5:14b\"");
    println!("\n  [models.\"qwen2.5:14b\"]");
    println!("  context        = 32768");
    println!("  temperature    = 0.2");
    println!("  top_p          = 0.95");
    println!("  top_k          = 40");
    println!("  repeat_penalty = 1.1");
    println!("  use_system_prompt = false  # send only the project's own instructions");
    println!("  reasoning      = \"keep\"    # keep | compress | drop the chain of thought");
    println!("  reasoning_threshold = 800  # tokens of thinking before compressing is worth it");
    println!("\nreasoning decides what happens to a thinking model's chain of thought once its");
    println!("turn is over. 'keep' sends it back with the rest of the history, which is what");
    println!("llama.cpp's --reasoning-preserve is for; 'drop' strips it, as most clients do;");
    println!("'compress' replaces it with its hard points, at the cost of one extra request");
    println!("per reply. /reasoning switches it for one session.");
    println!("\nInstructions: ~/.haico/HAICO.md holds preferences that apply everywhere; a");
    println!("HAICO.md, AGENTS.md or CLAUDE.md in the working directory holds the project's");
    println!("own, and takes precedence over both. Both are sent even to a model configured");
    println!("with use_system_prompt = false.");
    println!("\nAnything left out is omitted from the request, so the server's own setting");
    println!("for that model still applies. top_k and repeat_penalty are extensions that");
    println!("llama.cpp, Ollama, LM Studio and vLLM understand, not OpenAI API fields.");
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
            url_supplied: false,
            searxng: None,
            context_arg: None,
            temperature_arg: None,
            max_tokens_arg: None,
            top_p_arg: None,
            top_k_arg: None,
            repeat_penalty_arg: None,
            file: toml::from_str(toml_text).expect("test config should parse"),
            config_path: None,
            resume: false,
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
            repeat_penalty = 1.15
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
        assert_eq!(small.repeat_penalty.value, Some(1.15));
        // ...and a model that does not name them leaves them unset, not inherited from a
        // sibling model's section.
        assert_eq!(big.top_k.value, None);
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
        assert_eq!(bare.repeat_penalty.value, None);
    }


    #[test]
    fn a_model_can_be_configured_without_haico_s_own_prompt() {
        use crate::{compose_system_prompt, resolve_settings};

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
        )
        .expect("project instructions still make a system message");
        assert!(bare.contains("USER RULES"));
        assert!(bare.contains("PROJECT RULES"));
        assert!(!bare.contains("CUSTOM"));
        assert!(!bare.contains("# Tone and style"), "got: {}", bare);

        // ...and with nothing to say at all there is no system message to send.
        assert!(compose_system_prompt(Some("CUSTOM"), "raw-model", false, None, None).is_none());

        // On: unchanged from before — custom prompt if given, ours otherwise, project
        // instructions appended either way.
        let custom =
            compose_system_prompt(Some("CUSTOM"), "guided-model", true, None, Some(&project))
                .expect("a prompt is in force");
        assert!(custom.starts_with("CUSTOM"));
        assert!(custom.contains("PROJECT RULES"));
        let ours =
            compose_system_prompt(None, "guided-model", true, None, None).expect("built-in");
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

}
