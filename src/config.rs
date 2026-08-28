//! Command line, environment and ~/.haico/config.toml, layered.

use anyhow::{Context, Result};
use getopts::Options;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::PathBuf;

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

    // Which way round the terminal's own colors run: "light" or "dark". Top level only,
    // and deliberately not a per-model setting — it is a property of the screen haico is
    // running on, which does not change when the model does. /light and /dark override it
    // for the session.
    pub(crate) theme: Option<String>,

    // Directories the file tools may read but never write, on top of the working
    // directory. Empty by default and deliberately so: the value of the boundary is that
    // nothing outside it is reachable unless it was named on purpose.
    #[serde(default)]
    pub(crate) read_paths: Vec<String>,

    // Top-level fallbacks, used when the selected model has no entry of its own.
    pub(crate) context: Option<usize>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<u64>,
    pub(crate) repeat_penalty: Option<f64>,

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
    }
}

// A saved conversation. Versioned so a later format change can be recognized and declined
// rather than deserializing into something subtly wrong.

pub(crate) const DEFAULT_CONTEXT_TOKENS: usize = 131_072;

// Fraction of the window at which compaction is worth suggesting. Late enough not to nag,
// early enough to leave room for the summarization round-trip itself.

pub(crate) const COMPACT_HINT_FRACTION: f64 = 0.75;


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
    println!("\nConfig file: ~/.haico/config.toml — supplies defaults for all of the above,");
    println!("plus per-model sections. Command line and environment take precedence over it.");
    println!("\n  url   = \"http://127.0.0.1:1234\"");
    println!("  token = \"sk-lm-...\"");
    println!("  model = \"qwen2.5:14b\"");
    println!("  theme = \"dark\"          # the terminal's own colors: \"light\" or \"dark\"");
    println!("\n  [models.\"qwen2.5:14b\"]");
    println!("  context        = 32768");
    println!("  temperature    = 0.2");
    println!("  top_p          = 0.95");
    println!("  top_k          = 40");
    println!("  repeat_penalty = 1.1");
    println!("  use_system_prompt = false  # send only the project's own instructions");
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
        }
    }


    #[test]
    fn the_theme_belongs_to_the_screen_rather_than_to_a_model() {
        let config = config_with("theme = \"dark\"");
        assert_eq!(config.file.theme.as_deref(), Some("dark"));

        // And per model it is refused outright rather than accepted and ignored: the
        // screen does not change when the model does, so a key that looks like it works
        // there would only ever be a surprise.
        let per_model = "[models.\"a\"]\ntheme = \"dark\"";
        assert!(toml::from_str::<FileConfig>(per_model).is_err());
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
