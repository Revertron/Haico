//! The two `mcp.json` files: reading them, merging them, and saying what each entry is.
//!
//! The format is Claude Code's (`{"mcpServers": {...}}`), so a server set up for one tool
//! copies into the other unchanged. haico's own keys (`autoAllow`, `readOnly`, `tools`,
//! `disabledTools`, the timeouts) sit beside the standard ones in the same entry.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// How long a server gets to start and answer `initialize` and `tools/list`. `uvx` and `npx`
/// download the server on first use, which is what the margin is for.
pub(crate) const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Which of the two files an entry came from. A project entry needs the user's approval
/// before it runs; a global one is the user's own and does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scope {
    Global,
    Project,
}

impl Scope {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Scope::Global => "global",
            Scope::Project => "project",
        }
    }
}

/// `autoAllow`: either the whole server or the named tools.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub(crate) enum AutoAllow {
    All(bool),
    Tools(Vec<String>),
}

impl AutoAllow {
    pub(crate) fn covers(&self, tool: &str) -> bool {
        match self {
            AutoAllow::All(all) => *all,
            AutoAllow::Tools(tools) => tools.iter().any(|t| t == tool),
        }
    }
}

/// One entry under `mcpServers`, as written. Unknown keys are an error: a misspelled
/// `autoAllow` silently ignored is a server that keeps asking with nobody knowing why, and a
/// misspelled `disabledTools` is a tool offered that was meant to be hidden.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawEntry {
    #[serde(rename = "type")]
    kind: Option<String>,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    cwd: Option<String>,
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    disabled: bool,
    tools: Option<Vec<String>>,
    #[serde(default)]
    disabled_tools: Vec<String>,
    auto_allow: Option<AutoAllow>,
    #[serde(default)]
    read_only: Vec<String>,
    startup_timeout: Option<u64>,
    timeout: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawFile {
    #[serde(default)]
    mcp_servers: BTreeMap<String, Value>,
}

/// How to reach a server.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Transport {
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<PathBuf>,
    },
    Http {
        url: String,
        headers: BTreeMap<String, String>,
    },
}

/// One server, read, expanded and ready to start.
#[derive(Debug, Clone)]
pub(crate) struct ServerConfig {
    pub(crate) name: String,
    pub(crate) scope: Scope,
    pub(crate) transport: Transport,
    pub(crate) disabled: bool,
    pub(crate) tools: Option<Vec<String>>,
    pub(crate) disabled_tools: Vec<String>,
    pub(crate) auto_allow: Option<AutoAllow>,
    pub(crate) read_only: Vec<String>,
    pub(crate) startup_timeout: Duration,
    pub(crate) timeout: Option<Duration>,
    /// What the user approves a project entry under: the entry as written, before
    /// expansion. Expanded values can hold secrets from the environment, and a changed
    /// environment variable is not a changed file.
    pub(crate) fingerprint: String,
}

impl ServerConfig {
    /// Whether this tool is offered at all, by the entry's `tools` and `disabledTools`.
    pub(crate) fn offers(&self, tool: &str) -> bool {
        self.tools.as_ref().is_none_or(|only| only.iter().any(|t| t == tool))
            && !self.disabled_tools.iter().any(|t| t == tool)
    }

    pub(crate) fn auto_allows(&self, tool: &str) -> bool {
        self.auto_allow.as_ref().is_some_and(|a| a.covers(tool))
    }

    /// How the entry is shown in the approval prompt and in `/mcp`: what will run.
    pub(crate) fn describe(&self) -> String {
        match &self.transport {
            Transport::Stdio { command, args, .. } => {
                std::iter::once(command.as_str()).chain(args.iter().map(String::as_str)).collect::<Vec<_>>().join(" ")
            }
            Transport::Http { url, .. } => url.clone(),
        }
    }
}

/// Everything read from both files: the servers, and what went wrong reading them.
#[derive(Debug, Default)]
pub(crate) struct Loaded {
    pub(crate) servers: Vec<ServerConfig>,
    /// One line per problem: a file that does not parse, an entry with an unknown key, a
    /// variable that is not set. Said once at startup; the rest of the file still loads.
    pub(crate) errors: Vec<String>,
}

/// The global file.
pub(crate) fn global_path() -> Option<PathBuf> {
    crate::store::path("mcp.json")
}

/// The project file, under the directory haico runs in.
pub(crate) fn project_path(cwd: &Path) -> PathBuf {
    cwd.join(".haico").join("mcp.json")
}

/// Read both files and merge them: a project entry replaces a global one of the same name
/// whole, so `"disabled": true` in the project turns a global server off there.
pub(crate) fn load(cwd: &Path) -> Loaded {
    let lookup = |name: &str| std::env::var(name).ok();
    let mut loaded = Loaded::default();
    let global = match global_path() {
        Some(path) => read_file(&path, Scope::Global, cwd, &lookup, &mut loaded.errors),
        None => Vec::new(),
    };
    let project = read_file(&project_path(cwd), Scope::Project, cwd, &lookup, &mut loaded.errors);
    loaded.servers = merge(global, project);
    loaded
}

pub(crate) fn merge(global: Vec<ServerConfig>, project: Vec<ServerConfig>) -> Vec<ServerConfig> {
    let mut merged: BTreeMap<String, ServerConfig> = BTreeMap::new();
    for server in global.into_iter().chain(project) {
        merged.insert(server.name.clone(), server);
    }
    merged.into_values().collect()
}

fn read_file(
    path: &Path,
    scope: Scope,
    cwd: &Path,
    lookup: &dyn Fn(&str) -> Option<String>,
    errors: &mut Vec<String>,
) -> Vec<ServerConfig> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            errors.push(format!("{}: {}", crate::util::display_path(path), e));
            return Vec::new();
        }
    };
    let shown = crate::util::display_path(path);
    match parse(&text, scope, cwd, lookup) {
        Ok((servers, problems)) => {
            errors.extend(problems.into_iter().map(|p| format!("{}: {}", shown, p)));
            servers
        }
        Err(e) => {
            errors.push(format!("{}: {}", shown, e));
            Vec::new()
        }
    }
}

/// Parse one file's text. `Err` when the file as a whole is unreadable; otherwise the
/// servers that parsed and a problem line for each entry that did not.
pub(crate) fn parse(
    text: &str,
    scope: Scope,
    cwd: &Path,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<(Vec<ServerConfig>, Vec<String>), String> {
    let file: RawFile = serde_json::from_str(text).map_err(|e| {
        // Nearly always a Windows path pasted as is: `"C:\Users\..."` is `\U`, which JSON
        // has no escape for. The parser's words do not say that.
        if e.to_string().contains("invalid escape") {
            format!("{} (a Windows path needs \\\\ or / between its parts)", e)
        } else {
            e.to_string()
        }
    })?;
    let mut servers = Vec::new();
    let mut problems = Vec::new();
    for (name, value) in file.mcp_servers {
        match entry(&name, &value, scope, cwd, lookup) {
            Ok(server) => servers.push(server),
            Err(problem) => problems.push(format!("server \"{}\": {}", name, problem)),
        }
    }
    Ok((servers, problems))
}

fn entry(
    name: &str,
    value: &Value,
    scope: Scope,
    cwd: &Path,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<ServerConfig, String> {
    let raw: RawEntry = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    let expand = |text: &str| expand_vars(text, lookup);

    let kind = raw.kind.as_deref().unwrap_or(if raw.url.is_some() { "http" } else { "stdio" });
    let transport = match kind {
        "stdio" => {
            let command = raw.command.as_deref().ok_or("no \"command\"")?;
            Transport::Stdio {
                command: expand(command)?,
                args: raw.args.iter().map(|a| expand(a)).collect::<Result<_, _>>()?,
                env: raw.env.iter().map(|(k, v)| Ok((k.clone(), expand(v)?))).collect::<Result<_, String>>()?,
                // Relative to the project for a project entry, which is what "." there means;
                // a global entry has no project of its own, so it is relative to wherever
                // haico was started, which is the same directory.
                cwd: raw.cwd.as_deref().map(|c| expand(c).map(|c| cwd.join(c))).transpose()?,
            }
        }
        "http" | "streamable-http" | "streamableHttp" => Transport::Http {
            url: expand(raw.url.as_deref().ok_or("no \"url\"")?)?,
            headers: raw.headers.iter().map(|(k, v)| Ok((k.clone(), expand(v)?))).collect::<Result<_, String>>()?,
        },
        // The old HTTP+SSE transport is deprecated in the spec and rmcp no longer has a
        // client for it; saying so beats a connection error about a protocol mismatch.
        "sse" => return Err("the \"sse\" transport is not supported; use \"http\" (streamable HTTP)".to_string()),
        other => return Err(format!("unknown \"type\": \"{}\"", other)),
    };

    let mut fingerprinted = value.clone();
    if let Some(map) = fingerprinted.as_object_mut() {
        // Turning a server off and on again is not a new server to trust.
        map.remove("disabled");
    }

    Ok(ServerConfig {
        name: name.to_string(),
        scope,
        transport,
        disabled: raw.disabled,
        tools: raw.tools,
        disabled_tools: raw.disabled_tools,
        auto_allow: raw.auto_allow,
        read_only: raw.read_only,
        startup_timeout: raw.startup_timeout.map(Duration::from_secs).unwrap_or(DEFAULT_STARTUP_TIMEOUT),
        timeout: raw.timeout.map(Duration::from_secs),
        fingerprint: format!("{:016x}", fnv1a(fingerprinted.to_string().as_bytes())),
    })
}

/// FNV-1a: the fingerprint is stored, so it has to come out the same on every build, which
/// `DefaultHasher` does not promise (see `session::path_digest`).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3))
}

/// `${VAR}` and `${VAR:-default}`. An unset variable with no default is an error rather than
/// an empty string: `Bearer ` sent as a token fails somewhere far from the cause.
pub(crate) fn expand_vars(text: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find('}') else {
            return Err(format!("unclosed \"${{\" in \"{}\"", text));
        };
        let inner = &after[..end];
        let (name, default) = match inner.split_once(":-") {
            Some((name, default)) => (name, Some(default)),
            None => (inner, None),
        };
        match lookup(name).filter(|v| !v.is_empty()).or_else(|| default.map(str::to_string)) {
            Some(value) => out.push_str(&value),
            None => return Err(format!("environment variable {} is not set", name)),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn env(name: &str) -> Option<String> {
        match name {
            "TOKEN" => Some("secret".to_string()),
            "EMPTY" => Some(String::new()),
            _ => None,
        }
    }

    fn parse_ok(value: Value, scope: Scope) -> Vec<ServerConfig> {
        let (servers, problems) = parse(&value.to_string(), scope, Path::new("/proj"), &env).unwrap();
        assert!(problems.is_empty(), "{problems:?}");
        servers
    }

    #[test]
    fn variables_expand_with_defaults_and_unset_ones_are_errors() {
        assert_eq!(expand_vars("Bearer ${TOKEN}", &env).unwrap(), "Bearer secret");
        assert_eq!(expand_vars("${NOPE:-x}-${TOKEN}", &env).unwrap(), "x-secret");
        // Set but empty counts as unset, so the default still applies.
        assert_eq!(expand_vars("${EMPTY:-d}", &env).unwrap(), "d");
        assert!(expand_vars("${NOPE}", &env).unwrap_err().contains("NOPE"));
        assert!(expand_vars("${TOKEN", &env).is_err());
        assert_eq!(expand_vars("no vars $HOME", &env).unwrap(), "no vars $HOME");
    }

    #[test]
    fn claude_code_entries_parse_as_stdio_and_http() {
        let servers = parse_ok(
            json!({"mcpServers": {
                "git": {"command": "uvx", "args": ["mcp-server-git"], "env": {"T": "${TOKEN}"}},
                "docs": {"type": "http", "url": "http://localhost/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}},
                "bare": {"url": "http://localhost/other"}
            }}),
            Scope::Global,
        );
        let by_name = |n: &str| servers.iter().find(|s| s.name == n).unwrap();
        assert_eq!(
            by_name("git").transport,
            Transport::Stdio {
                command: "uvx".into(),
                args: vec!["mcp-server-git".into()],
                env: [("T".to_string(), "secret".to_string())].into(),
                cwd: None
            }
        );
        let Transport::Http { headers, .. } = &by_name("docs").transport else { panic!() };
        assert_eq!(headers["Authorization"], "Bearer secret");
        // A url alone means HTTP, as in Claude Code.
        assert!(matches!(by_name("bare").transport, Transport::Http { .. }));
        assert_eq!(by_name("git").startup_timeout, DEFAULT_STARTUP_TIMEOUT);
        assert_eq!(by_name("git").timeout, None);
    }

    #[test]
    fn a_bad_entry_is_reported_and_the_rest_still_load() {
        let text = json!({"mcpServers": {
            "typo": {"command": "x", "autoAlow": true},
            "sse": {"type": "sse", "url": "http://x"},
            "unset": {"command": "${NOPE}"},
            "fine": {"command": "x"}
        }})
        .to_string();
        let (servers, problems) = parse(&text, Scope::Global, Path::new("/"), &env).unwrap();
        assert_eq!(servers.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["fine"]);
        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("autoAlow")));
        assert!(parse("{not json", Scope::Global, Path::new("/"), &env).is_err());
    }

    #[test]
    fn the_project_entry_replaces_the_global_one_whole() {
        let global = parse_ok(
            json!({"mcpServers": {"db": {"command": "g", "autoAllow": true}, "keep": {"command": "k"}}}),
            Scope::Global,
        );
        let project = parse_ok(json!({"mcpServers": {"db": {"command": "p", "disabled": true}}}), Scope::Project);
        let merged = merge(global, project);
        assert_eq!(merged.len(), 2);
        let db = merged.iter().find(|s| s.name == "db").unwrap();
        assert_eq!(db.scope, Scope::Project);
        assert!(db.disabled);
        assert!(db.auto_allow.is_none(), "nothing of the global entry is kept");
    }

    #[test]
    fn tool_filters_and_auto_allow() {
        let servers = parse_ok(
            json!({"mcpServers": {
                "a": {"command": "x", "tools": ["q", "r"], "disabledTools": ["r"], "autoAllow": ["q"]},
                "b": {"command": "x", "autoAllow": true},
                "c": {"command": "x", "autoAllow": false}
            }}),
            Scope::Global,
        );
        let a = &servers[0];
        assert!(a.offers("q") && !a.offers("r") && !a.offers("s"));
        assert!(a.auto_allows("q") && !a.auto_allows("r"));
        assert!(servers[1].auto_allows("anything"));
        assert!(!servers[2].auto_allows("anything"));
    }

    #[test]
    fn the_fingerprint_follows_the_entry_as_written_but_not_disabled() {
        let fp = |v: Value| parse_ok(json!({"mcpServers": {"s": v}}), Scope::Project)[0].fingerprint.clone();
        let base = fp(json!({"command": "x", "args": ["a"]}));
        assert_eq!(base, fp(json!({"command": "x", "args": ["a"], "disabled": false})));
        assert_ne!(base, fp(json!({"command": "x", "args": ["b"]})));
        assert_ne!(base, fp(json!({"command": "x", "args": ["a"], "autoAllow": true})));
        assert_ne!(base, fp(json!({"command": "x", "args": ["a"], "readOnly": ["q"]})));
        // Written with a variable, so a changed environment is not a changed entry.
        let with_var = fp(json!({"command": "x", "env": {"T": "${TOKEN}"}}));
        assert_eq!(with_var, fp(json!({"command": "x", "env": {"T": "${TOKEN}"}})));
    }

    #[test]
    fn a_relative_cwd_is_the_projects() {
        let servers = parse_ok(json!({"mcpServers": {"s": {"command": "x", "cwd": "tools"}}}), Scope::Project);
        let Transport::Stdio { cwd, .. } = &servers[0].transport else { panic!() };
        assert_eq!(cwd.as_deref(), Some(Path::new("/proj/tools")));
    }
}
