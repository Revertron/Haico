//! The two `hooks.json` files: reading them, and saying what each entry runs.
//!
//! The layout is Claude Code's (`{"hooks": {"<Event>": [{"matcher": …, "hooks": [{"type":
//! "command", …}]}]}}`), so a hook written for one tool copies into the other with only its
//! matcher changed — haico's tools have their own names (`run_command`, not `Bash`).
//! Unlike `mcp.json` the two files add up rather than replace each other: a project's hooks
//! run beside the user's own, never instead of them.

use crate::mcp::config::fnv1a;
use crate::mcp::Scope;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The points of a session a hook can run at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Event {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PermissionRequest,
    PostToolUse,
    PostToolUseFailure,
    Stop,
    Notification,
}

impl Event {
    pub(crate) const ALL: [Event; 8] = [
        Event::SessionStart,
        Event::UserPromptSubmit,
        Event::PreToolUse,
        Event::PermissionRequest,
        Event::PostToolUse,
        Event::PostToolUseFailure,
        Event::Stop,
        Event::Notification,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Event::SessionStart => "SessionStart",
            Event::UserPromptSubmit => "UserPromptSubmit",
            Event::PreToolUse => "PreToolUse",
            Event::PermissionRequest => "PermissionRequest",
            Event::PostToolUse => "PostToolUse",
            Event::PostToolUseFailure => "PostToolUseFailure",
            Event::Stop => "Stop",
            Event::Notification => "Notification",
        }
    }

    fn parse(name: &str) -> Option<Event> {
        Event::ALL.into_iter().find(|e| e.name() == name)
    }

    /// Whether the matcher means anything here. The rest run on every occurrence, and a
    /// matcher written on one is said to be ignored rather than silently dropped.
    fn takes_matcher(self) -> bool {
        !matches!(self, Event::UserPromptSubmit | Event::Stop)
    }
}

/// What a group's `matcher` selects: a tool name for the tool events, the source for
/// `SessionStart`, the notification type for `Notification`.
#[derive(Debug, Clone)]
pub(crate) enum Matcher {
    All,
    /// `a|b` or `a,b`: whole names, as Claude Code reads a matcher of plain characters.
    Names(Vec<String>),
    /// Anything else is a regular expression, unanchored.
    Regex(regex::Regex),
}

impl Matcher {
    fn parse(text: Option<&str>) -> Result<Matcher, String> {
        let text = text.map(str::trim).unwrap_or_default();
        if text.is_empty() || text == "*" {
            return Ok(Matcher::All);
        }
        let plain = text.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ' ' | ',' | '|'));
        if plain {
            let names = text.split(['|', ',']).map(str::trim).filter(|n| !n.is_empty()).map(str::to_string).collect();
            return Ok(Matcher::Names(names));
        }
        regex::Regex::new(text).map(Matcher::Regex).map_err(|e| format!("matcher \"{}\" is not a valid regex: {}", text, e))
    }

    pub(crate) fn matches(&self, subject: &str) -> bool {
        match self {
            Matcher::All => true,
            Matcher::Names(names) => names.iter().any(|n| n == subject),
            Matcher::Regex(re) => re.is_match(subject),
        }
    }
}

/// What a hook does when it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Action {
    /// A shell line (cmd.exe on Windows, `sh` elsewhere — what `run_command` uses), or with
    /// `args` the program itself, started with no shell in between.
    Command { command: String, args: Option<Vec<String>> },
    /// The event POSTed to `url` as JSON; the reply's body is read as a command's stdout.
    Http {
        url: String,
        /// As written: `$VAR` and `${VAR}` in a value are filled in when the request is made,
        /// from `allowed_env_vars` only.
        headers: BTreeMap<String, String>,
        allowed_env_vars: Vec<String>,
    },
}

/// One hook to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Handler {
    pub(crate) action: Action,
    /// None unless the entry sets one: a hook runs until it ends, and Esc stops it.
    pub(crate) timeout: Option<Duration>,
}

impl Handler {
    /// What the hook runs, as `/hooks` and the approval prompt show it. An HTTP hook's
    /// header *names* only: their values are often tokens.
    pub(crate) fn describe(&self) -> String {
        match &self.action {
            Action::Command { command, args: Some(args) } => std::iter::once(command.as_str())
                .chain(args.iter().map(String::as_str))
                .collect::<Vec<_>>()
                .join(" "),
            Action::Command { command, args: None } => command.clone(),
            Action::Http { url, headers, .. } if headers.is_empty() => format!("POST {}", url),
            Action::Http { url, headers, .. } => {
                format!("POST {} (headers: {})", url, headers.keys().cloned().collect::<Vec<_>>().join(", "))
            }
        }
    }
}

/// One handler under one event and matcher, and which file it came from.
#[derive(Debug, Clone)]
pub(crate) struct Rule {
    pub(crate) event: Event,
    pub(crate) matcher: Matcher,
    /// As written, for showing.
    pub(crate) matcher_text: Option<String>,
    pub(crate) handler: Handler,
    pub(crate) scope: Scope,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    #[serde(default)]
    hooks: BTreeMap<String, Vec<Value>>,
}

// Unknown keys are errors here as in mcp.json: a misspelled `matcher` silently ignored is a
// hook that runs on every tool, and Claude Code's `async`, `if` or `onFailure` copied over
// would otherwise run as something they are not.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGroup {
    matcher: Option<String>,
    hooks: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawHandler {
    #[serde(rename = "type")]
    kind: String,
    command: Option<String>,
    args: Option<Vec<String>>,
    url: Option<String>,
    headers: Option<BTreeMap<String, String>>,
    allowed_env_vars: Option<Vec<String>>,
    timeout: Option<u64>,
}

/// One file, read.
#[derive(Debug, Default)]
pub(crate) struct Parsed {
    pub(crate) rules: Vec<Rule>,
    /// What the user approves a project file under: the file as parsed, so reformatting it
    /// is not a new question and changing any command is.
    pub(crate) fingerprint: String,
}

/// Both files, read.
#[derive(Debug, Default)]
pub(crate) struct Loaded {
    pub(crate) global: Vec<Rule>,
    pub(crate) project: Parsed,
    /// One line per problem: a file that does not parse, a handler with an unknown key.
    /// The rest of the file still loads.
    pub(crate) errors: Vec<String>,
}

pub(crate) fn global_path() -> Option<PathBuf> {
    crate::store::path("hooks.json")
}

pub(crate) fn project_path(cwd: &Path) -> PathBuf {
    cwd.join(".haico").join("hooks.json")
}

pub(crate) fn load(cwd: &Path) -> Loaded {
    let mut loaded = Loaded::default();
    if let Some(path) = global_path() {
        loaded.global = read_file(&path, Scope::Global, &mut loaded.errors).rules;
    }
    loaded.project = read_file(&project_path(cwd), Scope::Project, &mut loaded.errors);
    loaded
}

fn read_file(path: &Path, scope: Scope, errors: &mut Vec<String>) -> Parsed {
    let shown = crate::util::display_path(path);
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Parsed::default(),
        Err(e) => {
            errors.push(format!("{}: {}", shown, e));
            return Parsed::default();
        }
    };
    match parse(&text, scope) {
        Ok((parsed, problems)) => {
            errors.extend(problems.into_iter().map(|p| format!("{}: {}", shown, p)));
            parsed
        }
        Err(e) => {
            errors.push(format!("{}: {}", shown, e));
            Parsed::default()
        }
    }
}

/// Parse one file's text. `Err` when the file as a whole is unreadable; otherwise the rules
/// that parsed and a problem line for each piece that did not.
pub(crate) fn parse(text: &str, scope: Scope) -> Result<(Parsed, Vec<String>), String> {
    let value: Value = serde_json::from_str(text).map_err(|e| {
        // Nearly always a Windows path pasted as is, as in mcp.json.
        if e.to_string().contains("invalid escape") {
            format!("{} (a Windows path needs \\\\ or / between its parts)", e)
        } else {
            e.to_string()
        }
    })?;
    let fingerprint = format!("{:016x}", fnv1a(value.to_string().as_bytes()));
    let file: RawFile = serde_json::from_value(value).map_err(|e| e.to_string())?;

    let mut rules = Vec::new();
    let mut problems = Vec::new();
    for (name, groups) in file.hooks {
        let Some(event) = Event::parse(&name) else {
            let known: Vec<&str> = Event::ALL.iter().map(|e| e.name()).collect();
            problems.push(format!("unknown event \"{}\" (haico has {})", name, known.join(", ")));
            continue;
        };
        for (i, group) in groups.into_iter().enumerate() {
            let at = format!("{}[{}]", name, i);
            let group: RawGroup = match serde_json::from_value(group) {
                Ok(group) => group,
                Err(e) => {
                    problems.push(format!("{}: {}", at, e));
                    continue;
                }
            };
            if group.matcher.as_deref().is_some_and(|m| !m.trim().is_empty() && m.trim() != "*") && !event.takes_matcher() {
                problems.push(format!("{}: {} takes no matcher; it runs on every occurrence", at, name));
            }
            let matcher = match Matcher::parse(group.matcher.as_deref()) {
                Ok(matcher) => matcher,
                Err(e) => {
                    problems.push(format!("{}: {}", at, e));
                    continue;
                }
            };
            for (j, raw) in group.hooks.into_iter().enumerate() {
                match handler(raw) {
                    Ok(handler) => rules.push(Rule {
                        event,
                        matcher: matcher.clone(),
                        matcher_text: group.matcher.clone().filter(|m| !m.trim().is_empty()),
                        handler,
                        scope,
                    }),
                    Err(e) => problems.push(format!("{}.hooks[{}]: {}", at, j, e)),
                }
            }
        }
    }
    Ok((Parsed { rules, fingerprint }, problems))
}

fn handler(raw: Value) -> Result<Handler, String> {
    let raw: RawHandler = serde_json::from_value(raw).map_err(|e| e.to_string())?;
    // A key of the other type is refused rather than ignored: a `url` on a command hook, or
    // a `command` on an HTTP one, is an entry that means something other than it says.
    let stray = |keys: &[(&str, bool)]| match keys.iter().find(|(_, present)| *present) {
        Some((key, _)) => Err(format!("\"{}\" does not belong to a \"{}\" hook", key, raw.kind)),
        None => Ok(()),
    };
    let action = match raw.kind.as_str() {
        "command" => {
            stray(&[("url", raw.url.is_some()), ("headers", raw.headers.is_some()), ("allowedEnvVars", raw.allowed_env_vars.is_some())])?;
            let command = raw.command.filter(|c| !c.trim().is_empty()).ok_or("no \"command\"")?;
            Action::Command { command, args: raw.args }
        }
        "http" => {
            stray(&[("command", raw.command.is_some()), ("args", raw.args.is_some())])?;
            let url = raw.url.ok_or("no \"url\"")?;
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return Err(format!("\"url\" must start with http:// or https://, got \"{}\"", url));
            }
            Action::Http {
                url,
                headers: raw.headers.unwrap_or_default(),
                allowed_env_vars: raw.allowed_env_vars.unwrap_or_default(),
            }
        }
        other => {
            return Err(format!("\"type\": \"{}\" is not supported; haico runs \"command\" and \"http\" hooks", other))
        }
    };
    Ok(Handler { action, timeout: raw.timeout.map(Duration::from_secs) })
}

/// `$VAR` and `${VAR}` in an HTTP hook's header value, filled in from the environment for the
/// variables the entry lists in `allowedEnvVars` and with nothing for the rest, as in Claude
/// Code. The list is what keeps a project's hooks.json from posting a secret it was never
/// meant to see to a URL of its choosing.
pub(crate) fn interpolate(text: &str, allowed: &[String], lookup: &dyn Fn(&str) -> Option<String>) -> String {
    let is_name = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('$') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        let (name, tail) = if let Some(braced) = after.strip_prefix('{') {
            match braced.find('}') {
                Some(end) => (&braced[..end], &braced[end + 1..]),
                // No closing brace: not a reference, kept as written.
                None => ("", after),
            }
        } else {
            let end = after.find(|c: char| !is_name(c)).unwrap_or(after.len());
            (&after[..end], &after[end..])
        };
        if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
            out.push('$');
            rest = after;
            continue;
        }
        if allowed.iter().any(|a| a == name) {
            out.push_str(&lookup(name).unwrap_or_default());
        }
        rest = tail;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse_ok(value: Value) -> Vec<Rule> {
        let (parsed, problems) = parse(&value.to_string(), Scope::Global).unwrap();
        assert!(problems.is_empty(), "{problems:?}");
        parsed.rules
    }

    #[test]
    fn claude_code_layout_parses() {
        let rules = parse_ok(json!({"hooks": {
            "PreToolUse": [{"matcher": "run_command", "hooks": [
                {"type": "command", "command": "check.sh", "timeout": 30},
                {"type": "command", "command": "node", "args": ["a.js", "--fix"]}
            ]}],
            "Stop": [{"hooks": [{"type": "command", "command": "tests.sh"}]}]
        }}));
        assert_eq!(rules.len(), 3);
        let pre: Vec<&Rule> = rules.iter().filter(|r| r.event == Event::PreToolUse).collect();
        assert_eq!(pre[0].handler.timeout, Some(Duration::from_secs(30)));
        assert_eq!(pre[1].handler.describe(), "node a.js --fix");
        assert!(rules.iter().any(|r| r.event == Event::Stop && r.handler.timeout.is_none()));
    }

    #[test]
    fn http_hooks_parse_and_mixed_keys_are_refused() {
        let rules = parse_ok(json!({"hooks": {"PreToolUse": [{"hooks": [
            {"type": "http", "url": "http://localhost:8080/pre", "timeout": 5,
             "headers": {"Authorization": "Bearer $TOKEN"}, "allowedEnvVars": ["TOKEN"]}
        ]}]}}));
        assert_eq!(
            rules[0].handler.action,
            Action::Http {
                url: "http://localhost:8080/pre".into(),
                headers: [("Authorization".to_string(), "Bearer $TOKEN".to_string())].into(),
                allowed_env_vars: vec!["TOKEN".into()],
            }
        );
        // The token's name is shown, never its value.
        assert_eq!(rules[0].handler.describe(), "POST http://localhost:8080/pre (headers: Authorization)");

        let text = json!({"hooks": {"Stop": [{"hooks": [
            {"type": "http", "url": "http://x", "command": "c"},
            {"type": "http"},
            {"type": "http", "url": "localhost:8080"},
            {"type": "command", "command": "c", "url": "http://x"},
            {"type": "prompt", "prompt": "is it done?"}
        ]}]}})
        .to_string();
        let (parsed, problems) = parse(&text, Scope::Global).unwrap();
        assert!(parsed.rules.is_empty());
        assert_eq!(problems.len(), 5, "{problems:?}");
    }

    #[test]
    fn only_allowed_variables_reach_a_header() {
        let env = |name: &str| match name {
            "TOKEN" => Some("secret".to_string()),
            "HOME" => Some("/home/me".to_string()),
            _ => None,
        };
        let allowed = vec!["TOKEN".to_string(), "UNSET".to_string()];
        assert_eq!(interpolate("Bearer $TOKEN", &allowed, &env), "Bearer secret");
        assert_eq!(interpolate("Bearer ${TOKEN}x", &allowed, &env), "Bearer secretx");
        // Set, but not listed: nothing. Listed, but unset: nothing.
        assert_eq!(interpolate("[$HOME][${UNSET}]", &allowed, &env), "[][]");
        // Not references at all.
        assert_eq!(interpolate("$5 and $ and ${open", &allowed, &env), "$5 and $ and ${open");
    }

    #[test]
    fn matchers_are_names_or_regexes() {
        let m = |t: &str| Matcher::parse(Some(t)).unwrap();
        assert!(m("*").matches("anything"));
        assert!(m("").matches("anything"));
        assert!(m("edit|write_file").matches("write_file"));
        assert!(m("edit, write_file").matches("edit"));
        // Whole names: `edit` does not match `multi_edit`.
        assert!(!m("edit").matches("multi_edit"));
        // A regex is unanchored.
        assert!(m("mcp__db__.*").matches("mcp__db__query"));
        assert!(m("edit$").matches("multi_edit"));
        assert!(Matcher::parse(Some("(")).is_err());
    }

    #[test]
    fn bad_pieces_are_reported_and_the_rest_still_load() {
        let text = json!({"hooks": {
            "PreToolUse": [
                {"matcher": "x", "hooks": [{"type": "http", "url": "http://x", "command": "c"}]},
                {"matcher": "x", "hooks": [{"type": "command", "command": "c", "async": true}]},
                {"matcher": "x", "hooks": [{"type": "command", "command": "fine"}]}
            ],
            "SubagentStop": [{"hooks": [{"type": "command", "command": "c"}]}],
            "Stop": [{"matcher": "x", "hooks": [{"type": "command", "command": "s"}]}]
        }})
        .to_string();
        let (parsed, problems) = parse(&text, Scope::Global).unwrap();
        assert_eq!(parsed.rules.iter().map(|r| r.handler.describe()).collect::<Vec<_>>(), ["fine", "s"]);
        assert_eq!(problems.len(), 4, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("SubagentStop")));
        assert!(problems.iter().any(|p| p.contains("async")));
        assert!(problems.iter().any(|p| p.contains("takes no matcher")));
        assert!(parse("{not json", Scope::Global).is_err());
        assert!(parse(r#"{"hook": {}}"#, Scope::Global).is_err());
    }

    #[test]
    fn the_fingerprint_ignores_formatting_but_not_commands() {
        let fp = |text: &str| parse(text, Scope::Project).unwrap().0.fingerprint;
        let base = fp(r#"{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"a"}]}]}}"#);
        assert_eq!(base, fp("{ \"hooks\" : { \"Stop\" : [ { \"hooks\" : [ { \"type\" : \"command\", \"command\" : \"a\" } ] } ] } }"));
        assert_ne!(base, fp(r#"{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"b"}]}]}}"#));
    }
}
