//! Hooks: the user's own commands, run by haico at fixed points of a session — before a tool
//! call, after one, when a prompt is sent, when the model wants to stop.
//!
//! Instructions ask the model; a hook does not. "Format every file after an edit" written in
//! HAICO.md is followed when the model remembers it, and a small local model often does not.
//! Written as a `PostToolUse` hook it happens every time, because haico runs it, not the model.
//!
//! **The protocol is Claude Code's.** A hook gets the event as JSON on stdin and answers with
//! its exit code: 0 is fine (and a JSON object on stdout says more), 2 blocks with stderr as
//! the reason, anything else is an error that is shown and changes nothing. Scripts carry
//! over unchanged apart from tool names in their matchers. An `http` hook is the same
//! conversation over a POST: the event is the body, and the reply's body is read as stdout.
//!
//! **Nothing a hook says is cut.** Its context reaches the model whole, the same rule as
//! skill descriptions and research answers, and a hook runs for as long as it runs unless its
//! entry sets a `timeout` — Esc stops it like any other wait in a turn.
//!
//! The core never prints, here as elsewhere: what happened goes out as `Notice`s.

use crate::mcp::trust::Trust;
use crate::mcp::Scope;
use crate::turn::{interruptible, EventSink, Notice, TurnLink, Waiting};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(crate) mod config;

pub(crate) use config::{Action, Event, Handler, Rule};

/// The store file approvals of project hooks are kept in, beside `mcp-approvals.json`.
const APPROVALS: &str = "hook-approvals.json";
/// The one "server" name hooks are approved under: the project's file is approved whole.
const APPROVAL_KEY: &str = "hooks";
/// Substituted in a command and its args, and set in the hook's environment.
pub(crate) const PROJECT_DIR_VAR: &str = "HAICO_PROJECT_DIR";

/// The hooks that may run: every global one, and the project's once approved.
#[derive(Clone, Default)]
pub(crate) struct Hooks {
    rules: Vec<Rule>,
    project_dir: PathBuf,
}

impl Hooks {
    #[cfg(test)]
    pub(crate) fn from_rules(rules: Vec<Rule>, project_dir: &Path) -> Self {
        Self { rules, project_dir: project_dir.to_path_buf() }
    }

    pub(crate) fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Whether anything is configured for `event` at all — the cheap question asked before
    /// building an input nobody will read.
    pub(crate) fn has(&self, event: Event) -> bool {
        self.rules.iter().any(|r| r.event == event)
    }

    /// The handlers for this occurrence, in file order, each once: the same command written
    /// in both files runs once, as in Claude Code.
    fn matching(&self, event: Event, subject: Option<&str>) -> Vec<Handler> {
        let mut handlers: Vec<Handler> = Vec::new();
        for rule in self.rules.iter().filter(|r| r.event == event) {
            if subject.is_some_and(|s| !rule.matcher.matches(s)) {
                continue;
            }
            if !handlers.contains(&rule.handler) {
                handlers.push(rule.handler.clone());
            }
        }
        handlers
    }

    /// Run the `Notification` hooks and do not wait for them. A notification is a side
    /// effect — a sound, a toast — and the prompt it announces must not wait on it.
    pub(crate) fn notify(&self, input: Value, notification_type: &str) {
        let handlers = self.matching(Event::Notification, Some(notification_type));
        if handlers.is_empty() {
            return;
        }
        let dir = self.project_dir.clone();
        tokio::spawn(async move {
            for (handler, result) in run_all(handlers, input, dir).await {
                match result {
                    Ok(output) if output.code == Some(0) => {}
                    Ok(output) => crate::debug::log("Notification hook", &failure_detail(&handler, &output)),
                    Err(e) => crate::debug::log("Notification hook", &format!("{}: {}", handler.describe(), e)),
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Loading and approval
// ---------------------------------------------------------------------------

/// Both files read, and what is needed to put them in force: the frontend asks about the
/// project's hooks when they are new or changed (`unapproved`, `approve`), then `finish`es.
///
/// The project's file is approved whole rather than hook by hook: hooks work together (a
/// `PreToolUse` check and the `PostToolUse` formatter beside it), and half of a set running
/// is a configuration nobody wrote.
pub(crate) struct Startup {
    cwd: PathBuf,
    loaded: config::Loaded,
    trust: Trust,
    approved: bool,
}

impl Startup {
    pub(crate) fn load(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
            loaded: config::load(cwd),
            trust: Trust::from_store_file(APPROVALS),
            approved: false,
        }
    }

    /// The project's hooks, when there are any and nobody has approved them as written now.
    pub(crate) fn unapproved(&self) -> Option<&[Rule]> {
        let project = &self.loaded.project;
        let approved =
            self.approved || self.trust.is_approved(&self.cwd, APPROVAL_KEY, &project.fingerprint);
        (!project.rules.is_empty() && !approved).then_some(project.rules.as_slice())
    }

    /// Approve for this run, and remember it. `Err` says only that remembering failed.
    pub(crate) fn approve(&mut self) -> anyhow::Result<()> {
        self.approved = true;
        self.trust.approve(&self.cwd, APPROVAL_KEY, &self.loaded.project.fingerprint)
    }

    /// The hooks in force, and the lines worth telling the user: problems in the files, and
    /// project hooks left off.
    pub(crate) fn finish(self) -> (Hooks, Vec<String>) {
        let skipped = self.unapproved().map(<[Rule]>::len);
        let Startup { cwd, loaded, .. } = self;
        let mut warnings: Vec<String> = loaded.errors.into_iter().map(|e| format!("hooks.json: {}", e)).collect();
        let mut rules = loaded.global;
        match skipped {
            Some(count) => warnings.push(format!(
                "{} project hook{} in {} not approved, not run. `/hooks reload` in the terminal asks again.",
                count,
                if count == 1 { "" } else { "s" },
                crate::util::display_path(&config::project_path(&cwd)),
            )),
            None => rules.extend(loaded.project.rules),
        }
        (Hooks { rules, project_dir: cwd }, warnings)
    }
}

/// The approval question for a project's hooks, as lines: every command, and when it runs.
pub(crate) fn approval_lines(rules: &[Rule]) -> Vec<String> {
    let mut lines = vec!["The project's .haico/hooks.json asks to run these hooks:".to_string()];
    lines.extend(rules.iter().map(|rule| format!("  {}", describe_rule(rule))));
    lines
}

/// One rule on one line: when, and what.
pub(crate) fn describe_rule(rule: &Rule) -> String {
    let when = match &rule.matcher_text {
        Some(matcher) => format!("{} [{}]", rule.event.name(), matcher),
        None => rule.event.name().to_string(),
    };
    format!("{}: {}", when, rule.handler.describe())
}

/// Where a rule came from, for `/hooks`.
pub(crate) fn scope_label(rule: &Rule) -> &'static str {
    match rule.scope {
        Scope::Global => "global",
        Scope::Project => "project",
    }
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// What one hook process did.
#[derive(Debug)]
pub(crate) struct Output {
    /// `None` when it was ended by a signal.
    pub(crate) code: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

/// Run every handler at once, as Claude Code does, and give the results back in the order the
/// handlers were written, so what they decide merges the same way every time.
///
/// Dropping the future (Esc) drops the set, which aborts every task, and each task's
/// `KillTreeOnDrop` takes its process tree with it.
async fn run_all(handlers: Vec<Handler>, input: Value, dir: PathBuf) -> Vec<(Handler, Result<Output, String>)> {
    let input = input.to_string();
    let mut set = tokio::task::JoinSet::new();
    for (i, handler) in handlers.iter().cloned().enumerate() {
        let (input, dir) = (input.clone(), dir.clone());
        set.spawn(async move { (i, run_handler(&handler, &input, &dir).await) });
    }
    let mut results: Vec<(usize, Result<Output, String>)> = Vec::with_capacity(handlers.len());
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(result) => results.push(result),
            // A panicked task; nothing is known about what it decided.
            Err(e) => crate::debug::log("hook task", &e.to_string()),
        }
    }
    results.sort_by_key(|(i, _)| *i);
    results.into_iter().map(|(i, result)| (handlers[i].clone(), result)).collect()
}

async fn run_handler(handler: &Handler, input: &str, dir: &Path) -> Result<Output, String> {
    let work = async {
        match &handler.action {
            Action::Command { command, args } => run_command(command, args.as_deref(), input, dir).await,
            Action::Http { url, headers, allowed_env_vars } => post(url, headers, allowed_env_vars, input).await,
        }
    };
    match handler.timeout {
        Some(limit) => tokio::time::timeout(limit, work)
            .await
            .map_err(|_| format!("did not finish within {} s", limit.as_secs()))?,
        None => work.await,
    }
}

/// An HTTP hook, answered the way Claude Code reads one: a 2xx with an empty body is a hook
/// with nothing to say, a 2xx with a JSON object is read exactly as a command's stdout, and
/// anything else is an error that blocks nothing. There is no exit code 2 over HTTP — a
/// block is the JSON's to say.
async fn post(
    url: &str,
    headers: &BTreeMap<String, String>,
    allowed: &[String],
    input: &str,
) -> Result<Output, String> {
    let lookup = |name: &str| std::env::var(name).ok();
    let mut request = reqwest::Client::new()
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(input.to_string());
    for (name, value) in headers {
        request = request.header(name, config::interpolate(value, allowed, &lookup));
    }
    let response = request.send().await.map_err(|e| format!("POST {}: {}", url, e))?;
    let status = response.status();
    let body = response.text().await.map_err(|e| format!("POST {}: reading the reply: {}", url, e))?;
    if !status.is_success() {
        return Err(match body.trim() {
            "" => format!("POST {}: HTTP {}", url, status),
            body => format!("POST {}: HTTP {}: {}", url, status, body),
        });
    }
    let body = body.trim();
    if !body.is_empty() && !matches!(serde_json::from_str::<Value>(body), Ok(Value::Object(_))) {
        return Err(format!("POST {}: the reply is not a JSON object: {}", url, body));
    }
    Ok(Output { code: Some(0), stdout: body.to_string(), stderr: String::new() })
}

async fn run_command(command_line: &str, args: Option<&[String]>, input: &str, dir: &Path) -> Result<Output, String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let shown_dir = dir.to_string_lossy();
    let expand = |text: &str| text.replace(&format!("${{{}}}", PROJECT_DIR_VAR), &shown_dir);
    let mut command = match args {
        Some(args) => {
            let mut command = tokio::process::Command::new(expand(command_line));
            command.args(args.iter().map(|a| expand(a)));
            command
        }
        None => crate::tools::shell_command(&expand(command_line)),
    };
    command
        .env(PROJECT_DIR_VAR, dir)
        // A pipe, not the console: cmd.exe rewrites the console's input mode only when its
        // stdin *is* the console (see `shell_builder`), and a pipe is what carries the event.
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    // No console of its own either, as for an MCP server: one sharing haico's would be a
    // second reader of the keyboard.
    #[cfg(windows)]
    command.creation_flags(windows::Win32::System::Threading::CREATE_NO_WINDOW.0);
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command.spawn().map_err(|e| format!("could not start: {}", e))?;
    let mut tree = crate::tools::KillTreeOnDrop { pid: child.id() };
    let (stdin, stdout, stderr) = (child.stdin.take(), child.stdout.take(), child.stderr.take());

    // Written while the output is read, not before: a hook that prints before it has read
    // all of a large input would otherwise wait on a full pipe while haico waits on it.
    let work = async {
        let write = async {
            if let Some(mut stdin) = stdin {
                // A hook that never reads its input closes the pipe early; that is its right.
                let _ = stdin.write_all(input.as_bytes()).await;
            }
        };
        let read = |pipe: Option<tokio::process::ChildStdout>| async move {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes).await;
            }
            bytes
        };
        let read_err = async {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = stderr {
                let _ = pipe.read_to_end(&mut bytes).await;
            }
            bytes
        };
        let ((), out, err, status) = tokio::join!(write, read(stdout), read_err, child.wait());
        (out, err, status)
    };
    // A timeout drops this future, and `tree` with it.
    let (out, err, status) = work.await;
    tree.disarm();
    let status = status.map_err(|e| e.to_string())?;
    Ok(Output {
        code: status.code(),
        stdout: String::from_utf8_lossy(&out).into_owned(),
        stderr: String::from_utf8_lossy(&err).into_owned(),
    })
}

fn failure_detail(handler: &Handler, output: &Output) -> String {
    let code = output.code.map_or("ended by a signal".to_string(), |c| format!("exit code {}", c));
    match output.stderr.trim() {
        "" => format!("{}: {}", handler.describe(), code),
        stderr => format!("{}: {}: {}", handler.describe(), code, stderr),
    }
}

// ---------------------------------------------------------------------------
// What the hooks said
// ---------------------------------------------------------------------------

/// A permission verdict from `PreToolUse` or `PermissionRequest`, with the hook's reason.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Decision {
    Allow(String),
    Ask(String),
    Deny(String),
}

impl Decision {
    fn weight(&self) -> u8 {
        match self {
            Decision::Allow(_) => 0,
            Decision::Ask(_) => 1,
            Decision::Deny(_) => 2,
        }
    }

    fn reason(&self) -> &str {
        match self {
            Decision::Allow(r) | Decision::Ask(r) | Decision::Deny(r) => r,
        }
    }
}

/// Everything the hooks of one occurrence said, merged. Which fields mean anything depends
/// on the event; the caller reads the ones that do.
#[derive(Debug, Default)]
pub(crate) struct Outcome {
    /// The strictest verdict: a deny beats an ask beats an allow.
    pub(crate) decision: Option<Decision>,
    /// Exit 2 or `"decision": "block"`: the prompt is not sent, the model does not stop, the
    /// tool's result gets the reason as feedback — the event says which.
    pub(crate) block: Option<String>,
    /// Arguments to run the call with instead. The last hook to give some wins.
    pub(crate) updated_input: Option<Value>,
    /// A result to give the model instead of the tool's.
    pub(crate) updated_output: Option<String>,
    /// For the model.
    pub(crate) context: Vec<String>,
    /// For the user (`systemMessage`, and stderr where the event shows it).
    pub(crate) messages: Vec<String>,
    /// `"continue": false`: the turn ends, with this reason.
    pub(crate) stop: Option<String>,
    /// Hooks that could not run or failed without blocking, as one line each.
    pub(crate) failures: Vec<(String, String)>,
}

impl Outcome {
    fn decide(&mut self, decision: Decision) {
        match &mut self.decision {
            Some(current) if current.weight() > decision.weight() => {}
            Some(current) if current.weight() == decision.weight() => {
                let joined = join_reasons(current.reason(), decision.reason());
                *current = match decision {
                    Decision::Allow(_) => Decision::Allow(joined),
                    Decision::Ask(_) => Decision::Ask(joined),
                    Decision::Deny(_) => Decision::Deny(joined),
                };
            }
            _ => self.decision = Some(decision),
        }
    }

    fn block(&mut self, reason: String) {
        self.block = Some(match self.block.take() {
            Some(before) => join_reasons(&before, &reason),
            None => reason,
        });
    }
}

fn join_reasons(a: &str, b: &str) -> String {
    match (a.is_empty(), b.is_empty()) {
        (true, _) => b.to_string(),
        (_, true) => a.to_string(),
        _ => format!("{}\n{}", a, b),
    }
}

pub(crate) fn interpret(event: Event, ran: Vec<(Handler, Result<Output, String>)>) -> Outcome {
    let mut outcome = Outcome::default();
    for (handler, result) in ran {
        let output = match result {
            Ok(output) => output,
            Err(e) => {
                outcome.failures.push((handler.describe(), e));
                continue;
            }
        };
        match output.code {
            Some(0) => {
                let stdout = output.stdout.trim();
                match serde_json::from_str::<Value>(stdout) {
                    Ok(Value::Object(fields)) => apply_json(event, &fields, &mut outcome),
                    // Plain text is context where Claude Code makes it context, and goes
                    // unread everywhere else.
                    _ if !stdout.is_empty() && matches!(event, Event::SessionStart | Event::UserPromptSubmit) => {
                        outcome.context.push(stdout.to_string())
                    }
                    _ => {}
                }
            }
            Some(2) => {
                let reason = output.stderr.trim().to_string();
                match event {
                    Event::PreToolUse => outcome.decide(Decision::Deny(reason)),
                    Event::UserPromptSubmit | Event::Stop | Event::PostToolUse | Event::PostToolUseFailure => {
                        outcome.block(reason)
                    }
                    Event::SessionStart if !reason.is_empty() => outcome.messages.push(reason),
                    // Claude Code ignores exit 2 here: a permission verdict is its JSON's to give.
                    Event::SessionStart | Event::PermissionRequest | Event::Notification => {}
                }
            }
            _ => outcome.failures.push((handler.describe(), failure_detail(&handler, &output))),
        }
    }
    outcome
}

fn apply_json(event: Event, fields: &Map<String, Value>, outcome: &mut Outcome) {
    let text = |value: Option<&Value>| value.and_then(Value::as_str).unwrap_or_default().to_string();

    if fields.get("continue") == Some(&Value::Bool(false)) {
        outcome.stop = Some(text(fields.get("stopReason")));
    }
    if let Some(message) = fields.get("systemMessage").and_then(Value::as_str).filter(|m| !m.is_empty()) {
        outcome.messages.push(message.to_string());
    }
    let reason = text(fields.get("reason"));
    match (event, fields.get("decision").and_then(Value::as_str)) {
        // The older spelling of a PreToolUse verdict, which Claude Code still reads.
        (Event::PreToolUse, Some("block")) => outcome.decide(Decision::Deny(reason)),
        (Event::PreToolUse, Some("approve")) => outcome.decide(Decision::Allow(reason)),
        (_, Some("block")) => outcome.block(reason),
        _ => {}
    }

    let Some(specific) = fields.get("hookSpecificOutput").and_then(Value::as_object) else { return };
    if let Some(context) = specific.get("additionalContext").and_then(Value::as_str).filter(|c| !c.is_empty()) {
        outcome.context.push(context.to_string());
    }
    match event {
        Event::PreToolUse => {
            let reason = text(specific.get("permissionDecisionReason"));
            match specific.get("permissionDecision").and_then(Value::as_str) {
                Some("allow") => outcome.decide(Decision::Allow(reason)),
                Some("ask") => outcome.decide(Decision::Ask(reason)),
                Some("deny") => outcome.decide(Decision::Deny(reason)),
                // "defer" and anything else: no verdict, the usual gates decide.
                _ => {}
            }
            if let Some(input) = specific.get("updatedInput").filter(|v| v.is_object()) {
                outcome.updated_input = Some(input.clone());
            }
        }
        Event::PermissionRequest => {
            let Some(decision) = specific.get("decision").and_then(Value::as_object) else { return };
            let message = text(decision.get("message"));
            match decision.get("behavior").and_then(Value::as_str) {
                Some("allow") => outcome.decide(Decision::Allow(message)),
                Some("deny") => outcome.decide(Decision::Deny(message)),
                _ => {}
            }
            if let Some(input) = decision.get("updatedInput").filter(|v| v.is_object()) {
                outcome.updated_input = Some(input.clone());
            }
        }
        Event::PostToolUse | Event::PostToolUseFailure => match specific.get("updatedToolOutput") {
            Some(Value::String(output)) => outcome.updated_output = Some(output.clone()),
            Some(Value::Null) | None => {}
            Some(other) => outcome.updated_output = Some(other.to_string()),
        },
        _ => {}
    }
}

/// Run the hooks for one occurrence inside a turn and report what went wrong with them.
/// `None` means the user pressed Esc while they ran.
///
/// What the hooks *decided* is the caller's to act on and announce: only it knows what a
/// block means for the event at hand.
pub(crate) async fn fire(
    hooks: &Hooks,
    tx: &EventSink,
    link: &TurnLink,
    event: Event,
    subject: Option<&str>,
    input: Value,
) -> Option<Outcome> {
    let handlers = hooks.matching(event, subject);
    if handlers.is_empty() {
        return Some(Outcome::default());
    }
    let ran = interruptible(tx, link, Waiting::Hook(event.name()), run_all(handlers, input, hooks.project_dir.clone())).await?;
    let outcome = interpret(event, ran);
    for (command, detail) in &outcome.failures {
        let _ = tx.send(Notice::HookFailed { event: event.name(), command: command.clone(), detail: detail.clone() }.into());
    }
    for message in &outcome.messages {
        let _ = tx.send(Notice::HookMessage { event: event.name(), text: message.clone() }.into());
    }
    if let Some(reason) = &outcome.stop {
        let _ = tx.send(Notice::HookStopped { event: event.name(), reason: reason.clone() }.into());
    }
    Some(outcome)
}

/// Context from a hook as the model reads it: labeled with the event, so a line that
/// appears under a tool result is not mistaken for part of the result.
pub(crate) fn context_note(event: Event, context: &[String]) -> Option<String> {
    (!context.is_empty()).then(|| format!("[{} hook]\n{}", event.name(), context.join("\n\n")))
}

#[cfg(test)]
mod tests;
