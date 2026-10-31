//! haico as an MCP *client*: the servers in `mcp.json`, and their tools offered to the model
//! beside haico's own.
//!
//! Not to be confused with `tools/`, which is haico as an MCP *server* (`--server`). That
//! side is unchanged and does not pass these tools on: a haico host pointing at a haico
//! server would otherwise see its own MCP servers twice, through a hop with no permission
//! gate.
//!
//! **Servers start in the background.** A server launched through `uvx` or `npx` can take
//! many seconds on first use, and the prompt should not wait for it. Each server's tools
//! join the request between turns, never mid-turn — the agent takes a snapshot when a turn
//! starts — because a tool list that changes under a running turn changes the prefix the
//! LLM server has cached, and the model would meet tools appearing between two rounds of
//! one task.
//!
//! **The hub says what happened as notices**, queued here and taken by whichever frontend
//! is running, the way timers are: the core never prints.

use crate::tools::ToolOutput;
use rmcp::model::{CallToolRequestParams, Tool};
use rmcp::service::{NotificationContext, RunningService};
use rmcp::{ClientHandler, RoleClient, ServiceExt};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

pub(crate) mod config;
pub(crate) mod trust;

pub(crate) use config::{Scope, ServerConfig, Transport};

/// What every MCP tool's name starts with, so the agent can tell one from haico's own tools
/// without asking the hub.
pub(crate) const PREFIX: &str = "mcp__";

/// A text result longer than this is written to the scratch directory instead of the
/// conversation, and the model is given its beginning and the file's path. Nothing is cut:
/// the rest is one `read` away. A database dump in the history would be re-sent with every
/// request for the rest of the session.
pub(crate) const SPILL_THRESHOLD: usize = 64 * 1024;
/// How much of a spilled result the model is shown, so it can tell whether it needs the rest.
pub(crate) const SPILL_HEAD: usize = 8 * 1024;
/// The last lines a stdio server wrote to stderr, kept for `/mcp`: a server that fails to
/// start usually says why there, and nowhere else.
const STDERR_LINES: usize = 20;

/// A server's state, as `/mcp` shows it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum State {
    Loading,
    Ready,
    Failed(String),
    Disabled,
    /// A project server the user has not approved (or whose entry changed since).
    NotApproved,
}

/// Why a tool counts as read-only, which decides whether plan mode offers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadOnly {
    /// The server's `readOnlyHint`.
    Hint,
    /// The entry's `readOnly` list.
    Config,
}

/// One MCP tool as the agent sees it.
#[derive(Debug, Clone)]
pub(crate) struct McpTool {
    /// `mcp__<server>__<tool>`, the name the model calls.
    pub(crate) name: String,
    pub(crate) server: String,
    /// The tool's own name on its server.
    pub(crate) tool: String,
    /// The OpenAI-format entry for the request's tools array.
    pub(crate) schema: Value,
    pub(crate) read_only: Option<ReadOnly>,
    pub(crate) auto_allow: bool,
}

/// Something a frontend should tell the user.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Notice {
    /// Servers have started loading.
    Loading(Vec<String>),
    /// Every server that was loading has finished, one way or the other.
    Loaded { servers: usize, tools: usize },
    Failed { server: String, error: String },
    /// A problem in one of the files.
    Config(String),
    /// Project servers left off because nobody approved them.
    NotApproved(Vec<String>),
    /// A server changed its tool list; the change reaches the model at the next turn.
    ToolsChanged { server: String, tools: usize },
    /// A server that was working stopped answering.
    Lost { server: String },
}

impl Notice {
    pub(crate) fn text(&self) -> String {
        match self {
            Notice::Loading(names) => format!("Loading MCP servers: {}…", names.join(", ")),
            Notice::Loaded { servers, tools } => format!(
                "MCP servers loaded: {} server{}, {} tool{}",
                servers,
                if *servers == 1 { "" } else { "s" },
                tools,
                if *tools == 1 { "" } else { "s" }
            ),
            Notice::Failed { server, error } => format!("MCP server \"{}\" failed: {}", server, error),
            Notice::Config(problem) => format!("mcp.json: {}", problem),
            Notice::NotApproved(names) => format!(
                "Project MCP server{} not approved, not started: {}. `/mcp reconnect <name>` in the \
                 terminal asks again.",
                if names.len() == 1 { "" } else { "s" },
                names.join(", ")
            ),
            Notice::ToolsChanged { server, tools } => {
                format!("MCP server \"{}\" now has {} tool{}", server, tools, if *tools == 1 { "" } else { "s" })
            }
            Notice::Lost { server } => format!("MCP server \"{}\" stopped responding", server),
        }
    }

    /// Whether this is bad news, for the frontend's color.
    pub(crate) fn is_problem(&self) -> bool {
        matches!(self, Notice::Failed { .. } | Notice::Config(_) | Notice::NotApproved(_) | Notice::Lost { .. })
    }
}

/// A server as `/mcp` shows it.
pub(crate) struct Status {
    pub(crate) name: String,
    pub(crate) scope: Scope,
    pub(crate) state: State,
    pub(crate) target: String,
    pub(crate) tools: Vec<McpTool>,
    pub(crate) stderr: Vec<String>,
}

type Service = RunningService<RoleClient, Handler>;

struct Server {
    config: ServerConfig,
    state: State,
    tools: Vec<Tool>,
    service: Option<Arc<Service>>,
    stderr: Arc<Mutex<VecDeque<String>>>,
    /// Bumped by every (re)connect, so a slow connection that finishes after a newer one
    /// started cannot overwrite it.
    generation: u64,
}

#[derive(Default)]
struct Inner {
    servers: Mutex<BTreeMap<String, Server>>,
    notices: Mutex<Vec<Notice>>,
    /// Woken when a notice is queued. One permit is stored if nobody is waiting, so a notice
    /// queued while the frontend was busy still wakes the next wait.
    notify: tokio::sync::Notify,
    /// How many servers are still loading.
    loading: Mutex<usize>,
    /// Where a long result is written. The session's scratch directory, set by the agent.
    spill_dir: Mutex<Option<PathBuf>>,
    spilled: std::sync::atomic::AtomicUsize,
    /// Hub-wide rather than per server: `stop_all` empties the map, and a per-server count
    /// starting again from zero could match a connection still in flight from before.
    generations: std::sync::atomic::AtomicU64,
}

/// The servers, shared: the tool server holds one, and every clone is the same hub.
#[derive(Clone, Default)]
pub(crate) struct Hub {
    inner: Arc<Inner>,
}

impl Hub {
    /// Take on these servers and start every one that is enabled and approved. `approved`
    /// answers for project servers only; global ones are the user's own.
    pub(crate) fn start(&self, servers: Vec<ServerConfig>, approved: &dyn Fn(&ServerConfig) -> bool) {
        let mut starting = Vec::new();
        let mut unapproved = Vec::new();
        {
            let mut map = self.inner.servers.lock().unwrap();
            for config in servers {
                let state = if config.disabled {
                    State::Disabled
                } else if config.scope == Scope::Project && !approved(&config) {
                    unapproved.push(config.name.clone());
                    State::NotApproved
                } else {
                    starting.push(config.name.clone());
                    State::Loading
                };
                map.insert(
                    config.name.clone(),
                    Server { config, state, tools: Vec::new(), service: None, stderr: Arc::default(), generation: 0 },
                );
            }
        }
        if !unapproved.is_empty() {
            self.push(Notice::NotApproved(unapproved));
        }
        if starting.is_empty() {
            return;
        }
        *self.inner.loading.lock().unwrap() += starting.len();
        self.push(Notice::Loading(starting.clone()));
        for name in starting {
            self.spawn_connect(&name);
        }
    }

    /// Say a problem in the files, for `start`'s caller.
    pub(crate) fn report_config_errors(&self, errors: Vec<String>) {
        for error in errors {
            self.push(Notice::Config(error));
        }
    }

    /// Stop a server and start it again — also how an approved-later server gets going.
    pub(crate) fn reconnect(&self, name: &str) -> Result<(), String> {
        {
            let mut map = self.inner.servers.lock().unwrap();
            let server = map.get_mut(name).ok_or_else(|| format!("no MCP server named \"{}\"", name))?;
            if server.state == State::Disabled {
                return Err(format!("\"{}\" is disabled in mcp.json", name));
            }
            if let Some(service) = server.service.take() {
                service.cancellation_token().cancel();
            }
            server.state = State::Loading;
            server.tools.clear();
        }
        *self.inner.loading.lock().unwrap() += 1;
        self.push(Notice::Loading(vec![name.to_string()]));
        self.spawn_connect(name);
        Ok(())
    }

    /// Connect a server over an in-memory pipe, for tests: haico's own tool server on the
    /// other end stands in for an MCP server, and the protocol between them is the real one.
    #[cfg(test)]
    pub(crate) async fn attach(&self, config: ServerConfig, transport: tokio::io::DuplexStream) {
        let name = config.name.clone();
        let generation = self.inner.generations.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        self.inner.servers.lock().unwrap().insert(
            name.clone(),
            Server { config, state: State::Loading, tools: Vec::new(), service: None, stderr: Arc::default(), generation },
        );
        *self.inner.loading.lock().unwrap() += 1;
        let handler = Handler { hub: Arc::downgrade(&self.inner), server: name.clone() };
        let connected = serve_and_list(handler, transport).await;
        self.finish_connect(&name, generation, connected);
    }

    /// Stop every server and forget them, for `/cd`: the next directory brings its own.
    /// A connection still being made is abandoned; it finds its server gone and drops.
    pub(crate) fn stop_all(&self) {
        let servers = std::mem::take(&mut *self.inner.servers.lock().unwrap());
        for server in servers.into_values() {
            if let Some(service) = server.service {
                service.cancellation_token().cancel();
            }
        }
        *self.inner.loading.lock().unwrap() = 0;
        self.inner.notices.lock().unwrap().clear();
    }

    fn spawn_connect(&self, name: &str) {
        let (config, generation, stderr) = {
            let mut map = self.inner.servers.lock().unwrap();
            let Some(server) = map.get_mut(name) else { return };
            server.generation = self.inner.generations.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            server.stderr.lock().unwrap().clear();
            (server.config.clone(), server.generation, Arc::clone(&server.stderr))
        };
        let hub = self.clone();
        tokio::spawn(async move {
            let handler = Handler { hub: Arc::downgrade(&hub.inner), server: config.name.clone() };
            let connected = tokio::time::timeout(config.startup_timeout, connect(&config, handler, stderr)).await;
            let connected = match connected {
                Ok(result) => result,
                Err(_) => Err(format!("did not start within {} s", config.startup_timeout.as_secs())),
            };
            hub.finish_connect(&config.name, generation, connected);
        });
    }

    fn finish_connect(&self, name: &str, generation: u64, connected: Result<(Service, Vec<Tool>), String>) {
        let notice = {
            let mut map = self.inner.servers.lock().unwrap();
            let Some(server) = map.get_mut(name).filter(|s| s.generation == generation) else {
                // Superseded by a reconnect, which does its own counting.
                return;
            };
            match connected {
                Ok((service, tools)) => {
                    server.tools = tools;
                    server.service = Some(Arc::new(service));
                    server.state = State::Ready;
                    None
                }
                Err(error) => {
                    // The last thing the server said is usually the reason.
                    let said = server.stderr.lock().unwrap().back().cloned();
                    let error = match said {
                        Some(line) => format!("{} ({})", error, line),
                        None => error,
                    };
                    server.state = State::Failed(error.clone());
                    Some(Notice::Failed { server: name.to_string(), error })
                }
            }
        };
        if let Some(notice) = notice {
            self.push(notice);
        }
        let finished = {
            let mut loading = self.inner.loading.lock().unwrap();
            *loading = loading.saturating_sub(1);
            *loading == 0
        };
        if finished {
            let tools = self.tools();
            let servers = self.inner.servers.lock().unwrap().values().filter(|s| s.state == State::Ready).count();
            self.push(Notice::Loaded { servers, tools: tools.len() });
        } else {
            // Wakes the prompt so the status bar's count moves.
            self.inner.notify.notify_one();
        }
    }

    fn push(&self, notice: Notice) {
        crate::debug::log("mcp", &notice.text());
        self.inner.notices.lock().unwrap().push(notice);
        self.inner.notify.notify_one();
    }

    /// The notices queued since the last call.
    pub(crate) fn take_notices(&self) -> Vec<Notice> {
        std::mem::take(&mut *self.inner.notices.lock().unwrap())
    }

    /// Resolves when something has happened: a notice was queued or a server finished
    /// loading. A frontend waits on it beside the keyboard.
    pub(crate) async fn changed(&self) {
        self.inner.notify.notified().await;
    }

    /// `(still loading, configured to run)`, for the status bar; `None` when nothing is
    /// loading.
    pub(crate) fn loading(&self) -> Option<(usize, usize)> {
        let loading = *self.inner.loading.lock().unwrap();
        if loading == 0 {
            return None;
        }
        let total = self
            .inner
            .servers
            .lock()
            .unwrap()
            .values()
            .filter(|s| !matches!(s.state, State::Disabled | State::NotApproved))
            .count();
        Some((loading, total))
    }

    /// Wait until no server is loading, for a one-shot run that must not send its only
    /// request before the tools are there. Bounded by the servers' own startup timeouts.
    pub(crate) async fn wait_loaded(&self) {
        while self.loading().is_some() {
            // `changed` can also fire for a notice; the loop re-checks.
            let _ = tokio::time::timeout(std::time::Duration::from_millis(250), self.changed()).await;
        }
    }

    pub(crate) fn set_spill_dir(&self, dir: Option<&Path>) {
        *self.inner.spill_dir.lock().unwrap() = dir.map(Path::to_path_buf);
    }

    /// Every tool of every ready server, as offered to the model.
    pub(crate) fn tools(&self) -> Vec<McpTool> {
        let map = self.inner.servers.lock().unwrap();
        let mut tools: Vec<McpTool> = Vec::new();
        // Two servers whose names differ only in characters a tool name cannot hold come out
        // the same; the first keeps the name rather than the model calling one and getting
        // the other.
        for tool in map.values().filter(|s| s.state == State::Ready).flat_map(server_tools) {
            if !tools.iter().any(|seen| seen.name == tool.name) {
                tools.push(tool);
            }
        }
        tools
    }

    pub(crate) fn status(&self) -> Vec<Status> {
        let map = self.inner.servers.lock().unwrap();
        map.values()
            .map(|server| Status {
                name: server.config.name.clone(),
                scope: server.config.scope,
                state: server.state.clone(),
                target: server.config.describe(),
                tools: server_tools(server),
                stderr: server.stderr.lock().unwrap().iter().cloned().collect(),
            })
            .collect()
    }

    /// The server entry behind a name, for the approval prompt.
    pub(crate) fn config(&self, name: &str) -> Option<ServerConfig> {
        self.inner.servers.lock().unwrap().get(name).map(|s| s.config.clone())
    }

    /// Run one tool. `name` is the model's (`mcp__server__tool`).
    pub(crate) async fn call(&self, name: &str, arguments: Value) -> anyhow::Result<ToolOutput> {
        let (service, tool, timeout) = {
            let map = self.inner.servers.lock().unwrap();
            let found = map.values().find_map(|server| {
                let tool = server_tools(server).into_iter().find(|t| t.name == name)?;
                Some((server, tool))
            });
            let Some((server, tool)) = found else {
                return Err(anyhow::anyhow!("Unknown tool: {}", name));
            };
            let service = server
                .service
                .clone()
                .filter(|_| server.state == State::Ready)
                .ok_or_else(|| anyhow::anyhow!("MCP server \"{}\" is not running", server.config.name))?;
            (service, tool, server.config.timeout)
        };

        let mut params = CallToolRequestParams::new(tool.tool.clone());
        params.arguments = match arguments {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => return Err(anyhow::anyhow!("arguments must be an object, got {}", other)),
        };
        let call = service.call_tool(params);
        let result = match timeout {
            Some(limit) => tokio::time::timeout(limit, call)
                .await
                .map_err(|_| anyhow::anyhow!("no answer from \"{}\" within {} s", tool.server, limit.as_secs()))?,
            None => call.await,
        };
        let result = match result {
            Ok(result) => result,
            Err(e) => {
                if service.is_closed() {
                    self.mark_lost(&tool.server);
                }
                return Err(anyhow::anyhow!("{}", e));
            }
        };
        let mut output = to_output(&result);
        output.text = self.spill(&tool, output.text);
        Ok(output)
    }

    fn mark_lost(&self, server: &str) {
        let lost = {
            let mut map = self.inner.servers.lock().unwrap();
            match map.get_mut(server) {
                Some(s) if s.state == State::Ready => {
                    s.state = State::Failed("connection lost".to_string());
                    s.service = None;
                    true
                }
                _ => false,
            }
        };
        if lost {
            self.push(Notice::Lost { server: server.to_string() });
        }
    }

    fn spill(&self, tool: &McpTool, text: String) -> String {
        if text.len() <= SPILL_THRESHOLD {
            return text;
        }
        let Some(dir) = self.inner.spill_dir.lock().unwrap().clone() else {
            // No scratch directory to put it in. Whole is the one honest alternative.
            return text;
        };
        // A JSON result goes to the file pretty-printed. As sent, it is often one line —
        // ClickHouse's `list_tables` was 729 KB of it, every Cyrillic letter a `\u` escape —
        // and `read` and `grep` work by lines: a grep for `"name"` matched the one line,
        // and the model concluded the database had three tables out of 54.
        let text = match serde_json::from_str::<Value>(&text) {
            Ok(value) => serde_json::to_string_pretty(&value).unwrap_or(text),
            Err(_) => text,
        };
        let n = self.inner.spilled.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let path = dir.join(format!("mcp-{}-{}-{}.txt", sanitize(&tool.server), sanitize(&tool.tool), n));
        if std::fs::write(&path, &text).is_err() {
            return text;
        }
        let head = crate::util::truncate_at_boundary(&text, SPILL_HEAD);
        format!(
            "{head}\n\n[haico] The result is {} bytes; the first {} are above. The whole result is in {} \
             — use read (with offset/limit) or grep on it for the rest.",
            text.len(),
            head.len(),
            crate::util::display_path(&path),
        )
    }
}

/// Both files read, and what is needed to start them: the frontend asks about the project
/// servers nobody approved yet (`unapproved`), records each yes (`approve`), then `start`s.
pub(crate) struct Startup {
    cwd: PathBuf,
    loaded: config::Loaded,
    trust: trust::Trust,
    /// Approved in this run, whether or not the store could record it: the user said yes.
    approved: Vec<String>,
}

impl Startup {
    pub(crate) fn load(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
            loaded: config::load(cwd),
            trust: trust::Trust::from_store(),
            approved: Vec::new(),
        }
    }

    /// Project servers that would run and have not been approved as they are written now.
    pub(crate) fn unapproved(&self) -> Vec<ServerConfig> {
        self.loaded
            .servers
            .iter()
            .filter(|s| s.scope == Scope::Project && !s.disabled)
            .filter(|s| !self.trust.is_approved(&self.cwd, &s.name, &s.fingerprint))
            .cloned()
            .collect()
    }

    /// Approve for this run, and remember it. `Err` says only that remembering failed.
    pub(crate) fn approve(&mut self, server: &ServerConfig) -> anyhow::Result<()> {
        self.approved.push(server.name.clone());
        self.trust.approve(&self.cwd, &server.name, &server.fingerprint)
    }

    /// Hand everything to the hub, which starts what may start in the background.
    pub(crate) fn start(self, hub: &Hub) {
        let Startup { cwd, loaded, trust, approved } = self;
        hub.report_config_errors(loaded.errors);
        hub.start(loaded.servers, &|s| {
            approved.contains(&s.name) || trust.is_approved(&cwd, &s.name, &s.fingerprint)
        });
    }
}

/// The approval question for one project server, as lines: what it runs, and what it would
/// be allowed to do unasked.
pub(crate) fn approval_lines(server: &ServerConfig) -> Vec<String> {
    let mut lines = vec![
        format!("The project's .haico/mcp.json asks to start MCP server \"{}\":", server.name),
        format!("  {}", server.describe()),
    ];
    if let Transport::Stdio { env, cwd, .. } = &server.transport {
        if !env.is_empty() {
            lines.push(format!("  env: {}", env.keys().cloned().collect::<Vec<_>>().join(", ")));
        }
        if let Some(cwd) = cwd {
            lines.push(format!("  in: {}", crate::util::display_path(cwd)));
        }
    }
    match &server.auto_allow {
        Some(config::AutoAllow::All(true)) => lines.push("  autoAllow: every tool runs without asking".to_string()),
        Some(config::AutoAllow::Tools(tools)) if !tools.is_empty() => {
            lines.push(format!("  autoAllow (run without asking): {}", tools.join(", ")))
        }
        _ => {}
    }
    if !server.read_only.is_empty() {
        lines.push(format!("  readOnly (offered in plan mode): {}", server.read_only.join(", ")));
    }
    lines
}

/// What one call to an MCP tool needs before it runs.
///
/// Plan mode offers only read-only tools, and refuses the rest by name for a model that
/// calls one it was not offered. Read-only does not mean unasked: the hint is the server's
/// word about itself, and a wrong one should cost a prompt, not a write. `autoAllow` is the
/// user's word, so it is the one thing that skips the reviewer and the prompt.
pub(crate) fn gate(mode: crate::modes::Mode, tool: &McpTool) -> crate::permissions::Gate {
    use crate::permissions::Gate;
    if mode == crate::modes::Mode::Plan && tool.read_only.is_none() {
        return Gate::Deny(format!(
            "'{}' is not available in plan mode: its server does not mark it read-only, and plan \
             mode changes nothing until a plan is approved. Finish investigating, then call \
             `present_plan` with what you propose to do.",
            tool.name
        ));
    }
    if tool.auto_allow {
        return Gate::Auto;
    }
    Gate::Ask {
        keys: vec![tool.name.clone()],
        label: format!("Allow all '{}' calls to MCP server \"{}\"", tool.tool, tool.server),
    }
}

fn server_tools(server: &Server) -> Vec<McpTool> {
    let config = &server.config;
    server
        .tools
        .iter()
        .filter(|tool| config.offers(&tool.name))
        .map(|tool| {
            let name = tool_name(&config.name, &tool.name);
            let read_only = if config.read_only.iter().any(|t| t == tool.name.as_ref()) {
                Some(ReadOnly::Config)
            } else if tool.annotations.as_ref().and_then(|a| a.read_only_hint) == Some(true) {
                Some(ReadOnly::Hint)
            } else {
                None
            };
            McpTool {
                schema: openai_schema(&name, tool),
                name,
                server: config.name.clone(),
                tool: tool.name.to_string(),
                read_only,
                auto_allow: config.auto_allows(&tool.name),
            }
        })
        .collect()
}

/// `mcp__<server>__<tool>`, in the characters OpenAI-style function names allow and no longer
/// than they allow. Claude Code's naming, so a model that has seen it there recognizes it.
pub(crate) fn tool_name(server: &str, tool: &str) -> String {
    let name = format!("{}{}__{}", PREFIX, sanitize(server), sanitize(tool));
    crate::util::truncate_at_boundary(&name, 64).to_string()
}

fn sanitize(text: &str) -> String {
    text.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

fn openai_schema(name: &str, tool: &Tool) -> Value {
    let mut parameters = Value::Object((*tool.input_schema).clone());
    crate::tools::simplify_schema(&mut parameters);
    // Some servers send `{}` or `{"type": "object"}` for a tool without arguments, and some
    // chat templates (llama.cpp's among them) choke on a function with no properties.
    if let Value::Object(map) = &mut parameters {
        map.entry("type").or_insert_with(|| json!("object"));
        map.entry("properties").or_insert_with(|| json!({}));
    }
    // Whole, never shortened: the description is how the model knows when to reach for it.
    let description = tool.description.as_deref().or(tool.title.as_deref()).unwrap_or("");
    json!({
        "type": "function",
        "function": { "name": name, "description": description, "parameters": parameters }
    })
}

/// A tool's result, in the shape haico's own tools return: text, images beside it, failed.
fn to_output(result: &rmcp::model::CallToolResult) -> ToolOutput {
    let mut texts: Vec<String> = Vec::new();
    let mut images = Vec::new();
    for block in &result.content {
        // Through JSON rather than rmcp's types: the wire shape is the spec's and stable,
        // where the Rust wrappers around it have changed between releases.
        let value = serde_json::to_value(block).unwrap_or_default();
        match value["type"].as_str().unwrap_or_default() {
            "text" => texts.push(value["text"].as_str().unwrap_or_default().to_string()),
            "image" => images.push((
                value["mimeType"].as_str().unwrap_or("image/png").to_string(),
                value["data"].as_str().unwrap_or_default().to_string(),
            )),
            "resource" => {
                let resource = &value["resource"];
                match resource["text"].as_str() {
                    Some(text) => texts.push(text.to_string()),
                    None => texts.push(format!(
                        "[binary resource {} ({}), not shown]",
                        resource["uri"].as_str().unwrap_or("?"),
                        resource["mimeType"].as_str().unwrap_or("unknown type")
                    )),
                }
            }
            "resource_link" => texts.push(format!(
                "[resource {}{}]",
                value["uri"].as_str().unwrap_or("?"),
                value["name"].as_str().map(|n| format!(" \"{}\"", n)).unwrap_or_default()
            )),
            "audio" => texts.push("[audio, not shown]".to_string()),
            _ => {}
        }
    }
    // A tool that answered only in structured content still answered.
    if texts.is_empty() {
        if let Some(structured) = &result.structured_content {
            texts.push(structured.to_string());
        }
    }
    ToolOutput { text: texts.join("\n"), images, failed: result.is_error.unwrap_or(false) }
}

/// Our side of each connection: it only listens for the tool list changing.
struct Handler {
    hub: Weak<Inner>,
    server: String,
}

impl ClientHandler for Handler {
    async fn on_tool_list_changed(&self, context: NotificationContext<RoleClient>) {
        let Some(inner) = self.hub.upgrade() else { return };
        let hub = Hub { inner };
        let Ok(tools) = context.peer.list_all_tools().await else { return };
        let count = {
            let mut map = hub.inner.servers.lock().unwrap();
            let Some(server) = map.get_mut(&self.server).filter(|s| s.state == State::Ready) else { return };
            server.tools = tools;
            server_tools(server).len()
        };
        hub.push(Notice::ToolsChanged { server: self.server.clone(), tools: count });
    }

    fn get_info(&self) -> rmcp::model::ClientConfig {
        let mut info = rmcp::model::ClientConfig::default();
        info.client_info.name = "haico".to_string();
        info.client_info.version = env!("CARGO_PKG_VERSION").to_string();
        info
    }
}

async fn connect(
    config: &ServerConfig,
    handler: Handler,
    stderr: Arc<Mutex<VecDeque<String>>>,
) -> Result<(Service, Vec<Tool>), String> {
    match &config.transport {
        Transport::Stdio { command, args, env, cwd } => {
            let transport = spawn_stdio(command, args, env, cwd.as_deref(), stderr)?;
            serve_and_list(handler, transport).await
        }
        Transport::Http { url, headers } => {
            use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
            use rmcp::transport::StreamableHttpClientTransport;
            let mut custom = std::collections::HashMap::new();
            for (key, value) in headers {
                let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                    .map_err(|e| format!("header \"{}\": {}", key, e))?;
                let value = reqwest::header::HeaderValue::from_str(value).map_err(|e| format!("header \"{}\": {}", key, e))?;
                custom.insert(name, value);
            }
            let transport = StreamableHttpClientTransport::from_config(
                StreamableHttpClientTransportConfig::with_uri(url.as_str()).custom_headers(custom),
            );
            serve_and_list(handler, transport).await
        }
    }
}

/// The MCP handshake, then the server's tools.
async fn serve_and_list<T, E, A>(handler: Handler, transport: T) -> Result<(Service, Vec<Tool>), String>
where
    T: rmcp::transport::IntoTransport<RoleClient, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    let service = handler.serve(transport).await.map_err(|e| e.to_string())?;
    let tools = service.list_all_tools().await.map_err(|e| format!("tools/list: {}", e))?;
    Ok((service, tools))
}

fn spawn_stdio(
    command: &str,
    args: &[String],
    env: &BTreeMap<String, String>,
    cwd: Option<&Path>,
    stderr: Arc<Mutex<VecDeque<String>>>,
) -> Result<rmcp::transport::TokioChildProcess, String> {
    use process_wrap::tokio::CommandWrap;

    let program = resolve_program(command);
    let mut cmd = tokio::process::Command::new(&program);
    cmd.args(args).envs(env);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let mut wrap = CommandWrap::from(cmd);
    #[cfg(windows)]
    {
        use windows::Win32::System::Threading::CREATE_NO_WINDOW;
        wrap.wrap(process_wrap::tokio::CreationFlags(CREATE_NO_WINDOW));
        wrap.wrap(process_wrap::tokio::JobObject);
    }
    #[cfg(unix)]
    {
        wrap.wrap(process_wrap::tokio::ProcessGroup::leader());
    }
    wrap.wrap(process_wrap::tokio::KillOnDrop);

    let (transport, child_stderr) = rmcp::transport::TokioChildProcess::builder(wrap)
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {}: {}", program.display(), e))?;

    // Drained continuously: a pipe nobody reads fills up, and a server blocked writing a log
    // line stops answering. Only the tail is kept.
    if let Some(child_stderr) = child_stderr {
        tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(child_stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut kept = stderr.lock().unwrap();
                if kept.len() == STDERR_LINES {
                    kept.pop_front();
                }
                kept.push_back(line);
            }
        });
    }
    Ok(transport)
}

/// Find a bare command the way a shell would. On Windows, `Command` looks for `<name>.exe`
/// only, and `npx`, `uvx` wrappers and the like are `.cmd` files — the example in every MCP
/// server's README would fail with "program not found".
fn resolve_program(command: &str) -> PathBuf {
    let path = PathBuf::from(command);
    if !cfg!(windows) || path.extension().is_some() || path.components().count() > 1 {
        return path;
    }
    let extensions: Vec<String> = std::env::var("PATHEXT")
        .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
        .split(';')
        .filter(|e| !e.is_empty())
        .map(str::to_string)
        .collect();
    let dirs = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default();
    for dir in dirs {
        for extension in &extensions {
            let candidate = dir.join(format!("{}{}", command, extension.to_lowercase()));
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    path
}

/// A hub with one server attached: haico's own tool server, under `name`, configured by the
/// `mcp.json` entry `entry` (which needs a `command`, never run).
#[cfg(test)]
pub(crate) async fn fixture_hub(hub: &Hub, name: &str, entry: Value) {
    let text = json!({"mcpServers": {name: entry}}).to_string();
    let (mut servers, problems) = config::parse(&text, Scope::Global, Path::new("/"), &|_| None).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
    let (client, server) = tokio::io::duplex(1 << 20);
    tokio::spawn(async move {
        if let Ok(running) = crate::testutil::test_server().serve(server).await {
            let _ = running.waiting().await;
        }
    });
    hub.attach(servers.remove(0), client).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_connected_server_lists_and_runs_its_tools_under_prefixed_names() {
        let hub = Hub::default();
        fixture_hub(&hub, "fix", json!({"command": "unused", "tools": ["calculator", "get_date"], "autoAllow": ["get_date"], "readOnly": ["get_date"]})).await;
        assert_eq!(hub.loading(), None);
        assert_eq!(hub.take_notices(), [Notice::Loaded { servers: 1, tools: 2 }]);

        let tools = hub.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["mcp__fix__calculator", "mcp__fix__get_date"]);
        let calculator = &tools[0];
        assert_eq!(calculator.schema["function"]["name"], "mcp__fix__calculator");
        assert!(calculator.schema["function"]["parameters"]["properties"]["operation"].is_object());
        assert_eq!((calculator.auto_allow, calculator.read_only), (false, None));
        assert_eq!((tools[1].auto_allow, tools[1].read_only), (true, Some(ReadOnly::Config)));

        let output = hub.call("mcp__fix__calculator", json!({"operation": "multiply", "a": 6, "b": 7})).await.unwrap();
        assert!(output.text.contains("42"), "{}", output.text);
        assert!(!output.failed);
        // Filtered out by `tools`, so not callable either.
        assert!(hub.call("mcp__fix__get_time", json!({})).await.is_err());
    }

    #[test]
    fn the_gate_follows_auto_allow_and_plan_mode_follows_read_only() {
        use crate::modes::Mode;
        use crate::permissions::Gate;
        let tool = |auto_allow, read_only| McpTool {
            name: "mcp__db__query".into(),
            server: "db".into(),
            tool: "query".into(),
            schema: json!({}),
            read_only,
            auto_allow,
        };
        assert!(matches!(gate(Mode::Normal, &tool(true, None)), Gate::Auto));
        assert!(matches!(gate(Mode::Auto, &tool(false, None)), Gate::Ask { keys, .. } if keys == ["mcp__db__query"]));
        assert!(matches!(gate(Mode::Plan, &tool(true, None)), Gate::Deny(_)));
        // Read-only gets it into plan mode, not past the prompt.
        assert!(matches!(gate(Mode::Plan, &tool(false, Some(ReadOnly::Hint))), Gate::Ask { .. }));
        assert!(matches!(gate(Mode::Plan, &tool(true, Some(ReadOnly::Config))), Gate::Auto));
    }

    #[test]
    fn tool_names_are_prefixed_sanitized_and_short_enough() {
        assert_eq!(tool_name("git", "status"), "mcp__git__status");
        assert_eq!(tool_name("my server", "do.it"), "mcp__my_server__do_it");
        assert!(tool_name("s", &"x".repeat(100)).len() <= 64);
    }

    #[test]
    fn a_tool_without_properties_gets_an_empty_object() {
        let tool: Tool = serde_json::from_value(json!({"name": "t", "inputSchema": {"type": "object"}})).unwrap();
        let schema = openai_schema("mcp__s__t", &tool);
        assert_eq!(schema["function"]["parameters"], json!({"type": "object", "properties": {}}));
        assert_eq!(schema["function"]["description"], "");
    }

    #[test]
    fn results_keep_text_images_resources_and_the_error_flag() {
        let result: rmcp::model::CallToolResult = serde_json::from_value(json!({
            "content": [
                {"type": "text", "text": "one"},
                {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                {"type": "resource", "resource": {"uri": "file:///a", "text": "two"}},
                {"type": "resource_link", "uri": "file:///b", "name": "b"}
            ],
            "isError": true
        }))
        .unwrap();
        let output = to_output(&result);
        assert_eq!(output.text, "one\ntwo\n[resource file:///b \"b\"]");
        assert_eq!(output.images, [("image/png".to_string(), "AAAA".to_string())]);
        assert!(output.failed);

        let structured: rmcp::model::CallToolResult =
            serde_json::from_value(json!({"content": [], "structuredContent": {"rows": 3}})).unwrap();
        assert_eq!(to_output(&structured).text, "{\"rows\":3}");
    }

    #[test]
    fn a_long_result_is_spilled_to_the_scratch_directory_whole() {
        let dir = crate::testutil::TempDir::new("mcp-spill");
        let hub = Hub::default();
        let tool = McpTool {
            name: "mcp__db__query".into(),
            server: "db".into(),
            tool: "query".into(),
            schema: json!({}),
            read_only: None,
            auto_allow: false,
        };
        let short = "x".repeat(100);
        // No directory: whole, in the conversation.
        let long = "y".repeat(SPILL_THRESHOLD + 1);
        assert_eq!(hub.spill(&tool, long.clone()), long);

        hub.set_spill_dir(Some(dir.path()));
        assert_eq!(hub.spill(&tool, short.clone()), short);
        let spilled = hub.spill(&tool, long.clone());
        assert!(spilled.len() < SPILL_HEAD + 1000, "{}", spilled.len());
        assert!(spilled.contains("mcp-db-query-1.txt"), "{spilled}");
        assert_eq!(std::fs::read_to_string(dir.join("mcp-db-query-1.txt")).unwrap(), long);

        // JSON is spilled a field to a line, escapes decoded, so `grep` finds each match.
        let rows: Vec<Value> = (0..2000).map(|i| json!({"name": format!("таблица_{i}")})).collect();
        let compact = serde_json::to_string(&json!({ "tables": rows })).unwrap();
        let escaped = compact.replace("таблица", "\\u0442\\u0430\\u0431\\u043b\\u0438\\u0446\\u0430");
        assert!(escaped.len() > SPILL_THRESHOLD && !escaped.contains('\n'));
        hub.spill(&tool, escaped);
        let written = std::fs::read_to_string(dir.join("mcp-db-query-2.txt")).unwrap();
        assert_eq!(written.lines().filter(|l| l.contains("\"name\": \"таблица_")).count(), 2000);
    }

    #[test]
    fn disabled_and_unapproved_servers_do_not_start() {
        let parse = |text: &str, scope| config::parse(text, scope, Path::new("/"), &|_| None).unwrap().0;
        let mut servers = parse(r#"{"mcpServers": {"off": {"command": "x", "disabled": true}}}"#, Scope::Global);
        servers.extend(parse(r#"{"mcpServers": {"repo": {"command": "x"}}}"#, Scope::Project));
        let hub = Hub::default();
        hub.start(servers, &|_| false);
        assert_eq!(hub.loading(), None);
        let states: Vec<State> = hub.status().into_iter().map(|s| s.state).collect();
        assert_eq!(states, [State::Disabled, State::NotApproved]);
        assert_eq!(hub.take_notices(), [Notice::NotApproved(vec!["repo".to_string()])]);
    }
}
