//! The agent core: everything a conversation with the model needs, with no
//! terminal attached.
//!
//! Until this existed, all of it lived as locals in `main()` — the client, the
//! tool router, the history, the permissions, the workspace, the mode — which is
//! why there was nothing for a second frontend to attach to. A GUI cannot borrow
//! a stack frame.
//!
//! **The rule that keeps this honest: nothing in here may name `out`, `Keys` or
//! `crossterm`.** If the core can print, it will print, and a second frontend
//! ends up reimplementing the loop instead of driving it. So the methods here
//! return what happened and let the caller say it — `resume` hands back what it
//! restored rather than announcing it, `clear` reports whether the project file
//! changed under it rather than printing that it did.
//!
//! What is deliberately *not* here yet is the turn loop itself. It is entangled
//! with output, with the interrupt key and with the permission prompt, and
//! moving it is the event-stream work (section 11.2 of `plan.md`) rather than
//! this. This is the state it will run against.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::config::{resolve_settings, Config, Reasoning, ResolvedSettings};
use crate::llm::LMStudioClient;
use crate::modes::{Mode, ReviewCache};
use crate::permissions::PermissionManager;
use crate::prompt::{compose_system_prompt, load_project_context, load_user_context};
use crate::session::{
    set_system_message,
    load_session, session_file_path, ReasoningStats,
};
use crate::tools::{mcp_tools_to_openai_format, MyMcpServer};
use crate::workspace::Workspace;

/// What a successful [`Agent::resume`] restored, for the caller to report.
pub(crate) struct Resumed {
    pub(crate) messages: usize,
    pub(crate) saved_at: String,
    /// The model the session was saved under, which need not be the current one.
    pub(crate) model: String,
}

/// What [`Agent::clear`] found when it re-read the project instructions. The
/// file is re-read there and only there because that is the moment a stale copy
/// starts to cost something: whatever the model knew from having read the file
/// is being thrown out with the history, leaving the system message as the only
/// copy of it.
pub(crate) enum ContextReload {
    /// The file is unchanged (or was absent and still is); nothing to say.
    Unchanged,
    /// Re-read from this file.
    Reloaded(String),
    /// It was there at startup and is not any more.
    Gone,
}

pub(crate) struct Agent {
    pub(crate) config: Config,
    pub(crate) client: LMStudioClient,
    pub(crate) server: MyMcpServer,
    /// The tool array sent with every request, built once from the router.
    pub(crate) tools: Vec<Value>,
    pub(crate) settings: ResolvedSettings,
    /// Pulled out of `settings` because `/context` is not the only thing that
    /// sets it and the value in force has to outlive a re-resolve.
    pub(crate) context_tokens: usize,
    /// Likewise: `/reasoning` overrides the resolved setting for the session
    /// without touching the config file, so the policy in force lives here.
    pub(crate) reasoning: Reasoning,
    pub(crate) reasoning_threshold: usize,
    /// Savings made on *this* history. Reset with the history by `clear`, and
    /// deliberately not by `/compact` or `/model` — neither discards the
    /// conversation, so the savings still stand.
    pub(crate) reasoning_stats: ReasoningStats,
    pub(crate) messages: Vec<Value>,
    /// `None` when the model is configured without haico's prompt and neither
    /// instruction file exists — then the conversation carries no system message.
    pub(crate) system_prompt: Option<String>,
    pub(crate) project_context: Option<(String, String)>,
    /// Read once at startup and never reloaded: it lives outside the project and
    /// nothing here writes to it, so the reload points the project file needs do
    /// not apply.
    pub(crate) user_context: Option<String>,
    pub(crate) session_path: Option<PathBuf>,
    /// Approvals for this session only, never persisted, so resuming never
    /// silently carries over permission to run something.
    pub(crate) permissions: PermissionManager,
    pub(crate) review_cache: ReviewCache,
    /// Not persisted either, and for the same reason: a session that happened to
    /// end in auto mode should not quietly resume deciding things for the user.
    pub(crate) mode: Mode,
    pub(crate) workspace: Workspace,
}

impl Agent {
    /// Build the agent for an already-chosen `model`. Choosing one is a
    /// conversation with the user, so it stays with the frontend; by the time
    /// this is called the answer is known.
    pub(crate) fn new(config: Config, model: String, cwd: &Path) -> Self {
        let mut client = LMStudioClient::new(config.llm_url.clone(), model, config.token.clone());
        let settings = resolve_settings(&config, &client.model);
        client.apply_settings(&settings);

        let server = MyMcpServer::new(client.clone(), true, config.searxng.clone(), true);
        let tools = mcp_tools_to_openai_format(&server);

        let project_context = load_project_context();
        let user_context = load_user_context();
        let system_prompt = compose_system_prompt(
            config.system_prompt.as_deref(),
            &client.model,
            settings.use_system_prompt,
            user_context.as_deref(),
            project_context.as_ref(),
        );

        let workspace = Workspace::new(cwd, &config.file.read_paths);
        let context_tokens = settings.context_tokens;
        let reasoning = settings.reasoning();
        let reasoning_threshold = settings.reasoning_threshold();

        let mut agent = Agent {
            config,
            client,
            server,
            tools,
            settings,
            context_tokens,
            reasoning,
            reasoning_threshold,
            reasoning_stats: ReasoningStats::default(),
            messages: Vec::new(),
            system_prompt,
            project_context,
            user_context,
            session_path: session_file_path(),
            permissions: PermissionManager::new(),
            review_cache: ReviewCache::new(),
            mode: Mode::Normal,
            workspace,
        };
        agent.seed_system_message();
        agent
    }

    /// Start the history with the system message, if there is one. The only way
    /// a system message is ever added, so `clear` and `new` cannot disagree
    /// about whether one belongs at the front.
    fn seed_system_message(&mut self) {
        if let Some(content) = &self.system_prompt {
            self.messages.push(json!({ "role": "system", "content": content }));
        }
    }

    /// Whether a saved conversation exists for this directory — the hint that
    /// `--resume` would find something.
    pub(crate) fn has_saved_session(&self) -> bool {
        self.session_path.as_deref().is_some_and(|p| p.exists())
    }

    /// Restore the saved conversation for this directory, returning what was
    /// restored so the caller can report it.
    ///
    /// Only the messages and the plan come back — never the stored system
    /// prompt, whose date, branch and git status would otherwise be whatever
    /// they were when the session was saved.
    pub(crate) fn resume(&mut self) -> Result<Resumed> {
        let path = self.session_path.as_deref().ok_or_else(|| anyhow!("no home directory"))?;
        let session = load_session(path)?;
        let restored = Resumed {
            messages: session.messages.len(),
            saved_at: session.saved_at.clone(),
            model: session.model.clone(),
        };
        self.messages.extend(session.messages);
        *self.server.todos.lock().unwrap() = session.todos;
        Ok(restored)
    }

    /// Throw away the conversation and everything scoped to it, and report
    /// whether the project's instructions changed while it ran.
    ///
    /// What survives is deliberate. Approvals in `permissions` stay, as they
    /// always have. What goes with the history: the review cache (an entry is
    /// keyed by a whole script or a whole diff, so a long session accumulates
    /// them, and no verdict is invalidated by this — the reviewer never saw the
    /// conversation); the plan (pinned above the input box, a plan for work that
    /// was just discarded would sit there for the rest of the session); and the
    /// reasoning savings, which were savings on the history that just went.
    pub(crate) fn clear(&mut self) -> ContextReload {
        let reloaded = load_project_context();
        let outcome = if reloaded == self.project_context {
            ContextReload::Unchanged
        } else {
            self.project_context = reloaded;
            // Only the prompt: the history is about to be cleared anyway, so
            // replacing the system message in it would be work thrown away.
            self.recompose_system_prompt();
            match &self.project_context {
                Some((name, _)) => ContextReload::Reloaded(name.clone()),
                None => ContextReload::Gone,
            }
        };

        self.messages.clear();
        self.seed_system_message();
        self.review_cache.clear();
        self.server.todos.lock().unwrap().clear();
        self.reasoning_stats = ReasoningStats::default();
        outcome
    }

    /// Recompose the system prompt from the current model, settings and
    /// instruction files. Returns whether it actually changed; touches only the
    /// prompt, never the history.
    fn recompose_system_prompt(&mut self) -> bool {
        let rebuilt = compose_system_prompt(
            self.config.system_prompt.as_deref(),
            &self.client.model,
            self.settings.use_system_prompt,
            self.user_context.as_deref(),
            self.project_context.as_ref(),
        );
        let changed = rebuilt != self.system_prompt;
        self.system_prompt = rebuilt;
        changed
    }

    /// Recompose the system prompt and, if it changed, replace it in the history
    /// in place so the conversation itself carries on unchanged.
    ///
    /// Both halves matter: composing without replacing leaves the conversation
    /// carrying the old prompt, and replacing without composing sends the new
    /// model the previous one's.
    pub(crate) fn rebuild_system_prompt(&mut self) {
        if self.recompose_system_prompt() {
            set_system_message(&mut self.messages, self.system_prompt.as_deref());
        }
    }

    /// Point the conversation at a different model, re-resolving everything that
    /// is configured per model.
    ///
    /// Re-resolving is the whole point: a 131072-token budget carried onto a 32k
    /// model would warn far too late to be any use. The conversation itself is
    /// kept — switching model is not starting over.
    pub(crate) fn switch_model(&mut self, model: String) {
        self.settings = resolve_settings(&self.config, &model);
        self.context_tokens = self.settings.context_tokens;
        // Re-resolved with the rest: whether a chain of thought is worth keeping
        // is a property of the model that produced it, so a policy chosen for a
        // thinking model has no business surviving a switch to one that does not
        // think at all.
        self.reasoning = self.settings.reasoning();
        self.reasoning_threshold = self.settings.reasoning_threshold();

        // Both copies. The REPL drives one and `web_fetch` summarizes through
        // the server's; leaving one behind would silently keep using the old
        // model for half the work.
        self.client.model = model.clone();
        self.client.apply_settings(&self.settings);
        self.server.llm_client.model = model;
        self.server.llm_client.apply_settings(&self.settings);

        // The reviewer is this model too, so its verdicts leave with it.
        self.review_cache.clear();
        self.rebuild_system_prompt();
    }

    /// The model currently in use.
    pub(crate) fn model(&self) -> &str {
        &self.client.model
    }

    /// The plan, as the todo tool last left it.
    pub(crate) fn plan(&self) -> Vec<crate::tools::TodoItem> {
        self.server.todos.lock().unwrap().clone()
    }
}
