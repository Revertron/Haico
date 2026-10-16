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
//! The turn loop lives here too, and it is why the rule above is written down.
//! It says what happens through an `EventSink`, shares exactly three things with
//! its frontend through a `TurnLink` (the mode, the queue of typed-ahead input,
//! and cancellation), and asks questions by sending an `Ask` carrying the reply
//! channel. It never draws, never reads a key, and never chooses a word a person
//! reads — the flavor text on the activity bar and in the turn summary is picked
//! by the frontend from a `Waiting` kind and an elapsed time.

use crate::subagent::run_sub_agent;
use crate::tools::SUB_AGENT;
use crate::trace::Caller;
use anyhow::{anyhow, Result};
use rmcp::handler::server::wrapper::Parameters;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::config::{
    autocompact_threshold, resolve_settings, save_active_provider, save_default_model,
    save_model_context, save_provider_model, Config,
    Reasoning, ResolvedSettings, Setting, AUTOCOMPACT_FRACTION, FROM_UNSET,
};
use crate::ask::{self, ASK_USER};
use crate::llm::{Chunk, LMStudioClient};
use crate::modes::{
    gate_for_mode, judge_tool_call, tools_for_mode, with_mode_note, Mode, PlanDecision,
    ReviewCache, Verdict, PRESENT_PLAN, START_PLAN,
};
use crate::permissions::{Gate, PermissionManager, PermissionResponse};
use crate::memory;
use crate::prompt::{compose_system_prompt, load_project_context, load_user_context};
// The one `render` import, and it earns its place: a diff is the *content* of a
// pending change, which the turn has to compute because only it has the old file.
// How much of it to show is the frontend's, and that half lives there.
use crate::render::file_change_diff;
use crate::repetition::Repeating;
use crate::scratch::Scratch;
use crate::skills;
use crate::session::{
    apply_reasoning, compact_history, estimate_tokens, load_session,
    reasoning_needs_a_request, reasoning_text, save_session, session_file_path,
    set_system_message, ReasoningOutcome, ReasoningStats,
};
use crate::tools::{mcp_tools_to_openai_format, parse_tool_arguments, repair_leaked_arguments, MyMcpServer, ReadParams};
use crate::turn::*;
use crate::util::{absolute_path, display_path, truncate_at_boundary};
use crate::workspace::{expand_home, Workspace};

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

/// A request's size as the server counted it, and haico's estimate of the history it was
/// sent with — the second is what lets `Agent::context_used` add only what came after.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Measured {
    pub(crate) prompt_tokens: usize,
    pub(crate) estimate: usize,
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
    /// What the server last said the request cost, for `context_used`. `None` until a reply
    /// reports usage, and forgotten when the model changes (another tokenizer).
    pub(crate) measured: Option<Measured>,
    /// `None` when the model is configured without haico's prompt and neither
    /// instruction file exists — then the conversation carries no system message.
    pub(crate) system_prompt: Option<String>,
    pub(crate) project_context: Option<(String, String)>,
    /// Read once at startup and never reloaded: it lives outside the project and
    /// nothing here writes to it, so the reload points the project file needs do
    /// not apply.
    pub(crate) user_context: Option<String>,
    /// The project's memory index — one line per stored fact — as the system prompt
    /// carries it.
    ///
    /// Refreshed only where the history is being thrown away or rewritten anyway
    /// (`clear`, and after a compaction), never after a `remember` lands mid-turn. The
    /// system message is the very front of the prefix the server caches, so rewriting
    /// it invalidates every token behind it and re-prefills the whole conversation —
    /// and it would buy nothing, because a fact the model just recorded is already in
    /// the history two messages up. It is when the history goes that the index has to
    /// be there instead.
    pub(crate) memory_index: Option<String>,
    /// The skills index — one line per skill — as the system prompt carries it. Refreshed
    /// at exactly the points `memory_index` is and for its reason: a skill dropped in
    /// mid-turn is not worth re-prefilling the whole conversation for.
    pub(crate) skill_index: Option<String>,
    /// The library index — one line per crate — refreshed with the other two. The library
    /// lives in haico's store rather than the project, but that changes nothing about when
    /// rewriting the system message is affordable.
    pub(crate) library_index: Option<String>,
    pub(crate) session_path: Option<PathBuf>,
    /// Approvals for this session only, never persisted, so resuming never
    /// silently carries over permission to run something.
    pub(crate) permissions: PermissionManager,
    pub(crate) review_cache: ReviewCache,
    /// Not persisted either, and for the same reason: a session that happened to
    /// end in auto mode should not quietly resume deciding things for the user.
    pub(crate) mode: Mode,
    pub(crate) workspace: Workspace,
    /// This session's scratch directory. Held here because it is the
    /// conversation's, not the machine's: `clear` renews it, and dropping the
    /// agent takes it away.
    scratch: Scratch,
}

/// Make the user's skills directory readable, if it exists.
///
/// It lives in haico's store, outside every project, and a skill's instructions routinely
/// say "read reference.md" or "see templates/". Readable only: following a skill is no
/// reason to be able to rewrite it. Only once it exists, so the prompt does not list a
/// read-only root that is not there.
fn allow_user_skills(workspace: &mut Workspace) {
    if let Some(dir) = skills::user_dir().filter(|d| d.is_dir()) {
        workspace.add_readable(&dir);
    }
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
        let memory_index = memory::index_block(cwd);
        let skill_index = skills::index_block(&skills::roots(cwd));
        let library_index = server.library_root.as_deref().and_then(crate::library::index_block);

        // Before the prompt, not after: the prompt names the scratch directory, and the
        // file tools can only write there because the workspace was told about it.
        let scratch = Scratch::new();
        let mut workspace = Workspace::new(cwd, &config.file.read_paths);
        allow_user_skills(&mut workspace);
        workspace.set_scratch(scratch.path());

        let system_prompt = compose_system_prompt(
            config.system_prompt.as_deref(),
            &client.model,
            settings.use_system_prompt,
            user_context.as_deref(),
            project_context.as_ref(),
            skill_index.as_deref(),
            library_index.as_deref(),
            memory_index.as_deref(),
            &workspace,
        );

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
            measured: None,
            system_prompt,
            project_context,
            user_context,
            memory_index,
            skill_index,
            library_index,
            // Never in a test binary. `run_turn` saves after every turn, so an agent
            // built by a test wrote a session named for whatever directory the suite
            // was run in straight into the user's own store, among their real saved
            // conversations — and overwrote it again on every run. Nulled here rather
            // than at each of the half-dozen places a test builds an agent, because
            // that list only grows and the next addition would leak again. A test that
            // wants persistence sets the field, and `session.rs` covers saving and
            // loading against an explicit temp path.
            session_path: if cfg!(test) { None } else { session_file_path() },
            permissions: PermissionManager::new(),
            review_cache: ReviewCache::new(),
            mode: Mode::Normal,
            workspace,
            scratch,
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
    /// scratch directory, whose contents are working notes for the conversation
    /// that is going. The reasoning savings go too — they were savings on the
    /// history that just went.
    pub(crate) fn clear(&mut self) -> ContextReload {
        let reloaded = load_project_context();
        let outcome = if reloaded == self.project_context {
            ContextReload::Unchanged
        } else {
            self.project_context = reloaded;
            match &self.project_context {
                Some((name, _)) => ContextReload::Reloaded(name.clone()),
                None => ContextReload::Gone,
            }
        };

        // The new directory has a new name, so the prompt is rebuilt whether the
        // project's instructions changed or not — otherwise the fresh
        // conversation would be handed the path of a directory that was just
        // deleted. Only the prompt: the history is about to be cleared anyway,
        // so replacing the system message in it would be work thrown away.
        self.scratch.renew();
        self.workspace.set_scratch(self.scratch.path());
        // Reloaded with the project's instructions, and for the same reason: anything
        // the last conversation remembered is about to be the only trace of it left.
        self.memory_index = memory::index_for_cwd();
        self.skill_index = skills::index_for_cwd();
        self.library_index = self.server.library_root.as_deref().and_then(crate::library::index_block);
        self.recompose_system_prompt();

        self.messages.clear();
        self.seed_system_message();
        self.review_cache.clear();
        self.server.todos.lock().unwrap().clear();
        // The timers go with the conversation that set them: a reminder to "check the
        // test run" arriving in a conversation that never started one is a riddle.
        self.server.timers.lock().unwrap().clear();
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
            self.skill_index.as_deref(),
            self.library_index.as_deref(),
            self.memory_index.as_deref(),
            &self.workspace,
        );
        let changed = rebuilt != self.system_prompt;
        self.system_prompt = rebuilt;
        changed
    }

    /// Move the conversation to another directory, for `/cd`.
    ///
    /// Everything that answers "which project is this" is re-derived here, in one
    /// place and in this order, because the process's own working directory is what
    /// nearly all of it reads: `load_project_context` opens `HAICO.md` by a relative
    /// path, `index_for_cwd` walks `.haico/memory`, and `build_system_prompt` asks git
    /// where it is. So the `set_current_dir` has to land first and the rest has to
    /// follow it — which is why this is a method rather than five calls in a frontend
    /// that would have to remember the order.
    ///
    /// The prompt is rebuilt in place rather than left to the `/clear` the caller goes
    /// on to suggest. It names the working directory, its branch and the roots the file
    /// tools may touch, and every one of those has just changed; a model still told the
    /// old ones would be writing into a project it has left and having the write
    /// refused. It costs a re-prefill of the history, which is what keeping the history
    /// across the move is worth.
    ///
    /// The history itself is deliberately kept — `/cd` moves, it does not start over —
    /// and that is the one thing the caller has to say out loud, because a conversation
    /// about the last project is still in front of the model.
    pub(crate) fn change_directory(&mut self, path: &Path) -> Result<PathBuf, String> {
        let requested = expand_home(&path.to_string_lossy());
        std::env::set_current_dir(&requested).map_err(|e| {
            format!("Could not move to {}: {}", display_path(&absolute_path(&requested)), e)
        })?;

        // Read back rather than kept from above, and the `\\?\` prefix is why. Handing
        // `set_current_dir` the canonicalized path leaves the process reporting that form
        // for the rest of the run, and `session_file_path` hashes whatever
        // `current_dir` says — so the same directory would answer to two different
        // session files depending on whether haico started here or walked here. Letting
        // the OS name it means everything downstream agrees with a fresh start.
        let target = std::env::current_dir()
            .map_err(|e| format!("Moved, but cannot tell where to: {}", e))?;

        // Where a session is saved is keyed by directory, so leaving this pointing at
        // the old one would file the rest of this conversation under the project the
        // user has just left. Nulled in a test binary for `Agent::new`'s reason.
        self.session_path = if cfg!(test) { None } else { session_file_path() };

        // Consent was given about a project, not about a machine: "yes, run `cargo test`"
        // was an answer about this code, and carrying it into somewhere else would run a
        // command the user approved against a repository they have never seen. Unlike
        // `clear`, which keeps them — clearing a conversation does not move the boundary
        // they were given inside of.
        self.permissions = PermissionManager::new();

        self.workspace.set_cwd(&target);
        self.project_context = load_project_context();
        self.memory_index = memory::index_for_cwd();
        self.skill_index = skills::index_for_cwd();
        self.library_index = self.server.library_root.as_deref().and_then(crate::library::index_block);
        self.rebuild_system_prompt();
        Ok(target)
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

    /// Re-read the memory index and put it in the prompt.
    ///
    /// Called where the history has just been summarized away. Up to that point a fact
    /// recorded this session was in the conversation itself and the prompt did not need
    /// to carry it; afterwards the conversation is a paragraph of notes, and the index
    /// is the only thing that still knows. Cheap here and nowhere else: compaction has
    /// already invalidated the server's prefix cache, so rewriting the system message
    /// costs a re-prefill that was going to happen regardless.
    pub(crate) fn refresh_memory(&mut self) {
        self.memory_index = memory::index_for_cwd();
        self.skill_index = skills::index_for_cwd();
        self.library_index = self.server.library_root.as_deref().and_then(crate::library::index_block);
        self.rebuild_system_prompt();
    }

    /// Re-read the skills index, for `/update-skills`. Returns whether it changed.
    ///
    /// Only on request, and only then does it touch the history: rewriting the system
    /// message costs a re-prefill of the whole conversation, which is the user's call to
    /// make after adding a skill mid-session, not something to do behind their back.
    pub(crate) fn reload_skills(&mut self) -> bool {
        // First, and whatever the index says: a skills directory created since startup is
        // exactly the case this command is for, and its skills are no use if the files
        // beside their SKILL.md cannot be read.
        allow_user_skills(&mut self.workspace);
        let reloaded = skills::index_for_cwd();
        if reloaded == self.skill_index {
            return false;
        }
        self.skill_index = reloaded;
        self.rebuild_system_prompt();
        true
    }

    /// Point the conversation at a different model, re-resolving everything that
    /// is configured per model, and record it as the one to start with next time.
    ///
    /// Re-resolving is the whole point: a 131072-token budget carried onto a 32k
    /// model would warn far too late to be any use. The conversation itself is
    /// kept — switching model is not starting over.
    ///
    /// Saving happens here rather than in either frontend because there are two of
    /// them and this is the one place both go through; the same choice made in two
    /// places is the one that ends up made in one. The return value says whether it
    /// reached the file, for the caller to word — the switch itself has already
    /// happened either way.
    pub(crate) fn switch_model(&mut self, model: String) -> Kept {
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
        self.server.llm_client.model = model.clone();
        self.server.llm_client.apply_settings(&self.settings);

        // The reviewer is this model too, so its verdicts leave with it.
        self.review_cache.clear();
        // Counted by the old model's tokenizer.
        self.measured = None;
        self.rebuild_system_prompt();

        // Into the provider's own section when there is one: the model is that server's,
        // and the parsed file moves with the one on disk, as `set_context` keeps it.
        match self.config.provider.clone() {
            Some(provider) => {
                if let Some(section) = self.config.file.provider_sections.get_mut(&provider) {
                    section.model = Some(model.clone());
                }
                self.keep(|path| save_provider_model(path, &provider, &model))
            }
            None => self.keep(|path| save_default_model(path, &model)),
        }
    }

    /// The provider in force, if the config names any.
    pub(crate) fn provider(&self) -> Option<&str> {
        self.config.provider.as_deref()
    }

    /// The providers the config file lists, in its order.
    pub(crate) fn providers(&self) -> &[String] {
        &self.config.file.providers
    }

    /// Where a configured provider is, how to authenticate to it, and the model it
    /// starts with — what a frontend needs to choose a model there before switching.
    pub(crate) fn provider_section(&self, name: &str) -> Option<&crate::config::ProviderFile> {
        self.config.file.provider_sections.get(name)
    }

    /// Move the conversation to another configured provider and `model` on it, and
    /// record both as the ones to start with next time.
    ///
    /// The model is chosen by the caller, because when the provider names none that is
    /// a question for the user about the new server's list. The conversation is kept,
    /// as `switch_model` keeps it: changing server is not starting over. The provider's
    /// url and token replace whatever the command line said — switching is the user
    /// saying where to go now.
    pub(crate) fn switch_provider(&mut self, name: &str, model: String) -> Kept {
        let Some(section) = self.config.file.provider_sections.get(name).cloned() else {
            return Kept::Failed(format!("there is no provider named '{name}'"));
        };
        let url = section.url.unwrap_or_default();
        let token = section.token.filter(|t| !t.trim().is_empty());

        self.config.provider = Some(name.to_string());
        self.config.llm_url = url.clone();
        self.config.token = token.clone();
        self.config.url_supplied = true;
        self.config.file.provider = Some(name.to_string());
        // Both copies, for the same reason as the model: `web_fetch` summarizes
        // through the server's, and leaving it behind would send half the work to
        // the old server.
        let base_url = crate::llm::normalize_base_url(&url);
        self.client.base_url = base_url.clone();
        self.client.token = token.clone();
        self.server.llm_client.base_url = base_url;
        self.server.llm_client.token = token;

        let model_kept = self.switch_model(model);
        let provider_kept = self.keep(|path| save_active_provider(path, name));
        match (model_kept, provider_kept) {
            (Kept::Failed(why), _) | (_, Kept::Failed(why)) => Kept::Failed(why),
            (kept, _) => kept,
        }
    }

    /// Put a change in the config file, so it is still true next run.
    ///
    /// Writing the file is not printing, so the core may do it; saying whether it
    /// worked is the frontend's, which is what `Kept` is carried back for. No
    /// config path means no home directory to keep one in — reported rather than
    /// swallowed, because a setting the user was told would last and did not is
    /// worse than one they know never made it.
    fn keep(&self, write: impl FnOnce(&Path) -> Result<()>) -> Kept {
        let Some(path) = self.config.config_path.as_deref() else {
            return Kept::Failed("haico found no home directory to keep a config in".to_string());
        };
        match write(path) {
            Ok(()) => Kept::Config,
            Err(e) => Kept::Failed(format!("{:#}", e)),
        }
    }

    /// The model currently in use.
    /// How many tokens the next request will take, as nearly as can be known: what the
    /// server counted for the last one, plus an estimate of only what has been added since.
    ///
    /// The estimate alone (characters over four, messages only) was what this used to be,
    /// and it ran 20% low on a p2p_auth run — Rust and JSON-escaped tool output tokenize
    /// nearer 3.3 characters to the token, and the tools array, sent with every request,
    /// was not counted at all. Auto mode compacts at 90% of the window by this number, so
    /// the history grew to 130,820 of 131,072 tokens with the estimate still at 107k: the
    /// reply was cut off mid-tool-call and the next request was refused outright.
    ///
    /// The measurement is trusted only while the history has grown since it was taken. A
    /// history that shrank (compaction, `/clear`) is a different history, and falls back to
    /// the estimate, with the tools array counted, until the next reply measures it again.
    pub(crate) fn context_used(&self) -> usize {
        let estimate = estimate_tokens(&self.messages);
        match self.measured {
            Some(Measured { prompt_tokens, estimate: then }) if estimate >= then => prompt_tokens + (estimate - then),
            _ => estimate + estimate_tokens(&self.tools),
        }
    }

    /// Whether `context_used` rests on a measurement right now, for saying so.
    pub(crate) fn context_is_measured(&self) -> bool {
        self.measured.is_some_and(|m| estimate_tokens(&self.messages) >= m.estimate)
    }

    pub(crate) fn model(&self) -> &str {
        &self.client.model
    }

    /// The plan, as the todo tool last left it.
    pub(crate) fn plan(&self) -> Vec<crate::tools::TodoItem> {
        self.server.todos.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// @mentions
// ---------------------------------------------------------------------------

/// Put the files a message named with `@` into the message, before it is sent.
///
/// The saving is a whole round. Left to itself the model reads the name, emits a
/// `read` call, waits for the server to prefill the reply and only then starts
/// working; on a local model that is seconds to minutes spent fetching something
/// the user already knew they wanted. Which file it is is not a judgment call
/// here — the user typed the path.
///
/// Three things are worth knowing about what this does beyond concatenating:
///
/// - **It reads through the `read` tool rather than `fs::read_to_string`.** That
///   is what gets the line-number prefixes the edit tool's callers expect, the
///   2000-line cap, the "not a text file" message for a binary, and — the part
///   that matters most — `mark_read`. Without that last one the model is handed
///   the contents and then refused the edit, because `check_may_modify` has
///   never seen the file this session. The turn saved by attaching would be
///   spent re-reading it, which is the whole feature undone.
/// - **A file outside the workspace widens it, to that one file.** Typing the
///   path is the consent; see `Workspace::attach` for why it stays one file.
///   Nothing has to be done to tell the model about it: `with_mode_note` builds
///   the workspace note from the workspace itself on every request, so a file
///   attached here is named in the one that follows. The system prompt does not
///   carry the boundary and is deliberately left alone.
/// - **The blocks go in front of the user's own words, not after.** The request
///   is what the model should read last, and a 2000-line attachment between the
///   instruction and the end of the message is a long way for a small model to
///   hold it. The `@name` is still in the text, and each block names itself by
///   that as well as by the full path, so the two ends tie together.
///
/// A mentioned **image** takes the other path out of this function. It cannot go
/// in the text, so the message stops being text: `content` becomes the
/// OpenAI-compatible parts array, the image parts first and one text part
/// carrying everything above. That shape is only used when there is an image in
/// it — a plain string is what every server handles, and there is no reason to
/// make one parse an array to be told a sentence.
///
/// Whether the model can *see* the image is deliberately not asked. There is no
/// reliable way to know from here: the name says nothing dependable, and no
/// OpenAI-compatible endpoint reports it. So the image is sent, and a text-only
/// model's server says so in its own words — which is a clearer answer than a
/// guess of haico's would be.
async fn attach_mentions(agent: &mut Agent, input: &str, tx: &EventSink) -> Value {
    let mentions = crate::mention::find(input);
    if mentions.is_empty() {
        return json!(input);
    }

    let mut blocks = String::new();
    let mut images: Vec<Value> = Vec::new();

    for mention in mentions {
        let shown = display_path(&mention.path);

        // Sniffed from the bytes, so a JPEG that someone renamed to .png is
        // still sent as a JPEG. See `mention::image_type`.
        if crate::mention::image_type_of(&mention.path).is_some() {
            match crate::mention::sendable_image(&mention.path) {
                Some((mime, bytes)) => {
                    let outside = agent.workspace.attach(&mention.path);
                    let _ = tx.send(
                        Notice::ImageAttached { path: shown, bytes: bytes.len(), outside }.into(),
                    );
                    images.push(json!({
                        "type": "image_url",
                        "image_url": { "url": data_url(&mime, &bytes) }
                    }));
                }
                None => {
                    let _ = tx.send(
                        Notice::FileNotAttached {
                            path: shown,
                            reason: "the file could not be read".to_string(),
                        }
                        .into(),
                    );
                }
            }
            continue;
        }

        let read = agent
            .server
            .read(Parameters(ReadParams {
                file_path: mention.path.display().to_string(),
                offset: None,
                limit: None,
            }))
            .await;

        let text = match read {
            Ok(result) if result.is_error != Some(true) => result
                .content
                .iter()
                .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            // Binary, unreadable, or gone in the moment between the parse and
            // now. Nothing is attached and the `@name` stays in the text as
            // written, so the model can still call `read` on it and be told the
            // same thing through the channel that knows how to say it.
            Ok(result) => {
                let reason = result
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                // The tool quotes back the path it was handed, which is the
                // verbatim form because that is what the workspace compares
                // against. `display_path` cannot reach it there — it strips a
                // prefix, and this one is in the middle of a sentence.
                let reason = reason.replace(r"\\?\", "");
                let _ = tx.send(Notice::FileNotAttached { path: shown, reason }.into());
                continue;
            }
            Err(e) => {
                let _ = tx
                    .send(Notice::FileNotAttached { path: shown, reason: format!("{:?}", e) }.into());
                continue;
            }
        };

        // After the read, not before: a file that turned out to be unreadable has
        // no business having widened the boundary on its way to failing.
        let outside = agent.workspace.attach(&mention.path);

        let _ = tx.send(
            Notice::FileAttached { path: shown.clone(), lines: text.lines().count(), outside }
                .into(),
        );

        blocks.push_str(&format!(
            "Contents of {}, referenced as {} in the message below:\n{}\n\n",
            shown, mention.written, text
        ));
    }

    let text = if blocks.is_empty() { input.to_string() } else { format!("{}{}", blocks, input) };

    if images.is_empty() {
        return json!(text);
    }

    // Images first and the text last, for the same reason the text blocks go
    // above the request: what the model is being asked to do should be the last
    // thing it reads.
    let mut parts = images;
    parts.push(json!({"type": "text", "text": text}));
    json!(parts)
}

/// Say something to the model in the user's voice, from haico rather than from
/// the user.
///
/// Folded into the last message when that is already the user's, because the
/// first round of a turn ends on the user's own message and two user messages in
/// a row is an order some chat templates refuse outright. Anywhere else — after
/// tool results — it is a message of its own, as the loop nudge is. The last
/// message is the tail of the prefix the server caches, so extending it
/// invalidates nothing that was not about to be re-sent anyway.
// Said once, to a reply that called nothing straight after an auto-compaction (see
// `just_compacted` in `run_turn`).
const AFTER_COMPACTION: &str = "You replied without calling a tool, right after your context was \
     compacted in the middle of the task. The work is not finished: carry on from the notes above \
     by calling a tool now.";

pub(crate) fn push_user_note(messages: &mut Vec<Value>, note: &str) {
    if let Some(last) = messages.last_mut().filter(|m| m["role"] == "user") {
        match &mut last["content"] {
            Value::String(text) => {
                text.push_str("\n\n");
                text.push_str(note);
                return;
            }
            Value::Array(parts) => {
                parts.push(json!({ "type": "text", "text": note }));
                return;
            }
            _ => {}
        }
    }
    messages.push(json!({ "role": "user", "content": note }));
}

/// Take every picture out of the history, leaving a line of text where each one
/// was. Returns how many went.
///
/// A message stripped down to nothing but images does not become an empty
/// message — it becomes the note, because a `content` of `[]` is a message some
/// servers reject and all of them find strange, and because the model should be
/// told the picture is gone rather than left wondering why its own question no
/// longer makes sense.
///
/// Not a general-purpose history edit. This runs at the top of a round, where
/// every tool call already has its reply, and it never removes a message — so no
/// `tool_call_id` can be orphaned by it.
fn strip_images(messages: &mut [Value]) -> usize {
    let mut removed = 0;

    for message in messages.iter_mut() {
        let Some(parts) = message["content"].as_array() else { continue };
        let images = parts.iter().filter(|part| part["type"] == "image_url").count();
        if images == 0 {
            continue;
        }
        removed += images;

        let mut kept: Vec<Value> = parts
            .iter()
            .filter(|part| part["type"] != "image_url")
            .cloned()
            .collect();
        kept.push(json!({
            "type": "text",
            "text": format!(
                "[{} image(s) were attached here and have been removed: the server would \
                 not accept them, so this model cannot see images. Work from the text.]",
                images
            )
        }));
        message["content"] = json!(kept);
    }

    removed
}

/// One image as the `data:` URL an OpenAI-compatible server expects.
///
/// A URL and not a file path, because the server is a separate process and may
/// be a separate machine — a path it could not open is the one thing this must
/// never send.
fn data_url(mime: &str, bytes: &[u8]) -> String {
    use base64::Engine;
    format!("data:{};base64,{}", mime, base64::engine::general_purpose::STANDARD.encode(bytes))
}

// A stuck model gives itself away by calling the same tool with the same arguments over and
// over with nothing in between, which can happen on the tenth round or the fiftieth. The first
// threshold says so in the conversation; the second gives up on the turn. Shared with the
// sub-agent's loop, which gets stuck the same way.
pub(crate) const LOOP_NUDGE: usize = 3;
pub(crate) const LOOP_BREAK: usize = 5;

// ---------------------------------------------------------------------------
// Clearing one call
// ---------------------------------------------------------------------------

/// What a call that may run carries with it to the call line.
pub(crate) struct Cleared {
    /// Why it runs unasked, when the auto-mode reviewer is the one who said yes.
    pub(crate) auto_note: Option<String>,
    /// The user was prompted, and the prompt has already shown the diff.
    pub(crate) prompted: bool,
}

/// Every gate one call has to pass before it runs: the mode, the workspace, the
/// permission rules, the auto-mode reviewer and the user, in that order.
///
/// `Err` carries the reply the call's `tool_call_id` is owed instead of a result;
/// the caller pushes it, because the caller knows which message list it belongs
/// in. A function rather than a block of `run_turn` because the sub-agent
/// (`crate::subagent`) makes calls too, and a call made one level down is still a
/// call on this machine: the same boundary, the same prompt, the same reviewer.
/// Two copies of this would be two answers to "may this run".
pub(crate) async fn authorize(
    agent: &mut Agent,
    tx: &EventSink,
    link: &TurnLink,
    tool_name: &str,
    arguments: &Value,
) -> std::result::Result<Cleared, String> {
    // Decide what this specific call needs: nothing (read-only), the user's consent under
    // a narrowly scoped key, or an outright refusal. The mode has the first word — plan
    // mode refuses anything that changes the machine, whatever the call would normally be
    // entitled to.
    let gate = gate_for_mode(link.mode(), &agent.workspace, tool_name, arguments);

    if let Gate::Deny(reason) = &gate {
        let _ = tx.send(Notice::Refused(reason.clone()).into());
        return Err(reason.clone());
    }

    let mut ask = match &gate {
        Gate::Ask { keys, label } if !agent.permissions.is_allowed(keys) => {
            Some((keys.clone(), label.clone()))
        }
        _ => None,
    };

    // Auto mode: the reviewer answers the routine yeses so the user does not have to.
    // Only a call that was going to interrupt is reviewed — an already-approved one costs
    // nothing and should stay free.
    //
    // An allow runs the call unasked, and a deny refuses it unasked. The deny used to fall
    // through to the prompt instead, which made auto mode useless for the one thing it is
    // for: handing over a task and walking away. A single suspicious call parked the whole
    // run on a dialog nobody was there to answer. Refusing costs little — the model is
    // told why and can find another way or stop and say what it needed.
    //
    // A review with no verdict (cut off at its token cap, an unreadable reply, a server
    // error) is not a judgment either way, and for the same reason it no longer falls
    // through to the prompt: nobody may be there. The model is told it was a temporary
    // error and to make the call again. Worded as a refusal, it was taken as final: an
    // unattended model whose `write_file` met a review cut off at its cap spent the next
    // forty minutes writing files through `echo` and PowerShell. If the reviewer keeps
    // failing, the identical-call loop guard bounds the retries.
    //
    // Only a review the user interrupted with Esc still reaches the prompt: that is the
    // one outcome that proves somebody is there.
    // Why the call is running unasked, and what to add to the prompt when it still has to
    // be asked. Only one is ever set.
    let mut auto_note: Option<String> = None;
    let mut review_note: Option<ReviewNote> = None;
    if link.mode() == Mode::Auto && ask.is_some() {
        // Transient: a review takes seconds, and a silent pause reads as a hang. Erased
        // once it lands, because the outcome belongs on the call line below — left
        // standing, every single call would cost two lines of log to say one thing.
        let verdict = interruptible(
            tx,
            link,
            Waiting::ReviewingCall,
            judge_tool_call(
                &agent.client,
                &mut agent.review_cache,
                &agent.workspace,
                tool_name,
                arguments,
            ),
        )
        .await;

        match verdict {
            Some(Ok(Verdict { allow: true, reason, cached })) => {
                // Marked, so reading back through a run still tells apart a call the
                // reviewer weighed from one it merely recognized.
                auto_note = Some(match cached {
                    true => format!("cached — {}", reason),
                    false => reason,
                });
                ask = None;
            }
            Some(Ok(Verdict { allow: false, reason, .. })) => {
                let _ = tx.send(
                    Notice::ReviewDenied {
                        tool: tool_name.to_string(),
                        arguments: arguments.clone(),
                        reason: reason.clone(),
                    }
                    .into(),
                );
                return Err(format!(
                    "Permission denied: this call was refused as unsafe ({}). It was not run. \
                     Do not retry it as is. Reach the goal a safer way if there is one; \
                     otherwise finish and tell the user what you needed done and why.",
                    reason
                ));
            }
            Some(Err(e)) => {
                let detail = format!("{:#}", e);
                let _ = tx.send(
                    Notice::ReviewFailed {
                        tool: tool_name.to_string(),
                        arguments: arguments.clone(),
                        detail,
                    }
                    .into(),
                );
                return Err(format!(
                    "Temporary error while evaluating the '{}' call, so it was not run. \
                     Nothing is known to be wrong with the call itself: make the same call \
                     again.",
                    tool_name
                ));
            }
            None => {
                review_note = Some(ReviewNote::Interrupted);
            }
        }
    }

    // Captured before the prompt consumes it: the diff under the call line is skipped
    // only when the prompt has already shown it.
    let prompted = ask.is_some();

    if let Some((gate_keys, allow_all_label)) = ask {
        let diff_content: Option<String> = file_change_diff(tool_name, arguments);

        // The call itself goes to the frontend, not a sentence about it: how much of a
        // command to show and where the reviewer's objection belongs are rendering
        // decisions.
        //
        // A terminal we cannot read from (piped stdin, no TTY) must not end the session,
        // and must never be taken as consent: refuse the call and tell the model.
        let answered = ask_frontend(tx, |reply| Ask::Permission {
            tool_name: tool_name.to_string(),
            arguments: arguments.clone(),
            review: review_note.clone(),
            allow_all_label: allow_all_label.clone(),
            diff: diff_content.clone(),
            reply,
        })
        .await;
        let Some(response) = answered else {
            let _ = tx.send(
                Notice::AskFailed {
                    what: "for permission",
                    detail: "no interactive terminal".to_string(),
                }
                .into(),
            );
            return Err(format!(
                "Permission for '{}' could not be requested (no interactive terminal), so it \
                 was not run.",
                tool_name
            ));
        };

        match response {
            PermissionResponse::Allow => {
                let _ = tx.send(Notice::PermissionAllowed.into());
            }
            PermissionResponse::AllowAll => {
                // Remembered under the narrow keys this call resolved to, not under the
                // tool name.
                agent.permissions.allow(&gate_keys);
                let _ = tx.send(Notice::PermissionAllowedAll(allow_all_label.clone()).into());
            }
            PermissionResponse::Reject(alternative) => {
                let _ = tx.send(Notice::PermissionDenied { alternative: alternative.clone() }.into());
                return Err(format!("Permission denied by user. User suggests: {}", alternative));
            }
        }
    }

    Ok(Cleared { auto_note, prompted })
}

// ---------------------------------------------------------------------------
// One turn
// ---------------------------------------------------------------------------

/// One turn: the user's message in, and everything that follows from it — rounds
/// of tool calls, the prompts they need answering, the reply as it streams.
///
/// It says what happened through `tx` and shares three things through `link`; it
/// holds no terminal, no keyboard and no renderer, and it must stay that way.
/// See the module header.
pub(crate) async fn run_turn(
    agent: &mut Agent,
    input: &str,
    // True when this turn is `/init` writing the project instruction file, so the
    // file it produces can be picked up at the end of it.
    initializing: bool,
    tx: &EventSink,
    link: &TurnLink,
) -> TurnEnd {
        // Wall clock, started here rather than at the first request: what the summary at
        // the end of the turn reports is how long the user waited, which includes the tool
        // runs, the permission prompts they answered and every round of the loop below —
        // not just the time the model spent generating.
        let turn_started = std::time::Instant::now();

        // Add user message, with any `@path` in it already read into it.
        let content = attach_mentions(agent, input, tx).await;
        agent.messages.push(json!({
            "role": "user",
            "content": content
        }));

        // Get LLM response and handle tool call chains. The cap stops a model that keeps
        // calling tools forever from spinning until the user kills the process. It does
        // not apply in auto mode: a fixed number of rounds is a poor stand-in for "this is
        // going nowhere", and a long job that hits it unattended has nobody to say
        // "continue". Repetition is the better signal, and it is what guards auto mode.
        const MAX_TOOL_ROUNDS: usize = 50;
        let mut round = 0usize;

        // The (tool, arguments) pair called last, and how long the unbroken run of it is.
        // Consecutive rather than counted across the turn: a model working through a list
        // of compiler errors legitimately runs the same build command a dozen times in one
        // turn, and every one of those is progress because there were edits in between. A
        // loop is the same call with nothing else happening between the tries. Any
        // different call resets the run, `nudged_at` with it, so a later loop still gets
        // its warning.
        let mut last_call: Option<(String, String)> = None;
        let mut run = (0usize, String::new());
        let mut nudged_at = 0usize;

        // Cleared when compacting fails to get the history back under the line, which
        // means the window is too small to hold even a summary of it. Trying again the
        // next round would summarize the summary and throw the round away with it, so
        // the turn runs on instead and lets the server error be the honest end of it.
        let mut autocompact = true;

        // Set by an auto-compaction and cleared by the next reply that calls a tool. While it
        // is set, a reply that calls nothing does not end the turn: the task was mid-flight
        // when the history was replaced, and the loop has just said so. On a p2p_auth run the
        // first reply after a compaction was one sentence of intent ("Let me check the current
        // state of the codebase") with no tool call, the turn ended on it, and the unattended
        // run exited as finished with 19 compile errors in its tests.
        let mut just_compacted = false;

        // Set once the recovery below has run, so a failure that has nothing to do with
        // images cannot make the turn retry forever.
        let mut stripped_images = false;

        // Replies in a row that were cut off for repeating themselves (see `repetition`).
        // In a row, not per turn: an auto-mode turn lives for hours, and a model that fell
        // into a loop once, was told, and went on to do an hour of real work has earned the
        // same second chance the next time. Two in a row means the nudge is not landing, and
        // a third try would most likely be the same loop at the same cost.
        const MAX_REPEATING_REPLIES: usize = 2;
        let mut repeating_replies = 0usize;

        let end = loop {
            round += 1;
            // Once a round, because that is when the history has changed and the
            // status bar is what reports it. The numbers, not the line: how wide
            // the bar is and what goes where is the frontend's business.
            let _ = tx.send(AgentEvent::Status {
                provider: agent.provider().map(str::to_string),
                model: agent.client.model.clone(),
                used_tokens: agent.context_used(),
                context_tokens: agent.context_tokens,
                effort: agent.reasoning_effort().map(str::to_string),
            });
            // Re-read every round rather than captured once: approving a plan can switch
            // into auto mode mid-turn, and the cap should stop applying from that point.
            if link.mode() != Mode::Auto && round > MAX_TOOL_ROUNDS {
                let _ = tx.send(Notice::RoundCapReached(MAX_TOOL_ROUNDS).into());
                break TurnEnd::Failed;
            }

            // Auto mode is meant to run unattended, so the one thing it must not do at the
            // edge of the window is stop and ask for /compact: there is nobody there to
            // type it, and a job left running overnight would be found dead at the prompt.
            // Checked here rather than at the end of a turn, where the old hint printed,
            // because a long job lives inside this loop for hours and may never reach the
            // end of a turn at all. The history is balanced at this point — every tool call
            // from the round before already has its reply — so replacing it wholesale
            // cannot orphan a tool_call_id, which is not true further down.
            if link.mode() == Mode::Auto
                && autocompact
                && agent.context_used() > autocompact_threshold(agent.context_tokens)
            {
                let (context_tokens, before) = (agent.context_tokens, agent.context_used());
                let compacted = interruptible(
                    tx,
                    link,
                    Waiting::Compacting,
                    compact_history(&agent.client, &mut agent.messages, context_tokens, before),
                )
                .await;

                match compacted {
                    Some(Ok(report)) => {
                        // The same reclaim /compact does, for the same reason: the cache is
                        // keyed to a history that no longer exists.
                        agent.review_cache.clear();
                        agent.refresh_memory();
                        let _ = tx.send(Notice::Compacted(report).into());
                        // The compacted history ends on the assistant agreeing that it has
                        // the notes, and asking for a reply straight after that is asking
                        // the model to talk past itself. This says the one thing the notes
                        // cannot: the task was mid-flight when the summarizer ran, and it
                        // is the same task that continues.
                        just_compacted = true;
                        agent.messages.push(json!({
                            "role": "user",
                            "content": "Your context was compacted mid-task. Continue the \
                                        work described in the notes above from where it left \
                                        off; do not start over."
                        }));

                        if agent.context_used() > autocompact_threshold(agent.context_tokens) {
                            autocompact = false;
                            let _ = tx.send(
                                Notice::CompactionStillLarge((AUTOCOMPACT_FRACTION * 100.0) as usize)
                                    .into(),
                            );
                        }
                    }
                    // Not fatal by itself: the request below may still fit, and if it does
                    // not, the server's own error ends the turn — exactly what would have
                    // happened with none of this here.
                    Some(Err(e)) => {
                        let _ = tx.send(Notice::CompactionFailed(format!("{:#}", e)).into());
                    }
                    // Esc during the summarization is Esc during the turn, as everywhere
                    // else in this loop.
                    None => {
                        let _ = tx.send(Notice::Interrupted.into());
                        break TurnEnd::Interrupted;
                    }
                }
            }


            // Whether anything has reached the screen for this reply, and the
            // half-rendered markdown line, both belong to the presenter — see
            // `Presenter::start_reply`. Per round, not per turn: a turn is many
            // rounds and each one streams its own reply.
            let _ = tx.send(AgentEvent::ReplyStart);
            // The count of characters streamed so far, which the activity bar
            // shows so a model that reasons for minutes before its first content
            // token is distinguishable from one that has hung, is now the pump's:
            // it sees every chunk go past and can count them itself.
            //
            // Rebuilt every round rather than once per turn: approving a plan switches the
            // mode mid-turn, and the very next request has to carry the full tool list.
            let request_messages = with_mode_note(link.mode(), &agent.workspace, &agent.messages);
            let request_tools = tools_for_mode(link.mode(), &agent.tools);
            // Of the history as it stands, not of `request_messages`: the mode note is not
            // kept, and `context_used` compares against the history it will be looking at.
            let estimate_sent = estimate_tokens(&agent.messages);

            // Esc abandons the turn instead of killing the process. It arrives at
            // the pump as a keystroke rather than as a signal, because raw mode is
            // on for the whole session — which is also what lets the next message
            // be typed while this one is still running — and reaches here as the
            // link's cancellation.
            let result = interruptible(
                tx,
                link,
                Waiting::Model,
                agent.client.chat_completion_measured(
                    Caller::Main,
                    request_messages,
                    request_tools,
                    // Both kinds of text are handed on the moment they arrive, so
                    // none of the branches below re-print anything from the
                    // finished message. What either one *looks* like — dim,
                    // italic, folded, markdown-wrapped — is the presenter's, and
                    // this no longer knows. The third carries no text at all,
                    // only how much of a tool call has landed: the call is shown
                    // whole when it is about to run, and once is enough.
                    |chunk| {
                        let _ = tx.send(match chunk {
                            Chunk::Reasoning(text) => AgentEvent::ReasoningChunk(text),
                            Chunk::Content(text) => AgentEvent::ReplyChunk(text),
                            Chunk::ToolFragment(characters) => {
                                AgentEvent::ToolCallChunk(characters)
                            }
                            // The failed attempt's reply is closed as one that ended, and
                            // the next attempt opens a reply of its own, so what it streams
                            // is never shown as the rest of the text the failure cut off.
                            // Nothing of it reaches the history: only the reply that comes
                            // back whole is returned.
                            Chunk::Retrying { error, retry, of, delay } => {
                                let _ = tx.send(AgentEvent::ReplyEnd);
                                let _ = tx.send(Notice::Retrying { error, retry, of, delay }.into());
                                AgentEvent::ReplyStart
                            }
                        });
                    },
                ),
            )
            .await;

            let Some(result) = result else {
                // The partial reply is dropped rather than stored: half a tool call in the
                // history is worse than no reply at all. Everything before this turn stays
                // consistent, so the conversation is still usable.
                let _ = tx.send(AgentEvent::ReplyEnd);
                let _ = tx.send(Notice::Interrupted.into());
                break TurnEnd::Interrupted;
            };

            let mut message = match result {
                Ok((message, ending)) => {
                    repeating_replies = 0;
                    if let Some(prompt_tokens) = ending.usage.as_ref().and_then(|u| u["prompt_tokens"].as_u64()) {
                        agent.measured = Some(Measured { prompt_tokens: prompt_tokens as usize, estimate: estimate_sent });
                    }
                    message
                }
                Err(e) => {
                    let _ = tx.send(AgentEvent::ReplyEnd);

                    // The stream was cut off because the reply had gone into a loop. The
                    // partial reply is dropped, exactly as an interrupt drops one — it is
                    // the loop, and half a tool call in the history is worse than none. The
                    // history is as consistent as it was before the request, so asking again
                    // is safe; asking again *unchanged* would mostly get the same loop back,
                    // which is why the model is shown what it was repeating first.
                    if let Some(repeating) = e.downcast_ref::<Repeating>() {
                        repeating_replies += 1;
                        let retrying = repeating_replies < MAX_REPEATING_REPLIES;
                        let _ = tx.send(
                            Notice::Repeating { passage: repeating.passage.clone(), retrying }.into(),
                        );
                        if !retrying {
                            break TurnEnd::Failed;
                        }
                        push_user_note(
                            &mut agent.messages,
                            &format!(
                                "Your previous reply was cut off and discarded because it had \
                                 started repeating the same passage over and over:\n\n> {}\n\n\
                                 Do not go back over that reasoning again. Decide on one \
                                 concrete next step and take it now — call a tool, try a \
                                 different approach, or say plainly what you are stuck on.",
                                repeating.passage
                            ),
                        );
                        continue;
                    }

                    // An image in the history is the one failure this can back
                    // out of, and the only one worth trying to: a model without
                    // a vision head rejects the request that carries a picture
                    // and every request after it, so the conversation is dead
                    // until the picture leaves. That is a real risk here and not
                    // a theoretical one, because `read` lets the *model* choose
                    // the file — it can point itself at a screenshot with no
                    // way to know whether it can see one.
                    //
                    // Once per turn, so an unrelated failure that keeps failing
                    // still ends at the prompt instead of looping. The images
                    // are gone from the history for good, which is the point:
                    // retrying with them in would fail the same way.
                    if !stripped_images && strip_images(&mut agent.messages) > 0 {
                        stripped_images = true;
                        let _ = tx.send(Notice::ImagesStripped(format!("{:#}", e)).into());
                        continue;
                    }

                    // A dropped connection, a server-side error, or an unparseable body
                    // must not end the session. Every tool call made so far already has its
                    // reply, so the history is consistent: drop back to the prompt.
                    let _ = tx.send(Notice::RequestFailed(format!("{:#}", e)).into());
                    break TurnEnd::Failed;
                }
            };

            // Both kinds of text were handed on as they arrived — including the case of a
            // model whose entire reply is chain of thought with no content at all, which
            // is why nothing is re-emitted from `message` here. All that is left is to say
            // the reply is over, and let the presenter close whatever it has open.
            let _ = tx.send(AgentEvent::ReplyEnd);

            // The one place the reasoning policy is applied, deliberately ahead of every
            // `agent.messages.push(message.clone())` below rather than at each of them: all three
            // push this same value, and a policy enforced at two of the three sites would
            // leave whole traces in the history depending on how the turn happened to end.
            //
            // Only the compression of a long enough trace costs a request, and only that
            // goes through `work_with_input` — several seconds with a frozen input box
            // every turn would be worse than the tokens are worth. Esc during it means
            // `None`, which lands in the same place a failure does: the original chain of
            // thought is kept and the turn carries on. Dropping a field, and declining a
            // trace under the threshold, are both over before a frame could be painted, so
            // they are awaited directly rather than announced as work in progress.
            if agent.reasoning != Reasoning::Keep {
                let note = if reasoning_needs_a_request(&message, agent.reasoning, agent.reasoning_threshold) {
                    interruptible(
                        tx,
                        link,
                        Waiting::CompressingReasoning,
                        apply_reasoning(&agent.client, &mut message, agent.reasoning, agent.reasoning_threshold),
                    )
                    .await
                    .flatten()
                } else {
                    apply_reasoning(&agent.client, &mut message, agent.reasoning, agent.reasoning_threshold).await
                };

                match note {
                    // Shown rather than folded, and wearing the chrome's light blue: the
                    // whole point of compressing is to find out whether it is paying for
                    // itself, which cannot be judged from a line that is hidden by
                    // default. Tinted, not painted — the bars are the only thing on
                    // screen that gets a background, and that is what makes them read as
                    // furniture rather than as output.
                    Some(ReasoningOutcome::Compressed { before, after }) => {
                        agent.reasoning_stats.record(ReasoningOutcome::Compressed { before, after });
                        // Read back out of the message rather than handed over with the
                        // figures: `apply_reasoning` has just written the summary into the
                        // reasoning field, and that field is the version the history now
                        // carries — reporting anything else would be reporting a text the
                        // conversation does not actually contain.
                        let summary = reasoning_text(&message).unwrap_or_default().to_string();
                        let _ = tx.send(Notice::ReasoningCompressed { before, after, summary }.into());
                    }
                    // Dropping says nothing the policy did not already say: it removes
                    // every trace, every turn, so a line per reply would be a line per
                    // reply saying the same thing.
                    Some(ReasoningOutcome::Dropped { tokens }) => {
                        let _ = tx.send(Notice::ReasoningDropped { tokens }.into());
                    }
                    None => {}
                }
            }

            // Check if model wants to call a tool
            if let Some(tool_calls) = message.get("tool_calls").filter(|c| !c.is_null()) {
                let tool_calls_array = match tool_calls.as_array() {
                    Some(arr) => arr,
                    None => {
                        let _ = tx.send(Notice::ToolCallsMalformed(format!("{:?}", tool_calls)).into());
                        break TurnEnd::Failed;
                    }
                };

                // Check if array is empty
                if tool_calls_array.is_empty() {
                    // No actual tool calls, this is the final response — already streamed.
                    agent.messages.push(message.clone());
                    if std::mem::take(&mut just_compacted) {
                        push_user_note(&mut agent.messages, AFTER_COMPACTION);
                        continue;
                    }
                    break TurnEnd::Answered;
                }
                just_compacted = false;

                // Add assistant message with tool call first
                agent.messages.push(message.clone());

                // Set when Ctrl+C lands during a tool call, so the turn ends rather than
                // only the tool loop.
                let mut interrupted = false;

                // Pictures the round's tools returned, waiting for the one place
                // an image is allowed to go. See where they are pushed below.
                let mut seen_images: Vec<Value> = Vec::new();

                for tool_call in tool_calls_array {

                    // Captured up front: every path out of this iteration owes the model a
                    // `role:"tool"` message under this id. Leaving one unanswered makes the
                    // next request fail server-side, which is far harder to recover from
                    // than telling the model its call was malformed.
                    let tool_call_id = tool_call.get("id").cloned().unwrap_or(Value::Null);
                    let function = &tool_call["function"];

                    let tool_name = match function["name"].as_str() {
                        Some(name) => name,
                        None => {
                            let detail = format!(
                                "Malformed tool call: no function name in {}",
                                truncate_at_boundary(&function.to_string(), 500)
                            );
                            let _ = tx.send(Notice::ToolCallUnnamed(detail.clone()).into());
                            agent.messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": detail
                            }));
                            continue;
                        }
                    };

                    let parsed = parse_tool_arguments(function);

                    // Counted here, before any of the branches below, so a call that is
                    // refused, denied *or malformed* counts too — retrying a call the
                    // workspace or the user has already turned down, unchanged and
                    // immediately, is a loop like any other. What it returned is
                    // deliberately not part of the key: a build that fails three times in a
                    // row with nothing done between the tries is stuck whatever the errors
                    // say.
                    //
                    // The malformed case has to be counted *before* the parse, not after,
                    // and that is not tidiness. The error arm below `continue`s, so
                    // counting after it meant a model repeating a call whose arguments are
                    // not JSON never advanced the run at all. In normal mode the round cap
                    // still stopped it; in auto mode, which skips that cap on purpose
                    // because there is nobody there to type "continue", nothing stopped it
                    // and the turn span forever unattended. Keyed on the canonical form
                    // when it parses (serde_json sorts object keys, so the same call always
                    // renders to the same string) and on the raw argument text when it does
                    // not, which is the only form a malformed call has.
                    let args_key = match &parsed {
                        Ok(arguments) => arguments.to_string(),
                        Err(_) => function.get("arguments").map(|a| a.to_string()).unwrap_or_default(),
                    };
                    let call = (tool_name.to_string(), args_key);
                    if last_call.as_ref() == Some(&call) {
                        run.0 += 1;
                    } else {
                        last_call = Some(call);
                        run = (1, tool_name.to_string());
                        nudged_at = 0;
                    }

                    let mut arguments = match parsed {
                        Ok(arguments) => arguments,
                        Err(detail) => {
                            let _ = tx.send(Notice::ToolArgumentsInvalid { tool: tool_name.to_string(), detail: detail.clone() }.into());
                            agent.messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": format!(
                                    "Could not run '{}': {}. Call it again with valid JSON arguments.",
                                    tool_name, detail
                                )
                            }));
                            continue;
                        }
                    };

                    // Before anything looks at the arguments, so the gate, the diff preview
                    // and the tool all see what the model meant rather than what it typed
                    // (see `repair_leaked_arguments`). Said in the tool's reply: a model
                    // that is never told keeps making the same slip, and this one made it
                    // in half the edits of a run.
                    let repaired = repair_leaked_arguments(&mut arguments);

                    // The model can put itself into plan mode, for when the user asked for
                    // a plan rather than for the work. Nothing to approve: this only ever
                    // takes capability away, and present_plan is where consent is needed.
                    if tool_name == START_PLAN {
                        let content = if link.mode() == Mode::Plan {
                            "Already in plan mode. Investigate with the reading tools, then \
                             call present_plan."
                        } else {
                            link.set_mode(Mode::Plan);
                            let _ = tx.send(Notice::PlanModeEntered.into());
                            "Plan mode is active. The tools that change this machine are no \
                             longer available to you. Investigate the real code with read, \
                             grep, ls and glob_search, then call present_plan with what you \
                             propose. Do not start work until the user approves it."
                        };
                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content
                        }));
                        continue;
                    }

                    // A finished plan is not a tool call so much as the end of plan mode,
                    // and the mode it ends lives out here rather than inside the tool.
                    if tool_name == PRESENT_PLAN {
                        let plan = arguments["plan"].as_str().unwrap_or("").trim().to_string();
                        if plan.is_empty() {
                            agent.messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": "The plan was empty. Put the plan itself in the \
                                            'plan' field and call present_plan again."
                            }));
                            continue;
                        }

                        // A terminal that cannot be read from must not be taken as
                        // approval — the same rule the permission prompt follows.
                        let decision =
                            ask_frontend(tx, |reply| Ask::Plan { plan: plan.clone(), reply }).await;
                        let content = match decision {
                            Some(PlanDecision::Approve) => {
                                link.set_mode(Mode::Normal);
                                let _ = tx.send(Notice::PlanApproved { auto: false }.into());
                                "The user approved the plan. Plan mode is over and the full \
                                 tool set is available again. Begin work now, following the \
                                 plan you just had approved."
                                    .to_string()
                            }
                            Some(PlanDecision::ApproveWithAuto) => {
                                link.set_mode(Mode::Auto);
                                let _ = tx.send(Notice::PlanApproved { auto: true }.into());
                                // The model is told nothing about the review. It should
                                // propose the same call either way, and knowing it is being
                                // checked is an invitation to write for the checker.
                                "The user approved the plan. Plan mode is over and the full \
                                 tool set is available again. Begin work now, following the \
                                 plan you just had approved."
                                    .to_string()
                            }
                            Some(PlanDecision::Revise(feedback)) => {
                                let _ = tx.send(Notice::PlanRejected.into());
                                format!(
                                    "The user did not approve the plan. Their corrections: {}\n\n\
                                     You are still in plan mode and must not change anything. \
                                     Fold this in — investigating further if you need to — and \
                                     call present_plan again.",
                                    feedback
                                )
                            }
                            None => {
                                let _ = tx.send(
                                    Notice::AskFailed {
                                        what: "about the plan",
                                        detail: "no interactive terminal".to_string(),
                                    }
                                    .into(),
                                );
                                "The plan could not be shown for approval (no interactive \
                                 terminal), so it is not approved. Do not start work."
                                    .to_string()
                            }
                        };

                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content
                        }));
                        continue;
                    }

                    // A question for the user is answered out here for the same reason a
                    // plan is: the keyboard belongs to the REPL, and the tool body runs in
                    // the MCP server too, where there is nobody to ask.
                    if tool_name == ASK_USER {
                        let question = arguments["question"].as_str().unwrap_or_default();
                        // Anything that is not a string is dropped rather than rendered as
                        // JSON: a model that sent objects here is better told the list was
                        // too short than shown its own braces back on screen.
                        let options: Vec<String> = arguments["options"]
                            .as_array()
                            .map(|items| {
                                items
                                    .iter()
                                    .filter_map(|item| item.as_str().map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default();

                        let content = match ask::normalize(question, &options) {
                            Err(reason) => reason,
                            Ok((question, options)) => {
                                let answered = ask_frontend(tx, |reply| Ask::User {
                                    question: question.clone(),
                                    options: options.clone(),
                                    reply,
                                })
                                .await;
                                match answered {
                                    Some(answer) => ask::reply_for(&answer, &options),
                                    // Nobody to ask is not the same as an answer, and it
                                    // must not stall the turn either: the model is told to
                                    // carry the decision itself.
                                    None => {
                                        let _ = tx.send(
                                            Notice::AskFailed {
                                                what: "the user",
                                                detail: "no interactive terminal".to_string(),
                                            }
                                            .into(),
                                        );
                                        "The question could not be put to the user (no \
                                         interactive terminal). Decide it yourself, say in \
                                         one line which way you went and why, and carry on."
                                            .to_string()
                                    }
                                }
                            }
                        };

                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content
                        }));
                        continue;
                    }

                    // Run out here rather than in the tool, because every call the helper
                    // makes is cleared through `authorize` — and that needs the agent.
                    // The call itself needs no clearing: it does nothing that its own
                    // calls do not each answer for.
                    if tool_name == SUB_AGENT {
                        let task = arguments["task"].as_str().unwrap_or_default().trim().to_string();
                        let content = if task.is_empty() {
                            "The task was empty. Put the whole task in the 'task' field and                              call sub_agent again."
                                .to_string()
                        } else {
                            let _ = tx.send(AgentEvent::ToolStarted {
                                name: tool_name.to_string(),
                                arguments: arguments.clone(),
                                auto_note: None,
                            });
                            let Some(report) = run_sub_agent(agent, &task, tx, link).await else {
                                let _ = tx.send(Notice::Interrupted.into());
                                agent.messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": "Interrupted by the user before the sub-agent finished."
                                }));
                                interrupted = true;
                                break;
                            };
                            let _ = tx.send(AgentEvent::DidTool {
                                name: tool_name.to_string(),
                                arguments: arguments.clone(),
                            });
                            let _ = tx.send(AgentEvent::ToolResult(report.clone()));
                            report
                        };
                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content
                        }));
                        continue;
                    }

                    let Cleared { auto_note, prompted } =
                        match authorize(agent, tx, link, tool_name, &arguments).await {
                            Ok(cleared) => cleared,
                            Err(content) => {
                                agent.messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": content
                                }));
                                continue;
                            }
                        };
                    // Execute the tool
                    // A slow tool (a big fetch, a long build) is interruptible too. The
                    // tool_call_id is still answered before breaking out — an unanswered one
                    // is what the next request gets rejected for.
                    let _ = tx.send(AgentEvent::ToolStarted { name: tool_name.to_string(), arguments: arguments.clone(), auto_note: auto_note.clone() });

                    // The diff belongs under the call line, not before it, and it has to
                    // appear on every path that skipped the prompt — a pre-approved write
                    // and now an auto-approved one both used to arrive as a path and a byte
                    // count, with no way to see what went into the file. Convenience should
                    // cost the prompt, not the oversight. Suppressed only when the prompt
                    // itself already showed the same diff a moment ago.
                    if !prompted {
                        if let Some(diff) = file_change_diff(tool_name, &arguments) {
                            let _ = tx.send(AgentEvent::ToolDiff(diff));
                        }
                    }


                    // Captured before the arguments move into the call. For a shell tool
                    // the command is the thing that ran, so naming the tool as well would
                    // only say "run_command" twice.
                    // Cloned before the arguments move into the call, because what
                    // ran has to be nameable after it has run. Cheap next to the
                    // call itself, and the alternative is the frontend inventing
                    // a description from the tool name alone.
                    let activity_arguments = arguments.clone();
                    // What a picture coming back from this call is called, asked
                    // while the arguments are still here to ask. `read` is the
                    // only tool that returns one, so its path is the only name
                    // worth looking for.
                    let image_label = arguments["file_path"]
                        .as_str()
                        .map(|path| display_path(std::path::Path::new(path)))
                        .unwrap_or_else(|| tool_name.to_string());
                    let executed = interruptible(
                        tx,
                        link,
                        Waiting::Tool(tool_name.to_string()),
                        agent.server.execute_tool(tool_name, arguments),
                    )
                    .await;

                    let Some(result) = executed else {
                        let _ = tx.send(Notice::Interrupted.into());
                        agent.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "Interrupted by the user before this tool finished."
                        }));
                        interrupted = true;
                        break;
                    };

                    // It ran, whatever it returned: a failed build is still the last
                    // thing that happened, and the bar saying so is the point.
                    let _ = tx.send(AgentEvent::DidTool {
                        name: tool_name.to_string(),
                        arguments: activity_arguments,
                    });

                    let tool_result = result.unwrap_or_else(|e| {
                        let _ = tx.send(Notice::ToolFailed(format!("{:?}", e)).into());
                        crate::tools::ToolOutput {
                            text: format!("Error: {}", e),
                            images: Vec::new(),
                        }
                    });

                    // Held for after the loop rather than pushed here. A picture
                    // cannot go in the `role: "tool"` reply — that message may
                    // carry text and nothing else, and a server rejects an image
                    // part in one outright — so it has to arrive as a user
                    // message, and a user message may not sit between an
                    // assistant turn and the tool results it is waiting for.
                    for (mime, data) in tool_result.images {
                        let _ = tx.send(
                            Notice::ImageAttached {
                                path: image_label.clone(),
                                // Back from base64 to the size of the thing it
                                // encodes, which is the figure that means
                                // something to whoever reads the line.
                                bytes: data.len() / 4 * 3,
                                outside: false,
                            }
                            .into(),
                        );
                        seen_images.push(json!({
                            "type": "image_url",
                            "image_url": { "url": format!("data:{};base64,{}", mime, data) }
                        }));
                    }
                    let tool_result = tool_result.text;

                    // The summary is the row that used to be all there was: the first
                    // line of the result, cut to fit. The result itself is kept behind it
                    // now rather than thrown away, and Ctrl+O opens it.
                    //
                    // research and library are the exception: each has already printed its
                    // progress and its whole answer while it ran, so a summary here would
                    // repeat the answer's first line immediately underneath the answer.
                    if tool_name != "research" && tool_name != "library" {
                        let _ = tx.send(AgentEvent::ToolResult(tool_result.clone()));
                    }

                    let tool_result = if repaired.is_empty() {
                        tool_result
                    } else {
                        format!(
                            "{tool_result}

[haico] Your `{names}` argument ended with a second copy of                              this call's other arguments written inside its text (`\", \"old_string\": \"...`).                              That copy was cut off before the tool ran, so only the text before it                              was used. End each string argument where its own text ends.",
                            names = repaired.join("`, `"),
                        )
                    };

                    // Add tool result
                    agent.messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result
                    }));
                }

                // The break above only leaves the tool loop; this leaves the turn.
                if interrupted {
                    break TurnEnd::Interrupted;
                }

                // The pictures the round's tools produced, in the one position
                // they are legal: after every tool call in the round has its
                // reply, as a user message. The wire format allows no image in a
                // `role: "tool"` message, and no user message before those
                // replies are complete — between them that leaves exactly here.
                //
                // A line of text goes with them because the images arrive with
                // no `tool_call_id` and nothing else ties them to the call that
                // fetched them. The tool's own reply named the file; this says
                // which message the picture is in.
                if !seen_images.is_empty() {
                    let count = seen_images.len();
                    let mut parts = seen_images;
                    parts.push(json!({
                        "type": "text",
                        "text": format!(
                            "{} named in the tool result{} above, attached here because an \
                             image cannot be sent inside a tool result.",
                            if count == 1 { "The image" } else { "The images" },
                            if count == 1 { "" } else { "s" }
                        )
                    }));
                    agent.messages.push(json!({"role": "user", "content": parts}));
                }

                // The tool log makes a loop obvious to whoever is watching it scroll past,
                // but the model reads none of that — every round looks to it like a fresh
                // decision, which is why it can repeat one call fifty times. Saying the
                // repetition out loud in the conversation is what breaks it, and it breaks
                // it far earlier than the round cap. That matters most in auto mode, where
                // the cap is the only backstop and nobody is watching it climb.
                if run.0 >= LOOP_BREAK {
                    let _ =
                        tx.send(Notice::LoopBroken { tool: run.1.clone(), count: run.0 }.into());
                    // Whatever was typed while this ran is deliberately left queued: it
                    // becomes the next prompt, which is where the user takes over.
                    break TurnEnd::Failed;
                }
                if run.0 >= LOOP_NUDGE && run.0 > nudged_at {
                    nudged_at = run.0;
                    let _ = tx.send(Notice::LoopNudge { tool: run.1.clone(), count: run.0 }.into());
                    // A user message rather than a system one: mid-conversation system
                    // messages are ignored by some local models and rejected outright by
                    // some servers, and this one has to land. The order is legal here for
                    // the same reason the queued text below is — every tool call in this
                    // round already has its result.
                    agent.messages.push(json!({
                        "role": "user",
                        "content": format!(
                            "You have called '{}' with the same arguments {} times in a \
                             row. Get out of the loop — calling it again will not tell you \
                             anything you have not already been told. Take a different \
                             approach, or stop and say what you are stuck on.",
                            run.1, run.0
                        )
                    }));
                }

                // Anything typed while this round ran joins the conversation here, where
                // every tool call already has its result. Any sooner would put a user
                // message between an assistant turn and the tool results it is waiting
                // for, which is a message order servers reject outright.
                // The echo carries what was typed; the message carries the same
                // text with its mentions read into it. Typing `@path` while the
                // model works is as ordinary as typing it at the prompt, and a
                // mention that expanded in one place and not the other would be
                // the harder thing to explain.
                for text in link.take_queued() {
                    let _ = tx.send(Notice::QueuedSent(text.clone()).into());
                    let content = attach_mentions(agent, &text, tx).await;
                    agent.messages.push(json!({"role": "user", "content": content}));
                }
                // Timers that came due while this round ran join at the same boundary and
                // for the same reason. Not sooner — and not by interrupting the round
                // either: a timer is a reminder, and the work in progress may well be the
                // thing it was set to check on.
                let fired = agent.server.take_due_timers();
                if !fired.is_empty() {
                    let _ = tx.send(Notice::TimersFired(fired.clone()).into());
                    agent.messages.push(json!({
                        "role": "user",
                        "content": crate::timers::message_for(&fired)
                    }));
                }

                // Continue the loop to get next response (might be more tool calls or final answer)
                continue;
            } else {
                // No tool call, this is the final response — already streamed above.
                agent.messages.push(message.clone());
                if std::mem::take(&mut just_compacted) {
                    push_user_note(&mut agent.messages, AFTER_COMPACTION);
                    continue;
                }
                break TurnEnd::Answered;
            }
        };

        // Every way out of the loop above lands here, an interrupt and a server error
        // included: the line says how long the turn ran for, not whether it got anywhere,
        // and a turn that ended badly after four minutes is exactly the one worth timing.
        // Dimmed for the same reason the chain of thought is — it is not the answer.
        let _ = tx.send(AgentEvent::Finished { elapsed: turn_started.elapsed() });

        // The file /init just wrote is the one that belongs in the system message, and what
        // was loaded at startup — usually nothing, since /init is what you run when the
        // project has no instructions yet — is now out of date. Re-read it here rather than
        // leaving it to the next startup, so the new rules are in force for the rest of this
        // session and survive /clear, which re-pushes this same content.
        if initializing {
            let reloaded = load_project_context();
            if reloaded != agent.project_context {
                agent.project_context = reloaded;
                if let Some((name, _)) = &agent.project_context {
                    let _ = tx.send(Notice::ProjectContextReloaded(name.clone()).into());
                }
                agent.system_prompt = compose_system_prompt(
                    agent.config.system_prompt.as_deref(),
                    &agent.client.model,
                    agent.settings.use_system_prompt,
                    agent.user_context.as_deref(),
                    agent.project_context.as_ref(),
                    agent.skill_index.as_deref(),
                    agent.library_index.as_deref(),
                    agent.memory_index.as_deref(),
                    &agent.workspace,
                );
                set_system_message(&mut agent.messages, agent.system_prompt.as_deref());
            }
        }

        // Persisted every turn, so a crash does not cost the session's history.
        // The input history is the frontend's and is saved by the pump.
        if let Some(path) = &agent.session_path {
            let todos = agent.plan();
            save_session(path, &agent.messages, &agent.client.model, &todos);
        }

        end
}

#[cfg(test)]
mod send_check {
    /// The GUI runs winit on the main thread and the conversation on a worker,
    /// so the agent has to be movable between them. Asserted at compile time
    /// rather than discovered when the frontend is half-written.
    const fn assert_send<T: Send>() {}
    #[allow(dead_code)]
    const _: () = assert_send::<super::Agent>();
}

// ---------------------------------------------------------------------------
// Model settings
// ---------------------------------------------------------------------------

/// How long a change just made is going to last, for the frontend to say.
///
/// Most settings are session-only on purpose — a temperature is something to try
/// against the task in hand — so `Session` is not a failure, it is the usual
/// answer. `Failed` is the one that has to be reported: a setting the user was
/// told would last and did not is worse than one they know never made it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Kept {
    /// This session only. Nothing was written, and nothing was meant to be.
    Session,
    /// Written to the config file, so it is still in force next run.
    Config,
    /// Meant for the config file and did not get there. Carries why, in words a
    /// frontend can show as they are.
    Failed(String),
}

/// A model setting a frontend can read and change.
///
/// These were only ever reachable through `/set`, which meant the parsing, the
/// canonical names and the "apply to both clients" step all lived inside one
/// slash command. They are settings rather than commands — a window puts them in
/// a dialog and never types a name at all — so the policy belongs here, and only
/// the wording stays with each frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Knob {
    /// The model's context window. Not a request field and not session-only:
    /// how big a model's window is is a fact about the model, true again the
    /// next time it is loaded, so setting it writes it into that model's
    /// section of the config file.
    Context,
    Temperature,
    TopP,
    TopK,
    MinP,
    RepeatPenalty,
    MaxTokens,
    /// A string rather than a number, and the only one: the level names come from the
    /// model's chat template, so there is nothing here to parse them against.
    ReasoningEffort,
    /// haico's own rather than the server's: it never goes into a request, so
    /// there is nothing to apply to the clients afterwards, and clearing it means
    /// falling back to the config file rather than letting the server decide.
    ReasoningThreshold,
}

impl Knob {
    /// Every knob, in the order a frontend should show them.
    pub(crate) const ALL: &'static [Knob] = &[
        Knob::Context,
        Knob::Temperature,
        Knob::TopP,
        Knob::TopK,
        Knob::MinP,
        Knob::RepeatPenalty,
        Knob::MaxTokens,
        Knob::ReasoningEffort,
        Knob::ReasoningThreshold,
    ];

    /// The canonical wire name — what the config file and the request field use.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Knob::Context => "context",
            Knob::Temperature => "temperature",
            Knob::TopP => "top_p",
            Knob::TopK => "top_k",
            Knob::MinP => "min_p",
            Knob::RepeatPenalty => "repeat_penalty",
            Knob::MaxTokens => "max_tokens",
            Knob::ReasoningEffort => "reasoning_effort",
            Knob::ReasoningThreshold => "reasoning_threshold",
        }
    }

    /// Resolve a name a person typed, including the abbreviations they use.
    pub(crate) fn parse(name: &str) -> Option<Knob> {
        match name.to_lowercase().as_str() {
            "context" | "context_length" | "ctx" => Some(Knob::Context),
            "temp" | "temperature" => Some(Knob::Temperature),
            "top_p" | "top-p" | "topp" => Some(Knob::TopP),
            "top_k" | "top-k" | "topk" => Some(Knob::TopK),
            "min_p" | "min-p" | "minp" => Some(Knob::MinP),
            "repeat_penalty" | "repeat-penalty" | "repeat" | "penalty" => {
                Some(Knob::RepeatPenalty)
            }
            "max_tokens" | "max-tokens" | "max" => Some(Knob::MaxTokens),
            "reasoning_effort" | "reasoning-effort" | "effort" => Some(Knob::ReasoningEffort),
            "reasoning_threshold" | "reasoning-threshold" | "threshold" => {
                Some(Knob::ReasoningThreshold)
            }
            _ => None,
        }
    }

    /// Whether this one is a count. `top_k` and `max_tokens` are: sending 40.0
    /// where a server wants an integer is a needless way to get a 400 back.
    fn is_count(self) -> bool {
        matches!(
            self,
            Knob::TopK | Knob::MaxTokens | Knob::ReasoningThreshold | Knob::Context
        )
    }
}

impl Agent {
    /// The knob's current value and where it came from, ready to show.
    pub(crate) fn knob(&self, knob: Knob) -> String {
        match knob {
            // Never absent: there is always a window in force, even if it is only
            // the built-in guess at one.
            Knob::Context => {
                format!("{}  [{}]", self.context_tokens, self.settings.context_source)
            }
            Knob::Temperature => self.settings.temperature.describe(),
            Knob::TopP => self.settings.top_p.describe(),
            Knob::TopK => self.settings.top_k.describe(),
            Knob::MinP => self.settings.min_p.describe(),
            Knob::RepeatPenalty => self.settings.repeat_penalty.describe(),
            Knob::MaxTokens => self.settings.max_tokens.describe(),
            Knob::ReasoningEffort => self.settings.reasoning_effort.describe(),
            // Never absent: unset means the configured default, not "no value".
            Knob::ReasoningThreshold => format!("{} tokens", self.reasoning_threshold),
        }
    }

    /// The value alone, with no provenance — what a form field is filled with.
    /// `None` for a knob that is not set, which a field shows as empty.
    pub(crate) fn knob_value(&self, knob: Knob) -> Option<String> {
        match knob {
            Knob::Context => Some(self.context_tokens.to_string()),
            Knob::Temperature => self.settings.temperature.value.map(|v| v.to_string()),
            Knob::TopP => self.settings.top_p.value.map(|v| v.to_string()),
            Knob::TopK => self.settings.top_k.value.map(|v| v.to_string()),
            Knob::MinP => self.settings.min_p.value.map(|v| v.to_string()),
            Knob::RepeatPenalty => self.settings.repeat_penalty.value.map(|v| v.to_string()),
            Knob::MaxTokens => self.settings.max_tokens.value.map(|v| v.to_string()),
            Knob::ReasoningEffort => self.settings.reasoning_effort.value.clone(),
            Knob::ReasoningThreshold => Some(self.reasoning_threshold.to_string()),
        }
    }

    /// Set a knob for this session, or clear it back to what is configured.
    ///
    /// `None`, an empty string, or one of the words a person types to mean
    /// "stop overriding it" all clear. `Err` carries what was wrong with the
    /// text, for the frontend to say in its own words; nothing changes when it
    /// does, so a rejected field leaves the session exactly as it was.
    ///
    /// `Ok` says how long the change lasts: `Session` for every request field,
    /// and `Config` for the one knob that is written back to the file.
    pub(crate) fn set_knob(&mut self, knob: Knob, value: Option<&str>) -> Result<Kept, String> {
        const SET_HERE: &str = "set here";
        let text = value.map(str::trim).filter(|v| !v.is_empty());
        let clearing = text.is_none() || matches!(text, Some("unset" | "default" | "-"));

        if knob == Knob::Context {
            return self.set_context(text.filter(|_| !clearing));
        }

        if knob == Knob::ReasoningThreshold {
            // Its own path, because it is not a request field: clearing means the
            // configured value rather than an absent one, and there is nothing to
            // push to the clients.
            self.reasoning_threshold = match text {
                _ if clearing => self.settings.reasoning_threshold(),
                Some(text) => text
                    .parse::<usize>()
                    .map_err(|_| format!("{} needs a whole number of tokens", knob.name()))?,
                None => unreachable!("not clearing implies text"),
            };
            return Ok(Kept::Session);
        }

        if clearing {
            match knob {
                Knob::Temperature => self.settings.temperature = Setting::unset(FROM_UNSET),
                Knob::TopP => self.settings.top_p = Setting::unset(FROM_UNSET),
                Knob::TopK => self.settings.top_k = Setting::unset(FROM_UNSET),
                Knob::MinP => self.settings.min_p = Setting::unset(FROM_UNSET),
                Knob::RepeatPenalty => self.settings.repeat_penalty = Setting::unset(FROM_UNSET),
                Knob::MaxTokens => self.settings.max_tokens = Setting::unset(FROM_UNSET),
                Knob::ReasoningEffort => {
                    self.settings.reasoning_effort = Setting::unset(FROM_UNSET)
                }
                Knob::ReasoningThreshold | Knob::Context => unreachable!("handled above"),
            }
        } else {
            let text = text.expect("not clearing implies text");
            if knob == Knob::ReasoningEffort {
                // Nothing to validate against. Whatever the template does not recognize it
                // ignores, and refusing an unknown level here would only mean haico deciding
                // which models exist.
                self.settings.reasoning_effort =
                    Setting { value: Some(text.to_string()), source: SET_HERE };
            } else if knob.is_count() {
                let parsed = text
                    .parse::<u64>()
                    .map_err(|_| format!("{} needs a whole number", knob.name()))?;
                let setting = Setting { value: Some(parsed), source: SET_HERE };
                match knob {
                    Knob::TopK => self.settings.top_k = setting,
                    Knob::MaxTokens => self.settings.max_tokens = setting,
                    _ => unreachable!("the only counts that reach here"),
                }
            } else {
                let parsed = text
                    .parse::<f64>()
                    .map_err(|_| format!("{} needs a number", knob.name()))?;
                let setting = Setting { value: Some(parsed), source: SET_HERE };
                match knob {
                    Knob::Temperature => self.settings.temperature = setting,
                    Knob::TopP => self.settings.top_p = setting,
                    Knob::MinP => self.settings.min_p = setting,
                    Knob::RepeatPenalty => self.settings.repeat_penalty = setting,
                    _ => unreachable!("the only reals that reach here"),
                }
            }
        }

        // Both clients, same as `switch_model`: the REPL drives one and
        // `web_fetch` summarizes through the server's, so leaving one behind
        // would silently keep the old sampling for half the work.
        self.client.apply_settings(&self.settings);
        self.server.llm_client.apply_settings(&self.settings);
        Ok(Kept::Session)
    }

    /// The context knob's own path, because it is nothing like the others: it is
    /// not a request field, there is nothing to push to the clients, and it is the
    /// one setting that outlives the session.
    ///
    /// `None` clears it, which means taking it back out of the config file and
    /// falling through the layers again rather than leaving a value nothing wrote.
    fn set_context(&mut self, tokens: Option<&str>) -> Result<Kept, String> {
        let tokens = match tokens {
            Some(text) => Some(
                text.parse::<usize>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| "context needs a whole number of tokens".to_string())?,
            ),
            None => None,
        };

        // The model in force is the one this is a fact about. A window set while one
        // model is loaded has no business following the user onto the next.
        let model = self.client.model.clone();
        let already = self.config.file.models.get(&model).and_then(|s| s.context);

        // The parsed file has to move with the one on disk: `switch_model` re-resolves
        // from it, so a switch away and straight back would otherwise hand back the
        // window this just replaced.
        self.config.file.models.entry(model.clone()).or_default().context = tokens;

        match tokens {
            Some(tokens) => {
                self.context_tokens = tokens;
                self.settings.context_tokens = tokens;
                // "set here" rather than the config's own provenance, even though that
                // is where it has just gone: `--context` on the command line outranks
                // the file, and a number typed at the prompt has to win for the rest of
                // the session whatever was passed at startup.
                self.settings.context_source = "set here";
            }
            None => {
                // Back through every layer rather than to the built-in default: the top
                // level of the file, or the command line, may still have an answer.
                let resolved = resolve_settings(&self.config, &model);
                self.context_tokens = resolved.context_tokens;
                self.settings.context_tokens = resolved.context_tokens;
                self.settings.context_source = resolved.context_source;
            }
        }

        // The file already says exactly this, so there is nothing to write. Worth the
        // check because a settings dialog hands back every field whether or not it was
        // touched, and rewriting the file on each apply would be a file rewritten for
        // nothing.
        if tokens.is_some() && tokens == already {
            return Ok(Kept::Config);
        }
        Ok(self.keep(|path| save_model_context(path, &model, tokens)))
    }

    /// The thinking level in force, if any. `None` means the field is left out of the
    /// request, so the model thinks as much as it thinks.
    pub(crate) fn reasoning_effort(&self) -> Option<&str> {
        self.settings.reasoning_effort.value.as_deref()
    }

    /// Change what happens to a reply's chain of thought once the turn moves on.
    pub(crate) fn set_reasoning(&mut self, reasoning: Reasoning) {
        self.reasoning = reasoning;
    }
}

#[cfg(test)]
mod knob_tests {
    use super::*;
    use crate::testutil::test_config;

    fn agent() -> Agent {
        let cwd = std::env::current_dir().unwrap();
        Agent::new(test_config("http://127.0.0.1:1"), "m".to_string(), &cwd)
    }

    /// An agent whose config file is a scratch one, for the two settings that are
    /// written back. `test_config` leaves the path `None` on purpose so that no other
    /// test can reach the user's own config; a test about saving has to supply one.
    fn agent_saving_to(name: &str) -> (Agent, std::path::PathBuf) {
        let path = crate::testutil::scratch(name, "");
        let mut config = test_config("http://127.0.0.1:1");
        config.config_path = Some(path.clone());
        let cwd = std::env::current_dir().unwrap();
        (Agent::new(config, "m".to_string(), &cwd), path)
    }

    /// Picking a model is answering "which model", which is asked once per machine
    /// rather than once per session.
    #[test]
    fn switching_model_records_it_as_the_default() {
        let (mut agent, path) = agent_saving_to("agent-saves-model.toml");

        assert_eq!(agent.switch_model("qwen2.5:14b".to_string()), Kept::Config);
        assert_eq!(agent.model(), "qwen2.5:14b");
        assert_eq!(
            crate::config::load_file_config(Some(&path)).model.as_deref(),
            Some("qwen2.5:14b")
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Switching provider moves both clients to the new server and remembers the
    /// provider, and the model chosen there goes into that provider's own section.
    #[test]
    fn switching_provider_moves_both_clients_and_records_it() {
        let text = "version = 1\nproviders = [\"lm\", \"ollama\"]\n\n\
            [lm]\nurl = \"http://lm:1234/v1\"\ntoken = \"sk\"\nmodel = \"m\"\n\n\
            [ollama]\nurl = \"http://ollama:11434\"\n";
        let path = crate::testutil::scratch("agent-switches-provider.toml", text);
        let mut config = test_config("http://lm:1234");
        config.file = crate::config::parse_file_config(text).unwrap();
        config.provider = Some("lm".to_string());
        config.token = Some("sk".to_string());
        config.config_path = Some(path.clone());
        let cwd = std::env::current_dir().unwrap();
        let mut agent = Agent::new(config, "m".to_string(), &cwd);

        assert_eq!(agent.switch_provider("ollama", "llama3.1:8b".to_string()), Kept::Config);
        assert_eq!(agent.provider(), Some("ollama"));
        assert_eq!(agent.model(), "llama3.1:8b");
        for client in [&agent.client, &agent.server.llm_client] {
            assert_eq!(client.base_url, "http://ollama:11434");
            assert_eq!(client.token, None, "the old server's token went along");
            assert_eq!(client.model, "llama3.1:8b");
        }
        let saved = crate::config::load_file_config(Some(&path));
        assert_eq!(saved.provider.as_deref(), Some("ollama"));
        assert_eq!(saved.provider_sections["ollama"].model.as_deref(), Some("llama3.1:8b"));
        assert!(saved.model.is_none());

        // And back: the token returns with its server, and /model on it saves there.
        agent.switch_provider("lm", "m".to_string());
        assert_eq!(agent.client.base_url, "http://lm:1234");
        assert_eq!(agent.client.token.as_deref(), Some("sk"));
        agent.switch_model("other".to_string());
        let saved = crate::config::load_file_config(Some(&path));
        assert_eq!(saved.provider_sections["lm"].model.as_deref(), Some("other"));

        assert!(matches!(agent.switch_provider("absent", "x".to_string()), Kept::Failed(_)));
        assert_eq!(agent.provider(), Some("lm"));

        let _ = std::fs::remove_file(&path);
    }

    /// Nowhere to save is reported rather than swallowed: a choice the user was told
    /// would last and did not is worse than one they know never made it. The switch
    /// itself happens either way.
    #[test]
    fn with_nowhere_to_save_the_switch_still_happens_and_says_so() {
        let mut agent = agent();
        assert!(matches!(agent.switch_model("other".to_string()), Kept::Failed(_)));
        assert_eq!(agent.model(), "other");
    }

    /// The context window is the one knob that outlives the session, and it belongs to
    /// the model rather than to the machine.
    #[test]
    fn the_context_window_is_kept_in_the_model_s_own_section() {
        use crate::config::{load_file_config, DEFAULT_CONTEXT_TOKENS};
        let (mut agent, path) = agent_saving_to("agent-saves-context.toml");

        assert_eq!(agent.set_knob(Knob::Context, Some("32768")), Ok(Kept::Config));
        assert_eq!(agent.context_tokens, 32768);
        assert_eq!(load_file_config(Some(&path)).models["m"].context, Some(32768));

        // Switching away and back re-resolves from the parsed config, which has to have
        // moved with the file or the window just set would be forgotten on the way back.
        agent.switch_model("other".to_string());
        assert_eq!(agent.context_tokens, DEFAULT_CONTEXT_TOKENS);
        agent.switch_model("m".to_string());
        assert_eq!(agent.context_tokens, 32768);

        // Clearing takes the key out and falls back through the layers, not straight to
        // the built-in default: the top level may still have an answer.
        agent.config.file.context = Some(4096);
        assert_eq!(agent.set_knob(Knob::Context, Some("unset")), Ok(Kept::Config));
        assert_eq!(agent.context_tokens, 4096);
        assert_eq!(load_file_config(Some(&path)).models["m"].context, None);

        let _ = std::fs::remove_file(&path);
    }

    /// A window of no tokens is not a window, and neither is half of one.
    #[test]
    fn the_context_window_refuses_nonsense_and_changes_nothing() {
        let (mut agent, path) = agent_saving_to("agent-context-nonsense.toml");
        agent.set_knob(Knob::Context, Some("32768")).unwrap();

        for bad in ["0", "lots", "32k", "-1", "8192.5"] {
            let complaint = agent.set_knob(Knob::Context, Some(bad)).unwrap_err();
            assert!(complaint.contains("whole number"), "unhelpful complaint: {complaint}");
            assert_eq!(agent.context_tokens, 32768, "{bad:?} should have changed nothing");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn names_and_abbreviations_resolve() {
        for knob in Knob::ALL {
            assert_eq!(Knob::parse(knob.name()), Some(*knob), "{} should parse", knob.name());
        }
        // The short forms people actually type.
        assert_eq!(Knob::parse("temp"), Some(Knob::Temperature));
        assert_eq!(Knob::parse("TOP-P"), Some(Knob::TopP));
        assert_eq!(Knob::parse("penalty"), Some(Knob::RepeatPenalty));
        assert_eq!(Knob::parse("threshold"), Some(Knob::ReasoningThreshold));
        assert_eq!(Knob::parse("effort"), Some(Knob::ReasoningEffort));
        assert_eq!(Knob::parse("nonsense"), None);
    }

    #[test]
    fn setting_and_clearing_round_trip() {
        let mut agent = agent();
        assert_eq!(agent.knob_value(Knob::Temperature), None, "starts unset");

        agent.set_knob(Knob::Temperature, Some("0.7")).unwrap();
        assert_eq!(agent.knob_value(Knob::Temperature), Some("0.7".to_string()));

        // Every spelling of "stop overriding it" clears.
        for clearing in ["unset", "default", "-", ""] {
            agent.set_knob(Knob::Temperature, Some("0.7")).unwrap();
            agent.set_knob(Knob::Temperature, Some(clearing)).unwrap();
            assert_eq!(
                agent.knob_value(Knob::Temperature),
                None,
                "{clearing:?} should clear it"
            );
        }
        agent.set_knob(Knob::Temperature, Some("0.7")).unwrap();
        agent.set_knob(Knob::Temperature, None).unwrap();
        assert_eq!(agent.knob_value(Knob::Temperature), None);
    }

    /// A count must not accept a decimal: sending 40.0 where the server wants an
    /// integer is a needless way to get a 400 back.
    #[test]
    fn counts_refuse_decimals_and_nothing_changes() {
        let mut agent = agent();
        agent.set_knob(Knob::TopK, Some("40")).unwrap();
        let complaint = agent.set_knob(Knob::TopK, Some("40.5")).unwrap_err();
        assert!(complaint.contains("whole number"), "unhelpful complaint: {complaint}");
        // Rejected means unchanged, not half-applied.
        assert_eq!(agent.knob_value(Knob::TopK), Some("40".to_string()));

        let complaint = agent.set_knob(Knob::Temperature, Some("hot")).unwrap_err();
        assert!(complaint.contains("number"), "unhelpful complaint: {complaint}");
    }

    /// The threshold is haico's own, not a request field: clearing it means the
    /// configured value rather than an absent one, so it is never `None`.
    #[test]
    fn the_reasoning_threshold_falls_back_rather_than_unsetting() {
        let mut agent = agent();
        let configured = agent.knob_value(Knob::ReasoningThreshold).unwrap();
        agent.set_knob(Knob::ReasoningThreshold, Some("1234")).unwrap();
        assert_eq!(agent.knob_value(Knob::ReasoningThreshold), Some("1234".to_string()));
        agent.set_knob(Knob::ReasoningThreshold, Some("unset")).unwrap();
        assert_eq!(agent.knob_value(Knob::ReasoningThreshold), Some(configured));
    }

    /// The one knob that is a string. It takes a level whole rather than parsing it,
    /// and still clears back to absent like the request fields it sits among — unlike
    /// the threshold above, which falls back to a configured value instead.
    #[test]
    fn the_effort_is_taken_whole_and_clears_to_absent() {
        let mut agent = agent();
        assert_eq!(agent.knob_value(Knob::ReasoningEffort), None, "starts unset");

        // A level haico has never heard of is still sent: the set belongs to the model.
        agent.set_knob(Knob::ReasoningEffort, Some("xhigh")).unwrap();
        assert_eq!(agent.knob_value(Knob::ReasoningEffort), Some("xhigh".to_string()));
        assert_eq!(agent.reasoning_effort(), Some("xhigh"));
        // And it reaches both clients, for the same reason the sampling does.
        assert_eq!(agent.client.reasoning_effort.as_deref(), Some("xhigh"));
        assert_eq!(agent.server.llm_client.reasoning_effort.as_deref(), Some("xhigh"));

        agent.set_knob(Knob::ReasoningEffort, Some("unset")).unwrap();
        assert_eq!(agent.knob_value(Knob::ReasoningEffort), None);
        assert_eq!(agent.reasoning_effort(), None);
        assert_eq!(agent.client.reasoning_effort, None, "the field is left out again");

        // "off" is stored and sent as typed, and also becomes the template switch.
        agent.set_knob(Knob::ReasoningEffort, Some("off")).unwrap();
        assert_eq!(agent.reasoning_effort(), Some("off"));
        assert_eq!(agent.client.reasoning_effort.as_deref(), Some("off"));
        assert_eq!(agent.client.enable_thinking, Some(false));
        assert_eq!(agent.server.llm_client.enable_thinking, Some(false));
        agent.set_knob(Knob::ReasoningEffort, Some("unset")).unwrap();
        assert_eq!(agent.client.enable_thinking, None, "unset lets the model think again");
    }

    /// Both clients get the sampling, because `web_fetch` summarizes through the
    /// server's copy and half-applied sampling is the kind of thing nobody
    /// notices until two identical prompts behave differently.
    #[test]
    fn sampling_reaches_both_clients() {
        let mut agent = agent();
        agent.set_knob(Knob::Temperature, Some("0.25")).unwrap();
        assert_eq!(agent.client.temperature, Some(0.25));
        assert_eq!(agent.server.llm_client.temperature, Some(0.25));
    }
}

/// The session's scratch directory, from the conversation's side: it has to be
/// somewhere the file tools may write, the model has to be told where it is, and
/// clearing the conversation has to take it away.
#[cfg(test)]
mod scratch_tests {
    use super::*;
    use crate::testutil::test_config;

    fn system_message(agent: &Agent) -> String {
        agent.messages[0]["content"].as_str().unwrap_or_default().to_string()
    }

    #[test]
    fn a_session_gets_a_writable_scratch_directory_and_clear_replaces_it() {
        let cwd = std::env::current_dir().unwrap();
        let mut agent = Agent::new(test_config("http://127.0.0.1:1"), "m".to_string(), &cwd);

        let first = agent.workspace.scratch().expect("a session has one").to_path_buf();
        assert!(first.is_dir());
        // Writable, or the model is being pointed at a directory the file tools refuse.
        assert!(agent.workspace.may_write(&first.join("draft.py")));
        // And named, or it is unguessable and the model invents "C:\Temp" instead.
        assert!(system_message(&agent).contains(&crate::util::display_path(&first)));

        std::fs::write(first.join("draft.py"), "print(1)").unwrap();
        agent.clear();

        let second = agent.workspace.scratch().expect("cleared, not removed").to_path_buf();
        assert_ne!(first, second);
        assert!(!first.exists(), "the cleared conversation's files survived");
        assert!(agent.workspace.may_write(&second.join("draft.py")));
        assert!(!agent.workspace.may_write(&first.join("draft.py")));

        // The prompt is rebuilt whether or not the project's instructions changed —
        // otherwise the fresh conversation is handed a path that was just deleted.
        let system = system_message(&agent);
        assert!(system.contains(&crate::util::display_path(&second)), "got: {}", system);
        assert!(!system.contains(&crate::util::display_path(&first)));
    }
}
