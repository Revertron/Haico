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
use crate::diff::file_change_diff;
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
        let diff_content = file_change_diff(tool_name, arguments);

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

// Get LLM response and handle tool call chains. The cap stops a model that keeps calling
// tools forever from spinning until the user kills the process. It does not apply in auto
// mode: a fixed number of rounds is a poor stand-in for "this is going nowhere", and a long
// job that hits it unattended has nobody to say "continue". Repetition is the better
// signal, and it is what guards auto mode.
const MAX_TOOL_ROUNDS: usize = 50;

// Replies in a row that were cut off for repeating themselves (see `repetition`). In a row,
// not per turn: an auto-mode turn lives for hours, and a model that fell into a loop once,
// was told, and went on to do an hour of real work has earned the same second chance the
// next time. Two in a row means the nudge is not landing, and a third try would most likely
// be the same loop at the same cost.
pub(crate) const MAX_REPEATING_REPLIES: usize = 2;

/// The identical-call guard: the (tool, arguments) pair called last, and how long the
/// unbroken run of it is.
///
/// Consecutive rather than counted across the turn: a model working through a list of
/// compiler errors legitimately runs the same build command a dozen times in one turn, and
/// every one of those is progress because there were edits in between. A loop is the same
/// call with nothing else happening between the tries. Any different call resets the run,
/// `nudged_at` with it, so a later loop still gets its warning.
#[derive(Default)]
pub(crate) struct LoopGuard {
    last: Option<(String, String)>,
    run: usize,
    nudged_at: usize,
}

/// What the guard has to say once a round's calls have all been answered.
pub(crate) enum LoopVerdict {
    Fine,
    /// The same call `count` times in a row: say so in the conversation, once per count.
    Nudge { tool: String, count: usize },
    /// …and this is where the loop gives up on it.
    Break { tool: String, count: usize },
}

impl LoopGuard {
    /// Count one call.
    ///
    /// Before anything else looks at it, so a call that is refused, denied *or malformed*
    /// counts too — retrying a call the workspace or the user has already turned down,
    /// unchanged and immediately, is a loop like any other. What it returned is
    /// deliberately not part of the key: a build that fails three times in a row with
    /// nothing done between the tries is stuck whatever the errors say.
    ///
    /// The malformed case has to be counted *before* the parse is acted on, not after, and
    /// that is not tidiness. Counting only calls that parsed meant a model repeating a call
    /// whose arguments are not JSON never advanced the run at all. In normal mode the round
    /// cap still stopped it; in auto mode, which skips that cap on purpose because there is
    /// nobody there to type "continue", nothing stopped it and the turn span forever
    /// unattended. Keyed on the canonical form when it parses (serde_json sorts object
    /// keys, so the same call always renders to the same string) and on the raw argument
    /// text when it does not, which is the only form a malformed call has.
    pub(crate) fn observe(&mut self, tool: &str, function: &Value, parsed: &std::result::Result<Value, String>) {
        let key = match parsed {
            Ok(arguments) => arguments.to_string(),
            Err(_) => function.get("arguments").map(|a| a.to_string()).unwrap_or_default(),
        };
        let call = (tool.to_string(), key);
        if self.last.as_ref() == Some(&call) {
            self.run += 1;
        } else {
            self.last = Some(call);
            self.run = 1;
            self.nudged_at = 0;
        }
    }

    /// Asked after every call of a round has its reply, which is the one point a nudge
    /// can join the conversation.
    pub(crate) fn verdict(&mut self) -> LoopVerdict {
        let tool = self.last.as_ref().map(|(tool, _)| tool.clone()).unwrap_or_default();
        if self.run >= LOOP_BREAK {
            return LoopVerdict::Break { tool, count: self.run };
        }
        if self.run >= LOOP_NUDGE && self.run > self.nudged_at {
            self.nudged_at = self.run;
            return LoopVerdict::Nudge { tool, count: self.run };
        }
        LoopVerdict::Fine
    }
}

// What the rounds of one turn carry from one to the next.
struct TurnState {
    round: usize,
    guard: LoopGuard,
    // Cleared when compacting fails to get the history back under the line, which means
    // the window is too small to hold even a summary of it. Trying again the next round
    // would summarize the summary and throw the round away with it, so the turn runs on
    // instead and lets the server error be the honest end of it.
    autocompact: bool,
    // Set by an auto-compaction and cleared by the next reply that calls a tool. While it
    // is set, a reply that calls nothing does not end the turn: the task was mid-flight
    // when the history was replaced, and the loop has just said so. On a p2p_auth run the
    // first reply after a compaction was one sentence of intent ("Let me check the current
    // state of the codebase") with no tool call, the turn ended on it, and the unattended
    // run exited as finished with 19 compile errors in its tests.
    just_compacted: bool,
    // Set once the image recovery has run, so a failure that has nothing to do with images
    // cannot make the turn retry forever.
    stripped_images: bool,
    repeating_replies: usize,
}

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
    // Wall clock, started here rather than at the first request: what the summary at the
    // end of the turn reports is how long the user waited, which includes the tool runs,
    // the permission prompts they answered and every round of the loop below — not just
    // the time the model spent generating.
    let turn_started = std::time::Instant::now();

    // Add user message, with any `@path` in it already read into it.
    let content = attach_mentions(agent, input, tx).await;
    agent.messages.push(json!({
        "role": "user",
        "content": content
    }));

    let mut state = TurnState {
        round: 0,
        guard: LoopGuard::default(),
        autocompact: true,
        just_compacted: false,
        stripped_images: false,
        repeating_replies: 0,
    };

    let end = loop {
        state.round += 1;
        // Once a round, because that is when the history has changed and the status bar is
        // what reports it. The numbers, not the line: how wide the bar is and what goes
        // where is the frontend's business.
        let _ = tx.send(AgentEvent::Status {
            provider: agent.provider().map(str::to_string),
            model: agent.client.model.clone(),
            used_tokens: agent.context_used(),
            context_tokens: agent.context_tokens,
            effort: agent.reasoning_effort().map(str::to_string),
        });
        // Re-read every round rather than captured once: approving a plan can switch into
        // auto mode mid-turn, and the cap should stop applying from that point.
        if link.mode() != Mode::Auto && state.round > MAX_TOOL_ROUNDS {
            let _ = tx.send(Notice::RoundCapReached(MAX_TOOL_ROUNDS).into());
            break TurnEnd::Failed;
        }

        if let Some(end) = autocompact_if_due(agent, tx, link, &mut state).await {
            break end;
        }

        // Whether anything has reached the screen for this reply, and the half-rendered
        // markdown line, both belong to the presenter — see `Presenter::start_reply`. Per
        // round, not per turn: a turn is many rounds and each one streams its own reply.
        let _ = tx.send(AgentEvent::ReplyStart);
        // The count of characters streamed so far, which the activity bar shows so a model
        // that reasons for minutes before its first content token is distinguishable from
        // one that has hung, is now the pump's: it sees every chunk go past and can count
        // them itself.
        //
        // Rebuilt every round rather than once per turn: approving a plan switches the mode
        // mid-turn, and the very next request has to carry the full tool list.
        let request_messages = with_mode_note(link.mode(), &agent.workspace, &agent.messages);
        let request_tools = tools_for_mode(link.mode(), &agent.tools);
        // Of the history as it stands, not of `request_messages`: the mode note is not
        // kept, and `context_used` compares against the history it will be looking at.
        let estimate_sent = estimate_tokens(&agent.messages);

        // Esc abandons the turn instead of killing the process. It arrives at the pump as a
        // keystroke rather than as a signal, because raw mode is on for the whole session —
        // which is also what lets the next message be typed while this one is still
        // running — and reaches here as the link's cancellation.
        let result = interruptible(
            tx,
            link,
            Waiting::Model,
            agent.client.chat_completion_measured(
                Caller::Main,
                request_messages,
                request_tools,
                // Both kinds of text are handed on the moment they arrive, so nothing
                // below re-prints anything from the finished message. What either one
                // *looks* like — dim, italic, folded, markdown-wrapped — is the
                // presenter's, and this no longer knows. The third carries no text at
                // all, only how much of a tool call has landed: the call is shown whole
                // when it is about to run, and once is enough.
                |chunk| {
                    let _ = tx.send(match chunk {
                        Chunk::Reasoning(text) => AgentEvent::ReasoningChunk(text),
                        Chunk::Content(text) => AgentEvent::ReplyChunk(text),
                        Chunk::ToolFragment(characters) => AgentEvent::ToolCallChunk(characters),
                        // The failed attempt's reply is closed as one that ended, and the
                        // next attempt opens a reply of its own, so what it streams is
                        // never shown as the rest of the text the failure cut off. Nothing
                        // of it reaches the history: only the reply that comes back whole
                        // is returned.
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
                state.repeating_replies = 0;
                if let Some(prompt_tokens) = ending.usage.as_ref().and_then(|u| u["prompt_tokens"].as_u64()) {
                    agent.measured = Some(Measured { prompt_tokens: prompt_tokens as usize, estimate: estimate_sent });
                }
                message
            }
            Err(e) => {
                let _ = tx.send(AgentEvent::ReplyEnd);
                match recover_from_failed_request(agent, tx, &mut state, e) {
                    Some(end) => break end,
                    None => continue,
                }
            }
        };

        // Both kinds of text were handed on as they arrived — including the case of a model
        // whose entire reply is chain of thought with no content at all, which is why
        // nothing is re-emitted from `message` here. All that is left is to say the reply
        // is over, and let the presenter close whatever it has open.
        let _ = tx.send(AgentEvent::ReplyEnd);

        apply_reasoning_policy(agent, tx, link, &mut message).await;

        // A reply with no `tool_calls`, a null one and an empty array are the same answer,
        // and they end the turn in the one place below.
        let calls = match message.get("tool_calls").filter(|c| !c.is_null()) {
            None => Vec::new(),
            Some(calls) => match calls.as_array() {
                Some(calls) => calls.clone(),
                None => {
                    let _ = tx.send(Notice::ToolCallsMalformed(format!("{:?}", calls)).into());
                    break TurnEnd::Failed;
                }
            },
        };

        // The reply goes into the history before any of its calls is answered: the tool
        // replies refer to it by id.
        agent.messages.push(message);

        if calls.is_empty() {
            // The final response, already streamed.
            if std::mem::take(&mut state.just_compacted) {
                push_user_note(&mut agent.messages, AFTER_COMPACTION);
                continue;
            }
            break TurnEnd::Answered;
        }
        state.just_compacted = false;

        if run_tool_calls(agent, tx, link, &calls, &mut state.guard).await == Round::Interrupted {
            break TurnEnd::Interrupted;
        }

        if let Some(end) = after_round(agent, tx, link, &mut state.guard).await {
            break end;
        }
        // On to the next response, which might be more tool calls or the final answer.
    };

    // Every way out of the loop above lands here, an interrupt and a server error included:
    // the line says how long the turn ran for, not whether it got anywhere, and a turn that
    // ended badly after four minutes is exactly the one worth timing. Dimmed for the same
    // reason the chain of thought is — it is not the answer.
    let _ = tx.send(AgentEvent::Finished { elapsed: turn_started.elapsed() });

    // The file /init just wrote is the one that belongs in the system message, and what was
    // loaded at startup — usually nothing, since /init is what you run when the project has
    // no instructions yet — is now out of date. Re-read it here rather than leaving it to
    // the next startup, so the new rules are in force for the rest of this session and
    // survive /clear, which re-pushes this same content.
    if initializing {
        let reloaded = load_project_context();
        if reloaded != agent.project_context {
            agent.project_context = reloaded;
            if let Some((name, _)) = &agent.project_context {
                let _ = tx.send(Notice::ProjectContextReloaded(name.clone()).into());
            }
            agent.recompose_system_prompt();
            set_system_message(&mut agent.messages, agent.system_prompt.as_deref());
        }
    }

    // Persisted every turn, so a crash does not cost the session's history. The input
    // history is the frontend's and is saved by the pump.
    if let Some(path) = &agent.session_path {
        let todos = agent.plan();
        if let Err(e) = save_session(path, &agent.messages, &agent.client.model, &todos) {
            let _ = tx.send(Notice::SessionNotSaved(format!("{:#}", e)).into());
        }
    }

    end
}

/// Compact the history before the round's request when auto mode has filled the window.
/// `Some` ends the turn.
///
/// Auto mode is meant to run unattended, so the one thing it must not do at the edge of the
/// window is stop and ask for /compact: there is nobody there to type it, and a job left
/// running overnight would be found dead at the prompt. Checked at the top of a round rather
/// than at the end of a turn, where the old hint printed, because a long job lives inside
/// the loop for hours and may never reach the end of a turn at all. The history is balanced
/// at this point — every tool call from the round before already has its reply — so
/// replacing it wholesale cannot orphan a tool_call_id, which is not true anywhere later in
/// the round.
async fn autocompact_if_due(
    agent: &mut Agent,
    tx: &EventSink,
    link: &TurnLink,
    state: &mut TurnState,
) -> Option<TurnEnd> {
    if link.mode() != Mode::Auto
        || !state.autocompact
        || agent.context_used() <= autocompact_threshold(agent.context_tokens)
    {
        return None;
    }

    match agent.compact(tx, link).await {
        Some(Ok(report)) => {
            let _ = tx.send(Notice::Compacted(report).into());
            // The compacted history ends on the assistant agreeing that it has the notes,
            // and asking for a reply straight after that is asking the model to talk past
            // itself. This says the one thing the notes cannot: the task was mid-flight
            // when the summarizer ran, and it is the same task that continues.
            state.just_compacted = true;
            agent.messages.push(json!({
                "role": "user",
                "content": "Your context was compacted mid-task. Continue the \
                            work described in the notes above from where it left \
                            off; do not start over."
            }));

            if agent.context_used() > autocompact_threshold(agent.context_tokens) {
                state.autocompact = false;
                let _ = tx.send(Notice::CompactionStillLarge((AUTOCOMPACT_FRACTION * 100.0) as usize).into());
            }
            None
        }
        // Not fatal by itself: the request may still fit, and if it does not, the server's
        // own error ends the turn — exactly what would have happened with none of this here.
        Some(Err(e)) => {
            let _ = tx.send(Notice::CompactionFailed(format!("{:#}", e)).into());
            None
        }
        // Esc during the summarization is Esc during the turn, as everywhere else in it.
        None => {
            let _ = tx.send(Notice::Interrupted.into());
            Some(TurnEnd::Interrupted)
        }
    }
}

impl Agent {
    /// Replace the history with notes on it, for `/compact` and for auto mode at the edge
    /// of the window. `None` means the user interrupted it.
    ///
    /// One function for both because the reclaim after it is easy to forget. The review
    /// cache is keyed to a history that no longer exists, and the history that held
    /// anything remembered this session has just become a paragraph of notes — the memory
    /// index is now the only thing that still knows, so it goes into the prompt here. The
    /// REPL's `/compact` used to make its own call and its own copy of the reclaim, and it
    /// ran outside the pump, so Esc could not stop it.
    pub(crate) async fn compact(&mut self, tx: &EventSink, link: &TurnLink) -> Option<anyhow::Result<String>> {
        let (context_tokens, before) = (self.context_tokens, self.context_used());
        let compacted = interruptible(
            tx,
            link,
            Waiting::Compacting,
            compact_history(&self.client, &mut self.messages, context_tokens, before),
        )
        .await;
        // On the one path that actually shrank the session. A compaction that failed
        // freed nothing to go with it.
        if let Some(Ok(_)) = &compacted {
            self.review_cache.clear();
            self.refresh_memory();
        }
        compacted
    }
}

/// `/compact`: the compaction, said through `tx` as auto-compaction says it.
pub(crate) async fn compact_on_request(agent: &mut Agent, tx: &EventSink, link: &TurnLink) {
    let _ = tx.send(match agent.compact(tx, link).await {
        Some(Ok(report)) => Notice::Compacted(report).into(),
        Some(Err(e)) => Notice::CompactionFailed(format!("{:#}", e)).into(),
        None => Notice::Interrupted.into(),
    });
}

/// What to do about a request that failed: `None` asks again, `Some` ends the turn.
fn recover_from_failed_request(
    agent: &mut Agent,
    tx: &EventSink,
    state: &mut TurnState,
    e: anyhow::Error,
) -> Option<TurnEnd> {
    // The stream was cut off because the reply had gone into a loop. The partial reply is
    // dropped, exactly as an interrupt drops one — it is the loop, and half a tool call in
    // the history is worse than none. The history is as consistent as it was before the
    // request, so asking again is safe; asking again *unchanged* would mostly get the same
    // loop back, which is why the model is shown what it was repeating first.
    if let Some(repeating) = e.downcast_ref::<Repeating>() {
        state.repeating_replies += 1;
        let retrying = state.repeating_replies < MAX_REPEATING_REPLIES;
        let _ = tx.send(Notice::Repeating { passage: repeating.passage.clone(), retrying }.into());
        if !retrying {
            return Some(TurnEnd::Failed);
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
        return None;
    }

    // An image in the history is the one failure this can back out of, and the only one
    // worth trying to: a model without a vision head rejects the request that carries a
    // picture and every request after it, so the conversation is dead until the picture
    // leaves. That is a real risk here and not a theoretical one, because `read` lets the
    // *model* choose the file — it can point itself at a screenshot with no way to know
    // whether it can see one.
    //
    // Once per turn, so an unrelated failure that keeps failing still ends at the prompt
    // instead of looping. The images are gone from the history for good, which is the
    // point: retrying with them in would fail the same way.
    if !state.stripped_images && strip_images(&mut agent.messages) > 0 {
        state.stripped_images = true;
        let _ = tx.send(Notice::ImagesStripped(format!("{:#}", e)).into());
        return None;
    }

    // A dropped connection, a server-side error, or an unparseable body must not end the
    // session. Every tool call made so far already has its reply, so the history is
    // consistent: drop back to the prompt.
    let _ = tx.send(Notice::RequestFailed(format!("{:#}", e)).into());
    Some(TurnEnd::Failed)
}

/// The one place the reasoning policy is applied, on the reply just finished and before it
/// is pushed — so the history only ever holds the post-policy version, however the round
/// goes on to end.
///
/// Only the compression of a long enough trace costs a request, and only that goes through
/// `interruptible` — several seconds with a frozen input box every turn would be worse than
/// the tokens are worth. Esc during it means `None`, which lands in the same place a failure
/// does: the original chain of thought is kept and the turn carries on. Dropping a field,
/// and declining a trace under the threshold, are both over before a frame could be
/// painted, so they are awaited directly rather than announced as work in progress.
async fn apply_reasoning_policy(agent: &mut Agent, tx: &EventSink, link: &TurnLink, message: &mut Value) {
    if agent.reasoning == Reasoning::Keep {
        return;
    }
    let note = if reasoning_needs_a_request(message, agent.reasoning, agent.reasoning_threshold) {
        interruptible(
            tx,
            link,
            Waiting::CompressingReasoning,
            apply_reasoning(&agent.client, message, agent.reasoning, agent.reasoning_threshold),
        )
        .await
        .flatten()
    } else {
        apply_reasoning(&agent.client, message, agent.reasoning, agent.reasoning_threshold).await
    };

    match note {
        // Shown rather than folded, and wearing the chrome's light blue: the whole point of
        // compressing is to find out whether it is paying for itself, which cannot be
        // judged from a line that is hidden by default. Tinted, not painted — the bars are
        // the only thing on screen that gets a background, and that is what makes them
        // read as furniture rather than as output.
        Some(ReasoningOutcome::Compressed { before, after }) => {
            agent.reasoning_stats.record(ReasoningOutcome::Compressed { before, after });
            // Read back out of the message rather than handed over with the figures:
            // `apply_reasoning` has just written the summary into the reasoning field, and
            // that field is the version the history now carries — reporting anything else
            // would be reporting a text the conversation does not actually contain.
            let summary = reasoning_text(message).unwrap_or_default().to_string();
            let _ = tx.send(Notice::ReasoningCompressed { before, after, summary }.into());
        }
        // Dropping says nothing the policy did not already say: it removes every trace,
        // every turn, so a line per reply would be a line per reply saying the same thing.
        Some(ReasoningOutcome::Dropped { tokens }) => {
            let _ = tx.send(Notice::ReasoningDropped { tokens }.into());
        }
        None => {}
    }
}

#[derive(PartialEq)]
enum Round {
    Done,
    Interrupted,
}

/// What one call is answered with, and whether the user stopped it.
struct CallReply {
    content: String,
    interrupted: bool,
}

impl From<String> for CallReply {
    fn from(content: String) -> Self {
        CallReply { content, interrupted: false }
    }
}

impl From<&str> for CallReply {
    fn from(content: &str) -> Self {
        content.to_string().into()
    }
}

/// Answer every call of one reply, in order.
///
/// **Every `tool_call_id` gets its `role:"tool"` reply here and nowhere else.** An
/// unanswered id makes the *next* request fail server-side, which is far harder to recover
/// from than telling the model its call was malformed. Each kind of call used to push its
/// own reply, eight sites that each had to remember; now a call only says what its reply
/// is, and this pushes it.
///
/// That includes the calls an interrupt never reached. Stopping at the first one left the
/// rest of the reply's ids unanswered, and the conversation then failed on its next
/// request.
async fn run_tool_calls(
    agent: &mut Agent,
    tx: &EventSink,
    link: &TurnLink,
    calls: &[Value],
    guard: &mut LoopGuard,
) -> Round {
    // Pictures the round's tools returned, waiting for the one place an image is allowed
    // to go. See where they are pushed below.
    let mut seen_images: Vec<Value> = Vec::new();
    let mut round = Round::Done;

    for tool_call in calls {
        let tool_call_id = tool_call.get("id").cloned().unwrap_or(Value::Null);
        let reply = if round == Round::Interrupted {
            CallReply::from("Not run: the user interrupted the turn before this call was reached.")
        } else {
            run_one_call(agent, tx, link, tool_call, guard, &mut seen_images).await
        };
        if reply.interrupted {
            round = Round::Interrupted;
        }
        agent.messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": reply.content
        }));
    }

    // An interrupted round ends the turn; whatever the user does next starts from the
    // replies alone.
    if round == Round::Interrupted {
        return round;
    }

    // The pictures the round's tools produced, in the one position they are legal: after
    // every tool call in the round has its reply, as a user message. The wire format allows
    // no image in a `role: "tool"` message, and no user message before those replies are
    // complete — between them that leaves exactly here.
    //
    // A line of text goes with them because the images arrive with no `tool_call_id` and
    // nothing else ties them to the call that fetched them. The tool's own reply named the
    // file; this says which message the picture is in.
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
    round
}

/// One call, from its raw form in the reply to the text its id is answered with.
async fn run_one_call(
    agent: &mut Agent,
    tx: &EventSink,
    link: &TurnLink,
    tool_call: &Value,
    guard: &mut LoopGuard,
    seen_images: &mut Vec<Value>,
) -> CallReply {
    let function = &tool_call["function"];

    let Some(tool_name) = function["name"].as_str() else {
        let detail = format!(
            "Malformed tool call: no function name in {}",
            truncate_at_boundary(&function.to_string(), 500)
        );
        let _ = tx.send(Notice::ToolCallUnnamed(detail.clone()).into());
        return detail.into();
    };

    let parsed = parse_tool_arguments(function);
    guard.observe(tool_name, function, &parsed);

    let mut arguments = match parsed {
        Ok(arguments) => arguments,
        Err(detail) => {
            let _ = tx.send(Notice::ToolArgumentsInvalid { tool: tool_name.to_string(), detail: detail.clone() }.into());
            return format!("Could not run '{}': {}. Call it again with valid JSON arguments.", tool_name, detail).into();
        }
    };

    // Before anything looks at the arguments, so the gate, the diff preview and the tool
    // all see what the model meant rather than what it typed (see
    // `repair_leaked_arguments`). Said in the tool's reply: a model that is never told
    // keeps making the same slip, and this one made it in half the edits of a run.
    let repaired = repair_leaked_arguments(&mut arguments);

    // The calls answered out here rather than by a tool body, each because what it needs
    // is the agent's or the frontend's, not the tool server's.
    match tool_name {
        START_PLAN => start_plan_reply(tx, link).into(),
        PRESENT_PLAN => present_plan_reply(tx, link, &arguments).await.into(),
        ASK_USER => ask_user_reply(tx, &arguments).await.into(),
        SUB_AGENT => sub_agent_reply(agent, tx, link, &arguments).await,
        _ => gated_call(agent, tx, link, tool_name, arguments, &repaired, seen_images).await,
    }
}

// The model can put itself into plan mode, for when the user asked for a plan rather than
// for the work. Nothing to approve: this only ever takes capability away, and present_plan
// is where consent is needed.
fn start_plan_reply(tx: &EventSink, link: &TurnLink) -> &'static str {
    if link.mode() == Mode::Plan {
        return "Already in plan mode. Investigate with the reading tools, then call present_plan.";
    }
    link.set_mode(Mode::Plan);
    let _ = tx.send(Notice::PlanModeEntered.into());
    "Plan mode is active. The tools that change this machine are no \
     longer available to you. Investigate the real code with read, \
     grep, ls and glob_search, then call present_plan with what you \
     propose. Do not start work until the user approves it."
}

// A finished plan is not a tool call so much as the end of plan mode, and the mode it ends
// lives out here rather than inside the tool.
async fn present_plan_reply(tx: &EventSink, link: &TurnLink, arguments: &Value) -> String {
    let plan = arguments["plan"].as_str().unwrap_or("").trim().to_string();
    if plan.is_empty() {
        return "The plan was empty. Put the plan itself in the 'plan' field and call present_plan again."
            .to_string();
    }

    // A terminal that cannot be read from must not be taken as approval — the same rule
    // the permission prompt follows.
    let decision = ask_frontend(tx, |reply| Ask::Plan { plan: plan.clone(), reply }).await;
    match decision {
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
            // The model is told nothing about the review. It should propose the same call
            // either way, and knowing it is being checked is an invitation to write for the
            // checker.
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
                Notice::AskFailed { what: "about the plan", detail: "no interactive terminal".to_string() }.into(),
            );
            "The plan could not be shown for approval (no interactive \
             terminal), so it is not approved. Do not start work."
                .to_string()
        }
    }
}

// A question for the user is answered out here for the same reason a plan is: the keyboard
// belongs to the frontend, and the tool body runs in the MCP server too, where there is
// nobody to ask.
async fn ask_user_reply(tx: &EventSink, arguments: &Value) -> String {
    let question = arguments["question"].as_str().unwrap_or_default();
    // Anything that is not a string is dropped rather than rendered as JSON: a model that
    // sent objects here is better told the list was too short than shown its own braces
    // back on screen.
    let options: Vec<String> = arguments["options"]
        .as_array()
        .map(|items| items.iter().filter_map(|item| item.as_str().map(str::to_string)).collect())
        .unwrap_or_default();

    let (question, options) = match ask::normalize(question, &options) {
        Err(reason) => return reason,
        Ok(normalized) => normalized,
    };
    let answered =
        ask_frontend(tx, |reply| Ask::User { question: question.clone(), options: options.clone(), reply }).await;
    match answered {
        Some(answer) => ask::reply_for(&answer, &options),
        // Nobody to ask is not the same as an answer, and it must not stall the turn
        // either: the model is told to carry the decision itself.
        None => {
            let _ = tx.send(Notice::AskFailed { what: "the user", detail: "no interactive terminal".to_string() }.into());
            "The question could not be put to the user (no \
             interactive terminal). Decide it yourself, say in \
             one line which way you went and why, and carry on."
                .to_string()
        }
    }
}

// Run out here rather than in the tool, because every call the helper makes is cleared
// through `authorize` — and that needs the agent. The call itself needs no clearing: it
// does nothing that its own calls do not each answer for.
async fn sub_agent_reply(agent: &mut Agent, tx: &EventSink, link: &TurnLink, arguments: &Value) -> CallReply {
    let task = arguments["task"].as_str().unwrap_or_default().trim().to_string();
    if task.is_empty() {
        return "The task was empty. Put the whole task in the 'task' field and call sub_agent again.".into();
    }
    let _ = tx.send(AgentEvent::ToolStarted {
        name: SUB_AGENT.to_string(),
        arguments: arguments.clone(),
        auto_note: None,
    });
    let Some(report) = run_sub_agent(agent, &task, tx, link).await else {
        let _ = tx.send(Notice::Interrupted.into());
        return CallReply {
            content: "Interrupted by the user before the sub-agent finished.".to_string(),
            interrupted: true,
        };
    };
    let _ = tx.send(AgentEvent::DidTool { name: SUB_AGENT.to_string(), arguments: arguments.clone() });
    let _ = tx.send(AgentEvent::ToolResult(report.clone()));
    report.into()
}

// Every other call: through the gates, then to the tool server.
async fn gated_call(
    agent: &mut Agent,
    tx: &EventSink,
    link: &TurnLink,
    tool_name: &str,
    arguments: Value,
    repaired: &[String],
    seen_images: &mut Vec<Value>,
) -> CallReply {
    let Cleared { auto_note, prompted } = match authorize(agent, tx, link, tool_name, &arguments).await {
        Ok(cleared) => cleared,
        Err(refusal) => return refusal.into(),
    };

    // A slow tool (a big fetch, a long build) is interruptible too. The tool_call_id is
    // still answered before the turn ends — an unanswered one is what the next request
    // gets rejected for.
    let _ = tx.send(AgentEvent::ToolStarted {
        name: tool_name.to_string(),
        arguments: arguments.clone(),
        auto_note: auto_note.clone(),
    });

    // The diff belongs under the call line, not before it, and it has to appear on every
    // path that skipped the prompt — a pre-approved write and now an auto-approved one both
    // used to arrive as a path and a byte count, with no way to see what went into the
    // file. Convenience should cost the prompt, not the oversight. Suppressed only when the
    // prompt itself already showed the same diff a moment ago.
    if !prompted {
        if let Some(diff) = file_change_diff(tool_name, &arguments) {
            let _ = tx.send(AgentEvent::ToolDiff(diff));
        }
    }

    // Cloned before the arguments move into the call, because what ran has to be nameable
    // after it has run. Cheap next to the call itself, and the alternative is the frontend
    // inventing a description from the tool name alone.
    let activity_arguments = arguments.clone();
    // What a picture coming back from this call is called, asked while the arguments are
    // still here to ask. `read` is the only tool that returns one, so its path is the only
    // name worth looking for.
    let image_label = arguments["file_path"]
        .as_str()
        .map(|path| display_path(std::path::Path::new(path)))
        .unwrap_or_else(|| tool_name.to_string());
    // Inside a progress scope, so a tool that runs a loop of its own can say what it is
    // doing as it goes (see `crate::progress`).
    let executed = interruptible(
        tx,
        link,
        Waiting::Tool(tool_name.to_string()),
        crate::progress::reporting(tx, agent.server.execute_tool(tool_name, arguments)),
    )
    .await;

    let Some((result, answer_shown)) = executed else {
        let _ = tx.send(Notice::Interrupted.into());
        return CallReply {
            content: "Interrupted by the user before this tool finished.".to_string(),
            interrupted: true,
        };
    };

    // It ran, whatever it returned: a failed build is still the last thing that happened,
    // and the bar saying so is the point.
    let _ = tx.send(AgentEvent::DidTool { name: tool_name.to_string(), arguments: activity_arguments });

    let tool_result = result.unwrap_or_else(|e| {
        let _ = tx.send(Notice::ToolFailed(format!("{:?}", e)).into());
        crate::tools::ToolOutput { text: format!("Error: {}", e), images: Vec::new() }
    });

    // Held for after the round rather than answered here. A picture cannot go in the
    // `role: "tool"` reply — that message may carry text and nothing else, and a server
    // rejects an image part in one outright — so it has to arrive as a user message, and a
    // user message may not sit between an assistant turn and the tool results it is
    // waiting for.
    for (mime, data) in tool_result.images {
        let _ = tx.send(
            Notice::ImageAttached {
                path: image_label.clone(),
                // Back from base64 to the size of the thing it encodes, which is the figure
                // that means something to whoever reads the line.
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

    // The summary is the row that used to be all there was: the first line of the result,
    // cut to fit. The result itself is kept behind it now rather than thrown away, and
    // Ctrl+O opens it.
    //
    // A call that showed its own answer while it ran (research, library) is the
    // exception: a summary here would repeat the answer's first line immediately
    // underneath the answer.
    if !answer_shown {
        let _ = tx.send(AgentEvent::ToolResult(tool_result.clone()));
    }

    if repaired.is_empty() {
        return tool_result.into();
    }
    format!(
        "{tool_result}\n\n[haico] Your `{names}` argument ended with a second copy of \
         this call's other arguments written inside its text (`\", \"old_string\": \"...`). \
         That copy was cut off before the tool ran, so only the text before it \
         was used. End each string argument where its own text ends.",
        names = repaired.join("`, `"),
    )
    .into()
}

/// What happens once every call of a round has its reply: the loop guard, then whatever
/// joined the conversation while the round ran. `Some` ends the turn.
async fn after_round(agent: &mut Agent, tx: &EventSink, link: &TurnLink, guard: &mut LoopGuard) -> Option<TurnEnd> {
    // The tool log makes a loop obvious to whoever is watching it scroll past, but the
    // model reads none of that — every round looks to it like a fresh decision, which is
    // why it can repeat one call fifty times. Saying the repetition out loud in the
    // conversation is what breaks it, and it breaks it far earlier than the round cap. That
    // matters most in auto mode, where the cap is the only backstop and nobody is watching
    // it climb.
    match guard.verdict() {
        LoopVerdict::Break { tool, count } => {
            let _ = tx.send(Notice::LoopBroken { tool, count }.into());
            // Whatever was typed while this ran is deliberately left queued: it becomes the
            // next prompt, which is where the user takes over.
            return Some(TurnEnd::Failed);
        }
        LoopVerdict::Nudge { tool, count } => {
            let _ = tx.send(Notice::LoopNudge { tool: tool.clone(), count }.into());
            // A user message rather than a system one: mid-conversation system messages are
            // ignored by some local models and rejected outright by some servers, and this
            // one has to land. The order is legal here for the same reason the queued text
            // below is — every tool call in this round already has its result.
            agent.messages.push(json!({
                "role": "user",
                "content": format!(
                    "You have called '{}' with the same arguments {} times in a \
                     row. Get out of the loop — calling it again will not tell you \
                     anything you have not already been told. Take a different \
                     approach, or stop and say what you are stuck on.",
                    tool, count
                )
            }));
        }
        LoopVerdict::Fine => {}
    }

    // Anything typed while this round ran joins the conversation here, where every tool
    // call already has its result. Any sooner would put a user message between an
    // assistant turn and the tool results it is waiting for, which is a message order
    // servers reject outright. The echo carries what was typed; the message carries the
    // same text with its mentions read into it. Typing `@path` while the model works is as
    // ordinary as typing it at the prompt, and a mention that expanded in one place and not
    // the other would be the harder thing to explain.
    for text in link.take_queued() {
        let _ = tx.send(Notice::QueuedSent(text.clone()).into());
        let content = attach_mentions(agent, &text, tx).await;
        agent.messages.push(json!({"role": "user", "content": content}));
    }
    // Timers that came due while this round ran join at the same boundary and for the same
    // reason. Not sooner — and not by interrupting the round either: a timer is a reminder,
    // and the work in progress may well be the thing it was set to check on.
    let fired = agent.server.take_due_timers();
    if !fired.is_empty() {
        let _ = tx.send(Notice::TimersFired(fired.clone()).into());
        agent.messages.push(json!({
            "role": "user",
            "content": crate::timers::message_for(&fired)
        }));
    }
    None
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

/// Tests that drive a whole turn against a fake LLM server. Separate from the
/// module above because they need a Tokio runtime and the output capture, and
/// because what they are pinning is different in kind: not a helper's output but
/// the loop's invariants — the ones whose failure mode is the *next* request
/// being rejected rather than anything visible in this one.
#[cfg(test)]
mod turn_tests {
    #![allow(unused_imports)]
    use super::*;
    use crate::ask::*;
    use crate::config::*;
    use crate::llm::*;
    use crate::modes::*;
    use crate::out;
    use crate::{permissions, testutil};
    use crate::permissions::*;
    use crate::session::*;
    use crate::tools::*;
    use crate::turn::*;
    use std::env;
    use serde_json::Value;
    use crate::testutil::{looping_stream, serve_sequence, test_config};

    /// One turn against `bodies`, returning the agent so the history can be
    /// inspected, plus how many requests the server actually served.
    async fn run_one(bodies: &'static [&'static str], input: &str) -> (Agent, usize) {
        run_one_in(bodies, input, Mode::Normal).await
    }

    /// As `run_one`, in a chosen mode. Auto mode matters here because it is the
    /// only one that skips `MAX_TOOL_ROUNDS`, so it is the only one where a
    /// runaway turn has nothing but the repetition guard to stop it.
    async fn run_one_in(
        bodies: &'static [&'static str],
        input: &str,
        mode: Mode,
    ) -> (Agent, usize) {
        run_and_watch(bodies, input, mode, Reasoning::Keep).await.0
    }

    /// As `run_one_in`, but also every event the turn emitted.
    ///
    /// This is what the event stream bought the tests: the turn is driven with no
    /// frontend at all — no terminal, no keyboard, no presenter — and what it
    /// *said* can be asserted on directly instead of inferred from output. An
    /// `Ask` arriving here is answered by nobody, so its reply channel drops and
    /// the turn hears a refusal, which is exactly what a test wants: no prompt
    /// can hang the suite.
    ///
    /// The reasoning policy is a parameter for the same reason the mode is: it
    /// changes what the turn does after a reply rather than what it says to the
    /// server, so a fixture cannot express it.
    async fn run_and_watch(
        bodies: &'static [&'static str],
        input: &str,
        mode: Mode,
        reasoning: Reasoning,
    ) -> ((Agent, usize), Vec<AgentEvent>) {
        let (url, served) = serve_sequence(bodies).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        agent.mode = mode;
        agent.reasoning = reasoning;

        let link = TurnLink::new(agent.mode);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, input, false, &tx, &link).await;
        agent.mode = link.mode();

        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        let count = *served.lock().unwrap();
        ((agent, count), seen)
    }

    /// Every `Notice` the turn emitted, in order. The notices are the part of the
    /// vocabulary that derives `PartialEq`, which is why assertions use them.
    fn notices(events: &[AgentEvent]) -> Vec<Notice> {
        events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Notice(notice) => Some(notice.clone()),
                _ => None,
            })
            .collect()
    }

    /// Every `role: "tool"` reply in the history, as `(tool_call_id, content)`.
    fn tool_replies(messages: &[Value]) -> Vec<(String, String)> {
        messages
            .iter()
            .filter(|m| m.get("role").and_then(Value::as_str) == Some("tool"))
            .map(|m| {
                (
                    m.get("tool_call_id").and_then(Value::as_str).unwrap_or_default().to_string(),
                    m.get("content").and_then(Value::as_str).unwrap_or_default().to_string(),
                )
            })
            .collect()
    }

    /// A `--prompt` run decides whether to exit, follow up or report a failure from
    /// what `run_turn` hands back, so an answer and a failed request must not look
    /// alike. A server that answers garbage is the failure that needs no timing.
    #[tokio::test]
    async fn a_turn_says_whether_it_ended_in_an_answer() {
        let _capture = out::capture_guard();
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"all done"}}]}"#;
        let cases: [(&'static [&'static str], TurnEnd); 2] =
            [(&[FINAL], TurnEnd::Answered), (&["not json"], TurnEnd::Failed)];
        for (bodies, expected) in cases {
            let (url, _) = serve_sequence(bodies).await;
            let cwd = env::current_dir().unwrap();
            let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
            agent.mode = Mode::Auto;
            let link = TurnLink::new(agent.mode);
            let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
            assert_eq!(run_turn(&mut agent, "go", false, &tx, &link).await, expected, "for {bodies:?}");
        }
    }

    /// A timer that comes due while a turn is working joins at the round boundary,
    /// after the tool replies — the only place a user message may go mid-turn — and
    /// says it came from a timer. One that is not due yet stays armed.
    #[tokio::test]
    async fn a_due_timer_joins_the_turn_after_the_tool_replies() {
        let _capture = out::capture_guard();
        let (url, served) = serve_sequence(&[
            r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"get_date","arguments":"{}"}}]}}]}"#,
            r#"{"choices":[{"message":{"role":"assistant","content":"checked"}}]}"#,
        ])
        .await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
        {
            use crate::timers::Schedule;
            use std::time::{Duration, Instant};
            let mut timers = agent.server.timers.lock().unwrap();
            // Set two minutes ago for one minute: due now.
            let past = Instant::now() - Duration::from_secs(120);
            timers.add(Schedule::Once(Duration::from_secs(60)), "check the tests", past, chrono::Local::now()).unwrap();
            timers.add(Schedule::Once(Duration::from_secs(3600)), "much later", Instant::now(), chrono::Local::now()).unwrap();
        }

        let link = TurnLink::new(Mode::Normal);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(run_turn(&mut agent, "go", false, &tx, &link).await, TurnEnd::Answered);
        assert_eq!(*served.lock().unwrap(), 2);

        let roles: Vec<&str> = agent.messages.iter().filter_map(|m| m["role"].as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "user", "assistant"]);
        let note = agent.messages[4]["content"].as_str().unwrap();
        assert!(note.starts_with("[Timer 1 fired"), "{note}");
        assert!(note.ends_with("check the tests"), "{note}");

        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(notices(&seen).iter().any(|n| matches!(n, Notice::TimersFired(f) if f.len() == 1 && f[0].id == 1)));
        let left: Vec<u32> = agent.server.timers.lock().unwrap().list().iter().map(|t| t.id).collect();
        assert_eq!(left, vec![2]);

        // And they go with the conversation.
        agent.clear();
        assert!(agent.server.timers.lock().unwrap().is_empty());
    }

    /// The invariant that costs the most when it breaks, and shows nothing when
    /// it does: **every `tool_call_id` must get a `role: "tool"` reply**, on
    /// every path. An unanswered id is accepted by this request and rejected by
    /// the *next* one, server-side, which is why it took a release to find and
    /// why it is worth a test that asserts on the history rather than the screen.
    ///
    /// Both malformed shapes are covered at once because both used to `continue`
    /// past the reply: a call with no function name, and one whose arguments are
    /// not JSON. Local models emit the second often enough that it was the
    /// original bug report.
    #[tokio::test]
    async fn every_tool_call_id_is_answered_even_when_malformed() {
        let _capture = out::capture_guard();
        const MALFORMED: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[
                {"id":"call_no_name","type":"function","function":{"arguments":"{}"}},
                {"id":"call_bad_args","type":"function","function":{"name":"read","arguments":"not json at all"}}
            ]}}]}"#;
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"all done"}}]}"#;

        let (agent, served) = run_one(&[MALFORMED, FINAL], "do something").await;
        let replies = tool_replies(&agent.messages);
        let ids: Vec<&str> = replies.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.contains(&"call_no_name") && ids.contains(&"call_bad_args"),
            "an id went unanswered — the next request would be rejected. Replies: {replies:?}"
        );

        // Answered *with something the model can act on*, not merely answered:
        // the point of replying to a malformed call is to say what was wrong.
        let bad_args = replies.iter().find(|(id, _)| id == "call_bad_args").unwrap();
        assert!(
            bad_args.1.contains("read"),
            "the reply does not name the tool that failed: {:?}",
            bad_args.1
        );

        // And the turn went on to a second request and finished on the answer,
        // rather than stopping at the malformed round.
        assert_eq!(served, 2, "the turn did not come back for a final answer");
        assert_eq!(
            agent.messages.last().and_then(|m| m.get("content")).and_then(Value::as_str),
            Some("all done")
        );
    }

    /// `@path` in a message puts the file in the message, which is the whole
    /// point: the turn must reach the model with the contents already there,
    /// rather than spending a round on a `read` the user has already asked for.
    #[tokio::test]
    async fn a_mentioned_file_arrives_in_the_message() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"seen"}}]}"#;

        let dir = testutil::TempDir::new("mention-turn");
        let file = dir.join("notes.txt");
        std::fs::write(&file, "the quick brown fox").unwrap();

        let input = format!("what is in @{}?", file.display());
        let (agent, served) = run_one(&[ANSWER], &input).await;

        // One request, not two: nothing was fetched by a tool call.
        assert_eq!(served, 1);
        let sent = agent
            .messages
            .iter()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap();
        assert!(sent.contains("the quick brown fox"), "contents missing from: {sent}");
        // The instruction is still the last thing the model reads, and the
        // reference it was written as is still in it.
        assert!(sent.trim_end().ends_with(&input), "the request is not last in: {sent}");

        // Read through the `read` tool, so the file counts as seen and an edit
        // is not refused for a file the model was just handed.
        assert!(
            agent.server.check_may_modify(&file).is_none(),
            "the mentioned file was not marked read — the saved round is spent re-reading it"
        );
    }

    /// A mentioned image cannot go in the text, so the message stops being
    /// text: `content` becomes the parts array, with the image as a data URL.
    #[tokio::test]
    async fn a_mentioned_image_becomes_a_content_part() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"seen"}}]}"#;

        let dir = testutil::TempDir::new("mention-image-turn");
        let png = dir.join("shot.png");
        // A signature and some bytes after it: nothing decodes this, and what
        // is asserted is the shape of the message, not the picture.
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03").unwrap();
        let notes = dir.join("notes.txt");
        std::fs::write(&notes, "some prose").unwrap();

        let input = format!("what is in @{} — see also @{}", png.display(), notes.display());
        let (agent, served) = run_one(&[ANSWER], &input).await;
        assert_eq!(served, 1);

        let message = agent
            .messages
            .iter()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .cloned()
            .unwrap();
        let parts = message["content"].as_array().expect("content should be a parts array");

        // The image first, the text last — the request is what the model should
        // read at the end.
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "image_url");
        assert!(
            parts[0]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,"),
            "not a png data URL: {}",
            parts[0]["image_url"]["url"]
        );
        assert_eq!(parts[1]["type"], "text");

        // The text part still carries the text file and the request itself, so
        // mixing the two kinds in one message loses neither.
        let text = parts[1]["text"].as_str().unwrap();
        assert!(text.contains("some prose"), "the text mention is missing: {text}");
        assert!(text.trim_end().ends_with(&input), "the request is not last in: {text}");

        // And the base64 is costed as an image rather than as prose. Left alone,
        // a screenshot reads as hundreds of thousands of tokens and auto mode
        // starts compacting a history that compaction cannot shrink. Measured
        // against the same message with a realistic payload in it, since the
        // twelve-byte one above would pass either way.
        let mut heavy = message.clone();
        heavy["content"][0]["image_url"]["url"] =
            Value::String(format!("data:image/png;base64,{}", "A".repeat(400_000)));
        let estimate = estimate_tokens(&[heavy]);
        assert!(
            estimate < 2000,
            "a 300 KB image is being counted as {estimate} tokens of text"
        );
    }

    /// The first reply after an auto-compaction may not end the turn by calling nothing: on a
    /// p2p_auth run it was one sentence of intent, and the unattended run exited as finished.
    #[tokio::test]
    async fn a_reply_that_calls_nothing_right_after_a_compaction_does_not_end_the_turn() {
        let _capture = out::capture_guard();
        let bodies: &'static [&'static str] = &[
            // The compactor's notes.
            r#"{"choices":[{"message":{"role":"assistant","content":"Task: x\nCurrent state: tests not run\nNext step: run the tests"}}]}"#,
            // Intent, and nothing called.
            r#"{"choices":[{"message":{"role":"assistant","content":"Let me check the current state."}}]}"#,
            // Sent back, it answers properly.
            r#"{"choices":[{"message":{"role":"assistant","content":"All tests pass."}}]}"#,
        ];
        let (url, served) = serve_sequence(bodies).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
        agent.mode = Mode::Auto;
        // Tiny, so the history is over the compaction line on the first round; and a history
        // with something in it, since one message is nothing to compact.
        agent.context_tokens = 100;
        agent.messages.push(serde_json::json!({"role": "user", "content": "build it"}));
        agent.messages.push(serde_json::json!({"role": "assistant", "content": "Built half of it."}));

        let link = TurnLink::new(Mode::Auto);
        let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(run_turn(&mut agent, "go", false, &tx, &link).await, TurnEnd::Answered);
        assert_eq!(*served.lock().unwrap(), 3);

        let last = agent.messages.last().unwrap();
        assert_eq!(last["content"], "All tests pass.");
        assert!(
            agent.messages.iter().any(|m| m["role"] == "user"
                && m["content"].as_str().is_some_and(|c| c.contains("You replied without calling a tool"))),
            "the note is missing"
        );
    }

    /// The size of the history comes from the server's count once one has arrived, plus an
    /// estimate of only what was added after it; a history that shrank since falls back to
    /// the estimate with the tools array counted. A p2p_auth run reached 130,820 of 131,072
    /// tokens on an estimate that still read 107k, so compaction never started.
    #[tokio::test]
    async fn the_context_is_sized_by_the_servers_count() {
        let _capture = out::capture_guard();
        let (mut agent, _) = run_one(
            &[r#"{"choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":50000,"completion_tokens":3,"total_tokens":50003}}"#],
            "hi",
        )
        .await;
        let measured = agent.measured.expect("the reply's usage was not recorded");
        assert_eq!(measured.prompt_tokens, 50000);
        assert!(agent.context_is_measured());

        // Everything after the measured request is estimated on top of the count.
        let grown = estimate_tokens(&agent.messages) - measured.estimate;
        assert_eq!(agent.context_used(), 50000 + grown);
        agent.messages.push(serde_json::json!({"role": "user", "content": "x".repeat(4000)}));
        assert_eq!(agent.context_used(), 50000 + grown + estimate_tokens(&agent.messages[agent.messages.len() - 1..]));

        // A history that shrank is not the one that was counted.
        agent.messages.truncate(1);
        assert!(!agent.context_is_measured());
        assert_eq!(agent.context_used(), estimate_tokens(&agent.messages) + estimate_tokens(&agent.tools));

        // A server that reports no usage leaves the estimate in charge.
        let (agent, _) = run_one(&[r#"{"choices":[{"message":{"role":"assistant","content":"hello"}}]}"#], "hi").await;
        assert!(agent.measured.is_none());
        assert_eq!(agent.context_used(), estimate_tokens(&agent.messages) + estimate_tokens(&agent.tools));
    }

    /// The edit from the p2p_auth trace: `new_string` carrying a second copy of `old_string`
    /// after an escaped quote. The file must get only the intended text, and the reply must
    /// say what was cut, or the model goes on making the same slip.
    #[tokio::test]
    async fn an_edit_with_its_old_string_leaked_into_new_string_is_repaired_and_said_to_be() {
        let _capture = out::capture_guard();

        let dir = testutil::TempDir::new("leaked-edit");
        let file = dir.join("lib.rs");
        std::fs::write(&file, "let (tx, mut rx) = channel(64);\nlet node = 1;\n").unwrap();
        let path = file.display().to_string();

        let call = |id: &str, name: &str, arguments: serde_json::Value| -> &'static str {
            Box::leak(
                format!(
                    r#"{{"choices":[{{"message":{{"role":"assistant","content":null,
                    "tool_calls":[{{"id":"{id}","type":"function","function":{{"name":"{name}","arguments":{}}}}}]}}}}]}}"#,
                    serde_json::to_string(&arguments.to_string()).unwrap()
                )
                .into_boxed_str(),
            )
        };
        let old = "let (tx, mut rx) = channel(64);";
        let leaked = format!("let (tx, rx) = channel(64);\", \"old_string\": \"{}\"}}", old);
        let edit = serde_json::json!({"file_path": path, "new_string": leaked, "old_string": old});
        let bodies: &'static [&'static str] = Box::leak(Box::new([
            call("c1", "read", serde_json::json!({"file_path": path})),
            call("c2", "edit", edit.clone()),
            r#"{"choices":[{"message":{"role":"assistant","content":"done"}}]}"#,
        ]));
        let (url, requests) = serve_sequence(bodies).await;

        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
        agent.workspace.add(dir.path()).unwrap();
        // Approved up front: in a test nobody answers the prompt, which reads as a refusal.
        if let permissions::Gate::Ask { keys, .. } = permissions::gate_for(&agent.workspace, "edit", &edit) {
            agent.permissions.allow(&keys);
        }

        let link = TurnLink::new(Mode::Normal);
        let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, "fix it", false, &tx, &link).await;
        assert_eq!(*requests.lock().unwrap(), 3);

        assert_eq!(std::fs::read_to_string(&file).unwrap(), "let (tx, rx) = channel(64);\nlet node = 1;\n");
        let reply = agent
            .messages
            .iter()
            .find(|m| m["tool_call_id"] == "c2")
            .and_then(|m| m["content"].as_str())
            .unwrap()
            .to_string();
        assert!(reply.contains("Your `new_string` argument ended with a second copy"), "{reply}");
    }

    /// `read` on an image returns the picture, and the picture cannot ride in
    /// the `role: "tool"` reply — that message may carry text and nothing else.
    /// So it arrives as a user message, and only once every tool call in the
    /// round has its reply.
    #[tokio::test]
    async fn reading_an_image_puts_it_in_a_user_message_after_the_tool_reply() {
        let _capture = out::capture_guard();

        let dir = testutil::TempDir::new("read-image");
        let png = dir.join("shot.png");
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03").unwrap();

        // The model asks to read it, then answers. The path is escaped for JSON
        // because a Windows path is full of backslashes.
        let call = format!(
            r#"{{"choices":[{{"message":{{"role":"assistant","content":null,
            "tool_calls":[{{"id":"c1","type":"function","function":{{"name":"read","arguments":{}}}}}]}}}}]}}"#,
            serde_json::to_string(
                &serde_json::json!({"file_path": png.display().to_string()}).to_string()
            )
            .unwrap()
        );
        let (url, requests) = serve_sequence(Box::leak(Box::new([
            &*Box::leak(call.into_boxed_str()),
            r#"{"choices":[{"message":{"role":"assistant","content":"a tiny png"}}]}"#,
        ])))
        .await;

        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
        // The temp directory is outside the project, so without this the read is
        // refused by the workspace and never reaches the image path at all.
        agent.workspace.attach(&png);

        // `read` is on the auto-allowed list, so no prompt stands in the way.
        let link = TurnLink::new(Mode::Normal);
        let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, "look at the screenshot", false, &tx, &link).await;
        assert_eq!(*requests.lock().unwrap(), 2);

        let roles: Vec<&str> =
            agent.messages.iter().filter_map(|m| m["role"].as_str()).collect();
        // system, user, assistant(tool_calls), tool, user(image), assistant
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "user", "assistant"]);

        // The tool reply is text and names the file. It must not have become an
        // array: a server rejects an image part in a tool message outright.
        let reply = &agent.messages[3];
        let text = reply["content"].as_str().expect("a tool reply must be a string");
        assert!(text.contains("PNG"), "the reply does not say what the file is: {text}");

        // And the picture is in the user message that follows it.
        let parts = agent.messages[4]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "image_url");
        assert!(parts[0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    /// A model with no vision head rejects the request carrying a picture, and
    /// every request after it, so the conversation is dead until the picture
    /// leaves. The turn takes it out once and tries again rather than handing
    /// back a history that can only fail.
    #[tokio::test]
    async fn a_refused_image_is_stripped_and_the_request_retried() {
        let _capture = out::capture_guard();

        let dir = testutil::TempDir::new("strip-image");
        let png = dir.join("shot.png");
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03").unwrap();

        // The server refuses the first request outright, then answers. That is
        // the shape of a text-only model being handed an image.
        let (url, served) = serve_sequence(&[
            "not json, and not a 200 either",
            r#"{"choices":[{"message":{"role":"assistant","content":"no picture here"}}]}"#,
        ])
        .await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);

        let link = TurnLink::new(Mode::Normal);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, &format!("look at @{}", png.display()), false, &tx, &link).await;

        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(
            notices(&seen).iter().any(|n| matches!(n, Notice::ImagesStripped(_))),
            "the retry never happened: {:?}",
            notices(&seen)
        );
        assert_eq!(*served.lock().unwrap(), 2, "the request was not tried a second time");

        // No image left anywhere, and the message that carried one still says
        // something rather than being emptied.
        for message in &agent.messages {
            if let Some(parts) = message["content"].as_array() {
                assert!(
                    !parts.iter().any(|p| p["type"] == "image_url"),
                    "an image survived the strip"
                );
                assert!(!parts.is_empty(), "a message was left with no content at all");
            }
        }
        let user = message_text(&agent.messages[1]);
        assert!(user.contains("removed"), "the model was not told why: {user}");
    }

    /// A message carrying an image must still reach a text summarizer as
    /// something, or the turn it began summarizes as though nobody spoke.
    #[test]
    fn an_image_message_still_has_text_for_the_summarizer() {
        let message = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "why is this button grey?"}
            ]
        });
        let text = message_text(&message);
        assert!(text.contains("why is this button grey?"));
        assert!(text.contains("[an image]"), "the image left no mark: {text}");
        assert!(!text.contains("AAAA"), "the base64 reached the summarizer: {text}");
    }

    /// A mentioned file outside every root comes into reach, and only that file
    /// does. `/add` widens to a directory because that is what it is asked for;
    /// a mention is asked about one file, and `@~/.ssh/config` must not hand
    /// over the keys beside it.
    #[tokio::test]
    async fn a_mention_outside_the_workspace_reaches_that_file_alone() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"seen"}}]}"#;

        // Outside the working directory the test agent is built in, which is the
        // crate root.
        let dir = testutil::TempDir::new("mention-outside");
        let named = dir.join("named.txt");
        let sibling = dir.join("sibling.txt");
        std::fs::write(&named, "in reach").unwrap();
        std::fs::write(&sibling, "not in reach").unwrap();

        let (agent, _) = run_one(&[ANSWER], &format!("read @{}", named.display())).await;

        assert!(agent.workspace.may_write(&named), "the named file is still refused");
        assert!(
            !agent.workspace.may_read(&sibling),
            "the mention opened the whole directory, not the one file"
        );
        // And the model is told, or it finds out through a refused call instead.
        // The note is built per request from the workspace, so this is what the
        // next request would carry rather than what the system prompt holds.
        let note = agent.workspace.note();
        assert!(
            note.contains(&display_path(&absolute_path(&named))),
            "the attached file is not in the workspace note: {note}"
        );
        assert!(
            !note.contains(&display_path(&absolute_path(&sibling))),
            "the note names a file that was never attached: {note}"
        );
    }

    /// A reply with no tool calls ends the turn on the answer, and the user's
    /// message and the assistant's both land in the history — the ordinary case,
    /// asserted so that the interesting ones above have a baseline.
    #[tokio::test]
    async fn a_plain_answer_ends_the_turn() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"hello back"}}]}"#;

        let (agent, served) = run_one(&[ANSWER], "hello").await;
        assert_eq!(served, 1, "a turn with no tool calls should take one request");
        let roles: Vec<&str> =
            agent.messages.iter().filter_map(|m| m.get("role").and_then(Value::as_str)).collect();
        assert_eq!(roles.last(), Some(&"assistant"));
        assert!(roles.contains(&"user"));
        assert!(tool_replies(&agent.messages).is_empty());
    }

    /// `MAX_TOOL_ROUNDS` is not the only thing stopping a runaway turn, and in
    /// auto mode it is not applied at all — the guard there is the identical call
    /// repeated with nothing in between. A server that answers with the same
    /// tool call forever must therefore still terminate.
    #[tokio::test]
    async fn an_identical_call_repeated_forever_still_terminates() {
        let _capture = out::capture_guard();
        // `serve_sequence` repeats its last body forever, so this is a model that
        // never stops asking for the same thing.
        const SAME: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"same","type":"function","function":{"name":"read","arguments":"also not json"}}]}}]}"#;

        let (agent, served) = run_one(&[SAME], "loop forever").await;
        // The bound is what matters, not its exact value: without one this test
        // would hang rather than fail.
        assert!(served > 1, "the loop never came back for a second round");
        // The claim is specifically that the *repetition* guard stopped it and
        // not the round cap: `LOOP_BREAK` is 5 against a cap of 50, so a count
        // in single figures is the identical-call run firing. A range rather
        // than an exact number, because the nudge rounds are free to move.
        assert!(
            served <= 8,
            "a repeated identical call ran for {served} rounds — that is the round cap \
             stopping it, not the loop guard"
        );
        // However it stopped, every id it made is still answered.
        let unanswered = agent
            .messages
            .iter()
            .filter(|m| m.get("tool_calls").is_some_and(|c| !c.is_null()))
            .count()
            - tool_replies(&agent.messages).len();
        assert_eq!(unanswered, 0, "a round left its tool call unanswered");
    }

    /// The same runaway turn in **auto mode**, which is where it actually
    /// mattered. Auto mode skips `MAX_TOOL_ROUNDS` deliberately — a fixed number
    /// of rounds is a poor stand-in for "this is going nowhere", and a long job
    /// that hits it unattended has nobody to say "continue" — so the repetition
    /// guard is the *only* thing bounding it.
    ///
    /// This is the test that found the bug it now guards: the run counter used to
    /// be incremented after `parse_tool_arguments`, whose error arm `continue`s,
    /// so a model repeating a call whose arguments are not JSON never advanced
    /// the run. In normal mode the cap still caught it. In auto mode nothing did,
    /// and the turn span forever with no user present to stop it. Written second,
    /// because writing it first would have hung the suite rather than failed.
    #[tokio::test]
    async fn auto_mode_still_bounds_a_repeated_malformed_call() {
        let _capture = out::capture_guard();
        const SAME: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"same","type":"function","function":{"name":"read","arguments":"still not json"}}]}}]}"#;

        let (agent, served) = run_one_in(&[SAME], "loop forever", Mode::Auto).await;
        assert!(
            served <= 8,
            "auto mode ran {served} rounds of an identical malformed call: with no round 
             cap there, the repetition guard is all there is and it did not fire"
        );
        assert_eq!(
            agent.messages.iter().filter(|m| m.get("tool_calls").is_some_and(|c| !c.is_null())).count()
                - tool_replies(&agent.messages).len(),
            0,
            "a round left its tool call unanswered"
        );
    }

    /// The failure that motivated `repetition`: a model that never calls a tool
    /// and never finishes, only restates itself, which the tool-call loop guard
    /// cannot see. The reply is cut off and dropped, the model is shown what it
    /// was repeating, and the next reply is taken normally. Auto mode, because
    /// unattended is where it cost the most.
    #[tokio::test]
    async fn a_reply_that_repeats_itself_is_dropped_and_the_model_told() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"fixed it"}}]}"#;
        let bodies: &'static [&'static str] = Box::leak(Box::new([looping_stream(), ANSWER]));

        let ((agent, served), events) = run_and_watch(bodies, "debug it", Mode::Auto, Reasoning::Keep).await;

        assert_eq!(served, 2);
        assert!(
            notices(&events).iter().any(|n| matches!(n, Notice::Repeating { retrying: true, .. })),
            "notices: {:?}",
            notices(&events)
        );
        // The looping reply never reached the history: neither its text nor the
        // content it would have ended on.
        let history = serde_json::to_string(&agent.messages).unwrap();
        assert!(!history.contains("finished anyway"), "the looping reply was stored");
        // Folded into the user's own message, since the loop was the turn's first reply.
        let user: Vec<&Value> = agent.messages.iter().filter(|m| m["role"] == "user").collect();
        assert_eq!(user.len(), 1, "two user messages in a row");
        let asked = user[0]["content"].as_str().unwrap();
        assert!(asked.starts_with("debug it") && asked.contains("repeating the same passage"), "{asked}");
        assert_eq!(agent.messages.last().unwrap()["content"], "fixed it");
    }

    /// And when telling it does not help, the turn gives up instead of paying
    /// for the same loop forever — which in auto mode, with no round cap, is
    /// the only thing that would stop it.
    #[tokio::test]
    async fn a_model_that_keeps_repeating_itself_ends_the_turn() {
        let _capture = out::capture_guard();
        let bodies: &'static [&'static str] = Box::leak(Box::new([looping_stream()]));
        let (url, served) = serve_sequence(bodies).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        agent.mode = Mode::Auto;
        let link = TurnLink::new(agent.mode);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();

        assert_eq!(run_turn(&mut agent, "debug it", false, &tx, &link).await, TurnEnd::Failed);
        assert_eq!(*served.lock().unwrap(), 2);
        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(notices(&seen).iter().any(|n| matches!(n, Notice::Repeating { retrying: false, .. })));
    }

    /// A review that ends without a verdict is reported to the model as a
    /// temporary failure to retry, and the user is not asked: auto mode is for
    /// walking away. Worded as a refusal it was taken as final — an unattended
    /// model whose `write_file` met a review cut off at its token cap wrote its
    /// files through `echo` for the rest of the run — and sent to the prompt, it
    /// parked the run on a dialog nobody was there to answer.
    #[tokio::test]
    async fn a_review_without_a_verdict_is_a_temporary_error_to_retry() {
        let _capture = out::capture_guard();
        const CALL: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"run_command","arguments":"{\"command\":\"cargo --version\"}"}}]}}]}"#;
        // What a review cut off at its cap comes back as: all reasoning, no answer.
        const CUT_OFF: &str = r#"{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"length"}]}"#;
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"done"}}]}"#;
        let (url, served) = serve_sequence(&[CALL, CUT_OFF, FINAL]).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        agent.mode = Mode::Auto;
        let link = TurnLink::new(agent.mode);
        // A frontend that is there but never answers, like a REPL whose user went
        // for coffee: a prompt sent to it would wait forever, so the timeout is
        // what turns "it asked" into a failure rather than a hung suite.
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();

        let turn = run_turn(&mut agent, "check the toolchain", false, &tx, &link);
        tokio::time::timeout(std::time::Duration::from_secs(10), turn)
            .await
            .expect("the turn waited on a prompt nobody answered");

        assert_eq!(*served.lock().unwrap(), 3, "call, review, then the reply to the retry note");
        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(!seen.iter().any(|e| matches!(e, AgentEvent::Ask(_))), "the user was asked");
        assert!(
            notices(&seen).iter().any(|n| matches!(n, Notice::ReviewFailed { tool, .. } if tool == "run_command")),
            "notices: {:?}",
            notices(&seen)
        );
        let replies = tool_replies(&agent.messages);
        assert_eq!(replies.len(), 1);
        let (id, content) = &replies[0];
        assert_eq!(id, "c1");
        assert!(
            content.starts_with("Temporary error while evaluating the 'run_command' call")
                && content.contains("make the same call again"),
            "{content}"
        );
    }

    /// The turn narrates itself, and this is the assertion that could not be
    /// written before: driven with no frontend at all, what it *said* is checked
    /// directly rather than inferred from terminal output.
    #[tokio::test]
    async fn a_malformed_call_is_narrated_as_well_as_answered() {
        let _capture = out::capture_guard();
        const MALFORMED: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"bad","type":"function","function":{"name":"read","arguments":"nope"}}]}}]}"#;
        const FINAL: &str = r#"{"choices":[{"message":{"role":"assistant","content":"done"}}]}"#;

        let (_, events) =
            run_and_watch(&[MALFORMED, FINAL], "go", Mode::Normal, Reasoning::Keep).await;
        let said = notices(&events);
        assert!(
            said.iter().any(|n| matches!(
                n,
                Notice::ToolArgumentsInvalid { tool, .. } if tool == "read"
            )),
            "the turn did not say which tool had bad arguments: {said:?}"
        );

        // A reply is opened and closed exactly once per round, on every path.
        // The presenter's fold and half-rendered markdown line depend on that
        // pairing, and a missing ReplyEnd leaks a fold into the next round.
        let starts = events.iter().filter(|e| matches!(e, AgentEvent::ReplyStart)).count();
        let ends = events.iter().filter(|e| matches!(e, AgentEvent::ReplyEnd)).count();
        assert_eq!(starts, 2, "two rounds should open two replies");
        assert_eq!(starts, ends, "every reply must be closed: {starts} opened, {ends} closed");

        // And the turn reported its own end, which is the line the summary is
        // drawn from.
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Finished { .. })));
    }

    /// Compression replaces the chain of thought in the history, so the
    /// transcript is the only place the summary can still be read — which is why
    /// the notice carries the text as well as the two figures. Reporting a saving
    /// and nothing else leaves the reader no way to judge whether what survived
    /// was worth keeping, and the original is gone by then.
    #[tokio::test]
    async fn a_compressed_trace_is_reported_with_the_text_that_replaced_it() {
        let _capture = out::capture_guard();
        // Well over the 800-token threshold, so the turn spends a request on
        // compressing it and the second body is the compressor's answer.
        // `serve_sequence` wants `'static` bodies and a trace this long has to be
        // built rather than written out, so it is leaked — a test process is the
        // one place where that costs nothing.
        let reply: &'static str = Box::leak(
            format!(
                "{{\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"Done.\",\
                 \"reasoning_content\":\"{}\"}}}}]}}",
                "z".repeat(8_000)
            )
            .into_boxed_str(),
        );
        const SUMMARY: &str = r#"{"choices":[{"message":{"role":"assistant",
            "content":"Tried the regex path, too slow."}}]}"#;
        let bodies: &'static [&'static str] = Box::leak(Box::new([reply, SUMMARY]));

        let ((agent, _), events) =
            run_and_watch(bodies, "go", Mode::Normal, Reasoning::Compress).await;
        let said = notices(&events);
        let reported = said.iter().find_map(|n| match n {
            Notice::ReasoningCompressed { summary, .. } => Some(summary.clone()),
            _ => None,
        });
        assert_eq!(
            reported.as_deref(),
            Some("Tried the regex path, too slow."),
            "the compression was reported without the text it produced: {said:?}"
        );
        // And what was reported is what the history actually carries — the point
        // of reading it back out of the message rather than passing it alongside
        // the figures.
        assert_eq!(
            agent.messages.last().unwrap()["reasoning_content"].as_str(),
            reported.as_deref()
        );
    }

    /// Cancellation is a message, not a dropped future — see
    /// `TurnLink::until_cancelled`. Cancelling before the turn starts means it
    /// stops at its first interruptible point, and the invariant still holds: the
    /// user message is in the history and nothing is left half-written.
    #[tokio::test]
    async fn a_cancelled_turn_stops_without_orphaning_anything() {
        let _capture = out::capture_guard();
        const ANSWER: &str = r#"{"choices":[{"message":{"role":"assistant","content":"hello"}}]}"#;

        let (url, served) = serve_sequence(&[ANSWER]).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        let link = TurnLink::new(agent.mode);
        // Cancelled up front: `until_cancelled` has to see a flag that was already
        // set, not only a transition after it subscribed.
        link.cancel();

        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        run_turn(&mut agent, "hello", false, &tx, &link).await;

        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            seen.push(event);
        }
        assert!(
            notices(&seen).contains(&Notice::Interrupted),
            "a cancelled turn should say so: {:?}",
            notices(&seen)
        );
        assert_eq!(*served.lock().unwrap(), 0, "cancelled before the first request, so none went");
        // The user's message is there and nothing else is: no assistant message
        // was pushed, so no tool call can be unanswered.
        let roles: Vec<&str> =
            agent.messages.iter().filter_map(|m| m.get("role").and_then(Value::as_str)).collect();
        assert_eq!(roles.last(), Some(&"user"));
        assert!(tool_replies(&agent.messages).is_empty());
    }

    /// Esc during the first of two calls in one reply. The turn used to stop at the
    /// call it interrupted and leave the second id unanswered, and the next request
    /// was rejected for it.
    #[tokio::test]
    async fn an_interrupt_mid_round_still_answers_every_call_of_the_reply() {
        let _capture = out::capture_guard();
        let sleeper = if cfg!(windows) { "ping -n 30 127.0.0.1 > NUL" } else { "sleep 30" };
        let reply = Box::leak(
            serde_json::json!({"choices": [{"message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "slow", "type": "function", "function": {
                        "name": "run_command",
                        "arguments": serde_json::json!({"command": sleeper}).to_string()}},
                    {"id": "never", "type": "function", "function": {
                        "name": "get_date", "arguments": "{}"}}
                ]
            }}]})
            .to_string()
            .into_boxed_str(),
        );
        let bodies: &'static [&'static str] = Box::leak(Box::new([&*reply]));

        let (url, served) = serve_sequence(bodies).await;
        let cwd = env::current_dir().unwrap();
        let mut agent = Agent::new(test_config(&url), "test-model".to_string(), &cwd);
        let link = TurnLink::new(agent.mode);
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();

        // The frontend: approve the command, and press Esc once it is running.
        let frontend = async {
            while let Some(event) = events.recv().await {
                match event {
                    AgentEvent::Ask(Ask::Permission { reply, .. }) => {
                        let _ = reply.send(PermissionResponse::Allow);
                    }
                    AgentEvent::ToolStarted { name, .. } if name == "run_command" => link.cancel(),
                    _ => {}
                }
            }
        };
        let turn = async {
            let end = run_turn(&mut agent, "go", false, &tx, &link).await;
            drop(tx);
            end
        };
        let (end, ()) = tokio::join!(turn, frontend);

        assert_eq!(end, TurnEnd::Interrupted);
        assert_eq!(*served.lock().unwrap(), 1);
        let replies = tool_replies(&agent.messages);
        let ids: Vec<&str> = replies.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["slow", "never"], "every call of the reply needs its reply");
        assert!(replies[1].1.contains("Not run"), "got: {}", replies[1].1);
    }
}
