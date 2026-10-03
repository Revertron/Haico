//! The tool server: shared state, dispatch, and the MCP handler.
//!
//! The tools themselves live in the submodules. rmcp generates one router per impl block,
//! and `new` merges them, which is what lets the tools be split across files at all.

use crate::llm::LMStudioClient;
use crate::util::{file_key, modified_time, truncate_at_boundary};
use anyhow::Result;
use lru::LruCache;
use rmcp::handler::server::{tool::ToolRouter, wrapper::Parameters};
use rmcp::model::ContentBlock;
use rmcp::{tool_handler, ServerHandler};
use serde_json::{json, Value};
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

mod basics;
mod crates;
mod files;
pub(crate) mod library;
mod memory;
mod research;
mod shell;
mod skills;
mod timers;
mod web;

pub(crate) use basics::*;
pub(crate) use crates::*;
pub(crate) use files::*;
pub(crate) use library::*;
pub(crate) use memory::*;
pub(crate) use research::*;
pub(crate) use shell::*;
pub(crate) use skills::*;
pub(crate) use timers::*;
pub(crate) use web::*;

#[derive(Clone)]
pub(crate) struct MyMcpServer {
    pub(crate) tool_router: ToolRouter<Self>,
    pub(crate) llm_client: LMStudioClient,
    // LRU cache of the last ~10 fetched pages, keyed by (upgraded) URL. Shared across
    // clones via Arc so web_fetch and web_get (and the REPL) hit the same cache.
    pub(crate) cache: Arc<Mutex<LruCache<String, FetchResult>>>,
    // Base URL of a SearxNG instance (--searxng / SEARXNG_URL). When set, web_search goes
    // through it instead of DuckDuckGo.
    pub(crate) searxng: Option<String>,
    // Files this session has seen, and what was true of them when it saw them. `edit` and
    // `write_file` consult this so a model cannot rewrite a file it has never looked at, or
    // one that changed underneath it since it looked. Shared across clones via Arc.
    pub(crate) seen_files: Arc<Mutex<std::collections::HashMap<PathBuf, Seen>>>,
    // The current task list, as last written by todo_write. Shared across clones so the
    // REPL and the tool see the same list.
    pub(crate) todos: Arc<Mutex<Vec<TodoItem>>>,
    // Contents of every file about to be changed, newest last, so a bad edit can be put
    // back without reaching for git.
    pub(crate) checkpoints: Arc<Mutex<Vec<FileCheckpoint>>>,
    // Long-running commands started with run_background, killed when the session ends.
    pub(crate) background: Arc<Mutex<BackgroundJobs>>,
    // True in the REPL, false in MCP server mode. Gates progress output: run_command echoes
    // a long command's output as it arrives, which is noise for an MCP host.
    pub(crate) interactive: bool,
    // The library the `library` tool reads, or None when that tool is off — no LLM to run
    // its librarian, or no articles to read. `run_command` checks it too: a failed build is
    // only pointed at the library when there is one to point at.
    pub(crate) library_root: Option<PathBuf>,
    // Fingerprints of the error sets a failed build has already been told to look up.
    // Once per distinct set of errors: a model rebuilding against the same errors has
    // either asked already or decided not to, and the same note under every rebuild is how
    // a model learns to skip notes.
    pub(crate) library_nudged: Arc<Mutex<std::collections::HashSet<u64>>>,
    // Timers the model has set on itself (`crate::timers`). Here rather than on the agent
    // because the tool that arms them runs on this struct; the agent and the frontends
    // reach them through `take_due_timers` and `next_timer_wake`.
    pub(crate) timers: Arc<Mutex<crate::timers::Timers>>,
}

// A file's contents immediately before a tool changed it.
//
// Bytes rather than a String: write_file can legitimately overwrite a binary file, and a
// checkpoint that could not represent it would be worse than useless — it would look like a
// working undo right up until it destroyed the file.

pub(crate) struct FileCheckpoint {
    pub(crate) path: PathBuf,
    // None means the file did not exist, so undoing means deleting it again.
    pub(crate) previous: Option<Vec<u8>>,
    pub(crate) description: String,
}

// How many changes can be undone. Bounded because the contents are held in memory.

pub(crate) const MAX_CHECKPOINTS: usize = 50;

// What was true of a file the last time this session saw it.
//
// The modification time is the cheap check and answers almost every call. The fingerprint is
// what settles a disagreement, and it exists because the timestamp turned out to be a liar:
// on Windows a stat taken shortly after a write can come back with the directory entry's
// older time rather than the file's real one (`fs::metadata` falls back to the directory
// entry when it cannot open a handle — a scanner holding the file for a moment is enough).
// A tool's own write then looked like somebody else's edit, and the model was told to re-read
// a file it had itself just written, over and over.

#[derive(Clone, Copy)]
pub(crate) struct Seen {
    // None when the stat failed; the fingerprint then carries the check on its own.
    pub(crate) modified: Option<std::time::SystemTime>,
    // Set only when we know the file's whole contents — that is, when we wrote them. `read`
    // sees a window of a file, never necessarily all of it, so it has no fingerprint to give.
    pub(crate) hash: Option<u64>,
}

// Fingerprint of a file's exact bytes.
//
// The standard hasher is enough: fingerprints are only ever compared with others taken in the
// same process, so its lack of a stability guarantee across Rust versions does not matter, and
// nothing here is persisted.

pub(crate) fn content_hash(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

// Key for `seen_files`. Canonicalized so "./src/a.rs", "src/a.rs" and an absolute path all
// land on the same entry; a path that cannot be canonicalized (a file being created) falls
// back to its literal form.

impl MyMcpServer {
    pub(crate) fn new(
        llm_client: LMStudioClient,
        expose_web_fetch: bool,
        searxng: Option<String>,
        interactive: bool,
    ) -> Self {
        // One router per impl block, merged here. This is what allows the tools to live in
        // separate files at all.
        // research before the web tools it wraps: a small model reads the list in order,
        // and the tool it should reach for first should not be the last one it sees.
        let mut tool_router = Self::basics_router()
            + Self::files_router()
            + Self::memory_router()
            + Self::skills_router()
            + Self::research_router()
            + Self::library_router()
            + Self::web_router()
            + Self::shell_router()
            + Self::crates_router()
            + Self::timers_router();
        // web_get (raw page text) is always available. web_fetch summarizes a page through
        // the LLM and research drives a whole loop through it, so both are exposed only when
        // we know where the LLM lives.
        if !expose_web_fetch {
            tool_router.remove_route("web_fetch");
            tool_router.remove_route("research");
        }
        // An MCP host has skills of its own, found in its own places and loaded its own way;
        // a second, differently-rooted set arriving through a tool would be two answers to
        // one question.
        if !interactive {
            tool_router.remove_route("skill");
            // A timer fires by putting a message into our own conversation loop, and an
            // MCP host has no loop of ours to put it into.
            tool_router.remove_route("timer");
        }
        // The library needs a model to run its librarian and articles for it to read. Off
        // in tests unless a test sets it, like `Agent::new`'s session path: a suite that read
        // the user's own store would pass or fail by what happens to be in it.
        let library_root = crate::library::user_dir()
            .filter(|_| expose_web_fetch && !cfg!(test))
            .filter(|root| !crate::library::Library::load(root).is_empty());
        if library_root.is_none() {
            tool_router.remove_route("library");
        }
        Self {
            tool_router,
            llm_client,
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(10).unwrap()))),
            searxng,
            seen_files: Arc::new(Mutex::new(std::collections::HashMap::new())),
            todos: Arc::new(Mutex::new(Vec::new())),
            checkpoints: Arc::new(Mutex::new(Vec::new())),
            background: Arc::new(Mutex::new(BackgroundJobs::default())),
            interactive,
            library_root,
            library_nudged: Arc::new(Mutex::new(std::collections::HashSet::new())),
            timers: Arc::new(Mutex::new(crate::timers::Timers::default())),
        }
    }


    // Kill every background job. Returns how many were still running.
    //
    // Called on every way out of the REPL. It does not cover haico being killed or panicking
    // — nothing short of a Windows Job Object would — so `/bg` exists to make survivors
    // visible when that happens.
    pub(crate) async fn stop_all_background(&self) -> usize {
        // Collected under the lock, killed outside it: killing awaits, and a std Mutex must
        // never be held across an await point.
        let running: Vec<(u32, Option<u32>)> = {
            let registry = self.background.lock().unwrap();
            registry
                .jobs
                .iter()
                .filter(|(_, job)| job.exit.lock().unwrap().is_none() && !job.stopped)
                .map(|(id, job)| (*id, job.pid))
                .collect()
        };

        for (id, pid) in &running {
            if let Some(pid) = pid {
                crate::tools::kill_tree_by_pid(*pid).await;
            }
            if let Some(job) = self.background.lock().unwrap().jobs.get_mut(id) {
                job.stopped = true;
            }
        }

        running.len()
    }

    pub(crate) fn checkpoint(&self, path: &std::path::Path, description: String) {
        let previous = fs::read(path).ok();

        let mut checkpoints = self.checkpoints.lock().unwrap();
        checkpoints.push(FileCheckpoint {
            path: path.to_path_buf(),
            previous,
            description,
        });
        // Oldest first, so trimming from the front keeps the most recent undoable.
        if checkpoints.len() > MAX_CHECKPOINTS {
            checkpoints.remove(0);
        }
    }

    // Put back the most recent change. Returns what was undone.

    pub(crate) fn undo_last(&self) -> std::result::Result<String, String> {
        let Some(checkpoint) = self.checkpoints.lock().unwrap().pop() else {
            return Err("Nothing to undo.".to_string());
        };

        match &checkpoint.previous {
            Some(contents) => {
                fs::write(&checkpoint.path, contents)
                    .map_err(|e| format!("Could not restore {}: {}", checkpoint.path.display(), e))?;
                // The file now holds what it did when it was last read, so a following edit
                // must not be turned away as stale.
                self.mark_written(&checkpoint.path, contents);
            }
            None => {
                // The change created this file, so undoing it means removing it. Already
                // gone is the desired end state, not an error.
                if checkpoint.path.exists() {
                    fs::remove_file(&checkpoint.path).map_err(|e| {
                        format!("Could not remove {}: {}", checkpoint.path.display(), e)
                    })?;
                }
                // Nothing there to have seen any more. Writing the path again is a creation,
                // which needs no prior read, so an entry describing a file that no longer
                // exists could only ever be wrong.
                self.seen_files.lock().unwrap().remove(&file_key(&checkpoint.path));
            }
        }

        Ok(checkpoint.description)
    }

    // Record that a file has been read, at the modification time it had when the read began.
    //
    // The time is the caller's rather than taken here: `read` streams the file, and the mtime
    // that matters is the one from before it started, so a file changed mid-read counts as
    // stale instead of as seen.

    pub(crate) fn mark_read(&self, path: &std::path::Path, modified: std::time::SystemTime) {
        let key = file_key(path);
        let mut seen = self.seen_files.lock().unwrap();
        // Reading a file cannot make us know less about it: if it has not moved on since a
        // tool wrote it, that write's fingerprint still describes what is there.
        let hash = seen
            .get(&key)
            .filter(|previous| previous.modified == Some(modified))
            .and_then(|previous| previous.hash);
        seen.insert(
            key,
            Seen {
                modified: Some(modified),
                hash,
            },
        );
    }

    // Record that a tool just wrote these exact bytes.
    //
    // The fingerprint comes from the bytes we handed to the filesystem, not from reading the
    // file back: it is the one thing here we know for certain, and it stays true however the
    // recorded timestamp turns out.

    pub(crate) fn mark_written(&self, path: &std::path::Path, bytes: &[u8]) {
        self.seen_files.lock().unwrap().insert(
            file_key(path),
            Seen {
                modified: modified_time(path),
                hash: Some(content_hash(bytes)),
            },
        );
    }

    // Gate a modification to an existing file on having read it first, and on it not having
    // changed since. Returns the message to hand back to the model when the edit must not
    // proceed.
    //
    // The tool descriptions have always said "Read the file first"; nothing enforced it, so
    // a model could rewrite a file from an assumption about its contents. The staleness half
    // catches the other case: the user edited the file in their editor after the model read
    // it, and a string replacement would now land somewhere unintended.

    pub(crate) fn check_may_modify(&self, path: &std::path::Path) -> Option<String> {
        // A file that does not exist yet is being created; there is nothing to have read.
        if !path.exists() {
            return None;
        }

        let key = file_key(path);
        let seen = self.seen_files.lock().unwrap().get(&key).copied();

        let Some(seen) = seen else {
            return Some(format!(
                "Refusing to modify {} because it has not been read in this session. \
                 Call read on it first, so the change is based on its actual contents.",
                path.display()
            ));
        };

        let now = modified_time(path);
        if now.is_some() && now == seen.modified {
            return None;
        }

        // The timestamp disagrees, which is not the same thing as the file having changed —
        // see `Seen`. When we know which bytes we left there, ask the file itself before
        // calling it somebody else's edit.
        if let Some(hash) = seen.hash {
            if fs::read(path).is_ok_and(|bytes| content_hash(&bytes) == hash) {
                // Same bytes, different timestamp: adopt the new one so the next call takes
                // the cheap path instead of re-reading the file every time.
                self.seen_files
                    .lock()
                    .unwrap()
                    .insert(key, Seen { modified: now, hash: Some(hash) });
                return None;
            }
        }

        Some(format!(
            "Refusing to modify {} because it changed on disk after you read it. \
             Call read on it again, then redo the change against the current contents.",
            path.display()
        ))
    }


    pub(crate) async fn execute_tool(&self, name: &str, arguments: Value) -> Result<ToolOutput> {
        let result = match name {
            "calculator" => {
                let params: CalculatorParams = serde_json::from_value(arguments)?;
                self.calculator(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "get_date" => {
                self.get_date().await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "get_time" => {
                self.get_time().await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "read" => {
                let params: ReadParams = serde_json::from_value(arguments)?;
                self.read(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "write_file" => {
                let params: WriteParams = serde_json::from_value(arguments)?;
                self.write_file(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "edit" => {
                let params: EditParams = serde_json::from_value(arguments)?;
                self.edit(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "glob_search" => {
                let params: GlobParams = serde_json::from_value(arguments)?;
                self.glob_search(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "grep" => {
                let params: GrepParams = serde_json::from_value(arguments)?;
                self.grep(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "web_fetch" => {
                let params: WebFetchParams = serde_json::from_value(arguments)?;
                self.web_fetch(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "web_get" => {
                let params: WebGetParams = serde_json::from_value(arguments)?;
                self.web_get(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "web_search" => {
                let params: WebSearchParams = serde_json::from_value(arguments)?;
                self.web_search(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "research" => {
                let params: ResearchParams = serde_json::from_value(arguments)?;
                self.research(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "library" => {
                let params: LibraryParams = serde_json::from_value(arguments)?;
                self.library(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "remember" => {
                let params: RememberParams = serde_json::from_value(arguments)?;
                self.remember(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "recall" => {
                let params: RecallParams = serde_json::from_value(arguments)?;
                self.recall(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "skill" => {
                let params: SkillParams = serde_json::from_value(arguments)?;
                self.skill(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "ls" => {
                let params: LSParams = serde_json::from_value(arguments)?;
                self.ls(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "run_background" => {
                let params: RunBackgroundParams = serde_json::from_value(arguments)?;
                self.run_background(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "background_output" => {
                let params: BackgroundOutputParams = serde_json::from_value(arguments)?;
                self.background_output(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "background_stop" => {
                let params: BackgroundStopParams = serde_json::from_value(arguments)?;
                self.background_stop(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "run_command" => {
                let params: RunCommandParams = serde_json::from_value(arguments)?;
                self.run_command(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "multi_edit" => {
                let params: MultiEditParams = serde_json::from_value(arguments)?;
                self.multi_edit(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "todo_write" => {
                let params: TodoWriteParams = serde_json::from_value(arguments)?;
                self.todo_write(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "ask_user" => {
                let params: AskUserParams = serde_json::from_value(arguments)?;
                self.ask_user(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "start_plan" => {
                self.start_plan().await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "present_plan" => {
                let params: PresentPlanParams = serde_json::from_value(arguments)?;
                self.present_plan(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "crate_info" => {
                let params: CrateInfoParams = serde_json::from_value(arguments)?;
                self.crate_info(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "crate_search" => {
                let params: CrateSearchParams = serde_json::from_value(arguments)?;
                self.crate_search(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "timer" => {
                let params: TimerParams = serde_json::from_value(arguments)?;
                self.timer(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "crate_api" => {
                let params: CrateApiParams = serde_json::from_value(arguments)?;
                self.crate_api(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            _ => return Err(anyhow::anyhow!("Unknown tool: {}", name)),
        };

        // Split the result by kind rather than flattening it to text. Flattening
        // is what this did for as long as every tool returned prose, and it is
        // why `read` on an image had nowhere to put the image.
        let text = result
            .content
            .iter()
            .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
            .collect::<Vec<_>>()
            .join("\n");

        let images = result
            .content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::Image(image) => {
                    Some((image.mime_type.clone(), image.data.clone()))
                }
                _ => None,
            })
            .collect();

        Ok(ToolOutput { text, images })
    }
}

/// What a tool produced, split by what the conversation can do with each half.
///
/// The split is forced by the wire format, not chosen. A `role: "tool"` message
/// may carry text parts and nothing else — servers reject an `image_url` there
/// outright — so a picture cannot travel back inside the reply to the call that
/// produced it. It comes back beside the text instead, and the turn puts it
/// where an image is allowed: a user message, after the round's tool replies.
///
/// MCP server mode never builds one of these. There the `CallToolResult` goes to
/// the host whole, image blocks and all, because MCP has a place for them.
#[derive(Debug, Default)]
pub(crate) struct ToolOutput {
    pub(crate) text: String,
    /// `(mime type, base64 data)`, in the order the tool produced them.
    pub(crate) images: Vec<(String, String)>,
}

// MCP server handler: wires the existing tool_router to handle list_tools/call_tool.
// The macro auto-generates a tools-enabled get_info() (server identity comes from Cargo.toml).

#[tool_handler(router = self.tool_router, name = "haico")]
impl ServerHandler for MyMcpServer {}

// Convert MCP tools to OpenAI tool format

pub(crate) fn mcp_tools_to_openai_format(server: &MyMcpServer) -> Vec<Value> {
    let tools = server.tool_router.list_all();
    tools
        .iter()
        .map(|tool| {
            let mut parameters = Value::Object((*tool.input_schema).clone());
            simplify_schema(&mut parameters);
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description.as_deref().unwrap_or(""),
                    "parameters": parameters
                }
            })
        })
        .collect()
}

/// The schema as the model reads it: schemars writes for validators, and three of its habits
/// cost a small model something and tell it nothing.
///
/// - `"type": ["integer", "null"]` for every `Option` field. Optional is already said by the
///   field's absence from `required`; the union only adds "you may send an explicit null",
///   which haico never needs (an omitted argument and a null one parse the same). Small models
///   copy it back as the string `"null"`, and the mixed shapes (`path` is `["string", "null"]`
///   in `grep` but `"string"` in `ls`) are also what made a dataset built from these schemas
///   unreadable by Arrow, which wants one type per column.
/// - `"format": "uint64"` and friends: Rust type names, not JSON Schema formats. `minimum: 0`,
///   which schemars writes alongside, already says unsigned.
/// - `"$schema"`: a dialect URL, 50 characters of nothing in every tool.
///
/// Only the copy sent to the model is simplified. MCP server mode hands the host rmcp's own
/// schema, since validating arguments is the host's business.
pub(crate) fn simplify_schema(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            map.remove("$schema");
            if map.get("format").and_then(Value::as_str).is_some_and(is_rust_number_format) {
                map.remove("format");
            }
            if let Some(Value::Array(types)) = map.get("type") {
                let kept: Vec<Value> = types.iter().filter(|t| t.as_str() != Some("null")).cloned().collect();
                // A field that is *only* null has nothing else to say; leave it alone.
                if !kept.is_empty() && kept.len() < types.len() {
                    let simple = if kept.len() == 1 { kept[0].clone() } else { Value::Array(kept) };
                    map.insert("type".to_string(), simple);
                }
            }
            map.values_mut().for_each(simplify_schema);
        }
        Value::Array(items) => items.iter_mut().for_each(simplify_schema),
        _ => {}
    }
}

fn is_rust_number_format(format: &str) -> bool {
    matches!(
        format,
        "int" | "int8" | "int16" | "int32" | "int64" | "uint" | "uint8" | "uint16" | "uint32" | "uint64" | "float" | "double"
    )
}

// Pull the arguments out of a tool call, tolerating the shapes local models actually emit.
//
// The OpenAI wire format says `function.arguments` is a JSON *string*, but plenty of
// local servers put a JSON object there instead, and some omit it entirely for a tool that
// takes no parameters. Treating any of those as a hard failure used to drop the tool call
// on the floor while its assistant message stayed in the history — leaving a tool_call_id
// no `role:"tool"` message ever answered, which the next request is rejected for.

pub(crate) fn parse_tool_arguments(function: &Value) -> std::result::Result<Value, String> {
    match function.get("arguments") {
        // No arguments at all: fine for a no-parameter tool like get_date.
        None | Some(Value::Null) => Ok(json!({})),
        Some(Value::String(raw)) => {
            let raw = raw.trim();
            if raw.is_empty() {
                return Ok(json!({}));
            }
            serde_json::from_str(raw).map_err(|e| {
                format!(
                    "arguments were not valid JSON ({}): {}",
                    e,
                    truncate_at_boundary(raw, 500)
                )
            })
        }
        Some(object @ Value::Object(_)) => Ok(object.clone()),
        Some(other) => Err(format!(
            "arguments had unexpected type: {}",
            truncate_at_boundary(&other.to_string(), 200)
        )),
    }
}

// Cut off a copy of a call's other arguments that a model wrote *inside* one of them, and
// name each argument that was cut.
//
// Observed on a p2p_auth run (qwen3.8-35b-a3b-distill): 14 of its 29 `edit` calls had a
// `new_string` ending in `", "old_string": "<the real old_string, escaped>"}` — the rest of the
// argument object written a second time as text, before the real `old_string` followed. The
// JSON was valid and `old_string` matched, so every one of them was applied, and the file got
// a literal `", "old_string": "...\n..."`: the build failed on "unknown start of token: \" and
// the model took it for the edit tool corrupting its files.
//
// Repaired rather than refused, because the evidence is exact: the leaked tail must decode to
// precisely the value the other argument really has. Code that merely contains such a string
// would have to end with an escaped copy of the call's own `old_string` to be touched, and
// what is left after the cut is what the model meant to write. The caller says it was done.
// Top-level string arguments only; that is where it has been seen.
pub(crate) fn repair_leaked_arguments(arguments: &mut Value) -> Vec<String> {
    let Some(object) = arguments.as_object_mut() else {
        return Vec::new();
    };
    let mut cuts: Vec<(String, usize)> = Vec::new();
    for (key, value) in object.iter() {
        let Some(text) = value.as_str() else { continue };
        for (other, other_value) in object.iter() {
            let Some(other_text) = other_value.as_str() else { continue };
            if other == key || other_text.is_empty() {
                continue;
            }
            let leak = regex::Regex::new(&format!(r#""\s*,\s*"{}"\s*:\s*""#, regex::escape(other))).unwrap();
            let found = leak.find_iter(text).find(|m| leaked_copy(&text[m.end()..]).as_deref() == Some(other_text));
            if let Some(found) = found {
                cuts.push((key.clone(), found.start()));
                break;
            }
        }
    }
    for (key, at) in &cuts {
        if let Some(Value::String(text)) = object.get_mut(key) {
            text.truncate(*at);
        }
    }
    let mut repaired: Vec<String> = cuts.into_iter().map(|(key, _)| key).collect();
    repaired.sort();
    repaired
}

// The value a leaked tail spells, if it is one: an escaped string body, its closing quote,
// and at most the object's closing brace after it.
fn leaked_copy(tail: &str) -> Option<String> {
    let tail = tail.trim_end();
    let tail = tail.strip_suffix('}').unwrap_or(tail).trim_end();
    let body = tail.strip_suffix('"')?;
    serde_json::from_str::<String>(&format!("\"{}\"", body)).ok()
}

// Fetch available models from LLM server

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

    #[test]
    fn simplify_schema_drops_nullability_rust_formats_and_the_dialect() {
        let mut schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "limit": {"type": ["integer", "null"], "format": "uint", "minimum": 0},
                "ratio": {"type": "number", "format": "double"},
                "url": {"type": "string", "format": "uri"},
                // A field named like a keyword is a field, not the keyword.
                "format": {"type": ["string", "null"]},
                "either": {"type": ["string", "integer", "null"]},
                "nothing": {"type": ["null"]},
                "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
            },
            "required": ["url"],
            "$defs": {"Item": {"type": "object", "properties": {"n": {"type": ["integer", "null"], "format": "int64"}}}}
        });
        simplify_schema(&mut schema);
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 0},
                    "ratio": {"type": "number"},
                    "url": {"type": "string", "format": "uri"},
                    "format": {"type": "string"},
                    "either": {"type": ["string", "integer"]},
                    "nothing": {"type": ["null"]},
                    "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
                },
                "required": ["url"],
                "$defs": {"Item": {"type": "object", "properties": {"n": {"type": "integer"}}}}
            })
        );
    }

    #[test]
    fn the_model_is_sent_simple_schemas_with_optionality_intact() {
        let server = test_server();
        let tools = mcp_tools_to_openai_format(&server);
        let text = serde_json::to_string(&tools).unwrap();
        assert!(!text.contains("\"null\""), "a null type reached the model");
        assert!(!text.contains("$schema") && !text.contains("\"uint"), "schemars noise reached the model");
        // Optional is still said by `required`, which the simplification doesn't touch.
        let read = tools.iter().find(|t| t["function"]["name"] == "read").unwrap();
        assert_eq!(read["function"]["parameters"]["required"], json!(["file_path"]));
        assert_eq!(read["function"]["parameters"]["properties"]["limit"]["type"], "integer");
        // MCP hosts still get rmcp's own schema.
        let raw = server.tool_router.list_all().into_iter().find(|t| t.name == "read").unwrap();
        assert!(serde_json::to_string(&raw.input_schema).unwrap().contains("\"null\""));
    }

    #[tokio::test]
    async fn undo_reverts_edits_creations_and_overwrites() {
        let server = test_server();

        // Nothing done yet.
        assert!(server.undo_last().is_err());

        let path = scratch("undo.rs", "original\n");
        read_file(&server, &path).await;

        try_edit(&server, &path, "original", "changed", false).await;
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "changed\n");

        let what = server.undo_last().unwrap();
        assert!(what.contains("edited"), "got: {}", what);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "original\n");

        // Undo restores the file to what was last read, so a following edit is not turned
        // away as stale — otherwise undo would leave the session stuck.
        let (ok, out) = try_edit(&server, &path, "original", "again", false).await;
        assert!(ok, "got: {}", out);

        // Undoing a creation removes the file rather than leaving an empty stub.
        let dir = path.parent().unwrap().to_path_buf();
        let created = dir.join("undo-created.rs");
        let _ = std::fs::remove_file(&created);
        server
            .write_file(rmcp::handler::server::wrapper::Parameters(
                crate::WriteParams {
                    file_path: created.display().to_string(),
                    content: "fn main() {}\n".to_string(),
                },
            ))
            .await
            .unwrap();
        assert!(created.exists());
        let what = server.undo_last().unwrap();
        assert!(what.contains("created"), "got: {}", what);
        assert!(!created.exists(), "undoing a creation should remove the file");

        // Undo works back through the stack, most recent first.
        server.undo_last().unwrap(); // the "again" edit
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "original\n");

        let _ = std::fs::remove_file(&path);
    }


    #[tokio::test]
    async fn checkpoints_preserve_bytes_not_just_text() {
        let server = test_server();

        // A checkpoint that could only hold valid UTF-8 would look like a working undo right
        // up until it destroyed the file.
        let dir = scratch("anchor2", "").parent().unwrap().to_path_buf();
        let binary = dir.join("undo-binary.bin");
        let original: Vec<u8> = vec![0x00, 0xFF, 0xFE, 0x80, 0x0A, 0x1B];
        std::fs::write(&binary, &original).unwrap();

        server.checkpoint(&binary, "overwrote binary".to_string());
        std::fs::write(&binary, "clobbered").unwrap();

        server.undo_last().unwrap();
        assert_eq!(std::fs::read(&binary).unwrap(), original);

        let _ = std::fs::remove_file(&binary);
    }


    #[test]
    fn accepts_the_tool_argument_shapes_local_models_emit() {
        // The wire format: a JSON string.
        assert_eq!(
            parse_tool_arguments(&json!({"arguments": "{\"a\":1}"})).unwrap(),
            json!({"a": 1})
        );
        // What plenty of local servers send instead: a bare object.
        assert_eq!(
            parse_tool_arguments(&json!({"arguments": {"a": 1}})).unwrap(),
            json!({"a": 1})
        );
        // A no-parameter tool, omitted or nulled or blank.
        assert_eq!(parse_tool_arguments(&json!({})).unwrap(), json!({}));
        assert_eq!(
            parse_tool_arguments(&json!({"arguments": null})).unwrap(),
            json!({})
        );
        assert_eq!(
            parse_tool_arguments(&json!({"arguments": "  "})).unwrap(),
            json!({})
        );
        // Genuinely broken input is reported, not silently dropped.
        assert!(parse_tool_arguments(&json!({"arguments": "{oops"})).is_err());
        assert!(parse_tool_arguments(&json!({"arguments": 42})).is_err());
    }

    #[test]
    fn a_copy_of_the_other_arguments_inside_one_is_cut_off() {
        // The shape from the p2p_auth trace, as the server delivered it.
        let raw = r#"{"file_path":"src/lib.rs","new_string":"        let (out_tx, out_rx) = channel(64);\n        insert(out_tx);\", \"old_string\": \"        let (out_tx, mut out_rx) = channel(64);\\n        insert(out_tx);\"}","old_string":"        let (out_tx, mut out_rx) = channel(64);\n        insert(out_tx);"}"#;
        let mut arguments = parse_tool_arguments(&json!({ "arguments": raw })).unwrap();
        assert_eq!(repair_leaked_arguments(&mut arguments), ["new_string"]);
        assert_eq!(arguments["new_string"], "        let (out_tx, out_rx) = channel(64);\n        insert(out_tx);");
        assert_eq!(arguments["old_string"], "        let (out_tx, mut out_rx) = channel(64);\n        insert(out_tx);");

        // Without the closing brace, and with spaces the model may or may not write.
        let mut arguments = json!({"new_string": "b\" ,  \"old_string\" : \"a\\nz\"", "old_string": "a\nz"});
        assert_eq!(repair_leaked_arguments(&mut arguments), ["new_string"]);
        assert_eq!(arguments["new_string"], "b");

        // Code that contains such text is left alone unless the tail is exactly the other
        // argument's value: here it spells something else.
        let json_in_code = r#"let body = "{\"new_string\": \"x\", \"old_string\": \"y\"}";"#;
        let mut arguments = json!({"new_string": json_in_code, "old_string": "let body = 1;"});
        assert!(repair_leaked_arguments(&mut arguments).is_empty());
        assert_eq!(arguments["new_string"], json_in_code);
        let spelled_out = r#"x", "old_string": "y"}"#;
        let mut arguments = json!({"new_string": spelled_out, "old_string": "z"});
        assert!(repair_leaked_arguments(&mut arguments).is_empty());
        assert_eq!(arguments["new_string"], spelled_out);

        // Non-object and non-string arguments are not touched.
        assert!(repair_leaked_arguments(&mut json!([1, 2])).is_empty());
        let mut arguments = json!({"command": "ls", "timeout": 5});
        assert!(repair_leaked_arguments(&mut arguments).is_empty());
    }

}
