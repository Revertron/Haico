use super::MyMcpServer;
use crate::util::{file_key, modified_time, truncate_at_boundary};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::io::{self, BufRead, BufReader};

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct ReadParams {
    #[schemars(description = "Absolute path to the file to read")]
    pub(crate) file_path: String,
    #[schemars(description = "Line number to start reading from (default: 0)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) offset: Option<usize>,
    #[schemars(description = "Maximum number of lines to read (default: 2000)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) limit: Option<usize>,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct WriteParams {
    #[schemars(description = "Absolute path where the file should be written")]
    pub(crate) file_path: String,
    #[schemars(description = "The complete content to write to the file")]
    pub(crate) content: String,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct EditParams {
    #[schemars(description = "Absolute path to the file to modify")]
    pub(crate) file_path: String,
    #[schemars(description = "The exact text to search for and replace")]
    pub(crate) old_string: String,
    #[schemars(description = "The text to replace it with (must be different from old_string)")]
    pub(crate) new_string: String,
    #[schemars(description = "Replace all occurrences of old_string (default false)")]
    #[serde(default)]
    pub(crate) replace_all: bool,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct GlobParams {
    #[schemars(description = "Glob pattern to match files (e.g., '**/*.rs', '*.txt')")]
    pub(crate) pattern: String,
    #[schemars(description = "Base directory to search in (default: current directory)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) path: Option<String>,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct GrepParams {
    #[schemars(description = "Regular expression pattern to search for")]
    pub(crate) pattern: String,
    #[schemars(description = "File or directory to search in (default: current directory)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) path: Option<String>,
    #[schemars(description = "Glob pattern to filter files (e.g., '*.rs', '*.{ts,tsx}')")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) glob: Option<String>,
    #[schemars(description = "Output format: 'content' (matching lines), 'files_with_matches' (file paths, default), or 'count' (match counts)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_mode: Option<String>,
    #[schemars(description = "Perform case-insensitive search (default: false)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) case_insensitive: Option<bool>,
    #[schemars(description = "Limit output to first N results")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) head_limit: Option<usize>,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct LSParams {
    #[schemars(description = "Path to the directory to list")]
    pub(crate) path: String,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct EditOperation {
    #[schemars(description = "The exact text to search for and replace")]
    pub(crate) old_string: String,
    #[schemars(description = "The text to replace it with")]
    pub(crate) new_string: String,
    #[schemars(description = "Replace all occurrences of old_string (default false)")]
    #[serde(default)]
    pub(crate) replace_all: bool,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct MultiEditParams {
    #[schemars(description = "Absolute path to the file to modify")]
    pub(crate) file_path: String,
    #[schemars(
        description = "The replacements to make, applied in order. Each one sees the result of the previous, so later edits must match the text as it will be by then."
    )]
    pub(crate) edits: Vec<EditOperation>,
}


pub(crate) fn build_glob_matcher(pattern: &str) -> std::result::Result<globset::GlobMatcher, String> {
    globset::Glob::new(pattern)
        .map(|g| g.compile_matcher())
        .map_err(|e| format!("Invalid glob pattern '{}': {}", pattern, e))
}

// Match a path against a glob, testing the bare file name too.
//
// A `*` never crosses a path separator, so a plain "*.rs" would match nothing when tested
// against "src/main.rs". Models overwhelmingly write the bare-extension form, so fall back
// to matching the file name — the same accommodation ripgrep's --glob makes.

pub(crate) fn glob_matches(matcher: &globset::GlobMatcher, path: &std::path::Path) -> bool {
    matcher.is_match(path)
        || path
            .file_name()
            .map_or(false, |name| matcher.is_match(std::path::Path::new(name)))
}

// Walk a tree the way a code search is expected to: honoring .gitignore (plus global and
// per-repo excludes), never descending into .git, and never following symlinks.
//
// Without this both grep and glob_search walk target/, node_modules/ and .git/. In a Rust
// project that is tens of thousands of build artifacts, which makes every search slow and
// buries the real matches — the single biggest thing that made these tools unusable.
//
// Dotfiles are deliberately still visible (.github/, .gitignore, .cargo/config.toml are
// all things a coding agent legitimately needs); only .git itself is cut out.

pub(crate) fn walk_source_files(
    root: &str,
    glob: Option<&str>,
    cancel: &AtomicBool,
) -> std::result::Result<Vec<PathBuf>, String> {
    let matcher = match glob {
        Some(pattern) => Some(build_glob_matcher(pattern)?),
        None => None,
    };

    let walker = ignore::WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true)
        .follow_links(false)
        .filter_entry(|entry| entry.file_name() != ".git")
        .build();

    let mut files = Vec::new();
    for entry in walker {
        // Over a tree with no .gitignore to thin it out the walk is the slow half all by
        // itself — ~/.cargo/registry is ninety thousand entries — so giving up has to be
        // possible here, and not only once the whole list has been collected.
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        let entry = match entry {
            Ok(entry) => entry,
            // An unreadable directory should not abort the whole search.
            Err(_) => continue,
        };

        if !entry.file_type().map_or(false, |t| t.is_file()) {
            continue;
        }

        if let Some(matcher) = &matcher {
            // Match against the path relative to the walk root. Walking "." yields
            // "./src/main.rs", and that leading "./" makes any pattern with a directory
            // component ("src/**/*.rs") match nothing — while a bare "*.rs" still works via
            // the file-name fallback, which is exactly how this stayed hidden.
            let path = entry.path();
            let relative = path.strip_prefix(root).unwrap_or(path);
            if !glob_matches(matcher, relative) && !glob_matches(matcher, path) {
                continue;
            }
        }

        files.push(entry.into_path());
    }

    Ok(files)
}

// Sets a search's cancel flag when the future that owns it goes away.
//
// Dropping the handle of a `spawn_blocking` task does not stop it — there is no way to
// abort a thread mid-call — so without this an interrupted grep would go on reading files
// for minutes after the user gave up on it, with nothing left to receive the answer. The
// flag is the only way back: the blocking half checks it and returns early.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

// The blocking half of `glob_search`: the walk, and the mtime of every match behind the sort.
fn glob_search_blocking(
    params: &GlobParams,
    cancel: &AtomicBool,
) -> std::result::Result<String, String> {
    let base_path = params.path.as_deref().unwrap_or(".");

    let mut matches = walk_source_files(base_path, Some(&params.pattern), cancel)?;

    matches.sort_by_cached_key(|path| {
        std::cmp::Reverse(fs::metadata(path).and_then(|m| m.modified()).ok())
    });

    // A broad pattern in a big tree can match thousands of files; dumping all of them
    // costs more context than the answer is worth.
    const MAX_RESULTS: usize = 500;
    let total = matches.len();
    let shown = total.min(MAX_RESULTS);

    Ok(if matches.is_empty() {
        format!("No files found matching pattern: {}", params.pattern)
    } else {
        let paths: Vec<String> = matches[..shown]
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        let mut output = format!("Found {} file(s):\n{}", total, paths.join("\n"));
        if total > shown {
            output.push_str(&format!(
                "\n\n[showing the {} most recently modified of {}; narrow the pattern to see more]",
                shown, total
            ));
        }
        output
    })
}

// The blocking half of `grep`: the walk, and then reading every file it turned up.
fn grep_blocking(
    params: &GrepParams,
    regex: &regex::Regex,
    cancel: &AtomicBool,
) -> std::result::Result<String, String> {
    let output_mode = params.output_mode.as_deref().unwrap_or("files_with_matches");
    let head_limit = params.head_limit;

    let search_path = params.path.as_deref().unwrap_or(".");
    let files = walk_source_files(search_path, params.glob.as_deref(), cancel)?;

    let mut results = Vec::new();
    let mut count = 0;
    let mut truncated = false;

    'files: for path in files {
        // Per file rather than per line: a file is the unit of work here, so the most a
        // late check costs is one more read, not the rest of the tree.
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        if let Ok(metadata) = fs::metadata(&path) {
            if metadata.len() > 10_000_000 {
                continue;
            }
        }

        // read_to_string fails on invalid UTF-8, which is what skips binaries.
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };

        let mut file_matches = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            if regex.is_match(line) {
                file_matches.push((line_num + 1, line));
            }
        }

        if file_matches.is_empty() {
            continue;
        }

        let path_str = path.display().to_string();

        match output_mode {
            "content" => {
                for (line_num, line) in file_matches {
                    // Cap the line itself: one minified bundle line would otherwise
                    // blow out the whole result.
                    results.push(format!(
                        "{}:{}:{}",
                        path_str,
                        line_num,
                        truncate_at_boundary(line, 2000)
                    ));
                    count += 1;
                    if head_limit.is_some_and(|limit| count >= limit) {
                        truncated = true;
                        break 'files;
                    }
                }
            }
            "count" => {
                results.push(format!("{}:{}", path_str, file_matches.len()));
                count += 1;
            }
            _ => {
                results.push(path_str);
                count += 1;
            }
        }

        if head_limit.is_some_and(|limit| count >= limit) {
            truncated = true;
            break;
        }
    }

    Ok(if results.is_empty() {
        format!("No matches found for pattern: {}", params.pattern)
    } else if truncated {
        format!("{}\n\n[stopped at head_limit]", results.join("\n"))
    } else {
        results.join("\n")
    })
}

// Drain one of a child process's pipes to completion, optionally echoing each finished line
// so a long-running command shows progress instead of going silent until it exits.
//
// Bytes rather than tokio's line reader: Windows console tools still emit legacy code pages,
// and a lines() iterator stops at the first invalid UTF-8 byte, silently truncating the
// output. Decoding lossily per line preserves what the previous wait_with_output() returned.
//
// Echo goes to stderr, never stdout: in MCP server mode stdout carries the JSON-RPC stream
// and a stray byte there breaks the protocol.
// Output accumulates in a shared buffer rather than being returned, so the caller can read
// what arrived even when it has to give up on the pump. A killed command's grandchildren can
// keep the pipe open indefinitely (killing `cmd /C ping` leaves `ping` holding it), and
// waiting on the task would then hang long past the timeout that was supposed to bound it.

#[tool_router(router = files_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(description = "Use whenever you need to see what a file actually contains rather than assume — inspecting source before changing it, checking config or data, or quoting exact text. Always Read a file before you Edit or Write it. Reads files from the filesystem. Requires absolute path. Reads up to 2000 lines (use offset/limit for large files). Lines >2000 chars truncated.")]
    pub(crate) async fn read(&self, params: Parameters<ReadParams>) -> Result<CallToolResult, McpError> {
        let file_path = PathBuf::from(&params.0.file_path);

        if !file_path.exists() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "File not found: {}",
                params.0.file_path
            ))]));
        }

        if !file_path.is_file() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Path is not a file: {}",
                params.0.file_path
            ))]));
        }

        let file = fs::File::open(&file_path).map_err(|e| {
            McpError::internal_error(format!("Failed to open file: {}", e), None)
        })?;

        // Captured before reading, recorded only once the read succeeds: if the file changes
        // while we stream it, the next edit is treated as stale rather than as based on
        // current contents, and a read that failed must not count as having seen the file.
        let mtime_at_open = modified_time(&file_path);

        let reader = BufReader::new(file);
        let offset = params.0.offset.unwrap_or(0);
        let limit = params.0.limit.unwrap_or(2000);

        let mut output = String::new();
        let mut line_num = 1;

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(line) => line,
                // A binary file is a normal thing to stumble into, not an internal fault.
                // Saying so plainly stops the model retrying the same read forever.
                Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                    return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                        "{} is not a text file (invalid UTF-8 near line {}). The read, edit \
                         and write tools only handle text.",
                        params.0.file_path, line_num
                    ))]));
                }
                Err(e) => {
                    return Err(McpError::internal_error(
                        format!("Failed to read line: {}", e),
                        None,
                    ));
                }
            };

            if line_num > offset && line_num <= offset + limit {
                let truncated = if line.len() > 2000 {
                    format!("{}...[truncated]", truncate_at_boundary(&line, 2000))
                } else {
                    line
                };
                output.push_str(&format!("{:6}→{}\n", line_num, truncated));
            }

            line_num += 1;

            if line_num > offset + limit {
                break;
            }
        }

        if output.is_empty() {
            output = "File is empty or offset is beyond file length".to_string();
        }

        // The read completed, so the file now counts as seen — at the modification time it
        // had when we opened it, not now.
        if let Some(mtime) = mtime_at_open {
            self.seen_files
                .lock()
                .unwrap()
                .insert(file_key(&file_path), mtime);
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
    }


    #[tool(description = "Use to create a brand-new file or completely replace an existing file's contents; to change only part of a file, prefer Edit instead. Writes a file to the filesystem, overwriting existing files (Read first for existing files). Prefer editing over creating new files. Don't proactively create docs/READMEs. No emojis unless requested.")]
    pub(crate) async fn write_file(&self, params: Parameters<WriteParams>) -> Result<CallToolResult, McpError> {
        let file_path = PathBuf::from(&params.0.file_path);

        // Overwriting an existing file destroys whatever was there; creating a new one does
        // not, so only the former has to have been read.
        if let Some(refusal) = self.check_may_modify(&file_path) {
            return Ok(CallToolResult::error(vec![ContentBlock::text(refusal)]));
        }

        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                McpError::internal_error(format!("Failed to create directories: {}", e), None)
            })?;
        }

        self.checkpoint(
            &file_path,
            format!(
                "{} {}",
                if file_path.exists() { "overwrote" } else { "created" },
                params.0.file_path
            ),
        );

        fs::write(&file_path, &params.0.content).map_err(|e| {
            McpError::internal_error(format!("Failed to write file: {}", e), None)
        })?;

        // The file now matches what the model just produced, so a follow-up edit is working
        // from known contents and must not be rejected as unread or stale.
        self.mark_read(&file_path);

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Successfully wrote {} bytes to {}",
            params.0.content.len(),
            params.0.file_path
        ))]))
    }


    #[tool(description = "Use to change part of an existing file — fixing a line, renaming a symbol, or adjusting a value — without rewriting the whole file. Performs exact string replacement. Must Read file first. Preserve exact indentation of lines. Prefer editing over creating new files. No emojis unless requested. old_string must be unique or use replace_all. Use replace_all for renaming variables.")]
    pub(crate) async fn edit(&self, params: Parameters<EditParams>) -> Result<CallToolResult, McpError> {
        let file_path = PathBuf::from(&params.0.file_path);

        if !file_path.exists() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "File not found: {}",
                params.0.file_path
            ))]));
        }

        if let Some(refusal) = self.check_may_modify(&file_path) {
            return Ok(CallToolResult::error(vec![ContentBlock::text(refusal)]));
        }

        if params.0.old_string == params.0.new_string {
            return Ok(CallToolResult::error(vec![ContentBlock::text(
                "No changes made - old_string and new_string are identical".to_string(),
            )]));
        }

        let content = fs::read_to_string(&file_path).map_err(|e| {
            McpError::internal_error(format!("Failed to read file: {}", e), None)
        })?;

        let matches = content.matches(&params.0.old_string).count();

        if matches == 0 {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "String not found in {}: '{}'",
                params.0.file_path,
                truncate_at_boundary(&params.0.old_string, 500)
            ))]));
        }

        // The description has always promised "old_string must be unique or use
        // replace_all", but nothing checked: a string occurring several times silently
        // edited the first one, which is rarely the one that was meant.
        if matches > 1 && !params.0.replace_all {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "old_string is not unique: it occurs {} times in {}. Include enough \
                 surrounding context to identify the one you mean, or pass \
                 replace_all: true to change all {} of them.",
                matches, params.0.file_path, matches
            ))]));
        }

        let new_content = if params.0.replace_all {
            content.replace(&params.0.old_string, &params.0.new_string)
        } else {
            // Exactly one occurrence, so replacen and find agree.
            content.replacen(&params.0.old_string, &params.0.new_string, 1)
        };

        self.checkpoint(
            &file_path,
            format!(
                "edited {} ({} replacement(s))",
                params.0.file_path, matches
            ),
        );

        fs::write(&file_path, &new_content).map_err(|e| {
            McpError::internal_error(format!("Failed to write file: {}", e), None)
        })?;

        // Keep the recorded mtime current so a follow-up edit is not rejected as stale.
        self.mark_read(&file_path);

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Successfully replaced {} occurrence(s) in {}",
            matches, params.0.file_path
        ))]))
    }


    #[tool(
        description = "Make several replacements in one file in a single call. Use this instead of repeating edit when a change touches a file in more than one place — renaming a symbol across a function, updating a signature and its callers, adjusting several constants together. The edits apply in order and each sees the result of the previous one. Either every edit applies or none do, so the file is never left half-changed. Same rules as edit: read the file first, and each old_string must be unique unless you set replace_all."
    )]
    pub(crate) async fn multi_edit(
        &self,
        params: Parameters<MultiEditParams>,
    ) -> Result<CallToolResult, McpError> {
        let file_path = PathBuf::from(&params.0.file_path);

        if !file_path.exists() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "File not found: {}",
                params.0.file_path
            ))]));
        }

        if let Some(refusal) = self.check_may_modify(&file_path) {
            return Ok(CallToolResult::error(vec![ContentBlock::text(refusal)]));
        }

        if params.0.edits.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(
                "No edits supplied".to_string(),
            )]));
        }

        let original = fs::read_to_string(&file_path).map_err(|e| {
            McpError::internal_error(format!("Failed to read file: {}", e), None)
        })?;

        // Applied to a working copy first. A partial application would leave the file in a
        // state neither the model nor the user asked for, and the model's next edit would
        // then be matching against text that no longer exists.
        let mut working = original.clone();
        let mut replacements = 0usize;

        for (index, edit) in params.0.edits.iter().enumerate() {
            let position = index + 1;

            if edit.old_string == edit.new_string {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Edit {} makes no change: old_string and new_string are identical. \
                     Nothing was written.",
                    position
                ))]));
            }

            let matches = working.matches(&edit.old_string).count();

            if matches == 0 {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Edit {} of {}: string not found: '{}'. Nothing was written. Remember \
                     that each edit applies to the result of the ones before it.",
                    position,
                    params.0.edits.len(),
                    truncate_at_boundary(&edit.old_string, 300)
                ))]));
            }

            if matches > 1 && !edit.replace_all {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Edit {} of {}: old_string is not unique, it occurs {} times. Nothing \
                     was written. Add surrounding context, or set replace_all on that edit.",
                    position,
                    params.0.edits.len(),
                    matches
                ))]));
            }

            working = if edit.replace_all {
                working.replace(&edit.old_string, &edit.new_string)
            } else {
                working.replacen(&edit.old_string, &edit.new_string, 1)
            };
            replacements += matches;
        }

        self.checkpoint(
            &file_path,
            format!(
                "multi-edited {} ({} edit(s), {} replacement(s))",
                params.0.file_path,
                params.0.edits.len(),
                replacements
            ),
        );

        fs::write(&file_path, &working).map_err(|e| {
            McpError::internal_error(format!("Failed to write file: {}", e), None)
        })?;

        self.mark_read(&file_path);

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Applied {} edit(s), {} replacement(s), to {}",
            params.0.edits.len(),
            replacements,
            params.0.file_path
        ))]))
    }


    #[tool(description = "Use when you know how a file is named or where it lives but not its exact path — locating files by name or extension across a tree. Fast file pattern matching that supports glob patterns (\"*.rs\", \"**/*.{ts,tsx}\"). Respects .gitignore, so build output (target/, node_modules/) is skipped automatically. Returns paths sorted by modification time, newest first. Batch multiple searches when useful.")]
    pub(crate) async fn glob_search(&self, params: Parameters<GlobParams>) -> Result<CallToolResult, McpError> {
        // Walking a tree and stat-ing every match is blocking work; see `grep` below for
        // why none of it may happen on this future's own thread.
        let cancel = Arc::new(AtomicBool::new(false));
        let _abort = CancelOnDrop(cancel.clone());

        let params = params.0;
        let output = tokio::task::spawn_blocking(move || glob_search_blocking(&params, &cancel))
            .await
            .map_err(|e| McpError::internal_error(format!("glob_search failed: {}", e), None))?
            .map_err(|e| McpError::invalid_params(e, None))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
    }


    #[tool(description = "Use when you need to find where something appears in the code — a function, variable, string literal, or any regex pattern — across many files at once. Searches file contents using regex. Respects .gitignore, so build output (target/, node_modules/) is skipped automatically. Filter by glob (\"*.rs\", \"**/*.{ts,tsx}\"). Output modes: content (lines), files_with_matches (paths, default), count. Escape literal braces.")]
    pub(crate) async fn grep(&self, params: Parameters<GrepParams>) -> Result<CallToolResult, McpError> {
        let params = params.0;

        // Compiled before anything is spawned: a bad pattern is the caller's own mistake
        // and should come straight back, not after a thread has been started for it.
        let regex = if params.case_insensitive.unwrap_or(false) {
            regex::RegexBuilder::new(&params.pattern)
                .case_insensitive(true)
                .build()
        } else {
            regex::Regex::new(&params.pattern)
        }.map_err(|e| {
            McpError::invalid_params(format!("Invalid regex pattern: {}", e), None)
        })?;

        // Reading every file in a tree is blocking work, and in the REPL this future is
        // polled by the same `select!` that reads the keyboard: run inline, a grep over
        // ~/.cargo/registry — ninety thousand files, and it is in `read_paths` for anyone
        // who wants crate sources readable — froze the bar, the spinner and Esc itself for
        // as long as it ran. There is no await inside the search to hand control back with.
        //
        // So it goes to the blocking pool, and the flag is set when this future is dropped,
        // which is exactly what Esc does — otherwise the detached thread would keep reading
        // for minutes with nobody left to receive the answer.
        let cancel = Arc::new(AtomicBool::new(false));
        let _abort = CancelOnDrop(cancel.clone());

        let output = tokio::task::spawn_blocking(move || grep_blocking(&params, &regex, &cancel))
            .await
            .map_err(|e| McpError::internal_error(format!("grep failed: {}", e), None))?
            .map_err(|e| McpError::invalid_params(e, None))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
    }

    // Shared helper: fetch a URL and return its cleaned, size-limited text. Used by both
    // web_fetch (which then summarizes via the LLM) and web_get (which returns text directly).

    #[tool(description = "Use when you need to see what a directory actually contains — discovering files and subfolders, confirming a path exists, or checking a parent directory before writing into it. Lists directory contents with file/folder information. Prefer Glob/Grep when you already know what you're looking for.")]
    pub(crate) async fn ls(&self, params: Parameters<LSParams>) -> Result<CallToolResult, McpError> {
        let path = PathBuf::from(&params.0.path);

        if !path.exists() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Path not found: {}",
                params.0.path
            ))]));
        }

        if !path.is_dir() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Path is not a directory: {}",
                params.0.path
            ))]));
        }

        let mut entries = Vec::new();

        match fs::read_dir(&path) {
            Ok(dir_entries) => {
                for entry in dir_entries {
                    if let Ok(entry) = entry {
                        let file_name = entry.file_name().to_string_lossy().to_string();
                        let metadata = entry.metadata().ok();

                        let entry_type = if entry.path().is_dir() {
                            "DIR "
                        } else {
                            "FILE"
                        };

                        let size = metadata
                            .as_ref()
                            .map(|m| m.len())
                            .map(|s| {
                                if s < 1024 {
                                    format!("{} B", s)
                                } else if s < 1024 * 1024 {
                                    format!("{:.1} KB", s as f64 / 1024.0)
                                } else if s < 1024 * 1024 * 1024 {
                                    format!("{:.1} MB", s as f64 / (1024.0 * 1024.0))
                                } else {
                                    format!("{:.1} GB", s as f64 / (1024.0 * 1024.0 * 1024.0))
                                }
                            })
                            .unwrap_or_else(|| "?".to_string());

                        entries.push(format!("{} {:>10}  {}", entry_type, size, file_name));
                    }
                }
            }
            Err(e) => {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Failed to read directory: {}",
                    e
                ))]));
            }
        }

        entries.sort();

        let output = if entries.is_empty() {
            format!("Directory is empty: {}", params.0.path)
        } else {
            format!("Contents of {}:\n\n{}", params.0.path, entries.join("\n"))
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
    }

}

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use crate::config::*;
    use super::CancelOnDrop;
    use std::sync::atomic::AtomicBool;
    use crate::llm::*;
    use crate::permissions::*;
    use crate::render::*;
    use crate::session::*;
    use crate::testutil::*;
    use crate::tools::*;
    use crate::util::*;
    use serde_json::json;

    #[test]
    fn glob_matches_full_path_or_bare_file_name() {
        let rs = build_glob_matcher("*.rs").unwrap();
        // The bare-extension form models actually write has to reach nested files.
        assert!(glob_matches(&rs, std::path::Path::new("src/main.rs")));
        assert!(glob_matches(&rs, std::path::Path::new("main.rs")));
        assert!(!glob_matches(&rs, std::path::Path::new("src/main.toml")));

        // A pattern with a separator is matched against the path.
        let nested = build_glob_matcher("src/**/*.rs").unwrap();
        assert!(glob_matches(&nested, std::path::Path::new("src/a/b.rs")));

        // Brace expansion, which the `glob` crate could not do.
        let web = build_glob_matcher("**/*.{ts,tsx}").unwrap();
        assert!(glob_matches(&web, std::path::Path::new("app/x.tsx")));
        assert!(glob_matches(&web, std::path::Path::new("x.ts")));
        assert!(!glob_matches(&web, std::path::Path::new("x.js")));

        assert!(build_glob_matcher("[").is_err());
    }


    #[test]
    fn walker_skips_build_output_and_the_git_dir() {
        // Runs against haico's own tree, which always has a target/ and a .git/ when the
        // tests are run — the exact case that used to bury every search result.
        let files = walk_source_files(".", None, &AtomicBool::new(false)).unwrap();
        assert!(!files.is_empty());

        let has_component = |name: &str| {
            files
                .iter()
                .any(|p| p.components().any(|c| c.as_os_str() == name))
        };
        assert!(!has_component("target"), "target/ was not skipped");
        assert!(!has_component(".git"), ".git/ was not skipped");

        // Dotfiles a coding agent legitimately needs stay visible.
        assert!(files.iter().any(|p| p.file_name().unwrap() == ".gitignore"));

        let rust = walk_source_files(".", Some("*.rs"), &AtomicBool::new(false)).unwrap();
        assert!(rust.iter().any(|p| p.ends_with("main.rs")));
        assert!(rust.iter().all(|p| p.extension().unwrap() == "rs"));

        // Patterns carrying a directory component have to work too. Walking "." yields
        // "./src/main.rs", and matching that raw makes every such pattern return nothing —
        // a bare "*.rs" hid it, because it matches on the file name instead.
        for pattern in ["src/*.rs", "src/**/*.rs", "**/*.rs"] {
            let found = walk_source_files(".", Some(pattern), &AtomicBool::new(false)).unwrap();
            assert!(
                found.iter().any(|p| p.ends_with("main.rs")),
                "pattern {} found {:?}",
                pattern,
                found
            );
        }

        // And a directory pattern must still exclude what it does not name.
        let only_src = walk_source_files(".", Some("src/*.rs"), &AtomicBool::new(false)).unwrap();
        assert!(only_src.iter().all(|p| p.to_string_lossy().contains("src")));

        assert!(walk_source_files(".", Some("["), &AtomicBool::new(false)).is_err());
    }


    #[tokio::test]
    async fn edit_refuses_a_non_unique_old_string() {
        let server = test_server();
        let path = scratch("dup.rs", "let x = 1;\nlet y = 1;\nlet z = 2;\n");
        read_file(&server, &path).await;

        // "= 1;" appears twice: silently editing the first one is the bug being fixed.
        let (ok, message) = try_edit(&server, &path, "= 1;", "= 9;", false).await;
        assert!(!ok, "expected a refusal, got: {}", message);
        assert!(message.contains("not unique"), "got: {}", message);
        assert!(message.contains("occurs 2 times"), "got: {}", message);
        // The file is untouched.
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "let x = 1;\nlet y = 1;\nlet z = 2;\n"
        );

        // replace_all is the documented way through, and reports the real count.
        let (ok, message) = try_edit(&server, &path, "= 1;", "= 9;", true).await;
        assert!(ok, "got: {}", message);
        assert!(message.contains("2 occurrence"), "got: {}", message);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "let x = 9;\nlet y = 9;\nlet z = 2;\n"
        );

        // So is enough surrounding context to make it unique.
        let (ok, _) = try_edit(&server, &path, "let z = 2;", "let z = 3;", false).await;
        assert!(ok);
        assert!(std::fs::read_to_string(&path).unwrap().contains("let z = 3;"));

        let _ = std::fs::remove_file(&path);
    }


    #[tokio::test]
    async fn edit_requires_reading_the_file_first_and_notices_staleness() {
        let server = test_server();
        let path = scratch("gate.rs", "original\n");

        // Never read: refused.
        let (ok, message) = try_edit(&server, &path, "original", "changed", false).await;
        assert!(!ok);
        assert!(message.contains("has not been read"), "got: {}", message);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "original\n");

        // Read, then edit: allowed.
        read_file(&server, &path).await;
        let (ok, message) = try_edit(&server, &path, "original", "changed", false).await;
        assert!(ok, "got: {}", message);

        // A second edit straight after the first is not treated as stale, even though the
        // write just changed the file's mtime.
        let (ok, message) = try_edit(&server, &path, "changed", "changed twice", false).await;
        assert!(ok, "got: {}", message);

        // Somebody else edits the file behind our back.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&path, "edited in another editor\n").unwrap();
        let (ok, message) = try_edit(&server, &path, "edited", "clobbered", false).await;
        assert!(!ok);
        assert!(message.contains("changed on disk"), "got: {}", message);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "edited in another editor\n"
        );

        let _ = std::fs::remove_file(&path);
    }


    #[tokio::test]
    async fn write_file_gates_overwrites_but_not_new_files() {
        let server = test_server();
        let dir = scratch("anchor", "").parent().unwrap().to_path_buf();
        let fresh = dir.join("brand-new.rs");
        let _ = std::fs::remove_file(&fresh);

        let write = |path: std::path::PathBuf, content: &str| {
            let content = content.to_string();
            let server = server.clone();
            async move {
                server
                    .write_file(rmcp::handler::server::wrapper::Parameters(
                        crate::WriteParams {
                            file_path: path.display().to_string(),
                            content,
                        },
                    ))
                    .await
                    .unwrap()
            }
        };

        // Creating a file that does not exist needs no prior read.
        let result = write(fresh.clone(), "fn main() {}\n").await;
        assert!(!result.is_error.unwrap_or(false), "{}", result_text(&result));
        assert_eq!(std::fs::read_to_string(&fresh).unwrap(), "fn main() {}\n");

        // And the write itself counts as knowing the contents, so an immediate overwrite
        // is fine rather than being refused as unread.
        let result = write(fresh.clone(), "fn main() { todo!() }\n").await;
        assert!(!result.is_error.unwrap_or(false), "{}", result_text(&result));

        // But a pre-existing file this session has never seen is refused.
        let unseen = scratch("unseen.rs", "important\n");
        let result = write(unseen.clone(), "gone\n").await;
        assert!(result.is_error.unwrap_or(false));
        assert!(result_text(&result).contains("has not been read"));
        assert_eq!(std::fs::read_to_string(&unseen).unwrap(), "important\n");

        let _ = std::fs::remove_file(&fresh);
        let _ = std::fs::remove_file(&unseen);
    }


    #[tokio::test]
    async fn multi_edit_applies_in_order_or_not_at_all() {
        let server = test_server();
        let path = scratch("multi.rs", "fn one() {}\nfn two() {}\nfn three() {}\n");
        read_file(&server, &path).await;

        // Applied in sequence, each seeing the previous result.
        let (ok, out) = try_multi_edit(
            &server,
            &path,
            vec![("fn one", "fn first", false), ("fn two", "fn second", false)],
        )
        .await;
        assert!(ok, "got: {}", out);
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("fn first"));
        assert!(content.contains("fn second"));
        assert!(content.contains("fn three"));

        // A later edit may depend on an earlier one's output.
        let (ok, out) = try_multi_edit(
            &server,
            &path,
            vec![("fn first", "fn alpha", false), ("fn alpha() {}", "fn alpha() { todo!() }", false)],
        )
        .await;
        assert!(ok, "got: {}", out);
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("fn alpha() { todo!() }"));

        // If any edit fails, none are written — a half-applied file is worse than none,
        // because the model's next edit would target text that no longer exists.
        let before = std::fs::read_to_string(&path).unwrap();
        let (ok, out) = try_multi_edit(
            &server,
            &path,
            vec![
                ("fn second", "fn beta", false),
                ("fn nonexistent", "fn gamma", false),
            ],
        )
        .await;
        assert!(!ok);
        assert!(out.contains("Edit 2 of 2"), "got: {}", out);
        assert!(out.contains("Nothing was written"), "got: {}", out);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);

        // Non-unique strings are refused per-edit, same as `edit`.
        let (ok, out) = try_multi_edit(&server, &path, vec![("fn ", "pub fn ", false)]).await;
        assert!(!ok);
        assert!(out.contains("not unique"), "got: {}", out);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);

        // ...and replace_all is the way through.
        let (ok, _) = try_multi_edit(&server, &path, vec![("fn ", "pub fn ", true)]).await;
        assert!(ok);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap().matches("pub fn ").count(),
            3
        );

        let _ = std::fs::remove_file(&path);
    }


    #[tokio::test]
    async fn read_reports_a_binary_file_instead_of_failing_opaquely() {
        let server = test_server();

        let dir = scratch("anchor3", "").parent().unwrap().to_path_buf();
        let binary = dir.join("blob.bin");
        std::fs::write(&binary, [0x00u8, 0xFF, 0xFE, 0x80]).unwrap();

        let result = server
            .read(rmcp::handler::server::wrapper::Parameters(crate::ReadParams {
                file_path: binary.display().to_string(),
                offset: None,
                limit: None,
            }))
            .await
            .unwrap();

        // A clean tool error, not an internal fault: the model needs to understand it should
        // stop rather than retry the same read.
        assert!(result.is_error.unwrap_or(false));
        let text = result_text(&result);
        assert!(text.contains("not a text file"), "got: {}", text);

        // And a failed read must not count as having seen the file, or the write gate would
        // wave through an overwrite based on contents nobody ever looked at.
        assert!(server.check_may_modify(&binary).is_some());

        let _ = std::fs::remove_file(&binary);
    }

    #[test]
    fn a_cancelled_walk_stops_where_it_is() {
        // Set before the walk starts, so there is nothing to race with: what it proves is
        // that the flag is looked at at all, which is what an interrupted grep depends on.
        let cancel = AtomicBool::new(true);
        let files = walk_source_files(".", None, &cancel).unwrap();
        assert!(files.is_empty(), "a cancelled walk returned {} files", files.len());
    }

    #[test]
    fn the_guard_cancels_the_search_when_the_future_holding_it_goes_away() {
        // The guard lives in the async fn's own frame, so this drop is what Esc does: the
        // future is dropped mid-await and the detached thread has to be told to stop.
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        {
            let _abort = CancelOnDrop(cancel.clone());
            assert!(!cancel.load(std::sync::atomic::Ordering::Relaxed));
        }
        assert!(
            cancel.load(std::sync::atomic::Ordering::Relaxed),
            "the search was left running after its future was dropped"
        );
    }

    #[tokio::test]
    async fn grep_still_returns_its_matches_from_the_thread_pool() {
        // The search moved off the future's own thread; the result must not have moved
        // with it. Both the REPL and the MCP router come through execute_tool.
        let server = crate::testutil::test_server();
        let found = server
            .execute_tool(
                "grep",
                serde_json::json!({
                    "pattern": "fn walk_source_files",
                    "path": "src",
                    "output_mode": "files_with_matches"
                }),
            )
            .await
            .unwrap();
        assert!(found.contains("files.rs"), "got: {}", found);

        // Joined at runtime: written out in one piece, the needle would be sitting in
        // this very file and the search would find it.
        let needle = format!("{}{}", "nothing-matches", "-this-xyzzy");
        let missing = server
            .execute_tool("grep", serde_json::json!({"pattern": needle}))
            .await
            .unwrap();
        assert!(missing.starts_with("No matches found"), "got: {}", missing);

        // An unparseable pattern still comes back as an error rather than reaching a thread.
        assert!(server
            .execute_tool("grep", serde_json::json!({"pattern": "("}))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn a_search_hands_control_back_before_it_reads_anything() {
        // The property the REPL depends on, and the one a future rewrite could quietly
        // take away: the first poll must spawn the work and return, not do it. The bar,
        // the spinner and Esc all live in the `select!` that polls this future, and while
        // a poll is running none of them exist.
        use std::future::Future;

        let server = crate::testutil::test_server();
        let mut search = Box::pin(server.execute_tool(
            "grep",
            serde_json::json!({"pattern": "fn ", "path": "src", "output_mode": "count"}),
        ));

        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(
            search.as_mut().poll(&mut cx).is_pending(),
            "the search did its reading on the first poll, which is the freeze itself"
        );

        // And it still finishes, with the counts it was asked for.
        let counted = search.await.unwrap();
        assert!(counted.contains("files.rs:"), "got: {}", counted);
    }
}
