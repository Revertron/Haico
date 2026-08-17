use anyhow::{bail, Context, Result};
use reqwest;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, tool_handler, ServerHandler, ServiceExt, ErrorData as McpError};
use rmcp::handler::server::{tool::ToolRouter, wrapper::Parameters};
use rmcp::transport::stdio;
use serde_json::{json, Value};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::collections::HashSet;
use std::process::Command;
use std::env;
use std::sync::{Arc, Mutex};
use std::num::NonZeroUsize;
use lru::LruCache;
use chrono;
use tokio;
use getopts::Options;
use crossterm::{
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
    event::{self, Event, KeyCode, KeyEventKind},
    style::{Color, ResetColor, SetBackgroundColor, SetForegroundColor, Attribute, SetAttribute},
    execute,
    cursor,
};
use similar::{ChangeTag, TextDiff};

const SYSTEM_PROMPT: &str = include_str!("system_prompt.txt");

fn run_git_command(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
}

fn get_main_branch() -> String {
    // Try to get the default branch from the remote
    if let Some(remote_head) = run_git_command(&["symbolic-ref", "refs/remotes/origin/HEAD"]) {
        // Output format: "refs/remotes/origin/main"
        if let Some(branch) = remote_head.strip_prefix("refs/remotes/origin/") {
            return branch.to_string();
        }
    }

    // Try to get from git config
    if let Some(default_branch) = run_git_command(&["config", "--get", "init.defaultBranch"]) {
        if !default_branch.is_empty() {
            return default_branch;
        }
    }

    // Fallback: check if common branches exist
    for branch in &["main", "master"] {
        if run_git_command(&["rev-parse", "--verify", branch]).is_some() {
            return branch.to_string();
        }
    }

    // Ultimate fallback
    "main".to_string()
}

fn build_system_prompt() -> String {
    let pwd = env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_string()))
        .unwrap_or_else(|| ".".to_string());

    let is_repo = run_git_command(&["rev-parse", "--git-dir"]).is_some();

    let platform = if cfg!(target_os = "windows") {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    };

    let os_version = std::env::consts::OS;

    let date = chrono::Local::now().format("%Y-%m-%d").to_string();

    let (branch, git_status, last_5_commits) = if is_repo {
        let branch = run_git_command(&["rev-parse", "--abbrev-ref", "HEAD"])
            .unwrap_or_else(|| "unknown".to_string());

        let status = run_git_command(&["status", "--short"])
            .unwrap_or_else(|| "".to_string());

        let commits = run_git_command(&["log", "--oneline", "-5"])
            .unwrap_or_else(|| "".to_string());

        (branch, status, commits)
    } else {
        ("N/A".to_string(), "".to_string(), "".to_string())
    };

    let main_branch = get_main_branch();

    SYSTEM_PROMPT
        .replace("$pwd", &pwd)
        .replace("$is_repo", &is_repo.to_string())
        .replace("$platform", platform)
        .replace("$os_version", os_version)
        .replace("$date", &date)
        .replace("$branch", &branch)
        .replace("$main_branch", &main_branch)
        .replace("$git_status", &git_status)
        .replace("$last_5_commits", &last_5_commits)
}

// Model response structures
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

// Permission tracking system
#[derive(Debug, Clone)]
struct PermissionManager {
    allowed_tools: HashSet<String>, // Tools that are always allowed
}

impl PermissionManager {
    fn new() -> Self {
        Self {
            allowed_tools: HashSet::new(),
        }
    }

    fn is_allowed(&self, tool_name: &str) -> bool {
        self.allowed_tools.contains(tool_name)
    }

    fn allow_tool(&mut self, tool_name: String) {
        self.allowed_tools.insert(tool_name);
    }
}

enum PermissionResponse {
    Allow,
    AllowAll,
    Reject(String),
}

// Generate a colored diff with line numbers, showing only context around changes
fn generate_diff(file_path: &str, old_content: &str, new_content: &str) -> Result<String> {
    let diff = TextDiff::from_lines(old_content, new_content);
    let mut output = Vec::new();

    writeln!(&mut output, "File: {}", file_path)?;

    const CONTEXT_LINES: usize = 5;

    // First pass: collect all changes with their positions
    #[derive(Clone)]
    struct Change {
        tag: ChangeTag,
        text: String,
        old_line: Option<usize>,
        new_line: Option<usize>,
    }

    let mut changes = Vec::new();
    let mut old_line_num = 1;
    let mut new_line_num = 1;

    for change in diff.iter_all_changes() {
        let text = change.as_str().unwrap_or("");
        let text_trimmed = text.trim_end_matches('\n').trim_end_matches('\r').to_string();

        match change.tag() {
            ChangeTag::Delete => {
                changes.push(Change {
                    tag: ChangeTag::Delete,
                    text: text_trimmed,
                    old_line: Some(old_line_num),
                    new_line: None,
                });
                old_line_num += 1;
            }
            ChangeTag::Insert => {
                changes.push(Change {
                    tag: ChangeTag::Insert,
                    text: text_trimmed,
                    old_line: None,
                    new_line: Some(new_line_num),
                });
                new_line_num += 1;
            }
            ChangeTag::Equal => {
                changes.push(Change {
                    tag: ChangeTag::Equal,
                    text: text_trimmed,
                    old_line: Some(old_line_num),
                    new_line: Some(new_line_num),
                });
                old_line_num += 1;
                new_line_num += 1;
            }
        }
    }

    // Second pass: identify change regions
    let mut change_indices = Vec::new();
    for (i, change) in changes.iter().enumerate() {
        if !matches!(change.tag, ChangeTag::Equal) {
            change_indices.push(i);
        }
    }

    if change_indices.is_empty() {
        writeln!(&mut output, "No changes")?;
        return Ok(String::from_utf8(output)?);
    }

    // Group consecutive changes into regions with context
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut current_start = change_indices[0].saturating_sub(CONTEXT_LINES);
    let mut current_end = change_indices[0] + CONTEXT_LINES;

    for &idx in &change_indices[1..] {
        let region_start = idx.saturating_sub(CONTEXT_LINES);
        let region_end = idx + CONTEXT_LINES;

        // If regions overlap or are adjacent, merge them
        if region_start <= current_end + 1 {
            current_end = region_end;
        } else {
            // Save current region and start a new one
            regions.push((current_start, current_end.min(changes.len() - 1)));
            current_start = region_start;
            current_end = region_end;
        }
    }
    regions.push((current_start, current_end.min(changes.len() - 1)));

    // Render regions
    for (region_idx, (start, end)) in regions.iter().enumerate() {
        if region_idx > 0 {
            writeln!(&mut output, "       ...")?;
        }

        for i in *start..=*end {
            let change = &changes[i];
            match change.tag {
                ChangeTag::Delete => {
                    if let Some(line_num) = change.old_line {
                        execute!(&mut output, SetForegroundColor(Color::DarkGrey))?;
                        write!(&mut output, "{:6}", line_num)?;
                        execute!(&mut output, ResetColor)?;
                        write!(&mut output, " ")?;
                        execute!(&mut output, SetBackgroundColor(Color::Red))?;
                        write!(&mut output, "- {}", change.text)?;
                        execute!(&mut output, ResetColor)?;
                        writeln!(&mut output)?;
                    }
                }
                ChangeTag::Insert => {
                    if let Some(line_num) = change.new_line {
                        execute!(&mut output, SetForegroundColor(Color::DarkGrey))?;
                        write!(&mut output, "{:6}", line_num)?;
                        execute!(&mut output, ResetColor)?;
                        write!(&mut output, " ")?;
                        execute!(&mut output, SetBackgroundColor(Color::Green))?;
                        write!(&mut output, "+ {}", change.text)?;
                        execute!(&mut output, ResetColor)?;
                        writeln!(&mut output)?;
                    }
                }
                ChangeTag::Equal => {
                    if let Some(line_num) = change.old_line {
                        execute!(&mut output, SetForegroundColor(Color::DarkGrey))?;
                        write!(&mut output, "{:6}", line_num)?;
                        execute!(&mut output, ResetColor)?;
                        write!(&mut output, " ")?;
                        writeln!(&mut output, "  {}", change.text)?;
                    }
                }
            }
        }
    }

    Ok(String::from_utf8(output)?)
}

// Ask user for permission to execute a tool
fn ask_permission(tool_name: &str, description: &str, diff: Option<&str>) -> Result<PermissionResponse> {
    println!("🔒 Allow this tool usage?");
    println!("{}\n", description);

    if let Some(diff_content) = diff {
        println!("{}", diff_content);
    }

    println!("1. Allow once");
    println!("2. Allow all '{}' tool usage", tool_name);
    println!("3. Reject and provide alternative");
    io::stdout().flush()?;

    // Enable raw mode
    enable_raw_mode()?;

    let response = loop {
        if let Event::Key(key_event) = event::read()? {
            // Only process key press events, ignore key release
            if key_event.kind != KeyEventKind::Press {
                continue;
            }

            match key_event.code {
                KeyCode::Char('1') => {
                    break PermissionResponse::Allow;
                }
                KeyCode::Char('2') => {
                    break PermissionResponse::AllowAll;
                }
                KeyCode::Char('3') => {
                    // Disable raw mode to get text input
                    disable_raw_mode()?;
                    println!("3\n");

                    print!("> ");
                    io::stdout().flush()?;

                    let mut alternative = String::new();
                    io::stdin().read_line(&mut alternative)?;

                    break PermissionResponse::Reject(alternative.trim().to_string());
                }
                _ => {
                    // Invalid key, continue waiting
                    continue;
                }
            }
        }
    };

    // Disable raw mode if not already disabled
    let _ = disable_raw_mode();

    // Drain any remaining events from the input buffer to prevent key queue issues
    use std::time::Duration;
    while event::poll(Duration::from_millis(0))? {
        let _ = event::read()?;
    }

    Ok(response)
}

// Normalize an LLM server URL down to a bare origin, so the "/v1/..." paths below always
// concatenate cleanly. LM Studio's UI hands out its base URL with "/v1" already appended
// (and OpenAI SDKs expect it that way), which would otherwise build
// "/v1/v1/chat/completions" — a path LM Studio rejects while still answering HTTP 200.
fn normalize_base_url(url: &str) -> String {
    let url = url.trim().trim_end_matches('/');
    url.strip_suffix("/v1")
        .unwrap_or(url)
        .trim_end_matches('/')
        .to_string()
}

// Explain a 401 in terms of what the user can actually change.
fn unauthorized_message(had_token: bool) -> String {
    if had_token {
        "LLM server rejected the request (401 Unauthorized): the configured token was not \
         accepted. Check --token / LM_API_TOKEN."
            .to_string()
    } else {
        "LLM server rejected the request (401 Unauthorized): it requires an API token, but \
         none was configured. Pass --token or set LM_API_TOKEN."
            .to_string()
    }
}

// LM Studio client
#[derive(Clone)]
struct LMStudioClient {
    client: reqwest::Client,
    base_url: String,
    // Model to request. An empty string means "no model configured": the "model" field is
    // then omitted from requests entirely, so the server answers with the model it already
    // has loaded.
    model: String,
    // Bearer token for servers with authentication enabled (LM Studio's "API token").
    // None means send no Authorization header, which is what a default local server wants.
    token: Option<String>,
}

impl LMStudioClient {
    fn new(base_url: String, model: String, token: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: normalize_base_url(&base_url),
            model,
            token,
        }
    }

    // Attach the bearer token, when one is configured.
    fn authorize(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => req.bearer_auth(token),
            None => req,
        }
    }

    // Add the "model" field to a request body, unless no model is configured.
    fn set_model(&self, body: &mut Value) {
        if !self.model.is_empty() {
            body["model"] = json!(self.model);
        }
    }

    // POST a chat-completions request. An auth failure is reported as such, instead of
    // surfacing later as a confusing "failed to parse response" on the error JSON.
    async fn post_chat(&self, body: &Value) -> Result<Value> {
        let response = self
            .authorize(
                self.client
                    .post(format!("{}/v1/chat/completions", self.base_url)),
            )
            .json(body)
            .send()
            .await
            .context("Failed to send request to LM Studio")?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            bail!("{}", unauthorized_message(self.token.is_some()));
        }

        response
            .json::<Value>()
            .await
            .context("Failed to parse response")
    }

    async fn chat_completion(&self, messages: Vec<Value>, tools: Vec<Value>) -> Result<Value> {
        let mut body = json!({
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        });
        self.set_model(&mut body);

        self.post_chat(&body).await
    }

    async fn simple_completion(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let messages = vec![
            json!({
                "role": "system",
                "content": system_prompt
            }),
            json!({
                "role": "user",
                "content": user_prompt
            })
        ];

        let mut body = json!({
            "messages": messages
        });
        self.set_model(&mut body);

        let result = self.post_chat(&body).await?;

        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(content)
    }
}

// Tool parameter structs
#[derive(Serialize, Deserialize, JsonSchema)]
struct CalculatorParams {
    #[schemars(description = "The arithmetic operation to perform: 'add', 'subtract', 'multiply', or 'divide'")]
    operation: String,
    #[schemars(description = "The first operand")]
    a: f64,
    #[schemars(description = "The second operand")]
    b: f64,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct ReadParams {
    #[schemars(description = "Absolute path to the file to read")]
    file_path: String,
    #[schemars(description = "Line number to start reading from (default: 0)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<usize>,
    #[schemars(description = "Maximum number of lines to read (default: 2000)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct WriteParams {
    #[schemars(description = "Absolute path where the file should be written")]
    file_path: String,
    #[schemars(description = "The complete content to write to the file")]
    content: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct EditParams {
    #[schemars(description = "Absolute path to the file to modify")]
    file_path: String,
    #[schemars(description = "The exact text to search for and replace")]
    old_string: String,
    #[schemars(description = "The text to replace it with (must be different from old_string)")]
    new_string: String,
    #[schemars(description = "Replace all occurrences of old_string (default false)")]
    #[serde(default)]
    replace_all: bool,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct GlobParams {
    #[schemars(description = "Glob pattern to match files (e.g., '**/*.rs', '*.txt')")]
    pattern: String,
    #[schemars(description = "Base directory to search in (default: current directory)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct GrepParams {
    #[schemars(description = "Regular expression pattern to search for")]
    pattern: String,
    #[schemars(description = "File or directory to search in (default: current directory)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[schemars(description = "Glob pattern to filter files (e.g., '*.rs', '*.{ts,tsx}')")]
    #[serde(skip_serializing_if = "Option::is_none")]
    glob: Option<String>,
    #[schemars(description = "Output format: 'content' (matching lines), 'files_with_matches' (file paths, default), or 'count' (match counts)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    output_mode: Option<String>,
    #[schemars(description = "Perform case-insensitive search (default: false)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    case_insensitive: Option<bool>,
    #[schemars(description = "Limit output to first N results")]
    #[serde(skip_serializing_if = "Option::is_none")]
    head_limit: Option<usize>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct WebFetchParams {
    #[schemars(description = "The URL to fetch content from")]
    url: String,
    #[schemars(description = "A prompt describing what information to extract from the fetched page")]
    prompt: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct WebGetParams {
    #[schemars(description = "The URL to fetch content from")]
    url: String,
}

// Outcome of fetching and cleaning a web page: either a cross-host redirect notice to
// surface to the caller, or the cleaned, size-limited page text.
#[derive(Clone)]
enum FetchResult {
    Redirect(String),
    Text { final_url: String, text: String },
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct WebSearchParams {
    #[schemars(description = "The search query to submit to the search engine")]
    query: String,
    #[schemars(description = "Maximum number of search results to return (default: 10)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    max_results: Option<usize>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct LSParams {
    #[schemars(description = "Path to the directory to list")]
    path: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct RunCommandParams {
    #[schemars(description = "The shell command to execute")]
    command: String,
    #[schemars(description = "Timeout in seconds (default: 120, max: 600)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<u64>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct CrateInfoParams {
    #[schemars(description = "The exact crate name as published on crates.io (e.g. 'serde', 'tokio', 'reqwest')")]
    name: String,
    #[schemars(description = "Optional exact version to inspect (e.g. '1.0.210'). Defaults to the latest stable release.")]
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct CrateSearchParams {
    #[schemars(description = "Free-text search query, e.g. 'async http client' or 'json serialization'")]
    query: String,
    #[schemars(description = "Maximum number of results to return (default: 15, max: 50)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct CrateApiParams {
    #[schemars(description = "The exact crate name as published on crates.io (e.g. 'serde')")]
    name: String,
    #[schemars(description = "Crate version, or 'latest'. Defaults to 'latest'.")]
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    #[schemars(description = "Optional case-sensitive substring to filter the public API by item path (e.g. 'Serialize', 'Client::new', 'builder'). Omit to list the entire public API.")]
    #[serde(skip_serializing_if = "Option::is_none")]
    item: Option<String>,
}

// Base endpoints for the Rust API tools.
const CRATES_IO_API: &str = "https://crates.io/api/v1";
const DOCS_RS: &str = "https://docs.rs";

// Shared HTTP client for the Rust API tools (crates.io / docs.rs). crates.io's crawler
// policy requires a descriptive, identifying User-Agent, so we always set one.
fn rust_api_client() -> Result<reqwest::Client, McpError> {
    reqwest::Client::builder()
        .user_agent("haico/0.1.0 (Rust crate API lookup tool)")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| McpError::internal_error(format!("Failed to build HTTP client: {}", e), None))
}

// The Firefox release we impersonate in the web tools. Bump this occasionally: a
// user-agent that trails the real release train by too many versions is itself a
// bot signal.
const FIREFOX_VERSION: &str = "153";

// Shared HTTP client for the web tools (web_search, web_get, web_fetch). Many sites
// challenge or reject requests whose headers look like a bare HTTP library, so we send
// the same header set a desktop Firefox sends for a top-level page load.
//
// This only disguises the headers. The TLS ClientHello and HTTP/2 settings are still
// reqwest's, so services that fingerprint at that level (Cloudflare, DataDome) can
// still tell us apart from a real browser.
fn browser_client() -> Result<reqwest::Client, McpError> {
    use reqwest::header::{
        HeaderMap, HeaderValue, ACCEPT, ACCEPT_ENCODING, ACCEPT_LANGUAGE,
        UPGRADE_INSECURE_REQUESTS,
    };

    let mut headers = HeaderMap::new();
    // Verbatim from DocumentAcceptHeader() in Firefox's nsHttpHandler.cpp.
    headers.insert(
        ACCEPT,
        HeaderValue::from_static(
            "text/html,application/xhtml+xml,application/xml;q=0.9,\
             image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
        ),
    );
    headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("en-US,en;q=0.5"));
    // Safe to advertise only because the matching reqwest features are enabled in
    // Cargo.toml; reqwest keeps this header as written and still decodes the response.
    headers.insert(
        ACCEPT_ENCODING,
        HeaderValue::from_static("gzip, deflate, br, zstd"),
    );
    headers.insert(UPGRADE_INSECURE_REQUESTS, HeaderValue::from_static("1"));
    // Fetch metadata for a user-initiated, top-level navigation. web_search overrides
    // Sec-Fetch-Site because it sends a Referer.
    headers.insert("Sec-Fetch-Dest", HeaderValue::from_static("document"));
    headers.insert("Sec-Fetch-Mode", HeaderValue::from_static("navigate"));
    headers.insert("Sec-Fetch-Site", HeaderValue::from_static("none"));
    headers.insert("Sec-Fetch-User", HeaderValue::from_static("?1"));
    headers.insert("Priority", HeaderValue::from_static("u=0, i"));

    reqwest::Client::builder()
        .user_agent(format!(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{v}.0) Gecko/20100101 Firefox/{v}.0",
            v = FIREFOX_VERSION
        ))
        .default_headers(headers)
        // Anti-bot flows commonly set a cookie and redirect; without a jar we drop it.
        .cookie_store(true)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| McpError::internal_error(format!("Failed to build HTTP client: {}", e), None))
}

// Resolve a search result's URL into an absolute, scheme-carrying one.
//
// DuckDuckGo's HTML results print the URL without a scheme in the link text
// ("rust-lang.org/tools/install") and wrap the real target in a protocol-relative
// redirect ("//duckduckgo.com/l/?uddg=<percent-encoded target>&rut=..."). Models that
// get the bare text back often fail to prepend a scheme and then can't fetch the page,
// so prefer the target decoded out of the redirect and fall back to the display text
// with https:// prepended.
fn resolve_result_url(href: Option<&str>, display_text: &str) -> String {
    if let Some(href) = href {
        let absolute = if href.starts_with("//") {
            format!("https:{}", href)
        } else {
            href.to_string()
        };

        if let Ok(parsed) = url::Url::parse(&absolute) {
            // query_pairs() percent-decodes, so this is already the plain target URL.
            if let Some((_, target)) = parsed.query_pairs().find(|(key, _)| key == "uddg") {
                return target.into_owned();
            }
            if parsed.scheme() == "http" || parsed.scheme() == "https" {
                return absolute;
            }
        }
    }

    let text = display_text.trim();
    if text.is_empty() || text.starts_with("http://") || text.starts_with("https://") {
        text.to_string()
    } else {
        format!("https://{}", text)
    }
}

// One parsed search result. Both backends (DuckDuckGo and SearxNG) produce these, and
// format_search_results renders them, so the tool's output looks the same either way.
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

fn format_search_results(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No search results found for query: '{}'", query);
    }

    let body = results
        .iter()
        .enumerate()
        .map(|(index, result)| {
            format!(
                "{}. {}\n   URL: {}\n   {}\n",
                index + 1,
                result.title,
                if result.url.is_empty() { "N/A" } else { &result.url },
                result.snippet
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "Search results for '{}' ({} results):\n\n{}",
        query,
        results.len(),
        body
    )
}

// A SearxNG instance's JSON answer. Only the fields we render are declared.
#[derive(Deserialize)]
struct SearxngResponse {
    #[serde(default)]
    results: Vec<SearxngResult>,
}

#[derive(Deserialize)]
struct SearxngResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
}

// Search through a SearxNG instance instead of DuckDuckGo (--searxng / SEARXNG_URL).
//
// JSON is the format we want, but "json" is not in the default `search.formats` of
// SearxNG's settings.yml, so an instance that hasn't enabled it answers a format=json
// request with a 403 (or an HTML error page). Try JSON first and fall back to scraping
// the normal HTML result page, which every instance serves.
//
// Unlike the DuckDuckGo path there is no silent fallback across providers: --searxng was
// passed deliberately, so a broken instance is reported rather than worked around.
async fn searxng_search(
    base_url: &str,
    query: &str,
    max_results: usize,
) -> Result<CallToolResult, McpError> {
    let client = browser_client()?;
    let search_url = format!("{}/search?q={}", base_url, urlencoding::encode(query));

    let response = client
        .get(format!("{}&format=json", search_url))
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|e| {
            McpError::internal_error(
                format!("Failed to reach the SearxNG instance at {}: {}", base_url, e),
                None,
            )
        })?;

    if response.status().is_success() {
        let body = response.text().await.map_err(|e| {
            McpError::internal_error(format!("Failed to read response body: {}", e), None)
        })?;

        if let Ok(parsed) = serde_json::from_str::<SearxngResponse>(&body) {
            let results: Vec<SearchResult> = parsed
                .results
                .into_iter()
                .take(max_results)
                .map(|r| SearchResult {
                    title: r.title,
                    url: r.url,
                    snippet: r.content,
                })
                .collect();
            return Ok(CallToolResult::success(vec![ContentBlock::text(
                format_search_results(query, &results),
            )]));
        }
    }

    // JSON is not enabled on this instance; scrape the HTML result page instead.
    let html = client
        .get(&search_url)
        .send()
        .await
        .map_err(|e| {
            McpError::internal_error(
                format!("Failed to reach the SearxNG instance at {}: {}", base_url, e),
                None,
            )
        })?
        .error_for_status()
        .map_err(|e| {
            McpError::internal_error(
                format!("SearxNG at {} rejected the search: {}", base_url, e),
                None,
            )
        })?
        .text()
        .await
        .map_err(|e| {
            McpError::internal_error(format!("Failed to read response body: {}", e), None)
        })?;

    let results = parse_searxng_html(&html, max_results);

    // An anti-bot challenge (common on public instances) is served as a normal HTTP 200
    // page, and a URL that isn't a SearxNG instance at all answers with something else
    // entirely. Both scrape to zero results, so check for the result page's own markup
    // before reporting a genuine "no results".
    if results.is_empty() && !html.contains("id=\"urls\"") && !html.contains("id=\"results\"") {
        return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
            "SearxNG at {} did not return a results page. The instance may be behind an \
             anti-bot challenge, or the URL may not point at a SearxNG instance. Enabling \
             'json' under search.formats in the instance's settings.yml avoids HTML parsing \
             entirely.",
            base_url
        ))]));
    }

    Ok(CallToolResult::success(vec![ContentBlock::text(
        format_search_results(query, &results),
    )]))
}

// Scrape SearxNG's "simple" theme result page. Results are <article class="result">, with
// the target on the <h3> link and repeated on the a.url_header wrapper above it (older
// SearxNG put that class on a.url_wrapper instead, so accept either as a fallback).
fn parse_searxng_html(html: &str, max_results: usize) -> Vec<SearchResult> {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);
    let result_selector = Selector::parse(".result").unwrap();
    let title_selector = Selector::parse("h3 a").unwrap();
    let url_selector = Selector::parse("a.url_header, a.url_wrapper").unwrap();
    let snippet_selector = Selector::parse("p.content").unwrap();

    let mut results = Vec::new();

    for result in document.select(&result_selector) {
        if results.len() >= max_results {
            break;
        }

        let title_element = result.select(&title_selector).next();

        let title = title_element
            .map(|e| e.text().collect::<String>())
            .unwrap_or_default()
            .trim()
            .to_string();

        if title.is_empty() {
            continue;
        }

        let url = title_element
            .map(|e| resolve_result_url(e.value().attr("href"), ""))
            .filter(|u| !u.is_empty())
            .or_else(|| {
                result
                    .select(&url_selector)
                    .next()
                    .map(|e| resolve_result_url(e.value().attr("href"), ""))
            })
            .unwrap_or_default();

        let snippet = result
            .select(&snippet_selector)
            .next()
            .map(|e| e.text().collect::<String>())
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        results.push(SearchResult { title, url, snippet });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::{normalize_base_url, parse_searxng_html, resolve_result_url};

    #[test]
    fn normalizes_base_urls() {
        // The forms haico already documents are left alone.
        assert_eq!(
            normalize_base_url("http://127.0.0.1:1234"),
            "http://127.0.0.1:1234"
        );
        // The forms LM Studio's UI and OpenAI SDKs hand out.
        assert_eq!(
            normalize_base_url("http://127.0.0.1:1234/v1"),
            "http://127.0.0.1:1234"
        );
        assert_eq!(
            normalize_base_url("http://127.0.0.1:1234/v1/"),
            "http://127.0.0.1:1234"
        );
        assert_eq!(
            normalize_base_url("  http://127.0.0.1:1234/  "),
            "http://127.0.0.1:1234"
        );
        // Bracketed IPv6 hosts survive intact.
        assert_eq!(
            normalize_base_url("http://[225:ca89:40a8:611e:78b8:ab81:999a:d4d7]:1234/v1"),
            "http://[225:ca89:40a8:611e:78b8:ab81:999a:d4d7]:1234"
        );
        // Known trade-off: stripping is unconditional, so a proxy whose base path really
        // does end in "/v1" loses it. Documented rather than special-cased.
        assert_eq!(
            normalize_base_url("http://example.com/api/v1"),
            "http://example.com/api"
        );
    }

    #[test]
    fn decodes_duckduckgo_redirect() {
        assert_eq!(
            resolve_result_url(
                Some("//duckduckgo.com/l/?uddg=https%3A%2F%2Frust%2Dlang.org%2Ftools%2Finstall%2F&rut=ae4a"),
                "rust-lang.org/tools/install",
            ),
            "https://rust-lang.org/tools/install/"
        );
    }

    #[test]
    fn keeps_a_plain_absolute_href() {
        assert_eq!(
            resolve_result_url(Some("https://example.com/page"), "example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn falls_back_to_display_text_with_a_scheme() {
        assert_eq!(
            resolve_result_url(None, "example.com/page"),
            "https://example.com/page"
        );
        assert_eq!(
            resolve_result_url(None, "https://example.com/page"),
            "https://example.com/page"
        );
        assert_eq!(resolve_result_url(None, ""), "");
    }

    #[test]
    fn parses_searxng_html_results() {
        // Shape of SearxNG's "simple" theme result page (searx/templates/simple/macros.html).
        let html = r#"
            <div id="urls">
              <article class="result result-default category-general">
                <a href="https://rust-lang.org/tools/install" class="url_header" rel="noreferrer">
                  <div class="url_wrapper"><span class="url_o1"><span class="url_i1">rust-lang.org</span></span></div>
                </a>
                <h3><a href="https://rust-lang.org/tools/install" rel="noreferrer">Install Rust</a></h3>
                <p class="content">Rustup is
                   the recommended installer.</p>
              </article>
              <article class="result result-default category-general">
                <a href="https://doc.rust-lang.org/book/" class="url_header" rel="noreferrer">
                  <div class="url_wrapper"><span class="url_i1">doc.rust-lang.org</span></div>
                </a>
                <h3><a>The Rust Book</a></h3>
                <p class="content empty_element">This site did not provide any description.</p>
              </article>
            </div>
        "#;

        let results = parse_searxng_html(html, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Install Rust");
        assert_eq!(results[0].url, "https://rust-lang.org/tools/install");
        // Whitespace inside the snippet is collapsed onto one line.
        assert_eq!(results[0].snippet, "Rustup is the recommended installer.");
        // With no href on the <h3> link, the URL comes from the a.url_header wrapper.
        assert_eq!(results[1].url, "https://doc.rust-lang.org/book/");

        // max_results truncates.
        assert_eq!(parse_searxng_html(html, 1).len(), 1);
    }
}

// MCP Server with tools
#[derive(Clone)]
struct MyMcpServer {
    tool_router: ToolRouter<Self>,
    llm_client: LMStudioClient,
    // LRU cache of the last ~10 fetched pages, keyed by (upgraded) URL. Shared across
    // clones via Arc so web_fetch and web_get (and the REPL) hit the same cache.
    cache: Arc<Mutex<LruCache<String, FetchResult>>>,
    // Base URL of a SearxNG instance (--searxng / SEARXNG_URL). When set, web_search goes
    // through it instead of DuckDuckGo.
    searxng: Option<String>,
}

#[tool_router]
impl MyMcpServer {
    fn new(llm_client: LMStudioClient, expose_web_fetch: bool, searxng: Option<String>) -> Self {
        let mut tool_router = Self::tool_router();
        // web_get (raw page text) is always available. web_fetch additionally summarizes
        // via the LLM, so only expose it when we know where the LLM lives.
        if !expose_web_fetch {
            tool_router.remove_route("web_fetch");
        }
        Self {
            tool_router,
            llm_client,
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(10).unwrap()))),
            searxng,
        }
    }

    #[tool(description = "Use for exact arithmetic instead of computing in your head — reach for this whenever a task involves numbers to add, subtract, multiply, or divide, especially with large values or decimals, or when a wrong result would be costly. Performs one basic arithmetic operation (add, subtract, multiply, divide) on two numbers.")]
    async fn calculator(&self, params: Parameters<CalculatorParams>) -> Result<CallToolResult, McpError> {
        let operation = params.0.operation.as_str();
        let a = params.0.a;
        let b = params.0.b;

        let result = match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Ok(CallToolResult::error(vec![ContentBlock::text("Division by zero")]));
                }
                a / b
            }
            _ => return Ok(CallToolResult::error(vec![ContentBlock::text("Unknown operation")])),
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(format!("Result: {}", result))]))
    }

    #[tool(description = "Call this whenever anything depends on today's date — you have no reliable internal clock and your best guess is usually stale or wrong, so never assume the date. Use it before answering 'what is today', resolving relative dates ('yesterday', 'next Monday', 'in two weeks'), computing ages, durations or deadlines, or naming files and logs by date. Returns the current local date in YYYY-MM-DD format (e.g. 2026-06-22).")]
    async fn get_date(&self) -> Result<CallToolResult, McpError> {
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        Ok(CallToolResult::success(vec![ContentBlock::text(date)]))
    }

    #[tool(description = "Call this whenever you need the current wall-clock time or timezone — you cannot know it on your own and must not guess. Use it for 'what time is it', scheduling, measuring how long something takes, timestamping, or converting between time zones. Returns the current local time in HH:MM:SS format with UTC offset (e.g. 14:30:05 +03:00).")]
    async fn get_time(&self) -> Result<CallToolResult, McpError> {
        let time = chrono::Local::now().format("%H:%M:%S %:z").to_string();
        Ok(CallToolResult::success(vec![ContentBlock::text(time)]))
    }

    #[tool(description = "Use whenever you need to see what a file actually contains rather than assume — inspecting source before changing it, checking config or data, or quoting exact text. Always Read a file before you Edit or Write it. Reads files from the filesystem. Requires absolute path. Reads up to 2000 lines (use offset/limit for large files). Lines >2000 chars truncated.")]
    async fn read(&self, params: Parameters<ReadParams>) -> Result<CallToolResult, McpError> {
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

        let reader = BufReader::new(file);
        let offset = params.0.offset.unwrap_or(0);
        let limit = params.0.limit.unwrap_or(2000);

        let mut output = String::new();
        let mut line_num = 1;

        for line_result in reader.lines() {
            let line = line_result.map_err(|e| {
                McpError::internal_error(format!("Failed to read line: {}", e), None)
            })?;

            if line_num > offset && line_num <= offset + limit {
                let truncated = if line.len() > 2000 {
                    format!("{}...[truncated]", &line[..2000])
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

        Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
    }

    #[tool(description = "Use to create a brand-new file or completely replace an existing file's contents; to change only part of a file, prefer Edit instead. Writes a file to the filesystem, overwriting existing files (Read first for existing files). Prefer editing over creating new files. Don't proactively create docs/READMEs. No emojis unless requested.")]
    async fn write_file(&self, params: Parameters<WriteParams>) -> Result<CallToolResult, McpError> {
        let file_path = PathBuf::from(&params.0.file_path);

        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                McpError::internal_error(format!("Failed to create directories: {}", e), None)
            })?;
        }

        fs::write(&file_path, &params.0.content).map_err(|e| {
            McpError::internal_error(format!("Failed to write file: {}", e), None)
        })?;

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Successfully wrote {} bytes to {}",
            params.0.content.len(),
            params.0.file_path
        ))]))
    }

    #[tool(description = "Use to change part of an existing file — fixing a line, renaming a symbol, or adjusting a value — without rewriting the whole file. Performs exact string replacement. Must Read file first. Preserve exact indentation of lines. Prefer editing over creating new files. No emojis unless requested. old_string must be unique or use replace_all. Use replace_all for renaming variables.")]
    async fn edit(&self, params: Parameters<EditParams>) -> Result<CallToolResult, McpError> {
        let file_path = PathBuf::from(&params.0.file_path);

        if !file_path.exists() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "File not found: {}",
                params.0.file_path
            ))]));
        }

        let content = fs::read_to_string(&file_path).map_err(|e| {
            McpError::internal_error(format!("Failed to read file: {}", e), None)
        })?;

        let new_content = if params.0.replace_all {
            content.replace(&params.0.old_string, &params.0.new_string)
        } else {
            if let Some(pos) = content.find(&params.0.old_string) {
                let mut result = String::with_capacity(content.len());
                result.push_str(&content[..pos]);
                result.push_str(&params.0.new_string);
                result.push_str(&content[pos + params.0.old_string.len()..]);
                result
            } else {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "String not found in file: '{}'",
                    params.0.old_string
                ))]));
            }
        };

        if content == new_content {
            return Ok(CallToolResult::error(vec![ContentBlock::text(
                "No changes made - old_string not found or identical to new_string".to_string()
            )]));
        }

        fs::write(&file_path, &new_content).map_err(|e| {
            McpError::internal_error(format!("Failed to write file: {}", e), None)
        })?;

        let count = if params.0.replace_all {
            content.matches(&params.0.old_string).count()
        } else {
            1
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Successfully replaced {} occurrence(s) in {}",
            count, params.0.file_path
        ))]))
    }

    #[tool(description = "Use when you know how a file is named or where it lives but not its exact path — locating files by name or extension across a tree. Fast file pattern matching that supports glob patterns (\"**/*.js\"). Returns paths sorted by modification time. Batch multiple searches when useful.")]
    async fn glob_search(&self, params: Parameters<GlobParams>) -> Result<CallToolResult, McpError> {
        let base_path = params.0.path.as_deref().unwrap_or(".");
        let pattern = if params.0.path.is_some() {
            format!("{}/{}", base_path, params.0.pattern)
        } else {
            params.0.pattern.clone()
        };

        let mut matches: Vec<PathBuf> = Vec::new();

        for entry in glob::glob(&pattern).map_err(|e| {
            McpError::invalid_params(format!("Invalid glob pattern: {}", e), None)
        })? {
            match entry {
                Ok(path) => matches.push(path),
                Err(e) => eprintln!("Error reading glob entry: {}", e),
            }
        }

        matches.sort_by(|a, b| {
            let a_time = fs::metadata(a).and_then(|m| m.modified()).ok();
            let b_time = fs::metadata(b).and_then(|m| m.modified()).ok();
            b_time.cmp(&a_time)
        });

        let output = if matches.is_empty() {
            format!("No files found matching pattern: {}", params.0.pattern)
        } else {
            let paths: Vec<String> = matches
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            format!("Found {} file(s):\n{}", paths.len(), paths.join("\n"))
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
    }

    #[tool(description = "Use when you need to find where something appears in the code — a function, variable, string literal, or any regex pattern — across many files at once. Searches file contents using regex. Filter by glob (\"*.js\") or type (\"js\", \"py\"). Output modes: content (lines), files_with_matches (paths, default), count. Escape literal braces. Use multiline:true for cross-line patterns.")]
    async fn grep(&self, params: Parameters<GrepParams>) -> Result<CallToolResult, McpError> {
        let pattern = params.0.pattern.clone();
        let case_insensitive = params.0.case_insensitive.unwrap_or(false);
        let output_mode = params.0.output_mode.as_deref().unwrap_or("files_with_matches");
        let head_limit = params.0.head_limit;

        let regex = if case_insensitive {
            regex::RegexBuilder::new(&pattern)
                .case_insensitive(true)
                .build()
        } else {
            regex::Regex::new(&pattern)
        }.map_err(|e| {
            McpError::invalid_params(format!("Invalid regex pattern: {}", e), None)
        })?;

        let search_path = params.0.path.as_deref().unwrap_or(".");
        let mut results = Vec::new();
        let mut count = 0;

        let walker = walkdir::WalkDir::new(search_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file());

        for entry in walker {
            if let Ok(metadata) = entry.metadata() {
                if metadata.len() > 10_000_000 {
                    continue;
                }
            }

            if let Ok(content) = fs::read_to_string(entry.path()) {
                let mut file_matches = Vec::new();

                for (line_num, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        file_matches.push((line_num + 1, line.to_string()));
                    }
                }

                if !file_matches.is_empty() {
                    let path_str = entry.path().display().to_string();

                    match output_mode {
                        "content" => {
                            for (line_num, line) in file_matches {
                                results.push(format!("{}:{}:{}", path_str, line_num, line));
                                count += 1;
                                if let Some(limit) = head_limit {
                                    if count >= limit {
                                        break;
                                    }
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

                    if let Some(limit) = head_limit {
                        if count >= limit {
                            break;
                        }
                    }
                }
            }
        }

        let output = if results.is_empty() {
            format!("No matches found for pattern: {}", pattern)
        } else {
            results.join("\n")
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
    }

    // Shared helper: fetch a URL and return its cleaned, size-limited text. Used by both
    // web_fetch (which then summarizes via the LLM) and web_get (which returns text directly).
    async fn fetch_clean_text(&self, url: &str) -> Result<FetchResult, McpError> {
        // Validate URL
        let parsed_url = url::Url::parse(url).map_err(|e| {
            McpError::invalid_params(format!("Invalid URL: {}", e), None)
        })?;

        // Upgrade HTTP to HTTPS if needed
        let url_string = if parsed_url.scheme() == "http" {
            url.replace("http://", "https://")
        } else {
            url.to_string()
        };

        // Serve repeat fetches of the same page from the LRU cache. The lock is dropped
        // before any .await below, so it is never held across an await point.
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(hit) = cache.get(&url_string) {
                return Ok(hit.clone());
            }
        }

        // Fetch the content
        let client = browser_client()?;

        let response = client
            .get(&url_string)
            .send()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to fetch URL: {}", e), None)
            })?;

        // Check if we got redirected to a different host
        let final_url = response.url().to_string();
        if response.url().host_str() != parsed_url.host_str() {
            let result = FetchResult::Redirect(format!(
                "URL redirected to different host: {}\nPlease request this URL again if you want to fetch it.",
                final_url
            ));
            self.cache.lock().unwrap().put(url_string.clone(), result.clone());
            return Ok(result);
        }

        // Get the response body
        let html = response
            .text()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to read response body: {}", e), None)
            })?;

        // Convert HTML to plain text
        let text_content = html2text::from_read(html.as_bytes(), 100)
            .map_err(|e| {
                McpError::internal_error(format!("Failed to convert HTML to text: {}", e), None)
            })?;

        // Trim and clean up the text
        let cleaned_text = text_content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        // Limit size to avoid context overflow
        const MAX_LENGTH: usize = 65536;
        let text = if cleaned_text.len() > MAX_LENGTH {
            cleaned_text[..MAX_LENGTH].to_string()
        } else {
            cleaned_text
        };

        let result = FetchResult::Text { final_url, text };
        self.cache.lock().unwrap().put(url_string.clone(), result.clone());
        Ok(result)
    }

    #[tool(description = "Use when you have a specific URL and want a focused answer pulled from it rather than the whole page — the page is fetched, converted to readable text, and an assistant extracts just what your prompt asks for. Takes a URL and a prompt describing what information to extract. Use web_get instead when you want the full raw text yourself.")]
    async fn web_fetch(&self, params: Parameters<WebFetchParams>) -> Result<CallToolResult, McpError> {
        let (final_url, truncated_text) = match self.fetch_clean_text(&params.0.url).await? {
            FetchResult::Redirect(message) => {
                return Ok(CallToolResult::success(vec![ContentBlock::text(message)]));
            }
            FetchResult::Text { final_url, text } => (final_url, text),
        };

        // Use LLM to extract information based on the prompt
        let system_prompt = "You are a helpful assistant that extracts specific information from web page content. \
                            Be concise and focus only on answering the user's query based on the provided content.";

        let user_prompt = format!(
            "Web page URL: {}\n\nContent:\n{}\n\n---\n\nTask: {}",
            final_url, truncated_text, params.0.prompt
        );

        let extracted_info = self.llm_client
            .simple_completion(system_prompt, &user_prompt)
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to extract information with LLM: {}", e), None)
            })?;

        let result_text = format!(
            "URL: {}\n\n=== Extracted Information ===\n\n{}",
            final_url, extracted_info
        );

        Ok(CallToolResult::success(vec![ContentBlock::text(result_text)]))
    }

    #[tool(description = "Use when you have a URL and want to read or process the full page yourself, with no filtering — fetches the page and returns the cleaned text directly (no summarization). Choose web_fetch instead when you only need a specific fact and want it extracted for you.")]
    async fn web_get(&self, params: Parameters<WebGetParams>) -> Result<CallToolResult, McpError> {
        let (final_url, text) = match self.fetch_clean_text(&params.0.url).await? {
            FetchResult::Redirect(message) => {
                return Ok(CallToolResult::success(vec![ContentBlock::text(message)]));
            }
            FetchResult::Text { final_url, text } => (final_url, text),
        };

        let result_text = format!(
            "URL: {}\n\n=== Page Content ===\n\n{}",
            final_url, text
        );

        Ok(CallToolResult::success(vec![ContentBlock::text(result_text)]))
    }

    #[tool(description = "Use when you need information you don't have or that may be newer than your knowledge, or when you don't yet have a URL to open — search the web by keywords to discover relevant pages. Searches using DuckDuckGo (or a configured SearxNG instance) and returns results with titles, URLs, and snippets; follow up with web_fetch or web_get on a promising URL.")]
    async fn web_search(&self, params: Parameters<WebSearchParams>) -> Result<CallToolResult, McpError> {
        let query = params.0.query.clone();
        let max_results = params.0.max_results.unwrap_or(10);

        // A configured SearxNG instance replaces DuckDuckGo entirely.
        if let Some(base_url) = &self.searxng {
            return searxng_search(base_url, &query, max_results).await;
        }

        // URL encode the query
        let encoded_query = urlencoding::encode(&query);
        let search_url = format!("https://html.duckduckgo.com/html/?q={}", encoded_query);

        // Fetch search results. browser_client() already carries the Firefox header set;
        // this request additionally looks like the search form on duckduckgo.com was
        // submitted, so the Referer and Sec-Fetch-Site have to agree with each other.
        let client = browser_client()?;

        let response = client
            .get(&search_url)
            .header(reqwest::header::REFERER, "https://duckduckgo.com/")
            .header("Sec-Fetch-Site", "same-site")
            .send()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to fetch search results: {}", e), None)
            })?;

        // DuckDuckGo answers requests it flags as automated with an HTTP 202 anti-bot
        // challenge page (an "anomaly-modal" CAPTCHA) instead of results. Detect it and
        // report clearly, otherwise the empty parse below falls through to a misleading
        // "No search results found" message.
        let status = response.status();

        let html = response
            .text()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to read response body: {}", e), None)
            })?;

        if status == reqwest::StatusCode::ACCEPTED || html.contains("anomaly-modal") {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "DuckDuckGo blocked this search with a bot-detection challenge (HTTP {}). \
                 Web search is temporarily unavailable — please try again later or configure \
                 an alternative search provider.",
                status.as_u16()
            ))]));
        }

        // Parse HTML
        use scraper::{Html, Selector};

        let document = Html::parse_document(&html);

        // DuckDuckGo HTML result selectors
        let result_selector = Selector::parse(".result").unwrap();
        let title_selector = Selector::parse(".result__a").unwrap();
        let snippet_selector = Selector::parse(".result__snippet").unwrap();
        let url_selector = Selector::parse(".result__url").unwrap();

        let mut results = Vec::new();

        for (index, result) in document.select(&result_selector).enumerate() {
            if index >= max_results {
                break;
            }

            let title = result
                .select(&title_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            let snippet = result
                .select(&snippet_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            let url = result
                .select(&url_selector)
                .next()
                .map(|e| resolve_result_url(e.value().attr("href"), &e.text().collect::<String>()))
                .filter(|u| !u.is_empty())
                // Some layouts only carry the target on the title link.
                .or_else(|| {
                    result
                        .select(&title_selector)
                        .next()
                        .map(|e| resolve_result_url(e.value().attr("href"), ""))
                })
                .unwrap_or_default();

            if !title.is_empty() {
                results.push(SearchResult { title, url, snippet });
            }
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(
            format_search_results(&query, &results),
        )]))
    }

    #[tool(description = "Use when you need to see what a directory actually contains — discovering files and subfolders, confirming a path exists, or checking a parent directory before writing into it. Lists directory contents with file/folder information. Prefer Glob/Grep when you already know what you're looking for.")]
    async fn ls(&self, params: Parameters<LSParams>) -> Result<CallToolResult, McpError> {
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

    #[tool(description = "Use to run shell commands and CLIs when no dedicated tool fits — building, testing, running scripts, or git/gh operations. Execute a shell command (cmd on Windows, sh elsewhere). Verify parent dirs with LS before creating files. Quote paths with spaces. Timeout: 120s default, 600s max. AVOID: find/grep (use Grep/Glob), cat/head/tail/ls (use Read/LS). Chain with && (avoid ;, which does not chain in Windows cmd), no newlines. Prefer absolute paths over cd.\n\nGit commits: 1) Run parallel: git status/diff/log 2) Draft message (why not what), check secrets 3) Add files, commit using repeated -m flags for multi-line messages (e.g. git commit -m \"summary\" -m \"details\"), check status 4) Retry once if pre-commit modifies files. Never update config or use -i flag. Don't push unless requested.\n\nPull requests: 1) Parallel: git status/diff, check remote, git log 2) Analyze all commits, draft summary 3) Create branch, push, gh pr create with the body passed via --body-file (a temp file), since heredocs don't work in cmd.\n\nGitHub: Use gh for issues/PRs/checks/releases.")]
    async fn run_command(&self, params: Parameters<RunCommandParams>) -> Result<CallToolResult, McpError> {
        let command = params.0.command.clone();
        let timeout_secs = params.0.timeout.unwrap_or(120).min(600);

        use tokio::process::Command;
        use std::time::Duration;

        // Determine shell based on platform
        let (shell, shell_arg) = if cfg!(target_os = "windows") {
            ("cmd", "/C")
        } else {
            ("sh", "-c")
        };

        let child = Command::new(shell)
            .arg(shell_arg)
            .arg(&command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                McpError::internal_error(format!("Failed to spawn command: {}", e), None)
            })?;

        // Wait with timeout
        let output_result = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            child.wait_with_output()
        ).await;

        match output_result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                let mut result_text = String::new();

                if !stdout.is_empty() {
                    result_text.push_str(&stdout);
                }

                if !stderr.is_empty() {
                    if !result_text.is_empty() {
                        result_text.push_str("\n--- stderr ---\n");
                    }
                    result_text.push_str(&stderr);
                }

                if result_text.is_empty() {
                    result_text = format!("Command completed successfully (exit code: {})", output.status.code().unwrap_or(0));
                }

                // Truncate if too long
                const MAX_OUTPUT: usize = 30000;
                if result_text.len() > MAX_OUTPUT {
                    result_text = format!(
                        "{}\n\n[Output truncated - {} of {} characters shown]",
                        &result_text[..MAX_OUTPUT],
                        MAX_OUTPUT,
                        result_text.len()
                    );
                }

                if output.status.success() {
                    Ok(CallToolResult::success(vec![ContentBlock::text(result_text)]))
                } else {
                    Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                        "Command failed with exit code: {}\n\n{}",
                        output.status.code().unwrap_or(-1),
                        result_text
                    ))]))
                }
            }
            Ok(Err(e)) => {
                Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Failed to execute command: {}",
                    e
                ))]))
            }
            Err(_) => {
                // Timeout - the child is already consumed, just return error
                Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Command timed out after {} seconds",
                    timeout_secs
                ))]))
            }
        }
    }

    #[tool(description = "Before adding or upgrading a dependency, call this to get ground truth from crates.io instead of guessing version numbers or feature flags from memory — which is frequently stale or wrong and leads to Cargo.toml that won't resolve. Given a crate name, returns its description, the latest stable version, recent versions (yanked ones flagged), the exact feature flags you can enable, license, minimum supported Rust version (MSRV), useful links (docs.rs / repository / homepage), and a ready-to-paste `name = \"version\"` line. Use it whenever you write a [dependencies] entry, pick a version, or need to know which features to turn on.")]
    async fn crate_info(&self, params: Parameters<CrateInfoParams>) -> Result<CallToolResult, McpError> {
        let name = params.0.name.trim().to_string();
        if name.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text("Crate name must not be empty")]));
        }
        let requested_version = params.0.version.clone();

        let client = rust_api_client()?;
        let url = format!("{}/crates/{}", CRATES_IO_API, name);
        let resp = client.get(&url).send().await.map_err(|e| {
            McpError::internal_error(format!("Failed to query crates.io: {}", e), None)
        })?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Crate '{}' not found on crates.io. Check the spelling, or use crate_search to discover the right name.",
                name
            ))]));
        }
        if !resp.status().is_success() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "crates.io returned HTTP {} for crate '{}'.",
                resp.status().as_u16(),
                name
            ))]));
        }

        let body: Value = resp.json().await.map_err(|e| {
            McpError::internal_error(format!("Failed to parse crates.io response: {}", e), None)
        })?;

        let c = &body["crate"];
        let versions = body["versions"].as_array().cloned().unwrap_or_default();

        // Pick which version to detail: the requested one, else latest stable, else newest.
        let target_version = requested_version.clone()
            .or_else(|| c["max_stable_version"].as_str().map(|s| s.to_string()))
            .or_else(|| c["newest_version"].as_str().map(|s| s.to_string()))
            .or_else(|| c["max_version"].as_str().map(|s| s.to_string()));
        let version_obj = target_version.as_ref().and_then(|tv| {
            versions.iter().find(|v| v["num"].as_str() == Some(tv.as_str()))
        });

        let mut out = String::new();
        out.push_str(&format!("Crate: {}\n", c["name"].as_str().unwrap_or(&name)));
        if let Some(desc) = c["description"].as_str() {
            if !desc.trim().is_empty() {
                out.push_str(&format!("Description: {}\n", desc.trim()));
            }
        }
        out.push_str(&format!("Latest stable: {}\n", c["max_stable_version"].as_str().unwrap_or("-")));
        if c["newest_version"].as_str() != c["max_stable_version"].as_str() {
            out.push_str(&format!("Newest (incl. pre-release): {}\n", c["newest_version"].as_str().unwrap_or("-")));
        }
        out.push_str(&format!("Total downloads: {}\n", c["downloads"].as_u64().unwrap_or(0)));

        for (label, key) in [("Documentation", "documentation"), ("Repository", "repository"), ("Homepage", "homepage")] {
            if let Some(v) = c[key].as_str() {
                if !v.is_empty() {
                    out.push_str(&format!("{}: {}\n", label, v));
                }
            }
        }
        out.push_str(&format!("docs.rs: {}/{}\n", DOCS_RS, name));

        if let Some(tv) = &target_version {
            out.push_str(&format!("\n=== Version {} ===\n", tv));
            if let Some(vo) = version_obj {
                if let Some(lic) = vo["license"].as_str() {
                    if !lic.is_empty() {
                        out.push_str(&format!("License: {}\n", lic));
                    }
                }
                if let Some(msrv) = vo["rust_version"].as_str() {
                    out.push_str(&format!("Minimum Rust (MSRV): {}\n", msrv));
                }
                if vo["yanked"].as_bool() == Some(true) {
                    out.push_str("⚠ This version is YANKED.\n");
                }
                if let Some(feats) = vo["features"].as_object() {
                    if feats.is_empty() {
                        out.push_str("Features: (none)\n");
                    } else {
                        out.push_str(&format!("Features ({}):\n", feats.len()));
                        let mut names: Vec<&String> = feats.keys().collect();
                        names.sort();
                        for fname in names {
                            let subs = feats[fname].as_array()
                                .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", "))
                                .unwrap_or_default();
                            if subs.is_empty() {
                                out.push_str(&format!("  - {}\n", fname));
                            } else {
                                out.push_str(&format!("  - {} = [{}]\n", fname, subs));
                            }
                        }
                    }
                }
            } else if requested_version.is_some() {
                out.push_str("(Requested version was not found in the published version list.)\n");
            }
        }

        out.push_str("\nRecent versions:\n");
        for v in versions.iter().take(10) {
            let num = v["num"].as_str().unwrap_or("?");
            let yanked = if v["yanked"].as_bool() == Some(true) { " (yanked)" } else { "" };
            out.push_str(&format!("  {}{}\n", num, yanked));
        }

        if let Some(stable) = c["max_stable_version"].as_str() {
            out.push_str(&format!("\nCargo.toml: {} = \"{}\"\n", name, stable));
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(out)]))
    }

    #[tool(description = "When you know what a crate should DO but not its exact name, search crates.io by keywords instead of inventing a plausible-sounding crate that may not exist. Returns the top matching crates with their latest version, download counts (a good popularity/quality signal) and one-line descriptions. Use it to discover the right dependency, then follow up with crate_info for versions/features and crate_api for the exact public API signatures.")]
    async fn crate_search(&self, params: Parameters<CrateSearchParams>) -> Result<CallToolResult, McpError> {
        let query = params.0.query.trim().to_string();
        if query.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text("Search query must not be empty")]));
        }
        let limit = params.0.limit.unwrap_or(15).clamp(1, 50);

        let client = rust_api_client()?;
        let url = format!(
            "{}/crates?q={}&per_page={}&sort=relevance",
            CRATES_IO_API,
            urlencoding::encode(&query),
            limit
        );
        let resp = client.get(&url).send().await.map_err(|e| {
            McpError::internal_error(format!("Failed to query crates.io: {}", e), None)
        })?;
        if !resp.status().is_success() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "crates.io search returned HTTP {}.",
                resp.status().as_u16()
            ))]));
        }

        let body: Value = resp.json().await.map_err(|e| {
            McpError::internal_error(format!("Failed to parse crates.io response: {}", e), None)
        })?;

        let crates = body["crates"].as_array().cloned().unwrap_or_default();
        let total = body["meta"]["total"].as_u64().unwrap_or(crates.len() as u64);

        if crates.is_empty() {
            return Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "No crates found for query '{}'.",
                query
            ))]));
        }

        let mut out = String::new();
        out.push_str(&format!(
            "Found {} crate(s) for '{}' (showing {}):\n\n",
            total,
            query,
            crates.len()
        ));
        for cr in &crates {
            let cname = cr["name"].as_str().unwrap_or("?");
            let vers = cr["max_stable_version"].as_str()
                .or_else(|| cr["max_version"].as_str())
                .unwrap_or("?");
            let downloads = cr["downloads"].as_u64().unwrap_or(0);
            let desc = cr["description"].as_str().unwrap_or("").trim();
            out.push_str(&format!("{} v{}  ({} downloads)\n", cname, vers, downloads));
            if !desc.is_empty() {
                out.push_str(&format!("    {}\n", desc));
            }
        }
        out.push_str("\nUse crate_info for versions/features, or crate_api for the public API signatures.");

        Ok(CallToolResult::success(vec![ContentBlock::text(out)]))
    }

    #[tool(description = "This is your antidote to hallucinated APIs. Before you write `use` statements, call a function, construct a type, or implement a trait from an external crate, call this to fetch the crate's REAL public API — exact function signatures, struct/enum definitions, trait methods and their bounds — parsed straight from the official rustdoc JSON on docs.rs. Pass a crate name (and optionally a version, default 'latest', and an `item` substring like 'Serialize' or 'Client::new' to zoom in on one item). Large crates expose a lot of API, so use the `item` filter to stay focused. If you're unsure whether a method or path exists, check here first instead of guessing. Note: docs.rs only has rustdoc JSON for builds since 2025-05-23, so very old versions may be unavailable.")]
    async fn crate_api(&self, params: Parameters<CrateApiParams>) -> Result<CallToolResult, McpError> {
        use std::io::Read;

        let name = params.0.name.trim().to_string();
        if name.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text("Crate name must not be empty")]));
        }
        let version = params.0.version.clone().unwrap_or_else(|| "latest".to_string());
        let item_filter = params.0.item.clone();

        let client = rust_api_client()?;
        let url = format!("{}/crate/{}/{}/json.gz", DOCS_RS, name, version);
        let resp = client.get(&url).send().await.map_err(|e| {
            McpError::internal_error(format!("Failed to download rustdoc JSON: {}", e), None)
        })?;

        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "No rustdoc JSON available for '{}' {}. docs.rs only builds rustdoc JSON for successful builds since 2025-05-23, so older or failed-build versions have none. Try 'latest' or a newer version.",
                name, version
            ))]));
        }
        if !status.is_success() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "docs.rs returned HTTP {} for '{}' {}.",
                status.as_u16(), name, version
            ))]));
        }

        let compressed = resp.bytes().await.map_err(|e| {
            McpError::internal_error(format!("Failed to read rustdoc JSON body: {}", e), None)
        })?;

        // The `.gz` resource is a gzip file (not transfer-encoding), so decompress it ourselves.
        let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
        let mut json_bytes = Vec::new();
        decoder.read_to_end(&mut json_bytes).map_err(|e| {
            McpError::internal_error(format!("Failed to decompress rustdoc JSON: {}", e), None)
        })?;

        // public-api parses from a file path, so stage the JSON in a temp file.
        let safe = |s: &str| s.replace(['/', '\\', ':'], "_");
        let tmp_path = std::env::temp_dir().join(format!(
            "haico-rustdoc-{}-{}-{}.json",
            safe(&name), safe(&version), std::process::id()
        ));
        std::fs::write(&tmp_path, &json_bytes).map_err(|e| {
            McpError::internal_error(format!("Failed to write temp file: {}", e), None)
        })?;

        // Strip the universal blanket impls (Into/TryFrom/Borrow/Any/ToOwned/…) and auto-trait
        // impls (Send/Sync/Unpin/Freeze/…) that repeat identically for every type and drown out
        // the crate's real API. Auto-derived impls (Clone/Debug/Default/…) are kept — they carry
        // useful signal about a type and aren't per-type noise.
        let build_result = public_api::Builder::from_rustdoc_json(&tmp_path)
            .omit_blanket_impls(true)
            .omit_auto_trait_impls(true)
            .build();
        let _ = std::fs::remove_file(&tmp_path);

        let public_api = match build_result {
            Ok(api) => api,
            Err(e) => {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Could not parse the rustdoc JSON for '{}' {} (likely a rustdoc-json format version this tool doesn't support yet): {}",
                    name, version, e
                ))]));
            }
        };

        let all_lines: Vec<String> = public_api.items().map(|it| it.to_string()).collect();
        let total_items = all_lines.len();

        let lines: Vec<String> = match &item_filter {
            Some(filter) => all_lines.into_iter().filter(|l| l.contains(filter.as_str())).collect(),
            None => all_lines,
        };

        if let Some(filter) = &item_filter {
            if lines.is_empty() {
                return Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                    "No public API items in '{}' {} match '{}'. The crate exposes {} public items total; try a different substring or omit the filter.",
                    name, version, filter, total_items
                ))]));
            }
        }

        let shown = lines.len();
        let mut out = match &item_filter {
            Some(f) => format!("Public API of '{}' {} matching '{}' ({} of {} items):\n\n", name, version, f, shown, total_items),
            None => format!("Public API of '{}' {} ({} items):\n\n", name, version, total_items),
        };

        let joined = lines.join("\n");
        const MAX_LENGTH: usize = 60000;
        if joined.len() > MAX_LENGTH {
            let mut truncated = joined[..MAX_LENGTH].to_string();
            if let Some(pos) = truncated.rfind('\n') {
                truncated.truncate(pos);
            }
            let kept = truncated.lines().count();
            out.push_str(&truncated);
            out.push_str(&format!(
                "\n\n… output truncated ({} of {} lines shown). Pass an 'item' filter to narrow the results.",
                kept, shown
            ));
        } else {
            out.push_str(&joined);
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(out)]))
    }

    // Helper to execute tools by name
    async fn execute_tool(&self, name: &str, arguments: Value) -> Result<String> {
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
            "ls" => {
                let params: LSParams = serde_json::from_value(arguments)?;
                self.ls(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "run_command" => {
                let params: RunCommandParams = serde_json::from_value(arguments)?;
                self.run_command(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "crate_info" => {
                let params: CrateInfoParams = serde_json::from_value(arguments)?;
                self.crate_info(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "crate_search" => {
                let params: CrateSearchParams = serde_json::from_value(arguments)?;
                self.crate_search(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "crate_api" => {
                let params: CrateApiParams = serde_json::from_value(arguments)?;
                self.crate_api(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            _ => return Err(anyhow::anyhow!("Unknown tool: {}", name)),
        };

        // Extract text from result
        let text = result
            .content
            .iter()
            .filter_map(|c| {
                c.as_text().map(|t| t.text.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(text)
    }
}

// MCP server handler: wires the existing tool_router to handle list_tools/call_tool.
// The macro auto-generates a tools-enabled get_info() (server identity comes from Cargo.toml).
#[tool_handler(router = self.tool_router, name = "haico")]
impl ServerHandler for MyMcpServer {}

// Convert MCP tools to OpenAI tool format
fn mcp_tools_to_openai_format(server: &MyMcpServer) -> Vec<Value> {
    let tools = server.tool_router.list_all();
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description.as_deref().unwrap_or(""),
                    "parameters": tool.input_schema
                }
            })
        })
        .collect()
}

// Fetch available models from LLM server
async fn fetch_available_models(base_url: &str, token: Option<&str>) -> Result<Vec<String>> {
    let client = reqwest::Client::new();
    let request = client.get(format!("{}/v1/models", normalize_base_url(base_url)));
    let request = match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    };

    let response = request
        .send()
        .await
        .context("Failed to fetch models from server")?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        bail!("{}", unauthorized_message(token.is_some()));
    }

    let models_response: ModelsResponse = response
        .json()
        .await
        .context("Failed to parse models response")?;

    let model_ids: Vec<String> = models_response
        .data
        .into_iter()
        .map(|m| m.id)
        .collect();

    Ok(model_ids)
}

// Analyze text to determine if it contains Markdown formatting
fn is_markdown(text: &str) -> bool {
    let mut markdown_score = 0;
    let lines: Vec<&str> = text.lines().collect();

    if lines.is_empty() {
        return false;
    }

    for line in &lines {
        let trimmed = line.trim();

        // Check for headers (lines starting with #)
        if trimmed.starts_with('#') && trimmed.len() > 1 {
            markdown_score += 2;
        }

        // Check for unordered list items (lines starting with -)
        if trimmed.starts_with('-') && trimmed.len() > 1 {
            markdown_score += 1;
        }

        // Check for unordered list items (lines starting with *)
        if trimmed.starts_with('*') && trimmed.len() > 1 && !trimmed.starts_with("**") {
            markdown_score += 1;
        }

        // Check for bold text markers (**text**)
        if line.contains("**") {
            // Count pairs of **
            let count = line.matches("**").count();
            if count >= 2 {
                markdown_score += 1;
            }
        }

        // Check for italic markers (*text* or _text_) - only if they appear in pairs and not part of snake_case
        // Count underscores only if surrounded by non-word chars to avoid snake_case false positives
        let underscore_count = line.matches('_').count();
        if underscore_count >= 2 {
            // Check if they look like markdown (not snake_case)
            // Simple heuristic: if there's whitespace around them, they're likely markdown
            if line.contains(" _") || line.contains("_ ") {
                markdown_score += 1;
            }
        }

        // Check for code blocks (```)
        if trimmed.starts_with("```") {
            markdown_score += 3;
        }

        // Check for inline code (`text`)
        if line.contains('`') && !trimmed.starts_with("```") {
            markdown_score += 1;
        }

        // Check for links ([text](url))
        if line.contains("[") && line.contains("](") {
            markdown_score += 2;
        }

        // Check for ordered list (1. text)
        if trimmed.len() > 2 {
            let chars: Vec<char> = trimmed.chars().collect();
            if chars[0].is_numeric() && chars.len() > 1 && chars[1] == '.' {
                markdown_score += 1;
            }
        }
    }

    // Determine threshold: if we have enough Markdown indicators relative to text length
    // Consider it Markdown if score is at least 3, or if score/lines ratio is > 0.2
    let ratio = markdown_score as f64 / lines.len() as f64;
    markdown_score >= 3 || ratio > 0.2
}

// Render Markdown text with colors and formatting for terminal output
fn render_markdown(text: &str) -> Result<()> {
    let mut in_code_block = false;
    let mut code_lang = String::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Handle code blocks
        if trimmed.starts_with("```") {
            if in_code_block {
                // End of code block
                in_code_block = false;
                code_lang.clear();
            } else {
                // Start of code block
                in_code_block = true;
                code_lang = trimmed.trim_start_matches("```").to_string();
            }
            continue;
        }

        if in_code_block {
            // Render code with dark grey background
            execute!(io::stdout(), SetBackgroundColor(Color::Grey), SetForegroundColor(Color::Black))?;
            print!("  {}", line);
            execute!(io::stdout(), ResetColor)?;
            println!();
            continue;
        }

        // Handle headers
        if trimmed.starts_with("# ") {
            execute!(io::stdout(), SetForegroundColor(Color::Cyan))?;
            println!("\n{}", trimmed.trim_start_matches("# "));
            execute!(io::stdout(), ResetColor)?;
            continue;
        } else if trimmed.starts_with("## ") {
            execute!(io::stdout(), SetForegroundColor(Color::Cyan))?;
            println!("\n{}", trimmed.trim_start_matches("## "));
            execute!(io::stdout(), ResetColor)?;
            continue;
        } else if trimmed.starts_with("### ") {
            execute!(io::stdout(), SetForegroundColor(Color::Blue))?;
            println!("\n{}", trimmed.trim_start_matches("### "));
            execute!(io::stdout(), ResetColor)?;
            continue;
        }

        // Process inline formatting
        render_inline_markdown(line)?;
    }

    Ok(())
}

// Render a line with inline Markdown formatting (bold, italic, code, links)
fn render_inline_markdown(line: &str) -> Result<()> {
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check for inline code (`code`)
        if chars[i] == '`' {
            if let Some(end) = find_closing('`', &chars, i + 1) {
                execute!(io::stdout(), SetBackgroundColor(Color::Grey), SetForegroundColor(Color::Black))?;
                print!(" ");
                for j in (i + 1)..end {
                    print!("{}", chars[j]);
                }
                print!(" ");
                execute!(io::stdout(), ResetColor)?;
                i = end + 1;
                continue;
            }
        }

        // Check for bold (**text**)
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_closing("**", &chars, i + 2) {
                execute!(io::stdout(), SetAttribute(Attribute::Bold))?;
                for j in (i + 2)..end {
                    print!("{}", chars[j]);
                }
                execute!(io::stdout(), SetAttribute(Attribute::Reset))?;
                i = end + 2;
                continue;
            }
        }

        // Check for italic (*text* or _text_)
        if chars[i] == '*' || chars[i] == '_' {
            let marker = chars[i];
            if let Some(end) = find_closing(marker, &chars, i + 1) {
                // Make sure it's not part of **
                if marker == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
                    print!("{}", chars[i]);
                    i += 1;
                    continue;
                }
                execute!(io::stdout(), SetForegroundColor(Color::Green))?;
                for j in (i + 1)..end {
                    print!("{}", chars[j]);
                }
                execute!(io::stdout(), ResetColor)?;
                i = end + 1;
                continue;
            }
        }

        // Check for links ([text](url))
        if chars[i] == '[' {
            if let Some(close_bracket) = find_closing(']', &chars, i + 1) {
                if close_bracket + 1 < chars.len() && chars[close_bracket + 1] == '(' {
                    if let Some(close_paren) = find_closing(')', &chars, close_bracket + 2) {
                        // Render link text in blue
                        execute!(io::stdout(), SetForegroundColor(Color::Blue))?;
                        for j in (i + 1)..close_bracket {
                            print!("{}", chars[j]);
                        }
                        execute!(io::stdout(), ResetColor)?;

                        // Render URL in dark grey
                        execute!(io::stdout(), SetForegroundColor(Color::DarkGrey))?;
                        print!(" (");
                        for j in (close_bracket + 2)..close_paren {
                            print!("{}", chars[j]);
                        }
                        print!(")");
                        execute!(io::stdout(), ResetColor)?;

                        i = close_paren + 1;
                        continue;
                    }
                }
            }
        }

        // Regular character
        print!("{}", chars[i]);
        i += 1;
    }

    println!();
    Ok(())
}

// Find closing character from a starting position
fn find_closing(marker: char, chars: &[char], start: usize) -> Option<usize> {
    for i in start..chars.len() {
        if chars[i] == marker {
            return Some(i);
        }
    }
    None
}

// Find closing double-character marker (like **)
fn find_double_closing(marker: &str, chars: &[char], start: usize) -> Option<usize> {
    let marker_chars: Vec<char> = marker.chars().collect();
    if marker_chars.len() != 2 {
        return None;
    }

    for i in start..(chars.len() - 1) {
        if chars[i] == marker_chars[0] && chars[i + 1] == marker_chars[1] {
            return Some(i);
        }
    }
    None
}

// Render text (auto-detect Markdown and render accordingly)
fn render_text(text: &str) -> Result<()> {
    if is_markdown(text) {
        render_markdown(text)
    } else {
        println!("{}", text);
        Ok(())
    }
}

// Interactive model selection using arrow keys
fn select_model_interactive(models: Vec<String>) -> Result<String> {
    if models.is_empty() {
        return Err(anyhow::anyhow!("No models available"));
    }

    let mut selected_index = 0;

    println!("Select a model (use Up/Down arrows, Enter to confirm):\n");

    // Enable raw mode for key events
    enable_raw_mode()?;

    loop {
        // Clear screen and move cursor to top
        execute!(io::stdout(), cursor::MoveTo(0, 0), Clear(ClearType::All))?;

        println!("Select a model (use Up/Down arrows, Enter to confirm):\n");

        // Display models
        for (index, model) in models.iter().enumerate() {
            if index == selected_index {
                println!("> {} <", model);
            } else {
                println!("  {}  ", model);
            }
        }

        io::stdout().flush()?;

        // Read key event
        if let Event::Key(key_event) = event::read()? {
            // Only process key press events, ignore key release
            if key_event.kind != KeyEventKind::Press {
                continue;
            }

            match key_event.code {
                KeyCode::Up => {
                    if selected_index > 0 {
                        selected_index -= 1;
                    }
                }
                KeyCode::Down => {
                    if selected_index < models.len() - 1 {
                        selected_index += 1;
                    }
                }
                KeyCode::Enter => {
                    // Selection confirmed
                    break;
                }
                KeyCode::Esc => {
                    disable_raw_mode()?;
                    return Err(anyhow::anyhow!("Model selection cancelled"));
                }
                _ => {
                    // Ignore other keys
                }
            }
        }
    }

    disable_raw_mode()?;

    // Clear screen one final time and show selection
    execute!(io::stdout(), cursor::MoveTo(0, 0), Clear(ClearType::All))?;
    println!("Selected model: {}\n", models[selected_index]);

    Ok(models[selected_index].clone())
}

// Configuration structure
struct Config {
    llm_url: String,
    token: Option<String>,
    system_prompt: Option<String>,
    model: Option<String>,
    server: bool,
    url_supplied: bool,
    searxng: Option<String>,
}

// Parse configuration from command-line arguments and environment variables
fn parse_config() -> Result<Config> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optopt("u", "url", "LLM server URL (default: http://127.0.0.1:11434 for Ollama)", "URL");
    opts.optopt("s", "system-prompt", "System prompt (string or path to text file)", "PROMPT");
    opts.optopt("m", "model", "Model name (skip model selection)", "MODEL");
    opts.optopt("t", "token", "API token for servers with authentication enabled", "TOKEN");
    opts.optopt("", "searxng", "Base URL of a SearxNG instance to search through instead of DuckDuckGo", "URL");
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

    // Parse LLM URL
    // Priority: command-line arg > environment variable > default
    let llm_url = if let Some(url) = matches.opt_str("u") {
        url
    } else if let Ok(url) = env::var("LLM_URL") {
        url
    } else {
        // Default to Ollama's port (previously was LM Studio's 15200)
        "http://127.0.0.1:11434".to_string()
    };

    // Whether the URL was set explicitly (flag or env var) vs. falling back to the default.
    // web_fetch's LLM summarization requires the URL to be supplied explicitly, not defaulted.
    let url_supplied = matches.opt_present("u") || env::var("LLM_URL").is_ok();

    // Parse API token, for servers that have authentication enabled.
    // Priority: command-line arg > environment variable > none.
    let token = matches
        .opt_str("t")
        .or_else(|| env::var("LM_API_TOKEN").ok())
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
        None
    };

    // Parse model
    let model = matches.opt_str("m");

    // Run as an stdio MCP server instead of the interactive REPL
    let server = matches.opt_present("server");

    // SearxNG instance for web_search, replacing DuckDuckGo when set.
    // Priority: command-line arg > environment variable > none (DuckDuckGo).
    let searxng = matches
        .opt_str("searxng")
        .or_else(|| env::var("SEARXNG_URL").ok())
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty());

    Ok(Config {
        llm_url,
        token,
        system_prompt,
        model,
        server,
        url_supplied,
        searxng,
    })
}

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
    println!("\nEnvironment variables:");
    println!("  LLM_URL       LLM server URL (overridden by --url)");
    println!("  LM_API_TOKEN  API token for authenticated servers (overridden by --token)");
    println!("  SEARXNG_URL   SearxNG instance for web_search (overridden by --searxng)");
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
}

// Run haico as an stdio MCP server. stdout/stdin carry the JSON-RPC protocol, so this
// function must NOT print to stdout — all diagnostics go to stderr via eprintln!.
async fn run_mcp_server(config: Config) -> Result<()> {
    eprintln!("[haico] starting MCP server (stdio)...");

    // web_fetch summarizes pages via the LLM, so it is only exposed when --url (or LLM_URL)
    // says explicitly where that LLM lives. Otherwise it is hidden and only web_get (raw
    // cleaned page text, no LLM round-trip) is exposed. Every other tool works regardless.
    if !config.url_supplied {
        eprintln!("[haico] note: no LLM URL configured (pass --url or set LLM_URL); \
                   exposing web_get (raw page text) and hiding web_fetch (summarization).");
    }

    // --model is optional here: with none given we omit the "model" field from requests,
    // so the LLM server uses whatever model it already has loaded.
    let model = config.model.unwrap_or_default();
    if config.url_supplied && model.is_empty() {
        eprintln!("[haico] no --model given; using the model already loaded on the LLM server.");
    }

    if let Some(searxng) = &config.searxng {
        eprintln!("[haico] web_search will query SearxNG at {}.", searxng);
    }

    let lm_client = LMStudioClient::new(config.llm_url, model, config.token);
    let server = MyMcpServer::new(lm_client, config.url_supplied, config.searxng);

    eprintln!("[haico] MCP server ready; serving over stdio.");
    let running = server
        .serve(stdio())
        .await
        .context("failed to start MCP stdio server")?;
    running
        .waiting()
        .await
        .context("MCP server terminated with error")?;
    eprintln!("[haico] MCP server stopped.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse configuration first so we know the mode before any stdout output.
    let config = parse_config()?;

    // Server mode: hand off to the MCP stdio server and return (no banner, no REPL).
    if config.server {
        return run_mcp_server(config).await;
    }

    println!("          Welcome to Haico!");
    println!("=====================================\n");
    println!("Connecting to LLM server at: {}", config.llm_url);
    if let Some(searxng) = &config.searxng {
        println!("Searching through SearxNG at: {}", searxng);
    }

    // Determine which model to use
    let selected_model = if let Some(model_name) = config.model {
        // Model specified via command-line, use it directly
        println!("Using specified model: {}\n", model_name);
        model_name
    } else {
        // No model specified, fetch available models and let user select
        println!("Fetching available models...");
        let models = match fetch_available_models(&config.llm_url, config.token.as_deref()).await {
            Ok(models) => models,
            Err(e) => {
                println!("Error connecting to LLM: {e}");
                return Ok(());
            }
        };

        if models.is_empty() {
            return Err(anyhow::anyhow!("No models available from server"));
        }

        // Let user select a model interactively
        select_model_interactive(models)?
    };

    // Initialize LM Studio client with selected model
    let lm_client = LMStudioClient::new(
        config.llm_url,
        selected_model,
        config.token,
    );

    // Initialize MCP server with tools
    let server = MyMcpServer::new(lm_client.clone(), true, config.searxng);
    let openai_tools = mcp_tools_to_openai_format(&server);

    // println!("Available tools:");
    // for tool in server.tool_router.list_all() {
    //     println!("  - {}: {}", tool.name, tool.description.as_deref().unwrap_or(""));
    // }
    // println!();

    // Determine system prompt to use
    let system_prompt_content = if let Some(custom_prompt) = config.system_prompt {
        // Use custom prompt if provided
        custom_prompt
    } else {
        // Use default prompt with environment variables
        build_system_prompt()
    };

    // Conversation history
    let mut messages: Vec<Value> = vec![];
    messages.push(json!({
        "role": "system",
        "content": system_prompt_content
    }));

    // Permission manager
    let mut perm_manager = PermissionManager::new();

    // Main loop
    loop {
        // Read user input
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "/exit" || input == "/quit" {
            println!("Goodbye!");
            break;
        }

        if input == "/clear" || input == "/new" || input == "/reset" {
            messages.clear();
            messages.push(json!({
                "role": "system",
                "content": &system_prompt_content
            }));
            continue;
        }

        // Add user message
        messages.push(json!({
            "role": "user",
            "content": input
        }));

        // Get LLM response and handle tool call chains
        loop {
            println!("Fancying around...");
            let response = lm_client
                .chat_completion(messages.clone(), openai_tools.clone())
                .await?;

            let choice = &response["choices"][0];
            let message = &choice["message"];

            // Check if model wants to call a tool
            if let Some(tool_calls) = message.get("tool_calls") {
                let tool_calls_array = match tool_calls.as_array() {
                    Some(arr) => arr,
                    None => {
                        eprintln!("[ERROR] tool_calls is not an array: {:?}", tool_calls);
                        break;
                    }
                };

                // Check if array is empty
                if tool_calls_array.is_empty() {
                    // No actual tool calls, this is the final response
                    let assistant_reply = message["content"].as_str().unwrap_or("");
                    print!("\n● ");
                    let _ = render_text(assistant_reply);
                    println!();
                    messages.push(message.clone());
                    break;
                }

                // Display assistant's reasoning/content before tool calls if present
                let content = message.get("content").and_then(|c| c.as_str()).filter(|s| !s.is_empty());
                let reasoning = message.get("reasoning").and_then(|r| r.as_str()).filter(|s| !s.is_empty());

                if let Some(text) = content.or(reasoning) {
                    print!("\n● ");
                    let _ = render_text(text.trim());
                    println!();
                }

                // Add assistant message with tool call first
                messages.push(message.clone());

                for tool_call in tool_calls_array {
                    //println!("[DEBUG] Processing tool call: {}", serde_json::to_string_pretty(&tool_call).unwrap_or_default());

                    let function = &tool_call["function"];
                    let tool_name = match function["name"].as_str() {
                        Some(name) => name,
                        None => {
                            eprintln!("[ERROR] Could not extract tool name from: {:?}", function);
                            continue;
                        }
                    };

                    let args_str = match function["arguments"].as_str() {
                        Some(s) => s,
                        None => {
                            eprintln!("[ERROR] Could not extract arguments string from: {:?}", function);
                            continue;
                        }
                    };

                    let arguments: Value = match serde_json::from_str(args_str) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("[ERROR] Failed to parse arguments JSON: {} - Raw: {}", e, args_str);
                            continue;
                        }
                    };

                    // Check if permission is needed for this tool
                    let needs_permission = matches!(
                        tool_name,
                        "web_search" | "web_fetch" | "web_get" | "read" | "write_file" | "edit" | "run_command" | "ls"
                            | "crate_info" | "crate_search" | "crate_api"
                    );

                    if needs_permission && !perm_manager.is_allowed(tool_name) {
                        // Generate diff if needed
                        let diff_content: Option<String> = match tool_name {
                            "write_file" => {
                                let file_path = arguments["file_path"].as_str().unwrap_or("");
                                let new_content = arguments["content"].as_str().unwrap_or("");

                                // Check if file exists and read old content
                                if PathBuf::from(file_path).exists() {
                                    if let Ok(old_content) = fs::read_to_string(file_path) {
                                        generate_diff(file_path, &old_content, new_content).ok()
                                    } else {
                                        None
                                    }
                                } else {
                                    // New file - show all as additions
                                    generate_diff(file_path, "", new_content).ok()
                                }
                            }
                            "edit" => {
                                let file_path = arguments["file_path"].as_str().unwrap_or("");
                                let old_string = arguments["old_string"].as_str().unwrap_or("");
                                let new_string = arguments["new_string"].as_str().unwrap_or("");
                                let replace_all = arguments["replace_all"].as_bool().unwrap_or(false);

                                // Read current content and generate preview
                                if let Ok(content) = fs::read_to_string(file_path) {
                                    let new_content = if replace_all {
                                        content.replace(old_string, new_string)
                                    } else {
                                        if let Some(pos) = content.find(old_string) {
                                            let mut result = String::with_capacity(content.len());
                                            result.push_str(&content[..pos]);
                                            result.push_str(new_string);
                                            result.push_str(&content[pos + old_string.len()..]);
                                            result
                                        } else {
                                            content.clone()
                                        }
                                    };
                                    generate_diff(file_path, &content, &new_content).ok()
                                } else {
                                    None
                                }
                            }
                            _ => None
                        };

                        // Create human-readable description
                        let description = match tool_name {
                            "web_search" => {
                                let query = arguments["query"].as_str().unwrap_or("unknown");
                                format!("Search the web for: '{}'", query)
                            }
                            "web_fetch" | "web_get" => {
                                let url = arguments["url"].as_str().unwrap_or("unknown");
                                format!("Fetch content from URL: {}", url)
                            }
                            "read" => {
                                let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
                                let offset = arguments["offset"].as_u64().map(|o| o.to_string()).unwrap_or_else(|| "none".to_string());
                                let limit = arguments["limit"].as_u64().map(|l| l.to_string()).unwrap_or_else(|| "none".to_string());
                                format!("Read file: {} (offset: {}, limit: {})", file_path, offset, limit)
                            }
                            "write_file" => {
                                let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
                                let size = arguments["content"].as_str().map(|s| s.len()).unwrap_or(0);
                                format!("Write {} bytes to file: {}", size, file_path)
                            }
                            "edit" => {
                                let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
                                let from = arguments["old_string"].as_str().unwrap_or("unknown");
                                let to = arguments["new_string"].as_str().unwrap_or("unknown");
                                let replace_all = arguments["replace_all"].as_bool().unwrap_or(false);
                                if replace_all {
                                    format!("Edit file (replace all '{from}'->'{to}') in {}", file_path)
                                } else {
                                    format!("Edit file (replace once '{from}'->'{to}') in {}", file_path)
                                }
                            }
                            "run_command" => {
                                let command = arguments["command"].as_str().unwrap_or("unknown");
                                let cmd_display = if command.len() > 50 {
                                    format!("{}...", &command[..50])
                                } else {
                                    command.to_string()
                                };
                                format!("Execute command: {}", cmd_display)
                            }
                            "ls" => {
                                let path = arguments["path"].as_str().unwrap_or("unknown");
                                format!("List directory contents: {}", path)
                            }
                            "crate_info" => {
                                let name = arguments["name"].as_str().unwrap_or("unknown");
                                format!("Look up crate '{}' on crates.io", name)
                            }
                            "crate_search" => {
                                let query = arguments["query"].as_str().unwrap_or("unknown");
                                format!("Search crates.io for: '{}'", query)
                            }
                            "crate_api" => {
                                let name = arguments["name"].as_str().unwrap_or("unknown");
                                format!("Fetch the public API of crate '{}' from docs.rs", name)
                            }
                            _ => format!("Execute '{}' tool", tool_name)
                        };

                        // Ask for permission with diff if available
                        match ask_permission(tool_name, &description, diff_content.as_deref())? {
                            PermissionResponse::Allow => {
                                // Allow once, proceed with execution
                                println!("✓ Permission granted for this request\n");
                            }
                            PermissionResponse::AllowAll => {
                                // Allow all future uses
                                perm_manager.allow_tool(tool_name.to_string());
                                println!("✓ Permission granted for all future '{}' requests\n", tool_name);
                            }
                            PermissionResponse::Reject(alternative) => {
                                // User rejected, send alternative to LLM
                                println!("✗ Permission denied. User suggestion: {}\n", alternative);

                                messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": format!("Permission denied by user. User suggests: {}", alternative)
                                }));
                                continue;
                            }
                        }
                    }

                    //println!("\n[Calling tool: {} with args: {}]", tool_name, arguments);

                    // Execute the tool
                    let tool_result = server.execute_tool(tool_name, arguments).await.unwrap_or_else(|e| {
                        eprintln!("[ERROR] Tool execution failed: {:?}", e);
                        format!("Error: {}", e)
                    });

                    //println!("[Tool result: {}]\n", tool_result);

                    // Add tool result
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result
                    }));
                }

                // Continue the loop to get next response (might be more tool calls or final answer)
                continue;
            } else {
                // No tool call, this is the final response
                let assistant_reply = message["content"].as_str().unwrap_or("");
                print!("\n● ");
                let _ = render_text(assistant_reply);
                println!();
                messages.push(message.clone());
                break;
            }
        }
    }

    Ok(())
}