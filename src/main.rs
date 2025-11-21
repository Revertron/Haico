use anyhow::{Context, Result};
use reqwest;
use rmcp::model::{CallToolResult, Content};
use rmcp::{tool, tool_router, ErrorData as McpError};
use rmcp::handler::server::{tool::ToolRouter, wrapper::Parameters};
use serde_json::{json, Value};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use tokio;

// LM Studio client
struct LMStudioClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl LMStudioClient {
    fn new(base_url: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            model,
        }
    }

    async fn chat_completion(
        &self,
        messages: Vec<Value>,
        tools: Vec<Value>,
    ) -> Result<Value> {
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&json!({
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"
            }))
            .send()
            .await
            .context("Failed to send request to LM Studio")?;

        let result = response
            .json::<Value>()
            .await
            .context("Failed to parse response")?;

        Ok(result)
    }
}

// Tool parameter structs
#[derive(Serialize, Deserialize, JsonSchema)]
struct CalculatorParams {
    operation: String,
    a: f64,
    b: f64,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct ReadParams {
    file_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct WriteParams {
    file_path: String,
    content: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct EditParams {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct GlobParams {
    pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct GrepParams {
    pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    glob: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    case_insensitive: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    head_limit: Option<usize>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct WebFetchParams {
    url: String,
    prompt: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct WebSearchParams {
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_results: Option<usize>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct LSParams {
    path: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct BashParams {
    command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<u64>,
}

// MCP Server with tools
#[derive(Clone)]
struct MyMcpServer {
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MyMcpServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Perform basic arithmetic operations (add, subtract, multiply, divide)")]
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
                    return Ok(CallToolResult::error(vec![Content::text("Division by zero")]));
                }
                a / b
            }
            _ => return Ok(CallToolResult::error(vec![Content::text("Unknown operation")])),
        };

        Ok(CallToolResult::success(vec![Content::text(format!("Result: {}", result))]))
    }

    #[tool(description = "Reads a file from the local filesystem with optional offset and limit for pagination")]
    async fn read(&self, params: Parameters<ReadParams>) -> Result<CallToolResult, McpError> {
        let file_path = PathBuf::from(&params.0.file_path);

        if !file_path.exists() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "File not found: {}",
                params.0.file_path
            ))]));
        }

        if !file_path.is_file() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
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

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Writes content to a file, creating or overwriting it")]
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

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Successfully wrote {} bytes to {}",
            params.0.content.len(),
            params.0.file_path
        ))]))
    }

    #[tool(description = "Performs exact string replacement in a file")]
    async fn edit(&self, params: Parameters<EditParams>) -> Result<CallToolResult, McpError> {
        let file_path = PathBuf::from(&params.0.file_path);

        if !file_path.exists() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
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
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String not found in file: '{}'",
                    params.0.old_string
                ))]));
            }
        };

        if content == new_content {
            return Ok(CallToolResult::error(vec![Content::text(
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

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Successfully replaced {} occurrence(s) in {}",
            count, params.0.file_path
        ))]))
    }

    #[tool(description = "Fast file pattern matching using glob patterns like **/*.rs or src/**/*.json")]
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

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Search for patterns in files using regex. Supports filtering by glob patterns and multiple output modes")]
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

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Fetches content from a URL and converts HTML to readable text. Takes a URL and a prompt describing what information to extract")]
    async fn web_fetch(&self, params: Parameters<WebFetchParams>) -> Result<CallToolResult, McpError> {
        let url = params.0.url.clone();
        let prompt = params.0.prompt.clone();

        // Validate URL
        let parsed_url = url::Url::parse(&url).map_err(|e| {
            McpError::invalid_params(format!("Invalid URL: {}", e), None)
        })?;

        // Upgrade HTTP to HTTPS if needed
        let url_string = if parsed_url.scheme() == "http" {
            url.replace("http://", "https://")
        } else {
            url.clone()
        };

        // Fetch the content
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (compatible; MCPBot/1.0)")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| {
                McpError::internal_error(format!("Failed to build HTTP client: {}", e), None)
            })?;

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
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "URL redirected to different host: {}\nPlease call web_fetch again with this URL if you want to fetch it.",
                final_url
            ))]));
        }

        // Get the response body
        let html = response
            .text()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to read response body: {}", e), None)
            })?;

        // Convert HTML to plain text
        let text_content = html2text::from_read(html.as_bytes(), 100);

        // Trim and clean up the text
        let cleaned_text = text_content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        // Limit output size
        const MAX_LENGTH: usize = 50000;
        let output = if cleaned_text.len() > MAX_LENGTH {
            format!(
                "{}\n\n[Content truncated - {} of {} characters shown]",
                &cleaned_text[..MAX_LENGTH],
                MAX_LENGTH,
                cleaned_text.len()
            )
        } else {
            cleaned_text
        };

        let result_text = format!(
            "URL: {}\nPrompt: {}\n\n=== Content ===\n\n{}",
            final_url, prompt, output
        );

        Ok(CallToolResult::success(vec![Content::text(result_text)]))
    }

    #[tool(description = "Searches the web using DuckDuckGo and returns search results with titles, URLs, and snippets")]
    async fn web_search(&self, params: Parameters<WebSearchParams>) -> Result<CallToolResult, McpError> {
        let query = params.0.query.clone();
        let max_results = params.0.max_results.unwrap_or(10);

        // URL encode the query
        let encoded_query = urlencoding::encode(&query);
        let search_url = format!("https://html.duckduckgo.com/html/?q={}", encoded_query);

        // Fetch search results
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| {
                McpError::internal_error(format!("Failed to build HTTP client: {}", e), None)
            })?;

        let response = client
            .get(&search_url)
            .send()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to fetch search results: {}", e), None)
            })?;

        let html = response
            .text()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to read response body: {}", e), None)
            })?;

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
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            if !title.is_empty() {
                results.push(format!(
                    "{}. {}\n   URL: {}\n   {}\n",
                    index + 1,
                    title,
                    if url.is_empty() { "N/A" } else { &url },
                    snippet
                ));
            }
        }

        let output = if results.is_empty() {
            format!("No search results found for query: '{}'", query)
        } else {
            format!(
                "Search results for '{}' ({} results):\n\n{}",
                query,
                results.len(),
                results.join("\n")
            )
        };

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Lists directory contents with file/folder information")]
    async fn ls(&self, params: Parameters<LSParams>) -> Result<CallToolResult, McpError> {
        let path = PathBuf::from(&params.0.path);

        if !path.exists() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Path not found: {}",
                params.0.path
            ))]));
        }

        if !path.is_dir() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
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
                return Ok(CallToolResult::error(vec![Content::text(format!(
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

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Executes a bash command with optional timeout (default 120s, max 600s)")]
    async fn bash(&self, params: Parameters<BashParams>) -> Result<CallToolResult, McpError> {
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
                    Ok(CallToolResult::success(vec![Content::text(result_text)]))
                } else {
                    Ok(CallToolResult::error(vec![Content::text(format!(
                        "Command failed with exit code: {}\n\n{}",
                        output.status.code().unwrap_or(-1),
                        result_text
                    ))]))
                }
            }
            Ok(Err(e)) => {
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to execute command: {}",
                    e
                ))]))
            }
            Err(_) => {
                // Timeout - the child is already consumed, just return error
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "Command timed out after {} seconds",
                    timeout_secs
                ))]))
            }
        }
    }

    // Helper to execute tools by name
    async fn execute_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let result = match name {
            "calculator" => {
                let params: CalculatorParams = serde_json::from_value(arguments)?;
                self.calculator(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
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
            "web_search" => {
                let params: WebSearchParams = serde_json::from_value(arguments)?;
                self.web_search(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "ls" => {
                let params: LSParams = serde_json::from_value(arguments)?;
                self.ls(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            "bash" => {
                let params: BashParams = serde_json::from_value(arguments)?;
                self.bash(Parameters(params)).await.map_err(|e| anyhow::anyhow!("{:?}", e))?
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

#[tokio::main]
async fn main() -> Result<()> {
    println!("LM Studio + MCP Tool Calling Example");
    println!("=====================================\n");

    // Initialize LM Studio client
    let lm_client = LMStudioClient::new(
        "http://127.0.0.1:15200".to_string(),
        "openai/gpt-oss-120b".to_string(), // Replace with actual model name
    );

    // Initialize MCP server with tools
    let server = MyMcpServer::new();
    let openai_tools = mcp_tools_to_openai_format(&server);

    println!("Available tools:");
    for tool in server.tool_router.list_all() {
        println!("  - {}: {}", tool.name, tool.description.as_deref().unwrap_or(""));
    }
    println!();

    // Conversation history
    let mut messages: Vec<Value> = vec![];

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

        if input == "/clear" || input == "/new" {
            messages.clear();
            continue;
        }

        // Add user message
        messages.push(json!({
            "role": "user",
            "content": input
        }));

        // Get LLM response and handle tool call chains
        loop {
            println!("[DEBUG] Getting response from LLM...");
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
                    println!("\n ● {}\n", assistant_reply);
                    messages.push(message.clone());
                    break;
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

                    println!("\n[Calling tool: {} with args: {}]", tool_name, arguments);

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
                println!("\n ● {}\n", assistant_reply);
                messages.push(message.clone());
                break;
            }
        }
    }

    Ok(())
}