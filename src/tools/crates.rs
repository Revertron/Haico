use super::MyMcpServer;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct CrateInfoParams {
    #[schemars(description = "The exact crate name as published on crates.io (e.g. 'serde', 'tokio', 'reqwest')")]
    pub(crate) name: String,
    #[schemars(description = "Optional exact version to inspect (e.g. '1.0.210'). Defaults to the latest stable release.")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) version: Option<String>,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct CrateSearchParams {
    #[schemars(description = "Free-text search query, e.g. 'async http client' or 'json serialization'")]
    pub(crate) query: String,
    #[schemars(description = "Maximum number of results to return (default: 15, max: 50)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) limit: Option<usize>,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct CrateApiParams {
    #[schemars(description = "The exact crate name as published on crates.io (e.g. 'serde')")]
    pub(crate) name: String,
    #[schemars(description = "Crate version, or 'latest'. Defaults to 'latest'.")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) version: Option<String>,
    #[schemars(description = "Optional case-sensitive substring to filter the public API by item path (e.g. 'Serialize', 'Client::new', 'builder'). Omit to list the entire public API.")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) item: Option<String>,
}

// Compile a user-supplied glob ("*.rs", "**/*.{ts,tsx}") into a matcher.
//
// globset rather than the `glob` crate because the tool descriptions promise brace
// expansion, which `glob::Pattern` does not support.

pub(crate) const CRATES_IO_API: &str = "https://crates.io/api/v1";
pub(crate) const DOCS_RS: &str = "https://docs.rs";

// Shared HTTP client for the Rust API tools (crates.io / docs.rs). crates.io's crawler
// policy requires a descriptive, identifying User-Agent, so we always set one.

pub(crate) fn rust_api_client() -> Result<reqwest::Client, McpError> {
    reqwest::Client::builder()
        .user_agent("haico/0.1.0 (Rust crate API lookup tool)")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| McpError::internal_error(format!("Failed to build HTTP client: {}", e), None))
}

// The Firefox release we impersonate in the web tools. Bump this occasionally: a
// user-agent that trails the real release train by too many versions is itself a
// bot signal.

#[tool_router(router = crates_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(description = "Before adding or upgrading a dependency, call this to get ground truth from crates.io instead of guessing version numbers or feature flags from memory — which is frequently stale or wrong and leads to Cargo.toml that won't resolve. Given a crate name, returns its description, the latest stable version, recent versions (yanked ones flagged), the exact feature flags you can enable, license, minimum supported Rust version (MSRV), useful links (docs.rs / repository / homepage), and a ready-to-paste `name = \"version\"` line. Use it whenever you write a [dependencies] entry, pick a version, or need to know which features to turn on.")]
    pub(crate) async fn crate_info(&self, params: Parameters<CrateInfoParams>) -> Result<CallToolResult, McpError> {
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
    pub(crate) async fn crate_search(&self, params: Parameters<CrateSearchParams>) -> Result<CallToolResult, McpError> {
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
    pub(crate) async fn crate_api(&self, params: Parameters<CrateApiParams>) -> Result<CallToolResult, McpError> {
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
}
