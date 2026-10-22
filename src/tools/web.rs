use crate::trace::Caller;
use super::MyMcpServer;
use crate::util::truncate_at_boundary;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::readable;

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct WebFetchParams {
    #[schemars(description = "The URL to fetch content from")]
    pub(crate) url: String,
    #[schemars(description = "A prompt describing what information to extract from the fetched page")]
    pub(crate) prompt: String,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct WebGetParams {
    #[schemars(description = "The URL to fetch content from")]
    pub(crate) url: String,
}

// Outcome of fetching and cleaning a web page: the cleaned, size-limited page text, or
// one of the two reasons there is no page text to hand back. Both of those carry the
// sentence to give the caller rather than an error, because a model that is told a URL
// redirected elsewhere or was blocked by a challenge can act on it, and an error struct
// printed at it cannot be acted on at all.

#[derive(Clone)]
pub(crate) enum FetchResult {
    Redirect(String),
    Blocked(String),
    Text { final_url: String, text: String },
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct WebSearchParams {
    #[schemars(description = "The search query to submit to the search engine")]
    pub(crate) query: String,
    #[schemars(description = "Maximum number of search results to return (default: 10)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_results: Option<usize>,
}


// The Firefox release we impersonate in the web tools. Bump this occasionally: a
// user-agent that trails the real release train by too many versions is itself a
// bot signal.
pub(crate) const FIREFOX_VERSION: &str = "153";

// How long the web tools remember the cookies sites give them. The user's choice: long
// enough that an anti-bot cookie earned on one fetch is still there for the next few, the
// way a browser keeps it; short enough that a session does not carry one site's tracking
// cookies around for hours.
pub(crate) const COOKIE_LIFETIME: std::time::Duration = std::time::Duration::from_secs(10 * 60);

// The web client's cookie jar, emptied once its cookies are `COOKIE_LIFETIME` old.
//
// The client lives as long as the server, so connections are pooled across fetches — and so
// would reqwest's own jar be, keeping every cookie for the whole session. reqwest's `Jar` has
// no notion of age, so this wraps one and starts a fresh one when the oldest cookie in it
// passes the lifetime. That is per jar, not per cookie: a cookie set late in a jar's life is
// forgotten with the rest, sooner than ten minutes. Per-cookie ages would mean re-implementing
// cookie domain and path matching here, which `Jar` already does correctly.
pub(crate) struct ShortLivedJar {
    // The jar, and when the first cookie went into it — None while it is empty, so a jar
    // that has sat unused does not expire the first cookie the moment it arrives.
    state: std::sync::Mutex<(reqwest::cookie::Jar, Option<std::time::Instant>)>,
    lifetime: std::time::Duration,
}

impl ShortLivedJar {
    pub(crate) fn new(lifetime: std::time::Duration) -> Self {
        Self { state: std::sync::Mutex::new((reqwest::cookie::Jar::default(), None)), lifetime }
    }

    fn set_at(&self, headers: &mut dyn Iterator<Item = &reqwest::header::HeaderValue>, url: &url::Url, now: std::time::Instant) {
        let mut headers = headers.peekable();
        if headers.peek().is_none() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        self.expire(&mut state, now);
        state.1.get_or_insert(now);
        reqwest::cookie::CookieStore::set_cookies(&state.0, &mut headers, url);
    }

    fn cookies_at(&self, url: &url::Url, now: std::time::Instant) -> Option<reqwest::header::HeaderValue> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        self.expire(&mut state, now);
        reqwest::cookie::CookieStore::cookies(&state.0, url)
    }

    fn expire(&self, state: &mut (reqwest::cookie::Jar, Option<std::time::Instant>), now: std::time::Instant) {
        if state.1.is_some_and(|first| now.duration_since(first) >= self.lifetime) {
            *state = (reqwest::cookie::Jar::default(), None);
        }
    }
}

impl reqwest::cookie::CookieStore for ShortLivedJar {
    fn set_cookies(&self, headers: &mut dyn Iterator<Item = &reqwest::header::HeaderValue>, url: &url::Url) {
        self.set_at(headers, url, std::time::Instant::now());
    }

    fn cookies(&self, url: &url::Url) -> Option<reqwest::header::HeaderValue> {
        self.cookies_at(url, std::time::Instant::now())
    }
}

// The HTTP client for the web tools (web_search, web_get, web_fetch). Many sites
// challenge or reject requests whose headers look like a bare HTTP library, so we send
// the same header set a desktop Firefox sends for a top-level page load.
//
// This only disguises the headers. The TLS ClientHello and HTTP/2 settings are still
// reqwest's, so services that fingerprint at that level (Cloudflare, DataDome) can
// still tell us apart from a real browser.
//
// Built once per server (`MyMcpServer::web_client`) rather than per call, so fetches reuse
// connections and a cookie earned on one is still there for the next (see `ShortLivedJar`).
// Per server and not in a static, because a client keeps its pooled connections on the
// runtime that opened them, and each test runs its own.

pub(crate) fn browser_client() -> reqwest::Client {
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
        .cookie_provider(std::sync::Arc::new(ShortLivedJar::new(COOKIE_LIFETIME)))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        // Fails only where `Client::new` itself would panic (no TLS backend); see llm.rs.
        .expect("HTTP client")
}

// Why a fetched page's status is read before its body.
//
// A blocked request still answers with a body, and that body is still valid HTML, so the
// readability pass downstream turns one into a page like any other. Stack Overflow behind
// Cloudflare came back to `web_get` as the single line "Enable JavaScript and cookies to
// continue", reported as a successful read — the model was told it had read the question.
//
// Two signals, because they do not always agree. The status catches an ordinary failure
// (404, 500, a plain 403). `cf-mitigated: challenge` is what Cloudflare labels its own
// interstitials with, and it does not only send them with a 4xx, so a challenge served
// with a 200 is only visible in the header.

pub(crate) fn blocked_reason(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
) -> Option<String> {
    let challenged = headers
        .get("cf-mitigated")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("challenge"));

    if challenged {
        return Some(format!(
            "blocked by a bot-detection challenge (HTTP {}). The page was not served; \
             its content is not available to fetch",
            status.as_u16()
        ));
    }

    if !status.is_success() {
        return Some(format!("the server answered HTTP {}", status));
    }

    None
}

// Resolve a search result's URL into an absolute, scheme-carrying one.
//
// DuckDuckGo's HTML results print the URL without a scheme in the link text
// ("rust-lang.org/tools/install") and wrap the real target in a protocol-relative
// redirect ("//duckduckgo.com/l/?uddg=<percent-encoded target>&rut=..."). Models that
// get the bare text back often fail to prepend a scheme and then can't fetch the page,
// so prefer the target decoded out of the redirect and fall back to the display text
// with https:// prepended.

pub(crate) fn resolve_result_url(href: Option<&str>, display_text: &str) -> String {
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

// What web_search goes through instead of DuckDuckGo, chosen at start-up from --searxng
// or --tinyfish (or their environment and config equivalents). Unset means DuckDuckGo.

#[derive(Clone)]
pub(crate) enum SearchBackend {
    // Base URL of a SearxNG instance, without the trailing slash.
    Searxng(String),
    // A Tinyfish Search API key.
    Tinyfish(String),
}

// One parsed search result. Every backend (DuckDuckGo, SearxNG, Tinyfish) produces these,
// and format_search_results renders them, so the tool's output looks the same either way.

pub(crate) struct SearchResult {
    pub(crate) title: String,
    pub(crate) url: String,
    pub(crate) snippet: String,
}


pub(crate) fn format_search_results(query: &str, results: &[SearchResult]) -> String {
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
pub(crate) struct SearxngResponse {
    #[serde(default)]
    pub(crate) results: Vec<SearxngResult>,
}


#[derive(Deserialize)]
pub(crate) struct SearxngResult {
    #[serde(default)]
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) url: String,
    #[serde(default)]
    pub(crate) content: String,
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

pub(crate) async fn searxng_search(
    client: &reqwest::Client,
    base_url: &str,
    query: &str,
    max_results: usize,
) -> Result<CallToolResult, McpError> {
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

pub(crate) const TINYFISH_URL: &str = "https://api.search.tinyfish.ai";

// The Tinyfish Search API's answer. Only the fields we render are declared; `date` comes
// with news and with some web results.

#[derive(Deserialize)]
pub(crate) struct TinyfishResponse {
    #[serde(default)]
    pub(crate) results: Vec<TinyfishResult>,
}


#[derive(Deserialize)]
pub(crate) struct TinyfishResult {
    #[serde(default)]
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) url: String,
    #[serde(default)]
    pub(crate) snippet: String,
    #[serde(default)]
    pub(crate) date: Option<String>,
}

// Search through the Tinyfish Search API (--tinyfish / TINYFISH_API_KEY). `endpoint` is
// TINYFISH_URL outside the tests.
//
// One request, one page of results: every page is a request against the key's rate limit
// (30 a minute by default), and a model asking for more than a page is rarely helped by
// the second one.
//
// As with SearxNG there is no fallback to DuckDuckGo: the key was given deliberately, so
// a rejected key or a spent allowance is reported, and says which, because the model can
// do nothing about either and the user can.

pub(crate) async fn tinyfish_search(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    query: &str,
    max_results: usize,
) -> Result<CallToolResult, McpError> {
    let response = client
        .get(format!("{}?query={}", endpoint, urlencoding::encode(query)))
        .header("X-API-Key", api_key)
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|e| {
            McpError::internal_error(format!("Failed to reach Tinyfish search: {}", e), None)
        })?;

    let status = response.status();
    let body = response.text().await.map_err(|e| {
        McpError::internal_error(format!("Failed to read response body: {}", e), None)
    })?;

    if !status.is_success() {
        let reason = match status.as_u16() {
            401 => "the API key was rejected. Check --tinyfish, TINYFISH_API_KEY or `tinyfish` \
                    in the config."
                .to_string(),
            402 => "the free daily allowance is used up and the wallet is empty.".to_string(),
            429 => "too many searches in the last minute; wait a minute before searching again."
                .to_string(),
            // Anything else is passed on as the service said it. Capped, because a proxy in
            // front of the API may answer with a whole HTML page.
            _ => truncate_at_boundary(body.trim(), 1000).to_string(),
        };
        return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
            "Tinyfish search failed (HTTP {}): {}",
            status.as_u16(),
            reason
        ))]));
    }

    let parsed = serde_json::from_str::<TinyfishResponse>(&body).map_err(|e| {
        McpError::internal_error(format!("Tinyfish returned an unreadable answer: {}", e), None)
    })?;

    Ok(CallToolResult::success(vec![ContentBlock::text(
        format_search_results(query, &tinyfish_results(parsed, max_results)),
    )]))
}


// The date leads the snippet: a model judging whether a page is current has nothing else
// to go on before opening it.

pub(crate) fn tinyfish_results(response: TinyfishResponse, max_results: usize) -> Vec<SearchResult> {
    response
        .results
        .into_iter()
        .take(max_results)
        .map(|r| SearchResult {
            title: r.title,
            url: r.url,
            snippet: match r.date.as_deref().map(str::trim).filter(|d| !d.is_empty()) {
                Some(date) => format!("[{}] {}", date, r.snippet),
                None => r.snippet,
            },
        })
        .collect()
}

// Scrape SearxNG's "simple" theme result page. Results are <article class="result">, with
// the target on the <h3> link and repeated on the a.url_header wrapper above it (older
// SearxNG put that class on a.url_wrapper instead, so accept either as a fallback).

pub(crate) fn parse_searxng_html(html: &str, max_results: usize) -> Vec<SearchResult> {
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


#[tool_router(router = web_router, vis = "pub(crate)")]
impl MyMcpServer {
    pub(crate) async fn fetch_clean_text(&self, url: &str) -> Result<FetchResult, McpError> {
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
        let client = &self.web_client;

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

        // Nothing is cached on this path, deliberately: a challenge is a statement about
        // this request, not about the URL, so storing it would hand the same stub back to
        // every later fetch in the session — including the research sub-agent's, which
        // shares this cache — without ever asking the site again.
        if let Some(reason) = blocked_reason(response.status(), response.headers()) {
            return Ok(FetchResult::Blocked(format!(
                "Failed to fetch {}: {}",
                final_url, reason
            )));
        }

        // Get the response body
        let html = response
            .text()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to read response body: {}", e), None)
            })?;

        // Convert the page to readable text: the article subtree only, the way a browser's
        // reader mode presents it.
        let cleaned_text = readable::html_to_text(&html, &final_url);

        // Limit size to avoid context overflow. The cut is made at a line boundary so the tail
        // of the text is not a half-finished sentence, and never inside a multi-byte character.
        const MAX_LENGTH: usize = 65536;
        let text = if cleaned_text.len() > MAX_LENGTH {
            let head = truncate_at_boundary(&cleaned_text, MAX_LENGTH);
            let end = head.rfind('\n').unwrap_or(head.len());
            format!("{}\n\n[Page truncated]", &cleaned_text[..end])
        } else {
            cleaned_text
        };

        let result = FetchResult::Text { final_url, text };
        self.cache.lock().unwrap().put(url_string.clone(), result.clone());
        Ok(result)
    }


    #[tool(description = "Use when you already have a specific URL and suspect that it is very big, and want a focused answer pulled from it rather than the whole page — the page is fetched, converted to readable text, and an assistant extracts just what your prompt asks for. Takes a URL and a prompt describing what information to extract. This tool saves context, but takes more time to process. Use web_get instead when you want the full raw text yourself. Use research instead when you do not already have the URL — research is this same tool with the finding of the page included.")]
    pub(crate) async fn web_fetch(&self, params: Parameters<WebFetchParams>) -> Result<CallToolResult, McpError> {
        let (final_url, truncated_text) = match self.fetch_clean_text(&params.0.url).await? {
            FetchResult::Redirect(message) => {
                return Ok(CallToolResult::success(vec![ContentBlock::text(message)]));
            }
            FetchResult::Blocked(message) => {
                return Ok(CallToolResult::error(vec![ContentBlock::text(message)]));
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
            .simple_completion(Caller::Summarizer, system_prompt, &user_prompt)
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


    #[tool(description = "Use when you already have the exact URL and want to read or process the full page yourself, with no filtering — fetches the page and returns the cleaned text directly (no summarization). Choose web_fetch instead if you suspect that the page will be very big, but you only need a specific fact and want it extracted for you. Choose research instead whenever you do not already know which page holds the answer, or when one page is unlikely to settle it — reading pages yourself to find out is what research is for.")]
    pub(crate) async fn web_get(&self, params: Parameters<WebGetParams>) -> Result<CallToolResult, McpError> {
        let (final_url, text) = match self.fetch_clean_text(&params.0.url).await? {
            FetchResult::Redirect(message) => {
                return Ok(CallToolResult::success(vec![ContentBlock::text(message)]));
            }
            FetchResult::Blocked(message) => {
                return Ok(CallToolResult::error(vec![ContentBlock::text(message)]));
            }
            FetchResult::Text { final_url, text } => (final_url, text),
        };

        let result_text = format!(
            "URL: {}\n\n=== Page Content ===\n\n{}",
            final_url, text
        );

        Ok(CallToolResult::success(vec![ContentBlock::text(result_text)]))
    }


    #[tool(description = "Use when the snippets alone will answer you — search the web by keywords and get back titles, URLs and snippets, using DuckDuckGo (or the configured SearxNG or Tinyfish). If the snippets settle it, you are done. If they do not, and you are about to start opening the results one by one, stop and call research with the question instead: it does the searching and the reading and hands back the answer, without the pages landing in your context.")]
    pub(crate) async fn web_search(&self, params: Parameters<WebSearchParams>) -> Result<CallToolResult, McpError> {
        let query = params.0.query.clone();
        let max_results = params.0.max_results.unwrap_or(10);

        // A configured backend replaces DuckDuckGo entirely.
        match &self.search {
            Some(SearchBackend::Searxng(base_url)) => {
                return searxng_search(&self.web_client, base_url, &query, max_results).await;
            }
            Some(SearchBackend::Tinyfish(api_key)) => {
                return tinyfish_search(&self.web_client, TINYFISH_URL, api_key, &query, max_results)
                    .await;
            }
            None => {}
        }

        // URL encode the query
        let encoded_query = urlencoding::encode(&query);
        let search_url = format!("https://html.duckduckgo.com/html/?q={}", encoded_query);

        // Fetch search results. browser_client() already carries the Firefox header set;
        // this request additionally looks like the search form on duckduckgo.com was
        // submitted, so the Referer and Sec-Fetch-Site have to agree with each other.
        let client = &self.web_client;

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

}

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
    fn web_cookies_are_forgotten_once_the_jar_is_ten_minutes_old() {
        use super::ShortLivedJar;
        use reqwest::header::HeaderValue;
        use std::time::{Duration, Instant};

        let jar = ShortLivedJar::new(Duration::from_secs(600));
        let url: url::Url = "https://example.com/".parse().unwrap();
        let start = Instant::now();

        // Nothing stored yet, and an empty Set-Cookie list starts no clock.
        assert!(jar.cookies_at(&url, start).is_none());
        jar.set_at(&mut std::iter::empty(), &url, start);

        let challenge = HeaderValue::from_static("challenge=passed; Path=/");
        jar.set_at(&mut std::iter::once(&challenge), &url, start);
        // Sent back on the next fetch, as a browser would…
        assert_eq!(jar.cookies_at(&url, start + Duration::from_secs(599)).unwrap(), "challenge=passed");
        // …and to that site only.
        let other: url::Url = "https://other.example/".parse().unwrap();
        assert!(jar.cookies_at(&other, start + Duration::from_secs(1)).is_none());

        // Ten minutes after the first cookie went in, the jar is emptied…
        assert!(jar.cookies_at(&url, start + Duration::from_secs(600)).is_none());
        // …and a cookie arriving afterwards starts a fresh ten minutes of its own.
        let later = start + Duration::from_secs(900);
        jar.set_at(&mut std::iter::once(&challenge), &url, later);
        assert!(jar.cookies_at(&url, later + Duration::from_secs(599)).is_some());
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
    fn reports_an_unsuccessful_status() {
        use reqwest::header::HeaderMap;
        use reqwest::StatusCode;

        assert!(blocked_reason(StatusCode::OK, &HeaderMap::new()).is_none());

        let reason = blocked_reason(StatusCode::NOT_FOUND, &HeaderMap::new()).unwrap();
        assert!(reason.contains("404"), "{}", reason);
    }


    #[test]
    fn names_a_cloudflare_challenge_even_on_a_200() {
        use reqwest::header::{HeaderMap, HeaderValue};
        use reqwest::StatusCode;

        // The header Stack Overflow's 403 interstitial carries, verbatim.
        let mut headers = HeaderMap::new();
        headers.insert("cf-mitigated", HeaderValue::from_static("challenge"));

        let reason = blocked_reason(StatusCode::FORBIDDEN, &headers).unwrap();
        assert!(reason.contains("bot-detection challenge"), "{}", reason);

        // A challenge does not have to arrive with a failing status, which is the whole
        // reason the header is consulted as well.
        let reason = blocked_reason(StatusCode::OK, &headers).unwrap();
        assert!(reason.contains("bot-detection challenge"), "{}", reason);

        // Some other value in that header is not a challenge.
        let mut other = HeaderMap::new();
        other.insert("cf-mitigated", HeaderValue::from_static("rewrite"));
        assert!(blocked_reason(StatusCode::OK, &other).is_none());
    }


    // Answers one request with `response` verbatim (status line and all) and hands back the
    // request head, so a test can see the query string and the key header.
    async fn serve_raw(response: &'static str) -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let head = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let sink = head.clone();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                match socket.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => request.extend_from_slice(&buf[..n]),
                }
            }
            *sink.lock().unwrap() = String::from_utf8_lossy(&request).to_string();
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
        });
        (format!("http://{}", addr), head)
    }

    #[tokio::test]
    async fn tinyfish_search_sends_the_key_and_renders_the_results() {
        const RESPONSE: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 342\r\nConnection: close\r\n\r\n{\"query\":\"rust async\",\"results\":[{\"position\":1,\"site_name\":\"rust-lang.org\",\"title\":\"Async Book\",\"snippet\":\"Asynchronous programming in Rust.\",\"url\":\"https://rust-lang.github.io/async-book/\",\"date\":\"2026-01-02\"},{\"position\":2,\"site_name\":\"tokio.rs\",\"title\":\"Tokio\",\"snippet\":\"A runtime.\",\"url\":\"https://tokio.rs/\"}],\"total_results\":2,\"page\":0}";
        let (url, head) = serve_raw(RESPONSE).await;

        let result = super::tinyfish_search(&reqwest::Client::new(), &url, "secret", "rust async", 10)
            .await
            .unwrap();
        let text = result_text(&result);
        assert_ne!(result.is_error, Some(true), "{text}");
        assert!(text.contains("(2 results)"), "{text}");
        assert!(text.contains("1. Async Book\n   URL: https://rust-lang.github.io/async-book/\n   [2026-01-02] Asynchronous"), "{text}");
        assert!(text.contains("2. Tokio\n   URL: https://tokio.rs/\n   A runtime."), "{text}");

        let head = head.lock().unwrap().to_lowercase();
        assert!(head.starts_with("get /?query=rust%20async "), "{head}");
        assert!(head.contains("x-api-key: secret\r\n"), "{head}");
    }

    #[tokio::test]
    async fn tinyfish_search_names_a_rejected_key() {
        const RESPONSE: &str = "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
        let (url, _head) = serve_raw(RESPONSE).await;

        let result = super::tinyfish_search(&reqwest::Client::new(), &url, "wrong", "q", 10).await.unwrap();
        let text = result_text(&result);
        assert_eq!(result.is_error, Some(true), "{text}");
        assert!(text.contains("HTTP 401") && text.contains("API key was rejected"), "{text}");
    }

    #[test]
    fn tinyfish_results_stop_at_max_results() {
        let response: super::TinyfishResponse = serde_json::from_value(json!({
            "results": [
                {"title": "a", "url": "https://a/", "snippet": "first", "date": " "},
                {"title": "b", "url": "https://b/", "snippet": "second"},
            ]
        }))
        .unwrap();
        let results = super::tinyfish_results(response, 1);
        assert_eq!(results.len(), 1);
        // A blank date adds nothing to the snippet.
        assert_eq!(results[0].snippet, "first");
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
