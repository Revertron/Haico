//! The `library` sub-agent: a question goes in, an answer from the library's articles comes
//! out, and the articles read to get there never touch the main conversation.
//!
//! The same shape as `research`, for the same reason — an article is thousands of tokens, and
//! once in the main history it stays there: re-sent with every later request, re-prefilled
//! whenever the server's cache is lost, and competing for a small model's attention long after
//! the question it answered. The differences are what it reads and how cheaply: the articles
//! are local markdown (`crate::library`), so a lookup is a file read rather than a fetch, and
//! the model's own round trips are the whole cost.
//!
//! **It is bounded by time, not by how much it reads.** The windows haico runs against are
//! 128k–262k, so a librarian could read every article on a crate and still fit; what it cannot
//! do is take a minute over it, because the main model — and on an unattended run, the whole
//! task — waits for the answer. So lookups stop at `TIME_BUDGET`, and the round still running
//! at that moment is dropped (which also stops the server generating it) — except the first
//! round, which always runs whole: a librarian cut off before its first lookup has nothing to
//! answer from.
//!
//! **Its two tools are not router tools.** `library_search` and `library_read` exist only
//! inside this loop, defined here as JSON and dispatched here, so the main model never sees
//! them and cannot spend its own context reading articles directly — which is the one thing
//! this tool exists to prevent. It also means `library` can never appear in its own list.

use super::MyMcpServer;
use crate::library::Library;
use crate::session::without_think_block;
use crate::trace::Caller;
use crossterm::style::{ResetColor, SetForegroundColor};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

const SEARCH: &str = "library_search";
const READ: &str = "library_read";

// How long the librarian may spend looking things up. The answer that follows is not
// timed: it is one request with no tools, and cutting it off would throw away everything read.
pub(crate) const TIME_BUDGET: Duration = Duration::from_secs(120);

// Not the budget — time is — but the backstop under it. Searches and reads are instant, so a
// model that loops on them could otherwise fit dozens into the time, each round re-sending all
// the ones before.
const MAX_LOOKUPS: usize = 24;

// The round that calls no tool and says nothing either — see research's MAX_ROUNDS.
const MAX_ROUNDS: usize = MAX_LOOKUPS + 2;

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct LibraryParams {
    #[schemars(
        description = "The question, naming the crate and what you are trying to do — or the \
                       exact compiler error, with the code it points at."
    )]
    pub(crate) question: String,
}

// The two tools, as the sub-agent is offered them.
fn library_tools() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": SEARCH,
                "description": "Search the library's articles. Returns the best-matching \
                                sections, each with an id to read, its heading and the line \
                                that matched. Search with the words the answer would contain: \
                                type and function names, error codes like E0599, the crate's \
                                name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keywords to search for."}
                    },
                    "required": ["query"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": READ,
                "description": "Read one section of an article by its id, like 'tokio/select#2'. \
                                An article id without '#' returns the article's list of \
                                sections instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "A section id from library_search, or an article id."}
                    },
                    "required": ["id"]
                }
            }
        }),
    ]
}

// What the sub-agent is told. The catalog goes in whole: with a few dozen articles it is
// cheaper to show the sub-agent every title than to make it search for one it could have
// seen was there.
fn library_prompt(catalog: &str) -> String {
    format!(
        "You answer programming questions from a library of articles about Rust crates. You \
         are given one question and two tools: {search} and {read}.\n\n\
         The library's articles:\n{catalog}\n\
         How to work:\n\
         - If an article above is plainly about the question, read its outline with {read} \
           and then the sections you need. Otherwise start with {search}.\n\
         - Read the sections that answer the question, not whole articles.\n\
         - Stop as soon as you can answer. Your time for lookups is short: prefer reading the \
           one section that answers over searching again. When you are told to stop, answer \
           from what you have read.\n\n\
         When you have the answer, reply with text and no tool call:\n\
         - Answer the question directly.\n\
         - Copy code, signatures and names exactly as the article gives them — the person \
           reading your answer is going to compile it.\n\
         - Say which version the answer applies to when the article says.\n\
         - If the library does not cover the question, say so plainly in one sentence. Do \
           not fill the gap from memory: a clear \"the library does not cover this\" sends \
           the reader to other documentation, and a guess sends them to a compiler error.\n\n\
         Do not ask questions back — there is nobody to answer them. Do not describe what you \
         are about to do: either call a tool or give the answer.",
        search = SEARCH,
        read = READ,
        catalog = catalog,
    )
}

/// Whether an answer is the librarian saying the library had nothing on (part of) the
/// question. Kept in step with `NOT_COVERED` in `scripts/trace_to_dataset.py`.
pub(crate) fn says_not_covered(answer: &str) -> bool {
    static NOT_COVERED: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    NOT_COVERED
        .get_or_init(|| {
            regex::Regex::new(
                r"(?i)\b(?:library|articles?)\b[^.\n]{0,40}\b(?:does not|doesn't|do not|don't)\b[^.\n]{0,20}\b(?:cover|mention|say|describe|answer|include)",
            )
            .unwrap()
        })
        .is_match(answer)
}

#[tool_router(router = library_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(
        description = "Ask the library: articles written about the Rust crates listed under \
                       Library in your system prompt, and how to use them correctly. A \
                       librarian reads the articles and hands back the answer, with the exact \
                       code, so the articles never enter your context. Use this BEFORE writing \
                       code against one of those crates when you are not sure of its API, and \
                       ALWAYS when a build fails with errors involving one of them — pass the \
                       error message itself. It is local and quick, so ask rather than guess: \
                       one question to the library costs less than one failed build. Ask one \
                       specific question, like \"mio 1.0: how do I register a TcpListener \
                       for readable events?\". For anything the library does not cover, use \
                       crate_api or research."
    )]
    pub(crate) async fn library(
        &self,
        params: Parameters<LibraryParams>,
    ) -> Result<CallToolResult, McpError> {
        let question = params.0.question.trim().to_string();
        if question.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(
                "The question was empty. Put the question itself in the 'question' field \
                 and call library again.",
            )]));
        }
        let Some(root) = self.library_root.clone() else {
            return Ok(CallToolResult::error(vec![ContentBlock::text(
                "The library is not available in this session. Use crate_api or research.",
            )]));
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(
            self.run_library(&root, &question).await,
        )]))
    }
}

impl MyMcpServer {
    fn library_note(&self, line: &str) {
        if self.interactive {
            outln!("{}{}{}", SetForegroundColor(crate::out::DIM), line, ResetColor);
        }
    }

    // The loop. Never returns an error, for research's reason: what went wrong is still
    // something to tell the model in words it can act on.
    pub(crate) async fn run_library(&self, root: &Path, question: &str) -> String {
        self.run_library_within(root, question, TIME_BUDGET).await
    }

    // The loop, with the budget as a parameter so a test can spend it without waiting for it.
    async fn run_library_within(&self, root: &Path, question: &str, budget: Duration) -> String {
        // No line announcing the question: the call line above says it (`tool_call_summary`).
        // A line printed from here raced that one — the tool starts the moment its call is
        // sent to the frontend, often before the frontend has closed the reasoning fold — and
        // landed inside the fold, hidden under "(thinking)".
        crate::debug::log("library question", question);

        // Loaded per call, so an article added mid-session is there for the next question.
        let library = Library::load(root);
        if library.is_empty() {
            return "The library is empty, so nothing could be looked up. Use crate_api or \
                    research instead."
                .to_string();
        }

        let client = self.llm_client.for_sub_agent();
        let tools = library_tools();
        let mut messages = vec![
            json!({"role": "system", "content": library_prompt(&library.catalog())}),
            json!({"role": "user", "content": question}),
        ];

        // Sections actually read, so the answer can say where it came from — and so the
        // model, or the person reading the transcript, can check it against the article.
        let mut sources: Vec<String> = Vec::new();
        let mut spent = 0usize;
        let deadline = Instant::now() + budget;

        for round in 0..MAX_ROUNDS {
            let remaining = deadline.saturating_duration_since(Instant::now());
            // The first round always runs whole: until it has, there is nothing to answer from.
            // By round, not by lookups made, so a first call that fails does not leave every
            // later round exempt from the clock too.
            if spent >= MAX_LOOKUPS || (round > 0 && remaining.is_zero()) {
                break;
            }

            let request = client.chat_completion_streaming(Caller::Library, messages.clone(), tools.clone(), |_| {});
            let reply = if round == 0 {
                Ok(request.await)
            } else {
                tokio::time::timeout(remaining, request).await
            };
            let message = match reply {
                // Out of time mid-round: the half-generated reply is dropped, and with it the
                // stream, which is what stops the server spending more on it.
                Err(_) => {
                    self.library_note("  Out of time for lookups");
                    break;
                }
                Ok(Ok(message)) => message,
                Ok(Err(e)) => {
                    crate::debug::log("library failed", &format!("{:#}", e));
                    return format!(
                        "The librarian could not be reached: {}. Nothing was looked up. Try \
                         again, or use crate_api.",
                        e
                    );
                }
            };

            let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
            if calls.is_empty() {
                let answer = message["content"].as_str().unwrap_or("").trim();
                if !answer.is_empty() {
                    return self.finish_library(question, &without_think_block(answer), &sources);
                }
                break;
            }

            messages.push(message.clone());

            // Every tool_call_id gets a reply, on every path — the main loop's invariant,
            // for the main loop's reason.
            for call in &calls {
                let id = call["id"].as_str().unwrap_or_default().to_string();
                let name = call["function"]["name"].as_str().unwrap_or_default();

                let content = if name != SEARCH && name != READ {
                    format!(
                        "'{}' is not available for this task. You have exactly two tools: {} \
                         and {}.",
                        name, SEARCH, READ
                    )
                } else if spent >= MAX_LOOKUPS {
                    "That is enough lookups. Answer the question now, from what you have \
                     already read."
                        .to_string()
                } else {
                    match super::parse_tool_arguments(&call["function"]) {
                        Ok(arguments) => {
                            spent += 1;
                            self.library_lookup(&library, name, &arguments, &mut sources)
                        }
                        Err(e) => e,
                    }
                };

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": content,
                }));
            }
        }

        // Out of time or lookups, or stalled: one more request with no tools at all, so there
        // is nothing left to call — research's ending, for research's reason.
        messages.push(json!({
            "role": "user",
            "content": format!(
                "Stop looking things up and answer now: {}\n\nUse only what you have already \
                 read. If it did not answer the question, say so plainly rather than guessing.",
                question
            ),
        }));

        match client.chat_completion_plain(Caller::Library, messages, None).await {
            Ok(answer) => self.finish_library(question, without_think_block(&answer), &sources),
            Err(e) => {
                crate::debug::log("library answer failed", &format!("{:#}", e));
                format!(
                    "The librarian read {} section(s) but could not produce an answer: {}.",
                    sources.len(),
                    e
                )
            }
        }
    }

    fn library_lookup(
        &self,
        library: &Library,
        name: &str,
        arguments: &Value,
        sources: &mut Vec<String>,
    ) -> String {
        if name == SEARCH {
            let query = arguments["query"].as_str().unwrap_or_default();
            self.library_note(&format!("  Searching: {}", query));
            let hits = library.search(query);
            if hits.is_empty() {
                return format!(
                    "Nothing in the library matches '{}'. Try other words — a type or function \
                     name, or the crate's name — or read an article's outline from the list \
                     you were given.",
                    query
                );
            }
            let mut out = String::new();
            for hit in hits {
                let heading = if hit.heading.is_empty() { "(introduction)".to_string() } else { hit.heading };
                out.push_str(&format!("- {} — {} > {}\n  {}\n", hit.id, hit.title, heading, hit.snippet));
            }
            return out;
        }

        let id = arguments["id"].as_str().unwrap_or_default();
        self.library_note(&format!("  Reading {}", id));
        match library.read(id) {
            Ok(text) => {
                if id.contains('#') && !sources.iter().any(|s| s == id) {
                    sources.push(id.trim().to_string());
                }
                text
            }
            Err(e) => e,
        }
    }

    fn finish_library(&self, question: &str, answer: &str, sources: &[String]) -> String {
        // Never cut. What comes back from here is usually code the model is about to copy,
        // and an example stopped mid-block is worse than none: the model compiles the half it
        // got. The length is already bounded by whatever caps the librarian's generation
        // (max_tokens, or the server's own limit), and against a 128k window a long answer
        // is still far cheaper than the articles it saves reading.
        let answer = answer.trim();
        crate::trace::library_answer(question, answer);
        crate::debug::log("library answer", answer);

        if self.interactive {
            outln!(
                "{}Library answer:{} {}",
                SetForegroundColor(crate::out::DIM),
                ResetColor,
                answer
            );
        }

        let mut result = format!("Library answer for: {}\n\n{}", question, answer);
        if !sources.is_empty() {
            result.push_str("\n\nSections read:\n");
            for id in sources {
                result.push_str(&format!("- {}\n", id));
            }
        }
        // The prompt tells the librarian to say plainly when the library has nothing, so the
        // reader goes elsewhere rather than guessing — but a small model handed "not covered"
        // and nothing more tends to guess anyway, from memory, which is the one thing the
        // answer was meant to prevent. Naming where to go next is what makes it act on it.
        if says_not_covered(answer) {
            result.push_str(
                "\n\nThe library does not cover this. Look it up with research to get needed info.",
            );
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{serve_sequence, TempDir};
    use std::fs;

    fn library_with_select(label: &str) -> TempDir {
        let dir = TempDir::new(label);
        fs::create_dir_all(dir.join("tokio")).unwrap();
        fs::write(
            dir.join("tokio/select.md"),
            "# tokio::select!\n\n## Loops\n\nPin the future with tokio::pin! before the loop.\n",
        )
        .unwrap();
        dir
    }

    fn server_at(url: String, root: &Path) -> crate::MyMcpServer {
        let mut server = crate::MyMcpServer::new(
            crate::LMStudioClient::new(url, String::new(), None),
            true,
            None,
            false,
        );
        server.library_root = Some(root.to_path_buf());
        server
    }

    const SEARCHES: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"library_search","arguments":"{\"query\":\"pin select loop\"}"}}]}}]}"#;
    const READS: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c2","type":"function","function":{"name":"library_read","arguments":"{\"id\":\"tokio/select#0\"}"}}]}}]}"#;
    const ANSWERS: &str = r#"{"choices":[{"message":{"role":"assistant","content":"Pin it with tokio::pin! before the loop."}}]}"#;
    const CALLS_RESEARCH: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"research","arguments":"{\"question\":\"x\"}"}}]}}]}"#;

    #[tokio::test]
    async fn a_search_and_a_read_become_an_answer_with_its_sections() {
        let dir = library_with_select("library-tool-answer");
        let (url, served) = serve_sequence(&[SEARCHES, READS, ANSWERS]).await;
        let answer = server_at(url, dir.path())
            .run_library(dir.path(), "how do I use select! in a loop?")
            .await;

        assert_eq!(*served.lock().unwrap(), 3);
        assert!(answer.starts_with("Library answer for:"), "{}", answer);
        assert!(answer.contains("tokio::pin!"), "{}", answer);
        assert!(answer.contains("Sections read:\n- tokio/select#0"), "{}", answer);
    }

    #[tokio::test]
    async fn an_answer_the_library_does_not_cover_points_to_research() {
        const NOT_COVERED: &str = r#"{"choices":[{"message":{"role":"assistant","content":"The library does not cover mio's Registry."}}]}"#;
        let dir = library_with_select("library-tool-uncovered");
        let (url, _) = serve_sequence(&[NOT_COVERED]).await;
        let answer = server_at(url, dir.path()).run_library(dir.path(), "q").await;
        assert!(answer.ends_with("Look it up with research to get needed info."), "{}", answer);

        let (url, _) = serve_sequence(&[ANSWERS]).await;
        let answer = server_at(url, dir.path()).run_library(dir.path(), "q").await;
        assert!(!answer.contains("research"), "{}", answer);
    }

    #[tokio::test]
    async fn a_tool_outside_the_library_is_answered_rather_than_dropped() {
        let dir = library_with_select("library-tool-foreign");
        let (url, served) = serve_sequence(&[CALLS_RESEARCH, ANSWERS]).await;
        let answer = server_at(url, dir.path()).run_library(dir.path(), "q").await;
        assert_eq!(*served.lock().unwrap(), 2);
        assert!(answer.contains("tokio::pin!"), "{}", answer);
    }

    #[tokio::test]
    async fn the_lookup_backstop_stops_a_librarian_that_loops_on_instant_lookups() {
        let dir = library_with_select("library-tool-budget");
        let (url, served) = serve_sequence(&[SEARCHES]).await;
        let answer = server_at(url, dir.path()).run_library(dir.path(), "q").await;

        // A local server answers in milliseconds, so time never runs out here: the count does.
        // One request per lookup, then the forced answer asked with no tools.
        assert_eq!(*served.lock().unwrap(), MAX_LOOKUPS + 1);
        assert!(answer.contains("could not produce an answer"), "{}", answer);
    }

    #[tokio::test]
    async fn out_of_time_the_librarian_answers_after_its_first_lookup() {
        let dir = library_with_select("library-tool-time");
        let (url, served) = serve_sequence(&[SEARCHES, ANSWERS]).await;
        let answer = server_at(url, dir.path())
            .run_library_within(dir.path(), "q", Duration::ZERO)
            .await;

        // No time at all still buys the first lookup — without it there is nothing to answer
        // from — and then the forced answer, and nothing else.
        assert_eq!(*served.lock().unwrap(), 2);
        assert!(answer.contains("tokio::pin!"), "{}", answer);
    }

    #[tokio::test]
    async fn an_empty_library_or_question_never_reaches_the_model() {
        let empty = TempDir::new("library-tool-empty");
        let server = crate::testutil::test_server();
        let answer = server.run_library(empty.path(), "q").await;
        assert!(answer.contains("library is empty"), "{}", answer);

        let result = server
            .library(Parameters(LibraryParams { question: " ".into() }))
            .await
            .unwrap();
        assert!(result.content[0].as_text().unwrap().text.contains("empty"));

        // No root at all: the tool says where to go instead.
        let result = server
            .library(Parameters(LibraryParams { question: "q".into() }))
            .await
            .unwrap();
        assert!(result.content[0].as_text().unwrap().text.contains("crate_api"));
    }

    #[tokio::test]
    async fn an_unreachable_model_is_reported_as_advice() {
        let dir = library_with_select("library-tool-down");
        let answer = crate::testutil::test_server().run_library(dir.path(), "q").await;
        assert!(answer.contains("could not be reached"), "{}", answer);
    }
}
