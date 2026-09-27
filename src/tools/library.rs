//! The `library` sub-agent: a question goes in, an answer from the library's articles comes
//! out, and the articles read to get there never touch the main conversation.
//!
//! The same shape as `research`, for the same reason — an article is thousands of tokens,
//! and three of them in the main history push a 32k model's own working memory out of the
//! window. The differences are what it reads and how cheaply: the articles are local
//! markdown (`crate::library`), so a lookup is a file read rather than a fetch, and the
//! round trips are the whole cost.
//!
//! **Its two tools are not router tools.** `library_search` and `library_read` exist only
//! inside this loop, defined here as JSON and dispatched here, so the main model never sees
//! them and cannot spend its own context reading articles directly — which is the one thing
//! this tool exists to prevent. It also means `library` can never appear in its own list.

use super::MyMcpServer;
use crate::library::Library;
use crate::session::without_think_block;
use crate::trace::Caller;
use crate::util::truncate_at_boundary;
use crossterm::style::{Color, ResetColor, SetForegroundColor};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

const SEARCH: &str = "library_search";
const READ: &str = "library_read";

// Searches and section reads together. Fewer than research's ten: a section is chosen from
// a ranked list with its heading already visible, so there is much less reading of the
// wrong thing to budget for.
const LOOKUP_BUDGET: usize = 8;

// The round that calls no tool and says nothing either — see research's MAX_ROUNDS.
const MAX_ROUNDS: usize = LOOKUP_BUDGET + 2;

// Longer than research's answer may be, because what comes back from here is usually code
// the model is about to copy, and a cut-off example is worse than none.
const MAX_ANSWER_CHARS: usize = 6000;

const MAX_ECHOED_QUESTION: usize = 200;

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
         - Stop as soon as you can answer. You have at most {budget} lookups (searches and \
           reads together); when they run out you must answer from what you have read.\n\n\
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
        budget = LOOKUP_BUDGET,
    )
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
            outln!("{}{}{}", SetForegroundColor(Color::DarkGrey), line, ResetColor);
        }
    }

    // The loop. Never returns an error, for research's reason: what went wrong is still
    // something to tell the model in words it can act on.
    pub(crate) async fn run_library(&self, root: &Path, question: &str) -> String {
        self.library_note(&format!(
            "Library lookup for: {}",
            truncate_at_boundary(question, MAX_ECHOED_QUESTION)
        ));
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

        for _ in 0..MAX_ROUNDS {
            if spent >= LOOKUP_BUDGET {
                break;
            }

            let message = match client
                .chat_completion_streaming(Caller::Library, messages.clone(), tools.clone(), |_| {})
                .await
            {
                Ok(message) => message,
                Err(e) => {
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
                } else if spent >= LOOKUP_BUDGET {
                    format!(
                        "The budget of {} lookups is spent. Answer the question now, from what \
                         you have already read.",
                        LOOKUP_BUDGET
                    )
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

        // Out of budget, or stalled: one more request with no tools at all, so there is
        // nothing left to call — research's ending, for research's reason.
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
        let answer = truncate_at_boundary(answer.trim(), MAX_ANSWER_CHARS);
        crate::debug::log("library answer", answer);

        if self.interactive {
            outln!(
                "{}Library answer:{} {}",
                SetForegroundColor(Color::DarkGrey),
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
    async fn a_tool_outside_the_library_is_answered_rather_than_dropped() {
        let dir = library_with_select("library-tool-foreign");
        let (url, served) = serve_sequence(&[CALLS_RESEARCH, ANSWERS]).await;
        let answer = server_at(url, dir.path()).run_library(dir.path(), "q").await;
        assert_eq!(*served.lock().unwrap(), 2);
        assert!(answer.contains("tokio::pin!"), "{}", answer);
    }

    #[tokio::test]
    async fn the_lookup_budget_stops_a_librarian_that_never_answers() {
        let dir = library_with_select("library-tool-budget");
        let (url, served) = serve_sequence(&[SEARCHES]).await;
        let answer = server_at(url, dir.path()).run_library(dir.path(), "q").await;

        // One request per lookup, then the forced answer asked with no tools.
        assert_eq!(*served.lock().unwrap(), LOOKUP_BUDGET + 1);
        assert!(answer.contains("could not produce an answer"), "{}", answer);
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
