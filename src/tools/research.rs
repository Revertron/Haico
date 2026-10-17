//! The research sub-agent: a question goes in, an answer comes out, and the pages read to
//! get there never touch the main conversation.
//!
//! The other two sub-agents in haico — the auto-mode reviewer in `modes.rs` and the
//! reasoning compressor in `session.rs` — are single round trips: ask, get an answer. This
//! one is a small agentic loop of its own, because looking something up on the web is not
//! one question. It is a search, a page that turns out to be the wrong version, another
//! search, and then the paragraph that actually answers it.
//!
//! Doing that in the main conversation costs the main conversation. Every search result
//! block and every 60 KB page stays in the history for the rest of the session. On the
//! 128k–262k windows haico runs against they would fit; the cost is everything after — each
//! later request carries them, a lost prefix cache re-prefills them, compaction comes sooner,
//! and a small model's attention is spread over pages about a question settled long ago.
//! Here they land in a message list that is thrown away when the call returns; what the
//! model gets back is the answer and the URLs it came from, a few hundred tokens at most.
//!
//! What this does *not* buy is time or compute. The server still prefills every page, once
//! per round, and on a local model a full ten-lookup research call is minutes rather than
//! seconds. That is the trade: the main context is the scarce thing, not the GPU's patience.
//! It is also why this runs on the same client as everything else — a second model would
//! want a second slot on a GPU that is already full.

use crate::trace::Caller;
use super::MyMcpServer;
use crate::session::without_think_block;
use crate::util::truncate_at_boundary;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// The only two tools the sub-agent gets. Named here rather than filtered by "anything web"
// so that adding a web tool later does not silently widen what research can reach — and so
// that `research` itself can never appear in its own tool list.
pub(crate) const RESEARCH_TOOLS: &[&str] = &["web_search", "web_get"];

// How many lookups — searches and page reads together — one research call may spend.
const LOOKUP_BUDGET: usize = 10;

// Rounds of model → tools → model. The budget above bounds every round that calls a tool,
// so this only exists for the round that calls none and says nothing either: without it,
// a model that answers with an empty message loops forever.
const MAX_ROUNDS: usize = LOOKUP_BUDGET + 2;

// How much of one page the sub-agent is shown: as much as `web_get` returns at all, 64 KB,
// the user's choice. It used to be 8000, which kept each round cheap — every round re-sends
// every page read so far — but hid the answer whenever it was further down a long page,
// with no way to read on. The cost is the other way round now: ten full pages are ~160k
// tokens re-sent each round, more than a 131k window holds.
const MAX_PAGE_CHARS: usize = 64 * 1024;


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct ResearchParams {
    #[schemars(
        description = "The question to research. One specific question, naming the library, \
                       tool and version it is about — not a topic."
    )]
    pub(crate) question: String,
}

// What the sub-agent is told.
//
// Written for a small local model driving two tools with nobody watching: every instruction
// is about when to stop rather than what to try, because the failure mode here is not a bad
// search, it is a model that keeps searching after it already has the answer.

fn research_prompt() -> String {
    format!(
        "You are a research assistant. You are given one question and two tools: \
         web_search and web_get. Find the answer on the web and report it.\n\n\
         How to work:\n\
         - Start with web_search, unless the question already contains a URL to open.\n\
         - Open the most promising results with web_get. Prefer official documentation, a \
           project's own repository and its release notes over blog posts and forum answers.\n\
         - If a page turns out to be about a different version than the one asked about, say \
           so and look for the right one rather than answering from the wrong one.\n\
         - Stop as soon as you can answer. You have at most {} lookups (searches and page \
           reads together) for the whole task; when they run out you must answer from what \
           you have already read.\n\n\
         When you have the answer, reply with text and no tool call:\n\
         - Answer the question directly, in a few sentences.\n\
         - Include the exact code, signature, flag or version number the answer turns on, \
           if there is one.\n\
         - Say which version the answer applies to whenever that matters.\n\
         - If the web did not answer it, say that plainly. A clear \"the documentation does \
           not cover this\" is worth more than a confident guess, because the person reading \
           your answer is going to write code from it.\n\n\
         Do not ask questions back — there is nobody to answer them. Do not describe what \
         you are about to do: either call a tool or give the answer.",
        LOOKUP_BUDGET
    )
}

#[tool_router(router = research_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(
        description = "Ask a question about anything outside this machine and get back the \
                       answer and its sources. A research assistant runs the searches and reads \
                       the pages itself, so the pages never enter your context. Use this \
                       INSTEAD of web_search and web_get whenever any of these is true: you do \
                       not know a library's API, syntax or current usage; an error message or \
                       compiler complaint means nothing to you; you need to know what changed \
                       between two versions; a documentation page you tried did not answer it; \
                       or you have already made one web call about this and are about to make \
                       another. That last one is the important one — two lookups on the same \
                       question is the signal to hand it over, not to keep going. Ask one \
                       specific question naming the library and version, like \"what is the \
                       catch-all route pattern in actix-web 4?\", not a topic. Only use \
                       web_get instead when you already know the exact URL that holds the \
                       answer."
    )]
    pub(crate) async fn research(
        &self,
        params: Parameters<ResearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let question = params.0.question.trim().to_string();
        if question.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(
                "The question was empty. Put the question itself in the 'question' field \
                 and call research again.",
            )]));
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(
            self.run_research(&question).await,
        )]))
    }
}

impl MyMcpServer {
    // One line of progress, in the same grey the tool call lines use.
    //
    // Only in the REPL: in MCP server mode stdout carries the protocol, and a stray line on
    // it is a parse error at the host rather than a cosmetic problem.
    // The loop itself.
    //
    // Never returns an error: a research call that failed is still something to tell the
    // model about in words it can act on, and an `Err` here would surface as a tool failure
    // that says nothing about whether to try a different question or give up on the web.
    async fn run_research(&self, question: &str) -> String {
        // Named on the call line instead, for the library's reason (see `run_library_within`).
        crate::debug::log("research question", question);

        // One sub-agent copy for the whole loop, so the main conversation's thinking level
        // does not ride along. This is where it would cost the most: up to ten rounds, each
        // one prefilling every page fetched so far, and none of the thinking kept — what
        // comes back is the answer and its sources. `sub_agent_effort` is where to turn it
        // down for a model that takes a level.
        let client = self.llm_client.for_sub_agent();

        // Built from the real router rather than written out here, so the descriptions the
        // sub-agent reads are the same ones the main model reads.
        let tools: Vec<Value> = super::mcp_tools_to_openai_format(self)
            .into_iter()
            .filter(|tool| {
                RESEARCH_TOOLS.contains(&tool["function"]["name"].as_str().unwrap_or_default())
            })
            .collect();

        let mut messages = vec![
            json!({"role": "system", "content": research_prompt()}),
            json!({"role": "user", "content": question}),
        ];

        // URLs actually opened, in the order they were opened, so the answer can carry its
        // sources back. The page cache is shared with the main model's own web_get, so a
        // source it decides to read for itself afterwards costs nothing to fetch again.
        let mut sources: Vec<String> = Vec::new();
        let mut spent = 0usize;

        for _ in 0..MAX_ROUNDS {
            if spent >= LOOKUP_BUDGET {
                break;
            }

            // Streaming with a no-op sink: this is the only call in the client that sends a
            // tools array, and the sub-agent's own text is not for the screen. It also
            // carries the fallback for a server that ignores "stream", which a hand-rolled
            // request here would have to repeat.
            let message = match client
                .chat_completion_streaming(Caller::Research, messages.clone(), tools.clone(), |_| {})
                .await
            {
                Ok(message) => message,
                // A reply cut off for going round in a loop is not a server that cannot be
                // reached, which is what it used to be reported as. Whatever was read so far
                // is still worth an answer, so it goes to the forced one below — the request
                // with no tools, which is the least likely to loop the same way again.
                Err(e) if e.downcast_ref::<crate::repetition::Repeating>().is_some() => {
                    crate::debug::log("research reply repeated itself", &format!("{:#}", e));
                    break;
                }
                Err(e) => {
                    crate::debug::log("research failed", &format!("{:#}", e));
                    return format!(
                        "The research assistant could not be reached: {}. Nothing was looked \
                         up. Answer from what you know, or use web_search and web_get \
                         yourself.",
                        e
                    );
                }
            };

            let calls = message["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default();

            if calls.is_empty() {
                let answer = message["content"].as_str().unwrap_or("").trim();
                if !answer.is_empty() {
                    return self.finish(question, answer, &sources);
                }
                // No call and no words. Nothing here will improve by asking again the same
                // way, so fall through to the forced answer below.
                break;
            }

            messages.push(message.clone());

            // Every tool_call_id gets a reply, on every path — the same invariant the main
            // loop lives by. An unanswered id makes the *next* request fail server-side, and
            // here that would strand the loop with no way to say why.
            for call in &calls {
                // As the value it came as, the way the main loop answers it: a missing id
                // stays missing rather than becoming "", which matches nothing either.
                let id = call["id"].clone();
                let name = call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();

                let content = if !RESEARCH_TOOLS.contains(&name.as_str()) {
                    // The descriptions of web_search and web_get both mention web_fetch,
                    // which is not on the table here, so this is a call the sub-agent will
                    // genuinely make rather than a theoretical one.
                    format!(
                        "'{}' is not available for this task. You have exactly two tools: \
                         web_search and web_get.",
                        name
                    )
                } else if spent >= LOOKUP_BUDGET {
                    format!(
                        "The budget of {} lookups is spent. Answer the question now, from \
                         what you have already read.",
                        LOOKUP_BUDGET
                    )
                } else {
                    match super::parse_tool_arguments(&call["function"]) {
                        Ok(arguments) => {
                            spent += 1;
                            self.run_lookup(&name, arguments, &mut sources).await
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

        // The budget ran out, or the sub-agent stalled. Ask once more with no tools at all:
        // a request carrying a tools array invites another tool call, which is exactly what
        // there is no longer any budget for.
        messages.push(json!({
            "role": "user",
            "content": format!(
                "Stop looking things up and answer now: {}\n\nUse only what you have already \
                 read. If it did not answer the question, say so plainly rather than guessing.",
                question
            ),
        }));

        match client.chat_completion_plain(Caller::Research, messages, None).await {
            Ok(answer) => self.finish(question, without_think_block(&answer), &sources),
            Err(e) => {
                crate::debug::log("research answer failed", &format!("{:#}", e));
                format!(
                    "The research assistant read {} page(s) but could not produce an answer: \
                     {}. {}",
                    sources.len(),
                    e,
                    if sources.is_empty() {
                        "Try web_search yourself.".to_string()
                    } else {
                        format!("These pages looked relevant: {}", sources.join(", "))
                    }
                )
            }
        }
    }

    // Run one of the two allowed tools and return what the sub-agent should see.
    //
    // Dispatch goes through `execute_tool` rather than calling the tool methods directly, so
    // there is still exactly one place that knows how to turn a name and arguments into a
    // result. The name has already been checked against `RESEARCH_TOOLS` by the caller,
    // which is what keeps `research` from reaching anything else through this door.
    async fn run_lookup(
        &self,
        name: &str,
        arguments: Value,
        sources: &mut Vec<String>,
    ) -> String {
        let is_page = name == "web_get";

        if is_page {
            let url = arguments["url"].as_str().unwrap_or_default().to_string();
            crate::progress::step(format!("Checking {}", url));
            if !url.is_empty() && !sources.contains(&url) {
                sources.push(url);
            }
        } else {
            crate::progress::step(format!("Searching: {}", arguments["query"].as_str().unwrap_or_default()));
        }

        // `.text` and not the whole output: the two tools reachable from here
        // return prose, and an image part could not go into this sub-agent's
        // message list anyway — it has no user message to put one in.
        match self.execute_tool(name, arguments).await.map(|output| output.text) {
            Ok(text) if is_page && text.len() > MAX_PAGE_CHARS => {
                format!(
                    "{}\n\n[Cut off at {} characters. If the answer was further down, say so \
                     and open a more specific page rather than this one again.]",
                    truncate_at_boundary(&text, MAX_PAGE_CHARS),
                    MAX_PAGE_CHARS
                )
            }
            Ok(text) => text,
            // Handed back rather than raised: a search that was blocked or a page that 404s
            // is information the sub-agent can act on by trying something else.
            Err(e) => format!("Error: {}", e),
        }
    }

    // Show the answer, and shape the text the main model receives.
    fn finish(&self, question: &str, answer: &str, sources: &[String]) -> String {
        // Never cut, for the library's reason: an answer stopped mid-way is worse than a long
        // one, and the model acts on the half it got. It used to be cut at 4000 characters.
        let answer = answer.trim();
        crate::trace::research_answer(question, answer, sources);
        crate::debug::log("research answer", answer);

        crate::progress::answer("Got answer", answer);

        let mut result = format!("Research answer for: {}\n\n{}", question, answer);
        if !sources.is_empty() {
            result.push_str("\n\nSources read:\n");
            for url in sources {
                result.push_str(&format!("- {}\n", url));
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::test_server;

    fn names_of(tools: &[serde_json::Value]) -> Vec<String> {
        tools
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    #[test]
    fn the_sub_agent_gets_two_tools_and_never_itself() {
        // Built the way the REPL builds it, so `research` really is on the main model's
        // list — a filter that excluded it because it was never there would prove nothing.
        let server = server_at("http://127.0.0.1:1".to_string());
        let all = super::super::mcp_tools_to_openai_format(&server);
        assert!(names_of(&all).contains(&"research".to_string()));

        let offered: Vec<serde_json::Value> = all
            .into_iter()
            .filter(|tool| {
                RESEARCH_TOOLS.contains(&tool["function"]["name"].as_str().unwrap_or_default())
            })
            .collect();
        let names = names_of(&offered);

        assert_eq!(names.len(), 2, "got {:?}", names);
        assert!(names.contains(&"web_search".to_string()));
        assert!(names.contains(&"web_get".to_string()));
        // The whole point of naming the two tools rather than filtering on "web": research
        // must not be able to call itself.
        assert!(!names.contains(&"research".to_string()));
    }

    #[test]
    fn research_is_not_offered_when_there_is_no_llm_to_run_it() {
        // test_server is built the way MCP server mode builds it without --url: no LLM
        // endpoint, so the two tools that need one are not advertised at all.
        let names = names_of(&super::super::mcp_tools_to_openai_format(&test_server()));
        assert!(!names.contains(&"research".to_string()), "got {:?}", names);
        assert!(!names.contains(&"web_fetch".to_string()));
        assert!(names.contains(&"web_get".to_string()));
    }

    #[tokio::test]
    async fn an_empty_question_is_refused_without_reaching_the_model() {
        // test_server points at a port nothing listens on, so a call that did reach the
        // model would fail rather than return this.
        let server = test_server();
        let result = server
            .research(Parameters(ResearchParams {
                question: "   ".to_string(),
            }))
            .await
            .unwrap();

        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("empty"), "got: {}", text);
    }

    // A model that asks for a tool it does not have, and then answers.
    //
    // web_fetch is the one it will actually reach for: the descriptions of both allowed
    // tools name it, and it is not on the table here.
    const CALLS_A_TOOL_IT_DOES_NOT_HAVE: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"web_fetch","arguments":"{\"url\":\"https://example.com\",\"prompt\":\"x\"}"}}]}}]}"#;

    const ANSWERS: &str = r#"{"choices":[{"message":{"role":"assistant","content":"In actix-web 4 the catch-all is .default_service(web::route().to(handler))."}}]}"#;

    // Always asks for another page, and never answers. The URL is a port nothing listens
    // on, so the lookup fails immediately and without leaving the machine.
    const ALWAYS_FETCHES: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"web_get","arguments":"{\"url\":\"http://127.0.0.1:1/nothing\"}"}}]}}]}"#;

    fn server_at(url: String) -> crate::MyMcpServer {
        crate::MyMcpServer::new(
            crate::LMStudioClient::new(url, String::new(), None),
            true,
            None,
            // Not interactive, as an MCP server is: research's own two tools are the same
            // either way, and its progress goes nowhere outside a turn's scope.
            false,
        )
    }

    #[tokio::test]
    async fn a_tool_the_sub_agent_does_not_have_is_answered_rather_than_dropped() {
        let (url, served) =
            crate::testutil::serve_sequence(&[CALLS_A_TOOL_IT_DOES_NOT_HAVE, ANSWERS]).await;
        let answer = server_at(url)
            .run_research("what is the catch-all route in actix-web 4?")
            .await;

        // Two rounds: the refusal went back as a tool message the model could act on,
        // rather than orphaning its tool_call_id and failing the next request.
        assert_eq!(*served.lock().unwrap(), 2);
        assert!(answer.contains("Research answer for:"), "got: {}", answer);
        assert!(answer.contains("default_service"), "got: {}", answer);
    }

    #[tokio::test]
    async fn a_research_reply_that_loops_is_answered_from_what_was_read() {
        let bodies: &'static [&'static str] =
            Box::leak(Box::new([crate::testutil::looping_stream(), ANSWERS]));
        let (url, served) = crate::testutil::serve_sequence(bodies).await;
        let answer = server_at(url).run_research("what is the catch-all route in actix-web 4?").await;

        // The loop was cut off, and the next request was the forced answer — not a
        // "could not be reached", which is what a looping reply used to be told as.
        assert_eq!(*served.lock().unwrap(), 2);
        assert!(!answer.contains("could not be reached"), "got: {}", answer);
        assert!(answer.contains("default_service"), "got: {}", answer);
    }

    #[tokio::test]
    async fn the_lookup_budget_stops_a_sub_agent_that_never_answers() {
        let (url, served) = crate::testutil::serve_sequence(&[ALWAYS_FETCHES]).await;
        let answer = server_at(url).run_research("something unanswerable").await;

        // One request per lookup, then exactly one more: the forced answer, asked with no
        // tools at all so there is nothing left to call.
        assert_eq!(*served.lock().unwrap(), LOOKUP_BUDGET + 1);
        // That last request got the same "call web_get" body back, which carries no answer,
        // so what the model is told is what actually happened rather than an invention.
        assert!(answer.contains("could not produce an answer"), "got: {}", answer);
    }

    #[tokio::test]
    async fn an_unreachable_model_is_reported_as_advice_not_as_an_error() {
        let server = test_server();
        let answer = server.run_research("what is the catch-all route in actix-web 4?").await;

        // Never Err: the model is told what happened in terms it can act on.
        assert!(answer.contains("could not be reached"), "got: {}", answer);
        assert!(answer.contains("web_search"), "got: {}", answer);
    }
}
