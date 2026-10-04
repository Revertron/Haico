//! The general sub-agent: the model hands over a task, a helper with an empty context does
//! it with its own tools, and only the helper's report comes back.
//!
//! The other sub-agents each do one fixed job: the reviewer judges a call, the compressors
//! shorten a trace or a history, research looks something up on the web, the librarian
//! reads articles. This one does whatever it is asked, and what it buys is the same thing
//! research buys, spent on the local machine instead of the web: a short main history. A
//! 4,000-line build log read to find one error, a directory read to answer "who calls X",
//! a test run read to list what failed — done in the main loop, all of that stays in the
//! history for the rest of the session, re-sent with every request and re-prefilled with
//! every lost cache. Here it lands in a message list dropped when the call returns.
//!
//! **Every call the helper makes is cleared through `agent::authorize`, the main loop's own
//! gate.** That is why this lives on the core's side of the seam rather than in
//! `tools/` beside research: the workspace boundary, the permission rules, plan mode, the
//! auto-mode reviewer and the user's prompt all live on the agent, and a call made one
//! level down is still a call on this machine. A helper that bypassed them would be a way
//! for the model to read what it was refused, or run what it was never allowed to.
//!
//! **Its tools are named in `SUB_AGENT_TOOLS`, never filtered by "everything read-only".**
//! No file writes: the main model edits from contents it has read itself, and a change
//! made down here would be one it only knows about from a summary. No `sub_agent`,
//! `research` or anything else that starts a loop of its own. `run_command` is on the list
//! because digesting command output is half of what this is for, and it meets its gate
//! like any other call.
//!
//! **There is no cap on calls or on the report.** What bounds the loop is what bounds the
//! main one in auto mode — the identical-call guard and the repetition guard on the
//! stream — plus the window itself: once the helper's own history reaches the auto-compact
//! line, it is made to answer from what it has. Esc ends it like any other wait.

use serde_json::{json, Value};

use crate::agent::{authorize, push_user_note, Agent, LOOP_BREAK, LOOP_NUDGE};
use crate::config::autocompact_threshold;
use crate::prompt::build_sub_agent_prompt;
use crate::repetition::Repeating;
use crate::session::{estimate_tokens, without_think_block};
use crate::tools::{parse_tool_arguments, repair_leaked_arguments};
use crate::trace::Caller;
use crate::turn::*;

/// The tools the helper is offered. See the module header for why each kind is left out.
pub(crate) const SUB_AGENT_TOOLS: &[&str] = &[
    "read",
    "grep",
    "glob_search",
    "ls",
    "run_command",
    "background_output",
    "recall",
    "library",
    "crate_info",
    "crate_search",
    "crate_api",
    "web_search",
    "web_get",
    "calculator",
    "get_date",
    "get_time",
];

// Replies in a row cut off for repeating themselves before the helper is made to answer
// instead. The main loop's own figure, for its reason.
const MAX_REPEATING_REPLIES: usize = 2;

/// Run one task to its report. `None` means the user interrupted it, and the caller owes
/// its own `tool_call_id` the interrupted reply.
///
/// Never an error otherwise: a helper that failed is still something to tell the model in
/// words it can act on — "nothing was read, do it yourself" — where a bare failure says
/// nothing about whether to try again.
pub(crate) async fn run_sub_agent(
    agent: &mut Agent,
    task: &str,
    tx: &EventSink,
    link: &TurnLink,
) -> Option<String> {
    crate::debug::log("sub-agent task", task);

    // A copy for the whole loop, so the main conversation's thinking level does not ride
    // along — `sub_agent_effort` is where to turn it down, as for research.
    let client = agent.client.for_sub_agent();

    // The mode's own list first: in plan mode `run_command` is not offered here either,
    // and a tool the helper cannot see is one it does not spend a round on.
    let tools: Vec<Value> = crate::modes::tools_for_mode(link.mode(), &agent.tools)
        .into_iter()
        .filter(|tool| SUB_AGENT_TOOLS.contains(&tool["function"]["name"].as_str().unwrap_or_default()))
        .collect();
    let offered: Vec<String> = tools
        .iter()
        .filter_map(|tool| tool["function"]["name"].as_str().map(str::to_string))
        .collect();

    // The workspace note goes in for the reason it goes into every main request: a refusal
    // the model was never warned of is one it retries with a different path.
    let system = format!(
        "{}\n{}",
        build_sub_agent_prompt(&agent.client.model, &agent.workspace),
        agent.workspace.note()
    );
    let mut messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": task}),
    ];

    // The server's count of the last request and the estimate it was sent with, so the
    // window check below is measured rather than guessed — `Agent::context_used`'s reason.
    let mut measured: Option<(usize, usize)> = None;
    let tools_estimate = estimate_tokens(&tools);
    let limit = autocompact_threshold(agent.context_tokens);

    let mut calls_made = 0usize;
    let mut repeating_replies = 0usize;
    let mut last_call: Option<(String, String)> = None;
    let mut run = 0usize;
    let mut nudged_at = 0usize;

    // Why the loop stopped without a report, said to the helper in the forced request.
    let stop_reason = loop {
        let estimate = estimate_tokens(&messages);
        let used = match measured {
            Some((prompt_tokens, then)) if estimate >= then => prompt_tokens + (estimate - then),
            _ => estimate + tools_estimate,
        };
        if used > limit {
            break "Your context is nearly full.";
        }

        let reply = interruptible(
            tx,
            link,
            Waiting::Tool("sub_agent".to_string()),
            client.chat_completion_measured(Caller::SubAgent, messages.clone(), tools.clone(), |_| {}),
        )
        .await?;

        let message = match reply {
            Ok((message, ending)) => {
                repeating_replies = 0;
                if let Some(tokens) = ending.usage.as_ref().and_then(|u| u["prompt_tokens"].as_u64()) {
                    measured = Some((tokens as usize, estimate));
                }
                message
            }
            Err(e) => {
                if let Some(repeating) = e.downcast_ref::<Repeating>() {
                    repeating_replies += 1;
                    if repeating_replies >= MAX_REPEATING_REPLIES {
                        break "Your replies keep repeating themselves.";
                    }
                    push_user_note(
                        &mut messages,
                        &format!(
                            "Your previous reply was cut off and discarded because it had \
                             started repeating the same passage over and over:\n\n> {}\n\n\
                             Decide on one concrete next step and take it now, or give your \
                             report.",
                            repeating.passage
                        ),
                    );
                    continue;
                }
                crate::debug::log("sub-agent failed", &format!("{:#}", e));
                return Some(format!(
                    "The sub-agent failed after {} tool call(s): {:#}. Nothing it read reached \
                     you. Do the task yourself with the other tools, or try sub_agent again if \
                     the error looks temporary.",
                    calls_made, e
                ));
            }
        };

        let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
        if calls.is_empty() {
            let content = message["content"].as_str().unwrap_or("");
            let report = without_think_block(content);
            if !report.is_empty() {
                return Some(finish(report, calls_made));
            }
            // No call and no words. Asking the same way again would most likely get the
            // same nothing, so ask the way that sends no tools.
            break "You replied with nothing.";
        }

        messages.push(message.clone());

        // Every tool_call_id gets a reply, on every path, before anything else can happen:
        // an unanswered one makes the next request fail server-side.
        for call in &calls {
            let id = call["id"].clone();
            let name = call["function"]["name"].as_str().unwrap_or_default().to_string();
            let parsed = parse_tool_arguments(&call["function"]);

            // The main loop's identical-call guard, keyed the same way.
            let key = match &parsed {
                Ok(arguments) => arguments.to_string(),
                Err(_) => call["function"]["arguments"].to_string(),
            };
            let this = (name.clone(), key);
            if last_call.as_ref() == Some(&this) {
                run += 1;
            } else {
                last_call = Some(this);
                run = 1;
                nudged_at = 0;
            }

            let content = if !offered.contains(&name) {
                format!(
                    "'{}' is not available to you. Your tools are: {}.",
                    name,
                    offered.join(", ")
                )
            } else {
                match parsed {
                    Err(detail) => format!(
                        "Could not run '{}': {}. Call it again with valid JSON arguments.",
                        name, detail
                    ),
                    Ok(mut arguments) => {
                        repair_leaked_arguments(&mut arguments);
                        match authorize(agent, tx, link, &name, &arguments).await {
                            Err(refusal) => refusal,
                            Ok(_) => {
                                let _ = tx.send(
                                    Notice::SubAgentStep { tool: name.clone(), arguments: arguments.clone() }
                                        .into(),
                                );
                                calls_made += 1;
                                let executed = interruptible(
                                    tx,
                                    link,
                                    Waiting::Tool(name.clone()),
                                    agent.server.execute_tool(&name, arguments),
                                )
                                .await?;
                                match executed {
                                    // An image cannot go into a tool reply, and the helper's
                                    // report is text anyway: it is told the picture exists.
                                    Ok(output) if !output.images.is_empty() => format!(
                                        "{}\n\n[{} image(s) not shown: you work from text only.]",
                                        output.text,
                                        output.images.len()
                                    ),
                                    Ok(output) => output.text,
                                    Err(e) => format!("Error: {}", e),
                                }
                            }
                        }
                    }
                }
            };

            messages.push(json!({"role": "tool", "tool_call_id": id, "content": content}));
        }

        if run >= LOOP_BREAK {
            break "You have been making the same call over and over.";
        }
        if run >= LOOP_NUDGE && run > nudged_at {
            nudged_at = run;
            messages.push(json!({
                "role": "user",
                "content": format!(
                    "You have made the same '{}' call {} times in a row. Calling it again will \
                     not tell you anything new. Take a different approach, or give your report.",
                    last_call.as_ref().map(|(name, _)| name.as_str()).unwrap_or_default(),
                    run
                ),
            }));
        }
    };

    // Asked with no tools array at all: a request carrying one invites another call, which
    // is exactly what is over.
    messages.push(json!({
        "role": "user",
        "content": format!(
            "{} Stop now and give your report on the task from what you have already found. \
             Say plainly what you did not get to.",
            stop_reason
        ),
    }));
    let answer = interruptible(
        tx,
        link,
        Waiting::Tool("sub_agent".to_string()),
        client.chat_completion_plain(Caller::SubAgent, messages, None),
    )
    .await?;

    Some(match answer {
        Ok(answer) if !without_think_block(&answer).is_empty() => {
            finish(without_think_block(&answer), calls_made)
        }
        Ok(_) => format!(
            "The sub-agent made {} tool call(s) and then gave no report ({}). Do the task \
             yourself, or hand it over again more narrowly.",
            calls_made,
            stop_reason.trim_end_matches('.').to_lowercase()
        ),
        Err(e) => format!(
            "The sub-agent made {} tool call(s) but could not produce a report: {:#}. Do the \
             task yourself, or hand it over again more narrowly.",
            calls_made, e
        ),
    })
}

// The text the main model receives.
fn finish(report: &str, calls_made: usize) -> String {
    crate::debug::log("sub-agent report", report);
    format!(
        "{}\n\n[haico] Report from the sub-agent ({} tool call(s)). What it read is not in your \
         context: read a file yourself before editing it.",
        report.trim(),
        calls_made
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::Mode;
    use crate::testutil::test_config;

    fn agent_at(url: &str) -> Agent {
        let cwd = std::env::current_dir().unwrap();
        Agent::new(test_config(url), "m".to_string(), &cwd)
    }

    // Drains everything the loop sent, so the test can look at what was shown.
    fn sink() -> (EventSink, tokio::sync::mpsc::UnboundedReceiver<AgentEvent>) {
        tokio::sync::mpsc::unbounded_channel()
    }

    fn steps(rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEvent>) -> Vec<String> {
        let mut seen = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let AgentEvent::Notice(Notice::SubAgentStep { tool, .. }) = event {
                seen.push(tool);
            }
        }
        seen
    }

    const READS_CARGO_TOML: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{\"file_path\":\"Cargo.toml\"}"}}]}}]}"#;
    const REPORTS: &str = r#"{"choices":[{"message":{"role":"assistant","content":"The package is named haico."}}]}"#;
    const WRITES_A_FILE: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"write_file","arguments":"{\"file_path\":\"x.txt\",\"content\":\"x\"}"}}]}}]}"#;
    const READS_OUTSIDE: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{\"file_path\":\"/definitely/outside/secret.txt\"}"}}]}}]}"#;

    #[test]
    fn the_helper_gets_no_writes_and_never_itself() {
        let agent = agent_at("http://127.0.0.1:1");
        let names: Vec<&str> = agent
            .tools
            .iter()
            .filter_map(|tool| tool["function"]["name"].as_str())
            .collect();
        // On the main model's list, so leaving it out below is a decision, not an accident.
        assert!(names.contains(&"sub_agent"), "got {:?}", names);
        for left_out in ["sub_agent", "write_file", "edit", "multi_edit", "research", "run_background", "remember"] {
            assert!(!SUB_AGENT_TOOLS.contains(&left_out), "{left_out} must not be offered");
        }
    }

    #[tokio::test]
    async fn a_report_comes_back_with_what_it_read_left_behind() {
        let (url, served) = crate::testutil::serve_sequence(&[READS_CARGO_TOML, REPORTS]).await;
        let mut agent = agent_at(&url);
        let history = agent.messages.len();
        let (tx, mut rx) = sink();
        let link = TurnLink::new(Mode::Normal);

        let report = run_sub_agent(&mut agent, "What is the package called?", &tx, &link)
            .await
            .expect("not interrupted");

        assert_eq!(*served.lock().unwrap(), 2);
        assert!(report.starts_with("The package is named haico."), "got: {}", report);
        assert!(report.contains("1 tool call"), "got: {}", report);
        assert_eq!(steps(&mut rx), vec!["read".to_string()]);
        // The whole point: the file it read went into its own list, not the agent's.
        assert_eq!(agent.messages.len(), history);
    }

    #[tokio::test]
    async fn a_tool_it_was_not_given_is_answered_and_never_run() {
        let (url, served) = crate::testutil::serve_sequence(&[WRITES_A_FILE, REPORTS]).await;
        let mut agent = agent_at(&url);
        let (tx, mut rx) = sink();
        let link = TurnLink::new(Mode::Normal);

        let report = run_sub_agent(&mut agent, "task", &tx, &link).await.unwrap();

        // Two requests: the refusal went back under the call's id and the helper went on.
        assert_eq!(*served.lock().unwrap(), 2);
        assert!(report.contains("haico"), "got: {}", report);
        assert!(steps(&mut rx).is_empty());
        assert!(!std::path::Path::new("x.txt").exists());
    }

    #[tokio::test]
    async fn the_workspace_boundary_holds_one_level_down() {
        let (url, _) = crate::testutil::serve_sequence(&[READS_OUTSIDE, REPORTS]).await;
        let mut agent = agent_at(&url);
        let (tx, mut rx) = sink();
        let link = TurnLink::new(Mode::Normal);

        run_sub_agent(&mut agent, "task", &tx, &link).await.unwrap();

        let mut refused = false;
        let mut ran = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                AgentEvent::Notice(Notice::Refused(reason)) => {
                    refused = reason.contains("outside the workspace")
                }
                AgentEvent::Notice(Notice::SubAgentStep { .. }) => ran = true,
                _ => {}
            }
        }
        assert!(refused, "the read outside the workspace was not refused");
        assert!(!ran, "a refused call must not run");
    }

    #[tokio::test]
    async fn the_loop_guard_makes_a_stuck_helper_answer() {
        // Reads the same file forever; the last body served repeats, so the forced answer
        // also gets a tool call back, carries no words, and is reported as exactly that.
        let (url, served) = crate::testutil::serve_sequence(&[READS_CARGO_TOML]).await;
        let mut agent = agent_at(&url);
        let (tx, _rx) = sink();
        let link = TurnLink::new(Mode::Normal);

        let report = run_sub_agent(&mut agent, "task", &tx, &link).await.unwrap();

        // LOOP_BREAK identical calls, then the one request that sends no tools.
        assert_eq!(*served.lock().unwrap(), LOOP_BREAK + 1);
        assert!(report.contains("could not produce a report"), "got: {}", report);
        assert!(report.contains("5 tool call(s)"), "got: {}", report);
    }

    #[tokio::test]
    async fn an_unreachable_server_is_reported_as_advice() {
        let mut agent = agent_at("http://127.0.0.1:1");
        let (tx, _rx) = sink();
        let link = TurnLink::new(Mode::Normal);

        let report = run_sub_agent(&mut agent, "task", &tx, &link).await.unwrap();
        assert!(report.contains("failed after 0 tool call(s)"), "got: {}", report);
    }
}
