//! Plan mode and auto mode: two ways of answering the permission question.
//!
//! Normal mode asks the user about every gated call. Plan mode refuses to change anything
//! until a written plan has been approved. Auto mode asks a second copy of the model,
//! given no conversation history, whether the call looks safe; a yes runs it unasked, and
//! anything else falls through to the user with the reason attached. The reviewer can
//! spend the user's consent, never withhold it.

use crate::llm::LMStudioClient;
use crate::permissions::{gate_for, runs_arbitrary_code, command_segments, Gate, AUTO_ALLOWED_TOOLS};
use crate::render::file_change_diff;
use anyhow::{anyhow, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use serde_json::{json, Value};
use std::io::{self, Write};
use std::path::PathBuf;

// Which set of rules is in force. One enum rather than independent flags: plan mode
// refuses the writes that auto mode exists to judge, so combining them leaves the judge
// with nothing to do.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Mode {
    #[default]
    Normal,
    Plan,
    Auto,
}

impl Mode {
    // The prompt prefix, so the current mode is visible on the line being typed rather
    // than only in the message that switched to it.
    pub(crate) fn prompt(&self) -> &'static str {
        match self {
            Mode::Normal => "> ",
            Mode::Plan => "plan> ",
            Mode::Auto => "auto> ",
        }
    }

    pub(crate) fn describe(&self) -> &'static str {
        match self {
            Mode::Normal => "normal — every gated call asks you",
            Mode::Plan => "plan — research and propose only, no changes until you approve",
            Mode::Auto => "auto — routine calls are reviewed and run; the rest still ask you",
        }
    }
}

// Tools that change nothing on this machine, so plan mode can offer them.
//
// The web tools are not in AUTO_ALLOWED_TOOLS because they send content off the machine,
// and that stays true here: plan mode offers them still carrying their normal prompt.
// Research is most of what planning is, and cutting it off would make plans worse.
const PLAN_MODE_READS: &[&str] = &["web_search", "web_get", "web_fetch"];

// The two tools that move between modes rather than doing anything to the machine. Each
// is offered only in the mode it leads out of: nothing to enter when already planning,
// nothing to present when not.
pub(crate) const START_PLAN: &str = "start_plan";
pub(crate) const PRESENT_PLAN: &str = "present_plan";

pub(crate) fn plan_mode_allows(tool_name: &str) -> bool {
    tool_name == PRESENT_PLAN
        || AUTO_ALLOWED_TOOLS.contains(&tool_name)
        || PLAN_MODE_READS.contains(&tool_name)
}

// What the model is told while planning.
//
// Refusing the write tools through tool errors alone is not enough: a model told "no" only
// after the fact spends rounds retrying variations, which is how the 50-round cap gets hit
// with nothing to show. Appended to the system message for the request only, so it is never
// persisted into the session and never lingers after the mode changes.
const PLAN_MODE_NOTE: &str = "\n\n# Plan mode is active\n\n\
You are planning, not building. Nothing you do right now may change this machine: no \
writing or editing files, no shell commands, no starting processes. Those tools are not \
available to you in this mode.\n\n\
Investigate first with the reading tools — read, ls, grep, glob_search — until you \
understand the actual code involved rather than guessing at it. Then call `present_plan` \
with your plan.\n\n\
A good plan names the specific files and functions it will touch, says what will change in \
each, and calls out anything you are unsure about. It is for a human to read and correct, \
so keep it concrete and short enough to check.\n\n\
The user will approve the plan or send back corrections. Do not begin work until they have \
approved it — you will be told when they do.";

// The permission decision for one call, once the mode has had its say.

pub(crate) fn gate_for_mode(mode: Mode, tool_name: &str, arguments: &Value) -> Gate {
    match mode {
        // Changing mode touches nothing, and entering plan mode only ever takes capability
        // away — asking permission to become more restricted would be noise.
        _ if tool_name == START_PLAN || tool_name == PRESENT_PLAN => Gate::Auto,
        Mode::Plan if !plan_mode_allows(tool_name) => Gate::Deny(format!(
            "'{}' is not available in plan mode, which changes nothing until a plan is \
             approved. Finish investigating with the reading tools, then call \
             `present_plan` with what you propose to do.",
            tool_name
        )),
        // Plan mode leaves the allowed tools on whatever gate they normally carry, so a
        // web fetch still asks. Auto mode changes who answers the prompt, not what counts
        // as needing one.
        _ => gate_for(tool_name, arguments),
    }
}

// The tool list to advertise for this mode.
//
// Filtering matters more than the deny message above: a tool the model cannot see is one
// it does not spend a round trying. present_plan is hidden outside plan mode for the same
// reason — there is nothing to approve when the model is already free to act.
pub(crate) fn tools_for_mode(mode: Mode, tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter(|tool| {
            let name = tool["function"]["name"].as_str().unwrap_or_default();
            match mode {
                Mode::Plan => plan_mode_allows(name),
                // start_plan stays visible here — this is the mode it leads out of.
                _ => name != PRESENT_PLAN,
            }
        })
        .cloned()
        .collect()
}

// The messages to send for one request, with any mode instructions attached.
//
// Appended to the system message rather than pushed as a trailing one: a system message in
// last position is handled inconsistently across local servers, and the note has to be
// reliable to be worth having.
pub(crate) fn with_mode_note(mode: Mode, messages: &[Value]) -> Vec<Value> {
    let mut request = messages.to_vec();
    if mode == Mode::Plan {
        if let Some(system) = request
            .first_mut()
            .filter(|m| m["role"] == "system")
            .and_then(|m| m.get_mut("content"))
            .and_then(|c| c.as_str().map(str::to_string))
        {
            request[0]["content"] = json!(format!("{}{}", system, PLAN_MODE_NOTE));
        }
    }
    request
}

// What the user decided about a proposed plan.

pub(crate) enum PlanDecision {
    Approve,
    // Approve, and have the reviewer look over each gated call from here on. Approving a
    // plan is the moment the user knows what is coming next, which is exactly when they
    // can reasonably decide to loosen their grip on the individual steps.
    ApproveWithAuto,
    Revise(String),
}

// Show a plan and ask what to do with it.

pub(crate) fn review_plan(plan: &str) -> Result<PlanDecision> {
    println!("\n📋 Proposed plan\n");
    // Rendered rather than printed raw: plans come back as markdown, and an unrendered
    // one is a wall of asterisks exactly when it most needs reading.
    if crate::render::render_markdown(plan).is_err() {
        println!("{}", plan);
    }
    println!();
    // Same shape as the permission prompt: the approvals first, the refusal last, so the
    // key that means "no" is in the same place in both.
    println!("1. Approve and start work");
    println!("2. Approve and enable auto mode (routine calls run without asking)");
    println!("3. Send back corrections");
    io::stdout().flush()?;

    enable_raw_mode()?;
    let decision = loop {
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('1') => break PlanDecision::Approve,
                KeyCode::Char('2') => break PlanDecision::ApproveWithAuto,
                KeyCode::Char('3') => {
                    disable_raw_mode()?;
                    println!("3\n");
                    print!("What should change? > ");
                    io::stdout().flush()?;
                    let mut feedback = String::new();
                    io::stdin().read_line(&mut feedback)?;
                    break PlanDecision::Revise(feedback.trim().to_string());
                }
                _ => continue,
            }
        }
    };
    let _ = disable_raw_mode();

    use std::time::Duration;
    while event::poll(Duration::from_millis(0))? {
        let _ = event::read()?;
    }

    Ok(decision)
}

// The safety reviewer's verdict on one call.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Verdict {
    pub(crate) allow: bool,
    pub(crate) reason: String,
}

// The reviewer is deliberately given no conversation history.
//
// That is the whole value of it: the model that just decided it wanted to run something is
// under pressure to finish the task, and a fresh reader of the same call is not. It is a
// filter against drift and accidents, not a security boundary — same weights, and content
// fetched from the web can end up inside the arguments it is asked to bless.
const JUDGE_PROMPT: &str = "You are reviewing one action a coding agent wants to take on a \
developer's machine. You are shown only the action itself, with no conversation history — \
judge it on what is in front of you.\n\n\
Allow ordinary development work: reading and editing files inside the project, building, \
running tests, formatting, git status and diff and commit, starting a local dev server, \
installing declared project dependencies.\n\n\
Deny anything that:\n\
- deletes or overwrites files outside the project directory\n\
- reads or transmits credentials, tokens, private keys, or .env contents\n\
- downloads code and runs it in the same step (curl or wget piped into a shell)\n\
- is hard to undo: force pushing, publishing a package, deploying, dropping a database, \
rewriting git history, disabling a safety check\n\
- changes system settings, installs system-wide packages, or touches another user's data\n\
- looks like it is exfiltrating the project's contents somewhere\n\n\
If you cannot tell what an action does, deny it. Uncertainty is a reason to involve the \
human, not to wave it through.\n\n\
Reply with one JSON object and nothing else:\n\
{\"verdict\": \"allow\", \"reason\": \"<one short sentence>\"}\n\
or\n\
{\"verdict\": \"deny\", \"reason\": \"<one short sentence>\"}";

// A script named on a command line, if one of the arguments points at a real file.
//
// Heuristic on purpose: `python x.py`, `node server.js` and `powershell -File .\x.ps1` all
// name the file differently, and `sh -c '...'` names no file at all. Taking the first
// argument that exists on disk covers the case that matters — a script the agent wrote
// itself moments earlier — and simply finds nothing when it does not apply.
pub(crate) fn referenced_script(command: &str) -> Option<PathBuf> {
    for segment in command_segments(command) {
        if !runs_arbitrary_code(&segment) {
            continue;
        }
        for token in segment.split_whitespace().skip(1) {
            let token = token.trim_matches(['"', '\''].as_slice());
            if token.starts_with('-') || token.is_empty() {
                continue;
            }
            let path = PathBuf::from(token);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

// Everything the reviewer gets to see about a call.

pub(crate) fn build_evidence(tool_name: &str, arguments: &Value) -> String {
    let mut evidence = format!("Tool: {}\n", tool_name);

    // For a write, the diff carries the path and the content both, so the raw arguments
    // would only repeat the file back a second time.
    if let Some(diff) = file_change_diff(tool_name, arguments) {
        evidence.push_str("\nThe change it would make:\n\n");
        evidence.push_str(&diff);
        evidence.push('\n');
        return evidence;
    }

    let pretty = serde_json::to_string_pretty(arguments)
        .unwrap_or_else(|_| arguments.to_string());
    evidence.push_str("\nArguments:\n");
    evidence.push_str(&pretty);
    evidence.push('\n');

    // A command that runs a script says nothing on its own — the agent wrote the script,
    // so the file is the actual proposal. Sent whole: a truncated script is one whose
    // interesting half may be the part that was cut.
    if matches!(tool_name, "run_command" | "run_background") {
        let command = arguments["command"].as_str().unwrap_or_default();
        if let Some(path) = referenced_script(command) {
            evidence.push_str(&format!("\nContents of {}, which this command runs:\n\n", path.display()));
            match std::fs::read_to_string(&path) {
                Ok(source) => evidence.push_str(&source),
                Err(_) => evidence.push_str("(not readable as text)"),
            }
            evidence.push('\n');
        }
    }

    evidence
}

// Pull a verdict out of whatever the model actually replied with.
//
// Local models wrap JSON in fences and prepend a <think> block, and neither is worth
// failing over. Anything genuinely unparseable is an error rather than a guess: a verdict
// invented here would carry the same weight as a real one.
pub(crate) fn parse_verdict(raw: &str) -> Result<Verdict> {
    let mut text = raw.trim();

    // Reasoning models emit their scratchpad first; it is full of the words "allow" and
    // "deny" in every combination, so it has to go before anything else is read.
    if let Some(end) = text.find("</think>") {
        text = text[end + "</think>".len()..].trim();
    }

    let start = text.find('{').ok_or_else(|| anyhow!("no JSON object in the reply"))?;
    let end = text.rfind('}').ok_or_else(|| anyhow!("no JSON object in the reply"))?;
    if end < start {
        return Err(anyhow!("no JSON object in the reply"));
    }

    let parsed: Value = serde_json::from_str(&text[start..=end])
        .map_err(|e| anyhow!("reply was not valid JSON: {}", e))?;

    let verdict = parsed["verdict"]
        .as_str()
        .ok_or_else(|| anyhow!("reply had no 'verdict' field"))?
        .trim()
        .to_lowercase();

    let allow = if verdict.starts_with("allow") {
        true
    } else if verdict.starts_with("deny") {
        false
    } else {
        return Err(anyhow!("unrecognized verdict '{}'", verdict));
    };

    let reason = parsed["reason"].as_str().unwrap_or("").trim().to_string();

    Ok(Verdict { allow, reason })
}

// Ask the reviewer about one call.

pub(crate) async fn judge_tool_call(client: &LMStudioClient, tool_name: &str, arguments: &Value) -> Result<Verdict> {
    let messages = vec![
        json!({"role": "system", "content": JUDGE_PROMPT}),
        json!({"role": "user", "content": build_evidence(tool_name, arguments)}),
    ];

    // Generous for a two-field answer, because the answer is not all that gets generated:
    // a reasoning model thinks first, and a cap that cuts it off mid-thought yields a reply
    // with no JSON in it at all. That surfaced as "Review failed: no JSON object" on a
    // perfectly ordinary `cargo build`.
    let reply = client.chat_completion_plain(messages, 1000).await?;
    parse_verdict(&reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plan_mode_refuses_to_change_anything() {
        // Reading is how planning happens, so it must not prompt.
        for tool in ["read", "grep", "ls", "glob_search"] {
            assert!(
                matches!(gate_for_mode(Mode::Plan, tool, &json!({})), Gate::Auto),
                "{} should stay available while planning",
                tool
            );
        }

        // Everything that touches the machine is refused outright, not merely asked about:
        // a plan the user has not seen yet cannot be consented to one call at a time.
        for tool in ["write_file", "edit", "multi_edit", "run_command", "run_background"] {
            let gate = gate_for_mode(Mode::Plan, tool, &json!({"command": "ls", "file_path": "a.rs"}));
            assert!(matches!(gate, Gate::Deny(_)), "{} must be denied in plan mode", tool);
        }

        // Research still reaches the network, and still asks first.
        assert!(matches!(
            gate_for_mode(Mode::Plan, "web_search", &json!({"query": "x"})),
            Gate::Ask { .. }
        ));

        // Normal and auto mode are unchanged by any of this.
        for mode in [Mode::Normal, Mode::Auto] {
            assert!(matches!(
                gate_for_mode(mode, "write_file", &json!({"file_path": "a.rs"})),
                Gate::Ask { .. }
            ));
        }
    }

    #[test]
    fn the_advertised_tool_list_follows_the_mode() {
        let tools: Vec<Value> = [
            "read", "write_file", "run_command", "web_search", START_PLAN, PRESENT_PLAN,
        ]
        .iter()
        .map(|name| json!({"function": {"name": name}}))
        .collect();

        let names = |mode: Mode| -> Vec<String> {
            tools_for_mode(mode, &tools)
                .iter()
                .map(|t| t["function"]["name"].as_str().unwrap().to_string())
                .collect()
        };

        // A tool the model cannot see is one it does not waste a round on.
        let planning = names(Mode::Plan);
        assert!(planning.contains(&"read".to_string()));
        assert!(planning.contains(&"web_search".to_string()));
        assert!(planning.contains(&PRESENT_PLAN.to_string()));
        assert!(!planning.contains(&"write_file".to_string()));
        assert!(!planning.contains(&"run_command".to_string()));
        // Already planning: nothing to enter.
        assert!(!planning.contains(&START_PLAN.to_string()));

        // Outside plan mode the pairing is the other way round — the model can put itself
        // into plan mode, but has no plan outstanding to present.
        let working = names(Mode::Normal);
        assert!(working.contains(&"write_file".to_string()));
        assert!(working.contains(&START_PLAN.to_string()));
        assert!(!working.contains(&PRESENT_PLAN.to_string()));
        assert_eq!(names(Mode::Auto), working);
    }

    #[test]
    fn changing_mode_never_asks_permission() {
        // Entering plan mode only removes capability, and present_plan is itself the
        // consent step — prompting for either would be asking to be asked.
        for mode in [Mode::Normal, Mode::Plan, Mode::Auto] {
            for tool in [START_PLAN, PRESENT_PLAN] {
                assert!(
                    matches!(gate_for_mode(mode, tool, &json!({"plan": "do the thing"})), Gate::Auto),
                    "{} should not prompt in {:?} mode",
                    tool,
                    mode
                );
            }
        }
    }

    #[test]
    fn the_plan_note_rides_on_the_system_message_only_while_planning() {
        let messages = vec![
            json!({"role": "system", "content": "base prompt"}),
            json!({"role": "user", "content": "do a thing"}),
        ];

        let planning = with_mode_note(Mode::Plan, &messages);
        let system = planning[0]["content"].as_str().unwrap();
        assert!(system.starts_with("base prompt"));
        assert!(system.contains("Plan mode is active"));
        // The rest of the conversation is untouched...
        assert_eq!(planning[1], messages[1]);
        // ...and so is the caller's history, or the note would be persisted and outlive
        // the mode that set it.
        assert_eq!(messages[0]["content"], "base prompt");

        for mode in [Mode::Normal, Mode::Auto] {
            assert_eq!(with_mode_note(mode, &messages), messages);
        }
    }

    #[test]
    fn a_verdict_survives_the_wrapping_local_models_add() {
        let bare = parse_verdict(r#"{"verdict": "allow", "reason": "edits project source"}"#).unwrap();
        assert!(bare.allow);
        assert_eq!(bare.reason, "edits project source");

        // Fenced, which is what most local models actually send.
        assert!(parse_verdict("```json\n{\"verdict\":\"deny\",\"reason\":\"deletes /etc\"}\n```")
            .unwrap()
            .allow
            .eq(&false));

        // A reasoning preamble mentions both words repeatedly before deciding; reading it
        // as part of the answer would make the verdict a coin toss.
        let reasoned = parse_verdict(
            "<think>Should I allow this? It could deny... allow seems fine</think>\n\
             {\"verdict\": \"deny\", \"reason\": \"pipes curl into sh\"}",
        )
        .unwrap();
        assert!(!reasoned.allow);
        assert_eq!(reasoned.reason, "pipes curl into sh");

        // Prose around the object is tolerated, since the object is unambiguous.
        assert!(parse_verdict("Here you go: {\"verdict\":\"allow\"} — hope that helps")
            .unwrap()
            .allow);

        // Anything genuinely unreadable is an error. Guessing here would produce a verdict
        // indistinguishable from a real one.
        assert!(parse_verdict("looks fine to me").is_err());
        assert!(parse_verdict("{\"verdict\": \"maybe\"}").is_err());
        assert!(parse_verdict("{\"reason\": \"no verdict field\"}").is_err());
        assert!(parse_verdict("").is_err());
    }

    #[test]
    fn the_reviewer_is_shown_the_script_a_command_would_run() {
        let script = crate::testutil::scratch("deploy.py", "import os\nos.system('rm -rf /')\n");

        let command = format!("python {}", script.display());
        assert_eq!(referenced_script(&command).as_deref(), Some(script.as_path()));

        // The command line alone says nothing; the file is the actual proposal.
        let evidence = build_evidence("run_command", &json!({"command": command}));
        assert!(evidence.contains("rm -rf /"), "got: {}", evidence);

        // Inline code names no file, and is already visible in the command itself.
        assert_eq!(referenced_script("sh -c 'echo hi'"), None);
        // An ordinary program is not an interpreter, so nothing is read on its behalf.
        assert_eq!(referenced_script("cargo build --release"), None);
        // A script that does not exist yet cannot be read, and must not break the call.
        assert_eq!(referenced_script("python nope-not-here.py"), None);

        let _ = std::fs::remove_file(&script);
    }

    #[test]
    fn a_write_is_shown_as_its_diff_rather_than_its_arguments() {
        // A path that does not exist yet, so the diff is the whole new file.
        let file = std::env::temp_dir().join("haico-judge-write.rs");
        let _ = std::fs::remove_file(&file);

        let evidence = build_evidence(
            "write_file",
            &json!({"file_path": file.display().to_string(), "content": "const KEY: &str = \"x\";\n"}),
        );
        assert!(evidence.contains("Tool: write_file"));
        assert!(evidence.contains("const KEY"), "got: {}", evidence);

        // A tool with no file involved still gets its arguments through.
        let evidence = build_evidence("web_search", &json!({"query": "rust tokio select"}));
        assert!(evidence.contains("rust tokio select"), "got: {}", evidence);
    }
}
