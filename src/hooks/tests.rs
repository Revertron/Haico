// Holding `out::capture_guard()` across awaits, as the turn tests do: each test has a runtime
// of its own, so nothing else waits on the lock.
#![allow(clippy::await_holding_lock)]

use super::*;
use crate::agent::{run_turn, Agent};
use crate::modes::Mode;
use crate::testutil::{serve_sequence, test_config, TempDir};
use crate::turn::{AgentEvent, TurnEnd};
use serde_json::json;

fn handler(command: &str) -> Handler {
    Handler { action: Action::Command { command: command.to_string(), args: None }, timeout: None }
}

fn output(code: i32, stdout: &str, stderr: &str) -> Result<Output, String> {
    Ok(Output { code: Some(code), stdout: stdout.to_string(), stderr: stderr.to_string() })
}

fn rule(event: Event, matcher: Option<&str>, handler: Handler) -> Rule {
    let (parsed, problems) = config::parse(
        &json!({"hooks": {event.name(): [{"matcher": matcher.unwrap_or("*"), "hooks": [{"type": "command", "command": "x"}]}]}})
            .to_string(),
        Scope::Global,
    )
    .unwrap();
    assert!(problems.is_empty(), "{problems:?}");
    Rule { handler, ..parsed.rules.into_iter().next().unwrap() }
}

/// A hook script in `dir` that prints `stdout`, writes `stderr` and exits with `code`. The
/// texts go through files rather than `echo`, so JSON reaches the hook's stdout unquoted on
/// both shells.
fn script(dir: &TempDir, name: &str, stdout: &str, stderr: &str, code: i32) -> Handler {
    std::fs::write(dir.join(&format!("{name}.out")), stdout).unwrap();
    std::fs::write(dir.join(&format!("{name}.err")), stderr).unwrap();
    if cfg!(windows) {
        let path = dir.join(&format!("{name}.cmd"));
        std::fs::write(
            &path,
            format!(
                "@echo off\r\ntype \"%~dp0{name}.out\"\r\ntype \"%~dp0{name}.err\" 1>&2\r\nexit /b {code}\r\n"
            ),
        )
        .unwrap();
        handler(&format!("\"{}\"", path.display()))
    } else {
        let path = dir.join(&format!("{name}.sh"));
        std::fs::write(
            &path,
            format!("d=$(dirname \"$0\")\ncat \"$d/{name}.out\"\ncat \"$d/{name}.err\" >&2\nexit {code}\n"),
        )
        .unwrap();
        handler(&format!("sh \"{}\"", path.display()))
    }
}

/// An exec-form hook that copies its stdin to `path`.
fn stdin_recorder(path: &Path) -> Handler {
    let (command, args) = if cfg!(windows) {
        (
            "powershell",
            vec![
                "-NoProfile".to_string(),
                "-Command".to_string(),
                format!("[Console]::In.ReadToEnd() | Set-Content -NoNewline -LiteralPath '{}'", path.display()),
            ],
        )
    } else {
        ("sh", vec!["-c".to_string(), format!("cat > '{}'", path.display())])
    };
    Handler { action: Action::Command { command: command.to_string(), args: Some(args) }, timeout: None }
}

fn http(url: &str) -> Handler {
    Handler {
        action: Action::Http { url: url.to_string(), headers: Default::default(), allowed_env_vars: Vec::new() },
        timeout: None,
    }
}

/// A server that answers one request with `status` and `body`, and hands back the whole
/// request it got — head and body — as text.
async fn serve_reply(status: &'static str, body: &'static str) -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/hook", listener.local_addr().unwrap());
    let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = captured.clone();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        // The head, then as much body as Content-Length says.
        loop {
            let n = socket.read(&mut buf).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&request).to_string();
            if let Some(head_end) = text.find("\r\n\r\n") {
                let length: usize = text[..head_end]
                    .lines()
                    .find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0)))
                    .unwrap_or(0);
                if request.len() >= head_end + 4 + length {
                    break;
                }
            }
        }
        *sink.lock().unwrap() = String::from_utf8_lossy(&request).to_string();
        let response = format!(
            "HTTP/1.1 {}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
            status,
            body.len(),
            body
        );
        let _ = socket.write_all(response.as_bytes()).await;
    });
    (url, captured)
}

// ---------------------------------------------------------------------------
// What the hooks said
// ---------------------------------------------------------------------------

#[test]
fn exit_two_blocks_by_event_and_other_codes_are_failures() {
    let o = interpret(Event::PreToolUse, vec![(handler("a"), output(2, "", "no rm\n"))]);
    assert_eq!(o.decision, Some(Decision::Deny("no rm".to_string())));

    let o = interpret(Event::Stop, vec![(handler("a"), output(2, "", "tests fail"))]);
    assert_eq!(o.block.as_deref(), Some("tests fail"));

    // Not blocking, and not read: a permission verdict is the JSON's to give.
    let o = interpret(Event::PermissionRequest, vec![(handler("a"), output(2, "", "x"))]);
    assert!(o.decision.is_none() && o.block.is_none() && o.failures.is_empty());

    let o = interpret(Event::PreToolUse, vec![(handler("a"), output(1, "", "boom")), (handler("b"), Err("could not start".into()))]);
    assert!(o.decision.is_none());
    assert_eq!(o.failures.len(), 2);
    assert!(o.failures[0].1.contains("exit code 1") && o.failures[0].1.contains("boom"));
}

#[test]
fn plain_stdout_is_context_only_where_claude_code_makes_it_so() {
    let o = interpret(Event::UserPromptSubmit, vec![(handler("a"), output(0, "branch: main\n", ""))]);
    assert_eq!(o.context, vec!["branch: main"]);
    let o = interpret(Event::PreToolUse, vec![(handler("a"), output(0, "chatter", ""))]);
    assert!(o.context.is_empty());
}

#[test]
fn the_json_answers_are_read() {
    let pre = json!({
        "systemMessage": "careful",
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": "a person should look",
            "updatedInput": {"command": "cargo test -q"},
            "additionalContext": "tests are slow here"
        }
    });
    let o = interpret(Event::PreToolUse, vec![(handler("a"), output(0, &pre.to_string(), ""))]);
    assert_eq!(o.decision, Some(Decision::Ask("a person should look".to_string())));
    assert_eq!(o.updated_input, Some(json!({"command": "cargo test -q"})));
    assert_eq!(o.context, vec!["tests are slow here"]);
    assert_eq!(o.messages, vec!["careful"]);

    let request = json!({"hookSpecificOutput": {"hookEventName": "PermissionRequest",
        "decision": {"behavior": "deny", "message": "not on Fridays"}}});
    let o = interpret(Event::PermissionRequest, vec![(handler("a"), output(0, &request.to_string(), ""))]);
    assert_eq!(o.decision, Some(Decision::Deny("not on Fridays".to_string())));

    let post = json!({"decision": "block", "reason": "lint failed",
        "hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedToolOutput": "redacted"}});
    let o = interpret(Event::PostToolUse, vec![(handler("a"), output(0, &post.to_string(), ""))]);
    assert_eq!(o.block.as_deref(), Some("lint failed"));
    assert_eq!(o.updated_output.as_deref(), Some("redacted"));

    let stop = json!({"continue": false, "stopReason": "budget spent"});
    let o = interpret(Event::Stop, vec![(handler("a"), output(0, &stop.to_string(), ""))]);
    assert_eq!(o.stop.as_deref(), Some("budget spent"));
}

#[test]
fn the_strictest_verdict_wins() {
    let allow = json!({"hookSpecificOutput": {"permissionDecision": "allow", "permissionDecisionReason": "fine"}});
    let deny = json!({"hookSpecificOutput": {"permissionDecision": "deny", "permissionDecisionReason": "no"}});
    let o = interpret(
        Event::PreToolUse,
        vec![
            (handler("a"), output(0, &allow.to_string(), "")),
            (handler("b"), output(0, &deny.to_string(), "")),
            (handler("c"), output(0, &allow.to_string(), "")),
        ],
    );
    assert_eq!(o.decision, Some(Decision::Deny("no".to_string())));
}

#[test]
fn handlers_match_by_subject_and_run_once() {
    let hooks = Hooks::from_rules(
        vec![
            rule(Event::PreToolUse, Some("run_command"), handler("a")),
            rule(Event::PreToolUse, Some("edit|write_file"), handler("b")),
            rule(Event::PreToolUse, None, handler("a")),
            rule(Event::Stop, None, handler("c")),
        ],
        Path::new("."),
    );
    let names = |subject| hooks.matching(Event::PreToolUse, Some(subject)).into_iter().map(|h| h.describe()).collect::<Vec<_>>();
    assert_eq!(names("run_command"), ["a"]);
    assert_eq!(names("edit"), ["b", "a"]);
    assert!(hooks.has(Event::Stop) && !hooks.has(Event::SessionStart));
}

// ---------------------------------------------------------------------------
// Running them
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_hook_gets_the_event_on_stdin_and_answers_on_stdout() {
    let dir = TempDir::new("hook-run");
    let recorded = dir.join("input.json");
    let ran = run_all(
        vec![stdin_recorder(&recorded), script(&dir, "say", r#"{"systemMessage":"hi"}"#, "", 0)],
        json!({"hook_event_name": "PreToolUse", "tool_name": "run_command"}),
        dir.path().to_path_buf(),
    )
    .await;
    let o = interpret(Event::PreToolUse, ran);
    assert!(o.failures.is_empty(), "{:?}", o.failures);
    assert_eq!(o.messages, vec!["hi"]);
    let input: Value = serde_json::from_str(&std::fs::read_to_string(&recorded).unwrap()).unwrap();
    assert_eq!(input["tool_name"], "run_command");
}

#[tokio::test]
async fn a_timeout_set_on_the_entry_is_kept() {
    let dir = TempDir::new("hook-timeout");
    let mut slow = if cfg!(windows) {
        handler("ping -n 30 127.0.0.1 >nul")
    } else {
        handler("sleep 30")
    };
    slow.timeout = Some(std::time::Duration::from_secs(1));
    let started = std::time::Instant::now();
    let ran = run_all(vec![slow], json!({}), dir.path().to_path_buf()).await;
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
    assert!(matches!(&ran[0].1, Err(e) if e.contains("within 1 s")), "{:?}", ran[0].1);
}

#[tokio::test]
async fn an_http_hook_posts_the_event_and_its_reply_is_read_as_stdout() {
    let (url, request) =
        serve_reply("200 OK", r#"{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"no"}}"#)
            .await;
    // A variable every machine has, so nothing needs setting for the test.
    let var = if cfg!(windows) { "SystemRoot" } else { "HOME" };
    let value = std::env::var(var).unwrap();
    let mut hook = http(&url);
    hook.action = Action::Http {
        url,
        headers: [
            ("X-Allowed".to_string(), format!("v=${{{var}}}")),
            ("X-Not-Listed".to_string(), "v=$PATH".to_string()),
        ]
        .into(),
        allowed_env_vars: vec![var.to_string()],
    };
    let ran = run_all(vec![hook], json!({"hook_event_name": "PreToolUse"}), PathBuf::from(".")).await;
    let o = interpret(Event::PreToolUse, ran);
    assert_eq!(o.decision, Some(Decision::Deny("no".to_string())));

    let request = request.lock().unwrap().clone();
    let lower = request.to_lowercase();
    assert!(request.starts_with("POST /hook "), "{request}");
    assert!(lower.contains("content-type: application/json"), "{request}");
    assert!(lower.contains(&format!("x-allowed: v={}", value.to_lowercase())), "{request}");
    assert!(lower.contains("x-not-listed: v=\r\n"), "{request}");
    assert!(request.ends_with(r#"{"hook_event_name":"PreToolUse"}"#), "{request}");
}

#[tokio::test]
async fn an_http_hook_that_says_nothing_is_fine_and_a_bad_reply_blocks_nothing() {
    let (url, _) = serve_reply("204 No Content", "").await;
    let o = interpret(Event::PreToolUse, run_all(vec![http(&url)], json!({}), PathBuf::from(".")).await);
    assert!(o.failures.is_empty() && o.decision.is_none(), "{:?}", o.failures);

    // Plain text is not context over HTTP, even where a command's would be.
    let (url, _) = serve_reply("200 OK", "on branch main").await;
    let o = interpret(Event::UserPromptSubmit, run_all(vec![http(&url)], json!({}), PathBuf::from(".")).await);
    assert!(o.context.is_empty());
    assert!(o.failures[0].1.contains("not a JSON object"), "{:?}", o.failures);

    let (url, _) = serve_reply("500 Internal Server Error", "down").await;
    let o = interpret(Event::PreToolUse, run_all(vec![http(&url)], json!({}), PathBuf::from(".")).await);
    assert!(o.decision.is_none());
    assert!(o.failures[0].1.contains("500") && o.failures[0].1.contains("down"), "{:?}", o.failures);

    // Nobody listening.
    let o = interpret(Event::PreToolUse, run_all(vec![http("http://127.0.0.1:9/hook")], json!({}), PathBuf::from(".")).await);
    assert_eq!(o.failures.len(), 1);
}

// ---------------------------------------------------------------------------
// In a turn
// ---------------------------------------------------------------------------

const DATE_CALL: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,
    "tool_calls":[{"id":"c1","type":"function","function":{"name":"get_date","arguments":"{}"}}]}}]}"#;
const DONE: &str = r#"{"choices":[{"message":{"role":"assistant","content":"done"}}]}"#;

async fn turn(bodies: &'static [&'static str], rules: Vec<Rule>, dir: &TempDir) -> (Agent, usize, TurnEnd, Vec<Notice>) {
    let (url, served) = serve_sequence(bodies).await;
    let cwd = std::env::current_dir().unwrap();
    let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
    agent.hooks = Hooks::from_rules(rules, dir.path());
    let link = TurnLink::new(Mode::Normal);
    let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
    // The frontend: every question is dropped unanswered, which the turn hears as a refusal.
    let frontend = async {
        let mut notices = Vec::new();
        while let Some(event) = events.recv().await {
            if let AgentEvent::Notice(notice) = event {
                notices.push(notice);
            }
        }
        notices
    };
    let turn = async {
        let end = run_turn(&mut agent, "go", false, &tx, &link).await;
        drop(tx);
        end
    };
    let (end, notices) = tokio::join!(turn, frontend);
    let count = *served.lock().unwrap();
    (agent, count, end, notices)
}

fn tool_reply(agent: &Agent) -> String {
    agent.messages.iter().find(|m| m["role"] == "tool").unwrap()["content"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn a_pre_tool_use_deny_answers_the_call_without_running_it() {
    let _capture = crate::out::capture_guard();
    let dir = TempDir::new("hook-pre-deny");
    let deny = script(&dir, "deny", "", "dates are off limits", 2);
    let (agent, served, end, notices) =
        turn(&[DATE_CALL, DONE], vec![rule(Event::PreToolUse, Some("get_date"), deny)], &dir).await;
    assert_eq!((served, end), (2, TurnEnd::Answered));
    let reply = tool_reply(&agent);
    assert!(reply.starts_with("Blocked by the user's PreToolUse hook (dates are off limits)"), "{reply}");
    assert!(notices.iter().any(|n| matches!(n, Notice::HookBlocked { event: "PreToolUse", .. })));
}

#[tokio::test]
async fn a_pre_tool_use_ask_puts_even_a_free_call_to_the_user() {
    let _capture = crate::out::capture_guard();
    let dir = TempDir::new("hook-pre-ask");
    let ask = script(&dir, "ask", r#"{"hookSpecificOutput":{"permissionDecision":"ask"}}"#, "", 0);
    let (agent, _, _, _) = turn(&[DATE_CALL, DONE], vec![rule(Event::PreToolUse, None, ask)], &dir).await;
    // Nobody answers in a test, which is a refusal.
    assert!(tool_reply(&agent).contains("could not be requested"), "{}", tool_reply(&agent));
}

#[tokio::test]
async fn post_tool_use_context_goes_under_the_result() {
    let _capture = crate::out::capture_guard();
    let dir = TempDir::new("hook-post");
    let note = script(&dir, "note", r#"{"hookSpecificOutput":{"additionalContext":"the clock is UTC"}}"#, "", 0);
    let (agent, _, _, _) = turn(&[DATE_CALL, DONE], vec![rule(Event::PostToolUse, Some("get_date"), note)], &dir).await;
    let reply = tool_reply(&agent);
    assert!(reply.ends_with("[PostToolUse hook]\nthe clock is UTC"), "{reply}");
}

#[tokio::test]
async fn a_blocked_prompt_is_never_sent_or_kept() {
    let _capture = crate::out::capture_guard();
    let dir = TempDir::new("hook-prompt");
    let block = script(&dir, "block", "", "no secrets in prompts", 2);
    let (agent, served, end, notices) = turn(&[DONE], vec![rule(Event::UserPromptSubmit, None, block)], &dir).await;
    assert_eq!((served, end), (0, TurnEnd::Failed));
    assert!(!agent.messages.iter().any(|m| m["role"] == "user"));
    assert!(notices.contains(&Notice::HookBlocked { event: "UserPromptSubmit", reason: "no secrets in prompts".into() }));
}

#[tokio::test]
async fn prompt_context_joins_the_message() {
    let _capture = crate::out::capture_guard();
    let dir = TempDir::new("hook-prompt-context");
    let context = script(&dir, "context", "on branch main", "", 0);
    let (agent, served, _, _) = turn(&[DONE], vec![rule(Event::UserPromptSubmit, None, context)], &dir).await;
    assert_eq!(served, 1);
    let user = agent.messages.iter().find(|m| m["role"] == "user").unwrap()["content"].as_str().unwrap();
    assert_eq!(user, "go\n\n[UserPromptSubmit hook]\non branch main");
}

/// The hook objects once, then lets the model stop: the second time it finds its marker.
#[tokio::test]
async fn a_stop_hook_sends_the_model_back_to_work() {
    let _capture = crate::out::capture_guard();
    let dir = TempDir::new("hook-stop");
    let once = if cfg!(windows) {
        let path = dir.join("once.cmd");
        std::fs::write(
            &path,
            "@echo off\r\nif exist \"%~dp0marker\" exit /b 0\r\necho.> \"%~dp0marker\"\r\necho run the tests first 1>&2\r\nexit /b 2\r\n",
        )
        .unwrap();
        handler(&format!("\"{}\"", path.display()))
    } else {
        let path = dir.join("once.sh");
        std::fs::write(
            &path,
            "d=$(dirname \"$0\")\n[ -e \"$d/marker\" ] && exit 0\ntouch \"$d/marker\"\necho run the tests first >&2\nexit 2\n",
        )
        .unwrap();
        handler(&format!("sh \"{}\"", path.display()))
    };
    let (agent, served, end, notices) = turn(&[DONE, DONE], vec![rule(Event::Stop, None, once)], &dir).await;
    assert_eq!((served, end), (2, TurnEnd::Answered));
    let roles: Vec<&str> = agent.messages.iter().filter_map(|m| m["role"].as_str()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "user", "assistant"]);
    assert_eq!(agent.messages[3]["content"], "[Stop hook]\nrun the tests first");
    assert!(notices.contains(&Notice::HookContinued("run the tests first".into())));
}

#[tokio::test]
async fn continue_false_ends_the_turn_after_the_round() {
    let _capture = crate::out::capture_guard();
    let dir = TempDir::new("hook-continue");
    let halt = script(&dir, "halt", r#"{"continue":false,"stopReason":"enough"}"#, "", 0);
    let (agent, served, end, notices) =
        turn(&[DATE_CALL, DONE], vec![rule(Event::PostToolUse, None, halt)], &dir).await;
    assert_eq!((served, end), (1, TurnEnd::Failed));
    // The call still has its reply.
    assert!(!tool_reply(&agent).is_empty());
    assert!(notices.contains(&Notice::HookStopped { event: "PostToolUse", reason: "enough".into() }));
}

#[tokio::test]
async fn session_start_context_goes_into_the_system_prompt() {
    let dir = TempDir::new("hook-session");
    let context = script(&dir, "context", "toolchain: 1.92", "", 0);
    let (url, _) = serve_sequence(&[DONE]).await;
    let cwd = std::env::current_dir().unwrap();
    let mut agent = Agent::new(test_config(&url), "m".to_string(), &cwd);
    agent.hooks = Hooks::from_rules(vec![rule(Event::SessionStart, Some("startup"), context)], dir.path());
    let link = TurnLink::new(Mode::Normal);
    let (tx, _events) = tokio::sync::mpsc::unbounded_channel();

    // Matched by source: `clear` is not `startup`.
    agent.session_start("clear", &tx, &link).await;
    assert!(agent.session_context.is_none());
    agent.session_start("startup", &tx, &link).await;
    assert_eq!(agent.session_context.as_deref(), Some("toolchain: 1.92"));
    let system = agent.messages[0]["content"].as_str().unwrap();
    assert!(system.ends_with("toolchain: 1.92"), "{system}");
}
