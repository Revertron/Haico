//! The `timer` tool over the store in `crate::timers`.
//!
//! One tool with an action rather than five tools, for the reason `remember` is one: every
//! tool is a paragraph in every request, and a small model picks between three actions of
//! one tool more reliably than between five similar names. The store knows nothing about
//! the clock; this supplies it.

use super::MyMcpServer;
use crate::timers::{Cron, Fired, Schedule};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::CallToolResult;
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct TimerParams {
    // In the schema as well as the description, as with `remember`: a server that
    // constrains decoding can then only produce one of these, and `normalize_action` is
    // the net under the ones that do not.
    #[schemars(
        extend("enum" = ["once", "every", "cron", "list", "cancel"]),
        description = "'once' fires one time after 'seconds' (like setTimeout). 'every' fires repeatedly every 'seconds' (like setInterval), at least 60. 'cron' fires at the local times a five-field cron expression in 'cron' matches. 'list' shows what is armed. 'cancel' disarms the timer with 'id'."
    )]
    pub(crate) action: String,
    #[schemars(description = "For 'once' and 'every': the delay, or the period, in seconds. 300 is five minutes.")]
    pub(crate) seconds: Option<u64>,
    #[schemars(
        description = "For 'cron': minute hour day-of-month month day-of-week, in local time. '*/10 * * * *' is every ten minutes, '0 9 * * mon-fri' is 9:00 on weekdays. @hourly and @daily work too."
    )]
    pub(crate) cron: Option<String>,
    #[schemars(
        description = "For 'once', 'every' and 'cron': the message you will be sent when it fires. Write it as an instruction to your future self that makes sense on its own: 'Check whether the test run in background job 2 has finished and report the failures', not 'check'."
    )]
    pub(crate) prompt: Option<String>,
    #[schemars(description = "For 'cancel': the id the timer was given when it was set.")]
    pub(crate) id: Option<u32>,
}

/// Map whatever the model called the action onto one of the five, or `None`. The
/// JavaScript names are on the list because they are the names the model was just
/// told these work like.
fn normalize_action(raw: &str) -> Option<&'static str> {
    match super::action_key(raw).as_str() {
        "once" | "timeout" | "settimeout" | "after" | "delay" | "oneshot" | "later" | "in" => Some("once"),
        "every" | "interval" | "setinterval" | "repeat" | "recurring" | "periodic" => Some("every"),
        "cron" | "schedule" | "crontab" | "at" => Some("cron"),
        "list" | "show" | "ls" | "status" | "get" => Some("list"),
        "cancel" | "stop" | "delete" | "remove" | "clear" | "cleartimeout" | "clearinterval"
        | "disarm" | "unset" => Some("cancel"),
        _ => None,
    }
}

use super::{error, success};

impl MyMcpServer {
    /// When the frontend should next look at the timers, or `None` with none armed.
    pub(crate) fn next_timer_wake(&self) -> Option<Instant> {
        self.timers.lock().unwrap().next_wake(Instant::now())
    }

    /// Take the timers that are due now.
    pub(crate) fn take_due_timers(&self) -> Vec<Fired> {
        self.timers.lock().unwrap().take_due(Instant::now(), chrono::Local::now())
    }

    /// What is armed, one line per timer, for the model and for `/timers` alike.
    pub(crate) fn describe_timers(&self) -> String {
        let timers = self.timers.lock().unwrap();
        if timers.is_empty() {
            return "No timers are armed.".to_string();
        }
        let now = Instant::now();
        timers
            .list()
            .iter()
            .map(|t| {
                format!(
                    "Timer {}: {}, next at {} (in {}) — {}",
                    t.id,
                    t.schedule.describe(),
                    t.due_wall.format("%Y-%m-%d %H:%M:%S"),
                    crate::timers::human(t.due.saturating_duration_since(now)),
                    t.prompt
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(crate) fn timer_in(&self, params: TimerParams) -> CallToolResult {
        let Some(action) = normalize_action(&params.action) else {
            return error(format!(
                "'{}' is not something timer can do. Use action 'once', 'every', 'cron', \
                 'list' or 'cancel'.",
                params.action
            ));
        };

        let schedule = match action {
            "list" => return success(self.describe_timers()),
            "cancel" => {
                let Some(id) = params.id else {
                    return error(format!("Pass the 'id' of the timer to cancel.\n{}", self.describe_timers()));
                };
                let cancelled = self.timers.lock().unwrap().cancel(id);
                return match cancelled {
                    Some(timer) => success(format!(
                        "Timer {} ({}) cancelled; it will not fire again.",
                        timer.id,
                        timer.schedule.describe()
                    )),
                    None => error(format!(
                        "There is no timer {} — it may already have fired.\n{}",
                        id,
                        self.describe_timers()
                    )),
                };
            }
            "cron" => {
                let Some(expression) = params.cron.as_deref().filter(|c| !c.trim().is_empty()) else {
                    return error("Pass the cron expression in 'cron', like '*/10 * * * *'.".to_string());
                };
                match Cron::parse(expression) {
                    Ok(cron) => Schedule::Cron(cron),
                    Err(why) => return error(why),
                }
            }
            _ => {
                let Some(seconds) = params.seconds else {
                    return error(format!("Pass the {} in 'seconds'.", if action == "once" { "delay" } else { "period" }));
                };
                let duration = Duration::from_secs(seconds);
                if action == "once" { Schedule::Once(duration) } else { Schedule::Every(duration) }
            }
        };

        let prompt = params.prompt.unwrap_or_default();
        let mut timers = self.timers.lock().unwrap();
        match timers.add(schedule, &prompt, Instant::now(), chrono::Local::now()) {
            Ok(timer) => {
                let after = match timer.schedule {
                    Schedule::Once(_) => "It fires once.".to_string(),
                    _ => format!(
                        "It keeps firing ({}) until you cancel it with action 'cancel' and id {}.",
                        timer.schedule.describe(),
                        timer.id
                    ),
                };
                success(format!(
                    "Timer {} set: first fires at {} (in {}). {} When it fires you will be sent \
                     a message headed '[Timer {} fired ...]' carrying your prompt. Nothing \
                     happens before then, so carry on with anything else, or end your turn.",
                    timer.id,
                    timer.due_wall.format("%H:%M:%S"),
                    crate::timers::human(timer.due.saturating_duration_since(Instant::now())),
                    after,
                    timer.id
                ))
            }
            Err(why) => error(why),
        }
    }
}

#[tool_router(router = timers_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(
        description = "Set a timer that sends you a message later, so you can come back to something without the user having to remind you: check on a long build or test run, poll for a deploy or a CI result, retry something that is rate-limited, or do a recurring job on a schedule. Three kinds: 'once' after N seconds (like setTimeout), 'every' N seconds (like setInterval), and 'cron' for wall-clock times ('0 9 * * mon-fri'). When it fires you get your own prompt back as a new message, after your current step or once you are idle. Timers last only as long as this session. Cancel a repeating one as soon as it has done its job. Do not use this to wait on something you can simply run and wait for now."
    )]
    pub(crate) async fn timer(&self, params: Parameters<TimerParams>) -> Result<CallToolResult, McpError> {
        Ok(self.timer_in(params.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::*;

    fn call(server: &MyMcpServer, args: serde_json::Value) -> (bool, String) {
        let params: TimerParams = serde_json::from_value(args).unwrap();
        let result = server.timer_in(params);
        (!result.is_error.unwrap_or(false), result_text(&result))
    }

    #[test]
    fn timers_are_set_listed_and_cancelled_through_one_tool() {
        let server = test_server();
        let (ok, out) = call(&server, serde_json::json!({"action": "setTimeout", "seconds": 120, "prompt": "check job 1"}));
        assert!(ok, "{out}");
        assert!(out.starts_with("Timer 1 set"), "{out}");
        assert!(out.contains("fires once"), "{out}");

        let (ok, out) = call(&server, serde_json::json!({"action": "interval", "seconds": 600, "prompt": "poll CI"}));
        assert!(ok, "{out}");
        assert!(out.contains("until you cancel it") && out.contains("id 2"), "{out}");

        let (ok, out) = call(&server, serde_json::json!({"action": "cron", "cron": "0 9 * * mon-fri", "prompt": "standup notes"}));
        assert!(ok, "{out}");

        let (_, out) = call(&server, serde_json::json!({"action": "list"}));
        assert!(out.contains("Timer 1: once, after 2 min"), "{out}");
        assert!(out.contains("Timer 2: every 10 min"), "{out}");
        assert!(out.contains("Timer 3: cron '0 9 * * mon-fri'"), "{out}");

        let (ok, out) = call(&server, serde_json::json!({"action": "clearInterval", "id": 2}));
        assert!(ok, "{out}");
        let (ok, out) = call(&server, serde_json::json!({"action": "cancel", "id": 2}));
        assert!(!ok && out.contains("no timer 2"), "{out}");
        assert_eq!(server.timers.lock().unwrap().list().len(), 2);
    }

    #[test]
    fn a_call_missing_what_its_action_needs_says_which_field() {
        let server = test_server();
        let (ok, out) = call(&server, serde_json::json!({"action": "once", "prompt": "x"}));
        assert!(!ok && out.contains("'seconds'"), "{out}");
        let (ok, out) = call(&server, serde_json::json!({"action": "cron", "prompt": "x"}));
        assert!(!ok && out.contains("'cron'"), "{out}");
        let (ok, out) = call(&server, serde_json::json!({"action": "cron", "cron": "* * * * * *", "prompt": "x"}));
        assert!(!ok && out.contains("exactly five"), "{out}");
        let (ok, out) = call(&server, serde_json::json!({"action": "cancel"}));
        assert!(!ok && out.contains("'id'"), "{out}");
        let (ok, out) = call(&server, serde_json::json!({"action": "snooze"}));
        assert!(!ok && out.contains("not something timer can do"), "{out}");
    }

    #[test]
    fn the_timer_tool_is_offered_only_where_something_can_deliver_it() {
        let client = crate::LMStudioClient::new("http://127.0.0.1:1".to_string(), String::new(), None);
        let interactive = MyMcpServer::new(client, false, None, true);
        let tool = interactive.tool_router.list_all().into_iter().find(|t| t.name == "timer").expect("timer is routed");
        let schema = serde_json::to_string(&tool.input_schema).unwrap();
        assert!(schema.contains(r#"["once","every","cron","list","cancel"]"#), "{schema}");
        // An MCP host has no loop of ours to send the message into.
        assert!(!test_server().tool_router.list_all().iter().any(|t| t.name == "timer"));
    }
}
