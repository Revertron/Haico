use super::MyMcpServer;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use chrono;

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct CalculatorParams {
    #[schemars(description = "The arithmetic operation to perform: 'add', 'subtract', 'multiply', or 'divide'")]
    pub(crate) operation: String,
    #[schemars(description = "The first operand")]
    pub(crate) a: f64,
    #[schemars(description = "The second operand")]
    pub(crate) b: f64,
}


#[derive(Serialize, Deserialize, JsonSchema, Clone, PartialEq)]
pub(crate) struct TodoItem {
    #[schemars(description = "What needs doing, phrased as an action (e.g. 'Add the --context flag')")]
    pub(crate) task: String,
    // The three values are in the schema and not only in the description because a server
    // that builds a decoding grammar from the schema (llama.cpp, LM Studio) can then only
    // ever produce one of them — a constrained model cannot mis-say the word, whereas a
    // description is a suggestion and the first thing to lose an argument with the model's
    // own prior. The field stays a String rather than becoming a Rust enum so that a server
    // which constrains nothing still reaches `normalize_status` instead of failing to
    // deserialize: the schema in front, the synonym table as the net under it.
    #[schemars(
        extend("enum" = ["pending", "in_progress", "completed"]),
        description = "One of: 'pending', 'in_progress', 'completed'. Keep exactly one item in_progress at a time."
    )]
    pub(crate) status: String,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct TodoWriteParams {
    #[schemars(
        description = "The complete list, in order. This replaces the previous list entirely, so include finished items with status 'completed' rather than dropping them."
    )]
    pub(crate) todos: Vec<TodoItem>,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct AskUserParams {
    #[schemars(
        description = "The question, as one plain sentence. Say what you are deciding and why it matters, not just 'which one?'."
    )]
    pub(crate) question: String,
    #[schemars(
        description = "The answers to offer, two to eight of them, in the order you would recommend. Write each one as long as it needs to be — a whole sentence naming the trade-off is far more use than a bare word. Do not add an 'other' or 'you decide' option: one is always added for you."
    )]
    pub(crate) options: Vec<String>,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct PresentPlanParams {
    #[schemars(
        description = "The plan itself, as markdown. Name the specific files and functions you will change and what each change does, in the order you will do them. Call out anything you are unsure about so the user can correct it. Keep it short enough to be checked in one read."
    )]
    pub(crate) plan: String,
}


/// Map whatever the model called a status onto one of the three real ones, or `None`.
///
/// `completed` is canonical because it is the word every agent-tuned model already has:
/// Claude Code's own todo tool ships `pending` / `in_progress` / `completed`, so that is
/// what the training data says, and a local model half a context window into a job falls
/// back on that prior long before it re-reads this tool's description. Renaming `done`
/// to match it turns the most common failure into no failure at all.
///
/// The rest of the table is the same argument applied to the words the prior does not
/// cover. A rejected call is not free: it costs a whole round trip on a local model to
/// say nothing except "use the other word", and it lands in the history where the next
/// turn reads it. Anything still unrecognized is refused rather than guessed at — a
/// status nobody meant, silently filed as pending, is a task the model believes it has
/// already dealt with.
pub(crate) fn normalize_status(raw: &str) -> Option<&'static str> {
    let key: String = raw
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c == '-' || c == ' ' { '_' } else { c })
        .collect();
    match key.as_str() {
        "pending" | "todo" | "to_do" | "not_started" | "open" | "new" | "queued" | "waiting" => {
            Some("pending")
        }
        "in_progress" | "inprogress" | "doing" | "active" | "working" | "started" | "wip"
        | "current" => Some("in_progress"),
        "completed" | "complete" | "done" | "finished" | "closed" | "x" => Some("completed"),
        _ => None,
    }
}

#[tool_router(router = basics_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(description = "Use for exact arithmetic instead of computing in your head — reach for this whenever a task involves numbers to add, subtract, multiply, or divide, especially with large values or decimals, or when a wrong result would be costly. Performs one basic arithmetic operation (add, subtract, multiply, divide) on two numbers.")]
    pub(crate) async fn calculator(&self, params: Parameters<CalculatorParams>) -> Result<CallToolResult, McpError> {
        let operation = params.0.operation.as_str();
        let a = params.0.a;
        let b = params.0.b;

        let result = match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Ok(CallToolResult::error(vec![ContentBlock::text("Division by zero")]));
                }
                a / b
            }
            _ => return Ok(CallToolResult::error(vec![ContentBlock::text("Unknown operation")])),
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(format!("Result: {}", result))]))
    }


    #[tool(description = "Call this whenever anything depends on today's date — you have no reliable internal clock and your best guess is usually stale or wrong, so never assume the date. Use it before answering 'what is today', resolving relative dates ('yesterday', 'next Monday', 'in two weeks'), computing ages, durations or deadlines, or naming files and logs by date. Returns the current local date in YYYY-MM-DD format (e.g. 2026-06-22).")]
    pub(crate) async fn get_date(&self) -> Result<CallToolResult, McpError> {
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        Ok(CallToolResult::success(vec![ContentBlock::text(date)]))
    }


    #[tool(description = "Call this whenever you need the current wall-clock time or timezone — you cannot know it on your own and must not guess. Use it for 'what time is it', scheduling, measuring how long something takes, timestamping, or converting between time zones. Returns the current local time in HH:MM:SS format with UTC offset (e.g. 14:30:05 +03:00).")]
    pub(crate) async fn get_time(&self) -> Result<CallToolResult, McpError> {
        let time = chrono::Local::now().format("%H:%M:%S %:z").to_string();
        Ok(CallToolResult::success(vec![ContentBlock::text(time)]))
    }


    #[tool(
        description = "Track a multi-step task so you do not lose the thread. Call this at the start of any job that needs more than a couple of steps, writing out the whole plan; then call it again after each step to mark what is done and what you are on now. Keeping the list in front of you is what stops you forgetting a requirement halfway through, repeating work already finished, or stopping early while items are still pending. Pass the complete list every time — it replaces the previous one, so carry finished items over with status 'completed'. Skip it for genuinely single-step requests, where it is just noise."
    )]
    pub(crate) async fn todo_write(
        &self,
        params: Parameters<TodoWriteParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut todos = params.0.todos;

        for item in &mut todos {
            match normalize_status(&item.status) {
                Some(status) => item.status = status.to_string(),
                None => {
                    let hint = format!(
                        "Invalid status '{}' for '{}'. Use one of: pending, in_progress, completed.",
                        item.status, item.task
                    );
                    return Ok(CallToolResult::error(vec![ContentBlock::text(hint)]));
                }
            }
        }

        let in_progress = todos
            .iter()
            .filter(|item| item.status == "in_progress")
            .count();
        if in_progress > 1 {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "{} items are marked in_progress. Work on one at a time: mark the rest \
                 pending until you get to them.",
                in_progress
            ))]));
        }

        *self.todos.lock().unwrap() = todos.clone();

        // Nothing to print in the REPL: the list is pinned above the input box, and the
        // next draw picks the new one up from here. A piped session has no bottom region
        // to pin anything to, so there it still goes into the transcript — buried under
        // the next few tool calls, but shown, which beats not shown at all.
        if self.interactive && !crate::out::has_region() {
            outln!("\n  Plan:");
            for item in &todos {
                let mark = match item.status.as_str() {
                    "completed" => "x",
                    "in_progress" => "»",
                    _ => " ",
                };
                outln!("   [{}] {}", mark, item.task);
            }
            outln!();
        }

        let done = todos
            .iter()
            .filter(|item| item.status == "completed")
            .count();
        let total = todos.len();
        let remaining: Vec<&str> = todos
            .iter()
            .filter(|item| item.status != "completed")
            .map(|item| item.task.as_str())
            .collect();

        let summary = if remaining.is_empty() {
            format!("All {} item(s) done.", total)
        } else {
            format!(
                "{}/{} done. Still to do: {}",
                done,
                total,
                remaining.join("; ")
            )
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(summary)]))
    }

    #[tool(
        description = "Ask the user a question and let them pick the answer. Reach for this the moment you are genuinely unsure and the answer would change what you build: two designs that both look reasonable, which library or dependency to use, a requirement that could mean either of two things, a name or format to settle on, whether to replace something or keep it. Ask at the point the decision comes up, not after you have already built one of the branches. One question now is cheaper than the wrong thing built well — and far cheaper than deciding, doubting the decision, and deciding again. If you notice yourself turning the same choice over for a second time, that is the signal: stop and ask. Offer the alternatives you are actually weighing, spelled out; the user can always type an answer of their own. Do not ask about anything you could settle yourself by reading the code, and do not ask the user to re-confirm work they already asked for."
    )]
    pub(crate) async fn ask_user(&self, params: Parameters<AskUserParams>) -> Result<CallToolResult, McpError> {
        // In the REPL this never runs: the turn loop intercepts the call and puts the
        // question on the screen itself, because the keyboard lives out there. Reaching
        // this body means there is nobody on the other end to ask — an MCP client calling
        // the tool directly — so the question is still checked over, and then the model is
        // told plainly that it is on its own.
        match crate::ask::normalize(&params.0.question, &params.0.options) {
            Err(reason) => Ok(CallToolResult::error(vec![ContentBlock::text(reason)])),
            Ok(_) => Ok(CallToolResult::success(vec![ContentBlock::text(
                "There is nobody to ask in this context. Decide it yourself, say in one \
                 line which way you went and why, and carry on.",
            )])),
        }
    }

    #[tool(description = "Switch yourself into plan mode, where you can read and investigate but cannot change anything until the user approves a plan. Call this first, before doing anything else, whenever the user asks for a plan, a proposal, an approach, or how you would go about something — rather than asking you to just do it. Also reach for it when a request is large or ambiguous enough that building the wrong thing would waste real work. Do not use it for small, clear changes the user plainly wants carried out now. Once in plan mode, investigate the code and then call present_plan.")]
    pub(crate) async fn start_plan(&self) -> Result<CallToolResult, McpError> {
        // As with present_plan, the REPL intercepts this and flips the mode itself. This
        // body is what an MCP client sees, where there is no mode to enter.
        Ok(CallToolResult::success(vec![ContentBlock::text(
            "There is no plan mode in this context. Investigate with the reading tools and \
             write out your plan for the caller instead of making changes.",
        )]))
    }

    #[tool(description = "Present your finished plan to the user for approval. Only available in plan mode, and the only way out of it: nothing you propose can be built until the user has approved it here. Investigate the real code with read, grep and glob_search first, then submit a plan that names the specific files and functions involved. The user will either approve it, at which point you start work, or send back corrections for you to fold in and submit again.")]
    pub(crate) async fn present_plan(&self, params: Parameters<PresentPlanParams>) -> Result<CallToolResult, McpError> {
        // In the REPL this never runs: the turn loop intercepts the call, shows the plan
        // and asks the user, because the mode it would change lives out there. Reaching
        // this body means plan mode is not in play at all — an MCP client calling the tool
        // directly — and there is nobody in that path to approve anything.
        let plan = params.0.plan.trim();
        if plan.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(
                "The plan was empty. Send the plan itself in the 'plan' field.",
            )]));
        }
        Ok(CallToolResult::success(vec![ContentBlock::text(
            "Plan recorded. No approval step is available in this context, so treat it as \
             pending and wait for the caller.",
        )]))
    }

    // Snapshot a file just before changing it. A file that does not exist yet is recorded as
    // absent, so undoing its creation removes it rather than leaving an empty stub.
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

    #[tokio::test]
    async fn todo_write_tracks_progress_and_rejects_incoherent_lists() {
        let server = test_server();

        let write = |items: Vec<(&str, &str)>| {
            let todos = items
                .into_iter()
                .map(|(task, status)| crate::TodoItem {
                    task: task.to_string(),
                    status: status.to_string(),
                })
                .collect();
            let server = server.clone();
            async move {
                let result = server
                    .todo_write(rmcp::handler::server::wrapper::Parameters(
                        crate::TodoWriteParams { todos },
                    ))
                    .await
                    .unwrap();
                (!result.is_error.unwrap_or(false), result_text(&result))
            }
        };

        // The reply names what is left, so the model is reminded every time it checks in.
        let (ok, out) = write(vec![
            ("Add the flag", "in_progress"),
            ("Write tests", "pending"),
            ("Update docs", "pending"),
        ])
        .await;
        assert!(ok, "got: {}", out);
        assert!(out.contains("0/3 done"), "got: {}", out);
        assert!(out.contains("Write tests"), "got: {}", out);
        assert_eq!(server.todos.lock().unwrap().len(), 3);

        let (ok, out) = write(vec![
            ("Add the flag", "completed"),
            ("Write tests", "in_progress"),
            ("Update docs", "pending"),
        ])
        .await;
        assert!(ok);
        assert!(out.contains("1/3 done"), "got: {}", out);
        // A finished item must not still be listed as outstanding.
        assert!(!out.contains("Add the flag"), "got: {}", out);

        let (ok, out) = write(vec![
            ("Add the flag", "completed"),
            ("Write tests", "completed"),
            ("Update docs", "completed"),
        ])
        .await;
        assert!(ok);
        assert!(out.contains("All 3 item(s) done"), "got: {}", out);

        // Two things at once is how a model loses track of both.
        let (ok, out) = write(vec![("a", "in_progress"), ("b", "in_progress")]).await;
        assert!(!ok);
        assert!(out.contains("in_progress"), "got: {}", out);

        // A typo'd status is caught rather than stored and silently ignored later.
        let (ok, out) = write(vec![("a", "compelted")]).await;
        assert!(!ok);
        assert!(out.contains("Invalid status"), "got: {}", out);

        // Neither rejection disturbed the last good list.
        assert_eq!(server.todos.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn todo_write_takes_the_words_a_model_actually_uses() {
        let server = test_server();

        let todos = vec![
            ("Add the flag", "COMPLETED"),
            ("Write tests", "Done"),
            ("Update docs", "in-progress"),
            ("Ship it", "not started"),
        ]
        .into_iter()
        .map(|(task, status)| crate::TodoItem {
            task: task.to_string(),
            status: status.to_string(),
        })
        .collect();

        let result = server
            .todo_write(rmcp::handler::server::wrapper::Parameters(
                crate::TodoWriteParams { todos },
            ))
            .await
            .unwrap();
        assert!(!result.is_error.unwrap_or(false), "got: {}", result_text(&result));
        assert!(result_text(&result).contains("2/4 done"), "got: {}", result_text(&result));

        // Stored canonically, so everything downstream still only has three words to know.
        let stored: Vec<String> = server
            .todos
            .lock()
            .unwrap()
            .iter()
            .map(|item| item.status.clone())
            .collect();
        assert_eq!(
            stored,
            ["completed", "completed", "in_progress", "pending"]
        );
    }

    #[test]
    fn todo_schema_names_the_three_statuses() {
        // The description is only a suggestion; the enum in the schema is what a server
        // doing constrained decoding turns into a grammar, so it has to reach the wire.
        let server = test_server();
        let tool = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|tool| tool.name == "todo_write")
            .expect("todo_write is routed");
        let schema = serde_json::to_string(&tool.input_schema).unwrap();
        assert!(
            schema.contains(r#"["pending","in_progress","completed"]"#),
            "got: {}",
            schema
        );
    }
}
