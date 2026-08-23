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


#[derive(Serialize, Deserialize, JsonSchema, Clone)]
pub(crate) struct TodoItem {
    #[schemars(description = "What needs doing, phrased as an action (e.g. 'Add the --context flag')")]
    pub(crate) task: String,
    #[schemars(
        description = "One of: 'pending', 'in_progress', 'done'. Keep exactly one item in_progress at a time."
    )]
    pub(crate) status: String,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct TodoWriteParams {
    #[schemars(
        description = "The complete list, in order. This replaces the previous list entirely, so include finished items with status 'done' rather than dropping them."
    )]
    pub(crate) todos: Vec<TodoItem>,
}


#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct PresentPlanParams {
    #[schemars(
        description = "The plan itself, as markdown. Name the specific files and functions you will change and what each change does, in the order you will do them. Call out anything you are unsure about so the user can correct it. Keep it short enough to be checked in one read."
    )]
    pub(crate) plan: String,
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
        description = "Track a multi-step task so you do not lose the thread. Call this at the start of any job that needs more than a couple of steps, writing out the whole plan; then call it again after each step to mark what is done and what you are on now. Keeping the list in front of you is what stops you forgetting a requirement halfway through, repeating work already finished, or stopping early while items are still pending. Pass the complete list every time — it replaces the previous one, so carry finished items over with status 'done'. Skip it for genuinely single-step requests, where it is just noise."
    )]
    pub(crate) async fn todo_write(
        &self,
        params: Parameters<TodoWriteParams>,
    ) -> Result<CallToolResult, McpError> {
        const VALID: [&str; 3] = ["pending", "in_progress", "done"];

        for item in &params.0.todos {
            if !VALID.contains(&item.status.as_str()) {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Invalid status '{}' for '{}'. Use one of: pending, in_progress, done.",
                    item.status, item.task
                ))]));
            }
        }

        let in_progress = params
            .0
            .todos
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

        *self.todos.lock().unwrap() = params.0.todos.clone();

        // Shown to the user as it changes, so the plan is visible rather than buried in
        // tool output only the model sees.
        if self.interactive {
            println!("\n  Plan:");
            for item in &params.0.todos {
                let mark = match item.status.as_str() {
                    "done" => "x",
                    "in_progress" => "»",
                    _ => " ",
                };
                println!("   [{}] {}", mark, item.task);
            }
            println!();
        }

        let done = params
            .0
            .todos
            .iter()
            .filter(|item| item.status == "done")
            .count();
        let total = params.0.todos.len();
        let remaining: Vec<&str> = params
            .0
            .todos
            .iter()
            .filter(|item| item.status != "done")
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
            ("Add the flag", "done"),
            ("Write tests", "in_progress"),
            ("Update docs", "pending"),
        ])
        .await;
        assert!(ok);
        assert!(out.contains("1/3 done"), "got: {}", out);
        // A finished item must not still be listed as outstanding.
        assert!(!out.contains("Add the flag"), "got: {}", out);

        let (ok, out) = write(vec![
            ("Add the flag", "done"),
            ("Write tests", "done"),
            ("Update docs", "done"),
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

}
