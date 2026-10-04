//! The `sub_agent` tool's schema and description. The loop itself is `crate::subagent`,
//! on the core's side of the seam, because every call it makes has to clear the same
//! gates the main model's calls do — and the workspace, the permissions and the reviewer
//! live on the agent, not on this struct.

use super::MyMcpServer;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub(crate) const SUB_AGENT: &str = "sub_agent";

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct SubAgentParams {
    #[schemars(
        description = "The whole task, written for someone who has seen none of this \
                       conversation: which files, logs, commands or directories to look at, \
                       what to find out, and exactly what to report back — a verdict, a list, \
                       the lines that matter, a yes or no with the reason. Paths must be \
                       spelled out in full. The answer you get back is only as good as this \
                       description."
    )]
    pub(crate) task: String,
}

#[tool_router(router = subagent_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(
        description = "Hand a self-contained task to a helper with a fresh, empty context and \
                       get back only its short report. The helper can read and search files, \
                       list directories, run commands, read background job output and look \
                       things up on the web; it cannot edit files. Everything it reads stays \
                       in its own context and is thrown away when it answers, so your context \
                       only grows by the report. Use it whenever answering would mean reading \
                       a lot to keep a little: a long log or build output to find the one \
                       error in; a large file or a whole directory to pull a few facts out of; \
                       a test run whose failures you need summarized; \"which of these files \
                       call X and how\"; checking a claim across many files and returning a \
                       verdict. Do not use it for a single small file you are about to edit — \
                       read that yourself, since you need its exact contents to edit it. \
                       Describe the task completely (it has not seen this conversation) and \
                       say what shape the answer should have."
    )]
    pub(crate) async fn sub_agent(
        &self,
        params: Parameters<SubAgentParams>,
    ) -> Result<CallToolResult, McpError> {
        // In the REPL this never runs: the turn loop intercepts the call and runs the loop
        // itself, because the gates every inner call has to clear live on the agent. The
        // route is removed in MCP server mode, so reaching this body means a caller found a
        // way to it anyway — and it has no gates to offer such a caller.
        let _ = params;
        Ok(CallToolResult::error(vec![ContentBlock::text(
            "sub_agent is only available inside haico's own conversation loop. Do the work \
             with the other tools directly.",
        )]))
    }
}
