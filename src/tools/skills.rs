//! The `skill` tool over the store in `crate::skills`.
//!
//! The store takes roots and does filesystem work; this supplies the roots for the working
//! directory and words the answer. Same split as `memory`, for the same reason.

use super::MyMcpServer;
use crate::skills;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct SkillParams {
    #[schemars(description = "The name of the skill to load, exactly as your system prompt lists it.")]
    pub(crate) name: String,
}

pub(crate) fn skill_in(roots: &[PathBuf], params: SkillParams) -> CallToolResult {
    let Some(skill) = skills::resolve(roots, &params.name) else {
        // The whole list rather than a bare "not found": the usual reason a name misses is
        // that the model half-remembered one, and the names are what fix the second try.
        let known: Vec<String> = skills::discover(roots).into_iter().map(|s| s.name).collect();
        let listing = if known.is_empty() {
            "No skills are installed, so carry on without one.".to_string()
        } else {
            format!("Available skills: {}.", known.join(", "))
        };
        return CallToolResult::error(vec![ContentBlock::text(format!(
            "There is no skill named '{}'. {}",
            params.name.trim(),
            listing
        ))]);
    };
    match skills::body(&skill) {
        Some(body) => {
            CallToolResult::success(vec![ContentBlock::text(skills::invocation(&skill, &body, None))])
        }
        None => CallToolResult::error(vec![ContentBlock::text(format!(
            "Could not read the instructions for skill '{}'.",
            skill.name
        ))]),
    }
}

#[tool_router(router = skills_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(
        description = "Load a skill: a set of instructions someone wrote for doing one particular kind of task well in this project or on this machine. Your system prompt lists every available skill by name with a line saying when it applies. When the user's request matches one of those lines, call this with its name BEFORE starting the work, then follow the instructions it returns — they may point at scripts or reference files in the skill's directory. It reads a small local file, so it is cheap: load the skill rather than guessing at what it says."
    )]
    pub(crate) async fn skill(
        &self,
        params: Parameters<SkillParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(skill_in(&skills::roots_for_cwd(), params.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use std::fs;

    fn text_of(result: &CallToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|block| block.as_text().map(|t| t.text.clone()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn a_skill_comes_back_with_its_directory() {
        let dir = TempDir::new("tool-skill-load");
        let skill_dir = dir.path().join("bump-version");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), "---\ndescription: d\n---\nRun scripts/bump.py.").unwrap();

        let result = skill_in(&[dir.path().to_path_buf()], SkillParams { name: "bump-version".into() });
        assert!(!result.is_error.unwrap_or(false));
        let text = text_of(&result);
        assert!(text.contains("Run scripts/bump.py."), "{}", text);
        assert!(text.contains(&crate::util::display_path(&skill_dir)), "{}", text);
    }

    #[test]
    fn a_miss_lists_what_there_is() {
        let dir = TempDir::new("tool-skill-miss");
        for name in ["alpha", "beta"] {
            fs::create_dir_all(dir.path().join(name)).unwrap();
            fs::write(dir.path().join(name).join("SKILL.md"), "x").unwrap();
        }
        let result = skill_in(&[dir.path().to_path_buf()], SkillParams { name: "gamma".into() });
        assert!(result.is_error.unwrap_or(false));
        let text = text_of(&result);
        assert!(text.contains("alpha") && text.contains("beta"), "{}", text);
    }
}
