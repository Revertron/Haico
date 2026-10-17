//! The two tools over the project memory store in `crate::memory`.
//!
//! `remember` writes a fact, `recall` reads one back in full. The store itself knows
//! nothing about either — it takes a root and does filesystem work, which is what lets
//! it be tested without a working directory to itself.

use super::MyMcpServer;
use crate::memory::{self, Memory};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::CallToolResult;
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct RememberParams {
    // The enum is in the schema as well as the description for the same reason
    // `TodoItem::status` carries one: a server that builds a decoding grammar from the
    // schema can then only ever produce one of the three, and `normalize_action` is the
    // net under the servers that constrain nothing.
    #[schemars(
        extend("enum" = ["add", "update", "delete"]),
        description = "One of: 'add' to record a new fact, 'update' to correct one that turned out wrong, 'delete' to forget one."
    )]
    pub(crate) action: String,
    #[schemars(
        description = "A short kebab-case name for this fact, which is also its identity: 'gui-is-behind-a-feature-flag', 'tests-must-run-in-parallel'. Name the fact, not the topic. For update and delete, pass the name of the existing memory exactly as your system prompt lists it."
    )]
    pub(crate) name: String,
    #[schemars(
        description = "One sentence stating the fact, for add and update. This line — and only this line — is what you are shown in every future session, so write the fact itself, not a label for it: 'the gui feature is off by default, so cargo build alone never compiles gui.rs' rather than 'build flags'."
    )]
    pub(crate) description: Option<String>,
    #[schemars(
        description = "The whole fact, for add and update: the detail, the reason, and what to do about it. A paragraph or two. This is what you get back when you call recall on this name."
    )]
    pub(crate) fact: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct RecallParams {
    #[schemars(
        description = "The name of the memory to read, as your system prompt lists it."
    )]
    pub(crate) name: String,
}

/// Map whatever the model called the action onto one of the three, or `None`.
///
/// Same bargain as `normalize_status`: a rejected call costs a whole round trip on a
/// local model to be told nothing except "use the other word", and that refusal lands in
/// the history where the next turn reads it. Anything still unrecognized is refused
/// rather than guessed at — guessing between "add" and "delete" is not a mistake worth
/// making on the user's behalf.
fn normalize_action(raw: &str) -> Option<&'static str> {
    match super::action_key(raw).as_str() {
        "add" | "create" | "new" | "write" | "save" | "store" | "remember" | "record"
        | "insert" => Some("add"),
        "update" | "edit" | "change" | "modify" | "replace" | "revise" | "correct" | "fix" => {
            Some("update")
        }
        "delete" | "remove" | "forget" | "drop" | "erase" | "clear" => Some("delete"),
        _ => None,
    }
}

use super::{error, success};

/// What is on file, for a message that has to tell the model where it went wrong.
///
/// Always the whole list rather than a count: the usual reason a name misses is that the
/// model half-remembered one, and the names are what turn a second failed guess into a
/// correct call.
fn known_names(root: &Path) -> String {
    let all = memory::load_all(root);
    if all.is_empty() {
        return "Nothing is stored yet.".to_string();
    }
    format!(
        "Stored memories: {}.",
        all.iter().map(|m| m.name.as_str()).collect::<Vec<_>>().join(", ")
    )
}

pub(crate) fn remember_in(root: &Path, params: RememberParams) -> CallToolResult {
    let Some(action) = normalize_action(&params.action) else {
        return error(format!(
            "'{}' is not something remember can do. Use action 'add', 'update' or 'delete'.",
            params.action
        ));
    };

    let Some(name) = memory::slug(&params.name) else {
        return error(
            "That name has no letters or digits in it. Give the memory a short kebab-case \
             name that says what the fact is about, like 'tests-must-run-in-parallel'."
                .to_string(),
        );
    };

    let existing = memory::load_one(root, &name);

    match action {
        "add" => {
            if let Some(existing) = existing {
                return error(format!(
                    "A memory named '{}' already exists: {}. Call remember again with \
                     action 'update' to correct it, or pick a different name for a \
                     separate fact.",
                    existing.name, existing.description
                ));
            }

            let count = memory::load_all(root).len();
            if count >= memory::MAX_MEMORIES {
                return error(format!(
                    "This project already has {} memories, which is the limit. Delete one \
                     that no longer holds before adding another.",
                    count
                ));
            }

            let Some(fact) = params.fact.as_deref().map(str::trim).filter(|f| !f.is_empty()) else {
                return error(
                    "Nothing to remember: pass the fact itself in 'fact'.".to_string(),
                );
            };
            if let Some(refusal) = too_long(fact) {
                return error(refusal);
            }

            // A missing description is derived rather than refused. It is the field that
            // matters most — it is the one that reaches every future session — but
            // bouncing the call costs a round trip on a local model and usually gets the
            // same fact back with a worse summary. Deriving it and saying so lets the
            // model improve it in one update if it cares to.
            let (description, derived) = match params.description.as_deref().map(str::trim) {
                Some(given) if !given.is_empty() => (given.to_string(), false),
                _ => (first_line(fact), true),
            };

            let memory = Memory { name, description, body: fact.to_string() };
            match memory::save(root, &memory) {
                Ok(_) => {
                    let mut text = format!(
                        "Remembered as '{}', in {}/{}.md. Every future session in this \
                         project is shown this line: {}",
                        memory.name, memory::MEMORY_DIR, memory.name, memory.description
                    );
                    if derived {
                        text.push_str(
                            "\n\nThat summary was taken from the first line of the fact \
                             because none was given. If it does not state the fact on its \
                             own, call remember with action 'update' and a better one.",
                        );
                    }
                    success(text)
                }
                Err(e) => error(format!("Could not write the memory: {}", e)),
            }
        }

        "update" => {
            // The name the model typed may not be the one on disk; resolving it here is
            // what stops an update silently becoming a second, nearly identical memory.
            let Some(resolved) = memory::resolve(root, &params.name) else {
                return error(format!(
                    "There is no memory named '{}' to update. {} Use action 'add' if this \
                     is a new fact.",
                    name,
                    known_names(root)
                ));
            };
            let Some(current) = memory::load_one(root, &resolved) else {
                return error(format!("Could not read the memory named '{}'.", resolved));
            };

            let fact = params.fact.as_deref().map(str::trim).filter(|f| !f.is_empty());
            let description = params.description.as_deref().map(str::trim).filter(|d| !d.is_empty());
            if fact.is_none() && description.is_none() {
                return error(format!(
                    "Nothing to change in '{}'. Pass a new 'fact', a new 'description', or \
                     both.",
                    resolved
                ));
            }
            if let Some(fact) = fact {
                if let Some(refusal) = too_long(fact) {
                    return error(refusal);
                }
            }

            let updated = Memory {
                name: resolved,
                description: description.map(str::to_string).unwrap_or(current.description),
                body: fact.map(str::to_string).unwrap_or(current.body),
            };
            match memory::save(root, &updated) {
                Ok(_) => success(format!(
                    "Updated '{}'. It now reads: {}",
                    updated.name, updated.description
                )),
                Err(e) => error(format!("Could not write the memory: {}", e)),
            }
        }

        _ => {
            let Some(resolved) = memory::resolve(root, &params.name) else {
                return error(format!(
                    "There is no memory named '{}' to delete. {}",
                    name,
                    known_names(root)
                ));
            };
            match memory::remove(root, &resolved) {
                Ok(true) => success(format!("Forgot '{}'.", resolved)),
                Ok(false) => error(format!("There is no memory named '{}' to delete.", resolved)),
                Err(e) => error(format!("Could not delete the memory: {}", e)),
            }
        }
    }
}

pub(crate) fn recall_in(root: &Path, params: RecallParams) -> CallToolResult {
    let Some(resolved) = memory::resolve(root, &params.name) else {
        return error(format!(
            "There is no memory named '{}'. {}",
            params.name.trim(),
            known_names(root)
        ));
    };
    let Some(memory) = memory::load_one(root, &resolved) else {
        return error(format!("Could not read the memory named '{}'.", resolved));
    };
    success(format!("# {}\n\n{}", memory.name, memory.body))
}

// The description is one line, so a fact that arrives as a paragraph needs cutting
// somewhere. Its first sentence is a better guess than its first hundred characters.
fn first_line(fact: &str) -> String {
    let first = fact.lines().find(|l| !l.trim().is_empty()).unwrap_or(fact);
    match first.find(". ") {
        Some(stop) => memory::one_line(&first[..stop + 1]),
        None => memory::one_line(first),
    }
}

// Refused rather than truncated. A memory silently stored with its second half missing
// would be recalled later as a fact that stops mid-argument, and nothing would say why.
fn too_long(fact: &str) -> Option<String> {
    let length = fact.chars().count();
    if length <= memory::MAX_FACT_CHARS {
        return None;
    }
    Some(format!(
        "That fact is {} characters, and a memory holds at most {}. A memory is one fact \
         worth carrying into every future session, not a document — cut it down to the \
         part that would change what you do, or split it into two memories.",
        length,
        memory::MAX_FACT_CHARS
    ))
}

fn project_root() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[tool_router(router = memory_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(
        description = "Record a fact about this project so it outlives this conversation. Call it the moment the user tells you to remember something — 'remember that...', 'don't forget...', 'keep in mind...' — and call it unprompted when you learn something about this project that would have saved you time had you known it at the start: a build flag that is not obvious, a test that must be run a particular way, a directory that is generated, a convention nobody wrote down, a decision and the reason behind it. Do not record what the code already says, what the project's instruction file already says, or anything that is only true of this one conversation. Memories are listed in your system prompt in every future session, one line each, so they cost something to keep: add the ones that will change what you do, and use action 'update' to correct one that turns out wrong or 'delete' when the user says to forget it."
    )]
    pub(crate) async fn remember(
        &self,
        params: Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(remember_in(&project_root(), params.0))
    }

    #[tool(
        description = "Read one stored project memory in full. Your system prompt lists every memory this project has by name with a one-line summary; when one of those lines touches what you are about to do, call this with its name to get the whole fact — the detail, the reason and what to do about it — before you act on the summary alone. It reads a small local file, so it is cheap: prefer it to guessing, and prefer it to asking the user something they already told you once."
    )]
    pub(crate) async fn recall(
        &self,
        params: Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(recall_in(&project_root(), params.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn add(root: &Path, name: &str, description: &str, fact: &str) -> CallToolResult {
        remember_in(
            root,
            RememberParams {
                action: "add".to_string(),
                name: name.to_string(),
                description: Some(description.to_string()),
                fact: Some(fact.to_string()),
            },
        )
    }

    fn text_of(result: &CallToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|block| block.as_text().map(|t| t.text.clone()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn failed(result: &CallToolResult) -> bool {
        result.is_error.unwrap_or(false)
    }

    #[test]
    fn a_fact_is_written_and_read_back() {
        let dir = TempDir::new("tool-memory-roundtrip");
        let added = add(dir.path(), "gui-feature", "gui.rs needs --features gui", "The gui feature is off by default.");
        assert!(!failed(&added), "{}", text_of(&added));

        let recalled = recall_in(dir.path(), RecallParams { name: "gui-feature".to_string() });
        assert!(text_of(&recalled).contains("The gui feature is off by default."));
    }

    #[test]
    fn adding_over_an_existing_name_says_to_update_instead() {
        let dir = TempDir::new("tool-memory-duplicate");
        add(dir.path(), "same", "first version", "body one");
        let again = add(dir.path(), "same", "second version", "body two");

        assert!(failed(&again));
        assert!(text_of(&again).contains("update"), "{}", text_of(&again));
        // And the first one is untouched, rather than half-overwritten.
        assert!(text_of(&recall_in(dir.path(), RecallParams { name: "same".to_string() }))
            .contains("body one"));
    }

    #[test]
    fn an_update_keeps_the_field_it_was_not_given() {
        let dir = TempDir::new("tool-memory-partial");
        add(dir.path(), "partial", "the original summary", "the original body");

        let updated = remember_in(
            dir.path(),
            RememberParams {
                action: "update".to_string(),
                name: "partial".to_string(),
                description: None,
                fact: Some("a corrected body".to_string()),
            },
        );
        assert!(!failed(&updated), "{}", text_of(&updated));
        assert!(text_of(&updated).contains("the original summary"));
        assert!(text_of(&recall_in(dir.path(), RecallParams { name: "partial".to_string() }))
            .contains("a corrected body"));
    }

    #[test]
    fn an_update_to_a_missing_memory_is_not_quietly_an_add() {
        let dir = TempDir::new("tool-memory-missing-update");
        let result = remember_in(
            dir.path(),
            RememberParams {
                action: "update".to_string(),
                name: "never-stored".to_string(),
                description: Some("d".to_string()),
                fact: Some("f".to_string()),
            },
        );
        assert!(failed(&result));
        assert!(memory::load_all(dir.path()).is_empty());
    }

    #[test]
    fn a_missing_name_comes_back_with_the_names_that_do_exist() {
        let dir = TempDir::new("tool-memory-names");
        add(dir.path(), "first-fact", "d", "b");
        add(dir.path(), "second-fact", "d", "b");

        let missed = recall_in(dir.path(), RecallParams { name: "third".to_string() });
        assert!(failed(&missed));
        let text = text_of(&missed);
        assert!(text.contains("first-fact") && text.contains("second-fact"), "{}", text);
    }

    #[test]
    fn the_model_s_own_word_for_the_action_is_accepted() {
        let dir = TempDir::new("tool-memory-synonyms");
        let created = remember_in(
            dir.path(),
            RememberParams {
                action: "create".to_string(),
                name: "synonym".to_string(),
                description: Some("d".to_string()),
                fact: Some("b".to_string()),
            },
        );
        assert!(!failed(&created), "{}", text_of(&created));

        let forgotten = remember_in(
            dir.path(),
            RememberParams {
                action: "forget".to_string(),
                name: "synonym".to_string(),
                description: None,
                fact: None,
            },
        );
        assert!(!failed(&forgotten), "{}", text_of(&forgotten));
        assert!(memory::load_all(dir.path()).is_empty());
    }

    #[test]
    fn an_unrecognized_action_is_refused_rather_than_guessed_at() {
        let dir = TempDir::new("tool-memory-bad-action");
        let result = remember_in(
            dir.path(),
            RememberParams {
                action: "merge".to_string(),
                name: "x".to_string(),
                description: Some("d".to_string()),
                fact: Some("b".to_string()),
            },
        );
        assert!(failed(&result));
        assert!(memory::load_all(dir.path()).is_empty());
    }

    #[test]
    fn a_missing_description_is_derived_and_said_to_be() {
        let dir = TempDir::new("tool-memory-derived");
        let added = remember_in(
            dir.path(),
            RememberParams {
                action: "add".to_string(),
                name: "derived".to_string(),
                description: None,
                fact: Some("The build needs nightly. There is a reason, and it is long.".to_string()),
            },
        );
        assert!(!failed(&added), "{}", text_of(&added));
        assert_eq!(
            memory::load_one(dir.path(), "derived").unwrap().description,
            "The build needs nightly."
        );
        assert!(text_of(&added).contains("update"), "the model was not told it was derived");
    }

    #[test]
    fn an_oversized_fact_is_refused_rather_than_stored_half_way() {
        let dir = TempDir::new("tool-memory-oversize");
        let huge = "x".repeat(memory::MAX_FACT_CHARS + 1);
        let result = add(dir.path(), "huge", "d", &huge);

        assert!(failed(&result));
        assert!(memory::load_all(dir.path()).is_empty(), "a refused fact was stored anyway");
    }

    #[test]
    fn a_name_that_slugs_to_nothing_is_refused() {
        let dir = TempDir::new("tool-memory-empty-name");
        let result = add(dir.path(), "!!!", "d", "b");
        assert!(failed(&result));
        assert!(memory::load_all(dir.path()).is_empty());
    }
}
