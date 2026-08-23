//! Which paths the file tools are allowed to touch.
//!
//! The working directory is writable, `/add` makes another directory writable for the rest
//! of the session, and `read_paths` in the config file names directories that may be read
//! but never written. Everything else is refused outright rather than offered to the user
//! as a prompt: the point is that a model which has been talked into reading `~/.ssh` or a
//! `.env` two directories over does not get the chance to ask.
//!
//! This bounds the **file tools**, not the machine. `run_command` runs a shell, and a shell
//! can read and write anywhere the user can — that boundary is the command approval in
//! `permissions.rs`, and it is a separate question from this one.

use crate::util::{absolute_path, display_path};
use serde_json::Value;
use std::path::{Path, PathBuf};

// The file tools, and whether the call is about to change something.
//
// Anything not named here is not path-bounded, so it passes straight through: the workspace
// has nothing to say about a web search or a calculation.
fn path_argument(tool_name: &str, arguments: &Value) -> Option<(PathBuf, bool)> {
    let (raw, writes) = match tool_name {
        "read" => (arguments["file_path"].as_str(), false),
        "write_file" | "edit" | "multi_edit" => (arguments["file_path"].as_str(), true),
        "ls" => (arguments["path"].as_str(), false),
        // Optional for these two, and absent means "the working directory", which is always
        // allowed — so a missing path is not a denial.
        "glob_search" | "grep" => (arguments["path"].as_str(), false),
        _ => return None,
    };

    let raw = raw.map(str::trim).filter(|p| !p.is_empty())?;
    Some((PathBuf::from(raw), writes))
}

#[derive(Debug, Clone)]
pub(crate) struct Workspace {
    // Readable and writable. The working directory is always the first entry.
    writable: Vec<PathBuf>,
    // Readable only, from the config file. A path here can never become writable, because
    // the whole reason to list it is that it is somewhere the model has business reading
    // and no business changing.
    readable: Vec<PathBuf>,
}

impl Workspace {
    pub(crate) fn new(cwd: &Path, readable: &[String]) -> Self {
        Self {
            writable: vec![absolute_path(cwd)],
            readable: readable
                .iter()
                .map(|p| absolute_path(&expand_home(p)))
                .collect(),
        }
    }

    // Add a directory to the session's writable set.
    //
    // It has to exist: `/add` is typed by hand, and a typo that silently widened the
    // boundary to a directory that does not exist yet would be discovered later, by the
    // model writing somewhere unexpected.
    pub(crate) fn add(&mut self, path: &Path) -> Result<PathBuf, String> {
        let resolved = absolute_path(&expand_home(&path.to_string_lossy()));

        if !resolved.is_dir() {
            return Err(format!(
                "{} is not a directory that exists — nothing was added.",
                display_path(&resolved)
            ));
        }
        if self.writable.iter().any(|root| root == &resolved) {
            return Err(format!("{} is already in the workspace.", display_path(&resolved)));
        }

        self.writable.push(resolved.clone());
        Ok(resolved)
    }

    pub(crate) fn may_read(&self, path: &Path) -> bool {
        let path = absolute_path(path);
        self.writable.iter().chain(self.readable.iter()).any(|root| path.starts_with(root))
    }

    pub(crate) fn may_write(&self, path: &Path) -> bool {
        let path = absolute_path(path);
        self.writable.iter().any(|root| path.starts_with(root))
    }

    // The root a path falls under, so an approval can be keyed to it. Approving writes in
    // one project must not silently extend to a directory added later.
    pub(crate) fn writable_root_for(&self, path: &Path) -> Option<&Path> {
        let path = absolute_path(path);
        self.writable
            .iter()
            .find(|root| path.starts_with(root))
            .map(|root| root.as_path())
    }

    pub(crate) fn writable_roots(&self) -> &[PathBuf] {
        &self.writable
    }

    pub(crate) fn readable_roots(&self) -> &[PathBuf] {
        &self.readable
    }

    // What the model is told about the boundary, before it walks into it.
    //
    // Discovering the workspace through refusals costs a round each time and reads as the
    // tool being broken. It also has to be said out loud that a second root exists at all:
    // a model working on a backend has no other way to learn that the client directory
    // beside it was added and is fair game.
    pub(crate) fn note(&self) -> String {
        let list = |roots: &[PathBuf]| {
            roots
                .iter()
                .map(|r| format!("- {}", display_path(r)))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let mut note = format!(
            "\n\n# Workspace\n\nThe file tools may read and write only under these \
             directories:\n{}",
            list(&self.writable)
        );

        if !self.readable.is_empty() {
            note.push_str(&format!(
                "\n\nThese may be read but never written:\n{}",
                list(&self.readable)
            ));
        }

        note.push_str(
            "\n\nAny other path is refused outright, for reads as well as writes, and there \
             is no prompt to approve it. Do not try to reach a refused path another way — \
             say what you need and let the user decide whether to add it.",
        );
        note
    }

    // Why this call cannot proceed, if it cannot.
    //
    // The message is written for the model as much as the user: it says which boundary was
    // hit and that the user is the one who can move it, so the answer is to ask rather than
    // to try the same path a different way.
    pub(crate) fn refusal(&self, tool_name: &str, arguments: &Value) -> Option<String> {
        let (path, writes) = path_argument(tool_name, arguments)?;
        let allowed = if writes { self.may_write(&path) } else { self.may_read(&path) };
        if allowed {
            return None;
        }

        let resolved = absolute_path(&path);
        let roots = self
            .writable
            .iter()
            .map(|r| display_path(r))
            .collect::<Vec<_>>()
            .join(", ");

        // A path inside a read-only root that was asked to be written gets told exactly
        // that, rather than the generic "outside the workspace" — otherwise the model
        // retries the same write, having been told something that looks untrue.
        if writes && self.may_read(&path) {
            return Some(format!(
                "Refused: '{}' would write to {}, which is configured as read-only. \
                 Writable directories are: {}. Ask the user to run '/add <path>' if this \
                 file really needs changing.",
                tool_name,
                display_path(&resolved),
                roots
            ));
        }

        Some(format!(
            "Refused: '{}' tried to reach {}, which is outside the workspace. The workspace \
             is: {}. Do not try to reach this path another way — ask the user to run \
             '/add <path>' if you genuinely need it.",
            tool_name,
            display_path(&resolved),
            roots
        ))
    }
}

// Expand a leading "~" against the user's home directory, so a hand-written config file can
// say "~/.cargo/registry" without knowing which drive the profile lives on.
fn expand_home(path: &str) -> PathBuf {
    let trimmed = path.trim();
    let Some(rest) = trimmed.strip_prefix('~') else {
        return PathBuf::from(trimmed);
    };
    let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
        return PathBuf::from(trimmed);
    };
    let rest = rest.trim_start_matches(['/', '\\'].as_slice());
    if rest.is_empty() {
        PathBuf::from(home)
    } else {
        PathBuf::from(home).join(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scratch_dirs() -> (PathBuf, PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!("haico-ws-{}", std::process::id()));
        let project = base.join("project");
        let client = base.join("client");
        let elsewhere = base.join("elsewhere");
        for dir in [&project, &client, &elsewhere] {
            std::fs::create_dir_all(dir).unwrap();
        }
        (project, client, elsewhere)
    }

    #[test]
    fn the_working_directory_is_writable_and_everywhere_else_is_not() {
        let (project, _, elsewhere) = scratch_dirs();
        let workspace = Workspace::new(&project, &[]);

        assert!(workspace.may_write(&project.join("src/main.rs")));
        assert!(workspace.may_read(&project.join("Cargo.toml")));

        // The default posture: nothing outside the project, not even a read. This is the
        // whole point — a secret two directories over is not reachable by asking nicely.
        assert!(!workspace.may_read(&elsewhere.join("secrets.env")));
        assert!(!workspace.may_write(&elsewhere.join("secrets.env")));
    }

    #[test]
    fn adding_a_directory_makes_it_a_first_class_part_of_the_workspace() {
        let (project, client, elsewhere) = scratch_dirs();
        let mut workspace = Workspace::new(&project, &[]);

        assert!(!workspace.may_write(&client.join("app.ts")));
        workspace.add(&client).unwrap();
        assert!(workspace.may_write(&client.join("app.ts")));
        assert!(workspace.may_read(&client.join("app.ts")));
        // The original root is untouched by the addition.
        assert!(workspace.may_write(&project.join("src/main.rs")));
        // And nothing else came along with it.
        assert!(!workspace.may_read(&elsewhere.join("secrets.env")));

        // Each root is its own approval key, so allowing writes in one project does not
        // silently cover a directory added afterwards.
        assert_ne!(
            workspace.writable_root_for(&project.join("a.rs")),
            workspace.writable_root_for(&client.join("b.ts"))
        );

        // A typo must not widen the boundary to a path that does not exist.
        assert!(workspace.add(Path::new("definitely-not-here-xyz")).is_err());
        // Nor should the same directory pile up twice.
        assert!(workspace.add(&client).is_err());
        assert_eq!(workspace.writable_roots().len(), 2);
    }

    #[test]
    fn configured_read_paths_can_be_read_but_never_written() {
        let (project, library, _) = scratch_dirs();
        let workspace = Workspace::new(&project, &[library.display().to_string()]);

        assert!(workspace.may_read(&library.join("src/lib.rs")));
        // Readable is not a step on the way to writable: the reason to list a path is that
        // the model has business reading it and none changing it.
        assert!(!workspace.may_write(&library.join("src/lib.rs")));
        assert_eq!(workspace.readable_roots().len(), 1);
    }

    #[test]
    fn refusals_name_the_boundary_that_was_hit() {
        let (project, library, elsewhere) = scratch_dirs();
        let workspace = Workspace::new(&project, &[library.display().to_string()]);

        // In-bounds work says nothing at all.
        assert!(workspace
            .refusal("write_file", &json!({"file_path": project.join("a.rs").display().to_string()}))
            .is_none());
        assert!(workspace
            .refusal("read", &json!({"file_path": library.join("lib.rs").display().to_string()}))
            .is_none());

        // A read right out of bounds, and the message points at the way to fix it rather
        // than leaving the model to invent a workaround.
        let denied = workspace
            .refusal("read", &json!({"file_path": elsewhere.join("secrets.env").display().to_string()}))
            .expect("should refuse");
        assert!(denied.contains("outside the workspace"), "got: {}", denied);
        assert!(denied.contains("/add"), "got: {}", denied);

        // A write into a read-only root is told precisely that, or it retries the same
        // write having been told something that reads as untrue.
        let denied = workspace
            .refusal("edit", &json!({"file_path": library.join("lib.rs").display().to_string()}))
            .expect("should refuse");
        assert!(denied.contains("read-only"), "got: {}", denied);

        // Tools with no path are not this module's business.
        assert!(workspace.refusal("web_search", &json!({"query": "x"})).is_none());
        assert!(workspace.refusal("run_command", &json!({"command": "ls"})).is_none());
        // An absent optional path means the working directory, which is always fine.
        assert!(workspace.refusal("grep", &json!({"pattern": "fn"})).is_none());
    }

    #[test]
    fn a_relative_path_cannot_climb_out_of_the_workspace() {
        let (project, _, elsewhere) = scratch_dirs();
        let workspace = Workspace::new(&project, &[]);

        // The escape that matters: dots are resolved before the comparison, so this lands
        // in `elsewhere` and is refused, rather than passing because the string starts
        // with the project root.
        let climbing = project.join("..").join("elsewhere").join("secrets.env");
        assert!(!workspace.may_read(&climbing));
        assert!(workspace.refusal("read", &json!({"file_path": climbing.display().to_string()})).is_some());

        let _ = elsewhere;
    }
}
