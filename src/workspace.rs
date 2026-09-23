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
    // This session's scratch directory, which is also in `writable` — it is remembered
    // separately because it is the one writable root that is not the user's work, and
    // both the system prompt and the auto-mode reviewer have to say so.
    scratch: Option<PathBuf>,
    // Single files pulled in by an `@mention` that fell outside everything above. Readable
    // and writable, like `writable`, and kept in their own list for two reasons: a file
    // path in a list every other entry of which is a directory reads as a mistake wherever
    // that list is shown, and `/add` and a mention are asking for different things — one
    // widens the boundary to a tree the user named, the other to the one file they did.
    attached: Vec<PathBuf>,
}

impl Workspace {
    pub(crate) fn new(cwd: &Path, readable: &[String]) -> Self {
        Self {
            writable: vec![absolute_path(cwd)],
            readable: readable
                .iter()
                .map(|p| absolute_path(&expand_home(p)))
                .collect(),
            scratch: None,
            attached: Vec::new(),
        }
    }

    // Point the workspace at this session's scratch directory, replacing whatever was
    // there before.
    //
    // Replacing rather than adding: `/clear` deletes the old directory and makes a new
    // one, and a writable root that no longer exists would stay in the list, be shown by
    // `/workspace`, and let a write recreate a directory nothing is going to clean up.
    pub(crate) fn set_scratch(&mut self, path: Option<&Path>) {
        if let Some(old) = self.scratch.take() {
            self.writable.retain(|root| root != &old);
        }
        if let Some(path) = path {
            let resolved = absolute_path(path);
            self.writable.push(resolved.clone());
            self.scratch = Some(resolved);
        }
    }

    // Move the working root, for `/cd`.
    //
    // Replaces the first entry rather than pushing a second one: the working directory is
    // always `writable[0]`, and a `/cd` that only added would quietly widen the boundary to
    // two projects where the user asked to move between them. Everything else stays — a
    // directory `/add`ed this session and a file pulled in by a mention were both named by
    // hand, and moving house is not a reason to decide the user stopped meaning them.
    //
    // A later entry naming the directory being moved into goes, because it is this one now:
    // the user may well have `/add`ed the project before deciding to work in it, and
    // `/workspace` would otherwise list it twice.
    pub(crate) fn set_cwd(&mut self, cwd: &Path) {
        let resolved = absolute_path(cwd);
        self.writable.remove(0);
        self.writable.retain(|root| root != &resolved);
        self.writable.insert(0, resolved);
    }

    pub(crate) fn scratch(&self) -> Option<&Path> {
        self.scratch.as_deref()
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

    // Make one file readable and writable for the rest of the session.
    //
    // Called for a file the user named with `@` that no root covers. Typing the path is
    // the consent — the same consent `/add` asks for, given about one file instead of a
    // directory — so there is no prompt. It stays one file: a mention must not open the
    // directory around it, or `@~/.ssh/config` would hand over every key beside it.
    //
    // Says whether it actually widened anything, because that is the part worth telling
    // the user about; a mention inside the project is not news.
    pub(crate) fn attach(&mut self, path: &Path) -> bool {
        let resolved = absolute_path(path);
        if self.may_read(&resolved) {
            return false;
        }
        self.attached.push(resolved);
        true
    }

    pub(crate) fn attached_files(&self) -> &[PathBuf] {
        &self.attached
    }

    // Make a directory readable for the rest of the session, as if it had been in
    // `read_paths` from the start.
    //
    // For directories haico itself knows the model has business reading — the user's own
    // skills — that may not have existed at startup. Says whether anything was added, so a
    // caller can do this unconditionally and repeatedly.
    pub(crate) fn add_readable(&mut self, path: &Path) -> bool {
        let resolved = absolute_path(path);
        if self.readable.iter().any(|root| root == &resolved) {
            return false;
        }
        self.readable.push(resolved);
        true
    }

    pub(crate) fn may_read(&self, path: &Path) -> bool {
        let path = absolute_path(path);
        self.attached.iter().any(|file| file == &path)
            || self.writable.iter().chain(self.readable.iter()).any(|root| path.starts_with(root))
    }

    pub(crate) fn may_write(&self, path: &Path) -> bool {
        let path = absolute_path(path);
        self.attached.iter().any(|file| file == &path)
            || self.writable.iter().any(|root| path.starts_with(root))
    }

    // The root a path falls under, so an approval can be keyed to it. Approving writes in
    // one project must not silently extend to a directory added later.
    //
    // An attached file is its own root, and is checked first: it is the narrower answer
    // wherever both could match, and it is the only answer when the file sits outside
    // every directory in the workspace — which is the case it exists for.
    pub(crate) fn writable_root_for(&self, path: &Path) -> Option<&Path> {
        let path = absolute_path(path);
        self.attached
            .iter()
            .find(|file| *file == &path)
            .or_else(|| self.writable.iter().find(|root| path.starts_with(root)))
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

        // Named individually, because they are the one part of the boundary the model
        // cannot infer from a directory: the file is fair game and everything beside it in
        // the same directory is not, and a model that assumed otherwise would spend a
        // round finding out through a refusal.
        if !self.attached.is_empty() {
            note.push_str(&format!(
                "\n\nThese single files may also be read and written, because the user \
                 named them in a message. Only these files — not the rest of the \
                 directories they are in:\n{}",
                list(&self.attached)
            ));
        }

        note.push_str(
            "\n\nAny other path is refused outright, for reads as well as writes, and there \
             is no prompt to approve it. Do not try to reach a refused path another way — \
             say what you need and let the user decide whether to add it.",
        );
        note
    }

    // What the auto-mode reviewer is told about the boundary.
    //
    // Separate wording from `note()` because the reviewer is told to deny anything that
    // touches files "outside the project directory", and it is shown one call with no
    // conversation behind it. Without this it reads a write into a directory under the
    // system temp folder as exactly that — while the temp directory in question is the
    // one haico created for the session and told the agent to use.
    pub(crate) fn review_note(&self) -> String {
        let list = |roots: &[PathBuf]| {
            roots
                .iter()
                .map(|r| {
                    let scratch = self.scratch.as_deref() == Some(r.as_path());
                    if scratch {
                        format!(
                            "- {} — this session's scratch directory, created by the tool \
                             itself under the system temp folder. Writing, overwriting and \
                             deleting throwaway files here is ordinary work.",
                            display_path(r)
                        )
                    } else {
                        format!("- {}", display_path(r))
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let mut note = format!(
            "\n\nThe agent is working under these directories. For the rules above they all \
             count as its project:\n{}",
            list(&self.writable)
        );

        if !self.readable.is_empty() {
            note.push_str(&format!(
                "\n\nThese it may read but never write:\n{}",
                list(&self.readable)
            ));
        }

        // The reviewer sees one call with no conversation behind it, so a write to a file
        // in someone's home directory reads as exactly the thing it is told to deny —
        // while the user typed that path themselves one message ago.
        if !self.attached.is_empty() {
            note.push_str(&format!(
                "\n\nThe user named these single files in a message, which is what put them \
                 in reach. Reading and changing them is what was asked for, wherever they \
                 happen to live:\n{}",
                list(&self.attached)
            ));
        }

        note.push_str(
            "\n\nThe file tools already refuse every other path before it reaches you. A \
             shell command is not bounded that way — it can reach anything the user can, \
             which is the case you are here to judge.",
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
pub(crate) fn expand_home(path: &str) -> PathBuf {
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
    use crate::testutil::TempDir;
    use serde_json::json;

    // Real directories, not invented paths: every comparison here goes through
    // `absolute_path`, which canonicalizes, and canonicalizing only resolves a path that
    // exists. The guard comes back with them and has to be *bound* — `let (_, project,
    // ..)` drops the directory before the first assertion runs.
    fn scratch_dirs() -> (TempDir, PathBuf, PathBuf, PathBuf) {
        let base = TempDir::new("ws");
        let project = base.subdir("project");
        let client = base.subdir("client");
        let elsewhere = base.subdir("elsewhere");
        (base, project, client, elsewhere)
    }

    #[test]
    fn the_working_directory_is_writable_and_everywhere_else_is_not() {
        let (_base, project, _, elsewhere) = scratch_dirs();
        let workspace = Workspace::new(&project, &[]);

        assert!(workspace.may_write(&project.join("src/main.rs")));
        assert!(workspace.may_read(&project.join("Cargo.toml")));

        // The default posture: nothing outside the project, not even a read. This is the
        // whole point — a secret two directories over is not reachable by asking nicely.
        assert!(!workspace.may_read(&elsewhere.join("secrets.env")));
        assert!(!workspace.may_write(&elsewhere.join("secrets.env")));
    }

    /// `/cd` moves the boundary rather than widening it: the project being left stops
    /// being writable, while a directory the user named by hand this session survives
    /// the move. The scratch directory does too — it belongs to the conversation, which
    /// `/cd` does not end.
    #[test]
    fn changing_the_working_directory_moves_the_root_and_keeps_what_was_added() {
        let (base, project, client, elsewhere) = scratch_dirs();
        let scratch = base.subdir("scratch");
        let mut workspace = Workspace::new(&project, &[]);
        workspace.set_scratch(Some(&scratch));
        workspace.add(&client).unwrap();

        workspace.set_cwd(&elsewhere);

        assert!(workspace.may_write(&elsewhere.join("src/main.rs")));
        assert!(!workspace.may_read(&project.join("src/main.rs")));
        assert!(workspace.may_write(&client.join("app.ts")));
        assert!(workspace.may_write(&scratch.join("notes.md")));
        // Still the first entry, which is what everything else here assumes it is.
        assert_eq!(workspace.writable_roots()[0], absolute_path(&elsewhere));
    }

    /// Moving into a directory that was `/add`ed earlier leaves one entry, not two —
    /// `/workspace` would otherwise list the same path twice.
    #[test]
    fn moving_into_an_added_directory_does_not_list_it_twice() {
        let (_base, project, client, _) = scratch_dirs();
        let mut workspace = Workspace::new(&project, &[]);
        workspace.add(&client).unwrap();

        workspace.set_cwd(&client);

        assert_eq!(workspace.writable_roots(), [absolute_path(&client)]);
    }

    #[test]
    fn adding_a_directory_makes_it_a_first_class_part_of_the_workspace() {
        let (_base, project, client, elsewhere) = scratch_dirs();
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

    // The scratch directory is a writable root like any other, or the model is told to
    // put its throwaway files somewhere the file tools then refuse to write.
    #[test]
    fn the_scratch_directory_is_writable_and_is_replaced_rather_than_piled_up() {
        let (_base, project, first, second) = scratch_dirs();
        let mut workspace = Workspace::new(&project, &[]);

        assert!(!workspace.may_write(&first.join("draft.py")));
        assert!(workspace.scratch().is_none());

        workspace.set_scratch(Some(&first));
        assert!(workspace.may_write(&first.join("draft.py")));
        assert_eq!(workspace.writable_roots().len(), 2);
        assert_eq!(workspace.scratch(), workspace.writable_roots().last().map(|p| p.as_path()));

        // /clear deletes the old directory and makes another. The root has to move with
        // it — left behind, a write would recreate a directory nothing will clean up.
        workspace.set_scratch(Some(&second));
        assert!(workspace.may_write(&second.join("draft.py")));
        assert!(!workspace.may_write(&first.join("draft.py")));
        assert_eq!(workspace.writable_roots().len(), 2);

        // And a session that could not make one is not left with a phantom root.
        workspace.set_scratch(None);
        assert!(!workspace.may_read(&second.join("draft.py")));
        assert_eq!(workspace.writable_roots().len(), 1);
        assert!(workspace.may_write(&project.join("src/main.rs")));
    }

    // The reviewer denies writes "outside the project directory" and is shown one call
    // with no history, so the scratch directory has to arrive labelled — otherwise the
    // one place haico told the agent to write is the one it gets refused for.
    #[test]
    fn the_reviewer_is_told_which_directory_is_scratch() {
        let (_base, project, scratch, library) = scratch_dirs();
        let mut workspace = Workspace::new(&project, &[library.display().to_string()]);
        workspace.set_scratch(Some(&scratch));

        let note = workspace.review_note();
        assert!(note.contains(&display_path(&project)));
        assert!(note.contains(&display_path(&scratch)));
        assert!(note.contains("scratch directory"), "got: {}", note);
        assert!(note.contains("read but never write"), "got: {}", note);
        assert!(note.contains(&display_path(&library)));

        // The project's own root must not be described as scratch.
        let line = note
            .lines()
            .find(|l| l.contains(&display_path(&project)))
            .unwrap();
        assert!(!line.contains("scratch"), "got: {}", line);
    }

    #[test]
    fn configured_read_paths_can_be_read_but_never_written() {
        let (_base, project, library, _) = scratch_dirs();
        let workspace = Workspace::new(&project, &[library.display().to_string()]);

        assert!(workspace.may_read(&library.join("src/lib.rs")));
        // Readable is not a step on the way to writable: the reason to list a path is that
        // the model has business reading it and none changing it.
        assert!(!workspace.may_write(&library.join("src/lib.rs")));
        assert_eq!(workspace.readable_roots().len(), 1);
    }

    #[test]
    fn a_readable_root_added_later_is_readable_once_and_never_writable() {
        let (_base, project, library, _) = scratch_dirs();
        let mut workspace = Workspace::new(&project, &[]);
        assert!(!workspace.may_read(&library.join("src/lib.rs")));

        assert!(workspace.add_readable(&library));
        assert!(!workspace.add_readable(&library), "added twice");
        assert!(workspace.may_read(&library.join("src/lib.rs")));
        assert!(!workspace.may_write(&library.join("src/lib.rs")));
        assert_eq!(workspace.readable_roots().len(), 1);
    }

    #[test]
    fn refusals_name_the_boundary_that_was_hit() {
        let (_base, project, library, elsewhere) = scratch_dirs();
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

    // Creating a symlink needs privilege or developer mode on Windows, so the test below
    // reports rather than fails when it cannot make one.
    #[cfg(windows)]
    fn link_dir(target: &Path, link: &Path) -> bool {
        std::os::windows::fs::symlink_dir(target, link).is_ok()
    }
    #[cfg(not(windows))]
    fn link_dir(target: &Path, link: &Path) -> bool {
        std::os::unix::fs::symlink(target, link).is_ok()
    }

    #[test]
    fn a_symlink_out_of_the_workspace_cannot_be_used_to_climb_back_in() {
        let base = std::env::temp_dir().join(format!("haico-link-{}", std::process::id()));
        let project = base.join("project");
        let outside = base.join("outside");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.env"), "AWS_SECRET=hunter2\n").unwrap();

        let link = project.join("link");
        let _ = std::fs::remove_dir_all(&link);
        if !link_dir(&outside, &link) {
            eprintln!("skipping: this machine will not let the test create a symlink");
            let _ = std::fs::remove_dir_all(&base);
            return;
        }

        let workspace = Workspace::new(&project, &[]);

        // Straight through the link, which is the vector that works everywhere: the file
        // really is outside, and canonicalization resolves the link and refuses it.
        assert!(!workspace.may_read(&link.join("secret.env")));
        assert!(workspace
            .refusal("read", &json!({"file_path": link.join("secret.env").display().to_string()}))
            .is_some());

        // ".." cancelling the symlink is where the two platforms genuinely disagree, so
        // the invariant worth asserting is not a fixed verdict but that the gate matches
        // whichever file the OS would actually open.
        let cancelled = project.join("link").join("..").join("secret.env");

        // POSIX resolves the symlink before the "..", so this lands outside `project`
        // altogether. Collapsing ".." textually first — which is what the resolver used to
        // do — made it read as "project/secret.env" and pass, while the tool opening the
        // original string followed the link and left the workspace.
        #[cfg(not(windows))]
        {
            assert!(
                !workspace.may_read(&cancelled),
                "a symlink cancelled by '..' must not pass as an in-workspace path"
            );
            assert!(!workspace.may_write(&cancelled));
        }

        // Win32 normalizes ".." textually before the filesystem ever sees the path, so
        // here it genuinely *is* project/secret.env — verified by opening both forms and
        // getting the inside file. Allowing it is correct rather than a hole.
        #[cfg(windows)]
        {
            std::fs::write(project.join("secret.env"), "inside\n").unwrap();
            assert!(workspace.may_read(&cancelled));
            assert_eq!(
                std::fs::read_to_string(&cancelled).unwrap(),
                std::fs::read_to_string(project.join("secret.env")).unwrap(),
                "the gate allowed this path, so it must be the in-project file"
            );
        }

        // Ordinary work inside the project is untouched by any of this.
        assert!(workspace.may_write(&project.join("src").join("main.rs")));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_relative_path_cannot_climb_out_of_the_workspace() {
        let (_base, project, _, elsewhere) = scratch_dirs();
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
