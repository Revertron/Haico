//! Which project servers the user has approved, kept in the store.
//!
//! A project's `.haico/mcp.json` arrives with the repository, and an entry in it is a
//! command line that runs on this machine the moment haico starts. Cloning a repository is
//! not consent to that, so a project server runs only once the user has said yes to it —
//! to that exact entry. The approval is stored under the project's directory and the
//! entry's fingerprint, so editing the entry (a different command, a new `autoAllow`) asks
//! again, and the same entry in another checkout is another question.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const FILE: &str = "mcp-approvals.json";

/// project directory → server name → fingerprint.
type Approvals = BTreeMap<String, BTreeMap<String, String>>;

/// The store's approvals, read from and written to `path`. `None` (no store) approves
/// nothing and remembers nothing.
pub(crate) struct Trust {
    path: Option<PathBuf>,
}

impl Trust {
    pub(crate) fn from_store() -> Self {
        Self::from_store_file(FILE)
    }

    /// The same bookkeeping in another file of the store, for the project's hooks: a
    /// `.haico/hooks.json` is commands from the repository just as `mcp.json` is.
    pub(crate) fn from_store_file(file: &str) -> Self {
        // Never the user's own file from a test binary, for `Agent::new`'s reason.
        Self { path: if cfg!(test) { None } else { crate::store::path(file) } }
    }

    #[cfg(test)]
    pub(crate) fn at(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn read(&self) -> Approvals {
        self.path
            .as_deref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub(crate) fn is_approved(&self, project: &Path, server: &str, fingerprint: &str) -> bool {
        self.read()
            .get(&project_key(project))
            .and_then(|servers| servers.get(server))
            .is_some_and(|stored| stored == fingerprint)
    }

    pub(crate) fn approve(&self, project: &Path, server: &str, fingerprint: &str) -> anyhow::Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        let mut approvals = self.read();
        approvals.entry(project_key(project)).or_default().insert(server.to_string(), fingerprint.to_string());
        crate::util::write_atomically(path, serde_json::to_string_pretty(&approvals)?.as_bytes())
    }
}

// Case-folded on Windows, where `E:\proj` and `e:\proj` are one directory.
fn project_key(project: &Path) -> String {
    let text = crate::util::absolute_path(project).to_string_lossy().into_owned();
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_approval_holds_for_that_project_and_that_entry_only() {
        let dir = crate::testutil::TempDir::new("mcp-trust");
        let trust = Trust::at(dir.join("approvals.json"));
        let project = dir.join("proj");
        assert!(!trust.is_approved(&project, "db", "aaaa"));
        trust.approve(&project, "db", "aaaa").unwrap();
        assert!(trust.is_approved(&project, "db", "aaaa"));
        // The entry changed: ask again.
        assert!(!trust.is_approved(&project, "db", "bbbb"));
        // Another checkout of the same repository is another question.
        assert!(!trust.is_approved(&dir.join("other"), "db", "aaaa"));
        // Re-approving replaces the old fingerprint.
        trust.approve(&project, "db", "bbbb").unwrap();
        assert!(trust.is_approved(&project, "db", "bbbb"));
        assert!(!trust.is_approved(&project, "db", "aaaa"));
    }

    #[test]
    fn without_a_store_nothing_is_approved() {
        let trust = Trust { path: None };
        trust.approve(Path::new("/p"), "s", "f").unwrap();
        assert!(!trust.is_approved(Path::new("/p"), "s", "f"));
    }
}
