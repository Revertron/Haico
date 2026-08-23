//! Small helpers with no home of their own.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub(crate) fn run_git_command(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
}


pub(crate) fn truncate_at_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// Resolve "." and ".." textually, without touching the filesystem.
//
// Done before canonicalizing so a ".." buried in a not-yet-existing part of the path cannot
// smuggle a write out of the project: "proj/nothere/../../../evil.txt" has no directory to
// canonicalize, so only lexical resolution catches it.
fn lexical_normalize(path: &std::path::Path) -> PathBuf {
    use std::path::Component;

    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // Never pop past the root or the drive prefix.
                if matches!(
                    out.components().next_back(),
                    Some(Component::Normal(_))
                ) {
                    out.pop();
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

// Resolve a path to an absolute, canonical form without requiring it to exist yet.
//
// Canonicalizing only when the whole path exists was a real bug on Windows: canonicalize
// returns a verbatim path ("\\?\E:\proj"), so a file whose parent directory had not been
// created yet came back as a plain "E:\proj\sites\index.html". starts_with then compared a
// plain path against a verbatim one and always said false, which made every write into a
// new subdirectory look like it was outside the project.
//
// Canonicalizing the deepest part that *does* exist and re-attaching the rest keeps both
// sides in the same form no matter how much of the path is still hypothetical.
pub(crate) fn absolute_path(path: &std::path::Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    // Canonicalize first, and resolve ".." lexically only over the part that could not be.
    //
    // Doing it the other way round is a traversal: with `link` a symlink out of the
    // workspace, "W/link/../secret" cancels textually to "W/secret" and passes the boundary
    // check, while the tool then opens the original string and the OS — which resolves the
    // symlink before the ".." — lands outside. The gate would approve one file and the tool
    // open another. Letting canonicalize see the path first means symlinks and ".." in the
    // existing prefix are resolved by the same rules the tool will use.
    let mut trailing: Vec<std::ffi::OsString> = Vec::new();
    let mut current = absolute.as_path();
    loop {
        if let Ok(canonical) = fs::canonicalize(current) {
            let mut resolved = canonical;
            for part in trailing.iter().rev() {
                resolved.push(part);
            }
            // Only the tail that does not exist yet can still hold "..", and it has no
            // symlinks in it by definition, so collapsing it textually is safe here.
            return lexical_normalize(&resolved);
        }
        // Walk up until something exists. A drive root or "/" always does, so this
        // terminates well before running out of components in practice.
        //
        // The last *component* rather than file_name(): file_name() is None for a "..",
        // which would abandon the walk and fall back to the purely lexical answer this
        // exists to avoid.
        match (current.parent(), current.components().next_back()) {
            (Some(parent), Some(component)) => {
                trailing.push(component.as_os_str().to_os_string());
                current = parent;
            }
            _ => return lexical_normalize(&absolute),
        }
    }
}

// A path as a person should see it.
//
// Canonicalizing on Windows yields the verbatim form (\\?\E:\proj), which is the right
// thing to compare against — that prefix is why a not-yet-created path once read as being
// outside its own project — and the wrong thing to put in front of a user. Stripped for
// display only; every comparison still runs on the full form.
pub(crate) fn display_path(path: &std::path::Path) -> String {
    let shown = path.display().to_string();
    shown
        .strip_prefix(r"\\?\")
        .map(str::to_string)
        .unwrap_or(shown)
}

// Decide what a tool call needs before it runs.
//
// The keys are what makes "allow all" mean something narrower than "this tool, forever":
// a shell command is remembered by its program and subcommand, and a write outside the
// working directory is remembered per file rather than blanket-approved.

pub(crate) fn file_key(path: &std::path::Path) -> PathBuf {
    // Same resolver as everything else, so a file tracked before it existed and the same
    // file looked up afterwards land on one key.
    absolute_path(path)
}

pub(crate) fn modified_time(path: &std::path::Path) -> Option<std::time::SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}


#[cfg(test)]
mod tests {
    #![allow(unused_imports)]

    // Superseded in production by Workspace::may_write, which does the same comparison
    // against several roots. Kept here because these cases are the regression coverage for
    // the Windows verbatim-path bug, and what they are really exercising is absolute_path.
    fn is_inside_cwd(path: &std::path::Path) -> bool {
        let Ok(cwd) = std::env::current_dir() else {
            return false;
        };
        absolute_path(path).starts_with(absolute_path(&cwd))
    }
    use crate::config::*;
    use crate::llm::*;
    use crate::permissions::*;
    use crate::render::*;
    use crate::session::*;
    use crate::testutil::*;
    use crate::tools::*;
    use crate::util::*;
    use serde_json::json;

    #[test]
    fn truncates_without_splitting_characters() {
        // Short enough to pass through untouched.
        assert_eq!(truncate_at_boundary("hello", 10), "hello");
        assert_eq!(truncate_at_boundary("hello", 5), "hello");
        // ASCII cuts exactly at the limit.
        assert_eq!(truncate_at_boundary("hello", 3), "hel");

        // Each of these is 2 bytes, so a limit of 3 falls inside the second one and has to
        // back up to 2. Slicing directly would panic here.
        assert_eq!(truncate_at_boundary("привет", 3), "п");
        assert_eq!(truncate_at_boundary("привет", 4), "пр");

        // A 4-byte character backs all the way off when the limit lands inside it.
        assert_eq!(truncate_at_boundary("🦀", 3), "");
        assert_eq!(truncate_at_boundary("🦀", 4), "🦀");
        assert_eq!(truncate_at_boundary("a🦀", 2), "a");
    }

    #[test]
    fn paths_resolve_consistently_before_the_directories_exist() {
        use std::path::Path;

        let cwd = std::env::current_dir().unwrap();
        let project = absolute_path(&cwd);

        // The reported bug: writing into a subdirectory that has not been created yet was
        // judged to be outside the project. On Windows canonicalize returns a verbatim
        // "\\?\E:\..." path, so a not-yet-existing target came back as plain "E:\..." and
        // starts_with could never match.
        let scaffolded = cwd.join("sites").join("site1").join("index.html");
        assert!(!scaffolded.exists(), "test assumes this path is absent");
        assert!(
            absolute_path(&scaffolded).starts_with(&project),
            "{:?} should be inside {:?}",
            absolute_path(&scaffolded),
            project
        );
        assert!(is_inside_cwd(&scaffolded));

        // Relative paths reach the same answer.
        assert!(is_inside_cwd(Path::new("sites/site1/index.html")));
        assert!(is_inside_cwd(Path::new("./Cargo.toml")));

        // Existing files were already fine; make sure they stayed fine.
        assert!(is_inside_cwd(&cwd.join("Cargo.toml")));

        // Genuinely outside is still outside.
        assert!(!is_inside_cwd(&std::env::temp_dir().join("haico-elsewhere.txt")));
        assert!(!is_inside_cwd(&cwd.join("..").join("sibling.txt")));

        // ".." hidden behind a directory that does not exist must not escape the project.
        // Nothing there can be canonicalized, so only lexical resolution catches it.
        let sneaky = cwd.join("nothere").join("..").join("..").join("evil.txt");
        assert!(
            !is_inside_cwd(&sneaky),
            "{:?} escaped the project",
            absolute_path(&sneaky)
        );

        // ...while a ".." that stays within the project is still within it.
        let round_trip = cwd.join("src").join("..").join("Cargo.toml");
        assert!(is_inside_cwd(&round_trip));
        assert_eq!(absolute_path(&round_trip), absolute_path(&cwd.join("Cargo.toml")));
    }

    #[test]
    fn file_key_is_stable_across_a_file_coming_into_existence() {
        // seen_files is keyed by this: a file written and then edited must hash to one
        // entry, or the read-before-edit gate would reject the model's own new file.
        let dir = std::env::temp_dir().join(format!("haico-key-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("later.rs");
        let _ = std::fs::remove_file(&target);

        let before = file_key(&target);
        std::fs::write(&target, "fn main() {}").unwrap();
        let after = file_key(&target);
        assert_eq!(before, after);

        // Different spellings of the same file agree too.
        assert_eq!(after, file_key(&dir.join(".").join("later.rs")));

        let _ = std::fs::remove_file(&target);
        let _ = std::fs::remove_dir(&dir);
    }
}
