//! Project memory: the facts about this project that outlive a conversation.
//!
//! A session ends and everything learned in it goes with it — which build flag is a
//! trap, which test must not be run a certain way, which directory is generated. The
//! user types it again next time, or does not, and the model rediscovers it the
//! expensive way. This is where those facts survive.
//!
//! Only the project's. There is no user-wide store on purpose: a fact worth keeping is
//! almost always a fact about *this* code, and one that really does hold everywhere
//! already has a home in `~/.haico/HAICO.md`, written by the user rather than by the
//! model.
//!
//! **One fact per file, and the directory is the truth.** `.haico/memory/<name>.md`
//! holds a fact; `MEMORY.md` beside them is regenerated after every change and is never
//! read back. That is deliberate. An index that is a cache of the directory can
//! disagree with it, and the disagreement would be silent in the worst way: a fact
//! deleted by hand would go on being advertised in the system prompt, and the `recall`
//! it invited would come back empty. Nothing here ever trusts the index, so there is
//! nothing to keep in sync — the file exists for the user to read, not for haico.
//!
//! **The index lines go into the system prompt; the bodies do not.** That is what keeps
//! a hundred facts affordable: the prompt pays for a line each, and only the fact that
//! is actually about the work at hand is paid for in full.
//!
//! That shape is what `description` is for, and why it is a whole statement rather than
//! a topic. "The GUI is behind a feature flag, so `cargo build` alone never compiles it"
//! answers itself from the index and saves the `recall` entirely; "build flags" forces a
//! round trip to learn anything at all — and a round trip on a local model is seconds,
//! not milliseconds. The summary is the fast path, not a table of contents.

use std::fs;
use std::path::{Path, PathBuf};

/// Where a project's memories live, relative to its root.
pub(crate) const MEMORY_DIR: &str = ".haico/memory";

// Regenerated from the directory after every change; never parsed. Named in capitals
// because it is the one file in there a person is meant to open.
const INDEX_FILE: &str = "MEMORY.md";

/// Caps. A memory is a fact, not a document, and the index is a prompt section that is
/// sent on every single request — both of these are small on purpose.
pub(crate) const MAX_MEMORIES: usize = 100;
pub(crate) const MAX_FACT_CHARS: usize = 4000;
pub(crate) const MAX_DESCRIPTION_CHARS: usize = 200;
const MAX_NAME_CHARS: usize = 64;

pub(crate) struct Memory {
    /// The file's stem, and the only identity a memory has. See `parse`.
    pub(crate) name: String,
    /// One line. This is the part that reaches the model without being asked for.
    pub(crate) description: String,
    pub(crate) body: String,
}

pub(crate) fn dir(root: &Path) -> PathBuf {
    root.join(".haico").join("memory")
}

fn path_for(root: &Path, name: &str) -> PathBuf {
    dir(root).join(format!("{}.md", name))
}

/// Turn whatever the model called a memory into a filename.
///
/// Unicode-aware rather than ASCII-only: a project discussed in Russian produces facts
/// named in Russian, and slugging those to nothing would leave every one of them
/// fighting over the same empty name.
///
/// This is also the whole of the path safety here. Everything that is not alphanumeric
/// becomes a dash, so `..`, `/` and `\` cannot survive it — a name can only ever land
/// inside the memory directory, whatever the model asks for.
pub(crate) fn slug(raw: &str) -> Option<String> {
    let mut out = String::new();
    let mut gap = false;
    for c in raw.trim().chars() {
        if c.is_alphanumeric() {
            if gap && !out.is_empty() {
                out.push('-');
            }
            gap = false;
            out.extend(c.to_lowercase());
        } else {
            gap = true;
        }
    }
    let capped: String = out.chars().take(MAX_NAME_CHARS).collect();
    let trimmed = capped.trim_matches('-');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Flatten to something that can be one line of an index.
pub(crate) fn one_line(text: &str) -> String {
    let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flattened.chars().count() <= MAX_DESCRIPTION_CHARS {
        return flattened;
    }
    let cut: String = flattened.chars().take(MAX_DESCRIPTION_CHARS - 1).collect();
    // Back off to the last space so the line ends on a word rather than part way
    // through one, which is what the model reads back to itself next session.
    match cut.rfind(' ') {
        Some(space) if space > cut.len() / 2 => format!("{}…", &cut[..space]),
        _ => format!("{}…", cut),
    }
}

/// Read one file into a `Memory`.
///
/// The identity is the file's stem, never the `name:` in the frontmatter. They are
/// written to agree, but if they ever disagree the stem is the one `recall` can find
/// and the one `delete` can remove — believing the frontmatter would produce a memory
/// that advertises a name nothing answers to.
///
/// Frontmatter is optional. A file dropped in by hand with no header at all is still a
/// fact worth keeping; its first non-empty line becomes the description, which is what
/// a person writing one would have meant anyway.
fn parse(stem: &str, text: &str) -> Memory {
    let text = text.trim_start_matches('\u{feff}');
    let mut description = String::new();
    let mut body = text;

    if let Some(rest) = text.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            let (front, after) = rest.split_at(end);
            for line in front.lines() {
                if let Some((key, value)) = line.split_once(':') {
                    if key.trim() == "description" {
                        description = one_line(value);
                    }
                }
            }
            body = after.trim_start_matches('\n').trim_start_matches("---");
        }
    }

    let body = body.trim().to_string();
    if description.is_empty() {
        description = one_line(body.lines().find(|l| !l.trim().is_empty()).unwrap_or(""));
    }

    Memory { name: stem.to_string(), description, body }
}

/// Every memory this project has, by name.
///
/// Errors are swallowed throughout: a memory directory that cannot be read is a project
/// with no memories, which is a state haico has to work in anyway. Failing a startup
/// over it would trade a small loss for a total one.
pub(crate) fn load_all(root: &Path) -> Vec<Memory> {
    let Ok(entries) = fs::read_dir(dir(root)) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        // The generated index is not one of the facts it indexes.
        if path.file_name().and_then(|n| n.to_str()) == Some(INDEX_FILE) {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        out.push(parse(stem, &text));
    }

    // Sorted so the prompt section is stable between sessions. An index whose lines
    // moved around for no reason would look like a change to every diff that saw it,
    // and to every prefix cache.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

pub(crate) fn load_one(root: &Path, name: &str) -> Option<Memory> {
    let text = fs::read_to_string(path_for(root, name)).ok()?;
    Some(parse(name, &text))
}

/// The stored name the model meant, which is often not the one it typed.
///
/// Exact match first, then the slugged form, then a unique partial. A near miss that
/// resolves here costs nothing; the same near miss left to fail costs a round trip on a
/// local model, and the model's second guess is rarely better than its first.
pub(crate) fn resolve(root: &Path, wanted: &str) -> Option<String> {
    let all = load_all(root);
    let slugged = slug(wanted)?;

    if all.iter().any(|m| m.name == slugged) {
        return Some(slugged);
    }

    let hits: Vec<&Memory> = all
        .iter()
        .filter(|m| m.name.contains(&slugged) || slugged.contains(&m.name))
        .collect();
    // Only when it is unambiguous. Picking one of several would answer a question the
    // model did not ask, and it would look like a successful recall of the wrong fact.
    match hits.as_slice() {
        [only] => Some(only.name.clone()),
        _ => None,
    }
}

/// Write one memory, creating the store if this is the first.
pub(crate) fn save(root: &Path, memory: &Memory) -> std::io::Result<PathBuf> {
    let dir = dir(root);
    fs::create_dir_all(&dir)?;
    ignore_ourselves(root);

    let path = path_for(root, &memory.name);
    let text = format!(
        "---\nname: {}\ndescription: {}\n---\n\n{}\n",
        memory.name,
        one_line(&memory.description),
        memory.body.trim()
    );
    fs::write(&path, text)?;
    write_index(root);
    Ok(path)
}

/// Returns whether there was anything to remove.
pub(crate) fn remove(root: &Path, name: &str) -> std::io::Result<bool> {
    let path = path_for(root, name);
    if !path.exists() {
        return Ok(false);
    }
    fs::remove_file(&path)?;
    write_index(root);
    Ok(true)
}

/// Keep the store out of the project's history without touching the project's own
/// `.gitignore`.
///
/// A `.gitignore` containing `*` inside `.haico/` ignores the whole directory including
/// itself, which is the same result as adding a line to the repository's ignore file and
/// costs nothing outside the directory haico already owns. Editing a tracked file the
/// user did not ask about — to make a diff they did not ask for, in a repo that may have
/// its own conventions for that file — is not haico's business.
fn ignore_ourselves(root: &Path) {
    let path = root.join(".haico").join(".gitignore");
    if path.exists() {
        return;
    }
    let _ = fs::write(path, "*\n");
}

/// Regenerate `MEMORY.md` from the directory.
///
/// Best-effort, and deliberately not reported: this file is a convenience for whoever
/// opens the folder, and failing a `remember` that already wrote its fact — because the
/// summary of it could not be rewritten — would throw away the part that mattered to
/// protect the part that did not.
fn write_index(root: &Path) {
    let memories = load_all(root);
    let mut text = String::from(
        "# Project memory\n\n\
         Written by haico, regenerated whenever a memory changes — edit the files beside \
         this one, not this one. Each line names a memory and summarizes it; the file it \
         names holds the whole fact.\n\n",
    );
    for memory in &memories {
        text.push_str(&format!("- [{0}]({0}.md) — {1}\n", memory.name, memory.description));
    }
    if memories.is_empty() {
        text.push_str("_Nothing remembered yet._\n");
    }
    let _ = fs::write(dir(root).join(INDEX_FILE), text);
}

/// The index as the system prompt carries it, or `None` when there is nothing to carry.
pub(crate) fn index_block(root: &Path) -> Option<String> {
    let memories = load_all(root);
    if memories.is_empty() {
        return None;
    }
    let mut out = String::new();
    for memory in &memories {
        out.push_str(&format!("- {} — {}\n", memory.name, memory.description));
    }
    Some(out)
}

/// The index for the directory haico was started in.
///
/// The one entry point the rest of the program uses, so that "the project" means the
/// same directory everywhere. Everything above takes a root instead, which is what makes
/// it testable at all — the tests run in parallel in one process, where changing the
/// working directory is not something a test may do.
pub(crate) fn index_for_cwd() -> Option<String> {
    index_block(&std::env::current_dir().ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn memory(name: &str, description: &str, body: &str) -> Memory {
        Memory {
            name: name.to_string(),
            description: description.to_string(),
            body: body.to_string(),
        }
    }

    #[test]
    fn a_saved_memory_comes_back_whole() {
        let dir = TempDir::new("memory-roundtrip");
        save(dir.path(), &memory("gui-is-behind-a-feature", "cargo build does not compile gui.rs", "The gui feature is off by default.")).unwrap();

        let loaded = load_one(dir.path(), "gui-is-behind-a-feature").unwrap();
        assert_eq!(loaded.name, "gui-is-behind-a-feature");
        assert_eq!(loaded.description, "cargo build does not compile gui.rs");
        assert_eq!(loaded.body, "The gui feature is off by default.");
    }

    #[test]
    fn the_store_ignores_itself_so_the_project_s_gitignore_is_left_alone() {
        let dir = TempDir::new("memory-gitignore");
        save(dir.path(), &memory("a", "a", "a")).unwrap();

        let ignore = dir.path().join(".haico").join(".gitignore");
        assert_eq!(fs::read_to_string(ignore).unwrap(), "*\n");
        assert!(!dir.path().join(".gitignore").exists(), "the project's own file was touched");
    }

    #[test]
    fn a_name_cannot_escape_the_memory_directory() {
        // The slug is the whole of the path safety, so this is the test that matters
        // most in the file: every separator and every dot has to be gone.
        assert_eq!(slug("../../etc/passwd").unwrap(), "etc-passwd");
        assert_eq!(slug("..\\..\\windows").unwrap(), "windows");
        assert_eq!(slug("Build Flags!").unwrap(), "build-flags");
        assert_eq!(slug("...").unwrap_or_default(), "");
        assert!(slug("   ").is_none());
    }

    #[test]
    fn a_name_in_another_alphabet_survives_slugging() {
        // ASCII-only slugging would leave every Russian-named fact called "".
        assert_eq!(slug("сборка требует флага").unwrap(), "сборка-требует-флага");
    }

    #[test]
    fn the_index_is_rebuilt_from_the_directory_not_from_itself() {
        let dir = TempDir::new("memory-index");
        save(dir.path(), &memory("one", "the first fact", "body one")).unwrap();
        save(dir.path(), &memory("two", "the second fact", "body two")).unwrap();

        // A fact deleted behind haico's back. The index still names it until something
        // rewrites it, which is exactly why nothing ever reads the index back.
        fs::remove_file(dir.path().join(".haico/memory/one.md")).unwrap();

        let block = index_block(dir.path()).unwrap();
        assert!(!block.contains("one"), "a deleted fact was still advertised:\n{}", block);
        assert!(block.contains("two — the second fact"));

        // And the file catches up the next time anything is written.
        save(dir.path(), &memory("three", "the third fact", "body three")).unwrap();
        let written = fs::read_to_string(dir.path().join(".haico/memory/MEMORY.md")).unwrap();
        assert!(!written.contains("[one]"));
        assert!(written.contains("[two]") && written.contains("[three]"));
    }

    #[test]
    fn the_index_is_not_mistaken_for_a_fact() {
        let dir = TempDir::new("memory-index-not-a-fact");
        save(dir.path(), &memory("real", "a real fact", "body")).unwrap();

        let names: Vec<String> = load_all(dir.path()).into_iter().map(|m| m.name).collect();
        assert_eq!(names, vec!["real".to_string()]);
    }

    #[test]
    fn a_near_miss_resolves_and_an_ambiguous_one_does_not() {
        let dir = TempDir::new("memory-resolve");
        save(dir.path(), &memory("tests-need-parallelism", "d", "b")).unwrap();

        assert_eq!(resolve(dir.path(), "tests-need-parallelism").unwrap(), "tests-need-parallelism");
        assert_eq!(resolve(dir.path(), "Tests Need Parallelism").unwrap(), "tests-need-parallelism");
        assert_eq!(resolve(dir.path(), "parallelism").unwrap(), "tests-need-parallelism");

        save(dir.path(), &memory("tests-need-a-server", "d", "b")).unwrap();
        assert!(resolve(dir.path(), "tests").is_none(), "an ambiguous name picked one anyway");
    }

    #[test]
    fn a_hand_written_file_without_frontmatter_is_still_a_memory() {
        let dir = TempDir::new("memory-handwritten");
        fs::create_dir_all(super::dir(dir.path())).unwrap();
        fs::write(super::dir(dir.path()).join("by-hand.md"), "The build needs nightly.\n\nMore detail.\n").unwrap();

        let loaded = load_one(dir.path(), "by-hand").unwrap();
        assert_eq!(loaded.description, "The build needs nightly.");
        assert!(loaded.body.starts_with("The build needs nightly."));
    }

    #[test]
    fn the_file_stem_wins_over_disagreeing_frontmatter() {
        let dir = TempDir::new("memory-stem-wins");
        fs::create_dir_all(super::dir(dir.path())).unwrap();
        fs::write(
            super::dir(dir.path()).join("actual-name.md"),
            "---\nname: something-else\ndescription: d\n---\n\nbody\n",
        )
        .unwrap();

        // Believing the frontmatter would advertise a name that recall cannot open.
        assert_eq!(load_all(dir.path())[0].name, "actual-name");
    }

    #[test]
    fn a_description_is_flattened_and_capped() {
        let long = "word ".repeat(100);
        let line = one_line(&format!("first\nsecond\t{}", long));
        assert!(!line.contains('\n'));
        assert!(line.chars().count() <= MAX_DESCRIPTION_CHARS);
        assert!(line.ends_with('…'));
    }

    #[test]
    fn removing_says_whether_there_was_anything_to_remove() {
        let dir = TempDir::new("memory-remove");
        save(dir.path(), &memory("gone", "d", "b")).unwrap();

        assert!(remove(dir.path(), "gone").unwrap());
        assert!(!remove(dir.path(), "gone").unwrap());
        assert!(index_block(dir.path()).is_none());
    }
}
