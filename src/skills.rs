//! Skills: packaged instructions for one kind of task, loaded only when the task comes up.
//!
//! The format is the one other agents already use — a directory holding a `SKILL.md` whose
//! frontmatter carries `name` and `description`, and whatever scripts, templates or reference
//! files the instructions point at beside it. Inventing a format of our own would mean every
//! skill somebody already wrote for another agent has to be rewritten before haico can use it.
//!
//! **The index goes into the system prompt; the bodies do not.** Same bargain as project
//! memory: a line per skill is affordable on every request, a whole `SKILL.md` per skill is
//! not. The model sees the line, and calls `skill` for the rest when the task in hand is the
//! one the line describes. That is why the description matters more than the name — it is
//! the only part that decides whether the skill is ever used, so it has to say *when*.
//!
//! Where they come from, most specific first — the first one found under a name wins:
//!
//! - the project's `.haico/skills`, `.agents/skills` and `.claude/skills`, in the order
//!   `PROJECT_CONTEXT_FILES` reads the instruction files, and for the same reason: a project
//!   set up for another agent should work here without being copied;
//! - the user's own `skills` directory in haico's store, for skills that hold everywhere.
//!
//! The project's win over the user's for the reason the project's instruction file wins over
//! the user's: it is the more specific of the two.

use std::fs;
use std::path::{Path, PathBuf};

/// Where a project keeps its skills, relative to its root, in precedence order.
pub(crate) const PROJECT_SKILL_DIRS: &[&str] = &[".haico/skills", ".agents/skills", ".claude/skills"];

const SKILL_FILE: &str = "SKILL.md";

/// A description is the one part of a skill sent on every request, so it is capped. Longer
/// than a memory's, because a good one names both what the skill does and when to use it,
/// and cutting it before the "when" would throw away the half that matters.
const MAX_DESCRIPTION_CHARS: usize = 300;

/// A `SKILL.md` is meant to be a page or two; anything past this is cut and said to be.
/// The same cap the instruction files get — both are instructions arriving whole.
const MAX_BODY_BYTES: usize = 32_768;

pub(crate) struct Skill {
    /// The directory's name, and the only identity a skill has — the same rule memory
    /// follows for the same reason: it is the name that can actually be found again, so a
    /// frontmatter `name` that disagrees with it would advertise something nothing answers to.
    pub(crate) name: String,
    pub(crate) description: String,
    /// The skill's own directory. Handed to the model with the body, because the body
    /// refers to its scripts and references by paths relative to it.
    pub(crate) dir: PathBuf,
}

/// The user's own skills, in haico's store.
pub(crate) fn user_dir() -> Option<PathBuf> {
    crate::store::path("skills")
}

/// Every directory skills are looked for in, most specific first.
pub(crate) fn roots(project: &Path) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = PROJECT_SKILL_DIRS.iter().map(|d| project.join(d)).collect();
    roots.extend(user_dir());
    roots
}

/// The roots for the directory haico is in now. The one entry point the rest of the program
/// uses; everything else takes roots, which is what lets the tests run in parallel without
/// changing the working directory.
pub(crate) fn roots_for_cwd() -> Vec<PathBuf> {
    match std::env::current_dir() {
        Ok(cwd) => roots(&cwd),
        Err(_) => user_dir().into_iter().collect(),
    }
}

/// Every skill under `roots`, one per name, sorted by name.
///
/// Errors are swallowed the way memory swallows them: a directory that cannot be read is one
/// with no skills in it, and failing a startup over it would trade a small loss for a total one.
pub(crate) fn discover(roots: &[PathBuf]) -> Vec<Skill> {
    let mut out: Vec<Skill> = Vec::new();
    for root in roots {
        let Ok(entries) = fs::read_dir(root) else {
            continue;
        };
        let mut here = Vec::new();
        for entry in entries.flatten() {
            let dir = entry.path();
            let Some(name) = dir.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
                continue;
            };
            // Earlier roots are more specific, so a name already found shadows this one.
            if out.iter().any(|s| s.name == name) {
                continue;
            }
            let Ok(text) = fs::read_to_string(dir.join(SKILL_FILE)) else {
                continue;
            };
            let (front, body) = split_frontmatter(&text);
            let description = match front_value(front, "description") {
                Some(d) if !d.is_empty() => one_line(&d),
                // Same fallback as a hand-dropped memory: the first thing it says is what a
                // person writing it would have meant.
                _ => one_line(body.lines().find(|l| !l.trim().is_empty()).unwrap_or("")),
            };
            here.push(Skill { name, description, dir });
        }
        out.extend(here);
    }
    // Sorted so the prompt section is stable between sessions — a reordered index would be a
    // change to every prefix cache that saw it.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The skill the model meant, which is often not quite the name it typed.
///
/// Exact, then ignoring case, then a unique partial — the same forgiveness as
/// `memory::resolve`, and for its reason: a near miss that resolves costs nothing, and the
/// same miss left to fail costs a round trip on a local model.
pub(crate) fn resolve(roots: &[PathBuf], wanted: &str) -> Option<Skill> {
    let wanted = wanted.trim().trim_start_matches('/');
    if wanted.is_empty() {
        return None;
    }
    let mut all = discover(roots);
    if let Some(i) = all.iter().position(|s| s.name == wanted) {
        return Some(all.swap_remove(i));
    }
    let lower = wanted.to_lowercase();
    if let Some(i) = all.iter().position(|s| s.name.to_lowercase() == lower) {
        return Some(all.swap_remove(i));
    }
    let hits: Vec<usize> = all
        .iter()
        .enumerate()
        .filter(|(_, s)| s.name.to_lowercase().contains(&lower))
        .map(|(i, _)| i)
        .collect();
    match hits.as_slice() {
        [only] => Some(all.swap_remove(*only)),
        _ => None,
    }
}

/// A skill's instructions: the body of its `SKILL.md`, frontmatter removed.
pub(crate) fn body(skill: &Skill) -> Option<String> {
    let text = fs::read_to_string(skill.dir.join(SKILL_FILE)).ok()?;
    let (_, body) = split_frontmatter(&text);
    let body = body.trim();
    let cut = crate::util::truncate_at_boundary(body, MAX_BODY_BYTES);
    let mut out = cut.to_string();
    if cut.len() < body.len() {
        out.push_str("\n\n[skill instructions truncated]");
    }
    Some(out)
}

/// The index as the system prompt carries it, or `None` when there is nothing to carry.
pub(crate) fn index_block(roots: &[PathBuf]) -> Option<String> {
    let skills = discover(roots);
    if skills.is_empty() {
        return None;
    }
    let mut out = String::new();
    for skill in &skills {
        out.push_str(&format!("- {} — {}\n", skill.name, skill.description));
    }
    Some(out)
}

pub(crate) fn index_for_cwd() -> Option<String> {
    index_block(&roots_for_cwd())
}

/// What a typed `/name args` becomes: the skill's instructions as the user's message.
///
/// Written that way for `/init`'s reason — an ordinary turn gets the tools, the permission
/// prompts and `/undo` for free. `None` when the first word names no skill, so the caller
/// can send the line on as it was typed.
pub(crate) fn expand_command(roots: &[PathBuf], input: &str) -> Option<String> {
    let rest = input.strip_prefix('/')?;
    let (name, args) = match rest.split_once(char::is_whitespace) {
        Some((name, args)) => (name, args.trim()),
        None => (rest, ""),
    };
    // Exact only. Forgiving a model's near miss is cheap; running a skill the user did not
    // type because their typo was a unique substring of one is not.
    let skill = discover(roots).into_iter().find(|s| s.name == name)?;
    let body = body(&skill)?;
    Some(invocation(&skill, &body, if args.is_empty() { None } else { Some(args) }))
}

/// The skill as the model receives it, whether it asked for it or the user typed it.
pub(crate) fn invocation(skill: &Skill, body: &str, args: Option<&str>) -> String {
    let mut out = format!(
        "# Skill: {}\n\nDirectory: {}\nPaths in these instructions are relative to that \
         directory: read its files with `read` using the full path, and run its scripts with \
         `run_command`.\n",
        skill.name,
        crate::util::display_path(&skill.dir)
    );
    if let Some(args) = args {
        out.push_str(&format!("\nThe user invoked this skill with: {}\n", args));
    }
    out.push_str("\n---\n\n");
    out.push_str(body);
    out
}

/// Split `---`-fenced frontmatter from the body. No frontmatter is an empty one.
fn split_frontmatter(text: &str) -> (&str, &str) {
    let text = text.trim_start_matches('\u{feff}');
    if let Some(rest) = text.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            let (front, after) = rest.split_at(end);
            let after = &after[4..];
            // The rest of the closing fence's line, whatever it is, is not body.
            let after = after.split_once('\n').map(|(_, b)| b).unwrap_or("");
            return (front, after);
        }
    }
    ("", text)
}

/// One key from the frontmatter.
///
/// Not a YAML parser, and not trying to be: it covers what `SKILL.md` files actually contain —
/// a plain value, a quoted one, or a `>`/`|` block with its lines indented beneath. Pulling in
/// a YAML dependency for two keys would be the bigger risk of the two.
fn front_value(front: &str, key: &str) -> Option<String> {
    let mut lines = front.lines().peekable();
    while let Some(line) = lines.next() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        if k.trim() != key || line.starts_with(char::is_whitespace) {
            continue;
        }
        let v = v.trim();
        if v.starts_with('>') || v.starts_with('|') {
            let mut block = Vec::new();
            while let Some(next) = lines.peek() {
                if !next.trim().is_empty() && !next.starts_with(char::is_whitespace) {
                    break;
                }
                block.push(next.trim());
                lines.next();
            }
            return Some(block.join(" "));
        }
        let unquoted = v
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .or_else(|| v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
            .unwrap_or(v);
        return Some(unquoted.to_string());
    }
    None
}

fn one_line(text: &str) -> String {
    let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flattened.chars().count() <= MAX_DESCRIPTION_CHARS {
        return flattened;
    }
    let cut: String = flattened.chars().take(MAX_DESCRIPTION_CHARS - 1).collect();
    format!("{}…", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn write_skill(root: &Path, name: &str, text: &str) {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(SKILL_FILE), text).unwrap();
    }

    #[test]
    fn a_skill_is_found_by_its_directory_and_described_by_its_frontmatter() {
        let dir = TempDir::new("skills-basic");
        write_skill(
            dir.path(),
            "release",
            "---\nname: something-else\ndescription: \"Use when cutting a release.\"\n---\n\n# Steps\n\nTag it.\n",
        );

        let skills = discover(&[dir.path().to_path_buf()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "release");
        assert_eq!(skills[0].description, "Use when cutting a release.");
        assert_eq!(body(&skills[0]).unwrap(), "# Steps\n\nTag it.");
    }

    #[test]
    fn a_folded_description_is_read_whole() {
        let dir = TempDir::new("skills-folded");
        write_skill(
            dir.path(),
            "pdf",
            "---\nname: pdf\ndescription: >\n  Work with PDF files.\n  Use when a PDF is involved.\nlicense: MIT\n---\nbody\n",
        );
        let skills = discover(&[dir.path().to_path_buf()]);
        assert_eq!(skills[0].description, "Work with PDF files. Use when a PDF is involved.");
    }

    #[test]
    fn without_frontmatter_the_first_line_describes_it() {
        let dir = TempDir::new("skills-bare");
        write_skill(dir.path(), "bare", "\nFormat the changelog.\n\nMore.\n");
        let skills = discover(&[dir.path().to_path_buf()]);
        assert_eq!(skills[0].description, "Format the changelog.");
    }

    #[test]
    fn a_directory_without_skill_md_is_not_a_skill() {
        let dir = TempDir::new("skills-empty");
        fs::create_dir_all(dir.path().join("not-a-skill")).unwrap();
        assert!(discover(&[dir.path().to_path_buf()]).is_empty());
        assert!(index_block(&[dir.path().to_path_buf()]).is_none());
    }

    #[test]
    fn the_more_specific_root_wins_a_shared_name() {
        let project = TempDir::new("skills-project");
        let user = TempDir::new("skills-user");
        write_skill(project.path(), "deploy", "---\ndescription: project deploy\n---\nP");
        write_skill(user.path(), "deploy", "---\ndescription: user deploy\n---\nU");
        write_skill(user.path(), "lint", "---\ndescription: user lint\n---\nL");

        let roots = [project.path().to_path_buf(), user.path().to_path_buf()];
        let skills = discover(&roots);
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].description, "project deploy");
        assert_eq!(skills[1].name, "lint");
    }

    #[test]
    fn a_near_miss_resolves_only_when_it_is_unambiguous() {
        let dir = TempDir::new("skills-resolve");
        write_skill(dir.path(), "release-notes", "x");
        write_skill(dir.path(), "release-tag", "x");
        write_skill(dir.path(), "changelog", "x");
        let roots = [dir.path().to_path_buf()];

        assert_eq!(resolve(&roots, "Changelog").unwrap().name, "changelog");
        assert_eq!(resolve(&roots, "notes").unwrap().name, "release-notes");
        assert!(resolve(&roots, "release").is_none());
        assert!(resolve(&roots, "../etc").is_none());
    }

    #[test]
    fn a_typed_command_expands_into_the_instructions_with_its_arguments() {
        let dir = TempDir::new("skills-command");
        write_skill(dir.path(), "review", "---\ndescription: d\n---\nCheck the diff.");
        let roots = [dir.path().to_path_buf()];

        let expanded = expand_command(&roots, "/review src/main.rs please").unwrap();
        assert!(expanded.contains("Check the diff."), "{}", expanded);
        assert!(expanded.contains("src/main.rs please"), "{}", expanded);
        assert!(expanded.contains(&crate::util::display_path(&dir.path().join("review"))));

        // Exact names only, and nothing that is not a command.
        assert!(expand_command(&roots, "/rev").is_none());
        assert!(expand_command(&roots, "review").is_none());
    }
}
