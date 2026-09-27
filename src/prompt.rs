//! The system prompt and the project's own instruction file.

use crate::util::{display_path, run_git_command, truncate_at_boundary};
use crate::workspace::Workspace;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const SYSTEM_PROMPT: &str = include_str!("system_prompt.txt");


pub(crate) fn get_main_branch() -> String {
    // Try to get the default branch from the remote
    if let Some(remote_head) = run_git_command(&["symbolic-ref", "refs/remotes/origin/HEAD"]) {
        // Output format: "refs/remotes/origin/main"
        if let Some(branch) = remote_head.strip_prefix("refs/remotes/origin/") {
            return branch.to_string();
        }
    }

    // Try to get from git config
    if let Some(default_branch) = run_git_command(&["config", "--get", "init.defaultBranch"]) {
        if !default_branch.is_empty() {
            return default_branch;
        }
    }

    // Fallback: check if common branches exist
    for branch in &["main", "master"] {
        if run_git_command(&["rev-parse", "--verify", branch]).is_some() {
            return branch.to_string();
        }
    }

    // Ultimate fallback
    "main".to_string()
}


pub(crate) fn build_system_prompt(model: &str, workspace: &Workspace) -> String {
    let pwd = env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_string()))
        .unwrap_or_else(|| ".".to_string());

    let is_repo = run_git_command(&["rev-parse", "--git-dir"]).is_some();

    let platform = if cfg!(target_os = "windows") {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    };

    let os_version = std::env::consts::OS;

    // Naming the shell is not a detail: cmd and sh disagree on the things a model reaches for
    // without thinking — `;`, `/dev/null`, heredocs — and "Platform: win32" above was evidently
    // not enough to stop a POSIX one-liner being written for it.
    let shell = if cfg!(target_os = "windows") {
        "cmd.exe — use cmd syntax: chain with && or &, never ';'; no /dev/null and no heredocs"
    } else {
        "sh"
    };

    let date = chrono::Local::now().format("%Y-%m-%d").to_string();

    let (branch, git_status, last_5_commits) = if is_repo {
        let branch = run_git_command(&["rev-parse", "--abbrev-ref", "HEAD"])
            .unwrap_or_else(|| "unknown".to_string());

        let status = run_git_command(&["status", "--short"])
            .unwrap_or_else(|| "".to_string());

        let commits = run_git_command(&["log", "--oneline", "-5"])
            .unwrap_or_else(|| "".to_string());

        (branch, status, commits)
    } else {
        ("N/A".to_string(), "".to_string(), "".to_string())
    };

    let main_branch = get_main_branch();

    // Named because it is otherwise unguessable. A model told to write a throwaway script
    // and given nowhere to put it picks "C:\Temp" or "/tmp" — a path the workspace then
    // refuses, on a machine whose real temp directory it was never shown.
    let scratch_dir = match workspace.scratch() {
        Some(path) => display_path(path),
        None => "none — this session has none, so keep temporary files in the working \
                 directory"
            .to_string(),
    };

    let read_paths = if workspace.readable_roots().is_empty() {
        "none".to_string()
    } else {
        workspace
            .readable_roots()
            .iter()
            .map(|r| display_path(r))
            .collect::<Vec<_>>()
            .join(", ")
    };

    SYSTEM_PROMPT
        .replace("$pwd", &pwd)
        .replace("$is_repo", &is_repo.to_string())
        .replace("$platform", platform)
        .replace("$os_version", os_version)
        .replace("$shell", shell)
        .replace("$date", &date)
        .replace("$scratch_dir", &scratch_dir)
        .replace("$read_paths", &read_paths)
        .replace("$branch", &branch)
        .replace("$main_branch", &main_branch)
        .replace("$git_status", &git_status)
        .replace("$model", if model.is_empty() { "unknown" } else { model })
        .replace("$last_5_commits", &last_5_commits)
}

// Assemble the system message: haico's own prompt — the built-in one, or whatever
// --system-prompt supplied — with the user's global instructions and then the project's
// appended.
//
// The order is the precedence order, least specific first: ours, the user's, the project's.
// Each section says it overrides what is above it, so the most specific file wins simply by
// being last.
//
// `use_system_prompt` comes from the model's own config section. A model fine-tuned with
// its own baked-in instructions can be made worse by a second set layered on top, so ours
// is dropped for it while the user's and the project's still go through — those are the
// user's own words, not haico's, and dropping them would silently ignore what they wrote.
//
// None when that leaves nothing to say at all: such a model, with neither file present,
// gets no system message rather than an empty one.

pub(crate) fn compose_system_prompt(
    custom: Option<&str>,
    model: &str,
    use_system_prompt: bool,
    user_context: Option<&str>,
    project_context: Option<&(String, String)>,
    skills: Option<&str>,
    library: Option<&str>,
    memory: Option<&str>,
    workspace: &Workspace,
) -> Option<String> {
    let mut content = if !use_system_prompt {
        String::new()
    } else if let Some(custom) = custom {
        custom.to_string()
    } else {
        build_system_prompt(model, workspace)
    };

    // Appended to whichever prompt is in use, so a custom --system-prompt gets these too,
    // and a model configured without ours still gets them.
    if let Some(user) = user_context {
        append_section(
            &mut content,
            &format!("# User instructions ({})", user_context_file()),
            "The following comes from this user and takes precedence over the general \
             guidance above wherever the two disagree.",
            user,
        );
    }

    if let Some((name, project)) = project_context {
        append_section(
            &mut content,
            &format!("# Project instructions ({})", name),
            "The following comes from this project and takes precedence over the general \
             guidance above wherever the two disagree.",
            project,
        );
    }

    // After the instruction files and claiming no precedence over them: a skill is how to
    // do one kind of task, not a rule about this project, and it is not even in force until
    // it is loaded. Only the index is here — the body costs a `skill` call, and only for
    // the task it is about.
    if let Some(skills) = skills {
        append_section(
            &mut content,
            "# Skills",
            "Instructions for particular kinds of task, one line each saying when it applies. \
             When the user's request matches one, call `skill` with its name before starting \
             and follow what it returns.",
            skills,
        );
    }

    // Next to skills and for the same reason claiming no precedence: it says what the
    // `library` tool knows about, not what to do. It is the half of the library's nudge
    // that works before anything has gone wrong — a model that sees "tokio" listed here
    // can ask before writing the code, rather than only after the build fails.
    if let Some(library) = library {
        append_section(
            &mut content,
            "# Library",
            "Articles on these crates and topics are in haico's library, one line each. When \
             you write code against one of them and are not certain of its API, or a build \
             fails with errors involving one or with one of the compiler errors listed, call \
             `library` with a specific question before guessing.",
            library,
        );
    }

    // Last, and phrased differently from the two above on purpose. The instruction
    // files tell the model what to do; this tells it what was true of the project when
    // somebody last looked. Handing it the same "takes precedence" sentence would make
    // a remembered fact outrank the code it is a fact about, which is exactly backwards
    // when the two disagree — the memory is the thing that went stale.
    if let Some(memory) = memory {
        append_section(
            &mut content,
            &format!("# Project memory ({}/)", crate::memory::MEMORY_DIR),
            "Facts recorded about this project in earlier sessions, one line each. These \
             are observations, not instructions: where one disagrees with what you can \
             see in the code now, the code is right and the memory is worth correcting \
             with `remember`. Call `recall` with a name to read the whole of that fact \
             before acting on its summary alone.",
            memory,
        );
    }

    if content.trim().is_empty() {
        None
    } else {
        Some(content)
    }
}

// The precedence sentence is dropped when this section is the whole message — with nothing
// above it there is nothing for it to take precedence over, and the model would be reading
// a reference to guidance that was never sent.
fn append_section(content: &mut String, heading: &str, precedence: &str, body: &str) {
    if content.is_empty() {
        *content = format!("{}\n\n{}", heading, body);
    } else {
        content.push_str(&format!("\n\n{}\n\n{}\n\n{}", heading, precedence, body));
    }
}

// Model response structures

pub(crate) const PROJECT_CONTEXT_FILES: &[&str] = &["HAICO.md", "AGENTS.md", "CLAUDE.md"];

// Load the project's own instructions, if it has any.
//
// Without this there is no way to tell the model a project's conventions except by retyping
// them every session or editing the global system prompt. Capped because an oversized file
// would silently eat the context window it is meant to make good use of.

pub(crate) fn load_project_context() -> Option<(String, String)> {
    const MAX_BYTES: usize = 32_768;

    for name in PROJECT_CONTEXT_FILES {
        let path = PathBuf::from(name);
        let Ok(contents) = fs::read_to_string(&path) else {
            continue;
        };
        if contents.trim().is_empty() {
            continue;
        }

        let trimmed = truncate_at_boundary(&contents, MAX_BYTES);
        let mut text = trimmed.to_string();
        if trimmed.len() < contents.len() {
            text.push_str("\n\n[project context truncated]");
        }
        return Some((name.to_string(), text));
    }

    None
}

// The user's own instructions, in force in every directory.
//
// The resolved path rather than a fixed string, which is what this was while the answer
// was `~/.haico/HAICO.md` on every platform. It is no longer the same place on each, and
// both readers of this — the person being told where their instructions came from, and the
// model being told which file to edit when asked to change them — need the real one.
pub(crate) fn user_context_file() -> String {
    crate::store::display("HAICO.md")
}

// Next to config.toml rather than in the home directory itself: preferences that hold
// everywhere ("always American English", "never commit unless asked") belong in one place
// instead of being copied into every project's instruction file.
pub(crate) fn user_context_path() -> Option<PathBuf> {
    crate::store::path("HAICO.md")
}

// Read once at startup and kept for the session. Unlike the project's file this one is not
// re-read on /clear or after /init: it lives outside the project, nothing haico does writes
// to it, and editing it mid-session is not a thing that happens.

pub(crate) fn load_user_context() -> Option<String> {
    const MAX_BYTES: usize = 32_768;

    let contents = fs::read_to_string(user_context_path()?).ok()?;
    if contents.trim().is_empty() {
        return None;
    }

    let trimmed = truncate_at_boundary(&contents, MAX_BYTES);
    let mut text = trimmed.to_string();
    if trimmed.len() < contents.len() {
        text.push_str("\n\n[user context truncated]");
    }
    Some(text)
}

pub(crate) const INIT_PROMPT: &str = include_str!("init_prompt.txt");

// The first instruction file the project actually has, with its contents.
fn existing_context_file(root: &Path) -> Option<(&'static str, String)> {
    for name in PROJECT_CONTEXT_FILES {
        let Ok(contents) = fs::read_to_string(root.join(name)) else {
            continue;
        };
        if contents.trim().is_empty() {
            continue;
        }
        return Some((name, contents));
    }
    None
}

// Which file `/init` writes.
//
// Not always HAICO.md: load_project_context reads the *first* of PROJECT_CONTEXT_FILES
// that exists, so writing HAICO.md into a project that already has its own CLAUDE.md would
// shadow it from then on, leaving two files that disagree and no sign of which one is in
// force. Whatever is already being loaded is what gets improved.

pub(crate) fn init_target_file(root: &Path) -> &'static str {
    existing_context_file(root)
        .map(|(name, _)| name)
        .unwrap_or("HAICO.md")
}

// The instruction `/init` sends, with everything haico can cheaply find out about the
// project already attached to it.
//
// The facts are gathered here rather than left to the model to discover because the models
// haico talks to are small local ones: asked to "analyze the project" from nothing, they
// tend to call `ls` once, guess the rest, and write a confident file full of build commands
// that do not exist. Handing them the listing, the manifests, the README and the recent
// commits costs nothing and changes what comes back.

pub(crate) fn build_init_prompt(root: &Path) -> String {
    // Enough of a README to tell what the project is and to show which commands it
    // documents; enough of an existing instruction file to improve it rather than guess.
    const README_LINES: usize = 40;
    const README_BYTES: usize = 4_096;
    const EXISTING_BYTES: usize = 20_480;

    let mut prompt = INIT_PROMPT.replace("$target_file", init_target_file(root));

    prompt.push_str("\n# What haico already found\n\n");
    prompt.push_str(&format!("Working directory: {}\n", root.display()));

    let root_arg = root.to_string_lossy().to_string();
    if let Some(branch) = run_git_command(&["-C", &root_arg, "rev-parse", "--abbrev-ref", "HEAD"]) {
        prompt.push_str(&format!("Git branch: {}\n", branch));
    }

    let (entries, hidden) = top_level_entries(root);
    if !entries.is_empty() {
        prompt.push_str("\nTop-level entries:\n");
        for entry in &entries {
            prompt.push_str(&format!("  {}\n", entry));
        }
        if hidden > 0 {
            prompt.push_str(&format!("  ... and {} more\n", hidden));
        }
    }

    let manifests = manifest_files(root);
    if !manifests.is_empty() {
        prompt.push_str("\nBuild and manifest files (this is where the commands are):\n");
        for manifest in &manifests {
            prompt.push_str(&format!("  {}\n", manifest));
        }
    }

    if let Some((name, text)) = readme(root) {
        let head = text.lines().take(README_LINES).collect::<Vec<_>>().join("\n");
        let head = truncate_at_boundary(&head, README_BYTES);
        prompt.push_str(&format!(
            "\n{} (first {} lines):\n\n{}\n",
            name, README_LINES, head
        ));
    }

    if let Some(log) = run_git_command(&["-C", &root_arg, "log", "--oneline", "-15"]) {
        if !log.is_empty() {
            prompt.push_str(&format!("\nRecent commits:\n\n{}\n", log));
        }
    }

    // Last, and with its own instruction, because "here is the file you are replacing" is
    // the one fact the model most often acts on wrongly — throwing away hand-written rules
    // it could not verify in one session.
    if let Some((name, contents)) = existing_context_file(root) {
        let trimmed = truncate_at_boundary(&contents, EXISTING_BYTES);
        prompt.push_str(&format!(
            "\n# {} already exists\n\nImprove it instead of starting over: keep what is still \
             true, correct what is wrong, add what is missing. Do not drop a section because \
             you have not checked it — check it.\n\n{}\n",
            name, trimmed
        ));
        if trimmed.len() < contents.len() {
            prompt.push_str(
                "\n[cut short here — `read` the file in full and use `edit` rather than \
                 rewriting it whole]\n",
            );
        }
    }

    prompt
}

// Sorted, directories marked, and the noise every project has left out. The count of what
// was dropped goes with it so the model knows the listing is partial.
fn top_level_entries(root: &Path) -> (Vec<String>, usize) {
    const MAX_ENTRIES: usize = 60;
    const SKIP: &[&str] = &[".git", "target", "node_modules", ".venv", "venv", "dist", "build"];

    let Ok(dir) = fs::read_dir(root) else {
        return (Vec::new(), 0);
    };

    let mut names: Vec<String> = dir
        .flatten()
        .filter(|entry| !SKIP.contains(&entry.file_name().to_string_lossy().as_ref()))
        .map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                format!("{}/", name)
            } else {
                name
            }
        })
        .collect();
    names.sort();

    let hidden = names.len().saturating_sub(MAX_ENTRIES);
    names.truncate(MAX_ENTRIES);
    (names, hidden)
}

// Manifests one level down as well as at the top, because in a workspace or a monorepo the
// top-level listing says nothing at all about how anything is built.
fn manifest_files(root: &Path) -> Vec<String> {
    const NAMES: &[&str] = &[
        "Cargo.toml", "package.json", "pyproject.toml", "setup.py", "requirements.txt",
        "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "CMakeLists.txt",
        "Makefile", "makefile", "justfile", "Justfile", "composer.json", "Gemfile",
        "deno.json", "tsconfig.json", "Dockerfile", "docker-compose.yml",
    ];
    const SKIP: &[&str] = &[".git", "target", "node_modules", ".venv", "venv", "dist", "build"];
    const MAX: usize = 25;

    let mut found = Vec::new();

    for name in NAMES {
        if root.join(name).is_file() {
            found.push(name.to_string());
        }
    }

    if let Ok(dir) = fs::read_dir(root) {
        let mut subdirs: Vec<PathBuf> = dir
            .flatten()
            .filter(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|entry| entry.path())
            .filter(|path| {
                !path
                    .file_name()
                    .map(|n| SKIP.contains(&n.to_string_lossy().as_ref()))
                    .unwrap_or(true)
            })
            .collect();
        subdirs.sort();

        for subdir in subdirs {
            for name in NAMES {
                if found.len() >= MAX {
                    return found;
                }
                if subdir.join(name).is_file() {
                    let parent = subdir.file_name().unwrap_or_default().to_string_lossy();
                    found.push(format!("{}/{}", parent, name));
                }
            }
        }
    }

    found
}

fn readme(root: &Path) -> Option<(&'static str, String)> {
    const NAMES: &[&str] = &["README.md", "readme.md", "README.rst", "README.txt", "README"];

    for name in NAMES {
        if let Ok(contents) = fs::read_to_string(root.join(name)) {
            if !contents.trim().is_empty() {
                return Some((name, contents));
            }
        }
    }
    None
}

// Settings that vary by model rather than by server. Context window is a genuine property
// of the model; sampling is usually chosen per model too.
//
// Every field is optional and nothing has a built-in default, because "unset" has to stay
// distinguishable from "set to the default". LM Studio lets you tune sampling per model in
// its own UI, and a client that always sends temperature would silently override that with
// no indication why the model behaves differently through haico.
// Key names match the wire field exactly, so what you write here is what the server sees.
//
// top_k, min_p and repeat_penalty are not part of the OpenAI API — they are extensions
// that llama.cpp, Ollama, LM Studio and vLLM all understand. A strictly-OpenAI endpoint will
// ignore or reject them, which is why nothing is sent unless it was asked for.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    // Its own directory rather than testutil::scratch, because build_init_prompt lists the
    // whole directory and one test's fixture would show up in another's listing.
    fn init_dir(name: &str) -> TempDir {
        TempDir::new(&format!("init-{}", name))
    }

    // The order is the whole point: the project's file has to come after the user's, or
    // "takes precedence over the guidance above" in each section says the wrong thing and
    // the more specific file is the one that loses.
    fn ws() -> Workspace {
        Workspace::new(&env::current_dir().unwrap(), &[])
    }

    #[test]
    fn user_instructions_sit_between_ours_and_the_project_s() {
        let project = ("HAICO.md".to_string(), "PROJECT RULES".to_string());

        let both = compose_system_prompt(
            Some("OUR PROMPT"),
            "some-model",
            true,
            Some("USER RULES"),
            Some(&project),
            Some("- a-skill - SKILLFUL"),
            Some("- tokio: LIBRARIAN"),
            Some("- a-remembered-fact - REMEMBERED"),
            &ws(),
        )
        .expect("everything is present");

        let ours = both.find("OUR PROMPT").unwrap();
        let user = both.find("USER RULES").unwrap();
        let proj = both.find("PROJECT RULES").unwrap();
        assert!(ours < user && user < proj, "wrong order:\n{}", both);
        // The path itself is the platform's now, so the assertion is that the heading
        // names the real file rather than that it names a particular string.
        assert!(
            both.contains(&format!("# User instructions ({})", user_context_file())),
            "the heading does not name the instructions file:\n{}",
            both
        );
        assert!(user_context_file().ends_with("HAICO.md"));

        // Memory comes after both instruction files, and is the one section that does
        // not claim to take precedence over what is above it: a fact about the project
        // must not outrank an instruction from the user or from the project itself.
        let memory = both.find("REMEMBERED").unwrap();
        assert!(proj < memory, "memory is not last: {}", both);

        // Skills sit between the instruction files and memory, and claim no precedence
        // either: a skill is how to do one kind of task, not a rule that outranks the user's.
        let skills = both.find("SKILLFUL").unwrap();
        assert!(proj < skills && skills < memory, "skills out of place: {}", both);
        // The library sits between skills and memory, claiming no precedence either.
        let library = both.find("LIBRARIAN").unwrap();
        assert!(skills < library && library < memory, "library out of place: {}", both);
        let section = &both[both.find("# Library").unwrap()..both.find("# Project memory").unwrap()];
        assert!(!section.contains("takes precedence"), "got: {}", section);
        // A string continued without its trailing backslash keeps the next line's indentation,
        // and the model reads sentences with runs of spaces in the middle of them.
        assert!(!both.contains("   "), "a run of spaces in the prompt:\n{}", both);
        let section = &both[both.find("# Skills").unwrap()..both.find("# Project memory").unwrap()];
        assert!(!section.contains("takes precedence"), "got: {}", section);
        let section = &both[both.find("# Project memory").unwrap()..];
        assert!(!section.contains("takes precedence"), "got: {}", section);

        // Alone it is still a system message, and without anything above it the sentence
        // about taking precedence is left out.
        let alone =
            compose_system_prompt(None, "some-model", false, Some("USER RULES"), None, None, None, None, &ws())
                .expect("the user's own file is enough on its own");
        assert!(alone.starts_with("# User instructions"), "got: {}", alone);
        assert!(!alone.contains("takes precedence"), "got: {}", alone);
    }

    // The two lines this whole thing exists for: a model that is not told the real temp
    // directory writes to "C:\Temp" and has the write refused, and one that is not told
    // what it may read discovers the read-only roots by being refused too.
    #[test]
    fn the_env_block_names_the_scratch_directory_and_the_read_only_roots() {
        let base = init_dir("env-block");
        let scratch = base.subdir("scratch");
        let library = base.subdir("library");

        let mut workspace = Workspace::new(base.path(), &[library.display().to_string()]);
        workspace.set_scratch(Some(&scratch));

        let prompt = build_system_prompt("some-model", &workspace);

        let shown = crate::util::display_path(workspace.scratch().unwrap());
        assert!(prompt.contains(&format!("Scratch directory: {}", shown)), "got: {}", prompt);
        assert!(prompt.contains("Read-only directories:"), "got: {}", prompt);
        assert!(prompt.contains(&crate::util::display_path(&library)), "got: {}", prompt);
        assert!(!prompt.contains("$scratch_dir") && !prompt.contains("$read_paths"));

        // Nothing configured is "none" rather than an empty line, which reads as a
        // substitution that failed.
        let bare = build_system_prompt("some-model", &Workspace::new(base.path(), &[]));
        assert!(bare.contains("Read-only directories: none"), "got: {}", bare);
        assert!(bare.contains("Scratch directory: none —"), "got: {}", bare);
    }

    #[test]
    fn init_prompt_carries_the_project_facts() {
        let dir = init_dir("facts");
        fs::write(dir.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
        fs::write(dir.join("README.md"), "# Demo\n\nRun with `cargo run`.\n").unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("target")).unwrap();

        let prompt = build_init_prompt(dir.path());

        assert!(prompt.contains("HAICO.md"), "no target file named:\n{}", prompt);
        assert!(prompt.contains("Cargo.toml"));
        assert!(prompt.contains("src/"));
        assert!(prompt.contains("Run with `cargo run`."));
        // The build directory is noise in every project that has one.
        assert!(!prompt.contains("target/"), "listed target/:\n{}", prompt);
    }

    // Writing HAICO.md next to an existing CLAUDE.md would shadow it, so /init has to
    // improve whichever file load_project_context is already reading.
    #[test]
    fn init_improves_the_instruction_file_the_project_already_has() {
        let dir = init_dir("existing");
        fs::write(dir.join("CLAUDE.md"), "# Rules\n\nNever touch generated code.\n").unwrap();

        assert_eq!(init_target_file(dir.path()), "CLAUDE.md");

        let prompt = build_init_prompt(dir.path());
        assert!(prompt.contains("`CLAUDE.md`"), "wrong target:\n{}", prompt);
        assert!(!prompt.contains("`HAICO.md`"));
        assert!(prompt.contains("Never touch generated code."));
        assert!(prompt.contains("Improve it instead of starting over"));
    }

    // An empty file is not instructions; load_project_context skips it, so this must too,
    // or /init would "improve" a file nothing reads.
    #[test]
    fn init_ignores_an_empty_instruction_file() {
        let dir = init_dir("empty");
        fs::write(dir.join("AGENTS.md"), "\n  \n").unwrap();

        assert_eq!(init_target_file(dir.path()), "HAICO.md");
    }
}

