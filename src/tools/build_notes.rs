//! What a failed cargo build is told about the library: which covered crates it involves,
//! and the ready-made question that points the model at the article.
//!
//! Decided from the output, never the exit status — models pipe builds through `| tail`,
//! whose status is always 0. Once per distinct error set, and once for "hangs" however it
//! was caught (`BuildNudges`), since the same note under every rebuild is how a model
//! learns to skip notes.
//!
//! It used to be split three ways: the analysis in `library.rs` beside the store and the
//! search, the state that makes it once-only as two loose mutexes on the tool server, and
//! the decision in `shell.rs`. The store is `library.rs`'s; reading a build is this file's.

use crate::library::{one_line, Library};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Whether one word of a shell command is cargo itself, by path or quoted.
pub(crate) fn is_cargo_program(word: &str) -> bool {
    let program = word.trim_matches('"');
    let program = program.rsplit(['/', '\\']).next().unwrap_or(program);
    program == "cargo" || program == "cargo.exe"
}

fn code_name(crate_name: &str) -> String {
    crate_name.replace('-', "_").to_lowercase()
}

/// The dependencies a `Cargo.toml` names, from every dependency table it has.
///
/// Renamed dependencies (`web = { package = "actix-web" }`) are reported by their package,
/// which is the name the library files them under.
pub(crate) fn cargo_dependencies(manifest: &str) -> Vec<String> {
    let Ok(table) = manifest.parse::<toml::Table>() else {
        return Vec::new();
    };
    let mut tables: Vec<&toml::Table> = Vec::new();
    for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(t) = table.get(key).and_then(|v| v.as_table()) {
            tables.push(t);
        }
        if let Some(t) = table
            .get("workspace")
            .and_then(|w| w.get(key))
            .and_then(|v| v.as_table())
        {
            tables.push(t);
        }
    }
    if let Some(targets) = table.get("target").and_then(|v| v.as_table()) {
        for target in targets.values() {
            for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
                if let Some(t) = target.get(key).and_then(|v| v.as_table()) {
                    tables.push(t);
                }
            }
        }
    }

    let mut out = Vec::new();
    for t in tables {
        for (name, spec) in t {
            let name = spec
                .get("package")
                .and_then(|p| p.as_str())
                .unwrap_or(name)
                .to_string();
            if !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

/// The error lines of a failed cargo build, deduplicated, in the order they appeared.
///
/// Only the headlines — `error[E0599]: no method named ...` — and not cargo's own closing
/// remarks ("could not compile", "aborting due to"), which say nothing about what went
/// wrong and are the same for every failure.
pub(crate) fn compile_errors(output: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        let is_error = line.starts_with("error[E") || line.starts_with("error:");
        if !is_error {
            continue;
        }
        let noise = [
            "error: could not compile",
            "error: aborting due to",
            "error: test failed",
            "error: process didn't exit",
            "error: Recipe",
        ];
        if noise.iter().any(|n| line.starts_with(n)) {
            continue;
        }
        let line = line.to_string();
        if !out.contains(&line) {
            out.push(line);
        }
    }
    out
}

/// Whether a shell command is a cargo build of some kind — the commands whose failures are
/// compile errors worth looking up.
pub(crate) fn is_cargo_build(command: &str) -> bool {
    let words: Vec<&str> = command.split_whitespace().collect();
    words.iter().enumerate().any(|(at, word)| {
        if !is_cargo_program(word) {
            return false;
        }
        // `cargo +nightly build`: a toolchain may come before the subcommand. It used to be
        // matched by a second rule that knew neither paths nor the short spellings, so
        // `cargo +nightly t` was not a build.
        let mut rest = words[at + 1..].iter().skip_while(|word| word.starts_with('+'));
        rest.next().is_some_and(|subcommand| {
            matches!(*subcommand, "build" | "b" | "check" | "c" | "test" | "t" | "run" | "r" | "clippy" | "bench")
        })
    })
}

/// What a test run that compiled says went wrong: tests that failed, tests libtest reported as
/// still running after a minute, and the panic that best says why.
#[derive(Default, Debug, PartialEq)]
pub(crate) struct TestTrouble {
    pub(crate) failed: Vec<String>,
    pub(crate) hung: Vec<String>,
    pub(crate) panic: Option<String>,
}

/// Read from every part of libtest's output that survives a `| tail`: the per-test
/// `test name ... FAILED` lines scroll off first, so the closing `failures:` list is read too —
/// the second run this was built for piped its tests through `tail -40`, and the only failing
/// run it had was reported by that list alone.
pub(crate) fn test_trouble(output: &str) -> TestTrouble {
    let mut trouble = TestTrouble::default();
    let add = |list: &mut Vec<String>, name: &str| {
        let name = name.trim().to_string();
        if !name.is_empty() && !list.contains(&name) {
            list.push(name);
        }
    };
    let mut in_list = false;
    let mut panics: Vec<(bool, String)> = Vec::new();
    let mut lines = output.lines().peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if in_list {
            // The list is bare names, four spaces in, ending at a blank line.
            if raw.starts_with("    ") && !line.is_empty() && !line.contains(' ') {
                add(&mut trouble.failed, line);
                continue;
            }
            in_list = false;
        }
        if line == "failures:" {
            in_list = true;
            continue;
        }
        if let Some((_, place)) = line.split_once(" panicked at ") {
            // The message is the next line. A panic in the code under test says more than one in
            // the test that noticed it (`client thread panicked: Any { .. }`), so those come first.
            if let Some(message) = lines.peek().map(|l| l.trim()).filter(|l| !l.is_empty()) {
                let in_tests = place.starts_with("tests") || place.contains("/tests/") || place.contains("\\tests\\");
                panics.push((in_tests, message.to_string()));
            }
            continue;
        }
        let Some(rest) = line.strip_prefix("test ") else { continue };
        if let Some(name) = rest.strip_suffix(" ... FAILED") {
            add(&mut trouble.failed, name);
        } else if let Some((name, _)) = rest.split_once(" has been running for over") {
            add(&mut trouble.hung, name);
        }
    }
    trouble.panic = panics
        .iter()
        .find(|(in_tests, _)| !in_tests)
        .or_else(|| panics.first())
        .map(|(_, message)| one_line(message, 200));
    trouble
}

/// The crates a hang is worth asking about, runtime first, out of those a project involves.
///
/// A hang is almost always in whatever schedules the I/O: a blocked runtime thread, a read that
/// waits on a reply nobody sends, an accept loop on a blocking socket. Listing every covered crate
/// the project depends on said nothing about where to look: a p2p_auth run's note named nine
/// (anyhow, chacha20poly1305, clap, ed25519-dalek, ...) for a tokio test that hung because a
/// listener built with `from_std` was never made non-blocking, and its example question began
/// with "anyhow". Empty when none of them is involved, and then the note names them all.
const HANG_SUSPECTS: &[&str] =
    &["tokio", "mio", "tokio-util", "futures", "hyper", "axum", "actix-web", "reqwest", "tonic", "async-std", "smol"];

/// The one line that follows the hang note on later hangs: every second one from the second on,
/// since the full note went out with the first (see `BuildNudges`).
///
/// A line rather than the note again, and with the count in it, so that it says something new
/// each time: "4th time" is what tells a model the approach it keeps retrying is not working.
/// Once only was the rule before, and on a p2p_auth run it meant one note and then ten silent
/// timeouts over 29 minutes, the model rewriting its tests without once asking the library,
/// which had the cause (a listener converted with `from_std` and never made non-blocking).
pub(crate) fn hang_reminder(crates: &[String], times: usize) -> Option<String> {
    let narrowed = hang_suspects(crates);
    let crates: &[String] = if narrowed.is_empty() { crates } else { &narrowed };
    let first = crates.first()?;
    Some(format!(
        "[haico] Hung again, the {} time this session. The usual causes of a hang are written up \
         in the library; ask before trying the same approach again: library(question: \
         \"{}: a test hangs and never finishes. What are the usual causes?\").",
        ordinal(times),
        first
    ))
}

fn ordinal(n: usize) -> String {
    let suffix = match (n % 10, n % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

fn hang_suspects(crates: &[String]) -> Vec<String> {
    HANG_SUSPECTS
        .iter()
        .filter(|suspect| crates.iter().any(|c| c == *suspect))
        .map(|suspect| suspect.to_string())
        .collect()
}

/// The note for a build that compiled and then failed at runtime — tests failing, tests hanging,
/// or the whole command killed at its timeout.
///
/// A compile error names the API it is about; a runtime failure names nothing, and that is where
/// a model stops asking and starts guessing. The first run this was built for compiled its mio
/// server and then spent forty minutes splitting the test suite by hand to find a hang, with the
/// causes (EOF ignored, a connection dropped before its last reply was flushed) written up in the
/// library the whole time. So the note names the symptom, and hands over a question phrased the
/// way the articles' tags are: by what went wrong, not by what caused it.
pub(crate) fn runtime_failure_note(crates: &[String], trouble: &TestTrouble, timed_out: bool) -> Option<String> {
    if crates.is_empty() || (trouble.failed.is_empty() && trouble.hung.is_empty() && !timed_out) {
        return None;
    }
    let hanging = timed_out || !trouble.hung.is_empty();
    let narrowed = if hanging { hang_suspects(crates) } else { Vec::new() };
    let crates: &[String] = if narrowed.is_empty() { crates } else { &narrowed };
    let names = match crates {
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {}", init.join(", "), last),
        [] => unreachable!(),
    };
    let listed = |tests: &[String]| {
        let shown: Vec<&str> = tests.iter().take(3).map(String::as_str).collect();
        let more = tests.len().saturating_sub(3);
        if more > 0 { format!("{} (and {} more)", shown.join(", "), more) } else { shown.join(", ") }
    };
    let (what, symptom) = if hanging {
        let which = if trouble.hung.is_empty() { String::new() } else { format!(": {}", listed(&trouble.hung)) };
        (format!("Something hangs{}", which), "a test hangs and never finishes".to_string())
    } else {
        let symptom = match &trouble.panic {
            // The panic is the searchable part: "I/O source already registered with a Registry"
            // finds the section on registration rules, a test's name finds nothing.
            Some(panic) => format!("test {} fails with: {}", trouble.failed[0], panic),
            None => format!("test {} fails", trouble.failed[0]),
        };
        (format!("Tests fail: {}", listed(&trouble.failed)), symptom)
    };
    // Says what the articles are, not what they contain: the first wording promised "hangs, lost
    // replies, resets and busy loops", true of mio and tokio and false of a web framework, and
    // was handed to a model whose failing test was a path assertion in an actix-files server.
    Some(format!(
        "[haico] {what}, in a project that uses {names}. The library has articles on {names}, \
         written from how code using them actually behaves and fails. Before debugging \
         further, call library and describe the symptom — for example: \
         library(question: \"{krate}: {symptom}. What are the usual causes?\").",
        what = what,
        names = names,
        krate = crates[0],
        symptom = symptom.replace('"', "'"),
    ))
}

/// The directory a command starts by moving into: `cd app && cargo build`, `cd /d "C:\x y" && ..`.
/// Only a leading `cd` joined by `&&` or `;`, since that is the one whose effect reaches the
/// rest of the line.
pub(crate) fn leading_cd(command: &str) -> Option<PathBuf> {
    let re = regex::Regex::new(r#"^\s*cd\s+(?:/d\s+)?(?:"([^"]+)"|'([^']+)'|([^\s&;|]+))\s*(?:&&|;)"#).ok()?;
    let caps = re.captures(command)?;
    let dir = caps.get(1).or(caps.get(2)).or(caps.get(3))?.as_str();
    Some(PathBuf::from(dir))
}

/// The library's crates that a failed build involves: named in the errors, or depended on
/// by the project being built. Errors first, because a crate named in the error itself is
/// the likelier culprit.
pub(crate) fn involved_crates(library_crates: &[String], output: &str, dependencies: &[String]) -> Vec<String> {
    let lower = output.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    for name in library_crates {
        let code = code_name(name);
        // `mio::Poll`, `use mio;`, or a registry path like `mio-1.0.2\src`.
        // Anchored at the start of a path, not just a word: `\bio::` would find a crate
        // called `io` in every `std::io::`.
        let pattern = format!(
            r"(?m)(?:^|[^\w:]){code}::|`{code}`|\b{name}-\d+\.\d+",
            code = regex::escape(&code),
            name = regex::escape(&name.to_lowercase())
        );
        let named = regex::Regex::new(&pattern).is_ok_and(|re| re.is_match(&lower));
        if named && !out.contains(name) {
            out.push(name.clone());
        }
    }
    for name in library_crates {
        let depended = dependencies.iter().any(|d| code_name(d) == code_name(name));
        if depended && !out.contains(name) {
            out.push(name.clone());
        }
    }
    out
}

/// The subjects a failed build's note names, in the order it names them: whatever explains the
/// first error, then whatever explains the others, then the crates the other errors name, then
/// the crates the build merely depends on.
///
/// The order matters because the note's example question pairs its first subject with the
/// first error. Ordered by the library instead, it once suggested asking "ed25519-dalek — how
/// do I fix: unresolved imports `tokio::sync..`": both crates were involved, and ed25519-dalek
/// sorts first. Within each group, error codes and phrases come before crate names — a code is
/// the most direct evidence there is, and for the language itself (a borrow error names no crate
/// and depends on none) it is the only evidence there is.
pub(crate) fn subjects_for_build(library: &Library, errors: &[String], crates: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut add = |names: Vec<String>| {
        for name in names {
            if !out.contains(&name) {
                out.push(name);
            }
        }
    };
    if let Some(first) = errors.first() {
        add(library.subjects_for_errors(std::slice::from_ref(first)));
        add(involved_crates(crates, first, &[]));
    }
    add(library.subjects_for_errors(errors));
    add(involved_crates(crates, &errors.join("\n"), &[]));
    add(crates.to_vec());
    out
}

/// The note added under a failed build's output, or `None` when there is nothing to say.
///
/// It names the crates and quotes the first error as a ready-made question, because a small
/// model told only "you could use the library" rarely works out what to ask — and one that
/// is handed the question usually does ask it.
pub(crate) fn build_failure_note(subjects: &[String], errors: &[String]) -> Option<String> {
    if subjects.is_empty() || errors.is_empty() {
        return None;
    }
    let names = match subjects {
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {}", init.join(", "), last),
        [] => unreachable!(),
    };
    let first = one_line(&errors[0], 200);
    Some(format!(
        "[haico] The library has articles on {names}, which this build involves. Before \
         changing the code again, call library with a question about these errors — \
         for example: library(question: \"{crate} — how do I fix: {first}\"). It answers \
         from documentation written for exactly this, which is more reliable than guessing \
         at the API.",
        names = names,
        crate = subjects[0],
        first = first.replace('"', "'"),
    ))
}


// The fingerprint every hang shares, so the full hang note goes out once per hang however it
// was caught, and finished tests can clear it for the next one.
const HANG_KEY: &str = "runtime:hang";

/// What the build notes remember between commands, kept together because they change
/// together: a run of tests that finishes clears the hang from both.
#[derive(Default)]
pub(crate) struct BuildNudges {
    // Fingerprints of the error sets a failed build has already been told to look up. Once
    // per distinct set of errors: a model rebuilding against the same errors has either asked
    // already or decided not to.
    nudged: HashSet<u64>,
    // Cargo runs in a row that hung (killed at the timeout, or reported by libtest as running
    // for over a minute) since tests last ran to the end. The full hang note is given once;
    // this count is what the short reminders after it carry.
    hang_count: usize,
}

impl BuildNudges {
    /// The library note for a failed build, if this failure deserves one.
    ///
    /// Compile errors get the note that quotes the first error; a build that compiled and
    /// then failed its tests, hung, or was killed at the timeout gets the one that names the
    /// symptom.
    pub(crate) fn note_for(
        &mut self,
        root: &Path,
        command: &str,
        output: &str,
        cwd: Option<&str>,
        timed_out: bool,
    ) -> Option<String> {
        if !is_cargo_build(command) {
            return None;
        }
        let errors = compile_errors(output);
        let trouble = test_trouble(output);
        if errors.is_empty() && trouble.failed.is_empty() && trouble.hung.is_empty() && !timed_out {
            // Tests that ran to the end mean the hang is solved, so a later one is a new
            // problem: the count starts over and it gets the full note again. Only a test run
            // counts — a `cargo check` passing between two hanging test runs solves nothing.
            if output.contains("test result:") {
                self.hang_count = 0;
                self.nudged.remove(&crate::tools::content_hash(HANG_KEY.as_bytes()));
            }
            return None;
        }
        let hanging = errors.is_empty() && (timed_out || !trouble.hung.is_empty());
        if hanging {
            self.hang_count += 1;
        }

        let library = Library::load(root);
        // The explicit cwd, then a leading `cd <dir> &&` — which is how models usually say
        // where to build — then the process's own directory.
        let dir = match cwd.filter(|c| !c.trim().is_empty()) {
            Some(dir) => PathBuf::from(dir),
            None => match leading_cd(command) {
                Some(dir) => dir,
                None => std::env::current_dir().ok()?,
            },
        };
        let dependencies = std::fs::read_to_string(dir.join("Cargo.toml"))
            .map(|manifest| cargo_dependencies(&manifest))
            .unwrap_or_default();
        let crates = involved_crates(&library.crates(), output, &dependencies);
        let (note, key) = if errors.is_empty() {
            let note = runtime_failure_note(&crates, &trouble, timed_out)?;
            // A hang is one symptom however it was caught — a timeout or libtest's warning —
            // so it is fingerprinted as one; each set of failing tests is another.
            let key = if hanging { HANG_KEY.to_string() } else { format!("runtime:{}", trouble.failed.join("\n")) };
            (note, key)
        } else {
            let subjects = subjects_for_build(&library, &errors, &crates);
            (build_failure_note(&subjects, &errors)?, errors.join("\n"))
        };

        // Recorded only once there is a note to give, so an error set that got none (no
        // covered crate involved) does not use up the one it would get later.
        if !self.nudged.insert(crate::tools::content_hash(key.as_bytes())) {
            // The full note went out already. A hang alone gets a reminder after it, a line
            // with the count, on every second hang: often enough not to leave a stuck model
            // alone for half an hour, rarely enough not to hound one making progress.
            let hangs = self.hang_count;
            if hanging && hangs >= 2 && hangs % 2 == 0 {
                return hang_reminder(&crates, hangs);
            }
            return None;
        }
        Some(note)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use std::fs;

    fn write(root: &Path, id: &str, text: &str) {
        let path = root.join(format!("{}.md", id));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    #[test]
    fn the_note_leads_with_what_explains_the_first_error() {
        let dir = TempDir::new("library-first-error");
        write(dir.path(), "ed25519-dalek/basics", "# Ed25519\n\nx\n");
        write(dir.path(), "tokio/basics", "# Tokio\n\nx\n");
        write(dir.path(), "rust/ownership", "---\ntopic: Rust ownership\ncodes: E0382\n---\n# Ownership\n\nx\n");
        let library = Library::load(dir.path());
        // The case that was wrong: two crates involved, and the one that sorts first is not the
        // one the first error is about.
        let errors = vec![
            "error[E0432]: unresolved imports `tokio::sync::Mutex`".to_string(),
            "error[E0382]: borrow of moved value: `key`".to_string(),
        ];
        let crates = vec!["ed25519-dalek".to_string(), "tokio".to_string()];
        let subjects = subjects_for_build(&library, &errors, &crates);
        assert_eq!(subjects, ["tokio", "Rust ownership", "ed25519-dalek"]);
        let note = build_failure_note(&subjects, &errors).unwrap();
        assert!(note.contains("library(question: \"tokio — how do I fix: error[E0432]"), "{}", note);

        // A first error explained by its code leads with the article for that code.
        let errors = vec![errors[1].clone(), errors[0].clone()];
        assert_eq!(subjects_for_build(&library, &errors, &crates), ["Rust ownership", "tokio", "ed25519-dalek"]);
    }

    #[test]
    fn dependencies_come_from_every_table_and_by_their_package_name() {
        let deps = cargo_dependencies(
            "[package]\nname = \"x\"\n\n[dependencies]\ntokio = \"1\"\nweb = { package = \"actix-web\", version = \"4\" }\n\n\
[dev-dependencies]\ngetopts = \"0.2\"\n\n[target.'cfg(unix)'.dependencies]\nmio = \"1\"\n",
        );
        assert_eq!(deps, ["tokio", "actix-web", "getopts", "mio"]);
        assert!(cargo_dependencies("not toml [").is_empty());
    }

    #[test]
    fn only_the_error_headlines_of_a_build_count() {
        let output = "   Compiling x v0.1.0\n\
error[E0599]: no method named `poll` found for struct `mio::Poll`\n  --> src/main.rs:3:5\n\
error[E0599]: no method named `poll` found for struct `mio::Poll`\n\
warning: unused import\n\
error: aborting due to 2 previous errors\n\
error: could not compile `x` (bin \"x\")\n";
        assert_eq!(
            compile_errors(output),
            ["error[E0599]: no method named `poll` found for struct `mio::Poll`"]
        );
    }

    #[test]
    fn cargo_builds_are_recognized_and_other_commands_are_not() {
        assert!(is_cargo_build("cargo build --release"));
        assert!(is_cargo_build("cd app && cargo check"));
        assert!(is_cargo_build("cargo +nightly test"));
        assert!(is_cargo_build("C:\\Users\\me\\.cargo\\bin\\cargo.exe t"));
        // A toolchain with the short spellings and a path, which the toolchain case used to
        // know neither of.
        assert!(is_cargo_build("cargo +nightly t"));
        assert!(is_cargo_build("\"C:\\cargo\\bin\\cargo.exe\" +stable b"));
        assert!(!is_cargo_build("cargo +nightly fmt"));
        assert!(!is_cargo_build("cargo fmt"));
        assert!(!is_cargo_build("echo cargo"));
        assert!(!is_cargo_build("cargo add tokio"));
    }

    #[test]
    fn a_leading_cd_names_the_build_directory() {
        assert_eq!(leading_cd("cd app && cargo build"), Some(PathBuf::from("app")));
        assert_eq!(leading_cd(r#"cd /d "C:\my dir" && cargo test"#), Some(PathBuf::from(r"C:\my dir")));
        assert_eq!(leading_cd(r"  cd E:\x\y; cargo test"), Some(PathBuf::from(r"E:\x\y")));
        assert_eq!(leading_cd("cargo build"), None);
        assert_eq!(leading_cd("cd app"), None);
        assert_eq!(leading_cd("echo cd app && cargo build"), None);
    }

    #[test]
    fn failing_and_hanging_tests_are_read_from_libtest_output() {
        let output = "running 3 tests\n\
test reverses_a_line ... ok\n\
test burst ... FAILED\n\
test oversized has been running for over 60 seconds\n\
test burst ... FAILED\n";
        let trouble = test_trouble(output);
        assert_eq!(trouble.failed, ["burst"]);
        assert_eq!(trouble.hung, ["oversized"]);
        assert_eq!(test_trouble("test result: ok. 3 passed"), TestTrouble::default());
    }

    #[test]
    fn the_tail_of_a_failing_run_still_names_its_tests_and_its_panic() {
        // Exactly what `cargo test 2>&1 | tail -40` left of a real run: no `... FAILED` lines.
        let tail = "thread 'quit' (33684) panicked at tests\\server.rs:55:18:\n\
client thread panicked: Any { .. }\n\
\n\
---- line_sent_one_byte_at_a_time stdout ----\n\
\n\
thread '<unnamed>' (17664) panicked at src\\lib.rs:102:10:\n\
failed to register listener: Custom { kind: AlreadyExists, error: \"I/O source already registered with a `Registry`\" }\n\
\n\
failures:\n    large_reply_burst\n    line_sent_one_byte_at_a_time\n\
\n\
test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s\n";
        let trouble = test_trouble(tail);
        assert_eq!(trouble.failed, ["large_reply_burst", "line_sent_one_byte_at_a_time"]);
        // The panic in src/ wins over the test's own "client thread panicked".
        let panic = trouble.panic.clone().unwrap();
        assert!(panic.starts_with("failed to register listener"), "{}", panic);

        let note = runtime_failure_note(&["mio".to_string()], &trouble, false).unwrap();
        assert!(note.contains("already registered with a `Registry`"), "{}", note);

        // The first `failures:` block holds `---- name stdout ----` sections, not names.
        let details = "failures:\n\n---- a stdout ----\nboom\n";
        assert!(test_trouble(details).failed.is_empty());
    }

    #[test]
    fn a_hang_note_names_the_runtime_not_every_dependency() {
        // The p2p_auth run's crates, in the order the note listed them.
        let crates: Vec<String> = ["anyhow", "chacha20poly1305", "clap", "ed25519-dalek", "hkdf", "rand", "sha2", "tokio", "x25519-dalek"]
            .iter()
            .map(|c| c.to_string())
            .collect();
        let hung = TestTrouble { hung: vec!["test_two_nodes_exchange_messages".into()], ..Default::default() };
        let note = runtime_failure_note(&crates, &hung, false).unwrap();
        assert!(note.contains("in a project that uses tokio. The library has articles on tokio,"), "{note}");
        assert!(note.contains("library(question: \"tokio: a test hangs"), "{note}");
        assert!(!note.contains("anyhow"), "{note}");
        // Killed at the timeout is the same symptom.
        let note = runtime_failure_note(&crates, &TestTrouble::default(), true).unwrap();
        assert!(note.contains("uses tokio.") && !note.contains("clap"), "{note}");

        // A failing (not hanging) test still names them all: an assertion can be about anything.
        let failed = TestTrouble { failed: vec!["roundtrip".into()], ..Default::default() };
        assert!(runtime_failure_note(&crates, &failed, false).unwrap().contains("anyhow"));
        // No runtime involved: nothing to narrow to, so nothing is dropped.
        let sync_only = vec!["rayon".to_string(), "crossbeam-channel".to_string()];
        let note = runtime_failure_note(&sync_only, &hung, false).unwrap();
        assert!(note.contains("rayon and crossbeam-channel"), "{note}");
    }

    #[test]
    fn the_runtime_note_names_the_symptom() {
        let mio = vec!["mio".to_string()];
        let hung = TestTrouble { hung: vec!["oversized".into()], ..Default::default() };
        let note = runtime_failure_note(&mio, &hung, false).unwrap();
        assert!(note.contains("Something hangs: oversized"), "{}", note);
        assert!(note.contains("mio: a test hangs and never finishes"), "{}", note);

        // A command killed at its timeout is a hang even when libtest never said so.
        let note = runtime_failure_note(&mio, &TestTrouble::default(), true).unwrap();
        assert!(note.contains("Something hangs, in a project"), "{}", note);

        let failed = TestTrouble { failed: vec!["burst".into()], ..Default::default() };
        let note = runtime_failure_note(&mio, &failed, false).unwrap();
        assert!(note.contains("Tests fail: burst"), "{}", note);
        assert!(note.contains("mio: test burst fails"), "{}", note);

        assert!(runtime_failure_note(&mio, &TestTrouble::default(), false).is_none());
        assert!(runtime_failure_note(&[], &hung, true).is_none());
    }

    #[test]
    fn crates_named_in_the_errors_come_before_mere_dependencies() {
        let library = vec!["actix-web".to_string(), "mio".to_string(), "tokio".to_string()];
        let output = "error[E0433]: failed to resolve: could not find `web` in `actix_web`\n\
  --> C:\\Users\\me\\.cargo\\registry\\src\\index\\mio-1.0.2\\src\\poll.rs:3:1";
        let involved = involved_crates(&library, output, &["tokio".to_string(), "serde".to_string()]);
        assert_eq!(involved, ["actix-web", "mio", "tokio"]);
        assert!(involved_crates(&library, "error: x", &[]).is_empty());

        // A path that merely passes through a crate's name is not that crate.
        let io = vec!["io".to_string()];
        assert!(involved_crates(&io, "expected `std::io::Result<()>`", &[]).is_empty());
        assert_eq!(involved_crates(&io, "use io::Read;", &[]), ["io"]);
    }

    #[test]
    fn the_note_hands_the_model_a_question() {
        let note = build_failure_note(
            &["mio".to_string(), "tokio".to_string()],
            &["error[E0599]: no method named \"poll\"".to_string()],
        )
        .unwrap();
        assert!(note.contains("mio and tokio"), "{}", note);
        assert!(note.contains("library(question: \"mio — how do I fix: error[E0599]"), "{}", note);
        assert!(build_failure_note(&[], &["e".into()]).is_none());
        assert!(build_failure_note(&["mio".into()], &[]).is_none());
    }
}
