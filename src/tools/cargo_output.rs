//! Cargo's output, shortened before it is stored, so the history does not fill with lines that
//! never mattered.
//!
//! Done at the source and nowhere later, because of the server's prefix cache: a tool result is
//! the tail of the history when it is stored, so filtering it costs nothing, while removing the
//! same lines from an older result would re-prefill every turn behind it. What is removed is
//! what a model never acts on: progress lines (`Compiling`, `Locking`, the `Adding` lines for
//! every package a lock pulled in), passing tests, and — only while the build has errors — the
//! bodies of its warnings, which are what fills a screen ahead of the error that ended the
//! build. Errors are never touched, and a build with warnings only (clippy, a check after the
//! errors are fixed) keeps them whole, because then the warnings are what it was run for.
//!
//! Measured on a p2p_auth run: of the 80 lines one `cargo test | head -80` let through, 44 were
//! `Compiling` lines, and the errors it was run for were all cut off by the `head`.
//!
//! The note says what was left out, so a model reading a short build does not take it for the
//! whole of it.

use regex::Regex;
use std::sync::LazyLock;

/// Lines that report progress and nothing else.
static PROGRESS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^\s*(Compiling|Checking|Downloaded|Downloading|Updating|Locking|Blocking|Fresh|Documenting|Unpacking|Removing|Packaging|Verifying|Archiving)\s",
    )
    .unwrap()
});

/// `Adding foo v1.2.3` with nothing after it: a package a lock pulled in. The one worth keeping,
/// `Adding foo v1.2.3 to dependencies`, says so and does not match.
static LOCKED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*Adding \S+ v\S+\s*$").unwrap());

static PASSED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^test \S.* \.\.\. ok$").unwrap());

/// The headline of a compiler diagnostic. Its body runs to the next blank line: rustc never
/// leaves one inside a diagnostic, and always leaves one after it.
static WARNING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^warning(\[\w+\])?: ").unwrap());
static ERROR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^error(\[E\d{4}\])?: ").unwrap());

/// Whether a command runs cargo at all, whatever the subcommand.
pub(crate) fn runs_cargo(command: &str) -> bool {
    command.split_whitespace().any(super::build_notes::is_cargo_program)
}

/// The output with the lines that never matter left out, and a note saying what went, or the
/// output unchanged and `None` when nothing did.
pub(crate) fn condense(command: &str, output: &str) -> (String, Option<String>) {
    if !runs_cargo(command) {
        return (output.to_string(), None);
    }
    // Warning bodies only go when something failed to compile; the error count is taken from
    // headlines that are rustc's, not from cargo's own "error: could not compile" summary, which
    // has the same shape and is also an error.
    let has_errors = output.lines().any(|line| ERROR.is_match(line));

    let mut kept: Vec<&str> = Vec::new();
    let (mut progress, mut passed, mut warnings) = (0usize, 0usize, 0usize);
    let mut lines = output.lines().peekable();
    while let Some(line) = lines.next() {
        if PROGRESS.is_match(line) || LOCKED.is_match(line) {
            progress += 1;
            continue;
        }
        if PASSED.is_match(line.trim_end()) {
            passed += 1;
            continue;
        }
        if has_errors && WARNING.is_match(line) {
            // The headline, and where it points; the snippet, labels and notes go.
            let mut body: Vec<&str> = Vec::new();
            while let Some(next) = lines.peek() {
                // A new headline ends it too: cargo's one-line "generated 1 warning" is followed
                // straight by the next diagnostic, with no blank line between.
                if next.trim().is_empty() || WARNING.is_match(next) || ERROR.is_match(next) || PROGRESS.is_match(next) {
                    break;
                }
                body.push(lines.next().unwrap());
            }
            kept.push(line);
            if let Some(location) = body.iter().find(|l| l.trim_start().starts_with("-->")) {
                kept.push(location);
            }
            // A one-line warning (cargo's "generated 3 warnings") had no body to remove.
            if !body.is_empty() {
                warnings += 1;
            }
            continue;
        }
        kept.push(line);
    }

    let mut parts: Vec<String> = Vec::new();
    if progress > 0 {
        parts.push(format!("{} progress line{} (Compiling, Locking and the like)", progress, plural(progress)));
    }
    if passed > 0 {
        parts.push(format!("{} passing test{}", passed, plural(passed)));
    }
    if warnings > 0 {
        parts.push(format!(
            "the bodies of {} warning{} (headline and location kept; the build has errors, which are shown in full)",
            warnings,
            plural(warnings)
        ));
    }
    if parts.is_empty() {
        return (output.to_string(), None);
    }

    let mut text = kept.join("\n");
    if output.ends_with('\n') {
        text.push('\n');
    }
    let note = format!("[haico] Shortened this cargo output: left out {}.", join_list(&parts));
    (text, Some(note))
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn join_list(parts: &[String]) -> String {
    match parts {
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {}", init.join(", "), last),
        [] => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shaped like the p2p_auth run's `cargo test` output, cut down.
    const FAILED_BUILD: &str = "   Compiling cfg-if v1.0.5
   Compiling tokio v1.53.2
   Compiling p2p_auth v0.1.0 (E:\\p2p_auth)
warning: variable does not need to be mutable
   --> src\\lib.rs:108:5
    |
108 |     mut stream: TcpStream,
    |     ----^^^^^^
    |     |
    |     help: remove this `mut`
    |
    = note: `#[warn(unused_mut)]` (part of `#[warn(unused)]`) on by default

warning: `p2p_auth` (lib) generated 1 warning
error[E0425]: cannot find value `bob_pk` in this scope
  --> tests\\integration.rs:40:23
   |
40 |     alice_trust.insert(bob_pk);
   |                        ^^^^^^ not found in this scope
help: the leading underscore in `_bob_pk` marks it as unused, consider renaming it
   |
37 |     let bob_pk = bob_id.verifying_key();
   |         ~~~~~~

error: could not compile `p2p_auth` (test \"integration\") due to 1 previous error
";

    #[test]
    fn a_failed_build_keeps_its_errors_whole_and_its_warnings_short() {
        let (text, note) = condense("cargo test 2>&1", FAILED_BUILD);
        assert!(!text.contains("Compiling"), "{text}");
        // The warning: headline and location, nothing else.
        assert!(text.contains("warning: variable does not need to be mutable\n   --> src\\lib.rs:108:5\n"), "{text}");
        assert!(!text.contains("help: remove this `mut`"), "{text}");
        assert!(text.contains("warning: `p2p_auth` (lib) generated 1 warning"), "{text}");
        // The error, every line of it, help included.
        let error = &FAILED_BUILD[FAILED_BUILD.find("error[E0425]").unwrap()..];
        assert!(text.ends_with(error), "{text}");
        let note = note.unwrap();
        assert!(note.starts_with("[haico] Shortened this cargo output: left out 3 progress lines"), "{note}");
        assert!(note.contains("the bodies of 1 warning "), "{note}");
    }

    #[test]
    fn warnings_without_errors_are_what_was_asked_for() {
        let clippy = "    Checking p2p_auth v0.1.0\nwarning: this `if` has identical blocks\n --> src/lib.rs:3:5\n  |\n3 |     if a { 1 } else { 1 }\n  |     ^^^^^^^^^^^^^^^^^^^^^\n\nwarning: `p2p_auth` (lib) generated 1 warning\n    Finished `dev` profile\n";
        let (text, note) = condense("cargo clippy", clippy);
        assert!(text.contains("3 |     if a { 1 } else { 1 }"), "{text}");
        assert!(text.contains("Finished"), "{text}");
        assert_eq!(note.unwrap(), "[haico] Shortened this cargo output: left out 1 progress line (Compiling, Locking and the like).");
    }

    #[test]
    fn passing_tests_go_and_failures_stay() {
        let run = "running 3 tests\ntest a ... ok\ntest b ... FAILED\ntest c ... ok\n\nfailures:\n\n---- b stdout ----\nthread 'b' panicked at src/lib.rs:9:5:\nboom\n\ntest result: FAILED. 2 passed; 1 failed; 0 ignored\n";
        let (text, note) = condense("cargo test", run);
        // Through cmd or a pipe a line can arrive with a trailing space or a \r; the
        // trailing space must not hide a pass, and line endings come back as plain newlines.
        let (crlf, _) = condense("cargo test", "test a ... ok \r\ntest b ... FAILED\r\n");
        assert_eq!(crlf, "test b ... FAILED\n");
        assert!(!text.contains("test a ... ok") && !text.contains("test c ... ok"), "{text}");
        for kept in ["running 3 tests", "test b ... FAILED", "panicked at src/lib.rs:9:5", "test result: FAILED"] {
            assert!(text.contains(kept), "{kept} is missing: {text}");
        }
        assert!(note.unwrap().contains("2 passing tests"));
    }

    #[test]
    fn cargo_add_keeps_what_was_added_and_its_features() {
        let add = "    Updating crates.io index\n      Adding ed25519-dalek v3.0.0 to dependencies\n             Features:\n             + fast\n             - serde\n     Locking 2 packages to latest compatible versions\n      Adding curve25519-dalek v5.0.0\n      Adding subtle v2.6.1\n";
        let (text, note) = condense("cargo add ed25519-dalek", add);
        assert_eq!(text, "      Adding ed25519-dalek v3.0.0 to dependencies\n             Features:\n             + fast\n             - serde\n");
        assert!(note.unwrap().contains("4 progress lines"));
    }

    #[test]
    fn other_commands_and_quiet_output_are_left_alone() {
        let (text, note) = condense("git status", "   Compiling nothing, this is git\n");
        assert_eq!((text.as_str(), note), ("   Compiling nothing, this is git\n", None));
        let (text, note) = condense("cargo build", "    Finished `dev` profile\n");
        assert_eq!((text.as_str(), note), ("    Finished `dev` profile\n", None));
        assert!(runs_cargo("cd p2p && C:\\Users\\me\\.cargo\\bin\\cargo.exe build"));
        assert!(!runs_cargo("echo cargoes"));
    }
}
