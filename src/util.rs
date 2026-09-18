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

// Long output cut in the middle, keeping both ends.
//
// The reason a command was run is almost never in the first 30 KB of what it printed. A build
// lists every crate it compiles and only then says what went wrong; a test run lists every test
// and only then the failures. Keeping the head is therefore the one way of truncating that
// reliably throws away the part that was worth having — a `cargo build` over a few hundred
// crates came back as nothing but `Compiling` lines, with the error that ended it cut off. The
// tail gets the larger share of the budget for the same reason.
//
// Cuts fall on line boundaries so that neither end starts or stops mid-message. The exception is
// a tail whose first newline is most of the way through it — one very long line, minified JSON or
// a single-line log — where snapping forward would drop the payload the caller asked to keep.
pub(crate) fn truncate_middle(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }

    let head = truncate_at_boundary(s, max_bytes / 3);
    let head_end = match head.rfind('\n') {
        Some(i) => i + 1,
        None => head.len(),
    };

    let mut tail_start = s.len() - (max_bytes - max_bytes / 3);
    while tail_start < s.len() && !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    if let Some(i) = s[tail_start..].find('\n') {
        let snapped = tail_start + i + 1;
        if s.len() - snapped >= (s.len() - tail_start) / 2 {
            tail_start = snapped;
        }
    }

    let omitted = &s[head_end..tail_start];
    format!(
        "{}\n[... {} lines omitted, {} characters ...]\n{}",
        &s[..head_end],
        omitted.lines().count(),
        omitted.len(),
        &s[tail_start..]
    )
}

// Terminal escape sequences, removed.
//
// The diff behind a permission prompt is rendered for a terminal, so it is full of color
// codes — and the very same string is what the auto-mode reviewer is handed to judge. To a
// model those are pure noise: tokens spent on "ESC[38;5;8m" before a single line of the
// code is read, and one more thing in the evidence that is not the change itself.
//
// CSI (ESC [ ... final byte in @..~) is what crossterm emits and all this has to recognize.
// Any other escape is dropped together with the byte after it, which is right for the
// two-byte forms and at least leaves nothing dangling from the ones that never turn up here.
pub(crate) fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        if chars.next() == Some('[') {
            for c in chars.by_ref() {
                if ('\u{40}'..='\u{7e}').contains(&c) {
                    break;
                }
            }
        }
    }
    out
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


// Line endings, and why the file tools have to think about them at all.
//
// `read` shows the model the file through `BufRead::lines`, which strips `\n` *and* `\r\n`.
// So the model is shown every file as if it were LF, and has no way to discover otherwise —
// `grep` goes through `str::lines` and strips the same byte. `edit` then matched the
// model's LF-shaped `old_string` against bytes still holding their `\r`, and every
// multi-line edit to a CRLF file failed with "String not found", which tells the model
// nothing it can act on: more context, less context, re-reading the file are all still LF,
// so it retried until the round limit. Single-line edits worked the whole time, because a
// one-line `old_string` has no newline in it to disagree about.
//
// The fix is to meet the model where `read` left it: translate its strings to whatever the
// file on disk actually uses, rather than asking it to guess a byte it was never shown.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LineEnding {
    Lf,
    Crlf,
}

/// What the file uses, by majority. Counting rather than sampling the first line is what
/// gets a file haico itself has damaged right: a CRLF file with three lone LFs spliced into
/// it by an older edit is still a CRLF file, and normalizing it to CRLF is the repair.
///
/// A file with no line breaks at all reports `Lf`, which costs nothing — there is no ending
/// in it to preserve, so both answers translate to the same bytes.
pub(crate) fn dominant_line_ending(text: &str) -> LineEnding {
    let crlf = text.matches("\r\n").count();
    // Every `\r\n` contains a `\n`, so the lone ones are what is left over.
    let lone_lf = text.matches('\n').count() - crlf;
    if crlf > lone_lf {
        LineEnding::Crlf
    } else {
        LineEnding::Lf
    }
}

/// Rewrite every line break in `text` as `ending`, leaving everything else alone.
///
/// Borrows when the text already agrees, so the common LF-file edit does not copy the whole
/// file to discover there was nothing to do.
///
/// A lone `\r` is deliberately not a line break here: `BufRead::lines` does not split on one
/// either, so as far as the model can see it is an ordinary character in the middle of a
/// line, and turning it into a newline would invent a line the model never read.
pub(crate) fn with_line_endings(text: &str, ending: LineEnding) -> std::borrow::Cow<'_, str> {
    if already_uses(text, ending) {
        return std::borrow::Cow::Borrowed(text);
    }
    // Via LF, so the conversion is idempotent and a mixed file lands wholly on one ending
    // rather than on whichever one it happened to have first.
    let lf = text.replace("\r\n", "\n");
    std::borrow::Cow::Owned(match ending {
        LineEnding::Lf => lf,
        LineEnding::Crlf => lf.replace('\n', "\r\n"),
    })
}

fn already_uses(text: &str, ending: LineEnding) -> bool {
    match ending {
        LineEnding::Lf => !text.contains("\r\n"),
        // Every `\n` must have its `\r`; a lone one means there is work to do.
        LineEnding::Crlf => text.matches('\n').count() == text.matches("\r\n").count(),
    }
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
    fn middle_truncation_keeps_both_ends() {
        // Short enough to pass through untouched.
        assert_eq!(truncate_middle("hello\nworld\n", 100), "hello\nworld\n");

        // The shape this exists for: a build's progress first, its error last.
        let mut log = String::new();
        for i in 0..2000 {
            log.push_str(&format!("Compiling crate-{}\n", i));
        }
        log.push_str("error[E0308]: mismatched types\n");

        let cut = truncate_middle(&log, 3000);
        assert!(cut.len() < 3400, "budget blown: {} bytes", cut.len());
        assert!(cut.starts_with("Compiling crate-0\n"));
        assert!(
            cut.ends_with("error[E0308]: mismatched types\n"),
            "the error at the end is what must survive"
        );
        assert!(cut.contains("lines omitted"));
        // The tail is the larger half, so it holds more of the log than the head does.
        let marker = cut.find("[...").unwrap();
        assert!(cut.len() - marker > marker, "the head kept more than the tail");
    }

    #[test]
    fn middle_truncation_cuts_on_boundaries() {
        // Cuts land on line ends, so no line is served in halves.
        let lines: String = (0..100).map(|i| format!("line-{:03}\n", i)).collect();
        let cut = truncate_middle(&lines, 300);
        for line in cut.lines().filter(|l| l.starts_with("line-")) {
            assert_eq!(line.len(), 8, "half a line survived: {:?}", line);
        }

        // One long line with no newline to snap to: both ends still come back whole-ish
        // rather than the tail being dropped for want of a boundary.
        let blob = format!("head\n{}tail", "x".repeat(5000));
        let cut = truncate_middle(&blob, 300);
        assert!(cut.starts_with("head\n"));
        assert!(cut.ends_with("tail"), "the payload was snapped away: {:?}", cut);

        // Multi-byte characters are never split.
        let wide: String = std::iter::repeat("привет\n").take(500).collect();
        let cut = truncate_middle(&wide, 301);
        assert!(cut.contains("привет"));
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

    #[test]
    fn ansi_escapes_come_off_before_a_model_sees_them() {
        // A delete line exactly as generate_diff writes it: grey line number, red
        // background, white text, and a reset after each.
        let colored = "\x1b[38;5;8m     6\x1b[0m \x1b[48;5;9m\x1b[38;5;15m- use std::path::Path;\x1b[0m";
        assert_eq!(strip_ansi(colored), "     6 - use std::path::Path;");

        // Text with nothing to strip is returned unchanged, byte for byte.
        assert_eq!(strip_ansi("plain line"), "plain line");
        // A lone escape at the end must not run off the end of the string.
        assert_eq!(strip_ansi("tail\x1b"), "tail");
    }

    #[test]
    fn line_endings_are_decided_by_majority_and_converted_both_ways() {
        assert_eq!(dominant_line_ending("a\nb\n"), LineEnding::Lf);
        assert_eq!(dominant_line_ending("a\r\nb\r\n"), LineEnding::Crlf);
        // No line breaks at all: nothing to preserve, and Lf converts to the same bytes.
        assert_eq!(dominant_line_ending("one line"), LineEnding::Lf);

        // The case this exists for: a CRLF file an older haico spliced a lone LF into is
        // still a CRLF file, and the majority is what says so.
        assert_eq!(dominant_line_ending("a\r\nb\nc\r\nd\r\n"), LineEnding::Crlf);
        // And the mirror image, so a stray CRLF pasted into an LF file does not flip it.
        assert_eq!(dominant_line_ending("a\nb\r\nc\nd\n"), LineEnding::Lf);

        assert_eq!(with_line_endings("a\nb\n", LineEnding::Crlf), "a\r\nb\r\n");
        assert_eq!(with_line_endings("a\r\nb\r\n", LineEnding::Lf), "a\nb\n");
        // Idempotent, and a mixed input lands wholly on one ending either way.
        assert_eq!(with_line_endings("a\r\nb\nc", LineEnding::Crlf), "a\r\nb\r\nc");
        assert_eq!(with_line_endings("a\r\nb\nc", LineEnding::Lf), "a\nb\nc");

        // A lone \r is a character inside a line, not a line break: BufRead::lines does not
        // split on one, so the model never saw a line boundary there to preserve.
        assert_eq!(with_line_endings("a\rb\n", LineEnding::Crlf), "a\rb\r\n");
        assert_eq!(with_line_endings("a\rb\r\n", LineEnding::Lf), "a\rb\n");

        // Text that already agrees is handed straight back rather than copied.
        assert!(matches!(
            with_line_endings("a\nb\n", LineEnding::Lf),
            std::borrow::Cow::Borrowed(_)
        ));
        assert!(matches!(
            with_line_endings("a\r\nb\r\n", LineEnding::Crlf),
            std::borrow::Cow::Borrowed(_)
        ));
    }
}
