//! `@path` in a typed message: the file goes into the message.
//!
//! Without this a message that names a file costs a whole round — the model
//! reads the name, emits a `read` call, the server prefills the reply, and only
//! then does the work start. On a local model that round is seconds to minutes,
//! and it is entirely predictable: the user already knows which file they meant,
//! which is why they typed it. So the frontend does the read before the request
//! goes out and the model gets the contents in the same breath as the request.
//!
//! Only the parsing lives here. Reading, attaching and the workspace widening
//! are `agent::attach_mentions`, which is where the tool server and the
//! workspace are.
//!
//! **A mention is only a mention if it names a file that exists.** Everything
//! else — an email address, a Rust attribute someone pasted, a bare `@` — is
//! left in the text untouched. That is the whole defense against false
//! positives, and it is why resolution happens here rather than after parsing:
//! deciding where the token ends needs the filesystem too (see `TRAILING`).

use crate::util::absolute_path;
use crate::workspace::expand_home;
use std::path::PathBuf;

/// A file the user named with `@`.
pub(crate) struct Mention {
    /// Exactly as it was typed, `@` included. The attached block names the file
    /// by this as well as by its full path, so the model can tie the contents to
    /// the reference it can still see in the message.
    pub(crate) written: String,
    /// Absolute and canonical, in the same form the workspace compares against.
    pub(crate) path: PathBuf,
}

// Characters that end a sentence rather than a file name. Trimmed off the tail
// one at a time, and only until what is left names a file — so "@src/main.rs."
// finds the file and a directory genuinely called "build)" would still resolve
// on the first try. Doing it the other way round, trimming unconditionally,
// would make any file whose name really ends in one of these unreachable.
const TRAILING: &[char] = &['.', ',', ';', ':', '!', '?', ')', ']', '}', '>', '"', '\'', '`'];

// …and the ones that can stand immediately before it without joining the `@` to
// a word. The mirror of the set above, so "(@path)" works from both ends.
const OPENERS: &[char] = &['(', '[', '{', '<', '"', '\'', '`'];

/// Every file the text mentions, in the order they were typed, without repeats.
///
/// Repeats are dropped by resolved path rather than by what was typed: naming
/// the same file once relatively and once absolutely is one attachment, not two.
pub(crate) fn find(text: &str) -> Vec<Mention> {
    let mut found: Vec<Mention> = Vec::new();
    let mut cursor = 0;

    while let Some(offset) = text[cursor..].find('@') {
        let at = cursor + offset;

        // A word boundary, so "user@example.com" and "a@b" are not mentions. An
        // opening bracket or quote counts as one too: "(@src/main.rs)" is how a
        // path lands in an aside, and it is not the shape a false positive
        // takes — every one of those has a word character on the left.
        let starts_word = at == 0
            || text[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_whitespace() || OPENERS.contains(&c));

        // The token runs to the next whitespace: paths hold slashes, dots,
        // colons and backslashes, and there is no smaller set that covers
        // "C:\Users\x\file.rs" without also cutting it in half.
        let token_start = at + 1;
        let token_end = text[token_start..]
            .find(char::is_whitespace)
            .map(|p| token_start + p)
            .unwrap_or(text.len());
        let token = &text[token_start..token_end];

        // Past the whole token either way, so a token that was not a mention
        // cannot have a second '@' inside it scanned as one.
        cursor = token_end.max(at + 1);

        if !starts_word || token.is_empty() {
            continue;
        }

        let Some((written, path)) = resolve(token) else {
            continue;
        };
        if found.iter().any(|m| m.path == path) {
            continue;
        }
        found.push(Mention { written: format!("@{}", written), path });
    }

    found
}

/// The longest prefix of `token` that names an existing file, with the text that
/// prefix was written as.
fn resolve(token: &str) -> Option<(String, PathBuf)> {
    let mut candidate = token;
    loop {
        if !candidate.is_empty() {
            let resolved = absolute_path(&expand_home(candidate));
            if resolved.is_file() {
                return Some((candidate.to_string(), resolved));
            }
        }
        // Nothing left to try once the tail is not punctuation.
        candidate = candidate.strip_suffix(TRAILING)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    #[test]
    fn a_mention_is_the_file_it_names() {
        let dir = TempDir::new("mention");
        let file = dir.join("notes.txt");
        std::fs::write(&file, "hello").unwrap();

        let text = format!("look at @{}", file.display());
        let found = find(&text);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, absolute_path(&file));
        assert_eq!(found[0].written, format!("@{}", file.display()));
    }

    #[test]
    fn a_name_that_is_not_a_file_stays_text() {
        let dir = TempDir::new("mention-miss");
        // A path under a real directory, so only the missing file distinguishes
        // this from the case above.
        let text = format!("see @{}", dir.join("absent.txt").display());
        assert!(find(&text).is_empty());

        // The shapes this must never claim: an address, a handle, a lone '@'.
        assert!(find("mail me at user@example.com").is_empty());
        assert!(find("thanks @someone").is_empty());
        assert!(find("cost is 3 @ 5").is_empty());
    }

    #[test]
    fn punctuation_after_a_mention_is_not_part_of_the_name() {
        let dir = TempDir::new("mention-punct");
        let file = dir.join("main.rs");
        std::fs::write(&file, "fn main() {}").unwrap();

        // The three ways a path lands next to punctuation in ordinary prose.
        for text in [
            format!("fix @{}.", file.display()),
            format!("fix @{}, then build", file.display()),
            format!("fix (@{})", file.display()),
        ] {
            let found = find(&text);
            assert_eq!(found.len(), 1, "in: {}", text);
            assert_eq!(found[0].path, absolute_path(&file), "in: {}", text);
        }
    }

    #[test]
    fn a_file_named_twice_is_attached_once() {
        let dir = TempDir::new("mention-dup");
        let file = dir.join("dup.txt");
        std::fs::write(&file, "x").unwrap();

        let text = format!("@{} and again @{}", file.display(), file.display());
        assert_eq!(find(&text).len(), 1);
    }

    #[test]
    fn a_directory_is_not_a_file() {
        // Only files are attached, so a directory mention is left as text and the
        // model can list it itself if it wants to.
        let dir = TempDir::new("mention-dir");
        let text = format!("look in @{}", dir.path().display());
        assert!(find(&text).is_empty());
    }

    #[test]
    fn a_relative_mention_resolves_against_the_working_directory() {
        // Cargo.toml is at the root of this crate and the tests run there, so a
        // bare relative name is enough to prove the resolution without a
        // process-wide chdir — which would race every other test in the binary.
        let found = find("what does @Cargo.toml say?");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].written, "@Cargo.toml");
        assert!(found[0].path.ends_with("Cargo.toml"));
    }
}
