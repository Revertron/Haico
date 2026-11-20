//! Pictures pasted into the input box: saved to the scratch directory, standing in the
//! message as `[Image-0001]` until it is sent.
//!
//! The box holds text and nothing else, so a picture cannot sit in it. What sits there
//! instead is a token the user can see, move and delete like any other word, and the
//! picture waits on disk under the matching name. Only when the message goes does
//! `agent::attach_mentions` swap the token for the picture — so deleting the token before
//! sending is how a paste is taken back, with nothing to keep in sync.
//!
//! The scratch directory is the conversation's, which is the lifetime a paste has: `/clear`
//! takes the files with it, and a `[Image-0001]` recalled from the history after that names
//! nothing and stays text, exactly as an `@path` to a deleted file does.

use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// A token written in the box: the number is the file's, digits as written.
static TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[Image-(\d+)\]").unwrap());

/// …and the file it stands for, which is the same number lower-cased into a file name.
static FILE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^image-(\d+)\.png$").unwrap());

fn file_name(number: &str) -> String {
    format!("image-{}.png", number)
}

/// Write one pasted PNG into `dir` under the next free number, and return the token that
/// stands for it.
///
/// The next number is one past the highest already there rather than a counter kept in
/// memory: the directory is the only thing that knows which numbers are taken, and it is
/// renewed by `/clear`, which is exactly when the numbering should start again.
pub(crate) fn save(dir: &Path, png: &[u8]) -> std::io::Result<String> {
    let highest = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            FILE.captures(&name)?[1].parse::<u64>().ok()
        })
        .max()
        .unwrap_or(0);
    let number = format!("{:04}", highest + 1);
    std::fs::write(dir.join(file_name(&number)), png)?;
    Ok(format!("[Image-{}]", number))
}

/// Every token in `text` whose picture is in `dir`, in the order typed, without repeats.
///
/// A token with no file behind it is not one: it stays in the text as written, which is
/// all it is — someone may well have typed those characters meaning just them.
pub(crate) fn find(text: &str, dir: &Path) -> Vec<(String, PathBuf)> {
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    for captures in TOKEN.captures_iter(text) {
        let token = captures[0].to_string();
        let path = dir.join(file_name(&captures[1]));
        if path.is_file() && !found.iter().any(|(seen, _)| *seen == token) {
            found.push((token, path));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    #[test]
    fn each_paste_takes_the_next_number_after_the_highest() {
        let dir = TempDir::new("pasted-numbers");
        assert_eq!(save(dir.path(), b"one").unwrap(), "[Image-0001]");
        assert_eq!(save(dir.path(), b"two").unwrap(), "[Image-0002]");
        assert_eq!(std::fs::read(dir.join("image-0002.png")).unwrap(), b"two");

        // A gap is not refilled: a token deleted from the box may still be in the history,
        // and handing its number to a different picture would attach the wrong one.
        std::fs::remove_file(dir.join("image-0001.png")).unwrap();
        assert_eq!(save(dir.path(), b"three").unwrap(), "[Image-0003]");

        // Other files in the scratch directory are the model's, and none of them counts.
        std::fs::write(dir.join("image-9999.txt"), "notes").unwrap();
        assert_eq!(save(dir.path(), b"four").unwrap(), "[Image-0004]");
    }

    #[test]
    fn a_token_is_a_picture_only_if_its_file_is_there() {
        let dir = TempDir::new("pasted-find");
        let token = save(dir.path(), b"png").unwrap();

        let text = format!("compare {} with [Image-0042], and {} again", token, token);
        let found = find(&text, dir.path());
        assert_eq!(found.len(), 1, "a missing file or a repeat was attached: {:?}", found);
        assert_eq!(found[0].0, "[Image-0001]");
        assert_eq!(found[0].1, dir.join("image-0001.png"));

        assert!(find("[image-0001] [Image-] [Image-x]", dir.path()).is_empty());
    }
}
