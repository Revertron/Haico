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

/// The image formats a vision model is reachable with, as the MIME type that
/// goes in the data URL.
///
/// **Recognized by the bytes on disk, never by the extension.** A `.png` that is
/// really a JPEG is common enough — every screenshot tool that renames rather
/// than converts makes one — and the MIME type in a data URL is not decoration:
/// a server that believes it decodes the wrong thing and fails somewhere far
/// from here. The four are what OpenAI-compatible servers agree on; anything
/// else is left to the text path, which says "not a text file" and is right.
pub(crate) fn image_type(bytes: &[u8]) -> Option<&'static str> {
    // Each of these is the format's own signature, taken from its specification
    // rather than guessed from a sample.
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && bytes[8..12] == *b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// One image, in a form a model server will actually take: its bytes and the
/// MIME type to declare for them.
///
/// **WEBP is re-encoded to PNG, and that is the whole reason this exists.** The
/// two servers haico is pointed at disagree about webp in opposite directions,
/// so there is no label that satisfies both:
///
/// - LM Studio validates the *string* in the data URL and rejects `image/webp`
///   outright, with `'url' field must be a base64 encoded image` — which reads
///   like a corrupt payload and is nothing of the kind. Its decoder sniffs the
///   content, so the same bytes labelled `image/png` are decoded correctly.
/// - llama.cpp is the mirror image. It takes any label and loads through
///   `stb_image`, which has no webp support at all; there is a
///   `decode_webp_with_ffmpeg` path, so a build without ffmpeg cannot read one
///   however it is announced.
///
/// Mislabelling therefore fixes one server and not the other, and lies in the
/// URL besides. Re-encoding is true on both, and the cost is paid once per
/// image on a path that is already about to base64 a few hundred kilobytes.
///
/// PNG, JPEG and GIF are passed through untouched. All three are accepted by
/// every server tested, and re-encoding a JPEG to PNG would multiply its size
/// for nothing.
pub(crate) fn sendable_image(path: &std::path::Path) -> Option<(String, Vec<u8>)> {
    let mime = image_type_of(path)?;
    let bytes = std::fs::read(path).ok()?;

    if mime != "image/webp" {
        return Some((mime.to_string(), bytes));
    }

    // A webp that will not decode is left as a webp rather than dropped: the
    // request then fails the way it did before, which is a worse outcome than
    // this function working but a better one than silently attaching nothing.
    let Ok(decoded) = image::load_from_memory(&bytes) else {
        return Some((mime.to_string(), bytes));
    };
    let mut png = std::io::Cursor::new(Vec::new());
    match decoded.write_to(&mut png, image::ImageFormat::Png) {
        Ok(()) => Some(("image/png".to_string(), png.into_inner())),
        Err(_) => Some((mime.to_string(), bytes)),
    }
}

/// As above, for a file on disk, reading only as much of it as the question
/// needs.
///
/// Twelve bytes rather than the whole file: an image is read whole a moment
/// later anyway, but a text file must not be read twice over just to establish
/// that it is one.
pub(crate) fn image_type_of(path: &std::path::Path) -> Option<&'static str> {
    use std::io::Read;
    let mut header = Vec::with_capacity(12);
    std::fs::File::open(path).ok()?.take(12).read_to_end(&mut header).ok()?;
    image_type(&header)
}

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
    fn an_image_is_recognized_by_its_bytes_and_not_its_name() {
        // The smallest valid file of each kind is not needed — only the
        // signature is read, and reading only the signature is the point.
        assert_eq!(image_type(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(image_type(&[0xFF, 0xD8, 0xFF, 0xE0, 0x00]), Some("image/jpeg"));
        assert_eq!(image_type(b"GIF89a....."), Some("image/gif"));
        assert_eq!(image_type(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));

        // Text, an empty file, and a RIFF container that is a WAV rather than a
        // WEBP — the one case a four-byte check would get wrong.
        assert_eq!(image_type(b"fn main() {}"), None);
        assert_eq!(image_type(b""), None);
        assert_eq!(image_type(b"RIFF\0\0\0\0WAVEfmt "), None);

        // And the name never enters into it: a JPEG called .png is a JPEG,
        // which matters because the MIME type in the data URL is what the
        // server decodes by.
        let dir = TempDir::new("mention-image");
        let lying = dir.join("screenshot.png");
        std::fs::write(&lying, [0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0]).unwrap();
        assert_eq!(image_type_of(&lying), Some("image/jpeg"));

        let text = dir.join("notes.png");
        std::fs::write(&text, "not an image at all").unwrap();
        assert_eq!(image_type_of(&text), None);
    }

    #[test]
    fn a_webp_is_re_encoded_and_the_others_are_left_alone() {
        let dir = TempDir::new("mention-webp");

        // A real one-pixel WEBP, so this exercises the decoder rather than the
        // signature check. Lossless VP8L, which is the smallest valid file.
        let webp = dir.join("tiny.webp");
        let bytes = [
            b'R', b'I', b'F', b'F', 0x1a, 0, 0, 0, b'W', b'E', b'B', b'P', b'V', b'P', b'8', b'L',
            0x0e, 0, 0, 0, 0x2f, 0, 0, 0, 0x10, 0x07, 0x10, 0x11, 0x11, 0x88, 0x88, 0xfe, 0x07,
            0x00,
        ];
        std::fs::write(&webp, bytes).unwrap();

        let (mime, converted) = sendable_image(&webp).expect("a webp should be sendable");
        // The point of the whole function: what goes out is not what is on disk.
        // LM Studio refuses the string "image/webp" and llama.cpp cannot decode
        // the bytes, so neither the label nor the payload may survive.
        assert_eq!(mime, "image/png");
        assert_eq!(image_type(&converted), Some("image/png"));

        // A PNG is passed through byte for byte. Re-encoding one would be work
        // for nothing, and re-encoding a JPEG would multiply its size.
        let png = dir.join("plain.png");
        std::fs::write(&png, &converted).unwrap();
        let (mime, again) = sendable_image(&png).unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(again, converted, "a png was re-encoded when it did not need to be");
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
