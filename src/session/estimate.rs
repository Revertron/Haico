//! What a conversation costs in tokens, estimated.

use serde_json::Value;


// What one image is counted as. A round figure, and honestly so: a vision model
// tiles an image and charges by the tile, so the true number depends on
// dimensions this never decodes, and it lands somewhere in the high hundreds to
// low thousands for anything screenshot-sized. Being roughly right here is the
// entire requirement — what it feeds is a percentage on a status bar and the
// threshold auto mode compacts at.
//
// What it must not be is the alternative. A data URL is four base64 characters
// for every three bytes of image, so counting the message literally makes a
// two-megabyte screenshot read as ~700k tokens: twenty times a 32k window, an
// instant autocompact in auto mode, and compaction cannot shrink an image, so
// the turn would go round that loop rather than answering.
const IMAGE_TOKENS: usize = 1000;

pub(crate) fn estimate_tokens(messages: &[Value]) -> usize {
    messages.iter().map(estimate_message).sum()
}

fn estimate_message(message: &Value) -> usize {
    match message.get("content").and_then(Value::as_array) {
        // The ordinary shape, and every message that carries no image:
        // characters over four across the whole thing, envelope included.
        None => message.to_string().chars().count() / 4,
        // A content-parts array, costed a part at a time so an image's base64 —
        // the bulk of the message and none of its meaning — is replaced rather
        // than counted.
        Some(parts) => parts
            .iter()
            .map(|part| {
                if part["type"] == "image_url" {
                    IMAGE_TOKENS
                } else {
                    part.to_string().chars().count() / 4
                }
            })
            .sum(),
    }
}

// Flatten the conversation for a summarizer, dropping the system prompt (it is re-attached
// unchanged) and trimming tool output, which is the bulk of a long session and the part
// least worth preserving verbatim.

/// What a message says, whatever shape its content is in.
///
/// A content-parts array reads as an empty string through `as_str`, and the
/// caller below drops an empty one — so without this a turn that started with an
/// image summarized as though the user had said nothing at all. The image itself
/// cannot go to a text summarizer, so it leaves a mark saying it was there;
/// "the user asked about this" is the part the summary needs to keep.
pub(crate) fn message_text(message: &Value) -> String {
    match &message["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("text") => part["text"].as_str().unwrap_or("").to_string(),
                Some("image_url") => "[an image]".to_string(),
                _ => String::new(),
            })
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

