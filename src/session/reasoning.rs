//! The reasoning policy: what happens to a reply's chain of thought before it joins the
//! history — kept, compressed or dropped.

use crate::trace::Caller;
use crate::config::Reasoning;
use crate::llm::LMStudioClient;
use serde_json::{json, Value};
use std::collections::HashSet;
// The key `into_message` writes a chain of thought under, and the one the thinking-model
// chat templates read back. Only the one spelling is written here even though the stream
// may have arrived under either, because the assembler has already normalized it.

pub(super) const REASONING_KEY: &str = "reasoning_content";

// What the compressor is told to produce.
//
// The voice matters more than it looks. With reasoning preserved, the template feeds this
// field back to the model as its own thinking, so a summary written as minutes about a
// thought ("The model considered X, then rejected it") is further out of distribution than
// the long thought it replaces. Asking for first-person compressed thinking keeps the
// replacement the same kind of text as the original.
//
// Asking for the first person was not enough on its own, which is what the rest of the
// wording is for. A compressor handed a trace as a user turn reads it as a question and
// answers it, and one came back in the first person of an *advisor*: "My intent: steer them
// to capture just owned locals", then a "Concrete fix:" and a fenced Rust block. Stored as
// the reply's own chain of thought, that left the history carrying a thought in which "I"
// was somebody else and the model itself was "them". The round after it did not continue
// the work — it continued the voice, and its thinking became commentary on whether there
// had been any previous thinking at all to quote. So the trace is named as the model's own
// before anything is asked of it, answering and advising are refused outright, and the
// pronouns that carried the other voice are named rather than left to be inferred from
// "first person".

//
// The trace goes inside `<trace>` tags, and the prompt names them, because a bare trace as the
// whole user turn was read as part of the instruction instead. At temperature 0 and on a trace
// no different from the ones before it, the compressor concluded "there's no actual chain of
// thought provided ... the entire content IS the instruction itself" and refused: "There's no
// prior chain of thought here to condense". Twice in 53 compressions, in two separate runs, and
// the refusal was stored as the model's thinking both times. The tags leave nothing to decide
// about where the instruction ends and the text begins.

const COMPRESS_PROMPT: &str = "The text inside <trace> tags in the next message is your own \
     chain of thought from a moment ago — not a question, not a request, and nobody is waiting \
     on an answer to it. \
     Rewrite it shorter so you can carry on from it without re-reading it. Do not solve the \
     problem it is about, do not propose a fix, do not address anyone. Keep only: approaches \
     you tried or considered and then rejected, and why; facts you established along the \
     way; and the intent you ended on. Drop restatements of the question, anything already \
     visible in the conversation, and any working you would reach the same way again. Write \
     it as thinking rather than as a report about thinking: first person, the way the trace \
     itself is written — \"I\" is you, and there is no \"they\", no \"the model\" and no \
     \"the user\". No preamble, no headings, no fenced code, no more than a short paragraph.";

// Uncapped: the compressor is the same model, so it thinks before it summarizes, and a cap
// that lands mid-thought would leave nothing but an unfinished scratchpad to store. Letting
// it run to its own stop is what `chat_completion_plain`'s finish_reason check is for — a
// reply that still hit "length" (the model's own context limit, not an artificial one here)
// is treated as a failure rather than trusted as a summary.

// The chain of thought on an assistant message, if it has one worth acting on.

pub(crate) fn reasoning_text(message: &Value) -> Option<&str> {
    message[REASONING_KEY]
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

// A reasoning model asked to summarize thinks first, and its scratchpad arrives ahead of
// the answer in the same field. Kept out of the stored summary for the obvious reason —
// storing the thinking about the compression instead of the compression would defeat the
// whole exercise.

pub(crate) fn without_think_block(text: &str) -> &str {
    match text.find("</think>") {
        Some(end) => text[end + "</think>".len()..].trim(),
        None => text.trim(),
    }
}

// What the policy did to one reply, in numbers rather than in prose.
//
// Returned instead of a formatted line because two callers want different things from it:
// the transcript wants a sentence, and the session totals want the figures. Formatting it
// here would mean the totals had to parse their own output back.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReasoningOutcome {
    Dropped { tokens: usize },
    Compressed { before: usize, after: usize },
}

// What compression has bought on the conversation so far, for /context.
//
// Cleared by /clear with the history it describes, and only by that: /compact and /model
// both keep the conversation, so the savings made on it still stand. Kept out here rather
// than derived from `messages` because it cannot be derived — once a trace is replaced, the
// history holds the compressed version and no record of what it cost before.

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReasoningStats {
    pub(crate) turns: usize,
    pub(crate) before: usize,
    pub(crate) after: usize,
}

impl ReasoningStats {
    pub(crate) fn record(&mut self, outcome: ReasoningOutcome) {
        if let ReasoningOutcome::Compressed { before, after } = outcome {
            self.turns += 1;
            self.before += before;
            self.after += after;
        }
    }

    pub(crate) fn saved(&self) -> usize {
        self.before.saturating_sub(self.after)
    }
}

// Whether applying the policy will cost a request to the server.
//
// The activity bar is drawn before the work starts, so the caller has to know this without
// doing the work first. Dropping a field and declining a trace too short to be worth
// compressing are both over before a frame could be painted; announcing either as work in
// progress would report something that is not happening.

pub(crate) fn reasoning_needs_a_request(
    message: &Value,
    policy: Reasoning,
    threshold: usize,
) -> bool {
    policy == Reasoning::Compress
        && reasoning_text(message).is_some_and(|text| estimate_reasoning_tokens(text) >= threshold)
}

// Apply the session's reasoning policy to the reply that just finished.
//
// Called on the tail of the conversation and only there, which is what makes this safe to
// do every turn. The KV cache on the server is a prefix cache: rewriting the message that
// was just generated invalidates from that message forward, and nothing is in front of it
// yet, so the re-prefill is this reply and no more. Rewriting anything older would
// invalidate every turn after it and cost far more than the tokens it saved.
//
// That same property is what stops a trace being compressed twice. Once this returns, the
// message moves into the history and is never revisited, so there is no state to track and
// no marker to write into the message — a compressed trace is simply one that is already
// behind the write point.
//
// A failure here keeps the original. Compression is an optimization, and losing a chain of
// thought to a summarizer that timed out would be a worse outcome than the tokens it was
// trying to save.

pub(crate) async fn apply_reasoning(
    client: &LMStudioClient,
    message: &mut Value,
    policy: Reasoning,
    threshold: usize,
) -> Option<ReasoningOutcome> {
    let Some(original) = reasoning_text(message) else {
        return None;
    };

    match policy {
        Reasoning::Keep => None,

        Reasoning::Drop => {
            let tokens = estimate_reasoning_tokens(original);
            if let Some(object) = message.as_object_mut() {
                object.remove(REASONING_KEY);
            }
            Some(ReasoningOutcome::Dropped { tokens })
        }

        Reasoning::Compress => {
            let before = estimate_reasoning_tokens(original);
            if before < threshold {
                return None;
            }

            let Some(summary) = compress_trace(client, original).await else {
                return None;
            };
            let after = estimate_reasoning_tokens(&summary);
            message[REASONING_KEY] = json!(summary);
            Some(ReasoningOutcome::Compressed { before, after })
        }
    }
}

// One chain of thought shortened by the compressor, or `None` when the compressor failed or
// gave back something that is not a shorter summary of it. Shared by the reasoning policy
// and by compaction, which uses it on the largest traces when a transcript is over budget.
pub(crate) async fn compress_trace(client: &LMStudioClient, original: &str) -> Option<String> {
    let before = estimate_reasoning_tokens(original);
    // A sub-agent copy, like the reviewer and research: the main conversation's
    // thinking level is no more relevant to summarizing a trace than its temperature
    // is. It is not asked to skip thinking, though — that was tried, and a compressor
    // answering cold reads the trace it was handed as a question and answers it,
    // which is the adviser voice the prompt above now argues against.
    let summary = match client
        .for_sub_agent()
        .chat_completion_plain(
            Caller::Compressor,
            vec![
                json!({"role": "system", "content": COMPRESS_PROMPT}),
                json!({"role": "user", "content": format!(
                    "The chain of thought to shorten:\n\n<trace>\n{}\n</trace>",
                    original
                )}),
            ],
            None,
        )
        .await
    {
        Ok(reply) => without_think_block(&reply).to_string(),
        Err(e) => {
            crate::debug::log("reasoning compression failed", &format!("{:#}", e));
            return None;
        }
    };

    // A summary that came back empty, or that is no shorter than what it replaces,
    // is not worth the substitution: the original is at least the model's own words
    // in its own voice, and a compressor that expanded the trace has misunderstood
    // the job. Both happen with small models often enough to be worth the guard.
    let after = estimate_reasoning_tokens(&summary);
    if summary.is_empty() || after >= before {
        return None;
    }
    if !reads_as_a_summary(original, &summary) {
        crate::debug::log("reasoning compression rejected, not a summary of the trace", &summary);
        return None;
    }

    Some(summary)
}

// Whether a compressor's reply is a summary of the trace rather than something else — a
// refusal, an answer to a question the trace was mistaken for, a reply to somebody.
//
// Judged by vocabulary: a summary is made of the trace's own words, so nearly all of its
// longer words appear in the trace, while a reply about the request is made of words about
// requests. Measured over every compression recorded so far (53, several runs): real summaries
// share at least 63% of their distinct words of four or more characters with their trace, the
// two refusals 34% and 42%. Identifiers need no special treatment — `bound_addr` or `E0308`
// in a summary are exactly the words that come from the trace. A reply too short to carry
// evidence either way is let through; the length guard above has already seen it.

const MIN_SHARED_WORDS: f64 = 0.5;
const MIN_WORDS_TO_JUDGE: usize = 8;

pub(super) fn reads_as_a_summary(original: &str, summary: &str) -> bool {
    fn words(text: &str) -> HashSet<String> {
        text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|w| w.chars().count() >= 4)
            .map(str::to_lowercase)
            .collect()
    }
    let summary_words = words(summary);
    if summary_words.len() < MIN_WORDS_TO_JUDGE {
        return true;
    }
    let original_words = words(original);
    let shared = summary_words.iter().filter(|w| original_words.contains(*w)).count();
    shared as f64 / summary_words.len() as f64 >= MIN_SHARED_WORDS
}

// The same chars/4 estimate `estimate_tokens` uses, applied to one string rather than to a
// whole conversation. Rough, and deliberately the same roughness: a threshold tuned against
// one number and reported against another would be confusing to reason about.

pub(crate) fn estimate_reasoning_tokens(text: &str) -> usize {
    text.chars().count() / 4
}


