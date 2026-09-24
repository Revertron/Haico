//! Spotting a reply that has started repeating itself, while it is still streaming.
//!
//! The loop detection in `run_turn` watches tool calls, and a model can get stuck without
//! making any. The case that motivated this: a 35B model debugging a hung mio server wrote
//! one reasoning block that restated the same paragraph 126 times, word for word, over
//! thirteen minutes, and nothing in haico noticed — the reply was still arriving, so to
//! the turn loop it was a model thinking hard. With no `max_tokens` set and a 262k window
//! it would have run for hours. Unattended in `--prompt` mode that is the whole run gone.
//!
//! The test is one exact repeat: every `CHECK_EVERY` bytes, the latest `NEEDLE` bytes are
//! looked for anywhere earlier in the last `WINDOW`. A model does not write the same 800
//! bytes twice by accident — a degenerate loop is its own most likely continuation feeding
//! back into itself, so the second copy comes out identical to the first. Waiting for more
//! copies only pays for them. The needle's length is what keeps honest replies clear, and
//! it was chosen by replaying ~1450 real reasoning traces and answers: at 200 and 400 bytes
//! the hits were reasoning that drafted a block of code and later quoted it again; from 600
//! up the only hit left was a trace that really had restated a 1000-character paragraph
//! word for word before wandering off — a loop that happened to end, and one worth cutting
//! short anyway. Newlines play no part, so a loop that never ends a line is caught the same
//! way, and a cycle shorter than the needle is too, since the search allows the earlier
//! copy to overlap the latest one.

use std::fmt;

/// Bytes of the latest text that must already have appeared. Long enough that
/// a quoted snippet cannot trip it; see the module header for how it was chosen.
const NEEDLE: usize = 800;
/// How far back the earlier copy may be, which bounds the longest cycle that
/// can be seen — far past the paragraph-sized loops actually observed.
const WINDOW: usize = 32 * 1024;
/// How much new text arrives between checks. The search is over at most
/// `WINDOW` bytes, so checking every half kilobyte costs nothing next to the
/// time a model takes to produce one.
const CHECK_EVERY: usize = 512;

/// One stream's worth of text — one per kind, since a reasoning trace and the
/// answer after it are different texts and a loop never spans the two.
#[derive(Default)]
pub(crate) struct RepetitionGuard {
    /// The tail of the stream, trimmed back to `WINDOW` now and then rather
    /// than on every push.
    tail: String,
    /// Bytes pushed since the last check.
    unchecked: usize,
}

impl RepetitionGuard {
    /// Takes the next piece of the stream, and says what is repeating once the
    /// latest text is a copy of something already said.
    pub(crate) fn push(&mut self, text: &str) -> Option<Repeating> {
        self.tail.push_str(text);
        self.unchecked += text.len();
        if self.unchecked < CHECK_EVERY {
            return None;
        }
        self.unchecked = 0;

        if self.tail.len() > 2 * WINDOW {
            let cut = floor_boundary(&self.tail, self.tail.len() - WINDOW);
            self.tail.drain(..cut);
        }
        let end = self.tail.len();
        if end <= NEEDLE {
            return None;
        }

        let needle = &self.tail[floor_boundary(&self.tail, end - NEEDLE)..];
        let start = floor_boundary(&self.tail, end.saturating_sub(WINDOW));
        // Everything but the last character, so the needle cannot find itself
        // while an earlier copy may still overlap it: a cycle of 300 bytes puts
        // the previous copy of the latest 800 just 300 bytes back.
        let earlier = &self.tail[start..floor_boundary(&self.tail, end - 1)];
        if !earlier.contains(needle) {
            return None;
        }
        Some(Repeating { passage: sample(needle) })
    }
}

/// A stream cut off because it was going round in a loop. An error type of its
/// own so the turn can tell it apart from a dropped connection: this one is
/// worth another try with a word to the model, a network failure is not.
#[derive(Debug)]
pub(crate) struct Repeating {
    /// A line of what was repeating, for the user and for the nudge — the
    /// model is far more likely to step out of a loop it is shown.
    pub(crate) passage: String,
}

impl fmt::Display for Repeating {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the model got stuck repeating itself: \"{}\"", self.passage)
    }
}

impl std::error::Error for Repeating {}

/// The longest line of the needle, since the needle itself starts wherever the
/// byte count happened to fall — usually mid-word.
fn sample(needle: &str) -> String {
    let line = needle
        .lines()
        .map(str::trim)
        .max_by_key(|l| l.len())
        .unwrap_or("");
    crate::util::truncate_at_boundary(line, 120).to_string()
}

fn floor_boundary(text: &str, mut at: usize) -> usize {
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `text` in pieces the size a server sends them, returning the first
    /// verdict, and how far into the text it came.
    fn feed(text: &str) -> Option<(Repeating, usize)> {
        let mut guard = RepetitionGuard::default();
        let mut fed = 0;
        let chars: Vec<char> = text.chars().collect();
        for piece in chars.chunks(7) {
            let piece: String = piece.iter().collect();
            fed += piece.len();
            if let Some(found) = guard.push(&piece) {
                return Some((found, fed));
            }
        }
        None
    }

    /// Distinct lines, so text built from it never repeats by construction.
    fn honest(lines: usize) -> String {
        honest_from(0, lines)
    }

    fn honest_from(first: usize, lines: usize) -> String {
        (first..first + lines).map(|i| format!("step {i} changes line {} of the parser\n", i * 7)).collect()
    }

    const PARAGRAPH: &str = "So after \"DBG registered conn token 1\", run() did NOT print DBG \
        events NOR loop back. It just... stopped printing. The only explanation: run() \
        returned (thread exited) right there, or panicked silently. If run() had returned \
        Ok or Err right after accept_new, the server thread would exit, main's client read \
        would time out at 3s, and SERVER RUN RESULT would print. We do not see it. So run() \
        did not return either: it is stuck somewhere between the registration and the next \
        print, and join blocks.\n\nThe ONLY code between the registration and the next \
        poll is the insert into the connections map.\n\n";

    #[test]
    fn a_paragraph_restated_once_is_caught_on_the_second_copy() {
        let start = honest(40);
        let (found, at) = feed(&format!("{start}{}", PARAGRAPH.repeat(50))).expect("a loop");
        // Before the third copy has even started.
        assert!(at < start.len() + 2 * PARAGRAPH.len() + CHECK_EVERY, "caught only at {at}");
        assert!(found.passage.len() > 20, "passage: {:?}", found.passage);
    }

    #[test]
    fn a_cycle_shorter_than_the_needle_is_caught() {
        // Overlapping copies: the previous copy of the latest 800 bytes starts
        // only one cycle back.
        let text = format!("Let me think. {}", "and then I check the flag again, ".repeat(400));
        assert!(feed(&text).is_some());
    }

    #[test]
    fn a_loop_after_a_long_honest_start_is_caught() {
        let start = honest(3000);
        let (_, at) = feed(&format!("{start}{}", PARAGRAPH.repeat(10))).expect("a loop");
        assert!(at > start.len());
    }

    #[test]
    fn quoting_the_same_snippet_again_is_not_a_loop() {
        // The false positive that set the needle's length: reasoning drafts code,
        // thinks about it, then quotes it again before writing it.
        let snippet = "```rust\nApp::new()\n    .app_data(state.clone())\n    .route(\"/todos\", \
                       web::get().to(list_todos))\n    .route(\"/todos\", web::post().to(create_todo))\n    \
                       .route(\"/todos/{id}\", web::get().to(get_todo))\n    .route(\"/todos/{id}\", \
                       web::patch().to(update_todo))\n    .route(\"/todos/{id}\", web::delete().to(delete_todo))\n```\n";
        let text = format!(
            "{}Draft:\n{snippet}{}So the final version is:\n{snippet}{}",
            honest_from(0, 20),
            honest_from(100, 30),
            honest_from(200, 20)
        );
        assert!(feed(&text).is_none());
    }

    #[test]
    fn repetitive_but_varying_code_is_not_a_loop() {
        let tests: String = (0..300)
            .map(|i| format!("    #[test]\n    fn case_{i}() {{\n        let mut cache = LruCache::new(2);\n        cache.put({i}, {});\n        assert_eq!(cache.get(&{i}), Some(&{}));\n    }}\n\n", i * 3, i * 3))
            .collect();
        assert!(feed(&tests).is_none());
    }

    #[test]
    fn a_copy_older_than_the_window_is_not_a_loop() {
        let text = format!("{PARAGRAPH}{}{PARAGRAPH}", honest(WINDOW / 30));
        assert!(feed(&text).is_none());
    }

    #[test]
    fn multibyte_text_never_splits_a_character() {
        // Would panic on a slice through a character if the boundaries were wrong.
        let text = "Проверяю флаг ещё раз — ничего не меняется. ".repeat(300);
        assert!(feed(&text).is_some());
    }
}
