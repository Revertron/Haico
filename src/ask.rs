//! The question the model asks the user, and the dialog that answers it.
//!
//! A model that hits a fork it cannot resolve — two reasonable designs, which dependency,
//! what a vague requirement actually meant — has no way to say so mid-turn. What it does
//! instead is decide, doubt the decision, decide again, and burn rounds circling a choice
//! a human could settle in one keypress. `ask_user` is that keypress.
//!
//! The dialog lives here rather than in the tool because the keyboard does: the REPL owns
//! `Keys`, and the tool body runs in both the REPL and an MCP server, where there is
//! nobody on the other end at all.

use crate::input::{echo, read_text, Editor, Input, Keys};
use crate::modes::PlanDecision;
use crate::permissions::PermissionResponse;
use anyhow::{anyhow, Result};
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};

pub(crate) const ASK_USER: &str = "ask_user";

// The most options a question may carry.
//
// Nine rows is what the digit keys can address, and the typed answer takes one of them.
// The ceiling is not really about the keyboard though: a question with a dozen answers is
// one the user has to study rather than answer, which is the opposite of the point.
pub(crate) const MAX_OPTIONS: usize = 8;

// The row that is always there, whatever the model offered. Nothing the model can think of
// covers every case, and a question that only offers its own guesses is a leading question.
const OWN_ANSWER: &str = "Something else — type my own answer";

// What came back from the dialog.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Answer {
    // One of the offered options, by index.
    Chose(usize),
    Typed(String),
    // Escaped out, or typed nothing. The model is told to decide for itself rather than
    // ask again — a question nobody wants to answer must not become a loop.
    Dismissed,
}

// Check a question over before it reaches the screen, and tidy what survives.
//
// Shared with the tool body so the model gets the same complaint whichever path it took.

pub(crate) fn normalize(
    question: &str,
    options: &[String],
) -> std::result::Result<(String, Vec<String>), String> {
    let question = question.trim();
    if question.is_empty() {
        return Err("The question was empty. Put the question itself in the 'question' \
                    field, then call ask_user again."
            .to_string());
    }

    // Blanks and repeats are dropped rather than refused: they are a slip in how the list
    // was written, not a sign the question is a bad one.
    let mut cleaned: Vec<String> = Vec::new();
    for option in options {
        let option = option.trim();
        if option.is_empty() || cleaned.iter().any(|kept| kept == option) {
            continue;
        }
        cleaned.push(option.to_string());
    }

    if cleaned.len() < 2 {
        return Err(format!(
            "A question needs at least two options to choose between; this one had {}. \
             Offer the alternatives you are actually weighing. If there is nothing to \
             choose between, do not call ask_user — end your turn with the question \
             instead.",
            cleaned.len()
        ));
    }

    if cleaned.len() > MAX_OPTIONS {
        return Err(format!(
            "{} options is more than anyone can weigh in one read. Narrow it to the {} \
             that actually matter — the user can always type an answer of their own.",
            cleaned.len(),
            MAX_OPTIONS
        ));
    }

    Ok((question.to_string(), cleaned))
}

// What the model is told, once the user has answered.
//
// Every wording here ends the question rather than inviting another one. An answer is a
// decision the user has now made, and a dismissal is permission to make it without them.

pub(crate) fn reply_for(answer: &Answer, options: &[String]) -> String {
    match answer {
        Answer::Chose(index) => format!(
            "The user chose option {}: \"{}\". That is settled — build it that way and do \
             not ask about this again.",
            index + 1,
            options[*index]
        ),
        Answer::Typed(text) => format!(
            "The user answered in their own words: \"{}\". That is settled — follow it and \
             do not ask about this again.",
            text
        ),
        Answer::Dismissed => "The user did not answer. Decide it yourself, say in one line \
                              which way you went and why, and carry on — do not ask again."
            .to_string(),
    }
}

// Wrap plain text into physical rows.
//
// The region's rows are physical: whatever draws them owes the writer rows that fit,
// because a row the terminal wrapped on its own is a row `out` cannot count back over when
// it erases. Options are whole sentences here — the model is told to write them as long as
// they need to be — so this breaks on spaces, and mid-word only when a single word is
// wider than the terminal.

fn wrap(text: &str, width: usize, first: &str, rest: &str) -> Vec<String> {
    let mut rows = Vec::new();
    let mut line = first.to_string();
    // Whether the row so far is nothing but its prefix, which is what decides whether the
    // next word needs a space in front of it.
    let mut bare = true;

    for word in text.split_whitespace() {
        let mut word = word;
        loop {
            let space = usize::from(!bare);
            let room = width.saturating_sub(line.chars().count() + space);

            if word.chars().count() <= room {
                if !bare {
                    line.push(' ');
                }
                line.push_str(word);
                bare = false;
                break;
            }

            // It does not fit. Move to the next row — unless this row is already bare, in
            // which case the word is wider than the row itself and has to be cut. The
            // alternative is a row that wraps itself, and the region's height is then
            // wrong for as long as the dialog is up.
            if !bare {
                rows.push(std::mem::replace(&mut line, rest.to_string()));
                bare = true;
                continue;
            }

            let room = width.saturating_sub(line.chars().count()).max(1);
            let taken: String = word.chars().take(room).collect();
            word = &word[taken.len()..];
            line.push_str(&taken);
            rows.push(std::mem::replace(&mut line, rest.to_string()));
        }
    }

    rows.push(line);
    rows
}

// The dialog as physical rows, with the row the cursor belongs on.
//
// Pure, so the layout can be tested without a terminal: the caller hands it to the writer.

pub(crate) fn dialog_rows(
    options: &[String],
    selected: usize,
    width: usize,
) -> (Vec<String>, usize) {
    // A column short of the width, for the same reason the status bars are: the last cell
    // is where a terminal decides to wrap on its own.
    let width = width.saturating_sub(1).max(20);

    let mut rows: Vec<String> = Vec::new();
    let mut caret = 0;

    // The row the model never wrote comes last, so the numbering of what it did offer is
    // the numbering it gave them.
    let items = options.iter().map(String::as_str).chain([OWN_ANSWER]);

    for (index, option) in items.enumerate() {
        let chosen = index == selected;
        if chosen {
            caret = rows.len();
        }
        let marker = if chosen { "❯ " } else { "  " };
        let first = format!("{}{}. ", marker, index + 1);
        // Continuation lines line up under the text, not under the number.
        let rest = " ".repeat(first.chars().count());

        for row in wrap(option, width, &first, &rest) {
            rows.push(match chosen {
                true => format!("{}{}{}", SetForegroundColor(Color::Cyan), row, ResetColor),
                false => row,
            });
        }
    }

    // Wrapped like the options are, because on a narrow terminal the hint is the longest
    // line on the screen and a row that wraps itself is one the writer cannot erase.
    let hint = format!(
        "↑/↓ and Enter to choose, 1–{} to answer straight away, Esc to skip",
        options.len() + 1
    );
    for row in wrap(&hint, width, "  ", "  ") {
        rows.push(format!(
            "{}{}{}",
            SetForegroundColor(crate::out::DIM),
            row,
            ResetColor
        ));
    }

    (rows, caret)
}

// Put the question on the screen and wait for an answer.
//
// The options are drawn into the bottom region rather than printed, so moving the
// selection redraws in place instead of scrolling a new copy past every keypress. The
// question itself is printed above it and stays in the scrollback, where it reads as part
// of the conversation; the list comes off once it has been answered, and the answer takes
// its place.

pub(crate) async fn ask_question(
    keys: &mut Keys,
    question: &str,
    options: &[String],
) -> Result<Answer> {
    outln!(
        "\n{}? {}{}",
        SetAttribute(Attribute::Bold),
        question,
        SetAttribute(Attribute::Reset)
    );

    // The typed answer is the last row, one past the offered options.
    let typed_row = options.len();
    let mut selected = 0usize;

    let answer = loop {
        let width = crate::out::terminal_width().max(20);
        let (rows, caret) = dialog_rows(options, selected, width);
        crate::out::set_region(rows, caret, 0);

        let Some(event) = keys.next().await else {
            crate::out::clear_region();
            return Err(anyhow!("no interactive terminal to ask the user"));
        };
        let Event::Key(key) = event else {
            // A resize or a paste: redrawn at the top of the loop, at the new width.
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Up => selected = selected.saturating_sub(1),
            KeyCode::Down => selected = (selected + 1).min(typed_row),
            // A digit answers outright, the way the permission prompt's digits do. Enter
            // is for whoever would rather look before committing.
            KeyCode::Char(digit @ '1'..='9') => {
                let index = digit as usize - '1' as usize;
                if index <= typed_row {
                    break Some(index);
                }
            }
            KeyCode::Enter => break Some(selected),
            KeyCode::Esc => break None,
            // Ctrl+C abandons a half-typed line at the prompt; here there is nothing to
            // abandon but the question, so it means the same as Esc.
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break None,
            _ => {}
        }
    };

    crate::out::clear_region();
    // Whatever was typed while the question stood was not an answer to it, and must not
    // fall into the editor below either.
    keys.drain();

    let Some(answer) = answer else {
        outln!("  ✗ skipped\n");
        return Ok(Answer::Dismissed);
    };

    if answer < typed_row {
        outln!("  ✓ {}\n", options[answer]);
        return Ok(Answer::Chose(answer));
    }

    const PROMPT: &str = "  ✎ ";
    let mut editor = Editor::new(Vec::new());
    let typed = match read_text(keys, &mut editor, PROMPT).await {
        Input::Text(text) => text,
        // Backing out of the answer is not an answer, so it counts as skipping the
        // question rather than as an empty reply.
        _ => String::new(),
    };
    keys.drain();

    let typed = typed.trim().to_string();
    if typed.is_empty() {
        outln!("  ✗ skipped\n");
        return Ok(Answer::Dismissed);
    }

    outln!("{}\n", echo(PROMPT, &typed));
    Ok(Answer::Typed(typed))
}

// The permission prompt: the call described, its diff when it writes a file, and three
// answers. Here beside the question dialog for the question dialog's reason — it reads the
// keyboard, which the gate in `permissions.rs` must not.
pub(crate) async fn ask_permission(
    keys: &mut Keys,
    allow_all_label: &str,
    description: &str,
    diff: Option<&str>,
) -> Result<PermissionResponse> {
    outln!("⚑ Allow this tool usage?");
    outln!("{}\n", description);

    if let Some(diff_content) = diff {
        outln!("{}", diff_content);
    }

    outln!("1. Allow once");
    outln!("2. {}", allow_all_label);
    outln!("3. Reject and provide alternative");
    crate::out::flush();

    let response = loop {
        let Some(event) = keys.next().await else {
            // The keyboard reader is gone, which is not consent.
            return Err(anyhow!("no interactive terminal to ask for permission"));
        };

        // Only a closed channel means nobody is there to answer. Everything else that can
        // arrive here is not an answer but is not an absent user either, and matching the
        // key event directly above used to lump the two together: on Windows the console
        // emits a focus event whenever the window is clicked away from or minimized, so
        // stepping away from the keyboard refused the pending call - and every call after
        // it - as if the terminal had gone.
        let Event::Key(key_event) = event else {
            continue;
        };

        // Only process key press events, ignore key release
        if key_event.kind != KeyEventKind::Press {
            continue;
        }

        match key_event.code {
            KeyCode::Char('1') => {
                break PermissionResponse::Allow;
            }
            KeyCode::Char('2') => {
                break PermissionResponse::AllowAll;
            }
            KeyCode::Char('3') => {
                outln!("3\n");

                let mut editor = Editor::new(Vec::new());
                let alternative = match read_text(keys, &mut editor, "> ").await {
                    Input::Text(text) => text,
                    // Backing out of the question is not an answer to it, so it is the
                    // same as rejecting with nothing to say.
                    _ => String::new(),
                };
                outln!("{}", echo("> ", &alternative));

                break PermissionResponse::Reject(alternative.trim().to_string());
            }
            _ => {
                // Invalid key, continue waiting
                continue;
            }
        }
    };

    // Anything typed while the question was on screen was not meant for whatever comes
    // after it.
    keys.drain();

    Ok(response)
}

// Show a plan and ask what to do with it. Here for the permission prompt's reason, and laid out
// like it.

pub(crate) async fn review_plan(keys: &mut Keys, plan: &str) -> Result<PlanDecision> {
    outln!("\n≡ Proposed plan\n");
    // Rendered rather than printed raw: plans come back as markdown, and an unrendered
    // one is a wall of asterisks exactly when it most needs reading.
    out!("{}", crate::render::render_markdown(plan));
    outln!();
    // Same shape as the permission prompt: the approvals first, the refusal last, so the
    // key that means "no" is in the same place in both.
    outln!("1. Approve and start work");
    outln!("2. Approve and enable auto mode (routine calls run without asking)");
    outln!("3. Send back corrections");
    crate::out::flush();

    let decision = loop {
        let Some(event) = keys.next().await else {
            return Err(anyhow!("no interactive terminal to review the plan"));
        };
        // A focus change or a resize is not a decision, but it is not a missing terminal
        // either; see the permission prompt for the failure that motivated the split.
        let Event::Key(key) = event else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match key.code {
            KeyCode::Char('1') => break PlanDecision::Approve,
            KeyCode::Char('2') => break PlanDecision::ApproveWithAuto,
            KeyCode::Char('3') => {
                outln!("3\n");
                const PROMPT: &str = "What should change? > ";
                let mut editor = Editor::new(Vec::new());
                let feedback = match read_text(keys, &mut editor, PROMPT).await {
                    Input::Text(text) => text,
                    // Backing out is not corrections; the plan stands unapproved and the
                    // model is told only that.
                    _ => String::new(),
                };
                outln!("{}", echo(PROMPT, &feedback));
                break PlanDecision::Revise(feedback.trim().to_string());
            }
            _ => continue,
        }
    };

    keys.drain();

    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::out::visible_width;

    fn options(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_question_is_checked_over_before_it_reaches_the_screen() {
        // The ordinary case: trimmed, and handed back in the order it was offered.
        let (question, offered) = normalize(
            "  Which HTTP client?  ",
            &options(&["reqwest", "  ureq  ", "hyper"]),
        )
        .unwrap();
        assert_eq!(question, "Which HTTP client?");
        assert_eq!(offered, options(&["reqwest", "ureq", "hyper"]));

        // Blanks and repeats are a slip in the list, not a bad question: dropped quietly.
        let (_, offered) = normalize("Which?", &options(&["a", "", "  ", "a", "b"])).unwrap();
        assert_eq!(offered, options(&["a", "b"]));

        // A question with nothing to choose between is a sentence, not a dialog — and the
        // model is told what to do with it instead.
        let refused = normalize("Which?", &options(&["only one"])).unwrap_err();
        assert!(refused.contains("at least two"), "got: {}", refused);
        assert!(refused.contains("end your turn"), "got: {}", refused);

        // Dropping duplicates can be what takes a list under two.
        assert!(normalize("Which?", &options(&["a", "a"])).is_err());

        let refused = normalize("  ", &options(&["a", "b"])).unwrap_err();
        assert!(refused.contains("empty"), "got: {}", refused);

        let many: Vec<String> = (0..MAX_OPTIONS + 1).map(|i| format!("option {}", i)).collect();
        let refused = normalize("Which?", &many).unwrap_err();
        assert!(refused.contains("Narrow it"), "got: {}", refused);
        // The last one that still fits does not.
        assert!(normalize("Which?", &many[..MAX_OPTIONS]).is_ok());
    }

    #[test]
    fn every_answer_ends_the_question_rather_than_inviting_another() {
        let offered = options(&["reqwest", "ureq"]);

        let reply = reply_for(&Answer::Chose(1), &offered);
        assert!(reply.contains("ureq"), "got: {}", reply);
        assert!(reply.contains("option 2"), "got: {}", reply);
        assert!(reply.contains("do not ask about this again"), "got: {}", reply);

        let reply = reply_for(&Answer::Typed("use ureq, but wrap it".to_string()), &offered);
        assert!(reply.contains("use ureq, but wrap it"), "got: {}", reply);
        assert!(reply.contains("do not ask about this again"), "got: {}", reply);

        // Skipping the question hands the decision back, rather than leaving the model
        // with nothing to do but ask it again.
        let reply = reply_for(&Answer::Dismissed, &offered);
        assert!(reply.contains("Decide it yourself"), "got: {}", reply);
        assert!(reply.contains("do not ask again"), "got: {}", reply);
    }

    #[test]
    fn rows_fit_the_terminal_however_long_an_option_is() {
        // The model is told to write options as long as they need to be, and the region's
        // rows are physical: one that wraps itself is one the writer cannot erase.
        const WIDTH: usize = 40;
        let offered = options(&[
            "Use reqwest, which is already a dependency and brings tokio along with it",
            "supercalifragilisticexpialidociousantidisestablishmentarianism",
        ]);

        let (rows, _) = dialog_rows(&offered, 0, WIDTH);
        for row in &rows {
            assert!(
                visible_width(row) < WIDTH,
                "{:?} is {} wide",
                row,
                visible_width(row)
            );
        }

        // Nothing is lost to the wrapping: the words come back in order.
        let text: String = rows
            .iter()
            .map(|row| strip_escapes(row))
            .collect::<Vec<_>>()
            .join(" ");
        let words: Vec<&str> = text.split_whitespace().collect();
        assert!(
            words.windows(3).any(|w| w == ["already", "a", "dependency"]),
            "got: {}",
            text
        );
        assert!(text.contains("supercalifragilistic"), "got: {}", text);
    }

    #[test]
    fn the_typed_answer_is_always_the_last_row_and_the_caret_follows_the_selection() {
        let offered = options(&["a", "b", "c"]);

        let (rows, caret) = dialog_rows(&offered, 0, 80);
        assert_eq!(caret, 0);
        // Three options, the typed answer, and the hint line.
        assert_eq!(rows.len(), 5);
        assert!(strip_escapes(&rows[3]).contains("4. Something else"), "got: {:?}", rows);
        assert!(strip_escapes(&rows[4]).contains("Esc to skip"), "got: {:?}", rows);

        // The marker and the caret both move with the selection.
        let (rows, caret) = dialog_rows(&offered, 2, 80);
        assert_eq!(caret, 2);
        assert!(strip_escapes(&rows[2]).starts_with("❯ 3."), "got: {:?}", rows);
        assert!(strip_escapes(&rows[0]).starts_with("  1."), "got: {:?}", rows);

        // Which includes the row the model never wrote.
        let (rows, caret) = dialog_rows(&offered, 3, 80);
        assert_eq!(caret, 3);
        assert!(strip_escapes(&rows[3]).starts_with("❯ 4."), "got: {:?}", rows);
    }

    fn strip_escapes(text: &str) -> String {
        let mut plain = String::new();
        let mut chars = text.chars();
        while let Some(ch) = chars.next() {
            if ch != '\u{1b}' {
                plain.push(ch);
                continue;
            }
            if chars.next() == Some('[') {
                for ch in chars.by_ref() {
                    if ('@'..='~').contains(&ch) {
                        break;
                    }
                }
            }
        }
        plain
    }
}
