//! Turning [`AgentEvent`]s into terminal output. The other half of `src/turn.rs`.
//!
//! Everything the turn loop used to know about escapes, folds, previews and
//! wording now lives here, which is the point: `render.rs` is called from this
//! side of the seam and never from the core, so terminal knowledge cannot follow
//! the loop when the loop moves.

use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};

use crate::input::{echo, PROMPT};
use crate::render::{change_preview, tool_call_line, tool_result_line, MarkdownStream};
use crate::turn::{AgentEvent, Notice};

/// Renders events, and owns the state rendering needs: the markdown stream, and
/// whether anything has been put on screen for the reply in progress.
///
/// Those three were locals in the turn loop, and they are a large part of why it
/// could not be moved: `markdown.finish()` has to be called by whoever called
/// `markdown.push()`, and a fold opened for a chain of thought has to be closed
/// by whoever opened it. They are display state, so they belong with the display.
pub(crate) struct Presenter {
    markdown: MarkdownStream,
    /// Assistant text has already been printed as it arrived, so nothing
    /// re-prints it from the finished message.
    streamed_any: bool,
    /// The same for the chain of thought. Tracked separately because it is not
    /// markdown, does not go through the stream, and can be the *whole* reply —
    /// a model that only ever thought leaves a fold open with no content behind it.
    streamed_thinking: bool,
}

impl Presenter {
    pub(crate) fn new() -> Self {
        Presenter {
            markdown: MarkdownStream::new(),
            streamed_any: false,
            streamed_thinking: false,
        }
    }

    /// Ready for the next reply. Per reply, not per turn: a turn is many rounds
    /// and each round streams one.
    pub(crate) fn start_reply(&mut self) {
        self.markdown = MarkdownStream::new();
        self.streamed_any = false;
        self.streamed_thinking = false;
    }

    /// Whether anything at all has reached the screen for this reply.
    fn showed_anything(&self) -> bool {
        self.streamed_any || self.streamed_thinking
    }

    pub(crate) fn show(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::ReasoningChunk(text) => {
                if !self.streamed_thinking {
                    self.streamed_thinking = true;
                    // Folded from the start rather than after the fact: the
                    // reasoning can run for pages, and the activity bar is
                    // already counting the characters as they arrive, so nothing
                    // about the wait goes unreported while it is closed. Ctrl+O
                    // opens it, and it stays open.
                    crate::out::begin_fold(&format!(
                        "{}{}○ (thinking){}",
                        SetAttribute(Attribute::Italic),
                        SetForegroundColor(Color::DarkGrey),
                        ResetColor
                    ));
                    out!(
                        "{}{}○ {}",
                        SetAttribute(Attribute::Italic),
                        SetForegroundColor(Color::DarkGrey),
                        ResetColor
                    );
                }
                // Italic as well as dim: a tool call and its result are dim too,
                // and until this the three were the same grey. A terminal that
                // cannot do italic is no worse off than before — conhost drops
                // the attribute and prints the dim text it printed already.
                //
                // Painted per piece rather than left switched on across the
                // stream: an unfinished line is redrawn above the input box every
                // time the box moves, and a color left open at the end of it
                // would bleed into the box itself.
                out!(
                    "{}{}{}{}",
                    SetAttribute(Attribute::Italic),
                    SetForegroundColor(Color::DarkGrey),
                    text,
                    ResetColor
                );
                crate::out::flush();
            }
            AgentEvent::ReplyChunk(text) => {
                if !self.streamed_any {
                    self.streamed_any = true;
                    // The answer starts on a row of its own so it is not mistaken
                    // for the tail of the thinking above it. The reasoning is over
                    // the moment the answer begins, so its fold closes here and
                    // the answer is not swept into it.
                    if self.streamed_thinking {
                        crate::out::end_fold();
                        crate::out::ensure_line_start();
                        outln!();
                    }
                    out!("● ");
                }
                out!("{}", self.markdown.push(&text));
                crate::out::flush();
            }
            AgentEvent::ReplyEnd => {
                // Unconditional: a reply that was nothing but reasoning ends with
                // its fold still open, and every path out of a reply sends this.
                crate::out::end_fold();
                if self.streamed_any {
                    out!("{}", self.markdown.finish(false));
                }
                if self.showed_anything() {
                    crate::out::ensure_line_start();
                }
            }
            AgentEvent::ToolStarted { name, arguments, auto_note } => {
                outln!("{}", tool_call_line(&name, &arguments, auto_note.as_deref()));
            }
            AgentEvent::ToolDiff(diff) => {
                // A few lines of the change, with the whole diff behind them —
                // the terminal's decision, not the turn's. It has a height, and
                // the fold is what it has instead of scrollback.
                if let Some(preview) = change_preview(&diff) {
                    crate::out::foldable(preview.trim_end_matches('\n'), &diff);
                }
            }
            AgentEvent::ToolResult(result) => {
                // The summary is the row that used to be all there was: the first
                // line, cut to fit. The result itself is kept behind it now rather
                // than thrown away, and Ctrl+O opens it.
                if let Some(line) = tool_result_line(&result) {
                    crate::out::foldable(&line, &result);
                }
            }
            AgentEvent::Finished { verb, elapsed } => {
                // Dimmed for the same reason the chain of thought is: it is not
                // the answer.
                outln!(
                    "\n{}{}{}\n",
                    SetForegroundColor(Color::DarkGrey),
                    crate::turn_summary(verb, elapsed, chrono::Local::now()),
                    ResetColor
                );
            }
            AgentEvent::Notice(notice) => self.notice(notice),
        }
    }

    fn notice(&mut self, notice: Notice) {
        match notice {
            Notice::Interrupted => outln!("\n[interrupted]\n"),
            Notice::RequestFailed(e) => {
                // A reply that got part-way out needs a blank row between it and
                // the error, or the two run together. Decided here because it is
                // a question about what is already on screen, which is the one
                // thing the turn has no way of knowing.
                if self.showed_anything() {
                    outln!();
                }
                outln!("\n[error] {}", e);
                outln!("Conversation kept. Type a message to retry.");
            }
            Notice::RoundCapReached(cap) => {
                outln!("\n[stopped after {} tool-call rounds without a final answer]", cap)
            }
            // Wearing the chrome's light blue, tinted rather than painted: the
            // bars are the only thing on screen with a background, and that is
            // what makes them read as furniture rather than as output.
            Notice::Compacted(report) => outln!(
                "{}{}{}",
                SetForegroundColor(crate::input::BAR_BACKGROUND),
                report,
                ResetColor
            ),
            Notice::CompactionStillLarge(percent) => outln!(
                "[the summary is still over {}% of the window; carrying on without compacting again]",
                percent
            ),
            Notice::CompactionFailed(e) => outln!("\n[error] could not compact: {}\n", e),
            // Shown rather than folded: the whole point of compressing is to find
            // out whether it is paying for itself, which cannot be judged from a
            // line hidden by default.
            Notice::ReasoningCompressed { before, after } => outln!(
                "{}Compressed reasoning from {} to {} tokens.{}",
                SetForegroundColor(crate::input::BAR_BACKGROUND),
                before,
                after,
                ResetColor
            ),
            // Folded, because dropping says nothing the policy did not already
            // say: it removes every trace, every turn, so a visible line per
            // reply would be a line per reply saying the same thing.
            Notice::ReasoningDropped { tokens } => {
                let line = format!("[reasoning dropped (~{} tokens)]", tokens);
                crate::out::foldable(&line, &line);
            }
            Notice::ToolCallsMalformed(shown) => {
                outln!("[ERROR] tool_calls is not an array: {}", shown)
            }
            Notice::ToolCallUnnamed(detail) => outln!("[ERROR] {}", detail),
            Notice::ToolArgumentsInvalid { tool, detail } => {
                outln!("[ERROR] {}: {}", tool, detail)
            }
            Notice::ToolFailed(e) => outln!("[ERROR] Tool execution failed: {}", e),
            Notice::PlanModeEntered => {
                outln!("\n📋 Plan mode — investigating before changing anything.\n")
            }
            Notice::PlanApproved { auto: false } => outln!("\n✓ Plan approved — starting work.\n"),
            Notice::PlanApproved { auto: true } => {
                outln!("\n✓ Plan approved — starting work in auto mode.");
                // Said plainly, because "auto" sounds like it will stop asking
                // and right now it does not.
                outln!("  Each gated call is reviewed first; you still answer the prompt.\n");
            }
            Notice::PlanRejected => outln!("\n✗ Plan sent back for revision.\n"),
            Notice::AskFailed { what, detail } => {
                outln!("\n[error] could not ask {}: {}", what, detail)
            }
            Notice::Refused(reason) => outln!("\n⛔ {}\n", reason),
            Notice::PermissionAllowed => outln!("✓ Permission granted for this request\n"),
            Notice::PermissionAllowedAll(label) => outln!("✓ {}\n", label),
            Notice::PermissionDenied { alternative } => {
                outln!("✗ Permission denied. User suggestion: {}\n", alternative)
            }
            Notice::LoopBroken { tool, count } => outln!(
                "\n[stopped: '{}' was called {} times in a row with the same arguments]\n",
                tool, count
            ),
            Notice::LoopNudge { tool, count } => outln!(
                "\n[looping: '{}' called {} times in a row — telling the model to stop]\n",
                tool, count
            ),
            Notice::QueuedSent(text) => outln!("{}", echo(PROMPT, &text)),
            Notice::ProjectContextReloaded(name) => {
                outln!("\nLoaded project context from {}", name)
            }
        }
    }
}
