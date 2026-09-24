//! Turning [`AgentEvent`]s into terminal output. The other half of `src/turn.rs`.
//!
//! Everything the turn loop used to know about escapes, folds, previews and
//! wording now lives here, which is the point: `render.rs` is called from this
//! side of the seam and never from the core, so terminal knowledge cannot follow
//! the loop when the loop moves.

use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};

use crate::input::{echo, PROMPT};
use crate::render::{change_preview, tool_call_line, tool_result_line, MarkdownStream};
use serde_json::Value;

use crate::render::indented_command;
use crate::turn::{AgentEvent, Notice, ReviewNote};

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
    /// What the row above the input box says once the turn is over: the last
    /// thing that actually happened, rather than an empty row. Chrome, so the
    /// pump reads it rather than the presenter drawing it.
    last_activity: Option<String>,
}

impl Presenter {
    pub(crate) fn new() -> Self {
        Presenter {
            markdown: MarkdownStream::new(),
            streamed_any: false,
            streamed_thinking: false,
            last_activity: None,
        }
    }

    /// The row above the input box: the last thing that happened.
    pub(crate) fn last_activity(&self) -> Option<&str> {
        self.last_activity.as_deref()
    }

    /// Ready for the next reply. Per reply, not per turn: a turn is many rounds
    /// and each round streams one.
    fn start_reply(&mut self) {
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
            AgentEvent::ReplyStart => self.start_reply(),
            AgentEvent::ReasoningChunk(text) => {
                if !self.streamed_thinking {
                    self.streamed_thinking = true;
                    // A row of its own, with a blank one above it, for the
                    // same reason the answer gets one: the thinking is not a
                    // continuation of the prompt, or of the tool result it
                    // follows.
                    crate::out::ensure_line_start();
                    outln!();
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
                // The summary is the row that used to be all there was: the
                // first line, whole. The result itself is kept behind it now
                // rather than thrown away, and Ctrl+O opens it.
                if let Some(line) = tool_result_line(&result) {
                    crate::out::foldable(&line, &result);
                }
            }
            AgentEvent::Finished { elapsed } => {
                // Dimmed for the same reason the chain of thought is: it is not
                // the answer. The verb is chosen here because it is flavor text —
                // one of a long list, picked fresh each turn — and choosing words
                // is the display's job, not the turn's.
                outln!(
                    "\n{}{}{}\n",
                    SetForegroundColor(Color::DarkGrey),
                    crate::turn_summary(crate::get_done_verb(), elapsed, chrono::Local::now()),
                    ResetColor
                );
            }
            // Chrome, not transcript: the row above the input box. Phrased here
            // because "ran cargo build" is wording, and the turn only knows that
            // `run_command` finished with those arguments.
            AgentEvent::DidTool { name, arguments } => {
                self.last_activity = Some(
                    match crate::render::tool_call_summary(&name, &arguments) {
                        summary if summary.is_empty() => format!("ran {}", name),
                        // For a shell tool the command *is* the thing that ran,
                        // so naming the tool as well would say "run_command" twice.
                        summary if matches!(name.as_str(), "run_command" | "run_background") => {
                            format!("ran {}", summary)
                        }
                        summary => format!("ran {} {}", name, summary),
                    },
                );
            }
            AgentEvent::Notice(notice) => self.notice(notice),
            // The pump's, not the presenter's: the activity bar and the status
            // bar are chrome drawn around the transcript rather than lines in
            // it, and a question needs answering rather than showing. Listed
            // explicitly instead of caught by `_` so that a new variant still
            // fails to compile here until someone has decided which side it is.
            AgentEvent::Busy(_)
            | AgentEvent::Status { .. }
            | AgentEvent::ToolCallChunk(_)
            | AgentEvent::Ask(_) => {}
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
            // The figures are shown rather than folded: the whole point of
            // compressing is to find out whether it is paying for itself, which
            // cannot be judged from a line hidden by default.
            Notice::ReasoningCompressed { before, after, summary } => {
                outln!(
                    "{}Compressed reasoning from {} to {} tokens.{}",
                    SetForegroundColor(crate::input::BAR_BACKGROUND),
                    before,
                    after,
                    ResetColor
                );
                // The figures say the trace shrank; they do not say whether what
                // came back is still worth having, and the original is gone from
                // the history by now. So the summary goes behind a fold of its
                // own, opened by the same Ctrl+O as the thinking it replaced —
                // dim and italic for the same reason that is: it is not the
                // answer. Written per line rather than left switched on, so a
                // color cannot bleed past the block.
                //
                // `begin_fold` rather than `foldable` because of what the inline
                // renderer does with each: `foldable` would print the summary row
                // and nothing else, leaving a label there for a fold that cannot
                // open. This way the text itself is simply printed under the line
                // above, which already says what it is.
                if !summary.is_empty() {
                    crate::out::begin_fold(&format!(
                        "{}{}○ (compressed reasoning){}",
                        SetAttribute(Attribute::Italic),
                        SetForegroundColor(Color::DarkGrey),
                        ResetColor
                    ));
                    for line in summary.lines() {
                        outln!(
                            "{}{}      {}{}",
                            SetAttribute(Attribute::Italic),
                            SetForegroundColor(Color::DarkGrey),
                            line,
                            ResetColor
                        );
                    }
                    crate::out::end_fold();
                }
            }
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
                // Said plainly, because "auto" does not say which way a doubtful
                // call goes: it is refused, not put to the user.
                outln!("  Each gated call is reviewed first; what the review rejects is refused.\n");
            }
            Notice::PlanRejected => outln!("\n✗ Plan sent back for revision.\n"),
            Notice::AskFailed { what, detail } => {
                outln!("\n[error] could not ask {}: {}", what, detail)
            }
            Notice::Refused(reason) => outln!("\n⛔ {}\n", reason),
            Notice::ReviewDenied { tool, arguments, reason } => {
                outln!("{}", tool_call_line(&tool, &arguments, None));
                outln!("⛔ Refused by review: {}\n", reason)
            }
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
            // Under the echoed message, indented, because that is what it is
            // about — and one line each however long the file was, since the
            // contents went to the model and not to the screen.
            Notice::FileAttached { path, lines, outside } => outln!(
                "  ⊕ {} ({} line{}){}",
                path,
                lines,
                if lines == 1 { "" } else { "s" },
                if outside { " — outside the workspace, added for this session" } else { "" }
            ),
            // Kilobytes rather than lines, and the icon differs, because what
            // the request carries is what the reader wants to weigh up: an
            // image is worth ten of the text file next to it in the list.
            Notice::ImageAttached { path, bytes, outside } => outln!(
                "  ⊕ {} (image, {} KB){}",
                path,
                bytes.div_ceil(1024),
                if outside { " — outside the workspace, added for this session" } else { "" }
            ),
            Notice::FileNotAttached { path, reason } => {
                outln!("  ⊘ {} was not attached: {}", path, reason)
            }
            // Two lines, because both halves are news: what happened to the
            // conversation, and what the server said that prompted it. The
            // second is folded — it is a server error, wanted only if the
            // retry fails too.
            Notice::ImagesStripped(detail) => {
                outln!(
                    "\n⊘ The request was refused with an image in it, so the images were \
                     removed and it was sent again. This model most likely cannot see them."
                );
                let summary = detail.lines().next().unwrap_or_default();
                crate::out::foldable(
                    &format!("  the server said: {}", summary),
                    &format!("  the server said: {}", detail),
                );
            }
            Notice::QueuedSent(text) => outln!("{}", echo(PROMPT, &text)),
            Notice::ProjectContextReloaded(name) => {
                outln!("\nLoaded project context from {}", name)
            }
        }
    }
}

/// The sentence a permission prompt puts above the buttons.
///
/// Rendering, not policy: the turn sends the call it wants to make and this
/// decides how to say it — how much of a command to show, whether a byte count
/// or a diff is the useful summary, and where the reviewer's objection goes.
/// It lived in the turn loop until the event stream gave it somewhere better.
pub(crate) fn permission_description(
    tool_name: &str,
    arguments: &Value,
    review: Option<&ReviewNote>,
) -> String {
    let mut description = match tool_name {
        "web_search" => {
            let query = arguments["query"].as_str().unwrap_or("unknown");
            format!("Search the web for: '{}'", query)
        }
        "web_fetch" | "web_get" => {
            let url = arguments["url"].as_str().unwrap_or("unknown");
            format!("Fetch content from URL: {}", url)
        }
        "write_file" => {
            let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
            let size = arguments["content"].as_str().map(|s| s.len()).unwrap_or(0);
            format!("Write {} bytes to file: {}", size, file_path)
        }
        "edit" => {
            let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
            let from = arguments["old_string"].as_str().unwrap_or("unknown");
            let to = arguments["new_string"].as_str().unwrap_or("unknown");
            let replace_all = arguments["replace_all"].as_bool().unwrap_or(false);
            if replace_all {
                format!("Edit file (replace all '{from}'->'{to}') in {}", file_path)
            } else {
                format!("Edit file (replace once '{from}'->'{to}') in {}", file_path)
            }
        }
        "multi_edit" => {
            let file_path = arguments["file_path"].as_str().unwrap_or("unknown");
            let count = arguments["edits"].as_array().map(|e| e.len()).unwrap_or(0);
            format!("Apply {} edit(s) to file: {}", count, file_path)
        }
        // The command is laid out over several indented rows rather than cut to a
        // width: a generated one-liner cut short is a script approved sight unseen.
        "run_background" => {
            let command = arguments["command"].as_str().unwrap_or("unknown");
            format!("Start in the background:\n{}", indented_command(command))
        }
        "run_command" => {
            let command = arguments["command"].as_str().unwrap_or("unknown");
            format!("Execute command:\n{}", indented_command(command))
        }
        _ => format!("Execute '{}' tool", tool_name),
    };

    // Why the reviewer gave no verdict goes inside the prompt rather than on a
    // line of its own: it is the reason this prompt is being shown at all.
    if let Some(review) = review {
        let line = match review {
            ReviewNote::Failed(detail) => format!("🤖 Review failed: {}", detail),
            ReviewNote::Interrupted => "🤖 Review interrupted.".to_string(),
        };
        description.push_str(&format!("\n\n{}", line));
    }
    description
}
