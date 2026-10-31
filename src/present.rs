//! Turning [`AgentEvent`]s into terminal output. The other half of `src/turn.rs`.
//!
//! Everything the turn loop used to know about escapes, folds, previews and
//! wording now lives here, which is the point: `render.rs` is called from this
//! side of the seam and never from the core, so terminal knowledge cannot follow
//! the loop when the loop moves.

use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};

use crate::input::{echo, echo_message, PROMPT};
use crate::render::{change_preview, tool_call_line, tool_result_line, MarkdownStream};
use serde_json::Value;

use crate::render::indented_command;
use crate::turn::{AgentEvent, Notice};

/// A review on screen, waiting for its verdict.
struct Review {
    /// The "Reviewing tool call…" row the verdict replaces, when there is one to
    /// replace: none for a remembered verdict, which has no wait to show, and none
    /// under the inline renderer, which cannot take a row back.
    mark: Option<crate::out::Mark>,
    /// Under the call it belongs to: a step of the sub-agent's sits one level deeper.
    indent: &'static str,
}

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
    review: Option<Review>,
}

impl Presenter {
    pub(crate) fn new() -> Self {
        Presenter {
            markdown: MarkdownStream::new(),
            streamed_any: false,
            streamed_thinking: false,
            last_activity: None,
            review: None,
        }
    }

    /// Put the reviewer's verdict where the review said it would go. `dim` for the
    /// outcomes that let the call go on, which are not news; a refusal is. The icon
    /// alone takes `color`, so the outcome reads at a glance down a long run while
    /// the reason keeps the color of the line it is on.
    fn verdict(&mut self, icon: &str, color: Option<Color>, text: &str, dim: bool) {
        let Review { mark, indent } = self.review.take().unwrap_or(Review { mark: None, indent: "    " });
        // One row, however the reviewer worded it: a placeholder is one logical line and
        // is replaced by one. Not clipped — this is the whole account of why consent was
        // spent, or refused, on the user's behalf.
        let text: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let base = if dim { SetForegroundColor(crate::out::DIM).to_string() } else { String::new() };
        let icon = match color {
            // Back to the line's own color after it, not to the terminal's default: on a
            // dim line that would leave the reason in full white.
            Some(color) => format!("{}{}{}{}", SetForegroundColor(color), icon, ResetColor, base),
            None => icon.to_string(),
        };
        let line = format!("{}{}{} {}{}", base, indent, icon, text, ResetColor);
        crate::out::settle(mark, &line);
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
                        SetForegroundColor(crate::out::DIM),
                        ResetColor
                    ));
                    out!(
                        "{}{}○ {}",
                        SetAttribute(Attribute::Italic),
                        SetForegroundColor(crate::out::DIM),
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
                    SetForegroundColor(crate::out::DIM),
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
            // Already on screen when the review put it there.
            AgentEvent::ToolStarted { announced: true, .. } => {}
            AgentEvent::ToolStarted { name, arguments, announced: false } => {
                outln!("{}", tool_call_line(&name, &arguments));
            }
            AgentEvent::ReviewStarted { name, arguments, nested, diff, cached } => {
                if nested {
                    outln!("{}", sub_agent_step_line(&name, &arguments));
                } else {
                    outln!("{}", tool_call_line(&name, &arguments));
                }
                // Before the verdict, not after: the change is what is being judged.
                if let Some(diff) = diff {
                    show_diff(&diff);
                }
                let indent = if nested { "      " } else { "    " };
                let mark = match cached {
                    true => None,
                    false => crate::out::placeholder(&format!(
                        "{}{}⁇ Reviewing tool call…{}",
                        SetForegroundColor(crate::out::DIM),
                        indent,
                        ResetColor
                    )),
                };
                self.review = Some(Review { mark, indent });
            }
            AgentEvent::ToolDiff(diff) => show_diff(&diff),
            AgentEvent::ToolResult { text, failed } => {
                // The summary is the row that used to be all there was: the
                // first line, whole. The result itself is kept behind it now
                // rather than thrown away, and Ctrl+O opens it.
                if let Some(line) = tool_result_line(&text, failed) {
                    crate::out::foldable(&line, &text);
                }
            }
            AgentEvent::Finished { elapsed } => {
                // Dimmed for the same reason the chain of thought is: it is not
                // the answer. The verb is chosen here because it is flavor text —
                // one of a long list, picked fresh each turn — and choosing words
                // is the display's job, not the turn's.
                outln!(
                    "\n{}{}{}\n",
                    SetForegroundColor(crate::out::DIM),
                    turn_summary(get_done_verb(), elapsed, chrono::Local::now()),
                    ResetColor
                );
            }
            // Chrome, not transcript: the row above the input box. Phrased here
            // because "ran cargo build" is wording, and the turn only knows that
            // `run_command` finished with those arguments.
            AgentEvent::DidTool { name, arguments } => {
                self.last_activity = Some(
                    match crate::summary::tool_call_summary(&name, &arguments) {
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
            Notice::SessionNotSaved(e) => outln!("\n[error] the session was not saved: {}\n", e),
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
                        SetForegroundColor(crate::out::DIM),
                        ResetColor
                    ));
                    for line in summary.lines() {
                        outln!(
                            "{}{}      {}{}",
                            SetAttribute(Attribute::Italic),
                            SetForegroundColor(crate::out::DIM),
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
                outln!("\n≡ Plan mode — investigating before changing anything.\n")
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
            Notice::Refused(reason) => outln!("\n✘ {}\n", reason),
            Notice::ReviewAllowed { reason, cached } => {
                let how = if cached { "Allowed (cached)" } else { "Allowed" };
                self.verdict("✓", Some(Color::Green), &format!("{}: {}", how, reason), true)
            }
            // The blank row after these is the one a refusal always had: the call
            // ends here, and the model's next move starts a new paragraph.
            Notice::ReviewDenied { reason } => {
                self.verdict("✘", Some(Color::Red), &format!("Refused by review: {}", reason), false);
                outln!();
            }
            Notice::ReviewFailed { detail } => {
                self.verdict(
                    "⚠",
                    Some(Color::Yellow),
                    &format!("Review gave no verdict ({}): not run, the model will retry", detail),
                    false,
                );
                outln!();
            }
            Notice::ReviewInterrupted => self.verdict("↪", None, "Review interrupted: asking you", true),
            Notice::PermissionAllowed => outln!("✓ Permission granted for this request\n"),
            Notice::PermissionAllowedAll(label) => outln!("✓ {}\n", label),
            Notice::PermissionDenied { alternative } => {
                outln!("✗ Permission denied. User suggestion: {}\n", alternative)
            }
            Notice::LoopBroken { tool, count } => outln!(
                "\n[stopped: '{}' was called {} times in a row with the same arguments]\n",
                tool, count
            ),
            Notice::Repeating { passage, retry } => {
                match retry {
                    Some((retry, of)) => outln!(
                        "\n[repeating itself: reply cut off and dropped — asking again ({} of {})]",
                        retry, of
                    ),
                    None => outln!("\n[stopped: the model kept repeating itself]"),
                }
                outln!("  \"{}\"\n", passage);
            }
            Notice::Retrying { error, retry, of, delay } => outln!(
                "
[server error: {} — retrying in {}s ({} of {})]
",
                error,
                delay.as_secs(),
                retry,
                of
            ),
            // Indented under the sub_agent call line, and dim like research's own
            // progress: these are the steps of one call, not calls of their own.
            Notice::SubAgentStep { announced: true, .. } => {}
            Notice::SubAgentStep { tool, arguments, announced: false } => {
                outln!("{}", sub_agent_step_line(&tool, &arguments))
            }
            Notice::LookupStep(step) => {
                outln!("{}  {}{}", SetForegroundColor(crate::out::DIM), step, ResetColor)
            }
            Notice::LookupAnswer { label, answer } => {
                outln!("{}{}:{} {}", SetForegroundColor(crate::out::DIM), label, ResetColor, answer)
            }
            Notice::CommandOutput(line) => outln!("  │ {}", line.trim_end_matches('\r')),
            // Nothing to print in the REPL: the list is pinned above the input box, and the
            // next draw picks the new one up from the agent. A piped session has no bottom
            // region to pin anything to, so there it still goes into the transcript —
            // buried under the next few tool calls, but shown, which beats not shown at all.
            Notice::PlanUpdated(todos) => {
                if !crate::out::has_region() {
                    outln!("\n  Plan:");
                    for item in &todos {
                        let mark = match item.status.as_str() {
                            "completed" => "x",
                            "in_progress" => "»",
                            _ => " ",
                        };
                        outln!("   [{}] {}", mark, item.task);
                    }
                    outln!();
                }
            }
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
            Notice::QueuedSent(text) => outln!("{}", echo_message(PROMPT, &text)),
            Notice::TimersFired(fired) => outln!("{}", timers_fired(&fired)),
            Notice::ProjectContextReloaded(name) => {
                outln!("\nLoaded project context from {}", name)
            }
        }
    }
}

/// A few lines of the change, with the whole diff behind them — the terminal's
/// decision, not the turn's. It has a height, and the fold is what it has instead
/// of scrollback.
fn show_diff(diff: &crate::diff::Diff) {
    let diff = crate::render::diff_text(diff);
    if let Some(preview) = change_preview(&diff) {
        crate::out::foldable(preview.trim_end_matches('\n'), &diff);
    }
}

/// One of the sub-agent's calls: indented under the `sub_agent` call line, and dim
/// like research's own progress, because it is a step of one call rather than a
/// call of its own.
fn sub_agent_step_line(tool: &str, arguments: &Value) -> String {
    format!(
        "{}  ↳ {} {}{}",
        SetForegroundColor(crate::out::DIM),
        tool,
        crate::summary::tool_call_summary(tool, arguments),
        ResetColor
    )
}

/// The sentence a permission prompt puts above the buttons.
///
/// Rendering, not policy: the turn sends the call it wants to make and this
/// decides how to say it — how much of a command to show, and whether a byte
/// count or a diff is the useful summary. It lived in the turn loop until the
/// event stream gave it somewhere better.
pub(crate) fn permission_description(tool_name: &str, arguments: &Value) -> String {
    match tool_name {
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
    }
}

/// The line a timer firing leaves in the transcript, in place of the echo a typed
/// message gets. Shared by the notice and the idle prompt, which fires timers without a
/// turn running to send one.
pub(crate) fn timers_fired(fired: &[crate::timers::Fired]) -> String {
    fired
        .iter()
        .map(|f| {
            let head = format!("◷ Timer {} ({}): ", f.id, f.schedule.describe());
            echo(&head, &f.prompt)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// What /context says: how full the window is, and whether reasoning compression is paying.
///
/// `used` is `Agent::context_used`, and `measured` whether that rests on the server's own
/// count of the last request rather than on the estimate alone.
pub(crate) fn report_context(
    messages: &[Value],
    used: usize,
    measured: bool,
    context_tokens: usize,
    reasoning: &crate::session::ReasoningStats,
) {
    let turns = messages.iter().filter(|m| m["role"] == "user").count();
    let percent = (used as f64 / context_tokens as f64) * 100.0;

    outln!(
        "{} messages, {} user turn(s), ~{} of {} tokens ({:.1}%, {}).",
        messages.len(),
        turns,
        used,
        context_tokens,
        percent,
        if measured { "from the server's count of the last request" } else { "estimated" }
    );

    // Only once there is something to report. A line of zeroes under every /context would
    // be noise for the default policy, which never compresses anything.
    if reasoning.turns > 0 {
        outln!(
            "Reasoning compressed on {} repl{}: ~{} -> ~{} tokens, saving ~{}.",
            reasoning.turns,
            if reasoning.turns == 1 { "y" } else { "ies" },
            reasoning.before,
            reasoning.after,
            reasoning.saved()
        );
    }

    if used > crate::config::compact_hint_threshold(context_tokens) {
        outln!("Run /compact to replace the history with a summary.");
    }
    outln!();
}

pub(crate) fn get_thinking_message() -> &'static str {
    use std::time::{SystemTime, UNIX_EPOCH};
    const MESSAGES: &[&str] = &[
        // First step
        "Fancying around…",
        "Thinking deeply…",
        "Connecting the dots…",
        "Pondering possibilities…",
        "Weighing options…",
        "Untangling logic…",
        "Simulating neurons…",
        "Brewing ideas…",
        "Scanning the horizon…",
        "Assembling the puzzle…",

        // Technical / Agent-y
        "Parsing intent…",
        "Traversing the graph…",
        "Running inference…",
        "Compiling thoughts…",
        "Optimizing the plan…",
        "Executing mental threads…",
        "Querying the knowledge base…",
        "Running a quick sanity check…",
        "Bootstrapping context…",
        "Refining the approach…",

        // Whimsical / Personable
        "Staring into the void (productively)…",
        "Consulting the oracle…",
        "Summoning the code spirits…",
        "Bribing the compiler…",
        "Rubber-ducking internally…",
        "Channeling Ada Lovelace…",
        "Polishing the crystal ball…",
        "Negotiating with entropy…",
        "Warming up the neurons…",
        "Reticulating splines…",

        // Concise / Dry
        "Processing…",
        "Analyzing…",
        "Evaluating…",
        "Computing…",
        "Reasoning…",
        "Synthesizing…",
        "Deliberating…",
        "Calculating…",
        "Inspecting…",
        "Contemplating…",

        // Creative / Poetic
        "Following the thread…",
        "Tracing constellations…",
        "Weaving the narrative…",
        "Tuning the frequency…",
        "Listening to the silence…",
        "Reading between the lines…",
        "Chasing the spark…",
        "Stirring the cauldron…",
        "Sharpening the quill…",
        "Lighting the lantern…",
    ];

    let idx = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as usize % MESSAGES.len();

    MESSAGES[idx]
}

// The verb in the line printed when a turn ends. A companion to the list above: the
// spinner says what it is doing in the present tense, this says what it did.
//
// Every one of these has to read as an intransitive past tense — "Baked for 3m 34s" —
// so verbs that want an object ("Untangled", "Assembled") stay out however well they
// fit the spinner.
pub(crate) fn get_done_verb() -> &'static str {
    use std::time::{SystemTime, UNIX_EPOCH};
    const VERBS: &[&str] = &[
        // Kitchen
        "Baked",
        "Brewed",
        "Simmered",
        "Stewed",
        "Marinated",
        "Percolated",
        "Steeped",
        "Kneaded",

        // Thought
        "Pondered",
        "Ruminated",
        "Cogitated",
        "Deliberated",
        "Mused",
        "Reasoned",
        "Reflected",
        "Deduced",

        // Puttering
        "Noodled",
        "Tinkered",
        "Puttered",
        "Pottered",
        "Fiddled",
        "Meandered",
        "Rummaged",
        "Foraged",

        // Machinery
        "Churned",
        "Whirred",
        "Hummed",
        "Ticked",
        "Crunched",
        "Toiled",
        "Labored",
        "Computed",
    ];

    let idx = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as usize % VERBS.len();

    VERBS[idx]
}

// How long the turn took, in the largest two units that carry information.
//
// Seconds are dropped past the hour on purpose: a turn that ran for over an hour is not
// one anybody is counting seconds on, and "1h 4m 12s" is three numbers to read where two
// say the same thing.
pub(crate) fn format_elapsed(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

// The line that closes a turn: what it did, how long for, and the time it finished at.
//
// Both halves earn their place. A turn on a local model can run for minutes with its own
// start scrolled far off the top of the terminal, so the duration is not recoverable from
// what is on screen; and a turn that finished while nobody was watching wants a clock
// time, because a duration alone does not say how stale the answer is.
pub(crate) fn turn_summary(verb: &str, elapsed: std::time::Duration, finished: chrono::DateTime<chrono::Local>) -> String {
    format!(
        "✻ {} for {} · done {}",
        verb,
        format_elapsed(elapsed),
        finished.format("%-I:%M %p")
    )
}

#[cfg(test)]
mod wording_tests {
    use super::*;
    use chrono::TimeZone;
    use std::time::Duration;

    #[test]
    fn elapsed_uses_two_units_at_most() {
        assert_eq!(format_elapsed(Duration::from_millis(400)), "0s");
        assert_eq!(format_elapsed(Duration::from_secs(12)), "12s");
        assert_eq!(format_elapsed(Duration::from_secs(59)), "59s");
        // The minute boundary keeps the seconds, even when there are none of them: a bare
        // "1m" reads as a rounded figure where "1m 0s" reads as the measurement it is.
        assert_eq!(format_elapsed(Duration::from_secs(60)), "1m 0s");
        assert_eq!(format_elapsed(Duration::from_secs(214)), "3m 34s");
        assert_eq!(format_elapsed(Duration::from_secs(3599)), "59m 59s");
        // Past the hour the seconds go, rather than becoming a third number.
        assert_eq!(format_elapsed(Duration::from_secs(3600)), "1h 0m");
        assert_eq!(format_elapsed(Duration::from_secs(3852)), "1h 4m");
    }

    #[test]
    fn summary_reads_as_one_sentence() {
        let finished = chrono::Local.with_ymd_and_hms(2026, 8, 28, 2, 13, 0).unwrap();
        assert_eq!(
            turn_summary("Baked", Duration::from_secs(214), finished),
            "✻ Baked for 3m 34s · done 2:13 AM"
        );

        // The hour is unpadded — "2:13 AM", never "02:13 AM" — and the afternoon still
        // reads on a twelve-hour clock.
        let finished = chrono::Local.with_ymd_and_hms(2026, 8, 28, 14, 5, 0).unwrap();
        assert_eq!(
            turn_summary("Pottered", Duration::from_secs(9), finished),
            "✻ Pottered for 9s · done 2:05 PM"
        );
    }

    #[test]
    fn every_done_verb_reads_intransitively() {
        // The list is picked from by clock, so nothing here can check the choice — what it
        // can check is that whatever comes out slots into the sentence without a space or
        // a stray capital in the middle of it.
        let verb = get_done_verb();
        assert!(!verb.is_empty());
        assert!(verb.chars().all(|c| c.is_ascii_alphabetic()));
        assert!(verb.starts_with(|c: char| c.is_uppercase()));
    }
}

#[cfg(test)]
mod presenter_tests {
    use super::*;
    use crate::out;
    use crate::tools::TodoItem;
    use serde_json::json;

    // What the presenter printed for these events, escapes and all.
    fn shown(events: Vec<AgentEvent>) -> String {
        let _capture = out::capture_guard();
        out::start_capture();
        let mut presenter = Presenter::new();
        for event in events {
            presenter.show(event);
        }
        out::take_capture()
    }

    #[test]
    fn a_tool_result_shows_its_first_line_and_keeps_the_rest_folded() {
        let text = shown(vec![AgentEvent::ToolResult { text: "\n  first line\nsecond\nthird".into(), failed: false }]);
        assert!(text.contains("first line"), "got: {text:?}");
        // Folded away, not lost: the fold is collapsed by default, so the rest is not on
        // screen — and an empty first line is skipped rather than shown as a blank row.
        assert!(!text.contains("second"), "got: {text:?}");
    }

    #[test]
    fn a_failed_tool_result_is_shown_in_red() {
        // Without it, under NO_COLOR crossterm writes `red` as an empty SGR, which a rendered
        // line holds only when another test has not already taken the switch.
        out::own_colors();
        let red = crossterm::style::SetForegroundColor(crossterm::style::Color::Red).to_string();
        let failed = shown(vec![AgentEvent::ToolResult { text: "error[E0308]: mismatched types".into(), failed: true }]);
        assert!(failed.contains(&red), "got: {failed:?}");
        let passed = shown(vec![AgentEvent::ToolResult { text: "ok".into(), failed: false }]);
        assert!(!passed.contains(&red), "got: {passed:?}");
    }

    #[test]
    fn a_lookup_and_a_command_report_their_steps_indented_under_the_call() {
        let text = shown(vec![
            Notice::LookupStep("Searching: tokio select".into()).into(),
            Notice::CommandOutput("   Compiling haico v0.2.27\r".into()).into(),
        ]);
        let plain = crate::util::strip_ansi(&text);
        assert!(plain.contains("  Searching: tokio select"), "got: {plain:?}");
        // A carriage return a command ended its line with is not drawn.
        assert!(plain.contains("  │    Compiling haico v0.2.27\n"), "got: {plain:?}");
    }

    #[test]
    fn a_lookup_answer_is_shown_whole_under_its_label() {
        let text = crate::util::strip_ansi(&shown(vec![
            Notice::LookupAnswer { label: "Library answer", answer: "Use biased; to poll in order.".into() }.into(),
        ]));
        assert!(text.contains("Library answer: Use biased; to poll in order."), "got: {text:?}");
    }

    #[test]
    fn the_plan_is_printed_when_there_is_no_box_to_pin_it_above() {
        // A test has no input region, which is the piped session's case.
        let todos = vec![
            TodoItem { task: "write the parser".into(), status: "completed".into() },
            TodoItem { task: "test it".into(), status: "in_progress".into() },
        ];
        let text = shown(vec![Notice::PlanUpdated(todos).into()]);
        assert!(text.contains("[x] write the parser"), "got: {text:?}");
        assert!(text.contains("[»] test it"), "got: {text:?}");
    }

    #[test]
    fn a_session_that_was_not_saved_is_an_error_the_user_sees() {
        let text = shown(vec![Notice::SessionNotSaved("could not write session.json".into()).into()]);
        assert!(text.contains("[error] the session was not saved: could not write session.json"), "got: {text:?}");
    }

    fn reviewed(nested: bool, cached: bool) -> AgentEvent {
        AgentEvent::ReviewStarted {
            name: "run_command".into(),
            arguments: json!({"command": "cargo build"}),
            nested,
            diff: None,
            cached,
        }
    }

    // The call is on screen before the verdict and is not drawn again when it starts;
    // the verdict sits under it. Inline there is no placeholder to replace, so the
    // verdict is the only row the review adds.
    #[test]
    fn a_reviewed_call_is_shown_once_with_its_verdict_under_it() {
        let text = crate::util::strip_ansi(&shown(vec![
            reviewed(false, false),
            Notice::ReviewAllowed { reason: "builds the project".into(), cached: false }.into(),
            AgentEvent::ToolStarted {
                name: "run_command".into(),
                arguments: json!({"command": "cargo build"}),
                announced: true,
            },
        ]));
        assert_eq!(text.matches("cargo build").count(), 1, "got: {text:?}");
        let call = text.find("→ run_command").expect("no call line");
        let verdict = text.find("    ✓ Allowed: builds the project").expect("no verdict");
        assert!(call < verdict, "got: {text:?}");
        assert!(!text.contains("Reviewing"), "got: {text:?}");
    }

    // The icon carries the outcome's color and the reason goes back to the line's own:
    // dim after an allow, the plain text color after a refusal.
    #[test]
    fn the_verdict_icon_is_green_for_an_allow_and_red_for_a_refusal() {
        // Under NO_COLOR crossterm would write the colors as empty SGRs; see
        // `a_failed_tool_result_is_shown_in_red`.
        out::own_colors();
        let green = SetForegroundColor(Color::Green).to_string();
        let red = SetForegroundColor(Color::Red).to_string();
        let dim = SetForegroundColor(out::DIM).to_string();

        let allowed = shown(vec![
            reviewed(false, false),
            Notice::ReviewAllowed { reason: "fine".into(), cached: false }.into(),
        ]);
        assert!(allowed.contains(&format!("{green}✓{ResetColor}{dim} Allowed: fine")), "got: {allowed:?}");

        let denied = shown(vec![reviewed(false, false), Notice::ReviewDenied { reason: "no".into() }.into()]);
        assert!(denied.contains(&format!("{red}✘{ResetColor} Refused by review: no")), "got: {denied:?}");
    }

    #[test]
    fn a_remembered_verdict_says_so() {
        let text = crate::util::strip_ansi(&shown(vec![
            reviewed(false, true),
            Notice::ReviewAllowed { reason: "seen before".into(), cached: true }.into(),
        ]));
        assert!(text.contains("✓ Allowed (cached): seen before"), "got: {text:?}");
    }

    // A step of the sub-agent's is drawn as a step, and its verdict one level deeper.
    #[test]
    fn a_reviewed_sub_agent_step_is_indented_under_the_call() {
        let text = crate::util::strip_ansi(&shown(vec![
            reviewed(true, false),
            Notice::ReviewDenied { reason: "not in the workspace".into() }.into(),
        ]));
        assert!(text.contains("  ↳ run_command cargo build\n"), "got: {text:?}");
        assert!(text.contains("\n      ✘ Refused by review: not in the workspace\n"), "got: {text:?}");
        assert!(!text.contains("→"), "got: {text:?}");
    }

    #[test]
    fn the_permission_question_names_what_the_call_will_do() {
        assert_eq!(
            permission_description("write_file", &json!({"file_path": "a.rs", "content": "12345"})),
            "Write 5 bytes to file: a.rs"
        );
        assert_eq!(
            permission_description("edit", &json!({"file_path": "a.rs", "old_string": "x", "new_string": "y", "replace_all": true})),
            "Edit file (replace all 'x'->'y') in a.rs"
        );
        assert_eq!(
            permission_description("web_search", &json!({"query": "axum routing"})),
            "Search the web for: 'axum routing'"
        );
    }
}
