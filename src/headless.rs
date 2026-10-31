//! `--prompt`: one task, run unattended in auto mode, and then exit.
//!
//! The third consumer of the turn's events, beside the REPL's pump and the window.
//! It renders through the same `Presenter` as the REPL — a log of an unattended run
//! should read like a session did — but owns no keyboard and draws no chrome, so it
//! works the same in a terminal, a pipe, a scheduled task or CI.
//!
//! What makes it more than "the REPL with one line of input" is deciding when the
//! run is over. A turn ending is not the task ending: small local models routinely
//! stop to announce the next step ("Now I will run the tests."), to ask whether they
//! should go on, or with half their own plan still pending — all of which a person
//! at the prompt answers with "continue", and nobody here will. So a turn that ends
//! looking unfinished is followed up, a bounded number of times, and only a turn
//! that ends looking finished exits cleanly.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::agent::{run_turn, Agent};
use crate::config::Config;
use crate::modes::{Mode, PlanDecision};
use crate::present::Presenter;
use crate::skills;
use crate::tools::TodoItem;
use crate::turn::{AgentEvent, Ask, TurnEnd, TurnLink};
use crate::util::display_path;

/// The run finished: the model's last turn ended in an answer that did not look
/// like it was waiting for anything.
pub(crate) const EXIT_DONE: i32 = 0;
/// A turn ended without an answer — the server failed, or the loop guard gave up.
pub(crate) const EXIT_FAILED: i32 = 1;
/// The model kept stopping short of the end and the follow-ups ran out.
pub(crate) const EXIT_UNFINISHED: i32 = 2;
/// Ctrl+C, by the shell convention for a process ended by SIGINT.
pub(crate) const EXIT_INTERRUPTED: i32 = 130;

/// How many times a turn that stopped short is sent back to work before the run
/// gives up. Enough for a model that pauses between phases of a job; few enough
/// that one which will never finish costs minutes rather than the night.
const MAX_FOLLOW_UPS: usize = 3;

/// Said once, with the task, because every habit it heads off is one the model
/// learned from a user being there: asking before doing, stopping to report
/// progress, handing a decision back.
const UNATTENDED_NOTE: &str = "This task is running unattended: nobody will answer a \
question, review a plan or approve anything, and the run ends as soon as you reply \
without calling a tool — unless a timer you set is still armed, in which case it waits \
for it, so cancel any you no longer need. Work until the task is completely done and verified, deciding \
anything ambiguous yourself. Then finish with a short summary of what you did.";

/// Run `task` in the current directory and return the process's exit code.
///
/// An `Err` is only ever a failure to start — no model to use, no server to reach.
/// Everything that can go wrong once the agent exists is an exit code instead,
/// so a script calling this can tell "never ran" from "ran and did not finish".
pub(crate) async fn run(config: Config, task: String) -> Result<i32> {
    // Checked once: where the output goes does not change during a run.
    use std::io::IsTerminal;
    crate::out::set_plain(!std::io::stdout().is_terminal());

    if let Some(notice) = crate::config::starter_config_notice(&config) {
        outln!("{}", notice);
    }
    if let Some(notice) = crate::config::config_error_notice(&config) {
        outln!("{}", notice);
    }
    match &config.provider {
        Some(provider) => outln!("Connecting to {} at: {}", provider, config.llm_url),
        None => outln!("Connecting to LLM server at: {}", config.llm_url),
    }
    let model = crate::repl::choose_model(&config, None).await?;

    let cwd = std::env::current_dir()?;
    let resume = config.resume;
    let mut agent = Agent::new(config, model, &cwd);
    // Auto mode is the whole premise: every call that would have asked is put to
    // the reviewer instead, and its no is a refusal the model is told about rather
    // than a dialog nobody is there to answer.
    agent.mode = Mode::Auto;

    if let Some((name, _)) = &agent.project_context {
        outln!("Loaded project context from {}", name);
    }
    // Saved only when continuing one. The session file is per directory, and a fresh
    // run saving would replace whatever conversation the user left here at the prompt —
    // on every run, for a task on a schedule. `--resume` is the opt-in: it says this
    // directory's conversation is the one being worked on, so it is kept up to date.
    if !resume {
        agent.session_path = None;
    } else {
        match agent.resume() {
            Ok(restored) => outln!("Resumed {} message(s) from {}.", restored.messages, restored.saved_at),
            Err(e) => outln!("Could not resume: {:#}. Starting fresh.", e),
        }
    }
    // Waited for, unlike at the REPL: this run sends its first request once and has no
    // later turn for a slow server's tools to join. Bounded by each server's startup
    // timeout. Unapproved project servers are skipped — there is nobody to ask, and an
    // unattended run is the last place to start a command a cloned repository supplied.
    crate::mcp::Startup::load(&cwd).start(&agent.server.mcp);
    print_mcp_notices(&agent);
    agent.server.mcp.wait_loaded().await;
    print_mcp_notices(&agent);
    outln!("Running unattended in {} (auto mode).\n", display_path(&cwd));

    let mut input = first_message(&cwd, &task);
    // The echo shows the task as given rather than with the note appended: the note
    // is the same paragraph on every run, and the log is read for what was asked.
    let mut echoed = crate::input::echo(crate::input::PROMPT, task.trim());
    let mut follow_ups = 0;
    let code = loop {
        outln!("{}", echoed);
        let ended = drive(&mut agent, &input).await;
        // A server lost mid-run, or one whose tool list changed.
        print_mcp_notices(&agent);
        match ended {
            TurnEnd::Answered => {}
            TurnEnd::Interrupted => break EXIT_INTERRUPTED,
            TurnEnd::Failed => {
                outln!("\nStopped: the last turn ended without an answer.");
                break EXIT_FAILED;
            }
        }
        let Some(follow_up) = follow_up(&agent.plan(), last_reply(&agent.messages)) else {
            // Finished — unless the model left itself a timer, which is a statement that
            // it is not finished yet, only waiting. Its firing is a fresh piece of work,
            // so it gets a fresh set of follow-ups.
            match wait_for_timers(&agent).await {
                Waited::Fired(fired) => {
                    follow_ups = 0;
                    echoed = crate::present::timers_fired(&fired);
                    input = crate::timers::message_for(&fired);
                    continue;
                }
                Waited::Nothing => break EXIT_DONE,
                Waited::Interrupted => break EXIT_INTERRUPTED,
            }
        };
        if follow_ups == MAX_FOLLOW_UPS {
            outln!(
                "\nStopped: the model still had not finished after {} follow-ups.",
                MAX_FOLLOW_UPS
            );
            break EXIT_UNFINISHED;
        }
        follow_ups += 1;
        echoed = crate::input::echo(crate::input::PROMPT, &follow_up);
        input = follow_up;
    };

    // The REPL does this on its way out for the same reason: a dev server started
    // by the task must not outlive the run holding a port.
    let stopped = agent.server.stop_all_background().await;
    if stopped > 0 {
        outln!("Stopped {} background job(s).", stopped);
    }
    agent.server.mcp.stop_all();
    Ok(code)
}

fn print_mcp_notices(agent: &Agent) {
    for notice in agent.server.mcp.take_notices() {
        let mark = if notice.is_problem() { "⚠ " } else { "" };
        outln!("{}{}", mark, notice.text());
    }
}

/// The task as the model first sees it. A task that names a skill (`/release 1.2`)
/// expands into that skill's instructions, exactly as typing it at the prompt does,
/// so a scheduled run can be a one-word command.
fn first_message(cwd: &Path, task: &str) -> String {
    let task = skills::expand_command(&skills::roots(cwd), task.trim())
        .unwrap_or_else(|| task.trim().to_string());
    format!("{}\n\n({})", task, UNATTENDED_NOTE)
}

/// One turn, rendered as it happens, with its questions answered by nobody.
///
/// Ctrl+C cancels the turn the way Esc does in the REPL, so it stops somewhere it
/// chose with every tool call answered and the session saved. A second one does not
/// wait for that: someone pressing it twice wants the process gone.
async fn drive(agent: &mut Agent, input: &str) -> TurnEnd {
    let link = TurnLink::new(agent.mode);
    let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
    let mut presenter = Presenter::new();

    let end = {
        let turn = run_turn(agent, input, false, &tx, &link);
        tokio::pin!(turn);
        let mut cancelled = false;
        loop {
            tokio::select! {
                end = &mut turn => break end,
                Some(event) = events.recv() => show(&mut presenter, event),
                // The first Ctrl+C asks the turn to stop where it chooses, with its
                // history balanced; the second stops waiting for it. That used to be
                // `process::exit`, which skipped everything on the way out — background
                // jobs were left running holding their ports, and the trace never got
                // its `end` line. Dropping the turn instead lets `run` stop the jobs and
                // `main` end the trace, which is still immediate: nothing is awaited but
                // the kills. The history dropped mid-round is never saved, since the
                // save is the turn's own last step.
                _ = tokio::signal::ctrl_c() => {
                    if cancelled {
                        break TurnEnd::Interrupted;
                    }
                    cancelled = true;
                    link.cancel();
                }
            }
        }
    };
    // The turn narrates its own ending, so what it said last may still be queued.
    while let Ok(event) = events.try_recv() {
        show(&mut presenter, event);
    }
    // A plan approved mid-turn hands the mode back through the link.
    agent.mode = link.mode();
    end
}

fn show(presenter: &mut Presenter, event: AgentEvent) {
    let AgentEvent::Ask(ask) = event else {
        presenter.show(event);
        return;
    };
    match ask {
        // Approved, and in auto mode, because a plan nobody can review is either
        // approved or the run is over — and refusing it would leave the model told
        // not to start work with nobody to change that. Printed, because this is the
        // only place the plan's text appears at all, and a log that reads "plan
        // approved" without the plan is no record of what was agreed to.
        Ask::Plan { plan, reply } => {
            outln!("\nPlan (approved automatically — nobody is here to review it):\n{}", plan);
            let _ = reply.send(PlanDecision::ApproveWithAuto);
        }
        // Dropped unanswered, which the turn hears as "could not ask": a question
        // comes back telling the model to decide for itself, and a call the reviewer
        // gave no verdict on is refused. Never consent — the same rule as a REPL
        // with no terminal.
        Ask::User { .. } | Ask::Permission { .. } => {}
    }
}

enum Waited {
    Fired(Vec<crate::timers::Fired>),
    /// No timer is armed, so there is nothing to wait for.
    Nothing,
    Interrupted,
}

/// Sleep until an armed timer fires, or return at once when none is.
///
/// A recurring timer keeps the run alive for as long as it is armed. That is the point
/// of setting one in an unattended run — "poll CI every ten minutes until it is green" —
/// and the model was told the run waits on it, so cancelling it is how it says it is
/// done. Ctrl+C still ends the wait, and the log says what it is waiting for.
async fn wait_for_timers(agent: &Agent) -> Waited {
    if agent.server.next_timer_wake().is_none() {
        return Waited::Nothing;
    }
    outln!("\nWaiting for timers (Ctrl+C to stop):\n{}", agent.server.describe_timers());
    loop {
        let Some(wake) = agent.server.next_timer_wake() else {
            return Waited::Nothing;
        };
        tokio::select! {
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(wake)) => {}
            _ = tokio::signal::ctrl_c() => return Waited::Interrupted,
        }
        let fired = agent.server.take_due_timers();
        if !fired.is_empty() {
            return Waited::Fired(fired);
        }
    }
}

/// The text of the history's final message, when it is the assistant's.
fn last_reply(messages: &[Value]) -> Option<&str> {
    messages
        .last()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .map(|m| m.get("content").and_then(Value::as_str).unwrap_or_default())
}

/// What to say to a model whose turn ended looking unfinished, or `None` when it
/// looks done.
///
/// Three signs, checked in order of how sure they are. Unfinished items in its own
/// plan are the model's own statement that there is more to do. A reply ending on a
/// question is a model waiting for an answer that is not coming. And an empty reply
/// says nothing at all, which is not a summary of finished work. Anything else is
/// taken at its word: second-guessing a model that says it is done would mean asking
/// it again, and a model asked "are you sure?" tends to find something more to do.
fn follow_up(plan: &[TodoItem], reply: Option<&str>) -> Option<String> {
    let unfinished: Vec<&TodoItem> =
        plan.iter().filter(|item| item.status != "completed").collect();
    if !unfinished.is_empty() {
        let list = unfinished
            .iter()
            .map(|item| format!("- [{}] {}", item.status, item.task))
            .collect::<Vec<_>>()
            .join("\n");
        return Some(format!(
            "You stopped, but your plan still has unfinished items:\n{}\n\nNobody is \
             watching this run, so nobody will tell you to go on. Continue with them now. \
             If one cannot be done or turned out not to be needed, mark it completed with \
             todo_write and say why in your final summary.",
            list
        ));
    }

    let reply = reply.unwrap_or_default().trim();
    if reply.is_empty() {
        return Some(
            "Your reply was empty. If the task is finished, reply with a short summary of \
             what you did. If it is not, carry on with it."
                .to_string(),
        );
    }
    if announces_a_next_step(reply) {
        return Some(
            "You said what you would do next, but stopped without doing it. Nobody is watching \
             this run, so nothing will happen until you call a tool. Do it now, and carry on \
             until the task is done."
                .to_string(),
        );
    }
    let last_line = reply.lines().last().unwrap_or_default().trim();
    if last_line.ends_with('?') {
        return Some(
            "Nobody is watching this run, so that question will not be answered. Decide \
             it yourself, say in one line which way you went, and carry on until the task \
             is done. If it already is, reply with a short summary and nothing else."
                .to_string(),
        );
    }
    None
}

/// Whether a short reply ends by announcing what it will do instead of having done it — the
/// habit the module doc names first ("Now I will run the tests."). Short, because a summary of
/// finished work runs longer and may well end on a plan for later; "Let me know" is a
/// sign-off, not a plan. Measured case: a p2p_auth run's turn ended on "Let me check the
/// current state of the codebase and see what errors remain." and the run exited as finished.
const ANNOUNCEMENT_MAX: usize = 400;

fn announces_a_next_step(reply: &str) -> bool {
    static ANNOUNCES: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r"(?i)^(?:(?:now|next|first|then|ok|okay|alright|great)[,.!]?\s+)?(?:let me|let's|i'll|i will|i'm going to|i am going to|i need to|going to)\b",
        )
        .unwrap()
    });
    if reply.len() > ANNOUNCEMENT_MAX {
        return false;
    }
    // The last sentence: after the last ". ", "! " or line break. Not after every '.', which
    // would cut "tests/integration.rs." at the file extension.
    let text = reply.trim();
    let start = [text.rfind(". "), text.rfind("! "), text.rfind('\n')]
        .into_iter()
        .flatten()
        .max()
        .map_or(0, |at| at + 1);
    let last = text[start..].trim();
    ANNOUNCES.is_match(last) && !last.to_lowercase().starts_with("let me know")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(task: &str, status: &str) -> TodoItem {
        TodoItem { task: task.to_string(), status: status.to_string() }
    }

    #[test]
    fn a_finished_answer_ends_the_run() {
        let plan = [item("update deps", "completed"), item("commit", "completed")];
        assert_eq!(follow_up(&plan, Some("Updated 4 crates and committed.")), None);
        assert_eq!(follow_up(&[], Some("Done.")), None);
    }

    #[test]
    fn unfinished_plan_items_send_the_model_back_to_work() {
        let plan = [item("update deps", "completed"), item("commit", "pending")];
        let nudge = follow_up(&plan, Some("Dependencies updated.")).unwrap();
        assert!(nudge.contains("- [pending] commit"), "{nudge}");
        assert!(!nudge.contains("update deps"), "finished items are not repeated: {nudge}");
    }

    #[test]
    fn a_question_or_silence_is_not_a_finish() {
        let asked = follow_up(&[], Some("I found two outdated crates.\n\nShall I update them?"));
        assert!(asked.unwrap().contains("will not be answered"));
        assert!(follow_up(&[], Some("  \n")).unwrap().contains("empty"));
        assert!(follow_up(&[], None).is_some());
        // A question in the middle of a summary is not one waiting for an answer.
        assert_eq!(follow_up(&[], Some("Why was it pinned? An old bug. Unpinned it.")), None);
    }

    #[test]
    fn an_announced_next_step_is_not_a_finish() {
        for stopped in [
            "Let me check the current state of the codebase and see what errors remain.",
            "The handshake compiles. Now I will run the tests.",
            "Next, I'll fix the remaining two errors in tests/integration.rs.",
            "I need to update the imports first.",
        ] {
            let nudge = follow_up(&[], Some(stopped));
            assert!(nudge.is_some_and(|n| n.contains("stopped without doing it")), "{stopped}");
        }
        for finished in [
            "Done: all 6 tests pass three times in a row and clippy is clean.",
            "Fixed the parser. Let me know if you want the error messages changed.",
            // A long summary may end on a plan for later; it is still a summary.
            &format!("{} I'll leave the TLS support for a later version.", "Implemented the node. ".repeat(30)),
        ] {
            assert_eq!(follow_up(&[], Some(finished)), None, "{finished}");
        }
    }

    #[test]
    fn the_last_reply_is_read_only_off_an_assistant_message() {
        let history = vec![
            serde_json::json!({"role": "user", "content": "go"}),
            serde_json::json!({"role": "assistant", "content": "done"}),
        ];
        assert_eq!(last_reply(&history), Some("done"));
        assert_eq!(last_reply(&history[..1]), None);
        let null_content = [serde_json::json!({"role": "assistant", "content": null})];
        assert_eq!(last_reply(&null_content), Some(""));
    }

    #[test]
    fn the_task_carries_the_unattended_note() {
        let message = first_message(&std::env::temp_dir(), "  Update dependencies and commit ");
        assert!(message.starts_with("Update dependencies and commit\n\n("));
        assert!(message.contains("nobody will answer"));
    }
}
