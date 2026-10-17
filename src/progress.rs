//! What a tool says while it runs, delivered to whichever frontend started the call.
//!
//! `research` and `library` each run a loop of their own inside one tool call, and
//! minutes of it on a local model; a line per lookup and the answer as it lands are
//! what show the call is going anywhere. They used to `outln!` those lines straight to
//! the terminal from inside the tool. That put terminal output in the core — the GUI
//! never saw any of it — and forced `run_turn` to name the two tools, so it could skip
//! the result summary that would repeat the answer they had just printed.
//!
//! A tool body cannot be handed the turn's event channel: its signature is rmcp's, and
//! the same body answers MCP hosts, which have no channel at all. So the channel rides
//! in a task-local instead, set by the turn around the one call (`reporting`), and the
//! tool speaks into it with `step` and `answer`. Outside such a scope — MCP server
//! mode, a sub-agent's own calls, a test calling the tool directly — both are silent,
//! which is exactly what the `interactive` checks they replace used to decide.

use crate::turn::{EventSink, Notice};
use std::cell::Cell;
use std::future::Future;

struct Sink {
    tx: EventSink,
    // Whether the call showed its own answer, so the turn does not show a summary of it
    // again directly underneath.
    answered: Cell<bool>,
}

tokio::task_local! {
    static SINK: Sink;
}

/// Run `call` with what it reports going to `tx`. The flag says whether it showed its
/// own answer.
pub(crate) async fn reporting<F: Future>(tx: &EventSink, call: F) -> (F::Output, bool) {
    let sink = Sink { tx: tx.clone(), answered: Cell::new(false) };
    SINK.scope(sink, async {
        let output = call.await;
        (output, SINK.with(|sink| sink.answered.get()))
    })
    .await
}

// Said to the turn listening, if one is. A frontend that has gone away is no reason for a
// tool to stop, so a failed send is dropped as quietly as an absent scope.
fn tell(notice: Notice) {
    let _ = SINK.try_with(|sink| {
        let _ = sink.tx.send(notice.into());
    });
}

/// One step of a call's own loop: a search, a page, a section read.
pub(crate) fn step(line: impl Into<String>) {
    tell(Notice::LookupStep(line.into()));
}

/// The answer a call's loop arrived at, shown whole as it lands.
pub(crate) fn answer(label: &'static str, answer: &str) {
    let _ = SINK.try_with(|sink| sink.answered.set(true));
    tell(Notice::LookupAnswer { label, answer: answer.to_string() });
}

/// The task list as `todo_write` just set it. Whether to show it — the REPL pins it
/// above the input box already — is the frontend's call.
pub(crate) fn plan(todos: &[crate::tools::TodoItem]) {
    tell(Notice::PlanUpdated(todos.to_vec()));
}

/// The scope's channel, for work the call hands to a task of its own — `run_command`'s
/// output pumps — which the task-local does not follow into.
pub(crate) fn current() -> Option<Reporter> {
    SINK.try_with(|sink| Reporter(sink.tx.clone())).ok()
}

#[derive(Clone)]
pub(crate) struct Reporter(EventSink);

impl Reporter {
    /// One line of a running command's output, as it arrives.
    pub(crate) fn output(&self, line: &str) {
        let _ = self.0.send(Notice::CommandOutput(line.to_string()).into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::AgentEvent;

    #[tokio::test]
    async fn a_tool_speaks_to_the_turn_that_called_it_and_to_nobody_else() {
        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        let ((), answered) = reporting(&tx, async {
            step("Searching: tokio select");
            answer("Library answer", "use biased;");
        })
        .await;
        assert!(answered);
        let said: Vec<AgentEvent> = std::iter::from_fn(|| events.try_recv().ok()).collect();
        assert!(matches!(&said[..], [
            AgentEvent::Notice(Notice::LookupStep(step)),
            AgentEvent::Notice(Notice::LookupAnswer { label: "Library answer", answer }),
        ] if step == "Searching: tokio select" && answer == "use biased;"));

        // A call made with no turn listening — MCP mode, a sub-agent — says nothing, and
        // does not fail for having nobody to say it to.
        step("unheard");
        answer("Library answer", "unheard");
        assert!(events.try_recv().is_err());

        // And a call that shows no answer of its own leaves the summary to the turn.
        let ((), answered) = reporting(&tx, async { step("Searching") }).await;
        assert!(!answered);
    }
}
