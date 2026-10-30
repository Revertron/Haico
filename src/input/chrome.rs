//! The chrome around the input box: the activity bar above it, the status bar below,
//! the plan panel and the queued messages, and the one function that draws them.

use super::*;
// Everything the bottom of the screen shows besides what is being typed: what the
// model is busy with above the box, and what the session is set to below it.
pub(crate) struct Chrome<'a> {
    pub(crate) mode: Mode,
    pub(crate) busy: Option<&'a str>,
    // What the top bar says when nothing is running. Ignored while `busy` is set, since
    // what is happening now always beats what happened last.
    pub(crate) last: Option<&'a str>,
    // The right-hand end of the status bar: which model, and how full its context is.
    pub(crate) status: &'a str,
    // The plan as todo_write last left it, drawn above everything else. Empty until the
    // model writes one, and then the panel takes no rows at all.
    pub(crate) todos: &'a [TodoItem],
    // Messages typed while this was running that have not reached the model yet. Held
    // above the bar rather than written into the transcript, because the transcript is
    // what was said and these have not been said yet.
    pub(crate) queued: &'a [String],
    // How much of it to show. Owned by the caller, since it outlives any one draw.
    pub(crate) plan: PlanView,
}

// The chrome's colors. The bars are the only thing on screen that is painted rather than
// merely tinted, which is what makes them read as furniture instead of as output.
//
// A 256-color light blue rather than an RGB one: it lands the same on any terminal from
// the last twenty years, and the exact shade is not worth a truecolor requirement.
// Shared rather than private because the transcript borrows the shade for the line that
// reports a compressed chain of thought — that line is haico talking about the reply
// rather than part of it, and wearing the chrome's color is what says so. Kept as one
// constant so the two cannot drift apart into two nearly-identical blues.
pub(crate) const BAR_BACKGROUND: Color = Color::AnsiValue(153);
pub(super) const BAR_TEXT: Color = Color::Black;

// A message waiting its turn. Painted rather than tinted, like the bars — but only behind
// the text itself, and in a gray that says furniture rather than message: it is on screen
// because it has *not* happened yet, and the moment it does it moves into the transcript
// as an ordinary line.
const QUEUED_BACKGROUND: Color = Color::AnsiValue(252);
const QUEUED_TEXT: Color = Color::Black;

// Paint one row of chrome the whole way across.
//
// Two things this owes the writer. The row stops a column short of the width, because a
// row the terminal wrapped on its own is a row `out` cannot count and the region's height
// is wrong from then on — with a background that last cell simply stays unpainted. And
// line breaks and tabs become spaces, because a summary of a multi-line shell command
// would otherwise put line feeds inside a row that is supposed to be exactly one.
//
// Runs of spaces are left alone, and that is not an oversight: the gap that right-aligns
// the status bar is a run of spaces, and collapsing it flattens the bar to the left.
pub(super) fn bar(content: &str, width: usize) -> String {
    let room = width.saturating_sub(1);
    if room == 0 {
        return String::new();
    }

    let content: String = content
        .chars()
        .map(|ch| if ch.is_whitespace() { ' ' } else { ch })
        .collect();

    // Cut by characters: what reaches here is plain text, painted here rather than
    // carrying escapes of its own, so there is no sequence to cut in half.
    let content = fit(&content, room);

    let pad = room.saturating_sub(content.chars().count());
    format!(
        "{}{}{}{}{}",
        SetBackgroundColor(BAR_BACKGROUND),
        SetForegroundColor(BAR_TEXT),
        content,
        " ".repeat(pad),
        ResetColor
    )
}

// What the activity bar says while something is running: the label, plus how much reply
// has arrived so far once that is anything at all.
//
// Four characters to the token, the same rule of thumb `estimate_tokens` uses, so the two
// figures on screen cannot disagree about what a token is. Nothing is bracketed until the
// first token lands — through the silence before it, "(↓ 0)" says less than no bracket at
// all, and the silence before it is exactly what this exists for.
// Takes a count rather than a shared counter: it used to be read while the future
// filling it was still running, so it had to be an `AtomicUsize` behind a
// reference. Now every chunk passes through the pump, which counts them and calls
// this on the same thread — there is nothing left to share.
pub(crate) fn busy_label(busy: &str, characters: usize) -> String {
    match characters / 4 {
        0 => busy.to_string(),
        tokens if tokens < 1000 => format!("{} (↓ {})", busy, tokens),
        tokens => format!("{} (↓ {:.1}k)", busy, tokens as f64 / 1000.0),
    }
}

// The bar above the box: what is running, or what ran last.
//
// It used to exist only while something was running, which meant the input box jumped a
// row every time work started or stopped. Always drawn, so it does not.
fn activity_bar(chrome: &Chrome, width: usize) -> String {
    let text = chrome.busy.or(chrome.last).unwrap_or("ready");
    bar(&format!("○ {}", text), width)
}

// One row: the mode on the left, the model and context on the right. Kept a column
// short of the terminal width so it can never be the thing that wraps — a row the
// terminal wrapped on its own is a row the writer cannot count.
pub(super) fn status_bar(mode: Mode, status: &str, width: usize) -> String {
    let left = mode.label();
    let room = width.saturating_sub(1);

    // On a narrow terminal something has to give, and it is the front of the status:
    // a truncated model name still tells you which one it is, while a truncated
    // context figure is worse than none.
    let room_for_status = room.saturating_sub(left.chars().count() + 1);
    let length = status.chars().count();
    let status: String = if length > room_for_status {
        status.chars().skip(length - room_for_status).collect()
    } else {
        status.to_string()
    };

    let used = left.chars().count() + status.chars().count();
    let gap = room.saturating_sub(used).max(1);
    bar(&format!("{}{}{}", left, " ".repeat(gap), status), width)
}

// What was typed while something was running, one row each, above the bar.
//
// These used to go straight into the transcript, which read as though they had already
// been sent — a message queued behind a long tool call would sit in the scrollback for a
// minute or more looking exactly like one the model had already been given. Held here
// instead, in gray, until the round ends and they are actually pushed into the
// conversation.
//
// Flattened and cut to one row apiece for the same reason `bar` does it: a region row
// that turns out to be two is a row the writer cannot count.
pub(super) fn queued_rows(queued: &[String], width: usize) -> Vec<String> {
    let room = width.saturating_sub(1);
    if room == 0 {
        return Vec::new();
    }

    queued
        .iter()
        .map(|text| {
            let flattened: String = text
                .chars()
                .map(|ch| if ch.is_whitespace() { ' ' } else { ch })
                .collect();
            let content = fit(&format!("{}{}", PROMPT, flattened.trim()), room);
            format!(
                "{}{}{}{}",
                SetBackgroundColor(QUEUED_BACKGROUND),
                SetForegroundColor(QUEUED_TEXT),
                content,
                ResetColor
            )
        })
        .collect()
}

// The plan, pinned above the activity bar.
//
// todo_write used to print the list as ordinary output, which meant it scrolled away
// behind the next few tool calls — exactly when it is most use. Pinned, it is on screen
// for as long as the plan exists, and that is what the windowing pays for: a long plan
// would otherwise take the transcript's rows away for the rest of the turn.
//
// The ceiling on the windowed view, whatever the height of the terminal, heading and "…"
// lines included.
pub(super) const PLAN_MAX_ROWS: usize = 10;

// Cut a plain-text row down to the room it has, for the same reason `bar` does it: a row
// the terminal wraps on its own is a row the writer cannot count, and the region's height
// is wrong from then on.
pub(crate) fn fit(text: &str, columns: usize) -> String {
    let length = text.chars().count();
    if length <= columns || columns == 0 {
        return text.to_string();
    }
    text.chars().take(columns - 1).chain(['…']).collect()
}

pub(super) fn dim(text: String) -> String {
    format!("{}{}{}", SetForegroundColor(crate::out::DIM), text, ResetColor)
}

// `room` is how many rows the panel may take at most — what the box and the bars leave.
// Even the full view is bounded by it: the writer erases the region by counting rows back
// from the caret, so a region taller than the screen would have it walking back over rows
// that scrolled away.
pub(crate) fn plan_rows(
    todos: &[TodoItem],
    width: usize,
    room: usize,
    view: PlanView,
) -> Vec<String> {
    if todos.is_empty() || room == 0 {
        return Vec::new();
    }

    let budget = match view {
        PlanView::Collapsed => 1,
        // A third of what is free, because the transcript above is what the panel borrows
        // from, and never more than the ceiling however tall the terminal is.
        PlanView::Windowed => (room / 3).clamp(4, PLAN_MAX_ROWS),
        PlanView::Full => room,
    }
    .min(room);

    let columns = width.saturating_sub(1);
    let done = todos
        .iter()
        .filter(|item| item.status == "completed")
        .count();
    let mark = |item: &TodoItem| match item.status.as_str() {
        "completed" => "x",
        "in_progress" => "»",
        _ => " ",
    };

    // What the window is kept around: whatever is being worked on, or failing that the
    // first thing still to do. A finished list shows its end, which is the part that says
    // it is finished.
    let focus = todos
        .iter()
        .position(|item| item.status == "in_progress")
        .or_else(|| todos.iter().position(|item| item.status != "completed"))
        .unwrap_or(todos.len() - 1);

    let heading = format!("  Plan  {}/{}", done, todos.len());

    // The one-row form: asked for, or all there is room for. Below four rows a windowed
    // list is mostly ellipses, and this says the same thing in less space.
    if budget < 4 && todos.len() + 1 > budget {
        let item = &todos[focus];
        let line = format!("{}  [{}] {}", heading, mark(item), item.task);
        return vec![dim(fit(&line, columns))];
    }

    // One row of the budget goes to the heading, and one to each "…" that turns out to be
    // needed. Both are assumed and a row handed back when the window sits against an end,
    // which is shorter than solving it exactly and lands on the same answer.
    let body = budget - 1;
    let (mut start, mut end) = (0, todos.len());
    if todos.len() > body {
        let shown = body.saturating_sub(2).max(1);
        start = focus.saturating_sub(shown / 2).min(todos.len() - shown);
        end = start + shown;
        if start == 0 {
            end = (end + 1).min(todos.len());
        }
        if end == todos.len() {
            start = start.saturating_sub(1);
        }
    }

    let mut rows = vec![dim(fit(&heading, columns))];
    if start > 0 {
        rows.push(dim(fit(&format!("   … {} above", start), columns)));
    }
    for item in &todos[start..end] {
        let line = fit(&format!("   [{}] {}", mark(item), item.task), columns);
        // Only what is left to do is worth full brightness: the finished items are there
        // to say the plan is being followed, not to be read again.
        rows.push(if item.status == "completed" { dim(line) } else { line });
    }
    if end < todos.len() {
        rows.push(dim(fit(
            &format!("   … +{} more", todos.len() - end),
            columns,
        )));
    }
    rows
}

// How far an event asks the transcript to move, if it asks at all. Negative is backward,
// into what has already gone past.
//
// The wheel only ever arrives because the full-screen renderer asked the terminal for
// button reporting; under the inline renderer the terminal keeps the wheel to itself and
// `out::scroll` does nothing anyway. Home and End are already the editor's, so paging is
// on PageUp and PageDown, which nothing else wanted.
pub(crate) fn scroll_rows(event: &Event) -> Option<isize> {
    // A page is a screenful less two rows, so a couple of lines of context carry over and
    // the reader can see where they were.
    let page = || out::viewport_rows().saturating_sub(2).max(1) as isize;
    match event {
        Event::Mouse(mouse) => match mouse.kind {
            MouseEventKind::ScrollUp => Some(-3),
            MouseEventKind::ScrollDown => Some(3),
            _ => None,
        },
        Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
            KeyCode::PageUp => Some(-page()),
            KeyCode::PageDown => Some(page()),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn draw(editor: &Editor, chrome: &Chrome) {
    let width = terminal_width();
    let (mut rows, caret_row, caret_column) = editor.rows(PROMPT, width);

    // The command list opens upward, between the activity bar and the box, so the box and
    // everything under it stay where they were while it comes and goes.
    let suggestions = editor.suggestion_rows(width);
    let suggestion_count = suggestions.len();
    rows.splice(0..0, suggestions);

    rows.insert(0, activity_bar(chrome, width));
    rows.push(status_bar(chrome.mode, chrome.status, width));

    // Between the plan and the bar: near enough to the box to read as something the user
    // typed, above the bar because what is waiting to be sent is not what is running.
    let waiting = queued_rows(chrome.queued, width);
    let waiting_rows = waiting.len();
    rows.splice(0..0, waiting);

    // What is left for the plan once the box, the bars and the blank row above them have
    // had their rows, less two: one so the transcript never disappears entirely, and one
    // for a streamed line that has not ended yet, which is drawn above the region and
    // counts against the screen just the same.
    let room = terminal_height().saturating_sub(rows.len() + 3);
    let plan = plan_rows(chrome.todos, width, room, chrome.plan);

    // The plan sits above the activity bar, and one blank row sits above the lot: without
    // it the pinned block butts straight up against the transcript and the last line of
    // output reads as another row of chrome. Everything below them — the box included —
    // is that many rows further down than it used to be.
    let above = plan.len() + waiting_rows + suggestion_count + 2;
    rows.splice(0..0, plan);
    rows.insert(0, String::new());

    out::set_region(rows, caret_row + above, caret_column);
}
