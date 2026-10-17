//! The keyboard: one reader, and the line editor it feeds.
//!
//! rustyline used to own both, and it owned them by blocking: nothing else could
//! happen while a line was being typed, and nothing could be typed while anything
//! else was happening. Reading the keyboard on its own thread and handing the events
//! to an editor we drive ourselves is what makes the second half possible, and it is
//! also the only way to get an input box that grows over more than one line.

use crate::modes::Mode;
use crate::out;
use crate::tools::TodoItem;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind};
use crate::commands::{self, Args, Command, Suggestion};
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor};
use crossterm::tty::IsTty;
use std::fs;
use std::io;
use std::path::Path;
use tokio::sync::mpsc::{error::TryRecvError, unbounded_channel, UnboundedReceiver, UnboundedSender};

// The one thing that reads the keyboard.
//
// Raw mode is on for the whole session now, so a second reader anywhere would race
// this one for keystrokes and each would get an arbitrary half of what was typed.
// Everything that needs a key — the permission prompt, the plan review, the model
// picker — takes events from here instead of calling `event::read` itself.
pub(crate) struct Keys {
    events: UnboundedReceiver<Event>,
}

impl Keys {
    // `vt_input` says the console is delivering everything as escape sequences, which on
    // Windows is the price of the mouse reporting anything at all. Crossterm's Windows
    // reader has no escape parser — it maps one input record to one event — so in that
    // mode the byte stream has to be read and parsed here instead.
    pub(crate) fn start(vt_input: bool) -> Self {
        let (sender, events) = unbounded_channel();

        #[cfg(windows)]
        if vt_input {
            vt::start(sender);
            return Keys { events };
        }
        #[cfg(not(windows))]
        let _ = vt_input;

        // A plain thread rather than a task: `event::read` blocks, and blocking a
        // tokio worker would stall every other future on that thread. It ends when
        // the receiver goes, which is when the process is on its way out anyway.
        std::thread::spawn(move || loop {
            match event::read() {
                Ok(event) => {
                    if sender.send(event).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        });
        Keys { events }
    }

    pub(crate) async fn next(&mut self) -> Option<Event> {
        self.events.recv().await
    }

    // Throw away anything typed ahead. A prompt that has just been answered should not
    // also consume the keystrokes that arrived while the user was reading it.
    pub(crate) fn drain(&mut self) {
        while !matches!(self.events.try_recv(), Err(TryRecvError::Empty | TryRecvError::Disconnected)) {}
    }
}

pub(crate) fn stdin_is_a_terminal() -> bool {
    io::stdin().is_tty()
}

// Piped input has no keyboard: no raw mode, no region, no editing. Kept because
// scripting haico — feeding it a file of commands — is worth not breaking, and
// because the alternative is a program that only works when a human is watching.
pub(crate) fn read_piped_line() -> Option<String> {
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line.trim_end_matches(['\r', '\n']).to_string()),
    }
}

// What is typed after. The mode used to be spelled out here as well, which was one
// place too many once the status bar carries it.
pub(crate) const PROMPT: &str = "> ";

// What came out of the editor.
pub(crate) enum Input {
    Text(String),
    // Ctrl+C: the half-typed line is abandoned, the session is not.
    Interrupted,
    // Ctrl+D on an empty line, or the keyboard going away.
    Eof,
    // Shift+Tab. The editor does not own the mode, so it says so and lets the caller
    // decide whether cycling means anything where it is being used.
    CycleMode,
    // Ctrl+T, and the same arrangement: the editor does not own the plan panel either,
    // and the prompts that draw no panel simply ignore this.
    TogglePlan,
    // Ctrl+O: show what the transcript folded away, or fold it back. The state lives in the
    // writer rather than here, so unlike the two above this carries nothing — the caller
    // just tells `out` to flip it.
    ToggleExpand,
    // Nothing was typed: the deadline `read_command` was given came round, so a timer may
    // be due. The half-typed line stays in the editor for when the prompt comes back.
    Wake,
}

// How much of the plan the panel shows, cycled with Ctrl+T.
//
// The windowed default is a compromise, and both ends of it are sometimes what is wanted:
// the whole list while checking what the model set out to do, one line while reading the
// transcript underneath it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum PlanView {
    // One row: how far along the plan is, and what is being worked on.
    Collapsed,
    #[default]
    Windowed,
    // As much of it as the screen can spare, which is still not unbounded.
    Full,
}

impl PlanView {
    // Smallest, middle, largest, and round again. Starting from the middle, the first
    // press opens the panel up and the second folds it away.
    pub(crate) fn next(self) -> Self {
        match self {
            PlanView::Collapsed => PlanView::Windowed,
            PlanView::Windowed => PlanView::Full,
            PlanView::Full => PlanView::Collapsed,
        }
    }
}
mod chrome;
mod editor;
pub(crate) use chrome::*;
pub(crate) use editor::*;


// What a submitted entry looks like once it is part of the scrollback rather than
// something being edited.
pub(crate) fn echo(prompt: &str, text: &str) -> String {
    let indent = " ".repeat(prompt.chars().count());
    text.split('\n')
        .enumerate()
        .map(|(index, line)| {
            if index == 0 {
                format!("{}{}", prompt, line)
            } else {
                format!("{}{}", indent, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// Read one entry with nothing else going on. The editor is redrawn after every
// keystroke, which is cheap: it is a handful of rows, and the writer only touches the
// ones below the transcript.
pub(crate) async fn read_text(keys: &mut Keys, editor: &mut Editor, prompt: &str) -> Input {
    loop {
        let (rows, caret_row, caret_column) = editor.rows(prompt, terminal_width());
        out::set_region(rows, caret_row, caret_column);
        let Some(event) = keys.next().await else {
            return Input::Eof;
        };
        if let Some(delta) = scroll_rows(&event) {
            out::scroll(delta);
            continue;
        }
        match event {
            Event::Key(key) => match editor.key(key) {
                // This prompt draws no plan panel, so there is nothing to fold.
                Some(Input::TogglePlan) => {}
                Some(Input::ToggleExpand) => out::toggle_expand(),
                Some(input) => {
                    out::follow();
                    out::clear_region();
                    return input;
                }
                None => {}
            },
            Event::Paste(text) => editor.paste(&text),
            // Redrawn at the top of the loop with the new width.
            Event::Resize(..) => {}
            _ => {}
        }
    }
}

// The main prompt: the full bottom of the screen, and the one key that means something
// only here.
pub(crate) async fn read_command(
    keys: &mut Keys,
    editor: &mut Editor,
    mode: &mut Mode,
    status: &str,
    last: Option<&str>,
    todos: &[TodoItem],
    plan: &mut PlanView,
    wake: Option<std::time::Instant>,
) -> Input {
    // Made once rather than per keystroke: typing must not keep pushing the deadline back.
    let deadline = async {
        match wake {
            Some(at) => tokio::time::sleep_until(tokio::time::Instant::from_std(at)).await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(deadline);
    loop {
        draw(
            editor,
            &Chrome {
                mode: *mode,
                busy: None,
                last,
                status,
                todos,
                // Nothing can be waiting here: a queued message becomes the next prompt
                // rather than reaching this one.
                queued: &[],
                plan: *plan,
            },
        );
        let event = tokio::select! {
            event = keys.next() => event,
            _ = &mut deadline => return Input::Wake,
        };
        let Some(event) = event else {
            return Input::Eof;
        };
        if let Some(delta) = scroll_rows(&event) {
            out::scroll(delta);
            continue;
        }
        match event {
            Event::Key(key) => match editor.key(key) {
                Some(Input::CycleMode) => *mode = mode.next(),
                Some(Input::TogglePlan) => *plan = plan.next(),
                Some(Input::ToggleExpand) => out::toggle_expand(),
                Some(input) => {
                    // Whatever was just typed is about to be answered, and the answer goes
                    // on the end: a view still held back would miss its own reply.
                    out::follow();
                    out::clear_region();
                    return input;
                }
                None => {}
            },
            Event::Paste(text) => editor.paste(&text),
            Event::Resize(..) => {}
            _ => {}
        }
    }
}

// What to do with a key that arrives while the model is working.
pub(crate) enum WhileWorking {
    // Give up on this turn.
    Interrupt,
    // Typed and submitted: it goes in the queue for when the turn is over.
    Queued(String),
    // Anything else — still editing, or nothing that means something here.
    Nothing,
}

// One keystroke's worth of the live input box, for use in a select! arm alongside
// whatever the model is doing.
pub(crate) fn while_working(
    editor: &mut Editor,
    mode: &mut Mode,
    plan: &mut PlanView,
    event: Event,
) -> WhileWorking {
    // Reading back through the transcript while the model works is the whole point of
    // being able to scroll at all, so this comes before anything that might consume the
    // key. What arrives while the view is held back leaves it where it is.
    if let Some(delta) = scroll_rows(&event) {
        out::scroll(delta);
        return WhileWorking::Nothing;
    }

    match event {
        Event::Key(key) => {
            if key.kind == KeyEventKind::Press && key.code == KeyCode::Esc {
                return WhileWorking::Interrupt;
            }
            match editor.key(key) {
                Some(Input::Text(text)) => {
                    out::follow();
                    WhileWorking::Queued(text)
                }
                // Ctrl+C means the same thing here as Esc does.
                Some(Input::Interrupted) => WhileWorking::Interrupt,
                Some(Input::CycleMode) => {
                    *mode = mode.next();
                    WhileWorking::Nothing
                }
                // Nothing to report: the caller redraws after every key it is handed.
                Some(Input::TogglePlan) => {
                    *plan = plan.next();
                    WhileWorking::Nothing
                }
                Some(Input::ToggleExpand) => {
                    out::toggle_expand();
                    WhileWorking::Nothing
                }
                // Ctrl+D is for leaving, and leaving mid-turn is what Esc is for.
                // The editor never produces a wake; only `read_command`'s deadline does.
                Some(Input::Eof) | Some(Input::Wake) | None => WhileWorking::Nothing,
            }
        }
        Event::Paste(text) => {
            editor.paste(&text);
            WhileWorking::Nothing
        }
        _ => WhileWorking::Nothing,
    }
}

// Reading the Windows console as a byte stream.
//
// Only reached when the full-screen renderer has put the console into VT input mode, which
// it does because that is the only configuration where the wheel reports at all *and*
// Shift+drag still selects text. The cost is that every key now arrives as an escape
// sequence, and crossterm reports those a character at a time: an arrow key comes through
// as "[", "A", then Up, and a mouse report comes through as its characters and nothing
// else. Filtering that after the fact would mean guessing whether a "[" starts a sequence
// or is someone typing `arr[i]`, and guessing wrong eats pasted code. Parsing the bytes
// removes the guess.
//
// `terminput` is crossterm's own escape parser published separately, so this is the same
// parsing that happens on Unix, just applied where crossterm declines to.
#[cfg(windows)]
mod vt;


// Reading the clipboard, for the paste the terminal is no longer doing for us.
//
// Windows only, and deliberately so: this exists because VT input mode stops the console
// acting on Shift+Insert itself. Every other terminal still handles that key on its own
// and never forwards it, so there is nothing here to do and no dependency worth taking on
// to do it.
mod clipboard;


pub(crate) fn load_history(path: &Path) -> Vec<String> {
    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.replace("\\n", "\n").replace("\\\\", "\\"))
        .collect()
}

pub(crate) fn save_history(path: &Path, history: &[String]) {
    // The directory may not exist yet. It did not have to be made while the history was
    // `~/.haico_history` and its parent was the home directory; now that it lives inside
    // haico's own store, a first run with no config file has nothing to write into.
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let body: String = history
        .iter()
        .map(|entry| format!("{}\n", entry.replace('\\', "\\\\").replace('\n', "\\n")))
        .collect();
    let _ = fs::write(path, body);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyEventKind, KeyModifiers};

    fn press(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    fn type_text(editor: &mut Editor, text: &str) {
        for ch in text.chars() {
            editor.key(press(KeyCode::Char(ch), KeyModifiers::NONE));
        }
    }

    #[test]
    fn shift_enter_grows_the_entry_and_enter_submits_all_of_it() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "first");
        editor.key(press(KeyCode::Enter, KeyModifiers::SHIFT));
        type_text(&mut editor, "second");

        let (rows, caret_row, caret_column) = editor.rows("> ", 80);
        assert_eq!(rows, vec!["> first".to_string(), "  second".to_string()]);
        assert_eq!((caret_row, caret_column), (1, 8));

        match editor.key(press(KeyCode::Enter, KeyModifiers::NONE)) {
            Some(Input::Text(text)) => assert_eq!(text, "first\nsecond"),
            _ => panic!("Enter should submit"),
        }
        // And the editor is empty again afterwards.
        assert_eq!(editor.rows("> ", 80).0, vec!["> ".to_string()]);
    }

    #[test]
    fn a_long_line_wraps_and_the_caret_follows_it() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, &"x".repeat(12));

        // Width 10, prompt 2 wide: "> " plus 8 x's, then the remaining 4. One word with
        // nowhere to break still breaks at the edge, and the prompt's own space is not a
        // break — a row holding just ">" would be worse than the cut.
        let (rows, caret_row, caret_column) = editor.rows("> ", 10);
        assert_eq!(rows, vec!["> xxxxxxxx".to_string(), "xxxx".to_string()]);
        assert_eq!((caret_row, caret_column), (1, 4));
    }

    #[test]
    fn a_wrapped_line_breaks_between_words_and_the_caret_follows_it() {
        let mut editor = Editor::new(Vec::new());
        editor.paste("the quick brown fox jumps");

        // Width 20, prompt 2 wide: "brown" ends at column 17 and "fox" will not fit, so
        // the break is the space after it rather than four characters into the word.
        let (rows, caret_row, caret_column) = editor.rows("> ", 21);
        assert_eq!(
            rows,
            vec!["> the quick brown".to_string(), "fox jumps".to_string()]
        );
        assert_eq!((caret_row, caret_column), (1, 9));

        // Back to the "f" of "fox", which is where the second row starts.
        for _ in 0..9 {
            editor.key(press(KeyCode::Left, KeyModifiers::NONE));
        }
        let (.., caret_row, caret_column) = editor.rows("> ", 21);
        assert_eq!((caret_row, caret_column), (1, 0));

        // And one more, onto the space the break swallowed. It has no column of its own
        // on the row below, so it keeps the one at the end of the row it broke.
        editor.key(press(KeyCode::Left, KeyModifiers::NONE));
        let (.., caret_row, caret_column) = editor.rows("> ", 21);
        assert_eq!((caret_row, caret_column), (0, 17));
    }

    #[test]
    fn editing_keys_work_on_characters_not_bytes() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "привет мир");
        editor.key(press(KeyCode::Char('w'), KeyModifiers::CONTROL));
        assert_eq!(editor.rows("> ", 80).0, vec!["> привет ".to_string()]);

        editor.key(press(KeyCode::Backspace, KeyModifiers::NONE));
        editor.key(press(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0, vec!["> приве".to_string()]);

        editor.key(press(KeyCode::Home, KeyModifiers::NONE));
        editor.key(press(KeyCode::Delete, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0, vec!["> риве".to_string()]);
    }

    #[test]
    fn the_arrows_reach_the_history_only_past_the_edge_of_the_entry() {
        let mut editor = Editor::new(vec!["older".to_string(), "newer".to_string()]);
        type_text(&mut editor, "draft");
        editor.key(press(KeyCode::Enter, KeyModifiers::SHIFT));
        type_text(&mut editor, "second");

        // From the second row, Up is a cursor move.
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0.len(), 2);

        // From the first, it is the history — and what was being typed comes back.
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0, vec!["> newer".to_string()]);
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(editor.rows("> ", 80).0, vec!["> older".to_string()]);
        editor.key(press(KeyCode::Down, KeyModifiers::NONE));
        editor.key(press(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(
            editor.rows("> ", 80).0,
            vec!["> draft".to_string(), "  second".to_string()]
        );
    }

    #[test]
    fn a_pasted_block_becomes_one_entry_rather_than_several() {
        let mut editor = Editor::new(Vec::new());
        editor.paste("one\ntwo\r\nthree");
        match editor.key(press(KeyCode::Enter, KeyModifiers::NONE)) {
            Some(Input::Text(text)) => assert_eq!(text, "one\ntwo\nthree"),
            _ => panic!("Enter should submit"),
        }
    }

    #[test]
    fn the_status_bar_never_reaches_the_edge_of_the_screen() {
        // A row exactly as wide as the terminal wraps, and a wrapped row is one the
        // writer cannot count — so the bar has to stop a column short whatever is in it.
        for width in [20usize, 40, 80] {
            for status in ["", "m", "a-model-with-a-long-name  ctx 97%"] {
                let bar = status_bar(Mode::Auto, status, width);
                assert!(
                    crate::out::visible_width(&bar) <= width - 1,
                    "width {} overflowed with {:?}: {:?}",
                    width,
                    status,
                    bar
                );
            }
        }
    }

    #[test]
    fn the_activity_bar_counts_up_only_once_there_is_something_to_count() {
        // Work that produces no stream of its own says only what it is doing, and
        // so does a stream that has not delivered a token yet: the whole point is
        // the silence before the first one, and "(↓ 0)" would fill it with nothing.
        assert_eq!(busy_label("Thinking…", 0), "Thinking…");
        assert_eq!(busy_label("Thinking…", 3), "Thinking…");

        // Four characters to the token, the same rule `estimate_tokens` uses.
        assert_eq!(busy_label("Thinking…", 4), "Thinking… (↓ 1)");
        assert_eq!(busy_label("Thinking…", 2596), "Thinking… (↓ 649)");

        // Thousands become the short form rather than a five-digit number nobody reads.
        assert_eq!(busy_label("Thinking…", 4000), "Thinking… (↓ 1.0k)");
        assert_eq!(busy_label("Thinking…", 94_000), "Thinking… (↓ 23.5k)");
    }

    #[test]
    fn the_box_sits_between_the_line_above_it_and_the_bar_below() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "hello");

        let drawn = |chrome: &Chrome| {
            let _guard = crate::out::capture_guard();
            crate::out::start_capture();
            draw(&editor, chrome);
            crate::out::take_capture()
        };

        let written = drawn(&Chrome {
            mode: Mode::Plan,
            busy: Some("thinking"),
            last: Some("ran cargo build"),
            status: "a-model  ctx 3%",
            todos: &[],
            queued: &[],
            plan: PlanView::Windowed,
        });

        let busy = written.find("○ thinking").expect("the busy line");
        let box_row = written.find("> hello").expect("the input box");
        // The two spaces before "ctx" are the separator status_line puts there. Only line
        // breaks are flattened, so a run of spaces has to survive the bar intact — the gap
        // that right-aligns this one is made of them.
        let bar = written.rfind("a-model  ctx 3%").expect("the status bar");
        assert!(busy < box_row && box_row < bar, "wrong order in {:?}", written);

        // Both bars are painted, and black on light blue is what says "chrome" rather
        // than "something the model said".
        let painted = format!("{}{}", SetBackgroundColor(BAR_BACKGROUND), SetForegroundColor(BAR_TEXT));
        assert_eq!(written.matches(&painted).count(), 2, "both bars paint: {:?}", written);

        // Nothing running: the top bar stays on screen showing what last happened, so the
        // box never jumps a row between turns.
        let idle = drawn(&Chrome {
            mode: Mode::Plan,
            busy: None,
            last: Some("ran cargo build"),
            status: "a-model  ctx 3%",
            todos: &[],
            queued: &[],
            plan: PlanView::Windowed,
        });
        assert!(idle.contains("○ ran cargo build"), "got {:?}", idle);

        // ...and before anything has happened at all, it still says something.
        let fresh = drawn(&Chrome {
            mode: Mode::Plan,
            busy: None,
            last: None,
            status: "a-model  ctx 3%",
            todos: &[],
            queued: &[],
            plan: PlanView::Windowed,
        });
        assert!(fresh.contains("○ ready"), "got {:?}", fresh);
    }


    #[test]
    fn an_unsent_message_comes_back_to_the_box_ahead_of_what_is_half_typed() {
        // Nothing being typed: it simply lands, ready to send or edit.
        let mut editor = Editor::new(Vec::new());
        editor.restore("what about the tests");
        let (rows, caret_row, caret_column) = editor.rows(PROMPT, 80);
        assert_eq!(rows, vec!["> what about the tests".to_string()]);
        assert_eq!((caret_row, caret_column), (0, 22));

        // Something being typed: the unsent message goes in front of it rather than
        // through the middle of it, and the caret stays with the new text.
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "half typed");
        editor.restore("first message");
        let (rows, caret_row, caret_column) = editor.rows(PROMPT, 80);
        assert_eq!(
            rows,
            vec!["> first message".to_string(), "  half typed".to_string()]
        );
        assert_eq!((caret_row, caret_column), (1, 2));

        // Two of them keep their own lines, so Enter still sends what was typed.
        let mut editor = Editor::new(Vec::new());
        editor.restore("one\ntwo");
        let (rows, ..) = editor.rows(PROMPT, 80);
        assert_eq!(rows, vec!["> one".to_string(), "  two".to_string()]);
    }
    #[test]
    fn a_queued_message_waits_above_the_bar_rather_than_in_the_transcript() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "hello");

        let _guard = crate::out::capture_guard();
        crate::out::start_capture();
        draw(
            &editor,
            &Chrome {
                mode: Mode::Normal,
                busy: Some("thinking"),
                last: None,
                status: "a-model  ctx 3%",
                todos: &[],
                queued: &["ask about the tests".to_string()],
                plan: PlanView::Windowed,
            },
        );
        let written = crate::out::take_capture();

        let waiting = written
            .find("> ask about the tests")
            .expect("the queued message");
        let bar = written.find("○ thinking").expect("the activity bar");
        let box_row = written.rfind("> hello").expect("the input box");
        assert!(
            waiting < bar && bar < box_row,
            "wrong order in {:?}",
            written
        );
    }

    #[test]
    fn a_queued_message_takes_exactly_one_row_however_it_was_typed() {
        let rows = queued_rows(&["one\ntwo\nthree".to_string(), "x".repeat(200)], 40);
        assert_eq!(rows.len(), 2, "one row each: {:?}", rows);
        for row in &rows {
            // A region row that turns out to be two is a row the writer cannot count.
            assert!(!row.contains('\n'), "more than one row: {:?}", row);
            assert!(crate::out::visible_width(row) < 40, "too wide: {:?}", row);
        }
        assert!(rows[0].contains("one two three"), "got {:?}", rows[0]);
    }

    fn todo(task: &str, status: &str) -> TodoItem {
        TodoItem {
            task: task.to_string(),
            status: status.to_string(),
        }
    }

    // A plan long enough that no view can show all of it, with the work part way down.
    fn long_plan() -> Vec<TodoItem> {
        let mut plan: Vec<TodoItem> = (0..20)
            .map(|index| todo(&format!("step {}", index), "completed"))
            .collect();
        plan[12].status = "in_progress".to_string();
        for item in plan.iter_mut().skip(13) {
            item.status = "pending".to_string();
        }
        plan
    }

    #[test]
    fn the_plan_panel_stays_short_and_keeps_the_current_item_in_view() {
        // Nothing pinned until there is a plan: the panel costs no rows at all.
        assert!(plan_rows(&[], 80, 20, PlanView::Windowed).is_empty());

        let short = vec![
            todo("write the config", "completed"),
            todo("write the server", "in_progress"),
            todo("write main", "pending"),
        ];
        let rows = plan_rows(&short, 80, 20, PlanView::Windowed);
        assert_eq!(rows.len(), 4, "a heading and one row each: {:?}", rows);
        assert!(rows[0].contains("Plan  1/3"), "got {:?}", rows[0]);
        assert!(rows[2].contains("[»] write the server"), "got {:?}", rows[2]);

        // Longer than the panel: windowed around whatever is in progress, with both ends
        // saying how much is out of sight rather than silently dropping it.
        let long = long_plan();
        for room in [6usize, 20, 60] {
            let rows = plan_rows(&long, 80, room, PlanView::Windowed);
            let budget = (room / 3).clamp(4, PLAN_MAX_ROWS).min(room);
            assert!(rows.len() <= budget, "{} rows for room {}: {:?}", rows.len(), room, rows);
            assert!(
                rows.iter().any(|row| row.contains("step 12")),
                "the item being worked on has to be in the window: {:?}",
                rows
            );
            assert!(rows.iter().any(|row| row.contains("above")), "{:?}", rows);
            assert!(rows.iter().any(|row| row.contains("more")), "{:?}", rows);
        }

        // And a row that wrapped would be a row the writer cannot count.
        let wide = vec![todo(&"a very long task ".repeat(20), "pending")];
        for row in plan_rows(&wide, 40, 20, PlanView::Windowed) {
            assert!(crate::out::visible_width(&row) < 40, "too wide: {:?}", row);
        }
    }

    #[test]
    fn ctrl_t_cycles_the_panel_between_its_three_sizes() {
        let mut editor = Editor::new(Vec::new());
        let mut view = PlanView::default();
        let mut cycle = || {
            match editor.key(press(KeyCode::Char('t'), KeyModifiers::CONTROL)) {
                Some(Input::TogglePlan) => view = view.next(),
                other => panic!("Ctrl+T should ask for the panel, got {:?}", other.is_some()),
            }
            view
        };
        // The default is the middle size; the first press opens it up, the second folds
        // it away, and the third is back where it started.
        assert_eq!(PlanView::default(), PlanView::Windowed);
        assert_eq!(cycle(), PlanView::Full);
        assert_eq!(cycle(), PlanView::Collapsed);
        assert_eq!(cycle(), PlanView::Windowed);

        let long = long_plan();

        // Folded away: one row, whatever the plan and whatever the room, and it still
        // names what is being worked on.
        let rows = plan_rows(&long, 80, 40, PlanView::Collapsed);
        assert_eq!(rows.len(), 1, "got {:?}", rows);
        assert!(rows[0].contains("Plan  12/20"), "got {:?}", rows[0]);
        assert!(rows[0].contains("[»] step 12"), "got {:?}", rows[0]);

        // Opened up: all twenty and the heading, since there is room for them.
        let rows = plan_rows(&long, 80, 40, PlanView::Full);
        assert_eq!(rows.len(), 21, "got {:?}", rows);
        assert!(rows[1].contains("[x] step 0"), "got {:?}", rows[1]);
        assert!(rows[20].contains("[ ] step 19"), "got {:?}", rows[20]);

        // ...but never past what the screen can spare. The full view is bounded by the
        // room it is given, because a region taller than the terminal scrolls its own top
        // away and the writer then erases rows that are no longer there.
        for room in [1usize, 2, 3, 7, 12] {
            for view in [PlanView::Collapsed, PlanView::Windowed, PlanView::Full] {
                let rows = plan_rows(&long, 80, room, view);
                assert!(
                    rows.len() <= room,
                    "{:?} took {} rows out of {}: {:?}",
                    view,
                    rows.len(),
                    room,
                    rows
                );
            }
        }
    }

    #[test]
    fn the_plan_sits_above_the_activity_bar_and_the_caret_moves_down_with_it() {
        let mut editor = Editor::new(Vec::new());
        type_text(&mut editor, "hello");
        let todos = vec![todo("write the server", "in_progress")];

        let _guard = crate::out::capture_guard();
        crate::out::start_capture();
        draw(
            &editor,
            &Chrome {
                mode: Mode::Normal,
                busy: Some("thinking"),
                last: None,
                status: "a-model  ctx 3%",
                todos: &todos,
                queued: &[],
                plan: PlanView::Windowed,
            },
        );
        let written = crate::out::take_capture();

        let plan = written.find("Plan  0/1").expect("the plan");
        let busy = written.find("○ thinking").expect("the busy line");
        let box_row = written.find("> hello").expect("the input box");
        assert!(plan < busy && busy < box_row, "wrong order in {:?}", written);

        // Two plan rows, the bar, the box, the status bar: the caret belongs on the fourth
        // row of five, so the writer walks back exactly one from the last and along to
        // column 7. An offset that forgot the plan is invisible in the text above and
        // obvious on screen.
        use crossterm::cursor::{MoveRight, MoveToColumn, MoveUp};
        let landing = format!("{}{}{}", MoveUp(1), MoveToColumn(0), MoveRight(7));
        assert!(written.contains(&landing), "the caret lands in the wrong place: {:?}", written);
    }

    #[test]
    fn a_bar_stays_one_row_whatever_it_is_handed() {
        // A multi-line shell command is a plausible summary, and a line feed inside a row
        // the writer counts as one is how the region's height goes wrong.
        for width in [20usize, 40, 80] {
            for content in [
                "",
                "○ ran cargo build",
                "○ ran for f in *.rs\ndo\n  echo $f\ndone",
                &"○ ran ".to_string().repeat(40),
            ] {
                let row = bar(content, width);
                assert!(!row.contains('\n'), "line feed in {:?}", row);
                // Painted the whole way across, and never wider than the writer allows.
                assert_eq!(
                    crate::out::visible_width(&row),
                    width - 1,
                    "width {} with {:?}: {:?}",
                    width,
                    content,
                    row
                );
            }
        }
    }

    #[test]
    fn history_survives_a_multi_line_entry() {
        let directory = std::env::temp_dir().join("haico-history-test");
        let _ = fs::create_dir_all(&directory);
        let path = directory.join("history");

        let entries = vec!["plain".to_string(), "two\nlines".to_string(), "back\\slash".to_string()];
        save_history(&path, &entries);
        assert_eq!(load_history(&path), entries);

        let _ = fs::remove_file(&path);
    }

    fn with_commands(history: Vec<String>) -> Editor {
        let mut editor = Editor::new(history);
        editor.set_commands(commands::available(&[]));
        editor
    }

    fn listed(editor: &Editor) -> Vec<String> {
        editor
            .suggestion_rows(120)
            .iter()
            .map(|row| crate::util::strip_ansi(row))
            .collect()
    }

    fn submitted(input: Option<Input>) -> Option<String> {
        match input {
            Some(Input::Text(text)) => Some(text),
            _ => None,
        }
    }

    #[test]
    fn a_slash_opens_the_list_and_enter_runs_the_picked_command() {
        let mut editor = with_commands(Vec::new());
        type_text(&mut editor, "/com");
        let rows = listed(&editor);
        assert!(rows[0].starts_with("› /compact"), "{:?}", rows);
        assert!(rows[0].contains("Replace the history"), "{:?}", rows);

        let sent = submitted(editor.key(press(KeyCode::Enter, KeyModifiers::NONE)));
        assert_eq!(sent.as_deref(), Some("/compact"));
        assert!(listed(&editor).is_empty(), "the list outlived the entry");
        assert_eq!(editor.history(), ["/compact".to_string()]);
    }

    #[test]
    fn a_command_that_needs_an_argument_is_put_in_the_box_instead() {
        let mut editor = with_commands(Vec::new());
        type_text(&mut editor, "/ad");
        assert!(submitted(editor.key(press(KeyCode::Enter, KeyModifiers::NONE))).is_none());
        assert_eq!(editor.lines, ["/add ".to_string()]);
        assert!(listed(&editor).is_empty(), "a space ends the command, and the list with it");

        type_text(&mut editor, "src");
        let sent = submitted(editor.key(press(KeyCode::Enter, KeyModifiers::NONE)));
        assert_eq!(sent.as_deref(), Some("/add src"));
    }

    #[test]
    fn the_arrows_move_the_pick_and_tab_fills_it_in() {
        let mut editor = with_commands(Vec::new());
        type_text(&mut editor, "/re");
        let rows = listed(&editor);
        assert!(rows[0].starts_with("› /reasoning") && rows[1].starts_with("  /resume"), "{:?}", rows);

        editor.key(press(KeyCode::Down, KeyModifiers::NONE));
        assert!(listed(&editor)[1].starts_with("› /resume"));
        // Up from the top wraps round to the bottom, and back.
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert!(listed(&editor).last().unwrap().starts_with("› "));
        editor.key(press(KeyCode::Down, KeyModifiers::NONE));
        editor.key(press(KeyCode::Down, KeyModifiers::NONE));

        // /resume takes nothing after it, so it goes in without a space.
        editor.key(press(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(editor.lines, ["/resume".to_string()]);
    }

    #[test]
    fn typing_more_starts_the_pick_from_the_top_again() {
        let mut editor = with_commands(Vec::new());
        type_text(&mut editor, "/c");
        editor.key(press(KeyCode::Down, KeyModifiers::NONE));
        type_text(&mut editor, "o");
        assert!(listed(&editor)[0].starts_with("› "));
    }

    #[test]
    fn the_list_shows_five_rows_and_scrolls_to_keep_the_pick_in_view() {
        let mut editor = with_commands(Vec::new());
        type_text(&mut editor, "/");
        assert_eq!(listed(&editor).len(), SUGGESTION_ROWS);
        for _ in 0..7 {
            editor.key(press(KeyCode::Down, KeyModifiers::NONE));
        }
        let rows = listed(&editor);
        assert_eq!(rows.len(), SUGGESTION_ROWS);
        assert!(rows.last().unwrap().starts_with("› "), "{:?}", rows);
    }

    #[test]
    fn esc_closes_the_list_until_the_entry_changes() {
        let mut editor = with_commands(Vec::new());
        type_text(&mut editor, "/com");
        editor.key(press(KeyCode::Esc, KeyModifiers::NONE));
        assert!(listed(&editor).is_empty());
        // Closed, Enter sends what was typed rather than what was picked.
        let mut closed = with_commands(Vec::new());
        type_text(&mut closed, "/com");
        closed.key(press(KeyCode::Esc, KeyModifiers::NONE));
        let sent = submitted(closed.key(press(KeyCode::Enter, KeyModifiers::NONE)));
        assert_eq!(sent.as_deref(), Some("/com"));

        type_text(&mut editor, "p");
        assert!(!listed(&editor).is_empty());
    }

    #[test]
    fn history_is_walked_past_a_command_without_the_list_taking_the_arrows() {
        let mut editor = with_commands(vec!["first".to_string(), "/compact".to_string()]);
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(editor.lines, ["/compact".to_string()]);
        assert!(listed(&editor).is_empty());
        editor.key(press(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(editor.lines, ["first".to_string()]);
    }

    #[test]
    fn a_path_or_a_dialog_reply_is_not_taken_for_a_command() {
        let mut editor = with_commands(Vec::new());
        type_text(&mut editor, "/etc/hosts");
        assert!(listed(&editor).is_empty());
        let sent = submitted(editor.key(press(KeyCode::Enter, KeyModifiers::NONE)));
        assert_eq!(sent.as_deref(), Some("/etc/hosts"));

        // The dialogs build their editors without a command list.
        let mut dialog = Editor::new(Vec::new());
        type_text(&mut dialog, "/com");
        assert!(listed(&dialog).is_empty());
        let sent = submitted(dialog.key(press(KeyCode::Enter, KeyModifiers::NONE)));
        assert_eq!(sent.as_deref(), Some("/com"));
    }

    #[test]
    fn the_list_sits_between_the_activity_bar_and_the_box() {
        let mut editor = with_commands(Vec::new());
        type_text(&mut editor, "/com");

        let _guard = crate::out::capture_guard();
        crate::out::start_capture();
        draw(
            &editor,
            &Chrome {
                mode: Mode::Normal,
                busy: None,
                last: None,
                status: "a-model  ctx 3%",
                todos: &[],
                queued: &[],
                plan: PlanView::Windowed,
            },
        );
        let written = crate::out::take_capture();

        let bar = written.find("○ ready").expect("the activity bar");
        let list = written.find("Replace the history").expect("the list");
        let box_row = written.find("> /com").expect("the input box");
        assert!(bar < list && list < box_row, "wrong order in {:?}", written);

        // The caret still lands in the box, one row up from the status bar: an offset that
        // forgot the list would put it in the list instead.
        use crossterm::cursor::{MoveRight, MoveToColumn, MoveUp};
        let landing = format!("{}{}{}", MoveUp(1), MoveToColumn(0), MoveRight(6));
        assert!(written.contains(&landing), "the caret lands in the wrong place: {:?}", written);
    }
}
