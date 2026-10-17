use super::{Event, UnboundedSender};
use std::io::Read;

pub(super) fn start(sender: UnboundedSender<Event>) {
    let sizes = sender.clone();
    std::thread::spawn(move || read_stream(sender));
    std::thread::spawn(move || watch_size(sizes));
}

fn read_stream(sender: UnboundedSender<Event>) {
    let mut stdin = std::io::stdin();
    let mut chunk = [0u8; 1024];
    let mut buffer: Vec<u8> = Vec::new();

    loop {
        // The mode has to be right at the moment the read starts, not merely at startup:
        // anything sharing this console can leave it in cooked mode, and a read that
        // begins there is echoed by the console and returns nothing until Enter.
        crate::out::reassert_input_mode();
        let read = match stdin.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(read) => read,
        };
        // A read that filled the chunk did not reach the end of what was typed — the rest
        // is already buffered behind it — and the cut falls wherever the boundary lands,
        // including between an Escape and the sequence it introduces. That matters for one
        // rule below and nothing else.
        let cut_short = read == chunk.len();
        if !feed(&mut buffer, &chunk[..read], cut_short, &sender) {
            return;
        }
    }
}

// Everything that becomes of a chunk once it has been read, and all of the reader the
// tests can reach — a console is not something they have. `false` means the receiver has
// gone and there is nobody left to read for.
//
// `cut_short` is the one thing this cannot work out for itself, and the Escape rule below
// is the only thing that wants it.
fn feed(
    buffer: &mut Vec<u8>,
    bytes: &[u8],
    cut_short: bool,
    sender: &UnboundedSender<Event>,
) -> bool {
    buffer.extend_from_slice(bytes);

    loop {
        if buffer.is_empty() {
            return true;
        }
        // An Escape all on its own is an Escape; an Escape with anything behind it
        // is the start of a sequence. The console writes a sequence in one piece, so an
        // Escape that is all the buffer holds is one that arrived alone — which is what
        // lets Esc interrupt a turn without waiting on a timer.
        //
        // Unless the read was cut short, in which case an Escape at the end of the buffer
        // is as likely to be the head of a sequence whose tail is in the next read: taking
        // it for a keypress would interrupt the turn *and* type the remainder —
        // `[<64;65;59M` — into the input box. So it waits for the rest of itself instead.
        // The cost is a real Esc pressed as the last byte of a burst that exactly filled
        // the chunk, which waits for the next key rather than acting; pressing it again is
        // then a short read, and works.
        if !cut_short && buffer.len() == 1 && buffer[0] == 0x1B {
            buffer.clear();
            return sender.send(Event::Key(super::KeyCode::Esc.into())).is_ok();
        }
        match take_event(buffer) {
            Some(Some(event)) => {
                if sender.send(event).is_err() {
                    return false;
                }
            }
            // Consumed, but nothing haico has a use for.
            Some(None) => {}
            // Incomplete: the rest of it is still on its way.
            None => return true,
        }
    }
}

// One event off the front of the buffer, or `None` while it is still incomplete.
//
// The shortest prefix that parses is the event, and is also how many bytes it took.
// Asking about the whole buffer instead would be wrong in both directions: `parse_from`
// ignores trailing bytes, so it would swallow the keystroke behind this one.
fn take_event(buffer: &mut Vec<u8>) -> Option<Option<Event>> {
    // A leading Escape is never the whole event here — the caller has already dealt
    // with the case where it is — so the search starts past it.
    let first = if buffer[0] == 0x1B { 2 } else { 1 };
    for end in first..=buffer.len() {
        match terminput::Event::parse_from(&buffer[..end]) {
            Ok(Some(parsed)) => {
                buffer.drain(..end);
                return Some(
                    terminput_crossterm::to_crossterm(parsed)
                        .ok()
                        .map(restore_shift),
                );
            }
            Ok(None) => continue,
            Err(_) => {
                // Nothing beginning here can ever parse, so it goes rather than
                // blocking every keystroke that follows it.
                buffer.drain(..1);
                return Some(None);
            }
        }
    }
    None
}

// The one thing the byte stream cannot say, asked of Windows directly.
//
// Enter is the single byte 0x0D whether or not Shift was held. The legacy encoding has
// a modifier field only for keys sent as CSI sequences — that is how `ESC[1;5D` manages
// to mean Ctrl+Left — and a key sent as a bare control character has nowhere to put
// one. So the state is gone before any parser could see it, and no amount of parsing
// brings it back.
//
// But it is only gone from the *encoding*. The key is still physically down: this runs
// within microseconds of the console writing the byte, and nobody releases Shift that
// fast. `GetAsyncKeyState` reports the live physical state and, unlike `GetKeyState`,
// needs no message queue — which a console program does not have.
//
// Deliberately only Enter. Every other key carries its own modifiers correctly, and
// second-guessing state we already have would be strictly worse than trusting it.
fn restore_shift(event: Event) -> Event {
    let Event::Key(mut key) = event else {
        return event;
    };
    if key.code == super::KeyCode::Enter
        && key.modifiers.is_empty()
        && shift_is_down()
    {
        key.modifiers |= super::KeyModifiers::SHIFT;
    }
    Event::Key(key)
}

fn shift_is_down() -> bool {
    extern "system" {
        fn GetAsyncKeyState(key: i32) -> i16;
    }
    const VK_SHIFT: i32 = 0x10;
    // The high bit is "down right now"; the low bit is "was pressed since last asked"
    // and is not what this wants.
    unsafe { GetAsyncKeyState(VK_SHIFT) as u16 & 0x8000 != 0 }
}

// VT input reports keys and the mouse and says nothing about the window changing size,
// and the console records that used to carry that are no longer being read. Polling is
// the whole of the fix: dragging a window edge is a human-speed thing, and four times a
// second is faster than anyone can do it.
fn watch_size(sender: UnboundedSender<Event>) {
    let mut last = crossterm::terminal::size().unwrap_or((80, 24));
    loop {
        std::thread::sleep(std::time::Duration::from_millis(250));
        let Ok(size) = crossterm::terminal::size() else {
            continue;
        };
        if size != last {
            last = size;
            if sender.send(Event::Resize(size.0, size.1)).is_err() {
                return;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::{feed, take_event};
    use crossterm::event::{Event, KeyCode, MouseEventKind};

    // One chunk through the reader, and everything it sent. No console is involved, which
    // is the whole reason `feed` is a function of its own.
    fn feed_chunk(buffer: &mut Vec<u8>, bytes: &[u8], cut_short: bool) -> Vec<Event> {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        assert!(feed(buffer, bytes, cut_short, &sender), "the receiver is still here");
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        events
    }

    // The console does not always hand a sequence over in one piece. A read that fills the
    // chunk cuts the burst wherever the boundary falls, and it can fall directly after the
    // Escape — at which point calling it a keypress interrupts the turn *and* types the
    // rest of the report into the input box.
    #[test]
    fn an_escape_left_by_a_cut_read_waits_for_the_rest_of_its_sequence() {
        let mut buffer = Vec::new();

        let events = feed_chunk(&mut buffer, b"\x1b", true);
        assert!(events.is_empty(), "{:?}", events);
        assert_eq!(buffer, b"\x1b", "the Escape was acted on rather than held");

        let events = feed_chunk(&mut buffer, b"[<64;65;59M", false);
        assert_eq!(events.len(), 1, "{:?}", events);
        assert!(matches!(events[0], Event::Mouse(_)), "{:?}", events[0]);
        assert!(buffer.is_empty());
    }

    // And the rule that guard qualifies still holds: nothing behind it and nothing cut off
    // it, so it is the key, which is what lets Esc interrupt a turn with no timer involved.
    #[test]
    fn an_escape_that_is_all_there_is_still_interrupts() {
        let mut buffer = Vec::new();

        let events = feed_chunk(&mut buffer, b"\x1b", false);

        assert_eq!(events.len(), 1, "{:?}", events);
        assert!(matches!(events[0], Event::Key(key) if key.code == KeyCode::Esc));
        assert!(buffer.is_empty());
    }

    // Being cut short bears on the Escape rule and on nothing else: a chunk that did end on
    // a sequence boundary still delivers everything in it.
    #[test]
    fn a_cut_read_still_delivers_the_sequences_it_did_contain() {
        let mut buffer = Vec::new();

        let events = feed_chunk(&mut buffer, b"\x1b[A\x1b[B", true);

        assert_eq!(events.len(), 2, "{:?}", events);
        assert!(matches!(events[0], Event::Key(key) if key.code == KeyCode::Up));
        assert!(matches!(events[1], Event::Key(key) if key.code == KeyCode::Down));
        assert!(buffer.is_empty());
    }

    fn drain(bytes: &[u8]) -> (Vec<Event>, Vec<u8>) {
        let mut buffer = bytes.to_vec();
        let mut events = Vec::new();
        while !buffer.is_empty() {
            match take_event(&mut buffer) {
                Some(Some(event)) => events.push(event),
                Some(None) => {}
                None => break,
            }
        }
        (events, buffer)
    }

    // The whole reason the bytes are parsed rather than crossterm's per-record events
    // being filtered: an arrow key is one event, not "[" then "A" then Up.
    #[test]
    fn an_arrow_key_is_one_event_and_not_its_characters() {
        let (events, rest) = drain(b"\x1b[A");

        assert_eq!(events.len(), 1, "{:?}", events);
        assert!(matches!(events[0], Event::Key(key) if key.code == KeyCode::Up));
        assert!(rest.is_empty());
    }

    // `parse_from` is happy to ignore trailing bytes, so asking it about the whole
    // buffer would take the next keystroke along with this one. Only the shortest
    // prefix that parses may be consumed.
    #[test]
    fn two_sequences_running_together_stay_two_events() {
        let (events, rest) = drain(b"\x1b[A\x1b[B");

        assert_eq!(events.len(), 2, "{:?}", events);
        assert!(matches!(events[0], Event::Key(key) if key.code == KeyCode::Up));
        assert!(matches!(events[1], Event::Key(key) if key.code == KeyCode::Down));
        assert!(rest.is_empty());
    }

    // The point of the exercise.
    #[test]
    fn a_wheel_report_becomes_a_scroll_event() {
        let (events, _) = drain(b"\x1b[<64;10;5M");

        assert_eq!(events.len(), 1, "{:?}", events);
        let Event::Mouse(mouse) = events[0] else {
            panic!("not a mouse event: {:?}", events[0]);
        };
        assert!(
            matches!(mouse.kind, MouseEventKind::ScrollUp | MouseEventKind::ScrollDown),
            "{:?}",
            mouse.kind
        );
    }

    // A filter working on crossterm's per-character events could not tell this from the
    // start of a sequence, and eating it would corrupt pasted code.
    #[test]
    fn ordinary_text_with_a_bracket_in_it_survives() {
        let (events, rest) = drain(b"arr[i]");

        let typed: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Key(key) => match key.code {
                    KeyCode::Char(ch) => Some(ch),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(typed, "arr[i]");
        assert!(rest.is_empty());
    }

    // Half a sequence is not an event yet, and must not be mistaken for one.
    #[test]
    fn an_unfinished_sequence_waits_for_the_rest_of_itself() {
        let mut buffer = b"\x1b[".to_vec();

        assert!(take_event(&mut buffer).is_none());
        assert_eq!(buffer, b"\x1b[", "an incomplete sequence was consumed");

        buffer.push(b'C');
        let event = take_event(&mut buffer).expect("now complete").expect("an event");
        assert!(matches!(event, Event::Key(key) if key.code == KeyCode::Right));
    }
}
