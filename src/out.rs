//! The one place that writes to the terminal.
//!
//! Everything the REPL shows goes through here rather than reaching `println!`
//! directly. Two things need that. Output arriving while the model works has to be
//! able to step around a live input line, which only works if one writer knows what
//! is currently on screen; and raw mode drops the terminal's own newline
//! translation, so "\n" has to become "\r\n" in one place rather than at two
//! hundred call sites.
//!
//! Nothing turns raw mode on yet. `set_raw` is here for the input rework and stays
//! off until then, so what reaches the terminal today is byte for byte what the
//! `println!` calls this replaced produced.

use crossterm::terminal::{Clear, ClearType};
use std::fmt;
use std::io::{self, Write};
use std::sync::{Mutex, MutexGuard, OnceLock};

struct Out {
    raw: bool,
    // Whether a transient status line is sitting on the current row waiting to be
    // erased. Anything else printed has to erase it first, or the two overlap.
    transient: bool,
    // Set by the tests so they can assert on what would have been written.
    capture: Option<String>,
}

impl Out {
    // Show something, erasing a transient status line first if one is up.
    fn emit(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.drop_transient();
        self.put(text);
    }

    fn put(&mut self, text: &str) {
        // Raw mode turns off the terminal's newline translation, so a bare line feed
        // moves down a row without returning to column 0 and the output walks
        // diagonally off the right of the screen. A line feed that already has its
        // carriage return is left alone.
        if self.raw && text.contains('\n') {
            let mut translated = String::with_capacity(text.len() + 8);
            let mut previous = '\0';
            for ch in text.chars() {
                if ch == '\n' && previous != '\r' {
                    translated.push('\r');
                }
                translated.push(ch);
                previous = ch;
            }
            self.write_out(&translated);
        } else {
            self.write_out(text);
        }
    }

    fn write_out(&mut self, text: &str) {
        if let Some(buffer) = &mut self.capture {
            buffer.push_str(text);
        } else {
            let _ = io::stdout().write_all(text.as_bytes());
        }
    }

    fn drop_transient(&mut self) {
        if !self.transient {
            return;
        }
        self.transient = false;
        self.put(&format!("\r{}", Clear(ClearType::CurrentLine)));
    }
}

// A panic while the lock is held would otherwise poison it and take the whole
// interface down with it, which is a bad trade for a line of output.
fn lock() -> MutexGuard<'static, Out> {
    static OUT: OnceLock<Mutex<Out>> = OnceLock::new();
    OUT.get_or_init(|| {
        Mutex::new(Out {
            raw: false,
            transient: false,
            capture: None,
        })
    })
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn write_fmt(args: fmt::Arguments) {
    lock().emit(&args.to_string());
}

pub(crate) fn write_line(args: fmt::Arguments) {
    let mut text = args.to_string();
    text.push('\n');
    lock().emit(&text);
}

pub(crate) fn flush() {
    if lock().capture.is_none() {
        let _ = io::stdout().flush();
    }
}

// Whether line feeds need a carriage return in front of them. Off until the input
// rework takes raw mode for the whole session, which is the only caller it is
// waiting for.
#[allow(dead_code)]
pub(crate) fn set_raw(raw: bool) {
    lock().raw = raw;
}

// A line that is going to be erased again: a spinner, a "working on it" note. It
// stays on its row until something else is printed or `clear_transient` runs.
pub(crate) fn transient(text: &str) {
    let mut state = lock();
    state.drop_transient();
    state.put(text);
    state.transient = true;
    if state.capture.is_none() {
        let _ = io::stdout().flush();
    }
}

pub(crate) fn clear_transient() {
    lock().drop_transient();
}

#[cfg(test)]
pub(crate) fn start_capture() {
    let mut state = lock();
    state.raw = false;
    state.transient = false;
    state.capture = Some(String::new());
}

#[cfg(test)]
pub(crate) fn take_capture() -> String {
    let mut state = lock();
    state.raw = false;
    state.capture.take().unwrap_or_default()
}

macro_rules! out {
    ($($arg:tt)*) => ($crate::out::write_fmt(format_args!($($arg)*)));
}

macro_rules! outln {
    () => ($crate::out::write_line(format_args!("")));
    ($($arg:tt)*) => ($crate::out::write_line(format_args!($($arg)*)));
}

#[cfg(test)]
mod tests {
    // One test at a time: the writer is a global, so two tests capturing at once
    // would read each other's output.
    use std::sync::Mutex;
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn raw_mode_returns_the_carriage_on_every_line_feed() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

        super::start_capture();
        outln!("one");
        out!("two\nthree");
        assert_eq!(super::take_capture(), "one\ntwo\nthree");

        super::start_capture();
        super::set_raw(true);
        outln!("one");
        out!("two\nthree");
        // And text that already has its carriage returns is passed through as it is.
        out!("\r\nfour");
        assert_eq!(super::take_capture(), "one\r\ntwo\r\nthree\r\nfour");
    }

    #[test]
    fn a_transient_line_is_erased_by_whatever_comes_next() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

        super::start_capture();
        super::transient("working…");
        outln!("done");
        let written = super::take_capture();
        assert!(written.starts_with("working…\r"), "got {:?}", written);
        assert!(written.ends_with("done\n"), "got {:?}", written);

        // And erasing it twice does not emit the escape twice.
        super::start_capture();
        super::transient("working…");
        super::clear_transient();
        super::clear_transient();
        let written = super::take_capture();
        assert_eq!(written.matches('\r').count(), 1, "got {:?}", written);
    }
}
