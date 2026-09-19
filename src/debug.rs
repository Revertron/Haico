//! An optional log file, for failures that only show up mid-session.
//!
//! Off unless `HAICO_DEBUG` is set, and a file rather than stderr because the REPL holds
//! the terminal in raw mode for the whole session: a diagnostic printed while the model is
//! streaming would land in the middle of the drawn screen and shift everything below it.
//! A file can be tailed from a second window, and is still there after the run that
//! produced it — which is the case that matters for something that happens "sometimes".

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

// Resolved once. The variable is read at startup and cannot change afterwards, and a run
// with logging off should not pay for an env lookup at every point that offers to log.
fn log_path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        let flag = std::env::var("HAICO_DEBUG").ok()?;
        // "0" is off, so the variable can stay exported in a shell profile and be turned
        // on for one run without editing it.
        if flag.is_empty() || flag == "0" {
            return None;
        }
        crate::store::path("debug.log")
    })
    .as_ref()
}

// Checked by callers before building an entry, because the bodies are not cheap: the
// reviewer's evidence is a whole diff, and a reply can be kilobytes of reasoning.
pub(crate) fn enabled() -> bool {
    log_path().is_some()
}

// Append one entry, indenting the body so a multi-line reply stays visibly one entry.
//
// Every failure here is swallowed. A debug log that reported its own problems into a
// raw-mode terminal would damage the session it was added to explain, and a missing entry
// is a smaller loss than that.
pub(crate) fn log(section: &str, body: &str) {
    let Some(path) = log_path() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }

    let mut entry = format!(
        "[{}] {}\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
        section
    );
    for line in body.lines() {
        entry.push_str("    ");
        entry.push_str(line);
        entry.push('\n');
    }

    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(entry.as_bytes());
    }
}
