#[cfg(windows)]
pub(super) fn read() -> Option<String> {
    // Empty is not worth a paste, and a clipboard holding an image rather than text
    // reports an error rather than something to insert.
    clipboard_win::get_clipboard_string()
        .ok()
        .filter(|text| !text.is_empty())
}

#[cfg(not(windows))]
pub(super) fn read() -> Option<String> {
    None
}

// What a mouse selection copies. Windows text on the clipboard ends its lines in CR LF,
// and a bare LF pasted into an older program runs every line together.
#[cfg(windows)]
pub(super) fn write(text: &str) {
    let _ = clipboard_win::set_clipboard_string(&text.replace('\n', "\r\n"));
}

// Anywhere else there is no clipboard API this can count on — X11, Wayland, a session over
// SSH — but the terminal usually has one, and OSC 52 asks it.
#[cfg(not(windows))]
pub(super) fn write(text: &str) {
    crate::out::copy_through_terminal(text);
}
