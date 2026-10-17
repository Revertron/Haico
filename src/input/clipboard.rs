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
