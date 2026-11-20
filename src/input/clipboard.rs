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

// A picture on the clipboard, as PNG bytes.
//
// The registered "PNG" format comes first, and is what browsers, Snipping Tool and most
// editors put there: it is the picture exactly, transparency included. Only without it does
// this fall back to the bitmap every image copy offers, which Windows hands over as a BMP
// and which goes out as PNG all the same — a BMP is a format no model server takes.
#[cfg(windows)]
pub(super) fn read_image() -> Option<Vec<u8>> {
    use clipboard_win::{formats, is_format_avail, raw, register_format, Clipboard};

    let (png, bmp) = {
        // Held only while copying out: an open clipboard is closed to every other program,
        // and the conversion below can take a while on a large screenshot.
        let _open = Clipboard::new_attempts(10).ok()?;
        let mut png = Vec::new();
        let format = register_format("PNG").filter(|format| is_format_avail(format.get()));
        if let Some(format) = format {
            let _ = raw::get_vec(format.get(), &mut png);
        }
        let mut bmp = Vec::new();
        if png.is_empty() && is_format_avail(formats::CF_BITMAP) {
            let _ = raw::get_bitmap(&mut bmp);
        }
        (png, bmp)
    };

    if crate::mention::image_type(&png) == Some("image/png") {
        return Some(trim_png(png));
    }
    if bmp.is_empty() {
        return None;
    }
    let decoded = image::load_from_memory_with_format(&bmp, image::ImageFormat::Bmp).ok()?;
    let mut out = std::io::Cursor::new(Vec::new());
    decoded.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

// The clipboard reports the size of the memory block, not of the picture, and the block is
// rounded up — so a PNG read from it can trail bytes past its end. Decoders skip them; cut
// anyway, since every one of them would be base64'd into each request that carries it.
#[cfg(windows)]
fn trim_png(mut png: Vec<u8>) -> Vec<u8> {
    // The last chunk is IEND: a four-byte type with no data, then a four-byte checksum.
    if let Some(at) = png.windows(4).rposition(|window| window == b"IEND") {
        png.truncate((at + 8).min(png.len()));
    }
    png
}

// Anywhere else the terminal does text pasting itself, but it cannot paste a picture, so
// this asks whichever clipboard tool the desktop has. Wayland's first, then X11's; both
// fail at once when their display is not there, and neither is a dependency.
#[cfg(not(windows))]
pub(super) fn read_image() -> Option<Vec<u8>> {
    use std::process::{Command, Stdio};
    const TOOLS: &[(&str, &[&str])] = &[
        ("wl-paste", &["--no-newline", "--type", "image/png"]),
        ("xclip", &["-selection", "clipboard", "-target", "image/png", "-out"]),
    ];
    TOOLS.iter().find_map(|(tool, args)| {
        let output = Command::new(tool)
            .args(*args)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        let png = output.stdout;
        (output.status.success() && crate::mention::image_type(&png) == Some("image/png"))
            .then_some(png)
    })
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
