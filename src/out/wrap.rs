//! Widths and word wrap: how many cells a string takes, and where a line breaks so that
//! a word still arriving is free to move to the next row.

use super::*;

// How wide a string prints. Escape sequences take no space, and counting them would
// make every colored line look several times longer than it is.
// The pictographs a terminal draws two cells wide even though the Unicode tables call
// them one, sorted so they can be searched. These are the characters with the Emoji
// property whose *default* presentation is text rather than color: the standard gives
// them an East Asian Width of Ambiguous or Neutral, meaning one cell, and every terminal
// with a color emoji font draws them from it at two, because that is the glyph the font
// has. `unicode-width` answers with the standard, correctly, and is one cell short.
//
// This is the gap that broke a table: the frame was measured with ☑ at one cell and each
// row drawn a cell wider than the frame around it.
//
// It is deliberately the Emoji property *minus* the characters that are also ordinary
// text. ©, ®, ™, ‼, ⁉, ℹ, Ⓜ, ▪ and ▫ carry the property too, and widening those would
// break far more prose than it fixed. A pictograph found missing later is one line here.
// Nothing needs removing for the emoji that are already two cells by the standard —
// ✅, ❌, ⚡ and the astral ones — and an entry that overlaps them costs nothing.
pub(super) const WIDE_TEXT_EMOJI: &[(u32, u32)] = &[
    (0x2194, 0x2199), // ↔ ↕ ↖ ↗ ↘ ↙
    (0x21A9, 0x21AA), // ↩ ↪
    (0x2328, 0x2328), // ⌨
    (0x23CF, 0x23CF), // ⏏
    (0x23ED, 0x23EF), // ⏭ ⏮ ⏯
    (0x23F1, 0x23F2), // ⏱ ⏲
    (0x23F8, 0x23FA), // ⏸ ⏹ ⏺
    (0x25B6, 0x25B6), // ▶
    (0x25C0, 0x25C0), // ◀
    (0x2600, 0x2604), // ☀ ☁ ☂ ☃ ☄
    (0x260E, 0x260E), // ☎
    (0x2611, 0x2611), // ☑
    (0x2618, 0x2618), // ☘
    (0x261D, 0x261D), // ☝
    (0x2620, 0x2620), // ☠
    (0x2622, 0x2623), // ☢ ☣
    (0x2626, 0x2626), // ☦
    (0x262A, 0x262A), // ☪
    (0x262E, 0x262F), // ☮ ☯
    (0x2638, 0x263A), // ☸ ☹ ☺
    (0x2640, 0x2640), // ♀
    (0x2642, 0x2642), // ♂
    (0x265F, 0x2660), // ♟ ♠
    (0x2663, 0x2663), // ♣
    (0x2665, 0x2666), // ♥ ♦
    (0x2668, 0x2668), // ♨
    (0x267B, 0x267B), // ♻
    (0x267E, 0x267E), // ♾
    (0x2692, 0x2697), // ⚒ ⚓ ⚔ ⚕ ⚖ ⚗
    (0x2699, 0x2699), // ⚙
    (0x269B, 0x269C), // ⚛ ⚜
    (0x26A0, 0x26A0), // ⚠
    (0x26A7, 0x26A7), // ⚧
    (0x26B0, 0x26B1), // ⚰ ⚱
    (0x26C8, 0x26C8), // ⛈
    (0x26CF, 0x26CF), // ⛏
    (0x26D1, 0x26D1), // ⛑
    (0x26D3, 0x26D3), // ⛓
    (0x26E9, 0x26E9), // ⛩
    (0x26F0, 0x26F1), // ⛰ ⛱
    (0x26F4, 0x26F4), // ⛴
    (0x26F7, 0x26F9), // ⛷ ⛸ ⛹
    (0x2702, 0x2702), // ✂
    (0x2708, 0x2709), // ✈ ✉
    (0x270C, 0x270D), // ✌ ✍
    (0x270F, 0x270F), // ✏
    (0x2712, 0x2712), // ✒
    (0x2714, 0x2714), // ✔
    (0x2716, 0x2716), // ✖
    (0x271D, 0x271D), // ✝
    (0x2721, 0x2721), // ✡
    (0x2733, 0x2734), // ✳ ✴
    (0x2744, 0x2744), // ❄
    (0x2747, 0x2747), // ❇
    (0x2763, 0x2764), // ❣ ❤
    (0x27A1, 0x27A1), // ➡
    (0x2934, 0x2935), // ⤴ ⤵
    (0x2B05, 0x2B07), // ⬅ ⬆ ⬇
];

// How many cells one character takes. Two for CJK and the emoji, none for a combining
// mark or a variation selector, one for nearly everything else.
//
// A control character is none: it prints nothing. The exception is a tab, which is kept
// at one because that is what the wrapper counts it as when it treats it as a blank —
// neither is the eight columns a terminal actually gives it, and disagreeing with
// ourselves would be worse than being uniformly approximate.
pub(super) fn cell_width(ch: char) -> usize {
    let code = ch as u32;
    // Almost everything printed is ASCII, and the table starts far above it.
    if code >= WIDE_TEXT_EMOJI[0].0 && code <= WIDE_TEXT_EMOJI[WIDE_TEXT_EMOJI.len() - 1].1 {
        let found = WIDE_TEXT_EMOJI.binary_search_by(|&(first, last)| {
            if code < first {
                std::cmp::Ordering::Greater
            } else if code > last {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        });
        if found.is_ok() {
            return 2;
        }
    }
    UnicodeWidthChar::width(ch).unwrap_or(usize::from(ch == '\t'))
}

pub(crate) fn visible_width(text: &str) -> usize {
    let mut width = 0;
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            width += cell_width(ch);
            continue;
        }
        // CSI sequences run until a byte in @..~; anything else is a two-character
        // escape. Either way none of it lands on the screen.
        if let Some('[') = chars.next() {
            for ch in chars.by_ref() {
                if ('@'..='~').contains(&ch) {
                    break;
                }
            }
        }
    }
    width
}

// The width text is broken at. A row exactly as wide as the terminal wraps in some
// terminals and not others, so everything stops a column short — the same margin the
// input box keeps, for the same reason.
pub(crate) fn wrap_width() -> usize {
    terminal_width().saturating_sub(1).max(1)
}

// Which colors are still switched on at the end of a row, given the ones that were on at
// the start of it. Only SGR sequences — the ones ending in "m" — say anything about color;
// a cursor move carries nothing over. A reset closes everything, so it clears what has
// been collected; anything else joins it in the order it was written, which is the order
// it has to be written again in to put the terminal back the way the row left it.
pub(super) fn open_colors(open: &str, row: &str) -> String {
    let mut colors = open.to_string();
    let mut chars = row.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            continue;
        }
        let escape = take_escape(&mut chars);
        let Some(params) = escape
            .strip_prefix("\u{1b}[")
            .and_then(|rest| rest.strip_suffix('m'))
        else {
            continue;
        };
        // "\x1b[m" and "\x1b[0m" are the same reset; a list of them still resets.
        if params.split(';').all(|param| param.is_empty() || param == "0") {
            colors.clear();
        } else {
            colors.push_str(&escape);
        }
    }
    colors
}

// Consume one escape sequence, the ESC itself already taken, and give it back whole. A
// sequence cut in half paints everything after it, so nothing here may ever split one.
pub(super) fn take_escape(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut escape = String::from('\u{1b}');
    match chars.next() {
        Some('[') => {
            escape.push('[');
            for ch in chars.by_ref() {
                escape.push(ch);
                if ('@'..='~').contains(&ch) {
                    break;
                }
            }
        }
        Some(ch) => escape.push(ch),
        None => {}
    }
    escape
}

// Where the wrapped rows of a line start, so they line up under the text they continue
// rather than under whatever introduced it: past the line's own indentation, and past a
// bullet or a numbered item if it has one. Without this a wrapped list item runs back to
// column zero and reads as the start of a new paragraph.
pub(super) fn hanging_indent(line: &str, width: usize) -> usize {
    let mut chars = line.chars().peekable();
    let mut indent = 0usize;
    loop {
        match chars.peek() {
            Some('\u{1b}') => {
                chars.next();
                take_escape(&mut chars);
            }
            Some(' ') => {
                chars.next();
                indent += 1;
            }
            _ => break,
        }
    }

    // A marker can have colors around it — the reply's own "● " arrives that way — so it
    // is matched against what actually prints rather than against the raw characters.
    let mut visible = String::new();
    let mut rest = chars.clone();
    while visible.chars().count() < 6 {
        match rest.next() {
            Some('\u{1b}') => {
                take_escape(&mut rest);
            }
            Some(ch) => visible.push(ch),
            None => break,
        }
    }

    let marker = ["- ", "* ", "+ ", "• ", "● ", "○ ", "→ "]
        .into_iter()
        .find(|marker| visible.starts_with(marker))
        .map(|marker| marker.chars().count())
        .or_else(|| numbered_marker(&visible))
        .unwrap_or(0);

    // Half the row is as far as a hanging indent can usefully go; a deeply indented code
    // line past that would leave a column too narrow to read.
    (indent + marker).min(width / 2)
}

// "1. " or "12) ": the width of the number together with the punctuation after it.
pub(super) fn numbered_marker(visible: &str) -> Option<usize> {
    let digits = visible.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let mut after = visible.chars().skip(digits);
    match (after.next(), after.next()) {
        (Some('.') | Some(')'), Some(' ')) => Some(digits + 2),
        _ => None,
    }
}

// Breaking output into rows at spaces rather than wherever the right edge happens to fall.
//
// Left to itself the terminal cuts words in half, which is what a streamed reply of any
// length used to look like. Doing it here costs two things the terminal never has to
// think about. Escape sequences carry the colors and move the cursor nowhere: they take
// no width and must survive whole. And a run with no space in it that is wider than a
// whole row — a URL, a hash, a base64 blob — has to be broken anyway, because refusing
// would push it off the edge, which is worse than the break that was going to happen
// regardless.
//
// The state is carried across calls because a line does not arrive in one piece: the
// reply's "● " and the words after it are separate writes, and only the first of them
// says where the wrapped rows should line up.
#[derive(Clone)]
pub(super) struct Wrap {
    pub(super) width: usize,
    // Columns already used on the current screen row.
    pub(super) column: usize,
    // What a wrapped row of the current line is indented by, worked out once from the
    // start of the line and kept for the rest of it.
    pub(super) indent: String,
    // Nothing has been written on this line yet, so its indent is still to be found.
    pub(super) fresh: bool,
    // Nothing but the indent on the current row, so breaking again would only leave an
    // empty row behind.
    pub(super) row_empty: bool,
}

impl Wrap {
    pub(super) fn new(width: usize) -> Self {
        Wrap {
            width: width.max(1),
            column: 0,
            indent: String::new(),
            fresh: true,
            row_empty: true,
        }
    }

    // A line ended for real, by a newline of its own or by the input box taking the row.
    pub(super) fn reset_line(&mut self) {
        self.column = 0;
        self.indent.clear();
        self.fresh = true;
        self.row_empty = true;
    }

    // The same text with the breaks put in. Newlines that were already there keep their
    // meaning: they end a line, and the next one works out its own indent.
    pub(super) fn feed(&mut self, text: &str) -> String {
        let mut wrapped = String::with_capacity(text.len() + 16);
        for (index, line) in text.split('\n').enumerate() {
            if index > 0 {
                wrapped.push('\n');
                self.reset_line();
            }
            self.line(line, &mut wrapped);
        }
        wrapped
    }

    pub(super) fn line(&mut self, line: &str, out: &mut String) {
        if line.is_empty() {
            return;
        }
        if self.fresh {
            self.indent = " ".repeat(hanging_indent(line, self.width));
            self.fresh = false;
        }

        // Almost everything printed is short. Passing it through untouched keeps the
        // common case byte-for-byte what it was before any of this existed.
        let visible = visible_width(line);
        if self.column + visible <= self.width {
            out.push_str(line);
            self.column += visible;
            self.row_empty &= visible == 0;
            return;
        }

        // A word is a run of non-blanks together with any escape sequence touching it:
        // the color has to travel to whichever row the word lands on.
        let mut word = String::new();
        let mut word_width = 0usize;
        let mut gap = String::new();
        let mut gap_width = 0usize;
        let mut chars = line.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '\u{1b}' => word.push_str(&take_escape(&mut chars)),
                // A stray carriage return prints nothing and belongs where it was written.
                '\r' => word.push('\r'),
                ' ' | '\t' => {
                    if !word.is_empty() {
                        self.place(out, &word, word_width, &gap, gap_width);
                        word.clear();
                        word_width = 0;
                        gap.clear();
                        gap_width = 0;
                    }
                    gap.push(ch);
                    gap_width += 1;
                }
                _ => {
                    word.push(ch);
                    word_width += cell_width(ch);
                }
            }
        }
        self.place(out, &word, word_width, &gap, gap_width);
    }

    // One word, with the blanks that stood in front of it. They are held back until here
    // so that a break can drop them instead of leaving them dangling past the edge.
    pub(super) fn place(
        &mut self,
        out: &mut String,
        word: &str,
        word_width: usize,
        gap: &str,
        gap_width: usize,
    ) {
        if word.is_empty() {
            return;
        }
        // Escape sequences on their own — the reset closing a colored run — take no width
        // and belong exactly where they were written.
        if word_width == 0 {
            out.push_str(gap);
            self.column += gap_width;
            out.push_str(word);
            return;
        }
        if self.column + gap_width + word_width <= self.width {
            out.push_str(gap);
            self.column += gap_width;
        } else {
            if !self.row_empty {
                self.break_row(out);
            }
            if self.column + word_width > self.width {
                return self.hard(out, word);
            }
        }
        out.push_str(word);
        self.column += word_width;
        self.row_empty = false;
    }

    pub(super) fn break_row(&mut self, out: &mut String) {
        out.push('\n');
        out.push_str(&self.indent);
        self.column = self.indent.chars().count();
        self.row_empty = true;
    }

    // A word too long for a row of its own, broken wherever the edge falls. There is no
    // better place for it, and letting it run would only move the same break one column
    // to the right.
    pub(super) fn hard(&mut self, out: &mut String, word: &str) {
        let mut chars = word.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\u{1b}' {
                let escape = take_escape(&mut chars);
                out.push_str(&escape);
                continue;
            }
            if ch == '\r' {
                out.push('\r');
                continue;
            }
            // A two-cell character cannot straddle the edge, so the room for it is
            // asked for before it is written rather than after.
            let cells = cell_width(ch);
            if self.column + cells > self.width && !self.row_empty {
                self.break_row(out);
            }
            out.push(ch);
            self.column += cells;
            self.row_empty = false;
        }
    }
}
