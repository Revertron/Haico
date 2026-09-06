//! Markdown into a Lumio view, one block at a time, as it streams.
//!
//! The terminal's markdown lives in `render.rs` and is not reusable here: it
//! settles lines against a width and emits SGR escapes, so what comes out is a
//! `String` of terminal instructions. This produces spans and blocks instead.
//!
//! ## Why streaming is the whole difficulty
//!
//! A reply arrives a few characters at a time, and markdown is not decidable
//! from a prefix. `- a` is a list item until a second line makes it something
//! else; ``` opens a fence that is not a fence until it closes; a line of `---`
//! under a paragraph turns that paragraph into a heading retroactively. So the
//! renderer cannot commit to the shape of the text it has just been handed.
//!
//! It commits to everything *before* the last block boundary and lays the rest
//! down provisionally, replacing it on the next chunk with
//! [`RichText::truncate`]. Two offsets are all the state that needs: how much
//! markdown has been settled, and how long the view's text was at that point.
//! Everything before the cut keeps its layout, so a chunk costs one block rather
//! than one reply — which is the whole reason Lumio's blocks became the
//! relayout unit.
//!
//! One `RichText` per message, not one per block. That is deliberate and it is
//! the reason `truncate` had to exist: a message rendered as a stack of views
//! cannot be selected in one drag, and selecting a whole answer is something a
//! person does several times an hour.

use lumio::prelude::*;
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

/// Colours and sizes. Small enough to keep here rather than in a theme, and the
/// GUI's own: the turn has no opinion about any of it.
const CODE_BAND: u32 = 0xFF232323;
const CODE_INK: u32 = 0xFFC8D4C8;
const HEADING_INK: u32 = 0xFFE8E8E8;
const QUOTE_INK: u32 = 0xFF9AA6B2;
const LINK_INK: u32 = 0xFF6FA8DC;
const RULE_INK: u32 = 0xFF5A5A5A;
/// Indent per nesting level, and the hanging indent that keeps a wrapped list
/// item under its own text rather than under its bullet.
const INDENT: f32 = 16.0;
const HANGING: f32 = 14.0;

/// Streams markdown into one `RichText`.
///
/// Holds no reference to the view: the caller owns it, and hands it in on each
/// chunk. That keeps this usable from anywhere the view can be reached, which in
/// the GUI means inside a closure on the UI thread.
pub(crate) struct MarkdownView {
    /// Everything received so far, because a later chunk can change what an
    /// earlier line means.
    source: String,
    /// Bytes of `source` rendered as blocks that can no longer change.
    settled_source: usize,
    /// The view's text length at that point — where `truncate` cuts back to.
    settled_text: usize,
    /// Base text size in dip; headings scale off it.
    size: f32,
}

impl MarkdownView {
    pub(crate) fn new(size: f32) -> Self {
        MarkdownView { source: String::new(), settled_source: 0, settled_text: 0, size }
    }

    /// Add a chunk and redraw whatever it could have changed.
    pub(crate) fn push(&mut self, view: &RichText, chunk: &str) {
        self.source.push_str(chunk);
        self.rerender(view);
    }

    /// Nothing more is coming, so the last block is final too.
    pub(crate) fn finish(&mut self, view: &RichText) {
        self.rerender(view);
        self.settled_source = self.source.len();
        self.settled_text = view.text_len();
    }

    /// Render everything after the settled point, advancing it over whatever can
    /// no longer change.
    fn rerender(&mut self, view: &RichText) {
        // Throw away the provisional tail. Blocks before this keep their layout.
        view.truncate(self.settled_text);

        let pending = &self.source[self.settled_source..];
        let boundary = last_block_boundary(pending);
        let (settled, provisional) = pending.split_at(boundary);

        if !settled.is_empty() {
            render(view, settled, self.size);
            // Recorded before the provisional tail goes down, so the next chunk
            // cuts back to exactly here.
            self.settled_source += settled.len();
            self.settled_text = view.text_len();
        }
        if !provisional.is_empty() {
            render(view, provisional, self.size);
        }
    }
}

/// The byte offset in `md` after the last point where a block certainly ended.
///
/// A blank line ends a block in markdown, so the text before the final blank
/// line cannot be reinterpreted by anything that comes later — with one
/// exception that matters: inside a fenced code block a blank line means
/// nothing, and the fence itself is not closed until its matching ``` arrives.
/// So fences are tracked and a boundary inside an open one is not a boundary.
fn last_block_boundary(md: &str) -> usize {
    let mut in_fence = false;
    let mut boundary = 0;
    let mut offset = 0;
    let mut blank_run = 0usize;

    for line in md.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            // A closing fence ends its block, and the line after it is a fresh
            // start — but only once the newline has actually arrived, or the
            // fence marker itself might still be growing.
            if !in_fence && line.ends_with('\n') {
                boundary = offset + line.len();
            }
            blank_run = 0;
        } else if !in_fence && line.trim().is_empty() {
            blank_run += 1;
            if line.ends_with('\n') {
                boundary = offset + line.len();
            }
        } else {
            blank_run = 0;
        }
        offset += line.len();
    }
    let _ = blank_run;
    boundary
}

/// Append `md` to `view` as blocks and spans.
fn render(view: &RichText, md: &str, size: f32) {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);

    let mut state = Render {
        view,
        size,
        style: SpanStyle::default(),
        stack: Vec::new(),
        list: Vec::new(),
        // Asked of the view rather than assumed: a streamed chunk renders into a
        // message that already has blocks in it, and its first block is not the
        // message's first. Getting this wrong swallowed every separator after the
        // first chunk.
        at_start: view.text_len() == 0,
        pending_block: None,
        pending_item: false,
        // False even mid-message, and that is correct rather than a shortcut: a
        // boundary only ever falls on a blank line, a tight list has none
        // between its items, so a settled prefix can never end part-way through
        // one. A loose list does have blank lines between items — and there the
        // separator is what the author asked for.
        last_item: false,
    };
    for event in Parser::new_ext(md, options) {
        state.event(event);
    }
    state.flush_block();
}

/// A list being rendered: `Some(n)` for an ordered list at item `n`.
type ListLevel = Option<u64>;

struct Render<'a> {
    view: &'a RichText,
    size: f32,
    /// The span style built up by the inline tags currently open.
    style: SpanStyle,
    /// Styles to return to as those tags close.
    stack: Vec<SpanStyle>,
    list: Vec<ListLevel>,
    /// Whether anything has been written into the current block yet, so a
    /// separator is only added between blocks and not before the first.
    at_start: bool,
    /// The block to start when the first text of it arrives. Deferred, because
    /// starting a block that then gets no text would leave an empty one in the
    /// middle of the message.
    pending_block: Option<BlockStyle>,
    /// Whether the pending block, and the last one opened, are list items.
    ///
    /// Blocks are separated by a blank line, because Lumio stacks them without
    /// spacing and two paragraphs with nothing between them read as one. List
    /// items are the exception: a blank line between every bullet turns a tight
    /// list into a loose one, and markdown means those differently.
    pending_item: bool,
    last_item: bool,
}

impl Render<'_> {
    fn event(&mut self, event: Event) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => self.text(&text),
            // Inline code: a span, not a block, and monospace rather than merely
            // tinted — which is what the family field in `SpanStyle` was for.
            Event::Code(text) => {
                let mut style = self.style.clone().code();
                style.background = Some(CODE_BAND);
                self.write(&text, style);
            }
            Event::SoftBreak => self.write(" ", self.style.clone()),
            Event::HardBreak => self.write("\n", self.style.clone()),
            Event::Rule => {
                self.block(BlockStyle::default());
                self.write("────────", SpanStyle::default().color(RULE_INK));
            }
            // Raw HTML is shown as what it is rather than interpreted: a model
            // emitting a tag meant it literally often enough, and interpreting
            // it would mean a second parser and a sanitizer.
            Event::Html(text) | Event::InlineHtml(text) => {
                self.write(&text, self.style.clone().code())
            }
            Event::FootnoteReference(name) => {
                self.write(&format!("[{name}]"), self.style.clone())
            }
            Event::TaskListMarker(done) => {
                self.write(if done { "[x] " } else { "[ ] " }, self.style.clone())
            }
            Event::InlineMath(text) | Event::DisplayMath(text) => {
                self.write(&text, self.style.clone().code())
            }
        }
    }

    fn start(&mut self, tag: Tag) {
        self.stack.push(self.style.clone());
        match tag {
            Tag::Paragraph => self.block(self.indented(BlockStyle::default())),
            Tag::Heading { level, .. } => {
                self.block(BlockStyle::default());
                // Scaled rather than sized absolutely, so the whole message
                // scales with one number.
                let scale = match level {
                    HeadingLevel::H1 => 1.5,
                    HeadingLevel::H2 => 1.3,
                    HeadingLevel::H3 => 1.15,
                    _ => 1.05,
                };
                self.style = self.style.clone().bold().size(self.size * scale).color(HEADING_INK);
            }
            Tag::CodeBlock(kind) => {
                // The reason `BlockStyle::pre` exists: whitespace is content
                // here, so indentation survives, and the band makes the whole
                // thing read as one box rather than a stack of tinted strips.
                self.block(
                    self.indented(BlockStyle::pre())
                        .background(CODE_BAND)
                        .padding(8.0)
                        .hanging(HANGING),
                );
                self.style = self.style.clone().code().color(CODE_INK);
                // The language is not shown: it is a hint for highlighting,
                // which is `syntect`'s job and not done yet.
                if let CodeBlockKind::Fenced(_) = kind {}
            }
            Tag::BlockQuote(_) => {
                self.block(self.indented(BlockStyle::default()).indent(INDENT));
                self.style = self.style.clone().italic().color(QUOTE_INK);
            }
            Tag::List(start) => self.list.push(start),
            Tag::Item => {
                // Each item is its own block, which is what gives it a hanging
                // indent: the bullet sits at the left edge and a wrapped line
                // lines up under the text instead of under the bullet.
                let depth = self.list.len().max(1) as f32;
                self.block_as(
                    BlockStyle::default().indent(INDENT * depth).hanging(HANGING),
                    true,
                );
                let marker = match self.list.last_mut() {
                    Some(Some(n)) => {
                        let marker = format!("{n}. ");
                        *n += 1;
                        marker
                    }
                    _ => "• ".to_string(),
                };
                self.write(&marker, self.style.clone().color(QUOTE_INK));
            }
            Tag::Emphasis => self.style = self.style.clone().italic(),
            Tag::Strong => self.style = self.style.clone().bold(),
            Tag::Strikethrough => self.style = self.style.clone().strike(),
            Tag::Link { dest_url, .. } => {
                self.style = self.style.clone().link(&dest_url).color(LINK_INK)
            }
            // Tables are laid out as preformatted text for now: a real one wants
            // a TableView, which cannot live inside a RichText, and putting it
            // outside would break selecting the message in one drag.
            Tag::Table(_) | Tag::TableHead | Tag::TableRow => {
                self.block(self.indented(BlockStyle::pre()));
                self.style = self.style.clone().code();
            }
            Tag::TableCell => self.write("  ", self.style.clone()),
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        if let Some(style) = self.stack.pop() {
            self.style = style;
        }
        match tag {
            // A closed block means the next thing starts a new one.
            TagEnd::Paragraph
            | TagEnd::Heading(_)
            | TagEnd::CodeBlock
            | TagEnd::BlockQuote(_)
            | TagEnd::Item
            | TagEnd::Table
            | TagEnd::TableHead => self.pending_block = Some(BlockStyle::default()),
            TagEnd::List(_) => {
                self.list.pop();
                self.pending_block = Some(BlockStyle::default());
            }
            TagEnd::TableRow => self.write("\n", self.style.clone()),
            _ => {}
        }
    }

    fn text(&mut self, text: &str) {
        self.write(text, self.style.clone());
    }

    /// A block style carrying the current list indent, so a code block or a
    /// paragraph inside a list item lines up with the item's text.
    fn indented(&self, style: BlockStyle) -> BlockStyle {
        match self.list.len() {
            0 => style,
            depth => style.indent(INDENT * depth as f32),
        }
    }

    /// Note that the next text belongs to a new block with this style.
    fn block(&mut self, style: BlockStyle) {
        self.block_as(style, false);
    }

    fn block_as(&mut self, style: BlockStyle, item: bool) {
        self.pending_block = Some(style);
        self.pending_item = item;
    }

    fn flush_block(&mut self) {
        self.pending_block = None;
    }

    /// Write text, opening any deferred block first.
    fn write(&mut self, text: &str, style: SpanStyle) {
        if text.is_empty() {
            return;
        }
        if let Some(block) = self.pending_block.take() {
            // A blank line between blocks — except between two list items, where
            // it would turn a tight list loose. Never before the first block.
            let run_on = self.pending_item && self.last_item;
            if !self.at_start && !run_on {
                self.view.push("\n", SpanStyle::default());
            }
            self.view.start_block(block);
            self.at_start = false;
            self.last_item = self.pending_item;
        }
        self.view.push(text, style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The boundary finder is the whole streaming story, so it is tested on its
    /// own: everything else follows from it being right.
    #[test]
    fn a_blank_line_settles_what_came_before_it() {
        // Nothing is settled until a block has certainly ended.
        assert_eq!(last_block_boundary("a paragraph still arriv"), 0);
        assert_eq!(last_block_boundary("a paragraph\n"), 0);
        // The blank line is the end of the block, and it is included.
        assert_eq!(last_block_boundary("a paragraph\n\n"), "a paragraph\n\n".len());
        assert_eq!(last_block_boundary("one\n\ntwo"), "one\n\n".len());
        // The *last* one, not the first.
        assert_eq!(last_block_boundary("one\n\ntwo\n\nthree"), "one\n\ntwo\n\n".len());
    }

    #[test]
    fn a_blank_line_inside_a_fence_is_not_a_boundary() {
        // This is the case that makes a naive split wrong: the blank line is
        // part of the code, and the fence is not a block until it closes.
        let md = "```rust\nfn a() {}\n\nfn b() {}\n";
        assert_eq!(last_block_boundary(md), 0, "an open fence settles nothing");

        // Closed, so the fence itself is a finished block.
        let closed = "```rust\nfn a() {}\n\nfn b() {}\n```\n";
        assert_eq!(last_block_boundary(closed), closed.len());

        // And a fence still being typed settles nothing, even though "``" looks
        // like ordinary text.
        assert_eq!(last_block_boundary("text\n\n``"), "text\n\n".len());
    }

    /// Rendering the same markdown in one go, or in chunks, has to produce the
    /// same text. This is the property `truncate` was added for, checked at the
    /// level that uses it.
    #[test]
    fn streaming_and_whole_rendering_agree() {
        let md = "# Heading\n\nSome **bold** prose and `code`.\n\n```rust\nfn main() {\n    ok();\n}\n```\n\n- one\n- two\n\nAfter the list.";

        let whole = {
            let view = RichText::new(lumio::types::rect((0, 0), (0, 0)), 15.0);
            let mut md_view = MarkdownView::new(15.0);
            md_view.push(&view, md);
            md_view.finish(&view);
            view.get_plain_text()
        };

        // A byte at a time is the worst case: every partial construct is seen.
        let streamed = {
            let view = RichText::new(lumio::types::rect((0, 0), (0, 0)), 15.0);
            let mut md_view = MarkdownView::new(15.0);
            let mut at = 0;
            while at < md.len() {
                let mut next = at + 1;
                while next < md.len() && !md.is_char_boundary(next) {
                    next += 1;
                }
                md_view.push(&view, &md[at..next]);
                at = next;
            }
            md_view.finish(&view);
            view.get_plain_text()
        };

        assert_eq!(whole, streamed, "a streamed reply rendered differently from a whole one");
        // And the code block kept its indentation, which is the thing that was
        // impossible before Lumio's preformatted blocks.
        assert!(whole.contains("    ok();"), "the code block lost its indentation: {whole:?}");
    }

    #[test]
    fn structure_reaches_the_view() {
        let view = RichText::new(lumio::types::rect((0, 0), (0, 0)), 15.0);
        let mut md_view = MarkdownView::new(15.0);
        md_view.push(&view, "text\n\n```\ncode\n```\n\n- item\n");
        md_view.finish(&view);
        let text = view.get_plain_text();
        assert!(text.contains("text"));
        assert!(text.contains("code"));
        assert!(text.contains("• item"), "list marker missing: {text:?}");
    }
}
