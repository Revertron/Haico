//! Reading-mode extraction for fetched web pages.
//!
//! A raw page converted straight to text carries the whole site around with it: navigation,
//! sidebars, footers, cookie banners and a footnote for every link in them. This module does
//! what a browser's reader mode does — Readability picks out the article subtree first, and
//! only that is converted to text.
//!
//! Tables are pulled out before the text conversion and re-rendered as markdown, because
//! html2text draws them as box art padded to the column width, which is both unreadable and
//! by far the largest part of the output on a table-heavy page.

use dom_smoothie::{Config, Readability};
use scraper::node::Node;
use scraper::{ElementRef, Html, Selector};

/// Wrapping width handed to html2text. Deliberately far wider than a terminal: the consumer is
/// a language model, and hard wrapping only splits sentences and URLs across lines.
const WRAP_WIDTH: usize = 3000;

/// Marks where a table was lifted out of the HTML. Carries digits so the decoration-only line
/// filter in `tidy` never mistakes it for markup.
const TABLE_MARK: &str = "%%HAICO-TABLE-";

/// Below this many characters an extracted article is treated as a miss (a page that is mostly
/// not prose), and the whole document is converted instead.
const MIN_ARTICLE_CHARS: usize = 200;

/// Convert a fetched HTML page to compact readable text.
///
/// `url` is the page's final URL, used to resolve relative links into absolute ones.
pub fn html_to_text(html: &str, url: &str) -> String {
    let Some((title, content)) = extract_article(html, url) else {
        return pipeline(html);
    };

    let mut body = pipeline(&content);

    // Readability can succeed and still return very little on a page that is mostly not prose.
    // The whole document is a better answer than a stub.
    if body.len() < MIN_ARTICLE_CHARS {
        let whole = pipeline(html);
        if whole.len() > body.len() {
            body = whole;
        }
    }

    match title.trim() {
        "" => body,
        title => format!("# {}\n\n{}", title, body),
    }
}

/// Run one chunk of HTML through the full conversion: tables out, text render, cleanup, tables
/// back in.
fn pipeline(html: &str) -> String {
    let (without_tables, tables) = lift_tables(html);

    let text = match html2text::config::plain()
        .no_link_wrapping()
        .string_from_read(without_tables.as_bytes(), WRAP_WIDTH)
    {
        Ok(text) => text,
        // html2text only fails on malformed input; the page is worth returning either way.
        Err(_) => return String::new(),
    };

    restore_tables(&tidy(&text), &tables)
}

/// Extract the article subtree, returning its title and HTML. `None` when the page has no
/// recognizable article (Readability's own verdict).
fn extract_article(html: &str, url: &str) -> Option<(String, String)> {
    let config = Config {
        // Long pages are the ones that need cleaning most; do not give up on them.
        max_elements_to_parse: 0,
        ..Default::default()
    };

    let mut readability = Readability::new(html, Some(url), Some(config)).ok()?;
    let article = readability.parse().ok()?;

    Some((article.title.to_string(), article.content.to_string()))
}

/// Replace every top-level `<table>` with a placeholder, returning the stripped HTML and the
/// tables rendered as markdown.
///
/// The tables are swapped out as text rather than converted in place so that html2text still
/// sees one continuous document — its link footnotes are numbered per conversion, so splitting
/// the page into several conversions would restart the numbering.
fn lift_tables(html: &str) -> (String, Vec<String>) {
    // Tag matching is done on an ASCII-lowercased copy. Only ASCII case is folded, so byte
    // offsets stay valid in the original.
    let lower = html.to_ascii_lowercase();

    let mut out = String::with_capacity(html.len());
    let mut tables = Vec::new();
    let mut pos = 0;

    while let Some(start) = find_tag(&lower, pos, "<table") {
        let Some(end) = table_end(&lower, start) else {
            break;
        };

        let markdown = match table_to_markdown(&html[start..end]) {
            Some(markdown) => markdown,
            // An empty or unparseable table: leave it for html2text to deal with.
            None => {
                out.push_str(&html[pos..end]);
                pos = end;
                continue;
            }
        };

        out.push_str(&html[pos..start]);
        // A <pre> keeps the placeholder on a line of its own instead of being absorbed into a
        // neighboring paragraph.
        out.push_str(&format!("<pre>{}{}%%</pre>", TABLE_MARK, tables.len()));
        tables.push(markdown);
        pos = end;
    }

    out.push_str(&html[pos..]);
    (out, tables)
}

/// Find `tag` at or after `from`, where the tag name is followed by a delimiter rather than
/// more name characters (so `<table` does not match `<tablet`).
fn find_tag(lower: &str, from: usize, tag: &str) -> Option<usize> {
    let mut at = from;
    while let Some(offset) = lower[at..].find(tag) {
        let start = at + offset;
        let next = lower[start + tag.len()..].chars().next();
        match next {
            Some(c) if c.is_whitespace() || c == '>' || c == '/' => return Some(start),
            None => return None,
            _ => at = start + tag.len(),
        }
    }
    None
}

/// Byte offset just past the `</table>` matching the `<table>` at `start`, counting nested
/// tables on the way.
fn table_end(lower: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut at = start + "<table".len();

    loop {
        let close = find_tag(lower, at, "</table")?;
        let open = find_tag(lower, at, "<table");

        if let Some(open) = open {
            if open < close {
                depth += 1;
                at = open + "<table".len();
                continue;
            }
        }

        if depth == 0 {
            let gt = lower[close..].find('>')?;
            return Some(close + gt + 1);
        }

        depth -= 1;
        at = close + "</table".len();
    }
}

/// Render one table as a markdown table, or `None` if it holds no rows.
fn table_to_markdown(table_html: &str) -> Option<String> {
    let fragment = Html::parse_fragment(table_html);
    let table = fragment.select(&Selector::parse("table").ok()?).next()?;
    let row_selector = Selector::parse("tr").ok()?;
    let cell_selector = Selector::parse("th, td").ok()?;

    // `Some` marks a cell claimed by an earlier row's rowspan or an earlier cell's colspan.
    let mut grid: Vec<Vec<Option<String>>> = Vec::new();

    let rows = table
        .select(&row_selector)
        .filter(|row| nearest_ancestor(*row, "table").map(|t| t.id()) == Some(table.id()));

    for (r, row) in rows.enumerate() {
        let cells = row
            .select(&cell_selector)
            .filter(|cell| nearest_ancestor(*cell, "tr").map(|t| t.id()) == Some(row.id()));

        let mut c = 0;
        for cell in cells {
            while claimed(&grid, r, c) {
                c += 1;
            }

            let colspan = span_attr(cell, "colspan");
            let rowspan = span_attr(cell, "rowspan");
            let text = cell_markdown(cell);

            for dr in 0..rowspan {
                for dc in 0..colspan {
                    // Markdown has no spans: the top-left cell keeps the text and the cells it
                    // covers are blanked, which keeps every later column in its own place.
                    let value = if dr == 0 && dc == 0 { text.clone() } else { String::new() };
                    put(&mut grid, r + dr, c + dc, value);
                }
            }

            c += colspan;
        }
    }

    // Spacer rows carry no text at all and would render as a row of empty markdown cells.
    grid.retain(|row| {
        row.iter()
            .any(|cell| cell.as_deref().is_some_and(|cell| !cell.is_empty()))
    });

    let width = grid.iter().map(|row| row.len()).max().unwrap_or(0);
    if width == 0 {
        return None;
    }

    let mut out = String::new();
    for (r, row) in grid.iter().enumerate() {
        out.push('|');
        for c in 0..width {
            let cell = row.get(c).and_then(|cell| cell.as_deref()).unwrap_or("");
            out.push(' ');
            out.push_str(cell);
            out.push_str(" |");
        }
        out.push('\n');

        // Markdown needs the separator after the first row, which is the header row whether or
        // not the page marked it up as one.
        if r == 0 {
            out.push('|');
            for _ in 0..width {
                out.push_str(" --- |");
            }
            out.push('\n');
        }
    }

    Some(out)
}

/// The nearest ancestor element with the given tag name, if any.
fn nearest_ancestor<'a>(element: ElementRef<'a>, name: &str) -> Option<ElementRef<'a>> {
    element
        .ancestors()
        .filter_map(ElementRef::wrap)
        .find(|ancestor| ancestor.value().name() == name)
}

/// A `colspan`/`rowspan` value, clamped to something a page could plausibly mean.
fn span_attr(cell: ElementRef, name: &str) -> usize {
    cell.value()
        .attr(name)
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 64)
}

fn claimed(grid: &[Vec<Option<String>>], r: usize, c: usize) -> bool {
    grid.get(r).and_then(|row| row.get(c)).is_some_and(Option::is_some)
}

fn put(grid: &mut Vec<Vec<Option<String>>>, r: usize, c: usize, value: String) {
    while grid.len() <= r {
        grid.push(Vec::new());
    }
    while grid[r].len() <= c {
        grid[r].push(None);
    }
    grid[r][c] = Some(value);
}

/// Cell contents as a single line of markdown, keeping the inline markup that carries meaning.
fn cell_markdown(cell: ElementRef) -> String {
    let mut out = String::new();
    for child in cell.children() {
        write_inline(child, &mut out);
    }

    // Links are inline here rather than footnoted: html2text numbers the footnotes of the
    // surrounding page, and it never sees these cells.
    out.split_whitespace().collect::<Vec<_>>().join(" ").replace('|', "\\|")
}

fn write_inline(node: ego_tree::NodeRef<Node>, out: &mut String) {
    match node.value() {
        Node::Text(text) => out.push_str(text),
        Node::Element(element) => {
            let mut inner = String::new();
            for child in node.children() {
                write_inline(child, &mut inner);
            }

            match element.name() {
                "br" => out.push(' '),
                "strong" | "b" => wrap(out, &inner, "**"),
                "em" | "i" => wrap(out, &inner, "*"),
                "code" | "kbd" | "samp" => wrap(out, &inner, "`"),
                "a" => match element.attr("href") {
                    // Anchor-icon links have no text of their own; there is nothing to label.
                    Some(href) if !inner.trim().is_empty() => {
                        out.push_str(&format!("[{}]({})", inner.trim(), href))
                    }
                    _ => out.push_str(&inner),
                },
                _ => out.push_str(&inner),
            }
        }
        _ => {}
    }
}

fn wrap(out: &mut String, inner: &str, marker: &str) {
    if inner.trim().is_empty() {
        out.push_str(inner);
        return;
    }
    out.push_str(marker);
    out.push_str(inner.trim());
    out.push_str(marker);
}

/// Tidy up html2text's output: drop leftover markup, and keep blank lines as block separators
/// without letting them pile up.
fn tidy(text: &str) -> String {
    let mut out: Vec<&str> = Vec::new();

    for line in text.lines() {
        let line = line.trim_end();
        let trimmed = line.trim();

        // Leftovers from stripped images and empty anchors: a line with markup but no content.
        if !trimmed.is_empty() && !trimmed.chars().any(|c| c.is_alphanumeric()) {
            continue;
        }

        // Collapse runs of blank lines rather than dropping them, so paragraph and section
        // boundaries survive into the text.
        if trimmed.is_empty() && out.last().is_some_and(|last| last.trim().is_empty()) {
            continue;
        }

        out.push(line);
    }

    while out.first().is_some_and(|line| line.trim().is_empty()) {
        out.remove(0);
    }
    while out.last().is_some_and(|line| line.trim().is_empty()) {
        out.pop();
    }

    out.join("\n")
}

/// Put the markdown tables back where their placeholders ended up.
fn restore_tables(text: &str, tables: &[String]) -> String {
    if tables.is_empty() {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        match placeholder_index(line.trim()) {
            Some(index) if index < tables.len() => out.push_str(tables[index].trim_end()),
            _ => out.push_str(line),
        }
        out.push('\n');
    }

    out.truncate(out.trim_end().len());
    out
}

fn placeholder_index(line: &str) -> Option<usize> {
    line.strip_prefix(TABLE_MARK)?.strip_suffix("%%")?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_spans_keep_columns_aligned() {
        let html = r#"<table>
            <tr><th>Category</th><th>Benchmark</th><th>Score</th></tr>
            <tr><td rowspan="2">Agentic</td><td>MCP Atlas</td><td><strong>75.5</strong></td></tr>
            <tr><td>DeepSearch QA</td><td>74.6</td></tr>
        </table>"#;

        let markdown = table_to_markdown(html).unwrap();
        let lines: Vec<&str> = markdown.lines().collect();

        assert_eq!(lines[0], "| Category | Benchmark | Score |");
        assert_eq!(lines[1], "| --- | --- | --- |");
        assert_eq!(lines[2], "| Agentic | MCP Atlas | **75.5** |");
        // The rowspan continuation must leave the first column empty, not shift the row left.
        assert_eq!(lines[3], "|  | DeepSearch QA | 74.6 |");
    }

    #[test]
    fn cell_pipes_are_escaped() {
        let html = "<table><tr><td>a | b</td><td>c</td></tr></table>";
        assert_eq!(table_to_markdown(html).unwrap().lines().next().unwrap(), "| a \\| b | c |");
    }

    #[test]
    fn nested_tables_do_not_leak_rows() {
        let html = "<table><tr><td>outer<table><tr><td>inner</td></tr></table></td></tr></table>";
        let markdown = table_to_markdown(html).unwrap();
        assert_eq!(markdown.lines().count(), 2);
    }

    #[test]
    fn tables_are_lifted_and_restored() {
        let html = "<p>before</p><table><tr><td>cell</td></tr></table><p>after</p>";
        let (stripped, tables) = lift_tables(html);

        assert!(!stripped.contains("<table"));
        assert_eq!(tables.len(), 1);

        let text = pipeline(html);
        assert!(text.contains("before"));
        assert!(text.contains("| cell |"));
        assert!(text.contains("after"));
    }

    #[test]
    fn paragraph_breaks_survive_but_do_not_pile_up() {
        assert_eq!(tidy("a\n\n\n\nb"), "a\n\nb");
        assert_eq!(tidy("  \na\n***\nb\n  "), "a\nb");
    }
}
