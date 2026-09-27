//! The library: long articles about crates and topics, searched and read by the `library`
//! sub-agent on the main model's behalf.
//!
//! Where project memory holds short facts about *this* code, the library holds what a
//! model needs to know about the crates the code is written against — how mio's event loop
//! is meant to be driven, what changed in actix-web 4, which tokio types are `Send`. That
//! knowledge is what a small local model gets wrong, and it is the same in every project,
//! so the library lives in haico's store rather than in the project.
//!
//! **Articles are markdown files under `library/` in the store**, conventionally
//! `library/<crate>/<topic>.md`. Frontmatter is optional and carries `title`, `crate`,
//! `versions` and `summary`; without it the title is the first heading and the crate is the
//! directory the file sits in. An article's identity is its path, `tokio/select`, for the
//! reason a memory's is its file stem: that is the name that can actually be opened again.
//!
//! **Search is by section, not by article.** An article is split at its headings, and a
//! query is ranked against sections with BM25. Keyword ranking rather than embeddings,
//! because the GPU is already holding the coding model — the argument `research` makes for
//! not loading a second model — and because what gets searched for here is mostly exact
//! tokens: `E0599`, `Poll::poll`, `default_service`. Those are what keyword ranking is good
//! at and what embeddings blur.
//!
//! Everything is re-read from disk on every call. A library is dozens of files, a search
//! is a millisecond, and a cache would be one more thing that can disagree with the
//! directory — the lesson memory's index already taught.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// A section is what the sub-agent reads in one call. Articles are written to be read a
/// section at a time, so this only bites on a section that should have been split; the cap
/// is `research`'s page cap, for the same reason — every round re-sends every section read so
/// far, and the librarian works against a clock.
pub(crate) const MAX_SECTION_CHARS: usize = 8000;

/// How many sections one search returns.
const SEARCH_HITS: usize = 8;

/// How much of an article's summary the index carries per line.
const MAX_SUMMARY_CHARS: usize = 200;

// BM25's two knobs, at the values every implementation defaults to.
const K1: f64 = 1.2;
const B: f64 = 0.75;

// How many times a heading's words count against the body's. A heading is a statement of
// what the section is about, so "Graceful shutdown" matching the query is worth more than
// the word "shutdown" appearing once in a paragraph about something else.
const HEADING_WEIGHT: usize = 3;

pub(crate) struct Article {
    /// The path under the library root, `/`-separated, without `.md`: `tokio/select`.
    pub(crate) id: String,
    pub(crate) title: String,
    /// The crate the article is about, as written in `Cargo.toml` (`actix-web`), or empty
    /// for an article that is about Rust itself rather than one crate.
    pub(crate) crate_name: String,
    pub(crate) versions: Option<String>,
    pub(crate) summary: String,
    pub(crate) sections: Vec<Section>,
}

pub(crate) struct Section {
    /// "Graceful shutdown", or "Channels > Bounded channels" under a parent heading. Empty
    /// for the text before the first heading.
    pub(crate) heading: String,
    pub(crate) text: String,
}

pub(crate) struct Library {
    pub(crate) articles: Vec<Article>,
}

pub(crate) struct Hit {
    /// `tokio/select#2` — what `read` takes.
    pub(crate) id: String,
    pub(crate) title: String,
    pub(crate) heading: String,
    pub(crate) snippet: String,
}

/// The library in haico's store.
pub(crate) fn user_dir() -> Option<PathBuf> {
    crate::store::path("library")
}

impl Library {
    /// Every article under `root`, sorted by id.
    ///
    /// A directory that cannot be read is an empty library, and a file that cannot be read
    /// is left out: failing a whole lookup over one unreadable file would trade a small loss
    /// for a total one, which is the rule skills and memory already follow.
    pub(crate) fn load(root: &Path) -> Library {
        let mut articles = Vec::new();
        collect(root, root, &mut articles);
        // Sorted so the prompt's index is the same between sessions — a reordered index is a
        // change to every prefix cache that saw it.
        articles.sort_by(|a, b| a.id.cmp(&b.id));
        Library { articles }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.articles.is_empty()
    }

    /// The crates the library has something on, as `Cargo.toml` spells them, deduplicated.
    pub(crate) fn crates(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for article in &self.articles {
            if !article.crate_name.is_empty() && !out.contains(&article.crate_name) {
                out.push(article.crate_name.clone());
            }
        }
        out.sort();
        out
    }

    /// The best-matching sections for a query, best first.
    pub(crate) fn search(&self, query: &str) -> Vec<Hit> {
        let terms: Vec<String> = {
            let mut seen = HashSet::new();
            tokens(query).into_iter().filter(|t| seen.insert(t.clone())).collect()
        };
        if terms.is_empty() {
            return Vec::new();
        }

        // One document per section. The article's title and crate go into every one of its
        // sections, so "tokio select" finds the right article even when a section's own text
        // never says "tokio".
        let mut docs: Vec<(usize, usize, HashMap<String, usize>, usize)> = Vec::new();
        for (a, article) in self.articles.iter().enumerate() {
            let mut header = tokens(&article.title);
            header.extend(tokens(&article.crate_name));
            for (s, section) in article.sections.iter().enumerate() {
                let mut all = header.clone();
                // Tags count as much as the heading: they are the author saying what the
                // section is for, in the words someone with the problem would use.
                let mut heading = tokens(&section.heading);
                heading.extend(tokens(section_tags(&section.text).unwrap_or("")));
                for _ in 0..HEADING_WEIGHT {
                    all.extend(heading.iter().cloned());
                }
                all.extend(tokens(&section.text));
                let len = all.len();
                let mut counts = HashMap::new();
                for token in all {
                    *counts.entry(token).or_insert(0usize) += 1;
                }
                docs.push((a, s, counts, len));
            }
        }
        if docs.is_empty() {
            return Vec::new();
        }

        let n = docs.len() as f64;
        let average = docs.iter().map(|d| d.3).sum::<usize>() as f64 / n;
        let idf: HashMap<&str, f64> = terms
            .iter()
            .map(|term| {
                let df = docs.iter().filter(|d| d.2.contains_key(term)).count() as f64;
                // The "+ 1" form, which never goes negative: a term in most sections still
                // counts for a little rather than counting against a match.
                (term.as_str(), ((n - df + 0.5) / (df + 0.5) + 1.0).ln())
            })
            .collect();

        let mut scored: Vec<(f64, usize, usize)> = docs
            .iter()
            .filter_map(|(a, s, counts, len)| {
                let score: f64 = terms
                    .iter()
                    .filter_map(|term| {
                        let tf = *counts.get(term)? as f64;
                        let norm = K1 * (1.0 - B + B * *len as f64 / average);
                        Some(idf[term.as_str()] * tf * (K1 + 1.0) / (tf + norm))
                    })
                    .sum();
                (score > 0.0).then_some((score, *a, *s))
            })
            .collect();
        scored.sort_by(|x, y| y.0.total_cmp(&x.0));

        scored
            .into_iter()
            .take(SEARCH_HITS)
            .map(|(_, a, s)| {
                let article = &self.articles[a];
                let section = &article.sections[s];
                Hit {
                    id: format!("{}#{}", article.id, s),
                    title: article.title.clone(),
                    heading: section.heading.clone(),
                    snippet: snippet(&section.text, &terms),
                }
            })
            .collect()
    }

    /// What `read` hands back for an id: one section for `article#n`, or the article's
    /// outline for a bare `article`.
    ///
    /// The outline rather than the whole article, because a whole article is exactly the
    /// thing that does not fit — the sub-agent picks the sections it needs from the list.
    pub(crate) fn read(&self, id: &str) -> Result<String, String> {
        let id = id.trim().trim_start_matches('/').replace('\\', "/");
        let (article_id, number) = match id.split_once('#') {
            Some((a, n)) => (a.trim_end_matches(".md"), Some(n.trim())),
            None => (id.trim_end_matches(".md"), None),
        };

        let Some(article) = self.find(article_id) else {
            return Err(format!(
                "There is no article '{}'. Use an id exactly as library_search returned it, \
                 like 'tokio/select#2', or search again.",
                article_id
            ));
        };

        let Some(number) = number else {
            return Ok(outline(article));
        };
        let section = number
            .parse::<usize>()
            .ok()
            .and_then(|n| article.sections.get(n).map(|s| (n, s)));
        let Some((n, section)) = section else {
            return Err(format!(
                "'{}' has no section {}. Its sections are:\n{}",
                article.id,
                number,
                section_list(article)
            ));
        };

        let mut out = format!("# {} — {}", article.title, article.id);
        if let Some(versions) = &article.versions {
            out.push_str(&format!(" (versions: {})", versions));
        }
        out.push('\n');
        if !section.heading.is_empty() {
            out.push_str(&format!("## {}\n", section.heading));
        }
        out.push('\n');
        let text = section.text.trim();
        let cut = crate::util::truncate_at_boundary(text, MAX_SECTION_CHARS);
        out.push_str(cut);
        if cut.len() < text.len() {
            out.push_str(&format!(
                "\n\n[Section cut off at {} characters.]",
                MAX_SECTION_CHARS
            ));
        }
        if n + 1 < article.sections.len() {
            out.push_str(&format!(
                "\n\n(Next section: {}#{} — {})",
                article.id,
                n + 1,
                article.sections[n + 1].heading
            ));
        }
        Ok(out)
    }

    /// The article an id names, forgiving case — a model that types `Tokio/Select` meant it.
    fn find(&self, id: &str) -> Option<&Article> {
        self.articles
            .iter()
            .find(|a| a.id == id)
            .or_else(|| self.articles.iter().find(|a| a.id.eq_ignore_ascii_case(id)))
    }

    /// One line per article: id, title, versions and summary. Given to the sub-agent whole,
    /// so it can open an article it can see is the right one without searching first.
    pub(crate) fn catalog(&self) -> String {
        let mut out = String::new();
        for article in &self.articles {
            out.push_str(&format!("- {} — {}", article.id, article.title));
            if let Some(versions) = &article.versions {
                out.push_str(&format!(" ({})", versions));
            }
            if !article.summary.is_empty() {
                out.push_str(&format!(": {}", article.summary));
            }
            out.push('\n');
        }
        out
    }
}

/// The index as the system prompt carries it: one line per crate, naming its articles.
///
/// Per crate rather than per article, because what the main model needs from it is "the
/// library knows about mio", not a table of contents — the sub-agent has the table of
/// contents. `None` for an empty library, so the prompt says nothing about a tool that
/// would have nothing to say.
pub(crate) fn index_block(root: &Path) -> Option<String> {
    let library = Library::load(root);
    if library.is_empty() {
        return None;
    }
    let mut groups: Vec<(String, Vec<&Article>)> = Vec::new();
    for article in &library.articles {
        let key = if article.crate_name.is_empty() {
            "general".to_string()
        } else {
            article.crate_name.clone()
        };
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, list)) => list.push(article),
            None => groups.push((key, vec![article])),
        }
    }
    groups.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out = String::new();
    for (name, articles) in groups {
        let titles: Vec<&str> = articles.iter().map(|a| a.title.as_str()).collect();
        out.push_str(&format!("- {}: {}\n", name, titles.join("; ")));
    }
    Some(out)
}

// Walk the tree, reading every `.md` file as an article.
fn collect(root: &Path, dir: &Path, out: &mut Vec<Article>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(root, &path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        let id = relative
            .with_extension("")
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        out.push(parse_article(id, &text));
    }
}

pub(crate) fn parse_article(id: String, text: &str) -> Article {
    use crate::skills::{front_value, split_frontmatter};

    let (front, body) = split_frontmatter(text);
    let mut sections = split_sections(body);

    // A body that opens with a single `#` heading has that heading as its document title.
    // It is dropped from every section's heading path whether or not the frontmatter also
    // names a title, because under it every section would read "Title > Imports", repeating
    // the article's name in each search hit and in the headings the ranking weights up.
    let top = sections
        .iter()
        .find(|s| !s.heading.is_empty())
        .filter(|_| top_level(body))
        .map(|first| first.heading.split(" > ").next().unwrap_or_default().to_string());
    if let Some(top) = &top {
        for section in &mut sections {
            section.heading = strip_title(&section.heading, top);
        }
    }

    // The title is the frontmatter's, then that top heading, then the article's name.
    let title = front_value(front, "title")
        .filter(|t| !t.is_empty())
        .or(top)
        .unwrap_or_else(|| id.rsplit('/').next().unwrap_or(&id).to_string());
    // A title heading with nothing under it is now a section with no heading and no text:
    // nothing to read, and a search hit on it would be a read spent on an empty page.
    if sections.len() > 1 {
        sections.retain(|s| !(s.heading.is_empty() && s.text.trim().is_empty()));
    }

    // The directory is the crate unless the frontmatter says otherwise; an article at the
    // top of the library is about no crate in particular.
    let crate_name = front_value(front, "crate")
        .or_else(|| id.rsplit_once('/').map(|(dir, _)| dir.split('/').next().unwrap_or(dir).to_string()))
        .unwrap_or_default();

    let summary = front_value(front, "summary")
        .or_else(|| front_value(front, "description"))
        .unwrap_or_else(|| {
            // The first paragraph line is what the author would have said it was about.
            sections
                .iter()
                .flat_map(|s| s.text.lines())
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with("```"))
                .unwrap_or("")
                .to_string()
        });
    let summary = one_line(&summary, MAX_SUMMARY_CHARS);

    Article {
        id,
        title,
        crate_name,
        versions: front_value(front, "versions").filter(|v| !v.is_empty()),
        summary,
        sections,
    }
}

// Whether the body's first heading is a single `#` — the markdown convention for "this is
// the document's title" rather than its first section.
fn top_level(body: &str) -> bool {
    body.lines()
        .find_map(heading_level)
        .is_some_and(|(level, _)| level == 1)
}

fn strip_title(heading: &str, title: &str) -> String {
    if heading == title {
        return String::new();
    }
    heading
        .strip_prefix(title)
        .and_then(|rest| rest.strip_prefix(" > "))
        .unwrap_or(heading)
        .to_string()
}

// A markdown ATX heading's level and text, or None for any other line.
fn heading_level(line: &str) -> Option<(usize, String)> {
    let level = line.chars().take_while(|&c| c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &line[level..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    Some((level, rest.trim().trim_end_matches('#').trim().to_string()))
}

/// Split a markdown body at its headings.
///
/// Headings inside code fences are code, not structure — a Rust article's code blocks are
/// full of `#[derive(..)]` and shell comments, and taking those as headings would cut the
/// article into nonsense. Each section is named by its heading's path, so a `###` under
/// "Channels" reads "Channels > Bounded", which says where it is without the outline.
fn split_sections(body: &str) -> Vec<Section> {
    let mut sections = vec![Section {
        heading: String::new(),
        text: String::new(),
    }];
    let mut path: Vec<(usize, String)> = Vec::new();
    let mut fence: Option<&str> = None;

    for line in body.lines() {
        let trimmed = line.trim_start();
        let marker = if trimmed.starts_with("```") {
            Some("```")
        } else if trimmed.starts_with("~~~") {
            Some("~~~")
        } else {
            None
        };
        match (fence, marker) {
            (None, Some(m)) => fence = Some(m),
            (Some(open), Some(m)) if open == m => fence = None,
            _ => {}
        }

        if fence.is_none() && marker.is_none() {
            if let Some((level, text)) = heading_level(line) {
                while path.last().is_some_and(|(l, _)| *l >= level) {
                    path.pop();
                }
                path.push((level, text));
                sections.push(Section {
                    heading: path.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join(" > "),
                    text: String::new(),
                });
                continue;
            }
        }
        let current = sections.last_mut().unwrap();
        current.text.push_str(line);
        current.text.push('\n');
    }

    // The text before the first heading is a section only if there is any.
    if sections[0].text.trim().is_empty() && sections.len() > 1 {
        sections.remove(0);
    }
    sections
}

fn outline(article: &Article) -> String {
    let mut out = format!("# {} — {}\n", article.title, article.id);
    if let Some(versions) = &article.versions {
        out.push_str(&format!("Versions: {}\n", versions));
    }
    if !article.summary.is_empty() {
        out.push_str(&format!("{}\n", article.summary));
    }
    out.push_str("\nSections (read one with library_read and its id):\n");
    out.push_str(&section_list(article));
    out
}

fn section_list(article: &Article) -> String {
    article
        .sections
        .iter()
        .enumerate()
        .map(|(n, s)| {
            let heading = if s.heading.is_empty() { "(introduction)" } else { &s.heading };
            format!("- {}#{} — {} ({} chars)\n", article.id, n, heading, s.text.trim().len())
        })
        .collect()
}

/// A section's tags: its first non-empty line, when that line starts with `Tags:`.
///
/// They exist for the gap between how a problem is described and how it is explained. A model
/// with a stuck server asks about "hangs" and "never answers"; the section that fixes it is
/// titled "Draining readiness" and talks about `WouldBlock`, and no stemmer joins those words.
/// Kept in the text rather than hidden, because the librarian reading the section benefits
/// from the same line: it says what the section is the answer to.
pub(crate) fn section_tags(text: &str) -> Option<&str> {
    let first = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    let (key, rest) = first.split_once(':')?;
    key.trim().eq_ignore_ascii_case("tags").then_some(rest.trim())
}

/// Words for ranking: lowercase runs of letters, digits and underscores, plus the parts of
/// a snake_case name, so `spawn_blocking` is found by "spawn blocking" and by itself.
///
/// Rust paths come apart at `::` on their own, which is what makes `tokio::sync::mpsc` match
/// a question that only says "mpsc".
pub(crate) fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        let word = word.trim_matches('_').to_lowercase();
        if word.chars().count() < 2 || STOPWORDS.contains(&word.as_str()) {
            continue;
        }
        if word.contains('_') {
            for part in word.split('_') {
                if part.chars().count() >= 2 && !STOPWORDS.contains(&part) {
                    out.push(part.to_string());
                }
            }
        }
        out.push(word);
    }
    out
}

// Words a question is made of that say nothing about what it is asking. BM25's IDF would
// mostly discount them anyway; dropping them keeps a question phrased as a whole sentence
// from being ranked on "how" and "the".
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "has", "have", "how", "what",
    "when", "where", "which", "who", "why", "with", "this", "that", "from", "into", "does", "did",
    "was", "were", "will", "would", "should", "could", "is", "it", "in", "on", "of", "to", "an",
    "as", "at", "be", "by", "do", "if", "or", "so", "use", "using", "get", "my", "me", "there",
    "their", "then", "than", "its", "about",
];

// The line of a section that best shows why it matched: the one holding the most distinct
// query words, cut short. Falls back to the section's first line.
fn snippet(text: &str, terms: &[String]) -> String {
    let best = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|line| {
            let words: HashSet<String> = tokens(line).into_iter().collect();
            (terms.iter().filter(|t| words.contains(*t)).count(), line)
        })
        .fold((0usize, ""), |best, next| if next.0 > best.0 { next } else { best });
    let line = if best.0 > 0 {
        best.1
    } else {
        text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("")
    };
    one_line(line, 160)
}

fn one_line(text: &str, max: usize) -> String {
    let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flattened.chars().count() <= max {
        return flattened;
    }
    let cut: String = flattened.chars().take(max - 1).collect();
    format!("{}…", cut.trim_end())
}

/// Crate names as they appear in Rust code: `actix-web` is written `actix_web`.
fn code_name(crate_name: &str) -> String {
    crate_name.replace('-', "_").to_lowercase()
}

/// The dependencies a `Cargo.toml` names, from every dependency table it has.
///
/// Renamed dependencies (`web = { package = "actix-web" }`) are reported by their package,
/// which is the name the library files them under.
pub(crate) fn cargo_dependencies(manifest: &str) -> Vec<String> {
    let Ok(table) = manifest.parse::<toml::Table>() else {
        return Vec::new();
    };
    let mut tables: Vec<&toml::Table> = Vec::new();
    for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(t) = table.get(key).and_then(|v| v.as_table()) {
            tables.push(t);
        }
        if let Some(t) = table
            .get("workspace")
            .and_then(|w| w.get(key))
            .and_then(|v| v.as_table())
        {
            tables.push(t);
        }
    }
    if let Some(targets) = table.get("target").and_then(|v| v.as_table()) {
        for target in targets.values() {
            for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
                if let Some(t) = target.get(key).and_then(|v| v.as_table()) {
                    tables.push(t);
                }
            }
        }
    }

    let mut out = Vec::new();
    for t in tables {
        for (name, spec) in t {
            let name = spec
                .get("package")
                .and_then(|p| p.as_str())
                .unwrap_or(name)
                .to_string();
            if !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

/// The error lines of a failed cargo build, deduplicated, in the order they appeared.
///
/// Only the headlines — `error[E0599]: no method named ...` — and not cargo's own closing
/// remarks ("could not compile", "aborting due to"), which say nothing about what went
/// wrong and are the same for every failure.
pub(crate) fn compile_errors(output: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        let is_error = line.starts_with("error[E") || line.starts_with("error:");
        if !is_error {
            continue;
        }
        let noise = [
            "error: could not compile",
            "error: aborting due to",
            "error: test failed",
            "error: process didn't exit",
            "error: Recipe",
        ];
        if noise.iter().any(|n| line.starts_with(n)) {
            continue;
        }
        let line = line.to_string();
        if !out.contains(&line) {
            out.push(line);
        }
    }
    out
}

/// Whether a shell command is a cargo build of some kind — the commands whose failures are
/// compile errors worth looking up.
pub(crate) fn is_cargo_build(command: &str) -> bool {
    let words: Vec<&str> = command.split_whitespace().collect();
    words.windows(2).any(|pair| {
        let program = pair[0].trim_matches('"');
        let program = program.rsplit(['/', '\\']).next().unwrap_or(program);
        (program == "cargo" || program == "cargo.exe")
            && matches!(
                pair[1],
                "build" | "b" | "check" | "c" | "test" | "t" | "run" | "r" | "clippy" | "bench"
            )
    }) || words.windows(3).any(|w| {
        // `cargo +nightly build`
        (w[0] == "cargo" || w[0] == "cargo.exe")
            && w[1].starts_with('+')
            && matches!(w[2], "build" | "check" | "test" | "run" | "clippy")
    })
}

/// What a test run that compiled says went wrong: tests that failed, and tests libtest reported
/// as still running after a minute.
#[derive(Default, Debug, PartialEq)]
pub(crate) struct TestTrouble {
    pub(crate) failed: Vec<String>,
    pub(crate) hung: Vec<String>,
}

pub(crate) fn test_trouble(output: &str) -> TestTrouble {
    let mut trouble = TestTrouble::default();
    for line in output.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("test ") else { continue };
        if let Some(name) = rest.strip_suffix(" ... FAILED") {
            let name = name.trim().to_string();
            if !trouble.failed.contains(&name) {
                trouble.failed.push(name);
            }
        } else if let Some((name, _)) = rest.split_once(" has been running for over") {
            let name = name.trim().to_string();
            if !trouble.hung.contains(&name) {
                trouble.hung.push(name);
            }
        }
    }
    trouble
}

/// The note for a build that compiled and then failed at runtime — tests failing, tests hanging,
/// or the whole command killed at its timeout.
///
/// A compile error names the API it is about; a runtime failure names nothing, and that is where
/// a model stops asking and starts guessing. The first run this was built for compiled its mio
/// server and then spent forty minutes splitting the test suite by hand to find a hang, with the
/// causes (EOF ignored, a connection dropped before its last reply was flushed) written up in the
/// library the whole time. So the note names the symptom, and hands over a question phrased the
/// way the articles' tags are: by what went wrong, not by what caused it.
pub(crate) fn runtime_failure_note(crates: &[String], trouble: &TestTrouble, timed_out: bool) -> Option<String> {
    if crates.is_empty() || (trouble.failed.is_empty() && trouble.hung.is_empty() && !timed_out) {
        return None;
    }
    let names = match crates {
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {}", init.join(", "), last),
        [] => unreachable!(),
    };
    let listed = |tests: &[String]| {
        let shown: Vec<&str> = tests.iter().take(3).map(String::as_str).collect();
        let more = tests.len().saturating_sub(3);
        if more > 0 { format!("{} (and {} more)", shown.join(", "), more) } else { shown.join(", ") }
    };
    let hanging = timed_out || !trouble.hung.is_empty();
    let (what, symptom) = if hanging {
        let which = if trouble.hung.is_empty() { String::new() } else { format!(": {}", listed(&trouble.hung)) };
        (format!("Something hangs{}", which), "a test hangs and never finishes".to_string())
    } else {
        (format!("Tests fail: {}", listed(&trouble.failed)), format!("test {} fails", trouble.failed[0]))
    };
    Some(format!(
        "[haico] {what}, in a project that uses {names}. The library's articles on {names} cover \
         the usual causes of hangs, lost replies, resets and busy loops, by symptom. Before \
         debugging further, call library and describe the symptom — for example: \
         library(question: \"{krate}: {symptom}. What are the usual causes?\").",
        what = what,
        names = names,
        krate = crates[0],
        symptom = symptom.replace('"', "'"),
    ))
}

/// The directory a command starts by moving into: `cd app && cargo build`, `cd /d "C:\x y" && ..`.
/// Only a leading `cd` joined by `&&` or `;`, since that is the one whose effect reaches the
/// rest of the line.
pub(crate) fn leading_cd(command: &str) -> Option<PathBuf> {
    let re = regex::Regex::new(r#"^\s*cd\s+(?:/d\s+)?(?:"([^"]+)"|'([^']+)'|([^\s&;|]+))\s*(?:&&|;)"#).ok()?;
    let caps = re.captures(command)?;
    let dir = caps.get(1).or(caps.get(2)).or(caps.get(3))?.as_str();
    Some(PathBuf::from(dir))
}

/// The library's crates that a failed build involves: named in the errors, or depended on
/// by the project being built. Errors first, because a crate named in the error itself is
/// the likelier culprit.
pub(crate) fn involved_crates(library_crates: &[String], output: &str, dependencies: &[String]) -> Vec<String> {
    let lower = output.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    for name in library_crates {
        let code = code_name(name);
        // `mio::Poll`, `use mio;`, or a registry path like `mio-1.0.2\src`.
        // Anchored at the start of a path, not just a word: `\bio::` would find a crate
        // called `io` in every `std::io::`.
        let pattern = format!(
            r"(?m)(?:^|[^\w:]){code}::|`{code}`|\b{name}-\d+\.\d+",
            code = regex::escape(&code),
            name = regex::escape(&name.to_lowercase())
        );
        let named = regex::Regex::new(&pattern).is_ok_and(|re| re.is_match(&lower));
        if named && !out.contains(name) {
            out.push(name.clone());
        }
    }
    for name in library_crates {
        let depended = dependencies.iter().any(|d| code_name(d) == code_name(name));
        if depended && !out.contains(name) {
            out.push(name.clone());
        }
    }
    out
}

/// The note added under a failed build's output, or `None` when there is nothing to say.
///
/// It names the crates and quotes the first error as a ready-made question, because a small
/// model told only "you could use the library" rarely works out what to ask — and one that
/// is handed the question usually does ask it.
pub(crate) fn build_failure_note(crates: &[String], errors: &[String]) -> Option<String> {
    if crates.is_empty() || errors.is_empty() {
        return None;
    }
    let names = match crates {
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {}", init.join(", "), last),
        [] => unreachable!(),
    };
    let first = one_line(&errors[0], 200);
    Some(format!(
        "[haico] The library has articles on {names}, which this build involves. Before \
         changing the code again, call library with a question about these errors — \
         for example: library(question: \"{crate} — how do I fix: {first}\"). It answers \
         from documentation written for exactly this, which is more reliable than guessing \
         at the API.",
        names = names,
        crate = crates[0],
        first = first.replace('"', "'"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn write(root: &Path, id: &str, text: &str) {
        let path = root.join(format!("{}.md", id));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    const SELECT: &str = "---\nsummary: Waiting on several futures at once.\nversions: \"1.x\"\n---\n\
# tokio::select!\n\nIntro text.\n\n## Cancellation\n\nThe losing branches are dropped.\n\n\
```rust\n# hidden doc line\nlet x = 1;\n```\n\n## Loops\n\nUse `biased;` to poll in order.\n\n\
### Borrowing in loops\n\nPin the future with tokio::pin! before the loop.\n";

    #[test]
    fn an_article_is_split_at_its_headings_and_not_inside_code() {
        let article = parse_article("tokio/select".into(), SELECT);
        assert_eq!(article.title, "tokio::select!");
        assert_eq!(article.crate_name, "tokio");
        assert_eq!(article.versions.as_deref(), Some("1.x"));
        assert_eq!(article.summary, "Waiting on several futures at once.");

        let headings: Vec<&str> = article.sections.iter().map(|s| s.heading.as_str()).collect();
        // The title heading is the article's, not a section path prefix; the `#` inside the
        // code block is code.
        assert_eq!(headings, ["", "Cancellation", "Loops", "Loops > Borrowing in loops"]);
        assert!(article.sections[1].text.contains("# hidden doc line"));
    }

    #[test]
    fn frontmatter_can_name_the_crate_and_the_title() {
        let article = parse_article(
            "web/routing".into(),
            "---\ntitle: Routing\ncrate: actix-web\n---\n## Scopes\n\ntext\n",
        );
        assert_eq!(article.title, "Routing");
        assert_eq!(article.crate_name, "actix-web");
        assert_eq!(article.summary, "text");
    }

    #[test]
    fn a_title_heading_is_not_repeated_in_front_of_every_section_even_with_a_frontmatter_title() {
        let article = parse_article(
            "mio/basics".into(),
            "---\ntitle: \"mio: basics\"\n---\n# mio: the basics\n\nIntro.\n\n## Imports\n\nx\n\n### Read\n\ny\n",
        );
        assert_eq!(article.title, "mio: basics");
        let headings: Vec<&str> = article.sections.iter().map(|s| s.heading.as_str()).collect();
        assert_eq!(headings, ["", "Imports", "Imports > Read"]);
    }

    #[test]
    fn a_top_level_article_has_no_crate() {
        let article = parse_article("lifetimes".into(), "Borrowing rules.\n");
        assert_eq!(article.crate_name, "");
        assert_eq!(article.title, "lifetimes");
        assert_eq!(article.sections.len(), 1);
    }

    #[test]
    fn search_ranks_the_section_that_answers_first() {
        let dir = TempDir::new("library-search");
        write(dir.path(), "tokio/select", SELECT);
        write(
            dir.path(),
            "mio/poll",
            "# Polling\n\n## Registering\n\nCall registry().register(&mut socket, TOKEN, Interest::READABLE).\n\n\
## Events\n\nPoll::poll(&mut events, timeout) fills events.\n",
        );
        let library = Library::load(dir.path());
        assert_eq!(library.articles.len(), 2);

        let hits = library.search("how do I pin a future in a select loop?");
        assert_eq!(hits[0].id, "tokio/select#3", "{:?}", hits.iter().map(|h| &h.id).collect::<Vec<_>>());
        assert!(hits[0].snippet.contains("tokio::pin"), "{}", hits[0].snippet);

        // Paths and snake_case come apart into the words a question uses.
        let hits = library.search("Interest READABLE register");
        assert_eq!(hits[0].id, "mio/poll#0");
        assert!(library.search("zebra").is_empty());
        assert!(library.search("the and of").is_empty());
    }

    #[test]
    fn tags_under_a_heading_rank_a_section_for_words_its_text_never_uses() {
        let dir = TempDir::new("library-tags");
        write(
            dir.path(),
            "mio/loop",
            "# Loop

## Draining readiness

Tags: hangs, stalls, never answered

Read until WouldBlock.

## Hangs in general

A server can stop for many reasons.
",
        );
        let library = Library::load(dir.path());
        // "stalls" appears only in the tags, so without them this section could not match at all.
        assert_eq!(library.search("server stalls")[0].id, "mio/loop#0");
        // Weighted like the heading, the tag outranks a section that merely says "hangs" once
        // in its heading and body.
        let hits = library.search("never answered hangs");
        assert_eq!(hits[0].id, "mio/loop#0", "{:?}", hits.iter().map(|h| &h.id).collect::<Vec<_>>());

        assert_eq!(section_tags("
  Tags: a, b
text"), Some("a, b"));
        assert_eq!(section_tags("tags:x"), Some("x"));
        assert_eq!(section_tags("Text first.
Tags: a"), None);
        assert_eq!(section_tags("Contags: a"), None);
    }

    #[test]
    fn read_returns_a_section_or_an_outline_and_refuses_what_is_not_there() {
        let dir = TempDir::new("library-read");
        write(dir.path(), "tokio/select", SELECT);
        let library = Library::load(dir.path());

        let section = library.read("tokio/select#2").unwrap();
        assert!(section.contains("biased;"), "{}", section);
        assert!(section.contains("versions: 1.x"), "{}", section);
        assert!(section.contains("Next section: tokio/select#3"), "{}", section);

        let outline = library.read("Tokio/Select").unwrap();
        assert!(outline.contains("tokio/select#1 — Cancellation"), "{}", outline);

        assert!(library.read("tokio/nothing").unwrap_err().contains("no article"));
        let missing = library.read("tokio/select#9").unwrap_err();
        assert!(missing.contains("tokio/select#2"), "{}", missing);
    }

    #[test]
    fn the_index_groups_articles_by_crate() {
        let dir = TempDir::new("library-index");
        write(dir.path(), "tokio/select", SELECT);
        write(dir.path(), "tokio/channels", "# Channels\n\nmpsc.\n");
        write(dir.path(), "mio/poll", "# Polling\n\nx\n");
        let index = index_block(dir.path()).unwrap();
        assert_eq!(index, "- mio: Polling\n- tokio: Channels; tokio::select!\n");

        let empty = TempDir::new("library-empty");
        assert!(index_block(empty.path()).is_none());
        assert!(index_block(&empty.join("missing")).is_none());
    }

    #[test]
    fn dependencies_come_from_every_table_and_by_their_package_name() {
        let deps = cargo_dependencies(
            "[package]\nname = \"x\"\n\n[dependencies]\ntokio = \"1\"\nweb = { package = \"actix-web\", version = \"4\" }\n\n\
[dev-dependencies]\ngetopts = \"0.2\"\n\n[target.'cfg(unix)'.dependencies]\nmio = \"1\"\n",
        );
        assert_eq!(deps, ["tokio", "actix-web", "getopts", "mio"]);
        assert!(cargo_dependencies("not toml [").is_empty());
    }

    #[test]
    fn only_the_error_headlines_of_a_build_count() {
        let output = "   Compiling x v0.1.0\n\
error[E0599]: no method named `poll` found for struct `mio::Poll`\n  --> src/main.rs:3:5\n\
error[E0599]: no method named `poll` found for struct `mio::Poll`\n\
warning: unused import\n\
error: aborting due to 2 previous errors\n\
error: could not compile `x` (bin \"x\")\n";
        assert_eq!(
            compile_errors(output),
            ["error[E0599]: no method named `poll` found for struct `mio::Poll`"]
        );
    }

    #[test]
    fn cargo_builds_are_recognized_and_other_commands_are_not() {
        assert!(is_cargo_build("cargo build --release"));
        assert!(is_cargo_build("cd app && cargo check"));
        assert!(is_cargo_build("cargo +nightly test"));
        assert!(is_cargo_build("C:\\Users\\me\\.cargo\\bin\\cargo.exe t"));
        assert!(!is_cargo_build("cargo fmt"));
        assert!(!is_cargo_build("echo cargo"));
        assert!(!is_cargo_build("cargo add tokio"));
    }

    #[test]
    fn a_leading_cd_names_the_build_directory() {
        assert_eq!(leading_cd("cd app && cargo build"), Some(PathBuf::from("app")));
        assert_eq!(leading_cd(r#"cd /d "C:\my dir" && cargo test"#), Some(PathBuf::from(r"C:\my dir")));
        assert_eq!(leading_cd(r"  cd E:\x\y; cargo test"), Some(PathBuf::from(r"E:\x\y")));
        assert_eq!(leading_cd("cargo build"), None);
        assert_eq!(leading_cd("cd app"), None);
        assert_eq!(leading_cd("echo cd app && cargo build"), None);
    }

    #[test]
    fn failing_and_hanging_tests_are_read_from_libtest_output() {
        let output = "running 3 tests\n\
test reverses_a_line ... ok\n\
test burst ... FAILED\n\
test oversized has been running for over 60 seconds\n\
test burst ... FAILED\n";
        let trouble = test_trouble(output);
        assert_eq!(trouble.failed, ["burst"]);
        assert_eq!(trouble.hung, ["oversized"]);
        assert_eq!(test_trouble("test result: ok. 3 passed"), TestTrouble::default());
    }

    #[test]
    fn the_runtime_note_names_the_symptom() {
        let mio = vec!["mio".to_string()];
        let hung = TestTrouble { failed: vec![], hung: vec!["oversized".into()] };
        let note = runtime_failure_note(&mio, &hung, false).unwrap();
        assert!(note.contains("Something hangs: oversized"), "{}", note);
        assert!(note.contains("mio: a test hangs and never finishes"), "{}", note);

        // A command killed at its timeout is a hang even when libtest never said so.
        let note = runtime_failure_note(&mio, &TestTrouble::default(), true).unwrap();
        assert!(note.contains("Something hangs, in a project"), "{}", note);

        let failed = TestTrouble { failed: vec!["burst".into()], hung: vec![] };
        let note = runtime_failure_note(&mio, &failed, false).unwrap();
        assert!(note.contains("Tests fail: burst"), "{}", note);
        assert!(note.contains("mio: test burst fails"), "{}", note);

        assert!(runtime_failure_note(&mio, &TestTrouble::default(), false).is_none());
        assert!(runtime_failure_note(&[], &hung, true).is_none());
    }

    #[test]
    fn crates_named_in_the_errors_come_before_mere_dependencies() {
        let library = vec!["actix-web".to_string(), "mio".to_string(), "tokio".to_string()];
        let output = "error[E0433]: failed to resolve: could not find `web` in `actix_web`\n\
  --> C:\\Users\\me\\.cargo\\registry\\src\\index\\mio-1.0.2\\src\\poll.rs:3:1";
        let involved = involved_crates(&library, output, &["tokio".to_string(), "serde".to_string()]);
        assert_eq!(involved, ["actix-web", "mio", "tokio"]);
        assert!(involved_crates(&library, "error: x", &[]).is_empty());

        // A path that merely passes through a crate's name is not that crate.
        let io = vec!["io".to_string()];
        assert!(involved_crates(&io, "expected `std::io::Result<()>`", &[]).is_empty());
        assert_eq!(involved_crates(&io, "use io::Read;", &[]), ["io"]);
    }

    #[test]
    fn the_note_hands_the_model_a_question() {
        let note = build_failure_note(
            &["mio".to_string(), "tokio".to_string()],
            &["error[E0599]: no method named \"poll\"".to_string()],
        )
        .unwrap();
        assert!(note.contains("mio and tokio"), "{}", note);
        assert!(note.contains("library(question: \"mio — how do I fix: error[E0599]"), "{}", note);
        assert!(build_failure_note(&[], &["e".into()]).is_none());
        assert!(build_failure_note(&["mio".into()], &[]).is_none());
    }
}
