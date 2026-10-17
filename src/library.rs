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

use crate::util::flatten;
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
    /// The crates the article is about, as written in `Cargo.toml` (`actix-web`), or none
    /// for an article that is about Rust itself. Usually one; several for an article about a
    /// trait crate and the crates implementing it (`aead, aes-gcm, chacha20poly1305`), since a
    /// project depends on `aes-gcm` and never names `aead`, and one article per implementation
    /// would repeat the whole trait in each.
    pub(crate) crates: Vec<String>,
    /// What an article that is not about a crate is about: "Rust ownership and borrowing".
    /// Set by the frontmatter's `topic`, and what the index and the notes call it.
    pub(crate) topic: String,
    /// Compiler error codes the article explains (`E0502`), from the frontmatter's `codes`.
    /// They are how a build is pointed at an article when no crate is involved at all: a
    /// borrow-checker error names no crate and depends on none, so nothing else could.
    pub(crate) codes: Vec<String>,
    /// Phrases from error headlines the article explains, from the frontmatter's `matches`
    /// (separated by `;`), compared case-insensitively. For the errors a code can't pick out:
    /// "cannot be sent between threads safely" is E0277, which is also every other unsatisfied
    /// trait bound, and "lifetime may not live long enough" has no code at all.
    pub(crate) matches: Vec<String>,
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
            for name in &article.crates {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
        }
        out.sort();
        out
    }

    /// The names of the articles' subjects (each crate, else the topic) that explain any of
    /// these error headlines — by one of the article's codes, or one of its phrases —
    /// deduplicated, in library order.
    ///
    /// One name per crate, never `subject()`'s joined list: the note merges these with the
    /// crates the build depends on, and a joined "ed25519-dalek, x25519-dalek, ed25519" is not
    /// equal to any of them, so the note named each of those crates twice.
    pub(crate) fn subjects_for_errors(&self, errors: &[String]) -> Vec<String> {
        let codes = error_codes(&errors.join("\n"));
        let lowered: Vec<String> = errors.iter().map(|e| e.to_lowercase()).collect();
        let mut out: Vec<String> = Vec::new();
        for article in &self.articles {
            let by_code = article.codes.iter().any(|c| codes.contains(c));
            let by_phrase = article.matches.iter().any(|m| lowered.iter().any(|e| e.contains(m.as_str())));
            if by_code || by_phrase {
                let names = if article.crates.is_empty() { vec![article.topic.clone()] } else { article.crates.clone() };
                for name in names {
                    if !name.is_empty() && !out.contains(&name) {
                        out.push(name);
                    }
                }
            }
        }
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
            for name in &article.crates {
                header.extend(tokens(name));
            }
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
            if !article.codes.is_empty() {
                out.push_str(&format!(" [errors {}]", article.codes.join(", ")));
            }
            if !article.summary.is_empty() {
                out.push_str(&format!(": {}", article.summary));
            }
            out.push('\n');
        }
        out
    }
}

impl Article {
    /// What the article is about, as the index and the notes name it: its crates, else its topic.
    pub(crate) fn subject(&self) -> String {
        if !self.crates.is_empty() { self.crates.join(", ") } else { self.topic.clone() }
    }
}

/// The index as the system prompt carries it: one line per crate or topic, naming its articles
/// and the compiler errors they explain.
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
        let key = match article.subject() {
            subject if subject.is_empty() => "general".to_string(),
            subject => subject,
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
        out.push_str(&format!("- {}: {}", name, titles.join("; ")));
        let mut codes: Vec<&str> = articles.iter().flat_map(|a| a.codes.iter().map(String::as_str)).collect();
        codes.sort_unstable();
        codes.dedup();
        if !codes.is_empty() {
            out.push_str(&format!(" (compiler errors {})", codes.join(", ")));
        }
        out.push('\n');
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
    // top of the library is about no crate in particular, and neither is one with a `topic` —
    // `library/rust/ownership.md` is not about a crate called "rust", and treating it as one
    // would have the notes looking for `rust::` in compiler output.
    let topic = front_value(front, "topic").unwrap_or_default();
    let crates: Vec<String> = match front_value(front, "crate") {
        Some(list) => list.split(',').map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect(),
        None if !topic.is_empty() => Vec::new(),
        None => id.rsplit_once('/').map(|(dir, _)| vec![dir.split('/').next().unwrap_or(dir).to_string()]).unwrap_or_default(),
    };
    let codes = front_value(front, "codes").map(|list| error_codes(&list)).unwrap_or_default();
    let matches = front_value(front, "matches")
        .map(|list| {
            list.split(';').map(|m| m.trim().to_lowercase()).filter(|m| !m.is_empty()).collect()
        })
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
    // Whole: a summary cut short is one the librarian cannot judge the article by.
    let summary = flatten(&summary);

    Article {
        id,
        title,
        crates,
        topic,
        codes,
        matches,
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
// query words, whole — a line cut at 160 characters, as it was, could stop before the words
// that made it the match. Falls back to the section's first line.
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
    flatten(line)
}


/// The distinct error codes (`E0502`) in some text, in order of appearance, uppercased.
pub(crate) fn error_codes(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split(|c: char| !c.is_ascii_alphanumeric()) {
        let upper = word.to_ascii_uppercase();
        let is_code = upper.len() == 5 && upper.starts_with('E') && upper[1..].chars().all(|c| c.is_ascii_digit());
        if is_code && !out.contains(&upper) {
            out.push(upper);
        }
    }
    out
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
    fn a_long_summary_is_kept_whole() {
        let summary = format!("{}and the part that says when it applies.", "Covers a great deal of ground ".repeat(20));
        let article = parse_article("tokio/long".into(), &format!("---\nsummary: {}\n---\n# Long\n\nx\n", summary));
        assert_eq!(article.summary, summary);
    }

    #[test]
    fn an_article_is_split_at_its_headings_and_not_inside_code() {
        let article = parse_article("tokio/select".into(), SELECT);
        assert_eq!(article.title, "tokio::select!");
        assert_eq!(article.crates, ["tokio"]);
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
        assert_eq!(article.crates, ["actix-web"]);
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
    fn a_topic_article_has_no_crate_and_declares_its_error_codes() {
        let article = parse_article(
            "rust/ownership".into(),
            "---\ntopic: Rust ownership and borrowing\ncodes: E0499, e0502 E0505,E0597, not-a-code\n---\n# Borrowing\n\nx\n",
        );
        assert!(article.crates.is_empty());
        assert_eq!(article.subject(), "Rust ownership and borrowing");
        assert_eq!(article.codes, ["E0499", "E0502", "E0505", "E0597"]);

        // A crate named outright still wins over the topic.
        let both = parse_article("x/y".into(), "---\ncrate: mio\ntopic: T\n---\nx\n");
        assert_eq!(both.subject(), "mio");
    }

    #[test]
    fn topics_are_found_by_error_code_and_indexed_with_their_codes() {
        let dir = TempDir::new("library-codes");
        write(dir.path(), "rust/ownership", "---\ntopic: Rust ownership and borrowing\ncodes: E0502, E0499\n---\n# Borrowing\n\nx\n");
        write(dir.path(), "mio/poll", "# Polling\n\nx\n");
        let library = Library::load(dir.path());

        let errors = |e: &str| vec![e.to_string()];
        assert_eq!(library.subjects_for_errors(&errors("error[E0502]: cannot borrow")), ["Rust ownership and borrowing"]);
        assert!(library.subjects_for_errors(&errors("error[E0308]: mismatched types")).is_empty());
        // A topic is not a crate: nothing goes looking for it in Cargo.toml or in paths.
        assert_eq!(library.crates(), ["mio"]);

        let index = index_block(dir.path()).unwrap();
        assert!(index.contains("- Rust ownership and borrowing: Borrowing (compiler errors E0499, E0502)"), "{}", index);
        assert!(index.contains("- mio: Polling\n"), "{}", index);
        assert!(library.catalog().contains("[errors E0502, E0499]"), "{}", library.catalog());

        assert_eq!(
            error_codes("error[E0502]: cannot borrow\nerror[E0502]: again\nerror[e0499]: x E123 E12345"),
            ["E0502", "E0499"]
        );
    }

    #[test]
    fn a_phrase_picks_out_an_error_its_code_cannot() {
        let dir = TempDir::new("library-matches");
        write(
            dir.path(),
            "rust/threads",
            "---\ntopic: Rust threads\nmatches: cannot be sent between threads safely; Cannot be shared between threads safely\n---\n# Threads\n\nx\n",
        );
        let library = Library::load(dir.path());
        let send = vec!["error[E0277]: `Rc<i32>` cannot be sent between threads safely".to_string()];
        let shared = vec!["error[E0277]: `RefCell<u8>` CANNOT be shared between threads safely".to_string()];
        let other = vec!["error[E0277]: the trait bound `Foo: Display` is not satisfied".to_string()];
        assert_eq!(library.subjects_for_errors(&send), ["Rust threads"]);
        assert_eq!(library.subjects_for_errors(&shared), ["Rust threads"]);
        // The same code, a different error: not this article's.
        assert!(library.subjects_for_errors(&other).is_empty());
    }

    #[test]
    fn a_note_names_each_crate_of_a_matched_article_once() {
        // Observed on a p2p_auth run: an article on several crates, matched by a phrase, was
        // named by its joined crate list, which the dependencies did not dedupe against, so
        // the note read "ed25519-dalek, x25519-dalek, ed25519, ..., x25519-dalek, ed25519-dalek".
        let dir = TempDir::new("library-multi-crate-note");
        write(
            dir.path(),
            "crypto/ed25519",
            "---\ncrate: ed25519-dalek, x25519-dalek, ed25519\nmatches: signature::Signer<\n---\n# Signing\n\nx\n",
        );
        let library = Library::load(dir.path());
        let errors = vec!["error[E0599]: the trait bound `SigningKey: signature::Signer<Sig>` is not satisfied".to_string()];
        assert_eq!(library.subjects_for_errors(&errors), ["ed25519-dalek", "x25519-dalek", "ed25519"]);

        let crates = vec!["tokio".to_string(), "x25519-dalek".to_string(), "ed25519-dalek".to_string()];
        let subjects = crate::tools::build_notes::subjects_for_build(&library, &errors, &crates);
        let mut unique = subjects.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(subjects.len(), unique.len(), "{subjects:?}");
    }

    #[test]
    fn a_top_level_article_has_no_crate() {
        let article = parse_article("lifetimes".into(), "Borrowing rules.\n");
        assert!(article.crates.is_empty());
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
    fn an_article_can_cover_a_trait_crate_and_its_implementations() {
        let dir = TempDir::new("library-crates");
        write(dir.path(), "aead/basics", "---\ncrate: aead, aes-gcm , chacha20poly1305\n---\n# AEAD\n\n## Nonces\n\nNever reuse one.\n");
        let library = Library::load(dir.path());
        assert_eq!(library.articles[0].crates, ["aead", "aes-gcm", "chacha20poly1305"]);
        assert_eq!(library.crates(), ["aead", "aes-gcm", "chacha20poly1305"]);
        assert_eq!(index_block(dir.path()).unwrap(), "- aead, aes-gcm, chacha20poly1305: AEAD\n");
        // A project that depends on one implementation is pointed at the article.
        let deps = ["aes-gcm".to_string()];
        assert_eq!(crate::tools::build_notes::involved_crates(&library.crates(), "error[E0599]: x", &deps), ["aes-gcm"]);
        // And every crate's name finds its sections, though the text names none of them.
        assert_eq!(library.search("chacha20poly1305 nonce")[0].id, "aead/basics#0");
    }

}
