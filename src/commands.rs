//! The REPL's slash commands as data, for the list that opens above the input box once a
//! `/` is typed.
//!
//! The commands themselves are still the chain of `if input == "/..."` in `main.rs`; this
//! is only what the list offers. A test reads `main.rs` to check that every name here is
//! one the chain answers to, since a suggestion that runs nothing is worse than none.

use crate::skills;
use std::path::PathBuf;

// Whether a command needs something typed after it. It decides what Enter on a picked
// suggestion does: a command that can run bare runs, one that cannot is put in the box
// with a space after it, waiting for the rest.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Args {
    None,
    Optional,
    Required,
}

#[derive(Clone, Debug)]
pub(crate) struct Command {
    pub(crate) name: String,
    // Other spellings that run the same thing. Matched, never shown: listing /clear, /new
    // and /reset as three rows would spend three of the five on one command.
    pub(crate) aliases: &'static [&'static str],
    pub(crate) args: Args,
    pub(crate) about: String,
}

// Kept to one line each — the list cuts them at the edge of the screen. `/help` has the
// long versions.
const BUILT_IN: &[(&str, &[&str], Args, &str)] = &[
    ("help", &[], Args::None, "Show the commands and the keys"),
    ("plan", &[], Args::None, "Research and propose only; nothing changes until approved"),
    ("auto", &[], Args::None, "Review each gated call; routine ones run without asking"),
    ("normal", &[], Args::None, "Back to answering every prompt yourself"),
    ("mode", &[], Args::None, "Show which mode is in force"),
    ("add", &[], Args::Required, "Let the file tools read and write another directory too"),
    ("workspace", &[], Args::None, "Show which paths the file tools may touch"),
    ("cd", &[], Args::Optional, "Move to another project; with no path, show where you are"),
    ("init", &[], Args::None, "Study the project and write its instruction file"),
    ("tools", &[], Args::None, "List available tools and whether they prompt"),
    ("model", &[], Args::Optional, "Switch model and save it as the default"),
    ("provider", &[], Args::Optional, "Switch to another server from the config's providers"),
    ("config", &[], Args::None, "Show the settings in force and where each came from"),
    ("set", &[], Args::Optional, "Show or set context, temperature and the other knobs"),
    ("temp", &[], Args::Optional, "Show or set the temperature"),
    ("effort", &[], Args::Optional, "Show or set the reasoning effort; on|off switch thinking"),
    ("bg", &[], Args::Optional, "List background jobs, or stop one or all of them"),
    ("timers", &[], Args::Optional, "List the timers the model set, or cancel them"),
    ("undo", &[], Args::None, "Revert the last file change made by a tool"),
    ("todo", &[], Args::None, "Show the current plan, if the model made one"),
    ("context", &[], Args::None, "Show how much context the conversation is using"),
    ("reasoning", &[], Args::Optional, "Keep, compress or drop the model's chain of thought"),
    ("compact", &[], Args::None, "Replace the history with a summary, freeing context"),
    ("resume", &[], Args::None, "Load this directory's saved conversation"),
    ("clear", &["new", "reset"], Args::None, "Start over, keeping the system prompt"),
    ("skills", &[], Args::None, "List the skills the model can load"),
    ("update-skills", &[], Args::None, "Re-read the skills directories into the system prompt"),
    ("exit", &["quit"], Args::None, "Leave"),
];

// The built-ins, then the skills under `roots`. A skill named like a built-in is left out:
// the built-in is matched first in `main.rs`, so typing that name never reaches the skill.
pub(crate) fn available(roots: &[PathBuf]) -> Vec<Command> {
    let mut commands: Vec<Command> = BUILT_IN
        .iter()
        .map(|(name, aliases, args, about)| Command {
            name: name.to_string(),
            aliases,
            args: *args,
            about: about.to_string(),
        })
        .collect();
    for skill in skills::discover(roots) {
        let taken = commands
            .iter()
            .any(|command| command.name == skill.name || command.aliases.contains(&skill.name.as_str()));
        if !taken {
            commands.push(Command {
                name: skill.name,
                aliases: &[],
                args: Args::Optional,
                about: skill.description,
            });
        }
    }
    commands
}

pub(crate) fn for_cwd() -> Vec<Command> {
    available(&skills::roots_for_cwd())
}

// One row of the list: which command, and which characters of its name matched, to be
// drawn in bold. No span when it was an alias that matched, since the alias is not shown.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Suggestion {
    pub(crate) index: usize,
    pub(crate) hit: Option<(usize, usize)>,
}

// Everything `query` (what follows the `/`) picks out: exact names first, then names that
// start with it, then names that merely contain it — so `/com` puts `/compact` above
// `/autocompact`, and a whole name typed out is always the first row, which is what Enter
// takes. Within a rank the table's order holds, built-ins ahead of skills.
pub(crate) fn suggest(commands: &[Command], query: &str) -> Vec<Suggestion> {
    let query = query.to_lowercase();
    let mut ranked: Vec<(u8, Suggestion)> = Vec::new();
    for (index, command) in commands.iter().enumerate() {
        let name = command.name.to_lowercase();
        let best = if let Some(at) = name.find(&query) {
            let rank = if name == query { 0 } else if at == 0 { 1 } else { 3 };
            let start = name[..at].chars().count();
            Some((rank, Some((start, start + query.chars().count()))))
        } else {
            command
                .aliases
                .iter()
                .filter_map(|alias| {
                    alias.find(&query).map(|at| {
                        if *alias == query {
                            0
                        } else if at == 0 {
                            2
                        } else {
                            4
                        }
                    })
                })
                .min()
                .map(|rank| (rank, None))
        };
        if let Some((rank, hit)) = best {
            ranked.push((rank, Suggestion { index, hit }));
        }
    }
    ranked.sort_by_key(|(rank, _)| *rank);
    ranked.into_iter().map(|(_, suggestion)| suggestion).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn names(commands: &[Command], query: &str) -> Vec<String> {
        suggest(commands, query)
            .into_iter()
            .map(|suggestion| commands[suggestion.index].name.clone())
            .collect()
    }

    #[test]
    fn every_built_in_is_a_command_main_answers_to() {
        let main = include_str!("main.rs");
        for (name, aliases, _, _) in BUILT_IN {
            for spelling in std::iter::once(name).chain(aliases.iter()) {
                assert!(
                    main.contains(&format!("\"/{}\"", spelling)),
                    "/{} is suggested but main.rs never matches it",
                    spelling
                );
            }
        }
    }

    #[test]
    fn a_prefix_ranks_above_a_match_in_the_middle() {
        let commands = available(&[]);
        let found = names(&commands, "com");
        assert_eq!(found.first().map(String::as_str), Some("compact"));

        let found = names(&commands, "set");
        assert_eq!(found.first().map(String::as_str), Some("set"));
        assert!(!found.contains(&"reset".to_string()), "an alias is not a row of its own");
    }

    #[test]
    fn the_matched_characters_are_marked_and_an_alias_match_marks_none() {
        let commands = available(&[]);
        let hits = suggest(&commands, "xyz");
        assert!(hits.is_empty());

        let compact = suggest(&commands, "pac")
            .into_iter()
            .find(|suggestion| commands[suggestion.index].name == "compact")
            .unwrap();
        assert_eq!(compact.hit, Some((3, 6)));

        let clear = suggest(&commands, "res")
            .into_iter()
            .find(|suggestion| commands[suggestion.index].name == "clear")
            .expect("/reset is an alias of /clear");
        assert_eq!(clear.hit, None);
        // `resume` starts with it and is a name, so it comes before the alias.
        assert_eq!(names(&commands, "res")[0], "resume");
    }

    #[test]
    fn an_empty_query_lists_everything() {
        let commands = available(&[]);
        assert_eq!(suggest(&commands, "").len(), commands.len());
    }

    #[test]
    fn skills_join_the_list_unless_a_built_in_has_their_name() {
        let root = TempDir::new("commands-skills");
        for (name, description) in [("review", "Review the diff"), ("compact", "Shadowed")] {
            let dir = root.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {}\ndescription: {}\n---\nbody\n", name, description),
            )
            .unwrap();
        }
        let commands = available(&[root.path().to_path_buf()]);
        let review = commands.iter().find(|command| command.name == "review").unwrap();
        assert_eq!(review.about, "Review the diff");
        assert_eq!(review.args, Args::Optional);
        assert_eq!(commands.iter().filter(|command| command.name == "compact").count(), 1);
    }
}
