//! The REPL's slash commands as data: the list that opens above the input box once a `/`
//! is typed, the command half of `/help`, and the names `repl::command_for` dispatches on.
//!
//! One table for all three, because they were three: the list here, a chain of
//! `if input == "/..."` that ran the commands, and a hand-written `/help` that had already
//! drifted from both. A test in `repl.rs` checks every name here runs something, since a
//! suggestion that runs nothing is worse than none.

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

pub(crate) struct BuiltIn {
    pub(crate) name: &'static str,
    pub(crate) aliases: &'static [&'static str],
    pub(crate) args: Args,
    // What follows the name in `/help` — `<path>`, `[name]`.
    pub(crate) usage: &'static str,
    // One line — the list cuts it at the edge of the screen.
    pub(crate) about: &'static str,
    // What `/help` adds under that line, already wrapped. Empty for most.
    pub(crate) more: &'static [&'static str],
}

const fn built_in(
    name: &'static str,
    aliases: &'static [&'static str],
    args: Args,
    usage: &'static str,
    about: &'static str,
    more: &'static [&'static str],
) -> BuiltIn {
    BuiltIn { name, aliases, args, usage, about, more }
}

pub(crate) const BUILT_IN: &[BuiltIn] = &[
    built_in("help", &[], Args::None, "", "Show the commands and the keys", &[]),
    built_in("plan", &[], Args::None, "", "Research and propose only; nothing changes until approved", &[]),
    built_in("auto", &[], Args::None, "", "Review each gated call; routine ones run without asking", &[]),
    built_in("normal", &[], Args::None, "", "Back to answering every prompt yourself", &[]),
    built_in("mode", &[], Args::None, "", "Show which mode is in force", &[]),
    built_in("add", &[], Args::Required, "<path>", "Let the file tools read and write another directory too", &[]),
    built_in("workspace", &[], Args::None, "", "Show which paths the file tools may touch", &[]),
    built_in("cd", &[], Args::Optional, "[path]", "Move to another project; with no path, show where you are", &[]),
    built_in("init", &[], Args::None, "", "Study the project and write its instruction file", &[]),
    built_in("tools", &[], Args::None, "", "List available tools and whether they prompt", &[]),
    built_in("mcp", &[], Args::Optional, "[reconnect <name>]", "List the MCP servers and their tools, or restart one", &[
        "Servers come from mcp.json in the data directory and .haico/mcp.json",
        "in the project. 'reconnect' also asks again about a project server",
        "that was not approved.",
    ]),
    built_in("model", &[], Args::Optional, "[name]", "Switch model and save it as the default", &[
        "With no name, pick from the server's list.",
    ]),
    built_in("provider", &[], Args::Optional, "[name]", "Switch to another server from the config's providers", &[
        "Saved as the default; with no name, pick from the list.",
    ]),
    built_in("config", &[], Args::None, "", "Show the settings in force and where each came from", &[]),
    built_in("set", &[], Args::Optional, "[name] [val]", "Show or set context, temperature and the other knobs", &[
        "The knobs: context, temperature, top_p, top_k, min_p, repeat_penalty,",
        "max_tokens, reasoning_effort and reasoning_threshold ('/set <name> unset'",
        "to undo). All last for the session except 'context', which is saved for",
        "this model.",
    ]),
    built_in("temp", &[], Args::Optional, "[val]", "Show or set the temperature", &[]),
    built_in("effort", &[], Args::Optional, "[level]", "Show or set the reasoning effort; on|off switch thinking", &[]),
    built_in("bg", &[], Args::Optional, "[stop [id]]", "List background jobs, or stop one or all of them", &[]),
    built_in("timers", &[], Args::Optional, "[cancel [id]]", "List the timers the model set, or cancel them", &[]),
    built_in("undo", &[], Args::None, "", "Revert the last file change made by a tool", &[]),
    built_in("todo", &[], Args::None, "", "Show the current plan, if the model made one", &[]),
    built_in("context", &[], Args::None, "", "Show how much context the conversation is using", &[]),
    built_in("reasoning", &[], Args::Optional, "[what]", "Keep, compress or drop the model's chain of thought", &[]),
    built_in("compact", &[], Args::None, "", "Replace the history with a summary, freeing context", &[]),
    built_in("resume", &[], Args::None, "", "Load this directory's saved conversation", &[
        "The conversation in progress is dropped.",
    ]),
    built_in("clear", &["new", "reset"], Args::None, "", "Start over, keeping the system prompt", &[]),
    built_in("skills", &[], Args::None, "", "List the skills the model can load", &[]),
    built_in("update-skills", &[], Args::None, "", "Re-read the skills directories into the system prompt", &[]),
    built_in("exit", &["quit"], Args::None, "", "Leave (Ctrl+D also works)", &[]),
];

/// The built-in a typed name means, under its own name or an alias.
pub(crate) fn built_in_named(name: &str) -> Option<&'static BuiltIn> {
    BUILT_IN.iter().find(|command| command.name == name || command.aliases.contains(&name))
}

// The built-ins, then the skills under `roots`. A skill named like a built-in is left out:
// the built-in is matched first, so typing that name never reaches the skill.
pub(crate) fn available(roots: &[PathBuf]) -> Vec<Command> {
    let mut commands: Vec<Command> = BUILT_IN
        .iter()
        .map(|command| Command {
            name: command.name.to_string(),
            aliases: command.aliases,
            args: command.args,
            about: command.about.to_string(),
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
