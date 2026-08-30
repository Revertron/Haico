//! What a tool call needs before it runs, and how the user is asked.

use crate::input::{echo, read_text, Editor, Input, Keys};
use crate::util::truncate_at_boundary;
use crate::workspace::Workspace;
use anyhow::{anyhow, Result};
use crossterm::event::{Event, KeyCode, KeyEventKind};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;

pub(crate) const AUTO_ALLOWED_TOOLS: &[&str] = &[
    "calculator",
    "get_date",
    "get_time",
    "read",
    "ls",
    "glob_search",
    "grep",
    "crate_info",
    "crate_search",
    "crate_api",
    "todo_write",
    "background_output",
    "background_stop",
    // Not read-only so much as inert: it changes nothing on this machine, and it is the
    // user answering it. Prompting for permission to ask a question would be a question
    // about a question. Listed here rather than special-cased because plan mode reads the
    // same list, and planning is where the model is least sure of anything.
    "ask_user",
];

// Programs where the bare name says almost nothing about what is about to happen, so the
// subcommand belongs in the permission key: approving "cargo build" must not also approve
// "cargo publish".

pub(crate) const SUBCOMMAND_PROGRAMS: &[&str] = &[
    "git", "cargo", "npm", "npx", "pnpm", "yarn", "gh", "docker", "kubectl", "make", "go",
    "pip", "pip3", "python", "python3", "dotnet", "systemctl", "brew", "apt", "winget",
];

// What has to happen before a tool call runs.

pub(crate) enum Gate {
    // Read-only: run it without interrupting the user.
    Auto,
    // Needs consent. "Allow all" remembers every key, and a later call is silent only when
    // all of its keys are already remembered.
    Ask { keys: Vec<String>, label: String },
    // Refused outright, with the reason handed back to the model.
    Deny(String),
}

// Permission tracking system

#[derive(Debug, Clone)]
pub(crate) struct PermissionManager {
    pub(crate) allowed: HashSet<String>,
}


impl PermissionManager {
    pub(crate) fn new() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    // Every key must be approved. An empty key list is never allowed — that would turn a
    // malformed call into a silently permitted one.
    pub(crate) fn is_allowed(&self, keys: &[String]) -> bool {
        !keys.is_empty() && keys.iter().all(|key| self.allowed.contains(key))
    }

    pub(crate) fn allow(&mut self, keys: &[String]) {
        for key in keys {
            self.allowed.insert(key.clone());
        }
    }
}


pub(crate) enum PermissionResponse {
    Allow,
    AllowAll,
    Reject(String),
}

// Split a shell command into the pieces that will actually run.
//
// Without this a chain rides in on one approved prefix: having allowed "git status", the
// model could run "git status && rm -rf build" with no further prompt. Quoting is not
// parsed, so a separator inside a quoted string splits too — that only ever produces an
// extra unrecognized segment, which prompts rather than silently allowing.

pub(crate) fn command_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();

    // Quoting has to be tracked, or a separator inside a quoted argument splits the command
    // into nonsense. A PowerShell one-liner full of ';' inside its -Command string produced
    // a dozen fake "commands" in the approval prompt, one of which was the interpreter
    // itself — so the prompt buried a blanket grant in noise.
    let mut in_single = false;
    let mut in_double = false;

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(c);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(c);
            }
            '&' | '|' if !in_single && !in_double && chars.peek() == Some(&c) => {
                chars.next();
                segments.push(std::mem::take(&mut current));
            }
            ';' | '\n' | '|' if !in_single && !in_double => {
                segments.push(std::mem::take(&mut current))
            }
            _ => current.push(c),
        }
    }
    segments.push(current);

    segments
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// Interpreters, and the flags that hand them a program to run inline.
//
// `powershell -Command "..."` or `sh -c "..."` can do anything at all, so the program name
// says nothing about what will happen. Approving one of these by prefix would be approving
// every future script the same interpreter is handed.
const INTERPRETERS: &[&str] = &[
    "sh", "bash", "zsh", "dash", "ksh", "fish", "powershell", "powershell.exe", "pwsh",
    "pwsh.exe", "cmd", "cmd.exe", "python", "python3", "node", "perl", "ruby", "deno", "php",
];

// Whether this segment hands work to an interpreter.
//
// Originally this only caught inline code (`-c`, `-Command`), on the theory that running a
// script *file* was more like running a program. That was too fine a distinction:
// `powershell -ExecutionPolicy Bypass -File .\test3.ps1` offered "allow all 'powershell'",
// and since the agent writes the .ps1 itself, approving that is approving anything it later
// chooses to write. The program name never says what will run, whichever way the script
// arrives, so every interpreter invocation is scoped to its exact command.
//
// Running the same command again is still remembered, so a repeated `python test.py` costs
// one approval, not one per run. Only a different script asks again.
pub(crate) fn runs_arbitrary_code(segment: &str) -> bool {
    let Some(program) = segment
        .split_whitespace()
        .next()
        .map(|p| p.trim_matches(['"', '\''].as_slice()))
    else {
        return false;
    };
    let program = program
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(program)
        .to_lowercase();

    INTERPRETERS.contains(&program.as_str())
}

// The identifying prefix of one command: the program, plus its subcommand where the program
// alone is just a multiplexer.

pub(crate) fn command_prefix(segment: &str) -> String {
    let mut parts = segment
        .split_whitespace()
        .map(|p| p.trim_matches(['"', '\''].as_slice()))
        .filter(|p| !p.is_empty());

    let Some(program) = parts.next() else {
        return String::new();
    };

    // Strip any directory part so "/usr/bin/git" and "git" share one key.
    let program = program
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(program)
        .to_lowercase();

    if SUBCOMMAND_PROGRAMS.contains(&program.as_str()) {
        if let Some(subcommand) = parts.next().filter(|s| !s.starts_with('-')) {
            return format!("{} {}", program, subcommand);
        }
    }

    program
}

// Commands that are catastrophic and essentially never what was meant.
//
// This is a guard against a model's accident, not a security boundary: anything determined
// can express the same effect in a form this does not recognize. It is deliberately narrow,
// so it does not train the user to ignore it.

pub(crate) fn destructive_command(segment: &str) -> Option<&'static str> {
    let normalized = segment.split_whitespace().collect::<Vec<_>>().join(" ");
    let lowered = normalized.to_lowercase();

    // A recursive delete of the filesystem root or the home directory, in any flag order.
    for prefix in ["rm -rf", "rm -fr", "rm -r -f", "rm -f -r", "rm --recursive --force"] {
        if let Some(rest) = lowered.strip_prefix(prefix) {
            let target = rest.trim();
            if matches!(target, "/" | "/*" | "~" | "~/" | "~/*" | "$home" | "$home/*") || target.starts_with("/ ")
            {
                return Some("recursive delete of the filesystem root or home directory");
            }
        }
    }

    if lowered.starts_with("mkfs") || lowered.contains(" mkfs.") {
        return Some("filesystem format");
    }
    if lowered.contains("dd ") && lowered.contains("of=/dev/") {
        return Some("raw write to a block device");
    }
    if lowered.replace(' ', "").contains(":(){:|:&};:") {
        return Some("fork bomb");
    }
    if lowered.starts_with("format c:") || lowered.starts_with("format /") {
        return Some("drive format");
    }
    if lowered.starts_with("del /f /s /q c:\\") || lowered.starts_with("rd /s /q c:\\") {
        return Some("recursive delete of the system drive");
    }

    None
}

// Resolve a path to an absolute one without requiring it to exist yet.

pub(crate) fn gate_for(workspace: &Workspace, tool_name: &str, arguments: &Value) -> Gate {
    // Before anything else, including the auto-allow list below. The read-only tools are on
    // that list, so checking the boundary afterwards would let every read skip it — and
    // reading is how a secret leaves a machine.
    if let Some(refusal) = workspace.refusal(tool_name, arguments) {
        return Gate::Deny(refusal);
    }

    if AUTO_ALLOWED_TOOLS.contains(&tool_name) {
        return Gate::Auto;
    }

    match tool_name {
        "run_command" | "run_background" => {
            let command = arguments["command"].as_str().unwrap_or_default();
            let segments = command_segments(command);

            if segments.is_empty() {
                return Gate::Deny("refusing to run an empty command".to_string());
            }

            for segment in &segments {
                if let Some(reason) = destructive_command(segment) {
                    return Gate::Deny(format!(
                        "Refusing to run '{}': {}. If this really is what you want, run it \
                         yourself outside the agent.",
                        truncate_at_boundary(segment, 200),
                        reason
                    ));
                }
            }

            // An interpreter cannot be approved by program name: "powershell" says nothing
            // about what it will be handed, and the agent writes those scripts itself.
            // Approval is for this exact command.
            if segments.iter().any(|segment| runs_arbitrary_code(segment)) {
                return Gate::Ask {
                    keys: vec![format!("run_command:exact:{}", command.trim())],
                    label: "Allow this exact command again (it runs a script, so approving \
                            the interpreter by name would cover anything it writes next)"
                        .to_string(),
                };
            }

            let mut prefixes: Vec<String> = segments
                .iter()
                .map(|segment| command_prefix(segment))
                .filter(|prefix| !prefix.is_empty())
                .collect();
            prefixes.sort();
            prefixes.dedup();

            if prefixes.is_empty() {
                return Gate::Deny("refusing to run an unrecognizable command".to_string());
            }

            // A long chain would otherwise produce a label nobody reads, which is how a
            // broad grant slips past unnoticed.
            const MAX_SHOWN: usize = 4;
            let shown = prefixes
                .iter()
                .take(MAX_SHOWN)
                .map(|p| format!("'{}'", p))
                .collect::<Vec<_>>()
                .join(", ");
            let label = if prefixes.len() > MAX_SHOWN {
                format!("Allow all {} and {} more command(s)", shown, prefixes.len() - MAX_SHOWN)
            } else {
                format!("Allow all {} commands", shown)
            };

            Gate::Ask {
                keys: prefixes
                    .iter()
                    .map(|prefix| format!("run_command:{}", prefix))
                    .collect(),
                label,
            }
        }

        "write_file" | "edit" | "multi_edit" => {
            // Anything outside the workspace was refused above, so this lands in one of the
            // writable roots. Keyed to that root rather than to the workspace as a whole:
            // approving writes in the backend must not silently cover a client directory
            // added an hour later.
            let path = PathBuf::from(arguments["file_path"].as_str().unwrap_or_default());
            let root = workspace.writable_root_for(&path);
            // The key keeps the full canonical form, so two roots can never collide; the
            // label drops the \\?\ prefix, which is noise to everyone reading the prompt.
            let key = root.map(|r| r.display().to_string()).unwrap_or_default();
            let shown = root.map(crate::util::display_path).unwrap_or_default();

            Gate::Ask {
                keys: vec![format!("{}:root:{}", tool_name, key)],
                label: format!("Allow all '{}' calls under {}", tool_name, shown),
            }
        }

        // Everything else that reaches the network with user-supplied content.
        _ => Gate::Ask {
            keys: vec![tool_name.to_string()],
            label: format!("Allow all '{}' tool usage", tool_name),
        },
    }
}

// Generate a colored diff with line numbers, showing only context around changes

pub(crate) async fn ask_permission(
    keys: &mut Keys,
    allow_all_label: &str,
    description: &str,
    diff: Option<&str>,
) -> Result<PermissionResponse> {
    outln!("🔒 Allow this tool usage?");
    outln!("{}\n", description);

    if let Some(diff_content) = diff {
        outln!("{}", diff_content);
    }

    outln!("1. Allow once");
    outln!("2. {}", allow_all_label);
    outln!("3. Reject and provide alternative");
    crate::out::flush();

    let response = loop {
        let Some(event) = keys.next().await else {
            // The keyboard reader is gone, which is not consent.
            return Err(anyhow!("no interactive terminal to ask for permission"));
        };

        // Only a closed channel means nobody is there to answer. Everything else that can
        // arrive here is not an answer but is not an absent user either, and matching the
        // key event directly above used to lump the two together: on Windows the console
        // emits a focus event whenever the window is clicked away from or minimized, so
        // stepping away from the keyboard refused the pending call - and every call after
        // it - as if the terminal had gone.
        let Event::Key(key_event) = event else {
            continue;
        };

        // Only process key press events, ignore key release
        if key_event.kind != KeyEventKind::Press {
            continue;
        }

        match key_event.code {
            KeyCode::Char('1') => {
                break PermissionResponse::Allow;
            }
            KeyCode::Char('2') => {
                break PermissionResponse::AllowAll;
            }
            KeyCode::Char('3') => {
                outln!("3\n");

                let mut editor = Editor::new(Vec::new());
                let alternative = match read_text(keys, &mut editor, "> ").await {
                    Input::Text(text) => text,
                    // Backing out of the question is not an answer to it, so it is the
                    // same as rejecting with nothing to say.
                    _ => String::new(),
                };
                outln!("{}", echo("> ", &alternative));

                break PermissionResponse::Reject(alternative.trim().to_string());
            }
            _ => {
                // Invalid key, continue waiting
                continue;
            }
        }
    };

    // Anything typed while the question was on screen was not meant for whatever comes
    // after it.
    keys.drain();

    Ok(response)
}

// Normalize an LLM server URL down to a bare origin, so the "/v1/..." paths below always
// concatenate cleanly. LM Studio's UI hands out its base URL with "/v1" already appended
// (and OpenAI SDKs expect it that way), which would otherwise build
// "/v1/v1/chat/completions" — a path LM Studio rejects while still answering HTTP 200.

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use crate::config::*;
    use crate::llm::*;
    use crate::permissions::*;
    use crate::render::*;
    use crate::session::*;
    use crate::testutil::*;
    use crate::tools::*;
    use crate::util::*;
    use serde_json::json;

    // The default posture: the working directory and nothing else.
    fn ws() -> crate::workspace::Workspace {
        crate::workspace::Workspace::new(&std::env::current_dir().unwrap(), &[])
    }

    #[test]
    fn splits_chained_shell_commands() {
        use crate::command_segments;

        assert_eq!(command_segments("cargo build"), vec!["cargo build"]);
        // The whole point: a chain must not be treated as one approved command.
        assert_eq!(
            command_segments("git status && rm -rf build"),
            vec!["git status", "rm -rf build"]
        );
        assert_eq!(
            command_segments("a ; b || c | d"),
            vec!["a", "b", "c", "d"]
        );
        assert_eq!(command_segments("   "), Vec::<String>::new());
    }


    #[test]
    fn separators_inside_quotes_do_not_split_a_command() {
        use crate::permissions::command_segments;

        // The reported case: a PowerShell one-liner whose -Command string is full of ';'
        // and '|'. Splitting on those produced a dozen fake commands, and the approval
        // prompt then offered to allow "'$writer.flush()'", "'powershell'" and friends —
        // burying a blanket grant for the interpreter in noise.
        let command = "powershell -Command \"$s = New-Object System.Net.Sockets.TcpClient; \
                       $writer.WriteLine('GET / HTTP/1.1'); $writer.Flush(); $s.Close()\"";
        assert_eq!(command_segments(command).len(), 1);

        // Single quotes protect too, and both kinds nest inside the other.
        assert_eq!(command_segments("sh -c 'a; b | c'").len(), 1);
        assert_eq!(command_segments("echo \"it's fine; really\"").len(), 1);

        // Real separators outside quotes still split, or the chain check would be useless.
        assert_eq!(
            command_segments("git status && rm -rf build"),
            vec!["git status", "rm -rf build"]
        );
        assert_eq!(
            command_segments("echo \"a; b\" && rm -rf /tmp/x"),
            vec!["echo \"a; b\"", "rm -rf /tmp/x"]
        );
    }

    #[test]
    fn an_inline_script_is_approved_only_as_itself() {
        use crate::permissions::{gate_for, runs_arbitrary_code, Gate, PermissionManager};

        assert!(runs_arbitrary_code("powershell -Command \"$x = 1\""));
        assert!(runs_arbitrary_code("sh -c 'rm -rf build'"));
        assert!(runs_arbitrary_code("python -c \"import os\""));
        assert!(runs_arbitrary_code("C:\\Windows\\System32\\cmd.exe /c dir"));
        // A script file is no narrower than inline code when the agent wrote the file.
        assert!(runs_arbitrary_code("powershell -ExecutionPolicy Bypass -File .\\test3.ps1"));
        assert!(runs_arbitrary_code("python script.py"));
        assert!(runs_arbitrary_code("node server.js"));
        // Ordinary programs stay prefix-scoped, or every build would ask again.
        assert!(!runs_arbitrary_code("cargo build"));
        assert!(!runs_arbitrary_code("git status"));

        let keys_for = |command: &str| match gate_for(&ws(), "run_command", &json!({"command": command})) {
            Gate::Ask { keys, label } => (keys, label),
            other => panic!(
                "expected a prompt, got {}",
                match other {
                    Gate::Auto => "auto".to_string(),
                    Gate::Deny(r) => r,
                    _ => unreachable!(),
                }
            ),
        };

        let (keys, label) = keys_for("powershell -Command \"$x = 1; $x\"");
        assert!(label.contains("exact command"), "got: {}", label);
        // The screenshot case: a script file, not inline code.
        let (file_keys, _) = keys_for("powershell -ExecutionPolicy Bypass -File .\test3.ps1");
        assert!(!file_keys.iter().any(|k| k == "run_command:powershell"), "got: {:?}", file_keys);
        // Never the bare interpreter: that would wave through every future script.
        assert!(!keys.iter().any(|k| k == "run_command:powershell"), "got: {:?}", keys);

        let mut permissions = PermissionManager::new();
        permissions.allow(&keys);
        // The same script again is remembered...
        assert!(permissions.is_allowed(&keys_for("powershell -Command \"$x = 1; $x\"").0));
        // ...but a different one still asks, which is the whole point.
        assert!(!permissions.is_allowed(&keys_for("powershell -Command \"rm -rf C:\\\"").0));
        assert!(!permissions.is_allowed(&keys_for("powershell Get-Process").0));
    }

    #[test]
    fn a_long_chain_does_not_produce_an_unreadable_label() {
        use crate::permissions::{gate_for, Gate};

        let Gate::Ask { keys, label } = gate_for(&ws(), 
            "run_command",
            &json!({"command": "a && b && c && d && e && f && g"}),
        ) else {
            panic!("expected a prompt");
        };
        // Every prefix is still approved; only the label is condensed, because a label
        // nobody reads is how a broad grant slips past.
        assert_eq!(keys.len(), 7);
        assert!(label.contains("3 more"), "got: {}", label);
        assert!(label.len() < 90, "label is {} chars: {}", label.len(), label);
    }

    #[test]
    fn derives_permission_prefixes() {
        use crate::command_prefix;

        // Multiplexers keep their subcommand: approving one must not approve the rest.
        assert_eq!(command_prefix("git status --short"), "git status");
        assert_eq!(command_prefix("cargo build --release"), "cargo build");
        assert_ne!(command_prefix("cargo publish"), command_prefix("cargo build"));
        // A flag is not a subcommand.
        assert_eq!(command_prefix("git --version"), "git");
        // Plain programs key on the program alone, path and case stripped.
        assert_eq!(command_prefix("/usr/bin/ls -la"), "ls");
        assert_eq!(command_prefix("C:\\Windows\\System32\\WHERE.exe rustc"), "where.exe");
        assert_eq!(command_prefix(""), "");
    }


    #[test]
    fn allow_all_does_not_leak_across_commands() {
        use crate::{gate_for, Gate, PermissionManager};

        let gate_keys = |command: &str| match gate_for(&ws(), "run_command", &json!({"command": command})) {
            Gate::Ask { keys, .. } => keys,
            Gate::Auto => panic!("run_command must never be auto-allowed"),
            Gate::Deny(reason) => panic!("unexpected deny: {}", reason),
        };

        let mut permissions = PermissionManager::new();
        permissions.allow(&gate_keys("git status"));

        // The approved command runs silently.
        assert!(permissions.is_allowed(&gate_keys("git status --short")));
        // A different subcommand, a different program, and a chain that merely starts with
        // the approved command all still prompt.
        assert!(!permissions.is_allowed(&gate_keys("git push")));
        assert!(!permissions.is_allowed(&gate_keys("cargo build")));
        assert!(!permissions.is_allowed(&gate_keys("git status && rm -rf build")));

        // Approving the chain approves both halves, and nothing else.
        permissions.allow(&gate_keys("git status && cargo build"));
        assert!(permissions.is_allowed(&gate_keys("cargo build --release")));
        assert!(!permissions.is_allowed(&gate_keys("cargo publish")));

        // An empty approval set is never a pass.
        assert!(!PermissionManager::new().is_allowed(&[]));
    }


    #[test]
    fn read_only_tools_are_auto_allowed_and_writes_are_not() {
        use crate::{gate_for, Gate};

        for tool in ["read", "ls", "grep", "glob_search", "crate_info", "calculator"] {
            assert!(
                matches!(gate_for(&ws(), tool, &json!({})), Gate::Auto),
                "{} should not interrupt the user",
                tool
            );
        }

        for tool in ["write_file", "edit", "run_command", "web_search", "web_fetch"] {
            assert!(
                !matches!(gate_for(&ws(), tool, &json!({"command": "ls", "file_path": "a.rs"})), Gate::Auto),
                "{} must never be auto-allowed",
                tool
            );
        }
    }


    #[test]
    fn work_outside_the_workspace_is_refused_rather_than_offered() {
        use crate::{gate_for, Gate};

        // Inside the working directory: one shared approval for the whole root.
        let keys_for = |path: &str| match gate_for(&ws(), "edit", &json!({"file_path": path})) {
            Gate::Ask { keys, .. } => keys,
            other => panic!(
                "expected a prompt, got {}",
                match other {
                    Gate::Auto => "auto".to_string(),
                    Gate::Deny(reason) => reason,
                    _ => unreachable!(),
                }
            ),
        };
        assert_eq!(keys_for("src/main.rs"), keys_for("Cargo.toml"));

        // Outside it there is no prompt to answer. This used to offer a per-file approval;
        // it is now refused outright, because a prompt is something a model can talk its
        // way through and the point of the boundary is that it cannot.
        let outside = std::env::temp_dir().join("haico-elsewhere.rs");
        let denied = gate_for(&ws(), "edit", &json!({"file_path": outside.display().to_string()}));
        assert!(matches!(denied, Gate::Deny(_)), "a write outside must be denied");

        // And the same for reads, which is the new part: they were auto-allowed with no
        // path check at all, so anything on the machine was one tool call away.
        let secret = std::env::temp_dir().join("haico-secrets.env");
        assert!(
            matches!(gate_for(&ws(), "read", &json!({"file_path": secret.display().to_string()})), Gate::Deny(_)),
            "a read outside the workspace must be denied"
        );
        assert!(
            matches!(gate_for(&ws(), "ls", &json!({"path": std::env::temp_dir().display().to_string()})), Gate::Deny(_)),
            "listing outside the workspace must be denied"
        );

        // Reads inside it are still free — the boundary must not cost a prompt per file.
        assert!(matches!(gate_for(&ws(), "read", &json!({"file_path": "Cargo.toml"})), Gate::Auto));
    }


    #[test]
    fn refuses_catastrophic_commands() {
        use crate::{gate_for, Gate};

        let denied = |command: &str| {
            matches!(
                gate_for(&ws(), "run_command", &json!({"command": command})),
                Gate::Deny(_)
            )
        };

        assert!(denied("rm -rf /"));
        assert!(denied("rm -fr /*"));
        assert!(denied("rm -rf ~"));
        assert!(denied("mkfs.ext4 /dev/sda1"));
        assert!(denied("dd if=/dev/zero of=/dev/sda"));
        // Hidden at the tail of a chain, where a prefix check alone would miss it.
        assert!(denied("cargo build && rm -rf /"));

        // Ordinary work is not caught: a deny-list that fires on real commands just trains
        // the user to ignore it.
        assert!(!denied("rm -rf build"));
        assert!(!denied("rm -rf ./target/debug"));
        assert!(!denied("cargo build --release"));
        assert!(!denied("git clean -fdx"));
    }

}
