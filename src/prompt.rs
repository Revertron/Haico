//! The system prompt and the project's own instruction file.

use crate::util::{run_git_command, truncate_at_boundary};
use std::env;
use std::fs;
use std::path::PathBuf;

pub(crate) const SYSTEM_PROMPT: &str = include_str!("system_prompt.txt");


pub(crate) fn get_main_branch() -> String {
    // Try to get the default branch from the remote
    if let Some(remote_head) = run_git_command(&["symbolic-ref", "refs/remotes/origin/HEAD"]) {
        // Output format: "refs/remotes/origin/main"
        if let Some(branch) = remote_head.strip_prefix("refs/remotes/origin/") {
            return branch.to_string();
        }
    }

    // Try to get from git config
    if let Some(default_branch) = run_git_command(&["config", "--get", "init.defaultBranch"]) {
        if !default_branch.is_empty() {
            return default_branch;
        }
    }

    // Fallback: check if common branches exist
    for branch in &["main", "master"] {
        if run_git_command(&["rev-parse", "--verify", branch]).is_some() {
            return branch.to_string();
        }
    }

    // Ultimate fallback
    "main".to_string()
}


pub(crate) fn build_system_prompt(model: &str) -> String {
    let pwd = env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_string()))
        .unwrap_or_else(|| ".".to_string());

    let is_repo = run_git_command(&["rev-parse", "--git-dir"]).is_some();

    let platform = if cfg!(target_os = "windows") {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    };

    let os_version = std::env::consts::OS;

    let date = chrono::Local::now().format("%Y-%m-%d").to_string();

    let (branch, git_status, last_5_commits) = if is_repo {
        let branch = run_git_command(&["rev-parse", "--abbrev-ref", "HEAD"])
            .unwrap_or_else(|| "unknown".to_string());

        let status = run_git_command(&["status", "--short"])
            .unwrap_or_else(|| "".to_string());

        let commits = run_git_command(&["log", "--oneline", "-5"])
            .unwrap_or_else(|| "".to_string());

        (branch, status, commits)
    } else {
        ("N/A".to_string(), "".to_string(), "".to_string())
    };

    let main_branch = get_main_branch();

    SYSTEM_PROMPT
        .replace("$pwd", &pwd)
        .replace("$is_repo", &is_repo.to_string())
        .replace("$platform", platform)
        .replace("$os_version", os_version)
        .replace("$date", &date)
        .replace("$branch", &branch)
        .replace("$main_branch", &main_branch)
        .replace("$git_status", &git_status)
        .replace("$model", if model.is_empty() { "unknown" } else { model })
        .replace("$last_5_commits", &last_5_commits)
}

// Model response structures

pub(crate) const PROJECT_CONTEXT_FILES: &[&str] = &["HAICO.md", "AGENTS.md", "CLAUDE.md"];

// Load the project's own instructions, if it has any.
//
// Without this there is no way to tell the model a project's conventions except by retyping
// them every session or editing the global system prompt. Capped because an oversized file
// would silently eat the context window it is meant to make good use of.

pub(crate) fn load_project_context() -> Option<(String, String)> {
    const MAX_BYTES: usize = 32_768;

    for name in PROJECT_CONTEXT_FILES {
        let path = PathBuf::from(name);
        let Ok(contents) = fs::read_to_string(&path) else {
            continue;
        };
        if contents.trim().is_empty() {
            continue;
        }

        let trimmed = truncate_at_boundary(&contents, MAX_BYTES);
        let mut text = trimmed.to_string();
        if trimmed.len() < contents.len() {
            text.push_str("\n\n[project context truncated]");
        }
        return Some((name.to_string(), text));
    }

    None
}

// Settings that vary by model rather than by server. Context window is a genuine property
// of the model; sampling is usually chosen per model too.
//
// Every field is optional and nothing has a built-in default, because "unset" has to stay
// distinguishable from "set to the default". LM Studio lets you tune sampling per model in
// its own UI, and a client that always sends temperature would silently override that with
// no indication why the model behaves differently through haico.
// Key names match the wire field exactly, so what you write here is what the server sees.
//
// top_k and repeat_penalty are not part of the OpenAI API — they are extensions that
// llama.cpp, Ollama, LM Studio and vLLM all understand. A strictly-OpenAI endpoint will
// ignore or reject them, which is why nothing is sent unless it was asked for.
