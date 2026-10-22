# Haico

**Haico** (Home AI Coder) is a coding agent for local LLMs. It is a single Rust binary: an
interactive terminal assistant that talks to any OpenAI-compatible server — **Ollama**,
**LM Studio**, **llama-server** — and runs tools on your machine on the model's behalf: reading
and editing files, running commands and builds, searching the web. The same binary can also run
as a **stdio MCP server**, exposing those tools to other MCP hosts.

It is built for models that run on your own GPU: tool descriptions written for small models,
sub-agents that keep long pages and logs out of the main history, loop and repetition guards.
It's designed primarily for efficient coding in Rust, but maybe used for other languages and environments too.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Web search](#web-search)
- [Using the REPL](#using-the-repl)
- [Unattended runs](#unattended-runs)
- [Project instructions and memory](#project-instructions-and-memory)
- [Skills](#skills)
- [The library](#the-library)
- [Tracing](#tracing)
- [MCP Server Mode](#mcp-server-mode)
- [License](#license)

---

## Features

- **File tools** – `read`, `write_file`, `edit`, `multi_edit`, `glob_search`, `grep`, `ls`.
  Every change can be reverted with `/undo`, byte for byte.
- **Shell** – `run_command` with a timeout (up to an hour), and `run_background` /
  `background_output` / `background_stop` for servers and long builds. Cargo output is shortened
  before the model sees it, never dropping an error.
- **Web** – `web_search` through one of three backends — DuckDuckGo (default), your own
  [SearxNG](#searxng), or the hosted [Tinyfish Search API](#tinyfish) — plus `web_get` (a page
  as clean text) and `web_fetch` (a page summarized by the same LLM).
- **Rust crates info** – `crate_info`, `crate_search` and `crate_api` look up crates.io and docs.rs.
- **Integrated sub-agents** – introduced to save context, `research` answers a question from the web in its own context,
  and universal `sub_agent` reads the long log or the big file for the model; only their answers enter the main
  history.
- **The library** – also a sub-agent, your own articles about the crates you build on, any topics like APIs,
  answered from by a librarian sub-agent and suggested automatically when a build fails.
- **Modes** – *manual* (you approve each risky call), *plan* (read-only until you approve a
  plan), and *auto* (a second, context-free copy of the model with specially tuned system-prompt reviews each call).
- **Planning** – `todo_write` keeps a visible task list; `ask_user` lets the model ask you a
  question with choices.
- **Project memory** – facts about the project that outlive a conversation (`remember`, `recall`).
- **Skills** – reusable instructions in the `SKILL.md` format other coding agents use.
- **Timers** – the model can remind itself later: once, every N seconds, or on a cron schedule.
- **Sessions** – conversations are saved per directory and continued with `--resume`;
  `/compact` (or automatic compaction near the context limit) replaces the history with notes.
- **Unattended runs** – `-p "task"` runs one task in auto mode and exits with a status code.
- **Tool subsets** – `--tools read,grep,ls` offers the model only the tools you list.
- **Tracing** – every model call recorded as JSON lines, for building training data.
- **MCP Server Mode** – the same tools over the Model Context Protocol.

---

## Installation

### Prerequisites
- A recent stable Rust toolchain (installed via `rustup`).
- An OpenAI-compatible LLM server: **Ollama**, **LM Studio**, **llama-server**, or similar.

```bash
git clone https://github.com/Revertron/Haico.git
cd haico
cargo build --release
```

The binary is `target/release/haico` (`haico.exe` on Windows).

There is also an experimental window frontend behind `--features gui`. It is off by default
and needs the Lumio GUI library checked out next to this repository (`../Lumio`).

---

## Quick start

```bash
# Ollama on its default port, picking a model from the server's list
haico

# LM Studio, with a model chosen up front
haico --url http://127.0.0.1:1234 --model tiel-coder-35b-a3b-mtp

# Continue the conversation you left in this directory
haico --resume
```

On first start Haico writes a commented starter config and tells you where it is. Every setting
in it is optional.

---

## Configuration

Settings are resolved per setting, in this order:

1. Command-line arguments
2. Environment variables
3. The model's own `[models."<name>"]` section in the config file
4. The config file's top level
5. Built-in defaults

`/config` in the REPL shows every setting in force and where it came from.

### The config file

`config.toml` lives in Haico's data directory, along with the user instructions, saved
sessions, input history, skills and library:

| Platform | Directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Haico` |
| macOS | `~/Library/Application Support/Haico` |
| Linux | `$XDG_DATA_HOME/haico`, or `~/.local/share/haico` |

Unknown keys are an error, not silently ignored, so a typo is caught.

```toml
version = 1

# Servers to switch between with /provider; each is a section below.
providers = ["lmstudio", "ollama"]
provider  = "lmstudio"              # the one to start with (default: the first)

searxng    = "http://127.0.0.1:8888"  # or tinyfish = "your-api-key", not both
read_paths = ["C:/src/shared-lib"]  # readable by the file tools, never writable
tui        = "fullscreen"           # or "inline"
mouse      = "scroll"               # or "off"

[lmstudio]
url   = "http://127.0.0.1:1234"
token = "sk-lm-..."
model = "qwen2.5:14b"               # the model to start with on this server

[ollama]
url = "http://127.0.0.1:11434"

[models."tiel-coder-35b-a3b-mtp"]
context          = 131072
temperature      = 0.3
top_p            = 0.95
top_k            = 40
min_p            = 0.05
repeat_penalty   = 1.05
reasoning_effort = "high"   # sent to the server as-is; "on"/"off" toggle thinking
sub_agent_effort = "low"    # the same for Haico's own sub-agents
reasoning        = "keep"   # keep | compress | drop the chain of thought after a turn
use_system_prompt = false   # send only your own and the project's instructions
```

A file without `version` is read the older way, with `url`, `token` and `model` at the top level.
`--url`, `--token` and `--model` still override the provider's own values one by one.

Two choices are written back to the file, editing only that key and keeping your comments: the
model picked with `/model` (and the provider picked with `/provider`), and a model's context
window set with `/set context`. Other `/set` changes last for the session.

### Command-line options

```
    --provider NAME       Use this provider from the config file for this run
-u, --url URL             LLM server URL (default: http://127.0.0.1:11434 for Ollama)
-m, --model MODEL         Model name (skip model selection)
-t, --token TOKEN         API token for servers with authentication enabled
-s, --system-prompt TEXT  System prompt (string or path to text file)
    --searxng URL         SearxNG instance to search through instead of DuckDuckGo
    --tinyfish KEY        Tinyfish Search API key, to search through Tinyfish instead
    --context TOKENS      Model context window in tokens (default: 131072)
    --temperature VALUE   Sampling settings; each is left to the server unless given
    --max-tokens N
    --top-p VALUE
    --top-k N
    --min-p VALUE
    --repeat-penalty VALUE
-r, --resume              Continue this directory's previous conversation
-p, --prompt TASK         Run this task unattended in auto mode, then exit
    --tools LIST          Offer the model only these tools (comma-separated)
-d, --dir DIR             Work in DIR instead of the current directory
    --trace FILE          Append every model call to FILE as JSON lines
    --trace-lib FILE      Append each library question and answer to FILE
    --trace-research FILE Append each research question and answer to FILE
    --trace-gaps FILE     Append each question the library could not answer to FILE
    --server              Run as a stdio MCP server (no REPL, no prompts)
    --gui                 Open a window instead of the terminal (needs --features gui)
-h, --help                Print help, including a commented config example
```

### Environment variables

| Variable | Meaning |
| --- | --- |
| `HAICO_PROVIDER` | Provider from the config file (overridden by `--provider`) |
| `LLM_URL` | LLM server URL (overridden by `--url`) |
| `LM_API_TOKEN` | API token for authenticated servers (overridden by `--token`) |
| `SEARXNG_URL` | SearxNG instance for `web_search` (overridden by `--searxng`) |
| `TINYFISH_API_KEY` | Tinyfish key for `web_search` (overridden by `--tinyfish`) |
| `HAICO_CONTEXT` | Model context window in tokens (overridden by `--context`) |
| `HAICO_TUI` | `inline` or `fullscreen`, overriding `tui` |
| `HAICO_DEBUG` | Set to `1` to append diagnostics to `debug.log` in the data directory |
| `NO_COLOR` | Turn colors off |

### The server URL

The URL is normalized before use: a trailing `/` or `/v1` is stripped, so
`http://127.0.0.1:1234`, `http://127.0.0.1:1234/` and `http://127.0.0.1:1234/v1` are
equivalent. (LM Studio's UI and OpenAI SDKs show the base URL with `/v1` appended; without
stripping it, Haico would request `/v1/v1/chat/completions`, which LM Studio rejects while
still answering HTTP 200.)

Requests identify themselves as `User-Agent: Haico/<version>`. Server-side failures (timeouts,
429s, 5xx, dropped connections) are retried with backoff — 2, 4, 8, 16 and 32 seconds, or the
server's `Retry-After` up to a minute — and the retry is shown in the transcript.

### Authentication

LM Studio can require an API token (Settings → Developer → enable authentication). Put it in
the provider's section of the config, pass it with `--token`, or keep it out of your shell
history with `LM_API_TOKEN`:

```bash
LM_API_TOKEN=sk-lm-... haico --url http://127.0.0.1:1234
```

Without a token, an authenticated server answers every request with `401`, and Haico says so
directly.

---

## Web search

`web_search` goes through exactly one backend, chosen at startup (SearxNG and Tinyfish are named
in the startup banner):

| Backend | How to choose it | Good for |
| --- | --- | --- |
| DuckDuckGo | the default; nothing to set | trying Haico out — it throttles and eventually answers with a bot-detection challenge |
| [SearxNG](#searxng) | `--searxng URL`, `SEARXNG_URL`, or `searxng = "..."` | a self-hosted metasearch engine with no limits but your own |
| [Tinyfish](#tinyfish) | `--tinyfish KEY`, `TINYFISH_API_KEY`, or `tinyfish = "..."` | a hosted API, with no server to run |

Whichever is set replaces DuckDuckGo entirely: there is no fallback between backends, so a
failing one is reported rather than silently worked around. Only one of `searxng` and
`tinyfish` may be set at a time. The command line outranks the environment, which outranks the
config file, so `--tinyfish` overrides a `searxng` in the file, and `--searxng ""` switches back
to DuckDuckGo for one run.

All three feed the same tools: the model searches with `web_search`, and `research` uses it to
find pages before reading them in its own conversation. The backend changes nothing else.

### SearxNG

Point `--searxng` (or `searxng` in the config) at a [SearxNG](https://docs.searxng.org/)
instance — your own, ideally. Pass the instance's base URL; Haico appends `/search` itself.

Haico asks for JSON first, which SearxNG does **not** enable by default. Add it to the
instance's `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

Without it, Haico falls back to scraping the HTML result page, which works but depends on
SearxNG's markup. Most *public* instances sit behind an anti-bot challenge that blocks both;
a self-hosted instance is the reliable option.

### Tinyfish

The [Tinyfish Search API](https://docs.tinyfish.ai/search-api) is the hosted option: pass your
API key with `--tinyfish` (or `TINYFISH_API_KEY`, or `tinyfish` in the config). Keep the key
out of your shell history with the environment variable:

```bash
TINYFISH_API_KEY=your-key haico
```

Each search is one request for one page of results. A rejected key (`401`), a spent allowance
(`402`) and the rate limit (`429`, 30 searches a minute by default) are each reported to the
model as such.

---

## Using the REPL

Type what you want in plain language. The model reads, searches, edits and runs commands,
asking first for anything that changes your machine.

### Modes

Shift+Tab cycles between them, or use the commands:

- **Normal** (`/normal`) – read-only tools run on their own; edits, shell commands and web
  access ask. You can approve a command once, or a whole command family for the session.
  Approvals are never saved.
- **Plan** (`/plan`) – the model may only read and research. It finishes by presenting a plan;
  nothing changes until you approve it.
- **Auto** (`/auto`) – a second copy of the model, which sees only the call and not the
  conversation, reviews each gated call. Routine ones run; refused ones are explained to the
  model instead of stopping for you.

Whatever the mode, the file tools can only write inside the working directory (plus anything you
`/add`), destructive shell commands are refused outright, and commands that run an interpreter
need approval of the exact command line.

### Commands

| Command | What it does |
| --- | --- |
| `/help` | Show the full list, with keyboard shortcuts |
| `/plan`, `/auto`, `/normal`, `/mode` | Switch modes, or show the current one |
| `/add <path>`, `/workspace` | Let the file tools reach another directory; show what they may touch |
| `/cd [path]` | Move to another project |
| `/init` | Study the project and write its instruction file |
| `/model [name]`, `/provider [name]` | Switch model or server, and save it as the default |
| `/config` | Show the settings in force and where each came from |
| `/set [name] [value]` | Show or change context, sampling and reasoning settings (`/temp`, `/effort` for short) |
| `/reasoning [keep\|compress\|drop]` | What happens to the model's chain of thought after a turn |
| `/context`, `/compact` | Show context use; replace the history with notes |
| `/undo` | Revert the last file change made by a tool |
| `/todo` | Show the model's task list |
| `/bg [stop [id]]` | List or stop background jobs |
| `/timers [cancel [id]]` | List or cancel the model's timers |
| `/tools` | List the tools and whether each one asks |
| `/skills`, `/update-skills`, `/<skill> [args]` | List, reload or run skills |
| `/resume`, `/clear` | Load this directory's saved conversation; start over |
| `/exit` | Leave (Ctrl+D also works) |

Typing `/` opens a list of matching commands and skills above the input box. A mistyped command
is answered with "Unknown command" instead of being sent to the model.

### In a message and at the keyboard

- `@path` attaches a file to your message, so the model does not spend a round reading it.
  Images (PNG, JPEG, GIF, WEBP) go as pictures to models that can see them.
- **Shift+Enter** adds a line; **up/down** walk the input history.
- While the model works, keep typing — **Enter** queues the message for the end of the turn,
  and **Esc** interrupts.
- **Ctrl+O** unfolds what the transcript folded away: full tool output, whole diffs, reasoning.
- **Ctrl+T** cycles the plan panel.

The default fullscreen view scrolls with the mouse wheel and PageUp/PageDown. Set
`tui = "inline"` (or `HAICO_TUI=inline`) to use the terminal's own scrollback instead. On
Windows, hold Shift to select text while the wheel is on, or set `mouse = "off"`.

---

## Unattended runs

`-p` runs one task in the current directory (or `--dir`) in auto mode, with no keyboard, and
exits:

```bash
haico -d ~/src/project -p "Make the failing test in tests/parse.rs pass"
```

The model decides its own questions, and anything the reviewer refuses is not run. A turn that
stops with its plan unfinished, ends on a question, or says nothing is sent back to work up to
three times. Exit codes: `0` done, `1` a turn failed, `2` still unfinished after the follow-ups,
`130` interrupted. Timers the model set keep the run alive until they fire.

A fresh `-p` run does not save its conversation, so a scheduled run never replaces the one you
left in that directory; `-p --resume` continues that conversation and saves it.

### Limiting the tools

`--tools` offers the model only the tools you list, in a `-p` run or in the REPL:

```bash
haico -p "Explain how config.rs resolves settings" --tools read,grep,glob_search,ls
```

A call to any other tool is refused and the model is told which tools it has; the `sub_agent`
gets the same subset. A misspelled name stops the run with the list of real ones, and an empty
list (`--tools ""`) offers no tools at all, for a task that is only text. The list narrows what
the model is offered, not what Haico runs on its behalf: `--tools research` still lets the
research sub-agent search and read pages.

---

## Project instructions and memory

- **Your instructions** – `HAICO.md` in Haico's data directory, applied everywhere.
- **The project's instructions** – the first of `HAICO.md`, `AGENTS.md` or `CLAUDE.md` in the
  working directory. It takes precedence over yours. `/init` writes one for you.
- **Project memory** – the model records facts with `remember` (it does so when you ask it to
  remember something) in `.haico/memory/`, one Markdown file per fact. An index of them is in
  every conversation; the model reads one in full with `recall`. Edit or delete the files by
  hand whenever you like — the directory is the truth, and it ignores itself in git.

---

## Skills

A skill is a set of instructions for doing one kind of task well — cutting a release, writing a
migration, reviewing a diff — kept in a file instead of being typed into every conversation.
Haico uses the same format as other coding agents, so skills written for them work as they are.

Each skill is a directory holding a `SKILL.md`, plus any scripts, templates or reference files
its instructions mention:

```
release/
├── SKILL.md
└── scripts/
    └── bump_version.py
```

```markdown
---
name: release
description: Cut a new release. Use when the user asks to release, tag or publish a version.
---

1. Run `python scripts/bump_version.py <version>`.
2. Update CHANGELOG.md with the commits since the last tag.
3. Commit, then tag the commit `v<version>`.
```

The directory's name is the skill's name. The `description` is what the model sees in every
conversation, so write it to say **when** the skill applies, not just what it is about.

### Where skills are found

Most specific first — when two places have a skill with the same name, the earlier one wins:

1. `.haico/skills/` in the project
2. `.agents/skills/` in the project
3. `.claude/skills/` in the project
4. `skills/` in Haico's data directory, for skills you want in every project

### How they are used

Only one line per skill — its name and description — goes into the system prompt, so having
many skills costs little. When a request matches a description, the model calls the `skill`
tool to load the full instructions, then follows them. Loading a skill never asks for
permission, since it only reads a file; anything the skill then tells the model to run goes
through the usual permission prompts.

You can also run a skill yourself by typing its name as a command, with anything after it passed
along:

```
/release 1.4.0
```

`/skills` lists every skill found and where it lives; `/update-skills` re-reads the directories
after you add or edit one mid-session. Skills are available in the REPL only; in
[MCP Server Mode](#mcp-server-mode) the host is expected to have its own.

---

## The library

The library is a set of Markdown articles about the crates your projects are built on — the
async runtime, the web framework, the error-handling crate — kept in `library/` in Haico's data
directory as `library/<crate>/<topic>.md`. The articles are your own; Haico ships none.

The model asks it questions with the `library` tool. A librarian sub-agent searches the
articles, reads the relevant sections and hands back one answer, so the articles themselves never
fill the main conversation. When a `cargo build`, `check` or `test` fails, hangs or times out
and one of the crates involved has articles, the command's output ends with a note suggesting a
question to ask.

Articles take optional frontmatter:

```markdown
---
title: Shutting down a Tokio runtime
crate: tokio
versions: "1"
summary: Why a runtime hangs on drop, and how to stop its tasks.
codes: E0277
matches: cannot be sent between threads safely
---
```

`crate:` may list several crates; `topic:` makes an article about the language instead of a
crate; `codes:` and `matches:` tie it to compiler errors, so a failing build points at it even
when no crate is named. A section starting with a `Tags:` line is found by those words too.

`scripts/check_library.py` compiles every `rust` block in the articles against the versions
pinned in a `check.toml` beside them, checks that `compile_fail` blocks fail with the stated
error code, and with `--run` runs them, so an article cannot teach a wrong signature.

---

## Tracing

`--trace runs.jsonl` appends every model call as one JSON line: the request exactly as sent and
the whole reply, reasoning included, tagged with which conversation made it (main, research,
reviewer, compressor, compactor, summarizer). Nothing is rewritten afterwards, so compaction
mid-run loses nothing.

```bash
haico -p "Add a --verbose flag" --trace runs.jsonl
python scripts/trace_to_dataset.py runs.jsonl -o dataset.jsonl
```

`--trace-lib`, `--trace-research` and `--trace-gaps` record library answers, research answers
and the questions the library could not answer, one question/answer pair per line.

---

## MCP Server Mode

Haico can also run as a **stdio MCP server**, exposing its tools to any MCP host (such as
**LM Studio**):

```bash
haico --server
```

In server mode Haico speaks the Model Context Protocol over stdin/stdout, so it prints no banner
or prompts (diagnostics go to stderr) and does **not** select a model interactively. The host
decides when to call each tool, including any confirmation — Haico's permission prompts, modes
and workspace limits do not apply. Tools that need Haico's own loop (`skill`, `timer`,
`sub_agent`) are not offered.

### Using with LM Studio

Build a release binary and register Haico in LM Studio's `mcp.json` (Program ▸ Edit `mcp.json`):

```json
{
  "mcpServers": {
    "haico": {
      "command": "C:\\path\\to\\haico\\target\\release\\haico.exe",
      "args": ["--server", "--url", "http://127.0.0.1:1234", "--model", "tiel-coder-35b-a3b-mtp"],
      "env": { "LM_API_TOKEN": "sk-lm-..." }
    }
  }
}
```

On other platforms use the plain path to the `haico` binary. The `env` entry is only needed when
the LLM server requires a token.

### Web tools in server mode

- **`web_get(url)`** – fetches a page and returns its cleaned text, with no LLM call. Always
  available.
- **`web_fetch(url, prompt)`**, **`research(question)`** and **`library(question)`** – use the
  LLM themselves, which takes a second round trip that can exceed the host's request timeout.
  They are offered only when Haico is told where the LLM is (`--url`, `LLM_URL` or the config);
  without a model, they use whatever the server has loaded. `library` also needs articles to
  answer from.

---
## Contribution

Any pull requests created in this repo without prior conversation with the author of Haico and his approval are rejected.

---

## License

This project is licensed under the **MIT OR Apache-2.0 License**.
