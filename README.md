# Haico

[![Build Status](https://img.shields.io/github/workflow/status/Revertron/CI)](https://github.com/Revertron/actions)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE.md)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Skills](#skills)
- [MCP Server Mode](#mcp-server-mode)
- [License](#license)

---

## Overview

**Haico** is a Rust‑based interactive CLI assistant that integrates an LLM client (via **LM Studio** or **Ollama**) with a custom “MCP” tool router.  
The system enables seamless interaction between a language model and programmable actions on the host environment.

### How it works
1️⃣ **User Input** → The CLI reads user commands.
2️⃣ **LLM Interaction** → The input is sent to the LLM, which generates a response that may include tool calls.
3️⃣ **Tool Parsing** → The system parses any tool invocation from the model’s output.
4️⃣ **Permission Prompt** → A permission manager asks the user for safe execution of potentially risky tools (e.g., web access or shell commands).
5️⃣ **Execution** → The selected tool is executed, and its result is fed back into the conversation.

---

## Features
- **Calculator** – arithmetic operations (`add`, `subtract`, `multiply`, `divide`).
- **File Operations** – read, write, edit files (including exact string replacement).
- **Glob Search** – fast pattern matching across directories (`**/*.rs`, etc.).
- **Regex Grep** – search for patterns in files with optional case‑insensitivity.
- **Web Tools** – `web_fetch` (fetch a page and summarize it via the LLM), `web_get` (fetch a page and return its cleaned text), and `web_search` (DuckDuckGo, or a SearxNG instance via `--searxng`).
- **Directory Listing** – list contents of a folder.
- **Command Execution** – `run_command` runs shell commands with timeout controls.
- **Skills** – reusable instructions for particular tasks, in the same `SKILL.md` format other coding agents use; the model loads one when a request matches it, or you run it with `/<name>`.
- **MCP Server Mode** – run as a stdio MCP server so other MCP hosts (e.g. LM Studio) can use Haico's tools.

All tools are described via JSON schemas, making them easy to extend or integrate with other systems.

---

## Installation
### Prerequisites
- Rust ≥ 1.56 (installed via `rustup`).
- **Ollama** or **LM Studio** (or any OpenAI-compatible LLM API server).

```bash
# Clone the repository
git clone https://github.com/Revertron/haico.git
cd haico

# Build the project
cargo build --release
```

Optional: Install additional dependencies if required (e.g., `reqwest` for web fetching, `regex` crate).

---

## Configuration

Haico supports flexible configuration through command-line arguments and environment variables, making it compatible with both **Ollama** and **LM Studio**.

### Command-Line Options

```bash
haico [OPTIONS]

Options:
  -u, --url <URL>              LLM server URL (default: http://127.0.0.1:11434)
  -m, --model <MODEL>          Model name (skips interactive selection)
  -t, --token <TOKEN>          API token for servers with authentication enabled
  -s, --system-prompt <PROMPT> System prompt (string or path to text file)
      --searxng <URL>          SearxNG instance for web_search (default: DuckDuckGo)
      --server                 Run as a stdio MCP server (no REPL; for MCP hosts like LM Studio)
  -h, --help                   Print help message
```

The URL is normalized before use: a trailing `/` or `/v1` is stripped, so
`http://127.0.0.1:1234`, `http://127.0.0.1:1234/`, and `http://127.0.0.1:1234/v1` are
equivalent. (LM Studio's UI and OpenAI SDKs present the base URL with `/v1` appended;
without stripping, that would build `/v1/v1/chat/completions`, which LM Studio rejects
while still answering HTTP 200.)

### Environment Variables

- **`LLM_URL`** - LLM server URL (overridden by `--url`)
- **`LM_API_TOKEN`** - API token for authenticated servers (overridden by `--token`)
- **`SEARXNG_URL`** - SearxNG instance for `web_search` (overridden by `--searxng`)

### Configuration Priority

Parameters are resolved in the following order:
1. Command-line arguments (highest priority)
2. Environment variables
3. Default values (lowest priority)

### Examples

#### Connect to Ollama (default)
```bash
# Uses default Ollama port (11434)
./haico

# Explicitly specify Ollama URL
./haico --url http://127.0.0.1:11434
```

#### Connect to LM Studio
```bash
# LM Studio uses port 1234 by default
./haico --url http://127.0.0.1:1234

# Or use environment variable
LLM_URL=http://127.0.0.1:1234 ./haico
```

#### Connect to a server with authentication enabled

LM Studio can require an API token (Settings → Developer → enable authentication). Pass it
with `--token`, or keep it out of your shell history via `LM_API_TOKEN`:

```bash
./haico --url http://127.0.0.1:1234 --token sk-lm-...

# Preferred: token via environment, not on the command line
LM_API_TOKEN=sk-lm-... ./haico --url http://127.0.0.1:1234
```

Without a token, an authenticated server answers every request with `401` and haico
reports it directly rather than failing to parse the error body.

#### Search through SearxNG instead of DuckDuckGo

`web_search` uses DuckDuckGo by default, which throttles and eventually answers with a
bot-detection challenge. Point `--searxng` at a [SearxNG](https://docs.searxng.org/)
instance (your own, ideally) to search through it instead:

```bash
./haico --searxng http://127.0.0.1:8080

# Or via environment variable
SEARXNG_URL=http://127.0.0.1:8080 ./haico
```

Pass the instance's base URL; haico appends `/search` itself. When set, SearxNG replaces
DuckDuckGo entirely — there is no fallback between providers, so a failing instance is
reported rather than silently worked around.

Haico first asks for JSON, which parses cleanly. SearxNG does **not** enable that by
default, so add it to the instance's `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

Without it, haico falls back to scraping the HTML result page, which works but is tied to
SearxNG's page markup. Note that most *public* instances run behind an anti-bot challenge
that blocks both paths; a self-hosted instance is the reliable option.

#### Skip Model Selection
```bash
# Directly specify model name
./haico --model "qwen2.5:14b"

# Works with any server
./haico --url http://127.0.0.1:1234 --model "llama-3.2-3b-instruct"
```

#### Custom System Prompt

You can customize the system prompt either as a string or from a file:

```bash
# Use inline string
./haico --system-prompt "You are a helpful coding assistant specializing in Rust"

# Load from file
./haico --system-prompt ./my_custom_prompt.txt
```

**Note:** The system prompt file can be a plain text file containing your custom instructions. If the path doesn't exist, it will be treated as a literal string.

#### Complete Example
```bash
# Connect to LM Studio with specific model and custom prompt
./haico \
  --url http://127.0.0.1:1234 \
  --model "llama-3.2-3b-instruct" \
  --system-prompt "./prompts/coding_assistant.txt"
```

---

## Usage
Run the CLI assistant:

```
./target/release/haico.exe
```

or:

```bash
./target/release/haico
```

You’ll see a prompt where you can type natural language commands.  
Examples:
- **Calculate:** `What is 5 + 3?`
- **Read file:** `Show me the contents of README.md`.
- **Search web:** `Find articles about Rust concurrency.`

The assistant will automatically parse tool calls, ask for permission (if needed), and execute them.

---

## Skills

A skill is a set of instructions for doing one kind of task well — cutting a release, writing a migration, reviewing a diff — kept in a file instead of being typed into every conversation. Haico uses the same format as other coding agents, so skills written for them work as they are.

Each skill is a directory holding a `SKILL.md`, plus any scripts, templates or reference files its instructions mention:

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

The directory's name is the skill's name. The `description` is what the model sees in every conversation, so write it to say **when** the skill applies, not just what it is about.

### Where skills are found

Most specific first — when two places have a skill with the same name, the earlier one wins:

1. `.haico/skills/` in the project
2. `.agents/skills/` in the project
3. `.claude/skills/` in the project
4. `skills/` in Haico's own data directory, for skills you want in every project — `%LOCALAPPDATA%\Haico\skills` on Windows, `~/Library/Application Support/Haico/skills` on macOS, `$XDG_DATA_HOME/haico/skills` (or `~/.local/share/haico/skills`) on Linux

### How they are used

Only one line per skill — its name and description — goes into the system prompt, so having many skills costs little. When a request matches a description, the model calls the `skill` tool to load the full instructions, then follows them. Loading a skill never asks for permission, since it only reads a file; anything the skill then tells the model to run goes through the usual permission prompts.

You can also run a skill yourself by typing its name as a command, with anything after it passed along:

```
/release 1.4.0
```

| Command | What it does |
| --- | --- |
| `/skills` | List every skill found, with its description and where it lives |
| `/update-skills` | Re-read the skill directories after adding or editing one mid-session |
| `/<name> [args]` | Run a skill directly |

Skills are available in the interactive REPL only; in [MCP Server Mode](#mcp-server-mode) the host is expected to have its own.

---

## MCP Server Mode

In addition to the interactive REPL, Haico can run as a **stdio MCP server**, exposing its tools to any MCP host (such as **LM Studio**). Pass `--server` to start in this mode:

```bash
./haico --server
```

In server mode Haico speaks the Model Context Protocol over stdin/stdout, so it prints no banner or prompts (diagnostics go to stderr) and does **not** select a model interactively. The MCP host launches the binary and decides when to call each tool — including any confirmation, since Haico's interactive permission prompts do not apply in server mode.

### Using with LM Studio

Build a release binary (`cargo build --release`) and register Haico in LM Studio's `mcp.json` (Program ▸ Edit `mcp.json`):

```json
{
  "mcpServers": {
    "haico": {
      "command": "C:\\path\\to\\haico\\target\\release\\haico.exe",
      "args": ["--server", "--url", "http://127.0.0.1:1234", "--model", "qwen2.5:14b"]
    }
  }
}
```

On non-Windows platforms use the plain path to the `haico` binary. Haico's tools then appear in LM Studio and can be called by the model.

If the LLM server Haico talks to requires an API token, pass it through here too — either
as another entry in `args` (`"--token", "sk-lm-..."`) or, to keep the secret out of
`mcp.json`, via the host's environment:

```json
{
  "mcpServers": {
    "haico": {
      "command": "C:\\path\\to\\haico\\target\\release\\haico.exe",
      "args": ["--server", "--url", "http://127.0.0.1:1234", "--model", "qwen2.5:14b"],
      "env": { "LM_API_TOKEN": "sk-lm-..." }
    }
  }
}
```

### Web tools in server mode

The two page-fetching tools differ, and which are exposed depends on configuration:

- **`web_get(url)`** – fetches a page and returns its cleaned text directly, with no LLM call. Always available.
- **`web_fetch(url, prompt)`** – fetches a page and uses the LLM to summarize/extract per the prompt. Because this makes a second LLM round-trip (which can exceed the host's request timeout on large pages), it is only exposed when an LLM is explicitly configured — i.e. when **both** `--url` and `--model` are passed. Otherwise it is hidden and only `web_get` is offered.

When a model is configured, both tools are exposed and the calling model picks whichever it needs. (In the interactive REPL a model is always selected, so both are always available there.)

---

## License
This project is licensed under the **MIT License**. See the [LICENSE.md](LICENSE.md) file for details.
