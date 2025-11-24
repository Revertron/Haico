# Haico

[![Build Status](https://img.shields.io/github/workflow/status/Revertron/CI)](https://github.com/Revertron/actions)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE.md)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
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
- **Web Fetch & Search** – retrieve and parse web content using DuckDuckGo.
- **Directory Listing** – list contents of a folder.
- **Bash Execution** – run shell commands with timeout controls.

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
  -s, --system-prompt <PROMPT> System prompt (string or path to text file)
  -h, --help                   Print help message
```

### Environment Variables

- **`LLM_URL`** - LLM server URL (overridden by `--url`)

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

## License
This project is licensed under the **MIT License**. See the [LICENSE.md](LICENSE.md) file for details.
