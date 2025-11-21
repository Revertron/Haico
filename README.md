# Coding MCP Server (working name)

[![Build Status](https://img.shields.io/github/workflow/status/Revertron/CI)](https://github.com/Revertron/actions)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE.md)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Coding MCP Server** is a Rust‑based interactive CLI assistant that integrates an LLM client (via **LM Studio**) with a custom “MCP” tool router.  
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
- **LM Studio** client (or any compatible LLM API).

```bash
# Clone the repository
git clone https://github.com/Revertron/coding-mcp-server.git
cd coding-mcp-server

# Build the project
cargo build --release
```

Optional: Install additional dependencies if required (e.g., `reqwest` for web fetching, `regex` crate).

---

## Usage
Run the CLI assistant:

```bash
./target/release/coding_mcp_server
```

You’ll see a prompt where you can type natural language commands.  
Examples:
- **Calculate:** `What is 5 + 3?`
- **Read file:** `Show me the contents of README.md`.
- **Search web:** `Find articles about Rust concurrency.`

The assistant will automatically parse tool calls, ask for permission (if needed), and execute them.

---

## Contributing
We welcome contributions! Please follow these steps:
1️⃣ Fork the repository.
2️⃣ Create a new branch (`git checkout -b feature-name`).
3️⃣ Implement your changes.
4️⃣ Run tests (`cargo test`) to ensure stability.
5️⃣ Submit a pull request with a clear description of your changes.

Feel free to open issues for bugs, feature requests, or documentation improvements.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE.md](LICENSE.md) file for details.
