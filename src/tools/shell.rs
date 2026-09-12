use super::MyMcpServer;
use crate::util::{truncate_at_boundary, truncate_middle};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct RunCommandParams {
    #[schemars(description = "The shell command to execute")]
    pub(crate) command: String,
    #[schemars(description = "Timeout in seconds (default: 120, max: 600)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) timeout: Option<u64>,
    #[schemars(
        description = "Directory to run the command in. Prefer this over a leading 'cd', which only applies within a single call."
    )]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cwd: Option<String>,
}


// A command's output, with a ceiling.
//
// A background dev server logs for as long as it runs, so the buffer has to be bounded. The
// oldest bytes go first — the newest are what anyone is asking about — and `dropped` records
// how many were discarded so a reader's absolute offset still means something afterwards.
pub(crate) struct OutputBuffer {
    data: Vec<u8>,
    dropped: usize,
    cap: usize,
}

impl OutputBuffer {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            data: Vec::new(),
            dropped: 0,
            cap,
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
        if self.data.len() > self.cap {
            let excess = self.data.len() - self.cap;
            self.data.drain(..excess);
            self.dropped += excess;
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.data
    }

    // Total bytes ever written, whether or not they are still held.
    pub(crate) fn total(&self) -> usize {
        self.dropped + self.data.len()
    }

    // Everything past an absolute offset, clamped to what survives.
    pub(crate) fn since(&self, offset: usize) -> &[u8] {
        &self.data[offset.saturating_sub(self.dropped).min(self.data.len())..]
    }

    pub(crate) fn dropped(&self) -> usize {
        self.dropped
    }
}

pub(crate) async fn pump_output<R>(mut reader: R, echo: bool, sink: Arc<Mutex<OutputBuffer>>)
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;

    let mut echoed_to = 0usize;
    let mut chunk = [0u8; 8192];

    loop {
        let read = match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };

        let mut collected = sink.lock().unwrap();
        collected.push(&chunk[..read]);

        if echo {
            let data = collected.bytes();
            while let Some(offset) = data[echoed_to..].iter().position(|&b| b == b'\n') {
                let end = echoed_to + offset;
                let line = String::from_utf8_lossy(&data[echoed_to..end]);
                outln!("  │ {}", line.trim_end_matches('\r'));
                echoed_to = end + 1;
            }
        }
    }

    // A final line with no trailing newline (a progress bar, a prompt) still deserves showing.
    if echo {
        let collected = sink.lock().unwrap();
        let data = collected.bytes();
        if echoed_to < data.len() {
            let line = String::from_utf8_lossy(&data[echoed_to..]);
            outln!("  │ {}", line.trim_end_matches('\r'));
        }
    }
}

// Kill a spawned shell *and everything it started*.
//
// A plain kill() only ends the shell itself. `cmd /C cargo build` that times out would leave
// rustc running, still holding the output pipe and still burning CPU, with nothing left to
// reap it. Neither platform makes this automatic, so each needs its own mechanism: taskkill
// walks the tree on Windows, and on Unix the child is made a process-group leader at spawn
// so the whole group can be signalled at once.

pub(crate) async fn kill_tree_by_pid(pid: u32) {
    #[cfg(target_os = "windows")]
    let mut killer = {
        let mut c = tokio::process::Command::new("taskkill");
        c.args(["/T", "/F", "/PID", &pid.to_string()]);
        c
    };

    // Negative pid means "the whole process group", which is why spawn sets one up.
    #[cfg(not(target_os = "windows"))]
    let mut killer = {
        let mut c = tokio::process::Command::new("kill");
        c.args(["-TERM", &format!("-{}", pid)]);
        c
    };

    killer
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let _ = killer.status().await;
}

pub(crate) async fn kill_process_tree(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        kill_tree_by_pid(pid).await;
    }

    // Always reap the direct child, whether or not the tree kill above worked.
    let _ = child.kill().await;
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct RunBackgroundParams {
    #[schemars(description = "The shell command to start in the background")]
    pub(crate) command: String,
    #[schemars(description = "Directory to run the command in")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cwd: Option<String>,
    #[schemars(
        description = "Text to wait for in the output before returning, e.g. 'Listening on'. Matched case-insensitively. Use it whenever something must be ready before the next step, so you do not act before a server is accepting connections."
    )]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) wait_for: Option<String>,
    #[schemars(description = "How long to wait for wait_for, in seconds (default: 30, max: 300)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) timeout: Option<u64>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct BackgroundOutputParams {
    #[schemars(description = "Job id. Omit to list every background job and its status.")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<u32>,
    #[schemars(
        description = "Return everything captured so far instead of only what is new since your last read (default: false)"
    )]
    #[serde(default)]
    pub(crate) from_start: bool,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub(crate) struct BackgroundStopParams {
    #[schemars(description = "Job id to stop. Omit to stop every running job.")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<u32>,
}

// One background command.
//
// The Child itself is owned by a waiter task so its exit status can be recorded the moment
// it happens; killing therefore goes through the pid rather than the handle.
pub(crate) struct BackgroundJob {
    pub(crate) command: String,
    pub(crate) pid: Option<u32>,
    pub(crate) output: Arc<Mutex<OutputBuffer>>,
    // Absolute offset already handed to the model, so a read returns only what is new.
    pub(crate) cursor: usize,
    pub(crate) started: std::time::Instant,
    pub(crate) exit: Arc<Mutex<Option<i32>>>,
    pub(crate) stopped: bool,
}

impl BackgroundJob {
    pub(crate) fn status(&self) -> String {
        // "stopped" wins over any exit code, because killing the tree is itself what
        // produced that code. Reporting "exited (1)" for a server the model deliberately
        // stopped would read as a crash and send it looking for a bug that is not there.
        if self.stopped {
            return "stopped".to_string();
        }
        match *self.exit.lock().unwrap() {
            Some(code) => format!("exited ({})", code),
            None => format!("running for {}s", self.started.elapsed().as_secs()),
        }
    }
}

#[derive(Default)]
pub(crate) struct BackgroundJobs {
    pub(crate) next_id: u32,
    pub(crate) jobs: std::collections::BTreeMap<u32, BackgroundJob>,
}

// A dev server can log indefinitely, so what it produced an hour ago is not worth holding.
const BACKGROUND_OUTPUT_CAP: usize = 256 * 1024;

// What one finished command may hand back to the model. Anything longer is cut in the middle
// rather than at the end — see truncate_middle for why.
const MAX_OUTPUT: usize = 30000;

// Build the shell invocation for a command line.
//
// On Windows this is deliberately not the obvious `Command::new("cmd").arg("/C").arg(command)`.
// A Windows process is handed one flat command line rather than an argument array, so `arg` has
// to serialize back into a string, and it does that by the MSVC C runtime's rules: an argument
// containing spaces is wrapped in quotes and every quote inside it is escaped as `\"`. cmd.exe
// does not parse by those rules and has no escape character at all, so the backslashes arrive
// literally and every quote in the command is destroyed. `curl -H "Host: example.com" URL`
// reached curl as `-H`, `"Host:` and a stray `example.com"` — the header was never sent, the
// request fell through to the default vhost, and the model was left concluding its own code was
// broken. A JSON `-d` payload and a quoted path with spaces failed the same way, the latter
// while the tool description was busy advising "quote paths with spaces". `raw_arg` appends the
// text with no quoting or escaping whatsoever; it exists for exactly this case.
//
// The wrapping quotes and `/S` go together, and neither works without the other. Plain `/C`
// preserves the outer quotes only under a list of conditions (two quotes exactly, no `&<>()@^|`
// between them, the text between them naming a real executable) and otherwise strips the first
// and *last* quote of the whole line — which cuts apart any command whose program path is
// itself quoted, as `"C:\Program Files\x.exe" --flag` is. `/S` makes the strip unconditional,
// so it removes exactly the pair added here and leaves everything between them untouched.
//
// Unix never had the problem: `sh -c` receives a real argument array, so nothing is
// re-serialized and the command reaches the shell as written.
pub(crate) fn shell_command(command: &str) -> tokio::process::Command {
    #[cfg(target_os = "windows")]
    let builder = {
        let mut c = tokio::process::Command::new("cmd");
        c.arg("/S").arg("/C").raw_arg(format!("\"{}\"", command));
        c
    };

    #[cfg(not(target_os = "windows"))]
    let builder = {
        let mut c = tokio::process::Command::new("sh");
        c.arg("-c").arg(command);
        c
    };

    builder
}

// What a command chained with `;` is told when it is refused on Windows.
//
// It has to name the replacement, not just the problem: a model that is only told `;` failed
// re-sends the same line with the halves reordered.
#[cfg(target_os = "windows")]
pub(crate) const SEMICOLON_REFUSAL: &str = "Not run: `;` does not chain commands in Windows cmd — \
    it is an ordinary argument delimiter, so everything after it would be handed to the first \
    program as extra arguments instead of being run. Use `&&` to run the next command only if \
    this one succeeds, `&` to run it either way, or send separate run_command calls.";

// Does this command line try to chain with `;`?
//
// cmd.exe has no `;` operator, and the way that fails is worse than not running at all:
// `curl -sS URL ; if exist hdr.txt type hdr.txt` reached curl as one long argument list, so
// curl went off trying to resolve `if`, `exist` and `hdr.txt` as hostnames. The errors all
// come from the *first* program, which reads like that program misbehaving rather than like
// the second command never having existed — the model debugs the wrong thing. The tool
// description has said "chain with &&" all along and a small model writes `;` anyway, so it is
// caught here instead, where the answer can name the fix.
//
// Only a `;` with whitespace on one side of it is a separator. `set PATH=C:\a;C:\b` and
// `for %i in (a;b) do ...` are ordinary cmd lines and must still run.
//
// Double quotes are tracked and single quotes deliberately are not: `'` has no quoting meaning
// in cmd, so a `;` between single quotes really is a separator there. This is the one place
// that differs from `command_segments`, which is cross-platform and must assume sh.
#[cfg(target_os = "windows")]
pub(crate) fn chains_with_semicolon(command: &str) -> bool {
    let chars: Vec<char> = command.chars().collect();
    let mut in_double = false;

    for (i, &c) in chars.iter().enumerate() {
        match c {
            '"' => in_double = !in_double,
            ';' if !in_double => {
                let before = i.checked_sub(1).map(|j| chars[j]);
                let after = chars.get(i + 1).copied();
                if before.is_none_or(char::is_whitespace) || after.is_none_or(char::is_whitespace) {
                    return true;
                }
            }
            _ => {}
        }
    }

    false
}

#[tool_router(router = shell_router, vis = "pub(crate)")]
impl MyMcpServer {
    #[tool(description = "Use to run shell commands and CLIs when no dedicated tool fits — building, testing, running scripts, or git/gh operations. Execute a shell command (cmd on Windows, sh elsewhere). Output is streamed to the terminal as it arrives, so long builds show progress. Pass cwd to choose the working directory instead of a leading \"cd\", whose effect ends with the call. Verify parent dirs with LS before creating files. Quote paths with spaces. Timeout: 120s default, 600s max; on timeout the command and everything it spawned are killed. AVOID: find/grep (use Grep/Glob), cat/head/tail/ls (use Read/LS). Chain with && (avoid ;, which does not chain in Windows cmd), no newlines. Prefer absolute paths over cd.\n\nGit commits: 1) Run parallel: git status/diff/log 2) Draft message (why not what), check secrets 3) Add files, commit using repeated -m flags for multi-line messages (e.g. git commit -m \"summary\" -m \"details\"), check status 4) Retry once if pre-commit modifies files. Never update config or use -i flag. Don't push unless requested.\n\nPull requests: 1) Parallel: git status/diff, check remote, git log 2) Analyze all commits, draft summary 3) Create branch, push, gh pr create with the body passed via --body-file (a temp file), since heredocs don't work in cmd.\n\nGitHub: Use gh for issues/PRs/checks/releases.")]
    pub(crate) async fn run_command(&self, params: Parameters<RunCommandParams>) -> Result<CallToolResult, McpError> {
        let command = params.0.command.clone();
        let timeout_secs = params.0.timeout.unwrap_or(120).min(600);

        #[cfg(target_os = "windows")]
        if chains_with_semicolon(&command) {
            return Ok(CallToolResult::error(vec![ContentBlock::text(SEMICOLON_REFUSAL)]));
        }

        use std::time::Duration;

        let mut builder = shell_command(&command);
        builder
            // No stdin, and on Windows that is what keeps the terminal working at all.
            //
            // Measured in a real console: cmd.exe sets the console input mode to its own
            // idea of one about ten milliseconds after it starts — ENABLE_PROCESSED_INPUT,
            // ENABLE_LINE_INPUT and ENABLE_ECHO_INPUT all back on — and never puts back what
            // it found. So raw mode was off from the first run_command onward while VT input
            // and the terminal's mouse reporting were both still on, and the next wheel
            // report was read a line at a time and echoed by the console into the input box
            // as `^[[<64;65;59M` with nothing reaching haico until Enter. It only does that
            // when its stdin is the console handle: the same command with no stdin leaves
            // the mode untouched, and ordinary programs (git, cargo, rustc, taskkill) never
            // touch it at all, which is what pins this on the shell rather than on them.
            //
            // Nothing is lost. A child sharing the console keyboard was already competing
            // with `Keys` for keystrokes, so a command that prompts could never have been
            // answered anyway; now it reads EOF and fails at once instead of waiting on a
            // terminal it cannot have. `run_background` has taken the same line all along.
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Make the child a process-group leader so a timeout can signal the whole group,
        // not just the shell. Windows gets the same effect from taskkill /T instead.
        #[cfg(not(target_os = "windows"))]
        builder.process_group(0);

        // An explicit directory beats a leading "cd", whose effect ends with the call.
        if let Some(cwd) = params.0.cwd.as_deref().filter(|c| !c.trim().is_empty()) {
            let dir = PathBuf::from(cwd);
            if !dir.is_dir() {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Working directory not found: {}",
                    cwd
                ))]));
            }
            builder.current_dir(&dir);
        }

        let mut child = builder.spawn().map_err(|e| {
            McpError::internal_error(format!("Failed to spawn command: {}", e), None)
        })?;

        // Both pipes are drained concurrently with the wait. Reading them only after the
        // process exits can deadlock: a command that fills the pipe buffer blocks forever
        // waiting for someone to read it.
        let echo = self.interactive;
        let stdout_buf = Arc::new(Mutex::new(OutputBuffer::new(usize::MAX)));
        let stderr_buf = Arc::new(Mutex::new(OutputBuffer::new(usize::MAX)));
        let stdout_task = child
            .stdout
            .take()
            .map(|pipe| tokio::spawn(pump_output(pipe, echo, stdout_buf.clone())));
        let stderr_task = child
            .stderr
            .take()
            .map(|pipe| tokio::spawn(pump_output(pipe, echo, stderr_buf.clone())));

        let status = match tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await
        {
            Ok(Ok(status)) => Some(status),
            Ok(Err(e)) => {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Failed to execute command: {}",
                    e
                ))]));
            }
            Err(_) => {
                // Kill it, along with anything it spawned. The previous code dropped the
                // timed-out child still running, so a hung command outlived the tool call
                // that started it.
                kill_process_tree(&mut child).await;
                None
            }
        };

        // Whatever the command managed to produce is worth returning, timeout or not.
        //
        // The pumps normally finish the moment the pipes close, so the grace period is only
        // ever spent when something still holds a pipe open — a killed command's surviving
        // grandchild, or a spawned background process. Abandoning them there is the point:
        // the shared buffers still hold everything read up to that moment.
        let drain = |task: Option<tokio::task::JoinHandle<()>>| async move {
            let Some(mut handle) = task else { return };
            if tokio::time::timeout(Duration::from_secs(2), &mut handle)
                .await
                .is_err()
            {
                handle.abort();
            }
        };
        drain(stdout_task).await;
        drain(stderr_task).await;

        let stdout_bytes = stdout_buf.lock().unwrap().bytes().to_vec();
        let stderr_bytes = stderr_buf.lock().unwrap().bytes().to_vec();

        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let stderr = String::from_utf8_lossy(&stderr_bytes);

        let mut result_text = String::new();
        if !stdout.is_empty() {
            result_text.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result_text.is_empty() {
                result_text.push_str("\n--- stderr ---\n");
            }
            result_text.push_str(&stderr);
        }

        let Some(status) = status else {
            let detail = if result_text.is_empty() {
                String::new()
            } else {
                format!(
                    "\n\nOutput before it was killed:\n{}",
                    truncate_middle(&result_text, MAX_OUTPUT)
                )
            };
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Command timed out after {} seconds and was killed.{}",
                timeout_secs, detail
            ))]));
        };

        if result_text.is_empty() {
            result_text = format!(
                "Command completed successfully (exit code: {})",
                status.code().unwrap_or(0)
            );
        }

        // Cut in the middle, not at the end: a build says what went wrong last.
        if result_text.len() > MAX_OUTPUT {
            result_text = truncate_middle(&result_text, MAX_OUTPUT);
        }

        if status.success() {
            Ok(CallToolResult::success(vec![ContentBlock::text(result_text)]))
        } else {
            Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Command failed with exit code: {}\n\n{}",
                status.code().unwrap_or(-1),
                result_text
            ))]))
        }
    }

    #[tool(
        description = "Start a long-running command that must keep running while you do other things — a dev server, a file watcher, a database. Returns a job id immediately instead of blocking. Pass wait_for with a string the command prints once it is ready (e.g. 'Listening on') so you do not move on before it can be reached; without it you get about a second and a half of output, enough to catch an immediate failure such as a port already in use. Read later output with background_output and stop it with background_stop. If wait_for times out but the job is still running, the program is probably buffering its output rather than failing — probe it directly instead of waiting again. All jobs are killed when the session ends. Use run_command for anything that finishes on its own."
    )]
    pub(crate) async fn run_background(
        &self,
        params: Parameters<RunBackgroundParams>,
    ) -> Result<CallToolResult, McpError> {
        use std::time::{Duration, Instant};

        let command = params.0.command.clone();
        let timeout_secs = params.0.timeout.unwrap_or(30).min(300);

        #[cfg(target_os = "windows")]
        if chains_with_semicolon(&command) {
            return Ok(CallToolResult::error(vec![ContentBlock::text(SEMICOLON_REFUSAL)]));
        }

        let mut builder = shell_command(&command);
        builder
            // No stdin: a background process that blocks on a prompt would hang unseen.
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        #[cfg(not(target_os = "windows"))]
        builder.process_group(0);

        if let Some(cwd) = params.0.cwd.as_deref().filter(|c| !c.trim().is_empty()) {
            let dir = PathBuf::from(cwd);
            if !dir.is_dir() {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Working directory not found: {}",
                    cwd
                ))]));
            }
            builder.current_dir(&dir);
        }

        let mut child = builder.spawn().map_err(|e| {
            McpError::internal_error(format!("Failed to start command: {}", e), None)
        })?;
        let pid = child.id();

        // stdout and stderr share one buffer so the output reads in the order a terminal
        // would have shown it. Never echoed: a server's request log interleaved with the
        // assistant's streaming reply is unreadable.
        let output = Arc::new(Mutex::new(OutputBuffer::new(BACKGROUND_OUTPUT_CAP)));
        if let Some(pipe) = child.stdout.take() {
            tokio::spawn(pump_output(pipe, false, output.clone()));
        }
        if let Some(pipe) = child.stderr.take() {
            tokio::spawn(pump_output(pipe, false, output.clone()));
        }

        // A waiter owns the Child so the exit status is recorded the moment it happens.
        // Without it a server that crashed would just look silent.
        let exit = Arc::new(Mutex::new(None));
        let exit_writer = exit.clone();
        tokio::spawn(async move {
            if let Ok(status) = child.wait().await {
                *exit_writer.lock().unwrap() = Some(status.code().unwrap_or(-1));
            }
        });

        let started = Instant::now();
        let wait_for = params.0.wait_for.clone().filter(|w| !w.trim().is_empty());
        let deadline = match &wait_for {
            Some(_) => started + Duration::from_secs(timeout_secs),
            // No pattern given: just long enough to catch an immediate failure.
            None => started + Duration::from_millis(1500),
        };
        let needle = wait_for.as_ref().map(|w| w.to_lowercase());

        let mut matched = false;
        let mut exited = false;
        loop {
            if exit.lock().unwrap().is_some() {
                exited = true;
                break;
            }
            if let Some(needle) = &needle {
                let seen =
                    String::from_utf8_lossy(output.lock().unwrap().bytes()).to_lowercase();
                if seen.contains(needle) {
                    matched = true;
                    break;
                }
            }
            if Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let (captured, seen_bytes) = {
            let buffer = output.lock().unwrap();
            (
                String::from_utf8_lossy(buffer.bytes()).to_string(),
                buffer.total(),
            )
        };

        let id = {
            let mut registry = self.background.lock().unwrap();
            registry.next_id += 1;
            let id = registry.next_id;
            registry.jobs.insert(
                id,
                BackgroundJob {
                    command: command.clone(),
                    pid,
                    output: output.clone(),
                    // What is shown below counts as read, so the next call returns only
                    // what arrived after it.
                    cursor: seen_bytes,
                    started,
                    exit: exit.clone(),
                    stopped: false,
                },
            );
            id
        };

        let elapsed = started.elapsed().as_secs_f32();
        let headline = if exited {
            let code = exit.lock().unwrap().unwrap_or(-1);
            format!(
                "Job {} exited immediately with code {} after {:.1}s — it is NOT running.",
                id, code, elapsed
            )
        } else if matched {
            format!(
                "Job {} started; matched {:?} after {:.1}s.",
                id,
                wait_for.as_deref().unwrap_or(""),
                elapsed
            )
        } else if wait_for.is_some() {
            format!(
                "Job {} started, but {:?} did not appear within {}s. It is still running — \
                 this often just means the program buffers its output when it is not writing \
                 to a terminal, so the banner is stuck in its buffer rather than missing. \
                 Probe it directly (curl, a port check) instead of waiting longer, or \
                 restart it unbuffered (python -u, stdbuf -oL). background_output({}) shows \
                 whatever has arrived.",
                id,
                wait_for.as_deref().unwrap_or(""),
                timeout_secs,
                id
            )
        } else {
            format!("Job {} started and is running.", id)
        };

        let body = if captured.trim().is_empty() {
            "(no output yet)".to_string()
        } else {
            truncate_at_boundary(&captured, 8000).to_string()
        };

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "{}\nCommand: {}\n\n{}",
            headline, command, body
        ))]))
    }

    #[tool(
        description = "Read what a background job has printed since you last looked — use it to check whether a server came up, or to see why it failed. With no id, lists every job with its status. Returns only new output unless from_start is set, so calling it repeatedly does not re-read the same log."
    )]
    pub(crate) async fn background_output(
        &self,
        params: Parameters<BackgroundOutputParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut registry = self.background.lock().unwrap();

        let Some(id) = params.0.id else {
            if registry.jobs.is_empty() {
                return Ok(CallToolResult::success(vec![ContentBlock::text(
                    "No background jobs.".to_string(),
                )]));
            }
            let listing = registry
                .jobs
                .iter()
                .map(|(id, job)| format!("  {}  [{}]  {}", id, job.status(), job.command))
                .collect::<Vec<_>>()
                .join("\n");
            return Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "Background jobs:\n{}",
                listing
            ))]));
        };

        let Some(job) = registry.jobs.get_mut(&id) else {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "No background job with id {}.",
                id
            ))]));
        };

        let status = job.status();
        let command = job.command.clone();
        let (text, dropped) = {
            let buffer = job.output.lock().unwrap();
            let from = if params.0.from_start {
                buffer.dropped()
            } else {
                job.cursor
            };
            let text = String::from_utf8_lossy(buffer.since(from)).to_string();
            let dropped = buffer.dropped();
            let total = buffer.total();
            drop(buffer);
            job.cursor = total;
            (text, dropped)
        };

        let mut body = if text.trim().is_empty() {
            "(nothing new)".to_string()
        } else {
            truncate_at_boundary(&text, 20000).to_string()
        };
        if params.0.from_start && dropped > 0 {
            body = format!("[{} earlier bytes dropped]\n{}", dropped, body);
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Job {} [{}]: {}\n\n{}",
            id, status, command, body
        ))]))
    }

    #[tool(
        description = "Stop a background job and everything it started. With no id, stops all of them. Do this as soon as a server is no longer needed, rather than leaving it holding a port."
    )]
    pub(crate) async fn background_stop(
        &self,
        params: Parameters<BackgroundStopParams>,
    ) -> Result<CallToolResult, McpError> {
        // pids are collected under the lock and killed outside it: killing awaits, and a
        // std Mutex must never be held across an await point.
        let targets: Vec<(u32, Option<u32>, String)> = {
            let registry = self.background.lock().unwrap();
            registry
                .jobs
                .iter()
                .filter(|(id, job)| {
                    params.0.id.is_none_or(|wanted| wanted == **id)
                        && job.exit.lock().unwrap().is_none()
                        && !job.stopped
                })
                .map(|(id, job)| (*id, job.pid, job.command.clone()))
                .collect()
        };

        if targets.is_empty() {
            let message = match params.0.id {
                Some(id) => format!("Job {} is not running.", id),
                None => "No running background jobs.".to_string(),
            };
            return Ok(CallToolResult::success(vec![ContentBlock::text(message)]));
        }

        let mut stopped = Vec::new();
        for (id, pid, command) in targets {
            if let Some(pid) = pid {
                kill_tree_by_pid(pid).await;
            }
            if let Some(job) = self.background.lock().unwrap().jobs.get_mut(&id) {
                job.stopped = true;
            }
            stopped.push(format!("  {}  {}", id, command));
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Stopped {} job(s):\n{}",
            stopped.len(),
            stopped.join("\n")
        ))]))
    }
}

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

    async fn run(
        server: &crate::MyMcpServer,
        command: &str,
        timeout: Option<u64>,
        cwd: Option<String>,
    ) -> (bool, String) {
        let result = server
            .run_command(rmcp::handler::server::wrapper::Parameters(
                crate::RunCommandParams {
                    command: command.to_string(),
                    timeout,
                    cwd,
                },
            ))
            .await
            .unwrap();
        (!result.is_error.unwrap_or(false), result_text(&result))
    }


    #[test]
    fn output_buffer_drops_the_oldest_and_keeps_offsets_meaningful() {
        let mut buffer = OutputBuffer::new(10);
        buffer.push(b"abcde");
        assert_eq!(buffer.bytes(), b"abcde");
        assert_eq!(buffer.total(), 5);
        assert_eq!(buffer.dropped(), 0);
        // A reader that has consumed 2 bytes sees the rest.
        assert_eq!(buffer.since(2), b"cde");

        // Past the cap, the oldest go and `dropped` accounts for them, so an absolute
        // offset taken earlier still resolves to the right place.
        buffer.push(b"fghijklmn");
        assert_eq!(buffer.bytes(), b"efghijklmn");
        assert_eq!(buffer.total(), 14);
        assert_eq!(buffer.dropped(), 4);
        assert_eq!(buffer.since(10), b"klmn");
        // An offset pointing at bytes that are gone yields what survives, not a panic.
        assert_eq!(buffer.since(0), b"efghijklmn");
        // An offset past the end is empty rather than out of bounds.
        assert_eq!(buffer.since(999), b"");
    }

    async fn background(
        server: &super::super::MyMcpServer,
        command: &str,
        wait_for: Option<&str>,
        timeout: Option<u64>,
    ) -> String {
        let result = server
            .run_background(rmcp::handler::server::wrapper::Parameters(
                super::RunBackgroundParams {
                    command: command.to_string(),
                    cwd: None,
                    wait_for: wait_for.map(|w| w.to_string()),
                    timeout,
                },
            ))
            .await
            .unwrap();
        result_text(&result)
    }

    async fn bg_output(server: &super::super::MyMcpServer, id: Option<u32>) -> String {
        let result = server
            .background_output(rmcp::handler::server::wrapper::Parameters(
                super::BackgroundOutputParams { id, from_start: false },
            ))
            .await
            .unwrap();
        result_text(&result)
    }

    #[tokio::test]
    async fn background_waits_for_readiness_then_reports_only_new_output() {
        let server = test_server();

        // Prints a marker, then keeps running. wait_for must return as soon as the marker
        // appears rather than burning the whole timeout.
        let command = if cfg!(windows) {
            "echo starting && echo LISTENING ON 8080 && ping -n 20 127.0.0.1 > NUL"
        } else {
            "echo starting; echo LISTENING ON 8080; sleep 20"
        };

        let started = std::time::Instant::now();
        let out = background(&server, command, Some("listening on"), Some(20)).await;
        let elapsed = started.elapsed();

        assert!(out.contains("matched"), "got: {}", out);
        assert!(out.contains("LISTENING ON 8080"), "got: {}", out);
        // Matched case-insensitively and returned promptly, not after the 20s timeout.
        assert!(elapsed < std::time::Duration::from_secs(10), "took {:?}", elapsed);

        // Everything already shown counts as read, so a follow-up returns nothing new.
        let again = bg_output(&server, Some(1)).await;
        assert!(again.contains("nothing new"), "got: {}", again);
        assert!(again.contains("running"), "got: {}", again);

        // Listing shows it without an id.
        let listing = bg_output(&server, None).await;
        assert!(listing.contains("Background jobs:"), "got: {}", listing);

        let stopped = server.stop_all_background().await;
        assert_eq!(stopped, 1);
    }

    #[tokio::test]
    async fn background_reports_a_command_that_dies_at_once() {
        let server = test_server();

        // The port-already-in-use case: it must come back as "NOT running" with the output,
        // not as a cheerful job id the model then curls against.
        let out = background(&server, "echo boom && exit 7", Some("never appears"), Some(10)).await;
        assert!(out.contains("exited immediately"), "got: {}", out);
        assert!(out.contains("code 7"), "got: {}", out);
        assert!(out.contains("NOT running"), "got: {}", out);
        assert!(out.contains("boom"), "got: {}", out);

        // Nothing to stop, and saying so beats pretending it worked.
        let result = server
            .background_stop(rmcp::handler::server::wrapper::Parameters(
                super::BackgroundStopParams { id: Some(1) },
            ))
            .await
            .unwrap();
        assert!(result_text(&result).contains("not running"));

        assert_eq!(server.stop_all_background().await, 0);
    }

    #[tokio::test]
    async fn background_stop_kills_the_whole_tree() {
        let server = test_server();

        let sleeper = if cfg!(windows) {
            "ping -n 30 127.0.0.1 > NUL"
        } else {
            "sleep 30"
        };
        background(&server, sleeper, None, None).await;

        let started = std::time::Instant::now();
        let stopped = server.stop_all_background().await;
        assert_eq!(stopped, 1);
        // Killing only the shell would leave ping alive holding the pipe; the pump task
        // would then outlive the session, which is the whole point of the tree kill.
        assert!(
            started.elapsed() < std::time::Duration::from_secs(8),
            "stopping took {:?}",
            started.elapsed()
        );

        // A second pass has nothing left to do.
        assert_eq!(server.stop_all_background().await, 0);
        let listing = bg_output(&server, None).await;
        assert!(listing.contains("stopped"), "got: {}", listing);
    }

    #[tokio::test]
    async fn run_command_reports_output_status_and_directory() {
        let server = test_server();

        let (ok, out) = run(&server, "echo hello", None, None).await;
        assert!(ok, "got: {}", out);
        assert!(out.contains("hello"), "got: {}", out);

        // A failing command is surfaced as an error with its exit code, not as success.
        let (ok, out) = run(&server, "exit 3", None, None).await;
        assert!(!ok);
        assert!(out.contains("exit code: 3"), "got: {}", out);

        // cwd is honored, so the model does not have to rely on a leading "cd".
        let dir = scratch("marker.txt", "x").parent().unwrap().to_path_buf();
        let (ok, out) = run(
            &server,
            if cfg!(windows) { "cd" } else { "pwd" },
            None,
            Some(dir.display().to_string()),
        )
        .await;
        assert!(ok, "got: {}", out);
        let reported = out.to_lowercase().replace('\\', "/");
        let expected = dir.display().to_string().to_lowercase().replace('\\', "/");
        // The temp dir may be reported via a different prefix form; compare the tail.
        let tail = expected.rsplit('/').next().unwrap();
        assert!(reported.contains(tail), "got: {} (wanted {})", out, expected);

        // A directory that does not exist is reported rather than silently ignored.
        let (ok, out) = run(&server, "echo x", None, Some("no/such/dir".to_string())).await;
        assert!(!ok);
        assert!(out.contains("Working directory not found"), "got: {}", out);
    }


    #[tokio::test]
    async fn run_command_delivers_quoted_arguments_to_the_child_intact() {
        let server = test_server();

        // A quoted argument holding a space must reach the child as one argument. Windows
        // hands a process a single command line rather than an argv, and passing the command
        // through `Command::arg` re-escaped every quote as `\"` on the way — which cmd.exe
        // has no rule for — so `-H "Host: example.com"` arrived as `"Host:` plus a stray
        // `example.com"`, and the header was silently never sent.
        //
        // This has to be checked against a real child process: a shell builtin is handed the
        // text before any argv is built, so it would pass even while every external command
        // was broken.
        let haystack = scratch("quoted-args-haystack.txt", "alpha beta gamma\n");
        let search = if cfg!(windows) {
            format!("findstr /c:\"beta gamma\" \"{}\"", haystack.display())
        } else {
            format!("grep -F \"beta gamma\" \"{}\"", haystack.display())
        };
        let (ok, out) = run(&server, &search, None, None).await;
        assert!(ok, "got: {}", out);
        assert!(out.contains("alpha beta gamma"), "got: {}", out);

        // The tool description tells the model to quote paths with spaces, so that has to
        // work; it did not.
        let spaced = scratch("name with spaces.txt", "payload-token\n");
        let show = if cfg!(windows) { "type" } else { "cat" };
        let (ok, out) = run(&server, &format!("{} \"{}\"", show, spaced.display()), None, None).await;
        assert!(ok, "got: {}", out);
        assert!(out.contains("payload-token"), "got: {}", out);

        // Nothing may inject backslashes into the command on the way to the shell. cmd's echo
        // prints its argument verbatim, so this reports exactly what cmd received: it used to
        // be `{\"key\": \"value\"}`. Windows-only — sh consumes the quotes itself.
        if cfg!(windows) {
            let (ok, out) = run(&server, r#"echo {"key": "value"}"#, None, None).await;
            assert!(ok, "got: {}", out);
            assert!(out.contains(r#"{"key": "value"}"#), "got: {}", out);
            assert!(!out.contains('\\'), "backslash injected: {}", out);
        }

        // A command whose program path is itself quoted is the case that plain `/C` cuts in
        // half, by stripping the first and last quote of the whole line rather than a pair.
        // `/S` is what makes the strip apply only to the quotes haico added.
        let quoted_program = if cfg!(windows) {
            format!("\"{}\\System32\\find.exe\" \"beta\" \"{}\"",
                std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".into()),
                haystack.display())
        } else {
            format!("\"/bin/grep\" -F \"beta gamma\" \"{}\"", haystack.display())
        };
        let (ok, out) = run(&server, &quoted_program, None, None).await;
        assert!(ok, "got: {}", out);
        assert!(out.contains("alpha beta gamma"), "got: {}", out);
    }


    #[tokio::test]
    async fn run_command_kills_a_command_that_times_out() {
        let server = test_server();

        // Sleeps well past the timeout. The old code dropped the timed-out child while it
        // was still running, so the process outlived the tool call that started it.
        let sleeper = if cfg!(windows) {
            "ping -n 30 127.0.0.1 > NUL"
        } else {
            "sleep 30"
        };

        let started = std::time::Instant::now();
        let (ok, out) = run(&server, sleeper, Some(2), None).await;
        let elapsed = started.elapsed();

        assert!(!ok);
        assert!(out.contains("timed out"), "got: {}", out);
        assert!(out.contains("killed"), "got: {}", out);
        // Tight on purpose. Killing only the shell leaves the grandchild holding the output
        // pipe, and draining it then blocks until the command finishes on its own — which
        // showed up as this test taking ~30s instead of ~2s. The timing is the evidence that
        // the whole process tree really went away.
        assert!(
            elapsed < std::time::Duration::from_secs(8),
            "took {:?}: the command outlived its timeout, so the tree kill did not work",
            elapsed
        );
    }


    #[tokio::test]
    async fn run_command_gives_the_child_no_stdin_of_its_own() {
        let server = test_server();

        // Reads stdin to the end and prints what it got, which is nothing.
        let reader = if cfg!(windows) { "sort" } else { "cat" };

        let started = std::time::Instant::now();
        let (ok, out) = run(&server, reader, Some(5), None).await;
        let elapsed = started.elapsed();

        assert!(ok, "got: {}", out);
        // The timing is the assertion. With the console's own stdin handed to the child, this
        // waits for a line that only the user can type — and on Windows the shell has by then
        // put the console back into cooked mode, which is the bug this pins down: the input
        // box fills with echoed mouse reports and the keyboard stops reaching haico.
        assert!(
            elapsed < std::time::Duration::from_secs(4),
            "took {:?}: the child was waiting on a terminal it should not have",
            elapsed
        );
    }

    #[tokio::test]
    async fn run_command_survives_output_larger_than_a_pipe_buffer() {
        let server = test_server();

        // Reading the pipes only after the process exits deadlocks once the command fills
        // the OS pipe buffer (~64KB). Draining concurrently is what makes this terminate.
        let noisy = if cfg!(windows) {
            "for /L %i in (1,1,2000) do @echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        } else {
            "for i in $(seq 1 2000); do echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; done"
        };

        let (ok, out) = run(&server, noisy, Some(60), None).await;
        assert!(ok, "got: {}", truncate_at_boundary(&out, 300));
        // It terminates at all, and returns the 30k cap's worth rather than deadlocking or
        // coming back empty.
        assert!(out.contains("lines omitted"), "got {} bytes", out.len());
        assert!(out.len() > 29_000, "only got {} bytes", out.len());
        // The cut is in the middle, so the command's own last line is still the last line here.
        assert!(out.trim_end().ends_with('a'), "the tail was cut instead of the middle");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn semicolon_is_a_separator_only_where_cmd_would_have_run_two_commands() {
        use crate::tools::shell::chains_with_semicolon;

        assert!(chains_with_semicolon("curl URL ; type hdr.txt"));
        assert!(chains_with_semicolon("curl URL; type hdr.txt"));
        assert!(chains_with_semicolon("cargo build ;"));

        // Ordinary cmd lines that happen to contain a semicolon, and must still run.
        assert!(!chains_with_semicolon("set PATH=C:\\a;C:\\b"));
        assert!(!chains_with_semicolon("for %i in (a;b) do @echo %i"));
        assert!(!chains_with_semicolon("cargo build"));

        // Quoted text is an argument, not a chain — but only double quotes quote in cmd, so a
        // semicolon between single quotes is still a separator there.
        assert!(!chains_with_semicolon("echo \"a ; b\""));
        assert!(chains_with_semicolon("echo 'a ; b'"));
    }

    #[cfg(target_os = "windows")]
    #[tokio::test]
    async fn run_command_refuses_a_semicolon_chain_and_says_what_to_use_instead() {
        let server = test_server();

        // Without the refusal cmd hands "; echo second" to the first echo as arguments, so the
        // command "succeeds" having run only half of what was asked.
        let (ok, out) = run(&server, "echo first ; echo second", Some(30), None).await;
        assert!(!ok, "the chain should have been refused, got: {}", out);
        assert!(out.contains("&&"), "the refusal must name the fix, got: {}", out);
    }

}
