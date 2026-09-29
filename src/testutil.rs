//! Fixtures shared by more than one module's tests.
#![allow(unused_imports)]

use crate::config::*;
use crate::llm::*;
use crate::tools::*;
use serde_json::json;

pub(crate) async fn serve_once(body: &'static str, content_type: &'static str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();

        // Consume the request head so the client is never left blocked on a write.
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            match socket.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => request.extend_from_slice(&buf[..n]),
            }
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n{}",
            content_type,
            body.len(),
            body
        );
        let _ = socket.write_all(response.as_bytes()).await;
        let _ = socket.flush().await;
    });

    format!("http://{}", addr)
    }

// A server that answers by path, for as long as the test needs it rather than once.
//
// Probing for a loaded model asks two different endpoints and stops at whichever answers,
// so which one replies — and which 404s — is the whole thing being tested.
pub(crate) async fn serve_routes(routes: &'static [(&'static str, &'static str)]) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0u8; 4096];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => request.extend_from_slice(&buf[..n]),
                    }
                }

                let head = String::from_utf8_lossy(&request);
                let path = head.split_whitespace().nth(1).unwrap_or("");
                // Closed after every reply rather than kept alive: a pooled connection to
                // a socket this handler has already dropped is a retry the test does not
                // need to be flaky about.
                let response = match routes.iter().find(|(route, _)| *route == path) {
                    Some((_, body)) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    ),
                    None => "HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n".to_string(),
                };
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
            });
        }
    });

    format!("http://{}", addr)
}

    // Like serve_once, but hands back what the client actually sent.

pub(crate) async fn serve_once_capturing(
    body: &'static str,
    content_type: &'static str,
    ) -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = captured.clone();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();

        // Read the head, then the body named by Content-Length.
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        let head_end = loop {
            match socket.read(&mut buf).await {
                Ok(0) | Err(_) => break None,
                Ok(n) => {
                    request.extend_from_slice(&buf[..n]);
                    if let Some(pos) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        break Some(pos + 4);
                    }
                }
            }
        };

        if let Some(head_end) = head_end {
            let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
            let length: usize = head
                .split("content-length:")
                .nth(1)
                .and_then(|rest| rest.split(['\r', '\n']).next())
                .and_then(|value| value.trim().parse().ok())
                .unwrap_or(0);

            while request.len() < head_end + length {
                match socket.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => request.extend_from_slice(&buf[..n]),
                }
            }
            *sink.lock().unwrap() =
                String::from_utf8_lossy(&request[head_end..]).to_string();
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n{}",
            content_type,
            body.len(),
            body
        );
        let _ = socket.write_all(response.as_bytes()).await;
        let _ = socket.flush().await;
    });

    (format!("http://{}", addr), captured)
    }

// A chat-completions server that answers a different body each time, for testing a loop
// rather than a single call.
//
// The last body repeats for every request after it, which is what lets a test say "keep
// asking for a tool call forever" in one entry and then check that something else stopped
// the loop. The count of requests served comes back alongside the URL, because for a loop
// the interesting assertion is usually how many rounds it took rather than what it said.

pub(crate) async fn serve_sequence(
    bodies: &'static [&'static str],
    ) -> (String, std::sync::Arc<std::sync::Mutex<usize>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = std::sync::Arc::new(std::sync::Mutex::new(0usize));
    let count = served.clone();

    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let count = count.clone();
            tokio::spawn(async move {
                // Head, then the body named by Content-Length: a client left blocked on a
                // write it cannot finish is a test that hangs instead of failing.
                let mut request = Vec::new();
                let mut buf = [0u8; 4096];
                let head_end = loop {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break None,
                        Ok(n) => {
                            request.extend_from_slice(&buf[..n]);
                            if let Some(pos) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                                break Some(pos + 4);
                            }
                        }
                    }
                };

                if let Some(head_end) = head_end {
                    let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
                    let length: usize = head
                        .split("content-length:")
                        .nth(1)
                        .and_then(|rest| rest.split(['\r', '\n']).next())
                        .and_then(|value| value.trim().parse().ok())
                        .unwrap_or(0);
                    while request.len() < head_end + length {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => request.extend_from_slice(&buf[..n]),
                        }
                    }
                }

                let index = {
                    let mut served = count.lock().unwrap();
                    let index = *served;
                    *served += 1;
                    index
                };
                let body = bodies[index.min(bodies.len() - 1)];

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
            });
        }
    });

    (format!("http://{}", addr), served)
}


pub(crate) fn test_server() -> crate::MyMcpServer {
    crate::MyMcpServer::new(
        crate::LMStudioClient::new("http://127.0.0.1:1".to_string(), String::new(), None),
        false,
        None,
        false,
    )
    }

// A directory under the OS temp folder that takes itself away again.
//
// The fixtures used to leave theirs behind. Each one is named for the process and the
// thread, so nothing collided and every run looked clean — while the temp folder
// collected one directory per test per run, thousands of them before anyone noticed.
//
// Removed on the way in as well as out: a run killed halfway through (Ctrl+C, a
// harness panic) leaves the directory, and the next run to land on the same name would
// otherwise find a fixture it did not plant.
pub(crate) struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    // `label` only has to tell the fixtures apart — the process and thread are already
    // in the name, which is what keeps two tests running at once out of each other's way.
    pub(crate) fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "haico-{}-{}-{:?}",
            label,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub(crate) fn join(&self, tail: &str) -> std::path::PathBuf {
        self.path.join(tail)
    }

    // A subdirectory, created, for a fixture that needs several.
    pub(crate) fn subdir(&self, name: &str) -> std::path::PathBuf {
        let path = self.path.join(name);
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

thread_local! {
    // The scratch directory belongs to the running test, and a thread-local is the only
    // place a guard for it can live: `scratch` hands back a plain path to thirty-odd
    // call sites, none of which is holding anything, and a test binary has no exit hook
    // to sweep up at the end. libtest gives each test its own thread, so the directory
    // is made when the first scratch file is asked for and removed when that thread
    // ends — including when the test panicked, which is the one case where the files
    // would have been worth keeping. Measured with `--test-threads=1` as well, where the
    // tests are serialized but still each get a thread; a harness that ever stopped doing
    // that would leave the directory behind, which is what every run did before this.
    static SCRATCH_DIR: TempDir = TempDir::new("test");
}

    // A scratch file with a unique name, in a directory that goes with the test.

pub(crate) fn scratch(name: &str, contents: &str) -> std::path::PathBuf {
    SCRATCH_DIR.with(|dir| {
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    })
    }


pub(crate) fn result_text(result: &rmcp::model::CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
    }


pub(crate) async fn read_file(server: &crate::MyMcpServer, path: &std::path::Path) {
    server
        .read(rmcp::handler::server::wrapper::Parameters(crate::ReadParams {
            file_path: path.display().to_string(),
            offset: None,
            limit: None,
        }))
        .await
        .unwrap();
    }


pub(crate) async fn try_edit(
    server: &crate::MyMcpServer,
    path: &std::path::Path,
    old: &str,
    new: &str,
    replace_all: bool,
    ) -> (bool, String) {
    let result = server
        .edit(rmcp::handler::server::wrapper::Parameters(crate::EditParams {
            file_path: path.display().to_string(),
            old_string: old.to_string(),
            new_string: new.to_string(),
            replace_all,
        }))
        .await
        .unwrap();
    let failed = result.is_error.unwrap_or(false);
    (!failed, result_text(&result))
    }


pub(crate) async fn try_multi_edit(
    server: &crate::MyMcpServer,
    path: &std::path::Path,
    edits: Vec<(&str, &str, bool)>,
    ) -> (bool, String) {
    let edits = edits
        .into_iter()
        .map(|(old_string, new_string, replace_all)| crate::EditOperation {
            old_string: old_string.to_string(),
            new_string: new_string.to_string(),
            replace_all,
        })
        .collect();
    let result = server
        .multi_edit(rmcp::handler::server::wrapper::Parameters(
            crate::MultiEditParams {
                file_path: path.display().to_string(),
                edits,
            },
        ))
        .await
        .unwrap();
    (!result.is_error.unwrap_or(false), result_text(&result))
    }


pub(crate) fn settings_of(
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    min_p: Option<f64>,
    repeat_penalty: Option<f64>,
    ) -> crate::ResolvedSettings {
    // A fn, not a closure: closures cannot be generic, and these are f64 and u64.
    fn at<T>(value: Option<T>) -> crate::Setting<T> {
        crate::Setting { value, source: "test" }
    }
    crate::ResolvedSettings {
        context_tokens: 32768,
        context_source: "test",
        temperature: at(temperature),
        max_tokens: at(max_tokens),
        top_p: at(top_p),
        top_k: at(top_k),
        min_p: at(min_p),
        repeat_penalty: at(repeat_penalty),
        // Not a seventh positional argument: that would be paid for by every call site to
        // serve the one test that cares, which sets the field on the result instead.
        reasoning_effort: at(None),
        sub_agent_effort: None,
        use_system_prompt: true,
        reasoning_setting: at(None),
        reasoning_threshold_setting: at(None),
    }
    }


/// A `Config` aimed at `url`, with everything else at its quietest: no config
/// file, no search backend, no resume, and the inline renderer — a test owns no
/// terminal to take the alternate screen of.
///
/// Spelled out field by field rather than derived from `Default`: the real
/// defaults are decided by `parse_config`, and a `Default` impl on `Config`
/// would be a second answer to that question, findable from production code and
/// wrong there.
pub(crate) fn test_config(url: &str) -> crate::Config {
    crate::Config {
        llm_url: url.to_string(),
        token: None,
        system_prompt: None,
        model: None,
        server: false,
        gui: false,
        url_supplied: true,
        searxng: None,
        context_arg: None,
        temperature_arg: None,
        max_tokens_arg: None,
        top_p_arg: None,
        top_k_arg: None,
        min_p_arg: None,
        repeat_penalty_arg: None,
        file: Default::default(),
        config_path: None,
        config_created: false,
        resume: false,
        prompt: None,
        trace: None,
        dir: None,
        tui: crate::Tui::Inline,
        mouse: Default::default(),
    }
}
