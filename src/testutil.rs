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


pub(crate) fn test_server() -> crate::MyMcpServer {
    crate::MyMcpServer::new(
        crate::LMStudioClient::new("http://127.0.0.1:1".to_string(), String::new(), None),
        false,
        None,
        false,
    )
    }

    // A scratch file with a unique name, cleaned up by the caller.

pub(crate) fn scratch(name: &str, contents: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "haico-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, contents).unwrap();
    path
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
        repeat_penalty: at(repeat_penalty),
        use_system_prompt: true,
    }
    }

