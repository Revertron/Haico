//! The OpenAI-compatible client, including streaming.

use crate::config::ResolvedSettings;
use crate::util::truncate_at_boundary;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
pub(crate) struct ModelsResponse {
    pub(crate) data: Vec<ModelInfo>,
}


#[derive(Debug, Deserialize)]
pub(crate) struct ModelInfo {
    pub(crate) id: String,
}

// Tools that only inspect local state, or query a fixed, benign endpoint. Prompting for
// these is pure friction: a single task easily makes dozens of such calls, and a user asked
// every time learns within minutes to answer "allow all" reflexively — including on
// run_command, which is exactly how a session ends up with an unrestricted shell.
//
// read/ls/grep/glob_search can reach outside the working directory. That is deliberate: a
// coding agent legitimately reads ~/.cargo/config.toml, and confining reads would break more
// than it protects. What they cannot do is change anything.

pub(crate) fn normalize_base_url(url: &str) -> String {
    let url = url.trim().trim_end_matches('/');
    url.strip_suffix("/v1")
        .unwrap_or(url)
        .trim_end_matches('/')
        .to_string()
}

// Truncate to at most `max_bytes`, never splitting a multi-byte character.
//
// `String` indexing is by byte offset, so `&s[..n]` panics outright when `n` lands inside
// a UTF-8 sequence. Every size limit in the codebase has to go through here; a single
// Cyrillic character or emoji in a long line, in command output, or in a command string
// would otherwise abort the process and take the whole conversation with it.

pub(crate) fn unauthorized_message(had_token: bool) -> String {
    if had_token {
        "LLM server rejected the request (401 Unauthorized): the configured token was not \
         accepted. Check --token / LM_API_TOKEN."
            .to_string()
    } else {
        "LLM server rejected the request (401 Unauthorized): it requires an API token, but \
         none was configured. Pass --token or set LM_API_TOKEN."
            .to_string()
    }
}

// LM Studio client

#[derive(Clone)]
pub(crate) struct LMStudioClient {
    pub(crate) client: reqwest::Client,
    pub(crate) base_url: String,
    // Model to request. An empty string means "no model configured": the "model" field is
    // then omitted from requests entirely, so the server answers with the model it already
    // has loaded.
    pub(crate) model: String,
    // Bearer token for servers with authentication enabled (LM Studio's "API token").
    // None means send no Authorization header, which is what a default local server wants.
    pub(crate) token: Option<String>,
    // Sampling, when configured. None means the field is left out of the request entirely,
    // so the server's own per-model settings apply — sending a made-up default here would
    // silently override what the user tuned in LM Studio's UI.
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<u64>,
    pub(crate) repeat_penalty: Option<f64>,
}


impl LMStudioClient {
    pub(crate) fn new(base_url: String, model: String, token: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: normalize_base_url(&base_url),
            model,
            token,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            repeat_penalty: None,
        }
    }

    // Apply the settings resolved for the model currently in use.
    pub(crate) fn apply_settings(&mut self, settings: &ResolvedSettings) {
        self.temperature = settings.temperature.value;
        self.max_tokens = settings.max_tokens.value;
        self.top_p = settings.top_p.value;
        self.top_k = settings.top_k.value;
        self.repeat_penalty = settings.repeat_penalty.value;
    }

    // Add the sampling fields that are actually configured, and only those.
    pub(crate) fn set_sampling(&self, body: &mut Value) {
        if let Some(temperature) = self.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(max_tokens) = self.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(top_p) = self.top_p {
            body["top_p"] = json!(top_p);
        }
        // Not OpenAI fields, but understood by llama.cpp, Ollama, LM Studio and vLLM.
        // Still only sent when explicitly configured, so a strict endpoint never sees them
        // unless the user asked for them.
        if let Some(top_k) = self.top_k {
            body["top_k"] = json!(top_k);
        }
        if let Some(repeat_penalty) = self.repeat_penalty {
            body["repeat_penalty"] = json!(repeat_penalty);
        }
    }

    // Attach the bearer token, when one is configured.
    pub(crate) fn authorize(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => req.bearer_auth(token),
            None => req,
        }
    }

    // Add the "model" field to a request body, unless no model is configured.
    pub(crate) fn set_model(&self, body: &mut Value) {
        if !self.model.is_empty() {
            body["model"] = json!(self.model);
        }
    }

    // POST a chat-completions request. An auth failure is reported as such, instead of
    // surfacing later as a confusing "failed to parse response" on the error JSON.
    pub(crate) async fn post_chat(&self, body: &Value) -> Result<Value> {
        let response = self
            .authorize(
                self.client
                    .post(format!("{}/v1/chat/completions", self.base_url)),
            )
            .json(body)
            .send()
            .await
            .context("Failed to send request to LM Studio")?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            bail!("{}", unauthorized_message(self.token.is_some()));
        }

        response
            .json::<Value>()
            .await
            .context("Failed to parse response")
    }

    // A single completion with no tools and no streaming, for asking the model something
    // that is not part of the conversation.
    //
    // Deliberately not sharing the streaming path: that one always sends a tools array,
    // and a request carrying tools invites a tool call back instead of the plain answer
    // wanted here. The user's sampling settings are skipped too — a safety review wants
    // the same answer every time, not whatever temperature suits writing code.
    pub(crate) async fn chat_completion_plain(
        &self,
        messages: Vec<Value>,
        max_tokens: u64,
    ) -> Result<String> {
        let mut body = json!({
            "messages": messages,
            "stream": false,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        });
        self.set_model(&mut body);

        let response = self.post_chat(&body).await?;

        if let Some(error) = response["error"]["message"].as_str() {
            bail!("{}", truncate_at_boundary(error, 500));
        }

        let message = &response["choices"][0]["message"];
        let content = message["content"].as_str().unwrap_or("").trim();
        if !content.is_empty() {
            return Ok(content.to_string());
        }

        // Some models put everything in the reasoning field and send empty content. Falling
        // back to it beats reporting an empty reply as a server failure. Both spellings are
        // in the wild, and a reviewer verdict lost to the wrong key would gate a call that
        // the model actually approved.
        for key in ["reasoning_content", "reasoning"] {
            let reasoning = message[key].as_str().unwrap_or("").trim();
            if !reasoning.is_empty() {
                return Ok(reasoning.to_string());
            }
        }

        bail!("the model returned an empty reply")
    }

    // Stream a chat completion, handing each piece of assistant text to `on_text` as it
    // arrives, and returning the fully reassembled `message` object.
    //
    // Without this a local model at ~15 tok/s leaves the user staring at a blank terminal
    // for a minute before anything appears. Servers that do not actually implement SSE are
    // handled too: if no `data:` event ever arrives, the body is parsed as an ordinary
    // non-streaming completion instead of failing.
    pub(crate) async fn chat_completion_streaming(
        &self,
        messages: Vec<Value>,
        tools: Vec<Value>,
        mut on_chunk: impl FnMut(Chunk),
    ) -> Result<Value> {
        let mut body = json!({
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": true
        });
        self.set_model(&mut body);
        self.set_sampling(&mut body);

        let mut response = self
            .authorize(
                self.client
                    .post(format!("{}/v1/chat/completions", self.base_url)),
            )
            .json(&body)
            .send()
            .await
            .context("Failed to send request to LM Studio")?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            bail!("{}", unauthorized_message(self.token.is_some()));
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!(
                "LLM server returned {}: {}",
                status,
                truncate_at_boundary(body.trim(), 1000)
            );
        }

        let mut assembler = StreamAssembler::default();
        // Bytes, not a String: a chunk boundary can fall inside a multi-byte character, and
        // decoding early would turn it into a replacement character.
        let mut buffer: Vec<u8> = Vec::new();
        let mut raw: Vec<u8> = Vec::new();
        let mut saw_event = false;
        let mut done = false;

        while let Some(chunk) = response
            .chunk()
            .await
            .context("Connection dropped mid-stream")?
        {
            raw.extend_from_slice(&chunk);
            buffer.extend_from_slice(&chunk);

            // A newline is always a character boundary, so a complete line decodes cleanly.
            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);

                let Some(payload) = line.trim().strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();

                if payload == "[DONE]" {
                    done = true;
                    break;
                }
                if payload.is_empty() {
                    continue;
                }

                saw_event = true;

                // A malformed event should not abandon a response that is otherwise fine.
                let Ok(parsed) = serde_json::from_str::<Value>(payload) else {
                    continue;
                };

                if let Some(error) = parsed.get("error").filter(|e| !e.is_null()) {
                    bail!(
                        "LLM server returned an error mid-stream: {}",
                        error
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or(&error.to_string())
                    );
                }

                for piece in assembler.push(&parsed) {
                    on_chunk(piece);
                }
            }

            if done {
                break;
            }
        }

        // The server ignored "stream": true and answered with an ordinary completion.
        if !saw_event {
            let parsed: Value = serde_json::from_slice(&raw)
                .context("LLM server sent neither a stream nor a parseable response")?;

            if let Some(error) = parsed.get("error").filter(|e| !e.is_null()) {
                bail!(
                    "LLM server returned an error: {}",
                    error
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or(&error.to_string())
                );
            }

            let message = parsed["choices"][0]
                .get("message")
                .filter(|m| m.is_object())
                .cloned()
                .context("LLM server returned no message")?;

            // Handed to the caller in the same two flavors the streaming path uses, so the
            // REPL has one way of showing a reply rather than one per transport. The whole
            // thing arrives at once here, which is the nature of a server that ignored
            // "stream": true — there was nothing to show earlier.
            for key in ["reasoning", "reasoning_content"] {
                if let Some(text) = message
                    .get(key)
                    .and_then(|r| r.as_str())
                    .filter(|s| !s.is_empty())
                {
                    on_chunk(Chunk::Reasoning(text.to_string()));
                }
            }
            if let Some(text) = message
                .get("content")
                .and_then(|c| c.as_str())
                .filter(|s| !s.is_empty())
            {
                on_chunk(Chunk::Content(text.to_string()));
            }
            return Ok(message);
        }

        Ok(assembler.into_message())
    }

    pub(crate) async fn simple_completion(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let messages = vec![
            json!({
                "role": "system",
                "content": system_prompt
            }),
            json!({
                "role": "user",
                "content": user_prompt
            })
        ];

        let mut body = json!({
            "messages": messages
        });
        self.set_model(&mut body);
        self.set_sampling(&mut body);

        let result = self.post_chat(&body).await?;

        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(content)
    }
}

// One tool call being reassembled from stream fragments.

#[derive(Default, Clone)]
pub(crate) struct StreamToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

// Reassembles an OpenAI-style streamed response into the same `message` object the
// non-streaming endpoint returns, so the REPL's tool handling is identical either way.
//
// Content arrives token by token. Tool calls arrive as fragments keyed by `index`: the id
// and name normally come complete in the first fragment for that index, while `arguments`
// is split across many and has to be concatenated in arrival order. Some servers repeat the
// id and name in every fragment, so those are set once and later copies ignored rather than
// appended — the failure mode of the alternative ("readreadread") is much worse than not
// supporting the rare server that genuinely fragments a name.

#[derive(Default)]
pub(crate) struct StreamAssembler {
    pub(crate) content: String,
    pub(crate) reasoning: String,
    pub(crate) tool_calls: Vec<StreamToolCall>,
}

// A piece of the reply as it lands, tagged with which of the two streams it came from.
//
// They are kept apart rather than concatenated because they are shown differently: the
// answer is rendered as markdown, the chain of thought is dimmed. Reasoning used to be
// swallowed here and printed only once the response was complete, which meant a model
// that thinks for fifteen minutes before its first content token left the screen blank
// for fifteen minutes and then emptied its whole preamble onto it at once.
pub(crate) enum Chunk {
    Content(String),
    Reasoning(String),
}

impl StreamAssembler {
    // Feed one parsed `data:` payload. Returns whatever newly arrived text it carried, in
    // arrival order, so the caller can print it the moment it lands. Usually nothing or a
    // single piece; a delta carrying both kinds yields both, thinking first.
    pub(crate) fn push(&mut self, chunk: &Value) -> Vec<Chunk> {
        let mut pieces = Vec::new();
        let Some(delta) = chunk
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("delta"))
        else {
            return pieces;
        };

        // Some models stream their chain of thought separately, under either key.
        for key in ["reasoning", "reasoning_content"] {
            if let Some(text) = delta
                .get(key)
                .and_then(|r| r.as_str())
                .filter(|s| !s.is_empty())
            {
                self.reasoning.push_str(text);
                pieces.push(Chunk::Reasoning(text.to_string()));
            }
        }

        if let Some(calls) = delta.get("tool_calls").and_then(|c| c.as_array()) {
            for call in calls {
                let index = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                if self.tool_calls.len() <= index {
                    self.tool_calls.resize(index + 1, StreamToolCall::default());
                }
                let slot = &mut self.tool_calls[index];

                if let Some(id) = call.get("id").and_then(|i| i.as_str()) {
                    if slot.id.is_empty() {
                        slot.id = id.to_string();
                    }
                }
                if let Some(function) = call.get("function") {
                    if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                        if slot.name.is_empty() {
                            slot.name = name.to_string();
                        }
                    }
                    if let Some(arguments) = function.get("arguments").and_then(|a| a.as_str()) {
                        slot.arguments.push_str(arguments);
                    }
                }
            }
        }

        if let Some(text) = delta
            .get("content")
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
        {
            self.content.push_str(text);
            pieces.push(Chunk::Content(text.to_string()));
        }

        pieces
    }

    pub(crate) fn into_message(self) -> Value {
        let mut message = json!({
            "role": "assistant",
            "content": self.content,
        });

        // Echoed back under the key the thinking-model chat templates read (Qwen3,
        // DeepSeek-R1 and friends), whichever of the two keys it arrived under, so the
        // model is shown its own chain of thought on the next round instead of having it
        // silently dropped by the template.
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = json!(self.reasoning);
        }

        // A fragment that never carried a name is a hole in the stream, not a call.
        let calls: Vec<Value> = self
            .tool_calls
            .into_iter()
            .enumerate()
            .filter(|(_, call)| !call.name.is_empty())
            .map(|(index, call)| {
                json!({
                    // The id is echoed back as tool_call_id, so it cannot be left empty
                    // when a server omits it.
                    "id": if call.id.is_empty() { format!("call_{}", index) } else { call.id },
                    "type": "function",
                    "function": { "name": call.name, "arguments": call.arguments }
                })
            })
            .collect();

        if !calls.is_empty() {
            message["tool_calls"] = json!(calls);
        }

        message
    }
}

// Tool parameter structs

pub(crate) async fn fetch_available_models(base_url: &str, token: Option<&str>) -> Result<Vec<String>> {
    let client = reqwest::Client::new();
    let request = client.get(format!("{}/v1/models", normalize_base_url(base_url)));
    let request = match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    };

    let response = request
        .send()
        .await
        .context("Failed to fetch models from server")?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        bail!("{}", unauthorized_message(token.is_some()));
    }

    let models_response: ModelsResponse = response
        .json()
        .await
        .context("Failed to parse models response")?;

    let model_ids: Vec<String> = models_response
        .data
        .into_iter()
        .map(|m| m.id)
        .collect();

    Ok(model_ids)
}

// Analyze text to determine if it contains Markdown formatting

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

    #[test]
    fn reassembles_streamed_content_and_tool_calls() {
        use crate::StreamAssembler;

        let mut assembler = StreamAssembler::default();
        let mut printed = String::new();

        // Text arrives token by token; each token is handed back for immediate printing.
        for token in ["Let", " me", " look"] {
            for piece in assembler.push(&json!({"choices": [{"delta": {"content": token}}]})) {
                match piece {
                    crate::Chunk::Content(text) => printed.push_str(&text),
                    crate::Chunk::Reasoning(text) => panic!("not thinking: {}", text),
                }
            }
        }
        assert_eq!(printed, "Let me look");

        // A tool call: id and name complete in the first fragment, arguments split across
        // several. The repeated id/name in the second fragment must not be appended twice.
        assert!(assembler
            .push(&json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "id": "call_abc",
                "function": {"name": "read", "arguments": "{\"file"}
            }]}}]}))
            .is_empty());
        assembler.push(&json!({"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_abc",
            "function": {"name": "read", "arguments": "_path\":\"a.rs\"}"}
        }]}}]}));

        // A second, interleaved call at a different index.
        assembler.push(&json!({"choices": [{"delta": {"tool_calls": [{
            "index": 1,
            "id": "call_def",
            "function": {"name": "get_date", "arguments": ""}
        }]}}]}));

        // A chunk with no delta at all (keepalive / usage-only) is harmless.
        assert!(assembler.push(&json!({"choices": [{"finish_reason": "tool_calls"}]})).is_empty());

        let message = assembler.into_message();
        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "Let me look");

        let calls = message["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_abc");
        assert_eq!(calls[0]["function"]["name"], "read");
        assert_eq!(calls[0]["function"]["arguments"], "{\"file_path\":\"a.rs\"}");
        assert_eq!(calls[1]["id"], "call_def");
        assert_eq!(calls[1]["function"]["name"], "get_date");

        // The reassembled arguments must survive the same parse the REPL does.
        assert_eq!(
            parse_tool_arguments(&calls[0]["function"]).unwrap(),
            json!({"file_path": "a.rs"})
        );
    }


    #[test]
    fn stream_without_tool_calls_or_content_stays_well_formed() {
        use crate::StreamAssembler;

        // Reasoning-only stream: kept, but not presented as content.
        let mut assembler = StreamAssembler::default();
        assembler.push(&json!({"choices": [{"delta": {"reasoning": "hmm"}}]}));
        let message = assembler.into_message();
        assert_eq!(message["content"], "");
        assert_eq!(message["reasoning_content"], "hmm");
        assert!(message.get("tool_calls").is_none());

        // A server that omits the id still yields something we can echo back as
        // tool_call_id, and a nameless fragment is dropped rather than sent as a call.
        let mut assembler = StreamAssembler::default();
        assembler.push(&json!({"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"name": "get_time", "arguments": ""}
        }]}}]}));
        assembler.push(&json!({"choices": [{"delta": {"tool_calls": [{
            "index": 1, "function": {"arguments": "{}"}
        }]}}]}));
        let message = assembler.into_message();
        let calls = message["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_0");
        assert_eq!(calls[0]["function"]["name"], "get_time");
    }

    // Serve one canned HTTP response on an ephemeral port and hand back its address.
    // Exercises the real reqwest -> SSE-decoding path without contacting anything.

    #[tokio::test]
    async fn configured_sampling_reaches_the_wire() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";

        // Nothing configured: the fields must be absent, not sent as defaults.
        let (url, captured) = serve_once_capturing(sse, "text/event-stream").await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        client
            .chat_completion_streaming(vec![], vec![], |_| {})
            .await
            .unwrap();
        let sent: serde_json::Value =
            serde_json::from_str(&captured.lock().unwrap()).expect("a JSON request body");
        assert_eq!(sent["stream"], true);
        assert!(sent.get("temperature").is_none(), "sent: {}", sent);
        assert!(sent.get("max_tokens").is_none(), "sent: {}", sent);

        // A strictly-OpenAI endpoint must never see the extensions unasked.
        assert!(sent.get("top_k").is_none(), "sent: {}", sent);
        assert!(sent.get("repeat_penalty").is_none(), "sent: {}", sent);

        // Configured: they must actually arrive.
        let (url, captured) = serve_once_capturing(sse, "text/event-stream").await;
        let mut client = crate::LMStudioClient::new(url, "m".to_string(), None);
        client.apply_settings(&settings_of(Some(0.2), Some(512), None, Some(40), Some(1.15)));
        client
            .chat_completion_streaming(vec![], vec![], |_| {})
            .await
            .unwrap();
        let sent: serde_json::Value =
            serde_json::from_str(&captured.lock().unwrap()).expect("a JSON request body");
        assert_eq!(sent["temperature"], 0.2);
        assert_eq!(sent["max_tokens"], 512);
        // Extensions rather than OpenAI fields, but the local servers this targets read
        // them, so they have to arrive under exactly these names.
        assert_eq!(sent["top_k"], 40);
        assert_eq!(sent["repeat_penalty"], 1.15);
        // top_k is a count: sending 40.0 where an integer is wanted invites a 400.
        assert!(sent["top_k"].is_u64(), "top_k went out as {}", sent["top_k"]);
        // Still absent when not configured, even with its neighbours set.
        assert!(sent.get("top_p").is_none(), "sent: {}", sent);
    }


    #[tokio::test]
    async fn streams_server_sent_events_end_to_end() {
        // Deliberately awkward: a blank-line-separated stream, one event carrying both text
        // and a tool-call fragment, an unparseable event to skip, and a trailing [DONE].
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n",
            ": keepalive comment\n\n",
            "data: not-json\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"get_date\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let url = serve_once(body, "text/event-stream").await;

        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        let mut streamed = String::new();
        let message = client
            .chat_completion_streaming(vec![], vec![], |chunk| {
                if let crate::Chunk::Content(text) = chunk {
                    streamed.push_str(&text);
                }
            })
            .await
            .unwrap();

        // Text reached the caller incrementally, not just in the final message.
        assert_eq!(streamed, "Hi there");
        assert_eq!(message["content"], "Hi there");
        assert_eq!(message["tool_calls"][0]["function"]["name"], "get_date");
    }


    #[tokio::test]
    async fn thinking_reaches_the_caller_while_it_is_still_thinking() {
        // A reasoning model: a long preamble under `reasoning_content`, then the answer.
        // The preamble has to arrive piece by piece and stay tagged as thinking — held
        // back until the end, it used to leave the screen blank for as long as the model
        // took and then land in one block that read like the answer.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Yes, \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"wait…\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"42\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let url = serve_once(body, "text/event-stream").await;

        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        // Recorded in arrival order, so the test also pins that thinking precedes content.
        let mut order = Vec::new();
        let message = client
            .chat_completion_streaming(vec![], vec![], |chunk| match chunk {
                crate::Chunk::Reasoning(text) => order.push(format!("think:{}", text)),
                crate::Chunk::Content(text) => order.push(format!("say:{}", text)),
            })
            .await
            .unwrap();

        assert_eq!(order, ["think:Yes, ", "think:wait…", "say:42"]);
        // And it is still echoed back to the model, which is what the chat templates read.
        assert_eq!(message["reasoning_content"], "Yes, wait…");
        assert_eq!(message["content"], "42");
    }


    #[tokio::test]
    async fn falls_back_when_the_server_ignores_stream() {
        // A server that answers "stream": true with an ordinary completion body.
        let body = concat!(
            "{\"choices\":[{\"message\":{\"role\":\"assistant\",",
            "\"content\":\"plain answer\"}}]}"
        );
        let url = serve_once(body, "application/json").await;

        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        let mut streamed = String::new();
        let message = client
            .chat_completion_streaming(vec![], vec![], |chunk| {
                if let crate::Chunk::Content(text) = chunk {
                    streamed.push_str(&text);
                }
            })
            .await
            .unwrap();

        assert_eq!(message["content"], "plain answer");
        // The caller still gets the text, so the REPL prints it exactly once either way.
        assert_eq!(streamed, "plain answer");
    }


    #[tokio::test]
    async fn reports_a_mid_stream_server_error() {
        let body = "data: {\"error\":{\"message\":\"context overflow\"}}\n\n";
        let url = serve_once(body, "text/event-stream").await;

        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        let error = client
            .chat_completion_streaming(vec![], vec![], |_| {})
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("context overflow"),
            "unexpected error: {}",
            error
        );
    }

    // A server wired to nothing: the file tools never touch the LLM client.

    #[test]
    fn unset_sampling_is_left_out_of_the_request_entirely() {
        use crate::LMStudioClient;

        // Sending a made-up default would silently override whatever the user tuned for
        // that model in LM Studio's own UI, with nothing to indicate why.
        let client = LMStudioClient::new("http://x".to_string(), "m".to_string(), None);
        let mut body = json!({"messages": []});
        client.set_sampling(&mut body);
        for field in ["temperature", "max_tokens", "top_p", "top_k", "repeat_penalty"] {
            assert!(body.get(field).is_none(), "{} should not be sent", field);
        }

        // Only what is configured gets sent.
        let mut client = LMStudioClient::new("http://x".to_string(), "m".to_string(), None);
        client.apply_settings(&settings_of(Some(0.25), None, Some(0.8), None, None));
        let mut body = json!({"messages": []});
        client.set_sampling(&mut body);
        assert_eq!(body["temperature"], 0.25);
        assert_eq!(body["top_p"], 0.8);
        assert!(body.get("max_tokens").is_none());
    }


    #[test]
    fn normalizes_base_urls() {
        // The forms haico already documents are left alone.
        assert_eq!(
            normalize_base_url("http://127.0.0.1:1234"),
            "http://127.0.0.1:1234"
        );
        // The forms LM Studio's UI and OpenAI SDKs hand out.
        assert_eq!(
            normalize_base_url("http://127.0.0.1:1234/v1"),
            "http://127.0.0.1:1234"
        );
        assert_eq!(
            normalize_base_url("http://127.0.0.1:1234/v1/"),
            "http://127.0.0.1:1234"
        );
        assert_eq!(
            normalize_base_url("  http://127.0.0.1:1234/  "),
            "http://127.0.0.1:1234"
        );
        // Bracketed IPv6 hosts survive intact.
        assert_eq!(
            normalize_base_url("http://[225:ca89:40a8:611e:78b8:ab81:999a:d4d7]:1234/v1"),
            "http://[225:ca89:40a8:611e:78b8:ab81:999a:d4d7]:1234"
        );
        // Known trade-off: stripping is unconditional, so a proxy whose base path really
        // does end in "/v1" loses it. Documented rather than special-cased.
        assert_eq!(
            normalize_base_url("http://example.com/api/v1"),
            "http://example.com/api"
        );
    }

}
