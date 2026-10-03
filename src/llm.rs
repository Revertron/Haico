//! The OpenAI-compatible client, including streaming.

use crate::config::ResolvedSettings;
use crate::repetition::RepetitionGuard;
use crate::trace::{self, Caller};
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
    pub(crate) min_p: Option<f64>,
    pub(crate) repeat_penalty: Option<f64>,
    // How hard to think. Not a sampler — the server hands it to the chat template, which is
    // what decides where its own level names set the budget — but it lives with the sampling
    // for the same reason: None leaves the field out, so the model's default survives.
    pub(crate) reasoning_effort: Option<String>,
    // Sent as chat_template_kwargs.enable_thinking, which is how a Qwen3-style template is
    // told not to think at all — `reasoning_effort` cannot say it, because "off" is not one
    // of the levels every model offers. Nothing sets it at the moment: `for_sub_agent` did,
    // and that is the change this comment outlived. Kept because it is still the only way to
    // say the thing, for the next model that needs it said.
    pub(crate) enable_thinking: Option<bool>,
    // The level a sub-agent's own conversation asks for, straight from the config. Read
    // only by `for_sub_agent` — the main conversation never sends this one.
    pub(crate) sub_agent_effort: Option<String>,
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
            min_p: None,
            repeat_penalty: None,
            reasoning_effort: None,
            enable_thinking: None,
            sub_agent_effort: None,
        }
    }

    /// A copy of this client for a sub-agent's own conversation.
    ///
    /// The main conversation's effort deliberately does not carry over. A sub-agent is
    /// summarizing, judging or looking something up, and asking it to think as hard as the
    /// coding model spends the user's waiting time on a chain of thought that is dropped
    /// with the message list when the call returns.
    ///
    /// Unset says nothing at all rather than saying "do not think", which is what it said
    /// first. Forbidding it through the template broke the sub-agents it was meant to make
    /// cheaper: the compressor, answering cold, read the trace it was handed as a question
    /// and answered it — a fix with a fenced code block, written in the voice of an adviser
    /// rather than in the model's own, and stored as that reply's chain of thought. The
    /// brief deliberation was what the summary was being made out of. A level named in the
    /// config is still sent as it stands, and the template decides what it means.
    pub(crate) fn for_sub_agent(&self) -> Self {
        let mut client = self.clone();
        client.reasoning_effort = self.sub_agent_effort.clone();
        client
    }

    // Apply the settings resolved for the model currently in use.
    pub(crate) fn apply_settings(&mut self, settings: &ResolvedSettings) {
        self.temperature = settings.temperature.value;
        self.max_tokens = settings.max_tokens.value;
        self.top_p = settings.top_p.value;
        self.top_k = settings.top_k.value;
        self.min_p = settings.min_p.value;
        self.repeat_penalty = settings.repeat_penalty.value;
        self.reasoning_effort = settings.reasoning_effort.value.clone();
        // Not applied to this client's own requests — `for_sub_agent` is what reads it.
        self.sub_agent_effort = settings.sub_agent_effort.clone();
    }

    // How much to think. Its own method rather than part of `set_sampling` because the
    // plain completion builds its body by hand and skips the sampling entirely, and the
    // calls that go through it are exactly the ones that most need this said.
    pub(crate) fn set_thinking(&self, body: &mut Value) {
        // Sent as the user typed it, for the same reason it is stored as a string: the level
        // names are the template's, so a value haico does not recognize is far more likely to
        // be a model it has not met than a typo worth refusing to send.
        if let Some(reasoning_effort) = &self.reasoning_effort {
            body["reasoning_effort"] = json!(reasoning_effort);
        }
        if let Some(enable_thinking) = self.enable_thinking {
            body["chat_template_kwargs"] = json!({ "enable_thinking": enable_thinking });
        }
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
        if let Some(min_p) = self.min_p {
            body["min_p"] = json!(min_p);
        }
        if let Some(repeat_penalty) = self.repeat_penalty {
            body["repeat_penalty"] = json!(repeat_penalty);
        }
        self.set_thinking(body);
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

    // `post_chat`, recorded in the trace when there is one. The reply is recorded as the
    // assistant message rather than the whole envelope, so every `call` line in the trace
    // has the same shape whichever path produced it; a server error in a 200 body is
    // recorded as the error it is.
    async fn post_chat_traced(&self, caller: Caller, body: &Value) -> Result<Value> {
        let started = std::time::Instant::now();
        let response = self.post_chat(body).await;
        if trace::enabled() {
            let ending;
            let outcome = match &response {
                Ok(reply) => match reply["error"]["message"].as_str() {
                    Some(error) => Err(error.to_string()),
                    None => {
                        ending = Ending::of(reply);
                        Ok(trace::Reply { message: &reply["choices"][0]["message"], ending: &ending })
                    }
                },
                Err(e) => Err(format!("{:#}", e)),
            };
            trace::call(caller, body, outcome, started.elapsed());
        }
        response
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
        caller: Caller,
        messages: Vec<Value>,
        max_tokens: Option<u64>,
    ) -> Result<String> {
        let mut body = json!({
            "messages": messages,
            "stream": false,
            "temperature": 0.0,
        });
        if let Some(max_tokens) = max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        self.set_model(&mut body);
        // Every caller of this is a sub-agent, and every one of them reaches it through a
        // `for_sub_agent` copy — so what this sends is the sub-agent's thinking, never the
        // main conversation's. A main-client call landing here would send the main effort,
        // which is the bug this comment exists to make visible if one is ever added.
        self.set_thinking(&mut body);

        let response = self.post_chat_traced(caller, &body).await?;

        // Logged whole, before anything is picked out of it: when this reply cannot be
        // parsed the interesting part is usually what the server did instead of answering
        // — a finish_reason of "length" for a model still thinking, an empty message with
        // the text in a field nobody read, a prompt that filled the window on its own.
        // None of that survives into the one-line error the caller ends up reporting.
        if crate::debug::enabled() {
            crate::debug::log("plain completion reply", &describe_plain_reply(&response));
        }

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
        //
        // Only trusted when the model actually finished. A reply cut short by max_tokens
        // while still inside its own reasoning (finish_reason "length") leaves that field
        // holding an unfinished scratchpad, not an answer — returning it verbatim is how a
        // compressor's own mid-thought about how to summarize ends up stored as the
        // compressed trace it was asked to produce instead.
        let finish_reason = response["choices"][0]["finish_reason"].as_str();
        if finish_reason != Some("length") {
            for key in ["reasoning_content", "reasoning"] {
                let reasoning = message[key].as_str().unwrap_or("").trim();
                if !reasoning.is_empty() {
                    return Ok(reasoning.to_string());
                }
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
        caller: Caller,
        messages: Vec<Value>,
        tools: Vec<Value>,
        on_chunk: impl FnMut(Chunk),
    ) -> Result<Value> {
        self.chat_completion_measured(caller, messages, tools, on_chunk).await.map(|(message, _)| message)
    }

    // The same, also handing back how the reply ended and what the server says it cost.
    // `usage.prompt_tokens` is what the main loop sizes the history by (`Agent::context_used`):
    // haico's own estimate ran 20% low on a p2p_auth run, so the history reached the
    // window's last token while the estimate still put it under the compaction line.
    pub(crate) async fn chat_completion_measured(
        &self,
        caller: Caller,
        messages: Vec<Value>,
        tools: Vec<Value>,
        on_chunk: impl FnMut(Chunk),
    ) -> Result<(Value, Ending)> {
        let mut body = json!({
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": true
        });
        self.set_model(&mut body);
        self.set_sampling(&mut body);

        // Recorded once the stream is over, as the message it reassembled into: that is
        // what the model produced, whereas the fragments are only how it travelled. A
        // stream dropped by an interrupt never gets here — the future is dropped with it.
        let tracing = trace::enabled();
        // Asked for on every request now, not only while tracing as it once was: the count
        // is what decides when an unattended run compacts, and an estimate is what let one
        // run into the end of its window. Every server this targets accepts it (OpenAI's own
        // field; LM Studio, llama.cpp and Ollama send the final usage chunk), and one that
        // ignores it just leaves the count to the estimate.
        body["stream_options"] = json!({ "include_usage": true });
        let started = std::time::Instant::now();
        let reply = self.stream_chat(&body, on_chunk).await;
        if tracing {
            let outcome = match &reply {
                Ok((message, ending)) => Ok(trace::Reply { message, ending }),
                Err(e) => Err(format!("{:#}", e)),
            };
            trace::call(caller, &body, outcome, started.elapsed());
        }
        reply
    }

    async fn stream_chat(
        &self,
        body: &Value,
        mut on_chunk: impl FnMut(Chunk),
    ) -> Result<(Value, Ending)> {
        let mut response = self
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
        // One per kind of text: see `repetition`. Returning drops `response`, and with it
        // the connection, which is what tells the server to stop generating — a loop cut
        // off here is a loop that stops costing GPU time too.
        let mut reasoning_guard = RepetitionGuard::default();
        let mut content_guard = RepetitionGuard::default();
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
                    let looping = match &piece {
                        Chunk::Reasoning(text) => reasoning_guard.push(text),
                        Chunk::Content(text) => content_guard.push(text),
                        Chunk::ToolFragment(_) => None,
                    };
                    on_chunk(piece);
                    if let Some(repeating) = looping {
                        return Err(anyhow::Error::new(repeating));
                    }
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
            return Ok((message, Ending::of(&parsed)));
        }

        let ending = std::mem::take(&mut assembler.ending);
        Ok((assembler.into_message(), ending))
    }

    pub(crate) async fn simple_completion(
        &self,
        caller: Caller,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String> {
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

        let result = self.post_chat_traced(caller, &body).await?;

        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(content)
    }
}

// A non-streaming reply, laid out for the debug log.
//
// Written field by field rather than as the raw JSON because the text is the point: in a
// JSON dump a reasoning block arrives as one enormous line with every newline escaped,
// which is exactly the thing that has to be read to see where the answer went.
fn describe_plain_reply(response: &Value) -> String {
    let choice = &response["choices"][0];
    let mut out = format!(
        "finish_reason: {}\n",
        choice["finish_reason"].as_str().unwrap_or("(none)")
    );

    if let Some(usage) = response["usage"].as_object() {
        // The token counts say whether the cap was the problem and, just as often,
        // whether the prompt left any room for an answer at all.
        let mut counts: Vec<String> = usage
            .iter()
            .filter_map(|(k, v)| v.as_u64().map(|n| format!("{}={}", k, n)))
            .collect();
        counts.sort();
        if !counts.is_empty() {
            out.push_str(&format!("usage: {}\n", counts.join(" ")));
        }
    }

    if let Some(error) = response["error"]["message"].as_str() {
        out.push_str(&format!("error: {}\n", error));
    }

    let message = &choice["message"];
    let mut had_text = false;
    for key in ["content", "reasoning_content", "reasoning"] {
        let text = message[key].as_str().unwrap_or("");
        if !text.is_empty() {
            out.push_str(&format!("{} ({} bytes):\n{}\n", key, text.len(), text));
            had_text = true;
        }
    }
    if !had_text {
        out.push_str("no text in content, reasoning_content or reasoning\n");
    }

    out
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
    // How the reply ended and what it cost, as far as the server says. Kept out of the
    // message because the message goes back into the history, and neither is part of
    // what the model said; the trace and the context count want them. A stream that hit
    // `max_tokens` mid-thought looks exactly like a finished one without `finish_reason` —
    // the one thing a dataset built from the trace most needs to be able to tell.
    pub(crate) ending: Ending,
}

/// The two things about a reply that are not the reply. `usage` only arrives on a stream
/// when asked for (`stream_options.include_usage`), which the main stream always does.
#[derive(Default, Debug, Clone, PartialEq)]
pub(crate) struct Ending {
    pub(crate) finish_reason: Option<String>,
    pub(crate) usage: Option<Value>,
}

impl Ending {
    /// Read off a whole (non-streamed) completion.
    fn of(response: &Value) -> Ending {
        Ending {
            finish_reason: response["choices"][0]["finish_reason"].as_str().map(str::to_string),
            usage: response.get("usage").filter(|u| u.is_object()).cloned(),
        }
    }
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
    /// How many characters of a tool call just arrived: the name and the
    /// arguments, not the id a server made up for it.
    ///
    /// A count and not the text, deliberately. The call is shown once, whole,
    /// when it is about to run, and a file's worth of arguments has no business
    /// being streamed onto the screen on the way there as well — a caller that
    /// never receives the text cannot print it by accident. What it is for is the
    /// activity bar: without it a model spending a minute writing one long
    /// argument looks exactly like one that has hung.
    ToolFragment(usize),
}

impl StreamAssembler {
    // Feed one parsed `data:` payload. Returns whatever newly arrived text it carried, in
    // arrival order, so the caller can print it the moment it lands. Usually nothing or a
    // single piece; a delta carrying both kinds yields both, thinking first.
    pub(crate) fn push(&mut self, chunk: &Value) -> Vec<Chunk> {
        let mut pieces = Vec::new();
        // Before the delta is looked for: the usage chunk has an empty `choices`, and the
        // finishing chunk often an empty delta.
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.ending.usage = Some(usage.clone());
        }
        if let Some(reason) = chunk["choices"][0]["finish_reason"].as_str() {
            self.ending.finish_reason = Some(reason.to_string());
        }
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
            // What this delta actually added to the calls being assembled. Counted as it
            // is stored rather than as it arrives: a server that repeats the name on every
            // fragment would otherwise have the count run away from what the model wrote.
            let mut fragment = 0usize;
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
                            fragment += name.chars().count();
                        }
                    }
                    if let Some(arguments) = function.get("arguments").and_then(|a| a.as_str()) {
                        slot.arguments.push_str(arguments);
                        fragment += arguments.chars().count();
                    }
                }
            }
            if fragment > 0 {
                pieces.push(Chunk::ToolFragment(fragment));
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

// Which models the server actually has in memory, as opposed to merely knows about.
//
// Best effort by design, and it never fails: neither endpoint is part of the
// OpenAI-compatible surface, so a server that has never heard of either simply looks like
// one with nothing loaded — which is the case the picker already handles. An empty answer
// is not an error worth showing anybody.
//
// `/v1/models` is no help for this. Depending on the version it lists everything
// downloaded or only what is in memory, and it carries no field to tell the two apart.
pub(crate) async fn fetch_loaded_models(base_url: &str, token: Option<&str>) -> Vec<String> {
    // LM Studio's own REST API, v1 — released with LM Studio 0.4.0 and the one it now
    // recommends. `/api/v0` is still answered and is still the fallback below, because a
    // v1 request to anything older 404s.
    //
    // The state is the shape of `loaded_instances` rather than a string: a model with
    // none is merely downloaded. And the name worth reporting is the *instance's* id, not
    // the entry's `key` — the key carries the quantization suffix and names the file on
    // disk (`nail-qwen3.6-35b@q4_k_xl`), while the instance answers to the bare name
    // (`nail-qwen3.6-35b`). Both are accepted in a request's `model` field, so taking the
    // key would look like it worked while naming something other than what is in memory.
    #[derive(Deserialize)]
    struct LmStudioV1 {
        #[serde(default)]
        models: Vec<LmStudioV1Model>,
    }
    #[derive(Deserialize)]
    struct LmStudioV1Model {
        #[serde(default)]
        key: String,
        #[serde(default, rename = "type")]
        kind: String,
        #[serde(default)]
        loaded_instances: Vec<LmStudioInstance>,
    }
    #[derive(Deserialize)]
    struct LmStudioInstance {
        #[serde(default)]
        id: String,
    }

    // LM Studio's v0: every model it knows, each with the state it is in.
    #[derive(Deserialize)]
    struct LmStudioModels {
        #[serde(default)]
        data: Vec<LmStudioModel>,
    }
    #[derive(Deserialize)]
    struct LmStudioModel {
        id: String,
        #[serde(default)]
        state: String,
        #[serde(default, rename = "type")]
        kind: String,
    }

    // Embedding models are loaded alongside chat ones and would be a nonsense choice
    // here. Matched by prefix because the two APIs disagree on the word: v0 says
    // "embeddings" and v1 says "embedding", and a filter copied from one to the other
    // would quietly stop excluding anything.
    fn is_embedding(kind: &str) -> bool {
        kind.starts_with("embedding")
    }

    // Ollama's: only what is loaded right now, which is exactly the question being asked.
    #[derive(Deserialize)]
    struct OllamaRunning {
        #[serde(default)]
        models: Vec<OllamaModel>,
    }
    #[derive(Deserialize)]
    struct OllamaModel {
        #[serde(default)]
        model: String,
        #[serde(default)]
        name: String,
    }

    let base = normalize_base_url(base_url);
    let client = reqwest::Client::new();
    let get = |path: &str| {
        let request = client.get(format!("{}{}", base, path));
        match token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
        .send()
    };

    if let Ok(response) = get("/api/v1/models").await {
        if response.status().is_success() {
            if let Ok(listed) = response.json::<LmStudioV1>().await {
                let mut loaded = Vec::new();
                for model in listed.models {
                    if is_embedding(&model.kind) {
                        continue;
                    }
                    let LmStudioV1Model { key, loaded_instances, .. } = model;
                    for instance in loaded_instances {
                        // The key only as a last resort: an instance with no id of its own
                        // is still something running, and naming it by its file beats
                        // dropping it from the list entirely.
                        let name = if instance.id.is_empty() { key.clone() } else { instance.id };
                        if !name.is_empty() {
                            loaded.push(name);
                        }
                    }
                }
                if !loaded.is_empty() {
                    return loaded;
                }
            }
        }
    }

    if let Ok(response) = get("/api/v0/models").await {
        if response.status().is_success() {
            if let Ok(listed) = response.json::<LmStudioModels>().await {
                let loaded: Vec<String> = listed
                    .data
                    .into_iter()
                    .filter(|model| model.state == "loaded" && !is_embedding(&model.kind))
                    .map(|model| model.id)
                    .collect();
                if !loaded.is_empty() {
                    return loaded;
                }
            }
        }
    }

    if let Ok(response) = get("/api/ps").await {
        if response.status().is_success() {
            if let Ok(running) = response.json::<OllamaRunning>().await {
                return running
                    .models
                    .into_iter()
                    .map(|model| if model.model.is_empty() { model.name } else { model.model })
                    .filter(|name| !name.is_empty())
                    .collect();
            }
        }
    }

    Vec::new()
}

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

    /// A reply cut off by `max_tokens` and a finished one differ only in these, and the
    /// usage chunk arrives with an empty `choices` — the case the delta lookup skips.
    #[test]
    fn a_stream_keeps_how_it_ended_out_of_the_message() {
        use crate::StreamAssembler;

        let mut assembler = StreamAssembler::default();
        assembler.push(&json!({"choices": [{"delta": {"content": "half a th"}}]}));
        assembler.push(&json!({"choices": [{"delta": {}, "finish_reason": "length"}]}));
        assembler.push(&json!({"choices": [], "usage": {"completion_tokens": 512}}));

        let ending = std::mem::take(&mut assembler.ending);
        assert_eq!(ending.finish_reason.as_deref(), Some("length"));
        assert_eq!(ending.usage, Some(json!({"completion_tokens": 512})));
        let message = assembler.into_message();
        assert!(message.get("finish_reason").is_none() && message.get("usage").is_none());
    }

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
                    crate::Chunk::ToolFragment(n) => panic!("not a call: {} chars", n),
                }
            }
        }
        assert_eq!(printed, "Let me look");

        // A tool call: id and name complete in the first fragment, arguments split across
        // several. The repeated id/name in the second fragment must not be appended twice.
        // Nothing to print — what comes back is the size of what was stored, for the bar.
        let fragment = assembler.push(&json!({"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_abc",
            "function": {"name": "read", "arguments": "{\"file"}
        }]}}]}));
        assert!(
            matches!(fragment[..], [crate::Chunk::ToolFragment(n)] if n == "read{\"file".len()),
            "expected one counted fragment"
        );
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


    // The activity bar counts what has arrived, and a model writing out a long argument
    // used to leave it sitting still for the length of the write — which looks exactly
    // like a hang. What is counted is what was *stored*: a server repeating the id and
    // the name on every fragment must not inflate it, and the id is the server's own
    // bookkeeping rather than anything the model spent time generating.
    #[test]
    fn a_tool_call_being_written_out_is_counted_but_not_handed_over_as_text() {
        use crate::StreamAssembler;

        let mut assembler = StreamAssembler::default();
        let mut counted = 0usize;
        let mut push = |delta| {
            for piece in assembler.push(&json!({"choices": [{"delta": delta}]})) {
                match piece {
                    crate::Chunk::ToolFragment(n) => counted += n,
                    crate::Chunk::Content(text) | crate::Chunk::Reasoning(text) => {
                        panic!("a call was handed over as text: {}", text)
                    }
                }
            }
        };

        push(json!({"tool_calls": [{
            "index": 0,
            "id": "call_abc",
            "function": {"name": "write_file", "arguments": "{\"path\":"}
        }]}));
        // The same id and the same name again, as several servers send them.
        push(json!({"tool_calls": [{
            "index": 0,
            "id": "call_abc",
            "function": {"name": "write_file", "arguments": "\"a.rs\"}"}
        }]}));

        assert_eq!(counted, "write_file".len() + "{\"path\":\"a.rs\"}".len());
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
            .chat_completion_streaming(Caller::Main, vec![], vec![], |_| {})
            .await
            .unwrap();
        let sent: serde_json::Value =
            serde_json::from_str(&captured.lock().unwrap()).expect("a JSON request body");
        assert_eq!(sent["stream"], true);
        assert!(sent.get("temperature").is_none(), "sent: {}", sent);
        assert!(sent.get("max_tokens").is_none(), "sent: {}", sent);

        // A strictly-OpenAI endpoint must never see the extensions unasked.
        assert!(sent.get("top_k").is_none(), "sent: {}", sent);
        assert!(sent.get("min_p").is_none(), "sent: {}", sent);
        assert!(sent.get("repeat_penalty").is_none(), "sent: {}", sent);

        // Configured: they must actually arrive.
        let (url, captured) = serve_once_capturing(sse, "text/event-stream").await;
        let mut client = crate::LMStudioClient::new(url, "m".to_string(), None);
        client.apply_settings(&settings_of(
            Some(0.2),
            Some(512),
            None,
            Some(40),
            Some(0.05),
            Some(1.15),
        ));
        client
            .chat_completion_streaming(Caller::Main, vec![], vec![], |_| {})
            .await
            .unwrap();
        let sent: serde_json::Value =
            serde_json::from_str(&captured.lock().unwrap()).expect("a JSON request body");
        assert_eq!(sent["temperature"], 0.2);
        assert_eq!(sent["max_tokens"], 512);
        // Extensions rather than OpenAI fields, but the local servers this targets read
        // them, so they have to arrive under exactly these names.
        assert_eq!(sent["top_k"], 40);
        assert_eq!(sent["min_p"], 0.05);
        assert_eq!(sent["repeat_penalty"], 1.15);
        // top_k is a count: sending 40.0 where an integer is wanted invites a 400.
        assert!(sent["top_k"].is_u64(), "top_k went out as {}", sent["top_k"]);
        // Still absent when not configured, even with its neighbours set.
        assert!(sent.get("top_p").is_none(), "sent: {}", sent);
    }



    // The shapes in these three are trimmed from what a real LM Studio answered, which is
    // the only way to be sure of them: neither API publishes a response schema.

    #[tokio::test]
    async fn the_v1_api_reports_the_instance_that_is_running_not_the_file_on_disk() {
        // `loaded_instances` is the load state — an empty one means merely downloaded.
        // The instance's id is the bare name and the entry's `key` carries the quant
        // suffix; both are accepted as a request's `model`, so taking the key would look
        // like it worked while naming something other than what is in memory. Note v1
        // spells the embedding type in the singular, where v0 uses the plural.
        const LM_STUDIO_V1: &str = r#"{"models":[
            {"key":"qwen2.5-14b@q4_k_xl","type":"llm",
             "loaded_instances":[{"id":"qwen2.5-14b","config":{"context_length":262144}}]},
            {"key":"llama-3.1-8b@q8_0","type":"llm","loaded_instances":[]},
            {"key":"nomic-embed@f16","type":"embedding",
             "loaded_instances":[{"id":"nomic-embed"}]}
        ]}"#;
        let url = serve_routes(&[("/api/v1/models", LM_STUDIO_V1)]).await;
        assert_eq!(fetch_loaded_models(&url, None).await, ["qwen2.5-14b"]);
    }

    #[tokio::test]
    async fn an_lm_studio_too_old_for_v1_is_still_asked_the_old_way() {
        // v1 arrived in LM Studio 0.4.0; anything older 404s it and answers v0. Falling
        // through matters more than it looks — concluding "nothing is loaded" would send
        // the picker's user to choose a model that is already in memory.
        const LM_STUDIO_V0: &str = r#"{"data":[
            {"id":"qwen2.5-14b","type":"llm","state":"loaded"},
            {"id":"llama-3.1-8b","type":"llm","state":"not-loaded"},
            {"id":"nomic-embed","type":"embeddings","state":"loaded"}
        ]}"#;
        let url = serve_routes(&[("/api/v0/models", LM_STUDIO_V0)]).await;
        assert_eq!(fetch_loaded_models(&url, None).await, ["qwen2.5-14b"]);
    }

    #[tokio::test]
    async fn a_v1_server_with_nothing_loaded_does_not_stop_the_fallbacks() {
        // Every entry present and none of them running. The v1 arm returns nothing rather
        // than an answer, so the probe carries on to the next server instead of reporting
        // an empty list as fact.
        const EMPTY_V1: &str = r#"{"models":[{"key":"qwen2.5-14b@q4_k_xl","type":"llm","loaded_instances":[]}]}"#;
        const OLLAMA: &str = r#"{"models":[{"name":"qwen2.5:14b","model":"qwen2.5:14b"}]}"#;
        let url = serve_routes(&[("/api/v1/models", EMPTY_V1), ("/api/ps", OLLAMA)]).await;
        assert_eq!(fetch_loaded_models(&url, None).await, ["qwen2.5:14b"]);
    }

    #[tokio::test]
    async fn ollama_is_asked_when_lm_studio_does_not_answer() {
        // The other server answers a different endpoint and 404s this one, so the probe
        // has to fall through rather than conclude that nothing is loaded.
        const OLLAMA: &str = r#"{"models":[{"name":"qwen2.5:14b","model":"qwen2.5:14b"}]}"#;
        let url = serve_routes(&[("/api/ps", OLLAMA)]).await;
        assert_eq!(fetch_loaded_models(&url, None).await, ["qwen2.5:14b"]);

        // And a server that has heard of neither simply has nothing running, which is not
        // an error: the picker takes it from there.
        let url = serve_routes(&[("/v1/models", r#"{"data":[{"id":"m"}]}"#)]).await;
        assert!(fetch_loaded_models(&url, None).await.is_empty());
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
            .chat_completion_streaming(Caller::Main, vec![], vec![], |chunk| {
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
            .chat_completion_streaming(Caller::Main, vec![], vec![], |chunk| match chunk {
                crate::Chunk::Reasoning(text) => order.push(format!("think:{}", text)),
                crate::Chunk::Content(text) => order.push(format!("say:{}", text)),
                crate::Chunk::ToolFragment(n) => order.push(format!("call:{}", n)),
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
            .chat_completion_streaming(Caller::Main, vec![], vec![], |chunk| {
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
            .chat_completion_streaming(Caller::Main, vec![], vec![], |_| {})
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
        for field in [
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "reasoning_effort",
        ] {
            assert!(body.get(field).is_none(), "{} should not be sent", field);
        }

        // Only what is configured gets sent.
        let mut client = LMStudioClient::new("http://x".to_string(), "m".to_string(), None);
        client.apply_settings(&settings_of(Some(0.25), None, Some(0.8), None, None, None));
        let mut body = json!({"messages": []});
        client.set_sampling(&mut body);
        assert_eq!(body["temperature"], 0.25);
        assert_eq!(body["top_p"], 0.8);
        assert!(body.get("max_tokens").is_none());

        // reasoning_effort goes out as the string it came in as, unparsed: "high" and a
        // level only one model has both have to survive the trip, because what the names
        // mean is the chat template's business rather than haico's.
        let mut settings = settings_of(None, None, None, None, None, None);
        settings.reasoning_effort =
            crate::Setting { value: Some("high".to_string()), source: "test" };
        let mut client = LMStudioClient::new("http://x".to_string(), "m".to_string(), None);
        client.apply_settings(&settings);
        let mut body = json!({"messages": []});
        client.set_sampling(&mut body);
        assert_eq!(body["reasoning_effort"], "high");
    }


    /// A sub-agent asks for its own thinking, never the main conversation's: a copy that
    /// quietly inherited "high" would spend the user's waiting time on a chain of thought
    /// that is dropped when the call returns. Unset is the interesting half — it says
    /// nothing rather than saying "do not think". Forbidding it outright was tried, and the
    /// compressor answering cold read the trace it was handed as a question and answered it.
    #[test]
    fn a_sub_agent_asks_for_its_own_thinking() {
        use crate::LMStudioClient;

        let mut settings = settings_of(None, None, None, None, None, None);
        settings.reasoning_effort =
            crate::Setting { value: Some("high".to_string()), source: "test" };
        let mut client = LMStudioClient::new("http://x".to_string(), "m".to_string(), None);
        client.apply_settings(&settings);

        // Nothing configured for sub-agents: nothing is said about thinking either way, so
        // the model's own default applies, and the main conversation's "high" is gone rather
        // than carried over.
        let helper = client.for_sub_agent();
        let mut body = json!({"messages": []});
        helper.set_thinking(&mut body);
        assert!(body.get("reasoning_effort").is_none(), "the main effort must not ride along");
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "a sub-agent told not to think answers cold, which is what broke the compressor"
        );

        // A level named in the config is sent instead, and then nothing is said about
        // enable_thinking — asking for a level and forbidding thinking at once is the one
        // request that cannot mean anything.
        settings.sub_agent_effort = Some("low".to_string());
        client.apply_settings(&settings);
        let helper = client.for_sub_agent();
        let mut body = json!({"messages": []});
        helper.set_thinking(&mut body);
        assert_eq!(body["reasoning_effort"], "low");
        assert!(body.get("chat_template_kwargs").is_none());

        // And the main client is untouched by any of it.
        let mut body = json!({"messages": []});
        client.set_thinking(&mut body);
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("chat_template_kwargs").is_none());
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

    #[tokio::test]
    async fn plain_completion_trusts_the_reasoning_field_only_once_the_model_is_done() {
        // Empty content, finish_reason "stop": this model always answers through the
        // reasoning field, and that reply is complete, so it is the answer.
        let body = json!({"choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "", "reasoning_content": "the answer"}
        }]})
        .to_string();
        let url = crate::testutil::serve_once(
            Box::leak(body.into_boxed_str()),
            "application/json",
        )
        .await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        let reply = client
            .chat_completion_plain(Caller::Reviewer, vec![json!({"role": "user", "content": "hi"})], Some(1000))
            .await
            .unwrap();
        assert_eq!(reply, "the answer");

        // Empty content, finish_reason "length": max_tokens landed while the model was
        // still inside its own reasoning, so that field holds an unfinished scratchpad,
        // not an answer. Trusting it verbatim is how a compressor's thinking about how to
        // summarize ends up stored as the summary itself, so this must fail instead.
        let body = json!({"choices": [{
            "finish_reason": "length",
            "message": {"role": "assistant", "content": "", "reasoning_content": "still working out how to"}
        }]})
        .to_string();
        let url = crate::testutil::serve_once(
            Box::leak(body.into_boxed_str()),
            "application/json",
        )
        .await;
        let client = crate::LMStudioClient::new(url, "m".to_string(), None);
        assert!(client
            .chat_completion_plain(Caller::Reviewer, vec![json!({"role": "user", "content": "hi"})], Some(1000))
            .await
            .is_err());
    }

}
