use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::pin::Pin;

mod telemetry;
pub mod types;
pub mod macros;
pub mod tools;

pub use tools::{OllamaTool, ToolRegistry};
pub use ollama_client_macros::OllamaTool;

pub use types::{
    ChatRequest, ChatRequestBuilder, ChatResponse, EmbedInput, EmbedRequest, EmbedResponse,
    FunctionCall, FunctionDefinition, Message, Model, ModelDetails, ModelFamily, ModelInfo,
    ToolCall, ToolFormat, Tool, StreamChunk,
    inject_tools_as_function_tag, inject_tools_as_hermes_xml, inject_tools_as_json_prompt,
    parse_function_tag_tool_calls, parse_hermes_tool_calls, parse_json_tool_call,
};

pub type ChatResponseStream = Pin<Box<dyn Stream<Item = Result<ChatResponse, OllamaError>> + Send>>;
pub type ParsedStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, OllamaError>> + Send>>;

#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("HTTP Error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API Error: {0}")]
    Api(Value),
    #[error("JSON Error: {error} (payload: {data})")]
    Json {
        data: String,
        #[source]
        error: serde_json::Error,
    },
}

impl OllamaError {
    async fn from_response(
        response: reqwest::Response,
        context: Option<serde_json::Value>,
    ) -> Self {
        let status = response.status();
        let text = match response.text().await {
            Ok(text) => text,
            Err(error) => return Self::Http(error),
        };
        let message = match serde_json::from_str::<Value>(&text) {
            Ok(error) => error,
            Err(_) => serde_json::Value::String(text),
        };

        Self::Api(serde_json::json!({
            "status": status.as_u16(),
            "message": message,
            "context": context.unwrap_or_default(),
        }))
    }
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    http_client: Client,
    api_url: String,
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self {
            http_client: Client::new(),
            api_url: "http://127.0.0.1:11434/api".to_string(),
        }
    }
}

impl OllamaClient {
    /// Create a new Ollama client.
    ///
    /// Pass the base API URL (e.g. `http://127.0.0.1:11434/api`).
    /// For the default localhost URL, use [`OllamaClient::default()`] instead.
    pub fn new(api_url: String) -> Self {
        OllamaClient {
            api_url,
            ..Default::default()
        }
    }

    /// Provide a pre-configured [`reqwest::Client`] to use for the Ollama
    /// client.
    ///
    /// This can be used to configure things like timeouts, proxies, etc.
    pub fn with_client(mut self, http_client: Client) -> Self {
        self.http_client = http_client;
        self
    }

    /// Set the API URL for the Ollama client.
    pub fn with_api_url(mut self, api_url: String) -> Self {
        self.api_url = api_url;
        self
    }

    /// List all available models.
    pub async fn list_models(&self) -> Result<Vec<types::Model>, OllamaError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "ollama_client_rs.list_models"
        );
        crate::telemetry::telemetry_info!("list_models started");

        let url = format!("{}/tags", self.api_url);

        let response = match self.http_client.get(&url).send().await {
            Ok(response) => response,
            Err(error) => {
                let error = OllamaError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    "list_models request failed"
                );
                return Err(error);
            }
        };

        if !response.status().is_success() {
            let error = OllamaError::from_response(response, None).await;
            crate::telemetry::telemetry_error!(
                error_kind = crate::telemetry::ollama_error_kind(&error),
                "list_models API failure"
            );
            return Err(error);
        }

        let response: types::ModelListResponse = match response.json().await {
            Ok(response) => response,
            Err(error) => {
                let error = OllamaError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    "list_models response parsing failed"
                );
                return Err(error);
            }
        };

        crate::telemetry::telemetry_info!(
            model_count = response.models.len(),
            "list_models completed"
        );

        Ok(response.models)
    }

    /// Returns [`ModelInfo`] for the given model name, combining heuristics
    /// with any information available from the Ollama API.
    ///
    /// This is a best-effort call — if the model is not available locally the
    /// heuristic-only `ModelInfo::from_name` result is returned.
    pub async fn model_info(&self, model: &str) -> ModelInfo {
        // Start with heuristics.
        let mut info = ModelInfo::from_name(model);

        // Try to enrich from the Ollama show API.
        let url = format!("{}/show", self.api_url);
        let body = serde_json::json!({ "name": model });
        if let Ok(response) = self.http_client.post(&url).json(&body).send().await {
            if response.status().is_success() {
                if let Ok(value) = response.json::<Value>().await {
                    // Extract context length from model details if available.
                    if let Some(ctx) = value
                        .pointer("/model_info/llama.context_length")
                        .or_else(|| value.pointer("/parameters/num_ctx"))
                        .and_then(|v| v.as_u64())
                    {
                        info.context_length = ctx as u32;
                    }
                }
            }
        }

        info
    }

    /// Pull (download) a model from the Ollama registry.
    ///
    /// Returns a stream of progress events as JSON values.
    pub async fn pull_model(
        &self,
        model: &str,
        insecure: bool,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Value, OllamaError>> + Send>>, OllamaError> {
        let url = format!("{}/pull", self.api_url);
        let body = serde_json::json!({
            "name": model,
            "insecure": insecure,
            "stream": true,
        });

        let response = self
            .http_client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(OllamaError::Http)?;

        if !response.status().is_success() {
            return Err(OllamaError::from_response(response, None).await);
        }

        let bytes = response.bytes_stream();
        let stream = try_stream! {
            let mut buffer = String::new();
            futures_util::pin_mut!(bytes);

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(OllamaError::Http)?;
                let text = std::str::from_utf8(&chunk)
                    .map_err(|e| OllamaError::Api(serde_json::json!({"error": e.to_string()})))?;
                buffer.push_str(text);

                while let Some(idx) = buffer.find('\n') {
                    let line = buffer[..idx].trim().to_string();
                    buffer.drain(..=idx);
                    if line.is_empty() {
                        continue;
                    }
                    let value = serde_json::from_str::<Value>(&line).map_err(|error| {
                        OllamaError::Json { data: line.clone(), error }
                    })?;
                    yield value;
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Send a chat request to the Ollama `/api/chat` endpoint.
    pub async fn chat(
        &self,
        request: &ChatRequest,
    ) -> Result<ChatResponse, OllamaError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "ollama_client_rs.chat",
            model = request.model.as_str(),
            messages_count = request.messages.len(),
            tools_count = request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        );
        crate::telemetry::telemetry_info!("chat started");

        let url = format!("{}/chat", self.api_url);

        // Ensure stream is false for a standard chat query if not set by caller.
        let mut request_clone = request.clone();
        if request_clone.stream.is_none() {
            request_clone.stream = Some(false);
        }

        let response = match self.http_client.post(&url).json(&request_clone).send().await {
            Ok(response) => response,
            Err(error) => {
                let error = OllamaError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    "chat request failed"
                );
                return Err(error);
            }
        };

        if !response.status().is_success() {
            let error = OllamaError::from_response(response, None).await;
            crate::telemetry::telemetry_error!(
                error_kind = crate::telemetry::ollama_error_kind(&error),
                "chat API failure"
            );
            return Err(error);
        }

        let mut chat_response: ChatResponse = match response.json().await {
            Ok(response) => response,
            Err(error) => {
                let error = OllamaError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    "chat response parsing failed"
                );
                return Err(error);
            }
        };

        // Normalise: if tool_calls is empty but content looks like a tool call,
        // parse it and populate tool_calls so callers don't need to check content.
        if chat_response.message.tool_calls.as_ref().is_none_or(|v| v.is_empty()) {
            let parsed = chat_response.extract_tool_calls();
            if !parsed.is_empty() {
                chat_response.message.tool_calls = Some(parsed);
            }
        }

        crate::telemetry::telemetry_info!("chat completed");

        Ok(chat_response)
    }

    /// Send a streaming chat request to the Ollama `/api/chat` endpoint.
    ///
    /// The stream yields [`ChatResponse`] chunks as they arrive. The final
    /// chunk has `done = true` and contains the full usage statistics.
    ///
    /// **Improvement over v0.2.x**: the stream parser now handles chunk
    /// boundaries that fall mid-JSON-line correctly, and emits a synthetic
    /// `done` chunk if the server closes the connection without one.
    pub async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<ChatResponseStream, OllamaError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "ollama_client_rs.chat_stream",
            model = request.model.as_str(),
            messages_count = request.messages.len(),
            tools_count = request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        );
        crate::telemetry::telemetry_info!("chat_stream started");

        let url = format!("{}/chat", self.api_url);
        let mut request_clone = request.clone();
        request_clone.stream = Some(true);

        let response = match self.http_client.post(&url).json(&request_clone).send().await {
            Ok(response) => response,
            Err(error) => {
                let error = OllamaError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    "chat_stream request failed"
                );
                return Err(error);
            }
        };

        if !response.status().is_success() {
            let error = OllamaError::from_response(response, None).await;
            crate::telemetry::telemetry_error!(
                error_kind = crate::telemetry::ollama_error_kind(&error),
                "chat_stream API failure"
            );
            return Err(error);
        }

        let model_name = request.model.clone();
        let bytes = response.bytes_stream();
        let stream = try_stream! {
            let mut buffer = String::new();
            let mut saw_done = false;
            futures_util::pin_mut!(bytes);

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(OllamaError::Http)?;
                let text = std::str::from_utf8(&chunk)
                    .map_err(|e| OllamaError::Api(serde_json::json!({"error": e.to_string()})))?;
                buffer.push_str(text);

                // Process all complete newline-terminated lines in the buffer.
                while let Some(idx) = buffer.find('\n') {
                    let line = buffer[..idx].trim().to_string();
                    buffer.drain(..=idx);
                    if line.is_empty() {
                        continue;
                    }
                    crate::telemetry::telemetry_debug!("chat_stream processing chunk");
                    let mut chunk = serde_json::from_str::<ChatResponse>(&line).map_err(|error| {
                        OllamaError::Json {
                            data: line.clone(),
                            error,
                        }
                    })?;

                    // Normalise tool calls from content for models that don't
                    // use the native tool_calls field.
                    if chunk.done {
                        saw_done = true;
                        if chunk.message.tool_calls.as_ref().is_none_or(|v| v.is_empty()) {
                            let parsed = chunk.extract_tool_calls();
                            if !parsed.is_empty() {
                                chunk.message.tool_calls = Some(parsed);
                            }
                        }
                    }

                    yield chunk;
                }
            }

            // Handle any remaining data in the buffer (no trailing newline).
            let tail = buffer.trim().to_string();
            if !tail.is_empty() {
                let mut chunk = serde_json::from_str::<ChatResponse>(&tail).map_err(|error| {
                    OllamaError::Json {
                        data: tail.clone(),
                        error,
                    }
                })?;
                if chunk.done {
                    saw_done = true;
                    if chunk.message.tool_calls.as_ref().is_none_or(|v| v.is_empty()) {
                        let parsed = chunk.extract_tool_calls();
                        if !parsed.is_empty() {
                            chunk.message.tool_calls = Some(parsed);
                        }
                    }
                }
                yield chunk;
            }

            // If the server closed the connection without a done=true chunk,
            // emit a synthetic one so callers always see a terminal event.
            if !saw_done {
                crate::telemetry::telemetry_debug!("chat_stream emitting synthetic done chunk");
                yield ChatResponse {
                    model: model_name,
                    created_at: String::new(),
                    message: types::Message {
                        role: "assistant".to_string(),
                        content: String::new(),
                        name: None,
                        images: None,
                        audio: None,
                        video_frames: None,
                        thinking: None,
                        tool_calls: None,
                    },
                    done: true,
                    done_reason: Some("stop".to_string()),
                    total_duration: None,
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: None,
                    eval_duration: None,
                };
            }
        };

        Ok(Box::pin(stream))
    }

    /// Send a streaming chat request to the Ollama `/api/chat` endpoint and parses the stream
    /// into reasoning and content chunks. This is useful for dealing with Gemma 4's "Thinking"
    /// mode tokens dynamically.
    pub async fn chat_stream_parsed(
        &self,
        request: &ChatRequest,
    ) -> Result<ParsedStream, OllamaError> {
        let stream = self.chat_stream(request).await?;
        let parsed_stream = try_stream! {
            let mut is_thinking = false;
            let mut thinking_buffer = String::new();
            
            futures_util::pin_mut!(stream);
            
            while let Some(chunk) = stream.next().await {
                let response = chunk?;
                let content = response.message.content;
                
                let mut current_idx = 0;
                while current_idx < content.len() {
                    let slice = &content[current_idx..];
                    
                    if is_thinking {
                        if let Some(end_idx) = slice.find("<channel|>") {
                            let part = &slice[..end_idx];
                            thinking_buffer.push_str(part);
                            if !thinking_buffer.is_empty() {
                                yield StreamChunk::Reasoning(thinking_buffer.clone());
                                thinking_buffer.clear();
                            }
                            is_thinking = false;
                            current_idx += end_idx + "<channel|>".len();
                        } else {
                            thinking_buffer.push_str(slice);
                            // Yield incrementally, or buffer
                            yield StreamChunk::Reasoning(thinking_buffer.clone());
                            thinking_buffer.clear();
                            break;
                        }
                    } else {
                        if let Some(start_idx) = slice.find("<|channel>thought\n") {
                            let part = &slice[..start_idx];
                            if !part.is_empty() {
                                yield StreamChunk::Content(part.to_string());
                            }
                            is_thinking = true;
                            current_idx += start_idx + "<|channel>thought\n".len();
                        } else {
                            if !slice.is_empty() {
                                yield StreamChunk::Content(slice.to_string());
                            }
                            break;
                        }
                    }
                }
            }
        };

        Ok(Box::pin(parsed_stream))
    }

    /// Send an embedding request to the Ollama `/api/embed` endpoint.
    pub async fn embed(
        &self,
        request: &EmbedRequest,
    ) -> Result<EmbedResponse, OllamaError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "ollama_client_rs.embed",
            model = request.model.as_str(),
        );
        crate::telemetry::telemetry_info!("embed started");

        let url = format!("{}/embed", self.api_url);

        let response = match self.http_client.post(&url).json(request).send().await {
            Ok(response) => response,
            Err(error) => {
                let error = OllamaError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    "embed request failed"
                );
                return Err(error);
            }
        };

        if !response.status().is_success() {
            let error = OllamaError::from_response(response, None).await;
            crate::telemetry::telemetry_error!(
                error_kind = crate::telemetry::ollama_error_kind(&error),
                "embed API failure"
            );
            return Err(error);
        }

        let embed_response: EmbedResponse = match response.json().await {
            Ok(response) => response,
            Err(error) => {
                let error = OllamaError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    "embed response parsing failed"
                );
                return Err(error);
            }
        };

        crate::telemetry::telemetry_info!(
            embeddings_count = embed_response.embeddings.len(),
            "embed completed"
        );

        Ok(embed_response)
    }
}
