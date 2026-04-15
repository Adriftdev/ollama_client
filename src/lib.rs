use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::pin::Pin;

mod telemetry;
pub mod types;

use types::{ChatRequest, ChatResponse, EmbedRequest, EmbedResponse};

pub type ChatResponseStream = Pin<Box<dyn Stream<Item = Result<ChatResponse, OllamaError>> + Send>>;

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
            api_url: "http://127.0.0.1:11434/api".to_string(), // Adjust based on env if desired, defaulting to localhost
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
    ///
    /// This is useful for testing purposes or connecting to a remote Ollama
    /// instance.
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

        // Ensure stream is false for a standard chat query if not set by caller
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

        let chat_response: ChatResponse = match response.json().await {
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

        crate::telemetry::telemetry_info!(
            "chat completed"
        );

        Ok(chat_response)
    }

    /// Send a streaming chat request to the Ollama `/api/chat` endpoint.
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

        let bytes = response.bytes_stream();
        let stream = try_stream! {
            let mut buffer = String::new();
            futures_util::pin_mut!(bytes);

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(OllamaError::Http)?;
                let text = std::str::from_utf8(&chunk)
                    .map_err(|error| OllamaError::Api(serde_json::json!({"error": error.to_string()})))?;
                buffer.push_str(text);

                while let Some(index) = buffer.find('\n') {
                    let line = buffer[..index].trim().to_string();
                    buffer.drain(..=index);
                    if line.is_empty() {
                        continue;
                    }
                    crate::telemetry::telemetry_debug!("chat_stream processing chunk");
                    let chunk = serde_json::from_str::<ChatResponse>(&line).map_err(|error| {
                        OllamaError::Json {
                            data: line.clone(),
                            error,
                        }
                    })?;
                    yield chunk;
                }
            }

            let tail = buffer.trim().to_string();
            if !tail.is_empty() {
                let chunk = serde_json::from_str::<ChatResponse>(&tail).map_err(|error| {
                    OllamaError::Json {
                        data: tail.clone(),
                        error,
                    }
                })?;
                yield chunk;
            }
        };

        Ok(Box::pin(stream))
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
