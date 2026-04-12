use reqwest::Client;
use serde_json::Value;

pub mod agentic;
mod telemetry;
pub mod types;

use std::future::Future;
use std::pin::Pin;

use types::{ChatRequest, ChatResponse};

type SyncFunctionHandler = Box<dyn Fn(&mut Value) -> Result<Value, String> + Send + Sync>;
type AsyncFunctionHandler = Box<
    dyn Fn(&mut Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> + Send + Sync,
>;

/// A handler for tool/function calls that can be either sync or async.
pub enum FunctionHandler {
    /// A synchronous function handler.
    Sync(SyncFunctionHandler),
    /// An asynchronous function handler.
    Async(AsyncFunctionHandler),
}

impl FunctionHandler {
    /// Executes the handler, automatically handling whether it's sync or async.
    pub async fn execute(&self, params: &mut Value) -> Result<Value, String> {
        match self {
            FunctionHandler::Sync(handler) => handler(params),
            FunctionHandler::Async(handler) => handler(params).await,
        }
    }
}

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
    #[error("Function execution error: {0}")]
    FunctionExecution(String),
    #[error("Tool loop exceeded the maximum number of round trips ({max_round_trips})")]
    LoopLimitExceeded { max_round_trips: usize },
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

    /// Send a chat request with automatic function-calling loop support.
    ///
    /// The function handlers are used to execute any tool calls that the model
    /// requests. The loop continues until the model produces a final text
    /// response or the round-trip limit is exceeded.
    pub async fn chat_with_function_calling(
        &self,
        request: ChatRequest,
        function_handlers: &agentic::tool_runtime::ToolRegistry,
    ) -> Result<ChatResponse, OllamaError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "ollama_client_rs.chat_with_function_calling",
            model = request.model.as_str(),
            messages_count = request.messages.len(),
            tools_count = request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            function_handler_count = function_handlers.len()
        );
        crate::telemetry::telemetry_info!("chat_with_function_calling started");
        
        let toolbox = agentic::tool_runtime::Toolbox::empty();
        let tool_view = agentic::tool_runtime::ToolRegistryView::all(&toolbox, function_handlers);
        let result = match agentic::tool_runtime::execute_tool_loop(
            self,
            request,
            Some(&tool_view),
            &agentic::tool_runtime::ToolRuntimeConfig::default(),
        )
        .await
        {
            Ok(result) => result,
            Err(error) => {
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    function_handler_count = function_handlers.len(),
                    "chat_with_function_calling failed"
                );
                return Err(error);
            }
        };

        crate::telemetry::telemetry_info!(
            round_trips = result.trace.round_trips,
            tool_call_count = result.trace.calls.len(),
            "chat_with_function_calling completed"
        );

        Ok(result.response)
    }
}
