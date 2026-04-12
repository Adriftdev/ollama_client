use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    types::{ChatRequest, ChatResponse, Message, Tool},
    FunctionHandler, OllamaClient, OllamaError,
};

pub type ToolRegistry = HashMap<String, FunctionHandler>;

#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, OllamaError>;
}

#[async_trait]
impl ModelBackend for OllamaClient {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, OllamaError> {
        OllamaClient::chat(self, request).await
    }
}

#[derive(Debug, Clone)]
pub struct ToolRuntimeConfig {
    pub max_round_trips: usize,
    pub allow_parallel_calls: bool,
}

impl Default for ToolRuntimeConfig {
    fn default() -> Self {
        Self {
            max_round_trips: 8,
            allow_parallel_calls: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallRecord {
    pub round_trip: usize,
    pub id: Option<String>,
    pub name: String,
    pub arguments: Value,
    pub response: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolTrace {
    pub round_trips: usize,
    pub calls: Vec<ToolCallRecord>,
}

#[derive(Debug, Clone)]
pub struct ToolRunResult {
    pub response: ChatResponse,
    pub trace: ToolTrace,
}

#[derive(Debug, Clone)]
pub struct Toolbox {
    tools: Vec<Tool>,
}

impl Toolbox {
    pub fn new(tools: Vec<Tool>) -> Self {
        Self { tools }
    }

    pub fn empty() -> Self {
        Self { tools: vec![] }
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    pub fn available_tool_names(&self, handlers: &ToolRegistry) -> Vec<String> {
        let mut names = self
            .tools
            .iter()
            .map(|t| t.function.name.clone())
            .collect::<Vec<_>>();

        for name in handlers.keys() {
            if !names.iter().any(|existing| existing == name) {
                names.push(name.clone());
            }
        }

        names.sort();
        names
    }

    pub fn missing_allowed_names(
        &self,
        handlers: &ToolRegistry,
        allowed_names: &[String],
    ) -> Vec<String> {
        let known = self
            .available_tool_names(handlers)
            .into_iter()
            .collect::<HashSet<_>>();

        allowed_names
            .iter()
            .filter(|name| !known.contains(*name))
            .cloned()
            .collect()
    }

    pub fn select_tools(&self, allowed_names: &[String]) -> Vec<Tool> {
        if allowed_names.is_empty() {
            return vec![];
        }

        let allowed = allowed_names.iter().cloned().collect::<HashSet<_>>();
        self.tools
            .iter()
            .filter(|tool| allowed.contains(&tool.function.name))
            .cloned()
            .collect()
    }

    pub fn filter_handlers<'a>(
        &'a self,
        handlers: &'a ToolRegistry,
        allowed_names: &[String],
    ) -> ToolRegistryView<'a> {
        if allowed_names.is_empty() {
            return ToolRegistryView::empty(self.tools.clone());
        }

        let allowed = allowed_names.iter().cloned().collect::<HashSet<_>>();
        let filtered_handlers = handlers
            .iter()
            .filter(|(name, _)| allowed.contains(*name))
            .map(|(name, handler)| (name.as_str(), handler))
            .collect::<HashMap<_, _>>();

        ToolRegistryView {
            tools: self.select_tools(allowed_names),
            handlers: filtered_handlers,
        }
    }
}

pub struct AgentTools {
    pub toolbox: Toolbox,
    pub handlers: ToolRegistry,
}

impl AgentTools {
    pub fn new(tools: Vec<Tool>, handlers: ToolRegistry) -> Self {
        Self {
            toolbox: Toolbox::new(tools),
            handlers,
        }
    }

    pub fn available_tool_names(&self) -> Vec<String> {
        self.toolbox.available_tool_names(&self.handlers)
    }

    pub fn missing_allowed_names(&self, allowed_names: &[String]) -> Vec<String> {
        self.toolbox
            .missing_allowed_names(&self.handlers, allowed_names)
    }

    pub fn all(&self) -> ToolRegistryView<'_> {
        ToolRegistryView::all(&self.toolbox, &self.handlers)
    }

    pub fn select(&self, allowed_names: &[String]) -> ToolRegistryView<'_> {
        self.toolbox.filter_handlers(&self.handlers, allowed_names)
    }
}

pub struct ToolRegistryView<'a> {
    tools: Vec<Tool>,
    handlers: HashMap<&'a str, &'a FunctionHandler>,
}

impl<'a> ToolRegistryView<'a> {
    pub fn empty(tools: Vec<Tool>) -> Self {
        Self {
            tools,
            handlers: HashMap::new(),
        }
    }

    pub fn all(toolbox: &'a Toolbox, handlers: &'a ToolRegistry) -> Self {
        Self {
            tools: toolbox.tools.clone(),
            handlers: handlers
                .iter()
                .map(|(name, handler)| (name.as_str(), handler))
                .collect(),
        }
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    fn handler(&self, name: &str) -> Option<&FunctionHandler> {
        self.handlers.get(name).copied()
    }
}

pub async fn execute_tool_loop<B: ModelBackend>(
    backend: &B,
    mut request: ChatRequest,
    tools: Option<&ToolRegistryView<'_>>,
    config: &ToolRuntimeConfig,
) -> Result<ToolRunResult, OllamaError> {
    let _span = crate::telemetry::telemetry_span_guard!(
        info,
        "ollama_client_rs.tool_loop",
        model = request.model.as_str(),
        max_round_trips = config.max_round_trips,
        request_tools_count = request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        has_tool_view = tools.is_some()
    );
    crate::telemetry::telemetry_info!("tool_loop started");

    if request.tools.is_none() || request.tools.as_ref().unwrap().is_empty() {
        if let Some(tool_view) = tools {
            request.tools = Some(tool_view.tools().to_vec());
            crate::telemetry::telemetry_debug!(
                request_tools_count = tool_view.tools().len(),
                "tool_loop populated request tools from tool view"
            );
        }
    }

    let mut trace = ToolTrace::default();

    for round_trip in 1..=config.max_round_trips {
        crate::telemetry::telemetry_debug!(round_trip, "tool_loop round trip started");
        let response = backend.chat(&request).await?;

        let tool_calls = match &response.message.tool_calls {
            Some(calls) if !calls.is_empty() => calls.clone(),
            _ => {
                trace.round_trips = round_trip;
                crate::telemetry::telemetry_info!(
                    round_trips = trace.round_trips,
                    total_calls = trace.calls.len(),
                    "tool_loop completed without further tool calls"
                );
                return Ok(ToolRunResult { response, trace });
            }
        };

        let Some(tool_view) = tools else {
            let error = OllamaError::FunctionExecution(
                "Model requested tool calls but no tool handlers were provided".to_string(),
            );
            crate::telemetry::telemetry_warn!(
                error_kind = crate::telemetry::ollama_error_kind(&error),
                round_trip,
                "tool_loop missing tool handlers"
            );
            return Err(error);
        };

        // Persist the assistant message containing the tool_calls
        request.messages.push(response.message.clone());

        for call in tool_calls {
            let function_call = &call.function;
            let Some(handler) = tool_view.handler(&function_call.name) else {
                let error = OllamaError::FunctionExecution(format!(
                    "Unknown function: {}",
                    function_call.name
                ));
                crate::telemetry::telemetry_warn!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    tool_name = function_call.name.as_str(),
                    round_trip,
                    "tool_loop unknown tool requested"
                );
                return Err(error);
            };

            crate::telemetry::telemetry_debug!(
                tool_name = function_call.name.as_str(),
                round_trip,
                "tool_loop executing function call"
            );
            
            let mut arguments = function_call.arguments.clone();
            let result = handler.execute(&mut arguments).await.map_err(|error| {
                let error = OllamaError::FunctionExecution(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::ollama_error_kind(&error),
                    tool_name = function_call.name.as_str(),
                    round_trip,
                    "tool_loop handler execution failed"
                );
                error
            })?;

            trace.calls.push(ToolCallRecord {
                round_trip,
                id: None,
                name: function_call.name.clone(),
                arguments,
                response: result.clone(),
            });

            // Push the tool response into context
            request.messages.push(Message {
                role: "tool".to_string(),
                content: match result {
                    Value::String(s) => s,
                    _ => serde_json::to_string(&result).unwrap_or_default(),
                },
                name: Some(function_call.name.clone()),
                images: None,
                tool_calls: None,
            });
        }
    }

    let error = OllamaError::LoopLimitExceeded {
        max_round_trips: config.max_round_trips,
    };
    crate::telemetry::telemetry_warn!(
        error_kind = crate::telemetry::ollama_error_kind(&error),
        max_round_trips = config.max_round_trips,
        total_calls = trace.calls.len(),
        "tool_loop exceeded round trip limit"
    );
    Err(error)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{execute_tool_loop, ToolRegistry, ToolRegistryView, ToolRuntimeConfig, Toolbox};
    use crate::{
        agentic::test_support::{
            response_with_text, response_with_tool_calls, ScriptedBackend,
        },
        types::{FunctionCall, FunctionDefinition, Tool, ToolCall},
        FunctionHandler, OllamaError,
    };

    fn weather_toolbox() -> (Toolbox, ToolRegistry) {
        let tools = vec![
            Tool {
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    name: "get_weather".to_string(),
                    description: "Fetches weather".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "location": { "type": "string" }
                        },
                        "required": ["location"]
                    }),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    name: "get_timezone".to_string(),
                    description: "Fetches timezone".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "location": { "type": "string" }
                        },
                        "required": ["location"]
                    }),
                },
            },
        ];

        let mut handlers = ToolRegistry::new();
        handlers.insert(
            "get_weather".to_string(),
            FunctionHandler::Sync(Box::new(|args| {
                Ok(json!({ "forecast": format!("sunny in {}", args["location"]) }))
            })),
        );
        handlers.insert(
            "get_timezone".to_string(),
            FunctionHandler::Sync(Box::new(|args| {
                Ok(json!({ "timezone": format!("tz for {}", args["location"]) }))
            })),
        );

        (Toolbox::new(tools), handlers)
    }

    fn tool_call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            function: FunctionCall {
                name: name.to_string(),
                arguments: args,
            },
        }
    }

    #[tokio::test]
    async fn executes_multiple_tool_calls() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|request| {
                assert_eq!(request.messages.len(), 1);
                Ok(response_with_tool_calls(vec![
                    tool_call("get_weather", json!({"location": "London"})),
                    tool_call("get_timezone", json!({"location": "London"})),
                ]))
            }),
            Box::new(|request| {
                // 1 original user + 1 assistant (tool_calls) + 2 tool responses
                assert_eq!(request.messages.len(), 4);
                Ok(response_with_text("done"))
            }),
        ]);
        let (toolbox, handlers) = weather_toolbox();
        let selection = ToolRegistryView::all(&toolbox, &handlers);

        let request = crate::types::ChatRequest {
            model: "test".to_string(),
            messages: vec![crate::agentic::build_user_message("Help me")],
            tools: None,
            format: None,
            options: None,
            stream: Some(false),
            keep_alive: None,
        };

        let result = execute_tool_loop(
            &backend,
            request,
            Some(&selection),
            &ToolRuntimeConfig::default(),
        )
        .await
        .expect("tool loop should succeed");

        assert_eq!(result.trace.calls.len(), 2);
        assert_eq!(result.trace.round_trips, 2);
    }

    #[tokio::test]
    async fn returns_unknown_tool_error_when_handler_is_missing() {
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_tool_calls(vec![
                tool_call("missing_tool", json!({})),
            ]))
        })]);
        let toolbox = Toolbox::empty();
        let handlers = ToolRegistry::new();
        let selection = ToolRegistryView::all(&toolbox, &handlers);

        let request = crate::types::ChatRequest {
            model: "test".to_string(),
            messages: vec![crate::agentic::build_user_message("Help me")],
            tools: None,
            format: None,
            options: None,
            stream: Some(false),
            keep_alive: None,
        };

        let error = execute_tool_loop(
            &backend,
            request,
            Some(&selection),
            &ToolRuntimeConfig::default(),
        )
        .await
        .expect_err("unknown tool should fail");

        assert!(matches!(
            error,
            OllamaError::FunctionExecution(message) if message.contains("missing_tool")
        ));
    }

    #[tokio::test]
    async fn propagates_handler_failures() {
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_tool_calls(vec![
                tool_call("get_weather", json!({"location": "London"})),
            ]))
        })]);
        let tools = vec![Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: "Fetches weather".to_string(),
                parameters: json!({}),
            },
        }];
        let mut handlers = ToolRegistry::new();
        handlers.insert(
            "get_weather".to_string(),
            FunctionHandler::Sync(Box::new(|_| Err("upstream failed".to_string()))),
        );
        let toolbox = Toolbox::new(tools);
        let selection = ToolRegistryView::all(&toolbox, &handlers);

        let request = crate::types::ChatRequest {
            model: "test".to_string(),
            messages: vec![crate::agentic::build_user_message("Help me")],
            tools: None,
            format: None,
            options: None,
            stream: Some(false),
            keep_alive: None,
        };

        let error = execute_tool_loop(
            &backend,
            request,
            Some(&selection),
            &ToolRuntimeConfig::default(),
        )
        .await
        .expect_err("handler failures should propagate");

        assert!(matches!(
            error,
            OllamaError::FunctionExecution(message) if message.contains("upstream failed")
        ));
    }

    #[tokio::test]
    async fn enforces_the_tool_loop_limit() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_tool_calls(vec![
                    tool_call("get_weather", json!({"location": "London"})),
                ]))
            }),
            Box::new(|_| {
                Ok(response_with_tool_calls(vec![
                    tool_call("get_weather", json!({"location": "London"})),
                ]))
            }),
        ]);
        let (toolbox, handlers) = weather_toolbox();
        let selection = ToolRegistryView::all(&toolbox, &handlers);

        let request = crate::types::ChatRequest {
            model: "test".to_string(),
            messages: vec![crate::agentic::build_user_message("Help me")],
            tools: None,
            format: None,
            options: None,
            stream: Some(false),
            keep_alive: None,
        };

        let error = execute_tool_loop(
            &backend,
            request,
            Some(&selection),
            &ToolRuntimeConfig {
                max_round_trips: 1,
                allow_parallel_calls: false,
            },
        )
        .await
        .expect_err("loop limit should be enforced");

        assert!(matches!(
            error,
            OllamaError::LoopLimitExceeded { max_round_trips: 1 }
        ));
    }
}
