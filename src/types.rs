use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Model family detection ────────────────────────────────────────────────

/// The family of a model, detected from its name string.
///
/// This is used to select the correct tool-calling format and prompt template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFamily {
    /// Google Gemma family (gemma, gemma2, gemma3, gemma4, functiongemma).
    Gemma,
    /// Alibaba Qwen family (qwen, qwen2, qwen2.5, qwen3, qwq).
    Qwen,
    /// Meta Llama family (llama2, llama3, llama3.1, llama3.2, llama3.3).
    Llama,
    /// Mistral AI family (mistral, mixtral, mistral-nemo, mistral-small).
    Mistral,
    /// DeepSeek family (deepseek, deepseek-r1, deepseek-v2, deepseek-v3).
    DeepSeek,
    /// Phi family (phi, phi3, phi4).
    Phi,
    /// Command-R family (command-r, command-r-plus).
    CommandR,
    /// Unknown / generic model.
    Unknown,
}

impl ModelFamily {
    /// Detect the model family from a model name string.
    ///
    /// The name is lowercased before matching, so `"Gemma3:27b"` and
    /// `"gemma3:27b"` both return [`ModelFamily::Gemma`].
    pub fn from_model_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        // Check most specific prefixes first to avoid false positives.
        if lower.contains("deepseek") {
            Self::DeepSeek
        } else if lower.contains("functiongemma") || lower.contains("gemma") {
            Self::Gemma
        } else if lower.contains("qwq") || lower.contains("qwen") {
            Self::Qwen
        } else if lower.contains("llama") {
            Self::Llama
        } else if lower.contains("mistral") || lower.contains("mixtral") {
            Self::Mistral
        } else if lower.contains("phi") {
            Self::Phi
        } else if lower.contains("command-r") || lower.contains("command_r") {
            Self::CommandR
        } else {
            Self::Unknown
        }
    }

    /// Returns `true` if this model family uses prompt-injected tool calling
    /// (i.e. tool definitions must be embedded as text in the system prompt
    /// rather than passed as the native `tools` field).
    pub fn uses_prompt_injected_tools(&self) -> bool {
        matches!(self, Self::Gemma)
    }

    /// Returns the recommended [`ToolFormat`] for this model family.
    pub fn tool_format(&self) -> ToolFormat {
        match self {
            Self::Gemma => ToolFormat::PromptInjectedJson,
            Self::Qwen => ToolFormat::HermesXml,
            Self::Llama => ToolFormat::NativeFunctionTag,
            Self::Mistral | Self::CommandR => ToolFormat::Native,
            Self::DeepSeek => ToolFormat::Native,
            Self::Phi => ToolFormat::Native,
            Self::Unknown => ToolFormat::Native,
        }
    }
}

/// The wire format used to communicate tool calls to/from a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolFormat {
    /// Standard Ollama native tool-calling (passed as `tools` JSON field).
    /// The model returns tool calls in the `tool_calls` field of the response.
    Native,
    /// Tool definitions are injected into the system prompt as a JSON array.
    /// The model returns a JSON object in the `content` field.
    /// Used by Gemma.
    PromptInjectedJson,
    /// Tool definitions are injected into the system prompt using Hermes XML
    /// `<tools>` tags. The model returns `<tool_call>` XML in content.
    /// Used by Qwen.
    HermesXml,
    /// Tool definitions are injected into the system prompt as JSON.
    /// The model returns `<function=name>{args}</function>` in content.
    /// Used by Llama 3.x.
    NativeFunctionTag,
}

// ─── Model info ────────────────────────────────────────────────────────────

/// Metadata about a specific model instance.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The full model name as returned by Ollama (e.g. `"qwen2.5:14b"`).
    pub name: String,
    /// The detected model family.
    pub family: ModelFamily,
    /// The recommended tool-calling format for this model.
    pub tool_format: ToolFormat,
    /// Whether this model supports extended thinking / reasoning output.
    pub supports_thinking: bool,
    /// Approximate context window in tokens (0 = unknown).
    pub context_length: u32,
}

impl ModelInfo {
    /// Build a `ModelInfo` from a model name, using heuristics.
    pub fn from_name(name: &str) -> Self {
        let family = ModelFamily::from_model_name(name);
        let tool_format = family.tool_format();
        let lower = name.to_lowercase();

        let supports_thinking = lower.contains("qwq")
            || lower.contains("deepseek-r1")
            || lower.contains("deepseek_r1")
            || lower.contains(":thinking")
            || lower.contains("-thinking");

        // Rough context-length heuristics based on known model sizes.
        let context_length = if lower.contains("gemma3") || lower.contains("gemma4") {
            if lower.contains("27b") || lower.contains("e4b") || lower.contains("12b") {
                131072
            } else {
                32768
            }
        } else if lower.contains("qwen") {
            if lower.contains("72b") || lower.contains("110b") {
                131072
            } else {
                32768
            }
        } else if lower.contains("llama3") {
            128000
        } else if lower.contains("deepseek") {
            65536
        } else {
            32768
        };

        Self {
            name: name.to_string(),
            family,
            tool_format,
            supports_thinking,
            context_length,
        }
    }
}

// ─── Chat types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Output format: `"json"` or a JSON Schema object for structured output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,

    /// Advanced model options (temperature, top_p, num_ctx, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl ChatRequest {
    /// Create a minimal non-streaming chat request.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: None,
            format: None,
            options: None,
            stream: Some(false),
            keep_alive: None,
        }
    }

    /// Returns a [`ChatRequestBuilder`] for fluent construction.
    pub fn builder(model: impl Into<String>) -> ChatRequestBuilder {
        ChatRequestBuilder::new(model)
    }
}

/// Fluent builder for [`ChatRequest`] with model-aware tool injection.
#[derive(Debug, Default)]
pub struct ChatRequestBuilder {
    model: String,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    format: Option<Value>,
    options: Option<Value>,
    stream: Option<bool>,
    keep_alive: Option<String>,
}

impl ChatRequestBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn format(mut self, format: Value) -> Self {
        self.format = Some(format);
        self
    }

    pub fn options(mut self, options: Value) -> Self {
        self.options = Some(options);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Build the [`ChatRequest`], automatically injecting tools into the system
    /// prompt for models that require it (e.g. Gemma).
    pub fn build(self) -> ChatRequest {
        let family = ModelFamily::from_model_name(&self.model);
        let tool_format = family.tool_format();

        let (messages, native_tools) = if self.tools.is_empty() {
            (self.messages, None)
        } else {
            match tool_format {
                ToolFormat::Native => (self.messages, Some(self.tools)),
                ToolFormat::PromptInjectedJson => {
                    let messages =
                        inject_tools_as_json_prompt(self.messages, &self.tools);
                    (messages, None)
                }
                ToolFormat::HermesXml => {
                    let messages =
                        inject_tools_as_hermes_xml(self.messages, &self.tools);
                    (messages, None)
                }
                ToolFormat::NativeFunctionTag => {
                    let messages =
                        inject_tools_as_function_tag(self.messages, &self.tools);
                    (messages, None)
                }
            }
        };

        ChatRequest {
            model: self.model,
            messages,
            tools: native_tools,
            format: self.format,
            options: self.options,
            stream: self.stream,
            keep_alive: self.keep_alive,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String, // "system", "user", "assistant", "tool"
    pub content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            name: None,
            images: None,
            thinking: None,
            tool_calls: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            name: None,
            images: None,
            thinking: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            name: None,
            images: None,
            thinking: None,
            tool_calls: None,
        }
    }

    pub fn tool_result(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            name: Some(name.into()),
            images: None,
            thinking: None,
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tool {
    pub r#type: String, // typically "function"
    pub function: FunctionDefinition,
}

impl Tool {
    /// Convenience constructor for a function tool.
    pub fn function(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value, // JSON schema object
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

impl ChatResponse {
    /// Extract tool calls from this response, checking both the native
    /// `tool_calls` field and falling back to parsing the `content` field
    /// for models that emit tool calls as plain text (Gemma, Llama, etc.).
    pub fn extract_tool_calls(&self) -> Vec<ToolCall> {
        // 1. Native tool_calls field (Ollama standard, Qwen, Mistral, etc.)
        // Filter out any entries with empty names — Ollama sometimes parses
        // Gemma's malformed `{}` outputs as ToolCall entries with empty names.
        if let Some(tool_calls) = &self.message.tool_calls {
            let valid: Vec<ToolCall> = tool_calls
                .iter()
                .filter(|tc| !tc.function.name.trim().is_empty())
                .cloned()
                .collect();
            if !valid.is_empty() {
                return valid;
            }
        }

        // 2. Hermes XML format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        // Used by Qwen when Ollama doesn't parse it automatically.
        if let Some(calls) = parse_hermes_tool_calls(&self.message.content) {
            if !calls.is_empty() {
                return calls;
            }
        }

        // 3. Gemma JSON format: {"name": "...", "parameters": {...}}
        // or {"name": "...", "arguments": {...}}
        if let Some(calls) = parse_json_tool_call(&self.message.content) {
            if !calls.is_empty() {
                return calls;
            }
        }

        // 4. Llama function tag format: <function=name>{args}</function>
        if let Some(calls) = parse_function_tag_tool_calls(&self.message.content) {
            if !calls.is_empty() {
                return calls;
            }
        }

        Vec::new()
    }

    /// Returns `true` if this response contains any tool calls (native or
    /// parsed from content).
    pub fn has_tool_calls(&self) -> bool {
        !self.extract_tool_calls().is_empty()
    }
}

// ─── Embed types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbedInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbedRequest {
    pub model: String,
    pub input: EmbedInput,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u64>,
}

// ─── Model list types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelListResponse {
    pub models: Vec<Model>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub families: Option<Vec<String>>,
    pub parameter_size: String,
    pub quantization_level: String,
}

// ─── Tool injection helpers ────────────────────────────────────────────────

/// Inject tool definitions as a JSON array into the system prompt.
///
/// This is the format required by Gemma models, which do not support the
/// native Ollama `tools` field.
///
/// The injected text follows Google's recommended format:
/// ```text
/// You have access to functions. If you decide to invoke any of the function(s),
/// you MUST put it in the format of
/// {"name": function name, "parameters": dictionary of argument name and its value}
///
/// You SHOULD NOT include any other text in the response if you call a function
///
/// [{"name": "...", "description": "...", "parameters": {...}}, ...]
/// ```
pub fn inject_tools_as_json_prompt(mut messages: Vec<Message>, tools: &[Tool]) -> Vec<Message> {
    if tools.is_empty() {
        return messages;
    }

    let tool_defs: Vec<Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            })
        })
        .collect();

    let tool_json = serde_json::to_string_pretty(&tool_defs).unwrap_or_default();

    let injection = format!(
        "You have access to functions. If you decide to invoke any of the function(s), \
you MUST put it in the format of\n\
{{\"name\": function name, \"parameters\": dictionary of argument name and its value}}\n\n\
You SHOULD NOT include any other text in the response if you call a function\n\n\
{tool_json}"
    );

    // Prepend to existing system message or insert a new one at the front.
    if let Some(sys) = messages.iter_mut().find(|m| m.role == "system") {
        sys.content = format!("{injection}\n\n{}", sys.content);
    } else {
        messages.insert(0, Message::system(injection));
    }

    messages
}

/// Inject tool definitions using Hermes XML format into the system prompt.
///
/// Used by Qwen models when Ollama doesn't handle the Hermes template
/// automatically.
///
/// ```text
/// # Tools
///
/// You may call one or more functions to assist with the user query.
///
/// <tools>
/// {"type": "function", "function": {"name": "...", ...}}
/// </tools>
/// ```
pub fn inject_tools_as_hermes_xml(mut messages: Vec<Message>, tools: &[Tool]) -> Vec<Message> {
    if tools.is_empty() {
        return messages;
    }

    let tool_lines: Vec<String> = tools
        .iter()
        .map(|t| serde_json::to_string(t).unwrap_or_default())
        .collect();

    let tools_block = tool_lines.join("\n");

    let injection = format!(
        "# Tools\n\n\
You may call one or more functions to assist with the user query.\n\n\
You are provided with function signatures within <tools></tools> XML tags:\n\
<tools>\n\
{tools_block}\n\
</tools>"
    );

    if let Some(sys) = messages.iter_mut().find(|m| m.role == "system") {
        sys.content = format!("{}\n\n{injection}", sys.content);
    } else {
        messages.insert(0, Message::system(injection));
    }

    messages
}

/// Inject tool definitions for Llama-style `<function=name>` format.
pub fn inject_tools_as_function_tag(mut messages: Vec<Message>, tools: &[Tool]) -> Vec<Message> {
    if tools.is_empty() {
        return messages;
    }

    let tool_lines: Vec<String> = tools
        .iter()
        .map(|t| {
            let schema = serde_json::to_string(&t.function.parameters).unwrap_or_default();
            format!(
                "Use the function '{}' to: {}\n{}",
                t.function.name, t.function.description, schema
            )
        })
        .collect();

    let tools_block = tool_lines.join("\n\n");

    let injection = format!(
        "You have access to the following functions:\n\n\
{tools_block}\n\n\
If a function is called, return ONLY the function call in this exact format:\n\
<function=FUNCTION_NAME>{{\"param\": \"value\"}}</function>"
    );

    if let Some(sys) = messages.iter_mut().find(|m| m.role == "system") {
        sys.content = format!("{injection}\n\n{}", sys.content);
    } else {
        messages.insert(0, Message::system(injection));
    }

    messages
}

// ─── Tool call parsing helpers ─────────────────────────────────────────────

/// Parse Hermes-style XML tool calls from a content string.
///
/// Looks for patterns like:
/// ```text
/// <tool_call>
/// {"name": "get_weather", "arguments": {"location": "London"}}
/// </tool_call>
/// ```
pub fn parse_hermes_tool_calls(content: &str) -> Option<Vec<ToolCall>> {
    if !content.contains("<tool_call>") {
        return None;
    }

    let mut calls = Vec::new();
    let mut remaining = content;

    while let Some(start) = remaining.find("<tool_call>") {
        let after_open = &remaining[start + "<tool_call>".len()..];
        if let Some(end) = after_open.find("</tool_call>") {
            let json_str = after_open[..end].trim();
            if let Ok(value) = serde_json::from_str::<Value>(json_str) {
                if let Some(call) = value_to_tool_call(value) {
                    calls.push(call);
                }
            }
            remaining = &after_open[end + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    if calls.is_empty() { None } else { Some(calls) }
}

/// Parse a Gemma-style JSON tool call from a content string.
///
/// Gemma outputs either:
/// - `{"name": "fn", "parameters": {...}}`
/// - `{"name": "fn", "arguments": {...}}`
///
/// The content may contain only the JSON object, or have surrounding text.
pub fn parse_json_tool_call(content: &str) -> Option<Vec<ToolCall>> {
    let trimmed = content.trim();

    // Quick pre-filter: if the content doesn't contain a "name" key at all,
    // there is no point trying to parse it as a tool call.  This avoids
    // treating bare `{}` or code snippets as tool calls.
    if !trimmed.contains("\"name\"") {
        return None;
    }

    // Try the whole content as a single JSON object first.
    if trimmed.starts_with('{') {
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            if let Some(call) = value_to_tool_call(value) {
                return Some(vec![call]);
            }
        }
    }

    // Try to find a JSON object anywhere in the content.
    if let Some(start) = trimmed.find('{') {
        let candidate = &trimmed[start..];
        // Find the matching closing brace.
        let mut depth = 0i32;
        let mut end = None;
        for (i, ch) in candidate.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end) = end {
            let json_str = &candidate[..end];
            if let Ok(value) = serde_json::from_str::<Value>(json_str) {
                if let Some(call) = value_to_tool_call(value) {
                    return Some(vec![call]);
                }
            }
        }
    }

    None
}

/// Parse Llama-style `<function=name>{args}</function>` tool calls.
pub fn parse_function_tag_tool_calls(content: &str) -> Option<Vec<ToolCall>> {
    if !content.contains("<function=") {
        return None;
    }

    let mut calls = Vec::new();
    let mut remaining = content;

    while let Some(start) = remaining.find("<function=") {
        let after_tag = &remaining[start + "<function=".len()..];
        if let Some(name_end) = after_tag.find('>') {
            let name = after_tag[..name_end].trim().to_string();
            let after_name = &after_tag[name_end + 1..];
            if let Some(close) = after_name.find("</function>") {
                let args_str = after_name[..close].trim();
                let arguments = serde_json::from_str::<Value>(args_str)
                    .unwrap_or(Value::Object(serde_json::Map::new()));
                calls.push(ToolCall {
                    id: None,
                    function: FunctionCall {
                        index: None,
                        name,
                        arguments,
                    },
                });
                remaining = &after_name[close + "</function>".len()..];
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if calls.is_empty() { None } else { Some(calls) }
}

/// Convert a JSON `Value` to a `ToolCall` if it looks like a tool call object.
///
/// Supports both `{"name": ..., "arguments": ...}` and
/// `{"name": ..., "parameters": ...}` formats.
fn value_to_tool_call(value: Value) -> Option<ToolCall> {
    let obj = value.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();

    // Reject empty or whitespace-only names — Gemma sometimes outputs `{}`
    // or `{"name": ""}` before it finds the correct format.
    if name.trim().is_empty() {
        return None;
    }

    // Reject names that look like JSON fragments (Gemma sometimes outputs
    // the whole tool schema as the name field).
    if name.trim_start().starts_with('{') || name.trim_start().starts_with('[') {
        return None;
    }

    let arguments = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    Some(ToolCall {
        id: None,
        function: FunctionCall {
            index: None,
            name,
            arguments,
        },
    })
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_model_family_detection() {
        assert_eq!(ModelFamily::from_model_name("gemma3:27b"), ModelFamily::Gemma);
        assert_eq!(ModelFamily::from_model_name("gemma4:latest"), ModelFamily::Gemma);
        assert_eq!(ModelFamily::from_model_name("functiongemma:270m"), ModelFamily::Gemma);
        assert_eq!(ModelFamily::from_model_name("qwen2.5:14b"), ModelFamily::Qwen);
        assert_eq!(ModelFamily::from_model_name("qwen3:8b"), ModelFamily::Qwen);
        assert_eq!(ModelFamily::from_model_name("qwq:32b"), ModelFamily::Qwen);
        assert_eq!(ModelFamily::from_model_name("llama3.3:70b"), ModelFamily::Llama);
        assert_eq!(ModelFamily::from_model_name("llama3.2:3b"), ModelFamily::Llama);
        assert_eq!(ModelFamily::from_model_name("mistral:7b"), ModelFamily::Mistral);
        assert_eq!(ModelFamily::from_model_name("mixtral:8x7b"), ModelFamily::Mistral);
        assert_eq!(ModelFamily::from_model_name("deepseek-r1:7b"), ModelFamily::DeepSeek);
        assert_eq!(ModelFamily::from_model_name("phi4:latest"), ModelFamily::Phi);
        assert_eq!(ModelFamily::from_model_name("unknown-model:latest"), ModelFamily::Unknown);
    }

    #[test]
    fn test_gemma_uses_prompt_injected_tools() {
        assert!(ModelFamily::Gemma.uses_prompt_injected_tools());
        assert!(!ModelFamily::Qwen.uses_prompt_injected_tools());
        assert!(!ModelFamily::Llama.uses_prompt_injected_tools());
    }

    #[test]
    fn test_parse_hermes_tool_calls() {
        let content = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "London"}}
</tool_call>"#;
        let calls = parse_hermes_tool_calls(content).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["location"], "London");
    }

    #[test]
    fn test_parse_hermes_multiple_tool_calls() {
        let content = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "London"}}
</tool_call>
<tool_call>
{"name": "get_time", "arguments": {"timezone": "UTC"}}
</tool_call>"#;
        let calls = parse_hermes_tool_calls(content).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn test_parse_json_tool_call_gemma_format() {
        // Gemma uses "parameters" key
        let content = r#"{"name": "get_weather", "parameters": {"location": "London"}}"#;
        let calls = parse_json_tool_call(content).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["location"], "London");
    }

    #[test]
    fn test_parse_json_tool_call_with_surrounding_text() {
        let content = r#"I'll call the weather function: {"name": "get_weather", "parameters": {"location": "Paris"}}"#;
        let calls = parse_json_tool_call(content).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_function_tag_tool_calls() {
        let content = r#"<function=get_weather>{"location": "London"}</function>"#;
        let calls = parse_function_tag_tool_calls(content).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["location"], "London");
    }

    #[test]
    fn test_inject_tools_as_json_prompt_no_existing_system() {
        let tools = vec![Tool::function(
            "get_weather",
            "Gets the weather",
            json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        )];
        let messages = vec![Message::user("What's the weather?")];
        let result = inject_tools_as_json_prompt(messages, &tools);
        assert_eq!(result[0].role, "system");
        assert!(result[0].content.contains("get_weather"));
        assert!(result[0].content.contains("You have access to functions"));
    }

    #[test]
    fn test_inject_tools_as_json_prompt_prepends_to_existing_system() {
        let tools = vec![Tool::function(
            "get_weather",
            "Gets the weather",
            json!({"type": "object", "properties": {}}),
        )];
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("What's the weather?"),
        ];
        let result = inject_tools_as_json_prompt(messages, &tools);
        assert_eq!(result[0].role, "system");
        assert!(result[0].content.contains("You have access to functions"));
        assert!(result[0].content.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_chat_request_builder_gemma_injects_tools() {
        let tools = vec![Tool::function(
            "search",
            "Search the web",
            json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        )];
        let request = ChatRequest::builder("gemma3:27b")
            .messages(vec![Message::user("Search for Rust")])
            .tools(tools)
            .build();

        // For Gemma, tools should be injected into system prompt, not native field.
        assert!(request.tools.is_none());
        assert_eq!(request.messages[0].role, "system");
        assert!(request.messages[0].content.contains("search"));
    }

    #[test]
    fn test_chat_request_builder_qwen_uses_native_tools() {
        let tools = vec![Tool::function(
            "search",
            "Search the web",
            json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        )];
        let request = ChatRequest::builder("qwen2.5:14b")
            .messages(vec![Message::user("Search for Rust")])
            .tools(tools)
            .build();

        // Qwen via Ollama uses native tool_calls, Ollama handles Hermes template.
        // But we inject as Hermes XML for safety when Ollama doesn't handle it.
        // The native tools field should be None (we inject into system prompt).
        assert!(request.tools.is_none());
    }

    #[test]
    fn test_extract_tool_calls_from_native_field() {
        let response = ChatResponse {
            model: "qwen2.5:14b".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            message: Message {
                role: "assistant".to_string(),
                content: String::new(),
                name: None,
                images: None,
                thinking: None,
                tool_calls: Some(vec![ToolCall {
                    id: Some("call_1".to_string()),
                    function: FunctionCall {
                        index: None,
                        name: "get_weather".to_string(),
                        arguments: json!({"location": "London"}),
                    },
                }]),
            },
            done: true,
            done_reason: Some("tool_calls".to_string()),
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
        };

        let calls = response.extract_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_extract_tool_calls_from_content_gemma() {
        let response = ChatResponse {
            model: "gemma3:27b".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            message: Message {
                role: "assistant".to_string(),
                content: r#"{"name": "get_weather", "parameters": {"location": "London"}}"#
                    .to_string(),
                name: None,
                images: None,
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

        let calls = response.extract_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["location"], "London");
    }
}
