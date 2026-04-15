/// Gemma tool calling example: prompt-injected tool definitions.
///
/// Gemma models do not support the native Ollama `tools` field. This example
/// shows how `ChatRequestBuilder` automatically injects tool definitions into
/// the system prompt and how `ChatResponse::extract_tool_calls()` parses the
/// model's JSON response back into structured `ToolCall` objects.
///
/// Run with:
/// ```
/// OLLAMA_MODEL=gemma3:4b cargo run --example gemma_tools
/// ```
use ollama_client_rs::{
    types::{ChatRequest, Message, ModelFamily, ModelInfo, Tool},
    OllamaClient,
};
use serde_json::json;
use std::env;

fn calculator_tool() -> Tool {
    Tool::function(
        "calculate",
        "Perform a mathematical calculation",
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g. '2 + 2 * 3'"
                }
            },
            "required": ["expression"]
        }),
    )
}

fn search_tool() -> Tool {
    Tool::function(
        "web_search",
        "Search the web for current information",
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }),
    )
}

fn dispatch_tool(name: &str, args: &serde_json::Value) -> String {
    match name {
        "calculate" => {
            let expr = args["expression"].as_str().unwrap_or("0");
            // Simulated evaluation — in production use a real math library.
            format!("{{\"result\": \"evaluated: {expr}\", \"note\": \"simulated\"}}")
        }
        "web_search" => {
            let query = args["query"].as_str().unwrap_or("");
            format!(
                "{{\"results\": [\"Result 1 for '{query}'\", \"Result 2 for '{query}'\"], \
                 \"note\": \"simulated\"}}"
            )
        }
        other => format!("{{\"error\": \"unknown tool: {other}\"}}"),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    let client = OllamaClient::new(ollama_url);

    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma3:4b".to_string());

    // Show model info so we can confirm the family was detected correctly.
    let info = ModelInfo::from_name(&model);
    println!("Model         : {}", info.name);
    println!("Family        : {:?}", info.family);
    println!("Tool format   : {:?}", info.tool_format);
    println!(
        "Prompt inject : {}\n",
        info.family.uses_prompt_injected_tools()
    );

    if info.family != ModelFamily::Gemma {
        println!(
            "Warning: this example is designed for Gemma models. \
             For {}, consider using the tool_calling example instead.",
            info.family.tool_format().as_name()
        );
    }

    let tools = vec![calculator_tool(), search_tool()];

    // The builder detects Gemma and automatically injects tools into the
    // system prompt instead of passing them as the native `tools` field.
    let mut messages = vec![
        Message::system("You are a helpful assistant with access to tools."),
        Message::user("What is 42 * 17, and can you search for the latest Rust news?"),
    ];

    println!("User: {}\n", messages.last().unwrap().content);

    let request = ChatRequest::builder(&model)
        .messages(messages.clone())
        .tools(tools.clone())
        .build();

    // Verify that tools were injected into the system prompt (not native field).
    assert!(
        request.tools.is_none(),
        "Gemma should not use native tools field"
    );
    println!(
        "System prompt (first 200 chars):\n  {}\n",
        &request.messages[0].content[..request.messages[0].content.len().min(200)]
    );

    let response = client.chat(&request).await?;
    println!("Raw response content:\n  {}\n", response.message.content);

    // extract_tool_calls() handles Gemma's JSON format automatically.
    let tool_calls = response.extract_tool_calls();

    if tool_calls.is_empty() {
        println!("No tool calls detected. Model replied directly:\n{}", response.message.content);
        return Ok(());
    }

    println!("Detected {} tool call(s):", tool_calls.len());
    messages.push(response.message.clone());

    for call in &tool_calls {
        println!("  → {}({:?})", call.function.name, call.function.arguments);
        let result = dispatch_tool(&call.function.name, &call.function.arguments);
        println!("  ← {result}");
        messages.push(Message::tool_result(&call.function.name, result));
    }

    // Send tool results back for the final answer.
    let request = ChatRequest::builder(&model)
        .messages(messages)
        .tools(tools)
        .build();

    let final_response = client.chat(&request).await?;
    println!("\nFinal answer:\n{}", final_response.message.content);

    Ok(())
}

// Helper to get a human-readable name for ToolFormat.
trait ToolFormatName {
    fn as_name(&self) -> &'static str;
}

impl ToolFormatName for ollama_client_rs::types::ToolFormat {
    fn as_name(&self) -> &'static str {
        match self {
            ollama_client_rs::types::ToolFormat::Native => "Native",
            ollama_client_rs::types::ToolFormat::PromptInjectedJson => "PromptInjectedJson",
            ollama_client_rs::types::ToolFormat::HermesXml => "HermesXml",
            ollama_client_rs::types::ToolFormat::NativeFunctionTag => "NativeFunctionTag",
        }
    }
}
