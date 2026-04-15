/// Tool calling example: native function calling for Qwen / Mistral models.
///
/// This example demonstrates how to define tools, send a request that triggers
/// a tool call, dispatch the call, and send the result back for a final answer.
///
/// Run with:
/// ```
/// OLLAMA_MODEL=qwen2.5:7b cargo run --example tool_calling
/// ```
use ollama_client_rs::{
    types::{ChatRequest, Message, Tool},
    OllamaClient,
};
use serde_json::json;
use std::env;

// ── Simulated tool implementations ────────────────────────────────────────

fn get_weather(location: &str, unit: &str) -> String {
    // In a real application this would call a weather API.
    format!(
        "{{\"location\": \"{location}\", \"temperature\": 18, \"unit\": \"{unit}\", \
         \"condition\": \"partly cloudy\", \"humidity\": 72}}"
    )
}

fn get_time(timezone: &str) -> String {
    format!("{{\"timezone\": \"{timezone}\", \"time\": \"14:32\", \"date\": \"2025-04-15\"}}")
}

// ── Tool definitions ───────────────────────────────────────────────────────

fn weather_tool() -> Tool {
    Tool::function(
        "get_weather",
        "Get the current weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. 'London, UK'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }),
    )
}

fn time_tool() -> Tool {
    Tool::function(
        "get_time",
        "Get the current time in a given timezone",
        json!({
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'Europe/London'"
                }
            },
            "required": ["timezone"]
        }),
    )
}

// ── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    let client = OllamaClient::new(ollama_url);

    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());
    println!("Using model: {model}");
    println!("Note: this example works best with qwen2.5:7b or larger.\n");

    let tools = vec![weather_tool(), time_tool()];

    // ── Turn 1: user asks a question that requires tool use ────────────────
    let mut messages = vec![
        Message::system(
            "You are a helpful assistant. Use the available tools to answer questions accurately.",
        ),
        Message::user("What's the weather like in Tokyo right now, and what time is it there?"),
    ];

    println!("User: {}\n", messages.last().unwrap().content);

    let request = ChatRequest::builder(&model)
        .messages(messages.clone())
        .tools(tools.clone())
        .build();

    let response = client.chat(&request).await?;

    // The builder automatically handles tool injection for the model family.
    // `extract_tool_calls()` checks both the native field and content fallback.
    let tool_calls = response.extract_tool_calls();

    if tool_calls.is_empty() {
        // Model answered directly without tool use.
        println!("Assistant (direct): {}", response.message.content);
        return Ok(());
    }

    println!("Assistant wants to call {} tool(s):", tool_calls.len());

    // ── Dispatch each tool call ────────────────────────────────────────────
    // Add the assistant's tool-call message to the conversation.
    messages.push(response.message.clone());

    for call in &tool_calls {
        println!("  → {}({:?})", call.function.name, call.function.arguments);

        let result = match call.function.name.as_str() {
            "get_weather" => {
                let location = call.function.arguments["location"]
                    .as_str()
                    .unwrap_or("unknown");
                let unit = call.function.arguments["unit"]
                    .as_str()
                    .unwrap_or("celsius");
                get_weather(location, unit)
            }
            "get_time" => {
                let tz = call.function.arguments["timezone"]
                    .as_str()
                    .unwrap_or("UTC");
                get_time(tz)
            }
            other => format!("{{\"error\": \"unknown tool: {other}\"}}"),
        };

        println!("  ← {result}");

        // Add the tool result to the conversation.
        messages.push(Message::tool_result(&call.function.name, result));
    }

    // ── Turn 2: send tool results back for the final answer ────────────────
    let request = ChatRequest::builder(&model)
        .messages(messages)
        .tools(tools)
        .build();

    let final_response = client.chat(&request).await?;
    println!("\nAssistant: {}", final_response.message.content);

    Ok(())
}
