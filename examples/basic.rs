use futures_util::StreamExt as _;
/// Basic example: list models, simple chat, and structured JSON output.
///
/// Run with:
/// ```
/// OLLAMA_HOST=http://127.0.0.1:11434/api cargo run --example basic
/// ```
use ollama_client_rs::{chat_request, messages, types::Options, OllamaClient, StreamChunk};
use serde_json::json;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The client defaults to http://127.0.0.1:11434/api if OLLAMA_HOST is not set.
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    println!("Connecting to Ollama at {}...\n", ollama_url);

    let client = OllamaClient::new(ollama_url);

    // ── 1. List available models ──────────────────────────────────────────
    println!("=== Available Models ===");
    let models = client.list_models().await?;
    if models.is_empty() {
        println!("No models found. Pull one first: `ollama pull qwen2.5:7b`");
        return Ok(());
    }
    for model in &models {
        println!("  • {} ({})", model.name, model.details.parameter_size);
    }

    // Pick a model to use for the rest of the examples.
    let target_model = models
        .iter()
        .find(|m| m.name.contains("qwen") || m.name.contains("gemma4") || m.name.contains("llama"))
        .map(|m| m.name.clone())
        .unwrap_or_else(|| models[0].name.clone());

    println!("\nUsing model: {target_model}\n");

    // ── 2. Simple chat using the new helper constructors ──────────────────
    println!("=== Simple Chat ===");

    let request = chat_request! {
        model: &target_model,
        messages: messages!(
            system: "You are a concise assistant. Answer in one sentence.",
            user: "Why is the sky blue?"
        ),
        options: serde_json::to_value(Options::gemma4_optimal(32768)).unwrap(),
    };

    let mut response = client.chat_stream_parsed(&request).await?;

    while let Some(parsed) = response.next().await {
        match parsed {
            Ok(chunk) => match chunk {
                StreamChunk::Reasoning(reasoning) => {
                    print!("{}", reasoning);
                }
                StreamChunk::Content(content) => {
                    print!("{}", content);
                }
            },
            Err(e) => eprintln!("Error parsing response: {e}"),
        }
    }

    // ── 3. Fluent builder API ─────────────────────────────────────────────
    println!("\n=== Builder API ===");

    let request = chat_request! {
        model: &target_model,
        messages: messages!(
            system: "You are a helpful culinary assistant.",
            user: "Give me a recipe for chocolate chip cookies in JSON format."
        ),
        format: json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "ingredients": { "type": "array", "items": { "type": "string" } },
                "instructions": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["name", "ingredients", "instructions"]
        }),
        options: serde_json::to_value(Options::gemma4_optimal(32768)).unwrap(),

    };

    let mut response = client.chat_stream_parsed(&request).await?;
    while let Some(parsed) = response.next().await {
        match parsed {
            Ok(chunk) => match chunk {
                StreamChunk::Reasoning(reasoning) => {
                    print!("{}", reasoning);
                }
                StreamChunk::Content(content) => {
                    print!("{}", content);
                }
            },
            Err(e) => eprintln!("Error parsing response: {e}"),
        }
    }

    // ── 4. Model info heuristics ──────────────────────────────────────────
    println!("\n=== Model Info ===");
    let info = client.model_info(&target_model).await;
    println!("  Name            : {}", info.name);
    println!("  Family          : {:?}", info.family);
    println!("  Tool format     : {:?}", info.tool_format);
    println!("  Supports think  : {}", info.supports_thinking);
    println!("  Context length  : {} tokens", info.context_length);

    Ok(())
}
