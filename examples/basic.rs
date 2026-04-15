/// Basic example: list models, simple chat, and structured JSON output.
///
/// Run with:
/// ```
/// OLLAMA_HOST=http://127.0.0.1:11434/api cargo run --example basic
/// ```
use ollama_client_rs::{
    types::{ChatRequest, Message},
    OllamaClient,
};
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
        .find(|m| m.name.contains("qwen") || m.name.contains("gemma") || m.name.contains("llama"))
        .map(|m| m.name.clone())
        .unwrap_or_else(|| models[0].name.clone());

    println!("\nUsing model: {target_model}\n");

    // ── 2. Simple chat using the new helper constructors ──────────────────
    println!("=== Simple Chat ===");
    let request = ChatRequest::new(
        &target_model,
        vec![
            Message::system("You are a concise assistant. Answer in one sentence."),
            Message::user("Why is the sky blue?"),
        ],
    );

    let response = client.chat(&request).await?;
    println!("Reply: {}", response.message.content);
    if let (Some(prompt_tokens), Some(output_tokens)) =
        (response.prompt_eval_count, response.eval_count)
    {
        println!("Tokens: {prompt_tokens} prompt + {output_tokens} output");
    }

    // ── 3. Fluent builder API ─────────────────────────────────────────────
    println!("\n=== Builder API ===");
    let request = ChatRequest::builder(&target_model)
        .message(Message::system("You are a helpful culinary assistant."))
        .message(Message::user(
            "Give me a recipe for chocolate chip cookies in JSON format.",
        ))
        .format(serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "ingredients": { "type": "array", "items": { "type": "string" } },
                "instructions": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["name", "ingredients", "instructions"]
        }))
        .build();

    let response = client.chat(&request).await?;
    println!("JSON recipe:\n{}", response.message.content);

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
