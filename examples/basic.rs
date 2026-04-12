use ollama_client_rs::{
    types::{ChatRequest, Message},
    OllamaClient,
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // The client defaults to http://127.0.0.1:11434/api if OLLAMA_HOST is not set.
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    println!("Connecting to Ollama at {}...", ollama_url);

    let client = OllamaClient::new(ollama_url);

    // 1. Basic List Models
    println!("=== Available Models ===");
    let models = client.list_models().await?;
    for model in models.iter() {
        println!("- {}", model.name);
    }

    // Attempt to use a known model if available, otherwise use the first one
    let target_model = if models.iter().any(|m| m.name.starts_with("gemma4")) {
        "gemma4:e2b".to_string()
    } else if let Some(first) = models.first() {
        first.name.clone()
    } else {
        println!("No models found in Ollama instance.");
        return Ok(());
    };

    println!("\n=== Target Model: {} ===", target_model);

    // 2. Chat completion
    println!("\n=== Chat Request ===");
    let request = ChatRequest {
        model: target_model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Why is the sky blue?".to_string(),
            name: None,
            images: None,
            tool_calls: None,
        }],
        tools: None,
        format: None,
        options: None,
        stream: Some(false),
        keep_alive: None,
    };

    let response = client.chat(&request).await?;
    println!("Response:\n{}", response.message.content);

    // 3. Structured JSON Request
    println!("\n=== Structured JSON Request ===");
    let json_request = ChatRequest {
        model: target_model.clone(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful culinary assistant.".to_string(),
                name: None,
                images: None,
                tool_calls: None,
            },
            Message {
                role: "user".to_string(),
                content: "Give me a recipe for chocolate chip cookies in JSON format.".to_string(),
                name: None,
                images: None,
                tool_calls: None,
            },
        ],
        tools: None,
        format: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "ingredients": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "instructions": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["name", "ingredients", "instructions"]
        })),
        options: None,
        stream: Some(false),
        keep_alive: None,
    };

    let json_response = client.chat(&json_request).await?;
    println!("JSON Response:\n{}", json_response.message.content);

    Ok(())
}
