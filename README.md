# ollama_client_rs

`ollama_client_rs` is a transport-focused Rust SDK for the [Ollama](https://ollama.com/) API.

It provides:

- typed request and response models
- synchronous chat requests
- streaming chat requests
- embeddings
- model listing
- lightweight telemetry hooks

This crate does not own orchestration, planning, RAG, or tool-loop behavior. Those higher-level workflows should live in the application layer, such as RAIN.

## Basic usage

```rust
use ollama_client_rs::{
    types::{ChatRequest, Message},
    OllamaClient,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::default();
    let response = client
        .chat(&ChatRequest {
            model: "gemma4:latest".to_string(),
            messages: vec![Message::user("Summarize this project in two sentences.")],
            ..Default::default()
        })
        .await?;

    println!("{}", response.message.content.unwrap_or_default());
    Ok(())
}
```

## Position in the stack

- Use `ollama_client_rs` when you want a low-level SDK for Ollama.
- Use RAIN when you want agentic execution, tool orchestration, retrieval, planning, or multi-step workflows.
