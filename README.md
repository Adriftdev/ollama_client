# Ollama Client RS

`ollama_client_rs` is a Rust-based, local-first SDK for interacting with [Ollama](https://ollama.com/), providing idiomatic Rust bindings for chat, functional calling, and advanced agentic workflows such as **supervisor orchestration**, **app-owned RAG**, and **bounded planning cycles**.

This project shares the exact same philosophy and workflow architectures as the [gemini_client](https://github.com/Adriftdev/gemini-client) toolkit, enabling developers to build state-of-the-art agentic behaviors entirely locally without relying on external or paid cloud APIs.

## Features

- **Idiomatic Client**: Native, fully async `OllamaClient` using `reqwest` and `tokio`.
- **Function Calling Framework**: High-level tooling for deterministic and cyclical tool loops, supporting both `Sync` and `Async` closures.
- **Auto-Retrying Execution**: Configurable round-trip loop limits that automatically submit model function request formats and auto-append returned results.
- **Bounded Planner**: An agent mode specifically requesting JSON format responses for formulating executing step-by-step sequential plans, handling mid-flight adjustments.
- **Supervisor Workflow**: Multi-agent topological loop implementing deterministically bound assignments across worker, reviewer, and synthesizer capabilities. 
- **App-Owned RAG**: Built-in retrieval abstraction for simple chunk selection, truncation, and validation with robust citation checks.

## Usage

Start your local Ollama instance (e.g. `llama3.1` is recommended for high functioning tool capabilities) and then experiment with the crate.

### Basic Chat

```rust
use ollama_client_rs::{OllamaClient, types::ChatRequest, agentic::build_user_message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::new("http://127.0.0.1:11434/api".to_string());
    
    let request = ChatRequest {
        model: "llama3.1:8b".to_string(),
        messages: vec![build_user_message("Why is the sky blue?")],
        tools: None,
        format: None,
        options: None,
        stream: Some(false),
        keep_alive: None,
    };

    let response = client.chat(&request).await?;
    println!("Response: {}", response.message.content);
    Ok(())
}
```

## Examples

To view complex agentic integrations run the following examples available in the `/examples` directory:

```sh
# Basic chat and structured JSON
cargo run --example basic

# Continuous Tool execution
cargo run --example custom_tool

# Multi-Agent orchestrator using assignments, workflow blackboard and revisions
cargo run --example supervisor_workflow

# Step-by-step plan verification and execution loop
cargo run --example plan_and_execute
```

## Telemetry
This project utilizes `tracing` for thorough and advanced logging. Set `RUST_LOG=debug` or `RUST_LOG=info` for deeper insights into internal states across long-running tool execution or planning loops.

## License
MIT
