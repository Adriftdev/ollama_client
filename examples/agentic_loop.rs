/// Agentic loop example: a simple ReAct-style agent that uses tools iteratively.
///
/// This example demonstrates a full agent loop:
///   1. Send user message + tools to the model
///   2. If the model calls a tool, dispatch it and add the result
///   3. Repeat until the model gives a final answer (no more tool calls)
///
/// Works with any model — uses `ChatRequestBuilder` which automatically
/// selects the correct tool format for the detected model family.
///
/// Run with:
/// ```
/// OLLAMA_MODEL=qwen2.5:7b cargo run --example agentic_loop
/// ```
use ollama_client_rs::{
    types::{ChatRequest, Message, ModelInfo, Tool},
    OllamaClient,
};
use serde_json::json;
use std::collections::HashMap;
use std::env;

// ── Tool registry ─────────────────────────────────────────────────────────

type ToolFn = Box<dyn Fn(&serde_json::Value) -> String + Send + Sync>;

fn build_tool_registry() -> HashMap<String, ToolFn> {
    let mut registry: HashMap<String, ToolFn> = HashMap::new();

    registry.insert(
        "read_file".to_string(),
        Box::new(|args| {
            let path = args["path"].as_str().unwrap_or("unknown");
            // Simulated file read.
            match path {
                "config.toml" => r#"[server]
host = "localhost"
port = 8080
workers = 4

[database]
url = "postgres://localhost/mydb"
pool_size = 10"#
                    .to_string(),
                "README.md" => "# My Project\n\nA sample Rust project.\n\n## Setup\n\nRun `cargo build`.".to_string(),
                _ => format!("Error: file not found: {path}"),
            }
        }),
    );

    registry.insert(
        "list_directory".to_string(),
        Box::new(|args| {
            let path = args["path"].as_str().unwrap_or(".");
            match path {
                "." | "./" => json!({
                    "files": ["Cargo.toml", "README.md", "config.toml"],
                    "directories": ["src", "tests", "examples"]
                })
                .to_string(),
                "src" => json!({
                    "files": ["main.rs", "lib.rs", "config.rs"],
                    "directories": []
                })
                .to_string(),
                _ => format!("{{\"error\": \"directory not found: {path}\"}}"),
            }
        }),
    );

    registry.insert(
        "run_command".to_string(),
        Box::new(|args| {
            let cmd = args["command"].as_str().unwrap_or("");
            match cmd {
                "cargo check" => "Compiling my_project v0.1.0\n    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.34s".to_string(),
                "cargo test" => "running 5 tests\ntest test_config ... ok\ntest test_server ... ok\ntest result: ok. 5 passed; 0 failed".to_string(),
                _ => format!("Error: command not allowed: {cmd}"),
            }
        }),
    );

    registry
}

fn build_tools() -> Vec<Tool> {
    vec![
        Tool::function(
            "read_file",
            "Read the contents of a file",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to read" }
                },
                "required": ["path"]
            }),
        ),
        Tool::function(
            "list_directory",
            "List files and directories in a path",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory path to list" }
                },
                "required": ["path"]
            }),
        ),
        Tool::function(
            "run_command",
            "Run a shell command (limited to safe commands)",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run. Allowed: 'cargo check', 'cargo test'"
                    }
                },
                "required": ["command"]
            }),
        ),
    ]
}

// ── Agent loop ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    let client = OllamaClient::new(ollama_url);

    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());
    let info = ModelInfo::from_name(&model);
    println!("Agent using model: {} ({:?})\n", info.name, info.family);

    let tools = build_tools();
    let tool_registry = build_tool_registry();

    let task = "Explore the project structure, read the config file, and run the tests. \
                Give me a summary of what you found.";

    println!("Task: {task}\n");
    println!("{}", "─".repeat(60));

    let mut messages = vec![
        Message::system(
            "You are a helpful software engineering assistant. Use the available tools \
             to explore the project and answer questions. Be thorough but concise.",
        ),
        Message::user(task),
    ];

    const MAX_TURNS: usize = 10;
    let mut turn = 0;

    loop {
        turn += 1;
        if turn > MAX_TURNS {
            println!("\n[Agent] Reached maximum turns ({MAX_TURNS}). Stopping.");
            break;
        }

        println!("\n[Turn {turn}] Thinking...");

        let request = ChatRequest::builder(&model)
            .messages(messages.clone())
            .tools(tools.clone())
            .build();

        let response = client.chat(&request).await?;
        let tool_calls = response.extract_tool_calls();

        if tool_calls.is_empty() {
            // No tool calls — this is the final answer.
            println!("\n[Agent] Final answer:\n");
            println!("{}", response.message.content);
            break;
        }

        // Show thinking text if any.
        if !response.message.content.is_empty() {
            let preview = if response.message.content.len() > 150 {
                format!("{}...", &response.message.content[..150])
            } else {
                response.message.content.clone()
            };
            println!("[Agent] Thinking: {preview}");
        }

        // Add the assistant's tool-call message to history.
        messages.push(response.message.clone());

        // Dispatch each tool call.
        for call in &tool_calls {
            println!(
                "[Tool] {}({})",
                call.function.name,
                serde_json::to_string(&call.function.arguments).unwrap_or_default()
            );

            let result = if let Some(tool_fn) = tool_registry.get(&call.function.name) {
                tool_fn(&call.function.arguments)
            } else {
                format!("{{\"error\": \"unknown tool: {}\"}}", call.function.name)
            };

            let preview = if result.len() > 120 {
                format!("{}...", &result[..120])
            } else {
                result.clone()
            };
            println!("[Tool] Result: {preview}");

            messages.push(Message::tool_result(&call.function.name, result));
        }
    }

    println!("\n{}", "─".repeat(60));
    println!("Agent completed in {turn} turn(s).");
    Ok(())
}
