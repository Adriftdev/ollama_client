use ollama_client_rs::{
    agentic::tool_runtime::{execute_tool_loop, AgentTools, ToolRuntimeConfig},
    types::{ChatRequest, FunctionDefinition, Message, Tool},
    FunctionHandler, OllamaClient,
};
use serde_json::{json, Value};
use std::{env, sync::Arc};
use tokio::sync::Mutex;

// Here is our mock state simulating an email system
#[derive(Default)]
struct EmailService {
    outbox: Vec<(String, String)>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    println!("Connecting to Ollama at {}...", ollama_url);

    let client = OllamaClient::new(ollama_url);

    // We should pick a capable tool calling model
    let target_model = "gemma4:e2b".to_string();

    // 1. Define the shared state we want to act on
    let service = Arc::new(Mutex::new(EmailService::default()));

    // 2. Define the tool spec (What the model sees)
    let send_email_spec = Tool {
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: "send_email".to_string(),
            description: "Sends an email to the specified recipient.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "to": { "type": "string", "description": "The email address of the recipient" },
                    "body": { "type": "string", "description": "The contents of the email" }
                },
                "required": ["to", "body"]
            }),
        },
    };

    // 3. Define the tool implementation
    let service_clone = service.clone();
    let handler = FunctionHandler::Async(Box::new(move |args: &mut Value| {
        let service_ref = service_clone.clone();
        let args_val = args.clone();
        Box::pin(async move {
            let to = args_val["to"].as_str().unwrap_or("").to_string();
            let body = args_val["body"].as_str().unwrap_or("").to_string();

            if to.is_empty() || body.is_empty() {
                return Ok(json!({ "error": "Missing required fields 'to' or 'body'." }));
            }

            println!("\n>> [TOOL EXECUTED] Sending email to {}...", to);
            let mut svc = service_ref.lock().await;
            svc.outbox.push((to.clone(), body));

            Ok(json!({
                "status": "success",
                "message": format!("Email successfully queued for {}", to)
            }))
        })
            as std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>>
    }));

    let mut agent_tools_map = std::collections::HashMap::new();
    agent_tools_map.insert("send_email".to_string(), handler);

    let tools = AgentTools::new(vec![send_email_spec], agent_tools_map);

    println!("\n=== Starting Tool Loop ===");
    println!("Asking the model to send an email to bob@example.com greeting them.");

    let request = ChatRequest {
        model: target_model,
        messages: vec![Message {
            role: "user".to_string(),
            content:
                "Please send a friendly greeting email to bob@example.com introducing yourself."
                    .to_string(),
            name: None,
            images: None,
            tool_calls: None,
        }],
        tools: Some(tools.toolbox.tools().to_vec()),
        format: None,
        options: None,
        stream: Some(false),
        keep_alive: None,
    };

    let mut runtime_config = ToolRuntimeConfig::default();
    runtime_config.max_round_trips = 3;

    let tool_result =
        execute_tool_loop(&client, request, Some(&tools.all()), &runtime_config).await?;

    println!("\n=== Final Response ===");
    println!("{}", tool_result.response.message.content);

    println!("\n=== Validation ===");
    let svc = service.lock().await;
    println!("Emails in outbox: {}", svc.outbox.len());
    for (to, body) in &svc.outbox {
        println!(" - To: {}\n   Body: {}", to, body);
    }

    Ok(())
}
