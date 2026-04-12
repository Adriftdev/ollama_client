use ollama_client_rs::{
    agentic::{
        planning::{PlanningConfig, PlanningSession},
        rag::{RagError, RagQuery, RetrievedChunk, Retriever},
        tool_runtime::AgentTools,
    },
    types::{FunctionDefinition, Tool},
    FunctionHandler, OllamaClient,
};
use serde_json::{json, Value};
use std::env;

struct DummyRetriever;
#[async_trait::async_trait]
impl Retriever for DummyRetriever {
    async fn retrieve(&self, _query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        Ok(vec![])
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    println!("Connecting to Ollama at {}...", ollama_url);

    let client = OllamaClient::new(ollama_url);
    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string());
    println!("=== Target Model: {} ===", model);

    // 1. Setup tools
    let mut agent_tools_map = std::collections::HashMap::new();

    // Tool: Calculator
    let calculator_spec = Tool {
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: "calculate".to_string(),
            description: "Evaluates a simple math expression (e.g., '10 + 5').".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": { "type": "string" }
                },
                "required": ["expression"]
            }),
        },
    };

    let calc_handler = FunctionHandler::Async(Box::new(move |args: &mut Value| {
        let args_val = args.clone();
        Box::pin(async move {
            let expr = args_val["expression"].as_str().unwrap_or("0").to_string();
            println!(">> [TOOL EXECUTED] Calculating: {}...", expr);

            // Very naive mock calculator for demonstration
            let result = if expr.contains('+') {
                let parts: Vec<&str> = expr.split('+').collect();
                let a: i32 = parts[0].trim().parse().unwrap_or(0);
                let b: i32 = parts[1].trim().parse().unwrap_or(0);
                a + b
            } else {
                0
            };

            Ok(json!({ "result": result }))
        })
            as std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>>
    }));

    // Tool: Length Converter
    let converter_spec = Tool {
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: "meters_to_feet".to_string(),
            description: "Converts meters to feet.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "meters": { "type": "number" }
                },
                "required": ["meters"]
            }),
        },
    };

    let convert_handler = FunctionHandler::Async(Box::new(move |args: &mut Value| {
        let args_val = args.clone();
        Box::pin(async move {
            let meters = args_val["meters"].as_f64().unwrap_or(0.0);
            println!(">> [TOOL EXECUTED] Converting {} meters to feet...", meters);

            let feet = meters * 3.28084;
            Ok(json!({ "feet": feet }))
        })
            as std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>>
    }));

    agent_tools_map.insert("calculate".to_string(), calc_handler);
    agent_tools_map.insert("meters_to_feet".to_string(), convert_handler);

    let agent_tools = AgentTools::new(vec![calculator_spec, converter_spec], agent_tools_map);

    // 2. Setup the Planning Workflow
    let config = PlanningConfig::default();
    let session = PlanningSession::new(&client, config);

    // 3. Run the workflow
    let task = "I have a rod that is 5 meters long. Another is 7 meters long. What is their combined length in feet?";
    println!("\n=== Starting Plan-and-Execute Workflow ===");
    println!("Task: {}\n", task);

    let outcome = session
        .run::<DummyRetriever>(&model, task, Some(&agent_tools), None)
        .await?;

    println!("\n=== Workflow Completed ===");
    println!("\nFinal Answer:\n{}", outcome.final_answer);

    // Let's print out the plan execution memory
    println!("\n=== Working Memory ===");
    for (step_id, result) in &outcome.working_memory.entries {
        println!(
            "\nStep Idea: {}\nResult: {}",
            step_id,
            serde_json::to_string_pretty(result).unwrap_or_default()
        );
    }

    Ok(())
}
