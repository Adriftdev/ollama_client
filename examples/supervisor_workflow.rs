use ollama_client_rs::{
    agentic::{
        multi_agent::{SupervisorConfig, SupervisorWorkflow},
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

    // Choose models. Different roles could technically use different models
    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string());
    println!("=== Target Model: {} ===", model);

    // 1. Setup a custom tool that the agents can use
    let mut agent_tools_map = std::collections::HashMap::new();
    let current_weather_spec = Tool {
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_current_weather".to_string(),
            description: "Gets the current weather for a city.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
        },
    };

    let weather_handler = FunctionHandler::Async(Box::new(move |args: &mut Value| {
        let args_val = args.clone();
        Box::pin(async move {
            let location = args_val["location"]
                .as_str()
                .unwrap_or("Unknown")
                .to_string();
            println!(">> [TOOL EXECUTED] Getting weather for {}...", location);

            // Mock weather data
            Ok(json!({
                "location": location,
                "temperature": "72F",
                "condition": "Sunny"
            }))
        })
            as std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>>
    }));

    agent_tools_map.insert("get_current_weather".to_string(), weather_handler);
    let agent_tools = AgentTools::new(vec![current_weather_spec], agent_tools_map);

    // 2. Setup the Supervisor Workflow
    let mut config = SupervisorConfig::default();
    config.max_assignments = 3;

    let workflow = SupervisorWorkflow::new(&client, config);

    // 3. Run the workflow
    let task = "What is the current weather in San Francisco? Please provide a nice message explaining what to wear.";
    println!("\n=== Starting Supervisor Workflow ===");
    println!("Task: {}\n", task);

    // We pass None for RAG retriever in this example
    let outcome = workflow
        .run::<DummyRetriever>(&model, task, Some(&agent_tools), None)
        .await?;

    println!("\n=== Workflow Completed ===");
    println!("\nFinal Answer:\n{}", outcome.final_answer);

    // Let's print out what artifacts were accepted
    println!("\n=== Accepted Artifacts ===");
    for (i, artifact) in outcome.accepted_artifacts.iter().enumerate() {
        println!(
            "\nArtifact {}:\nRole: {}\nTask: {}\nContent: {}",
            i + 1,
            artifact.agent_role,
            artifact.assignment_task,
            artifact.content
        );
    }

    Ok(())
}
