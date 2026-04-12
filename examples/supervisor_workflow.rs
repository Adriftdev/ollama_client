use async_trait::async_trait;
use ollama_client_rs::{
    agentic::{
        multi_agent::{SupervisorConfig, SupervisorWorkflow},
        rag::{RagError, RagQuery, RetrievedChunk, Retriever},
    },
    OllamaClient,
};
use std::env;

struct LocalRetriever {
    chunks: Vec<RetrievedChunk>,
}

#[async_trait]
impl Retriever for LocalRetriever {
    async fn retrieve(&self, query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        Ok(self.chunks.iter().take(query.top_k).cloned().collect())
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

    let retriever = LocalRetriever {
        chunks: vec![RetrievedChunk {
            id: "doc-1".to_string(),
            source: "local".to_string(),
            title: "Incident playbook".to_string(),
            content: "Major incidents should include current status, customer impact, and next update time."
                .to_string(),
            score: 0.99,
            metadata: None,
        }],
    };

    // 2. Setup the Supervisor Workflow
    let config = SupervisorConfig {
        max_assignments: 3,
        ..Default::default()
    };

    let workflow = SupervisorWorkflow::new(&client, config);

    // 3. Run the workflow
    let task = "Prepare a short stakeholder update for incident INC-204.";

    let outcome = workflow.run(&model, task, None, Some(&retriever)).await?;

    println!("\nFinal Answer:\n{}", outcome.final_answer);

    Ok(())
}
