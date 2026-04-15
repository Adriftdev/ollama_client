/// Pull model example: download a model with progress reporting.
///
/// Run with:
/// ```
/// PULL_MODEL=qwen2.5:0.5b cargo run --example pull_model
/// ```
use futures_util::StreamExt;
use ollama_client_rs::OllamaClient;
use std::env;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    let client = OllamaClient::new(ollama_url);

    let model = env::var("PULL_MODEL").unwrap_or_else(|_| "qwen2.5:0.5b".to_string());
    println!("Pulling model: {model}\n");

    let mut stream = client.pull_model(&model, false).await?;

    let mut last_status = String::new();
    while let Some(event) = stream.next().await {
        let event = event?;

        let status = event["status"].as_str().unwrap_or("").to_string();
        let completed = event["completed"].as_u64().unwrap_or(0);
        let total = event["total"].as_u64().unwrap_or(0);

        if total > 0 {
            let pct = (completed as f64 / total as f64 * 100.0) as u64;
            let bar_len = 40usize;
            let filled = (pct as usize * bar_len / 100).min(bar_len);
            let bar = format!(
                "[{}{}] {:>3}%  {:.1} / {:.1} MB",
                "#".repeat(filled),
                " ".repeat(bar_len - filled),
                pct,
                completed as f64 / 1_048_576.0,
                total as f64 / 1_048_576.0,
            );
            print!("\r{status:<20} {bar}");
            io::stdout().flush()?;
        } else if status != last_status {
            println!("{status}");
            last_status = status;
        }
    }

    println!("\n\nDone! Model '{model}' is ready to use.");
    Ok(())
}
