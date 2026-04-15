/// Streaming example: receive tokens as they are generated.
///
/// Run with:
/// ```
/// cargo run --example streaming
/// ```
use futures_util::StreamExt;
use ollama_client_rs::{
    types::{ChatRequest, Message},
    OllamaClient,
};
use std::env;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    let client = OllamaClient::new(ollama_url);

    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());
    println!("Streaming from {model}...\n");

    let request = ChatRequest::builder(&model)
        .message(Message::system(
            "You are a creative writer. Write vivid, engaging prose.",
        ))
        .message(Message::user(
            "Write a short paragraph describing a thunderstorm at sea.",
        ))
        .stream(true)
        .build();

    let mut stream = client.chat_stream(&request).await?;

    let mut total_prompt_tokens = 0u64;
    let mut total_output_tokens = 0u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        // Print each token as it arrives.
        if !chunk.message.content.is_empty() {
            print!("{}", chunk.message.content);
            std::io::stdout().flush()?;
        }

        // The final chunk carries usage statistics.
        if chunk.done {
            println!(); // newline after the last token
            if let (Some(p), Some(o)) = (chunk.prompt_eval_count, chunk.eval_count) {
                total_prompt_tokens = p;
                total_output_tokens = o;
            }
            if let Some(reason) = &chunk.done_reason {
                println!("\nFinish reason : {reason}");
            }
        }
    }

    println!(
        "Tokens used   : {total_prompt_tokens} prompt + {total_output_tokens} output = {}",
        total_prompt_tokens + total_output_tokens
    );

    Ok(())
}
