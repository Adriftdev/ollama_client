/// Multi-turn conversation example: maintain a conversation history.
///
/// This example shows how to build a simple REPL that maintains full
/// conversation context across turns, using the new `Message` helper
/// constructors.
///
/// Run with:
/// ```
/// cargo run --example multi_turn
/// ```
use ollama_client_rs::{
    types::{ChatRequest, Message},
    OllamaClient,
};
use std::env;
use std::io::{self, BufRead, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    let client = OllamaClient::new(ollama_url);

    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());
    println!("Multi-turn chat with {model}");
    println!("Type your message and press Enter. Type 'quit' to exit.\n");

    let mut history: Vec<Message> = vec![Message::system(
        "You are a helpful, friendly assistant. Keep responses concise.",
    )];

    let stdin = io::stdin();
    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut line = String::new();
        stdin.lock().read_line(&mut line)?;
        let input = line.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        if input.is_empty() {
            continue;
        }

        history.push(Message::user(input));

        let request = ChatRequest::new(&model, history.clone());
        match client.chat(&request).await {
            Ok(response) => {
                let reply = response.message.content.trim().to_string();
                println!("Assistant: {reply}\n");
                // Add the assistant's reply to history for the next turn.
                history.push(Message::assistant(&reply));
            }
            Err(e) => {
                eprintln!("Error: {e}");
                // Remove the user message we just added so the history stays consistent.
                history.pop();
            }
        }
    }

    println!("\nConversation had {} turns.", (history.len() - 1) / 2);
    Ok(())
}
