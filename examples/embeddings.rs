/// Embeddings example: generate vector embeddings and compute cosine similarity.
///
/// Run with:
/// ```
/// OLLAMA_MODEL=nomic-embed-text cargo run --example embeddings
/// ```
use ollama_client_rs::{
    types::{EmbedInput, EmbedRequest},
    OllamaClient,
};
use std::env;

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    let client = OllamaClient::new(ollama_url);

    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
    println!("Generating embeddings with {model}...\n");

    let sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep neural networks learn representations from data.",
        "The weather today is sunny with a light breeze.",
    ];

    // Embed all sentences in a single batch request.
    let request = EmbedRequest {
        model: model.clone(),
        input: EmbedInput::Multiple(sentences.iter().map(|s| s.to_string()).collect()),
        truncate: Some(true),
        options: None,
        keep_alive: None,
    };

    let response = client.embed(&request).await?;
    println!(
        "Generated {} embeddings of dimension {}.\n",
        response.embeddings.len(),
        response.embeddings.first().map(|e| e.len()).unwrap_or(0)
    );

    // Compute pairwise cosine similarities.
    println!("Pairwise cosine similarities:");
    println!("{:<55} {:<55} {:>8}", "Sentence A", "Sentence B", "Score");
    println!("{}", "-".repeat(120));

    for i in 0..sentences.len() {
        for j in (i + 1)..sentences.len() {
            let sim = cosine_similarity(&response.embeddings[i], &response.embeddings[j]);
            let a = if sentences[i].len() > 52 {
                format!("{}...", &sentences[i][..52])
            } else {
                sentences[i].to_string()
            };
            let b = if sentences[j].len() > 52 {
                format!("{}...", &sentences[j][..52])
            } else {
                sentences[j].to_string()
            };
            println!("{a:<55} {b:<55} {:>8.4}", sim);
        }
    }

    // Semantic search: find the most similar sentence to a query.
    println!("\n── Semantic search ──────────────────────────────────────");
    let query = "A dog and a fox in a field.";
    println!("Query: {query}");

    let query_req = EmbedRequest {
        model,
        input: EmbedInput::Single(query.to_string()),
        truncate: Some(true),
        options: None,
        keep_alive: None,
    };
    let query_resp = client.embed(&query_req).await?;
    let query_vec = &query_resp.embeddings[0];

    let mut ranked: Vec<(usize, f32)> = sentences
        .iter()
        .enumerate()
        .map(|(i, _)| (i, cosine_similarity(query_vec, &response.embeddings[i])))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nRanked results:");
    for (rank, (idx, score)) in ranked.iter().enumerate() {
        println!("  {}. [{:.4}] {}", rank + 1, score, sentences[*idx]);
    }

    Ok(())
}
