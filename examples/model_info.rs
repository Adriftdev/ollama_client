/// Model info example: detect model families and their tool-calling capabilities.
///
/// This example shows how `ModelInfo::from_name()` and `OllamaClient::model_info()`
/// can be used to make model-aware decisions about prompt formatting and tool use.
///
/// Run with:
/// ```
/// cargo run --example model_info
/// ```
use ollama_client_rs::{
    types::{ModelFamily, ModelInfo, ToolFormat},
    OllamaClient,
};
use std::env;

fn tool_format_description(fmt: &ToolFormat) -> &'static str {
    match fmt {
        ToolFormat::Native => "Native Ollama tools field (OpenAI-compatible JSON schema)",
        ToolFormat::PromptInjectedJson => "Tools injected into system prompt as JSON text",
        ToolFormat::HermesXml => "Hermes-style <tool_call> XML tags in content",
        ToolFormat::NativeFunctionTag => "Llama-style <function=name>{args}</function> tags",
    }
}

fn family_notes(family: &ModelFamily) -> &'static str {
    match family {
        ModelFamily::Gemma => {
            "Gemma does not support native tool calling. Tools must be injected \
             as text in the system prompt and parsed from the model's content response."
        }
        ModelFamily::Qwen => {
            "Qwen uses the Hermes tool-call format. Ollama handles this natively \
             but smaller variants may put tool calls in content instead of tool_calls."
        }
        ModelFamily::Llama => {
            "Llama 3.1+ supports function-tag tool calling. 8B models are unreliable; \
             70B+ recommended for production tool use."
        }
        ModelFamily::Mistral => {
            "Mistral uses OpenAI-compatible native tool calling. Works well across sizes."
        }
        ModelFamily::DeepSeek => {
            "DeepSeek-R1 supports native tool calling and extended thinking. \
             Use the :thinking variant for reasoning tasks."
        }
        ModelFamily::Phi => {
            "Phi-3/4 supports native tool calling via Ollama. Small but capable."
        }
        ModelFamily::CommandR => {
            "Command-R supports native tool calling and is optimised for RAG workloads."
        }
        ModelFamily::Unknown => {
            "Unknown model family. Defaulting to native tool format — \
             may need manual adjustment."
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama_url =
        env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434/api".to_string());
    let client = OllamaClient::new(ollama_url);

    // ── Heuristic detection from model name ──────────────────────────────
    println!("=== Heuristic Model Family Detection ===\n");
    let test_models = vec![
        "gemma3:4b",
        "gemma3:27b",
        "qwen2.5:7b",
        "qwen2.5:72b",
        "llama3.1:8b",
        "llama3.2:3b",
        "mistral:7b",
        "deepseek-r1:14b",
        "deepseek-r1:70b:thinking",
        "phi4:14b",
        "phi3:mini",
        "unknown-model:latest",
    ];

    for name in &test_models {
        let info = ModelInfo::from_name(name);
        println!("  {:<35} family={:<12} format={:<22} ctx={:<8} think={}",
            name,
            format!("{:?}", info.family),
            format!("{:?}", info.tool_format),
            info.context_length,
            info.supports_thinking,
        );
    }

    // ── Detailed info for each family ────────────────────────────────────
    println!("\n=== Detailed Family Notes ===\n");
    let families = [
        ModelFamily::Gemma,
        ModelFamily::Qwen,
        ModelFamily::Llama,
        ModelFamily::Mistral,
        ModelFamily::DeepSeek,
        ModelFamily::Phi,
        ModelFamily::CommandR,
        ModelFamily::Unknown,
    ];

    for family in &families {
        let fmt = family.tool_format();
        println!("▸ {:?}", family);
        println!("  Tool format : {}", tool_format_description(&fmt));
        println!("  Notes       : {}", family_notes(family));
        println!();
    }

    // ── Live model info from Ollama ───────────────────────────────────────
    println!("=== Live Model Info from Ollama ===\n");
    match client.list_models().await {
        Ok(models) if !models.is_empty() => {
            for model in &models {
                let info = client.model_info(&model.name).await;
                println!("  {:<40} {:?} / {:?} / ctx={}",
                    model.name,
                    info.family,
                    info.tool_format,
                    info.context_length,
                );
            }
        }
        Ok(_) => println!("  No models found locally. Pull one with: ollama pull qwen2.5:7b"),
        Err(e) => println!("  Could not connect to Ollama: {e}"),
    }

    Ok(())
}
