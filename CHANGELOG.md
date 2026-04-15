# Changelog

All notable changes to `ollama_client_rs` are documented here.

## [0.3.0] — 2025-04-15

### Added

**Model-aware tool calling** — the crate now understands the different tool-call
formats used by each model family and selects the correct one automatically.

| Type | Details |
|---|---|
| `ModelFamily` | Enum covering Gemma, Qwen, Llama, Mistral, DeepSeek, Phi, CommandR, Unknown |
| `ToolFormat` | Enum: `Native`, `PromptInjectedJson`, `HermesXml`, `NativeFunctionTag` |
| `ModelInfo` | Struct with `family`, `tool_format`, `context_length`, `supports_thinking` |
| `ModelInfo::from_name()` | Heuristic detection from model name string |
| `OllamaClient::model_info()` | Enriches heuristics with live Ollama API data |

**`ChatRequestBuilder`** — fluent builder API for constructing chat requests:
- Automatically injects tools into the system prompt for Gemma (prompt-injected format)
- Passes tools as native Ollama `tools` field for all other families
- Helper constructors: `ChatRequest::new()`, `ChatRequest::builder()`

**`Message` helper constructors** — `Message::system()`, `Message::user()`,
`Message::assistant()`, `Message::tool_result()` for ergonomic message building.

**`Tool::function()`** — convenience constructor for function-type tools.

**`ChatResponse::extract_tool_calls()`** — smart extraction that:
1. Returns native `tool_calls` field if populated
2. Falls back to parsing `content` for Gemma JSON format
3. Falls back to parsing Hermes `<tool_call>` XML tags
4. Falls back to parsing Llama `<function=name>` tags

**`OllamaClient::pull_model()`** — stream model downloads with progress events.

**Streaming improvements**:
- Buffer now handles chunk boundaries that fall mid-JSON-line (previously could panic)
- Synthetic `done=true` chunk emitted if server closes connection without one
- Tool calls are normalised on the final streaming chunk automatically

**Automatic tool-call normalisation** — both `chat()` and `chat_stream()` now
call `extract_tool_calls()` on the response and populate `message.tool_calls`
if it was empty but content contained parseable tool calls. Callers no longer
need to check both fields.

### New examples

| Example | Description |
|---|---|
| `basic` | List models, simple chat, structured JSON output, model info |
| `streaming` | Token-by-token streaming with usage statistics |
| `tool_calling` | Full two-turn tool-use loop for native models (Qwen, Mistral) |
| `gemma_tools` | Prompt-injected tool calling for Gemma with content parsing |
| `multi_turn` | Interactive REPL with full conversation history |
| `embeddings` | Batch embeddings, cosine similarity, semantic search |
| `model_info` | Model family detection and capability overview |
| `pull_model` | Download a model with a progress bar |
| `agentic_loop` | Full ReAct-style agent loop with tool dispatch |

### Changed

- `ChatRequest` now derives `Clone` to support multi-turn conversation patterns
- `EmbedInput` supports both `Single(String)` and `Multiple(Vec<String>)` variants
- `Cargo.toml`: added `keywords` and `categories` for crates.io discoverability

### Fixed

- Streaming parser no longer fails when a TCP chunk boundary falls mid-JSON-line
- `chat()` and `chat_stream()` now correctly report tool calls for models that
  embed them in `content` rather than the native `tool_calls` field

---

## [0.2.0]

Initial public release with basic chat, streaming, and embeddings support.
