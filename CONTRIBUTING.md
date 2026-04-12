# Contributing to ollama-client-rs

Thank you for your interest in contributing to `ollama-client-rs`. This project aims to provide a high-performance, idiomatic Rust client for the Ollama API, with a specialized `agentic` layer for deterministic multi-agent orchestration.

This library is part of a broader ecosystem alongside `gemini-client-rs`. Both libraries share architectural philosophy, module structure, and test patterns. **If you contribute to one, please keep the other in mind.**

As a systems-oriented project, we prioritize **reliability, determinism, and architectural clarity**.

---

## 🏗️ Getting Started

### Prerequisites

- **Rust**: Latest stable version (Edition 2021).
- **Ollama**: A running [Ollama](https://ollama.ai) instance with at least one model pulled (e.g. `ollama pull llama3.1:8b`).

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Adriftdev/ollama-client.git
   cd ollama-client
   ```

2. **Configure Environment** (optional):
   By default the client connects to `http://127.0.0.1:11434/api`. Override with:
   ```env
   OLLAMA_HOST=http://your-host:11434/api
   OLLAMA_MODEL=llama3.1:8b
   ```

3. **Run Examples**:
   Verify your setup by running a basic example:
   ```bash
   cargo run --example basic
   ```

---

## 🏛️ Ecosystem Parity

This crate is designed to converge with `gemini-client-rs` into a unified agentic ecosystem. When making changes, follow these rules:

1. **Mirror module structure**: Both libraries share `agentic/tool_runtime.rs`, `agentic/planning.rs`, `agentic/rag.rs`, `agentic/multi_agent.rs`, and `agentic/test_support.rs`. New agentic modules should be added to both.
2. **Mirror test coverage**: Every unit test in `gemini-client-rs` should have a counterpart here, adapted for Ollama types.
3. **Consistent visibility**: Internal helpers are `pub(crate)`. Only core types (`OllamaClient`, error enums, agentic structs) are `pub`.
4. **Consistent telemetry**: Use the `telemetry_*!` macros from `telemetry.rs`, never raw `tracing::*` calls.

---

## 🛠️ Architectural Philosophy

When contributing to the `agentic` module or core client, adhere to these principles:

1. **Occam's Razor**: Prefer the simplest implementation that satisfies the requirements. Avoid over-engineering orchestrators unless scale or resilience demands it.
2. **Deterministic Orchestration**: Higher-level patterns (Supervisor, Worker, etc.) should have predictable state transitions. Avoid hidden side effects in agent loops.
3. **Store and Forward (Persistence)**: For agentic workflows requiring long-running state, utilize local disk-backed buffers or persistent blackboards.
4. **Transparent Proxy**: Ensure the low-level client remains a clean relay for the Ollama API, preserving byte-for-byte fidelity where possible.

---

## 💻 Coding Standards

### 1. Idiomatic Rust
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Use `thiserror` for library-level error definitions in `src/`.
- Do **not** add `anyhow` as a library dependency — it is acceptable in examples and dev-dependencies only.

### 2. Linting & Formatting
We maintain strict quality gates. PRs will not be merged if they contain Clippy warnings.
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

### 3. Concurrency
- Use `tokio` for async operations.
- Prefer ownership and message passing over shared mutable state.
- When shared state is necessary, use `Arc<RwLock<T>>` or `Arc<Mutex<T>>`.

---

## 🧪 Testing Protocol

Every PR must be verified across multiple feature configurations:

1. **Core Tests**:
   ```bash
   cargo test
   ```

2. **Feature-Specific Tests**:
   Verify the `tracing` feature and other optional dependencies:
   ```bash
   cargo test --features tracing
   ```

3. **Example Verification**:
   If you modify the `agentic` module, you MUST verify the corresponding examples:
   ```bash
   cargo check --examples
   cargo run --example supervisor_workflow
   ```

---

## 📬 Contribution Workflow

1. **Open an Issue**: Discuss major changes before implementation.
2. **Fork and Branch**: Work on a feature branch (`feat/your-feature` or `fix/your-fix`).
3. **Conventional Commits**: We recommend using conventional commit messages (e.g., `feat: add RAG caching`).
4. **PR Review**: All PRs require at least one approval. Ensure all CI checks pass.

---

## 🛡️ Security

- **Never commit your `.env` file or API keys**.
- If you find a security vulnerability, please report it via GitHub Issues (marking as private if possible) or contact the maintainers directly.
