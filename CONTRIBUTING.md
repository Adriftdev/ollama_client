# Contributing to ollama-client-rs

Thank you for your interest in contributing to `ollama-client-rs`. This project provides a high-performance, idiomatic, and infrastructure-first Rust client for the Ollama API.

We focus on being a **precision thin layer**—prioritizing raw API fidelity, reliability, and architectural clarity.

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
   Verify your setup by running the basic example:
   ```bash
   cargo run --example basic
   ```

---

## 🏛️ Ecosystem Alignment

This crate is part of the `rain` ecosystem. To maintain consistency across clients (e.g., `gemini-client-rs`), we adhere to a shared set of interface patterns:

1. **Builder Pattern**: Clients should always implement `new`, `with_client`, and `with_api_url`.
2. **Pinned Streams**: All streaming methods must return `Pin<Box<dyn Stream>>` to simplify caller integration.
3. **Standardized Telemetry**: Use the internal `telemetry_*!` macros. Never use raw `tracing` calls directly in the client logic.
4. **Error Mapping**: Maintain a flat, descriptive `OllamaError` enum using `thiserror`.

---

## 🛠️ Architectural Philosophy

When contributing to the core client, adhere to these principles:

1. **Thin Layer Philosophy**: The client is a transport and mapping layer. Avoid adding complex state machines, orchestration logic, or "agentic" capabilities. These belong in higher-level crates (like `rain`).
2. **Transparent Proxy**: Preserving byte-for-byte fidelity and API structure is a priority. Avoid abstractions that hide underlying API features.
3. **Rust Type Safety**: Leverage Rust's type system to make API constraints explicit and compile-time safe.
4. **Zero-Overhead Abstractions**: Ensure the mapping from request structs to JSON is efficient and follows the public API documentation precisely.

---

## 💻 Coding Standards

### 1. Idiomatic Rust
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Use `thiserror` for all library-level error definitions.
- Avoid `anyhow` in the core library; it is reserved for examples and tests.

### 2. Linting & Formatting
We maintain strict quality gates. PRs will not be merged if they contain Clippy warnings.
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

### 3. Telemetry
Always use the standardized macros to ensure consistent mapping of error kinds and span data for cloud-native observability.

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
   Ensure basic examples remain functional:
   ```bash
   cargo check --examples
   ```

---

## 📬 Contribution Workflow

1. **Open an Issue**: Discuss major changes before implementation.
2. **Fork and Branch**: Work on a feature branch (`feat/your-feature` or `fix/your-fix`).
3. **Conventional Commits**: We use conventional commit messages (e.g., `feat: add tool calling support`).
4. **PR Review**: All PRs require approval and passing CI checks.

---

## 🛡️ Security

- Report security vulnerabilities via GitHub Issues or contact the maintainers directly.
