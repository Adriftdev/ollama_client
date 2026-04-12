use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use serde_json::json;

use crate::{
    agentic::{
        rag::{RagError, RagQuery, RetrievedChunk, Retriever},
        tool_runtime::ModelBackend,
    },
    types::{ChatRequest, ChatResponse},
    OllamaError,
};

pub(crate) type ScriptedResponse = Box<
    dyn Fn(&ChatRequest) -> Result<ChatResponse, OllamaError> + Send + Sync,
>;

pub(crate) struct ScriptedBackend {
    responses: Arc<Mutex<VecDeque<ScriptedResponse>>>,
}

impl ScriptedBackend {
    pub(crate) fn new(responses: Vec<ScriptedResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
        }
    }
}

#[async_trait]
impl ModelBackend for ScriptedBackend {
    async fn chat(
        &self,
        request: &ChatRequest,
    ) -> Result<ChatResponse, OllamaError> {
        let response = self
            .responses
            .lock()
            .expect("response queue lock")
            .pop_front()
            .ok_or_else(|| OllamaError::Api(json!({"error": "no scripted response remaining"})))?;

        response(request)
    }
}

pub(crate) struct StaticRetriever {
    chunks: Vec<RetrievedChunk>,
}

impl StaticRetriever {
    pub(crate) fn new(chunks: Vec<RetrievedChunk>) -> Self {
        Self { chunks }
    }
}

#[async_trait]
impl Retriever for StaticRetriever {
    async fn retrieve(&self, query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        Ok(self.chunks.iter().take(query.top_k).cloned().collect())
    }
}

pub(crate) fn response_with_text(text: &str) -> ChatResponse {
    ChatResponse {
        model: "test-model".to_string(),
        created_at: "2024-01-01T00:00:00Z".to_string(),
        message: crate::types::Message {
            role: "assistant".to_string(),
            content: text.to_string(),
            name: None,
            images: None,
            tool_calls: None,
        },
        done: true,
        total_duration: None,
        load_duration: None,
        prompt_eval_count: None,
        prompt_eval_duration: None,
        eval_count: None,
        eval_duration: None,
    }
}

pub(crate) fn response_with_tool_calls(
    calls: Vec<crate::types::ToolCall>,
) -> ChatResponse {
    ChatResponse {
        model: "test-model".to_string(),
        created_at: "2024-01-01T00:00:00Z".to_string(),
        message: crate::types::Message {
            role: "assistant".to_string(),
            content: String::new(),
            name: None,
            images: None,
            tool_calls: Some(calls),
        },
        done: true,
        total_duration: None,
        load_duration: None,
        prompt_eval_count: None,
        prompt_eval_duration: None,
        eval_count: None,
        eval_duration: None,
    }
}
