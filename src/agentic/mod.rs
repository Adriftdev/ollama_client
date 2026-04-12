use serde_json::Value;

use crate::types::{ChatRequest, ChatResponse, Message};

pub mod multi_agent;
pub mod planning;
pub mod rag;
pub mod tool_runtime;

#[cfg(test)]
pub(crate) mod test_support;

pub(crate) fn build_system_message(text: Option<&str>) -> Option<Message> {
    text.map(|value| Message {
        role: "system".to_string(),
        content: value.to_string(),
        images: None,
        tool_calls: None,
        name: None,
    })
}

pub(crate) fn build_user_message(text: &str) -> Message {
    Message {
        role: "user".to_string(),
        content: text.to_string(),
        images: None,
        tool_calls: None,
        name: None,
    }
}

pub(crate) fn extract_text_response(response: &ChatResponse) -> Option<String> {
    let content = &response.message.content;
    if content.trim().is_empty() {
        None
    } else {
        Some(content.clone())
    }
}

pub(crate) fn request_with_json_response(
    system_instruction: Option<&str>,
    user_prompt: String,
    schema: Value,
) -> ChatRequest {
    let mut messages = vec![];
    if let Some(sys_msg) = build_system_message(system_instruction) {
        messages.push(sys_msg);
    }
    messages.push(build_user_message(&user_prompt));

    ChatRequest {
        model: String::new(), // To be filled by caller
        messages,
        format: Some(schema),
        options: None,
        stream: Some(false),
        tools: None,
        keep_alive: None,
    }
}
