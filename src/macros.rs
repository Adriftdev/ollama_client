/// Macro to easily construct a `ChatRequest` from a model and a sequence of messages.
///
/// Under the hood, this macro leverages the `ChatRequestBuilder` and accepts optional arguments
/// to further customize the request such as adding tools, format, options, streaming, and keep_alive flags.
///
/// # Examples
/// ```rust
/// use ollama_client_rs::{chat_request, messages};
///
/// let req = chat_request!(
///     model: "gemma4:latest",
///     messages: messages!(
///         system: "You are a helpful assistant.",
///         user: "What's the weather like?",
///     )
/// );
/// ```
#[macro_export]
macro_rules! chat_request {
    (
        model: $model:expr,
        messages: $messages:expr
        $(, tools: $tools:expr)?
        $(, format: $format:expr)?
        $(, options: $options:expr)?
        $(, stream: $stream:expr)?
        $(, keep_alive: $keep_alive:expr)?
        $(,)?
    ) => {{
        let mut builder = $crate::types::ChatRequestBuilder::new($model)
            .messages($messages);
        
        $(builder = builder.tools($tools);)?
        $(builder = builder.format($format);)?
        $(builder = builder.options($options);)?
        $(builder = builder.stream($stream);)?
        $(builder = builder.keep_alive($keep_alive);)?
        
        builder.build()
    }};
}

/// Macro to easily construct a list of `Message` structs.
///
/// # Examples
/// ```rust
/// use ollama_client_rs::messages;
///
/// let msgs = messages!(
///     system: "You are a helpful assistant.",
///     user: "What's the meaning of life?",
/// );
/// ```
#[macro_export]
macro_rules! messages {
    ($( $role:ident : $content:expr ),* $(,)?) => {{
        let mut messages = Vec::new();
        $(
            let msg = match stringify!($role) {
                "system" => $crate::types::Message::system($content),
                "user" => $crate::types::Message::user($content),
                "assistant" => $crate::types::Message::assistant($content),
                _ => {
                    $crate::types::Message {
                        role: stringify!($role).to_string(),
                        content: $content.into(),
                        name: None,
                        images: None,
                        audio: None,
                        video_frames: None,
                        thinking: None,
                        tool_calls: None,
                    }
                }
            };
            messages.push(msg);
        )*
        messages
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatRequest, Message};

    #[test]
    fn test_messages_macro() {
        let msgs = messages!(
            system: "You are an AI.",
            user: "Hello!",
        );
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "You are an AI.");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "Hello!");
    }

    #[test]
    fn test_chat_request_macro() {
        let msgs = messages!(user: "Hi.");
        let req = chat_request!(
            model: "test_model",
            messages: msgs,
            stream: true,
            keep_alive: "5m"
        );
        assert_eq!(req.model, "test_model");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.stream, Some(true));
        assert_eq!(req.keep_alive, Some("5m".to_string()));
    }
}
