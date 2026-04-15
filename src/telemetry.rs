#[cfg(feature = "tracing")]
use crate::OllamaError;

#[cfg(feature = "tracing")]
#[allow(dead_code)]
pub(crate) struct SpanGuard(Option<tracing::Span>);

#[cfg(not(feature = "tracing"))]
pub(crate) struct SpanGuard;

impl SpanGuard {
    #[cfg(feature = "tracing")]
    pub(crate) fn from_info_span(span: tracing::Span) -> Self {
        Self(Some(span))
    }

    #[cfg(not(feature = "tracing"))]
    pub(crate) fn from_info_span() -> Self {
        Self
    }
}

#[cfg(feature = "tracing")]
pub(crate) fn ollama_error_kind(error: &OllamaError) -> &'static str {
    match error {
        OllamaError::Http(_) => "http",
        OllamaError::Api(_) => "api",
        OllamaError::Json { .. } => "json",
    }
}

macro_rules! telemetry_span_guard {
    (info, $name:expr $(, $($field:tt)+)?) => {{
        #[cfg(feature = "tracing")]
        {
            $crate::telemetry::SpanGuard::from_info_span(
                tracing::info_span!($name $(, $($field)+)?)
            )
        }
        #[cfg(not(feature = "tracing"))]
        {
            $crate::telemetry::SpanGuard::from_info_span()
        }
    }};
}

macro_rules! telemetry_info {
    ($($tt:tt)*) => {{
        #[cfg(feature = "tracing")]
        {
            tracing::info!($($tt)*);
        }
    }};
}

macro_rules! telemetry_debug {
    ($($tt:tt)*) => {{
        #[cfg(feature = "tracing")]
        {
            tracing::debug!($($tt)*);
        }
    }};
}

macro_rules! telemetry_error {
    ($($tt:tt)*) => {{
        #[cfg(feature = "tracing")]
        {
            tracing::error!($($tt)*);
        }
    }};
}

pub(crate) use telemetry_debug;
pub(crate) use telemetry_error;
pub(crate) use telemetry_info;
pub(crate) use telemetry_span_guard;
