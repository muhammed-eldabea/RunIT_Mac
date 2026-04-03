/// OpenAI-compatible HTTP server — Phase 7.
///
/// Routes:
///   POST /v1/chat/completions   — chat inference (blocking + SSE streaming)
///   GET  /v1/models             — list loaded models
///   GET  /health                — liveness probe

pub mod types;
pub mod handlers;

use std::sync::{Arc, Mutex};

use axum::{Router, routing::{get, post}};

#[cfg(target_os = "macos")]
use crate::forward::Executor;
use bare_metal_tokenizer::Tokenizer;

/// Shared application state passed to every handler via `Arc<AppState>`.
pub struct AppState {
    /// Human-readable model identifier reported in API responses.
    pub model_id:  String,
    /// HuggingFace-compatible tokenizer.
    pub tokenizer: Tokenizer,
    /// The inference executor — wrapped in Mutex because Metal command queues
    /// are not Send+Sync. `spawn_blocking` takes the lock for each request.
    #[cfg(target_os = "macos")]
    pub executor:  Mutex<Executor>,
}

/// Build the axum `Router` with all routes wired up.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health",               get(handlers::health))
        .route("/v1/models",            get(handlers::list_models))
        .route("/v1/chat/completions",  post(handlers::chat_completions))
        .with_state(state)
}
