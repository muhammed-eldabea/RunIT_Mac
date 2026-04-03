/// OpenAI-compatible request / response types — Phase 7.

use serde::{Deserialize, Serialize};

// ── Request types ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role:    String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model:             Option<String>,
    pub messages:          Vec<ChatMessage>,
    #[serde(default)]
    pub stream:            bool,
    #[serde(default = "default_temperature")]
    pub temperature:       f32,
    #[serde(default = "default_top_p")]
    pub top_p:             f32,
    pub top_k:             Option<usize>,
    pub max_tokens:        Option<usize>,
    pub seed:              Option<u64>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p()        -> f32 { 0.9 }

// ── Response types ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens:     usize,
    pub completion_tokens: usize,
    pub total_tokens:      usize,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role:    String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index:         usize,
    pub message:       ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id:      String,
    pub object:  String,
    pub created: u64,
    pub model:   String,
    pub choices: Vec<Choice>,
    pub usage:   Usage,
}

// ── Streaming (SSE) delta types ───────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role:    Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index:         usize,
    pub delta:         DeltaMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id:      String,
    pub object:  String,
    pub created: u64,
    pub model:   String,
    pub choices: Vec<StreamChoice>,
}

// ── Models list ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id:      String,
    pub object:  String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data:   Vec<ModelObject>,
}

// ── Error response ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetail {
    pub message: String,
    pub r#type:  String,
    pub code:    Option<String>,
}
