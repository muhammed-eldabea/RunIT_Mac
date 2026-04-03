/// HTTP route handlers — Phase 7.
///
/// POST /v1/chat/completions  — blocking (non-streaming) and SSE streaming
/// GET  /v1/models            — list available models
/// GET  /health               — liveness probe

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
};
use axum::response::sse::{Event, KeepAlive};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::types::*;
use super::AppState;
use crate::sampler::{SamplerConfig, SimpleRng, sample};
use crate::chat_template::{format_prompt, ChatMessage as TplMessage, SpecialTokens};

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn new_id() -> String {
    format!("chatcmpl-{:x}", unix_now())
}

// ── GET /health ───────────────────────────────────────────────────────────────

pub async fn health() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

// ── GET /v1/models ────────────────────────────────────────────────────────────

pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let list = ModelList {
        object: "list".into(),
        data: vec![ModelObject {
            id:       state.model_id.clone(),
            object:   "model".into(),
            created:  unix_now(),
            owned_by: "bare-metal-mx-llm".into(),
        }],
    };
    Json(list)
}

// ── POST /v1/chat/completions ─────────────────────────────────────────────────

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Detect system message (first message with role="system")
    let system = req.messages.iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone());

    // Non-system messages for template
    let non_system: Vec<TplMessage> = req.messages.iter()
        .filter(|m| m.role != "system")
        .map(|m| TplMessage { role: m.role.clone(), content: m.content.clone() })
        .collect();

    let prompt_str = format_prompt(&non_system, system.as_deref());

    let max_new_tokens = req.max_tokens.unwrap_or(256);
    let sampler_cfg = SamplerConfig {
        temperature: req.temperature,
        top_k:       req.top_k.unwrap_or(50),
        top_p:       req.top_p,
        seed:        req.seed.unwrap_or(42),
    };

    if req.stream {
        stream_response(state, prompt_str, max_new_tokens, sampler_cfg).await
    } else {
        blocking_response(state, prompt_str, max_new_tokens, sampler_cfg).await
    }
}

// ── Non-streaming path ────────────────────────────────────────────────────────

async fn blocking_response(
    state:       Arc<AppState>,
    prompt_str:  String,
    max_new:     usize,
    sampler_cfg: SamplerConfig,
) -> Response {
    let model_id = state.model_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        generate_blocking(&state, &prompt_str, max_new, sampler_cfg)
    }).await;

    match result {
        Ok(Ok((text, prompt_len, completion_len))) => {
            let resp = ChatCompletionResponse {
                id:      new_id(),
                object:  "chat.completion".into(),
                created: unix_now(),
                model:   model_id,
                choices: vec![Choice {
                    index: 0,
                    message: ResponseMessage { role: "assistant".into(), content: text },
                    finish_reason: "stop".into(),
                }],
                usage: Usage {
                    prompt_tokens:     prompt_len,
                    completion_tokens: completion_len,
                    total_tokens:      prompt_len + completion_len,
                },
            };
            Json(resp).into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e),
        Err(e)     => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

// ── SSE streaming path ────────────────────────────────────────────────────────

async fn stream_response(
    state:       Arc<AppState>,
    prompt_str:  String,
    max_new:     usize,
    sampler_cfg: SamplerConfig,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

    tokio::task::spawn_blocking(move || {
        let id       = new_id();
        let model_id = state.model_id.clone();

        // Send role delta first
        let role_chunk = ChatCompletionChunk {
            id: id.clone(), object: "chat.completion.chunk".into(),
            created: unix_now(), model: model_id.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: DeltaMessage { role: Some("assistant".into()), content: None },
                finish_reason: None,
            }],
        };
        let _ = tx.blocking_send(Ok(Event::default()
            .data(serde_json::to_string(&role_chunk).unwrap_or_default())));

        if let Err(e) = generate_stream(&state, &prompt_str, max_new, sampler_cfg,
                                        &id, &model_id, &tx) {
            let _ = tx.blocking_send(Ok(Event::default().data(format!("[ERROR] {e}"))));
        }

        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

// ── Core generation helpers (called from spawn_blocking) ──────────────────────

#[cfg(target_os = "macos")]
fn generate_blocking(
    state:       &Arc<AppState>,
    prompt_str:  &str,
    max_new:     usize,
    sampler_cfg: SamplerConfig,
) -> Result<(String, usize, usize), String> {
    use crate::kv_cache::KvCache;

    let tok_ids = state.tokenizer.encode(prompt_str, false)
        .map_err(|e| e.to_string())?;
    let prompt_len = tok_ids.len();
    let max_seq    = prompt_len + max_new + 8;

    let executor = state.executor.lock().map_err(|e| e.to_string())?;
    let mut kv = KvCache::new(&executor.ctx, &executor.config, max_seq);

    let logits = executor.prefill(&tok_ids, &mut kv).map_err(|e| e.to_string())?;

    let special = SpecialTokens::default();
    let mut rng  = SimpleRng::new(sampler_cfg.seed);
    let mut out  = Vec::<u32>::with_capacity(max_new);

    let mut next = sample(&logits, &sampler_cfg, &mut rng);
    out.push(next);

    for pos in (prompt_len as u32)..(prompt_len + max_new) as u32 {
        if special.is_stop(next) { break; }
        let logits = executor.forward(next, pos, &mut kv).map_err(|e| e.to_string())?;
        next = sample(&logits, &sampler_cfg, &mut rng);
        out.push(next);
    }

    let text = state.tokenizer.decode(&out, true).unwrap_or_default();
    Ok((text, prompt_len, out.len()))
}

#[cfg(not(target_os = "macos"))]
fn generate_blocking(
    _state: &Arc<AppState>, _prompt: &str, _max_new: usize, _cfg: SamplerConfig,
) -> Result<(String, usize, usize), String> {
    Err("Metal GPU required (macOS only)".into())
}

#[cfg(target_os = "macos")]
fn generate_stream(
    state:       &Arc<AppState>,
    prompt_str:  &str,
    max_new:     usize,
    sampler_cfg: SamplerConfig,
    id:          &str,
    model_id:    &str,
    tx:          &mpsc::Sender<Result<Event, std::convert::Infallible>>,
) -> Result<(), String> {
    use crate::kv_cache::KvCache;

    let tok_ids = state.tokenizer.encode(prompt_str, false)
        .map_err(|e| e.to_string())?;
    let prompt_len = tok_ids.len();
    let max_seq    = prompt_len + max_new + 8;

    let executor = state.executor.lock().map_err(|e| e.to_string())?;
    let mut kv = KvCache::new(&executor.ctx, &executor.config, max_seq);

    let logits = executor.prefill(&tok_ids, &mut kv).map_err(|e| e.to_string())?;

    let special = SpecialTokens::default();
    let mut rng  = SimpleRng::new(sampler_cfg.seed);
    let mut next = sample(&logits, &sampler_cfg, &mut rng);

    for pos in (prompt_len as u32)..(prompt_len + max_new) as u32 {
        if special.is_stop(next) { break; }

        if let Ok(text) = state.tokenizer.decode(&[next], true) {
            if !text.is_empty() {
                let chunk = ChatCompletionChunk {
                    id: id.to_string(), object: "chat.completion.chunk".into(),
                    created: unix_now(), model: model_id.to_string(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: DeltaMessage { role: None, content: Some(text) },
                        finish_reason: None,
                    }],
                };
                let _ = tx.blocking_send(Ok(Event::default()
                    .data(serde_json::to_string(&chunk).unwrap_or_default())));
            }
        }

        let logits = executor.forward(next, pos, &mut kv).map_err(|e| e.to_string())?;
        next = sample(&logits, &sampler_cfg, &mut rng);
    }

    // Final stop chunk
    let final_chunk = ChatCompletionChunk {
        id: id.to_string(), object: "chat.completion.chunk".into(),
        created: unix_now(), model: model_id.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: DeltaMessage { role: None, content: None },
            finish_reason: Some("stop".into()),
        }],
    };
    let _ = tx.blocking_send(Ok(Event::default()
        .data(serde_json::to_string(&final_chunk).unwrap_or_default())));

    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn generate_stream(
    _state: &Arc<AppState>, _prompt: &str, _max_new: usize, _cfg: SamplerConfig,
    _id: &str, _model_id: &str,
    _tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
) -> Result<(), String> {
    Err("Metal GPU required (macOS only)".into())
}

// ── Error helper ──────────────────────────────────────────────────────────────

fn error_response(status: StatusCode, msg: &str) -> Response {
    let body = ApiError {
        error: ApiErrorDetail {
            message: msg.to_string(),
            r#type:  "internal_error".into(),
            code:    None,
        },
    };
    (status, Json(body)).into_response()
}
