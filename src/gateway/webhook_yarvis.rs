//! Yarvis router integration — `/api/chat` endpoint.
//!
//! Provides a full agent-loop endpoint for the Yarvis router, which forwards
//! Teams webhooks and email hooks to ZeroClaw. Supports two delivery modes:
//!
//! 1. **HTTP response** (default): agent reply returned in the JSON body.
//! 2. **Channel direct** (when `teams_context` is present): reply sent directly
//!    to Teams via Bot Framework, HTTP body confirms delivery.
//!
//! ## Wire format
//!
//! ```text
//! POST /api/chat
//! Authorization: Bearer zc_<token>
//! { "message": "...", "session_id": "...", "teams_context": { ... } }
//! ```
//!
//! Response (non-channel): `{ "reply": "...", "model": "..." }`
//! Response (channel):     `{ "delivered_via_channel": true, "model": "..." }`

use super::{client_key_from_request, AppState, RATE_LIMIT_WINDOW_SECS};
use crate::channels::traits::Channel;
use crate::memory::MemoryCategory;
use crate::providers;
use axum::{
    extract::{ConnectInfo, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Request body for `POST /api/chat`.
#[derive(Debug, Deserialize)]
pub struct ApiChatBody {
    /// The user message to process.
    pub message: String,

    /// Optional session ID for memory scoping.
    #[serde(default)]
    pub session_id: Option<String>,

    /// Optional reply channel context for direct delivery.
    /// When present, ZeroClaw sends replies directly to the channel (e.g. Teams)
    /// instead of returning them in the HTTP body.
    /// Wire format: `"teams_context": { ... }` — preserved for router compatibility.
    #[serde(default, rename = "teams_context")]
    pub reply_channel: Option<serde_json::Value>,
}

fn api_chat_memory_key() -> String {
    format!("api_chat_msg_{}", Uuid::new_v4())
}

/// Derive a session state file path from a session ID.
/// This enables conversation history persistence across `/api/chat` calls
/// with the same session_id, matching how channels maintain history.
fn session_state_path(config: &crate::config::Config, session_id: Option<&str>) -> Option<PathBuf> {
    let sid = session_id?;
    if sid.is_empty() {
        return None;
    }
    // Sanitize session ID for use as filename (replace non-alphanumeric with _)
    let safe_id: String = sid
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let dir = PathBuf::from(&config.workspace_dir)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("sessions")
        .join("api_chat");
    Some(dir.join(format!("{safe_id}.json")))
}

/// Record LlmResponse + RequestLatency + AgentEnd observability events.
fn record_completion(
    state: &AppState,
    provider: &str,
    model: &str,
    duration: Duration,
    error: Option<&str>,
) {
    state
        .observer
        .record_event(&crate::observability::ObserverEvent::LlmResponse {
            provider: provider.to_string(),
            model: model.to_string(),
            duration,
            success: error.is_none(),
            error_message: error.map(String::from),
            input_tokens: None,
            output_tokens: None,
        });
    state
        .observer
        .record_metric(&crate::observability::traits::ObserverMetric::RequestLatency(
            duration,
        ));
    state
        .observer
        .record_event(&crate::observability::ObserverEvent::AgentEnd {
            provider: provider.to_string(),
            model: model.to_string(),
            duration,
            tokens_used: None,
            cost_usd: None,
        });
}

/// Sanitize an agent response before sending to the caller.
fn sanitize_response(response: &str) -> String {
    crate::channels::sanitize_channel_response(response, &[])
}

/// Detect error strings that mean the session history on disk is structurally
/// unusable for this provider (tool_use blocks without matching tool_result,
/// adjacent same-role messages, etc.). Without repair, every subsequent turn
/// reloads the same history and hits the same 400, trapping the user in an
/// endless "Please try again" loop.
fn is_session_poisoning_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    // Anthropic's well-known failure modes when history is structurally broken.
    lower.contains("tool_use")
        || lower.contains("tool_result")
        || lower.contains("tool_call_id")
        || lower.contains("messages.")
        || lower.contains("unexpected `role`")
}

/// Reset a session by removing its on-disk history file. The next turn with
/// the same session_id will start fresh. Idempotent: missing file is not an
/// error. `reason` is logged so operators can tell apart `/new`-initiated
/// resets from auto-heal resets after provider errors. Returns true if a
/// file was actually removed.
fn reset_session_history(
    config: &crate::config::Config,
    session_id: Option<&str>,
    reason: &str,
) -> bool {
    let Some(path) = session_state_path(config, session_id) else {
        return false;
    };
    if !path.exists() {
        return false;
    }
    match std::fs::remove_file(&path) {
        Ok(()) => {
            tracing::warn!(
                session_id = session_id.unwrap_or(""),
                path = %path.display(),
                reason,
                "session history reset"
            );
            true
        }
        Err(e) => {
            tracing::error!(
                session_id = session_id.unwrap_or(""),
                path = %path.display(),
                reason,
                error = %e,
                "failed to reset session history"
            );
            false
        }
    }
}

/// User-facing reply for the poisoned-session recovery case. Tells the user
/// we reset memory without revealing internals. Stays inside SOUL rule 8
/// ("never mention API behavior in user-facing messages").
const SESSION_RESET_REPLY: &str =
    "Something got tangled in my memory of our conversation, so I reset our thread. \
     Could you send your request again?";

/// Generic fallback for provider errors that don't look like session poisoning
/// (rate limits, auth, transient network). Advice to "try again" is honest here.
const TRANSIENT_ERROR_REPLY: &str =
    "Sorry, I couldn't process your message. Please try again.";

/// Reply when the user explicitly sends `/new` to reset the conversation.
const NEW_SESSION_REPLY: &str = "Fresh start. What can I help you with?";

/// Detect an explicit user-initiated session reset. Matches `/new` (with or
/// without trailing whitespace/punctuation). Case-insensitive so `/New` works.
/// Mirrors upstream's channel-layer `/new` command but operates at the
/// `/api/chat` surface used by the Yarvis router.
fn is_new_session_command(message: &str) -> bool {
    let trimmed = message.trim();
    trimmed.eq_ignore_ascii_case("/new") || trimmed.eq_ignore_ascii_case("/reset")
}

/// Run the agent loop with session history persistence.
///
/// If a `session_id` is provided, conversation history is loaded from and saved
/// to a session file, enabling multi-turn conversations across `/api/chat` calls.
/// This matches how channels (Telegram, Discord, etc.) maintain per-sender history.
async fn run_with_session(
    config: crate::config::Config,
    message: &str,
    session_id: Option<&str>,
) -> anyhow::Result<String> {
    let session_path = session_state_path(&config, session_id);

    // Use the full `run()` function with a session state file when available.
    // This loads prior conversation history and saves after the turn,
    // matching channel behavior.
    Box::pin(crate::agent::run(
        config,
        Some(message.to_string()),
        None,  // provider_override
        None,  // model_override
        -1.0,  // temperature (-1 = use config default)
        vec![], // peripheral_overrides
        false, // interactive (false = non-interactive/daemon mode)
        session_path,
        None,  // allowed_tools
    ))
    .await
}

/// `POST /api/chat` — full agent loop with tools and memory.
///
/// Request:  `{ "message": "...", "session_id": "...", "teams_context": { ... } }`
/// Response: `{ "reply": "...", "model": "..." }` or `{ "delivered_via_channel": true }`
pub async fn handle_api_chat(
    State(state): State<AppState>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Json<ApiChatBody>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    // ── Rate limit ──
    let rate_key =
        client_key_from_request(Some(peer_addr), &headers, state.trust_forwarded_headers);
    if !state.rate_limiter.allow_webhook(&rate_key) {
        tracing::warn!("/api/chat rate limit exceeded");
        let err = serde_json::json!({
            "error": "Too many chat requests. Please retry later.",
            "retry_after": RATE_LIMIT_WINDOW_SECS,
        });
        return (StatusCode::TOO_MANY_REQUESTS, Json(err));
    }

    // ── Auth: require at least one layer for non-loopback ──
    if !state.pairing.require_pairing()
        && state.webhook_secret_hash.is_none()
        && !peer_addr.ip().is_loopback()
    {
        tracing::warn!("/api/chat: rejected unauthenticated non-loopback request");
        let err = serde_json::json!({
            "error": "Unauthorized — configure pairing or X-Webhook-Secret for non-local access"
        });
        return (StatusCode::UNAUTHORIZED, Json(err));
    }

    // ── Bearer token auth (pairing) ──
    if state.pairing.require_pairing() {
        if let Err(e) = state.auth_limiter.check_rate_limit(&rate_key) {
            tracing::warn!("/api/chat: auth rate limit exceeded for {rate_key}");
            let err = serde_json::json!({
                "error": format!("Too many auth attempts. Try again in {}s.", e.retry_after_secs),
                "retry_after": e.retry_after_secs,
            });
            return (StatusCode::TOO_MANY_REQUESTS, Json(err));
        }
        let auth = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth.strip_prefix("Bearer ").unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            state.auth_limiter.record_attempt(&rate_key);
            tracing::warn!("/api/chat: rejected — not paired / invalid bearer token");
            let err = serde_json::json!({
                "error": "Unauthorized — pair first via POST /pair, then send Authorization: Bearer <token>"
            });
            return (StatusCode::UNAUTHORIZED, Json(err));
        }
    }

    // ── Parse body ──
    let Json(chat_body) = match body {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("/api/chat JSON parse error: {e}");
            let err = serde_json::json!({
                "error": "Invalid JSON body. Expected: {\"message\": \"...\"}"
            });
            return (StatusCode::BAD_REQUEST, Json(err));
        }
    };

    let message = chat_body.message.trim();
    let session_id = chat_body
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if message.is_empty() {
        let err = serde_json::json!({ "error": "Message cannot be empty" });
        return (StatusCode::BAD_REQUEST, Json(err));
    }

    // ── Explicit session reset via /new or /reset ──
    // Handled before anything that touches the LLM: the user asked us to
    // forget the thread, so we clear on-disk history and short-circuit.
    // Still goes through the reply_channel if Teams provided one.
    if is_new_session_command(message) {
        let reset_config = state.config.lock().clone();
        let reset_happened = reset_session_history(&reset_config, session_id, "user_command");
        tracing::info!(
            session_id = session_id.unwrap_or(""),
            reset = reset_happened,
            "/api/chat: /new session reset requested"
        );

        let reply_channel: Option<Box<dyn Channel>> = chat_body
            .reply_channel
            .as_ref()
            .and_then(crate::channels::reply_channel_from_json);
        if let Some(ref ch) = reply_channel {
            let msg = crate::channels::SendMessage::new(NEW_SESSION_REPLY, "");
            let _ = ch.send(&msg).await;
            let body = serde_json::json!({
                "delivered_via_channel": true,
                "model": state.model,
                "session_id": chat_body.session_id,
                "session_reset": reset_happened,
                "command": "new",
            });
            return (StatusCode::OK, Json(body));
        }

        let body = serde_json::json!({
            "reply": NEW_SESSION_REPLY,
            "model": state.model,
            "session_id": chat_body.session_id,
            "session_reset": reset_happened,
            "command": "new",
        });
        return (StatusCode::OK, Json(body));
    }

    // ── Auto-save to memory ──
    if state.auto_save {
        let key = api_chat_memory_key();
        let _ = state
            .mem
            .store(&key, message, MemoryCategory::Conversation, session_id)
            .await;
    }

    // ── Observability ──
    let provider_label = state
        .config
        .lock()
        .default_provider
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let model_label = state.model.clone();
    let started_at = Instant::now();

    state
        .observer
        .record_event(&crate::observability::ObserverEvent::AgentStart {
            provider: provider_label.clone(),
            model: model_label.clone(),
        });
    state
        .observer
        .record_event(&crate::observability::ObserverEvent::LlmRequest {
            provider: provider_label.clone(),
            model: model_label.clone(),
            messages_count: 1,
        });

    // ── Channel direct reply path ──
    let reply_channel: Option<Box<dyn Channel>> = chat_body
        .reply_channel
        .as_ref()
        .and_then(crate::channels::reply_channel_from_json);

    if let Some(ref ch) = reply_channel {
        let _ = ch.start_typing("").await;

        let config = state.config.lock().clone();
        let result = run_with_session(config, message, session_id).await;

        let _ = ch.stop_typing("").await;

        let duration = started_at.elapsed();

        match result {
            Ok(response) => {
                let safe_response = sanitize_response(&response);

                record_completion(&state, &provider_label, &model_label, duration, None);

                let msg = crate::channels::SendMessage::new(&safe_response, "");
                let send_result = ch.send(&msg).await;

                if let Err(e) = send_result {
                    tracing::error!("/api/chat channel direct reply failed: {e}");
                    let err = serde_json::json!({
                        "error": "Failed to deliver reply via channel",
                        "reply": safe_response,
                    });
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(err));
                }

                let body = serde_json::json!({
                    "delivered_via_channel": true,
                    "model": state.model,
                    "session_id": chat_body.session_id,
                });
                return (StatusCode::OK, Json(body));
            }
            Err(e) => {
                let raw = e.to_string();
                let sanitized = providers::sanitize_api_error(&raw);
                record_completion(
                    &state,
                    &provider_label,
                    &model_label,
                    duration,
                    Some(&sanitized),
                );

                let poisoned = is_session_poisoning_error(&raw);
                let reset_config = state.config.lock().clone();
                let reset_happened = poisoned && reset_session_history(&reset_config, session_id, "poisoned_history");

                tracing::error!(
                    poisoned,
                    reset = reset_happened,
                    "/api/chat provider error (channel path): {sanitized}"
                );

                let reply_text = if reset_happened {
                    SESSION_RESET_REPLY
                } else {
                    TRANSIENT_ERROR_REPLY
                };
                let msg = crate::channels::SendMessage::new(reply_text, "");
                let _ = ch.send(&msg).await;

                let err = serde_json::json!({
                    "error": "LLM request failed",
                    "session_reset": reset_happened,
                    "delivered_via_channel": true,
                });
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(err));
            }
        }
    }

    // ── Run the full agent loop (non-channel path) ──
    let config = state.config.lock().clone();
    match run_with_session(config, message, session_id).await {
        Ok(response) => {
            let safe_response = sanitize_response(&response);
            let duration = started_at.elapsed();

            record_completion(&state, &provider_label, &model_label, duration, None);

            let body = serde_json::json!({
                "reply": safe_response,
                "model": state.model,
                "session_id": chat_body.session_id,
            });
            (StatusCode::OK, Json(body))
        }
        Err(e) => {
            let duration = started_at.elapsed();
            let raw = e.to_string();
            let sanitized = providers::sanitize_api_error(&raw);

            record_completion(
                &state,
                &provider_label,
                &model_label,
                duration,
                Some(&sanitized),
            );

            let poisoned = is_session_poisoning_error(&raw);
            let reset_config = state.config.lock().clone();
            let reset_happened = poisoned && reset_session_history(&reset_config, session_id, "poisoned_history");

            tracing::error!(
                poisoned,
                reset = reset_happened,
                "/api/chat provider error: {sanitized}"
            );

            let err = serde_json::json!({
                "error": "LLM request failed",
                "session_reset": reset_happened,
                "reply": if reset_happened { SESSION_RESET_REPLY } else { TRANSIENT_ERROR_REPLY },
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_poisoning_detector_catches_tool_use_errors() {
        // The actual Anthropic 400 from the Camino log.
        let err = "Anthropic API error (400 Bad Request): {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"messages.4: `tool_use` ids were found without `tool_result` blocks immediately after: toolu_01KMoWJVyKjAAWJDk4wGm39T. Each `tool_use` block must have a corresponding `tool_result` block in the next message.\"}}";
        assert!(is_session_poisoning_error(err));
    }

    #[test]
    fn session_poisoning_detector_catches_openai_tool_call_id_errors() {
        let err = "OpenAI error: tool_call_id not found for message";
        assert!(is_session_poisoning_error(err));
    }

    #[test]
    fn session_poisoning_detector_catches_adjacent_role_errors() {
        let err = "invalid_request_error: messages.2: unexpected `role`: \"assistant\" after \"assistant\"";
        assert!(is_session_poisoning_error(err));
    }

    #[test]
    fn session_poisoning_detector_ignores_rate_limit() {
        let err = "Anthropic API error (429): rate limit exceeded, retry after 30s";
        assert!(!is_session_poisoning_error(err));
    }

    #[test]
    fn session_poisoning_detector_ignores_auth_errors() {
        let err = "Anthropic API error (401): invalid x-api-key";
        assert!(!is_session_poisoning_error(err));
    }

    #[test]
    fn session_poisoning_detector_ignores_network_errors() {
        let err = "connection reset by peer";
        assert!(!is_session_poisoning_error(err));
    }

    #[test]
    fn reset_session_history_removes_file_when_present() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let sessions = tmp.path().join("sessions").join("api_chat");
        std::fs::create_dir_all(&sessions).unwrap();
        let session_file = sessions.join("sess-123.json");
        std::fs::write(&session_file, "{\"version\":1,\"history\":[]}").unwrap();

        let mut config = crate::config::Config::default();
        config.workspace_dir = workspace.clone();

        assert!(session_file.exists());
        let removed = reset_session_history(&config, Some("sess-123"), "test");
        assert!(removed);
        assert!(!session_file.exists());
    }

    #[test]
    fn reset_session_history_returns_false_when_no_session_id() {
        let config = crate::config::Config::default();
        assert!(!reset_session_history(&config, None, "test"));
        assert!(!reset_session_history(&config, Some(""), "test"));
    }

    #[test]
    fn reset_session_history_returns_false_when_file_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let mut config = crate::config::Config::default();
        config.workspace_dir = workspace.clone();
        assert!(!reset_session_history(&config, Some("nonexistent"), "test"));
    }

    #[test]
    fn new_session_command_matches_canonical_forms() {
        assert!(is_new_session_command("/new"));
        assert!(is_new_session_command("  /new  "));
        assert!(is_new_session_command("/New"));
        assert!(is_new_session_command("/NEW"));
        assert!(is_new_session_command("/reset"));
        assert!(is_new_session_command("/Reset"));
    }

    #[test]
    fn new_session_command_rejects_near_misses() {
        assert!(!is_new_session_command("new"));
        assert!(!is_new_session_command("/news"));
        assert!(!is_new_session_command("/new chat"));
        assert!(!is_new_session_command("please /new"));
        assert!(!is_new_session_command("what's new?"));
        assert!(!is_new_session_command(""));
        assert!(!is_new_session_command("   "));
    }
}
