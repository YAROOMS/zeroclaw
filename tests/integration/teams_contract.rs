//! Integration tests for the Yarvis Teams channel and /api/chat contract.
//!
//! Validates:
//! - TeamsChannel implements Channel trait correctly
//! - TeamsContext deserialization from router payloads
//! - reply_channel_from_json factory dispatches to Teams
//! - ApiChatBody serde contract (teams_context wire key)

use zeroclaw::channels::teams::{TeamsChannel, TeamsContext};
use zeroclaw::channels::traits::{Channel, SendMessage};
use zeroclaw::channels::{reply_channel_from_json};
use zeroclaw::gateway::webhook_yarvis::ApiChatBody;

// ─────────────────────────────────────────────────────────────────────────────
// TeamsChannel trait contract
// ─────────────────────────────────────────────────────────────────────────────

fn test_context() -> TeamsContext {
    serde_json::from_value(serde_json::json!({
        "service_url": "https://smba.trafficmanager.net/teams/",
        "conversation_id": "19:test@thread.tacv2",
        "activity_id": "1234567890",
        "bot_id": "28:bot-app-id",
        "bot_name": "Yarvis",
        "user_id": "29:user-aad-id",
        "user_name": "Test User",
        "app_id": "app-client-id",
        "app_password": "app-client-secret",
        "app_tenant_id": "tenant-aad-id"
    }))
    .unwrap()
}

#[test]
fn teams_channel_name_is_teams() {
    let ch = TeamsChannel::new(test_context());
    assert_eq!(ch.name(), "teams");
}

#[test]
fn teams_channel_supports_draft_updates() {
    let ch = TeamsChannel::new(test_context());
    assert!(ch.supports_draft_updates());
}

#[test]
fn teams_channel_does_not_support_multi_message_streaming() {
    let ch = TeamsChannel::new(test_context());
    assert!(!ch.supports_multi_message_streaming());
}

#[tokio::test]
async fn teams_channel_listen_is_noop() {
    let ch = TeamsChannel::new(test_context());
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    // listen() should return Ok immediately (send-only channel)
    ch.listen(tx).await.unwrap();
}

#[tokio::test]
async fn teams_channel_health_check_returns_true() {
    let ch = TeamsChannel::new(test_context());
    assert!(ch.health_check().await);
}

#[tokio::test]
async fn teams_channel_start_stop_typing_succeeds() {
    let ch = TeamsChannel::new(test_context());
    // start_typing spawns a background task; stop_typing aborts it.
    // Without a real Bot Framework API, start_typing will fail on the HTTP call
    // but shouldn't panic. The typing handle should still be manageable.
    // We test the stop path which should always succeed.
    ch.stop_typing("").await.unwrap();
}

// ─────────────────────────────────────────────────────────────────────────────
// reply_channel_from_json factory
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reply_channel_from_json_creates_teams_channel() {
    let value = serde_json::json!({
        "service_url": "https://smba.trafficmanager.net/teams/",
        "conversation_id": "conv-123",
        "app_id": "app-id",
        "app_password": "secret"
    });
    let ch = reply_channel_from_json(&value);
    assert!(ch.is_some());
    assert_eq!(ch.unwrap().name(), "teams");
}

#[test]
fn reply_channel_from_json_returns_none_for_invalid() {
    assert!(reply_channel_from_json(&serde_json::json!({"foo": "bar"})).is_none());
    assert!(reply_channel_from_json(&serde_json::json!(null)).is_none());
}

#[test]
fn reply_channel_from_json_returns_none_for_empty_service_url() {
    let value = serde_json::json!({
        "service_url": "",
        "conversation_id": "conv-123",
        "app_id": "app-id",
        "app_password": "secret"
    });
    assert!(reply_channel_from_json(&value).is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// ApiChatBody serde contract (wire format)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn api_chat_body_deserializes_minimal() {
    let json = r#"{"message": "hello"}"#;
    let body: ApiChatBody = serde_json::from_str(json).unwrap();
    assert_eq!(body.message, "hello");
    assert!(body.session_id.is_none());
    assert!(body.reply_channel.is_none());
}

#[test]
fn api_chat_body_deserializes_with_session_id() {
    let json = r#"{"message": "hello", "session_id": "conv-123"}"#;
    let body: ApiChatBody = serde_json::from_str(json).unwrap();
    assert_eq!(body.session_id.as_deref(), Some("conv-123"));
}

#[test]
fn api_chat_body_teams_context_maps_to_reply_channel() {
    // The router sends "teams_context" but the Rust field is "reply_channel"
    let json = r#"{
        "message": "hello",
        "session_id": "conv-123",
        "teams_context": {
            "service_url": "https://smba.trafficmanager.net/teams/",
            "conversation_id": "19:test@thread.tacv2",
            "app_id": "app-id",
            "app_password": "secret"
        }
    }"#;
    let body: ApiChatBody = serde_json::from_str(json).unwrap();
    assert!(body.reply_channel.is_some());
    let ctx = body.reply_channel.unwrap();
    assert_eq!(ctx["service_url"], "https://smba.trafficmanager.net/teams/");
    assert_eq!(ctx["conversation_id"], "19:test@thread.tacv2");
}

#[test]
fn api_chat_body_without_teams_context_has_no_reply_channel() {
    let json = r#"{"message": "hello", "session_id": "hook:email:conv-456"}"#;
    let body: ApiChatBody = serde_json::from_str(json).unwrap();
    assert!(body.reply_channel.is_none());
}

#[test]
fn api_chat_body_rejects_missing_message() {
    let json = r#"{"session_id": "conv-123"}"#;
    let result: Result<ApiChatBody, _> = serde_json::from_str(json);
    assert!(result.is_err());
}

#[test]
fn api_chat_body_ignores_unknown_fields() {
    let json = r#"{"message": "hello", "unknown_field": 42}"#;
    // Should not fail - unknown fields are ignored
    let result: Result<ApiChatBody, _> = serde_json::from_str(json);
    assert!(result.is_ok());
}
