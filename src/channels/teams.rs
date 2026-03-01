//! Microsoft Teams channel — Bot Framework REST API integration.
//!
//! Unlike most ZeroClaw channels, Teams operates in a "send-only" webhook model:
//! an external router (or Azure Bot Service) receives incoming webhooks and forwards
//! messages to ZeroClaw's `/api/chat` endpoint. ZeroClaw then sends replies, typing
//! indicators, and Adaptive Cards directly to Teams via the Bot Framework REST API.
//!
//! ## Architecture
//!
//! ```text
//! Teams → Router/Azure Bot Service → POST /api/chat { teams_context }
//!                                         │
//!                                         └─ TeamsChannel::send() → Teams
//! ```
//!
//! ## Usage
//!
//! The `TeamsChannel` is constructed per-request from the `teams_context` payload
//! and used for the duration of that request. It is not a long-lived channel like
//! Slack or Discord — there is no `listen()` loop.
//!
//! ## Security
//!
//! - Bot Framework credentials (`app_id`, `app_password`) are passed per-request
//!   in the POST body, never stored on disk inside the agent container.
//! - OAuth tokens are cached in-memory only, keyed by `(app_id, tenant_id)`.
//! - The existing pairing auth on `/api/chat` prevents unauthorized callers.

use super::traits::{Channel, ChannelMessage, SendMessage};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

// ══════════════════════════════════════════════════════════════════════════════
// TeamsContext — per-request conversation context
// ══════════════════════════════════════════════════════════════════════════════

/// Teams conversation context passed from the router in the `/api/chat` body.
#[derive(Deserialize, Clone)]
pub struct TeamsContext {
    pub service_url: String,
    pub conversation_id: String,
    pub activity_id: Option<String>,
    pub bot_id: Option<String>,
    pub bot_name: Option<String>,
    pub user_id: Option<String>,
    pub user_name: Option<String>,
    pub app_id: String,
    pub app_password: String,
    pub app_tenant_id: Option<String>,
}

impl std::fmt::Debug for TeamsContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamsContext")
            .field("service_url", &self.service_url)
            .field("conversation_id", &self.conversation_id)
            .field("activity_id", &self.activity_id)
            .field("bot_id", &self.bot_id)
            .field("bot_name", &self.bot_name)
            .field("user_id", &self.user_id)
            .field("user_name", &self.user_name)
            .field("app_id", &self.app_id)
            .field("app_password", &"[REDACTED]")
            .field("app_tenant_id", &self.app_tenant_id)
            .finish()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// TeamsChannel — Channel trait implementation
// ══════════════════════════════════════════════════════════════════════════════

/// MS Teams channel using the Bot Framework REST API.
///
/// Constructed per-request from a `TeamsContext`. Implements the `Channel` trait
/// so ZeroClaw's agent loop can send replies, typing indicators, and draft
/// updates through the standard channel interface.
pub struct TeamsChannel {
    ctx: TeamsContext,
    typing_handle: Mutex<Option<JoinHandle<()>>>,
}

impl TeamsChannel {
    /// Create a new Teams channel from a per-request context.
    pub fn new(ctx: TeamsContext) -> Self {
        Self {
            ctx,
            typing_handle: Mutex::new(None),
        }
    }

    /// Access the underlying context.
    pub fn context(&self) -> &TeamsContext {
        &self.ctx
    }

    /// Send an Adaptive Card to the conversation.
    ///
    /// This is a Teams-specific extension not covered by the generic `Channel` trait.
    pub async fn send_card(&self, card: serde_json::Value) -> Result<String> {
        send_card_activity(&self.ctx, card).await
    }

    /// Construct a `TeamsChannel` from an opaque JSON blob (the reply channel payload).
    ///
    /// Returns `None` if the JSON doesn't deserialize into a valid `TeamsContext`
    /// or if required fields are empty.
    pub fn try_from_reply_channel(value: &serde_json::Value) -> Option<Box<dyn Channel>> {
        let ctx: TeamsContext = serde_json::from_value(value.clone()).ok()?;
        if ctx.service_url.is_empty() || ctx.app_id.is_empty() {
            return None;
        }
        Some(Box::new(TeamsChannel::new(ctx)))
    }
}

#[async_trait]
impl Channel for TeamsChannel {
    fn name(&self) -> &str {
        "teams"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        if let Some((card, remaining)) = extract_adaptive_card(&message.content) {
            send_card_activity(&self.ctx, card).await?;
            if !remaining.is_empty() {
                send_reply(&self.ctx, &remaining).await?;
            }
        } else {
            send_reply(&self.ctx, &message.content).await?;
        }
        Ok(())
    }

    async fn listen(&self, _tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> Result<()> {
        // Teams messages arrive via webhook to the router, not via a listen loop.
        // This is a no-op; the channel is send-only.
        tracing::debug!("TeamsChannel::listen() is a no-op — messages arrive via /api/chat");
        Ok(())
    }

    async fn health_check(&self) -> bool {
        // We could ping the Bot Framework API, but token acquisition is the
        // real health signal and that happens on first send.
        true
    }

    async fn start_typing(&self, _recipient: &str) -> Result<()> {
        let handle = spawn_typing_loop(self.ctx.clone());
        *self.typing_handle.lock() = Some(handle);
        Ok(())
    }

    async fn stop_typing(&self, _recipient: &str) -> Result<()> {
        if let Some(h) = self.typing_handle.lock().take() {
            h.abort();
        }
        Ok(())
    }

    fn supports_draft_updates(&self) -> bool {
        true
    }

    async fn send_draft(&self, message: &SendMessage) -> Result<Option<String>> {
        let activity_id = send_reply(&self.ctx, &message.content).await?;
        Ok(Some(activity_id))
    }

    async fn update_draft(
        &self,
        _recipient: &str,
        message_id: &str,
        text: &str,
    ) -> Result<Option<String>> {
        update_activity(&self.ctx, message_id, text).await?;
        Ok(None) // Same message ID
    }

    async fn finalize_draft(&self, _recipient: &str, message_id: &str, text: &str) -> Result<()> {
        update_activity(&self.ctx, message_id, text).await
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Shared HTTP client
// ══════════════════════════════════════════════════════════════════════════════

/// Returns a cached HTTP client with proxy configuration.
fn http_client() -> reqwest::Client {
    crate::config::build_runtime_proxy_client("channel.teams")
}

// ══════════════════════════════════════════════════════════════════════════════
// OAuth token cache
// ══════════════════════════════════════════════════════════════════════════════

/// Cached OAuth token with expiry.
struct CachedToken {
    token: String,
    expires_at: Instant,
}

/// Cache key: (app_id, tenant_id or "common").
type TokenCacheKey = (String, String);

fn token_cache() -> &'static Mutex<HashMap<TokenCacheKey, CachedToken>> {
    static CACHE: OnceLock<Mutex<HashMap<TokenCacheKey, CachedToken>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Token lifetime buffer — cache tokens for 50 minutes (they're valid ~60 min).
const TOKEN_CACHE_TTL: Duration = Duration::from_secs(50 * 60);

/// Acquire an OAuth token for the Bot Framework REST API.
///
/// Tokens are cached in-memory per `(app_id, tenant_id)` for ~50 minutes.
async fn get_bot_token(
    app_id: &str,
    app_password: &str,
    tenant_id: Option<&str>,
) -> Result<String> {
    let tenant = tenant_id.unwrap_or("botframework.com");
    let cache_key = (app_id.to_string(), tenant.to_string());

    // Check cache
    {
        let cache = token_cache().lock();
        if let Some(cached) = cache.get(&cache_key) {
            if cached.expires_at > Instant::now() {
                return Ok(cached.token.clone());
            }
        }
    }

    // Fetch new token
    let token_url = format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
        tenant
    );

    let resp = http_client()
        .post(&token_url)
        .form(&[
            ("grant_type", "client_credentials"),
            ("client_id", app_id),
            ("client_secret", app_password),
            ("scope", "https://api.botframework.com/.default"),
        ])
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("Bot Framework token request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Bot Framework token request returned {status}: {body}");
    }

    #[derive(Deserialize)]
    struct TokenResponse {
        access_token: String,
    }

    let token_resp: TokenResponse = resp
        .json()
        .await
        .context("Failed to parse Bot Framework token response")?;

    // Cache
    {
        let mut cache = token_cache().lock();
        cache.insert(
            cache_key,
            CachedToken {
                token: token_resp.access_token.clone(),
                expires_at: Instant::now() + TOKEN_CACHE_TTL,
            },
        );
    }

    Ok(token_resp.access_token)
}

// ══════════════════════════════════════════════════════════════════════════════
// Bot Framework REST API calls
// ══════════════════════════════════════════════════════════════════════════════

/// Normalize the service URL to ensure it ends without a trailing slash.
fn normalize_service_url(url: &str) -> &str {
    url.trim_end_matches('/')
}

/// Bot Framework activity payload for sending messages.
#[derive(Serialize)]
struct BotActivity {
    r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(rename = "textFormat", skip_serializing_if = "Option::is_none")]
    text_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attachments: Option<Vec<CardAttachment>>,
    from: ActivityAccount,
    #[serde(skip_serializing_if = "Option::is_none")]
    recipient: Option<ActivityAccount>,
    #[serde(rename = "replyToId", skip_serializing_if = "Option::is_none")]
    reply_to_id: Option<String>,
}

#[derive(Serialize)]
struct ActivityAccount {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Serialize)]
struct CardAttachment {
    #[serde(rename = "contentType")]
    content_type: &'static str,
    content: serde_json::Value,
}

/// Response from Bot Framework when creating/sending an activity.
#[derive(Deserialize)]
struct SendResponse {
    id: Option<String>,
}

/// POST or PUT a Bot Framework activity. Returns the response.
async fn post_activity(
    ctx: &TeamsContext,
    url: &str,
    activity: &BotActivity,
    method: reqwest::Method,
    timeout: Duration,
    context_msg: &'static str,
) -> Result<reqwest::Response> {
    let token = get_bot_token(&ctx.app_id, &ctx.app_password, ctx.app_tenant_id.as_deref()).await?;

    let resp = http_client()
        .request(method, url)
        .bearer_auth(&token)
        .json(activity)
        .timeout(timeout)
        .send()
        .await
        .context(context_msg)?;

    Ok(resp)
}

/// Build the activities URL for a conversation.
fn activities_url(ctx: &TeamsContext) -> String {
    let base_url = normalize_service_url(&ctx.service_url);
    format!(
        "{}/v3/conversations/{}/activities",
        base_url, ctx.conversation_id
    )
}

/// Build the `from` account for activities.
fn bot_account(ctx: &TeamsContext) -> ActivityAccount {
    ActivityAccount {
        id: ctx.bot_id.clone().unwrap_or_default(),
        name: ctx.bot_name.clone(),
    }
}

/// Build the optional `recipient` account for activities.
fn user_account(ctx: &TeamsContext) -> Option<ActivityAccount> {
    ctx.user_id.as_ref().map(|id| ActivityAccount {
        id: id.clone(),
        name: ctx.user_name.clone(),
    })
}

/// Send a text reply to a Teams conversation. Returns the activity ID.
pub async fn send_reply(ctx: &TeamsContext, text: &str) -> Result<String> {
    let url = activities_url(ctx);
    let activity = BotActivity {
        r#type: "message",
        text: Some(text.to_string()),
        text_format: Some("markdown"),
        attachments: None,
        from: bot_account(ctx),
        recipient: user_account(ctx),
        reply_to_id: ctx.activity_id.clone(),
    };

    let resp = post_activity(
        ctx,
        &url,
        &activity,
        reqwest::Method::POST,
        Duration::from_secs(15),
        "Failed to send Teams reply",
    )
    .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Teams send_reply returned {status}: {body}");
    }

    let send_resp: SendResponse = resp
        .json()
        .await
        .context("Failed to parse Teams send_reply response")?;

    Ok(send_resp.id.unwrap_or_default())
}

/// Send an Adaptive Card reply to a Teams conversation. Returns the activity ID.
pub async fn send_card_activity(ctx: &TeamsContext, card: serde_json::Value) -> Result<String> {
    let url = activities_url(ctx);
    let activity = BotActivity {
        r#type: "message",
        text: None,
        text_format: None,
        attachments: Some(vec![CardAttachment {
            content_type: "application/vnd.microsoft.card.adaptive",
            content: card,
        }]),
        from: bot_account(ctx),
        recipient: user_account(ctx),
        reply_to_id: ctx.activity_id.clone(),
    };

    let resp = post_activity(
        ctx,
        &url,
        &activity,
        reqwest::Method::POST,
        Duration::from_secs(15),
        "Failed to send Teams card",
    )
    .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Teams send_card returned {status}: {body}");
    }

    let send_resp: SendResponse = resp
        .json()
        .await
        .context("Failed to parse Teams send_card response")?;

    Ok(send_resp.id.unwrap_or_default())
}

/// Update an existing message in a Teams conversation.
pub async fn update_activity(ctx: &TeamsContext, activity_id: &str, text: &str) -> Result<()> {
    let base_url = normalize_service_url(&ctx.service_url);
    let url = format!(
        "{}/v3/conversations/{}/activities/{}",
        base_url, ctx.conversation_id, activity_id
    );

    let activity = BotActivity {
        r#type: "message",
        text: Some(text.to_string()),
        text_format: Some("markdown"),
        attachments: None,
        from: bot_account(ctx),
        recipient: None,
        reply_to_id: None,
    };

    let resp = post_activity(
        ctx,
        &url,
        &activity,
        reqwest::Method::PUT,
        Duration::from_secs(15),
        "Failed to update Teams activity",
    )
    .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Teams update_activity returned {status}: {body}");
    }

    Ok(())
}

/// Send a typing indicator to a Teams conversation.
pub async fn send_typing(ctx: &TeamsContext) -> Result<()> {
    let url = activities_url(ctx);
    let activity = BotActivity {
        r#type: "typing",
        text: None,
        text_format: None,
        attachments: None,
        from: bot_account(ctx),
        recipient: user_account(ctx),
        reply_to_id: None,
    };

    let resp = post_activity(
        ctx,
        &url,
        &activity,
        reqwest::Method::POST,
        Duration::from_secs(10),
        "Failed to send Teams typing indicator",
    )
    .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        tracing::warn!("Teams send_typing returned {status}: {body}");
    }

    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// Typing indicator loop
// ══════════════════════════════════════════════════════════════════════════════

/// Spawn a background task that sends typing indicators every 3 seconds.
///
/// Returns a `JoinHandle` that should be aborted when the agent finishes.
/// Teams typing indicators expire after ~3 seconds, so we must repeat them.
pub fn spawn_typing_loop(ctx: TeamsContext) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let _ = send_typing(&ctx).await;

        let mut interval = tokio::time::interval(Duration::from_secs(3));
        interval.tick().await; // consume the immediate tick

        loop {
            interval.tick().await;
            if let Err(e) = send_typing(&ctx).await {
                tracing::debug!("Typing indicator loop error (non-fatal): {e}");
                break;
            }
        }
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Adaptive Card extraction
// ══════════════════════════════════════════════════════════════════════════════

/// Try to extract an Adaptive Card JSON from the agent response text.
///
/// LLMs return Adaptive Cards in various forms:
/// 1. Raw JSON: `{"type": "AdaptiveCard", ...}`
/// 2. Wrapped in a markdown code block: ````json\n{...}\n````
/// 3. Code block with surrounding text: `Here's your card:\n```json\n{...}\n```\nEnjoy!`
///
/// Returns `Some((card_json, remaining_text))` if found.
/// `remaining_text` is any text outside the code block (may be empty).
pub fn extract_adaptive_card(text: &str) -> Option<(serde_json::Value, String)> {
    let trimmed = text.trim();

    // Try raw JSON first (entire response is the card)
    if let Some(card) = try_parse_adaptive_card(trimmed) {
        return Some((card, String::new()));
    }

    // Find a markdown code block anywhere in the text and check if it's an Adaptive Card.
    // Scan for ``` boundaries — handles ```json, ```, etc.
    let mut search_from = 0;
    while let Some(start) = trimmed[search_from..].find("```") {
        let abs_start = search_from + start;
        // Skip past the opening ``` and any language tag (e.g., "json")
        let after_backticks = abs_start + 3;
        // Find the end of the opening line (language hint)
        let content_start = trimmed[after_backticks..]
            .find('\n')
            .map(|i| after_backticks + i + 1)
            .unwrap_or(after_backticks);

        // Find closing ```
        if let Some(end_offset) = trimmed[content_start..].find("```") {
            let content_end = content_start + end_offset;
            let inner = trimmed[content_start..content_end].trim();

            if let Some(card) = try_parse_adaptive_card(inner) {
                // Collect text before and after the code block
                let block_end = content_end + 3;
                let before = trimmed[..abs_start].trim();
                let after = trimmed[block_end..].trim();
                let remaining = match (before.is_empty(), after.is_empty()) {
                    (true, true) => String::new(),
                    (false, true) => before.to_string(),
                    (true, false) => after.to_string(),
                    (false, false) => format!("{before}\n{after}"),
                };
                return Some((card, remaining));
            }

            // Not an Adaptive Card, keep searching after this block
            search_from = content_end + 3;
        } else {
            break;
        }
    }

    None
}

fn try_parse_adaptive_card(text: &str) -> Option<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    if value.get("type")?.as_str()? == "AdaptiveCard" {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_service_url_strips_trailing_slash() {
        assert_eq!(
            normalize_service_url("https://smba.trafficmanager.net/teams/"),
            "https://smba.trafficmanager.net/teams"
        );
        assert_eq!(
            normalize_service_url("https://smba.trafficmanager.net/teams"),
            "https://smba.trafficmanager.net/teams"
        );
    }

    #[test]
    fn teams_context_deserializes_minimal() {
        let json = r#"{
            "service_url": "https://smba.trafficmanager.net/teams/",
            "conversation_id": "conv-123",
            "app_id": "app-id",
            "app_password": "app-password"
        }"#;
        let ctx: TeamsContext = serde_json::from_str(json).unwrap();
        assert_eq!(ctx.conversation_id, "conv-123");
        assert!(ctx.activity_id.is_none());
        assert!(ctx.app_tenant_id.is_none());
    }

    #[test]
    fn teams_context_deserializes_full() {
        let json = r#"{
            "service_url": "https://smba.trafficmanager.net/teams/",
            "conversation_id": "conv-123",
            "activity_id": "act-456",
            "bot_id": "bot-789",
            "bot_name": "ZeroClaw",
            "user_id": "user-abc",
            "user_name": "Test User",
            "app_id": "app-id",
            "app_password": "app-password",
            "app_tenant_id": "tenant-xyz"
        }"#;
        let ctx: TeamsContext = serde_json::from_str(json).unwrap();
        assert_eq!(ctx.activity_id.as_deref(), Some("act-456"));
        assert_eq!(ctx.bot_id.as_deref(), Some("bot-789"));
        assert_eq!(ctx.app_tenant_id.as_deref(), Some("tenant-xyz"));
    }

    #[test]
    fn teams_context_debug_redacts_password() {
        let ctx = TeamsContext {
            service_url: "https://example.com".to_string(),
            conversation_id: "conv".to_string(),
            activity_id: None,
            bot_id: None,
            bot_name: None,
            user_id: None,
            user_name: None,
            app_id: "id".to_string(),
            app_password: "super-secret-password".to_string(),
            app_tenant_id: None,
        };
        let debug_output = format!("{:?}", ctx);
        assert!(!debug_output.contains("super-secret-password"));
        assert!(debug_output.contains("[REDACTED]"));
    }

    #[test]
    fn bot_activity_serializes_text_message() {
        let activity = BotActivity {
            r#type: "message",
            text: Some("Hello!".to_string()),
            text_format: Some("markdown"),
            attachments: None,
            from: ActivityAccount {
                id: "bot-1".to_string(),
                name: Some("Bot".to_string()),
            },
            recipient: Some(ActivityAccount {
                id: "user-1".to_string(),
                name: None,
            }),
            reply_to_id: Some("act-1".to_string()),
        };
        let json = serde_json::to_value(&activity).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["text"], "Hello!");
        assert_eq!(json["textFormat"], "markdown");
        assert_eq!(json["replyToId"], "act-1");
        assert!(json.get("attachments").is_none());
    }

    #[test]
    fn bot_activity_serializes_typing() {
        let activity = BotActivity {
            r#type: "typing",
            text: None,
            text_format: None,
            attachments: None,
            from: ActivityAccount {
                id: "bot-1".to_string(),
                name: None,
            },
            recipient: None,
            reply_to_id: None,
        };
        let json = serde_json::to_value(&activity).unwrap();
        assert_eq!(json["type"], "typing");
        assert!(json.get("text").is_none());
        assert!(json.get("recipient").is_none());
    }

    #[test]
    fn bot_activity_serializes_card_attachment() {
        let card = serde_json::json!({
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [{"type": "TextBlock", "text": "Hello"}]
        });
        let activity = BotActivity {
            r#type: "message",
            text: None,
            text_format: None,
            attachments: Some(vec![CardAttachment {
                content_type: "application/vnd.microsoft.card.adaptive",
                content: card.clone(),
            }]),
            from: ActivityAccount {
                id: "bot-1".to_string(),
                name: None,
            },
            recipient: None,
            reply_to_id: None,
        };
        let json = serde_json::to_value(&activity).unwrap();
        assert_eq!(
            json["attachments"][0]["contentType"],
            "application/vnd.microsoft.card.adaptive"
        );
        assert_eq!(json["attachments"][0]["content"]["type"], "AdaptiveCard");
    }

    #[test]
    fn teams_channel_name() {
        let ctx = TeamsContext {
            service_url: "https://example.com".to_string(),
            conversation_id: "conv".to_string(),
            activity_id: None,
            bot_id: None,
            bot_name: None,
            user_id: None,
            user_name: None,
            app_id: "id".to_string(),
            app_password: "pass".to_string(),
            app_tenant_id: None,
        };
        let channel = TeamsChannel::new(ctx);
        assert_eq!(channel.name(), "teams");
        assert!(channel.supports_draft_updates());
    }

    // ── Adaptive Card extraction tests ──

    #[test]
    fn extract_adaptive_card_raw_json() {
        let text = r#"{"type": "AdaptiveCard", "version": "1.5", "body": []}"#;
        let (card, remaining) = extract_adaptive_card(text).unwrap();
        assert_eq!(card["type"], "AdaptiveCard");
        assert_eq!(card["version"], "1.5");
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_adaptive_card_markdown_code_block() {
        let text = "```json\n{\"type\": \"AdaptiveCard\", \"version\": \"1.5\", \"body\": []}\n```";
        let (card, remaining) = extract_adaptive_card(text).unwrap();
        assert_eq!(card["type"], "AdaptiveCard");
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_adaptive_card_plain_code_block() {
        let text = "```\n{\"type\": \"AdaptiveCard\", \"version\": \"1.5\", \"body\": []}\n```";
        let (card, remaining) = extract_adaptive_card(text).unwrap();
        assert_eq!(card["type"], "AdaptiveCard");
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_adaptive_card_with_surrounding_text() {
        let text = "Here's your card:\n```json\n{\"type\": \"AdaptiveCard\", \"version\": \"1.5\", \"body\": []}\n```\nUpdated the title!";
        let (card, remaining) = extract_adaptive_card(text).unwrap();
        assert_eq!(card["type"], "AdaptiveCard");
        assert_eq!(remaining, "Here's your card:\nUpdated the title!");
    }

    #[test]
    fn extract_adaptive_card_text_only_after() {
        let text = "```json\n{\"type\": \"AdaptiveCard\", \"version\": \"1.5\", \"body\": []}\n```\nEnjoy!";
        let (card, remaining) = extract_adaptive_card(text).unwrap();
        assert_eq!(card["type"], "AdaptiveCard");
        assert_eq!(remaining, "Enjoy!");
    }

    #[test]
    fn extract_adaptive_card_not_a_card() {
        assert!(extract_adaptive_card("Hello, how can I help?").is_none());
        assert!(extract_adaptive_card(r#"{"type": "other", "data": 1}"#).is_none());
        assert!(extract_adaptive_card("```json\n{\"key\": \"value\"}\n```").is_none());
    }

    #[test]
    fn extract_adaptive_card_with_whitespace() {
        let text = "  \n{\"type\": \"AdaptiveCard\", \"version\": \"1.5\", \"body\": []}\n  ";
        let (card, _) = extract_adaptive_card(text).unwrap();
        assert_eq!(card["type"], "AdaptiveCard");
    }

    // ── Reply channel factory tests ──

    #[test]
    fn try_from_reply_channel_valid() {
        let value = serde_json::json!({
            "service_url": "https://smba.trafficmanager.net/teams/",
            "conversation_id": "conv-123",
            "app_id": "app-id",
            "app_password": "app-password"
        });
        let ch = TeamsChannel::try_from_reply_channel(&value);
        assert!(ch.is_some());
        assert_eq!(ch.unwrap().name(), "teams");
    }

    #[test]
    fn try_from_reply_channel_empty_service_url() {
        let value = serde_json::json!({
            "service_url": "",
            "conversation_id": "conv-123",
            "app_id": "app-id",
            "app_password": "app-password"
        });
        assert!(TeamsChannel::try_from_reply_channel(&value).is_none());
    }

    #[test]
    fn try_from_reply_channel_empty_app_id() {
        let value = serde_json::json!({
            "service_url": "https://smba.trafficmanager.net/teams/",
            "conversation_id": "conv-123",
            "app_id": "",
            "app_password": "app-password"
        });
        assert!(TeamsChannel::try_from_reply_channel(&value).is_none());
    }

    #[test]
    fn try_from_reply_channel_invalid_json() {
        let value = serde_json::json!({"foo": "bar"});
        assert!(TeamsChannel::try_from_reply_channel(&value).is_none());
    }
}
