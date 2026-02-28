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

// ══════════════════════════════════════════════════════════════════════════════
// TeamsContext — per-request conversation context
// ══════════════════════════════════════════════════════════════════════════════

/// Teams conversation context passed from the router in the `/api/chat` body.
#[derive(Debug, Deserialize, Clone)]
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
}

impl TeamsChannel {
    /// Create a new Teams channel from a per-request context.
    pub fn new(ctx: TeamsContext) -> Self {
        Self { ctx }
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
}

#[async_trait]
impl Channel for TeamsChannel {
    fn name(&self) -> &str {
        "teams"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        send_reply(&self.ctx, &message.content).await?;
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
        send_typing(&self.ctx).await
    }

    async fn stop_typing(&self, _recipient: &str) -> Result<()> {
        // Teams typing indicators expire naturally after ~3 seconds.
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

    async fn finalize_draft(
        &self,
        _recipient: &str,
        message_id: &str,
        text: &str,
    ) -> Result<()> {
        update_activity(&self.ctx, message_id, text).await
    }
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

    let client = reqwest::Client::new();
    let resp = client
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
    content_type: String,
    content: serde_json::Value,
}

/// Send a text reply to a Teams conversation. Returns the activity ID.
pub async fn send_reply(ctx: &TeamsContext, text: &str) -> Result<String> {
    let token = get_bot_token(&ctx.app_id, &ctx.app_password, ctx.app_tenant_id.as_deref()).await?;
    let base_url = normalize_service_url(&ctx.service_url);
    let url = format!(
        "{}/v3/conversations/{}/activities",
        base_url, ctx.conversation_id
    );

    let activity = BotActivity {
        r#type: "message",
        text: Some(text.to_string()),
        text_format: Some("markdown"),
        attachments: None,
        from: ActivityAccount {
            id: ctx.bot_id.clone().unwrap_or_default(),
            name: ctx.bot_name.clone(),
        },
        recipient: ctx.user_id.as_ref().map(|id| ActivityAccount {
            id: id.clone(),
            name: ctx.user_name.clone(),
        }),
        reply_to_id: ctx.activity_id.clone(),
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .bearer_auth(&token)
        .json(&activity)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .context("Failed to send Teams reply")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Teams send_reply returned {status}: {body}");
    }

    #[derive(Deserialize)]
    struct SendResponse {
        id: Option<String>,
    }

    let send_resp: SendResponse = resp
        .json()
        .await
        .context("Failed to parse Teams send_reply response")?;

    Ok(send_resp.id.unwrap_or_default())
}

/// Send an Adaptive Card reply to a Teams conversation. Returns the activity ID.
pub async fn send_card_activity(ctx: &TeamsContext, card: serde_json::Value) -> Result<String> {
    let token = get_bot_token(&ctx.app_id, &ctx.app_password, ctx.app_tenant_id.as_deref()).await?;
    let base_url = normalize_service_url(&ctx.service_url);
    let url = format!(
        "{}/v3/conversations/{}/activities",
        base_url, ctx.conversation_id
    );

    let activity = BotActivity {
        r#type: "message",
        text: None,
        text_format: None,
        attachments: Some(vec![CardAttachment {
            content_type: "application/vnd.microsoft.card.adaptive".to_string(),
            content: card,
        }]),
        from: ActivityAccount {
            id: ctx.bot_id.clone().unwrap_or_default(),
            name: ctx.bot_name.clone(),
        },
        recipient: ctx.user_id.as_ref().map(|id| ActivityAccount {
            id: id.clone(),
            name: ctx.user_name.clone(),
        }),
        reply_to_id: ctx.activity_id.clone(),
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .bearer_auth(&token)
        .json(&activity)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .context("Failed to send Teams card")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Teams send_card returned {status}: {body}");
    }

    #[derive(Deserialize)]
    struct SendResponse {
        id: Option<String>,
    }

    let send_resp: SendResponse = resp
        .json()
        .await
        .context("Failed to parse Teams send_card response")?;

    Ok(send_resp.id.unwrap_or_default())
}

/// Update an existing message in a Teams conversation.
pub async fn update_activity(ctx: &TeamsContext, activity_id: &str, text: &str) -> Result<()> {
    let token = get_bot_token(&ctx.app_id, &ctx.app_password, ctx.app_tenant_id.as_deref()).await?;
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
        from: ActivityAccount {
            id: ctx.bot_id.clone().unwrap_or_default(),
            name: ctx.bot_name.clone(),
        },
        recipient: None,
        reply_to_id: None,
    };

    let client = reqwest::Client::new();
    let resp = client
        .put(&url)
        .bearer_auth(&token)
        .json(&activity)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .context("Failed to update Teams activity")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Teams update_activity returned {status}: {body}");
    }

    Ok(())
}

/// Send a typing indicator to a Teams conversation.
pub async fn send_typing(ctx: &TeamsContext) -> Result<()> {
    let token = get_bot_token(&ctx.app_id, &ctx.app_password, ctx.app_tenant_id.as_deref()).await?;
    let base_url = normalize_service_url(&ctx.service_url);
    let url = format!(
        "{}/v3/conversations/{}/activities",
        base_url, ctx.conversation_id
    );

    let activity = BotActivity {
        r#type: "typing",
        text: None,
        text_format: None,
        attachments: None,
        from: ActivityAccount {
            id: ctx.bot_id.clone().unwrap_or_default(),
            name: ctx.bot_name.clone(),
        },
        recipient: ctx.user_id.as_ref().map(|id| ActivityAccount {
            id: id.clone(),
            name: ctx.user_name.clone(),
        }),
        reply_to_id: None,
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .bearer_auth(&token)
        .json(&activity)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("Failed to send Teams typing indicator")?;

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
                content_type: "application/vnd.microsoft.card.adaptive".to_string(),
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
        assert_eq!(json["attachments"][0]["contentType"], "application/vnd.microsoft.card.adaptive");
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
}
