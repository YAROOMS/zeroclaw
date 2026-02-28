//! Gateway re-exports for Teams channel helpers.
//!
//! The actual implementation lives in [`crate::channels::teams`]. This module
//! re-exports the public API so existing gateway code (`openclaw_compat.rs`)
//! can continue importing from `super::teams_reply` without change.

pub use crate::channels::teams::{
    send_card_activity as send_card, send_reply, spawn_typing_loop, TeamsContext,
};
