//! Masking secrets (API keys, bot tokens, heartbeat ping URLs) out of log output. Port of
//! `fplopt.redact`; Rust has no httpx logger to filter, so callers redact what they log.

use std::collections::BTreeSet;
use std::sync::{LazyLock, Mutex};

use regex::Regex;

static SECRET_QUERY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)\b(apikey|api_key|token)=[^&\s"']+"#).unwrap());
static BOT_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/bot[^/]+/").unwrap());
// healthchecks.io ping URLs are https://hc-ping.com/<uuid>: the UUID is the secret.
static UUID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b").unwrap()
});

static SECRETS: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// Mask this exact string from now on, e.g. a ping URL whose path is a secret but not a
/// UUID (healthchecks.io ping-key URLs, self-hosted instances).
pub fn register_secret(value: &str) {
    let value = value.trim();
    if value.chars().count() >= 8 {
        // never mask short, common strings
        SECRETS.lock().unwrap().insert(value.to_owned());
    }
}

/// Mask registered secrets, UUIDs, API keys in query strings and Telegram bot tokens in
/// URL paths.
pub fn redact(text: &str) -> String {
    let mut text = text.to_owned();
    let mut secrets: Vec<String> = SECRETS.lock().unwrap().iter().cloned().collect();
    secrets.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
    for secret in secrets {
        text = text.replace(&secret, "REDACTED");
    }
    let text = UUID.replace_all(&text, "REDACTED");
    let text = SECRET_QUERY.replace_all(&text, "${1}=REDACTED");
    BOT_TOKEN.replace_all(&text, "/botREDACTED/").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_query_keys_and_bot_tokens() {
        assert_eq!(
            redact("https://x.test/v4/odds/?apiKey=abc123&regions=uk"),
            "https://x.test/v4/odds/?apiKey=REDACTED&regions=uk"
        );
        assert_eq!(
            redact("https://api.telegram.org/bot123:AA-bb_c/sendMessage"),
            "https://api.telegram.org/botREDACTED/sendMessage"
        );
    }

    #[test]
    fn masks_malformed_bot_tokens() {
        assert_eq!(
            redact("POST https://api.telegram.org/bot12 3:x#y/sendMessage"),
            "POST https://api.telegram.org/botREDACTED/sendMessage"
        );
    }

    #[test]
    fn masks_uuids() {
        let uuid = "0b4c1f7e-9a2d-4e4b-8f3a-5d6c7b8a9e01";
        assert!(!redact(&format!("GET https://hc-ping.com/{uuid} 200")).contains(uuid));
    }

    #[test]
    fn masks_registered_secrets_but_not_short_ones() {
        register_secret("https://hc.example.test/ping/AbCdEfGhIjKl/slug");
        register_secret("short");
        let masked = redact("GET https://hc.example.test/ping/AbCdEfGhIjKl/slug failed short");
        assert_eq!(masked, "GET REDACTED failed short");
    }
}
