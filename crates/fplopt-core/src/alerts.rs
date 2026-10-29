//! Admin alerts via the Telegram Bot API (plain HTTP; no bot framework needed). Port of
//! `fplopt.alerts`.

use reqwest::blocking::Client;
use serde_json::json;

use crate::http;
use crate::redact::redact;

pub const MAX_LEN: usize = 4000; // Telegram's limit is 4096 characters
const API: &str = "https://api.telegram.org";

/// Best effort: never fails, so a failed alert can't hide the original error.
pub fn send_admin_alert(text: &str, token: Option<&str>, chat_id: Option<&str>) -> bool {
    send_admin_alert_via(API, None, text, token, chat_id)
}

/// `send_admin_alert` against another API base, with an optional client (tests).
pub fn send_admin_alert_via(
    api: &str,
    client: Option<&Client>,
    text: &str,
    token: Option<&str>,
    chat_id: Option<&str>,
) -> bool {
    let configured = (token.filter(|t| !t.is_empty()), chat_id.filter(|c| !c.is_empty()));
    let (Some(token), Some(chat_id)) = configured else {
        log::warn!("telegram alert not configured; message was: {}", redact(text));
        return false;
    };
    let owned;
    let client = match client {
        Some(client) => client,
        None => match http::client() {
            Ok(c) => {
                owned = c;
                &owned
            }
            Err(err) => {
                log::error!("telegram alert failed: {}", http::failure_detail(&err));
                return false;
            }
        },
    };
    let text: String = text.chars().take(MAX_LEN).collect();
    let result = client
        .post(format!("{api}/bot{}/sendMessage", token.trim()))
        .json(&json!({"chat_id": chat_id, "text": text}))
        .send()
        .and_then(|response| response.error_for_status());
    match result {
        Ok(_) => true,
        Err(err) => {
            // Never log the error's message: it can contain the bot token from the URL.
            log::error!("telegram alert failed: {}", http::failure_detail(&err));
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::testing::serve_once;
    use serde_json::Value;

    #[test]
    fn posts_message_to_chat() {
        let (base, seen) = serve_once(200);
        assert!(send_admin_alert_via(&base, None, "boom", Some("T"), Some("42")));
        let seen = seen.recv().unwrap();
        assert_eq!((seen.method.as_str(), seen.path.as_str()), ("POST", "/botT/sendMessage"));
        let body: Value = serde_json::from_str(&seen.body).unwrap();
        assert_eq!(body, json!({"chat_id": "42", "text": "boom"}));
    }

    #[test]
    fn unconfigured_is_noop() {
        assert!(!send_admin_alert("boom", None, Some("42")));
        assert!(!send_admin_alert("boom", Some("T"), None));
    }

    #[test]
    fn failure_returns_false() {
        let (base, _seen) = serve_once(500);
        assert!(!send_admin_alert_via(&base, None, "boom", Some("T"), Some("42")));
    }

    #[test]
    fn long_messages_are_truncated() {
        let (base, seen) = serve_once(200);
        send_admin_alert_via(&base, None, &"x".repeat(5000), Some("T"), Some("42"));
        let body: Value = serde_json::from_str(&seen.recv().unwrap().body).unwrap();
        assert_eq!(body["text"].as_str().unwrap().len(), MAX_LEN);
    }

    #[test]
    fn transport_error_returns_false() {
        // Port 9 on localhost: nothing listens, the connection is refused.
        assert!(!send_admin_alert_via("http://127.0.0.1:9", None, "b", Some("T"), Some("42")));
    }

    #[test]
    fn failure_detail_never_contains_the_token() {
        let (base, _seen) = serve_once(401);
        let err = Client::new()
            .post(format!("{base}/bot123:SECRET/sendMessage"))
            .send()
            .and_then(|r| r.error_for_status())
            .unwrap_err();
        let detail = http::failure_detail(&err);
        assert!(!detail.contains("SECRET"));
        assert!(detail.contains("HTTP 401"));
    }
}
