//! Best-effort heartbeat pings to an external dead-man's switch such as healthchecks.io,
//! which alerts when the pings stop (including when the whole server is down). Port of
//! `fplopt.heartbeat`.

use reqwest::blocking::Client;

use crate::http;
use crate::redact::register_secret;

/// GET `url`. Never fails, and never logs the URL: its path is usually a secret.
pub fn send_heartbeat(url: &str, client: Option<&Client>) -> bool {
    register_secret(url);
    if let Ok(parsed) = reqwest::Url::parse(url) {
        register_secret(parsed.as_str()); // the normalised form, if it differs
    }
    let owned;
    let client = match client {
        Some(client) => client,
        None => match http::client() {
            Ok(c) => {
                owned = c;
                &owned
            }
            Err(err) => {
                log::warn!("heartbeat ping failed: {}", http::failure_detail(&err));
                return false;
            }
        },
    };
    match client.get(url).send().and_then(|r| r.error_for_status()) {
        Ok(_) => {
            log::info!("heartbeat ping sent");
            true
        }
        Err(err) => {
            // Never log the error's message: it can contain the URL.
            log::warn!("heartbeat ping failed: {}", http::failure_detail(&err));
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::testing::serve_once;
    use crate::redact::redact;

    #[test]
    fn pings_the_url_with_get() {
        let (base, seen) = serve_once(200);
        let url = format!("{base}/0b4c1f7e-9a2d-4e4b-8f3a-5d6c7b8a9e01");
        assert!(send_heartbeat(&url, None));
        let seen = seen.recv().unwrap();
        assert_eq!(seen.method, "GET");
        assert_eq!(seen.path, "/0b4c1f7e-9a2d-4e4b-8f3a-5d6c7b8a9e01");
    }

    #[test]
    fn http_errors_return_false() {
        for status in [404, 500] {
            let (base, _seen) = serve_once(status);
            assert!(!send_heartbeat(&format!("{base}/ping"), None));
        }
    }

    #[test]
    fn transport_error_returns_false() {
        assert!(!send_heartbeat("http://127.0.0.1:9/ping/abc", None));
    }

    #[test]
    fn non_uuid_ping_urls_are_redacted_once_used() {
        let (base, _seen) = serve_once(200);
        let url = format!("{base}/ping/AbCdEfGhIjKlMnOpQrStUv/fplopt-archiver");
        send_heartbeat(&url, None);
        assert!(!redact(&format!("GET {url} failed")).contains("AbCdEfGhIjKlMnOpQrStUv"));
    }
}
