//! The Odds API (v4): live EPL match odds. Port of `fplopt.adapters.odds`.

use anyhow::Result;
use reqwest::header::HeaderMap;
use serde_json::Value;

use super::OddsSource;
use crate::Failure;
use crate::fetch::{FetchError, Http};

pub const BASE_URL: &str = "https://api.the-odds-api.com/v4";
pub const SPORT: &str = "soccer_epl";
pub const MARKETS: [&str; 2] = ["h2h", "totals"];
pub const REGIONS: [&str; 1] = ["uk"];
pub const LOW_CREDIT_WARNING: f64 = 50.0;

pub struct OddsClient<'a> {
    http: &'a Http,
    api_key: String,
    base: String,
}

impl<'a> OddsClient<'a> {
    pub fn new(http: &'a Http, api_key: &str, base_url: &str) -> Self {
        fplopt_core::redact::register_secret(api_key);
        Self { http, api_key: api_key.to_owned(), base: base_url.trim_end_matches('/').to_owned() }
    }
}

impl OddsSource for OddsClient<'_> {
    /// Current EPL odds. Costs len(MARKETS) * len(REGIONS) credits per call. Errors
    /// (`OddsApiError`) never contain the request URL (it holds the key).
    fn epl_odds(&self) -> Result<Vec<u8>> {
        let url = format!("{}/sports/{SPORT}/odds/", self.base);
        let (regions, markets) = (REGIONS.join(","), MARKETS.join(","));
        let params = [
            ("apiKey", self.api_key.as_str()),
            ("regions", regions.as_str()),
            ("markets", markets.as_str()),
            ("oddsFormat", "decimal"),
            ("dateFormat", "iso"),
        ];
        match self.http.get_json(&url, &params, 3) {
            Ok(response) => {
                log_credits(&response.headers);
                Ok(response.body)
            }
            // Status errors are reported; anything else (transport, invalid payload) is
            // passed on as is: its message names the URL without the query.
            Err(FetchError::Status { status, body, .. }) => {
                let failure = format!("HTTP {status} {}", error_code(&body));
                Err(Failure::error(
                    "OddsApiError",
                    format!("odds request failed: {}", failure.trim_end()),
                ))
            }
            Err(err) => Err(err.into()),
        }
    }
}

/// The API's `error_code` (e.g. OUT_OF_USAGE_CREDITS), never echoing anything else.
fn error_code(body: &[u8]) -> String {
    let Ok(Value::Object(payload)) = serde_json::from_slice::<Value>(body) else {
        return String::new();
    };
    match payload.get("error_code") {
        Some(Value::String(code)) if is_identifier(code) => code.clone(),
        _ => String::new(),
    }
}

/// Python's `str.isidentifier`, near enough for API error codes.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c == '_' || c.is_alphabetic())
        && chars.all(|c| c == '_' || c.is_alphanumeric())
}

fn log_credits(headers: &HeaderMap) {
    let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let remaining = get("x-requests-remaining");
    let show = |v: &Option<String>| v.clone().unwrap_or_else(|| "None".to_owned());
    log::info!(
        "odds api credits remaining={} used={} last={}",
        show(&remaining),
        show(&get("x-requests-used")),
        show(&get("x-requests-last"))
    );
    let low = remaining.as_deref().and_then(|r| r.trim().parse::<f64>().ok());
    if low.is_some_and(|r| r < LOW_CREDIT_WARNING) {
        log::warn!("odds api credits low: {} remaining this month", show(&remaining));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::testing::{no_wait, serve};

    #[test]
    fn sends_params_and_returns_body() {
        let server =
            serve(|_, _| (200, vec![("x-requests-remaining", "10".into())], b"[]".to_vec()));
        let http = no_wait();
        let odds = OddsClient::new(&http, "KEY12345", &server.base);
        assert_eq!(odds.epl_odds().unwrap(), b"[]");
        let seen = server.seen.lock().unwrap();
        assert_eq!(seen[0].path, "/sports/soccer_epl/odds/");
        assert_eq!(
            seen[0].query,
            "apiKey=KEY12345&regions=uk&markets=h2h%2Ctotals&oddsFormat=decimal&dateFormat=iso"
        );
    }

    #[test]
    fn errors_name_the_code_but_never_the_key() {
        let body =
            br#"{"message": "Usage quota has been reached", "error_code": "OUT_OF_USAGE_CREDITS"}"#;
        let server = serve(move |_, _| (401, vec![], body.to_vec()));
        let http = no_wait();
        let err = OddsClient::new(&http, "KEY12345", &server.base).epl_odds().unwrap_err();
        assert_eq!(crate::kind_name(&err), "OddsApiError");
        assert_eq!(err.to_string(), "odds request failed: HTTP 401 OUT_OF_USAGE_CREDITS");
        assert_eq!(server.requests(), 1);
    }

    #[test]
    fn odd_error_codes_are_dropped() {
        assert_eq!(error_code(br#"{"error_code": "a b"}"#), "");
        assert_eq!(error_code(b"<html>"), "");
        assert_eq!(error_code(br#"{"error_code": "INVALID_KEY"}"#), "INVALID_KEY");
    }
}
