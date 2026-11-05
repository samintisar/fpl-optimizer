//! ESPN's public site API (unofficial, no key): soccer scoreboards and match summaries.
//! Port of `fplopt.adapters.espn`.

use anyhow::Result;
use serde_json::Value;

use super::EspnSource;
use crate::Failure;
use crate::fetch::Http;

pub const BASE_URL: &str = "https://site.api.espn.com/apis/site/v2/sports/soccer";
/// The PL, domestic cups, the Community Shield and UEFA club competitions.
pub const LEAGUES: [&str; 8] = [
    "eng.1",
    "eng.fa",
    "eng.league_cup",
    "eng.charity",
    "uefa.champions",
    "uefa.europa",
    "uefa.europa.conf",
    "uefa.super_cup",
];
pub const SCOREBOARD_LIMIT: &str = "1000";
pub const ATTEMPTS: u32 = 3; // per request: a hanging ESPN costs at most three timeouts

/// The body is a JSON object with `key` (ESPN answers some errors with 200).
pub fn require_key(content: &[u8], key: &str) -> Result<(), String> {
    match serde_json::from_slice::<Value>(crate::fetch::strip_bom(content)) {
        Err(_) => Err("response is not JSON".to_owned()),
        Ok(Value::Object(payload)) if payload.contains_key(key) => Ok(()),
        Ok(_) => Err(format!("response has no {}", crate::py_repr_str(key))),
    }
}

pub struct EspnClient<'a> {
    http: &'a Http,
    base: String,
}

impl<'a> EspnClient<'a> {
    pub fn new(http: &'a Http, base_url: &str) -> Self {
        Self { http, base: base_url.trim_end_matches('/').to_owned() }
    }

    fn url(&self, league: &str, endpoint: &str) -> Result<String> {
        if !LEAGUES.contains(&league) {
            return Err(Failure::error(
                "ValueError",
                format!("unknown ESPN league {}", crate::py_repr_str(league)),
            ));
        }
        Ok(format!("{}/{league}/{endpoint}", self.base))
    }
}

impl EspnSource for EspnClient<'_> {
    /// Every event of a calendar year (`period` 'YYYY') or month ('YYYYMM').
    fn scoreboard(&self, league: &str, period: &str) -> Result<Vec<u8>> {
        let url = self.url(league, "scoreboard")?;
        let params = [("dates", period), ("limit", SCOREBOARD_LIMIT)];
        let validate = |body: &[u8]| require_key(body, "events");
        Ok(self.http.get(&url, &params, Some(&validate), ATTEMPTS)?.body)
    }

    /// One match: header, rosters (lineups, substitutes) and key events (subs, cards).
    fn summary(&self, league: &str, event_id: &str) -> Result<Vec<u8>> {
        let url = self.url(league, "summary")?;
        let validate = |body: &[u8]| require_key(body, "header");
        Ok(self.http.get(&url, &[("event", event_id)], Some(&validate), ATTEMPTS)?.body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::testing::{no_wait, serve};

    #[test]
    fn requests_and_validation() {
        let server = serve(|req, n| {
            let body = match (req.path.as_str(), n) {
                (_, 0) => b"{\"error\": 1}".to_vec(), // a 200 without the key: retried
                (p, _) if p.ends_with("/scoreboard") => b"{\"events\": []}".to_vec(),
                _ => b"{\"header\": {}}".to_vec(),
            };
            (200, vec![], body)
        });
        let http = no_wait();
        let espn = EspnClient::new(&http, &server.base);
        assert_eq!(espn.scoreboard("eng.fa", "202610").unwrap(), b"{\"events\": []}");
        assert_eq!(espn.summary("eng.1", "123").unwrap(), b"{\"header\": {}}");
        let seen = server.seen.lock().unwrap();
        assert_eq!(
            (seen[1].path.as_str(), seen[1].query.as_str()),
            ("/eng.fa/scoreboard", "dates=202610&limit=1000")
        );
        assert_eq!(
            (seen[2].path.as_str(), seen[2].query.as_str()),
            ("/eng.1/summary", "event=123")
        );
    }

    #[test]
    fn unknown_league_is_refused() {
        let http = no_wait();
        let err =
            EspnClient::new(&http, "http://127.0.0.1:9").scoreboard("esp.1", "2026").unwrap_err();
        assert_eq!(err.to_string(), "unknown ESPN league 'esp.1'");
    }

    #[test]
    fn key_check() {
        assert_eq!(require_key(b"[]", "events"), Err("response has no 'events'".to_owned()));
        assert_eq!(require_key(b"x", "events"), Err("response is not JSON".to_owned()));
    }
}
