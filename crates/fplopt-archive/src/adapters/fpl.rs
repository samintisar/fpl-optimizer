//! Fantasy Premier League public API. Port of `fplopt.adapters.fpl` (the archiver's
//! endpoints; the entry endpoints move with `fplopt.entry` in batch 4).

use anyhow::Result;

use super::FplSource;
use crate::fetch::{DEFAULT_ATTEMPTS, Http};

pub const BASE_URL: &str = "https://fantasy.premierleague.com/api";

pub struct FplClient<'a> {
    http: &'a Http,
    base: String,
}

impl<'a> FplClient<'a> {
    pub fn new(http: &'a Http, base_url: &str) -> Self {
        Self { http, base: base_url.trim_end_matches('/').to_owned() }
    }

    fn get(&self, path: &str) -> Result<Vec<u8>> {
        Ok(self.http.get_json(&format!("{}/{path}", self.base), &[], DEFAULT_ATTEMPTS)?.body)
    }
}

impl FplSource for FplClient<'_> {
    fn bootstrap_static(&self) -> Result<Vec<u8>> {
        self.get("bootstrap-static/")
    }

    fn fixtures(&self) -> Result<Vec<u8>> {
        self.get("fixtures/")
    }

    fn element_summary(&self, element_id: i64) -> Result<Vec<u8>> {
        self.get(&format!("element-summary/{element_id}/"))
    }

    fn event_live(&self, gw: i64) -> Result<Vec<u8>> {
        self.get(&format!("event/{gw}/live/"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::testing::{no_wait, serve};

    #[test]
    fn endpoints() {
        let server =
            serve(|req, _| (200, vec![], format!("{{\"p\": \"{}\"}}", req.path).into_bytes()));
        let http = no_wait();
        let fpl = FplClient::new(&http, &format!("{}/api/", server.base));
        assert_eq!(fpl.bootstrap_static().unwrap(), br#"{"p": "/api/bootstrap-static/"}"#);
        assert_eq!(fpl.fixtures().unwrap(), br#"{"p": "/api/fixtures/"}"#);
        assert_eq!(fpl.element_summary(7).unwrap(), br#"{"p": "/api/element-summary/7/"}"#);
        assert_eq!(fpl.event_live(3).unwrap(), br#"{"p": "/api/event/3/live/"}"#);
    }
}
