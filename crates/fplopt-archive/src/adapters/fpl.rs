//! Fantasy Premier League public API. Port of `fplopt.adapters.fpl` (the archiver's
//! endpoints, and a manager's team for `fplopt_optimize::entry`).

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

    // A manager's team (public, no login). Errors carry the `FetchError` (a 404: no such
    // team; `FetchError::status`).
    pub fn entry_history(&self, entry_id: i64) -> Result<Vec<u8>> {
        self.get(&format!("entry/{entry_id}/history/"))
    }

    pub fn entry_picks(&self, entry_id: i64, gw: i64) -> Result<Vec<u8>> {
        self.get(&format!("entry/{entry_id}/event/{gw}/picks/"))
    }

    pub fn entry_transfers(&self, entry_id: i64) -> Result<Vec<u8>> {
        self.get(&format!("entry/{entry_id}/transfers/"))
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
        assert_eq!(fpl.entry_history(9).unwrap(), br#"{"p": "/api/entry/9/history/"}"#);
        assert_eq!(fpl.entry_picks(9, 4).unwrap(), br#"{"p": "/api/entry/9/event/4/picks/"}"#);
        assert_eq!(fpl.entry_transfers(9).unwrap(), br#"{"p": "/api/entry/9/transfers/"}"#);
    }

    #[test]
    fn a_missing_team_is_a_404_status() {
        let server = serve(|_, _| (404, vec![], b"{\"detail\": \"Not found.\"}".to_vec()));
        let http = no_wait();
        let fpl = FplClient::new(&http, &format!("{}/api", server.base));
        let err = fpl.entry_history(1).unwrap_err();
        let fetch = err.downcast_ref::<crate::fetch::FetchError>().unwrap();
        assert_eq!(fetch.status(), Some(404));
    }
}
