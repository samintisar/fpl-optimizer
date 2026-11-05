//! football-data.co.uk: EPL results + pre-match odds, one CSV per season. Port of
//! `fplopt.adapters.football_data`.

use anyhow::Result;
use fplopt_core::seasons::football_data_code;

use super::FootballDataSource;
use crate::fetch::Http;

pub const BASE_URL: &str = "https://www.football-data.co.uk/mmz4281";

/// football-data CSVs start with the `Div` column (after an optional UTF-8 BOM).
///
/// Seen 2016/17-2026/27: `Div,Date,HomeTeam,...` (to 2018/19), `Div,Date,Time,...` (from
/// 2019/20), and a UTF-8 BOM before `Div` in some seasons (2021/22, 2024/25 onwards).
pub fn require_csv(content: &[u8]) -> Result<(), String> {
    let head = &content[..content.len().min(64)];
    let head = head.strip_prefix(b"\xef\xbb\xbf").unwrap_or(head);
    if String::from_utf8_lossy(head).trim_start().starts_with("Div,") {
        Ok(())
    } else {
        Err("response is not a football-data CSV".to_owned())
    }
}

pub struct FootballDataClient<'a> {
    http: &'a Http,
    base: String,
}

impl<'a> FootballDataClient<'a> {
    pub fn new(http: &'a Http, base_url: &str) -> Self {
        Self { http, base: base_url.trim_end_matches('/').to_owned() }
    }
}

impl FootballDataSource for FootballDataClient<'_> {
    /// The season's E0 (Premier League) CSV, e.g. 2016 -> mmz4281/1617/E0.csv.
    fn epl_season(&self, start_year: i32) -> Result<Vec<u8>> {
        let url = format!("{}/{}/E0.csv", self.base, football_data_code(start_year));
        Ok(self.http.get(&url, &[], Some(&require_csv), 3)?.body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::testing::{no_wait, serve};

    #[test]
    fn csv_validation() {
        assert!(require_csv(b"Div,Date,HomeTeam\nE0,13/08/16,Hull\n").is_ok());
        assert!(require_csv(b"\xef\xbb\xbfDiv,Date,Time").is_ok());
        assert!(require_csv(b"  \nDiv,Date").is_ok());
        assert!(require_csv(b"<html>").is_err());
    }

    #[test]
    fn season_url() {
        let server = serve(|req, _| (200, vec![], format!("Div,{}", req.path).into_bytes()));
        let http = no_wait();
        let fd = FootballDataClient::new(&http, &server.base);
        assert_eq!(fd.epl_season(2016).unwrap(), b"Div,/1617/E0.csv");
    }
}
