//! One adapter per data source (CLAUDE.md invariant), each returning raw response bytes.
//! Port of `fplopt.adapters`. The jobs take the sources as traits so tests can fake them.

pub mod espn;
pub mod football_data;
pub mod fpl;
pub mod fplcache;
pub mod odds;
pub mod vaastav;

use std::collections::BTreeMap;
use std::io::Read;

use anyhow::Result;

pub trait FplSource {
    fn bootstrap_static(&self) -> Result<Vec<u8>>;
    fn fixtures(&self) -> Result<Vec<u8>>;
    fn element_summary(&self, element_id: i64) -> Result<Vec<u8>>;
    fn event_live(&self, gw: i64) -> Result<Vec<u8>>;
}

pub trait OddsSource {
    fn epl_odds(&self) -> Result<Vec<u8>>;
}

pub trait PropsSource {
    fn epl_events(&self) -> Result<Vec<u8>>;
    fn event_props(&self, event_id: &str) -> Result<Vec<u8>>;
}

pub trait FootballDataSource {
    fn epl_season(&self, start_year: i32) -> Result<Vec<u8>>;
}

pub trait EspnSource {
    fn scoreboard(&self, league: &str, period: &str) -> Result<Vec<u8>>;
    fn summary(&self, league: &str, event_id: &str) -> Result<Vec<u8>>;
}

pub trait VaastavSource {
    /// path -> blob sha for every file at `commit`.
    fn tree(&self, commit: &str) -> Result<BTreeMap<String, String>>;
    /// The file's bytes at `commit`, verified against its blob sha.
    fn file(&self, commit: &str, path: &str, blob_sha: &str) -> Result<Vec<u8>>;
}

pub trait FplcacheSource {
    fn head_commit(&self) -> Result<String>;
    /// The repo tarball's gzip stream at `commit` (read with `fplcache::for_each_member`).
    fn tarball(&self, commit: &str) -> Result<Box<dyn Read>>;
}

/// Base URL `default`, or the `FPLOPT_<NAME>_URL` environment variable when set: lets the
/// batch 1 parity harness point the archiver at a local replay server. Not for production.
pub fn base_url(name: &str, default: &str) -> String {
    std::env::var(format!("FPLOPT_{name}_URL"))
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
        .trim_end_matches('/')
        .to_owned()
}
