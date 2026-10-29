//! Shared build plumbing: the build context, raw CSV reading (pandas `read_csv` semantics
//! where the builders rely on them), run discovery and UK time helpers. Port of
//! `fplopt.build.common`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Europe::London;
use fplopt_archive::raw_store::{RawStore, TS_PATTERN, parse_ts};
use serde_json::Value;

use crate::table::{Days, Micros, Table, read_parquet};

pub const MANIFEST: &str = "_manifest.json.gz";
pub const HOUR: Micros = 3_600_000_000;
pub const DAY: Micros = 24 * HOUR;
/// State fixed before deadline t is available this long before it (PLAN §4).
pub const KNOWN_BEFORE_DEADLINE: Micros = HOUR;
pub const EPOCH: Micros = 0;

/// What a builder reads from: the raw store, earlier-built tables in `data_dir` (cached),
/// and the hand-maintained configs in `config_dir`.
pub struct BuildContext {
    pub store: RawStore,
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    cache: HashMap<String, Table>,
}

impl BuildContext {
    pub fn new(
        store: RawStore,
        data_dir: impl Into<PathBuf>,
        config_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            store,
            data_dir: data_dir.into(),
            config_dir: config_dir.into(),
            cache: HashMap::new(),
        }
    }

    pub fn table_path(&self, name: &str) -> PathBuf {
        self.data_dir.join(format!("{name}.parquet"))
    }

    pub fn table(&mut self, name: &str) -> Result<&Table> {
        if !self.cache.contains_key(name) {
            let path = self.table_path(name);
            if !path.exists() {
                bail!(
                    "table {name:?} has not been built ({}); run `fplopt build {name}` first",
                    path.display()
                );
            }
            let table = read_parquet(&path, None)?;
            self.cache.insert(name.to_string(), table);
        }
        Ok(&self.cache[name])
    }

    /// A clone of a built table (for builders that hold several at once).
    pub fn owned(&mut self, name: &str) -> Result<Table> {
        Ok(self.table(name)?.clone())
    }

    pub fn remember(&mut self, name: &str, table: Table) {
        self.cache.insert(name.to_string(), table);
    }

    pub fn forget(&mut self, name: &str) {
        self.cache.remove(name);
    }

    pub fn root(&self) -> &Path {
        &self.store.root
    }
}

// --- raw CSV ---------------------------------------------------------------------------------

/// pandas' default `na_values`: cells read as missing.
const NA_VALUES: &[&str] = &[
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
];

/// Strict UTF-8 (BOM stripped), else latin-1.
pub fn decode_text(data: &[u8]) -> String {
    let data = data.strip_prefix(b"\xef\xbb\xbf").unwrap_or(data);
    match std::str::from_utf8(data) {
        Ok(text) => text.to_string(),
        Err(_) => data.iter().map(|&b| b as char).collect(),
    }
}

/// A raw CSV: stripped header names and the rows' cells as text (missing cells `None`).
pub struct Csv {
    pub header: Vec<String>,
    pub rows: Vec<Vec<Option<String>>>,
}

impl Csv {
    pub fn read(path: &Path) -> Result<Csv> {
        let text = decode_text(&RawStore::read_bytes(path)?);
        Csv::parse(&text).with_context(|| format!("read {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Csv> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(text.as_bytes());
        let mut records = reader.records();
        let header: Vec<String> = match records.next() {
            Some(record) => record?.iter().map(|h| h.trim().to_string()).collect(),
            None => Vec::new(),
        };
        let mut rows = Vec::new();
        for record in records {
            let record = record?;
            // pandas skips blank lines.
            if record.len() == 1 && record[0].trim().is_empty() {
                continue;
            }
            let mut row: Vec<Option<String>> = record
                .iter()
                .map(|cell| (!NA_VALUES.contains(&cell)).then(|| cell.to_string()))
                .collect();
            row.resize(header.len(), None);
            rows.push(row);
        }
        Ok(Csv { header, rows })
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.header.iter().position(|h| h == name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.index(name).is_some()
    }

    pub fn require(&self, name: &str) -> Result<usize> {
        self.index(name).with_context(|| format!("CSV has no column {name:?}"))
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The column's cells (missing = `None`); an absent column is all `None`.
    pub fn cells(&self, name: &str) -> Vec<Option<&str>> {
        match self.index(name) {
            Some(i) => self.rows.iter().map(|row| row[i].as_deref()).collect(),
            None => vec![None; self.rows.len()],
        }
    }

    /// Rows with a value in `name` (pandas `dropna(subset=[name])`).
    pub fn drop_na(mut self, name: &str) -> Csv {
        if let Some(i) = self.index(name) {
            self.rows.retain(|row| row[i].is_some());
        }
        self
    }
}

/// A cell as pandas reads a numeric column: int64 when every value is integral.
pub fn parse_int(cell: Option<&str>, what: &str) -> Result<i64> {
    let text = cell.with_context(|| format!("{what}: missing value"))?;
    parse_int_text(text).with_context(|| format!("{what}: not an integer: {text:?}"))
}

fn parse_int_text(text: &str) -> Option<i64> {
    let text = text.trim();
    if let Ok(v) = text.parse::<i64>() {
        return Some(v);
    }
    let f: f64 = text.parse().ok()?;
    (f.fract() == 0.0 && f.is_finite()).then_some(f as i64)
}

pub fn parse_opt_int(cell: Option<&str>, what: &str) -> Result<Option<i64>> {
    cell.map(|text| {
        parse_int_text(text).with_context(|| format!("{what}: not an integer: {text:?}"))
    })
    .transpose()
}

pub fn parse_opt_float(cell: Option<&str>, what: &str) -> Result<Option<f64>> {
    cell.map(|text| {
        text.trim().parse::<f64>().with_context(|| format!("{what}: not a number: {text:?}"))
    })
    .transpose()
}

/// pandas' boolean parsing ("True"/"TRUE"/"true", "False"/...).
pub fn parse_bool(cell: Option<&str>, what: &str) -> Result<bool> {
    match cell {
        Some("True" | "TRUE" | "true") => Ok(true),
        Some("False" | "FALSE" | "false") => Ok(false),
        Some(text) => {
            // A numeric column cast with astype(bool).
            match text.trim().parse::<f64>() {
                Ok(v) => Ok(v != 0.0),
                Err(_) => bail!("{what}: not a boolean: {text:?}"),
            }
        }
        None => bail!("{what}: missing value"),
    }
}

/// `astype(str)` of a pandas cell: missing -> "nan".
pub fn as_str(cell: Option<&str>) -> String {
    cell.unwrap_or("nan").to_string()
}

// --- JSON helpers ----------------------------------------------------------------------------

pub fn json_i64(value: &Value, what: &str) -> Result<i64> {
    match value {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
            .with_context(|| format!("{what}: not an integer: {n}")),
        Value::String(s) => {
            parse_int_text(s).with_context(|| format!("{what}: not an integer: {s:?}"))
        }
        Value::Bool(b) => Ok(i64::from(*b)),
        other => bail!("{what}: not an integer: {other}"),
    }
}

pub fn json_opt_i64(value: &Value, what: &str) -> Result<Option<i64>> {
    if value.is_null() { Ok(None) } else { json_i64(value, what).map(Some) }
}

pub fn json_f64(value: &Value, what: &str) -> Result<f64> {
    match value {
        Value::Number(n) => n.as_f64().with_context(|| format!("{what}: not a number")),
        Value::String(s) => {
            s.trim().parse().with_context(|| format!("{what}: not a number: {s:?}"))
        }
        other => bail!("{what}: not a number: {other}"),
    }
}

pub fn json_opt_f64(value: &Value, what: &str) -> Result<Option<f64>> {
    if value.is_null() { Ok(None) } else { json_f64(value, what).map(Some) }
}

/// Python truthiness of a JSON value, as `astype(bool)` sees it.
pub fn json_bool(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

pub fn json_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

// --- runs ------------------------------------------------------------------------------------

/// (run directory, manifest) of every timestamp-named complete run under
/// `<source>/<endpoint>/`, newest first: the manifest lists every expected file as written
/// and none failed.
pub fn complete_runs(
    store: &RawStore,
    source: &str,
    endpoint: &str,
) -> Result<Vec<(PathBuf, Value)>> {
    let directory = store.root.join(source).join(endpoint);
    let mut runs: Vec<(DateTime<Utc>, PathBuf)> = Vec::new();
    if directory.is_dir() {
        for entry in std::fs::read_dir(&directory)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            if path.is_dir() && TS_PATTERN.is_match(name) {
                runs.push((parse_ts(name)?, path));
            }
        }
    }
    runs.sort_by_key(|run| std::cmp::Reverse(run.0));
    let mut out = Vec::new();
    for (_, run) in runs {
        let manifest_path = run.join(MANIFEST);
        if !manifest_path.exists() {
            continue;
        }
        let manifest = RawStore::read_json(&manifest_path)?;
        let as_strings = |v: &Value| -> Option<Vec<String>> {
            let mut items: Vec<String> = v.as_array()?.iter().map(py_str).collect();
            items.sort();
            Some(items)
        };
        let failed_empty =
            manifest.get("failed").and_then(Value::as_array).is_some_and(Vec::is_empty);
        let expected = manifest.get("expected").and_then(as_strings);
        let written = manifest.get("written").and_then(as_strings);
        if failed_empty && expected.is_some() && expected == written {
            out.push((run, manifest));
        } else {
            log::warn!("skipping incomplete run {}", run.display());
        }
    }
    Ok(out)
}

/// Python's `str()` of a JSON value.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

pub fn latest_complete_run(store: &RawStore, source: &str, endpoint: &str) -> Result<PathBuf> {
    complete_runs(store, source, endpoint)?
        .into_iter()
        .next()
        .map(|(run, _)| run)
        .with_context(|| format!("no complete run under {source}/{endpoint}"))
}

/// Newest complete run whose manifest `season` is `season`; runs without one are skipped.
pub fn latest_complete_season_run(
    store: &RawStore,
    source: &str,
    endpoint: &str,
    season: i32,
) -> Result<Option<PathBuf>> {
    for (run, manifest) in complete_runs(store, source, endpoint)? {
        match manifest.get("season") {
            None | Some(Value::Null) => {
                log::warn!("run {} has no season in its manifest; skipped", run.display());
            }
            Some(value) => {
                if json_i64(value, "manifest season")? == i64::from(season) {
                    return Ok(Some(run));
                }
            }
        }
    }
    Ok(None)
}

/// A bootstrap snapshot: (time, path, source 'fplcache' | 'fpl').
#[derive(Clone, Debug)]
pub struct Bootstrap {
    pub taken_at: DateTime<Utc>,
    pub path: PathBuf,
    pub source: &'static str,
}

/// Every archived bootstrap-static snapshot (fplcache mirror and our own), by (time, source).
pub fn fplcache_and_own_bootstraps(store: &RawStore) -> Result<Vec<Bootstrap>> {
    let mut entries: Vec<Bootstrap> = store
        .entries("fplcache", "bootstrap-static", ".json.xz")?
        .into_iter()
        .map(|(taken_at, path)| Bootstrap { taken_at, path, source: "fplcache" })
        .collect();
    entries.extend(
        store
            .entries("fpl", "bootstrap-static", ".json.gz")?
            .into_iter()
            .map(|(taken_at, path)| Bootstrap { taken_at, path, source: "fpl" }),
    );
    entries.sort_by(|a, b| (a.taken_at, a.source).cmp(&(b.taken_at, b.source)));
    Ok(entries)
}

/// The newest `.csv.gz` (or other suffix) entry of a store directory.
pub fn latest_entry(
    store: &RawStore,
    source: &str,
    endpoint: &str,
    suffix: &str,
) -> Result<Option<PathBuf>> {
    Ok(store.entries(source, endpoint, suffix)?.pop().map(|(_, p)| p))
}

/// Sorted subdirectories of `dir` (empty if it is not a directory).
pub fn subdirs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

pub fn file_name(path: &Path) -> &str {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("")
}

// --- time ------------------------------------------------------------------------------------

pub fn micros(ts: &DateTime<Utc>) -> Micros {
    ts.timestamp_micros()
}

pub fn from_micros(us: Micros) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(us).expect("timestamp in range")
}

/// ISO-8601 text -> UTC microseconds (pandas `to_datetime(utc=True, format="ISO8601")`,
/// truncated to microseconds).
pub fn parse_iso(text: &str) -> Result<Micros> {
    let text = text.trim();
    if let Ok(ts) = DateTime::parse_from_rfc3339(text) {
        return Ok(ts.with_timezone(&Utc).timestamp_micros());
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f%:z", "%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%d %H:%M:%S%.f%:z"] {
        if let Ok(ts) = DateTime::parse_from_str(text, format) {
            return Ok(ts.with_timezone(&Utc).timestamp_micros());
        }
    }
    // Naive times count as UTC (utc=True).
    for format in
        ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"]
    {
        if let Ok(ts) = NaiveDateTime::parse_from_str(text, format) {
            return Ok(ts.and_utc().timestamp_micros());
        }
    }
    if let Some(stripped) = text.strip_suffix('Z')
        && let Ok(ts) = NaiveDateTime::parse_from_str(stripped, "%Y-%m-%dT%H:%M:%S%.f")
    {
        return Ok(ts.and_utc().timestamp_micros());
    }
    if let Ok(day) = NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        return Ok(day.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp_micros());
    }
    bail!("not an ISO-8601 time: {text:?}")
}

pub fn days_from_date(day: NaiveDate) -> Days {
    (day - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as Days
}

pub fn date_from_days(days: Days) -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1).unwrap() + chrono::Duration::days(i64::from(days))
}

/// The UK-local calendar date of a UTC time.
pub fn uk_date(us: Micros) -> NaiveDate {
    from_micros(us).with_timezone(&London).date_naive()
}

/// `day` at `hour`:00 UK local time, in UTC (never ambiguous: clocks change at 01:00-02:00).
pub fn uk_wall_clock(day: NaiveDate, hour: u32) -> Micros {
    let local = day.and_hms_opt(hour, 0, 0).unwrap();
    London
        .from_local_datetime(&local)
        .earliest()
        .expect("UK wall-clock time exists")
        .with_timezone(&Utc)
        .timestamp_micros()
}

/// GW lockdown: 09:00 UK on the day after the UK-local date of the GW's last kickoff.
pub fn lockdown_time(last_kickoff: Micros) -> Micros {
    uk_wall_clock(uk_date(last_kickoff).succ_opt().unwrap(), 9)
}

/// 1 June of a season's start year, 00:00 UTC (fixture list publication, PLAN §4).
pub fn schedule_published_at(season: i64) -> Micros {
    Utc.with_ymd_and_hms(season as i32, 6, 1, 0, 0, 0).unwrap().timestamp_micros()
}

/// Publication, or if later the lockdown after the previous season's last kickoff.
pub fn schedule_available_at(season: i64, previous_last_kickoff: Option<Micros>) -> Micros {
    let published = schedule_published_at(season);
    match previous_last_kickoff {
        Some(kickoff) => published.max(lockdown_time(kickoff)),
        None => published,
    }
}

/// The season in progress at a UTC time (July cutoff).
pub fn season_of(us: Micros) -> i64 {
    let ts = from_micros(us);
    i64::from(if ts.month() >= 7 { ts.year() } else { ts.year() - 1 })
}

/// The season a kickoff belongs to: seasons start in August and end by July (2019/20 ended
/// 2020-07-26).
pub fn season_of_aug(us: Micros) -> i64 {
    let ts = from_micros(us);
    i64::from(if ts.month() >= 8 { ts.year() } else { ts.year() - 1 })
}

pub static UNSCHEDULED_AT: LazyLock<Micros> =
    LazyLock::new(|| Utc.with_ymd_and_hms(2100, 1, 1, 0, 0, 0).unwrap().timestamp_micros());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lockdown_across_clock_change() {
        // Last kickoff Sat 2026-10-24 (BST); lockdown Sun 25 Oct 09:00 GMT = 09:00 UTC.
        let kickoff = parse_iso("2026-10-24T18:30:00Z").unwrap();
        assert_eq!(lockdown_time(kickoff), parse_iso("2026-10-25T09:00:00Z").unwrap());
        let kickoff = parse_iso("2026-08-22T23:30:00Z").unwrap(); // Sun 00:30 BST
        assert_eq!(lockdown_time(kickoff), parse_iso("2026-08-24T08:00:00Z").unwrap());
    }

    #[test]
    fn csv_na_values_and_blank_lines() {
        let csv = Csv::parse(" a ,b\n1,NA\n\n,x\n").unwrap();
        assert_eq!(csv.header, ["a", "b"]);
        assert_eq!(csv.rows, vec![vec![Some("1".into()), None], vec![None, Some("x".into())]]);
    }

    #[test]
    fn iso_fractions_truncate_to_micros() {
        assert_eq!(parse_iso("1970-01-01T00:00:01.0000019Z").unwrap(), 1_000_001);
    }
}
