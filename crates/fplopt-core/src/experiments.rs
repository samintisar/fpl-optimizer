//! The experiment log, `results/experiments.csv` (PLAN §5): one row per run. Port of
//! `log_experiment` / `read_experiments` / `family_variants` in `fplopt.backtest.evaluate`,
//! writing the same columns and JSON layout.

use std::fs::{self, OpenOptions};
use std::io;
use std::path::Path;

use anyhow::Result;
use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use serde_json::Value;
use serde_json::ser::Formatter;

pub const EXPERIMENT_COLUMNS: [&str; 7] =
    ["timestamp", "git_sha", "command", "config", "metrics", "n_variants", "family"];

/// The commit this binary was built from (see build.rs), or "unknown".
pub const GIT_SHA: &str = env!("FPLOPT_GIT_SHA");

/// The log's rows as column -> value maps (empty if the file is missing or empty); columns
/// missing from an older log read as "".
pub fn read_experiments(path: &Path) -> Result<Vec<Vec<(String, String)>>> {
    if !path.exists() || fs::metadata(path)?.len() == 0 {
        return Ok(Vec::new());
    }
    let mut reader = csv::Reader::from_path(path)?;
    let header: Vec<String> = reader.headers()?.iter().map(str::to_owned).collect();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record?;
        let row = EXPERIMENT_COLUMNS
            .iter()
            .map(|&column| {
                let value = header
                    .iter()
                    .position(|h| h == column)
                    .and_then(|i| record.get(i))
                    .unwrap_or("");
                (column.to_owned(), value.to_owned())
            })
            .collect();
        rows.push(row);
    }
    Ok(rows)
}

/// How many runs of comparison family `family` the log already holds.
pub fn family_variants(path: &Path, family: &str) -> Result<usize> {
    let rows = read_experiments(path)?;
    Ok(rows.iter().filter(|row| row.iter().any(|(k, v)| k == "family" && v == family)).count())
}

/// Append one row: UTC timestamp, git sha, command, config and metrics as JSON (sorted
/// keys, Python's `json.dumps` separators; non-finite floats are already null in a
/// `serde_json::Value`), n_variants, family. LF line endings; the header is written if the
/// file is new or empty, and a log with an older header is rewritten with these columns.
pub fn log_experiment(
    path: &Path,
    command: &str,
    config: &Value,
    metrics: &Value,
    n_variants: usize,
    family: &str,
) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let new = !path.exists() || fs::metadata(path)?.len() == 0;
    if !new {
        let mut reader = csv::Reader::from_path(path)?;
        let header: Vec<String> = reader.headers()?.iter().map(str::to_owned).collect();
        if header != EXPERIMENT_COLUMNS {
            let old = read_experiments(path)?;
            let mut writer = writer(fs::File::create(path)?);
            writer.write_record(EXPERIMENT_COLUMNS)?;
            for row in old {
                writer.write_record(row.iter().map(|(_, v)| v))?;
            }
            writer.flush()?;
        }
    }
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = writer(file);
    if new {
        writer.write_record(EXPERIMENT_COLUMNS)?;
    }
    writer.write_record([
        Utc::now().to_rfc3339_opts(SecondsFormat::Secs, false).as_str(),
        GIT_SHA,
        command,
        &dumps(config)?,
        &dumps(metrics)?,
        &n_variants.to_string(),
        family,
    ])?;
    writer.flush()?;
    Ok(())
}

fn writer<W: io::Write>(inner: W) -> csv::Writer<W> {
    csv::WriterBuilder::new().terminator(csv::Terminator::Any(b'\n')).from_writer(inner)
}

/// `json.dumps(value, sort_keys=True)`: ", " and ": " separators. Keys come out sorted
/// because `serde_json::Map` is a BTreeMap (no `preserve_order` feature).
pub fn dumps(value: &Value) -> Result<String> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonFormatter);
    value.serialize(&mut ser)?;
    Ok(String::from_utf8(out)?)
}

struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_matches_python_layout() {
        let value = json!({"b": [1, 2.5, null], "a": {"y": "x", "x": true}, "nan": f64::NAN});
        assert_eq!(
            dumps(&value).unwrap(),
            r#"{"a": {"x": true, "y": "x"}, "b": [1, 2.5, null], "nan": null}"#
        );
    }

    #[test]
    fn appends_rows_with_header_and_counts_family() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results").join("experiments.csv");
        log_experiment(&path, "fplopt-rs x", &json!({"k": 1}), &json!({}), 1, "fam").unwrap();
        log_experiment(&path, "fplopt-rs y", &json!({}), &json!({"m": 2}), 2, "fam").unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains('\r'));
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], EXPERIMENT_COLUMNS.join(","));
        assert!(lines[1].contains(r#",fplopt-rs x,"{""k"": 1}",{},1,fam"#), "{}", lines[1]);
        assert_eq!(family_variants(&path, "fam").unwrap(), 2);
        assert_eq!(family_variants(&path, "other").unwrap(), 0);
    }

    #[test]
    fn rewrites_an_older_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.csv");
        fs::write(&path, "timestamp,git_sha,command,config,metrics,n_variants\nt,s,c,{},{},1\n")
            .unwrap();
        log_experiment(&path, "c2", &json!({}), &json!({}), 1, "f").unwrap();
        let rows = read_experiments(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][6], ("family".to_owned(), String::new()));
        assert_eq!(rows[1][6], ("family".to_owned(), "f".to_owned()));
    }
}
