//! Append-only store for raw source data: compressed files with a UTC timestamp in the
//! path. Port of `fplopt.ingest.raw_store`.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDateTime, Utc};
use flate2::Compression;
use regex::Regex;
use serde_json::Value;

use crate::Failure;
use crate::fetch::strip_bom;

pub const TS_FORMAT: &str = "%Y-%m-%dT%H%M%SZ";
pub const SUFFIX: &str = ".json.gz";
pub static TS_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{6}Z$").unwrap());

pub fn format_ts(ts: &DateTime<Utc>) -> String {
    ts.format(TS_FORMAT).to_string()
}

pub fn parse_ts(text: &str) -> Result<DateTime<Utc>> {
    Ok(NaiveDateTime::parse_from_str(text, TS_FORMAT)?.and_utc())
}

/// Deterministic gzip (no mtime, best compression), as used for every `.gz` file. The
/// compressed bytes may differ from Python's zlib output; the content is what matters.
pub fn gzip_bytes(content: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::GzBuilder::new()
        .mtime(0)
        .write(Vec::with_capacity(content.len() / 4), Compression::best());
    encoder.write_all(content).expect("writing to a Vec cannot fail");
    encoder.finish().expect("writing to a Vec cannot fail")
}

/// Refuse anything that could escape the store: absolute paths, '..', empty parts, '\\',
/// and ':' (a Windows drive prefix such as 'C:x' would replace the store root).
fn check_relative(text: &str, what: &str) -> Result<()> {
    let bad = text.is_empty()
        || text.starts_with('/')
        || text.contains('\\')
        || text.contains(':')
        || text.split('/').any(|part| matches!(part, "" | "." | ".."));
    if bad {
        return Err(Failure::error(
            "ValueError",
            format!("unsafe {what}: {}", crate::py_repr_str(text)),
        ));
    }
    Ok(())
}

/// Raw files are always compressed; the suffix says how (e.g. `.csv.gz`, `.json.xz`).
fn check_suffix(suffix: &str) -> Result<()> {
    let ok = suffix.len() >= 2
        && suffix.starts_with('.')
        && !suffix.contains('/')
        && !suffix.contains('\\')
        && (suffix.ends_with(".gz") || suffix.ends_with(".xz"));
    if !ok {
        return Err(Failure::error(
            "ValueError",
            format!(
                "suffix must be a compressed file suffix like .json.gz: {}",
                crate::py_repr_str(suffix)
            ),
        ));
    }
    Ok(())
}

/// Persist a new directory entry. POSIX only (Windows cannot open directories).
fn fsync_dir(directory: &Path) -> std::io::Result<()> {
    if cfg!(unix) { fs::File::open(directory)?.sync_all() } else { Ok(()) }
}

/// Writes each file exactly once; existing files are never overwritten.
///
/// Layout: single-file snapshots at `<source>/<endpoint>/<ts><suffix>`; grouped runs
/// written with a name at `<source>/<endpoint>/<ts>/<name><suffix>` (`endpoint` and `name`
/// may contain `/`). `write` takes JSON and gzips it; `write_bytes` stores
/// already-compressed bytes verbatim, with a suffix saying how they are compressed.
///
/// `entries` / `times` / `latest` list single-file snapshots with the given suffix only;
/// grouped runs are not listed, and files whose names are not timestamps are ignored.
#[derive(Debug, Clone)]
pub struct RawStore {
    pub root: PathBuf,
}

impl RawStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn directory(&self, source: &str, endpoint: &str) -> Result<PathBuf> {
        check_relative(source, "source")?;
        if source.contains('/') {
            return Err(Failure::error(
                "ValueError",
                format!("unsafe source: {}", crate::py_repr_str(source)),
            ));
        }
        check_relative(endpoint, "endpoint")?;
        Ok(self.root.join(source).join(endpoint))
    }

    /// Where a file fetched at `fetched_at` lives (whether or not it exists yet).
    pub fn path_for(
        &self,
        source: &str,
        endpoint: &str,
        fetched_at: &DateTime<Utc>,
        suffix: &str,
        name: Option<&str>,
    ) -> Result<PathBuf> {
        let directory = self.directory(source, endpoint)?;
        check_suffix(suffix)?;
        let stamp = format_ts(fetched_at);
        match name {
            None => Ok(directory.join(format!("{stamp}{suffix}"))),
            Some(name) => {
                check_relative(name, "name")?;
                Ok(directory.join(stamp).join(format!("{name}{suffix}")))
            }
        }
    }

    /// Store `data` verbatim; the caller compresses it to match `suffix`. Fails with
    /// `AlreadyExists` instead of overwriting.
    pub fn write_bytes(
        &self,
        source: &str,
        endpoint: &str,
        data: &[u8],
        fetched_at: &DateTime<Utc>,
        suffix: &str,
        name: Option<&str>,
    ) -> Result<PathBuf> {
        let path = self.path_for(source, endpoint, fetched_at, suffix, name)?;
        let parent = path.parent().context("store path has no parent")?;
        fs::create_dir_all(parent)?;
        let file_name = path.file_name().context("store path has no file name")?;
        let tmp =
            parent.join(format!(".{}.{}.tmp", file_name.to_string_lossy(), std::process::id()));
        let result = (|| -> std::io::Result<()> {
            let mut fh = fs::File::create(&tmp)?;
            fh.write_all(data)?;
            fh.flush()?;
            fh.sync_all()?;
            drop(fh);
            fs::hard_link(&tmp, &path)?; // atomic, and fails instead of overwriting
            fsync_dir(parent)
        })();
        let _ = fs::remove_file(&tmp);
        result?;
        Ok(path)
    }

    /// Gzip and store JSON; refuses anything that isn't (e.g. an HTML error page).
    pub fn write(
        &self,
        source: &str,
        endpoint: &str,
        content: &[u8],
        fetched_at: &DateTime<Utc>,
        name: Option<&str>,
    ) -> Result<PathBuf> {
        serde_json::from_slice::<serde::de::IgnoredAny>(strip_bom(content))?;
        self.write_bytes(source, endpoint, &gzip_bytes(content), fetched_at, SUFFIX, name)
    }

    /// (timestamp, path) of every single-file snapshot with `suffix`, oldest first.
    pub fn entries(
        &self,
        source: &str,
        endpoint: &str,
        suffix: &str,
    ) -> Result<Vec<(DateTime<Utc>, PathBuf)>> {
        let directory = self.directory(source, endpoint)?;
        check_suffix(suffix)?;
        if !directory.is_dir() {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        for entry in fs::read_dir(&directory)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            let Some(stem) = name.strip_suffix(suffix) else { continue };
            if TS_PATTERN.is_match(stem)
                && let Ok(ts) = parse_ts(stem)
            {
                entries.push((ts, path));
            }
        }
        entries.sort();
        Ok(entries)
    }

    pub fn json_entries(
        &self,
        source: &str,
        endpoint: &str,
    ) -> Result<Vec<(DateTime<Utc>, PathBuf)>> {
        self.entries(source, endpoint, SUFFIX)
    }

    pub fn times(&self, source: &str, endpoint: &str) -> Result<Vec<DateTime<Utc>>> {
        Ok(self.json_entries(source, endpoint)?.into_iter().map(|(ts, _)| ts).collect())
    }

    pub fn latest(&self, source: &str, endpoint: &str) -> Result<Option<PathBuf>> {
        Ok(self.json_entries(source, endpoint)?.pop().map(|(_, path)| path))
    }

    /// The decompressed content of a stored file (by its final suffix: `.gz` or `.xz`).
    pub fn read_bytes(path: &Path) -> Result<Vec<u8>> {
        let data = fs::read(path)?;
        decompress(path, &data)
    }

    pub fn read_json(path: &Path) -> Result<Value> {
        Ok(serde_json::from_slice(strip_bom(&Self::read_bytes(path)?))?)
    }
}

/// Decompress raw-store bytes by the path's final suffix.
pub fn decompress(path: &Path, data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    match path.extension().and_then(|e| e.to_str()) {
        Some("gz") => {
            flate2::read::MultiGzDecoder::new(data).read_to_end(&mut out)?;
        }
        Some("xz") => {
            lzma_rs::xz_decompress(&mut std::io::BufReader::new(data), &mut out)
                .map_err(|e| Failure::error("LZMAError", e.to_string()))?;
        }
        _ => {
            let name =
                path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            return Err(Failure::error(
                "ValueError",
                format!("not a compressed raw file: {}", crate::py_repr_str(&name)),
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(h: u32, m: u32, s: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 8, 21, h, m, s).unwrap()
    }

    #[test]
    fn writes_gzipped_json_at_timestamped_path() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let path =
            store.write("fpl", "bootstrap-static", b"{\"a\": 1}", &at(1, 2, 3), None).unwrap();
        assert_eq!(path, dir.path().join("fpl/bootstrap-static/2026-08-21T010203Z.json.gz"));
        assert_eq!(RawStore::read_bytes(&path).unwrap(), b"{\"a\": 1}");
        assert_eq!(RawStore::read_json(&path).unwrap()["a"], 1);
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0], "mtime must be zero");
    }

    #[test]
    fn never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        store.write("fpl", "fixtures", b"[]", &at(1, 0, 0), None).unwrap();
        let err = store.write("fpl", "fixtures", b"[1]", &at(1, 0, 0), None).unwrap_err();
        assert_eq!(crate::kind_name(&err), "FileExistsError");
        let path = store.latest("fpl", "fixtures").unwrap().unwrap();
        assert_eq!(RawStore::read_bytes(&path).unwrap(), b"[]");
        let leftovers: Vec<_> = fs::read_dir(path.parent().unwrap()).unwrap().collect();
        assert_eq!(leftovers.len(), 1, "no temp files left behind");
    }

    #[test]
    fn refuses_non_json() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let err = store.write("fpl", "fixtures", b"<html>", &at(1, 0, 0), None).unwrap_err();
        assert_eq!(crate::kind_name(&err), "JSONDecodeError");
        assert!(store.latest("fpl", "fixtures").unwrap().is_none());
    }

    #[test]
    fn grouped_runs_are_not_listed() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let p = store.write("fpl", "element-summary", b"{}", &at(1, 0, 0), Some("7")).unwrap();
        assert!(p.ends_with("fpl/element-summary/2026-08-21T010000Z/7.json.gz"));
        assert!(store.times("fpl", "element-summary").unwrap().is_empty());
    }

    #[test]
    fn entries_sorted_and_filtered() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        store.write("odds", "soccer_epl", b"[]", &at(3, 0, 0), None).unwrap();
        store.write("odds", "soccer_epl", b"[]", &at(1, 0, 0), None).unwrap();
        fs::write(dir.path().join("odds/soccer_epl/notes.json.gz"), b"x").unwrap();
        assert_eq!(store.times("odds", "soccer_epl").unwrap(), vec![at(1, 0, 0), at(3, 0, 0)]);
    }

    #[test]
    fn rejects_unsafe_paths_and_suffixes() {
        let store = RawStore::new("r");
        let ts = at(1, 0, 0);
        for (source, endpoint, name) in [
            ("../x", "e", None),
            ("a/b", "e", None),
            ("fpl", "/abs", None),
            ("fpl", "a/../b", None),
            ("fpl", "a//b", None),
            ("fpl", "C:x", None),
            ("fpl", "a\\b", None),
            ("fpl", "e", Some("..")),
            ("fpl", "e", Some("")),
        ] {
            assert!(
                store.path_for(source, endpoint, &ts, SUFFIX, name).is_err(),
                "{source} {endpoint} {name:?}"
            );
        }
        for suffix in [".json", "json.gz", ".", "/x.gz", ".a\\b.gz"] {
            assert!(store.path_for("fpl", "e", &ts, suffix, None).is_err(), "{suffix}");
        }
        assert!(store.path_for("fpl", "e", &ts, ".csv.gz", None).is_ok());
    }

    #[test]
    fn reads_xz() {
        let dir = tempfile::tempdir().unwrap();
        let mut compressed = Vec::new();
        lzma_rs::xz_compress(&mut &b"{\"x\": 2}"[..], &mut compressed).unwrap();
        let store = RawStore::new(dir.path());
        let p = store
            .write_bytes(
                "fplcache",
                "bootstrap-static",
                &compressed,
                &at(1, 0, 0),
                ".json.xz",
                None,
            )
            .unwrap();
        assert_eq!(RawStore::read_json(&p).unwrap()["x"], 2);
    }
}
