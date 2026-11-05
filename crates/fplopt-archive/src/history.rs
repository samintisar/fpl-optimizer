//! One-off historical backfills into raw/: football-data seasons, vaastav, fplcache. Port
//! of `fplopt.ingest.history`.

use std::cell::RefCell;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, TimeZone, Utc};
use fplopt_core::pyjson;
use fplopt_core::seasons::{football_data_code, season_label, season_start_year};
use regex::Regex;
use serde::Serialize;

use crate::adapters::fplcache::{Member, for_each_member};
use crate::adapters::{FootballDataSource, FplcacheSource, VaastavSource};
use crate::jobs::{
    run_independently, snapshot_football_data, snapshot_football_data_current, step,
};
use crate::pytime::isoformat;
use crate::raw_store::{RawStore, gzip_bytes};
use crate::{Clock, Failure, kind_name, one_line, py_repr_str, py_repr_strs, stop};

pub const FIRST_SEASON: i32 = 2016;
pub const FOOTBALL_DATA_PAUSE: Duration = Duration::from_secs(1);
pub const VAASTAV_PAUSE: Duration = Duration::from_millis(50);

/// Every EPL season CSV from `first_season` (default 2016/17) to the current season. All
/// seasons are attempted; failures are raised together (SnapshotError, one line). The
/// current season follows the daily job's rule: a 404 in July/August means "not published
/// yet". Returns the number of seasons written.
pub fn backfill_football_data(
    store: &RawStore,
    fd: &dyn FootballDataSource,
    clock: &dyn Clock,
    pause: Duration,
    first_season: i32,
) -> Result<usize> {
    let current = season_start_year(&clock.now());
    let written = RefCell::new(Vec::new());
    let steps = (first_season..=current)
        .map(|start_year| {
            let written = &written;
            step(format!("E0/{}", football_data_code(start_year)), move || {
                if start_year > first_season {
                    clock.sleep(pause);
                }
                stop::check()?;
                let path = if start_year == current {
                    snapshot_football_data_current(store, fd, clock)?
                } else {
                    Some(snapshot_football_data(store, fd, start_year, clock)?)
                };
                if path.is_some() {
                    written.borrow_mut().push(start_year);
                }
                Ok(())
            })
        })
        .collect();
    run_independently(steps)?;
    let n = written.borrow().len();
    log::info!("archived {n} football-data seasons");
    Ok(n)
}

// --- vaastav -------------------------------------------------------------------------------

/// 2016-17 ... 2025-26; 2026/27 comes from our own archive.
pub const VAASTAV_SEASONS: std::ops::Range<i32> = 2016..2026;
static VAASTAV_FILE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^data/(?P<season>\d{4}-\d{2})/(?:players_raw|player_idlist|cleaned_players|fixtures|teams|id_dict|gws/merged_gw|understat/[^/]+)\.csv$",
    )
    .unwrap()
});
pub const VAASTAV_ROOT_FILES: [&str; 1] = ["data/master_team_list.csv"];

/// The vaastav files we mirror, sorted. Excluded on purpose: `gws/xP*.csv` (lookahead
/// leak), `gws/gwN.csv` and `players/` (duplicated by merged_gw), `fbref/`, `managers/`,
/// `cleaned_merged_seasons*.csv` (incomplete) and the stale 2026-27 folder.
pub fn select_vaastav_paths<'a>(paths: impl IntoIterator<Item = &'a String>) -> Vec<String> {
    let seasons: Vec<String> = VAASTAV_SEASONS.map(season_label).collect();
    let mut selected: Vec<String> = paths
        .into_iter()
        .filter(|path| {
            VAASTAV_ROOT_FILES.contains(&path.as_str())
                || VAASTAV_FILE
                    .captures(path)
                    .is_some_and(|m| seasons.iter().any(|s| s == &m["season"]))
        })
        .cloned()
        .collect();
    selected.sort();
    selected
}

/// Store name of a vaastav file: 'data/2016-17/gws/merged_gw.csv' -> '2016-17/gws/merged_gw'.
pub fn vaastav_name(path: &str) -> Result<&str> {
    path.strip_prefix("data/").and_then(|p| p.strip_suffix(".csv")).ok_or_else(|| {
        Failure::error("ValueError", format!("not a vaastav data CSV: {}", py_repr_str(path)))
    })
}

#[derive(Serialize)]
struct VaastavManifest<'a> {
    commit: &'a str,
    run_at: String,
    finished_at: String,
    expected: &'a [String],
    written: &'a [String],
    failed: &'a [String],
}

/// Mirror the selected vaastav files at `commit` under one run directory:
/// raw/vaastav/data/<run_at>/<path without 'data/' and '.csv'>.csv.gz, plus
/// _manifest.json.gz (commit, run_at, finished_at, expected, written, failed) written even
/// when aborted. Continues past single failures (including paths the store refuses, which
/// are not downloaded); aborts after `max_consecutive_failures` in a row; fails if anything
/// failed. Returns the number of files written.
pub fn backfill_vaastav(
    store: &RawStore,
    vaastav: &dyn VaastavSource,
    commit: &str,
    clock: &dyn Clock,
    pause: Duration,
    max_consecutive_failures: usize,
) -> Result<usize> {
    let blobs = vaastav.tree(commit)?;
    let paths = select_vaastav_paths(blobs.keys());
    if paths.is_empty() {
        return Err(Failure::error(
            "RuntimeError",
            format!("no vaastav files selected at {commit} (layout changed?)"),
        ));
    }
    let run_at = clock.now();
    let mut written = Vec::new();
    let mut failed = Vec::new();
    let outcome = (|| -> Result<()> {
        let mut consecutive = 0;
        for path in &paths {
            stop::check()?;
            let result = (|| -> Result<()> {
                let name = vaastav_name(path)?;
                // Validate the store path before downloading (e.g. ':' is refused).
                store.path_for("vaastav", "data", &run_at, ".csv.gz", Some(name))?;
                let content = vaastav.file(commit, path, &blobs[path])?;
                store.write_bytes(
                    "vaastav",
                    "data",
                    &gzip_bytes(&content),
                    &run_at,
                    ".csv.gz",
                    Some(name),
                )?;
                Ok(())
            })();
            match result {
                Ok(()) => {
                    written.push(path.clone());
                    consecutive = 0;
                }
                Err(err) => {
                    if err.is::<stop::Terminated>() {
                        return Err(err);
                    }
                    log::error!("vaastav {path} failed: {err:#}");
                    failed.push(path.clone());
                    consecutive += 1;
                    if consecutive >= max_consecutive_failures {
                        return Err(Failure::error(
                            "RuntimeError",
                            format!(
                                "aborting vaastav backfill after {consecutive} consecutive failures (last {path}: {})",
                                kind_name(&err)
                            ),
                        ));
                    }
                }
            }
            clock.sleep(pause);
        }
        Ok(())
    })();
    let manifest = VaastavManifest {
        commit,
        run_at: isoformat(&run_at),
        finished_at: isoformat(&clock.now()),
        expected: &paths,
        written: &written,
        failed: &failed,
    };
    let saved = pyjson::dumps(&manifest).and_then(|text| {
        store.write("vaastav", "data", text.as_bytes(), &run_at, Some("_manifest"))
    });
    if let Err(err) = saved {
        log::error!("could not write vaastav manifest: {err:#}");
    }
    outcome?;
    if !failed.is_empty() {
        return Err(Failure::error(
            "RuntimeError",
            format!(
                "{} of {} vaastav files failed (first: {})",
                failed.len(),
                paths.len(),
                py_repr_strs(&failed[..failed.len().min(5)])
            ),
        ));
    }
    log::info!("archived {} vaastav files at {commit}", paths.len());
    Ok(written.len())
}

// --- fplcache ------------------------------------------------------------------------------

static FPLCACHE_MEMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[^/]+/cache/(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<H>\d{2})(?P<M>\d{2})\.json\.xz$")
        .unwrap()
});
/// Real snapshots are ~0.1-0.3 MB compressed...
pub const MAX_FPLCACHE_MEMBER_BYTES: u64 = 64 << 20;
/// ...and ~1-2 MB decompressed.
pub const MAX_FPLCACHE_JSON_BYTES: usize = 256 << 20;
pub const FPLCACHE_PROGRESS_EVERY: usize = 500;

/// 'fplcache-<sha>/cache/2021/4/18/1641.json.xz' -> 2021-04-18 16:41 UTC; None for any other
/// file. A snapshot-shaped name with an impossible date is an error (`ValueError`).
pub fn fplcache_snapshot_time(member_name: &str) -> Result<Option<DateTime<Utc>>> {
    let Some(m) = FPLCACHE_MEMBER.captures(member_name) else { return Ok(None) };
    let num = |key: &str| m[key].parse::<u32>().expect("digits");
    let (y, mo, d, hh, mm) = (num("y") as i32, num("m"), num("d"), num("H"), num("M"));
    match Utc.with_ymd_and_hms(y, mo, d, hh, mm, 0).single() {
        Some(ts) => Ok(Some(ts)),
        None => Err(Failure::error(
            "ValueError",
            format!("impossible snapshot time {y}-{mo}-{d} {hh}:{mm}"),
        )),
    }
}

/// A writer that fails past `limit` bytes (a decompression bomb would otherwise exhaust
/// memory before the manifest is written).
struct Bounded {
    out: Vec<u8>,
    limit: usize,
}

impl std::io::Write for Bounded {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.out.len() + buf.len() > self.limit {
            return Err(std::io::Error::other("limit"));
        }
        self.out.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Decompress one complete .xz stream, refusing output over `limit` bytes and anything
/// after the stream.
pub fn unxz_bounded(data: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut input = data;
    let mut out = Bounded { out: Vec::new(), limit };
    if let Err(err) = lzma_rs::xz_decompress(&mut input, &mut out) {
        let too_big = out.out.len() + 1 > limit
            || matches!(&err, lzma_rs::error::Error::IoError(e) if e.to_string() == "limit");
        return Err(if too_big {
            Failure::error("ValueError", format!("decompresses to more than {limit} bytes"))
        } else if err.to_string().contains("Unexpected data after last XZ block") {
            // lzma-rs reads past the stream itself and refuses what follows
            Failure::error("ValueError", "unexpected data after the xz stream")
        } else {
            Failure::error("ValueError", format!("truncated xz data ({err})"))
        });
    }
    if !input.is_empty() {
        return Err(Failure::error("ValueError", "unexpected data after the xz stream"));
    }
    Ok(out.out)
}

/// Archive one snapshot member. True if written, false if an identical copy exists; fails
/// if it cannot be archived.
fn archive_fplcache_member(
    store: &RawStore,
    member: Member<'_>,
    snapshot_at: &DateTime<Utc>,
) -> Result<bool> {
    if member.size > MAX_FPLCACHE_MEMBER_BYTES {
        return Err(Failure::error(
            "ValueError",
            format!("member too large: {} bytes", member.size),
        ));
    }
    let mut data = Vec::with_capacity(member.size as usize);
    member.content.read_to_end(&mut data)?;
    let path = store.path_for("fplcache", "bootstrap-static", snapshot_at, ".json.xz", None)?;
    if path.exists() {
        if std::fs::read(&path)? == data {
            return Ok(false);
        }
        return Err(Failure::error(
            "ValueError",
            "differs from the archived copy (upstream rewrote history?)",
        ));
    }
    let json = unxz_bounded(&data, MAX_FPLCACHE_JSON_BYTES)?;
    serde_json::from_slice::<serde::de::IgnoredAny>(crate::fetch::strip_bom(&json))?;
    store.write_bytes("fplcache", "bootstrap-static", &data, snapshot_at, ".json.xz", None)?;
    Ok(true)
}

#[derive(Serialize)]
struct FailedMember {
    member: String,
    reason: String,
}

#[derive(Serialize)]
struct FplcacheManifest<'a> {
    commit: &'a str,
    run_at: String,
    finished_at: String,
    written: usize,
    skipped: usize,
    failed: &'a [FailedMember],
    error: Option<String>,
}

/// Mirror every fplcache snapshot into raw/fplcache/bootstrap-static/<snapshot time>.json.xz,
/// byte for byte. Idempotent: identical files already present are skipped; a present file
/// with different bytes is a failure (upstream rewrote history). Each new file must
/// decompress to JSON. Writes a run manifest raw/fplcache/runs/<run_at>.json.gz with commit,
/// written, skipped, failed (member names + reason) and error (a download/tar failure that
/// stopped the run, else null) - also when failures occurred - then fails if anything
/// failed. Returns the number of files written.
pub fn backfill_fplcache(
    store: &RawStore,
    fplcache: &dyn FplcacheSource,
    clock: &dyn Clock,
) -> Result<usize> {
    let commit = fplcache.head_commit()?;
    let run_at = clock.now();
    let (mut written, mut skipped) = (0usize, 0usize);
    let mut failed: Vec<FailedMember> = Vec::new();
    let outcome = fplcache.tarball(&commit).and_then(|stream| {
        for_each_member(stream, |member| {
            stop::check()?;
            if !member.is_file {
                return Ok(());
            }
            let name = member.name.clone();
            let result = fplcache_snapshot_time(&name).and_then(|snapshot_at| match snapshot_at {
                None => Ok(None),
                Some(at) => archive_fplcache_member(store, member, &at).map(Some),
            });
            match result {
                Ok(None) => return Ok(()),
                Ok(Some(true)) => written += 1,
                Ok(Some(false)) => skipped += 1,
                Err(err) => {
                    log::error!("fplcache {name} failed: {err:#}");
                    failed.push(FailedMember { member: name, reason: one_line(&err) });
                }
            }
            let done = written + skipped + failed.len();
            if done.is_multiple_of(FPLCACHE_PROGRESS_EVERY) {
                log::info!(
                    "fplcache: {done} snapshots ({written} written, {skipped} skipped, {} failed)",
                    failed.len()
                );
            }
            Ok(())
        })
    });
    let manifest = FplcacheManifest {
        commit: &commit,
        run_at: isoformat(&run_at),
        finished_at: isoformat(&clock.now()),
        written,
        skipped,
        failed: &failed,
        error: outcome.as_ref().err().map(one_line),
    };
    let saved = pyjson::dumps(&manifest)
        .and_then(|text| store.write("fplcache", "runs", text.as_bytes(), &run_at, None));
    if let Err(err) = saved {
        log::error!("could not write fplcache run manifest: {err:#}");
    }
    outcome?;
    let total = written + skipped + failed.len();
    if !failed.is_empty() {
        let first: Vec<&str> = failed.iter().take(5).map(|f| f.member.as_str()).collect();
        return Err(Failure::error(
            "RuntimeError",
            format!(
                "{} of {total} fplcache snapshots failed (first: {})",
                failed.len(),
                py_repr_strs(&first)
            ),
        ));
    }
    log::info!("fplcache at {commit}: {written} written, {skipped} already archived");
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vaastav_selection() {
        let paths: Vec<String> = [
            "data/2016-17/players_raw.csv",
            "data/2016-17/gws/merged_gw.csv",
            "data/2016-17/gws/gw1.csv",
            "data/2016-17/gws/xP1.csv",
            "data/2016-17/understat/understat_team.csv",
            "data/2016-17/understat/x/y.csv",
            "data/2026-27/players_raw.csv",
            "data/2015-16/players_raw.csv",
            "data/master_team_list.csv",
            "data/cleaned_merged_seasons.csv",
        ]
        .map(String::from)
        .into();
        assert_eq!(
            select_vaastav_paths(&paths),
            [
                "data/2016-17/gws/merged_gw.csv",
                "data/2016-17/players_raw.csv",
                "data/2016-17/understat/understat_team.csv",
                "data/master_team_list.csv",
            ]
        );
        assert_eq!(
            vaastav_name("data/2016-17/gws/merged_gw.csv").unwrap(),
            "2016-17/gws/merged_gw"
        );
        assert!(vaastav_name("x/2016-17.csv").is_err());
    }

    #[test]
    fn fplcache_times() {
        let at = fplcache_snapshot_time("fplcache-abc/cache/2021/4/18/1641.json.xz").unwrap();
        assert_eq!(at, Some(Utc.with_ymd_and_hms(2021, 4, 18, 16, 41, 0).unwrap()));
        assert_eq!(fplcache_snapshot_time("fplcache-abc/README.md").unwrap(), None);
        assert!(fplcache_snapshot_time("fplcache-abc/cache/2021/13/18/1641.json.xz").is_err());
    }

    #[test]
    fn xz_limits() {
        let mut compressed = Vec::new();
        lzma_rs::xz_compress(&mut &[b'x'; 1000][..], &mut compressed).unwrap();
        assert_eq!(unxz_bounded(&compressed, 1000).unwrap().len(), 1000);
        assert!(unxz_bounded(&compressed, 999).unwrap_err().to_string().contains("more than 999"));
        assert!(unxz_bounded(&compressed[..compressed.len() - 4], 5000).is_err());
        let mut trailing = compressed.clone();
        trailing.extend_from_slice(b"zz");
        assert!(
            unxz_bounded(&trailing, 5000).unwrap_err().to_string().contains("after the xz stream")
        );
    }
}
