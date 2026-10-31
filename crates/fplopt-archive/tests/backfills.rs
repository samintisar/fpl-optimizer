//! Historical backfills and the ESPN archive against fake sources (tests/test_history.py,
//! tests/test_ingest_espn.py).

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Cursor, Read, Write};
use std::time::Duration;

use anyhow::{Result, bail};
use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use fplopt_archive::adapters::{EspnSource, FootballDataSource, FplcacheSource, VaastavSource};
use fplopt_archive::raw_store::RawStore;
use fplopt_archive::{Clock, espn, history};
use serde_json::{Value, json};

struct FakeClock(Cell<DateTime<Utc>>);

impl Clock for FakeClock {
    fn now(&self) -> DateTime<Utc> {
        self.0.set(self.0.get() + TimeDelta::seconds(1));
        self.0.get()
    }

    fn sleep(&self, _: Duration) {}
}

fn clock(y: i32, m: u32, d: u32) -> FakeClock {
    FakeClock(Cell::new(Utc.with_ymd_and_hms(y, m, d, 2, 30, 0).unwrap()))
}

fn new_store() -> (tempfile::TempDir, RawStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = RawStore::new(dir.path());
    (dir, store)
}

// --- football-data -------------------------------------------------------------------------

struct Fd {
    fail: BTreeSet<i32>,
    calls: RefCell<Vec<i32>>,
}

impl FootballDataSource for Fd {
    fn epl_season(&self, y: i32) -> Result<Vec<u8>> {
        self.calls.borrow_mut().push(y);
        if self.fail.contains(&y) {
            bail!("down {y}");
        }
        Ok(format!("Div,Date\nE0,{y}\n").into_bytes())
    }
}

#[test]
fn football_data_backfill_attempts_every_season() {
    let (_d, store) = new_store();
    let fd = Fd { fail: [2018].into(), calls: RefCell::new(vec![]) };
    let err =
        history::backfill_football_data(&store, &fd, &clock(2026, 10, 1), Duration::ZERO, 2016)
            .unwrap_err();
    assert_eq!(err.to_string(), "E0/1819: Error: down 2018");
    assert_eq!(fd.calls.borrow().as_slice(), (2016..=2026).collect::<Vec<_>>());
    assert_eq!(store.entries("football-data", "E0/1617", ".csv.gz").unwrap().len(), 1);
    assert!(store.entries("football-data", "E0/1819", ".csv.gz").unwrap().is_empty());
}

// --- vaastav -------------------------------------------------------------------------------

struct Vaastav {
    files: BTreeMap<String, Vec<u8>>,
    fail: BTreeSet<String>,
}

impl VaastavSource for Vaastav {
    fn tree(&self, _commit: &str) -> Result<BTreeMap<String, String>> {
        Ok(self
            .files
            .iter()
            .map(|(p, c)| (p.clone(), fplopt_archive::adapters::vaastav::git_blob_sha(c)))
            .collect())
    }

    fn file(&self, _commit: &str, path: &str, _sha: &str) -> Result<Vec<u8>> {
        if self.fail.contains(path) {
            bail!("no {path}");
        }
        Ok(self.files[path].clone())
    }
}

fn vaastav(paths: &[&str]) -> Vaastav {
    Vaastav {
        files: paths.iter().map(|p| (p.to_string(), format!("csv of {p}").into_bytes())).collect(),
        fail: BTreeSet::new(),
    }
}

#[test]
fn vaastav_mirrors_selected_files_with_a_manifest() {
    let (dir, store) = new_store();
    let source = vaastav(&[
        "data/2016-17/players_raw.csv",
        "data/2016-17/gws/gw1.csv",
        "data/master_team_list.csv",
    ]);
    let c = clock(2026, 10, 1);
    assert_eq!(
        history::backfill_vaastav(&store, &source, "abc", &c, Duration::ZERO, 10).unwrap(),
        2
    );
    let run =
        std::fs::read_dir(dir.path().join("vaastav/data")).unwrap().next().unwrap().unwrap().path();
    let got = RawStore::read_bytes(&run.join("2016-17/players_raw.csv.gz")).unwrap();
    assert_eq!(got, b"csv of data/2016-17/players_raw.csv");
    let manifest = RawStore::read_json(&run.join("_manifest.json.gz")).unwrap();
    assert_eq!(manifest["commit"], "abc");
    assert_eq!(
        manifest["expected"],
        json!(["data/2016-17/players_raw.csv", "data/master_team_list.csv"])
    );
    assert_eq!(manifest["failed"], json!([]));
    let text =
        String::from_utf8(RawStore::read_bytes(&run.join("_manifest.json.gz")).unwrap()).unwrap();
    assert!(
        text.starts_with(
            r#"{"commit": "abc", "run_at": "2026-10-01T02:30:01+00:00", "finished_at": "#
        ),
        "{text}"
    );
}

#[test]
fn vaastav_failures_are_collected() {
    let (_d, store) = new_store();
    let mut source = vaastav(&["data/2016-17/players_raw.csv", "data/2016-17/teams.csv"]);
    source.fail.insert("data/2016-17/teams.csv".into());
    let err =
        history::backfill_vaastav(&store, &source, "abc", &clock(2026, 10, 1), Duration::ZERO, 10)
            .unwrap_err();
    assert_eq!(err.to_string(), "1 of 2 vaastav files failed (first: ['data/2016-17/teams.csv'])");
    let err = history::backfill_vaastav(
        &store,
        &vaastav(&["README.md"]),
        "abc",
        &clock(2026, 10, 2),
        Duration::ZERO,
        10,
    )
    .unwrap_err();
    assert!(err.to_string().starts_with("no vaastav files selected"));
}

// --- fplcache ------------------------------------------------------------------------------

fn xz(json: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    lzma_rs::xz_compress(&mut &json[..], &mut out).unwrap();
    out
}

fn tar_gz(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (name, content) in files {
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, name, &content[..]).unwrap();
    }
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    gz.write_all(&builder.into_inner().unwrap()).unwrap();
    gz.finish().unwrap()
}

struct Fplcache(Vec<u8>);

impl FplcacheSource for Fplcache {
    fn head_commit(&self) -> Result<String> {
        Ok("c".repeat(40))
    }

    fn tarball(&self, _commit: &str) -> Result<Box<dyn Read>> {
        Ok(Box::new(Cursor::new(self.0.clone())))
    }
}

fn runs(store: &RawStore) -> Vec<Value> {
    store.times("fplcache", "runs").unwrap();
    store
        .json_entries("fplcache", "runs")
        .unwrap()
        .iter()
        .map(|(_, p)| RawStore::read_json(p).unwrap())
        .collect()
}

#[test]
fn fplcache_mirrors_snapshots_verbatim_and_is_idempotent() {
    let (dir, store) = new_store();
    let good = xz(br#"{"events": []}"#);
    let tarball = tar_gz(&[
        ("fplcache-x/README.md", b"hi".to_vec()),
        ("fplcache-x/cache/2021/4/18/1641.json.xz", good.clone()),
        ("fplcache-x/cache/2021/4/18/2241.json.xz", xz(b"not json")),
        ("fplcache-x/cache/2021/13/18/0041.json.xz", good.clone()),
    ]);
    let err = history::backfill_fplcache(&store, &Fplcache(tarball.clone()), &clock(2026, 10, 1))
        .unwrap_err();
    assert!(err.to_string().starts_with(
        "2 of 3 fplcache snapshots failed (first: ['fplcache-x/cache/2021/4/18/2241.json.xz', "
    ));
    let path = dir.path().join("fplcache/bootstrap-static/2021-04-18T164100Z.json.xz");
    assert_eq!(std::fs::read(&path).unwrap(), good);
    let manifest = &runs(&store)[0];
    assert_eq!((manifest["written"].clone(), manifest["skipped"].clone()), (json!(1), json!(0)));
    assert_eq!(
        manifest["failed"][0]["reason"].as_str().unwrap().split(':').next(),
        Some("JSONDecodeError")
    );
    assert_eq!(manifest["error"], Value::Null);

    // Again: the good one is skipped, a rewritten one is a failure.
    let rewritten =
        tar_gz(&[("fplcache-x/cache/2021/4/18/1641.json.xz", xz(br#"{"events": [1]}"#))]);
    history::backfill_fplcache(
        &store,
        &Fplcache(tar_gz(&[("fplcache-x/cache/2021/4/18/1641.json.xz", good)])),
        &clock(2026, 10, 2),
    )
    .unwrap();
    let err =
        history::backfill_fplcache(&store, &Fplcache(rewritten), &clock(2026, 10, 3)).unwrap_err();
    assert!(err.to_string().starts_with("1 of 1"));
    let manifests = runs(&store);
    assert_eq!(
        (manifests[1]["written"].clone(), manifests[1]["skipped"].clone()),
        (json!(0), json!(1))
    );
    assert!(
        manifests[2]["failed"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("differs from the archived copy")
    );
}

#[test]
fn fplcache_truncated_download_is_recorded_in_the_manifest() {
    let (_d, store) = new_store();
    let tarball = tar_gz(&[("fplcache-x/cache/2021/4/18/1641.json.xz", xz(&[b' '; 20000]))]);
    let cut = tarball[..tarball.len() / 2].to_vec();
    assert!(history::backfill_fplcache(&store, &Fplcache(cut), &clock(2026, 10, 1)).is_err());
    let manifest = &runs(&store)[0];
    assert!(manifest["error"].as_str().is_some_and(|e| !e.is_empty()), "{manifest}");
}

// --- ESPN ----------------------------------------------------------------------------------

struct Espn {
    completed: Cell<bool>,
    summaries: RefCell<Vec<String>>,
}

impl EspnSource for Espn {
    fn scoreboard(&self, league: &str, period: &str) -> Result<Vec<u8>> {
        let events = if league == "eng.1" && period == "202610" {
            json!([{"id": "7", "date": "2026-10-04T14:00Z", "status": {"type": {"completed": true}},
                    "competitions": [{"competitors": [{"team": {"id": "1"}}, {"team": {"id": "2"}}]}]}])
        } else {
            json!([])
        };
        Ok(serde_json::to_vec(&json!({"events": events}))?)
    }

    fn summary(&self, league: &str, event_id: &str) -> Result<Vec<u8>> {
        self.summaries.borrow_mut().push(format!("{league}/{event_id}"));
        let status = json!({"type": {"completed": self.completed.get()}});
        Ok(serde_json::to_vec(&json!({"header": {"competitions": [{"status": status}]}}))?)
    }
}

#[test]
fn espn_recent_snapshot_is_resumable() {
    let (dir, store) = new_store();
    let source = Espn { completed: Cell::new(false), summaries: RefCell::new(vec![]) };
    let ids = BTreeSet::new();
    espn::snapshot_espn_recent(&store, &source, &ids, &clock(2026, 10, 5), Duration::ZERO).unwrap();
    // 4 months (late Sep to early Dec) x 8 leagues of scoreboards, one shared run timestamp
    let boards = dir.path().join("espn/eng.1/scoreboard/202610");
    assert_eq!(std::fs::read_dir(&boards).unwrap().count(), 1);
    assert_eq!(store.times("espn", "uefa.super_cup/scoreboard/202612").unwrap().len(), 1);
    assert_eq!(source.summaries.borrow().len(), 1);
    // an incomplete summary is fetched again; a completed one is not
    source.completed.set(true);
    espn::snapshot_espn_recent(&store, &source, &ids, &clock(2026, 10, 6), Duration::ZERO).unwrap();
    espn::snapshot_espn_recent(&store, &source, &ids, &clock(2026, 10, 7), Duration::ZERO).unwrap();
    assert_eq!(source.summaries.borrow().as_slice(), ["eng.1/7", "eng.1/7"]);
}
