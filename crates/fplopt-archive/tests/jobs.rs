//! The archiver jobs against fake sources: port of the behaviour tests in tests/test_jobs.py.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, bail};
use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use fplopt_archive::adapters::{FootballDataSource, FplSource, OddsSource};
use fplopt_archive::fetch::FetchError;
use fplopt_archive::jobs::*;
use fplopt_archive::raw_store::RawStore;
use fplopt_archive::{Clock, kind_name};
use serde_json::{Value, json};

fn deadline() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 10, 10, 0, 0).unwrap()
}

fn daily() -> DateTime<Utc> {
    deadline() - TimeDelta::hours(8)
}

/// Advances one second per `now` so consecutive writes get distinct timestamps.
struct FakeClock {
    t: Cell<DateTime<Utc>>,
    sleeps: RefCell<Vec<Duration>>,
}

fn clock(start: DateTime<Utc>) -> FakeClock {
    FakeClock { t: Cell::new(start), sleeps: RefCell::new(Vec::new()) }
}

impl Clock for FakeClock {
    fn now(&self) -> DateTime<Utc> {
        self.t.set(self.t.get() + TimeDelta::seconds(1));
        self.t.get()
    }

    fn sleep(&self, d: Duration) {
        self.sleeps.borrow_mut().push(d);
    }
}

/// Always the same instant.
struct Frozen(DateTime<Utc>);

impl Clock for Frozen {
    fn now(&self) -> DateTime<Utc> {
        self.0
    }

    fn sleep(&self, _: Duration) {}
}

/// `checked` marks those GW ids `data_checked`; the bootstrap lists GWs 1..max(checked,
/// n_events), one week apart from `first_deadline`.
#[derive(Default)]
struct FakeFpl {
    fail_ids: HashSet<i64>,
    element_ids: Vec<i64>,
    fail_fixtures: bool,
    checked: Vec<i64>,
    fail_live: HashSet<i64>,
    first_deadline: Option<DateTime<Utc>>,
    n_events: i64,
    summary_calls: RefCell<Vec<i64>>,
    live_calls: RefCell<Vec<i64>>,
}

fn fpl() -> FakeFpl {
    FakeFpl { element_ids: vec![1, 2], n_events: 1, ..Default::default() }
}

fn checked(gws: &[i64]) -> FakeFpl {
    FakeFpl { checked: gws.to_vec(), ..fpl() }
}

fn two_gw(gws: &[i64]) -> FakeFpl {
    FakeFpl { first_deadline: Some(deadline() - TimeDelta::weeks(1)), n_events: 2, ..checked(gws) }
}

impl FakeFpl {
    fn bootstrap(&self) -> Value {
        let first = self.first_deadline.unwrap_or_else(deadline);
        let last = self.checked.iter().copied().max().unwrap_or(1).max(self.n_events);
        let events: Vec<Value> = (1..=last)
            .map(|gw| {
                let at = first + TimeDelta::weeks(gw - 1);
                json!({"id": gw, "deadline_time": at.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                       "finished": self.checked.contains(&gw), "data_checked": self.checked.contains(&gw)})
            })
            .collect();
        let elements: Vec<Value> = self.element_ids.iter().map(|i| json!({"id": i})).collect();
        json!({"events": events, "elements": elements})
    }
}

impl FplSource for FakeFpl {
    fn bootstrap_static(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&self.bootstrap())?)
    }

    fn fixtures(&self) -> Result<Vec<u8>> {
        if self.fail_fixtures {
            bail!("fixtures down");
        }
        Ok(b"[]".to_vec())
    }

    fn element_summary(&self, id: i64) -> Result<Vec<u8>> {
        self.summary_calls.borrow_mut().push(id);
        if self.fail_ids.contains(&id) {
            bail!("boom");
        }
        Ok(serde_json::to_vec(&json!({"id": id, "history": []}))?)
    }

    fn event_live(&self, gw: i64) -> Result<Vec<u8>> {
        self.live_calls.borrow_mut().push(gw);
        if self.fail_live.contains(&gw) {
            bail!("live {gw} down\nsecond line");
        }
        Ok(b"{\"elements\": []}".to_vec())
    }
}

struct FakeOdds {
    failures: Cell<u32>,
    message: &'static str,
}

fn odds() -> FakeOdds {
    FakeOdds { failures: Cell::new(0), message: "odds down" }
}

impl OddsSource for FakeOdds {
    fn epl_odds(&self) -> Result<Vec<u8>> {
        if self.failures.get() > 0 {
            self.failures.set(self.failures.get() - 1);
            bail!("{}", self.message);
        }
        Ok(b"[]".to_vec())
    }
}

struct FakeFootballData {
    status: Option<u16>,
}

impl FootballDataSource for FakeFootballData {
    fn epl_season(&self, _start_year: i32) -> Result<Vec<u8>> {
        match self.status {
            Some(status) => Err(FetchError::Status {
                status,
                reason: "Not Found".into(),
                url: "https://x".into(),
                retry_after: None,
                body: vec![],
            }
            .into()),
            None => Ok(b"Div,Date,HomeTeam,AwayTeam\r\nE0,13/08/16,Burnley,Swansea\r\n".to_vec()),
        }
    }
}

fn new_store() -> (tempfile::TempDir, RawStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = RawStore::new(dir.path());
    (dir, store)
}

fn n(store: &RawStore, source: &str, endpoint: &str) -> usize {
    store.times(source, endpoint).unwrap().len()
}

fn live_times(store: &RawStore, gw: i64) -> usize {
    n(store, "fpl", &format!("event-live/2026-27/{gw}"))
}

fn summary_runs(root: &Path) -> Vec<PathBuf> {
    let dir = root.join("fpl/element-summary");
    let mut runs: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries.map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect(),
        Err(_) => vec![],
    };
    runs.sort();
    runs
}

fn daily_run(
    store: &RawStore,
    fpl: &FakeFpl,
    odds: Option<&dyn OddsSource>,
    at: DateTime<Utc>,
) -> Result<()> {
    run_daily(store, fpl, odds, &clock(at), None, None)
}

fn write_manifest(store: &RawStore, when: DateTime<Utc>, extra: Value) {
    let mut manifest =
        json!({"run_at": when.to_rfc3339(), "expected": [1, 2], "written": [1, 2], "failed": []});
    for (k, v) in extra.as_object().unwrap() {
        manifest[k] = v.clone();
    }
    store
        .write(
            "fpl",
            "element-summary",
            &serde_json::to_vec(&manifest).unwrap(),
            &when,
            Some("_manifest"),
        )
        .unwrap();
}

#[test]
fn daily_writes_bootstrap_fixtures_and_odds() {
    let (_d, store) = new_store();
    daily_run(&store, &fpl(), Some(&odds()), daily()).unwrap();
    assert_eq!(n(&store, "fpl", "bootstrap-static"), 1);
    assert_eq!(n(&store, "fpl", "fixtures"), 1);
    assert_eq!(n(&store, "odds", "soccer_epl"), 1);
}

#[test]
fn failed_fixtures_leaves_window_open_and_does_not_block_odds() {
    let (_d, store) = new_store();
    let bad = FakeFpl { fail_fixtures: true, ..fpl() };
    let err = daily_run(&store, &bad, Some(&odds()), daily()).unwrap_err();
    assert!(err.to_string().contains("fixtures down"));
    assert_eq!(n(&store, "fpl", "bootstrap-static"), 0);
    assert_eq!(n(&store, "odds", "soccer_epl"), 1);
}

#[test]
fn combined_error_is_one_line_naming_every_source() {
    let (_d, store) = new_store();
    let bad = FakeFpl { fail_fixtures: true, ..fpl() };
    let odds = FakeOdds { failures: Cell::new(1), message: "odds down\ndetails on a second line" };
    let err = daily_run(&store, &bad, Some(&odds), daily()).unwrap_err();
    assert_eq!(kind_name(&err), "SnapshotError");
    assert_eq!(err.to_string(), "fpl: Error: fixtures down; odds: Error: odds down");
}

#[test]
fn tick_snapshots_an_empty_store_then_once_per_window() {
    let (_d, store) = new_store();
    let fpl = fpl();
    assert!(run_tick(&store, &fpl, None, &clock(daily())).unwrap());
    assert!(!run_tick(&store, &fpl, None, &clock(deadline() - TimeDelta::hours(3))).unwrap());
    assert!(run_tick(&store, &fpl, None, &clock(deadline() - TimeDelta::minutes(110))).unwrap());
    assert!(!run_tick(&store, &fpl, None, &clock(deadline() - TimeDelta::minutes(95))).unwrap());
    assert_eq!(n(&store, "fpl", "bootstrap-static"), 2);
    assert!(
        !run_tick(&store, &fpl, Some(&odds()), &clock(deadline() + TimeDelta::minutes(5))).unwrap()
    );
}

#[test]
fn tick_retries_odds_until_captured_in_window() {
    let (_d, store) = new_store();
    let fpl = fpl();
    daily_run(&store, &fpl, Some(&odds()), daily()).unwrap();
    let flaky = FakeOdds { failures: Cell::new(1), message: "odds down" };
    let err = run_tick(&store, &fpl, Some(&flaky), &clock(deadline() - TimeDelta::minutes(110)))
        .unwrap_err();
    assert!(err.to_string().contains("odds down"));
    assert_eq!(n(&store, "fpl", "bootstrap-static"), 2);
    assert!(
        run_tick(&store, &fpl, Some(&flaky), &clock(deadline() - TimeDelta::minutes(95))).unwrap()
    );
    assert_eq!(n(&store, "odds", "soccer_epl"), 2);
    assert_eq!(n(&store, "fpl", "bootstrap-static"), 2);
    assert!(
        !run_tick(&store, &fpl, Some(&flaky), &clock(deadline() - TimeDelta::minutes(80))).unwrap()
    );
}

#[test]
fn backfill_archives_every_element_under_one_run() {
    let (dir, store) = new_store();
    let c = clock(deadline());
    assert_eq!(
        backfill_element_summaries(&store, &fpl(), &c, ELEMENT_SUMMARY_PAUSE, 10).unwrap(),
        2
    );
    let runs = summary_runs(dir.path());
    assert_eq!(runs.len(), 1);
    let mut names: Vec<String> = std::fs::read_dir(&runs[0])
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, ["1.json.gz", "2.json.gz", "_manifest.json.gz"]);
    assert_eq!(c.sleeps.borrow().as_slice(), [ELEMENT_SUMMARY_PAUSE; 2]);
}

#[test]
fn manifest_bytes_match_python() {
    let (_d, store) = new_store();
    let at = deadline();
    backfill_element_summaries(&store, &checked(&[1]), &Frozen(at), Duration::ZERO, 10).unwrap();
    let path =
        store.path_for("fpl", "element-summary", &at, ".json.gz", Some("_manifest")).unwrap();
    // json.dumps of Python's manifest dict, in its key order
    assert_eq!(
        String::from_utf8(RawStore::read_bytes(&path).unwrap()).unwrap(),
        r#"{"run_at": "2026-10-10T10:00:00+00:00", "finished_at": "2026-10-10T10:00:00+00:00", "season": 2026, "through_event": 1, "expected": [1, 2], "written": [1, 2], "failed": []}"#
    );
}

#[test]
fn backfill_tolerates_a_bootstrap_archived_in_the_same_second() {
    let (_d, store) = new_store();
    let fpl = fpl();
    let frozen = Frozen(deadline());
    snapshot_fpl(&store, &fpl, &frozen).unwrap();
    assert_eq!(backfill_element_summaries(&store, &fpl, &frozen, Duration::ZERO, 10).unwrap(), 2);
    assert_eq!(store.times("fpl", "bootstrap-static").unwrap(), [deadline()]);
}

#[test]
fn backfill_continues_past_failures_then_fails() {
    let (dir, store) = new_store();
    let bad = FakeFpl { fail_ids: [1].into(), ..fpl() };
    let err = backfill_element_summaries(&store, &bad, &clock(deadline()), Duration::ZERO, 10)
        .unwrap_err();
    assert_eq!(err.to_string(), "1 of 2 element summaries failed: [1]");
    let run = &summary_runs(dir.path())[0];
    assert!(run.join("2.json.gz").exists() && run.join("_manifest.json.gz").exists());
    assert!(!run.join("1.json.gz").exists());
}

#[test]
fn backfill_aborts_after_consecutive_failures_and_still_writes_manifest() {
    let (dir, store) = new_store();
    let bad = FakeFpl { fail_ids: [2, 3].into(), element_ids: vec![1, 2, 3, 4], ..fpl() };
    let err = backfill_element_summaries(&store, &bad, &clock(deadline()), Duration::ZERO, 2)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "aborting backfill after 2 consecutive failures (last element 3: Error)"
    );
    assert_eq!(bad.summary_calls.borrow().as_slice(), [1, 2, 3]);
    let manifest =
        RawStore::read_json(&summary_runs(dir.path())[0].join("_manifest.json.gz")).unwrap();
    assert_eq!(manifest["expected"], json!([1, 2, 3, 4]));
    assert_eq!(manifest["written"], json!([1]));
    assert_eq!(manifest["failed"], json!([2, 3]));
}

#[test]
fn backfill_consecutive_counter_resets_on_success() {
    let (_d, store) = new_store();
    let bad = FakeFpl { fail_ids: [1, 3, 5].into(), element_ids: vec![1, 2, 3, 4, 5, 6], ..fpl() };
    let err = backfill_element_summaries(&store, &bad, &clock(deadline()), Duration::ZERO, 2)
        .unwrap_err();
    assert!(err.to_string().starts_with("3 of 6"));
    assert_eq!(bad.summary_calls.borrow().len(), 6);
}

#[test]
fn football_data_by_season_code_and_404_grace() {
    let (_d, store) = new_store();
    let fd = FakeFootballData { status: None };
    let path = snapshot_football_data(&store, &fd, 2016, &clock(deadline())).unwrap();
    assert!(path.to_string_lossy().replace('\\', "/").contains("football-data/E0/1617/"));
    assert!(path.to_string_lossy().ends_with(".csv.gz"));
    let missing = FakeFootballData { status: Some(404) };
    let august = Utc.with_ymd_and_hms(2026, 8, 1, 2, 30, 0).unwrap();
    assert!(snapshot_football_data_current(&store, &missing, &clock(august)).unwrap().is_none());
    let err = snapshot_football_data_current(&store, &missing, &clock(deadline())).unwrap_err();
    assert_eq!(kind_name(&err), "HTTPStatusError");
    let path = snapshot_football_data_current(&store, &fd, &clock(deadline())).unwrap().unwrap();
    assert!(path.to_string_lossy().replace('\\', "/").contains("E0/2627/"));
}

#[test]
fn run_independently_runs_every_step() {
    let ran = RefCell::new(Vec::new());
    let err = run_independently(vec![
        step("a", || {
            ran.borrow_mut().push("a");
            bail!("first\nsecond")
        }),
        step("b", || {
            ran.borrow_mut().push("b");
            Ok(())
        }),
        step("c", || bail!("")),
    ])
    .unwrap_err();
    assert_eq!(ran.borrow().as_slice(), ["a", "b"]);
    assert_eq!(err.to_string(), "a: Error: first; c: Error: ");
}

#[test]
fn checked_events_tolerates_missing_flag() {
    let b = json!({"events": [{"id": 1, "data_checked": true}, {"id": 3, "data_checked": true},
                              {"id": 2, "data_checked": false}, {"id": 4}]});
    assert_eq!(checked_events(&b).unwrap(), [1, 3]);
    assert!(checked_events(&json!({"events": null})).unwrap().is_empty());
}

#[test]
fn daily_without_finalised_gws_skips_post_lockdown() {
    let (dir, store) = new_store();
    let fpl = fpl();
    daily_run(&store, &fpl, None, daily()).unwrap();
    assert!(fpl.live_calls.borrow().is_empty() && fpl.summary_calls.borrow().is_empty());
    assert!(!dir.path().join("fpl/event-live").exists());
}

#[test]
fn daily_archives_finalised_gws_and_refreshes_element_summaries() {
    let (dir, store) = new_store();
    let fpl = checked(&[1, 2]);
    let c = clock(daily());
    run_daily(&store, &fpl, None, &c, None, None).unwrap();
    assert_eq!(fpl.live_calls.borrow().as_slice(), [1, 2]);
    assert_eq!((live_times(&store, 1), live_times(&store, 2)), (1, 1));
    assert_eq!(fpl.summary_calls.borrow().as_slice(), [1, 2]);
    assert!(!c.sleeps.borrow().is_empty());
    let runs = summary_runs(dir.path());
    let manifest = RawStore::read_json(&runs[0].join("_manifest.json.gz")).unwrap();
    assert_eq!(
        (manifest["season"].clone(), manifest["through_event"].clone()),
        (json!(2026), json!(2))
    );
    assert_eq!(latest_complete_element_summary(&store).unwrap(), (2026, 2));
}

#[test]
fn post_lockdown_is_idempotent_and_picks_up_new_gws() {
    let (dir, store) = new_store();
    daily_run(&store, &checked(&[1, 2]), None, daily()).unwrap();
    let again = checked(&[1, 2]);
    daily_run(&store, &again, None, daily() + TimeDelta::days(1)).unwrap();
    assert!(again.live_calls.borrow().is_empty() && again.summary_calls.borrow().is_empty());
    assert_eq!(summary_runs(dir.path()).len(), 1);
    let next = checked(&[1, 2, 3]);
    daily_run(&store, &next, None, daily() + TimeDelta::days(7)).unwrap();
    assert_eq!(next.live_calls.borrow().as_slice(), [3]);
    assert_eq!(next.summary_calls.borrow().as_slice(), [1, 2]);
    assert_eq!(latest_complete_element_summary(&store).unwrap(), (2026, 3));
}

#[test]
fn failed_summary_run_is_retried_next_day() {
    let (_d, store) = new_store();
    let bad = FakeFpl { fail_ids: [2].into(), ..checked(&[1]) };
    let err = daily_run(&store, &bad, None, daily()).unwrap_err();
    assert!(err.to_string().contains("post-lockdown"));
    assert_eq!(latest_complete_element_summary(&store).unwrap(), (0, 0));
    let retry = checked(&[1]);
    daily_run(&store, &retry, None, daily() + TimeDelta::days(1)).unwrap();
    assert!(retry.live_calls.borrow().is_empty());
    assert_eq!(retry.summary_calls.borrow().as_slice(), [1, 2]);
    assert_eq!(latest_complete_element_summary(&store).unwrap(), (2026, 1));
}

#[test]
fn event_live_failure_does_not_block_other_gws_or_summaries() {
    let (_d, store) = new_store();
    let bad = FakeFpl { fail_live: [1].into(), ..checked(&[1, 2]) };
    let err = daily_run(&store, &bad, None, daily()).unwrap_err();
    let message = err.to_string();
    assert!(!message.contains('\n'));
    assert!(message.contains("post-lockdown") && message.contains("live 1 down"), "{message}");
    assert_eq!((live_times(&store, 1), live_times(&store, 2)), (0, 1));
    assert_eq!(bad.summary_calls.borrow().as_slice(), [1, 2]);
    let retry = checked(&[1, 2]);
    daily_run(&store, &retry, None, daily() + TimeDelta::days(1)).unwrap();
    assert_eq!(retry.live_calls.borrow().as_slice(), [1]);
    assert!(retry.summary_calls.borrow().is_empty());
}

#[test]
fn event_live_puts_season_in_the_path() {
    let (_d, store) = new_store();
    let next = FakeFpl {
        first_deadline: Some(Utc.with_ymd_and_hms(2027, 8, 14, 10, 0, 0).unwrap()),
        ..checked(&[1])
    };
    let written =
        snapshot_event_live(&store, &fpl(), &next.bootstrap(), &clock(daily()), Duration::ZERO)
            .unwrap();
    assert_eq!(written, [1]);
    assert_eq!(n(&store, "fpl", "event-live/2027-28/1"), 1);
}

#[test]
fn post_lockdown_skips_missing_or_stale_bootstraps() {
    let (_d, store) = new_store();
    let fpl = checked(&[1]);
    snapshot_post_lockdown(&store, &fpl, &clock(daily())).unwrap();
    store
        .write("fpl", "bootstrap-static", &fpl.bootstrap_static().unwrap(), &daily(), None)
        .unwrap();
    snapshot_post_lockdown(&store, &fpl, &clock(daily() + TimeDelta::hours(7))).unwrap();
    assert!(fpl.live_calls.borrow().is_empty() && fpl.summary_calls.borrow().is_empty());
    snapshot_post_lockdown(&store, &fpl, &clock(daily() + TimeDelta::hours(5))).unwrap();
    assert_eq!(fpl.live_calls.borrow().as_slice(), [1]);
}

#[test]
fn manifests_that_do_not_count() {
    let (dir, store) = new_store();
    write_manifest(&store, daily(), json!({})); // old style: no season / through_event
    assert_eq!(latest_complete_element_summary(&store).unwrap(), (0, 0));
    write_manifest(
        &store,
        daily() + TimeDelta::hours(1),
        json!({"season": 2026, "through_event": 3, "written": [1], "failed": [2]}),
    );
    write_manifest(
        &store,
        daily() + TimeDelta::hours(2),
        json!({"season": 2026, "through_event": 4, "written": [1]}),
    );
    write_manifest(
        &store,
        daily() + TimeDelta::hours(3),
        json!({"season": true, "through_event": 9}),
    );
    assert_eq!(latest_complete_element_summary(&store).unwrap(), (0, 0));
    write_manifest(
        &store,
        daily() + TimeDelta::hours(4),
        json!({"season": 2026, "through_event": 2}),
    );
    // a run killed before its manifest, a non-timestamp directory and a stray file
    store
        .write("fpl", "element-summary", b"{}", &(daily() + TimeDelta::days(1)), Some("1"))
        .unwrap();
    let root = dir.path().join("fpl/element-summary");
    std::fs::create_dir(root.join("scratch")).unwrap();
    std::fs::write(root.join("notes.txt"), "x").unwrap();
    let broken = store
        .path_for(
            "fpl",
            "element-summary",
            &(daily() + TimeDelta::days(2)),
            ".json.gz",
            Some("_manifest"),
        )
        .unwrap();
    std::fs::create_dir_all(broken.parent().unwrap()).unwrap();
    std::fs::write(&broken, b"not gzip").unwrap();
    assert_eq!(latest_complete_element_summary(&store).unwrap(), (2026, 2));
}

#[test]
fn season_compares_before_gameweek() {
    let (_d, store) = new_store();
    let at = |m, d| Utc.with_ymd_and_hms(2027, m, d, 2, 30, 0).unwrap();
    write_manifest(&store, at(5, 26), json!({"season": 2026, "through_event": 38}));
    let new_season =
        |gws: &[i64]| FakeFpl { first_deadline: Some(at(8, 14)), ..checked(gws) }.bootstrap();
    assert!(!element_summaries_due(&store, &new_season(&[])).unwrap());
    assert!(element_summaries_due(&store, &new_season(&[1])).unwrap());
    write_manifest(&store, at(8, 20), json!({"season": 2027, "through_event": 1}));
    assert!(!element_summaries_due(&store, &new_season(&[1])).unwrap());
}

#[test]
fn summary_manifest_tolerates_bootstrap_without_events() {
    struct NoEvents;
    impl FplSource for NoEvents {
        fn bootstrap_static(&self) -> Result<Vec<u8>> {
            Ok(br#"{"events": [], "elements": [{"id": 1}]}"#.to_vec())
        }
        fn fixtures(&self) -> Result<Vec<u8>> {
            unreachable!()
        }
        fn element_summary(&self, _: i64) -> Result<Vec<u8>> {
            Ok(b"{}".to_vec())
        }
        fn event_live(&self, _: i64) -> Result<Vec<u8>> {
            unreachable!()
        }
    }
    let (dir, store) = new_store();
    assert_eq!(
        backfill_element_summaries(&store, &NoEvents, &clock(daily()), Duration::ZERO, 10).unwrap(),
        1
    );
    let manifest =
        RawStore::read_json(&summary_runs(dir.path())[0].join("_manifest.json.gz")).unwrap();
    assert_eq!(
        (manifest["season"].clone(), manifest["through_event"].clone()),
        (Value::Null, json!(0))
    );
}

#[test]
fn pre_deadline_tick_runs_post_lockdown() {
    let in_window = deadline() - TimeDelta::minutes(110);
    let (_d, store) = new_store();
    daily_run(&store, &two_gw(&[]), None, daily()).unwrap();
    let fpl = two_gw(&[1]);
    assert!(run_tick(&store, &fpl, None, &clock(in_window)).unwrap());
    assert_eq!(fpl.live_calls.borrow().as_slice(), [1]);
    assert_eq!(fpl.summary_calls.borrow().as_slice(), [1, 2]);
    assert_eq!(latest_complete_element_summary(&store).unwrap(), (2026, 1));

    let (_d, store) = new_store();
    daily_run(&store, &two_gw(&[1]), None, daily()).unwrap();
    let fpl = two_gw(&[1]);
    assert!(run_tick(&store, &fpl, None, &clock(in_window)).unwrap());
    assert_eq!(n(&store, "fpl", "bootstrap-static"), 3); // daily, its summary run, tick
    assert!(fpl.live_calls.borrow().is_empty() && fpl.summary_calls.borrow().is_empty());
}

#[test]
fn tick_outside_window_skips_post_lockdown_and_failures_do_not_hide_snapshots() {
    let (_d, store) = new_store();
    daily_run(&store, &two_gw(&[]), None, daily()).unwrap();
    store
        .write(
            "fpl",
            "bootstrap-static",
            &two_gw(&[1]).bootstrap_static().unwrap(),
            &(daily() + TimeDelta::hours(1)),
            None,
        )
        .unwrap();
    let fpl = two_gw(&[1]);
    assert!(!run_tick(&store, &fpl, None, &clock(daily() + TimeDelta::hours(2))).unwrap());
    assert!(fpl.live_calls.borrow().is_empty());

    let (_d, store) = new_store();
    daily_run(&store, &two_gw(&[]), None, daily()).unwrap();
    let bad = FakeFpl { fail_live: [1].into(), ..two_gw(&[1]) };
    let err = run_tick(&store, &bad, Some(&odds()), &clock(deadline() - TimeDelta::minutes(110)))
        .unwrap_err();
    assert!(err.to_string().contains("post-lockdown"));
    assert_eq!(n(&store, "fpl", "bootstrap-static"), 3);
    assert_eq!(n(&store, "odds", "soccer_epl"), 1);
    assert_eq!(bad.summary_calls.borrow().as_slice(), [1, 2]);
}

#[test]
fn a_teams_csv_problem_fails_only_the_espn_step() {
    struct NoEspn;
    impl fplopt_archive::adapters::EspnSource for NoEspn {
        fn scoreboard(&self, _: &str, _: &str) -> Result<Vec<u8>> {
            unreachable!()
        }
        fn summary(&self, _: &str, _: &str) -> Result<Vec<u8>> {
            unreachable!()
        }
    }
    let (dir, store) = new_store();
    let missing = dir.path().join("missing.csv");
    let espn = DailyEspn { source: &NoEspn, teams_csv: &missing };
    let err = run_daily(&store, &fpl(), None, &clock(daily()), None, Some(espn)).unwrap_err();
    assert!(err.to_string().starts_with("espn: FileNotFoundError"), "{err}");
    assert_eq!(n(&store, "fpl", "bootstrap-static"), 1);
}
