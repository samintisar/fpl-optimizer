import gzip
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from fplopt.ingest.jobs import (
    SnapshotError,
    backfill_element_summaries,
    checked_events,
    element_summaries_due,
    latest_complete_element_summary,
    run_daily,
    run_independently,
    run_tick,
    snapshot_event_live,
    snapshot_football_data,
    snapshot_football_data_current,
    snapshot_fpl,
    snapshot_post_lockdown,
)
from fplopt.ingest.raw_store import RawStore

DEADLINE = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)


class MultiLineFailingOdds:
    def epl_odds(self):
        raise RuntimeError("odds down\ndetails on a second line")


class FakeFpl:
    """`checked` marks those GW ids `data_checked` (finalised); the bootstrap lists GWs
    1..max(checked), one week apart from `first_deadline`."""

    def __init__(
        self,
        fail_ids=(),
        element_ids=(1, 2),
        fail_fixtures=False,
        clock=None,
        checked=(),
        fail_live=(),
        first_deadline=DEADLINE,
    ):
        self.fail_ids = set(fail_ids)
        self.element_ids = list(element_ids)
        self.fail_fixtures = fail_fixtures
        self.clock = clock
        self.checked = set(checked)
        self.fail_live = set(fail_live)
        self.first_deadline = first_deadline
        self.fetched = []
        self.summary_calls = []
        self.live_calls = []

    def events(self):
        return [
            {
                "id": gw,
                "deadline_time": (self.first_deadline + timedelta(weeks=gw - 1)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "finished": gw in self.checked,
                "data_checked": gw in self.checked,
            }
            for gw in range(1, max(self.checked, default=1) + 1)
        ]

    def bootstrap_static(self):
        if self.clock:
            self.fetched.append(("bootstrap", self.clock()))
        return json.dumps(
            {
                "events": self.events(),
                "elements": [{"id": i} for i in self.element_ids],
            }
        ).encode()

    def fixtures(self):
        if self.fail_fixtures:
            raise RuntimeError("fixtures down")
        if self.clock:
            self.fetched.append(("fixtures", self.clock()))
        return b"[]"

    def element_summary(self, element_id):
        self.summary_calls.append(element_id)
        if element_id in self.fail_ids:
            raise RuntimeError("boom")
        return json.dumps({"id": element_id, "history": []}).encode()

    def event_live(self, gw):
        self.live_calls.append(gw)
        if gw in self.fail_live:
            raise RuntimeError(f"live {gw} down\nsecond line")
        return json.dumps({"elements": []}).encode()


class FakeOdds:
    def epl_odds(self):
        return b"[]"


class Clock:
    """Advances one second per call so consecutive writes get distinct timestamps."""

    def __init__(self, start):
        self.t = start

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


class FlakyOdds:
    def __init__(self, failures):
        self.failures = failures

    def epl_odds(self):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("odds down")
        return b"[]"


def no_sleep(_seconds):
    pass


def test_daily_writes_bootstrap_fixtures_and_odds(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(), FakeOdds(), Clock(DEADLINE - timedelta(hours=8)))
    assert len(store.times("fpl", "bootstrap-static")) == 1
    assert len(store.times("fpl", "fixtures")) == 1
    assert len(store.times("odds", "soccer_epl")) == 1


def test_daily_without_odds_key_still_archives_fpl(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(), None, Clock(DEADLINE - timedelta(hours=8)))
    assert store.times("odds", "soccer_epl") == []
    assert len(store.times("fpl", "fixtures")) == 1


def test_failed_fixtures_leaves_window_open(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(RuntimeError):
        run_daily(
            store, FakeFpl(fail_fixtures=True), None, Clock(DEADLINE - timedelta(minutes=110))
        )
    assert store.times("fpl", "bootstrap-static") == []


def test_fpl_failure_does_not_block_odds(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(RuntimeError, match="fixtures down"):
        run_daily(
            store, FakeFpl(fail_fixtures=True), FakeOdds(), Clock(DEADLINE - timedelta(hours=8))
        )
    assert len(store.times("odds", "soccer_epl")) == 1


def test_timestamps_are_taken_after_each_fetch(tmp_path):
    clock = Clock(DEADLINE - timedelta(hours=8))
    fpl = FakeFpl(clock=clock)
    store = RawStore(tmp_path)
    run_daily(store, fpl, None, clock)
    fetched = dict(fpl.fetched)
    assert store.times("fpl", "fixtures")[0] > fetched["fixtures"]
    assert store.times("fpl", "bootstrap-static")[0] > fetched["bootstrap"]


def test_backfill_run_timestamp_is_after_bootstrap_fetch(tmp_path):
    clock = Clock(DEADLINE)
    fpl = FakeFpl(clock=clock)
    store = RawStore(tmp_path)
    backfill_element_summaries(store, fpl, clock, sleep=no_sleep)
    assert store.times("fpl", "bootstrap-static")[0] > dict(fpl.fetched)["bootstrap"]


def test_tick_snapshots_an_empty_store(tmp_path):
    store = RawStore(tmp_path)
    assert run_tick(store, FakeFpl(), None, Clock(DEADLINE - timedelta(hours=8))) is True
    assert len(store.times("fpl", "bootstrap-static")) == 1


def test_tick_runs_once_per_pre_deadline_window(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl()
    run_daily(store, fpl, None, Clock(DEADLINE - timedelta(hours=8)))
    assert run_tick(store, fpl, None, Clock(DEADLINE - timedelta(hours=3))) is False
    assert run_tick(store, fpl, None, Clock(DEADLINE - timedelta(minutes=110))) is True
    assert run_tick(store, fpl, None, Clock(DEADLINE - timedelta(minutes=95))) is False
    assert len(store.times("fpl", "bootstrap-static")) == 2


def test_backfill_archives_every_element_under_one_run(tmp_path):
    store = RawStore(tmp_path)
    count = backfill_element_summaries(store, FakeFpl(), Clock(DEADLINE), sleep=no_sleep)
    assert count == 2
    run_dirs = list((tmp_path / "fpl" / "element-summary").iterdir())
    assert len(run_dirs) == 1
    assert sorted(p.name for p in run_dirs[0].iterdir()) == [
        "1.json.gz",
        "2.json.gz",
        "_manifest.json.gz",
    ]


def test_backfill_tolerates_a_bootstrap_archived_in_the_same_second(tmp_path):
    # The daily FPL step and the element-summary run can both archive a bootstrap in one second.
    store = RawStore(tmp_path)
    fpl = FakeFpl()
    frozen = lambda: DEADLINE  # noqa: E731
    snapshot_fpl(store, fpl, frozen)
    assert backfill_element_summaries(store, fpl, frozen, sleep=no_sleep) == 2
    assert store.times("fpl", "bootstrap-static") == [DEADLINE]
    manifest = store.read_json(store.path_for("fpl", "element-summary", DEADLINE, name="_manifest"))
    assert manifest["written"] == [1, 2]


def test_backfill_continues_past_failures_then_raises(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(RuntimeError, match="1 of 2"):
        backfill_element_summaries(store, FakeFpl(fail_ids={1}), Clock(DEADLINE), sleep=no_sleep)
    run_dir = next((tmp_path / "fpl" / "element-summary").iterdir())
    assert sorted(p.name for p in run_dir.iterdir()) == ["2.json.gz", "_manifest.json.gz"]


def test_backfill_aborts_after_consecutive_failures(tmp_path):
    fpl = FakeFpl(fail_ids={1, 2, 3}, element_ids=(1, 2, 3, 4, 5))
    with pytest.raises(RuntimeError, match="3 consecutive"):
        backfill_element_summaries(
            RawStore(tmp_path), fpl, Clock(DEADLINE), sleep=no_sleep, max_consecutive_failures=3
        )
    assert fpl.summary_calls == [1, 2, 3]


def test_tick_retries_odds_until_captured_in_window(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl()
    odds = FlakyOdds(failures=1)
    run_daily(store, fpl, FakeOdds(), Clock(DEADLINE - timedelta(hours=8)))
    with pytest.raises(RuntimeError, match="odds down"):
        run_tick(store, fpl, odds, Clock(DEADLINE - timedelta(minutes=110)))
    assert len(store.times("fpl", "bootstrap-static")) == 2
    assert run_tick(store, fpl, odds, Clock(DEADLINE - timedelta(minutes=95))) is True
    assert len(store.times("odds", "soccer_epl")) == 2
    assert len(store.times("fpl", "bootstrap-static")) == 2
    assert run_tick(store, fpl, odds, Clock(DEADLINE - timedelta(minutes=80))) is False


def test_tick_retries_fpl_after_failure_in_window(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(), None, Clock(DEADLINE - timedelta(hours=8)))
    with pytest.raises(RuntimeError, match="fixtures down"):
        run_tick(store, FakeFpl(fail_fixtures=True), None, Clock(DEADLINE - timedelta(minutes=110)))
    assert run_tick(store, FakeFpl(), None, Clock(DEADLINE - timedelta(minutes=95))) is True
    assert len(store.times("fpl", "bootstrap-static")) == 2


def test_tick_after_deadline_does_nothing(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(), None, Clock(DEADLINE - timedelta(hours=8)))
    assert run_tick(store, FakeFpl(), FakeOdds(), Clock(DEADLINE + timedelta(minutes=5))) is False


def test_backfill_consecutive_counter_resets_on_success(tmp_path):
    fpl = FakeFpl(fail_ids={1, 3, 5}, element_ids=(1, 2, 3, 4, 5, 6))
    with pytest.raises(RuntimeError, match="3 of 6"):
        backfill_element_summaries(
            RawStore(tmp_path), fpl, Clock(DEADLINE), sleep=no_sleep, max_consecutive_failures=2
        )
    assert fpl.summary_calls == [1, 2, 3, 4, 5, 6]


def test_backfill_writes_manifest_even_when_aborted(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl(fail_ids={2, 3}, element_ids=(1, 2, 3, 4))
    with pytest.raises(RuntimeError, match="2 consecutive"):
        backfill_element_summaries(
            store, fpl, Clock(DEADLINE), sleep=no_sleep, max_consecutive_failures=2
        )
    run_dir = next((tmp_path / "fpl" / "element-summary").iterdir())
    manifest = store.read_json(run_dir / "_manifest.json.gz")
    assert manifest["expected"] == [1, 2, 3, 4]
    assert manifest["written"] == [1]
    assert manifest["failed"] == [2, 3]


def test_combined_error_is_one_line_naming_every_source(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl(fail_fixtures=True)
    with pytest.raises(RuntimeError) as info:
        run_daily(store, fpl, MultiLineFailingOdds(), Clock(DEADLINE - timedelta(hours=8)))
    message = str(info.value)
    assert "\n" not in message
    assert "fixtures down" in message
    assert "odds down" in message


CSV = b"Div,Date,HomeTeam,AwayTeam\r\nE0,13/08/16,Burnley,Swansea\r\n"


class FakeFootballData:
    def __init__(self, fail=None):
        self.fail = fail
        self.seasons = []

    def epl_season(self, start_year):
        self.seasons.append(start_year)
        if self.fail is not None:
            raise self.fail
        return CSV


def not_found():
    request = httpx.Request("GET", "https://x")
    return httpx.HTTPStatusError(
        "404", request=request, response=httpx.Response(404, request=request)
    )


def test_snapshot_football_data_writes_gzipped_csv_by_season_code(tmp_path):
    store = RawStore(tmp_path)
    path = snapshot_football_data(store, FakeFootballData(), 2016, Clock(DEADLINE))
    assert path.parent == tmp_path / "football-data" / "E0" / "1617"
    assert path.name.endswith(".csv.gz")
    assert RawStore.read_bytes(path) == CSV
    assert len(store.times("football-data", "E0/1617", suffix=".csv.gz")) == 1


def test_current_season_404_in_august_is_tolerated(tmp_path):
    store = RawStore(tmp_path)
    fd = FakeFootballData(fail=not_found())
    clock = Clock(datetime(2026, 8, 10, 3, 0, tzinfo=UTC))
    assert snapshot_football_data_current(store, fd, clock) is None
    assert fd.seasons == [2026]
    assert not (tmp_path / "football-data").exists()


def test_current_season_404_in_october_raises(tmp_path):
    fd = FakeFootballData(fail=not_found())
    with pytest.raises(httpx.HTTPStatusError):
        snapshot_football_data_current(RawStore(tmp_path), fd, Clock(DEADLINE))


def test_current_season_snapshot_uses_season_in_progress(tmp_path):
    store = RawStore(tmp_path)
    fd = FakeFootballData()
    path = snapshot_football_data_current(store, fd, Clock(datetime(2027, 1, 15, tzinfo=UTC)))
    assert fd.seasons == [2026]
    assert path.parent.name == "2627"


def test_daily_also_archives_current_football_data_csv(tmp_path):
    store = RawStore(tmp_path)
    fd = FakeFootballData()
    run_daily(store, FakeFpl(), FakeOdds(), Clock(DEADLINE - timedelta(hours=8)), football_data=fd)
    assert len(store.times("fpl", "bootstrap-static")) == 1
    assert len(store.times("odds", "soccer_epl")) == 1
    assert len(store.times("football-data", "E0/2627", suffix=".csv.gz")) == 1


def test_daily_football_data_failure_does_not_block_fpl_or_odds(tmp_path):
    store = RawStore(tmp_path)
    fd = FakeFootballData(fail=RuntimeError("fd down"))
    with pytest.raises(SnapshotError) as info:
        run_daily(
            store, FakeFpl(), FakeOdds(), Clock(DEADLINE - timedelta(hours=8)), football_data=fd
        )
    assert "football-data" in str(info.value)
    assert "fd down" in str(info.value)
    assert len(store.times("fpl", "bootstrap-static")) == 1
    assert len(store.times("odds", "soccer_epl")) == 1


def test_daily_without_football_data_source_skips_it(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(), None, Clock(DEADLINE - timedelta(hours=8)))
    assert not (tmp_path / "football-data").exists()


def test_run_independently_runs_every_step_and_names_failures():
    ran = []

    def fail():
        ran.append("b")
        raise ValueError("bad\nsecond line")

    with pytest.raises(SnapshotError) as info:
        run_independently(
            [("a", lambda: ran.append("a")), ("b", fail), ("c", lambda: ran.append("c"))]
        )
    assert ran == ["a", "b", "c"]
    assert str(info.value) == "b: ValueError: bad"


# --- post-lockdown ingest ----------------------------------------------------------------

DAILY = DEADLINE - timedelta(hours=8)


def event_live_times(store, gw, season="2026-27"):
    return store.times("fpl", f"event-live/{season}/{gw}")


def summary_runs(tmp_path):
    root = tmp_path / "fpl" / "element-summary"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def write_manifest(store, when, **fields):
    manifest = {"run_at": when.isoformat(), "expected": [1, 2], "written": [1, 2], "failed": []}
    manifest.update(fields)
    return store.write(
        "fpl", "element-summary", json.dumps(manifest).encode(), when, name="_manifest"
    )


def bootstrap(first_deadline, checked):
    return json.loads(FakeFpl(checked=checked, first_deadline=first_deadline).bootstrap_static())


def test_checked_events_lists_finalised_gws_and_tolerates_missing_flag():
    events = [
        {"id": 1, "data_checked": True},
        {"id": 3, "data_checked": True},
        {"id": 2, "data_checked": False},
        {"id": 4},
    ]
    assert checked_events({"events": events}) == [1, 3]


def test_daily_without_finalised_gws_skips_post_lockdown(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl()
    run_daily(store, fpl, None, Clock(DAILY), sleep=no_sleep)
    assert fpl.live_calls == []
    assert fpl.summary_calls == []
    assert not (tmp_path / "fpl" / "event-live").exists()


def test_daily_archives_finalised_gws_and_refreshes_element_summaries(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl(checked=(1, 2))
    sleeps = []
    run_daily(store, fpl, None, Clock(DAILY), sleep=sleeps.append)
    assert fpl.live_calls == [1, 2]
    assert len(event_live_times(store, 1)) == 1
    assert len(event_live_times(store, 2)) == 1
    assert fpl.summary_calls == [1, 2]
    assert sleeps  # the injected sleep paces the element-summary run
    (run,) = summary_runs(tmp_path)
    manifest = store.read_json(run / "_manifest.json.gz")
    assert manifest["season"] == 2026
    assert manifest["through_event"] == 2
    assert latest_complete_element_summary(store) == (2026, 2)


def test_post_lockdown_reads_the_bootstrap_fetched_by_the_same_daily_run(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(), None, Clock(DAILY), sleep=no_sleep)
    fpl = FakeFpl(checked=(1,))
    run_daily(store, fpl, None, Clock(DAILY + timedelta(days=1)), sleep=no_sleep)
    assert fpl.live_calls == [1]


def test_daily_again_with_same_bootstrap_fetches_nothing_new(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(checked=(1, 2)), None, Clock(DAILY), sleep=no_sleep)
    fpl = FakeFpl(checked=(1, 2))
    run_daily(store, fpl, None, Clock(DAILY + timedelta(days=1)), sleep=no_sleep)
    assert fpl.live_calls == []
    assert fpl.summary_calls == []
    assert len(summary_runs(tmp_path)) == 1
    assert len(event_live_times(store, 1)) == 1


def test_newly_finalised_gw_gets_event_live_and_a_new_summary_run(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(checked=(1, 2)), None, Clock(DAILY), sleep=no_sleep)
    fpl = FakeFpl(checked=(1, 2, 3))
    run_daily(store, fpl, None, Clock(DAILY + timedelta(days=7)), sleep=no_sleep)
    assert fpl.live_calls == [3]
    assert fpl.summary_calls == [1, 2]
    assert len(summary_runs(tmp_path)) == 2
    assert latest_complete_element_summary(store) == (2026, 3)


def test_failed_summary_run_is_retried_next_day(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(SnapshotError) as info:
        run_daily(store, FakeFpl(checked=(1,), fail_ids={2}), None, Clock(DAILY), sleep=no_sleep)
    assert "post-lockdown" in str(info.value)
    assert latest_complete_element_summary(store) == (0, 0)
    fpl = FakeFpl(checked=(1,))
    run_daily(store, fpl, None, Clock(DAILY + timedelta(days=1)), sleep=no_sleep)
    assert fpl.live_calls == []  # event-live for GW1 was archived on the first day
    assert fpl.summary_calls == [1, 2]
    assert latest_complete_element_summary(store) == (2026, 1)


def test_event_live_failure_does_not_block_other_gws_or_summaries(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl(checked=(1, 2), fail_live={1})
    with pytest.raises(SnapshotError) as info:
        run_daily(store, fpl, None, Clock(DAILY), sleep=no_sleep)
    message = str(info.value)
    assert "\n" not in message
    assert "post-lockdown" in message
    assert "live 1 down" in message
    assert event_live_times(store, 1) == []
    assert len(event_live_times(store, 2)) == 1
    assert fpl.summary_calls == [1, 2]
    retry = FakeFpl(checked=(1, 2))
    run_daily(store, retry, None, Clock(DAILY + timedelta(days=1)), sleep=no_sleep)
    assert retry.live_calls == [1]
    assert retry.summary_calls == []


def test_snapshot_event_live_puts_season_in_the_path(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl()
    new_season = bootstrap(datetime(2027, 8, 14, 10, 0, tzinfo=UTC), (1,))
    assert snapshot_event_live(store, fpl, new_season, Clock(DAILY)) == [1]
    assert len(event_live_times(store, 1, season="2027-28")) == 1


def test_post_lockdown_without_any_bootstrap_is_a_no_op(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl(checked=(1,))
    snapshot_post_lockdown(store, fpl, Clock(DAILY), sleep=no_sleep)
    assert fpl.live_calls == []
    assert fpl.summary_calls == []


def archive_bootstrap(store, when, checked):
    store.write("fpl", "bootstrap-static", FakeFpl(checked=checked).bootstrap_static(), when)


def test_post_lockdown_skips_a_stale_bootstrap(tmp_path, caplog):
    # e.g. today's FPL step failed during the July reset: the newest bootstrap is last
    # season's, but event/{gw}/live would return new-season data under the old season's label.
    store = RawStore(tmp_path)
    archive_bootstrap(store, DAILY, checked=(1,))
    fpl = FakeFpl(checked=(1,))
    snapshot_post_lockdown(store, fpl, Clock(DAILY + timedelta(hours=7)), sleep=no_sleep)
    assert fpl.live_calls == []
    assert fpl.summary_calls == []
    assert "stale" in caplog.text


def test_post_lockdown_uses_a_bootstrap_from_the_last_six_hours(tmp_path):
    store = RawStore(tmp_path)
    archive_bootstrap(store, DAILY, checked=(1,))
    fpl = FakeFpl(checked=(1,))
    snapshot_post_lockdown(store, fpl, Clock(DAILY + timedelta(hours=5)), sleep=no_sleep)
    assert fpl.live_calls == [1]
    assert fpl.summary_calls == [1, 2]


def test_daily_with_failed_fpl_step_skips_post_lockdown_on_a_stale_bootstrap(tmp_path):
    store = RawStore(tmp_path)
    archive_bootstrap(store, DAILY, checked=(1,))
    fpl = FakeFpl(checked=(1,), fail_fixtures=True)
    with pytest.raises(SnapshotError) as info:
        run_daily(store, fpl, None, Clock(DAILY + timedelta(days=1)), sleep=no_sleep)
    assert "post-lockdown" not in str(info.value)
    assert fpl.live_calls == []


def test_tick_on_empty_store_does_not_run_post_lockdown(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl(checked=(1,))
    assert run_tick(store, fpl, None, Clock(DAILY)) is True
    assert len(store.times("fpl", "bootstrap-static")) == 1
    assert fpl.live_calls == []
    assert fpl.summary_calls == []


def test_old_style_manifest_counts_as_none(tmp_path):
    store = RawStore(tmp_path)
    write_manifest(store, DAILY)  # pre-Phase-1a run: no season / through_event
    assert latest_complete_element_summary(store) == (0, 0)
    assert element_summaries_due(store, bootstrap(DEADLINE, (1,))) is True


def test_old_style_manifest_on_server_triggers_a_fresh_run(tmp_path):
    store = RawStore(tmp_path)
    write_manifest(store, DAILY - timedelta(days=30))
    fpl = FakeFpl(checked=(1, 2, 3, 4, 5))
    run_daily(store, fpl, None, Clock(DAILY), sleep=no_sleep)
    assert fpl.summary_calls == [1, 2]
    assert latest_complete_element_summary(store) == (2026, 5)


def test_incomplete_manifests_do_not_count(tmp_path):
    store = RawStore(tmp_path)
    write_manifest(store, DAILY, season=2026, through_event=3, written=[1], failed=[2])
    write_manifest(
        store, DAILY + timedelta(hours=1), season=2026, through_event=4, written=[1], failed=[]
    )
    assert latest_complete_element_summary(store) == (0, 0)


def test_latest_complete_ignores_stray_directories_and_runs_without_manifest(tmp_path):
    store = RawStore(tmp_path)
    good = write_manifest(store, DAILY, season=2026, through_event=2)
    # A run that died before writing its manifest (e.g. killed by a timeout).
    store.write("fpl", "element-summary", b"{}", DAILY + timedelta(days=1), name="1")
    # A non-timestamp directory holding a "better" manifest, and a stray file.
    root = tmp_path / "fpl" / "element-summary"
    better = {"expected": [], "written": [], "failed": [], "season": 2099, "through_event": 9}
    (root / "scratch").mkdir()
    (root / "scratch" / "_manifest.json.gz").write_bytes(gzip.compress(json.dumps(better).encode()))
    (root / "notes.txt").write_text("x")
    assert good.exists()
    assert latest_complete_element_summary(store) == (2026, 2)


def test_latest_complete_skips_unreadable_manifests(tmp_path):
    store = RawStore(tmp_path)
    write_manifest(store, DAILY, season=2026, through_event=2)
    broken = store.path_for("fpl", "element-summary", DAILY + timedelta(days=1), name="_manifest")
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not gzip")
    assert latest_complete_element_summary(store) == (2026, 2)


def test_latest_complete_compares_season_before_gameweek(tmp_path):
    store = RawStore(tmp_path)
    write_manifest(store, datetime(2027, 8, 20, 2, 30, tzinfo=UTC), season=2027, through_event=1)
    # Written later but describing an older season (e.g. a stale bootstrap): still older.
    write_manifest(store, datetime(2027, 8, 21, 2, 30, tzinfo=UTC), season=2026, through_event=38)
    assert latest_complete_element_summary(store) == (2027, 1)


def test_new_season_gw1_is_due_after_last_season_gw38(tmp_path):
    store = RawStore(tmp_path)
    write_manifest(store, datetime(2027, 5, 26, tzinfo=UTC), season=2026, through_event=38)
    new_season = datetime(2027, 8, 14, 10, 0, tzinfo=UTC)
    assert element_summaries_due(store, bootstrap(new_season, ())) is False
    assert element_summaries_due(store, bootstrap(new_season, (1,))) is True
    write_manifest(store, datetime(2027, 8, 20, tzinfo=UTC), season=2027, through_event=1)
    assert element_summaries_due(store, bootstrap(new_season, (1,))) is False


def test_summary_manifest_tolerates_bootstrap_without_events(tmp_path):
    class NoEvents(FakeFpl):
        def bootstrap_static(self):
            return json.dumps({"events": [], "elements": [{"id": 1}]}).encode()

    store = RawStore(tmp_path)
    assert backfill_element_summaries(store, NoEvents(), Clock(DAILY), sleep=no_sleep) == 1
    (run,) = summary_runs(tmp_path)
    manifest = store.read_json(run / "_manifest.json.gz")
    assert manifest["season"] is None
    assert manifest["through_event"] == 0
    assert latest_complete_element_summary(store) == (0, 0)
