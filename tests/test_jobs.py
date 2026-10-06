import json
from datetime import UTC, datetime, timedelta

import pytest

from fplopt.ingest.jobs import backfill_element_summaries, run_daily, run_tick
from fplopt.ingest.raw_store import RawStore

DEADLINE = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)


class MultiLineFailingOdds:
    def epl_odds(self):
        raise RuntimeError("odds down\ndetails on a second line")


class FakeFpl:
    def __init__(self, fail_ids=(), element_ids=(1, 2), fail_fixtures=False, clock=None):
        self.fail_ids = set(fail_ids)
        self.element_ids = list(element_ids)
        self.fail_fixtures = fail_fixtures
        self.clock = clock
        self.fetched = []
        self.summary_calls = []

    def bootstrap_static(self):
        if self.clock:
            self.fetched.append(("bootstrap", self.clock()))
        return json.dumps(
            {
                "events": [{"deadline_time": "2026-10-10T10:00:00Z"}],
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
