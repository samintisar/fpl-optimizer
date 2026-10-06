from datetime import UTC, datetime, timedelta

import httpx
import pytest

from fplopt.ingest.history import backfill_football_data
from fplopt.ingest.jobs import SnapshotError
from fplopt.ingest.raw_store import RawStore

NOW = datetime(2026, 10, 6, 3, 0, tzinfo=UTC)


class Clock:
    """Advances one second per call so consecutive writes get distinct timestamps."""

    def __init__(self, start):
        self.t = start

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


def no_sleep(_seconds):
    pass


def csv_for(start_year):
    return f"Div,Date,HomeTeam\r\nE0,{start_year},X\r\n".encode()


class FakeFootballData:
    def __init__(self, fail=(), not_found=()):
        self.fail = set(fail)
        self.not_found = set(not_found)
        self.seasons = []

    def epl_season(self, start_year):
        self.seasons.append(start_year)
        if start_year in self.fail:
            raise RuntimeError(f"season {start_year} down")
        if start_year in self.not_found:
            request = httpx.Request("GET", "https://x")
            raise httpx.HTTPStatusError(
                "404", request=request, response=httpx.Response(404, request=request)
            )
        return csv_for(start_year)


def test_backfill_football_data_fetches_every_season_in_order(tmp_path):
    store = RawStore(tmp_path)
    fd = FakeFootballData()
    sleeps = []
    assert backfill_football_data(store, fd, Clock(NOW), sleep=sleeps.append) == 11
    assert fd.seasons == list(range(2016, 2027))
    assert len(sleeps) == 10  # between requests, not after the last
    for year, code in ((2016, "1617"), (2020, "2021"), (2026, "2627")):
        path = store.latest("football-data", f"E0/{code}", suffix=".csv.gz")
        assert RawStore.read_bytes(path) == csv_for(year)


def test_backfill_football_data_continues_past_a_failing_season(tmp_path):
    store = RawStore(tmp_path)
    fd = FakeFootballData(fail={2018})
    with pytest.raises(SnapshotError, match="1819") as info:
        backfill_football_data(store, fd, Clock(NOW), sleep=no_sleep)
    assert "\n" not in str(info.value)
    assert fd.seasons == list(range(2016, 2027))
    assert store.times("football-data", "E0/1819", suffix=".csv.gz") == []
    assert len(store.times("football-data", "E0/1920", suffix=".csv.gz")) == 1


def test_backfill_football_data_tolerates_unpublished_new_season_in_august(tmp_path):
    store = RawStore(tmp_path)
    fd = FakeFootballData(not_found={2026})
    august = datetime(2026, 8, 1, tzinfo=UTC)
    assert backfill_football_data(store, fd, Clock(august), sleep=no_sleep) == 10
    assert fd.seasons == list(range(2016, 2027))
