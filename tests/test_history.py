from datetime import UTC, datetime, timedelta

import httpx
import pytest

from fplopt.adapters.vaastav import git_blob_sha
from fplopt.ingest.history import (
    backfill_football_data,
    backfill_vaastav,
    select_vaastav_paths,
    vaastav_name,
)
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


# --- vaastav -----------------------------------------------------------------------------

SELECTED = {
    "data/master_team_list.csv": b"season,team,team_name\n2016-17,1,Arsenal\n",
    "data/2016-17/gws/merged_gw.csv": b"name,round,total_points\nA,1,2\n",
    "data/2021-22/understat/Martin_Ødegaard_1.csv": b"goals,xG\n1,0.4\n",
}
UNSELECTED = {"data/2021-22/gws/xP3.csv": b"name,xP\nA,9.9\n"}


class FakeVaastav:
    def __init__(self, files=None, fail=()):
        self.files = dict(SELECTED | UNSELECTED) if files is None else files
        self.fail = set(fail)
        self.tree_calls = []
        self.file_calls = []

    def tree(self, commit):
        self.tree_calls.append(commit)
        return {path: git_blob_sha(data) for path, data in self.files.items()}

    def file(self, commit, path, blob_sha):
        self.file_calls.append((commit, path, blob_sha))
        if path in self.fail:
            raise RuntimeError(f"{path} down")
        return self.files[path]


def vaastav_run_dir(tmp_path):
    (run_dir,) = (tmp_path / "vaastav" / "data").iterdir()
    return run_dir


def test_select_vaastav_paths_keeps_wanted_files_for_mirrored_seasons():
    keep = [
        "data/master_team_list.csv",
        "data/2016-17/gws/merged_gw.csv",
        "data/2016-17/players_raw.csv",
        "data/2018-19/fixtures.csv",
        "data/2019-20/teams.csv",
        "data/2021-22/id_dict.csv",
        "data/2021-22/understat/Aaron_Cresswell_534.csv",
        "data/2021-22/understat/understat_team.csv",
        "data/2023-24/player_idlist.csv",
        "data/2024-25/cleaned_players.csv",
        "data/2025-26/gws/merged_gw.csv",
    ]
    drop = [
        "data/2021-22/gws/xP3.csv",
        "data/2021-22/gws/gw1.csv",
        "data/2021-22/players/Aaron_Cresswell_534/gw.csv",
        "data/2026-27/players_raw.csv",
        "data/2015-16/players_raw.csv",
        "data/cleaned_merged_seasons.csv",
        "data/2021-22/fbref/x.csv",
        "README.md",
        "data/2016-17/players_raw.csv.bak",
    ]
    assert select_vaastav_paths(drop + keep[::-1]) == sorted(keep)


def test_vaastav_name_strips_data_prefix_and_csv_suffix():
    assert vaastav_name("data/2016-17/gws/merged_gw.csv") == "2016-17/gws/merged_gw"
    assert vaastav_name("data/master_team_list.csv") == "master_team_list"
    with pytest.raises(ValueError):
        vaastav_name("README.md")


def test_backfill_vaastav_mirrors_selected_files_under_one_run(tmp_path):
    store = RawStore(tmp_path)
    fake = FakeVaastav()
    sleeps = []
    assert backfill_vaastav(store, fake, "abc123", Clock(NOW), sleep=sleeps.append) == 3
    assert fake.tree_calls == ["abc123"]
    assert sorted(path for _, path, _ in fake.file_calls) == sorted(SELECTED)
    assert all(commit == "abc123" for commit, _, _ in fake.file_calls)
    assert len(sleeps) == 3
    run_dir = vaastav_run_dir(tmp_path)
    for path, data in SELECTED.items():
        stored = run_dir / (path.removeprefix("data/").removesuffix(".csv") + ".csv.gz")
        assert RawStore.read_bytes(stored) == data
    assert (
        RawStore.read_bytes(run_dir / "2016-17" / "gws" / "merged_gw.csv.gz")
        == (SELECTED["data/2016-17/gws/merged_gw.csv"])
    )
    manifest = RawStore.read_json(run_dir / "_manifest.json.gz")
    assert manifest["commit"] == "abc123"
    assert manifest["expected"] == sorted(SELECTED)
    assert manifest["written"] == sorted(SELECTED)
    assert manifest["failed"] == []
    assert {"run_at", "finished_at"} <= manifest.keys()


def test_backfill_vaastav_continues_past_a_failing_file_then_raises(tmp_path):
    store = RawStore(tmp_path)
    fake = FakeVaastav(fail={"data/2016-17/gws/merged_gw.csv"})
    with pytest.raises(RuntimeError, match="1 of 3"):
        backfill_vaastav(store, fake, "abc123", Clock(NOW), sleep=no_sleep)
    run_dir = vaastav_run_dir(tmp_path)
    assert not (run_dir / "2016-17").exists()
    assert (run_dir / "master_team_list.csv.gz").exists()
    manifest = RawStore.read_json(run_dir / "_manifest.json.gz")
    assert manifest["failed"] == ["data/2016-17/gws/merged_gw.csv"]
    assert len(manifest["written"]) == 2


def test_backfill_vaastav_aborts_after_consecutive_failures_and_keeps_manifest(tmp_path):
    store = RawStore(tmp_path)
    fake = FakeVaastav(fail=set(SELECTED))
    with pytest.raises(RuntimeError, match="2 consecutive"):
        backfill_vaastav(
            store, fake, "abc123", Clock(NOW), sleep=no_sleep, max_consecutive_failures=2
        )
    assert len(fake.file_calls) == 2
    manifest = RawStore.read_json(vaastav_run_dir(tmp_path) / "_manifest.json.gz")
    assert len(manifest["failed"]) == 2
    assert manifest["written"] == []


def test_backfill_vaastav_records_a_path_the_store_cannot_hold_as_failed(tmp_path):
    store = RawStore(tmp_path)
    files = dict(SELECTED) | {"data/2021-22/understat/Bad:Name_1.csv": b"x\n"}
    fake = FakeVaastav(files=files)
    with pytest.raises(RuntimeError, match="1 of 4"):
        backfill_vaastav(store, fake, "abc123", Clock(NOW), sleep=no_sleep)
    assert len(fake.file_calls) == 3  # the unsafe path is not even downloaded
    manifest = RawStore.read_json(vaastav_run_dir(tmp_path) / "_manifest.json.gz")
    assert manifest["failed"] == ["data/2021-22/understat/Bad:Name_1.csv"]
    assert len(manifest["written"]) == 3


def test_backfill_vaastav_refuses_an_empty_selection(tmp_path):
    fake = FakeVaastav(files=UNSELECTED)
    with pytest.raises(RuntimeError, match="no vaastav files"):
        backfill_vaastav(RawStore(tmp_path), fake, "abc123", Clock(NOW), sleep=no_sleep)
    assert not (tmp_path / "vaastav").exists()
