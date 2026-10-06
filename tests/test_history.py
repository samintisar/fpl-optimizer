import io
import json
import lzma
import tarfile
import zlib
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from fplopt.adapters.fplcache import FplcacheClient
from fplopt.adapters.http import make_client
from fplopt.adapters.vaastav import git_blob_sha
from fplopt.ingest.history import (
    backfill_football_data,
    backfill_fplcache,
    backfill_vaastav,
    fplcache_snapshot_time,
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


# --- fplcache ----------------------------------------------------------------------------

FC_SHA = "0123456789abcdef0123456789abcdef01234567"
FC_PREFIX = f"fplcache-{FC_SHA}"
SNAP_A = lzma.compress(json.dumps({"events": [{"id": 1}], "n": "a"}).encode())
SNAP_B = lzma.compress(json.dumps({"events": [{"id": 1}], "n": "b"}).encode())
NAME_A = f"{FC_PREFIX}/cache/2021/4/18/1641.json.xz"
NAME_B = f"{FC_PREFIX}/cache/2021/12/3/0905.json.xz"
TS_A = datetime(2021, 4, 18, 16, 41, tzinfo=UTC)
TS_B = datetime(2021, 12, 3, 9, 5, tzinfo=UTC)
FC_MEMBERS = {
    FC_PREFIX: None,
    f"{FC_PREFIX}/README.md": b"# fplcache",
    f"{FC_PREFIX}/cache": None,
    NAME_A: SNAP_A,
    f"{FC_PREFIX}/cache/2021/4/18/notes.txt": b"notes",
    NAME_B: SNAP_B,
}


def make_tarball(members):
    """An in-memory .tar.gz; a None value makes a directory member."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def truncated_at_member(members, index, extra=0):
    """A .tar.gz body that decompresses to exactly the tar bytes before member `index`'s
    header plus `extra` bytes (a full flush at the cut point), i.e. a download cut there."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
        cut = tar.getmembers()[index].offset + extra
    compressor = zlib.compressobj(wbits=31)
    return compressor.compress(raw[:cut]) + compressor.flush(zlib.Z_FULL_FLUSH)


def fplcache_client(members, broken_after=None, body=None):
    """FplcacheClient over MockTransport serving the commits API and a streamed tarball
    (`body` overrides the .tar.gz built from `members`)."""
    body = make_tarball(members) if body is None else body

    def chunks():
        for start in range(0, len(body), 4096):
            if broken_after is not None and start >= broken_after:
                raise httpx.ReadError("connection reset")
            yield body[start : start + 4096]

    def handler(request):
        if request.url.path == "/repos/Randdalf/fplcache/commits/main":
            return httpx.Response(200, json={"sha": FC_SHA})
        if request.url.path == f"/Randdalf/fplcache/tar.gz/{FC_SHA}":
            return httpx.Response(200, content=chunks())
        return httpx.Response(404)

    return FplcacheClient(make_client(httpx.MockTransport(handler)))


def snapshot_path(store, ts):
    return store.path_for("fplcache", "bootstrap-static", ts, suffix=".json.xz")


def latest_fplcache_manifest(store):
    return RawStore.read_json(store.latest("fplcache", "runs"))


def test_fplcache_snapshot_time_parses_unpadded_and_padded_dates():
    assert fplcache_snapshot_time(NAME_A) == TS_A
    assert fplcache_snapshot_time("x/cache/2022/08/05/0000.json.xz") == datetime(
        2022, 8, 5, tzinfo=UTC
    )


def test_fplcache_snapshot_time_ignores_other_files():
    assert fplcache_snapshot_time(f"{FC_PREFIX}/README.md") is None
    assert fplcache_snapshot_time(f"{FC_PREFIX}/cache/2021/4/18/notes.txt") is None
    assert fplcache_snapshot_time("cache/2021/4/18/1641.json.xz") is None


def test_fplcache_snapshot_time_rejects_an_impossible_date():
    with pytest.raises(ValueError):
        fplcache_snapshot_time(f"{FC_PREFIX}/cache/2021/13/1/0000.json.xz")


def test_backfill_fplcache_mirrors_snapshots_byte_for_byte(tmp_path):
    store = RawStore(tmp_path)
    assert backfill_fplcache(store, fplcache_client(FC_MEMBERS), Clock(NOW)) == 2
    assert snapshot_path(store, TS_A).read_bytes() == SNAP_A
    assert snapshot_path(store, TS_B).read_bytes() == SNAP_B
    assert store.times("fplcache", "bootstrap-static", suffix=".json.xz") == [TS_A, TS_B]
    manifest = latest_fplcache_manifest(store)
    assert manifest["commit"] == FC_SHA
    assert (manifest["written"], manifest["skipped"], manifest["failed"]) == (2, 0, [])
    assert manifest["error"] is None


def test_backfill_fplcache_second_run_skips_identical_files(tmp_path):
    store = RawStore(tmp_path)
    backfill_fplcache(store, fplcache_client(FC_MEMBERS), Clock(NOW))
    later = Clock(NOW + timedelta(days=1))
    assert backfill_fplcache(store, fplcache_client(FC_MEMBERS), later) == 0
    manifest = latest_fplcache_manifest(store)
    assert (manifest["written"], manifest["skipped"], manifest["failed"]) == (0, 2, [])
    assert len(store.times("fplcache", "runs")) == 2


def test_backfill_fplcache_records_bad_members_and_raises(tmp_path):
    store = RawStore(tmp_path)
    corrupt = f"{FC_PREFIX}/cache/2021/5/1/0000.json.xz"
    not_json = f"{FC_PREFIX}/cache/2021/5/2/0000.json.xz"
    members = FC_MEMBERS | {corrupt: b"not xz at all", not_json: lzma.compress(b"<html>")}
    with pytest.raises(RuntimeError, match="2 of 4 fplcache snapshots failed"):
        backfill_fplcache(store, fplcache_client(members), Clock(NOW))
    assert snapshot_path(store, TS_A).read_bytes() == SNAP_A
    assert snapshot_path(store, TS_B).read_bytes() == SNAP_B
    assert not snapshot_path(store, datetime(2021, 5, 1, tzinfo=UTC)).exists()
    manifest = latest_fplcache_manifest(store)
    assert manifest["commit"] == FC_SHA
    assert manifest["written"] == 2
    assert [entry["member"] for entry in manifest["failed"]] == [corrupt, not_json]
    assert all(entry["reason"] for entry in manifest["failed"])


def test_backfill_fplcache_fails_when_archived_copy_differs(tmp_path):
    store = RawStore(tmp_path)
    store.write_bytes("fplcache", "bootstrap-static", b"older bytes", TS_A, suffix=".json.xz")
    with pytest.raises(RuntimeError, match="1 of 2 fplcache snapshots failed"):
        backfill_fplcache(store, fplcache_client(FC_MEMBERS), Clock(NOW))
    assert snapshot_path(store, TS_A).read_bytes() == b"older bytes"
    assert snapshot_path(store, TS_B).read_bytes() == SNAP_B
    manifest = latest_fplcache_manifest(store)
    assert [entry["member"] for entry in manifest["failed"]] == [NAME_A]


def test_backfill_fplcache_writes_manifest_when_the_download_breaks(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(httpx.ReadError):
        backfill_fplcache(store, fplcache_client(FC_MEMBERS, broken_after=0), Clock(NOW))
    manifest = latest_fplcache_manifest(store)
    assert manifest["commit"] == FC_SHA
    assert "ReadError" in manifest["error"]


def test_backfill_fplcache_fails_on_a_tarball_truncated_at_a_member_boundary(tmp_path):
    store = RawStore(tmp_path)
    body = truncated_at_member(FC_MEMBERS, list(FC_MEMBERS).index(NAME_B))
    with pytest.raises(EOFError, match="truncated"):
        backfill_fplcache(store, fplcache_client(FC_MEMBERS, body=body), Clock(NOW))
    assert not snapshot_path(store, TS_B).exists()
    manifest = latest_fplcache_manifest(store)
    assert manifest["commit"] == FC_SHA
    assert "truncated" in manifest["error"]


def test_backfill_fplcache_records_a_download_cut_inside_a_member(tmp_path):
    store = RawStore(tmp_path)
    body = truncated_at_member(FC_MEMBERS, list(FC_MEMBERS).index(NAME_B), extra=512 + 10)
    with pytest.raises(EOFError, match="truncated"):
        backfill_fplcache(store, fplcache_client(FC_MEMBERS, body=body), Clock(NOW))
    assert not snapshot_path(store, TS_B).exists()
    assert "truncated" in latest_fplcache_manifest(store)["error"]


def test_unxz_bounded_refuses_oversized_output():
    from fplopt.ingest.history import _unxz_bounded

    blob = lzma.compress(b"0" * 10_000)
    assert _unxz_bounded(blob, 10_000) == b"0" * 10_000
    with pytest.raises(ValueError, match="more than 9999"):
        _unxz_bounded(blob, 9_999)
    with pytest.raises(ValueError, match="truncated"):
        _unxz_bounded(blob[:-8], 10_000)
