import gzip
import lzma
from datetime import UTC, datetime, timedelta, timezone

import pytest

from fplopt.ingest.raw_store import RawStore, format_ts, gzip_bytes, parse_ts

T0 = datetime(2026, 10, 5, 3, 0, 0, tzinfo=UTC)


def test_write_creates_gzipped_file_with_timestamp_path(tmp_path):
    store = RawStore(tmp_path)
    path = store.write("fpl", "bootstrap-static", b'{"a": 1}', T0)
    assert path == tmp_path / "fpl" / "bootstrap-static" / "2026-10-05T030000Z.json.gz"
    assert gzip.decompress(path.read_bytes()) == b'{"a": 1}'


def test_write_never_overwrites(tmp_path):
    store = RawStore(tmp_path)
    store.write("fpl", "fixtures", b"[]", T0)
    with pytest.raises(FileExistsError):
        store.write("fpl", "fixtures", b"[1]", T0)
    assert store.read_json(store.latest("fpl", "fixtures")) == []


def test_write_rejects_non_json(tmp_path):
    with pytest.raises(ValueError):
        RawStore(tmp_path).write("fpl", "fixtures", b"<html>The game is being updated.</html>", T0)
    assert not (tmp_path / "fpl").exists()


def test_write_with_name_groups_files_under_run_timestamp(tmp_path):
    path = RawStore(tmp_path).write("fpl", "element-summary", b"{}", T0, name="17")
    assert path == tmp_path / "fpl" / "element-summary" / "2026-10-05T030000Z" / "17.json.gz"


def test_times_and_latest(tmp_path):
    store = RawStore(tmp_path)
    assert store.times("fpl", "bootstrap-static") == []
    assert store.latest("fpl", "bootstrap-static") is None
    later = T0 + timedelta(hours=2)
    store.write("fpl", "bootstrap-static", b'{"n": 2}', later)
    store.write("fpl", "bootstrap-static", b'{"n": 1}', T0)
    assert store.times("fpl", "bootstrap-static") == [T0, later]
    assert store.read_json(store.latest("fpl", "bootstrap-static")) == {"n": 2}


def test_timestamps_are_normalised_to_utc():
    plus_one = timezone(timedelta(hours=1))
    assert format_ts(datetime(2026, 10, 5, 4, 0, 0, tzinfo=plus_one)) == "2026-10-05T030000Z"
    assert parse_ts("2026-10-05T030000Z") == T0


def test_naive_timestamp_rejected(tmp_path):
    with pytest.raises(ValueError):
        RawStore(tmp_path).write("fpl", "fixtures", b"[]", datetime(2026, 10, 5))


def test_unrecognised_files_are_ignored(tmp_path):
    store = RawStore(tmp_path)
    store.write("fpl", "fixtures", b"[]", T0)
    (tmp_path / "fpl" / "fixtures" / "backup.json.gz").write_bytes(b"")
    assert store.times("fpl", "fixtures") == [T0]
    assert store.latest("fpl", "fixtures").name == "2026-10-05T030000Z.json.gz"


def test_failed_write_leaves_no_temp_file(tmp_path):
    store = RawStore(tmp_path)
    store.write("fpl", "fixtures", b"[]", T0)
    with pytest.raises(FileExistsError):
        store.write("fpl", "fixtures", b"[1]", T0)
    names = [p.name for p in (tmp_path / "fpl" / "fixtures").iterdir()]
    assert names == ["2026-10-05T030000Z.json.gz"]


CSV = b"Div,Date,HomeTeam\nE0,13/08/2016,Burnley\n"


def test_write_bytes_stores_bytes_verbatim_and_never_overwrites(tmp_path):
    store = RawStore(tmp_path)
    data = gzip_bytes(CSV)
    path = store.write_bytes("football-data", "E0/1617", data, T0, suffix=".csv.gz")
    assert path == tmp_path / "football-data" / "E0" / "1617" / "2026-10-05T030000Z.csv.gz"
    assert path.read_bytes() == data
    with pytest.raises(FileExistsError):
        store.write_bytes("football-data", "E0/1617", b"other", T0, suffix=".csv.gz")
    assert path.read_bytes() == data
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_write_bytes_with_nested_name(tmp_path):
    path = RawStore(tmp_path).write_bytes(
        "vaastav", "data", gzip_bytes(CSV), T0, suffix=".csv.gz", name="2016-17/gws/merged_gw"
    )
    run_dir = tmp_path / "vaastav" / "data" / "2026-10-05T030000Z"
    assert path == run_dir / "2016-17" / "gws" / "merged_gw.csv.gz"
    assert RawStore.read_bytes(path) == CSV


def test_path_for_matches_write_bytes(tmp_path):
    store = RawStore(tmp_path)
    expected = store.path_for("fplcache", "bootstrap-static", T0, suffix=".json.xz")
    assert not expected.exists()
    written = store.write_bytes(
        "fplcache", "bootstrap-static", lzma.compress(b"{}"), T0, suffix=".json.xz"
    )
    assert written == expected
    assert store.path_for("fpl", "element-summary", T0, name="17") == (
        tmp_path / "fpl" / "element-summary" / "2026-10-05T030000Z" / "17.json.gz"
    )


@pytest.mark.parametrize(
    ("source", "endpoint", "name"),
    [
        ("fpl", "../x", None),
        ("fpl", "x", "a/../b"),
        ("fpl", "x", "/abs"),
        ("fpl", "x", "a\\b"),
        ("fpl", "x", "a/"),
        ("fpl", "x", "C:evil"),
        ("fpl", "/abs", None),
        ("fpl", "a//b", None),
        ("fpl", "./x", None),
        ("", "x", None),
        ("..", "x", None),
        ("a/b", "x", None),
    ],
)
def test_unsafe_path_parts_rejected(tmp_path, source, endpoint, name):
    store = RawStore(tmp_path)
    with pytest.raises(ValueError):
        store.write_bytes(source, endpoint, gzip_bytes(CSV), T0, suffix=".csv.gz", name=name)
    with pytest.raises(ValueError):
        store.path_for(source, endpoint, T0, name=name)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("suffix", [".csv", "csv.gz", ".json", ".gz/x", "/../x.gz", ""])
def test_uncompressed_or_odd_suffix_rejected(tmp_path, suffix):
    with pytest.raises(ValueError):
        RawStore(tmp_path).write_bytes("football-data", "E0/1617", CSV, T0, suffix=suffix)
    assert list(tmp_path.iterdir()) == []


def test_listing_is_suffix_aware(tmp_path):
    store = RawStore(tmp_path)
    later = T0 + timedelta(hours=6)
    store.write_bytes("fplcache", "bootstrap-static", lzma.compress(b"{}"), T0, suffix=".json.xz")
    store.write("fplcache", "bootstrap-static", b"{}", later)
    assert store.times("fplcache", "bootstrap-static", suffix=".json.xz") == [T0]
    assert store.times("fplcache", "bootstrap-static") == [later]
    assert store.latest("fplcache", "bootstrap-static", suffix=".json.xz").name == (
        "2026-10-05T030000Z.json.xz"
    )
    assert store.entries("fplcache", "bootstrap-static", suffix=".json.xz") == [
        (T0, tmp_path / "fplcache" / "bootstrap-static" / "2026-10-05T030000Z.json.xz")
    ]
    assert store.entries("fplcache", "missing") == []


def test_read_json_handles_xz_and_gz(tmp_path):
    store = RawStore(tmp_path)
    xz_path = store.write_bytes(
        "fplcache", "bootstrap-static", lzma.compress(b'{"a": 1}'), T0, suffix=".json.xz"
    )
    gz_path = store.write("fpl", "bootstrap-static", b'{"b": 2}', T0)
    assert RawStore.read_json(xz_path) == {"a": 1}
    assert RawStore.read_json(gz_path) == {"b": 2}


def test_read_bytes_round_trips_csv(tmp_path):
    path = RawStore(tmp_path).write_bytes(
        "football-data", "E0/1617", gzip_bytes(CSV), T0, suffix=".csv.gz"
    )
    assert RawStore.read_bytes(path) == CSV


def test_read_bytes_rejects_unknown_compression(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(CSV)
    with pytest.raises(ValueError):
        RawStore.read_bytes(path)


def test_gzip_bytes_is_deterministic():
    assert gzip_bytes(CSV) == gzip_bytes(CSV)
    assert gzip.decompress(gzip_bytes(CSV)) == CSV
