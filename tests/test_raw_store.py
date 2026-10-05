import gzip
from datetime import UTC, datetime, timedelta, timezone

import pytest

from fplopt.ingest.raw_store import RawStore, format_ts, parse_ts

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
