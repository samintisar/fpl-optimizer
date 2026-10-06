import datetime as dt
import json
import logging
import lzma
from datetime import UTC, datetime

import pandas as pd
import pyarrow.parquet as pq
import pytest

from fplopt.build import BUILDERS, ORDER, snapshots
from fplopt.build.common import BuildContext, TableValidationError
from fplopt.build.snapshots import COLUMNS, build_player_snapshot
from fplopt.ingest.raw_store import RawStore

OLD = datetime(2022, 7, 1, 12, 0, tzinfo=UTC)  # still the 2021/22 game (calendar says 2022)
NEW = datetime(2026, 10, 5, 3, 0, tzinfo=UTC)


def element(id_, code, team, element_type=3, **extra):
    base = {
        "id": id_,
        "code": code,
        "team": team,
        "element_type": element_type,
        "now_cost": 55,
        "status": "a",
        "chance_of_playing_next_round": None,
        "chance_of_playing_this_round": None,
        "news": "",
        "news_added": None,
        "selected_by_percent": "12.3",
        "ep_next": "4.5",
        "ep_this": "3.0",
        "form": "2.5",
        "penalties_order": None,
        "corners_and_indirect_freekicks_order": None,
        "direct_freekicks_order": None,
        "transfers_in_event": 10,
        "transfers_out_event": 4,
        "cost_change_event": 0,
    }
    return base | extra


def bootstrap(first_deadline, teams, elements):
    return {
        "events": [
            {"id": 2, "deadline_time": "2099-01-01T00:00:00Z"},
            {"id": 1, "deadline_time": first_deadline},
        ],
        "teams": [{"id": i, "code": code} for i, code in teams],
        "elements": elements,
    }


OLD_SNAPSHOT = bootstrap(
    "2021-08-13T17:30:00Z",
    teams=[(1, 3), (2, 7)],
    elements=[
        element(
            2,
            200,
            2,
            status="d",
            chance_of_playing_next_round=75,
            chance_of_playing_this_round=50,
            news="Knock - 75% chance of playing",
            news_added="2022-05-20T10:15:30.123456Z",
            penalties_order=1,
        ),
        element(1, 100, 1, element_type=1, selected_by_percent="0.0", form="0.0"),
    ],
)
NEW_SNAPSHOT = bootstrap(
    "2026-08-14T17:30:00Z",
    teams=[(1, 7), (2, 3)],  # ids reshuffled between seasons; codes are stable
    elements=[
        element(1, 300, 2, team_join_date="2024-07-04", ep_next=None, direct_freekicks_order=2),
        element(5, 999, 1, element_type=5),  # assistant manager -> dropped
        element(4, 100, 1, element_type=1, team_join_date=None),
    ],
)


def write_fplcache(store, ts, payload):
    data = lzma.compress(json.dumps(payload).encode())
    store.write_bytes("fplcache", "bootstrap-static", data, ts, suffix=".json.xz")


def write_own(store, ts, payload):
    store.write("fpl", "bootstrap-static", json.dumps(payload).encode(), ts)


@pytest.fixture
def ctx(tmp_path):
    store = RawStore(tmp_path / "raw")
    write_fplcache(store, OLD, OLD_SNAPSHOT)
    write_fplcache(store, NEW, NEW_SNAPSHOT)
    write_own(store, NEW, NEW_SNAPSHOT)  # same moment, other archive: both kept
    return BuildContext(store, tmp_path / "data")


def run(ctx, **kwargs):
    kwargs.setdefault("jobs", 1)
    kwargs.setdefault("first_season", None)  # no season-coverage check
    summary = build_player_snapshot(ctx, **kwargs)
    return summary, pd.read_parquet(ctx.table_path("player_snapshot"))


def test_columns_dtypes_and_order(ctx):
    _, out = run(ctx)
    assert list(out.columns) == list(COLUMNS)
    dtypes = out.dtypes.astype(str).to_dict()
    assert dtypes["snapshot_at"] == "datetime64[us, UTC]"
    assert dtypes["news_added"] == "datetime64[us, UTC]"
    assert dtypes["available_at"] == "datetime64[us, UTC]"
    assert dtypes["season"] == "int64"
    assert dtypes["player_key"] == "int64"
    assert dtypes["chance_of_playing_next_round"] == "Int64"
    assert dtypes["penalties_order"] == "Int64"
    assert dtypes["selected_by_percent"] == "Float64"
    assert dtypes["team_join_date"] == "date32[day][pyarrow]"
    assert dtypes["source"] == "str"
    assert (out["event_time"] == out["snapshot_at"]).all()
    assert (out["available_at"] == out["snapshot_at"]).all()


def test_rows_sorted_by_key_with_both_sources_and_managers_dropped(ctx):
    summary, out = run(ctx)
    keys = list(zip(out["snapshot_at"], out["source"], out["element_id"], strict=True))
    assert keys == [
        (pd.Timestamp(OLD), "fplcache", 1),
        (pd.Timestamp(OLD), "fplcache", 2),
        (pd.Timestamp(NEW), "fpl", 1),
        (pd.Timestamp(NEW), "fpl", 4),
        (pd.Timestamp(NEW), "fplcache", 1),
        (pd.Timestamp(NEW), "fplcache", 4),
    ]
    assert 999 not in set(out["player_key"])
    # build() logs rows and seasons from the returned frame
    assert len(summary) == len(out)
    assert sorted(summary["season"].unique()) == [2021, 2026]


def test_season_from_bootstrap_not_timestamp_and_team_codes(ctx):
    _, out = run(ctx)
    old = out[out["snapshot_at"] == pd.Timestamp(OLD)].set_index("element_id")
    assert set(old["season"]) == {2021}
    assert old.loc[1, "team_key"] == 3 and old.loc[2, "team_key"] == 7
    new = out[(out["snapshot_at"] == pd.Timestamp(NEW)) & (out["source"] == "fpl")]
    new = new.set_index("element_id")
    assert set(new["season"]) == {2026}
    assert new.loc[1, "team_key"] == 3 and new.loc[4, "team_key"] == 7
    assert new.loc[1, "player_key"] == 300 and new.loc[4, "player_key"] == 100


def test_value_conversions_and_missing_keys_are_null(ctx):
    _, out = run(ctx)
    old = out[out["snapshot_at"] == pd.Timestamp(OLD)].set_index("element_id")
    assert old.loc[2, "selected_by_percent"] == 12.3
    assert old.loc[2, "ep_next"] == 4.5 and old.loc[2, "form"] == 2.5
    assert old.loc[1, "form"] == 0.0
    assert old.loc[2, "chance_of_playing_next_round"] == 75
    assert old.loc[2, "chance_of_playing_this_round"] == 50
    assert pd.isna(old.loc[1, "chance_of_playing_next_round"])
    assert old.loc[2, "news_added"] == pd.Timestamp("2022-05-20T10:15:30.123456Z")
    assert pd.isna(old.loc[1, "news_added"])
    assert old.loc[2, "penalties_order"] == 1 and pd.isna(old.loc[1, "penalties_order"])
    assert old["team_join_date"].isna().all()  # key absent from old snapshots
    new = out[(out["snapshot_at"] == pd.Timestamp(NEW)) & (out["source"] == "fpl")]
    new = new.set_index("element_id")
    assert new.loc[1, "team_join_date"] == dt.date(2024, 7, 4)
    assert pd.isna(new.loc[4, "team_join_date"])
    assert pd.isna(new.loc[1, "ep_next"]) and new.loc[1, "ep_this"] == 3.0
    assert new.loc[1, "direct_freekicks_order"] == 2


def test_rebuild_is_byte_identical_across_chunking_and_jobs(ctx, monkeypatch):
    monkeypatch.setattr(snapshots, "IN_FLIGHT_PER_JOB", 1)  # 2 in flight < 3 snapshots: refills
    run(ctx)
    path = ctx.table_path("player_snapshot")
    first = path.read_bytes()
    run(ctx)
    assert path.read_bytes() == first
    # One row group per season when chunks are large; content equal whatever the chunk size.
    assert pq.ParquetFile(path).num_row_groups == 2
    run(ctx, chunk_snapshots=1)
    assert pq.ParquetFile(path).num_row_groups == 3
    small_chunks = path.read_bytes()
    run(ctx, chunk_snapshots=1, jobs=2)
    assert path.read_bytes() == small_chunks
    assert not list(ctx.data_dir.glob("*.tmp"))


def test_invalid_chunk_fails_and_writes_nothing(ctx):
    bad = bootstrap("2026-08-14T17:30:00Z", [(1, 3)], [element(1, 1, 1, now_cost=-5)])
    write_own(ctx.store, datetime(2026, 10, 6, tzinfo=UTC), bad)
    with pytest.raises(TableValidationError, match="now_cost"):
        run(ctx)
    assert not ctx.table_path("player_snapshot").exists()
    assert not list(ctx.data_dir.glob("*.tmp"))


def test_missing_required_key_fails_naming_the_file(ctx):
    broken = element(1, 1, 1)
    del broken["now_cost"]
    payload = bootstrap("2026-08-14T17:30:00Z", [(1, 3)], [broken])
    write_own(ctx.store, datetime(2026, 10, 6, tzinfo=UTC), payload)
    with pytest.raises(ValueError, match=r"fpl bootstrap 2026-10-06T000000Z\.json\.gz.*now_cost"):
        run(ctx)
    assert not ctx.table_path("player_snapshot").exists()


def test_missing_seasons_fail(ctx):
    with pytest.raises(ValueError, match=r"missing season\(s\) 2022, 2023, 2024, 2025"):
        run(ctx, first_season=2021)  # 2021 and 2026 only
    with pytest.raises(ValueError, match="missing season"):
        run(ctx, first_season=2020)
    assert not ctx.table_path("player_snapshot").exists()


def test_player_dim_check_reports_missing_keys(ctx, caplog):
    run(ctx)  # no player_dim yet: check skipped
    pd.DataFrame({"player_key": [100, 200]}).to_parquet(ctx.table_path("player_dim"))
    with caplog.at_level(logging.WARNING, logger="fplopt.build.snapshots"):
        run(ctx)
    assert "1 player_key(s) not in player_dim" in caplog.text
    assert "300" in caplog.text


def test_registered_after_team_dim():
    assert "player_snapshot" in BUILDERS
    assert BUILDERS["player_snapshot"].schema is None
    assert ORDER.index("player_snapshot") > ORDER.index("team_dim")
