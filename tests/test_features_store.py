from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from fplopt.build.common import schedule_published_at
from fplopt.features.store import AsOfView, DataStore

UTC_US = pd.DatetimeTZDtype("us", "UTC")


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC").as_unit("us")


T1, T2, T3 = ts("2026-08-01 10:00"), ts("2026-08-02 10:00"), ts("2026-08-03 10:00")
DEADLINE = ts("2026-08-02 12:00")


def player_snapshot() -> pd.DataFrame:
    rows = [
        # snapshot_at, source, player_key, now_cost
        (T3, "fpl", 1, 99),  # after the deadline
        (T1, "fplcache", 1, 50),
        (T2, "fpl", 1, 52),  # ties fplcache at T2: 'fpl' wins
        (T2, "fplcache", 1, 51),
        (T1, "fplcache", 2, 60),  # stale but still the newest for player 2
        (T3, "fplcache", 3, 70),  # only after the deadline
    ]
    df = pd.DataFrame(rows, columns=["snapshot_at", "source", "player_key", "now_cost"])
    df["snapshot_at"] = df["snapshot_at"].astype(UTC_US)
    return df.assign(season=2026, event_time=df["snapshot_at"], available_at=df["snapshot_at"])


def player_match() -> pd.DataFrame:
    available = [DEADLINE - pd.Timedelta(microseconds=1), DEADLINE, DEADLINE + pd.Timedelta(1, "h")]
    df = pd.DataFrame(
        {
            "player_key": [1, 2, 3],
            "fixture_key": [10, 20, 30],
            "minutes": [90, 45, 0],
            "available_at": pd.Series(available).astype(UTC_US),
        }
    )
    return df.assign(event_time=df["available_at"])


def gameweek() -> pd.DataFrame:
    published = schedule_published_at(2026)
    df = pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "gw": [1, 2, 3],
            "gw_index": [1, 2, 3],
            "deadline_time": pd.Series([T1, DEADLINE, T3 + pd.Timedelta(days=7)]).astype(UTC_US),
        }
    )
    return df.assign(event_time=df["deadline_time"], available_at=published)


def schedule() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "fixture_key": [101, 102, 201],
            "season": [2026, 2026, 2025],
            "gw": pd.array([1, 2, 1], dtype="Int64"),
            "gw_index": pd.array([1, 2, 1], dtype="Int64"),
            "kickoff_time": pd.Series(
                [T1 + pd.Timedelta(hours=3), DEADLINE + pd.Timedelta(hours=3), ts("2025-08-16")]
            ).astype(UTC_US),
            "home_team_key": [1, 3, 1],
            "away_team_key": [2, 4, 2],
            "schedule_source": "final",
        }
    )
    published = df["season"].map(schedule_published_at).astype(UTC_US)
    return df.assign(event_time=df["kickoff_time"], available_at=published)


SNAP_AT = ts("2026-08-02 06:00")


def fixture_snapshot() -> pd.DataFrame:
    """Two snapshots of 2026: at T1 (as final) and at SNAP_AT (fixture 102 moved to GW 3)."""
    final = schedule().query("season == 2026")
    frames = []
    for at, gw102 in ((T1, 2), (SNAP_AT, 3)):
        snap = final[["season", "fixture_key", "kickoff_time", "home_team_key", "away_team_key"]]
        snap = snap.assign(gw=pd.array([1, gw102], dtype="Int64"), snapshot_at=at)
        frames.append(snap)
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_at"] = df["snapshot_at"].astype(UTC_US)
    return df.assign(event_time=df["snapshot_at"], available_at=df["snapshot_at"])


def tables() -> dict[str, pd.DataFrame]:
    return {
        "player_snapshot": player_snapshot(),
        "player_match": player_match(),
        "gameweek": gameweek(),
        "schedule": schedule(),
        "fixture_snapshot": fixture_snapshot(),
    }


@pytest.fixture
def store():
    return DataStore(tables=tables())


def test_table_keeps_rows_strictly_before_the_deadline(store):
    view = store.as_of(DEADLINE)
    assert isinstance(view, AsOfView)
    assert view.deadline == DEADLINE
    out = view.table("player_match")
    assert out["fixture_key"].tolist() == [10]
    assert store.as_of(DEADLINE + pd.Timedelta(microseconds=1)).table("player_match")[
        "fixture_key"
    ].tolist() == [10, 20]


def test_table_filters_on_available_at_not_event_time():
    df = player_match().assign(event_time=ts("2000-01-01"))
    view = DataStore(tables={"player_match": df}).as_of(DEADLINE)
    assert view.table("player_match")["fixture_key"].tolist() == [10]


def test_table_selects_columns(store):
    out = store.as_of(DEADLINE).table("player_match", columns=["fixture_key", "minutes"])
    assert list(out.columns) == ["fixture_key", "minutes"]
    with pytest.raises(KeyError, match="nope"):
        store.as_of(DEADLINE).table("player_match", columns=["nope"])


def test_naive_deadline_is_rejected(store):
    with pytest.raises(ValueError, match="tz-aware"):
        store.as_of(pd.Timestamp("2026-08-02 12:00"))
    with pytest.raises(ValueError, match="tz-aware"):
        store.as_of(datetime(2026, 8, 2, 12))


def test_non_utc_deadline_is_converted(store):
    view = store.as_of(DEADLINE.tz_convert("Europe/London"))
    assert str(view.deadline.tz) == "UTC"
    assert view.deadline == DEADLINE


def test_unknown_table_lists_the_registered_names(store):
    with pytest.raises(KeyError, match="player_snapshot"):
        store.as_of(DEADLINE).table("players")
    with pytest.raises(KeyError, match="player_snapshot"):
        DataStore(tables={"players": player_match()})


def test_table_not_provided_in_memory(store):
    with pytest.raises(KeyError, match="team_rating"):
        store.as_of(DEADLINE).table("team_rating")


def test_latest_picks_newest_snapshot_per_group_before_the_deadline(store):
    out = store.as_of(DEADLINE).latest("player_snapshot", by=["player_key"])
    assert out["player_key"].tolist() == [1, 2]
    assert out["now_cost"].tolist() == [52, 60]
    assert out["source"].tolist() == ["fpl", "fplcache"]
    assert out["snapshot_at"].tolist() == [T2, T1]
    later = store.as_of(T3 + pd.Timedelta(seconds=1)).latest("player_snapshot", by=["player_key"])
    assert later["now_cost"].tolist() == [99, 60, 70]


def test_latest_selected_columns_start_with_by(store):
    out = store.as_of(DEADLINE).latest("player_snapshot", by=["player_key"], columns=["now_cost"])
    assert list(out.columns) == ["player_key", "now_cost"]


def test_latest_prefers_fpl_regardless_of_input_order():
    df = player_snapshot()
    for seed in range(5):
        shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        out = DataStore(tables={"player_snapshot": shuffled}).as_of(DEADLINE)
        assert out.latest("player_snapshot", by=["player_key"])["now_cost"].tolist() == [52, 60]


def test_latest_groups_on_null_keys():
    odds = pd.DataFrame(
        {
            "fixture_key": [1, 1, 1, 1],
            "market": ["h2h", "h2h", "totals", "totals"],
            "line": pd.array([None, None, 2.5, 2.5], dtype="Float64"),
            "price": [2.0, 2.1, 1.9, 1.8],
            "snapshot_at": pd.Series([T1, T2, T1, T2]).astype(UTC_US),
        }
    )
    odds = odds.assign(event_time=odds["snapshot_at"], available_at=odds["snapshot_at"])
    view = DataStore(tables={"odds_snapshot": odds}).as_of(DEADLINE)
    out = view.latest("odds_snapshot", by=["fixture_key", "market", "line"])
    assert out["price"].tolist() == [2.1, 1.8]


def test_latest_with_no_groups_is_the_single_newest_row(store):
    out = store.as_of(DEADLINE).latest("player_snapshot", by=[], columns=["now_cost"])
    assert out["now_cost"].tolist() == [52]


def test_latest_is_for_snapshot_tables_only(store):
    with pytest.raises(ValueError, match="snapshot"):
        store.as_of(DEADLINE).latest("player_match", by=["player_key"])


def test_latest_before_any_snapshot_is_empty_with_columns(store):
    out = store.as_of(T1).latest("player_snapshot", by=["player_key"], columns=["now_cost"])
    assert out.empty and list(out.columns) == ["player_key", "now_cost"]
    assert out["now_cost"].dtype == np.dtype("int64")


def test_schedule_switches_from_final_to_snapshot():
    no_snapshots = {**tables(), "fixture_snapshot": fixture_snapshot().iloc[0:0]}
    final = DataStore(tables=no_snapshots).as_of(DEADLINE).schedule(2026)
    assert final["schedule_source"].unique().tolist() == ["final"]
    assert final["gw"].tolist() == [1, 2]

    store = DataStore(tables=tables())
    first = store.as_of(T1).schedule(2026)  # T1 snapshot is not yet visible at T1
    assert first["schedule_source"].unique().tolist() == ["final"]
    early = store.as_of(T1 + pd.Timedelta(hours=1)).schedule(2026)
    assert early["schedule_source"].unique().tolist() == ["snapshot"]
    assert early["gw"].tolist() == [1, 2]
    moved = store.as_of(DEADLINE).schedule(2026)
    assert moved["fixture_key"].tolist() == [101, 102]
    assert moved["gw"].tolist() == [1, 3]
    assert moved["gw_index"].tolist() == [1, 3]
    assert list(moved.columns) == list(final.columns)
    # 2025 has no snapshot: the final schedule.
    assert store.as_of(DEADLINE).schedule(2025)["schedule_source"].tolist() == ["final"]


def test_schedule_is_empty_before_publication(store):
    out = store.as_of(ts("2026-05-01")).schedule(2026)
    assert out.empty and "fixture_key" in out.columns


def test_gameweek_for_deadline(store):
    assert store.as_of(DEADLINE).gameweek_for_deadline() == (2026, 2)
    with pytest.raises(LookupError):
        store.as_of(DEADLINE + pd.Timedelta(minutes=1)).gameweek_for_deadline()


def test_returned_frames_are_independent_copies(store):
    view = store.as_of(DEADLINE)
    out = view.table("player_match")
    out.loc[:, "minutes"] = -1
    latest = view.latest("player_snapshot", by=["player_key"])
    latest.loc[:, "now_cost"] = -1
    assert view.table("player_match")["minutes"].tolist() == [90]
    assert view.latest("player_snapshot", by=["player_key"])["now_cost"].tolist() == [52, 60]


def test_input_frames_are_not_mutated():
    df = player_snapshot().sample(frac=1, random_state=1)
    before = df.copy()
    DataStore(tables={"player_snapshot": df}).as_of(DEADLINE).latest(
        "player_snapshot", by=["player_key"]
    )
    pd.testing.assert_frame_equal(df, before)


def test_reads_parquet_lazily_and_matches_in_memory(tmp_path):
    for name, df in tables().items():
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    on_disk = DataStore(tmp_path).as_of(DEADLINE)
    in_memory = DataStore(tables=tables()).as_of(DEADLINE)
    for name in ("player_match", "gameweek"):
        pd.testing.assert_frame_equal(on_disk.table(name), in_memory.table(name))
    pd.testing.assert_frame_equal(
        on_disk.latest("player_snapshot", by=["player_key"]),
        in_memory.latest("player_snapshot", by=["player_key"]),
    )
    pd.testing.assert_frame_equal(on_disk.schedule(2026), in_memory.schedule(2026))
    with pytest.raises(FileNotFoundError, match="team_rating"):
        on_disk.table("team_rating")


def test_views_of_one_store_share_loaded_data(tmp_path):
    for name, df in tables().items():
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    store = DataStore(tmp_path)
    a = store.as_of(DEADLINE).latest("player_snapshot", by=["player_key"])
    b = store.as_of(T3 + pd.Timedelta(seconds=1)).latest("player_snapshot", by=["player_key"])
    assert len(a) == 2 and len(b) == 3


def test_missing_available_at_values_are_rejected():
    df = player_match()
    df.loc[0, "available_at"] = pd.NaT
    with pytest.raises(ValueError, match="available_at"):
        DataStore(tables={"player_match": df}).as_of(DEADLINE).table("player_match")
