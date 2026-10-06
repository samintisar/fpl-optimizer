from datetime import UTC, datetime, timedelta
from functools import partial

import pandas as pd
import pyarrow.parquet as pq
import pytest
from synthetic_raw import CURRENT_AT, World, bootstrap
from test_build_players import current_history
from test_build_understat import two_seasons

from fplopt.build import BUILDERS, ORDER, Builder, build
from fplopt.build.common import EPOCH, schedule_published_at
from fplopt.build.snapshots import build_player_snapshot
from fplopt.build.tables import KINDS, TABLES, TableSpec


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_table_specs_are_well_formed():
    for name, spec in TABLES.items():
        assert isinstance(spec, TableSpec) and spec.name == name
        assert spec.kind in KINDS
        assert spec.key and len(set(spec.key)) == len(spec.key)
        assert (spec.snapshot_col is not None) == (spec.kind == "snapshot"), name
    kinds = {name: spec.kind for name, spec in TABLES.items()}
    assert {n for n, k in kinds.items() if k == "static"} == {
        "team_dim",
        "player_dim",
        "understat_map",
    }
    assert {n for n, k in kinds.items() if k == "schedule"} == {"schedule", "gameweek"}
    assert {n for n, k in kinds.items() if k == "snapshot"} == {
        "player_snapshot",
        "fixture_snapshot",
        "odds_snapshot",
    }


def test_synthetic_build_all_writes_exactly_the_registered_tables(world, monkeypatch):
    fx23, fx24 = two_seasons(world)
    world.add_fplcache_bootstrap(datetime(2024, 6, 1, tzinfo=UTC), bootstrap(2023, fx23))
    world.add_fplcache_bootstrap(datetime(2025, 6, 1, tzinfo=UTC), bootstrap(2024, fx24))
    # 2026-27 from our own archive: two fixture snapshots, an element-summary run.
    fx26 = world.add_current_season(finished_through=2, fd_rows=1000)
    world.add_own_fixtures(CURRENT_AT + timedelta(days=1), fx26)
    world.add_element_summary(CURRENT_AT, current_history(fx26), 2026, through_event=2)
    snapshots = partial(build_player_snapshot, jobs=1, first_season=None)
    monkeypatch.setitem(BUILDERS, "player_snapshot", Builder(snapshots, None))
    # team_dim is already written by World (rebuilding it checks real club names).
    build([name for name in ORDER if name != "team_dim"], world.ctx)

    written = sorted(p.name.removesuffix(".parquet") for p in world.ctx.data_dir.glob("*.parquet"))
    assert written == sorted(TABLES)
    for name, spec in TABLES.items():
        df = pd.read_parquet(world.ctx.table_path(name))
        columns = set(pq.read_schema(world.ctx.table_path(name)).names)
        assert {"event_time", "available_at", *spec.key} <= columns, name
        assert not df.duplicated(list(spec.key)).any(), name
        if spec.kind == "static":
            assert (df["available_at"] == EPOCH).all(), name
        elif spec.kind == "snapshot":
            assert (df["available_at"] == df[spec.snapshot_col]).all(), name
        elif spec.kind == "schedule":
            published = df["season"].map(schedule_published_at)
            assert (df["available_at"] == published).all(), name
    for name in ("schedule", "fixture_snapshot", "player_gw", "player_gw_ownership"):
        assert len(pd.read_parquet(world.ctx.table_path(name))), name


def test_static_tables_declare_public_columns_and_a_visibility_source():
    for name, spec in TABLES.items():
        if spec.kind != "static":
            assert spec.public_columns == () and spec.visible_via is None, name
            continue
        assert len(spec.key) == 1 and spec.key[0] == spec.public_columns[0], name
        table, column = spec.visible_via
        assert TABLES[table].kind != "static", name
    assert TABLES["team_dim"].public_columns == ("team_key", "short_name", "fpl_names")
    assert TABLES["player_dim"].public_columns == (
        "player_key",
        "first_name",
        "second_name",
        "web_name",
    )
    assert TABLES["understat_map"].public_columns == ("understat_id", "player_key")
