from datetime import UTC, datetime, timedelta
from functools import partial

import pandas as pd
import pytest
from synthetic_raw import CURRENT_AT, World, bootstrap
from test_build_players import current_history
from test_build_understat import two_seasons

from fplopt.build import BUILDERS, ORDER, Builder, build
from fplopt.build.snapshots import build_player_snapshot
from fplopt.build.tables import TABLES


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Every table of a synthetic world: vaastav 2023 + 2024 (fplcache bootstraps after
    each season), 2026 from our own archive (bootstrap 1 Sep, fixtures 2 Sep, GW1-2 played).
    Shared by the feature and leakage tests: copy before modifying (see `tables`)."""
    world = World(tmp_path_factory.mktemp("features"))
    fx23, fx24 = two_seasons(world)
    world.add_fplcache_bootstrap(datetime(2024, 6, 1, tzinfo=UTC), bootstrap(2023, fx23))
    world.add_fplcache_bootstrap(datetime(2025, 6, 1, tzinfo=UTC), bootstrap(2024, fx24))
    fx26 = world.add_current_season(finished_through=2, fd_rows=1000)
    world.add_own_fixtures(CURRENT_AT + timedelta(days=1), fx26)
    world.add_element_summary(CURRENT_AT, current_history(fx26), 2026, through_event=2)
    snapshots = partial(build_player_snapshot, jobs=1, first_season=None)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(BUILDERS, "player_snapshot", Builder(snapshots, None))
        build([name for name in ORDER if name != "team_dim"], world.ctx)
    return {name: pd.read_parquet(world.ctx.table_path(name)) for name in TABLES}


@pytest.fixture
def tables(built):
    """A private copy of the synthetic tables (tests may modify it)."""
    return {name: df.copy() for name, df in built.items()}
