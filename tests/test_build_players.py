from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from synthetic_raw import (
    CURRENT_AT,
    REPO_CONFIG,
    TEAM_CODES,
    World,
    bootstrap,
    merged_gw,
    players_raw,
    season_fixtures,
)

from fplopt.build import BUILDERS, build
from fplopt.build.common import write_table
from fplopt.build.players import (
    PlayerMatchError,
    assemble_player_match,
    first_visible_rows,
    match_rows,
    player_dim_from_seasons,
    player_gw_ownership_table,
    registration_available_at,
    registration_stints,
)
from fplopt.build.teams import TeamResolver, team_dim_from_config


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


LATER_AT = CURRENT_AT + timedelta(days=1)
HOUR = pd.Timedelta(hours=1)


def utc(text):
    return pd.Timestamp(text, tz="UTC").as_unit("us")


def current_history(fixtures):
    """element-summary history rows for the finished 2026 fixtures (xG as strings)."""
    played = fixtures[fixtures["finished"]].copy()
    played["event"] = played["event"].astype(int)
    history = merged_gw(played).drop(columns=["name", "GW", "xP"])
    history["starts"] = 1
    history["defensive_contribution"] = 3
    history["expected_goals"] = "0.25"
    history["expected_assists"] = "0.10"
    history["expected_goals_conceded"] = "1.30"
    return history


def build_players(world):
    """fixture … player_dim. Written directly, not via `build()`, which would also run the
    `understat` builder after player_match / player_dim: these worlds have no Understat."""
    build(["fixture", "gameweek"], world.ctx)
    world.write_player_snapshot()
    for name in ("player_gw", "player_season", "player_match", "player_dim"):
        builder = BUILDERS[name]
        df = builder.run(world.ctx)
        write_table(df, name, builder.schema, world.ctx.data_dir, builder.sort_by)
        world.ctx.forget(name)
    return world.ctx.table("player_match")


def test_player_tables_end_to_end(world):
    # 2019-20: a phantom copy of fixture 5 filed under round 29 (vaastav COVID quirk).
    fx19 = season_fixtures(2019)
    merged19 = merged_gw(fx19)
    phantom = merged19[merged19["fixture"] == 5].assign(round=29, GW=29, minutes=0)
    world.add_vaastav_season(2019, merged=pd.concat([merged19, phantom], ignore_index=True))
    # 2024-25: an assistant manager (element_type 5) with a row.
    fx24 = season_fixtures(2024)
    merged24 = merged_gw(fx24)
    manager_row = merged24.iloc[[0]].assign(element=21, minutes=0, goals_scored=0)
    raw24 = players_raw(2024)
    manager = raw24.iloc[[0]].assign(id=21, code=999999, element_type=5)
    world.add_vaastav_season(
        2024,
        fixtures=fx24,
        merged=pd.concat([merged24, manager_row]),
        players=pd.concat([raw24, manager]),
    )
    # 2025-26: an exact duplicate row; `team` names present; element 21 (code 100021) joins
    # team id 1 before GW10 (rows from GW10 on, 0 minutes).
    fx25 = season_fixtures(2025)
    merged25 = merged_gw(fx25)
    names = team_dim_from_config(REPO_CONFIG / "teams.csv").set_index("team_key")["fpl_names"]
    merged25["team"] = [names[TEAM_CODES[e - 1]].split(";")[0] for e in merged25["element"]]
    signing = merged25[(merged25["element"] == 1) & (merged25["round"] >= 10)].assign(
        element=21, minutes=0, goals_scored=0, clean_sheets=0, total_points=0
    )
    raw25 = players_raw(2025)
    raw25 = pd.concat([raw25, raw25.iloc[[0]].assign(id=21, code=100021, web_name="New")])
    world.add_vaastav_season(
        2025, merged=pd.concat([merged25, merged25.iloc[[3]], signing]), players=raw25
    )
    # 2026-27: element-summary run through GW1 (GW2 is finished but not yet in the run); a
    # newer bootstrap lists element 21 (code 100022), registered after the run: no rows.
    fx26 = world.add_current_season(finished_through=2)
    world.add_element_summary(CURRENT_AT, current_history(fx26), 2026, through_event=1)
    later = bootstrap(2026, season_fixtures(2026), finished_through=2)
    later["elements"].append({**later["elements"][1], "id": 21, "code": 100022})
    world.add_own_bootstrap(LATER_AT, later)

    pm = build_players(world)
    assert pm.groupby("season").size().to_dict() == {2019: 760, 2024: 760, 2025: 789, 2026: 20}
    assert not pm["player_key"].eq(999999).any()
    assert pm.duplicated(["player_key", "fixture_key"]).sum() == 0
    f5 = pm[pm["fixture_key"] == 2019005]
    assert f5["gw"].tolist() == [1, 1] and (f5["minutes"] == 90).all()

    # Team from the fixture; the player of team id i is element i (code 100000 + i).
    fixture = world.ctx.table("fixture").set_index("fixture_key")
    row = pm[(pm["fixture_key"] == 2025001) & pm["was_home"]].iloc[0]
    assert row["team_key"] == fixture.loc[2025001, "home_team_key"]
    assert row["opponent_team_key"] == fixture.loc[2025001, "away_team_key"]
    assert row["player_key"] == 100000 + TEAM_CODES.index(row["team_key"]) + 1
    assert row["available_at"] == fixture.loc[2025001, "available_at"]
    assert row["event_time"] == fixture.loc[2025001, "kickoff_time"]

    s19 = pm[pm["season"] == 2019]
    assert s19["starts"].isna().all() and s19["fpl_xg"].isna().all()
    s26 = pm[pm["season"] == 2026]
    assert (s26["source"] == "fpl").all() and (s26["gw"] == 1).all()
    assert s26["fpl_xg"].tolist() == [0.25] * 20
    assert (s26["starts"] == 1).all() and (s26["defensive_contribution"] == 3).all()
    assert s26["us_xg"].isna().all()
    assert (pm.loc[pm["season"] < 2026, "source"] == "vaastav").all()

    player_season = world.ctx.table("player_season")
    assert "team_key" not in player_season.columns  # per match: player_match; as-of: snapshot
    assert player_season.groupby("season").size().to_dict() == {
        2019: 20,
        2024: 20,
        2025: 21,
        2026: 21,
    }
    # event_time = deadline of the player's first visible player_gw row that season,
    # available_at = that row's (an hour before the deadline for players registered at
    # launch; a later signing without snapshots is visible from the next GW); a player
    # without rows: the time of the bootstrap that lists him.
    deadline = world.ctx.table("gameweek").set_index(["season", "gw"])["deadline_time"]
    ps = player_season.set_index(["season", "player_key"])
    assert ps.loc[(2025, 100021), "event_time"] == deadline[(2025, 10)]
    assert ps.loc[(2025, 100021), "available_at"] == deadline[(2025, 10)]
    signing = world.ctx.table("player_gw").set_index(["player_key", "season", "gw"])
    assert signing.loc[(100021, 2025, 10), "available_at"] == deadline[(2025, 10)]
    assert signing.loc[(100021, 2025, 11), "available_at"] == deadline[(2025, 11)] - HOUR
    listed = ps.loc[(2026, 100022)]
    assert listed["available_at"] == listed["event_time"] == pd.Timestamp(LATER_AT).as_unit("us")
    regular = ps.drop([(2025, 100021), (2026, 100022)])
    gw1 = pd.Series([deadline[(season, 1)] for season, _ in regular.index], index=regular.index)
    assert (regular["event_time"] == gw1).all()
    assert (regular["available_at"] == gw1 - HOUR).all()
    player_dim = world.ctx.table("player_dim")
    assert len(player_dim) == 22
    assert player_dim["opta_code"].iloc[0] == f"p{player_dim['player_key'].iloc[0]}"
    assert "first_season" not in player_dim.columns and "last_season" not in player_dim.columns


def test_player_match_rows_wait_for_the_players_first_listing(world):
    # 2026-27 with snapshots from before GW1. Element 21 (code 100022, team id 1) is first
    # listed at LATER_AT, after the GW1 and GW2 lockdowns, yet his element-summary history has
    # (0-minute) rows for both GWs.
    preseason = bootstrap(2026, season_fixtures(2026), finished_through=0)
    world.add_own_bootstrap(datetime(2026, 7, 20, tzinfo=UTC), preseason)
    fx26 = world.add_current_season(finished_through=2)
    later = bootstrap(2026, season_fixtures(2026), finished_through=2)
    later["elements"].append({**later["elements"][0], "id": 21, "code": 100022})
    world.add_own_bootstrap(LATER_AT, later)
    history = current_history(fx26)
    zeros = {"minutes": 0, "goals_scored": 0, "assists": 0, "clean_sheets": 0, "starts": 0}
    signing = history[history["element"] == 1].assign(element=21, **zeros)
    world.add_element_summary(LATER_AT, pd.concat([history, signing]), 2026, through_event=2)

    pm = build_players(world)
    listed_at = pd.Timestamp(LATER_AT).as_unit("us")
    ps = world.ctx.table("player_season").set_index(["season", "player_key"])
    assert ps.loc[(2026, 100022), "available_at"] == listed_at
    lockdown = world.ctx.table("gameweek").set_index(["season", "gw"])["lockdown_time"]
    assert lockdown[(2026, 2)] < listed_at
    late = pm["player_key"] == 100022
    assert pm.loc[late, "gw"].tolist() == [1, 2]
    assert (pm.loc[late, "available_at"] == listed_at).all()
    gw_lockdown = lockdown.reindex(pd.MultiIndex.from_frame(pm[["season", "gw"]])).to_numpy()
    assert (pm.loc[~late, "available_at"] == gw_lockdown[~late.to_numpy()]).all()
    fixture = world.ctx.table("fixture").set_index("fixture_key")
    assert (pm["event_time"] == fixture.loc[pm["fixture_key"], "kickoff_time"].to_numpy()).all()


def test_current_season_stays_buildable_after_next_season_data_arrives(world, caplog):
    # 2026-27 complete through GW2 (football-data published for all of it).
    fx26 = world.add_current_season(finished_through=2, fd_rows=1000)
    world.add_element_summary(CURRENT_AT, current_history(fx26), 2026, through_event=2)
    # June 2027: FPL resets. Our archive gets the 2027-28 bootstrap and fixture list (nothing
    # played), an element-summary run with zero history rows, and a newer run whose manifest
    # has no season (complete for no season).
    reset_at = datetime(2027, 6, 20, tzinfo=UTC)
    fx27 = season_fixtures(2027).assign(finished=False, team_h_score=None, team_a_score=None)
    world.add_own_bootstrap(reset_at, bootstrap(2027, season_fixtures(2027), finished_through=0))
    world.add_own_fixtures(reset_at, fx27)
    no_history = pd.DataFrame({"element": pd.Series(dtype="int64")})
    world.add_element_summary(reset_at, no_history, 2027, 0, elements=list(range(1, 21)))
    world.add_element_summary(reset_at + timedelta(hours=1), no_history, None, 0, elements=[1])

    pm = build_players(world)
    fixture = world.ctx.table("fixture")
    assert fixture.groupby("season").size().to_dict() == {2026: 380, 2027: 380}
    assert fixture.loc[fixture["season"] == 2026, "finished"].sum() == 20
    assert pm.groupby("season").size().to_dict() == {2026: 40}
    gameweek = world.ctx.table("gameweek")
    assert gameweek.groupby("season").size().to_dict() == {2026: 38, 2027: 38}
    player_season = world.ctx.table("player_season").set_index(["season", "player_key"])
    assert player_season.groupby(level="season").size().to_dict() == {2026: 20, 2027: 20}
    reset = pd.Timestamp(reset_at).as_unit("us")
    assert (player_season.loc[2027, "available_at"] == reset).all()  # listed, no rows yet
    assert "has no season" in caplog.text


def test_goal_sum_mismatch_fails_the_build(world):
    fx = season_fixtures(2023)
    merged = merged_gw(fx)
    merged.loc[0, "goals_scored"] += 1
    world.add_vaastav_season(2023, fixtures=fx, merged=merged)
    with pytest.raises(PlayerMatchError, match="1 fixture"):
        build_players(world)


def test_player_dim_takes_newest_names():
    seasons = pd.DataFrame(
        {
            "player_key": [7, 7, 9],
            "season": [2017, 2016, 2016],
            "first_name": ["Heung-Min", "Heung-min", "A"],
            "second_name": ["Son", "Son", "B"],
            "web_name": ["Son", "Son", "B"],
        }
    )
    dim = player_dim_from_seasons(seasons).set_index("player_key")
    assert dim.loc[7, "first_name"] == "Heung-Min"
    assert dim.loc[9, "opta_code"] == "p9"


# --- assemble_player_match units --------------------------------------------------------


def small_inputs(season=2022, gw=10):
    fixture = pd.DataFrame(
        {
            "season": [season],
            "fpl_fixture_id": [100],
            "fixture_key": [season * 1000 + 100],
            "gw": pd.array([gw], dtype="Int64"),
            "kickoff_time": [utc("2022-10-15T14:00")],
            "home_team_key": [3],
            "away_team_key": [8],
            "home_goals": pd.array([1], dtype="Int64"),
            "away_goals": pd.array([0], dtype="Int64"),
        }
    )
    gameweek = pd.DataFrame(
        {"season": [season], "gw": [gw], "lockdown_time": [utc("2022-10-17T08:00")]}
    )
    player_season = pd.DataFrame(
        {"season": [season] * 2, "element_id": [1, 2], "player_key": [501, 502]}
    )
    merged = pd.DataFrame(
        {
            "element": [1, 2],
            "fixture": [100, 100],
            "round": [gw, gw],
            "was_home": [True, False],
            "opponent_team": [2, 1],
            "team": ["Arsenal", "Chelsea"],
            "starts": [0, 0],
            **{name: [0, 0] for name in ["minutes", "assists", "clean_sheets", "goals_conceded"]},
            **{
                name: [0, 0]
                for name in [
                    "own_goals",
                    "penalties_saved",
                    "penalties_missed",
                    "yellow_cards",
                    "red_cards",
                    "saves",
                    "bonus",
                    "bps",
                    "total_points",
                ]
            },
            "goals_scored": [1, 0],
        }
    )
    team_codes = {season: {1: 3, 2: 8}}
    resolver = TeamResolver(team_dim_from_config(REPO_CONFIG / "teams.csv"))
    return merged, fixture, gameweek, player_season, team_codes, resolver


def assemble(merged, fixture, gameweek, player_season, team_codes, resolver, season=2022):
    raw = match_rows(merged, season, "vaastav")
    return assemble_player_match(raw, fixture, gameweek, player_season, team_codes, resolver)


LATE_COLUMNS = ["starts", "fpl_xg", "fpl_xa", "fpl_xgc"]
PLACEHOLDER_XG = {
    "expected_goals": [0.0, 0.0],
    "expected_assists": [0.0, 0.0],
    "expected_goals_conceded": [0.0, 0.0],
}


def test_starts_and_fpl_x_null_for_early_2022_gws_only():
    merged, *rest = small_inputs(gw=15)
    early = assemble(merged.assign(**PLACEHOLDER_XG), *rest)
    assert early[LATE_COLUMNS].isna().all().all()
    merged, *rest = small_inputs(gw=16)
    later = assemble(merged.assign(**PLACEHOLDER_XG), *rest)
    assert later["starts"].tolist() == [0, 0]
    assert later["fpl_xg"].tolist() == [0.0, 0.0] and later["fpl_xgc"].tolist() == [0.0, 0.0]


def test_placeholder_zero_block_fails_the_build(world):
    # FPL xG all 0 in GW3 although goals were scored: a placeholder block, not data.
    fx = season_fixtures(2023)
    merged = merged_gw(fx).assign(expected_goals=0.3)
    merged.loc[merged["round"] == 3, "expected_goals"] = 0.0
    world.add_vaastav_season(2023, fixtures=fx, merged=merged)
    with pytest.raises(PlayerMatchError, match="placeholder") as failure:
        build_players(world)
    assert "1 placeholder-zero block" in str(failure.value)
    assert str(failure.value).endswith("2023   3 fpl_xg")


def test_team_name_and_opponent_mismatches_fail():
    merged, *rest = small_inputs()
    bad_name = merged.assign(team=["Chelsea", "Chelsea"])
    with pytest.raises(PlayerMatchError, match="`team` name mismatch"):
        assemble(bad_name, *rest)
    bad_opponent = merged.assign(opponent_team=[1, 1])
    with pytest.raises(PlayerMatchError, match="opponent_team mismatch"):
        assemble(bad_opponent, *rest)


def test_unknown_element_and_leftover_duplicates_fail():
    merged, fixture, gameweek, player_season, team_codes, resolver = small_inputs()
    with pytest.raises(PlayerMatchError, match="not a registered player"):
        assemble(merged, fixture, gameweek, player_season.iloc[:1], team_codes, resolver)
    doubled = pd.concat([merged, merged.assign(minutes=[5, 5])])
    with pytest.raises(PlayerMatchError, match="duplicate"):
        assemble(doubled, fixture, gameweek, player_season, team_codes, resolver)


def test_phantom_drop_never_loses_an_element_fixture_pair():
    # Both rows of (element 1, fixture 100) are filed under GW29, not the fixture's GW10:
    # dropping them as phantoms would silently lose the pair.
    merged, *rest = small_inputs(gw=10)
    filed_elsewhere = merged.iloc[[0]].assign(round=29)
    rows = pd.concat([filed_elsewhere, filed_elsewhere.assign(minutes=5), merged.iloc[[1]]])
    with pytest.raises(PlayerMatchError, match="would be lost"):
        assemble(rows, *rest)


# --- player_gw / player_gw_ownership ----------------------------------------------------


def double_gw_world(world, *, conflict: bool = False):
    """2022-23 with FPL's cancelled GW7 (rounds 7 and 8 both form GW8: a double for every
    club) and per-GW prices/ownership; 2026-27 through GW2 from element-summary."""
    fx22 = season_fixtures(2022, gw_numbers=[*range(1, 7), 8, *range(8, 39)])
    merged = merged_gw(fx22)
    merged["value"] = 50 + merged["round"]
    merged["selected"] = 1000 * merged["element"] + merged["round"]
    merged["transfers_in"] = merged["round"]
    merged["transfers_out"] = merged["element"]
    if conflict:
        first = merged.index[(merged["element"] == 3) & (merged["round"] == 8)][0]
        merged.loc[first, "value"] = 99
    world.add_vaastav_season(2022, fixtures=fx22, merged=merged)
    fx26 = world.add_current_season(finished_through=2, fd_rows=1000)
    history = current_history(fx26)
    history["value"] = 60 + history["round"]
    world.add_element_summary(CURRENT_AT, history, 2026, through_event=2)


def test_player_gw_one_row_per_gw_and_double(world):
    double_gw_world(world)
    world.write_player_snapshot()
    build(["fixture", "gameweek", "player_gw", "player_gw_ownership"], world.ctx)

    gw = world.ctx.table("player_gw")
    assert list(gw.columns) == [
        "player_key",
        "season",
        "gw",
        "team_key",
        "element_type",
        "value",
        "event_time",
        "available_at",
    ]
    assert gw.groupby("season").size().to_dict() == {2022: 20 * 37, 2026: 20 * 2}
    assert not gw.duplicated(["player_key", "season", "gw"]).any()
    assert 7 not in set(gw.loc[gw["season"] == 2022, "gw"])
    deadline = world.ctx.table("gameweek").set_index(["season", "gw"])["deadline_time"]
    g = gw.set_index(["player_key", "season", "gw"])
    assert g.loc[(100003, 2022, 8), "value"] == 58  # one row for the double
    assert g.loc[(100003, 2022, 8), "team_key"] == TEAM_CODES[2]
    assert g.loc[(100003, 2022, 8), "element_type"] == 4  # (3 % 4) + 1
    assert g.loc[(100005, 2026, 2), "value"] == 62
    at = pd.Series([deadline[(s, w)] for _, s, w in g.index], index=g.index)
    # Price at the deadline: fixed before it.
    assert (g["event_time"] == at).all() and (g["available_at"] == at - HOUR).all()

    own = world.ctx.table("player_gw_ownership")
    assert list(own.columns) == [
        "player_key",
        "season",
        "gw",
        "selected",
        "transfers_in",
        "transfers_out",
        "event_time",
        "available_at",
    ]
    o = own.set_index(["player_key", "season", "gw"])
    assert list(o.index) == list(g.index)
    assert o.loc[(100003, 2022, 8), ["selected", "transfers_in", "transfers_out"]].tolist() == [
        3008,
        8,
        3,
    ]
    # Ownership after GW t's transfers: final only at the deadline.
    assert (o["event_time"] == at).all() and (o["available_at"] == at).all()


def test_player_gw_fails_when_a_double_disagrees(world):
    double_gw_world(world, conflict=True)
    world.write_player_snapshot()
    build(["fixture", "gameweek"], world.ctx)
    with pytest.raises(PlayerMatchError, match="differ within a GW"):
        build(["player_gw"], world.ctx)


# --- registration timing ----------------------------------------------------------------

D1, D2, D3 = utc("2021-08-13T17:30"), utc("2021-08-21T10:00"), utc("2021-08-28T10:00")
DEADLINES = {1: D1, 2: D2, 3: D3}


def gw_rows(rows):
    """(player_key, gw, team_key) -> player GW rows of season 2021 with deadlines."""
    df = pd.DataFrame(rows, columns=["player_key", "gw", "team_key"])
    return df.assign(
        season=2021,
        gw_index=df["gw"],
        deadline_time=df["gw"].map(DEADLINES).astype("datetime64[us, UTC]"),
    )


def stints(rows):
    """(player_key, team_key, first listed) -> registration stints of season 2021."""
    df = pd.DataFrame(rows, columns=["player_key", "team_key", "snapshot_at"])
    df["snapshot_at"] = df["snapshot_at"].map(utc)
    return df.assign(season=2021)[["snapshot_at", "season", "player_key", "team_key"]]


CLUB_FIRST = pd.DataFrame({"season": 2021, "team_key": [10, 20], "first_gw_index": [1, 2]})


def available(rows, stint_rows=(), coverage_start=None):
    df = gw_rows(rows)
    out = registration_available_at(df, stints(list(stint_rows)), coverage_start, CLUB_FIRST)
    return dict(zip(zip(df["player_key"], df["gw"], strict=True), out, strict=True))


def test_registration_with_snapshots_waits_for_the_first_listing_with_that_club():
    out = available(
        [
            (1, 1, 10),
            (1, 2, 10),
            (1, 3, 10),  # listed since before GW1
            (2, 2, 10),
            (2, 3, 10),  # added the day after the GW2 deadline
            (3, 1, 10),
            (3, 2, 20),
            (3, 3, 20),  # moved within the last hour before GW2
            (4, 1, 10),
            (4, 2, 20),  # moved to club 20 after the GW2 deadline
            (5, 3, 30),  # never listed with club 30
        ],  # fmt: skip
        [
            (1, 10, "2021-08-01T00:00"),
            (2, 10, "2021-08-22T08:00"),
            (3, 10, "2021-08-01T00:00"),
            (3, 20, "2021-08-21T09:30"),
            (4, 10, "2021-08-01T00:00"),
            (4, 20, "2021-08-23T02:00"),
            (5, 10, "2021-08-01T00:00"),
        ],
        coverage_start=utc("2021-08-01"),
    )
    assert out[(1, 1)] == D1 - HOUR and out[(1, 3)] == D3 - HOUR
    assert out[(2, 2)] == utc("2021-08-22T08:00")  # not visible at the GW2 deadline
    assert out[(2, 3)] == D3 - HOUR
    assert out[(3, 1)] == D1 - HOUR
    assert out[(3, 2)] == utc("2021-08-21T09:30")  # listed before the deadline: visible
    assert out[(3, 3)] == D3 - HOUR
    assert out[(4, 2)] == utc("2021-08-23T02:00")
    assert out[(5, 3)] == D3  # never listed: from the next GW


def test_registration_without_snapshots_trusts_only_game_launch_and_unchanged_clubs():
    out = available(
        [
            (1, 1, 10),
            (1, 2, 10),
            (1, 3, 10),  # registered at launch
            (2, 2, 20),
            (2, 3, 20),  # club 20 blanks GW1: GW2 is its first GW
            (3, 2, 10),
            (3, 3, 10),  # signed after GW1
            (4, 1, 10),
            (4, 2, 20),
            (4, 3, 20),  # moved from club 10 to 20
        ],  # fmt: skip
    )
    assert [out[(1, gw)] for gw in (1, 2, 3)] == [D1 - HOUR, D2 - HOUR, D3 - HOUR]
    assert [out[(2, gw)] for gw in (2, 3)] == [D2 - HOUR, D3 - HOUR]
    assert [out[(3, gw)] for gw in (2, 3)] == [D2, D3 - HOUR]
    assert [out[(4, gw)] for gw in (1, 2, 3)] == [D1 - HOUR, D2, D3 - HOUR]


def test_registration_rule_switches_where_snapshot_coverage_starts():
    # Coverage starts between the GW2 and GW3 deadlines: GW1-2 use the launch rule.
    out = available(
        [(1, 2, 10), (1, 3, 10), (2, 3, 10)],
        [(1, 10, "2021-08-25T00:00"), (2, 10, "2021-08-29T00:00")],
        coverage_start=utc("2021-08-25"),
    )
    assert out[(1, 2)] == D2  # first row not at the club's first GW
    assert out[(1, 3)] == D3 - HOUR  # listed before the GW3 deadline
    assert out[(2, 3)] == utc("2021-08-29T00:00")


def test_registration_stints_are_runs_of_the_same_club():
    snaps = pd.DataFrame(
        {
            "snapshot_at": [utc(f"2021-08-0{d}") for d in (1, 2, 3, 4, 1, 2)],
            "season": 2021,
            "player_key": [1, 1, 1, 1, 2, 2],
            "team_key": [10, 10, 20, 10, 30, 30],
            "now_cost": 50,
        }
    )
    out = registration_stints(snaps.sample(frac=1, random_state=0))
    assert out.to_dict("list") == {
        "snapshot_at": [utc("2021-08-01"), utc("2021-08-03"), utc("2021-08-04"), utc("2021-08-01")],
        "season": [2021] * 4,
        "player_key": [1, 1, 1, 2],
        "team_key": [10, 20, 10, 30],
    }


def test_ownership_and_player_season_never_precede_the_registration():
    gw = gw_rows([(1, 1, 10), (1, 2, 10), (2, 2, 10), (2, 3, 10)])
    gw = gw.assign(selected=5, transfers_in=1, transfers_out=0)
    known = {(1, 1): D1 - HOUR, (1, 2): D2 - HOUR, (2, 2): utc("2021-08-30"), (2, 3): D3}
    player_gw = gw.assign(
        event_time=gw["deadline_time"],
        available_at=[known[(p, g)] for p, g in zip(gw["player_key"], gw["gw"], strict=True)],
    )
    player_gw["available_at"] = player_gw["available_at"].astype("datetime64[us, UTC]")
    own = player_gw_ownership_table(gw, player_gw).set_index(["player_key", "gw"])
    assert own.loc[(1, 1), "available_at"] == D1  # final at the deadline
    assert own.loc[(2, 2), "available_at"] == utc("2021-08-30")  # registered later still
    assert own.loc[(2, 3), "available_at"] == D3

    first = first_visible_rows(player_gw).set_index("player_key")
    assert first.loc[1, "first_event_time"] == D1
    assert first.loc[1, "first_available_at"] == D1 - HOUR
    # Player 2's GW3 row is visible (D3) before his GW2 row (30 Aug).
    assert first.loc[2, "first_event_time"] == D3
    assert first.loc[2, "first_available_at"] == D3


def test_rebuilding_player_season_rebuilds_player_match(caplog):
    """player_match rows wait for the player's first listing in player_season (#24), so a
    player_season rebuild also rebuilds player_match (and, through it, understat and
    team_match)."""
    from fplopt.build import _resolve

    caplog.set_level("INFO", logger="fplopt.build")
    assert _resolve(["player_season"]) == [
        "player_season",
        "player_match",
        "understat",
        "team_match",
    ]
    assert "also rebuilding player_match: its rows depend on player_season" in caplog.text
    assert _resolve(["player_gw"]) == ["player_gw"]


def test_rescored_points_name_a_season_without_backtest_rules():
    """A newly archived season (e.g. 2027/28) has no backtest rules yet: the build stops
    with what to add, not an opaque error from the rules loader."""
    from fplopt.build.players import PlayerMatchError, rescored_points

    matches = pd.DataFrame({"season": [2027], "player_key": [1]})
    seasons = pd.DataFrame({"season": [2027], "player_key": [1], "element_type": [3]})
    with pytest.raises(PlayerMatchError, match=r"2027-28 has no backtest rules.*rules export"):
        rescored_points(matches, seasons)


def test_rescored_points_put_every_season_on_the_backtest_rules(built):
    """`rescored_points` = the row re-scored from its stats under its season's backtest
    rules (on real data it equals `total_points` for 2016/17-2024/25 except GK goals, 6 then
    and 10 now: tests/test_backtest_scoring.py; synthetic raw points are arbitrary)."""
    from fplopt.backtest.rules import backtest_rules
    from fplopt.backtest.scoring import score_matches

    matches = built["player_match"]
    positions = built["player_season"].set_index(["season", "player_key"])["element_type"]
    element_type = positions.reindex(pd.MultiIndex.from_frame(matches[["season", "player_key"]]))
    matches = matches.assign(element_type=element_type.to_numpy())
    for season, rows in matches.groupby("season"):
        expected = score_matches(rows, backtest_rules(int(season)))["points"]
        assert rows["rescored_points"].tolist() == expected.tolist(), season
