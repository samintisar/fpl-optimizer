import pandas as pd
import pytest
from synthetic_raw import (
    CURRENT_AT,
    REPO_CONFIG,
    TEAM_CODES,
    World,
    merged_gw,
    players_raw,
    season_fixtures,
)

from fplopt.build import build
from fplopt.build.players import (
    PlayerMatchError,
    assemble_player_match,
    match_rows,
    player_dim_from_seasons,
)
from fplopt.build.teams import TeamResolver, team_dim_from_config


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


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
    build(["fixture", "gameweek", "player_season", "player_dim", "player_match"], world.ctx)
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
    # 2025-26: an exact duplicate row; `team` names present.
    fx25 = season_fixtures(2025)
    merged25 = merged_gw(fx25)
    names = team_dim_from_config(REPO_CONFIG / "teams.csv").set_index("team_key")["fpl_names"]
    merged25["team"] = [names[TEAM_CODES[e - 1]].split(";")[0] for e in merged25["element"]]
    world.add_vaastav_season(2025, merged=pd.concat([merged25, merged25.iloc[[3]]]))
    # 2026-27: element-summary run through GW1 (GW2 is finished but not yet in the run).
    fx26 = world.add_current_season(finished_through=2)
    world.add_element_summary(CURRENT_AT, current_history(fx26), 2026, through_event=1)

    pm = build_players(world)
    assert pm.groupby("season").size().to_dict() == {2019: 760, 2024: 760, 2025: 760, 2026: 20}
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
    assert player_season.groupby("season").size().to_dict() == {
        2019: 20,
        2024: 20,
        2025: 20,
        2026: 20,
    }
    first_deadline = world.ctx.table("gameweek").groupby("season")["deadline_time"].min()
    assert (player_season["available_at"] == player_season["season"].map(first_deadline)).all()
    player_dim = world.ctx.table("player_dim")
    assert len(player_dim) == 20
    assert player_dim["opta_code"].iloc[0] == f"p{player_dim['player_key'].iloc[0]}"
    assert (player_dim["first_season"] == 2019).all() and (player_dim["last_season"] == 2026).all()


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
    assert dim.loc[7, ["first_season", "last_season"]].tolist() == [2016, 2017]
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


def test_starts_null_for_early_2022_gws_only():
    early = assemble(*small_inputs(gw=15))
    assert early["starts"].isna().all()
    later = assemble(*small_inputs(gw=16))
    assert later["starts"].tolist() == [0, 0]


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
    with pytest.raises(PlayerMatchError, match="not in player_season"):
        assemble(merged, fixture, gameweek, player_season.iloc[:1], team_codes, resolver)
    doubled = pd.concat([merged, merged.assign(minutes=[5, 5])])
    with pytest.raises(PlayerMatchError, match="duplicate"):
        assemble(doubled, fixture, gameweek, player_season, team_codes, resolver)
