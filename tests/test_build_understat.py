import pandas as pd
import pytest
from synthetic_raw import REPO_CONFIG, TEAM_CODES, World, football_data, merged_gw, season_fixtures

from fplopt.build import build
from fplopt.build.teams import TeamResolver, team_dim_from_config
from fplopt.build.understat import (
    UnderstatMappingError,
    candidate_pairs,
    map_understat,
    mapping_problems,
    name_scores,
    normalise_name,
    read_overrides,
    team_sides,
    understat_team_rows,
)

CONFIG = pd.read_csv(REPO_CONFIG / "teams.csv")
US_NAMES = dict(zip(CONFIG["team_key"], CONFIG["understat"], strict=True))
EMPTY_OVERRIDES = pd.DataFrame(
    {"understat_id": pd.Series(dtype="Int64"), "player_key": pd.Series(dtype="Int64")}
)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def us_team(team_id):
    return US_NAMES[TEAM_CODES[team_id - 1]]


# --- synthetic Understat files ----------------------------------------------------------


def player_file_rows(fixtures, season, team_id, xg=0.3):
    """Understat career-log rows of the only player of `team_id` (90 minutes every match)."""
    rows = []
    for fx in fixtures.itertuples(index=False):
        if team_id not in (fx.team_h, fx.team_a):
            continue
        scored = fx.team_h_score if fx.team_h == team_id else fx.team_a_score
        rows.append(
            {
                "goals": scored,
                "shots": 2,
                "xG": xg,
                "time": 90,
                "position": "FW",
                "h_team": us_team(fx.team_h),
                "a_team": us_team(fx.team_a),
                "h_goals": fx.team_h_score,
                "a_goals": fx.team_a_score,
                "date": fx.kickoff_time[:10],
                "id": season * 1000 + fx.id,
                "season": season,
                "roster_id": 1,
                "xA": 0.1,
                "assists": 0,
                "key_passes": 1,
                "npg": scored,
                "npxG": xg,
                "xGChain": 0.0,
                "xGBuildup": 0.0,
            }
        )
    return pd.DataFrame(rows)


def team_file_rows(fixtures, team_id, xg=1.2):
    rows = []
    for fx in fixtures.itertuples(index=False):
        if team_id not in (fx.team_h, fx.team_a):
            continue
        home = fx.team_h == team_id
        rows.append(
            {
                "h_a": "h" if home else "a",
                "xG": xg,
                "xGA": 0.8,
                "npxG": xg - 0.1,
                "npxGA": 0.7,
                "scored": fx.team_h_score if home else fx.team_a_score,
                "missed": fx.team_a_score if home else fx.team_h_score,
                "date": fx.kickoff_time.replace("T", " ").removesuffix("Z"),
            }
        )
    return pd.DataFrame(rows)


def two_seasons(world, *, skip_player=None):
    """2023 and 2024 vaastav seasons, Understat files for every player (in the 2024-25
    folder; player 1's file also stale in 2023-24) and team files, fd xG for 2024."""
    fx23 = world.add_vaastav_season(2023)
    fx24 = season_fixtures(2024)
    merged24 = merged_gw(fx24).assign(expected_goals=0.5)
    world.add_vaastav_season(2024, fixtures=fx24, merged=merged24)
    fd24 = football_data(fx24, 2024).assign(HxG=1.5, AxG=0.5)
    world.football_data(2024, fd24, at=pd.Timestamp("2026-10-07", tz="UTC").to_pydatetime())
    for team_id in range(1, 21):
        if team_id == skip_player:
            continue
        career = pd.concat(
            [
                player_file_rows(fx23, 2023, team_id),
                player_file_rows(fx24, 2024, team_id),
                # Another league and a pre-2016 EPL season: no fixture.
                pd.DataFrame(
                    [
                        {
                            **player_file_rows(fx24, 2024, team_id).iloc[0].to_dict(),
                            "h_team": "Barcelona",
                            "a_team": "Real Madrid",
                            "id": 1,
                        },
                        {
                            **player_file_rows(fx24, 2024, team_id).iloc[0].to_dict(),
                            "season": 2014,
                            "date": "2014-09-01",
                            "id": 2,
                        },
                    ]
                ),
            ]
        )
        world.vaastav(f"2024-25/understat/Player_{team_id}_{900 + team_id}", career)
    stale = player_file_rows(fx23, 2023, 1, xg=0.9)
    world.vaastav("2023-24/understat/Player_1_901", stale)
    names = pd.DataFrame({"id": [901], "player_name": ["Dara O&#039;Shea"]})
    world.vaastav("2024-25/understat/understat_player", names)
    for team_id in range(1, 21):
        file = f"understat_{us_team(team_id).replace(' ', '_')}"
        world.vaastav(f"2023-24/understat/{file}", team_file_rows(fx23, team_id))
        # The newer folder repeats last season's rows (with other values) and adds its own.
        both = pd.concat([team_file_rows(fx23, team_id, xg=9.9), team_file_rows(fx24, team_id)])
        world.vaastav(f"2024-25/understat/{file}", both)
    return fx23, fx24


ALL = ["fixture", "gameweek", "player_season", "player_dim", "player_match", "understat"]


def test_understat_build_end_to_end(world):
    two_seasons(world)
    build([*ALL, "team_match"], world.ctx)

    upm = world.ctx.table("understat_player_match")
    assert upm["understat_id"].nunique() == 20
    assert len(upm) == 20 * 38 * 2  # no Barcelona or 2014 rows, no duplicates
    p1 = upm[upm["understat_id"] == 901]
    assert (p1["us_xg"] == 0.3).all()  # the newest folder wins over the stale 0.9 copy
    assert (p1["understat_name"] == "Dara O'Shea").all()
    assert upm.loc[upm["understat_id"] == 902, "understat_name"].iloc[0] == "Player 2"

    mapping = world.ctx.table("understat_map").set_index("understat_id")
    assert len(mapping) == 20 and (mapping["method"] == "auto").all()
    assert mapping.loc[901, "player_key"] == 100001  # minutes agree, name doesn't
    player_dim = world.ctx.table("player_dim").set_index("player_key")
    assert player_dim.loc[100005, "understat_id"] == 905

    pm = world.ctx.table("player_match")
    assert pm["us_minutes"].notna().all() and (pm["us_xg"] == 0.3).all()

    tm = world.ctx.table("team_match")
    assert len(tm) == 2 * 380 * 2
    t23 = tm[tm["season"] == 2023]
    assert (t23["us_xg"] == 1.2).all()  # own season's folder preferred over the newer copy
    assert t23["fd_xg"].isna().all() and t23["fpl_xg"].isna().all()
    t24 = tm[tm["season"] == 2024]
    assert (t24.loc[t24["is_home"], "fd_xg"] == 1.5).all()
    assert (t24.loc[~t24["is_home"], "fd_xg"] == 0.5).all()
    assert (t24.loc[~t24["is_home"], "fd_xga"] == 1.5).all()
    assert (t24["fpl_xg"] == 0.5).all() and (t24["fpl_xga"] == 0.5).all()
    assert (tm["available_at"] > tm["event_time"]).all()


def test_unmapped_player_in_covered_season_fails(world):
    two_seasons(world, skip_player=7)
    with pytest.raises(UnderstatMappingError, match="1 FPL player"):
        build(ALL, world.ctx)


def test_override_exempts_a_player_without_understat(world, tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "teams.csv").write_text((REPO_CONFIG / "teams.csv").read_text("utf-8"), "utf-8")
    (config / "overrides.csv").write_text(
        "source,source_id,season,player_key,note\nunderstat,,,100007,no Understat file\n",
        "utf-8",
    )
    world.ctx.config_dir = config
    two_seasons(world, skip_player=7)
    build(ALL, world.ctx)
    assert 100007 not in set(world.ctx.table("understat_map")["player_key"].dropna())


# --- mapping units ----------------------------------------------------------------------


def walker_inputs():
    """Kyle Walker (90') and Kyle Walker-Peters (60') play the same five fixtures."""
    fixtures = range(1, 6)
    fpl_apps = pd.DataFrame(
        {
            "player_key": [100] * 5 + [200] * 5,
            "fixture_key": [*fixtures, *fixtures],
            "season": 2021,
            "minutes": [90] * 5 + [60] * 5,
        }
    )
    upm = pd.DataFrame(
        {
            "understat_id": [10] * 5 + [20] * 5,
            "fixture_key": [*fixtures, *fixtures],
            "season": 2021,
            "us_minutes": [90] * 5 + [61] * 5,
        }
    )
    us_names = {10: "Kyle Walker", 20: "Kyle Walker-Peters"}
    fpl_names = {100: {"Kyle Walker-Peters", "Walker-Peters"}, 200: {"Kyle Walker", "Walker"}}
    return upm, fpl_apps, us_names, fpl_names


def scored_pairs(upm, fpl_apps, us_names, fpl_names):
    pairs = candidate_pairs(upm, fpl_apps)
    pairs["name_score"] = name_scores(pairs, us_names, fpl_names)
    return pairs


def test_mapping_prefers_minutes_over_similar_names():
    # FPL player 100 plays 90' (like Understat "Kyle Walker") but is *named* Walker-Peters:
    # the names are useless (token sets contain each other), the minutes decide.
    upm, fpl_apps, us_names, fpl_names = walker_inputs()
    pairs = scored_pairs(upm, fpl_apps, us_names, fpl_names)
    assert (pairs["name_score"] == 100).all()
    mapping = map_understat(pairs, EMPTY_OVERRIDES).set_index("understat_id")
    assert mapping.loc[10, "player_key"] == 100
    assert mapping.loc[20, "player_key"] == 200
    assert mapping.loc[10, "coverage"] == 1.0


def test_override_wins_and_can_unmap():
    upm, fpl_apps, us_names, fpl_names = walker_inputs()
    pairs = scored_pairs(upm, fpl_apps, us_names, fpl_names)
    overrides = pd.DataFrame(
        {
            "understat_id": pd.array([10, 20], dtype="Int64"),
            "player_key": pd.array([200, pd.NA], dtype="Int64"),
        }
    )
    mapping = map_understat(pairs, overrides).set_index("understat_id")
    assert mapping.loc[10, "player_key"] == 200 and mapping.loc[10, "method"] == "override"
    assert pd.isna(mapping.loc[20, "player_key"])


def test_single_appearance_with_minutes_off_maps_by_name():
    # A substitute keeper: Understat 24', FPL 35' — no close-minutes fixture, same name.
    fpl_apps = pd.DataFrame(
        {"player_key": [1, 2], "fixture_key": [7, 7], "season": 2021, "minutes": [35, 24]}
    )
    upm = pd.DataFrame(
        {"understat_id": [5], "fixture_key": [7], "season": 2021, "us_minutes": [24]}
    )
    pairs = scored_pairs(
        upm, fpl_apps, {5: "Kristoffer Klaesson"}, {1: {"Kristoffer Klaesson"}, 2: {"Rúben Neves"}}
    )
    mapping = map_understat(pairs, EMPTY_OVERRIDES)
    assert mapping["player_key"].tolist() == [1]


def test_mapping_problems_id_dict_contradiction_and_duplicates():
    upm, fpl_apps, us_names, fpl_names = walker_inputs()
    mapping = map_understat(scored_pairs(upm, fpl_apps, us_names, fpl_names), EMPTY_OVERRIDES)
    oracle = pd.DataFrame({"understat_id": [10, 20], "player_key": [100, 200], "season": 2021})
    assert mapping_problems(mapping, upm, fpl_apps, oracle) == []
    swapped = oracle.assign(player_key=[200, 100])
    problems = mapping_problems(mapping, upm, fpl_apps, swapped)
    assert len(problems) == 1 and "2 id_dict pair(s) contradicted" in problems[0]
    doubled = pd.concat([mapping, mapping.assign(understat_id=30)])
    assert any(
        "player_key mapped more than once" in p
        for p in mapping_problems(doubled, upm, fpl_apps, oracle)
    )


def test_read_overrides(tmp_path):
    (tmp_path / "overrides.csv").write_text(
        "source,source_id,season,player_key,note\n"
        "understat,123,,4567,forced\n"
        "understat,124,,,not in FPL\n"
        "understat,,,89,no Understat\n"
        "other,9,,9,ignored\n",
        "utf-8",
    )
    overrides = read_overrides(tmp_path)
    assert overrides["understat_id"].tolist()[:2] == [123, 124]
    assert pd.isna(overrides["understat_id"].iloc[2])
    assert overrides["player_key"].iloc[0] == 4567 and pd.isna(overrides["player_key"].iloc[1])
    assert len(read_overrides(tmp_path / "missing")) == 0


def test_normalise_name():
    assert normalise_name("N&#039;Golo Kanté") == "n golo kante"
    assert normalise_name("Martin Ødegaard") == "martin odegaard"
    assert normalise_name("Łukasz Fabiański") == "lukasz fabianski"


# --- team_match units -------------------------------------------------------------------


def test_understat_team_rows_fail_on_goal_mismatch():
    fixtures = season_fixtures(2023).head(1)  # team 1 (home) v team 20
    fixture = pd.DataFrame(
        {
            "fixture_key": [2023001],
            "season": [2023],
            "kickoff_time": pd.to_datetime(fixtures["kickoff_time"], utc=True).astype(
                "datetime64[us, UTC]"
            ),
            "available_at": [pd.Timestamp("2023-08-07 08:00", tz="UTC").as_unit("us")],
            "home_team_key": [TEAM_CODES[fixtures["team_h"].iloc[0] - 1]],
            "away_team_key": [TEAM_CODES[fixtures["team_a"].iloc[0] - 1]],
            "home_goals": pd.array(fixtures["team_h_score"], dtype="Int64"),
            "away_goals": pd.array(fixtures["team_a_score"], dtype="Int64"),
            "finished": [True],
        }
    )
    sides = team_sides(fixture)
    home_id = int(fixtures["team_h"].iloc[0])
    raw = team_file_rows(fixtures, home_id).assign(team_name=us_team(home_id), folder_season=2023)
    resolver = TeamResolver(team_dim_from_config(REPO_CONFIG / "teams.csv"))
    rows = understat_team_rows(raw, sides, resolver)
    assert rows["fixture_key"].tolist() == [2023001]
    with pytest.raises(UnderstatMappingError, match="disagree with the result"):
        understat_team_rows(raw.assign(scored=raw["scored"] + 1), sides, resolver)
    with pytest.raises(UnderstatMappingError, match="match no fixture"):
        understat_team_rows(raw.assign(date="2023-01-01 15:00:00"), sides, resolver)
