"""Minutes-model history features (`fplopt.features.history`): start inference, lags from
earlier fixtures only, targets, suspensions, and the prediction frame."""

import numpy as np
import pandas as pd
import pytest
from synthetic_season import espn_tables, synthetic_tables

from fplopt.build import espn
from fplopt.features.history import (
    LAG_FEATURES,
    PREDICTION_COLUMNS,
    TARGETS,
    infer_starts,
    prediction_frame,
    training_frame,
)
from fplopt.features.store import DataStore

UTC_US = pd.DatetimeTZDtype("us", "UTC")
SEASON = 2022
FIRST_DEADLINE = pd.Timestamp("2022-08-05 17:30", tz="UTC").as_unit("us")


def deadline(gw: int, season: int = SEASON) -> pd.Timestamp:
    return FIRST_DEADLINE + pd.DateOffset(years=season - SEASON) + pd.Timedelta(days=7 * (gw - 1))


def league(rows: list[dict], n_gws: int = 12, seasons: tuple[int, ...] = (SEASON,)) -> dict:
    """In-memory tables for `training_frame`: one fixture per (season, gw, team) unless a row
    gives `fixture` (a double: give the second one `fixture` and `day`). Rows: player, team,
    gw, minutes; optional season, starts (None = unknown), yellow, red, element_type, value,
    fixture, day (kickoff = deadline + day days, default 1)."""
    gameweeks = pd.DataFrame(
        [
            {"season": s, "gw": gw, "gw_index": gw, "deadline_time": deadline(gw, s)}
            for s in seasons
            for gw in range(1, n_gws + 1)
        ]
    )
    gameweeks["deadline_time"] = gameweeks["deadline_time"].astype(UTC_US)
    gameweeks = gameweeks.assign(
        event_time=gameweeks["deadline_time"],
        available_at=pd.Timestamp("2022-06-01", tz="UTC").as_unit("us"),
    )
    matches, registered = [], []
    for r in rows:
        season = r.get("season", SEASON)
        at = deadline(r["gw"], season)
        kickoff = at + pd.Timedelta(days=r.get("day", 1))
        fixture = r.get("fixture", season * 10_000 + r["gw"] * 100 + r["team"])
        matches.append(
            {
                "player_key": r["player"],
                "season": season,
                "fixture_key": fixture,
                "gw": r["gw"],
                "team_key": r["team"],
                "kickoff_time": kickoff,
                "minutes": r["minutes"],
                "starts": r.get("starts"),
                "yellow_cards": r.get("yellow", 0),
                "red_cards": r.get("red", 0),
                "event_time": kickoff,
                "available_at": at + pd.Timedelta(days=4),
            }
        )
        registered.append(
            {
                "player_key": r["player"],
                "season": season,
                "gw": r["gw"],
                "team_key": r["team"],
                "element_type": r.get("element_type", 3),
                "value": r.get("value", 50),
                "event_time": at,
                "available_at": at - pd.Timedelta(hours=1),
            }
        )
    player_match = pd.DataFrame(matches)
    player_match["starts"] = pd.array(player_match["starts"], dtype="Int64")
    for column in ("kickoff_time", "event_time", "available_at"):
        player_match[column] = player_match[column].astype(UTC_US)
    player_gw = pd.DataFrame(registered).drop_duplicates(["player_key", "season", "gw"])
    for column in ("event_time", "available_at"):
        player_gw[column] = player_gw[column].astype(UTC_US)
    first = player_gw.sort_values("available_at").drop_duplicates(["player_key", "season"])
    player_season = first[["player_key", "season", "element_type", "event_time", "available_at"]]
    return {
        "gameweek": gameweeks,
        "player_match": player_match,
        "player_gw": player_gw.reset_index(drop=True),
        "player_season": player_season.reset_index(drop=True),
        **espn_tables(),
    }


def frame_at(tables: dict, gw: int, season: int = SEASON) -> pd.DataFrame:
    return training_frame(DataStore(tables=tables).as_of(deadline(gw, season)))


def row(frame: pd.DataFrame, player: int, gw: int, season: int = SEASON) -> pd.Series:
    found = frame[
        (frame["player_key"] == player) & (frame["gw"] == gw) & (frame["season"] == season)
    ]
    assert len(found) == 1
    return found.iloc[0]


def team(team_key: int, gw: int, minutes: list[int], first_player: int = 1, **extra) -> list[dict]:
    return [
        {"player": first_player + i, "team": team_key, "gw": gw, "minutes": m, **extra}
        for i, m in enumerate(minutes)
    ]


# --- start inference ----------------------------------------------------------------------


def test_the_eleven_players_with_most_minutes_start_ties_by_player_key():
    minutes = [90] * 9 + [45, 45, 45, 20, 0]  # players 10-12 tie for the last two places
    frame = pd.DataFrame(
        {"fixture_key": 1, "team_key": 1, "player_key": range(1, 15), "minutes": minutes}
    )
    started = infer_starts(frame)
    assert started.tolist() == [True] * 11 + [False] * 3
    shuffled = frame.sample(frac=1, random_state=3)
    assert infer_starts(shuffled).equals(started.loc[shuffled.index])


def test_with_fewer_than_eleven_on_the_pitch_only_they_start():
    frame = pd.DataFrame(
        {"fixture_key": 1, "team_key": 7, "player_key": range(1, 15), "minutes": [90] * 9 + [0] * 5}
    )
    assert infer_starts(frame).tolist() == [True] * 9 + [False] * 5


def test_red_cards_rank_by_minutes_like_everyone_else():
    # Player 11 is sent off after 35 minutes: still in the top 11 (subs played 25 and 10).
    frame = pd.DataFrame(
        {
            "fixture_key": [1] * 13 + [2] * 13,
            "team_key": 1,
            "player_key": list(range(1, 14)) * 2,
            # Fixture 2: sent off after 15 minutes while a sub played 30, the known error.
            "minutes": [90] * 10 + [35, 25, 10] + [90] * 10 + [15, 30, 10],
        }
    )
    started = infer_starts(frame).to_numpy()
    assert started[:13].tolist() == [True] * 11 + [False] * 2
    assert started[13:].tolist() == [True] * 10 + [False, True, False]


def test_real_starts_win_over_the_inference():
    rows = team(1, 1, [90] * 10 + [45, 45])
    rows[10]["starts"], rows[11]["starts"] = 0, 1  # the real starter has the higher key
    for r in rows[:10]:
        r["starts"] = 1
    frame = frame_at(league(rows), 2)
    assert frame.sort_values("player_key")["start"].tolist() == [1.0] * 10 + [0.0, 1.0]
    assert not frame["start_inferred"].any()
    inferred = frame_at(league(team(1, 1, [90] * 10 + [45, 45])), 2)
    assert inferred.sort_values("player_key")["start"].tolist() == [1.0] * 11 + [0.0]
    assert inferred["start_inferred"].all()


# --- lags ---------------------------------------------------------------------------------


def history_rows() -> list[dict]:
    """Player 1 (team 1) over GW1-8 with minutes; team-mates 2-12 always play 90, so player
    1 starts when he plays 90 (ties go to the lowest key) and is a sub otherwise."""
    minutes = [90, 90, 0, 0, 0, 30, 90, 70]
    rows = []
    for gw, m in enumerate(minutes, start=1):
        rows.append({"player": 1, "team": 1, "gw": gw, "minutes": m, "yellow": int(gw in (1, 7))})
        rows += [{"player": p, "team": 1, "gw": gw, "minutes": 90} for p in range(2, 13)]
    return rows


def test_lags_use_only_earlier_gameweeks():
    frame = frame_at(league(history_rows()), 12)
    gw7 = row(frame, 1, 7)  # earlier: 90, 90, 0, 0, 0, 30 (GW6: a sub, 30 minutes)
    assert gw7["start_rate_l1"] == 0.0
    assert gw7["minutes_l1"] == 30.0
    assert gw7["start_rate_l3"] == 0.0
    assert gw7["minutes_l3"] == pytest.approx(10.0)
    assert gw7["start_rate_l5"] == pytest.approx(1 / 5)
    assert gw7["start_rate_l10"] == pytest.approx(2 / 6)
    assert gw7["minutes_l10"] == pytest.approx(210 / 6)
    assert gw7["n_long"] == 6
    assert gw7["absent_run"] == 0
    assert gw7["returned"] == 1  # GW6: an appearance after 3 games without minutes
    assert gw7["yellows_season"] == 1
    assert gw7["days_since_app"] == pytest.approx(6.0)  # GW6 kickoff to the GW7 deadline
    assert row(frame, 1, 6)["absent_run"] == 3
    assert row(frame, 1, 6)["returned"] == 0
    assert row(frame, 1, 8)["yellows_season"] == 2
    first = row(frame, 1, 1)
    assert np.isnan(first["start_rate_l1"]) and np.isnan(first["days_since_app"])
    assert first["n_long"] == 0


def test_club_continuity_is_the_share_of_starters_kept():
    rows = []
    for gw in (1, 2, 3):
        rows += team(1, gw, [90] * 11 + [0])  # the same XI every week
    rows += team(2, 1, [90] * 11 + [0] * 4, first_player=21)  # 21-31 start
    rows += team(2, 2, [0] * 4 + [90] * 11, first_player=21)  # 25-35: 7 kept
    rows += team(2, 3, [90] * 15, first_player=21)
    frame = frame_at(league(rows), 6)
    assert np.isnan(row(frame, 1, 2)["club_continuity"])  # one earlier fixture only
    assert row(frame, 1, 3)["club_continuity"] == pytest.approx(1.0)
    assert row(frame, 21, 3)["club_continuity"] == pytest.approx(7 / 11)
    assert row(frame, 35, 3)["club_continuity"] == pytest.approx(7 / 11)


def test_club_fixtures_season_counts_the_clubs_earlier_fixtures():
    frame = frame_at(league(history_rows()), 12)
    assert [row(frame, 1, gw)["club_fixtures_season"] for gw in (1, 2, 7)] == [0, 1, 6]
    assert row(frame, 5, 8)["club_fixtures_season"] == 7


def test_a_planted_future_row_changes_no_earlier_feature():
    rows = history_rows()
    base = frame_at(league(rows), 12)
    planted = rows + [{"player": 1, "team": 1, "gw": 10, "minutes": 90, "yellow": 1, "red": 1}]
    planted += [{"player": 13, "team": 1, "gw": 9, "minutes": 90}]
    later = frame_at(league(planted), 12)
    columns = [*LAG_FEATURES, *TARGETS]
    earlier = later[later["gw"] <= 8].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier[columns], base[columns])


def test_only_visible_rows_are_used():
    frame = frame_at(league(history_rows()), 5)  # GW5's deadline: GW1-4 played and visible
    assert sorted(frame["gw"].unique()) == [1, 2, 3, 4]


def test_both_fixtures_of_a_double_see_the_history_as_of_the_deadline():
    rows = [{"player": 1, "team": 1, "gw": 1, "minutes": 90}]
    rows += [
        {"player": 1, "team": 1, "gw": 2, "minutes": 0, "fixture": 21},
        {"player": 1, "team": 1, "gw": 2, "minutes": 90, "fixture": 22, "day": 4},
    ]
    frame = frame_at(league(rows), 4)
    second = frame[frame["fixture_key"] == 22].iloc[0]
    first = frame[frame["fixture_key"] == 21].iloc[0]
    for column in ("start_rate_l1", "minutes_l1", "n_long", "absent_run", "days_since_app"):
        assert second[column] == first[column]
    assert second["minutes_l1"] == 90.0
    assert second["rest_days"] == pytest.approx(3.0)  # the club's previous fixture


def test_targets():
    rows = team(1, 1, [90] * 10 + [59, 30, 0])
    frame = frame_at(league(rows), 2).sort_values("player_key")
    assert frame["minutes"].tolist() == [90.0] * 10 + [59.0, 30.0, 0.0]
    assert frame["start"].tolist() == [1.0] * 11 + [0.0, 0.0]
    assert frame["sixty"].tolist() == [1.0] * 10 + [0.0, 0.0, 0.0]
    assert frame["sub"].tolist() == [0.0] * 11 + [1.0, 0.0]


def test_start_shares_rest_competition_and_club_spells():
    rows = []
    for gw in range(1, 6):
        rows += team(1, gw, [90] * 11 + [10 * (gw % 2)], element_type=2)  # 11 DEF start
        rows += team(2, gw, [90] * 11, first_player=101, element_type=2)
    # Player 12 moves from team 1 to team 2 at GW4.
    for r in rows:
        if r["player"] == 12 and r["gw"] >= 4:
            r["team"] = 2
    last_season = [
        {"player": 1, "team": 1, "gw": gw, "minutes": 90, "season": 2021} for gw in range(1, 11)
    ]
    tables = league(rows + last_season, seasons=(2021, SEASON))
    frame = frame_at(tables, 8)
    gw5 = row(frame, 1, 5)
    assert gw5["start_share_season"] == 1.0
    assert gw5["start_share_last"] == pytest.approx(10 / 38)
    assert np.isnan(row(frame, 2, 5)["start_share_last"])
    assert np.isnan(row(frame, 1, 1)["start_share_season"])  # the club's first fixture
    assert gw5["rest_days"] == pytest.approx(7.0)
    assert row(frame, 1, 1)["rest_days"] == 30.0  # capped (last season)
    assert gw5["comp_start_sum"] == pytest.approx(10.0)  # 10 DEF team-mates always start
    assert gw5["comp_rank"] == 1.0
    assert row(frame, 12, 5)["comp_rank"] == 12.0
    assert row(frame, 12, 3)["fixtures_at_club"] == 2
    assert row(frame, 12, 4)["fixtures_at_club"] == 0  # a new signing
    assert row(frame, 12, 5)["fixtures_at_club"] == 1
    assert row(frame, 12, 5)["start_share_season"] == 0.0  # 0 starts in team 2's 4 fixtures


# --- suspensions --------------------------------------------------------------------------


def card_rows(cards: dict[int, tuple[int, int]], n_gws: int = 34) -> list[dict]:
    """Player 1 starts every GW; cards = {gw: (yellows, reds)}."""
    out = []
    for gw in range(1, n_gws + 1):
        yellow, red = cards.get(gw, (0, 0))
        out.append({"player": 1, "team": 1, "gw": gw, "minutes": 90, "yellow": yellow, "red": red})
    return out


@pytest.mark.parametrize(
    ("cards", "banned_gws"),
    [
        ({3: (1, 0), 5: (1, 0), 8: (1, 0), 10: (1, 0), 12: (1, 0)}, [13]),  # 5th yellow, GW12
        ({3: (2, 0), 10: (3, 0)}, [11]),  # 5 by GW10
        ({20: (5, 0)}, []),  # the 5th yellow after GW19: no ban
        ({2: (4, 0), 19: (1, 0), 25: (5, 0)}, [20, 26, 27]),  # 5 by GW19, 10 by GW32
        ({7: (0, 1)}, [8]),  # red card: one match
        ({33: (15, 0)}, [34]),  # 15 yellows: three matches (only GW34 is left)
    ],
)
def test_suspensions(cards, banned_gws):
    # Without ESPN data a red card is one certain match; yellow bans are certain.
    frame = frame_at(league(card_rows(cards), n_gws=40), 40)
    assert frame.loc[frame["ban_kind"] == 2, "gw"].tolist() == banned_gws
    assert (frame["ban_kind"] != 1).all()
    assert frame["banned"].equals((frame["ban_kind"] > 0).astype("float64"))
    assert frame["ban_kind"].dtype == "float64"


def test_a_ban_carries_into_the_next_season():
    rows = [{**r, "season": 2021} for r in card_rows({34: (0, 1)})]
    rows += [{**r, "season": 2022} for r in card_rows({}, n_gws=3)]
    frame = frame_at(league(rows, n_gws=40, seasons=(2021, 2022)), 5)
    assert frame.loc[frame["banned"] == 1, ["season", "gw"]].values.tolist() == [[2022, 1]]


# --- the prediction frame -----------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    return synthetic_tables(seasons=(2023,), n_clubs=6, seed=4)


def test_prediction_frame_has_one_row_per_pool_player_and_horizon_fixture(synthetic):
    gameweeks = synthetic["gameweek"]
    at = gameweeks.loc[gameweeks["gw"] == 4, "deadline_time"].iloc[0]
    view = DataStore(tables=synthetic).as_of(at)
    frame = prediction_frame(view)
    assert list(frame.columns) == list(PREDICTION_COLUMNS)
    assert frame.index.equals(pd.RangeIndex(len(frame)))
    pool = synthetic["player_snapshot"]
    n_players = pool.loc[pool["snapshot_at"] < at, "player_key"].nunique()
    assert len(frame) == n_players * 6  # no blanks or doubles: one fixture per GW, h 0-5
    assert sorted(frame["horizon"].unique()) == [0, 1, 2, 3, 4, 5]
    assert frame.equals(prediction_frame(view))


def test_prediction_lags_match_the_training_rows_of_that_gameweek(synthetic):
    gameweeks = synthetic["gameweek"]
    store = DataStore(tables=synthetic)
    for gw in (2, 6, 9):
        at = gameweeks.loc[gameweeks["gw"] == gw, "deadline_time"].iloc[0]
        predicted = prediction_frame(store.as_of(at))
        predicted = predicted[predicted["horizon"] == 0].set_index("player_key")
        later = training_frame(store.as_of(gameweeks["deadline_time"].max()))
        trained = later[later["gw"] == gw].set_index("player_key")
        common = predicted.index.intersection(trained.index)
        assert len(common) > 50
        history_only = [c for c in LAG_FEATURES if c not in ("price",)]
        pd.testing.assert_frame_equal(
            predicted.loc[common, history_only], trained.loc[common, history_only]
        )


def prediction_tables(cards: dict[int, tuple[int, int]]) -> dict:
    """`league` tables of player 1 (team 1, cards = {gw: (yellows, reds)}) up to GW5, with
    team 1's schedule (one fixture per GW) and his GW6 registration: predict at GW6."""
    rows = card_rows(cards, n_gws=5)
    tables = league(rows, n_gws=12)
    gw6 = (
        tables["player_gw"]
        .iloc[[-1]]
        .assign(
            gw=6,
            event_time=deadline(6),
            available_at=deadline(6) - pd.Timedelta(hours=1),
        )
    )
    tables["player_gw"] = pd.concat([tables["player_gw"], gw6], ignore_index=True)
    gameweeks = tables["gameweek"]
    schedule = pd.DataFrame(
        {
            "fixture_key": [SEASON * 10_000 + gw * 100 + 1 for gw in range(1, 13)],
            "season": SEASON,
            "gw": range(1, 13),
            "gw_index": range(1, 13),
            "kickoff_time": [deadline(gw) + pd.Timedelta(days=1) for gw in range(1, 13)],
            "home_team_key": 1,
            "away_team_key": 2,
            "schedule_source": "final",
        }
    )
    schedule["kickoff_time"] = schedule["kickoff_time"].astype(UTC_US)
    schedule = schedule.assign(
        event_time=schedule["kickoff_time"], available_at=gameweeks["available_at"].iloc[0]
    )
    snapshot = synthetic_tables(seasons=(2023,), n_clubs=2)["player_snapshot"].iloc[:0]
    fixture_snapshot = synthetic_tables(seasons=(2023,), n_clubs=2)["fixture_snapshot"]
    return {
        **tables,
        "schedule": schedule,
        "player_snapshot": snapshot,
        "fixture_snapshot": fixture_snapshot,
    }


def test_a_ban_in_the_prediction_frame():
    tables = prediction_tables({5: (0, 1)})
    frame = prediction_frame(DataStore(tables=tables).as_of(deadline(6)))
    mine = frame[frame["player_key"] == 1]
    assert mine["ban_kind"].tolist() == [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert mine["reds_l3"].iloc[0] == 1.0


GW5_FIXTURE = SEASON * 10_000 + 5 * 100 + 1


def sent_off(tables: dict, second_yellow: bool, cup_days: list[float]) -> dict:
    """`tables` plus player 1's red card in GW5 on ESPN (mapped; `second_yellow`) and an
    EFL Cup match of team 1 `days` after the GW5 deadline per entry of `cup_days` (known 10
    days before kickoff)."""
    kickoff = deadline(5) + pd.Timedelta(days=1)
    times = {"event_time": kickoff, "available_at": kickoff + pd.Timedelta(hours=3)}
    match = {
        "espn_event_id": 9001,
        "league": "eng.1",
        "season": SEASON,
        "kickoff_time": kickoff,
        "home_espn_id": "9001",
        "away_espn_id": "9002",
        "home_team_key": 1,
        "away_team_key": 2,
        "fixture_key": GW5_FIXTURE,
        "status": "STATUS_FULL_TIME",
        "extra_time": False,
        **times,
    }
    player = {
        "espn_athlete_id": 501,
        "espn_event_id": 9001,
        "season": SEASON,
        "league": "eng.1",
        "team_key": 1,
        "athlete_name": "Player One",
        "starter": True,
        "minute_on": 0,
        "minute_off": 70,
        "minutes": 70,
        "yellows": 2 if second_yellow else 0,
        "red": True,
        "second_yellow": second_yellow,
        **times,
    }
    mapping = espn.empty_map()
    mapping = pd.concat(
        [mapping, pd.DataFrame({"espn_athlete_id": [501], "player_key": [1], **times})],
        ignore_index=True,
    ).astype(mapping.dtypes.to_dict())
    cups = []
    for i, days in enumerate(cup_days):
        cup_kickoff = deadline(5) + pd.Timedelta(days=days)
        cups.append(
            {
                "espn_event_id": 9100 + i,
                "league": "eng.league_cup",
                "season": SEASON,
                "kickoff_time": cup_kickoff,
                "home_team_key": 1,
                "away_team_key": None,
                "played": cup_kickoff < deadline(6),
                "event_time": cup_kickoff,
                "available_at": cup_kickoff - pd.Timedelta(days=10),
            }
        )
    return {
        **tables,
        "espn_match": espn._frame([match], espn.MATCH_SCHEMA),
        "espn_player_match": espn._frame([player], espn.PLAYER_SCHEMA),
        "espn_map": mapping,
        "club_match": espn._frame(cups, espn.CLUB_MATCH_SCHEMA),
    }


def predicted_kinds(tables: dict) -> list[float]:
    frame = prediction_frame(DataStore(tables=tables).as_of(deadline(6)))
    return frame.loc[frame["player_key"] == 1, "ban_kind"].tolist()


@pytest.mark.parametrize(
    ("second_yellow", "cup_days", "kinds"),
    [
        (True, [], [2, 0, 0, 0, 0, 0]),
        (True, [4], [0] * 6),  # the cup tie between GW5 and GW6 serves the ban
        (False, [], [2, 1, 1, 0, 0, 0]),  # straight red: 1 certain + 2 likely
        (False, [4], [1, 1, 0, 0, 0, 0]),
        (False, [4, 9], [1, 0, 0, 0, 0, 0]),  # a scheduled cup match after the deadline
        (False, [4, 30], [1, 1, 0, 0, 0, 0]),  # ... after the first two horizon fixtures
        (False, [9], [2, 1, 0, 0, 0, 0]),  # between the first two horizon fixtures
    ],
)
def test_a_domestic_cup_match_serves_a_red_card_ban(second_yellow, cup_days, kinds):
    tables = sent_off(
        prediction_tables({5: (1 if second_yellow else 0, 1)}), second_yellow, cup_days
    )
    assert predicted_kinds(tables) == [float(k) for k in kinds]


def test_a_cup_match_not_yet_known_does_not_serve_the_ban():
    tables = sent_off(prediction_tables({5: (0, 1)}), False, [9])
    cups = tables["club_match"]
    tables["club_match"] = cups.assign(available_at=deadline(6) + pd.Timedelta(hours=1))
    assert predicted_kinds(tables) == [2.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_training_rows_carry_the_served_ban():
    tables = sent_off(league(card_rows({5: (1, 1)}, n_gws=8), n_gws=12), False, [4])
    frame = frame_at(tables, 9)
    mine = frame[frame["player_key"] == 1].set_index("gw")
    assert mine.loc[6:8, "ban_kind"].tolist() == [1.0, 1.0, 0.0]
    assert mine.loc[6:8, "banned"].tolist() == [1.0, 1.0, 0.0]
    without = frame_at(sent_off(league(card_rows({5: (1, 1)}, n_gws=8), n_gws=12), False, []), 9)
    mine = without[without["player_key"] == 1].set_index("gw")
    assert mine.loc[5:9, "ban_kind"].tolist() == [0.0, 2.0, 1.0, 1.0]
