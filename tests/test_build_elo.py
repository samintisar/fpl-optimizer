import pandas as pd
import pytest
from synthetic_raw import World, football_data, season_fixtures

from fplopt.build import build
from fplopt.build.common import TableValidationError, validate_table
from fplopt.build.elo import (
    HOME_ADVANTAGE,
    INITIAL_RATING,
    K_FACTOR,
    SCHEMA,
    elo_ratings,
    goal_multiplier,
)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def utc(text):
    return pd.Timestamp(text, tz="UTC").as_unit("us")


def matches(rows, season=2010):
    """rows: (kickoff, home, away, home_goals, away_goals[, season])."""
    out = []
    for row in rows:
        kickoff, home, away, hg, ag = row[:5]
        out.append(
            {
                "season": row[5] if len(row) > 5 else season,
                "fixture_key": pd.NA,
                "kickoff_time": utc(kickoff),
                "home_team_key": home,
                "away_team_key": away,
                "home_goals": hg,
                "away_goals": ag,
            }
        )
    df = pd.DataFrame(out)
    return df.astype({"fixture_key": "Int64", "kickoff_time": "datetime64[us, UTC]"})


def played(out):
    """Match rows only (pre-season rows have no kickoff)."""
    return out[out["kickoff_time"].notna()]


def by_team(out):
    return played(out).set_index(["team_key", "kickoff_time"])


def expected_home(diff):
    return 1 / (1 + 10 ** (-(diff + HOME_ADVANTAGE) / 400))


def test_one_match_updates_both_sides_symmetrically():
    out = by_team(elo_ratings(matches([("2010-08-14T14:00", 3, 8, 1, 0)])))
    e = expected_home(0)
    delta = K_FACTOR * (1 - e)
    home, away = out.loc[(3, utc("2010-08-14T14:00"))], out.loc[(8, utc("2010-08-14T14:00"))]
    assert home["rating_before"] == away["rating_before"] == INITIAL_RATING
    assert home["rating_after"] == pytest.approx(INITIAL_RATING + delta)
    assert away["rating_after"] == pytest.approx(INITIAL_RATING - delta)
    assert home["expected_score"] == pytest.approx(e)
    assert away["expected_score"] == pytest.approx(1 - e)
    assert bool(home["is_home"]) and not bool(away["is_home"])
    assert home["opponent_team_key"] == 8 and away["opponent_team_key"] == 3
    assert home["available_at"] == utc("2010-08-14T16:00")


def test_draw_between_equal_teams_lowers_home_rating():
    out = by_team(elo_ratings(matches([("2010-08-14T14:00", 3, 8, 1, 1)])))
    assert out.loc[(3, utc("2010-08-14T14:00")), "rating_after"] < INITIAL_RATING
    assert out.loc[(8, utc("2010-08-14T14:00")), "rating_after"] > INITIAL_RATING


def test_goal_multiplier():
    assert [goal_multiplier(d) for d in (0, 1, -1, 2, -2, 3, 5)] == [
        1.0,
        1.0,
        1.0,
        1.5,
        1.5,
        14 / 8,
        2.0,
    ]
    out = by_team(elo_ratings(matches([("2010-08-14T14:00", 3, 8, 0, 3)])))
    gain = out.loc[(8, utc("2010-08-14T14:00")), "rating_after"] - INITIAL_RATING
    assert gain == pytest.approx(K_FACTOR * 14 / 8 * expected_home(0))


def test_ratings_carry_over_and_promoted_clubs_start_at_relegated_mean():
    rows = [
        # season 2010: clubs 1-4; 3 and 4 relegated
        ("2010-08-14T14:00", 1, 3, 3, 0),
        ("2010-08-14T14:00", 2, 4, 0, 1),
        ("2010-08-21T14:00", 3, 4, 2, 2),
        # season 2011: 1, 2 stay; 5 and 6 promoted
        ("2011-08-13T14:00", 1, 5, 0, 0, 2011),
        ("2011-08-13T14:00", 6, 2, 1, 0, 2011),
    ]
    out = played(elo_ratings(matches(rows)))
    end_2010 = out[out["season"] == 2010].sort_values("kickoff_time").groupby("team_key").last()
    relegated_mean = end_2010.loc[[3, 4], "rating_after"].mean()
    first_2011 = out[out["season"] == 2011].set_index("team_key")
    assert first_2011.loc[5, "rating_before"] == pytest.approx(relegated_mean)
    assert first_2011.loc[6, "rating_before"] == pytest.approx(relegated_mean)
    assert first_2011.loc[1, "rating_before"] == end_2010.loc[1, "rating_after"]
    assert first_2011.loc[2, "rating_before"] == end_2010.loc[2, "rating_after"]


def test_processing_order_is_kickoff_then_home_key():
    shuffled = matches([("2010-08-21T14:00", 3, 8, 0, 1), ("2010-08-14T14:00", 8, 3, 2, 0)]).iloc[
        ::-1
    ]
    a = elo_ratings(shuffled)
    b = elo_ratings(shuffled.iloc[::-1])
    pd.testing.assert_frame_equal(a, b)
    second = by_team(a).loc[(3, utc("2010-08-21T14:00"))]
    first = by_team(a).loc[(3, utc("2010-08-14T14:00"))]
    assert second["rating_before"] == first["rating_after"]


def test_schema_rejects_broken_continuity():
    out = elo_ratings(matches([("2010-08-14T14:00", 3, 8, 1, 0), ("2010-08-21T14:00", 8, 3, 1, 0)]))
    validate_table(out, "team_rating", SCHEMA)
    broken = out.copy()
    late = broken["kickoff_time"] == utc("2010-08-21T14:00")
    broken.loc[late, "rating_before"] += 1.0
    with pytest.raises(TableValidationError, match="continuous"):
        validate_table(broken, "team_rating", SCHEMA)


def test_build_team_rating_end_to_end(world):
    world.football_data(2015, football_data(season_fixtures(2015), 2015))
    world.add_vaastav_season(2016, fixtures_csv=False, teams_csv=False)
    build(["fixture", "team_rating"], world.ctx)

    rating = world.ctx.table("team_rating")
    pre_season = rating[rating["kickoff_time"].isna()]
    assert pre_season.groupby("season").size().to_dict() == {2015: 20, 2016: 20}
    rating = played(rating)
    assert rating.groupby("season").size().to_dict() == {2015: 760, 2016: 760}
    pre = rating[rating["season"] == 2015]
    assert pre["fixture_key"].isna().all()
    local = pre["kickoff_time"].dt.tz_convert("Europe/London")
    assert (local.dt.strftime("%H:%M") == "15:00").all()
    fixture = world.ctx.table("fixture").set_index("fixture_key")
    post = rating[rating["season"] == 2016]
    assert post["fixture_key"].notna().all()
    assert (post["kickoff_time"] == post["fixture_key"].map(fixture["kickoff_time"])).all()
    assert (rating["event_time"] == rating["kickoff_time"]).all()
    # same 20 clubs both seasons: 2016 starts from 2015's end ratings
    end_2015 = pre.sort_values("kickoff_time").groupby("team_key")["rating_after"].last()
    start_2016 = post.sort_values("kickoff_time").groupby("team_key")["rating_before"].first()
    pd.testing.assert_series_equal(end_2015, start_2016, check_names=False)


def test_pre_season_rows_carry_the_season_start_rating():
    """One row per club and season before its first match: rating_before = rating_after =
    the seeded (promoted) or carried-over rating, no fixture fields, known from 1 June."""
    rows = [
        ("2010-08-14T14:00", 1, 3, 3, 0),
        ("2010-08-14T14:00", 2, 4, 0, 1),
        ("2010-08-21T14:00", 3, 4, 2, 2),
        ("2011-08-13T14:00", 1, 5, 0, 0, 2011),
        ("2011-08-13T14:00", 6, 2, 1, 0, 2011),
    ]
    # Club 7 is in the 2011 season but has not played yet (current season).
    out = elo_ratings(matches(rows), {2011: {1, 2, 5, 6, 7}})
    validate_table(out, "team_rating", SCHEMA)
    pre = out[out["kickoff_time"].isna()].set_index(["season", "team_key"])
    assert sorted(pre.index) == [(2010, t) for t in (1, 2, 3, 4)] + [
        (2011, t) for t in (1, 2, 5, 6, 7)
    ]
    assert (pre["rating_before"] == pre["rating_after"]).all()
    assert pre["fixture_key"].isna().all() and pre["opponent_team_key"].isna().all()
    assert pre["is_home"].isna().all() and pre["expected_score"].isna().all()
    assert (pre["event_time"] == pre["available_at"]).all()
    assert (pre.loc[2010, "available_at"] == utc("2010-06-01")).all()
    assert (pre.loc[2011, "available_at"] == utc("2011-06-01")).all()
    assert (pre.loc[2010, "rating_after"] == INITIAL_RATING).all()

    games = by_team(out)
    end_2010 = played(out[out["season"] == 2010]).groupby("team_key")["rating_after"].last()
    relegated_mean = end_2010.loc[[3, 4]].mean()
    for club in (5, 6, 7):
        assert pre.loc[(2011, club), "rating_after"] == pytest.approx(relegated_mean)
    for club in (1, 2):
        assert pre.loc[(2011, club), "rating_after"] == end_2010.loc[club]
    # The first match continues from the pre-season row.
    assert (
        games.loc[(5, utc("2011-08-13T14:00")), "rating_before"]
        == pre.loc[(2011, 5), "rating_after"]
    )


def test_pre_season_rows_wait_for_the_previous_season_to_end():
    """A season that ends after 1 June (2019/20 ended in July 2020) delays the next season's
    start ratings until its last result is in."""
    rows = [
        ("2010-08-14T14:00", 1, 2, 1, 0),
        ("2011-07-20T18:00", 2, 1, 2, 0),  # same season, played after 1 June 2011
        ("2011-09-12T14:00", 1, 2, 0, 0, 2011),
    ]
    out = elo_ratings(matches(rows))
    validate_table(out, "team_rating", SCHEMA)
    pre = out[out["kickoff_time"].isna() & (out["season"] == 2011)]
    assert len(pre) == 2
    assert (pre["available_at"] == utc("2011-07-20T20:00")).all()
    assert (pre["event_time"] == pre["available_at"]).all()


def test_schema_rejects_a_pre_season_row_that_moves_the_rating():
    out = elo_ratings(matches([("2010-08-14T14:00", 3, 8, 1, 0)]))
    broken = out.copy()
    pre = broken["kickoff_time"].isna() & (broken["team_key"] == 3)
    broken.loc[pre, "rating_after"] += 1.0
    with pytest.raises(TableValidationError):
        validate_table(broken, "team_rating", SCHEMA)
