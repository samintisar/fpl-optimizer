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


def by_team(out):
    return out.set_index(["team_key", "kickoff_time"])


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
    out = elo_ratings(matches(rows))
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
