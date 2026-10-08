"""Player goal/assist shares and penalties (`fplopt.models.shares`) on hand-made rows and
the synthetic league."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.features.store import DataStore
from fplopt.models.gbm import GbmParams
from fplopt.models.minutes import MinutesParams, fit_minutes, predict_minutes
from fplopt.models.shares import (
    FPL_PEN_XG,
    SHARES_COLUMNS,
    SharesFit,
    SharesParams,
    fit_shares,
    marcel_shares,
    match_xg,
    predict_shares,
    taker_probabilities,
)
from fplopt.models.team import fit_team, team_lambdas

SMALL = GbmParams(num_boost_round=20, min_data_in_leaf=20)
MINUTES = MinutesParams(start=SMALL, sixty=SMALL, sub=SMALL)
SEASON, GW = 2023, 5


@pytest.fixture(scope="module")
def league():
    return synthetic_tables(seasons=(2022, 2023), n_clubs=6, seed=3)


def deadline(tables, season=SEASON, gw=GW):
    gameweeks = tables["gameweek"]
    match = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return match["deadline_time"].iloc[0]


def view(tables, season=SEASON, gw=GW):
    return DataStore(tables=tables).as_of(deadline(tables, season, gw))


@pytest.fixture(scope="module")
def inputs(league):
    """The minutes and team frames at the test deadline (computed once; the shares tests
    change only what shares read)."""
    v = view(league)
    minutes = predict_minutes(v, fit_minutes(v, MINUTES))
    team = team_lambdas(v, fit_team(v))
    return minutes, team


def predict(tables, inputs, params=None):
    v = view(tables)
    fit = fit_shares(v, params)
    return fit, predict_shares(v, fit, *inputs)


def copy(tables):
    return {name: df.copy() for name, df in tables.items()}


# --- npxG sources ---------------------------------------------------------------------------


def source_frame(**columns):
    base = {
        "minutes": 90.0,
        "goals_scored": 0.0,
        "penalties_missed": 0.0,
        "fpl_xg": np.nan,
        "fpl_xa": np.nan,
        "us_goals": np.nan,
        "us_npg": np.nan,
        "us_npxg": np.nan,
        "us_xa": np.nan,
        "main_taker": False,
    }
    base.update(columns)
    n = max(len(v) for v in columns.values() if isinstance(v, list))
    return pd.DataFrame({k: v if isinstance(v, list) else [v] * n for k, v in base.items()})


def test_npxg_comes_from_understat_then_fpl_then_nothing():
    frame = source_frame(
        us_npxg=[0.3, np.nan, np.nan, np.nan],
        us_goals=[2.0, np.nan, np.nan, np.nan],
        us_npg=[1.0, np.nan, np.nan, np.nan],
        us_xa=[0.2, np.nan, np.nan, np.nan],
        fpl_xg=[9.0, 1.1, 1.1, np.nan],
        fpl_xa=[9.0, 0.4, np.nan, np.nan],
        goals_scored=[2.0, 1.0, 1.0, 0.0],
        penalties_missed=[1.0, 0.0, 0.0, 0.0],
        main_taker=[False, False, True, False],
    )
    out = match_xg(frame, k_goals=1.2, k_assists=1.5, pen_pi=(0.6, 0.1))
    assert list(out["xg_source"]) == ["understat", "fpl", "fpl", "none"]
    assert list(out["xa_source"]) == ["understat", "fpl", "none", "none"]
    # Understat: its own npxG; penalty goals us_goals - us_npg, attempts + the miss.
    assert out["npxg"].iloc[0] == 0.3 and out["xa"].iloc[0] == 0.2
    assert out["pen_goals"].iloc[0] == 1 and out["pen_attempts"].iloc[0] == 2
    # FPL: one candidate (1.1 >= 0.79 and a goal), times pi by the main-taker flag.
    for row, pi in ((1, 0.1), (2, 0.6)):
        assert out["pen_goals"].iloc[row] == pytest.approx(pi)
        assert out["npxg"].iloc[row] == pytest.approx(1.2 * (1.1 - FPL_PEN_XG * pi))
    assert out["xa"].iloc[1] == pytest.approx(0.6)
    assert np.isnan(out["npxg"].iloc[3]) and np.isnan(out["xa"].iloc[3])
    assert np.isnan(out["pen_attempts"].iloc[3])


def test_penalty_candidates_need_the_xg_and_a_goal():
    frame = source_frame(
        fpl_xg=[0.5, 1.6, 1.6, 1.6, 2.4],
        goals_scored=[1.0, 2.0, 0.0, 2.0, 1.0],
        penalties_missed=[0.0, 0.0, 0.0, 1.0, 0.0],
        main_taker=True,
    )
    out = match_xg(frame, pen_pi=(1.0, 1.0))
    # 0.5 < 0.79: none; 1.6: two; no goal: none; 1.6 with a miss: one; 2.4 but one goal: one.
    np.testing.assert_allclose(out["pen_goals"], [0, 2, 0, 1, 1])
    np.testing.assert_allclose(out["pen_attempts"], [0, 2, 0, 2, 1])
    np.testing.assert_allclose(
        out["npxg"], np.clip([0.5, 1.6 - 1.58, 1.6, 1.6 - 1.58, 1.61], 0, None)
    )


# --- penalty takers -------------------------------------------------------------------------


def test_taker_probabilities_follow_the_order_and_sum_to_one():
    groups = np.array([1, 1, 1, 1, 2, 2, 2])
    rank = np.array([1.0, 2.0, np.nan, np.nan, 1.0, 2.0, np.nan])
    q = np.array([0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.0])
    f = np.array([1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0])  # club 2's first choice is out
    g = np.array([0.2, 0.1, 0.3, 0.4, 0.5, 0.1, 0.3])
    p = taker_probabilities(groups, rank, q, f, g)
    leftover = 0.2 * 0.2
    spread = g[:4] * f[:4] / (g[:4] * f[:4]).sum()
    np.testing.assert_allclose(p[:4], [0.8, 0.2 * 0.8, 0, 0] + leftover * spread)
    assert p[4] == 0  # off the pitch: never takes one
    np.testing.assert_allclose(p[5], 0.8 + 0.2 * 0.1 / 0.4)
    np.testing.assert_allclose([p[:4].sum(), p[4:].sum()], [1.0, 1.0])
    # Row order doesn't matter.
    order = np.array([6, 2, 0, 5, 3, 1, 4])
    shuffled = taker_probabilities(groups[order], rank[order], q[order], f[order], g[order])
    np.testing.assert_allclose(shuffled, p[order])


# --- the Marcel prior -----------------------------------------------------------------------


def toy_fit(**params):
    return SharesFit(
        cutoff=pd.Timestamp("2023-08-01", tz="UTC"),
        params=SharesParams(**params),
        k_goals=1.0,
        k_assists=1.0,
        pen_pi=(0.5, 0.2),
        ratio_goals=1.0,
        ratio_assists=1.0,
        ratio_team=1.0,
        team_output=1.5,
        position_goal=((3, 0.1), (4, 0.3)),
        position_assist=((3, 0.12), (4, 0.08)),
        price_goal=((3, np.log(0.1), 0.5, 60.0), (4, np.log(0.3), 0.2, 70.0)),
        price_assist=((3, np.log(0.12), 0.1, 60.0), (4, np.log(0.08), 0.1, 70.0)),
        pen_rate=0.13,
        conversion=0.8,
        team_pen=(),
        assist_fraction=0.88,
        own_goal_fraction=0.03,
        taker_q=0.85,
        n_rows=0,
    )


def share_rows(player_key, team_key, season, n, x_per_match, xa_per_match=0.1, fallback=False):
    """n full matches with team output 1.5 per match and the given npxG / xA per match."""
    return pd.DataFrame(
        {
            "player_key": player_key,
            "team_key": team_key,
            "season": season,
            "x_goals": x_per_match,
            "x_assists": xa_per_match,
            "d": 1.5,
            "on_pitch": 1.0,
            "fb_goals": fallback,
            "fb_assists": fallback,
            "pen_goals": 0.0,
            "penalties_missed": 0.0,
        },
        index=range(n),
    )


POOL = pd.DataFrame(
    {
        "player_key": [1, 2, 3, 4, 5],
        "team_key": [10, 10, 10, 20, 20],
        "element_type": [4, 4, 3, 4, 4],
        "price": [90, 70, 80, 70, 70],
    }
)


def test_marcel_recovers_planted_shares_and_shrinks_small_samples():
    rows = pd.concat(
        [
            share_rows(1, 10, 2023, 200, 0.75),  # share 0.5, lots of minutes
            share_rows(2, 10, 2023, 1, 0.75),  # share 0.5, one match
        ],
        ignore_index=True,
    )
    out = marcel_shares(rows, POOL, toy_fit(), SEASON).set_index("player_key")
    d0 = 480 / 90 * 1.5
    w = 3.0  # current season
    expected = (w * 200 * 0.75 + 0.3 * d0) / (w * 200 * 1.5 + d0)
    assert out.loc[1, "g"] == pytest.approx(expected) and abs(out.loc[1, "g"] - 0.5) < 0.01
    assert out.loc[2, "g"] == pytest.approx((w * 0.75 + 0.3 * d0) / (w * 1.5 + d0))
    assert 0.3 < out.loc[2, "g"] < 0.4  # mostly the FWD average
    assert not out.loc[1, "new"] and not out.loc[2, "new"]
    # New players: position + price.
    assert out.loc[3, "new"] and out.loc[3, "g"] == pytest.approx(0.1 * np.exp(0.5 * 2.0))
    assert out.loc[4, "g"] == pytest.approx(0.3)
    # Stronger prior, more shrinkage.
    strong = marcel_shares(rows, POOL, toy_fit(prior_minutes=5000.0), SEASON)
    assert abs(strong.set_index("player_key").loc[1, "g"] - 0.3) < abs(out.loc[1, "g"] - 0.3)


def test_season_weights_and_goal_fallback():
    rows = pd.concat(
        [
            share_rows(1, 10, 2023, 10, 0.75),  # 0.5 this season
            share_rows(1, 10, 2022, 10, 0.15),  # 0.1 last season
            share_rows(1, 10, 2018, 50, 3.0),  # too old: ignored
            share_rows(2, 10, 2023, 20, 0.75, fallback=True),
            share_rows(5, 20, 2023, 20, 0.75),
        ],
        ignore_index=True,
    )
    fit = toy_fit(season_weights=(3.0, 2.0, 1.0, 1.0), prior_minutes=1e-6)
    out = marcel_shares(rows, POOL, fit, SEASON).set_index("player_key")
    assert out.loc[1, "g"] == pytest.approx((3 * 0.5 + 2 * 0.1) / 5, rel=1e-6)
    fit = toy_fit(goals_weight=0.25)
    out = marcel_shares(rows, POOL, fit, SEASON).set_index("player_key")
    d0 = 480 / 90 * 1.5
    # The fallback rows weigh 0.25: player 2 sits nearer the prior than player 5.
    assert out.loc[2, "g"] == pytest.approx((0.75 * 20 * 0.75 + 0.3 * d0) / (0.75 * 30 + d0))
    assert out.loc[5, "g"] == pytest.approx((3 * 20 * 0.75 + 0.3 * d0) / (3 * 30 + d0))


def test_club_change_keeps_the_old_share_with_extra_shrinkage():
    rows = pd.concat(
        [
            share_rows(4, 99, 2022, 30, 0.75),  # player 4 now at club 20, was at 99
            share_rows(5, 20, 2022, 30, 0.75),  # player 5 stayed
        ],
        ignore_index=True,
    )
    out = marcel_shares(rows, POOL, toy_fit(club_change_weight=0.5), SEASON)
    out = out.set_index("player_key")
    d0 = 480 / 90 * 1.5
    assert out.loc[5, "g"] == pytest.approx((2 * 30 * 0.75 + 0.3 * d0) / (2 * 30 * 1.5 + d0))
    assert out.loc[4, "g"] == pytest.approx((30 * 0.75 + 0.3 * d0) / (30 * 1.5 + d0))
    assert 0.3 < out.loc[4, "g"] < out.loc[5, "g"] < 0.5
    same = marcel_shares(rows, POOL, toy_fit(club_change_weight=1.0), SEASON)
    assert same.set_index("player_key").loc[4, "g"] == pytest.approx(out.loc[5, "g"])


# --- the synthetic league -------------------------------------------------------------------


def test_output_schema_order_and_determinism(league, inputs):
    fit, out = predict(league, inputs)
    minutes, _ = inputs
    assert list(out.columns) == list(SHARES_COLUMNS)
    assert (out.dtypes.iloc[:7] == "int64").all() and (out.dtypes.iloc[7:] == "float64").all()
    assert out.index.equals(pd.RangeIndex(len(out)))
    keys = ["player_key", "horizon", "fixture_key"]
    assert out.equals(out.sort_values(keys, kind="mergesort").reset_index(drop=True))
    pd.testing.assert_frame_equal(out[minutes.columns[:7]], minutes[minutes.columns[:7]])
    assert (out[["e_np_goals", "e_pen_goals", "e_assists", "p_taker"]] >= 0).all().all()
    np.testing.assert_allclose(out["e_goals"], out["e_np_goals"] + out["e_pen_goals"])
    again = fit_shares(view(league))
    assert again == fit
    pd.testing.assert_frame_equal(predict_shares(view(league), again, *inputs), out)
    assert {season for season, _, _ in fit.coverage} == {2022, 2023}


def test_team_sums_match_the_team_lambdas(league, inputs):
    fit, out = predict(league, inputs)
    _, team = inputs
    sums = out.groupby(["fixture_key", "team_key"])[["e_np_goals", "e_pen_goals", "e_assists"]]
    sums = sums.sum().join(team.set_index(["fixture_key", "team_key"])["lambda_for"])
    rates = dict((key, rate * conversion) for key, rate, conversion in fit.team_pen)
    pen = np.array([rates.get(t, fit.pen_rate * fit.conversion) for _, t in sums.index])
    lam = sums["lambda_for"].to_numpy()
    np.testing.assert_allclose(sums["e_pen_goals"], pen, rtol=1e-9)
    np.testing.assert_allclose(sums["e_np_goals"], lam * (1 - fit.own_goal_fraction) - pen)
    np.testing.assert_allclose(sums["e_assists"], lam * fit.assist_fraction)
    # Shares are normalized over the expected XI.
    minutes, _ = inputs
    f = minutes["e_minutes"].to_numpy() / 90
    weighted = out.assign(gf=out["goal_share"] * f, af=out["assist_share"] * f)
    totals = weighted.groupby(["fixture_key", "team_key"])[["gf", "af"]].sum()
    np.testing.assert_allclose(totals, 1.0)


def test_fit_rates_come_from_visible_history(league):
    fit = fit_shares(view(league))
    at = deadline(league)
    teams = league["team_match"]
    teams = teams[teams["available_at"] < at]
    assert fit.assist_fraction == pytest.approx(
        league["player_match"].loc[league["player_match"]["available_at"] < at, "assists"].sum()
        / teams["goals_for"].sum()
    )
    matches = league["player_match"]
    matches = matches[matches["available_at"] < at]
    assert fit.own_goal_fraction == pytest.approx(
        matches["own_goals"].sum() / teams["goals_for"].sum()
    )
    assert 0 < fit.conversion < 1 and fit.pen_rate > 0
    assert fit.position_goal[-1][1] > fit.position_goal[1][1]  # FWD > DEF


def test_source_scale_is_fitted_on_the_overlap(league):
    tables = copy(league)
    matches = tables["player_match"]
    matches["fpl_xg"] = (matches["us_npxg"] * 1.25).round(4)
    matches["fpl_xa"] = (matches["us_xa"] * 0.8).round(4)
    fit = fit_shares(view(tables))
    assert fit.k_goals == pytest.approx(0.8, abs=0.01)
    assert fit.k_assists == pytest.approx(1.25, abs=0.02)
    assert fit.pen_pi[1] < 0.2  # big chances without a penalty: π shrinks below its default


def test_fpl_xg_takes_over_without_understat(league, inputs):
    tables = copy(league)
    matches = tables["player_match"]
    current = matches["season"] == SEASON
    for column in ("us_npxg", "us_xa", "us_goals", "us_npg", "us_xg"):
        matches.loc[current, column] = pd.NA
    teams = tables["team_match"]
    teams.loc[teams["season"] == SEASON, ["us_xg", "us_npxg"]] = np.nan
    _, base = predict(league, inputs)
    _, live = predict(tables, inputs)
    assert np.corrcoef(base["e_np_goals"], live["e_np_goals"])[0, 1] > 0.98
    assert np.corrcoef(base["e_assists"], live["e_assists"])[0, 1] > 0.98


def test_planted_shares_are_recovered(league, inputs):
    tables = copy(league)
    matches = tables["player_match"]
    teams = tables["team_match"].set_index(["fixture_key", "team_key"])["us_npxg"]
    star = 100_000 + 100 * 2 + 3  # a club-2 defender (slots: 2 GK, 5 DEF, 5 MID, 3 FWD)
    rows = (matches["player_key"] == star) & (matches["minutes"] > 0)
    team_npxg = np.array(
        [
            teams.loc[(f, t)]
            for f, t in zip(
                matches.loc[rows, "fixture_key"], matches.loc[rows, "team_key"], strict=True
            )
        ],
        dtype="float64",
    )
    matches.loc[rows, "us_npxg"] = 0.6 * team_npxg * matches.loc[rows, "minutes"] / 90
    _, base = predict(league, inputs)
    _, planted = predict(tables, inputs)
    mine = lambda out: out[out["player_key"] == star]  # noqa: E731
    assert (mine(planted)["e_np_goals"] > 3 * mine(base)["e_np_goals"]).all()
    club = planted[(planted["team_key"] == 2) & (planted["horizon"] == 0)]
    assert club.sort_values("goal_share")["player_key"].iloc[-1] == star


def test_penalty_takers_from_the_snapshot_listing(league, inputs):
    tables = copy(league)
    snaps = tables["player_snapshot"]
    taker, second = 100_000 + 100 * 3 + 9, 100_000 + 100 * 3 + 12
    snaps.loc[snaps["player_key"] == taker, "penalties_order"] = 1
    snaps.loc[snaps["player_key"] == second, "penalties_order"] = 2
    fit, out = predict(tables, inputs)
    club = out[(out["team_key"] == 3) & (out["horizon"] == 0)].set_index("player_key")
    assert club["p_taker"].idxmax() == taker
    assert club["e_pen_goals"].idxmax() == taker
    assert club["e_pen_goals"].nlargest(2).index[1] == second
    # p_taker is the conditional: q when the first choice; below q for the second.
    assert club.loc[taker, "p_taker"] == pytest.approx(fit.taker_q, abs=0.06)
    snaps.loc[snaps["player_key"] == taker, "penalties_order"] = 3
    _, out = predict(tables, inputs)
    club = out[(out["team_key"] == 3) & (out["horizon"] == 0)].set_index("player_key")
    assert club["p_taker"].idxmax() == second


def test_penalty_takers_from_history_without_snapshots(inputs):
    tables = synthetic_tables(seasons=(2022, 2023), n_clubs=6, seed=3, snapshots=False)
    matches = tables["player_match"]
    taker = 100_000 + 100 * 4 + 10
    rows = (matches["player_key"] == taker) & (matches["season"] == 2022)
    rows &= matches["minutes"] > 0
    index = matches.index[rows][:4]
    matches.loc[index, "goals_scored"] += 1
    matches.loc[index, "us_goals"] = matches.loc[index, "goals_scored"]
    fit, out = predict(tables, inputs)
    club = out[(out["team_key"] == 4) & (out["horizon"] == 0)].set_index("player_key")
    assert club["e_pen_goals"].idxmax() == taker
    assert club.loc[taker, "p_taker"] > 0.5


def test_only_visible_data_is_used(league, inputs):
    tables = copy(league)
    at = deadline(tables)
    rng = np.random.default_rng(0)
    for name in ("player_match", "team_match", "player_gw", "player_snapshot"):
        frame = tables[name]
        future = frame["available_at"] >= at
        n = int(future.sum())
        if name == "player_match":
            frame.loc[future, "goals_scored"] = rng.integers(0, 5, n)
            frame.loc[future, "us_npxg"] = rng.random(n) * 3
            frame.loc[future, "us_goals"] = 4
            frame.loc[future, "us_npg"] = 0
            frame.loc[future, "assists"] = 3
        if name == "team_match":
            frame.loc[future, "goals_for"] = 9
            frame.loc[future, "us_npxg"] = 0.01
        if name == "player_gw":
            frame.loc[future, "value"] = 150
        if name == "player_snapshot":
            frame.loc[future, "penalties_order"] = 1
    fit, out = predict(league, inputs)
    corrupted_fit, corrupted = predict(tables, inputs)
    assert corrupted_fit == fit
    pd.testing.assert_frame_equal(corrupted, out)


def test_params_validate():
    with pytest.raises(ValueError):
        SharesParams(season_weights=())
    with pytest.raises(ValueError):
        SharesParams(prior_minutes=0)
    assert dataclasses.replace(SharesParams(), prior_minutes=240.0).prior_minutes == 240.0
