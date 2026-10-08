"""The walk-forward xP evaluation (fplopt.evaluate.run) on a synthetic league."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest.policies import target_xp
from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.simulator import Caches, HoldoutError, read_outcomes, season_schedule
from fplopt.evaluate import run
from fplopt.evaluate.metrics import diebold_mariano, lineup_regret, mse
from fplopt.evaluate.run import (
    PREDICTION_COLUMNS,
    REGRET_COLUMNS,
    candidate_keys,
    deadline_squads,
    evaluate,
    model_seasons,
)
from fplopt.features.store import DataStore
from fplopt.models import MODELS
from fplopt.models.baseline import xp_rolling

# 2020 is history only for most tests; 2021 is develop, 2023 validate (10 clubs, 18 GWs).
SEASONS = (2020, 2021, 2022, 2023)
BLANK = (2023, 5, 1)  # club 1's GW5 fixture is played in GW7


@pytest.fixture(scope="module")
def store():
    return DataStore(tables=synthetic_tables(seasons=SEASONS, n_clubs=10, blank=BLANK))


@pytest.fixture(scope="module")
def result(store):
    return evaluate(store, ["rolling", "ep_next"], [2021, 2023], n_random=2)


def test_predictions_schema_and_rows(store, result):
    predictions = result.predictions
    assert list(predictions.columns) == [name for name, _ in PREDICTION_COLUMNS]
    assert predictions.dtypes.astype(str).tolist() == [str(d) for _, d in PREDICTION_COLUMNS]
    caches = Caches()
    for season in (2021, 2023):
        schedule = season_schedule(store, season, 1)
        for model in ("rolling", "ep_next"):
            part = predictions[(predictions["model"] == model) & (predictions["season"] == season)]
            assert part["deadline"].nunique() == len(schedule) == 18
            row = schedule.iloc[4]
            frame = caches.xp(store, model, store.as_of(row["deadline_time"]))
            at = part[part["deadline"] == row["deadline_time"]]
            assert len(at) == len(frame)
            assert at["xp"].tolist() == frame["xp"].tolist()
            assert (at["target_gw"].to_numpy() == frame["gw"].to_numpy()).all()
    # Sorted by model (given order), deadline, player_key, horizon.
    assert predictions["model"].iloc[0] == "rolling" and predictions["model"].iloc[-1] == "ep_next"
    keys = predictions[predictions["model"] == "rolling"][["deadline", "player_key", "horizon"]]
    assert keys.equals(keys.sort_values(list(keys.columns), kind="mergesort"))


def test_realized_points_are_the_backtest_outcomes(store, result):
    predictions = result.predictions
    rules = backtest_rules(2023)
    schedule = season_schedule(store, 2023, 1).set_index("gw")
    for gw in (5, 7, 12):
        realized = read_outcomes(store, rules, 2023, gw, schedule.loc[gw, "lockdown_time"]).realized
        rows = predictions[(predictions["season"] == 2023) & (predictions["target_gw"] == gw)]
        keys = rows["player_key"].astype(int)
        expected = [realized.get(k, (0, 0)) for k in keys]
        assert rows["points"].tolist() == [float(p) for p, _ in expected]
        assert rows["minutes"].tolist() == [float(m) for _, m in expected]
    # Club 1 blanks in GW5: no fixture, 0 points; it doubles in GW7.
    gw5 = predictions[(predictions["season"] == 2023) & (predictions["target_gw"] == 5)]
    assert gw5["n_fixtures"].isin([0, 1]).all() and (gw5["n_fixtures"] == 0).any()
    gw7 = predictions[(predictions["season"] == 2023) & (predictions["target_gw"] == 7)]
    assert (gw7["n_fixtures"] == 2).any()


def test_xp_metrics_recomputed(result):
    predictions = result.predictions
    rows = {(r["model"], r["split"], r["horizon"]): r for r in result.metrics["xp"]}
    assert {split for _, split, _ in rows} == {"all", "develop", "validate"}
    for model in ("rolling", "ep_next"):
        mine = predictions[predictions["model"] == model]
        h0 = mine[mine["horizon"] == 0]
        assert rows[(model, "all", "0")]["mse"] == pytest.approx(mse(h0["xp"], h0["points"]))
        validate = h0[h0["season"] == 2023]
        assert rows[(model, "validate", "0")]["mse"] == pytest.approx(
            mse(validate["xp"], validate["points"])
        )
        later = mine[mine["horizon"].between(1, 5)]
        assert rows[(model, "all", "1-5")]["mse"] == pytest.approx(
            mse(later["xp"], later["points"])
        )
        # 10 clubs: every player is within the default top-N, so all rows are candidates.
        assert rows[(model, "all", "0")]["mse_candidates"] == rows[(model, "all", "0")]["mse"]
    bands = [b for b in result.metrics["bands"] if b["model"] == "rolling"]
    bands = [b for b in bands if b["split"] == "all" and b["horizon"] == "0"]
    h0 = predictions[(predictions["model"] == "rolling") & (predictions["horizon"] == 0)]
    assert sum(b["n"] for b in bands) == len(h0)
    low = h0[h0["xp"] < 0.5]
    assert bands[0]["hi"] == 0.5 and bands[0]["n"] == len(low)
    assert bands[0]["mse"] == pytest.approx(mse(low["xp"], low["points"]))
    seasons = {(r["model"], r["season"]) for r in result.metrics["by_season"]}
    assert seasons == {(m, s) for m in ("rolling", "ep_next") for s in (2021, 2023)}


def test_dm_rows_on_the_common_sample(result):
    predictions = result.predictions
    dm = {(r["metric"], r["split"], r["horizon"]): r for r in result.metrics["dm"]}
    row = dm[("mse", "all", "0")]
    assert (row["a"], row["b"]) == ("rolling", "ep_next")
    h0 = predictions[predictions["horizon"] == 0]
    a = h0[h0["model"] == "rolling"].set_index(["deadline", "player_key"])
    b = h0[h0["model"] == "ep_next"].set_index(["deadline", "player_key"])
    expected = diebold_mariano(
        (a["xp"] - a["points"]) ** 2,
        (b["xp"].reindex(a.index) - a["points"]) ** 2,
        a.index.get_level_values("deadline"),
    )
    assert row["mean_diff"] == pytest.approx(expected.mean_diff)
    assert row["p_a_better"] == pytest.approx(expected.p_a_better)
    assert row["n_clusters"] == 36 and row["seasons"] == [2021, 2023]
    assert dm[("mse", "all", "3")]["lag"] == 3 and dm[("mse", "all", "1-5")]["lag"] == 5
    assert ("xi_regret", "validate", "0") in dm and ("captain_regret", "develop", "0") in dm


def test_regrets_use_the_same_squads_for_every_model(store, result):
    regrets = result.regrets
    assert list(regrets.columns) == [name for name, _ in REGRET_COLUMNS]
    squads = regrets.groupby(["model", "deadline"])["squad"].apply(tuple).unstack("model")
    assert (squads["rolling"] == squads["ep_next"]).all()
    assert (regrets["xi_regret"] == regrets["best_points"] - regrets["points"]).all()
    assert (regrets["xi_regret"] >= 0).all() and (regrets["captain_regret"] >= 0).all()
    # One deadline recomputed independently.
    rules = backtest_rules(2023)
    row = season_schedule(store, 2023, 1).iloc[6]
    view = store.as_of(row["deadline_time"])
    outcomes = read_outcomes(store, rules, 2023, int(row["gw"]), row["lockdown_time"]).realized
    squads = deadline_squads(view, rules, n_random=2)
    assert [s for s, _ in squads] == ["template", "random0", "random1"]
    frame = MODELS["ep_next"](view)
    for squad_id, state in squads:
        squad = pd.DataFrame(
            {
                "player_key": [h.player_key for h in state.holdings],
                "element_type": [h.element_type for h in state.holdings],
            }
        )
        expected = lineup_regret(squad, target_xp(frame), outcomes, rules)
        got = regrets[
            (regrets["model"] == "ep_next")
            & (regrets["deadline"] == row["deadline_time"])
            & (regrets["squad"] == squad_id)
        ].iloc[0]
        assert (got["points"], got["xi_regret"], got["captain_regret"]) == (
            expected.points,
            expected.xi_regret,
            expected.captain_regret,
        )
    summary = {(r["model"], r["split"]): r for r in result.metrics["regret"]}
    mine = regrets[(regrets["model"] == "rolling") & (regrets["season"] == 2021)]
    assert summary[("rolling", "develop")]["xi_regret"] == pytest.approx(mine["xi_regret"].mean())


def test_template_is_skipped_where_it_is_refused():
    # Without snapshots no ownership is visible at GW1 (GW1's is visible from GW2).
    tables = synthetic_tables(seasons=(2023,), n_clubs=10, snapshots=False)
    store = DataStore(tables=tables)
    rules = backtest_rules(2023)
    schedule = season_schedule(store, 2023, 1)
    first = deadline_squads(store.as_of(schedule["deadline_time"].iloc[0]), rules, 2, seed=5)
    assert [s for s, _ in first] == ["random5", "random6"]
    second = deadline_squads(store.as_of(schedule["deadline_time"].iloc[1]), rules, 1)
    assert [s for s, _ in second] == ["template", "random0"]


def test_ep_next_runs_only_from_2021(store):
    out = evaluate(store, ["rolling", "ep_next"], [2020, 2021], n_random=0)
    seasons = out.predictions.groupby("model")["season"].unique()
    assert sorted(seasons["rolling"]) == [2020, 2021] and list(seasons["ep_next"]) == [2021]
    assert out.metrics["seasons"] == {"rolling": [2020, 2021], "ep_next": [2021]}
    for row in out.metrics["dm"]:
        assert row["seasons"] == [2021]
    assert model_seasons(["ep_next_fade", "rolling"], [2019, 2022]) == {
        "ep_next_fade": (2022,),
        "rolling": (2019, 2022),
    }


def test_refusals():
    store = DataStore(tables={})  # never read: the checks come first
    with pytest.raises(HoldoutError, match="holdout"):
        evaluate(store, ["rolling"], [2024, 2025])
    with pytest.raises(ValueError, match="unknown"):
        evaluate(store, ["nope"], [2023])
    with pytest.raises(ValueError, match="unique"):
        evaluate(store, ["rolling", "rolling"], [2023])
    with pytest.raises(ValueError, match="no season"):
        evaluate(store, ["rolling", "ep_next"], [2019, 2020])
    with pytest.raises(ValueError, match="no models"):
        evaluate(store, [], [2023])


def test_jobs_two_equals_jobs_one(store, result):
    parallel = evaluate(store, ["rolling", "ep_next"], [2021, 2023], n_random=2, jobs=2)
    pd.testing.assert_frame_equal(parallel.predictions, result.predictions)
    pd.testing.assert_frame_equal(parallel.regrets, result.regrets)
    dump = lambda m: json.dumps(m, sort_keys=True, allow_nan=False)  # noqa: E731
    assert dump(parallel.metrics) == dump(result.metrics)


def test_candidate_keys_union_of_top_n():
    pool = pd.DataFrame(
        {"player_key": [1, 2, 3, 4, 5, 6], "element_type": [1, 1, 2, 2, 2, 4]}, dtype="int64"
    )

    def frame(values):
        keys = [k for k in range(1, 7) for _ in range(2)]
        xp = [v / 2 for v in values for _ in range(2)]  # horizon sum = value
        return pd.DataFrame({"player_key": keys, "horizon": [0, 1] * 6, "xp": xp})

    a = frame([5, 1, 3, 2, 2, 0])
    b = frame([1, 5, 0, 1, 4, 0])
    prune = {1: 1, 2: 1, 4: 1}
    assert candidate_keys({"a": a}, pool, prune).tolist() == [1, 3, 6]
    assert candidate_keys({"b": b}, pool, prune).tolist() == [2, 5, 6]
    assert candidate_keys({"a": a, "b": b}, pool, prune).tolist() == [1, 2, 3, 5, 6]
    # Ties go to the smaller key; a position without N keeps everyone.
    assert candidate_keys({"a": frame([1, 1, 2, 2, 0, 0])}, pool, {1: 1, 2: 1}).tolist() == [
        1,
        3,
        6,
    ]


# --- fake models --------------------------------------------------------------------------------


def xp_zero(view):
    frame = xp_rolling(view)
    return frame.assign(xp=0.0)


def xp_components(view):
    frame = xp_rolling(view)
    return frame.assign(p_cs=0.3, e_goals=np.where(frame["xp"] > 0, 0.2, 0.0))


def xp_stranger(view):
    frame = xp_rolling(view)
    return pd.concat([frame, frame.iloc[:1].assign(player_key=10**9)], ignore_index=True)


def test_fake_models_and_component_hook(store, monkeypatch):
    monkeypatch.setitem(MODELS, "zero", xp_zero)
    monkeypatch.setitem(MODELS, "with_components", xp_components)
    out = evaluate(store, ["zero", "with_components"], [2023], n_random=1)
    assert {"p_cs", "e_goals"} <= set(out.predictions.columns)
    zero = out.predictions[out.predictions["model"] == "zero"]
    assert zero["p_cs"].isna().all()
    h0 = zero[zero["horizon"] == 0]
    xp_row = next(
        r
        for r in out.metrics["xp"]
        if (r["model"], r["split"], r["horizon"]) == ("zero", "all", "0")
    )
    assert xp_row["mse"] == pytest.approx(float((h0["points"] ** 2).mean()))
    components = {(r["model"], r["component"], r["horizon"]): r for r in out.metrics["components"]}
    assert ("zero", "p_cs", "0") not in components
    cs = components[("with_components", "p_cs", "0")]
    mine = out.predictions[
        (out.predictions["model"] == "with_components") & (out.predictions["horizon"] == 0)
    ]
    single = mine[mine["n_fixtures"] == 1]
    assert cs["n"] == len(single)
    assert cs["brier"] == pytest.approx(float(((0.3 - (single["clean_sheets"] >= 1)) ** 2).mean()))
    goals = components[("with_components", "e_goals", "0")]
    assert goals["n"] == len(mine) and math.isfinite(goals["log_likelihood"])
    # The zero model never beats rolling-like predictions in this league.
    dm = next(
        r
        for r in out.metrics["dm"]
        if (r["metric"], r["split"], r["horizon"]) == ("mse", "all", "0")
    )
    assert dm["mean_diff"] > 0 and dm["p_b_better"] < 0.10


def test_rows_outside_the_pool_are_refused(store, monkeypatch):
    monkeypatch.setitem(MODELS, "stranger", xp_stranger)
    with pytest.raises(ValueError, match="outside the pool"):
        evaluate(store, ["stranger"], [2023], n_random=0)


def test_module_constants_match_the_cli():
    from fplopt import cli

    assert {m for m in run.FIRST_SEASON} == set(cli.EP_NEXT_MODELS)
    assert set(run.FIRST_SEASON.values()) == {cli.EP_NEXT_FIRST_SEASON}
