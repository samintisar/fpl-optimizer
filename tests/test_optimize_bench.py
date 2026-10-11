"""Deadline sampling and summaries of the optimizer benchmark (fplopt.optimize.bench; Phase 4
plan, Task 3). The end-to-end run is tests/test_cli.py::test_optimize_bench_end_to_end."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pulp")
pytest.importorskip("highspy")

from synthetic_season import synthetic_tables  # noqa: E402

from fplopt.features.store import DataStore  # noqa: E402
from fplopt.optimize.bench import (  # noqa: E402
    BenchResult,
    bench_cases,
    bench_deadlines,
    summarize,
    summary_text,
)


@pytest.fixture(scope="module")
def league():
    return DataStore(tables=synthetic_tables(seasons=(2022, 2023), n_clubs=10))


def test_deadlines_round_robin_over_seasons_never_the_holdout(league) -> None:
    picked = bench_deadlines(league, 4, seed=1, seasons=(2022, 2023, 2025))
    assert len(picked) == len(set(picked)) == 4
    assert sorted({s for s, _ in picked}) == [2022, 2023]
    assert sum(s == 2022 for s, _ in picked) == 2
    assert picked == bench_deadlines(league, 4, seed=1, seasons=(2022, 2023, 2025))
    assert picked != bench_deadlines(league, 4, seed=2, seasons=(2022, 2023, 2025))


def test_deadlines_only_in_the_past_and_validated(league) -> None:
    gameweeks = league.as_of(pd.Timestamp("2200-01-01", tz="UTC")).table("gameweek")
    first = gameweeks[gameweeks["season"] == 2023].sort_values("gw_index")
    cutoff = first["deadline_time"].iloc[3]  # only gw_index 1-3 of 2023 have passed
    picked = bench_deadlines(league, 50, seed=0, seasons=(2023,), now=cutoff)
    assert sorted(g for _, g in picked) == sorted(first["gw_index"].iloc[:3].tolist())
    with pytest.raises(ValueError, match="at least one"):
        bench_deadlines(league, 0, seed=0)
    with pytest.raises(ValueError, match="no gameweeks"):
        bench_deadlines(league, 1, seed=0, seasons=(2025,))


def test_cases_cover_template_random_and_both_xp_models(league) -> None:
    cases = bench_cases(league, 2, seed=5)
    assert len(cases) == 8
    assert {c.start for c in cases} == {"template", "random:5", "random:6"}
    assert {c.xp for c in cases} == {"ep_next", "rolling"}
    assert cases[0].name.startswith("2022-23 gw")


def test_summary_text_is_ascii() -> None:
    cases = pd.DataFrame(
        {
            "input_s": [0.01, 0.02],
            "build_s": [0.03, 0.04],
            "solve_s": [0.5, 2.0],
            "gap": [0.0, 0.004],
            "top3_s": [1.0, 5.0],
            "chips_s": [8.0, 12.0],
            "relax_s": [5.0, 6.0],
            "chip_solve_s": [3.0, 6.0],
            "n_scenarios": [205, 224],
            "n_relaxations": [50, 60],
            "n_solves": [2, 3],
            "n_candidates": [80, 90],
            "chips_all_s": [400.0, None],
            "bound_matches_all": [True, None],
            "chips_prune_loss": [0.0, None],
        }
    )
    prune = pd.DataFrame(
        {
            "variant": ["none", "default", "none", "default"],
            "n_candidates": [700, 80, 700, 90],
            "solve_s": [40.0, 1.0, 50.0, 2.0],
            "loss": [0.0, 0.0, 0.0, 0.2],
        }
    )
    summary = summarize(BenchResult(cases, prune, {}))
    assert summary["chips_all"] == {
        "n_cases": 1,
        "matches": 1,
        "undetermined": 0,
        "all_s": {"median": 400.0, "p90": 400.0, "max": 400.0},
        "bound_s": {"median": 8.0, "p90": 8.0, "max": 8.0},
        "prune_loss": {"median": 0.0, "p90": 0.0, "max": 0.0},
    }
    assert summary["prune"]["default"]["n_loss_over_0.1"] == 1
    text = summary_text(summary)
    assert text.isascii()  # printed to Windows consoles (cp1252)
    assert "same best plan in 1/1" in text and "default" in text


def test_same_plan_is_undetermined_when_a_limit_stopped_a_solve() -> None:
    """The bench runs under a wall-clock limit: the bound and exhaustive searches are only
    compared when every solve of both reached the gap."""
    from types import SimpleNamespace

    from fplopt.optimize.bench import _same_plan

    def plans(status, scenario="none", total=10.0):
        best = SimpleNamespace(scenario=scenario, total_objective=total)
        return SimpleNamespace(solver_status=status, best=best)

    limited = "Optimal (1 chip scenario solve(s) stopped at a limit)"
    assert _same_plan(plans("Optimal"), plans("Optimal")) is True
    assert _same_plan(plans("Optimal"), plans("Optimal", "bboost@gw3")) is False
    assert _same_plan(plans("Optimal"), plans("Optimal", total=10.5)) is False
    assert _same_plan(plans("TimeLimit"), plans("Optimal")) is None
    assert _same_plan(plans("Optimal"), plans(limited)) is None
    cases = pd.DataFrame(
        {
            "chips_all_s": [400.0, 500.0],
            "chips_s": [8.0, 9.0],
            "chips_prune_loss": [0.0, 0.0],
            "bound_matches_all": [True, None],
        }
    )
    timings = {k: [1.0, 1.0] for k in ("input_s", "build_s", "solve_s", "gap", "top3_s")}
    timings |= {k: [1.0, 1.0] for k in ("relax_s", "chip_solve_s", "n_scenarios")}
    timings |= {k: [1, 1] for k in ("n_relaxations", "n_solves", "n_candidates")}
    summary = summarize(BenchResult(cases.assign(**timings), pd.DataFrame(), {}))
    assert summary["chips_all"]["matches"] == 1 and summary["chips_all"]["undetermined"] == 1
    assert "same best plan in 1/2 (1 undetermined" in summary_text(summary)


class _MinutesCaches:
    """`Caches` whose xP frames carry the minutes columns (as v1's do), on rolling xP:
    expected minutes 90 per GW for players with positive xP, else 10; P(play) 0.9."""

    def __init__(self) -> None:
        from fplopt.backtest.simulator import Caches

        self._caches = Caches()

    def pool(self, store, view):
        return self._caches.pool(store, view)

    def xp(self, store, model, view):
        frame = self._caches.xp(store, "rolling", view)
        minutes = frame["xp"].gt(0).map({True: 90.0, False: 10.0}).astype("float64")
        return frame.assign(p_play=0.9, e_minutes=minutes)


def test_bench_case_measures_minutes_floors_and_fixed_bench(league) -> None:
    """Task 8: frames with minutes get the floor variants (loss vs floor 0) and a no-chip
    solve with the fixed bench weights; the no-floor references stay floor-free."""
    from fplopt.optimize import OptimizerParams
    from fplopt.optimize.bench import BenchCase, bench_case

    season, gw_index = bench_deadlines(league, 1, seed=3, seasons=(2023,))[0]
    case = BenchCase(season, max(gw_index, 3), "random:1", "v1")
    params = OptimizerParams(horizon=2, prune_n={1: 4, 2: 8, 3: 8, 4: 5}, time_limit=60.0)
    row, prune = bench_case(
        league,
        _MinutesCaches(),
        case,
        params,
        chips_all=False,
        prune_study=True,
        floors=(0.0, 60.0, 500.0),
    )
    assert row["minutes_bench"] and "fixed_bench_solve_s" in row
    variants = [r["variant"] for r in prune]
    assert variants[-3:] == ["floor_0", "floor_60", "floor_500"]
    by = {r["variant"]: r for r in prune}
    assert by["floor_0"]["floor_loss"] == 0.0 and "floor_loss" not in by["none"]
    # No player reaches 500 expected minutes over 2 GWs: only the owned squad is left.
    assert by["floor_500"]["n_candidates"] == 15
    assert by["floor_500"]["floor_loss"] >= -1e-6
    assert by["none"]["n_candidates"] >= by["floor_60"]["n_candidates"]
    summary = summarize(BenchResult(pd.DataFrame([row]), pd.DataFrame(prune), {}))
    assert "floor_loss" in summary["prune"]["floor_60"]
    assert "floor_loss" not in summary["prune"]["none"]
    text = summary_text(summary)
    assert "Minutes floors" in text and text.isascii()
