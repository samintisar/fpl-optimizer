"""The Phase 6 holdout run's fixed parts (`fplopt.backtest.holdout`): the gate, the
run-once and clean-checkout guards and the input pre-flight."""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest import holdout
from fplopt.backtest.evaluate import log_experiment
from fplopt.features.store import DataStore


def windows(diffs, xg, a="sys", b="base", starts=2):
    """A per-decision frame: one window per value at gw_index 1, 5, 9, ..., each start equal."""
    rows = []
    for i, (d, x) in enumerate(zip(diffs, xg, strict=True)):
        for s in range(starts):
            rows.append(
                {
                    "season": holdout.SEASON,
                    "start_id": f"s{s}",
                    "policy_a": a,
                    "policy_b": b,
                    "gw_index": 1 + 4 * i,
                    "diff": d,
                    "diff_xg": x,
                }
            )
    return pd.DataFrame(rows)


def test_gate_passes_on_a_clear_edge_with_the_xg_sign():
    diffs = [8.0, 3.0, 6.0, 9.0, 2.0, 7.0, 5.0, 4.0, 10.0, 6.0]
    result = holdout.gate(windows(diffs, [1.0] * 10), "sys", "base", "a")
    assert result.passed and result.n_windows == 10
    assert result.mean == pytest.approx(6.0) and result.p < 0.001
    assert result.xg_mean == pytest.approx(1.0)
    assert result.bootstrap.mean == pytest.approx(6.0)


def test_gate_fails_on_the_xg_sign_or_the_p_value():
    diffs = [8.0, 3.0, 6.0, 9.0, 2.0, 7.0, 5.0, 4.0, 10.0, 6.0]
    assert not holdout.gate(windows(diffs, [-1.0] * 10), "sys", "base", "a").passed
    noisy = [20.0, -18.0, 15.0, -12.0, 9.0, -10.0, 14.0, -16.0, 11.0, -9.0]
    result = holdout.gate(windows(noisy, [1.0] * 10), "sys", "base", "a")
    assert result.mean > 0 and result.p >= holdout.ALPHA and not result.passed
    no_xg = holdout.gate(windows(diffs, [None] * 10), "sys", "base", "a")
    assert np.isnan(no_xg.xg_mean) and not no_xg.passed


def test_gate_needs_the_pair():
    with pytest.raises(ValueError, match="no per-decision rows"):
        holdout.gate(windows([1.0, 2.0], [1.0, 1.0]), "sys", "other", "b")


def test_already_run(tmp_path):
    log = tmp_path / "experiments.csv"
    assert not holdout.already_run(log)
    log_experiment(log, "fplopt backtest compare", {}, {}, 1, "something else")
    assert not holdout.already_run(log)
    log_experiment(log, "fplopt holdout run", {}, {}, 1, holdout.FAMILY)
    assert holdout.already_run(log)


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_dirty_checkout_ignores_only_the_experiment_log(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "experiments.csv").write_text("a\n", encoding="utf-8")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    assert holdout.dirty_checkout(tmp_path) == []
    (tmp_path / "results" / "experiments.csv").write_text("a\nb\n", encoding="utf-8")
    assert holdout.dirty_checkout(tmp_path) == []
    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("", encoding="utf-8")
    assert sorted(line[3:] for line in holdout.dirty_checkout(tmp_path)) == ["code.py", "new.py"]


def test_dirty_checkout_without_git(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(holdout.subprocess, "run", no_git)
    with pytest.raises(RuntimeError, match="can't check the checkout"):
        holdout.dirty_checkout(tmp_path)


def test_preflight_lists_every_deadline():
    store = DataStore(tables=synthetic_tables(seasons=(2023,), n_clubs=10))
    table, problems = holdout.preflight(store, 2023)
    assert list(table["gw"]) == sorted(table["gw"]) and len(table) == table["gw"].nunique()
    assert (table["pool"] > 0).all()
    assert set(table.columns) == {"gw", "deadline", "pool", "ep_next", "clubs_without_players"}
    flagged = {p.split(":")[0] for p in problems}
    gws = set(table.loc[table["ep_next"] < 0.95 * table["pool"], "gw"])
    gws |= set(table.loc[table["clubs_without_players"].map(len) > 0, "gw"])
    assert flagged == {f"GW{g}" for g in gws}
