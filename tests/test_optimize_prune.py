"""Candidate pruning (fplopt.optimize.prune; Phase 4 plan, Decisions *Pruning*)."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from optimize_fixtures import (
    RULES,
    correlated_xp,
    first_valid_squad,
    grid_pool,
    make_state,
)

from fplopt.optimize import OptimizerParams, PlanInput
from fplopt.optimize.prune import prune


def run(keys, et, price, xp, owned=(), buyable=None, prune_n=None, dominated=True):
    keys = np.asarray(keys)
    return prune(
        keys=keys,
        element_type=np.asarray(et),
        price=np.asarray(price),
        xp=np.asarray(xp, dtype="float64").reshape(len(keys), -1),
        owned=np.isin(keys, list(owned)),
        buyable=np.ones(len(keys), bool) if buyable is None else np.asarray(buyable),
        decay=0.85,
        horizons=np.arange(np.asarray(xp).reshape(len(keys), -1).shape[1]),
        rules=RULES,
        prune_n=prune_n,
        dominated=dominated,
    ).tolist()


def test_top_n_by_xp_and_by_value_per_position() -> None:
    # 6 FWDs: xp 10, 9, 8, 1, 0.9, 0.1; prices 100, 100, 100, 10, 45, 45.
    keys = [1, 2, 3, 4, 5, 6]
    xp = [10, 9, 8, 1, 0.9, 0.1]
    price = [100, 100, 100, 10, 45, 45]
    kept = run(keys, [4] * 6, price, xp, prune_n={4: 2}, dominated=False)
    assert kept == [1, 2, 4]  # best two by xp (1, 2) and by xp/price (4: 0.1, 1: 0.1 tie → 1)


def test_owned_always_kept_and_unlisted_positions_untouched() -> None:
    keys = [1, 2, 3, 4, 5]
    et = [4, 4, 4, 3, 3]
    xp = [5, 4, 0, 0, 0]
    kept = run(keys, et, [50] * 5, xp, owned=[3], prune_n={4: 1}, dominated=True)
    assert 3 in kept and 1 in kept and 2 not in kept
    # MIDs (3) aren't in prune_n: both kept by top-N; neither dominated by 5 (squad_select).
    assert {4, 5} <= set(kept)


def test_dominance_keeps_enough_dominators() -> None:
    # 4 GKs (squad_select 2): 1 and 2 dominate 3 and 4; 3 dominates 4.
    keys = [1, 2, 3, 4]
    xp = [[5, 5], [4, 6], [3, 3], [3, 2]]
    price = [40, 40, 45, 45]
    assert run(keys, [1] * 4, price, xp) == [1, 2]
    # Only one dominator (2 needed): kept.
    assert run([1, 3], [1, 1], [40, 45], [[5, 5], [3, 3]]) == [1, 3]
    # Cheaper but worse in one GW: not dominated.
    assert run(keys, [1] * 4, price, [[5, 5], [4, 6], [3, 7], [3, 2]]) == [1, 2, 3]


def test_identical_players_tie_by_key_and_departed_never_dominate() -> None:
    keys = [7, 3, 5, 9]
    assert run(keys, [1] * 4, [40] * 4, [2.0] * 4) == [3, 5]
    # 3 and 5 can't be bought (left the game; owned): they dominate nobody.
    kept = run(keys, [1] * 4, [40] * 4, [2.0] * 4, owned=[3, 5], buyable=[True, False, False, True])
    assert kept == [3, 5, 7, 9]


def test_deterministic_under_input_order() -> None:
    rng = np.random.default_rng(0)
    n = 60
    keys = rng.permutation(1000)[:n]
    et = rng.integers(1, 5, n)
    price = rng.integers(40, 120, n)
    xp = rng.uniform(0, 8, (n, 3)).round(1)
    a = run(keys, et, price, xp, prune_n={1: 3, 2: 5, 3: 5, 4: 3})
    order = rng.permutation(n)
    b = run(keys[order], et[order], price[order], xp[order], prune_n={1: 3, 2: 5, 3: 5, 4: 3})
    assert a == b


def plans(seed: int, n_gws: int = 2):
    """(pruned, dominance-only, unpruned) plan objectives on a random instance."""
    from fplopt.optimize import optimize

    pool = grid_pool(range(1, 9), {1: 2, 2: 5, 3: 5, 4: 3}, seed=seed)
    xp = correlated_xp(pool, n_gws, seed)
    state = make_state(pool, first_valid_squad(pool), ft=2)
    out = []
    base = OptimizerParams(horizon=n_gws, mip_gap=0.0)
    for params in (
        base,
        OptimizerParams(horizon=n_gws, mip_gap=0.0, prune_n=None),
        OptimizerParams(horizon=n_gws, mip_gap=0.0, prune_n=None, prune_dominated=False),
    ):
        problem = PlanInput.from_context(state, pool, xp, RULES, params)
        assert set(state.player_keys) <= {p.player_key for p in problem.players}
        out.append((len(problem.players), optimize(problem, params).objective))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_pruning_keeps_the_optimum_on_random_instances(seed: int) -> None:
    pytest.importorskip("highspy")
    (n_pruned, pruned), (n_dom, dominance), (n_all, unpruned) = plans(seed)
    assert n_pruned <= n_dom < n_all
    assert dominance == pytest.approx(unpruned, abs=1e-6)
    # Top-N is a heuristic: allow the default MIP gap.
    assert pruned >= unpruned * (1 - OptimizerParams().mip_gap) - 1e-6


def test_default_params_are_read_only() -> None:
    params = OptimizerParams()
    assert isinstance(params.prune_n, MappingProxyType)
    assert dict(params.prune_n) == {1: 10, 2: 30, 3: 30, 4: 15}
    assert params.ft_state_value(1) == 0 and params.ft_state_value(3) == pytest.approx(3.6)
    with pytest.raises(ValueError):
        OptimizerParams(ft_value={2: -1.0})
