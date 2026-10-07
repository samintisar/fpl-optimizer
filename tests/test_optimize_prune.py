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


def run(keys, et, price, xp, owned=(), buyable=None, prune_n=None, dominated=True, clubs=None):
    """`prune` on lists; every player has his own club unless `clubs` says otherwise."""
    keys = np.asarray(keys)
    return prune(
        keys=keys,
        element_type=np.asarray(et),
        team_key=np.asarray(keys if clubs is None else clubs),
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


def test_dominance_needs_dominators_from_enough_clubs() -> None:
    # GKs: squad_select 2 + 15 // 3 full clubs = dominators from 7 clubs needed.
    keys = list(range(1, 9))
    xp = [[5, 5]] * 7 + [[3, 3]]
    price = [40] * 7 + [45]
    # 1-7 (identical: tie by key, so 7 has 6 dominators) dominate 8, from 7 clubs.
    assert run(keys, [1] * 8, price, xp) == keys[:7]
    # The same 7 dominators from 3 clubs: the club cap could block them all, so 8 stays.
    assert run(keys, [1] * 8, price, xp, clubs=[1, 1, 1, 2, 2, 2, 3, 4]) == keys
    # Six dominators: kept.
    assert run(keys[1:], [1] * 7, price[1:], xp[1:]) == keys[1:]
    # Cheaper but worse in one GW: not dominated.
    worse = [[5, 5]] * 6 + [[5, 1], [3, 3]]
    assert run(keys, [1] * 8, price, worse) == keys


def test_identical_players_tie_by_key_and_departed_never_dominate() -> None:
    keys = [7, 3, 5, 9, 11, 13, 15, 17, 19]
    # Sorted 3, 5, ..., 19: each is dominated by the smaller keys; 17 and 19 have 7+.
    assert run(keys, [1] * 9, [40] * 9, [2.0] * 9) == [3, 5, 7, 9, 11, 13, 15]
    # 3 and 5 can't be bought (left the game; owned): they dominate nobody.
    buyable = [k not in (3, 5) for k in keys]
    kept = run(keys, [1] * 9, [40] * 9, [2.0] * 9, owned=[3, 5], buyable=buyable)
    assert kept == sorted(keys)


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
    from fplopt.optimize.model import solve_plan

    # 12 clubs: dominance needs dominators from up to 10 clubs (squad_select + 5).
    pool = grid_pool(range(1, 13), {1: 1, 2: 3, 3: 3, 4: 2}, seed=seed)
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
        out.append((len(problem.players), solve_plan(problem, params).objective))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_pruning_keeps_the_optimum_on_random_instances(seed: int) -> None:
    pytest.importorskip("highspy")
    (n_pruned, pruned), (n_dom, dominance), (n_all, unpruned) = plans(seed)
    assert n_pruned <= n_dom <= n_all
    assert dominance == pytest.approx(unpruned, abs=1e-6)
    # Top-N is a heuristic: allow the default MIP gap.
    assert pruned >= unpruned * (1 - OptimizerParams().mip_gap) - 1e-6


def test_default_params_are_read_only() -> None:
    params = OptimizerParams()
    assert isinstance(params.prune_n, MappingProxyType)
    assert dict(params.prune_n) == {1: 20, 2: 60, 3: 60, 4: 30}
    assert params.ft_state_value(1) == 0 and params.ft_state_value(3) == pytest.approx(3.6)
    with pytest.raises(ValueError):
        OptimizerParams(ft_value={2: -1.0})
