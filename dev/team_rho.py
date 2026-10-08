"""Fit the Dixon-Coles ρ of the team model once on develop (Phase 5 plan, Task 2; PLAN §6.1).
Dev only: never imported by `fplopt`.

For every develop match (2016/17–2022/23) with football-data pre-match `avg` odds, and each
ρ on a grid: de-vig (power), solve the market λ under that ρ (`market_lambdas`), and score
the realized result under the Dixon-Coles distribution with (λ_home, λ_away, ρ). ρ is the
argmax of the summed log-likelihood (profile likelihood): the low-score correction under
which the market-implied score distributions fit the results best. A coarse grid, then a
fine one around its maximum. The result is hard-coded as `fplopt.models.team.RHO`.

Reads the built tables read-only through `DataStore` (as of 2023-07-01: develop results, no
validate or holdout match).

    uv run python dev/team_rho.py [--data-dir PATH] [--devig power|shin]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fplopt.features.store import DataStore
from fplopt.models import team

DEVELOP = range(2016, 2023)
AS_OF = "2023-07-01T00:00Z"  # after develop's last match, before 2023/24


def develop_markets(store: DataStore, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(de-vigged markets, realized scores) of develop matches with pre-match odds."""
    view = store.as_of(AS_OF)
    matches = view.table("fixture", columns=["fixture_key", "season", "home_goals", "away_goals"])
    matches = matches[matches["season"].isin(DEVELOP) & matches["home_goals"].notna()]
    odds = team.visible_odds(view, matches["fixture_key"])
    markets = team.market_probabilities(odds, method)
    matches = matches[matches["fixture_key"].isin(markets["fixture_key"])]
    return markets, matches.sort_values("fixture_key").reset_index(drop=True)


def log_likelihood(markets: pd.DataFrame, scores: pd.DataFrame, rho: float) -> float:
    solved = team.market_lambdas(markets, rho)
    solved = solved.merge(scores, on="fixture_key")
    solved = solved[solved["success"]]
    matrix = team.dc_matrix(solved["lambda_home"], solved["lambda_away"], rho)
    home = np.minimum(solved["home_goals"].to_numpy(dtype="int64"), team.MAX_GOALS)
    away = np.minimum(solved["away_goals"].to_numpy(dtype="int64"), team.MAX_GOALS)
    return float(np.log(matrix[np.arange(len(solved)), home, away]).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = os.environ.get("FPLOPT_DATA_DIR", str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--data-dir", default=default)
    parser.add_argument("--devig", default="power", choices=team.DEVIG_METHODS)
    args = parser.parse_args()
    store = DataStore(data_dir=args.data_dir)
    markets, scores = develop_markets(store, args.devig)
    print(f"{len(scores)} develop matches with pre-match odds ({args.devig} de-vig)")
    start = time.perf_counter()
    coarse = {round(r, 4): log_likelihood(markets, scores, r) for r in np.arange(-0.2, 0.051, 0.02)}
    best = max(coarse, key=coarse.get)
    fine = {
        round(r, 4): log_likelihood(markets, scores, r)
        for r in np.arange(best - 0.02, best + 0.0201, 0.005)
    }
    grid = {**coarse, **fine}
    for rho in sorted(grid):
        print(f"rho {rho:+.3f}  log-lik {grid[rho]:.2f}  (vs rho 0: {grid[rho] - grid[0.0]:+.2f})")
    best = max(grid, key=grid.get)
    print(f"best rho {best:+.3f} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
