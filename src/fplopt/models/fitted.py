"""Walk-forward fitted xP models (Phase 5 plan, *Model structure*; PLAN §6 *Phase 5
decisions*).

A `FittedModel` splits an xP model into `fit(view) -> fitted` (frozen parameters) and
`predict(view, fitted) -> xP frame`. Called on a view, it fits on
`view.earlier(cutoff(view))` and predicts at the view. The cutoff is the deadline of the
season's latest refit GW at or before the view's (`gw_index` 1, 1 + every, 1 + 2·every, …).
So it is one callable on the view like every other model, and the corrupt-the-future check
covers the fit too. Callers that predict at many deadlines (the backtester's `Caches`)
memoize `fit` by cutoff; nothing here keeps state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fplopt.features.store import AsOfView

REFIT_EVERY = 4  # GWs between walk-forward refits


def refit_deadline(view: AsOfView, every: int = REFIT_EVERY) -> pd.Timestamp:
    """Deadline of the latest refit GW (`gw_index` ≡ 1 mod `every`) of the view's season at
    or before the view's GW."""
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    season, gw = view.gameweek_for_deadline()
    gameweeks = view.table("gameweek", columns=["season", "gw", "gw_index", "deadline_time"])
    gameweeks = gameweeks[gameweeks["season"] == season]
    target = int(gameweeks.loc[gameweeks["gw"] == gw, "gw_index"].iloc[0])
    refit = gameweeks[gameweeks["gw_index"] == 1 + (target - 1) // every * every]
    if len(refit) != 1:
        raise LookupError(f"season {season}: {len(refit)} gameweeks are the refit GW of GW {gw}")
    return pd.Timestamp(refit["deadline_time"].iloc[0])


@dataclass(frozen=True)
class FittedModel:
    """An xP model fitted walk-forward: `fit` on the view at the refit cutoff, `predict`
    at the deadline. `fit` and `predict` must be module-level functions (the backtester
    pickles models into worker processes)."""

    fit: Callable[[AsOfView], Any]
    predict: Callable[[AsOfView, Any], pd.DataFrame]
    refit_every: int = REFIT_EVERY

    def cutoff(self, view: AsOfView) -> pd.Timestamp:
        return refit_deadline(view, self.refit_every)

    def __call__(self, view: AsOfView) -> pd.DataFrame:
        return self.predict(view, self.fit(view.earlier(self.cutoff(view))))
