"""xP models (PLAN §5 *Baselines*, §6). Every model takes one argument, an `AsOfView`
(`fplopt.features.store.DataStore(data_dir).as_of(deadline)`), reads only through it
(feature builders and view methods) and returns an xP frame: one row per `player_pool` player
and GW from the target GW to `HORIZON` GWs later, columns `player_key, gw, gw_index,
horizon, xp` (int64 x 4, float64; 0 when unknown), sorted by (player_key, horizon).

`MAX_HORIZON` (= `HORIZON` + 1) is the number of GWs a frame covers at most: a planner
horizon above it would be silently cut to it, so the policies refuse one.

A model may add float columns after those (`v1`'s components); consumers that need only
xP read only the five standard columns.

Models fitted walk-forward are `FittedModel`s (`fplopt.models.fitted`): callable on the view
like the others, with `fit` / `predict` exposed so callers can memoize fits by cutoff.

`MODELS` is the registry the backtester, the CLI and the corrupt-the-future check
(`model:<name>`) use. Model modules follow the feature modules' static rules
(tests/test_features_architecture.py): no file access, `AsOfView` only from the store, no
state between calls.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from fplopt.features.store import AsOfView
from fplopt.models.assemble import fit_v1, predict_v1
from fplopt.models.baseline import HORIZON, xp_ep_next, xp_ep_next_fade, xp_rolling
from fplopt.models.fitted import FittedModel

XpModel = Callable[[AsOfView], pd.DataFrame]
MAX_HORIZON = HORIZON + 1  # GWs in an xP frame at most: the target GW and HORIZON more

MODELS: dict[str, XpModel] = {
    "rolling": xp_rolling,
    "ep_next": xp_ep_next,
    "ep_next_fade": xp_ep_next_fade,
    # Phase 5 component model (fplopt.models.assemble): walk-forward fits, calibrated xP;
    # its frames carry per player-GW components (assemble.GW_COMPONENTS) after the xP
    # columns, which the evaluation scores and the backtester ignores.
    "v1": FittedModel(fit_v1, predict_v1),
}

__all__ = ("MAX_HORIZON", "MODELS", "FittedModel", "XpModel")
