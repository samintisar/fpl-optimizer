"""xP models (PLAN §5 *Baselines*, §6). Every model takes one argument, an `AsOfView`
(`fplopt.features.store.DataStore(data_dir).as_of(deadline)`), reads only through it
(feature builders and view methods) and returns an xP frame: one row per `player_pool` player
and GW from the target GW to `HORIZON` GWs later, columns `player_key, gw, gw_index,
horizon, xp` (int64 x 4, float64; 0 when unknown), sorted by (player_key, horizon).

`MODELS` is the registry the backtester, the CLI and the corrupt-the-future check
(`model:<name>`) use. Model modules follow the feature modules' static rules
(tests/test_features_architecture.py): no file access, `AsOfView` only from the store, no
state between calls.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from fplopt.features.store import AsOfView
from fplopt.models.baseline import xp_ep_next, xp_ep_next_fade, xp_rolling

XpModel = Callable[[AsOfView], pd.DataFrame]

MODELS: dict[str, XpModel] = {
    "rolling": xp_rolling,
    "ep_next": xp_ep_next,
    "ep_next_fade": xp_ep_next_fade,
}

__all__ = ("MODELS", "XpModel")
