"""Point-in-time features. Every builder takes one argument, an `AsOfView`
(`DataStore(data_dir).as_of(deadline)`), and reads only through it (PLAN §4)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from fplopt.features.baseline import (
    availability,
    ep_next,
    player_pool,
    recent_form,
    team_strength,
    upcoming_fixtures,
)
from fplopt.features.store import AsOfView, DataStore

FeatureBuilder = Callable[[AsOfView], pd.DataFrame]

FEATURES: dict[str, FeatureBuilder] = {
    "player_pool": player_pool,
    "availability": availability,
    "ep_next": ep_next,
    "recent_form": recent_form,
    "upcoming_fixtures": upcoming_fixtures,
    "team_strength": team_strength,
}


def compute_features(view: AsOfView) -> dict[str, pd.DataFrame]:
    """Every registered feature at the view's deadline, in FEATURES order."""
    return {name: builder(view) for name, builder in FEATURES.items()}


__all__ = ["FEATURES", "AsOfView", "DataStore", "FeatureBuilder", "compute_features"]
