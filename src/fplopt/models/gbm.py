"""The only LightGBM entry point (PLAN §4 determinism, §6 *Phase 5 decisions*).

Every fit uses `DETERMINISTIC` (fixed seed, one thread, `deterministic`, row-wise
histograms), so the same rows give a byte-identical booster and the corrupt-the-future
check can compare predictions exactly. Callers sort their rows by key before fitting. Model
modules import `train` / `Gbm` from here, never `lightgbm` itself
(tests/test_features_architecture.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

import lightgbm
import numpy as np
import pandas as pd

DETERMINISTIC = MappingProxyType(
    {
        "seed": 0,
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
)


@dataclass(frozen=True)
class GbmParams:
    """Learning parameters (tuned on develop); `DETERMINISTIC` is always added."""

    objective: str = "binary"
    num_boost_round: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 200
    feature_fraction: float = 1.0
    lambda_l2: float = 1.0

    def lightgbm_params(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "lambda_l2": self.lambda_l2,
            **DETERMINISTIC,
        }


@dataclass(frozen=True)
class Gbm:
    """A fitted booster and the feature columns it reads, in order."""

    booster: lightgbm.Booster
    features: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[list(self.features)].to_numpy(dtype="float64")
        return np.asarray(self.booster.predict(x, num_threads=1), dtype="float64")


def train(
    frame: pd.DataFrame,
    features: Sequence[str],
    label: str,
    weight: str | None = None,
    params: GbmParams | None = None,
) -> Gbm:
    """Fit on `frame` (rows in a fixed order; features as float64, NaN = missing)."""
    params = GbmParams() if params is None else params
    features = tuple(features)
    data = lightgbm.Dataset(
        frame[list(features)].to_numpy(dtype="float64"),
        label=frame[label].to_numpy(dtype="float64"),
        weight=None if weight is None else frame[weight].to_numpy(dtype="float64"),
        feature_name=list(features),
        params={"verbosity": -1},
    )
    booster = lightgbm.train(params.lightgbm_params(), data, num_boost_round=params.num_boost_round)
    return Gbm(booster, features)
