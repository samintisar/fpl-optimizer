import numpy as np
import pandas as pd
import pytest

from fplopt.backtest.simulator import Caches
from fplopt.features.store import DataStore
from fplopt.models import MODELS
from fplopt.models.fitted import FittedModel, refit_deadline
from fplopt.models.gbm import GbmParams, train

UTC_US = pd.DatetimeTZDtype("us", "UTC")
START = pd.Timestamp("2026-08-14 17:30", tz="UTC")
DEADLINES = [START + pd.Timedelta(days=7 * i) for i in range(10)]
FITS: list[pd.Timestamp] = []


def gameweek() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "season": 2026,
            "gw": range(1, 11),
            "gw_index": range(1, 11),
            "deadline_time": pd.Series(DEADLINES).astype(UTC_US),
        }
    )
    published = pd.Timestamp("2026-06-01", tz="UTC").as_unit("us")
    return df.assign(event_time=df["deadline_time"], available_at=published)


def store() -> DataStore:
    return DataStore(tables={"gameweek": gameweek()})


def fit_cutoff(view):
    FITS.append(view.deadline)
    return view.deadline


def predict_with(view, fitted):
    return pd.DataFrame({"deadline": [view.deadline], "fitted": [fitted]})


FAKE = FittedModel(fit=fit_cutoff, predict=predict_with)


@pytest.mark.parametrize(
    ("gw", "every", "refit_gw"),
    [(1, 4, 1), (4, 4, 1), (5, 4, 5), (8, 4, 5), (9, 4, 9), (7, 1, 7), (10, 3, 10)],
)
def test_refit_deadline_is_the_latest_refit_gw(gw, every, refit_gw):
    view = store().as_of(DEADLINES[gw - 1])
    assert refit_deadline(view, every) == DEADLINES[refit_gw - 1]


def test_refit_deadline_rejects_a_bad_cadence():
    with pytest.raises(ValueError, match="every"):
        refit_deadline(store().as_of(DEADLINES[0]), 0)


def test_a_fitted_model_fits_at_the_cutoff_and_predicts_at_the_deadline():
    FITS.clear()
    out = FAKE(store().as_of(DEADLINES[6]))
    assert FITS == [DEADLINES[4]]
    assert out["deadline"].tolist() == [DEADLINES[6]]
    assert out["fitted"].tolist() == [DEADLINES[4]]


def test_caches_memoize_fits_by_cutoff(monkeypatch):
    monkeypatch.setitem(MODELS, "fake", FAKE)
    FITS.clear()
    caches, data = Caches(), store()
    frames = [caches.xp(data, "fake", data.as_of(d)) for d in DEADLINES[4:9]]
    assert FITS == [DEADLINES[4], DEADLINES[8]]  # GW5-8 share one fit, GW9 refits
    for frame, deadline in zip(frames, DEADLINES[4:9], strict=True):
        pd.testing.assert_frame_equal(frame, FAKE(data.as_of(deadline)))
    assert caches.xp(data, "fake", data.as_of(DEADLINES[5])) is frames[1]
    with pytest.raises(ValueError, match="not fitted walk-forward"):
        caches.fit(data, "rolling", data.as_of(DEADLINES[0]))


def synthetic(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype("float64")
    frame = pd.DataFrame(x, columns=["a", "b", "c"]).assign(y=y, w=rng.uniform(0.5, 2, n))
    frame.loc[::17, "c"] = np.nan  # missing values are allowed
    return frame


def test_gbm_fits_are_byte_identical_and_use_weights():
    frame = synthetic()
    params = GbmParams(num_boost_round=30, min_data_in_leaf=20)
    first = train(frame, ["a", "b", "c"], "y", weight="w", params=params)
    second = train(frame, ["a", "b", "c"], "y", weight="w", params=params)
    assert first.booster.model_to_string() == second.booster.model_to_string()
    assert first.features == ("a", "b", "c")
    p = first.predict(frame)
    assert p.dtype == np.float64 and ((p > 0) & (p < 1)).all()
    np.testing.assert_array_equal(p, second.predict(frame[["c", "b", "a"]]))  # by name
    unweighted = train(frame, ["a", "b", "c"], "y", params=params)
    assert unweighted.booster.model_to_string() != first.booster.model_to_string()
