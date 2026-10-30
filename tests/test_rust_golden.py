"""Golden files the Rust port (crates/) is tested against: Python's outputs on a fixed set of
inputs. This test fails if Python's behaviour drifts from a golden file; regenerate with
`uv run python tests/test_rust_golden.py` (and the Rust tests then check the new values).
`names.json.gz` comes from real data (`dev/name_golden.py --data data`); here it is only
re-checked against the installed rapidfuzz and `normalise_name`. `bot/` is a scripted bot
session on the synthetic league (`tests/bot_replay.py`): Python's replies and database rows,
which `crates/fplopt-bot/tests/golden.rs` replays. `lgbm.json.gz`: LightGBM boosters the wheel
trains through `fplopt.models.gbm` on synthetic rows (the model text and predictions
`crates/fplopt-lgbm` must reproduce byte for byte, with the parameter strings `lightgbm.train`
hands the C API)."""

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fplopt.seasons import (
    DEVELOP_SEASONS,
    HOLDOUT_SEASONS,
    VALIDATE_SEASONS,
    bootstrap_season,
    football_data_code,
    parse_season_label,
    season_label,
    season_start_year,
)

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "rust_golden"

PARSE_CASES = [
    "2016-17", "2099-00", "1999-00", "2025-26", "2016-18", "16-17", "2016/17", "2016-17 ",
    " 2016-17", "", "2016-1", "2016-170", "abcd-ef", "2016-16", "0000-01", "9999-00",
]  # fmt: skip


def seasons_golden() -> dict:
    years = range(1990, 2101)
    parsed = {}
    for label in PARSE_CASES:
        try:
            parsed[label] = parse_season_label(label)
        except ValueError:
            parsed[label] = None
    instants = []
    for year in range(2015, 2031):
        for month in range(1, 13):
            first = datetime(year, month, 1, tzinfo=UTC)
            for when in (first, first - timedelta(seconds=1)):
                instants.append([when.isoformat(), season_start_year(when)])
    bootstraps = [
        {"events": [{"id": 2, "deadline_time": "2026-08-28T17:30:00Z"},
                    {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"}]},
        {"events": [{"id": 1, "deadline_time": "2020-09-12T10:00:00Z"}]},
        {"events": [{"id": 38, "deadline_time": "2017-05-21T13:30:00Z"},
                    {"id": 1, "deadline_time": "2016-08-13T10:30:00Z"}]},
    ]  # fmt: skip
    return {
        "season_label": {str(y): season_label(y) for y in years},
        "football_data_code": {str(y): football_data_code(y) for y in years},
        "parse_season_label": parsed,
        "season_start_year": instants,
        "bootstrap_season": [[b, bootstrap_season(b)] for b in bootstraps],
        "develop": sorted(DEVELOP_SEASONS),
        "validate": sorted(VALIDATE_SEASONS),
        "holdout": sorted(HOLDOUT_SEASONS),
    }


def test_seasons_golden_is_current():
    stored = json.loads((GOLDEN / "seasons.json").read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(seasons_golden()))


def test_names_golden_is_current():
    from rapidfuzz import fuzz

    from fplopt.build.understat import normalise_name

    golden = json.loads(gzip.decompress((GOLDEN / "names.json.gz").read_bytes()))
    assert {n: normalise_name(n) for n in golden["names"]} == golden["names"]
    assert all(fuzz.token_set_ratio(a, b) == score for a, b, score in golden["pairs"])


LGBM_CASES = [
    # (name, seed, rows, weights, GbmParams overrides)
    ("weighted", 0, 600, "decay", {"num_boost_round": 30, "num_leaves": 7, "min_data_in_leaf": 20}),
    ("unit_weight", 1, 500, "ones", {"num_boost_round": 25, "num_leaves": 5,
                                     "min_data_in_leaf": 30}),
    ("no_weight", 2, 400, None, {"num_boost_round": 20, "feature_fraction": 0.5, "lambda_l2": 0.3}),
]  # fmt: skip


def lgbm_golden() -> dict:
    import numpy as np
    import pandas as pd
    from lightgbm.basic import _param_dict_to_str

    from fplopt.models.gbm import GbmParams, train

    cases = []
    for name, seed, n, weights, overrides in LGBM_CASES:
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.integers(0, 90, size=n).astype("float64"),
                "c": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)),
                "d": rng.choice([1.0, 2.0, 3.0, 4.0], size=n),
            }
        )
        logit = 0.8 * frame["a"] + frame["b"] / 45 - 1 + frame["c"].fillna(0.5) * frame["d"] / 2
        frame["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("float64")
        if weights == "decay":
            frame["w"] = 0.85 ** rng.integers(0, 6, size=n)
        elif weights == "ones":
            frame["w"] = 1.0
        params = GbmParams(**overrides)
        weight = None if weights is None else "w"
        gbm = train(frame, list("abcd"), "y", weight=weight, params=params)
        text = gbm.booster.model_to_string()
        text = text[: text.rindex("\npandas_categorical:")]  # the package's own suffix
        lgb = params.lightgbm_params()
        rounds = params.num_boost_round
        x = frame[list("abcd")].to_numpy()
        cases.append(
            {
                "name": name,
                "features": list("abcd"),
                "x": [[None if np.isnan(v) else v for v in row] for row in x],
                "label": frame["y"].tolist(),
                "weight": None if weights is None else frame["w"].tolist(),
                "dataset_params": _param_dict_to_str(
                    {"verbosity": -1, **lgb, "num_iterations": rounds}
                ),
                "booster_params": _param_dict_to_str({**lgb, "num_iterations": rounds}),
                "rounds": rounds,
                "model": text,
                "predictions": gbm.predict(frame).tolist(),
            }
        )
    return {"cases": cases}


def test_lgbm_golden_is_current():
    stored = json.loads(gzip.decompress((GOLDEN / "lgbm.json.gz").read_bytes()))
    assert stored == json.loads(json.dumps(lgbm_golden()))


def test_bot_golden_is_current(tmp_path):
    import shutil

    from bot_replay import dump_db, replay

    root = GOLDEN / "bot"
    shutil.copytree(root / "state", tmp_path / "state")
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    results = replay(script, tmp_path / "state", root / "fpl")
    assert results == json.loads((root / "results.json").read_text(encoding="utf-8"))
    db = json.loads(json.dumps(dump_db(tmp_path / "state" / "bot.sqlite")))
    assert db == json.loads((root / "db.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    from bot_replay import golden

    golden(GOLDEN / "bot")
    GOLDEN.mkdir(parents=True, exist_ok=True)
    text = json.dumps(seasons_golden(), indent=1, sort_keys=True) + "\n"
    (GOLDEN / "seasons.json").write_text(text, encoding="utf-8", newline="\n")
    lgbm = json.dumps(lgbm_golden(), separators=(",", ":")).encode()
    (GOLDEN / "lgbm.json.gz").write_bytes(gzip.compress(lgbm, mtime=0))
