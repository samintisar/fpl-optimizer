"""Golden files the Rust port (crates/) is tested against: Python's outputs on a fixed set of
inputs. This test fails if Python's behaviour drifts from a golden file; regenerate with
`uv run python tests/test_rust_golden.py` (and the Rust tests then check the new values)."""

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


if __name__ == "__main__":
    GOLDEN.mkdir(parents=True, exist_ok=True)
    text = json.dumps(seasons_golden(), indent=1, sort_keys=True) + "\n"
    (GOLDEN / "seasons.json").write_text(text, encoding="utf-8", newline="\n")
