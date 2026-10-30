"""Chip terminal values estimated from backtests (chip pre-registration §3,
docs/superpowers/plans/2026-10-11-chip-policy.md; PLAN §7 *Chips*: "the expected best use of
that chip in the rest of its window beyond the horizon, estimated from backtest
distributions").

Input: the GW rows of a `chipgains` run (S0: the optimizer without chips, recording each
chip's ex-ante gain at every GW: `gain_tc`, `gain_bb`, `gain_fh`, `gain_wc`). For a chip and
`L` GWs left in its window after the horizon, the value is the mean, over runs (season,
start) and window positions with exactly `L` GWs left, of the best gain among those `L` GWs
(the GWs `stop - L + 1 .. stop` of the window, by `gw_index`). Develop seasons only: the
validate seasons must not inform what is tested on them.

Windows are the 2026/27 rules' on `gw_index` (`backtest_rules`), as in every develop
season (PLAN §5). The best of `L` noisy forecasts overstates what a manager can get, which
the policy spec's `chip_shrink` (chosen on develop) corrects.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from fplopt.seasons import DEVELOP_SEASONS

__all__ = (
    "CHIP_VALUES_FILE",
    "DOUBLE_VALUES_FILE",
    "GAIN_COLUMNS",
    "MIN_CELL",
    "SPLITS",
    "WINDOW_VALUES_FILE",
    "estimate_loso",
    "estimate_terminal_values",
    "load_loso_values",
    "load_terminal_values",
)

GAIN_COLUMNS = {"3xc": "gain_tc", "bboost": "gain_bb", "freehit": "gain_fh", "wildcard": "gain_wc"}
# The estimated table (`fplopt chips values`; the chip study's, PLAN §12).
CHIP_VALUES_FILE = Path(__file__).resolve().parents[3] / "config" / "chips" / "terminal-values.json"
# Chip policy 2 (docs/superpowers/specs/2026-10-30-chip-policy-2-design.md): per-window
# and per-window-and-double tables, each with a leave-one-season-out section.
WINDOW_VALUES_FILE = CHIP_VALUES_FILE.with_name("terminal-values-window.json")
DOUBLE_VALUES_FILE = CHIP_VALUES_FILE.with_name("terminal-values-double.json")
SPLITS = ("pooled", "window", "double")
# A double-split cell with fewer (run, window) observations falls back to its window value.
MIN_CELL = 30


def load_terminal_values(
    shrink: float = 1.0, path: Path | None = None
) -> dict[str, tuple[float, ...]]:
    """The estimated terminal value table times `shrink` (ValueError if it hasn't been
    estimated yet or `shrink` < 0): `OptimizerParams.chip_value_left`."""
    path = CHIP_VALUES_FILE if path is None else path
    if not math.isfinite(shrink) or shrink < 0:
        raise ValueError(f"chip_shrink must be a finite number >= 0, got {shrink}")
    if not path.is_file():
        raise ValueError(f"chip_shrink: no estimated chip values at {path} (fplopt chips values)")
    table = json.loads(path.read_text(encoding="utf-8"))["values"]
    return {name: tuple(shrink * float(v) for v in values) for name, values in table.items()}


def load_loso_values(
    shrink: float = 1.0, path: Path | None = None
) -> dict[int, dict[str, tuple[float, ...]]]:
    """The leave-one-season-out tables of a v2 file (season → table, each × `shrink`):
    season s's table was estimated without s. ValueError if the file has none."""
    path = WINDOW_VALUES_FILE if path is None else path
    if not math.isfinite(shrink) or shrink < 0:
        raise ValueError(f"chip_shrink must be a finite number >= 0, got {shrink}")
    if not path.is_file():
        raise ValueError(f"loso: no estimated chip values at {path} (fplopt chips values)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "loso" not in payload:
        raise ValueError(f"{path}: no leave-one-season-out tables (chips values --loso)")
    return {
        int(season): {name: tuple(shrink * float(v) for v in values) for name, values in t.items()}
        for season, t in payload["loso"].items()
    }


def estimate_terminal_values(
    gws: pd.DataFrame,
    split: str = "pooled",
    doubles: Mapping[tuple[int, int], frozenset[int]] | None = None,
    horizon: int = 6,
) -> dict[str, tuple[float, ...]]:
    """Per table key, the values for L = 1 .. the longest window (module docstring).
    `split`: "pooled" keys by chip name (all windows together); "window" by `name@k`, k the
    window's ordinal; "double" by `name@k+d` as well as `name@k`, d = 1 when a double was
    known (`doubles[(season, decision gw_index)]`, `chip_doubles.known_doubles_by_gw`) in the
    window's last L GWs at the decision GW max(1, stop - L - horizon + 1); a `+d` cell with
    fewer than `MIN_CELL` observations takes the `name@k` value."""
    from fplopt.backtest.rules import backtest_rules

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "double" and doubles is None:
        raise ValueError("split='double' needs the known doubles (chip_doubles)")
    seasons = sorted(int(s) for s in gws["season"].unique())
    outside = [s for s in seasons if s not in DEVELOP_SEASONS]
    if outside:
        raise ValueError(f"chip values are estimated on develop seasons only, got {outside}")
    missing = [c for c in GAIN_COLUMNS.values() if c not in gws.columns]
    if missing or gws[list(GAIN_COLUMNS.values())].isna().all().all():
        raise ValueError("no chip gains in these GW rows: run `backtest run --spec chipgains:...`")
    rules = backtest_rules(max(DEVELOP_SEASONS))
    sums: dict[tuple[str, int], float] = defaultdict(float)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    # Keys in a fixed order (chip, window, +0/+1) with their longest window, whatever the data.
    longest: dict[str, int] = {}
    chip_windows: dict[str, list[tuple[int, int]]] = {}
    for name in GAIN_COLUMNS:
        chip_windows[name] = sorted((w.start, w.stop) for w in rules.chips if w.name == name)
        for k, (start, stop) in enumerate(chip_windows[name], start=1):
            length = stop - start + 1
            keys = [name] if split == "pooled" else [f"{name}@{k}"]
            if split == "double":
                keys += [f"{name}@{k}+0", f"{name}@{k}+1"]
            for key in keys:
                longest[key] = max(longest.get(key, 0), length)
    runs = [
        (int(season), run)
        for (season, _, _), run in gws.groupby(["season", "policy", "start_id"], sort=True)
    ]
    for name, column in GAIN_COLUMNS.items():
        windows = chip_windows[name]
        for season, run in runs:
            gains = {
                int(i): float(g)
                for i, g in zip(run["gw_index"], run[column], strict=True)
                if not pd.isna(g)
            }
            for k, (start, stop) in enumerate(windows, start=1):
                base = name if split == "pooled" else f"{name}@{k}"
                best = -math.inf
                for left in range(1, stop - start + 2):  # GWs left: stop-left+1 .. stop
                    gw = stop - left + 1
                    if gw in gains:
                        best = max(best, gains[gw])
                    if best == -math.inf:
                        continue
                    cell = [base]
                    if split == "double":
                        decision = max(1, stop - left - horizon + 1)
                        ahead = doubles.get((season, decision), frozenset())
                        d = int(any(stop - left < g <= stop for g in ahead))
                        cell.append(f"{base}+{d}")
                    for key in cell:
                        sums[key, left] += best
                        counts[key, left] += 1
    out = {}
    for key, length in longest.items():
        parent = key.split("+")[0] if "+" in key else None
        values: list[float] = []
        for left in range(1, length + 1):
            src = key
            if parent is not None and counts.get((key, left), 0) < MIN_CELL:
                src = parent
            n = counts.get((src, left), 0)
            values.append(sums[src, left] / n if n else (values[-1] if values else 0.0))
        out[key] = tuple(max(0.0, v) for v in values)
    return out


def estimate_loso(
    gws: pd.DataFrame,
    split: str = "window",
    doubles: Mapping[tuple[int, int], frozenset[int]] | None = None,
    horizon: int = 6,
) -> dict[int, dict[str, tuple[float, ...]]]:
    """Season s → `estimate_terminal_values` on every other season's rows: the table a
    backtest of season s uses, so no season's own outcomes inform its holding values."""
    seasons = sorted(int(s) for s in gws["season"].unique())
    if len(seasons) < 2:
        raise ValueError(f"leave-one-season-out needs at least two seasons, got {seasons}")
    return {
        s: estimate_terminal_values(gws[gws["season"] != s], split, doubles, horizon)
        for s in seasons
    }
