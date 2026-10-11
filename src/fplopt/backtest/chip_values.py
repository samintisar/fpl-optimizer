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

import math

import pandas as pd

from fplopt.seasons import DEVELOP_SEASONS

__all__ = ("GAIN_COLUMNS", "estimate_terminal_values")

GAIN_COLUMNS = {"3xc": "gain_tc", "bboost": "gain_bb", "freehit": "gain_fh", "wildcard": "gain_wc"}


def estimate_terminal_values(gws: pd.DataFrame) -> dict[str, tuple[float, ...]]:
    """Per chip name, the values for L = 1 .. the longest window (module docstring)."""
    from fplopt.backtest.rules import backtest_rules

    seasons = sorted(int(s) for s in gws["season"].unique())
    outside = [s for s in seasons if s not in DEVELOP_SEASONS]
    if outside:
        raise ValueError(f"chip values are estimated on develop seasons only, got {outside}")
    missing = [c for c in GAIN_COLUMNS.values() if c not in gws.columns]
    if missing or gws[list(GAIN_COLUMNS.values())].isna().all().all():
        raise ValueError("no chip gains in these GW rows: run `backtest run --spec chipgains:...`")
    rules = backtest_rules(max(DEVELOP_SEASONS))
    out = {}
    for name, column in GAIN_COLUMNS.items():
        windows = [(w.start, w.stop) for w in rules.chips if w.name == name]
        longest = max(stop - start + 1 for start, stop in windows)
        sums = [0.0] * longest
        counts = [0] * longest
        for _, run in gws.groupby(["season", "policy", "start_id"], sort=True):
            gains = {
                int(i): float(g)
                for i, g in zip(run["gw_index"], run[column], strict=True)
                if not pd.isna(g)
            }
            for start, stop in windows:
                best = -math.inf
                for left in range(1, stop - start + 2):  # GWs left: stop-left+1 .. stop
                    gw = stop - left + 1
                    if gw in gains:
                        best = max(best, gains[gw])
                    if best > -math.inf:
                        sums[left - 1] += best
                        counts[left - 1] += 1
        values = []
        for total, n in zip(sums, counts, strict=True):
            values.append(total / n if n else (values[-1] if values else 0.0))
        out[name] = tuple(max(0.0, v) for v in values)
    return out
