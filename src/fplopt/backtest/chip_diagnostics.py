"""Chip diagnostics (chip research, sub-project 1; spec
docs/superpowers/specs/2026-10-29-chip-diagnostics-design.md): where chip value comes from,
read from backtest GW rows (`backtest run` -> gws.parquet) and the schedule. Pure functions
over frames; `dev/chip_diagnostics.py` loads the inputs and writes the tables.

S0 rows (`chipgains`) carry each chip's ex-ante gain (`gain_tc`, `gain_bb`, `gain_fh`,
`gain_wc`); since S0 plays no chip, its `captain_points` and `bench_points` are the realized
gains a Triple Captain and a Bench Boost would have added in that GW.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = (
    "CELL",
    "KEYS",
    "WC_SPAN",
    "attribute_chips",
    "gap_by_cell",
    "gap_by_chip",
    "paired",
)

KEYS = ["season", "start_id", "gw_index"]
WC_SPAN = 6  # GWs a Wildcard is credited with: its own and the next 5 (the plan horizon)
CELL = 8  # GWs per cell, as the chip study's full-run window t-test


def attribute_chips(gws: pd.DataFrame) -> pd.Series:
    """Per row, the chip its points are attributed to: the chip played that GW, else
    "wildcard" within `WC_SPAN` GWs of a Wildcard in the same run (season, start_id), else
    "none". Index as `gws`."""
    out = pd.Series("none", index=gws.index, dtype="object")
    ordered = gws.sort_values(KEYS, kind="mergesort")
    for _, run in ordered.groupby(["season", "start_id"], sort=False):
        left = 0  # GWs of a Wildcard's span still to credit
        for idx, chip in zip(run.index, run["chip"], strict=True):
            if isinstance(chip, str) and chip:
                out[idx] = chip
                if chip == "wildcard":
                    left = WC_SPAN - 1
                    continue
            elif left > 0:
                out[idx] = "wildcard"
            left = max(0, left - 1)
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """The GW rows of `a` and `b` matched on `KEYS`: `diff` / `diff_xg` (a − b net points,
    realized / xG-scored) and each side's attributed chip (`chip_a`, `chip_b`)."""
    cols = KEYS + ["net_points", "xg_net_points"]
    left = a[cols].assign(chip=attribute_chips(a))
    right = b[cols].assign(chip=attribute_chips(b))
    m = left.merge(right, on=KEYS, suffixes=("_a", "_b"))
    if not len(m) == len(a) == len(b):
        raise ValueError(f"runs don't match: {len(a)} and {len(b)} rows, {len(m)} paired")
    m["diff"] = (m["net_points_a"] - m["net_points_b"]).astype("float64")
    m["diff_xg"] = (m["xg_net_points_a"] - m["xg_net_points_b"]).astype("float64")
    return m


def gap_by_chip(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a − b by the pair of attributed chips: n, mean_diff, mean_diff_xg and
    contribution_per_gw (the group's summed diff over all paired rows: the contributions sum
    to the mean gap per GW). Sorted by contribution, largest first."""
    m = paired(a, b)
    total = len(m)
    table = (
        m.groupby(["chip_a", "chip_b"])
        .agg(
            n=("diff", "size"),
            mean_diff=("diff", "mean"),
            mean_diff_xg=("diff_xg", "mean"),
            sum_diff=("diff", "sum"),
        )
        .reset_index()
    )
    table["contribution_per_gw"] = table["sum_diff"] / total
    return table.drop(columns="sum_diff").sort_values(
        "contribution_per_gw", ascending=False, key=np.abs, ignore_index=True
    )


def gap_by_cell(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a − b per `CELL`-GW cell of gw_index (1 = GWs 1–8): n, mean_diff, mean_diff_xg."""
    m = paired(a, b)
    m["cell"] = (m["gw_index"] - 1) // CELL + 1
    return (
        m.groupby("cell")
        .agg(n=("diff", "size"), mean_diff=("diff", "mean"), mean_diff_xg=("diff_xg", "mean"))
        .reset_index()
    )
