"""Price sizing 2 (spec docs/superpowers/specs/2026-11-06-price-sizing-design.md): the money
perfect timing within a GW would have saved on our system's transfers.

    uv run python dev/price_timing.py --run results/price-sizing-base \
        --data C:/Users/samin/Documents/GitHub/fpl-optimizer/data --out results/price-sizing-timing

For each transfer (out o, in i) at a deadline after gw_index 1, over a window ending at the
deadline (24 h, and since the previous deadline): sell gain = o's best selling price in the
window minus at the deadline; buy gain = i's deadline price minus its lowest in the window.
Purchase prices are replayed from the run (a start squad's at the start GW's deadline price).
Seasons 2021 and 2022 only (develop seasons with daily snapshots). Writes timing.json:
tenths of £m per season and start, per window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SEASONS = (2021, 2022)


def sell_price(purchase: int, price: int) -> int:
    return price if price <= purchase else purchase + (price - purchase) // 2


def timing_gain(purchase, window_out, deadline_out, window_in, deadline_in) -> tuple[int, int]:
    at = sell_price(purchase, deadline_out)
    best = max([sell_price(purchase, p) for p in window_out] + [at])
    return best - at, deadline_in - min(list(window_in) + [deadline_in])


def replay(gws: pd.DataFrame, deadline_price: dict, snaps: dict, window) -> list[dict]:
    """deadline_price[(season, gw, player_key)] = price at that GW's deadline; snaps[key] =
    [(snapshot_at, price)] sorted; window: a Timedelta, or None for "since the previous
    deadline". One row per GW with transfers after gw_index 1."""
    out = []
    for (season, start_id), unit in gws.groupby(["season", "start_id"], sort=True):
        unit = unit.sort_values("gw_index")
        first_gw = int(unit["gw"].iloc[0])
        purchase: dict[int, int] = {}
        prev_deadline = None
        for row in unit.itertuples():
            pairs = json.loads(row.transfers)
            start = prev_deadline if window is None else row.deadline - window
            if window is None and start is None:
                start = row.deadline - pd.Timedelta(days=7)
            sell_gain = buy_gain = 0
            for o, i in pairs:
                p_o = purchase.get(o, deadline_price.get((season, first_gw, o)))
                d_o = deadline_price.get((season, row.gw, o))
                d_i = deadline_price.get((season, row.gw, i))
                if row.gw_index > 1 and None not in (p_o, d_o, d_i):
                    w_o = [p for t, p in snaps.get(o, []) if start <= t < row.deadline]
                    w_i = [p for t, p in snaps.get(i, []) if start <= t < row.deadline]
                    s, b = timing_gain(p_o, w_o, d_o, w_i, d_i)
                    sell_gain, buy_gain = sell_gain + s, buy_gain + b
                purchase.pop(o, None)
                if d_i is not None:
                    purchase[i] = d_i
            if pairs and row.gw_index > 1:
                out.append(
                    {
                        "season": season,
                        "start_id": start_id,
                        "gw": row.gw,
                        "sell_gain": sell_gain,
                        "buy_gain": buy_gain,
                    }
                )
            prev_deadline = row.deadline
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gws = pd.read_parquet(args.run / "gws.parquet")
    gws["deadline"] = pd.to_datetime(gws["deadline"], utc=True)
    gws = gws[gws["season"].isin(SEASONS) & ~gws["start_id"].str.contains(r"\+bank")]
    pg = pd.read_parquet(
        args.data / "player_gw.parquet", columns=["season", "gw", "player_key", "value"]
    )
    pg = pg[pg["season"].isin(SEASONS)]
    deadline_price = {
        (int(s), int(g), int(k)): int(v)
        for s, g, k, v in pg[["season", "gw", "player_key", "value"]].itertuples(index=False)
    }
    snap = pd.read_parquet(
        args.data / "player_snapshot.parquet",
        columns=["season", "player_key", "snapshot_at", "now_cost"],
    )
    snap = snap[snap["season"].isin(SEASONS)].dropna().sort_values("snapshot_at")
    snap["snapshot_at"] = pd.to_datetime(snap["snapshot_at"], utc=True)
    snaps: dict[int, list] = {}
    for k, t, p in snap[["player_key", "snapshot_at", "now_cost"]].itertuples(index=False):
        snaps.setdefault(int(k), []).append((t, int(p)))
    result = {}
    for name, window in (("24h", pd.Timedelta(hours=24)), ("week", None)):
        rows = pd.DataFrame(replay(gws, deadline_price, snaps, window))
        rows["gain"] = rows["sell_gain"] + rows["buy_gain"]
        per_unit = rows.groupby(["season", "start_id"])["gain"].sum()
        result[name] = {
            "per_unit": {f"{s} {i}": int(v) for (s, i), v in per_unit.items()},
            "mean_per_season": float(per_unit.mean()),
            "sell_share": float(rows["sell_gain"].sum() / max(rows["gain"].sum(), 1)),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["mean_per_season"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
