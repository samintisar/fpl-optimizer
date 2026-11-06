"""Price sizing 2 (spec docs/superpowers/specs/2026-11-06-price-sizing-design.md): the money
perfect timing within a GW would have saved on our system's transfers.

    uv run python dev/price_timing.py --run results/price-sizing-base \
        --data C:/Users/samin/Documents/GitHub/fpl-optimizer/data --out results/price-sizing-timing

For each transfer (out o, in i) at a deadline after gw_index 1, over a window ending at the
deadline (24 h, and since the previous deadline): sell gain = o's best selling price in the
window minus at the deadline; buy gain = i's deadline price minus its lowest in the window.
Purchase prices are replayed from the run (a start squad's at the start GW's deadline price);
deadline prices are the last snapshot before the deadline. Sell and buy best times are taken
independently and a bought player's purchase stays at the deadline price, so the figure is a
loose upper bound. Free Hit GWs are skipped (purchases kept as before them); a wildcard is
treated like any other GW. Seasons 2021 and 2022 only (develop seasons with daily snapshots).
Writes timing.json:
tenths of £m per season and start, per window; `mean_per_season` is over the @1/@2 units,
`mean_gw20_38` over the @20 units.
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


def deadline_prices(deadlines: dict, snap: pd.DataFrame, fallback: dict | None = None) -> dict:
    """{(season, gw, player_key): price at that GW's deadline}: the last `player_snapshot`
    row (`now_cost`) with snapshot_at < the deadline, as the backtester prices; `fallback`
    (player_gw's `value`) only fills keys the snapshots lack. deadlines: {(season, gw): ts}."""
    out = {}
    for (season, gw), deadline in deadlines.items():
        s = snap[(snap["season"] == season) & (snap["snapshot_at"] < deadline)]
        last = s.sort_values("snapshot_at").groupby("player_key")["now_cost"].last()
        for k, v in last.items():
            out[(season, gw, int(k))] = int(v)
    for key, v in (fallback or {}).items():
        out.setdefault(key, v)
    return out


def replay(
    gws: pd.DataFrame, deadline_price: dict, snaps: dict, window, stats: dict | None = None
) -> list[dict]:
    """deadline_price[(season, gw, player_key)] = price at that GW's deadline; snaps[key] =
    [(snapshot_at, price)] sorted; window: a Timedelta, or None for "since the previous
    deadline". One row per GW with transfers, except each unit's first row (its start squad
    did not exist before that deadline). Free Hit rows are skipped and leave the purchase
    prices as they were (the squad reverts); a wildcard is treated like any other GW.
    `stats["skipped"]` counts transfers skipped for a missing price (the bought player's
    price unknown means a later sale of him is skipped too)."""
    stats = stats if stats is not None else {}
    stats.setdefault("skipped", 0)
    out = []
    for (season, start_id), unit in gws.groupby(["season", "start_id"], sort=True):
        unit = unit.sort_values("gw_index")
        first_gw = int(unit["gw"].iloc[0])
        purchase: dict[int, int] = {}
        unknown: set[int] = set()
        prev_deadline = None
        for n, row in enumerate(unit.itertuples()):
            if getattr(row, "chip", None) == "freehit":
                prev_deadline = row.deadline
                continue
            pairs = json.loads(row.transfers)
            start = prev_deadline if window is None else row.deadline - window
            if window is None and start is None:
                start = row.deadline - pd.Timedelta(days=7)
            sell_gain = buy_gain = 0
            for o, i in pairs:
                p_o = None if o in unknown else purchase.get(o)
                if p_o is None and o not in purchase and o not in unknown:
                    p_o = deadline_price.get((season, first_gw, o))
                d_o = deadline_price.get((season, row.gw, o))
                d_i = deadline_price.get((season, row.gw, i))
                if n > 0:
                    if None in (p_o, d_o, d_i):
                        stats["skipped"] += 1
                    else:
                        w_o = [p for t, p in snaps.get(o, []) if start <= t < row.deadline]
                        w_i = [p for t, p in snaps.get(i, []) if start <= t < row.deadline]
                        s, b = timing_gain(p_o, w_o, d_o, w_i, d_i)
                        sell_gain, buy_gain = sell_gain + s, buy_gain + b
                purchase.pop(o, None)
                unknown.discard(o)
                if d_i is not None:
                    purchase[i] = d_i
                else:
                    unknown.add(i)
            if pairs and n > 0:
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


def summarise(rows: list[dict], gws: pd.DataFrame) -> dict:
    """Per-unit totals over every (season, start_id) in gws (0 for no transfers); the mean
    season is over the full-season units (@1, or @2 after the GW1 fallback), GW20-38 apart."""
    df = pd.DataFrame(rows, columns=["season", "start_id", "gw", "sell_gain", "buy_gain"])
    df["gain"] = df["sell_gain"] + df["buy_gain"]
    units = pd.MultiIndex.from_frame(gws[["season", "start_id"]].drop_duplicates())
    per_unit = df.groupby(["season", "start_id"])["gain"].sum().reindex(units, fill_value=0)
    ends = per_unit.index.get_level_values("start_id").str.extract(r"@(\d+)$")[0].to_numpy()
    full = per_unit[[e in ("1", "2") for e in ends]]
    late = per_unit[[e == "20" for e in ends]]
    return {
        "per_unit": {f"{s} {i}": int(v) for (s, i), v in per_unit.items()},
        "mean_per_season": float(full.mean()) if len(full) else float("nan"),
        "mean_gw20_38": float(late.mean()) if len(late) else float("nan"),
        "sell_share": float(df["sell_gain"].sum() / max(df["gain"].sum(), 1)),
    }


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
    fallback = {
        (int(s), int(g), int(k)): int(v)
        for s, g, k, v in pg[["season", "gw", "player_key", "value"]].itertuples(index=False)
    }
    snap = pd.read_parquet(
        args.data / "player_snapshot.parquet",
        columns=["season", "player_key", "snapshot_at", "now_cost"],
    )
    snap = snap[snap["season"].isin(SEASONS)].dropna().sort_values("snapshot_at")
    snap["snapshot_at"] = pd.to_datetime(snap["snapshot_at"], utc=True)
    deadlines = {
        (int(s), int(g)): d
        for s, g, d in gws.drop_duplicates(["season", "gw"])[
            ["season", "gw", "deadline"]
        ].itertuples(index=False)
    }
    deadline_price = deadline_prices(deadlines, snap, fallback)
    snaps: dict[int, list] = {}
    for k, t, p in snap[["player_key", "snapshot_at", "now_cost"]].itertuples(index=False):
        snaps.setdefault(int(k), []).append((t, int(p)))
    result = {}
    for name, window in (("24h", pd.Timedelta(hours=24)), ("week", None)):
        stats: dict = {}
        rows = replay(gws, deadline_price, snaps, window, stats)
        result[name] = {**summarise(rows, gws), "skipped_transfers": stats["skipped"]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["mean_per_season"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
