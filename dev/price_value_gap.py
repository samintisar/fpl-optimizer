"""Price sizing 1a (spec docs/superpowers/specs/2026-11-06-price-sizing-design.md): FPL team
value of sampled managers vs our system's 2026/27 backtest.

    uv run python dev/price_value_gap.py fetch --out results/price-sizing-managers [--seed 0]
    uv run python dev/price_value_gap.py gap --managers results/price-sizing-managers \
        --ours results/price-sizing-2026 [--entry 4570069]

`fetch` samples 500 ranks in 1-10k and 500 in 10k-100k of the overall league (314) as it
stands now (survivorship: managers ranked there now), saves each standings page and
history as .json.gz under --out (resumable; never raw/), one request per --pause seconds,
and writes managers.parquet (group, rank, entry, gw, value, bank). `gap` compares their
median FPL team value with ours (`market_value` in gws.parquet) at the last GW both have
and scales the gap to 38 GWs linearly; writes gap.json next to the managers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://fantasy.premierleague.com/api/"
GROUPS = {"top10k": (1, 10_000), "10k-100k": (10_001, 100_000)}
PER_PAGE = 50


def sample_ranks(seed: int = 0, n: int = 500) -> dict[str, list[int]]:
    rng = random.Random(seed)
    return {g: sorted(rng.sample(range(lo, hi + 1), n)) for g, (lo, hi) in GROUPS.items()}


def page_of(rank: int) -> int:
    return (rank - 1) // PER_PAGE + 1


def slot_of(rank: int) -> int:
    return (rank - 1) % PER_PAGE


def get_cached(path: Path, url: str, pause: float) -> dict:
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    req = urllib.request.Request(API + url, headers={"User-Agent": "fpl-optimizer"})
    for attempt in range(5):
        try:
            body = urllib.request.urlopen(req, timeout=30).read()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(30 * (attempt + 1))
    time.sleep(pause)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body))
    return json.loads(body)


def fetch(out: Path, seed: int, pause: float) -> pd.DataFrame:
    rows = []
    seen: set[int] = set()  # an entry drawn twice (or in both groups) counts once
    for group, ranks in sample_ranks(seed).items():
        for i, rank in enumerate(ranks):
            page_path = out / "standings" / f"{page_of(rank)}.json.gz"
            page = get_cached(
                page_path,
                f"leagues-classic/314/standings/?page_standings={page_of(rank)}",
                pause,
            )
            entry = page["standings"]["results"][slot_of(rank)]["entry"]
            if entry in seen:
                continue
            seen.add(entry)
            history = get_cached(
                out / "history" / f"{entry}.json.gz", f"entry/{entry}/history/", pause
            )
            fetched_at = pd.Timestamp(page_path.stat().st_mtime, unit="s", tz="UTC")
            for h in history["current"]:
                rows.append(
                    {
                        "group": group,
                        "rank": rank,
                        "entry": entry,
                        "gw": h["event"],
                        "value": h["value"],
                        "bank": h["bank"],
                        "page_fetched_at": fetched_at,
                    }
                )
            if i % 50 == 0:
                print(f"{group}: {i}/{len(ranks)}", flush=True)
    managers = pd.DataFrame(rows)
    managers.to_parquet(out / "managers.parquet", index=False)
    return managers


def value_gap(
    managers: pd.DataFrame, ours: pd.DataFrame, n_gws: int = 38, fpl_lag: int = 0
) -> dict:
    """Per group: the group's median `value` at the last GW both sides have, ours
    (`market_value` of our row for that GW), the gap and the gap scaled to n_gws.

    fpl_lag: 0 if FPL's history `value` for event g is the squad priced at g's deadline,
    like our row g (verified on entry 4570069, GW3-6: value[g] = squad at g's prices + bank);
    1 if it is priced at g+1's deadline, then the managers' event g is compared with our
    row g+1 (their gw is shifted by +1)."""
    managers = managers.assign(gw=managers["gw"] + fpl_lag)
    out = {}
    for group, m in managers.groupby("group"):
        gw = int(min(m["gw"].max(), ours["gw"].max()))
        at = m[m["gw"] == gw]["value"]
        mine = int(ours.loc[ours["gw"] == gw, "market_value"].iloc[0])
        g = float(at.median()) - mine
        out[group] = {
            "gw": gw,
            "n": int(at.size),
            "median": float(at.median()),
            "q25": float(at.quantile(0.25)),
            "q75": float(at.quantile(0.75)),
            "ours": mine,
            "gap": g,
            "gap_season": g * n_gws / gw,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--pause", type=float, default=1.0)
    g = sub.add_parser("gap")
    g.add_argument("--managers", type=Path, required=True)
    g.add_argument("--ours", type=Path, required=True)
    g.add_argument("--entry", type=int, default=4570069)
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.out, args.seed, args.pause)
        return
    managers = pd.read_parquet(args.managers / "managers.parquet")
    gws = pd.read_parquet(args.ours / "gws.parquet")
    ours = gws[gws["start_id"].str.match(r"^template@\d+$")].sort_values("gw")
    assert ours["start_id"].nunique() == 1, ours["start_id"].unique()
    result = value_gap(managers, ours[["gw", "market_value"]])
    history = get_cached(
        args.managers / "history" / f"{args.entry}.json.gz", f"entry/{args.entry}/history/", 1.0
    )
    result["user"] = {
        str(h["event"]): {"value": h["value"], "bank": h["bank"]} for h in history["current"]
    }
    result["paths"] = managers.groupby(["group", "gw"])["value"].median().unstack(0).to_dict()
    result["ours_path"] = dict(
        zip(ours["gw"].astype(int), ours["market_value"].astype(int), strict=True)
    )
    (args.managers / "gap.json").write_text(json.dumps(result, indent=2, default=int))
    print(json.dumps({k: result[k] for k in GROUPS}, indent=2))


if __name__ == "__main__":
    main()
