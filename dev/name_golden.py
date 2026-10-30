"""Writes `tests/fixtures/rust_golden/names.json.gz`: every name the Understat and ESPN maps
normalise, and the (source name, FPL name) pairs they score with rapidfuzz's
`token_set_ratio` on real data, with Python's results (Rust migration batch 2: the Rust
port of the matcher is tested against it, `crates/fplopt-build/src/fuzz.rs`). Dev only.

Pairs: every one scoring >= 50 (the only scores that can change a map: every accept rule
needs name >= 50, and ties are ranked among accepted pairs only) plus a fixed sample of
`LOW_SAMPLE` below (all 1.8M pairs would be 15 MB).

    uv run python dev/name_golden.py --data data

`tests/test_rust_golden.py` re-checks the file against the installed rapidfuzz (no data
needed), so a rapidfuzz upgrade that changes a score fails there.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fplopt.build import espn, understat  # noqa: E402

GOLDEN = REPO / "tests" / "fixtures" / "rust_golden" / "names.json.gz"
MIN_KEPT_SCORE = 50.0
LOW_SAMPLE = 20_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="a full build's data/")
    args = parser.parse_args()
    player_match = pd.read_parquet(args.data / "player_match.parquet")
    player_season = pd.read_parquet(args.data / "player_season.parquet")
    fpl_apps = understat.fpl_appearances(player_match)
    fpl_names = understat.fpl_name_variants(player_season)

    sources = []
    upm = pd.read_parquet(args.data / "understat_player_match.parquet")
    us_names = dict(zip(upm["understat_id"], upm["understat_name"], strict=True))
    sources.append((understat.candidate_pairs(upm, fpl_apps), us_names, "understat_id"))
    tables = {
        name: pd.read_parquet(args.data / f"{name}.parquet")
        for name in ("espn_match", "espn_player_match")
    }
    played = espn.pl_appearances(tables).dropna(subset=["minutes_src"])
    played = played[played["minutes_src"] > 0].astype({"minutes_src": "int64"})
    if len(played):
        pairs = understat.candidate_pairs(
            played, fpl_apps, id_col=espn.ID, minutes_col="minutes_src"
        )
        names = dict(zip(played[espn.ID], played["athlete_name"], strict=True))
        sources.append((pairs, names, espn.ID))

    raw_names: set[str] = {n for names in fpl_names.values() for n in names}
    scored: set[tuple[str, str]] = set()
    for pairs, names, id_col in sources:
        raw_names.update(str(n) for n in names.values())
        norm = {k: understat.normalise_name(v) for k, v in names.items()}
        fpl_norm = {k: {understat.normalise_name(n) for n in v} for k, v in fpl_names.items()}
        for uid, key in zip(pairs[id_col], pairs["player_key"], strict=True):
            a = norm.get(uid, "")
            scored.update((a, b) for b in fpl_norm.get(int(key), ()))
    golden = {
        "names": {n: understat.normalise_name(n) for n in sorted(raw_names)},
        "pairs": [[a, b, fuzz.token_set_ratio(a, b)] for a, b in sorted(scored)],
    }
    high = [p for p in golden["pairs"] if p[2] >= MIN_KEPT_SCORE]
    low = [p for p in golden["pairs"] if p[2] < MIN_KEPT_SCORE]
    sample = random.Random(0).sample(low, min(LOW_SAMPLE, len(low)))
    golden["pairs"] = sorted(high + sample)
    GOLDEN.write_bytes(gzip.compress(json.dumps(golden).encode(), mtime=0))
    print(f"{GOLDEN}: {len(golden['names'])} names, {len(golden['pairs'])} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
