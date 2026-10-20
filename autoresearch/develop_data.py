"""A develop-only copy of `data/` for the research loop (`autoresearch/program.md`).

Every table with a `season` column keeps seasons 2016/17-2022/23 only; the static tables
(`player_dim`, `team_dim`, `understat_map`) are copied whole (identity rows, PLAN §4). With
this as the research checkout's `data/`, no validate (2023/24-2024/25), holdout (2025/26) or
live (2026/27) row exists where experiments run.

    uv run python autoresearch/develop_data.py <full data dir> <out dir>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from fplopt.build.tables import TABLES

LAST_DEVELOP_SEASON = 2022


def main() -> None:
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for path in sorted(source.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if "season" in frame.columns:
            frame = frame[frame["season"] <= LAST_DEVELOP_SEASON].reset_index(drop=True)
            kept = sorted(frame["season"].unique().tolist())
            print(f"{path.name}: {len(frame)} rows, seasons {kept[:1]}..{kept[-1:]}")
        elif path.stem in TABLES and TABLES[path.stem].kind == "static":
            print(f"{path.name}: static, {len(frame)} rows")
        else:  # a dated table without seasons would carry every season across
            sys.exit(f"{path.name} has no season column and is not a static table")
        frame.to_parquet(tmp / path.name, index=False)
    (tmp / "DEVELOP_ONLY").write_text(
        f"seasons <= {LAST_DEVELOP_SEASON} only (autoresearch/develop_data.py)\n", encoding="utf-8"
    )
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


if __name__ == "__main__":
    main()
