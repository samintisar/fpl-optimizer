# Chip diagnostics (chip research, sub-project 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure where chip value comes from under today's v1: three develop backtests, a tested analysis module, and a report that ranks sub-project 2's ideas.

**Architecture:** Spec `docs/superpowers/specs/2026-10-29-chip-diagnostics-design.md`.
- Three `backtest run`s (S0 `chipgains`, H, P) on develop write `gws.parquet`.
- `fplopt.backtest.chip_diagnostics` holds pure functions over those frames and the schedule (no file reads).
- `dev/chip_diagnostics.py` loads the runs and the tables, calls the functions, and writes CSVs and `comparisons.json` to `results/chips-diag/`.
- The report is written by hand from those tables.

**Tech Stack:** pandas, numpy; `fplopt.backtest.evaluate.full_run_window_test`, `fplopt.backtest.chip_values`, `fplopt.backtest.rules.backtest_rules`, `fplopt.features.history.training_frame`.

**Conventions:**
- Work in this worktree (branch `feat/chip-diagnostics`).
- The dev PC's data lives in the main checkout. `MAIN=/c/Users/samin/Documents/GitHub/fpl-optimizer`, and every `fplopt` command runs with `FPLOPT_RAW_DIR=$MAIN/raw FPLOPT_DATA_DIR=$MAIN/data`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run `uv run pytest`, check the summary line rather than a piped tail, then `uv run ruff check . && uv run ruff format .`.

**Facts the code relies on** (checked 2026-10-29):

*GW rows (`gws.parquet`):*
- Keys: `season`, `start_id` (e.g. `template@2`, `random0@1`, `random1@20`), `policy`, `gw`, `gw_index`.
- `chip`: `3xc`, `bboost`, `freehit`, `wildcard` or null.
- Points columns: `net_points`, `xg_net_points`, `captain` (player_key), `captain_points` (the raw points of the player who got the armband), `bench_points` (points of the squad players who did not count; 0 under Bench Boost).
- `bank` and `free_transfers`.
- The chip columns are filled in `chipgains` runs only:
  - `captain_xp`, `bench_xp`;
  - `gain_tc` (= captain xP);
  - `gain_bb` (bench xP net of bench weights);
  - `gain_fh`, `gain_wc` (solve deltas);
  - `blanks`.

*Realized gains of the matching chips in S0 rows (S0 never plays a chip):*
- Triple Captain: `captain_points`, the extra armband multiple.
- Bench Boost: `bench_points`.

*Tables:*
- `schedule`: `fixture_key`, `season`, `gw`, `gw_index`, `home_team_key`, `away_team_key`, …
- `player_gw`: `player_key`, `season`, `gw`, `team_key`, …

---

## File map

- Create: `src/fplopt/backtest/chip_diagnostics.py`: the functions in Tasks 2–6.
- Create: `tests/test_chip_diagnostics.py`.
- Create: `dev/chip_diagnostics.py`: the driver (Task 7).
- Create: `docs/research/2026-10-chip-diagnostics.md`: the report (Task 9).
- Modify: `results/experiments.csv`: the three runs' rows, appended by `backtest run`.

**Spec deviations, made because of the data:**
- **Question 5 covers Triple Captain only.** The GW rows don't record which players were on the bench, so the congestion check for Bench Boost picks is out.
- **The three comparisons are computed in the driver from the run frames** with `full_run_window_test`, as `results/chips-validate.py` did. They are not separate `backtest compare` runs, which would re-simulate every arm. They are written to `results/chips-diag/comparisons.json` and quoted in the report.
- **Single / double / blank** is the captain's own fixture count for Triple Captain. For the other chips it's per GW: whether any club has a double, or a blank.

---

### Task 1: data and the three runs (background)

**Files:** none in the repo; `$MAIN/raw` gains the server's newer files, `$MAIN/data` is rebuilt.

- [ ] **Step 1: copy the raw files the dev PC lacks** (Git Bash; raw/ is append-only, so missing paths are the whole difference; the server is only read)

```bash
MAIN=/c/Users/samin/Documents/GitHub/fpl-optimizer
ssh fplopt-server 'cd ~/fpl-optimizer && find raw -type f ! -name ".*" | LC_ALL=C sort' > /tmp/raw-server.txt
(cd $MAIN && find raw -type f | LC_ALL=C sort) > /tmp/raw-local.txt
LC_ALL=C comm -23 /tmp/raw-server.txt /tmp/raw-local.txt > /tmp/raw-missing.txt
wc -l < /tmp/raw-missing.txt
ssh fplopt-server 'cd ~/fpl-optimizer && nice -n 15 tar cf - -T -' < /tmp/raw-missing.txt | (cd $MAIN && tar xf -)
ls $MAIN/raw
```
Expected: several thousand files; `raw/espn` now listed.

- [ ] **Step 2: rebuild data/**

```bash
FPLOPT_RAW_DIR=$MAIN/raw FPLOPT_DATA_DIR=$MAIN/data uv run fplopt build all
ls $MAIN/data | grep -E "espn|club_match"
```
Expected: the build ends without an error and lists `club_match.parquet`, `espn_map.parquet`, `espn_match.parquet`, `espn_player_match.parquet` and `espn_schedule_snapshot.parquet`.

- [ ] **Step 3: start the three runs in the background** (about 1 h each at `--jobs 14`; they don't need any later task)

```bash
BASE=horizon=6,decay=0.75,max_hits=1,hit_margin=2
COMMON="--seasons 2016-2022 --starts template@1,random:3@1,random:3@20 --jobs 14 --family chip-diagnostics-2026-10"
export FPLOPT_RAW_DIR=$MAIN/raw FPLOPT_DATA_DIR=$MAIN/data
( uv run fplopt backtest run --spec chipgains:v1:$BASE $COMMON --out results/chips-diag-s0 > results/chips-diag-s0.log 2>&1 &&
  uv run fplopt backtest run --spec chiprules:v1:$BASE,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87 $COMMON --out results/chips-diag-h > results/chips-diag-h.log 2>&1 &&
  uv run fplopt backtest run --spec optimizer:v1:$BASE,chips=1,chip_shrink=1.0 $COMMON --out results/chips-diag-p > results/chips-diag-p.log 2>&1 )
```
Run it with the Bash tool's `run_in_background`. Expected: three `gws.parquet` files with 1,448 rows each, and three new rows in `results/experiments.csv`.

---

### Task 2: chip attribution and the paired gap

**Files:**
- Create: `src/fplopt/backtest/chip_diagnostics.py`
- Test: `tests/test_chip_diagnostics.py`

- [ ] **Step 1: write the failing tests**

```python
"""Tests for fplopt.backtest.chip_diagnostics (spec 2026-10-29-chip-diagnostics-design)."""

import numpy as np
import pandas as pd
import pytest

from fplopt.backtest import chip_diagnostics as cd


def gws(season=2018, start="template@1", n=10, chips=None, points=None):
    """GW rows of one run: gw_index 1..n, `chips` = {gw_index: chip}, `points` = {gw_index: net}."""
    chips, points = chips or {}, points or {}
    return pd.DataFrame(
        {
            "season": season,
            "start_id": start,
            "gw": range(1, n + 1),
            "gw_index": range(1, n + 1),
            "chip": [chips.get(i) for i in range(1, n + 1)],
            "net_points": [points.get(i, 50) for i in range(1, n + 1)],
            "xg_net_points": [float(points.get(i, 50)) for i in range(1, n + 1)],
        }
    )


def test_a_wildcard_is_attributed_its_gw_and_the_next_five():
    run = gws(chips={2: "wildcard", 4: "3xc"})
    got = list(cd.attribute_chips(run))
    assert got == [
        "none", "wildcard", "wildcard", "3xc", "wildcard", "wildcard", "wildcard",
        "none", "none", "none",
    ]


def test_attribution_is_per_run():
    a = gws(start="template@1", n=3, chips={3: "wildcard"})
    b = gws(start="random0@1", n=3)
    got = cd.attribute_chips(pd.concat([a, b], ignore_index=True))
    assert list(got) == ["none", "none", "wildcard", "none", "none", "none"]


def test_gap_by_chip_contributions_sum_to_the_mean_gap():
    a = gws(chips={3: "3xc"}, points={3: 70, 5: 60})
    b = gws(points={5: 40})
    table = cd.gap_by_chip(a, b)
    assert table["contribution_per_gw"].sum() == pytest.approx((20 + 20) / 10)
    row = table.set_index(["chip_a", "chip_b"]).loc[("3xc", "none")]
    assert row["n"] == 1 and row["mean_diff"] == 20


def test_gap_by_cell_uses_8_gw_cells():
    a = gws(n=16, points={1: 58, 9: 42})
    b = gws(n=16)
    table = cd.gap_by_cell(a, b).set_index("cell")
    assert table.loc[1, "mean_diff"] == pytest.approx(8 / 8)
    assert table.loc[2, "mean_diff"] == pytest.approx(-8 / 8)


def test_paired_refuses_runs_that_dont_match():
    with pytest.raises(ValueError, match="rows"):
        cd.paired(gws(n=10), gws(n=9))
```

- [ ] **Step 2: run them and check they fail**

Run: `uv run pytest tests/test_chip_diagnostics.py -q`
Expected: the collection fails with `ImportError` (no module `chip_diagnostics`).

- [ ] **Step 3: implement**

```python
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
        .agg(n=("diff", "size"), mean_diff=("diff", "mean"), mean_diff_xg=("diff_xg", "mean"),
             sum_diff=("diff", "sum"))
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
```

- [ ] **Step 4: run the tests and check they pass**

Run: `uv run pytest tests/test_chip_diagnostics.py -q`
Expected: `5 passed`.

- [ ] **Step 5: commit**

```bash
git add src/fplopt/backtest/chip_diagnostics.py tests/test_chip_diagnostics.py
git commit -m "feat(backtest): chip diagnostics, the paired gap by attributed chip and by cell"
```

---

### Task 3: GW kinds (doubles, blanks) and gain distributions

**Files:**
- Modify: `src/fplopt/backtest/chip_diagnostics.py`
- Test: `tests/test_chip_diagnostics.py`

- [ ] **Step 1: write the failing tests** (append)

```python
def schedule_rows(rows):
    """rows = [(season, gw, home, away)]."""
    return pd.DataFrame(
        [
            {"fixture_key": k, "season": s, "gw": g, "home_team_key": h, "away_team_key": a}
            for k, (s, g, h, a) in enumerate(rows)
        ]
    )


def test_gw_kind_counts_the_captains_fixtures_and_the_gws_doubles_and_blanks():
    # clubs 1-4; GW1 normal, GW2 club 1 twice, clubs 3-4 blank
    schedule = schedule_rows(
        [(2018, 1, 1, 2), (2018, 1, 3, 4), (2018, 2, 1, 2), (2018, 2, 1, 2)]
    )
    player_gw = pd.DataFrame(
        {"player_key": [7, 7, 8], "season": 2018, "gw": [1, 2, 2], "team_key": [1, 1, 3]}
    )
    rows = pd.DataFrame(
        {"season": 2018, "start_id": "t", "gw": [1, 2, 2], "gw_index": [1, 2, 2],
         "captain": [7, 7, 8]}
    )
    got = cd.gw_kind(schedule, player_gw, rows)
    assert list(got["captain_fixtures"]) == [1, 2, 0]
    assert list(got["captain_kind"]) == ["single", "double", "blank"]
    assert list(got["n_double_clubs"]) == [0, 1, 1]
    assert list(got["n_blank_clubs"]) == [0, 2, 2]
    assert list(got["gw_type"]) == ["normal", "double+blank", "double+blank"]


def test_gain_distribution_quantiles_per_chip_and_group():
    rows = pd.DataFrame(
        {"start_id": ["template@1"] * 4 + ["random0@1"] * 4,
         "gain_tc": [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0],
         "gain_bb": np.nan, "gain_fh": np.nan, "gain_wc": np.nan}
    )
    rows["start_type"] = cd.start_type(rows["start_id"])
    table = cd.gain_distribution(rows, ["start_type"])
    tc = table[table["chip"] == "3xc"].set_index("start_type")
    assert tc.loc["template", "median"] == 2.5 and tc.loc["random", "p90"] == 10
    assert tc.loc["template", "n"] == 4
    assert set(table["chip"]) == {"3xc"}  # chips without gains are left out
```

- [ ] **Step 2: run them and check they fail**

Run: `uv run pytest tests/test_chip_diagnostics.py -q -k "gw_kind or distribution"`
Expected: FAIL, `AttributeError: module ... has no attribute 'gw_kind'`.

- [ ] **Step 3: implement** (append to the module, and add `"gain_distribution", "gw_kind", "start_type"` to `__all__`)

```python
from fplopt.backtest.chip_values import GAIN_COLUMNS


def start_type(start_id: pd.Series) -> pd.Series:
    """"template" or "random" from a start id (template@1, random0@20, ...)."""
    return start_id.str.extract(r"^(template|random)", expand=False)


def _kind(n: int) -> str:
    return "blank" if n == 0 else "single" if n == 1 else "double"


def gw_kind(schedule: pd.DataFrame, player_gw: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per row of `rows` (season, gw, captain = player_key), in row order:
    `captain_fixtures` (his club's league fixtures that GW, his club from `player_gw`),
    `captain_kind` (blank / single / double), `n_double_clubs` and `n_blank_clubs` (the
    season's clubs with ≥ 2 / 0 fixtures that GW; a season's clubs are the schedule's) and
    `gw_type` (normal, double, blank or double+blank)."""
    sides = pd.concat(
        [
            schedule[["season", "gw", "home_team_key"]].rename(columns={"home_team_key": "team_key"}),
            schedule[["season", "gw", "away_team_key"]].rename(columns={"away_team_key": "team_key"}),
        ]
    )
    sides = sides[sides["team_key"] > 0]
    counts = sides.groupby(["season", "gw", "team_key"]).size().rename("n").reset_index()
    clubs = sides.drop_duplicates(["season", "team_key"])[["season", "team_key"]]
    gws_ = schedule.drop_duplicates(["season", "gw"])[["season", "gw"]]
    grid = gws_.merge(clubs, on="season").merge(counts, on=["season", "gw", "team_key"], how="left")
    grid["n"] = grid["n"].fillna(0).astype("int64")
    per_gw = grid.groupby(["season", "gw"]).agg(
        n_double_clubs=("n", lambda s: int((s >= 2).sum())),
        n_blank_clubs=("n", lambda s: int((s == 0).sum())),
    )
    club = player_gw[["player_key", "season", "gw", "team_key"]].rename(columns={"player_key": "captain"})
    out = rows[["season", "gw", "captain"]].reset_index(drop=True)
    out = out.merge(club, on=["season", "gw", "captain"], how="left")
    out = out.merge(counts, on=["season", "gw", "team_key"], how="left")
    out["captain_fixtures"] = out["n"].fillna(0).astype("int64")
    out["captain_kind"] = out["captain_fixtures"].map(_kind)
    out = out.join(per_gw, on=["season", "gw"])
    out[["n_double_clubs", "n_blank_clubs"]] = out[["n_double_clubs", "n_blank_clubs"]].fillna(0).astype("int64")
    double, blank = out["n_double_clubs"] > 0, out["n_blank_clubs"] > 0
    out["gw_type"] = np.select(
        [double & blank, double, blank], ["double+blank", "double", "blank"], "normal"
    )
    out.index = rows.index
    return out[["captain_fixtures", "captain_kind", "n_double_clubs", "n_blank_clubs", "gw_type"]]


def gain_distribution(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Each chip's ex-ante gain (`GAIN_COLUMNS`) per group of `by`: n, mean, median, p75,
    p90. Rows without a gain (chip unavailable) are left out, chips without any gain too."""
    parts = []
    for chip, column in GAIN_COLUMNS.items():
        part = rows[by + [column]].rename(columns={column: "gain"}).dropna(subset=["gain"])
        if part.empty:
            continue
        g = part.groupby(by)["gain"]
        table = pd.DataFrame(
            {"n": g.size(), "mean": g.mean(), "median": g.median(),
             "p75": g.quantile(0.75), "p90": g.quantile(0.9)}
        ).reset_index()
        parts.append(table.assign(chip=chip))
    return pd.concat(parts, ignore_index=True)[["chip", *by, "n", "mean", "median", "p75", "p90"]]
```
Put the `GAIN_COLUMNS` import with the module's other imports at the top.

- [ ] **Step 4: run the tests and check they pass**

Run: `uv run pytest tests/test_chip_diagnostics.py -q`
Expected: `7 passed`.

- [ ] **Step 5: commit**

```bash
git add src/fplopt/backtest/chip_diagnostics.py tests/test_chip_diagnostics.py
git commit -m "feat(backtest): chip diagnostics, GW kinds and gain distributions"
```

---

### Task 4: forecast vs outcome

**Files:** Modify `src/fplopt/backtest/chip_diagnostics.py`, `tests/test_chip_diagnostics.py`

- [ ] **Step 1: write the failing tests** (append)

```python
def test_forecast_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    predicted = pd.Series(rng.uniform(2, 12, 2000))
    realized = 1.0 + 0.5 * predicted + rng.normal(0, 0.5, 2000)
    got = cd.forecast_slope(predicted, realized)
    assert got["slope"] == pytest.approx(0.5, abs=0.02)
    assert got["intercept"] == pytest.approx(1.0, abs=0.15)
    assert got["se"] < 0.01 and got["n"] == 2000
    assert got["bias"] == pytest.approx(float((realized - predicted).mean()))


def test_forecast_slope_skips_missing_pairs_and_needs_three():
    got = cd.forecast_slope(pd.Series([1.0, 2.0, np.nan, 4.0]), pd.Series([1.0, 2.0, 3.0, np.nan]))
    assert got["n"] == 2 and np.isnan(got["slope"])
```

- [ ] **Step 2: run them and check they fail**

Run: `uv run pytest tests/test_chip_diagnostics.py -q -k forecast`
Expected: FAIL, `AttributeError ... 'forecast_slope'`.

- [ ] **Step 3: implement** (append; add `"forecast_slope"` to `__all__`)

```python
def forecast_slope(predicted: pd.Series, realized: pd.Series) -> dict[str, float]:
    """OLS of realized on predicted over the pairs with both: slope, its SE, intercept, bias
    (mean realized − predicted) and n. Slope < 1: the larger the forecast gain, the more it
    overstates (the optimizer's curse). NaN statistics under 3 pairs."""
    frame = pd.DataFrame({"x": predicted, "y": realized}).astype("float64").dropna()
    n = len(frame)
    nan = float("nan")
    if n < 3:
        return {"slope": nan, "se": nan, "intercept": nan, "bias": nan, "n": n}
    x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
    xc = x - x.mean()
    sxx = float(xc @ xc)
    slope = float(xc @ (y - y.mean()) / sxx) if sxx > 0 else nan
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    se = float(np.sqrt(resid @ resid / (n - 2) / sxx)) if sxx > 0 else nan
    return {"slope": slope, "se": se, "intercept": intercept, "bias": float((y - x).mean()), "n": n}
```

- [ ] **Step 4: run the tests and check they pass**

Run: `uv run pytest tests/test_chip_diagnostics.py -q`
Expected: `9 passed`.

- [ ] **Step 5: commit**

```bash
git add src/fplopt/backtest/chip_diagnostics.py tests/test_chip_diagnostics.py
git commit -m "feat(backtest): chip diagnostics, forecast-vs-outcome slope"
```

---

### Task 5: the value of holding a chip

**Files:** Modify `src/fplopt/backtest/chip_diagnostics.py`, `tests/test_chip_diagnostics.py`

`holding_value` uses the same windows and best-of-`L` rule as `chip_values.estimate_terminal_values`: the 2026/27 windows on `gw_index`, from `backtest_rules(max(DEVELOP_SEASONS))`, and for `L` the best gain among GWs `stop − L + 1 .. stop`. Its `ex_ante` column, pooled over a chip's windows weighted by `n`, must equal that function's output (before its clip at 0), which the test pins. `realized` is the realized gain at the ex-ante best GW, for chips that have a realized column.

- [ ] **Step 1: write the failing tests** (append)

```python
from fplopt.backtest.chip_values import estimate_terminal_values
from fplopt.backtest.rules import backtest_rules
from fplopt.seasons import DEVELOP_SEASONS


def season_rows(seed, start="template@1", season=2018):
    rng = np.random.default_rng(seed)
    n = 38
    return pd.DataFrame(
        {
            "season": season, "start_id": start, "policy": "s0",
            "gw_index": np.arange(1, n + 1),
            "gain_tc": rng.uniform(3, 12, n), "gain_bb": rng.uniform(0, 8, n),
            "gain_fh": rng.uniform(-2, 10, n), "gain_wc": rng.uniform(-5, 20, n),
            "captain_points": rng.integers(0, 20, n), "bench_points": rng.integers(0, 15, n),
        }
    )


def test_holding_value_ex_ante_matches_the_terminal_value_estimate():
    # estimate_terminal_values pools a chip's windows per L: compare the n-weighted mean
    rows = pd.concat([season_rows(1), season_rows(2, start="random0@1")], ignore_index=True)
    table = {c: (1.0,) * 19 for c in ("3xc", "bboost", "freehit", "wildcard")}
    got = cd.holding_value(rows, {"3xc": "captain_points", "bboost": "bench_points"}, table)
    pooled = (
        got.assign(w=got["ex_ante"] * got["n"])
        .groupby(["chip", "left"])[["w", "n"]].sum()
        .pipe(lambda t: t["w"] / t["n"])
    )
    for chip, values in estimate_terminal_values(rows).items():
        mine = pooled.loc[chip].sort_index().to_numpy()[: len(values)]
        np.testing.assert_allclose(np.clip(mine, 0, None), values)


def test_holding_value_realized_is_taken_at_the_ex_ante_best_gw():
    rows = season_rows(3)
    rules = backtest_rules(max(DEVELOP_SEASONS))
    stop = max(w.stop for w in rules.chips if w.name == "3xc" and w.start == 1)
    # L = 1: the window's last GW only
    last = rows.loc[rows["gw_index"] == stop].iloc[0]
    # L = 2: the better of the last two by gain_tc
    two = rows[rows["gw_index"].isin([stop - 1, stop])].sort_values("gain_tc").iloc[-1]
    got = cd.holding_value(rows, {"3xc": "captain_points"}, {"3xc": (2.0,) * 19})
    tc = got[(got["chip"] == "3xc") & (got["window_start"] == 1)].set_index("left")
    assert tc.loc[1, "realized"] == last["captain_points"]
    assert tc.loc[2, "realized"] == two["captain_points"]
    assert tc.loc[1, "table"] == 2.0
    assert tc.loc[1, "realized_ratio"] == pytest.approx(last["captain_points"] / 2.0)
    fh = got[got["chip"] == "freehit"]
    assert fh["realized"].isna().all()  # no realized column given
```

- [ ] **Step 2: run them and check they fail**

Run: `uv run pytest tests/test_chip_diagnostics.py -q -k holding`
Expected: FAIL, `AttributeError ... 'holding_value'`.

- [ ] **Step 3: implement** (append; add `"holding_value"` to `__all__`; imports at the top: `import math`, `from fplopt.seasons import DEVELOP_SEASONS`)

```python
def holding_value(
    rows: pd.DataFrame, realized: dict[str, str], table: dict[str, tuple[float, ...]]
) -> pd.DataFrame:
    """Per chip, window and GWs left `L`: `ex_ante` (the mean over runs of the best ex-ante
    gain among the window's last L GWs: `chip_values.estimate_terminal_values`, unclipped
    and per window), `realized` (the mean realized gain at that ex-ante best GW, from the
    column `realized[chip]`; NaN for chips without one), `table` (`table[chip][L − 1]`, the
    value P plans with), `exante_ratio` and `realized_ratio` (each over `table`), and n."""
    from fplopt.backtest.rules import backtest_rules

    rules = backtest_rules(max(DEVELOP_SEASONS))
    out = []
    for chip, column in GAIN_COLUMNS.items():
        real_col = realized.get(chip)
        for window in [w for w in rules.chips if w.name == chip]:
            span = window.stop - window.start + 1
            sums = np.zeros(span)
            real_sums = np.zeros(span)
            counts = np.zeros(span, dtype="int64")
            for _, run in rows.groupby(["season", "policy", "start_id"], sort=True):
                by_gw = run.set_index("gw_index")
                best, best_real = -math.inf, math.nan
                for left in range(1, span + 1):
                    gw = window.stop - left + 1
                    if gw in by_gw.index and not pd.isna(by_gw.at[gw, column]):
                        gain = float(by_gw.at[gw, column])
                        if gain > best:
                            best = gain
                            best_real = float(by_gw.at[gw, real_col]) if real_col else math.nan
                    if best > -math.inf:
                        sums[left - 1] += best
                        real_sums[left - 1] += best_real
                        counts[left - 1] += 1
            values = table.get(chip, ())
            for left in range(1, span + 1):
                n = int(counts[left - 1])
                ex_ante = sums[left - 1] / n if n else math.nan
                real = real_sums[left - 1] / n if n and real_col else math.nan
                tab = float(values[left - 1]) if left <= len(values) else math.nan
                out.append(
                    {
                        "chip": chip, "window_start": window.start, "left": left, "n": n,
                        "ex_ante": ex_ante, "realized": real, "table": tab,
                        "exante_ratio": ex_ante / tab if tab else math.nan,
                        "realized_ratio": real / tab if tab else math.nan,
                    }
                )
    return pd.DataFrame(out)
```


- [ ] **Step 4: run the tests and check they pass**

Run: `uv run pytest tests/test_chip_diagnostics.py -q`
Expected: `11 passed`.

- [ ] **Step 5: commit**

```bash
git add src/fplopt/backtest/chip_diagnostics.py tests/test_chip_diagnostics.py
git commit -m "feat(backtest): chip diagnostics, the value of holding a chip (ex-ante and realized)"
```

---

### Task 6: congestion split

**Files:** Modify `src/fplopt/backtest/chip_diagnostics.py`, `tests/test_chip_diagnostics.py`

- [ ] **Step 1: write the failing tests** (append)

```python
def test_captain_congestion_flags_a_european_week_or_short_rest():
    features = pd.DataFrame(
        {"player_key": [7, 7, 7, 8], "season": 2018, "gw": [1, 2, 2, 2],
         "europe_week": [0.0, 0.0, 1.0, 0.0], "rest_days_all": [7.0, 6.0, 3.0, 2.5]}
    )
    rows = pd.DataFrame({"season": 2018, "gw": [1, 2, 2, 3], "captain": [7, 7, 8, 7]})
    got = cd.captain_congestion(rows, features)
    assert list(got) == [False, True, True, False]  # GW3: no feature row -> not flagged


def test_ratio_split_and_weighted_share():
    rows = pd.DataFrame(
        {"flag": [True, True, False, False], "xp": [5.0, 5.0, 5.0, 5.0],
         "real": [2.0, 4.0, 6.0, 6.0]}
    )
    got = cd.ratio_split(rows, "flag", "xp", "real").set_index("flag")
    assert got.loc[True, "ratio"] == pytest.approx(0.6) and got.loc[False, "ratio"] == 1.2
    assert got.loc[True, "n"] == 2
    assert cd.weighted_share(pd.Series([True, False, False]), pd.Series([2.0, 1.0, 1.0])) == 0.5
```

- [ ] **Step 2: run them and check they fail**

Run: `uv run pytest tests/test_chip_diagnostics.py -q -k "congestion or ratio"`
Expected: FAIL, `AttributeError`.

- [ ] **Step 3: implement** (append; add `"CONGESTED_REST", "captain_congestion", "ratio_split", "weighted_share"` to `__all__`)

```python
CONGESTED_REST = 3.0  # rest_days_all at or below: a short-rest fixture


def captain_congestion(rows: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Per row (season, gw, captain = player_key): True if any of his fixtures that GW has
    `europe_week` = 1 or `rest_days_all` ≤ `CONGESTED_REST`. `features`: player_key, season,
    gw, europe_week, rest_days_all (one row per player fixture, e.g. `training_frame`). Rows
    without a feature row are False."""
    flag = (features["europe_week"] == 1) | (features["rest_days_all"] <= CONGESTED_REST)
    per = (
        features.assign(flag=flag)
        .groupby(["player_key", "season", "gw"])["flag"].any()
        .rename_axis(["captain", "season", "gw"])
    )
    keys = pd.MultiIndex.from_frame(rows[["captain", "season", "gw"]])
    return pd.Series(per.reindex(keys).fillna(False).to_numpy(dtype=bool), index=rows.index)


def ratio_split(rows: pd.DataFrame, flag: str, xp: str, real: str) -> pd.DataFrame:
    """Per value of `flag`: n, the summed `xp` and `real`, and ratio = Σ real / Σ xp."""
    g = rows.groupby(flag)
    table = pd.DataFrame({"n": g.size(), "xp": g[xp].sum(), "realized": g[real].sum()})
    table["ratio"] = table["realized"] / table["xp"]
    return table.reset_index()


def weighted_share(flag: pd.Series, weight: pd.Series) -> float:
    """Σ weight over flagged rows / Σ weight: the share a chip-gain-weighted pick would land
    in flagged fixtures."""
    w = weight.astype("float64").clip(lower=0)
    return float(w[flag.astype(bool)].sum() / w.sum())
```

- [ ] **Step 4: run the tests and check they pass**

Run: `uv run pytest tests/test_chip_diagnostics.py -q`
Expected: `13 passed`.

- [ ] **Step 5: commit**

```bash
git add src/fplopt/backtest/chip_diagnostics.py tests/test_chip_diagnostics.py
git commit -m "feat(backtest): chip diagnostics, congestion split for captains"
```

---

### Task 7: the driver script

**Files:** Create `dev/chip_diagnostics.py`

- [ ] **Step 1: write it**

```python
"""Chip diagnostics driver (spec docs/superpowers/specs/2026-10-29-chip-diagnostics-design.md):
reads the S0 / H / P develop runs, writes the tables to results/chips-diag/.

    FPLOPT_DATA_DIR=<data> uv run python dev/chip_diagnostics.py \
        --s0 results/chips-diag-s0 --h results/chips-diag-h --p results/chips-diag-p \
        [--out results/chips-diag] [--upcoming 2026:10-19]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fplopt.backtest import chip_diagnostics as cd
from fplopt.backtest.chip_values import load_terminal_values
from fplopt.backtest.evaluate import full_run_window_test
from fplopt.features.history import training_frame
from fplopt.features.store import DataStore
from fplopt.settings import Settings

ARMS = ("S0", "H", "P")
COMPARISONS = (("P", "H"), ("P", "S0"), ("H", "S0"))


def load(run: Path) -> pd.DataFrame:
    return pd.read_parquet(run / "gws.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=Path, required=True)
    ap.add_argument("--h", type=Path, required=True)
    ap.add_argument("--p", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/chips-diag"))
    ap.add_argument("--upcoming", default=None, help="SEASON:FROM-TO, e.g. 2026:10-19")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Settings.from_env().data_dir
    runs = {"S0": args.s0, "H": args.h, "P": args.p}
    gws = {arm: load(path) for arm, path in runs.items()}
    (out / "inputs.json").write_text(
        json.dumps({arm: str(p.resolve()) for arm, p in runs.items()}, indent=2), encoding="utf-8"
    )

    # Q1: comparisons, gap by chip and by cell, chip timing
    comparisons = {}
    for a, b in COMPARISONS:
        m = cd.paired(gws[a], gws[b])
        real, xg = full_run_window_test(m, "diff"), full_run_window_test(m, "diff_xg")
        comparisons[f"{a} vs {b}"] = {
            "mean_per_gw": real.mean, "p_one_sided": real.p_one_sided, "n_cells": real.n_cells,
            "xg_mean_per_gw": xg.mean,
            "season_totals": m.groupby(["season", "start_id"])["diff"].sum()
            .groupby("season").mean().to_dict(),
        }
        cd.gap_by_chip(gws[a], gws[b]).to_csv(out / f"q1-gap-chip-{a}-{b}.csv", index=False)
        cd.gap_by_cell(gws[a], gws[b]).to_csv(out / f"q1-gap-cell-{a}-{b}.csv", index=False)
        print(f"{a} vs {b}: {real.mean:+.3f}/GW, p {real.p_one_sided:.4f}, xG {xg.mean:+.3f}/GW")
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2, default=str), encoding="utf-8")
    timing = pd.concat(
        [g.dropna(subset=["chip"]).assign(arm=arm)[["arm", "chip", "gw_index"]] for arm, g in gws.items()]
    )
    timing.groupby(["arm", "chip", "gw_index"]).size().rename("n").reset_index().to_csv(
        out / "q1-timing.csv", index=False
    )

    # Q2: gain distributions (S0)
    s0 = gws["S0"].copy()
    schedule = pd.read_parquet(data_dir / "schedule.parquet")
    player_gw = pd.read_parquet(data_dir / "player_gw.parquet")
    s0 = s0.join(cd.gw_kind(schedule, player_gw, s0))
    s0["start_type"] = cd.start_type(s0["start_id"])
    s0["ft"] = s0["free_transfers"].clip(upper=3)
    s0["bank_band"] = pd.cut(s0["bank"], [-1, 5, 20, 50, 10_000], labels=["<=0.5", "0.6-2.0", "2.1-5.0", ">5.0"])
    for by in (["gw_index"], ["captain_kind"], ["gw_type"], ["start_type"], ["ft"], ["bank_band"]):
        cd.gain_distribution(s0, by).to_csv(out / f"q2-gain-{by[0]}.csv", index=False)

    # Q3: forecast vs outcome
    slopes = {
        "3xc (S0, all rows)": cd.forecast_slope(s0["gain_tc"], s0["captain_points"]),
        "bboost (S0, all rows)": cd.forecast_slope(s0["gain_bb"], s0["bench_points"]),
    }
    for chip, gain, real in (("3xc", "gain_tc", "captain_points"), ("bboost", "gain_bb", "bench_points")):
        top = s0[s0[gain] >= s0[gain].quantile(0.9)]
        slopes[f"{chip} (S0, top decile)"] = cd.forecast_slope(top[gain], top[real])
    for arm in ("H", "P"):
        m = cd.paired(gws[arm], gws["S0"]).merge(
            s0[cd.KEYS + ["gain_fh", "gain_wc"]], on=cd.KEYS
        )
        played = gws[arm].set_index(cd.KEYS)["chip"]
        m["played"] = played.reindex(pd.MultiIndex.from_frame(m[cd.KEYS])).to_numpy()
        fh = m[m["played"] == "freehit"]
        slopes[f"freehit ({arm} vs S0, that GW)"] = cd.forecast_slope(fh["gain_fh"], fh["diff"])
        m = m.sort_values(cd.KEYS)
        m["diff_6"] = (
            m.groupby(["season", "start_id"])["diff"]
            .transform(lambda s: s[::-1].rolling(cd.WC_SPAN, min_periods=1).sum()[::-1])
        )
        wc = m[m["played"] == "wildcard"]
        slopes[f"wildcard ({arm} vs S0, 6 GWs)"] = cd.forecast_slope(wc["gain_wc"], wc["diff_6"])
    pd.DataFrame(slopes).T.to_csv(out / "q3-forecast-slopes.csv")

    # Q4: holding value
    table = load_terminal_values(1.0)
    hv = cd.holding_value(s0, {"3xc": "captain_points", "bboost": "bench_points"}, table)
    hv.to_csv(out / "q4-holding-value.csv", index=False)

    # Q5: captain congestion (training_frame as of the first deadline after develop)
    gameweek = pd.read_parquet(data_dir / "gameweek.parquet")
    after = gameweek.loc[(gameweek["season"] == 2023) & (gameweek["gw"] == 1), "deadline_time"].iloc[0]
    features = training_frame(DataStore(data_dir).as_of(after))
    features = features[["player_key", "season", "gw", "europe_week", "rest_days_all"]]
    s0["congested"] = cd.captain_congestion(s0, features)
    cd.ratio_split(s0, "congested", "captain_xp", "captain_points").to_csv(
        out / "q5-captain-congestion-s0.csv", index=False
    )
    shares = {"S0 gain-weighted": cd.weighted_share(s0["congested"], s0["gain_tc"])}
    for arm in ("H", "P"):
        tc = gws[arm][gws[arm]["chip"] == "3xc"]
        shares[f"{arm} TC picks"] = float(cd.captain_congestion(tc, features).mean())
    (out / "q5-tc-congestion-share.json").write_text(json.dumps(shares, indent=2), encoding="utf-8")

    # Q6: doubles and blanks ahead (as-of schedule now; no scoring)
    if args.upcoming:
        season, span = args.upcoming.split(":")
        lo, hi = (int(x) for x in span.split("-"))
        store = DataStore(data_dir)
        future = gameweek[(gameweek["season"] == int(season)) & (gameweek["deadline_time"] > pd.Timestamp.now(tz="UTC"))]
        view = store.as_of(future["deadline_time"].min())
        sched = view.table("schedule")
        sched = sched[(sched["season"] == int(season)) & sched["gw"].between(lo, hi)]
        rows = sched[["season", "gw"]].drop_duplicates().assign(captain=-1)
        kinds = cd.gw_kind(sched, player_gw.iloc[:0], rows)
        rows.join(kinds)[["season", "gw", "n_double_clubs", "n_blank_clubs", "gw_type"]].to_csv(
            out / "q6-upcoming.csv", index=False
        )
    print(f"tables in {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: check the APIs it calls exist with these signatures**

Run: `uv run python -c "from fplopt.settings import Settings; from fplopt.features.store import DataStore; import inspect; print(inspect.signature(Settings.from_env)); print(inspect.signature(DataStore))"`
Expected: `Settings.from_env` takes no required arguments, or only an optional env, and `DataStore` takes a data directory as its first argument. If a name differs, for example the settings constructor, use the real one (`grep -n "def \|class " src/fplopt/settings.py`) and keep the rest.

- [ ] **Step 3: lint**

Run: `uv run ruff check dev/chip_diagnostics.py && uv run ruff format dev/chip_diagnostics.py`
Expected: clean.

- [ ] **Step 4: commit**

```bash
git add dev/chip_diagnostics.py
git commit -m "dev: chip diagnostics driver (tables for the report)"
```

---

### Task 8: run the analysis

- [ ] **Step 1: check the runs finished** (from Task 1 Step 3)

Run: `uv run python -c "import pandas as pd; [print(r, len(pd.read_parquet(f'results/{r}/gws.parquet'))) for r in ('chips-diag-s0','chips-diag-h','chips-diag-p')]"`
Expected: 1,448 rows each, and every solve Optimal (`grep -c "solver" results/chips-diag-*.log` shows no warnings).

- [ ] **Step 2: run the driver**

```bash
FPLOPT_DATA_DIR=$MAIN/data uv run python dev/chip_diagnostics.py --s0 results/chips-diag-s0 --h results/chips-diag-h --p results/chips-diag-p --upcoming 2026:10-19
```
Expected: three comparison lines (`P vs H`, `P vs S0`, `H vs S0`), then `tables in results/chips-diag`. A crash on real data means real data differs from the synthetic tests, for example a dtype or an empty group. In that case, add a test that reproduces it in `tests/test_chip_diagnostics.py`, then fix it.

- [ ] **Step 3: the user's `/chips` for section 6** (read-only, on the server, with the scratch script already used on 2026-10-28)

```bash
scp "C:/Users/samin/AppData/Local/Temp/claude/C--Users-samin-Documents-GitHub-fpl-optimizer--claude-worktrees-fpl-bot-plan-improvements-22577f/382d5683-019a-4c9d-8cf6-cb7a345e5924/scratchpad/chips_check.py" fplopt-server:/tmp/chips_check.py
ssh fplopt-server 'cd ~/fpl-optimizer && ~/.local/bin/uv run python /tmp/chips_check.py' > results/chips-diag/q6-user-chips.txt
```
Expected: scenario objectives and the best plan's chips for entry 4570069.

- [ ] **Step 4: commit the experiment-log rows**

```bash
git add results/experiments.csv
git commit -m "results: experiment log, chip diagnostics develop runs (S0, H, P)"
```

---

### Task 9: the report

**Files:** Create `docs/research/2026-10-chip-diagnostics.md`

- [ ] **Step 1: write it from the tables** with these sections and contents:
  1. **The gaps under today's model:** `comparisons.json` against Oct 11's develop numbers (P vs H +0.14/GW, H vs S0 +2.39/GW).
  2. **Where the gap comes from:** the top rows of `q1-gap-chip-P-H.csv` and `q1-gap-chip-H-S0.csv`, the cell pattern, and timing (median `gw_index` per chip and arm from `q1-timing.csv`).
  3. **What a chip is worth, and when:** `q2-gain-*.csv`; doubles vs singles for Triple Captain, blanks for Free Hit, template vs random.
  4. **Forecast vs outcome:** `q3-forecast-slopes.csv`, per chip slope ± SE, and what a slope below 1 means for the thresholds and the table.
  5. **Holding value:** `q4-holding-value.csv`, the realized/table and ex-ante/table ratios per chip at L = 1, 5, 10 and 19. For FH/WC, the ex-ante ratio × Q3's slope.
  6. **Congestion:** `q5-*`.
  7. **Ranked ideas for sub-project 2:** one line each, with the evidence and a plausible develop gain per GW. Rank by gain × confidence.
  8. **Your set-1 chips (untested research):** `q6-upcoming.csv` (doubles and blanks GW10–19), `q6-user-chips.txt`, and what 3–6 imply.
  9. Every number states its source file. Develop seasons only, except section 8.

- [ ] **Step 2: commit**

```bash
git add docs/research/2026-10-chip-diagnostics.md
git commit -m "docs(research): chip diagnostics report"
```

- [ ] **Step 3: publish it as a private page** (Artifact, built from the report's content) and give the user the link. Then agree with the user on the ranked idea list as sub-project 2's starting point.

- [ ] **Step 4: PR** `feat/chip-diagnostics` → main: the module, tests, driver, report and log rows. Merge only on the user's go-ahead.
