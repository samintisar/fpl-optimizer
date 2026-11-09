# Price-Change Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the perfect-foresight upper bound of (1) planning with future prices and (2) timing transfers within a GW, and decide by the spec's fixed thresholds whether to build either.

**Architecture:** Two small Rust additions to the backtester: a `+bank<n>` start modifier and a `market_value` column in `gws.parquet`. Three backtests run on the server. Three analysis scripts in `dev/` (`price_value_gap.py`, `price_timing.py`, `price_sizing.py`) read their outputs, FPL's public API and `data/`, and write JSON. A write-up records the numbers and the decision.

**Tech Stack:** Rust (`crates/fplopt-backtest`, `crates/fplopt-cli`), Python 3 + pandas + scipy for `dev/` scripts (`uv run python`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-price-sizing-design.md`. Branch `feat/price-sizing` (from `main` 8f8c8f9), worktree `C:\Users\samin\Documents\GitHub\fpl-optimizer\.claude\worktrees\v1r`.

**Facts checked while writing this plan:**
- `gws.parquet` already has `season, start_id, gw, gw_index, deadline, net_points, bank, squad_value` (selling prices, bank excluded) and `transfers` (JSON text `[[out_key, in_key], ...]`), written in `crates/fplopt-cli/src/backtest.rs:300-335` from `SimRow` (`crates/fplopt-backtest/src/simulator.rs:190-220`, filled at ~line 340).
- FPL's `entry/{id}/history/` `value` is team value at **market** prices (the user's GW9 `value` 1030 ≈ Σ `now_cost` 1032 of his picks today). Whether it includes `bank` is checked in Task 4 Step 6.
- Start states: `StartSpec` and `parse_start_specs` in `crates/fplopt-backtest/src/evaluate.rs:33-125`; tests near line 1060.
- `dev/backtest_parity.py::compare_tables` (line 187) fails on any column-list difference, so the new Rust-only column must be dropped there.
- Data: `data/player_snapshot.parquet` (`snapshot_at, season, player_key, now_cost, …`), `data/player_gw.parquet` (`player_key, season, gw, value, …`). The worktree has no `data/`; the main checkout (`C:\Users\samin\Documents\GitHub\fpl-optimizer\data`) and the server (`~/fpl-optimizer/data`) do.
- The live spec (`holdout.SYSTEM`, `crates/fplopt-bot/src/planner.rs:44`): `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`.

**Rules for every task:** never touch season 2025 (the holdout); `raw/` is never written; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never push or run anything on the server without the controller's go-ahead (the controller asks the user).

---

### Task 1: `+bank<n>` start modifier (Rust)

**Files:**
- Modify: `crates/fplopt-backtest/src/evaluate.rs` (StartSpec, `start_id`, `build`, `parse_start_specs`, tests)
- Modify: `crates/fplopt-cli/src/backtest.rs` (the `--starts` doc comment near line 57)

- [ ] **Step 1: Write the failing tests** in `evaluate.rs`'s `#[cfg(test)] mod tests`, next to the existing `parse_start_specs` test (~line 1060):

```rust
#[test]
fn bank_modifier_parses_and_labels() {
    let specs =
        parse_start_specs("template+bank10@1,random:2+bank5@20,squad:dir/t.toml+bank10@2")
            .unwrap();
    let ids: Vec<String> = specs.iter().map(|s| s.start_id(None)).collect();
    assert_eq!(
        ids,
        ["template+bank10@1", "random0+bank5@20", "random1+bank5@20", "squad:t+bank10@2"]
    );
    assert_eq!(specs[0].extra_bank, 10);
    assert_eq!(specs[1].extra_bank, 5);
    assert!(parse_start_specs("template+bank@1").is_err());
    assert!(parse_start_specs("template+bank0@1").is_err());
    assert!(parse_start_specs("template+bankx@1").is_err());
    // a base start and its modified twin may run together
    assert!(parse_start_specs("template@1,template+bank10@1").is_ok());
    // no modifier: extra_bank 0, ids unchanged
    let plain = parse_start_specs("template@1").unwrap();
    assert_eq!(plain[0].extra_bank, 0);
    assert_eq!(plain[0].start_id(None), "template@1");
}

#[test]
fn extra_bank_adds_to_the_bank_only() {
    let state = SquadState {
        season: 2020,
        gw_index: 1,
        holdings: Vec::new(),
        bank: 7,
        free_transfers: 1,
        chips_used: Vec::new(),
        freehit_backup: None,
        freehit_bank: None,
    };
    let more = with_extra_bank(state.clone(), 10);
    assert_eq!(more.bank, 17);
    assert_eq!(SquadState { bank: 7, ..more }, state);
}
```

If `SquadState` isn't in scope in the tests module, add `use fplopt_optimize::state::SquadState;` (it is `pub struct SquadState` in `crates/fplopt-optimize/src/state.rs:45`; check how `evaluate.rs` already imports it and reuse that path).

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p fplopt-backtest --lib bank_modifier extra_bank`
Expected: compile errors (`extra_bank` field and `with_extra_bank` missing).

- [ ] **Step 3: Implement**

In `StartSpec` add the field and document it:

```rust
/// One start state: the template squad, a random squad with `seed`, or a real squad from a
/// squad file (`squad`; Rust only), at `gw_index`, with `extra_bank` tenths of £m added to the
/// bank (`+bank<n>`; Rust only, for the price sizing study).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartSpec {
    pub random: bool,
    pub gw_index: i64,
    pub seed: u64,
    pub squad: Option<PathBuf>,
    pub extra_bank: i64,
}
```

`start_id` puts the modifier before `@`:

```rust
pub fn start_id(&self, gw_index: Option<i64>) -> String {
    let gw = gw_index.unwrap_or(self.gw_index);
    let bank = if self.extra_bank > 0 { format!("+bank{}", self.extra_bank) } else { String::new() };
    if let Some(path) = &self.squad {
        let stem = path.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default();
        return format!("squad:{stem}{bank}@{gw}");
    }
    if self.random {
        format!("random{}{bank}@{gw}", self.seed)
    } else {
        format!("template{bank}@{gw}")
    }
}
```

`build` applies it after building the base state:

```rust
fn build(&self, store: &DataStore, season: i64, gw_index: i64, rules: &Rules) -> Result<SquadState> {
    let state = if let Some(path) = &self.squad {
        squad_file_state(store, season, gw_index, path, rules)?
    } else {
        let deadline = season_schedule(store, season, gw_index)?[0].deadline;
        let view = store.as_of(deadline);
        if self.random {
            random_state(&view, rules, self.seed, Some(MAX_BANK))?
        } else {
            template_state(&view, rules)?
        }
    };
    Ok(with_extra_bank(state, self.extra_bank))
}
```

and the helper (module level, next to `StartSpec`):

```rust
/// The start state with `extra` tenths of £m more in the bank (nothing else changes).
fn with_extra_bank(mut state: SquadState, extra: i64) -> SquadState {
    state.bank += extra;
    state
}
```

In `parse_start_specs`, right after `let gw_index ...` and the `gw_index < 1` check, split off the modifier from `kind`, then match on the base kind; every `StartSpec { .. }` literal gets `extra_bank`:

```rust
let (kind, extra_bank) = match kind.rsplit_once("+bank") {
    None => (kind, 0),
    Some((base, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
        let n: i64 = n.parse()?;
        if n < 1 {
            bail!("bad start spec '{item}': +bank<n> needs n >= 1 (tenths of £m)");
        }
        (base, n)
    }
    Some(_) => return Err(bad()),
};
```

Update the `bad()` message and the function's doc comment to list `+bank<n>` ("`template+bank10@1`: the template at 1 with £1.0m more in the bank; any start kind"). Update the `--starts` doc comment in `crates/fplopt-cli/src/backtest.rs` (~line 57) the same way.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fplopt-backtest --lib`
Expected: all pass, including the existing `parse_start_specs` test.

- [ ] **Step 5: Workspace check**

Run: `cargo test --workspace` then `cargo clippy --workspace --all-targets -- -D warnings` then `cargo fmt --all`
Expected: pass, no warnings. Fix any other `StartSpec { .. }` literal the compiler reports by adding `extra_bank: 0`.

- [ ] **Step 6: Commit**

```bash
git add crates/fplopt-backtest/src/evaluate.rs crates/fplopt-cli/src/backtest.rs
git commit -m "feat(backtest): +bank<n> start modifier for the price sizing study

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `market_value` column (Rust) and the parity script

**Files:**
- Modify: `crates/fplopt-backtest/src/state.rs` (new `market_value` fn + test)
- Modify: `crates/fplopt-backtest/src/simulator.rs` (`SimRow.market_value`, filled next to `squad_value`)
- Modify: `crates/fplopt-cli/src/backtest.rs` (column after `squad_value`, ~line 324)
- Modify: `dev/backtest_parity.py` (`compare_tables`, line 187)

- [ ] **Step 1: Write the failing test** in `state.rs`'s tests module (create `#[cfg(test)] mod tests { use super::*; ... }` at the end of the file if there is none):

```rust
#[test]
fn market_value_is_current_prices_plus_bank() {
    let h = |key, purchase, price| Holding {
        player_key: key,
        element_type: 2,
        team_key: 1,
        purchase_price: purchase,
        price,
    };
    let state = SquadState {
        season: 2020,
        gw_index: 5,
        holdings: vec![h(1, 50, 53), h(2, 60, 58)],
        bank: 12,
        free_transfers: 1,
        chips_used: Vec::new(),
        freehit_backup: None,
        freehit_bank: None,
    };
    assert_eq!(market_value(&state), 53 + 58 + 12);
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `cargo test -p fplopt-backtest --lib market_value`
Expected: compile error, `market_value` not found.

- [ ] **Step 3: Implement** in `state.rs`, next to `squad_value` (line ~61):

```rust
/// FPL's team value: the holdings at their current prices plus the bank (what FPL's
/// `entry/{id}/history/` reports as `value`; `squad_value` is the selling-price sum instead).
pub fn market_value(state: &SquadState) -> i64 {
    state.holdings.iter().map(|h| h.price).sum::<i64>() + state.bank
}
```

Add it to the module doc's list (line 2). In `simulator.rs`: import it next to `squad_value` (line 29), add `pub market_value: i64,` after `pub squad_value: i64,` in `SimRow` with a doc comment `/// state::market_value of gw_state (Rust only; dev/backtest_parity.py ignores it).`, and fill `market_value: market_value(&gw_state),` after `squad_value: ...` (~line 351). In `backtest.rs` add `.with("market_value", r(&|g| g.row.market_value))` right after the `squad_value` line.

- [ ] **Step 4: Parity script** — in `dev/backtest_parity.py`, add above `compare_tables`:

```python
RUST_ONLY = ("market_value",)  # gws.parquet columns Python never wrote (price sizing, 2026-11)
```

and as the first lines of `compare_tables`:

```python
    left, right = pd.read_parquet(a), pd.read_parquet(b)
    left = left.drop(columns=[c for c in RUST_ONLY if c in left.columns])
    right = right.drop(columns=[c for c in RUST_ONLY if c in right.columns])
```

(replacing the existing `left, right = ...` line).

- [ ] **Step 5: Run everything**

Run: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all`, `uv run ruff check dev/backtest_parity.py && uv run ruff format dev/backtest_parity.py`
Expected: pass. If a Rust golden test compares `gws` columns against a Python golden (search `crates/` tests for `squad_value`), drop `market_value` there the same way and say so in the commit message.

- [ ] **Step 6: Commit**

```bash
git add crates/fplopt-backtest/src/state.rs crates/fplopt-backtest/src/simulator.rs crates/fplopt-cli/src/backtest.rs dev/backtest_parity.py
git commit -m "feat(backtest): market_value column (FPL's team value) in gws.parquet

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `dev/price_value_gap.py` (fetch + gap)

**Files:**
- Create: `dev/price_value_gap.py`
- Create: `tests/test_price_sizing_scripts.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_price_sizing_scripts.py`:

```python
"""Unit tests of the price sizing scripts in dev/ (spec 2026-11-06-price-sizing-design.md)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "dev" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gap = _load("price_value_gap")


def test_sample_ranks_is_seeded_and_in_range():
    a, b = gap.sample_ranks(0, 500), gap.sample_ranks(0, 500)
    assert a == b
    assert len(a["top10k"]) == 500 and len(set(a["top10k"])) == 500
    assert min(a["top10k"]) >= 1 and max(a["top10k"]) <= 10_000
    assert min(a["10k-100k"]) >= 10_001 and max(a["10k-100k"]) <= 100_000
    assert gap.sample_ranks(1, 500) != a


def test_page_and_slot_of_rank():
    assert gap.page_of(1) == 1 and gap.slot_of(1) == 0
    assert gap.page_of(50) == 1 and gap.slot_of(50) == 49
    assert gap.page_of(51) == 2 and gap.slot_of(51) == 0


def test_value_gap_at_last_common_gw():
    managers = pd.DataFrame(
        {
            "group": ["top10k"] * 4,
            "entry": [1, 1, 2, 2],
            "gw": [1, 2, 1, 2],
            "value": [1000, 1010, 1000, 1030],
        }
    )
    ours = pd.DataFrame({"gw": [1, 2, 3], "market_value": [1000, 1005, 1007]})
    out = gap.value_gap(managers, ours, n_gws=38)
    g = out["top10k"]
    assert g["gw"] == 2
    assert g["median"] == 1020 and g["ours"] == 1005 and g["gap"] == 15
    assert g["gap_season"] == pytest.approx(15 * 38 / 2)
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_price_sizing_scripts.py -q`
Expected: FAIL (`dev/price_value_gap.py` missing).

- [ ] **Step 3: Implement `dev/price_value_gap.py`:**

```python
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
    for group, ranks in sample_ranks(seed).items():
        for i, rank in enumerate(ranks):
            page = get_cached(
                out / "standings" / f"{page_of(rank)}.json.gz",
                f"leagues-classic/314/standings/?page_standings={page_of(rank)}",
                pause,
            )
            entry = page["standings"]["results"][slot_of(rank)]["entry"]
            history = get_cached(
                out / "history" / f"{entry}.json.gz", f"entry/{entry}/history/", pause
            )
            for h in history["current"]:
                rows.append(
                    {
                        "group": group,
                        "rank": rank,
                        "entry": entry,
                        "gw": h["event"],
                        "value": h["value"],
                        "bank": h["bank"],
                    }
                )
            if i % 50 == 0:
                print(f"{group}: {i}/{len(ranks)}", flush=True)
    managers = pd.DataFrame(rows)
    managers.to_parquet(out / "managers.parquet", index=False)
    return managers


def value_gap(managers: pd.DataFrame, ours: pd.DataFrame, n_gws: int = 38) -> dict:
    """Per group: the group's median `value` at the last GW both sides have, ours
    (`market_value` of our row for that GW), the gap and the gap scaled to n_gws."""
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
    ours = gws[gws["start_id"].str.startswith("template")].sort_values("gw")
    result = value_gap(managers, ours[["gw", "market_value"]])
    history = get_cached(
        args.managers / "history" / f"{args.entry}.json.gz", f"entry/{args.entry}/history/", 1.0
    )
    result["user"] = {
        str(h["event"]): {"value": h["value"], "bank": h["bank"]} for h in history["current"]
    }
    result["paths"] = managers.groupby(["group", "gw"])["value"].median().unstack(0).to_dict()
    result["ours_path"] = dict(zip(ours["gw"].astype(int), ours["market_value"].astype(int)))
    (args.managers / "gap.json").write_text(json.dumps(result, indent=2, default=int))
    print(json.dumps({k: result[k] for k in GROUPS}, indent=2))


if __name__ == "__main__":
    main()
```

(Format with ruff; the long lines above wrap.)

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_price_sizing_scripts.py -q`
Expected: 3 passed.

- [ ] **Step 5: Lint**

Run: `uv run ruff check dev/price_value_gap.py tests/test_price_sizing_scripts.py && uv run ruff format dev/price_value_gap.py tests/test_price_sizing_scripts.py`

- [ ] **Step 6: VERIFY FPL's `value` includes the bank** (no code change unless it fails). Run:

```bash
uv run python -c "import json,urllib.request as u;r=u.Request('https://fantasy.premierleague.com/api/entry/4570069/history/',headers={'User-Agent':'fpl-optimizer'});h=json.load(u.urlopen(r))['current'];print([(x['event'],x['value'],x['bank']) for x in h[:3]])"
```

Expected: GW1 `value` within a few tenths of 1000 whatever `bank` is (everyone starts at £100.0m including the bank). If instead `value + bank` ≈ 1000, FPL's value excludes the bank: then `value_gap` must compare `value + bank`; change `fetch` to store `value` as `h["value"] + h["bank"]`, re-run the tests, and note it in the commit.

- [ ] **Step 7: Commit**

```bash
git add dev/price_value_gap.py tests/test_price_sizing_scripts.py
git commit -m "feat(dev): price sizing value-gap script

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `dev/price_timing.py` (perfect timing gain)

**Files:**
- Create: `dev/price_timing.py`
- Modify: `tests/test_price_sizing_scripts.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_price_sizing_scripts.py`):

```python
timing = _load("price_timing")


def test_selling_price_rule():
    assert timing.sell_price(50, 53) == 51  # half the rise, rounded down
    assert timing.sell_price(50, 54) == 52
    assert timing.sell_price(50, 48) == 48  # a fall is taken in full
    assert timing.sell_price(50, 50) == 50


def test_timing_gain_sell_before_drop_buy_before_rise():
    # sold player: purchase 50, was 52 in the window, 50 at the deadline -> sell at 51 not 50
    # bought player: 60 in the window, 61 at the deadline -> 1 saved
    assert timing.timing_gain(50, [52, 51], 50, [60, 61], 61) == (1, 1)


def test_timing_gain_is_zero_without_changes():
    assert timing.timing_gain(50, [50, 50], 50, [60, 60], 60) == (0, 0)
    assert timing.timing_gain(50, [], 50, [], 60) == (0, 0)


def test_replay_tracks_purchase_prices():
    gws = pd.DataFrame(
        {
            "season": [2021, 2021, 2021],
            "start_id": ["template@1"] * 3,
            "gw": [1, 2, 3],
            "gw_index": [1, 2, 3],
            "deadline": pd.to_datetime(
                ["2021-08-13 17:30", "2021-08-21 10:00", "2021-08-28 10:00"], utc=True
            ),
            "transfers": ["[]", "[[1, 2]]", "[[2, 3]]"],
        }
    )
    deadline_price = {
        (2021, 1, 1): 50,
        (2021, 2, 1): 50,
        (2021, 2, 2): 60,
        (2021, 3, 2): 61,
        (2021, 3, 3): 70,
    }
    snaps = {1: [], 2: [(pd.Timestamp("2021-08-28 01:30", tz="UTC"), 62)], 3: []}
    rows = timing.replay(gws, deadline_price, snaps, window=pd.Timedelta(hours=24))
    # GW3: player 2 bought at 60, 62 on the night before the deadline, 61 at the deadline:
    # selling price 61 vs 60 -> 1 tenth; nothing on the buy side
    assert [(r["gw"], r["sell_gain"], r["buy_gain"]) for r in rows] == [(2, 0, 0), (3, 1, 0)]
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_price_sizing_scripts.py -q`
Expected: the new tests fail (module missing).

- [ ] **Step 3: Implement `dev/price_timing.py`:**

```python
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
```

Note for the implementer: `snapshot_at` and `deadline` must both be tz-aware UTC for the comparisons; if `snapshot_at` loads naive, `tz_localize("UTC")` it. If `player_snapshot` holds more than one row per player and time (several sources), keep it: duplicates don't change a max/min.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_price_sizing_scripts.py -q`
Expected: 7 passed.

- [ ] **Step 5: Lint** — `uv run ruff check dev/price_timing.py tests/test_price_sizing_scripts.py && uv run ruff format dev/price_timing.py tests/test_price_sizing_scripts.py`

- [ ] **Step 6: Commit**

```bash
git add dev/price_timing.py tests/test_price_sizing_scripts.py
git commit -m "feat(dev): price sizing perfect-timing script

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `dev/price_sizing.py` (points per £1m, combine, decide)

**Files:**
- Create: `dev/price_sizing.py`
- Modify: `tests/test_price_sizing_scripts.py`

- [ ] **Step 1: Write the failing tests** (append):

```python
sizing = _load("price_sizing")


def _gws(start_ids, points):
    rows = []
    for sid, pts in zip(start_ids, points):
        for gw, p in enumerate(pts, start=1):
            rows.append({"season": 2020, "start_id": sid, "gw": gw, "net_points": p})
    return pd.DataFrame(rows)


def test_points_per_million_pairs_by_start():
    base = _gws(["template@1", "random0@1", "template@20"], [[50, 50], [40, 40], [30]])
    bank = _gws(
        ["template+bank10@1", "random0+bank10@1", "template+bank10@20"], [[52, 51], [41, 42], [30]]
    )
    out = sizing.points_per_million(base, bank, extra_bank=10)
    assert out["1"]["n"] == 2 and out["1"]["mean"] == pytest.approx((3 + 3) / 2)
    assert out["20"]["n"] == 1 and out["20"]["mean"] == 0


def test_decision_thresholds():
    d = sizing.decide(planning_points=19.9, timing_points=10.0)
    assert d == {"planning": False, "timing": True}
    assert sizing.decide(20.0, 9.9) == {"planning": True, "timing": False}
```

- [ ] **Step 2: Run to see them fail** — `uv run pytest tests/test_price_sizing_scripts.py -q`; expected: the new tests fail.

- [ ] **Step 3: Implement `dev/price_sizing.py`:**

```python
"""Price sizing: points per £1m (develop backtests with and without +bank10), the two upper
bounds and the decision, with the spec's fixed thresholds
(docs/superpowers/specs/2026-11-06-price-sizing-design.md §3).

    uv run python dev/price_sizing.py --base results/price-sizing-base \
        --bank results/price-sizing-bank10 --gap results/price-sizing-managers/gap.json \
        --timing results/price-sizing-timing/timing.json --out results/price-sizing
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

PLANNING_MIN = 20.0  # points a season (spec §3, fixed before any number was seen)
TIMING_MIN = 10.0
GAP_GROUP = "top10k"  # the upper bound uses the larger, top-10k gap


def _totals(gws: pd.DataFrame) -> pd.Series:
    return gws.groupby(["season", "start_id"])["net_points"].sum()


def points_per_million(base: pd.DataFrame, bank: pd.DataFrame, extra_bank: int) -> dict:
    """Mean (bank − base) season total per start GW, scaled to £1m, with an 80% t CI."""
    b = _totals(base)
    m = _totals(bank)
    m.index = pd.MultiIndex.from_tuples(
        [(s, i.replace(f"+bank{extra_bank}", "")) for s, i in m.index], names=b.index.names
    )
    diff = (m - b).dropna() * 10 / extra_bank
    out = {}
    for gw, d in diff.groupby(diff.index.get_level_values("start_id").str.split("@").str[1]):
        n = int(d.size)
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        half = float(stats.t.ppf(0.9, n - 1) * se) if n > 1 else float("nan")
        out[gw] = {
            "n": n,
            "mean": float(d.mean()),
            "se": se,
            "ci80": [float(d.mean()) - half, float(d.mean()) + half],
        }
    return out


def decide(planning_points: float, timing_points: float) -> dict:
    return {"planning": planning_points >= PLANNING_MIN, "timing": timing_points >= TIMING_MIN}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("base", "bank", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--gap", type=Path, required=True)
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--extra-bank", type=int, default=10)
    args = ap.parse_args()
    rate = points_per_million(
        pd.read_parquet(args.base / "gws.parquet"),
        pd.read_parquet(args.bank / "gws.parquet"),
        args.extra_bank,
    )
    # GW1 starts before 2021/22 fall back to gw_index 2; pool them with GW1 as the season rate
    early = [v for k, v in rate.items() if k in ("1", "2")]
    n = sum(v["n"] for v in early)
    season_rate = sum(v["mean"] * v["n"] for v in early) / n
    gap = json.loads(args.gap.read_text())[GAP_GROUP]
    timing = json.loads(args.timing.read_text())
    planning_points = gap["gap_season"] / 10 * season_rate
    timing_points = timing["24h"]["mean_per_season"] / 10 * season_rate
    result = {
        "points_per_million": rate,
        "season_rate": season_rate,
        "gap": gap,
        "planning_points": planning_points,
        "timing_points_24h": timing_points,
        "timing_points_week": timing["week"]["mean_per_season"] / 10 * season_rate,
        "thresholds": {"planning": PLANNING_MIN, "timing": TIMING_MIN},
        "decision": decide(planning_points, timing_points),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sizing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests** — `uv run pytest tests/test_price_sizing_scripts.py -q`; expected: 9 passed.

- [ ] **Step 5: Lint** — `uv run ruff check dev/price_sizing.py tests/test_price_sizing_scripts.py && uv run ruff format dev/price_sizing.py tests/test_price_sizing_scripts.py`

- [ ] **Step 6: Commit**

```bash
git add dev/price_sizing.py tests/test_price_sizing_scripts.py
git commit -m "feat(dev): price sizing combine-and-decide script

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Run the backtests on the server (controller only)

The controller does this task, not a subagent: it pushes a branch and uses the server. **Ask the user before pushing `feat/price-sizing` and before starting server jobs.** The live bot runs from `~/fpl-optimizer` and must not be disturbed: use a separate worktree, never `git checkout` in `~/fpl-optimizer`.

- [ ] **Step 1: Push the branch** (after the user's OK): `git push -u origin feat/price-sizing`

- [ ] **Step 2: Server worktree and build**

```bash
ssh fplopt-server 'cd ~/fpl-optimizer && git fetch -q origin feat/price-sizing && git worktree add ~/price-sizing origin/feat/price-sizing && cd ~/price-sizing && ln -s ~/fpl-optimizer/data data && git submodule update --init --recursive && PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH cargo build --release -p fplopt-cli 2>&1 | tail -2'
```

(If the submodule build is slow, `CARGO_TARGET_DIR=~/fpl-optimizer/target` reuses the live build's dependency artefacts but would overwrite the live binary: don't. A fresh build takes ~10–20 min.)

- [ ] **Step 3: VERIFY season 2026 runs** (the spec's VERIFY):

```bash
ssh fplopt-server 'cd ~/price-sizing && nice ./target/release/fplopt-rs backtest run --seasons 2026 --spec optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --starts template@1 --jobs 2 --family "price sizing" --out results/price-sizing-2026 2>&1 | tail -15'
```

Expected: GW1–9 rows (the GWs with results). If refused or no rows: per the spec, record the refusal and compare against the user's own entry instead (in `price_value_gap.py gap`, use the `user` history as `ours`: `market_value` = `value`). Confirm `market_value` is in `results/price-sizing-2026/gws.parquet`.

- [ ] **Step 4: Develop backtests** (background; ~1–3 h each at `--jobs 12`, both share the xP cache):

```bash
ssh fplopt-server 'cd ~/price-sizing && nohup sh -c "nice ./target/release/fplopt-rs backtest run --seasons 2016-2022 --spec optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --starts template@1,random:3@1,template@20,random:3@20 --jobs 12 --family \"price sizing\" --out results/price-sizing-base && nice ./target/release/fplopt-rs backtest run --seasons 2016-2022 --spec optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --starts template+bank10@1,random:3+bank10@1,template+bank10@20,random:3+bank10@20 --jobs 12 --family \"price sizing\" --out results/price-sizing-bank10" > price-sizing.log 2>&1 &'
```

Run it with the Bash tool's `run_in_background` waiting on a loop that exits when both `gws.parquet` files exist or the log shows an error; never poll in the foreground.

- [ ] **Step 5: Copy results back**

```bash
scp -r fplopt-server:price-sizing/results/price-sizing-2026 fplopt-server:price-sizing/results/price-sizing-base fplopt-server:price-sizing/results/price-sizing-bank10 results/
ssh fplopt-server 'cd ~/price-sizing && git diff -U0 results/experiments.csv' 
```

Append the new `results/experiments.csv` rows (family `price sizing`) to the local `results/experiments.csv` by hand-copying the added lines; commit them in Task 7.

---

### Task 7: Run the analysis, write up, record the decision (controller)

**Files:**
- Create: `docs/research/2026-11-price-sizing.md`
- Modify: `docs/PLAN.md` (§1 non-goals line, §12 decision row)
- Modify: `results/experiments.csv` (rows from Task 6)

- [ ] **Step 1: Fetch managers** (~35 min; locally, background):

`uv run python dev/price_value_gap.py fetch --out results/price-sizing-managers`

- [ ] **Step 2: Gap** — `uv run python dev/price_value_gap.py gap --managers results/price-sizing-managers --ours results/price-sizing-2026`

- [ ] **Step 3: Timing** — `uv run python dev/price_timing.py --run results/price-sizing-base --data C:/Users/samin/Documents/GitHub/fpl-optimizer/data --out results/price-sizing-timing` (the main checkout's `data/`; check it covers 2021–2022 snapshots: `player_snapshot` rows with `season` 2021 and 2022 exist).

- [ ] **Step 4: Combine** — `uv run python dev/price_sizing.py --base results/price-sizing-base --bank results/price-sizing-bank10 --gap results/price-sizing-managers/gap.json --timing results/price-sizing-timing/timing.json --out results/price-sizing`

- [ ] **Step 5: Write `docs/research/2026-11-price-sizing.md`**, in the style of `docs/research/2026-11-chip-policy-2.md`:
  - the question and the thresholds (fixed in the spec before the numbers);
  - 1a: the value paths (ours, top-10k and 10k–100k medians and quartiles, the user's entry), the GW9 gap and the season scaling, with the survivorship caveat;
  - 1b: points per £1m at GW1 and GW20 with 80% CIs, n;
  - 2: perfect timing £ per season (24 h and week), the sell share, converted to points;
  - the decision from `sizing.json`, verbatim, and what happens next (a full-foresight design, a timing-hint design, or stop).

- [ ] **Step 6: PLAN** — §12 gets a 2026-11-06 row with the two upper bounds and the decision. If both are below the thresholds, §1's "Price-change prediction." becomes "Price-change prediction (measured 2026-11: upper bounds X and Y points a season, `docs/research/2026-11-price-sizing.md`)."; otherwise leave §1 for the next design to change.

- [ ] **Step 7: Commit**

```bash
git add docs/research/2026-11-price-sizing.md docs/PLAN.md results/experiments.csv
git commit -m "docs: price sizing results and decision

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Clean up the server worktree** (after the user has the results): `ssh fplopt-server 'cd ~/fpl-optimizer && git worktree remove ~/price-sizing --force'`, then report to the user and ask whether to open a PR for the branch.
