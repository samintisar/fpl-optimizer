# Noise Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the standard error of eight comparison methods on real planner pairs and on placebo pairs, then adopt the cheapest method that reaches ≤ 4.5 points a season (spec `docs/superpowers/specs/2026-11-07-noise-study-design.md`).

**Architecture:**
- A placebo setting (`xp_jitter`, `jitter_seed`) on the Rust optimizer policy multiplies each player's xP by (1 + σ·ε), with ε drawn deterministically from a hash. Two seeds make two equal-in-expectation planners.
- `dev/noise_study.py` drives `fplopt-rs backtest compare` for each comparison and window length, then reads `paired.parquet` / `per_decision.parquet` to put every method on one scale (points per season), with its SE, a placebo check and server time.

**Tech Stack:** Rust (`crates/fplopt-backtest`), Python 3.12 + pandas (`dev/`, `tests/`), the server `ssh fplopt-server`.

**Rules for every task:**
- Cargo always with `-j 2` (the local linker runs out of memory otherwise).
- Nothing new goes into `src/fplopt` (Python package). Dev scripts in `dev/` are fine.
- Never read 2025/26 (the holdout) or 2023/24–2024/25.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The worktree hook rejects bash commands that chain `cd` with git. Run git with `git -C <path>` or as separate simple commands.

---

## File structure

- Create `crates/fplopt-backtest/src/jitter.rs`: `XpJitter { sigma, seed }`, the hash, ε, and `XpJitter::apply(&XpFrame, season, gw_index) -> XpFrame`. One responsibility: the placebo wobble.
- Modify `crates/fplopt-backtest/src/lib.rs`: `pub mod jitter;`.
- Modify `crates/fplopt-backtest/src/policies.rs`:
  - the field `jitter: Option<XpJitter>` on `Policy::Optimizer`;
  - its name suffix `jit=σ/seed`;
  - applied in `optimizer_plans`.
- Modify `crates/fplopt-backtest/src/specs.rs`: the keys `xp_jitter` (Float) and `jitter_seed` (Int) on `optimizer`, their defaults, and their checks.
- Modify `crates/fplopt-backtest/src/probes.rs` (line ~125) and `policies.rs` (line ~648): the other `Policy::Optimizer { .. }` constructors get `jitter: None`.
- Create `dev/noise_study.py`: `run` (drive the compares) and `report` (methods table, choice).
- Create `tests/test_noise_study.py`: the pure functions.
- Create `docs/research/2026-11-noise-study.md`; modify `docs/PLAN.md` (§7 evaluation design, §12 row) and `CLAUDE.md` (the standard comparison). These are written in Task 5, after the run.

---

### Task 1: The placebo wobble in Rust (`xp_jitter`, `jitter_seed`)

**Files:**
- Create: `crates/fplopt-backtest/src/jitter.rs`
- Modify: `crates/fplopt-backtest/src/lib.rs`, `crates/fplopt-backtest/src/policies.rs`, `crates/fplopt-backtest/src/specs.rs`, `crates/fplopt-backtest/src/probes.rs`
- Test: unit tests in `jitter.rs` and `specs.rs`

- [ ] **Step 1: Write `jitter.rs` with its tests (failing until it compiles into the crate)**

```rust
//! The placebo wobble on xP (`xp_jitter`, `jitter_seed` on the optimizer policy; noise study,
//! `docs/superpowers/specs/2026-11-07-noise-study-design.md`): each player's xP for a GW is
//! multiplied by (1 + sigma * eps), eps ~ N(0, 1) drawn from a hash of (seed, season, decision
//! gw_index, player_key, GW). Two seeds give two planners that are equal in expectation, so
//! their compared difference is 0 in truth. Deterministic and independent of thread order; one
//! eps per player and GW (a double GW's fixture rows share it).

use fplopt_optimize::XpFrame;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XpJitter {
    pub sigma: f64,
    pub seed: u64,
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f64 {
    // (0, 1): never 0, so ln is finite.
    ((h >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// eps ~ N(0, 1) for (seed, season, decision gw_index, player_key, GW), by Box-Muller.
pub fn epsilon(seed: u64, season: i64, gw_index: i64, player_key: i64, gw: i64) -> f64 {
    let mut h = splitmix64(seed);
    for v in [season, gw_index, player_key, gw] {
        h = splitmix64(h ^ v as u64);
    }
    let (u1, u2) = (unit(h), unit(splitmix64(h ^ 0xD1B5_4A32_D192_ED03)));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

impl XpJitter {
    /// A copy of `frame` with every non-null xP multiplied by (1 + sigma * eps).
    pub fn apply(&self, frame: &XpFrame, season: i64, gw_index: i64) -> XpFrame {
        let mut out = frame.clone();
        for i in 0..out.len() {
            if let Some(x) = out.xp[i] {
                let eps = epsilon(self.seed, season, gw_index, out.player_key[i], out.gw[i]);
                out.xp[i] = Some(x * (1.0 + self.sigma * eps));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> XpFrame {
        XpFrame {
            player_key: vec![1, 1, 1, 2],
            gw: vec![5, 5, 6, 5],
            gw_index: vec![5, 5, 6, 5],
            horizon: vec![0, 0, 1, 0],
            xp: vec![Some(4.0), Some(2.0), Some(3.0), None],
            ..XpFrame::default()
        }
    }

    #[test]
    fn deterministic_and_seed_dependent() {
        assert_eq!(epsilon(1, 2022, 5, 10, 5), epsilon(1, 2022, 5, 10, 5));
        assert_ne!(epsilon(1, 2022, 5, 10, 5), epsilon(2, 2022, 5, 10, 5));
        assert_ne!(epsilon(1, 2022, 5, 10, 5), epsilon(1, 2022, 6, 10, 5));
    }

    #[test]
    fn standard_normal_moments() {
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|k| epsilon(7, 2022, 1, k, 3)).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn apply_scales_xp_one_eps_per_player_gw() {
        let j = XpJitter { sigma: 0.05, seed: 3 };
        let out = j.apply(&frame(), 2022, 5);
        let e = epsilon(3, 2022, 5, 1, 5);
        assert_eq!(out.xp[0], Some(4.0 * (1.0 + 0.05 * e)));
        // the double GW's second fixture row shares the eps
        assert_eq!(out.xp[1], Some(2.0 * (1.0 + 0.05 * e)));
        assert_eq!(out.xp[3], None);
        assert_eq!(out.player_key, frame().player_key);
    }

    #[test]
    fn zero_sigma_is_identity() {
        let out = XpJitter { sigma: 0.0, seed: 9 }.apply(&frame(), 2022, 5);
        assert_eq!(out.xp, frame().xp);
    }
}
```

In `crates/fplopt-backtest/src/lib.rs`, add `pub mod jitter;` next to the other `pub mod` lines (alphabetical order: after `pub mod gw_score;` or wherever `j` sorts).

Check `XpFrame` is exported as `fplopt_optimize::XpFrame`. Run `grep -n "pub use" crates/fplopt-optimize/src/lib.rs`. If it isn't, use the path policies.rs uses (`grep -n "XpFrame" crates/fplopt-backtest/src/policies.rs | head -3`).

- [ ] **Step 2: Run the jitter tests**

Run: `cargo test -j 2 -p fplopt-backtest jitter`
Expected: 4 passed.

- [ ] **Step 3: Write the failing spec tests** (append to `mod tests` in `crates/fplopt-backtest/src/specs.rs`, next to `price_keys_parse_and_label`)

```rust
    #[test]
    fn jitter_keys_parse_and_label() {
        let base = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2";
        let spec = parse_policy_spec(&format!("{base},xp_jitter=0.05,jitter_seed=3")).unwrap();
        let policy = spec.build().unwrap();
        assert_eq!(policy.name(), "optimizer(v1,mh=1,m=2.0,d=0.75,jit=0.05/3)");
        let Policy::Optimizer { jitter, .. } = &policy else { panic!("not an optimizer") };
        assert_eq!(*jitter, Some(crate::jitter::XpJitter { sigma: 0.05, seed: 3 }));
        // the default (0) is dropped and leaves the live planner's name and spec unchanged
        let plain = parse_policy_spec(&format!("{base},xp_jitter=0")).unwrap();
        assert_eq!(plain, parse_policy_spec(base).unwrap());
        assert_eq!(plain.build().unwrap().name(), "optimizer(v1,mh=1,m=2.0,d=0.75)");
        // seed without a wobble, negative sigma, and the key on other policies are refused
        assert!(parse_policy_spec(&format!("{base},jitter_seed=3")).is_err());
        assert!(parse_policy_spec(&format!("{base},xp_jitter=-0.1")).is_err());
        assert!(parse_policy_spec(&format!("{base},xp_jitter=0.05,jitter_seed=-1")).is_err());
        assert!(parse_policy_spec("chipgains:v1:xp_jitter=0.05").is_err());
        // seed defaults to 0
        let s0 = parse_policy_spec(&format!("{base},xp_jitter=0.05")).unwrap().build().unwrap();
        assert!(s0.name().ends_with("jit=0.05/0)"), "{}", s0.name());
    }
```

Run: `cargo test -j 2 -p fplopt-backtest jitter_keys`
Expected: FAIL to compile (`jitter` is not a field of `Policy::Optimizer`).

- [ ] **Step 4: Add the keys in `specs.rs`**

In `params_of`, the `"optimizer"` arm's `keys.extend([...])` list gets two more entries after `("prices", Kind::Prices)`:

```rust
                ("prices", Kind::Prices),
                ("xp_jitter", Kind::Float),
                ("jitter_seed", Kind::Int),
```

In `key_default`, add:

```rust
        ("optimizer", "xp_jitter") => Some(Value::Float(0.0)),
```

In `PolicySpec::build`'s `"optimizer"` arm, after `let tc_rule = ...;`:

```rust
                let sigma = params.remove("xp_jitter").map_or(0.0, Value::f);
                let seed = params.remove("jitter_seed").map(Value::i);
                if sigma < 0.0 {
                    bail!("xp_jitter must be >= 0, got {}", py_float_repr(sigma));
                }
                if seed.is_some_and(|s| s < 0) {
                    bail!("jitter_seed must be >= 0");
                }
                if seed.is_some() && sigma == 0.0 {
                    bail!("jitter_seed needs xp_jitter > 0");
                }
                let jitter = (sigma > 0.0)
                    .then(|| crate::jitter::XpJitter { sigma, seed: seed.unwrap_or(0) as u64 });
```

and change the constructor line to:

```rust
                Policy::Optimizer { xp_model, params: opt, chips, extra, prices, jitter }
```

- [ ] **Step 5: Add the field, the name and the application in `policies.rs`**

The variant:

```rust
    Optimizer {
        xp_model: String,
        params: OptimizerParams,
        chips: bool,
        extra: ChipPolicy2,
        prices: PriceSource,
        /// The noise study's placebo wobble on xP (`crate::jitter`); None for every real policy.
        jitter: Option<crate::jitter::XpJitter>,
    },
```

`validate` (line ~349): change the pattern to `Policy::Optimizer { xp_model, params, chips, extra, prices, .. }`.

`name` (line ~423): change the pattern to `Policy::Optimizer { xp_model, params, chips, extra, prices, jitter }`. After the `if *prices != PriceSource::Fixed { ... }` block, add:

```rust
                if let Some(j) = jitter {
                    parts.push(format!("jit={}/{}", r(j.sigma), j.seed));
                }
```

`optimizer_plans` (line ~529): change the pattern to `let Policy::Optimizer { params, chips, extra, prices, jitter, .. } = self else {`. Replace the `PlanInput::from_context(...)` statement with:

```rust
        let jittered;
        let frame = match jitter {
            Some(j) => {
                jittered = j.apply(ctx.frame, ctx.state.season, ctx.state.gw_index);
                &jittered
            }
            None => ctx.frame,
        };
        let mut problem = PlanInput::from_context(ctx.state, ctx.pool, frame, ctx.rules, &params)?;
```

The other constructors, `policies.rs` line ~648 and `probes.rs` line ~125, each get `jitter: None,` as their last field. Find any remaining ones with `cargo build -j 2 -p fplopt-backtest`; the compiler names each missing field.

- [ ] **Step 6: Run the backtest crate's tests and clippy**

Run: `cargo test -j 2 -p fplopt-backtest`
Expected: all pass, including `jitter_keys_parse_and_label` and the 4 jitter tests.
Run: `cargo clippy -j 2 --workspace --all-targets -- -D warnings` and `cargo fmt --all`.
Expected: no warnings.

- [ ] **Step 7: Run the CLI's tests (spec parsing is exercised there too)**

Run: `cargo test -j 2 -p fplopt-cli`
Expected: all pass. (A `fplopt-highs` githash test can flake in full local workspace runs. That's environmental; don't chase it.)

- [ ] **Step 8: Commit**

```bash
git add crates/fplopt-backtest/src/jitter.rs crates/fplopt-backtest/src/lib.rs crates/fplopt-backtest/src/policies.rs crates/fplopt-backtest/src/specs.rs crates/fplopt-backtest/src/probes.rs
git commit -m "feat(backtest): xp_jitter placebo wobble on the optimizer policy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The study's pure functions (`dev/noise_study.py`)

**Files:**
- Create: `dev/noise_study.py` (pure functions only in this task; the commands come in Task 3)
- Test: `tests/test_noise_study.py`

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("noise_study", REPO / "dev" / "noise_study.py")
ns = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ns
_spec.loader.exec_module(ns)


def test_comparisons():
    c = ns.comparisons()
    assert list(c)[:2] == ["C1", "C2"] and len(c) == 12
    assert c["C1"]["a"] == ns.LIVE + ",prices=forecast,price_k=0.5" and c["C1"]["b"] == ns.LIVE
    assert c["C1"]["seasons"] == "2022" and c["C2"]["seasons"] == "2021,2022"
    assert "hit_margin=3" in c["C2"]["a"] and "hit_margin=2" not in c["C2"]["a"]
    assert c["P1"]["a"].endswith("xp_jitter=0.05,jitter_seed=1")
    assert c["P1"]["b"].endswith("xp_jitter=0.05,jitter_seed=2")
    assert c["P10"]["b"].endswith("jitter_seed=20")


def test_compare_cmd():
    comp = ns.comparisons()["C2"]
    cmd = ns.compare_cmd("bin", comp, 2, Path("o"), jobs=14, experiments=Path("e.csv"))
    assert cmd[:3] == ["bin", "backtest", "compare"]
    j = " ".join(cmd)
    assert "--k 2" in j and "--continuation own" in j and "--reference b" in j
    assert "--seasons 2021,2022" in j and f"--starts {ns.STARTS}" in j
    assert "--family" in cmd and cmd[cmd.index("--family") + 1] == ns.FAMILY
    full = ns.compare_cmd("bin", comp, None, Path("o"), jobs=14, experiments=Path("e.csv"))
    assert "--no-per-decision" in full and "--k" not in full


def _paired():
    # 1 season, 2 starts, 8 GWs: start s0 diff 1 each GW, s1 diff 3 -> per-GW mean 2
    rows = [
        {"season": 2022, "start_id": s, "gw_index": g, "diff": d, "diff_xg": d / 2}
        for s, d in (("s0", 1), ("s1", 3))
        for g in range(1, 9)
    ]
    return pd.DataFrame(rows)


def test_whole_season_scale_and_se():
    mean, se, n = ns.whole_season(_paired(), "diff")
    assert n == 2 and mean == pytest.approx(2 * 38) and se == 0.0
    df = _paired()
    df.loc[(df.start_id == "s0") & (df.gw_index <= 4), "diff"] = 5  # block 1 mean 4, block 2 mean 2
    mean, se, n = ns.whole_season(df, "diff")
    assert mean == pytest.approx(3 * 38)
    assert se == pytest.approx(math.sqrt(2) / math.sqrt(2) * 38)  # sd of (4, 2) = sqrt(2)


def _decisions(k):
    rows = []
    for g, d in ((1, 2), (1 + k, 4), (1 + 2 * k, 6)):
        for s in ("s0", "s1"):
            rows.append({"season": 2022, "start_id": s, "gw_index": g, "diff": d, "diff_xg": None})
    return pd.DataFrame(rows)


def test_per_decision_scale_and_se():
    mean, se, n = ns.per_decision(_decisions(2), 2, "diff")
    assert n == 3 and mean == pytest.approx(4 * 19)  # 38 / k = 19 decisions a season
    assert se == pytest.approx(2 / math.sqrt(3) * 19)
    mean, se, n = ns.per_decision(_decisions(2), 2, "diff_xg")  # all null -> no windows
    assert n == 0 and math.isnan(mean) and math.isnan(se)


def test_z_sd():
    assert ns.z_sd([(1.0, 1.0), (-1.0, 1.0)]) == pytest.approx(math.sqrt(2))
    assert math.isnan(ns.z_sd([(1.0, 0.0)]))


def test_choose():
    rows = [
        {"method": "whole/realized", "se_C1": 9.0, "se_C2": 8.0, "z_sd": 1.0, "minutes": 10},
        {"method": "k1/xg", "se_C1": 4.0, "se_C2": 4.4, "z_sd": 1.1, "minutes": 60},
        {"method": "k2/xg", "se_C1": 4.2, "se_C2": 4.0, "z_sd": 1.2, "minutes": 40},
        {"method": "k4/xg", "se_C1": 3.0, "se_C2": 3.0, "z_sd": 1.6, "minutes": 30},
    ]
    assert ns.choose(rows) == "k2/xg"
    assert ns.choose(rows[:1]) is None
```

Run: `uv run pytest tests/test_noise_study.py -q`
Expected: FAIL (`dev/noise_study.py` doesn't exist).

- [ ] **Step 2: Write the pure functions**

```python
"""Noise study (spec docs/superpowers/specs/2026-11-07-noise-study-design.md): which comparison
method sees 10 points a season.

    uv run python dev/noise_study.py run --out results/noise-study [--jobs 14] [--only C1,P3]
    uv run python dev/noise_study.py report --out results/noise-study

`run` launches `fplopt-rs backtest compare` for every comparison (C1 price, C2 hit margin,
P1-P10 placebo pairs): once without per-decision (the whole-season method's cost) and once per
window length k (1, 2, 4; each arm continues its own plan, states from B). Resumable: a run whose
summary.json exists is skipped. Wall times go to OUT/times.json. `report` puts the 8 methods
(whole season, per decision k = 1, 2, 4; realized and xG-scored points) on one scale, points per
season, and picks the cheapest with SE <= 4.5 on C1 and C2 and placebo z sd <= 1.25.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

LIVE = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2"
STARTS = (
    "template@1,random:3@1,template@10,random:3@10,"
    "template@20,random:3@20,template@29,random:3@29"
)
KS = (1, 2, 4)
SEASON_GWS = 38
BLOCK = 4  # GWs per block of the whole-season method (PLAN §7's window t-test)
TARGET_SE = 4.5  # points a season
Z_SD_MAX = 1.25
SIGMA = 0.05
N_PLACEBO = 10
FAMILY = "noise study"
SCORES = {"realized": "diff", "xg": "diff_xg"}


def comparisons() -> dict[str, dict]:
    out = {
        "C1": {"a": LIVE + ",prices=forecast,price_k=0.5", "b": LIVE, "seasons": "2022"},
        "C2": {"a": LIVE.replace("hit_margin=2", "hit_margin=3"), "b": LIVE, "seasons": "2021,2022"},
    }
    for s in range(1, N_PLACEBO + 1):
        jit = f",xp_jitter={SIGMA},jitter_seed="
        out[f"P{s}"] = {"a": f"{LIVE}{jit}{2 * s - 1}", "b": f"{LIVE}{jit}{2 * s}", "seasons": "2022"}
    return out


def compare_cmd(binary, comp: dict, k: int | None, out: Path, jobs: int, experiments: Path):
    """The compare command: k None = whole season only (--no-per-decision)."""
    cmd = [str(binary), "backtest", "compare", "--a", comp["a"], "--b", comp["b"]]
    cmd += ["--seasons", comp["seasons"], "--starts", STARTS, "--jobs", str(jobs)]
    cmd += ["--family", FAMILY, "--experiments", str(experiments), "--out", str(out)]
    if k is None:
        cmd.append("--no-per-decision")
    else:
        cmd += ["--continuation", "own", "--reference", "b", "--k", str(k), "--stride", str(k)]
    return cmd


def _stats(values: pd.Series, scale: float) -> tuple[float, float, int]:
    n = int(values.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(values.mean()) * scale
    se = float(values.std(ddof=1)) / math.sqrt(n) * scale if n > 1 else float("nan")
    return mean, se, n


def whole_season(paired: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Per-GW A - B averaged over the starts live in it, averaged per non-overlapping BLOCK-GW
    block within each season; mean and SE over blocks, scaled to a season (x SEASON_GWS)."""
    d = paired.dropna(subset=[col])
    per_gw = d.groupby(["season", "gw_index"])[col].mean().reset_index()
    per_gw["block"] = (per_gw["gw_index"] - 1) // BLOCK
    blocks = per_gw.groupby(["season", "block"])[col].mean()
    return _stats(blocks, SEASON_GWS)


def per_decision(dec: pd.DataFrame, k: int, col: str) -> tuple[float, float, int]:
    """Each decision window's A - B (summed over its k GWs) averaged over the starts live in it;
    windows of all seasons pooled; mean and SE over windows x (SEASON_GWS / k)."""
    d = dec.dropna(subset=[col])
    windows = d.groupby(["season", "gw_index"])[col].mean()
    return _stats(windows, SEASON_GWS / k)


def season_clustered_se(dec_or_paired: pd.DataFrame, k: int | None, col: str) -> float:
    """Diagnostic: SE over per-season means (C2 has 2 seasons, so this is rough)."""
    d = dec_or_paired.dropna(subset=[col])
    scale = SEASON_GWS if k is None else SEASON_GWS / k
    per = d.groupby(["season", "gw_index"])[col].mean().groupby("season").mean() * scale
    return float(per.std(ddof=1) / math.sqrt(per.size)) if per.size > 1 else float("nan")


def z_sd(pairs: list[tuple[float, float]]) -> float:
    """sd of z = mean / SE over placebo pairs (honest SEs: about 1)."""
    zs = [m / s for m, s in pairs if s and not math.isnan(s) and s > 0]
    if len(zs) < 2:
        return float("nan")
    mu = sum(zs) / len(zs)
    return math.sqrt(sum((z - mu) ** 2 for z in zs) / (len(zs) - 1))


def choose(rows: list[dict]) -> str | None:
    """The cheapest method (minutes) with se_C1, se_C2 <= TARGET_SE and z_sd <= Z_SD_MAX."""
    ok = [
        r
        for r in rows
        if r["se_C1"] <= TARGET_SE and r["se_C2"] <= TARGET_SE and r["z_sd"] <= Z_SD_MAX
    ]
    return min(ok, key=lambda r: r["minutes"])["method"] if ok else None
```

(Task 3 adds `import argparse`, `import json`, `import subprocess` and `import time` with the commands that use them.)

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_noise_study.py -q`
Expected: 7 passed.
Run: `uv run ruff format dev/noise_study.py tests/test_noise_study.py`, then `uv run ruff check dev/noise_study.py tests/test_noise_study.py`.
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add dev/noise_study.py tests/test_noise_study.py
git commit -m "feat(dev): noise study's scales, placebo check and choice rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `run` and `report` commands, smoke-tested on a real tiny compare

**Files:**
- Modify: `dev/noise_study.py`
- Test: `tests/test_noise_study.py` (one test for `method_rows` on synthetic run dirs)

- [ ] **Step 1: Write the failing test for the report's table**

```python
def _write_run(d: Path, k, diff_a: float):
    d.mkdir(parents=True)
    (d / "summary.json").write_text("{}")
    paired = pd.DataFrame(
        {
            "season": 2022,
            "start_id": ["s0"] * 8 + ["s1"] * 8,
            "gw_index": list(range(1, 9)) * 2,
            "diff": [diff_a, -diff_a] * 8,
            "diff_xg": [diff_a / 2, -diff_a / 2] * 8,
        }
    )
    paired.to_parquet(d / "paired.parquet")
    if k is not None:
        dec = pd.DataFrame(
            {
                "season": 2022,
                "start_id": ["s0", "s1"] * 3,
                "gw_index": [1, 1, 1 + k, 1 + k, 1 + 2 * k, 1 + 2 * k],
                "diff": [1.0, 2.0, 3.0, 1.0, 0.0, 2.0],
                "diff_xg": [1.0, 2.0, 3.0, 1.0, 0.0, 2.0],
            }
        )
        dec.to_parquet(d / "per_decision.parquet")


def test_method_rows(tmp_path):
    times = {}
    for cid in ns.comparisons():
        for k in (None, *ns.KS):
            _write_run(tmp_path / cid / ns.run_name(k), k, 1.0)
            times[f"{cid}/{ns.run_name(k)}"] = 60.0
    (tmp_path / "times.json").write_text(json.dumps(times))
    rows = ns.method_rows(tmp_path)
    assert [r["method"] for r in rows] == [
        "whole/realized", "whole/xg", "k1/realized", "k1/xg",
        "k2/realized", "k2/xg", "k4/realized", "k4/xg",
    ]
    k2 = rows[4]
    assert k2["n_C1"] == 3 and k2["mean_C1"] == pytest.approx(1.5 * 19)
    assert k2["minutes"] == pytest.approx(2.0)  # C1 + C2 run times, 60 s each
```

Add `import json` to the test file's imports.

Run: `uv run pytest tests/test_noise_study.py::test_method_rows -q`
Expected: FAIL (`run_name` / `method_rows` not defined).

- [ ] **Step 2: Add `run_name`, `method_rows`, `run`, `report` and `main`**

Add `import argparse`, `import json`, `import subprocess` and `import time` to the imports, then append:

```python
def run_name(k: int | None) -> str:
    return "full" if k is None else f"k{k}"


def _method_runs():
    yield "whole", None
    for k in KS:
        yield f"k{k}", k


def _measure(run_dir: Path, k: int | None, col: str) -> tuple[float, float, int]:
    if k is None:
        return whole_season(pd.read_parquet(run_dir / "paired.parquet"), col)
    return per_decision(pd.read_parquet(run_dir / "per_decision.parquet"), k, col)


def method_rows(out: Path) -> list[dict]:
    """One row per method: C1 and C2 mean / SE / n per season, placebo z sd and mean z,
    C2's season-clustered SE (diagnostic), minutes (C1 + C2 run times)."""
    times = json.loads((out / "times.json").read_text()) if (out / "times.json").exists() else {}
    comps = comparisons()
    placebos = [c for c in comps if c.startswith("P")]
    rows = []
    for name, k in _method_runs():
        for score, col in SCORES.items():
            row = {"method": f"{name}/{score}"}
            for cid in ("C1", "C2"):
                mean, se, n = _measure(out / cid / run_name(k), k, col)
                row |= {f"mean_{cid}": mean, f"se_{cid}": se, f"n_{cid}": n}
            src = "paired.parquet" if k is None else "per_decision.parquet"
            row["se_C2_season_clustered"] = season_clustered_se(
                pd.read_parquet(out / "C2" / run_name(k) / src), k, col
            )
            pairs = [_measure(out / p / run_name(k), k, col)[:2] for p in placebos]
            row["z_sd"] = z_sd(pairs)
            zs = [m / s for m, s in pairs if s and s > 0]
            row["z_mean"] = sum(zs) / len(zs) if zs else float("nan")
            row["minutes"] = sum(times.get(f"{c}/{run_name(k)}", float("nan")) for c in ("C1", "C2")) / 60
            row["pass"] = (
                row["se_C1"] <= TARGET_SE and row["se_C2"] <= TARGET_SE and row["z_sd"] <= Z_SD_MAX
            )
            rows.append(row)
    return rows


def cmd_run(args) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    times_path = out / "times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else {}
    only = set(args.only.split(",")) if args.only else None
    for cid, comp in comparisons().items():
        if only and cid not in only:
            continue
        for k in (None, *KS):
            run_dir = out / cid / run_name(k)
            if (run_dir / "summary.json").exists():
                continue
            cmd = compare_cmd(args.binary, comp, k, run_dir, args.jobs, args.experiments)
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            with (run_dir / "compare.log").open("w", encoding="utf-8") as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"{cid}/{run_name(k)} failed ({r.returncode}); see {run_dir / 'compare.log'}")
                return r.returncode
            times[f"{cid}/{run_name(k)}"] = time.monotonic() - t0
            times_path.write_text(json.dumps(times, indent=2))
            print(f"{cid}/{run_name(k)}: {times[f'{cid}/{run_name(k)}'] / 60:.1f} min", flush=True)
    return 0


def cmd_report(args) -> int:
    rows = method_rows(args.out)
    chosen = choose(rows)
    payload = {
        "target_se": TARGET_SE,
        "z_sd_max": Z_SD_MAX,
        "n_placebo": N_PLACEBO,
        "sigma": SIGMA,
        "starts": STARTS,
        "methods": rows,
        "chosen": chosen,
    }
    (args.out / "noise.json").write_text(json.dumps(payload, indent=2))
    head = f"{'method':<14} {'C1 mean':>8} {'C1 se':>6} {'C2 mean':>8} {'C2 se':>6} {'C2 se/s':>7}"
    print(head + f" {'z sd':>5} {'z mean':>6} {'min':>6} pass")
    for r in rows:
        print(
            f"{r['method']:<14} {r['mean_C1']:>8.1f} {r['se_C1']:>6.2f} {r['mean_C2']:>8.1f} "
            f"{r['se_C2']:>6.2f} {r['se_C2_season_clustered']:>7.2f} {r['z_sd']:>5.2f} "
            f"{r['z_mean']:>6.2f} {r['minutes']:>6.1f} {'yes' if r['pass'] else 'no'}"
        )
    print(f"chosen: {chosen}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--binary", type=Path, default=Path("target/release/fplopt-rs"))
    r.add_argument("--jobs", type=int, default=14)
    r.add_argument("--experiments", type=Path, default=Path("results/experiments.csv"))
    r.add_argument("--only", help="comma-separated comparison ids, e.g. C1,P3")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raise SystemExit(cmd_run(args) if args.cmd == "run" else cmd_report(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the tests and lint**

Run: `uv run pytest tests/test_noise_study.py -q`
Expected: 8 passed.
Run: `uv run ruff format dev/noise_study.py tests/test_noise_study.py`, then `uv run ruff check dev/noise_study.py tests/test_noise_study.py`.
Expected: clean.

- [ ] **Step 4: Smoke test against the real binary (local, tiny)**

Build: `cargo build -j 2 --release -p fplopt-cli`. Then, with `FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data`, run one real compare with a cheap pair and the placebo keys:

```bash
target/release/fplopt-rs backtest compare --a "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,xp_jitter=0.05,jitter_seed=1" --b "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,xp_jitter=0.05,jitter_seed=2" --seasons 2022 --starts template@29 --continuation own --reference b --k 2 --stride 2 --jobs 2 --family "noise smoke" --experiments <scratch>/exp.csv --out <scratch>/noise-smoke/P1/k2
```

where `<scratch>` is `C:\Users\samin\AppData\Local\Temp\claude\6ff2f6c53ca963aa\50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e\scratchpad`.

Expected:
- the policy labels end with `jit=0.05/1)` and `jit=0.05/2)`;
- `paired.parquet` and `per_decision.parquet` exist with `diff` and `diff_xg`;
- at least one window has `same_decision == False` (the wobble changes decisions).

Then in Python, call `ns.per_decision(pd.read_parquet(...per_decision.parquet), 2, "diff")` and `ns.whole_season(pd.read_parquet(...paired.parquet), "diff")`. Both return finite means.

- [ ] **Step 5: Commit**

```bash
git add dev/noise_study.py tests/test_noise_study.py
git commit -m "feat(dev): noise study run and report commands

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The study on the server (controller; after the user's go-ahead)

- [ ] **Setup:**
  - push the branch;
  - on the server, reuse `~/fpl-pricec`: `git fetch` then `git checkout --detach <tip>`;
  - rebuild with `PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH cargo build --release -p fplopt-cli`;
  - run `uv sync --all-extras` (already done once).
- [ ] **Order:**
  - run C1 and C2 first: `.venv/bin/python dev/noise_study.py run --out results/noise-study --jobs 14 --only C1,C2`. Look at the times, then estimate and tell the user the placebo time;
  - then run the rest: `run --out results/noise-study --jobs 14`;
  - launch it detached (`setsid nohup … &`) with a status file, as for Stage C.
- [ ] **Report:**
  - run `.venv/bin/python dev/noise_study.py report --out results/noise-study`;
  - copy `noise.json`, the `noise study` rows of the server worktree's `results/experiments.csv` and the printed table back to the local worktree.

### Task 5: Write up and adopt (controller)

- [ ] **`docs/research/2026-11-noise-study.md`:**
  - the question and the target;
  - the comparisons and the methods;
  - the methods table;
  - the placebo check;
  - the chosen method, with its cost per comparison;
  - what the C1 numbers say about the price planner (descriptive only: develop 2022/23, no gate);
  - and if nothing passes, the best SE reached and the B options.
- [ ] **PLAN:**
  - a §12 row dated the day of the run;
  - in §7's evaluation design, a "Standard comparison (from 2026-11)" bullet naming the chosen method and its flags.
- [ ] **CLAUDE.md:** one line under Commands pointing at `dev/noise_study.py` and naming the standard comparison flags.
- [ ] **Experiment log:** append the `noise study` rows to `results/experiments.csv`.
- [ ] **Commit and push**, then remove the server worktree (`git worktree remove ~/fpl-pricec`), unless the user wants the early-move experiment next, which can reuse it.
