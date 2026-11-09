# Night Price Model (Gate N) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A player-night model of P(price rises tonight) and P(price falls tonight), from FPL's snapshots before 22:00 UK, and its develop gate (Gate N) on 2022/23 against two baselines.

**Architecture:** A view-only feature builder in `fplopt-features` computes, per player, the state of the price-change counter at the view's cutoff; the same code computes every night of a season from one read for training. `fplopt-models` turns nights and the next morning's prices into labelled rows and fits two binary LightGBM boosters. A `research night-eval` command fits walk-forward, scores 2022/23's nights by log loss against a base-rate and a one-feature baseline with Diebold–Mariano tests, and reports Gate N.

**Tech Stack:** Rust (`crates/fplopt-features`, `crates/fplopt-models`, `crates/fplopt-backtest`, `crates/fplopt-cli`), the pinned LightGBM (`fplopt-lgbm`, objective `binary`, already used by v1's components).

**Spec:** `docs/superpowers/specs/2026-11-09-move-timing-design.md`, section "The nightly model" (read it first). This plan covers the model and Gate N only; policy T (simulator, spec keys, comparison) gets its own plan after Gate N's result. Standing rules: new production code in Rust only; never read 2025/26 (holdout); 2023/24–2024/25 stay unused (the eval command refuses them); `raw/` is append-only. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Worktree:** `C:\Users\samin\Documents\GitHub\fpl-optimizer\.claude\worktrees\v1r`, branch `feat/price-sizing`. Local data: `FPLOPT_DATA_DIR=../../../data` (relative: the session refuses command paths containing "GitHub"); it has snapshots from 2021-04-18 (2020/21's last 65 nights) through 2022/23 and beyond. Plain git commands, one per call; no stash; no heredocs (use the Edit/Write tools). If a link fails with LNK errors, rerun with `-j 2`.

**Verified facts (2026-11-09):**
- `player_snapshot` columns used: `snapshot_at, season, player_key, now_cost, transfers_in_event, transfers_out_event, selected_by_percent, status, chance_of_playing_next_round, source` (see `crates/fplopt-features/src/price.rs` `SNAPSHOT_READ`); where an `fpl` and an `fplcache` row share a time the `fpl` row wins.
- In 2021/22–2022/23 snapshots are at about 01–03, 06–07, 12–13 and 18–19 UK; prices change about 02:30 UK.
- `fplopt_models::gbm::{GbmParams, train, Gbm}`: `GbmParams::default()` is objective `binary`; `Gbm::predict` returns probabilities for it.
- `fplopt_build::common::{uk_date, uk_wall_clock}` give UK dates and UK wall-clock times in UTC (`fplopt-models` depends on `fplopt-build`; `fplopt-features` depends on `fplopt-store` only and must stay that way).
- `crate::eval::diebold_mariano(&[(Micros, f64, f64)], lag)` in `fplopt-backtest` returns pairs including `p_a_better` (as `price_eval.rs` uses it).

---

## File structure

| File | Change |
|---|---|
| `crates/fplopt-features/src/night.rs` | new: `SeasonSnaps`, `features_at`, `night_features`, `COLUMNS` |
| `crates/fplopt-features/src/price.rs` | `status_code` → `pub(crate)` |
| `crates/fplopt-features/src/lib.rs` | `pub mod night;`, `night_features` in `FRAMES` |
| `crates/fplopt-models/src/night.rs` | new: `nights`, `night_labels`, `training_rows`, `NightParams`, `fit_night`, `predict_night` |
| `crates/fplopt-models/src/lib.rs` | `pub mod night;` |
| `crates/fplopt-backtest/src/night_eval.rs` | new: `evaluate_nights`, `night_metrics`, `rows_table` |
| `crates/fplopt-backtest/src/lib.rs` | `pub mod night_eval;` |
| `crates/fplopt-cli/src/research.rs` | `research night-eval` |
| `CLAUDE.md` | one Commands line |

---

### Task 1: Night features (view-only)

**Files:**
- Create: `crates/fplopt-features/src/night.rs`
- Modify: `crates/fplopt-features/src/price.rs` (`fn status_code` → `pub(crate) fn status_code`)
- Modify: `crates/fplopt-features/src/lib.rs` (`pub mod night;`; add `("night_features", night::night_features),` at the end of `FRAMES`, and mention it in `FRAMES`' doc comment next to `price_features`)

- [ ] **Step 1: Write the failing tests** (bottom of the new `night.rs`)

The fixture: GW1 deadline 2021-08-13 17:30, GW2 2021-08-21 10:00 UTC; player 1 rose 75 → 76 between the Aug 14 and Aug 18 snapshots; an `fplcache` row at Aug 20 12:00 shares the time with an `fpl` row and must lose; rows on Aug 22 are after the cutoff.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_store::DataStore;
    use std::collections::HashMap;

    const MINUTE: Micros = 60_000_000;
    /// 2021-08-01 00:00 UTC.
    const AUG_1: Micros = 1_627_776_000_000_000;

    fn aug(day: i64, hour: i64, minute: i64) -> Micros {
        AUG_1 + (day - 1) * DAY + hour * HOUR + minute * MINUTE
    }

    /// (at, source, player, cost, in, out, own, status, chance)
    type Row = (Micros, &'static str, i64, i64, i64, i64, f64, &'static str, Option<i64>);

    fn store(rows: &[Row]) -> DataStore {
        let col = |f: &dyn Fn(&Row) -> i64| Column::I64(rows.iter().map(f).collect());
        let snapshots = Table::new()
            .with("snapshot_at", Column::Ts(rows.iter().map(|r| r.0).collect()))
            .with("source", Column::Str(rows.iter().map(|r| r.1.to_string()).collect()))
            .with("season", col(&|_| 2021))
            .with("player_key", col(&|r| r.2))
            .with("now_cost", col(&|r| r.3))
            .with("transfers_in_event", col(&|r| r.4))
            .with("transfers_out_event", col(&|r| r.5))
            .with("selected_by_percent", Column::OptF64(rows.iter().map(|r| Some(r.6)).collect()))
            .with("status", Column::Str(rows.iter().map(|r| r.7.to_string()).collect()))
            .with(
                "chance_of_playing_next_round",
                Column::OptI64(rows.iter().map(|r| r.8).collect()),
            )
            .with("available_at", Column::Ts(rows.iter().map(|r| r.0).collect()));
        let gameweek = Table::new()
            .with("season", Column::I64(vec![2021; 2]))
            .with("gw", Column::I64(vec![1, 2]))
            .with("gw_index", Column::I64(vec![1, 2]))
            .with("deadline_time", Column::Ts(vec![aug(13, 17, 30), aug(21, 10, 0)]))
            .with("available_at", Column::Ts(vec![0; 2]));
        DataStore::from_tables(HashMap::from([
            ("gameweek".to_string(), gameweek),
            ("player_snapshot".to_string(), snapshots),
        ]))
        .unwrap()
    }

    fn rows() -> Vec<Row> {
        vec![
            (aug(14, 0, 0), "fpl", 1, 75, 1_000, 0, 10.0, "a", None),
            (aug(18, 0, 0), "fpl", 1, 76, 5_000, 0, 12.0, "a", None),
            (aug(20, 12, 0), "fplcache", 1, 80, 0, 0, 1.0, "i", Some(0)),
            (aug(20, 12, 0), "fpl", 1, 76, 9_000, 1_000, 13.9, "a", None),
            (aug(14, 0, 0), "fpl", 2, 50, 0, 500, 5.0, "a", None),
            (aug(18, 0, 0), "fpl", 2, 50, 0, 1_000, 5.0, "a", None),
            (aug(20, 12, 0), "fpl", 2, 50, 0, 2_000, 5.0, "d", Some(75)),
            (aug(22, 0, 0), "fpl", 1, 99, 50_000, 0, 40.0, "a", None),
            (aug(22, 0, 0), "fpl", 2, 99, 50_000, 0, 40.0, "a", None),
        ]
    }

    fn close(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-9
    }

    #[test]
    fn night_features_on_an_early_view_before_gw2() {
        let s = store(&rows());
        let cutoff = aug(20, 21, 0);
        let out = night_features(&s.as_of_early(aug(21, 10, 0), cutoff).unwrap()).unwrap();
        assert_eq!(out.i64("player_key").unwrap(), &[1, 2]);
        let v = |c: &str| out.f64(c).unwrap().to_vec();
        let expect: &[(&str, [f64; 2])] = &[
            ("price", [76.0, 50.0]),
            ("net_event", [8_000.0, -2_000.0]),
            ("net_since_change", [7_000.0, -2_000.0]),
            ("net_24h", [3_000.0, -1_000.0]),
            ("own", [13.9, 5.0]),
            ("net_per_own", [7_000.0 / 14.0, -2_000.0 / 5.1]),
            ("nights_since_change", [2.875, f64::NAN]),
            ("rises_gw", [1.0, 0.0]),
            ("falls_gw", [0.0, 0.0]),
            ("status_code", [0.0, 1.0]),
            ("chance_next", [f64::NAN, 75.0]),
            ("red_lock", [0.0, 0.0]),
            ("hours_to_deadline", [13.0, 13.0]),
            ("gw_index", [2.0, 2.0]),
        ];
        assert_eq!(out.width(), COLUMNS.len() + 1);
        for (c, want) in expect {
            let got = v(c);
            assert!(close(got[0], want[0]) && close(got[1], want[1]), "{c}: {got:?} vs {want:?}");
        }
    }

    #[test]
    fn features_at_equals_night_features_on_the_view() {
        let s = store(&rows());
        let late = s.as_of(aug(23, 0, 0));
        let season = SeasonSnaps::read(&late, 2021).unwrap();
        for cutoff in [aug(15, 21, 0), aug(18, 21, 0), aug(20, 21, 0)] {
            let view = s.as_of_early(aug(21, 10, 0), cutoff).unwrap();
            let a = night_features(&view).unwrap();
            let b = features_at(&season, cutoff, aug(21, 10, 0), aug(13, 17, 30), 2);
            assert_eq!(a.i64("player_key").unwrap(), b.i64("player_key").unwrap());
            for c in COLUMNS {
                let (x, y) = (a.f64(c).unwrap(), b.f64(c).unwrap());
                assert!(x.iter().zip(y).all(|(p, q)| close(*p, *q)), "{c} at {cutoff}");
            }
        }
    }

    #[test]
    fn red_lock_marks_a_red_flag_cleared_straight_to_available_within_a_week() {
        let mut r = rows();
        // player 2: injured on Aug 14, available again on Aug 18 (lock); Aug 20 'd' after it
        r[4].7 = "i";
        r[5].7 = "a";
        let s = store(&r);
        let out = night_features(&s.as_of_early(aug(21, 10, 0), aug(20, 21, 0)).unwrap()).unwrap();
        assert_eq!(out.f64("red_lock").unwrap(), &[0.0, 1.0]);
        // more than a week after Aug 18 the lock has passed (the fixture has no GW3, so call
        // features_at directly rather than through a view)
        let season = SeasonSnaps::read(&s.as_of(aug(23, 0, 0)), 2021).unwrap();
        let later = features_at(&season, aug(25, 21, 0), aug(28, 10, 0), aug(21, 10, 0), 3);
        assert_eq!(later.f64("red_lock").unwrap()[1], 0.0);
    }

    #[test]
    fn a_view_before_any_snapshot_has_no_rows() {
        let s = store(&rows());
        let out = night_features(&s.as_of_early(aug(21, 10, 0), aug(13, 21, 0)).unwrap()).unwrap();
        assert_eq!(out.height(), 0);
        assert_eq!(out.width(), COLUMNS.len() + 1);
    }
}
```
(Fixture arithmetic, player 1 at Aug 20 21:00: GW2 starts at the GW1 deadline Aug 13 17:30; the change is at Aug 18 00:00 (75 → 76); the snapshot before it (Aug 14) had net 1,000, the newest has 9,000 − 1,000 = 8,000, so since the change 7,000; the newest snapshot at or before Aug 19 21:00 is Aug 18 (5,000), so 24 h: 3,000; nights since the change (Aug 20 21:00 − Aug 18 00:00) = 2.875 days; hours to the deadline 13. Player 2: no change, so since the change = since the GW start = −2,000. Remove the leftover `late`/`if let` lines in `red_lock_…` if clippy flags them; they only document why the helper is called directly.)

- [ ] **Step 2: Run them to make sure they fail**

Run: `cargo test -p fplopt-features night`
Expected: compile errors (module `night` items missing).

- [ ] **Step 3: Implement `night.rs`** (top of the file)

```rust
//! Night price features (move timing, `docs/superpowers/specs/2026-11-09-move-timing-design.md`):
//! per player in the season's newest snapshot before the view's cutoff (a night's decision
//! moment, 22:00 UK), the state of FPL's price-change counter. `night_features(view)` is the
//! view-only builder (`FRAMES`, so `check leakage` covers it); `features_at` computes any
//! cutoff from one `SeasonSnaps` read, so training reads the snapshots once a season. Both
//! use only snapshots strictly before the cutoff.

use std::collections::BTreeMap;

use anyhow::Result;
use fplopt_store::{AsOfView, Column, Micros, Table};

use crate::baseline::target;
use crate::price::status_code;

const DAY: Micros = 86_400_000_000;
const HOUR: Micros = 3_600_000_000;
/// A red flag cleared straight to available locks the price for a week.
const LOCK: Micros = 7 * DAY;

/// The feature columns after `player_key`, in output order (all F64, NaN where undefined).
pub const COLUMNS: &[&str] = &[
    "price",
    "net_event",
    "net_since_change",
    "net_24h",
    "own",
    "net_per_own",
    "nights_since_change",
    "rises_gw",
    "falls_gw",
    "status_code",
    "chance_next",
    "red_lock",
    "hours_to_deadline",
    "gw_index",
];

const READ: &[&str] = &[
    "snapshot_at",
    "season",
    "player_key",
    "now_cost",
    "transfers_in_event",
    "transfers_out_event",
    "selected_by_percent",
    "status",
    "chance_of_playing_next_round",
    "source",
];

/// One snapshot of a player (one per time: an `fpl` row wins a tie with `fplcache`).
#[derive(Clone, Debug)]
pub struct NightSnap {
    pub at: Micros,
    pub cost: f64,
    pub net: f64,
    pub own: Option<f64>,
    pub status: Option<String>,
    pub chance: Option<f64>,
}

/// A season's player snapshots as one read sees them: per player, by time.
pub struct SeasonSnaps {
    pub by_player: BTreeMap<i64, Vec<NightSnap>>,
}

impl SeasonSnaps {
    /// `season`'s snapshots visible in `view` (available_at < its cutoff).
    pub fn read(view: &AsOfView, season: i64) -> Result<SeasonSnaps> {
        let t = view.table("player_snapshot", Some(READ))?;
        let seasons = t.opt_i64("season")?;
        let players = t.opt_i64("player_key")?;
        let times = t.opt_i64("snapshot_at")?;
        let sources = t.opt_str("source")?;
        let costs = t.opt_f64("now_cost")?;
        let ins = t.opt_f64("transfers_in_event")?;
        let outs = t.opt_f64("transfers_out_event")?;
        let owns = t.opt_f64("selected_by_percent")?;
        let statuses = t.opt_str("status")?;
        let chances = t.opt_f64("chance_of_playing_next_round")?;
        let mut raw: BTreeMap<i64, Vec<(Micros, bool, NightSnap)>> = BTreeMap::new();
        for i in 0..t.height() {
            let (Some(player), Some(at)) = (players[i], times[i]) else { continue };
            if seasons[i] != Some(season) {
                continue;
            }
            let net = match (ins[i], outs[i]) {
                (Some(a), Some(b)) => a - b,
                _ => f64::NAN,
            };
            let snap = NightSnap {
                at,
                cost: costs[i].unwrap_or(f64::NAN),
                net,
                own: owns[i],
                status: statuses[i].clone(),
                chance: chances[i],
            };
            raw.entry(player).or_default().push((at, sources[i].as_deref() == Some("fpl"), snap));
        }
        let mut by_player = BTreeMap::new();
        for (player, mut rows) in raw {
            rows.sort_by_key(|r| (r.0, r.1));
            let mut kept: Vec<NightSnap> = Vec::with_capacity(rows.len());
            for (at, _, snap) in rows {
                // sorted with the 'fpl' row last at a tie: a later row at the same time wins
                if kept.last().is_some_and(|s| s.at == at) {
                    kept.pop();
                }
                kept.push(snap);
            }
            by_player.insert(player, kept);
        }
        Ok(SeasonSnaps { by_player })
    }

    /// The newest snapshot time strictly before `cutoff` (None before the first).
    fn newest_before(&self, cutoff: Micros) -> Option<Micros> {
        self.by_player
            .values()
            .filter_map(|rows| {
                let k = rows.partition_point(|s| s.at < cutoff);
                (k > 0).then(|| rows[k - 1].at)
            })
            .max()
    }
}

/// The latest deadline of `season` at or before `cutoff` (the current GW's start: FPL resets
/// `transfers_in_event` there), i64::MIN before the first.
pub fn gw_start(view: &AsOfView, season: i64, cutoff: Micros) -> Result<Micros> {
    let g = view.table("gameweek", Some(&["season", "deadline_time"]))?;
    let seasons = g.i64("season")?;
    let deadlines = g.opt_i64("deadline_time")?;
    Ok((0..g.height())
        .filter(|&i| seasons[i] == season)
        .filter_map(|i| deadlines[i])
        .filter(|&d| d <= cutoff)
        .max()
        .unwrap_or(i64::MIN))
}

/// One player's first 12 feature columns (`COLUMNS[..12]`, price … red_lock) from his
/// snapshots before `cutoff` (`rows` sorted by time, all before the cutoff, non-empty).
fn player_row(rows: &[NightSnap], cutoff: Micros, gw_start: Micros) -> [f64; 12] {
    let last = &rows[rows.len() - 1];
    let since_gw = |s: &NightSnap| if s.at >= gw_start { s.net } else { 0.0 };
    let changes: Vec<usize> = (1..rows.len())
        .filter(|&i| {
            let (a, b) = (rows[i - 1].cost, rows[i].cost);
            a.is_finite() && b.is_finite() && a != b
        })
        .collect();
    let net_since_change = match changes.last() {
        Some(&c) if rows[c].at >= gw_start => last.net - since_gw(&rows[c - 1]),
        _ => last.net,
    };
    let k = rows.partition_point(|s| s.at <= cutoff - DAY);
    let net_24h = if k > 0 { last.net - since_gw(&rows[k - 1]) } else { last.net };
    let nights_since_change =
        changes.last().map_or(f64::NAN, |&c| (cutoff - rows[c].at) as f64 / DAY as f64);
    let in_gw = |up: bool| {
        changes
            .iter()
            .filter(|&&c| rows[c].at >= gw_start && (rows[c].cost > rows[c - 1].cost) == up)
            .count() as f64
    };
    let red = |s: &NightSnap| matches!(s.status.as_deref(), Some("i" | "s" | "u"));
    let locked = (1..rows.len()).any(|i| {
        rows[i].at >= cutoff - LOCK && rows[i].status.as_deref() == Some("a") && red(&rows[i - 1])
    });
    let own = last.own.unwrap_or(f64::NAN);
    [
        last.cost,
        last.net,
        net_since_change,
        net_24h,
        own,
        net_since_change / (own + 0.1),
        nights_since_change,
        in_gw(true),
        in_gw(false),
        status_code(last.status.as_deref()),
        last.chance.unwrap_or(f64::NAN),
        if locked { 1.0 } else { 0.0 },
    ]
}

/// Features at `cutoff` for the GW whose deadline is `deadline` (index `gw_index`, started at
/// `gw_start`): one row per player in the newest snapshot before the cutoff, sorted by
/// player_key; empty before the season's first snapshot.
pub fn features_at(
    season: &SeasonSnaps,
    cutoff: Micros,
    deadline: Micros,
    gw_start: Micros,
    gw_index: i64,
) -> Table {
    let newest = season.newest_before(cutoff);
    let mut keys = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); COLUMNS.len()];
    if let Some(newest) = newest {
        for (&player, rows) in &season.by_player {
            let k = rows.partition_point(|s| s.at < cutoff);
            if k == 0 || rows[k - 1].at != newest {
                continue;
            }
            keys.push(player);
            let r = player_row(&rows[..k], cutoff, gw_start);
            for (j, v) in r.into_iter().enumerate() {
                cols[j].push(v);
            }
            cols[12].push((deadline - cutoff) as f64 / HOUR as f64);
            cols[13].push(gw_index as f64);
        }
    }
    let mut t = Table::new().with("player_key", Column::I64(keys));
    for (name, values) in COLUMNS.iter().zip(cols) {
        t = t.with(name, Column::F64(values));
    }
    t
}

/// The night features at the view's cutoff, for the GW of its deadline.
pub fn night_features(view: &AsOfView) -> Result<Table> {
    let (season, _, gw_index) = target(view)?;
    let snaps = SeasonSnaps::read(view, season)?;
    let start = gw_start(view, season, view.cutoff())?;
    Ok(features_at(&snaps, view.cutoff(), view.deadline(), start, gw_index))
}
```
The order of `player_row`'s array must match `COLUMNS[..12]`. `hours_to_deadline` and `gw_index` are columns 12 and 13. If `target(view)` needs tables the fixture lacks (it reads `gameweek` only), report it.

- [ ] **Step 4: Run the tests and the architecture checks**

Run: `cargo test -p fplopt-features`
Expected: all pass (the new four and the existing ones).
Run: `cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all`.

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-features/src/night.rs crates/fplopt-features/src/price.rs crates/fplopt-features/src/lib.rs
```
```bash
git commit -m "feat(features): night price features (state of the price counter at a cutoff)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Nights, labels, training rows, the model

**Files:**
- Create: `crates/fplopt-models/src/night.rs`
- Modify: `crates/fplopt-models/src/lib.rs` (`pub mod night;`)

- [ ] **Step 1: Write the failing tests** (bottom of `night.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_build::common::uk_wall_clock;
    use fplopt_store::DataStore;
    use std::collections::HashMap;

    const DAY: Micros = 86_400_000_000;

    fn date(m: u32, d: u32) -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd_opt(2021, m, d).unwrap()
    }

    /// GW1 deadline Aug 13 17:30 UTC, GW2 Aug 21 10:00 UTC; player 1 costs 75 until a
    /// snapshot at 06:30 UK on Aug 16 shows 76; player 2 stays 50. Snapshots at 19:00 and
    /// 06:30 UK each day Aug 13–22.
    fn store() -> DataStore {
        let mut rows: Vec<(Micros, i64, i64)> = Vec::new();
        for d in 13..=22 {
            for (h, m) in [(6, 30), (19, 0)] {
                let at = uk_wall_clock(date(8, d), h) + m * 60_000_000;
                let up = at >= uk_wall_clock(date(8, 16), 6);
                rows.push((at, 1, if up { 76 } else { 75 }));
                rows.push((at, 2, 50));
            }
        }
        let n = rows.len();
        let snapshots = Table::new()
            .with("snapshot_at", Column::Ts(rows.iter().map(|r| r.0).collect()))
            .with("source", Column::Str(vec!["fpl".to_string(); n]))
            .with("season", Column::I64(vec![2021; n]))
            .with("player_key", Column::I64(rows.iter().map(|r| r.1).collect()))
            .with("now_cost", Column::I64(rows.iter().map(|r| r.2).collect()))
            .with("transfers_in_event", Column::I64(vec![0; n]))
            .with("transfers_out_event", Column::I64(vec![0; n]))
            .with("selected_by_percent", Column::OptF64(vec![Some(5.0); n]))
            .with("status", Column::Str(vec!["a".to_string(); n]))
            .with("chance_of_playing_next_round", Column::OptI64(vec![None; n]))
            .with("available_at", Column::Ts(rows.iter().map(|r| r.0).collect()));
        let gw1 = uk_wall_clock(date(8, 13), 18) + 30 * 60_000_000;
        let gw2 = uk_wall_clock(date(8, 21), 11);
        let gameweek = Table::new()
            .with("season", Column::I64(vec![2021; 2]))
            .with("gw", Column::I64(vec![1, 2]))
            .with("gw_index", Column::I64(vec![1, 2]))
            .with("deadline_time", Column::Ts(vec![gw1, gw2]))
            .with("available_at", Column::Ts(vec![0; 2]));
        DataStore::from_tables(HashMap::from([
            ("gameweek".to_string(), gameweek),
            ("player_snapshot".to_string(), snapshots),
        ]))
        .unwrap()
    }

    #[test]
    fn nights_run_from_the_first_deadlines_date_to_the_night_before_the_last_deadline() {
        let s = store();
        let view = s.as_of(uk_wall_clock(date(8, 25), 0));
        let n = nights(&view, 2021).unwrap();
        // Aug 13 22:00 UK is after GW1's deadline and before GW2's; the last night with a
        // later deadline is Aug 20.
        assert_eq!(n.first().unwrap().cutoff, uk_wall_clock(date(8, 13), 22));
        assert_eq!(n.last().unwrap().cutoff, uk_wall_clock(date(8, 20), 22));
        assert_eq!(n.len(), 8);
        assert!(n.iter().all(|x| x.deadline == uk_wall_clock(date(8, 21), 11)));
        assert!(n.iter().all(|x| x.gw_index == 2));
        let first = &n[0];
        assert_eq!(first.label_from, uk_wall_clock(date(8, 14), 4));
        assert_eq!(first.label_to, uk_wall_clock(date(8, 14), 12));
    }

    #[test]
    fn labels_are_the_next_mornings_price_minus_the_evenings() {
        let s = store();
        let rows = training_rows(&s, &[2021], uk_wall_clock(date(8, 25), 0)).unwrap();
        let nights = rows.opt_i64("night").unwrap();
        let keys = rows.i64("player_key").unwrap();
        let dp = rows.f64("dp").unwrap();
        let rise = rows.f64("rise").unwrap();
        let fall = rows.f64("fall").unwrap();
        let aug15 = uk_wall_clock(date(8, 15), 22);
        for i in 0..rows.height() {
            let expect = if keys[i] == 1 && nights[i] == Some(aug15) { 1.0 } else { 0.0 };
            assert_eq!(dp[i], expect, "row {i}");
            assert_eq!(rise[i], expect);
            assert_eq!(fall[i], 0.0);
        }
        // 8 nights × 2 players
        assert_eq!(rows.height(), 16);
    }

    #[test]
    fn training_rows_see_nothing_at_or_after_the_fit_cutoff() {
        let s = store();
        // fit at Aug 17 00:00 UK: nights whose label window ends before it (Aug 13..15)
        let cut = uk_wall_clock(date(8, 17), 0);
        let rows = training_rows(&s, &[2021], cut).unwrap();
        let nights = rows.opt_i64("night").unwrap();
        assert!(nights.iter().all(|n| n.unwrap() + DAY / 2 <= cut));
        assert_eq!(rows.height(), 3 * 2);
    }

    #[test]
    fn training_rows_refuse_the_holdout() {
        let s = store();
        assert!(training_rows(&s, &[2025], uk_wall_clock(date(8, 25), 0)).is_err());
    }

    #[test]
    fn a_fit_predicts_probabilities_for_every_row() {
        let s = store();
        let rows = training_rows(&s, &[2021], uk_wall_clock(date(8, 25), 0)).unwrap();
        let params = NightParams { gbm: GbmParams { num_boost_round: 5, min_data_in_leaf: 1, ..NightParams::default().gbm } };
        let fit = fit_night(&rows, &params).unwrap();
        let p = predict_night(&fit, &rows).unwrap();
        assert_eq!(p.height(), rows.height());
        for c in ["p_rise", "p_fall"] {
            assert!(p.f64(c).unwrap().iter().all(|v| (0.0..=1.0).contains(v)), "{c}");
        }
    }
}
```
(`training_rows` with a single rise: the boosters' probabilities just need to be valid; the fit's quality is Gate N's job.)

- [ ] **Step 2: Run them to make sure they fail**

Run: `cargo test -p fplopt-models night`
Expected: compile errors.

- [ ] **Step 3: Implement `night.rs`**

```rust
//! The night price model (move timing, `docs/superpowers/specs/2026-11-09-move-timing-design.md`):
//! P(rise tonight) and P(fall tonight) per player from `fplopt_features::night` at 22:00 UK,
//! labelled by the next morning's price (the first snapshot from 04:00 to 12:00 UK; FPL
//! changes prices at about 02:30). Two binary boosters, fitted on the nights of earlier
//! seasons only; the holdout season is refused.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Result, bail};
use fplopt_build::common::{uk_date, uk_wall_clock};
use fplopt_features::night::{COLUMNS, SeasonSnaps, features_at, gw_start};
use fplopt_store::{AsOfView, Column, DataStore, Micros, Table};

use crate::gbm::{Gbm, GbmParams, train};
use crate::price::refuse_holdout;

/// The decision moment and the label window, UK wall-clock hours.
pub const NIGHT_HOUR: u32 = 22;
pub const MORNING_FROM_HOUR: u32 = 4;
pub const MORNING_TO_HOUR: u32 = 12;

/// The model's feature columns (all of the night features).
pub const FEATURES: &[&str] = COLUMNS;

/// One night of a season: the decision moment, the GW it is before and the label window.
#[derive(Clone, Debug, PartialEq)]
pub struct Night {
    pub season: i64,
    pub cutoff: Micros,
    pub deadline: Micros,
    pub gw_index: i64,
    pub label_from: Micros,
    pub label_to: Micros,
}

/// Every night of `season` in the view's gameweek table: each UK date from the date of the
/// season's first deadline, at 22:00 UK, that is before a later deadline of the season (the
/// GW it decides for is that next deadline's).
pub fn nights(view: &AsOfView, season: i64) -> Result<Vec<Night>> {
    let g = view.table("gameweek", Some(&["season", "gw_index", "deadline_time"]))?;
    let seasons = g.i64("season")?;
    let indices = g.i64("gw_index")?;
    let deadlines = g.opt_i64("deadline_time")?;
    let mut gws: Vec<(Micros, i64)> = (0..g.height())
        .filter(|&i| seasons[i] == season)
        .filter_map(|i| deadlines[i].map(|d| (d, indices[i])))
        .collect();
    gws.sort();
    gws.dedup();
    let (Some(first), Some(last)) = (gws.first(), gws.last()) else { return Ok(vec![]) };
    let mut out = Vec::new();
    let mut day = uk_date(first.0);
    while uk_wall_clock(day, NIGHT_HOUR) < last.0 {
        let cutoff = uk_wall_clock(day, NIGHT_HOUR);
        if let Some(&(deadline, gw_index)) = gws.iter().find(|(d, _)| *d > cutoff) {
            let next = day.succ_opt().expect("a next day");
            out.push(Night {
                season,
                cutoff,
                deadline,
                gw_index,
                label_from: uk_wall_clock(next, MORNING_FROM_HOUR),
                label_to: uk_wall_clock(next, MORNING_TO_HOUR),
            });
        }
        day = day.succ_opt().expect("a next day");
    }
    Ok(out)
}

/// The labelled rows of `seasons`' nights whose label window ends at or before `cutoff`:
/// `season`, `night` (Ts), `player_key`, `FEATURES`, `dp` (the first snapshot price in the
/// label window minus the newest before the night; NaN without one), `rise`, `fall` (1/0,
/// NaN with dp). Features see only snapshots before each night (`features_at`, equal to
/// `night_features` on that night's view), labels only rows before `cutoff`. Sorted by
/// (season, night, player_key).
pub fn training_rows(store: &DataStore, seasons: &[i64], cutoff: Micros) -> Result<Table> {
    refuse_holdout(seasons)?;
    let view = store.as_of(cutoff);
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); FEATURES.len()];
    let (mut ss, mut ns, mut ks, mut dps) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &season in seasons.iter().collect::<BTreeSet<_>>() {
        let snaps = SeasonSnaps::read(&view, season)?;
        for night in nights(&view, season)? {
            if night.label_to > cutoff {
                continue;
            }
            let start = gw_start(&view, season, night.cutoff)?;
            let f = features_at(&snaps, night.cutoff, night.deadline, start, night.gw_index);
            let keys = f.i64("player_key")?;
            let price = f.f64("price")?;
            for (i, &key) in keys.iter().enumerate() {
                let morning = snaps.by_player.get(&key).and_then(|rows| {
                    rows.iter()
                        .find(|s| s.at >= night.label_from && s.at < night.label_to)
                        .map(|s| s.cost)
                });
                ss.push(season);
                ns.push(night.cutoff);
                ks.push(key);
                dps.push(morning.map_or(f64::NAN, |m| m - price[i]));
                for (j, c) in FEATURES.iter().enumerate() {
                    cols[j].push(f.f64(c)?[i]);
                }
            }
        }
    }
    let flag = |up: bool| -> Vec<f64> {
        dps.iter()
            .map(|&d: &f64| if d.is_nan() { f64::NAN } else if (d > 0.0) == up && d != 0.0 { 1.0 } else { 0.0 })
            .collect()
    };
    let (rise, fall) = (flag(true), flag(false));
    let mut t = Table::new()
        .with("season", Column::I64(ss))
        .with("night", Column::Ts(ns))
        .with("player_key", Column::I64(ks));
    for (name, values) in FEATURES.iter().zip(cols) {
        t = t.with(name, Column::F64(values));
    }
    Ok(t.with("dp", Column::F64(dps)).with("rise", Column::F64(rise)).with("fall", Column::F64(fall)))
}

/// The boosters' learning parameters (one setting, no grid: Gate N is pre-registered).
#[derive(Clone, Debug, PartialEq)]
pub struct NightParams {
    pub gbm: GbmParams,
}

impl Default for NightParams {
    fn default() -> Self {
        NightParams {
            gbm: GbmParams {
                objective: "binary",
                num_boost_round: 300,
                learning_rate: 0.05,
                num_leaves: 31,
                min_data_in_leaf: 200,
                feature_fraction: 0.9,
                lambda_l2: 1.0,
            },
        }
    }
}

/// The two boosters and the seasons fitted on.
pub struct NightFit {
    pub rise: Gbm,
    pub fall: Gbm,
    pub seasons: Vec<i64>,
}

/// One binary booster per target on the rows with a finite label, in the rows' order, on
/// `features`; the holdout season is refused.
pub fn fit_on(rows: &Table, features: &[&'static str], params: &NightParams) -> Result<NightFit> {
    let seasons: Vec<i64> =
        rows.i64("season")?.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    refuse_holdout(&seasons)?;
    let boost = |label: &str| -> Result<Gbm> {
        let y = rows.f64(label)?;
        let keep: Vec<usize> = (0..rows.height()).filter(|&i| y[i].is_finite()).collect();
        if keep.is_empty() {
            bail!("no training row has a finite {label}");
        }
        train(&rows.take(&keep), features, label, None, &params.gbm)
    };
    Ok(NightFit { rise: boost("rise")?, fall: boost("fall")?, seasons })
}

/// The model: both boosters on every night feature.
pub fn fit_night(rows: &Table, params: &NightParams) -> Result<NightFit> {
    fit_on(rows, FEATURES, params)
}

/// `player_key, p_rise, p_fall` for `features`' rows, in order.
pub fn predict_night(fit: &NightFit, features: &Table) -> Result<Table> {
    Ok(Table::new()
        .with("player_key", Column::I64(features.i64("player_key")?.to_vec()))
        .with("p_rise", Column::F64(fit.rise.predict(features)?))
        .with("p_fall", Column::F64(fit.fall.predict(features)?)))
}
```
Check `Table::take`, `Column::Ts`, `opt_i64` on a Ts column and `refuse_holdout`'s visibility (`pub` in `price.rs`) against the real code, and simplify `flag` (rise = dp > 0, fall = dp < 0). `HashMap` may end up unused; drop it.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fplopt-models night`, then `cargo test -p fplopt-models`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-models/src/night.rs crates/fplopt-models/src/lib.rs
```
```bash
git commit -m "feat(models): night price model (nights, labels, training rows, boosters)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `research night-eval` and Gate N

**Files:**
- Create: `crates/fplopt-backtest/src/night_eval.rs`
- Modify: `crates/fplopt-backtest/src/lib.rs` (`pub mod night_eval;`)
- Modify: `crates/fplopt-cli/src/research.rs` (`ResearchJob::NightEval(NightEvalArgs)`, dispatch, `fn night_eval`, an args test)

- [ ] **Step 1: Write the failing tests** (bottom of `night_eval.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn row(night: Micros, truth_rise: f64, p: f64, b0: f64, b1: f64) -> NightRow {
        NightRow {
            season: 2022,
            night,
            player_key: night,
            rise: truth_rise,
            fall: 0.0,
            p_rise: p,
            p_fall: 0.02,
            b0_rise: b0,
            b0_fall: 0.02,
            b1_rise: b1,
            b1_fall: 0.02,
        }
    }

    #[test]
    fn log_loss_is_the_mean_binary_cross_entropy() {
        let l = log_loss(&[(1.0, 0.8), (0.0, 0.2)]);
        assert!((l - (-(0.8f64).ln())).abs() < 1e-12);
        // clipped: a certain wrong prediction is finite
        assert!(log_loss(&[(1.0, 0.0)]).is_finite());
    }

    #[test]
    fn seasons_refused_outside_develop() {
        assert!(check_seasons(&[2022], 2020).is_ok());
        assert!(check_seasons(&[2021], 2021).is_err()); // not after --first-fit
        assert!(check_seasons(&[2023], 2020).is_err()); // kept clean
        assert!(check_seasons(&[2025], 2020).is_err()); // holdout
        assert!(check_seasons(&[2022], 2019).is_err()); // no snapshots before 2020/21
    }

    #[test]
    fn the_gate_needs_both_targets_to_beat_both_baselines() {
        // the model is better on every night: the rise half passes; the fall probabilities
        // are equal across arms (0.02), so the fall half fails and so does the gate (a DM
        // test on identical losses must not report p < 0.10: check it returns NaN or ≥ 0.10)
        let rows: Vec<NightRow> =
            (0..40).map(|n| row(n, (n % 2) as f64, if n % 2 == 1 { 0.9 } else { 0.1 }, 0.5, 0.4 + 0.1 * (n % 2) as f64)).collect();
        let m = night_metrics(&rows);
        let gate = gate_of(&m);
        assert_eq!(gate.len(), 4);
        assert!(gate.iter().filter(|g| g.target == "rise").all(|g| g.pass));
        assert!(gate.iter().filter(|g| g.target == "fall").all(|g| !g.pass));
    }

    #[test]
    fn calibration_bins_cover_every_scored_row() {
        let rows: Vec<NightRow> = (0..100).map(|n| row(n, (n % 3 == 0) as i64 as f64, n as f64 / 100.0, 0.3, 0.3)).collect();
        let bins = calibration(&rows, |r| r.p_rise, |r| r.rise);
        assert_eq!(bins.iter().map(|b| b.n).sum::<usize>(), 100);
    }
}
```
(Adapt the test fixtures to the final `NightRow` and metric types; keep the four checks: log loss, season refusals, the gate's both-targets-both-baselines rule, calibration covering all rows.)

- [ ] **Step 2: Run them to make sure they fail**

Run: `cargo test -p fplopt-backtest night_eval`
Expected: compile errors.

- [ ] **Step 3: Implement `night_eval.rs`**

```rust
//! Gate N (`docs/superpowers/specs/2026-11-09-move-timing-design.md`, "The nightly model"):
//! walk-forward evaluation of the night price model (`research night-eval`).
//!
//! For each evaluated season S (develop only: 2023/24–2024/25 are kept clean and the holdout
//! is never read) the model and the baselines are fitted on the nights of first_fit..S−1 at
//! S's first night (labels before it), then score every night of S whose label is known
//! (from a view after the season: evaluation only). Baselines: B0 the training base rate of
//! each target; B1 the same booster on `net_per_own` alone (a threshold rule on net transfers
//! per owner). Metrics per target: log loss of the model and both baselines, one-sided
//! Diebold–Mariano tests of the model against each on per-night mean log loss (lag 7,
//! clustered by night), and a calibration table (10 bins of the model's probability).
//! Gate N passes when the model beats both baselines on both targets with p < 0.10.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use fplopt_core::pyjson::Ordered;
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_features::night::{SeasonSnaps, features_at, gw_start};
use fplopt_models::night::{NightParams, fit_night, fit_on, nights, predict_night, training_rows};
use fplopt_store::{Column, DataStore, Micros, Table};

use crate::eval::diebold_mariano;

/// Nights' log losses overlap for about a week (the counter carries over nights).
pub const DM_LAG: usize = 7;
/// The gate's one-sided level.
pub const ALPHA: f64 = 0.10;
/// The last develop season with snapshots; later seasons are kept clean (spec, Measurement).
pub const LAST_DEVELOP: i64 = 2022;
/// Snapshots start in April 2021 (2020/21's last nights).
pub const FIRST_FIT_MIN: i64 = 2020;
const EPS: f64 = 1e-15;

/// One scored player-night: the truths and each arm's probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct NightRow {
    pub season: i64,
    pub night: Micros,
    pub player_key: i64,
    pub rise: f64,
    pub fall: f64,
    pub p_rise: f64,
    pub p_fall: f64,
    pub b0_rise: f64,
    pub b0_fall: f64,
    pub b1_rise: f64,
    pub b1_fall: f64,
}

pub fn check_seasons(seasons: &[i64], first_fit: i64) -> Result<()> {
    if first_fit < FIRST_FIT_MIN {
        bail!("--first-fit {first_fit}: no snapshots before {}", season_label(FIRST_FIT_MIN as i32));
    }
    for &s in seasons {
        if HOLDOUT_SEASONS.contains(&(s as i32)) {
            bail!("{} is the holdout season", season_label(s as i32));
        }
        if s > LAST_DEVELOP {
            bail!("{} is kept clean until a frozen policy passes on develop", season_label(s as i32));
        }
        if s <= first_fit {
            bail!("{} is not after --first-fit {first_fit}", season_label(s as i32));
        }
    }
    Ok(())
}

pub fn log_loss(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    pairs
        .iter()
        .map(|&(y, p)| {
            let p = p.clamp(EPS, 1.0 - EPS);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum::<f64>()
        / n
}

/// The scored rows of season `s`: fit on first_fit..s−1 at s's first night, predict every
/// night of s with a known label.
pub fn season_rows(store: &DataStore, s: i64, first_fit: i64, params: &NightParams) -> Result<Vec<NightRow>> {
    let after = store.as_of(Micros::MAX);
    // the evaluated season's nights (labels from a view after the season: evaluation only)
    let season_nights = nights(&after, s)?;
    let Some(first) = season_nights.first() else { bail!("no nights in {}", season_label(s as i32)) };
    let fit_seasons: Vec<i64> = (first_fit..s).collect();
    let train_rows = training_rows(store, &fit_seasons, first.cutoff)?;
    let model = fit_night(&train_rows, params)?;
    let b1 = fit_on(&train_rows, &["net_per_own"], params)?;
    let base = |c: &str| -> Result<f64> {
        let v: Vec<f64> = train_rows.f64(c)?.iter().copied().filter(|x| x.is_finite()).collect();
        Ok(v.iter().sum::<f64>() / v.len() as f64)
    };
    let (b0_rise, b0_fall) = (base("rise")?, base("fall")?);
    let scored = training_rows(store, &[s], Micros::MAX)?;
    let p = predict_night(&model, &scored)?;
    let q = predict_night(&b1, &scored)?;
    let (nights_col, keys) = (scored.opt_i64("night")?, scored.i64("player_key")?);
    let (rise, fall) = (scored.f64("rise")?, scored.f64("fall")?);
    let (pr, pf, qr, qf) = (p.f64("p_rise")?, p.f64("p_fall")?, q.f64("p_rise")?, q.f64("p_fall")?);
    Ok((0..scored.height())
        .filter(|&i| rise[i].is_finite())
        .map(|i| NightRow {
            season: s,
            night: nights_col[i].expect("a night"),
            player_key: keys[i],
            rise: rise[i],
            fall: fall[i],
            p_rise: pr[i],
            p_fall: pf[i],
            b0_rise,
            b0_fall,
            b1_rise: qr[i],
            b1_fall: qf[i],
        })
        .collect())
}
```
Note on `training_rows(store, &[s], Micros::MAX)`: features at each night still see only snapshots before that night (`features_at`); the labels come from after the season, which is the evaluation's truth. It must never be used to fit. If `refuse_holdout` or `store.as_of(Micros::MAX)` reads the holdout season's rows, check that `training_rows` only reads `s`'s snapshots (it filters by season) and that `s` is never the holdout (`check_seasons`). `SeasonSnaps`, `features_at`, `gw_start` may end up unused here; drop the import.

Then the metrics:
```rust
/// Per target, the log losses and the model-vs-baseline tests.
pub struct GateRow {
    pub target: &'static str,
    pub baseline: &'static str,
    pub value_model: f64,
    pub value_baseline: f64,
    pub p: f64,
    pub pass: bool,
}

pub struct Bin {
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
    pub mean_p: f64,
    pub rate: f64,
}

/// 10 equal-width bins of the probability.
pub fn calibration(rows: &[NightRow], p: impl Fn(&NightRow) -> f64, y: impl Fn(&NightRow) -> f64) -> Vec<Bin> {
    (0..10)
        .map(|b| {
            let (lo, hi) = (b as f64 / 10.0, (b + 1) as f64 / 10.0);
            let inside: Vec<&NightRow> = rows
                .iter()
                .filter(|r| {
                    let v = p(r);
                    v >= lo && (v < hi || (b == 9 && v <= hi))
                })
                .collect();
            let n = inside.len();
            let mean = |f: &dyn Fn(&NightRow) -> f64| {
                if n == 0 { f64::NAN } else { inside.iter().map(|r| f(r)).sum::<f64>() / n as f64 }
            };
            Bin { lo, hi, n, mean_p: mean(&|r| p(r)), rate: mean(&|r| y(r)) }
        })
        .collect()
}

type Pick = fn(&NightRow) -> (f64, f64, f64, f64);

const TARGETS: [(&str, Pick); 2] = [
    ("rise", |r| (r.rise, r.p_rise, r.b0_rise, r.b1_rise)),
    ("fall", |r| (r.fall, r.p_fall, r.b0_fall, r.b1_fall)),
];

fn rec(pairs: Vec<(&str, Ordered)>) -> Ordered {
    Ordered::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Per night (in night order) the model's and a baseline's mean log loss.
fn per_night(rows: &[NightRow], pick: Pick, baseline: usize) -> Vec<(Micros, f64, f64)> {
    let mut by: BTreeMap<Micros, (Vec<(f64, f64)>, Vec<(f64, f64)>)> = BTreeMap::new();
    for r in rows {
        let (y, p, b0, b1) = pick(r);
        let e = by.entry(r.night).or_default();
        e.0.push((y, p));
        e.1.push((y, if baseline == 0 { b0 } else { b1 }));
    }
    by.into_iter().map(|(n, (a, b))| (n, log_loss(&a), log_loss(&b))).collect()
}

/// The metrics object: `n_rows`, `n_nights`, `log_loss` (per target: model, b0, b1), `dm`
/// (model vs each baseline: value_a, value_b, then `diebold_mariano`'s pairs), `calibration`
/// (per target and bin: lo, hi, n, mean_p, rate), `gate` (per target and baseline: p, pass)
/// and `gate_pass`.
pub fn night_metrics(rows: &[NightRow]) -> Ordered {
    let nights: std::collections::BTreeSet<Micros> = rows.iter().map(|r| r.night).collect();
    let (mut losses, mut dm, mut cal, mut gate) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut all_pass = true;
    for (target, pick) in TARGETS {
        let arm = |k: usize| {
            let pairs: Vec<(f64, f64)> = rows
                .iter()
                .map(|r| {
                    let (y, p, b0, b1) = pick(r);
                    (y, [p, b0, b1][k])
                })
                .collect();
            log_loss(&pairs)
        };
        losses.push(rec(vec![
            ("target", Ordered::Str(target.to_owned())),
            ("model", Ordered::Float(arm(0))),
            ("b0", Ordered::Float(arm(1))),
            ("b1", Ordered::Float(arm(2))),
        ]));
        for (baseline, name) in [(0, "b0"), (1, "b1")] {
            let loss = per_night(rows, pick, baseline);
            let mean = |f: &dyn Fn(&(Micros, f64, f64)) -> f64| {
                loss.iter().map(f).sum::<f64>() / loss.len() as f64
            };
            let test = diebold_mariano(&loss, DM_LAG);
            let p = test
                .iter()
                .find(|(k, _)| *k == "p_a_better")
                .and_then(|(_, v)| match v {
                    Ordered::Float(x) => Some(*x),
                    _ => None,
                })
                .unwrap_or(f64::NAN);
            let pass = p < ALPHA;
            all_pass &= pass;
            let mut pairs = vec![
                ("target", Ordered::Str(target.to_owned())),
                ("baseline", Ordered::Str(name.to_owned())),
                ("value_a", Ordered::Float(mean(&|l| l.1))),
                ("value_b", Ordered::Float(mean(&|l| l.2))),
            ];
            pairs.extend(test);
            dm.push(rec(pairs));
            gate.push(rec(vec![
                ("target", Ordered::Str(target.to_owned())),
                ("baseline", Ordered::Str(name.to_owned())),
                ("p", Ordered::Float(p)),
                ("pass", Ordered::Bool(pass)),
            ]));
        }
        for b in calibration(rows, |r| pick(r).1, |r| pick(r).0) {
            cal.push(rec(vec![
                ("target", Ordered::Str(target.to_owned())),
                ("lo", Ordered::Float(b.lo)),
                ("hi", Ordered::Float(b.hi)),
                ("n", Ordered::Int(b.n as i128)),
                ("mean_p", Ordered::Float(b.mean_p)),
                ("rate", Ordered::Float(b.rate)),
            ]));
        }
    }
    rec(vec![
        ("n_rows", Ordered::Int(rows.len() as i128)),
        ("n_nights", Ordered::Int(nights.len() as i128)),
        ("log_loss", Ordered::Array(losses)),
        ("dm", Ordered::Array(dm)),
        ("calibration", Ordered::Array(cal)),
        ("gate", Ordered::Array(gate)),
        ("gate_pass", Ordered::Bool(all_pass)),
    ])
}

/// The `gate` rows of `night_metrics`' object.
pub fn gate_of(metrics: &Ordered) -> Vec<GateRow> {
    let Ordered::Object(top) = metrics else { return vec![] };
    let Some((_, Ordered::Array(rows))) = top.iter().find(|(k, _)| k == "gate") else { return vec![] };
    rows.iter()
        .filter_map(|r| {
            let Ordered::Object(r) = r else { return None };
            let get = |k: &str| r.iter().find(|(n, _)| n == k).map(|(_, v)| v);
            let text = |k: &str| match get(k) {
                Some(Ordered::Str(s)) => Some(s.clone()),
                _ => None,
            };
            let target = if text("target")? == "rise" { "rise" } else { "fall" };
            let baseline = if text("baseline")? == "b0" { "b0" } else { "b1" };
            let p = match get("p") {
                Some(Ordered::Float(x)) => *x,
                _ => f64::NAN,
            };
            let pass = matches!(get("pass"), Some(Ordered::Bool(true)));
            Some(GateRow { target, baseline, value_model: f64::NAN, value_baseline: f64::NAN, p, pass })
        })
        .collect()
}
```
Check `Ordered`'s real variants (`Str`, `Float`, `Int`, `Bool`, `Array`, `Object`) and the exact key and value type `diebold_mariano` uses for `p_a_better` (see `fplopt-backtest/src/eval.rs` ~853); adjust the matches to them. `GateRow`'s `value_model`/`value_baseline` may be dropped if unused.

`rows_table(&[NightRow]) -> Table` with columns `season, night (Ts), player_key, rise, fall, p_rise, p_fall, b0_rise, b0_fall, b1_rise, b1_fall`.

`evaluate_nights(store, seasons, first_fit, params) -> Result<(Vec<NightRow>, Ordered)>`: `season_rows` for each season in order (sequential is fine: one or two seasons), then `night_metrics`.

In `research.rs`, mirror `price_eval`: `NightEvalArgs { seasons, first_fit (default 2020), family (default "night price model"), experiments, data_dir, out }` (no `--params`, no `--jobs`, no xP cache); `night_eval(args)`: parse seasons, `check_seasons` (usage error, exit 2, on failure), open the store, `evaluate_nights`, write `predictions.parquet` and `metrics.json` (command, config, family, n_variants, runtime_seconds, metrics), `log_experiment` **before** printing, then print the log-loss table, the DM table and "Gate N: PASS/FAIL" with the four gate rows. Add an args test like `price_eval_args`. Module doc line for `research night-eval`.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fplopt-backtest night_eval`, `cargo test -p fplopt-cli research`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-backtest/src/night_eval.rs crates/fplopt-backtest/src/lib.rs crates/fplopt-cli/src/research.rs
```
```bash
git commit -m "feat(research): night-eval: walk-forward night price model and Gate N" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Leakage and a local smoke run

**Files:** none (checks), unless a check fails.

- [ ] **Step 1: Leakage check of the night features**

Run: `cargo build --release -p fplopt-cli`
Run: `FPLOPT_DATA_DIR=../../../data target/release/fplopt-rs.exe check leakage --deadlines 2 --only night_features`
Expected: exit 0, no `leak:` lines. (`check leakage` builds each frame at deadline views; `night_features` at a deadline view is the night features with cutoff = deadline, and it must not change when everything at or after the deadline is corrupted.)

- [ ] **Step 2: The early-view equality on real data** (a throwaway test run, not committed)

Run the `features_at_equals_night_features_on_the_view` idea on real data once: a small `#[ignore]` test in `fplopt-models/src/night.rs` that opens `DataStore::open("../../../data")` (skip if absent), picks 5 nights of 2022 from `nights`, and checks `training_rows(&s, &[2022], cutoff)`'s feature columns for those nights equal `night_features(&s.as_of_early(night.deadline, night.cutoff))`. Run it with `cargo test -p fplopt-models -- --ignored night_real`. Keep the test (ignored by default) and commit it with the next step.

- [ ] **Step 3: Timing**

Run: `FPLOPT_DATA_DIR=../../../data target/release/fplopt-rs.exe research night-eval --seasons 2021 --first-fit 2020 --family "night smoke" --experiments C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/night-smoke/exp.csv --out C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/night-smoke/run`
Expected: finishes (report the runtime and the row and night counts: about 250,000 player-nights and 330–370 nights in 2021/22, and rise/fall counts near 563 / 1,354). The smoke run evaluates 2021/22 from a fit on 2020/21's 65 nights only, so it tests the mechanics without showing 2022/23, Gate N's season; 2022/23 is evaluated once, in Task 5's logged run. Nothing is tuned on the smoke result.

- [ ] **Step 4: Commit** (only if a test or fix was added)

```bash
git add crates/fplopt-models/src/night.rs
```
```bash
git commit -m "test(models): night features equal on real early views (ignored)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4b: Calibration in Gate N (amended 2026-11-09, before Gate N's run)

The smoke run on 2021/22 (282 nights and 186,929 player-nights; the plan's earlier estimate of 330–370 nights was wrong) beat both baselines on log loss. Above p = 0.1, though, its probabilities were overconfident: predicted 0.55 against an observed 0.14 for rises, and 0.55 against 0.24 for falls. The spec's "Calibration" bullet (Gate N) now adds two things:

- **Platt recalibration** of the model and of B1, using out-of-fold predictions on the training rows. The folds are 5 contiguous blocks of training nights.
- **A calibration condition**, per target: among rows with calibrated p ≥ 0.3, n ≥ 30 and the observed rate ≥ the mean calibrated p − 0.10.

What changes in the code:

- **`fplopt_models::night`** gets:
  - `Platt { a, b }`, where p' = σ(a + b·logit(p)), with p clipped to [1e-6, 1 − 1e-6];
  - `fit_platt(&[(y, p)])`, a Newton/IRLS logistic fit with two parameters;
  - `oof_predictions(rows, features, params, folds = 5)`, out-of-fold predictions by contiguous night blocks;
  - `CalibratedFit { fit, rise, fall }`, built by `fit_calibrated(rows, features, params)`, plus `predict_calibrated`.
- **`night_eval`**:
  - `NightRow` gets `raw_p_rise`/`raw_p_fall` and `raw_b1_rise`/`raw_b1_fall`.
  - `p_*` and `b1_*` hold the calibrated probabilities.
  - The metrics add `log_loss_raw`, the Platt coefficients per season and target, and `calibration_gate` (per target: n, mean_p, rate, pass).
  - `gate_pass` requires the four DM tests and both calibration conditions.
  - The CLI prints the calibration-gate rows.
- **Tests:**
  - Platt recovers known coefficients on synthetic data.
  - The out-of-fold predictions never use a row's own fold.
  - The calibration condition fails with n < 30, or when the observed rate < mean_p − 0.10.
  - `gate_pass` needs both the DM tests and the calibration condition.
- **The smoke run on 2021** is run again once the changes are in, as a mechanics check.

### Task 5 (controller): Gate N run, docs

- [ ] **Step 1:** Add to `CLAUDE.md` Commands, after `research price-eval`:
`- \`fplopt-rs research night-eval --seasons 2022 [--first-fit 2020] [--family NAME] [--experiments PATH] [--data-dir data] --out DIR\` — Gate N of the move timing study (\`docs/superpowers/specs/2026-11-09-move-timing-design.md\`; Rust only, \`fplopt_backtest::night_eval\`, model \`fplopt_models::night\`, features \`fplopt_features::night\`): P(rise tonight) and P(fall tonight) per player at 22:00 UK, labelled by the next morning's price; fitted walk-forward on the nights of --first-fit..S−1 (snapshots start April 2021), scored on S's nights by log loss against the base rate (B0) and the booster on net_per_own alone (B1), one-sided Diebold–Mariano per night (lag 7), calibration; seasons after 2022/23 refused (kept clean), the holdout refused; writes predictions.parquet and metrics.json, logged before printing (default family \`night price model\`)`
- [ ] **Step 2:** Full checks: `cargo test --workspace`, clippy, fmt, `uv run ruff check .`.
- [ ] **Step 3:** The Gate N run (local, develop data; ~minutes): the Task 4 Step 3 command with `--family "night price model gate n" --experiments results/experiments.csv --out results/night-gate-n`.
- [ ] **Step 4:** Note `docs/research/2026-11-night-price-model.md` (counts, log losses, DM, calibration, the gate verdict, the smoke run disclosed) and a PLAN §12 row; commit; push (approved within this flow).
- [ ] **Step 5:** If Gate N passes, write policy T's plan; if it fails, stop and report (T is not run).
