# Chip Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The bot's 24-hour alert carries a chip section: the chip planner's call, the simple rules' call, unused chips with expiry warnings, and the evidence. Every call is logged per user and deadline, and the frozen V6's call is logged from GW20.

**Architecture:**
- The rule baseline's logic moves from `fplopt-backtest` into a new shared module, `fplopt_optimize::chip_rules`. That module has the thresholds, the last-chance rule, the rule chooser and the per-GW chip gains. The backtest calls the moved code, so its results are unchanged.
- The bot computes the calls in a new module, `fplopt_bot::chip_calls`, from:
  - the user's no-chip plan (`UserPlan`);
  - the `/chips` search (`ChipOptions`);
  - from GW20, a second search with V6's settings.
- `messages::chip_section` renders the calls. `BotService::alert_user` prepends the section and logs a `chip_calls` row.
- A `chip_alerts` flag keeps the section out of `bot replay`, so the Python golden and `dev/bot_parity.py` still compare byte for byte.

**Tech Stack:** Rust (workspace `crates/`), HiGHS through `fplopt-optimize`, rusqlite. Spec: `docs/superpowers/specs/2026-11-05-chip-alerts-design.md`.

**Conventions for every task:**
- Work in `C:\Users\samin\Documents\GitHub\fpl-optimizer\.claude\worktrees\v1r` on branch `feat/chip-alerts`.
- Build with `-j 2`.
- `FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data` for anything that reads `data/`.
- Before each commit:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets -j 2 -- -D warnings`
  - `cargo test -p <crate> -j 2`
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never push.
- In Bash, don't chain `cd` with `&&`. Use absolute paths or PowerShell.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `crates/fplopt-optimize/src/chip_rules.rs` | create | Rule thresholds (`RuleThresholds`, `H`), `CHIP_PRIORITY`, `CHIP_DIAGNOSTICS`, `last_chance`, `rule_chip`, `chip_gains`, `one_gw_params`, `target_xp_of` |
| `crates/fplopt-optimize/src/lib.rs` | modify | `pub mod chip_rules;` |
| `crates/fplopt-backtest/src/policies.rs` | modify | `chip_gains`/`choose`/`one_gw_params`/`target_xp_of`/the constants now call or re-export `chip_rules` |
| `crates/fplopt-bot/src/chip_calls.rs` | create | `ChipCalls`, `OpenChip`, `blanks_from_frame`, `v6_params`, `compute_chip_calls`, `calls_changed` |
| `crates/fplopt-bot/src/lib.rs` | modify | `pub mod chip_calls;` |
| `crates/fplopt-bot/src/messages.rs` | modify | `chip_section(calls) -> String` |
| `crates/fplopt-bot/src/store.rs` | modify | `chip_calls` table, `record_chip_call`, `chip_calls`, `ChipCallRow` |
| `crates/fplopt-bot/src/service.rs` | modify | `chip_alerts` flag, `v6_planner`, alert integration |
| `crates/fplopt-bot/src/replay.rs` | modify | Script key `chip_alerts` |
| `crates/fplopt-cli/src/bot.rs` | modify | Turn `chip_alerts` on for `bot run` / `live tick` |
| `crates/fplopt-bot/tests/chip_alerts.rs` | create | Replay with `chip_alerts: true` on the synthetic league |
| `docs/PLAN.md`, `CLAUDE.md`, the spec | modify | Docs |

---

### Task 1: Shared rule logic in `fplopt_optimize::chip_rules`

**Files:**
- Create: `crates/fplopt-optimize/src/chip_rules.rs`
- Modify: `crates/fplopt-optimize/src/lib.rs`, `crates/fplopt-backtest/src/policies.rs` (the `chip_gains` at ~line 840, `choose` at ~553, the helpers at ~759–829, and the constants at 38–42)

The backtest's `blanks(ctx)` stays in the backtest, because it reads the `AsOfView`. `chip_gains` gets the blank count as an argument.

- [ ] **Step 0: Build the baseline binary before changing anything.**
  - Run `cargo build --release -p fplopt-cli -j 2`.
  - Copy `target/release/fplopt-rs.exe` to `C:\Users\samin\AppData\Local\Temp\claude\6ff2f6c53ca963aa\50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e\scratchpad\fplopt-rs-before.exe`.
  - With `FPLOPT_DATA_DIR` set, run it once:

    `fplopt-rs-before.exe backtest run --seasons 2021 --spec "chiprules:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87" --starts template@1 --jobs 4 --no-xp-cache --out <scratchpad>/chiprules-before`
  - Then `git checkout -- results/experiments.csv`.

- [ ] **Step 1: Write the failing tests** in `crates/fplopt-optimize/src/chip_rules.rs` (a `#[cfg(test)] mod tests` at the bottom of the new file).

  `rule_chip` and `last_chance` need a `SquadState` and `Rules`. Build `Rules` the way `crates/fplopt-optimize/src/state.rs`'s tests do; look for an existing helper such as `test_rules()` or `backtest_rules`, and reuse it. Each test sets the chip windows explicitly. The windows are 2026/27's two halves:
  - `wildcard`: 1 = 2..19, 2 = 20..38;
  - `freehit`: 3 = 2..19, 6 = 20..38;
  - `bboost`: 4 = 1..19, 7 = 20..38;
  - `3xc`: 5 = 1..19, 8 = 20..38.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn gains(pairs: &[(&str, f64)]) -> BTreeMap<String, Option<f64>> {
        pairs.iter().map(|(k, v)| (k.to_string(), Some(*v))).collect()
    }

    #[test]
    fn thresholds_fire_in_priority_order() {
        let (state, rules) = state_at(10, &[]); // gw_index 10, no chips used
        let g = gains(&[("captain_xp", 9.0), ("bench_xp", 12.0), ("blanks", 0.0)]);
        // bboost is before 3xc in CHIP_PRIORITY
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap().as_deref(), Some("bboost"));
        let g = gains(&[("captain_xp", 9.0), ("bench_xp", 5.0), ("blanks", 0.0)]);
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap().as_deref(), Some("3xc"));
        let g = gains(&[("captain_xp", 5.0), ("bench_xp", 5.0), ("blanks", 3.0)]);
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap(), None);
        let g = gains(&[("captain_xp", 5.0), ("bench_xp", 5.0), ("blanks", 4.0)]);
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap().as_deref(), Some("freehit"));
    }

    #[test]
    fn last_chance_when_chips_outnumber_gws() {
        // gw_index 19: four first-half chips unused, one GW left -> freehit first (priority)
        let (state, rules) = state_at(19, &[]);
        assert_eq!(last_chance(&state, &rules).unwrap().as_deref(), Some("freehit"));
        // gw_index 18 with only bboost and freehit left (wildcard and 3xc used): 2 chips, 2 GWs
        let (state, rules) = state_at(18, &[(1, 4), (5, 7)]);
        assert_eq!(last_chance(&state, &rules).unwrap(), None);
        // gw_index 19 with the same two left: 2 chips, 1 GW -> freehit
        let (state, rules) = state_at(19, &[(1, 4), (5, 7)]);
        assert_eq!(last_chance(&state, &rules).unwrap().as_deref(), Some("freehit"));
        // and rule_chip returns the forced chip whatever the gains
        let g = gains(&[("captain_xp", 0.0), ("bench_xp", 0.0), ("blanks", 0.0)]);
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap().as_deref(), Some("freehit"));
    }

    #[test]
    fn h_is_the_chosen_rule_baseline() {
        // docs/superpowers/plans/2026-10-11-chip-policy.md: t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87
        assert_eq!(H, RuleThresholds { t_tc: 8.3, t_bb: 10.9, k_fh: 4, t_wc: 12.87 });
    }
}
```

  `state_at(gw_index, chips_used)` is a test helper you write in the same module. It builds a 15-player `SquadState` with `SquadState::new(2026, gw_index, holdings, 0, 1)` and sets `chips_used`, plus the `Rules` with the windows above.

- [ ] **Step 2: Run them and see them fail.**

  Run: `cargo test -p fplopt-optimize -j 2 chip_rules`

  Expected: compile errors (the module doesn't exist yet).

- [ ] **Step 3: Write the module.** Move the code from `crates/fplopt-backtest/src/policies.rs` and keep the arithmetic and order exactly as they are.

```rust
//! The chip study's rule baseline H (docs/superpowers/plans/2026-10-11-chip-policy.md) and
//! the per-GW chip quantities it reads, shared by the backtester (`chiprules`, `chipgains`)
//! and the bot's chip suggestions.

use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use fplopt_build::scoring::Rules;

use crate::chips::make_scenario;
use crate::model::{SolveError, solve_plan};
use crate::plans::Plan;
use crate::problem::{PlanInput, XpFrame};
use crate::state::{InvalidDecision, PoolPlayer, SquadState, chip_window};
use crate::{ChipScenario, OptimizerParams};

/// Order the rule policy considers chips in (one per GW), and forced "last chance" plays.
pub const CHIP_PRIORITY: [&str; 4] = ["freehit", "wildcard", "bboost", "3xc"];
/// Per-GW chip quantities (`chip_gains`), recorded by the simulator.
pub const CHIP_DIAGNOSTICS: [&str; 7] =
    ["captain_xp", "bench_xp", "gain_tc", "gain_bb", "gain_fh", "gain_wc", "blanks"];

/// The rule thresholds: Triple Captain if the captain's xP ≥ `t_tc`, Bench Boost if the
/// bench's xP ≥ `t_bb`, Free Hit if ≥ `k_fh` squad players blank, Wildcard if its plan beats
/// the no-chip plan by ≥ `t_wc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuleThresholds {
    pub t_tc: f64,
    pub t_bb: f64,
    pub k_fh: i64,
    pub t_wc: f64,
}

/// H as chosen on develop: `chiprules:v1:…,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87`.
pub const H: RuleThresholds = RuleThresholds { t_tc: 8.3, t_bb: 10.9, k_fh: 4, t_wc: 12.87 };

fn chip_available(state: &SquadState, name: &str, rules: &Rules) -> bool {
    chip_window(state, name, rules).is_ok()
}

/// The chip the rules must play now or lose (more unused chips in a window than GWs left in
/// it, first in `CHIP_PRIORITY`), or None.
pub fn last_chance(state: &SquadState, rules: &Rules) -> Result<Option<String>> {
    let playable: Vec<&str> =
        CHIP_PRIORITY.iter().copied().filter(|c| chip_available(state, c, rules)).collect();
    let mut windows = HashMap::new();
    for c in &playable {
        windows.insert(*c, chip_window(state, c, rules)?);
    }
    let stops: HashMap<i64, i64> = rules.chips.iter().map(|w| (w.chip_id, w.stop)).collect();
    for chip in &playable {
        let stop = stops[&windows[chip]];
        let same: Vec<&str> =
            playable.iter().copied().filter(|c| stops[&windows[c]] == stop).collect();
        if same.len() as i64 > stop - state.gw_index {
            return Ok(Some(same[0].to_owned()));
        }
    }
    Ok(None)
}

/// The chip H plays this GW: the last chance, else the first chip in `CHIP_PRIORITY` whose
/// threshold fires, else None.
pub fn rule_chip(
    state: &SquadState,
    rules: &Rules,
    gains: &BTreeMap<String, Option<f64>>,
    t: &RuleThresholds,
) -> Result<Option<String>> {
    if let Some(chip) = last_chance(state, rules)? {
        return Ok(Some(chip));
    }
    let g = |k: &str| gains.get(k).copied().flatten();
    let fires = |c: &str| match c {
        "freehit" => g("blanks").unwrap_or(0.0) >= t.k_fh as f64,
        "wildcard" => g("gain_wc").is_some_and(|x| x >= t.t_wc),
        "bboost" => g("bench_xp").unwrap_or(0.0) >= t.t_bb,
        "3xc" => g("captain_xp").unwrap_or(0.0) >= t.t_tc,
        _ => false,
    };
    Ok(CHIP_PRIORITY
        .iter()
        .copied()
        .filter(|c| chip_available(state, c, rules))
        .find(|c| fires(c))
        .map(str::to_owned))
}

/// `player_key`'s target-GW xP in `frame` (horizon 0, summed over a double; null → 0).
pub fn target_xp_of(frame: &XpFrame, key: i64) -> f64 {
    let mut total = 0.0;
    for i in 0..frame.len() {
        if frame.player_key[i] == key && frame.horizon[i] == 0 {
            total += frame.xp[i].unwrap_or(0.0);
        }
    }
    total
}

/// `params` for a 1-GW plan (the Free Hit gain): the expected-minutes floor scaled to one GW.
pub fn one_gw_params(params: &OptimizerParams) -> OptimizerParams {
    OptimizerParams {
        horizon: 1,
        min_minutes: params.min_minutes / params.horizon as f64,
        ..params.clone()
    }
}

fn chip_objective(problem: &PlanInput, params: &OptimizerParams, chip: &str) -> Result<Option<f64>> {
    match make_scenario(problem, &BTreeMap::from([(0, chip.to_owned())])) {
        Err(InvalidDecision(_)) => Ok(None),
        Ok(scenario) => match solve_plan(problem, params, None, &[], &scenario) {
            Ok(plan) => Ok(Some(plan.objective)),
            Err(SolveError::Infeasible(_) | SolveError::LimitReached { .. }) => Ok(None),
            Err(SolveError::Other(e)) => Err(e),
        },
    }
}

fn no_chip_objective(problem: &PlanInput, params: &OptimizerParams) -> Result<Option<f64>> {
    match solve_plan(problem, params, None, &[], &ChipScenario::none()) {
        Ok(plan) => Ok(Some(plan.objective)),
        Err(SolveError::Infeasible(_) | SolveError::LimitReached { .. }) => Ok(None),
        Err(SolveError::Other(e)) => Err(e),
    }
}

/// The `CHIP_DIAGNOSTICS` for the no-chip `plan` of `problem` (built from `state`, `pool`,
/// `frame`, `rules` with `params`); `blanks` is the caller's count of squad players whose club
/// has no target-GW fixture.
#[allow(clippy::too_many_arguments)]
pub fn chip_gains(
    state: &SquadState,
    pool: &[PoolPlayer],
    frame: &XpFrame,
    rules: &Rules,
    problem: &PlanInput,
    params: &OptimizerParams,
    plan: &Plan,
    blanks: i64,
) -> Result<BTreeMap<String, Option<f64>>> {
    // body: the backtest's `chip_gains` with `ctx.state/pool/frame/rules` replaced by the
    // arguments and `blanks(ctx)?` by `blanks`; same order of solves and float operations.
}
```

  Write the `chip_gains` body by moving the backtest's body (policies.rs ~840–878) across unchanged, apart from the renames in the comment. `make_scenario` returns `Result<_, InvalidDecision>`. Check its signature in `crates/fplopt-optimize/src/chips.rs` and adjust the match if it differs. If `make_scenario` or `InvalidDecision` aren't `pub` where you import them, make them `pub(crate)` or `pub`.

  In `crates/fplopt-optimize/src/lib.rs`, add `pub mod chip_rules;`.

- [ ] **Step 4: Point the backtest at the module.** In `crates/fplopt-backtest/src/policies.rs`:
  - Replace the two `pub const`s at lines 38–42 with `pub use fplopt_optimize::chip_rules::{CHIP_DIAGNOSTICS, CHIP_PRIORITY};`.
  - Replace `pub fn one_gw_params` with `pub use fplopt_optimize::chip_rules::one_gw_params;`.
  - Delete the local `target_xp_of`, `chip_objective` and `no_chip_objective`. Import `fplopt_optimize::chip_rules::target_xp_of` and keep `solve_with_chip`.
  - Turn `pub fn chip_gains(ctx, problem, params, plan)` into a thin wrapper:

```rust
pub fn chip_gains(
    ctx: &DecisionContext,
    problem: &PlanInput,
    params: &OptimizerParams,
    plan: &Plan,
) -> Result<BTreeMap<String, Option<f64>>> {
    fplopt_optimize::chip_rules::chip_gains(
        ctx.state, ctx.pool, ctx.frame, ctx.rules, problem, params, plan, blanks(ctx)?,
    )
}
```

  - The body of `fn choose` becomes:

```rust
        let Policy::ChipRules { t_tc, t_bb, k_fh, t_wc, .. } = self else {
            bail!("choose: not a chip rule policy");
        };
        let t = fplopt_optimize::chip_rules::RuleThresholds {
            t_tc: *t_tc, t_bb: *t_bb, k_fh: *k_fh, t_wc: *t_wc,
        };
        fplopt_optimize::chip_rules::rule_chip(ctx.state, ctx.rules, gains, &t)
```

  Keep `rule_tc` as it is.

- [ ] **Step 5: Run the tests.**

  Run: `cargo test -p fplopt-optimize -j 2 chip_rules`, then `cargo test -p fplopt-backtest -j 2`.

  Expected: all pass.

- [ ] **Step 6: Check the backtest is unchanged.**
  1. Run `cargo build --release -p fplopt-cli -j 2`.
  2. Run Step 0's command with `target/release/fplopt-rs.exe` and `--out <scratchpad>/chiprules-after`.
  3. Run `git checkout -- results/experiments.csv`.
  4. Compare the two runs: `C:/Users/samin/Documents/GitHub/fpl-optimizer/.venv/Scripts/python.exe -I -c "import pandas as pd; a=pd.read_parquet(r'<scratchpad>/chiprules-before/gws.parquet'); b=pd.read_parquet(r'<scratchpad>/chiprules-after/gws.parquet'); pd.testing.assert_frame_equal(a,b); print('identical', len(a))"`.

  Expected: `identical 38`.

- [ ] **Step 7: Commit.**

```bash
git add crates/fplopt-optimize/src/chip_rules.rs crates/fplopt-optimize/src/lib.rs crates/fplopt-backtest/src/policies.rs
git commit -m "refactor(chips): rule baseline H and chip gains move to fplopt_optimize::chip_rules (backtest unchanged)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The bot's chip calls (`fplopt_bot::chip_calls`)

**Files:**
- Create: `crates/fplopt-bot/src/chip_calls.rs`
- Modify: `crates/fplopt-bot/src/lib.rs` (`pub mod chip_calls;`)

**What it computes** (spec §1–3), for one user at one deadline:

```rust
/// One unused chip whose window is open at the context's GW.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenChip {
    pub name: String,         // "bboost", ...
    pub last_gw: i64,         // the window's stop as a GW number
    pub best_gw: Option<i64>, // the planner's most promising GW for it alone, if solved
    pub expiring: bool,       // stop − gw_index ≤ 3
}

/// The chip calls for one alert (logged in `chip_calls`, shown by `messages::chip_section`).
#[derive(Clone, Debug, PartialEq)]
pub struct ChipCalls {
    pub gw: i64,
    /// The planner's chip for this GW (None: hold) and its gain over the no-chip scenario.
    pub planner: Option<String>,
    pub planner_gain: f64,
    /// H's chip for this GW and the numbers it read.
    pub rules: Option<String>,
    pub gains: BTreeMap<String, Option<f64>>,
    /// The chip H must play now or lose (overrides the planner's line).
    pub forced: Option<String>,
    pub open: Vec<OpenChip>,
    /// V6's chip for this GW: None before GW20 (not computed), Some(None) = hold.
    pub v6: Option<Option<String>>,
    /// P's solved scenarios (label, total objective), as `chip_checks` stores them.
    pub scenarios: Vec<(String, f64)>,
}
```

- [ ] **Step 1: Write the failing tests** (a `#[cfg(test)] mod tests` in `chip_calls.rs`):

```rust
    #[test]
    fn blanks_count_squad_players_of_clubs_with_all_zero_xp() {
        // club 1 plays (one player > 0), club 2 blanks (all 0 at horizon 0), club 3 has a
        // player with null xP and one with 0 -> blank
        let frame = XpFrame {
            player_key: vec![10, 11, 20, 21, 30, 31, 10],
            gw: vec![9; 7],
            gw_index: vec![9, 9, 9, 9, 9, 9, 10],
            horizon: vec![0, 0, 0, 0, 0, 0, 1],
            xp: vec![Some(0.0), Some(3.2), Some(0.0), Some(0.0), None, Some(0.0), Some(5.0)],
            p_play: None, p_play_gw: None, e_minutes: None,
        };
        let pool = vec![
            pool_player(10, 1), pool_player(11, 1), pool_player(20, 2),
            pool_player(21, 2), pool_player(30, 3), pool_player(31, 3),
        ];
        let squad_teams = [1, 1, 2, 3, 3]; // team_key of each holding
        assert_eq!(blanks_from_frame(&frame, &pool, &squad_teams), 3);
    }

    #[test]
    fn expiring_is_three_gws_or_fewer() {
        assert!(is_expiring(16, 19));
        assert!(!is_expiring(15, 19));
    }

    #[test]
    fn best_gw_from_scenario_label() {
        assert_eq!(scenario_gw("bboost@gw16"), Some(16));
        assert_eq!(scenario_gw("none"), None);
    }

    #[test]
    fn changed_compares_planner_rules_and_forced_only() {
        let a = calls(Some("bboost"), None, None);
        let mut b = a.clone();
        b.planner_gain += 1.0;
        b.scenarios.clear();
        assert!(!calls_changed(&a, &b));
        b.rules = Some("3xc".into());
        assert!(calls_changed(&a, &b));
    }
```

  `pool_player(key, team)` builds a `PoolPlayer { player_key, element_type: 3, team_key: team, price: 50 }`. `calls(planner, rules, forced)` builds a `ChipCalls` with `gw: 10`, `planner_gain: 2.0`, empty gains, open, scenarios and `v6: None`.

- [ ] **Step 2: Run them and see them fail.**

  Run: `cargo test -p fplopt-bot -j 2 chip_calls`

  Expected: compile errors.

- [ ] **Step 3: Implement.**

```rust
//! The chip suggestions in the bot's alerts (spec docs/superpowers/specs/
//! 2026-11-05-chip-alerts-design.md): the chip planner P (`/chips`'s search), the rule
//! baseline H (`fplopt_optimize::chip_rules`), the open chips with expiry, and from GW20 the
//! frozen chip-policy-2 V6 (logged only).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use fplopt_optimize::chip_rules::{H, chip_gains, last_chance, rule_chip};
use fplopt_optimize::state::PoolPlayer;
use fplopt_optimize::{OptimizerParams, PlanInput, XpFrame};

use crate::planner::{ChipOptions, UserPlan, live_params, load_terminal_values};

/// Chip policy 2's frozen V6, live variant (docs/research/2026-11-chip-policy-2.md §5).
pub const V6_SHRINK: f64 = 0.8;
pub const V6_FH_SCALE: f64 = 0.4;
pub const V6_TABLE: &str = "config/chips/terminal-values-window.json";
/// V6 is logged from the second chip set's first GW.
pub const V6_FROM_GW_INDEX: i64 = 20;
/// Expiry warning when a window's stop is this close (stop − gw_index ≤ 3: four GWs left).
pub const EXPIRY_GWS: i64 = 3;

pub fn is_expiring(gw_index: i64, stop: i64) -> bool {
    stop - gw_index <= EXPIRY_GWS
}

/// `bboost@gw16` -> 16 (a single-chip scenario's GW); None for `none` or a malformed label.
pub fn scenario_gw(label: &str) -> Option<i64> {
    label.split_once("@gw").and_then(|(_, g)| g.parse().ok())
}

/// V6's settings: the live settings, the per-window table × 0.8, Free Hit gains × 0.4.
pub fn v6_params() -> Result<OptimizerParams> {
    let params = OptimizerParams {
        chip_value_left: Some(load_terminal_values(V6_SHRINK, Path::new(V6_TABLE))?),
        fh_scale: V6_FH_SCALE,
        ..live_params()
    };
    params.validate()?;
    Ok(params)
}

/// Squad players (by club, `squad_teams`) whose club blanks the target GW: every pool player
/// of the club has xP 0 or null at horizon 0 (a v1 frame gives a blank 0; a club that plays
/// always has someone above 0).
pub fn blanks_from_frame(frame: &XpFrame, pool: &[PoolPlayer], squad_teams: &[i64]) -> i64 {
    let team_of: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.team_key)).collect();
    let mut playing: HashSet<i64> = HashSet::new();
    for i in 0..frame.len() {
        if frame.horizon[i] == 0 && frame.xp[i].unwrap_or(0.0) > 0.0 {
            if let Some(t) = team_of.get(&frame.player_key[i]) {
                playing.insert(*t);
            }
        }
    }
    squad_teams.iter().filter(|t| !playing.contains(t)).count() as i64
}

/// Whether the call shown to the user changed (planner, rules or forced chip).
pub fn calls_changed(a: &ChipCalls, b: &ChipCalls) -> bool {
    (a.planner.as_deref(), a.rules.as_deref(), a.forced.as_deref())
        != (b.planner.as_deref(), b.rules.as_deref(), b.forced.as_deref())
}

/// The calls for `plan`'s user from P's search `options` and, from GW20, V6's `v6` search.
pub fn compute_chip_calls(
    plan: &UserPlan,
    options: &ChipOptions,
    v6: Option<&ChipOptions>,
) -> Result<ChipCalls> {
    let context = &plan.context;
    let gw_of: HashMap<i64, i64> = context.gameweeks.iter().map(|(g, i)| (*i, *g)).collect();
    // P
    let objectives = &options.plans.scenario_objectives;
    let base = objectives.iter().rev().find(|(l, _)| l == "none").map_or(0.0, |(_, v)| *v);
    let best = options.plans.best();
    let planner = best.first().chip.clone();
    let planner_gain = if planner.is_some() { best.total_objective() - base } else { 0.0 };
    // H on the bot's no-chip plan
    let problem = PlanInput::from_context(
        &plan.state, &context.pool, &context.xp, &plan.rules, &plan.params,
    )?;
    let squad_teams: Vec<i64> = plan.state.holdings.iter().map(|h| h.team_key).collect();
    let blanks = blanks_from_frame(&context.xp, &context.pool, &squad_teams);
    let no_chip = crate::planner::ranked(&plan.plans)[0];
    let gains = chip_gains(
        &plan.state, &context.pool, &context.xp, &plan.rules, &problem, &plan.params,
        no_chip, blanks,
    )?;
    let rules = rule_chip(&plan.state, &plan.rules, &gains, &H)?;
    let forced = last_chance(&plan.state, &plan.rules)?;
    // open chips, in window order
    let mut windows: Vec<_> = plan.rules.chips.iter().collect();
    windows.sort_by_key(|w| (w.start, w.chip_id));
    let used: HashSet<i64> = plan.state.chips_used.iter().map(|(c, _)| *c).collect();
    let open = windows
        .into_iter()
        .filter(|w| !used.contains(&w.chip_id) && w.start <= context.gw_index && context.gw_index <= w.stop)
        .map(|w| OpenChip {
            name: w.name.clone(),
            last_gw: gw_of.get(&w.stop).copied().unwrap_or(w.stop),
            best_gw: options.single(&w.name).and_then(|(label, _)| scenario_gw(label)),
            expiring: is_expiring(context.gw_index, w.stop),
        })
        .collect();
    let v6 = v6.map(|o| o.plans.best().first().chip.clone());
    Ok(ChipCalls {
        gw: context.gw,
        planner,
        planner_gain,
        rules,
        gains,
        forced,
        open,
        v6,
        scenarios: objectives.clone(),
    })
}
```

  Check the field names against the real types:
  - `Rules.chips` items `{chip_id, name, start, stop}` (see `messages.rs::chips_text`);
  - `PlanSet.best()`, `Plan.first()`, `GwPlan.chip`;
  - `ChipOptions.single`.

  If a window with the same name opens later (the second half), it isn't open now. The filter on `start <= gw_index` handles that.

- [ ] **Step 4: Run the tests and see them pass.**

  Run: `cargo test -p fplopt-bot -j 2 chip_calls`

  Expected: 4 pass.

- [ ] **Step 5: Commit.**

```bash
git add crates/fplopt-bot/src/chip_calls.rs crates/fplopt-bot/src/lib.rs
git commit -m "feat(bot): chip calls for alerts (planner, rules H, open chips, V6 from GW20)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The chip section text (`messages::chip_section`)

**Files:**
- Modify: `crates/fplopt-bot/src/messages.rs`

The text is HTML for Telegram: `<b>` and escaped text only. Use `e()` on anything that isn't a fixed string, and use `chip_label` for names.

**Rendering rules** (spec §1–4):

1. **Headline:**
   - `forced` Some(c): `<b>Chip this week:</b> play {label(c)} now, or lose it`
   - else `planner` Some(c): `<b>Chip this week:</b> play {label(c)} ({gain:+.1} xP vs keeping it)`
     - if c is `freehit` or `wildcard`, append ` · see /chips for the chip squad`
   - else: `<b>Chip this week:</b> hold`
2. **Rules line** (indented two spaces):
   - H's call equals the headline chip (the planner's, or forced if set):
     - `  Simple rules agree: {reason}`
     - when both hold: `  Simple rules agree: hold`
   - Otherwise: `  Simple rules: {play {label} | hold} ({reason})`
   - Reasons:
     - `3xc`: `captain xP {captain_xp:.1} ≥ 8.3`
     - `bboost`: `bench xP {bench_xp:.1} ≥ 10.9`
     - `freehit`: `{blanks} squad players without a match ≥ 4`
     - `wildcard`: `wildcard gain {gain_wc:.1} ≥ 12.9`
     - a last-chance play: `last chance before the window closes`
     - hold: `captain xP {:.1}, bench xP {:.1}`
3. **Unused chips:** `Unused chips: ` followed by the open chips joined with `, `. Each is `{label}` plus a parenthesised list:
   - `expires after GW{last}` if expiring, else `until GW{last}`;
   - then `best GW: GW{best}` if it differs from this GW, or `playing now` if the headline plays it.

   If no chips are open: `Unused chips: none`.
4. **Evidence:** `Evidence: on 2023/24 and 2024/25 this planner scored +4.2 pts/GW over playing no chips; not proven better than the simple rules.`
5. **Failure:** `chip_failure_section()` returns `<b>Chip this week:</b> chip check failed this time: try /chips`.

The lines are joined with `\n`, and a blank line follows the section (the caller adds it).

- [ ] **Step 1: Write the failing tests** (in the existing `mod tests` of `messages.rs`). Build `ChipCalls` values directly:

```rust
    fn chip_calls(planner: Option<&str>, rules: Option<&str>, forced: Option<&str>) -> ChipCalls {
        ChipCalls {
            gw: 13,
            planner: planner.map(Into::into),
            planner_gain: 7.44,
            rules: rules.map(Into::into),
            gains: [("captain_xp", 7.0), ("bench_xp", 11.6), ("blanks", 0.0), ("gain_wc", 3.0)]
                .iter().map(|(k, v)| (k.to_string(), Some(*v))).collect(),
            forced: forced.map(Into::into),
            open: vec![
                OpenChip { name: "freehit".into(), last_gw: 19, best_gw: Some(16), expiring: false },
                OpenChip { name: "bboost".into(), last_gw: 19, best_gw: Some(13), expiring: false },
            ],
            v6: None,
            scenarios: vec![],
        }
    }

    #[test]
    fn chip_section_play_and_agree() {
        assert_eq!(
            chip_section(&chip_calls(Some("bboost"), Some("bboost"), None)),
            "<b>Chip this week:</b> play Bench Boost (+7.4 xP vs keeping it)\n\
             \x20 Simple rules agree: bench xP 11.6 ≥ 10.9\n\
             Unused chips: Free Hit (until GW19, best GW: GW16), Bench Boost (until GW19, playing now)\n\
             Evidence: on 2023/24 and 2024/25 this planner scored +4.2 pts/GW over playing no chips; \
             not proven better than the simple rules."
        );
    }

    #[test]
    fn chip_section_hold_and_differ() {
        let text = chip_section(&chip_calls(None, Some("bboost"), None));
        assert!(text.starts_with("<b>Chip this week:</b> hold\n  Simple rules: play Bench Boost (bench xP 11.6 ≥ 10.9)\n"));
    }

    #[test]
    fn chip_section_forced_and_expiring() {
        let mut calls = chip_calls(None, Some("freehit"), Some("freehit"));
        calls.open[0].expiring = true;
        let text = chip_section(&calls);
        assert!(text.starts_with("<b>Chip this week:</b> play Free Hit now, or lose it\n  Simple rules agree: last chance before the window closes\n"));
        assert!(text.contains("Free Hit (expires after GW19, playing now)"));
    }

    #[test]
    fn chip_section_free_hit_points_to_chips() {
        let text = chip_section(&chip_calls(Some("freehit"), None, None));
        assert!(text.starts_with("<b>Chip this week:</b> play Free Hit (+7.4 xP vs keeping it) · see /chips for the chip squad\n"));
        assert!(text.contains("\n  Simple rules: hold (captain xP 7.0, bench xP 11.6)\n"));
    }
```

  `+7.4`: `planner_gain` 7.44 formatted with `{:+.1}`. When a call is forced, the rules' reason is the last-chance text. `rules == forced` always holds then, because `rule_chip` returns the last chance first.

- [ ] **Step 2: Run them and see them fail.**

  Run: `cargo test -p fplopt-bot -j 2 chip_section`

- [ ] **Step 3: Implement** `pub fn chip_section(calls: &ChipCalls) -> String` and `pub fn chip_failure_section() -> String` in `messages.rs`, following the rules above. Import `crate::chip_calls::ChipCalls` (and `OpenChip` in the tests).

- [ ] **Step 4: Run the tests and see them pass.**

  Run: `cargo test -p fplopt-bot -j 2 chip_section`

- [ ] **Step 5: Commit.**

```bash
git add crates/fplopt-bot/src/messages.rs
git commit -m "feat(bot): chip section text for alerts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The `chip_calls` table

**Files:**
- Modify: `crates/fplopt-bot/src/store.rs` (the schema string near the `chip_checks` table at line ~86, the methods near `record_chip_check` at ~512)

- [ ] **Step 1: Write the failing test** (in `store.rs`'s `mod tests`, next to the `chip_checks` round trip at ~line 800; reuse its setup for a user):

```rust
    #[test]
    fn chip_calls_round_trip() {
        let (users, uid) = store_with_user(); // the helper the chip_checks test uses
        let gains = r#"{"bench_xp": 11.6, "blanks": 0.0}"#;
        users
            .record_chip_call(&ChipCallRow {
                user_id: uid, season: 2026, gw: 13, kind: "alert24".into(),
                context_id: Some("ctx".into()), planner_call: "bboost".into(),
                planner_gain: Some(7.4), rules_call: "bboost".into(), forced: None,
                v6_call: None, gains: gains.into(), scenarios: "{}".into(), error: None,
                created_at: String::new(),
            })
            .unwrap();
        let rows = users.chip_calls(uid).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].planner_call, "bboost");
        assert_eq!(rows[0].gains, gains);
        assert!(!rows[0].created_at.is_empty());
        assert_eq!(users.chip_call(uid, 2026, 13, "alert24").unwrap().unwrap().rules_call, "bboost");
        assert!(users.chip_call(uid, 2026, 13, "alert2").unwrap().is_none());
    }
```

  If the `chip_checks` test builds its store differently, use its setup and keep the assertions.

- [ ] **Step 2: Run it and see it fail.**

  Run: `cargo test -p fplopt-bot -j 2 chip_calls_round_trip`

- [ ] **Step 3: Implement.** Add this to the schema string (`CREATE TABLE IF NOT EXISTS`, so existing databases gain the table on open):

```sql
CREATE TABLE IF NOT EXISTS chip_calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    kind TEXT NOT NULL,
    context_id TEXT,
    created_at TEXT NOT NULL,
    planner_call TEXT NOT NULL,
    planner_gain REAL,
    rules_call TEXT NOT NULL,
    forced TEXT,
    v6_call TEXT,
    gains TEXT NOT NULL,
    scenarios TEXT NOT NULL,
    error TEXT
);
```

  Add `pub struct ChipCallRow` with the columns above, all `pub`. Call and chip fields are `String` (`"none"` for hold), and the optional columns are `Option`. Then add:
  - `record_chip_call(&self, row: &ChipCallRow)`: inserts with `created_at = now()`, ignoring `row.created_at`.
  - `chip_calls(&self, user_id) -> Vec<ChipCallRow>`: ordered by `call_id`.
  - `chip_call(&self, user_id, season, gw, kind) -> Option<ChipCallRow>`: the newest by `call_id`.

  Follow `record_chip_check`/`chip_checks` for the `self.with(|db| …)` pattern. A store test probably checks that the schema matches Python's. If so, it compares table lists, and the new table must be added to its expected list (or excluded) with a comment: "Rust-only table (chip alerts, 2026-11-05)".

- [ ] **Step 4: Run the store tests.**

  Run: `cargo test -p fplopt-bot -j 2 store`

  Expected: pass, including any schema test you updated.

- [ ] **Step 5: Commit.**

```bash
git add crates/fplopt-bot/src/store.rs
git commit -m "feat(bot): chip_calls table (alert chip calls; the chip-policy-2 shadow log)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Alerts carry the section; replay flag; CLI flag; integration test

**Files:**
- Modify: `crates/fplopt-bot/src/service.rs` (`BotService` struct ~185, `new` ~197, `alert_user` ~716)
- Modify: `crates/fplopt-bot/src/replay.rs` (~92–115)
- Modify: `crates/fplopt-cli/src/bot.rs` (~90)
- Create: `crates/fplopt-bot/tests/chip_alerts.rs`

**Behaviour:**
- `BotService` gets two new fields:
  - `pub chip_alerts: bool`, false in `new`;
  - `pub v6_planner: ChipPlanner`. In `new` it's a closure that calls `chip_options(fpl, entry, context, ft, &v6, made)`, where `v6 = v6_params()`, read once. If `v6_params()` fails (no table), the closure returns that error.
- In `alert_user`, after `make_plan` succeeds and only when `self.chip_alerts`:
  1. Run `chip_step(user, &plan, kind) -> Option<ChipCalls>`:
     - `ft = self.users.free_transfers(...)`, `made = self.users.transfers(...)`;
     - `options = (self.chip_planner)(self.fpl.as_ref(), entry_id, &plan.context, ft, &made)`;
     - on `alert24` with `plan.context.gw_index >= V6_FROM_GW_INDEX`, `v6 = (self.v6_planner)(...)`. A V6 error is logged (`warn!`) and V6 counts as not computed;
     - `compute_chip_calls(&plan, &options, v6.as_ref())`.

     Any error returns `None`, and the error is logged with `warn!`.
  2. **Section text:**
     - `alert24`: `chip_section(&calls)` or `chip_failure_section()`.
     - `alert2`: read `self.users.chip_call(user_id, season, gw, "alert24")`. Show the section only if this run's calls exist and differ from the logged ones (compare the planner, rules and forced strings, with `"none"` for None). Otherwise show none.
  3. **Silence:** today `alert2` stays silent when `changes_text` is empty. It now stays silent only if the changes are empty **and** no chip section is shown. If the chip changed but the transfers didn't, send `recommendation_text(&plan, Some(&format!("GW{} update", context.gw)), &changes)` with the section in front.
  4. **The message:** `format!("{section}\n\n{text}")` when there's a section.
  5. **Record** a `chip_calls` row for every alert run with `chip_alerts` on: sent, silent, or failed (`error = Some(..)`, calls `"none"`, `gains = "{}"`, `scenarios = "{}"`).
     - Serialize `gains` with `fplopt_core::pyjson::dumps` (or `serde_json::to_string`) on the `BTreeMap`.
     - Serialize `scenarios` with the same `ordered_object` used for `chip_checks`. Make it `pub(crate)` if needed.
     - Write the row right after the alert's `recommendations` row.
- `replay.rs`:
  - `service.chip_alerts = script["chip_alerts"].as_bool().unwrap_or(false);`
  - when it is true, set `service.v6_planner` from `params_with(v6_params()?, script.get("v6_params")…)`, in the same way `chip_params` is handled.
  - Update the module doc: "`chip_alerts` (default false): alerts carry the chip section (Rust only; not in Python's golden)".
- `crates/fplopt-cli/src/bot.rs`: after `BotService::new(...)` at line ~90, set `service.chip_alerts = true;`. This builder serves both `bot run` and `live tick`. If they're built separately, set it in both. A one-line comment cites the spec.

- [ ] **Step 1: Write the failing integration test** `crates/fplopt-bot/tests/chip_alerts.rs`.

  Base it on `crates/fplopt-bot/tests/golden.rs`, which shows how the synthetic league in `tests/fixtures/rust_golden/bot/` is copied to a temp dir and replayed. Read `script.json` there to find the alert steps: steps with `"live": "tick"` or similar, and a `now`.

  The test:
  - Loads `script.json` and sets `"chip_alerts": true`.
  - Replays it into a temp state dir with `fplopt_bot::replay::replay(&script, &state, &fpl)`.
  - In the results, finds each alert message sent to a registered user. Results hold the sent texts; see how `golden.rs` compares `results.json`.
  - Asserts:
    - every `alert24` text starts with `<b>Chip this week:</b>`;
    - it contains `Evidence: on 2023/24 and 2024/25`;
    - after the section it carries the same recommendation text as Python's golden for that step. So `results.json`'s text for the step equals the Rust text with the section and the blank line after it cut off.
  - Opens `bot.sqlite` with `UserStore::new` and asserts that `chip_calls(user)` has one row per `alert24` run, with `kind == "alert24"`.
  - Asserts that a second replay with `chip_alerts` absent gives exactly `results.json`. That's the existing golden, unchanged.

  Synthetic league or no alert step: if the golden has no `alert24` step, add a minimal script in the test itself (built as JSON in code). It registers the first user from `script.json`, sets `now` to 23 h before the next deadline in the fixture bootstrap, and runs a tick. Copy the step shapes from `script.json`.

- [ ] **Step 2: Run it and see it fail.**

  Run: `cargo test -p fplopt-bot -j 2 --test chip_alerts`

  Expected: FAIL (no chip section yet).

- [ ] **Step 3: Implement the behaviour above** in `service.rs`, `replay.rs` and `fplopt-cli/src/bot.rs`.

- [ ] **Step 4: Run the bot and CLI tests.**

  Run:
  - `cargo test -p fplopt-bot -j 2`
  - `cargo test -p fplopt-cli -j 2`

  Expected: all pass, including `tests/golden.rs` unchanged and the new `chip_alerts`.

- [ ] **Step 5: Run the whole workspace and clippy.**

  Run:
  - `cargo clippy --workspace --all-targets -j 2 -- -D warnings`
  - `cargo test --workspace -j 2`

  Expected: clean, all pass.

- [ ] **Step 6: Commit.**

```bash
git add crates/fplopt-bot/src/service.rs crates/fplopt-bot/src/replay.rs crates/fplopt-cli/src/bot.rs crates/fplopt-bot/tests/chip_alerts.rs
git commit -m "feat(bot): alerts carry the chip section and log chip_calls (off in bot replay)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Docs

**Files:**
- Modify: `docs/PLAN.md`:
  - §8, the *Built in Phase 7* paragraph that says the bot's plans and alerts "play no chips";
  - §12, the decision log table: add a row at the end.
- Modify: `CLAUDE.md`, the `fplopt-rs bot run` line.
- Modify: `docs/superpowers/specs/2026-11-05-chip-alerts-design.md`, the Logging section. It says "written in the same transaction"; change that to "written right after the alert's row".

- [ ] **Step 1: PLAN §8.** After "…until a chip policy passes its own test (§11).", add:

  "**2026-11-05:** alerts suggest chips (the user's decision). Each `alert24` opens with a chip section: the chip planner's call (the `/chips` search), the rule baseline H's call, unused chips with expiry warnings from four GWs before a window closes ("play now, or lose it" when H's last-chance rule fires), and the evidence line. `alert2` repeats it only if the call changed. Calls are logged in `chip_calls`, with the frozen chip-policy-2 V6's call from GW20, which is that study's shadow log. The user still plays chips in FPL. `/plan`'s transfer plan stays the no-chip plan. Spec `docs/superpowers/specs/2026-11-05-chip-alerts-design.md`."

- [ ] **Step 2: PLAN §12 row.**

  `| 2026-11-05 | Alerts suggest chips: the chip planner's call next to the rule baseline H's, with expiry warnings; logged in chip_calls (V6 from GW20). The user plays chips. | The chip study's validate run: the planner beat no chips (+4.18/GW, p 0.025) but not H at the pre-registered threshold; a season without chips understates the system (holdout 2,317 with no chips vs ~2,400 for the top 10k). A suggestion tested against no chips, not a policy that passed the full gate. |`

- [ ] **Step 3: CLAUDE.md.** In the `fplopt-rs bot run` bullet, after "long polling over the Bot API (…)", add:

  "; alerts carry a chip section (planner and rule calls, expiry; `chip_calls` table; `chip_alerts`, on in `bot run`/`live tick`, off in `bot replay` unless the script sets it, so the Python golden still matches)".

- [ ] **Step 4: Spec wording fix** as listed above.

- [ ] **Step 5: Commit.**

```bash
git add docs/PLAN.md CLAUDE.md docs/superpowers/specs/2026-11-05-chip-alerts-design.md
git commit -m "docs: chip suggestions in alerts (PLAN §8, §12, CLAUDE.md)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## After the tasks

- Final review of the whole branch, then:
  - `cargo test --workspace -j 2`
  - `cargo clippy --workspace --all-targets -j 2 -- -D warnings`
  - `uv run python dev/bot_parity.py --help`, to confirm the gate still runs with the replay default off. A full run needs a server state copy, so it's not done here.
- No push, PR or server deploy without the user's OK. The deploy is: merge to `main`, then on the server `git pull`, `cargo build --release`, and restart the units. That happens between two GWs.
