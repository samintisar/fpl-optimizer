# Bot Messaging UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Rust Telegram bot's messages in plain FPL words, with reasoning a casual player can check. That reasoning covers each transfer's points, prices and fixtures, the captain's points, and the gain over saving the transfer.

**Architecture:** A new `summary.rs` gathers the facts of a plan into plain structs (no text). `messages.rs` turns those structs into Telegram HTML (no arithmetic). The plan context gains an optional `fixtures` frame. The bot golden is re-recorded from Rust (texts no longer match Python's).

**Tech Stack:** Rust (`crates/fplopt-bot`), the Parquet tables in `fplopt-build`, the `fplopt-store` as-of view, and Python only for the golden check and `dev/bot_parity.py`.

**Spec:** `docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md`.

## Ground rules for the implementer

- Work in this worktree. Run cargo from the repo root.
- After any task that changes a reply text, re-record the golden (Task 1 adds the switch) and read the diff of `tests/fixtures/rust_golden/bot/results.json`. Every changed line must be one this task meant to change.
- Checks before each commit:
  ```bash
  cargo test -p fplopt-bot
  ```
  ```bash
  cargo clippy -p fplopt-bot --all-targets -- -D warnings
  ```
  ```bash
  cargo fmt --all
  ```
- Names, reasons and anything from FPL go through `crate::util::escape` (imported as `e`). Only `<b>` tags are written raw.
- The golden league's players are called `P100109` and so on, and its context has no `fixtures` frame. So golden texts show no fixtures lines. That is expected.
- The golden script plans with `horizon: 1`. That's why golden texts say "this GW" rather than "over the next 6 GWs".

## File map

| File | Change |
| --- | --- |
| `crates/fplopt-bot/src/context.rs` | `Fixture`; `PlanContext.fixtures: Option<Arc<Vec<Fixture>>>`; read `fixtures.parquet` when present |
| `crates/fplopt-bot/src/builder.rs` | `fixtures_table`; `NewContext.fixtures`; write the frame |
| `crates/fplopt-bot/src/summary.rs` (new) | `horizon_points`, `gw_fixtures`, `tied`, `PlanSummary`, `plan_summary`, `PlanChange`, `plan_change` |
| `crates/fplopt-bot/src/messages.rs` | `render_plan`, `plan_text`, `update_text`, `UNCHANGED`, new `chip_section`/`chip_failure_section`/`chips_text`/`squad_text`; remove `recommendation_text`, `changes_text`, `prices_line` |
| `crates/fplopt-bot/src/service.rs` | new texts for every user command; use `plan_text`/`plan_change`/`update_text` |
| `crates/fplopt-bot/src/lib.rs` | `pub mod summary;` |
| `crates/fplopt-bot/tests/golden.rs` | `FPLOPT_UPDATE_GOLDEN=1` re-records `results.json` and `db.json` |
| `crates/fplopt-bot/tests/chip_alerts.rs` | new header/marker strings |
| `crates/fplopt-bot/tests/summary.rs` (new) | `plan_summary` on the golden league |
| `tests/test_rust_golden.py` | skip `test_bot_golden_is_current` |
| `dev/bot_parity.py` | compare rows and decisions, not texts |
| `CLAUDE.md` | batch 5 line |

---

### Task 1: Golden re-record switch; retire Python text parity

**Files:**
- Modify: `crates/fplopt-bot/tests/golden.rs`
- Modify: `tests/test_rust_golden.py:222`
- Modify: `dev/bot_parity.py` (`check`)

- [ ] **Step 1: Add the re-record switch to `golden.rs`**

In `bot_session_matches_python`, right after `let results = fplopt_bot::replay::replay(...)`, insert:

```rust
    if std::env::var("FPLOPT_UPDATE_GOLDEN").is_ok_and(|v| v == "1") {
        // The bot's texts are Rust's own since 2026-11-11 (spec
        // docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md): re-record.
        let write = |name: &str, value: &Value| {
            let text = serde_json::to_string_pretty(value).unwrap() + "\n";
            std::fs::write(golden.join(name), text).unwrap();
        };
        write("results.json", &results);
        write("db.json", &dump_db(&state.join("bot.sqlite")));
        let _ = std::fs::remove_dir_all(&state);
        return;
    }
```

Rename the test to `bot_session_matches_golden`. Replace the module doc's first sentence:

```rust
//! The synthetic bot golden (`tests/fixtures/rust_golden/bot/`: script, state and FPL
//! responses written by Python's `tests/bot_replay.py`; `results.json` and `db.json`
//! recorded from this replay with `FPLOPT_UPDATE_GOLDEN=1`): the scripted session must give
//! the recorded replies and alert texts and the same `bot.sqlite` rows.
```

In the assertion message, change `python:` to `golden:`.

- [ ] **Step 2: Re-record once with the current texts and check that nothing but formatting moved**

```bash
FPLOPT_UPDATE_GOLDEN=1 cargo test -p fplopt-bot --test golden bot_session_matches_golden
```
```bash
cargo test -p fplopt-bot --test golden
```
```bash
uv run python -c "import json;a=json.load(open('tests/fixtures/rust_golden/bot/results.json',encoding='utf-8'));print(len(a))"
```

Expected: PASS, then `55`. `git diff --stat tests/fixtures/rust_golden/bot` shows `results.json` and `db.json` changed by JSON layout (indent 2 vs Python's 1). Confirm `results.json`'s content is the same:

```bash
git show HEAD:tests/fixtures/rust_golden/bot/results.json > "$TEMP/old-results.json"
```
```bash
uv run python -c "import json,os;a=json.load(open(os.environ['TEMP']+'/old-results.json',encoding='utf-8'));b=json.load(open('tests/fixtures/rust_golden/bot/results.json',encoding='utf-8'));print('same' if a==b else 'DIFFERENT')"
```

Expected: `same`. If `db.json` has float differences in `plans` (the old golden.rs accepted them via `same_plans`), that's fine: from now on it is Rust's own output. Don't use `git stash` here: the stash is shared with other worktrees.

- [ ] **Step 3: Skip the Python golden check**

In `tests/test_rust_golden.py`, decorate `test_bot_golden_is_current`:

```python
@pytest.mark.skip(
    reason="results.json and db.json are recorded from the Rust bot since 2026-11-11 "
    "(FPLOPT_UPDATE_GOLDEN=1 cargo test -p fplopt-bot --test golden); its texts are its own"
)
def test_bot_golden_is_current(tmp_path):
```

Add `import pytest` at the top if it's missing.

- [ ] **Step 4: Stop `dev/bot_parity.py` comparing texts**

Add above `def check`:

```python
def _without_texts(result: Any) -> Any:
    """A step result with reply and alert texts blanked: the Rust bot's texts are its own
    since 2026-11-11 (docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md), so the
    gate compares who got a message, outcomes, rows and decisions."""
    if not isinstance(result, dict):
        return result
    out = dict(result)
    for key in ("replies", "texts"):
        if key in out:
            out[key] = [[chat, None] for chat, _ in out[key]]
    return out
```

In `check`, replace `py, rs = session["python"], session["rust"]` with:

```python
    py = [_without_texts(r) for r in session["python"]]
    rs = [_without_texts(r) for r in session["rust"]]
```

Replace `rest = {k for k in a if a.get(k) != b.get(k)}` with:

```python
        rest = {k for k in a if k != "message" and a.get(k) != b.get(k)}
```

Check that `Any` is imported from `typing` (it is used in the file; add it if not). Update the module docstring line "every step's replies and alert texts identical, byte for byte;" to "every step's replies and alerts going to the same chats with the same outcomes (texts are the Rust bot's own since 2026-11-11);".

- [ ] **Step 5: Run the checks**

```bash
cargo test -p fplopt-bot
```
```bash
uv run pytest tests/test_rust_golden.py -q
```
```bash
uv run ruff check dev/bot_parity.py tests/test_rust_golden.py
```

Expected: all pass, with 1 skipped in pytest.

- [ ] **Step 6: Commit**

```bash
git add crates/fplopt-bot/tests/golden.rs tests/fixtures/rust_golden/bot tests/test_rust_golden.py dev/bot_parity.py
```
```bash
git commit -m "bot: golden recorded from Rust; parity gate stops comparing texts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Fixtures in the plan context

**Files:**
- Modify: `crates/fplopt-bot/src/context.rs`
- Modify: `crates/fplopt-bot/src/builder.rs`

- [ ] **Step 1: Write the failing test (context.rs)**

Append to `crates/fplopt-bot/src/context.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_build::table::{Column, Table, write_parquet};

    fn golden_context_dir() -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rust_golden/bot/state/context");
        let id = std::fs::read_dir(&root).unwrap().next().unwrap().unwrap().file_name();
        root.join(id)
    }

    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }

    #[test]
    fn a_context_without_fixtures_loads_with_none() {
        let context = read_context(&golden_context_dir()).unwrap();
        assert!(context.fixtures.is_none());
    }

    #[test]
    fn fixtures_are_read_when_present() {
        let dir = std::env::temp_dir().join(format!("fplopt-ctx-fixtures-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        copy(&golden_context_dir(), &dir);
        let table = Table::new()
            .with("gw", Column::I64(vec![6, 6, 7]))
            .with("team_key", Column::I64(vec![1, 1, 1]))
            .with("opponent", Column::Str(vec!["BRE".into(), "LEE".into(), "WOL".into()]))
            .with("home", Column::Bool(vec![true, false, true]));
        write_parquet(&table, &dir.join("fixtures.parquet")).unwrap();
        let context = read_context(&dir).unwrap();
        let fixtures = context.fixtures.unwrap();
        assert_eq!(fixtures.len(), 3);
        assert_eq!(
            fixtures[1],
            Fixture { gw: 6, team_key: 1, opponent: "LEE".into(), home: false }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
```

- [ ] **Step 2: Run it to see it fail**

```bash
cargo test -p fplopt-bot --lib context::tests
```

Expected: compile error, `no field fixtures` / `cannot find struct Fixture`.

- [ ] **Step 3: Implement it in context.rs**

Add after `ContextPlayer`:

```rust
/// A row of the context's optional `fixtures` frame: one of club `team_key`'s games in GW
/// `gw` against `opponent` (FPL short name), at home or away. A double GW has two rows, a
/// blank none. Contexts written before 2026-11-11 have no frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fixture {
    pub gw: i64,
    pub team_key: i64,
    pub opponent: String,
    pub home: bool,
}
```

Add this field to `PlanContext`, after `prices_at`:

```rust
    /// The fixtures frame, sorted by (gw, team_key) as written; None in older contexts.
    pub fixtures: Option<Arc<Vec<Fixture>>>,
```

In `read_context`, before `Ok(PlanContext {`:

```rust
    let fixtures_path = directory.join("fixtures.parquet");
    let fixtures = if fixtures_path.is_file() {
        let table = read_parquet(&fixtures_path, None)?;
        let opponents = table.opt_str("opponent")?;
        let home = table.opt_bool("home")?;
        let rows: Vec<Fixture> = int_column(&table, "gw", "fixtures")?
            .into_iter()
            .zip(int_column(&table, "team_key", "fixtures")?)
            .zip(opponents)
            .zip(home)
            .map(|(((gw, team_key), opponent), home)| Fixture {
                gw,
                team_key,
                opponent: opponent.unwrap_or_default(),
                home: home.unwrap_or(false),
            })
            .collect();
        Some(Arc::new(rows))
    } else {
        None
    };
```

Add `fixtures,` to the struct literal. Update the module doc's frame list: "... `gameweeks` (gw, gw_index) and, since 2026-11-11, `fixtures` (gw, team_key, opponent, home; optional)."

`PlanContext` is built in other places too (tests, `chip_calls.rs`, `replay.rs`). Find them and add `fixtures: None` wherever a struct literal fails to compile:

```bash
cargo build -p fplopt-bot --all-targets 2>&1 | grep -B2 "missing field .fixtures."
```

- [ ] **Step 4: Run the tests**

```bash
cargo test -p fplopt-bot --lib context::tests
```

Expected: PASS (2 tests).

- [ ] **Step 5: Write the failing builder test**

Append to `mod tests` in `crates/fplopt-bot/src/builder.rs`:

```rust
    #[test]
    fn fixture_rows_are_both_sides_of_each_game_from_the_planned_gw() {
        let schedule = Table::new()
            .with("season", Column::I64(vec![2026, 2026, 2026, 2025]))
            .with("gw", Column::OptI64(vec![Some(11), Some(12), None, Some(12)]))
            .with("fixture_key", Column::I64(vec![1, 2, 3, 4]))
            .with("home_team_key", Column::I64(vec![10, 20, 10, 10]))
            .with("away_team_key", Column::I64(vec![20, 30, 30, 20]));
        let teams = Table::new()
            .with("team_key", Column::I64(vec![10, 20, 30]))
            .with("short_name", Column::Str(vec!["ARS".into(), "BRE".into(), "LEE".into()]));
        let rows = fixture_rows(&schedule, &teams, 2026, 12).unwrap();
        // Only GW12 of 2026 (GW11 is past, the unscheduled game has no GW, 2025 is another
        // season): BRE (20) at home to LEE (30), both sides.
        assert_eq!(rows.i64("gw").unwrap(), &[12, 12]);
        assert_eq!(rows.i64("team_key").unwrap(), &[20, 30]);
        assert_eq!(rows.str("opponent").unwrap(), &["LEE".to_string(), "BRE".to_string()]);
        assert_eq!(rows.bool("home").unwrap(), &[true, false]);
    }
```

- [ ] **Step 6: Run it to see it fail**

```bash
cargo test -p fplopt-bot --lib builder::tests::fixture_rows
```

Expected: compile error, `cannot find function fixture_rows`.

- [ ] **Step 7: Implement it in builder.rs**

Add `"fixtures"` to `FRAMES`, so it is `const FRAMES: [&str; 6] = ["xp", "pool", "players", "prices", "gameweeks", "fixtures"];`. Add `pub fixtures: Table,` to `NewContext`, and in `frame()` add `"fixtures" => &self.fixtures,` before the `_` arm.

Add above `build_context`:

```rust
/// The `fixtures` frame (context docs): both sides of every scheduled game of `season` in
/// GWs ≥ `gw`, sorted by (gw, team_key, fixture_key), from the as-of `schedule` and
/// `team_dim`'s short names.
pub fn fixture_rows(schedule: &Table, teams: &Table, season: i64, gw: i64) -> Result<Table> {
    let short: std::collections::HashMap<i64, String> = teams
        .i64("team_key")?
        .iter()
        .copied()
        .zip(teams.opt_str("short_name")?.into_iter().map(Option::unwrap_or_default))
        .collect();
    let (seasons, gws) = (schedule.i64("season")?, schedule.opt_i64("gw")?);
    let fixture_keys = schedule.i64("fixture_key")?;
    let (home, away) = (schedule.i64("home_team_key")?, schedule.i64("away_team_key")?);
    let mut rows: Vec<(i64, i64, i64, String, bool)> = Vec::new();
    for i in 0..schedule.height() {
        let Some(g) = gws[i] else { continue };
        if seasons[i] != season || g < gw {
            continue;
        }
        let name = |k: i64| short.get(&k).cloned().unwrap_or_default();
        rows.push((g, home[i], fixture_keys[i], name(away[i]), true));
        rows.push((g, away[i], fixture_keys[i], name(home[i]), false));
    }
    rows.sort_by_key(|r| (r.0, r.1, r.2));
    Ok(Table::new()
        .with("gw", Column::I64(rows.iter().map(|r| r.0).collect()))
        .with("team_key", Column::I64(rows.iter().map(|r| r.1).collect()))
        .with("opponent", Column::Str(rows.iter().map(|r| r.3.clone()).collect()))
        .with("home", Column::Bool(rows.iter().map(|r| r.4).collect())))
}
```

In `build_context`, before `let xp = ...`:

```rust
    let schedule = view.table(
        "schedule",
        Some(&["season", "gw", "fixture_key", "home_team_key", "away_team_key"]),
    )?;
    let teams = view.table("team_dim", Some(&["team_key", "short_name"]))?;
    let fixtures = fixture_rows(&schedule, &teams, season, gw)?;
```

Add `fixtures,` to the `NewContext` literal. If `view.table` returns a type other than `fplopt_store::Table` (= `fplopt_build::table::Table`), adjust the import. The `Table`/`Column` imports at the top of builder.rs come from `fplopt_store`, which re-exports `fplopt_build::table`.

- [ ] **Step 8: Run the tests and clippy**

```bash
cargo test -p fplopt-bot
```
```bash
cargo clippy -p fplopt-bot --all-targets -- -D warnings
```

Expected: PASS. Python's context reader reads only its five named frames (`src/fplopt/bot/context.py` `_FRAMES`), so a Rust context with `fixtures.parquet` still loads in a rollback.

- [ ] **Step 9: Commit**

```bash
git add crates/fplopt-bot/src
```
```bash
git commit -m "bot: fixtures frame in the plan context" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Summary helpers (`horizon_points`, `gw_fixtures`, `tied`)

**Files:**
- Create: `crates/fplopt-bot/src/summary.rs`
- Modify: `crates/fplopt-bot/src/lib.rs`
- Modify: `crates/fplopt-bot/src/messages.rs` (remove `tied`)

- [ ] **Step 1: Write the failing tests**

Create `crates/fplopt-bot/src/summary.rs`:

```rust
//! The facts of a plan for the bot's messages (spec
//! docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md): plain data, no text.
//! `messages` words them.

use fplopt_optimize::{Plan, XpFrame};

use crate::context::Fixture;

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> XpFrame {
        XpFrame {
            player_key: vec![1, 1, 1, 1, 2],
            gw: vec![6, 7, 7, 8, 6],
            gw_index: vec![6, 7, 7, 8, 6],
            horizon: vec![0, 1, 1, 2, 0],
            xp: vec![Some(2.0), Some(3.0), Some(1.5), Some(f64::NAN), None],
            p_play: None,
            p_play_gw: None,
            e_minutes: None,
        }
    }

    #[test]
    fn horizon_points_add_a_players_rows_in_the_gws() {
        // GW7 is a double (two rows); NaN and missing count 0.
        assert_eq!(horizon_points(&frame(), 1, &[6, 7]), Some(6.5));
        assert_eq!(horizon_points(&frame(), 1, &[6, 7, 8]), Some(6.5));
        assert_eq!(horizon_points(&frame(), 2, &[6]), Some(0.0));
        assert_eq!(horizon_points(&frame(), 3, &[6]), None);
    }

    #[test]
    fn gw_fixtures_group_games_per_gw_and_mark_blanks() {
        let f = |gw, team_key, opponent: &str, home| Fixture {
            gw,
            team_key,
            opponent: opponent.into(),
            home,
        };
        let fixtures =
            vec![f(6, 1, "BRE", true), f(6, 1, "LEE", false), f(8, 1, "WOL", true), f(6, 2, "X", true)];
        assert_eq!(
            gw_fixtures(&fixtures, 1, &[6, 7, 8]),
            vec![
                (6, vec![("BRE".to_string(), true), ("LEE".to_string(), false)]),
                (7, vec![]),
                (8, vec![("WOL".to_string(), true)]),
            ]
        );
    }
}
```

Add `pub mod summary;` to `crates/fplopt-bot/src/lib.rs` (keep the list alphabetical, after `service`).

- [ ] **Step 2: Run them to see them fail**

```bash
cargo test -p fplopt-bot --lib summary
```

Expected: compile error, `cannot find function horizon_points`.

- [ ] **Step 3: Implement**

Insert above `#[cfg(test)]` in `summary.rs`:

```rust
/// A player's expected points summed over `gws` (every row of a player-GW counts, so a
/// double adds both games; NaN or missing counts 0); None if the frame has no row for him in
/// those GWs.
pub fn horizon_points(xp: &XpFrame, player_key: i64, gws: &[i64]) -> Option<f64> {
    let mut total = None;
    for i in 0..xp.len() {
        if xp.player_key[i] == player_key && gws.contains(&xp.gw[i]) {
            let x = xp.xp[i].filter(|v| !v.is_nan()).unwrap_or(0.0);
            total = Some(total.unwrap_or(0.0) + x);
        }
    }
    total
}

/// A club's games per GW in `gws` (in that order): (gw, [(opponent, home)]), empty for a
/// blank.
pub fn gw_fixtures(fixtures: &[Fixture], team_key: i64, gws: &[i64]) -> Vec<(i64, Vec<(String, bool)>)> {
    gws.iter()
        .map(|&gw| {
            let games = fixtures
                .iter()
                .filter(|f| f.gw == gw && f.team_key == team_key)
                .map(|f| (f.opponent.clone(), f.home))
                .collect();
            (gw, games)
        })
        .collect()
}

/// Plans whose objectives differ by less than the solver's gap: equally good as far as the
/// solver can tell (PLAN §8 live finding).
pub fn tied(a: &Plan, b: &Plan) -> bool {
    let gap = a.mip_gap.max(b.mip_gap) * a.total_objective().abs();
    (a.total_objective() - b.total_objective()).abs() <= gap
}
```

In `messages.rs`, delete `fn tied` and add `use crate::summary::tied;`. (`Plan`, `GwPlan` and `XpFrame` are re-exported at `fplopt_optimize`'s root.)

- [ ] **Step 4: Run the tests**

```bash
cargo test -p fplopt-bot
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-bot/src
```
```bash
git commit -m "bot: summary helpers (horizon points, fixtures per GW, ties)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `PlanSummary` from a `UserPlan`

**Files:**
- Modify: `crates/fplopt-bot/src/summary.rs`
- Create: `crates/fplopt-bot/tests/summary.rs`

- [ ] **Step 1: Write the failing integration test**

Create `crates/fplopt-bot/tests/summary.rs`:

```rust
//! `plan_summary` on the synthetic golden league (`tests/fixtures/rust_golden/bot/`): team 7
//! at GW6 with the script's planner settings (horizon 1).

use std::path::{Path, PathBuf};

use fplopt_bot::context::ContextStore;
use fplopt_bot::planner::{live_params, plan_user};
use fplopt_bot::replay::FileFpl;
use fplopt_bot::summary::{horizon_points, plan_summary};
use fplopt_optimize::io::params_with;
use serde_json::Value;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

#[test]
fn plan_summary_of_the_golden_team() {
    let root = repo();
    std::env::set_current_dir(&root).unwrap(); // rules are read relative to the repo root
    let golden = root.join("tests/fixtures/rust_golden/bot");
    let script: Value =
        serde_json::from_str(&std::fs::read_to_string(golden.join("script.json")).unwrap())
            .unwrap();
    let context = ContextStore::new(golden.join("state/context")).latest().unwrap().unwrap();
    let fpl = FileFpl { dir: golden.join("fpl"), down: Default::default() };
    let params = params_with(live_params(), &script["params"]).unwrap();
    let plan = plan_user(&fpl, 7, &context, None, &params, &[]).unwrap();
    let s = plan_summary(&plan).unwrap();

    assert_eq!((s.gw, s.horizon), (6, 1));
    let best = fplopt_bot::planner::ranked(&plan.plans)[0];
    let first = best.first();
    assert_eq!(s.moves.len(), first.transfers.len());
    assert_eq!(s.hit_points, plan.rules.hit_cost * first.hits);
    let names = context.names();
    for (m, t) in s.moves.iter().zip(&first.transfers) {
        assert_eq!(m.out_name, names[&t.out_key]);
        assert_eq!(m.in_name, names[&t.in_key]);
        assert_eq!(Some(m.in_points), horizon_points(&context.xp, t.in_key, &[6]));
        assert!(m.in_fixtures.is_none(), "the golden context has no fixtures frame");
        let pool_price = context.pool.iter().find(|p| p.player_key == t.in_key).unwrap().price;
        assert_eq!(m.buy_price, pool_price);
    }
    assert_eq!(s.captain.name, names[&first.captain]);
    assert_eq!(Some(s.captain.points), horizon_points(&context.xp, first.captain, &[6]));
    assert_eq!(s.gain_vs_roll, best.gain_vs_roll);
    assert_eq!(s.bank_after, first.bank);
    // 5 free transfers, 5 moves: max(5 - 5, 0) + 1.
    assert_eq!(s.free_transfers_next, (plan.state.free_transfers - first.transfers.len() as i64).max(0) + 1);
    assert_eq!(s.team.iter().map(|(_, names)| names.len()).sum::<usize>(), 11);
    assert_eq!(s.bench.len(), 4);
    let starters: Vec<&String> = s.team.iter().flat_map(|(_, names)| names).collect();
    assert_eq!(starters.iter().filter(|n| n.ends_with(" (C)")).count(), 1);
    assert_eq!(starters.iter().filter(|n| n.ends_with(" (V)")).count(), 1);
    assert_eq!(s.alternatives.len(), plan.plans.plans.len() - 1);
    assert!(s.later.is_empty(), "a 1-GW horizon has no later moves");
    assert!(!s.told_transfers);
    assert_eq!(s.prices_fallback, Some(context.bootstrap_at.unwrap_or(context.created_at)));
}
```

`prices_fallback` is `Some(..)` here because the test plans on the stored context without live prices (`prices_at` is None).

- [ ] **Step 2: Run it to see it fail**

```bash
cargo test -p fplopt-bot --test summary
```

Expected: compile error, `cannot find function plan_summary`.

- [ ] **Step 3: Implement**

At the top of `summary.rs`, replace the imports with:

```rust
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use fplopt_optimize::state::selling_price;
use fplopt_optimize::{GwPlan, Plan, XpFrame};

use crate::context::Fixture;
use crate::planner::{UserPlan, ranked};
```

Add after `tied`:

```rust
/// Positions in lineup order.
pub const POSITIONS: [(i64, &str); 4] = [(1, "GK"), (2, "DEF"), (3, "MID"), (4, "FWD")];

/// One of this GW's transfers in plan 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Move {
    pub out_name: String,
    pub in_name: String,
    /// What the sale earns and the buy costs (tenths of £m).
    pub sell_price: i64,
    pub buy_price: i64,
    /// Expected points over the plan's horizon GWs (undecayed).
    pub out_points: f64,
    pub in_points: f64,
    /// The incoming player's games in his next 3 GWs, or None without a fixtures frame.
    pub in_fixtures: Option<Vec<(i64, Vec<(String, bool)>)>>,
}

/// A player and his expected points this GW.
#[derive(Clone, Debug, PartialEq)]
pub struct Pick {
    pub name: String,
    pub points: f64,
}

/// Another plan: its first-GW moves (out, in names), hit points, gain over rolling and
/// whether the solver can't tell it from plan 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Alternative {
    pub moves: Vec<(String, String)>,
    pub hit_points: i64,
    pub gain: Option<f64>,
    pub tied: bool,
}

/// The facts of a user's plan (spec, *Summary step*).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanSummary {
    pub gw: i64,
    pub deadline: DateTime<Utc>,
    /// GWs in plan 1's horizon.
    pub horizon: usize,
    pub moves: Vec<Move>,
    pub hit_points: i64,
    pub captain: Pick,
    pub vice: Pick,
    pub gain_vs_roll: Option<f64>,
    pub bank_after: i64,
    pub free_transfers_next: i64,
    /// Starters by position ("GK", names with " (C)"/" (V)"), positions without starters left
    /// out.
    pub team: Vec<(&'static str, Vec<String>)>,
    pub bench: Vec<String>,
    pub alternatives: Vec<Alternative>,
    /// Rolling the transfer: free transfers next GW, and whether it ties plan 1. None when the
    /// planner gave no roll plan or plan 1 makes no transfers.
    pub roll: Option<(i64, bool)>,
    /// Plan 1's later moves: (gw, out, in).
    pub later: Vec<(i64, String, String)>,
    pub stopped_early: bool,
    /// When FPL's live prices couldn't be fetched: the time of the prices used.
    pub prices_fallback: Option<DateTime<Utc>>,
    /// The plan starts from transfers the user recorded with /transfer.
    pub told_transfers: bool,
}

fn name(names: &HashMap<i64, String>, key: i64) -> String {
    names.get(&key).cloned().unwrap_or_else(|| format!("player {key}"))
}

fn moves_of(g: &GwPlan, names: &HashMap<i64, String>) -> Vec<(String, String)> {
    g.transfers.iter().map(|t| (name(names, t.out_key), name(names, t.in_key))).collect()
}

/// Free transfers at the next GW after making `n` transfers with `ft` available (FPL's
/// roll-over, capped; top-ups ignored).
fn next_free_transfers(ft: i64, n: i64, cap: i64) -> i64 {
    ((ft - n).max(0) + 1).min(cap)
}

/// The summary of `plan`, or None when the planner found no plan.
pub fn plan_summary(plan: &UserPlan) -> Option<PlanSummary> {
    let context = &plan.context;
    if plan.plans.plans.is_empty() {
        return None;
    }
    let names = context.names();
    let order = ranked(&plan.plans);
    let best = order[0];
    let first = best.first();
    let gws: Vec<i64> = best.gws.iter().map(|g| g.gw).collect();
    let points = |key: i64, gws: &[i64]| horizon_points(&context.xp, key, gws).unwrap_or(0.0);
    let pool: HashMap<i64, (i64, i64, i64)> = context
        .pool
        .iter()
        .map(|p| (p.player_key, (p.price, p.team_key, p.element_type)))
        .collect();
    let next3: Vec<i64> = context
        .gameweeks
        .iter()
        .map(|g| g.0)
        .filter(|&g| g >= first.gw)
        .take(3)
        .collect();
    let moves = first
        .transfers
        .iter()
        .map(|t| {
            let sell_price = plan
                .state
                .holdings
                .iter()
                .find(|h| h.player_key == t.out_key)
                .map_or(0, |h| selling_price(h, plan.rules.sell_on_fee));
            let (buy_price, team_key, _) = pool.get(&t.in_key).copied().unwrap_or((0, 0, 0));
            Move {
                out_name: name(&names, t.out_key),
                in_name: name(&names, t.in_key),
                sell_price,
                buy_price,
                out_points: points(t.out_key, &gws),
                in_points: points(t.in_key, &gws),
                in_fixtures: context.fixtures.as_ref().map(|f| gw_fixtures(f, team_key, &next3)),
            }
        })
        .collect();
    let mark = |k: i64| {
        if k == first.vice {
            " (V)"
        } else if k == first.captain {
            " (C)"
        } else {
            ""
        }
    };
    let team = POSITIONS
        .iter()
        .filter_map(|&(pos, label)| {
            let players: Vec<String> = first
                .starters
                .iter()
                .filter(|k| pool.get(k).map_or(0, |p| p.2) == pos)
                .map(|&k| format!("{}{}", name(&names, k), mark(k)))
                .collect();
            (!players.is_empty()).then_some((label, players))
        })
        .collect();
    let cap = plan.rules.max_free_transfers;
    let ft = plan.state.free_transfers;
    let roll = match &plan.plans.roll {
        Some(r) if !first.transfers.is_empty() => {
            Some((next_free_transfers(ft, 0, cap), tied(best, r)))
        }
        _ => None,
    };
    Some(PlanSummary {
        gw: context.gw,
        deadline: context.deadline,
        horizon: best.gws.len(),
        moves,
        hit_points: plan.rules.hit_cost * first.hits,
        captain: Pick { name: name(&names, first.captain), points: points(first.captain, &[first.gw]) },
        vice: Pick { name: name(&names, first.vice), points: points(first.vice, &[first.gw]) },
        gain_vs_roll: best.gain_vs_roll,
        bank_after: first.bank,
        free_transfers_next: next_free_transfers(ft, first.transfers.len() as i64, cap),
        team,
        bench: first.bench.iter().map(|&k| name(&names, k)).collect(),
        alternatives: order[1..]
            .iter()
            .map(|p| Alternative {
                moves: moves_of(p.first(), &names),
                hit_points: plan.rules.hit_cost * p.first().hits,
                gain: p.gain_vs_roll,
                tied: tied(best, p),
            })
            .collect(),
        roll,
        later: best.gws[1..]
            .iter()
            .flat_map(|g| {
                moves_of(g, &names).into_iter().map(move |(o, i)| (g.gw, o, i))
            })
            .collect(),
        stopped_early: !best.optimal(),
        prices_fallback: match context.prices_at {
            Some(_) => None,
            None => Some(context.bootstrap_at.unwrap_or(context.created_at)),
        },
        told_transfers: plan.spec.notes.iter().any(|n| n.starts_with("transfers you told the bot")),
    })
}
```

The told-transfers note is written by `apply_made_transfers` (`crates/fplopt-bot/src/planner.rs:177`) and starts with "transfers you told the bot about (/transfer)".

- [ ] **Step 4: Run the tests**

```bash
cargo test -p fplopt-bot --test summary
```
```bash
cargo test -p fplopt-bot --lib summary
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-bot/src/summary.rs crates/fplopt-bot/tests/summary.rs
```
```bash
git commit -m "bot: PlanSummary, the facts of a user's plan" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The `/plan` and 24-hour alert text

**Files:**
- Modify: `crates/fplopt-bot/src/messages.rs`
- Modify: `crates/fplopt-bot/src/service.rs`
- Modify: `crates/fplopt-bot/tests/chip_alerts.rs`

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `messages.rs`:

```rust
    use crate::summary::{Alternative, Move, Pick, PlanSummary};
    use chrono::TimeZone;

    fn summary() -> PlanSummary {
        PlanSummary {
            gw: 12,
            deadline: chrono::Utc.with_ymd_and_hms(2026, 11, 21, 11, 0, 0).unwrap(),
            horizon: 6,
            moves: vec![Move {
                out_name: "Wissa".into(),
                in_name: "Gonzalo".into(),
                sell_price: 69,
                buy_price: 71,
                out_points: 26.1,
                in_points: 31.4,
                in_fixtures: Some(vec![
                    (12, vec![("BRE".into(), true)]),
                    (13, vec![("LEE".into(), false), ("WOL".into(), true)]),
                    (14, vec![]),
                ]),
            }],
            hit_points: 0,
            captain: Pick { name: "Haaland".into(), points: 7.8 },
            vice: Pick { name: "Salah".into(), points: 6.9 },
            gain_vs_roll: Some(2.44),
            bank_after: 3,
            free_transfers_next: 1,
            team: vec![
                ("GK", vec!["Raya".into()]),
                ("DEF", vec!["Gabriel".into(), "Timber".into()]),
                ("FWD", vec!["Haaland (C)".into(), "Gonzalo".into()]),
            ],
            bench: vec!["Dúbravka".into(), "Ings".into()],
            alternatives: vec![
                Alternative {
                    moves: vec![("Wissa".into(), "Strand Larsen".into())],
                    hit_points: 0,
                    gain: Some(2.3),
                    tied: true,
                },
                Alternative {
                    moves: vec![("Mbeumo".into(), "Palmer".into()), ("Ings".into(), "Wood".into())],
                    hit_points: 4,
                    gain: Some(1.8),
                    tied: false,
                },
            ],
            roll: Some((2, false)),
            later: vec![(14, "Saka".into(), "Kudus".into())],
            stopped_early: false,
            prices_fallback: None,
            told_transfers: false,
        }
    }

    #[test]
    fn plan_text_in_full() {
        assert_eq!(
            render_plan(&summary()),
            "<b>GW12 plan</b> · deadline Sat 21 Nov 11:00 UK\n\
             \n\
             <b>This week's moves</b>\n\
             Sell Wissa (£6.9m) → buy Gonzalo (£7.1m)\n\
             \x20 26.1 → 31.4 pts over the next 6 GWs\n\
             \x20 Gonzalo's next games: BRE (H), LEE (A) + WOL (H), no game\n\
             Captain Haaland (7.8 pts expected) · vice Salah (6.9)\n\
             Gain: about +2.4 pts vs saving your free transfer\n\
             Afterwards: £0.3m in the bank, 1 free transfer next GW\n\
             \n\
             <b>Your team</b>\n\
             GK Raya\n\
             DEF Gabriel, Timber\n\
             FWD Haaland (C), Gonzalo\n\
             Bench Dúbravka, Ings\n\
             \n\
             <b>Close alternatives</b>\n\
             · Sell Wissa → buy Strand Larsen: +2.3 pts (just as good)\n\
             · Sell Mbeumo → buy Palmer, sell Ings → buy Wood (−4 pts hit): +1.8 pts\n\
             · Save your transfer: 2 free transfers next GW\n\
             \n\
             Looking ahead (may change): GW14 sell Saka → buy Kudus\n\
             \n\
             Made a transfer already? FPL hides it until the deadline, so tell me with /transfer."
        );
    }

    #[test]
    fn plan_text_variants() {
        let mut s = summary();
        s.hit_points = 4;
        s.horizon = 1;
        s.stopped_early = true;
        s.told_transfers = true;
        s.prices_fallback = Some(chrono::Utc.with_ymd_and_hms(2026, 11, 21, 9, 0, 0).unwrap());
        s.roll = Some((2, true));
        s.later.clear();
        let text = render_plan(&s);
        assert!(text.contains("<b>This week's moves</b> (costs a 4-pt hit)\n"), "{text}");
        assert!(text.contains("\n  26.1 → 31.4 pts this GW\n"), "{text}");
        assert!(text.contains("· Save your transfer: just as good (2 free transfers next GW)"));
        assert!(!text.contains("Looking ahead"));
        assert!(text.ends_with(
            "This plan was found under a time limit; there may be a slightly better one.\n\
             Couldn't reach FPL: prices from Sat 21 Nov 09:00 UK.\n\
             Includes the transfers you told me about (/transfer list).\n\
             Made a transfer already? FPL hides it until the deadline, so tell me with /transfer."
        ));

        let mut s = summary();
        s.moves.clear();
        s.roll = None;
        s.gain_vs_roll = None;
        s.free_transfers_next = 2;
        let text = render_plan(&s);
        assert!(
            text.contains(
                "<b>This week's moves:</b> none, save your transfer (2 free transfers next GW)\n\
                 Captain Haaland"
            ),
            "{text}"
        );
        assert!(!text.contains("Gain:"));
        assert!(!text.contains("Afterwards"));
    }

    #[test]
    fn plan_text_escapes_names_and_fits_one_message() {
        let mut s = summary();
        s.moves[0].in_name = "A&B <x>".into();
        let text = render_plan(&s);
        assert!(text.contains("buy A&amp;B &lt;x&gt;"), "{text}");
        // Worst case: 3 hits' worth of moves with long names and two alternatives.
        let long = "Abcdefghijklmnopqrstuvwxyz-Abcdefghij";
        s.moves = vec![s.moves[0].clone(); 5];
        for m in &mut s.moves {
            m.out_name = long.into();
            m.in_name = long.into();
        }
        s.hit_points = 12;
        assert!(render_plan(&s).chars().count() < MAX_MESSAGE);
    }
```

- [ ] **Step 2: Run them to see them fail**

```bash
cargo test -p fplopt-bot --lib messages
```

Expected: compile error, `cannot find function render_plan`.

- [ ] **Step 3: Implement the wording**

In `messages.rs`, add `use crate::summary::{PlanSummary, plan_summary};` and the functions below. Then:
- Delete `recommendation_text`, `lineup` and `POSITIONS`.
- Keep `prices_line` and `moves`: the old `chips_text` still uses them until Task 8, which deletes `prices_line`.
- Remove any imports that become unused (clippy runs with `-D warnings`).

```rust
fn money(tenths: i64) -> String {
    format!("£{:.1}m", tenths as f64 / 10.0)
}

fn free_transfers(n: i64) -> String {
    if n == 1 { "1 free transfer".into() } else { format!("{n} free transfers") }
}

/// "this GW" or "over the next N GWs".
pub fn over(horizon: usize) -> String {
    if horizon == 1 { "this GW".into() } else { format!("over the next {horizon} GWs") }
}

/// "sell A → buy B, sell C → buy D" (escaped); "no transfers" when empty.
fn sell_buy(moves: &[(String, String)]) -> String {
    if moves.is_empty() {
        return "no transfers".into();
    }
    let text: Vec<String> =
        moves.iter().map(|(o, i)| format!("sell {} {ARROW} buy {}", e(o), e(i))).collect();
    text.join(", ")
}

/// `text` with its first letter in upper case.
fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or(String::new(), |c| c.to_uppercase().collect::<String>() + chars.as_str())
}

fn games(fixtures: &[(i64, Vec<(String, bool)>)]) -> String {
    let per_gw: Vec<String> = fixtures
        .iter()
        .map(|(_, games)| {
            if games.is_empty() {
                return "no game".to_string();
            }
            let g: Vec<String> = games
                .iter()
                .map(|(o, home)| format!("{} ({})", e(o), if *home { "H" } else { "A" }))
                .collect();
            g.join(" + ")
        })
        .collect();
    per_gw.join(", ")
}

/// The plan message (`/plan`, the 24-hour alert).
pub fn render_plan(s: &PlanSummary) -> String {
    let mut lines =
        vec![format!("<b>GW{} plan</b> · deadline {} UK", s.gw, uk_time(s.deadline)), String::new()];
    if s.moves.is_empty() {
        lines.push(format!(
            "<b>This week's moves:</b> none, save your transfer ({} next GW)",
            free_transfers(s.free_transfers_next)
        ));
    } else {
        let hit = if s.hit_points > 0 {
            format!(" (costs a {}-pt hit)", s.hit_points)
        } else {
            String::new()
        };
        lines.push(format!("<b>This week's moves</b>{hit}"));
        for m in &s.moves {
            lines.push(format!(
                "Sell {} ({}) {ARROW} buy {} ({})",
                e(&m.out_name),
                money(m.sell_price),
                e(&m.in_name),
                money(m.buy_price)
            ));
            lines.push(format!(
                "  {:.1} {ARROW} {:.1} pts {}",
                m.out_points,
                m.in_points,
                over(s.horizon)
            ));
            if let Some(f) = &m.in_fixtures {
                lines.push(format!("  {}'s next games: {}", e(&m.in_name), games(f)));
            }
        }
    }
    lines.push(format!(
        "Captain {} ({:.1} pts expected) · vice {} ({:.1})",
        e(&s.captain.name),
        s.captain.points,
        e(&s.vice.name),
        s.vice.points
    ));
    if !s.moves.is_empty() {
        if let Some(g) = s.gain_vs_roll {
            lines.push(format!("Gain: about {g:+.1} pts vs saving your free transfer"));
        }
        lines.push(format!(
            "Afterwards: {} in the bank, {} next GW",
            money(s.bank_after),
            free_transfers(s.free_transfers_next)
        ));
    }
    lines.push(String::new());
    lines.push("<b>Your team</b>".into());
    for (label, names) in &s.team {
        lines.push(format!("{label} {}", e(&names.join(", "))));
    }
    lines.push(format!("Bench {}", e(&s.bench.join(", "))));
    if !s.alternatives.is_empty() || s.roll.is_some() {
        lines.push(String::new());
        lines.push("<b>Close alternatives</b>".into());
        for a in &s.alternatives {
            let hit = if a.hit_points > 0 {
                format!(" (−{} pts hit)", a.hit_points)
            } else {
                String::new()
            };
            let gain = a.gain.map_or(String::new(), |g| format!(": {g:+.1} pts"));
            let tie = if a.tied { " (just as good)" } else { "" };
            lines.push(format!("· {}{hit}{gain}{tie}", capitalized(&sell_buy(&a.moves))));
        }
        if let Some((ft, tie)) = s.roll {
            lines.push(if tie {
                format!("· Save your transfer: just as good ({} next GW)", free_transfers(ft))
            } else {
                format!("· Save your transfer: {} next GW", free_transfers(ft))
            });
        }
    }
    if !s.later.is_empty() {
        let mut by_gw: Vec<(i64, Vec<(String, String)>)> = Vec::new();
        for (gw, o, i) in &s.later {
            match by_gw.last_mut() {
                Some((g, moves)) if g == gw => moves.push((o.clone(), i.clone())),
                _ => by_gw.push((*gw, vec![(o.clone(), i.clone())])),
            }
        }
        let parts: Vec<String> = by_gw
            .iter()
            .map(|(gw, moves)| format!("GW{gw} {}", sell_buy(moves)))
            .collect();
        lines.push(String::new());
        lines.push(format!("Looking ahead (may change): {}", parts.join("; ")));
    }
    lines.push(String::new());
    if s.stopped_early {
        lines.push("This plan was found under a time limit; there may be a slightly better one.".into());
    }
    if let Some(at) = s.prices_fallback {
        lines.push(format!("Couldn't reach FPL: prices from {} UK.", uk_time(at)));
    }
    if s.told_transfers {
        lines.push("Includes the transfers you told me about (/transfer list).".into());
    }
    lines.push(
        "Made a transfer already? FPL hides it until the deadline, so tell me with /transfer."
            .into(),
    );
    lines.join("\n")
}

/// `render_plan` of `plan`'s summary, or the no-plan text.
pub fn plan_text(plan: &UserPlan) -> String {
    match plan_summary(plan) {
        Some(s) => render_plan(&s),
        None => format!(
            "<b>GW{} plan</b> · deadline {} UK\n\nI couldn't find a plan this time. The admin has been told.",
            plan.context.gw,
            uk_time(plan.context.deadline)
        ),
    }
}
```

Add `UserPlan` to the `use crate::planner::{...}` import. The "Gain:" and "Afterwards:" lines show only when there are moves (spec: no-move week). `uk_time` (`util.rs`) formats `%a %d %b %H:%M` in UK time, so the tests expect `Sat 21 Nov 11:00` (GMT in November).

- [ ] **Step 4: Use it in service.rs**

In `service.rs`, replace `recommendation_text(&result, None, &[])` (in `plan` and `make_plan`) with `plan_text(&result)`. In `alert_user`, keep `changes_text` deciding whether an `alert2` is sent until Task 6, and send the full plan for now:

```rust
        let text = match (kind, &last) {
            ("alert2", Some(last)) => {
                let changes = changes_text(last.decision.as_ref(), &decision, &context.names());
                if changes.is_empty() && section.is_none() {
                    return finish(base, AlertOutcome::Silent);
                }
                plan_text(&plan)
            }
            _ => plan_text(&plan),
        };
```

Update the `use crate::messages::{...}` import. `changes_text` stays in `messages.rs` until Task 6.

- [ ] **Step 5: Update `chip_alerts.rs` markers**

In `crates/fplopt-bot/tests/chip_alerts.rs`, the assertions `rest.starts_with("<b>GW6 plan</b>")` stay true. In the test around line 302-305, change `rest.starts_with("<b>GW6 update</b>")` to `rest.starts_with("<b>GW6 plan</b>")` and delete `assert!(!rest.contains("Transfers changed"), ...)`. Task 6 sets the final check.

- [ ] **Step 6: Run the tests, re-record the golden, read the diff**

```bash
cargo test -p fplopt-bot --lib messages
```
```bash
FPLOPT_UPDATE_GOLDEN=1 cargo test -p fplopt-bot --test golden
```
```bash
cargo test -p fplopt-bot
```
```bash
git diff tests/fixtures/rust_golden/bot/results.json | head -80
```

Expected: the tests PASS. In the diff, the `/plan` and `alert24` texts are now in the new layout, and no other reply changed. Check the step-21 `/plan` text by eye against the spec.

- [ ] **Step 7: Commit**

```bash
git add crates/fplopt-bot tests/fixtures/rust_golden/bot
```
```bash
git commit -m "bot: plain-words plan message with reasoning" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The "plan changed" update

**Files:**
- Modify: `crates/fplopt-bot/src/summary.rs`
- Modify: `crates/fplopt-bot/src/messages.rs`
- Modify: `crates/fplopt-bot/src/service.rs:824-833`
- Modify: `crates/fplopt-bot/tests/chip_alerts.rs`

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `summary.rs`:

```rust
    #[test]
    fn plan_change_compares_moves_captain_and_vice() {
        let names: HashMap<i64, String> =
            [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")].into_iter().map(|(k, v)| (k, v.to_string())).collect();
        let before = serde_json::json!({"transfers": [[1, 2]], "captain": 3, "vice": 4});
        let same = crate::planner::Decision { transfers: vec![[1, 2]], captain: 3, vice: 4 };
        assert!(change_of(Some(&before), &same, &names, &|_| None).is_none());
        assert!(change_of(None, &same, &names, &|_| None).is_none());
        let after = crate::planner::Decision { transfers: vec![[1, 5]], captain: 3, vice: 2 };
        let points = |k: i64| Some(k as f64 * 10.0);
        let c = change_of(Some(&before), &after, &names, &points).unwrap();
        assert_eq!(c.was_moves, vec![("A".to_string(), "B".to_string())]);
        assert_eq!(c.now_moves, vec![("A".to_string(), "E".to_string())]);
        assert!(c.moves_changed);
        assert_eq!(c.captain, ("C".to_string(), None));
        assert_eq!(c.vice, ("B".to_string(), Some("D".to_string())));
        assert_eq!(c.now_points, vec![("E".to_string(), 50.0)]);
        assert_eq!(c.was_points, vec![("B".to_string(), 20.0)]);
    }
```

Add to `mod tests` in `messages.rs`:

```rust
    #[test]
    fn update_text_says_what_changed() {
        let change = crate::summary::PlanChange {
            gw: 12,
            deadline: chrono::Utc.with_ymd_and_hms(2026, 11, 21, 11, 0, 0).unwrap(),
            horizon: 6,
            was_moves: vec![("Wissa".into(), "Strand Larsen".into())],
            now_moves: vec![("Wissa".into(), "Gonzalo".into())],
            moves_changed: true,
            captain: ("Haaland".into(), None),
            vice: ("Salah".into(), Some("Saka".into())),
            now_points: vec![("Gonzalo".into(), 31.4)],
            was_points: vec![("Strand Larsen".into(), 30.6)],
        };
        assert_eq!(
            update_text(&change),
            "<b>GW12 plan changed</b> · deadline Sat 21 Nov 11:00 UK\n\
             Now: sell Wissa, buy Gonzalo (was: sell Wissa, buy Strand Larsen)\n\
             Captain: Haaland (unchanged)\n\
             Vice: Salah (was Saka)\n\
             Why (pts over the next 6 GWs): now Gonzalo 31.4; before Strand Larsen 30.6\n\
             Full plan and team: /plan"
        );
        let mut c = change;
        c.moves_changed = false;
        c.now_points.clear();
        c.was_points.clear();
        let text = update_text(&c);
        assert!(!text.contains("Now:") && !text.contains("Why"), "{text}");
    }
```

- [ ] **Step 2: Run them to see them fail**

```bash
cargo test -p fplopt-bot --lib
```

Expected: compile errors, `cannot find function change_of` / `update_text`.

- [ ] **Step 3: Implement `PlanChange`**

Add to `summary.rs`:

```rust
/// What changed in plan 1's decision since the last alert (spec, *Update alert*).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanChange {
    pub gw: i64,
    pub deadline: DateTime<Utc>,
    pub horizon: usize,
    pub was_moves: Vec<(String, String)>,
    pub now_moves: Vec<(String, String)>,
    pub moves_changed: bool,
    /// (now, Some(was) if it changed)
    pub captain: (String, Option<String>),
    pub vice: (String, Option<String>),
    /// Incoming players new to the plan / dropped from it, with their horizon points (players
    /// without xP rows left out).
    pub now_points: Vec<(String, f64)>,
    pub was_points: Vec<(String, f64)>,
}

fn pairs(transfers: &serde_json::Value) -> Vec<(i64, i64)> {
    transfers
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|p| (p[0].as_i64().unwrap_or_default(), p[1].as_i64().unwrap_or_default()))
                .collect()
        })
        .unwrap_or_default()
}

/// The change from `before` (the stored decision JSON) to `after`, or None if nothing
/// changed (or there is no earlier decision). `points(key)`: a player's horizon points.
pub fn change_of(
    before: Option<&serde_json::Value>,
    after: &crate::planner::Decision,
    names: &HashMap<i64, String>,
    points: &dyn Fn(i64) -> Option<f64>,
) -> Option<PlanChange> {
    let before = before?;
    let was = pairs(&before["transfers"]);
    let now: Vec<(i64, i64)> = after.transfers.iter().map(|p| (p[0], p[1])).collect();
    let (mut a, mut b) = (was.clone(), now.clone());
    a.sort();
    b.sort();
    let captain = before["captain"].as_i64().unwrap_or_default();
    let vice = before["vice"].as_i64().unwrap_or_default();
    let moves_changed = a != b;
    if !moves_changed && captain == after.captain && vice == after.vice {
        return None;
    }
    let named = |m: &[(i64, i64)]| -> Vec<(String, String)> {
        m.iter().map(|&(o, i)| (name(names, o), name(names, i))).collect()
    };
    let ins = |m: &[(i64, i64)], other: &[(i64, i64)]| -> Vec<(String, f64)> {
        m.iter()
            .filter(|(_, i)| !other.iter().any(|(_, j)| j == i))
            .filter_map(|&(_, i)| points(i).map(|p| (name(names, i), p)))
            .collect()
    };
    let changed = |was: i64, now: i64| (name(names, now), (was != now).then(|| name(names, was)));
    Some(PlanChange {
        gw: 0,
        deadline: DateTime::<Utc>::MIN_UTC,
        horizon: 0,
        was_moves: named(&was),
        now_moves: named(&now),
        moves_changed,
        captain: changed(captain, after.captain),
        vice: changed(vice, after.vice),
        now_points: if moves_changed { ins(&now, &was) } else { Vec::new() },
        was_points: if moves_changed { ins(&was, &now) } else { Vec::new() },
    })
}

/// `change_of` for `plan` (its GW, deadline, horizon and xP filled in).
pub fn plan_change(before: Option<&serde_json::Value>, plan: &UserPlan) -> Option<PlanChange> {
    let best = *ranked(&plan.plans).first()?;
    let gws: Vec<i64> = best.gws.iter().map(|g| g.gw).collect();
    let decision = crate::planner::decision_summary(plan);
    let points = |k: i64| horizon_points(&plan.context.xp, k, &gws);
    let mut change = change_of(before, &decision, &plan.context.names(), &points)?;
    change.gw = plan.context.gw;
    change.deadline = plan.context.deadline;
    change.horizon = best.gws.len();
    Some(change)
}
```

Add `serde_json` as a path in `summary.rs` (the crate already depends on it).

- [ ] **Step 4: Implement `update_text`**

Add to `messages.rs` (and import `PlanChange`):

```rust
/// The re-run's message when only the chip call changed (after the chip section).
pub const UNCHANGED: &str = "Your transfers and captain are unchanged: /plan";

fn plain_moves(moves: &[(String, String)]) -> String {
    if moves.is_empty() {
        return "no transfers".into();
    }
    let parts: Vec<String> =
        moves.iter().map(|(o, i)| format!("sell {}, buy {}", e(o), e(i))).collect();
    parts.join("; ")
}

/// The "plan changed" update (spec, *Update alert*).
pub fn update_text(c: &PlanChange) -> String {
    let mut lines = vec![format!("<b>GW{} plan changed</b> · deadline {} UK", c.gw, uk_time(c.deadline))];
    if c.moves_changed {
        lines.push(format!(
            "Now: {} (was: {})",
            plain_moves(&c.now_moves),
            plain_moves(&c.was_moves)
        ));
    }
    for (label, (now, was)) in [("Captain", &c.captain), ("Vice", &c.vice)] {
        lines.push(match was {
            Some(was) => format!("{label}: {} (was {})", e(now), e(was)),
            None => format!("{label}: {} (unchanged)", e(now)),
        });
    }
    if !c.now_points.is_empty() || !c.was_points.is_empty() {
        let list = |p: &[(String, f64)]| {
            let parts: Vec<String> = p.iter().map(|(n, x)| format!("{} {x:.1}", e(n))).collect();
            parts.join(", ")
        };
        let what = if c.horizon == 1 { "pts this GW".to_string() } else { format!("pts over the next {} GWs", c.horizon) };
        let mut why = format!("Why ({what}): now {}", list(&c.now_points));
        if !c.was_points.is_empty() {
            why += &format!("; before {}", list(&c.was_points));
        }
        lines.push(why);
    }
    lines.push("Full plan and team: /plan".into());
    lines.join("\n")
}
```

Delete `changes_text` and its `pairs` helper from `messages.rs`, and its test `changes` (replaced by the `summary.rs` test).

- [ ] **Step 5: Wire the `alert2` path in service.rs**

Replace the `let text = match (kind, &last) { ... };` block (lines ~824-833) with:

```rust
        let text = match (kind, &last) {
            ("alert2", Some(last)) => match plan_change(last.decision.as_ref(), &plan) {
                None if section.is_none() => return finish(base, AlertOutcome::Silent),
                None => UNCHANGED.to_string(),
                Some(change) => update_text(&change),
            },
            _ => plan_text(&plan),
        };
```

Update the imports (`crate::summary::plan_change`, `crate::messages::{UNCHANGED, update_text, ...}`), and remove `changes_text`.

- [ ] **Step 6: Update `chip_alerts.rs`**

In the test that checks an `alert2` sent because the chip call changed (Task 5 left it asserting `rest.starts_with("<b>GW6 plan</b>")`), replace that assertion with:

```rust
    assert_eq!(rest, fplopt_bot::messages::UNCHANGED, "{rest}");
```

Also update the module doc of `chip_alerts.rs` to say the texts are compared with the recorded golden, not Python's.

If a step in the test replays an `alert2` whose transfers did change, assert `rest.starts_with("<b>GW6 plan changed</b>")` there instead.

- [ ] **Step 7: Run, re-record, read the diff**

```bash
cargo test -p fplopt-bot --lib
```
```bash
FPLOPT_UPDATE_GOLDEN=1 cargo test -p fplopt-bot --test golden
```
```bash
cargo test -p fplopt-bot
```
```bash
git diff tests/fixtures/rust_golden/bot/results.json | grep -n "plan changed" | head
```

Expected: PASS. The golden's two `alert2` sends are now "GW6 plan changed" texts with "Now:" lines.

- [ ] **Step 8: Commit**

```bash
git add crates/fplopt-bot tests/fixtures/rust_golden/bot
```
```bash
git commit -m "bot: short 'plan changed' update instead of the full plan" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Chip section in the alert

**Files:**
- Modify: `crates/fplopt-bot/src/messages.rs` (`chip_section`, `chip_failure_section`, their tests)
- Modify: `crates/fplopt-bot/tests/chip_alerts.rs`

- [ ] **Step 1: Rewrite the tests first**

Replace the five `chip_section_*` tests in `messages.rs` with:

```rust
    #[test]
    fn chip_section_play_and_agree() {
        assert_eq!(
            chip_section(&chip_calls(Some("bboost"), Some("bboost"), None)),
            "<b>Chip:</b> play Bench Boost this week (+7.4 pts vs keeping it)\n\
             \x20 The simple rule agrees: your bench is expected to score 11.6 (it plays at 10.9+)\n\
             Chips left: Free Hit (until GW19, best week GW16), Bench Boost (until GW19, playing now)"
        );
    }

    #[test]
    fn chip_section_hold_and_differ() {
        let text = chip_section(&chip_calls(None, Some("bboost"), None));
        assert!(text.starts_with(
            "<b>Chip:</b> keep your chips this week\n  A simpler rule would play Bench Boost (your bench is expected to score 11.6; it plays at 10.9+)\n"
        ), "{text}");
        assert!(text.ends_with("Chips left: Free Hit (until GW19, best week GW16), Bench Boost (until GW19)"), "{text}");
    }

    #[test]
    fn chip_section_forced_and_expiring() {
        let mut calls = chip_calls(None, Some("freehit"), Some("freehit"));
        calls.gw = 18;
        calls.open[0].expiring = true;
        let text = chip_section(&calls);
        assert!(text.starts_with(
            "<b>Chip:</b> play Free Hit now (2 chips left and 2 GWs before they expire after GW19) · your Free Hit team: /chips\n  The simple rule agrees: last chance before they expire\n"
        ), "{text}");
        assert!(text.contains("Free Hit (expires after GW19, playing now)"));
    }

    #[test]
    fn chip_section_forced_chip_does_not_override_the_planner() {
        let mut calls = chip_calls(Some("bboost"), Some("freehit"), Some("freehit"));
        calls.gw = 18;
        for o in &mut calls.open {
            o.expiring = true;
        }
        calls.open[0].best_gw = Some(19);
        assert_eq!(
            chip_section(&calls),
            "<b>Chip:</b> play Bench Boost this week (+7.4 pts vs keeping it)\n\
             \x20 2 chips left and 2 GWs before they expire after GW19: play one each GW\n\
             \x20 A simpler rule would play Free Hit (last chance before they expire)\n\
             Chips left: Free Hit (expires after GW19, best week GW19), Bench Boost (expires after GW19, playing now)"
        );
    }

    #[test]
    fn chip_section_free_hit_points_to_chips() {
        let text = chip_section(&chip_calls(Some("freehit"), None, None));
        assert!(text.starts_with("<b>Chip:</b> play Free Hit this week (+7.4 pts vs keeping it) · your Free Hit team: /chips\n"), "{text}");
        assert!(text.contains("\n  A simpler rule would keep them (captain expected 7.0, bench expected 11.6)\n"), "{text}");
    }
```

The "differ" case uses "; it plays at" inside the parentheses, so the brackets don't nest (spec wording adapted for that).

- [ ] **Step 2: Run them to see them fail**

```bash
cargo test -p fplopt-bot --lib chip_section
```

Expected: FAIL (old texts).

- [ ] **Step 3: Rewrite `chip_section` and `chip_failure_section`**

Replace both function bodies in `messages.rs` with:

```rust
pub fn chip_section(calls: &ChipCalls) -> String {
    let forced = calls.forced.as_deref();
    let headline = calls.planner.as_deref().or(forced);
    let window = forced.and_then(|f| {
        let last = calls.open.iter().find(|o| o.name == f)?.last_gw;
        let left = calls.open.iter().filter(|o| o.last_gw == last).count();
        Some((left, last - calls.gw + 1, last))
    });
    let mut lines = Vec::new();
    lines.push(match (calls.planner.as_deref(), forced, window) {
        (Some(c), _, _) => format!(
            "<b>Chip:</b> play {} this week ({:+.1} pts vs keeping it)",
            e(chip_label(c)),
            calls.planner_gain
        ),
        (None, Some(c), Some((left, gws, last))) => format!(
            "<b>Chip:</b> play {} now ({left} chips left and {gws} GWs before they expire after GW{last})",
            e(chip_label(c))
        ),
        (None, Some(c), None) => format!("<b>Chip:</b> play {} now, or lose it", e(chip_label(c))),
        (None, None, _) => "<b>Chip:</b> keep your chips this week".to_string(),
    });
    if let Some(c @ ("freehit" | "wildcard")) = headline {
        lines[0].push_str(&format!(" \u{b7} your {} team: /chips", chip_label(c)));
    }
    if calls.planner.is_some()
        && let Some((left, gws, last)) = window
    {
        lines.push(format!(
            "  {left} chips left and {gws} GWs before they expire after GW{last}: play one each GW"
        ));
    }
    let gain = |key: &str| match calls.gains.get(key).copied().flatten() {
        Some(v) => format!("{v:.1}"),
        None => "n/a".to_string(),
    };
    let rules = calls.rules.as_deref();
    // (what the rule saw, its threshold) — joined with " (… )" when agreeing, "; " inside
    // the brackets when not.
    let (seen, threshold): (String, Option<&str>) = match rules {
        Some(c) if calls.forced.as_deref() == Some(c) => {
            ("last chance before they expire".to_string(), None)
        }
        Some("3xc") => (format!("your captain is expected to score {}", gain("captain_xp")), Some("8.3+")),
        Some("bboost") => (format!("your bench is expected to score {}", gain("bench_xp")), Some("10.9+")),
        Some("freehit") => (
            format!("{} of your players have no game", gain("blanks").trim_end_matches(".0")),
            Some("4+"),
        ),
        Some("wildcard") => (format!("a Wildcard gains {} pts", gain("gain_wc")), Some("12.9+")),
        Some(_) => (String::new(), None),
        None => (
            format!("captain expected {}, bench expected {}", gain("captain_xp"), gain("bench_xp")),
            None,
        ),
    };
    if rules == headline {
        lines.push(match (rules, threshold) {
            (None, _) => "  The simple rule agrees: keep them".to_string(),
            (Some(_), Some(t)) => format!("  The simple rule agrees: {} (it plays at {t})", e(&seen)),
            (Some(_), None) => format!("  The simple rule agrees: {}", e(&seen)),
        });
    } else {
        let play = match rules {
            Some(c) => format!("play {}", e(chip_label(c))),
            None => "keep them".to_string(),
        };
        let why = match threshold {
            Some(t) => format!("{}; it plays at {t}", e(&seen)),
            None => e(&seen),
        };
        lines.push(format!("  A simpler rule would {play} ({why})"));
    }
    if calls.open.is_empty() {
        lines.push("Chips left: none".to_string());
    } else {
        let parts: Vec<String> = calls
            .open
            .iter()
            .map(|o| {
                let mut note = if o.expiring {
                    format!("expires after GW{}", o.last_gw)
                } else {
                    format!("until GW{}", o.last_gw)
                };
                if headline == Some(o.name.as_str()) {
                    note.push_str(", playing now");
                } else if let Some(best) = o.best_gw
                    && !(calls.planner.is_none() && best == calls.gw)
                {
                    note.push_str(&format!(", best week GW{best}"));
                }
                format!("{} ({note})", e(chip_label(&o.name)))
            })
            .collect();
        lines.push(format!("Chips left: {}", parts.join(", ")));
    }
    lines.join("\n")
}

/// The chip section when the calls could not be computed.
pub fn chip_failure_section() -> String {
    "<b>Chip:</b> couldn't check chips this time, try /chips".to_string()
}
```

Keep the doc comment above `chip_section`. Drop its last sentence about the evidence line if there is one.

- [ ] **Step 4: Update `chip_alerts.rs`**

Set `const HEADER: &str = "<b>Chip:</b>";`. Replace `const EVIDENCE: &str = "Evidence: on 2023/24 and 2024/25";` with `const CHIPS_LEFT: &str = "Chips left:";`, and rename its uses (`section.contains(EVIDENCE)` becomes `section.contains(CHIPS_LEFT)`).

- [ ] **Step 5: Run, re-record, check**

```bash
cargo test -p fplopt-bot --lib chip_section
```
```bash
FPLOPT_UPDATE_GOLDEN=1 cargo test -p fplopt-bot --test golden
```
```bash
cargo test -p fplopt-bot
```

Expected: PASS. The golden replays run without chip alerts, so its `results.json` shouldn't change in this task. Confirm with `git diff --stat tests/fixtures/rust_golden/bot`.

- [ ] **Step 6: Commit**

```bash
git add crates/fplopt-bot tests/fixtures/rust_golden/bot
```
```bash
git commit -m "bot: plain-words chip section in the alert" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `/chips`

**Files:**
- Modify: `crates/fplopt-bot/src/messages.rs` (`chips_text`)
- Modify: `crates/fplopt-bot/src/service.rs` (`chips_text(&options, 6)` → `chips_text(&options)`)

`/chips` is the only user of its facts, so it reads `ChipOptions` directly instead of going through a summary struct. The spec is amended to say so in Task 11.

- [ ] **Step 1: Write the failing check against the golden**

`ChipOptions` is too big to build by hand, so this text is checked on the golden replay. Add to `crates/fplopt-bot/tests/golden.rs`, at the end of `bot_session_matches_golden` (before `remove_dir_all`):

```rust
    // /chips (step 15 and 35 of the script) in the new wording.
    let chips: Vec<&str> = results
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r.get("replies")?.as_array()?.first()?.get(1)?.as_str())
        .filter(|t| t.starts_with("<b>GW6 chips</b>"))
        .collect();
    assert!(!chips.is_empty());
    for text in chips {
        assert!(text.contains(" · for information: your /plan doesn't use them\n"), "{text}");
        assert!(text.contains("<b>Each chip in its best week</b>"), "{text}");
        assert!(text.contains("\nChips left: "), "{text}");
        assert!(text.contains("A chip only shows a gain if it beats saving it for later."));
        assert!(!text.contains("solver") && !text.contains("xP"), "{text}");
        assert!(text.chars().count() <= fplopt_bot::messages::MAX_MESSAGE);
    }
```

- [ ] **Step 2: Run it to see it fail**

```bash
cargo test -p fplopt-bot --test golden
```

Expected: FAIL on the new assertions.

- [ ] **Step 3: Rewrite `chips_text`**

Replace `chips_text` in `messages.rs` with:

```rust
/// `/chips` (information only, spec *`/chips`*).
pub fn chips_text(options: &ChipOptions) -> String {
    let (context, rules, params) = (&options.context, &options.rules, &options.params);
    let gw_map = context.gw_index_map();
    let gw_of: HashMap<i64, i64> = gw_map.iter().map(|(g, i)| (*i, *g)).collect();
    let used: BTreeSet<i64> = options.state.chips_used.iter().map(|(cid, _)| *cid).collect();
    let mut windows: Vec<_> = rules.chips.iter().collect();
    windows.sort_by_key(|w| (w.stop, w.start, w.chip_id));
    let objectives = &options.plans.scenario_objectives;
    let base = objectives.iter().rev().find(|(l, _)| l == "none").map_or(0.0, |(_, v)| *v);
    let best = options.plans.best();
    let names = context.names();
    let mut lines = vec![
        format!("<b>GW{} chips</b> · for information: your /plan doesn't use them", context.gw),
        if best.scenario == "none" {
            format!("Best {}: keep your chips", over(best.gws.len()))
        } else {
            format!(
                "Best {}: {} ({:+.1} pts vs keeping your chips)",
                over(best.gws.len()),
                e(&scenario_label(&best.scenario)),
                best.total_objective() - base
            )
        },
    ];
    let first = best.first();
    match first.chip.as_deref() {
        Some(chip @ ("freehit" | "wildcard")) if best.scenario != "none" => {
            let positions: HashMap<i64, i64> =
                context.pool.iter().map(|p| (p.player_key, p.element_type)).collect();
            lines.push(String::new());
            lines.push(format!("<b>Your {} team</b>", chip_label(chip)));
            for (pos, label) in crate::summary::POSITIONS {
                let players: Vec<&str> = first
                    .starters
                    .iter()
                    .filter(|k| positions.get(k).copied() == Some(pos))
                    .map(|&k| name(&names, k))
                    .collect();
                if !players.is_empty() {
                    lines.push(format!("{label} {}", e(&players.join(", "))));
                }
            }
            let bench: Vec<&str> = first.bench.iter().map(|&k| name(&names, k)).collect();
            lines.push(format!("Bench {}", e(&bench.join(", "))));
            lines.push(format!("Captain {}", e(name(&names, first.captain))));
        }
        Some(_) if best.scenario != "none" => {
            lines.push(format!(
                "This week with it: {}; captain {}",
                moves(first, &names, rules.hit_cost),
                e(name(&names, first.captain))
            ));
        }
        _ => {}
    }
    if !options.singles.is_empty() {
        let mut singles: Vec<(&str, f64)> = CHIP_LABELS
            .iter()
            .filter_map(|(chip, _)| options.single(chip))
            .map(|(label, value)| (label.as_str(), value - base))
            .collect();
        singles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        lines.push(String::new());
        lines.push("<b>Each chip in its best week</b>".into());
        for (label, gain) in singles {
            lines.push(format!("· {}: {gain:+.1}", e(&scenario_label(label))));
        }
    }
    // Chips left, grouped by the GW their window ends.
    let mut groups: Vec<(i64, Vec<&str>)> = Vec::new();
    for w in windows.iter().filter(|w| !used.contains(&w.chip_id) && w.stop >= context.gw_index) {
        let last = gw_of.get(&w.stop).copied().unwrap_or(w.stop);
        match groups.last_mut() {
            Some((g, chips)) if *g == last => chips.push(chip_label(&w.name)),
            _ => groups.push((last, vec![chip_label(&w.name)])),
        }
    }
    let left: Vec<String> =
        groups.iter().map(|(last, chips)| format!("{} (until GW{last})", chips.join(", "))).collect();
    lines.push(String::new());
    lines.push(format!("Chips left: {}", e(if left.is_empty() { "none" } else { &left.join("; ") })));
    lines.push("A chip only shows a gain if it beats saving it for later.".into());
    let solved: BTreeSet<&str> = objectives
        .iter()
        .map(|(l, _)| l.as_str())
        .chain(options.singles.iter().map(|(_, (l, _))| l.as_str()))
        .collect();
    let stopped = options
        .plans
        .search
        .not_optimal
        .iter()
        .filter(|(l, _)| !solved.contains(l.as_str()))
        .count();
    if stopped > 0 {
        lines.push("Some options ran out of time, so this may be incomplete.".into());
    }
    lines.push(if params.chip_value_left.is_some() {
        "Tested on 2023–25: beat playing no chips, not clearly better than simple rules.".into()
    } else {
        "The chip planner isn't tested yet: weigh it against your own judgement.".into()
    });
    lines.join("\n")
}
```

Then:
- Reword `moves` (used by "This week with it"): `"no transfers"` stays, and the hit becomes `format!(" (−{} pts hit)", hit_cost * g.hits)`.
- Delete `prices_line` (no users left), and remove `chip_value`, `py_g` and any other imports that are now unused.
- `stopped` now counts unsolved options that hit the time limit. The old code intersected `scenario_bounds` minus solved with `not_optimal`. `search.not_optimal` is `Vec<(String, String)>` (label, status).
- In `service.rs`, change `chips_text(&options, 6)` to `chips_text(&options)`.

- [ ] **Step 4: Run, re-record, read**

```bash
cargo test -p fplopt-bot --lib
```
```bash
FPLOPT_UPDATE_GOLDEN=1 cargo test -p fplopt-bot --test golden
```
```bash
cargo test -p fplopt-bot
```
```bash
uv run python -c "import json;r=json.load(open('tests/fixtures/rust_golden/bot/results.json',encoding='utf-8'));print(r[15]['replies'][0][1])"
```

Expected: PASS. The printed `/chips` text shows "Best this GW: Free Hit GW6 (+13.5 pts vs keeping your chips)", the Free Hit team by position, the chips in their best weeks sorted by gain, and "Chips left: Bench Boost, Triple Captain, Wildcard, Free Hit (until GW19); … (until GW38)". The step index may differ; find the `/chips` reply.

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-bot tests/fixtures/rust_golden/bot
```
```bash
git commit -m "bot: plain-words /chips with the chip team" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Command texts

**Files:**
- Modify: `crates/fplopt-bot/src/service.rs`
- Modify: `crates/fplopt-bot/src/messages.rs` (`squad_text`)

- [ ] **Step 1: Write the failing test for `squad_text`**

Add to `mod tests` in `messages.rs`:

```rust
    #[test]
    fn squad_text_lists_the_squad_by_position() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rust_golden/bot/state/context");
        let id = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().file_name();
        let context = crate::context::read_context(&dir.join(id)).unwrap();
        let keys: HashMap<i64, i64> = context.pool.iter().map(|p| (p.player_key, p.element_type)).collect();
        // Two GKs and three outfield players of the context, as a squad spec.
        let pick = |t: i64, n: usize| -> Vec<i64> {
            context
                .players
                .iter()
                .filter(|p| keys.get(&p.player_key) == Some(&t))
                .take(n)
                .map(|p| p.element_id)
                .collect()
        };
        let mut elements = pick(1, 2);
        elements.extend(pick(4, 3));
        let players = elements
            .iter()
            .map(|&element_id| fplopt_optimize::entry::SquadPlayer {
                element_id,
                purchase_price: 50,
                name: String::new(),
            })
            .collect();
        let mut spec = SquadSpec::new(context.season, context.gw, players, 5, 2);
        spec.chips_used = vec![("wildcard".into(), 4)];
        let text = squad_text(&context, &spec);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("GK "), "{text}");
        assert!(lines[1].starts_with("FWD "), "{text}");
        assert_eq!(lines[2], "£0.5m in the bank · 2 free transfers");
        assert_eq!(lines[3], "Chips played: Wildcard (GW4)");
    }
```

Add `use std::path::Path;` to the test module.

- [ ] **Step 2: Run it to see it fail**

```bash
cargo test -p fplopt-bot --lib squad_text
```

Expected: compile error (`squad_text` takes 3 arguments).

- [ ] **Step 3: Rewrite `squad_text`**

```rust
/// The imported squad by position, bank and free transfers, chips played (/register).
pub fn squad_text(context: &PlanContext, spec: &SquadSpec) -> String {
    let names: HashMap<i64, &str> =
        context.players.iter().map(|p| (p.element_id, p.web_name.as_str())).collect();
    let keys = context.element_keys();
    let positions: HashMap<i64, i64> =
        context.pool.iter().map(|p| (p.player_key, p.element_type)).collect();
    let position = |element: i64| keys.get(&element).and_then(|k| positions.get(k)).copied();
    let label = |element: i64| {
        names.get(&element).map_or_else(|| element.to_string(), |n| n.to_string())
    };
    let mut lines = Vec::new();
    for (pos, tag) in crate::summary::POSITIONS {
        let players: Vec<String> = spec
            .players
            .iter()
            .filter(|p| position(p.element_id) == Some(pos))
            .map(|p| label(p.element_id))
            .collect();
        if !players.is_empty() {
            lines.push(format!("{tag} {}", e(&players.join(", "))));
        }
    }
    let other: Vec<String> = spec
        .players
        .iter()
        .filter(|p| position(p.element_id).is_none())
        .map(|p| label(p.element_id))
        .collect();
    if !other.is_empty() {
        lines.push(format!("Others {}", e(&other.join(", "))));
    }
    let used: Vec<String> = spec
        .chips_used
        .iter()
        .map(|(n, gw)| format!("{} (GW{gw})", chip_label(n)))
        .collect();
    let used = used.join(", ");
    lines.push(format!(
        "{} in the bank · {}",
        money(spec.bank),
        free_transfers(spec.free_transfers)
    ));
    lines.push(format!("Chips played: {}", e(if used.is_empty() { "none" } else { &used })));
    lines.join("\n")
}
```

`season_label` is no longer used here; remove its import if it's unused.

- [ ] **Step 4: Replace the command texts in service.rs**

Make these exact replacements:

```rust
pub const HELP: &str = "<b>FPL optimizer</b>: transfer and captain picks for your FPL team. You make the final call.

/register &lt;team ID&gt; · link your FPL team (the number in your team's web address)
/plan · this week's moves, captain and team
/chips · when to play your chips (for information)
/transfer &lt;out&gt; -&gt; &lt;in&gt; · a transfer you've already made this week (FPL hides it until the deadline); /transfer list, undo, clear
/setft &lt;n&gt; · fix the number of free transfers (/setft clear to go back to FPL's count)
/alerts on|off · a message a day before each deadline, plus a heads-up if the plan changes
/help · this list";
pub const PENDING: &str = "Thanks! The admin has been asked to let you in.";
pub const NOT_READY: &str = "I'm still updating for the next deadline. Try again in a few minutes.";
const REGISTER_FIRST: &str = "First link your FPL team: /register &lt;team ID&gt;";
```

| Where | Old | New |
| --- | --- | --- |
| `active` | `"Send /start first."` | `"Send /start to begin."` |
| `unknown` | `"Unknown command. /help lists them."` | `"I don't know that command. /help lists them."` |
| `register` usage | `"Usage: /register &lt;team ID&gt;, the number in your FPL team's URL (fantasy.premierleague.com/entry/<b>1234567</b>/event/…)."` | `"Send /register &lt;team ID&gt;. It's the number in your team's web address: fantasy.premierleague.com/entry/<b>1234567</b>/event/…"` |
| `register` not found | `format!("FPL has no team with ID {entry_id}.")` | `format!("FPL has no team with ID {entry_id}. Check the number in your team's web address.")` |
| `register` no context | `format!("Registered team {entry_id}.\n{NOT_READY}")` | `format!("Linked team {entry_id}.\n{NOT_READY}")` |
| `register` can't plan | `"Registered team {entry_id}, but its squad can't be planned yet: {}"` | `"Linked team {entry_id}, but I can't plan it yet: {}"` |
| `register` success | `format!("Registered.\n\n{text}\n\n/plan for the GW{} plan.", context.gw)` with `squad_text(&context, &spec, entry_id)` | `format!("Linked team {entry_id}.\n\n{text}\n\n/plan for this week's moves.")` with `squad_text(&context, &spec)` |
| `plan`, `chips` entry error | `format!("Can't plan team {entry_id}: {}", e(&message))` | `format!("I couldn't plan your team: {}", e(&message))` |
| `transfer` list | `format!("GW{gw} transfers you've told me about: {}", …)` | `format!("Transfers you've told me about for GW{gw}: {}", …)` |
| `transfer` usage | `"Usage: /transfer Wissa -&gt; Gonzalo (or /transfer Wissa Gonzalo); /transfer list, /transfer undo, /transfer clear"` | `"Send /transfer Wissa -&gt; Gonzalo (or /transfer Wissa Gonzalo). Also /transfer list, /transfer undo, /transfer clear."` |
| `transfer` recorded | `"Recorded for GW{gw}: {}. All GW{gw} transfers: {}. Bank £{:.1}m, {} FT left. /plan and the alerts now plan from this squad."` | see below |
| `setft` clear | `format!("GW{}: free transfers from the import again.", context.gw)` | `format!("GW{}: back to FPL's free transfer count.", context.gw)` |
| `setft` usage | `format!("Usage: /setft &lt;0-{cap}&gt; or /setft clear")` | `format!("Send /setft 0–{cap}, or /setft clear.")` |
| `setft` set | `"GW{} (deadline {}): planning with {n} free transfers."` | `format!("GW{}: planning with {n} free transfers. (/setft clear to go back to FPL's count)", context.gw)` |
| `alerts` usage | `format!("Alerts are {state}. Usage: /alerts on\|off")` | `format!("Alerts are {state}. Send /alerts on or /alerts off.")` |
| `alerts` set | `format!("Alerts {choice}.")` | `if choice == "on" { "Alerts on: a message a day before each deadline.".to_string() } else { "Alerts off.".to_string() }` |
| `alert_user` entry error | `format!("Couldn't plan GW{} for team {entry}: {}", context.gw, e(&message))` | `format!("I couldn't plan your team for GW{}: {}", context.gw, e(&message))` (drop the now-unused `entry` binding) |

The new `transfer` recorded text:

```rust
        let n_made = all.len();
        Ok(reply(
            chat_id,
            format!(
                "Got it: {} for GW{gw}{}. {} in the bank, {} left. Your plans now start from this team.",
                listed(&[new_move]),
                if n_made > 1 { format!(" ({n_made} transfers this GW: {})", listed(&all)) } else { String::new() },
                money_text(spec.bank),
                free_left(spec.free_transfers)
            ),
        ))
```

Add the two helpers to `service.rs`, or make `money`/`free_transfers` in `messages.rs` `pub` and import them as `money as money_text` and `free_transfers as free_left`. Making them `pub` is the DRY choice.

The `resolve` error "No player called 'Nobody' this season." (`service.rs:113`) becomes:

```rust
        0 => Err(format!("No player called {} this season. Check the spelling as FPL writes it.", py_repr_str(name))),
```

Admin texts (`ADMIN_HELP`, `/allow`, `/block`, `/users`, "New user asks for access", "That's you.", "You have access.") stay unchanged.

- [ ] **Step 5: Run, re-record, read every changed reply**

```bash
cargo test -p fplopt-bot
```
```bash
FPLOPT_UPDATE_GOLDEN=1 cargo test -p fplopt-bot --test golden
```
```bash
cargo test -p fplopt-bot
```
```bash
uv run python -c "import json;r=json.load(open('tests/fixtures/rust_golden/bot/results.json',encoding='utf-8'));[print('---',i,'\n',t) for i,x in enumerate(r) if x for c,t in x.get('replies',[])]" | less
```

Expected: PASS. Every user reply reads as in the spec's *Other commands* section, and admin replies are unchanged.

- [ ] **Step 6: Commit**

```bash
git add crates/fplopt-bot tests/fixtures/rust_golden/bot
```
```bash
git commit -m "bot: plain-words command replies" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Clean-up, full checks, docs

**Files:**
- Modify: `crates/fplopt-bot/src/messages.rs` (module doc)
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md`

- [ ] **Step 1: Module doc of messages.rs**

Replace the module doc with:

```rust
//! Telegram message texts (spec docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md):
//! Telegram HTML (`<b>` and escaped text only), short plain-FPL lines for a phone, worded
//! from `summary`'s facts. `plan_text` is the plan (`/plan`, the 24-hour alert),
//! `update_text` the re-run's "plan changed", `chip_section` the alerts' chip lines,
//! `chips_text` `/chips`; `split_message` cuts a text to Telegram's 4,096 character limit at
//! line breaks. The texts are the Rust bot's own since 2026-11-11 (no longer Python's).
```

- [ ] **Step 2: Find leftover jargon**

```bash
grep -n "xP\|solver\| FT\b\|snapshot\|Usage:" crates/fplopt-bot/src/messages.rs crates/fplopt-bot/src/service.rs | grep -v "^\s*//" | grep "\""
```

Expected: only admin `Usage: /allow …` and log/warn strings (not sent to users). Fix any user-facing hit.

- [ ] **Step 3: CLAUDE.md**

In the `fplopt-rs bot run` bullet, replace "every command's and alert's text (Python's, byte for byte)" with "every command's and alert's text (the Rust bot's own since 2026-11-11, spec `docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md`; plain FPL words, per-transfer points, prices and fixtures)". Replace "so the Python golden still matches" with "the golden's `results.json`/`db.json` are recorded from Rust: `FPLOPT_UPDATE_GOLDEN=1 cargo test -p fplopt-bot --test golden` after any text change or after `tests/test_rust_golden.py` rewrites the bot fixtures". In the `bot_parity.py` gate text, add "(texts no longer compared)".

- [ ] **Step 4: Amend the spec to match the plan**

In `docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md`:
- **`ChipsSummary` paragraph:** replace it with "`/chips` is worded straight from `ChipOptions` (it is the only user of those facts)."
- **Deadline format:** in the examples, `Sat 21 Nov, 11:00 UK` becomes `Sat 21 Nov 11:00 UK` (the bot's `uk_time` format).
- **"Chips left" line:** replace the example with "Chips left: Wildcard, Free Hit, Bench Boost, Triple Captain (until GW19); Wildcard, Free Hit, Bench Boost, Triple Captain (until GW38)" (grouped by window end).
- **Plan errors:** replace "I couldn't plan your team: <reason>. Try again later; the admin has been told." with "I couldn't plan your team: <reason>". These are problems with the squad that the admin isn't told about.
- **Python reader:** replace "which is checked in the implementation" with "checked: `_FRAMES` in `src/fplopt/bot/context.py` lists the five frames it reads".
- **Disagreeing simple rule:** the line reads "(your bench is expected to score 11.6; it plays at 10.9+)".

- [ ] **Step 5: Full checks**

```bash
cargo test --workspace
```
```bash
cargo clippy --workspace --all-targets -- -D warnings
```
```bash
cargo fmt --all -- --check
```
```bash
uv run pytest tests/test_rust_golden.py -q
```
```bash
uv run ruff check . && uv run ruff format --check .
```

Expected: all pass (pytest: 1 skipped).

- [ ] **Step 6: Commit**

```bash
git add -A crates/fplopt-bot CLAUDE.md docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md
```
```bash
git commit -m "docs: bot texts are Rust's own; spec matches the build" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Real-names check before deploy (needs the user)

**Files:** none in the repo.

- [ ] **Step 1: Ask the user before touching the server.** Ask: "Copy the server's `state/` (with `bot.sqlite` as a backup copy, never the live file) and replay `/plan`, `/chips` and both alerts locally on it?" Go on only after a yes.

- [ ] **Step 2: Replay on the copy**

This follows the `bot_parity.py` setup in `deploy/README.md`:
1. Write a short script JSON with `/plan`, `/chips` and both alerts for the user's registered chat.
2. Run it:
   ```bash
   cargo run --release -p fplopt-cli -- bot replay --script S --state DIR --fpl DIR --out out.json
   ```
3. Read every message. Check:
   - real names;
   - fixtures appear (the server's context needs a `live prepare` from this build to get a `fixtures.parquet`; build one locally with `fplopt-rs live prepare` against the copied `data/`);
   - every message is under 4,096 characters;
   - no jargon.

- [ ] **Step 3: Report the sample texts to the user.** Don't deploy without a go-ahead.
