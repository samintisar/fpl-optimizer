# Chip suggestions in the bot's alerts: design

**Date:** 2026-11-05. **Status:** approved by the user in chat. Written in Rust on `feat/chip-alerts` (from `rust-migration`). It goes live with the Rust server switch.

## Why

- The bot's plans and alerts play no chips. Chips are "information only" behind `/chips` (PLAN §8, §12 2026-10-11).
- Top managers play every chip: eight a season since 2025/26. A season without chips understates where the system would finish:
  - the 2025/26 holdout scored 2,317 from the template start with no chips, about rank 110k;
  - the top-10k cutoff was about 2,400.
- The chip study's validate run (2023/24–2024/25) found:
  - the chip planner P beat playing no chips: **+4.18 pts/GW, p 0.025**, about +130 a season;
  - it failed the gate only because it also had to beat the simple rules H (+1.70/GW, p 0.083, needed < 0.0296).
- The user's rule is "tested before it drives advice". P was tested against no chips and passed, so it may drive a suggestion.
- The user decided on 2026-11-05:
  - the bot suggests chips in its alerts, the planner's call next to the rules' call, with the evidence stated;
  - the user still plays chips in FPL;
  - `/plan`'s transfer plan stays the no-chip plan.

## What the user sees

The 24-hour alert (`alert24`) gets a chip section above the transfer plan, for example:

```
Chip this week: play Bench Boost (+7.4 xP vs keeping it)
  Simple rules agree: bench xP 11.6 ≥ 10.9
Unused chips: Free Hit (expires after GW19, best GW: GW16), Bench Boost (GW19, playing now)
Evidence: on 2023/24 and 2024/25 this planner scored +4.2 pts/GW over playing no chips;
not proven better than the simple rules.
```

### 1. The planner's call (P)

- **The search:** `chip_options` on the user's squad, exactly as `/chips` runs it.
  - Same `chip_params()`: the live settings, estimated terminal values at `chip_shrink` 1.0, the pooled table.
  - Same FPL live prices and made transfers.
- **Play:** if the best scenario plays a chip in the first GW, the line is "Chip this week: play <chip> (±x xP vs keeping it)". The xP is the best scenario's objective minus the no-chip one.
- **Hold:** otherwise the line is "Chip this week: hold". Each unused, open chip then gets its most promising GW from `ChipOptions::single`.
- **Free Hit or Wildcard:** the line adds "see /chips for the chip squad", because the alert's transfer plan stays the no-chip plan.

### 2. The rules' call (H)

- **The thresholds,** fixed as chosen on develop (`chiprules:…,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87`):
  - Triple Captain if the captain's xP ≥ 8.3;
  - Bench Boost if the bench xP ≥ 10.9;
  - Free Hit if ≥ 4 of the 15 squad players' clubs have no fixture;
  - Wildcard if the Wildcard plan beats the no-chip plan by ≥ 12.87 xP.
- **Last chance:** H's rule plays a chip when the chips left in a window outnumber its GWs.
- **The numbers** are the backtest's `chip_gains` on the bot's no-chip plan, in the same order as `ChipRules::choose`.
- **Shown as:**
  - "Simple rules agree: <reason>" when H and P make the same call;
  - "Simple rules: <their call> (<reason>)" when they differ.

### 3. Unused chips and expiry

- Each unused chip whose window is open lists its last GW and its best GW, or "playing now".
- **Warning:** from 4 GWs before a window closes (`stop − gw ≤ 3`), the chip gets "expires after GW<stop>".
- **Forced:** when H's last-chance rule fires, the planner line becomes "play <chip> now, or lose it", whatever P says.
  - P's terminal value for a chip whose window ends in the horizon is already 0, so P rarely disagrees.

### 4. The evidence line

It is fixed text, as above. It stays until a chip policy passes the full gate.

### 5. The 2-hour update (`alert2`)

- **Chip call changed** (P's or H's) since this GW's `alert24` call: the update shows the chip section.
- **Nothing changed:** the update leaves the chip section out.
- **Silent alert:** if the chip call and the transfer plan are both unchanged, the update stays silent, as now.

### 6. Failure

- If the chip search fails or times out for a user, the section is one line: "Chip check failed this time: try /chips".
- The transfer alert still goes out. A chip failure never fails an alert.

## Logging (also the chip-policy-2 shadow log)

**New table** `chip_calls`, written in the same transaction as the alert's `recommendations` row. Columns:

- `user_id`, `season`, `gw`, `kind`, `context_id`, `created_at`;
- `planner_call`, `rules_call`, `v6_call` (chip name or `none`);
- `planner_gain`;
- `gains`: H's numbers, as JSON;
- `scenarios`: P's solved scenarios, as `chip_checks` stores them.

**`v6_call`:**
- It is the frozen chip-policy-2 V6: `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1,chip_shrink=0.8,chip_split=1,fh_scale=0.4`, with the per-window table, live variant (docs/research/2026-11-chip-policy-2.md §5).
- It is computed for `alert24` only, from GW20 on (gw_index ≥ 20), as that spec requires. Before GW20 it is null.
- It is logged only, never shown.
- This row is the shadow logger that spec's *Code* item 6 asks for, so no separate logger is built.

**What the log is for:** scoring at season end. That means P vs H vs V6 vs what the user actually played, from `entry/{id}/history/` chips.

## Code

1. **Share the rule logic.**
   - Move `chip_gains`, `blanks`, `ChipRules::choose` (as a free function over the four thresholds) and `CHIP_DIAGNOSTICS` into `fplopt_optimize::chip_rules`, so the bot can call them without depending on `fplopt-backtest`.
   - Backtest behaviour is unchanged: the policies call the moved functions.
   - Check: `cargo test --workspace`, plus one `backtest run` with `chiprules` and one with `chipgains` on a short season range before and after the move, with identical `gws.parquet`.
2. **`fplopt_bot::chip_calls`:** given a user's no-chip plan and context, it returns:
   - P's call, H's call and gains, the expiry list and the forced flag;
   - from GW20 on, V6's call.

   V6 needs the per-window table and `fh_scale`. They are wired through the same params the backtest's `ChipPolicy2` uses, read once at bot start like `chip_params()`.
3. **`messages::chip_section(calls) -> String`:** the HTML above, escaped like every message. Labels from `chip_label`.
4. **`BotService::alert_user`:**
   - computes the calls after `make_plan`;
   - prepends the section on `alert24`;
   - on `alert2`, compares with the logged `alert24` call;
   - records `chip_calls`.

   A flag `chip_alerts: bool` on `BotService`:
   - `true` in `bot run` and `live tick`;
   - `false` by default in `bot replay`, where a script may set `"chip_alerts": true`.

   So the Python-generated golden (`tests/fixtures/rust_golden/bot/`) and `dev/bot_parity.py` still compare byte for byte. **Intentional divergence from Python:** Rust's live alerts carry the section and Python's don't. That's fine because the bot gates have run (df34009, 000670b) and Python is deleted after the switch.
5. **Store:** the `chip_calls` table is created with `CREATE TABLE IF NOT EXISTS` (the schema convention), plus `record_chip_call` and `chip_calls(user_id)`.
6. **Docs:**
   - PLAN §8: alerts carry chip suggestions.
   - PLAN §12: a decision row for 2026-11-05.
   - CLAUDE.md: the `live tick` / bot lines.
   - deploy/README: nothing new to deploy, since `config/chips/*` is already in the repo.

## Tests

- **Unit, the call logic.** Built on a synthetic plan context:
  - P plays, P holds, H agrees or differs;
  - last chance forced, expiry warning at `stop − gw` 3 vs 4;
  - chip search error leads to the failure line;
  - V6 null before GW20 and set from GW20.
- **Unit, the text.** `chip_section` for each case above, with exact strings.
- **Store.** A `chip_calls` round trip; the alert row and the chip row are written together.
- **Replay.** A new Rust-only golden (`crates/fplopt-bot/tests/chip_alerts.rs`):
  - the synthetic league with `chip_alerts: true`, covering an `alert24` with a chip, then an `alert2` with a changed call and one with an unchanged call;
  - checks the texts and the `chip_calls` rows.
- **Unchanged:** the existing bot golden and `dev/bot_parity.py` with `chip_alerts` off; the backtest checks from Code item 1.
- **Server.** After the switch, check the first live `alert24`'s section against `/chips` for the same user.

## Not in scope

- The bot playing chips for the user (FPL has no API for it).
- Chips in `/plan`'s transfer plan.
- A new chip policy or new thresholds.
- `/plan` showing the chip section. `/chips` already covers on-demand use.

## Timing

- **Live:** with the server switch (about GW12–13).
- **The user's first-half Bench Boost and Free Hit** (window ends GW19) are therefore covered by alerts from about GW13.
- **Until then:** the user checks `/chips` before each deadline.
- **V6 logging** starts at GW20 (deadline about 2026-12-29), after the switch.
