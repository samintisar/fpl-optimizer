# Bot messaging UX: plain words and reasoning

Date: 2026-11-11. Scope: the Rust Telegram bot (`crates/fplopt-bot`), which has been the live bot since the 2026-11-05 switch.

## Goal

The bot's readers are casual FPL players who know the game. They know captains, benches, free transfers and chips, but not models or solvers. Today's messages use system jargon: "xP", "FT", "solver's precision", "xP data from", "Save the FT", notes about replayed imports and "Simple rules agree: bench xP 11.6 ≥ 10.9". They also print everything at once. The new messages:

- use plain FPL words: "pts", "expected points", "free transfer", "−4 pts hit";
- lead with what to do this week;
- show the reasoning a reader can check: each transfer's points out and in, prices, the incoming player's next fixtures, the captain's expected points and the gain over saving the transfer;
- keep short one-line alternatives;
- cut every line that only matters to the system's author.

Not in scope: changing any decision (plans, chips, alerts timing), the admin texts (`/allow`, `/block`, `/users`, the access request) and the Python bot.

## Decisions

- **Rust only.** The texts change in Rust. The Python bot stays as the frozen rollback with its old texts. Byte-for-byte text parity with Python is retired. `dev/bot_parity.py` keeps comparing database rows and decisions, and the bot golden is re-recorded from Rust.
- **Structure.** `messages.rs` splits into two steps:
  1. A **summary step** (`summary.rs`) builds plain data structs from `UserPlan`, `ChipOptions`, `ChipCalls` and the context.
  2. A **wording step** (`messages.rs`) turns those structs into Telegram HTML.

  The plan, the update and `/chips` share the summary step. The summary step holds no text; the wording step does no arithmetic.
- **Fixtures** come from a new `fixtures` frame in the plan context: each club's fixtures for the planned GWs, with the opponent's `team_dim.short_name` and home/away, read from `schedule` at the context's deadline view. Contexts without the frame (written before this change) still load, and their messages leave out the fixtures line.

## Summary step (`crates/fplopt-bot/src/summary.rs`)

`PlanSummary` (from a `UserPlan`):
- `gw`, `deadline`.
- `moves`: this GW's transfers in plan 1. Each move holds:
  - the out and in names, the selling price and the buying price;
  - each player's points summed over the plan's horizon GWs from the xP frame (undecayed);
  - the in player's next 3 fixtures (`(short_name, home)`), or none without the frame.
- `hit_points` this GW (`hits × hit_cost`).
- `captain` and `vice`: names and this GW's expected points.
- `gain_vs_roll`: plan 1's existing `gain_vs_roll`.
- `bank_after`; `free_transfers_next`: plan 1's FTs at the next GW.
- `team`: starters by position (C/V marked) and the bench in order.
- `alternatives`: plans 2.. in ranked order. Each holds its first-GW moves (names), its hit points, its `gain_vs_roll` and `tied` (the existing `tied` test against plan 1).
- `roll`: when present, the roll plan's free transfers next GW and whether it is tied with plan 1.
- `later`: plan 1's transfers in later GWs, as `(gw, out, in)`.
- `stopped_early`: plan 1 not optimal.
- `prices_fallback`: `Some(time)` only when FPL's live prices could not be fetched.
- `told_transfers`: true when the plan includes transfers recorded with `/transfer`.

`PlanChange` (the update alert): the earlier and new first-GW moves, the earlier and new captain and vice, and the new points for any changed incoming player. Built from the stored decision, as `changes_text` is today.

`ChipsSummary` (from `ChipOptions`):
- the best scenario and its gain over keeping every chip;
- the chip GW's team (Free Hit or Wildcard) or its moves and captain (Bench Boost or Triple Captain);
- each chip in its best week with its gain;
- chips left, grouped by window;
- whether any option ran out of time;
- whether the planner is the tested one (`chip_value_left`).

The alert's chip section reads `ChipCalls` directly. It needs no new struct, only new wording.

## Wording

General rules: short lines for a phone; "UK" once, on the deadline; no "solver", "precision", "snapshot", "import", "xP" or "FT"; every error says what to do next.

### `/plan` and the 24-hour alert

```
GW12 plan · deadline Sat 21 Nov, 11:00 UK

This week's moves
Sell Wissa (£6.9m) → buy Gonzalo (£7.1m)
  26.1 → 31.4 pts over the next 6 GWs
  Gonzalo's next games: BRE (H), LEE (A), WOL (H)
Captain Haaland (7.8 pts expected) · vice Salah (6.9)
Gain: about +2.4 pts vs saving your free transfer
Afterwards: £0.3m in the bank, 1 free transfer next GW

Your team
GK Raya
DEF Gabriel, Timber, Muñoz, Hall
MID Salah (V), Saka, Mbeumo, Semenyo
FWD Haaland (C), Gonzalo
Bench Dúbravka, Mitchell, Andersen, Ings

Close alternatives
· Sell Wissa → buy Strand Larsen: +2.3 pts (just as good)
· Sell Mbeumo → buy Palmer: +1.8 pts
· Save your transfer: 2 free transfers next GW

Looking ahead (may change): GW14 sell Saka → buy Kudus

Made a transfer already? FPL hides it until the deadline, so tell me with /transfer.
```

The variants:
- **A hit:** "This week's moves (costs a 4-pt hit)".
- **No transfers:** "This week's moves: none, save your transfer (2 free transfers next GW)". The alternatives still follow.
- **A tie with the roll plan:** "· Save your transfer: just as good (2 free transfers next GW)".
- **"Gain" line:** shown only when `gain_vs_roll` exists.
- **"Looking ahead":** shown only when `later` is non-empty, with several moves joined by "; ".
- **Planner stopped early:** "This plan was found under a time limit; there may be a slightly better one."
- **Price fallback:** "Couldn't reach FPL: prices from 09:00 UK."
- **`told_transfers`:** "Includes the transfers you told me about (/transfer list)."
- **No plan:** "I couldn't find a plan this time. The admin has been told."
- **Alerts:** the chip section goes above the plan, followed by a blank line (as today).

Removed: the team ID, the "xP data from" time, the live-prices line, "Chips are your call…", the notes list, the per-GW transfer list of plan 1 and the "within the solver's precision" sentences.

### Update alert (about 2 h before the deadline, sent only on a change)

```
GW12 plan changed · deadline Sat 21 Nov, 11:00 UK
Now: sell Wissa, buy Gonzalo (was: sell Wissa, buy Strand Larsen)
Captain: Haaland (unchanged)
Why: Gonzalo now 31.4 pts over 6 GWs vs Strand Larsen 30.6
Full plan and team: /plan
```

- "Now" and "was" show this GW's moves ("no transfers" when empty). The "Now/was" line appears only when the moves changed.
- The captain and vice lines show "(was X)" or "(unchanged)".
- The "Why" line compares the new incoming players' horizon points with the earlier ones', for players still in the frame. It is left out when the moves are unchanged.
- The chip rules for `alert2` don't change (`service.rs` `chip_text`). The chip section, when shown, goes above the update. When only the chip call changed, the update is the chip section, then "Your transfers and captain are unchanged: /plan".
- The re-run's first alert for a user this GW (no `alert24` sent) is still the full plan, not an update.

### Chip section in the 24-hour alert

```
Chip: play Bench Boost this week (+7.4 pts vs keeping it)
  The simple rule agrees: your bench is expected to score 11.6 (it plays at 10.9+)
Chips left: Free Hit (until GW19, best week GW16), Bench Boost (until GW19)
```

The headline:

| Case | Headline |
| --- | --- |
| Keep | "Chip: keep your chips this week" |
| Last chance | "Chip: play Free Hit now (2 chips left and 2 GWs before they expire after GW19)" |
| Planner plays a chip while another is forced | "  2 chips left and 2 GWs before GW19: play one each GW" |
| Free Hit or Wildcard headline | append " · your Free Hit team: /chips" (or Wildcard) |
| Chip check failed | "Chip: couldn't check chips this time, try /chips" |

The simple rule's line:
- "The simple rule agrees: …" when it matches the headline, or "A simpler rule would play X (…)" / "A simpler rule would keep them (…)" when it doesn't.
- The reasons: captain "your captain is expected to score 7.0 (it plays at 8.3+)"; bench as above; Free Hit "4 of your players have no game (it plays at 4+)"; Wildcard "a Wildcard gains 3.0 pts (it plays at 12.9+)"; last chance "last chance before they expire"; keep "captain expected 7.0, bench expected 11.6".

The chip list:
- "Chips left" notes show "expires after GW19" for an expiring chip, ", playing now" for the headline chip and ", best week GWn". The rule that hides the best week next to a keep call stays.
- The "Evidence" line moves to `/chips`.

### `/chips`

```
GW12 chips · for information, the plan above doesn't use them
Best over the next 6 GWs: Free Hit this week (+13.5 pts vs keeping your chips)

Your Free Hit team
GK …
DEF …
MID …
FWD …
Bench …
Captain Haaland

Each chip in its best week
· Free Hit GW12: +13.5
· Wildcard GW12: +11.5
· Triple Captain GW12: +5.0
· Bench Boost GW15: −1.0

Chips left: Wildcard, Free Hit, Bench Boost, Triple Captain (until GW19), and a second set until GW38
A chip only shows a gain if it beats saving it for later.
Tested on 2023–25: beat playing no chips, not clearly better than simple rules.
```

- **Best scenario with Bench Boost or Triple Captain:** "This week with it: <moves>; captain X" in place of the team.
- **Best is no chip:** "Best over the next 6 GWs: keep your chips".
- **Multi-chip scenarios:** they read as today's scenario labels ("Wildcard GW5 + Triple Captain GW9").
- **Untested planner** (no `chip_value_left`): the last line reads "The chip planner isn't tested yet: weigh it against your own judgement."
- **An option ran out of time:** "Some options ran out of time, so this may be incomplete."
- **Removed:** the "Options the solver checked" list, the "ruled out" count and the "How to read it" paragraph.
- **The checks list** is sorted by gain, descending.

### Other commands

- **`/help`:**
  ```
  <b>FPL optimizer</b>: transfer and captain picks for your FPL team. You make the final call.

  /register <team ID> · link your FPL team (the number in your team's web address)
  /plan · this week's moves, captain and team
  /chips · when to play your chips (for information)
  /transfer <out> -> <in> · a transfer you've already made this week (FPL hides it until the deadline); /transfer list, undo, clear
  /setft <n> · fix the number of free transfers (/setft clear to go back to FPL's count)
  /alerts on|off · a message a day before each deadline, plus a heads-up if the plan changes
  /help · this list
  ```
- **Not ready:** "I'm still updating for the next deadline. Try again in a few minutes."
- **Pending access:** "Thanks! The admin has been asked to let you in."
- **Register first:** "First link your FPL team: /register <team ID>".
- **`/register`:**
  - Usage: "Send /register <team ID>. It's the number in your team's web address: fantasy.premierleague.com/entry/<b>1234567</b>/event/…".
  - Unknown team: "FPL has no team with ID 8. Check the number in your team's web address."
  - Success: "Linked team 4570069." followed by the squad by position (the plan's team layout without C/V), "£0.5m in the bank · 2 free transfers", "Chips played: Wildcard (GW4)" and "/plan for this week's moves."
  - Can't plan the squad yet: "Linked team 4570069, but I can't plan it yet: <reason>".
- **`/transfer`:**
  - Recorded: "Got it: Wissa → Gonzalo for GW12. £0.3m in the bank, 0 free transfers left. Your plans now start from this team."
  - List: "Transfers you've told me about for GW12: …" (or "none").
  - Undo or clear: "Removed 1. GW12 transfers now: …".
  - Usage: "Send /transfer Wissa -> Gonzalo (or /transfer Wissa Gonzalo). Also /transfer list, /transfer undo, /transfer clear."
  - No player: "No player called 'Nobody' this season. Check the spelling as FPL writes it."
  - Refused: "Can't record that transfer: <reason>".
- **`/setft`:** "GW12: planning with 2 free transfers. (/setft clear to go back to FPL's count)". Clear: "GW12: back to FPL's free transfer count." Usage: "Send /setft 0–5, or /setft clear."
- **`/alerts`:** "Alerts on: a message a day before each deadline." / "Alerts off." Usage: "Alerts are on. Send /alerts on or /alerts off."
- **Errors:**
  - "Can't plan team X: …" becomes "I couldn't plan your team: <reason>. Try again later; the admin has been told." The admin is told whenever the current code tells them.
  - Unknown command: "I don't know that command. /help lists them."
  - The generic apology stays.

Reasons passed through from planner errors (`EntryError` messages) are shown as they are. Rewording them is out of scope.

## Context: the `fixtures` frame

- **Contents:** `builder.rs` writes `fixtures.parquet` next to the other frames, with columns `gw`, `team_key`, `opponent_short_name`, `home` (bool). It covers every fixture of the context's season in GWs ≥ the planned GW, from `schedule` (and `team_dim` for short names) at the deadline's view.
- **Doubles and blanks:** a double GW gives two rows, which read "BRE (H) + LEE (A)" as one GW. A blank gives none, which reads "no game".
- **Loading:** `read_context` loads the frame when present. `PlanContext` gets `fixtures: Option<…>`.
- **Python:** `meta.json` is unchanged. Python's context reader ignores unknown files, which is checked in the implementation; it must still read Rust contexts during a rollback.

## Testing

- **Summary unit tests** on a small synthetic context:
  - horizon points and prices per move;
  - fixtures (a single, a double and a blank) and none without the frame;
  - captain and vice points;
  - ties with alternatives and with the roll plan;
  - the hit;
  - no transfers;
  - `later`;
  - `PlanChange`'s moves, captain and why.
- **Wording tests** with exact strings for the plan (normal, hit, no transfers, stopped early, price fallback), the update, every chip-section case in the table above (replacing today's Python-copied tests), `/chips` (Free Hit team, Bench Boost moves, keep, untested, ran out of time) and `/help`.
- **Bot golden:**
  - `tests/fixtures/rust_golden/bot/results.json` is re-recorded from Rust's `bot replay`, with a regen switch on `crates/fplopt-bot/tests/golden.rs` (env `FPLOPT_UPDATE_GOLDEN=1`). The diff is reviewed once by eye.
  - `chip_alerts.rs` is re-recorded the same way.
  - `tests/bot_replay.py` stays as the Python side, but it no longer writes the golden.
- **`dev/bot_parity.py`:** compares database rows and each user's decision (transfers, captain, vice). It no longer compares reply texts.
- **Length:** a test that the worst-case `/chips` (Free Hit with 15 players) and a plan with 3 hits and 2 alternatives each fit in one 4,096-character message. `split_message` stays.
- **Escaping:** every name and reason goes through `escape`. One test uses a name with `&` and `<`.

## Rollout

- **Before deploying:** run `fplopt-rs bot replay` on a copy of the server's `state/`, with a real script, and read the messages with real names.
- **On deploy:** the next `live tick` rebuilds the context (the code changed), so fixtures appear from the first alert.
- **Rollback:** the Python units are the rollback, with their old texts.
- **CLAUDE.md:** the batch 5 line says the bot's texts are "Python's, byte for byte". Change it to say the bot's texts are Rust's own since this change, and that the golden is recorded from Rust.
