# Value of information: which xP input is worth points to the planner

**Date:** 2026-11-09. **Status:** approved by the owner in chat, 2026-11-09. Plan: `docs/superpowers/plans/2026-11-09-value-of-information.md` (its last section lists three deviations from this spec, recorded before any code). Review: `docs/research/2026-11-system-review.md` §4–5.

## Why

- With chips played the system sits at about 2,325–2,345 against a target of 2,400. The planner, chip-timing and price items in the review sum to 15–35 points. None is the right size.
- xP accuracy is. A 5% random error on xP cost the planner about 25 points a season (the noise study's placebo arms), and the November v1 upgrade was worth about +173 a season on validate.
- The research loop improves the model on the data it has, judged on component error. Nobody has measured what each component is worth **in season points**, or what a new source of information (team news, lineups, player odds) could be worth at most. Without that, the next data source is a guess.

This study answers one question per component: if the planner knew it perfectly, how many points a season would it gain? A component whose ceiling is small is closed. A component whose ceiling is large gets a sourcing spec.

**How it differs from the research loop:** it changes no model. It replaces a component's forecast with the truth, inside the backtester only, and reads the planner's points.

## Oracle components

v1 assembles xP per fixture from a `Fixture` record (`crates/fplopt-models/src/assemble.rs:202-240`): `p_start`, `p_60`, `p_play`, `e_minutes`, the team's `lambda_for` / `lambda_against` / `p_cs`, `e_goals`, `e_assists`, and the small terms. Each arm overwrites some of those fields from realized data before the points are assembled. Everything else is v1's.

| Arm | What is replaced | Horizon | The information it stands for |
|---|---|---|---|
| **M0** | `p_play`, `p_start` ← whether he played / started | next gameweek only | Perfect team news for the coming gameweek |
| **M0h** | As M0, for a random half of the players (seeded by player and gameweek) | next gameweek only | A source that is right half the time v1 is unsure |
| **MA** | As M0 | all six gameweeks | Perfect knowledge of injuries, bans and rotation ahead |
| **A0** | `p_play` ← played or not; `p_start` kept as v1's share of it | next gameweek only | Availability only (injury news), without the lineup |
| **T0** | `lambda_for`, `lambda_against`, `p_cs` ← from closing odds | next gameweek only | The market at kick-off instead of at the deadline |
| **R** | A team's expected non-penalty goals and assists, re-split among its players by their realized season npxG and xA per 90 × expected minutes (players with ≥ 450 minutes with xG; team totals stay v1's). As built; see the plan's *Changes during the build* | all six | Knowing how a team's output divides between its players (what player odds would add) |

- In M0, MA and A0, expected minutes and the per-fixture goal and assist expectations are rescaled with the new probabilities, as v1's own calibration step does (`assemble.rs:315-345`), so per-90 rates are unchanged.
- "Started" is FPL's `starts` from 2022/23 gameweek 16 and the inferred starts before (the 11 with the most minutes, 98.65% accurate; PLAN §6).
- T0 uses `odds_snapshot` rows with `is_closing`. R uses Understat rates and needs xG coverage, so it runs on 2019/20–2022/23 only.
- **No arm uses realized goals, assists, clean sheets or points.** Those are luck, and an oracle on them says nothing about information. Minutes after kick-off (a player injured at 20 minutes) are also left to v1: M0 and MA replace who plays and starts, not how long.

## Where it lives

- `fplopt-backtest` only, next to `oracle.rs`: an xP source `v1+oracle-<ARM>` built from v1's fixtures and the realized tables read at the season's end. Spec form `optimizer:v1+oracle-M0:horizon=6,...`.
- It reads the future by design. `optimize plan`, the bot, `live` and `models eval`'s gates refuse it, and `check leakage` lists it as a deliberate exception, exactly as `prices=oracle`.
- Cached in the xP disk cache under its own model name, so v1's cache is untouched.
- The holdout is refused as everywhere.

## Measurement

- **Seasons:** develop 2016/17–2022/23 (R: 2019/20–2022/23). No validate, no holdout: this is a ceiling, not a claim about a policy.
- **Starts:** `template@1,random:3@1,template@20,random:3@20`, as Gate A.
- **Arms vs L** (`optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`, no chips), whole-season paired difference per (season, start) unit, mean with an 80% t interval over season means. Family `value of information`.
- **Also reported per arm:** transfers and hits a season, XI and captain regret, and the split of the gain between gameweek-1 and gameweek-20 starts.
- **Why whole seasons and not the standard comparison:** the effects are expected to be tens of points, far above the whole-season SE (about 8), and better information pays partly through later gameweeks, which the one-gameweek comparison does not count.
- **Cost:** six backtest runs of about 20–25 minutes each at `--jobs 14` on the server, plus the base run from the cache. About 2.5 hours.

## Reading the result (fixed before any run)

| Ceiling (points a season, lower end of the 80% interval) | Decision |
|---|---|
| ≥ 60 | Worth a source. A sourcing spec follows for that component, with its own test |
| 20 to 60 | Noted. Worth work only if a cheap source exists |
| < 20 | Closed, with the number |

- A real source captures part of a ceiling. M0h gives the first estimate of how fast the gain falls with accuracy; a third is the planning figure until then.
- **M0 against A0** separates lineup news from injury news. Injury news is partly public before the deadline (flags, press conferences); the full lineup is not, since teams are announced after most deadlines.
- **M0 against MA** says whether the value is in the coming gameweek or in the weeks after.
- If every arm is below 20, the gap to 2,400 is not an information problem that we can buy, and the note says so.

## Addendum (2026-11-10): splitting MA by what is knowable

Approved by the owner in chat, 2026-11-10, after the first results (`docs/research/2026-11-value-of-information.md`: MA +264, everything else under the bar). MA knows things no source can supply. Three more arms split it. Written before any of them ran.

| Arm | What is replaced | Seasons | The question |
|---|---|---|---|
| **MAf** | As MA, only for players **flagged at the deadline**: status not `a` in the newest `player_snapshot` of the season visible in the deadline's view | 2021/22–2022/23 | The ceiling for news on known absences (return dates, doubts) |
| **MAu** | As MA, only for players **not** flagged at the deadline (the complement of MAf, including players without a snapshot row) | 2021/22–2022/23 | Rotation, role changes and injuries still to come |
| **M2** | As MA, only for the coming gameweek and the two after (horizon ≤ 2) | 2016/17–2022/23 | How far ahead the knowledge has to reach |

- The flag set is read from the deadline's as-of view, like every v1 input. A deadline with no snapshot of its season yet has an empty flag set (MAf changes nothing there, MAu is MA); the run logs each deadline's row counts.
- **Flag arms, measurement:** flags exist from 2021/22, so two develop seasons only. To cut the noise within a season they run on eight full-season starts (`template@1,random:7@1`), with the base and MA re-run on the same starts. Reported: mean gain per season and over the 16 units, and MAf and MAu as shares of MA on the same units. Two seasons give no usable interval over season means (t with one degree of freedom), so **these two arms are descriptive: no decision threshold**. 2022/23 has been Gate N's test season and Stage C's develop season; it is used here for diagnosis only.
- **M2, measurement:** as the first six arms (same starts, same base run, same interval and thresholds).
- **Reading, fixed now:**
  - MAf ≥ half of MA on these units: the first source to try is FPL's own news text and `chance_of_playing` beyond return dates, with its own spec and test. MAf under a fifth of MA: known absences are closed as a source.
  - MAu carrying most of MA: the value is in rotation and future injuries, largely out of reach; it goes back to the minutes model.
  - M2 against M0 (+86) and MA (+264) places the value along the horizon. If M2's lower bound is ≥ 60, a source that only sees two to three weeks ahead is still worth a spec.
- MAf and MAu need not sum to MA (the planner's choices interact).
- **Cost:** four runs of about 7 minutes and one of about 20, locally. Family `value of information`.

### Club arms (2026-11-10, the owner's question)

After the flag arms (MAf +12, MAu +183 of MA's +203 on the same 16 units), the owner asked whether MA's gain is mostly Manchester City's rotation. Written before either arm ran; what was known then: City has the lowest XI retention to the next league match over the seven seasons (0.71 against a league mean of about 0.80), and with MA the planner captains City players in 26% of gameweeks against 14% without.

| Arm | What is replaced | Seasons |
|---|---|---|
| **MAc** | As MA, only in Manchester City's fixtures (`team_dim.short_name` `MCI`) | 2016/17–2022/23 |
| **MAnc** | As MA, in every other club's fixtures | 2016/17–2022/23 |

- Measured as the first six arms (same starts, same base run, same interval). Descriptive: they locate MA's gain, they are not a source.
- **Reading, fixed now:** MAnc's lower bound ≥ 60 means the pattern is league-wide and City is not a special case. MAc at half of MA or more means a City-specific treatment (a rotation model or a planner discount for that club) is worth its own spec.

## What could follow (not in this spec)

- **If M0 or A0 is large:** a team-news source. The first candidates are ones we may use: press-conference dates and FPL's own news text (already archived, parsed for return dates only), the player odds archived since 2026-10-30, and the user's own `/override`. Predicted-lineup sites are subject to the no-scraping rule and their terms.
- **If R is large:** player odds as a feature once a season is archived, which was the plan (PLAN §3).
- **If T0 is large:** a later odds snapshot, or moving the alert closer to the deadline.

## Risks

- **A ceiling is not a gain.** Gate A's price oracle said +255 and the forecast delivered nothing. These arms avoid that trap in one way (no outcome is leaked, only who plays) but not entirely: who starts is itself partly a consequence of form and fitness the public cannot see.
- **Perfect starts overstate.** Lineups come out after most deadlines, so M0 is not attainable; A0 and M0h are the realistic rows.
- **Final fixture lists** make every arm and the base slightly optimistic before 2026 (PLAN §3). It affects both sides of each pair.
- **Inferred starts before 2022/23** add a little noise to M0 and MA (1.35% of player-fixtures).
