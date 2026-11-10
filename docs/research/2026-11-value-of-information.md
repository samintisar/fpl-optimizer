# Value of information: knowing who plays over the next six gameweeks is the only large input

**Date:** 2026-11-10. **Spec:** `docs/superpowers/specs/2026-11-09-value-of-information-design.md`. **Plan:** `docs/superpowers/plans/2026-11-09-value-of-information.md` (its last two sections list every deviation). **Runs:** seven `backtest run`s on this machine from commit d44c5ce, family `value of information` in `results/experiments.csv`; outputs in `results/voi/` (gitignored); report `dev/info_value.py`.

## Question

If the planner knew one xP input perfectly, how many season points would it gain? Each arm is v1 with that input replaced by the truth, in the backtester only. No arm uses a realized goal, assist, clean sheet or point. The decision rule was fixed before any run, on the lower end of the 80% interval: ≥ 60 worth a source, 20–60 noted, < 20 closed.

## Result

Gain in season points over the live planner (`optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`, no chips), full-season starts (4 a season), mean over season means with an 80% t interval:

| Arm | What it knows | Seasons | Gain | 80% interval | Half-season starts | Decision |
|---|---|---|---|---|---|---|
| **MA** | Who plays and starts in each of the six horizon gameweeks | 7 | **+264** | +203 to +326 | +126 | **source** |
| M0 | Who plays and starts in the coming gameweek | 7 | +86 | +31 to +141 | +72 | noted |
| M0h | M0 for a fixed half of the players | 7 | +67 | +16 to +119 | +43 | closed |
| R | How a team's goals and assists divide between its players (true season rates) | 4 | +53 | −23 to +129 | +9 | closed |
| A0 | Who plays in the coming gameweek, not who starts | 7 | +16 | −30 to +61 | +25 | closed |
| T0 | Team goal expectations from closing odds, coming gameweek | 4 | +4 | −18 to +26 | −3 | closed |

By season (gain over the base; base = the live planner's season total):

| Season | Base | MA | M0 | M0h | A0 | T0 | R |
|---|---|---|---|---|---|---|---|
| 2016/17 | 2,040 | +290 | +146 | +46 | −43 | | |
| 2017/18 | 2,147 | +341 | +86 | −80 | +50 | | |
| 2018/19 | 2,083 | +400 | +206 | +180 | +138 | | |
| 2019/20 | 2,054 | +296 | +197 | +96 | +80 | −8 | +176 |
| 2020/21 | 2,187 | +163 | +29 | +35 | +36 | −25 | +60 |
| 2021/22 | 2,513 | +295 | −28 | +187 | −48 | +14 | −44 |
| 2022/23 | 2,449 | +66 | −35 | +8 | −103 | +37 | +20 |

Checks on the runs: every arm changed every row it was meant to (0 fixtures left unknown; at most 0.01% of rows had no rate to scale). R replaced 59% of player-fixtures (players with at least 450 minutes with xG). T0 and R have 24 base units with no counterpart because they ran on 2019/20–2022/23 only.

## Reading it

1. **Minutes over the horizon is the one large input.** MA gains 264 points a season, positive in all seven seasons, and its lower bound (203) is far above the bar. Nothing else comes close. This agrees with PLAN §6 ("minutes is the biggest lever") and puts a number on it in planner points for the first time.
2. **Most of that is beyond the coming gameweek.** M0 has a third of MA's gain (86 of 264). Knowing next week's lineups is worth much less than knowing who will be available and picked in weeks two to six, because transfers are held for about nine gameweeks and autosubs and the vice-captain already cover a no-show this week.
3. **Injury news without the lineup is worth little** (A0 +16). The step from "plays" to "starts" carries the value: rotation and bench roles, not absences.
4. **A better team model has nothing left to give at one gameweek.** Closing odds in place of deadline odds: +4 (−18 to +26). The market at the deadline is already as good as the market at kick-off for this purpose.
5. **Player scoring rates are unresolved.** R's mean is +53 but its interval covers zero on four seasons, and it is carried by 2019/20, when the Understat data behind v1's rates was thinnest. By the rule it is closed. It is also a weaker oracle than the others (it only re-splits a team's total among players with season rates).
6. **The per-season numbers are too noisy to split by era.** A season's gain on four starts moves by 100 points or more between arms that should be ordered: M0h (half of M0's knowledge) beats M0 by 215 points in 2021/22 and loses to it by 165 in 2017/18. So the pattern "M0 is worth nothing once status flags exist" (−28 and −35 in 2021/22 and 2022/23) is two noisy seasons, and I would not act on it. MA is the exception: its gain is large enough to show in every season, including +295 in 2021/22 with flags.

## What MA is not

MA is a ceiling, and a loose one:

- **It knows injuries that have not happened yet.** A player hurt in next week's match is already marked absent for the weeks after. No source can supply that.
- **It knows the manager's future selections**, including rotation that the manager has not decided.
- **What a source could supply is narrower:** return dates of players already injured or suspended, who is first choice in a position after a transfer or a managerial change, and planned rotation around cup and European matches. The first of these was measured before and dropped (PLAN, Phase 5: P(start) is already calibrated within injury groups), but that test scored calibration, not planner points.

So the result says where to look, not what is attainable. The next question is how much of the 264 sits in information that exists at the deadline.

## Decisions

| Input | Decision | Why |
|---|---|---|
| Who plays over the horizon (MA) | **Source: goes to a follow-up spec** | Lower bound 203 ≥ 60 |
| Next-gameweek lineups (M0) | Noted | Lower bound 31; a real lineup source is right far less than always, and M0h's lower bound is under 20 |
| Availability only (A0) | Closed | +16, interval covers zero |
| Closing odds (T0) | Closed | +4 |
| Player rates (R) | Closed by the rule; revisit with the props archive | Interval covers zero on four seasons; PLAN §3 already plans to test player odds once a season is archived |

## Proposed next step (the owner's call)

Split MA by what is knowable at the deadline, with the same code and three more runs of about 20 minutes each:

1. **MA on players flagged at the deadline only** (status not `a` in the newest snapshot): the ceiling for return dates and news on known absences. 2021/22–2022/23 only, since flags start in 2021.
2. **MA on unflagged players only:** rotation and injuries still to come. The unknowable part is in here.
3. **MA at horizons 1–2 only:** how far ahead the knowledge has to reach.

If (1) is large, the first source to try is the one already in the archive: FPL's own news text and `chance_of_playing`, which v1 parses for return dates only. If the value is mostly in (2), it is largely out of reach and the minutes model's own accuracy (the research loop's ground) is the only way at it.

## Caveats

- Develop seasons only; no validate, no holdout. A ceiling is not a policy and needs neither.
- Before 2026 the backtest sees the final fixture list (PLAN §3), in the base and in every arm.
- Starts before 2022/23 gameweek 16 are inferred from minutes (98.65% accurate).
- Whole-season totals on four starts a season: the SE over season means is 32–46 points for the seven-season arms, so only effects of about 60 points or more are resolved. T0's is 13 because its decisions rarely differ from the base's.
- With any oracle the planner takes more hits (18–25 hit points a season against 12): better information makes more moves look worth four points.
