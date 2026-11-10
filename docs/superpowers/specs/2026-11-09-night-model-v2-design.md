# Night price model v2: evening snapshots, repaired features, nightly recalibration

**Date:** 2026-11-09. **Status:** closed unbuilt by the owner, 2026-11-10 (the review was closed; `docs/research/2026-11-system-review.md`, *Closed*). Nothing is built. Review: `docs/research/2026-11-system-review.md` §1.

## Why

Gate N failed on the calibration of rises. The review found that about half of that failure was an artefact, and that the rest has identifiable causes:

- Gate N's season, 2022/23, contains a 44-night price freeze (the World Cup). It holds 179 of the 1,046 rows the rise condition was judged on, with no rise possible. Before the freeze the model was calibrated at T's threshold (0.63 predicted, 0.56 realized, 126 rows).
- The transfer counter drops 15% of every gameweek's net transfers, and nothing is scaled for the number of managers (24% more in 2022/23 than in the training season).
- From mid-January 2023 rises still ran at 0.25 against 0.44 predicted. Refitting the calibration map on the season's own last 60 nights brought the whole season to 0.36 against 0.46 at p ≥ 0.3 and 0.56 against 0.62 at p ≥ 0.5. That check was made after seeing the data.

**Size:** timing planned moves has a ceiling of about 19 points a season (£2.5–3.2m paid for price moves in the week before a transfer, review §1.1). A model that catches a third to a half, less the cost of deciding on older news, is worth **+3 to +8 points a season**. This spec is worth doing only because the model and policy code already exist.

## Part 0: evening snapshots (needs the owner's go-ahead for the server)

The 2026/27 archive has one `bootstrap-static` snapshot a night (about 02:00 UK) and one before each deadline. A night model needs the evening state.

- The archiver takes two more `bootstrap-static` snapshots a day: **22:00 UK** (the decision moment) and **01:00 UK** (the latest useful state before the change). Two requests a day, same paths and format as today's.
- `build` adds `total_players` from each bootstrap into a new table `game_snapshot` (`snapshot_at`, `season`, `total_players`, `available_at` = `snapshot_at`). fplcache's raw files carry the same field, so history is covered.
- Nothing downstream changes. This part can ship on its own, and each night without it is a night the gate below cannot use.

## Part 1: features (`fplopt_features::night`)

1. **Exact gameweek totals.** At each passed deadline the offset adds the player's final `transfers_in − transfers_out` for that gameweek from `player_gw_ownership` (visible after the deadline), not his last snapshot's value. Where that row is missing, the last snapshot's value as today.
2. **Manager scaling.** `net_since_change`, `net_24h` and `net_event` are divided by `total_players` / 10⁷ from the newest `game_snapshot`. `net_per_own` becomes net transfers per owner (owners = ownership share × `total_players`).
3. **Frozen periods.** A night belongs to a freeze when no price in the game changed on it and on the 6 nights before it. Freeze nights are not training rows and are not scored, and every player's counter restarts at the first night after a freeze. In the data this is 2022-11-12 to 2022-12-26 only. The rule reads only past nights, so it is valid as of each night.
4. **New column** `hours_since_snapshot`, because training snapshots (fplcache, about 18–19 UK) are older at the decision moment than live ones.

`check leakage` covers the frame as it does today.

## Part 2: nightly recalibration (`fplopt_models::night`)

- The boosters and the out-of-fold Platt map are fitted as now, on the training seasons.
- Each morning the map for the coming night is refitted on the current season's last 60 labelled nights: a logistic regression of the label on the logit of the training-calibrated probability. Until the season has 30 labelled nights with at least 20 events of the target, the training map is used unchanged.
- One setting (60 nights), fixed here. No grid.

## Fits and data

| Use | Seasons |
|---|---|
| Training | 2020/21's snapshot nights, 2021/22, 2022/23 (now a training season; it is spent as a test season) |
| Tuning | None. Booster parameters stay the plan's defaults; the only new setting is fixed above |
| Sanity check, not a gate | Train on 2021/22, score 2022/23 with parts 1–2, reported next to Gate N's table. It cannot pass or fail anything: 2022/23 chose these repairs |
| **Gate N2** | **2026/27 nights after this spec's tag**, prospective |
| Policy T's check | 2023/24–2024/25, once, only after N2 passes and T passes on develop |
| Never | 2025/26 |

2023/24–2024/25 night labels are not read before T's check.

## Gate N2 (prospective, run once)

- **Rows:** every pool player on each night from the first evening snapshot after the tag `night-model-v2-gate` until 60 nights are labelled. The label is as in Gate N.
- **Model:** fitted once at the tag on the training seasons plus 2026/27's nights before the tag (fplcache and our snapshots); the map recalibrated nightly as in part 2. Predictions are written each night to `state/` before the morning snapshot exists, so nothing can be revised.
- **Pass, per target, all of:**
  - log loss below the base rate and below the net-transfers-per-owner booster, one-sided Diebold–Mariano by night, lag 7, p < 0.10;
  - among rows with p ≥ 0.3: at least 30 rows, and the observed rate ≥ mean p − 0.10.
- **Reported only:** the same at p ≥ 0.5, per month, and the count of rows above each threshold a night (the volume T would act on).
- **Run rules:** as Gate N. Family `night price model gate n2`, logged before anything is printed, refused on a dirty checkout or off the tag.
- **Outcomes:** both targets pass → T is built for both sides. Only falls pass → T for sales only. Neither → the timing line is closed, with the number that closed it.

## After N2: policy T

As specified in `2026-11-09-move-timing-design.md`, unchanged: a planned move is made at night when its expected saving is at least τ = 0.5 tenths, and the price never decides who is moved. Develop is 2021/22–2022/23 (value gain V from the snapshots, information cost C from the comparison, net > 0 at one-sided p < 0.10), then one run on 2023/24–2024/25 with the same gate. Since the standard comparison counts one gameweek's points, V and C are reported separately as that spec already says.

## Not in this spec

- Any use of price in choosing players. A tie-break among near-equal plans is sized separately, with oracle prices first (review §5, item 7).
- The bot's "move tonight" alert. It follows T's check, with its own spec.

## Risks

- **Sixty nights may hold too few confident rises.** At 2022/23's recalibrated rate (343 rows a season at p ≥ 0.3) 60 nights give about 80, above the 30 the condition needs; if fewer than 30 arrive the gate waits for more nights, to at most 120.
- **Training and live snapshots differ in age.** `hours_since_snapshot` is the mitigation; the nightly map absorbs a level shift.
- **Wildcard and Free Hit transfers** are in the public counters and are believed not to count. Not fixable from our data; the nightly map absorbs part of it.
- **The prize is small.** If N2 passes and T nets +3 points, that is a real but minor gain. The review's chip and planner items are larger.
