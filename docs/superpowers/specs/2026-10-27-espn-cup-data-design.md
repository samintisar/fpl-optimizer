# Cup and European matches from ESPN — design

2026-10-27. Approved in chat; source of the implementation plan.

## Why

The v1 minutes model knows only league fixtures (PLAN §6.3: "European/cup matches … no
source"). Two consequences:

1. **Bans are served in the wrong games.** `fplopt.features.history` serves every red-card
   ban in the club's next *league* fixture. A red in a domestic match bans the club's next
   *domestic* matches (PL, FA Cup, EFL Cup), so a ban is often served in a cup tie. Live
   case: Palmer (Chelsea) was sent off in GW8 v Tottenham, serves the ban in the EFL Cup at
   Liverpool on 2026-10-28 and is available for GW9 (FPL: status `a`, 100%); the bot gives
   him P(play) 0.07 in GW9. The reverse case (a cup red banning the next PL match) is missed
   entirely, and every red counts as one match (straight reds are mostly three).
2. **Congestion and cup minutes are invisible.** Rest days count league fixtures only; a
   midweek European match, a rotation player starting a cup tie, or a player returning from
   injury in a cup game do not reach the model.

## Source

ESPN's public site API (unofficial, no key), probed 2026-10-27 on EFL Cup 2016/17,
FA Cup 2016/17, UCL 2017/18, UEL 2023/24, EFL Cup 2025/26 and 2026/27:

- `site.api.espn.com/apis/site/v2/sports/soccer/<league>/scoreboard?dates=YYYYMMDD` — the
  day's matches (past and scheduled);
- `…/<league>/summary?event=<id>` — `rosters` (per team: `starter`, `subbedIn`,
  `subbedOut`, athlete id and name), key events with clocks (substitutions, cards) and per
  player card counts in most competitions.

Leagues: `eng.1`, `eng.fa`, `eng.league_cup`, `eng.charity`, `uefa.champions`,
`uefa.europa`, `uefa.europa.conf`, `uefa.super_cup`.

Rejected: ESPN cup/European matches only with name matching (fragile: "Gabriel", youth
players in EFL Cup ties); Transfermarkt / FBref (scraping blocked, terms forbid it).

Risk: the API is undocumented and may change. Everything fetched is archived in `raw/`, so
a change stops new data, never history; the adapter isolates the format.

## 1. Data

**Adapter** `src/fplopt/adapters/espn.py`: `EspnClient` over `fplopt.adapters.http`
(`scoreboard(league, date)`, `summary(league, event_id)`, raw bytes), a polite rate limit
(~1 request/s) and retries like the other adapters.

**Raw** (append-only, compressed, timestamped):
`raw/espn/<league>/<season>/scoreboard_<YYYYMMDD>_<fetched_at>.json.gz` and
`raw/espn/<league>/<season>/<event_id>_<fetched_at>.json.gz`.

- `fplopt backfill espn [--seasons 2016-2026]`: walks each season's scoreboards (1 July –
  30 June) per league, keeps matches with a club in `team_dim` for that season, fetches
  their summaries; resumable (skips event ids already archived with a finished status).
  ~7,000 summaries, 2–3 h. Runs on the server (`deploy/README.md`).
- Live: the daily job (`fplopt snapshot daily`) also fetches the scoreboards from 7 days
  back to 35 days ahead for every league and the summaries of finished matches not yet
  archived. The future scoreboards are the as-of record of scheduled cup fixtures.

**Tables** (`fplopt.build.tables.TABLES`, built in `src/fplopt/build/espn.py`):

| table | kind | key | contents |
|---|---|---|---|
| `espn_match` | event | `espn_event_id` | league, season, kickoff, home/away `team_key` (null for non-FPL clubs), status, `fixture_key` for `eng.1` |
| `espn_player_match` | event | `espn_athlete_id`, `espn_event_id` | season, team_key, starter, minute on, minute off, minutes, yellows, red, second_yellow |
| `espn_schedule_snapshot` | snapshot | `snapshot_at`, `espn_event_id` | season, league, kickoff, home/away `team_key`, as published (live archive only) |
| `espn_map` | static | `espn_athlete_id` | `player_key`, names (public: id, player_key) |

- `event_time` = kickoff; `available_at` = kickoff + 3 h (**VERIFY**: ESPN's lineups and
  cards are final by then; check on a week of live matches).
- Minutes: a starter plays from 0, a substitute from his `subbedIn` clock; off at his
  `subbedOut` clock or a red card, else the match end (90, or 120 after extra time;
  stoppage time ignored like FPL).
- `second_yellow`: a red card with an earlier yellow for the same player in the match.
- `config/teams.csv` gains an `espn` column (club names).

**Mapping** (`espn_map`, all seasons at once): the Understat rules (`fplopt.build.understat`)
applied to `eng.1` matches — candidates share a PL fixture with both appearing; accept on
minutes agreement (|Δ| ≤ 5) and overlap, or overlap + name, or name alone with a common
fixture; greedy one-to-one by coverage; `config/overrides.csv` rows with `source=espn` win.
ESPN athlete ids are stable across seasons and competitions, so the PL mapping carries to
cup rows. Athletes never in a PL match (opponents' players, academy players who never played
PL) stay unmapped and are dropped from features.

**Build checks** (fail the build, listing offenders):

- coverage: every finished PL fixture from 2016/17 has an `eng.1` match;
- one-to-one mapping; every FPL player with PL minutes > 0 is mapped (exceptions by
  override);
- minutes oracle: each pair's ESPN and FPL PL minutes agree (|Δ| ≤ 5) in ≥ 90% of common
  fixtures;
- cards: ESPN PL reds equal `player_match.red_cards`.

## 2. Bans

Rules (Premier League / FA handbook, checked 2026-10-27): red-card bans apply across the
PL, FA Cup and EFL Cup; yellow-card accumulation bans stay in the competition where the
cards were shown; UEFA bans apply to UEFA matches only.

In `fplopt.features.history`:

- **Domestic red** (any of `eng.1`, `eng.fa`, `eng.league_cup`): bans the club's next
  domestic matches in date order (league or cup). Length: second yellow → 1 match; straight
  red → match 1 **certain**, matches 2–3 **likely** (a straight red is 3 matches except
  denying an obvious goal-scoring opportunity, 1; ESPN does not tell them apart).
- **PL yellows** (5 by the club's 19th PL match → 1, 10 by the 32nd → 2, 15 → 3): unchanged,
  PL cards and PL fixtures only. Cup yellows ignored (they bar cup matches only).
- **European cards**: ignored.
- **FPL return dates** ("Suspended until DD Mon"): still win where present.

Frames: `ban_remaining` becomes `ban_certain` and `ban_likely` — the PL fixtures still to
miss after the domestic cup matches in between consume part of the ban. Training rows carry
`banned_certain` / `banned_likely`; prediction rows count the club's domestic cup matches
known as of the deadline (§3, *Visibility*) between the deadline and each horizon fixture.

`fplopt.models.minutes`: the `ban_residual` post-processing is fitted separately per kind
(`MinutesFit.ban_start` / `ban_sub` per certain / likely), so the data says how often a
"likely" ban is real (and absorbs rescinded reds and appeals).

## 3. Congestion and cup-minutes features

**Visibility of future non-league matches** (prediction frame and ban counting):

- live archive: a match is known at a deadline if an `espn_schedule_snapshot` before the
  deadline lists it;
- before the archive: a club's match in a competition is known once the club's previous
  match in that competition has kicked off (the draw follows the previous round); a club's
  first match in a competition (or phase: league phase, group stage) is known 14 days
  before kickoff. **VERIFY** with the leakage check (corrupt-the-future on the new
  features) and against the live archive once it has a few weeks.

**Features** (`LAG_FEATURES`; frozen at the deadline for every horizon fixture except the
schedule-based ones, which are per fixture):

- `rest_days_all`: days since the club's previous match in any competition (cap 30); the
  league-only `rest_days` stays;
- `days_to_next`: days from the PL fixture to the club's next known match in any
  competition (cap 30);
- `europe_week`: 1 if the club has a known European match within 4 days before or after
  the PL fixture;
- `cup_minutes_7d`: his minutes in non-league matches in the 7 days before the deadline
  (training: before the row's GW deadline);
- `cup_started_last`: 1 if he started the club's latest non-league match since his last PL
  row, 0 if he was in the squad and did not start, NaN if none;
- `days_since_app`: now counts non-league appearances too.

Dropped from PLAN §6.3's list stays: manager tenure, age.

## Validation and rollout

Three PRs:

1. **Data**: adapter, backfill, tables, mapping, build checks, daily fetch. The backfill
   runs on the server after merge.
2. **Bans**: the rules above, the split residual, the prediction-frame ban counting.
   Tests: Palmer (red in GW8, EFL Cup before GW9 → GW9 clear), an EFL Cup red banning the
   next PL fixture, second yellow vs straight red, FPL return date still winning. Real
   data: the rule-banned rows' start rate per kind vs main (~10% today).
3. **Features**: the new lags, minutes model refit.

PRs 2 and 3 each: calibration refit (`dev/calibrate_v1.py walk/fit --write`),
`fplopt check leakage`, and `models eval` paired against main on develop, then validate
(2023/24–2024/25), then live 2026 (monitor only) — the #47 gates: MSE h0 and h1–5 not worse
on validate, captain regret not significantly worse. Target: PRs 1–2 deployed before the
GW9 alert (2026-10-30 11:00 UTC); PR 3 when it passes the gates.

The research loop's develop data (`autoresearch/develop_data.py`) keeps the new event tables
by their `season` column; `espn_map` is static.

PLAN §6.3 and the `history.py` docstring are updated with the change ("Dropped (no
source)" loses European/cup matches).
