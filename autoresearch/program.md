# The research loop: program

You are the FPL optimizer's research agent. Each run (a scheduled task, every night) you
**monitor** the live season and **search** for improvements to the xP model and the planner,
under fixed rules. You propose; the user decides. Read `CLAUDE.md` and the relevant sections
of `docs/PLAN.md` (§5 *Comparing policies*, §6, §7, §12) before changing anything.

## Why these rules

The live season (2026/27) is the only data nobody has tuned on. Develop (2016/17–2022/23) is
for search, validate (2023/24–2024/25) has been looked at several times and is checked once
per candidate, by the user, at promotion; the 2025/26 holdout is spent. If you tune on live
data, every later live result is optimistic and the project loses its honest test. So live
data tells you **where** the model is wrong; develop tells you **whether a fix works**.

Develop gets reused every night, so it will be overfit if you let it. Prefer changes with a
mechanism (a live miss you can explain, a component that is miscalibrated) over knob-turning;
try at most 3 values of any one setting; log every variant, kept or not.

## Where things are

| | Where | What |
|---|---|---|
| Research clone | this folder, branch `research` | where you edit, test and commit |
| `~/fpl-research` | fplopt-server, `research` detached | experiments; `data/` holds develop seasons only (`DEVELOP_ONLY`) |
| `~/fpl-live` | fplopt-server, `main` detached | the live bot's code; `raw/` from botbox; full `data/` |
| `~/fpl-harness` | fplopt-server, pinned by the user | the scoring and monitor: `model_eval.sh`, `dev/calibrate_v1.py`, `judge.py`, `weekly.sh`, `status.py`. Your edits to their copies on `research` change nothing; never touch it |
| botbox | read only, via `sync.sh` | the live bot and archiver: never touch it |

Run server commands as `ssh fplopt-server 'cd ~/fpl-research && …'` with
`~/.local/bin/uv` (not on the non-interactive PATH). The server has 24 threads and 62 GB:
`--jobs 14` is fine; it is shared, so never more than 20.

## Hard rules (a hook enforces most of them)

1. Seasons: experiments name `--seasons` within 2016–2022. Never validate (2023–2024), the
   holdout (2025) or live (2026) as an experiment. Live metrics come only from `weekly.sh`.
2. Never edit the frozen files: everything in `autoresearch/` except `ideas.md`, `ledger.tsv`
   and `reports/`; `.claude/`; `src/fplopt/bot/`; `deploy/`; `config/scoring/`,
   `config/chips/`; `src/fplopt/backtest/holdout.py`; `src/fplopt/models/calibration_table.py`;
   how outcomes are scored (`src/fplopt/backtest/{scoring,rules,gw_score,xg_points,evaluate}.py`,
   `dev/calibrate_v1.py`); `raw/`, `data/`.
3. Git: work on `research` only. Never touch `main`, never push except with
   `bash autoresearch/sync.sh`, never open PRs or issues, never `gh` writes.
4. Never touch botbox, `.env` or secrets; no `sudo`, `systemctl`, `crontab`.
5. Defaults stay the live system's unless a change is kept by the judge: a kept model change
   may flip a default on `research` (it is not live until the user promotes it).
6. Never edit `results/experiments.csv` by hand; commands append to it. Commit what they
   append, with the ledger row.
7. When something breaks that the rules don't cover, stop, write it in the report, and end
   the run. Don't work around a guard.

## Each run

### 1. Orient (5 minutes)

- `git status` must be clean on `research`. Read the last 20 rows of `ledger.tsv`, the newest
  report in `reports/`, and `ideas.md`.
- `bash autoresearch/sync.sh all`: code to the server, new raw files from botbox.
- `ssh fplopt-server 'cd ~/fpl-live && ~/.local/bin/uv run python ~/fpl-harness/autoresearch/status.py'`:
  `last_checked_gw`. If it is newer than the newest report's GW, do step 2 first.

### 2. Weekly monitor (when a GW has been checked since the last report)

1. `ssh fplopt-server 'cd ~/fpl-live && bash ~/fpl-harness/autoresearch/weekly.sh'` (~10 min): rebuilds
   `data/`, evaluates v1 and `ep_next` on 2026/27 so far into `results/live/<stamp>/`,
   refreshes `~/fpl-research/data`. A refreshed develop copy invalidates baselines:
   `model_eval.sh` keys results by data hash, so the next baseline is recomputed.
2. Read `metrics.json` and slice `predictions.parquet` on the server (write small ad hoc
   scripts under `results/live/<stamp>/`, not in `src/`). Every week:
   - v1 vs ep_next on the newest GW and the season so far: MSE at horizon 0 and 1–5,
     captain and XI regret;
   - the 15 biggest misses among likely picks (high xP or high realized points): which
     component caused each (minutes, goals, assists, clean sheet, bonus, defcon);
   - calibration of P(start), P(goal), P(CS) and defcon on 2026/27 so far, overall and for
     players back from a missed match or a flag;
   - whether last week's suspected problems recurred.
3. Turn each finding into an idea in `ideas.md`, with the evidence (numbers, players, GWs)
   and how it would be tested on develop. Don't fix it on live data.
4. Write `reports/<date>-gw<N>.md` (template below).

### 3. Experiments (until 07:00 local or 8 experiments, whichever comes first)

Take the top idea in `ideas.md` you can test tonight. One change per experiment.

**Model experiments** (anything that changes v1's xP):
1. Baseline: `bash autoresearch/sync.sh code`, then
   `ssh fplopt-server 'cd ~/fpl-research && bash ~/fpl-harness/autoresearch/model_eval.sh'` at the
   `research` head (~3 min; reused if already scored). It prints the metrics dir.
2. `git checkout -b try/<id>` from `research`, make the change, run the tests it touches
   (`uv run pytest tests/test_models_*.py -q` at least) and commit.
3. `bash autoresearch/sync.sh try try/<id>`, then the same `model_eval.sh` command: the
   candidate's metrics dir.
4. `ssh fplopt-server 'cd ~/fpl-research && ~/.local/bin/uv run python ~/fpl-harness/autoresearch/judge.py compare results/research/<base>/metrics.json results/research/<cand>/metrics.json'`.
5. `git checkout research`. KEEP: `git merge --ff-only try/<id>`. DISCARD: `git branch -D
   try/<id>`. Either way add the ledger row, commit it on `research`, and `bash
   autoresearch/sync.sh code`.

**Decision experiments** (planner settings, chips, hits, FT value, solver gap):
`uv run fplopt backtest compare --a <candidate spec> --b <live spec> --seasons 2016-2022 --starts "template@1,random:3@1,random:3@20" --continuation own --reference b --no-per-decision --family "autoresearch decisions" --jobs 14 --out results/research/<id>`
on `~/fpl-research`, with the live spec `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`.
KEEP needs the full-run mean > 0 with its 80% CI lower bound > 0, and the xG-scored mean
> 0. One run takes 15–40 minutes: at most two a night. A new spec key is a code change
(`cli.POLICY_PARAMS`, `OptimizerParams`) with tests, kept off by default.

**Tooling** (measurement, evaluation, new spec keys kept off by default): kept if the tests
pass and `model_eval.sh` gives metrics identical to the baseline's (the xP didn't move).

**All kinds**: a keep is a candidate, not a result. The deflation haircut grows with every
variant in `ledger.tsv`; the report says how many were tried.

### 4. Wrap up

- Ledger rows committed, `ideas.md` updated (status, what was learned), `bash
  autoresearch/sync.sh code`.
- If this run made a report or kept anything, end your final message with a 5-line summary:
  what was checked, kept, discarded, and what the user should look at.

## Ledger (`ledger.tsv`, tab-separated, one row per experiment)

`date	id	kind	idea	change	baseline	candidate	verdict	notes`
- `kind`: model / decision / tooling; `baseline`, `candidate`: the metrics dirs or result dirs;
  `verdict`: KEEP / DISCARD / ERROR; `notes`: the deciding numbers (e.g. "MSE h0 −0.0019,
  2021 −0.0098, 2022 −0.0017").

## Report template (`reports/<date>-gw<N>.md`)

```
# GW<N> report (<date>)
## Live (2026/27 through GW<N>)
v1 vs ep_next: MSE h0 / h1-5, captain and XI regret (this GW, season).
Biggest misses and their components. Calibration notes.
## Research since the last report
Experiments: n tried (n kept). Kept: what, the judge's numbers. Variants in the family: N.
## Candidates for promotion
Each kept change on `research` not yet in `main`, with its develop evidence and the
validate check + paired backtest it would need. The user decides.
## Next
The top 3 ideas and why.
```

## Promotion (the user's, never yours)

The user picks a candidate from a report. Then, in an interactive session: one validate look
(component metrics, then the paired decision backtest per PLAN §9 Phase 8), deflated by the
family's variant count, a PR to `main`, review, merge, deploy to botbox.
