# The research loop

A scheduled agent that, every night, monitors the live season and searches for improvements
on develop, under fixed rules. It proposes; you promote. The agent's instructions are
[`program.md`](program.md); this page is for you.

## How it fits together

```
dev PC (Claude, research clone, branch `research`)
  │  edits, tests, commits; guard hook in .claude/settings.local.json
  │  bash autoresearch/sync.sh        code → server; raw: botbox → server (read only)
  ▼
fplopt-server (3090 box, 24 threads, 62 GB)
  ~/fpl-optimizer   hub repo (receives pushes)
  ~/fpl-research    `research` / `try/*` detached; data/ = develop seasons only
  ~/fpl-live        `main` detached (the bot's code); raw/ from botbox; full data/
  ~/fpl-harness     the scoring (model_eval.sh, dev/calibrate_v1.py, judge.py), pinned by you
```

The scoring runs from `~/fpl-harness`, not from the candidate's checkout, so a candidate
can't change how it is judged; the hook runs from `C:\Users\samin\.fpl-harness\guard.py`
for the same reason. Updating either is yours:

```bash
git push -f fplopt-server:fpl-optimizer <commit>:refs/heads/harness
ssh fplopt-server 'git -C ~/fpl-harness checkout -q --detach harness'
cp autoresearch/guard.py ~/.fpl-harness/guard.py
```

- **Nightly** (scheduled task in the desktop app): orient, the weekly monitor if a GW has
  been checked since the last report, then experiments until 07:00 local.
- **Weekly monitor** (`weekly.sh`): rebuild `data/`, score v1 and FPL's `ep_next` on 2026/27
  so far, slice the misses, write `reports/<date>-gw<N>.md` with ideas for develop.
- **Experiments**: model changes go through the frozen harness (`model_eval.sh`, v1's
  walk-forward xP on develop, ~3 min) and judge (`judge.py`, fixed keep rule); planner
  changes through `backtest compare` on develop. Every one goes in `ledger.tsv`.
- **Guards**: develop-only `data/` where experiments run; the hook (`guard.py`) refuses
  validate/holdout/live seasons, pushes other than `sync.sh`, `gh` writes, botbox, secrets
  and edits to the frozen files.

## Your part

- **Read the weekly report** (`reports/`, also summarized at the end of the task's run).
- **Promote** a candidate by asking in an interactive session: one validate look (component
  metrics, then the paired decision backtest, PLAN §9 Phase 8), deflated for the variants in
  `ledger.tsv`, a PR to `main`, review, merge, deploy to botbox.
- **Keep the PC awake overnight**: the task runs while the desktop app is open; a missed
  run starts at the next launch.
- **Change the rules** yourself: `program.md`, the judge, the harness and the guard are
  frozen for the loop.

## Setup (done 2026-10-14; repeat on a new machine)

1. `git worktree add ../fpl-autoresearch research` from the repo (branch `research`).
2. Copy `autoresearch/guard.py` to `~/.fpl-harness/guard.py`; in the research clone,
   `.claude/settings.local.json` with the hook (PreToolUse on Bash, PowerShell, Edit, Write,
   NotebookEdit: `python ~/.fpl-harness/guard.py`) and an allow-list for the loop's commands.
3. On fplopt-server: `git -C ~/fpl-optimizer worktree add --detach` for `~/fpl-research`,
   `~/fpl-live` and `~/fpl-harness` (the last at `harness`, above); copy `raw/` into
   `~/fpl-live`; `bash autoresearch/sync.sh all`; then `bash autoresearch/weekly.sh` once in
   `~/fpl-live` (builds data/ and the develop copy).
4. The scheduled task (desktop app), nightly, pointed at the research clone.
