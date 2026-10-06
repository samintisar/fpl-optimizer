# Deploying the archiver (Ubuntu)

Runs as your normal user with systemd **user** timers. Raw data lands in `~/fpl-optimizer/raw/` (gitignored).

## One-time setup

1. Clone and install. The repo is private, so give the server a read-only deploy key:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N "" -C "$USER@$(hostname) deploy"
   printf "Host github.com\n    IdentityFile ~/.ssh/github_deploy\n    IdentitiesOnly yes\n" >> ~/.ssh/config
   ```
   Add `~/.ssh/github_deploy.pub` as a read-only deploy key from a machine where `gh` is logged in
   (`gh repo deploy-key add <pubkey-file> --repo samintisar/fpl-optimizer`), then:
   ```bash
   git clone git@github.com:samintisar/fpl-optimizer.git ~/fpl-optimizer
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"   # put uv on PATH in this shell
   cd ~/fpl-optimizer && uv sync --locked
   ```
2. Telegram alerts: create a bot with @BotFather, send it any message, then read your chat id from
   `https://api.telegram.org/bot<token>/getUpdates` (`message.chat.id`).
3. Create `~/fpl-optimizer/.env` (then `chmod 600 .env`):
   ```
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_ADMIN_CHAT_ID=...
   ODDS_API_KEY=...
   ```
   `ODDS_API_KEY` is optional (free tier at the-odds-api.com, 500 credits/month); without it odds are skipped.
   `FPLOPT_RAW_DIR` is optional (default `raw`, relative to the repo).
4. First daily run, by hand. On a fresh server this is more than a smoke test: besides fixtures, bootstrap, odds and the football-data CSV, it archives `event/{gw}/live/` for every GW finalised so far and does a full element-summary run (about 700 requests, ~5 minutes). Run it **outside** the 2 hours before a deadline: its bootstrap snapshot counts as that window's pre-deadline snapshot, so the tick would skip it.
   ```bash
   .venv/bin/fplopt snapshot daily && ls raw/fpl/bootstrap-static
   ```
5. Alert test (should fail and send a Telegram message; the empty `ODDS_API_KEY` avoids spending odds credits):
   ```bash
   ODDS_API_KEY= FPLOPT_RAW_DIR=/proc/fplopt-alert-test .venv/bin/fplopt snapshot daily; echo "exit=$?"
   ```
6. Install and start the timers:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/systemd/* ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now fplopt-daily.timer fplopt-tick.timer
   sudo loginctl enable-linger "$USER"   # keep timers running when logged out
   ```
   If a unit fails for any reason, including a broken venv where the CLI can't even start, `fplopt-failure@.service` sends a Telegram message. A job failure the CLI already alerted on will therefore alert twice. That's intentional.
7. Optional: element-summary by hand (issue #13, ~5 minutes). On a fresh server step 4 already did a full run, and the daily job keeps it fresh (see below), so this is only for forcing a new run, e.g. after step 4's run failed, without waiting for the next daily run. Same deadline caveat as step 4:
   ```bash
   .venv/bin/fplopt backfill element-summary
   ```
   Each run writes `raw/fpl/element-summary/<timestamp>/_manifest.json.gz` listing expected, written and failed players.

## Operating

- Timers: `systemctl --user list-timers 'fplopt-*'`
- Logs: `journalctl --user -u fplopt-daily.service -u fplopt-tick.service --since today`
- Update: `cd ~/fpl-optimizer && git pull && uv sync --locked` (re-copy units if `deploy/systemd/` changed, then `systemctl --user daemon-reload`)
- **Units changed in Phase 1a:** `fplopt-daily.service` now has `TimeoutStartSec=40min` (was 20min). On a server set up before that, re-copy the units and reload when deploying:
  ```bash
  cp deploy/systemd/* ~/.config/systemd/user/ && systemctl --user daemon-reload
  ```
- Odds credits: each odds snapshot logs `credits remaining=…`; a warning is logged below 50 (free tier: 500/month; ~80 used).

### What the daily job archives

Steps run independently: one failing doesn't stop the others, and the job fails (and alerts) naming each failed step.

1. FPL `fixtures/` and `bootstrap-static/`.
2. Odds (if `ODDS_API_KEY` is set).
3. football-data.co.uk's CSV for the season in progress, `raw/football-data/E0/<code>/` (e.g. `2627`). A 404 in July/August means the new season's file isn't published yet and is only logged.
4. Post-lockdown, driven by the GWs that the latest bootstrap marks `data_checked` (FPL has confirmed bonus points):
   - `event/{gw}/live/` once per finalised GW, under `raw/fpl/event-live/<season>/<gw>/` (e.g. `2026-27/5`). GWs that are already archived are skipped, and failed GWs are retried the next day.
   - A fresh element-summary run (all players' per-GW history, about 700 requests, ~5 minutes) when a GW has been finalised since the newest *complete* run. Its manifest also records `season` and `through_event`. A run with failures, without a manifest, or with an older manifest that lacks these keys doesn't count, so the next day tries again. The first daily run after deploying Phase 1a therefore does a full run.
   - Does nothing on days when no new GW has been finalised.

The element-summary run has to fit in the daily unit's `TimeoutStartSec=40min`. If systemd stops the job (timeout or `systemctl stop`), the CLI turns SIGTERM into a normal exit so the run still writes its manifest; the run is incomplete, so the next day redoes it. A run killed outright (SIGKILL, power loss) leaves no manifest and is likewise redone.

## Historical backfills (one-off, Phase 1a)

Run these on the server as transient user units, so they survive SSH disconnects and send the usual failure alert. Lingering is already on (setup step 6).

> Verified on the server (systemd 255) on 2026-10-06: the `OnFailure=` property is accepted on transient units. `systemd-run` doesn't expand `%h`/`%n` on its command line, so these commands use `$HOME` and literal unit names. `fplopt-failure@<instance>.service` reads the instance (`%i`) as the name of the failed unit and passes it to `journalctl -u`.

```bash
cd ~/fpl-optimizer && git pull && ~/.local/bin/uv sync --locked

systemd-run --user --unit=fplopt-bf-football-data --working-directory="$HOME/fpl-optimizer" \
  -p OnFailure=fplopt-failure@fplopt-bf-football-data.service \
  "$HOME/fpl-optimizer/.venv/bin/fplopt" backfill football-data
systemd-run --user --unit=fplopt-bf-vaastav --working-directory="$HOME/fpl-optimizer" \
  -p OnFailure=fplopt-failure@fplopt-bf-vaastav.service \
  "$HOME/fpl-optimizer/.venv/bin/fplopt" backfill vaastav
systemd-run --user --unit=fplopt-bf-fplcache --working-directory="$HOME/fpl-optimizer" \
  -p OnFailure=fplopt-failure@fplopt-bf-fplcache.service \
  "$HOME/fpl-optimizer/.venv/bin/fplopt" backfill fplcache

journalctl --user -u fplopt-bf-vaastav -f          # follow one
systemctl --user status fplopt-bf-fplcache         # still running?
```

The working directory matters: the CLI loads `.env` from it (Telegram alerts) and resolves the default `raw/` against it. A unit that fails stays loaded, so run `systemctl --user reset-failed fplopt-bf-<name>` before starting it again under the same name.

| Job | Size / time | Writes | Check |
|---|---|---|---|
| `football-data` | 11 small CSVs, ~15 s | `raw/football-data/E0/{1617…2627}/<ts>.csv.gz` | `ls raw/football-data/E0` shows 11 seasons |
| `vaastav` | ~3,000 files at the pinned commit, ~10 min | `raw/vaastav/data/<run>/…csv.gz` + `_manifest.json.gz` | manifest `failed` is empty (below) |
| `fplcache` | ~7,950 snapshots, ~0.9 GB streamed, ~10–20 min | `raw/fplcache/bootstrap-static/<snapshot time>.json.xz` + `raw/fplcache/runs/<run>.json.gz` | run manifest `failed` is empty and `error` is null |

```bash
# vaastav: commit, expected / written counts, failures
zcat "$(ls raw/vaastav/data/*/_manifest.json.gz | tail -1)" | python3 -c \
  'import json,sys; m=json.load(sys.stdin); print(m["commit"], len(m["expected"]), len(m["written"]), m["failed"])'
# fplcache: newest run manifest (commit, written, skipped, failed, error), then the file count
zcat "$(ls raw/fplcache/runs/*.json.gz | tail -1)"; echo
ls raw/fplcache/bootstrap-static | wc -l
du -sh raw/*
```

All three are safe to re-run. football-data and vaastav write a new timestamped copy each time. fplcache skips snapshots it already has (byte-identical) and only adds new ones, so re-running it later tops up the mirror. A snapshot that changed upstream is reported as a failure and never overwritten.

## Rules config

`config/scoring/<season>.json` is exported from the newest archived bootstrap of that season: our own archive first, then the fplcache mirror. Export to a scratch directory, so the server checkout stays clean for `git pull`:

```bash
.venv/bin/fplopt rules export 2026-27 --out /tmp/fplopt-scoring
.venv/bin/fplopt rules export 2025-26 --out /tmp/fplopt-scoring
```

Then, from the repo root on the dev machine, copy them back and commit them:

```bash
mkdir -p config/scoring
scp 'fplopt-server:/tmp/fplopt-scoring/*.json' config/scoring/
```

Each file records the snapshot it came from (`source.path`, `source.snapshot_at`).

## Copy raw/ to the dev machine

For notebooks and Phase 1b, from the repo root on the dev machine (Git Bash):

```bash
ssh fplopt-server 'tar -C fpl-optimizer -cf - raw' | tar -C . -xf -
```

Expect over 1 GB, mostly fplcache. Files that already exist locally are overwritten with identical content, since raw files never change.

Then build the Parquet tables locally with `uv run fplopt build all` (~1.5 min, ~35 MB in `data/`). The server doesn't need `data/` yet.
