# Deploying the archiver (Ubuntu)

Runs as your normal user with systemd **user** timers. Raw data lands in `~/fpl-optimizer/raw/` (gitignored).

Commands run from the dev machine use `fplopt-server` as an SSH host alias for the server; define
your own in `~/.ssh/config` or substitute `user@host`.

## One-time setup

1. Clone and install (the repo is public, so no key is needed):
   ```bash
   git clone https://github.com/samintisar/fpl-optimizer.git ~/fpl-optimizer
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"   # put uv on PATH in this shell
   cd ~/fpl-optimizer && uv sync --locked --all-extras
   ```
   A server cloned earlier over SSH with a read-only deploy key keeps pulling as before.
2. Telegram alerts: create a bot with @BotFather, send it any message, then read your chat id from
   `https://api.telegram.org/bot<token>/getUpdates` (`message.chat.id`).
3. Create `~/fpl-optimizer/.env` (then `chmod 600 .env`):
   ```
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_ADMIN_CHAT_ID=...
   ODDS_API_KEY=...
   ```
   `ODDS_API_KEY` is optional (free tier at the-odds-api.com, 500 credits/month); without it odds are skipped.
   `HEALTHCHECK_PING_URL` is optional too: see [Heartbeat](#heartbeat-optional) below.
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
   systemctl --user enable --now fplopt-daily.timer fplopt-tick.timer fplopt-freshness.timer
   sudo loginctl enable-linger "$USER"   # keep timers running when logged out
   ```
   If a unit fails for any reason, including a broken venv where the CLI can't even start, `fplopt-failure@.service` sends a Telegram message. A job failure the CLI already alerted on will therefore alert twice. That's intentional.

   What the units do besides running the CLI:
   - **Network wait.** `After=network-online.target` does nothing for systemd *user* units, so a `Persistent=true` timer catching up at boot could start before DNS works. Each service's `ExecStartPre` polls `getent hosts fantasy.premierleague.com` every 5 s for up to ~2 minutes, then lets the job start either way: if the network never comes up, the job fails and alerts as usual. A boot-time run can therefore take up to 2 minutes longer to start.
   - **Freshness check** (`fplopt-freshness.timer`, daily at 12:00 UK, `Persistent=true`): `fplopt check freshness` fails, and alerts, if the newest `raw/fpl/bootstrap-static` snapshot is missing or more than 36 hours old (`--max-age-hours` to change). The daily job writes one every night, so this catches jobs or timers that stopped silently, e.g. linger disabled or a timer left disabled after an update. It runs on the same server, so it can't notice the server itself being down: for that, set up the [heartbeat](#heartbeat-optional).
7. Optional: element-summary by hand (issue #13, ~5 minutes). On a fresh server step 4 already did a full run, and the daily job keeps it fresh (see below), so this is only for forcing a new run, e.g. after step 4's run failed, without waiting for the next daily run. Same deadline caveat as step 4:
   ```bash
   .venv/bin/fplopt backfill element-summary
   ```
   Each run writes `raw/fpl/element-summary/<timestamp>/_manifest.json.gz` listing expected, written and failed players.

## Operating

- Timers: `systemctl --user list-timers 'fplopt-*'`
- Logs: `journalctl --user -u fplopt-daily.service -u fplopt-tick.service -u fplopt-freshness.service --since today`
- Archive age by hand: `.venv/bin/fplopt check freshness` (logs the newest bootstrap's age; exit 1 and an alert if over 36 h)
- Update: `cd ~/fpl-optimizer && git pull && uv sync --locked --all-extras` (re-copy units if `deploy/systemd/` changed, then `systemctl --user daemon-reload`). Keep `--all-extras`: a plain `uv sync --locked` uninstalls the model, optimizer and bot packages the [Telegram bot](#telegram-bot-phase-7) needs.
- **Units changed in Phase 1a:** `fplopt-daily.service` now has `TimeoutStartSec=40min` (was 20min). On a server set up before that, re-copy the units and reload when deploying:
  ```bash
  cp deploy/systemd/* ~/.config/systemd/user/ && systemctl --user daemon-reload
  ```
- **Units changed in Phase 1 ops hardening:** `fplopt-daily.service` and `fplopt-tick.service` gained the network-wait `ExecStartPre`, and `fplopt-freshness.service`/`.timer` are new. On a server set up before that, re-copy the units, reload and enable the new timer:
  ```bash
  cp deploy/systemd/* ~/.config/systemd/user/ && systemctl --user daemon-reload
  systemctl --user enable --now fplopt-freshness.timer
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

The pre-deadline tick runs the same post-lockdown step right after it takes a window's FPL snapshot, so a GW finalised after the 02:30 run (say on a Tuesday morning) is archived before a Tuesday-evening deadline. It's idempotent, so usually it does nothing; when a GW was newly finalised it adds event-live and one element-summary run (~5 minutes, inside the tick unit's `TimeoutStartSec=20min`). That run archives another bootstrap in the same, already-closed window, which is harmless. A failure there alerts but isn't retried by later ticks; the next daily run picks it up.

The element-summary run has to fit in the daily unit's `TimeoutStartSec=40min`. If systemd stops the job (timeout or `systemctl stop`), the CLI turns SIGTERM into a normal exit so the run still writes its manifest; the run is incomplete, so the next day redoes it. A run killed outright (SIGKILL, power loss) leaves no manifest and is likewise redone.

## Heartbeat (optional)

Telegram alerts need the server to be up. To also hear about the server being down (or the
timers stopping), give the jobs an external dead-man's switch: after every *successful*
`snapshot daily` or `snapshot tick`, the CLI sends a GET to `HEALTHCHECK_PING_URL`
(10 s timeout; a failed ping is logged as `heartbeat ping failed: <error type>` and never fails
the job). The URL is a secret (anyone holding it can ping), so it is never logged.

With [healthchecks.io](https://healthchecks.io) (free tier is plenty):

1. Sign up, then **Add Check**. Name it e.g. `fplopt archiver`.
2. Schedule: **Period 1 hour, Grace 1 hour**. The 15-minute tick pings on every successful run,
   including the usual runs that have nothing to do, so pings normally arrive about 4 times an
   hour; this alerts after roughly 2 hours of silence. (If only the daily job pinged, you'd use
   period 1 day and grace ~6 hours, and an outage could go unnoticed for most of a day.) Ticks
   that fail don't ping, so a tick failing for 2 hours alerts here as well as on Telegram.
3. Under **Integrations**, keep email and/or add Telegram.
4. Copy the ping URL (`https://hc-ping.com/<uuid>`) into `~/fpl-optimizer/.env`:
   ```
   HEALTHCHECK_PING_URL=https://hc-ping.com/<uuid>
   ```
5. Check it: `.venv/bin/fplopt snapshot tick` should log `heartbeat ping sent`, and the check
   turns green on healthchecks.io. No unit change or restart is needed: each run reads `.env`.

## Telegram bot (Phase 7)

The bot uses the same `TELEGRAM_BOT_TOKEN` as the failure alerts. Its design is in `docs/superpowers/plans/2026-10-11-phase-7-bot.md`. It has two parts:
- **`fplopt-bot.service`**: long polling (no public URL). It answers `/register`, `/plan`, `/chips` (information only, ~40–60 s), `/transfer` (moves already made for the coming GW), `/setft`, `/alerts`, `/help`, plus the admin's `/allow`, `/block` and `/users`.
- **`fplopt-live.timer`** (`fplopt live tick`): every 15 minutes, 5 minutes after the archiver tick. It does three things:
  - rebuilds `data/` and the **plan context** (the v1 xP for the next deadline, shared by every user) whenever something new was archived;
  - sends the full recommendation **24 hours before each deadline**, after taking a fresh FPL snapshot;
  - re-runs after the archiver's **pre-deadline snapshot**, and messages a user only if their plan changed.

State lives in `~/fpl-optimizer/state/` (gitignored; `FPLOPT_STATE_DIR` to move it):
- `bot.sqlite`: users, settings, `/setft` overrides, imported squads and every recommendation (the live log);
- `context/`: one directory per plan context. Each holds xP, pool and prices as Parquet, plus `meta.json` with the snapshot, git sha and deadline.

### Setup

1. LightGBM (the v1 minutes model) needs the OpenMP runtime, a system package. As an admin user:
   ```bash
   sudo apt install libgomp1
   ```
2. Install the extras and build the first context (about 4 minutes: `build all` then v1):
   ```bash
   cd ~/fpl-optimizer && git pull && ~/.local/bin/uv sync --locked --all-extras
   .venv/bin/fplopt live prepare
   ```
3. Install and start the units:
   ```bash
   cp deploy/systemd/* ~/.config/systemd/user/ && systemctl --user daemon-reload
   systemctl --user enable --now fplopt-bot.service fplopt-live.timer
   ```
4. In Telegram, send the bot `/start`. The admin chat (`TELEGRAM_ADMIN_CHAT_ID`) is active straight away. Anyone else is `pending` until the admin answers the request the bot forwards with `/allow <chat id>`. Then `/register <team ID>` and `/plan`.

### Operating

- Logs: `journalctl --user -u fplopt-bot.service -u fplopt-live.service --since today`
- Contexts: `ls state/context`. The newest is the one in use, and the 12 newest are kept.
- Memory: botbox has 3.8 GB and no swap.
  - `build all` peaks at ~1.7 GB, and the live job runs it in a subprocess.
  - v1 peaks at ~2.0 GB.
  - Only the live job runs these. The bot only reads contexts, and a `/plan` (import + solve) takes ~20 s and a few hundred MB.
- After an update (`git pull && uv sync --locked --all-extras`):
  - `systemctl --user restart fplopt-bot.service`;
  - the next live tick rebuilds the context, because a new git sha makes it stale.

## Historical backfills (one-off, Phase 1a)

Run these on the server as transient user units, so they survive SSH disconnects and send the usual failure alert. Lingering is already on (setup step 6).

> Verified on the server (systemd 255) on 2026-10-06: the `OnFailure=` property is accepted on transient units. `systemd-run` doesn't expand `%h`/`%n` on its command line, so these commands use `$HOME` and literal unit names. `fplopt-failure@<instance>.service` reads the instance (`%i`) as the name of the failed unit and passes it to `journalctl -u`.

```bash
cd ~/fpl-optimizer && git pull && ~/.local/bin/uv sync --locked --all-extras

systemd-run --user --unit=fplopt-bf-football-data --working-directory="$HOME/fpl-optimizer" \
  -p OnFailure=fplopt-failure@fplopt-bf-football-data.service \
  "$HOME/fpl-optimizer/.venv/bin/fplopt" backfill football-data
systemd-run --user --unit=fplopt-bf-vaastav --working-directory="$HOME/fpl-optimizer" \
  -p OnFailure=fplopt-failure@fplopt-bf-vaastav.service \
  "$HOME/fpl-optimizer/.venv/bin/fplopt" backfill vaastav
systemd-run --user --unit=fplopt-bf-fplcache --working-directory="$HOME/fpl-optimizer" \
  -p OnFailure=fplopt-failure@fplopt-bf-fplcache.service \
  "$HOME/fpl-optimizer/.venv/bin/fplopt" backfill fplcache
systemd-run --user --unit=fplopt-bf-espn --working-directory="$HOME/fpl-optimizer" \
  -p OnFailure=fplopt-failure@fplopt-bf-espn.service \
  "$HOME/fpl-optimizer/.venv/bin/fplopt" backfill espn   # ~2-3 h; resumable, re-run after a failure

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

## Rust archiver shadow (migration batch 1)

Gate (b) of the Rust migration (`docs/superpowers/plans/2026-10-28-rust-migration.md`), run in
the week before the switch to Rust (the whole migration switches over at once, from the
`rust-migration` branch). The Rust archiver runs beside the live one for a week, writing only
to `raw-shadow/` (gitignored), from its own timers 2 (tick) and 15 (daily) minutes after the live ones. It doesn't ping the heartbeat;
its failures alert as `fplopt-rs ... failed`. It doubles the archiver's API calls for that week,
odds credits included (~80 a month more, within the free 500).

Rust lives in the `fplopt` user's home (`~/.cargo`, installed with rustup, no sudo). The binary
is built in its own checkout of the branch, `~/fpl-optimizer-rs`; the live `~/fpl-optimizer`
stays on `main` (the shadow units run there only for `.env` and `config/`):

From batch 4 on the build also compiles HiGHS (the pinned submodule `third_party/HiGHS`, v1.15.1 like highspy), which needs CMake ≥ 3.15 and a C++ compiler (`cmake --version`, `g++ --version`; without sudo, `uv tool install cmake` puts CMake in `~/.local/bin`):

```bash
git clone -b rust-migration --recurse-submodules --shallow-submodules https://github.com/samintisar/fpl-optimizer.git ~/fpl-optimizer-rs
cd ~/fpl-optimizer-rs && nice -n 15 ~/.cargo/bin/cargo build --release --locked -j 8
cp deploy/systemd-shadow/* ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now fplopt-rs-shadow-tick.timer fplopt-rs-shadow-daily.timer
```

After a `git pull` (and `git submodule update --init`) in `~/fpl-optimizer-rs` during the week, rebuild there (the units run its `target/release/fplopt-rs`). Logs:
`journalctl --user -u 'fplopt-rs-shadow-*' --since today`. After a week, compare (`--since`
skips the first shadow daily, which backfills what `raw/` already had):

```bash
.venv/bin/python dev/shadow_compare.py --since <UTC time after the first shadow daily>
```

**Cutover** (at the switch, once every batch's gates pass): in `fplopt-tick.service`, `fplopt-daily.service` and
`fplopt-freshness.service`, `ExecStart` runs `%h/fpl-optimizer/target/release/fplopt-rs` instead
of `.venv/bin/fplopt` (same arguments); re-copy, `daemon-reload`, then disable and remove the
shadow timers and `raw-shadow/`. Rollback: revert the three units (the Python CLI stays
installed until batch 8).

**Bot cutover** (batch 5, at the same switch, between two gameweeks, never in the 24 h before a deadline): in `fplopt-bot.service`, `ExecStart` runs `%h/fpl-optimizer/target/release/fplopt-rs bot run` instead of `.venv/bin/fplopt bot run` (same environment: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ADMIN_CHAT_ID`, `FPLOPT_STATE_DIR`); re-copy, `daemon-reload`, `restart`. The Rust bot uses the same `state/bot.sqlite` and reads the plan contexts the live job writes (Rust's after the live cutover below). Before it, re-run the gate on the server against a fresh backup of `bot.sqlite` (`python3 -c "import sqlite3; s = sqlite3.connect('file:state/bot.sqlite?mode=ro', uri=True); d = sqlite3.connect('../fpl-optimizer-rs/gate/bot-copy.sqlite'); s.backup(d)"`) and the newest context, never the live file: `dev/bot_parity.py --server <a directory with both>`. Rollback: revert the unit.

**Live cutover** (batch 6, with the bot): in `fplopt-live.service` (the 15-minute alert job), `ExecStart` runs `%h/fpl-optimizer/target/release/fplopt-rs live tick` instead of `.venv/bin/fplopt live tick` (same environment; `live prepare` likewise when run by hand). The Rust job rebuilds `data/` with `fplopt-rs build all` in a subprocess and writes plan contexts from its own v1 model to the same `state/context/`; its first run after the switch sees a new commit and builds a fresh context (~a few minutes), so switch more than an hour before an alert window. The build needs LightGBM's sources (`git submodule update --init --recursive`: `third_party/LightGBM` and its `external_libs`) and, besides CMake and a C++ compiler, OpenMP (`libgomp`, part of gcc). Before it, run `dev/models_parity.py --models v1 --seasons 2025,2026 --plans` on the server (its `data/`, read only) and `dev/bot_parity.py --server <copy> --deadlines <the next GW> --rust-contexts`. Rollback: revert the unit; the Python job reads the Rust-built contexts (same layout) or rebuilds its own on the next commit change.

## Copy raw/ to the dev machine

For notebooks and Phase 1b, from the repo root on the dev machine (Git Bash):

```bash
ssh fplopt-server 'tar -C fpl-optimizer -cf - raw' | tar -C . -xf -
```

Expect over 1 GB, mostly fplcache. Files that already exist locally are overwritten with identical content, since raw files never change.

Then build the Parquet tables locally with `uv run fplopt build all` (~1.5 min, ~35 MB in `data/`). The server doesn't need `data/` yet.
