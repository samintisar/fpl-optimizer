# Deploying the archiver (Ubuntu)

Runs as your normal user with systemd **user** timers. Raw data lands in `~/fpl-optimizer/raw/` (gitignored).

## One-time setup

1. Clone and install:
   ```bash
   git clone https://github.com/samintisar/fpl-optimizer.git ~/fpl-optimizer
   curl -LsSf https://astral.sh/uv/install.sh | sh
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
4. Smoke test:
   ```bash
   .venv/bin/fplopt snapshot daily && ls raw/fpl/bootstrap-static
   ```
5. Alert test (should fail and send a Telegram message):
   ```bash
   FPLOPT_RAW_DIR=/proc/fplopt-alert-test .venv/bin/fplopt snapshot daily; echo "exit=$?"
   ```
6. Install and start the timers:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/systemd/* ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now fplopt-daily.timer fplopt-tick.timer
   sudo loginctl enable-linger "$USER"   # keep timers running when logged out
   ```
7. One-off backfill of this season's per-GW stats (issue #13, ~5 minutes). Run it **outside** the 2 hours before a deadline (its bootstrap snapshot counts as that window's snapshot):
   ```bash
   .venv/bin/fplopt backfill element-summary
   ```
   Each run writes `raw/fpl/element-summary/<timestamp>/_manifest.json.gz` listing expected, written and failed players.

## Operating

- Timers: `systemctl --user list-timers 'fplopt-*'`
- Logs: `journalctl --user -u fplopt-daily.service -u fplopt-tick.service --since today`
- Update: `cd ~/fpl-optimizer && git pull && uv sync --locked` (re-copy units if `deploy/systemd/` changed, then `systemctl --user daemon-reload`)
- Odds credits: each odds snapshot logs `credits remaining=…`; a warning is logged below 50 (free tier: 500/month; ~80 used).
