"""Command-line entry point.

`fplopt snapshot daily|tick` (archiver timers), `fplopt backfill element-summary|football-data|
vaastav|fplcache` (one-off backfills into raw/; football-data takes `--from-season YEAR`),
`fplopt rules export SEASON [--out DIR]` (config/scoring/<season>.json from an archived
bootstrap), `fplopt build TABLE|all` (raw/ -> data/<table>.parquet; no network),
`fplopt check freshness [--max-age-hours H]` (fails if the newest bootstrap snapshot is
missing or older than H hours, default 36: a dead-man's switch for the timers),
`fplopt check leakage [--deadlines N] [--seed S]` (the corrupt-the-future check of every
registered feature on data/ at a fixed list of edge deadlines plus N sampled ones, default 12,
outside the holdout). Every job
gets a `Context`; failures are logged and alerted to Telegram, and the exit code is 1.
After a successful `snapshot daily|tick`, HEALTHCHECK_PING_URL (if set) gets a best-effort
GET, for an external dead-man's switch that also notices the server being down.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

from fplopt.adapters.football_data import FootballDataClient
from fplopt.adapters.fpl import FplClient
from fplopt.adapters.fplcache import FplcacheClient
from fplopt.adapters.http import make_client
from fplopt.adapters.odds import OddsClient
from fplopt.adapters.vaastav import VaastavClient
from fplopt.alerts import send_admin_alert
from fplopt.heartbeat import send_heartbeat
from fplopt.ingest import health, history, jobs
from fplopt.ingest.raw_store import RawStore
from fplopt.redact import redact
from fplopt.seasons import parse_season_label
from fplopt.settings import Settings

log = logging.getLogger("fplopt")


@dataclass
class Context:
    """What a job needs: the raw store, a shared HTTP client, settings and parsed arguments."""

    store: RawStore
    http: httpx.Client
    settings: Settings
    args: argparse.Namespace

    @property
    def fpl(self) -> FplClient:
        return FplClient(self.http)

    @property
    def odds(self) -> OddsClient | None:
        key = self.settings.odds_api_key
        return OddsClient(self.http, key) if key else None


Job = Callable[[Context], object]

# Jobs whose success pings HEALTHCHECK_PING_URL: the scheduled archiver runs.
HEARTBEAT_JOBS = frozenset({"snapshot daily", "snapshot tick"})


# The build layer pulls in pandas and pandera (~2 s to import), which the archiver jobs that
# run every 15 minutes don't need, so it is imported only by the jobs that use it.
def _rules_export(c: Context) -> object:
    from fplopt.build import rules

    # The season label is parsed inside the job, so a bad label is alerted like any failure.
    return rules.export_rules(c.store, parse_season_label(c.args.season), Path(c.args.out))


def _build(c: Context) -> object:
    from fplopt.build import build
    from fplopt.build.common import BuildContext

    # Unknown table names fail inside the job, so they are logged and alerted (exit 1).
    return build([c.args.target], BuildContext(c.store, c.settings.data_dir))


def _check_leakage(c: Context) -> object:
    from fplopt.features.leakcheck import run_leakage_check

    return run_leakage_check(c.settings.data_dir, c.args.deadlines, c.args.seed)


JOBS: dict[str, Job] = {
    "snapshot daily": lambda c: jobs.run_daily(
        c.store, c.fpl, c.odds, football_data=FootballDataClient(c.http)
    ),
    "snapshot tick": lambda c: jobs.run_tick(c.store, c.fpl, c.odds),
    "backfill element-summary": lambda c: jobs.backfill_element_summaries(c.store, c.fpl),
    "backfill football-data": lambda c: history.backfill_football_data(
        c.store, FootballDataClient(c.http), first_season=c.args.from_season
    ),
    "backfill vaastav": lambda c: history.backfill_vaastav(c.store, VaastavClient(c.http)),
    "backfill fplcache": lambda c: history.backfill_fplcache(c.store, FplcacheClient(c.http)),
    "rules export": _rules_export,
    "build": _build,
    "check freshness": lambda c: health.check_freshness(
        c.store, jobs.utc_now(), timedelta(hours=c.args.max_age_hours)
    ),
    "check leakage": _check_leakage,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fplopt")
    groups = parser.add_subparsers(dest="group", required=True)
    snapshot = groups.add_parser("snapshot", help="archive API snapshots into raw/")
    snapshot.add_argument("command", choices=["daily", "tick"])
    backfill = groups.add_parser("backfill", help="one-off backfills into raw/")
    backfill.add_argument(
        "command", choices=["element-summary", "football-data", "vaastav", "fplcache"]
    )
    backfill.add_argument(
        "--from-season",
        type=int,
        default=history.FIRST_SEASON,
        metavar="YEAR",
        help=f"football-data only: first season's start year (default {history.FIRST_SEASON})",
    )
    rules_group = groups.add_parser("rules", help="per-season rules config")
    rules_group.add_argument("command", choices=["export"])
    rules_group.add_argument("season", help="season label, e.g. 2026-27")
    rules_group.add_argument(
        "--out", default="config/scoring", help="output directory (default: config/scoring)"
    )
    build_group = groups.add_parser("build", help="build data/<table>.parquet from raw/")
    build_group.add_argument("target", help="table name, or 'all'")
    check = groups.add_parser("check", help="archive health and leakage checks")
    check.add_argument("command", choices=["freshness", "leakage"])
    check.add_argument(
        "--max-age-hours",
        type=float,
        default=health.DEFAULT_MAX_AGE.total_seconds() / 3600,
        metavar="H",
        help="fail if the newest bootstrap snapshot is older than this (default 36)",
    )
    check.add_argument(
        "--deadlines",
        type=int,
        default=12,
        metavar="N",
        help="leakage only: random GW deadlines to check besides the fixed edges (default 12)",
    )
    check.add_argument(
        "--seed", type=int, default=0, metavar="S", help="leakage only: corruption seed"
    )
    return parser


def job_name(args: argparse.Namespace) -> str:
    """The JOBS key for parsed arguments: '<group> <command>', or just '<group>' for groups
    without a command (build)."""
    command = getattr(args, "command", None)
    return f"{args.group} {command}" if command else args.group


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs full request URLs at INFO, which would expose API keys and bot tokens.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _exit_on_sigterm(signum: int, frame: object) -> None:
    raise SystemExit(143)  # 128 + SIGTERM, the status a shell reports for a TERM-killed process


@contextmanager
def _sigterm_raises_system_exit() -> Iterator[None]:
    """Turn SIGTERM (systemd stop or TimeoutStartSec) into SystemExit while a job runs, so its
    `finally` blocks still write run manifests; the default action kills the process without
    unwinding. Restores the previous handler afterwards. A no-op where there is no SIGTERM
    or off the main thread (signal handlers can only be set there)."""
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(sigterm, _exit_on_sigterm)
    try:
        yield
    finally:
        signal.signal(sigterm, previous)


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    name = job_name(args)
    if settings is None:
        load_dotenv(Path.cwd() / ".env")
        settings = Settings.from_env()
    configure_logging()
    if not (settings.telegram_bot_token and settings.telegram_admin_chat_id):
        log.warning(
            "TELEGRAM_BOT_TOKEN/TELEGRAM_ADMIN_CHAT_ID not set: failures will not be alerted"
        )
    log.info("job %r starting (raw dir %s)", name, settings.raw_dir)
    try:
        store = RawStore(settings.raw_dir)
        with make_client() as http, _sigterm_raises_system_exit():
            JOBS[name](Context(store=store, http=http, settings=settings, args=args))
    except Exception as exc:
        log.exception("job %r failed", name)
        summary = str(exc).splitlines()[0] if str(exc) else ""
        send_admin_alert(
            redact(
                f"fplopt {name} failed on {socket.gethostname()}: {type(exc).__name__}: {summary}"
            ),
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_admin_chat_id,
        )
        return 1
    log.info("job %r done", name)
    if name in HEARTBEAT_JOBS and settings.healthcheck_ping_url:
        send_heartbeat(settings.healthcheck_ping_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
