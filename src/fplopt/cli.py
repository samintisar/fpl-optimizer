"""Command-line entry point: `fplopt snapshot daily|tick`, `fplopt backfill element-summary`."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from collections.abc import Callable, Sequence

from dotenv import find_dotenv, load_dotenv

from fplopt.adapters.fpl import FplClient
from fplopt.adapters.http import make_client
from fplopt.adapters.odds import OddsClient
from fplopt.alerts import send_admin_alert
from fplopt.ingest import jobs
from fplopt.ingest.raw_store import RawStore
from fplopt.settings import Settings

log = logging.getLogger("fplopt")

Job = Callable[[RawStore, FplClient, OddsClient | None], object]

JOBS: dict[str, Job] = {
    "snapshot daily": lambda store, fpl, odds: jobs.run_daily(store, fpl, odds),
    "snapshot tick": lambda store, fpl, odds: jobs.run_tick(store, fpl, odds),
    "backfill element-summary": lambda store, fpl, odds: jobs.backfill_element_summaries(
        store, fpl
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fplopt")
    groups = parser.add_subparsers(dest="group", required=True)
    snapshot = groups.add_parser("snapshot", help="archive API snapshots into raw/")
    snapshot.add_argument("command", choices=["daily", "tick"])
    backfill = groups.add_parser("backfill", help="one-off backfills into raw/")
    backfill.add_argument("command", choices=["element-summary"])
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs full request URLs at INFO, which would expose API keys and bot tokens.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    job_name = f"{args.group} {args.command}"
    if settings is None:
        load_dotenv(find_dotenv(usecwd=True))
        settings = Settings.from_env()
    configure_logging()
    store = RawStore(settings.raw_dir)
    with make_client() as http:
        fpl = FplClient(http)
        odds = OddsClient(http, settings.odds_api_key) if settings.odds_api_key else None
        try:
            JOBS[job_name](store, fpl, odds)
        except Exception as exc:
            log.exception("job %r failed", job_name)
            send_admin_alert(
                f"fplopt {job_name} failed on {socket.gethostname()}: {type(exc).__name__}: {exc}",
                token=settings.telegram_bot_token,
                chat_id=settings.telegram_admin_chat_id,
            )
            return 1
    log.info("job %r done", job_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
