"""Archiver jobs: fetch from adapters and append to the raw store."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx

from fplopt.ingest.raw_store import TS_PATTERN, RawStore, gzip_bytes, parse_ts
from fplopt.ingest.schedule import deadlines_from_bootstrap, pre_deadline_due
from fplopt.seasons import bootstrap_season, football_data_code, season_label, season_start_year

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]

MAX_CONSECUTIVE_FAILURES = 10

FOOTBALL_DATA_GRACE_MONTHS = (7, 8)  # new-season CSV may not exist yet in July/August

# Post-lockdown only trusts a bootstrap this recent (normally the daily run's own FPL step).
POST_LOCKDOWN_MAX_BOOTSTRAP_AGE = timedelta(hours=6)


def utc_now() -> datetime:
    return datetime.now(UTC)


class FplSource(Protocol):
    def bootstrap_static(self) -> bytes: ...

    def fixtures(self) -> bytes: ...

    def element_summary(self, element_id: int) -> bytes: ...

    def event_live(self, gw: int) -> bytes: ...


class OddsSource(Protocol):
    def epl_odds(self) -> bytes: ...


class FootballDataSource(Protocol):
    def epl_season(self, start_year: int) -> bytes: ...


def snapshot_fpl(store: RawStore, fpl: FplSource, now: Clock = utc_now) -> list[Path]:
    """Fixtures first, bootstrap last: the tick treats a bootstrap snapshot inside a
    pre-deadline window as done, so a failure part-way must leave the window open."""
    fixtures_content = fpl.fixtures()
    fixtures = store.write("fpl", "fixtures", fixtures_content, now())
    bootstrap_content = fpl.bootstrap_static()
    bootstrap = store.write("fpl", "bootstrap-static", bootstrap_content, now())
    return [fixtures, bootstrap]


def snapshot_odds(store: RawStore, odds: OddsSource | None, now: Clock = utc_now) -> Path | None:
    if odds is None:
        log.warning("ODDS_API_KEY not set; skipping odds snapshot")
        return None
    content = odds.epl_odds()
    return store.write("odds", "soccer_epl", content, now())


class SnapshotError(RuntimeError):
    """One or more sources failed; the others were still archived."""


def run_independently(steps: list[tuple[str, Callable[[], object]]]) -> None:
    """Run every step even if earlier ones fail; then raise one SnapshotError naming each
    failed step (single line)."""
    errors: list[str] = []
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            log.exception("%s snapshot failed", name)
            # First line only: the combined message must stay one line so no source's
            # error is cut off when the CLI alert keeps just the first line.
            summary = str(exc).splitlines()[0] if str(exc) else ""
            errors.append(f"{name}: {type(exc).__name__}: {summary}")
    if errors:
        raise SnapshotError("; ".join(errors))


def snapshot_football_data(
    store: RawStore, fd: FootballDataSource, start_year: int, now: Clock = utc_now
) -> Path:
    """Archive one season's football-data CSV under football-data/E0/<code>, e.g. E0/1617."""
    content = fd.epl_season(start_year)
    return store.write_bytes(
        "football-data",
        f"E0/{football_data_code(start_year)}",
        gzip_bytes(content),
        now(),
        suffix=".csv.gz",
    )


def snapshot_football_data_current(
    store: RawStore, fd: FootballDataSource, now: Clock = utc_now
) -> Path | None:
    """Daily: the in-progress season's CSV (results + odds are appended twice a week).
    A 404 is tolerated in July/August (file not published yet); otherwise it is an error."""
    current = now()
    try:
        return snapshot_football_data(store, fd, season_start_year(current), now)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404 and current.month in FOOTBALL_DATA_GRACE_MONTHS:
            log.warning("football-data CSV for the new season not published yet")
            return None
        raise


def run_daily(
    store: RawStore,
    fpl: FplSource,
    odds: OddsSource | None,
    now: Clock = utc_now,
    *,
    football_data: FootballDataSource | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Daily archive: FPL, odds, the current football-data CSV (if a source is given), then
    the post-lockdown step, last so it reads the bootstrap the FPL step just archived."""
    steps: list[tuple[str, Callable[[], object]]] = [
        ("fpl", lambda: snapshot_fpl(store, fpl, now)),
        ("odds", lambda: snapshot_odds(store, odds, now)),
    ]
    if football_data is not None:
        steps.append(
            ("football-data", lambda: snapshot_football_data_current(store, football_data, now))
        )
    steps.append(("post-lockdown", lambda: snapshot_post_lockdown(store, fpl, now, sleep=sleep)))
    run_independently(steps)


def run_tick(
    store: RawStore, fpl: FplSource, odds: OddsSource | None, now: Clock = utc_now
) -> bool:
    """Run every ~15 min. Inside each pre-deadline window, snapshot FPL and odds once each;
    a source that fails is retried on the next tick. Returns True if anything ran."""
    latest = store.latest("fpl", "bootstrap-static")
    if latest is None:
        # FPL and odds only: the post-lockdown step (possibly a full element-summary run)
        # belongs to the daily job, not a 15-minute tick.
        log.info("no bootstrap snapshot yet; taking one")
        run_independently(
            [
                ("fpl", lambda: snapshot_fpl(store, fpl, now)),
                ("odds", lambda: snapshot_odds(store, odds, now)),
            ]
        )
        return True
    deadlines = deadlines_from_bootstrap(store.read_json(latest))
    current = now()
    fpl_deadline = pre_deadline_due(current, deadlines, store.times("fpl", "bootstrap-static"))
    odds_due = (
        odds is not None
        and pre_deadline_due(current, deadlines, store.times("odds", "soccer_epl")) is not None
    )
    steps: list[tuple[str, Callable[[], object]]] = []
    if fpl_deadline is not None:
        log.info("pre-deadline FPL snapshot for deadline %s", fpl_deadline.isoformat())
        steps.append(("fpl", lambda: snapshot_fpl(store, fpl, now)))
    if odds_due:
        log.info("pre-deadline odds snapshot")
        steps.append(("odds", lambda: snapshot_odds(store, odds, now)))
    run_independently(steps)
    return fpl_deadline is not None or odds_due


def backfill_element_summaries(
    store: RawStore,
    fpl: FplSource,
    now: Clock = utc_now,
    pause_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> int:
    """Archive element-summary for every current player under one run timestamp.

    Writes `<run>/_manifest.json.gz` (expected, written and failed ids, plus the `season` and
    `through_event` of the bootstrap it fetched) even when aborted, so incomplete runs are
    identifiable. The bootstrap it archives counts as a snapshot for the pre-deadline tick,
    so avoid running this by hand inside a pre-deadline window (the daily job runs it at
    02:30 UK, and its own FPL step snapshots fixtures and bootstrap anyway).

    Element summaries in a run share `run_at` but are fetched later; treat the manifest's
    `finished_at` as the run's `available_at`.
    """
    bootstrap = fpl.bootstrap_static()
    run_at = now()
    try:
        store.write("fpl", "bootstrap-static", bootstrap, run_at)
    except FileExistsError:
        # The daily FPL step archived a bootstrap in this same second; that one stands.
        log.info("bootstrap for %s already archived; keeping it", run_at.isoformat())
    payload = json.loads(bootstrap)
    ids = [element["id"] for element in payload["elements"]]
    season = _season_or_none(payload)
    through_event = max(checked_events(payload), default=0)
    written: list[int] = []
    failed: list[int] = []
    consecutive = 0
    try:
        for element_id in ids:
            try:
                content = fpl.element_summary(element_id)
                store.write("fpl", "element-summary", content, run_at, name=str(element_id))
                written.append(element_id)
                consecutive = 0
            except Exception as exc:
                log.exception("element-summary %s failed", element_id)
                failed.append(element_id)
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    raise RuntimeError(
                        f"aborting backfill after {consecutive} consecutive failures "
                        f"(last element {element_id}: {type(exc).__name__})"
                    ) from exc
            sleep(pause_s)
    finally:
        manifest = {
            "run_at": run_at.isoformat(),
            "finished_at": now().isoformat(),
            "season": season,
            "through_event": through_event,
            "expected": ids,
            "written": written,
            "failed": failed,
        }
        try:
            store.write(
                "fpl", "element-summary", json.dumps(manifest).encode(), run_at, name="_manifest"
            )
        except Exception:
            log.exception("could not write backfill manifest")
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(ids)} element summaries failed: {failed}")
    log.info("archived %d element summaries", len(ids))
    return len(ids)


# --- post-lockdown ingest (issue #17) ------------------------------------------------------


def checked_events(bootstrap: Mapping[str, Any]) -> list[int]:
    """GWs whose data FPL has finalised (bonus confirmed): `data_checked` is true. Sorted."""
    events = bootstrap.get("events") or []
    return sorted(event["id"] for event in events if event.get("data_checked"))


def _season_or_none(bootstrap: Mapping[str, Any]) -> int | None:
    """Season of a bootstrap, or None if it has no usable events (e.g. during the reset)."""
    try:
        return bootstrap_season(bootstrap)
    except (KeyError, TypeError, ValueError):
        return None


def event_live_endpoint(season: int, gw: int) -> str:
    """'event-live/2026-27/5': season in the path because GW numbers repeat every season."""
    return f"event-live/{season_label(season)}/{gw}"


def snapshot_event_live(
    store: RawStore,
    fpl: FplSource,
    bootstrap: Mapping[str, Any],
    now: Clock = utc_now,
    *,
    sleep: Callable[[float], None] = time.sleep,
    pause_s: float = 0.25,
) -> list[int]:
    """Archive event/{gw}/live/ once per finalised GW, under fpl/event-live/<season>/<gw>.
    GWs already archived are skipped; every pending GW is attempted even if another fails
    (SnapshotError naming each failed GW). Returns the GWs written."""
    season = bootstrap_season(bootstrap)
    pending = [
        gw
        for gw in checked_events(bootstrap)
        if store.latest("fpl", event_live_endpoint(season, gw)) is None
    ]
    written: list[int] = []

    def step(gw: int) -> Callable[[], None]:
        def run() -> None:
            if gw != pending[0]:
                sleep(pause_s)
            content = fpl.event_live(gw)
            store.write("fpl", event_live_endpoint(season, gw), content, now())
            written.append(gw)

        return run

    run_independently([(f"gw {gw}", step(gw)) for gw in pending])
    if written:
        log.info("archived event-live for %s GWs %s", season_label(season), written)
    return written


def _complete_run_key(manifest: Any) -> tuple[int, int] | None:
    """(season, through_event) of a manifest whose run wrote every expected element; None for
    incomplete runs and for older manifests without these keys."""
    if not isinstance(manifest, dict):
        return None
    season, through_event = manifest.get("season"), manifest.get("through_event")
    expected, written = manifest.get("expected"), manifest.get("written")
    if not all(type(value) is int for value in (season, through_event)):
        return None
    if not (isinstance(expected, list) and isinstance(written, list)) or manifest.get("failed"):
        return None
    if not set(expected) <= set(written):
        return None
    return season, through_event


def latest_complete_element_summary(store: RawStore) -> tuple[int, int]:
    """(season, through_event) of the newest element-summary run whose manifest shows every
    expected element written; (0, 0) if none. "Newest" compares (season, through_event), not
    run time. Non-timestamp directories, runs without a manifest (killed mid-run) and older
    manifests without these keys count as none."""
    root = store.root / "fpl" / "element-summary"
    best = (0, 0)
    if not root.is_dir():
        return best
    for run_dir in root.iterdir():
        if not (run_dir.is_dir() and TS_PATTERN.match(run_dir.name)):
            continue
        manifest_path = store.path_for(
            "fpl", "element-summary", parse_ts(run_dir.name), name="_manifest"
        )
        if not manifest_path.is_file():
            continue
        try:
            manifest = store.read_json(manifest_path)
        except Exception:
            log.warning("unreadable element-summary manifest %s", manifest_path, exc_info=True)
            continue
        key = _complete_run_key(manifest)
        if key is not None and key > best:
            best = key
    return best


def element_summaries_due(store: RawStore, bootstrap: Mapping[str, Any]) -> bool:
    """True if a GW has been finalised since the newest complete element-summary run."""
    checked = checked_events(bootstrap)
    if not checked:
        return False
    return (bootstrap_season(bootstrap), checked[-1]) > latest_complete_element_summary(store)


def snapshot_post_lockdown(
    store: RawStore,
    fpl: FplSource,
    now: Clock = utc_now,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """After GW lockdown: event-live for new finalised GWs, and one fresh element-summary run
    (all players' per-GW history) if a GW was finalised since the last complete run. Reads the
    latest archived bootstrap; the two steps fail independently. A no-op when nothing new is
    finalised.

    Skipped (warning, not an error) unless that bootstrap is under
    POST_LOCKDOWN_MAX_BOOTSTRAP_AGE old: event/{gw}/live has no season in its URL, so a stale
    bootstrap (say today's FPL step failed during the July reset) would file new-season data
    under the old season."""
    entries = store.entries("fpl", "bootstrap-static")
    if not entries:
        log.warning("no archived bootstrap yet; skipping post-lockdown ingest")
        return
    fetched_at, latest = entries[-1]
    if now() - fetched_at > POST_LOCKDOWN_MAX_BOOTSTRAP_AGE:
        log.warning(
            "latest bootstrap (%s) is stale; skipping post-lockdown ingest",
            fetched_at.isoformat(),
        )
        return
    bootstrap = store.read_json(latest)
    if not checked_events(bootstrap):
        log.info("no finalised GWs yet; nothing to do after lockdown")
        return

    def refresh_element_summaries() -> None:
        if element_summaries_due(store, bootstrap):
            log.info("a GW was finalised since the last complete element-summary run")
            backfill_element_summaries(store, fpl, now, sleep=sleep)

    run_independently(
        [
            ("event-live", lambda: snapshot_event_live(store, fpl, bootstrap, now, sleep=sleep)),
            ("element-summary", refresh_element_summaries),
        ]
    )
