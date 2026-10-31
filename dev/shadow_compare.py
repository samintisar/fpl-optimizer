"""Rust migration batch 1, parity gate (b): the shadow archiver (`fplopt-rs`, units in
deploy/systemd-shadow/, writing raw-shadow/) against the live one (raw/) over the shadow
period. Dev only: never imported by `fplopt`; reads both trees, writes nothing but the
experiment-log row.

Every live snapshot from the shadow's first file on must have a shadow twin and vice versa:
single-file snapshots (bootstrap, fixtures, odds, football-data, ESPN scoreboards) pair up
per endpoint within `--tolerance` minutes (the shadow timers run 2 and 15 minutes after the
live ones); write-once files (event-live, ESPN summaries) and element-summary runs (their
file names) are compared as sets. Every shadow file must decompress and parse (JSON, or a
football-data CSV). Snapshots after the newest shadow file minus the tolerance are not
compared yet.

    uv run python dev/shadow_compare.py [--live raw] [--shadow raw-shadow] [--tolerance 20]
        [--since 2026-10-31T00:00]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shlex
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fplopt.backtest.evaluate import family_variants, log_experiment  # noqa: E402

TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{6}Z")
# Sources the archiver jobs write (not the one-off backfills: vaastav, fplcache).
SOURCES = ("fpl", "odds", "football-data", "espn")
WRITE_ONCE = (re.compile(r"^fpl/event-live/"), re.compile(r"^espn/[^/]+/summary/"))
FAMILY = "rust parity"


def parse_ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H%M%SZ").replace(tzinfo=UTC)


def scan(root: Path) -> tuple[dict[str, list[datetime]], dict[str, dict[datetime, set[str]]]]:
    """Single-file snapshots per endpoint, and grouped runs (endpoint -> run time -> names)."""
    singles: dict[str, list[datetime]] = defaultdict(list)
    groups: dict[str, dict[datetime, set[str]]] = defaultdict(lambda: defaultdict(set))
    for source in SOURCES:
        for path in (root / source).rglob("*") if (root / source).is_dir() else []:
            if not path.is_file() or path.name.startswith("."):
                continue
            rel = path.relative_to(root).as_posix()
            parts = rel.split("/")
            stamps = [i for i, part in enumerate(parts) if TS.fullmatch(part.split(".")[0])]
            if not stamps:
                continue
            i = stamps[-1]
            endpoint = "/".join(parts[:i])
            at = parse_ts(parts[i].split(".")[0])
            if i == len(parts) - 1:
                singles[endpoint].append(at)
            else:
                groups[endpoint][at].add("/".join(parts[i + 1 :]))
    return singles, groups


def valid(path: Path) -> str | None:
    """None if the file decompresses and parses, else why not."""
    try:
        data = gzip.decompress(path.read_bytes())
        if path.name.endswith(".csv.gz"):
            head = data[:64].decode("utf-8-sig", errors="replace").lstrip()
            if not head.startswith("Div,"):
                return "not a football-data CSV"
        else:
            json.loads(data)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return f"{type(exc).__name__}: {exc}"
    return None


def pair(live: list[datetime], shadow: list[datetime], tolerance: timedelta) -> tuple[list, list]:
    """Greedy in time order: each live snapshot takes the first unused shadow one within
    the tolerance after or before it."""
    unused = sorted(shadow)
    missing = []
    for at in sorted(live):
        match = next((s for s in unused if abs(s - at) <= tolerance), None)
        if match is None:
            missing.append(at)
        else:
            unused.remove(match)
    return missing, unused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", type=Path, default=REPO / "raw")
    parser.add_argument("--shadow", type=Path, default=REPO / "raw-shadow")
    parser.add_argument("--tolerance", type=float, default=20.0, help="minutes")
    parser.add_argument(
        "--since",
        type=lambda t: datetime.fromisoformat(t).replace(tzinfo=UTC),
        help="UTC start of the comparison (default: the first shadow file), e.g. to skip the "
        "shadow's first daily run, which backfills what the live archive already had",
    )
    parser.add_argument("--experiments", type=Path, default=REPO / "results" / "experiments.csv")
    args = parser.parse_args()
    tol = timedelta(minutes=args.tolerance)

    live_singles, live_groups = scan(args.live)
    shadow_singles, shadow_groups = scan(args.shadow)
    times = [t for ts in shadow_singles.values() for t in ts]
    times += [t for runs in shadow_groups.values() for t in runs]
    if not times:
        parser.error(f"no shadow snapshots in {args.shadow}")
    start, end = args.since or min(times) - tol, max(times) - tol

    def window(ts: list[datetime]) -> list[datetime]:
        return [t for t in ts if start <= t <= end]

    problems: list[str] = []
    endpoints = sorted(set(live_singles) | set(shadow_singles))
    compared = 0
    for endpoint in endpoints:
        live = window(live_singles.get(endpoint, []))
        shadow = window(shadow_singles.get(endpoint, []))
        if any(p.match(endpoint + "/") for p in WRITE_ONCE):
            # written once: present on both sides (whenever it was fetched in the window)
            if bool(live_singles.get(endpoint)) != bool(shadow_singles.get(endpoint)) and (
                live or shadow
            ):
                side = "shadow" if live else "live"
                problems.append(f"{endpoint}: missing in {side}")
            compared += 1
            continue
        missing, extra = pair(live, shadow, tol)
        compared += len(live)
        problems += [f"{endpoint}: live {t:%Y-%m-%d %H:%M} has no shadow twin" for t in missing]
        problems += [f"{endpoint}: shadow {t:%Y-%m-%d %H:%M} has no live twin" for t in extra]

    for endpoint in sorted(set(live_groups) | set(shadow_groups)):
        live = {t: n for t, n in live_groups.get(endpoint, {}).items() if start <= t <= end}
        shadow = {t: n for t, n in shadow_groups.get(endpoint, {}).items() if start <= t <= end}
        missing, extra = pair(list(live), list(shadow), timedelta(hours=1))
        problems += [f"{endpoint}: live run {t:%Y-%m-%d %H:%M} has no shadow twin" for t in missing]
        problems += [f"{endpoint}: shadow run {t:%Y-%m-%d %H:%M} has no live twin" for t in extra]
        for at, names in live.items():
            twin = next((s for s in shadow if abs(s - at) <= timedelta(hours=1)), None)
            if twin is not None and shadow[twin] != names:
                diff = sorted(names ^ shadow[twin])[:5]
                problems.append(f"{endpoint}: run {at:%Y-%m-%d %H:%M} file sets differ: {diff}")
            compared += 1

    invalid = []
    n_shadow = 0
    for source in SOURCES:
        root = args.shadow / source
        for path in root.rglob("*.gz") if root.is_dir() else []:
            n_shadow += 1
            why = valid(path)
            if why is not None:
                invalid.append(f"{path.relative_to(args.shadow).as_posix()}: {why}")
    problems += [f"invalid shadow file {line}" for line in invalid]

    passed = not problems
    print(f"shadow period {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")
    print(f"{compared} snapshots/runs compared, {n_shadow} shadow files parsed")
    for problem in problems[:50]:
        print(f"  {problem}")
    if len(problems) > 50:
        print(f"  ... and {len(problems) - 50} more")
    print("PASS" if passed else f"FAIL: {len(problems)} problems")
    log_experiment(
        args.experiments,
        shlex.join(["python", "dev/shadow_compare.py", *sys.argv[1:]]),
        {"gate": "batch 1 (b) archiver shadow", "tolerance_min": args.tolerance},
        {
            "passed": passed,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "compared": compared,
            "shadow_files": n_shadow,
            "problems": len(problems),
        },
        family_variants(args.experiments, FAMILY) + 1,
        FAMILY,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
