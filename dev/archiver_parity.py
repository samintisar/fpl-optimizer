"""Rust migration batch 1, parity gate (a): the Python and Rust archivers, fed the same
recorded HTTP responses, must write the same files (docs/superpowers/plans/
2026-10-28-rust-migration.md, Batch 1). Dev only: never imported by `fplopt`.

A local replay server answers every source's URLs from real payloads in the local `raw/`
(FPL bootstrap/fixtures/element-summary/event-live, odds, football-data CSVs, vaastav CSVs,
fplcache snapshots) plus synthetic ESPN scoreboards and summaries. Each scenario starts
both implementations from the same seed store: the Python jobs are called in process with
clients pointed at the replay server (as `fplopt.cli.JOBS` builds them), the Rust binary
runs as a subprocess with `FPLOPT_<SOURCE>_URL` overrides. The trees are then compared:
paths with timestamps masked, contents decompressed (`.xz` compared verbatim), and in run
manifests the `run_at`/`finished_at` values and the text of failure reasons masked (the
exception type is kept). Exit status must agree too. "-faults" scenarios add 404s, 5xx,
and invalid payloads.

    cargo build --release -p fplopt-cli
    uv run python dev/archiver_parity.py [--raw raw] [--bin target/release/fplopt-rs]
        [--only daily,tick-window] [--keep DIR]

Takes ~5 minutes (the fault scenarios wait out real retry backoffs). Logs one row to
results/experiments.csv (family `rust parity`) and exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fplopt.adapters.espn import LEAGUES, EspnClient  # noqa: E402
from fplopt.adapters.football_data import FootballDataClient  # noqa: E402
from fplopt.adapters.fpl import FplClient  # noqa: E402
from fplopt.adapters.fplcache import FplcacheClient  # noqa: E402
from fplopt.adapters.http import make_client  # noqa: E402
from fplopt.adapters.odds import OddsClient  # noqa: E402
from fplopt.adapters.vaastav import PINNED_COMMIT, VaastavClient, git_blob_sha  # noqa: E402
from fplopt.backtest.evaluate import family_variants, log_experiment  # noqa: E402
from fplopt.ingest import espn as espn_ingest  # noqa: E402
from fplopt.ingest import health, history, jobs  # noqa: E402
from fplopt.ingest.raw_store import RawStore  # noqa: E402
from fplopt.seasons import football_data_code  # noqa: E402

TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{6}Z")
ISO = re.compile(r'"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00"')
REASON = re.compile(r'"(reason|error)": "([A-Za-z_]+): [^"]*"')
FAMILY = "rust parity"
N_ELEMENTS = 25  # trimmed bootstrap: a full element-summary run in seconds
FPLCACHE_SHA = "a" * 40
ODDS_KEY = "parity-test-key"


# --- recorded responses ----------------------------------------------------------------------


def latest(directory: Path, pattern: str = "*") -> Path:
    paths = sorted(p for p in directory.glob(pattern) if TS.match(p.name))
    if not paths:
        raise SystemExit(f"no recordings under {directory}")
    return paths[-1]


def gunzip(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


class Recording:
    """URL -> (status, headers, body) from real payloads in `raw`; `faults` adds failures."""

    def __init__(self, raw: Path, faults: bool) -> None:
        self.raw = raw
        self.faults = faults
        self.routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
        self.unmatched: list[str] = []
        self._fpl()
        self._odds()
        self._props()
        self._football_data()
        self._espn()
        self._vaastav()
        self._fplcache()

    def add(self, path: str, body: bytes, status: int = 200, **query: str) -> None:
        self.routes[key(path, query)] = (status, {}, body)

    def bootstrap(self) -> dict:
        boot = json.loads(gunzip(latest(self.raw / "fpl" / "bootstrap-static", "*.json.gz")))
        boot["elements"] = boot["elements"][:N_ELEMENTS]
        return boot

    def _fpl(self) -> None:
        boot = self.bootstrap()
        self.add("/fpl/api/bootstrap-static/", json.dumps(boot).encode())
        self.add("/fpl/api/fixtures/", gunzip(latest(self.raw / "fpl" / "fixtures", "*.json.gz")))
        run = latest(self.raw / "fpl" / "element-summary")
        ids = [e["id"] for e in boot["elements"]]
        for i, element_id in enumerate(ids):
            path = run / f"{element_id}.json.gz"
            body = gunzip(path) if path.exists() else json.dumps({"history": []}).encode()
            status = 404 if self.faults and i == 2 else 200
            self.add(f"/fpl/api/element-summary/{element_id}/", body, status)
        live = self.raw / "fpl" / "event-live" / "2026-27"
        fallback = gunzip(latest(live / "1", "*.json.gz"))
        for event in boot["events"]:
            gw = event["id"]
            gw_dir = live / str(gw)
            body = gunzip(latest(gw_dir, "*.json.gz")) if gw_dir.is_dir() else fallback
            status = 503 if self.faults and gw == 2 else 200
            self.add(f"/fpl/api/event/{gw}/live/", body, status)

    def _odds(self) -> None:
        body = gunzip(latest(self.raw / "odds" / "soccer_epl", "*.json.gz"))
        query = {
            "apiKey": ODDS_KEY,
            "regions": "uk",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        self.routes[key("/odds/v4/sports/soccer_epl/odds/", query)] = (
            200,
            {"x-requests-remaining": "321", "x-requests-used": "179", "x-requests-last": "2"},
            body,
        )

    def _props(self) -> None:
        """Synthetic player props, relative to now (the tick-window scenario's deadline is in
        an hour): one event already started, two this GW (one failing with faults), one long
        after the next deadline. Bodies have unsorted keys, floats and non-ASCII names, which
        both archivers re-encode as `json.dumps(json.loads(...))`."""
        now = datetime.now(UTC)
        kickoffs = {"ev0": -1, "ev1": 3, "ev2": 5, "ev3": 24 * 200}
        events = [
            {
                "id": event_id,
                "sport_key": "soccer_epl",
                "commence_time": (now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "home_team": "Brighton and Hove Albion",
                "away_team": "Nottingham Forest",
            }
            for event_id, h in kickoffs.items()
        ]
        base = {"apiKey": ODDS_KEY, "dateFormat": "iso"}
        self.routes[key("/odds/v4/sports/soccer_epl/events", base)] = (
            200,
            {},
            json.dumps(events).encode(),
        )
        query = base | {
            "regions": "us",
            "oddsFormat": "decimal",
            "markets": "player_goal_scorer_anytime,player_assists,player_shots_on_target",
        }
        for event in events:
            body = (
                (
                    '{"id":"@ID","bookmakers":[{"key":"draftkings","title":"DraftKings","markets":'
                    '[{"key":"player_goal_scorer_anytime","outcomes":[{"name":"Yes","description":'
                    '"Martin \u00d8degaard","price":3.10,"point":0.5},{"name":"Yes","description":'
                    '"Jo\u00e3o Pedro","price":2.25}]}]}],"commence_time":"@AT"}'
                )
                .replace("@ID", event["id"])
                .replace("@AT", event["commence_time"])
                .encode()
            )
            status = 500 if self.faults and event["id"] == "ev2" else 200
            path = f"/odds/v4/sports/soccer_epl/events/{event['id']}/odds"
            self.routes[key(path, query)] = (status, {"x-requests-remaining": "300"}, body)

    def _football_data(self) -> None:
        for year in range(2016, 2027):
            code = football_data_code(year)
            directory = self.raw / "football-data" / "E0" / code
            body = gunzip(latest(directory, "*.csv.gz"))
            status = 500 if self.faults and year == 2019 else 200
            self.add(f"/fd/mmz4281/{code}/E0.csv", body, status)

    def _espn(self) -> None:
        """Synthetic: per league and period, a completed match of ours (Arsenal, ESPN 359),
        an unfinished one, someone else's, and one before July 2016; summaries completed."""
        periods = [str(y) for y in (2025, 2026)]
        now = datetime.now(UTC)
        months = espn_ingest.recent_periods(now)
        for league in LEAGUES:
            for period in periods + months:
                year = int(period[:4])
                date = f"{year}-0{min(int(period[4:] or 8), 9)}-15T19:00Z"
                events = [
                    event(f"{league}-{period}-1", date, True, ["359", "1"]),
                    event(f"{league}-{period}-2", date, False, ["359", "2"]),
                    event(f"{league}-{period}-3", date, True, ["7", "8"]),
                    event(f"{league}-{period}-4", "2016-06-30T19:00Z", True, ["359", "3"]),
                ]
                path = f"/espn/{league}/scoreboard"
                self.add(path, json.dumps({"events": events}).encode(), dates=period, limit="1000")
                for ev in events:
                    status = {"type": {"completed": True}}
                    body = {"header": {"id": ev["id"], "competitions": [{"status": status}]}}
                    if self.faults and ev["id"].endswith(f"{periods[-1]}-3"):
                        body = {"oops": True}  # a 200 without the key: invalid payload
                    self.add(f"/espn/{league}/summary", json.dumps(body).encode(), event=ev["id"])

    def _vaastav(self) -> None:
        run = latest(self.raw / "vaastav" / "data")
        files = {}
        for rel in ("master_team_list", "2016-17/players_raw", "2016-17/teams", "2024-25/fixtures"):
            source = run / f"{rel}.csv.gz"
            files[f"data/{rel}.csv"] = gunzip(source) if source.exists() else b"a,b\n1,2\n"
        files["data/2016-17/gws/xP1.csv"] = b"leak\n"  # not selected
        files["data/2026-27/players_raw.csv"] = b"stale\n"  # not selected
        files["README.md"] = b"readme\n"
        tree = [{"path": p, "type": "blob", "sha": git_blob_sha(c)} for p, c in files.items()]
        tree.append({"path": "data", "type": "tree", "sha": "0" * 40})
        repo = "vaastav/Fantasy-Premier-League"
        self.add(
            f"/gh/repos/{repo}/git/trees/{PINNED_COMMIT}",
            json.dumps({"tree": tree, "truncated": False}).encode(),
            recursive="1",
        )
        for path, content in files.items():
            status = 404 if self.faults and path.endswith("teams.csv") else 200
            self.add(f"/raw/{repo}/{PINNED_COMMIT}/{path}", content, status)

    def _fplcache(self) -> None:
        snapshots = sorted((self.raw / "fplcache" / "bootstrap-static").glob("*.json.xz"))[:4]
        members = [(f"fplcache-{FPLCACHE_SHA}/README.md", b"public domain\n")]
        for path in snapshots:
            at = datetime.strptime(path.name[:18], "%Y-%m-%dT%H%M%SZ")
            name = f"cache/{at.year}/{at.month}/{at.day}/{at:%H%M}.json.xz"
            members.append((f"fplcache-{FPLCACHE_SHA}/{name}", path.read_bytes()))
        members.append(
            (f"fplcache-{FPLCACHE_SHA}/cache/2021/13/1/0000.json.xz", snapshots[0].read_bytes())
        )
        members.append(
            (f"fplcache-{FPLCACHE_SHA}/cache/2021/5/1/0000.json.xz", lzma_xz(b"not json"))
        )
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name, content in members:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        repo = "Randdalf/fplcache"
        self.add(f"/gh/repos/{repo}/commits/main", json.dumps({"sha": FPLCACHE_SHA}).encode())
        self.add(f"/codeload/{repo}/tar.gz/{FPLCACHE_SHA}", buffer.getvalue())


def lzma_xz(data: bytes) -> bytes:
    import lzma

    return lzma.compress(data, format=lzma.FORMAT_XZ)


def event(event_id: str, date: str, completed: bool, teams: list[str]) -> dict:
    competitors = [{"team": {"id": t}} for t in teams]
    return {
        "id": event_id,
        "date": date,
        "status": {"type": {"completed": completed}},
        "competitions": [{"competitors": competitors}],
    }


def key(path: str, query: dict[str, str]) -> str:
    return path + "?" + "&".join(f"{k}={v}" for k, v in sorted(query.items()))


def serve(recording: Recording) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            url = urlsplit(self.path)
            k = key(url.path, dict(parse_qsl(url.query)))
            status, headers, body = recording.routes.get(k, (404, {}, b"{}"))
            if k not in recording.routes:
                recording.unmatched.append(k)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# --- scenarios -------------------------------------------------------------------------------


class Python:
    """The Python jobs with clients pointed at the replay server (as `fplopt.cli.JOBS`)."""

    def __init__(self, base: str, teams_csv: Path) -> None:
        self.base = base
        self.teams_csv = teams_csv

    def run(self, store: RawStore, job: str, from_season: int = 2016) -> bool:
        b = self.base
        with make_client() as http:
            fpl = FplClient(http, f"{b}/fpl/api")
            odds = OddsClient(http, ODDS_KEY, f"{b}/odds/v4")
            fd = FootballDataClient(http, f"{b}/fd/mmz4281")
            espn = EspnClient(http, f"{b}/espn")
            run: dict[str, Callable[[], object]] = {
                "snapshot daily": lambda: jobs.run_daily(
                    store, fpl, odds, football_data=fd, espn=espn, espn_teams_csv=self.teams_csv
                ),
                "snapshot tick": lambda: jobs.run_tick(store, fpl, odds, props=odds),
                "backfill element-summary": lambda: jobs.backfill_element_summaries(store, fpl),
                "backfill football-data": lambda: history.backfill_football_data(
                    store, fd, first_season=from_season
                ),
                "backfill espn": lambda: espn_ingest.backfill_espn(
                    store, espn, espn_ingest.espn_team_ids(self.teams_csv), first_year=from_season
                ),
                "backfill vaastav": lambda: history.backfill_vaastav(
                    store, VaastavClient(http, f"{b}/gh", f"{b}/raw")
                ),
                "backfill fplcache": lambda: history.backfill_fplcache(
                    store, FplcacheClient(http, f"{b}/gh", f"{b}/codeload")
                ),
                "check freshness": lambda: health.check_freshness(store, jobs.utc_now()),
            }
            try:
                run[job]()
            except Exception as exc:
                print(f"    python: {type(exc).__name__}: {str(exc).splitlines()[:1]}")
                return False
        return True


def run_rust(binary: Path, raw: Path, base: str, cwd: Path, job: str, from_season: int) -> bool:
    env = {
        **os.environ,
        "FPLOPT_RAW_DIR": str(raw),
        "FPLOPT_FPL_URL": f"{base}/fpl/api",
        "FPLOPT_ODDS_URL": f"{base}/odds/v4",
        "FPLOPT_FOOTBALL_DATA_URL": f"{base}/fd/mmz4281",
        "FPLOPT_ESPN_URL": f"{base}/espn",
        "FPLOPT_GITHUB_API_URL": f"{base}/gh",
        "FPLOPT_GITHUB_RAW_URL": f"{base}/raw",
        "FPLOPT_CODELOAD_URL": f"{base}/codeload",
        "ODDS_API_KEY": ODDS_KEY,
        # never alert or ping from a parity run (set, so a .env can't fill them in)
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_ADMIN_CHAT_ID": "",
        "HEALTHCHECK_PING_URL": "",
        "RUST_LOG": "warn",
    }
    args = [str(binary), *job.split()]
    if job in ("backfill football-data", "backfill espn"):
        args += ["--from-season", str(from_season)]
    done = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if "failed" in line]
        print(f"    rust: exit {done.returncode}: {lines[-1][:200] if lines else ''}")
    return done.returncode == 0


def seed_bootstrap(
    store: RawStore, recording: Recording, at: datetime, deadline_in: timedelta
) -> None:
    """A bootstrap archived at `at` whose next unfinished GW's deadline is `deadline_in`
    from now."""
    boot = recording.bootstrap()
    for ev in boot["events"]:
        if not ev["finished"]:
            ev["deadline_time"] = (datetime.now(UTC) + deadline_in).strftime("%Y-%m-%dT%H:%M:%SZ")
            break
    store.write("fpl", "bootstrap-static", json.dumps(boot).encode(), at)


SCENARIOS: dict[str, tuple[str, int, Callable[[RawStore, Recording], None] | None]] = {
    # name: (job, from_season, seed)
    "daily": ("snapshot daily", 2016, None),
    "tick-empty": ("snapshot tick", 2016, None),
    "tick-window": (
        "snapshot tick",
        2016,
        lambda s, r: seed_bootstrap(
            s, r, datetime.now(UTC) - timedelta(hours=3), timedelta(hours=1)
        ),
    ),
    "tick-idle": (
        "snapshot tick",
        2016,
        lambda s, r: seed_bootstrap(
            s, r, datetime.now(UTC) - timedelta(hours=3), timedelta(hours=5)
        ),
    ),
    "element-summary": ("backfill element-summary", 2016, None),
    "football-data": ("backfill football-data", 2016, None),
    "vaastav": ("backfill vaastav", 2016, None),
    "fplcache": ("backfill fplcache", 2016, None),
    "espn": ("backfill espn", 2025, None),
    "freshness-ok": (
        "check freshness",
        2016,
        lambda s, r: seed_bootstrap(
            s, r, datetime.now(UTC) - timedelta(hours=30), timedelta(days=3)
        ),
    ),
    "freshness-stale": (
        "check freshness",
        2016,
        lambda s, r: seed_bootstrap(
            s, r, datetime.now(UTC) - timedelta(hours=40), timedelta(days=3)
        ),
    ),
    "freshness-empty": ("check freshness", 2016, None),
}
FAULT_SCENARIOS = ("daily", "tick-window", "element-summary", "football-data", "vaastav", "espn")


# --- comparison ------------------------------------------------------------------------------


def contents(path: Path) -> bytes:
    data = path.read_bytes()
    if path.name.endswith(".gz"):
        data = gzip.decompress(data)
    rel = path.as_posix()
    if "_manifest" in path.name or "/fplcache/runs/" in rel:
        text = ISO.sub('"<iso>"', data.decode())
        data = REASON.sub(r'"\1": "\2"', text).encode()
    return data


def tree(root: Path) -> Counter:
    out: Counter = Counter()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[(TS.sub("<ts>", path.relative_to(root).as_posix()), contents(path))] += 1
    return out


def compare(py: Path, rs: Path) -> list[str]:
    a, b = tree(py), tree(rs)
    problems = []
    for (name, content), n in sorted((a - b).items()):
        twin = [c for (m, c) in b if m == name]
        why = "content differs" if twin else "missing in rust"
        problems.append(f"{why}: {name} (x{n}){diff_hint(content, twin)}")
    for (name, _), n in sorted((b - a).items()):
        if not any(m == name for (m, _) in a):
            problems.append(f"only in rust: {name} (x{n})")
    return problems


def diff_hint(content: bytes, twins: list[bytes]) -> str:
    if not twins or len(content) > 2000:
        return ""
    return f"\n      python: {content[:300]!r}\n      rust:   {twins[0][:300]!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=REPO / "raw", help="recordings (read only)")
    parser.add_argument("--bin", type=Path, default=REPO / "target" / "release" / "fplopt-rs")
    parser.add_argument("--only", help="comma-separated scenario names")
    parser.add_argument("--keep", type=Path, help="keep the output trees here")
    parser.add_argument("--experiments", type=Path, default=REPO / "results" / "experiments.csv")
    args = parser.parse_args()
    binary = args.bin if args.bin.exists() else args.bin.with_suffix(".exe")
    if not binary.exists():
        parser.error(f"no Rust binary at {args.bin} (cargo build --release -p fplopt-cli)")

    names = [(n, False) for n in SCENARIOS] + [(n, True) for n in FAULT_SCENARIOS]
    if args.only:
        wanted = set(args.only.split(","))
        names = [(n, f) for n, f in names if (f"{n}-faults" if f else n) in wanted]

    work = Path(tempfile.mkdtemp(prefix="archiver-parity-")) if args.keep is None else args.keep
    work.mkdir(parents=True, exist_ok=True)
    cwd = work / "cwd"  # the Rust job's working directory: config/teams.csv, no .env
    (cwd / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "config" / "teams.csv", cwd / "config" / "teams.csv")

    results: dict[str, bool] = {}
    started = time.monotonic()
    for name, faults in names:
        label = f"{name}-faults" if faults else name
        job, from_season, seed = SCENARIOS[name]
        recording = Recording(args.raw, faults)
        server, base = serve(recording)
        try:
            seed_dir = work / label / "seed"
            if seed is not None:
                seed(RawStore(seed_dir), recording)
            py_dir, rs_dir = work / label / "python", work / label / "rust"
            for directory in (py_dir, rs_dir):
                if seed_dir.exists():
                    shutil.copytree(seed_dir, directory)
                else:
                    directory.mkdir(parents=True)
            print(f"{label}: {job}", flush=True)
            py_ok = Python(base, cwd / "config" / "teams.csv").run(
                RawStore(py_dir), job, from_season
            )
            rs_ok = run_rust(binary, rs_dir, base, cwd, job, from_season)
        finally:
            server.shutdown()
        problems = compare(py_dir, rs_dir)
        if py_ok != rs_ok:
            word = {True: "ok", False: "failed"}
            problems.insert(0, f"exit status differs: python {word[py_ok]}, rust {word[rs_ok]}")
        unmatched = sorted(set(recording.unmatched))
        n_files = sum(tree(py_dir).values())
        status = "PASS" if not problems else "FAIL"
        print(
            f"  {status}: {n_files} files, {'ok' if py_ok else 'failed'} in both"
            if not problems
            else f"  {status}"
        )
        for problem in problems[:20]:
            print(f"    {problem}")
        if unmatched:
            print(f"    (unrecorded requests, answered 404: {unmatched[:5]})")
        results[label] = not problems

    passed = all(results.values())
    took = time.monotonic() - started
    verdict = "PASS" if passed else "FAIL"
    print(f"\n{verdict}: {sum(results.values())}/{len(results)} scenarios ({took:.0f} s)")
    family_n = family_variants(args.experiments, FAMILY) + 1
    log_experiment(
        args.experiments,
        shlex.join(["python", "dev/archiver_parity.py", *sys.argv[1:]]),
        {"gate": "batch 1 (a) archiver", "raw": str(args.raw), "binary": str(binary)},
        {"passed": passed, "scenarios": results},
        family_n,
        FAMILY,
    )
    if args.keep is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
