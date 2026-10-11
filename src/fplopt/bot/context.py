"""The plan context (Phase 7 plan §1): everything a user's plan for the next deadline needs
except the squad, computed once per change of the archive and shared by every user.

`build_context(store, now)` takes the next deadline after `now` in `data/` and, at that
deadline's view:
- the v1 xP frame (the system that passed Phase 6; `LIVE_XP`);
- the player pool;
- the season's players (element_id, player_key, web name);
- GW -> gw_index;
- each GW's deadline prices up to the planned one (the entry import's purchase prices of
  players held since a manager's first GW).

`ContextStore` writes a context to `<root>/<context_id>/` (Parquet + `meta.json`) under a
temporary name and renames it into place, so readers never see half a context; the newest
`context_id` (a UTC stamp) is the current one. The bot only reads contexts, never `data/`,
so a rebuild of `data/` can't race it.

`raw_fingerprint(raw_dir)` identifies the archive's content (the relative paths of every
FPL, odds and football-data file), so a new context is built only when something new was
archived (or the code or the next deadline changed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fplopt.features.store import DataStore

__all__ = (
    "LIVE_XP",
    "ContextStore",
    "PlanContext",
    "build_context",
    "git_sha",
    "next_gameweek",
    "raw_fingerprint",
)

log = logging.getLogger(__name__)

LIVE_XP = "v1"  # the xP model of the system that passed Phase 6 (`holdout.SYSTEM`)
FINGERPRINT_SOURCES = ("fpl", "odds", "football-data")
KEEP_CONTEXTS = 12  # ContextStore.prune keeps this many newest contexts
_FRAMES = ("xp", "pool", "players", "prices", "gameweeks")
_STAMP = "%Y-%m-%dT%H%M%S%fZ"  # sortable; microseconds, so two builds never collide


@dataclass(frozen=True)
class PlanContext:
    """Inputs for every plan at one deadline (module docstring).

    `xp`: the LIVE_XP frame; `pool`: `player_pool` at the deadline; `players`:
    `element_id, player_key, web_name` of the season; `prices`: `gw, element_id, price`
    (each GW's deadline price, GW 1..`gw`); `gameweeks`: `gw, gw_index` of the season.
    `fingerprint`: the archive's `raw_fingerprint` it was built from; `bootstrap_at`: the
    newest bootstrap snapshot's time; `git_sha`: the code's commit."""

    context_id: str
    season: int
    gw: int
    gw_index: int
    deadline: datetime
    created_at: datetime
    fingerprint: str
    bootstrap_at: datetime | None
    git_sha: str | None
    xp_model: str
    xp: pd.DataFrame = field(repr=False)
    pool: pd.DataFrame = field(repr=False)
    players: pd.DataFrame = field(repr=False)
    prices: pd.DataFrame = field(repr=False)
    gameweeks: pd.DataFrame = field(repr=False)

    @property
    def gw_index_map(self) -> dict[int, int]:
        return {
            int(g): int(i)
            for g, i in zip(self.gameweeks["gw"], self.gameweeks["gw_index"], strict=True)
        }

    @property
    def element_keys(self) -> dict[int, int]:
        """element_id -> player_key."""
        return {
            int(e): int(k)
            for e, k in zip(self.players["element_id"], self.players["player_key"], strict=True)
        }

    @property
    def names(self) -> dict[int, str]:
        """player_key -> web name."""
        return {
            int(k): str(n)
            for k, n in zip(self.players["player_key"], self.players["web_name"], strict=True)
        }

    def initial_prices(self, gw: int) -> dict[int, int]:
        """element_id -> price at GW `gw`'s deadline (the entry import's `initial_prices`)."""
        rows = self.prices[self.prices["gw"] == gw]
        return {int(e): int(p) for e, p in zip(rows["element_id"], rows["price"], strict=True)}

    def meta(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "season": self.season,
            "gw": self.gw,
            "gw_index": self.gw_index,
            "deadline": self.deadline.isoformat(),
            "created_at": self.created_at.isoformat(),
            "fingerprint": self.fingerprint,
            "bootstrap_at": None if self.bootstrap_at is None else self.bootstrap_at.isoformat(),
            "git_sha": self.git_sha,
            "xp_model": self.xp_model,
        }


def next_gameweek(store: DataStore, now: datetime) -> tuple[int, int] | None:
    """(season, gw_index) of the first deadline after `now` in the data, or None."""
    from fplopt.backtest.simulator import FAR_FUTURE

    gameweeks = store.as_of(FAR_FUTURE).table(
        "gameweek", columns=["season", "gw", "gw_index", "deadline_time"]
    )
    future = gameweeks[gameweeks["deadline_time"] > pd.Timestamp(now)]
    if future.empty:
        return None
    row = future.sort_values(["deadline_time", "season", "gw_index"], kind="mergesort").iloc[0]
    return int(row["season"]), int(row["gw_index"])


def build_context(
    store: DataStore,
    now: datetime,
    *,
    fingerprint: str = "",
    bootstrap_at: datetime | None = None,
    sha: str | None = None,
    xp_model: str = LIVE_XP,
) -> PlanContext | None:
    """The context for the next deadline after `now` (None if the data has none)."""
    from fplopt.backtest.simulator import season_schedule
    from fplopt.features.baseline import player_pool
    from fplopt.models import MODELS

    found = next_gameweek(store, now)
    if found is None:
        return None
    season, gw_index = found
    schedule = season_schedule(store, season, gw_index)
    deadline = schedule["deadline_time"].iloc[0]
    gw = int(schedule["gw"].iloc[0])
    view = store.as_of(deadline)
    pool = player_pool(view).reset_index(drop=True)
    players = view.table(
        "player_season", columns=["player_key", "season", "element_id", "web_name"]
    )
    players = (
        players[players["season"] == season]
        .drop_duplicates("element_id", keep="last")[["element_id", "player_key", "web_name"]]
        .sort_values("element_id", kind="mergesort")
        .reset_index(drop=True)
    )
    gameweeks = view.table("gameweek", columns=["season", "gw", "gw_index"])
    gameweeks = (
        gameweeks[gameweeks["season"] == season][["gw", "gw_index"]]
        .astype("int64")
        .sort_values("gw_index", kind="mergesort")
        .reset_index(drop=True)
    )
    key_elements = {
        int(k): int(e) for e, k in zip(players["element_id"], players["player_key"], strict=True)
    }
    price_rows = []
    for past_gw, past_index in zip(gameweeks["gw"], gameweeks["gw_index"], strict=True):
        if past_index > gw_index:
            continue
        past_deadline = season_schedule(store, season, int(past_index))["deadline_time"].iloc[0]
        past_pool = player_pool(store.as_of(past_deadline))
        for key, price in zip(past_pool["player_key"], past_pool["price"], strict=True):
            if int(key) in key_elements:
                price_rows.append((int(past_gw), key_elements[int(key)], int(price)))
    prices = pd.DataFrame(price_rows, columns=["gw", "element_id", "price"], dtype="int64")
    xp = MODELS[xp_model](view).reset_index(drop=True)
    created = datetime.now(UTC)
    return PlanContext(
        context_id=created.strftime(_STAMP),
        season=season,
        gw=gw,
        gw_index=gw_index,
        deadline=pd.Timestamp(deadline).to_pydatetime(),
        created_at=created,
        fingerprint=fingerprint,
        bootstrap_at=bootstrap_at,
        git_sha=sha,
        xp_model=xp_model,
        xp=xp,
        pool=pool,
        players=players,
        prices=prices,
        gameweeks=gameweeks,
    )


class ContextStore:
    """Plan contexts under `root` (module docstring)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._cache: PlanContext | None = None

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / "meta.json").is_file()
        )

    def write(self, context: PlanContext) -> Path:
        """Write `context` atomically; returns its directory."""
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / context.context_id
        if final.exists():
            raise FileExistsError(f"context {context.context_id} exists")
        tmp = self.root / f".{context.context_id}.{os.getpid()}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            for name in _FRAMES:
                getattr(context, name).to_parquet(tmp / f"{name}.parquet", index=False)
            (tmp / "meta.json").write_text(
                json.dumps(context.meta(), indent=2) + "\n", encoding="utf-8"
            )
            os.replace(tmp, final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return final

    def meta(self, context_id: str | None = None) -> dict[str, Any] | None:
        """The meta of `context_id` (default: the newest), without reading its frames."""
        ids = self.ids()
        context_id = context_id or (ids[-1] if ids else None)
        if context_id is None:
            return None
        return json.loads((self.root / context_id / "meta.json").read_text(encoding="utf-8"))

    def read(self, context_id: str) -> PlanContext:
        directory = self.root / context_id
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        frames = {name: pd.read_parquet(directory / f"{name}.parquet") for name in _FRAMES}
        return PlanContext(
            context_id=meta["context_id"],
            season=int(meta["season"]),
            gw=int(meta["gw"]),
            gw_index=int(meta["gw_index"]),
            deadline=datetime.fromisoformat(meta["deadline"]),
            created_at=datetime.fromisoformat(meta["created_at"]),
            fingerprint=str(meta["fingerprint"]),
            bootstrap_at=(
                None
                if meta["bootstrap_at"] is None
                else datetime.fromisoformat(meta["bootstrap_at"])
            ),
            git_sha=meta["git_sha"],
            xp_model=str(meta["xp_model"]),
            **frames,
        )

    def latest(self) -> PlanContext | None:
        """The newest context (cached until a newer one appears)."""
        ids = self.ids()
        if not ids:
            return None
        if self._cache is None or self._cache.context_id != ids[-1]:
            self._cache = self.read(ids[-1])
        return self._cache

    def prune(self, keep: int = KEEP_CONTEXTS) -> list[str]:
        """Delete all but the `keep` newest contexts (the recommendations in SQLite stay);
        returns the deleted ids."""
        ids = self.ids()
        old = ids[: max(0, len(ids) - keep)]
        for context_id in old:
            shutil.rmtree(self.root / context_id, ignore_errors=True)
        return old


def raw_fingerprint(raw_dir: Path, sources: tuple[str, ...] = FINGERPRINT_SOURCES) -> str:
    """sha256 over the sorted relative paths of every archived file of `sources` under
    `raw_dir` (temporary files, which start with '.', excluded): changes whenever anything
    new is archived."""
    raw_dir = Path(raw_dir)
    digest = hashlib.sha256()
    paths = []
    for source in sources:
        root = raw_dir / source
        if not root.is_dir():
            continue
        for directory, _, files in os.walk(root):
            for name in files:
                if not name.startswith("."):
                    paths.append(Path(directory, name).relative_to(raw_dir).as_posix())
    for path in sorted(paths):
        digest.update(path.encode("utf-8") + b"\n")
    return digest.hexdigest()


def git_sha(root: Path | None = None) -> str | None:
    """The code checkout's HEAD commit, or None if git can't tell."""
    root = Path(__file__).resolve().parents[3] if root is None else root
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None
