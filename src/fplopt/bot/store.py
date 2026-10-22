"""User state in SQLite (PLAN §2-3, Phase 7 plan §2): who may use the bot, their FPL team,
settings, per-GW overrides, the imported squads and every recommendation made (the live log
of PLAN §8 together with the plan contexts, `fplopt.bot.context`).

Everything is keyed by `user_id` (CLAUDE.md); the Telegram `chat_id` is only how a user is
reached. Nothing else about a user is stored. One connection per call site; WAL mode, so the
bot and the alert job can both write.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = (
    "KINDS",
    "STATUSES",
    "Recommendation",
    "User",
    "UserStore",
)

log = logging.getLogger(__name__)

STATUSES = ("pending", "active", "blocked")
# Waits (seconds) before retrying a connection that hit SQLite's "disk I/O error": seen twice
# on ubuntuworker (2026-10-21) opening the database, transient (the next run was fine).
OPEN_RETRIES = (1.0, 3.0)
# Recommendation kinds: the 24-hour alert, the silent re-run before the deadline, /plan.
KINDS = ("alert24", "alert2", "plan")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'blocked')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_settings (
    user_id INTEGER PRIMARY KEY REFERENCES users (user_id),
    entry_id INTEGER,
    alerts INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_overrides (
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    free_transfers INTEGER,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, season, gw)
);
CREATE TABLE IF NOT EXISTS user_state (
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    entry_id INTEGER NOT NULL,
    squad TEXT NOT NULL,
    context_id TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, season, gw)
);
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alert24', 'alert2', 'plan')),
    context_id TEXT,
    created_at TEXT NOT NULL,
    plans TEXT,
    decision TEXT,
    message TEXT,
    sent INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS recommendations_user_gw
    ON recommendations (user_id, season, gw, kind);
CREATE TABLE IF NOT EXISTS user_transfers (
    transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    element_out INTEGER NOT NULL,
    element_in INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chip_checks (
    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    context_id TEXT,
    created_at TEXT NOT NULL,
    best TEXT NOT NULL,
    scenarios TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alert_runs (
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alert24', 'alert2')),
    context_id TEXT NOT NULL,
    bootstrap_at TEXT NOT NULL,
    run_at TEXT NOT NULL,
    PRIMARY KEY (season, gw, kind)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class User:
    user_id: int
    chat_id: int
    status: str
    entry_id: int | None = None
    alerts: bool = True

    @property
    def active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class Recommendation:
    rec_id: int
    user_id: int
    season: int
    gw: int
    kind: str
    context_id: str | None
    created_at: str
    decision: dict[str, Any] | None
    sent: bool
    error: str | None


class UserStore:
    """The bot's SQLite database at `path` (created with its tables on first use)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._open()) as db:
            # WAL is a property of the database file: set once here, not on every connection
            # (switching it needs a write lock).
            db.execute("PRAGMA journal_mode=WAL")
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _open(self) -> sqlite3.Connection:
        """A connection that has read the database (its journal mode), retrying a transient
        disk I/O error (`OPEN_RETRIES`)."""
        for wait in (*OPEN_RETRIES, None):
            db = sqlite3.connect(self.path, timeout=30)
            try:
                db.execute("PRAGMA journal_mode").fetchone()
                return db
            except sqlite3.OperationalError as exc:
                db.close()
                if wait is None or "disk I/O error" not in str(exc):
                    raise
                log.warning("bot database: %s; retrying in %.0f s", exc, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(self._open()) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:  # one transaction: commit on success, roll back on error
                yield db

    # --- users ----------------------------------------------------------------------------

    def user(self, chat_id: int) -> User | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT u.user_id, u.chat_id, u.status, s.entry_id, s.alerts FROM users u "
                "LEFT JOIN user_settings s USING (user_id) WHERE u.chat_id = ?",
                (int(chat_id),),
            ).fetchone()
        return None if row is None else _user(row)

    def add_user(self, chat_id: int, status: str = "pending") -> User:
        """The user for `chat_id`, created with `status` if new (an existing user is
        returned unchanged)."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO users (chat_id, status, created_at) VALUES (?, ?, ?)",
                (int(chat_id), status, _now()),
            )
        user = self.user(chat_id)
        assert user is not None
        return user

    def set_status(self, chat_id: int, status: str) -> User | None:
        """Set an existing user's status; None if there is no such user."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._connect() as db:
            db.execute("UPDATE users SET status = ? WHERE chat_id = ?", (status, int(chat_id)))
        return self.user(chat_id)

    def users(self, status: str | None = None) -> list[User]:
        query = (
            "SELECT u.user_id, u.chat_id, u.status, s.entry_id, s.alerts FROM users u "
            "LEFT JOIN user_settings s USING (user_id)"
        )
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE u.status = ?"
            params = (status,)
        with self._connect() as db:
            rows = db.execute(query + " ORDER BY u.user_id", params).fetchall()
        return [_user(r) for r in rows]

    def alert_users(self) -> list[User]:
        """Active users with a team and alerts on."""
        return [u for u in self.users("active") if u.entry_id is not None and u.alerts]

    # --- settings and overrides -----------------------------------------------------------

    def _settings(self, db: sqlite3.Connection, user_id: int, **values: Any) -> None:
        db.execute(
            "INSERT INTO user_settings (user_id, updated_at) VALUES (?, ?) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id, _now()),
        )
        for key, value in values.items():
            db.execute(
                f"UPDATE user_settings SET {key} = ?, updated_at = ? WHERE user_id = ?",
                (value, _now(), user_id),
            )

    def set_entry(self, user_id: int, entry_id: int) -> None:
        with self._connect() as db:
            self._settings(db, user_id, entry_id=int(entry_id))

    def set_alerts(self, user_id: int, on: bool) -> None:
        with self._connect() as db:
            self._settings(db, user_id, alerts=int(bool(on)))

    def set_free_transfers(self, user_id: int, season: int, gw: int, n: int | None) -> None:
        """The free transfers to plan GW `gw` with (None clears the override)."""
        with self._connect() as db:
            db.execute(
                "INSERT INTO user_overrides (user_id, season, gw, free_transfers, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (user_id, season, gw) DO UPDATE SET "
                "free_transfers = excluded.free_transfers, updated_at = excluded.updated_at",
                (user_id, season, gw, n, _now()),
            )

    def free_transfers(self, user_id: int, season: int, gw: int) -> int | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT free_transfers FROM user_overrides "
                "WHERE user_id = ? AND season = ? AND gw = ?",
                (user_id, season, gw),
            ).fetchone()
        return None if row is None or row[0] is None else int(row[0])

    def add_transfer(self, user_id: int, season: int, gw: int, out: int, into: int) -> None:
        """A transfer the user already made for GW `gw` (element ids), after any before it."""
        with self._connect() as db:
            db.execute(
                "INSERT INTO user_transfers (user_id, season, gw, element_out, element_in, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, season, gw, int(out), int(into), _now()),
            )

    def transfers(self, user_id: int, season: int, gw: int) -> list[tuple[int, int]]:
        """The transfers the user made for GW `gw`, in order: (element out, element in)."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT element_out, element_in FROM user_transfers WHERE user_id = ? AND "
                "season = ? AND gw = ? ORDER BY transfer_id",
                (user_id, season, gw),
            ).fetchall()
        return [(int(r[0]), int(r[1])) for r in rows]

    def drop_transfers(self, user_id: int, season: int, gw: int, last_only: bool) -> int:
        """Remove the GW's transfers (or only the newest); returns how many were removed."""
        with self._connect() as db:
            if last_only:
                cursor = db.execute(
                    "DELETE FROM user_transfers WHERE transfer_id = (SELECT MAX(transfer_id) "
                    "FROM user_transfers WHERE user_id = ? AND season = ? AND gw = ?)",
                    (user_id, season, gw),
                )
            else:
                cursor = db.execute(
                    "DELETE FROM user_transfers WHERE user_id = ? AND season = ? AND gw = ?",
                    (user_id, season, gw),
                )
            return int(cursor.rowcount)

    # --- squads and recommendations -------------------------------------------------------

    def save_squad(
        self,
        user_id: int,
        season: int,
        gw: int,
        entry_id: int,
        squad: Mapping[str, Any],
        context_id: str | None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO user_state (user_id, season, gw, entry_id, squad, context_id, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (user_id, season, gw) "
                "DO UPDATE SET entry_id = excluded.entry_id, squad = excluded.squad, "
                "context_id = excluded.context_id, updated_at = excluded.updated_at",
                (user_id, season, gw, entry_id, json.dumps(squad), context_id, _now()),
            )

    def squad(self, user_id: int, season: int, gw: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT squad FROM user_state WHERE user_id = ? AND season = ? AND gw = ?",
                (user_id, season, gw),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def record(
        self,
        user_id: int,
        season: int,
        gw: int,
        kind: str,
        *,
        context_id: str | None,
        plans: Any = None,
        decision: Mapping[str, Any] | None = None,
        message: str | None = None,
        sent: bool = False,
        error: str | None = None,
    ) -> int:
        """Log a recommendation (or a failed attempt, with `error`); returns its rec_id."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO recommendations (user_id, season, gw, kind, context_id, "
                "created_at, plans, decision, message, sent, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    season,
                    gw,
                    kind,
                    context_id,
                    _now(),
                    None if plans is None else json.dumps(plans),
                    None if decision is None else json.dumps(decision),
                    message,
                    int(sent),
                    error,
                ),
            )
            return int(cursor.lastrowid or 0)

    def last_sent_alert(self, user_id: int, season: int, gw: int) -> Recommendation | None:
        """The newest alert (24-hour or re-run) sent to the user for this GW."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM recommendations WHERE user_id = ? AND season = ? AND gw = ? "
                "AND kind IN ('alert24', 'alert2') AND sent = 1 ORDER BY rec_id DESC LIMIT 1",
                (user_id, season, gw),
            ).fetchone()
        return None if row is None else _recommendation(row)

    def recommendations(self, user_id: int) -> list[Recommendation]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM recommendations WHERE user_id = ? ORDER BY rec_id", (user_id,)
            ).fetchall()
        return [_recommendation(r) for r in rows]

    def record_chip_check(
        self,
        user_id: int,
        season: int,
        gw: int,
        *,
        context_id: str | None,
        best: str,
        scenarios: Mapping[str, float],
    ) -> None:
        """Log a `/chips` answer: the best scenario and every solved one's objective."""
        with self._connect() as db:
            db.execute(
                "INSERT INTO chip_checks (user_id, season, gw, context_id, created_at, best, "
                "scenarios) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, season, gw, context_id, _now(), best, json.dumps(dict(scenarios))),
            )

    def chip_checks(self, user_id: int) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM chip_checks WHERE user_id = ? ORDER BY check_id", (user_id,)
            ).fetchall()
        return [{**dict(r), "scenarios": json.loads(r["scenarios"])} for r in rows]

    # --- alert runs -----------------------------------------------------------------------

    def alert_run(self, season: int, gw: int, kind: str) -> dict[str, str] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT context_id, bootstrap_at, run_at FROM alert_runs "
                "WHERE season = ? AND gw = ? AND kind = ?",
                (season, gw, kind),
            ).fetchone()
        return None if row is None else dict(row)

    def mark_alert_run(
        self, season: int, gw: int, kind: str, context_id: str, bootstrap_at: datetime
    ) -> None:
        """Mark the GW's alert of `kind` done; a second mark (e.g. a manual `fplopt live
        tick` next to the timer's) keeps the first."""
        with self._connect() as db:
            db.execute(
                "INSERT INTO alert_runs (season, gw, kind, context_id, bootstrap_at, run_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (season, gw, kind) DO NOTHING",
                (season, gw, kind, context_id, bootstrap_at.isoformat(), _now()),
            )


def _user(row: sqlite3.Row) -> User:
    return User(
        user_id=int(row["user_id"]),
        chat_id=int(row["chat_id"]),
        status=str(row["status"]),
        entry_id=None if row["entry_id"] is None else int(row["entry_id"]),
        alerts=True if row["alerts"] is None else bool(row["alerts"]),
    )


def _recommendation(row: sqlite3.Row) -> Recommendation:
    return Recommendation(
        rec_id=int(row["rec_id"]),
        user_id=int(row["user_id"]),
        season=int(row["season"]),
        gw=int(row["gw"]),
        kind=str(row["kind"]),
        context_id=row["context_id"],
        created_at=str(row["created_at"]),
        decision=None if row["decision"] is None else json.loads(row["decision"]),
        sent=bool(row["sent"]),
        error=row["error"],
    )
