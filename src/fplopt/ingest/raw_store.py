"""Append-only store for raw API responses: gzipped JSON with a UTC timestamp in the path."""

from __future__ import annotations

import gzip
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TS_FORMAT = "%Y-%m-%dT%H%M%SZ"
SUFFIX = ".json.gz"
TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}Z$")


def format_ts(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(UTC).strftime(TS_FORMAT)


def parse_ts(text: str) -> datetime:
    return datetime.strptime(text, TS_FORMAT).replace(tzinfo=UTC)


def _fsync_dir(directory: Path) -> None:
    """Persist a new directory entry. POSIX only (Windows cannot open directories)."""
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RawStore:
    """Writes each response exactly once; existing files are never overwritten.

    `times()` / `latest()` list single-file snapshots (`<endpoint>/<ts>.json.gz`) only.
    Grouped runs written with `name=` live in `<endpoint>/<ts>/` directories and are not
    listed. Files whose names are not timestamps are ignored.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write(
        self,
        source: str,
        endpoint: str,
        content: bytes,
        fetched_at: datetime,
        name: str | None = None,
    ) -> Path:
        json.loads(content)  # refuse to archive non-JSON, e.g. an HTML error page
        stamp = format_ts(fetched_at)
        directory = self.root / source / endpoint
        if name is None:
            path = directory / f"{stamp}{SUFFIX}"
        else:
            directory = directory / stamp
            path = directory / f"{name}{SUFFIX}"
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as fh:
                fh.write(gzip.compress(content, mtime=0))
                fh.flush()
                os.fsync(fh.fileno())
            os.link(tmp, path)  # atomic, and raises FileExistsError instead of overwriting
            _fsync_dir(path.parent)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    def _entries(self, source: str, endpoint: str) -> list[tuple[datetime, Path]]:
        directory = self.root / source / endpoint
        if not directory.is_dir():
            return []
        entries = []
        for path in directory.glob(f"*{SUFFIX}"):
            stem = path.name.removesuffix(SUFFIX)
            if TS_PATTERN.match(stem):
                entries.append((parse_ts(stem), path))
        return sorted(entries)

    def times(self, source: str, endpoint: str) -> list[datetime]:
        return [ts for ts, _ in self._entries(source, endpoint)]

    def latest(self, source: str, endpoint: str) -> Path | None:
        entries = self._entries(source, endpoint)
        return entries[-1][1] if entries else None

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(gzip.decompress(Path(path).read_bytes()))
