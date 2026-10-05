"""Append-only store for raw API responses: gzipped JSON with a UTC timestamp in the path."""

from __future__ import annotations

import gzip
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TS_FORMAT = "%Y-%m-%dT%H%M%SZ"
SUFFIX = ".json.gz"


def format_ts(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(UTC).strftime(TS_FORMAT)


def parse_ts(text: str) -> datetime:
    return datetime.strptime(text, TS_FORMAT).replace(tzinfo=UTC)


class RawStore:
    """Writes each response exactly once; existing files are never overwritten."""

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
        tmp.write_bytes(gzip.compress(content, mtime=0))
        try:
            os.link(tmp, path)  # atomic, and raises FileExistsError instead of overwriting
        finally:
            tmp.unlink()
        return path

    def times(self, source: str, endpoint: str) -> list[datetime]:
        directory = self.root / source / endpoint
        if not directory.is_dir():
            return []
        return sorted(parse_ts(p.name.removesuffix(SUFFIX)) for p in directory.glob(f"*{SUFFIX}"))

    def latest(self, source: str, endpoint: str) -> Path | None:
        times = self.times(source, endpoint)
        if not times:
            return None
        return self.root / source / endpoint / f"{format_ts(times[-1])}{SUFFIX}"

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(gzip.decompress(Path(path).read_bytes()))
