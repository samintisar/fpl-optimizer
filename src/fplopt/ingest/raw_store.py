"""Append-only store for raw source data: compressed files with a UTC timestamp in the path."""

from __future__ import annotations

import gzip
import json
import lzma
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TS_FORMAT = "%Y-%m-%dT%H%M%SZ"
SUFFIX = ".json.gz"
TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}Z$")

COMPRESSED_SUFFIXES = (".gz", ".xz")
_DECOMPRESS: dict[str, Callable[[bytes], bytes]] = {
    ".gz": gzip.decompress,
    ".xz": lzma.decompress,
}


def format_ts(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(UTC).strftime(TS_FORMAT)


def parse_ts(text: str) -> datetime:
    return datetime.strptime(text, TS_FORMAT).replace(tzinfo=UTC)


def gzip_bytes(content: bytes) -> bytes:
    """Deterministic gzip (no mtime), as used for every `.gz` file in the store."""
    return gzip.compress(content, mtime=0)


def _check_relative(text: str, what: str) -> None:
    """Reject anything that could escape the store: absolute paths, '..', empty parts, '\\',
    and ':' (a Windows drive prefix such as 'C:x' would replace the store root)."""
    parts = text.split("/")
    if (
        not text
        or text.startswith("/")
        or "\\" in text
        or ":" in text
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"unsafe {what}: {text!r}")


def _check_suffix(suffix: str) -> None:
    """Raw files are always compressed; the suffix says how (e.g. `.csv.gz`, `.json.xz`)."""
    if (
        len(suffix) < 2
        or not suffix.startswith(".")
        or "/" in suffix
        or "\\" in suffix
        or not suffix.endswith(COMPRESSED_SUFFIXES)
    ):
        raise ValueError(f"suffix must be a compressed file suffix like .json.gz: {suffix!r}")


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
    """Writes each file exactly once; existing files are never overwritten.

    Layout: single-file snapshots at `<source>/<endpoint>/<ts><suffix>`; grouped runs written
    with `name=` at `<source>/<endpoint>/<ts>/<name><suffix>` (`endpoint` and `name` may
    contain `/`). `write` takes JSON and gzips it; `write_bytes` stores already-compressed
    bytes verbatim, with a suffix saying how they are compressed (`.csv.gz`, `.json.xz`, ...).

    `entries()` / `times()` / `latest()` list single-file snapshots with the given suffix only.
    Grouped runs are not listed. Files whose names are not timestamps are ignored.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _directory(self, source: str, endpoint: str) -> Path:
        _check_relative(source, "source")
        if "/" in source:
            raise ValueError(f"unsafe source: {source!r}")
        _check_relative(endpoint, "endpoint")
        return self.root / source / endpoint

    def path_for(
        self,
        source: str,
        endpoint: str,
        fetched_at: datetime,
        *,
        suffix: str = SUFFIX,
        name: str | None = None,
    ) -> Path:
        """Where a file fetched at `fetched_at` lives (whether or not it exists yet)."""
        directory = self._directory(source, endpoint)
        _check_suffix(suffix)
        stamp = format_ts(fetched_at)
        if name is None:
            return directory / f"{stamp}{suffix}"
        _check_relative(name, "name")
        return directory / stamp / f"{name}{suffix}"

    def write_bytes(
        self,
        source: str,
        endpoint: str,
        data: bytes,
        fetched_at: datetime,
        *,
        suffix: str,
        name: str | None = None,
    ) -> Path:
        """Store `data` verbatim; the caller compresses it to match `suffix`."""
        path = self.path_for(source, endpoint, fetched_at, suffix=suffix, name=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.link(tmp, path)  # atomic, and raises FileExistsError instead of overwriting
            _fsync_dir(path.parent)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    def write(
        self,
        source: str,
        endpoint: str,
        content: bytes,
        fetched_at: datetime,
        name: str | None = None,
    ) -> Path:
        json.loads(content)  # refuse to archive non-JSON, e.g. an HTML error page
        return self.write_bytes(
            source, endpoint, gzip_bytes(content), fetched_at, suffix=SUFFIX, name=name
        )

    def entries(
        self, source: str, endpoint: str, suffix: str = SUFFIX
    ) -> list[tuple[datetime, Path]]:
        """(timestamp, path) of every single-file snapshot with `suffix`, oldest first."""
        directory = self._directory(source, endpoint)
        _check_suffix(suffix)
        if not directory.is_dir():
            return []
        entries = []
        for path in directory.glob(f"*{suffix}"):
            stem = path.name.removesuffix(suffix)
            if TS_PATTERN.match(stem):
                entries.append((parse_ts(stem), path))
        return sorted(entries)

    def times(self, source: str, endpoint: str, suffix: str = SUFFIX) -> list[datetime]:
        return [ts for ts, _ in self.entries(source, endpoint, suffix)]

    def latest(self, source: str, endpoint: str, suffix: str = SUFFIX) -> Path | None:
        entries = self.entries(source, endpoint, suffix)
        return entries[-1][1] if entries else None

    @staticmethod
    def read_bytes(path: Path) -> bytes:
        """The decompressed content of a stored file (by its final suffix: `.gz` or `.xz`)."""
        path = Path(path)
        decompress = _DECOMPRESS.get(path.suffix)
        if decompress is None:
            raise ValueError(f"not a compressed raw file: {path.name!r}")
        return decompress(path.read_bytes())

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(RawStore.read_bytes(path))
