"""Randdalf/fplcache: bootstrap-static snapshots ~4x/day since April 2021 (public domain)."""

from __future__ import annotations

import io
import re
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from fplopt.adapters.http import get_json_response

REPO = "Randdalf/fplcache"
API_URL = "https://api.github.com"
CODELOAD_URL = "https://codeload.github.com"

READ_BUFFER = 1 << 20
_SHA = re.compile(r"[0-9a-f]{40}")


class _ChunkReader(io.RawIOBase):
    """File-like view over an iterator of byte chunks, so tarfile can stream an HTTP body.
    Holds at most one chunk at a time."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = iter(chunks)
        self._pending = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self._pending:
            try:
                self._pending = memoryview(next(self._chunks))
            except StopIteration:
                return 0
        size = min(len(buffer), len(self._pending))
        buffer[:size] = self._pending[:size]
        self._pending = self._pending[size:]
        return size


class FplcacheClient:
    def __init__(
        self,
        client: httpx.Client,
        api_url: str = API_URL,
        codeload_url: str = CODELOAD_URL,
    ) -> None:
        self._client = client
        self._api = api_url.rstrip("/")
        self._codeload = codeload_url.rstrip("/")

    def head_commit(self, branch: str = "main") -> str:
        """The commit sha `branch` points at now."""
        url = f"{self._api}/repos/{REPO}/commits/{branch}"
        sha = get_json_response(self._client, url).json().get("sha")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError(f"no commit sha for {REPO}@{branch}")
        return sha

    @contextmanager
    def tarball(self, commit: str) -> Iterator[tarfile.TarFile]:
        """Stream the repo tarball at `commit` (~0.9 GB) without holding it in memory.

        The TarFile is in stream mode: iterate it once, in order, and read each member with
        `extractfile` before moving on. No retries: a failed download fails the caller."""
        url = f"{self._codeload}/{REPO}/tar.gz/{commit}"
        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            reader = io.BufferedReader(_ChunkReader(response.iter_bytes()), READ_BUFFER)
            with tarfile.open(fileobj=reader, mode="r|gz") as tar:
                yield tar
