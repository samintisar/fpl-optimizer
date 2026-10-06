"""Randdalf/fplcache: bootstrap-static snapshots ~4x/day since April 2021 (public domain)."""

from __future__ import annotations

import io
import re
import tarfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from fplopt.adapters.http import get_json_response

REPO = "Randdalf/fplcache"
API_URL = "https://api.github.com"
CODELOAD_URL = "https://codeload.github.com"

READ_BUFFER = 1 << 20
_SHA = re.compile(r"[0-9a-f]{40}")


def _gunzip(chunks: Iterator[bytes], max_out: int = READ_BUFFER) -> Iterator[bytes]:
    """Decompress a single-member gzip stream chunk by chunk (each output at most `max_out`
    bytes). Raises EOFError if the input ends before the gzip trailer (whose CRC and length
    zlib checks), and ValueError on bytes after it.

    tarfile's own `r|gz` does neither: a body cut at a 512-byte boundary reads as a clean
    end-of-archive, so a truncated download would look like a smaller, complete repo. Trailing
    bytes are an error, not ignored: codeload sends one gzip member, so anything after it is
    unexpected and could be archive content we would otherwise silently drop."""
    decompressor = zlib.decompressobj(wbits=31)
    for chunk in chunks:
        data = chunk
        while data:
            if decompressor.eof:
                raise ValueError("unexpected data after the end of the fplcache gzip stream")
            out = decompressor.decompress(data, max_out)
            data = decompressor.unconsumed_tail or decompressor.unused_data
            if out:
                yield out
            # A full output buffer may leave decompressed bytes pending with no input left.
            while len(out) == max_out and not decompressor.eof and not data:
                out = decompressor.decompress(b"", max_out)
                data = decompressor.unconsumed_tail
                if out:
                    yield out
    if not decompressor.eof:
        raise EOFError("fplcache tarball truncated (gzip stream ended early)")


class _ChunkReader(io.RawIOBase):
    """File-like view over an iterator of byte chunks, so tarfile can stream an HTTP body.
    Holds at most one chunk at a time. An error from the iterator is re-raised on every later
    read: tarfile would otherwise see the next read as a clean end of archive."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = iter(chunks)
        self._pending = memoryview(b"")
        self._error: BaseException | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self._pending:
            if self._error is not None:
                raise self._error
            try:
                self._pending = memoryview(next(self._chunks))
            except StopIteration:
                return 0
            except Exception as exc:
                self._error = exc
                raise
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
        `extractfile` before moving on. No retries: a failed download fails the caller, and so
        does a truncated one (EOFError) — see `_gunzip`. On a normal exit the rest of the body
        is read and checked, so leaving the block early still downloads everything."""
        url = f"{self._codeload}/{REPO}/tar.gz/{commit}"
        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            chunks = _gunzip(response.iter_bytes())
            reader = io.BufferedReader(_ChunkReader(chunks), READ_BUFFER)
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                yield tar
            # tarfile stops at the end-of-archive blocks; read the rest (padding and the gzip
            # trailer) so `_gunzip` can check the stream really ended there.
            while reader.read(READ_BUFFER):
                pass
