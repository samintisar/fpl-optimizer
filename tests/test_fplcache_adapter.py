import io
import os
import tarfile

import httpx
import pytest

from fplopt.adapters.fplcache import FplcacheClient, _ChunkReader
from fplopt.adapters.http import make_client

SHA = "0123456789abcdef0123456789abcdef01234567"
CHUNK = 64 * 1024


def make_tarball(members):
    """An in-memory .tar.gz with `members` ({name: bytes}) in insertion order."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def chunked(data, consumed):
    """Yield `data` in CHUNK-sized pieces, counting how many have been pulled."""
    for start in range(0, len(data), CHUNK):
        consumed.append(start)
        yield data[start : start + CHUNK]


def test_chunk_reader_reassembles_uneven_chunks():
    assert _ChunkReader(iter([b"ab", b"", b"cde"])).read() == b"abcde"
    reader = io.BufferedReader(_ChunkReader(iter([b"ab", b"", b"cde", b"f"])), 2)
    assert [reader.read(4), reader.read(4), reader.read(4)] == [b"abcd", b"ef", b""]


def test_head_commit_returns_sha_from_commits_api():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"sha": SHA, "commit": {"message": "x"}})

    assert FplcacheClient(make_client(httpx.MockTransport(handler))).head_commit() == SHA
    assert seen == ["https://api.github.com/repos/Randdalf/fplcache/commits/main"]


def test_head_commit_rejects_a_non_sha():
    client = make_client(httpx.MockTransport(lambda r: httpx.Response(200, json={"sha": "x"})))
    with pytest.raises(ValueError):
        FplcacheClient(client).head_commit()


def test_tarball_streams_members_without_reading_the_whole_body():
    first = b'{"events": []}'
    filler = os.urandom(4 * 1024 * 1024)  # incompressible, so the .tar.gz is ~4 MB
    body = make_tarball({f"fplcache-{SHA}/a.json": first, f"fplcache-{SHA}/filler.bin": filler})
    total_chunks = len(range(0, len(body), CHUNK))
    consumed = []
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=chunked(body, consumed))

    fplcache = FplcacheClient(make_client(httpx.MockTransport(handler)))
    with fplcache.tarball(SHA) as tar:
        members = iter(tar)
        member = next(members)
        assert member.name == f"fplcache-{SHA}/a.json"
        assert tar.extractfile(member).read() == first
        assert len(consumed) < total_chunks / 2  # still streaming, not buffered
        member = next(members)
        assert tar.extractfile(member).read() == filler
    assert len(consumed) == total_chunks
    assert seen == [f"https://codeload.github.com/Randdalf/fplcache/tar.gz/{SHA}"]


def test_tarball_raises_on_http_error():
    client = make_client(httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        with FplcacheClient(client).tarball(SHA):
            pass
