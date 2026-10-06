import gzip
import io
import os
import tarfile
import zlib

import httpx
import pytest

from fplopt.adapters.fplcache import FplcacheClient, _ChunkReader, _gunzip
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


def gzip_split_at_member(members, index, extra=0):
    """A .tar.gz of `members` deflated as two pieces, flushed so that the first piece alone
    decompresses to exactly the tar bytes before member `index`'s header plus `extra` bytes
    (extra=0: a cut at a member boundary). Returns (first piece, rest); together they are the
    complete body."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
        cut = tar.getmembers()[index].offset + extra
    compressor = zlib.compressobj(wbits=31)
    first = compressor.compress(raw[:cut]) + compressor.flush(zlib.Z_FULL_FLUSH)
    return first, compressor.compress(raw[cut:]) + compressor.flush()


def tarball_client(body):
    return FplcacheClient(
        make_client(httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    )


def read_all(fplcache):
    with fplcache.tarball(SHA) as tar:
        return {member.name: tar.extractfile(member).read() for member in tar}


def chunked(data, consumed):
    """Yield `data` in CHUNK-sized pieces, counting how many have been pulled."""
    for start in range(0, len(data), CHUNK):
        consumed.append(start)
        yield data[start : start + CHUNK]


def test_chunk_reader_reassembles_uneven_chunks():
    assert _ChunkReader(iter([b"ab", b"", b"cde"])).read() == b"abcde"
    reader = io.BufferedReader(_ChunkReader(iter([b"ab", b"", b"cde", b"f"])), 2)
    assert [reader.read(4), reader.read(4), reader.read(4)] == [b"abcd", b"ef", b""]


def test_gunzip_bounds_each_output_chunk():
    data = bytes(5 * 1024 * 1024)  # deflates ~1000:1, so one input chunk expands a lot
    body = gzip.compress(data)
    out = list(_gunzip(iter([body[:10], body[10:]]), max_out=64 * 1024))
    assert b"".join(out) == data
    assert max(len(piece) for piece in out) <= 64 * 1024


def test_gunzip_rejects_a_stream_without_its_trailer():
    body = gzip.compress(b"hello" * 100)
    with pytest.raises(EOFError):
        b"".join(_gunzip(iter([body[:-8]])))


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


TWO_MEMBERS = {f"fplcache-{SHA}/a.json": b'{"a": 1}', f"fplcache-{SHA}/b.json": b'{"b": 2}'}


def test_tarball_reads_a_complete_body():
    first, rest = gzip_split_at_member(TWO_MEMBERS, 1)
    assert read_all(tarball_client(first + rest)) == TWO_MEMBERS


def test_tarball_raises_when_cut_at_a_member_boundary():
    first, _ = gzip_split_at_member(TWO_MEMBERS, 1)
    with pytest.raises(EOFError, match="truncated"):
        read_all(tarball_client(first))


def test_tarball_raises_when_cut_inside_a_header():
    first, _ = gzip_split_at_member(TWO_MEMBERS, 1, extra=100)
    with pytest.raises(EOFError, match="truncated"):
        read_all(tarball_client(first))


def test_tarball_raises_on_data_after_the_gzip_stream():
    first, rest = gzip_split_at_member(TWO_MEMBERS, 1)
    with pytest.raises(ValueError, match="after the end"):
        read_all(tarball_client(first + rest + b"trailing"))
