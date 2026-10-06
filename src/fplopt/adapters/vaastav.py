"""vaastav/Fantasy-Premier-League at a pinned commit: git tree listing + raw file bytes."""

from __future__ import annotations

import hashlib
from urllib.parse import quote

import httpx

from fplopt.adapters.http import get_json_response, get_response

REPO = "vaastav/Fantasy-Premier-League"
PINNED_COMMIT = "9779cdbc0c07f6c900c2d0c181ddf6bb9c800f88"  # 2026-08-28, "Add 26/27 gw1 data"
API_URL = "https://api.github.com"
RAW_URL = "https://raw.githubusercontent.com"


def git_blob_sha(data: bytes) -> str:
    """The sha git uses for a file's content, to verify downloads against the tree listing."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


class BlobMismatch(Exception):
    """Downloaded bytes do not match the blob sha in the pinned tree."""


class VaastavClient:
    def __init__(
        self, client: httpx.Client, api_url: str = API_URL, raw_url: str = RAW_URL
    ) -> None:
        self._client = client
        self._api = api_url.rstrip("/")
        self._raw = raw_url.rstrip("/")

    def tree(self, commit: str) -> dict[str, str]:
        """path -> blob sha for every file at `commit`. Refuses a truncated listing."""
        url = f"{self._api}/repos/{REPO}/git/trees/{commit}"
        data = get_json_response(self._client, url, params={"recursive": "1"}).json()
        if data.get("truncated"):
            raise RuntimeError(f"git tree listing for {REPO}@{commit} is truncated")
        return {entry["path"]: entry["sha"] for entry in data["tree"] if entry["type"] == "blob"}

    def file(self, commit: str, path: str, blob_sha: str) -> bytes:
        """The file's bytes at `commit`, verified against its blob sha."""
        url = f"{self._raw}/{REPO}/{commit}/{quote(path, safe='/')}"
        content = get_response(self._client, url).content
        if git_blob_sha(content) != blob_sha:
            raise BlobMismatch(f"{path}: content does not match blob {blob_sha}")
        return content
