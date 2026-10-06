import httpx
import pytest

from fplopt.adapters.http import make_client
from fplopt.adapters.vaastav import (
    PINNED_COMMIT,
    BlobMismatch,
    VaastavClient,
    git_blob_sha,
)


def recording_client(respond):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return respond(request)

    return make_client(httpx.MockTransport(handler)), seen


def test_git_blob_sha_matches_git():
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_tree_hits_recursive_tree_url_and_keeps_only_blobs():
    listing = {
        "sha": PINNED_COMMIT,
        "truncated": False,
        "tree": [
            {"path": "data", "type": "tree", "sha": "t1"},
            {"path": "data/master_team_list.csv", "type": "blob", "sha": "b1"},
            {"path": "data/2016-17/gws/merged_gw.csv", "type": "blob", "sha": "b2"},
            {"path": "some-submodule", "type": "commit", "sha": "c1"},
        ],
    }
    client, seen = recording_client(lambda request: httpx.Response(200, json=listing))
    assert VaastavClient(client).tree(PINNED_COMMIT) == {
        "data/master_team_list.csv": "b1",
        "data/2016-17/gws/merged_gw.csv": "b2",
    }
    assert seen == [
        "https://api.github.com/repos/vaastav/Fantasy-Premier-League/git/trees/"
        f"{PINNED_COMMIT}?recursive=1"
    ]


def test_tree_refuses_truncated_listing():
    listing = {"truncated": True, "tree": [{"path": "a.csv", "type": "blob", "sha": "b"}]}
    client, _ = recording_client(lambda request: httpx.Response(200, json=listing))
    with pytest.raises(RuntimeError, match="truncated"):
        VaastavClient(client).tree(PINNED_COMMIT)


def test_file_quotes_path_and_returns_verified_bytes():
    body = b"goals,xG\n1,0.4\n"
    client, seen = recording_client(lambda request: httpx.Response(200, content=body))
    vaastav = VaastavClient(client)
    path = "data/2021-22/understat/Martin_Ødegaard_1.csv"
    assert vaastav.file(PINNED_COMMIT, path, git_blob_sha(body)) == body
    odd = "data/2021-22/understat/Dara_O&#039;Shea_8756.csv"
    vaastav.file(PINNED_COMMIT, odd, git_blob_sha(body))
    prefix = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/{PINNED_COMMIT}"
    assert seen == [
        f"{prefix}/data/2021-22/understat/Martin_%C3%98degaard_1.csv",
        f"{prefix}/data/2021-22/understat/Dara_O%26%23039%3BShea_8756.csv",
    ]


def test_file_raises_on_blob_sha_mismatch():
    client, _ = recording_client(lambda request: httpx.Response(200, content=b"truncated"))
    with pytest.raises(BlobMismatch):
        VaastavClient(client).file(PINNED_COMMIT, "data/x.csv", git_blob_sha(b"full content"))
