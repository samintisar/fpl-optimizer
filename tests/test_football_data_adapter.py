import httpx
import pytest

from fplopt.adapters.football_data import FootballDataClient, require_csv
from fplopt.adapters.http import InvalidPayload, make_client

# First bytes of real E0.csv files (checked 2026-10-06).
HEADER_1617 = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS\r\n"
HEADER_1920 = b"Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee\r\n"
HEADER_2526 = b"\xef\xbb\xbfDiv,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG\r\n"


def recording_client(body):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=body)

    return make_client(httpx.MockTransport(handler)), seen


def test_epl_season_hits_expected_url_and_returns_bytes_unchanged():
    body = HEADER_1617 + b"E0,13/08/16,Burnley,Swansea,0,1,A,0,0,D,J Gillett,10\r\n"
    client, seen = recording_client(body)
    assert FootballDataClient(client).epl_season(2016) == body
    assert seen == ["https://www.football-data.co.uk/mmz4281/1617/E0.csv"]


def test_epl_season_accepts_bom_prefixed_body():
    client, seen = recording_client(HEADER_2526)
    assert FootballDataClient(client).epl_season(2025) == HEADER_2526
    assert seen == ["https://www.football-data.co.uk/mmz4281/2526/E0.csv"]


@pytest.mark.parametrize("header", [HEADER_1617, HEADER_1920, HEADER_2526])
def test_require_csv_accepts_every_real_header_variant(header):
    require_csv(header)


@pytest.mark.parametrize(
    "body",
    [
        b"<html><body>Not Found</body></html>",
        b"<!DOCTYPE html>\n<html>",
        b"",
        b"Date,HomeTeam\r\n",
        b"\xef\xbb\xbf",
    ],
)
def test_require_csv_rejects_non_csv(body):
    with pytest.raises(InvalidPayload):
        require_csv(body)
