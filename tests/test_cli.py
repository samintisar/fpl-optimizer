import logging
import signal
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from fplopt import cli
from fplopt.adapters.football_data import FootballDataClient
from fplopt.adapters.fpl import FplClient
from fplopt.adapters.odds import OddsClient
from fplopt.settings import Settings


def make_settings(tmp_path):
    return Settings(
        raw_dir=tmp_path,
        odds_api_key=None,
        telegram_bot_token="T",
        telegram_admin_chat_id="42",
    )


def test_successful_job_returns_zero(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setitem(cli.JOBS, "snapshot daily", lambda c: ran.append(c.odds))
    assert cli.main(["snapshot", "daily"], settings=make_settings(tmp_path)) == 0
    assert ran == [None]  # no odds key -> no odds client


def test_failed_job_alerts_and_returns_one(tmp_path, monkeypatch):
    def boom(c):
        raise RuntimeError("fpl down")

    alerts = []
    monkeypatch.setitem(cli.JOBS, "snapshot tick", boom)
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append((text, kw)))
    assert cli.main(["snapshot", "tick"], settings=make_settings(tmp_path)) == 1
    text, kw = alerts[0]
    assert "snapshot tick failed" in text
    assert "fpl down" in text
    assert kw == {"token": "T", "chat_id": "42"}


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["snapshot", "weekly"])


@pytest.fixture(autouse=True)
def restore_logging():
    names = ["", "httpx", "httpcore"]
    levels = {name: logging.getLogger(name).level for name in names}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def test_logging_hides_httpx_request_urls():
    cli.configure_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_settings_from_env():
    env = {"FPLOPT_RAW_DIR": "/srv/raw", "ODDS_API_KEY": "", "TELEGRAM_ADMIN_CHAT_ID": " 42 \n"}
    s = Settings.from_env(env)
    assert s.raw_dir == Path("/srv/raw").resolve()
    assert s.odds_api_key is None
    assert s.telegram_admin_chat_id == "42"
    assert Settings.from_env({}).raw_dir == Path("raw").resolve()
    assert Settings.from_env({}).data_dir == Path("data").resolve()
    assert (
        Settings.from_env({"FPLOPT_DATA_DIR": "/srv/data"}).data_dir == Path("/srv/data").resolve()
    )


def test_alert_text_is_redacted_and_single_line(tmp_path, monkeypatch):
    def boom(c):
        raise RuntimeError("GET https://x.test/?apiKey=SECRET failed\nsecond line")

    alerts = []
    monkeypatch.setitem(cli.JOBS, "snapshot daily", boom)
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    assert cli.main(["snapshot", "daily"], settings=make_settings(tmp_path)) == 1
    assert "SECRET" not in alerts[0]
    assert "second line" not in alerts[0]


def test_odds_client_built_when_key_set(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setitem(cli.JOBS, "snapshot daily", lambda c: seen.append(c.odds))
    settings = replace(make_settings(tmp_path), odds_api_key="K")
    assert cli.main(["snapshot", "daily"], settings=settings) == 0
    assert isinstance(seen[0], OddsClient)


def test_unconfigured_telegram_warns_and_still_fails(tmp_path, monkeypatch, caplog):
    def boom(c):
        raise RuntimeError("x")

    monkeypatch.setitem(cli.JOBS, "snapshot tick", boom)
    settings = replace(make_settings(tmp_path), telegram_bot_token=None)
    assert cli.main(["snapshot", "tick"], settings=settings) == 1
    assert "will not be alerted" in caplog.text


def test_setup_failure_is_alerted(tmp_path, monkeypatch):
    def broken_client():
        raise OSError("bad CA bundle")

    alerts = []
    monkeypatch.setattr(cli, "make_client", broken_client)
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    assert cli.main(["snapshot", "daily"], settings=make_settings(tmp_path)) == 1
    assert "bad CA bundle" in alerts[0]


def test_rules_export_passes_season_and_out_through(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setitem(
        cli.JOBS, "rules export", lambda c: seen.append((c.args.season, c.args.out))
    )
    settings = make_settings(tmp_path)
    assert cli.main(["rules", "export", "2026-27"], settings=settings) == 0
    assert cli.main(["rules", "export", "2025-26", "--out", "x/y"], settings=settings) == 0
    assert seen == [("2026-27", "config/scoring"), ("2025-26", "x/y")]


@pytest.mark.parametrize("command", ["element-summary", "football-data", "vaastav", "fplcache"])
def test_backfill_commands_are_valid(tmp_path, monkeypatch, command):
    ran = []
    monkeypatch.setitem(cli.JOBS, f"backfill {command}", lambda c: ran.append(c.store.root))
    assert cli.main(["backfill", command], settings=make_settings(tmp_path)) == 0
    assert ran == [tmp_path]


def test_backfill_football_data_from_season(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        cli.history,
        "backfill_football_data",
        lambda store, fd, **kw: seen.append((type(fd), kw)),
    )
    settings = make_settings(tmp_path)
    assert cli.main(["backfill", "football-data"], settings=settings) == 0
    assert cli.main(["backfill", "football-data", "--from-season", "2005"], settings=settings) == 0
    assert seen == [
        (FootballDataClient, {"first_season": 2016}),
        (FootballDataClient, {"first_season": 2005}),
    ]


def test_every_parsed_command_has_a_job():
    parser = cli.build_parser()
    extras = {"rules export": ["2026-27"], "build": ["fixture"]}
    for name in cli.JOBS:
        args = parser.parse_args(name.split() + extras.get(name, []))
        assert cli.job_name(args) == name


def test_build_dispatches_target_with_data_dir(tmp_path, monkeypatch):
    seen = []
    import fplopt.build

    monkeypatch.setattr(
        fplopt.build, "build", lambda names, ctx: seen.append((names, ctx.store.root, ctx.data_dir))
    )
    settings = replace(make_settings(tmp_path), data_dir=tmp_path / "data")
    assert cli.main(["build", "fixture"], settings=settings) == 0
    assert cli.main(["build", "all"], settings=settings) == 0
    assert seen == [
        (["fixture"], tmp_path, tmp_path / "data"),
        (["all"], tmp_path, tmp_path / "data"),
    ]


def test_build_unknown_table_fails_with_exit_one(tmp_path, monkeypatch):
    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    settings = replace(make_settings(tmp_path), data_dir=tmp_path / "data")
    assert cli.main(["build", "nope"], settings=settings) == 1
    assert "unknown table 'nope'" in alerts[0]


def test_bad_season_label_alerts_and_returns_one(tmp_path, monkeypatch):
    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    out = tmp_path / "out"
    assert (
        cli.main(
            ["rules", "export", "2026/27", "--out", str(out)], settings=make_settings(tmp_path)
        )
        == 1
    )
    assert "rules export failed" in alerts[0]
    assert "not a season label" in alerts[0]
    assert not out.exists()


def test_daily_job_archives_football_data(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli.jobs, "run_daily", lambda store, fpl, odds, **kw: calls.append((fpl, odds, kw))
    )
    assert cli.main(["snapshot", "daily"], settings=make_settings(tmp_path)) == 0
    ((fpl, odds, kw),) = calls
    assert isinstance(fpl, FplClient)
    assert odds is None
    assert isinstance(kw["football_data"], FootballDataClient)


def test_sigterm_during_a_job_raises_system_exit_and_handler_is_restored(tmp_path, monkeypatch):
    if not hasattr(signal, "SIGTERM"):
        pytest.skip("no SIGTERM on this platform")
    before = signal.getsignal(signal.SIGTERM)
    seen = []

    def job(c):
        handler = signal.getsignal(signal.SIGTERM)
        seen.append(handler)
        handler(signal.SIGTERM, None)  # what systemd's stop/timeout would deliver

    alerts = []
    monkeypatch.setitem(cli.JOBS, "snapshot daily", job)
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    with pytest.raises(SystemExit) as info:
        cli.main(["snapshot", "daily"], settings=make_settings(tmp_path))
    assert info.value.code == 143
    assert callable(seen[0]) and seen[0] is not before
    assert signal.getsignal(signal.SIGTERM) is before
    assert alerts == []  # systemd's OnFailure unit reports a killed run


def test_cli_import_does_not_load_pandas():
    """The 15-minute archiver jobs shouldn't pay for importing the build layer."""
    import subprocess
    import sys

    code = "import sys, fplopt.cli; print('pandas' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_check_freshness_passes_store_and_max_age(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        cli.health,
        "check_freshness",
        lambda store, now, max_age: seen.append((store.root, now.tzinfo is not None, max_age)),
    )
    settings = make_settings(tmp_path)
    assert cli.main(["check", "freshness"], settings=settings) == 0
    assert cli.main(["check", "freshness", "--max-age-hours", "12.5"], settings=settings) == 0
    assert seen == [
        (tmp_path, True, timedelta(hours=36)),
        (tmp_path, True, timedelta(hours=12.5)),
    ]


def test_check_freshness_on_an_empty_archive_alerts_and_returns_one(tmp_path, monkeypatch):
    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    assert cli.main(["check", "freshness"], settings=make_settings(tmp_path)) == 1
    assert "check freshness failed" in alerts[0]
    assert "StaleArchiveError" in alerts[0]


def test_check_leakage_passes_data_dir_deadlines_and_seed(tmp_path, monkeypatch):
    from fplopt.features import leakcheck

    seen = []
    monkeypatch.setattr(
        leakcheck, "run_leakage_check", lambda data_dir, n, seed: seen.append((data_dir, n, seed))
    )
    settings = replace(make_settings(tmp_path), data_dir=tmp_path / "data")
    assert cli.main(["check", "leakage"], settings=settings) == 0
    assert cli.main(["check", "leakage", "--deadlines", "3", "--seed", "7"], settings=settings) == 0
    assert seen == [(tmp_path / "data", 12, 0), (tmp_path / "data", 3, 7)]


def test_check_leakage_failure_alerts_and_returns_one(tmp_path, monkeypatch):
    from fplopt.features import leakcheck

    def leaky(data_dir, n, seed):
        raise leakcheck.LeakageError("2 leak(s) in 1 feature(s) at 1 deadline(s); first: ...")

    alerts = []
    monkeypatch.setattr(leakcheck, "run_leakage_check", leaky)
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    assert cli.main(["check", "leakage"], settings=make_settings(tmp_path)) == 1
    assert "check leakage failed" in alerts[0] and "LeakageError: 2 leak(s)" in alerts[0]


def test_check_leakage_end_to_end_on_synthetic_tables(tmp_path, built, caplog, monkeypatch):
    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name, df in built.items():
        df.to_parquet(data_dir / f"{name}.parquet")
    settings = replace(make_settings(tmp_path), data_dir=data_dir)
    with caplog.at_level(logging.INFO):
        assert cli.main(["check", "leakage", "--deadlines", "3"], settings=settings) == 0
    assert "leakage check passed: 3 deadline(s)" in caplog.text
    # A missing table fails the job.
    (data_dir / "player_match.parquet").unlink()
    assert cli.main(["check", "leakage"], settings=settings) == 1
    assert len(alerts) == 1 and "player_match" in alerts[0]


@pytest.fixture
def heartbeats(monkeypatch):
    sent = []
    monkeypatch.setattr(cli, "send_heartbeat", lambda url: sent.append(url) or True)
    return sent


@pytest.mark.parametrize("command", ["daily", "tick"])
def test_successful_snapshot_pings_the_heartbeat_url(tmp_path, monkeypatch, heartbeats, command):
    monkeypatch.setitem(cli.JOBS, f"snapshot {command}", lambda c: None)
    settings = replace(make_settings(tmp_path), healthcheck_ping_url="https://hc.test/abc")
    assert cli.main(["snapshot", command], settings=settings) == 0
    assert heartbeats == ["https://hc.test/abc"]


def test_failed_snapshot_does_not_ping(tmp_path, monkeypatch, heartbeats):
    def boom(c):
        raise RuntimeError("fpl down")

    monkeypatch.setitem(cli.JOBS, "snapshot daily", boom)
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: None)
    settings = replace(make_settings(tmp_path), healthcheck_ping_url="https://hc.test/abc")
    assert cli.main(["snapshot", "daily"], settings=settings) == 1
    assert heartbeats == []


def test_other_jobs_and_unset_url_do_not_ping(tmp_path, monkeypatch, heartbeats):
    monkeypatch.setitem(cli.JOBS, "backfill vaastav", lambda c: None)
    monkeypatch.setitem(cli.JOBS, "snapshot tick", lambda c: None)
    settings = replace(make_settings(tmp_path), healthcheck_ping_url="https://hc.test/abc")
    assert cli.main(["backfill", "vaastav"], settings=settings) == 0
    assert cli.main(["snapshot", "tick"], settings=make_settings(tmp_path)) == 0
    assert heartbeats == []


def test_failed_ping_does_not_fail_the_job(tmp_path, monkeypatch):
    def unreachable(request):
        raise httpx.ConnectError("down", request=request)

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **kw: real_client(transport=httpx.MockTransport(unreachable)),
    )
    monkeypatch.setitem(cli.JOBS, "snapshot tick", lambda c: None)
    settings = replace(make_settings(tmp_path), healthcheck_ping_url="https://hc.test/abc")
    assert cli.main(["snapshot", "tick"], settings=settings) == 0


def test_settings_heartbeat_url_is_stripped_and_optional():
    url = "https://hc-ping.com/0b4c1f7e-9a2d-4e4b-8f3a-5d6c7b8a9e01"
    assert Settings.from_env({"HEALTHCHECK_PING_URL": f" {url}\n"}).healthcheck_ping_url == url
    assert Settings.from_env({"HEALTHCHECK_PING_URL": " "}).healthcheck_ping_url is None
    assert Settings.from_env({}).healthcheck_ping_url is None
