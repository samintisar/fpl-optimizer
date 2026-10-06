import logging
import signal
from dataclasses import replace
from pathlib import Path

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


def test_every_parsed_command_has_a_job():
    parser = cli.build_parser()
    extras = {"rules export": ["2026-27"], "build": ["fixture"]}
    for name in cli.JOBS:
        args = parser.parse_args(name.split() + extras.get(name, []))
        assert cli.job_name(args) == name


def test_build_dispatches_target_with_data_dir(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        cli, "build_tables", lambda names, ctx: seen.append((names, ctx.store.root, ctx.data_dir))
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
