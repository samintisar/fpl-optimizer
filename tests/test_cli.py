import argparse
import csv
import json
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
    extras = {
        "rules export": ["2026-27"],
        "build": ["fixture"],
        "backtest run": ["--seasons", "2023"],
        "backtest compare": ["--seasons", "2023", "--a", "greedy:rolling", "--b", "roll:rolling"],
        "optimize plan": ["--season", "2023", "--gw", "5"],
        "models eval": ["--models", "rolling", "--seasons", "2023"],
    }
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

    code = (
        "import sys, fplopt.cli; "
        "print('pandas' in sys.modules, any(m.startswith('fplopt.backtest') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


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
    from fplopt.features import leakcheck

    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name, df in built.items():
        df.to_parquet(data_dir / f"{name}.parquet")
    settings = replace(make_settings(tmp_path), data_dir=data_dir)
    with caplog.at_level(logging.INFO):
        assert cli.main(["check", "leakage", "--deadlines", "3"], settings=settings) == 0
    n = len(leakcheck.sample_deadlines(built, 3))  # the edges + 3 random deadlines
    assert n > 3 and f"leakage check passed: {n} deadline(s)" in caplog.text
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


# --- backtest run / compare -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seasons"),
    [
        ("2016-2024", tuple(range(2016, 2025))),
        ("2021,2023", (2021, 2023)),
        ("2023, 2021,2021", (2021, 2023)),
        ("2016-2018,2021-22,2026", (2016, 2017, 2018, 2021, 2026)),
        ("2019-20", (2019,)),
        ("2024", (2024,)),
    ],
)
def test_parse_seasons(text, seasons):
    assert cli.parse_seasons(text) == seasons


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("2016-2026", "2025-26 is the holdout season"),
        ("2025", "holdout"),
        ("2025-26", "holdout"),
        ("2015", "no backtest data for 2015-16"),
        ("2024-2027", "no backtest data for 2027-28"),
        ("2020-2018", "end before start"),
        ("2019-21", "not a season label"),
        ("16", "bad season"),
        ("2021;2022", "bad season"),
        ("", "bad season"),
    ],
)
def test_parse_seasons_rejects(text, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        cli.parse_seasons(text)


def test_format_seasons():
    assert cli.format_seasons((2016, 2017, 2018, 2021, 2023, 2024)) == "2016-2018,2021,2023-2024"


def test_parse_policy_spec():
    spec = cli.parse_policy_spec("greedy:rolling")
    assert spec == cli.PolicySpec("greedy", "rolling")
    assert str(spec) == "greedy:rolling"
    spec = cli.parse_policy_spec(" greedy : ep_next : threshold=2, max_transfers=2,decay=0.9 ")
    assert spec.params == (("decay", 0.9), ("max_transfers", 2), ("threshold", 2.0))
    assert str(spec) == "greedy:ep_next:decay=0.9,max_transfers=2,threshold=2.0"
    assert cli.parse_policy_spec(str(spec)) == spec
    assert spec.build().name == "greedy(ep_next,t=2.0,d=0.9,n=2)"
    assert cli.parse_policy_spec("roll:rolling").build().name == "roll(rolling)"


def test_parse_optimizer_policy_spec():
    spec = cli.parse_policy_spec("optimizer:ep_next_fade:max_hits=0,hit_margin=2,chips=1")
    assert spec.params == (("chips", 1), ("hit_margin", 2.0), ("max_hits", 0))
    assert str(spec) == "optimizer:ep_next_fade:chips=1,hit_margin=2.0,max_hits=0"
    assert cli.parse_policy_spec(str(spec)) == spec
    policy = spec.build()
    assert policy.name == "optimizer(ep_next_fade,mh=0,m=2.0,chips)"
    assert policy.chips and policy.params.max_hits == 0 and policy.params.hit_margin == 2.0
    unlimited = cli.parse_policy_spec("optimizer:rolling:max_hits=none,horizon=4,decay=0.9")
    assert unlimited.params == (("decay", 0.9), ("horizon", 4), ("max_hits", None))
    assert cli.parse_policy_spec(str(unlimited)) == unlimited
    assert unlimited.build().name == "optimizer(rolling,mh=inf,m=0.0,h=4,d=0.9)"
    itb = cli.parse_policy_spec("optimizer:rolling:itb_value=0.08,chips=0").build()
    assert itb.name == "optimizer(rolling,mh=0,m=0.0,itb=0.08)" and not itb.chips


def test_parse_optimizer_minutes_keys():
    """Phase 5b Task 8: the minutes-based bench weights can be switched off and the
    expected-minutes floor set from a spec; the defaults keep the name unchanged."""
    spec = cli.parse_policy_spec("optimizer:v1:bench_from_minutes=0,min_minutes=45")
    assert spec.params == (("bench_from_minutes", 0), ("min_minutes", 45.0))
    assert cli.parse_policy_spec(str(spec)) == spec
    policy = spec.build()
    assert policy.params.bench_from_minutes is False and policy.params.min_minutes == 45.0
    assert policy.name == "optimizer(v1,mh=0,m=0.0,bfm=0,minm=45.0)"
    default = cli.parse_policy_spec("optimizer:v1").build()
    assert default.params.bench_from_minutes is True
    assert default.name == "optimizer(v1,mh=0,m=0.0)"
    on = cli.parse_policy_spec("optimizer:v1:bench_from_minutes=1").build()
    assert on.name == default.name
    with pytest.raises(argparse.ArgumentTypeError, match="not a valid 0/1 flag"):
        cli.parse_policy_spec("optimizer:v1:bench_from_minutes=2")
    with pytest.raises(argparse.ArgumentTypeError, match="min_minutes must be"):
        cli.parse_policy_spec("optimizer:v1:min_minutes=-1")
    with pytest.raises(argparse.ArgumentTypeError, match="no parameter 'bench_from_minutes'"):
        cli.parse_policy_spec("greedy:v1:bench_from_minutes=0")
    # On a model without minutes the keys would change nothing but the variant count.
    for text in ("optimizer:rolling:min_minutes=90", "optimizer:ep_next:bench_from_minutes=0"):
        with pytest.raises(argparse.ArgumentTypeError, match="only apply to xP models"):
            cli.parse_policy_spec(text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("greedy", "expected name:xp"),
        ("greedy:rolling:threshold=1:x", "expected name:xp"),
        (":rolling", "expected name:xp"),
        ("best:rolling", "unknown policy 'best'"),
        ("optimizer:rolling:threshold=1", "no parameter 'threshold'"),
        ("optimizer:rolling:max_hits=-1", "not a valid int or none"),
        ("optimizer:rolling:max_hits=x", "not a valid int or none"),
        ("optimizer:rolling:chips=2", "not a valid 0/1 flag"),
        ("optimizer:rolling:decay=0", "0 < decay <= 1"),
        ("optimizer:rolling:horizon=0", "1 <= horizon <= 6"),
        ("optimizer:rolling:horizon=7", "1 <= horizon <= 6"),
        ("optimizer:rolling:hit_margin=-1", "hit_margin must be a finite number >= 0"),
        ("optimizer:rolling:itb_value=-0.5", "itb_value must be a finite number >= 0"),
        ("greedy:rolling:threshold=none", "not a valid float"),
        ("greedy:magic", "unknown xP model 'magic'"),
        ("greedy:rolling:foo=1", "no parameter 'foo'"),
        ("roll:rolling:threshold=1", "no parameter 'threshold'"),
        ("greedy:rolling:threshold", "bad policy parameter"),
        ("greedy:rolling:threshold=x", "not a valid float"),
        ("greedy:rolling:horizon=2.5", "not a valid int"),
        ("greedy:rolling:threshold=nan", "not finite"),
        ("greedy:rolling:horizon=0", "1 <= horizon <= 6"),
        ("greedy:rolling:horizon=8", "1 <= horizon <= 6"),
        ("greedy:rolling:max_transfers=-1", "max_transfers >= 0"),
        ("greedy:rolling:threshold=1,threshold=2", "duplicate policy parameter"),
    ],
)
def test_parse_policy_spec_rejects(text, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        cli.parse_policy_spec(text)


def test_cli_policy_and_model_names_match_the_backtester():
    from dataclasses import fields

    from fplopt.backtest.policies import GreedyPolicy
    from fplopt.models import MODELS

    assert set(cli.XP_MODELS) == set(MODELS)
    greedy = {f.name for f in fields(GreedyPolicy)} - {"xp_model"}
    assert set(cli.POLICY_PARAMS["greedy"]) == greedy
    from fplopt.optimize import OptimizerParams

    optimizer = set(cli.POLICY_PARAMS["optimizer"]) - {"chips"}
    assert optimizer <= {f.name for f in fields(OptimizerParams)}
    assert cli.EP_NEXT_MODELS <= set(MODELS)
    from fplopt.models import MAX_HORIZON

    assert cli.MAX_PLAN_HORIZON == MAX_HORIZON


@pytest.fixture(scope="module")
def league():
    """A small synthetic league (10 clubs, 18 GWs) for 2022/23 and 2023/24, with snapshots."""
    from synthetic_season import synthetic_tables

    from fplopt.features.store import DataStore

    return DataStore(tables=synthetic_tables(seasons=(2022, 2023), n_clubs=10))


@pytest.fixture
def backtest(tmp_path, monkeypatch, league):
    """Runs `fplopt backtest ...` on the synthetic league; returns (exit code, alerts)."""
    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    monkeypatch.setattr(cli, "open_data_store", lambda data_dir: league)

    def run(*argv):
        return cli.main(["backtest", *argv], settings=make_settings(tmp_path)), alerts

    return run


ROLL_VS_GREEDY = ["--a", "roll:rolling", "--b", "greedy:rolling"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--seasons", "2016-2026"], "2025-26 is the holdout season"),
        (["compare", "--seasons", "2025", *ROLL_VS_GREEDY], "holdout"),
        (["run", "--seasons", "2020-2023", "--xp", "ep_next"], "ep_next, which exists only from"),
        (
            ["compare", "--seasons", "2019,2023", "--a", "greedy:ep_next", "--b", "roll:rolling"],
            "drop 2019-20",
        ),
        (
            ["compare", "--seasons", "2020", *ROLL_VS_GREEDY, "--continuation", "roll:ep_next"],
            "roll:ep_next uses ep_next",
        ),
        (["run", "--seasons", "2021"], "no gameweek/player_match data for 2021-22"),
        (["run", "--seasons", "2023", "--starts", "template"], "--starts: bad start spec"),
        (["run", "--seasons", "2023", "--policy", "roll", "--horizon", "3"], "only for --policy"),
        (["run", "--seasons", "2023", "--horizon", "0"], "1 <= horizon <= 6"),
        (["run", "--seasons", "2023", "--horizon", "7"], "1 <= horizon <= 6"),
        (["compare", "--seasons", "2023", "--a", "roll:rolling", "--b", "roll:rolling"], "same"),
        (  # a default spelled out builds the same policy (same name)
            ["compare", "--seasons", "2023", "--a", "greedy:rolling"]
            + ["--b", "greedy:rolling:threshold=1.0"],
            "same policy",
        ),
        (["compare", "--seasons", "2023", *ROLL_VS_GREEDY, "--k", "0"], "must be >= 1"),
        (["compare", "--seasons", "2023", *ROLL_VS_GREEDY, "--stride", "0"], "must be >= 1"),
        (["compare", "--seasons", "2023", "--a", "greedy:x", "--b", "roll:rolling"], "unknown xP"),
        (
            ["compare", "--seasons", "2020,2023", "--a", "optimizer:ep_next_fade"]
            + ["--b", "greedy:rolling"],
            "optimizer:ep_next_fade uses ep_next",
        ),
        (["run", "--seasons", "2023", "--jobs", "0"], "--jobs must be >= 1"),
        (["run", "--seasons", "2023", "--spec", "roll:rolling", "--xp", "rolling"], "--spec"),
    ],
)
def test_backtest_usage_errors_exit_two_without_alert(backtest, capsys, tmp_path, argv, message):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as info:
        backtest(*argv, "--out", str(out))
    assert info.value.code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def _experiments(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_backtest_run_end_to_end(backtest, tmp_path, capsys):
    import pandas as pd

    out, log = tmp_path / "out", tmp_path / "results" / "experiments.csv"
    argv = ["run", "--seasons", "2023", "--starts", "template@1,random:2@10"]
    argv += ["--threshold", "0.5", "--out", str(out), "--experiments", str(log)]
    code, alerts = backtest(*argv)
    assert code == 0 and alerts == []
    gws = pd.read_parquet(out / "gws.parquet")
    assert set(gws["start_id"]) == {"template@1", "random0@10", "random1@10"}
    assert set(gws["policy"]) == {"greedy(rolling,t=0.5)"}
    assert len(gws) == 18 + 2 * 9
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["policies"] == ["greedy:rolling:threshold=0.5"]
    assert summary["starts"] == {"2023-24": ["random0@10", "random1@10", "template@1"]}
    (row,) = summary["summary"]["season_totals"]
    totals = gws.groupby("start_id")["net_points"].sum()
    assert row["total"] == pytest.approx(totals.mean())
    # The printed/JSON season table keeps full-season and mid-season starts apart.
    gw1, gw10 = summary["season_totals_by_start"]
    assert (gw1["start_gw"], gw1["starts"], gw1["template"]) == (1, 1, totals["template@1"])
    assert (gw10["start_gw"], gw10["starts"], gw10["template"]) == (10, 2, None)
    assert gw10["mean"] == pytest.approx(totals[["random0@10", "random1@10"]].mean())
    assert gw10["min"] == totals[["random0@10", "random1@10"]].min()
    printed = capsys.readouterr().out
    assert "Season totals" in printed and "2023-24" in printed
    assert "greedy(rolling,t=0.5)" in printed
    assert f" {int(totals.min())} " in printed and f" {int(totals.max())} " in printed
    (logged,) = _experiments(log)
    assert logged["n_variants"] == "1"
    assert logged["command"].startswith("fplopt backtest run --seasons 2023")
    assert json.loads(logged["config"])["seasons"] == [2023]
    metrics = json.loads(logged["metrics"])
    season_total = metrics["season_totals"]["2023-24 greedy(rolling,t=0.5)"]
    assert season_total == pytest.approx(totals.mean(), abs=0.01)
    # A second run appends to the log.
    argv = ["run", "--seasons", "2023", "--policy", "roll", "--starts", "random@15"]
    assert backtest(*argv, "--out", str(out), "--experiments", str(log))[0] == 0
    assert [r["n_variants"] for r in _experiments(log)] == ["1", "1"]


def test_backtest_run_default_out_dir(backtest, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, _ = backtest("run", "--seasons", "2023", "--policy", "roll", "--starts", "random@16")
    assert code == 0
    (run_dir,) = (tmp_path / "results").glob("*-run")
    assert (run_dir / "gws.parquet").exists() and (run_dir / "summary.json").exists()
    assert len(_experiments(tmp_path / "results" / "experiments.csv")) == 1


def test_backtest_compare_end_to_end(backtest, tmp_path, capsys):
    import pandas as pd

    out, log = tmp_path / "out", tmp_path / "experiments.csv"
    argv = ["compare", "--seasons", "2022-23,2023", "--a", "greedy:ep_next", "--b"]
    argv += ["greedy:rolling", "--starts", "random:2@12", "--k", "2", "--n-boot", "50"]
    code, alerts = backtest(*argv, "--out", str(out), "--experiments", str(log))
    assert code == 0 and alerts == []
    gws = pd.read_parquet(out / "gws.parquet")
    assert set(gws["policy"]) == {"greedy(ep_next,t=1.0)", "greedy(rolling,t=1.0)"}
    assert set(gws["season"]) == {2022, 2023}
    paired = pd.read_parquet(out / "paired.parquet")
    decisions = pd.read_parquet(out / "per_decision.parquet")
    assert len(paired) == 2 * 2 * 7  # seasons x starts x GWs 12-18
    # Non-overlapping k = 2 windows on the grid gw_index 1, 3, 5, ...: 13, 15, 17.
    assert len(decisions) == 2 * 2 * 3
    assert sorted(set(decisions["gw_index"])) == [13, 15, 17]
    assert decisions["k"].tolist() == [2, 2, 2] * 4 and (decisions["stride"] == 2).all()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["policy_a"] == "greedy(ep_next,t=1.0)"
    assert summary["continuation"] == "roll(rolling)"
    comparisons = summary["summary"]["comparisons"]
    methods = {(c["method"], c["split"], c["metric"]) for c in comparisons}
    assert methods == {
        (m, split, x)
        for m in ("full_run", "per_decision")
        for split in ("all", "develop", "validate")  # 2022-23 develop, 2023-24 validate
        for x in ("realized", "realized@xg", "xg")
    }
    (full,) = [
        c
        for c in comparisons
        if (c["method"], c["split"], c["metric"]) == ("full_run", "all", "realized")
    ]
    cells = paired.groupby(["season", "gw_index"])["diff"].mean()
    assert full["mean"] == pytest.approx(cells.mean())
    assert full["deflated_mean"] == full["mean"]  # the family's first variant
    assert 0 <= full["p_season_t"] <= 1  # two seasons
    printed = capsys.readouterr().out
    for text in ("Paired differences A - B", "per-decision (k=2, stride=2)", "full run"):
        assert text in printed
    for text in ("Per season", "80% CI", "realized@xg", "per-decision total (k=2)"):
        assert text in printed
    for text in ("season-t p", "deflated", "develop", "validate", "reference greedy(rolling"):
        assert text in printed
    assert "2022-23" in printed and "2023-24" in printed
    (logged,) = _experiments(log)
    assert logged["n_variants"] == "1" and logged["family"] == "greedy:rolling 2022-2023"
    config = json.loads(logged["config"])
    assert config["policies"] == ["greedy:ep_next", "greedy:rolling"]
    assert config["continuation"] == "roll:rolling" and config["k"] == 2
    assert config["stride"] == 2 and config["ci"] == 0.8 and config["reference"] == "b"
    assert len(json.loads(logged["metrics"])["comparisons"]) == 18


def test_backtest_compare_own_continuation_reference_and_family(backtest, tmp_path, capsys):
    """--continuation own --reference a: each arm continues with its own policy from A's
    states; the experiment family accumulates variants and deflates the mean."""
    import pandas as pd

    out, log = tmp_path / "out", tmp_path / "experiments.csv"
    argv = ["compare", "--seasons", "2023", "--a", "greedy:rolling:threshold=0.5", "--b"]
    argv += ["roll:rolling", "--starts", "random@13", "--k", "3", "--n-boot", "20"]
    argv += ["--continuation", "own", "--reference", "a", "--jobs", "1"]
    assert backtest(*argv, "--out", str(out), "--experiments", str(log))[0] == 0
    decisions = pd.read_parquet(out / "per_decision.parquet")
    assert (decisions["continuation"] == "own").all() and (decisions["reference"] == "a").all()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["continuation"] == "own" and summary["reference"] == "a"
    assert "continuation own, reference greedy(rolling,t=0.5)" in capsys.readouterr().out
    # Two more variants against the same B on the same seasons: N = 2, 3, deflated means.
    argv[4] = "greedy:rolling:threshold=1.5"
    assert backtest(*argv, "--out", str(out), "--experiments", str(log))[0] == 0
    argv[4] = "greedy:rolling:threshold=2.5"
    assert backtest(*argv, "--out", str(out), "--experiments", str(log), "--family", "mine")[0] == 0
    rows = _experiments(log)
    assert [r["family"] for r in rows] == ["roll:rolling 2023", "roll:rolling 2023", "mine"]
    assert [r["n_variants"] for r in rows] == ["1", "2", "1"]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_variants"] == 1 and summary["family"] == "mine"


def test_backtest_compare_stride_one_evaluates_every_gw(backtest, tmp_path):
    import pandas as pd

    out, log = tmp_path / "out", tmp_path / "experiments.csv"
    argv = ["compare", "--seasons", "2023", *ROLL_VS_GREEDY, "--starts", "random@12"]
    argv += ["--k", "3", "--stride", "1", "--n-boot", "20"]
    assert backtest(*argv, "--out", str(out), "--experiments", str(log))[0] == 0
    decisions = pd.read_parquet(out / "per_decision.parquet")
    assert decisions["gw_index"].tolist() == list(range(12, 19))  # the league has 18 GWs
    assert decisions["k"].tolist() == [3] * 5 + [2, 1]
    assert json.loads(_experiments(log)[0]["config"])["stride"] == 1


def test_backtest_compare_without_per_decision(backtest, tmp_path):
    out, log = tmp_path / "out", tmp_path / "experiments.csv"
    argv = ["compare", "--seasons", "2023", "--a", "greedy:rolling:threshold=0.5", "--b"]
    argv += ["roll:rolling", "--starts", "random@14", "--no-per-decision", "--n-boot", "20"]
    assert backtest(*argv, "--out", str(out), "--experiments", str(log))[0] == 0
    assert not (out / "per_decision.parquet").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {c["method"] for c in summary["summary"]["comparisons"]} == {"full_run"}
    assert summary["continuation"] is None
    assert json.loads(_experiments(log)[0]["config"])["continuation"] is None


def test_backtest_job_failure_is_alerted(backtest, monkeypatch, tmp_path):
    from fplopt.backtest import evaluate

    def boom(*args, **kwargs):
        raise RuntimeError("simulator broke")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "run_grid", boom)
    code, alerts = backtest("run", "--seasons", "2023", "--starts", "random@15")
    assert code == 1 and "backtest run failed" in alerts[0] and "simulator broke" in alerts[0]
    assert not (tmp_path / "results").exists()  # no empty output dir, no log row


def test_start_gw_of_keeps_fallback_starts_with_their_spec():
    from fplopt.backtest.evaluate import parse_start_specs

    groups = cli.start_gw_of(parse_start_specs("template@1,random:2@20"))
    assert groups == {
        "template@1": 1,
        "template@2": 1,
        "random0@20": 20,
        "random1@20": 20,
    }


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("highspy") is None, reason="needs the optimize extra"
)
def test_optimize_bench_end_to_end(tmp_path, monkeypatch, league, capsys):
    """`fplopt optimize bench` on the synthetic league: one deadline × template/random ×
    both xP models, the exhaustive chip search on the first case, the pruning variants."""
    import pandas as pd

    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    monkeypatch.setattr(cli, "open_data_store", lambda data_dir: league)
    out = tmp_path / "bench"
    argv = ["optimize", "bench", "--deadlines", "1", "--horizon", "2", "--all-chips", "1"]
    argv += ["--prune-n", "5,10,10,6"]  # small pools: synthetic instances are hard for HiGHS
    assert cli.main([*argv, "--out", str(out)], settings=make_settings(tmp_path)) == 0
    assert alerts == []
    cases = pd.read_csv(out / "cases.csv")
    assert len(cases) == 4
    assert set(cases["xp"]) == {"ep_next", "rolling"}
    assert set(cases["season"]) <= {2022, 2023}
    assert (cases["n_solves"] <= cases["n_scenarios"]).all()
    assert (cases["chips_objective"] >= cases["objective"] - 1e-6).all()
    assert cases["bound_matches_all"].iloc[0] and cases["bound_matches_all"].iloc[1:].isna().all()
    prune = pd.read_csv(out / "prune.csv")
    assert len(prune) == 4 * 5 and (prune["loss"] >= 0).all()
    assert cases["chips_prune_loss"].iloc[0] >= -1e-6
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["n_cases"] == 4
    assert summary["summary"]["chips_all"]["matches"] == 1
    assert summary["config"]["params"]["horizon"] == 2
    printed = capsys.readouterr().out
    assert "chips, bound search" in printed and "Pruning" in printed and "default" in printed
    # --prune-n sets the planner's pool; bad values are usage errors.
    argv = ["optimize", "bench", "--deadlines", "1", "--horizon", "1", "--no-prune-study"]
    out2 = tmp_path / "bench2"
    code = cli.main(
        [*argv, "--prune-n", "3,6,6,4", "--out", str(out2)], settings=make_settings(tmp_path)
    )
    assert code == 0
    config = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))["config"]
    assert config["params"]["prune_n"] == {"1": 3, "2": 6, "3": 6, "4": 4}
    assert not (out2 / "prune.csv").read_text(encoding="utf-8").strip()
    with pytest.raises(SystemExit):
        cli.main([*argv, "--prune-n", "3,6,6"], settings=make_settings(tmp_path))


HAS_HIGHS = __import__("importlib").util.find_spec("highspy") is not None


@pytest.mark.skipif(not HAS_HIGHS, reason="needs the optimize extra")
def test_backtest_compare_optimizer_with_jobs(backtest, tmp_path, capsys):
    """An optimizer spec against greedy in two worker processes: runs end to end, prints
    the transfer-gain table and logs --jobs."""
    import pandas as pd

    out, log = tmp_path / "out", tmp_path / "experiments.csv"
    spec = "optimizer:rolling:horizon=2,max_hits=0"
    argv = ["compare", "--seasons", "2023", "--a", spec, "--b", "greedy:rolling"]
    argv += ["--starts", "random:2@15", "--k", "2", "--n-boot", "20", "--jobs", "2"]
    code, alerts = backtest(*argv, "--out", str(out), "--experiments", str(log))
    assert code == 0 and alerts == []
    gws = pd.read_parquet(out / "gws.parquet")
    name = "optimizer(rolling,mh=0,m=0.0,h=2)"
    assert set(gws["policy"]) == {name, "greedy(rolling,t=1.0)"}
    assert (gws.loc[gws["policy"] == name, "hits"] == 0).all()
    assert {"pred_gain", "real_gain"} <= set(gws.columns)
    printed = capsys.readouterr().out
    assert "Predicted vs realized transfer gain" in printed and name in printed
    (logged,) = _experiments(log)
    assert json.loads(logged["config"])["jobs"] == 2
    assert "transfer_gains" in json.loads(logged["metrics"])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {r["policy"] for r in summary["summary"]["transfer_gains"]} == set(gws["policy"])


def test_backtest_run_with_a_spec(backtest, tmp_path):
    import pandas as pd

    out, log = tmp_path / "out", tmp_path / "experiments.csv"
    argv = ["run", "--seasons", "2023", "--spec", "greedy:ep_next_fade:threshold=0.5"]
    argv += ["--starts", "random@16", "--jobs", "1"]
    assert backtest(*argv, "--out", str(out), "--experiments", str(log))[0] == 0
    gws = pd.read_parquet(out / "gws.parquet")
    assert set(gws["policy"]) == {"greedy(ep_next_fade,t=0.5)"}
    assert json.loads(_experiments(log)[0]["config"])["policies"] == [
        "greedy:ep_next_fade:threshold=0.5"
    ]


@pytest.fixture
def plan(tmp_path, monkeypatch, league):
    """Runs `fplopt optimize plan ...` on the synthetic league; returns (exit code, alerts)."""
    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    monkeypatch.setattr(cli, "open_data_store", lambda data_dir: league)

    def run(*argv):
        return cli.main(["optimize", "plan", *argv], settings=make_settings(tmp_path)), alerts

    run.alerts = alerts
    return run


@pytest.mark.skipif(not HAS_HIGHS, reason="needs the optimize extra")
def test_optimize_plan_prints_the_top_plans_and_the_roll_plan(plan, capsys):
    argv = ["--season", "2023-24", "--gw", "6", "--start", "random:1", "--xp", "rolling"]
    code, alerts = plan(*argv, "--no-chips", "--horizon", "2", "--max-hits", "1")
    assert code == 0 and alerts == []
    printed = capsys.readouterr().out
    assert "2023-24 GW6 (gw_index 6), random:1 squad, xP rolling, horizon 2 GW(s)" in printed
    assert "no chips, max_hits 1" in printed
    for label in ("Plan #1: gain vs roll", "Plan #2", "Plan #3", "Roll plan: gain vs roll +0.00"):
        assert label in printed
    assert "transfers (out -> in)" in printed and "captain (vice)" in printed
    assert " -> " in printed  # the top plans transfer, by player name
    code, _ = plan("--season", "2023", "--gw", "7", "--xp", "ep_next", "--horizon", "1")
    assert code == 0
    printed = capsys.readouterr().out
    assert "template squad" in printed and "Chip scenarios solved" in printed


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--season", "2025", "--gw", "3"], "holdout"),
        (["--season", "2021-2023", "--gw", "3"], "give one season"),
        (["--season", "2020", "--gw", "3"], "--xp ep_next uses ep_next"),
        (["--season", "2023", "--gw", "3", "--max-hits", "x"], "--max-hits"),
        (["--season", "2023", "--gw", "3", "--start", "best"], "template or random:SEED"),
        (["--season", "2023", "--gw", "3", "--horizon", "0"], "--horizon must be in 1..6"),
        (["--season", "2023", "--gw", "3", "--horizon", "7"], "--horizon must be in 1..6"),
        (["--season", "2023", "--gw", "3", "--top-k", "0"], "--top-k must be >= 1"),
        (["--season", "2023", "--gw", "3", "--hit-margin", "-1"], "hit_margin must be"),
        (["--season", "2023", "--gw", "40", "--xp", "rolling"], "--gw: 2023-24 has no GW40"),
    ],
)
def test_optimize_plan_usage_errors(plan, capsys, argv, message):
    with pytest.raises(SystemExit) as info:
        plan(*argv)
    assert info.value.code == 2
    assert message in capsys.readouterr().err
    assert plan.alerts == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--horizon", "7"], "--horizon must be in 1..6"),
        (["--horizon", "0"], "--horizon must be in 1..6"),
        (["--deadlines", "0"], "--deadlines must be >= 1"),
        (["--all-chips", "-1"], "--all-chips >= 0"),
    ],
)
def test_optimize_bench_usage_errors(tmp_path, monkeypatch, league, capsys, argv, message):
    monkeypatch.setattr(cli, "open_data_store", lambda data_dir: league)
    with pytest.raises(SystemExit) as info:
        cli.main(["optimize", "bench", *argv], settings=make_settings(tmp_path))
    assert info.value.code == 2
    assert message in capsys.readouterr().err


# --- models eval ------------------------------------------------------------------------------


@pytest.fixture
def models_eval(tmp_path, monkeypatch, league):
    """Runs `fplopt models eval ...` on the synthetic league; returns (exit code, alerts)."""
    alerts = []
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append(text))
    monkeypatch.setattr(cli, "open_data_store", lambda data_dir: league)

    def run(*argv):
        return cli.main(["models", "eval", *argv], settings=make_settings(tmp_path)), alerts

    return run


def test_parse_models():
    assert cli.parse_models("rolling, ep_next") == ("rolling", "ep_next")
    for text, message in (("rolling,v0", "unknown xP model"), ("rolling,rolling", "repeated")):
        with pytest.raises(argparse.ArgumentTypeError, match=message):
            cli.parse_models(text)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--models", "rolling", "--seasons", "2024-2025"], "2025-26 is the holdout season"),
        (["--models", "rolling,nope", "--seasons", "2023"], "unknown xP model(s) nope"),
        (["--models", "rolling,rolling", "--seasons", "2023"], "repeated xP model"),
        (["--models", "ep_next", "--seasons", "2019-2020"], "ep_next, which exists only from"),
        (["--models", "rolling", "--seasons", "2021"], "no gameweek/player_match data for 2021"),
        (["--models", "rolling", "--seasons", "2023", "--jobs", "0"], "--jobs must be >= 1"),
        (["--models", "rolling", "--seasons", "2023", "--n-random", "-1"], "--n-random"),
        (["--seasons", "2023"], "--models"),
    ],
)
def test_models_eval_usage_errors(models_eval, capsys, tmp_path, argv, message):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as info:
        models_eval(*argv, "--out", str(out))
    assert info.value.code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_models_eval_end_to_end(models_eval, tmp_path, capsys):
    import pandas as pd

    out, log = tmp_path / "out", tmp_path / "results" / "experiments.csv"
    argv = ["--models", "rolling,ep_next", "--seasons", "2023", "--n-random", "1", "--jobs", "1"]
    code, alerts = models_eval(*argv, "--out", str(out), "--experiments", str(log))
    assert code == 0 and alerts == []
    predictions = pd.read_parquet(out / "predictions.parquet")
    assert set(predictions["model"]) == {"rolling", "ep_next"}
    assert set(predictions["season"]) == {2023} and predictions["deadline"].nunique() == 18
    regrets = pd.read_parquet(out / "regrets.parquet")
    assert set(regrets["squad"]) == {"template", "random0"}
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert payload["config"]["models"] == ["rolling", "ep_next"]
    assert payload["family"] == "eval rolling,ep_next 2023" and payload["n_variants"] == 1
    metrics = payload["metrics"]
    assert metrics["seasons"] == {"rolling": [2023], "ep_next": [2023]}
    h0 = next(
        r
        for r in metrics["xp"]
        if (r["model"], r["split"], r["horizon"]) == ("ep_next", "all", "0")
    )
    rows = predictions[(predictions["model"] == "ep_next") & (predictions["horizon"] == 0)]
    assert h0["mse"] == pytest.approx(float(((rows["xp"] - rows["points"]) ** 2).mean()))
    printed = capsys.readouterr().out
    assert "MSE h0" in printed and "Diebold-Mariano" in printed and "validate" in printed
    assert f"{h0['mse']:.3f}" in printed
    (logged,) = _experiments(log)
    assert logged["family"] == "eval rolling,ep_next 2023" and logged["n_variants"] == "1"
    assert logged["command"].startswith("fplopt models eval --models rolling,ep_next")
    assert set(json.loads(logged["metrics"])) == {"seasons", "xp", "regret", "dm"}
    # The same family again counts as a second variant.
    assert models_eval(*argv, "--out", str(out), "--experiments", str(log))[0] == 0
    assert [r["n_variants"] for r in _experiments(log)] == ["1", "2"]


def test_models_eval_default_out_dir(models_eval, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, _ = models_eval("--models", "rolling", "--seasons", "2022", "--n-random", "0")
    assert code == 0
    (run_dir,) = (tmp_path / "results").glob("*-eval")
    assert (run_dir / "predictions.parquet").exists() and (run_dir / "metrics.json").exists()
