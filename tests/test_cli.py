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


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("greedy", "expected name:xp"),
        ("greedy:rolling:threshold=1:x", "expected name:xp"),
        (":rolling", "expected name:xp"),
        ("optimizer:rolling", "unknown policy 'optimizer'"),
        ("greedy:magic", "unknown xP model 'magic'"),
        ("greedy:rolling:foo=1", "no parameter 'foo'"),
        ("roll:rolling:threshold=1", "no parameter 'threshold'"),
        ("greedy:rolling:threshold", "bad policy parameter"),
        ("greedy:rolling:threshold=x", "not a valid float"),
        ("greedy:rolling:horizon=2.5", "not a valid int"),
        ("greedy:rolling:threshold=nan", "not finite"),
        ("greedy:rolling:horizon=0", "horizon >= 1"),
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
        (["run", "--seasons", "2023", "--horizon", "0"], "horizon >= 1"),
        (["compare", "--seasons", "2023", "--a", "roll:rolling", "--b", "roll:rolling"], "same"),
        (["compare", "--seasons", "2023", *ROLL_VS_GREEDY, "--k", "0"], "must be >= 1"),
        (["compare", "--seasons", "2023", *ROLL_VS_GREEDY, "--stride", "0"], "must be >= 1"),
        (["compare", "--seasons", "2023", "--a", "greedy:x", "--b", "roll:rolling"], "unknown xP"),
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
    methods = {(c["method"], c["metric"]) for c in comparisons}
    assert methods == {
        (m, x) for m in ("full_run", "per_decision") for x in ("realized", "realized@xg", "xg")
    }
    (full,) = [c for c in comparisons if (c["method"], c["metric"]) == ("full_run", "realized")]
    cells = paired.groupby(["season", "gw_index"])["diff"].mean()
    assert full["mean"] == pytest.approx(cells.mean())
    printed = capsys.readouterr().out
    for text in ("Paired differences A - B", "per-decision (k=2, stride=2)", "full run"):
        assert text in printed
    for text in ("Per season", "80% CI", "realized@xg", "per-decision total (k=2)"):
        assert text in printed
    assert "2022-23" in printed and "2023-24" in printed
    (logged,) = _experiments(log)
    assert logged["n_variants"] == "2"
    config = json.loads(logged["config"])
    assert config["policies"] == ["greedy:ep_next", "greedy:rolling"]
    assert config["continuation"] == "roll:rolling" and config["k"] == 2
    assert config["stride"] == 2 and config["ci"] == 0.8
    assert len(json.loads(logged["metrics"])["comparisons"]) == 6


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
