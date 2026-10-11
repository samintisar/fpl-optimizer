"""Guards on the systemd unit files in deploy/systemd (they have no other tests)."""

import re
from pathlib import Path

import pytest

UNITS = Path(__file__).resolve().parents[1] / "deploy" / "systemd"


def service_settings(name: str) -> list[tuple[str, str]]:
    """(key, value) pairs of a unit's [Service] section, in order (keys may repeat)."""
    section = None
    pairs = []
    for line in (UNITS / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            section = line.strip("[]")
        elif section == "Service":
            key, _, value = line.partition("=")
            pairs.append((key.strip(), value.strip()))
    return pairs


JOB_UNITS = [
    "fplopt-daily.service",
    "fplopt-tick.service",
    "fplopt-freshness.service",
    "fplopt-live.service",
    "fplopt-bot.service",
]


@pytest.mark.parametrize("unit", JOB_UNITS)
def test_job_units_wait_for_dns_before_starting(unit):
    # After=network-online.target is a no-op in user units, so a Persistent timer firing at
    # boot could otherwise start before DNS works.
    (pre,) = [value for key, value in service_settings(unit) if key == "ExecStartPre"]
    assert pre.startswith("/bin/sh -c '")
    assert "getent hosts fantasy.premierleague.com" in pre
    # systemd expands $VAR and %x specifiers itself; the shell needs them escaped as $$ / %%.
    assert "$" not in pre.replace("$$", "")
    assert not re.search(r"%[^%]", pre.replace("%%", ""))
    assert pre.rstrip("'").endswith("exit 0")  # never blocks the job: it fails and alerts itself


def unit_settings(name: str, section: str) -> dict[str, str]:
    """A unit file section's settings (the last value of a repeated key)."""
    current = None
    out = {}
    for line in (UNITS / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            current = line.strip("[]")
        elif current == section:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


@pytest.mark.parametrize("unit", JOB_UNITS)
def test_job_units_alert_on_failure_and_run_from_the_checkout(unit):
    assert unit_settings(unit, "Unit")["OnFailure"] == "fplopt-failure@%n.service"
    service = unit_settings(unit, "Service")
    assert service["WorkingDirectory"] == "%h/fpl-optimizer"  # the CLI reads .env from it
    assert service["ExecStart"].startswith("%h/fpl-optimizer/.venv/bin/fplopt ")


def test_live_timer_runs_after_the_archiver_tick():
    live = unit_settings("fplopt-live.timer", "Timer")
    tick = unit_settings("fplopt-tick.timer", "Timer")
    assert tick["OnCalendar"] == "*:0/15" and live["OnCalendar"] == "*:05/15"
    assert live["Persistent"] == "true"
    assert unit_settings("fplopt-live.service", "Service")["ExecStart"].endswith("live tick")


def test_bot_service_restarts_and_stops_cleanly():
    service = unit_settings("fplopt-bot.service", "Service")
    assert service["Type"] == "simple" and service["Restart"] == "on-failure"
    assert service["SuccessExitStatus"] == "143"  # SIGTERM -> SystemExit(143) in the CLI
    assert service["ExecStart"].endswith("bot run")
    unit = unit_settings("fplopt-bot.service", "Unit")
    assert "StartLimitBurst" in unit and "StartLimitIntervalSec" in unit
    assert unit_settings("fplopt-bot.service", "Install")["WantedBy"] == "default.target"
