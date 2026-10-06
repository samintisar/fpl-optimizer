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


@pytest.mark.parametrize("unit", ["fplopt-daily.service", "fplopt-tick.service"])
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
