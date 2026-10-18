"""The research loop's guard hook and keep/discard judge (`autoresearch/`)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "autoresearch"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load("guard")
judge = load("judge")


@pytest.mark.parametrize(
    "command",
    [
        "uv run fplopt backtest compare --a x --b y --seasons 2016-2024",
        "ssh fplopt-server 'cd ~/fpl-research && uv run fplopt models eval --seasons 2023'",
        "uv run python dev/calibrate_v1.py walk --out x",  # defaults include validate
        "uv run fplopt models eval --models v1 --seasons 2026",  # live is monitor only
        "uv run fplopt holdout run",
        "git push origin research",
        "git checkout main",
        "gh pr create --title x",
        "ssh botbox 'ls'",
        "cat .env",
        "uv run fplopt backtest run --seasons 2016-2022 --data-dir ~/fpl-live/data",
        "sed -i 's/MIN_GAIN = 0.0005/MIN_GAIN = 0/' autoresearch/judge.py",
        "echo x > dev/calibrate_v1.py",
        "echo x >> autoresearch/judge.py",
        "python -c 'print(1)' 2>autoresearch/program.md",
        "git checkout try/x -- src/fplopt/backtest/scoring.py",
        "ssh fplopt-server 'cd ~/fpl-harness && git checkout -q try/x'",
        "cp guard.py C:/Users/samin/.fpl-harness/guard.py",
    ],
)
def test_the_guard_refuses(command):
    assert guard.check_command(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "uv run fplopt backtest compare --a x --b y --seasons 2016-2022 --jobs 14",
        "uv run fplopt models eval --models v1,rolling --seasons 2017,2021-22",
        "bash autoresearch/sync.sh all",
        "ssh fplopt-server 'cd ~/fpl-research && bash ~/fpl-harness/autoresearch/model_eval.sh'",
        "ssh fplopt-server 'cd ~/fpl-research && ~/.local/bin/uv run python "
        "~/fpl-harness/autoresearch/judge.py compare a/metrics.json b/metrics.json'",
        "cat autoresearch/judge.py",
        "uv run ruff format src/fplopt/models/minutes.py",
        "git checkout -b try/x",
        "ls results > /dev/null",
        "bash autoresearch/sync.sh code 2>&1 | tail -3",
        "uv run fplopt backtest run --seasons 2016-2022 > results/x.log 2>&1",
        "git commit -q -m 'x' -m 'Co-Authored-By: C <a@b.c>' autoresearch/ideas.md",
        "ssh fplopt-server 'cd ~/fpl-live && bash ~/fpl-harness/autoresearch/weekly.sh'",
        "ssh fplopt-server 'cd ~/fpl-live && ~/.local/bin/uv run python "
        "~/fpl-harness/autoresearch/status.py'",
        "git commit -m 'try: x' && git reset --hard HEAD~1",
        "gh issue view 39",
        "uv run pytest tests/test_models_availability.py -q",
    ],
)
def test_the_guard_allows(command):
    assert guard.check_command(command) is None


@pytest.mark.parametrize(
    ("path", "frozen"),
    [
        ("C:/x/fpl-autoresearch/autoresearch/judge.py", True),
        ("C:\\x\\fpl-autoresearch\\.claude\\settings.local.json", True),
        ("/x/src/fplopt/bot/planner.py", True),
        ("/x/config/scoring/2026-27.json", True),
        ("/x/autoresearch/ideas.md", False),
        ("/x/autoresearch/ledger.tsv", False),
        ("/x/src/fplopt/models/minutes.py", False),
    ],
)
def test_frozen_paths(path, frozen):
    assert (guard.check_path(path) is not None) == frozen


def hook(event):
    return subprocess.run(
        [sys.executable, str(ROOT / "guard.py")],
        input=json.dumps(event),
        text=True,
        capture_output=True,
    )


CLONE = "C:\\Users\\samin\\Documents\\GitHub\\fpl-autoresearch"
MAIN = "C:\\Users\\samin\\Documents\\GitHub\\fpl-optimizer"


def test_the_hook_blocks_with_exit_2_in_the_research_clone():
    command = {"command": "uv run fplopt holdout run"}
    done = hook({"tool_name": "Bash", "tool_input": command, "cwd": CLONE})
    assert done.returncode == 2 and "holdout" in done.stderr
    # A session started in the main checkout that works in the clone is guarded too.
    command = {"command": f"cd {CLONE} && uv run fplopt holdout run"}
    assert hook({"tool_name": "Bash", "tool_input": command, "cwd": MAIN}).returncode == 2
    path = {"file_path": CLONE + "\\autoresearch\\judge.py"}
    assert hook({"tool_name": "Edit", "tool_input": path, "cwd": MAIN}).returncode == 2


def test_the_hook_leaves_other_work_alone():
    command = {"command": "uv run fplopt backtest compare --seasons 2023-2024"}
    assert hook({"tool_name": "Bash", "tool_input": command, "cwd": MAIN}).returncode == 0
    path = {"file_path": MAIN + "\\autoresearch\\judge.py"}
    assert hook({"tool_name": "Edit", "tool_input": path, "cwd": MAIN}).returncode == 0


def metrics(mse_h0, mse_h15=4.68, seasons=None, **extra):
    base = {m: 1.0 for m in judge.METRICS} | {"mean_xp_h0": 1.317, "mean_points_h0": 1.327}
    mean = base | {"mse_h0": mse_h0, "mse_h15": mse_h15} | extra
    seasons = seasons or {str(s): mse_h0 for s in range(2017, 2023)}
    return {"mean": mean, "per_season": {s: base | {"mse_h0": v} for s, v in seasons.items()}}


def test_the_judge_keeps_a_clear_gain_and_discards_noise():
    base = metrics(4.2609)
    ok, _ = judge.compare(base, metrics(4.2590))
    assert ok
    ok, reasons = judge.compare(base, metrics(4.2607))  # below MIN_GAIN
    assert not ok and any(r.startswith("FAIL  MSE h0") for r in reasons)


def test_a_change_in_the_flag_seasons_only_can_be_kept():
    base_seasons = {str(s): 4.3 for s in range(2017, 2023)}
    cand_seasons = base_seasons | {"2021": 4.29, "2022": 4.296}
    ok, _ = judge.compare(metrics(4.3, seasons=base_seasons), metrics(4.2977, seasons=cand_seasons))
    assert ok


def test_the_judge_discards_a_worse_component():
    base = metrics(4.2609)
    ok, reasons = judge.compare(base, metrics(4.2590, brier_start_h0=1.01))
    assert not ok and any("brier_start_h0" in r and r.startswith("FAIL") for r in reasons)
