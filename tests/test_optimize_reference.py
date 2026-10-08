"""Differential check against open-fpl-solver (dev/reference_check.py) on 3 small real-data
instances. Skipped unless OPEN_FPL_SOLVER_DIR points at a clone of
solioanalytics/open-fpl-solver with its venv synced (`uv sync` in the clone) and the real
data/ exists (FPLOPT_DATA_DIR or the repo's data/); see dev/README.md."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OFS_DIR = os.environ.get("OPEN_FPL_SOLVER_DIR")
DATA_DIR = Path(os.environ.get("FPLOPT_DATA_DIR", REPO / "data"))

pytestmark = [
    pytest.mark.skipif(not OFS_DIR, reason="OPEN_FPL_SOLVER_DIR not set"),
    pytest.mark.skipif(
        not (DATA_DIR / "player_snapshot.parquet").exists(), reason="real data/ not available"
    ),
]


def _harness():
    spec = importlib.util.spec_from_file_location(
        "reference_check", REPO / "dev" / "reference_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def test_planner_matches_open_fpl_solver_on_small_instances():
    rc = _harness()
    rows = rc.check(DATA_DIR, Path(OFS_DIR).resolve(), rc.QUICK_SPECS, 1e-4, 300.0)
    assert len(rows) == len(rc.QUICK_SPECS)
    for row in rows:
        assert row.verdict.startswith("agree"), rc.table([row])
        assert abs(row.diff) <= row.tol, rc.table([row])
        if not row.first_equal:  # only an alternative optimum may differ
            assert abs(row.fixed_theirs - row.ours) <= row.tol, rc.table([row])
