import importlib

import pytest

SUBPACKAGES = [
    "adapters",
    "ingest",
    "build",
    "features",
    "models",
    "optimize",
    "backtest",
    "bot",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"fplopt.{name}")
