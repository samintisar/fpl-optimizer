"""The single access path (PLAN §4), enforced statically: feature modules read data only
through the `AsOfView` they are given. `store.py` (the reader) and `leakcheck.py` (the
harness, which loads and corrupts whole tables) are the only exceptions."""

import ast
import inspect
from pathlib import Path

import pytest

import fplopt.features
from fplopt.features import FEATURES

FEATURES_DIR = Path(fplopt.features.__file__).parent
EXEMPT = {"store.py", "leakcheck.py"}
FORBIDDEN_MODULES = (
    "pyarrow",
    "duckdb",
    "sqlite3",
    "pathlib",
    "os",
    "io",
    "glob",
    "shutil",
    "pickle",
    "importlib",
    "httpx",
    "fplopt.build",
    "fplopt.ingest",
    "fplopt.adapters",
)
FORBIDDEN_CALLS = {"open", "eval", "exec", "__import__"}
FILE_METHODS = ("read_", "to_parquet", "to_csv", "to_pickle", "to_feather", "to_sql")


def violations(source: str) -> list[str]:
    """What a feature module must not do: import I/O or build/ingest modules, call
    open/eval/exec or pandas file readers/writers, or reach into private attributes
    (e.g. `view._store`, the back door around the as-of filter)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                found += [f"imports {a.name}" for a in node.names if a.name.startswith("read_")]
            for name in names:
                if any(name == m or name.startswith(f"{m}.") for m in FORBIDDEN_MODULES):
                    found.append(f"imports {name}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                found.append(f"calls {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr.startswith(FILE_METHODS):
                found.append(f"calls .{func.attr}()")
        elif isinstance(node, ast.Attribute):
            private = node.attr.startswith("_") and not node.attr.startswith("__")
            if private and not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                found.append(f"accesses private attribute .{node.attr}")
    return found


def feature_modules() -> list[Path]:
    return sorted(p for p in FEATURES_DIR.glob("*.py") if p.name not in EXEMPT)


def test_feature_modules_exist():
    assert {p.name for p in feature_modules()} >= {"__init__.py", "baseline.py"}


@pytest.mark.parametrize("path", feature_modules(), ids=lambda p: p.name)
def test_feature_modules_read_only_through_the_view(path):
    assert violations(path.read_text(encoding="utf-8")) == []


def test_the_scanner_catches_each_kind_of_bypass():
    source = """
import pyarrow.parquet as pq
from pathlib import Path
from fplopt.build.common import BuildContext
from pandas import read_parquet
import os.path

def leaky(view):
    a = pd.read_parquet("data/player_match.parquet")
    b = open("data/x.json").read()
    c = view._store._frames["player_match"]
    df.to_parquet("out.parquet")
    return a
"""
    assert sorted(violations(source)) == sorted(
        [
            "imports pyarrow.parquet",
            "imports pathlib",
            "imports fplopt.build.common",
            "imports read_parquet",
            "imports os.path",
            "calls .read_parquet()",
            "calls open()",
            "accesses private attribute ._store",
            "accesses private attribute ._frames",
            "calls .to_parquet()",
        ]
    )
    assert violations("def f(view):\n    return view.table('x').__len__()\n") == []


def test_feature_builders_take_exactly_the_view():
    for name, builder in FEATURES.items():
        parameters = list(inspect.signature(builder).parameters.values())
        assert len(parameters) == 1, name
        assert parameters[0].annotation in ("AsOfView", fplopt.features.AsOfView), name
        assert builder.__module__.startswith("fplopt.features."), name
