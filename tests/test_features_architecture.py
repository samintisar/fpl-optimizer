"""The single access path (PLAN §4), enforced statically: feature modules read data only
through the `AsOfView` they are given, and keep no state between calls. `store.py` (the
reader) and `leakcheck.py` (the harness, which loads and corrupts whole tables) are the only
exceptions; every other module under `fplopt/features/` (subpackages included) is scanned.

Rules (`violations`):
- imports: an allowlist of pure modules (`ALLOWED_MODULES`), `AsOfView` from
  `fplopt.features.store` and sibling feature modules; nothing else (no `DataStore`, no
  `leakcheck`, no other fplopt module, no I/O library);
- no dynamic access: `getattr`/`setattr`/`delattr`/`vars`/`globals`/`locals`/`__import__`/
  `eval`/`exec`/`compile`/`open` calls, dunder attributes that reach into objects or
  modules (`__dict__`, `__globals__`, ...), private attributes (`view._store`);
- no pandas file readers/writers (`read_*`, `to_parquet`, ...);
- no state between calls: no `global`/`nonlocal`, no cache decorators, module-level values
  only immutable (constants, tuples, frozensets, a few immutable constructors; the `FEATURES`
  registry in `__init__` excepted), and no function mutates a module-level name.
"""

import ast
import inspect
from pathlib import Path

import pytest

import fplopt.features
from fplopt.features import FEATURES

FEATURES_DIR = Path(fplopt.features.__file__).parent
PACKAGE = "fplopt.features"
EXEMPT = {FEATURES_DIR / "store.py", FEATURES_DIR / "leakcheck.py"}
ALLOWED_MODULES = {
    "__future__",
    "collections",
    "collections.abc",
    "dataclasses",
    "logging",  # warnings about incomplete inputs (player_pool); output only
    "math",
    "numpy",
    "pandas",
    "typing",
}
# Names a feature module may import from the store: the view type, nothing that reads files.
STORE_NAMES = {"AsOfView"}
FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "globals",
    "locals",
    "breakpoint",
}
FORBIDDEN_DUNDERS = {
    "__dict__",
    "__globals__",
    "__builtins__",
    "__class__",
    "__bases__",
    "__mro__",
    "__subclasses__",
    "__code__",
    "__closure__",
    "__self__",
    "__func__",
    "__module__",
    "__getattribute__",
    "__loader__",
    "__spec__",
    "__import__",
    "__wrapped__",
}
FILE_METHODS = ("read_", "to_parquet", "to_csv", "to_pickle", "to_feather", "to_sql")
CACHE_DECORATORS = {"cache", "lru_cache", "cached_property"}
# Module-level calls that build immutable values.
IMMUTABLE_CALLS = {
    "frozenset",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "pd.DatetimeTZDtype",
    "pd.Timedelta",
    "pd.Timestamp",
    "logging.getLogger",
    "TypeVar",
}
MUTATORS = {
    "append",
    "extend",
    "insert",
    "pop",
    "popitem",
    "remove",
    "clear",
    "update",
    "setdefault",
    "add",
    "discard",
    "sort",
    "reverse",
    "__setitem__",
    "__delitem__",
}
# Module-level containers allowed per file (path relative to fplopt/features): the registry.
MUTABLE_ALLOWED = {"__init__.py": {"FEATURES"}}


def feature_modules() -> list[Path]:
    return sorted(p for p in FEATURES_DIR.rglob("*.py") if p not in EXEMPT)


def module_name(path: Path) -> str:
    parts = list(path.relative_to(FEATURES_DIR).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PACKAGE, *parts])


def sibling_modules() -> set[str]:
    return {module_name(p) for p in feature_modules()} - {PACKAGE}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _resolve(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    """Absolute module of a (possibly relative) `from ... import`."""
    if not node.level:
        return node.module or ""
    base = module.split(".") if is_package else module.split(".")[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def _import_violations(tree: ast.AST, module: str, is_package: bool) -> list[str]:
    siblings = sibling_modules()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    found.append(f"imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            source = _resolve(node, module, is_package)
            names = [alias.name for alias in node.names]
            found += [f"imports {name}" for name in names if name.startswith(FILE_METHODS)]
            if source in ALLOWED_MODULES or source in siblings:
                continue
            if source == f"{PACKAGE}.store":
                found += [
                    f"imports {name} from {source}" for name in names if name not in STORE_NAMES
                ]
            elif source == PACKAGE:
                found += [
                    f"imports {name} from {source}"
                    for name in names
                    if f"{PACKAGE}.{name}" not in siblings
                ]
            else:
                found.append(f"imports {source}")
    return found


def _module_level_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        for target in targets:
            names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def _immutable(node: ast.AST | None) -> bool:
    """A module-level value that cannot be mutated later."""
    if node is None or isinstance(node, ast.Constant | ast.Name | ast.Attribute):
        return True
    if isinstance(node, ast.Tuple):
        return all(_immutable(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp):
        return _immutable(node.operand)
    if isinstance(node, ast.BinOp):
        return _immutable(node.left) and _immutable(node.right)
    if isinstance(node, ast.BoolOp):
        return all(_immutable(v) for v in node.values)
    if isinstance(node, ast.Subscript):  # type aliases, e.g. Callable[[AsOfView], ...]
        return isinstance(node.value, ast.Name | ast.Attribute)
    if isinstance(node, ast.Call):
        return _dotted(node.func) in IMMUTABLE_CALLS
    return False


def _state_violations(tree: ast.Module, relative: str) -> list[str]:
    found = []
    allowed = MUTABLE_ALLOWED.get(relative, set())
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if isinstance(node, ast.AugAssign):
                found += [f"module-level {name} is updated in place" for name in names]
            elif not _immutable(node.value) and not set(names) <= allowed:
                found += [f"module-level {name} is mutable" for name in names]
    module_names = _module_level_names(tree)
    scopes = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    functions = [n for n in ast.walk(tree) if isinstance(n, scopes)]
    for function in functions:
        for node in ast.walk(function):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign | ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.Delete):
                targets = node.targets
            for target in targets:
                if isinstance(target, ast.Subscript | ast.Attribute):
                    root = target.value
                    while isinstance(root, ast.Subscript | ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in module_names:
                        found.append(f"mutates module-level {root.id}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATORS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_names
            ):
                found.append(f"mutates module-level {node.func.value.id}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Global | ast.Nonlocal):
            found.append(f"{type(node).__name__.lower()} {', '.join(node.names)}")
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            for decorator in node.decorator_list:
                if _dotted(decorator).split(".")[-1] in CACHE_DECORATORS:
                    found.append(f"cache decorator @{_dotted(decorator)}")
    return found


def violations(source: str, path: Path | None = None) -> list[str]:
    """What a feature module must not do (see the module docstring). `path` (default: a
    module directly in fplopt/features) resolves relative imports and per-file exceptions."""
    path = FEATURES_DIR / "example.py" if path is None else path
    tree = ast.parse(source)
    found = _import_violations(tree, module_name(path), path.name == "__init__.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                found.append(f"calls {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr.startswith(FILE_METHODS):
                found.append(f"calls .{func.attr}()")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_DUNDERS:
                found.append(f"accesses .{node.attr}")
            private = node.attr.startswith("_") and not node.attr.startswith("__")
            if private and not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                found.append(f"accesses private attribute .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_DUNDERS:
            found.append(f"uses {node.id}")
    found += _state_violations(tree, path.relative_to(FEATURES_DIR).as_posix())
    return found


def test_feature_modules_exist():
    assert {p.name for p in feature_modules()} >= {"__init__.py", "baseline.py"}


@pytest.mark.parametrize("path", feature_modules(), ids=lambda p: str(p.relative_to(FEATURES_DIR)))
def test_feature_modules_read_only_through_the_view(path):
    assert violations(path.read_text(encoding="utf-8"), path) == []


def test_subpackages_are_scanned(tmp_path, monkeypatch):
    sub = tmp_path / "extra"
    sub.mkdir()
    (sub / "more.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "store.py").write_text("", encoding="utf-8")
    monkeypatch.setitem(globals(), "FEATURES_DIR", tmp_path)
    monkeypatch.setitem(globals(), "EXEMPT", {tmp_path / "store.py"})
    assert [p.relative_to(tmp_path).as_posix() for p in feature_modules()] == ["extra/more.py"]
    assert module_name(sub / "more.py") == "fplopt.features.extra.more"


def test_the_scanner_catches_each_kind_of_bypass():
    source = """
import pyarrow.parquet as pq
from pathlib import Path
from fplopt.build.common import BuildContext
from pandas import read_parquet
import os.path
import fplopt.features.store
from fplopt.features.store import AsOfView, DataStore
from fplopt.features import leakcheck
from fplopt.features.leakcheck import load_tables
from .store import DataStore as Store
import importlib
from functools import lru_cache

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
            "imports fplopt.features.store",
            "imports DataStore from fplopt.features.store",
            "imports leakcheck from fplopt.features",
            "imports fplopt.features.leakcheck",
            "imports DataStore from fplopt.features.store",
            "imports importlib",
            "imports functools",
            "calls .read_parquet()",
            "calls open()",
            "accesses private attribute ._store",
            "accesses private attribute ._frames",
            "calls .to_parquet()",
        ]
    )
    assert violations("def f(view):\n    return view.table('x').__len__()\n") == []


def test_the_scanner_catches_dynamic_access_and_hidden_state():
    source = """
from functools import cache
import pandas as pd

SEEN = {}
ROWS = []
NAMES = set(["a"])
TOTAL = 0
TOTAL += 1

@cache
def cached(view):
    return 1

@functools.lru_cache(maxsize=None)
def cached_too(view):
    return 1

def dynamic(view):
    store = getattr(view, "_" + "store")
    setattr(view, "deadline", None)
    namespace = vars(view)
    g = globals()
    l = locals()
    m = __import__("os")
    d = view.__dict__
    f = dynamic.__globals__
    global TOTAL
    SEEN[view.deadline] = 1
    ROWS.append(view.deadline)
    return store

def outer():
    count = 0
    def inner():
        nonlocal count
"""
    found = violations(source)
    expected = [
        "imports functools",
        "module-level SEEN is mutable",
        "module-level ROWS is mutable",
        "module-level NAMES is mutable",
        "module-level TOTAL is updated in place",
        "cache decorator @cache",
        "cache decorator @functools.lru_cache",
        "calls getattr()",
        "calls setattr()",
        "calls vars()",
        "calls globals()",
        "calls locals()",
        "calls __import__()",
        "accesses .__dict__",
        "accesses .__globals__",
        "global TOTAL",
        "nonlocal count",
        "mutates module-level SEEN",
        "mutates module-level ROWS",
    ]
    for item in expected:
        assert item in found, item
    clean = """
import logging
from collections.abc import Callable

import pandas as pd

from fplopt.features.store import AsOfView

log = logging.getLogger(__name__)
UTC_US = pd.DatetimeTZDtype("us", "UTC")
COLUMNS = ("a", "b")
DTYPES = (("a", "int64"), ("b", "str"))
RATES = tuple(c for c in COLUMNS if c != "a")
Builder = Callable[[AsOfView], pd.DataFrame]

def f(view: AsOfView) -> pd.DataFrame:
    cache = {}
    cache["x"] = 1
    out = view.table("x", columns=list(COLUMNS))
    return out.astype(dict(DTYPES))
"""
    assert violations(clean) == []


def test_the_registry_is_the_only_allowed_module_level_container():
    registry = "from fplopt.features.baseline import f\n\nFEATURES = {'f': f}\n"
    assert violations(registry, FEATURES_DIR / "__init__.py") == []
    assert violations(registry, FEATURES_DIR / "baseline.py") == [
        "module-level FEATURES is mutable"
    ]
    mutate = registry + "\ndef register(f):\n    FEATURES['g'] = f\n"
    assert violations(mutate, FEATURES_DIR / "__init__.py") == ["mutates module-level FEATURES"]


def test_relative_imports_are_resolved():
    assert violations("from .baseline import player_pool\n") == []
    assert violations("from .store import AsOfView\n") == []
    assert violations("from .store import DataStore\n") == [
        "imports DataStore from fplopt.features.store"
    ]
    assert violations("from . import leakcheck\n") == ["imports leakcheck from fplopt.features"]
    assert violations("from ..build import tables\n") == ["imports fplopt.build"]


def test_feature_builders_take_exactly_the_view():
    for name, builder in FEATURES.items():
        parameters = list(inspect.signature(builder).parameters.values())
        assert len(parameters) == 1, name
        assert parameters[0].annotation in ("AsOfView", fplopt.features.AsOfView), name
        assert builder.__module__.startswith("fplopt.features."), name
