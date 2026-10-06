"""The single access path (PLAN §4), enforced statically: code that computes decision inputs
(feature builders, xP models; later the backtest policies) reads data only through the
`AsOfView` it is given, and keeps no state between calls.

What is scanned is configured in `TARGETS`, one `ScanTarget` per package (or per listed files
of a package):
- `fplopt/features/`: every module (subpackages included) except `store.py` (the reader) and
  `leakcheck.py` (the harness, which loads and corrupts whole tables); registry `FEATURES`;
- `fplopt/models/`: every module; may also import the scanned feature modules (and the
  `fplopt.features` package); registry `MODELS`.
A later scanned package adds an entry (e.g. `fplopt.backtest` with
`files=("policies.py", "start_states.py", "probes.py")`, trusting the features and models
targets, `extra_modules` for the pure backtest modules it uses and `registries` for its
registry).

Rules (`violations`, the same for every target):
- imports: an allowlist of pure modules (`PURE_MODULES`), the target's sibling modules,
  the modules scanned under its trusted targets, its `extra_modules`, and from `restricted`
  modules only the listed names (from `fplopt.features.store` only `AsOfView`); nothing else
  (no `DataStore`, no `leakcheck`, no other fplopt module, no I/O library). Importing a
  submodule by name (`from fplopt.features import leakcheck`) counts as importing it, and
  so does reaching it through a package name (`fplopt.features.store.DataStore` after
  `import fplopt.features.baseline`);
- no dynamic access: `getattr`/`setattr`/`delattr`/`vars`/`globals`/`locals`/`__import__`/
  `eval`/`exec`/`compile`/`open` calls, dunder attributes that reach into objects or
  modules (`__dict__`, `__globals__`, ...), private attributes (`view._store`);
- no pandas file readers/writers (`read_*`, `to_parquet`, ...);
- no state between calls: no `global`/`nonlocal`, no cache decorators, module-level values
  only immutable (constants, tuples, frozensets, a few immutable constructors; the target's
  `registries` excepted), and no function mutates a module-level name.
"""

import ast
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import fplopt
import fplopt.features
import fplopt.models
from fplopt.features import FEATURES
from fplopt.models import MODELS

SRC_DIR = Path(fplopt.__file__).parent.parent
FEATURES_DIR = Path(fplopt.features.__file__).parent
MODELS_DIR = Path(fplopt.models.__file__).parent
PURE_MODULES = frozenset(
    {
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
)
# Names a scanned module may import from the store: the view type, nothing that reads files.
STORE_NAMES = frozenset({"AsOfView"})
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


@dataclass(frozen=True)
class ScanTarget:
    """A package (or some of its files) held to the rules above."""

    package: str  # dotted, e.g. "fplopt.features"
    files: tuple[str, ...] | None = None  # paths relative to the package; None = every module
    exempt: frozenset[str] = frozenset()  # relative paths not scanned (files=None only)
    trusted: tuple["ScanTarget", ...] = ()  # their scanned modules are importable
    extra_modules: frozenset[str] = frozenset()  # further importable modules
    restricted: Mapping[str, frozenset[str]] = field(default_factory=dict)  # module -> names
    registries: Mapping[str, frozenset[str]] = field(default_factory=dict)  # file -> names
    root: Path | None = None  # default: the package's directory under src/

    @property
    def directory(self) -> Path:
        return self.root if self.root is not None else SRC_DIR.joinpath(*self.package.split("."))


FEATURES_TARGET = ScanTarget(
    package="fplopt.features",
    exempt=frozenset({"store.py", "leakcheck.py"}),
    restricted={"fplopt.features.store": STORE_NAMES},
    registries={"__init__.py": frozenset({"FEATURES"})},
)
MODELS_TARGET = ScanTarget(
    package="fplopt.models",
    trusted=(FEATURES_TARGET,),
    restricted={"fplopt.features.store": STORE_NAMES},
    registries={"__init__.py": frozenset({"MODELS"})},
)
TARGETS = (FEATURES_TARGET, MODELS_TARGET)


def scanned_files(target: ScanTarget) -> list[Path]:
    directory = target.directory
    if target.files is not None:
        return sorted(directory / name for name in target.files)
    exempt = {directory / name for name in target.exempt}
    return sorted(p for p in directory.rglob("*.py") if p not in exempt)


def module_name(path: Path, target: ScanTarget = FEATURES_TARGET) -> str:
    parts = list(path.relative_to(target.directory).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([target.package, *parts])


def scanned_modules(target: ScanTarget) -> set[str]:
    return {module_name(p, target) for p in scanned_files(target)}


def sibling_modules(target: ScanTarget = FEATURES_TARGET) -> set[str]:
    return scanned_modules(target) - {target.package}


def allowed_modules(target: ScanTarget) -> set[str]:
    """Modules a scanned module of `target` may import whole (any name from them)."""
    allowed = set(PURE_MODULES) | set(target.extra_modules) | sibling_modules(target)
    for trusted in target.trusted:
        allowed |= scanned_modules(trusted)
    return allowed


def known_modules() -> set[str]:
    """Every module of the fplopt package (to tell submodules from other imported names)."""
    out = set()
    for path in (SRC_DIR / "fplopt").rglob("*.py"):
        parts = list(path.relative_to(SRC_DIR).with_suffix("").parts)
        out.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return out


def feature_modules() -> list[Path]:
    return scanned_files(FEATURES_TARGET)


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


def _import_violations(
    tree: ast.AST, module: str, is_package: bool, target: ScanTarget
) -> list[str]:
    allowed = allowed_modules(target)
    siblings = sibling_modules(target)
    modules = known_modules() | siblings
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed:
                    found.append(f"imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            source = _resolve(node, module, is_package)
            names = [alias.name for alias in node.names]
            found += [f"imports {name}" for name in names if name.startswith(FILE_METHODS)]
            if source in allowed:
                # A submodule imported by name is an import of that module.
                found += [
                    f"imports {name} from {source}"
                    for name in names
                    if f"{source}.{name}" in modules and f"{source}.{name}" not in allowed
                ]
            elif source in target.restricted:
                found += [
                    f"imports {name} from {source}"
                    for name in names
                    if name not in target.restricted[source]
                ]
            elif source == target.package:
                found += [
                    f"imports {name} from {source}"
                    for name in names
                    if f"{source}.{name}" not in siblings
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


def _state_violations(tree: ast.Module, allowed: set[str]) -> list[str]:
    found = []
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


def violations(
    source: str, path: Path | None = None, target: ScanTarget = FEATURES_TARGET
) -> list[str]:
    """What a scanned module of `target` must not do (see the module docstring). `path`
    (default: a module directly in the target's package) resolves relative imports and
    per-file exceptions (registries)."""
    path = target.directory / "example.py" if path is None else path
    tree = ast.parse(source)
    module = module_name(path, target)
    found = _import_violations(tree, module, path.name == "__init__.py", target)
    allowed = allowed_modules(target)
    modules = known_modules()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                found.append(f"calls {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr.startswith(FILE_METHODS):
                found.append(f"calls .{func.attr}()")
        elif isinstance(node, ast.Attribute):
            # `import fplopt.features.baseline` binds `fplopt`: no reaching other modules
            # through it (`fplopt.features.store.DataStore`). Packages on the way to an
            # allowed module are fine.
            dotted = _dotted(node)
            if (
                dotted in modules
                and dotted not in allowed
                and not any(m.startswith(f"{dotted}.") for m in allowed)
            ):
                found.append(f"accesses {dotted}")
            if node.attr in FORBIDDEN_DUNDERS:
                found.append(f"accesses .{node.attr}")
            private = node.attr.startswith("_") and not node.attr.startswith("__")
            if private and not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                found.append(f"accesses private attribute .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_DUNDERS:
            found.append(f"uses {node.id}")
    relative = path.relative_to(target.directory).as_posix()
    found += _state_violations(tree, set(target.registries.get(relative, ())))
    return found


SCANNED = [(target, path) for target in TARGETS for path in scanned_files(target)]


def _scan_id(item: tuple[ScanTarget, Path]) -> str:
    target, path = item
    return f"{target.package.split('.')[-1]}/{path.relative_to(target.directory).as_posix()}"


def test_scanned_modules_exist():
    assert {p.name for p in feature_modules()} >= {"__init__.py", "baseline.py"}
    assert {p.name for p in scanned_files(MODELS_TARGET)} >= {"__init__.py", "baseline.py"}
    for target in TARGETS:
        for path in scanned_files(target):
            assert path.is_file(), path


@pytest.mark.parametrize("item", SCANNED, ids=[_scan_id(item) for item in SCANNED])
def test_scanned_modules_read_only_through_the_view(item):
    target, path = item
    assert violations(path.read_text(encoding="utf-8"), path, target) == []


def test_subpackages_are_scanned(tmp_path):
    sub = tmp_path / "extra"
    sub.mkdir()
    (sub / "more.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "store.py").write_text("", encoding="utf-8")
    target = ScanTarget("fplopt.features", exempt=frozenset({"store.py"}), root=tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in scanned_files(target)] == ["extra/more.py"]
    assert module_name(sub / "more.py", target) == "fplopt.features.extra.more"


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


# --- models ------------------------------------------------------------------------------


def test_model_modules_may_use_features_and_the_view():
    source = """
import logging

import numpy as np
import pandas as pd

import fplopt.features.baseline
from fplopt.features import FEATURES, AsOfView, baseline, compute_features
from fplopt.features.baseline import HORIZON, player_pool, upcoming_fixtures
from fplopt.features.store import AsOfView
from fplopt.models.baseline import xp_rolling
from .baseline import xp_ep_next

log = logging.getLogger(__name__)

def xp(view: AsOfView) -> pd.DataFrame:
    matches = view.table("player_match", columns=["player_key", "total_points"])
    return player_pool(view).merge(matches, on="player_key")
"""
    path = MODELS_DIR / "other.py"
    assert violations(source, path, MODELS_TARGET) == []


def test_the_scanner_catches_a_models_module_bypass():
    source = """
from pathlib import Path
import fplopt.features.store
from fplopt.features.store import AsOfView, DataStore, files_blocked
from fplopt.features import leakcheck, store
from fplopt.features.leakcheck import load_tables
from fplopt.build.tables import TABLES
from fplopt.backtest import simulator
from fplopt.models import MODELS
from ..features.store import DataStore as Store
from functools import lru_cache

FITTED = {}

def leaky(view):
    a = pd.read_parquet("data/player_match.parquet")
    b = view._store._frames["player_match"]
    c = getattr(view, "_store")
    FITTED[view.deadline] = a
    return a
"""
    found = violations(source, MODELS_DIR / "baseline.py", MODELS_TARGET)
    expected = [
        "imports pathlib",
        "imports fplopt.features.store",
        "imports DataStore from fplopt.features.store",
        "imports files_blocked from fplopt.features.store",
        "imports leakcheck from fplopt.features",
        "imports store from fplopt.features",
        "imports fplopt.features.leakcheck",
        "imports fplopt.build.tables",
        "imports fplopt.backtest",
        "imports MODELS from fplopt.models",
        "imports DataStore from fplopt.features.store",
        "imports functools",
        "module-level FITTED is mutable",
        "calls .read_parquet()",
        "accesses private attribute ._store",
        "accesses private attribute ._frames",
        "calls getattr()",
        "mutates module-level FITTED",
    ]
    assert sorted(found) == sorted(expected)


def test_no_reaching_other_modules_through_an_imported_package_name():
    source = """
import fplopt.features.baseline

def sneaky(view):
    store = fplopt.features.store.DataStore("data")
    tables = fplopt.build.tables.TABLES
    return fplopt.features.baseline.player_pool(store.as_of(view.deadline))
"""
    expected = [
        "accesses fplopt.features.store",
        "accesses fplopt.build.tables",
        "accesses fplopt.build",
    ]
    assert sorted(violations(source, MODELS_DIR / "x.py", MODELS_TARGET)) == sorted(expected)
    assert sorted(violations(source)) == sorted(expected)  # same in a feature module


def test_feature_modules_may_not_import_models():
    assert violations("from fplopt.models import MODELS\n") == ["imports fplopt.models"]
    assert violations("import fplopt.models.baseline\n") == ["imports fplopt.models.baseline"]


def test_each_target_allows_only_its_own_registry():
    registry = "from fplopt.models.baseline import xp_rolling\n\nMODELS = {'r': xp_rolling}\n"
    assert violations(registry, MODELS_DIR / "__init__.py", MODELS_TARGET) == []
    assert violations(registry, MODELS_DIR / "baseline.py", MODELS_TARGET) == [
        "module-level MODELS is mutable"
    ]
    features = "FEATURES = {}\n"
    assert violations(features, MODELS_DIR / "__init__.py", MODELS_TARGET) == [
        "module-level FEATURES is mutable"
    ]


def test_model_builders_take_exactly_the_view():
    assert list(MODELS) == ["rolling", "ep_next"]
    for name, model in MODELS.items():
        parameters = list(inspect.signature(model).parameters.values())
        assert len(parameters) == 1, name
        assert parameters[0].annotation in ("AsOfView", fplopt.features.AsOfView), name
        assert model.__module__.startswith("fplopt.models."), name


def test_a_target_of_listed_files_trusting_features_and_models(tmp_path):
    """How a later package joins the scan (e.g. fplopt/backtest/policies.py): listed files,
    trusted targets, extra pure modules of its own package and its registry."""
    (tmp_path / "policies.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "probes.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "simulator.py").write_text("open('data/x')\n", encoding="utf-8")  # not listed
    target = ScanTarget(
        package="fplopt.backtest",
        files=("policies.py", "probes.py"),
        trusted=(FEATURES_TARGET, MODELS_TARGET),
        extra_modules=frozenset({"fplopt.backtest.rules", "fplopt.backtest.state"}),
        restricted={"fplopt.features.store": STORE_NAMES},
        registries={"probes.py": frozenset({"PROBES"})},
        root=tmp_path,
    )
    assert [p.name for p in scanned_files(target)] == ["policies.py", "probes.py"]
    ok = """
from fplopt.features.baseline import player_pool
from fplopt.features.store import AsOfView
from fplopt.models import MODELS
from fplopt.models.baseline import xp_rolling
from fplopt.backtest.rules import Rules
from .state import SquadState
from .policies import x
"""
    assert violations(ok, tmp_path / "policies.py", target) == []
    probes = ok + "\nPROBES = {'a': x}\n"
    assert violations(probes, tmp_path / "probes.py", target) == []
    assert violations(probes, tmp_path / "policies.py", target) == [
        "module-level PROBES is mutable"
    ]
    bad = """
from fplopt.backtest.simulator import simulate
from fplopt.backtest import simulator
from fplopt.features.store import DataStore
from fplopt.features import leakcheck
"""
    assert sorted(violations(bad, tmp_path / "policies.py", target)) == sorted(
        [
            "imports fplopt.backtest.simulator",
            "imports simulator from fplopt.backtest",
            "imports DataStore from fplopt.features.store",
            "imports leakcheck from fplopt.features",
        ]
    )
