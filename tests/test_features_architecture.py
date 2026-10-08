"""The single access path (PLAN §4), enforced statically: code that computes decision inputs
(feature builders, xP models; later the backtest policies) reads data only through the
`AsOfView` it is given, and keeps no state between calls.

What is scanned is configured in `TARGETS`, one `ScanTarget` per package (or per listed files
of a package):
- `fplopt/features/`: every module (subpackages included) except `store.py` (the reader) and
  `leakcheck.py` (the harness, which loads and corrupts whole tables); registry `FEATURES`;
- `fplopt/models/`: every module except `gbm.py`; may also import the scanned feature
  modules (and the `fplopt.features` package), `gbm`, and, name by name (`module_attrs`),
  `scipy.optimize` / `scipy.stats` / `scipy.special`; registry `MODELS`;
- `fplopt/models/gbm.py`: the only LightGBM entry point; `lightgbm` only through the names
  it lists (`module_attrs`), no booster file methods (`save_model`, `model_file=`);
- `fplopt/optimize/`: every module except `bench.py` (the benchmark, an orchestrator that
  opens the `DataStore` and drives the simulator's caches, like the simulator itself); the
  MILP planner the optimizer policy calls at a deadline. May also import the pure backtest
  modules `rules`, `state`, `gw_score`, `itertools`, `types`, and, attribute by attribute
  (`module_attrs`), the solver (`pulp`, `highspy`: only the names the planner uses, none
  that reads or writes model files) and `time` (only `perf_counter`); no registry;
- `fplopt/backtest/`: only `policies.py`, `start_states.py`, `probes.py` (what decides at a
  deadline); may also import the scanned feature, model and optimizer modules, the pure
  backtest modules `rules`, `state`, `gw_score` and `fplopt.seasons`; registry `PROBES`.
A later scanned package adds an entry the same way.

Rules (`violations`, the same for every target):
- imports: an allowlist of pure modules (`PURE_MODULES`), the target's sibling modules,
  the modules scanned under its trusted targets, its `extra_modules`, and from `restricted`
  modules only the listed names (from `fplopt.features.store` only `AsOfView`); nothing else
  (no `DataStore`, no `leakcheck`, no other fplopt module, no I/O library). From
  `module_attrs` modules (importable whole) only the listed attributes may be used, as
  `module.attr` or `from module import attr`; a dotted one (`scipy.optimize`) only by
  `from ... import` or with an alias, since `import scipy.optimize` binds `scipy`. Importing a
  submodule by name (`from fplopt.features import leakcheck`) counts as importing it, and
  so does reaching it through a package name (`fplopt.features.store.DataStore` after
  `import fplopt.features.baseline`);
- no dynamic access: `getattr`/`setattr`/`delattr`/`vars`/`globals`/`locals`/`__import__`/
  `eval`/`exec`/`compile`/`open` calls, dunder attributes that reach into objects or
  modules (`__dict__`, `__globals__`, ...), private attributes (`view._store`);
- no pandas file readers/writers (`read_*`, `to_parquet`, ...), no solver or booster file
  methods (`readModel`, `writeModel`, `writeLP`, `fromMPS`, `toJson`, `save_model`, ...) and
  no `model_file=` keyword;
- no state between calls: no `global`/`nonlocal`, no cache decorators, module-level values
  only immutable (constants, tuples, frozensets, a few immutable constructors called with
  immutable arguments; the target's `registries` excepted), and no function mutates a
  module-level name (assigning into it, or calling a mutator on anything reached from it:
  `NAME.append`, `NAME[k].update`, `NAME.attr.clear`, ...).
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
# Solver methods that read or write model, solution, basis or option files (PuLP, highspy).
SOLVER_FILE_METHODS = frozenset(
    {
        "readModel",
        "writeModel",
        "writeLP",
        "writeMPS",
        "fromMPS",
        "fromJson",
        "toJson",
        "readSolution",
        "writeSolution",
        "readBasis",
        "writeBasis",
        "readOptions",
        "writeOptions",
        "writeInfo",
        "writePresolvedModel",
        "writeIIS",
        "save_model",  # LightGBM Booster
        "save_binary",  # LightGBM Dataset
    }
)
FILE_KEYWORDS = frozenset({"model_file"})  # lightgbm.Booster(model_file=...)
CACHE_DECORATORS = {"cache", "lru_cache", "cached_property"}
# Module-level calls that build immutable values, when their arguments are immutable
# (`_immutable_arg`): constants, names, tuples of those, nested immutable calls.
IMMUTABLE_CALLS = {
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
# Immutable constructors that freeze their arguments: they may also take a list, set or
# dict literal or a comprehension, as long as its elements are immutable (the fresh
# container is copied or only reachable read-only).
FREEZING_CALLS = {
    "frozenset",
    "tuple",
    "MappingProxyType",  # a read-only view of a literal nothing else references
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
    immutable_calls: frozenset[str] = frozenset()  # further immutable module-level calls
    freezing_calls: frozenset[str] = frozenset()  # further freezing ones (FREEZING_CALLS)
    # Modules importable whole, but only these attributes of them may be used.
    module_attrs: Mapping[str, frozenset[str]] = field(default_factory=dict)
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
# LightGBM: in-memory datasets and boosters only (no file names; `FILE_KEYWORDS`).
LIGHTGBM_NAMES = frozenset({"Booster", "Dataset", "train"})
GBM_TARGET = ScanTarget(
    package="fplopt.models",
    files=("gbm.py",),
    extra_modules=frozenset({"types"}),  # MappingProxyType: read-only parameters
    module_attrs={"lightgbm": LIGHTGBM_NAMES},
)
# scipy for the models' MLE fits and distributions; no I/O name.
SCIPY_NAMES = {
    "scipy.optimize": frozenset({"OptimizeResult", "brentq", "least_squares", "minimize"}),
    "scipy.special": frozenset({"expit", "gammaln", "logit", "xlogy"}),
    "scipy.stats": frozenset({"nbinom", "norm", "poisson"}),
}
MODELS_TARGET = ScanTarget(
    package="fplopt.models",
    exempt=frozenset({"gbm.py"}),  # scanned as GBM_TARGET
    trusted=(FEATURES_TARGET, GBM_TARGET),
    restricted={"fplopt.features.store": STORE_NAMES},
    registries={"__init__.py": frozenset({"MODELS"})},
    module_attrs=SCIPY_NAMES,
)
PURE_BACKTEST_MODULES = frozenset(
    {
        "fplopt.backtest.rules",  # reads config/scoring (rules), never data/
        "fplopt.backtest.state",
        "fplopt.backtest.gw_score",
    }
)
# The solver API the planner uses: the model is built in memory and solved in-process; no
# name that reads or writes files (`SOLVER_FILE_METHODS` covers the object methods).
PULP_NAMES = frozenset(
    {
        "HiGHS",
        "LpAffineExpression",
        "LpBinary",
        "LpInteger",
        "LpMaximize",
        "LpProblem",
        "LpSolveStatus",
        "lpSum_vars",
        "lpSum_vars_coefs",
    }
)
HIGHSPY_NAMES = frozenset({"HighsModelStatus", "HighsVarType", "kHighsInf"})
OPTIMIZE_TARGET = ScanTarget(
    package="fplopt.optimize",
    exempt=frozenset({"bench.py"}),  # the benchmark harness: opens the DataStore
    extra_modules=PURE_BACKTEST_MODULES
    | frozenset(
        {
            "itertools",
            "types",  # MappingProxyType: read-only parameter mappings
        }
    ),
    module_attrs={
        "pulp": PULP_NAMES,
        "highspy": HIGHSPY_NAMES,
        "time": frozenset({"perf_counter"}),  # solve-time statistics only
    },
    immutable_calls=frozenset({"ChipScenario"}),  # frozen dataclass of tuples (NO_CHIP)
)
BACKTEST_TARGET = ScanTarget(
    package="fplopt.backtest",
    files=("policies.py", "start_states.py", "probes.py"),
    trusted=(FEATURES_TARGET, MODELS_TARGET, OPTIMIZE_TARGET),
    extra_modules=PURE_BACKTEST_MODULES | frozenset({"fplopt.seasons"}),
    restricted={"fplopt.features.store": STORE_NAMES},
    registries={"probes.py": frozenset({"PROBES"})},
    freezing_calls=frozenset({"OptimizerParams"}),  # copies mappings into read-only ones
)
TARGETS = (FEATURES_TARGET, MODELS_TARGET, GBM_TARGET, OPTIMIZE_TARGET, BACKTEST_TARGET)


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
    """Modules a scanned module of `target` may import whole (any name from them, except
    `module_attrs` modules: only their listed names)."""
    allowed = set(PURE_MODULES) | set(target.extra_modules) | sibling_modules(target)
    allowed |= set(target.module_attrs)
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
                elif alias.name in target.module_attrs and "." in alias.name and not alias.asname:
                    found.append(f"imports {alias.name} without an alias")
        elif isinstance(node, ast.ImportFrom):
            source = _resolve(node, module, is_package)
            names = [alias.name for alias in node.names]
            found += [f"imports {name}" for name in names if name.startswith(FILE_METHODS)]
            if source in target.module_attrs:
                found += [
                    f"imports {name} from {source}"
                    for name in names
                    if name not in target.module_attrs[source]
                ]
            elif source in allowed:
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


@dataclass(frozen=True)
class Calls:
    """Immutable constructors: plain ones and freezing ones (FREEZING_CALLS)."""

    plain: frozenset[str] = frozenset(IMMUTABLE_CALLS)
    freezing: frozenset[str] = frozenset(FREEZING_CALLS)

    @classmethod
    def of(cls, target: ScanTarget) -> "Calls":
        return cls(
            frozenset(IMMUTABLE_CALLS) | target.immutable_calls,
            frozenset(FREEZING_CALLS) | target.freezing_calls,
        )


DEFAULT_CALLS = Calls()


def _immutable(node: ast.AST | None, calls: Calls = DEFAULT_CALLS) -> bool:
    """A module-level value that cannot be mutated later: constants, names, tuples and
    operators over those, type-alias subscripts, and immutable constructors (`calls`)
    whose arguments are immutable (`_immutable_arg`)."""
    if node is None or isinstance(node, ast.Constant | ast.Name | ast.Attribute):
        return True
    if isinstance(node, ast.Tuple):
        return all(_immutable(e, calls) for e in node.elts)
    if isinstance(node, ast.Starred):  # `*xs` or `*(x for ...)` inside a tuple literal
        if isinstance(node.value, ast.GeneratorExp | ast.ListComp | ast.SetComp):
            return _immutable(node.value.elt, calls)
        return _immutable(node.value, calls)
    if isinstance(node, ast.JoinedStr | ast.FormattedValue):  # f-strings
        return True
    if isinstance(node, ast.UnaryOp):
        return _immutable(node.operand, calls)
    if isinstance(node, ast.BinOp):
        return _immutable(node.left, calls) and _immutable(node.right, calls)
    if isinstance(node, ast.BoolOp):
        return all(_immutable(v, calls) for v in node.values)
    if isinstance(node, ast.Subscript):  # type aliases, e.g. Callable[[AsOfView], ...]
        return isinstance(node.value, ast.Name | ast.Attribute)
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name not in calls.plain | calls.freezing:
            return False
        freezes = name in calls.freezing
        args = [*node.args, *(k.value for k in node.keywords)]
        return all(_immutable_arg(a, calls, freezes) for a in args)
    return False


def _immutable_arg(node: ast.AST, calls: Calls, freezes: bool) -> bool:
    """An argument that leaves an immutable constructor's result immutable: an immutable
    value (`_immutable`); for a freezing constructor also a list/set/dict literal or a
    comprehension whose elements (keys and values) are immutable values."""
    if isinstance(node, ast.Starred):
        return _immutable_arg(node.value, calls, freezes)
    if _immutable(node, calls):
        return True
    if not freezes:
        return False
    if isinstance(node, ast.List | ast.Set):
        return all(_immutable(e, calls) for e in node.elts)
    if isinstance(node, ast.Dict):
        keys = all(_immutable(k, calls) for k in node.keys if k is not None)
        return keys and all(_immutable(v, calls) for v in node.values)
    if isinstance(node, ast.GeneratorExp | ast.ListComp | ast.SetComp):
        return _immutable(node.elt, calls)
    if isinstance(node, ast.DictComp):
        return _immutable(node.key, calls) and _immutable(node.value, calls)
    return False


def _root_name(node: ast.AST) -> str | None:
    """The name an expression like `A[k].b.c(...)[0]` is reached from (`A`), if any."""
    while isinstance(node, ast.Subscript | ast.Attribute | ast.Call):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _state_violations(
    tree: ast.Module, allowed: set[str], calls: Calls = DEFAULT_CALLS
) -> list[str]:
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if isinstance(node, ast.AugAssign):
                found += [f"module-level {name} is updated in place" for name in names]
            elif not _immutable(node.value, calls) and not set(names) <= allowed:
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
                and _root_name(node.func.value) in module_names
            ):
                found.append(f"mutates module-level {_root_name(node.func.value)}")
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
    # Local names bound to `module_attrs` modules (`import pulp`, `import pulp as pl`).
    restricted_aliases = {
        (alias.asname or alias.name): alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name in target.module_attrs
    }
    # Such a module may only appear as `module.attr`: bound to another name (`clock = time`),
    # passed on or stored, its attributes could be reached unchecked.
    attribute_bases = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in restricted_aliases
            and id(node) not in attribute_bases
        ):
            found.append(f"uses {restricted_aliases[node.id]} other than as an attribute base")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            source = restricted_aliases.get(node.value.id)
            if source is not None and node.attr not in target.module_attrs[source]:
                found.append(f"accesses {source}.{node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                found.append(f"calls {func.id}()")
            if isinstance(func, ast.Attribute) and (
                func.attr.startswith(FILE_METHODS) or func.attr in SOLVER_FILE_METHODS
            ):
                found.append(f"calls .{func.attr}()")
            found += [f"passes {k.arg}=" for k in node.keywords if k.arg in FILE_KEYWORDS]
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
    found += _state_violations(tree, set(target.registries.get(relative, ())), Calls.of(target))
    return found


SCANNED = [(target, path) for target in TARGETS for path in scanned_files(target)]


def _scan_id(item: tuple[ScanTarget, Path]) -> str:
    target, path = item
    return f"{target.package.split('.')[-1]}/{path.relative_to(target.directory).as_posix()}"


def test_scanned_modules_exist():
    assert {p.name for p in feature_modules()} >= {"__init__.py", "baseline.py"}
    assert {p.name for p in scanned_files(MODELS_TARGET)} >= {"__init__.py", "baseline.py"}
    assert [p.name for p in scanned_files(BACKTEST_TARGET)] == [
        "policies.py",
        "probes.py",
        "start_states.py",
    ]
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


def test_lightgbm_only_through_gbm_and_scipy_name_by_name():
    gbm = MODELS_DIR / "gbm.py"
    other = MODELS_DIR / "minutes.py"
    ok_gbm = """
import lightgbm

def fit(x, y):
    return lightgbm.train({}, lightgbm.Dataset(x, label=y))
"""
    assert violations(ok_gbm, gbm, GBM_TARGET) == []
    bad_gbm = """
import lightgbm
from lightgbm import cv

def fit(x, y, path):
    booster = lightgbm.Booster(model_file=path)
    booster.save_model(path)
    lightgbm.Dataset(x).save_binary(path)
    return lightgbm.cv({}, x)
"""
    assert sorted(violations(bad_gbm, gbm, GBM_TARGET)) == sorted(
        [
            "imports cv from lightgbm",
            "passes model_file=",
            "calls .save_model()",
            "calls .save_binary()",
            "accesses lightgbm.cv",
        ]
    )
    ok_models = """
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.stats import poisson
from fplopt.models.gbm import Gbm, train
from .gbm import GbmParams

def fit(x):
    return minimize(lambda p: poisson.logpmf(x, p).sum(), 1.0), opt.least_squares
"""
    assert violations(ok_models, other, MODELS_TARGET) == []
    bad_models = """
import lightgbm
import scipy.optimize
import scipy.io
from scipy.io import loadmat
from scipy.stats import gaussian_kde
import scipy.optimize as opt

def fit(x):
    return opt.root(x)
"""
    assert sorted(violations(bad_models, other, MODELS_TARGET)) == sorted(
        [
            "imports lightgbm",
            "imports scipy.optimize without an alias",
            "imports scipy.io",
            "imports scipy.io",
            "imports gaussian_kde from scipy.stats",
            "accesses scipy.optimize.root",
        ]
    )


def test_model_builders_take_exactly_the_view():
    assert list(MODELS) == ["rolling", "ep_next", "ep_next_fade"]
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


# --- backtest policies, start states, probes -----------------------------------------------

BACKTEST_DIR = BACKTEST_TARGET.directory


def test_backtest_modules_may_use_features_models_and_pure_backtest_modules():
    source = """
from collections import Counter

import numpy as np
import pandas as pd

from fplopt.backtest.gw_score import Lineup
from fplopt.backtest.rules import Rules, backtest_rules
from fplopt.backtest.state import Decision, SquadState, apply_decision
from fplopt.backtest.policies import GreedyPolicy
from .start_states import template_state
from fplopt.features.baseline import player_pool, pool_coverage
from fplopt.features.store import AsOfView
from fplopt.models import MODELS
from fplopt.seasons import HOLDOUT_SEASONS

def decide(view: AsOfView) -> pd.DataFrame:
    return player_pool(view)
"""
    for name in ("policies.py", "start_states.py"):
        assert violations(source, BACKTEST_DIR / name, BACKTEST_TARGET) == []
    probes = source + "\nPROBES = {'p': decide}\n"
    assert violations(probes, BACKTEST_DIR / "probes.py", BACKTEST_TARGET) == []
    assert violations(probes, BACKTEST_DIR / "policies.py", BACKTEST_TARGET) == [
        "module-level PROBES is mutable"
    ]


@pytest.mark.parametrize("name", ["policies.py", "start_states.py", "probes.py"])
def test_the_scanner_catches_a_backtest_module_bypass(name):
    source = """
from pathlib import Path
from fplopt.backtest.simulator import simulate
from fplopt.backtest import simulator
from fplopt.backtest.evaluate import paired
from fplopt.features.store import AsOfView, DataStore
from fplopt.features import leakcheck
from fplopt.features.leakcheck import load_tables
from fplopt.build.tables import TABLES
from functools import lru_cache
import fplopt.cli

SEEN = {}

def decide(ctx):
    a = pd.read_parquet("data/player_match.parquet")
    b = ctx.view._store._frames["player_match"]
    c = getattr(ctx.view, "_store")
    d = fplopt.features.store.DataStore("data")
    SEEN[ctx.view.deadline] = a
    return a
"""
    found = violations(source, BACKTEST_DIR / name, BACKTEST_TARGET)
    expected = [
        "imports pathlib",
        "imports fplopt.backtest.simulator",
        "imports simulator from fplopt.backtest",
        "imports fplopt.backtest.evaluate",
        "imports DataStore from fplopt.features.store",
        "imports leakcheck from fplopt.features",
        "imports fplopt.features.leakcheck",
        "imports fplopt.build.tables",
        "imports functools",
        "imports fplopt.cli",
        "module-level SEEN is mutable",
        "calls .read_parquet()",
        "accesses private attribute ._store",
        "accesses private attribute ._frames",
        "calls getattr()",
        "accesses fplopt.features.store",
        "mutates module-level SEEN",
    ]
    assert sorted(found) == sorted(expected)


def test_probes_take_exactly_the_view():
    from fplopt.backtest.probes import PROBES

    assert list(PROBES) == [
        "greedy_rolling_random0",
        "greedy_ep_next_template",
        "roll_rolling_template",
        "optimizer_ep_next_template",
        "optimizer_rolling_random0",
    ]
    for name, probe in PROBES.items():
        parameters = list(inspect.signature(probe).parameters.values())
        assert len(parameters) == 1, name
        assert parameters[0].annotation in ("AsOfView", fplopt.features.AsOfView), name
        assert probe.__module__ == "fplopt.backtest.probes", name


# --- optimizer -----------------------------------------------------------------------------

OPTIMIZE_DIR = OPTIMIZE_TARGET.directory


def test_the_optimizer_package_is_scanned_except_the_benchmark():
    names = {p.name for p in scanned_files(OPTIMIZE_TARGET)}
    assert {"model.py", "plans.py", "params.py", "problem.py", "search.py"} <= names
    assert "bench.py" not in names


def test_optimizer_modules_may_use_the_solver_and_pure_backtest_modules():
    source = """
import itertools
import math
import time
from types import MappingProxyType

import pulp

from fplopt.backtest.rules import Rules
from fplopt.backtest.state import Transfer
from fplopt.optimize.params import OptimizerParams
from .chips import ChipScenario

DEFAULTS = MappingProxyType({1: 2.0})
NO_CHIP = ChipScenario()

def solve(model):
    import highspy

    start = time.perf_counter()
    return pulp.LpProblem("x", pulp.LpMaximize), highspy.kHighsInf, start
"""
    assert violations(source, OPTIMIZE_DIR / "model.py", OPTIMIZE_TARGET) == []


def test_the_scanner_catches_an_optimizer_module_bypass():
    source = """
from pathlib import Path
from fplopt.features.store import DataStore
from fplopt.backtest.simulator import Caches
from fplopt.optimize import bench
import fplopt.features.baseline

PLANS = {}
OTHER = ChipScenario()

def plan(problem):
    PLANS[problem] = pd.read_parquet("data/x.parquet")
    return PLANS
"""
    found = violations(source, OPTIMIZE_DIR / "plans.py", OPTIMIZE_TARGET)
    expected = [
        "imports pathlib",
        "imports fplopt.features.store",
        "imports fplopt.backtest.simulator",
        "imports bench from fplopt.optimize",
        "imports fplopt.features.baseline",
        "module-level PLANS is mutable",
        "calls .read_parquet()",
        "mutates module-level PLANS",
    ]
    assert sorted(found) == sorted(expected)
    # ChipScenario() is immutable only in the optimizer package.
    assert violations("NO_CHIP = ChipScenario()\n") == ["module-level NO_CHIP is mutable"]


def test_policies_may_use_the_optimizer_but_not_its_benchmark():
    ok = "from fplopt.optimize import OptimizerParams, PlanInput, optimize\n"
    assert violations(ok, BACKTEST_DIR / "policies.py", BACKTEST_TARGET) == []
    bad = "from fplopt.optimize.bench import run_bench\nfrom fplopt.optimize import bench\n"
    assert sorted(violations(bad, BACKTEST_DIR / "policies.py", BACKTEST_TARGET)) == [
        "imports bench from fplopt.optimize",
        "imports fplopt.optimize.bench",
    ]


# --- review bypasses (Phase 4) ---------------------------------------------------------------


def test_immutable_constructors_need_immutable_arguments():
    """(a) MappingProxyType/ChipScenario/OptimizerParams are immutable only with immutable
    arguments: a proxy over a dict holding a list, a frozen dataclass holding a list, or a
    proxy over a mutable call's result can still be mutated."""
    bad = """
A = MappingProxyType({"a": []})
B = MappingProxyType(dict())
C = MappingProxyType(make_table())
D = tuple([[1], [2]])
E = frozenset(set_of_lists())
"""
    assert sorted(violations(bad, OPTIMIZE_DIR / "params.py", OPTIMIZE_TARGET)) == [
        f"module-level {name} is mutable" for name in "ABCDE"
    ]
    chips = "X = ChipScenario(chips=[(0, 'wildcard')])\nY = ChipScenario(((0, 'bboost'),))\n"
    assert violations(chips, OPTIMIZE_DIR / "chips.py", OPTIMIZE_TARGET) == [
        "module-level X is mutable"
    ]
    params = "P = OptimizerParams(prune_n={1: 8}, bench_weights=(0.1, 0.2, 0.3, 0.4))\n"
    assert violations(params, BACKTEST_DIR / "probes.py", BACKTEST_TARGET) == []
    leaky = "P = OptimizerParams(prune_n={1: []})\nQ = OptimizerParams(prune_n=loaded())\n"
    assert violations(leaky, BACKTEST_DIR / "probes.py", BACKTEST_TARGET) == [
        "module-level P is mutable",
        "module-level Q is mutable",
    ]
    ok = """
A = MappingProxyType({2: 2.0, 3: 1.6})
B = frozenset({"a", "b"})
C = tuple(c for c in COLUMNS if c != "a")
D = (("x", "int64"), *((f"{s}_y", "Float64") for s in COLUMNS))
NO_CHIP = ChipScenario()
"""
    assert violations(ok, OPTIMIZE_DIR / "params.py", OPTIMIZE_TARGET) == []


def test_mutating_through_a_module_level_name_is_caught():
    """(b) A mutator call whose receiver is reached from a module-level name: subscripts,
    attributes and call results of it."""
    source = """
SEEN = MappingProxyType({"a": ()})
PARAMS = OptimizerParams()

def f(view):
    SEEN["a"].append(1)
    PARAMS.prune_n.update({1: 2})
    SEEN.get("a").clear()
    local = {}
    local["k"].append(1)
    return local
"""
    found = violations(source, BACKTEST_DIR / "policies.py", BACKTEST_TARGET)
    assert found == [
        "mutates module-level SEEN",
        "mutates module-level PARAMS",
        "mutates module-level SEEN",
    ]


def test_solver_modules_are_restricted_to_the_names_the_planner_uses():
    """(c) pulp/highspy are importable, but only the in-memory API the planner uses; file
    methods are forbidden on any object."""
    source = """
import pulp
import highspy as hs
from pulp import LpProblem, writeLP
from highspy import Highs

def solve(lp):
    lp.writeLP("model.lp")
    lp.writeMPS("model.mps")
    p2 = pulp.LpProblem.fromMPS("x.mps")
    data = lp.toJson("x.json")
    h = hs.Highs()
    h.readModel("model.mps")
    h.writeModel("out.mps")
    cbc = pulp.PULP_CBC_CMD()
    return pulp.LpProblem("ok"), hs.kHighsInf, data, p2, h, cbc
"""
    found = violations(source, OPTIMIZE_DIR / "model.py", OPTIMIZE_TARGET)
    expected = [
        "imports writeLP from pulp",
        "imports Highs from highspy",
        "calls .writeLP()",
        "calls .writeMPS()",
        "calls .fromMPS()",
        "calls .toJson()",
        "accesses highspy.Highs",
        "calls .readModel()",
        "calls .writeModel()",
        "accesses pulp.PULP_CBC_CMD",
    ]
    assert sorted(found) == sorted(expected)


def test_time_is_restricted_to_perf_counter():
    """(d) `time` for solve-time statistics only: no clocks that could steer a decision."""
    source = """
import time
from time import sleep

def f():
    start = time.perf_counter()
    now = time.time()
    time.sleep(1)
    return time.perf_counter() - start, now, time.localtime()
"""
    found = violations(source, OPTIMIZE_DIR / "search.py", OPTIMIZE_TARGET)
    assert sorted(found) == sorted(
        [
            "imports sleep from time",
            "accesses time.time",
            "accesses time.sleep",
            "accesses time.localtime",
        ]
    )


def test_restricted_modules_cannot_be_rebound():
    """A restricted module bound to another name (`clock = time`) or passed on would let its
    other attributes through unchecked (`clock.time()`), so it may only be used as
    `module.attr`."""
    source = """
import time
import pulp as pl

def f(run):
    clock = time
    solver = pl
    run(time)
    pair = (time, 1)
    return clock.time(), solver.PULP_CBC_CMD(), pair, time.perf_counter(), pl.LpProblem("x")
"""
    found = violations(source, OPTIMIZE_DIR / "search.py", OPTIMIZE_TARGET)
    assert sorted(found) == sorted(
        [
            "uses time other than as an attribute base",
            "uses pulp other than as an attribute base",
            "uses time other than as an attribute base",
            "uses time other than as an attribute base",
        ]
    )
