"""catspace/registry.py -- resolve a component approach from its name.

End-to-end configs name approaches as strings ("encoder:jepa_tokenizer"); only this
module imports approach modules, so a config never hard-codes a deep import path and
approaches can move without touching call sites.

An approach is discovered, not registered: any directory under
research/components/<component>/approaches/<name>/ containing src/__init__.py is an
approach. Its src package must expose `build(**kwargs)` returning an object that
satisfies the component's Protocol in catspace.interfaces.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

COMPONENTS = ("encoder", "planner", "search", "memory")

_COMPONENTS_DIR = Path(__file__).resolve().parent / "research" / "components"


def _approach_dir(component: str, name: str) -> Path:
    return _COMPONENTS_DIR / component / "approaches" / name


@lru_cache(maxsize=None)
def list_approaches(component: str) -> tuple[str, ...]:
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}; expected one of {COMPONENTS}")
    base = _COMPONENTS_DIR / component / "approaches"
    if not base.is_dir():
        return ()
    return tuple(sorted(
        p.name for p in base.iterdir()
        if p.is_dir() and (p / "src" / "__init__.py").exists()))


def module_path(component: str, name: str) -> str:
    return f"catspace.research.components.{component}.approaches.{name}.src"


def load(component: str, name: str):
    """Import an approach's src package."""
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}; expected one of {COMPONENTS}")
    if not _approach_dir(component, name).is_dir():
        raise LookupError(
            f"no approach {component}:{name}; available: {list_approaches(component) or '(none)'}")
    return importlib.import_module(module_path(component, name))


def build(spec: str, **kwargs):
    """Build an approach from a "component:name" spec, e.g. "search:puct_mcts"."""
    component, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"spec must be 'component:approach', got {spec!r}")
    mod = load(component, name)
    factory = getattr(mod, "build", None)
    if factory is None:
        raise AttributeError(f"{module_path(component, name)} defines no build()")
    return factory(**kwargs)


def inventory() -> dict[str, tuple[str, ...]]:
    return {c: list_approaches(c) for c in COMPONENTS}
