"""catspace -- the wrapper: it wires the research components together and exposes
them to deployment.

Layout (see repo_structure.md):
  catspace/interfaces.py   the Protocols every component approach must satisfy
  catspace/registry.py     resolve "component:approach" -> implementation
  catspace/approaches/     end-to-end configs (which planner/search/encoder/memory)
  catspace/research/       components, tools, docs, infra -- the research tree
  catspace/deployment/     server + web, consumes an end-to-end config
"""
from __future__ import annotations

from catspace import registry
from catspace.interfaces import MovePrior, Region, SearchOutcome, SubgoalSelector, ValueModel

__all__ = [
    "registry",
    "Region",
    "SearchOutcome",
    "ValueModel",
    "MovePrior",
    "SubgoalSelector",
]
