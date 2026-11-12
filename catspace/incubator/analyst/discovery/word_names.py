"""PLAN 78.35 (2) NAMES: a word defined by a program is named FROM THE WORDS IT IS MADE OF, so the name already says something (a list of meaningless names was confusing). A word written by hand is
named by its author (word_registry.define requires it); this module only builds the name of the others and checks a name. Generic: no object of any game is named here.

  composed_name(term, used, *, max_parts=4, max_len=40) -> str
        the distinct relation names of the term's literals, in order of first occurrence, joined by "_" (a registered word's name is ONE part: it is not expanded); at most `max_parts` parts (the
        first ones) and `max_len` characters (cut at a part boundary, at least one part); the result is a legal relation symbol `[a-z][a-z0-9_]*`; when it is in `used` (the words and the relations
        of the description) `_2`, `_3` ... is appended, the first free. Deterministic.
  check_name(name, used)  raises ValueError when `name` is not a legal relation symbol or is already in `used`
"""
from __future__ import annotations

import re

from catspace.incubator.analyst.discovery import word_terms as WT

LEGAL = re.compile(r"[a-z][a-z0-9_]*$")


def check_name(name: str, used) -> None:
    if not isinstance(name, str) or not LEGAL.match(name):
        raise ValueError(f"{name!r} is not a legal relation symbol (lowercase letters, digits and _, starting with a letter)")
    if name in set(used):
        raise ValueError(f"the name {name} is already a word or a relation of the description")


def composed_name(term: tuple, used, *, max_parts: int = 4, max_len: int = 40) -> str:
    used = set(used)
    parts = [p for p in WT.relations(term) if LEGAL.match(p)][:max(1, max_parts)]
    if not parts:
        parts = ["word"]
    while len(parts) > 1 and len("_".join(parts)) > max_len:
        parts.pop()
    base = "_".join(parts)
    name, k = base, 2
    while name in used:
        name, k = f"{base}_{k}", k + 1
    return name
