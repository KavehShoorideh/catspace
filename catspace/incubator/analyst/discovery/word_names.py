"""PLAN 78.35 (2) NAMES: a word built from words is given a short NAME from a static list, as storms are (rules/concepts/names.txt: one lowercase, pronounceable, proper-name-like word per
line, in a fixed order, the alphabet cycling a b c .. z a b ..). A name carries no meaning (so none is wrong); the registry (word_registry.py) says what it is. The list is static and
deterministic: the same set of used names always gives the same next name. Generic: no object of any game is named here; a name must not be a relation of the description (the registry
checks that, with the vocabulary's composed text).

  names() -> [name ..]                  the list, in its order
  next_name(used) -> the first name of the list not in `used`; NameError when the list is used up
"""
from __future__ import annotations

import re
from functools import lru_cache

from catspace.io import paths

NAMES_FILE = paths.REPO_ROOT / "rules" / "concepts" / "names.txt"
LEGAL = re.compile(r"[a-z][a-z0-9_]*$")


class NamesExhausted(NameError):
    pass


@lru_cache(maxsize=1)
def _read() -> tuple:
    out = []
    for line in NAMES_FILE.read_text().splitlines():
        w = line.split("#", 1)[0].strip()
        if w:
            out.append(w)
    return tuple(out)


def names() -> list:
    return list(_read())


def next_name(used) -> str:
    """The first name of the list that is not in `used` (any container of names)."""
    used = set(used)
    for n in _read():
        if n not in used:
            return n
    raise NamesExhausted(f"all {len(_read())} names of {NAMES_FILE.name} are in use; the list must be extended")
