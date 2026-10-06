"""RULES: the watcher's knowledge. A rule has a KIND (what it forecasts), a CONDITION in a small language over
attributes (frames.attributes for a frame; position attributes for 'elsewhere' and 'result'), a PARENT (an exception
refines its parent and applies only where the parent does: a Ripple-Down tree), its CORNERSTONE positions, and an
English text. Its forecast is the COUNT of what happened each time its condition held, smoothed toward its parent's
forecast: a new rule with few cases says what its parent says (Kaveh 2026-10-05: "I don't expect the first rule to
immediately reduce surprise").

Kinds and their outcome buckets:
  frame      the mover's material change on one frame's square until the exchanges settle: -10..+10
  elsewhere  the mover's material change on squares that were not frames: -10..+10
  result     the game's result for the mover: loss, draw, win
  attack     after a SACRIFICE (watch/cases.py), whether its material comes back within 10 plies and stands: fails, pays

Selection among the rules that apply (Kaveh 2026-10-05): the more specific supersedes the less specific and the
older; a learned prior over rules comes later (`select` is the slot: POLICIES). Every applicable rule's probability of
what happened is logged, the training data for that prior.

Condition language (JSON): {"all": [c, ...]}, {"any": [c, ...]}, {"not": c}, {"flag": attr},
{"is": attr, "eq": value}, {"cmp": attr, "op": "<"|"<="|"=="|">="|">", "value": number | {"attr": name}}.
A REFERENCE (optional, material kinds): {"attr": name, "sign": 1 | -1} or {"sum": [terms]}. The rule then counts the
outcome's OFFSET from the reference ("the exchange goes as the exchange count says" is one rule for every piece
value); its forecast is that offset distribution shifted to the frame's reference, smoothed toward the parent in
pawns.
The rule file is append-only (artifacts/watch/rules.jsonl); counts are state, rebuilt by replaying the stream."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from catspace.io import paths

KINDS = {"frame": list(range(-10, 11)), "elsewhere": list(range(-10, 11)), "result": ["loss", "draw", "win"],
         "attack": ["fails", "pays"]}
ROOTS = {"frame": "F0", "elsewhere": "E0", "result": "R0", "attack": "A0"}
ALPHA = 4.0                                                                  # the parent's weight, in cases, in a rule's forecast
TRUST = 20.0                                                                 # evidence_blend: a rule is trusted n/(n+TRUST) against the rule it supersedes
OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b, ">=": lambda a, b: a >= b, ">": lambda a, b: a > b}


def record_root() -> Path:
    import os
    return Path(os.environ.get("CATSPACE_WATCH_RECORD") or (paths.REPO_ROOT / "artifacts" / "watch"))


def rules_path() -> Path:
    return record_root() / "rules.jsonl"


@dataclass
class Rule:
    id: str
    kind: str
    parent: str | None
    cond: dict
    text: str
    created: int                                                             # order of creation: newer supersedes older
    cornerstones: list = field(default_factory=list)
    author: str = "claude"
    time: str = ""
    ref: dict | None = None                                                  # forecast relative to this attribute expression


class LanguageError(ValueError):
    pass


def check(cond: dict, attrs: set[str] | None = None) -> None:
    """Raise LanguageError unless cond is a well-formed condition (over `attrs` when given)."""
    if not isinstance(cond, dict) or len(cond) == 0:
        raise LanguageError(f"not a condition: {cond!r}")
    if "all" in cond or "any" in cond:
        k = "all" if "all" in cond else "any"
        if set(cond) != {k} or not isinstance(cond[k], list):
            raise LanguageError(f"bad {k}: {cond!r}")
        for c in cond[k]:
            check(c, attrs)
    elif "not" in cond:
        if set(cond) != {"not"}:
            raise LanguageError(f"bad not: {cond!r}")
        check(cond["not"], attrs)
    elif "flag" in cond:
        if set(cond) != {"flag"} or (attrs is not None and cond["flag"] not in attrs):
            raise LanguageError(f"bad flag: {cond!r}")
    elif "is" in cond:
        if set(cond) != {"is", "eq"} or (attrs is not None and cond["is"] not in attrs):
            raise LanguageError(f"bad is: {cond!r}")
    elif "cmp" in cond:
        if set(cond) != {"cmp", "op", "value"} or cond["op"] not in OPS or (attrs is not None and cond["cmp"] not in attrs):
            raise LanguageError(f"bad cmp: {cond!r}")
        v = cond["value"]
        if isinstance(v, dict) and (set(v) != {"attr"} or (attrs is not None and v["attr"] not in attrs)):
            raise LanguageError(f"bad cmp value: {cond!r}")
    else:
        raise LanguageError(f"unknown condition: {cond!r}")


def holds(cond: dict, a: dict) -> bool:
    if "all" in cond:
        return all(holds(c, a) for c in cond["all"])
    if "any" in cond:
        return any(holds(c, a) for c in cond["any"])
    if "not" in cond:
        return not holds(cond["not"], a)
    if "flag" in cond:
        return bool(a[cond["flag"]])
    if "is" in cond:
        return a[cond["is"]] == cond["eq"]
    v = cond["value"]
    rhs = a[v["attr"]] if isinstance(v, dict) else v
    return OPS[cond["op"]](a[cond["cmp"]], rhs)


class RuleBase:
    """The rules (from the append-only file) and their counts (state)."""

    def __init__(self, rules: list[Rule], counts: dict | None = None):
        self.rules = {r.id: r for r in rules}
        for kind, rid in ROOTS.items():
            if rid not in self.rules:
                self.rules[rid] = Rule(rid, kind, None, {"all": []}, f"the base rate ({kind})", -1)
        self.children: dict = {}
        for r in self.rules.values():
            self.children.setdefault(r.parent, []).append(r.id)
        self.counts = counts if counts is not None else {}
        for rid, r in self.rules.items():
            self.counts.setdefault(rid, [0] * len(KINDS[r.kind]))

    @classmethod
    def load(cls, counts: dict | None = None, upto: int | None = None) -> "RuleBase":
        """The rules in the file (those created before `upto`, when given: a batch uses the rules that existed when
        it started)."""
        rules = []
        p = rules_path()
        if p.exists():
            retired = set()
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    if "retire" in d:                                         # Kaveh's gate: a retired rule (and every
                        retired.add(d["retire"])                             # exception under it) no longer forecasts
                    elif upto is None or d["created"] < upto:
                        rules.append(Rule(**d))
            gone = set(retired)
            while True:
                more = {r.id for r in rules if r.parent in gone} - gone
                if not more:
                    break
                gone |= more
            rules = [r for r in rules if r.id not in gone]
        return cls(rules, counts)

    def matching(self, kind: str, a: dict) -> list[str]:
        """Every rule of `kind` that applies: its condition holds and so does its parent's, up to the root."""
        out = []
        stack = [ROOTS[kind]]
        while stack:
            rid = stack.pop()
            if rid == ROOTS[kind] or holds(self.rules[rid].cond, a):
                out.append(rid)
                stack.extend(self.children.get(rid, []))
        return out

    def dist(self, rid: str, a: dict | None = None) -> list[float]:
        """The rule's forecast for attributes `a`: its counts (offsets from its reference, when it has one) shifted to
        the frame's reference and smoothed toward its parent's forecast (the root's toward uniform)."""
        r = self.rules[rid]
        k = len(KINDS[r.kind])
        prior = self.dist(r.parent, a) if r.parent else [1.0 / k] * k
        c = self.counts[rid]
        n = sum(c)
        if not r.ref:
            return [(c[i] + ALPHA * prior[i]) / (n + ALPHA) for i in range(k)]
        shift = ref_value(r.ref, a)
        out = []
        for i, x in enumerate(KINDS[r.kind]):
            j = x - shift
            out.append(((c[j + 10] if -10 <= j <= 10 else 0) + ALPHA * prior[i]) / (n + ALPHA))
        z = sum(out)
        return [v / z for v in out]

    def select(self, kind: str, a: dict, policy: str = "specific_newest") -> tuple[str, list[str]]:
        """-> (the rule that forecasts, every rule that applies)."""
        m = self.matching(kind, a)
        return POLICIES[policy](self, m), m

    def forecast(self, kind: str, a: dict, policy: str = "evidence_blend") -> tuple[list[float], str, list[str], dict]:
        """-> (the forecast, the rule that leads it, every rule that applies, how it was made). evidence_blend (Kaveh
        2026-10-05, a simple prior until a learned one): the leading rule (more specific, then newer) is trusted
        n/(n+TRUST) by its own cases n against the rule it supersedes (the leader once it is set aside)."""
        m = self.matching(kind, a)
        lead = _specific_newest(self, m)
        d = self.dist(lead, a)
        if policy != "evidence_blend":
            return d, lead, m, dict(lead=lead)
        rest = [r for r in m if r != lead]
        if not rest:
            return d, lead, m, dict(lead=lead)
        prev = _specific_newest(self, rest)
        n = sum(self.counts[lead])
        w = n / (n + TRUST)
        dp = self.dist(prev, a)
        return [w * x + (1 - w) * y for x, y in zip(d, dp)], lead, m, dict(lead=lead, superseded=prev, weight=w)

    def update(self, matched: list[str], outcome, a: dict | None = None, weight: dict | float = 1.0) -> None:
        """Count the outcome for every rule that applied (each rule estimates its own condition's frequencies); a rule
        with a reference counts the offset from it. `weight` (per rule, or one number) is the case's share of an
        observation: a game's result is ONE observation however many positions saw it."""
        for rid in matched:
            r = self.rules[rid]
            w = weight.get(rid, 1.0) if isinstance(weight, dict) else weight
            if r.ref:
                self.counts[rid][bucket(r.kind, outcome - ref_value(r.ref, a)) + 10] += w
            else:
                self.counts[rid][KINDS[r.kind].index(outcome)] += w


def _specific_newest(rb: RuleBase, matched: list[str]) -> str:
    """The more specific supersedes the less specific and the older (Kaveh 2026-10-05): among the applying rules with
    no applying exception under them, the newest."""
    ms = set(matched)
    leaves = [rid for rid in matched if not any(c in ms for c in rb.children.get(rid, []))]
    return max(leaves, key=lambda rid: rb.rules[rid].created)


POLICIES = {"specific_newest": _specific_newest}                            # the slot for a learned prior over rules


def ref_value(ref: dict, a: dict) -> int:
    terms = ref["sum"] if "sum" in ref else [ref]
    return int(sum(t.get("sign", 1) * a[t["attr"]] for t in terms))


def bucket(kind: str, x) -> object:
    if kind in ("result", "attack"):
        return x
    return max(-10, min(10, int(x)))


def add_rule(kind: str, parent: str, cond: dict, text: str, cornerstones: list, attrs: set[str] | None = None,
             ref: dict | None = None) -> Rule:
    """Append a rule to the file (validated). Its id is the kind's letter and a running number."""
    check(cond, attrs)
    if ref is not None:
        terms = ref["sum"] if "sum" in ref else [ref]
        if kind in ("result", "attack") or not all(set(t) <= {"attr", "sign"} and "attr" in t and (attrs is None or t["attr"] in attrs) for t in terms):
            raise LanguageError(f"bad reference: {ref!r}")
    existing = RuleBase.load()
    if parent not in existing.rules or existing.rules[parent].kind != kind:
        raise LanguageError(f"parent {parent} is not a {kind} rule")
    created = max([r.created for r in existing.rules.values()] + [0]) + 1
    rid = f"{ROOTS[kind][0]}{created}"
    r = Rule(rid, kind, parent, cond, text, created, cornerstones, "claude", time.strftime("%Y-%m-%d %H:%M %Z"), ref)
    p = rules_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(asdict(r), sort_keys=True) + "\n")
    return r


def retire(rid: str, note: str, by: str = "kaveh") -> None:
    """Retire a rule (append-only: the rule stays in the file, a retire line follows it)."""
    p = rules_path()
    with open(p, "a") as fh:
        fh.write(json.dumps(dict(retire=rid, note=note, by=by, time=time.strftime("%Y-%m-%d %H:%M %Z")), sort_keys=True) + "\n")


def bits(p: float) -> float:
    return -math.log2(max(p, 1e-12))
