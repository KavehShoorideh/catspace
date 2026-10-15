"""CHAIN LEMMAS by chunking (Kaveh 2026-10-15; docs/ANALYST_PLAN.md 40.3). The event loop (rust/analyst_core/src/reasoning/chain.rs) records, for a question with a SHAPE (SAFE, DEFLECT,
POKE), its PREMISE -- a canonical graph of the pieces it touches, relations from the vocabulary only (attacks, the piece kind, the side relative to the actor), squares abstracted so a mirror or a
translation gives the same string -- and its OUTCOME (the kinds of the chain below it in order, and its value). This module keeps the store, data/derived/chain_lemmas.json:

  lemmas[premise] = {id, kind, outcomes: {outcome: {count, value, evidence: [fens]}}, promoted, demoted, outcome, count, value, contradictions: [...]}

A premise whose chains all settled the same way, at least `threshold` times (a knob, never below 5), is PROMOTED: the question is then answered by it in one step (a stake above the stake knob,
and every few matches as a held-out check, are still run and compared). A lemma a chain contradicts is DEMOTED and the contradiction kept (a surprise record). STATUS OF EVERY LEMMA:
CONJECTURE -- no Lean proof exists for any of them (AGENTS rule 11 asks for one before a lemma is relied on; this phase only measures).
"""
from __future__ import annotations

import json
from pathlib import Path

from catspace.io import paths

STATUS = "CONJECTURE (no Lean proof)"
MIN_THRESHOLD = 5


def default_path() -> str:
    return paths.derived("chain_lemmas.json")


def load(path: str | None = None) -> dict:
    p = Path(path or default_path())
    if p.exists():
        return json.loads(p.read_text())
    return {"status": STATUS, "threshold": None, "lemmas": {}}


def merge(reports: list, path: str | None = None, threshold: int = 20) -> dict:
    """Fold the observations (and contradictions) of ledger sessions (`Ledger.lemma_report()`) into the store, promote, save. -> counts."""
    threshold = max(MIN_THRESHOLD, int(threshold))
    store = load(path)
    lem = store["lemmas"]
    for rep in reports:
        for pk, outs in rep.get("observations", {}).items():
            e = lem.setdefault(pk, {"id": f"L{len(lem) + 1}", "kind": pk.split("|")[0], "outcomes": {}, "promoted": False, "demoted": False, "contradictions": []})
            for sig, o in outs.items():
                s = e["outcomes"].setdefault(sig, {"count": 0, "value": o.get("value"), "evidence": []})
                s["count"] += o["count"]
                for f in o.get("evidence", []):
                    if f not in s["evidence"] and len(s["evidence"]) < 5:
                        s["evidence"].append(f)
        for c in rep.get("contradictions", []):
            e = lem.get(c["premise"])
            if e is not None:
                e["demoted"] = True
                e["contradictions"].append({k: c[k] for k in ("expected", "got", "fen")})
    n_new = 0
    for pk, e in lem.items():
        outs = e["outcomes"]
        top = max(outs, key=lambda k: outs[k]["count"])
        total = sum(o["count"] for o in outs.values())
        same = outs[top]["count"] == total              # it settled the same way every time
        was = e["promoted"]
        e["promoted"] = bool(same and total >= threshold and not e["demoted"])
        if e["promoted"]:
            e.update(outcome=top, count=total, value=outs[top]["value"])
        if e["promoted"] and not was:
            n_new += 1
        if not same and was:
            e["demoted"] = True
            e["contradictions"].append({"expected": e.get("outcome"), "got": "a different outcome observed", "fen": ""})
            e["promoted"] = False
    store.update(status=STATUS, threshold=threshold, lemmas=lem)
    Path(path or default_path()).write_text(json.dumps(store, indent=1))
    return summary(store)


def summary(store: dict) -> dict:
    lem = store["lemmas"]
    return dict(candidates=len(lem), promoted=sum(1 for e in lem.values() if e["promoted"]), demoted=sum(1 for e in lem.values() if e["demoted"]),
                contradictions=sum(len(e["contradictions"]) for e in lem.values()), observations=sum(o["count"] for e in lem.values() for o in e["outcomes"].values()))


def describe(e: dict) -> str:
    out = e.get("outcome") or max(e["outcomes"], key=lambda k: e["outcomes"][k]["count"])
    return f"{e['id']} [{e['kind']}] settles as '{out}' in {sum(o['count'] for o in e['outcomes'].values())} chains ({'promoted' if e['promoted'] else 'candidate'}; {STATUS})"
