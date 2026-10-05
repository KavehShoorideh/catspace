"""TRANSLATE: guards (and the hand-registered Lean ideas) into candidates in the handoff plan's schema -- named
hypotheses, a footprint, a protagonist, a bounded claim, a sentence. Day one, a mined guard becomes a Python
candidate (its frame literals and its exception terms are the hypotheses, evaluated by lemma_apply; the footprint is
the squares of an embedding of the frame); research slot R2 (the guard-to-Lean translator) gives it a Lean statement."""
from __future__ import annotations

from .. import basis
from ..contracts import Candidate, Claim, to_dict
from ..registry import REGISTERED


def registered(entry: dict, ctx) -> Candidate:
    return Candidate(entry["id"], [dict(name=h, text=h, lean=h) for h in entry["hypotheses"]], entry["footprint"], entry["protagonist"],
                     Claim(**entry["claim"]), None,
                     dict(theorem=entry["theorem"], detector=entry["detector"], tier=entry["tier"], bridges=entry["bridges"]),
                     "registered", entry["sentence"], ctx.prov("translate", [entry["theorem"]]))


def mined(g: dict, ctx) -> Candidate:
    from ...lemma_gen import guard_text, lit_text
    hyps = [dict(name=lit_text(r, tuple(sl), pol), text=lit_text(r, tuple(sl), pol), lean=None) for r, sl, pol in g["literals"]]
    terms = [tuple((r, tuple(sl), pol) for r, sl, pol in t) for t in g["terms"]]
    if terms:
        hyps.append(dict(name="no exception", text="unless " + guard_text(terms), lean=None))
    size = len(g["types"])
    hv = (g.get("value") or {}).get("holds")
    claim = Claim("material", size, float(hv) if hv is not None else 0.0, f"{g['claim']} (the frame's own outcome over its {size} plies)")
    actor = g["types"][g["slot"]]
    sentence = f"[{' '.join(g['types'])}] moving the {actor} {g['claim'].replace('acting ', '')} material" + (f", unless {guard_text(terms)}" if terms else "")
    return Candidate("mined:" + g["id"], hyps, "the squares of an embedding of the frame (lemma_apply)", f"slot {g['slot']} ({actor})",
                     claim, g["id"], None, "mined", sentence, ctx.prov("translate", [g["id"]]),
                     origin=dict(library=g["library"], index=g["index"]))


def run(ctx) -> dict:
    cfg = ctx.cfg("translate")
    active = basis.active(at=ctx.basis_before)                               # the basis the round started from
    cands = [registered(e, ctx) for e in REGISTERED if e["id"] not in active or cfg.get("recheck_registered", True)]
    for g in (ctx.load("mine") or {}).get("guards", []):
        cands.append(mined(g, ctx))
    ctx.log(f"OBS translate: {sum(c.source == 'registered' for c in cands)} registered (Lean), {sum(c.source == 'mined' for c in cands)} mined (Python)")
    return dict(candidates=[to_dict(c) for c in cands])
