"""USE: how each promoted idea is wired into the player, and the refusals. The hard rule, enforced here: a CLOSURE
(an idea's conclusion used as a search value) needs tier unconditional or forcing, a Lean theorem with standard
axioms and a clean bridge that checked at least one claim. A conjecture-tier idea may only ORDER (a tie-break among
the moves the tactics tie) or speak in a SENTENCE. The wiring for the player -- this round's accepted uses plus those
of every idea already active in the basis -- goes to active.json, which frame_puzzles reads with --loop-active."""
from __future__ import annotations

from .. import basis, registry
from ..contracts import Promoted, UseRecord, from_dict, to_dict
from ..store import write_json


def closure_ok(p: Promoted) -> tuple[bool, str]:
    ev = p.evidence
    br = ev.bridge or {}
    if p.unsound:
        return False, f"unsound: {p.reason}"
    if p.tier not in ("unconditional", "forcing"):
        return False, f"tier {p.tier}: a closure needs a proof"
    if not ev.theorem or not ev.axioms_ok:
        return False, "no Lean theorem with standard axioms"
    if br.get("wrong", 1) != 0 or br.get("checked", 0) <= 0:
        return False, f"bridge {br.get('checked', 0)} checked, {br.get('wrong')} wrong"
    return True, "proved, standard axioms, clean bridge"


def requested(p: Promoted) -> list[tuple[str, str]]:
    c = p.candidate
    if c.source == "registered":
        reg = registry.by_id(c.id) or {}
        return list(reg.get("uses", [])) + [("sentence", "explain: " + c.sentence)]
    if c.lean and c.lean.get("uses"):                                         # a translated idea that asks for its uses
        return [tuple(u) for u in c.lean["uses"]] + [("sentence", "explain: " + c.sentence)]
    return [("order", "frame_player.LOOP_ORDER: a tie-break among the moves the tactics tie, by the lemmas' composed verdicts"),
            ("sentence", "explain: " + c.sentence)]


def run(ctx) -> dict:
    out = ctx.load("prove") or {}
    promoted = [from_dict(Promoted, p) for p in out.get("promoted", [])]
    records = []
    for p in promoted:
        for mode, switch in requested(p):
            refused, reason = False, ""
            if mode == "closure":
                ok, why = closure_ok(p)
                refused, reason = (not ok), why
            records.append(UseRecord(p.candidate.id, mode, switch, refused, reason, ctx.prov("use", [p.candidate.id])))
    for r in records:
        if r.refused:
            ctx.log(f"REFUSED {r.mode} for {r.idea}: {r.reason}")
    active = wiring(ctx.round, promoted, records, before=basis.active(at=ctx.basis_before))
    write_json(ctx.dir / "active.json", active)
    ctx.log(f"OBS use: {sum(not r.refused for r in records)} uses accepted, {sum(r.refused for r in records)} refused; "
            f"active: closures {active['closures']}, order {len(active['order'])}")
    return dict(uses=[to_dict(r) for r in records], active=active, active_path=str(ctx.dir / "active.json"))


def wiring(round_: int, promoted: list[Promoted], records: list[UseRecord], before: dict | None = None) -> dict:
    """The player's wiring: the accepted uses of this round's ideas and of every idea active in the basis the round
    started from (`before`; None: today's basis). An idea this round re-checked and refused a closure loses its old
    closure too (an unsound re-check used to leave it wired: BUGS 2026-10-05); each idea is wired once, this round's
    record over an older one."""
    before = basis.active() if before is None else before
    accepted = {(r.idea, r.mode) for r in records if not r.refused}
    refused_now = {r.idea for r in records if r.refused and r.mode == "closure"}
    closures, order, sentences = set(), {}, {}
    for e in before.values():                                                # earlier rounds
        for u in e.get("uses", []):
            if u["refused"] or (u["mode"] == "closure" and u["idea"] in refused_now):
                continue
            if u["mode"] == "closure":
                closures.add(u["idea"])
            elif u["mode"] == "order":
                c = e["record"]["candidate"]
                order[c["id"]] = dict(id=c["id"], guard=c["guard"], **(c.get("origin") or {}))
    by_id = {p.candidate.id: p for p in promoted}
    for idea, mode in sorted(accepted):
        p = by_id[idea]
        if mode == "closure":
            closures.add(idea)
        elif mode == "order":
            order[idea] = dict(id=idea, guard=p.candidate.guard, **(p.candidate.origin or {}))
        elif mode == "sentence":
            sentences[idea] = p.candidate.sentence
    return dict(round=round_, closures=sorted(closures), order=list(order.values()), sentences=sentences)
