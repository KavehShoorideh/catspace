"""PROOF TERMS AND A KERNEL (Kaveh 2026-09-20: "take inspiration from proof assistant software"). The searcher
(the ledger, the goal search) PRODUCES a term: our move, and for every legal reply a justification -- a leaf
priced by the ledger (a HOLE: assumed, not proven), a mate by the rules (proven), or a LEMMA application (a family
of replies one answer refutes). The KERNEL checks a term knowing only the rules -- legality, completeness of the
replies, that each lemma's members really are what the lemma says and that its answer is legal after each --
and reports what is proven and what rests on holes. The searcher can be wrong; the kernel does not trust it."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .lemmas import discovered_check_families


@dataclass
class Just:
    kind: str                       # "leaf" (a ledger price: a hole) | "mate" (proven by the rules) | "lemma"
    value: float = 0.0
    lemma_text: str = ""
    answer: str = ""                # uci, for a lemma
    family: tuple = ()              # (mover square, opener square) for a discovered-check lemma


@dataclass
class Term:
    fen: str
    move: str                       # uci
    replies: dict = field(default_factory=dict)      # reply uci -> Just
    claim: float = 0.0              # the guaranteed gain the searcher claims (the min over replies)

    def holes(self) -> int:
        return sum(1 for j in self.replies.values() if j.kind == "leaf")

    def render(self) -> list[str]:
        b = chess.Board(self.fen); m = chess.Move.from_uci(self.move); nb = b.copy(stack=False); nb.push(m)
        out = [f"claim: after {b.san(m)} the ledger guarantees {self.claim:+g} against every reply ({len(self.replies)} replies: {self.holes()} priced leaves, "
               f"{sum(1 for j in self.replies.values() if j.kind == 'mate')} mates by the rules, {sum(1 for j in self.replies.values() if j.kind == 'lemma')} closed by lemma)"]
        seen = set()
        for u, j in self.replies.items():
            if j.kind == "lemma" and j.lemma_text not in seen:
                seen.add(j.lemma_text); out.append(f"  lemma: {j.lemma_text}")
        worst = min(self.replies.items(), key=lambda kv: kv[1].value, default=None)
        if worst:
            out.append(f"  the worst reply for us: {nb.san(chess.Move.from_uci(worst[0]))} -> {worst[1].value:+g} ({worst[1].kind})")
        return out


def check(term: Term) -> dict:
    """The kernel. Returns dict(ok, errors, proven_replies, holes)."""
    errors = []
    b = chess.Board(term.fen); m = chess.Move.from_uci(term.move)
    if m not in b.legal_moves:
        return dict(ok=False, errors=[f"{term.move} is not legal"], proven=0, holes=0)
    nb = b.copy(stack=False); nb.push(m)
    legal = {r.uci() for r in nb.legal_moves}
    missing = legal - set(term.replies); extra = set(term.replies) - legal
    if missing:
        errors.append(f"replies not covered: {sorted(missing)[:5]}")
    if extra:
        errors.append(f"replies that are not legal: {sorted(extra)[:5]}")
    fams = {(f.mover, f.opener): f for f in discovered_check_families(nb)}
    proven = 0
    for u, j in term.replies.items():
        if u not in legal:
            continue
        r = chess.Move.from_uci(u); after = nb.copy(stack=False); after.push(r)
        if j.kind == "mate":
            if not after.is_checkmate():
                errors.append(f"{u}: claimed mate, not mate")
            else:
                proven += 1
        elif j.kind == "lemma":
            fam = fams.get(tuple(j.family))
            if fam is None or r not in fam.members:
                errors.append(f"{u}: not a member of the claimed family {j.family}"); continue
            a = chess.Move.from_uci(j.answer)
            if a not in after.legal_moves:
                errors.append(f"{u}: the lemma's answer {j.answer} is not legal after it"); continue
            if a.to_square != fam.opener:
                errors.append(f"{u}: the answer does not capture the opener"); continue
            if not after.is_check():
                errors.append(f"{u}: claimed a discovered check, no check"); continue
            proven += 1
        elif j.kind != "leaf":
            errors.append(f"{u}: unknown justification {j.kind}")
    # the claim must not exceed the minimum of the justifications' values
    if term.replies and term.claim > min(j.value for j in term.replies.values()) + 1e-9:
        errors.append("the claim exceeds the minimum over the replies")
    return dict(ok=not errors, errors=errors, proven=proven, holes=term.holes())


def render_check(term: Term) -> list[str]:
    c = check(term)
    return term.render() + [f"  kernel: {'OK' if c['ok'] else 'REJECTED'}; {c['proven']} replies proven by the rules, {c['holes']} rest on the ledger's prices" + (f"; errors: {c['errors']}" if c["errors"] else "")]
