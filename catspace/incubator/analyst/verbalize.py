"""L6 -- VERBALIZE. Names and the idea grammar: records -> text, DETERMINISTICALLY. Every
sentence renders a field of the record; nothing is inferred here, so the prose is faithful by
construction. Evaluations are printed from White's point of view (the convention readers know)."""
from __future__ import annotations

import chess

from .dynamics import Effect
from .ideas import Candidate, IdeaRecord
from .relations import Relation

COLOR = {True: "White", False: "Black"}
PIECE = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}
PRIORITY = ("fork", "pinned_and_attacked", "hanging_through_pin", "disabled_attack", "disabled_defense",
            "pin", "skewer", "discovered_attack", "hanging", "attacked_by_lower_value",
            "battery", "overloaded", "xray", "attacks_king_zone", "loose")


def pc(sym: str, sq: int) -> str:
    if sym == "-":
        return chess.square_name(sq)
    return f"the {'white' if sym.isupper() else 'black'} {PIECE[sym.lower()]} on {chess.square_name(sq)}"


def rel_phrase(r: Relation) -> str:
    p = [pc(s, q) for s, q in zip(r.pieces, r.squares)]
    n = r.name
    if n == "hanging":
        return f"{p[0]} is hanging"
    if n == "loose":
        return f"{p[0]} is undefended"
    if n == "attacked_by_lower_value":
        return f"{p[0]} attacks {p[1]}, a bigger piece"
    if n == "pin":
        return f"{p[0]} pins {p[1]} to {p[2]}" + (" (absolute)" if "absolute" in r.tags else "")
    if n == "skewer":
        return f"{p[0]} skewers {p[1]} and {p[2]}" + (" with check" if "check" in r.tags else "")
    if n == "xray":
        return f"{p[0]} x-rays through {p[1]} to {p[2]}"
    if n == "fork":
        return f"{p[0]} forks " + " and ".join(p[1:])
    if n == "discovered_attack":
        return f"moving {p[1]} discovers an attack by {p[0]} on {p[2]}" + (" with check" if "check" in r.tags else "")
    if n == "battery":
        return f"{p[0]} and {p[1]} form a battery against {p[2]}"
    if n == "overloaded":
        return f"{p[0]} is overloaded, defending " + " and ".join(p[1:])
    if n == "disabled_attack":
        how = "cannot take" if "absolute" in r.tags else "cannot take without giving up"
        return f"{p[1]} {how} {p[3]}{'' if 'absolute' in r.tags else ' ' + p[2]}: it is pinned by {p[0]}" + (f" to {p[2]}" if "absolute" in r.tags else "")
    if n == "disabled_defense":
        return f"{p[1]} no longer really defends {p[3]}: it is pinned to {p[2]} by {p[0]}"
    if n == "pinned_and_attacked":
        return f"{p[1]} is pinned to {p[2]} by {p[0]} and attacked by {p[3]}: it cannot flee"
    if n == "hanging_through_pin":
        return f"{p[0]} is effectively hanging: its only defender{'s are' if len(p) > 2 else ' is'} pinned ({', '.join(p[1:])})"
    if n == "attacks_king_zone":
        return f"{p[0]} attacks {r.tags[0]} square(s) around {p[1]}"
    return f"{n} {' '.join(p)}"


def _top(rels: list[Relation], k: int) -> list[Relation]:
    return sorted(rels, key=lambda r: PRIORITY.index(r.name) if r.name in PRIORITY else 99)[:k]


def effect_phrase(e: Effect, k: int = 2) -> str:
    parts = []
    if e.captured:
        parts.append(f"{COLOR[e.mover]} takes the {e.captured[0]} ({e.captured[1]})")
    if e.gives_check:
        parts.append("with check")
    if e.created:
        parts.append("creating: " + "; ".join(rel_phrase(r) for r in _top(e.created, k)))
    if e.destroyed:
        parts.append("ending: " + "; ".join(rel_phrase(r) for r in _top(e.destroyed, k)))
    return f"{e.san} -- " + ", ".join(parts) if parts else ""


def cp_str(cp: int, side_white: bool, mate: int | None = None) -> str:
    if mate is not None:
        m = mate if side_white else -mate
        return f"#{m}" if m > 0 else f"#-{-m}"
    v = cp if side_white else -cp
    return f"{v/100:+.2f}"


def candidate_phrases(c: Candidate, n: int = 3, k: int = 2) -> str:
    return " ".join(p for p in (effect_phrase(e, k) for e in c.effects[:n]) if p)


def render(rec: IdeaRecord, *, verbose: bool = True) -> str:
    w = rec.side == "white"
    who = COLOR[w]
    num = rec.ply // 2 + 1
    out = [f"[{num}{'.' if w else '...'}] {who} to move, eval {cp_str(rec.eval_cp, w, rec.idea.mate if rec.idea else None)}"]
    if rec.played and rec.verdict in ("blunder", "mistake", "inaccuracy"):
        s = f"  {rec.played.san} was {'an' if rec.verdict[0] in 'aeiou' else 'a'} {rec.verdict}: eval {cp_str(rec.played.cp, w, rec.played.mate)} instead of {cp_str(rec.eval_cp, w)} with {rec.idea.san}."
        if rec.played.refutation:
            r = rec.played.refutation
            s += f" Punished by {r.san}: {r.line_san}." + (f" {candidate_phrases(r, 1)}" if candidate_phrases(r, 1) else "")
        out.append(s)
    for t in rec.threats[:2]:
        fe = effect_phrase(t.first_effects, 1) if t.first_effects else ""
        out.append(f"  If {who} does nothing: {t.san_line(chess.Board(rec.fen))}" + (f" -- {fe}" if fe else "") + f" (worth {t.gain_cp:+d} cp above a tempo).")
    if rec.zugzwang and rec.zugzwang.flagged:
        out.append(f"  Zugzwang: every move worsens {who}'s position (passing would be {rec.zugzwang.pass_cp - rec.zugzwang.best_cp:+d} cp better).")
    if rec.idea:
        s = f"  Idea: {rec.idea.san} -- {rec.idea.line_san}."
        ph = candidate_phrases(rec.idea)
        out.append(s + (f" {ph}" if ph else ""))
    for k in rec.killed[:3]:
        r = k.candidate.refutation
        s = f"  {k.candidate.san} no longer works after {k.killed_by} (was {cp_str(k.was_cp, w)}, now {cp_str(k.now_cp, w, k.candidate.mate)})"
        if r:
            s += f": refuted by {r.san}" + (f" -- {candidate_phrases(r, 1)}" if candidate_phrases(r, 1) else "")
        out.append(s + ".")
    if verbose:
        for a in rec.alternatives[:2]:
            r = a.refutation
            s = f"  {a.san} ({cp_str(a.cp, w, a.mate)}) fails"
            if r:
                s += f" to {r.san}" + (f" -- {candidate_phrases(r, 1)}" if candidate_phrases(r, 1) else "")
            out.append(s + ".")
    return "\n".join(out)
