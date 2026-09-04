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


_SYMVAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
_SUBJ = {"hanging": 0, "loose": 0, "attacked_by_lower_value": 1, "pin": 1, "skewer": 1, "xray": 1, "fork": 0,
         "discovered_attack": 2, "battery": 2, "overloaded": 0, "disabled_attack": 3, "disabled_defense": 3,
         "pinned_and_attacked": 1, "hanging_through_pin": 0}


def _top(rels: list[Relation], k: int) -> list[Relation]:
    def key(r):
        sym = r.pieces[_SUBJ.get(r.name, 0)] if r.pieces else "-"
        return (PRIORITY.index(r.name) if r.name in PRIORITY else 99, -_SYMVAL.get(sym.lower(), 0))
    return sorted(rels, key=key)[:k]


RESOLVED_WORD = {"hanging": "which was hanging", "pinned_and_attacked": "which could not flee",
                 "hanging_through_pin": "whose defenders were pinned", "fork": "the forking piece",
                 "loose": "which was undefended"}


def effect_phrase(e: Effect, k: int = 2) -> str:
    parts = []
    if e.captured:
        order = ("pinned_and_attacked", "hanging_through_pin", "hanging", "loose", "fork")
        why = [RESOLVED_WORD[r.name] for r in sorted(e.resolved, key=lambda r: order.index(r.name) if r.name in order else 99)
               if r.name in RESOLVED_WORD]
        parts.append(f"{COLOR[e.mover]} takes the {e.captured[0]} ({e.captured[1]})" + (f", {why[0]}" if why else ""))
    if e.gives_check:
        parts.append("with check")
    if e.uses:
        def use_phrase(r):
            if r.name in ("discovered_attack", "battery"):
                return rel_phrase(r).replace("moving " + pc(r.pieces[1], r.squares[1]) + " discovers", "discovering")
            return "cashing: " + rel_phrase(r)
        parts.append("; ".join(use_phrase(r) for r in _top(e.uses, 1)))
    if e.created:
        parts.append("creating: " + "; ".join(rel_phrase(r) for r in _top(e.created, k)))
    if e.escaped:
        parts.append("escaping: " + "; ".join(rel_phrase(r) for r in _top(e.escaped, 1)))
    if e.destroyed:
        parts.append("ending: " + "; ".join(rel_phrase(r) for r in _top(e.destroyed, 1)))
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
