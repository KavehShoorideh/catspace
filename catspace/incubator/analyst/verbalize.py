"""L6 -- VERBALIZE. Names and the idea grammar: records -> text, DETERMINISTICALLY. Every
sentence renders a field of the record; nothing is inferred here, so the prose is faithful by
construction. Evaluations are printed from White's point of view (the convention readers know)."""
from __future__ import annotations

import chess

from .engine import MATE_CP

from .dynamics import Effect
from .ideas import Candidate, IdeaRecord
from .relations import Relation, against
from .mates import mate_in_line
from .sequences import find_sequences
MOTIF = {"deflection": "deflection", "attraction": "attraction", "interference": "interference", "clearance": "clearance",
         "removal_of_guard": "removal of the guard"}


def motif_phrase(board: chess.Board, pv) -> str:
    """'deflection: the check pulls the black queen on d7 off the knight on d6' -- or ''."""
    seqs = find_sequences(board, list(pv)[:3], 3)
    if not seqs:
        return ""
    r = seqs[0]; n = r.names(); p = [pc(sym, sq) for sym, sq in zip(r.pieces, r.squares)]
    if r.name == "deflection":
        return f"deflection: {p[1]} is pulled off the defence of {p[2]}"
    if r.name == "attraction":
        return f"attraction: {p[2]} is lured to {n[1]} and hit there"
    if r.name == "interference":
        return f"interference: the move on {n[0]} cuts {p[1]} off from {p[2]}"
    if r.name == "clearance":
        return f"clearance: {n[0]} is vacated for the piece from {n[1]}"
    if r.name == "removal_of_guard":
        return f"removal of the guard: {p[0]} was defending {p[1]}"
    return ""
from .horizon import winp

COLOR = {True: "White", False: "Black"}
PIECE = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}
PRIORITY = ("fork", "pinned_and_attacked", "hanging_through_pin", "disabled_attack", "disabled_defense",
            "pin", "skewer", "discovered_attack", "hanging", "attacked_by_lower_value",
            "battery", "overloaded", "xray", "double_duty", "xray_defense", "alignment_own", "attacks_king_zone", "loose")


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
    if n == "xray_defense":
        return f"{p[0]} defends {p[2]} through {p[1]} (an x-ray defence)"
    if n == "alignment_own":
        if "behind_pawn" in r.tags:
            return f"{p[0]} stands behind {p[1]} on the file"
        if "back_rank" in r.tags:
            return f"{p[0]} and {p[2]} share a line with {p[1]} between (back-rank geometry)"
        return f"{p[0]}, {p[1]} and {p[2]} are aligned"
    if n == "double_duty":
        return f"{p[0]} does two jobs: attacks {p[1]} and defends {p[2]}" if "attack+defence" in r.tags else f"{p[0]} defends both {p[1]} and {p[2]}"
    if n == "double_attack":
        return f"{p[0]} attacks both " + " and ".join(p[1:])
    if n == "attacks_king_zone":
        return f"{p[0]} attacks {r.tags[0]} square(s) around {p[1]}"
    return f"{n} {' '.join(p)}"


from .rules import STANDARD as _STD
_SUBJ = {"hanging": 0, "loose": 0, "attacked_by_lower_value": 1, "pin": 1, "skewer": 1, "xray": 1, "fork": 0,
         "discovered_attack": 2, "battery": 2, "overloaded": 0, "disabled_attack": 3, "disabled_defense": 3,
         "pinned_and_attacked": 1, "hanging_through_pin": 0, "xray_defense": 2, "alignment_own": 0, "double_duty": 0}


def _top(rels: list[Relation], k: int) -> list[Relation]:
    def key(r):
        sym = r.pieces[_SUBJ.get(r.name, 0)] if r.pieces else "-"
        return (PRIORITY.index(r.name) if r.name in PRIORITY else 99, -_STD.of_symbol(sym))
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
    if abs(v) >= MATE_CP - 1000:                          # a folded mate score with the distance lost: say 'mate', not +99.85
        return "#" if v > 0 else "#-"
    return f"{v/100:+.2f}"


def gain_str(gain_cp: int) -> str:
    """A threat's gain over the tempo baseline; a mating threat says so instead of printing thousands of cp."""
    return "mates" if gain_cp >= MATE_CP - 2000 else f"{gain_cp:+d} cp above a tempo"


def candidate_phrases(c: Candidate, n: int = 3, k: int = 2) -> str:
    """LONG form: every notable fact per ply (kept behind --long)."""
    return " ".join(p for p in (effect_phrase(e, k) for e in c.effects[:n]) if p)


# =============================== TERSE FORM (default) ===============================
# One fact per ply; the fact a coach would say first. Preference: what the move did TO the
# opponent's pieces, then what it exposed of our own. (Kaveh 2026-09-04: "the first clause
# of each line was the one that mattered, and the rest was true clutter.")
_REAL = ("fork", "pin", "skewer", "pinned_and_attacked", "hanging_through_pin", "discovered_attack",
         "disabled_attack", "disabled_defense", "overloaded", "battery", "attacked_by_lower_value", "hanging")


def _subject_sym(r: Relation) -> str:
    return r.pieces[_SUBJ.get(r.name, 0)] if r.pieces else "-"


def _rank(r: Relation, mover: bool) -> tuple:
    sym = _subject_sym(r)
    about_opp = sym != "-" and (sym.isupper() != mover)
    liability = r.name in ("hanging", "loose") and not about_opp
    return (1 if liability else 0, 0 if about_opp else 1,
            PRIORITY.index(r.name) if r.name in PRIORITY else 99, -_STD.of_symbol(sym))


def favors_mover(r: Relation, mover: bool) -> bool:
    """Does this relation work FOR the side that just moved? (victim-owned facts are owned by the victim)"""
    return not against(r, mover)


def best_fact(e: Effect, strict: bool = False) -> str | None:
    """The single strongest clause for this move, or None if the move is quiet.
    strict: only facts that favour the mover (for 'why the refutation works')."""
    if e.san.endswith("#"):
        return "checkmate"
    if e.captured:
        order = ("pinned_and_attacked", "hanging_through_pin", "hanging", "loose")
        why = [RESOLVED_WORD[r.name] for r in sorted(e.resolved, key=lambda r: order.index(r.name) if r.name in order else 99)
               if r.name in RESOLVED_WORD]
        cash = [r for r in e.uses if r.name not in ("discovered_attack", "battery")]
        s = f"{COLOR[e.mover]} takes the {e.captured[0]} ({e.captured[1]})"
        if why:
            s += f", {why[0]}"
        elif cash:
            s += ", cashing the threat by " + pc(cash[0].pieces[0], cash[0].squares[0])
        if e.gives_check:
            s += ", with check"
        disc = [r for r in e.uses if r.name in ("discovered_attack", "battery")]
        if disc:                                   # a capture that also discovers an attack: say both
            r = disc[0]
            s += f", discovering an attack by {pc(r.pieces[0], r.squares[0])} on {pc(r.pieces[2], r.squares[2])}"
        return s
    disc = [r for r in e.uses if r.name in ("discovered_attack", "battery")]
    if disc:
        r = disc[0]
        return ("with check, " if e.gives_check else "") + f"discovering an attack by {pc(r.pieces[0], r.squares[0])} on {pc(r.pieces[2], r.squares[2])}"
    created = sorted((r for r in e.created if not strict or favors_mover(r, e.mover)), key=lambda r: _rank(r, e.mover))
    if created:
        ph = rel_phrase(created[0])
        return ("with check, " if e.gives_check and "with check" not in ph else "") + ph
    saved = [r for r in e.escaped if r.name in ("hanging", "pinned_and_attacked", "hanging_through_pin", "attacked_by_lower_value")
             and _STD.of_symbol(_subject_sym(r)) >= _STD.minor_value]
    if saved:
        return "saving " + pc(saved[0].pieces[_SUBJ.get(saved[0].name, 0)], saved[0].squares[_SUBJ.get(saved[0].name, 0)])
    if e.gives_check:
        return "with check"
    return None


def terse_line(c: Candidate, n: int = 3) -> str:
    parts = []
    for e in c.effects[:n]:
        f = best_fact(e)
        if f:
            parts.append(f"{e.san}: {f}")
    return "; ".join(parts)


def why_clause(r: Candidate | None, board: chess.Board | None = None) -> str:
    """The refutation's single strongest first-ply fact; a sequence motif over its first 3 plies wins if present."""
    if not r or not r.effects:
        return ""
    e0 = r.effects[0]
    big_capture = e0.captured and (e0.captured[1] >= 5 or any(x.name in ("hanging", "pinned_and_attacked") for x in e0.resolved))
    if board is not None and r.pv and not big_capture:          # a plain win of a big/hanging piece says it best
        m = motif_phrase(board, r.pv)
        if m:
            return f" -- {m}"
    f = best_fact(e0, strict=True)
    if not f and len(r.effects) > 2:                 # a quiet first move: say what the refuter's NEXT move does
        f2 = best_fact(r.effects[2], strict=True)
        f = f"then {r.effects[2].san}: {f2}" if f2 else None
    return f" -- {f}" if f else ""


def render(rec: IdeaRecord, *, long: bool = False, alternatives: bool = True, verbose: bool | None = None) -> str:
    if verbose is not None:                      # backwards compatibility with the S0 signature
        alternatives = verbose
    w = rec.side == "white"
    who = COLOR[w]
    num = rec.ply // 2 + 1
    ev = cp_str(rec.eval_cp, w, rec.idea.mate if rec.idea else None)
    out = [f"[{num}{'.' if w else '...'}] {who} to move, eval {ev}"]
    seen = set()

    def add(line):
        if line not in seen:
            seen.add(line); out.append(line)

    if rec.played and rec.verdict in ("blunder", "mistake", "inaccuracy"):
        art = "an" if rec.verdict[0] in "aeiou" else "a"
        s = f"  {rec.played.san} was {art} {rec.verdict}: {rec.idea.san} was better ({cp_str(rec.played.cp, w, rec.played.mate)} vs {ev})."
        r = rec.played.refutation
        if r:
            t = chess.Board(rec.fen); t.push(chess.Move.from_uci(rec.played.uci))
            mate = mate_in_line(t, r.pv[:10]) if r.mate is not None else None
            if mate:
                s += f" Punished by {r.san} -- {mate[1]}: {mate[2]}."
            else:
                s += f" Punished by {r.san}" + (f" ({r.line_san})" if long else "") + (candidate_phrases(r, 1) and f": {candidate_phrases(r, 1)}" if long else why_clause(r)) + "."
        add(s)
    shown = 0
    for t in rec.threats:
        if shown >= 1 and not (t.line.mate is not None or t.gain_cp >= 300):
            break
        fe = (effect_phrase(t.first_effects, 1) if long else best_fact(t.first_effects)) if t.first_effects else ""
        b = chess.Board(rec.fen)
        already = f" Already: {rel_phrase(t.structure)}." if t.structure is not None else ""
        add(f"  If {who} does nothing: {t.san_line(b, 4)}" + (f" -- {fe}" if fe else "") + f" ({gain_str(t.gain_cp)}).{already}")
        shown += 1
    if rec.zugzwang and rec.zugzwang.flagged:
        add(f"  Zugzwang: every move worsens {who}'s position (passing would be {rec.zugzwang.pass_cp - rec.zugzwang.best_cp:+d} cp better).")
    if rec.idea:
        mate = mate_in_line(chess.Board(rec.fen), rec.idea.pv[:10]) if rec.idea.mate is not None else None
        if mate and mate[0] % 2 == 1:                   # odd plies: the side to move delivers it
            n, pat, why = mate
            add(f"  Mate in {(n + 1) // 2} was available: {rec.idea.line_san} -- {pat}: {why}.")
        else:
            mp = motif_phrase(chess.Board(rec.fen), rec.idea.pv)
            body = candidate_phrases(rec.idea) if long else terse_line(rec.idea)
            add(f"  Idea: {rec.idea.san} -- {rec.idea.line_san}." + (f" {mp}." if mp else "") + (f" {body}" if body else ""))
    for k in rec.killed[:3 if long else 2]:
        r = k.candidate.refutation
        s = f"  {k.candidate.san} no longer works after {k.killed_by} ({cp_str(k.was_cp, w)} -> {cp_str(k.now_cp, w, k.candidate.mate)})"
        if r:
            s += f": {r.san}" + (f" -- {candidate_phrases(r, 1)}" if long and candidate_phrases(r, 1) else why_clause(r))
        add(s + ".")
    if alternatives:
        for a in rec.alternatives[:2 if long else 1]:
            r = a.refutation
            s = f"  {a.san} ({cp_str(a.cp, w, a.mate)}) fails"
            if r:
                s += f" to {r.san}" + (f" -- {candidate_phrases(r, 1)}" if long and candidate_phrases(r, 1) else why_clause(r))
            add(s + ".")
    return "\n".join(out)


# =============================== HEADLINE (one sentence per error) ===============================
# Kaveh 2026-09-04: explain what is immediately useful. Order of what to say first: a missed mate;
# what the move walked into / ignored / allowed (with the refutation's one fact); a horizon trap
# with its collapse depth; else the generic verdict. Then the ladder tag: shared with the
# player's level, or a slip. Computer moves (every level prefers the played move and the loss is
# small) are marked so the review can hide them.
NOUN = {"pin": "a pin", "fork": "a fork", "discovered": "a discovered attack", "skewer": "a skewer",
        "hanging": "a hanging piece", "attacked_by_lower": "an attack by a lesser piece", "overload": "an overloaded defender",
        "mate": "a mating attack", "capture": "a capture"}


def _noun(families: list[str]) -> str:
    fams = [f for f in families if f in NOUN and f != "capture"] or families
    return NOUN.get(fams[0], "a tactic") if fams else "a tactic"


def delta_str(rec: IdeaRecord) -> str:
    """'-1.0 pawns, -12 win%' -- the cost of the played move against the best, the number a player wants first."""
    if not rec.played:
        return ""
    d_cp = rec.played.cp - rec.eval_cp
    d_wp = winp(rec.played.cp) - winp(rec.eval_cp)
    folded = MATE_CP - 1000
    if abs(rec.played.cp) >= folded or abs(rec.eval_cp) >= folded:          # a mate score: pawns are meaningless
        if rec.eval_cp >= folded and rec.played.cp < folded and rec.idea and rec.idea.mate:
            return f"mate in {(rec.idea.mate + 1) // 2} was there, {d_wp:+.0f} win%"
        return f"{d_wp:+.0f} win%"
    return f"{d_cp/100:+.1f} pawns, {d_wp:+.0f} win%"


PURPOSE_NAMES = ("pin", "pinned_and_attacked", "hanging", "attacked_by_lower_value", "disabled_attack", "disabled_defense", "skewer", "fork")


def purpose(rec: IdeaRecord) -> str:
    """What the played move DID for its own side (the pin it broke, the piece it saved) -- so the critique
    acknowledges the point of the move before saying what it cost. Priority: a piece saved from capture,
    then a fork/skewer escaped, then a pin broken; ties go to the bigger piece."""
    pl = rec.played
    if not pl or not pl.effects:
        return ""
    e = pl.effects[0]; us = rec.side == "white"
    mates = [t for t in rec.threats if t.real and t.line.mate is not None and t.line.mate > 0]
    if mates and pl.mate is None:                                             # the move parried a mate threat: that IS its purpose
        b = chess.Board(rec.fen); nb = b.copy(stack=False); nb.push(chess.Move.null())
        return f"stopping {nb.san(chess.Move.from_uci(mates[0].line.move))}"
    cands = []
    for r in list(e.escaped) + list(e.destroyed) + list(e.resolved):
        if r.name not in PURPOSE_NAMES:
            continue
        if not against(r, us):                                                # a pin/skewer/fork BY them on us, or our piece in trouble
            continue
        if r.name == "pin":
            if "absolute" not in r.tags and _STD.of_symbol(r.pieces[1]) < _STD.minor_value:
                continue                                                      # a pawn's relative pin is not a purpose
            cands.append((2, -_STD.of_symbol(r.pieces[1]), f"unpinning {pc(r.pieces[1], r.squares[1])}"))
        elif r.name in ("hanging", "pinned_and_attacked", "attacked_by_lower_value"):
            i = 0 if r.name == "hanging" else 1                                # the piece in trouble: (pinner, PINNED, ...) / (attacker, TARGET)
            cands.append((0, -_STD.of_symbol(r.pieces[i]), f"saving {pc(r.pieces[i], r.squares[i])}"))
        elif r.name in ("fork", "skewer"):
            cands.append((1, 0.0, f"escaping the {r.name}"))
    return min(cands)[2] if cands else ""


def headline(rec: IdeaRecord, cause: dict | None, trap: dict | None, p_played: dict, p_best: dict, level: int) -> tuple[str, bool]:
    """(sentence, is_computer_move)"""
    w = rec.side == "white"
    pl = rec.played
    r = pl.refutation if pl else None
    _t = chess.Board(rec.fen); _t.push(chess.Move.from_uci(pl.uci)) if pl else None
    why = why_clause(r, _t).replace(" -- ", "")
    b = chess.Board(rec.fen)
    ev_pl, ev_best = cp_str(pl.cp, w, pl.mate), cp_str(rec.eval_cp, w, rec.idea.mate if rec.idea else None)
    mate = mate_in_line(b, rec.idea.pv[:10]) if rec.idea and rec.idea.mate is not None else None
    computer = False
    purp = purpose(rec)
    pl_name = f"{pl.san}" + (f" ({purp})" if purp else "")
    if mate and mate[0] % 2 == 1:
        s = f"You had mate in {(mate[0] + 1) // 2}: {rec.idea.line_san} -- {mate[1]}: {mate[2]}. You played {pl.san} ({delta_str(rec)})."
    elif cause and cause["cause"] == "positional":
        e0 = r.effects[0] if r and r.effects else None
        we_took = b.is_capture(chess.Move.from_uci(pl.uci)) and not b.is_en_passant(chess.Move.from_uci(pl.uci))
        if we_took and e0 and e0.captured:                      # we captured first, they recaptured
            ours = chess.piece_name(b.piece_at(chess.Move.from_uci(pl.uci).to_square).piece_type)
            head = f"{pl_name} traded {e0.captured[0]} for {ours}, an even trade" + (" (they recapture with check)" if e0.gives_check else "")
        else:
            what = f"{e0.captured[0]} for {r.effects[1].captured[0]}" if e0 and len(r.effects) > 1 and r.effects[1].captured else "a trade"
            head = f"{pl_name} allowed {r.san}, an even trade ({what}" + (", with check" if e0 and e0.gives_check else "") + ")"
        s = (f"{head}. The cost ({delta_str(rec)}) is positional -- what the trade does to the plans -- "
             f"which this layer does not explain. Better was {rec.idea.san} ({ev_best}).")
    elif cause and cause["cause"] == "missed_better" and rec.idea:
        e0 = rec.idea.effects[0] if rec.idea.effects else None
        gain = best_fact(e0, strict=True) if e0 else None
        gain = gain if gain and any(k in gain for k in ("takes", "hanging", "forks", "pins", "skewers", "checkmate", "discovering", "attacks", "with check")) else ""
        mp = motif_phrase(b, rec.idea.pv)
        s = f"{pl_name} missed {rec.idea.san}" + (f" -- {mp}" if mp else f" -- {gain}" if gain else "") + f" ({delta_str(rec)})."
    elif cause and cause["cause"] == "walked_into":
        e0 = r.effects[0] if r and r.effects else None
        if e0 and e0.captured and chess.Move.from_uci(r.uci).to_square == chess.Move.from_uci(pl.uci).to_square:
            _after = chess.Board(rec.fen); _after.push(chess.Move.from_uci(pl.uci))
            defended = bool(_after.attackers(_after.turn == chess.WHITE and chess.BLACK or chess.WHITE, chess.Move.from_uci(pl.uci).to_square))
            if defended:
                s = f"{pl.san} put the {e0.captured[0]} where {r.san} trades it off" + (" with check" if e0.gives_check else "") + f" ({delta_str(rec)}). Better was {rec.idea.san} ({ev_best})."
            else:
                s = f"{pl.san} hung the {e0.captured[0]}: {r.san} takes it" + (", with check" if e0.gives_check else "") + f" ({delta_str(rec)}). Better was {rec.idea.san} ({ev_best})."
        else:
            wr = cause.get("walked_into_rel")
            if wr:                                                            # name the liability the move created, in its own words
                rel = Relation(wr["name"], len(wr["squares"]), tuple(wr["squares"]), wr["owner"], tuple(wr["tags"]), tuple(wr["pieces"]))
                s = f"{pl_name} walked into it -- {rel_phrase(rel)} -- and {r.san if r else '?'} cashes it" + (f" ({why})" if why and why != rel_phrase(rel) else "") + f" ({delta_str(rec)}). Better was {rec.idea.san} ({ev_best})."
            else:
                s = f"{pl_name} walked into {_noun(cause['walked_into_hit'])}: {r.san if r else '?'}" + (f" -- {why}" if why else "") + f" ({delta_str(rec)}). Better was {rec.idea.san} ({ev_best})."
    elif cause and cause["cause"] == "ignored_threat":
        already = next((t.structure for t in rec.threats if t.structure is not None and r and t.line.move == r.uci), None)
        s = f"{pl.san} ignored a standing threat: {r.san if r else '?'}" + (f" -- {why}" if why else "") + (f" (already: {rel_phrase(already)})" if already else "") + f" ({delta_str(rec)}). Better was {rec.idea.san} ({ev_best})."
    elif cause and cause["cause"] == "enabled_tactic":
        s = f"{pl.san} allowed {_noun(cause['walked_into'] or [m for m in (r and cause.get('mechanisms', [])) or []])}: {r.san if r else '?'}" + (f" -- {why}" if why else "") + f" ({delta_str(rec)}). Better was {rec.idea.san} ({ev_best})."
    elif trap and trap.get("kind") == "horizon":
        s = f"{pl.san} looks fine for {trap['peak_depth']} plies and collapses at depth {trap['collapse_depth']}: {r.san if r else '?'}" + (f" -- {why}" if why else "") + f" ({delta_str(rec)}). Better was {rec.idea.san} ({ev_best})."
    else:
        first = terse_line(rec.idea, 1) if rec.idea else ""
        s = f"{pl.san} was {'an' if rec.verdict[0] in 'aeiou' else 'a'} {rec.verdict} ({delta_str(rec)}); {rec.idea.san} ({ev_best}) was better" + (f" -- {first.split(': ', 1)[1]}" if ': ' in first else "") + "."
        drop = winp(rec.eval_cp) - winp(pl.cp)
        if rec.verdict == "inaccuracy" and drop < 10 and p_played and all(p_played.get(k, 0) >= p_best.get(k, 0) for k in p_played):
            computer = True
    return s, computer


def ladder_tag(p_played: dict, p_best: dict, level: int, played_san: str, best_san: str | None) -> str:
    """The level-relative sentence, added once the player's level is measured."""
    pp = p_played.get(level, 0.0); out = ""
    if pp >= 0.30:
        out += f" A {level} plays {played_san} too ({pp*100:.0f}%)."
    elif pp <= 0.02:
        out += f" (A slip: a {level} almost never plays this.)"
    pb = p_best.get(level, 0.0)
    if best_san and pb >= 0.30:
        out += f" A {level} finds {best_san} ({pb*100:.0f}%)."
    return out
