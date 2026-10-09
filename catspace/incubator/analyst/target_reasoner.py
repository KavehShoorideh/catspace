"""TARGET-FIRST REASONING (Kaveh 2026-10-08, on puzzle NPpe0): "start from the highest-value goal -- can I do anything
against the king at this moment? No, but I can against the queen ... the queen is so oddly placed, almost trapped on
the edge ... a prior to recognize that ... the moment we threaten the queen it runs away, so either create a bigger
threat at the same time -- the bigger threat is against the king -- or keep chasing it until it cannot escape ...
then, what if we don't do this, what is their next threat? -- but we shouldn't start there."

THE PROCEDURE, for the side to move:
  1. TARGETS in order of value (the king first), ties broken by a VULNERABILITY PRIOR -- few safe escapes, few
     defenders, on the edge (analyst_core.target_profile).
  2. THE KING: only if we bear on it at all ("not enough pieces there" otherwise); a mating chain is a certified gain
     of MATE (the cornering search with the king as its focus).
  3. ANOTHER TARGET: the moves that matter to it (analyst_core.target_moves: they take it, or increase our coverage of
     its frame -- a clearance included), tried in two methods: A BIGGER THREAT AT THE SAME TIME (it attacks the target
     and gives check), then THE CHASE (attack it or take its escapes) -- each certified by the value search focused on
     the target (analyst_core.certified_gain). The first target with a certified gain decides; lower targets are not
     tried.
  4. THEIR NEXT THREAT (the null move): if it is bigger than what we found, defending comes first.
  5. THE TRACE: every target considered, every method and value, and for the best rejected candidates their
     refutation (analyst_core.best_reply) -- the "why not" of the explanation.
Certified gains are lower bounds (modulo frame independence on the opponent's side; see rust/analyst_core).
"""
from __future__ import annotations

import time

import chess

MATE = 50
FIRST, DEEP = 2_000_000, 1_500_000   # (2026-10-08: 200k was just under what 4-ply certificates need -- about 30 ms each)


def _their_threat(b: chess.Board) -> int:
    from .forcing_frame import _best_capture
    from .chess_domain import passed
    p = passed(b)
    return _best_capture(p) if p is not None else 0


def their_plan(b: chess.Board, plies: int = 4) -> dict:
    """THE SAME REASONING FROM THEIR SIDE (Kaveh 2026-10-08: "what are the threats my opponent has against my king, then
    my queen, then my major pieces -- the reverse of the same thing"): the target-first reasoner run for the opponent
    on the null-move position (as if we passed). -> its result (their move, certified gain, target, trace)."""
    from .chess_domain import passed
    p = passed(b)
    if p is None:                                                     # we are in check: the evasion is the frame's job
        return dict(move=None, gain=0, target=None, trace=[])
    return reason(p, plies, symmetric=False)


def _their_gain_after(b: chess.Board, m: chess.Move, plies: int = 4) -> int:
    """After our move m (their turn on the real board): their best certified gain by the same reasoning."""
    nb = b.copy(stack=False); nb.push(m)
    if nb.is_checkmate():
        return -1000
    return reason(nb, plies, symmetric=False)["gain"]


import os as _os
SYMMETRIC = _os.environ.get("CATSPACE_SYMMETRIC", "1") == "1"


def reason(b: chess.Board, plies: int = 4, max_cands: int = 8, symmetric: bool | None = None) -> dict:
    if symmetric is None:
        symmetric = SYMMETRIC
    import analyst_core
    t0 = time.time()
    fen = b.fen(); us = b.turn
    trace = []
    targets = []
    for sq, pc in b.piece_map().items():
        if pc.color == us or pc.piece_type == chess.PAWN:
            continue
        prof = analyst_core.target_profile(fen, chess.square_name(sq))
        if prof is None:
            continue
        value, esc, att, dfd, edge = prof
        prior = -esc - dfd + (1 if edge else 0)                     # the vulnerability prior: fewer escapes and defenders
        targets.append((value, prior, sq, prof))
    targets.sort(key=lambda x: (-x[0], -x[1]))
    best = None
    for value, prior, sq, (v, esc, att, dfd, edge) in targets:
        if best is not None and value <= best[1]:
            trace.append(f"stop: the next target is worth {value}, not more than the +{best[1]} found")
            break
        name = f"{chess.piece_name(b.piece_at(sq).piece_type)} {chess.square_name(sq)}"
        cands = analyst_core.target_moves(fen, chess.square_name(sq))
        if b.piece_at(sq).piece_type == chess.KING:
            # "the king can be attacked, but I don't have enough pieces there" (Kaveh): pursue the king only when two or
            # more of our pieces bear on its zone, or a check leaves it at most one safe escape
            zone = chess.SquareSet(chess.BB_KING_ATTACKS[sq]) | chess.SquareSet.from_square(sq)
            pressure = len({a for z in zone for a in b.attackers(us, z)})
            tight = [c for c in cands if c[3] and c[1] <= 1]
            if pressure < 2 and not tight:
                trace.append(f"{name}: {pressure} of our pieces bear on its zone and no check leaves it at most one flight -- "
                             f"skip ('not enough pieces there')")
                continue
            cands = tight + [c for c in cands if c not in tight]
            goal = MATE
        else:
            goal = 1
        trace.append(f"{name} (value {v}, safe escapes {esc}, our attackers {att}, defenders {dfd}{', on the edge' if edge else ''}): "
                     f"{len(cands)} moves matter to it")
        bigger = [c for c in cands if c[2] and c[3]]                  # method A: attack it AND give check
        chase = [c for c in cands if c not in bigger]                 # method B: the chase
        listed = {c[0] for c in cands}                                # method C: a check first, then the chase (2026-10-08,
        forcing = [(m.uci(), 99, False, True) for m in b.legal_moves  # 0t58R Nd6+, 0k5nO Nf6+: the check touches nothing yet)
                   if m.uci() not in listed and b.gives_check(m)]
        found = None
        for method, group in (("a bigger threat at the same time (check)", bigger), ("the chase", chase), ("a check first, then the chase", forcing)):
            for uci, e_left, attacks, check in group[:max_cands]:
                g = analyst_core.certified_gain(fen, uci, [chess.square_name(sq)], plies, FIRST)
                if g is not None and g < goal:                         # deepen only a search that FINISHED short (one
                    g2 = analyst_core.certified_gain(fen, uci, [chess.square_name(sq)], plies + 2, DEEP)   # that ran out
                    g = g2 if g2 is not None else g                    # at 4 plies runs out at 6)
                m = chess.Move.from_uci(uci)
                trace.append(f"    {method}: {b.san(m)} (escapes left {e_left}) -> certified {g if g is not None else 'budget'}")
                if g is not None and g >= goal and (found is None or g > found[1]):
                    found = (m, g, method)
            if found:
                break
        if found and (best is None or found[1] > best[1]):
            best = (found[0], found[1], name, found[2])
    # (targets are tried from the most valuable down; the search goes on while the NEXT target's value could beat the
    # best gain found -- a target worth v cannot yield more than v. 2026-10-08, 0ZLzT / 0jyzC: the first target gave +1
    # / +2, a lesser target +4 / +5)
    threat = _their_threat(b)
    out = dict(move=best[0] if best else None, gain=best[1] if best else 0, target=best[2] if best else None,
               method=best[3] if best else None, their_threat=threat, trace=trace)
    if symmetric:
        # THEIR PLAN, reasoned the same way (king first, then the queen, ...), and the decision between attack and defence
        theirs = their_plan(b, plies)
        tg = theirs["gain"]
        if tg > 0:
            trace.append(f"THEIR PLAN (as if we passed): {theirs['target']} by {theirs['method']}, certified +{tg}")
        ours_g = best[1] if best else 0
        # (no veto after our move: their full reasoning on the real board counts their ordinary recapture inside OUR
        # exchange, which our certified value already paid -- the double count of 2026-10-07 again; bug 2026-10-09)
        # DOES OUR ATTACK ALREADY ANSWER THEIRS? (2026-10-09: the symmetric step lost 11 and won 0 -- it defended where the
        # attack took their attacker or out-threatened them; Kaveh's defender options: capture the attacker, a bigger
        # threat.) Their PLANNED move after ours: illegal or certifying less than our gain -> attack.
        if best is not None and tg > ours_g:
            import analyst_core
            nb = best[0]
            after = b.copy(stack=False); after.push(nb)
            tm = theirs["move"]
            if after.is_checkmate() or tm is None or tm not in after.legal_moves:
                left = 0
            else:
                g = analyst_core.certified_gain(after.fen(), tm.uci(), [theirs["target"].split()[-1]], plies, FIRST)
                left = g if g is not None else tg
            trace.append(f"  after our {b.san(nb)} their plan {b.san(tm) if tm is not None and tm in b.legal_moves else '-'} certifies +{max(0, left)}")
            if ours_g >= max(0, left):
                tg = 0                                                 # our attack answers theirs: no defence needed
        if tg > ours_g:
            # DEFEND: the moves that touch their plan -- moves of the threatened piece, moves from or to their target's
            # frame, captures, checks -- each scored by what they can still certify after it (and ours, for ties)
            tsq = chess.parse_square(theirs["target"].split()[-1]) if theirs["target"] else None
            cands = [m for m in b.legal_moves if (tsq is not None and (m.from_square == tsq or tsq in b.attacks(m.to_square)))
                     or b.is_capture(m) or b.gives_check(m)
                     or (theirs["move"] is not None and m.to_square in (theirs["move"].from_square, theirs["move"].to_square))]
            import analyst_core
            scored = []
            tm = theirs["move"]
            for m in cands[:16]:                                        # ONE certificate per candidate (speed, 2026-10-09):
                nb = b.copy(stack=False); nb.push(m)                     # does THEIR planned move still certify after ours?
                if nb.is_checkmate():
                    scored.append((-1000, m)); continue
                if tm is None or tm not in nb.legal_moves:
                    scored.append((0, m)); continue
                g = analyst_core.certified_gain(nb.fen(), tm.uci(), [theirs["target"].split()[-1]], plies, FIRST)
                scored.append((g if g is not None else tg, m))
            if scored:
                scored.sort(key=lambda x: x[0])
                g, m = scored[0]
                trace.append(f"DEFEND: {b.san(m)} leaves them +{max(0, g)} (their plan was +{tg})")
                out.update(move=m, gain=-max(0, g), target=theirs["target"], method="defence", defend_first=True)
                out["ms"] = round(1000 * (time.time() - t0))
                return out
        if best is None:
            out.update(move=None, gain=0, target=None, method=None)
    if best:                                                          # why the strongest alternatives on the same target fail
        sqn = best[2].split()[-1]
        for uci, *_ in analyst_core.target_moves(fen, sqn)[:max_cands]:
            if uci == best[0].uci():
                continue
            rs = analyst_core.best_reply(fen, uci, [sqn], plies, FIRST)
            if rs and rs[0][1] < 1:
                m = chess.Move.from_uci(uci); nb = b.copy(stack=False); nb.push(m)
                trace.append(f"    why not {b.san(m)}: {nb.san(chess.Move.from_uci(rs[0][0]))} holds us to {rs[0][1]:+d}")
    out["ms"] = round(1000 * (time.time() - t0))
    return out


def explain(fen: str, setup: str | None = None) -> None:
    b = chess.Board(fen)
    if setup:
        b.push_uci(setup)
    r = reason(b)
    print(f"position {b.fen()} ({'White' if b.turn else 'Black'} to move)")
    for line in r["trace"]:
        print("  " + line)
    if r["move"]:
        print(f"  => {b.san(r['move'])}: target {r['target']}, by {r['method']}, certified +{r['gain']}; their next threat {r['their_threat']}; {r['ms']} ms")
    else:
        print(f"  => no certified plan ({r['ms']} ms): fall back to search")
