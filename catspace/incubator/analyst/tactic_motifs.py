"""TACTICAL MOTIFS, as complete a list as the literature gives (Kaveh 2026-10-07: "have you as the LLM look at the tactics
and label them as all these groups that you know from chess ... I want it to be all encompassing"), and a STRUCTURAL
LABELLER that reads a fired tactic (tactic_coverage.events_of: a visibly safe or defended piece lost, forced) at its start
and labels it with the motifs it finds. The labels are for REVIEW, not for play: several may apply, and what the labeller
cannot read is left for the LLM to label by eye.

THE LIST (the standard motif vocabulary: Lichess's puzzle themes, Weteschnik's "Understanding Chess Tactics", Nunn's
"Learn Chess Tactics", the mating-pattern catalogue):
  double attacks     fork (knight, pawn, queen, rook, bishop, king; royal fork), double attack with check, discovered
                     attack, discovered check, double check, battery
  lines              absolute pin, relative pin, skewer, x-ray, interference (a line of defence closed), clearance /
                     line opening, line closing, cross-pin
  removing the guard capture of the defender, deflection (the defender forced away), decoy / attraction (a piece lured
                     to a square), overloading (one defender, two duties), undermining (the supporting pawn removed),
                     pinned defender (the defender cannot recapture)
  tempo              zwischenzug (an in-between move), desperado (a doomed piece takes what it can), tempo gain with
                     check, counter-attack, quiet move, zugzwang
  confinement        trapped piece (no safe square), trapped queen, king hunt, smothered
  pawns              pawn fork, breakthrough, passed pawn / promotion, underpromotion, en passant, pawn lever
  the king           back-rank weakness, exposed king, mating net, the mating patterns (back rank, smothered, Anastasia,
                     Arabian, Boden, double bishop, dovetail, epaulette, hook, Greek gift / Lolli, Opera, Pillsbury,
                     swallow's tail, Damiano, corridor, ladder), perpetual check, stalemate (defensive)
  counting           insufficiently defended (more attackers than defenders arrive), loose piece (LPDO: a defended
                     piece whose defender is itself attacked), capture with tempo, exchange sacrifice, windmill

WHAT THE LABELLER READS, at the start (A to move, A's move m, the lost piece P of V), in this order; each test is a
chess fact on the boards, not a guess:
  discovered check / discovered attack   m moved a piece off a line of A's slider that now attacks P (or the king)
  double check                           after m the king is attacked by two pieces
  fork / double attack                   the moved piece attacks P and another piece worth 3+ (or the king): by the
                                         mover's kind; royal when the king and queen are the targets
  pawn fork                              the mover is a pawn attacking two pieces
  skewer / pin                           a slider attacks P through a piece of V on the same line (skewer: the front
                                         piece is dearer; pin: P is in front of a dearer piece or the king)
  pinned defender                        P's only defender is pinned to its king (cannot recapture)
  capture of the defender                m captured a defender of P
  deflection / overload                  P's defender moved away at V's next turn (to answer m), and P was then taken
  trapped piece                          after m, P is attacked and has no square where it is not lost
  x-ray                                  A's slider attacks P through another piece of A (a battery) or through P's own
                                         defender
  interference                           m placed a piece between P and its defender
  promotion / breakthrough               a pawn of A promotes or reaches the seventh in the window
  back rank                              P was lost to a back-rank motif (A's rook or queen on V's home rank with check)
  zwischenzug                            A's winning capture came after an in-between check or bigger capture
  insufficiently defended                none of the above: P was defended, but the exchange on its square loses by
                                         the count (a cheaper attacker arrived or a defender was taken first)
  (unlabelled)                           none: for the LLM's eye
"""
from __future__ import annotations

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)

MOTIFS = ["discovered check", "discovered attack", "double check", "royal fork", "knight fork", "pawn fork", "queen fork",
          "rook fork", "bishop fork", "king fork", "double attack with check", "skewer", "absolute pin", "relative pin",
          "pinned defender", "capture of the defender", "deflection / overload", "trapped piece", "x-ray", "interference",
          "promotion / breakthrough", "back rank", "zwischenzug", "insufficiently defended", "desperado", "en passant"]


def _between_line(a: int, b: int) -> list:
    """The squares strictly between a and b when they share a line, else []."""
    return [s for s in chess.SQUARES if chess.BB_SQUARES[s] & chess.between(a, b)] if chess.BB_SQUARES[b] & chess.BB_RAYS[a][b] else []


def _ray_through(b: chess.Board, slider: int, target: int):
    """Does the slider on `slider` attack along the line to `target` with exactly one piece between (the x-rayed one)?
    -> that piece's square or None."""
    between = _between_line(slider, target)
    if not between:
        return None
    occ = [s for s in between if b.piece_at(s) is not None]
    return occ[0] if len(occ) == 1 else None


def _attacks_through(b: chess.Board, slider_sq: int, target: int) -> bool:
    p = b.piece_at(slider_sq)
    if p is None or p.piece_type not in SLIDERS:
        return False
    ray = chess.BB_RAYS[slider_sq][target]
    if not ray:
        return False
    kind_ok = (p.piece_type == chess.QUEEN) or (p.piece_type == chess.ROOK and (chess.square_file(slider_sq) == chess.square_file(target) or chess.square_rank(slider_sq) == chess.square_rank(target))) \
        or (p.piece_type == chess.BISHOP and chess.square_file(slider_sq) != chess.square_file(target) and chess.square_rank(slider_sq) != chess.square_rank(target))
    return kind_ok


def label(b0: chess.Board, m: chess.Move, lost_sq: int, seq: list) -> list:
    """Labels for a fired tactic: `b0` the start (A to move), `m` A's move, `lost_sq` the lost piece's square at the start,
    `seq` the moves played from b0 (m first). -> [motif, ...] in order of evidence; [] when nothing is read."""
    A = b0.turn; V = not A
    P = b0.piece_at(lost_sq)
    b1 = b0.copy(stack=False); b1.push(m)
    labels = []
    mover_sq = m.to_square
    mover = b1.piece_at(mover_sq)
    king_v = b1.king(V)
    # where is P now (it did not move: V is to move after m)
    p_sq = lost_sq
    attackers_after = list(b1.attackers(A, p_sq))
    # m itself took P: the frame to read is the one BEFORE the capture (review 2026-10-07: after it, p_sq holds A's own
    # capturer and its defenders were read as P's attackers -- 51 of 95 such captures were "trapped piece")
    direct = m.to_square == lost_sq
    bf = b0 if direct else b1                                      # the board on which P stands attacked
    attackers_f = list(bf.attackers(A, p_sq))
    # discovered attack / check: a slider of A that attacks P (or the king) now and did not before, and m's origin lay on its line
    for s in attackers_after if not direct else []:
        pc = b1.piece_at(s)
        if pc.piece_type in SLIDERS and s != mover_sq and m.from_square in _between_line(s, p_sq) and s not in b0.attackers(A, p_sq):
            labels.append("discovered attack"); break
    if b1.is_check():
        checkers = list(b1.checkers())
        if len(checkers) >= 2:
            labels.append("double check")
        elif checkers and checkers[0] != mover_sq and m.from_square in _between_line(checkers[0], king_v):
            labels.append("discovered check")
    # fork / double attack by the mover
    if mover is not None and mover_sq in attackers_after:
        others = [t for t in b1.attacks(mover_sq) if t != p_sq and (q := b1.piece_at(t)) is not None and q.color == V
                  and (q.piece_type == chess.KING or VAL[q.piece_type] >= 3 or (VAL[q.piece_type] > VAL[mover.piece_type]))]
        if others:
            kinds = {b1.piece_at(t).piece_type for t in others} | {P.piece_type}
            if chess.KING in kinds and chess.QUEEN in kinds:
                labels.append("royal fork")
            elif mover.piece_type == chess.KNIGHT:
                labels.append("knight fork")
            elif mover.piece_type == chess.PAWN:
                labels.append("pawn fork")
            elif chess.KING in kinds:
                labels.append("double attack with check")
            else:
                labels.append({chess.QUEEN: "queen fork", chess.ROOK: "rook fork", chess.BISHOP: "bishop fork", chess.KING: "king fork"}.get(mover.piece_type, "double attack with check"))
    # skewer / pin along a line through P
    for s in attackers_f:
        pc = bf.piece_at(s)
        if pc.piece_type not in SLIDERS:
            continue
        beyond = [t for t in chess.SQUARES if chess.BB_SQUARES[t] & chess.BB_RAYS[s][p_sq] and t != s and t != p_sq
                  and p_sq in _between_line(s, t) and not [u for u in _between_line(p_sq, t) if bf.piece_at(u)]]
        for t in beyond:
            q = bf.piece_at(t)
            if q is not None and q.color == V and _attacks_through(bf, s, t):
                if q.piece_type == chess.KING or VAL[q.piece_type] > VAL[P.piece_type]:
                    labels.append("absolute pin" if q.piece_type == chess.KING else "relative pin"); break
    # skewer the other way: a slider attacks a dearer piece of V in front, and P behind it on the same line
    for s in list(bf.pieces(chess.QUEEN, A)) + list(bf.pieces(chess.ROOK, A)) + list(bf.pieces(chess.BISHOP, A)):
        for front in bf.attacks(s):
            q = bf.piece_at(front)
            if q is None or q.color != V or front == p_sq or not _attacks_through(bf, s, p_sq):
                continue
            if front in _between_line(s, p_sq) and not [u for u in _between_line(front, p_sq) if bf.piece_at(u)] and VAL[q.piece_type] > VAL[P.piece_type]:
                labels.append("skewer")
    # pinned defender
    defenders = list(bf.attackers(V, p_sq))
    if defenders and all(bf.is_pinned(V, d) for d in defenders) and attackers_f:
        labels.append("pinned defender")
    # capture of the defender
    if b0.is_capture(m) and m.to_square in b0.attackers(V, lost_sq):
        labels.append("capture of the defender")
    # deflection / overload: a defender of P at the start moved away at V's next move, then P was taken
    if len(seq) >= 3 and not direct:
        v_move = seq[1]
        b2 = b1.copy(stack=False); b2.push(v_move)                # (review 2026-10-07: the destination was tested against the
        if v_move.from_square in b0.attackers(V, lost_sq) and p_sq not in b2.attacks(v_move.to_square) \
                and any(x.to_square == p_sq for x in seq[2:4]):   # defenders' squares BEFORE the move: always true)
            labels.append("deflection / overload")
    # trapped piece: attacked after m, no move of P lands on a square where it is not attacked (or capturing with safety);
    # not read when m took P itself or V is in check (P's moves are then illegal for another reason)
    if attackers_after and P.piece_type != chess.PAWN and not direct and not b1.is_check():
        safe = False
        for mv in b1.legal_moves if b1.turn == V else []:
            if mv.from_square == p_sq:
                b2 = b1.copy(stack=False); b2.push(mv)
                if not b2.attackers(A, mv.to_square) or VAL.get((b1.piece_at(mv.to_square) or chess.Piece(chess.PAWN, A)).piece_type, 0) >= VAL[P.piece_type]:
                    safe = True; break
        if not safe:
            labels.append("trapped piece")
    # x-ray: an A slider attacks P through a piece of A (battery) that also attacks P
    for s in bf.pieces(chess.QUEEN, A) | bf.pieces(chess.ROOK, A) | bf.pieces(chess.BISHOP, A):
        through = _ray_through(bf, s, p_sq)
        if through is not None and _attacks_through(bf, s, p_sq) and (q := bf.piece_at(through)) is not None and q.color == A and through in attackers_f:
            labels.append("x-ray"); break
    # interference: m put a piece between P and one of its defenders
    for d in b0.attackers(V, lost_sq):
        if mover_sq in _between_line(d, lost_sq):
            labels.append("interference"); break
    # promotion / breakthrough in the window
    bb = b0.copy(stack=False); seventh = 6 if A == chess.WHITE else 1
    for i, x in enumerate(seq):                                    # A's own pawn promotes or reaches its seventh (review
        pc = bb.piece_at(x.from_square)                            # 2026-10-07: read from the start board, any colour)
        if i % 2 == 0 and pc is not None and pc.color == A and pc.piece_type == chess.PAWN and (x.promotion or chess.square_rank(x.to_square) == seventh):
            labels.append("promotion / breakthrough"); break
        bb.push(x)
    # back rank: a rook or queen of A gives check on V's home rank in the window
    home = 7 if V == chess.BLACK else 0
    bb = b0.copy(stack=False)
    for i, x in enumerate(seq):
        bb.push(x)
        if i % 2 == 0 and bb.is_check() and chess.square_rank(x.to_square) == home and (bb.piece_at(x.to_square) or chess.Piece(chess.PAWN, A)).piece_type in (chess.ROOK, chess.QUEEN):
            labels.append("back rank"); break
    # zwischenzug: A's capture of P came after an in-between check or bigger capture by A
    cap_idx = next((i for i, x in enumerate(seq) if x.to_square == p_sq and i % 2 == 0), None)
    if cap_idx is not None and cap_idx >= 2:
        bb = b0.copy(stack=False)
        for i, x in enumerate(seq[:cap_idx]):
            if i % 2 == 0 and i > 0 and (bb.gives_check(x) or (bb.is_capture(x) and VAL.get((bb.piece_at(x.to_square) or chess.Piece(chess.PAWN, V)).piece_type, 0) >= VAL[P.piece_type])):
                labels.append("zwischenzug"); break
            bb.push(x)
    # en passant
    if seq and b0.is_en_passant(seq[0]):
        labels.append("en passant")
    # insufficiently defended: nothing else, P was defended at the start and the count on its square loses
    if not labels and b0.attackers(V, lost_sq) and attackers_after:
        labels.append("insufficiently defended")
    return labels


def main(tag: str = "c300") -> None:
    """Label every fired tactic of a census (tactic_coverage.events_of: a visibly safe or defended piece lost within the
    window with the loser's material down and a Stockfish step) and write them for the LLM's review:
    data/derived/tactic_census/fired_tactics_<tag>_labelled.json. Prints the label counts (OBS)."""
    import json
    import sys
    from collections import Counter, defaultdict

    from catspace.io import paths
    from .tactic_census_measure import load
    from .tactic_coverage import WINDOW, events_of, game_moves
    rows = load(tag)
    by = defaultdict(dict)
    for r in rows:
        by[r["game"]][r["ply"]] = r
    out = []; counts = Counter(); n = 0
    for g in sorted(by):
        plies = by[g]; moves = game_moves(plies); last = max(plies)
        for t, sq, pt, kind, forced, net, c in events_of(plies):
            if kind == "hanging" or not forced:
                continue
            r = plies[t]; b = chess.Board(r["fen"])
            seq = [moves[k] for k in range(t, min(t + WINDOW, last)) if k in moves]
            labels = label(b, moves[t], sq, seq)
            n += 1; counts.update(labels or ["(unlabelled)"])
            out.append(dict(game=g, ply=t, fen=r["fen"], move=moves[t].uci(), lost=chess.square_name(sq), piece=chess.piece_name(pt),
                            kind=kind, net=net, seen=c, labels=labels, seq=[m.uci() for m in seq]))
    p = paths.derived("tactic_census", f"fired_tactics_{tag}_labelled.json")
    json.dump(out, open(p, "w"), indent=0)
    print(f"OBS {n} fired tactics of {tag} labelled -> {p}")
    print("OBS   labels: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))


if __name__ == "__main__":
    import sys
    main(sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "c300")
