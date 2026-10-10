"""THE PIECE LEDGER, MEASURED (Kaveh 2026-10-10: "if my bishop -- in the last move I calculated what my bishop is
doing. I don't want to recalculate this, I'll just cache it; I'll only recalculate it if something changed regarding my
bishop"; "keep track of the jobs of the pieces"). On consecutive census positions of a game: per piece that did not
move, its JOBS (threat.rs: the threats that appear when it is lifted) before and after the move. The cache rule: reuse
when the move touched none of the piece's SUPPORT (its square, the squares it attacks, its job targets, and the squares
between it and them); the rule's miss = a job set that changed although the support was untouched."""
import os
from collections import Counter
from multiprocessing import Pool

import chess

from catspace.incubator.analyst.tactic_census_measure import load


def _jobs(fen):
    import analyst_core
    return {sq: sorted(j) for sq, j in analyst_core.threat_check(fen)[2]}


def lines(b: chess.Board, y: int) -> set:
    """The squares whose change can change who attacks y (the path lemma): along each queen ray from y up to and
    including the SECOND piece met (an x-ray / battery), y's knight squares, king squares and pawn squares."""
    out = {y} | set(chess.SquareSet(chess.BB_KNIGHT_ATTACKS[y])) | set(chess.SquareSet(chess.BB_KING_ATTACKS[y]))
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        f, r, met = chess.square_file(y), chess.square_rank(y), 0
        while met < 2:
            f += df; r += dr
            if not (0 <= f < 8 and 0 <= r < 8):
                break
            sq = chess.square(f, r); out.add(sq)
            if b.piece_at(sq): met += 1
    return out


def one(pair):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    f0, f1 = pair
    b0, b1 = chess.Board(f0), chess.Board(f1)
    mv = None
    for m in b0.legal_moves:
        b0.push(m)
        hit = b0.board_fen() == b1.board_fen()
        b0.pop()
        if hit:
            mv = m
            break
    if mv is None:
        return Counter()
    touched = {mv.from_square, mv.to_square}
    if b0.is_castling(mv):
        touched |= {chess.H1, chess.F1, chess.A1, chess.D1} if b0.turn else {chess.H8, chess.F8, chess.A8, chess.D8}
    if b0.is_en_passant(mv):
        touched.add(mv.to_square + (-8 if b0.turn else 8))
    j0, j1 = _jobs(f0), _jobs(f1)
    c = Counter()
    for s, p in b0.piece_map().items():
        if s in touched or p.piece_type == chess.KING or b1.piece_at(s) != p:
            continue
        sup = {s} | set(b0.attacks(s))
        for t, _, _ in j0.get(chess.square_name(s), []):
            ts = chess.parse_square(t); sup.add(ts); sup |= set(chess.SquareSet(chess.between(s, ts)))
        same = j0.get(chess.square_name(s), []) == j1.get(chess.square_name(s), [])
        hit = bool(sup & touched)
        c[("touched" if hit else "untouched", "same" if same else "changed")] += 1
        # v2: the lines onto every square the piece bears on, its job targets and itself (the path lemma's support)
        sup2 = set()
        for y in sup:
            sup2 |= lines(b0, y)
        hit2 = bool(sup2 & touched)
        c[("v2 " + ("touched" if hit2 else "untouched"), "same" if same else "changed")] += 1
        # v3: only what a job can come from -- the piece, its job targets, the pieces of its own side it guards, their
        # king if it is a job target (its box), and the lines onto each of these
        core = {s} | {y for y in b0.attacks(s) if b0.color_at(y) == p.color}
        for t, _, mate in j0.get(chess.square_name(s), []):
            ts = chess.parse_square(t); core.add(ts)
            if mate: core |= set(chess.SquareSet(chess.BB_KING_ATTACKS[ts]))
        sup3 = set()
        for y in core:
            sup3 |= lines(b0, y)
        hit3 = bool(sup3 & touched)
        c[("v3 " + ("touched" if hit3 else "untouched"), "same" if same else "changed")] += 1
        c[("had jobs" if j0.get(chess.square_name(s)) else "no jobs", "same" if same else "changed")] += 1
    return c


if __name__ == "__main__":
    rows = load("c300")
    at = {(r["game"], r["ply"]): r["fen"] for r in rows}
    pairs = [(at[k], at[(k[0], k[1] + 1)]) for k in at if (k[0], k[1] + 1) in at][:6000]
    with Pool(os.cpu_count()) as p:
        tot = sum(p.map(one, pairs, chunksize=16), Counter())
    n = sum(v for k, v in tot.items() if k[0] in ("touched", "untouched"))
    un = tot[("untouched", "same")] + tot[("untouched", "changed")]
    print(f"OBS {len(pairs)} moves, {n} pieces that stayed put: jobs unchanged {100 * (tot[('touched','same')] + tot[('untouched','same')]) / n:.1f}%")
    print(f"OBS   support untouched by the move: {un} ({100 * un / n:.0f}% of pieces -- reusable); of these the jobs CHANGED anyway "
          f"{tot[('untouched','changed')]} ({100 * tot[('untouched','changed')] / max(un, 1):.1f}%: the cache rule's misses)")
    t = tot[("touched", "same")] + tot[("touched", "changed")]
    print(f"OBS   support touched: {t}; jobs changed {tot[('touched','changed')]} ({100 * tot[('touched','changed')] / max(t, 1):.0f}%)")
    u2 = tot[("v2 untouched", "same")] + tot[("v2 untouched", "changed")]
    print(f"OBS   v2 support (the lines onto every square it bears on): untouched {u2} ({100 * u2 / n:.0f}% reusable); jobs "
          f"changed anyway {tot[('v2 untouched','changed')]} ({100 * tot[('v2 untouched','changed')] / max(u2, 1):.2f}%)")
    u3 = tot[("v3 untouched", "same")] + tot[("v3 untouched", "changed")]
    print(f"OBS   v3 support (lines onto the piece, its job targets, the pieces it guards): untouched {u3} ({100 * u3 / n:.0f}% reusable); "
          f"jobs changed anyway {tot[('v3 untouched','changed')]} ({100 * tot[('v3 untouched','changed')] / max(u3, 1):.2f}%)")
    print(f"OBS   pieces with jobs before the move: same {tot[('had jobs','same')]}, changed {tot[('had jobs','changed')]}")
