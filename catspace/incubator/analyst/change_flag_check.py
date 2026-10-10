"""THE CHANGE FLAG FOR BLUNDERS, measured (Kaveh 2026-10-10: "keep track of changes -- see how this change changes the
state of another piece. If a piece is attacked twice and defended twice, it's stable ... now suddenly we enable this
tactic for them because we remove a defender ... that change should flag the blunder"). For sampled census positions
and every legal move M: the CHANGE FLAG = a threat of theirs that M creates (threat.rs static threats -- the static
exchange on each piece, promotion, trapped, mate by checks -- after M against before M); the FULL reasoning = their best
proved immediate win after M exceeding their best before (the planner's immediate win with them to move). Recall of the
flag on the full reasoning's new wins, and how many moves it flags."""
import os
import random
from collections import Counter
from multiprocessing import Pool

import chess


def one(fen):
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    b = chess.Board(fen)
    before_static = {(t, m) for t, v, m in analyst_core.threat_check(fen)[1] if v > 0}
    # their best immediate win before (they to move = the null move)
    nb = b.copy(); nb.push(chess.Move.null())
    if b.is_check():
        return Counter()
    imm0 = analyst_core.plan_trace(nb.fen())[5]
    v0 = imm0[2] if imm0 else 0
    c = Counter()
    for m in b.legal_moves:
        a = b.copy(); a.push(m)
        if a.is_checkmate():
            continue
        st = analyst_core.threat_check(a.fen())[0]                      # their threats, they to move
        flag = any(v > 0 and (t, mt) not in before_static for t, v, mt in st)
        imm = analyst_core.plan_trace(a.fen())[5]
        full = bool(imm and imm[2] > v0)
        c[(flag, full)] += 1
    return c


if __name__ == "__main__":
    from catspace.incubator.analyst.waiting_frames import _games
    fens = [f for _, fs in _games("c300", 100) for f in fs]
    fens = random.Random(3).sample(fens, 600)
    with Pool(os.cpu_count()) as p:
        tot = sum(p.map(one, fens, chunksize=4), Counter())
    n = sum(tot.values())
    tp, fn, fp = tot[(True, True)], tot[(False, True)], tot[(True, False)]
    print(f"OBS {len(fens)} positions, {n} moves: the full reasoning finds a new win of theirs after {tp + fn} moves; "
          f"the change flag catches {tp} ({100 * tp / max(tp + fn, 1):.1f}%), misses {fn}; flags {tp + fp} moves in all "
          f"({100 * (tp + fp) / n:.0f}% of moves; {fp} without a new proved win)")
