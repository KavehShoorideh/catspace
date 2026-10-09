"""ONE POOL OF FACTS FOR BOTH SIDES, and the choice made on it (Kaveh 2026-10-09: "one common pool of facts and reasons
and lemmas that apply both to us and to them ... interactions between the reasons we find for us and the reasons we
find for them"). The pool (analyst_core.fact_pool): our claims on the real position, theirs on the null-move position
(their reasons were we to pass), each with its frame, and the interactions between them. THE CHOICE: every legal move
m scores  our certified gain by m  minus  the largest of THEIR certified claims that SURVIVES m  -- a claim of theirs
survives unless m answers it: m moves or takes a piece of its frame, lands on or blocks one of its squares, makes
their claimed move illegal, or gives check. One computation over one pool, not two runs compared afterwards."""
from __future__ import annotations

import time

import chess


def choose(b: chess.Board) -> dict:
    import analyst_core
    t0 = time.time()
    ours, theirs, _ = analyst_core.fact_pool(b.fen())
    mine = {}
    for i, uci, name, k, cert, frame in ours:
        if cert:
            mine[uci] = max(mine.get(uci, 0), k)
    p = b.copy(stack=False); p.push(chess.Move.null()) if not b.is_check() else None
    th = [(chess.Move.from_uci(uci), name, k, {chess.parse_square(s) for s in frame}) for i, uci, name, k, cert, frame in theirs if cert and k >= 1]
    scored = []
    for m in b.legal_moves:
        nb = b.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            scored.append((1000, 1000, m, [])); continue
        gives_check = nb.is_check()
        surviving = []
        for r, name, k, frame in th:
            answered = gives_check or m.from_square in frame or m.to_square in frame or r not in nb.legal_moves
            if not answered:
                surviving.append((k, name, r))
        their = max((k for k, _, _ in surviving), default=0)
        scored.append((mine.get(m.uci(), 0) - their, mine.get(m.uci(), 0), m, surviving))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    best = scored[0]
    return dict(move=best[2], net=best[0], ours=best[1], their_left=[(k, n, p.san(r) if p.is_legal(r) else r.uci()) for k, n, r in best[3]],
                their_claims=[(p.san(r) if p.is_legal(r) else r.uci(), n, k) for r, n, k, _ in th], ms=round(1000 * (time.time() - t0)))
