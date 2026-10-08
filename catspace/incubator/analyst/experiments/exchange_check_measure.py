"""Does a mere checking reply (our king off every line to s, no enemy slider on its lines) break the claim? Frames failing
noCheck only through the checking-move clause (plus, allowed: noOtherCapture priced as the intermezzo, noArrival, noInterpose)."""
import time, chess
from collections import Counter
from catspace.incubator.analyst.exchange_law import HYPS, SLIDERS, hypotheses, see, participants, cheapest_capture, bridge, intermezzo, _knight_relation
from catspace.incubator.analyst.tactic_census_measure import load
rows = load("c300")[:8000]; t0 = time.time()
n = ok = ok_naive = 0; gap = Counter(); misses = []
for i, r in enumerate(rows):
    if i % 1000 == 0: print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(rows)} positions, {n} frames, {time.time()-t0:.0f}s", flush=True)
    b = chess.Board(r["fen"]); c = b.turn
    for s, pc in b.piece_map().items():
        if pc.color == c or pc.piece_type == chess.KING or not b.attackers(c, s):
            continue
        h = hypotheses(b, s)
        fails = [k for k in HYPS if not h[k]]
        if not h["target"] or "noCheck" not in fails or set(fails) - {"noCheck", "noOtherCapture", "noArrival", "noInterpose"}:
            continue
        k = b.king(c)
        if k is None or chess.BB_RAYS[s][k] or _knight_relation(s, k) or any(p.color != c and p.piece_type in SLIDERS and chess.BB_RAYS[q][k] for q, p in b.piece_map().items()):
            continue                                                   # the king-line clauses fail too: not this measurement
        A, D = participants(b, s); v = see(b, s)[0]; pred = v - intermezzo(b, s)
        exact = bridge(b, s, min(len(A) + len(D) + 2, 4))
        n += 1; ok += exact >= pred - 1e-9; ok_naive += exact >= v - 1e-9; gap[int(round(exact - pred))] += 1
        if exact < pred - 1e-9:
            misses.append((r["fen"], chess.square_name(s), v, pred, exact)); print("MISS", misses[-1], flush=True)
print(f"OBS frames failing noCheck only by a checking reply (king off the lines) in 8000 positions: {n}; exact >= see {ok_naive} ({100*ok_naive/max(n,1):.0f}%); exact >= see - intermezzo {ok} ({100*ok/max(n,1):.0f}%)")
print("OBS   exact minus predicted: " + ", ".join(f"{k:+d}: {v}" for k, v in sorted(gap.items())))
