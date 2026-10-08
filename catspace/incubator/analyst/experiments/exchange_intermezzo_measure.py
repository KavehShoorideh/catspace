"""Do independent frames add? On frames failing ONLY noOtherCapture: predicted = see(s) - their best static capture elsewhere
after our capture (0 if none pays); compare with the exact bridge."""
import chess
from collections import Counter
from catspace.incubator.analyst.exchange_law import HYPS, hypotheses, see, participants, cheapest_capture, bridge
from catspace.incubator.analyst.tactic_census_measure import load
import sys, time
rows = load("c300")[:8000]; misses = []; t0 = time.time()
n = ok_pred = ok_naive = 0; gap = Counter(); disjoint = Counter()
for i, r in enumerate(rows):
    if i % 1000 == 0: print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(rows)} positions, {n} frames, {time.time()-t0:.0f}s", flush=True)
    b = chess.Board(r["fen"]); c = b.turn
    for s, pc in b.piece_map().items():
        if pc.color == c or pc.piece_type == chess.KING or not b.attackers(c, s):
            continue
        h = hypotheses(b, s)
        fails = [k for k in HYPS if not h[k]]
        if not h["target"] or "noOtherCapture" not in fails or set(fails) - {"noOtherCapture", "noArrival", "noInterpose"}:
            continue
        A, D = participants(b, s); v = see(b, s)[0]
        b1 = b.copy(stack=False); b1.push(cheapest_capture(b, s, A))
        best = 0; where = None
        for q, p in b1.piece_map().items():
            if p.color == c and p.piece_type != chess.KING and q != s and b1.attackers(not c, q):
                g = see(b1, q)[0]
                if g > best: best, where = g, q
        pred = v - best
        depth = min(len(A) + len(D) + 2, 4)
        exact = bridge(b, s, depth)
        n += 1; ok_pred += exact >= pred - 1e-9; ok_naive += exact >= v - 1e-9
        if exact < pred - 1e-9: misses.append((r['fen'], chess.square_name(s), v, best, chess.square_name(where) if where is not None else '-', exact, depth)); print('MISS', misses[-1], flush=True)
        gap[int(round(exact - pred))] += 1
        if where is not None:
            parts = set(A) | set(D)
            touch = bool(set(b1.attackers(not c, where)) & parts) or bool(set(b1.attackers(c, where)) & parts)
            disjoint["frames touch" if touch else "disjoint"] += 1
            disjoint[("touch" if touch else "disjoint", "holds" if exact >= pred - 1e-9 else "FAILS")] += 1
print(f"OBS intermezzo frames (only noOtherCapture fails) in all positions (noOtherCapture fails, possibly with noArrival or noInterpose): {n}; exact >= see {ok_naive} ({100*ok_naive/max(n,1):.0f}%); exact >= see - their best capture elsewhere {ok_pred} ({100*ok_pred/max(n,1):.0f}%)")
print("OBS   exact minus predicted, histogram: " + ", ".join(f"{k:+d}: {v}" for k, v in sorted(gap.items())))
print("OBS   by whether the other frame shares a participant with s: " + ", ".join(f"{k}: {v}" for k, v in sorted(disjoint.items(), key=str)))
