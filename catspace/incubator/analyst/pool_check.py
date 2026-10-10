"""THE WAITING-FRAMES POOL IN THE PLANNER, measured (Kaveh 2026-10-10: wire the existing pool, waiting.rs, instead of
rebuilding it). Census games replayed through a game's PlanState with the pool off and on (set_pool): immediate wins,
the wins the pool's active frames add, the pool's armed threats, the "hands" warnings on decisions, time. Then
Stockfish (depth 12): the pool's armed decisions (the preparing move's loss against the best move), and the decisions
the pool warned about against the same number of unwarned decisions (the decision move's loss) -- a warning is worth
keeping only if warned decisions lose more."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter


def _game(args):
    gid, fens, on = args
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    analyst_core.set_pool(on)
    st = analyst_core.PlanState()
    out = []
    for fen in fens:
        tr, dec, proofs, nodes, ms, imm, _ = st.observe(fen)
        mv = None
        if dec:
            mv = dec[1] if dec[0] in ("", "defend", "cheapest") else dec[0]
        out.append(dict(fen=fen, imm=imm, dec=dec, mv=mv, ms=ms, proofs=proofs,
                        frame_win=bool(imm and "from the frame pool" in (imm[3] or "")),
                        pool_arm=bool(dec and dec[0] not in ("", "defend", "cheapest") and "from the frame pool" in (dec[3] or "")),
                        hands=[l for l in tr if l.startswith("hands:")]))
    return gid, out


_ENG = None


def _sf(args):
    global _ENG
    import chess, chess.engine
    fen, mv = args
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = chess.Board(fen)
    best = _ENG.analyse(b, chess.engine.Limit(depth=12))["score"].pov(b.turn).score(mate_score=3000)
    a = b.copy(); a.push_uci(mv)
    got = _ENG.analyse(a, chess.engine.Limit(depth=12))["score"].pov(b.turn).score(mate_score=3000)
    return best - got


def main(argv=None):
    from multiprocessing import Pool
    from .waiting_frames import _games
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    games = list(_games(a.tag, a.games))
    res = {}
    for on in (False, True):
        t0 = time.time()
        with Pool(a.workers) as p:
            res[on] = {g: rows for g, rows in p.imap_unordered(_game, [(g, f, on) for g, f in games], chunksize=1)}
        print(f"TIME pool {'on ' if on else 'off'}: {time.time() - t0:.0f} s", flush=True)
    off = [r for g in sorted(res[False]) for r in res[False][g]]
    on = [r for g in sorted(res[True]) for r in res[True][g]]
    n = len(on)
    for name, rs in (("off", off), ("on ", on)):
        ms = sorted(r["ms"] for r in rs)
        print(f"OBS pool {name}: {n} positions, immediate wins {sum(bool(r['imm']) for r in rs)}, armed decisions "
              f"{sum(bool(r['dec'] and r['dec'][0] not in ('', 'defend', 'cheapest')) for r in rs)}, proofs mean {sum(r['proofs'] for r in rs) / n:.1f}, ms median {ms[n // 2]:.1f}")
    gained = sum(1 for x, y in zip(off, on) if y["imm"] and not x["imm"])
    lost = sum(1 for x, y in zip(off, on) if x["imm"] and not y["imm"])
    print(f"OBS immediate wins gained {gained}, lost {lost}; wins from the pool's active frames {sum(r['frame_win'] for r in on)}; "
          f"armed decisions from the pool {sum(r['pool_arm'] for r in on)}; decisions warned 'hands them a tactic' {sum(bool(r['hands']) for r in on)}")
    # Stockfish: the pool's armed decisions; warned vs unwarned decisions
    arms = [(r["fen"], r["mv"]) for r in on if r["pool_arm"]]
    warned = [(r["fen"], r["mv"]) for r in on if r["hands"] and r["mv"]]
    unwarned = random.Random(0).sample([(r["fen"], r["mv"]) for r in on if not r["hands"] and r["mv"]], min(len(warned), 2000))
    t1 = time.time()
    with Pool(a.workers) as p:
        la = p.map(_sf, arms, chunksize=2); lw = p.map(_sf, warned, chunksize=2); lu = p.map(_sf, unwarned, chunksize=2)
    # the pool's immediate wins by claimed gain, against a sample of the planner's own played wins
    pw1 = [(r["fen"], r["imm"][1]) for r in on if r["frame_win"] and r["imm"][2] == 1]
    pw2 = [(r["fen"], r["imm"][1]) for r in on if r["frame_win"] and r["imm"][2] >= 2]
    own = random.Random(1).sample([(r["fen"], r["imm"][1]) for r in off if r["imm"] and r["dec"] and r["dec"][0] == ""], min(400, sum(1 for r in off if r["imm"])))
    with Pool(a.workers) as p:
        l1 = p.map(_sf, pw1, chunksize=2); l2 = p.map(_sf, pw2, chunksize=2); lo = p.map(_sf, own, chunksize=2)
    print(f"TIME Stockfish: {time.time() - t1:.0f} s")
    la, lw, lu = la, lw, lu
    for name, v in (("pool wins claiming 1 pawn", l1), ("pool wins claiming 2+", l2), ("the planner's own played wins (sample)", lo)):
        print(f"OBS {name}: n {len(v)}, loss median {med(v) if (med := (lambda v: sorted(v)[len(v) // 2] if v else None)) else None} cp, within 50 cp {100 * sum(x <= 50 for x in v) / max(len(v), 1):.0f}%, losing 150+ {100 * sum(x >= 150 for x in v) / max(len(v), 1):.0f}%")
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    for name, v in (("pool armed decisions", la), ("warned decisions", lw), ("unwarned decisions (sample)", lu)):
        print(f"OBS {name}: n {len(v)}, loss vs Stockfish's best median {med(v)} cp, losing 150+ cp {sum(x >= 150 for x in v)} ({100 * sum(x >= 150 for x in v) / max(len(v), 1):.0f}%)")
    for r in [r for r in on if r["hands"]][:4]:
        print(f"  {r['fen']}  {r['dec'] and r['dec'][3]} | {r['hands'][0][:120]}")
    for r in [r for r in on if r["pool_arm"]][:4]:
        print(f"  {r['fen']}  {r['dec'][3]}")


if __name__ == "__main__":
    main()
