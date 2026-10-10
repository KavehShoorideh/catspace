"""The pool's seeds in the planner and the direct blunder check, measured (2026-10-10; pool_check.py measured the full
pool). Census games through PlanState with FRAME_SEEDS off and on (the direct after-check on in both): immediate wins,
the wins the frame seeds add, proofs, time; Stockfish (depth 12) on the added wins against the planner's own; and the
decisions warned "after: ... lets them" against unwarned decisions."""
from __future__ import annotations

import argparse
import os
import random
import time

from .pool_check import _sf


def _game(args):
    gid, fens, on = args
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    analyst_core.set_frame_seeds(on)
    st = analyst_core.PlanState()
    out = []
    t0 = time.time()
    for fen in fens:
        tr, dec, proofs, nodes, ms, imm, _ = st.observe(fen)
        mv = (dec[1] if dec[0] in ("", "defend", "cheapest") else dec[0]) if dec else None
        out.append(dict(fen=fen, imm=imm, dec=dec, mv=mv, proofs=proofs, ms=ms,
                        warned=any(l.startswith("after:") for l in tr),
                        frame_win=bool(imm and any(l.startswith("frame seed:") and imm[3].split(" wins")[0] in l for l in tr))))
    return gid, out, time.time() - t0


def main(argv=None):
    from multiprocessing import Pool
    from .waiting_frames import _games
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    games = list(_games("c300", a.games))
    res = {}
    for on in (False, True):
        t0 = time.time()
        with Pool(a.workers) as p:
            res[on] = {g: rows for g, rows, _ in p.imap_unordered(_game, [(g, f, on) for g, f in games], chunksize=1)}
        print(f"TIME frame seeds {'on ' if on else 'off'}: {time.time() - t0:.0f} s", flush=True)
    off = [r for g in sorted(res[False]) for r in res[False][g]]
    on = [r for g in sorted(res[True]) for r in res[True][g]]
    n = len(on)
    for name, rs in (("off", off), ("on ", on)):
        ms = sorted(r["ms"] for r in rs)
        print(f"OBS frame seeds {name}: {n} positions, immediate wins {sum(bool(r['imm']) for r in rs)}, proofs mean {sum(r['proofs'] for r in rs) / n:.1f}, ms median {ms[n // 2]:.1f}")
    added = [(y["fen"], y["imm"][1]) for x, y in zip(off, on) if y["imm"] and not x["imm"]]
    lost = sum(1 for x, y in zip(off, on) if x["imm"] and not y["imm"])
    own = random.Random(1).sample([(r["fen"], r["imm"][1]) for r in off if r["imm"] and r["dec"] and r["dec"][0] == ""], 300)
    warned = [(r["fen"], r["mv"]) for r in on if r["warned"] and r["mv"]]
    unwarned = random.Random(0).sample([(r["fen"], r["mv"]) for r in on if not r["warned"] and r["mv"]], min(len(warned), 1000))
    with Pool(a.workers) as p:
        la = p.map(_sf, added, chunksize=2); lo = p.map(_sf, own, chunksize=2)
        lw = p.map(_sf, warned, chunksize=2); lu = p.map(_sf, unwarned, chunksize=2)
    print(f"OBS immediate wins added {len(added)}, lost {lost}")
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    for name, v in (("added wins", la), ("the planner's own wins (sample)", lo), ("decisions warned by the after-check", lw), ("unwarned decisions (sample)", lu)):
        print(f"OBS {name}: n {len(v)}, loss vs Stockfish's best median {med(v)} cp, within 50 cp {100 * sum(x <= 50 for x in v) / max(len(v), 1):.0f}%, losing 150+ {100 * sum(x >= 150 for x in v) / max(len(v), 1):.0f}%")


if __name__ == "__main__":
    main()
