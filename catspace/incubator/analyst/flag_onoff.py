"""A FLAG ON vs OFF, on the same positions (2026-10-11): the stateless planner run twice per census position, with a
switch off and on -- decisions and immediate wins compared, time and nodes both ways. The equality audit any
performance change must pass before it is switched on (a change that keeps decisions identical is exact; one that does
not is a design change and is reported as such, with the differing positions).

  python -m catspace.incubator.analyst.flag_onoff --flag net_bound --games 20
"""
from __future__ import annotations

import argparse
import os
import time


def _game(args):
    gid, fens, flag = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    setter = getattr(analyst_core, f"set_{flag}")
    out = []
    for fen in fens:
        setter(False)
        r0 = analyst_core.plan_trace(fen)
        setter(True)
        r1 = analyst_core.plan_trace(fen)
        setter(False)
        out.append(dict(fen=fen, d0=r0[1], d1=r1[1], i0=r0[5], i1=r1[5], ms0=r0[4], ms1=r1[4], n0=r0[3], n1=r1[3], p0=r0[2], p1=r1[2]))
    return out


def run(tag: str, games: int, skip: int, flag: str, workers: int) -> None:
    from multiprocessing import Pool
    from .waiting_frames import _games
    t0 = time.time()
    jobs = [(g, f, flag) for g, f in _games(tag, skip + games)[skip:]]
    with Pool(workers) as p:
        rows = [r for rs in p.imap_unordered(_game, jobs, chunksize=1) for r in rs]
    n = len(rows)
    same = [r for r in rows if r["d0"] == r["d1"] and r["i0"] == r["i1"]]
    print(f"OBS {flag} ON/OFF, games {skip + 1}-{skip + games}, {n} positions ({time.time() - t0:.0f} s on {workers} processes): "
          f"decisions and immediate wins identical {len(same)} ({100 * len(same) / n:.2f}%); "
          f"ms a position off {sum(r['ms0'] for r in rows) / n:.1f} -> on {sum(r['ms1'] for r in rows) / n:.1f}; "
          f"proofs {sum(r['p0'] for r in rows) / n:.1f} -> {sum(r['p1'] for r in rows) / n:.1f}; "
          f"nodes {sum(r['n0'] for r in rows) / n:,.0f} -> {sum(r['n1'] for r in rows) / n:,.0f}")
    for r in [r for r in rows if r not in same][:12]:
        print(f"OBS   off {r['i0'][3] if r['i0'] else None} / {r['d0'][3] if r['d0'] else None}; on {r['i1'][3] if r['i1'] else None} / {r['d1'][3] if r['d1'] else None}  {r['fen']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", required=True, help="the switch's name after set_ (e.g. net_bound)")
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    run(a.tag, a.games, a.skip, a.flag, a.workers)


if __name__ == "__main__":
    main()
