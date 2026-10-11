"""THE FRAME AUDIT (Kaveh 2026-10-11): the frame proofs (rust/analyst_core/src/lib.rs FRAME_PROOFS: a proof reads its
frame and the index of both sides' other frames, the opponent's other moves being one pass) against the exact proofs
(every reply searched, the whole board read at the leaf) on census positions, the planner run both ways:

  FALSE CLAIMS  questions the frame proof holds and the exact proof refutes -- the pass model's unsoundness, measured
  MISSED        questions the frame proof fails and the exact proof holds -- the composed leaf's weakness
  DECISIONS     positions where the planner's decision or best immediate win differs between the two
  COST          proofs, nodes and ms a position, both ways
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter


def _game(args):
    gid, fens = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    rows = []
    for fen in fens:
        analyst_core.set_frame_proofs(False)
        r0 = analyst_core.plan_trace(fen)
        analyst_core.set_frame_proofs(True)
        analyst_core.set_frame_audit(True)
        analyst_core.frame_audit_stats()
        r = analyst_core.plan_trace(fen)
        audit = analyst_core.frame_audit_stats()
        analyst_core.set_frame_audit(False)
        analyst_core.set_frame_proofs(False)
        rows.append(dict(fen=fen, dec=r[1], dec0=r0[1], imm=r[5], imm0=r0[5], proofs=r[2], proofs0=r0[2], nodes=r[3],
                         nodes0=r0[3], ms=r[4], ms0=r0[4], audit=audit))
    return gid, rows


def run(tag: str, games: int, workers: int) -> None:
    from multiprocessing import Pool
    from .waiting_frames import _games
    t0 = time.time()
    with Pool(workers) as p:
        res = list(p.imap_unordered(_game, _games(tag, games), chunksize=1))
    rows = [r for _, rs in res for r in rs]
    n = len(rows)
    a = [sum(r["audit"][i] for r in rows) for i in range(6)]
    q = sum(a[:3]); qv = sum(a[3:])
    print(f"OBS {len(res)} games, {n} positions ({time.time() - t0:.0f} s on {workers} processes)")
    print(f"OBS   questions {q}: agree {a[0]} ({100 * a[0] / max(q, 1):.1f}%), FALSE CLAIMS {a[1]} ({100 * a[1] / max(q, 1):.2f}%), "
          f"missed {a[2]} ({100 * a[2] / max(q, 1):.2f}%)")
    print(f"OBS   frame VALUES {qv} (both finished): equal {a[3]} ({100 * a[3] / max(qv, 1):.1f}%), frame LARGER {a[4]} ({100 * a[4] / max(qv, 1):.2f}%), "
          f"frame smaller {a[5]} ({100 * a[5] / max(qv, 1):.2f}%)")
    print(f"OBS   frame proofs: proofs a position {sum(r['proofs'] for r in rows) / n:.1f}, nodes {sum(r['nodes'] for r in rows) / n:,.0f}, "
          f"ms {sum(r['ms'] for r in rows) / n:.1f}  (the audit's exact re-runs included in ms)")
    print(f"OBS   exact proofs: proofs a position {sum(r['proofs0'] for r in rows) / n:.1f}, nodes {sum(r['nodes0'] for r in rows) / n:,.0f}, "
          f"ms {sum(r['ms0'] for r in rows) / n:.1f}")
    kinds = Counter(); ex = []
    for r in rows:
        if r["imm"] == r["imm0"] and r["dec"] == r["dec0"]:
            continue
        g, g0 = (r["imm"] or ("", "", 0, ""))[2], (r["imm0"] or ("", "", 0, ""))[2]
        if g < g0:
            k = "frame found a SMALLER immediate win"
        elif g > g0:
            k = "frame found a BIGGER immediate win (a false claim or a composed gain)"
        elif r["imm"] != r["imm0"]:
            k = "same gain, another move"
        else:
            d, d0 = r["dec"] or ("", "", 0, ""), r["dec0"] or ("", "", 0, "")
            if (d[0], d[1], d[2]) == (d0[0], d0[1], d0[2]):
                k = "same decided move and value, described differently"
            elif d[0] == d0[0] and d[2] == d0[2]:
                k = "same kind and value of decision, another move"
            else:
                k = "DIFFERENT decision (kind or value)"
        kinds[k] += 1
        if len(ex) < 12 and (k.startswith("DIFFERENT") or "SMALLER" in k or "BIGGER" in k):
            ex.append((k, r["imm"], r["imm0"], r["dec"], r["dec0"], r["fen"]))
    print(f"OBS   DECISIONS: {n - sum(kinds.values())} of {n} positions identical; differences {dict(kinds)}")
    for k, i, i0, d, d0, fen in ex:
        print(f"OBS     {k}: frame {i[3] if i else None} / {d[3] if d else None}; exact {i0[3] if i0 else None} / {d0[3] if d0 else None}  {fen}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    run(a.tag, a.games, a.workers)


if __name__ == "__main__":
    main()
