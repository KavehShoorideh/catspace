"""THE FRAME MEMORY, audited (Kaveh 2026-10-09: "make sure you do a bug sweep before trusting any results out of your new
system").

Every census game is replayed twice per position: the planner with the game's PlanState (rust/analyst_core/src/plan.rs:
the frame memory and the history order, kept across the game's moves) and the stateless planner. Reported:
  COST        new searches a move (fresh + re-proofs), replays, reuses, nodes, ms -- with and without the memory
  AGREEMENT   positions where the decision or the best immediate win differs; each listed -- a memory that changes an
              answer is a bug until shown otherwise (allowed: a stale "fails" can only MISS a tactic; a different
              choice among equal wins is reported apart)
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
    # FRAME PROOFS and ZONE REUSE (2026-10-11): both runs read their frames; only the memory reuses by zone
    analyst_core.set_frame_proofs(os.environ.get("CATSPACE_FRAME") == "1")
    analyst_core.set_zone_reuse(os.environ.get("CATSPACE_ZONE") == "1")
    analyst_core.set_lemma(os.environ.get("CATSPACE_LEMMA") == "1")
    analyst_core.set_cert_reuse(os.environ.get("CATSPACE_CERT") == "1")
    analyst_core.set_lemma_audit(os.environ.get("CATSPACE_LEMMA") == "1")
    st = analyst_core.PlanState()
    rows = []
    for fen in fens:
        analyst_core.zone_stats(); analyst_core.lemma_stats()
        r = st.observe(fen)
        z = analyst_core.zone_stats(); lm = analyst_core.lemma_stats()
        r0 = analyst_core.plan_trace(fen)
        rows.append(dict(fen=fen, dec=r[1], dec0=r0[1], imm=r[5], imm0=r0[5], proofs=r[2], proofs0=r0[2], nodes=r[3],
                         nodes0=r0[3], ms=r[4], ms0=r0[4], stats=r[6], zone=z, lemma=lm))
    return gid, rows


def run(tag: str, games: int, workers: int) -> None:
    from multiprocessing import Pool
    from .waiting_frames import _games
    t0 = time.time()
    with Pool(workers) as p:
        res = list(p.imap_unordered(_game, _games(tag, games), chunksize=1))
    rows = [r for _, rs in res for r in rs]
    n = len(rows)
    s = [sum(r["stats"][i] for r in rows) for i in range(12)]
    new = sorted(r["stats"][0] + r["stats"][4] for r in rows)
    print(f"OBS {len(res)} games, {n} positions ({time.time() - t0:.0f} s on {workers} processes)")
    print(f"OBS   with the memory: NEW searches a move median {new[n // 2]}, mean {sum(new) / n:.1f}, 90th percentile {new[int(0.9 * n)]}; "
          f"replays {s[1] / n:.1f}, exact reuses {s[2] / n:.1f}, ZONE reuses {s[9] / n:.1f}, CERTIFICATE reuses {s[10] / n:.1f} (broke {s[11] / n:.1f}), reused failures {s[3] / n:.1f} a move; nodes {sum(r['nodes'] for r in rows) / n:,.0f}; "
          f"ms {sum(r['ms'] for r in rows) / n:.1f}")
    p0 = sorted(r["proofs0"] for r in rows)
    print(f"OBS   without:         searches a move median {p0[n // 2]}, mean {sum(p0) / n:.1f}, 90th percentile {p0[int(0.9 * n)]}; "
          f"nodes {sum(r['nodes0'] for r in rows) / n:,.0f}; ms {sum(r['ms0'] for r in rows) / n:.1f}")
    z = [sum(r["zone"][i] for r in rows) for i in range(12)]
    print(f"OBS   dependency of a proof: core {z[8] / max(z[11], 1):.1f}, changed exchanges {z[9] / max(z[11], 1):.1f}, leaf reads {z[10] / max(z[11], 1):.1f} squares ({z[11]} proofs)")
    print(f"OBS   memory a move: fresh {s[0] / n:.1f}, re-proofs {s[4] / n:.1f}; ZONE attempts {z[0] / n:.2f} -- failed on pieces {z[1]}, attacks onto D {z[2]}, the one-step set {z[3]}, attacks onto it {z[4]}, the index {z[5]}; "
          f"|D| mean {z[6] / max(z[0], 1):.1f}, one-step set mean {z[7] / max(z[0] - z[1] - z[2], 1):.1f}")
    lm = [sum(r["lemma"][i] for r in rows) for i in range(15)]
    names = ["noKing", "target", "noAttack", "exchange", "escape", "check", "pin", "promotion", "defence", "intermezzo", "value<goal"]
    if sum(lm):
        print(f"OBS   LEMMA: holds {lm[0]} ({100 * lm[0] / max(sum(lm[:12]), 1):.1f}% of one-target questions); undecided by stopper " + ", ".join(f"{nm} {lm[1 + i]}" for i, nm in enumerate(names)))
        print(f"OBS   LEMMA audit vs the 2-ply search (where the lemma holds): agree {lm[12]}, FALSE CLAIMS {lm[13]}, search ran out {lm[14]}")
    print(f"OBS   gates fired a move: lone-attack {s[5] / n:.1f}, frozen (pin) {s[6] / n:.2f}, escape filter {s[7] / n:.2f}, cut by the history order {s[8] / n:.2f}")
    kinds = Counter(); ex = []
    for r in rows:
        if r["imm"] == r["imm0"] and r["dec"] == r["dec0"]:
            continue
        g, g0 = (r["imm"] or ("", "", 0, ""))[2], (r["imm0"] or ("", "", 0, ""))[2]
        if g < g0:
            k = "memory found a SMALLER immediate win (a stale 'fails' missed a tactic)"
        elif g > g0:
            k = "memory found a BIGGER immediate win (must be explained)"
        elif r["imm"] != r["imm0"]:
            k = "same gain, another move (order or history)"
        else:
            d, d0 = r["dec"] or ("", "", 0, ""), r["dec0"] or ("", "", 0, "")
            if (d[0], d[1], d[2]) == (d0[0], d0[1], d0[2]):
                k = "same decided move and value, described differently"
            elif d[0] == d0[0] and d[2] == d0[2]:
                k = "same kind and value of decision, another move (e.g. another defence of equal worth)"
            else:
                k = "DIFFERENT decision (kind or value)"
        kinds[k] += 1
        if len(ex) < 10 and (k.startswith("DIFFERENT") or "SMALLER" in k or "BIGGER" in k):
            ex.append((k, r["imm"], r["imm0"], r["dec"], r["dec0"], r["fen"]))
    print(f"OBS   AGREEMENT: {n - sum(kinds.values())} of {n} positions identical; differences {dict(kinds)}")
    for k, i, i0, d, d0, fen in ex:
        print(f"OBS     {k}: memory {i[3] if i else None} / {d[3] if d else None}; stateless {i0[3] if i0 else None} / {d0[3] if d0 else None}  {fen}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--frame", action="store_true", help="frame proofs in both runs")
    ap.add_argument("--zone", action="store_true", help="zone reuse in the memory run (needs --frame or --lemma)")
    ap.add_argument("--cert", action="store_true", help="failed questions re-checked by their stored refutation before any search")
    ap.add_argument("--lemma", action="store_true", help="the one-move threat lemma decides 2-ply questions first (audited against the search)")
    a = ap.parse_args(argv)
    os.environ["CATSPACE_FRAME"] = "1" if a.frame else "0"
    os.environ["CATSPACE_ZONE"] = "1" if a.zone else "0"
    os.environ["CATSPACE_LEMMA"] = "1" if a.lemma else "0"
    os.environ["CATSPACE_CERT"] = "1" if a.cert else "0"
    run(a.tag, a.games, a.workers)


if __name__ == "__main__":
    main()
