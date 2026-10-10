"""THE NUCLEUS, measured (Kaveh 2026-10-10: start the search from what changed -- the piece ledger's events, ledger.rs).
Census games replayed through a game's PlanState twice, the nucleus on and off (set_nucleus): immediate wins, the wins
the nucleus found (a freed piece's move making two threats), proofs and time; the first traces with events printed."""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter


def _game(args):
    gid, fens, on = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    analyst_core.set_nucleus(on)
    st = analyst_core.PlanState()
    out = []
    for fen in fens:
        tr, dec, proofs, nodes, ms, imm, _ = st.observe(fen)
        out.append(dict(fen=fen, imm=imm, proofs=proofs, ms=ms, events=[l for l in tr if l.startswith("what changed") or "what can it do" in l or "it threatens" in l or "freed" in l],
                        nucleus_win=bool(imm and "freed" in (imm[3] or ""))))
    return gid, out


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
        print(f"TIME nucleus {'on ' if on else 'off'}: {time.time() - t0:.0f} s", flush=True)
    off = [r for g in sorted(res[False]) for r in res[False][g]]
    on = [r for g in sorted(res[True]) for r in res[True][g]]
    n = len(on)
    for name, rs in (("off", off), ("on ", on)):
        ms = sorted(r["ms"] for r in rs)
        print(f"OBS nucleus {name}: {n} positions, immediate wins {sum(bool(r['imm']) for r in rs)}, proofs mean {sum(r['proofs'] for r in rs) / n:.1f}, ms median {ms[n // 2]:.1f}")
    ev = sum(bool(r["events"]) for r in on)
    gained = [(b, a) for a, b in zip(off, on) if b["imm"] and not a["imm"]]
    lost = [(b, a) for a, b in zip(off, on) if a["imm"] and not b["imm"]]
    print(f"OBS positions with ledger events {ev} ({100 * ev / n:.0f}%); wins by a freed piece's double threat {sum(r['nucleus_win'] for r in on)}; "
          f"immediate wins gained {len(gained)}, lost {len(lost)}")
    shown = 0
    for r in on:
        if r["nucleus_win"] or (shown < 3 and any("what can it do" in e for e in r["events"])):
            print(f"  {r['fen']}  -> {r['imm'] and r['imm'][3]}")
            for e in r["events"][:4]:
                print(f"     {e[:200]}")
            shown += 1
            if shown >= 8:
                break


if __name__ == "__main__":
    main()
