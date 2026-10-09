"""WAITING FRAMES, measured on real games (Kaveh 2026-10-09: "keep the frames and the facts in the frames ... a live
index of who's waiting on what condition"; "each frame also needs to know what is invalidated, so that it can re-fire
if something happens").

The engine is Rust (`analyst_core.FramePool`, `rust/analyst_core/src/waiting.rs`): every frame (a seed move, its focus
and goal) is ACTIVE (its certificate holds), DORMANT (one change to the board inside the frame would make it hold: the
conditions it waits on) or FAR; an active frame carries its BREAKERS (the changes their next move can make that break
it). The pool recomputes every frame every move (reuse by footprint audited 1.0% stale, off; --lazy turns it on).

This replays games through the pool and reports:
  ANTICIPATION   a frame turning active for the side to move: was it already waiting (dormant) for that side, already a
                 threat (active on the null move), seen as far, or new -- and how many plies ahead it was first seen
  CONDITIONS MET a dormant frame one of whose conditions came true on the board: did it then hold (or was the condition
                 not enough: a false alarm)
  BASE RATES     of the dormant frames, the share active within 2 / 4 / 8 plies, by family (the counting table the
                 policy over conditions starts from)
  STALENESS      with --lazy --audit, the same games recomputed in full: how often a reused frame's status was wrong
Games run in parallel processes (one Rust thread each); every finished game is checkpointed and a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import chess

from catspace.io import paths


def _atom_holds(b: chess.Board, owner: chess.Color, atom) -> bool:
    kind, role, sq = atom[0], atom[1], atom[2]
    pc = b.piece_at(chess.parse_square(sq))
    pt = chess.Piece.from_symbol(role).piece_type
    ours = pc is not None and pc.color == owner and pc.piece_type == pt
    theirs = pc is not None and pc.color != owner and pc.piece_type == pt
    return {"add ours": ours, "add theirs": theirs, "remove ours": not ours, "remove theirs": not theirs}[kind]


def _replay(args) -> dict:
    gid, fens, exact = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    pool = analyst_core.FramePool(exact)
    hist = []                                                          # per ply: {key: (status, alts, born)}
    t0 = time.time()
    for fen in fens:
        pool.observe(fen)
        snap = {}
        for owner, uci, name, focus, goal, val, st, alts, brk, born in pool.frames():
            snap[f"{owner}|{uci}|{','.join(focus)}"] = (st, [a[:3] for a in alts], born, name, val)
        hist.append(snap)
    return dict(game=gid, fens=fens, hist=hist, secs=time.time() - t0, reused=pool.reused, recomputed=pool.recomputed, exact=exact)


def _games(tag: str, limit: int) -> list:
    from .tactic_census_measure import load
    by = defaultdict(dict)
    for r in load(tag):
        by[r["game"]][r["ply"]] = r["fen"]
    out = []
    for g in sorted(by)[:limit]:
        plies = by[g]
        out.append((g, [plies[t] for t in sorted(plies)]))
    return out


def run(tag: str, limit: int, audit: bool, workers: int, lazy: bool = False) -> list:
    from multiprocessing import Pool
    ck = paths.experiment(f"waiting_frames_{tag}_{limit}{'_lazy' if lazy else ''}{'_audit' if audit else ''}.jsonl")
    done = {}
    if os.path.exists(ck):
        with open(ck) as fh:
            for line in fh:
                r = json.loads(line); done[(r["game"], r["exact"])] = r
    modes = (False, True) if (lazy and audit) else ((False,) if lazy else (True,))
    todo = [(g, f, e) for g, f in _games(tag, limit) for e in modes if (g, e) not in done]
    print(f"{len(done)} game runs checkpointed, {len(todo)} to go ({workers} processes)", flush=True)
    if todo:
        with Pool(workers) as p, open(ck, "a") as fh:
            for i, r in enumerate(p.imap_unordered(_replay, todo, chunksize=1)):
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done[(r["game"], r["exact"])] = r
                print(f"  game {r['game']} exact={r['exact']}: {len(r['fens'])} plies in {r['secs']:.0f} s "
                      f"({1000 * r['secs'] / max(len(r['fens']), 1):.0f} ms a ply) [{i + 1}/{len(todo)}]", flush=True)
    return list(done.values())


def report(runs: list) -> None:
    lazy = [r for r in runs if not r["exact"]] or runs
    plies = sum(len(r["fens"]) for r in lazy); secs = sum(r["secs"] for r in lazy)
    reused = sum(r["reused"] for r in lazy); recomputed = sum(r["recomputed"] for r in lazy)
    print(f"OBS {len(lazy)} games, {plies} positions: {1000 * secs / max(plies, 1):.0f} ms a position on one core; frames reused "
          f"{reused}, recomputed {recomputed} ({100 * reused / max(reused + recomputed, 1):.0f}% reused)")
    origin = Counter(); lead = defaultdict(list); met = Counter(); met_ex = []; base = defaultdict(lambda: Counter())
    for r in lazy:
        hist, fens = r["hist"], r["fens"]
        for t in range(len(hist)):
            b = chess.Board(fens[t]); stm = "w" if b.turn else "b"
            for key, (st, alts, born, name, val) in hist[t].items():
                owner = key[0]
                # ANTICIPATION: newly active for the side to move
                if st == "active" and owner == stm:
                    prev = hist[t - 1].get(key) if t > 0 else None
                    prev2 = hist[t - 2].get(key) if t > 1 else None
                    if prev2 is not None and prev2[0] == "active":
                        continue                                      # it was active for this side last turn too
                    o = "created by their last move (walk-in)" if prev is None and prev2 is None else (prev or prev2)[0]
                    if prev is not None and prev[0] == "active":
                        o = "a threat already (active on the null move)"
                    elif prev is not None and prev[0] == "dormant":
                        o = "waiting (dormant), their move met the condition"
                    origin[(name if name in ("check",) else "piece", o)] += 1
                    first = min((s for s in range(t) if key in hist[s] and hist[s][key][0] == "dormant"), default=None)
                    if first is not None:
                        lead[name].append(t - first)
                # CONDITIONS MET and BASE RATES: a dormant frame at t
                if st == "dormant":
                    nxt = hist[t + 1].get(key) if t + 1 < len(hist) else None
                    if t + 1 < len(hist):
                        b1 = chess.Board(fens[t + 1]); ob = chess.WHITE if owner == "w" else chess.BLACK
                        if any(_atom_holds(b1, ob, a) for a in alts):
                            res = nxt[0] if nxt else "gone"
                            met[res] += 1
                            if res != "active" and len(met_ex) < 6:
                                met_ex.append((fens[t + 1], key, res))
                    for k in (2, 4, 8):
                        if t + k <= len(hist) - 1:
                            hit = any(hist[s].get(key, ("",))[0] == "active" for s in range(t + 1, t + k + 1))
                            base[(name, k)][hit] += 1
    tot = sum(origin.values())
    print(f"OBS ANTICIPATION: {tot} frames turned active for the side to move; before that they were:")
    for (kind, o), n in sorted(origin.items(), key=lambda x: -x[1]):
        print(f"OBS   {kind:6s} {o:45s} {n:5d} ({100 * n / max(tot, 1):.0f}%)")
    for name, ls in sorted(lead.items()):
        ls.sort()
        print(f"OBS   waited first seen dormant, {name}: {len(ls)} frames, lead in plies median {ls[len(ls) // 2]}, max {ls[-1]}")
    mt = sum(met.values())
    print(f"OBS CONDITIONS MET: a dormant frame's condition came true {mt} times; next status {dict(met)} "
          f"(held {100 * met.get('active', 0) / max(mt, 1):.0f}%)")
    for fen, key, res in met_ex:
        print(f"OBS   not enough: {key} -> {res}  {fen}")
    print("OBS BASE RATES: dormant frames active within k plies (the counting table for the policy)")
    for name in sorted({n for n, _ in base}):
        row = "  ".join(f"k={k}: {base[(name, k)][True]}/{sum(base[(name, k)].values())} "
                        f"({100 * base[(name, k)][True] / max(sum(base[(name, k)].values()), 1):.0f}%)" for k in (2, 4, 8))
        print(f"OBS   {name:15s} {row}")
    ex = {r["game"]: r for r in runs if r["exact"]}
    if ex and any(not r["exact"] for r in runs):
        n = stale = 0; kinds = Counter()
        for r in lazy:
            e = ex.get(r["game"])
            if e is None:
                continue
            for a, b in zip(r["hist"], e["hist"]):
                for key, v in b.items():
                    n += 1
                    got = a.get(key, ("missing",))[0]
                    if got != v[0]:
                        stale += 1; kinds[(got, v[0])] += 1
        print(f"OBS STALENESS: {stale} of {n} frame statuses differ between reuse and full recomputation ({100 * stale / max(n, 1):.1f}%): "
              f"{dict(kinds.most_common(6))}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--lazy", action="store_true", help="reuse frames whose footprint no move touched")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t = time.time()
    runs = run(a.tag, a.games, a.audit, a.workers, a.lazy)
    report(runs)
    print(f"({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
