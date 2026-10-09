"""WAITING FRAMES, measured on real games (Kaveh 2026-10-09: "keep the frames and the facts in the frames ... a live
index of who's waiting on what condition"; "each frame also needs to know what is invalidated, so that it can re-fire
if something happens").

The engine is Rust (`analyst_core.FramePool`, `rust/analyst_core/src/waiting.rs`): every frame (a seed move, its focus
and goal) is ACTIVE (its certificate holds), DORMANT (one change to the board inside the frame would make it hold: the
conditions it waits on) or FAR; an active frame carries its BREAKERS (the changes their next move can make that break
it). The pool keeps every frame's stored proofs: after a move, the frames it created are searched, the ones it
destroyed dropped, every other one RE-CHECKED from its stored proofs (the same answers as a fresh search, cheaper).

This replays games through the pool and reports:
  ANTICIPATION   a frame turning active for the side to move: was it already waiting (dormant) for that side, already a
                 threat (active on the null move), seen as far, or new -- and how many plies ahead it was first seen
  CONDITIONS MET a dormant frame one of whose conditions came true on the board: did it then hold (or was the condition
                 not enough: a false alarm)
  BASE RATES     of the dormant frames, the share active within 2 / 4 / 8 plies, by family (the counting table the
                 policy over conditions starts from)
  AUDIT          with --audit, the same games from scratch: every status, condition and breaker compared, and the cost
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
    gid, fens, proofs = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    pool = analyst_core.FramePool(proofs)
    hist = []                                                          # per ply: {key: (status, alts, born)}
    t0 = time.time()
    for fen in fens:
        pool.observe(fen)
        snap = {}
        for owner, uci, name, focus, goal, val, st, alts, brk, born in pool.frames():
            snap[f"{owner}|{uci}|{','.join(focus)}"] = (st, [a[:3] for a in alts], born, name, val, [a[:3] for a in brk])
        hist.append(snap)
    return dict(game=gid, fens=fens, hist=hist, secs=time.time() - t0, rechecked=pool.rechecked, created=pool.created,
                nodes=pool.nodes, proofs=proofs, exact=True)


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


def run(tag: str, limit: int, audit: bool, workers: int) -> list:
    from multiprocessing import Pool
    ck = paths.experiment(f"waiting_frames_v2_{tag}_{limit}{'_audit' if audit else ''}.jsonl")
    done = {}
    if os.path.exists(ck):
        with open(ck) as fh:
            for line in fh:
                r = json.loads(line); done[(r["game"], r["proofs"])] = r
    modes = (True, False) if audit else (True,)
    todo = [(g, f, e) for g, f in _games(tag, limit) for e in modes if (g, e) not in done]
    print(f"{len(done)} game runs checkpointed, {len(todo)} to go ({workers} processes)", flush=True)
    if todo:
        with Pool(workers) as p, open(ck, "a") as fh:
            for i, r in enumerate(p.imap_unordered(_replay, todo, chunksize=1)):
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done[(r["game"], r["proofs"])] = r
                print(f"  game {r['game']} proofs={r['proofs']}: {len(r['fens'])} plies in {r['secs']:.0f} s "
                      f"({1000 * r['secs'] / max(len(r['fens']), 1):.0f} ms a ply) [{i + 1}/{len(todo)}]", flush=True)
    return list(done.values())


def report(runs: list) -> None:
    lazy = [r for r in runs if r["proofs"]]
    plies = sum(len(r["fens"]) for r in lazy); secs = sum(r["secs"] for r in lazy)
    rc = sum(r["rechecked"] for r in lazy); cr = sum(r["created"] for r in lazy); nodes = sum(r["nodes"] for r in lazy)
    print(f"OBS {len(lazy)} games, {plies} positions: {1000 * secs / max(plies, 1):.0f} ms a position on one core, "
          f"{nodes / max(plies, 1):,.0f} nodes; frames re-checked from stored proofs {rc}, created {cr}")
    origin = Counter(); lead = defaultdict(list); met = Counter(); met_ex = []; base = defaultdict(lambda: Counter())
    for r in lazy:
        hist, fens = r["hist"], r["fens"]
        for t in range(len(hist)):
            b = chess.Board(fens[t]); stm = "w" if b.turn else "b"
            for key, (st, alts, born, name, val, *_) in hist[t].items():
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
    scratch = {r["game"]: r for r in runs if not r["proofs"]}
    if scratch:
        n = st_diff = alt_diff = brk_diff = missing = 0; kinds = Counter(); ex = []
        s_secs = s_nodes = s_plies = 0
        for r in lazy:
            e = scratch.get(r["game"])
            if e is None:
                continue
            s_secs += e["secs"]; s_nodes += e["nodes"]; s_plies += len(e["fens"])
            for t, (a, b) in enumerate(zip(r["hist"], e["hist"])):
                for key, v in b.items():
                    n += 1
                    w = a.get(key)
                    if w is None:
                        missing += 1; continue
                    if w[0] != v[0]:
                        st_diff += 1; kinds[(w[0], v[0])] += 1
                        if len(ex) < 5:
                            ex.append((r["fens"][t], key, w[0], v[0]))
                    elif set(map(tuple, w[1])) != set(map(tuple, v[1])):
                        alt_diff += 1
                    elif set(map(tuple, w[5])) != set(map(tuple, v[5])):
                        brk_diff += 1
        print(f"OBS AUDIT against from-scratch on the same games: {n} frame records; status differs {st_diff} "
              f"{dict(kinds)}, conditions differ {alt_diff}, breakers differ {brk_diff}, missing {missing}")
        for fen, key, got, want in ex:
            print(f"OBS   differs: {key} stored proofs {got}, scratch {want}  {fen}")
        print(f"OBS COST from scratch: {1000 * s_secs / max(s_plies, 1):.0f} ms a position, {s_nodes / max(s_plies, 1):,.0f} nodes "
              f"-- stored proofs: {1000 * secs / max(plies, 1):.0f} ms, {nodes / max(plies, 1):,.0f} nodes")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t = time.time()
    runs = run(a.tag, a.games, a.audit, a.workers)
    report(runs)
    print(f"({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
