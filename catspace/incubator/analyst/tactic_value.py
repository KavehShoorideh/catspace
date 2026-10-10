"""WHICH TACTICS PAY, AND WHICH WAITING FRAMES ARE WORTH WATCHING (Kaveh 2026-10-09: "we need to know what tactics lead to
victory and what positions lead to what tactics, so that we can say: pins usually are good, this other tactic isn't so
good -- so we have a sense of what to look for"; "we don't want hypotheses that 98% don't work -- focus on the important
stuff").

From the frames replayed over games (waiting_frames.py), Stockfish's labels (tactic_reach.py: the best move's score and
line) and the game results:
  PAYOFF       a frame ACTIVE for the side to move, by family: how often Stockfish has the mover up 1.5 or more, how
               often the position carries a labelled tactic, and how often the mover went on to win the game -- against
               the same rates over all positions (the base)
  REACH        a DORMANT frame, by family and by the kind of condition it waits on: how often it turns active for its
               owner within 2 / 4 / 8 plies, and how often that activation is confirmed by Stockfish
  PRIORITY     the classes ordered by reach x confirmation: if condition searches ran only for the top classes, how many
               of the confirmed activations they cover and how many frames (the cost of the search) they take
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import chess

from catspace.io import paths


def _load(frames_file: str, tag: str):
    from .tactic_census_measure import load
    from .tactic_reach import is_tactic
    census = load(tag)
    plies = defaultdict(list)
    meta = {}
    for r in census:
        plies[r["game"]].append(r["ply"])
        meta[(r["game"], r["ply"])] = r
    labels = {(r["game"], r["ply"]): r for r in map(json.loads, open(paths.experiment(f"tactic_labels_{tag}_d12_pv.jsonl")))}
    runs = [json.loads(l) for l in open(paths.experiment(frames_file))]
    runs = [r for r in runs if r.get("proofs", True)]
    out = []
    for r in runs:
        ps = sorted(plies[r["game"]])
        if len(ps) != len(r["fens"]):
            continue
        rows = []
        for i, fen in enumerate(r["fens"]):
            lab = labels.get((r["game"], ps[i])); m = meta[(r["game"], ps[i])]
            stm = "w" if chess.Board(fen).turn else "b"
            good = lab is not None and lab["best"] is not None and lab["best"] >= 150
            tac = lab is not None and i > 0 and is_tactic(lab, r["fens"][i - 1], fen)
            won = m["result"] == 1                                       # the census result: from the side to move's view
            rows.append(dict(stm=stm, good=good, tactic=tac, won=won, hist=r["hist"][i]))
        out.append(rows)
    return out


def report(games) -> None:
    base = Counter(); pay = defaultdict(Counter)
    for rows in games:
        for row in rows:
            base["n"] += 1; base["good"] += row["good"]; base["tactic"] += row["tactic"]; base["won"] += row["won"]
            fams = {v[3] for k, v in row["hist"].items() if k[0] == row["stm"] and v[0] == "active"}
            for f in fams:
                pay[f]["n"] += 1; pay[f]["good"] += row["good"]; pay[f]["tactic"] += row["tactic"]; pay[f]["won"] += row["won"]
            if not fams:
                pay["(no active frame)"]["n"] += 1; pay["(no active frame)"]["good"] += row["good"]
                pay["(no active frame)"]["tactic"] += row["tactic"]; pay["(no active frame)"]["won"] += row["won"]
    pc = lambda c, k: f"{100 * c[k] / max(c['n'], 1):5.1f}%"
    print(f"OBS PAYOFF -- positions where the side to move has an ACTIVE frame of the family (a position may count in several)")
    print(f"OBS   {'family':20s} {'positions':>9s} {'SF >= +1.5':>11s} {'tactic label':>13s} {'won the game':>13s}")
    print(f"OBS   {'(all positions)':20s} {base['n']:9d} {pc(base, 'good'):>11s} {pc(base, 'tactic'):>13s} {pc(base, 'won'):>13s}")
    for f, c in sorted(pay.items(), key=lambda x: -x[1]["n"]):
        print(f"OBS   {f:20s} {c['n']:9d} {pc(c, 'good'):>11s} {pc(c, 'tactic'):>13s} {pc(c, 'won'):>13s}")

    reach = defaultdict(Counter)
    for rows in games:
        for t, row in enumerate(rows):
            for key, v in row["hist"].items():
                st, alts, born, name = v[0], v[1], v[2], v[3]
                if st not in ("dormant", "far"):
                    continue
                owner = key[0]
                kinds = sorted({a[0] for a in alts}) if st == "dormant" else ["(far: no condition)"]
                for kind in kinds:
                    cls = (name, st, kind)
                    reach[cls]["n"] += 1
                    for k in (2, 4, 8):
                        hit = None
                        for s in range(t + 1, min(t + k + 1, len(rows))):
                            w = rows[s]["hist"].get(key)
                            if w is not None and w[0] == "active" and rows[s]["stm"] == owner:
                                hit = s; break
                        if hit is not None:
                            reach[cls][f"a{k}"] += 1
                            reach[cls][f"g{k}"] += rows[hit]["good"]
    print(f"OBS REACH -- waiting frames by family, status and the kind of condition: active for their owner within k plies "
          f"(and of those, Stockfish-confirmed: the owner up 1.5 or more)")
    print(f"OBS   {'family':16s} {'status':8s} {'condition':22s} {'frames':>7s} {'k=2':>14s} {'k=4':>14s} {'k=8':>14s}")
    for cls, c in sorted(reach.items(), key=lambda x: -x[1]["n"]):
        if c["n"] < 30:
            continue
        cells = "".join(f"{100 * c[f'a{k}'] / c['n']:6.1f}% ({100 * c[f'g{k}'] / max(c[f'a{k}'], 1):3.0f}%)" for k in (2, 4, 8))
        print(f"OBS   {cls[0]:16s} {cls[1]:8s} {cls[2]:22s} {c['n']:7d} {cells}")
    ranked = sorted(((c["g4"] / c["n"], cls, c) for cls, c in reach.items() if c["n"] >= 30), key=lambda x: -x[0])
    tot_g = sum(c["g4"] for _, _, c in ranked); tot_n = sum(c["n"] for _, _, c in ranked)
    print(f"OBS PRIORITY -- classes by confirmed activations within 4 plies per frame; searching only the top classes:")
    cg = cn = 0
    for v, cls, c in ranked:
        cg += c["g4"]; cn += c["n"]
        print(f"OBS   {cls[0]:16s} {cls[1]:8s} {cls[2]:22s} {100 * v:5.2f}% per frame -> cumulative: {100 * cg / max(tot_g, 1):4.0f}% of the "
              f"confirmed activations for {100 * cn / max(tot_n, 1):4.0f}% of the frames")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--frames", default="waiting_frames_v2_c300_300.jsonl")
    a = ap.parse_args(argv)
    report(_load(a.frames, a.tag))


if __name__ == "__main__":
    main()
