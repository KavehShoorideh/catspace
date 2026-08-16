#!/usr/bin/env python
"""gen_tb_entry_labels.py -- WHICH TABLEBASE DO WE FALL INTO? (Kaveh 2026-08-16)

"record the point at which each game in the corpus reaches the five piece level, and
categorize which category of five piece it is... I wanna know where do we enter the tablebase
region, and that is the label. And then the concepts should predict which tablebase entrance
we're ending [in], and from there the game should be guided by the tablebase. So it's
deterministic... the labels should be the five piece tablebase entrance type."

WHY THIS IS A BETTER LABEL THAN THE FINAL RESULT. Final W/D/L is three classes and, on the
balanced pool, 78.6% of them are DRAW -- a label that is nearly constant carries nearly no
information, which is exactly what the committor kept failing to learn from. The 5-piece
entrance type is ~145 classes (the Syzygy 5-man table count), and it is the LAST point at which
anything is still uncertain: once the position is in the tablebase the rest of the game is
solved, so "which tablebase do I fall into" is the complete remaining question. It is also the
form a human plan actually takes -- "trade into a rook endgame" is a statement about which
entrance you are steering toward.

PIECE COUNT FALLS BY EXACTLY ONE PER CAPTURE and promotion does not change it, so every game
that goes below 6 men passes through 5 exactly once. The entry point is unambiguous: the FIRST
ply at which exactly 5 men remain.

Emits, per position row:
    tb_class   int16   index into the entrance-type vocabulary, or -1 for NEVER-ENTERS
    tb_plies   int32   plies from this row until entry, or -1
and the vocabulary itself, so the label is legible ("KRP|KR" = white rook+pawn vs black rook).

    .venv/bin/python -m ...gen_tb_entry_labels --games 4000 --n-piecedown 45906
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

# token vocabulary from jepa.tokenize: 0 empty, 1-6 white P N B R Q K, 7-12 black
_LETTER = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
_ORDER = "KQRBNP"          # conventional heavy-to-light, king first


def signature(tok_row):
    """(64,) uint8 -> 'KRP|KR', white side left of the bar. Colour-explicit ON PURPOSE: KRPvKR
    is a different proposition depending on who holds the pawn, and the committor needs to
    know which side is winning, not merely which table to open."""
    w = Counter(); b = Counter()
    for t in tok_row:
        if t == 0:
            continue
        if t <= 6:
            w[_LETTER[int(t)]] += 1
        else:
            b[_LETTER[int(t) - 6]] += 1
    def fmt(c):
        return "".join(p * c[p] for p in _ORDER if c[p])
    return f"{fmt(w)}|{fmt(b)}"


def canonical(sig):
    """colour-folded form, i.e. the Syzygy TABLE this maps to ('KRP|KR' and 'KR|KRP' are one
    table). Reported alongside so we can quote both counts honestly."""
    a, b = sig.split("|")
    return f"{a}v{b}" if a >= b else f"{b}v{a}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--men", type=int, default=5, help="tablebase level to detect entry into")
    ap.add_argument("--min-count", type=int, default=50,
                    help="entrance types rarer than this are folded into an OTHER class, since "
                         "a class with a handful of examples cannot be learned or evaluated")
    ap.add_argument("--save", action="store_true", help="write the npz (default: measure only)")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    tok, start, length = tr.tok, tr.start, tr.length
    n_pos, n_games = len(tok), len(start)
    print(f"[tb] {n_games:,} games | {n_pos:,} positions", flush=True)

    # men per position: a single vectorised pass over the whole corpus
    men = (tok != 0).sum(1).astype(np.int16)
    print(f"[tb] men per position: min {men.min()} max {men.max()} "
          f"mean {men.mean():.1f}", flush=True)

    tb_class = np.full(n_pos, -1, np.int16)
    tb_plies = np.full(n_pos, -1, np.int32)
    sigs, entry_row, entry_game = [], [], []
    for g in range(n_games):
        s, n = int(start[g]), int(length[g])
        m = men[s:s + n]
        hit = np.flatnonzero(m == args.men)
        if len(hit) == 0:
            continue                       # never enters -- stays -1 (censored)
        i = int(hit[0])                    # FIRST time at exactly `men` men
        sigs.append(signature(tok[s + i]))
        entry_row.append(s + i); entry_game.append(g)
        tb_plies[s:s + i + 1] = np.arange(i, -1, -1, dtype=np.int32)

    n_enter = len(sigs)
    print(f"[tb] games reaching {args.men} men: {n_enter:,} / {n_games:,} "
          f"({100 * n_enter / max(n_games, 1):.1f}%)", flush=True)
    if not n_enter:
        return

    cnt = Counter(sigs)
    tables = Counter(canonical(s) for s in sigs)
    print(f"[tb] DISTINCT entrance types: {len(cnt):,} colour-explicit | "
          f"{len(tables):,} colour-folded (= Syzygy tables touched)", flush=True)

    keep = {s for s, c in cnt.items() if c >= args.min_count}
    vocab = sorted(keep) + ["OTHER"]
    idx = {s: i for i, s in enumerate(vocab)}
    print(f"[tb] vocabulary: {len(keep):,} types with >={args.min_count} examples, "
          f"+1 OTHER (covers {sum(c for s, c in cnt.items() if s not in keep):,} entries)",
          flush=True)

    for r, sg in zip(entry_row, sigs):
        pass
    # assign the class to every row of the game that leads to that entry
    for g, r, sg in zip(entry_game, entry_row, sigs):
        s = int(start[g])
        tb_class[s:r + 1] = idx.get(sg, idx["OTHER"])

    lab = int((tb_class >= 0).sum())
    print(f"[tb] labelled positions: {lab:,} / {n_pos:,} ({100 * lab / n_pos:.1f}%)", flush=True)
    print(f"\n[tb] top 15 entrance types (colour-explicit, white|black):")
    for s, c in cnt.most_common(15):
        print(f"     {s:<14} {c:>7,}  ({100 * c / n_enter:4.1f}%)")
    pl = tb_plies[tb_plies >= 0]
    if len(pl):
        print(f"\n[tb] plies-to-entry over labelled rows: median {np.median(pl):.0f} "
              f"p90 {np.percentile(pl, 90):.0f} max {pl.max()}")

    if args.save:
        out = paths.derived(f"tb_entry_{args.men}men_{n_pos}.npz")
        np.savez_compressed(out, tb_class=tb_class, tb_plies=tb_plies,
                            vocab=np.array(vocab), men=men)
        print(f"\n[tb] wrote {out}", flush=True)
        with open(str(out).replace(".npz", "_vocab.json"), "w") as f:
            json.dump({"vocab": vocab, "counts": dict(cnt.most_common())}, f, indent=1)


if __name__ == "__main__":
    main()
