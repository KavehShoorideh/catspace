#!/usr/bin/env python
"""kittychess_arena.py -- promotion gate for the playable server (Kaveh 2026-08-08): the candidate
must BEAT the reigning champion head-to-head before it ships.

Paired-opening match: each round samples one short random opening prefix and plays it TWICE with
colors swapped (both engines are deterministic; unpaired games would repeat one game forever).
Candidate promotes iff its total score exceeds half the games. The champion registry
(artifacts/champions/) records the lineage: who beat whom, when, by how much.

    .venv/bin/python -m ...kittychess_arena --candidate <ckpt> [--rounds 10] [--promote]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import chess

from catspace.io import paths
from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess

CHAMP_DIR = os.path.join(paths.experiment(""), "champions")
CHAMP_FILE = os.path.join(CHAMP_DIR, "champion.json")


def load_champion():
    if os.path.exists(CHAMP_FILE):
        return json.load(open(CHAMP_FILE))
    return None


def play_one(white_eng, black_eng, opening, max_plies=300, search_depth=0):
    b = chess.Board()
    for u in opening:
        b.push_uci(u)
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        eng = white_eng if b.turn == chess.WHITE else black_eng
        if search_depth > 0:
            rows = eng.search(b, depth=search_depth)
            mv = rows[0]["mv"] if rows else None
        else:
            mv = eng.choose(b)
        if mv is None:
            break
        b.push(mv)
    out = b.outcome(claim_draw=True)
    if out is None or out.winner is None:
        return 0.5
    return 1.0 if out.winner == chess.WHITE else 0.0


def random_opening(rng, plies=6):
    while True:
        b = chess.Board()
        ucis = []
        ok = True
        for _ in range(plies):
            ms = list(b.legal_moves)
            if not ms:
                ok = False
                break
            m = rng.choice(ms)
            ucis.append(m.uci())
            b.push(m)
        if ok and not b.is_game_over():
            return ucis


def _share_tb(a, b):
    """ONE probe-cache connection per process. Two TB() instances hit sqlite's DELETE-mode
    writer lock and the second spins forever in the busy handler at ~0% CPU (sampled
    2026-08-08: nanosleep under sqliteDefaultBusyCallback -- every 'stalled arena' today).
    The arena is single-threaded, so sharing is safe."""
    if getattr(b, "tb", None) is not None and getattr(a, "tb", None) is not None:
        try:
            b.tb.close()
        except Exception:
            pass
        b.tb = a.tb


def report_pairs(pairs, a_name, b_name):
    """Pentanomial + pair-bootstrap verdict (fishtest-style, 2026-08-08). The game PAIR
    (one opening, colors swapped) is the independent unit -- with deterministic engines the
    pair outcome is a pure function of the opening, so all sampling variance is over openings
    and per-game counting overstates evidence by ~2x."""
    import numpy as np
    pairs = np.array(pairs, float)
    n = len(pairs)
    penta = {k: int((pairs == v).sum()) for k, v in
             [("0", 0.0), ("0.5", 0.5), ("1", 1.0), ("1.5", 1.5), ("2", 2.0)]}
    score = pairs.sum() / (2 * n)
    rng = np.random.default_rng(0)
    boots = np.array([pairs[rng.integers(0, n, n)].mean() / 2 for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_gt = float((boots > 0.5).mean())
    print(f"\n[arena] FINAL: {a_name} {pairs.sum()}/{2*n} ({score:.1%}) vs {b_name}")
    print(f"[arena] pentanomial (pair outcomes 0/0.5/1/1.5/2): {penta}")
    print(f"[arena] pair-bootstrap 95% CI on score: [{lo:.1%}, {hi:.1%}]  "
          f"P(score>50%) = {p_gt:.2f}  ({n} independent opening pairs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--rounds", type=int, default=10, help="paired openings -> 2x games")
    ap.add_argument("--cond-elo", type=float, default=None)
    ap.add_argument("--promote", action="store_true",
                    help="on a win, update the champion registry (page rebuild is separate)")
    ap.add_argument("--candidate-nav", default="db", choices=("db", "ab", "cascade"),
                    help="candidate navigation: db threat-first | ab A-steer+B-gate | cascade")
    ap.add_argument("--champion-nav", default="db", choices=("db", "ab", "cascade"))
    ap.add_argument("--champion-ckpt", default=None,
                    help="override the registry champion (e.g. same ckpt, other nav mode -- "
                         "Kaveh 2026-08-08 'try both ways, arena them')")
    ap.add_argument("--search-depth", type=int, default=0,
                    help=">0: engines play by SEARCH at this depth (the real play config; "
                         "the 1-ply chooser ignores the move-head entirely)")
    ap.add_argument("--candidate-head", action="store_true",
                    help="candidate uses the move-head root prior + internal ordering")
    ap.add_argument("--champion-head", action="store_true")
    ap.add_argument("--opening-seed", type=int, default=0,
                    help="seed for the random opening pairs (chunked matches: different seed per chunk = fresh independent pairs)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    os.makedirs(CHAMP_DIR, exist_ok=True)
    if args.champion_ckpt:
        print(f"[arena] HEAD-TO-HEAD: {args.candidate} ({args.candidate_nav}) vs "
              f"{args.champion_ckpt} ({args.champion_nav}) -- registry untouched")
        cand = KittyChess(args.candidate, args.device, args.cond_elo, nav=args.candidate_nav,
                          head_order=args.candidate_head)
        ch = KittyChess(args.champion_ckpt, args.device, args.cond_elo, nav=args.champion_nav,
                        head_order=args.champion_head)
        _share_tb(cand, ch)
        rng = random.Random(args.opening_seed)
        pairs = []
        for rd in range(args.rounds):
            op = random_opening(rng)
            p = (play_one(cand, ch, op, search_depth=args.search_depth)
                 + (1.0 - play_one(ch, cand, op, search_depth=args.search_depth)))
            pairs.append(p)
            print(f"[arena] round {rd+1}/{args.rounds}: candidate {sum(pairs)}/{2*len(pairs)}",
                  flush=True)
        report_pairs(pairs, f"candidate({args.candidate_nav})", f"champion({args.champion_nav})")
        return
    champ = load_champion()
    if champ is None:
        if args.promote:
            json.dump({"ckpt": args.candidate, "since": time.strftime("%Y-%m-%d %H:%M"),
                       "history": [{"event": "founding champion", "ckpt": args.candidate}]},
                      open(CHAMP_FILE, "w"), indent=1)
            print(f"[arena] no champion existed -- {args.candidate} enthroned by default")
        else:
            print("[arena] no champion registered; rerun with --promote to enthrone")
        return

    print(f"[arena] candidate {args.candidate}\n[arena] champion  {champ['ckpt']}")
    cand = KittyChess(args.candidate, args.device, args.cond_elo, nav=args.candidate_nav)
    ch = KittyChess(champ["ckpt"], args.device, args.cond_elo, nav=args.champion_nav)
    _share_tb(cand, ch)
    rng = random.Random(args.opening_seed)
    pairs = []
    for rd in range(args.rounds):
        op = random_opening(rng)
        p = play_one(cand, ch, op) + (1.0 - play_one(ch, cand, op))   # white then black
        pairs.append(p)
        print(f"[arena] round {rd+1}/{args.rounds}: candidate {sum(pairs)}/{2*len(pairs)}",
              flush=True)
    cscore, n = sum(pairs), 2 * len(pairs)
    report_pairs(pairs, "candidate", "champion")
    win = cscore > n / 2
    print(f"[arena] verdict: {'PROMOTE' if win else 'REJECTED (champion holds)'}")
    if win and args.promote:
        champ["history"].append({"event": "dethroned", "old": champ["ckpt"],
                                 "new": args.candidate, "score": f"{cscore}/{n}",
                                 "date": time.strftime("%Y-%m-%d %H:%M")})
        champ["ckpt"] = args.candidate
        champ["since"] = time.strftime("%Y-%m-%d %H:%M")
        json.dump(champ, open(CHAMP_FILE, "w"), indent=1)
        print(f"[arena] champion registry updated -> {args.candidate}")
        print("[arena] next: export_kitty_web + build_kitty_page + republish the artifact")


if __name__ == "__main__":
    main()
