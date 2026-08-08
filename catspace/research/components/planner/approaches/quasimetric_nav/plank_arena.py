#!/usr/bin/env python
"""plank_arena.py -- promotion gate for the playable server (Kaveh 2026-08-08): the candidate
must BEAT the reigning champion head-to-head before it ships.

Paired-opening match: each round samples one short random opening prefix and plays it TWICE with
colors swapped (both engines are deterministic; unpaired games would repeat one game forever).
Candidate promotes iff its total score exceeds half the games. The champion registry
(artifacts/champions/) records the lineage: who beat whom, when, by how much.

    .venv/bin/python -m ...plank_arena --candidate <ckpt> [--rounds 10] [--promote]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import chess

from catspace.io import paths
from catspace.research.components.planner.approaches.quasimetric_nav.chessplank import ChessPlank

CHAMP_DIR = os.path.join(paths.experiment(""), "champions")
CHAMP_FILE = os.path.join(CHAMP_DIR, "champion.json")


def load_champion():
    if os.path.exists(CHAMP_FILE):
        return json.load(open(CHAMP_FILE))
    return None


def play_one(white_eng, black_eng, opening, max_plies=300):
    b = chess.Board()
    for u in opening:
        b.push_uci(u)
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        eng = white_eng if b.turn == chess.WHITE else black_eng
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--rounds", type=int, default=10, help="paired openings -> 2x games")
    ap.add_argument("--cond-elo", type=float, default=None)
    ap.add_argument("--promote", action="store_true",
                    help="on a win, update the champion registry (page rebuild is separate)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    os.makedirs(CHAMP_DIR, exist_ok=True)
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
    cand = ChessPlank(args.candidate, args.device, args.cond_elo)
    ch = ChessPlank(champ["ckpt"], args.device, args.cond_elo)
    rng = random.Random(0)
    cscore, n = 0.0, 0
    for rd in range(args.rounds):
        op = random_opening(rng)
        cscore += play_one(cand, ch, op)          # candidate white
        cscore += 1.0 - play_one(ch, cand, op)    # candidate black (same opening)
        n += 2
        print(f"[arena] round {rd+1}/{args.rounds}: candidate {cscore}/{n}", flush=True)
    frac = cscore / n
    win = cscore > n / 2
    print(f"\n[arena] FINAL: candidate {cscore}/{n} ({frac:.1%}) -> "
          f"{'PROMOTE' if win else 'REJECTED (champion holds)'}")
    if win and args.promote:
        champ["history"].append({"event": "dethroned", "old": champ["ckpt"],
                                 "new": args.candidate, "score": f"{cscore}/{n}",
                                 "date": time.strftime("%Y-%m-%d %H:%M")})
        champ["ckpt"] = args.candidate
        champ["since"] = time.strftime("%Y-%m-%d %H:%M")
        json.dump(champ, open(CHAMP_FILE, "w"), indent=1)
        print(f"[arena] champion registry updated -> {args.candidate}")
        print("[arena] next: export_plank_web + build_plank_page + republish the artifact")


if __name__ == "__main__":
    main()
