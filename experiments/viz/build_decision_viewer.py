#!/usr/bin/env python
"""
experiments/viz/build_decision_viewer.py — plays FBBoardPolicy vs an opponent
(random or Stockfish) exactly like experiments/arena_real.py, but records
per-candidate scores + feared replies from FBBoardPolicy.move_scored() at
every FB ply, for the interactive decision_viewer.html template. Boards are
NOT pre-rendered to SVG here -- only FEN + the two last-move square names are
stored (a few dozen bytes vs. ~31KB per chess.svg.board() call); the template
renders whichever position is on screen client-side, on demand.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import numpy as np
import torch

from catspace.data.encode import encode_meta, encode_packed
from catspace.io.paths import derived_dir, generated_dir
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.policy_fb import FBBoardPolicy
from catspace.realboard import RandomBoardPolicy
from catspace.viz.build_html import build_html


def make_opponent(spec: str):
    if spec == "random":
        return RandomBoardPolicy(), "random"
    if spec.startswith("sf:"):
        from catspace.uci import UCIBoardPolicy
        arg = spec[3:]
        if arg.startswith("skill="):
            return UCIBoardPolicy(skill=int(arg[6:]), movetime=0.02), spec
        return UCIBoardPolicy(elo=int(arg), movetime=0.05), spec
    raise SystemExit(f"unknown opponent spec {spec!r} (use random | sf:<elo> | sf:skill=<k>)")


def reach_after(fb, board, z_unit, device, elo_cond: int):
    packed = encode_packed(board)[None]
    meta = encode_meta(board)[None]
    om = omega_ids(np.array([elo_cond]), np.array([elo_cond]), np.array([300.0]))
    planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
    om_t = torch.from_numpy(om).to(device)
    with torch.no_grad():
        f = fb.embed_F(planes, om_t)
    return float((f @ z_unit).item())


def sq(square: int) -> str:
    return chess.square_name(square)


def play_game(fb, z_white_unit, z_black_unit, fb_policy, opponent, fb_is_white,
             opening_plies, max_plies, rng, device, elo_cond, cand_cap=8):
    board = chess.Board()
    plies = []
    rand = RandomBoardPolicy()

    def record(mover, san, move, cands=None):
        z_unit = z_white_unit if fb_is_white else z_black_unit
        r = reach_after(fb, board, z_unit, device, elo_cond)
        entry = dict(ply=len(plies), mover=mover, san=san,
                    fen=board.fen(), last_from=sq(move.from_square) if move else None,
                    last_to=sq(move.to_square) if move else None,
                    reach_after=round(r, 4), cands=None)
        if cands is not None:
            trimmed = cands[:cand_cap]
            if not any(c["chosen"] for c in trimmed):
                trimmed = trimmed + [c for c in cands if c["chosen"]]
            entry["cands"] = trimmed
        plies.append(entry)
        return entry

    for _ in range(opening_plies):
        if board.is_game_over(claim_draw=True):
            break
        m = rand.move(board, rng)
        san = board.san(m)
        board.push(m)
        record("open", san, m)

    result, termination = "*", "PLY_CAP"
    while len(plies) < max_plies:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result, termination = outcome.result(), outcome.termination.name
            break
        is_fb_turn = (board.turn == chess.WHITE) == fb_is_white
        if is_fb_turn:
            move, cands = fb_policy.move_scored(board, rng)
            # feared_fen needs the position BEFORE this move for each candidate;
            # cheap now (just FEN strings), so compute it for every candidate.
            pre_board = board.copy(stack=False)
            for c in cands:
                if "feared_uci" not in c:
                    continue
                child = pre_board.copy(stack=False)
                child.push(chess.Move.from_uci(c["uci"]))
                feared_move = chess.Move.from_uci(c["feared_uci"])
                child.push(feared_move)
                c["feared_fen"] = child.fen()
                c["feared_from"] = sq(feared_move.from_square)
                c["feared_to"] = sq(feared_move.to_square)
            san = pre_board.san(move)
            board.push(move)
            record("fb", san, move, cands)
        else:
            move = opponent.move(board, rng)
            san = board.san(move)
            board.push(move)
            record("opp", san, move)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result, termination = outcome.result(), outcome.termination.name
    return dict(plies=plies, result=result, termination=termination, fb_is_white=fb_is_white)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponent", default="random")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--depth", type=int, default=2, choices=(1, 2))
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=200,
                    help="boards render client-side from FEN now, so this is cheap "
                         "to raise (matches arena_real.py's default)")
    ap.add_argument("--elo-cond", type=int, default=1800)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    t0 = time.time()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    fb, payload = load_ckpt(ckpt_path, device)
    fb.eval()
    if "MATE_W" not in payload.get("zgoals", {}):
        raise SystemExit("checkpoint has no zgoals -- finish a train_lichess_fb.py run first")
    zw = payload["zgoals"]["MATE_W"].numpy().astype(np.float32)
    zb = payload["zgoals"]["MATE_B"].numpy().astype(np.float32)
    z_white_unit = torch.from_numpy(zw / np.linalg.norm(zw)).to(device)
    z_black_unit = torch.from_numpy(zb / np.linalg.norm(zb)).to(device)
    fb_white = FBBoardPolicy(fb, payload["zgoals"]["MATE_W"], depth=args.depth,
                             elo=args.elo_cond, device=device)
    fb_black = FBBoardPolicy(fb, payload["zgoals"]["MATE_B"], depth=args.depth,
                             elo=args.elo_cond, device=device)
    opponent, opp_name = make_opponent(args.opponent)
    print(f"load: {time.time() - t0:.1f}s  step={payload.get('step', '?')} "
          f"opponent={opp_name} depth={args.depth} device={device}")

    t0 = time.time()
    games = []

    def run_games():
        for i in range(args.games):
            rng = np.random.default_rng([args.seed, i])
            fb_is_white = i % 2 == 0
            fb_policy = fb_white if fb_is_white else fb_black
            g = play_game(fb, z_white_unit, z_black_unit, fb_policy, opponent, fb_is_white,
                         args.opening_plies, args.max_plies, rng, device, args.elo_cond)
            name = f"FB as {'W' if fb_is_white else 'B'} vs {opp_name} #{i}"
            games.append(dict(name=name, result=g["result"], plies=g["plies"]))
            print(f"  game {i}: {name}  result={g['result']}  plies={len(g['plies'])}  "
                  f"({time.time() - t0:.1f}s elapsed)")

    if args.opponent.startswith("sf:"):
        with opponent:
            run_games()
    else:
        run_games()
    print(f"{len(games)} games played+recorded: {time.time() - t0:.1f}s")

    data = dict(meta=dict(title=f"catspace — decision viewer  ·  FB depth={args.depth} "
                          f"vs {opp_name}  ·  ckpt step {payload.get('step', '?')}"),
               games=games)
    out = Path(args.out) if args.out else generated_dir() / "decision-viewer.html"
    template = Path(__file__).resolve().parents[2] / "catspace" / "viz" / "templates" / "decision_viewer.html"
    build_html(template, data, out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
