#!/usr/bin/env python
"""
experiments/selfplay_generate.py — generate self-play games with the CURRENT
best FB checkpoint and write them as Lichess-shard-compatible npz files
(same schema as catspace/data/lichess.py::build_shards), so they plug
directly into LichessPairSource / train_lichess_fb.py unchanged.

2026-07-12 motivation (JOURNAL.md, Kaveh's "build all the self-play stuff"):
real self-play is the mechanism the literature (McGrath et al., AlphaZero)
actually credits with organic tactical-concept emergence: NEW games,
generated under the CURRENT policy, so the training distribution keeps
shifting toward what the model itself needs to see to improve -- the
actual PI-refinement step this project's roadmap has flagged since round 4.
(The round-11 --winner-pov-only filter, a cheap proxy for this, was removed
the same day self-play landed -- losing trajectories carry the "bad future"
signal the ply-gap-calibrated quasimetric needs.)

Move diversity: FBSearchPolicy/FBPlanPolicy are deterministic argmax, so
raw self-play would collapse to a handful of repeated games. Two cheap,
standard diversity sources, layered: (1) a few random opening plies
(play_board_game's existing opening_plies), (2) per-move epsilon-random
mixing (StochasticPolicy below) -- simpler than full temperature/Dirichlet
noise (AlphaZero's approach) but the same purpose, and cheap to reason
about/test.

Leakage discipline: this script is intentionally SEPARATE from
train_lichess_fb.py's audited batch_tensors/main (catspace/audit.py's
static_purity_check only re-scans those two functions + the planner's read
path) -- self-play may use UCIBoardPolicy/Stockfish as a SPARRING PARTNER
and records only the PLAYED MOVES and the GAME RESULT (win/loss/draw),
never a Stockfish evaluation score. That's categorically the same as an
existing human game against a strong opponent, not a new leak path -- no
eval_cp is ever attached to self-play shard rows (written as nan, matching
unannotated human games in the existing shard schema).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import numpy as np

from catspace.data.encode import encode_meta, encode_packed
from catspace.realboard import play_board_game
from catspace.io.paths import derived_dir

_RESULT_MAP = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "*": 0}


class StochasticPolicy:
    """Wraps any BoardPolicy with epsilon-random move mixing, for self-play
    diversity -- the underlying policy (FBSearchPolicy etc.) is otherwise
    deterministic argmax, which would collapse repeated self-play games to
    near-duplicates."""

    def __init__(self, inner, epsilon: float):
        self.inner = inner
        self.epsilon = epsilon

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        if rng.random() < self.epsilon:
            legal = list(board.legal_moves)
            return legal[int(rng.integers(len(legal)))]
        return self.inner.move(board, rng)


def make_selfplay_pair(fb, zgoals, device, max_nodes: int, beam: int, epsilon: float,
                       policy_cls) -> tuple:
    """(white_policy, black_policy), both wrapping the SAME fb weights with
    the color-appropriate zgoal, epsilon-random for diversity."""
    from catspace.nn.policy_fb import FBPlanPolicy
    if policy_cls is FBPlanPolicy:
        kwargs = dict(plan_nodes=max_nodes, plan_beam=beam, device=device)
    else:
        kwargs = dict(max_nodes=max_nodes, beam=beam, device=device)
    white = policy_cls(fb, zgoals["MATE_W"], **kwargs)
    black = policy_cls(fb, zgoals["MATE_B"], **kwargs)
    return StochasticPolicy(white, epsilon), StochasticPolicy(black, epsilon)


def positions_of_game(rec) -> list[dict]:
    """Mirrors data.lichess.positions_of's dict shape (packed/meta/ply/
    clock/eval_cp), replaying rec.moves to get one row per ply INCLUDING
    the final position (checkmate finals live there -- needed for zgoal
    rebuilding on self-play-inclusive checkpoints)."""
    board = chess.Board()
    out = [dict(packed=encode_packed(board), meta=encode_meta(board), ply=0,
               clock=float("nan"), eval_cp=float("nan"))]
    for uci in rec.moves:
        board.push(chess.Move.from_uci(uci))
        out.append(dict(packed=encode_packed(board), meta=encode_meta(board), ply=len(out),
                        clock=float("nan"), eval_cp=float("nan")))
    return out


def generate(fb, zgoals, device, n_games: int, out_dir: Path, max_nodes: int, beam: int,
            epsilon: float, opening_plies: int, max_plies: int, elo: int, seed: int,
            shard_positions: int, sf_opponent_frac: float, sf_skill: int,
            policy_cls, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = {k: [] for k in ("packed", "meta", "ply", "clock", "eval_cp", "result",
                            "white_elo", "black_elo", "game_id")}
    shard_idx = 0
    shards = []
    game_id = 0
    t0 = time.time()

    sf_opponent = None
    if sf_opponent_frac > 0:
        from catspace.uci import UCIBoardPolicy
        sf_opponent = UCIBoardPolicy(skill=sf_skill, movetime=0.02)
        sf_opponent.__enter__()

    def flush():
        nonlocal shard_idx
        if not buf["packed"]:
            return
        path = out_dir / f"shard_{shard_idx:05d}.npz"
        np.savez(
            path,
            packed=np.array(buf["packed"], dtype=np.uint64),
            meta=np.array(buf["meta"], dtype=np.uint8),
            ply=np.array(buf["ply"], dtype=np.int32),
            clock=np.array(buf["clock"], dtype=np.float32),
            eval_cp=np.array(buf["eval_cp"], dtype=np.float32),
            result=np.array(buf["result"], dtype=np.int8),
            white_elo=np.array(buf["white_elo"], dtype=np.uint16),
            black_elo=np.array(buf["black_elo"], dtype=np.uint16),
            game_id=np.array(buf["game_id"], dtype=np.uint32),
        )
        shards.append({"file": path.name, "n": len(buf["packed"])})
        shard_idx += 1
        for k in buf:
            buf[k] = []

    try:
        for i in range(n_games):
            rng = np.random.default_rng([seed, i])
            white, black = make_selfplay_pair(fb, zgoals, device, max_nodes, beam, epsilon,
                                              policy_cls)
            use_sf = sf_opponent is not None and rng.random() < sf_opponent_frac
            if use_sf:
                if rng.random() < 0.5:
                    white = sf_opponent
                else:
                    black = sf_opponent
            rec = play_board_game(white, black, opening_plies=opening_plies,
                                  max_plies=max_plies, rng=rng)
            result = _RESULT_MAP[rec.result]
            rows = positions_of_game(rec)
            for r in rows:
                buf["packed"].append(r["packed"]); buf["meta"].append(r["meta"])
                buf["ply"].append(r["ply"]); buf["clock"].append(r["clock"])
                buf["eval_cp"].append(r["eval_cp"]); buf["result"].append(result)
                buf["white_elo"].append(elo); buf["black_elo"].append(elo)
                # odd ids only: train_lichess_fb's holdout rule drops
                # game_id % 50 == 0 rows, and self-play data is too scarce
                # to silently lose 2% of it to a filter meant for the
                # abundant human shards (odd numbers are never % 50 == 0)
                buf["game_id"].append(2 * game_id + 1)
            game_id += 1
            if verbose and (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  game {i + 1:4d}/{n_games}  result={rec.result:>7}  "
                     f"plies={rec.n_plies:3d}  sf_opp={use_sf}  ({rate:.2f} games/s)", flush=True)
            if len(buf["packed"]) >= shard_positions:
                flush()
    finally:
        flush()
        if sf_opponent is not None:
            sf_opponent.__exit__(None, None, None)

    total = sum(s["n"] for s in shards)
    manifest = dict(n_shards=len(shards), n_games=game_id, total_positions=total,
                    max_nodes=max_nodes, beam=beam, epsilon=epsilon,
                    opening_plies=opening_plies, sf_opponent_frac=sf_opponent_frac,
                    sf_skill=sf_skill, elo=elo, seed=seed)
    import json
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", required=True,
                    help="output shard dir -- put it under data/selfplay/, NOT data/shards/ "
                         "(newest_shard_dir() treats every dir there as a human-data candidate)")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.08,
                    help="per-move probability of a uniform-random legal move (diversity)")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--elo", type=int, default=1800, help="omega Elo bin stamped on self-play rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-positions", type=int, default=200_000)
    ap.add_argument("--sf-opponent-frac", type=float, default=0.3,
                    help="fraction of games where one side is Stockfish instead of self-play "
                         "-- external grounding so the field doesn't only reinforce its own "
                         "blind spots. Records only moves+result, never an eval score.")
    ap.add_argument("--sf-skill", type=int, default=3)
    ap.add_argument("--policy", choices=("search", "plan"), default="search")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBPlanPolicy, FBSearchPolicy

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    if "MATE_W" not in payload.get("zgoals", {}):
        raise SystemExit("checkpoint has no zgoals -- finish a train_lichess_fb.py run first")
    zgoals = {k: v.cpu().numpy() for k, v in payload["zgoals"].items()}
    policy_cls = FBSearchPolicy if args.policy == "search" else FBPlanPolicy

    print(f"self-play: {args.games} games, policy={args.policy}, max_nodes={args.max_nodes}, "
         f"beam={args.beam}, epsilon={args.epsilon}, sf_opponent_frac={args.sf_opponent_frac}, "
         f"ckpt={args.ckpt or 'default'}, device={device}")
    out_dir = Path(args.out_dir).resolve()
    from catspace.io.paths import shards_dir
    if shards_dir().resolve() in out_dir.parents:
        raise SystemExit(
            f"refusing to write self-play shards under {shards_dir()} -- "
            "newest_shard_dir() would silently adopt them as the default HUMAN "
            "training set (this exact mistake burned the 2026-07-12 round-13 "
            "first launch, see JOURNAL.md). Use data/selfplay/<name> instead.")
    manifest = generate(fb, zgoals, device, args.games, out_dir, args.max_nodes,
                        args.beam, args.epsilon, args.opening_plies, args.max_plies, args.elo,
                        args.seed, args.shard_positions, args.sf_opponent_frac, args.sf_skill,
                        policy_cls, verbose=True)
    print(f"wrote {manifest['n_shards']} shard(s), {manifest['n_games']} games, "
         f"{manifest['total_positions']} positions -> {args.out_dir}")


if __name__ == "__main__":
    main()
