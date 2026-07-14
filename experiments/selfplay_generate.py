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


def positions_of_game(rec, start: chess.Board | None = None) -> list[dict]:
    """Mirrors data.lichess.positions_of's dict shape (packed/meta/ply/
    clock/eval_cp), replaying rec.moves to get one row per ply INCLUDING
    the final position (checkmate finals live there -- needed for zgoal
    rebuilding on self-play-inclusive checkpoints). `start` must be the
    same board the game was played from (endgame-curriculum starts)."""
    board = start.copy(stack=False) if start is not None else chess.Board()
    out = [dict(packed=encode_packed(board), meta=encode_meta(board), ply=0,
               clock=float("nan"), eval_cp=float("nan"))]
    for uci in rec.moves:
        board.push(chess.Move.from_uci(uci))
        out.append(dict(packed=encode_packed(board), meta=encode_meta(board), ply=len(out),
                        clock=float("nan"), eval_cp=float("nan")))
    return out


_ENDGAME_MENUS = {
    "krvk": [(chess.ROOK, chess.WHITE)],
    "kqvk": [(chess.QUEEN, chess.WHITE)],
    "krrvk": [(chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE)],
    "krrkbp": [(chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE),
               (chess.BISHOP, chess.BLACK), (chess.PAWN, chess.BLACK)],
    "kqvkp": [(chess.QUEEN, chess.WHITE), (chess.PAWN, chess.BLACK)],
}

# THE canonical toy start (Kaveh, 2026-07-14: no random placements -- Leela
# trains from one fixed start; state diversity must come from PLAY, so the
# data distribution is the reachable set, not scattered legal positions).
# Home-square-like placement; no castling rights (syzygy can't probe them).
# Verified: syzygy wdl=+2 (clean White win), White to move.
KRRKBP_FIXED_START = "2b1k3/3p4/8/8/8/8/8/R3K2R w - - 0 1"


def openings_from_fixed_start(rng: np.random.Generator, n: int, tb,
                              start_fen: str = KRRKBP_FIXED_START,
                              min_plies: int = 2, max_plies: int = 10,
                              require_wdl: int = 2) -> list[str]:
    """Sample n distinct White-to-move opening positions REACHED from the
    fixed start by uniform-random legal play (an even number of plies in
    [min_plies, max_plies]), keeping only positions the tablebase still
    scores wdl=`require_wdl` for the mover -- evals then measure conversion
    of a still-won game, and every position is play-reachable by
    construction. Returns FENs (dedup'd)."""
    out: list[str] = []
    seen = set()
    tries = 0
    while len(out) < n and tries < 200_000:
        tries += 1
        b = chess.Board(start_fen)
        plies = 2 * int(rng.integers(min_plies // 2, max_plies // 2 + 1))
        ok = True
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over(claim_draw=True):
                ok = False
                break
            b.push(moves[int(rng.integers(len(moves)))])
        if not ok or b.is_game_over(claim_draw=True) or b.turn != chess.WHITE:
            continue
        fen = b.fen()
        if fen in seen:
            continue
        w, _ = tb.wdl_dtz(b)
        if w == require_wdl:
            seen.add(fen)
            out.append(fen)
    return out


def random_endgame_start(rng: np.random.Generator, material: str | None = None) -> chess.Board | None:
    """Random legal winnable-material endgame start for curriculum generation
    (2026-07-12: the qm_fitness_probe found the learned distance completely
    FLAT against true distance-to-mate in endgame regions -- Spearman ~0 on
    KRvK where tablebase DTZ == plies-to-mate -- because human games rarely
    reach them. Games STARTED here produce real trajectories with real
    outcomes through exactly that blind region: outcome-grounded coverage,
    no oracle labels). `material` (e.g. 'krrkbp') restricts to one menu;
    None mixes all. Material mixes: KRvK, KQvK, KRRvK, KRRvKBP-family."""
    if material is not None:
        pieces = list(_ENDGAME_MENUS[material])
    else:
        menus = list(_ENDGAME_MENUS.values())
        pieces = list(menus[int(rng.integers(len(menus)))])
    for _ in range(200):
        n = 2 + len(pieces)
        squares = rng.choice(64, size=n, replace=False)
        board = chess.Board(None)
        board.set_piece_at(int(squares[0]), chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(int(squares[1]), chess.Piece(chess.KING, chess.BLACK))
        ok = True
        for sq, (pt, color) in zip(squares[2:], pieces):
            if pt == chess.PAWN and chess.square_rank(int(sq)) in (0, 7):
                ok = False
                break
            board.set_piece_at(int(sq), chess.Piece(pt, color))
        if not ok:
            continue
        board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        # is_valid() rejects kings-adjacent / side-not-to-move-in-check etc.
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            continue
        return board
    return None


def generate(fb, zgoals, device, n_games: int, out_dir: Path, max_nodes: int, beam: int,
            epsilon: float, opening_plies: int, max_plies: int, elo: int, seed: int,
            shard_positions: int, sf_opponent_frac: float, sf_skill: int,
            policy_cls, endgame_start_frac: float = 0.0, start_fens: list | None = None,
            sf_vs_sf: bool = False, endgame_material: str | None = None,
            sf_movetime: float = 0.02, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = {k: [] for k in ("packed", "meta", "ply", "clock", "eval_cp", "result",
                            "white_elo", "black_elo", "game_id")}
    shard_idx = 0
    shards = []
    game_id = 0
    t0 = time.time()

    sf_opponent = None
    if sf_opponent_frac > 0 or sf_vs_sf:
        from catspace.uci import UCIBoardPolicy
        sf_opponent = UCIBoardPolicy(skill=sf_skill, movetime=sf_movetime)
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
            if sf_vs_sf:
                # BOTH sides Stockfish -- strong, CORRECT conversions through the
                # endgame region (dense mate signal the FB self-play was too weak
                # to produce). One engine serves both sides (moves are sequential).
                # Records only moves + result, never an eval score: leakage-clean,
                # same status as a human game vs a strong opponent.
                white = black = sf_opponent
                use_sf = True
            else:
                white, black = make_selfplay_pair(fb, zgoals, device, max_nodes, beam, epsilon,
                                                  policy_cls)
                use_sf = sf_opponent is not None and rng.random() < sf_opponent_frac
                if use_sf:
                    if rng.random() < 0.5:
                        white = sf_opponent
                    else:
                        black = sf_opponent
            start = None
            if start_fens:
                # toy-scenario mode: every game launches from a fixed start
                # position (cycled), e.g. the KRRvKBP fixed set -- so coverage
                # is concentrated exactly on the region we're studying.
                start = chess.Board(start_fens[i % len(start_fens)])
            elif rng.random() < endgame_start_frac:
                start = random_endgame_start(rng, material=endgame_material)
            rec = play_board_game(white, black, start=start,
                                  opening_plies=0 if start is not None else opening_plies,
                                  max_plies=max_plies, rng=rng)
            result = _RESULT_MAP[rec.result]
            rows = positions_of_game(rec, start=start)
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
                    sf_skill=sf_skill, elo=elo, seed=seed,
                    endgame_start_frac=endgame_start_frac)
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
    ap.add_argument("--endgame-start-frac", type=float, default=0.0,
                    help="fraction of games started from random winnable-material endgame "
                         "positions (KRvK/KQvK/KRRvK/KRRvKBP-family) instead of the initial "
                         "position -- curriculum coverage for the endgame regions where the "
                         "qm_fitness_probe found distance-to-mate calibration completely flat")
    ap.add_argument("--sf-opponent-frac", type=float, default=0.3,
                    help="fraction of games where one side is Stockfish instead of self-play "
                         "-- external grounding so the field doesn't only reinforce its own "
                         "blind spots. Records only moves+result, never an eval score.")
    ap.add_argument("--sf-skill", type=int, default=3)
    ap.add_argument("--sf-movetime", type=float, default=0.02,
                    help="Stockfish seconds/move (raise for correct endgame conversion; "
                         "0.1 is plenty for KRRvKBP)")
    ap.add_argument("--sf-vs-sf", action="store_true",
                    help="BOTH sides Stockfish (no FB policy) -- strong, correct endgame "
                         "conversions for dense mate signal. Records only moves+result, never "
                         "an eval score (leakage-clean). Pair with --endgame-start-frac 1.0 "
                         "--endgame-material krrkbp for the KRRvKBP toy.")
    ap.add_argument("--endgame-material", choices=list(_ENDGAME_MENUS), default=None,
                    help="restrict --endgame-start-frac starts to one material (e.g. krrkbp); "
                         "default mixes all menus.")
    ap.add_argument("--policy", choices=("search", "plan"), default="search")
    ap.add_argument("--start-fens", default=None,
                    help="path to a JSON {'fens': [...]} of start positions; every game "
                         "launches from one (cycled) instead of the initial position -- "
                         "TOY-scenario mode, concentrates self-play on a specific region "
                         "(e.g. artifacts/experiments/krrkbp_fixed_set_n60.json). Overrides "
                         "--endgame-start-frac and --opening-plies.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBPlanPolicy, FBSearchPolicy

    device = pick_device(args.device)
    if args.sf_vs_sf:
        # no FB policy needed -- both sides Stockfish
        fb, zgoals, policy_cls = None, None, None
    else:
        fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
        if "MATE_W" not in payload.get("zgoals", {}):
            raise SystemExit("checkpoint has no zgoals -- finish a train_lichess_fb.py run first")
        zgoals = {k: v.cpu().numpy() for k, v in payload["zgoals"].items()}
        policy_cls = FBSearchPolicy if args.policy == "search" else FBPlanPolicy

    start_fens = None
    if args.start_fens:
        import json
        start_fens = json.loads(Path(args.start_fens).read_text())["fens"]
        print(f"toy-scenario mode: {len(start_fens)} fixed start positions from {args.start_fens}")

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
                        policy_cls, endgame_start_frac=args.endgame_start_frac,
                        start_fens=start_fens, sf_vs_sf=args.sf_vs_sf,
                        endgame_material=args.endgame_material, sf_movetime=args.sf_movetime,
                        verbose=True)
    print(f"wrote {manifest['n_shards']} shard(s), {manifest['n_games']} games, "
         f"{manifest['total_positions']} positions -> {args.out_dir}")


if __name__ == "__main__":
    main()
