"""FRAME: the batch a round learns from -- the positions where the player went wrong -- frozen by id.
- puzzles: the player, wired with the basis the round started from, on development puzzle rows (i % 97 == 5, never
  the held-out rows; at most 260 of them, beyond which they meet the general held-out set); each puzzle gives its
  first step the player does not solve outright: missed, or solved only inside a tie (Kaveh 2026-10-05);
- game losses: positions of development games (game_id % 97 == 5 in the 2019 Lichess records) where the side to move
  played a move its frames did not foresee losing (one-ply value >= 0 relative to passing, and the move's own
  forecast above -min_loss) and the game took at least `min_loss` pawns off it within `horizon` plies, the loss still
  standing after the mover's reply (an exchange in progress is not a loss); `per_game` caps positions per game (off).
Research slot: other framers (engine disagreements, the opponent's tactics, surprise) write the same Framed records."""
from __future__ import annotations

import json
import subprocess
import sys

import chess

from catspace.io import paths

from .. import basis
from ..contracts import Framed, to_dict
from ..store import write_json

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(b: chess.Board, side: bool) -> int:
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values())


def frame_squares(b: chess.Board, moves: list[chess.Move]) -> list[str]:
    from ...fates import square_frame
    sq: set = set()
    for m in moves:
        for s in (m.from_square, m.to_square):
            try:
                sq |= set(square_frame(b, s))
            except Exception:
                sq.add(s)
    return sorted(chess.square_name(s) for s in sq)


def frames_value(b: chess.Board, m: chess.Move) -> tuple[float, float]:
    """The frames' one-ply read of a move (the player's own static read; frame_player._one_ply_values for one move):
    -> (value, own). value = what it takes, less what the opponent's frames then win, plus what they win if we pass
    (relative to passing); own = what it takes less what they then win (the forecast of the move itself)."""
    from ... import frame_player as FP
    stand = FP.passed(b)
    before, book = FP.joint_threats_book(stand, FP.square_frame, select=FP.SELECT) if stand is not None else ((0.0, None, []), None)
    nb = b.copy(stack=False)
    nb.push(m)
    if nb.is_checkmate():
        return 1e6, 1e6
    after = FP.joint_threats(nb, FP.square_frame, select=FP.SELECT, parent=book, changed=FP.changed_squares(b, m))
    own = FP.captured_value(b, m) - after[0]
    return own + before[0], own


def loss_stands(boards: list, p: int, horizon: int, side: bool, min_loss: int) -> int | None:
    """The material the mover lost `horizon` plies after position p, when the loss still stands one ply later (after
    the mover's reply: an exchange in progress is not a loss) or the game ends there. -> the loss, or None. (The loss
    used to be read at exactly p + horizon, right after the opponent's move: 95 of round 1's 150 game positions were
    back above the threshold one ply later; the reviewer, 2026-10-05.)"""
    if p + horizon >= len(boards):
        return None
    base = material(boards[p], side)
    at = material(boards[p + horizon], side) - base
    later = material(boards[p + horizon + 1], side) - base if p + horizon + 1 < len(boards) else at
    worst_kept = max(at, later)
    return -worst_kept if worst_kept <= -min_loss else None


def puzzle_misses(ctx, cfg: dict, active_path: str) -> list[Framed]:
    puzzles = cfg.get("puzzles", 150)
    if puzzles > 260:                                                       # row 5 + 97 * 260 = 25,225 is also a general held-out row
        raise RuntimeError("development puzzle rows beyond 260 meet the general held-out set (i % 997 == 300); exclude it first")
    cmd = [sys.executable, "-m", "catspace.incubator.analyst.frame_puzzles", "--n", str(puzzles),
           "--stride", "97", "--offset", "5", "--show", "0", "--tag", f"loop{ctx.round:03d}dev", "--loop-active", active_path]
    ctx.log("frame: the player on development rows: " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, cwd=paths.REPO_ROOT, capture_output=True, text=True)
    (ctx.dir / "frame_player.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError("frame_puzzles failed: " + r.stderr[-2000:])
    dump = next(l.split("wrote ", 1)[1] for l in r.stdout.splitlines() if l.startswith("wrote "))
    res = json.load(open(dump))
    out, missed, tied = [], 0, 0
    for rec in res:
        # the first step the player misses (the solution outside the top moves), or solves only inside a tie with
        # other moves, where it picks at random (Kaveh 2026-10-05, on the reviewer's finding: 23 such puzzles of 150
        # were never framed); cfg "ties": False frames misses only
        ties = cfg.get("ties", True)                                        # Kaveh 2026-10-05: frame ties too
        step = next(((i, x) for i, x in enumerate(rec["steps"]) if not x["tied"] or (ties and (x.get("n_top") or 1) > 1)), None)
        if step is None:
            continue
        k, s = step
        b = chess.Board(s["fen"])
        alts = [u for u in (s.get("best") or []) if u != s["sol"]]
        played = chess.Move.from_uci(alts[0] if alts else s["best"][0])
        better = chess.Move.from_uci(s["sol"])
        kind = "miss" if not s["tied"] else "tie"
        missed += kind == "miss"
        tied += kind == "tie"
        out.append(Framed(f"puzzle:{rec['id']}:{k}", s["fen"], played.uci(), better.uci(), None, "puzzle-solution",
                          frame_squares(b, [played, better]), "puzzle",
                          dict(kind=kind, rating=rec["rating"], themes=rec["themes"][:6], v_top=s.get("v_top"), v_sol=s.get("v_sol"),
                               n_top=s.get("n_top"), dump=dump.rsplit("/", 1)[-1]),
                          ctx.prov("frame", [rec["id"]])))
    ctx.log(f"OBS frame: {len(res)} development puzzles: {missed} missed, {tied} solved only inside a tie")
    return out


def game_losses(ctx, cfg: dict) -> list[Framed]:
    import pyarrow.parquet as pq
    shard = cfg.get("shard", "records_00000.parquet")
    stride, residue = cfg.get("stride", 97), cfg.get("residue", 5)
    horizon, min_loss, min_ply = cfg.get("horizon", 4), cfg.get("min_loss", 2), cfg.get("min_ply", 10)
    cap, max_games, per_game = cfg.get("games_max_records", 150), cfg.get("games_max", 400), cfg.get("per_game")   # per_game: a cap
                                                                             # per game (DESIGN, asked 2026-10-05; None: no cap)
    t = pq.read_table(paths.REPO_ROOT / "data" / "records" / "lichess_2019-01" / shard, columns=["game_id", "moves", "white_elo", "black_elo"])
    out, games, foreseen, contributing = [], 0, 0, 0
    for row in t.to_pylist():
        if row["game_id"] % stride != residue:
            continue
        if games >= max_games or len(out) >= cap:
            break
        games += 1                                                          # counted when scanned (it counted the game it broke on)
        moves = row["moves"].split()
        boards = [chess.Board()]
        try:
            for u in moves:
                nb = boards[-1].copy(stack=False)
                nb.push_uci(u)
                boards.append(nb)
        except ValueError:
            continue
        taken = 0
        for p in range(min_ply, len(moves) - horizon + 1):                  # the last horizon may end on the final position
            b, side = boards[p], boards[p].turn
            lost = loss_stands(boards, p, horizon, side, min_loss)
            if lost is None:
                continue
            m = chess.Move.from_uci(moves[p])
            v, own = frames_value(b, m)
            if v < 0 or own <= -min_loss:                                   # the frames saw it coming (relative to passing, or
                foreseen += own <= -min_loss and v >= 0                     # the move's own forecast): no surprise
                continue
            out.append(Framed(f"game:{row['game_id']}:{p}", b.fen(), m.uci(), None, float(lost), "game-continuation",
                              frame_squares(b, [m]), "game",
                              dict(elo=row["white_elo"] if side else row["black_elo"], frames_value=v, frames_own=own,
                                   continuation=" ".join(moves[p:p + horizon + 1])),
                              ctx.prov("frame", [str(row["game_id"])])))
            taken += 1
            if len(out) >= cap or (per_game and taken >= per_game):
                break
        contributing += taken > 0
    ctx.log(f"OBS frame: {games} development games scanned, {contributing} contributed {len(out)} positions where the frames "
            f"forecast no loss and the game took >= {min_loss} pawns within {horizon} plies, still standing after the reply; "
            f"{foreseen} more were losses the move's own forecast saw")
    return out


def run(ctx) -> dict:
    from .use import wiring
    cfg = ctx.cfg("frame")
    if cfg.get("skip"):
        return dict(framed=[], skipped=True)
    active = wiring(ctx.round, [], [], before=basis.active(at=ctx.basis_before))   # the basis the round started from
    path = ctx.dir / "active_frame.json"
    write_json(path, active)
    framed = []
    if cfg.get("puzzles", 150):
        framed += puzzle_misses(ctx, cfg, str(path))
    if cfg.get("games_max", 400):
        framed += game_losses(ctx, cfg)
    write_json(ctx.record / "inputs_framed.json", [f.id for f in framed])
    return dict(framed=[to_dict(f) for f in framed], counts=dict(puzzle=sum(f.source == "puzzle" for f in framed), game=sum(f.source == "game" for f in framed)))
