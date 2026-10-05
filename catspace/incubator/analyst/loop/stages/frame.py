"""FRAME: the batch a round learns from -- the positions where the player went wrong -- frozen by id.
- puzzle misses: the player, wired with the basis as it stands, on development puzzle rows (i % 97 == 5, never the
  held-out rows); each puzzle it fails gives its first missed step: the position, the move it chose, the solution;
- game losses: positions of development games (game_id % 97 == 5 in the 2019 Lichess records) where the side to
  move played a move its frames rated as safe (one-ply frame value >= 0) and the game's continuation took at least
  `min_loss` pawns off it within `horizon` plies: the frames and the game disagree.
Research slot: other framers (engine disagreements, the opponent's tactics) write the same Framed records."""
from __future__ import annotations

import json
import subprocess
import sys

import chess

from catspace.io import paths

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


def frames_value(b: chess.Board, m: chess.Move) -> float:
    """The frames' one-ply value of a move (the player's own static read): what it takes, less what the opponent's
    frames then win, plus what they win if we pass (frame_player._one_ply_values, for one move)."""
    from ... import frame_player as FP
    stand = FP.passed(b)
    before, book = FP.joint_threats_book(stand, FP.square_frame, select=FP.SELECT) if stand is not None else ((0.0, None, []), None)
    nb = b.copy(stack=False)
    nb.push(m)
    if nb.is_checkmate():
        return 1e6
    after = FP.joint_threats(nb, FP.square_frame, select=FP.SELECT, parent=book, changed=FP.changed_squares(b, m))
    return FP.captured_value(b, m) - after[0] + before[0]


def puzzle_misses(ctx, cfg: dict, active_path: str) -> list[Framed]:
    cmd = [sys.executable, "-m", "catspace.incubator.analyst.frame_puzzles", "--n", str(cfg.get("puzzles", 150)),
           "--stride", "97", "--offset", "5", "--show", "0", "--tag", f"loop{ctx.round:03d}dev", "--loop-active", active_path]
    ctx.log("frame: the player on development rows: " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, cwd=paths.REPO_ROOT, capture_output=True, text=True)
    (ctx.dir / "frame_player.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError("frame_puzzles failed: " + r.stderr[-2000:])
    dump = next(l.split("wrote ", 1)[1] for l in r.stdout.splitlines() if l.startswith("wrote "))
    res = json.load(open(dump))
    out = []
    for rec in res:
        if rec["solved"]:
            continue
        k, s = next((i, x) for i, x in enumerate(rec["steps"]) if not x["tied"])
        b = chess.Board(s["fen"])
        played = chess.Move.from_uci(s["best"][0])
        better = chess.Move.from_uci(s["sol"])
        out.append(Framed(f"puzzle:{rec['id']}:{k}", s["fen"], played.uci(), better.uci(), None, "puzzle-solution",
                          frame_squares(b, [played, better]), "puzzle",
                          dict(rating=rec["rating"], themes=rec["themes"][:6], v_top=s.get("v_top"), v_sol=s.get("v_sol"), n_top=s.get("n_top"), dump=dump.rsplit("/", 1)[-1]),
                          ctx.prov("frame", [rec["id"]])))
    ctx.log(f"OBS frame: {len(res)} development puzzles, {len(out)} missed")
    return out


def game_losses(ctx, cfg: dict) -> list[Framed]:
    import pyarrow.parquet as pq
    shard = cfg.get("shard", "records_00000.parquet")
    stride, residue = cfg.get("stride", 97), cfg.get("residue", 5)
    horizon, min_loss, min_ply = cfg.get("horizon", 4), cfg.get("min_loss", 2), cfg.get("min_ply", 10)
    cap, max_games = cfg.get("games_max_records", 150), cfg.get("games_max", 400)
    t = pq.read_table(paths.REPO_ROOT / "data" / "records" / "lichess_2019-01" / shard, columns=["game_id", "moves", "white_elo", "black_elo"])
    out, games = [], 0
    for row in t.to_pylist():
        if row["game_id"] % stride != residue:
            continue
        games += 1
        if games > max_games or len(out) >= cap:
            break
        moves = row["moves"].split()
        boards = [chess.Board()]
        try:
            for u in moves:
                nb = boards[-1].copy(stack=False)
                nb.push_uci(u)
                boards.append(nb)
        except ValueError:
            continue
        for p in range(min_ply, len(moves) - horizon):
            b, side = boards[p], boards[p].turn
            realized = material(boards[p + horizon], side) - material(b, side)
            if realized > -min_loss:
                continue
            m = chess.Move.from_uci(moves[p])
            v = frames_value(b, m)
            if v < 0:
                continue                                                    # the frames saw it coming: no disagreement
            out.append(Framed(f"game:{row['game_id']}:{p}", b.fen(), m.uci(), None, float(-realized), "game-continuation",
                              frame_squares(b, [m]), "game",
                              dict(elo=row["white_elo"] if side else row["black_elo"], frames_value=v, continuation=" ".join(moves[p:p + horizon])),
                              ctx.prov("frame", [str(row["game_id"])])))
            if len(out) >= cap:
                break
    ctx.log(f"OBS frame: {games} development games scanned, {len(out)} positions where the frames said safe and the game took >= {min_loss} pawns within {horizon} plies")
    return out


def run(ctx) -> dict:
    from .use import wiring
    cfg = ctx.cfg("frame")
    if cfg.get("skip"):
        return dict(framed=[], skipped=True)
    active = wiring(ctx.round, [], [])                                       # the basis as it stands before this round
    path = ctx.dir / "active_frame.json"
    write_json(path, active)
    framed = []
    if cfg.get("puzzles", 150):
        framed += puzzle_misses(ctx, cfg, str(path))
    if cfg.get("games_max", 400):
        framed += game_losses(ctx, cfg)
    write_json(ctx.record / "inputs_framed.json", [f.id for f in framed])
    return dict(framed=[to_dict(f) for f in framed], counts=dict(puzzle=sum(f.source == "puzzle" for f in framed), game=sum(f.source == "game" for f in framed)))
