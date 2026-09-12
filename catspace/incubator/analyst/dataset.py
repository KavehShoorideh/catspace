"""dataset.py -- the knowledge dataset: one interaction graph per position from real games, with the engine's verdict
and the game's future (Kaveh 2026-09-09: "is there a trajectory in the knowledge?").

Per position: the graph's node matrix (pieces x NODE_QUESTIONS), the sparse pair relations, the position indicators,
the zones; the engine's best move and evaluation at a fixed depth; the move played; the game result from the side
to move's view; and the REGIME AHEAD: the position indicators 10 and 20 plies later in the same game (material,
queens, rooks connected, passed pawns, phase...), plus how and when the game ended. Sharded so eight processes can
extract in parallel; `merge` joins the shards.

    python -m catspace.incubator.analyst.dataset extract data/lichess/huvaxshtra_all.pgn --out data/derived/knowledge --shard 0/8 [--every 1 --depth 12]
    python -m catspace.incubator.analyst.dataset merge data/derived/knowledge
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .memo import Memo
from .graph import build, NODE_QUESTIONS, EDGE_RELATIONS, POSITION_QUESTIONS
from .rank import move_kind

MAX_PIECES = 32
AHEAD = (10, 20)
REGIME_KEYS = ("material_diff", "phase", "pieces", "queens_on", "rooks_connected_ours", "rooks_connected_theirs",
               "passed_ours", "passed_theirs", "runner_ours", "runner_theirs", "bishop_pair_ours", "bishop_pair_theirs",
               "pawns_ours", "pawns_theirs", "open_files")


def regime_of(board: chess.Board, side: chess.Color) -> list[float]:
    """The cheap regime indicators of a position from `side`'s view (no operators: material, queens, rooks, pawns)."""
    from .facts import value
    b = board
    def cnt(pt, c):
        return len(b.pieces(pt, c))
    us, them = side, not side
    rooks = {c: sorted(b.pieces(chess.ROOK, c)) for c in (us, them)}
    def connected(c):
        r = rooks[c]
        return float(len(r) >= 2 and any(r[j] in b.attacks(r[i]) for i in range(len(r)) for j in range(len(r)) if i != j))
    def passed(c):
        out = []
        for sq in b.pieces(chess.PAWN, c):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            ahead = [t for t in b.pieces(chess.PAWN, not c) if abs(chess.square_file(t) - f) <= 1 and ((chess.square_rank(t) > r) if c == chess.WHITE else (chess.square_rank(t) < r))]
            if not ahead:
                out.append((7 - r) if c == chess.WHITE else r)
        return out
    pu, pt_ = passed(us), passed(them)
    def pair(c):
        bs = list(b.pieces(chess.BISHOP, c))
        return float(len(bs) >= 2 and len({(chess.square_file(x) + chess.square_rank(x)) % 2 for x in bs}) == 2)
    mat = lambda c: sum(value(b, s) for s in chess.scan_forward(b.occupied) if b.piece_at(s).color == c and b.piece_at(s).piece_type != chess.KING)
    vals = dict(material_diff=mat(us) - mat(them), phase=sum(value(b, s) for s in chess.scan_forward(b.occupied) if b.piece_at(s).piece_type not in (chess.PAWN, chess.KING)),
                pieces=chess.popcount(b.occupied), queens_on=cnt(chess.QUEEN, us) + cnt(chess.QUEEN, them),
                rooks_connected_ours=connected(us), rooks_connected_theirs=connected(them), passed_ours=len(pu), passed_theirs=len(pt_),
                runner_ours=min(pu, default=0), runner_theirs=min(pt_, default=0), bishop_pair_ours=pair(us), bishop_pair_theirs=pair(them),
                pawns_ours=cnt(chess.PAWN, us), pawns_theirs=cnt(chess.PAWN, them),
                open_files=sum(1 for f in range(8) if not any(chess.square_file(t) == f for t in b.pieces(chess.PAWN, chess.WHITE) | b.pieces(chess.PAWN, chess.BLACK))))
    return [float(vals[k]) for k in REGIME_KEYS]


def extract(pgn_path: str, out_dir: str, shard: str = "0/1", every: int = 1, depth: int = 12, min_ply: int = 6, max_games: int = 0, log=print):
    from .engine import Engine
    i, k = (int(x) for x in shard.split("/"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    eng = Engine(depth=depth, cache_path=str(out / f"engine_{i}.sqlite"))
    memo = Memo()
    games = []
    with open(pgn_path) as f:
        gi = 0
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            if gi % k == i:
                games.append((gi, g))
            gi += 1
            if max_games and len(games) >= max_games:
                break
    nodes_l, mask_l, pos_l, edge_l, meta = [], [], [], [], []
    t0 = time.perf_counter(); n = 0
    for gi, game in games:
        moves = list(game.mainline_moves())
        boards = [game.board()]
        for m in moves:
            b = boards[-1].copy(stack=False); b.push(m); boards.append(b)
        res = game.headers.get("Result", "*")
        term = game.headers.get("Termination", "")
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res)
        for ply in range(min_ply, len(moves)):
            if ply % every:
                continue
            board = boards[ply]
            if board.is_game_over():
                break
            try:
                g = build(board, memo)
                line = eng.best(board, depth=depth)
            except Exception as ex:
                log(f"[dataset] {gi}:{ply} error {ex}", flush=True); continue
            side = board.turn
            nm = g.node_matrix(); n_p = nm.shape[0]
            padded = np.zeros((MAX_PIECES, len(NODE_QUESTIONS)), dtype=np.float32); padded[:n_p] = nm
            mask = np.zeros(MAX_PIECES, dtype=bool); mask[:n_p] = True
            # sparse pairs: (i, j, relation values...) as float16 rows
            rows = []
            col = {r: c for c, r in enumerate(EDGE_RELATIONS)}
            for (a, b_), rel in g.edges.items():
                v = np.zeros(len(EDGE_RELATIONS), dtype=np.float16)
                for r, x in rel.items():
                    v[col[r]] = x
                rows.append(np.concatenate([np.array([a, b_], dtype=np.float16), v]))
            edges = np.stack(rows) if rows else np.zeros((0, 2 + len(EDGE_RELATIONS)), dtype=np.float16)
            squares = np.array([nd.square for nd in g.nodes] + [64] * (MAX_PIECES - n_p), dtype=np.int8)
            types = np.array([nd.piece.piece_type * (1 if nd.piece.color == side else -1) for nd in g.nodes] + [0] * (MAX_PIECES - n_p), dtype=np.int8)
            zone_id = np.full(MAX_PIECES, -1, dtype=np.int8)
            for zi, z in enumerate(g.zones):
                for sq in z:
                    for j, nd in enumerate(g.nodes):
                        if nd.square == sq and zone_id[j] < 0:
                            zone_id[j] = zi
            ahead = {}
            for h in AHEAD:
                if ply + h < len(boards):
                    ahead[h] = regime_of(boards[ply + h], side)
                else:
                    ahead[h] = regime_of(boards[-1], side)
            score = None if white_score is None else (white_score if side == chess.WHITE else 1.0 - white_score)
            played = moves[ply]
            best = chess.Move.from_uci(line.move) if line else None
            meta.append(dict(game=gi, site=game.headers.get("Site", ""), ply=ply, fen=board.fen(), side=int(side), n_pieces=n_p,
                             best=line.move if line else "", best_cp=line.cp if line else None, best_mate=line.mate if line else None,
                             best_kind=move_kind(board, best) if best else "", played=played.uci(), played_kind=move_kind(board, played),
                             result=score, termination=term, plies_left=len(moves) - ply,
                             ahead={str(h): v for h, v in ahead.items()}, ended_within={str(h): int(ply + h >= len(moves)) for h in AHEAD},
                             zones=len(g.zones), temperatures=g.temperatures[:4], seconds=g.seconds, squares=squares.tolist(), types=types.tolist(), zone_id=zone_id.tolist()))
            nodes_l.append(padded); mask_l.append(mask); pos_l.append(g.position_vector()); edge_l.append(edges)
            n += 1
            if n % 50 == 0:
                log(f"[dataset {shard}] {n} positions, {time.perf_counter() - t0:.0f}s ({(time.perf_counter() - t0) / n:.2f}s each), game {gi}, engine calls {eng.calls} hits {eng.hits}", flush=True)
    eng.close()
    edge_offsets = np.cumsum([0] + [e.shape[0] for e in edge_l])
    np.savez_compressed(out / f"shard_{i}.npz", nodes=np.stack(nodes_l) if nodes_l else np.zeros((0, MAX_PIECES, len(NODE_QUESTIONS)), np.float32),
                        mask=np.stack(mask_l) if mask_l else np.zeros((0, MAX_PIECES), bool), position=np.stack(pos_l) if pos_l else np.zeros((0, len(POSITION_QUESTIONS)), np.float32),
                        edges=np.concatenate(edge_l) if edge_l else np.zeros((0, 2 + len(EDGE_RELATIONS)), np.float16), edge_offsets=edge_offsets,
                        node_questions=np.array(NODE_QUESTIONS), edge_relations=np.array(EDGE_RELATIONS), position_questions=np.array(POSITION_QUESTIONS), regime_keys=np.array(REGIME_KEYS))
    with open(out / f"shard_{i}.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    dt = time.perf_counter() - t0
    log(f"[dataset {shard}] done: {n} positions from {len(games)} games in {dt:.0f}s ({dt / max(1, n):.2f}s each); {memo.report()}", flush=True)
    return n, dt


def extract_fens(rows_path: str, out_dir: str, shard: str = "0/1", log=print):
    """Positions given as rows (fen, labels: uci -> win probability for every legal move, wp, cp) -- a labeled corpus
    such as ChessBench -- extracted to the same shard format. No game around them: no future, no played move
    (the best-labeled move stands in), a unique game id per position so no trajectory pairs form."""
    i, k = (int(x) for x in shard.split("/"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for n, l in enumerate(open(rows_path)) if n % k == i]
    memo = Memo()
    nodes_l, mask_l, pos_l, edge_l, meta = [], [], [], [], []
    t0 = time.perf_counter(); n = 0
    col = {r: c for c, r in enumerate(EDGE_RELATIONS)}
    for gi, row in enumerate(rows):
        board = chess.Board(row["fen"])
        if board.is_game_over() or not board.is_valid():
            continue
        try:
            g = build(board, memo)
        except Exception as ex:
            log(f"[dataset] {gi} error {ex}", flush=True); continue
        side = board.turn
        nm = g.node_matrix(); n_p = nm.shape[0]
        padded = np.zeros((MAX_PIECES, len(NODE_QUESTIONS)), dtype=np.float32); padded[:n_p] = nm
        mask = np.zeros(MAX_PIECES, dtype=bool); mask[:n_p] = True
        er = []
        for (a, b_), rel in g.edges.items():
            v = np.zeros(len(EDGE_RELATIONS), dtype=np.float16)
            for r, x in rel.items():
                v[col[r]] = x
            er.append(np.concatenate([np.array([a, b_], dtype=np.float16), v]))
        edges = np.stack(er) if er else np.zeros((0, 2 + len(EDGE_RELATIONS)), dtype=np.float16)
        squares = np.array([nd.square for nd in g.nodes] + [64] * (MAX_PIECES - n_p), dtype=np.int8)
        types = np.array([nd.piece.piece_type * (1 if nd.piece.color == side else -1) for nd in g.nodes] + [0] * (MAX_PIECES - n_p), dtype=np.int8)
        zone_id = np.full(MAX_PIECES, -1, dtype=np.int8)
        labels = row["labels"]; best = max(labels, key=labels.get)
        reg = regime_of(board, side)
        meta.append(dict(game=-1000000 - (gi * k + i), site="chessbench", ply=0, fen=row["fen"], side=int(side), n_pieces=n_p,
                         best=best, best_cp=int(round(row["cp"])), best_mate=None, best_kind=move_kind(board, chess.Move.from_uci(best)),
                         played=best, played_kind=move_kind(board, chess.Move.from_uci(best)), result=None, termination="", plies_left=0,
                         ahead={str(h): reg for h in AHEAD}, ended_within={str(h): 1 for h in AHEAD}, labels=labels, wp=row["wp"],
                         zones=len(g.zones), temperatures=g.temperatures[:4], seconds=g.seconds, squares=squares.tolist(), types=types.tolist(), zone_id=zone_id.tolist()))
        nodes_l.append(padded); mask_l.append(mask); pos_l.append(g.position_vector()); edge_l.append(edges)
        n += 1
        if n % 100 == 0:
            log(f"[dataset {shard}] {n} positions, {time.perf_counter() - t0:.0f}s ({(time.perf_counter() - t0) / n:.2f}s each)", flush=True)
    edge_offsets = np.cumsum([0] + [e.shape[0] for e in edge_l])
    np.savez_compressed(out / f"shard_{i}.npz", nodes=np.stack(nodes_l), mask=np.stack(mask_l), position=np.stack(pos_l), edges=np.concatenate(edge_l), edge_offsets=edge_offsets,
                        node_questions=np.array(NODE_QUESTIONS), edge_relations=np.array(EDGE_RELATIONS), position_questions=np.array(POSITION_QUESTIONS), regime_keys=np.array(REGIME_KEYS))
    with open(out / f"shard_{i}.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    log(f"[dataset {shard}] done: {n} positions in {time.perf_counter() - t0:.0f}s; {memo.report()}", flush=True)


def load(out_dir: str, shards=None, meta_only: bool = False):
    """All shards joined: arrays + the metadata rows (`meta_only`: the rows alone, for scripts that need ids and FENs)."""
    out = Path(out_dir)
    files = sorted(out.glob("shard_*.npz")) if shards is None else [out / f"shard_{i}.npz" for i in shards]
    metas = []
    for f in files:
        with open(f.with_suffix(".jsonl")) as fh:
            metas.extend(json.loads(l) for l in fh)
    if meta_only:
        return dict(meta=metas)
    parts = [np.load(f, allow_pickle=False) for f in files]
    edges = []; offsets = [0]
    for p in parts:
        e, o = p["edges"], p["edge_offsets"]
        for a, b in zip(o[:-1], o[1:]):
            edges.append(e[a:b]); offsets.append(offsets[-1] + (b - a))
    data = dict(nodes=np.concatenate([p["nodes"] for p in parts]), mask=np.concatenate([p["mask"] for p in parts]),
                position=np.concatenate([p["position"] for p in parts]), edges=edges, meta=metas)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "extract_fens", "merge"]); ap.add_argument("path"); ap.add_argument("--out", default="data/derived/knowledge")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--every", type=int, default=1); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        extract(a.path, a.out, a.shard, a.every, a.depth, max_games=a.max_games)
    elif a.cmd == "extract_fens":
        extract_fens(a.path, a.out, a.shard)
    else:
        d = load(a.path)
        print(f"{len(d['meta'])} positions; nodes {d['nodes'].shape}, position {d['position'].shape}, pairs {sum(len(e) for e in d['edges'])}")


if __name__ == "__main__":
    main()
