"""THE MINER (plan §37 phase three), first experiment: counterexample-driven refinement on the PIN.

Mathematicians' fourth mechanism: state the cheap over-general claim, collect its counterexamples EXHAUSTIVELY
(the tablebase can, Stockfish cannot), and read the minimal guard off them. The pin is the test because we know
the answer in advance: "a pinned piece does not move" is over-general, and the guard is "unless its move gives
check, or takes the pinner, or the pin is only relative and the loss behind it is worth less than the move wins".
If the miner recovers that guard from the corpus without it being written, the miner can be trusted where we do
not know the answer.

The claim is a CONSTRAINT lemma, stated with its frame so it transfers out of tablebase territory:
    frame: an enemy slider S, our piece A on the line between S and our piece B, value(B) >= value(A), nothing
           else on the segments S..A and A..B.
    claim: a move of A that leaves the line loses the result.
The oracle is the corpus's exact partition of moves (endgames_v2: every legal move's result).

    python -m catspace.incubator.analyst.mine pin [--max-rows 100000]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import chess
import numpy as np

from .endgames import load as load_rows
from .facts import value as piece_value

OUT = Path("data/derived/mine")


def pins_of(board: chess.Board, side: chess.Color) -> list[tuple[int, int, int, bool]]:
    """(A, S, B, absolute): our piece A shielded from the enemy slider S by nothing, with our B behind it.
    B is the king for an absolute pin. Frame: the segments S..A and A..B hold nothing else."""
    out = []
    for a, pa in board.piece_map().items():
        if pa.color != side:
            continue
        for s, ps in board.piece_map().items():
            if ps.color == side or ps.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
                continue
            df, dr = chess.square_file(a) - chess.square_file(s), chess.square_rank(a) - chess.square_rank(s)
            straight = (df == 0) != (dr == 0); diag = abs(df) == abs(dr) and df != 0
            if not ((straight and ps.piece_type in (chess.QUEEN, chess.ROOK)) or (diag and ps.piece_type in (chess.QUEEN, chess.BISHOP))):
                continue
            if any(board.piece_at(t) for t in chess.SquareSet.between(s, a)):
                continue                                                   # the frame: S sees A
            step = ((df > 0) - (df < 0), (dr > 0) - (dr < 0))
            f, r = chess.square_file(a) + step[0], chess.square_rank(a) + step[1]
            b_sq = None
            while 0 <= f < 8 and 0 <= r < 8:
                t = chess.square(f, r)
                if board.piece_at(t) is not None:
                    b_sq = t; break
                f += step[0]; r += step[1]
            if b_sq is None:
                continue
            pb = board.piece_at(b_sq)
            if pb.color != side:
                continue
            if piece_value(board, b_sq) < piece_value(board, a):
                continue
            out.append((a, s, b_sq, pb.piece_type == chess.KING))
    return out


def features(board: chess.Board, a: int, s: int, b_sq: int, m: chess.Move) -> dict:
    """The candidate guards, all from the rules: what the move of A does."""
    nb = board.copy(stack=False); nb.push(m)
    return dict(gives_check=nb.is_check(),
                takes_the_pinner=m.to_square == s,
                stays_on_the_line=m.to_square in chess.SquareSet.between(s, b_sq) or m.to_square == s,
                captures=board.is_capture(m),
                absolute=board.piece_at(b_sq).piece_type == chess.KING,
                b_defended=bool(board.attackers(board.turn, b_sq) - {a}),
                gain_ge_loss=(piece_value(board, m.to_square) if board.is_capture(m) else 0.0) >= piece_value(board, b_sq) - 0.5)


def pin(max_rows: int, log=print):
    rows = load_rows(Path("data/derived/endgames_v2"))[:max_rows]
    n_pos = n_pins = 0
    tally = Counter()                       # (keeps, guard-tuple) -> count
    by_guard = defaultdict(lambda: [0, 0])  # guard name -> [keeps, total] among off-line moves
    off_line_total = off_line_keeps = 0
    examples = defaultdict(list)
    for r in rows:
        b = chess.Board(r["fen"]); moves = r["moves"]
        if not moves:
            continue
        best = max(np.sign(v) for v in moves.values())
        ps = pins_of(b, b.turn)
        if not ps:
            continue
        n_pos += 1
        for a, s, b_sq, absolute in ps:
            n_pins += 1
            for m in b.legal_moves:
                if m.from_square != a or m.uci() not in moves:
                    continue
                f = features(b, a, s, b_sq, m)
                if f["stays_on_the_line"] and not f["takes_the_pinner"]:
                    continue                                    # not a move that breaks the pin
                keeps = np.sign(moves[m.uci()]) == best
                off_line_total += 1; off_line_keeps += keeps
                for g in ("gives_check", "takes_the_pinner", "captures", "absolute", "b_defended", "gain_ge_loss"):
                    if f[g]:
                        by_guard[g][0] += keeps; by_guard[g][1] += 1
                key = tuple(sorted(k for k in ("gives_check", "takes_the_pinner", "captures", "absolute", "gain_ge_loss") if f[k]))
                tally[(bool(keeps), key)] += 1
                if keeps and len(examples[key]) < 2:
                    examples[key].append((r["fen"], b.san(m)))
    log(f"[mine/pin] {len(rows)} positions scanned; {n_pos} with a pin in the frame ({n_pins} pins); {off_line_total} moves of the pinned piece that leave the line")
    log(f"[mine/pin] THE NAIVE CLAIM 'a pinned piece's move off the line loses the result': it is WRONG in {off_line_keeps}/{off_line_total} = {100 * off_line_keeps / max(1, off_line_total):.1f}% of cases")
    log("[mine/pin] candidate guards, each: among the off-line moves that satisfy it, how many keep the result")
    for g, (k, t) in sorted(by_guard.items(), key=lambda kv: -(kv[1][0] / max(1, kv[1][1]))):
        log(f"    {g:<18} keeps {k}/{t} = {100 * k / max(1, t):.1f}%   (the base rate is {100 * off_line_keeps / max(1, off_line_total):.1f}%)")
    log("[mine/pin] the conjunctions that survive, biggest first (keeps / total):")
    combos = defaultdict(lambda: [0, 0])
    for (keeps, key), c in tally.items():
        combos[key][0] += c * keeps; combos[key][1] += c
    for key, (k, t) in sorted(combos.items(), key=lambda kv: -kv[1][1])[:10]:
        name = " AND ".join(key) if key else "(no guard: a plain move off the line)"
        log(f"    {name:<60} {k}/{t} = {100 * k / max(1, t):.1f}%" + (f"   e.g. {examples[key][0][1]} in {examples[key][0][0]}" if examples[key] else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(off_line_total=int(off_line_total), off_line_keeps=int(off_line_keeps),
                   guards={g: [int(x) for x in v] for g, v in by_guard.items()},
                   combos={" AND ".join(k) or "(none)": [int(x) for x in v] for k, v in combos.items()}),
              open(OUT / "pin.json", "w"), indent=1)


def pin_games(n: int, depth: int, lo: int, hi: int, within: int = 50, log=print):
    """The same experiment where the pin actually lives: middlegame positions, the oracle a search (Stockfish at
    a fixed depth). 'Keeps' = the move is within `within` centipawns of the engine's best. Only the pinned piece's
    off-line moves are searched, so the cost is a handful of calls a position."""
    from .dataset import load as load_ds
    from .engine import Engine
    import random as _r
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    rows = [m for m in meta if lo <= m["ply"] <= hi]
    _r.Random(0).shuffle(rows)
    eng = Engine(depth=depth, cache_path=str(OUT / f"pin_oracle_d{depth}.sqlite"))
    by_guard = defaultdict(lambda: [0, 0]); combos = defaultdict(lambda: [0, 0]); examples = defaultdict(list)
    off_total = off_keeps = n_pos = 0
    import time as _t; t0 = _t.perf_counter()
    for m_ in rows:
        if n_pos >= n:
            break
        b = chess.Board(m_["fen"])
        if b.is_game_over():
            continue
        ps = pins_of(b, b.turn)
        if not ps:
            continue
        cand = []
        for a_, s_, b_sq, absolute in ps:
            for mv in b.legal_moves:
                if mv.from_square != a_:
                    continue
                f = features(b, a_, s_, b_sq, mv)
                if f["stays_on_the_line"] and not f["takes_the_pinner"]:
                    continue
                cand.append((mv, f))
        if not cand:
            continue
        line = eng.best(b, depth=depth)
        if line is None:
            continue
        n_pos += 1
        for mv, f in cand:
            e = eng.eval_move(b, mv, depth=depth)
            if e is None:
                continue
            keeps = (float(line.cp) - float(e.cp)) <= within
            off_total += 1; off_keeps += keeps
            for g in ("gives_check", "takes_the_pinner", "captures", "absolute", "b_defended", "gain_ge_loss"):
                if f[g]:
                    by_guard[g][0] += keeps; by_guard[g][1] += 1
            key = tuple(sorted(k for k in ("gives_check", "takes_the_pinner", "captures", "absolute", "gain_ge_loss") if f[k]))
            combos[key][0] += keeps; combos[key][1] += 1
            if keeps and len(examples[key]) < 2:
                examples[key].append((m_["fen"], b.san(mv)))
        if n_pos % 25 == 0:
            log(f"[mine/pin-games] {n_pos}/{n} positions with a pin, {off_total} moves judged, {_t.perf_counter() - t0:.0f}s, engine calls {eng.calls} hits {eng.hits}", flush=True)
    eng.close()
    log(f"[mine/pin-games] {n_pos} middlegame positions with a pin in the frame; {off_total} moves of the pinned piece off the line, judged by Stockfish depth {depth} (within {within} cp of its best)")
    log(f"[mine/pin-games] THE NAIVE CLAIM 'the pinned piece does not move': WRONG in {off_keeps}/{off_total} = {100 * off_keeps / max(1, off_total):.1f}% of cases")
    for g, (k, t) in sorted(by_guard.items(), key=lambda kv: -(kv[1][0] / max(1, kv[1][1]))):
        log(f"    {g:<18} keeps {k}/{t} = {100 * k / max(1, t):.1f}%   (the base rate is {100 * off_keeps / max(1, off_total):.1f}%)")
    log("[mine/pin-games] the conjunctions, biggest first:")
    for key, (k, t) in sorted(combos.items(), key=lambda kv: -kv[1][1])[:8]:
        name = " AND ".join(key) if key else "(no guard: a plain move off the line)"
        log(f"    {name:<58} {k}/{t} = {100 * k / max(1, t):.1f}%" + (f"   e.g. {examples[key][0][1]}" if examples[key] else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(positions=n_pos, off_total=off_total, off_keeps=int(off_keeps), depth=depth, within=within,
                   guards={g: [int(x) for x in v] for g, v in by_guard.items()},
                   combos={" AND ".join(k) or "(none)": [int(x) for x in v] for k, v in combos.items()}),
              open(OUT / "pin_games.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pin", "pin-games"]); ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--within", type=int, default=50); ap.add_argument("--plies", default="21,70")
    a = ap.parse_args(argv)
    if a.cmd == "pin":
        pin(a.max_rows)
    else:
        lo, hi = (int(x) for x in a.plies.split(","))
        pin_games(a.n, a.depth, lo, hi, a.within)


if __name__ == "__main__":
    main()
