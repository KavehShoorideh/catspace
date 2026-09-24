"""THE BEST GOAL, WITH A SOUND PRUNE (Kaveh 2026-09-24): "what is the best goal we can achieve in this position?"
Frames are the source of bounds and of achieved values; the choice is between MOVES. A move's optimistic value is
the most any frame around it could reach when extended: the worth of every enemy piece the frame's pieces can touch
within the episode, plus the library's largest claimed gain for the move. Best-first over our moves, a move is
opened only if its bound beats the best value already achieved (alpha). The opponent's frames are never pruned:
their best threat is what our move must answer. A prune that changes the answer is a bug: `audit` runs on and off."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess

from .chains import known_frames
from .chess_domain import VAL
from .fates import Outcome, outcome

MATE = 500.0                                                                 # values beyond this are mates: their own class, above any material bound


@dataclass
class Choice:
    move: chess.Move
    bound: float
    value: float | None = None       # None: pruned
    frame: list = field(default_factory=list)
    outcome: Outcome | None = None
    pruned: bool = False


def _blocks(board: chess.Board, sq: int) -> list:
    """(x, y) pairs where the piece on sq stands between a slider x that reaches sq and the next piece y beyond it on
    the same line: what leaves the line when sq is vacated."""
    out = []
    for x in chess.SQUARES:
        px = board.piece_at(x)
        if px is None or x == sq or px.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN) or sq not in board.attacks(x):
            continue
        df = (chess.square_file(sq) > chess.square_file(x)) - (chess.square_file(sq) < chess.square_file(x))
        dr = (chess.square_rank(sq) > chess.square_rank(x)) - (chess.square_rank(sq) < chess.square_rank(x))
        f, r = chess.square_file(sq) + df, chess.square_rank(sq) + dr
        while 0 <= f < 8 and 0 <= r < 8:
            y = chess.square(f, r)
            if board.piece_at(y) is not None:
                out.append((x, y)); break
            f += df; r += dr
    return out


def neighbourhood(board: chess.Board, move: chess.Move, max_pieces: int = 12) -> list:
    """The frame a move is judged in: the mover and its destination, then by hops over the rules' own graph -- the
    pieces attacking or defending the destination, what those pieces block (a piece standing between a slider and
    another piece: both ends join), and the attackers and defenders of what is uncovered -- uncovered pieces first,
    then by worth, capped. Generic: attack, defend and between are the only words used."""
    ids = [move.from_square]
    if board.piece_at(move.to_square) is not None:
        ids.append(move.to_square)
    frontier = [(0, move.to_square)]
    for hop in range(3):
        nxt = []
        for _, sq in sorted(frontier, key=lambda t: (t[0], -VAL[board.piece_at(t[1]).piece_type] if board.piece_at(t[1]) else 0)):
            for a in list(board.attackers(chess.WHITE, sq)) + list(board.attackers(chess.BLACK, sq)):
                if a not in ids and len(ids) < max_pieces:
                    ids.append(a); nxt.append((1, a))
            if board.piece_at(sq) is not None:
                for x, y in _blocks(board, sq):
                    for z in (x, y):
                        if z not in ids and len(ids) < max_pieces:
                            ids.append(z); nxt.append((0, z))                # uncovered by the blocker's departure: first in the next hop
        frontier = nxt
        if len(ids) >= max_pieces or not frontier:
            break
    return ids[:max_pieces]


def frames_for(app, board: chess.Board, square: int, max_pieces: int = 8) -> list:
    """The pieces of every proved frame embedded in the position that holds this square, merged (largest first)."""
    ids = [square]
    for fset, c in known_frames(app, board):
        if square in fset:
            for x in sorted(fset):
                if x not in ids and len(ids) < max_pieces:
                    ids.append(x)
    return ids


def reach_worth(board: chess.Board, our: list) -> float:
    """An optimistic bound: the worth of every enemy piece our frame pieces attack now, or could attack after one move
    of theirs (from any square they can move to) -- everything the frame could conceivably win within the episode."""
    us = board.turn
    targets = set()
    for sq in our:
        p = board.piece_at(sq)
        if p is None or p.color != us:
            continue
        targets |= set(board.attacks(sq))
        for t in board.attacks(sq):
            if board.piece_at(t) is None:
                nb = board.copy(stack=False); nb.remove_piece_at(sq); nb.set_piece_at(t, p)
                targets |= set(nb.attacks(t))
    return sum(VAL[board.piece_at(t).piece_type] for t in targets if board.piece_at(t) is not None and board.piece_at(t).color != us and board.piece_at(t).piece_type != chess.KING)


def move_bound(app, board: chess.Board, move: chess.Move, frame: list) -> float:
    """Optimistic value of the move in its frame: the worth of every enemy piece in the frame (the episode counts only
    the frame's pieces, so nothing more can ever be won there), never below the largest gain the library claims."""
    us = board.turn
    bound = sum(VAL[board.piece_at(x).piece_type] for x in frame if board.piece_at(x) is not None and board.piece_at(x).color != us and board.piece_at(x).piece_type != chess.KING)
    # a promotion gains without capturing (audit 2026-09-24: h8=Q read 8 against a bound of 5): every pawn of ours in
    # the frame that can reach the last rank within the episode adds a queen's worth less its own
    for x in frame:
        p = board.piece_at(x)
        if p is not None and p.color == us and p.piece_type == chess.PAWN:
            rank = chess.square_rank(x) if us else 7 - chess.square_rank(x)
            if rank >= 4:
                bound += VAL[chess.QUEEN] - VAL[chess.PAWN]
    if board.piece_at(move.from_square) is not None and board.piece_at(move.from_square).piece_type == chess.PAWN and move.promotion:
        bound += VAL[chess.QUEEN] - VAL[chess.PAWN]
    claimed = 0.0
    try:
        res = app.apply(board, actor=move.from_square, only_action=move)
        for vl in res["verdicts"].values():
            for c, pred, fired, ids in vl:
                v = (c.value or {}).get("exception" if fired else "holds", 0.0) or 0.0
                claimed = max(claimed, v)
    except Exception:
        pass
    return max(bound, claimed)


def best_goal(app, board: chess.Board, prune: bool = True, depth: int = 0, max_open: int = 64) -> dict:
    """Best-first over the side to move's legal moves: bounds first, then the frame episode for each opened move; a
    move whose bound is at or below the best achieved value is pruned (when `prune`). Returns the choices with
    values, the best, and how many episodes were run."""
    dom = app.dom
    choices = []
    for m in board.legal_moves:
        fr = neighbourhood(board, m)
        choices.append(Choice(m, move_bound(app, board, m, fr), frame=fr))
    choices.sort(key=lambda c: -c.bound)
    alpha = -1e9; opened = 0
    for c in choices:
        if prune and alpha >= MATE:
            c.pruned = True; continue                                          # a mate is found: nothing material beats it
        if prune and c.bound <= alpha and alpha > -1e9 and not (board.gives_check(c.move) or board.is_capture(c.move)):
            c.pruned = True; continue                                          # a forcing move may still mate (a mate is worth more than any frame): only quiet moves are pruned by the material bound
        if opened >= max_open:
            c.pruned = True; continue
        o = outcome(dom, board, c.frame, c.move, depth, app=None, relative=True)    # over standing still in its own frame: a quiet move earns nothing for a tactic that was there anyway
        opened += 1
        if o is None:
            continue
        c.value = (1000.0 if o.value >= MATE else -1000.0 if o.value <= -MATE else o.value); c.outcome = o   # a mate is a mate: the material behind it is not a ranking (the audit's last disagreement was 1000 vs 1001)
        alpha = max(alpha, c.value)
    scored = [c for c in choices if c.value is not None]
    best = max(scored, key=lambda c: c.value) if scored else None
    return dict(best=best, choices=choices, opened=opened)


def threats(app, board: chess.Board, depth: int = 0) -> dict:
    """The opponent's best goal if we pass: never pruned. None when we are in check (no pass)."""
    b = board.copy(stack=False)
    if b.is_check():
        return dict(best=None, choices=[], opened=0)
    b.turn = not b.turn
    if not b.is_valid():
        return dict(best=None, choices=[], opened=0)
    return best_goal(app, b, prune=False, depth=depth)


def decide(app, board: chess.Board, k: int = 5, depth: int = 0) -> list:
    """Kaveh 2026-09-24: "we still have to evaluate all those frames for our opponent". For the k best moves by our
    achieved value (plus the k best by bound among the pruned), the opponent's best goal AFTER the move, unpruned;
    score = our gain minus theirs. Returns [(score, choice, their best value)] best first."""
    r = best_goal(app, board, prune=True, depth=depth)
    cands = sorted([c for c in r["choices"] if c.value is not None], key=lambda c: -c.value)[:k]
    out = []
    for c in cands:
        nb = board.copy(stack=False); nb.push(c.move)
        if nb.is_game_over():
            out.append((c.value, c, 0.0)); continue
        t = best_goal(app, nb, prune=False, depth=depth)                       # their side, never pruned
        theirs = t["best"].value if t["best"] else 0.0
        out.append((c.value - max(0.0, theirs), c, theirs))
    out.sort(key=lambda x: -x[0])
    return out


def audit(app, boards: list, depth: int = 0, log=print) -> dict:
    """The prune must not change the answer: run on and off, compare the best move and value; report time and the
    episodes saved. Any disagreement is a bug and is named."""
    same = 0; diff = []; t_on = t_off = 0.0; opened_on = opened_off = 0
    for b in boards:
        t = time.perf_counter(); on = best_goal(app, b, prune=True, depth=depth); t_on += time.perf_counter() - t
        t = time.perf_counter(); off = best_goal(app, b, prune=False, depth=depth); t_off += time.perf_counter() - t
        opened_on += on["opened"]; opened_off += off["opened"]
        vo = on["best"].value if on["best"] else None; vf = off["best"].value if off["best"] else None
        if vo == vf:
            same += 1
        else:
            diff.append((b.fen(), b.san(on["best"].move) if on["best"] else None, vo, b.san(off["best"].move) if off["best"] else None, vf))
    log(f"[goals/audit] {len(boards)} positions: same best value {same}/{len(boards)}; episodes opened {opened_on} with the prune vs {opened_off} without; time {t_on:.1f}s vs {t_off:.1f}s")
    for d in diff[:5]:
        log(f"  DIFFERS {d}")
    return dict(same=same, n=len(boards), diff=diff, opened_on=opened_on, opened_off=opened_off, t_on=t_on, t_off=t_off)


def main(argv=None):
    import argparse, json, random
    from pathlib import Path
    from .chains import domain_of
    from .lemma_apply import Applier, load_library
    from .lemma_gen import OUT
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["best", "audit"]); ap.add_argument("--fen"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--library", default=str(OUT / "library_games_frame6k.json")); ap.add_argument("--proved", default=str(OUT / "proved_games_frame6k.json"))
    a = ap.parse_args(argv)
    app = Applier(domain_of(Path(a.library)), load_library(Path(a.library), Path(a.proved)))
    if a.cmd == "best":
        b = chess.Board(a.fen); r = best_goal(app, b); t = threats(app, b)
        print(f"opened {r['opened']} of {len(r['choices'])} moves; best by value first, then the pruned")
        for c in sorted(r["choices"], key=lambda c: (c.value is None, -(c.value or 0), -c.bound))[:10]:
            print(f"  {b.san(c.move):7s} bound {c.bound:5.1f}  " + ("pruned" if c.pruned else f"value {c.value:+.1f}" if c.value is not None else "-") + f"  frame {[chess.square_name(x) for x in c.frame]}")
        print("their best if we pass:", (t["best"].value if t["best"] else None))
        print("decision (our gain minus their best after the move):")
        for score, c, theirs in decide(app, b)[:5]:
            print(f"  {b.san(c.move):7s} score {score:+.1f} = ours {c.value:+.1f} - theirs {theirs:+.1f}")
    else:
        rows = [json.loads(l)["fen"] for l in open("data/derived/chessbench_positions.jsonl")]
        random.Random(5).shuffle(rows)
        boards = [chess.Board(f) for f in rows[: a.n] if not chess.Board(f).is_game_over()]
        audit(app, boards)


if __name__ == "__main__":
    main()
