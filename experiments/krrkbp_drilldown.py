#!/usr/bin/env python
"""
experiments/krrkbp_drilldown.py — WHY does the planner fail to convert KRRvKBP?

Plays the FB policy (White) from a chosen KRRvKBP position against Stockfish
(Black), and at every White move compares what the model DID to what the
tablebase says is OPTIMAL. Surfaces the mistakes concretely:
  - per White ply: model's move, the tablebase DTZ before/after (win-distance),
    whether the move PRESERVED the win / how much progress it threw away, and
    the model's own reach ranking of its top moves vs the tablebase-best move.
  - flags the WORST mistakes (biggest DTZ jump, or win->draw throws) and, for
    the single worst, dumps the full candidate table: for each legal move, the
    model's reach score AND the tablebase truth -- so we can SEE what the model
    ranked high that was actually bad, and what it ranked low that was the win.

This answers "what isn't it seeing": if the model's reach consistently ranks a
DTZ-increasing (progress-losing) move above the DTZ-minimizing one, that gap is
the blindness. Move labels (SAN) + whether a rook landed on a bishop-attackable
square are printed to make the pattern legible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.diagnostic_krrkbp import load_fixed_set
from catspace.io.paths import derived_dir


def tb_dtz(board, tb):
    try:
        return tb.probe_dtz(board), tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError):
        return None, None


def white_move_analysis(board, tb, pol, rng):
    """For White to move in a won position: model's move + reach ranking, the
    tablebase-optimal move, and how much win-distance the model's move cost."""
    _, cands = _reach_all(pol, board, rng)          # [(move, reach)] all legal, sorted desc
    model_move = cands[0][0]

    # tablebase truth per move: resulting position's DTZ from opponent POV
    # (negative when opp is losing); we want the move that keeps opp most-lost
    # (smallest |dtz|, i.e. fastest mate) while preserving the win.
    truth = []
    for m in board.legal_moves:
        b2 = board.copy(stack=False); b2.push(m)
        if b2.is_checkmate():
            truth.append((m, -10_000, 2)); continue
        dtz, wdl = tb_dtz(b2, tb)
        truth.append((m, (dtz if dtz is not None else 0), (wdl if wdl is not None else 0)))
    # a move PRESERVES the win iff opponent is now losing (wdl < 0)
    winning = [(m, dtz) for m, dtz, wdl in truth if wdl < 0]
    if winning:
        best_move, best_dtz = min(winning, key=lambda t: abs(t[1]))
    else:
        best_move, best_dtz = None, None
    model_dtz = dict((m, dtz) for m, dtz, _ in truth)[model_move]
    model_wdl = dict((m, wdl) for m, _, wdl in truth)[model_move]
    model_preserves = model_wdl < 0 or board.copy(stack=False).__class__  # wdl<0 or mate handled
    return dict(cands=cands, model_move=model_move, model_dtz=model_dtz,
                model_wdl=model_wdl, best_move=best_move, best_dtz=best_dtz, truth=truth)


def _reach_all(pol, board, rng):
    """(best_move, [(move, reach)...] sorted desc) using the policy's own reach
    over the immediate children (1-ply model view -- what it 'sees' shallowly)."""
    moves = list(board.legal_moves)
    succ = [board.copy(stack=False) for _ in moves]
    for b, m in zip(succ, moves):
        b.push(m)
    reach = pol._reach_batch(succ)
    order = np.argsort(-reach)
    cands = [(moves[i], float(reach[i])) for i in order]
    return cands[0][0], cands


def rook_on_bishop_attackable(board_after, bishop_color=chess.BLACK):
    """Did White just put a rook where the black bishop attacks it? (the concept
    the planner is supposed to learn: keep rooks off the bishop's diagonals.)"""
    bsq = [s for s, p in board_after.piece_map().items()
           if p.piece_type == chess.BISHOP and p.color == bishop_color]
    if not bsq:
        return False
    attacked = board_after.attacks(bsq[0])
    return any(p.piece_type == chess.ROOK and p.color == chess.WHITE and s in attacked
               for s, p in board_after.piece_map().items())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_fixed_set_n60.json")
    ap.add_argument("--index", type=int, default=None, help="position index; default: scan for a drawn one")
    ap.add_argument("--max-nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBSearchPolicy
    from catspace.realboard import play_board_game
    from catspace.uci import UCIBoardPolicy

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    pol = FBSearchPolicy(fb, payload["zgoals"]["MATE_W"], max_nodes=args.max_nodes,
                         beam=args.beam, device=device)
    tb = chess.syzygy.open_tablebase(args.syzygy_dir)
    positions = load_fixed_set(args.fixed_set)

    idx = args.index if args.index is not None else 0
    start = positions[idx]
    print(f"position {idx}: {start.fen()}")
    dtz0, _ = tb_dtz(start, tb)
    print(f"tablebase: winning for White, DTZ={dtz0}\n")

    # play it out vs Stockfish, analyzing every White move
    board = start.copy(stack=False)
    rng = np.random.default_rng([args.seed, idx])
    opp = UCIBoardPolicy(skill=0, movetime=0.02)
    mistakes = []
    with opp:
        ply = 0
        while ply < args.max_plies and not board.is_game_over(claim_draw=True):
            if board.turn == chess.WHITE:
                a = white_move_analysis(board, tb, pol, rng)
                b_after = board.copy(stack=False); b_after.push(a["model_move"])
                threw = (a["model_wdl"] >= 0) and not b_after.is_checkmate()
                # progress lost = how much bigger the model's resulting |DTZ| is
                # vs the best move's |DTZ| (both from opp POV; smaller=closer to mate)
                lost = (abs(a["model_dtz"]) - abs(a["best_dtz"])) if a["best_dtz"] is not None else 0
                san = board.san(a["model_move"])
                best_san = board.san(a["best_move"]) if a["best_move"] else "-"
                rook_hang = rook_on_bishop_attackable(b_after)
                rank_of_best = next((i for i, (m, _) in enumerate(a["cands"])
                                     if m == a["best_move"]), -1)
                mistakes.append(dict(ply=ply, san=san, best_san=best_san, threw=threw,
                                     lost=lost, rook_hang=rook_hang,
                                     best_rank=rank_of_best, analysis=a, fen=board.fen()))
                board.push(a["model_move"])
            else:
                board.push(opp.move(board, rng))
            ply += 1

    print(f"game result: {board.outcome(claim_draw=True).result() if board.outcome(claim_draw=True) else '*'} "
          f"({board.outcome(claim_draw=True).termination.name if board.outcome(claim_draw=True) else 'PLY_CAP'}) "
          f"after {ply} plies\n")

    # rank the mistakes: win-throws first, then biggest progress loss
    ranked = sorted(mistakes, key=lambda m: (not m["threw"], -m["lost"]))
    print("=== worst White decisions (win-throws, then biggest DTZ progress lost) ===")
    for m in ranked[:5]:
        flag = "THREW WIN" if m["threw"] else f"lost {m['lost']:+d} DTZ"
        print(f"  ply {m['ply']:3d}: played {m['san']:6s} (best {m['best_san']:6s}, "
              f"best ranked #{m['best_rank']+1} by model reach)  {flag}"
              f"{'  [rook -> bishop-attackable square]' if m['rook_hang'] else ''}")

    # dump the full candidate table for the single worst decision
    worst = ranked[0]
    print(f"\n=== WORST decision (ply {worst['ply']}): {worst['fen']} ===")
    a = worst["analysis"]
    dtz_by_move = {m: dtz for m, dtz, _ in a["truth"]}
    wdl_by_move = {m: wdl for m, _, wdl in a["truth"]}
    tmp = chess.Board(worst["fen"])
    print("  model_rank  move    reach       tb_wdl  tb_dtz   verdict")
    for r, (mv, reach) in enumerate(a["cands"][:8]):
        wdl = wdl_by_move[mv]; dtz = dtz_by_move[mv]
        verdict = "MATE" if wdl == 2 and False else ("wins" if wdl < 0 else ("DRAW-throw" if wdl == 0 else "loses"))
        star = " <- model plays" if r == 0 else (" <- tablebase-best" if mv == a["best_move"] else "")
        print(f"  #{r+1:<9d} {tmp.san(mv):6s} {reach:+.4f}   {wdl:+d}     {dtz:+5d}   {verdict}{star}")
    tb.close()


if __name__ == "__main__":
    main()
