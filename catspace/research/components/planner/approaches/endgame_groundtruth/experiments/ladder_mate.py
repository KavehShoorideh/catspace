#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/ladder_mate.py -- the two-rook ladder ("lawnmower") mate: KRRvK, the
project's cornering-the-king concept in its purest form and the EXECUTE phase of the
long/short engine (DECISIONS.md sec 3). Two rooks + king vs a lone king is a shallow,
fully forced win, but from a CENTRAL black king the mate is ~12-16 plies away and pure
MCTS+mate_stop has NO signal that prefers driving the king to the edge -- it wanders.
This measures, honestly, whether the bare search delivers the mate, and is the harness
we plug a mate-distance value into (the separate DTM head) if it does not.

White = MCTS (pure, uniform prior, constant value, mate_stop). Black = tablebase-optimal
defense (longest resistance), so a reported mate is a mate against best play. Reports mate
rate + median plies-to-mate by node budget, on random KRRvK with a central black king (the
hard case) unless --anywhere.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, tb_best_move
from catspace.io import paths

CENTER = [sq for sq in range(64) if 2 <= chess.square_file(sq) <= 5 and 2 <= chess.square_rank(sq) <= 5]
VALUE_C = 8.0   # squash center: value = tanh((C - dtm)/C), closer-to-mate reads higher (white-POV)


def make_tb_value(tb):
    """ORACLE ceiling: white-POV mate-distance value from tablebase DTZ (KRRvK is
    pawnless & capture-free under optimal play, so |DTZ| ~ plies-to-mate). Cached."""
    cache = {}

    def value_fn(boards):
        out = []
        for b in boards:
            k = b._transposition_key()
            if k not in cache:
                w, d = tb.wdl_dtz(b)
                cache[k] = 1.0 if w is None else float(np.tanh((VALUE_C - abs(d if d is not None else 30)) / VALUE_C))
            out.append(cache[k])
        return np.array(out, dtype=float)
    return value_fn


# MOVED to catspace/diagnostics.py (canonical; DIAGNOSTIC-ONLY rule documented there).
from catspace.research.tools.chess_specific.diagnostics import escape_volume  # noqa: F401


def make_constraint_value():
    """White-POV value = shrink the black king's escape volume (the cornering concept).
    No tablebase, no learned net -- the exact concept, as the search's guiding value."""
    def value_fn(boards):
        return np.array([float(np.tanh((VALUE_C - escape_volume(b)) / VALUE_C)) for b in boards])
    return value_fn


def make_dtm_value(ckpt, device="cpu"):
    """LEARNED head: white-POV mate-distance value from the separate DTM CNN
    (train_dtm_cnn.py). value = tanh((C - dtm_pred)/C)."""
    import torch
    from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.train_dtm_cnn import DTMNet
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
    dev = pick_device(device)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = DTMNet(c=st["c"]).to(dev); net.load_state_dict(st["state"]); net.eval()
    scale = st.get("scale", 20.0)

    def value_fn(boards):
        pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
        with torch.no_grad():
            pred = net(torch.from_numpy(feature_planes(pk, mt)).to(dev)).cpu().numpy() * scale
        return np.tanh((VALUE_C - pred) / VALUE_C)
    return value_fn


def random_krrvk(rng: np.random.Generator, central: bool = True, max_tries: int = 500):
    """Random legal K+R+R vs k, White to move, not already over, black not in check."""
    for _ in range(max_tries):
        bk = int(rng.choice(CENTER)) if central else int(rng.integers(64))
        occ = {bk}
        wk = int(rng.integers(64))
        if wk in occ or chess.square_distance(wk, bk) < 2:
            continue
        occ.add(wk)
        rs = [int(x) for x in rng.choice([s for s in range(64) if s not in occ], size=2, replace=False)]
        b = chess.Board(None)
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(rs[0], chess.Piece(chess.ROOK, chess.WHITE))
        b.set_piece_at(rs[1], chess.Piece(chess.ROOK, chess.WHITE))
        b.turn = chess.WHITE
        if b.is_valid() and not b.is_game_over():
            return b
    return None


_EVAL_CACHE = {}                    # shared across moves/games; transposition evals are free

def white_mcts(board, nodes, value_fn=None, policy_fn=None):
    reach = (lambda bs: np.zeros(len(bs), dtype=float))
    # MCTS only consults value_fn on the AZ path, which needs a policy_fn too.
    # Pure search => neither (reach_fn=0). Value-guided => uniform prior + value_fn.
    if value_fn is not None and policy_fn is None:
        policy_fn = lambda b: {mv: 1.0 / max(1, b.legal_moves.count()) for mv in b.legal_moves}
    if len(_EVAL_CACHE) > 400_000:
        _EVAL_CACHE.clear()
    m = MCTS(reach, max_nodes=nodes, mate_stop=True, pw_c=1.5, root_min_visits=10,
             value_fn=value_fn, policy_fn=policy_fn,
             eval_cache=_EVAL_CACHE if value_fn is not None else None, batch_leaves=8)
    root = m.run(board)
    # White to move: prefer most-visited, break ties toward the best (fastest-mate) value
    best = max(root.children, key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)))
    return best.move, m.evals_used         # evals_used = actual search nodes spent (<= cap)


def play_out(start, tb, nodes, max_plies=60, value_fn=None, policy_fn=None, white="mcts"):
    b = start.copy(stack=False)
    plies = 0; search_nodes = 0
    for _ in range(max_plies):
        if b.is_game_over(claim_draw=True):
            break
        if b.turn == chess.WHITE:
            if white == "tb":
                # TRIVIAL solved-endgame finisher: tablebase lookup, ZERO search nodes.
                b.push(tb_best_move(b, tb))
            else:
                mv, ev = white_mcts(b, nodes, value_fn, policy_fn)
                search_nodes += ev; b.push(mv)
        else:
            b.push(tb_best_move(b, tb))
        plies += 1
    out = b.outcome(claim_draw=True)
    mated = bool(out and out.winner == chess.WHITE)
    if mated:
        kind = "mate"
    elif out is not None:
        kind = "draw"          # stalemate / repetition / 50-move / insufficient
    else:
        kind = "timeout"       # hit max_plies still winning but not converted
    return mated, plies, kind, search_nodes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--nodes", type=int, nargs="+", default=[400, 800, 1600])
    ap.add_argument("--max-plies", type=int, default=60)
    ap.add_argument("--anywhere", action="store_true", help="black king anywhere (not just central)")
    ap.add_argument("--value", choices=["none", "tb", "dtm", "constraint"], default="none",
                    help="MCTS leaf value: none=pure search, tb=tablebase oracle (ceiling), "
                         "dtm=learned DTM head, constraint=king escape-volume (the cornering concept)")
    ap.add_argument("--dtm-ckpt", default=paths.sep("dtm_cnn.pt"))
    ap.add_argument("--white", choices=["mcts", "tb"], default="mcts",
                    help="tb = trivial tablebase-optimal finisher (solved endgame, 100%%)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); tb = TB(str(paths.syzygy_dir()))
    rng = np.random.default_rng(args.seed)
    starts = [random_krrvk(rng, central=not args.anywhere) for _ in range(args.n)]
    starts = [s for s in starts if s is not None]
    value_fn = {"none": None, "tb": make_tb_value(tb),
                "dtm": make_dtm_value(args.dtm_ckpt) if args.value == "dtm" else None,
                "constraint": make_constraint_value()}[args.value]
    print(f"[ladder] {len(starts)} KRRvK starts ({'anywhere' if args.anywhere else 'central king'})  "
          f"value={args.value}", flush=True)
    node_list = [0] if args.white == "tb" else args.nodes
    for nodes in node_list:
        res = [play_out(s, tb, nodes, args.max_plies, value_fn=value_fn, white=args.white) for s in starts]
        mates = [p for m, p, _, _ in res if m]
        snodes = [sn for m, _, _, sn in res if m]
        rate = len(mates) / len(res)
        med = float(np.median(mates)) if mates else float("nan")
        med_sn = float(np.median(snodes)) if snodes else 0.0
        nd = sum(k == "draw" for _, _, k, _ in res); nt = sum(k == "timeout" for _, _, k, _ in res)
        tag = "tb-move" if args.white == "tb" else f"value={args.value:4s} nodes={nodes:5d}"
        print(f"VERDICT LADDER_MATE {tag}  mate_rate={rate:.2f} ({len(mates)}/{len(res)})  "
              f"median_plies={med:.0f}  search_nodes_to_mate median={med_sn:,.0f}  "
              f"draws={nd} timeouts={nt}  [{time.time()-t0:.0f}s]", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
