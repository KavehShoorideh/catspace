#!/usr/bin/env python
"""experiments/build_jepa_corpus.py -- T1 corpus for the anchored JEPA (paper §3.2-3.3):
second pass over the [%eval]-annotated games emitting the three training streams:

  transitions : tokenized (s_t, a_t, s_{t+1}) sampled ~4/game -- feeds L_dyn AND the
                L_dest along-game bootstrap (CE(d(s_t), sg d(s_{t+1})))
  boundary    : the first position with <=--max-men pieces AND a successful Syzygy
                probe: tokens + material class (mat_sig) + exact W/D/L (white POV)
                -- the terminal clamp of L_dest. Games never reaching the boundary
                contribute bootstrap only (synthetic behavioural tails = later,
                per the paper's w_i-weighted continuation term -- deliberate v1 cut).
  contexts    : the mined checkpoint contexts (--checkpoints npz), tokenized -- feeds
                the aggregate any-event hazard (per-atom keys arrive at T2-T4).

Class vocab: mat_sig counts over boundary positions -> top --classes kept, rest OTHER.
Table-2 stats printed: boundary-reached-naturally %%, class histogram head.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import chess.pgn
import chess.syzygy
import numpy as np
import zstandard

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.encoder.jepa import tokenize, move_ids                    # noqa: E402
from catspace.endgame.material import mat_sig                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="data/raw/lichess_db_standard_rated_2019-01.pgn.zst")
    ap.add_argument("--checkpoints", default="data/derived/checkpoints/checkpoints_v1_full.npz")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--max-men", type=int, default=6)
    ap.add_argument("--per-game", type=int, default=4)
    ap.add_argument("--classes", type=int, default=150)
    ap.add_argument("--max-eval-games", type=int, default=1000000000)
    ap.add_argument("--out", default="data/derived/checkpoints/jepa_t1_corpus.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    tb = chess.syzygy.open_tablebase(args.syzygy)

    tr_s, tr_g, tr_a, tr_s1, tr_g1 = [], [], [], [], []
    bd_tok, bd_glob, bd_sig, bd_wdl = [], [], [], []
    n_seen = n_eval = n_boundary = 0
    buf: list[str] = []; in_moves = False; has_eval = False
    with open(args.pgn, "rb") as f:
        stream = io.TextIOWrapper(
            zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(f),
            encoding="utf-8", errors="ignore")
        for line in stream:
            buf.append(line)
            if line.startswith("1. ") or (in_moves and line.strip()):
                in_moves = True
                has_eval = has_eval or "%eval" in line
            if in_moves and not line.strip():
                n_seen += 1
                if has_eval:
                    game = chess.pgn.read_game(io.StringIO("".join(buf)))
                    if game is not None and game.end().ply() >= 16:
                        n_eval += 1
                        board = game.board()
                        moves = list(game.mainline_moves())
                        picks = set(rng.choice(len(moves) - 1,
                                               min(args.per_game, len(moves) - 1),
                                               replace=False))
                        hit_boundary = False
                        for i, mv in enumerate(moves):
                            if i in picks:
                                t, g = tokenize(board)
                                b2 = board.copy(stack=False); b2.push(mv)
                                t1, g1 = tokenize(b2)
                                tr_s.append(t); tr_g.append(g)
                                tr_a.append(move_ids(mv)); tr_s1.append(t1); tr_g1.append(g1)
                            if (not hit_boundary
                                    and len(board.piece_map()) <= args.max_men):
                                try:
                                    wdl = tb.probe_wdl(board)          # side-to-move POV
                                    stm_white = board.turn
                                    w = 0 if (wdl > 0) == stm_white and wdl != 0 else \
                                        (1 if wdl == 0 else 2)         # white POV W/D/L
                                    tt, gg = tokenize(board)
                                    bd_tok.append(tt); bd_glob.append(gg)
                                    bd_sig.append(mat_sig(board)); bd_wdl.append(w)
                                    hit_boundary = True; n_boundary += 1
                                except (KeyError, chess.syzygy.MissingTableError):
                                    pass
                            board.push(mv)
                buf, in_moves, has_eval = [], False, False
                if n_seen % 400_000 == 0:
                    print(f"  scanned {n_seen:,} | used {n_eval:,} | trans {len(tr_s):,} | "
                          f"boundary {n_boundary:,} [{time.time()-t0:.0f}s]", flush=True)
                if n_eval >= args.max_eval_games:
                    break
    print(f"TABLE2 games used {n_eval:,} | transitions {len(tr_s):,} | boundary reached "
          f"naturally (probeable <= {args.max_men} men) {n_boundary/max(n_eval,1):.1%}",
          flush=True)

    # class vocab: top-N by frequency, rest -> OTHER (=N)
    sigs, counts = np.unique(np.array(bd_sig), return_counts=True)
    keep = sigs[np.argsort(-counts)[:args.classes]]
    vocab = {s: i for i, s in enumerate(keep)}
    cls = np.array([vocab.get(s, args.classes) for s in bd_sig], np.int32)
    print(f"AUDIT classes: {len(sigs)} distinct sigs -> {len(keep)} kept + OTHER "
          f"({np.mean(cls == args.classes):.1%} OTHER) | top: "
          + " ".join(f"{s}:{c}" for s, c in
                     sorted(zip(sigs, counts), key=lambda x: -x[1])[:5]))
    wdl_arr = np.array(bd_wdl, np.int8)
    print(f"AUDIT boundary WDL (white POV): W {np.mean(wdl_arr == 0):.1%} "
          f"D {np.mean(wdl_arr == 1):.1%} L {np.mean(wdl_arr == 2):.1%}")

    # tokenize the mined checkpoint contexts (hazard stream)
    ck = dict(np.load(args.checkpoints, allow_pickle=True))
    print(f"[tokenize] {len(ck['cx_fen']):,} contexts", flush=True)
    cx_tok = np.zeros((len(ck["cx_fen"]), 64), np.uint8)
    cx_glob = np.zeros((len(ck["cx_fen"]), 6), np.uint8)
    for i, fen in enumerate(ck["cx_fen"]):
        t, g = tokenize(chess.Board(fen))
        cx_tok[i] = t; cx_glob[i] = g
        if i % 500_000 == 0:
            print(f"  {i:,}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tr_tok=np.stack(tr_s), tr_glob=np.stack(tr_g),
        tr_act=np.array(tr_a, np.int16),
        tr_tok1=np.stack(tr_s1), tr_glob1=np.stack(tr_g1),
        bd_tok=np.stack(bd_tok) if bd_tok else np.zeros((0, 64), np.uint8),
        bd_glob=np.stack(bd_glob) if bd_glob else np.zeros((0, 6), np.uint8),
        bd_class=cls, bd_wdl=wdl_arr,
        class_vocab=np.array(list(keep)),
        cx_tok=cx_tok, cx_glob=cx_glob,
        cx_gap_dec=ck["cx_gap_dec"], cx_end_dec=ck["cx_end_dec"],
        cx_gid=ck["cx_gid"], cx_elo_victim=ck["cx_elo_victim"],
        cx_elo_opp=ck["cx_elo_opp"],
        meta_max_men=args.max_men, meta_classes=args.classes)
    print(f"wrote {out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
