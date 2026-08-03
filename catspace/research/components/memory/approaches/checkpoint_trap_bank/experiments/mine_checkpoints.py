#!/usr/bin/env python
"""catspace/research/components/memory/approaches/checkpoint_trap_bank/experiments/mine_checkpoints.py -- Stage 1 of the anchored-JEPA plan (Kaveh's draft,
§3.2 / Figure 2): mine CHECKPOINTS from [%eval]-annotated lichess games.

A checkpoint is the position at which the victim still had the choice -- immediately
preceding a win-probability swing Delta > tau ATTRIBUTED TO THEIR MOVE, with the
pre-swing eval inside a near-equal window (conversion-of-advantage events are
excluded: they teach the scoreboard, not the structure). Every earlier position of
the same game is a CONTEXT example for that target, with the gap measured in VICTIM
DECISIONS; games without a checkpoint contribute CENSORED contexts (exposure = victim
decisions to game end), never negatives.

Corpus reality: ~6.9%% of 2019-01 games carry [%eval] (measured) -> ~690k annotated
games in the month. A raw-text '%%eval' pre-filter keeps parsing off the other 93%%.

Output npz:
  checkpoints: fen, gid, ply, victim_white, elo_victim, elo_opp, delta, w_pre
  contexts   : fen, gid, ply, elo_victim, elo_opp, gap_dec (victim decisions to the
               checkpoint; -1 = censored), ckpt_row (-1 = censored), end_dec
               (victim decisions to game end = censoring exposure)
Stats printed = the paper's Table 2 fields (games scanned / annotated / checkpoints
post-filter / per-game rate / censoring rate).
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import chess.pgn
import numpy as np
import zstandard
from catspace.io import paths


def winprob(cp: float) -> float:
    """lichess win-probability model, white POV."""
    return 1.0 / (1.0 + np.exp(-0.00368208 * cp))


GAPS = (1, 2, 3, 5, 8, 13, 21, 34)          # context sampling: victim decisions back


def mine_game(game, tau, near, cap, rng):
    """-> (checkpoints, contexts) lists for one annotated game."""
    try:
        elo_w = int(game.headers.get("WhiteElo", 0)); elo_b = int(game.headers.get("BlackElo", 0))
    except ValueError:
        return [], []
    if not elo_w or not elo_b:
        return [], []
    board = game.board()
    rows = []                                # (fen, ply, mover_white, w_before, w_after)
    ev_prev = 0.0                            # start position ~equal
    for node in game.mainline():
        sc = node.eval()
        if sc is None:
            return [], []                    # partially annotated: skip game
        ev = float(sc.white().score(mate_score=3200))
        rows.append((board.fen(), board.ply(), board.turn, winprob(ev_prev), winprob(ev)))
        ev_prev = ev
        board.push(node.move)
    if len(rows) < 16:
        return [], []
    cps, ctxs = [], []
    for i, (fen, ply, white_mv, w0, w1) in enumerate(rows):
        wv0 = w0 if white_mv else 1.0 - w0   # victim(=mover)-POV before/after their move
        wv1 = w1 if white_mv else 1.0 - w1
        if (wv0 - wv1) > tau and abs(wv0 - 0.5) <= near:
            cps.append(dict(fen=fen, ply=ply, victim_white=white_mv,
                            elo_victim=elo_w if white_mv else elo_b,
                            elo_opp=elo_b if white_mv else elo_w,
                            delta=float(wv0 - wv1), w_pre=float(wv0), row=i))
            if len(cps) >= cap:
                break
    # contexts: log-spaced victim-decision gaps back from each checkpoint
    for c in cps:
        vd = [i for i, r in enumerate(rows) if r[2] == c["victim_white"] and i < c["row"]]
        for g in GAPS:
            if g <= len(vd):
                i = vd[-g]
                ctxs.append(dict(fen=rows[i][0], ply=rows[i][1], elo_victim=c["elo_victim"],
                                 elo_opp=c["elo_opp"], gap_dec=g, ckpt=c,
                                 end_dec=sum(r[2] == c["victim_white"] for r in rows[i:])))
    if not cps and rng.random() < 0.34:      # censored contexts (balanced subsample)
        for victim_white in (True, False):
            vd = [i for i, r in enumerate(rows) if r[2] == victim_white]
            if len(vd) > 10:
                i = vd[rng.integers(4, len(vd) - 4)]
                ctxs.append(dict(fen=rows[i][0], ply=rows[i][1],
                                 elo_victim=elo_w if victim_white else elo_b,
                                 elo_opp=elo_b if victim_white else elo_w,
                                 gap_dec=-1, ckpt=None,
                                 end_dec=sum(r[2] == victim_white for r in rows[i:])))
    return cps, ctxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default=paths.raw("lichess_db_standard_rated_2019-01.pgn.zst"))
    ap.add_argument("--tau", type=float, default=0.2, help="swing threshold, win-prob units")
    ap.add_argument("--near", type=float, default=0.15,
                    help="near-equal window: |W_victim - 0.5| <= near at the pre-swing position")
    ap.add_argument("--cap", type=int, default=2, help="checkpoints per game cap")
    ap.add_argument("--max-eval-games", type=int, default=200_000)
    ap.add_argument("--out", default=paths.derived("checkpoints/checkpoints_v0.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)

    n_seen = n_eval = 0
    all_cps, all_ctx = [], []
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
            if in_moves and not line.strip():                # game block complete
                n_seen += 1
                if has_eval:
                    n_eval += 1
                    game = chess.pgn.read_game(io.StringIO("".join(buf)))
                    if game is not None:
                        cps, ctxs = mine_game(game, args.tau, args.near, args.cap, rng)
                        for c in cps:
                            c["gid"] = n_seen
                        for x in ctxs:
                            x["gid"] = n_seen
                        base = len(all_cps)
                        all_cps.extend(cps)
                        for x in ctxs:
                            x["ckpt_row"] = base + cps.index(x["ckpt"]) if x["ckpt"] else -1
                        all_ctx.extend(ctxs)
                buf, in_moves, has_eval = [], False, False
                if n_seen % 200_000 == 0:
                    print(f"  scanned {n_seen:,} | annotated {n_eval:,} | checkpoints "
                          f"{len(all_cps):,} | contexts {len(all_ctx):,} [{time.time()-t0:.0f}s]",
                          flush=True)
                if n_eval >= args.max_eval_games:
                    break
    ncens = sum(1 for x in all_ctx if x["gap_dec"] < 0)
    print(f"TABLE2 games scanned {n_seen:,} | annotated {n_eval:,} ({n_eval/max(n_seen,1):.1%}) | "
          f"checkpoints {len(all_cps):,} ({len(all_cps)/max(n_eval,1):.2f}/annotated game) | "
          f"contexts {len(all_ctx):,} (censored {ncens/max(len(all_ctx),1):.1%})", flush=True)
    if all_cps:
        d = np.array([c["delta"] for c in all_cps])
        e = np.array([c["elo_victim"] for c in all_cps])
        print(f"AUDIT delta median {np.median(d):.3f} p90 {np.percentile(d, 90):.3f} | "
              f"victim Elo median {np.median(e):.0f} | victim-white "
              f"{np.mean([c['victim_white'] for c in all_cps]):.1%}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ck_fen=np.array([c["fen"] for c in all_cps]),
        ck_gid=np.array([c["gid"] for c in all_cps], np.int64),
        ck_ply=np.array([c["ply"] for c in all_cps], np.int32),
        ck_victim_white=np.array([c["victim_white"] for c in all_cps], bool),
        ck_elo_victim=np.array([c["elo_victim"] for c in all_cps], np.int32),
        ck_elo_opp=np.array([c["elo_opp"] for c in all_cps], np.int32),
        ck_delta=np.array([c["delta"] for c in all_cps], np.float32),
        ck_w_pre=np.array([c["w_pre"] for c in all_cps], np.float32),
        cx_fen=np.array([x["fen"] for x in all_ctx]),
        cx_gid=np.array([x["gid"] for x in all_ctx], np.int64),
        cx_ply=np.array([x["ply"] for x in all_ctx], np.int32),
        cx_elo_victim=np.array([x["elo_victim"] for x in all_ctx], np.int32),
        cx_elo_opp=np.array([x["elo_opp"] for x in all_ctx], np.int32),
        cx_gap_dec=np.array([x["gap_dec"] for x in all_ctx], np.int32),
        cx_ckpt_row=np.array([x["ckpt_row"] for x in all_ctx], np.int64),
        cx_end_dec=np.array([x["end_dec"] for x in all_ctx], np.int32),
        meta_tau=args.tau, meta_near=args.near, meta_cap=args.cap)
    print(f"wrote {out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
