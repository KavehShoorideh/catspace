#!/usr/bin/env python
"""race_battery.py -- SubgoalFormer acceptance harness (docs/SUBGOALFORMER.md; Kaveh's
premove calculus as the graded object). KP-vs-K promotion races with the SYZYGY ORACLE:

  unstoppable   defender outside the square      -> TB says WIN  : p-hat(promote) must be ~1,
                                                                   premove-safe ON
  stoppable     defender inside the square       -> TB says DRAW : p-hat drops, premove OFF,
                                                                   worry mass on the blockade
  tempo         SAME placement, either side to   -> TB win/draw  : the sharpest minimal pair;
                move flips the TB verdict           split by turn  certificate must track turn

The promotion GOAL TOKEN is resolved empirically, never hand-mapped: the (head, code) that
most consistently flips when the pawn is force-promoted across the battery positions.

Graded: calibration (Brier + AUC of p-hat vs oracle) and worry report. An untrained
SubgoalFormer sits at chance -- the numbers are the training target, the harness is the
deliverable. Printed verdicts only.

    .venv/bin/python -m ...race_battery --ckpt <field.pt> --jqt <_jqt.pt>
"""
from __future__ import annotations

import argparse

import chess
import numpy as np
import torch


def gen_races(n_per=40, seed=0):
    """[(board, klass, tb_win)] verified against the engine's TB by the caller."""
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n_per * 3 and tries < 20000:
        tries += 1
        pf = int(rng.integers(0, 8))
        pr = int(rng.integers(3, 6))                      # pawn on rank 4-6 (0-idx)
        psq = chess.square(pf, pr)
        wk = chess.square(int(rng.integers(0, 8)), int(rng.integers(0, 2)))
        bk = chess.square(int(rng.integers(0, 8)), int(rng.integers(2, 8)))
        b = chess.Board(None)
        b.set_piece_at(psq, chess.Piece(chess.PAWN, chess.WHITE))
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        b.turn = chess.WHITE
        if not b.is_valid() or b.is_game_over():
            continue
        b2 = b.copy(); b2.turn = chess.BLACK
        if not b2.is_valid() or b2.is_game_over():
            continue
        out.append((b, b2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jqt", required=True)
    ap.add_argument("--leverage", default=None)
    ap.add_argument("--former", default=None, help="trained SubgoalFormer weights (else fresh)")
    ap.add_argument("--n-per", type=int, default=40)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
        GeoQuery, SubgoalFormer)
    eng = KittyChess(args.ckpt, args.device)
    gq = GeoQuery(eng, args.jqt, args.leverage, args.device)

    pairs = gen_races(args.n_per)
    cases = []                                            # (board, klass, tb_win 0/1)
    for bw, bb in pairs:
        ww, _ = eng.tb.wdl_dtz(bw)
        wb, _ = eng.tb.wdl_dtz(bb)
        if ww is None or wb is None:
            continue
        win_w, win_b = ww > 0, wb < 0                     # white-favorable, either mover
        if win_w and win_b:
            cases.append((bw, "unstoppable", 1))
        elif not win_w and not win_b:
            cases.append((bw, "stoppable", 0))
        elif win_w and not win_b:                         # tempo-decisive minimal pair
            cases.append((bw, "tempo-win", 1))
            cases.append((bb, "tempo-draw", 0))
    kl = {k: sum(1 for c in cases if c[1] == k) for k in
          ("unstoppable", "stoppable", "tempo-win", "tempo-draw")}
    print(f"[race] {len(cases)} oracle-labeled cases: {kl}")

    # resolve the promotion concept EMPIRICALLY: force-promote each pawn, diff the codes
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
    flips = np.zeros((gq.H, gq.C), np.int64)
    n_res = 0
    with torch.no_grad():
        for b, _k, _w in cases[:120]:
            psq = next(iter(b.pieces(chess.PAWN, chess.WHITE)))
            bq = b.copy()
            bq.remove_piece_at(psq)
            bq.set_piece_at(chess.square(chess.square_file(psq), 7),
                            chess.Piece(chess.QUEEN, chess.WHITE))
            if not bq.is_valid():
                continue
            rows = []
            for bd in (b, bq):
                tk, gl = tokenize(bd)
                rows.append((np.asarray(tk), np.asarray(gl)))
            tok = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.int64)).to(args.device)
            gl = torch.from_numpy(np.stack([r[1] for r in rows]).astype(np.float32)).to(args.device)
            phi = eng.net.backbone(tok, gl)
            _, ids = gq.jqt.target_codes(phi)
            ids = ids.cpu().numpy()
            for h in range(gq.H):
                if ids[0, h] != ids[1, h]:
                    flips[h, ids[1, h]] += 1
            n_res += 1
    h_g, c_g = np.unravel_index(flips.argmax(), flips.shape)
    print(f"[race] promotion concept resolved: h{h_g}/c{c_g} "
          f"(flips {flips.max()}/{n_res} = {flips.max()/max(n_res,1):.0%} consistent)")

    # DIRECT RULER PROBE (no planner): does P(activate promotion-concept) from the CDB
    # ruler alone separate TB-won from drawn races? This grades the JQT rulers themselves.
    import torch as _t
    ys_r, ps_r = [], []
    with _t.no_grad():
        for b, _k, tb_win in cases:
            z_us, _ = gq.state_embed(b)
            A = gq.jqt.anchors_for(_t.tensor([[int(h_g), int(c_g)]], device=gq.device)).float()
            dB = eng.net.dB(z_us[None], A)
            ps_r.append(float(_t.sigmoid(gq.jqt.activation_logit(dB))))
            ys_r.append(tb_win)
    ys_r, ps_r = np.array(ys_r), np.array(ps_r)
    o = np.argsort(ps_r); rk = np.empty(len(ps_r)); rk[o] = np.arange(len(ps_r))
    n1, n0 = int(ys_r.sum()), int((1 - ys_r).sum())
    auc_r = float((rk[ys_r == 1].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1))
    print(f"[race] RULER-ONLY  P(activate promo) AUC vs oracle: {auc_r:.3f}  "
          f"(mean p won {ps_r[ys_r==1].mean():.2f} vs drawn {ps_r[ys_r==0].mean():.2f})")

    former = SubgoalFormer(n_head=gq.H, n_code=gq.C)
    if args.former:
        former.load_state_dict(torch.load(args.former, map_location="cpu"))
        print(f"[race] loaded trained SubgoalFormer: {args.former}")
    else:
        print("[race] UNTRAINED SubgoalFormer -- numbers below are the chance baseline")

    hc = gq.candidates(k_lev=10, extra=[(int(h_g), int(c_g))])
    gi = 0                                                # goal token index (extras first)
    sides = torch.zeros(len(hc), dtype=torch.long)
    ps, ys, worries = [], [], []
    for b, klass, tb_win in cases:
        G, F = gq.geometry(b, hc)
        cert = former.certificate(torch.as_tensor(hc), sides, F, G, committed_idx=gi)
        ps.append(cert.p_hat); ys.append(tb_win)
        worries.append(float(cert.worry.max(initial=0.0)))
    ps, ys = np.array(ps), np.array(ys)
    brier = float(np.mean((ps - ys) ** 2))
    order = np.argsort(ps)
    ranks = np.empty(len(ps)); ranks[order] = np.arange(len(ps))
    n1, n0 = int(ys.sum()), int((1 - ys).sum())
    auc = float((ranks[ys == 1].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1))
    print(f"\n[race] VERDICT  brier {brier:.3f} (perfect 0, chance ~0.25)  AUC {auc:.3f} "
          f"(perfect 1, chance 0.5)")
    for k in ("unstoppable", "stoppable", "tempo-win", "tempo-draw"):
        m = np.array([c[1] == k for c in cases])
        if m.any():
            print(f"  {k:12s} mean p-hat {ps[m].mean():.2f}  mean max-worry "
                  f"{np.array(worries)[m].mean():.3f}  (n={m.sum()})")
    # the premove predicate on the extremes
    safe = np.array([p >= 0.97 and w <= 0.02 for p, w in zip(ps, worries)])
    if safe.any():
        print(f"  premove-safe flagged: {safe.sum()} cases, of which TB-won {ys[safe].mean():.0%} "
              f"(must be 100% when trained)")


if __name__ == "__main__":
    main()
