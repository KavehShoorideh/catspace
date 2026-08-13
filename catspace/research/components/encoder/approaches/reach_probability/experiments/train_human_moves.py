#!/usr/bin/env python
"""train_human_moves.py -- THE HUMAN PREDICTION LAYER (Kaveh 2026-08-13: "let's build the
human prediction layer").

P(move | position, mover-Elo) trained on the lichess corpus -- the Maia-role model, on OUR
frozen trunk. This replaces the engine's own E-softmax as the expectation over a human's
moves everywhere the play stack needs one:
    surprise ruler   bits = -log2 P_human(move) (excess-vs-entropy unchanged) -- the engine
                     stops "expecting Nh3": expectation means HUMANS, not its own taste
    ponder / prep    branch priors over the opponent's replies
    per-player       Elo bucket fitted per player by NLL on their logged moves; the
                     annealed temperature (stranger -> settled) applies on top

Architecture: pointwise legal-move scorer (the move-head pattern):
    ctx  = GELU(Linear([phi(board) ; elo_emb(bucket)]))           phi FROZEN
    logit(m) = MLP([ctx ; e_from(m) + e_to(m) + e_promo(m)])      softmax over LEGAL moves
Metrics: top-1 accuracy + NLL in BITS/MOVE (the surprise ruler's native unit).

    .venv/bin/python -m ...train_human_moves --ckpt <field.pt> --games 20000
"""
from __future__ import annotations

import argparse
import json
import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ELO_LO, ELO_HI, ELO_W = 800, 2400, 100          # buckets: <800, 800-899, ..., >=2400


def elo_bucket(elo):
    return int(np.clip((elo - ELO_LO) // ELO_W + 1, 0, (ELO_HI - ELO_LO) // ELO_W + 1))


N_BUCKET = (ELO_HI - ELO_LO) // ELO_W + 2


N_CLK = 4        # [log1p(self)/7, log1p(opp)/7, frac-of-base, low-time<30s]


def clk_feats(clk_self, clk_opp, base):
    """v2 (Kaveh 2026-08-13: 'as the time runs out, the move quality becomes lower'):
    the mover's TIME STATE. NaN clocks (untimed / missing) -> ample-time defaults."""
    import math as _m
    if not (clk_self == clk_self):                    # NaN
        clk_self = base
    if not (clk_opp == clk_opp):
        clk_opp = base
    return [(_m.log1p(max(clk_self, 0.0))) / 7.0, (_m.log1p(max(clk_opp, 0.0))) / 7.0,
            min(clk_self / max(base, 1.0), 1.5), 1.0 if clk_self < 30.0 else 0.0]


class HumanMoves(nn.Module):
    """v2: phi PLUS the quantized concept codes PLUS the clock state (Kaveh 2026-08-13:
    "why throw that away? use the concepts alongside the board embeddings" + time-state).
    codes give the head the discrete metastable plan-state for free AND make human
    tendencies readable per concept; heads=0 falls back to the v1 board-only form."""

    def __init__(self, d_in=192, d=192, d_move=64, heads=0, codes=64):
        super().__init__()
        self.heads = heads
        self.elo = nn.Embedding(N_BUCKET, 32)
        self.clkp = nn.Sequential(nn.Linear(N_CLK, 32), nn.GELU())
        if heads:
            self.code_emb = nn.ModuleList([nn.Embedding(codes, 32) for _ in range(heads)])
        self.ctx = nn.Sequential(nn.Linear(d_in + 32 + 32 + (32 if heads else 0), d),
                                 nn.GELU())
        self.e_from = nn.Embedding(64, d_move)
        self.e_to = nn.Embedding(64, d_move)
        self.e_promo = nn.Embedding(5, d_move)
        self.score = nn.Sequential(nn.Linear(d + d_move, d), nn.GELU(), nn.Linear(d, 1))

    def move_logits(self, phi, elo_b, mids, row_of, clk=None, code_ids=None):
        """phi (B,d_in), elo_b (B,), mids (M,3) legal moves flattened over the batch,
        row_of (M,), clk (B,N_CLK) or None, code_ids (B,heads) or None -> logits (M,)"""
        if clk is None:
            clk = torch.zeros(len(phi), N_CLK, device=phi.device)
            clk[:, 0] = clk[:, 1] = 1.0               # ample time
            clk[:, 2] = 1.0
        parts = [phi, self.elo(elo_b), self.clkp(clk)]
        if self.heads:
            if code_ids is None:
                code_ids = torch.zeros(len(phi), self.heads, dtype=torch.long,
                                       device=phi.device)
            parts.append(sum(self.code_emb[h](code_ids[:, h])
                             for h in range(self.heads)))
        c = self.ctx(torch.cat(parts, -1))
        m = self.e_from(mids[:, 0]) + self.e_to(mids[:, 1]) + self.e_promo(mids[:, 2])
        return self.score(torch.cat([c[row_of], m], -1)).squeeze(-1)

    @staticmethod
    def nll_loss(logits, row_of, played_ix, n_rows):
        """segment log-softmax CE over ragged legal-move sets. played_ix: flat index of the
        played move per row. -> (loss, bits (n_rows,), top1 (n_rows,))"""
        # stable segment logsumexp via scatter
        mx = torch.full((n_rows,), -1e30, device=logits.device)
        mx = mx.scatter_reduce(0, row_of, logits, reduce="amax", include_self=True)
        ex = torch.exp(logits - mx[row_of])
        se = torch.zeros(n_rows, device=logits.device).scatter_add(0, row_of, ex)
        logZ = mx + torch.log(se + 1e-12)
        logp_played = logits[played_ix] - logZ
        bits = -logp_played * (1.0 / float(np.log(2.0)))
        # top-1: is the played move the argmax of its segment?
        best = torch.full((n_rows,), -1e30, device=logits.device)
        best = best.scatter_reduce(0, row_of, logits, reduce="amax", include_self=True)
        top1 = (logits[played_ix] >= best - 1e-9).float()
        return -logp_played.mean(), bits, top1


def mid_of(mv: chess.Move):
    return (mv.from_square, mv.to_square,
            {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3,
             chess.KNIGHT: 4}.get(mv.promotion, 0))


def build_batch(samples, dev):
    """samples: (tok, glob, elo_b, legal_mids, played_j[, clk4]) -> tensors"""
    tok = torch.from_numpy(np.stack([s[0] for s in samples]).astype(np.int64)).to(dev)
    glob = torch.from_numpy(np.stack([s[1] for s in samples]).astype(np.float32)).to(dev)
    elo_b = torch.tensor([s[2] for s in samples], dtype=torch.long, device=dev)
    clk = torch.tensor([s[5] if len(s) > 5 else clk_feats(float("nan"), float("nan"), 600)
                        for s in samples], dtype=torch.float32, device=dev)
    mids, row_of, played_ix = [], [], []
    for i, s in enumerate(samples):
        played_ix.append(len(mids) + s[4])
        mids.extend(s[3])
        row_of.extend([i] * len(s[3]))
    return (tok, glob, elo_b,
            torch.tensor(mids, dtype=torch.long, device=dev),
            torch.tensor(row_of, dtype=torch.long, device=dev),
            torch.tensor(played_ix, dtype=torch.long, device=dev), clk)


def sample_plies(games, per_game, rng):
    """replay games, yield (tok, glob, elo_bucket, legal mids, played index) per sampled ply."""
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
    out = []
    for gid, _res, ucis, _flag, elos in games:
        b = chess.Board()
        idx = sorted(rng.choice(len(ucis), size=min(per_game, len(ucis)), replace=False))
        ptr = 0
        for t, u in enumerate(ucis):
            try:
                mv = chess.Move.from_uci(u)
            except Exception:
                break
            if mv not in b.legal_moves:
                break
            if ptr < len(idx) and t == idx[ptr]:
                ptr += 1
                legal = list(b.legal_moves)
                if 0 < len(legal) <= 128:
                    tk, gl = tokenize(b)
                    out.append((np.asarray(tk), np.asarray(gl),
                                elo_bucket(elos[0] if b.turn else elos[1]),
                                [mid_of(m) for m in legal], legal.index(mv)))
            b.push(mv)
    return out


def sample_shard_plies(shard_dir, n, rng):
    """v2 data path: the FULL-corpus shards (86M positions) carry per-position CLOCKS.
    The played move is derived by matching consecutive rows of a game (packed-board diff);
    the mover's remaining time before the move sits on the PREVIOUS row (the %clk of their
    own last move), the opponent's on the current row. NOTE the source filter dropped
    moves with <30s left -- the sub-30s cliff needs a re-extraction (min_clock_s=0).
    -> (tok, glob, elo_bucket, legal mids, played_j, clk4)"""
    import glob as _gl
    from catspace.research.tools.chess_specific.chessdata.encode import (
        board_from_packed, encode_packed)
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
        tokenize)
    files = sorted(_gl.glob(str(shard_dir) + "/shard_*.npz"))
    if not files:
        raise SystemExit(f"no shards under {shard_dir}")
    out = []
    per = max(1, n // len(files))
    for f in files:
        z = np.load(f)
        gid, ply, clk = z["game_id"], z["ply"], z["clock"]
        we, be = z["white_elo"], z["black_elo"]
        packed, meta = z["packed"], z["meta"]
        cand = np.flatnonzero((gid[1:] == gid[:-1]) & (ply[1:] == ply[:-1] + 1))
        take = rng.choice(cand, size=min(per, len(cand)), replace=False)
        for i in take:
            try:
                b1 = board_from_packed(packed[i], meta[i])
                legal = list(b1.legal_moves)
                if not (0 < len(legal) <= 128):
                    continue
                tgt = packed[i + 1]
                mv = None
                for j, m in enumerate(legal):
                    b1.push(m)
                    if (encode_packed(b1) == tgt).all():
                        mv = j
                        b1.pop()
                        break
                    b1.pop()
                if mv is None:
                    continue
                # clocks: mover's = previous row of the SAME game at ply-1, else NaN
                c_self = float(clk[i - 1]) if (i > 0 and gid[i - 1] == gid[i]
                                               and ply[i - 1] == ply[i] - 1) \
                    else float("nan")
                c_opp = float(clk[i])
                g0 = np.flatnonzero(gid == gid[i])[:4]
                base = float(np.nanmax(clk[g0])) if len(g0) else 600.0
                if not (base == base) or base <= 0:
                    base = 600.0
                elo = int(we[i]) if b1.turn else int(be[i])
                tk, gl = tokenize(b1)
                out.append((np.asarray(tk), np.asarray(gl), elo_bucket(elo),
                            [mid_of(m) for m in legal], mv,
                            clk_feats(c_self, c_opp, base)))
            except Exception:
                continue
        if len(out) >= n:
            break
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--shards", default=None,
                    help="shard dir (per-position clocks) -> v2 data path; else parquet")
    ap.add_argument("--codes", type=int, default=1,
                    help="1 = feed the quantized concept codes alongside phi (v2)")
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os, re
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
        load_net)
    dev = args.device
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    model, _pay = load_net(args.ckpt, dev)
    model.eval()
    import chess as _ch
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize as _tk0
    _t, _g = _tk0(_ch.Board())
    with torch.no_grad():
        d_in = model.backbone(torch.tensor([_t], dtype=torch.long, device=dev),
                              torch.tensor([_g], dtype=torch.float32, device=dev)).shape[-1]

    jqt = None
    if args.codes:
        try:
            from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
                JQTModule)
            pj = torch.load(next(pth for pth in (base + "_jqt.pt", stem + "_jqt.pt")
                                 if os.path.exists(pth)), map_location=dev,
                            weights_only=False)
            jqt = JQTModule(d_model=pj["d_in"], heads=pj["heads"], codes=pj["codes"],
                            d=pj["d"], square_codes=pj.get("square_codes", 0),
                            piece_codes=pj.get("piece_codes", 0)).to(dev)
            jqt.load_state_dict(pj["state_dict"], strict=False)
            jqt.eval()
            print(f"[human] concept codes IN ({pj['heads']}x{pj['codes']})", flush=True)
        except Exception as e:
            print(f"[human] no jqt sidecar, codes off ({e})", flush=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    if args.shards:
        plies = sample_shard_plies(args.shards, args.games * args.per_game, rng)
    else:
        games = T.load_human_games(args.games, seed=0)
        print(f"[human] {len(games)} lichess games loaded [{time.time()-t0:.0f}s]",
              flush=True)
        t0 = time.time()
        plies = sample_plies(games, args.per_game, rng)
    print(f"[human] {len(plies)} plies sampled [{time.time()-t0:.0f}s]", flush=True)
    rng.shuffle(plies)
    n_val = max(512, len(plies) // 20)
    val, tr_p = plies[:n_val], plies[n_val:]

    net = HumanMoves(d_in=d_in, heads=(jqt.heads if jqt is not None else 0),
                     codes=(jqt.codes if jqt is not None else 64)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    log = open(stem + "_human_steps.jsonl", "a", buffering=1)
    t0 = time.time()
    for step in range(args.steps):
        ixs = rng.choice(len(tr_p), size=args.batch, replace=False)
        tok, glob, elo_b, mids, row_of, played_ix, clk = build_batch(
            [tr_p[i] for i in ixs], dev)
        with torch.no_grad():
            phi = model.backbone(tok, glob)              # FROZEN trunk
            ids = jqt.target_codes(phi)[1] if jqt is not None else None
        logits = net.move_logits(phi, elo_b, mids, row_of, clk, ids)
        loss, _, _ = HumanMoves.nll_loss(logits, row_of, played_ix, len(ixs))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            with torch.no_grad():
                bits_all, top_all = [], []
                for a in range(0, min(len(val), 2048), 256):
                    tok, glob, elo_b, mids, row_of, played_ix, clk = build_batch(
                        val[a:a + 256], dev)
                    phi = model.backbone(tok, glob)
                    ids = jqt.target_codes(phi)[1] if jqt is not None else None
                    lg = net.move_logits(phi, elo_b, mids, row_of, clk, ids)
                    _, bits, top1 = HumanMoves.nll_loss(lg, row_of, played_ix,
                                                        len(val[a:a + 256]))
                    bits_all.append(bits.cpu()); top_all.append(top1.cpu())
                bits_m = float(torch.cat(bits_all).mean())
                top_m = float(torch.cat(top_all).mean())
            print(f"[human] step {step} loss {float(loss):.3f} | val {bits_m:.2f} bits/move "
                  f"top1 {top_m:.1%} [{(time.time()-t0)/60:.0f}m]", flush=True)
            log.write(json.dumps({"step": step, "loss": float(loss),
                                  "val_bits": bits_m, "val_top1": top_m}) + "\n")
    outp = args.out or (stem + "_human.pt")
    torch.save({"state_dict": net.state_dict(), "d_in": d_in,
                "heads": (jqt.heads if jqt is not None else 0),
                "codes": (jqt.codes if jqt is not None else 64),
                "elo_lo": ELO_LO, "elo_hi": ELO_HI, "elo_w": ELO_W}, outp)
    # context for the verdict: uniform-over-legal is ~5 bits; Maia-class models sit
    # near ~1.6-2.0 bits/move (top1 ~50%) on rapid pools
    print(f"[human] VERDICT val {bits_m:.2f} bits/move top1 {top_m:.1%} "
          f"(uniform ~5 bits) -> {outp}", flush=True)


def _tests():
    torch.manual_seed(0)
    ok = True
    net = HumanMoves(d_in=32, d=64, d_move=32)
    # ragged segment softmax: probabilities sum to 1 per board, CE decreases on a
    # planted preference (all boards secretly love the first legal move)
    B = 8
    phi = torch.randn(B, 32)
    elo_b = torch.randint(0, N_BUCKET, (B,))
    mids, row_of, played_ix = [], [], []
    rng = np.random.default_rng(0)
    for i in range(B):
        n = int(rng.integers(2, 9))
        played_ix.append(len(mids) + 0)                  # planted: move 0 is always played
        mids.extend([(int(rng.integers(0, 64)), int(rng.integers(0, 64)), 0)
                     for _ in range(n)])
        row_of.extend([i] * n)
    mids = torch.tensor(mids); row_of = torch.tensor(row_of)
    played_ix = torch.tensor(played_ix)
    lg = net.move_logits(phi, elo_b, mids, row_of)
    _, bits0, _ = HumanMoves.nll_loss(lg, row_of, played_ix, B)
    # per-board softmax sums to 1
    mx = torch.full((B,), -1e30).scatter_reduce(0, row_of, lg.detach(), reduce="amax",
                                                include_self=True)
    p = torch.exp(lg.detach() - mx[row_of])
    s = torch.zeros(B).scatter_add(0, row_of, p)
    p = p / s[row_of]
    sums = torch.zeros(B).scatter_add(0, row_of, p)
    ok &= bool(torch.allclose(sums, torch.ones(B), atol=1e-5))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(120):
        lg = net.move_logits(phi, elo_b, mids, row_of)
        loss, _, _ = HumanMoves.nll_loss(lg, row_of, played_ix, B)
        opt.zero_grad(); loss.backward(); opt.step()
    lg = net.move_logits(phi, elo_b, mids, row_of)
    _, bits1, top1 = HumanMoves.nll_loss(lg, row_of, played_ix, B)
    ok &= float(bits1.mean()) < float(bits0.mean()) - 0.5 and float(top1.mean()) > 0.9
    print(f"[human] segment softmax sums to 1; planted preference learned "
          f"({float(bits0.mean()):.2f} -> {float(bits1.mean()):.2f} bits, "
          f"top1 {float(top1.mean()):.0%})  {'OK' if ok else 'FAIL'}")
    # elo buckets: ends clamp, interior buckets distinct
    ok &= elo_bucket(200) == 0 and elo_bucket(3000) == N_BUCKET - 1
    ok &= elo_bucket(1500) != elo_bucket(1600)
    print(f"[human] elo bucketing clamps + separates  {'OK' if ok else 'FAIL'}")
    # v2: codes + clock inputs accepted, defaults hold, clk features sane
    net2 = HumanMoves(d_in=32, d=64, d_move=32, heads=2, codes=8)
    ids2 = torch.randint(0, 8, (B, 2))
    clk2 = torch.tensor([clk_feats(15.0, 300.0, 300.0)] * B)
    lg2 = net2.move_logits(phi, elo_b, mids, row_of, clk2, ids2)
    lg2d = net2.move_logits(phi, elo_b, mids, row_of)          # defaults: ample time
    ok &= lg2.shape == lg2d.shape == (len(mids),)
    cf = clk_feats(15.0, 300.0, 300.0)
    ok &= cf[3] == 1.0 and cf[2] < 0.1                          # low-time flag + low frac
    ok &= clk_feats(float("nan"), float("nan"), 600)[3] == 0.0  # NaN -> ample
    print(f"[human] v2 codes+clock path, low-time features  {'OK' if ok else 'FAIL'}")
    print("ALL HUMAN-MOVES TESTS PASSED" if ok else "TESTS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--ckpt" in sys.argv:
        main()
    else:
        _tests()
