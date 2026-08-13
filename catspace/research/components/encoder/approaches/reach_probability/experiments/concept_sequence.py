#!/usr/bin/env python
"""concept_sequence.py -- the SEQUENCE LAYER (Kaveh 2026-08-13: "a regular transformer that
can use the concepts and predict whether another concept will happen soon or whether a game
will come ... a prediction of future likely concepts for my opponent").

A CAUSAL transformer over the game's concept-activation stream -- the third prediction
mechanism, complementary to the other two:
    ConceptDynamics   (state, move) -> next codes         one step, move-conditioned
    GeoAttention      dA/dB relations between subgoals    relational, horizon-free
    THIS              activation HISTORY -> what fires SOON + who wins   sequence-aware

Per ply t the model reads the 8 global head codes + side-to-move and predicts:
    activation head:  P(code c NEWLY activates in head h within the next K plies),
                      K in HORIZONS -- "newly" = not active at t (occupancy of a code you
                      already hold is persistence, not prediction)
    outcome head:     P(white win / draw / black win) from the partial sequence

Trained as a FROZEN PROBE on coded games (the jqt4 charter: substrate first, probes second;
end-to-end inclusion is a jqt5 decision). Sidecar: <stem>_seq.pt; the server's 'futures'
panel serves it when present.

    .venv/bin/python -m ...concept_sequence --ckpt <field.pt> [--games 2000] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZONS = (2, 6, 12)          # plies: "soon", "this phase", "this plan"


class ConceptSequence(nn.Module):
    def __init__(self, heads=8, codes=64, d=128, layers=3, n_attn=4, max_len=256):
        super().__init__()
        self.heads, self.codes = heads, codes
        self.emb = nn.ModuleList([nn.Embedding(codes, d) for _ in range(heads)])
        self.side = nn.Embedding(2, d)
        self.pos = nn.Embedding(max_len, d)
        lyr = nn.TransformerEncoderLayer(d, n_attn, 4 * d, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(lyr, num_layers=layers)
        self.act_head = nn.Linear(d, heads * codes * len(HORIZONS))
        self.out_head = nn.Linear(d, 3)
        self.max_len = max_len

    def forward(self, ids, stm):
        """ids (B,T,H) codes; stm (B,T) 0=white to move. -> act logits (B,T,H,C,K), wdl (B,T,3)"""
        B, T, H = ids.shape
        x = sum(self.emb[h](ids[:, :, h]) for h in range(H))
        x = x + self.side(stm) + self.pos(torch.arange(T, device=ids.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        z = self.tr(x, mask=mask, is_causal=True)
        act = self.act_head(z).view(B, T, self.heads, self.codes, len(HORIZONS))
        return act, self.out_head(z)


def activation_targets(ids, horizons=HORIZONS):
    """ids (T,H) -> y (T,H,C,K): code c NEWLY active in head h within (t, t+K]."""
    T, H = ids.shape
    C = int(ids.max()) + 1 if T else 1
    y = np.zeros((T, H, C, len(horizons)), np.float32)
    for t in range(T):
        cur = ids[t]
        for ki, K in enumerate(horizons):
            fut = ids[t + 1:t + 1 + K]
            for h in range(H):
                for c in np.unique(fut[:, h]) if len(fut) else []:
                    if c != cur[h]:
                        y[t, h, int(c), ki] = 1.0
    return y


def _causal_check(model, ids, stm):
    """prediction at t must be blind to tokens after t."""
    model.eval()                     # dropout off: differences must come from LEAKAGE only
    with torch.no_grad():
        a1, _ = model(ids, stm)
        ids2 = ids.clone()
        ids2[:, ids.shape[1] // 2:] = (ids2[:, ids.shape[1] // 2:] + 7) % model.codes
        a2, _ = model(ids2, stm)
    t_probe = ids.shape[1] // 2 - 1
    return bool(torch.allclose(a1[:, :t_probe + 1], a2[:, :t_probe + 1], atol=1e-5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os, re
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
        JQTModule)
    from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa import (
        ReachViT)

    dev = args.device
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    pk = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = ReachViT(**pk["arch"]).to(dev)
    model.load_state_dict(pk["state_dict"], strict=False)
    model.eval()
    pj = torch.load(next(p for p in (base + "_jqt.pt", stem + "_jqt.pt")
                         if os.path.exists(p)), map_location=dev, weights_only=False)
    jqt = JQTModule(d_model=pj["d_in"], heads=pj["heads"], codes=pj["codes"], d=pj["d"],
                    square_codes=pj.get("square_codes", 0),
                    piece_codes=pj.get("piece_codes", 0)).to(dev)
    jqt.load_state_dict(pj["state_dict"], strict=False)
    jqt.eval()
    H_, C_ = pj["heads"], pj["codes"]

    tr = T.load_piecedown_games()
    g_of = tr.game_of_row()
    n_g = int(g_of.max()) + 1
    rng = np.random.default_rng(0)
    take = rng.permutation(n_g)[:args.games]
    # ---- code the games (frozen substrate) ---------------------------------------------
    seqs, stms, outs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for gi in take:
            rows = np.flatnonzero(g_of == gi)
            if len(rows) < 8 or len(rows) > 240:
                continue
            tok = torch.from_numpy(tr.tok[rows].astype(np.int64)).to(dev)
            glob = torch.from_numpy(tr.glob[rows].astype(np.float32)).to(dev)
            phi = model.backbone(tok, glob)
            _, ids = jqt.target_codes(phi)
            seqs.append(ids.cpu().numpy().astype(np.int16))
            stms.append((tr.glob[rows, 0] < 0.5).astype(np.int64))   # glob[0]=stm white flag
            res = float(tr.result[rows[0]]) if hasattr(tr, "result") else 0.5
            outs.append(0 if res == 1.0 else (2 if res == 0.0 else 1))
    print(f"[seq] coded {len(seqs)} games [{time.time()-t0:.0f}s]", flush=True)
    n_val = max(8, len(seqs) // 10)
    val_ix = set(range(len(seqs))[-n_val:])

    net = ConceptSequence(heads=H_, codes=C_).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def batch(ixs):
        L = max(len(seqs[i]) for i in ixs)
        ids = torch.zeros(len(ixs), L, H_, dtype=torch.long)
        stm = torch.zeros(len(ixs), L, dtype=torch.long)
        y = torch.zeros(len(ixs), L, H_, C_, len(HORIZONS))
        w = torch.zeros(len(ixs), L)
        o = torch.zeros(len(ixs), dtype=torch.long)
        for j, i in enumerate(ixs):
            s = seqs[i]
            ids[j, :len(s)] = torch.from_numpy(s.astype(np.int64))
            stm[j, :len(s)] = torch.from_numpy(stms[i][:len(s)])
            yt = activation_targets(s)
            y[j, :len(s), :, :yt.shape[2]] = torch.from_numpy(yt)
            w[j, :len(s)] = 1.0
            o[j] = outs[i]
        return (ids.to(dev), stm.to(dev), y.to(dev), w.to(dev), o.to(dev))

    tr_ix = [i for i in range(len(seqs)) if i not in val_ix]
    log = open((args.out or (stem + "_seq")) + "_steps.jsonl", "a", buffering=1)
    for step in range(args.steps):
        ixs = rng.choice(tr_ix, size=min(args.batch, len(tr_ix)), replace=False)
        ids, stm, y, w, o = batch(list(ixs))
        act, wdl = net(ids, stm)
        l_act = (F.binary_cross_entropy_with_logits(act, y, reduction="none")
                 .mean((2, 3, 4)) * w).sum() / w.sum()
        l_out = (F.cross_entropy(wdl.transpose(1, 2), o[:, None].expand(-1, wdl.shape[1]),
                                 reduction="none") * w).sum() / w.sum()
        loss = l_act + 0.3 * l_out
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                vi = list(val_ix)[:16]
                ids, stm, y, w, o = batch(vi)
                act, wdl = net(ids, stm)
                p = torch.sigmoid(act)[w.bool()]
                yv = y[w.bool()]
                # AUC via rank statistic on a subsample
                fl_p = p.reshape(-1).cpu().numpy()
                fl_y = yv.reshape(-1).cpu().numpy()
                sub = np.random.default_rng(1).choice(len(fl_p), size=min(200_000, len(fl_p)),
                                                      replace=False)
                fp, fy = fl_p[sub], fl_y[sub]
                if fy.sum() and (1 - fy).sum():
                    order = fp.argsort()
                    r = np.empty(len(fp)); r[order] = np.arange(1, len(fp) + 1)
                    auc = (r[fy == 1].sum() - fy.sum() * (fy.sum() + 1) / 2) / \
                          (fy.sum() * (1 - fy).sum())
                else:
                    auc = float("nan")
                acc = float((wdl[:, -1].argmax(-1) == o).float().mean())
            print(f"[seq] step {step} loss {float(loss):.4f} act-AUC(val) {auc:.3f} "
                  f"outcome-acc(end) {acc:.2f}", flush=True)
            log.write(json.dumps({"step": step, "loss": float(loss),
                                  "auc": float(auc), "out_acc": acc}) + "\n")
    outp = args.out or (stem + "_seq.pt")
    torch.save({"state_dict": net.state_dict(), "heads": H_, "codes": C_,
                "horizons": list(HORIZONS), "d": 128, "layers": 3}, outp)
    print(f"[seq] VERDICT act-AUC(val) {auc:.3f} outcome-acc(final-ply) {acc:.2f} -> {outp}",
          flush=True)


def _tests():
    torch.manual_seed(0)
    ok = True
    H, C = 4, 16
    net = ConceptSequence(heads=H, codes=C, d=64, layers=2, n_attn=2)
    # causality: future tokens must not leak backward
    ids = torch.randint(0, C, (2, 24, H))
    stm = torch.randint(0, 2, (2, 24))
    ok &= _causal_check(net, ids, stm)
    print(f"[seq] causal mask holds  {'OK' if ok else 'FAIL'}")
    # planted rule: code 3 in head 0 -> code 5 appears in head 1 within 2 plies.
    rng = np.random.default_rng(0)

    def mk():
        s = rng.integers(0, C, (32, H))
        s[:, 1] = np.where(s[:, 1] == 5, 6, s[:, 1])      # 5 only appears via the rule
        for t in range(30):
            if s[t, 0] == 3:
                s[t + 1 + int(rng.integers(0, 2)), 1] = 5
        return s

    seqs = [mk() for _ in range(64)]
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    ki = 0                                                # horizon K=2
    for step in range(150):
        ids = torch.from_numpy(np.stack(seqs[:32]).astype(np.int64))
        stm = torch.zeros(32, 32, dtype=torch.long)
        y = torch.stack([torch.from_numpy(activation_targets(s)) for s in seqs[:32]])
        act, _ = net(ids, stm)
        loss = F.binary_cross_entropy_with_logits(
            act[:, :, :, :y.shape[3]], y.float())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = mk()
        ids = torch.from_numpy(s.astype(np.int64))[None]
        act, _ = net(ids, torch.zeros(1, 32, dtype=torch.long))
        p = torch.sigmoid(act[0, :, 1, 5, ki])            # P(head1 code5 soon)
        trig = torch.tensor([s[t, 0] == 3 for t in range(32)])
        gap = float(p[trig].mean() - p[~trig].mean())
    ok &= gap > 0.15
    print(f"[seq] planted trigger->consequence learned: P-gap {gap:.2f}  "
          f"{'OK' if ok else 'FAIL'}")
    print("ALL CONCEPT-SEQUENCE TESTS PASSED" if ok else "TESTS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--ckpt" in sys.argv:
        main()
    else:
        _tests()
