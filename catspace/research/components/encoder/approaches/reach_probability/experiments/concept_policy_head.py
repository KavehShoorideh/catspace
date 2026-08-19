#!/usr/bin/env python
"""concept_policy_head.py -- behavioral cloning over ALL siblings: played vs every legal alternative.

"all the moves that stockfish did not play, let's label them as not played... we teach the system
that this is what you shouldn't play because stockfish didn't play it. We don't wanna do a random
move as a contrast -- every other move except the one stockfish played." (Kaveh 2026-08-19)

He is right, and this corrects the cef4 design two ways at once. cef4 contrasted the played move
against 7 RANDOM siblings -- mostly garbage moves -- so the learned boundary was "garbage vs
non-garbage" and transferred nothing (winning-track 49.5% vs 47.3% random). Taking ALL siblings
includes the hard negatives (good moves SF happened not to pick) with no engine query to select
them. And the readout is a DEDICATED scorer rather than the win-probability head, whose
calibration cef4 was quietly distorting.

WHAT THE LABEL MEANS, precisely. 42-52%% of legal moves also keep the game result (measured,
winning-track census), so "not played" is NOT "bad" -- it is "SF preferred otherwise". The loss
is therefore a RANKING: softmax over the played move and every legal sibling, cross-entropy to
the played slot, with label smoothing so equal-quality siblings are not crushed to zero. What
this learns is the demonstrator's preference -- behavioral cloning on trajectories, the
robotics-legal object -- and a perfect learner aces the winning-track test by construction,
since it picks SF's move.

WHAT IT DOES NOT LEARN, kept on record: value or reachability. No goal conditioning, silent off
the demonstrator's distribution. It is the POLICY PRIOR of the AlphaZero shape; the reach field
stays the value half.

    .venv/bin/python -m ...concept_policy_head --steps 6000 --device mps
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyScorer(nn.Module):
    """child concept state -> scalar preference. Scores any child of any node, which is what a
    search prior needs; the softmax is taken over whatever children are on the table."""

    def __init__(self, d_in, hidden=1024, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.f = nn.Sequential(*layers)

    def forward(self, z):
        return self.f(z).squeeze(-1)


def build_step(args, dev):
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_event_field import ConceptEvent
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    t0 = time.time()
    ck = torch.load(args.concept_ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    enc = ConceptEvent(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                       square_codes=A["square_codes"], hidden=A["hidden"],
                       n_res=len(vocab), factored=True, n_eg=len(egs),
                       cls_reach=cr, cls_eg=ce, cls_out=co).to(dev)
    enc.load_state_dict(ck["state_dict"]); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    d_state = A["heads"] * 32 + 64 * 32

    sb = np.load(paths.derived(f"sibling_contrast_{args.sib_k}_{n}.npz"))
    p_row = sb["p_row"]
    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    tr_m = ~test_g[gor[p_row]]
    D = {k: torch.from_numpy(sb[k]) for k in ("pos_tok", "pos_glb", "neg_tok", "neg_glb", "n_neg")}
    tr_i = np.flatnonzero(tr_m); te_i = np.flatnonzero(~tr_m)
    K = D["neg_tok"].shape[1]
    print(f"[cpl] parents: {len(tr_i):,} train / {len(te_i):,} held-out | up to {K} siblings "
          f"(mean {float(D['n_neg'].float().mean()):.1f}) | d_state {d_state} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    net = PolicyScorer(d_state, hidden=args.hidden, depth=args.depth).to(dev)
    print(f"[cpl] {sum(p.numel() for p in net.parameters()):,} params (encoder frozen)",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    probe = te_i[rng.choice(len(te_i), min(2000, len(te_i)), replace=False)]

    @torch.no_grad()
    def encode(tk, gb):
        z, _, _, _ = enc.encode(tk.long().to(dev), gb.float().to(dev))
        return z

    def batch_logits(idx, model):
        """-> (B, 1+K) logits with -inf on invalid slots; slot 0 = played."""
        B = len(idx)
        tk = torch.cat([D["pos_tok"][idx][:, None], D["neg_tok"][idx]], 1)
        gb = torch.cat([D["pos_glb"][idx][:, None], D["neg_glb"][idx]], 1)
        z = encode(tk.reshape(B * (K + 1), 64), gb.reshape(B * (K + 1), 6))
        sc = model(z).view(B, K + 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool),
                           torch.arange(K)[None, :].expand(B, K) < D["n_neg"][idx][:, None]],
                          1).to(dev)
        return sc.masked_fill(~valid, float("-inf")), valid

    def step_fn(model, step):
        model.train()
        idx = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
        lg, valid = batch_logits(idx, model)
        # ranking CE to the played slot, label-smoothed over the VALID slots -- "SF preferred
        # this move", never "the others are worthless"
        logp = F.log_softmax(lg / args.temp, -1)
        nv = valid.float().sum(-1, keepdim=True)
        smooth = args.smooth / nv.clamp(min=1)
        tgt = torch.where(valid, smooth.expand_as(logp), torch.zeros_like(logp))
        tgt[:, 0] = tgt[:, 0] + (1.0 - args.smooth)
        loss = -(tgt * logp.masked_fill(~valid, 0.0)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            hits1 = hits3 = tot = 0
            with torch.no_grad():
                for a in range(0, len(probe), 128):
                    ii = torch.from_numpy(probe[a:a + 128])
                    lg2, _ = batch_logits(ii, model)
                    rk = (lg2 > lg2[:, :1]).sum(-1)          # siblings strictly above played
                    hits1 += int((rk == 0).sum()); hits3 += int((rk < 3).sum()); tot += len(ii)
            nl = float(D["n_neg"][torch.from_numpy(probe)].float().mean()) + 1
            el = time.time() - t0
            print(f"[cpl] step {step:>6}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | loss {float(loss):.4f} | "
                  f"picks played {hits1/tot:.3f} top3 {hits3/tot:.3f} "
                  f"(chance {1/nl:.3f}) held-out", flush=True)
            m = {"top1": hits1 / tot, "top3": hits3 / tot}
            model.train()
        return {"loss": float(loss.detach()), **m}

    return net, step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept-ckpt", default="artifacts/experiments/cef3_step10000.pt")
    ap.add_argument("--sib-k", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/cpl1")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)
    net, step_fn = build_step(args, args.device)
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=args.device)
    standard_train(step_fn, net, cfg, args)


if __name__ == "__main__":
    main()
