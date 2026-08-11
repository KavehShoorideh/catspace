#!/usr/bin/env python
"""concept_dynamics.py -- the CONCEPT-TRANSITION head (Kaveh 2026-08-11): given the parent's
trunk encoding (board read ONCE) + the move as a token, predict the CHILD's vector-quantized
concept profile. "This move connects the rooks / trades the queens" -- move effects stated in
concepts, not distances. The planning inversion this powers: pick target concepts (subgoals),
rank moves by predicted activation -- searching in code space with no further board reads.

Why this beats the distance head on feeding: labels are FREE -- the frozen quantizer codes any
child in one forward, so every (parent, move, child) triple in the corpus is supervision.
First source: the wdl shards' 1.76M tokenized children with move ids.

    phi(parent)[128] + e(move) -> MLP -> 8 x 64 logits  (child's code per VQ head)
Baseline to beat: COPY the parent's codes (most moves flip few heads).

    .venv/bin/python -m ...concept_dynamics --ckpt <field.pt> [--steps 6000] [--save]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
    ConceptVQ)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


class ConceptDynamics(nn.Module):
    def __init__(self, d_in=128, heads=8, codes=64, d_move=48, hidden=384):
        super().__init__()
        self.heads, self.codes = heads, codes
        self.e_from = nn.Embedding(64, d_move)
        self.e_to = nn.Embedding(64, d_move)
        self.e_promo = nn.Embedding(5, d_move)
        self.net = nn.Sequential(nn.Linear(d_in + d_move, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, heads * codes))

    def forward(self, phi, mids):
        m = self.e_from(mids[:, 0]) + self.e_to(mids[:, 1]) + self.e_promo(mids[:, 2])
        return self.net(torch.cat([phi, m], -1)).view(len(phi), self.heads, self.codes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--game-transitions", type=int, default=0,
                    help="ALSO train on N corpus-derived consecutive-ply transitions "
                         "(Kaveh: feed it a lot more before judging)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    pv = torch.load(base + "_vq.pt", map_location=args.device, weights_only=False)
    vq = ConceptVQ(d_in=pv["d_in"], heads=pv["heads"], codes=pv["codes"]).to(args.device)
    vq.load_state_dict(pv["state_dict"]); vq.eval()
    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)

    ddir = paths.derived("wdl_labels")
    shards = [os.path.join(ddir, f) for f in sorted(os.listdir(ddir)) if f.endswith(".npz")]
    P_ROW, C_TOK, C_GLOB, MID, OFF = [], [], [], [], [0]
    basei = 0
    for f in shards:
        pk = np.load(f)
        P_ROW.append(pk["row"]); C_TOK.append(pk["tok"]); C_GLOB.append(pk["glob"])
        MID.append(pk["mid"]); OFF.extend((pk["off"][1:] + basei).tolist())
        basei += len(pk["tok"])
    P_ROW = np.concatenate(P_ROW); C_TOK = np.concatenate(C_TOK)
    C_GLOB = np.concatenate(C_GLOB); MID = np.concatenate(MID)
    OFF = np.array(OFF, np.int64)
    n_par = len(P_ROW)
    print(f"[dyn] {n_par:,} parents, {len(C_TOK):,} shard (move -> child) transitions")
    GT = None
    if args.game_transitions:
        # transitions FROM THE CORPUS ITSELF (2026-08-11: replay-order reconstruction was 30%
        # wrong -- guessed alignments burn us; derive instead): consecutive same-game rows are
        # (parent, child); the move is the unique legal move whose push reproduces the child
        # tokens. Cached per corpus.
        import chess as _ch
        from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (
            row_to_board)
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            tokenize as _tok, move_ids)
        cache = paths.derived(f"game_transitions_{len(tr.tok)}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            GT = {"par": z["par"], "mid": z["mid"]}
        else:
            gor = tr.game_of_row()
            cand = np.flatnonzero(gor[:-1] == gor[1:])
            rng0 = np.random.default_rng(0)
            cand = cand[rng0.choice(len(cand), min(args.game_transitions, len(cand)),
                                    replace=False)]
            par_l, mid_l, bad = [], [], 0
            for r in cand:
                b = row_to_board(tr.tok[r], tr.glob[r])
                if not b.is_valid():
                    bad += 1; continue
                child = tr.tok[r + 1]
                found = None
                for mv in b.legal_moves:
                    b.push(mv)
                    tk, _g = _tok(b)
                    b.pop()
                    if np.array_equal(child, np.asarray(tk)):
                        found = mv; break
                if found is None:
                    bad += 1; continue
                par_l.append(int(r)); mid_l.append(move_ids(found))
            GT = {"par": np.array(par_l, np.int64), "mid": np.array(mid_l, np.int64)}
            np.savez(cache, par=GT["par"], mid=GT["mid"])
            print(f"[dyn] derived {len(par_l):,} transitions ({bad:,} undecodable) -> {cache}")
        print(f"[dyn] + {len(GT['par']):,} whole-game transitions (corpus-derived)")

    # ---- precompute: parent phi + parent codes + child codes (frozen nets, batched) ----
    def phi_codes(tok, glob):
        with torch.no_grad():
            phi = net.backbone(torch.from_numpy(tok.astype(np.int64)).to(args.device),
                               torch.from_numpy(glob.astype(np.float32)).to(args.device))
            _, ids, _ = vq(phi)
        return phi.float().cpu(), ids.cpu()

    print("[dyn] featurizing parents...", flush=True)
    PPHI, PCODE = [], []
    for a in range(0, n_par, 4096):
        ph, ic = phi_codes(tr.tok[P_ROW[a:a+4096]], tr.glob[P_ROW[a:a+4096]])
        PPHI.append(ph); PCODE.append(ic)
    PPHI = torch.cat(PPHI); PCODE = torch.cat(PCODE)
    GPHI = GCODE_P = GCODE_C = None
    if GT is not None:
        print("[dyn] featurizing game-transition rows...", flush=True)
        gp, gc_p, gc_c = [], [], []
        for a in range(0, len(GT["par"]), 4096):
            rr = GT["par"][a:a+4096]
            ph, ic = phi_codes(tr.tok[rr], tr.glob[rr])
            gp.append(ph); gc_p.append(ic)
            _, icc = phi_codes(tr.tok[rr + 1], tr.glob[rr + 1])
            gc_c.append(icc)
        GPHI = torch.cat(gp); GCODE_P = torch.cat(gc_p); GCODE_C = torch.cat(gc_c)
    print("[dyn] featurizing children...", flush=True)
    CCODE = []
    for a in range(0, len(C_TOK), 4096):
        _, ic = phi_codes(C_TOK[a:a+4096], C_GLOB[a:a+4096])
        CCODE.append(ic)
    CCODE = torch.cat(CCODE)

    # child index -> parent index
    par_of_child = np.zeros(len(C_TOK), np.int64)
    for i in range(n_par):
        par_of_child[OFF[i]:OFF[i+1]] = i
    val_par = np.zeros(n_par, bool)
    val_par[np.random.default_rng(0).choice(n_par, n_par // 10, replace=False)] = True
    tr_ch = np.flatnonzero(~val_par[par_of_child])
    va_ch = np.flatnonzero(val_par[par_of_child])
    print(f"[dyn] transitions: fit {len(tr_ch):,}  val {len(va_ch):,}")

    model = ConceptDynamics(d_in=PPHI.shape[1], heads=pv["heads"], codes=pv["codes"]).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    MID_t = torch.from_numpy(MID.astype(np.int64))
    GMID_t = torch.from_numpy(GT["mid"].astype(np.int64)) if GT is not None else None
    n_gval = len(GT["par"]) // 10 if GT is not None else 0   # tail 10% of game transitions = val
    for step in range(args.steps):
        if GT is not None and step % 2 == 1:                 # alternate: shard / game batches
            gs = np.random.randint(0, len(GT["par"]) - n_gval, args.batch)
            logits = model(GPHI[gs].to(args.device), GMID_t[gs].to(args.device))
            tgt = GCODE_C[gs].to(args.device)
            loss = sum(torch.nn.functional.cross_entropy(logits[:, h], tgt[:, h])
                       for h in range(pv["heads"])) / pv["heads"]
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 1000 == 0:
                print(f"[dyn] step {step+1}: CE {float(loss):.4f} (game)", flush=True)
            continue
        sel = tr_ch[np.random.randint(0, len(tr_ch), args.batch)]
        pi = par_of_child[sel]
        logits = model(PPHI[pi].to(args.device), MID_t[sel].to(args.device))
        tgt = CCODE[sel].to(args.device)
        loss = sum(torch.nn.functional.cross_entropy(logits[:, h], tgt[:, h])
                   for h in range(pv["heads"])) / pv["heads"]
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"[dyn] step {step+1}: CE {float(loss):.4f}", flush=True)

    # ---- eval: per-head accuracy vs the COPY baseline; flip detection ----
    with torch.no_grad():
        sel = va_ch[np.random.randint(0, len(va_ch), 20000)]
        pi = par_of_child[sel]
        logits = model(PPHI[pi].to(args.device), MID_t[sel].to(args.device))
        pred = logits.argmax(-1).cpu()
        tgt = CCODE[sel]; par = PCODE[pi]
    acc = (pred == tgt).float().mean(0)
    copy_acc = (par == tgt).float().mean(0)
    changed = (tgt != par)
    flip_recall = (pred[changed] == tgt[changed]).float().mean()
    flip_base = float(changed.float().mean())
    print("\n[dyn] held-out per-head accuracy (model vs copy-parent baseline):")
    for h in range(pv["heads"]):
        print(f"  head {h}: model {acc[h]:.3f}  copy {copy_acc[h]:.3f}")
    print(f"[dyn] FLIPS (codes that change, {flip_base:.1%} of slots): "
          f"model predicts the new code {float(flip_recall):.1%} of the time (copy: 0%)")
    if args.save:
        torch.save({"state_dict": model.state_dict(), "heads": pv["heads"],
                    "codes": pv["codes"], "d_in": PPHI.shape[1]}, base + "_dyn.pt")
        print(f"[dyn] saved {base}_dyn.pt")


if __name__ == "__main__":
    main()
