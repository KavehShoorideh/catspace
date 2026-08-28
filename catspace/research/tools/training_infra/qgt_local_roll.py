"""qgt_local_roll.py -- LOCALITY of the discrete transition (Kaveh 2026-08-28:
"if I'm paying attention to one corner, I shouldn't have to pay attention to the
whole world... roll localized concepts forward").

Chess is not metrically local (a queen crosses the board in a move); the right
locality is GRAPH locality on the position's own attack adjacency -- which the board
graph already carries as att edges (vac/dsc make the long-range exceptions explicit).
This probe asks three questions of a trained qgt3d (discrete-arm) checkpoint, no
retraining:

  1. IS the learned transition natively local? Roll the full T along real chains and
     histogram every flipped bit by hop distance (in the CURRENT position's attack
     graph) from the played move's from/to squares.
  2. Can we roll ONLY a corner? Per ply, restrict the roll to the move's h-hop halo
     -- two strictness modes:
        update  full attention, but only halo squares may CHANGE (the frame-axiom
                mode: everything else copies forward)
        attn    the relational path may only LOOK AT the halo (tok_mask): resources
                truly confined to the corner
     Fidelity = bit agreement vs the full roll, at each horizon, per h; cost = mean
     active fraction of the board.
  3. The planner's question: FIX a corner at ply 0 and track it for 6 plies while
     the game plays on. Corner bits from the local roll vs the TRUE encoder's corner
     bits, against the full roll's same score.

    .venv/bin/python -m catspace.research.tools.training_infra.qgt_local_roll \
        --ckpt artifacts/experiments/qgt3d_latest.pt
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import torch


def halo(att_pairs, seeds, hops):
    """h-hop BFS neighbourhood of `seeds` in an undirected attack adjacency.
    att_pairs: (n,2) int square pairs. Returns a 64-bool mask (seeds always in)."""
    adj = [[] for _ in range(64)]
    for a, b in att_pairs:
        adj[a].append(b)
        adj[b].append(a)
    mask = np.zeros(64, bool)
    q = deque((int(s), 0) for s in seeds)
    for s in seeds:
        mask[int(s)] = True
    while q:
        u, d = q.popleft()
        if d >= hops:
            continue
        for v in adj[u]:
            if not mask[v]:
                mask[v] = True
                q.append((v, d + 1))
    return mask


def local_roll(bt, z0, moves, active_seq, attn=False):
    """Roll with a per-step ACTIVE mask: only active squares update; the rest copy.
    attn=True additionally restricts the relational path to the active set."""
    z, out = z0, []
    for m, act in zip(moves, active_seq):
        lg, _ = bt(z, m, tok_mask=act if attn else None)
        zb = torch.sign(lg)
        zb = torch.where(zb == 0, torch.ones_like(zb), zb)
        z = torch.where(act[..., None], zb, z)
        out.append(z)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/qgt3d_latest.pt")
    ap.add_argument("--chain-graphs", default="data/derived/gko3_chain_graphs_mm.npz")
    ap.add_argument("--chain-starts", default="data/derived/gko1_chain_rows.npz")
    ap.add_argument("--chain-values", default="data/derived/gko1_chain_values.npz")
    ap.add_argument("--chain-fens", default="data/derived/gko1_chain_fens.npz")
    ap.add_argument("--n-chains", type=int, default=192)
    ap.add_argument("--hops", default="0,1,2,3")
    ap.add_argument("--units", action="store_true")
    args = ap.parse_args()

    if args.units:
        return _units()

    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    from catspace.research.tools.training_infra.qgt_field import QGTField, BitTransition
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize, MV_CAP

    dev = resolve_device("auto")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"]
    GR = A.get("dynamics") == "graph"
    assert A.get("dynamics") in ("discrete", "graph"), \
        f"{args.ckpt} has no bit transition (dynamics={A.get('dynamics')})"
    model = QGTField(d=A["d"], layers=A["layers"], sq_bits=A["sq_bits"],
                     rvq_levels=A["rvq_levels"], mv_bits=A["mv_bits"],
                     qa_levels=A["qa_levels"], n_q=A["n_q"], qa_heads=A["attn_heads"],
                     top_k=A["top_k"], out_bits=A["out_bits"])
    if GR:
        from catspace.research.tools.training_infra.qgt_field import GraphBitTransition
        model.bit_trans = GraphBitTransition(
            sq_bits=A["sq_bits"], mv_bits=A["mv_bits"], n_q=A["n_q"],
            heads=A["attn_heads"], top_k=A["top_k"], out_bits=A["out_bits"],
            w_id_init=A.get("w_id_init", 0.5))
    else:
        model.bit_trans = BitTransition(sq_bits=A["sq_bits"], mv_bits=A["mv_bits"],
                                        n_q=A["n_q"], heads=A["attn_heads"],
                                        top_k=A["top_k"], out_bits=A["out_bits"],
                                        w_id_init=A.get("w_id_init", 0.5))
    missing = model.load_state_dict(ck["state_dict"], strict=False)
    assert not missing.missing_keys, f"missing: {missing.missing_keys[:4]}"
    model = model.to(dev).eval()
    print(f"[local] {args.ckpt} step {ck.get('step')} | w_id "
          f"{float(model.bit_trans.w_id):.3f}", flush=True)

    # ---- held-out chains, EXACTLY the trainer's game split ------------------------
    _chg = np.load(args.chain_graphs, mmap_mode="r")
    CH_KEYS = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att", "played")
    CHA = {k: _chg[k] for k in CH_KEYS}
    CH_prow = np.asarray(_chg["p_row"])
    cs_ = np.load(args.chain_starts)
    starts, games = cs_["start"], cs_["game"]
    pos = np.searchsorted(CH_prow, starts[:, None] + np.arange(7)[None, :])
    okc = (pos < len(CH_prow))
    okc &= CH_prow[np.minimum(pos, len(CH_prow) - 1)] == (starts[:, None] + np.arange(7))
    keep = okc.all(1)
    CH_pos = pos[keep]; games = games[keep]
    gset = np.unique(games)
    rngc = np.random.default_rng(int(A["seed"]) + 7)
    te_g = set(gset[rngc.random(len(gset)) < 0.08].tolist())
    ch_te = np.flatnonzero(np.array([g in te_g for g in games]))
    print(f"[local] {len(ch_te):,} held-out chains; probing {args.n_chains}", flush=True)

    # labels for DECISION fidelity (Kaveh 2026-08-28: "fidelity should be determined
    # by centipawn loss in the end" -- regret in win-prob units, the house metric;
    # mis-reconstructing pieces irrelevant to the action costs nothing here)
    _zv = np.load(args.chain_values)
    CHW = _zv["wp"].astype(np.float32)
    _wdl_raw = _zv["wdl"].astype(np.float32)
    cf = np.load(args.chain_fens)
    CH_STM = np.array([str(f).split()[1] == "w" for f in cf["fen"]])
    CH_SGN = np.where(CH_STM, 1.0, -1.0).astype(np.float32)
    CH_WDL = _wdl_raw.copy()
    CH_WDL[~CH_STM] = CH_WDL[~CH_STM][:, ::-1]
    CH_OK = np.isfinite(CH_WDL).all(-1) & (np.nan_to_num(_wdl_raw).sum(-1) > 0)
    S3 = np.array([1.0, 0.5, 0.0])

    N_NODES = 64 + MV_CAP

    def feat_chain(rows):
        R = {k: torch.from_numpy(np.ascontiguousarray(CHA[k][rows]))
             for k in CH_KEYS if k != "played"}
        n = len(rows)
        b = {k: R[k].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev, n_nodes=N_NODES)
        off = (torch.arange(n, device=dev) * N_NODES)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a2 = R[name].to(dev).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        played = torch.from_numpy(np.ascontiguousarray(CHA["played"][rows])
                                  .astype(np.int64)).to(dev)
        return sqf, mvf, edges, mask, played

    HOPS = [int(h) for h in args.hops.split(",")]
    STEPS = ((0, 2), (2, 4), (4, 6))    # (source row, landing ply): the MOVE unit
    LANDS = (2, 4, 6)
    hist = np.zeros(6)            # full-T flips by hop distance 0,1,2,3,>3,unreached
    n_flip = 0
    agree = {(h, m): np.zeros(3) for h in HOPS for m in ("update", "attn")}
    agree_n = 0
    act_frac = {h: [0.0, 0] for h in HOPS}
    corner = {"local": np.zeros(3), "full": np.zeros(3), "n": 0}   # vs TRUE bits
    DEC = ["full"] + [f"h{h}" for h in HOPS]
    vc = {s: {k: ([], []) for k in LANDS} for s in DEC}
    brg = {s: {k: [0.0, 0] for k in LANDS} for s in DEC}
    br_rnd = [0.0, 0]
    ero = np.random.default_rng(3)
    BK = 8

    bt = model.bit_trans

    def T_step(z, m, f_, t_, mask_=None):
        if GR:
            lg, _, _ = bt(z, m, f_, t_, tok_mask=mask_)
        else:
            lg, _ = bt(z, m, tok_mask=mask_)
        zb = torch.sign(lg)
        return torch.where(zb == 0, torch.ones_like(zb), zb)

    def roll_full(z0, packs):
        outs, z = [], z0
        for m, f_, t_ in packs:
            z = T_step(z, m, f_, t_)
            outs.append(z)
        return outs

    def roll_local(z0, packs, act_seq, attn=False):
        outs, z = [], z0
        for (m, f_, t_), act in zip(packs, act_seq):
            zb = T_step(z, m, f_, t_, mask_=act if attn else None)
            z = torch.where(act[..., None], zb, z)
            outs.append(z)
        return outs

    with torch.no_grad():
        for a in range(0, min(args.n_chains, len(ch_te)), 32):
            ci = CH_pos[ch_te[a:a + 32]]
            cb = len(ci)
            rows_all = ci.ravel()
            sqa, mva, eda, mka, pla = feat_chain(rows_all)
            c_all, h_all, _ = model.carrier(sqa, mva, eda, mka)
            q1 = model.rvq_bits(c_all)[0].reshape(cb, 7, 64, model.sq_bits)
            dq = model.dyn_mv_lfq(model.dyn_mv_proj(h_all))[0]
            mvc = dq[torch.arange(cb * 7, device=dev), pla].reshape(cb, 7, -1)
            mv_raw = CHA["mv"][rows_all].reshape(cb, 7, MV_CAP, -1)
            pl_np = CHA["played"][rows_all].reshape(cb, 7).astype(np.int64)
            frm = mv_raw[np.arange(cb)[:, None], np.arange(7)[None, :], pl_np, 1]
            to_ = mv_raw[np.arange(cb)[:, None], np.arange(7)[None, :], pl_np, 2]
            frm_t = torch.from_numpy(frm.astype(np.int64)).to(dev)
            to_t = torch.from_numpy(to_.astype(np.int64)).to(dev)
            att_np = CHA["att"][rows_all].reshape(cb, 7, -1, 2)
            n_att = CHA["n_att"][rows_all].reshape(cb, 7).astype(np.int64)
            packs = [(mvc[:, s], frm_t[:, s], to_t[:, s]) for s, _ in STEPS]
            # full roll + flip locality (cone measured from MY move; reply effects
            # are marginalised into the step, honest per the move unit) ------------
            full = roll_full(q1[:, 0], packs)
            prev = q1[:, 0]
            for i, (s_, land) in enumerate(STEPS):
                fl = (full[i] != prev).any(-1).cpu().numpy()
                prev = full[i]
                for r in range(cb):
                    pairs = att_np[r, s_][:n_att[r, s_]]
                    pairs = pairs[(pairs != 255).all(-1)]
                    seeds = [int(frm[r, s_]), int(to_[r, s_])]
                    d = np.full(64, 5)
                    d[seeds] = 0
                    for hh in (1, 2, 3):
                        grown = halo(pairs, seeds, hh)
                        d[grown & (d == 5)] = hh
                    d[(d == 5) & halo(pairs, seeds, 64)] = 4
                    for sq_i in np.flatnonzero(fl[r]):
                        hist[d[sq_i]] += 1
                        n_flip += 1
            # local rolls ----------------------------------------------------------
            loc_states = {}
            for h in HOPS:
                act = []
                for s_, _ in STEPS:
                    m_ = np.zeros((cb, 64), bool)
                    for r in range(cb):
                        pairs = att_np[r, s_][:n_att[r, s_]]
                        pairs = pairs[(pairs != 255).all(-1)]
                        m_[r] = halo(pairs, [int(frm[r, s_]), int(to_[r, s_])], h)
                    act.append(torch.from_numpy(m_).to(dev))
                    act_frac[h][0] += float(m_.mean()); act_frac[h][1] += 1
                for mode in ("update", "attn"):
                    loc = roll_local(q1[:, 0], packs, act, attn=(mode == "attn"))
                    if mode == "update":
                        loc_states[h] = loc
                    for i in range(3):
                        agree[(h, mode)][i] += float(
                            (loc[i] == full[i]).float().mean()) * cb
            agree_n += cb
            # fixed corner (h=2 @root move), tracked to plies 2/4/6 ----------------
            m0 = np.zeros((cb, 64), bool)
            for r in range(cb):
                pairs = att_np[r, 0][:n_att[r, 0]]
                pairs = pairs[(pairs != 255).all(-1)]
                m0[r] = halo(pairs, [int(frm[r, 0]), int(to_[r, 0])], 2)
            m0t = torch.from_numpy(m0).to(dev)
            locf = roll_local(q1[:, 0], packs, [m0t] * 3, attn=False)
            for i, (_, land) in enumerate(STEPS):
                true_b = q1[:, land]
                cm = m0t[..., None].expand_as(true_b)
                corner["local"][i] += float(locf[i][cm].eq(true_b[cm]).float().mean()) * cb
                corner["full"][i] += float(full[i][cm].eq(true_b[cm]).float().mean()) * cb
            corner["n"] += cb
            # DECISION fidelity (regret currency; branch step always full-T) -------
            W = model.readout.weight
            samp = {}
            for kf, land in STEPS:
                wpk = CHW[ci[:, kf]]
                cols = np.zeros((cb, BK), np.int64); ok_r = np.zeros(cb, bool)
                for r in range(cb):
                    v = np.flatnonzero(np.isfinite(wpk[r]))
                    if len(v) < 2:
                        continue
                    b_ = v[wpk[r][v].argmax()]; rest = v[v != b_]
                    cols[r, 0] = b_
                    cols[r, 1:] = ero.choice(rest, BK - 1, replace=len(rest) < BK - 1)
                    ok_r[r] = True
                samp[kf] = (cols, ok_r, wpk)
            jit = {kf: ero.random((int(samp[kf][1].sum()), BK)) * 1e-9 for kf in samp}
            states = {"full": full, **{f"h{h}": loc_states[h] for h in HOPS}}
            for srcn, zs_roll in states.items():
                for i, (kf, land) in enumerate(STEPS):
                    fin = CH_OK[ci[:, land]]
                    if fin.any():
                        zf = model.levels_from_bits(zs_roll[i])[0]
                        pv = torch.nn.functional.linear(
                            zf, W[:, :model.d_pos], model.readout.bias) + model.turn[
                            torch.from_numpy(
                                CH_STM[ci[:, land]].astype(np.int64)).to(dev)]
                        p = (torch.softmax(pv, -1).cpu().numpy() @ S3)[fin]
                        tw = CH_WDL[ci[:, land]][fin]
                        tw = tw / tw.sum(-1, keepdims=True).clip(min=1e-6)
                        vc[srcn][land][0].append(p)
                        vc[srcn][land][1].append(tw @ S3)
                    cols, ok_r, wpk = samp[kf]
                    if not ok_r.any():
                        continue
                    okt = torch.from_numpy(ok_r).to(dev); nv = int(ok_r.sum())
                    colst = torch.from_numpy(cols[ok_r]).to(dev)
                    codes = dq.reshape(cb, 7, MV_CAP, -1)[:, kf][okt].gather(
                        1, colst[..., None].expand(-1, -1, dq.shape[-1])
                        ).reshape(nv * BK, -1)
                    fb = mv_raw[:, kf][ok_r][np.arange(nv)[:, None], cols[ok_r], 1]
                    tb = mv_raw[:, kf][ok_r][np.arange(nv)[:, None], cols[ok_r], 2]
                    fbt = torch.from_numpy(fb.astype(np.int64)).to(dev).reshape(-1)
                    tbt = torch.from_numpy(tb.astype(np.int64)).to(dev).reshape(-1)
                    base = (q1[:, 0] if kf == 0 else zs_roll[i - 1])[okt]
                    base = base.reshape(nv, 1, 64, -1).expand(-1, BK, -1, -1
                                                              ).reshape(nv * BK, 64, -1)
                    zbr = T_step(base, codes, fbt, tbt)
                    zf = model.levels_from_bits(zbr)[0]
                    pvb = torch.nn.functional.linear(
                        zf, W[:, :model.d_pos], model.readout.bias)
                    stm_c = torch.from_numpy(
                        CH_STM[ci[:, kf]].astype(np.int64)[ok_r]).to(dev)
                    pvb = pvb + model.turn[stm_c.repeat_interleave(BK)]
                    sg = CH_SGN[ci[:, kf]][ok_r][:, None]
                    sc = sg * (pvb[:, 0] - pvb[:, 2]).reshape(nv, BK).cpu().numpy()
                    wsel = wpk[np.arange(cb)[ok_r][:, None], cols[ok_r]]
                    pick = (sc + jit[kf]).argmax(1)
                    reg = (wsel[:, 0] - wsel[np.arange(nv), pick]).clip(min=0)
                    brg[srcn][land][0] += float(reg.sum()); brg[srcn][land][1] += nv
                    if srcn == "full":
                        rr = (wsel[:, :1] - wsel).clip(min=0)
                        br_rnd[0] += float(rr[:, 1:].mean(1).sum()); br_rnd[1] += nv

    print(f"\n[local] FULL-T FLIP LOCALITY ({n_flip:,} square-flips over 3 move-"
          f"steps; cone from MY move, reply marginalised): "
          f"hop0 {hist[0]/max(n_flip,1):.1%}  hop1 {hist[1]/max(n_flip,1):.1%}  "
          f"hop2 {hist[2]/max(n_flip,1):.1%}  hop3 {hist[3]/max(n_flip,1):.1%}  "
          f">3 {hist[4]/max(n_flip,1):.1%}  unreached {hist[5]/max(n_flip,1):.1%}")
    print(f"[local] LOCAL-vs-FULL bit agreement at plies 2/4/6:")
    for h in HOPS:
        af = act_frac[h][0] / max(act_frac[h][1], 1)
        for mode in ("update", "attn"):
            v = agree[(h, mode)] / max(agree_n, 1)
            print(f"  h={h} {mode:6s} (active {af:.1%} of board): " +
                  "/".join(f"{x:.3f}" for x in v))
    cl = corner["local"] / max(corner["n"], 1)
    cfl = corner["full"] / max(corner["n"], 1)
    print(f"[local] FIXED CORNER (h=2 @root move) vs TRUE bits, plies 2/4/6:")
    print(f"  local  " + "/".join(f"{x:.3f}" for x in cl))
    print(f"  full   " + "/".join(f"{x:.3f}" for x in cfl))
    print(f"[local] DECISION FIDELITY (Kaveh: fidelity = regret in win-prob; branch "
          f"step always full-T, the LINE is rolled per source):")
    for srcn in DEC:
        vs = []
        for k in LANDS:
            if vc[srcn][k][0]:
                pp = np.concatenate(vc[srcn][k][0]); tt = np.concatenate(vc[srcn][k][1])
                vs.append(float(np.corrcoef(pp, tt)[0, 1])
                          if pp.std() > 1e-9 and tt.std() > 1e-9 else 0.0)
            else:
                vs.append(float("nan"))
        bs = [brg[srcn][k][0] / max(brg[srcn][k][1], 1) for k in LANDS]
        print(f"  {srcn:5s} vroll " + "/".join(f"{v:+.3f}" for v in vs) +
              "  brreg " + "/".join(f"{b:.3f}" for b in bs))
    print(f"  random-pick baseline {br_rnd[0] / max(br_rnd[1], 1):.3f}")
    print("[local] VERDICT inputs printed above -- interpretation belongs to the "
          "journal, not this script.")


def _units():
    torch.manual_seed(0)
    from catspace.research.tools.training_infra.qgt_field import BitTransition
    # halo: hand-built graph 0-1-2-3 chain + isolated 63
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    m = halo(pairs, [0], 1)
    assert m[0] and m[1] and not m[2] and not m[63], "1-hop halo wrong"
    m2 = halo(pairs, [0], 3)
    assert m2[:4].all() and not m2[4:].any(), "3-hop halo wrong"
    bt = BitTransition(sq_bits=8, mv_bits=8)
    with torch.no_grad():                       # break identity so updates matter
        bt.head[-1].weight.add_(2.0 * torch.randn_like(bt.head[-1].weight))
    z0 = torch.sign(torch.randn(4, 64, 8)); z0[z0 == 0] = 1.0
    mv = [torch.sign(torch.randn(4, 8)) for _ in range(3)]
    act = [torch.zeros(4, 64, dtype=torch.bool) for _ in range(3)]
    for a_ in act:
        a_[:, :8] = True
    with torch.no_grad():
        loc = local_roll(bt, z0, mv, act)
        full = bt.roll_bits(z0, mv)
    frozen = (loc[-1][:, 8:] == z0[:, 8:]).all()
    assert bool(frozen), "inactive squares changed under a local roll"
    moved = (full[-1] != z0).float().mean()
    assert float(moved) > 0.05, "perturbed T did not move bits -- test is vacuous"
    with torch.no_grad():
        allact = [torch.ones(4, 64, dtype=torch.bool)] * 3
        la = local_roll(bt, z0, mv, allact, attn=True)
    assert bool((la[-1] == full[-1]).all()), \
        "all-active attn-masked roll differs from the full roll"
    print("  halo BFS exact on a hand graph; inactive squares provably frozen; "
          "all-active attn mode == full roll; perturbed T moves bits")
    print("  LOCAL ROLL TESTS PASS")


if __name__ == "__main__":
    main()
