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
    assert A.get("dynamics") == "discrete", \
        f"{args.ckpt} is not a discrete-arm checkpoint (dynamics={A.get('dynamics')})"
    model = QGTField(d=A["d"], layers=A["layers"], sq_bits=A["sq_bits"],
                     rvq_levels=A["rvq_levels"], mv_bits=A["mv_bits"],
                     qa_levels=A["qa_levels"], n_q=A["n_q"], qa_heads=A["attn_heads"],
                     top_k=A["top_k"], out_bits=A["out_bits"])
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
    # accumulators -----------------------------------------------------------------
    hist = np.zeros(6)            # full-T flips by hop distance 0,1,2,3,>3,unreached
    n_flip = 0
    agree = {(h, m): np.zeros(6) for h in HOPS for m in ("update", "attn")}
    agree_n = 0
    act_frac = {h: [0.0, 0] for h in HOPS}
    corner = {"local": np.zeros(6), "full": np.zeros(6), "n": 0}   # vs TRUE bits

    bt = model.bit_trans
    with torch.no_grad():
        for a in range(0, min(args.n_chains, len(ch_te)), 32):
            ci = CH_pos[ch_te[a:a + 32]]
            cb = len(ci)
            # encode all 7 true rows: bits + dyn codes + move meta
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
            att_np = CHA["att"][rows_all].reshape(cb, 7, -1, 2)
            n_att = CHA["n_att"][rows_all].reshape(cb, 7).astype(np.int64)
            moves = [mvc[:, k] for k in range(6)]
            # full roll + flip locality --------------------------------------------
            full = bt.roll_bits(q1[:, 0], moves)
            prev = q1[:, 0]
            for k in range(6):
                fl = (full[k] != prev).any(-1).cpu().numpy()      # (cb,64) sq flipped
                prev = full[k]
                for r in range(cb):
                    pairs = att_np[r, k][:n_att[r, k]]
                    pairs = pairs[(pairs != 255).all(-1)]
                    seeds = [int(frm[r, k]), int(to_[r, k])]
                    d = np.full(64, 5)                            # 5 = unreached
                    d[seeds] = 0
                    for hh in (1, 2, 3):
                        grown = halo(pairs, seeds, hh)
                        d[grown & (d == 5)] = hh
                    d[(d == 5) & halo(pairs, seeds, 64)] = 4      # reachable, >3
                    for sq_i in np.flatnonzero(fl[r]):
                        hist[d[sq_i]] += 1
                        n_flip += 1
            # local rolls ----------------------------------------------------------
            for h in HOPS:
                act = []
                for k in range(6):
                    m_ = np.zeros((cb, 64), bool)
                    for r in range(cb):
                        pairs = att_np[r, k][:n_att[r, k]]
                        pairs = pairs[(pairs != 255).all(-1)]
                        m_[r] = halo(pairs, [int(frm[r, k]), int(to_[r, k])], h)
                    act.append(torch.from_numpy(m_).to(dev))
                    act_frac[h][0] += float(m_.mean()); act_frac[h][1] += 1
                for mode in ("update", "attn"):
                    loc = local_roll(bt, q1[:, 0], moves, act, attn=(mode == "attn"))
                    for k in range(6):
                        agree[(h, mode)][k] += float(
                            (loc[k] == full[k]).float().mean()) * cb
            agree_n += cb
            # fixed-corner tracking (h=2 corner at ply 0, update mode) -------------
            m0 = np.zeros((cb, 64), bool)
            for r in range(cb):
                pairs = att_np[r, 0][:n_att[r, 0]]
                pairs = pairs[(pairs != 255).all(-1)]
                m0[r] = halo(pairs, [int(frm[r, 0]), int(to_[r, 0])], 2)
            m0t = torch.from_numpy(m0).to(dev)
            loc = local_roll(bt, q1[:, 0], moves, [m0t] * 6, attn=False)
            for k in range(6):
                true_b = q1[:, k + 1]
                cm = m0t[..., None].expand_as(true_b)
                corner["local"][k] += float(loc[k][cm].eq(true_b[cm]).float().mean()) * cb
                corner["full"][k] += float(full[k][cm].eq(true_b[cm]).float().mean()) * cb
            corner["n"] += cb

    print(f"\n[local] FULL-T FLIP LOCALITY ({n_flip:,} square-flips over 6 plies): "
          f"hop0 {hist[0]/max(n_flip,1):.1%}  hop1 {hist[1]/max(n_flip,1):.1%}  "
          f"hop2 {hist[2]/max(n_flip,1):.1%}  hop3 {hist[3]/max(n_flip,1):.1%}  "
          f">3 {hist[4]/max(n_flip,1):.1%}  unreached {hist[5]/max(n_flip,1):.1%}")
    print(f"[local] LOCAL-vs-FULL bit agreement by horizon (1..6):")
    for h in HOPS:
        af = act_frac[h][0] / max(act_frac[h][1], 1)
        for mode in ("update", "attn"):
            v = agree[(h, mode)] / max(agree_n, 1)
            print(f"  h={h} {mode:6s} (active {af:.1%} of board): " +
                  "/".join(f"{x:.3f}" for x in v))
    cl = corner["local"] / max(corner["n"], 1)
    cf = corner["full"] / max(corner["n"], 1)
    print(f"[local] FIXED CORNER (h=2 @ply0) vs TRUE bits, horizons 1..6:")
    print(f"  local  " + "/".join(f"{x:.3f}" for x in cl))
    print(f"  full   " + "/".join(f"{x:.3f}" for x in cf))
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
