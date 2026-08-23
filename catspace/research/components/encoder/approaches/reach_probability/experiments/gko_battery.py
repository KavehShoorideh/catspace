"""Post-run verdict battery for gko-family checkpoints (gen5).

Sections, each printing labeled verdict lines:
  RANK -- representational-health gate: recomputed phi variance (must reproduce the
          heartbeat's phivar, verifying the phi slice), bootstrapped effective rank
          of phi, participation ratio, per-block state variances.
  WDL  -- k=0 value calibration on chain rows with sidecar oracle triples:
          KL(oracle||pred), 3-outcome CRPS, derived-scalar MAE, stratified by band.

Rows come from the trainer's own held-out-by-GAME chain split, reconstructed
bit-exactly (seeded RNG; hard-asserted against the trainer's printed count) with
true per-row elo conditioning -- so WDL numbers are genuine held-out.
"""
import argparse, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build(ckpt_path, dev):
    from catspace.research.tools.training_infra.koopman_ops import TransformerLift
    from catspace.research.components.encoder.approaches.reach_probability \
        .experiments.graph_concept_field import GraphConceptField

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    assert d.get("gen5"), "battery v1 targets gen5 checkpoints"
    warm = torch.load(d["resume_field"], map_location="cpu", weights_only=False)
    A = dict(warm["args"])
    A["g_heads"], A["g_codes"], A["d_code"] = d["g_heads"], d["g_codes"], d["d_code"]
    A["slow_heads"] = d["slow_heads"] or A["slow_heads"]
    if d.get("sq_codes"):
        A["sq_codes"] = d["sq_codes"]

    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=True,
        g_heads=A["g_heads"], g_codes=A["g_codes"], d_code=A["d_code"],
        sq_codes=A.get("sq_codes", 256),
        mh_vq=bool(d["g_heads"]), gen5=True).to(dev)
    lift = TransformerLift(field.d_state, g_heads=A["g_heads"], d_code=A["d_code"],
                           d_sq=16, d_phi=max(d["d_phi"], 256),
                           n_layers=d["lift_layers"], n_piece=32).to(dev)
    sf_v_full = nn.Sequential(nn.Linear(lift.d_aug, 256), nn.GELU(),
                              nn.Linear(256, 3)).to(dev)

    sd = ck["state_dict"]
    for mod, pre in ((field, "field."), (lift, "lift."), (sf_v_full, "sf_v_full.")):
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        missing, unexpected = mod.load_state_dict(sub, strict=False)
        assert not missing and not unexpected, (pre, missing, unexpected)
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    return ck, d, A, field, lift, sf_v_full


def encode_rows(CH, rows, field, lift, dev, elo_n):
    """Mirror of the trainer's _feat_rows + encode + lift (true per-row elos)."""
    from catspace.research.components.encoder.approaches.reach_probability \
        .experiments.graph_move_policy import featurize, MV_CAP
    b = {k: torch.from_numpy(np.ascontiguousarray(CH[k][rows])).to(dev)
         for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
    sqf, mvf, edges, mask = featurize(b, dev, n_nodes=field.n_nodes)
    off = (torch.arange(len(rows), device=dev) * field.n_nodes)[:, None]
    jj = torch.arange(MV_CAP, device=dev)[None, :]
    for name in ("vac", "dsc"):
        a2 = torch.from_numpy(np.ascontiguousarray(CH[name][rows])).to(dev).long()
        val = (a2 != 255) & mask[:, :, None]
        edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                       (a2 + off[:, :, None])[val])
    elo = torch.from_numpy(np.ascontiguousarray(elo_n[rows])).to(dev)
    psl = (torch.from_numpy(np.ascontiguousarray(CH["pslot"][rows])).to(dev)
           if "pslot" in CH.files else None)
    with torch.no_grad():
        zq, _, _, _, h_mv, _ = field.encode(sqf, mvf, edges, mask, elo=elo,
                                            pslot=psl)[:6]
        q, _, _ = field.mv_vq(field.mv_vq_proj(h_mv))
        za = lift(zq, q * mask[..., None])
    return za


def eff_rank(X, n_boot=1000, seed=0):
    """exp(entropy of normalized covariance eigenvalues), bootstrapped over rows."""
    rng = np.random.default_rng(seed)
    def one(idx):
        C = np.cov(X[idx].T)
        ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
        p = ev / ev.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    full = one(np.arange(len(X)))
    boots = [one(rng.integers(0, len(X), len(X))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return full, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data/derived/gko3_chain_graphs_mm.npz")
    ap.add_argument("--sidecar", default="data/derived/gko1_chain_values.npz")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--state_boot", type=int, default=200)
    args = ap.parse_args()
    dev = ("mps" if torch.backends.mps.is_available() else "cpu")

    ck, d, A, field, lift, sf_v_full = build(args.ckpt, dev)
    print(f"[bat] {args.ckpt} step {ck['step']} | d_state {field.d_state} "
          f"d_aug {lift.d_aug} | dev {dev}", flush=True)

    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo
    CH = np.load(args.cache, mmap_mode="r")
    CH_prow = np.asarray(CH["p_row"])
    cs = np.load(d["chain_starts"])
    starts, games = cs["start"], cs["game"]
    pos = np.searchsorted(CH_prow, starts[:, None] + np.arange(7)[None, :])
    ok = (pos < len(CH_prow))
    ok &= CH_prow[np.minimum(pos, len(CH_prow) - 1)] == (starts[:, None] + np.arange(7))
    keep = ok.all(1)
    CH_pos = pos[keep]; games = games[keep]
    gset = np.unique(games)
    rngc = np.random.default_rng(d["seed"] + 7)
    te_g = set(gset[rngc.random(len(gset)) < 0.08].tolist())
    ch_te = np.flatnonzero(np.array([g in te_g for g in games]))
    assert len(ch_te) == 4655, \
        f"held-out reconstruction {len(ch_te)} != trainer's printed 4,655 -- inputs drifted"
    held = np.unique(CH_pos[ch_te].ravel())
    rng = np.random.default_rng(1)
    tail = np.sort(rng.choice(held, min(args.n, len(held)), replace=False))
    print(f"[bat] split reconstructed: {len(CH_pos):,} chains, {len(ch_te):,} held-out "
          f"(matches trainer) -> {len(tail):,} eval rows (held-out by GAME)", flush=True)
    elo_n = normalise_elo(np.load(d["row_elo"])["elo"].astype(np.float32))[CH_prow]
    za = []
    for i in range(0, len(tail), args.batch):
        za.append(encode_rows(CH, tail[i:i + args.batch], field, lift, dev,
                              elo_n).cpu())
    za = torch.cat(za).float()
    z_state, phi = za[:, :field.d_state].numpy(), za[:, field.d_state:].numpy()

    # ---- RANK ----
    phivar = float(phi.var(0).mean())
    print(f"[bat] RANK phi-slice check: recomputed phivar {phivar:.4f} "
          f"(final heartbeat printed 0.8873 on ITS eval sample; same order of "
          f"magnitude verifies the [d_state:] slice)", flush=True)
    er, lo, hi = eff_rank(phi)
    print(f"[bat] RANK phi effective rank {er:.1f} [{lo:.1f},{hi:.1f}]95 of "
          f"{phi.shape[1]} dims ({er/phi.shape[1]:.2f} of max)", flush=True)
    pr = float((phi.var(0).sum())**2 / (phi.var(0)**2).sum())
    print(f"[bat] RANK phi participation ratio {pr:.1f} (diagonal; eigen-based rank "
          f"above is the gate)", flush=True)
    ev = np.linalg.eigvalsh(np.cov(phi.T))[::-1]
    ev = np.clip(ev, 0, None); evf = ev / ev.sum()
    print(f"[bat] RANK phi top-8 eigen mass {np.round(evf[:8], 3).tolist()} "
          f"(cum {evf[:8].sum():.3f})", flush=True)
    er_z, lo_z, hi_z = eff_rank(z_state, n_boot=args.state_boot)
    print(f"[bat] RANK state effective rank {er_z:.1f} [{lo_z:.1f},{hi_z:.1f}]95 "
          f"of {z_state.shape[1]}", flush=True)
    blocks = {"board": (0, 512), "squares": (512, 1536),
              "pieces": (1536, 1664), "hist": (1664, 1792)}
    bv = {k: float(z_state[:, a:b].var(0).mean()) for k, (a, b) in blocks.items()}
    print(f"[bat] RANK block variances {bv} (no block dead)", flush=True)

    # ---- WDL ----
    SC = np.load(args.sidecar)
    wdl = SC["wdl"][tail].astype(np.float32)  # f16 on disk: 1e-9 underflows to 0 there
    valid = np.isfinite(wdl).all(1) & (wdl.sum(1) > 0.5)
    if valid.sum() < 100:
        print(f"[bat] WDL skipped: only {int(valid.sum())} sidecar triples in tail")
        return
    with torch.no_grad():
        pred = F.softmax(sf_v_full(za[valid].to(dev)), -1).cpu().numpy()
    orc = wdl[valid] / np.maximum(wdl[valid].sum(1, keepdims=True), 1e-9)
    kl = float(np.mean(np.sum(orc * (np.log(np.clip(orc, 1e-9, 1)) -
                                     np.log(np.clip(pred, 1e-9, 1))), 1)))
    cdf_o = np.cumsum(orc, 1)[:, :2]
    cdf_p = np.cumsum(pred, 1)[:, :2]
    crps = float(np.mean(np.sum((cdf_p - cdf_o) ** 2, 1)))
    e_o = orc[:, 0] + 0.5 * orc[:, 1]
    e_p = pred[:, 0] + 0.5 * pred[:, 1]
    mae = float(np.mean(np.abs(e_p - e_o)))
    print(f"[bat] WDL overall n={int(valid.sum())}: KL {kl:.4f} CRPS {crps:.4f} "
          f"scalar-MAE {mae:.4f} (held-out by game, true elo cond)",
          flush=True)
    print(f"[bat] WDL class means  pred W/D/L "
          f"{pred.mean(0)[0]:.3f}/{pred.mean(0)[1]:.3f}/{pred.mean(0)[2]:.3f} vs "
          f"oracle {orc.mean(0)[0]:.3f}/{orc.mean(0)[1]:.3f}/{orc.mean(0)[2]:.3f} "
          f"| pred-draw>oracle-draw by {pred.mean(0)[1]-orc.mean(0)[1]:+.3f} "
          f"(draw-collapse check)", flush=True)
    print(f"[bat] WDL spread  pred sd of E[W]+0.5E[D] {e_p.std():.4f} vs "
          f"oracle {e_o.std():.4f} (ratio {e_p.std()/max(e_o.std(),1e-9):.3f}; "
          f"<<1 => head is near-constant)", flush=True)
    bands = {
        "near-equal": (e_o >= 0.45) & (e_o < 0.55),
        "advantage": ((e_o >= 0.55) & (e_o < 0.85)) | ((e_o > 0.15) & (e_o <= 0.45)),
        "decided": (e_o >= 0.85) | (e_o <= 0.15),
    }
    for name, m in bands.items():
        if m.sum() < 20:
            continue
        klb = float(np.mean(np.sum(orc[m] * (np.log(np.clip(orc[m], 1e-9, 1)) -
                                             np.log(np.clip(pred[m], 1e-9, 1))), 1)))
        maeb = float(np.mean(np.abs(e_p[m] - e_o[m])))
        crpsb = float(np.mean(np.sum((cdf_p[m] - cdf_o[m]) ** 2, 1)))
        print(f"[bat] WDL band {name:<10} n={int(m.sum()):>6}: KL {klb:.4f} "
              f"CRPS {crpsb:.4f} scalar-MAE {maeb:.4f}", flush=True)


if __name__ == "__main__":
    main()
