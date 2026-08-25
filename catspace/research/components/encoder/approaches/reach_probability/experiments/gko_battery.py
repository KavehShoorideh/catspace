"""Post-run verdict battery for gko-family checkpoints (gen5).

Sections, each printing labeled verdict lines:
  RANK -- representational-health gate: recomputed phi variance (must reproduce the
          heartbeat's phivar, verifying the phi slice), bootstrapped effective rank
          of phi, participation ratio, per-block state variances.
  WDL  -- k=0 value calibration on chain rows with sidecar oracle triples:
          KL(oracle||pred), 3-outcome CRPS, derived-scalar MAE, stratified by band.
  SHARP-- the sf_av per-move head on UNSEEN sf shard rows (converted straight out of
          the store into a holdout dir, so they never entered any training rotation):
          within-position Spearman (ORDERING) and absolute LEVEL calibration. These
          two dissociate: ordering is scale-free and survives thin data, level is
          what starves first, so reporting both separates "undertrained" from
          "miscalibrated" (2026-08-23 sf-starvation).

Rows come from the trainer's own held-out-by-GAME chain split, reconstructed
bit-exactly (seeded RNG; hard-asserted against the trainer's printed count) with
true per-row elo conditioning -- so WDL numbers are genuine held-out.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catspace.research.tools.npz_io import load_npz


def build(ckpt_path, dev):
    from catspace.research.tools.training_infra.koopman_ops import TransformerLift
    from catspace.research.components.encoder.approaches.reach_probability \
        .experiments.graph_concept_field import GraphConceptField

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    gen5 = bool(d.get("gen5"))
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
        mh_vq=bool(d["g_heads"]), gen5=gen5).to(dev)
    lift = TransformerLift(field.d_state, g_heads=A["g_heads"], d_code=A["d_code"],
                           d_sq=(16 if gen5 else 32), d_phi=max(d["d_phi"], 256),
                           n_layers=d["lift_layers"],
                           n_piece=(32 if gen5 else 0)).to(dev)
    # width from the checkpoint's own head: gen5 is 3-way WDL, gen4 is 64-bin HL
    _vb = ck["state_dict"].get("sf_v_full.2.bias")
    vb = _vb.shape[0] if _vb is not None else 3
    sf_v_full = nn.Sequential(nn.Linear(lift.d_aug, 256), nn.GELU(),
                              nn.Linear(256, vb)).to(dev)
    kb = ck["state_dict"]["sf_av.2.bias"].shape[0]
    sf_av = nn.Sequential(nn.Linear(field.d_state + A["d_model"], 512), nn.GELU(),
                          nn.Linear(512, kb)).to(dev)

    sd = ck["state_dict"]
    for mod, pre in ((field, "field."), (lift, "lift."),
                     (sf_v_full, "sf_v_full."), (sf_av, "sf_av.")):
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        if not sub:
            continue
        missing, unexpected = mod.load_state_dict(sub, strict=False)
        assert not missing and not unexpected, (pre, missing, unexpected)
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    return ck, d, A, field, lift, sf_v_full, sf_av, kb, gen5, vb


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


def sharp_section(args, field, sf_av, kb, gen5, dev, normalise_elo,
                  lift=None, sf_v_full=None, vb=3):
    """sf_av on UNSEEN sf rows: ORDERING (scale-free, survives thin data) vs
    absolute LEVEL (what starves first). Reporting both separates 'undertrained'
    from 'miscalibrated'. Works for gen4 and gen5 checkpoints."""
    # ---- SHARP (unseen sf shard rows; ordering vs level) ----
    import glob as _glob
    from catspace.research.tools.training_infra.hl_gauss import HLGauss
    from catspace.research.components.encoder.approaches.reach_probability \
        .experiments.graph_move_policy import featurize as _ftz, MV_CAP
    hf = sorted(_glob.glob(os.path.join(args.sharp_dir, "*.npz")))
    if hf:
        hl = HLGauss(k=kb)
        hl.centers = hl.centers.to(dev)
        sps, lvl_p, lvl_t = [], [], []
        sv_p, sv_t = [], []
        for f in hf:
            H = load_npz(f)
            for i0 in range(0, len(H["kind"]), 256):
                rr = np.arange(i0, min(i0 + 256, len(H["kind"])))
                b = {k: torch.from_numpy(np.ascontiguousarray(H[k][rr])).to(dev)
                     for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
                sqf, mvf, edges, mask = _ftz(b, dev, n_nodes=field.n_nodes)
                off = (torch.arange(len(rr), device=dev) * field.n_nodes)[:, None]
                jj = torch.arange(MV_CAP, device=dev)[None, :]
                for nm in ("vac", "dsc"):
                    a2 = torch.from_numpy(np.ascontiguousarray(H[nm][rr])).to(dev).long()
                    val = (a2 != 255) & mask[:, :, None]
                    edges[nm] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                                 (a2 + off[:, :, None])[val])
                elo = torch.from_numpy(normalise_elo(
                    np.full((len(rr), 2), 2750.0, np.float32))).to(dev)
                psl = (torch.from_numpy(np.ascontiguousarray(H["pslot"][rr])).to(dev)
                       if gen5 else None)
                with torch.no_grad():
                    zq, _, _, _, h_mv, _ = field.encode(sqf, mvf, edges, mask,
                                                        elo=elo, pslot=psl)[:6]
                    x = torch.cat([zq[:, None, :].expand(-1, MV_CAP, -1), h_mv], -1)
                    pv = hl.decode(sf_av(x)).cpu().numpy()
                    svv_batch = None
                    if lift is not None and sf_v_full is not None:
                        q, _, _ = field.mv_vq(field.mv_vq_proj(h_mv))
                        za = lift(zq, q * mask[..., None])
                        lg = sf_v_full(za)
                        if vb == 3:
                            pr = torch.softmax(lg, -1)
                            svv = (pr[:, 0] + 0.5 * pr[:, 1]).cpu().numpy()
                        else:
                            svv = hl.decode(lg).cpu().numpy()
                        svv_batch = svv
                tg = H["tgt"][rr]
                fin = np.isfinite(tg) & (tg > -1e8)
                for r in range(len(rr)):
                    m = fin[r]
                    if m.sum() > 3:
                        a = np.argsort(np.argsort(pv[r][m]))
                        b2 = np.argsort(np.argsort(tg[r][m]))
                        c = np.corrcoef(a, b2)[0, 1]
                        if np.isfinite(c):
                            sps.append(c)
                        lvl_p.append(pv[r][m].max()); lvl_t.append(tg[r][m].max())
                        if svv_batch is not None:
                            # append PRED and TARGET together, inside the same filter --
                            # accumulating preds per-batch and targets per-row silently
                            # misaligns them at the first skipped row and reads as
                            # corr ~ 0 for a perfectly good head.
                            sv_p.append(float(svv_batch[r]))
                            sv_t.append(float(tg[r][m].max()))
        lp, lt = np.array(lvl_p), np.array(lvl_t)
        print(f"[bat] SHARP n={len(sps)} unseen sf positions: "
              f"ORDERING spearman {np.mean(sps):.3f} "
              f"(gko2i-era held-out ~0.50, gko3a final 0.544)", flush=True)
        print(f"[bat] SHARP LEVEL best-move pred {lp.mean():.3f} vs SF {lt.mean():.3f} "
              f"| bias {lp.mean()-lt.mean():+.3f} | MAE {np.abs(lp-lt).mean():.3f} "
              f"| corr {np.corrcoef(lp, lt)[0,1]:.3f} "
              f"(untethered level = bias large and/or corr low)", flush=True)
        if sv_p:
            # STATE VALUE (sf_v_full) on the same unseen rows. l_sv is what teaches
            # this head material -- it is gated on kind==0, so sf starvation blinds
            # it. SF's state value is max-over-moves, exactly l_sv's target.
            a = np.array(sv_p); b3 = np.array(sv_t)
            assert len(a) == len(b3), (len(a), len(b3))
            print(f"[bat] VALUE sf_v_full on {len(a)} unseen positions: "
                  f"corr {np.corrcoef(a, b3)[0,1]:.3f} | MAE {np.abs(a-b3).mean():.3f} "
                  f"| pred sd {a.std():.3f} vs SF sd {b3.std():.3f} "
                  f"(ratio {a.std()/max(b3.std(),1e-9):.2f}; <<1 = flat/material-blind)",
                  flush=True)
    else:
        print(f"[bat] SHARP skipped: no shards in {args.sharp_dir}", flush=True)


def ops_section(ckpt, sharp_dir, n_pos=48):
    """Operator move-CONDITIONING on real positions: split K(m)z into the shared
    part (A z, identical for every move) and the move-dependent part. If the
    move-dependent term is a tiny fraction of the shared one, every child lands in
    nearly the same place and NO downstream head can separate siblings, however
    well calibrated it is (2026-08-23: gen5 measured 1.3pct).
    Reuses KoopmanEngine so both generations load through the same tested path."""
    import glob as _g, chess
    from catspace.research.components.planner.approaches.quasimetric_nav \
        .koopman_search import KoopmanEngine
    ke = KoopmanEngine(ckpt, reencode_budget=12)
    fens = []
    for f in sorted(_g.glob(os.path.join(sharp_dir, "*.npz"))):
        d = load_npz(f)
        fens.extend([x.decode() if isinstance(x, bytes) else str(x)
                     for x in d["fen"][:n_pos]])
        if len(fens) >= n_pos:
            break
    sh_l, md_l, sep_l, nrm_l = [], [], [], []
    for fen in fens[:n_pos]:
        try:
            bd = chess.Board(fen)
        except Exception:
            continue
        if bd.is_game_over() or not list(bd.legal_moves):
            continue
        za, cond, moves, _ = ke._encode(bd)
        if len(moves) < 2:
            continue
        with torch.no_grad():
            zin = za.expand(len(cond), -1)
            if ke.gen5:
                zac = ke.ops(zin, cond[:, 0], cond[:, 1], cond[:, 2])
                shared = zin @ ke.ops.A.T
                md = zac - shared
                sh_l.append(float(shared.norm(dim=-1).mean()))
                md_l.append(float(md.norm(dim=-1).mean()))
            else:
                zac = ke.ops(zin, cond)
            sep_l.append(float(torch.cdist(zac, zac).mean()))
            nrm_l.append(float(zac.norm(dim=-1).mean()))
    if sh_l:
        sh, md = float(np.mean(sh_l)), float(np.mean(md_l))
        print(f"[bat] OPS n={len(sep_l)} real positions: shared ||Az|| {sh:.2f} | "
              f"move-dep ||K z - Az|| {md:.4f} | RATIO {md/max(sh,1e-9):.5f} "
              f"(gko3a@8k baseline 0.0127; higher = more move-conditioning)",
              flush=True)
    print(f"[bat] OPS child separation {float(np.mean(sep_l)):.4f} "
          f"(rel. to |z| {float(np.mean(sep_l))/max(float(np.mean(nrm_l)),1e-9):.5f}) "
          f"| incumbent gko2i 1.2322 on midgame", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data/derived/gko3_chain_graphs_mm.npz")
    ap.add_argument("--sidecar", default="data/derived/gko1_chain_values.npz")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--state_boot", type=int, default=200)
    ap.add_argument("--sharp_dir", default="data/derived/gko1_holdout_sf")
    ap.add_argument("--ops_only", action="store_true",
                    help="operator move-conditioning decomposition only")
    ap.add_argument("--sharp_only", action="store_true",
                    help="SHARP section only -- works for gen4 and gen5 checkpoints")
    args = ap.parse_args()
    dev = ("mps" if torch.backends.mps.is_available() else "cpu")

    if args.ops_only:
        ops_section(args.ckpt, args.sharp_dir)
        return
    ck, d, A, field, lift, sf_v_full, sf_av, kb, gen5, vb = build(args.ckpt, dev)
    print(f"[bat] {args.ckpt} step {ck['step']} | d_state {field.d_state} "
          f"d_aug {lift.d_aug} | dev {dev}", flush=True)

    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo
    if args.sharp_only:
        sharp_section(args, field, sf_av, kb, gen5, dev, normalise_elo,
                      lift, sf_v_full, vb)
        return
    CH = load_npz(args.cache, mmap_mode="r")
    CH_prow = np.asarray(CH["p_row"])
    cs = load_npz(d["chain_starts"])
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
    sharp_section(args, field, sf_av, kb, gen5, dev, normalise_elo,
                  lift, sf_v_full, vb)

    SC = load_npz(args.sidecar)
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
