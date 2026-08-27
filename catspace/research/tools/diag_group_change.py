"""How many CONCEPT GROUPS change per 2 plies? The control for --w-slow.

Kaveh 2026-08-26: "we want the fewest number of concept changes per ply possible in order
for us to be able to correctly predict the evaluation."

qat5 trains with the group fused lasso and prints `moved N/97` in its heartbeat, but qat4
does not -- it ran without --w-slow, so the flag that computes the statistic was off. That
leaves the comparison unmatched. This scores ANY checkpoint on the same statistic, at equal
training length, so "the regulariser reduced group churn" is a measurement rather than a
guess.

Groups: 64 squares + 32 piece slots (alive at both ends) + 1 edge histogram = 97.
Reported at EVEN lag (2 ply), because at odd lag the side-to-move flip moves ~54% of bits
for representational reasons and would swamp the signal.
"""
import argparse, glob, numpy as np, torch, chess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", default="qat")
    ap.add_argument("--scope", default="all")
    ap.add_argument("--n", type=int, default=96)
    args = ap.parse_args()

    from catspace.research.tools.concept_planner import build
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    from catspace.research.tools.npz_io import load_npz
    dev = resolve_device("auto")
    # Load the TRUNK ONLY. This statistic reads field.encode() and never touches the
    # readout, and readout widths differ across runs (3072 vs 3592) -- load_state_dict
    # raises on a shape mismatch even with strict=False, so loading the whole model would
    # make this comparison impossible for exactly the checkpoints we want to compare.
    model, field, _, _ = build(args.ckpt, args.arm, dev, field_only=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    fsd = {k[len("field."):]: v for k, v in ck["state_dict"].items() if k.startswith("field.")}
    miss = field.load_state_dict(fsd, strict=False)
    print(f"  {args.ckpt}: trunk {len(fsd)} tensors, {len(miss.missing_keys)} missing\n")

    from catspace.research.tools.concept_planner import Planner
    P = Planner(model, field, dev)

    fens = []
    for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:6]:
        z = load_npz(f)
        if "wdl" in z.files and (z["kind"] == 0).any():
            fens += [str(x) for x in np.asarray(z["fen"])[:80]]
        if len(fens) >= args.n * 3:
            break

    par, gc = [], []
    for fn in fens:
        try:
            b0 = chess.Board(fn)
        except Exception:
            continue
        lm = list(b0.legal_moves)
        if not lm:
            continue
        b1 = b0.copy(); b1.push(lm[0])
        r = list(b1.legal_moves)
        if not r:
            continue
        b2 = b1.copy(); b2.push(r[0])
        par.append(b0); gc.append(b2)
        if len(par) >= args.n:
            break

    z0, _ = P._encode(par)
    z2, _ = P._encode(gc)
    B0 = field.g_heads * field.code_width
    S0 = B0 + 64 * field.sq_width
    P0 = field.d_state - field.edge_codes
    PW = max(1, (P0 - S0) // 32)
    n = len(par)

    sq0 = z0[:, B0:S0].reshape(n, 64, field.sq_width)
    sq2 = z2[:, B0:S0].reshape(n, 64, field.sq_width)
    d_sq = ((sq0 - sq2) ** 2).sum(-1).sqrt()
    zp = z0[:, S0:P0].reshape(n, 32, PW); tp = z2[:, S0:P0].reshape(n, 32, PW)
    alive = ((zp.abs().sum(-1) > 0) & (tp.abs().sum(-1) > 0)).float()
    d_pc = ((zp - tp) ** 2).sum(-1).sqrt() * alive
    d_hi = 0.5 * (z0[:, P0:] - z2[:, P0:]).abs().sum(-1)

    moved = ((d_sq > 1e-3).float().sum(-1) + (d_pc > 1e-3).float().sum(-1) +
             (d_hi > 1e-3).float())
    ngrp = 64 + alive.sum(-1) + 1
    grpchg = ((d_sq.sum(-1) + d_pc.sum(-1) + d_hi) / ngrp)
    bits = ((z0[:, :S0] > 0) != (z2[:, :S0] > 0)).float().mean()

    print(f"  {n} positions, 2-ply lag (same side to move)\n")
    print(f"  GROUPS MOVED            {float(moved.mean()):.1f} of {float(ngrp.mean()):.0f}"
          f"   ({100*float(moved.mean()/ngrp.mean()):.0f}% of groups touched)")
    print(f"    squares moved         {float((d_sq > 1e-3).float().sum(-1).mean()):.1f} of 64")
    print(f"    live pieces moved     {float((d_pc > 1e-3).float().sum(-1).mean()):.1f} "
          f"of {float(alive.sum(-1).mean()):.1f} alive")
    print(f"  group change magnitude  {float(grpchg.mean()):.3f}")
    print(f"  BITS changed            {100*float(bits):.1f}%")
    print(f"\n  Smeared vs concentrated: {100*float(bits):.1f}% of bits but "
          f"{100*float(moved.mean()/ngrp.mean()):.0f}% of groups means the change is spread "
          f"thin.\n  Group sparsity should cut the second number while leaving the first.")


if __name__ == "__main__":
    main()
