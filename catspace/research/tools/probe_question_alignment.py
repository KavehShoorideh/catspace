"""THE flagship faithfulness measurement, run for the first time.

Quantized attention exists so that "question 5 means ATTACKS" is a falsifiable claim
(the answer to Jain & Wallace). This was built, unit-gated, and then never once measured
on a trained model across six training runs. This probe measures it:

  for each head and each discrete question id, when a square-token asks that question,
  what fraction of its attention mass lands on squares that python-chess says stand in a
  REAL relation to it (attacks / defends / threatens)?

Reported against two nulls: the relation's density (what uniform attention would score)
and a per-board column-shuffle of the truth (what this attention scores on scrambled
targets). Lift < ~1.5x chance on every head = the questions are not typed relations,
which is what arXiv 2407.19532 predicts for discretisation without semantic pressure.

qat4 CAVEAT: it predates the all-family token path, so its attention consumed RAW 24-bit
square slices. The probe reproduces that exact path; running it through the new
fam_proj (random for this ckpt) would measure noise.
"""
import argparse, glob
import numpy as np
import torch
import chess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/qat4_regret_latest.pt")
    ap.add_argument("--n", type=int, default=160)
    args = ap.parse_args()
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    from catspace.research.tools.concept_planner import build, Planner
    from catspace.research.tools.npz_io import load_npz
    dev = resolve_device("auto")

    model, field, _, _ = build(args.ckpt, "qat", dev, field_only=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    field.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("field.")},
                          strict=False)
    att_sd = {k[5:]: v for k, v in sd.items() if k.startswith("attn.")}
    miss = model.attn.load_state_dict(att_sd, strict=False)
    assert not miss.missing_keys, f"attention tensors missing: {miss.missing_keys}"
    model.eval(); field.eval()
    print(f"  {args.ckpt}: attention fully restored ({len(att_sd)} tensors)")

    fens = []
    for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:6]:
        z = load_npz(f)
        if "wdl" in z.files and (z["kind"] == 0).any():
            fens += [str(x) for x in np.asarray(z["fen"])[:80]]
        if len(fens) >= args.n * 2:
            break
    boards = []
    for fn in fens:
        try:
            b = chess.Board(fn)
            boards.append(b)
        except Exception:
            continue
        if len(boards) >= args.n:
            break
    P = Planner(model, field, dev)
    b0 = field.g_heads * field.code_width

    A_all, Q_all = [], []
    with torch.no_grad():
        for i in range(0, len(boards), 32):
            zq, _ = P._encode(boards[i:i + 32])
            tok = zq[:, b0:b0 + 64 * field.sq_width].reshape(-1, 64, field.sq_width)
            _, aux = model.attn(tok)                     # RAW slices: qat4's trained path
            A_all.append(aux["attn"].cpu()); Q_all.append(aux["question"].cpu())
    A = torch.cat(A_all); Q = torch.cat(Q_all)           # (B,H,64,64), (B,64,H)
    B, H = A.shape[0], A.shape[1]
    print(f"  {B} positions encoded; {H} heads x {model.attn.n_q} questions\n")

    def truths(rel):
        T = torch.zeros(B, 64, 64)
        for bi, bd in enumerate(boards):
            for sq, pc in bd.piece_map().items():
                for t in bd.attacks(sq):
                    tp = bd.piece_at(t)
                    if rel == "attacks" or \
                       (rel == "defends" and tp and tp.color == pc.color) or \
                       (rel == "threatens" and tp and tp.color != pc.color):
                        T[bi, sq, t] = 1.0
        return T

    rng = np.random.default_rng(0)
    print(f"  {'relation':10s} {'density':>8s} | per-head precision (shuffled) | best lift")
    best_pairs = []
    for rel in ("attacks", "defends", "threatens"):
        T = truths(rel)
        dens = float(T.mean())
        Ts = T[:, :, torch.from_numpy(rng.permutation(64))]
        prec, sprec = [], []
        for h in range(H):
            a = A[:, h]
            prec.append(float((a * T).sum() / a.sum().clamp(min=1e-9)))
            sprec.append(float((a * Ts).sum() / a.sum().clamp(min=1e-9)))
        row = " ".join(f"{p:.3f}({s:.3f})" for p, s in zip(prec, sprec))
        print(f"  {rel:10s} {dens:>8.3f} | {row} | {max(prec)/max(dens,1e-9):.2f}x")
        # per-(head, question): does any QUESTION specialise in this relation?
        for h in range(H):
            for q in range(model.attn.n_q):
                m = (Q[:, :, h] == q)                     # (B,64) tokens asking q
                if int(m.sum()) < 200:
                    continue
                a = A[:, h][m]                            # (n,64) their attention rows
                t = T[m]
                p = float((a * t).sum() / a.sum().clamp(min=1e-9))
                best_pairs.append((p / max(dens, 1e-9), rel, h, q, p,
                                   float(m.float().mean())))
    best_pairs.sort(reverse=True)
    print(f"\n  top (head, question) pairs by lift over density:")
    print(f"  {'lift':>6s}  {'relation':10s} {'head':>4s} {'q':>3s} {'precision':>9s} {'usage':>6s}")
    for lf, rel, h, q, p, u in best_pairs[:8]:
        print(f"  {lf:>5.2f}x  {rel:10s} {h:>4d} {q:>3d} {p:>9.3f} {100*u:>5.1f}%")
    hist = torch.zeros(H, model.attn.n_q)
    for h in range(H):
        hist[h] = torch.bincount(Q[:, :, h].flatten(), minlength=model.attn.n_q).float()
    hist = hist / hist.sum(-1, keepdim=True)
    print(f"\n  question usage: max share per head "
          f"{[round(float(x),2) for x in hist.max(-1).values]} "
          f"(1.0 = a dead single-question head)")


if __name__ == "__main__":
    main()
