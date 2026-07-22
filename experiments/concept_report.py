#!/usr/bin/env python
"""experiments/concept_report.py -- a CONTRASTIVE visual report of the SAE concept DIRECTIONS (Kaposi
2026-07-21). A concept is a direction (vector) in the value field's embedding space -- the SAE decoder
atom. "Firing" is shorthand for "this position projects highly onto that direction". So we show each
concept as its AXIS: the positions at the POSITIVE pole (concept present) contrasted with the NEGATIVE
pole (its opposite/mirror), each board annotated with the ground-truth named features it actually has,
so a mislabel (an atom that merely co-occurs with a feature) is obvious. Maintained dictionary_learning
TopK SAE. Output: self-contained HTML.
"""
from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import chess
import chess.svg
import numpy as np
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device
from dictionary_learning.trainers import TopKTrainer
from experiments.concept_features import features as named_features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dict", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--tower", choices=["F", "B"], default="F")
    ap.add_argument("--per-atom", type=int, default=5, help="boards per pole")
    ap.add_argument("--max-atoms", type=int, default=26)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--out", default="artifacts/experiments/concept_report.html")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    ev = np.asarray(nz["eval_cp"]).astype(np.float32) if "eval_cp" in nz.files else None
    cand = np.flatnonzero(ply >= args.min_ply)
    pool = cand[rng.permutation(len(cand))[:min(len(cand), 80000)]]
    pcnt = np.unpackbits(P[pool].reshape(len(pool), -1).view(np.uint8), axis=1).sum(1)
    bins = np.digitize(pcnt, [8, 14, 20, 26]); per = args.n // 5
    idx = np.concatenate([pool[bins == b][:per] for b in range(5)]); idx = idx[rng.permutation(len(idx))]
    Pk, Mk = P[idx], M[idx]; evk = ev[idx] if ev is not None else None
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        emb = (fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)) if args.tower == "F"
               else fb.embed_B(t)).cpu().numpy()
    Xn = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)
    X = torch.from_numpy(Xn).float().to(dev)
    tr = TopKTrainer(steps=args.steps, activation_dim=X.shape[1], dict_size=args.dict, k=args.k, layer=0,
                     lm_name="catspace", device=dev, warmup_steps=max(1, args.steps // 10), seed=args.seed)
    for step in range(args.steps):
        tr.update(step, X[torch.from_numpy(rng.integers(0, len(X), size=1024)).to(dev)])
    with torch.no_grad():
        code = tr.ae.encode(X).cpu().numpy()
        D_dec = tr.ae.decoder.weight.detach().cpu().numpy().T      # (dict, dim): concept DIRECTION vectors
    proj = Xn @ D_dec.T                                            # (N, dict): signed coordinate along each direction

    alive = np.flatnonzero((code > 1e-6).mean(0) > 0.003)
    feats = [named_features(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl") and feats[0][n][1] == "bin"]
    short = {n: n.replace("_w", "").replace("_", " ") for n in fnames}
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])
    from experiments.conditional_concepts import openness as _openness
    pc_k = np.unpackbits(Pk.reshape(len(Pk), -1).view(np.uint8), axis=1).sum(1).astype(float)   # piece count
    opn_k = np.array([_openness(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))], float)

    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else 0.0
    print(f"[stage] SAE trained, {len(alive)} alive atoms ({time.time()-t0:.0f}s)", flush=True)

    # each atom's contrastive separation profile (feature prevalence at + pole minus at - pole)
    poles, seps = {}, {}
    for a in alive:
        pos200, neg200 = np.argsort(-proj[:, a])[:200], np.argsort(proj[:, a])[:200]
        poles[int(a)] = (pos200, neg200)
        seps[int(a)] = np.array([Fmat[pos200, j].mean() - Fmat[neg200, j].mean() for j in range(len(fnames))])
    # CANONICAL atom per named concept (the one that separates it most) -> clean 1:1 labels, rest novel
    concept_best, claimed = {}, {}
    for j in range(len(fnames)):
        ba = int(max(alive, key=lambda a: abs(seps[int(a)][j])))
        if abs(seps[ba][j]) > 0.30 and abs(seps[ba][j]) > claimed.get(ba, 0):
            concept_best[ba] = (short[fnames[j]], j); claimed[ba] = abs(seps[ba][j])
    rows = []
    for a in alive:
        a = int(a); jm = int(np.argmax(np.abs(seps[a])))
        label, lj = concept_best.get(a, ("novel", jm))
        prof = sorted(((short[fnames[j]], corr(code[:, a], Fmat[:, j])) for j in range(len(fnames))),
                      key=lambda kv: -abs(kv[1]))[:3]
        rows.append(dict(atom=a, feat=short[fnames[lj]], sep=float(seps[a][lj]),
                         pplus=float(Fmat[poles[a][0], lj].mean()), pminus=float(Fmat[poles[a][1], lj].mean()),
                         label=label, fire=float((code[:, a] > 1e-6).mean()),
                         ev=(corr(code[:, a], evk) if evk is not None else 0.0),
                         ph=corr(code[:, a], pc_k), op=corr(code[:, a], opn_k), prof=prof))
    rows.sort(key=lambda r: (r["label"] == "novel", -abs(r["sep"])))

    def annotate(i):
        tags = [short[n] for j, n in enumerate(fnames) if Fmat[i, j] > 0.5]
        return ", ".join(tags) if tags else "&mdash;"

    def mini(i, cls):
        b = board_from_packed(Pk[i], Mk[i])
        ecap = f" &middot; {evk[i]/100:+.1f}" if evk is not None else ""
        return (f"<div class='pos {cls}'>{chess.svg.board(b, size=132, coordinates=False)}"
                f"<div class=ann>{annotate(i)}</div><div class=fen>{html.escape(b.fen())}{ecap}</div></div>")

    def matched_pairs(a, n_pairs, pool=400):
        """positions identical EXCEPT along direction a: match high-projection to nearest low-projection
        in the RESIDUAL (embedding with the a-component removed)."""
        d = D_dec[a] / (np.linalg.norm(D_dec[a]) + 1e-9)
        pr = Xn @ d
        R = Xn - np.outer(pr, d)                                  # everything except the a direction
        hi, lo = np.argsort(-pr)[:pool], np.argsort(pr)[:pool]
        Dm = cdist(R[hi], R[lo])                                  # residual distance
        pairs, used = [], set()
        for hi_i in np.argsort(Dm.min(1)):                       # closest available match first
            row = Dm[hi_i].copy()
            for j in used:
                row[j] = np.inf
            lo_j = int(row.argmin())
            if not np.isfinite(row[lo_j]):
                break
            used.add(lo_j); pairs.append((int(hi[hi_i]), int(lo[lo_j])))
            if len(pairs) >= n_pairs:
                break
        return pairs

    css = """body{font-family:-apple-system,system-ui,sans-serif;margin:22px;background:#faf9f7;color:#1a1a1a}
    @media(prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}.atom{background:#1f1f1f;border-color:#333}}
    h1{font-size:21px}.sub{color:#888;font-size:13px;margin-bottom:18px;max-width:900px}
    .atom{border:1px solid #ddd;border-radius:10px;padding:12px 16px;margin:14px 0;background:#fff}
    .atom h2{font-size:15px;margin:0}.tag{color:#0a7d38}.novel{color:#a06000}
    .meta{color:#888;font-size:12px;margin:2px 0 3px}
    .prof{color:#555;font-size:12px;margin:0 0 10px}@media(prefers-color-scheme:dark){.prof{color:#aaa}}
    .pairs{display:flex;flex-wrap:wrap;gap:20px}
    .pair{display:flex;align-items:flex-start;gap:4px;padding:6px;border:1px solid #eee;border-radius:8px}
    @media(prefers-color-scheme:dark){.pair{border-color:#333}}
    .lbl{font-weight:700;font-size:15px;margin-top:56px;color:#999}.pos.has{outline:2px solid #0a7d38}
    .pos{width:132px}.pos svg{width:132px;height:132px}
    .ann{font-size:10px;color:#0a7d38;margin-top:2px;min-height:12px}
    .fen{font-family:ui-monospace,monospace;font-size:8px;color:#aaa;word-break:break-all;margin-top:1px}"""
    parts = [f"<style>{css}</style>", "<h1>Concept directions &mdash; catspace value field (IQE+QRL)</h1>",
             f"<div class=sub>Each atom is a <b>direction</b> in the {args.tower}-tower embedding "
             f"(dictionary_learning TopK SAE, dict={args.dict}, k={args.k}). Shown as <b>minimal pairs</b>: "
             "each pair is a position high on the direction (<b>+</b>, left) next to its nearest match with "
             "the direction removed (<b>&minus;</b>, right) &mdash; identical in the field's view except along "
             "this vector, so the difference between them <i>is</i> the concept. Green = ground-truth features "
             "each board actually has. Named concepts first, then novel directions.</div>"]
    for r in rows[:args.max_atoms]:
        cls = "novel" if r["label"] == "novel" else "tag"
        parts.append(f"<div class=atom><h2>Atom {r['atom']} &middot; <span class={cls}>{html.escape(r['label'])}</span></h2>")
        parts.append(f"<div class=meta>closest feature <b>{html.escape(r['feat'])}</b>: {r['pplus']:.0%} at + pole "
                     f"vs {r['pminus']:.0%} at &minus; pole (&Delta;{r['sep']:+.0%}) &middot; fires {100*r['fire']:.0f}%</div>")
        pd = "endgame" if r["ph"] < -0.15 else ("opening" if r["ph"] > 0.15 else "mid / any phase")
        od = " &middot; open positions" if r["op"] > 0.12 else (" &middot; closed positions" if r["op"] < -0.12 else "")
        profstr = ", ".join(f"{html.escape(n)} {c:+.2f}" for n, c in r["prof"])
        parts.append(f"<div class=prof><b>value {r['ev']:+.2f}</b> &middot; {pd}{od} &middot; correlates: {profstr}</div>")
        parts.append("<div class=pairs>")
        for hi_i, lo_j in matched_pairs(r["atom"], args.per_atom):
            parts.append(f"<div class=pair><div class=lbl>+</div>{mini(hi_i, '')}"
                         f"<div class=lbl>&minus;</div>{mini(lo_j, '')}</div>")
        parts.append("</div></div>")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"VERDICT CONCEPT_REPORT atoms={len(alive)} shown={min(len(rows),args.max_atoms)} -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
