"""regimes.py -- the QUANTIZED FUTURE (Kaveh 2026-09-11: "a future board embedding distribution predicted from the
current state; bonus points if it's quantized or interpretable; it has to predict the sharpness, not the details").

A regime CODE is a cell of a k-means quantizer fit on the graph's own position indicators: the 15 regime keys
(material, phase, pieces, queens, rooks connected, passed pawns and runners, bishop pairs, pawns, open files) and
6 SHARPNESS keys read off the tactical layer (the biggest threat in either ledger, the hottest zone, pieces
falling, a mate threat, king attackers, check). Every code is described in words by its centroid, so a predicted
distribution over codes reads as "queens off and a passed pawn of ours, 40%; the same with the king under
attack, 25%". The codebook is fit on the training games only and persisted beside the dataset.

    python -m catspace.incubator.analyst.regimes fit --data data/derived/knowledge --k 48
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .graph import POSITION_QUESTIONS
from .dataset import REGIME_KEYS

P = {q: i for i, q in enumerate(POSITION_QUESTIONS)}
SHARP_KEYS = ("threat_max", "hottest", "falls", "mate", "king_attackers", "in_check")
CODE_KEYS = tuple(REGIME_KEYS) + SHARP_KEYS
BIG = 3.0                                  # a threat worth a piece: the sharpness threshold


def vectors(position: np.ndarray) -> np.ndarray:
    """(N, 39) position indicators -> (N, 21) code coordinates."""
    reg = position[:, [P[k] for k in REGIME_KEYS]]
    threat = np.minimum(np.maximum(position[:, P["threat_ours"]], position[:, P["threat_theirs"]]), 10.0)
    hottest = np.minimum(position[:, P["hottest"]], 10.0)
    falls = position[:, P["falls_ours"]] + position[:, P["falls_theirs"]]
    mate = np.minimum(position[:, P["mate_ours"]] + position[:, P["mate_theirs"]], 1.0)
    ka = np.maximum(position[:, P["king_attackers_ours"]], position[:, P["king_attackers_theirs"]])
    chk = position[:, P["in_check"]]
    return np.concatenate([reg, np.stack([threat, hottest, falls, mate, ka, chk], 1)], 1).astype(np.float32)


class Codebook:
    def __init__(self, centroids: np.ndarray, mean: np.ndarray, std: np.ndarray, values: np.ndarray, counts: np.ndarray):
        self.centroids, self.mean, self.std, self.values, self.counts = centroids, mean, std, values, counts

    @property
    def k(self):
        return len(self.centroids)

    def assign(self, position: np.ndarray) -> np.ndarray:
        z = (vectors(position) - self.mean) / self.std
        d = ((z[:, None, :] - self.centroids[None]) ** 2).sum(-1)
        return d.argmin(1)

    def raw(self, code: int) -> dict:
        c = self.centroids[code] * self.std + self.mean
        return {k: float(v) for k, v in zip(CODE_KEYS, c)}

    def sharp(self) -> np.ndarray:
        """Which codes are sharp: a big threat in the ledger or a mate threat at the centroid."""
        c = self.centroids * self.std + self.mean
        return (c[:, CODE_KEYS.index("threat_max")] >= BIG) | (c[:, CODE_KEYS.index("mate")] >= 0.5)

    def describe(self, code: int) -> str:
        r = self.raw(code)
        bits = []
        m = r["material_diff"]
        bits.append("material even" if abs(m) < 0.75 else f"material {m:+.1f}")
        ph = r["phase"]
        bits.append("opening" if ph >= 55 else "middlegame" if ph >= 35 else "late middlegame" if ph >= 18 else "ending")
        bits.append("queens off" if r["queens_on"] < 0.5 else "queens on" if r["queens_on"] >= 1.5 else "one queen")
        if r["passed_ours"] >= 0.5:
            bits.append(f"a passed pawn of ours ({r['runner_ours']:.0f} to go)" if r["passed_ours"] < 1.5 else "passed pawns of ours")
        if r["passed_theirs"] >= 0.5:
            bits.append("a passed pawn of theirs" if r["passed_theirs"] < 1.5 else "passed pawns of theirs")
        if r["rooks_connected_ours"] >= 0.5:
            bits.append("our rooks connected")
        if r["bishop_pair_ours"] >= 0.5 and r["bishop_pair_theirs"] < 0.5:
            bits.append("our bishop pair")
        if r["open_files"] >= 2.5:
            bits.append(f"{r['open_files']:.0f} open files")
        if r["mate"] >= 0.5:
            bits.append("a mate threat")
        elif r["threat_max"] >= BIG:
            bits.append(f"a big threat ({r['threat_max']:.0f})")
        elif r["threat_max"] >= 1.0:
            bits.append("small threats")
        else:
            bits.append("no threats")
        if r["falls"] >= 1.5:
            bits.append("pieces hanging")
        if r["king_attackers"] >= 2.5:
            bits.append("a king under attack")
        if r["in_check"] >= 0.5:
            bits.append("in check")
        return ", ".join(bits)

    def save(self, path: str):
        np.savez(path, centroids=self.centroids, mean=self.mean, std=self.std, values=self.values, counts=self.counts, keys=np.array(CODE_KEYS))

    @classmethod
    def load(cls, path: str) -> "Codebook":
        z = np.load(path)
        return cls(z["centroids"], z["mean"], z["std"], z["values"], z["counts"])


def fit(position: np.ndarray, train_mask: np.ndarray, results: np.ndarray, k: int = 48, seed: int = 0) -> Codebook:
    """k-means on the standardized code coordinates of the training positions; a code's VALUE is the mean game
    result from the side to move's view over the training positions in it (0 loss, 0.5 draw, 1 win)."""
    from sklearn.cluster import KMeans
    v = vectors(position)
    mean, std = v[train_mask].mean(0), v[train_mask].std(0) + 1e-6
    z = (v - mean) / std
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(z[train_mask])
    cb = Codebook(km.cluster_centers_.astype(np.float32), mean, std, np.zeros(k), np.zeros(k))
    codes = cb.assign(position)
    vals, cnts = np.zeros(k), np.zeros(k)
    for c in range(k):
        sel = train_mask & (codes == c) & (results >= 0)
        cnts[c] = sel.sum(); vals[c] = results[sel].mean() if sel.any() else 0.5
    cb.values, cb.counts = vals, cnts
    return cb


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fit", "show"]); ap.add_argument("--data", default="data/derived/knowledge"); ap.add_argument("--k", type=int, default=48)
    a = ap.parse_args(argv)
    from .dataset import load
    d = load(a.data); meta = d["meta"]; pos = d["position"]
    games = sorted({m["game"] for m in meta}); rng = random.Random(0); rng.shuffle(games)
    n = len(games); hold = set(games[: n // 5])
    train = np.array([m["game"] not in hold for m in meta])
    res = np.array([-1.0 if m["result"] is None else m["result"] for m in meta])
    path = Path(a.data) / "codebook.npz"
    if a.cmd == "fit":
        cb = fit(pos, train, res, a.k); cb.save(str(path)); print(f"[regimes] fit {a.k} codes on {train.sum()} training positions -> {path}")
    else:
        cb = Codebook.load(str(path))
    codes = cb.assign(pos)
    index = {(m["game"], m["ply"]): i for i, m in enumerate(meta)}
    fut = np.array([index.get((m["game"], m["ply"] + 10), -1) for m in meta]); has = fut >= 0
    same = (codes[has] == codes[fut[has]]).mean()
    print(f"[regimes] {cb.k} codes; sharp codes {int(cb.sharp().sum())}; the code persists 10 plies later {same:.0%}; most common code holds {np.bincount(codes, minlength=cb.k).max() / len(codes):.0%}")
    order = np.argsort(-cb.counts)
    print(f"  {'code':>4}{'n':>7}{'value':>7}  description")
    for c in order:
        print(f"  {c:>4}{int(cb.counts[c]):>7}{cb.values[c]:>7.2f}  {cb.describe(int(c))}")


if __name__ == "__main__":
    main()
