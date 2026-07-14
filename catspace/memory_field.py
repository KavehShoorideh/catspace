"""
memory_field.py — the FAST field: an in-memory, per-move-updatable evidence store
over embedding space (Kaveh, 2026-07-14 two-timescale design).

The SLOW field is the trained embedding (stationary geometry, retrained at epoch
boundaries). This is the fast one: rows of search/rollout evidence keyed by
embedding location, written every move, queried by kNN with (simple, for now)
visit-count weighting. "The landscape has shifted" is a first-class operation:
add rows mid-game; distill into the slow net between games (the closed loop).

Schema note (Kaveh): rows generalise beyond scalar evidence -- a TACTIC-POTENTIAL
is a row whose key is a PRECONDITION region ("if opponent plays X, the state lands
here") and whose payload is a plan/tactic + payoff, cf. the 2026-07-10 conditional
capture-vector design. Same store, same kNN firing rule; payload just isn't a
scalar. Not built yet -- schema reserved via the `payload` dict.

Start-simple choices (upgrade hooks later): visit-count weighting (not the
competence head); cosine kNN (embeddings are L2-normalised); no eviction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MemoryField:
    def __init__(self, d: int):
        self.d = d
        self.E = np.zeros((0, d), dtype=np.float32)   # embedding keys (L2-normalised)
        self.rows: list[dict] = []                     # {fen, p_hat, n, plies, payload}

    def add(self, emb: np.ndarray, fen: str, p_hat: float, n: int,
            plies: float | None = None, payload: dict | None = None) -> None:
        e = np.asarray(emb, dtype=np.float32)
        e = e / max(float(np.linalg.norm(e)), 1e-9)
        self.E = np.vstack([self.E, e[None]])
        self.rows.append(dict(fen=fen, p_hat=float(p_hat), n=int(n),
                              plies=plies, payload=payload or {}))

    def query(self, emb: np.ndarray, k: int = 8) -> dict | None:
        """Visit-count-weighted local evidence around `emb`: returns
        {p_hat, plies, n_total, support} over the k nearest rows, or None if empty.
        support = mean cosine of the neighbours (how local the evidence is --
        callers can gate the blend on it)."""
        if len(self.rows) == 0:
            return None
        e = np.asarray(emb, dtype=np.float32)
        e = e / max(float(np.linalg.norm(e)), 1e-9)
        sims = self.E @ e
        idx = np.argsort(-sims)[:k]
        w = np.array([self.rows[i]["n"] for i in idx], dtype=np.float64)
        w = w / max(w.sum(), 1e-9)
        p = float(sum(w[j] * self.rows[i]["p_hat"] for j, i in enumerate(idx)))
        pl = [(w[j], self.rows[i]["plies"]) for j, i in enumerate(idx)
              if self.rows[i]["plies"] is not None]
        plies = float(sum(wj * x for wj, x in pl) / max(sum(wj for wj, _ in pl), 1e-9)) if pl else None
        return dict(p_hat=p, plies=plies,
                    n_total=int(sum(self.rows[i]["n"] for i in idx)),
                    support=float(sims[idx].mean()))

    # ---- persistence + bulk load -----------------------------------------
    def save(self, path):
        np.savez(Path(path), E=self.E,
                 rows=np.frombuffer(json.dumps(self.rows).encode(), dtype=np.uint8))

    @classmethod
    def load(cls, path) -> "MemoryField":
        z = np.load(Path(path))
        mf = cls(z["E"].shape[1] if z["E"].size else 64)
        mf.E = z["E"]
        mf.rows = json.loads(bytes(z["rows"]).decode())
        return mf

    @classmethod
    def from_certainty_table(cls, table_json, fb, device) -> "MemoryField":
        """Bulk-build from certainty_rollouts.py output: embed each fen with the
        SLOW field's F and store its rollout evidence."""
        import chess
        import torch
        from catspace.data.encode import encode_meta, encode_packed
        from catspace.nn.features import feature_planes, omega_ids
        rows = json.loads(Path(table_json).read_text())["rows"]
        omega = omega_ids(np.array([1800]), np.array([1800]), np.array([float("nan")]))[0]
        mf = cls(fb.d)
        B = 512
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            boards = [chess.Board(r["fen"]) for r in chunk]
            packed = np.stack([encode_packed(b) for b in boards])
            meta = np.stack([encode_meta(b) for b in boards])
            with torch.no_grad():
                pl = torch.from_numpy(feature_planes(packed, meta)).to(device)
                om = torch.from_numpy(np.tile(omega, (len(chunk), 1))).to(device)
                F = fb.embed_F(pl, om).cpu().numpy()
            for r, f in zip(chunk, F):
                mf.add(f, r["fen"], r["p_hat"], r["n"], r.get("plies"))
        return mf
