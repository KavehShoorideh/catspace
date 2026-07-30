"""catspace/memory/checkpoint_bank.py -- the Memory component of the traced engine:
an embedding bank of mined trap CONTEXTS, each linked to the CHECKPOINT (trap
position) its game actually sprang. Query with the current position's embedding ->
"positions like mine that led into traps" -> candidate trap structures with
exemplars, timing stats, and neighbour agreement.

Build once per encoder (the bank is encoder-specific); load at play time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build(checkpoints_npz: str, out: str, encoder="trunk", ckpt: str = "",
          sample: int = 150_000, seed: int = 0):
    """Embed a sample of mined contexts + their linked checkpoints -> bank npz."""
    rng = np.random.default_rng(seed)
    d = dict(np.load(checkpoints_npz, allow_pickle=True))
    link = d["cx_ckpt_row"]
    ok = link >= 0                                   # only contexts with a sprung trap
    idx = np.flatnonzero(ok)
    idx = np.sort(rng.choice(idx, min(sample, len(idx)), replace=False))
    fens = d["cx_fen"][idx]
    if encoder == "trunk":
        from tools.embed import trunk_encode
        emb = trunk_encode(list(fens))
    else:
        import chess
        from catspace.encoder.jepa import tokenize
        from catspace.train.scaffold import resolve_device
        from tools.embed import jepa_encode
        tg = [tokenize(chess.Board(f)) for f in fens]
        emb = jepa_encode(ckpt, np.stack([t for t, _ in tg]),
                          np.stack([g for _, g in tg]), resolve_device("auto"))
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    rows = link[idx]
    np.savez_compressed(
        out, emb=emb.astype(np.float32),
        cx_gap_dec=d["cx_gap_dec"][idx],
        ck_fen=d["ck_fen"][rows], ck_delta=d["ck_delta"][rows],
        ck_victim_white=d["ck_victim_white"][rows],
        ck_elo_victim=d["ck_elo_victim"][rows],
        meta_encoder=encoder, meta_ckpt=ckpt, meta_source=checkpoints_npz)
    print(f"[bank] wrote {out}: {len(emb)} contexts -> checkpoints "
          f"({encoder} encoder)")


class CheckpointBank:
    def __init__(self, path: str):
        d = dict(np.load(path, allow_pickle=True))
        self.emb = d["emb"]                          # (N, d), L2-normalized
        self.gap = d["cx_gap_dec"]
        self.ck_fen = d["ck_fen"]; self.ck_delta = d["ck_delta"]
        self.ck_victim_white = d["ck_victim_white"]
        self.encoder = str(d["meta_encoder"])

    def query(self, phi, k: int = 64, top_traps: int = 3, victim_white=None):
        """phi (d,) L2-normalized -> candidate traps: clusters of retrieved
        neighbours that share (near-identical) checkpoint positions.
        victim_white: restrict to traps whose historical victims are that color
        (pass the OPPONENT's color so candidates are traps for them, not us)."""
        q = phi / (np.linalg.norm(phi) + 1e-9)
        sim = self.emb @ q
        if victim_white is not None:
            sim = np.where(self.ck_victim_white == victim_white, sim, -np.inf)
        nn = np.argsort(-sim)[:k]
        # group neighbours by their checkpoint's piece-placement (trap identity)
        groups: dict[str, list[int]] = {}
        for i in nn:
            key = str(self.ck_fen[i]).split(" ")[0]
            groups.setdefault(key, []).append(int(i))
        cands = []
        for key, rows in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top_traps]:
            r = np.array(rows)
            cands.append(dict(
                exemplar_fen=str(self.ck_fen[r[0]]),
                support=len(r),                       # neighbour agreement
                agreement=len(r) / k,
                sim=float(sim[r].mean()),
                med_gap=float(np.median(self.gap[r])),   # victim decisions out
                med_delta=float(np.median(self.ck_delta[r])),
                victim_white=bool(np.median(self.ck_victim_white[r]) > 0.5)))
        return cands
