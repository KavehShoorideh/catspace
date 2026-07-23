"""catspace/vectordb.py -- Qdrant adapter (Kaveh 2026-07-25: 'a proper vector db').

The immortal banks become Qdrant collections: one point per exemplar --
vector = B-embedding under the current field, payload = {epd, sig}. Deterministic ids
(epd hash) make sync idempotent; re-running after a field swap re-upserts new vectors
under the same ids (embeddings are per-field, facts are forever).

  sync_banks(field_ckpt, prefix)   -- upsert win/loss/draw banks from <prefix>_*.fens
  query(collection, board, k)      -- nearest exemplars for a position (F-embedding query
                                      against B-vectors: the reachability lookup)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import chess
import numpy as np

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, timeout=30)


def _pid(epd: str) -> int:
    return int(hashlib.sha1(epd.encode()).hexdigest()[:15], 16)


def sync_banks(field_ckpt: str, prefix: str, device: str = "cpu", batch: int = 512) -> dict:
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from catspace.engine.fields import FieldModel
    from experiments.bootstrap_mate_engine import mat_sig
    fm = FieldModel(field_ckpt, device=device)
    cli = _client()
    out = {}
    for name, sfx in (("bank_win", "_bank.fens"), ("bank_loss", "_lossbank.fens"),
                      ("bank_draw", "_drawbank.fens")):
        p = Path(prefix + sfx)
        if not p.exists():
            out[name] = 0
            continue
        epds = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        if not epds:
            out[name] = 0
            continue
        boards = [chess.Board(e) for e in epds]
        dim = None
        n = 0
        for s in range(0, len(boards), batch):
            bs = boards[s:s + batch]
            E = fm.embed_B_boards(bs)
            if dim is None:
                dim = E.shape[1]
                if not cli.collection_exists(name):
                    cli.create_collection(name, vectors_config=VectorParams(
                        size=dim, distance=Distance.EUCLID))
            cli.upsert(name, points=[
                PointStruct(id=_pid(b.epd()), vector=E[i].tolist(),
                            payload={"epd": b.epd(), "sig": mat_sig(b)})
                for i, b in enumerate(bs)])
            n += len(bs)
        out[name] = n
    return out


def query(collection: str, board: chess.Board, field_ckpt: str, k: int = 8,
          device: str = "cpu"):
    from catspace.engine.fields import FieldModel
    fm = FieldModel(field_ckpt, device=device)
    F = fm.embed_F_boards([board])[0]
    cli = _client()
    res = cli.query_points(collection, query=F.tolist(), limit=k)
    return [{"epd": pt.payload["epd"], "sig": pt.payload.get("sig"),
             "score": pt.score} for pt in res.points]
