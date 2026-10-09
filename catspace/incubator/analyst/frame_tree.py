"""THE FRAME AS A TREE OF BRANCHES, summarized (Kaveh 2026-10-09): "what is the probability of me winning this rook?
of mating? of just moving pieces around and winning nothing?" -- analyst_core.frame_tree builds the tree (Maia's
probabilities at each of the opponent's branching points; above 1900 the opponent is perfect; the certified lemmas
re-evaluated on every branch; mate = 150); this summarizes it into outcome probabilities and the main paths."""
from __future__ import annotations

import time

import chess

from catspace.io import paths


def model(band: int | None) -> str:
    """The Maia model for a rating; None or above 1900: '' (a perfect opponent)."""
    if band is None or band > 1900:                                   # a perfect opponent: Maia-1900's plausible replies
        return "perfect:" + str(paths.assets_dir() / "engines" / "maia_onnx" / "maia-1900.onnx")   # plus every forcing one
    band = max(1100, min(1900, round(band / 100) * 100))
    return str(paths.assets_dir() / "engines" / "maia_onnx" / f"maia-{band}.onnx")


def summarize(b: chess.Board, m: chess.Move, band: int | None, depth: int = 3) -> dict:
    import analyst_core
    t = time.time()
    rows = analyst_core.frame_tree(b.fen(), m.uci(), model(band), 0.95, depth)
    P = lambda f: sum(p for p, v, _ in rows if f(v))
    out = dict(p_mate=P(lambda v: v >= 150), p_piece=P(lambda v: 3 <= v < 150), p_less=P(lambda v: 0 < v < 3),
               p_nothing=P(lambda v: v == 0), p_loses=P(lambda v: v < 0), expected=sum(p * v for p, v, _ in rows),
               ms=round(1000 * (time.time() - t)))

    def san_path(path):
        bb = b.copy(); out_ = []
        for step in path:
            u = step.split()[0]
            mv = chess.Move.from_uci(u)
            note = step[len(u):].strip()
            out_.append(bb.san(mv) + (f" {note}" if note else "")); bb.push(mv)
        return " ".join(out_)
    out["paths"] = [(p, v, san_path(pa)) for p, v, pa in sorted(rows, key=lambda r: -r[0])[:4]]
    return out
