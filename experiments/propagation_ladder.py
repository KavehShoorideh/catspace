#!/usr/bin/env python
"""experiments/propagation_ladder.py — does the field propagate an ENDGAME value
back to the OPENING through reachability? (Kaveh 2026-07-19: "the probability of
going from early bishop blunder to losing position is high" -- a piece-down
opening funnels to a piece-down endgame, which is clearly lost; in a quasimetric
field that is the triangle inequality d(open->MATE_W) <= d(open->losing_eg) +
d(losing_eg->MATE_W)).

Measures reach-to-WHITE-WIN (-quasimetric distance to MATE_W for a metric field,
else fb.score; higher = white closer to winning) along a ladder from an EQUAL
opening to a piece-down endgame. If the propagation reaches back, the Delta-vs-
equal lifts off zero even at the move-2 blunder. Baseline (incumbent
cert_base_full, quasimetric distance): endgame Delta +0.850 (98% of the KRvK
winning reference +0.866 -- captured), middlegame +0.161 (propagates), move-2
opening +0.045 (WEAK but nonzero -- propagation reaches move 2, ~5% of the
endgame magnitude). Re-run on a sharper field to see whether the move-2 Delta
lifts well above +0.045.

Usage:
  .venv/bin/python experiments/propagation_ladder.py --ckpt data/derived/sep/cert_base_full.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt


def _line(mvs):
    b = chess.Board()
    for m in mvs:
        b.push_san(m)
    return b.fen()


LADDER = [
    ("equal opening (1.d4 e6 2.Nc3 Nf6)", _line(["d4", "e6", "Nc3", "Nf6"])),
    ("BLACK down a bishop, move 2 (Ba3 bxa3)", _line(["d4", "e6", "Nc3", "Ba3", "bxa3"])),
    ("BLACK down a bishop, middlegame", "r2q1rk1/ppp2ppp/5n2/3pp3/3PP3/5N2/PPP2PPP/R2Q1RK1 w - - 0 1"),
    ("BLACK down a bishop, ENDGAME (KRB v KR)", "8/5k2/8/8/8/3K1B2/4R3/5r2 w - - 0 1"),
    ("reference: WHITE up a ROOK endgame (KRvK)", "8/5k2/8/8/8/3K4/4R3/8 w - - 0 1"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    args = ap.parse_args()
    dev = "cpu"
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.eval()
    quasi = bool(getattr(fb, "quasimetric", False))
    zW = pay["zgoals"]["MATE_W"].to(dev).float()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    def reachW(fen):
        b = chess.Board(fen)
        pl = feature_planes(encode_packed(b)[None], encode_meta(b)[None])
        with torch.no_grad():
            f = fb.embed_F(torch.from_numpy(pl), torch.from_numpy(np.tile(om, (1, 1))))
            # reach toward the white-win pole: -distance for a quasimetric, else score
            if quasi:
                return -float(fb.distance_matrix(f, zW[None, :])[:, 0])
            return float(fb.score(f, zW))

    print(f"ckpt={Path(args.ckpt).name} quasimetric={quasi}  step={pay.get('step','?')}")
    print("reach-to-WHITE-WIN (higher = white closer to winning); Delta vs the equal opening:")
    base = None
    open_delta = None
    for name, fen in LADDER:
        r = reachW(fen)
        if base is None:
            base = r
        d = r - base
        if "move 2" in name:
            open_delta = d
        print(f"  {r:+.4f}  (Δ{d:+.4f})  {name}")
    print(f"VERDICT PROPAGATION move2_opening_delta={open_delta:+.4f} "
          f"(>0 => the losing-endgame value reaches back to the move-2 blunder; "
          f"~0 => propagation dies before the opening)")


if __name__ == "__main__":
    main()
