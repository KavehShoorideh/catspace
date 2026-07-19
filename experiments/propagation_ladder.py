#!/usr/bin/env python
"""experiments/propagation_ladder.py — does the field propagate an ENDGAME value
back to the OPENING? (Kaveh 2026-07-19: a piece-down opening funnels to a
piece-down endgame that is clearly lost, so it should read as losing.)

COMPOSED (relaxation) distance, not a one-hop distance (Kaveh: "find the nearest
positions, take the distance to that position + distance to mate, then take the
min of that among all the neighbors"):

    d_hat(s -> win) = min over waypoints g of [ d(F(s), B(g)) + d(g -> mate) ]

The field is trusted only for the SHORT hop d(F(s), B(g)) to a nearby known
position g; g's distance-to-mate d(g->mate) is a KNOWN quantity -- exact Syzygy
DTM for the tablebase waypoints, 0 for a terminal (mate/draw) final. Minimizing
over waypoints picks the best intermediate. This is the triangle inequality made
non-parametric, and it never trusts the field's mushy long-range distance.

Waypoints (the "surfaces", from the vector DB of known positions):
  W (white win): DTM tablebase white-wins (d(g->mate)=dtm) + white-mate finals (0)
  L (black win): black-mate finals (d=0)
  D (draw):      draw finals (d=0)

reach = -d_hat (higher = closer). EXPECT for a black-down-material ladder:
reach->W UP, reach->L DOWN, reach->D DOWN, and the effect reaching FURTHER back
toward move 2 as the field sharpens. NOTE the two terms are commensurable (both
in plies) only for a DTM-aligned field; on a non-DTM field pass --dtm-scale to
rescale d(g->mate) into the field's distance units.

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
from catspace.io.paths import newest_shard_dir
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt


def _line(mvs):
    b = chess.Board()
    for m in mvs:
        b.push_san(m)
    return b.fen()


# PAIRED by stage: (stage, EQUAL fen, black-DOWN-a-bishop fen) at matched piece
# counts, so delta = down - equal isolates MATERIAL (not simplification). The
# question: does the material delta (W up, L down, D down) reach back to the
# OPENING, or only show up once you're near the endgame?
PAIRS = [
    ("opening   (move 2)",
     _line(["d4", "e6", "Nc3", "Nf6"]),                                  # equal
     _line(["d4", "e6", "Nc3", "Ba3", "bxa3"])),                         # black down a bishop
    ("middlegame",
     "r2q1rk1/ppp2ppp/2n2n2/3pp3/3PP3/2N2N2/PPP2PPP/R2Q1RK1 w - - 0 1",  # equal (both have 2 minors)
     "r2q1rk1/ppp2ppp/5n2/3pp3/3PP3/2N2N2/PPP2PPP/R2Q1RK1 w - - 0 1"),   # black missing a knight (~down a minor)
    ("endgame",
     "8/5k2/8/8/8/3K4/4R3/5r2 w - - 0 1",                                # KR v KR (equal, drawn)
     "8/5k2/8/8/8/3K4/4R3/8 w - - 0 1"),                                 # KRvK (white up a ROOK,
]                                                                        # clearly won + in DTM coverage)


def _draw_finals(shard_dir, cap):
    picks = []
    for path in sorted(shard_dir.glob("shard_*.npz")):
        npz = np.load(path)
        gid, result, packed, meta = npz["game_id"], npz["result"], npz["packed"], npz["meta"]
        last = np.flatnonzero(np.r_[np.diff(gid) != 0, True])
        for row in last:
            if int(result[row]) == 0:
                picks.append((packed[row], meta[row]))
        if len(picks) >= cap:
            break
    return picks[:cap]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--forced-mate", default="data/derived/forced_mate.npz",
                    help="full-board forced-mate anchors (gen_forced_mate_data.py) added "
                         "to the W/L surfaces so openings can compose through a nearby mate")
    ap.add_argument("--dtm-scale", type=float, default=1.0,
                    help="divide d(g->mate) plies by this to match the field's distance "
                         "units (1.0 for a DTM-aligned field; larger for a raw field)")
    ap.add_argument("--n-waypoints", type=int, default=1500, help="waypoints per surface")
    args = ap.parse_args()
    dev = "cpu"
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.eval()
    quasi = bool(getattr(fb, "quasimetric", False))
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    rng = np.random.default_rng(0)

    def embedF(fen):
        b = chess.Board(fen)
        pl = feature_planes(encode_packed(b)[None], encode_meta(b)[None])
        with torch.no_grad():
            return fb.embed_F(torch.from_numpy(pl), torch.from_numpy(np.tile(om, (1, 1))))

    def embedB(packed, meta):
        with torch.no_grad():
            return fb.embed_B(torch.from_numpy(feature_planes(packed, meta)))

    # ---- build waypoint banks: (B-embeddings, d(g->mate)) per surface -------
    from experiments.train_lichess_fb import collect_mate_finals
    shard_dir = newest_shard_dir()
    finals = collect_mate_finals(shard_dir)          # {+1: {...}, -1: {...}} checkmate finals
    banks = {}                                        # type -> (Bg tensor, dmate np.array)

    def _sample(packed, meta, n):
        if len(packed) > n:
            idx = rng.choice(len(packed), n, replace=False)
            return packed[idx], meta[idx]
        return packed, meta

    def _finals_arr(res):                             # list[(packed,meta)] -> (packed[], meta[])
        rows = finals.get(res, [])
        if not rows:
            return None, None
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])

    # forced-mate full-board anchors (gen_forced_mate_data.py): result +/-1, dtm plies.
    fm = np.load(args.forced_mate) if Path(args.forced_mate).exists() else None

    def _fm(res):
        if fm is None:
            return None
        m = fm["result"] == res
        p, mt, dm = fm["packed"][m], fm["meta"][m], fm["dtm"][m].astype(np.float32)
        if len(p) > args.n_waypoints:
            j = rng.choice(len(p), args.n_waypoints, replace=False)
            p, mt, dm = p[j], mt[j], dm[j]
        return p, mt, dm

    # W: DTM tablebase white-wins (d=dtm) + white-mate finals (d=0) + forced white-mates
    w_pk, w_mt, w_dm = [], [], []
    if Path(args.dtm_npz).exists():
        dz = np.load(args.dtm_npz)
        idx = rng.choice(len(dz["packed"]), min(args.n_waypoints, len(dz["packed"])), replace=False)
        w_pk.append(dz["packed"][idx]); w_mt.append(dz["meta"][idx])
        w_dm.append(dz["dtm"][idx].astype(np.float32))
    fpk, fmt = _finals_arr(1)
    if fpk is not None:
        pk, mt = _sample(fpk, fmt, args.n_waypoints)
        w_pk.append(pk); w_mt.append(mt); w_dm.append(np.zeros(len(pk), np.float32))
    if _fm(1) is not None:
        p, mt, dm = _fm(1); w_pk.append(p); w_mt.append(mt); w_dm.append(dm)
    if w_pk:
        banks["W(white-win)"] = (embedB(np.concatenate(w_pk), np.concatenate(w_mt)),
                                 np.concatenate(w_dm))
    # L: black-mate finals (d=0) + forced black-mates (d=dtm)
    l_pk, l_mt, l_dm = [], [], []
    lpk, lmt = _finals_arr(-1)
    if lpk is not None:
        pk, mt = _sample(lpk, lmt, args.n_waypoints)
        l_pk.append(pk); l_mt.append(mt); l_dm.append(np.zeros(len(pk), np.float32))
    if _fm(-1) is not None:
        p, mt, dm = _fm(-1); l_pk.append(p); l_mt.append(mt); l_dm.append(dm)
    if l_pk:
        banks["L(black-win)"] = (embedB(np.concatenate(l_pk), np.concatenate(l_mt)),
                                 np.concatenate(l_dm))
    # D: draw finals (d=0)
    dfin = _draw_finals(shard_dir, args.n_waypoints)
    if dfin:
        pk = np.stack([r[0] for r in dfin]); mt = np.stack([r[1] for r in dfin])
        banks["D(draw)"] = (embedB(pk, mt), np.zeros(len(pk), np.float32))

    print(f"ckpt={Path(args.ckpt).name} quasimetric={quasi} step={pay.get('step','?')} "
          f"dtm_scale={args.dtm_scale}")
    print("waypoint banks: " + ", ".join(f"{k}={b[0].shape[0]}" for k, b in banks.items()))
    print("COMPOSED reach = -min_g[ d(F(s),B(g)) + d(g->mate) ]  (higher=closer); Δ vs equal in [].")
    print("EXPECT black-down-material: W up, L down, D down.")

    def reach_composed(f, bank):
        Bg, dmate = bank
        with torch.no_grad():
            if quasi:
                dsg = fb.distance_matrix(f, Bg)[0].cpu().numpy()      # d(F(s), B(g)) for all g
            else:
                dsg = -fb.score(f, Bg).cpu().numpy()
        return -float(np.min(dsg + dmate / args.dtm_scale))           # -min composed distance

    keys = list(banks.keys())
    print("material delta = reach(black-down) - reach(equal), PAIRED at each stage:")
    print("  " + "".join(f"{'Δ'+k:>18s}" for k in keys) + "   stage")
    open_deltas = {}
    for stage, eq_fen, dn_fen in PAIRS:
        fe, fd = embedF(eq_fen), embedF(dn_fen)
        cells = []
        for k in keys:
            d = reach_composed(fd, banks[k]) - reach_composed(fe, banks[k])
            if stage.startswith("opening"):
                open_deltas[k] = d
            cells.append(f"{d:+.3f}")
        print("  " + "".join(f"{c:>18s}" for c in cells) + f"   {stage}")
    dstr = " ".join(f"{k.split('(')[0]}={open_deltas.get(k, float('nan')):+.3f}" for k in keys)
    print(f"VERDICT PROPAGATION opening_material_delta[{dstr}] "
          f"(>0 for W, <0 for L and D => the move-2 blunder already reads as "
          f"heading to a white win; magnitude vs the endgame stage = how far the "
          f"propagation reaches back)")


if __name__ == "__main__":
    main()
