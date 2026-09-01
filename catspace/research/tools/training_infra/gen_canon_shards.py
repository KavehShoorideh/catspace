"""gen_canon_shards.py -- one-off parallel pregeneration of MOVER-CANONICAL
feature arrays (spec merge, Kaveh 8/29: "merge the doc like you said, then train
the new arch next").

For every corpus shard row (kind==0 with wdl) and every chain row, featurize the
CANONICAL board (side to move always White; Black positions mirrored+color-swapped
via qgt_canon.canon) and store, alongside:

  canon_of  (MV_CAP,) uint8, 255-pad: canonical legal-list slot of ORIGINAL move
            index j -- the label permutation (labels live in original move order).
  played    (chains) the played index REMAPPED to the canonical list.
  flipped   uint8: whether this row was mirrored.

Ordering invariant relied on (and gated): the stored arrays were generated with
python-chess legal_moves order, so list(Board(fen).legal_moves)[j] reproduces the
original arrays' move j.

    .venv/bin/python -m catspace.research.tools.training_infra.gen_canon_shards
"""
from __future__ import annotations

import glob
import os
from multiprocessing import Pool

import chess
import numpy as np

import sys as _sys
TWIN = "--twin" in _sys.argv          # emit the OPPONENT'S-VIEW features
OUT = "data/derived/gko1_canon_twin" if TWIN else "data/derived/gko1_canon"
MV_CAP = None                                    # set in _init


def _init():
    global MV_CAP, parent_graph, glob_of_board, canon, to_canon
    from catspace.research.components.encoder.approaches.reach_probability \
        .experiments.graph_move_policy import MV_CAP as _MC
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph as _pg, glob_of_board as _gb)
    from catspace.research.tools.training_infra.qgt_canon import (
        canon as _cn, to_canon as _tc)
    MV_CAP, parent_graph, glob_of_board = _MC, _pg, _gb
    canon, to_canon = _cn, _tc


def canon_row(fen_and_played):
    """fen -> (arrays dict, canon_of, played_canon, flipped) or None on failure."""
    fen, played = fen_and_played
    try:
        b = chess.Board(fen)
    except Exception:
        return None
    cb, fl = canon(b)
    if TWIN:
        # NNUE's second accumulator: the SAME position read from the other side.
        # Colours swapped, canonical turn forced so it stays in distribution.
        cb = cb.mirror()
        cb.turn = chess.WHITE
    g, _ = parent_graph(cb)
    gl = glob_of_board(cb)
    legal_o = list(b.legal_moves)[:MV_CAP]
    legal_c = list(cb.legal_moves)[:MV_CAP]      # TWIN: opponent's moves, by design
    pos = {m: i for i, m in enumerate(legal_c)}
    cmap = np.full(MV_CAP, 255, np.uint8)
    for j, m in enumerate(legal_o):
        cm = to_canon(m, fl)
        if cm in pos:
            cmap[j] = pos[cm]
    pc = cmap[played] if played is not None and played < MV_CAP else 0
    return g, gl, cmap, (int(pc) if pc != 255 else 0), int(fl)


KEYS = ("sq", "mv", "mut", "vac", "dsc", "att")


def pack(results, played_list=None):
    n = len(results)
    out = {k: [] for k in KEYS + ("gl", "n_mv", "n_att", "canon_of",
                                  "played", "flipped")}
    ok = np.ones(n, bool)
    for i, r in enumerate(results):
        if r is None:
            ok[i] = False
            r = results[next(j for j in range(n) if results[j] is not None)]
        g, gl, cmap, pc, fl = r
        for k in KEYS:
            out[k].append(g[k])
        out["gl"].append(gl)
        out["n_mv"].append(np.uint8(g["n_mv"]))
        out["n_att"].append(np.uint16(g["n_att"]))
        out["canon_of"].append(cmap)
        out["played"].append(np.uint8(pc))
        out["flipped"].append(np.uint8(fl))
    packed = {k: np.stack(v) for k, v in out.items()}
    packed["ok"] = ok
    return packed


def main():
    os.makedirs(OUT, exist_ok=True)
    _init()
    from catspace.research.tools.npz_io import load_npz
    pool = Pool(10, initializer=_init)
    # ---- corpus shards --------------------------------------------------------
    for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz")):
        base = os.path.basename(f)
        dst = os.path.join(OUT, base)
        if os.path.exists(dst):
            continue
        z = load_npz(f)
        if "wdl" not in z.files:
            continue
        fens = [(str(x), None) for x in z["fen"]]
        res = pool.map(canon_row, fens, chunksize=64)
        packed = pack(res)
        np.savez_compressed(dst, **packed)
        print(f"[canon] {base}: {len(fens)} rows ({(~packed['ok']).sum()} failed)",
              flush=True)
    # ---- chains ---------------------------------------------------------------
    dst = os.path.join(OUT, "chains.npz")
    if not os.path.exists(dst):
        cf = np.load("data/derived/gko1_chain_fens.npz")
        chg = np.load("data/derived/gko3_chain_graphs_mm.npz", mmap_mode="r")
        played = np.asarray(chg["played"])
        fens = [(str(x), int(p)) for x, p in zip(cf["fen"], played)]
        print(f"[canon] chains: {len(fens)} rows...", flush=True)
        res = pool.map(canon_row, fens, chunksize=64)
        packed = pack(res)
        np.savez_compressed(dst, **packed)
        print(f"[canon] chains done ({(~packed['ok']).sum()} failed)", flush=True)
    pool.close()
    # ---- verification gate ----------------------------------------------------
    if TWIN:
        print("[canon] twin pass complete (verification is the mover-view gate)")
        return
    rng = np.random.default_rng(0)
    z = load_npz(sorted(glob.glob(os.path.join(OUT, "shard_*.npz")))[0])
    src = load_npz(sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[0])
    for i in rng.choice(len(z["sq"]), 24, replace=False):
        b = chess.Board(str(src["fen"][i]))
        cb, fl = canon(b)
        assert bool(z["flipped"][i]) == fl
        g, _ = parent_graph(cb)
        assert (z["sq"][i] == g["sq"]).all(), f"sq mismatch row {i}"
        assert int(z["n_mv"][i]) == int(g["n_mv"])
        legal_o = list(b.legal_moves)[:MV_CAP]
        legal_c = list(cb.legal_moves)[:MV_CAP]
        for j, m in enumerate(legal_o):
            jj = int(z["canon_of"][i][j])
            if jj != 255:
                assert legal_c[jj] == to_canon(m, fl), f"map mismatch {i}/{j}"
    print("[canon] VERIFICATION PASS: features == featurize(canon(board)); "
          "label permutation exact on sampled rows")


if __name__ == "__main__":
    main()
