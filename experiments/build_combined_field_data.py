#!/usr/bin/env python
"""experiments/build_combined_field_data.py -- Kaveh 2026-08-03: ONE field trained on BOTH the
lichess human data and the self-generated SF-vs-SF opening pool, so their basins can be compared
in a single coordinate system instead of across two separately-trained (and therefore unalignable)
phi spaces.

Emits a light METADATA npz only -- deliberately NOT a concatenated planes/features file:

  * `planes` is never read by train_iqe_head.py (it trains on the precomputed trunk-feature
    memmap), and concatenating it would cost ~16GB for nothing.
  * the two trunk-feature memmaps (~36GB each) are likewise NOT copied into one file. This npz
    records `source` + `local_row` per row, and the trainer gathers from the two memmaps in
    place -- saving ~72GB of disk and the copy time. See DualFeats in train_iqe_head.py.

Per-row fields computed here (all vectorized, no Python loops over rows):
  y          : mover-POV basin label in {WIN=0, DRAW=1, LOSS=2} -- the CE target.
  n_to_end   : plies from this row to the game's terminal position (>=0; floored to >=1 by
               reachability_target at use). At the terminal itself this is 0 -> 1 ply from its
               pole; one ply earlier the mover is the WINNER, also 1 ply from the win pole.
  is_terminal: the game's final position (`move == ""`, exactly one per game).
  is_tail    : within the last `--tail-plies` of the game -- the ONLY rows that get the radial
               anchor (see below).
  source     : 0 = human lichess, 1 = SF-vs-SF.  local_row: row index within that source's feats.
  game       : OFFSET so ids stay unique across sources. Critical: build_pairs() keys on `game`,
               so colliding ids would silently fabricate cross-dataset "same-game" pairs.

Two labelling choices worth stating rather than burying:

1. **`result` (realized), not `ending` (TB-corrected), is the DEFAULT basin label** -- switchable
   with --label. `ending` is TB-OVERRIDDEN at <=7 pieces and its generator docstring calls it
   "the committor target", which is tempting, but it is the wrong committor here, for two
   measured reasons:
     * It breaks the radial anchor's units. Under `ending`, SF-vs-SF gains 9,659 terminal rows
       labelled mover-WIN (vs exactly 0 under `result`): positions the tablebase scores as won
       that the game actually adjudicated/repeated into a draw. Anchoring those at n=1 ply from
       the win pole asserts "a win lands next ply" when a TB-won position can be 40+ plies from
       mate. n_to_end is a TRAJECTORY quantity, so its label must be the trajectory's outcome.
     * It erases the very difference being measured. A committor is defined with respect to the
       dynamics you are modelling; human and SF basins differ precisely BECAUSE their dynamics
       differ. TB-overriding both at <=7 pieces forces them to share a label exactly where
       endgame technique diverges most.
   `ending` is still carried in the npz -- it is the natural exact BOUNDARY CONDITION for a
   later committor term, which is how the MSM literature actually uses it.

2. **The radial anchor is restricted to tail rows, and assumes surprisal=0 there.** Regressing
   distance-to-pole on plies-remaining over ALL plies would train the field to predict game
   LENGTH -- contingent on both players' later choices, not a geometric property of the position.
   Within the last few plies the outcome is effectively locked, so P(outcome)~1 and the
   surprisal channel of reachability_target is ~0. That channel stays wired at the call site for
   when a policy model can supply a real P(path); v1 passes 0 and says so.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.losses import WIN, DRAW, LOSS

ENDING_WHITE_WIN, ENDING_WHITE_LOSS = 0, 5       # per gen_field_data_fullgame.py (draw = 1..4)


def white_pov_outcome(d, label):
    """-> (N,) int8 white-POV outcome in {+1,0,-1} from either labelling."""
    if label == "result":
        return d["result"].astype(np.int8)                   # what the game actually scored
    e = d["ending"]                                          # TB-overridden at <=7 pieces
    return np.where(e == ENDING_WHITE_WIN, 1,
                    np.where(e == ENDING_WHITE_LOSS, -1, 0)).astype(np.int8)


def basin_labels(white_pov, ply):
    """(N,) white-POV outcome + ply -> (N,) mover-POV basin label in {WIN,DRAW,LOSS}. Vectorized.

    Side-to-move convention is `stm_white = (ply % 2 == 1)` -- White moves on ODD ply. This
    matches gen_field_data_fullgame.py's own `stm_white = (ply % 2 == 1)`; the basin_umap work
    originally had it backwards (ply % 2 == 0) and it was a real, corrected bug. Do not "fix" it.
    """
    mover_is_white = (ply % 2 == 1)
    mover_wins = (white_pov == 1) == mover_is_white
    return np.where(white_pov == 0, DRAW, np.where(mover_wins, WIN, LOSS)).astype(np.int8)


def plies_to_end(game, ply):
    """(N,) -> plies from each row to its game's LAST row. Vectorized via np.maximum.at over a
    dense group index (no per-game Python loop -- there are ~200k games)."""
    _, inv = np.unique(game, return_inverse=True)
    last = np.zeros(inv.max() + 1, dtype=ply.dtype)
    np.maximum.at(last, inv, ply)
    return (last[inv] - ply).astype(np.int32)


def load_meta(path, source, game_offset, label):
    d = np.load(path, allow_pickle=True)
    ply = d["ply"].astype(np.int32)
    game = d["game"].astype(np.int64) + game_offset
    ending = d["ending"]
    move = d["move"]
    n_to_end = plies_to_end(game, ply)
    return dict(
        y=basin_labels(white_pov_outcome(d, label), ply),
        n_to_end=n_to_end,
        is_terminal=np.asarray([str(m) == "" for m in move], dtype=bool),
        game=game, ply=ply,
        dtz=d["dtz"].astype(np.int32),
        result=d["result"].astype(np.int8),
        ending=ending.astype(np.int8),
        source=np.full(len(ply), source, dtype=np.int8),
        local_row=np.arange(len(ply), dtype=np.int64),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human", default="data/derived/field_std_v1.npz")
    ap.add_argument("--sf", default="data/derived/opening_pool_sfsf.npz")
    ap.add_argument("--human-feats", default="data/derived/trunk_feats/t1-256x10__field_std_v1.npy")
    ap.add_argument("--sf-feats", default="data/derived/trunk_feats/t1-256x10__opening_pool_sfsf.npy")
    ap.add_argument("--tail-plies", type=int, default=4,
                    help="rows within this many plies of the end get the radial anchor "
                         "(gen_field_data_fullgame.py's --tail default is 4, so 4 is what exists)")
    ap.add_argument("--label", choices=["result", "ending"], default="result",
                    help="basin label source; see the module docstring for why `result` is default")
    ap.add_argument("--out", default="data/derived/field_combined_v1.npz")
    args = ap.parse_args()
    t0 = time.time()

    human = load_meta(args.human, source=0, game_offset=0, label=args.label)
    offset = int(human["game"].max()) + 1                    # keep ids unique across sources
    sf = load_meta(args.sf, source=1, game_offset=offset, label=args.label)

    out = {k: np.concatenate([human[k], sf[k]]) for k in human}
    out["is_tail"] = (out["n_to_end"] < args.tail_plies)

    N = len(out["y"])
    print(f"[combined] N={N:,}  (human {len(human['y']):,} + SF {len(sf['y']):,})  "
          f"game-id offset {offset:,}  [{time.time()-t0:.0f}s]")
    for name, m in [("human", out["source"] == 0), ("SF-vs-SF", out["source"] == 1)]:
        y = out["y"][m]
        print(f"  {name:9s} basins  win {int((y==WIN).sum()):>9,}  draw {int((y==DRAW).sum()):>9,}"
              f"  loss {int((y==LOSS).sum()):>9,}")
    term = out["is_terminal"]
    print(f"  terminals {int(term.sum()):,} | tail rows (<{args.tail_plies} to end) "
          f"{int(out['is_tail'].sum()):,}")
    for name, m in [("human", out["source"] == 0), ("SF-vs-SF", out["source"] == 1)]:
        yt = out["y"][term & m]
        print(f"  {name:9s} TERMINAL basins  win {int((yt==WIN).sum()):>7,}  "
              f"draw {int((yt==DRAW).sum()):>7,}  loss {int((yt==LOSS).sum()):>7,}"
              f"   <- win~0 is expected: at a terminal the mover is the side that got mated")
    # the pre-terminal row is what actually anchors the WIN pole
    pre = out["n_to_end"] == 1
    yp = out["y"][pre]
    print(f"  PRE-terminal rows {int(pre.sum()):,}: win {int((yp==WIN).sum()):,} "
          f"draw {int((yp==DRAW).sum()):,} loss {int((yp==LOSS).sum()):,}  <- the win-pole anchors")

    # sanity: every game contributes exactly one terminal, and n_to_end==0 there
    assert (out["n_to_end"][term] == 0).all(), "terminal rows must have n_to_end == 0"
    assert int(term.sum()) == len(np.unique(out["game"][term])), "one terminal per game"

    meta = dict(feats=[args.human_feats, args.sf_feats], tail_plies=args.tail_plies, label=args.label,
                human=args.human, sf=args.sf, game_offset=offset)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out, _meta=np.array([repr(meta)], dtype=object))
    print(f"wrote {args.out}  [{time.time()-t0:.0f}s]")
    print(f"NOTE feats are NOT concatenated (saves ~72GB): {args.human_feats} | {args.sf_feats}")


if __name__ == "__main__":
    main()
