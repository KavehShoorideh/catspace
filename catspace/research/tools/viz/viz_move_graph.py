#!/usr/bin/env python
"""viz_move_graph.py -- render the GNN's edge types around one piece in a real position.

Three panels, one per relation family, drawn with python-chess's SVG board:
    ATT      square -> square attack edges touching the piece (the real, static structure)
    SRC/DST  the piece's legal moves -- each arrow IS a move-node, connected to its from-square
             (src_of) and to-square (dst_of)
    MUT      one chosen move's hypothetical mutation edges: the squares the moved piece would
             NEWLY attack -- what the move-node broadcasts back onto the board

The sample is mined from the cached move graphs: a knight move that gives check and creates many
new attacks, so the mutation panel has something dramatic to say.

    .venv/bin/python -m catspace.research.tools.viz.viz_move_graph
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", default="data/derived/move_graphs_300000_6545146.npz")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/generated/move_graph_edges.html")
    args = ap.parse_args()

    import chess
    import chess.svg
    from catspace.research.tools.training_infra.concepts_reference import board_from_row
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)

    z = np.load(args.graphs)
    mv, p_row = z["mv"], z["p_row"]
    # a knight move (piece type 2), giving check, creating >= 6 new attacks
    cand = np.argwhere((mv[:, :, 0] == 2) & (mv[:, :, 6] == 1) & (mv[:, :, 7] >= 6))
    i, j = cand[np.random.default_rng(3).integers(0, len(cand))]
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=400, n_piecedown=args.n_piecedown)
    b = board_from_row(tr.tok[p_row[i]], tr.glob[p_row[i]])
    frm, to = int(mv[i, j, 1]), int(mv[i, j, 2])
    move = chess.Move(frm, to)
    piece = b.piece_at(frm)

    # panel 1: real attack edges touching this piece
    arr1 = [chess.svg.Arrow(frm, s, color="#2ca02c") for s in b.attacks(frm)]
    arr1 += [chess.svg.Arrow(a, frm, color="#d62728")
             for a in b.attackers(not piece.color, frm)]
    svg1 = chess.svg.board(b, arrows=arr1, size=340,
                           fill={frm: "#ffe08a"})

    # panel 2: this piece's legal moves = its move-nodes
    arr2 = [chess.svg.Arrow(m.from_square, m.to_square, color="#1f77b4")
            for m in b.legal_moves if m.from_square == frm]
    svg2 = chess.svg.board(b, arrows=arr2, size=340, fill={frm: "#ffe08a"})

    # panel 3: mutation edges of the chosen move
    pre = set(b.attacks(frm))
    b.push(move)
    new = sorted(set(b.attacks(to)) - pre)
    arr3 = [chess.svg.Arrow(to, s, color="#ff7f0e") for s in new]
    svg3 = chess.svg.board(b, arrows=arr3, size=340, lastmove=move,
                           fill={to: "#ffe08a"}, check=b.king(b.turn) if b.is_check() else None)
    b.pop()

    san = b.san(move)
    html = f"""<title>GNN edge types — {san}</title><meta charset="utf-8">
<style>body{{font-family:ui-monospace,Menlo,monospace;max-width:1160px;margin:1.5rem auto;
padding:0 1rem}} .row{{display:flex;gap:1.2rem;flex-wrap:wrap}} .p{{flex:1;min-width:340px}}
h1{{font-size:1.05rem}} h2{{font-size:.9rem;margin:.4rem 0}} p{{font-size:.8rem;line-height:1.45;
color:#555}} @media (prefers-color-scheme: dark){{p{{color:#aaa}}}}</style>
<h1>The GNN's edge types around the {piece.symbol().upper()} on
{chess.square_name(frm)} — sample move {san}</h1>
<p>One heterogeneous graph per position: 64 square-nodes + one node per legal move
(~{int(z["n_mv"][i])} here). Each relation has its own message weights.</p>
<div class="row">
<div class="p"><h2>1 · ATT — real structure (sq→sq)</h2>{svg1}
<p><b style="color:#2ca02c">green</b>: squares this knight attacks now.
<b style="color:#d62728">red</b>: enemy pieces attacking it. These edges exist for every piece
on the board; the panel shows one piece's slice. This is the graph the trunk had to discover
from raw tokens — here it is wired in.</p></div>
<div class="p"><h2>2 · SRC/DST — move-nodes (sq→mv)</h2>{svg2}
<p><b style="color:#1f77b4">blue</b>: each arrow IS a move-node. Its from-square feeds it via a
<i>src_of</i> edge, its to-square via <i>dst_of</i>. All ~{int(z["n_mv"][i])} legal moves are
scored in ONE forward pass — this is what replaced the ~34 per-child trunk encodes.</p></div>
<div class="p"><h2>3 · MUT — hypothetical (mv→sq), for {san}</h2>{svg3}
<p><b style="color:#ff7f0e">orange</b>: squares the knight would NEWLY attack after {san}
(check included). The move-node broadcasts its one-ply consequence back onto the board —
the exact information whose absence made every earlier scorer sibling-blind: the badness or
brilliance of a move is not on the board until you look one ply deep, so the graph looks
for it.</p></div>
</div>"""
    with open(args.out, "w") as f:
        f.write(html)
    print(f"sample: {san} from parent row {p_row[i]} | new attacks {len(new)} | wrote {args.out}")


if __name__ == "__main__":
    main()
