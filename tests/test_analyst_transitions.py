"""The transition table: coordinates, the contrast, the cells, and the planner's reading of a synthetic table."""
import chess
import numpy as np

from catspace.incubator.analyst import transitions as T


def test_coords_and_contrast_and_cells():
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    c = T.coords(b, chess.WHITE)
    assert c["queens_on"] == 2 and c["pawns_ours"] == 8 and c["phase"] == 62 and c["material"] == 0 and c["open_files"] == 0
    # a synthetic contrast: when the phase falls the result is 0.7, else 0.4
    rows = []
    for i in range(60):
        c0 = dict(c); c1 = dict(c); c1["phase"] = c["phase"] - (8 if i % 2 else 0)
        rows.append(("cell", c0, c1, 0.7 if i % 2 else 0.4))
    t = T.contrast(rows, None, min_n=10)
    assert abs(t["cell"]["phase"]["diff"] - 0.3) < 1e-9 and t["cell"]["phase"]["n_fell"] == 30
    assert T.cell_for(b, 250, 1650) == ["kaveh winning | even", "lichess 1400-1700 | even"]
    assert T.lookup({"cell": t["cell"]}, "cell", "phase")["diff"] > 0 and T.lookup({}, "x", "phase") is None
