"""OpponentModel component: maia2 move-distribution priors (poison-guarded).
The live-z estimator (catspace/style/live.py) is the upgrade slot here."""
from __future__ import annotations

import chess


def make_maia2_policy(m2, m2_inf, elo_opp: int, elo_self: int):
    """maia2 move distribution at the opponent's rating frame -> {Move: prior}.
    Poison-guarded (maia2 preprocessing IndexErrors on dict-gap positions, the
    m4f killer): any failure returns None -> the tree keeps its default priors."""
    import pandas as pd

    def opp_policy(board):
        try:
            df = pd.DataFrame({"fen": [board.fen()], "move": ["0000"],
                               "elo_self": [int(elo_opp)], "elo_oppo": [int(elo_self)]})
            df, _ = m2_inf.inference_batch(df, m2, verbose=False, batch_size=1,
                                           num_workers=0)
            out = {}
            for uci, p in df["move_probs"][0].items():
                try:
                    out[chess.Move.from_uci(uci)] = float(p)
                except ValueError:
                    pass
            return out or None
        except Exception:
            return None
    return opp_policy
