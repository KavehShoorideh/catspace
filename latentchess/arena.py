"""arena.py — evaluating a policy against an opponent over many games."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from latentchess.chain import TransitionChain, KIND_ONGOING
from latentchess.game import play_game
from latentchess.opponents import Opponent
from latentchess.planner.policy import Policy
from latentchess.util import auc as _auc


@dataclass
class ArenaResult:
    conversion: float          # mate rate over starts
    tempo: float                # mean (white moves) / ceil(dtm/2) over mates
    exact_dtm_rate: float        # fraction of ALL starts mating in exactly ceil(dtm/2) white moves
    rook_loss: float             # fraction ending drawn via a mid-move capture (not an explicit stalemate)
    cap_rate: float               # fraction hitting the ply cap unresolved
    n: int
    extra: dict = field(default_factory=dict)   # optional: via_<stratum>, win_draw_auc, ...


def tempo_ratio(white_moves: int, dtm_plies: float) -> float:
    return white_moves / max(1.0, np.ceil(dtm_plies / 2))


def evaluate(chain: TransitionChain, dtm: np.ndarray, white: Policy, black: Opponent,
             starts, cap: int = 70, seed: int = 99,
             track_stratum_cross: str | None = None,
             auc_scores: np.ndarray | None = None,
             auc_won_mask: np.ndarray | None = None) -> ArenaResult:
    rng = np.random.default_rng(seed)
    n = len(starts)
    mates = exact = rook_lost = crossed_at_mate = crossed_any = capped = 0
    ratios: list[float] = []
    for s0 in starts:
        s0 = int(s0)
        rec = play_game(chain, white, black, s0, cap=cap, rng=rng)
        wm = len(rec.states)
        crossed = (track_stratum_cross is not None
                   and any(st in chain.strata[track_stratum_cross] for st in rec.states[1:]))
        crossed_any += bool(crossed)
        if rec.result == "mate":
            mates += 1
            d0 = float(dtm[s0])
            ratios.append(tempo_ratio(wm, d0))
            exact += wm == int(np.ceil(d0 / 2))
            crossed_at_mate += bool(crossed)
        elif rec.result == "draw" and rec.final_kind == KIND_ONGOING:
            rook_lost += 1
        elif rec.result == "cap":
            capped += 1
    extra: dict = {}
    if track_stratum_cross is not None:
        # via_<stratum>: fraction of MATES that crossed (the KRkn convention);
        # <stratum>_drop_rate: fraction of ALL games that ever crossed (the KRRk convention)
        extra[f"via_{track_stratum_cross}"] = crossed_at_mate / max(mates, 1)
        extra[f"{track_stratum_cross}_drop_rate"] = crossed_any / n
    if auc_scores is not None and auc_won_mask is not None:
        extra["win_draw_auc"] = _auc(auc_scores[auc_won_mask], auc_scores[~auc_won_mask])
    return ArenaResult(
        conversion=mates / n,
        tempo=float(np.mean(ratios)) if ratios else float("nan"),
        exact_dtm_rate=exact / n,
        rook_loss=rook_lost / n,
        cap_rate=capped / n,
        n=n,
        extra=extra,
    )
