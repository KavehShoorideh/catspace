"""
train/curriculum.py — approximate policy iteration with an opponent curriculum.

Round k: white = eps-greedy on current reach scores, black = eps-optimal
(annealed toward 0, i.e. toward true optimal defense); sample games;
accumulate transition counts; re-estimate the successor measure; refresh
scores; evaluate vs true optimal defense.

Collapses the ~6 near-identical copies of this loop (exp_policy_iteration.py,
exp_krkn.py, exp_krkn2.py, exp_krrk.py, gen_ui_data_pi.py, region_map.py) into
one trainer, parameterized by the round schedule and the (optional)
reverse-start curriculum (dtm_cap, annealed outward -- exp_krkn2.py's
contribution on top of the plain PI loop).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp

from catspace.chain import TransitionChain, P_from_counts, counts_from_transitions
from catspace.scoring import TerminalScores, fill_terminal_state_scores
from catspace.planner.readout import ReplyAgg, greedy_policy
from catspace.planner.policy import TablePolicy, EpsGreedy
from catspace.opponents import EpsOptimalDTM
from catspace.game import rollout_transitions
from catspace.arena import evaluate, ArenaResult
from catspace.cone.tabular import TabularFB
from catspace.cone.embedding import make_goal, reach
from catspace.train.checkpoints import TrainerState, save_ckpt, load_ckpt, ckpt_exists


@dataclass
class Round:
    eps_white: float
    eps_black: float
    n_games: int
    dtm_cap: int | None = None


@dataclass
class CurriculumConfig:
    schedule: list[Round]
    gamma: float
    d: int
    goal_region: Callable[[TransitionChain, np.ndarray], np.ndarray]
    eval_n: int = 300
    train_cap: int = 120
    eval_cap: int = 70
    seed: int = 100
    agg: ReplyAgg = ReplyAgg.MEAN
    n_oversample: int = 8
    track_stratum_cross: str | None = None
    frac_curriculum: float = 0.7
    start_pool: np.ndarray | None = None   # restrict rollout starts to this stratum (e.g. KRkn only)


def curriculum_starts(dtm: np.ndarray, won: np.ndarray, dtm_cap: int | None, n: int,
                       frac_curriculum: float = 0.7, rng: np.random.Generator | None = None,
                       pool_all: np.ndarray | None = None) -> np.ndarray:
    """Reverse-start curriculum: frac_curriculum of starts drawn from WON
    states with dtm <= dtm_cap (annealed outward across rounds); the rest
    uniform. dtm_cap=None -> pure uniform starts. `pool_all` restricts BOTH
    the curriculum and uniform draws to a specific stratum (e.g. KRkn only,
    excluding the easier KRk stratum) -- defaults to all live states."""
    rng = rng if rng is not None else np.random.default_rng()
    if pool_all is None:
        pool_all = np.arange(len(dtm))
    if dtm_cap is None:
        return pool_all[rng.integers(0, len(pool_all), size=n)]
    mask = won[pool_all] & (dtm[pool_all] <= dtm_cap)
    pool = pool_all[mask]
    n_cur = int(frac_curriculum * n)
    return np.concatenate([
        pool[rng.integers(0, len(pool), size=n_cur)],
        pool_all[rng.integers(0, len(pool_all), size=n - n_cur)],
    ])


class CurriculumTrainer:
    def __init__(self, chain: TransitionChain, dtm: np.ndarray, b_opt: np.ndarray,
                 cfg: CurriculumConfig, ckpt_path: Path | None = None):
        self.chain = chain
        self.dtm = dtm
        self.won = np.isfinite(dtm)
        self.b_opt = b_opt
        self.cfg = cfg
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self.ts = TerminalScores.big()
        self.region = cfg.goal_region(chain, dtm)

    def run(self, log: Callable[[str], None] = print) -> list[ArenaResult]:
        chain, cfg = self.chain, self.cfg
        counts = sp.csr_matrix((chain.n, chain.n))
        scores = np.zeros(chain.n)
        k0 = 0
        if self.ckpt_path is not None and ckpt_exists(self.ckpt_path):
            state = load_ckpt(self.ckpt_path)
            counts, scores, k0 = state.counts, state.scores, state.round
            log(f"resumed at round {k0}")

        results: list[ArenaResult] = []
        pool_all = cfg.start_pool if cfg.start_pool is not None else np.arange(chain.n_live)
        won_idx = pool_all[self.won[pool_all]]
        for k in range(k0, len(cfg.schedule)):
            rnd = cfg.schedule[k]
            base_local = greedy_policy(scores, chain, cfg.agg, self.ts)
            white = EpsGreedy(TablePolicy(base_local), rnd.eps_white)
            black = EpsOptimalDTM(self.b_opt, rnd.eps_black)
            starts = curriculum_starts(self.dtm, self.won, rnd.dtm_cap, rnd.n_games,
                                        cfg.frac_curriculum, np.random.default_rng(cfg.seed + k),
                                        pool_all=cfg.start_pool)
            rows, cols, n_mate = rollout_transitions(chain, white, black, starts, cap=cfg.train_cap,
                                                      rng=np.random.default_rng(cfg.seed + k))
            counts = (counts + counts_from_transitions(rows, cols, chain.n)).tocsr()
            P, _visited = P_from_counts(counts, chain.terminals)
            emb = TabularFB.fit(P, cfg.gamma, cfg.d, n_oversample=cfg.n_oversample, seed=0)
            goal = make_goal("curriculum", self.region, emb)
            scores = fill_terminal_state_scores(reach(emb, goal), chain, self.ts)

            eval_local = greedy_policy(scores, chain, cfg.agg, self.ts)
            eval_white = TablePolicy(eval_local)
            eval_black = EpsOptimalDTM(self.b_opt, 0.0)
            rng_eval = np.random.default_rng(cfg.seed + 100000)
            eval_starts = won_idx[rng_eval.integers(0, len(won_idx), size=cfg.eval_n)]
            result = evaluate(chain, self.dtm, eval_white, eval_black, eval_starts, cap=cfg.eval_cap,
                               track_stratum_cross=cfg.track_stratum_cross,
                               auc_scores=scores[:chain.n_live], auc_won_mask=self.won)
            results.append(result)
            log(f"round {k}: data_mates={n_mate} conversion={result.conversion:.3f} "
                f"tempo={result.tempo:.2f} rook_loss={result.rook_loss:.3f} extra={result.extra}")

            if self.ckpt_path is not None:
                save_ckpt(TrainerState(round=k + 1, counts=counts, scores=scores, F=emb.F, B=emb.B),
                          self.ckpt_path)
        return results
