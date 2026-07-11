"""
Deterministic, seconds-scale end-to-end smoke tests: build a domain, train a
tiny curriculum, confirm it beats a random baseline. Full-scale KRkn training
is covered by the slow-marked test below and by experiments/train_krkn.py.
"""
import numpy as np
import pytest

from latentchess.domains import krk
from latentchess.opponents import optimal_reply_table, RandomOpponent
from latentchess.planner.policy import RandomPolicy
from latentchess.train.curriculum import CurriculumTrainer, CurriculumConfig, Round
from latentchess.arena import evaluate


def goal_region(chain, dtm):
    return np.concatenate([np.where(dtm <= 3)[0], [chain.terminals.mate]])


def test_krk_micro_curriculum_beats_random():
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm, _ = krk.compute_dtm(W, B)
    b_opt = optimal_reply_table(chain, dtm)

    cfg = CurriculumConfig(
        schedule=[Round(1.0, 1.0, 1500, None)],   # single random-play round
        gamma=0.92, d=16, goal_region=goal_region,
        eval_n=100, train_cap=60, eval_cap=60, seed=42,
    )
    trainer = CurriculumTrainer(chain, dtm, b_opt, cfg)
    results = trainer.run(log=lambda msg: None)
    learned_conversion = results[-1].conversion

    rng = np.random.default_rng(1)
    starts = rng.integers(0, chain.n_live, size=200)
    random_result = evaluate(chain, dtm, RandomPolicy(), RandomOpponent(), starts, cap=60)

    assert learned_conversion >= 2 * max(random_result.conversion, 1e-9)


@pytest.mark.slow
def test_krkn_full_curriculum_reproduces_headline():
    """Runs the full exp_krkn2.py-equivalent schedule (~2 min) and checks the
    headline numbers are in the right ballpark (training is a chaotic-ish
    stochastic process; RNG-stream differences vs the original mean this is
    a noise-band check, not bit-exact -- see tests/baselines/expected.json)."""
    from latentchess.domains import krkn
    from latentchess.planner.readout import ReplyAgg
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_krkn as tk

    chain = krkn.build_chain(verbose=False)
    dtm = krkn.compute_dtm(chain)
    b_opt = optimal_reply_table(chain, dtm)
    n2 = chain.strata["KRkn"].stop

    cfg = CurriculumConfig(
        schedule=tk.SCHEDULE, gamma=0.93, d=48, goal_region=tk.goal_region,
        eval_n=300, train_cap=120, eval_cap=70, seed=100, agg=ReplyAgg.MEAN,
        track_stratum_cross="KRk", start_pool=np.arange(n2),
    )
    trainer = CurriculumTrainer(chain, dtm, b_opt, cfg)
    results = trainer.run(log=lambda msg: None)
    final = results[-1]
    assert final.conversion > 0.4          # baseline documented 0.487
    assert final.extra["win_draw_auc"] > 0.6   # baseline documented 0.702
