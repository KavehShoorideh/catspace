import numpy as np
import pytest

from latentchess.domains import krk
from latentchess.opponents import optimal_reply_table
from latentchess.train.curriculum import CurriculumTrainer, CurriculumConfig, Round
from latentchess.train.checkpoints import ckpt_exists


def goal_region(chain, dtm):
    return np.concatenate([np.where(dtm <= 3)[0], [chain.terminals.mate]])


@pytest.fixture(scope="module")
def krk_setup():
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm, _ = krk.compute_dtm(W, B)
    b_opt = optimal_reply_table(chain, dtm)
    return chain, dtm, b_opt


def _cfg(rounds):
    return CurriculumConfig(
        schedule=rounds, gamma=0.92, d=16, goal_region=goal_region,
        eval_n=60, train_cap=60, eval_cap=60, seed=7,
    )


def test_checkpoint_resume_matches_straight_run(krk_setup, tmp_path):
    chain, dtm, b_opt = krk_setup
    rounds = [Round(1.0, 1.0, 300, None), Round(0.3, 0.2, 300, None)]

    straight = CurriculumTrainer(chain, dtm, b_opt, _cfg(rounds))
    straight_results = straight.run(log=lambda msg: None)

    ckpt = tmp_path / "ckpt"
    partial = CurriculumTrainer(chain, dtm, b_opt, _cfg(rounds[:1]), ckpt_path=ckpt)
    partial.run(log=lambda msg: None)
    assert ckpt_exists(ckpt)

    resumed = CurriculumTrainer(chain, dtm, b_opt, _cfg(rounds), ckpt_path=ckpt)
    resumed_results = resumed.run(log=lambda msg: None)

    assert resumed_results[-1].conversion == pytest.approx(straight_results[-1].conversion)
    assert resumed_results[-1].tempo == pytest.approx(straight_results[-1].tempo, nan_ok=True)
