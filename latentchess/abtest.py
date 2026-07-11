"""
abtest.py — paired, matched-seed method comparisons with anytime-valid
sequential tests (e-values / confidence sequences), for comparing the
mathematical methods this project exists to compare (embedding method x
quantizer x readout aggregation x depth x goal spec) with statistical rigor
rather than a single noisy run.

EValueTest is a sign-test e-process (Beta(1/2,1/2) mixture over the null
p=1/2 "A and B equally likely to win a decisive pair"): P(sup_n e_n >= 1/alpha
| H0) <= alpha at EVERY n, so `compare()` can stop as soon as a pair's
e-value crosses 1/alpha without inflating the false-positive rate -- the
family of tests fishtest's SPRT/GSPRT belongs to, but expressed directly as
a betting e-process rather than a likelihood-ratio test against a fixed
alternative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import betaln

from latentchess.arena import tempo_ratio
from latentchess.chain import TransitionChain
from latentchess.game import play_game
from latentchess.opponents import Opponent
from latentchess.planner.policy import Policy


@dataclass
class MethodSpec:
    name: str
    build: Callable[[], Policy]   # returns a FRESH policy each call


@dataclass
class PairedOutcome:
    start: int
    seed: int
    a: dict
    b: dict


def paired_eval(chain: TransitionChain, dtm: np.ndarray, ma: MethodSpec, mb: MethodSpec,
                 black_builder: Callable[[], Opponent], starts, cap: int = 120,
                 base_seed: int = 0, seed_offset: int = 0) -> list:
    """One paired outcome per start: methods A and B each play from the SAME
    start with the SAME rng seed (hence the same opponent-randomness stream
    at least up to the point their trajectories diverge) -- matched-seed
    pairing to isolate the effect of the method itself."""
    outcomes = []
    for i, s0 in enumerate(starts):
        s0 = int(s0)
        idx = seed_offset + i
        results = {}
        for tag, spec in (("a", ma), ("b", mb)):
            rng = np.random.default_rng([base_seed, idx])
            rec = play_game(chain, spec.build(), black_builder(), s0, cap=cap, rng=rng)
            win = 1 if rec.result == "mate" else 0
            moves = len(rec.states)
            tempo = tempo_ratio(moves, float(dtm[s0])) if win else float("nan")
            results[tag] = dict(win=win, moves=moves, tempo=tempo)
        outcomes.append(PairedOutcome(start=s0, seed=base_seed, a=results["a"], b=results["b"]))
    return outcomes


class EValueTest:
    """Anytime-valid sign-test e-process for paired binary outcomes (which of
    A/B won each decisive pair). Ties (equal outcome) are ignored -- they
    carry no information about which method is better."""

    def __init__(self):
        self.n = 0    # decisive pairs seen
        self.k = 0    # of which, A won
        self.e = 1.0

    def update(self, diff: float) -> float:
        if diff > 0:
            self.n += 1
            self.k += 1
        elif diff < 0:
            self.n += 1
        if self.n > 0:
            log_e = (self.n * np.log(2.0) + betaln(self.k + 0.5, self.n - self.k + 0.5)
                      - betaln(0.5, 0.5))
            self.e = float(np.exp(log_e))
        return self.e

    def reject_at(self, alpha: float) -> bool:
        return self.e >= 1.0 / alpha


def confidence_sequence(diffs: np.ndarray, alpha: float = 0.05,
                          lo: float = -1.0, hi: float = 1.0) -> tuple:
    """Time-uniform confidence sequence for the running mean of bounded
    diffs via a union-bounded Hoeffding argument -- conservative but valid
    at every n (no peeking penalty)."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return (lo, hi)
    mean = float(diffs.mean())
    eps = (hi - lo) * np.sqrt(np.log(2.0 * (n + 1) * (n + 2) / alpha) / (2.0 * n))
    return (mean - eps, mean + eps)


@dataclass
class ComparisonRow:
    method_a: str
    method_b: str
    n_pairs: int
    decisive: int
    a_wins: int
    b_wins: int
    e_value: float
    rejected: bool
    mean_win_diff: float
    ci: tuple
    mean_tempo_a: float
    mean_tempo_b: float


def compare(chain: TransitionChain, dtm: np.ndarray, methods: dict,
            black_builder: Callable[[], Opponent], starts, alpha: float = 0.05,
            batch: int = 50, early_stop: bool = True, base_seed: int = 0) -> list:
    """Every unordered pair of `methods`, streamed in chunks of `batch`
    starts so an e-value crossing 1/alpha can stop that pair early."""
    names = sorted(methods)
    rows = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            ma, mb = methods[na], methods[nb]
            test = EValueTest()
            diffs: list[float] = []
            a_wins = b_wins = decisive = 0
            tempos_a: list[float] = []
            tempos_b: list[float] = []
            n_done = 0
            for start_idx in range(0, len(starts), batch):
                chunk = starts[start_idx:start_idx + batch]
                outcomes = paired_eval(chain, dtm, ma, mb, black_builder, chunk,
                                        base_seed=base_seed, seed_offset=start_idx)
                for o in outcomes:
                    diff = o.a["win"] - o.b["win"]
                    diffs.append(diff)
                    if diff > 0:
                        a_wins += 1; decisive += 1
                    elif diff < 0:
                        b_wins += 1; decisive += 1
                    test.update(diff)
                    if not np.isnan(o.a["tempo"]):
                        tempos_a.append(o.a["tempo"])
                    if not np.isnan(o.b["tempo"]):
                        tempos_b.append(o.b["tempo"])
                n_done += len(chunk)
                if early_stop and test.reject_at(alpha):
                    break
            ci = confidence_sequence(np.array(diffs), alpha)
            rows.append(ComparisonRow(
                method_a=na, method_b=nb, n_pairs=n_done, decisive=decisive,
                a_wins=a_wins, b_wins=b_wins, e_value=test.e, rejected=test.reject_at(alpha),
                mean_win_diff=float(np.mean(diffs)) if diffs else 0.0, ci=ci,
                mean_tempo_a=float(np.mean(tempos_a)) if tempos_a else float("nan"),
                mean_tempo_b=float(np.mean(tempos_b)) if tempos_b else float("nan"),
            ))
    return rows
