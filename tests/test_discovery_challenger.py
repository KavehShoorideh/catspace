"""The challenger's two gates (challenger.py): the held-out excess test and the trap / blind-spot split."""
import numpy as np

from catspace.incubator.analyst.discovery import challenger as C


def claims(n_in, fail_in, n_out, fail_out):
    y = np.array([True] * fail_in + [False] * (n_in - fail_in) + [True] * fail_out + [False] * (n_out - fail_out))
    r = np.array([True] * n_in + [False] * n_out)
    return y, r


def test_excess_does_not_rise_with_the_base_rate():
    # the 2026-10-12 case: the frame fails ~73% everywhere; inside R 81% -- "base + 0.10" (0.83) barred it
    y, r = claims(300, 243, 700, 487)              # 0.81 inside, ~0.70 outside
    assert C.excess(y, r) < 0.01
    y, r = claims(300, 210, 700, 490)              # 0.70 inside and out: no excess
    assert C.excess(y, r) > 0.3


def test_excess_is_one_sided_and_handles_empty_regions():
    y, r = claims(300, 150, 700, 560)              # fails LESS inside R
    assert C.excess(y, r) > 0.99
    y = np.array([True, False, True]); r = np.array([True, True, True])
    assert C.excess(y, r) == 1.0                   # nothing outside R: no test


def test_split_into_traps_and_blind_spots():
    cand = lambda p, trap, checked=60: dict(p=p, trap=trap, checked=checked, lift=1.0, recall=0.5)
    cands = [cand(1e-6, 0.87), cand(1e-6, 0.15), cand(1e-6, 0.95, checked=5), cand(0.02, 0.9), cand(1e-6, 0.80)]
    traps, blinds = C.split(cands)                 # alpha 0.05 / 5 = 0.01
    assert [c["trap"] for c in traps] == [0.87, 0.80]
    assert [c["trap"] for c in blinds] == [0.15]


def test_a_blind_spot_on_the_traps_own_boards_is_not_answered_twice():
    c = lambda trap, lift, boards: dict(trap=trap, lift=lift, recall=0.5, boards=boards)
    trap = c(0.92, 1.1, ["a", "b", "c", "d"])
    twin = c(0.77, 1.3, ["a", "b", "c", "e"])      # 3 of its 4 refuted boards are the trap's
    other = c(0.15, 1.2, ["e", "f", "g", "a"])     # 1 of 4
    assert C.pick([trap], [twin, other]) == (trap, other)
    assert C.pick([], [twin, other]) == (None, twin)


def test_paired_e_value():
    from catspace.incubator.analyst.discovery.defenders import paired_e, E_PROMOTE
    held, lost = [1, 0, 3], [0, 0, 3]
    assert abs(paired_e([held] + [lost] * 49, [lost] * 50)[2] - 1) < 1e-9       # one board of 50: e = 1, not evidence
    w, l, e = paired_e([held] * 26 + [lost] * 24, [held] * 3 + [lost] * 47)
    assert (w, l) == (23, 0) and e > 1e5                                 # the round-2 trap stage's margin
    assert paired_e([[0, 0, 6]] * 40 + [[0, 0, 1]] * 10, [[0, 0, 2]] * 50)[:2] == (40, 10)   # ties on held/won: progress decides
    assert paired_e([held] * 10 + [lost], [lost] * 10 + [held])[2] < E_PROMOTE    # 10 to 1: e 15.5
    assert paired_e([held] * 11 + [lost], [lost] * 11 + [held])[2] >= E_PROMOTE   # 11 to 1: e 26


def test_choose_and_confirm_on_different_boards():
    from catspace.incubator.analyst.discovery.verifier import split
    assert split(list(range(100)), 50) == (list(range(50)), list(range(50, 100)))
    a, b = split(list(range(30)), 50)
    assert len(a) == len(b) == 15 and not set(a) & set(b)
