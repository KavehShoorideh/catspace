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
    assert paired_e([held] + [lost] * 49, [lost] * 50)[2] < 2                    # one board of 50: not evidence
    w, l, e = paired_e([held] * 26 + [lost] * 24, [held] * 3 + [lost] * 47)
    assert (w, l) == (23, 0) and e > 1e5                                         # the round-2 trap stage's margin
    assert paired_e([[0, 0, 6]] * 40 + [[0, 0, 1]] * 10, [[0, 0, 2]] * 50)[:2] == (40, 10)   # ties on held/won: progress decides
    assert paired_e([held] * 7 + [lost], [lost] * 7 + [held])[2] < E_PROMOTE      # 7 to 1: e 7.0
    assert paired_e([held] * 10 + [lost], [lost] * 10 + [held])[2] >= E_PROMOTE   # 10 to 1: e 30.9 (one-sided)


def test_one_vote_per_game():
    from catspace.incubator.analyst.discovery.defenders import votes
    a = [[1, 0, 0]] * 3 + [[0, 0, 0]] + [[1, 0, 0]]
    b = [[0, 0, 0]] * 3 + [[1, 0, 0]] + [[0, 0, 0]]
    assert votes(a, b) == (4, 1)                                                  # by board
    assert votes(a, b, ["g1", "g1", "g1", "g2", "g3"]) == (2, 1)                  # by game: g1 counts once


def test_the_one_sided_mixture():
    from catspace.incubator.analyst.discovery.defenders import sign_e
    assert sign_e(0, 0) == 1.0
    assert 1.7 < sign_e(10, 5) / 0.683 < 1.9                                      # about twice the two-sided 0.68 at 10 to 5
    assert sign_e(5, 10) < sign_e(10, 5)


def test_choose_and_confirm_on_different_games():
    from catspace.incubator.analyst.discovery.verifier import split
    boards = [f"b{i}" for i in range(100)]
    games = [i // 3 for i in range(100)]                                          # three boards a game
    choose, check, cg = split(boards, 50, games, seed="t")
    gc = {games[boards.index(b)] for b in choose}
    assert not gc & set(cg) and len(check) == len(cg) <= 50 and 47 <= len(choose) <= 50


def test_spread_takes_one_game_at_a_time():
    from catspace.incubator.analyst.discovery.verifier import spread, split
    boards = [f"b{i}" for i in range(60)]
    games = [0] * 40 + [1] * 10 + [2] * 5 + [3] * 5                              # one long game, three short
    out, og = spread(boards, games, 12)
    assert og == [0, 1, 2, 3] * 3
    choose, check, cg = split(out, 50, og, seed="t")
    assert len(choose) <= 6 and set(cg) and not {og[out.index(b)] for b in choose} & set(cg)


def test_the_phase_claim_is_tested_by_game():
    import numpy as np
    # 20 held-out games of 10 positions: the drawn ones spend 8 of 10 positions inside R, the won ones 2
    games = [g for g in range(20) for _ in range(10)]
    y = np.array([g < 10 for g in games])                                        # games 0-9 drawn
    inside = np.array([(i % 10) < (8 if g < 10 else 2) for i, g in enumerate(games)])
    held = np.ones(len(games), bool)
    assert C.by_game_p(inside, y, games, held) < 1e-3
    assert C.by_game_p(np.array([(i % 10) < 5 for i in range(200)]), y, games, held) > 0.3   # no difference
    assert C.by_game_p(inside, y, games, np.zeros(len(games), bool)) == 1.0                 # nothing held out


def test_data_phase_labels_a_game_by_its_phase():
    rows = lambda theirs_after: [dict(material="KRRvKR", lead=5, theirs=5, fen="a"), dict(material="KRRvKR", lead=5, theirs=5, fen="b"),
                                 dict(material="KRRvK", lead=10, theirs=theirs_after, fen="c")]
    res = [dict(seed=1, out="win", rows=rows(0)), dict(seed=2, out="draw", rows=[dict(material="KRRvKR", lead=5, theirs=5, fen="d")])]
    assert C.data_phase(res, "KRRvKR") == [("a", False, 1), ("b", False, 1), ("d", True, 2)]


def test_too_little_data_is_not_a_gap():
    import numpy as np
    games = list(range(100))
    train = np.array([g % 2 == 0 for g in games])
    few = np.array([g in (1, 3) for g in games])                                 # 2 refuted held-out claims
    assert "refuted held-out boards" in C.too_little(few, train, games, phase=False)
    many = np.array([g % 4 == 1 for g in games])                                 # 25 refuted of 50 held out
    assert C.too_little(many, train, games, phase=False) is None
    # by game: ONE drawn held-out game (10 positions) against 49 won -- a perfect predictor cannot pass
    pos = [g for g in games for _ in range(10)]
    y = np.array([g == 1 for g in pos]); tr = np.array([g % 2 == 0 for g in pos])
    assert "perfect predictor" in (C.too_little(y, tr, pos, phase=True) or "")
