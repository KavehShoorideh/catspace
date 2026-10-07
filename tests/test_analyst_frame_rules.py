"""The rules' hook at zero search (frame_player.RULES, Kaveh 2026-10-05): no rules = the baseline; a proved rule adds
exactly its claim to the moves it covers and nothing else; a conjecture only breaks ties, never demoting a tied move it
says nothing about and never touching a move below the top; the passer as the first proved rule credits its push by
the conservative gain, which equals the zone search's leaf credit."""
import random

import chess
import pytest

from catspace.incubator.analyst import frame_player as FP

FENS = [chess.STARTING_FEN,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "2kr3r/ppp2ppp/2nb1n2/7q/3P2b1/2P2N2/PP1NBPPP/R1BQ1RK1 b - - 0 1",
        "8/8/1pp1k1p1/4Pp1p/2PK1P1P/pP6/P7/8 w - - 2 36"]


@pytest.fixture()
def rules():
    old = list(FP.RULES)
    FP.RULES[:] = []
    yield FP.RULES
    FP.RULES[:] = old


def _values(fen):
    from catspace.incubator.analyst.fates import clear_caches
    clear_caches()
    return FP.choose(chess.Board(fen))


def test_a_rule_that_never_fires_changes_nothing(rules):
    base = {f: _values(f) for f in FENS}
    rules.append(FP.Rule("silent", "proved", lambda b: {}))
    rules.append(FP.Rule("silent2", "conjecture", lambda b: {}))
    for f in FENS:
        c = _values(f)
        assert c["values"] == base[f]["values"] and c["best"] == base[f]["best"] and c["rules"] == {}


def test_a_proved_rule_adds_exactly_its_claim(rules):
    base = _values(chess.STARTING_FEN)["values"]
    rules.append(FP.Rule("e4", "proved", lambda b: {"e2e4": (2.0, "a claim"), "zz9z": (5.0, "not a move")}))
    c = _values(chess.STARTING_FEN)
    assert c["values"]["e2e4"] == pytest.approx(base["e2e4"] + 2.0)
    assert all(c["values"][u] == base[u] for u in base if u != "e2e4")
    assert c["best"] == ["e2e4"] and c["rules"] == {"e2e4": ["e4: a claim"]} and FP.RULES_FIRED[0] == 1


def test_a_conjecture_only_breaks_ties(rules):
    base = _values(chess.STARTING_FEN)
    assert len(base["best"]) == 20                                             # the opening: every move tied at 0
    rules.append(FP.Rule("centre", "conjecture", lambda b: {"e2e4": (1.0, "centre"), "d2d4": (1.0, "centre"), "a2a3": (0.0, "edge")}))
    c = _values(chess.STARTING_FEN)
    assert set(c["best"]) == {"e2e4", "d2d4"}                                  # the claimed tied moves rank above the rest of the tie
    assert c["values"]["a2a3"] == pytest.approx(base["values"]["a2a3"] - 1e-3)
    assert c["values"]["g1f3"] == pytest.approx(base["values"]["g1f3"] - 1e-3) # an unclaimed tied move ranks below a positive claim
    rules[:] = [FP.Rule("doom", "conjecture", lambda b: {"e2e4": (-2.0, "loses")})]
    c3 = _values(chess.STARTING_FEN)
    assert c3["values"] == base["values"]                                      # a negative claim demotes nobody (conjecture)
    tactical = "2kr3r/ppp2ppp/2nb1n2/7q/3P2b1/2P2N2/PP1NBPPP/R1BQ1RK1 b - - 0 1"
    b0 = _values(tactical)
    below = next(u for u, v in b0["values"].items() if v < max(b0["values"].values()))
    rules[:] = [FP.Rule("lift", "conjecture", lambda b, u=below: {u: (9.0, "preferred")})]
    c2 = _values(tactical)
    assert c2["best"] == b0["best"] and c2["values"][below] == b0["values"][below]   # a move below the top is never lifted


def test_passer_rule_credits_the_push(rules):
    from catspace.incubator.analyst.lean_passer_v2_proto import QUEEN_GAIN, proved_passer_gain, proved_passer_gains, proved_passers, random_pawn_ending
    b = chess.Board("7k/8/P7/8/8/8/8/1K6 w - - 0 1")                             # a6 queens; the black king is far
    gains = proved_passer_gains(b)
    assert set(gains) == {chess.A6} and gains[chess.A6][0] == QUEEN_GAIN
    rules.append(FP.RULE_BOOK["passer"]())
    c = _values(b.fen())
    assert c["best"] == ["a6a7"] and c["values"]["a6a7"] == pytest.approx(QUEEN_GAIN) and "proved" in c["rules"]["a6a7"][0]
    rng = random.Random(3)
    n = 0
    for _ in range(300):
        pe = random_pawn_ending(rng)
        if pe is None:
            continue
        g = proved_passer_gains(pe)
        assert set(g) <= {sq for sq, _ in proved_passers(pe)} and all(0 < v <= QUEEN_GAIN for v, _ in g.values())
        assert proved_passer_gain(pe) == pytest.approx(max((v for v, _ in g.values()), default=0.0))
        n += 1
    assert n > 100


def test_veto_charges_only_what_the_ledger_did_not(rules):
    """A veto cuts a move by the certified gain less the ledger's own one-ply charge for it, and never lifts anything."""
    base = _values(chess.STARTING_FEN)["values"]
    rules.append(FP.Rule("v", "proved", lambda b, values, trace: {"e2e4": (3.0, "refuted")}, "veto", True))
    c = _values(chess.STARTING_FEN)
    assert c["values"]["e2e4"] == pytest.approx(base["e2e4"] - 3.0) and "e2e4" not in c["best"]
    assert all(c["values"][u] == base[u] for u in base if u != "e2e4")
    values = {"a": 2.0, "b": 2.0}; trace = {"a": ((1.5, None, []), (0.0, None, []))}
    rules[:] = [FP.Rule("v2", "proved", lambda b, values, trace: {"a": (2.0, "x")}, "veto", True)]
    FP.apply_rules(chess.Board(), values, trace)
    assert values["a"] == pytest.approx(1.5) and values["b"] == 2.0         # the ledger had charged 1.5 of the 2.0


@pytest.mark.skipif(not __import__("catspace.incubator.analyst.lean_ideas", fromlist=["default_binary"]).default_binary().exists(), reason="lean/ideas not built")
def test_refutation_vetoes_a_capture_the_kernel_refutes(rules):
    """Qxd5 grabs a pawn; the QUIET Nf4 then forks queen and rook (a check the frames would have read as a forcing
    move): the kernel certifies Black's gain and the veto removes Qxd5 from the top."""
    b = chess.Board("8/5ppk/6pp/3p3n/8/8/4R3/3Q2K1 w - - 0 1")
    nb = b.copy(); nb.push_uci("d1d5")                                           # the position's own properties, checked before the rule's:
    assert not nb.is_check() and chess.Move.from_uci("h5f4") in nb.legal_moves   # the capture gives no check, the fork is legal,
    nf = nb.copy(); nf.push_uci("h5f4")
    assert nf.is_attacked_by(chess.BLACK, chess.D5) and nf.is_attacked_by(chess.BLACK, chess.E2) and not nf.is_check()   # quiet,
    assert not any(nf.gives_check(m) for m in nf.legal_moves)                   # and White has no counter-check (three wrong positions today)
    from catspace.incubator.analyst.fates import clear_caches
    clear_caches()
    assert "d1d5" in FP.choose(b)["best"]                                       # the ledger likes the free pawn
    rules.append(FP.RULE_BOOK["refute"]())
    clear_caches()
    c = FP.choose(b)
    assert "d1d5" in c["rules"] and "certified by Lean for +" in c["rules"]["d1d5"][0]
    assert "d1d5" not in c["best"]


def test_passer_setup_rule_credits_the_setup_move_once():
    """The composition rule credits a king move after which the d-pawn queens whatever they reply; a promotion is
    left to the ledger and a proved passer's own push to the passer rule, so the two rules never add up."""
    from catspace.incubator.analyst.frame_player import _passer_fire, _passer_setup_fire
    b = chess.Board("1K6/3P2P1/8/k4P2/8/8/3PP3/8 w - - 0 1")
    direct = _passer_fire(b)
    setup = _passer_setup_fire(b)
    assert "b8c8" in setup and setup["b8c8"][0] > 0
    assert "d7d8q" not in setup
    assert not (set(direct) & set(setup))
