"""PLAN 63: a check that is also a threat counts as a follow-up of a delayed threat (63.1, Lean `Ideas.Forcing.checkThreat` / `followUp`), and the DUTY word 'pinned to a threat'
(63.2, Lean `Ideas.Duty.duty` / `dutyBound` / `pin_dutyBound`). pos2 (Kaveh): White to move a rook down, 1.Bf2! -- their Bh4 is pinned to the mate on h7."""
import chess
import pytest

import catspace.incubator.analyst  # noqa: F401
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R

POS2 = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"
H4 = chess.parse_square("h4")
BARE = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"                                      # Rh8+ is a bare check: nothing stands behind it
FROZEN = "2K5/6N1/1k3n1Q/8/8/2p5/8/8 w - - 0 1"                             # the black Nf6 is pinned to its king by the Qh6: no legal move


@pytest.fixture(autouse=True)
def _terms():
    BV.install_terms()


def _bf2():
    b = chess.Board(POS2)
    return b, chess.Move.from_uci("e3f2")


# ---- 63.1 ------------------------------------------------------------------------------------------------------------------------------------------

def test_63_1_pos2_bf2_is_a_delayed_threat_naming_the_check_that_is_a_threat():
    b, m = _bf2()
    why = R.delayed_threat(b, m, 3, "w")
    assert why is not None and why.startswith("delayed threat")
    assert "by a check that is also a threat" in why and "Qh7#" in why and ("after Be7" in why or "after Bf6" in why or "after Bg5" in why or "after Qe7" in why or "after Rg8" in why)


def test_63_1_a_bare_check_is_not_a_threat_and_a_discovered_check_with_a_mate_behind_is():
    b = chess.Board(BARE)
    assert R.check_threat(b, chess.Move.from_uci("h1h8"), 3, "w") is None
    b, m = _bf2()
    c = b.copy()
    c.push(m)
    c.push_uci("h4e7")                                                     # 1.Bf2 Be7: the h-file is open to the Qh2 behind the Nh3
    why = R.check_threat(c, chess.Move.from_uci("h3g1"), 3, "w")
    assert why == "check that also threatens mate (Qh7#)"


def test_63_1_the_filter_is_the_lean_definition_brute_force():
    """Lean `delayed_followUp`: delayed <=> after EVERY legal reply some NEW move of ours is a follow-up (a threat, a mate threat, or a check that is also a threat)."""
    b, m = _bf2()
    me = True
    c = b.copy()
    c.push(m)
    standing = {x.uci() for x in b.legal_moves if R._follow_up(b, x, 3, me)}
    ok = True
    for r in c.legal_moves:
        d = c.copy()
        d.push(r)
        ok = ok and any(R._follow_up(d, x, 3, me) for x in d.legal_moves if x.uci() not in standing)
    assert ok == (R.delayed_threat(b, m, 3, "w") is not None)


def test_63_1_a_check_followup_needs_the_threat_behind_it():
    """A reply after which only bare checks follow breaks the delayed threat: Bf2 with the h-file closed (Bh4 stays, the knight on h3 shields) -- the bishop retreats leave only checks that are threats."""
    b, m = _bf2()
    c = b.copy()
    c.push(m)
    c.push_uci("h4e7")
    ours = [x for x in c.legal_moves if R._follow_up(c, x, 3, True)]
    assert ours and all(R._follow_up(c, x, 3, True).startswith("check that also threatens") for x in ours)


def test_63_1_ror8t_kg3_is_a_delayed_threat_by_mate_threats_only():
    """RoR8t (black, Kg3!): with (b) a threat of mate counts at any stake, Kg3 is a delayed threat at the material stakes <= 5 (after every reply black has a mate threat: Re4 / Rd4 ...), and not at the
    stakes where only a mate counts as before. The DECISION is graded by the analyser run (the missing word is the endgame claim after Rg8+, not forcing)."""
    p = [x for x in PA.load_set("rand23_5") if x["id"] == "RoR8t"][0]
    b = chess.Board(p["fen"])
    m = chess.Move.from_uci(p["answer"])
    assert "delayed threat" in R.delayed_threat(b, m, 3, "b")
    assert R.delayed_threat(b, m, 9, "b") is None


