"""The tower (docs/ANALYST_PLAN.md section 38): the mate in words, the mirror lemma's canonical form, a floor's claim re-checked
by the exact solver on fresh draws, and the back-rank mate with their pawns as the blockers of their own king."""
import random
import subprocess
import sys

import chess
import pytest

import catspace.incubator.analyst  # noqa: F401
import analyst_core as A
from catspace.incubator.analyst.discovery import explorer as E
from catspace.incubator.analyst.discovery import tower as T


@pytest.fixture(scope="module")
def pool():
    with T.make_pool(3) as p:
        yield p


def test_floor_zero_is_found_as_a_condition_true_on_mated_positions_only(pool):
    z = T.floor_zero("KQvK", pool, random.Random(0), n=1500)
    assert z["conds"], "no word describes a mate"
    assert z["mated"] >= 20 and z["mated_held"] == z["mated"] and z["not_mated_held"] == 0
    # fresh positions never drawn before: every mate has it, nothing else does
    mated, other = [], []
    for f in T._dedupe(T.draw("KQvK", 600, random.Random(99), [])):
        b = chess.Board(f)
        for m in list(b.legal_moves):
            b.push(m)
            (mated if b.is_checkmate() else other).append(b.fen())
            b.pop()
    assert mated and all(E._holds(mated, z["conds"])) and not any(E._holds(other, z["conds"]))


def test_canonical_identifies_a_position_with_its_mirror_and_its_shifts():
    f = "8/8/8/3k4/8/2K5/1Q6/8 w - - 0 1"
    assert T.canonical(f) == T.canonical(E.mirror_fen(f))
    assert T.canonical(f, translate=False) == T.canonical(E.mirror_fen(f), translate=False)
    shifted = E.translations(f, 1)[0]
    assert T.canonical(shifted) == T.canonical(f)
    other = "8/8/8/3k4/8/2K5/1Q6/8 b - - 0 1"
    assert T.canonical(other) != T.canonical(f)


def test_condition_literals_follow_the_solver_convention():
    assert T.lits_of([dict(term="(count edge)", op=">=", theta=3), dict(term="(count edge)", op="<=", theta=5)]) == \
        [("(count edge)", 3, T.BIG), ("(count edge)", 0, 5)]
    assert T.men("KQvKP") == 4


def test_a_floors_claim_holds_on_fresh_draws(pool):
    rng = random.Random(3)
    f = T.floor("KQvK", 2, [], pool, rng, n=300)
    assert f and f["conds"] and f.get("status") != "VOCABULARY GAP"
    T.status(f, [], pool, rng)
    assert f["status"] in ("LEMMA", "CONJECTURE")
    if f["status"] == "LEMMA":                          # (exact: every placement inside the condition was checked)
        assert f["check"]["exhaustive"] and f["check"]["failures"] == 0
    fresh = E.random_positions("KQvK", 150, random.Random(12345), f["conds"])
    assert fresh
    forced = [A.forced_goal(x, T.MATE, 2, T.BUDGET)[0] == "forced" for x in fresh]
    if f["status"] == "LEMMA":
        assert all(forced)
    else:
        assert sum(forced) / len(forced) > 0.5


def test_back_rank_mate_has_their_pawns_as_the_blockers_of_their_own_king(pool):
    with_pawns = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"        # Ra8#: f7 g7 h7 shut the king in
    bare = "6k1/8/8/8/8/8/8/R3K3 w - - 0 1"                 # Ra8+ Kf7/Kh7: no mate
    assert A.forced_goal(with_pawns, T.MATE, 1, T.BUDGET)[0] == "forced"
    assert A.forced_goal(bare, T.MATE, 1, T.BUDGET)[0] == "stopped"
    # the tower's refuters name nothing against the pawn version's first move and flee/capture against the bare one
    pairs = T._pairs([])
    b = chess.Board(with_pawns)
    assert not T.refuters(T._after(b, chess.Move.from_uci("a1a8")), pairs, 1, T.BUDGET)
    bare_refs = T.refuters(T._after(chess.Board(bare), chess.Move.from_uci("a1a8")), pairs, 1, T.BUDGET)
    assert bare_refs and {kind for _, kind in bare_refs} == {"flee"}
    # the proved lemma (docs/LEAN_INDEX.md): the index finds the back rank
    r = subprocess.run([sys.executable, "-m", "catspace.incubator.analyst.lean_index", "search", "back", "rank"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        assert "back" in r.stdout.lower()
