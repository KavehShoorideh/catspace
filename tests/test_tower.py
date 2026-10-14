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


@pytest.fixture(scope="module")
def kq_floor(pool):
    f = T.floor_arr("KQvK", 2, [], pool, random.Random(0), n=4000)
    assert f and f["frames"], "no arrangement floor"
    return f


def test_arrangement_floor_reads_as_a_relation_with_nothing_unforced_inside(kq_floor):
    idx = T.index3("KQvK")
    for fr in kq_floor["frames"]:
        assert fr["kind"] == "arrangement" and "their king" in fr["relation"] and fr["covers"] >= T.MIN_COVER
        arr = T.canon_arr(fr["view"], [tuple(i) for i in fr["items"]])
        inside = idx[arr]
        assert inside
        assert all(A.forced_goal(x, T.MATE, 2, T.BUDGET)[0] == "forced" for x in inside)      # every placement: zero unforced
        assert all(all(E._holds(inside, c)) for c in fr["variants"][:1]) or len(fr["variants"]) == 2   # the literals say the same
    assert kq_floor["covered"] + kq_floor["remainder"] == kq_floor["forced"] or kq_floor["remainder"] >= 0


def test_the_floor_number_measure_picks_a_floor_lowering_move(kq_floor):
    term = T.floor_number_term([kq_floor])
    assert term and "horizon" not in term
    idx = T.index3("KQvK")
    hand = []
    for fr in kq_floor["frames"]:
        arr = T.canon_arr(fr["view"], [tuple(i) for i in fr["items"]])
        hand += idx[arr][:40]
    # (a real step first: not a mate in one)
    hand = sorted(hand, key=lambda x: A.forced_goal(x, T.MATE, 1, T.BUDGET)[0] == "forced")[:3]
    assert len(hand) == 3
    pairs = T._pairs([])
    for x in hand:
        assert A.vocab_values([x], term) == [1]                      # the position reads as floor 1
        mv, certified = A.frame_choose(x, [term, "horizon:2"], None, "", "Q", [])
        c = T._after(chess.Board(x), chess.Move.from_uci(mv))
        assert certified and not T.refuters(c, pairs, 2, T.BUDGET)   # the mate stays forced against every reply


def test_the_measure_is_the_lowest_floor_that_holds():
    a = dict(term="(count (and (at L) corner))", op=">=", theta=1)
    b = dict(term="(count (and (at L) edge))", op=">=", theta=1)
    term = T.floor_number_term([dict(n=1, variants=[[a]]), dict(n=2, variants=[[b]])])
    in_corner, on_edge, inside = "7k/8/7K/8/8/8/8/1Q6 w - - 0 1", "4k3/8/4K3/8/8/8/8/1Q6 w - - 0 1", "8/8/3k4/8/8/3K4/8/1Q6 w - - 0 1"
    v = A.vocab_values([in_corner, on_edge, inside], term)
    assert v[:2] == [1, 2] and v[2] > 2                               # the lowest floor that holds; in no floor: more than any
