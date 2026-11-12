"""Plan 78.38 stage 1: the ladder of unit sets, the summary with a fake labeller, falls, the open order, a set claim's shape and its drill. Sandbox store in tmp; no tablebase."""
import json

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import construction as C
from catspace.incubator.analyst.discovery import exact as EX
from catspace.incubator.analyst.discovery import game_store as GS

GAME = "chess"
KP_K = "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"
KQ_K = "8/8/8/4k3/8/8/8/3QK3 w - - 0 1"
KQ_K_SWAPPED = "3qk3/8/8/8/8/8/4K3/8 b - - 0 1"


def holds(start):
    """True where the side to move holds the first part of the set (the holder)."""
    return C.role_of(GAME, start, CE.state(GAME, start, [])[1]) == "holder"


class Fake(EX.ExactLabeller):
    source = "fake"

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def _class(self, variant, start, moves):
        return self.fn(start, moves)


def test_signature_is_canonical_under_swapping_the_sides():
    a, b = C.signature(GAME, KQ_K), C.signature(GAME, KQ_K_SWAPPED)
    assert a == b == ((("king", 1), ("queen", 1)), (("king", 1),))
    assert hash(a) == hash(b)
    assert C.sides_of(GAME, KQ_K) == ("white", "black") and C.sides_of(GAME, KQ_K_SWAPPED) == ("black", "white")
    assert C.size_of(a) == 3 and C.show(a) == "king+queen v king"
    assert C.signature(GAME, "8/8/8/4k3/8/8/8/4K3 w - - 0 1") == ((("king", 1),), (("king", 1),))


def test_the_ladder_for_sizes_two_and_three():
    two = C.ladder(GAME, 2)
    assert two == [((("king", 1),), (("king", 1),))]
    three = C.ladder(GAME, 3)
    assert three[0] == two[0] and all(C.size_of(u) == 3 for u in three[1:])
    assert len(three) == len(set(three)) == 1 + 5                      # each added kind once: the same set with the sides swapped is one set
    for u in three[1:]:
        assert u[0] != u[1] and sum(n for _k, n in u[1]) == 1


def test_role_of_the_mover():
    assert C.role_of(GAME, KQ_K, "white") == "holder" and C.role_of(GAME, KQ_K, "black") == "other"
    assert C.role_of(GAME, "8/8/8/4k3/8/8/8/4K3 w - - 0 1", "white") == "even"


def test_sample_gives_legal_live_distinct_positions_of_the_set():
    u = C.signature(GAME, KQ_K)
    pos = C.sample(GAME, u, 30, 1)
    assert 0 < len(pos) <= 30 and len(set(pos)) == len(pos)
    for p in pos:
        _f, mover, legal, outcome, _k = CE.state(GAME, p, [])
        assert C.signature(GAME, p) == u and legal and outcome is None
    assert C.sample(GAME, u, 30, 1) == pos


def test_observe_arithmetic_with_a_fake_labeller():
    u = C.signature(GAME, KQ_K)
    pos = C.sample(GAME, u, 40, 2)
    # holder to move wins; the other side to move loses, except one position the labeller does not cover and one draw
    other = [p for p in pos if not holds(p)]
    assert len(other) > 1
    gaps = {other[0]: None, other[1]: "draw"}
    lab = Fake(lambda start, moves: gaps.get(start, "win" if holds(start) else "loss"))
    s = C.observe(GAME, u, pos, lab)
    holders = [p for p in pos if holds(p)]
    assert s["positions"] == len(pos) and s["skipped"] == 1 and s["labelled"] == len(pos) - 1 and s["covered"] is True
    assert s["by_mover"]["holder"]["n"] == len(holders) and s["by_mover"]["holder"]["shares"]["win"] == 1.0
    o = s["by_mover"]["other"]
    assert o["n"] == len(other) - 1 and o["counts"]["draw"] == 1 and o["counts"]["loss"] == o["n"] - 1
    assert o["shares"]["loss"] == pytest.approx((o["n"] - 1) / o["n"]) and sum(o["shares"].values()) == pytest.approx(1.0)
    assert s["decisive_exists"] is True
    draws = C.observe(GAME, u, pos, Fake(lambda *_: "draw"))
    assert draws["decisive_exists"] is False and draws["by_mover"]["holder"]["shares"]["draw"] == 1.0
    none = C.observe(GAME, u, pos, Fake(lambda *_: None))
    assert none["covered"] is False and none["labelled"] == 0 and none["by_mover"] == {}


def test_falls_to_on_a_real_sample():
    u = C.signature(GAME, "8/8/8/4k3/8/3Q4/8/4K3 w - - 0 1")
    # queen takes nothing here; a position where the queen can be taken or the king captures it falls to the smaller set
    near = "8/8/8/8/8/3k4/3Q4/4K3 b - - 0 1"               # black king can capture the queen (defended by the king: that move is illegal) -> use a loose queen instead
    loose = "8/8/8/8/8/3k4/2Q5/7K b - - 0 1"
    f = C.falls_to(GAME, u, [loose])
    assert f == {((("king", 1),), (("king", 1),)): 1}
    assert C.falls_to(GAME, u, [near]) == {}
    pawn = C.signature(GAME, "8/4P3/8/8/8/8/k7/7K w - - 0 1")
    promo = C.falls_to(GAME, pawn, ["8/4P3/8/8/8/8/k7/7K w - - 0 1"])
    assert {C.show(t) for t in promo} == {"king+queen v king", "king+rook v king", "bishop+king v king", "king+knight v king"}


def test_open_order_puts_a_set_after_every_set_it_falls_to():
    a, b, c, d = [((("k", i),), ()) for i in range(4)]
    falls = {d: {c: 3, a: 1}, c: {b: 2}, b: {a: 1}, a: {}}
    order = C.open_order([d, c, b, a], falls)
    assert order == [a, b, c, d]
    assert C.open_order([a, b], {a: {b: 1}, b: {a: 1}}) == [a, b]                   # a cycle is broken, nothing is lost
    assert C.open_order([a, b], {a: {((("zzz", 9),), ()): 1}}) == [a, b]              # a target outside the ladder is no order


@pytest.fixture
def store(tmp_path):
    old = GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT
    d = tmp_path / "learned" / GAME
    d.mkdir(parents=True)
    rec = json.loads((GS.RULES / "learned" / GAME / "vocabulary.keys.json").read_text())
    rec["words"] = True
    (d / "vocabulary.keys.json").write_text(json.dumps(rec))
    GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = tmp_path / "learned", tmp_path / "snapshots"
    try:
        yield tmp_path
    finally:
        GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = old


def test_a_set_claim_has_the_claim_shape_and_its_drill_counts_right_and_wrong(store):
    u = C.signature(GAME, KQ_K)
    pos = C.sample(GAME, u, 60, 1)
    # the claim is right where the holder is to move (a win) and wrong at the one position labelled a draw
    odd = next(p for p in pos if holds(p))
    lab = Fake(lambda start, moves: "draw" if start == odd else ("win" if holds(start) else "loss"))
    s = C.observe(GAME, u, pos, lab)
    claims = C.set_claims(GAME, s, 0.9, vocabulary="keys", min_n=5)
    by = {c["meta"]["mover"]: c for c in claims}
    win = by["holder"]
    assert win["claim"] == {"class": "win"} and win["stats"] == {} and win["meta"]["vocabulary"] == "keys"
    assert win["body"] == [{"relation": "exactly_king_queen_against_king", "args": ["us"], "neg": False}]
    assert by["other"]["claim"] == {"class": "loss"} and by["other"]["body"][0]["args"] == ["them"]
    CE.check_lemma(json.dumps(win))
    d = {r["name"]: r for r in C.drill_set(GAME, u, pos, claims, lab, "keys")}
    holders = sum(1 for p in pos if holds(p))
    w = d[win["name"]]
    assert w["binds"] == holders and w["wrong"] == 1 and w["right"] == holders - 1
    assert w["exceptions"] == [{"position": odd, "exact_class": "draw"}]
    o = d[by["other"]["name"]]
    assert o["binds"] == len(pos) - holders and o["wrong"] == 0
    # the word says "exactly this set": it does not bind on another set
    other = C.signature(GAME, KP_K)
    pos2 = C.sample(GAME, other, 20, 1)
    lab2 = Fake(lambda *_: "win")
    z = C.drill_set(GAME, other, pos2, [win], lab2, "keys")[0]
    assert z["binds"] == 0
    assert C.define_set_word(GAME, u, "keys")["name"] == "exactly_king_queen_against_king"          # defined once, the same word again


def test_a_claim_needs_the_bar_and_enough_positions(store):
    u = C.signature(GAME, KQ_K)
    pos = C.sample(GAME, u, 40, 1)
    s = C.observe(GAME, u, pos, Fake(lambda start, moves: "win" if holds(start) else "draw"))
    assert [c["meta"]["mover"] for c in C.set_claims(GAME, s, 0.99, vocabulary="keys", min_n=5)] == ["holder", "other"]
    assert C.set_claims(GAME, s, 0.99, vocabulary="keys", min_n=10_000) == []
    mixed = C.observe(GAME, u, pos, Fake(lambda start, moves: ("win", "draw")[sum(map(ord, start)) % 2]))
    assert C.set_claims(GAME, mixed, 0.99, vocabulary="keys", min_n=5) == []


def test_the_sandbox_store_name_is_checked(tmp_path):
    with pytest.raises(ValueError):
        C.sandbox_store(GAME, "keys", "anything_else")
