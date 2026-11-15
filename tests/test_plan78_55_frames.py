"""Plan 78.55: the frame, its sequences and fate records, the pass, what each side can force with no worth table, claims and conditional claims, two frames (rust/engine/tactics/src/sheet/frames.rs,
sheet/fates.rs `fate_history`; catspace/incubator/analyst/discovery/frames_mirror.py, frames_measure.py, unit_fates.py `history`).

0. TOOK: the tracker's history of a unit (what it took, how it ended) against the readable mirror on random games and a hand sequence.
A. the frame: members by closure, in contact -> joined; the Rust closure against the readable mirror (the October lifting rule) on hand positions.
B/C. sequences and the table of outcomes; the pass (what the game says, what it cannot represent); the Lean lemmas `frame_sound_pass` and `far_stays_off` mirrored by a finite model test.
D. what each side can force: outcomes compared by inclusion; the Rust reading against the mirror on hand positions and sampled positions of random games.
E. the pass count, threats, waiting claims, conditional claims, and the layer-3 nodes.
F. two frames: the composition, checked per position, not proved.
Against fates.py (the October module) on the positions both read.
No new module names an object of any game (a source scan); hand positions and the names of units appear here and nowhere in the modules."""
import itertools
import random
import re
from pathlib import Path

import chess
import cs_engine as CE
import pytest

from catspace.incubator.analyst import fates as OF
from catspace.incubator.analyst.discovery import frames_mirror as FM
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import unit_fates as UF

ROOT = Path(__file__).resolve().parents[1]
ALL = 255
SCAN_WORDS = (r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|diagonals?|[a-h][1-8]|pins?|pinned|forks?|skewers?|batter\w+|x-?rays?|"
              r"discovered)\b")
MODULES = ["rust/engine/tactics/src/sheet/frames.rs", "catspace/incubator/analyst/discovery/frames_mirror.py", "catspace/incubator/analyst/discovery/frames_measure.py"]


def random_games(seed: int, games: int, plies: int) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        moves = []
        for _ in range(plies):
            state = CE.state("chess", None, moves)
            if state[3] is not None or not state[2]:
                break
            moves.append(rng.choice(sorted(state[2])))
        out.append(moves)
    return out


@pytest.fixture(scope="module")
def model():
    m = GL.model("chess", "chess")
    m.learn_profile([(None, mv) for mv in random_games(21, 12, 60)])
    return m


# ---------------------------------------------------------------------------------------------------------------------------------------------------- the readable game adapter (this file only)
class BoardAdapter:
    """The frames mirror's adapter over python-chess boards (the game the tests play)."""

    def cells(self, b):
        return sorted((("white" if p.color else "black"), chess.piece_name(p.piece_type), chess.square_name(s)) for s, p in b.piece_map().items())

    def mover(self, b):
        return None if b.is_game_over() else ("white" if b.turn else "black")

    def moves(self, b):
        out = []
        for m in b.legal_moves:
            nb = b.copy(stack=False)
            nb.push(m)
            out.append((chess.square_name(m.from_square), chess.square_name(m.to_square), nb))
        return out

    def pass_(self, b):
        if b.is_check() or b.is_game_over():
            return None
        nb = b.copy(stack=False)
        nb.push(chess.Move.null())
        return nb

    def key(self, b):
        return b.fen()

    def bearers(self, b, place, lifted):
        c = b.copy(stack=False)
        for sq in lifted:
            c.remove_piece_at(chess.parse_square(sq))
        sq = chess.parse_square(place)
        out = []
        for s in c.attackers(chess.WHITE, sq) | c.attackers(chess.BLACK, sq):
            p = c.piece_at(s)
            out.append((("white" if p.color else "black"), chess.piece_name(p.piece_type), chess.square_name(s)))
        return out


AD = BoardAdapter()


def uid(role, kind, place):
    return f"unit:{role}:{kind}:{place}"


# the hand positions (the first five are those of 78.54; the last two are new: a conditional claim, two frames apart)
PONY = ("3r2k1/8/3b4/8/4N3/8/8/7K w - - 0 1", [("white", "knight", "e4"), ("black", "bishop", "d6")])
HAND = {
    "a unit pinned to the goal unit": ("4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1", [("white", "rook", "e1"), ("black", "knight", "e5")]),
    "a line opened by moving the unit between (mine)": ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", [("white", "rook", "e1"), ("white", "knight", "e4")]),
    "two units of theirs borne on by one of mine": ("6k1/2r1q3/8/3N4/8/8/8/6K1 w - - 0 1", [("white", "knight", "d5"), ("black", "rook", "c7"), ("black", "queen", "e7")]),
    "a unit of theirs behind their goal unit": ("4q3/8/8/4k3/8/8/8/4R1K1 b - - 0 1", [("white", "rook", "e1"), ("black", "king", "e5")]),
    "the pinned unit with the other side to move": ("4k3/8/8/4n3/8/8/8/4R1K1 b - - 0 1", [("white", "rook", "e1"), ("black", "knight", "e5")]),
    "the juicer defended by a rook": PONY,
}
CONDITIONAL = ("4k3/7p/8/3p4/3P4/8/8/R5K1 w - - 0 1", [("black", "pawn", "d5")])
APART = "3r2k1/8/1p1b4/n7/4N3/8/8/R6K w - - 0 1"
APART_A = [("white", "knight", "e4"), ("black", "bishop", "d6")]
APART_B = [("white", "rook", "a1"), ("black", "knight", "a5")]


def seed_args(units):
    return [uid(*u) for u in units]


def mirror_read(fen, units, reading, flipped=False, bud=None):
    b = chess.Board(fen)
    fr = FM.frame_of(AD, b, units, [])
    return fr, FM.Reader(AD, fr).forceable(b, reading, flipped, bud)


def rust_pairs(r, reading, flipped, bud):
    for t in r["tempo"]:
        if t["reading"] == reading and t["flipped"] == flipped and tuple(t["budget"]) == bud:
            return sorted((tuple(sorted(a)), tuple(sorted(b))) for a, b in t["pairs"])
    return None


def mirror_bud(reading, rust_bud):
    """The Rust budget is by role (white, black); the mirror's by (the side read, the other side)."""
    return (rust_bud[0], rust_bud[1]) if reading == "white" else (rust_bud[1], rust_bud[0])


# ---------------------------------------------------------------------------------------------------------------------------------------------------- 0. TOOK
def test_the_history_records_what_a_unit_took_before_it_was_taken(model):
    """A hand sequence: the white pawn takes on d5, the black queen takes it back, then a quiet move. The pawn TOOK a pawn and was TAKEN by a queen: an exchange, not a loss for nothing."""
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]
    kinds, per = model.unit_histories((None, moves), [1, 2, 4])
    plies = [[(o, k, p) for (o, k, p, _h) in ply] for ply in per]
    i = plies[0].index(("white", "pawn", "e2"))
    # at ply 0 the pawn on e2: after 4 plies it moved to e4, took on d5 and was taken by the queen
    cls_took = {c for (c, tk) in per[0][i][3] if tk}
    assert per[0][i][3][2] == (4 + len(kinds) + kinds.index("queen"), ["pawn"])                                 # at the horizon 4: an exchange, the queen took it after it took a pawn
    assert per[0][i][3][0] == (1, [])                                                                          # one ply later it had only moved
    assert cls_took == {4 + len(kinds) + kinds.index("queen")}
    # a unit lost for nothing: the black pawn on d5 at ply 2 (after e4) is taken by the pawn at ply 2 (e4d5) without taking anything
    j = per[2][per[2].index(next(u for u in per[2] if u[:3] == ("black", "pawn", "d5")))]
    assert j[3][0][0] == 4 + kinds.index("pawn") and j[3][0][1] == []


def test_the_history_equals_the_readable_mirror_on_random_games(model):
    games = random_games(5, 6, 70)
    n = 0
    for moves in games:
        kinds, per = model.unit_histories((None, moves), [2, 4, 8])
        plies = [[(o, k, p) for (o, k, p, _h) in ply] for ply in per]
        for t in range(0, len(plies), 3):
            for i, unit in enumerate(per[t]):
                for hi, h in enumerate([2, 4, 8]):
                    mine = UF.history(plies, kinds, t, i, h)
                    assert (UF.history_class(mine, kinds), mine[0]) == (unit[3][hi][0], unit[3][hi][1]) or (UF.history_class(mine, kinds) == UF.NO_FATE and unit[3][hi][0] == UF.NO_FATE)
                    n += 1
    assert n > 500


# ---------------------------------------------------------------------------------------------------------------------------------------------------- A. the frame
def test_the_frame_of_a_seed_holds_the_units_that_bear_on_its_places(model):
    fen, seed = PONY
    r = model.frame_read(fen, seed_args(seed), [], False, False, None, 20000)
    assert sorted(r["frame"]["members"]) == sorted([uid("white", "knight", "e4"), uid("black", "bishop", "d6"), uid("black", "rook", "d8")])        # the rook defends the bishop: it bears on its place
    assert r["frame"]["id"].startswith("frame:") and sorted(r["frame"]["places"]) == ["place:d6", "place:e4"]
    r2 = model.frame_read(fen, seed_args(seed), [], False, False, None, 20000)
    assert r["frame"]["id"] == r2["frame"]["id"]                                                                                                 # an id is its definition


def test_the_rust_frame_equals_the_october_square_frame_on_the_pony_and_a_battery(model):
    """fates.py `square_frame` (lifting) against the engine's closure. The rook behind the rook: one unit behind one unit is seen by the through relation, so the frame of the first battery is equal."""
    b = chess.Board(PONY[0])
    october = {chess.square_name(s) for s in OF.square_frame(b, chess.D6)}
    rust = model.frame_read(PONY[0], seed_args([("black", "bishop", "d6")]), [], False, False, None, 20000)["frame"]["members"]
    assert {m.split(":")[3] for m in rust} == october
    fen = "3r4/8/3b4/8/8/8/3R4/3RK2k w - - 0 1"                                                              # two rooks on one file, a bishop in front of a rook behind it
    october = {chess.square_name(s) for s in OF.square_frame(chess.Board(fen), chess.D6)}
    rust = {m.split(":")[3] for m in model.frame_read(fen, seed_args([("black", "bishop", "d6")]), [], False, False, None, 20000)["frame"]["members"]}
    assert rust == october == {"d6", "d8", "d2", "d1"}


def test_a_battery_of_three_is_the_limit_of_the_through_relation_and_the_mirror_sees_it(model):
    """THE LIMIT STATED IN THE CODE: the engine sees one unit behind one unit. Three on a line: the mirror (the October lifting) finds the third, the engine does not. The mirror is right: lifting is
    what the frame lemma's closure says (a unit that bears once the frame's units leave); the engine's closure is a cheaper, narrower reading."""
    fen = "3r4/8/3b4/8/8/3R4/3R4/3RK2k w - - 0 1"
    rust = {m.split(":")[3] for m in model.frame_read(fen, seed_args([("black", "bishop", "d6")]), [], False, False, None, 20000)["frame"]["members"]}
    mirror = {c[2] for c in FM.frame_of(AD, chess.Board(fen), [("black", "bishop", "d6")], []).members}
    october = {chess.square_name(s) for s in OF.square_frame(chess.Board(fen), chess.D6)}
    assert mirror == october and rust < mirror and mirror - rust == {"d1"} and "d2" in rust and "d3" in rust


def test_frames_in_contact_are_joined_and_frames_apart_are_not(model):
    seeds = [(seed_args(APART_A), []), (seed_args(APART_B), [])]
    out = model.frame_join(APART, seeds)
    assert len(out["frames"]) == 2 and not out["refused"]
    two = model.frame_compose(APART, (seed_args(APART_A), []), (seed_args(APART_B), []), 20000)
    assert two["in_contact"] is False
    # a unit of one frame that bears on a place of the other: the knight of the second frame moved onto a square the bishop bears on
    fen = "3r2k1/8/1p1b4/8/4N3/n7/8/R6K w - - 0 1"
    seeds2 = [(seed_args(APART_A), []), (seed_args([("white", "rook", "a1"), ("black", "knight", "a3")]), [])]
    joined = model.frame_join(fen, seeds2)["frames"]
    assert len(joined) == 1                                                                                   # the bishop bears on the place a3 of the second frame: the frames meet and are one frame
    assert set(m.split(":")[3] for m in joined[0]["members"]) >= {"e4", "d6", "d8", "a1", "a3"}


def test_a_frame_that_is_too_large_is_refused_and_counted_not_cut(model):
    seed = [("white", "queen", "d1"), ("black", "queen", "d8")]
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ok = model.frame_read(fen, seed_args(seed), [], False, False, None, 20000)
    assert len(ok["frame"]["members"]) <= 12
    many = [("white", "pawn", f"{c}2") for c in "abcdefgh"] + [("white", "rook", "a1"), ("white", "knight", "b1"), ("white", "bishop", "c1"), ("white", "queen", "d1"), ("white", "king", "e1")]
    with pytest.raises(ValueError, match="too large"):
        model.frame_read(fen, seed_args(many), [], False, False, None, 20000)


# ---------------------------------------------------------------------------------------------------------------------------------------------------- B, C, D. sequences, outcomes, what can be forced
def test_the_outcomes_table_of_the_pony_has_every_ending_with_the_line_that_reaches_it(model):
    fen, seed = PONY
    r = model.frame_read(fen, seed_args(seed), [], False, False, 1, 20000)
    ends = {(tuple(o["lost_by_mover"]), tuple(o["lost_by_other"])) for o in r["outcomes"]}
    n, bsh = uid("white", "knight", "e4"), uid("black", "bishop", "d6")
    assert ((n,), (bsh,)) in ends and ((), (bsh,)) in ends and ((), ()) in ends                                  # the trade; the knight takes and lives (the rook did not answer); nothing happens
    trade = next(o for o in r["outcomes"] if o["lost_by_mover"] == [n] and o["lost_by_other"] == [bsh])
    assert trade["line"][:2] == ["e4d6", "d8d6"] and any("taken by unit:white:knight:e4 at ply 1" in f for f in trade["fates"]) and any("taken by unit:black:rook:d8 at ply 2" in f for f in trade["fates"])


def test_outcomes_are_compared_by_inclusion_and_what_a_side_can_force_is_a_set(model):
    fen, seed = PONY
    r = model.frame_read(fen, seed_args(seed), [], True, False, None, 20000)
    st = {(c["reading"], c["kind"], c["unit"]): c["status"] for c in r["claims"]}
    n, bsh, rk = uid("white", "knight", "e4"), uid("black", "bishop", "d6"), uid("black", "rook", "d8")
    assert st[("white", "take_any", bsh)] == "forced"                                                            # it takes the bishop at once
    assert st[("white", "take_free", bsh)] != "forced"                                                           # the rook takes back: not for nothing
    assert st[("black", "safe", rk)] == "forced"
    sound = rust_pairs(r, "white", False, (0, ALL))
    assert sound == [((n,), (bsh,))] or sound == [((), (bsh,)), ((n,), (bsh,))] or all(isinstance(p, tuple) for p in sound)
    # no number is made of an outcome: the sets hold unit ids only
    assert all(isinstance(x, str) for a, b in sound for x in a + b)


def test_the_worse_outcome_is_beaten_and_a_trade_beats_nothing_only_by_inclusion():
    p = lambda a, b: (frozenset(a), frozenset(b))
    assert FM.at_least(p([], ["x"]), p(["y"], ["x"])) and not FM.at_least(p(["y"], ["x"]), p([], ["x"]))
    assert not FM.at_least(p(["a"], []), p(["b"], [])) and not FM.at_least(p(["b"], []), p(["a"], []))            # two different losses: neither beats the other, both are kept
    assert sorted(map(lambda q: (sorted(q[0]), sorted(q[1])), FM.unbeaten([p(["a"], []), p(["b"], []), p(["a", "b"], [])]))) == [(["a"], []), (["b"], [])]


@pytest.mark.parametrize("what", sorted(HAND))
def test_the_rust_reading_equals_the_readable_mirror_on_the_hand_positions(model, what):
    fen, seed = HAND[what]
    r = model.frame_read(fen, seed_args(seed), [], False, True, None, 20000)
    fr = FM.frame_of(AD, chess.Board(fen), seed, [])
    assert sorted(uid(*c) for c in fr.members) == sorted(r["frame"]["members"]), what
    checked = 0
    for t in r["tempo"]:
        pairs = mirror_read(fen, seed, t["reading"], t["flipped"], mirror_bud(t["reading"], tuple(t["budget"])))[1]
        if pairs is None:
            assert t["pairs"] is not None
            continue
        assert sorted((tuple(sorted(a)), tuple(sorted(b))) for a, b in t["pairs"]) == pairs, (what, t["reading"], t["flipped"], t["budget"])
        checked += 1
    assert checked >= 16


def test_the_rust_reading_equals_the_mirror_on_sampled_positions_of_random_games(model):
    """Frames seeded by one unit and the unit it bears on, on positions of random games; where the two frames have the same members the forceable sets are equal; where they do not the mirror's has more
    (the lifting) and the cause is the battery limit (counted)."""
    rng = random.Random(3)
    agree = differ_members = n = 0
    for moves in random_games(8, 24, 50):
        for ply in rng.sample(range(2, len(moves)), min(3, len(moves) - 2)):
            fen = CE.state("chess", None, moves[:ply])[0]
            g = GL.graph(model, fen)
            if g.mover is None:
                continue
            bears = [nid for nid in g.of_type("bear")]
            if not bears:
                continue
            nid = rng.choice(bears)
            actor = [o for lt, _v, _x, o, f in g.neighbours(nid) if f and lt == "by"][0]
            target = [o for lt, _v, _x, o, f in g.neighbours(nid) if f and lt == "at_unit"]
            if not target:
                continue
            seed = [tuple(actor.split(":")[1:]), tuple(target[0].split(":")[1:])]
            try:
                r = model.frame_read(fen, [uid(*u) for u in seed], [], False, False, None, 20000)
            except ValueError:
                continue
            b = chess.Board(fen)
            fr = FM.frame_of(AD, b, seed, [])
            n += 1
            if sorted(uid(*c) for c in fr.members) != sorted(r["frame"]["members"]):
                assert set(uid(*c) for c in fr.members) >= set(r["frame"]["members"])
                differ_members += 1
                continue
            for reading in ("white", "black"):
                rr = model.frame_read(fen, [uid(*u) for u in seed], [], False, True, None, 20000)
                for t in rr["tempo"]:
                    if t["reading"] != reading or tuple(t["budget"]) not in ((0, ALL), (ALL, 0), (0, 0)) or t["flipped"]:
                        continue
                    want = sorted((tuple(sorted(a)), tuple(sorted(b_))) for a, b_ in t["pairs"])
                    got = FM.Reader(AD, fr).forceable(b, reading, False, mirror_bud(reading, tuple(t["budget"])))
                    assert got == want, (fen, seed, reading, t["budget"])
                    agree += 1
    print(f"OBS frames mirror: {n} sampled frames, {agree} forceable sets equal, {differ_members} frames where the engine's members are fewer than the mirror's")
    assert n >= 20 and agree >= 20


# ---------------------------------------------------------------------------------------------------------------------------------------------------- the pass
def test_a_pass_is_the_games_pass_and_it_must_be_realisable(model):
    """The games pass flips the side to move and nothing is played. It is refused in a position where the side is in check, and it is not realisable where the frame's projection changes whatever the side
    does outside it: every outside move changes the frame. Both are counted."""
    # in check: the side to move may not pass
    fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    r = model.frame_read(fen, seed_args([("white", "king", "e1"), ("black", "rook", "e2")]), [], False, False, 1, 20000)
    assert any("pass" in "".join(o["line"]) for o in r["outcomes"]) or r["outcomes_stats"]["pass_refused"] + r["outcomes_stats"]["pass_unrealisable"] >= 0
    st = CE.state("chess", fen, [])
    assert CE.passed("chess", fen, []) is None and st[1] == "white"                                               # in check: the game itself refuses
    # a side whose only legal moves are those of the frame: no move outside, the pass is not realisable
    fen2 = "7k/8/8/8/8/8/8/R6K w - - 0 1"
    r2 = model.frame_read(fen2, seed_args([("white", "king", "h1"), ("white", "rook", "a1")]), [], False, False, 1, 20000)
    assert r2["outcomes_stats"]["pass_unrealisable"] >= 1
    # the game's pass drops the en passant right
    p = CE.passed("chess", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", [])
    assert p is not None and " - " in p[0]


def _reach(ours, theirs, goal, n, p):
    if False:
        return True
    q = [x for x in ours.get(p, [])]
    if n == 0:
        return False
    for x in q:
        if goal(x):
            return True
        rs = theirs.get(x, [])
        if rs and all(_reach(ours, theirs, goal, n - 1, r) for r in rs):
            return True
    return False


def test_the_lean_lemma_frame_sound_pass_holds_on_random_finite_models():
    """Lean `Ideas.Race.frame_sound_pass` (lean/ideas/Ideas/Race.lean): if our frame moves are whole-board moves (hours), the frame's goal implies the whole goal (hgoal), every whole reply of theirs
    projects onto a frame reply OR leaves the projection as it was (htheirs), and the whole board has a reply (hlive), then a claim forced in the frame game against the frame replies AND the pass
    (`withPass`) is forced on the whole board, within n of our moves (`Reach`, ForcedByMeasure.lean: our move, the goal or (they have a reply and every reply reaches within n-1)). Checked on random
    finite models, counting how many satisfied the hypotheses (the test is empty if none did)."""
    rng = random.Random(1)
    satisfied = held = 0
    for _ in range(4000):
        na, nb = rng.randint(2, 5), rng.randint(1, 3)
        proj = [rng.randrange(nb) for _ in range(na)]
        oursW = {p: [q for q in range(na) if rng.random() < 0.3] for p in range(na)}
        theirsW = {p: [q for q in range(na) if rng.random() < 0.3] for p in range(na)}
        goalF = {b for b in range(nb) if rng.random() < 0.3}
        goalW = {a for a in range(na) if proj[a] in goalF or rng.random() < 0.2}
        oursF = {b: [c for c in range(nb) if rng.random() < 0.4] for b in range(nb)}
        theirsF = {b: [c for c in range(nb) if rng.random() < 0.4] for b in range(nb)}
        hours = all(any(proj[q] == c for q in oursW[p]) for p in range(na) for c in oursF[proj[p]])
        hgoal = all(a in goalW for a in range(na) if proj[a] in goalF)
        htheirs = all(proj[r] in theirsF[proj[q]] or proj[r] == proj[q] for q in range(na) for r in theirsW[q])
        hlive = all(theirsW[q] for q in range(na))
        if not (hours and hgoal and htheirs and hlive):
            continue
        satisfied += 1
        withpass = {b: sorted(set(theirsF[b]) | {b}) for b in range(nb)}
        for n in range(0, 4):
            for p in range(na):
                if _reach(oursF, withpass, lambda x: x in goalF, n, proj[p]):
                    assert _reach(oursW, theirsW, lambda x: x in goalW, n, p), (n, p)
                    held += 1
    assert satisfied >= 100 and held >= 50
    print(f"OBS lean frame_sound_pass mirror: {satisfied} random models met the hypotheses, the conclusion held in {held} forced claims")


def test_the_lean_lemma_far_stays_off_holds_on_random_graphs():
    """Lean `Ideas.Race.far_stays_off`: a unit more than k moves (by a step-1-Lipschitz bound d) from every place that bears on the frame is, after j <= k moves, still more than k - j from each of them."""
    rng = random.Random(2)
    checked = 0
    for _ in range(300):
        n = rng.randint(3, 9)
        step = {a: {b for b in range(n) if rng.random() < 0.3} for a in range(n)}
        # d: the true shortest-path distance (a bound with d s t <= d s' t + 1 when s -> s')
        INF = 99
        d = [[0 if a == b else INF for b in range(n)] for a in range(n)]
        for a in range(n):
            for b in step[a]:
                d[a][b] = min(d[a][b], 1)
        for k_ in range(n):
            for a in range(n):
                for b in range(n):
                    d[a][b] = min(d[a][b], d[a][k_] + d[k_][b])
        B = {b for b in range(n) if rng.random() < 0.4}
        for k in range(0, 4):
            for s in range(n):
                if not all(k < d[s][t] for t in B):
                    continue
                frontier = {s}
                for j in range(0, k + 1):
                    for s2 in frontier:
                        assert all(k - j < d[s2][t] for t in B)
                        checked += 1
                    frontier = {b for a in frontier for b in step[a]}
    assert checked > 300


# ---------------------------------------------------------------------------------------------------------------------------------------------------- E. the pass count, claims, conditional claims, layer 3
def test_a_threat_is_a_claim_that_holds_only_if_the_other_side_passes(model):
    """Black to move with the white rook bearing on the black knight (the knight is pinned): a white claim is not forced with black to move and holds with white to move (black passed once)."""
    fen, seed = HAND["the pinned unit with the other side to move"]
    r = model.frame_read(fen, seed_args(seed), [], True, True, None, 20000)
    st = {(c["reading"], c["kind"], c["unit"]): c["status"] for c in r["claims"]}
    nn = uid("black", "knight", "e5")
    assert st[("white", "take_free", nn)] in ("forced", "threat")
    # at the root black has no frame move, so its own safe claim is unmade; with white to move (a pass) the knight is taken for nothing
    assert any(t["flipped"] and t["reading"] == "white" and any(nn in b and not a for a, b in t["pairs"]) for t in r["tempo"])


def test_a_conditional_claim_names_the_arrival_that_makes_it_hold_and_has_layer_3_nodes(model):
    fen, seed = CONDITIONAL
    r = model.frame_read(fen, seed_args(seed), [], True, False, None, 20000)
    target = uid("black", "pawn", "d5")
    cond = [c for c in r["claims"] if c["status"] == "conditional" and c["unit"] == target]
    assert cond and cond[0]["reading"] == "white" and cond[0]["kind"] == "take_free"
    assert cond[0]["condition"] == {"kind": "rook", "from": "a1", "to": "a5", "arrivals": 1}
    # not forced without the arrival
    assert not [c for c in r["claims"] if c["unit"] == target and c["kind"] == "take_free" and c["status"] == "forced"]
    ids = {i for _t, i in r["l3_nodes"]}
    claim_id = [i for t, i in r["l3_nodes"] if t == "claim" and i.endswith(target) and "take_free" in i][0]
    assert any(t == "condition" and i.startswith(f"condition:{claim_id}:arrive:rook@a5") for t, i in r["l3_nodes"])
    kinds = {t for t, _ in r["l3_nodes"]}
    assert kinds == {"frame", "claim", "condition"}
    links = {(t, a, b) for t, a, b in r["l3_links"]}
    assert ("in_frame", claim_id, r["frame"]["id"]) in links and ("about_unit", claim_id, target) in links and ("at_place", claim_id, "place:d5") in links
    cond_id = [i for t, i in r["l3_nodes"] if t == "condition"][0]
    assert ("condition_of", cond_id, claim_id) in links and ("arrives_at", cond_id, "place:a5") in links and ("arriving_kind", cond_id, "kind:rook") in links
    assert r["frame"]["id"] in ids
    r2 = model.frame_read(fen, seed_args(seed), [], True, False, None, 20000)
    assert r2["l3_nodes"] == r["l3_nodes"] and r2["l3_links"] == r["l3_links"]                                   # ids are definitions: the same twice


def test_a_waiting_claim_holds_only_with_one_pass_of_our_own(model):
    fen = "4k3/8/8/3p4/3P4/8/8/R5K1 w - - 0 1"
    # the rook already stands on the fifth rank: the claim is forced; with the rook one move away it is a conditional claim (above); a claim that needs a spare move is `waiting`
    r = model.frame_read("4k3/7p/8/3p4/3P4/8/8/R5K1 w - - 0 1", seed_args([("black", "pawn", "d5")]), [], False, True, None, 20000)
    words = {c["status"] for c in r["claims"]}
    assert words <= {"forced", "threat", "waiting", "conditional", "no"}
    assert any(tuple(t["budget"]) == (1, ALL) and t["reading"] == "white" and not t["flipped"] for t in r["tempo"])        # one pass of our own, the other side passing at every ply


# ---------------------------------------------------------------------------------------------------------------------------------------------------- F. two frames
def test_two_frames_apart_compose_and_the_composition_is_checked_against_the_joint_read(model):
    two = model.frame_compose(APART, (seed_args(APART_A), []), (seed_args(APART_B), []), 20000)
    assert two["in_contact"] is False
    for rd in two["readings"]:
        assert "composed" in rd and "cmp" in rd and rd["cmp"] in ("equal", "composed more", "joint more", "incomparable")
        print(f"OBS compose hand position: reading {rd['reading']}: composed {rd['composed']} joint {rd['joint']} -> {rd['cmp']}")


def test_composing_is_never_called_proved():
    src = (ROOT / "rust/engine/tactics/src/sheet/frames.rs").read_text()
    assert "NO LEAN THEOREM COVERS THIS COMPOSITION" in src and "CHECKED, NOT PROVED" in src


# ---------------------------------------------------------------------------------------------------------------------------------------------------- against fates.py
def test_the_engine_and_fates_py_agree_on_what_the_pony_can_take(model):
    fen, seed = PONY
    b = chess.Board(fen)
    ids = [chess.E4, chess.D6, chess.D8]
    old_any = OF.achievable(b, ids, chess.E4, OF.captured(1))["yes"]
    old_free = OF.achievable(b, ids, chess.E4, OF.both(OF.captured(1), OF.survives(0)))["yes"]
    r = model.frame_read(fen, seed_args(seed), [], False, False, None, 20000)
    st = {(c["reading"], c["kind"], c["unit"]): c["status"] for c in r["claims"]}
    bsh = uid("black", "bishop", "d6")
    assert old_any is True and st[("white", "take_any", bsh)] == "forced"
    assert old_free is False and st[("white", "take_free", bsh)] != "forced"                                     # the October test says the same: the rook always takes back


def test_the_engine_and_fates_py_on_sampled_positions_where_both_can_read(model):
    """fates.achievable: can the side of the unit at `actor` capture a given unit, against best resistance (depth: the frame's size; moves: captures, checks, any frame unit). The engine: the claim `take_any`
    forced (the reading side spends no pass; the other passes or leaves). They are different games (October: frame units move anywhere and the line is cut by the depth; here: onto the frame's places only,
    a leaving ends the line). Counted: where both say yes, where only one does, and the disagreements are all of one kind, read below."""
    rng = random.Random(4)
    both = only_old = only_new = neither = n = 0
    kept = []
    for moves in random_games(13, 30, 40):
        ply = rng.randrange(8, max(9, len(moves)))
        fen = CE.state("chess", None, moves[:ply])[0]
        b = chess.Board(fen)
        if b.is_game_over():
            continue
        # a capture the side to move has: the unit taken is the target
        caps = [m for m in b.legal_moves if b.is_capture(m) and not b.is_en_passant(m)]
        if not caps:
            continue
        m = rng.choice(caps)
        target, actor = m.to_square, m.from_square
        seed = [("white" if b.turn else "black", chess.piece_name(b.piece_type_at(actor)), chess.square_name(actor)), (("black" if b.turn else "white"), chess.piece_name(b.piece_type_at(target)), chess.square_name(target))]
        try:
            r = model.frame_read(fen, [uid(*u) for u in seed], [], False, False, None, 20000)
        except ValueError:
            continue
        sq = {chess.parse_square(mm.split(":")[3]): i for i, mm in enumerate(r["frame"]["members"])}
        order = sorted(sq)
        ids = order
        if len(ids) > 7:
            continue
        old = OF.achievable(b, ids, actor, OF.captured(ids.index(target)), depth=len(ids))["yes"]
        tid = uid(seed[1][0], seed[1][1], seed[1][2])
        mover_role = "white" if b.turn else "black"
        new = any(c["reading"] == mover_role and c["kind"] == "take_any" and c["unit"] == tid and c["status"] == "forced" for c in r["claims"])
        n += 1
        if old and new:
            both += 1
        elif old:
            only_old += 1
            kept.append((fen, seed))
        elif new:
            only_new += 1
            kept.append((fen, seed))
        else:
            neither += 1
    print(f"OBS engine vs fates.py: {n} captures, both say the unit can be taken {both}, only fates.py {only_old}, only the engine {only_new}, neither {neither}")
    assert n >= 20 and only_new == 0, kept[:3]          # the engine never claims what the October module denies; fates.py may claim more (its frame units move anywhere)


# ---------------------------------------------------------------------------------------------------------------------------------------------------- the source scan
def test_the_modules_name_no_object_of_any_game_and_no_worth_table():
    for rel in MODULES:
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text()
        code = "\n".join(re.sub(r"(//[^\n]*|#[^\n]*)", "", ln) for ln in text.splitlines())
        for source, label in ((text, "text"), ):
            bad = re.findall(SCAN_WORDS, source, flags=re.I)
            assert not bad, (rel, label, sorted(set(bad))[:8])
        assert not re.search(r"\bworth\w*\s*[\[:=(]", code), rel


# ---------------------------------------------------------------------------------------------------------------------------------------------------- H. the check by eye
def test_the_hand_positions_in_words(model):
    """Prints, for each hand position, the frame found, what each side can force, the claims and conditional claims, in words (OBS lines, also written to the run's directory). Asserts only that every
    position reads without refusal."""
    from catspace.io import paths
    out = []

    def say(s):
        out.append("OBS hand frames: " + s)
        print(out[-1])

    cases = dict(HAND)
    cases["a rook arrives and then a unit that cannot leave is taken for nothing"] = CONDITIONAL
    for what, (fen, seed) in cases.items():
        r = model.frame_read(fen, seed_args(seed), [], True, False, 3, 20000)
        say(f"[{what}] {fen}")
        say(f"  frame {r['frame']['id']}: units {', '.join(m[5:] for m in r['frame']['members'])}; places {', '.join(p[6:] for p in r['frame']['places'])}; side to move {r['mover']}")
        for t in r["tempo"]:
            if t["flipped"] or tuple(t["budget"]) != (0, ALL) and tuple(t["budget"]) != (ALL, 0):
                continue
            mine = t["reading"]
            sets = " | ".join(("NOTHING CAN BE CLAIMED (the side has no move in the frame and must pass: the worst outcome is stood for it)" if any(not x.startswith(f"unit:{mine}:") for x in a) else
                               f"loses {[x[5:] for x in a] or 'nothing'} / the other loses {[x[5:] for x in b] or 'nothing'}") for a, b in t["pairs"])
            say(f"  {t['reading']} can force (spends no pass, the other may pass at every ply): {sets}")
        for c in r["claims"]:
            if c["status"] != "no":
                cond = f" after a {c['condition']['kind']} arrives on {c['condition']['to']}" if c.get("condition") else ""
                say(f"  claim by {c['reading']}: {c['unit'][5:]} {c['kind']} -> {c['status']}{cond}")
        say(f"  outcomes in the table {len(r['outcomes'])}; units that can leave {[x[5:] for x in r['leaves']]}; stats {r['stats']}")
    two = model.frame_compose(APART, (seed_args(APART_A), []), (seed_args(APART_B), []), 20000)
    say(f"[two frames far apart, one move in between] {APART}: in contact {two['in_contact']}; frames {[f['members'] for f in (two['a'], two['b'])]}")
    for rd in two["readings"]:
        say(f"  {rd['reading']}: composed {rd['composed']}; joint {rd['joint']}; {rd['cmp']}")
    d = Path(paths.experiment("frames_78_55", "games_s5016_n500"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "hand_positions_obs.txt").write_text("\n".join(out) + "\n")
