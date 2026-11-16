"""Plan 78.60: a line of the frame read ends for a reason (rust/engine/tactics/src/sheet/questions.rs, py/src/questions_py.rs, discovery/frames_questions_measure.py).

A. SETTLED: a line ends settled when neither side has a member that stands to be lost and no member is trapped; with nothing in contact the end rests on nothing (the claim is FORCED WITHIN THE FRAME).
B. PROGRESS: past the first plies a quiet move is read only as a reply to a standing loss; a line that comes back to the bearings it had seen ends as circling, with no claim.
C. THE EXCHANGE GOES ON BELOW only for a named cause (the only guard of a member; a line uncovered), kept with the question.
D. THE BAND: a taking is pruned by the estimate only if it loses at every corner of the stored bands.
E. THE CLASSES: forced / pruned / estimate / bound, by what a claim rests on.
F. SMALL ROOTS: a cap below 10 makes the frame the seed units and what bears on them.
With every rule off the read is exactly the 78.58 read (the counts below are the 78.58 hand counts), and with no kind values the 78.57 read. The comparison against the exhaustive read is run with the exchange on.
No module names an object of any game; hand positions appear here and nowhere in the modules."""
import random
import re
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import graph_layer1 as GL

ROOT = Path(__file__).resolve().parents[1]
SCAN_WORDS = (r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b")
MODULES = ["rust/engine/tactics/src/sheet/questions.rs", "rust/engine/py/src/questions_py.rs", "catspace/incubator/analyst/discovery/frames_questions_measure.py"]
WORTH = [("pawn", 590), ("knight", 1506), ("bishop", 1793), ("rook", 3163), ("queen", 6631), ("king", 13684)]
BANDS = [("pawn", 495, 694), ("knight", 1341, 1710), ("bishop", 1626, 2015), ("rook", 2913, 3468), ("queen", 6251, 7221)]
OFF = (False, False, False, False, 0)
FUSE = 3000
READ_MARKS = {"ended_by_bound", "prune_used", "pruned_by_estimate", "exchange_by_estimate", "circling"}


def uid(role, kind, place):
    return f"unit:{role}:{kind}:{place}"


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
    m.set_kind_values(WORTH)
    m.set_kind_bands(BANDS)
    return m


def read(model, pos, rules=OFF, kv=True, fuse=FUSE, cond=True):
    model.set_kind_values(WORTH if kv else [])
    if kv:
        model.set_kind_bands(BANDS)
    model.set_rules(*rules)
    try:
        return model.question_read(pos[0], pos[1], [], cond, fuse)
    finally:
        model.set_rules(*OFF)
        model.set_kind_values(WORTH)
        model.set_kind_bands(BANDS)


READS = {
    "a queen takes a pawn that a pawn guards": ("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", [uid("white", "queen", "d1"), uid("black", "pawn", "d5")], 149, 58),
    "a pawn takes a knight that a pawn guards": ("4k3/8/4p3/3n4/4P3/8/8/4K3 w - - 0 1", [uid("white", "pawn", "e4"), uid("black", "knight", "d5")], 59, 38),
    "a knight takes a knight that a pawn guards": ("4k3/8/4p3/3n4/8/2N5/8/4K3 w - - 0 1", [uid("white", "knight", "c3"), uid("black", "knight", "d5")], 150, 68),
    "two rooks against a guarded pawn": ("4k3/8/4p3/3p4/8/8/3R4/3RK3 w - - 0 1", [uid("white", "rook", "d2"), uid("black", "pawn", "d5")], 106, 54),
    "a pawn and a knight, two guards": ("4k3/8/2p1p3/3n4/4P3/8/8/3RK3 w - - 0 1", [uid("white", "pawn", "e4"), uid("black", "knight", "d5")], 204, 69),
}
APART = ("4k3/8/4p3/3n4/8/8/1B6/4K3 w - - 0 1", [uid("white", "bishop", "b2"), uid("black", "knight", "d5")])                 # nothing is in contact
GUARD = ("4k3/8/1b2p3/3n4/4P3/8/1R6/4K3 w - - 0 1", [uid("white", "pawn", "e4"), uid("black", "knight", "d5")])               # the knight is the only guard of the bishop
BEHIND = ("3rk3/8/8/8/1p6/3N4/8/3RK3 w - - 0 1", [uid("white", "knight", "d3"), uid("black", "pawn", "b4")])                  # the taker's leaving uncovers a line onto a member
NARROW = ("4k3/8/4p3/3n4/2B5/8/8/4K3 w - - 0 1", [uid("white", "bishop", "c4"), uid("black", "knight", "d5")])                 # a taking that loses by the point estimate but not across the band


def sampled(model, games, per, seedn, game_seed=8):
    """Frames seeded by a unit and the unit it bears on, on positions of random games: (position text, seed)."""
    rng = random.Random(seedn)
    for moves in random_games(game_seed, games, 60):
        for ply in rng.sample(range(2, len(moves)), min(per, len(moves) - 2)):
            fen = CE.state("chess", None, moves[:ply])[0]
            g = GL.graph(model, fen)
            bears = list(g.of_type("bear"))
            if g.mover is None or not bears:
                continue
            nid = rng.choice(bears)
            actor = [o for lt, _v, _x, o, f in g.neighbours(nid) if f and lt == "by"][0]
            target = [o for lt, _v, _x, o, f in g.neighbours(nid) if f and lt == "at_unit"]
            if target:
                yield fen, [uid(*actor.split(":")[1:]), uid(*target[0].split(":")[1:])]


def basis_counts(r):
    out = {}
    for c in r["claims"]:
        if c["status"] != "no":
            out[c["basis"]] = out.get(c["basis"], 0) + 1
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the old reads are kept

def test_with_every_rule_off_the_read_is_the_78_58_read_and_without_kind_values_the_78_57_read(model):
    for name, pos in READS.items():
        a = read(model, pos, OFF, kv=False)["stats"]
        b = read(model, pos, OFF, kv=True)["stats"]
        assert (a["questions"], b["questions"]) == (pos[2], pos[3]), name           # the counts the 78.58 build printed for these positions
        for k in ("end_settled", "end_settled_est", "end_circle", "band_held", "cause_guard", "cause_line", "est_stop", "quiet_skipped", "settle_trap_q"):
            assert a[k] == b[k] == 0, (name, k)
    print("OBS 78.60 rules off: the five hand reads ask the same questions as the 78.58 build printed (without kind values 149 59 150 106 204, with them 58 38 68 54 69)")


# ------------------------------------------------------------------------------------------------------------------------------ A. settled

def test_nothing_in_contact_ends_every_line_settled_and_the_claims_are_forced_within_the_frame(model):
    off, on = read(model, APART, OFF), read(model, APART, (True, False, False, False, 0))
    so, sn = off["stats"], on["stats"]
    print(f"OBS 78.60 settled, nothing in contact: questions {so['questions']} -> {sn['questions']}; lines ended by a bound {so['bounded']} -> {sn['bounded']}, settled {sn['end_settled']}; claims by basis {basis_counts(off)} -> {basis_counts(on)}")
    assert sn["end_settled"] > 0 and sn["bounded"] == 0 and so["bounded"] > 0
    assert set(basis_counts(off)) == {"bound"} and set(basis_counts(on)) == {"forced"}
    for c in on["claims"]:
        if c["status"] != "no":
            assert c["status_read"] == "forced" and not (READ_MARKS & set(c["marks"])), c


def test_a_contact_the_estimate_reads_as_no_loss_ends_settled_by_estimate_and_carries_the_mark(model):
    pos = READS["a pawn and a knight, two guards"]
    r = read(model, pos, (True, False, False, False, 0))
    st = r["stats"]
    print(f"OBS 78.60 settled by estimate: {st['end_settled']} lines settled clean, {st['end_settled_est']} settled with a contact read as no loss; trap questions asked to settle {st['settle_trap_q']}")
    assert st["end_settled_est"] > 0


def test_a_member_that_is_trapped_does_not_settle_the_line(model):
    # over positions of random games: some contacts are trapped members; the line is then read on, never settled
    rng = random.Random(5)
    trapped = settled = 0
    model.set_rules(True, False, False, False, 0)
    try:
        for moves in random_games(8, 20, 60):
            for ply in rng.sample(range(2, len(moves)), min(3, len(moves) - 2)):
                fen = CE.state("chess", None, moves[:ply])[0]
                g = GL.graph(model, fen)
                bears = list(g.of_type("bear"))
                if g.mover is None or not bears:
                    continue
                nid = rng.choice(bears)
                actor = [o for lt, _v, _x, o, f in g.neighbours(nid) if f and lt == "by"][0]
                target = [o for lt, _v, _x, o, f in g.neighbours(nid) if f and lt == "at_unit"]
                if not target:
                    continue
                try:
                    r = model.question_read(fen, [uid(*actor.split(":")[1:]), uid(*target[0].split(":")[1:])], [], False, 1500)
                except ValueError:
                    continue
                trapped += r["stats"]["settle_trapped"]
                settled += r["stats"]["end_settled"] + r["stats"]["end_settled_est"]
    finally:
        model.set_rules(*OFF)
    print(f"OBS 78.60 trap in settling: over sampled frames {settled} lines settled, {trapped} times a member was found trapped and the line was read on")
    assert settled > 0 and trapped > 0


# ------------------------------------------------------------------------------------------------------------------------------ B. progress

def test_a_line_that_comes_back_to_its_bearings_ends_as_circling_with_no_claim_and_quiet_moves_need_a_reason(model):
    r = read(model, APART, (False, True, False, False, 0))
    st = r["stats"]
    print(f"OBS 78.60 circling: lines ended circling {st['end_circle']}; claims made {sum(1 for c in r['claims'] if c['status'] != 'no')}")
    assert st["end_circle"] > 0
    assert all(c["status"] == "no" for c in r["claims"])                            # no claim rests on a circling line
    pos = ("4k3/8/8/8/8/8/3r4/R3K2R w - - 0 1", [uid("white", "rook", "a1"), uid("black", "rook", "d2")], ["place:b1", "place:c1", "place:d1"])    # a frame with empty places: a unit may arrive on them
    model.set_rules(False, True, False, False, 0)
    try:
        st = model.question_read(pos[0], pos[1], pos[2], True, 3000)["stats"]
    finally:
        model.set_rules(*OFF)
    model.set_rules(*OFF)
    base = model.question_read(pos[0], pos[1], pos[2], True, 3000)["stats"]
    print(f"OBS 78.60 progress: a frame with empty places: quiet moves not read for want of a reason {st['quiet_skipped']} (none without the rule: {base['quiet_skipped']}); questions {base['questions']} -> {st['questions']}")
    assert st["quiet_skipped"] > 0 and base["quiet_skipped"] == 0


def test_circling_never_gives_a_claim_through_the_memo(model):
    # the same frame read twice gives the same answer: a circling end is not kept in the memo across lines
    a = read(model, READS["a knight takes a knight that a pawn guards"][:2], (True, True, True, True, 0))
    b = read(model, READS["a knight takes a knight that a pawn guards"][:2], (True, True, True, True, 0))
    assert [(c["kind"], c["unit"], c["status"], c["marks"]) for c in a["claims"]] == [(c["kind"], c["unit"], c["status"], c["marks"]) for c in b["claims"]]
    assert a["stats"]["questions"] == b["stats"]["questions"]


# ------------------------------------------------------------------------------------------------------------------------------ C. the named cause

def test_the_exchange_goes_on_below_only_for_a_named_cause_and_the_cause_is_kept(model):
    g_old, g_new = read(model, GUARD, OFF), read(model, GUARD, (False, False, True, False, 0))
    b_new = read(model, BEHIND, (False, False, True, False, 0))
    print(f"OBS 78.60 cause: only-guard position: exchanges read on below {g_old['stats']['est_cont']} -> {g_new['stats']['est_cont']} (only guard {g_new['stats']['cause_guard']}, line uncovered {g_new['stats']['cause_line']}, stopped for want of a cause {g_new['stats']['est_stop']}); "
          f"line-behind position: line uncovered {b_new['stats']['cause_line']}; causes kept {g_new['causes'][:1]} {b_new['causes'][:1]}")
    assert g_new["stats"]["est_cont"] < g_old["stats"]["est_cont"] and g_new["stats"]["cause_guard"] > 0
    assert any(c[0] == "only_guard" for c in g_new["causes"])
    assert b_new["stats"]["cause_line"] > 0 and any(c[0] == "uncovered_line" for c in b_new["causes"])
    assert g_new["stats"]["est_cont"] == g_new["stats"]["cause_guard"] + g_new["stats"]["cause_line"]
    assert g_new["stats"]["est_stop"] + g_new["stats"]["est_cont"] == g_new["stats"]["est_read"]
    assert g_new["stats"]["questions"] <= g_old["stats"]["questions"]


# ------------------------------------------------------------------------------------------------------------------------------ D. the band

def test_a_taking_is_pruned_by_the_estimate_only_if_it_loses_across_the_band(model):
    flat = read(model, NARROW, OFF)["stats"]
    banded = read(model, NARROW, (False, False, False, True, 0))["stats"]
    print(f"OBS 78.60 band: a taking that loses by the point estimate by a narrow margin: pruned {flat['est_pruned']} without the band, {banded['est_pruned']} with it ({banded['band_held']} held back and read)")
    assert banded["band_held"] > 0 and banded["est_pruned"] < flat["est_pruned"]
    q = READS["a queen takes a pawn that a pawn guards"]
    a, b = read(model, q[:2], OFF)["stats"], read(model, q[:2], (False, False, False, True, 0))["stats"]
    assert a["est_pruned"] == b["est_pruned"] > 0 and b["band_held"] == 0      # a queen for a pawn loses at every corner: still pruned


# ------------------------------------------------------------------------------------------------------------------------------ E. the classes, and against the exhaustive read

def test_the_classes_separate_what_is_forced_from_what_rests_on_an_estimate_and_what_was_cut(model):
    r = read(model, READS["a pawn takes a knight that a pawn guards"][:2], (True, True, True, True, 0))
    got = basis_counts(r)
    print(f"OBS 78.60 classes on the pawn-takes-knight frame: claims by basis {got}")
    for c in r["claims"]:
        if c["status"] == "no":
            continue
        assert c["basis"] in ("forced", "pruned", "estimate", "bound")
        has_bound = bool({"ended_by_bound", "circling"} & set(c["marks"]))
        has_est = bool({"pruned_by_estimate", "exchange_by_estimate"} & set(c["marks"]))
        assert (c["basis"] == "bound") == has_bound
        if c["basis"] == "estimate":
            assert has_est and not has_bound
        if c["basis"] == "forced":
            assert not READ_MARKS & set(c["marks"]) and c["status_read"] == "forced"


def test_a_claim_forced_within_the_frame_is_never_one_the_exhaustive_read_denies_the_exchange_on(model):
    """The frames and rows of 78.57, now with the exchange, the band and every rule of 78.60 on: a forced taking read with no mark at all that the 78.55 read denies would be a failure."""
    model.set_kind_values(WORTH)
    model.set_kind_bands(BANDS)
    model.set_rules(True, True, True, True, 0)
    rng = random.Random(3)
    both = marked = weak = plain = old_only = n = forced_marks_free = 0
    open_cases = []
    try:
        for moves in random_games(8, 40, 60):
            for ply in rng.sample(range(2, len(moves)), min(4, len(moves) - 2)):
                fen = CE.state("chess", None, moves[:ply])[0]
                g = GL.graph(model, fen)
                if g.mover is None:
                    continue
                bears = list(g.of_type("bear"))
                if not bears:
                    continue
                nid = rng.choice(bears)
                actor = [o for lt, _v, _x, o, f in g.neighbours(nid) if f and lt == "by"][0]
                target = [o for lt, _v, _x, o, f in g.neighbours(nid) if f and lt == "at_unit"]
                if not target:
                    continue
                seed = [uid(*actor.split(":")[1:]), uid(*target[0].split(":")[1:])]
                try:
                    r0 = model.frame_read(fen, seed, [], False, False, None, 20000)
                    r1 = model.question_read(fen, seed, [], False, 2000)
                except ValueError:
                    continue
                if len(r0["frame"]["members"]) > 8:
                    continue
                n += 1
                old = {(c["reading"], c["kind"], c["unit"]): c["status"] for c in r0["claims"]}
                for c in r1["claims"]:
                    if c["kind"] == "safe" or c["status"] != "forced":
                        continue
                    key = (c["reading"], c["kind"], c["unit"])
                    if old.get(key) == "forced":
                        both += 1
                    elif "followed_off_frame" in c["marks"] or "taken_by_a_unit_off_frame" in c["marks"]:
                        marked += 1
                    elif c["basis"] != "forced":
                        weak += 1
                    else:
                        plain += 1
                        open_cases.append((fen, seed, key, c["marks"]))
                for key, st in old.items():
                    if key[1] != "safe" and st == "forced" and not any(c["status"] == "forced" and (c["reading"], c["kind"], c["unit"]) == key for c in r1["claims"]):
                        old_only += 1
    finally:
        model.set_rules(*OFF)
    print(f"OBS 78.60 against the exhaustive read, exchange on, every rule on: {n} sampled frames; forced takings in both {both}; only by the questions and marked (followed off the frame, or taken from off it) {marked}; "
          f"only by the questions, resting on a bound, a prune or an estimate {weak}; only by the questions with no mark at all {plain}; only by the exhaustive read {old_only}")
    for c in open_cases:
        print("OBS unmarked difference:", c)
    assert n >= 40 and both >= 5
    assert plain == 0, open_cases


# ------------------------------------------------------------------------------------------------------------------------------ F. small roots

def test_a_cap_below_ten_makes_the_frame_the_seed_units_and_what_bears_on_them(model):
    pos = ("4k3/8/8/8/8/8/3r4/R3K2R w - - 0 1", [uid("white", "rook", "a1"), uid("black", "rook", "d2")], ["place:b1", "place:c1", "place:d1"])
    model.set_kind_values(WORTH)
    model.set_rules(*OFF)
    big = model.question_read(pos[0], pos[1], pos[2], False, 1500)
    model.set_rules(False, False, False, False, 6)
    try:
        small = model.question_read(pos[0], pos[1], pos[2], False, 1500)
    finally:
        model.set_rules(*OFF)
    nb, ns = len(big["frame"]["members"]), len(small["frame"]["members"])
    print(f"OBS 78.60 small roots: one seed of two units and three empty places: {nb} members and {len(big['frame']['places'])} places at the cap of 10, {ns} members and {len(small['frame']['places'])} places at the cap of 6; questions {big['stats']['questions']} -> {small['stats']['questions']}")
    assert ns <= 6 and len(small["frame"]["places"]) == len(pos[1]) < len(big["frame"]["places"])      # no empty place of the seed: the seed units and what bears on them
    model.set_rules(False, False, False, False, 2)
    try:
        with pytest.raises(ValueError):
            model.question_read(pos[0], pos[1], pos[2], False, 1500)                  # more bearers than the cap: refused, never cut quietly
    finally:
        model.set_rules(*OFF)


# ------------------------------------------------------------------------------------------------------------------------------ the source scan and the hand positions in words

def test_the_modules_name_no_object_of_any_game():
    for rel in MODULES:
        text = (ROOT / rel).read_text()
        code = "\n".join(re.sub(r"(//[^\n]*|#[^\n]*)", "", ln) for ln in text.splitlines())
        code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
        bad = re.findall(SCAN_WORDS, code, flags=re.I)
        assert not bad, (rel, sorted(set(bad))[:8])


def test_the_hand_positions_in_words(model):
    from catspace.io import paths
    out = []

    def say(s):
        out.append("OBS 78.60 hand: " + s)
        print(out[-1])

    cases = {"nothing in contact (a bishop far from a guarded knight)": APART, "the knight guards a bishop and a pawn takes it": GUARD, "a rook with a rook behind it takes a rook": BEHIND,
             "a bishop takes a knight a pawn guards, a narrow loss": NARROW}
    cases.update({k: v[:2] for k, v in READS.items()})
    for what, pos in cases.items():
        r0 = read(model, pos, OFF)
        r1 = read(model, pos, (True, True, True, True, 0))
        a, b = r0["stats"], r1["stats"]
        say(f"[{what}] questions {a['questions']} with the exchange only, {b['questions']} with every rule; ends: settled {b['end_settled']}, settled by estimate {b['end_settled_est']}, bound {b['bounded']} (was {a['bounded']}), circling {b['end_circle']}; "
            f"exchanges read on for a cause {b['cause_guard'] + b['cause_line']}, stopped for want of one {b['est_stop']}; band held back {b['band_held']}; claims by basis {basis_counts(r0)} -> {basis_counts(r1)}")
    d = Path(paths.experiment("frames_78_60", "hand"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "hand_positions_obs.txt").write_text("\n".join(out) + "\n")
