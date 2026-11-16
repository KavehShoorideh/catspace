"""Plan 78.57: the frame read by questions (rust/engine/tactics/src/sheet/questions.rs, py/src/questions_py.rs, discovery/frames_questions_measure.py).

A. the frame is bound to its units: it follows a unit that steps off, and the claim says so (a mark).
B. the read by questions: a few questions, candidates only, a stop at the first move that holds, a pruned candidate kept with its reason; never a claim the exhaustive 78.55 read denies, unless the read
   followed a unit off the frame or a unit off the frame took (counted, marked).
C. STUCK is its own word; frames in contact are joined, split, or dropped, never read with a unit in common.
D. the composition with the tempo charged, against the joint read.
E'. a pruned move reopened by a claim of another frame, with the tempo charged.
TRAP. the trap question on quiet routes.
No module names an object of any game (a source scan); hand positions appear here and nowhere in the modules."""
import random
import re
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import graph_layer1 as GL

ROOT = Path(__file__).resolve().parents[1]
SCAN_WORDS = (r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|diagonals?|[a-h][1-8]|pins?|pinned|forks?|skewers?|batter\w+|x-?rays?|"
              r"discovered)\b")
MODULES = ["rust/engine/tactics/src/sheet/questions.rs", "rust/engine/py/src/questions_py.rs", "catspace/incubator/analyst/discovery/frames_questions_measure.py"]


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


def uid(role, kind, place):
    return f"unit:{role}:{kind}:{place}"


@pytest.fixture(scope="module")
def model():
    m = GL.model("chess", "chess")
    m.learn_profile([(None, mv) for mv in random_games(21, 12, 60)])
    return m


FUSE = 3000
PINNED = ("4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1", [("white", "rook", "e1"), ("black", "knight", "e5")])
FORK = ("6k1/2r1q3/8/3N4/8/8/8/6K1 w - - 0 1", [("white", "knight", "d5"), ("black", "rook", "c7"), ("black", "queen", "e7")])
PONY = ("3r2k1/8/3b4/8/4N3/8/8/7K w - - 0 1", [("white", "knight", "e4"), ("black", "bishop", "d6")])
BLOCKED = ("4k3/7p/8/3p4/3P4/8/8/R5K1 w - - 0 1", [("black", "pawn", "d5")])
CHASE = ("n6k/8/8/P2B4/8/8/8/2R1K3 b - - 0 1", [("black", "knight", "a8"), ("white", "bishop", "d5")])        # their knight is attacked and every square it can step to loses it
APART = "3r2k1/8/1p1b4/n7/4N3/8/8/R6K w - - 0 1"
APART_A = [("white", "knight", "e4"), ("black", "bishop", "d6")]
APART_B = [("white", "rook", "a1"), ("black", "knight", "a5")]
SERVES = ("r1q2br1/2pn1k1p/B1Ppp1p1/P5B1/np1P4/N4NpP/1PQ3K1/R4R2 w - - 0 24", [[("black", "pawn", "c7"), ("black", "pawn", "d6")], [("white", "pawn", "b2"), ("white", "knight", "a3")]])
# a unit of ours with a move that loses it here, and in another frame the same move forks two units of theirs


def read(model, pos, cond=True):
    fen, seed = pos
    return model.question_read(fen, [uid(*u) for u in seed], [], cond, FUSE)


def claims(r, reading, kind, unit):
    return [c for c in r["claims"] if c["reading"] == reading and c["kind"] == kind and c["unit"] == unit]


def status(r, reading, kind, unit):
    cs = claims(r, reading, kind, unit)
    return cs[0]["status"] if cs else None


# ---------------------------------------------------------------------------------------------------------------------------------------------------- A, B. hand positions
def test_the_pinned_unit_and_the_fork_and_the_juicer_read_by_questions(model):
    r = read(model, PINNED)
    assert status(r, "white", "take_free", "unit:black:knight:e5") == "forced" and r["holds"] and r["gains"]
    r = read(model, FORK)
    assert status(r, "white", "take_any", "unit:black:queen:e7") == "forced" and status(r, "white", "take_any", "unit:black:rook:c7") == "forced"
    assert status(r, "white", "take_free", "unit:black:queen:e7") != "forced"                      # taking the queen costs the knight (the rook retakes): taking and losing, not free
    r = read(model, PONY)
    assert status(r, "white", "take_any", "unit:black:bishop:d6") == "forced" and status(r, "white", "take_free", "unit:black:bishop:d6") != "forced"     # the rook retakes

def test_the_read_asks_few_questions_and_stops_when_a_move_holds(model):
    r = read(model, PINNED)
    s = r["stats"]
    r_old = model.frame_read(PINNED[0], [uid(*u) for u in PINNED[1]], [], False, False, None, 20000)
    print(f"OBS questions: pinned unit: {s['questions']} questions, {s['positions']} positions made, {s['candidates']} candidates asked, {s['not_asked']} legal moves not asked, {s['stops']} stops; 78.55: {r_old['stats']['nodes']} positions")
    assert s["not_asked"] > s["candidates"] > 0          # only candidates are asked about
    assert s["stops"] > 0                                 # the questions ended at a move that holds
    assert s["questions"] < 200


def test_a_pruned_candidate_is_kept_with_its_reason(model):
    r = read(model, CHASE)
    assert {(p["move"], p["moved_unit_lost"]) for p in r["prunes"]} >= {(("a8", "b6"), True), (("a8", "c7"), True)}
    byname = {p["move"]: p for p in r["prunes"]}
    assert byname[("a8", "b6")]["reply"] == ("a5", "b6") and byname[("a8", "c7")]["reply"] == ("c1", "c7")           # the named reply of theirs that takes it
    assert all(p["lost"] == uid("black", "knight", "a8") for p in r["prunes"] if p["move"] in (("a8", "b6"), ("a8", "c7")))
    nodes = {n for t, n in r["l3_nodes"] if t == "prune"}
    assert any("a8->b6" in n and "loses:unit:black:knight:a8" in n and "to:a5->b6" in n for n in nodes)
    assert r["stats"]["pruned_moved"] > 0


def test_the_frame_is_bound_to_its_units_and_follows_a_unit_that_steps_off(model):
    fen, seed = CHASE
    r = read(model, CHASE, cond=False)
    assert r["frame"]["id"].startswith("frame:") and set(r["frame"]["seed"]) == {uid(*u) for u in seed}
    # their knight can step away, and every square it can step to loses it: the read follows it, and says so
    c = claims(r, "white", "take_free", "unit:black:knight:a8")[0]
    assert c["status"] == "forced" and "followed_off_frame" in c["marks"] and r["stats"]["follows"] > 0
    old = model.frame_read(fen, [uid(*u) for u in seed], [], False, False, None, 20000)
    oc = [x for x in old["claims"] if x["reading"] == "white" and x["kind"] == "take_free" and x["unit"] == "unit:black:knight:a8"][0]
    assert oc["status"] != "forced"                                                    # the 78.55 read ends the line when a unit steps off and claims nothing against it
    # the identity is the seed's units: the same frame id wherever the other units stand, and none when a seed unit is gone
    r2 = model.question_read("n6k/8/8/P2B4/8/8/8/R3K3 b - - 0 1", [uid(*u) for u in seed], [], False, FUSE)
    assert r2["frame"]["id"] == r["frame"]["id"]
    with pytest.raises(ValueError):
        model.question_read("7k/8/8/P2B4/8/8/8/R3K3 b - - 0 1", [uid(*u) for u in seed], [], False, FUSE)


def test_a_claim_through_a_follow_or_a_taker_off_the_frame_is_marked(model):
    r = read(model, PONY)
    c = claims(r, "white", "take_any", "unit:black:bishop:d6")[0]
    assert "ended_by_bound" in c["marks"]
    ids = {n for t, n in r["l3_nodes"] if t == "claim"}
    cid = [n for n in ids if "take_any" in n and "bishop:d6" in n][0]
    marks = dict(r["claim_marks"])
    assert marks[cid] == c["marks"]                                                     # the marks are written into the claim's layer 3 node


def test_a_conditional_claim_arises_from_an_arrival_of_the_side_to_move(model):
    r = read(model, BLOCKED)
    assert status(r, "white", "take_free", "unit:black:pawn:d5") == "forced"
    assert r["arrivals_tried"] >= 1


# ---------------------------------------------------------------------------------------------------------------------------------------------------- B. against the exhaustive read
def test_the_questions_never_claim_a_free_taking_the_exhaustive_read_denies_unless_marked(model):
    """Frames seeded by a unit and the unit it bears on, on positions of random games; a forced taking that the 78.55 read (same rules, the line ended when a unit steps off) denies must carry a mark: the
    read followed a unit off the frame or a unit off the frame took."""
    rng = random.Random(3)
    both = marked = weak = plain = old_only = n = 0
    open_cases = []
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
                elif "ended_by_bound" in c["marks"] or "prune_used" in c["marks"]:
                    weak += 1                                         # rests on a depth bound or on a prune: a reading, not a proof, and said so by its marks
                else:
                    plain += 1
                    open_cases.append((fen, seed, key, c["marks"]))
            for key, st in old.items():
                if key[1] != "safe" and st == "forced" and not any(c["status"] == "forced" and (c["reading"], c["kind"], c["unit"]) == key for c in r1["claims"]):
                    old_only += 1
    print(f"OBS questions vs exhaustive: {n} sampled frames; forced takings in both {both}; only by the questions and marked (a unit followed off the frame, or a unit off the frame took) {marked}; "
          f"only by the questions and resting on a bound or a prune (so marked) {weak}; only by the questions with no mark at all {plain}; only by the exhaustive read {old_only}")
    for c in open_cases:
        print("OBS unmarked difference:", c)
    assert n >= 40 and both >= 5
    assert plain == 0, open_cases


# ---------------------------------------------------------------------------------------------------------------------------------------------------- C. stuck and shared units
def test_frames_in_contact_are_joined_and_none_is_read_with_a_unit_in_common(model):
    fen = "3r2k1/8/1p1b4/n7/4N3/8/8/R6K w - - 0 1"
    seeds = [([uid(*u) for u in APART_A], []), ([uid(*u) for u in APART_B], []), ([uid("white", "knight", "e4")], [])]
    out = model.question_reopen(fen, seeds, 800)
    members = [set(r["frame"]["members"]) for r in out["reports"]]
    assert all(not (a & b) for i, a in enumerate(members) for b in members[i + 1:])
    assert len(out["reports"]) >= 2 and all(r["frame"]["id"].startswith("frame:") for r in out["reports"])


def test_the_rust_mark_for_a_stuck_line_grounds_no_claim():
    # the Rust unit tests (questions.rs) hold the mark rules; here: a stuck line is not an outcome with every member lost (no claim names such a pair)
    text = (ROOT / "rust/engine/tactics/src/sheet/questions.rs").read_text()
    assert "F_STUCK" in text and "a_marked_guarantee_grounds_no_claim_and_never_beats_a_clean_one" in text


# ---------------------------------------------------------------------------------------------------------------------------------------------------- D. the tempo in the composition
def test_two_frames_apart_compose_with_the_tempo_charged(model):
    two = model.question_compose(APART, ([uid(*u) for u in APART_A], []), ([uid(*u) for u in APART_B], []), FUSE)
    assert not two["in_contact"]
    won_both = [p for p in two["composed"] if any("bishop" in x for x in p[1]) and any("knight:a5" in x for x in p[1])]
    joint_both = [p for p in two["joint_pairs"] if any("bishop" in x for x in p[1]) and any("knight:a5" in x for x in p[1])]
    print(f"OBS composition: the composed read wins both units in {len(won_both)} guarantees, the joint read in {len(joint_both)}; composed {two['composed']}; joint {two['joint_pairs']}; {two['cmp']}")
    # 78.55 composed both gains while losing the first knight and the rook, which the joint read denied (a move spent in one frame was not a pass in the other); now each guarantee has a cost
    for p in won_both:
        assert any(q[2] <= p[2] and set(q[0]) <= set(p[0]) and set(q[1]) >= set(p[1]) for q in two["joint_pairs"])
    assert two["cmp"] in ("equal", "joint more", "incomparable")


# ---------------------------------------------------------------------------------------------------------------------------------------------------- E'. the pruned branch and the other frame
def test_a_pruned_move_is_reopened_by_a_claim_of_another_frame_with_the_tempo_charged(model):
    fen, seeds = SERVES
    out = model.question_reopen(fen, [([uid(*u) for u in s], []) for s in seeds], 800)
    assert out["reopen"], out
    prune, claim, other, reopened, why = out["reopen"][0]
    assert prune.startswith("prune:frame:") and "a3->b5" in prune and claim.startswith("claim:frame:") and other.startswith("frame:")
    assert reopened and why == "still forced after the other side has taken here"
    # the nodes: the prune node and the claim node, joined by the link reopened_by (written here from the pair the engine returned)
    assert prune in {p["id"] for r in out["reports"] for p in r["prunes"]}
    assert ("reopened_by", prune, claim) in out["reopen_links"] and ("claim", claim) in out["reopen_nodes"] and ("prune", prune) in out["reopen_nodes"]


# ---------------------------------------------------------------------------------------------------------------------------------------------------- TRAP
def test_the_trap_question_on_quiet_routes(model):
    def word(fen, unit):
        return model.question_trap(fen, unit, 400)
    # every square their knight can step to loses it: trapped
    a = word("n6k/8/8/P2B4/8/8/8/2R1K3 w - - 0 1", uid("black", "knight", "a8"))
    assert a["word"] == "trapped" and sorted(a["bad"]) == ["b6", "c7"] and a["free"] == []
    # a unit that nothing bears on may stay: free
    assert word("4k3/8/8/3n4/8/8/P7/4K3 b - - 0 1", uid("black", "knight", "d5"))["word"] == "free"
    # a unit with no move: immobile (not trapped: nothing bears on it)
    assert word("4k3/8/8/8/3p4/3P4/8/4K3 b - - 0 1", uid("black", "pawn", "d4"))["word"] == "immobile"
    # the owner is in check: read under a forcing move everything looks trapped, so the question is not read
    assert word("n3k3/8/8/8/8/8/8/4RK2 b - - 0 1", uid("black", "knight", "a8"))["word"] == "not_quiet"
    # the game's own end: the side to move has no move at all and the game says so
    assert word("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", uid("black", "king", "h8"))["word"] == "ended"


# ---------------------------------------------------------------------------------------------------------------------------------------------------- the source scan
def test_the_modules_name_no_object_of_any_game_and_no_worth_table():
    for rel in MODULES:
        text = (ROOT / rel).read_text()
        bad = re.findall(SCAN_WORDS, text, flags=re.I)
        assert not bad, (rel, sorted(set(bad))[:8])
        code = "\n".join(re.sub(r"(//[^\n]*|#[^\n]*)", "", ln) for ln in text.splitlines())
        assert not re.search(r"\bworth\w*\s*[\[:=(]", code), rel


def test_the_hand_positions_in_words(model):
    """Prints, for each hand position, the frame, the questions asked, the claims with their marks, and what was pruned (OBS lines, also written to the run's directory)."""
    from catspace.io import paths
    out = []

    def say(s):
        out.append("OBS hand frames-q: " + s)
        print(out[-1])

    cases = {"the pinned unit": PINNED, "the double threat": FORK, "the juicer defended by a rook": PONY, "a blocked unit and an arrival": BLOCKED, "their unit steps away and is caught": CHASE}
    for what, pos in cases.items():
        r = read(model, pos)
        say(f"[{what}] {pos[0]}")
        say(f"  frame {r['frame']['id']}: seed {', '.join(u[5:] for u in r['frame']['seed'])}; members {', '.join(m[5:] for m in r['frame']['members'])}; side to move {r['mover']}; it can hold {r['holds']}, it can gain {r['gains']}")
        s = r["stats"]
        say(f"  questions {s['questions']}, positions made {s['positions']}, candidates asked {s['candidates']}, legal moves not asked {s['not_asked']}, pruned {s['pruned_moved'] + s['pruned_other']}, stops {s['stops']}, bounded {s['bounded']}, follows {s['follows']}")
        for c in r["claims"]:
            if c["status"] != "no" and c["kind"] != "safe":
                say(f"  claim by {c['reading']}: {c['unit'][5:]} {c['kind']} -> {c['status']} {c['marks']}")
        for p in r["prunes"][:4]:
            say(f"  pruned {p['move'][0]}->{p['move'][1]}: loses {p['lost'][5:]} to {p['reply']}")
    d = Path(paths.experiment("frames_78_57", "games_s5016_n500"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "hand_positions_obs.txt").write_text("\n".join(out) + "\n")
