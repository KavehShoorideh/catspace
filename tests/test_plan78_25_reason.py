"""PLAN 78.25 THE REASON OPERATOR (cs_tactics::reason): the reasoning chain -- the threats against the side to move by stake, the frame of the top one, the moves that
parry or counter it, the units that do not matter; with no threat, our own tactics. An order and an explanation, never a verdict: the ledger's value decides.
The positions are chess (the extension names them); the operator reads the Game trait only, so a second variant is run too."""
import json
import re

import cs_engine as CE
from catspace.incubator.analyst.discovery import lemma_store as LS

J = LS.judge_of("chess", "material")
MATE = "r5k1/8/8/8/8/1P6/2P2PPP/6K1 w - - 0 1"          # Black threatens Ra1 mate; the pawns on b3 and c2 are far from it
ROOK = "qr4k1/8/8/8/8/8/5PP1/R5K1 w - - 0 1"            # our rook on a1 is attacked by the queen; Rxa8 wins the queen for the rook
QUIET = "4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1"              # nothing against us; Qxc6 wins the knight
FORK = "6k1/2q2p1p/5b2/8/8/4N3/5PPP/R5K1 w - - 0 1"   # Bf6 attacks our rook on a1; Nd5 attacks their queen and the bishop
IN_CHECK = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
COVER = "4k3/8/8/3p4/4P3/8/5P2/4K3 w - - 0 1"           # their pawn attacks ours on e4; f3 guards it: they can still take, and win nothing
# a middle-game position with no threat against the mover, where our own census (one free move, then a gain) reads 36 positions: 0.3 s with no clock
SLOW = "g2g4 b7b5 a2a4 e7e5 b1a3 c7c5 a4b5 c8b7 g1f3 e5e4 f3g1 e4e3 g1f3 e3d2 e1d2 a7a6 b5a6 b8a6 c2c3 d8f6 h2h3 h7h5 g4h5".split()


def _legal(fen):
    return CE.state("chess", fen, [])[2]


def test_78_25_a_mate_threat_is_the_top_stake_and_only_parries_answer_it():
    r = CE.reason("chess", MATE, [], **J)
    print("OBS 78.25 mate:\n" + "\n".join(r["evidence"]))
    assert not r["must_answer"] and r["threats"][0]["goal"] and r["threats"][0]["line"][0] == "a8a1"
    assert r["threats"][0]["stake"] >= r["threats"][-1]["stake"]
    assert r["answers"] and all(a["kind"] == "parry" for a in r["answers"])
    assert any("cell a 1" in f for f in r["threats"][0]["frame"])          # the frame is facts, not squares
    irrelevant = [f for u in r["irrelevant"] for f in u["facts"]]
    assert any("b 3 white pawn" in f for f in irrelevant) and any("c 2 white pawn" in f for f in irrelevant)
    far = {m for u in r["irrelevant"] for m in u["moves"]}
    assert "b3b4" in far and far.isdisjoint(set(r["candidate_moves"]))
    # every candidate really avoids the mate: after it, no reply of Black's mates (the engine reads it: the ledger value is not a loss of the game)
    for mv in r["candidate_moves"]:
        after = CE.state("chess", MATE, [mv])
        assert after[3] is None                                              # the game goes on
        for reply in after[2]:
            assert CE.state("chess", MATE, [mv, reply])[3] != [0, 100]       # no reply ends the game in Black's favour
    assert "reason: candidates" in r["evidence"][-1]
    assert [e for e in r["evidence"] if e.startswith("reason: if we pass they win the game")]


def test_78_25_a_rook_under_attack_is_parried_by_moving_it_or_by_taking_the_attacker():
    r = CE.reason("chess", ROOK, [], **J)
    print("OBS 78.25 rook:\n" + "\n".join(r["evidence"]))
    top = r["threats"][0]
    assert not top["goal"] and top["stake"] > 0 and "r" in top["targets"][0]
    assert top["line"][-1] == "b8b1" or top["line"] == ["a8a1"]                                # the line ends at its last ledger event (no quiet tail)
    by = {a["move"]: a for a in r["answers"]}
    assert by["a1a8"]["kind"] == "parry" and by["a1a8"]["net"] > 0                             # taking the attacker: the threat's move is gone, and we gain
    assert by["a1c1"]["kind"] == "parry" and by["a1c1"]["net"] > -top["stake"]                 # moving the rook away
    assert by["a1a8"]["if_executed"] is None
    assert r["candidate_moves"][0] == "a1a8"                                                   # the higher net ranks first
    assert r["leaving"] > 0 and "f2f3" not in by                                               # a move that ignores the threat leaves it


def test_78_25_a_greater_threat_counters_a_threat_that_still_stands():
    """The owner's case: our rook is attacked; a move that leaves it there but attacks something bigger answers the threat, because carrying the threat out now
    costs them more than it wins. Bf6 hits Ra1; Nd5 forks the queen and the bishop: if Bxa1 then Nxc7."""
    r = CE.reason("chess", FORK, [], **J)
    print("OBS 78.25 fork:\n" + "\n".join(r["evidence"]))
    top = r["threats"][0]
    assert top["line"] == ["f6a1"] and top["stake"] == 5
    assert not any("f 2" in f or "f 3" in f for f in top["frame"])                             # the frame holds the threat's own facts only
    by = {a["move"]: a for a in r["answers"]}
    c = by["e3d5"]
    assert c["kind"] == "counter" and c["net"] > -top["stake"]
    assert c["executed_line"][:2] == ["f6a1", "d5c7"] and c["if_executed"] == 4                # they take the rook, we take the queen: 9 - 5
    assert by["a1b1"]["kind"] == "parry"
    assert by["e3c2"]["kind"] == "lessen" and by["e3c2"]["if_executed"] == -2                  # guarding the rook: they still take it, we take back
    assert "g1h1" not in by and "h2h3" not in by                                               # moves that ignore the threat leave it
    irrelevant = [f for u in r["irrelevant"] for f in u["facts"]]
    assert any("h 2 white pawn" in f for f in irrelevant) and any("g 1 white king" in f for f in irrelevant)
    text = "\n".join(r["evidence"])
    assert "reason: counters (the threat stands and is outbid): e3d5" in text


def test_78_25_a_guard_covers_a_threat_that_still_stands_and_wins_nothing():
    r = CE.reason("chess", COVER, [], **J)
    print("OBS 78.25 cover:\n" + "\n".join(r["evidence"]))
    by = {a["move"]: a for a in r["answers"]}
    assert by["f2f3"]["kind"] == "cover" and by["f2f3"]["if_executed"] == 0 and by["f2f3"]["executed_line"][:2] == ["d5e4", "f3e4"]
    assert by["e4d5"]["kind"] == "parry" and by["e4e5"]["kind"] == "parry"
    assert r["candidate_moves"][0] == "e4d5"                                                   # the parry that also gains ranks first
    assert "reason: covers (the threat stands and wins nothing): f2f3" in "\n".join(r["evidence"])
    assert not any(a["kind"] == "counter" for a in r["answers"])                               # a guard is no greater threat


def test_78_25_the_chain_keeps_to_its_share_of_the_clock_and_a_cut_is_a_gap(tmp_path):
    """Integration finding (2026-11-10): our own census ran under its node fuse alone and took 0.42 s of a 0.2 s decision. One deadline now holds for the whole chain."""
    log = tmp_path / "trace.jsonl"
    rows = CE.solve_batch([("chess", None, SLOW)], 0.2, reason=True, trace_path=str(log), threads=1, **J)
    assert rows[0][4] is None
    row = json.loads(log.read_text().splitlines()[0])
    step = [x for x in row["steps"] if x["operator"] == "reason"][0]
    print(f"OBS 78.25 clock: reason step {step['seconds']:.4f} s of a {row['seconds']:.3f} s decision, status {step['status']}, parts {[(p['name'], p['nodes']) for p in step['parts']]}")
    assert step["seconds"] < 0.2 * 0.1 + 0.03 and row["seconds"] < 0.26
    assert step["status"] == "Gap" and any(t.startswith("reason: GAP") for t in row["trace"])
    assert [p["name"] for p in step["parts"]] == ["threats", "answers", "ours"]


def test_78_25_with_no_threat_our_own_tactics_are_the_candidates():
    r = CE.reason("chess", QUIET, [], **J)
    print("OBS 78.25 quiet:\n" + "\n".join(r["evidence"]))
    assert not r["must_answer"] and r["threats"] == [] and r["answers"] == []
    assert r["evidence"][0].startswith("reason: no threat against us")
    assert [t for t in r["ours"] if t["defects"] == 0] and r["candidate_moves"][0] == "a4c6"
    # the threats we can create (one defect) follow the ones that work now
    later = CE.reason("chess", "4k3/8/2n5/8/8/8/8/3QK3 w - - 0 1", [], **J)
    assert later["ours"] and all(t["defects"] >= 0 for t in later["ours"])


def test_78_25_a_side_in_check_must_answer_and_every_move_is_read():
    r = CE.reason("chess", IN_CHECK, [], **J)
    print("OBS 78.25 check:\n" + "\n".join(r["evidence"]))
    assert r["must_answer"] and r["threats"] == []
    assert sorted(a["move"] for a in r["answers"]) == sorted(_legal(IN_CHECK)) and {a["kind"] for a in r["answers"]} == {"reply"}
    assert r["candidate_moves"][0] == "e1e2" and r["unread"] == []                              # Kxe2 wins the rook: ranked first by net


def test_78_25_another_variant_gives_a_report_and_a_cut_reading_is_a_gap_not_a_refutation():
    for v in ("kingofthehill", "antichess", "atomic", "3check"):
        r = CE.reason(v)
        assert r["evidence"] and r["readings"] >= 1 and isinstance(r["candidates"], list)
        print(f"OBS 78.25 {v}: {r['evidence'][0]} -- candidates {r['candidate_moves']}")
    cut = CE.reason("chess", ROOK, [], 2, 100, **J)                                             # a starved fuse
    print("OBS 78.25 starved: " + " | ".join(cut["evidence"][-2:]))
    assert cut["cut"] > 0 and cut["unread"] and any(e.startswith("reason: GAP") for e in cut["evidence"])
    unread = set(cut["unread"])
    assert unread.isdisjoint({u for x in cut["irrelevant"] for u in x["moves"]})               # unread is never reported irrelevant


def _norm(trace):
    return [re.sub(r"\(\d+ nodes\)", "(N nodes)", x) for x in trace]


def test_78_25_the_decision_is_unchanged_with_reason_off_and_carries_the_lines_with_it_on():
    legal = _legal(MATE)
    mv0, _p0, t0, _n0, _f0 = CE.best_move("chess", MATE, [], 0.5, **J)
    mv1, _p1, t1, _n1, _f1 = CE.best_move("chess", MATE, [], 0.5, reason=False, **J)
    assert _norm(t0)[0] == _norm(t1)[0] and not any(x.startswith("reason:") for x in t0 + t1)
    assert mv0 == mv1                                                                           # (the clock may cut the ledger at another depth: the move is the same here)
    mv2, _p2, t2, _n2, _f2 = CE.best_move("chess", MATE, [], 0.5, reason=True, **J)
    print("OBS 78.25 decide with reason:\n" + "\n".join(t2))
    assert any(x.startswith("reason: if we pass they win") for x in t2) and any("ledger search" in x for x in t2)
    assert mv2 in legal and mv2 not in ("b3b4",)
    row = CE.solve_batch([("chess", MATE, []), ("chess", None, [])], 0.3, reason=True, **{k: v for k, v in J.items()})
    assert all(r[4] is None and r[0] is not None for r in row)
    off = CE.solve_batch([("chess", MATE, [])], 0.3)
    assert off[0][4] is None


def test_78_25_the_layer_crate_names_no_chess_object():
    from pathlib import Path
    chess_words = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"     # (tests/test_plan78.py's list)
    p = Path(__file__).resolve().parents[1] / "rust" / "engine" / "tactics" / "src" / "reason.rs"
    assert re.search(chess_words, p.read_text(), flags=re.I) is None                              # not even in its comments
