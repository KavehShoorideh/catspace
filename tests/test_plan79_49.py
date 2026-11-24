"""PLAN 79.49 THE THREE LEMMA OBLIGATIONS PUT TO THE PROVER HARNESS (SYSTEM_SPEC 11 "One lemma machinery, used twice"; AGENTS.md rule 11; piece: the long-term memory's statements;
ledger rows 49, 33). THIS file is the gate of the PATH, not of any proof (what the tools found is in the plan's BUILT block):
  (1) the harness takes an obligation of a rule lemma: its goal is read, the definitions its statement rests on are shown from the library modules it imports, a tool whose scratch
      project is not installed is reported and skipped;
  (2) the path of part A on a small obligation whose proof the automation finds in seconds (a toy lemma in memory, never written under rules/): the harness's tools close it, the whole
      generated file passes the harness's acceptance check;
  (3) a proof text Lean rejects, or the acceptance check bans, leaves the document, the obligation and the standing as they were;
  (4) a proof the library already holds is accepted by the whole-file check and its uses are the index's theorems;
  (5) the new module names no game; the tests are in the gate.
No proof is written by hand in this file: the only proof texts are the ones the harness's tools found, the one the lemma's own document already holds, and two texts that must be refused.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"


def mods():
    from catspace.incubator.analyst.discovery import prover as PR
    from catspace.incubator.analyst.discovery import rule_lemma as RL
    from catspace.incubator.analyst.discovery import rule_lemma_certify as RC
    from catspace.incubator.analyst.discovery import rule_lemma_lean as RLL
    from catspace.incubator.analyst.discovery import rule_lemma_prove as RP
    return PR, RL, RC, RLL, RP


def toy(RL):
    """A lemma of the first game made in memory: a relation that copies `candidate`, and the lemma that a candidate row is a row of it."""
    base = RL.load("chess", "candidate_is_mover_unit")
    doc = copy.deepcopy(base.doc())
    doc["name"] = "toy_candidate_copy"
    doc["says"] = "A fixture of the tests: a candidate move is a row of a relation that copies candidate."
    doc["right"] = {"relation": "toy_cand", "places": 5}
    doc["delta"] = ["(signature toy_cand side file rank file rank)", "(<= (toy_cand ?c ?f1 ?r1 ?f2 ?r2) (candidate ?c ?f1 ?r1 ?f2 ?r2))"]
    doc["standing"] = {"is": "CONJECTURE"}
    doc["lean"] = {k: v for k, v in base.lean.items() if k not in ("proof", "uses")}
    return RL.parse(doc)


def test_the_harness_takes_a_rule_obligation(tmp_path):
    PR, RL, RC, RLL, RP = mods()
    lemma = RL.load("chess", "lone_rook_attacks_its_file")
    f = RP.open_copy(lemma, tmp_path / "open.lean")
    goals = PR.goals_of(f)
    assert len(goals) == 1 and goals[0].name == "lone_rook_attacks_its_file_implies"
    assert goals[0].full == RL.generated_theorem(lemma)
    ctx = PR.spec_context(f)
    assert "inductive attackBy" in ctx and len(ctx) < PR.SPEC_BUDGET + 4000        # the relation its statement names, from the library module it imports
    assert RP.open_copy(lemma, tmp_path / "again.lean").read_text() == f.read_text()   # (the open copy is the obligation as it is generated, nothing else)
    if not PR.tools_ready():                                                          # (Duper and LeanCopilot not installed: reported, not run)
        log = PR.Log(tmp_path / "run")
        assert PR.try_tactics(goals[0], "copilot", log, deadline=1e18) is None
        assert "not installed" in log.path.read_text()


def test_the_automation_closes_a_small_obligation_and_the_whole_file_is_accepted(tmp_path):
    PR, RL, RC, RLL, RP = mods()
    lemma = toy(RL)
    f = RP.open_copy(lemma, tmp_path / "toy_open.lean")
    s = PR.prove(str(f), "blind", ("automation",), fuse=300, log_print=lambda *_: None, tag="test79_49")
    g = next(iter(s["goals"].values()))
    assert g["closed_by"] and g["closed_by"].startswith("automation") and g["secs"] < 300
    t = RP.try_proof(lemma, g["proof"])
    assert t["accepted"], t["why"]
    assert "sorry" not in t["text"].replace("-- ", "")                                 # (the text the document would write holds the proof)
    assert RP.candidate(lemma, [], []).standing["theorem"] == RL.generated_theorem(lemma)


def test_a_rejected_proof_leaves_the_standing_as_it_was():
    PR, RL, RC, RLL, RP = mods()
    for name in ("exposed_by_cases", "lone_rook_attacks_its_file"):
        lemma = RL.load("chess", name)
        doc = RL.lemma_dir("chess") / f"{name}.json"
        ob = RL.obligation_path(lemma)
        before = (doc.read_text(), ob.read_text(), RC.standing(lemma))
        assert before[2][0] == "CERTIFIED"
        for text in ("intro h\nexact h", "sorry"):                                     # (Lean rejects the first; the check bans the second)
            r = RP.adopt(lemma, text)
            assert not r["accepted"] and not r["adopted"] and r["why"]
            assert r["standing"][0] == "CERTIFIED"
        assert (doc.read_text(), ob.read_text(), RC.standing(RL.load("chess", name))) == before


def test_a_proof_the_library_holds_is_accepted_whole_and_its_uses_are_the_indexs():
    PR, RL, RC, RLL, RP = mods()
    lemma = RL.load("chess", "candidate_is_mover_unit")
    assert RC.standing(lemma)[0] == "PROVED"
    text = "\n".join(lemma.lean["proof"])
    t = RP.try_proof(lemma, text)
    assert t["accepted"], t["why"]
    assert set(t["uses"]) == set(lemma.lean["uses"])
    assert t["text"] == RLL.path_of(lemma).read_text()                                 # (the document's own text, regenerated: nothing differs)
    # a banned word in a proof is refused by the harness's scan, whatever Lean says
    assert "decide" in PR.scan("intro h\ndecide")


def test_the_module_names_no_game_and_the_gate_has_the_test():
    src = (DISC / "rule_lemma_prove.py").read_text()
    assert not re.search(r"chess|checkers|breakthrough", src, re.I)
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_49" in gate
