"""The idea loop's framework (catspace/incubator/analyst/loop): contracts round-trip, the basis is append-only, the use
rules refuse unproved closures, a deliberately false idea turns the UNSOUND control red and cannot be gated continue,
the driver refuses a round without the previous gate and after a swap gate until the stage is replaced, the ledger
renders every section."""
import json
import shutil

import chess
import pytest

from catspace.incubator.analyst.lean_ideas import default_binary


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("CATSPACE_LOOP_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CATSPACE_LOOP_RECORD", str(tmp_path / "record"))
    return tmp_path


def _candidate(C, idea="x", lean=None, source="registered"):
    return C.Candidate(idea, [dict(name="h", text="h", lean=None)], "fp", "p", C.Claim("mate", 1, 0.0, "mates"), None, lean,
                       source, "s", C.Provenance(0, "translate"))


def test_contracts_round_trip():
    from catspace.incubator.analyst.loop import contracts as C
    p = C.Promoted(_candidate(C, lean=dict(theorem="T", tier="unconditional", bridges=[])), "unconditional",
                   C.Evidence("T", ["propext"], True, dict(checked=3, wrong=0, parts={}), None), False, "ok", C.Provenance(0, "prove"))
    q = C.from_dict(C.Promoted, json.loads(json.dumps(C.to_dict(p))))
    assert q == p and isinstance(q.candidate.claim, C.Claim) and isinstance(q.evidence, C.Evidence)


def test_basis_append_only(roots):
    from catspace.incubator.analyst.loop import basis, contracts as C
    rec = C.to_dict(C.Promoted(_candidate(C, "a"), "conjecture", C.Evidence(), False, "", C.Provenance(1, "prove")))
    basis.promote(1, [rec])
    rec2 = json.loads(json.dumps(rec)); rec2["candidate"]["id"] = "b"
    basis.promote(1, [rec2])
    assert set(basis.active()) == {"a", "b"}
    basis.retire(2, ["a"], "test")
    assert set(basis.active()) == {"b"}
    head = basis.verify()
    p = basis.path()
    lines = p.read_text().splitlines()
    tampered = json.loads(lines[0]); tampered["tier"] = "unconditional"
    p.write_text("\n".join([json.dumps(tampered, sort_keys=True, separators=(",", ":"))] + lines[1:]) + "\n")
    with pytest.raises(basis.BasisError):
        basis.verify()
    with pytest.raises(basis.BasisError):
        basis.promote(3, [rec])
    assert head


def test_use_rules():
    from catspace.incubator.analyst.loop import contracts as C
    from catspace.incubator.analyst.loop.stages.use import closure_ok
    good = C.Promoted(_candidate(C), "unconditional", C.Evidence("T", ["propext"], True, dict(checked=5, wrong=0), None), False, "", C.Provenance(0, "prove"))
    assert closure_ok(good)[0]
    for bad in (C.Promoted(_candidate(C), "conjecture", C.Evidence(held_out={"status": "PROVED"}), False, "", C.Provenance(0, "prove")),
                C.Promoted(_candidate(C), "unconditional", C.Evidence("T", ["sorryAx"], False, dict(checked=5, wrong=0), None), True, "", C.Provenance(0, "prove")),
                C.Promoted(_candidate(C), "unconditional", C.Evidence("T", ["propext"], True, dict(checked=5, wrong=1), None), True, "", C.Provenance(0, "prove")),
                C.Promoted(_candidate(C), "unconditional", C.Evidence("T", ["propext"], True, dict(checked=0, wrong=0), None), False, "", C.Provenance(0, "prove"))):
        assert not closure_ok(bad)[0]


def every_rook_check_mates(li):
    """A false claim for the test: 'a rook check on the back rank mates'. The luft on h7 refutes it."""
    b = chess.Board("6k1/5pp1/7p/8/8/8/8/3R2K1 w - - 0 1")
    b.push_uci("d1d8")
    return dict(checked=1, wrong=0 if b.is_checkmate() else 1, what="the test's false idea", examples=[])


@pytest.mark.skipif(not default_binary().exists(), reason="lean/ideas not built")
def test_false_idea_is_unsound_refused_and_blocks_continue(roots):
    from catspace.incubator.analyst.loop import contracts as C, gate
    from catspace.incubator.analyst.loop.ctx import Ctx
    from catspace.incubator.analyst.loop.ledger import controls, render
    from catspace.incubator.analyst.loop.stages import prove, use
    from catspace.incubator.analyst.loop.store import record_dir, round_dir, write_json
    d, rec = round_dir(7), record_dir(7)
    ctx = Ctx(7, d, rec, dict(prove=dict(extra_bridges={"false": every_rook_check_mates})), "test")
    cand = _candidate(C, "test:everyRookCheckMates", lean=dict(theorem="Ideas.backRankMate_sound", detector="bb", tier="unconditional", bridges=["false"]))
    cand.sentence = "a rook check on the back rank mates"
    write_json(d / "translate.json", dict(hash="x", output=dict(candidates=[C.to_dict(cand)])))
    out = prove.run(ctx)
    p = out["promoted"][0]
    assert p["unsound"] and p["tier"] == "conjecture" and p["evidence"]["bridge"]["wrong"] == 1
    write_json(d / "prove.json", dict(hash="y", output=out))
    import catspace.incubator.analyst.loop.registry as R
    R.REGISTERED.append(dict(id="test:everyRookCheckMates", uses=[("closure", "TacticalSpec.proved_mates=1")]))
    try:
        u = use.run(ctx)
    finally:
        R.REGISTERED.pop()
    closures = [x for x in u["uses"] if x["mode"] == "closure"]
    assert closures and all(x["refused"] for x in closures)
    assert "test:everyRookCheckMates" not in u["active"]["closures"]
    ctl = controls(dict(prove=out, use=u, measure={}), dict(basis_before_ids=[]))
    assert ctl["red"] and ctl["unsound"][0]["id"] == "test:everyRookCheckMates"
    text = render(7, dict(prove=out, use=u, measure={}), dict(basis_before_ids=[], code="test"), ctl, {}, {})
    assert "**RED**" in text
    write_json(rec / "controls.json", ctl)
    with pytest.raises(gate.GateError):
        gate.decide(7, "continue", "should be refused", commit=False)
    g = gate.decide(7, "rollback", "the false idea", commit=False)
    assert g["decision"] == "rollback"


def test_driver_refuses_without_gate_and_after_swap(roots):
    from catspace.incubator.analyst.loop import gate
    from catspace.incubator.analyst.loop.round import RoundError, check_gate, default_config
    from catspace.incubator.analyst.loop.store import record_dir, write_json
    with pytest.raises(RoundError):
        check_gate(1, default_config(1))
    write_json(record_dir(0) / "controls.json", dict(red=False))
    gate.decide(0, "continue", "baseline", commit=False)
    check_gate(1, default_config(1))
    with pytest.raises(RoundError):
        check_gate(2, default_config(2))
    write_json(record_dir(1) / "controls.json", dict(red=False))
    gate.decide(1, "swap", "try another miner", stage="mine", commit=False)
    with pytest.raises(RoundError):
        check_gate(2, default_config(2))
    cfg = default_config(2)
    cfg["stages"] = dict(mine="catspace.incubator.analyst.loop.stages.mine_v2")
    check_gate(2, cfg)


def test_ledger_renders_every_section():
    from catspace.incubator.analyst.loop.ledger import controls, render
    meas = dict(measurements=[
        dict(set="backrank", arm="baseline", puzzles=300, steps=658, tied=300, single=289, mean_nodes=1616.0, wall_s=133.0, gained=[], lost=[], paired=None),
        dict(set="backrank", arm="round", puzzles=300, steps=658, tied=299, single=289, mean_nodes=1271.0, wall_s=136.0, gained=[], lost=["abc"],
             paired=dict(before=1063544, after=836385, ratio=0.786, fewer=632, more=1, equal=25))],
        fire=dict(backrank={"lean:backRankMate": dict(positions=658, instances=600, proved=300, oneaway=50)}))
    prove = dict(promoted=[], not_promoted=[])
    ctl = controls(dict(prove=prove, measure=meas), dict(basis_before_ids=[]))
    assert ctl["results_fall"] and not ctl["red"]
    text = render(1, dict(prove=prove, measure=meas, frame=dict(counts=dict(puzzle=3, game=4)), mine=dict(guards=[])), dict(basis_before_ids=[], code="c"), ctl, dict(frame=1.0), {})
    for section in ("## Inputs", "## Mined", "## Promoted", "## Fired", "## Changed", "## Controls", "## Proposed next", "## Decision"):
        assert section in text
    assert "RESULTS FALL: **FLAG**" in text and "1,063,544 → 836,385" in text


@pytest.mark.skipif(shutil.which("lake") is None and not (__import__("pathlib").Path.home() / ".elan" / "bin" / "lake").exists(), reason="no Lean toolchain")
def test_axioms_script():
    from catspace.incubator.analyst.loop.stages.prove import axioms
    ax, _ = axioms(["Ideas.backRankMate_sound", "Chess.allSquares_nodup"])
    assert set(ax["Ideas.backRankMate_sound"]) <= {"propext", "Classical.choice", "Quot.sound"}
    assert any("native_decide" in a for a in ax["Chess.allSquares_nodup"])


def _puzzle(pid, steps):
    """A dump record: steps as (solution among the tied top moves, how many tied top moves)."""
    st = [dict(fen=f"{pid}-{i}", tied=t, n_top=k, nodes=10) for i, (t, k) in enumerate(steps)]
    return dict(id=pid, steps=st, solved=all(t for t, _ in steps), solved_unique=all(t and k == 1 for t, k in steps))


def test_expected_score_reads_a_tie_break_fairly():
    """A tie-break that is right more often than chance raises the expected score although tied-set membership can only
    fall (the round-1 misreading: 'lost 82 > gained 2' while the expected score rose)."""
    from catspace.incubator.analyst.loop.ledger import controls
    from catspace.incubator.analyst.loop.stages.measure import expected_score, pair, summarize
    assert expected_score(_puzzle("a", [(True, 4), (True, 2)])) == pytest.approx(1 / 8)
    assert expected_score(_puzzle("b", [(True, 1), (False, 3)])) == 0.0
    base = [_puzzle("p1", [(True, 4)]), _puzzle("p2", [(True, 4)]), _puzzle("p3", [(True, 5)])]
    arm = [_puzzle("p1", [(True, 1)]), _puzzle("p2", [(True, 1)]), _puzzle("p3", [(False, 1)])]   # right twice, wrong once
    p = pair(base, arm)
    assert p["lost"] == ["p3"] and not p["gained"]                       # the old count says: fell
    assert p["expected_before"] == pytest.approx(0.25 + 0.25 + 0.2) and p["expected_after"] == pytest.approx(2.0)
    assert p["up"] == ["p1", "p2"] and p["down"] == ["p3"]
    s = summarize(arm)
    m = dict(set="x", arm="round", lost=p["lost"], gained=p["gained"], expected=s["expected"], paired=p, steps=3, wall_s=None,
             vs_previous=dict(expected_before=p["expected_before"], expected_after=p["expected_after"], up=p["up"], down=p["down"]))
    assert not controls(dict(measure=dict(measurements=[m])), dict(basis_before_ids=[]))["results_fall"]
    worse = [_puzzle("p1", [(False, 1)]), _puzzle("p2", [(True, 1)]), _puzzle("p3", [(False, 1)])]   # right once in three still beats chance here
    q = pair(base, worse)
    assert q["expected_after"] == pytest.approx(1.0)
    awful = [_puzzle("p1", [(False, 1)]), _puzzle("p2", [(False, 1)]), _puzzle("p3", [(False, 1)])]
    r = pair(base, awful)
    m2 = dict(m, expected=0.0, lost=r["lost"], paired=r, vs_previous=dict(expected_before=r["expected_before"], expected_after=0.0, up=[], down=r["down"]))
    falls = controls(dict(measure=dict(measurements=[m2])), dict(basis_before_ids=[]))["results_fall"]
    assert falls and "plain" in falls[0]["why"][0] and "previous round" in falls[0]["why"][1]


def test_enrich_pairs_against_the_previous_kept_round(roots, monkeypatch):
    from catspace.incubator.analyst.loop.stages import measure as M
    dumps = {"base.json": [_puzzle("p1", [(True, 4)]), _puzzle("p2", [(True, 2)])],
             "prev.json": [_puzzle("p1", [(True, 2)]), _puzzle("p2", [(True, 2)])],
             "now.json": [_puzzle("p1", [(True, 1)]), _puzzle("p2", [(False, 1)])]}
    monkeypatch.setattr(M, "load_dump", lambda name: dumps[name])
    out = dict(measurements=[dict(set="s", arm="baseline", dump="base.json", lost=[], gained=[], paired=None),
                             dict(set="s", arm="round", dump="now.json", lost=["p2"], gained=[], paired=dict(before=1, after=1))])
    prev = dict(measurements=[dict(set="s", arm="round", dump="prev.json")])
    e, obs = M.enrich(out, prev)
    r = e["measurements"][1]
    assert r["expected"] == pytest.approx(1.0) and e["measurements"][0]["expected"] == pytest.approx(0.75)
    assert r["vs_previous"]["expected_before"] == pytest.approx(1.0) and r["vs_previous"]["up"] == ["p1"] and r["vs_previous"]["down"] == ["p2"]
    assert r["paired"]["before"] == 1 and r["paired"]["expected_before"] == pytest.approx(0.75)   # the old fields are kept
    assert obs and obs[0].startswith("OBS s: expected score")
    assert out["measurements"][1].get("expected") is None                 # enrich does not touch its input
