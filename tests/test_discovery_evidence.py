"""The surprise loop's evidence carried across rounds (surprise_loop.to_test / settle)."""
from catspace.incubator.analyst.discovery.surprise_loop import settle, to_test

FRAME = ["(count (nearer L W))"]


def pend(term, e, base=FRAME):
    return dict(term=term, schema="append", base=base, e=e, w=7, l=1, rounds=1)


def test_the_rounds_best_starts_with_no_evidence():
    assert to_test(dict(term="x", schema="append"), [], FRAME) == [dict(term="x", schema="append", w0=0, l0=0, rounds=0)]


def test_a_pending_candidate_carries_its_evidence():
    t = to_test(dict(term="x", schema="append"), [pend("y", 3.6), pend("z", 1.2)], FRAME)
    assert [(c["term"], c["w0"], c["l0"], c["rounds"]) for c in t] == [("x", 0, 0, 0), ("y", 7, 1, 1), ("z", 7, 1, 1)]


def test_the_same_candidate_is_tested_once_with_its_evidence():
    t = to_test(dict(term="y", schema="append"), [pend("y", 3.6)], FRAME)
    assert [(c["term"], c["w0"], c["l0"]) for c in t] == [("y", 7, 1)]


def test_a_pending_candidate_is_dropped_when_its_frame_changed():
    assert [c["term"] for c in to_test(dict(term="x", schema="append"), [pend("y", 3.6, base=[])], FRAME)] == ["x"]
    assert to_test(None, [pend("y", 3.6, base=[])], FRAME) == []


def test_settle():
    from catspace.incubator.analyst.discovery.defenders import sign_e
    mk = lambda term, w, l: dict(term=term, w=w, l=l, e=sign_e(w, l))
    a, b = mk("a", 20, 1), mk("b", 8, 2)
    assert settle([a, b], 20.0, 30) == (a, [])                                   # promoted at the bar
    lean, weak = mk("c", 19, 12), mk("d", 12, 19)
    assert settle([b, lean, weak], 20.0, 30) == (None, [b, lean])                 # both still able to reach 20 in 30 more games
    assert settle([lean], 20.0, 0) == (None, [])                                  # no games left in the run: dropped
    assert settle([lean], 20.0, 3) == (None, [])                                  # 22 to 12 at best: e 3.5 -- cannot reach the bar


def test_pooled_evidence_grows_with_a_steady_lean():
    from catspace.incubator.analyst.discovery.defenders import sign_e
    assert sign_e(19 * 4, 12 * 4) > sign_e(19, 12) and sign_e(19 * 8, 12 * 8) > 20


def test_futility_by_the_reverse_e_value():
    from catspace.incubator.analyst.discovery.surprise_loop import futile
    assert futile(21, 21, 0, 20.0)                       # no games left: nothing can reach the bar
    assert not futile(9, 5, 22, 20.0)                    # leaning, games left
    assert futile(40, 38, 10, 20.0)                      # a coin over 78 games, 10 left: its rate is below what is needed
    assert not futile(7, 1, 20, 20.0)


def test_complete_lines_leaves_a_partial_line(tmp_path):
    from catspace.incubator.analyst.discovery.proposers import _complete_lines
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n{"c"')
    assert _complete_lines(str(p)) == ['{"a": 1}', '{"b": 2}']
    assert _complete_lines(str(tmp_path / "missing")) == []
