"""PLAN 73.5b THE GENERATOR, AMENDED (docs/ANALYST_PLAN.md 73.5b): the criterion is the plan's own Lean obligation -- every sampled won position is at the floor
(the engine's n-move mate search) or admits a DROP of the measure by a strategy of at most d moves against every reply, never through a stalemate, a unit the
exchange law loses or a claimable draw. Tested: the lexicographic order of a tuple measure; the AND-OR drop on hand-checkable K+R v K positions (a drop in one
move, and one that needs a tempo move: d = 2); the never-rules refuse a stalemating drop; the graph's min-max equals the per-root search; the miner chooses a
zero-failure measure, the simplest of its level, on a small fixture; the technique question proposes only first moves of a dropping strategy and carries its
target; the agenda asks the store's mined entry (table off: the technique decides; table on: the table decides); the modules name no unit kind."""
import random
import re
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery import technique_measures as TM
from catspace.incubator.analyst.discovery import technique_miner as MN
from catspace.incubator.analyst.discovery import technique_question as TQ
from catspace.incubator.analyst.discovery import technique_sample as TS

REGION = "(count (reach L free))"


def _entry(q: str, d: int, n: int = 2, sig: str = "KRvK") -> dict:
    return dict(name="test technique", kind="TECHNIQUE", measure=dict(quantity=q, floor="", terminal="", depth=d, n=n, **{"class": sig}))


def _region(b: chess.Board) -> int:
    return chess.popcount(TM.region(b, chess.WHITE))


def test_the_lexicographic_tuple_orders_as_its_ranks():
    m = MN.measure_of(f"(lex {REGION} (dist W L) (dist L edge))")
    assert m.op == "lex" and m.ranks == 3 and m.parts == ("region", "kings_dist", "edge_dist")
    rng = random.Random(4)
    for _ in range(300):
        a, b = ({k: rng.randrange(0, MN.RADIX) for k in MN.KEYS} for _ in range(2))
        assert (m.value(a) < m.value(b)) == (m.tuple_of(a) < m.tuple_of(b))
        assert TQ.show(m, m.value(a)) == "(" + ", ".join(str(x) for x in m.tuple_of(a)) + ")"
    assert {len(MN.candidates(r)) for r in (2, 3)} == {8 * 7, 8 * 7 * 6}
    assert len(MN.candidates(1)) == 8 + 28 + 28
    assert all("tempo" not in LS.term_reads(a.sexpr) for a in MN.ATOMS)        # "TEMPO enters as d": no atom of a measure reads the clock
    assert MN.measure_of("the rows beyond the rook's wall") is None


def test_a_drop_in_one_move():
    """Kc8, Rb7 v Kg8: their king walks on e8-h8 (rank 7 is the rook's, d8 our king's): 4 squares. Kd8 takes e8 from it; whichever way it steps (f8, h8) it has
    f8-h8: 3. Every other move leaves it 4 or more."""
    b = chess.Board("2K3k1/1R6/8/8/8/8/8/8 w - - 0 1")
    assert _region(b) == 4
    after = {}
    for m in b.legal_moves:
        b.push(m)
        after[b.peek().uci()] = [(b.push(r), _region(b), b.pop())[1] for r in list(b.legal_moves)]
        b.pop()
    assert after["c8d8"] == [3, 3] and [u for u, v in after.items() if v and max(v) < 4] == ["c8d8"]
    got = TQ.technique_routes(b, _entry(REGION, 1))
    assert got["kind"] == "drop" and got["j"] == 1 and [r[1] for r in got["routes"]] == ["Kd8"]


def test_a_drop_that_needs_a_tempo_move():
    """Kf8, Re7 v Kd8: their king has a8-d8 (4). No move of ours lowers it against every reply (Kf7, Rf7, Rg7, Rh7 keep it 4, the rest open it): d = 1 fails.
    Two moves do: e.g. Rf7 (their only move Kc8), then Ke8 takes d8 and d7: a8-c8, 3."""
    b = chess.Board("3k1K2/4R3/8/8/8/8/8/8 w - - 0 1")
    assert _region(b) == 4
    assert TQ.technique_routes(b, _entry(REGION, 1))["kind"] == "silent"
    got = TQ.technique_routes(b, _entry(REGION, 2))
    assert got["kind"] == "drop" and got["j"] == 2 and "Rf7" in [r[1] for r in got["routes"]]
    b.push_san("Rf7")
    assert [r.uci() for r in b.legal_moves] == ["d8c8"]
    b.push_san("Kc8")
    b.push_san("Ke8")
    assert {(b.push(r), _region(b), b.pop())[1] for r in list(b.legal_moves)} == {3}


def test_the_never_rules_refuse_a_stalemating_drop():
    """Kb6, Rh7 v Ka8: Rb7 leaves their king one square (a8) -- the largest drop there is -- and stalemates: refused, by the per-root search and by the graph."""
    b = chess.Board("k7/7R/1K6/8/8/8/8/8 w - - 0 1")
    rb7 = chess.Move.from_uci("h7b7")
    b.push(rb7)
    assert b.is_stalemate() and _region(b) == 1
    b.pop()
    m = MN.measure_of(REGION)
    S = TQ.Search(m, chess.WHITE, m.value(TM.atom_values(b, chess.WHITE)), frozenset(), 0)
    assert S.first(b, rb7, 1, 0) is None
    g = MN.Graph(chess.WHITE)
    i = g.add(b)
    g.expand(i)
    assert g.refused["stalemate"] == 1
    c = chess.Board("8/8/3k4/8/2R5/8/8/4K3 w - - 0 1")                         # (the rook next to their king, unguarded: the exchange law loses it)
    c.push_san("Rc5")
    assert MN.unit_lost(c, chess.WHITE)


def test_the_floor_move_is_the_first_move_of_a_mate():
    """At the floor the technique plays the mate search's first move: after it, every reply is met by a mate (n = 2: the binder's mate_in_two; n = 3: the net)."""
    fens, _ = TS.uniform("KRvK", 300, 5)
    seen = set()
    for f in fens:
        b = chess.Board(f)
        for n in (2, 3):
            fm = TS.floor_move(b, n)
            if fm is None or fm[1] in seen or fm[1] == "mate_in_one":
                continue
            seen.add(fm[1])
            m = chess.Move.from_uci(fm[0])
            assert m in b.legal_moves
            b.push(m)
            for r in list(b.legal_moves):
                b.push(r)
                assert TS.net(b, n - 1) is not None, (f, fm, r.uci())
                b.pop()
            b.pop()
    assert {"mate_in_two", "net_mate_in_3"} <= seen, seen


def test_the_graph_min_max_is_the_per_root_search():
    fens, drawn = TS.uniform("KRvK", 40, 9)
    assert drawn["won"] == 40 and drawn["valid"] >= 40
    g = MN.Graph(chess.WHITE)
    roots = [g.add(chess.Board(f)) for f in fens]
    n = 0
    for d in (1, 2):
        g.grow(roots, d)
        for q in (REGION, f"(lex {REGION} (dist W L))", "(sum (dist L edge) (dist W L))"):
            m = MN.measure_of(q)
            for f, x in zip(fens, MN.drops(g, m, roots, d)):
                b = chess.Board(f)
                S = TQ.Search(m, chess.WHITE, m.value(TM.atom_values(b, chess.WHITE)), frozenset(), b.halfmove_clock)
                assert x == S.force(b, d, 0), (f, q, d)
                n += 1
    print(f"OBS 73.5b graph against the per-root search: {n} (position, measure, d) equal")


def _fixture() -> list:
    """Won K+Q v K positions with a mate in 2 to 4 (DTZ 3 to 7): a sample the miner settles at d <= 2."""
    rng = random.Random(21)
    rows = []
    while len(rows) < 16:
        b = TC.random_start(rng, "KQvK", 3)
        if TC.tablebase().probe_dtz(b) <= 7:
            f2 = TS.floor_word(b, 2)
            rows.append(dict(fen=b.fen(), source="fixture", floor2=f2, floor3=f2 or TS.floor_word(b, 3)))
    return rows


def test_the_miner_chooses_a_zero_failure_measure():
    rows = _fixture()
    lines = []
    got = MN.mine("KQvK", rows, log=lines.append)
    print("\n".join(x for x in lines if "CHOICE" in x))
    c = got["choice"]
    assert c is not None and c["d"] <= 2
    m = c["measure"]
    fl = {j for j, r in enumerate(rows) if r[f"floor{c['n']}"] is not None}
    for j, r in enumerate(rows):                                                 # zero failures: every position is at the floor or admits the drop
        if j in fl:
            continue
        b = chess.Board(r["fen"])
        S = TQ.Search(m, chess.WHITE, m.value(TM.atom_values(b, chess.WHITE)), frozenset(), 0)
        assert S.force(b, c["d"], 0), r["fen"]
    assert m.sexpr == c["zero"][0] and m.simplicity() == min(MN.measure_of(s).simplicity() for s in c["zero"])
    for lv in got["levels"][:-1]:                                                # every earlier level had a failure in every candidate
        assert all(k > 0 for _s, k in lv["n2"]) and all(k > 0 for _s, k in lv["n3"])
    assert any(x.startswith("OBS miner KQvK CHOICE: ") for x in lines)


def test_the_question_proposes_only_first_moves_of_a_dropping_strategy():
    b = chess.Board("3k1K2/4R3/8/8/8/8/8/8 w - - 0 1")
    m = MN.measure_of(REGION)
    got = TQ.technique_routes(b, _entry(REGION, 2))
    routes = {r[0] for r in got["routes"]}
    for mv in b.legal_moves:                                                     # a route forces the drop within 2; a move that is not one does not
        S = TQ.Search(m, chess.WHITE, got["target"], frozenset(), 0)
        assert (S.first(b, mv, 2, 0) is not None) == (mv.uci() in routes), mv.uci()
    nxt = got["next"]                                                            # the carry: the target and the positions after the first route and every reply
    first = chess.Move.from_uci(got["routes"][0][0])
    b.push(first)
    reply = next(iter(b.legal_moves))
    b.push(reply)
    assert " ".join(b.fen().split()[:4]) in nxt["expect"] and nxt["target"] == 4
    again = TQ.technique_routes(b, _entry(REGION, 2), carried=nxt)
    assert again["carried"] and again["target"] == 4 and again["j"] == 1        # the strategy's second move: the drop below the carried target, now
    c = chess.Board("2K3k1/1R6/8/8/8/8/8/8 w - - 0 1")
    hist = set()
    c.push_san("Kd8")
    c.push_san("Kh8")
    hist.add(" ".join(c.fen().split()[:4]))
    c.pop()
    c.pop()
    blocked = TQ.technique_routes(c, _entry(REGION, 1), history=frozenset(hist))     # a reply into a position the game has seen is a draw the never-rules refuse
    assert blocked["kind"] == "silent" and TQ.technique_routes(c, _entry(REGION, 1))["kind"] == "drop"


def test_the_store_entry_is_asked_by_the_agenda_and_the_table_outranks_it(monkeypatch):
    from catspace.incubator.analyst.discovery import agenda as AG
    from catspace.incubator.analyst.discovery import table as TB
    from catspace.incubator.analyst.discovery.state import State
    ents = TQ.entries_for(chess.Board("4K3/R7/8/8/2k5/8/8/8 w - - 0 1"), chess.WHITE)
    assert [e["name"] for e in ents] == ["mined technique KRvK"] and ents[0]["origin"] == "MINED" and ents[0]["status"] == "CONJECTURE"
    assert TQ.entries_for(chess.Board("4K3/R7/8/8/2k5/8/8/8 w - - 0 1"), chess.BLACK) == []      # the class is read from the side to move's units
    fen = "4K3/R7/8/8/2k5/8/8/8 w - - 0 1"
    monkeypatch.setattr(TB, "TABLE", False)
    ag = AG.Agenda(State(fen=fen, us="w", pieces=[]))
    ag.ask()
    d = ag.decision()
    print(next(ln for ln in ag.lines if ln.startswith("TECHNIQUE ")))
    assert d is not None and d.goal == "technique" and d.lemma == "mined technique KRvK" and not d.proved
    assert any(ln.startswith("DECISION TECHNIQUE: ") for ln in ag.lines) and ag.ctx.uncatalogued == 0
    assert ag.ctx.technique["mined technique KRvK"]["expect"]
    monkeypatch.setattr(TB, "TABLE", True)
    ag = AG.Agenda(State(fen=fen, us="w", pieces=[]))
    ag.ask()
    d = ag.decision()
    assert d is not None and d.proved_by == "table" and d.lemma == "syzygy"
    assert not any(ln.startswith("TECHNIQUE ") for ln in ag.lines)            # the table claimed: a position of the class WITH a claim asks no technique
    from catspace.incubator.analyst.discovery.state import Claim
    ag = AG.Agenda(State(fen=fen, us="w", pieces=[]))
    ag._tech = (ag.all[0], Claim("technique", "Kd8", 0, "mined technique KRvK", "x", move="e8d8", goal="technique", proved=False, word="mined technique KRvK"))
    ag.ask()
    assert ag.decision().proved_by == "table"                                   # were the technique's claim there, the table's proved claim still decides
    monkeypatch.setattr(TB, "TABLE", False)
    monkeypatch.setattr(AG, "TECHNIQUE", False)
    ag = AG.Agenda(State(fen=fen, us="w", pieces=[]))
    assert ag._tech is None and not any(ln.startswith("TECHNIQUE ") for ln in ag.lines)


def test_the_modules_name_no_unit_kind():
    root = Path(MN.__file__).parent
    for f in ("technique_sample.py", "technique_miner.py", "technique_question.py"):
        code = _code(root / f)
        assert not re.search(r"\b(ROOK|QUEEN|BISHOP|KNIGHT|PAWN|Role::)\b", code), f
        assert not re.search(r"['\"](R|Q|KRvK|KQvK)['\"]", code), f


def _code(p: Path) -> str:
    """The module's code without its docstrings and comments."""
    t = p.read_text()
    t = re.sub(r'"""(?:.|\n)*?"""', "", t)
    return "\n".join(ln.split("  #")[0] for ln in t.splitlines())


def test_the_entry_schema_takes_a_class_and_a_depth():
    e = LS.get("mined technique KRvK")
    assert e is not None and LS.check_entry(e) == []
    mm = e["measure"]
    assert set(LS.MEASURE_KEYS) <= set(mm) and mm["class"] == "KRvK" and mm["depth"] in MN.DEPTHS and mm["n"] in MN.NETS
    assert MN.measure_of(mm["quantity"]) is not None and "forced_by_measure" in mm["obligation"]
    assert e["lean"] == [] and e["status"] == "CONJECTURE"
    bad = dict(e, measure=dict(mm, **{"class": "KRK"}))
    assert any("no material signature" in x for x in LS.check_entry(bad))
    assert any("whole number" in x for x in LS.check_entry(dict(e, measure=dict(mm, depth=0))))
    assert LS.get("no measure-dropping move")["kind"] == "STOP"
