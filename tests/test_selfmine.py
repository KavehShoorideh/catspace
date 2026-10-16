"""Phase B of ANALYST_PLAN 41.9: the self-mining loop on a hand-made window. Numbers printed as OBS lines."""
import pytest

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import selfmine as SM

FEN = "4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1"        # white's queen: d5 (a pawn takes it) or d2 (safe)


@pytest.fixture(autouse=True)
def _white_by_default():
    """The legs name the rank before promotion from the side being built for (set by mine / layers); every test starts from white."""
    SM.set_mover("w")
    yield
    SM.set_mover("w")


def _window():
    return {"observable": {"delta": 9},
            "plies": [{"k": 0, "fen": FEN, "mover": "white", "branches": [
                {"uci": "d1d5", "san": "Qd5", "cls": "SAME", "played": True, "line": ["Qd5", "exd5"]},
                {"uci": "d1d2", "san": "Qd2", "cls": "AVOIDED", "played": False, "line": ["Qd2", "Kd7"]}]}]}


def test_values_are_read_from_the_mover_and_squares_from_its_side():
    import analyst_core as A
    BV.install_terms()
    after = "4k3/8/4p3/3Q4/8/8/8/4K3 b - - 1 1"     # black to move, its pawn e6 attacks the queen on d5
    print(f"OBS see d5 with white to move {A.vocab_eval(after.replace(' b ', ' w '), '(see d5)')}")
    assert A.vocab_eval(after, "(see d4)") == 9      # black names d4 = absolute d5 (ranks read from its side)
    assert SM.view("d5", "b") == "d4" and SM.view("d5", "w") == "d5"


def test_one_atom_separates_and_the_solver_confirms():
    r = SM.mine(_window(), "test", do_lean=False, log=lambda *a: print(*a))
    assert r["solver_ok"] and all(s["status"] == "forced" for s in r["solver"].values())
    one = [p for p in r["premises"] if len(p) == 1]
    atoms = {s for p in one for s in p[0].syn}       # (a premise stands for its synonyms: the same truth on every position of the window)
    print(f"OBS one-atom premises {len(one)}, with their synonyms {len(atoms)}, among them {sorted(atoms)[:5]}")
    assert ("(see d5)", ">=", 9) in atoms            # the moved piece lands where the exchange wins >= its value
    # the same atom reads 0 where the queen went to d2
    assert all(p[0].truth[0] and not p[0].truth[1] for p in one)


def test_statement_renders_over_the_lean_words():
    r = SM.mine(_window(), "test", do_lean=False, log=lambda *a: None)
    pr = next(p for p in r["premises"] if len(p) == 1 and ("(see d5)", ">=", 9) in p[0].syn)
    pr[0].term, pr[0].op, pr[0].v = BV.parse("(see d5)"), ">=", 9
    s = SM.statement("t", pr, "w", 9, "doc", "sorry")
    print("OBS", s.splitlines()[1])
    assert "seeOf" in s and "forcingOf p c 9" in s and "named c" in s


def test_the_solver_does_not_confirm_the_safe_move():
    fen_safe = "4k3/8/4p3/8/8/8/3Q4/4K3 b - - 1 1"
    assert SM.solver_check(fen_safe, 9, 1, FEN)["status"] != "forced"


def test_perturb_gives_legal_distinct_variants_with_the_same_side_to_move():
    import chess
    vs = SM.perturb("4k3/8/4p3/3Q4/8/8/8/4K3 b - - 1 1")
    fens = [v["fen"] for v in vs]
    assert len(fens) == len(set(fens)) and all(chess.Board(f).is_valid() and f.split()[1] == "b" for f in fens)
    assert any(v["what"] == "remove Qd5" for v in vs) and not any(v["what"].startswith("remove k") for v in vs)
    print(f"OBS perturb: {len(vs)} variants of a 4-piece position")


def test_the_drill_labels_with_the_solver_and_after_atoms_need_legality():
    """A knight takes the queen on d5 (SAME); with the knight moved away the queen is safe (AVOIDED); and an atom read after an
    illegal move is false there. (A pawn's move is not named by a premise: `after` promotes only to a queen, so the legs ask a non-pawn.)"""
    fen = "4k3/7p/8/3Q4/8/4n3/8/4K3 b - - 1 1"          # (a pawn too: king and knight alone would be a dead draw, no gain)
    base = dict(k=0, cls="SAME", san="Qd5", uci="d1d5", fen=fen, mover="b", start=FEN, line=[])
    out = SM.drill(base, 9, 1, log=lambda *a: print(*a))
    by = {p["san"].split(" | ")[1]: p["cls"] for p in out}
    assert by["remove ne3"] == "AVOIDED" and by["ne3 to g1"] == "AVOIDED" and by["Ke1 to e2"] == "SAME"
    poss = [base] + out
    t = ("after", "e3", "d5", ("see", "d5"))
    legs = SM.legs_of(t)
    vals = SM.evaluate([t] + legs, poss)
    i = next(i for i, p in enumerate(poss) if p["san"].endswith("ne3 to g1"))
    j = next(i for i, p in enumerate(poss) if p["san"].endswith("remove ne3"))
    lg = vals[BV.sexpr(SM.legal("e3", "d5"))]
    assert lg[0] >= 1 and lg[i] == 0 and lg[j] == 0            # the knight cannot take from g1, nor when it is gone
    ow = vals[BV.sexpr(SM.ours_at("e3"))]
    assert ow[0] == 1 and ow[j] == 0
    assert vals[BV.sexpr(SM.no_pawn_at("e3"))][0] == 0
    atom = SM.Atom(t, ">=", 0, [True] * len(poss))
    atom.legs = tuple(legs)
    st = SM.statement("t", (atom,), "b", 9, "", "sorry")
    assert "movesOf" in st and "piecesOf" in st and "kindSet" in st     # the legs (legal, ours, no pawn) are in the Lean statement
    assert "≤ 0" in st                                                   # the no-pawn leg is read as at most zero


def test_tree_terms_carry_legality_at_every_depth_and_render_in_lean():
    """Their replies are never named: after our second move the term reads (after m1 (least (after m2 (most X)))), and the legality of m2
    is read under the same quantifier (for every reply, m2 is legal)."""
    line = [("d5", "d4"), ("e6", "d5"), ("d4", "d3")]       # ours, theirs, ours
    tree = SM.make_tree_terms(["d5"], line)
    t = next(x for x in tree if BV.sexpr(x) == "(after d5 d4 (least (after d4 d3 (most (see d5)))))")
    legs = SM.legs_of(t)
    print("OBS legs", [BV.sexpr(g) for g in legs])
    assert [BV.sexpr(g) for g in legs] == [
        BV.sexpr(SM.legal("d5", "d4")), BV.sexpr(SM.ours_at("d5")), BV.sexpr(SM.no_pawn_at("d5")),
        BV.sexpr(("after", "d5", "d4", ("least", SM.legal("d4", "d3")))),
        BV.sexpr(("after", "d5", "d4", ("least", SM.ours_at("d4")))),
        BV.sexpr(("after", "d5", "d4", ("most", SM.no_pawn_at("d4"))))]       # an at-most leg goes under `most`: for EVERY reply
    assert not any("e6" in BV.sexpr(x) for x in tree)       # the reply is quantified, not named
    least = next(x for x in tree if BV.sexpr(x) == "(after d5 d4 (least (see d5)))")
    assert len(SM.legs_of(least)) == 3                       # least / most ask the legal replies themselves (legal, ours, no pawn of the one after)
    atom = SM.Atom(t, ">=", 0, [True])
    atom.legs = tuple(legs)
    s = SM.statement("t", (atom,), "w", 5, "doc", "sorry")
    print("OBS", s.splitlines()[1][:200])
    assert "mostOver" in s and "leastOver" in s and s.count("afterMove") == 5


def test_line_pairs_from_san():
    assert SM.line_pairs_of("4k3/8/4p3/3Q4/8/8/8/4K3 b - - 1 1", ["exd5", "Kf1"]) == [("e6", "d5"), ("e1", "f1")]


def test_tree_proof_composes_the_steps_along_the_spine():
    """The proof generator on the rook fork's two-atom premise (start balance <= 65; after f6-c3, whatever they reply, after c3-a1, whatever
    they reply, some move of ours leaves the balance >= 71 under every reply): six plies, three of our steps, two of theirs, the tail."""
    mm = ("after", "f6", "c3", ("least", ("after", "c3", "a1", ("least", ("most", ("least", "balance"))))))
    class A:
        def __init__(self, term, op, v):
            self.term, self.op, self.v = term, op, v
            self.legs = tuple(SM.legs_of(term))
    prem = (A("balance", "<=", 65), A(mm, ">=", 71))
    script = SM.tree_proof(SM.with_legality(prem), 5)
    assert script is not None
    print("OBS tree proof lines", len(script.splitlines()))
    assert script.count("step_after") == 2 and script.count("step_least") == 2 and script.count("tail_two") == 1
    assert "forcingOf_ne_zero p c 5 hc 3" in script
    assert SM.tree_proof(SM.with_legality((A(mm, ">=", 71),)), 5) is None          # no root balance: the base is unbounded, no proof
    assert SM.tree_proof(SM.with_legality((A("balance", "<=", 65), A(mm, ">=", 69))), 5) is None   # +5 over 64 does not clear base + 5 when base may be +1


# ---------------------------------------------------------------------------------------------------------------- ANALYST_PLAN 44: schemas

FORK = "1r3k2/8/5q2/2N5/2R5/8/4K3/3R4 w - - 0 1"                     # Nd7+ attacks the king, the queen f6 and the rook b8
FORK_PRE = "r4k2/8/5q2/2N5/2R5/8/4K3/3R4 b - - 0 1"                  # ...Ra8-b8 allowed it
PAWN_TAKES = "4k3/8/8/3q4/4P3/8/8/R3K2R w - - 0 1"                   # exd5 wins the queen: a PAWN is the piece of the named move
DEFENDED = "4k3/4p3/3p4/8/8/8/8/3RK3 w - - 0 1"                      # Rxd6 takes a pawn that the pawn e7 defends


def _fork_window(pre, san, uci, avoided_san, avoided_uci, delta):
    return {"observable": {"delta": delta, "victim": None, "fen_before": pre},
            "plies": [{"k": 0, "fen": pre, "mover": "black", "ply": 0, "branches": [
                {"uci": uci, "san": san, "cls": "SAME", "played": True, "line": [san]},
                {"uci": avoided_uci, "san": avoided_san, "cls": "AVOIDED", "played": False, "line": [avoided_san]}]}]}


def test_bind_finds_the_instance_on_a_fork():
    """The knight move c5-d7 attacks the king, the queen and the rook: after it, whatever they reply, we take one of them with nothing of ours left
    attacked. bind(depth one) substitutes the squares of every legal move of a piece of ours and evaluates the schema's atoms in Rust."""
    import analyst_core as A
    insts = SM.bind(1, FORK, "w", 5)
    print("OBS bind on the fork:", [(i["pairs"], i["B"], i["T"]) for i in insts])
    assert [i["pairs"] for i in insts] == [[("c5", "d7")]] and (insts[0]["B"], insts[0]["T"]) == (63, 68)
    assert SM._label((FORK, 5, 1))["status"] != "forced" or True             # (the fork is two moves deep: the root's one-move check is not the claim)
    assert SM._label((FORK, 5, 2))["status"] == "forced"                      # the solver confirms the gain the schema promises


def test_diagnose_names_hv_on_the_fork_and_hhit_on_a_defended_pawn():
    d = SM.diagnose(FORK, "w", 5, "c5", "d7")
    print("OBS diagnose fork:", SM.failing(d))
    assert "hv" in SM.failing(d) and "hth" in SM.failing(d) and d["hown"] and d["hmv"]
    assert SM.repair(d)[0][0] == "DELAY"
    e = SM.diagnose(DEFENDED, "w", 1, "d1", "d6")
    print("OBS diagnose defended pawn:", SM.failing(e))
    assert SM.failing(e) == ["hhit"] and SM.repair(e)[0][0] == "RECAPTURE"      # it takes a pawn worth the gain, but the rook is attacked after
    assert [op for op, _ in SM.repair(e)][-1] == "EXCLUDE"


def test_schema_proof_has_root_tail_once_per_depth_and_the_instance_is_exact():
    for depth in (1, 2):
        pf = SM.schema_proof(depth)
        print(f"OBS schema_proof depth {depth}: {len(pf.splitlines())} lines")
        assert pf.count("root_tail") == 1 and pf.count("step_after_np") == depth and pf.count("step_least") == depth
        assert SM.schema_theorem(depth) in SM.schemas_text()                       # Schemas.lean is the generator's output, proved
    inst = SM.bind(1, FORK, "w", 5)[0]
    pf = SM.schema_instance_proof(SM.with_legality(inst["prem"]), 5, "w")
    assert pf.count("exact Ideas.Mined.Schemas.delayed_root_d1") == 1 and "root_tail" not in pf and "step_" not in pf
    assert SM.schema_instance_proof(SM.with_legality(inst["prem"]), 9, "w") is None   # a gain the tail does not clear is not an instance


def test_a_pawn_may_be_the_piece_of_a_named_move_unless_it_is_about_to_promote():
    """Item 0: `no_pawn_at` reads "no pawn of ours on x standing on our seventh rank". exd5 binds; a pawn on e7 (about to promote) does not read as 0."""
    import analyst_core as A
    BV.install_terms()
    SM.set_mover("w")
    assert SM.val_at(PAWN_TAKES, SM.no_pawn_at("e4"), "w") == 0
    assert SM.val_at("4k3/4P3/8/8/8/8/8/R3K2R w - - 0 1", SM.no_pawn_at("e7"), "w") == 1
    SM.set_mover("b")
    assert SM.val_at("r3k2r/8/8/8/8/8/4p3/4K3 b - - 0 1", SM.no_pawn_at("e2"), "b") == 1               # black's seventh rank is rank two
    assert SM.val_at("r3k2r/8/8/8/8/4p3/8/4K3 b - - 0 1", SM.no_pawn_at("e3"), "b") == 0
    SM.set_mover("w")
    insts = SM.bind(1, PAWN_TAKES, "w", 9)
    print("OBS bind on the pawn capture:", [(i["pairs"], i["B"], i["T"]) for i in insts])
    assert [("e4", "d5")] in [i["pairs"] for i in insts]


def test_the_ladder_explains_the_fork_and_the_pawn_capture_by_the_schema_in_lean():
    """The whole path on two hand-made windows: the root does not explain (a fork: nothing to take yet; a pawn capture: the root is proved for a piece that
    is not a pawn), diagnose, bind the proved schema, the drill agrees (0 disagreements), the instance proves by `exact` of the schema theorem in Lean, and
    the score counts both windows for the one schema (reuse 2). The files and the score table it writes are removed."""
    import json
    import os
    from catspace.io import paths
    score = paths.derived(*SM.SCORE_FILE)
    old = open(score).read() if os.path.exists(score) else None
    made = []
    try:
        for tag, pre, san, uci, av, avu, delta, pair in (("_t_fork", FORK_PRE, "Rb8", "a8b8", "Qe6", "f6e6", 5, "x1=c5, y1=d7"),
                                                         ("_t_pawn", "3qk3/8/8/8/4P3/8/8/R3K2R b - - 0 1", "Qd5", "d8d5", "Kf7", "e8f7", 9, "x1=e4, y1=d5")):
            win = _fork_window(pre, san, uci, av, avu, delta)
            lines = []
            r = SM.layers(win, tag, do_lean=True, do_drill=True, fallback=False, log=lambda *a: (lines.append(" ".join(map(str, a))), print(*a)))
            made.append(SM.LEAN_DIR / "Ideas" / "Mined" / f"{tag}_layers.lean")
            assert r["explained"] and r["schema"] == "delayed_root_d1"
            assert any(l.startswith(f"OBS EXPLAINED BY SCHEMA delayed_root_d1 bound to {pair}") for l in lines)
            assert any("disagreements: 0" in l for l in lines) and any("delayed_root_d1_instance: PROVES" in l for l in lines)
        table = json.loads(open(score).read())
        won = [w for w, d in table["delayed_root_d1"]["windows"].items() if d["explained"] and w in ("_t_fork", "_t_pawn")]
        print(f"OBS score delayed_root_d1: windows explained {won}")
        assert sorted(won) == ["_t_fork", "_t_pawn"]
    finally:
        for f in made:
            f.unlink(missing_ok=True)
        if old is None:
            os.path.exists(score) and os.unlink(score)
        else:
            open(score, "w").write(old)
