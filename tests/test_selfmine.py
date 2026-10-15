"""Phase B of ANALYST_PLAN 41.9: the self-mining loop on a hand-made window. Numbers printed as OBS lines."""
import pytest

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import selfmine as SM

FEN = "4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1"        # white's queen: d5 (a pawn takes it) or d2 (safe)


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
    """The queen on d5 is lost (SAME); with the e6 pawn moved to e7 the queen is safe (AVOIDED); and an atom read after an illegal move is false."""
    base = dict(k=0, cls="SAME", san="Qd5", uci="d1d5", fen="4k3/8/4p3/3Q4/8/8/8/4K3 b - - 1 1", mover="b", start=FEN, line=[])
    out = SM.drill(base, 9, 1, log=lambda *a: print(*a))
    by = {p["san"].split(" | ")[1]: p["cls"] for p in out}
    assert by["remove pe6"] == "AVOIDED" and by["pe6 to e7"] == "AVOIDED" and by["Ke1 to e2"] == "SAME"
    poss = [base] + out
    terms = [SM.legal("e6", "d5"), ("after", "e6", "d5", ("see", "d5"))]
    vals = SM.evaluate(terms, poss)
    _, prem = SM.propose(terms, vals, poss, [0], [i for i, p in enumerate(poss) if p["cls"] == "AVOIDED"], 1, "b")
    assert prem and all(a.pair == ("e6", "d5") for a in prem[0])
    st = SM.statement("t", prem[0], "b", 9, "", "sorry")
    assert "movesOf" in st                              # the legality conjunct is in the Lean statement
    i = next(i for i, p in enumerate(poss) if p["san"].endswith("pe6 to e7"))
    assert not prem[0][0].truth[i]                       # the pawn cannot take from e7: the after-atom is false there
