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
