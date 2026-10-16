"""ANALYST_PLAN 42: the root (a hanging piece) and the ladder's reading of it. Numbers printed as OBS lines."""
import chess
import pytest

from catspace.incubator.analyst.discovery import selfmine as SM

HANG = "k7/8/8/3n4/8/2N5/8/4K1R1 w - - 0 1"      # the knight on d5 hangs to Nc3; nothing of ours is attacked after the capture
DEFENDED = "4k3/8/4p3/3n4/8/8/8/3QK3 w - - 0 1"  # the knight is defended once, by a pawn; the capturer is a queen


def test_root_reads_a_hanging_knight():
    rd = SM.root_readings(HANG, "w", 3)
    r = next(x for x in rd if x["san"] == "Nxd5")
    print(f"OBS root Nxd5 on the hanging knight: worth {r['worth']}, attacked after {r['attacked_after']}, k {r['k']}, applies {r['applies']}")
    assert (r["worth"], r["attacked_after"], r["k"], r["applies"]) == (3, 0, 3, True)
    import analyst_core as A
    assert A.forced_moves(HANG, 3, 1, SM.BUDGET)                       # the solver confirms the gain in one move and one reply
    assert SM._label((HANG, 3, 1))["status"] == "forced"


def test_root_instance_proves_in_lean():
    rd = SM.root_readings(HANG, "w", 3)
    r = next(x for x in rd if x["applies"])
    out = SM.LEAN_DIR / "Ideas" / "Mined" / "_test_root.lean"
    try:
        head = SM.LAYERS_HEAD.format(tag="t", delta=3, ns="_test_root")
        it = SM.lean_root_instance(r, "w", 3, out, head, True, lambda *a: print(*a))
        print(f"OBS hanging_instance proved {it['proved']}")
        assert it["proved"]
    finally:
        out.unlink(missing_ok=True)


def test_defended_piece_is_not_a_root_claim():
    rd = SM.root_readings(DEFENDED, "w", 3)
    r = next(x for x in rd if x["san"] == "Qxd5")
    lab = SM._label((DEFENDED, 3, 1))
    print(f"OBS defended knight, queen takes: worth {r['worth']}, our pieces attacked after {r['attacked_after']} (most worth {r['w']}), k {r['k']}, applies {r['applies']}; solver {lab['status']}")
    assert r["w"] == 9 and r["k"] == -6 and not r["applies"]
    assert lab["status"] != "forced"                                  # k < delta: no claim, and the solver agrees there is no forced gain


def test_root_proof_reads_only_the_root_atoms():
    atoms = SM.with_legality([SM.atom_of(t, op, v) for t, op, v in SM.root_atoms("c3", "d5", "w", 3)])
    proof = SM.root_proof(atoms, 3, "w")
    assert proof and "Ideas.Root.hanging" in proof
    assert SM.root_proof(atoms[:-1], 3, "w") is None                  # a piece of the root missing: no discharge


def test_material_after_a_capture_matches_python_chess():
    import analyst_core as A
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    BV.install_terms()
    b = chess.Board(HANG)
    theirs0 = 3                                                       # the knight; the king does not count
    b.push_san("Nxd5")
    mat = lambda col: sum({1: 1, 2: 3, 3: 3, 4: 5, 5: 9}.get(p.piece_type, 0) for p in b.piece_map().values() if p.color == col)
    got = A.vocab_eval(HANG, "(after c3 d5 material_them)")
    print(f"OBS black material after Nxd5: python-chess {mat(chess.BLACK)} (was {theirs0}), word {got}")
    assert mat(chess.BLACK) == theirs0 - 3 == 0 and mat(chess.WHITE) == 8
