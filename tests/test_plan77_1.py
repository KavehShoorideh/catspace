"""PLAN 77.1 (a) THE PORT OF THE ONE DEFINITION'S READING (docs/ANALYST_PLAN.md 77.1 (a) AMENDED): `regress.forcing_by_definition` reads its reply loop by ONE Rust call per
node (`analyst_core.definition_rows_pos`); the claims it reads are the store's GOAL entries a chase pursues, each by its bound reader; the per-move body
`_forcing_by_definition_py` is the oracle. Count budgets only."""
import copy
import re
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import regress as R

POSITIONS = ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "2r2rk1/R5pp/5p2/2q1p3/3pQ3/8/2P2PPP/R5K1 w - - 4 30",
             "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1", "8/8/8/K2pP2r/8/8/8/7k w - d6 0 2",
             "8/5k2/8/1P6/8/8/6K1/8 w - - 0 1", "8/2k5/8/P7/8/8/5n2/6K1 w - - 0 1", "6k1/8/8/8/8/1p6/8/4K2N w - - 0 1")


def _read(b: chess.Board, stake: int, new: bool, rust: bool) -> tuple:
    R._DEF.clear()
    R._BEFORE.clear()
    R.DEF_RUST = rust
    notes = []
    try:
        whys = [R.forcing_by_definition(b, m, stake, b.turn, new=new, notes=notes) for m in b.legal_moves]
    finally:
        R.DEF_RUST = True
    return whys, notes


def test_77_1a_the_rust_rows_equal_the_per_move_body():
    """Every legal move of each position and of its children, stakes 1 3 5 8 9 100, new and held claims: the same answer and the same notes."""
    n = differ = 0
    kinds = set()
    for fen in POSITIONS:
        b = chess.Board(fen)
        nodes = [b]
        for m in b.legal_moves:
            c = b.copy(stack=False)
            c.push(m)
            if not c.is_game_over():
                nodes.append(c)
        for x in nodes:
            for stake in (1, 3, 5, 8, 9, 100):
                for new in (True, False):
                    a, o = _read(x, stake, new, True), _read(x, stake, new, False)
                    n += 1
                    differ += a != o
                    kinds |= {w.split(" ")[0] for w in a[0] if w} | ({"note"} if a[1] else set())
    print(f"OBS 77.1a port equality (test): node readings {n}, differ {differ}; answer kinds seen {sorted(kinds)}")
    assert differ == 0
    assert {"check", "capture", "threat", "threat:", "note"} <= kinds          # (the promotion lemma's text and the survive clause's note are exercised)


def test_77_1a_the_claims_are_the_stores_goals(monkeypatch):
    """The claims read are the store's: the gain goal taken off the chase leaves the queen's attack on the knight no claim."""
    b = chess.Board("4k3/8/2n5/8/8/8/8/3QK3 w - - 0 1")                       # Qd5 / Qa4 attack the knight on c6
    m = chess.Move.from_uci("d1a4")
    assert _read(b, 3, True, True)[0][list(b.legal_moves).index(m)] == "threat 3"
    patched = copy.deepcopy(LS.load())
    patched["win"]["params"]["chase"] = False
    monkeypatch.setattr(LS, "_STORE", [patched])
    G.clear_caches() if hasattr(G, "clear_caches") else None
    rust, py = R._claim_readers()
    assert [r for r, _w in rust] == ["terminal"] and [g.name for g in py] == ["promotion"]
    assert _read(b, 3, True, True)[0][list(b.legal_moves).index(m)] is None
    print(f"OBS 77.1a claims from the store: readers {rust} + {[g.name for g in py]} with the gain goal off the chase")


def test_77_1a_the_rust_section_names_no_unit_kind():
    """The generic test, extended to the new Rust section through its code: no unit kind spelled (the readers are rule level: the kernel's terminal, the exchange law)."""
    src = (Path(R.__file__).parents[4] / "rust" / "analyst_core" / "src" / "regress.rs").read_text()
    sec = src.split("plan 77.1 (a): the one definition's reading", 1)[1].split("// ----------", 1)[0]
    code = "\n".join(ln.split("//")[0] for ln in sec.splitlines())
    hit = re.findall(r"Role::(?!ALL\b)\w+|\b(?:king|queen|rook|bishop|knight|pawn)\b", code)        # (Role::ALL is the kernel's list of kinds: it names none)
    assert hit == [], hit
    assert "fn definition_rows_pos" in code


def test_77_1d_the_definitions_glue_names_no_unit_kind_and_the_lemma_domains_are_the_stores(monkeypatch):
    """plan 77.1 (d) (1): the Python glue of the definition spells no kernel unit kind (chess.KNIGHT, board.rooks ...): the promotion lemmas' domains are their entries' `needs`;
    a patched entry moves where the lemma is asked."""
    import ast, inspect
    names = ("_kinds", "_promotion_lemmas", "_promotion_masks", "_promotion_claims", "_definition_node", "_claims_of", "forcing_by_definition", "_decide", "every_reply")
    kinds = {"PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING", "pawns", "knights", "bishops", "rooks", "queens", "kings"}
    for nm in names:
        tree = ast.parse(inspect.getsource(getattr(R, nm)).lstrip())
        hit = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in kinds]
        assert hit == [], (nm, hit)
    assert not hasattr(R, "_promotion_shut")
    b = chess.Board("8/5k2/8/1P6/8/8/6K1/8 w - - 0 1")
    assert R._promotion_claims(b, 1)                                              # the passer lemma's domain holds (their king alone)
    patched = copy.deepcopy(LS.load())
    for n in ("unstoppablePasser2_sound", "Ideas.Race.race_sound"):
        patched[n]["needs"]["theirs"] = "Q"                                        # a domain without their king: the board (their king alone) is outside it
    monkeypatch.setattr(LS, "_STORE", [patched])
    assert R._promotion_claims(b, 1) == {}                                        # outside every declared domain: no lemma asked
    print(f"OBS 77.1d domains from the store: {[(n, bin(m)) for n, m, _k, _r in R._promotion_lemmas()]}")
