"""PLAN 78 THE END-TO-END SYSTEM (docs/ANALYST_PLAN.md 78). 78.1 THE GAME LAYER: the `Game` trait (rust/engine/game), any GDL text as a game (the spec evaluator,
cs_gdl), the chess kernels for the perfect-information variants (shakmaty), certify by co-play; the variants' rules as DELTAS on rules/chess.gdl (cs_gdl::compose).
The layer crates name no chess object: only the kernel module and the GDL texts are chess (rules ingestion)."""
import re
from pathlib import Path

import pytest

import cs_engine as CE

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
BASE = (ROOT / "rules" / "chess.gdl").read_text()
DELTA = {v: (ROOT / "rules" / "variants" / f"{v}.gdl").read_text() for v in ("kingofthehill", "3check")}
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


def _code(p: Path) -> str:
    """The Rust code without its comments (//, //!, ///) and string literals."""
    t = re.sub(r'"(?:\\.|[^"\\])*"', '""', p.read_text())
    return "\n".join(ln.split("//")[0] for ln in t.splitlines())


def test_78_1_the_layer_crates_name_no_chess_object_but_the_kernel_module():
    """The generic test, extended to the Rust layers: every source of the engine workspace but the kernel module (rules ingestion) and its certify binary is free
    of chess words in its code (comments and strings aside)."""
    allowed = {ENGINE / "game" / "src" / "kernel.rs", ENGINE / "game" / "src" / "bin" / "certify_coplay.rs", ENGINE / "py" / "src" / "lib.rs"}
    seen = []
    for p in sorted(ENGINE.glob("*/src/**/*.rs")):
        if p in allowed:
            continue
        hit = re.search(CHESS_WORDS, _code(p), flags=re.I)
        assert hit is None, (str(p.relative_to(ROOT)), hit.group(0) if hit else None)
        seen.append(p.name)
    print(f"OBS 78.1 generic test: {len(seen)} layer sources free of chess words: {', '.join(seen)}")
    assert {"lib.rs", "gdl_game.rs", "certify.rs", "rng.rs"} <= set(seen)


def test_78_1_the_kernels_agree_with_their_gdl_texts_by_co_play_and_a_wrong_text_disagrees():
    for v in ("chess", "kingofthehill", "3check"):
        games, plies, states, bad = CE.certify_coplay(v, BASE if v == "chess" else DELTA[v], 3, 5, None if v == "chess" else BASE)
        print(f"OBS 78.1 co-play {v}: games {games}, plies {plies}, states {states}, disagreements {len(bad)}")
        assert games == 3 and plies > 0 and bad == []
    assert CE.certify_coplay("kingofthehill", BASE, 3, 5)[3]                         # the hill's kernel against plain chess: it disagrees (the control)
    assert CE.certify_coplay("3check", DELTA["kingofthehill"], 3, 5, BASE)[3]


def test_78_1_a_delta_must_retract_what_the_base_says():
    with pytest.raises(ValueError):
        CE.certify_coplay("kingofthehill", "(retract (<= terminal nothing_of_the_base))", 1, 1, BASE)


def test_78_1_every_variant_is_played_and_those_without_a_text_are_reported_uncertified():
    from catspace.incubator.analyst.discovery import certify as CF
    assert set(CE.variants()) == {"chess", "chess960", "crazyhouse", "atomic", "antichess", "kingofthehill", "3check", "racingkings", "horde"}
    un = CF.uncertified()
    print(f"OBS 78.1 uncertified variants (no GDL text yet): {un}")
    assert set(un) == set(CE.variants()) - set(CF.VARIANT_TEXTS)
    for v in CF.VARIANT_TEXTS:
        assert CF.status()[f"kernel {v} (cs_game)"] == "fresh"


# ---- 78.2 the thin approaches (tactics `forcing`, don't lose, planning `keep`) and the decision loop, one call per move

def _worth():
    from catspace.incubator.analyst.discovery import lemma_store as LS
    return {k: int(v) for k, v in LS.valuation().worth.items()}


def test_78_2_the_decision_proves_each_variants_own_goal():
    """The tactics layer reads only the Game trait's outcome: the same search proves a chess mate, a king walking onto the hill, a third check."""
    w = _worth()
    for v, fen, want in (("chess", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"), ("kingofthehill", "8/8/8/8/8/2K5/8/7k w - - 0 1", "c3d4"),
                         ("3check", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1+3 0 3", None)):
        mv, proved, trace, nodes, _f = CE.best_move(v, fen, [], 1.0, w)
        print(f"OBS 78.2 {v}: {mv} proved {proved}, {nodes} nodes -- {trace[1][:120]}")
        assert proved and (want is None or mv == want)


def test_78_2_every_variant_gets_a_legal_move_and_dont_lose_drops_a_losing_one():
    w = _worth()
    for v in CE.variants():
        mv, proved, trace, nodes, fen = CE.best_move(v, None, [], 0.2, w)
        assert mv in CE.state(v)[2], (v, mv)
    # chess, the fool's mate both ways: after 1.f3 e5 2.g4 Black's Qh4# is proved; after 1.f3 e5 White's 2.g4?? (it allows Qh4#) is dropped by don't-lose
    mv, proved, trace, nodes, fen = CE.best_move("chess", None, ["f2f3", "e7e5", "g2g4"], 0.5, w)
    assert mv == "d8h4" and proved                                                    # (Black to move: the fool's mate, proved)
    mv, proved, trace, nodes, fen = CE.best_move("chess", None, ["f2f3", "e7e5"], 0.5, w)
    assert mv != "g2g4" and any("don't lose" in x and "g2g4" in x for x in trace)     # (White: g4?? allows Qh4#, dropped)
