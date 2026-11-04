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
