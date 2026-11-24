"""PLAN 79.37, 79.38, 79.40, 79.43: make the generated Rust files again (rust/engine/game/tests/generated/<game>.rs and <game>_<variant>.rs) from the descriptions with the translator (catspace/incubator/analyst/discovery/gdl_to_rust.py).
This script names the games; the translator does not. Run it after the translator or a description changed: tests/test_plan79_37.py and tests/test_plan79_40.py fail on a file whose hashes differ.
Only the plain file of a game (no pass) and the file with every pass (`fast`) are kept in the repository (plan 79.40 part F). With --scratch the files of the single passes and of the four passes of 79.38 together (`prev`) are made under
generated/scratch (not kept: that folder is ignored); the tests that hold each pass to the plain module, and the clocks, are compiled with `--cfg singles` and read them there.
  .venv/bin/python scripts/gdl_to_rust_make.py [--scratch]
Plan 79.45: the first game also has a kept file with every pass and its lemma (`chess_lemma`); the folder rules/lemmas/<game>/ is read HERE (through rule_lemma and rule_lemma_certify), the translator is given only the lemmas of form SAME whose computed
standing is CERTIFIED or PROVED, and each other lemma is printed as refused. The scratch folder also holds `_p43` (every pass of 79.43: no gated, no lemma), `_lm_plain` (no pass, the lemma) and `_lm_nogate` (every pass but gated, the lemma).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from catspace.incubator.analyst.discovery import gdl_to_rust as T

GAMES = {"chess": [ROOT / "rules" / "chess.gdl"], "atomic": [ROOT / "rules" / "chess.gdl", ROOT / "rules" / "variants" / "atomic.gdl"]}
KEPT = {"": [], "_fast": list(T.PASSES)}
KEPT_LEMMA = {"chess": {"_lemma": list(T.PASSES)}}                  # plan 79.45: the game that has a lemma folder, every pass and its lemmas (the one file kept besides the plain and the fast ones)
COMBOS = {"bits_keep": ["bits", "keep"], "bits_order": ["bits", "order"], "bits_order_keep": ["bits", "order", "keep"], "bits_demand": ["bits", "demand"], "bits_dense": ["bits", "dense"], "bits_copy": ["bits", "copy"], "bits_delta": ["bits", "delta"]}
# plan 79.41: each new pass alone on top of bits (the clocks), and `prev`, the module of every pass of 79.40 (the one the new passes are clocked against); `prev38` is the four passes of 79.38
PASSES_7940 = ["dense", "demand", "copy", "delta", "bits", "order", "keep"]
PASSES_7941 = PASSES_7940 + ["scan", "cut", "shift", "base"]
NEW = [p for p in T.PASSES if p in PASSES_7941 and p not in PASSES_7940]
COMBOS.update({f"bits_{p}": ["bits", p] for p in NEW})
# plan 79.42: each new pass alone on top of every pass of 79.41 is `prev41_<pass>`; `prev40` is every pass of 79.40
NEW42 = [p for p in T.PASSES if p not in PASSES_7941 and p in ("start", "ends", "orient")]
PASSES_7942 = PASSES_7941 + NEW42
# plan 79.43: `prev` is every pass of 79.42, the module the new passes are clocked against (for the clocks it is made by the translator of the commit before the plan, since the estimate of the pass start was mended in the plan); `prev_42` is the same passes made by this translator; each new pass alone on top of it is `prev_<pass>`
NEW43 = [p for p in T.PASSES if p not in PASSES_7942]
SCRATCH = {"_prev": PASSES_7942, "_prev_42": PASSES_7942, "_prev_noworth": [p for p in T.PASSES if p != "worth"], "_prev41": PASSES_7941, "_prev40": PASSES_7940, "_prev38": ["dense", "demand", "copy", "delta"], **{f"_prev41_{p}": PASSES_7941 + [p] for p in NEW42}, **{f"_prev_{p}": PASSES_7942 + [p] for p in NEW43}, **{f"_{p}": [p] for p in T.PASSES}, **{f"_{k}": v for k, v in COMBOS.items()}}
# plan 79.45: the module of every pass of 79.43 (no gated, no lemma) is the one the lemma's clocks are held against; `_lm_plain` the rewrite with no pass; `_lm_nogate` the lemma with every pass but gated
PASSES_NOGATE = [p for p in T.PASSES if p != "gated"]
SCRATCH_LEMMA = {"chess": {"_lm_plain": [], "_lm_nogate": PASSES_NOGATE}}
SCRATCH.update({"_p43": PASSES_NOGATE})
OUT = ROOT / "rust" / "engine" / "game" / "tests" / "generated"
OUT_SCRATCH = OUT / "scratch"


def lemma_docs(game: str) -> tuple:
    """-> (the composed text of the game's lemmas, the lemmas the translator may be given). Reads rules/lemmas/<game>/; a lemma that is not of form SAME or whose computed standing is neither CERTIFIED nor PROVED is refused here and printed."""
    from catspace.incubator.analyst.discovery import rule_lemma as RL, rule_lemma_certify as RC
    docs, text = [], None
    for lem in RL.load_all(game):
        word, detail = RC.standing(lem)
        if lem.form != "SAME" or word not in ("CERTIFIED", "PROVED"):
            print(f"OBS 79.45 lemma {game}/{lem.name} not given to the translator: form {lem.form}, standing {word} ({detail})")
            continue
        text = text or RL.composed_text(lem)
        docs.append({"name": lem.name, "form": lem.form, "left": lem.left, "right": lem.right, "condition": lem.condition, "standing": {"is": word, **({"key": detail} if word == "CERTIFIED" else {"theorem": detail})}})
        print(f"OBS 79.45 lemma {game}/{lem.name} given to the translator: form {lem.form}, standing {word} ({detail})")
    return text, docs


def make(out: Path, variants: dict, lemma_variants: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for game, sources in GAMES.items():
        text = T.compose_text(sources) if len(sources) > 1 else sources[0].read_text()
        for variant, passes in variants.items():
            module = game + variant
            src = T.translate(text, module, passes)
            (out / f"{module}.rs").write_text(src)
            print(f"OBS 79.40 translate {module}: description {len(text)} bytes, passes {','.join(passes) or '-'}, Rust {len(src)} bytes {src.count(chr(10))} lines")
    for game, vs in (lemma_variants or {}).items():
        ltext, docs = lemma_docs(game)
        for variant, passes in vs.items():
            module = game + variant
            src = T.translate(ltext, module, passes, docs)
            (out / f"{module}.rs").write_text(src)
            print(f"OBS 79.45 translate {module}: composed description {len(ltext)} bytes, passes {','.join(passes) or '-'}, lemmas {','.join(x['name'] for x in docs)}, Rust {len(src)} bytes {src.count(chr(10))} lines")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    make(OUT, KEPT, KEPT_LEMMA)
    if "--scratch" in args:
        make(OUT_SCRATCH, SCRATCH, SCRATCH_LEMMA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
