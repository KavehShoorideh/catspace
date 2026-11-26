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
# 2026-11-24 (plan 79.47): the five more games of the seven certified kernels, each the base text composed with its delta by the same translator (the kept files are generated/<game>.rs and <game>_fast.rs; the Rust module is named by `rust_name`: a name that begins with a digit is spelled)
for _g in ("3check", "kingofthehill", "antichess", "chess960", "horde"):
    GAMES[_g] = [ROOT / "rules" / "chess.gdl", ROOT / "rules" / "variants" / f"{_g}.gdl"]
SCRATCH_GAMES = ("chess", "atomic")                                 # the files of the single passes and the clocks of the passes are made for the first two games only
PASSES_NOCARRY = [p for p in T.PASSES if p != "carry"]              # plan 79.54: the pass carry (answers kept from the last state) is in the modules `_carry`, beside the modules `_fast` of every other pass, until the clocks say which a game uses
KEPT = {"": [], "_fast": PASSES_NOCARRY, "_carry": list(T.PASSES)}
KEPT_SUPPORT = {"_support": PASSES_NOCARRY + ["support"]}      # plan 79.56 (pass 26): the module with the support filter, kept for every game where the shape is found (the pass is not one of PASSES: a caller asks for it by name)
KEPT_LEMMA = {"chess": {"_lemma": PASSES_NOCARRY}}                  # plan 79.45: the game that has a lemma folder, every pass and its lemmas (the one file kept besides the plain and the fast ones)
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
PASSES_7943 = [p for p in T.PASSES if p not in ("cones", "words", "tables", "topdown", "carry")]      # plan 79.52: the two passes of that plan, and plan 79.53 the pass tables, come after every pass of 79.43
PASSES_7952 = [p for p in T.PASSES if p not in ("tables", "topdown", "carry")]               # plan 79.53: every pass of 79.52 (the module the tables are clocked against)
NEW43 = [p for p in PASSES_7943 if p not in PASSES_7942]
SCRATCH = {"_prev": PASSES_7942, "_prev_42": PASSES_7942, "_prev_noworth": [p for p in PASSES_7943 if p != "worth"], "_prev41": PASSES_7941, "_prev40": PASSES_7940, "_prev38": ["dense", "demand", "copy", "delta"], **{f"_prev41_{p}": PASSES_7941 + [p] for p in NEW42}, **{f"_prev_{p}": PASSES_7942 + [p] for p in NEW43}, **{f"_{p}": [p] for p in T.PASSES}, **{f"_{k}": v for k, v in COMBOS.items()}}
# plan 79.45: the module of every pass of 79.43 (no gated, no lemma) is the one the lemma's clocks are held against; `_lm_plain` the rewrite with no pass; `_lm_nogate` the lemma with every pass but gated
PASSES_NOGATE = [p for p in PASSES_7943 if p != "gated"]
SCRATCH_LEMMA = {"chess": {"_lm_plain": [], "_lm_nogate": PASSES_NOGATE}}
SCRATCH.update({"_p43": PASSES_NOGATE})
# plan 79.52: each of the two new passes alone on top of every pass of 79.43 (`_p52_cones`, `_p52_words`); the module with both is `fast`
SCRATCH.update({"_p52_cones": PASSES_7943 + ["cones"], "_p52_words": PASSES_7943 + ["words"]})
# plan 79.53: the tables alone on top of every pass of 79.52 (`_p53_tables`); the module with every pass is `fast`
SCRATCH.update({"_p53_tables": PASSES_7952 + ["tables"], "_p53_topdown": PASSES_7952 + ["tables", "topdown"]})
# plan 79.53 (part B3): the way that answers each question of a game, chosen by the clocks of tests/translated_speed_bits.rs (`chess_the_times_of_plan_53`, `atomic_the_times_of_plan_53`) and recorded in scripts/gdl_to_rust_ways.json:
# {game: {"end": "bottom-up" | "top-down", "legal-turn": ...}}; a game with no entry is answered bottom-up. The script reads the file; it chooses nothing.
import json as _json
WAYS = _json.loads((ROOT / "scripts" / "gdl_to_rust_ways.json").read_text()) if (ROOT / "scripts" / "gdl_to_rust_ways.json").exists() else {}
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


def rust_name(game: str) -> str:
    """The name of the Rust module of a game: the game's name, with a digit it begins with spelled (a Rust name does not begin with one)."""
    digits = "zero one two three four five six seven eight nine".split()
    return digits[int(game[0])] + game[1:] if game[:1].isdigit() else game


def make(out: Path, variants: dict, lemma_variants: dict | None = None, games: tuple | None = None, support: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for game, sources in GAMES.items():
        if games is not None and game not in games:
            continue
        text = T.compose_text(sources) if len(sources) > 1 else sources[0].read_text()
        for variant, passes in variants.items():
            module = rust_name(game) + variant
            src = T.translate(text, module, passes, ways=WAYS.get(game))
            (out / f"{game}{variant}.rs").write_text(src)
            print(f"OBS 79.40 translate {game}{variant} (module {module}): description {len(text)} bytes, passes {','.join(passes) or '-'}, Rust {len(src)} bytes {src.count(chr(10))} lines")
    for variant, passes in KEPT_SUPPORT.items() if (support and games is None) else ():
        for game, sources in GAMES.items():
            text = T.compose_text(sources) if len(sources) > 1 else sources[0].read_text()
            module = rust_name(game) + variant
            src = T.translate(text, module, passes, ways=WAYS.get(game))
            heads = [x[3:] for x in src.split("\n", 12)[:12] if x.startswith("// support")]
            if "// support: test " not in src:
                (out / f"{game}{variant}.rs").unlink(missing_ok=True)
                print(f"OBS 79.56 translate {game}{variant}: the shape is not found or no question reads it, no module kept; " + " | ".join(heads))
                continue
            (out / f"{game}{variant}.rs").write_text(src)
            print(f"OBS 79.56 translate {game}{variant} (module {module}): passes {','.join(passes)}, Rust {len(src)} bytes {src.count(chr(10))} lines; " + " | ".join(h[:150] for h in heads[:1]))
    for game, vs in (lemma_variants or {}).items():
        ltext, docs = lemma_docs(game)
        for variant, passes in vs.items():
            module = rust_name(game) + variant
            src = T.translate(ltext, module, passes, docs, ways=WAYS.get(game))
            (out / f"{game}{variant}.rs").write_text(src)
            print(f"OBS 79.45 translate {module}: composed description {len(ltext)} bytes, passes {','.join(passes) or '-'}, lemmas {','.join(x['name'] for x in docs)}, Rust {len(src)} bytes {src.count(chr(10))} lines")


# plan 79.57: the placed questions asked for one unit (kind, side, place = the first four places), with the reads of the answer (this script may name the relations; the translator does not).
# plan 79.58 part A: the same two asked for one ROW (all six places bound: the reads of one row of the answer, for the measure of a support by row); part B: the allowed moves of one unit (placed_allowed: side and place = the first three places; made by `allowed_unfolded` below).
READS_KEYED = (("placed_bears", 4), ("placed_reaches", 4), ("placed_bears", 6), ("placed_reaches", 6), ("placed_allowed", 3))


def allowed_unfolded(text: str) -> list:
    """Plan 79.58 part B. The top-down way refuses a literal with a compound term, and the placed question for the allowed moves reads `(legal ?c (move ?f1 ?r1 ?f2 ?r2))`. A literal of a relation with a compound is UNFOLDED by the clauses of that relation whose head
    has the same functor (resolution, the bodies copied with the head's variables put for the literal's and every other variable renamed): the result is a relation `placed_allowed` of the same rows as `placed_ends`, with no compound term, made of the description's own clauses and no rule of the game.
    TAKES the composed text; RETURNS the lines to add (a signature and one clause for each pair of an ends clause and a legal clause of its functor)."""
    import re
    lines = text.splitlines()
    legal = [m for m in (re.match(r"\(<= \(legal (\?\w+) \((\w+) ([^()]*)\)\) (.*)\)$", l) for l in lines) if m]
    out = ["(signature placed_allowed side file rank file rank)"]
    for l in lines:
        e = re.match(r"\(<= \(placed_ends ([^()]*)\) \(legal (\?\w+) \((\w+) ([^()]*)\)\)\)$", l)
        if not e:
            continue
        head, c, functor, args = e.group(1), e.group(2), e.group(3), e.group(4).split()
        for m in legal:
            if m.group(2) != functor or len(m.group(3).split()) != len(args):
                continue
            sub = {m.group(1): c, **dict(zip(m.group(3).split(), args))}
            out.append("(<= (placed_allowed " + head + ") " + re.sub(r"\?\w+", lambda v: sub.get(v.group(0), v.group(0) + "_u"), m.group(4)) + ")")
    return out


OUT_PLACED = ROOT / "rust" / "engine" / "game" / "tests" / "generated_placed"


def make_placed(out: Path = OUT_PLACED) -> None:
    """2026-11-24 (plan 79.48 part C): the module of each of the seven games' descriptions composed with the additive text of the placed questions (rules/placed/), every pass, kept as generated_placed/<game>.rs (the Rust module is
    `placed_<rust_name>`). The composition is discovery/placed_questions.py's `composed_text`; the translator is given the text as it is."""
    from catspace.incubator.analyst.discovery import placed_questions as PQ
    out.mkdir(parents=True, exist_ok=True)
    for game in GAMES:
        text = PQ.composed_text(game)
        module = "placed_" + rust_name(game)
        src = T.translate(text, module, PASSES_NOCARRY)
        (out / f"{game}.rs").write_text(src)
        print(f"OBS 79.48 translate {game} (module {module}): composed description {len(text)} bytes, passes {','.join(PASSES_NOCARRY)}, Rust {len(src)} bytes {src.count(chr(10))} lines")
        # plan 79.57 (pass 27): the same module with the reads of an answer: the bearing and the reach of a unit, asked with the unit's kind, side and place as the key
        rmodule = module + "_reads"
        rsrc = T.translate(text + "\n" + "\n".join(allowed_unfolded(text)), rmodule, PASSES_NOCARRY + ["reads"], reads=READS_KEYED)
        (out / f"{game}_reads.rs").write_text(rsrc)
        print(f"OBS 79.57 translate {game}_reads (module {rmodule}): passes {','.join(PASSES_NOCARRY)},reads, Rust {len(rsrc)} bytes {rsrc.count(chr(10))} lines; " + " | ".join(x[3:] for x in rsrc.split(chr(10), 12)[:12] if x.startswith("// reads")))


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--placed" in args:
        make_placed()
        return 0
    make(OUT, KEPT, KEPT_LEMMA, None, True)
    if "--scratch" in args:
        make(OUT_SCRATCH, SCRATCH, SCRATCH_LEMMA, SCRATCH_GAMES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
