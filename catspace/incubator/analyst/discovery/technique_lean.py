"""PLAN 73.6 (1) THE STATEMENT: the engine states a TECHNIQUE entry's obligation in Lean; it does not prove it. From an entry of the lemma store (kind TECHNIQUE, its
measure declaring a class, a depth d and a floor n) this module EMITS `lean/ideas/Ideas/Mined/<module>.lean` over the chess-lean rule kernel the library already
uses -- generated from the entry by code, nothing typed by hand per ending:

  THE CLASS `inClass`, from the material signature (technique_corpus.parse_signature): the side to move has exactly the winning side's units, the other side
    exactly its own (each unit a square named by the vocabulary's names: W our king, P our other piece, L their king, O theirs ...; every other square empty);
    no castling right, no en-passant square; neither side in check; the side to move has a legal move -- the miner's class (technique_sample.whole_class: every
    legal placement, the winning side to move, not in check, not over). The table's "won" is NOT in the predicate (the kernel has no table): where a class holds
    a position that is not won, the obligation is false there -- in K+R v K every position of the class is won (the 2026-10-26 whole-class OBS line).
  THE MOVES: `theirs q r` -- a legal move of the side to move (the kernel's `legalMoves`, `applyMove`: Ideas.Game.chess.step); `ours p q` -- the same with the
    NEVER-RULES of the miner (technique_miner.Graph.expand): no stalemate (the kernel's `stalemate`), no unit of ours (not the king) that the exchange law loses
    (`unitLost`: the vocabulary's `seeOf`, the x-ray static exchange of Ideas.ExchangeLaw, positive on a square of ours -- technique_miner.unit_lost reads
    graph.see_for); THE CLAIMABLE DRAWS are NOT in `ours` (see the header the file carries: the narrowest faithful version, labelled).
  THE GOAL `mate q`: the kernel's `checkmate` (after our move, they are mated).
  THE MEASURE: every atom of the entry's term compiled to a Lean definition over positions -- a vocabulary term by base_vocabulary.lean (the mirror of
    Ideas.Base.Vocabulary), the miner's three readings the vocabulary has no word for by their own generic definitions emitted beside them (`distSet`: the
    king-step distance to the nearest square of a set, 0 for none; `areaOf`: the squares of the smallest rectangle of files and ranks holding a set, 0 for
    none; whose move: 0 when the measured side is to move); a sum, a product, and THE LEXICOGRAPHIC TUPLE as its mixed-radix number at technique_miner.RADIX
    (every atom of the class reads below it: asserted by the miner) -- `mu p`, read for the side to move.
  THE OBLIGATION `hprog`, `by sorry`: exactly the premise of the PROVED Ideas.Forced.forced_by_measure at k = max(d, n) --
    `∀ p, inClass p → Ideas.Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) k p` (a mate within n is a Reach of the goal within k).
  THE COROLLARY `forced_mate`, proved from hprog by forced_by_measure (one application, emitted): the mate from every position of the class within
    k * (mu p + 1) of our moves. While hprog is a sorry the corollary rests on it (its body says so, and the index marks it OPEN).

The file must BUILD with exactly one `sorry` (`lake build Ideas.Mined.<module>`) before any prover runs: `build`. The measure's Lean reading is compared with the
miner's on positions of the class (`check_measure`: `#eval` of `mu` on the kernel's own FEN parse) -- the statement is the miner's quantity, not a lookalike.
An alternative measure (a zero-failure measure of the miner's chosen level, plan 73.6 (2)) is emitted the same way into `<module>_z<i>.lean`.

  python -m catspace.incubator.analyst.discovery.technique_lean emit "mined technique KRvK" [--zero <saved miner json>] [--build] [--check-measure 40]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery import technique_measures as TM
from catspace.incubator.analyst.discovery import technique_miner as MN
from catspace.io import paths

LEAN_DIR = Path(__file__).resolve().parents[4] / "lean" / "ideas"
MINED_DIR = LEAN_DIR / "Ideas" / "Mined"
KIND = {ch: chess.piece_name(pt) for ch, pt in TC.UNITS.items()}     # a signature's letter -> the kernel's PieceKind name (the corpus's own table)
UNIT_NAMES = {True: ["P", "P2", "P3", "P4", "P5"], False: ["O", "O2", "O3", "O4", "O5"]}      # the vocabulary's names of the other units (base_vocabulary)


class Inexpressible(ValueError):
    """A part of the statement the kernel (or this compiler) cannot express: a FINDING, never papered over."""


# ---------------------------------------------------------------------------------------------------------------- names

def module_name(entry: dict, suffix: str = "") -> str:
    """The Lean module of an entry: lemma_store.slug with '_' (a Lean module name is an identifier; the store's slug has '-'), plus the suffix."""
    return LS.slug(entry["name"]).replace("-", "_") + suffix


def namespace(entry: dict, suffix: str = "") -> str:
    return f"Ideas.Mined.{module_name(entry, suffix)}"


def lean_path(entry: dict, suffix: str = "") -> Path:
    return MINED_DIR / f"{module_name(entry, suffix)}.lean"


# ---------------------------------------------------------------------------------------------------------------- the class

def units(sig: str) -> list:
    """The class's units -> [(name, ours?, kind)]: W our king, the other units of ours P, P2 ... (the signature's order), L their king, O, O2 ..."""
    ours, theirs = TC.parse_signature(sig)
    out = []
    for mine, side in ((True, ours), (False, theirs)):
        others = iter(UNIT_NAMES[mine])
        for ch in side:
            out.append((("W" if mine else "L") if ch == "K" else next(others), mine, KIND[ch]))
    return out


def class_def(sig: str) -> str:
    """THE CLASS predicate of a signature (see the docstring)."""
    us = units(sig)
    names = " ".join(n for n, _m, _k in us)
    same = [f"{a} ≠ {b}" for i, (a, ma, ka) in enumerate(us) for (b, mb, kb) in us[i + 1:] if (ma, ka) == (mb, kb)]
    boards = [f"p.board {n} = some ⟨{'p.turn' if m else 'Color.opposite p.turn'}, .{k}⟩" for n, m, k in us]
    rest = " → ".join(f"s ≠ {n}" for n, _m, _k in us)
    body = " ∧\n      ".join(same + boards + [f"(∀ s : Square, {rest} → p.board s = none)"])
    gloss = ", ".join(f"{'our' if m else 'their'} {k} on {n}" for n, m, k in us)
    return (f"/-- THE CLASS {sig}, the winning side to move: {gloss}, every other square empty (the material signature, the units by the vocabulary's names);\n"
            f"no castling right, no en-passant square; neither side in check; the side to move has a legal move (the miner's class: technique_sample.whole_class). -/\n"
            f"def inClass (p : Position) : Prop :=\n"
            f"  (∃ {names} : Square,\n      {body}) ∧\n"
            f"  p.castling = ∅ ∧ p.enPassant = none ∧ inCheckB p p.turn = false ∧ inCheckB p (Color.opposite p.turn) = false ∧ legalMoves p ≠ []\n")


# ---------------------------------------------------------------------------------------------------------------- the measure

HELPERS = {
    "distSet": ("/-- (the miner's reading `(dist X S)`, technique_measures.dist_to) the king-step distance from `s` to the nearest square of the set `A`; 0 when\n"
                "`A` is empty. -/\n"
                "def distSet (s : Square) (A : SquareSet) : Nat :=\n"
                "  match (allSquares.filter A).map (Ideas.Vocabulary.dist s) with\n"
                "  | [] => 0\n"
                "  | x :: xs => xs.foldl min x\n\n"
                "/-- `distSet` from an optional square: 0 when there is no square (the vocabulary's `onum`). -/\n"
                "def odistSet : OSq → SquareSet → Nat\n"
                "  | some s, A => distSet s A\n"
                "  | none, _ => 0\n"),
    "areaOf": ("/-- (the miner's reading `(area S)`, technique_measures.area) the squares of the smallest rectangle of files and ranks that holds the set `A`; 0 when\n"
               "`A` is empty. -/\n"
               "def areaOf (A : SquareSet) : Nat :=\n"
               "  match allSquares.filter A with\n"
               "  | [] => 0\n"
               "  | l => ((l.map (·.1.val)).foldl max 0 + 1 - (l.map (·.1.val)).foldl min 7) * ((l.map (·.2.val)).foldl max 0 + 1 - (l.map (·.2.val)).foldl min 7)\n"),
}


def _is_set(t) -> bool:
    return BV.PRIMITIVES.get(BV.head(t), ((), "", ""))[1] == "set"


def compile_atom(sexpr: str) -> tuple:
    """One atom's s-expression -> (its Lean expression over `p c W L P O`, the helpers it needs). The miner's readings the vocabulary has no word for (area,
    dist to a set, whose move) are compiled here; every other part by base_vocabulary.lean. Raises Inexpressible for what neither says (a term reading the
    game's history: a position does not carry it)."""
    t = BV.parse(sexpr)
    need = set()

    def vocab(u) -> str:
        s = BV.sexpr(u) if not isinstance(u, str) else u
        if any(h in s for h in ("ago", "seen", "delta", "gained", "lost")):
            raise Inexpressible(f"{sexpr}: reads the game's history, which a single position does not carry")
        try:
            return BV.lean(u)
        except KeyError as x:
            raise Inexpressible(f"{sexpr}: no Lean rendering for {x}") from None
    if t == "turn":
        return "(if p.turn = c then 0 else 1)", need
    if isinstance(t, tuple) and t[0] == "area":
        need.add("areaOf")
        return f"(areaOf {vocab(t[1])})", need
    if isinstance(t, tuple) and t[0] == "dist" and _is_set(t[2]):
        need.add("distSet")
        return f"(odistSet {vocab(t[1])} {vocab(t[2])})", need
    return vocab(t), need


def compile_measure(sexpr: str) -> dict:
    """The measure's term -> {atoms: [(name, sexpr, gloss, lean)], mu: the Lean body of mu over the atom names, helpers, op}. A sum and a product of two atoms;
    the lexicographic tuple as its mixed-radix number (technique_miner.RADIX)."""
    m = MN.measure_of(sexpr)
    if m is None:
        raise Inexpressible(f"{sexpr!r} is no measure of the miner's libraries")
    atoms, helpers = [], set()
    for i, k in enumerate(m.parts, 1):
        a = TM.ATOM_BY_KEY[k]
        expr, need = compile_atom(a.sexpr)
        helpers |= need
        atoms.append((f"atom{i}", a.sexpr, a.gloss, expr))
    call = [f"{n} p c W L P O" for n, _s, _g, _e in atoms]
    if m.op == "atom":
        mu = call[0]
    elif m.op == "sum":
        mu = f"{call[0]} + {call[1]}"
    elif m.op == "product":
        mu = f"{call[0]} * {call[1]}"
    else:
        mu = call[0]
        for c in call[1:]:
            mu = f"({mu}) * {MN.RADIX} + {c}"
    return dict(atoms=atoms, mu=mu, helpers=sorted(helpers), op=m.op, gloss=m.gloss)


# ---------------------------------------------------------------------------------------------------------------- the file

def _header(entry: dict, sexpr: str, k: int, suffix: str) -> str:
    mm = entry["measure"]
    alt = "" if not suffix else f" -- here the ALTERNATIVE measure {sexpr} (a zero-failure measure of the miner's chosen level, plan 73.6 (2)), not the entry's"
    return (
        f"/-! AUTO-GENERATED by catspace/incubator/analyst/discovery/technique_lean.py from the lemma store entry \"{entry['name']}\"\n"
        f"(catspace/incubator/analyst/discovery/lemmas/{LS.slug(entry['name'])}.json) -- do not edit by hand. PLAN 73.6 (1): THE STATEMENT of the entry's\n"
        f"obligation{alt}. The engine states it; it does not prove it: `hprog` is the one `sorry`, the provers' task (prover.py), and nothing here is typed per ending.\n\n"
        f"The entry: class {mm['class']}, measure {mm['quantity']}, depth d = {mm['depth']}, floor n = {mm['n']}, status {entry['status']}. `hprog` is the premise of\n"
        f"the PROVED Ideas.Forced.forced_by_measure at k = max(d, n) = {k}; `forced_mate` applies it.\n\n"
        f"THE NARROWEST FAITHFUL VERSION (labelled): the CLAIMABLE DRAWS are not in `ours`. The kernel states them over a game's HISTORY (Chess.threefoldClaimable\n"
        f"of a list of positions) and over a position's fifty-move clock (Chess.fiftyMoveClaimableAfter); `Ideas.Forced.Reach` is a relation of single positions,\n"
        f"so repetition is not expressible in it, and the fifty-move rule over the clock would make the obligation FALSE (from a position whose clock reads 99\n"
        f"every move of ours reaches a claimable draw) while the corollary's bound {k} * (mu p + 1) can exceed fifty moves. So this is the game of the moves, check,\n"
        f"mate and stalemate (a stalemate is no success, as in Reach) WITHOUT the claimable draws -- the miner's own reading (technique_miner's docstring: its\n"
        f"search needs neither). The class does not say \"won\" (the kernel has no table): in K+R v K every position of the class is won (2026-10-26 OBS).\n"
        f"The measure's Lean reading is checked against the miner's on positions of the class (technique_lean.check_measure). -/\n")


def emit_text(entry: dict, sexpr: "str | None" = None, suffix: str = "", proof: "dict | None" = None) -> str:
    """The Lean file of `entry` (its measure, or the alternative `sexpr`) -> the text. `proof` (plan 73.6 (4), only from an ACCEPTED run of prover.py):
    {proof: the tactic text of hprog, lemmas: the helper lemmas the tool wrote, tool: who found it, run: the run's id} -- written in place of the sorry, by code;
    nobody types it."""
    mm = entry.get("measure") or {}
    if entry.get("kind") != "TECHNIQUE" or not {"class", "depth", "n", "quantity"} <= set(mm):
        raise Inexpressible(f"{entry.get('name')}: not a TECHNIQUE with a class, a depth, a floor n and a quantity")
    sexpr = sexpr or mm["quantity"]
    k = max(int(mm["depth"]), int(mm["n"]))
    cm = compile_measure(sexpr)
    us = units(mm["class"])
    names = {n for n, _m, _k in us}
    P = ("((nthPiece p p.turn 1).getD ((0 : Fin 8), (0 : Fin 8)))" if "P" in names else "((0 : Fin 8), (0 : Fin 8))")
    O = ("((nthPiece p (Color.opposite p.turn) 1).getD ((0 : Fin 8), (0 : Fin 8)))" if "O" in names else "((0 : Fin 8), (0 : Fin 8))")
    out = ["import Ideas.ForcedByMeasure", "import Ideas.Base.Vocabulary", "", _header(entry, sexpr, k, suffix),
           "set_option linter.unusedVariables false", "", f"namespace {namespace(entry, suffix)}", "",
           "open Chess hiding between   -- the chess library's `between` is a Prop; the vocabulary's is a set", "open Ideas.Vocabulary", "",
           "/-! ## The class -/", "", class_def(mm["class"]),
           "/-! ## The moves -/", "",
           "/-- THEIRS: a legal move of the side to move (the kernel's `legalMoves` and `applyMove`; Ideas.Game.chess.step). -/",
           "def theirs (q r : Position) : Prop := ∃ m, m ∈ legalMoves q ∧ r = applyMove q m", "",
           "/-- A UNIT LOST by the exchange law: after our move (they to move), a piece of ours other than the king stands where they win by the exchange (the\n"
           "vocabulary's `seeOf`, the x-ray static exchange of Ideas.ExchangeLaw; technique_miner.unit_lost). -/",
           "def unitLost (q : Position) : Prop :=", "  ∃ s pc, q.board s = some pc ∧ pc.color = Color.opposite q.turn ∧ pc.kind ≠ .king ∧ 0 < seeOf q s", "",
           "/-- OURS: a legal move of the side to move under the miner's NEVER-RULES -- it does not stalemate, and it leaves no unit the exchange law loses (the\n"
           "claimable draws: see the header). -/",
           "def ours (p q : Position) : Prop := ∃ m, m ∈ legalMoves p ∧ q = applyMove p m ∧ ¬ stalemate q ∧ ¬ unitLost q", "",
           "/-- THE GOAL, read after our move: the side to move (they) is mated (the kernel's `checkmate`). -/",
           "def mate (q : Position) : Prop := checkmate q", "",
           f"/-! ## The measure {sexpr}, compiled from its term -/", ""]
    for h in cm["helpers"]:
        out += [HELPERS[h]]
    for n, s, g, e in cm["atoms"]:
        out += [f"/-- {s}: {g}. -/", f"def {n} (p : Position) (c : Color) (W L P O : Square) : Nat := {e}", ""]
    rank = (f"the lexicographic tuple as its mixed-radix number, radix {MN.RADIX} (technique_miner.RADIX: every atom of the class reads below it)"
            if cm["op"] == "lex" else cm["op"])
    out += [f"/-- THE MEASURE {sexpr} -- {cm['gloss']} -- read for the side to move, {rank}; the units by the vocabulary's names (W our king, L their king, P our\n"
            f"other piece, O theirs; a1 when absent). -/",
            "def mu (p : Position) : Nat :=",
            f"  let c := p.turn; let W := kingSqD p c; let L := kingSqD p (Color.opposite c); let P := {P}; let O := {O}",
            f"  {cm['mu']}", "",
            "/-! ## The obligation and the forced mate -/", ""]
    if proof is not None:
        out += [f"/-! The proof below was FOUND by {proof['tool']} (prover.py, run {proof['run']}) and accepted by its check (no sorry, the standard axioms\n"
                f"only, no enumeration); written here by technique_lean.emit_text, not by hand. -/", ""]
        if proof.get("lemmas"):
            out += [proof["lemmas"].rstrip(), ""]
    body = indent_proof(proof["proof"]).splitlines() if proof is not None else ["  sorry"]
    status = "PROVED (the proof found by the tools, below)" if proof is not None else "OPEN: the provers' task (plan 73.6 (3))"
    rests = "  -- from hprog, proved above" if proof is not None else "  -- rests on hprog: OPEN while hprog's proof is the sorry above"
    out += [f"/-- THE OBLIGATION (the entry's; the premise `hprog` of the PROVED Ideas.Forced.forced_by_measure at k = {k}): from every position of the class we force,\n"
            f"within {k} of our moves and against every reply, the mate or a position of the class whose measure is lower. {status}. -/",
            f"theorem hprog : ∀ p, inClass p → Ideas.Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) {k} p := by"] + body + ["",
            f"/-- THE FORCED MATE from every position of the class within {k} * (mu p + 1) of our moves: Ideas.Forced.forced_by_measure applied to `hprog`. -/",
            f"theorem forced_mate : ∀ p, inClass p → Ideas.Forced.Reach ours theirs mate (fun _ => False) ({k} * (mu p + 1)) p :=",
            rests,
            f"  Ideas.Forced.forced_by_measure inClass mu {k} hprog", "",
            f"end {namespace(entry, suffix)}", ""]
    return "\n".join(out)


def indent_proof(proof: str, k: int = 2) -> str:
    """A tactic proof as the block placed after `:= by`, every line at `k` spaces plus its own relative indentation -- ONE function for the scratch check
    (prover.py) and the file (emit_text), so what is written is what was accepted. A first line that came from the `by` line itself (no indentation, the
    rest indented) stands at the level of the rest."""
    lines = proof.strip("\n").splitlines()
    if not lines:
        return ""
    pads = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    rest = [len(l) - len(l.lstrip()) for l in lines[1:] if l.strip()]
    if rest and pads[0] == 0 and min(rest) > 0:
        base = min(rest)
        lines = [" " * base + lines[0].strip()] + lines[1:]
    pad = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join((" " * k + l[pad:]) if l.strip() else "" for l in lines)


def emit(entry: dict, sexpr: "str | None" = None, suffix: str = "", proof: "dict | None" = None) -> Path:
    p = lean_path(entry, suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(emit_text(entry, sexpr, suffix, proof))
    return p


SORRY_WARNING = re.compile(r"declaration uses [`']sorry[`']")


def build(module: str, timeout: int = 1800) -> dict:
    """`lake build Ideas.Mined.<module>` in lean/ideas -> {ok, sorries (the 'declaration uses sorry' warnings), errors (lines), secs, output}."""
    t0 = time.monotonic()
    r = subprocess.run(["lake", "build", f"Ideas.Mined.{module}"], cwd=LEAN_DIR, capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    errs = [ln for ln in out.splitlines() if "error" in ln.lower() and "Build completed" not in ln]
    own = re.compile(r"Ideas/Mined/" + re.escape(module) + r"\.lean:\d+:\d+: declaration uses [`']sorry[`']")      # (Lake replays the imports' warnings too)
    return dict(ok=r.returncode == 0, sorries=len(own.findall(out)), errors=errs, secs=round(time.monotonic() - t0, 1), output=out)


UNEVALUABLE = ("reach",)       # the vocabulary's Lean `reach` is `Nat.repeat step 64` over closures: its #eval is exponential (each step reads the last at 65 squares)


def check_measure(entry: dict, fens: list, sexpr: "str | None" = None, suffix: str = "", timeout: int = 1800) -> dict:
    """THE MEASURE'S READING compared, atom by atom: each atom of the emitted module, evaluated by Lean (`#eval` on the kernel's own `parseFen`, the units bound
    as `mu` binds them), against the miner's value (technique_measures.atom_values) on `fens` -> {n, atoms: {name: (sexpr, equal | None, skipped reason)},
    diffs [(atom, fen, python, lean)], secs}. An atom reading `reach` is NOT evaluated (UNEVALUABLE: the vocabulary's Lean `reach` iterates a closure 64 times,
    so its evaluation is exponential -- a finding about evaluating the statement, not about its truth); its agreement rests on the vocabulary's own tests
    (the Rust evaluator mirrors the Lean definition: tests/test_plan73_5.py reads the region atoms through it)."""

    m = MN.measure_of(sexpr or entry["measure"]["quantity"])
    ns = namespace(entry, suffix)
    cm = compile_measure(sexpr or entry["measure"]["quantity"])
    keys = dict(zip((n for n, _s, _g, _e in cm["atoms"]), m.parts))
    bind = ("(fun p => {f} p p.turn (kingSqD p p.turn) (kingSqD p (Color.opposite p.turn)) ((nthPiece p p.turn 1).getD ((0 : Fin 8), (0 : Fin 8))) "
            "((nthPiece p (Color.opposite p.turn) 1).getD ((0 : Fin 8), (0 : Fin 8))))")
    lines = [f"import Ideas.Mined.{module_name(entry, suffix)}", "", "open Chess hiding between", "open Ideas.Vocabulary", ""]
    todo, atoms = [], {}
    sexprs = {n: s for n, s, _g, _e in cm["atoms"]}
    for n, s, _g, _e in cm["atoms"]:
        if set(re.findall(r"[\w-]+", s)) & set(UNEVALUABLE):
            atoms[n] = (s, None, "not evaluated: the vocabulary's Lean `reach` iterates a closure 64 times (an exponential #eval)")
            continue
        todo.append(n)
        for f in fens:
            lines.append(f"#eval ((Chess.parseFen \"{f}\").map {bind.format(f=ns + '.' + n)})")
    t0 = time.monotonic()
    got = []
    if todo:
        scratch = Path(paths.experiment("prover", "scratch", f"check_measure_{module_name(entry, suffix)}.lean"))
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_text("\n".join(lines) + "\n")
        r = subprocess.run(["lake", "env", "lean", str(scratch)], cwd=LEAN_DIR, capture_output=True, text=True, timeout=timeout)
        got = re.findall(r"^(?:some (\d+)|none)$", r.stdout, re.M)
    diffs = []
    it = iter(got)
    for n in todo:
        eq = 0
        for f in fens:
            g = next(it, "x")
            b = chess.Board(f)
            py = TM.atom_values(b, b.turn)[keys[n]]
            le = int(g) if g and g != "x" else None
            if le == py:
                eq += 1
            else:
                diffs.append((n, f, py, le))
        atoms[n] = (sexprs[n], eq, None)
    return dict(n=len(fens), atoms=atoms, diffs=diffs, secs=round(time.monotonic() - t0, 1))


def class_fens(sig: str, n: int, seed: int = 7) -> list:
    """`n` random positions of the class (technique_corpus.random_start: legal, the winning side to move, not in check, not over)."""
    import random
    rng = random.Random(seed)
    return [TC.random_start(rng, sig, 1).fen() for _ in range(n)]


def zero_measures(saved: str) -> list:
    """The zero-failure measures of a saved miner result's chosen level (technique_miner.save), simplest first."""
    got = json.load(open(saved))
    return list((got.get("choice") or {}).get("zero") or [])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="emit the entry's statement (and the alternatives) into lean/ideas/Ideas/Mined/")
    e.add_argument("name", help="the store entry, e.g. 'mined technique KRvK'")
    e.add_argument("--zero", default=None, help="a saved miner result: also emit every zero-failure measure of its chosen level (<module>_z<i>.lean)")
    e.add_argument("--build", action="store_true", help="lake build each emitted module (it must build with exactly one sorry)")
    e.add_argument("--check-measure", type=int, default=0, help="compare Lean's mu with the miner's on this many random positions of the class")
    a = ap.parse_args(argv)
    entry = LS.load(force=True).get(a.name)
    if entry is None:
        raise SystemExit(f"no store entry {a.name!r}")
    todo = [(None, "")]
    if a.zero:
        own = entry["measure"]["quantity"]
        todo += [(s, f"_z{i}") for i, s in enumerate((s for s in zero_measures(a.zero) if s != own), 1)]
    for sexpr, suffix in todo:
        p = emit(entry, sexpr, suffix)
        print(f"OBS technique_lean emitted {os.path.relpath(p, str(paths.REPO_ROOT))}: {entry['name']}, measure {sexpr or entry['measure']['quantity']}")
        if a.build:
            b = build(module_name(entry, suffix))
            print(f"OBS technique_lean build Ideas.Mined.{module_name(entry, suffix)}: ok {b['ok']}, sorry warnings {b['sorries']}, error lines {len(b['errors'])}, "
                  f"{b['secs']} s" + ("".join("\n  " + x for x in b["errors"][:12]) if b["errors"] else ""))
        if a.check_measure and (not a.build or b["ok"]):
            c = check_measure(entry, class_fens(entry["measure"]["class"], a.check_measure), sexpr, suffix)
            print(f"OBS technique_lean measure check Ideas.Mined.{module_name(entry, suffix)} on {c['n']} random positions of the class: "
                  + "; ".join(f"{n} {s}: " + (f"Lean equals the miner on {eq} of {c['n']}" if why is None else why) for n, (s, eq, why) in c["atoms"].items())
                  + f"; {c['secs']} s" + ("".join(f"\n  DIFF {x}" for x in c["diffs"][:6]) if c["diffs"] else ""))


if __name__ == "__main__":
    main()
