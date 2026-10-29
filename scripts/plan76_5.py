"""plan 76.5 THE VALUATION AS ONE DEFINITION -- the harness (docs/ANALYST_PLAN.md 76.5).

  .venv/bin/python scripts/plan76_5.py scan [--all]
        (b) THE SCAN: the Rust crate (rust/analyst_core/src), catspace/, tests/ and lean/ideas/Ideas grepped for the hand valuation's numbers in a WORTH ROLE -- a
        table of unit kinds to numbers (a dict, a match, a list in the kinds' order), the pawn's worth written as a literal (an en-passant victim, a promotion's
        "less the pawn"), the promotion's stake (the queen less the pawn) and a threshold on a worth reading (">= 3": a piece, not a pawn) -- every hit printed
        with its scope: ENGINE (the crate, discovery/, and every catspace module discovery/ imports, found by the import graph), OUTSIDE (the rest of catspace/:
        no engine module imports it), TESTS, LEAN. An ENGINE hit is routed to the store or is a FINDING of FINDINGS below with its reason; `--all` also prints
        the OUTSIDE hits one by one (by default their count per directory). Exit 1 on an ENGINE hit that is neither.
  .venv/bin/python scripts/plan76_5.py swap mined|scaled|hand
        (d) THE EXPERIMENT'S SWITCH (a run-time choice of the harness, never committed): the five GOAL entries count in the mined valuation (`material (mined)`) or
        back in the hand one -- each GOAL file's `params.valuation`, nothing else; `hand` restores the files byte for byte (checked against git's copy). `scaled`:
        the scale control (a temporary entry, the hand worths and unit times two; `hand` removes it).
  .venv/bin/python scripts/plan76_5.py decisions <hand dir> <mined dir> [--stockfish]
        (d) every guard DECISION that differs between two snapshots (the ten + wide, tiers 0 and 2), with Stockfish depth 14's reading of both moves (a separate
        process, `--stockfish`: the grader's verdict; never a premise of the engine).
  .venv/bin/python scripts/plan76_5.py batch <tag> [--seconds 15]
        (d) the loop-2 batch (rand14_10 .. rand17_10, `analyse --seconds 15 --qlog --quiet`, CATSPACE_QLOG to a scratch log), graded by `question_prior grade
        --log` in its own process; run once with the GOAL entries on the hand valuation and once on the mined one (`swap`), one sitting each.
  .venv/bin/python scripts/plan76_5.py batch-diff <hand qlog> <mined qlog>
        (d) the batch's agendas whose decision, grade or proof differ between the two sittings.
Outputs under artifacts/experiments/plan76_5/ (untracked)."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust" / "analyst_core" / "src"
CATSPACE = ROOT / "catspace"
DISCOVERY = CATSPACE / "incubator" / "analyst" / "discovery"
TESTS = ROOT / "tests"
LEAN = ROOT / "lean" / "ideas" / "Ideas"

# THE HAND TABLE (the `material` entry's worths, written here ONLY as the pattern the scan looks for -- the scan's own fixture, never read as a worth)
HAND = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
_KIND_TOKENS = {
    "p": r"(?:chess\.PAWN|\bPAWN|Role::Pawn|['\"][pP]['\"]|\.pawn)", "n": r"(?:chess\.KNIGHT|\bKNIGHT|Role::Knight|['\"][nN]['\"]|\.knight)",
    "b": r"(?:chess\.BISHOP|\bBISHOP|Role::Bishop|['\"][bB]['\"]|\.bishop)", "r": r"(?:chess\.ROOK|\bROOK|Role::Rook|['\"][rR]['\"]|\.rook)",
    "q": r"(?:chess\.QUEEN|\bQUEEN|Role::Queen|['\"][qQ]['\"]|\.queen)"}
_ARROW = r"\s*(?::|=>)\s*"                                    # a dict's ':' or a match arm's '=>' (Lean's '| .pawn => 1' too)
_NUM = r"(-?\d+(?:\.0+)?)\b"
_PAIR = {k: re.compile(tok + r"(?:\s*\|\s*Role::\w+)*" + _ARROW + _NUM) for k, tok in _KIND_TOKENS.items()}
# a match arm naming two kinds at once (`Role::Knight | Role::Bishop => 3`): each kind in the alternation gets the number
_ALT = re.compile(r"((?:Role::\w+\s*\|\s*)+Role::\w+)" + _ARROW + _NUM)
_ROLE_LETTER = {"Pawn": "p", "Knight": "n", "Bishop": "b", "Rook": "r", "Queen": "q", "King": "k"}
_LIST = re.compile(r"[\[(]\s*1(?:\.0)?\s*,\s*3(?:\.0)?\s*,\s*3(?:\.0)?\s*,\s*5(?:\.0)?\s*,\s*9(?:\.0)?\b")      # the worths in the kinds' order, pawn .. queen
# the hand table's numbers in a DERIVED worth role: the pawn's worth as a literal, the promotion's stake (queen less pawn), a threshold on a worth reading
_DERIVED = [
    ("PAWN", re.compile(r"is_en_passant\(\)\s*\{\s*1\s*\}")),                                       # Rust: an en-passant victim worth 1
    ("PAWN", re.compile(r"(?:val|value|kernel_value|worth|worth_of)\(r\)\s*-\s*1\b")),            # Rust: a promotion's gain, the new unit less 1
    ("PAWN", re.compile(r"VAL\w*\[[\w.]*promotion\]\s*-\s*1\b")),                                   # Python: the same
    ("PAWN", re.compile(r"^\s*victim\s*=\s*1\s*$")),                                                # Python: an en-passant victim worth 1
    ("STAKE", re.compile(r"\bSTAKE_PROMOTE\w*\s*(?::\s*\w+)?\s*=\s*8\b")),                          # the promotion's stake written as 8 (q 9 less p 1)
    ("STAKE", re.compile(r"\bv\s*\+\s*8\b")),                                                       # ... added to a target's worth
    ("STAKE", re.compile(r"is_promotion\(\)\s*\{\s*8(?:\.0)?\s*\}")),                               # ... a promotion's gain written as 8
    ("STAKE", re.compile(r"\b8(?:\.0)? if [\w.]*promo")),                                           # ... the same in Python
    ("STAKE", re.compile(r"claim\(\"promotion\",\s*8\b")),                                          # ... a promotion claim's value
    ("STAKE", re.compile(r"\balone\s*>=\s*8\b")),                                                   # ... "a promotion (8) or a queen"
    ("STAKE", re.compile(r"\b(?:QUEEN|PROMO|PROMOTE)\w*\s*(?::\s*\w+\s*)?=\s*8(?:\.0)?\b")),       # ... a constant holding it
    ("THRESHOLD", re.compile(r"(?:kernel_value|\bval|worth_of|rules::val)\([^()]*\)\s*(?:<|<=|>|>=)\s*[2-9]\b")),   # a worth reading against a literal
    ("THRESHOLD", re.compile(r"\|\((?:v, _|_, v)\)\|\s*\*v\s*(?:<|<=|>|>=)\s*[2-9]\b")),            # a filter of (worth, square) pairs against a literal
    ("THRESHOLD", re.compile(r"thr\[x\]\s*>=\s*[2-9]\b")),                                          # a threat's worth (the exchange it wins) against a literal
    ("THRESHOLD", re.compile(r"need_left\.clamp\(1,\s*3\)")),                                       # the need between the unit and a minor's worth, written
    ("THRESHOLD", re.compile(r"\bv\s*<\s*3\s*\|\|")),                                               # rules.rs trapped: a worth below 3
    ("THRESHOLD", re.compile(r"kernel_value\([^()]*\)\.max\(1\)")),                                 # a worth at least the unit, written as 1
    # (plan 76.5 (d), found by the mined run's crash) AN EXCHANGE READ AGAINST 0 OR 1 -- "they win something / nothing": the unit of a gain written as a literal; it
    # agrees with the store's `unit` only while the unit divides every worth (the hand table; the scale control), not under the mined entry (unit 2, 5 - 4 = 1)
    ("UNIT", re.compile(r"(?:_see|see_for|see_on|rules::see|G\.see|\bsee)\(.*?\)\s*(?:>|>=|<|<=|==|!=)\s*[01]\b")),
]
# the royal's place in an order of worths, written as a number ("the king counts 100": Lean Ideas.ExchangeLaw.valueAt)
_ROYAL = [re.compile(r"Role::King\s*\{\s*[1-9]\d*(?:\.0)?\s*\}"), re.compile(r"Role::King\s*=>\s*100"), re.compile(r"chess\.KING:\s*100\b"),
          re.compile(r"\b[1-9]\d+\s+if\s+[^\n]*KING\s+else\b"), re.compile(r"KING\s+else\s+[1-9]\d+"), re.compile(r"\.king then 100\b")]


def pairs_of_line(text: str) -> dict:
    """The (unit kind -> number) pairs one line of code writes (a dict entry, a match arm, a Lean match case)."""
    got = {}
    for k, rx in _PAIR.items():
        for m in rx.finditer(text):
            got.setdefault(k, float(m.group(1)))
    for m in _ALT.finditer(text):
        for role in re.findall(r"Role::(\w+)", m.group(1)):
            k = _ROLE_LETTER.get(role)
            if k in HAND:
                got.setdefault(k, float(m.group(2)))
    return got


def table_hit(pairs: dict) -> "str | None":
    """Two or more kinds mapped to their hand worths and none to another number: the kinds (a TABLE hit), else None (a mapping that gives some kind another
    number -- python-chess's piece-type codes, the vocabulary's order of kinds, a variant's scale -- is no copy of the table)."""
    same = sorted(k for k, v in pairs.items() if HAND.get(k) == v)
    other = [k for k, v in pairs.items() if k in HAND and HAND[k] != v]
    return "".join(same) if len(same) >= 2 and not other else None


def hits_of_line(text: str) -> list:
    """The worth-role hits of one line of code besides a table spread over lines: [(category, what)] -- TABLE (the kinds in their order as a list),
    PAWN / STAKE / THRESHOLD (a derived literal), ROYAL (the royal unit's place in a worth order as a number)."""
    out = []
    if _LIST.search(text):
        out.append(("TABLE", "list"))
    for cat, rx in _DERIVED:
        if rx.search(text):
            out.append((cat, rx.pattern[:40]))
    if any(rx.search(text) for rx in _ROYAL):
        out.append(("ROYAL", "royal sentinel"))
    return out


def hits_of_file(lines: list) -> list:
    """[(line number, category, text)] of a file's code lines: a TABLE is a RUN of consecutive lines writing kind -> number pairs (a dict over two lines, a
    Lean match case per line), reported at its first line; the other categories per line."""
    out = []
    run: list = []
    for n, text in lines + [(len(lines) + 1, "")]:
        p = pairs_of_line(text)
        if p:
            run.append((n, text, p))
            continue
        if run:
            merged = {}
            for _n, _t, q in run:
                for k, v in q.items():
                    merged.setdefault(k, v)
            t = table_hit(merged)
            if t:
                out.append((run[0][0], "TABLE", " / ".join(x[1].strip() for x in run)))
            run = []
    for n, text in lines:
        for cat, _what in hits_of_line(text):
            out.append((n, cat, text.strip()))
    return sorted(out)


def _code_lines(p: Path):
    """(line number, text) of a file's CODE: Python docstrings and comments, Rust and Lean comments blanked (a comment quoting the hand worths is no copy)."""
    t = p.read_text(errors="replace")
    if p.suffix == ".py":
        t = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', lambda m: "\n" * m.group(0).count("\n"), t)
        lines = [ln.split("#")[0] for ln in t.splitlines()]
    elif p.suffix == ".rs":
        lines = [re.sub(r"//.*$", "", ln) for ln in t.splitlines()]
    else:
        t = re.sub(r"/-(?:.|\n)*?-/", lambda m: "\n" * m.group(0).count("\n"), t)
        lines = [re.sub(r"--.*$", "", ln) for ln in t.splitlines()]
    return list(enumerate(lines, 1))


def _module_file(name: str) -> "Path | None":
    parts = name.split(".")
    p = ROOT.joinpath(*parts)
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    return None


def _module_level(tree) -> list:
    """The statements a module runs when it is imported: its body and the blocks of its module-level if / try / with -- never a function's or a class's body
    (an import inside a function is the tool's, run when the function is: discovery's tools reach the older analyst modules only that way)."""
    out, todo = [], list(tree.body)
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        out.append(n)
        for field in ("body", "orelse", "finalbody", "handlers"):
            todo.extend(getattr(n, field, None) or [])
    return out


def _imports_of(p: Path) -> set:
    """Every catspace module a file imports when it is imported (module level; relative imports resolved)."""
    try:
        tree = ast.parse(p.read_text())
    except SyntaxError:
        return set()
    pkg = ".".join(p.relative_to(ROOT).with_suffix("").parts[:-1]) if p.name != "__init__.py" else ".".join(p.relative_to(ROOT).parts[:-1])
    out = set()
    for node in _module_level(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith("catspace")}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
                base = ".".join(up + ([base] if base else []))
            if not base.startswith("catspace"):
                continue
            out.add(base)
            out |= {f"{base}.{a.name}" for a in node.names}
    return out


def engine_modules() -> set:
    """THE ENGINE'S PYTHON: every module of discovery/ (its archive/ aside) and every catspace module reachable from them by the imports a module runs when it
    is imported (the agenda's, the words', the regression's: lean_passer_v2_proto and what it imports); an import inside a function is a tool's (the play server's
    frame player, the surprise loop's walkers) and does not make its module the engine's."""
    todo = [p for p in DISCOVERY.glob("*.py")]
    seen: set = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        for name in _imports_of(p):
            f = _module_file(name)
            if f is not None and f not in seen:
                todo.append(f)
    return {p for p in seen if "archive" not in p.parts}


# THE FINDINGS: engine hits that stay, each with its reason (a path and a pattern of its line -> why it is not a copy of the valuation, or why it cannot be routed)
_ROYAL_NUMBER = ("the royal as a NUMBER where it enters arithmetic, a print or a learned feature -- not a worth of the table (the store's royal is worth 0: its loss is "
                 "the terminal); the terminal's stand-in (Lean Ideas.ExchangeLaw.valueAt: 'the king counts 100'), whose own definition is the terminal GOAL entry's "
                 "stake, which the core does not hold (a later plan's, with regress.rs STAKE_MATE and waiting.rs MATE_GOAL); ")
FINDINGS = [
    ("rust/analyst_core/src/lib.rs", r"let v = if p\.role == Role::King \{ 100 \} else \{ kernel_value\(p\.role\) \};",
     _ROYAL_NUMBER + "lib.rs `target_profile` returns it to its callers (the older analyst tools: goal_search, target_reasoner -- no discovery module)"),
    ("rust/analyst_core/src/plan.rs", r"fn val\(r: Role\) -> i64 \{ if r == Role::King \{ 1000 \}",
     _ROYAL_NUMBER + "plan.rs's planner (PlanState: the surprise loop's and the older play engine's, not the agenda's) ranks the king target first and prints it capped at MATE_GOAL"),
    ("rust/analyst_core/src/reasoning.rs", r"fn val\(r: Role\) -> i64 \{ if r == Role::King \{ 100 \}",
     _ROYAL_NUMBER + "reasoning.rs's ledger (discovery/ledger.py's `A.ledger`, a tool, not the agenda) adds and prints it"),
    ("rust/analyst_core/src/waiting.rs", r"if pc\.role == Role::King \{ 10 \}",
     _ROYAL_NUMBER + "waiting.rs `gate_features`: a TRAINED gate's input (the king encoded 10, above the queen; frame_gate.py fits the gate on these features) -- "
     "a learned model's encoding, changed only by retraining"),
]


BY_CATEGORY = {
    "UNIT": "AN EXCHANGE READ AGAINST 0 ('they win nothing / something'): the unit of a gain written as a literal -- it reads as the store's `unit` while the unit "
            "divides every worth (the hand table: the guard identical; the scale control, every worth and the unit doubled, moved no move of the 218), not under "
            "the mined entry (unit 2 -- its least worth -- while a bishop 5 for a knight 4 gains 1); not routed in 76.5: the hanging test and its parries "
            "(reasoned.py), which met the regression's goal (`>= $unit`) and crashed the mined run, are -- the rest is the next step if a valuation whose unit is "
            "not the greatest common divisor of its worths is adopted",
}
UNIT_FINDINGS_AT_76_5 = 18           # the ceiling the test holds: routing one lowers it; a new one fails the test


def finding_for(rel: str, text: str, cat: str = "") -> "str | None":
    for path, rx, why in FINDINGS:
        if rel == path and re.search(rx, text):
            return why
    return BY_CATEGORY.get(cat)


def scan(show_all: bool = False, log=print) -> dict:
    """THE SCAN (see the docstring). -> {scope: [(path, line, category, text, finding or None)]}."""
    eng = engine_modules()
    files = sorted(CRATE.rglob("*.rs")) + sorted(CATSPACE.rglob("*.py")) + sorted(TESTS.rglob("*.py")) + sorted(LEAN.rglob("*.lean"))
    out: dict = {"ENGINE": [], "OUTSIDE": [], "TESTS": [], "LEAN": []}
    for p in files:
        if "__pycache__" in p.parts or ".lake" in p.parts:
            continue
        rel = str(p.relative_to(ROOT))
        if p.suffix == ".rs" or p in eng:
            scope = "ENGINE"
        elif p.suffix == ".lean":
            scope = "LEAN"
        elif TESTS in p.parents:
            scope = "TESTS"
        else:
            scope = "OUTSIDE"
        if rel == "scripts/plan76_5.py" or rel == "tests/test_plan76_5.py":
            continue
        for n, cat, text in hits_of_file(_code_lines(p)):
            out[scope].append((rel, n, cat, text, finding_for(rel, text, cat) if scope == "ENGINE" else None))
    eng_hits = out["ENGINE"]
    bare = [h for h in eng_hits if h[4] is None]
    log(f"OBS plan76.5 scan: engine modules (discovery/ and what it imports) {len(eng)}; hits ENGINE {len(eng_hits)} (routed-or-explained: FINDINGS {len(eng_hits) - len(bare)}, "
        f"UNEXPLAINED {len(bare)}), OUTSIDE {len(out['OUTSIDE'])}, TESTS {len(out['TESTS'])}, LEAN {len(out['LEAN'])}")
    for rel, n, cat, text, why in eng_hits:
        log(f"  ENGINE {'FINDING' if why else 'HIT'} {rel}:{n} [{cat}] {text[:150]}" + (f"\n      -- {why}" if why else ""))
    for scope in ("TESTS", "LEAN"):
        log(f"  {scope} FINDINGS -- {WHY_SCOPE[scope]}:")
        for rel, n, cat, text, _w in out[scope]:
            log(f"  {scope} FINDING {rel}:{n} [{cat}] {text[:150]}")
    if show_all:
        for rel, n, cat, text, _w in out["OUTSIDE"]:
            log(f"  OUTSIDE FINDING {rel}:{n} [{cat}] {text[:150]}")
    else:
        by = Counter(str(Path(rel).parent) for rel, *_ in out["OUTSIDE"])
        log(f"  OUTSIDE FINDINGS by directory ({WHY_SCOPE['OUTSIDE']}; `--all` prints each): " + "; ".join(f"{d} {c}" for d, c in sorted(by.items())))
    return out


WHY_SCOPE = {
    "TESTS": "a test's oracle or fixture quoting the hand worths: it fixes the default valuation the reading it tests is compared with (a legacy module's own table "
             "in test_analyst_tactic_coverage)",
    "LEAN": "a VALUATION in Lean, not a rule (Ideas.Fork.pieceValue, read by Ideas.Base.Vocabulary.worthOf / materialOf; Ideas.ExchangeLaw.valueAt's king 100): the "
            "theorems are stated over these worths -- the OBLIGATION in the two VALUATION entries' notes (they hold for any table with the properties they use); no "
            "proof rewritten in 76.5",
    "OUTSIDE": "no engine module imports them when it is imported: the older analyst modules (plans <= 60), the play server's frame player, research/ (the ML era: "
               "its own components, whose src/ must not import the incubator) -- not routed in 76.5; their decisions are not the agenda's",
}


# ------------------------------------------------------------------------------------------------------------------------------------------ (d) the experiment

GOAL_FILES = ("exchange", "mate", "promotion", "win", "save")
MINED, HAND_NAME = "material (mined)", "material"


def _store_dir() -> Path:
    from catspace.incubator.analyst.discovery import lemma_store as LS
    return LS.DIR


def _saved_dir() -> Path:
    from catspace.io import paths
    d = Path(paths.experiment("plan76_5", "goal_files", "x")).parent
    d.mkdir(parents=True, exist_ok=True)
    return d


SCALED = "material (scaled)"          # the scale control's entry: written by `swap scaled`, removed by `swap hand`, never committed


def swap(to: str, log=print) -> None:
    """(d) THE EXPERIMENT'S SWITCH: the five GOAL entries count in `material (mined)` (to='mined': each file's bytes saved first, then its `params.valuation`
    rewritten) or back in `material` (to='hand': the saved bytes written back, byte for byte, and checked against git's copy). Never committed.
    to='scaled': THE SCALE CONTROL -- a temporary entry `material (scaled)`, the hand worths and the hand unit times two, and the GOAL entries pointed at it: if
    every reading of a worth is the table's, the scale moves the printed values and no move (the terminal's stake, a GOAL entry's number, is not scaled)."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    store, saved = _store_dir(), _saved_dir()
    if to in ("mined", "scaled"):
        name = MINED if to == "mined" else SCALED
        if to == "scaled":
            e = json.loads((store / "material.json").read_text())
            e.update(name=SCALED, note="THE SCALE CONTROL of plan 76.5 (scripts/plan76_5.py swap scaled; never committed): the hand worths and unit times two.",
                     implementation=["scripts/plan76_5.py:swap"])
            e["params"] = dict(e["params"], worth={k: 2 * w for k, w in e["params"]["worth"].items()}, unit=2 * e["params"]["unit"])
            (store / f"{LS.slug(SCALED)}.json").write_text(json.dumps(e, indent=1) + "\n")
        for n in GOAL_FILES:
            p = store / f"{n}.json"
            raw = p.read_bytes()
            e = json.loads(raw)
            assert e["params"]["valuation"] == HAND_NAME, (n, e["params"]["valuation"])
            (saved / f"{n}.json").write_bytes(raw)
            e["params"]["valuation"] = name
            p.write_text(json.dumps(e, indent=1) + "\n")
        log(f"OBS 76.5 experiment: the GOAL entries {', '.join(GOAL_FILES)} count in {name!r} (the originals saved in {saved})")
        return
    for n in GOAL_FILES:
        (store / f"{n}.json").write_bytes((saved / f"{n}.json").read_bytes())
    extra = store / f"{LS.slug(SCALED)}.json"
    if extra.exists():
        extra.unlink()
    files = [str((store / f"{n}.json").relative_to(ROOT)) for n in GOAL_FILES]
    same = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--", *files]).returncode == 0
    log(f"OBS 76.5 experiment: the GOAL entries restored byte for byte from {saved}; git diff on them: {'none' if same else 'DIFFERS'}")
    if not same:
        raise SystemExit("the GOAL files differ from git's copy after the restore")


def _decision(path: str) -> str:
    return next((ln for ln in open(path).read().splitlines() if ln.startswith("DECISION ")), "DECISION ?")


def _uci(fen: str, san: str) -> "str | None":
    import chess
    try:
        return chess.Board(fen).parse_san(san).uci()
    except (ValueError, AssertionError):
        return None


def decisions(hand: str, mined: str, stockfish: bool = False, log=print) -> list:
    """(d) every guard DECISION that differs between the hand snapshot and the mined one (ten + wide, tiers 0 and 2) -> [(run, id, fen, hand line, mined line,
    the moves' Stockfish readings)]. `stockfish`: depth 14 reads both moves (verdict.verdict: RIGHT within its tolerance) -- this process runs no agenda."""
    import importlib.util as IU
    spec = IU.spec_from_file_location("trace_identity", ROOT / "scripts" / "trace_identity.py")
    TI = IU.module_from_spec(spec)
    spec.loader.exec_module(TI)
    fens = {}
    for sets in ("ten", "wide"):
        for pid, fen, _us in TI._cases(sets):
            fens[pid] = fen
    VD = None
    if stockfish:
        from catspace.incubator.analyst.discovery import verdict as VD
    out = []
    for sub in ("ten_t0", "ten_t2", "wide_t0", "wide_t2"):
        d0, d1 = Path(hand) / sub, Path(mined) / sub
        if not d1.is_dir():
            continue
        names = sorted(f.name for f in d1.iterdir() if f.name.endswith(".txt") and f.name.count(".") == 1)
        moved = []
        for f in names:
            x, y = _decision(str(d0 / f)), _decision(str(d1 / f))
            if x == y:
                continue
            pid = f[:-4]
            fen = fens.get(pid, "?")
            mx, my = x.split()[1], y.split()[1]
            sf = None
            if VD is not None and fen != "?":
                sf = tuple(VD.verdict(fen, _uci(fen, m), 14) if m not in ("None", "?") else ("none", None, None) for m in (mx, my))
            moved.append((sub, pid, fen, x[9:], y[9:], sf))
        log(f"OBS 76.5 experiment {sub}: DECISION lines differ in {len(moved)} of {len(names)}; the MOVE differs in "
            f"{sum(1 for m in moved if m[3].split()[0] != m[4].split()[0])}")
        out += moved
    if VD is not None:
        VD.close()
    for sub, pid, fen, x, y, sf in out:
        s = "" if sf is None else f"  || Stockfish d14: hand {sf[0][0]} (loss {sf[0][1]}), mined {sf[1][0]} (loss {sf[1][1]}), best {sf[0][2] or sf[1][2]}"
        log(f"  DECISION {sub} {pid} [{fen}]: hand {x[:90]} | mined {y[:90]}{s}")
    if VD is not None:
        moves = [m for m in out if m[3].split()[0] != m[4].split()[0]]
        tally = Counter(f"{m[5][0][0]} -> {m[5][1][0]}" for m in moves)
        log("OBS 76.5 experiment the moved decisions by Stockfish (hand -> mined): " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    return out


def batch(tag: str, seconds: float = 15.0, sets=("rand14_10", "rand15_10", "rand16_10", "rand17_10"), log=print) -> str:
    """(d) THE LOOP-2 BATCH: `analyse --set S --seconds 15 --qlog --quiet` for each set (the workers' pool; the question log to a fresh scratch file through
    CATSPACE_QLOG), then `question_prior grade --log` in its OWN process (Stockfish depth 14). -> the log's path. One sitting per valuation."""
    import os
    import time
    from catspace.io import paths
    qlog = paths.experiment("plan76_5", f"qlog_{tag}.jsonl")
    if os.path.exists(qlog):
        os.remove(qlog)
    env = dict(os.environ, CATSPACE_QLOG=qlog, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT))
    t0 = time.monotonic()
    for s in sets:
        r = subprocess.run([sys.executable, "-m", "catspace.incubator.analyst.discovery.analyse", "--set", s, "--seconds", str(seconds), "--qlog", "--quiet"],
                           cwd=str(ROOT), env=env, capture_output=True, text=True)
        for ln in (r.stdout + r.stderr).splitlines():
            if ln.startswith("OBS set wall time") or "Traceback" in ln or "Error" in ln:
                log(f"  [{tag}] {ln}")
    wall = time.monotonic() - t0
    g = subprocess.run([sys.executable, "-m", "catspace.incubator.analyst.discovery.question_prior", "grade", "--log", qlog], cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
    for ln in g.stdout.splitlines():
        if ln.startswith("OBS"):
            log(f"OBS 76.5 batch {tag} (wall {wall:.0f} s): {ln[4:]}")
    return qlog


def batch_rows(qlog: str) -> dict:
    """The graded decisions of a batch log: {(set, id): (decision, sf, proved)} (the last row of each agenda carries the decision)."""
    rows = [json.loads(ln) for ln in open(qlog) if ln.strip()]
    out = {}
    for r in rows:
        out[(r.get("set"), r["id"])] = (r.get("decision"), r.get("sf"), r.get("proved"))
    return out


def batch_diff(hand: str, mined: str, log=print) -> None:
    """(d) the batch's decisions that differ between the hand and the mined sitting, with both grades."""
    h, m = batch_rows(hand), batch_rows(mined)
    diff = [(k, h.get(k), m.get(k)) for k in sorted(set(h) | set(m), key=str) if h.get(k) != m.get(k)]
    log(f"OBS 76.5 batch: {len(diff)} of {len(set(h) | set(m))} agendas differ (decision, grade or proof)")
    for k, x, y in diff:
        log(f"  BATCH {k[0]} {k[1]}: hand {x} | mined {y}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "scan":
        got = scan("--all" in a)
        sys.exit(1 if any(h[4] is None for h in got["ENGINE"]) else 0)
    if a and a[0] == "swap":
        swap(a[1])
        sys.exit(0)
    if a and a[0] == "decisions":
        decisions(a[1], a[2], "--stockfish" in a)
        sys.exit(0)
    if a and a[0] == "batch":
        batch(a[1], float(a[a.index("--seconds") + 1]) if "--seconds" in a else 15.0)
        sys.exit(0)
    if a and a[0] == "batch-diff":
        batch_diff(a[1], a[2])
        sys.exit(0)
    sys.exit(__doc__)
