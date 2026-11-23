"""PLAN 77.1 (d) (2) THE CERTIFY STAGE (Kaveh 2026-11-04: "extracting rules from GDL and writing in Rust is one part of the rules ingestion process. But it has to be kept to a
minimum, and the verification code should be part of the machinery").

THE LINE: the GDL text (rules/chess.gdl) is the rules. A FAST BINDING -- code that answers a spec question faster than the spec evaluator does -- is admitted only with a
CERTIFICATE the engine keeps: the binding names what it ingests and its ORACLE, and is checked against it. Four kinds, each with its oracle:
  the KERNELS (python-chess; the native core's shakmaty): the legal moves, the next state after every move, the terminal reasons and verdict, the goals (and the native core's
      draws) -- against the spec read by discovery/gdl.py (the comparison scripts/crossplay.py ran, moved here; the script is now its caller);
  the CLAIM READERS of the one definition of forcing (regress.definition_rows_pos): "terminal" (a mate in one: some legal move ends the game in mate) against the SPEC (the spec's
      legal moves, its next states, its terminal reasons); "exchange" (the exchange law on each unit of theirs we attack) against the REGRESSION on the spec (the win goal);
  a LEMMA'S DECLARED DOMAIN (its store entry's `needs`): the lemma never fires outside it -- against the lemma itself, asked without the gate.
(The composition `terminal_in` -- a mate in n -- is machinery over the kernel's legal moves, next state and checkmate: its certificate is the kernel's.)

THE CERTIFICATE (lemmas/certificates/<binding>.json, one per binding, written by `run`): what the binding ingests, its oracle, the corpus, the positions checked, the disagreements
(with the first examples), the date, and the KEY -- a hash of the files the answer depends on (the GDL text, the binding's sources): a changed file makes the certificate STALE
(`stale()`; the test fails until `run` is repeated). ONLINE, the search keeps checking the cheap readers on a priced share of the positions it reads (`online_due`, the fuse
ONLINE_SHARE: the share of the definition's reading time the checks may take); a disagreement is recorded (FOUND) and the reader is DEMOTED for the process: the definition
then reads its claims through the oracle (regress._claims_binding_py, the regression) -- slower, right.

  .venv/bin/python -m catspace.incubator.analyst.discovery.certify run [--jobs 11]     # every binding on its corpus; the certificates written; OBS lines
  .venv/bin/python -m catspace.incubator.analyst.discovery.certify status              # each certificate: fresh / stale / failing
"""
from __future__ import annotations

import datetime
import hashlib
import json
import random
import sys
import time
from collections import Counter

import chess

import analyst_core as AC
from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery.game import CHESS
from catspace.io import paths

DIR = LS.DIR / "certificates"
ROOT = paths.REPO_ROOT
CORE = ROOT / "rust" / "analyst_core" / "src"
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
LOCK = ROOT / "rust" / "analyst_core" / "Cargo.lock"                 # (the kernel crate's version: shakmaty)
SPEC_READER = [DISC / "gdl.py", DISC / "gdl_chess.py"]               # the oracle of a rule: the spec read by gdl.py through the chess binding
GOALS = [LS.DIR / "mate.json", LS.DIR / "win.json", LS.DIR / "material.json"]      # the store entries the readers' claims are valued by
SEED = 77
GAMES = 6                 # the play-outs of the corpus (each to the kernel's terminal: the rules end every game)
PUZZLE_STEP = 25          # every PUZZLE_STEP-th distinct puzzle position (the whole sets' order; a sample, named)
ENDGAMES = 300            # random endgames for the lemmas' domains (kings, pawns and one or two other units, seeded)
EXAMPLES = 5              # the disagreements kept with a certificate (the count is kept whole)
ONLINE_SHARE = 0.02       # the online checks may take this share of the definition's reading time (a named fuse)

# the rule corners (scripts/crossplay.py's RULE_CORNERS, moved: the draws, the clock at its limit, castling through an attacked square, a right lost to a capture)
RULE_CORNERS = [
    "8/8/8/4k3/8/8/8/KN6 w - - 0 1", "8/8/8/4k3/8/8/8/KB6 w - - 0 1", "8/8/8/4k3/8/8/8/KBB5 w - - 0 1", "8/8/8/4k3/8/8/8/KNN5 w - - 0 1",
    "8/8/8/4k3/8/8/2b5/KB6 w - - 0 1", "8/8/8/4k3/8/8/3b4/KB6 w - - 0 1", "8/8/8/4k3/8/8/8/K7 w - - 0 1", "8/8/8/4k3/8/8/2n5/KB6 w - - 0 1",
    "7k/8/8/8/8/8/8/K6R w - - 99 80", "7k/8/8/8/8/8/8/K6R w - - 100 80", "7k/5Q2/8/8/8/8/8/K7 w - - 99 80",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "r3k2r/8/8/8/8/5n2/8/R3K2R w KQkq - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "4k3/1P6/8/8/8/8/6p1/4K2R b K - 0 1",
]


# ------------------------------------------------------------------------------------------------------------------------------------------ the kernels against the spec

def _rust_state(q) -> frozenset:
    """The spec's state of a native position (its exact FEN, the raw e.p. square it keeps)."""
    return GC.state_of(chess.Board(q.fen_exact()), ep=q.ep_square())


def _fluents(diff) -> list:
    return sorted({rel.split(".", 1)[1] for rel, _a in diff})


def move_kind(b: chess.Board, uci: str) -> str:
    """The class of a move for a disagreement's signature (the kernel's names)."""
    m = chess.Move.from_uci(uci)
    p = b.piece_at(m.from_square)
    if p is None:
        return "none"
    if b.is_castling(m):
        return "castle"
    if b.is_en_passant(m):
        return "en passant"
    if m.promotion:
        return "promotion"
    return chess.piece_name(p.piece_type) + (" capture" if b.is_capture(m) else "")


def cover_move(cov: Counter, b: chess.Board, m: chess.Move, ns) -> None:
    """What a compared move covered (the coverage the cross-play prints)."""
    if b.is_castling(m):
        cov["castle"] += 1
    if b.is_en_passant(m):
        cov["en passant"] += 1
    if m.promotion:
        cov["promotion " + chess.piece_name(m.promotion)] += 1
    if b.is_capture(m):
        cov["capture"] += 1
    if any(rel == "true.ep_square" for rel, _a in ns):
        cov["e.p. square set"] += 1
    rights_before = len(chess.SquareSet(b.clean_castling_rights()))
    rights_after = sum(1 for rel, _a in ns if rel == "true.castling_right")
    if rights_after < rights_before:
        cov["castling right lost"] += 1
    clock = next((int(a[0]) for rel, a in ns if rel == "true.halfmove"), None)
    if clock == GC.CLOCK_LIMIT:
        cov["clock at its limit after the move"] += 1


def kernels_against_spec(fen: str, rust: bool = True, prog=None) -> dict:
    """The spec against the kernels at one position (plan 76.0 (b)'s comparison). -> {'n_moves', 'n_next', 'coverage': Counter, 'disagree': [(kernel, class, signature, detail)]}."""
    prog = prog or GC.program()
    b = chess.Board(fen)
    st = GC.state_of(b)
    r = GC.Reading(st, prog)
    out = []
    cov: Counter = Counter()
    k_legal = sorted(m.uci() for m in b.legal_moves)
    if r.legal != k_legal:
        spec_only, kern_only = sorted(set(r.legal) - set(k_legal)), sorted(set(k_legal) - set(r.legal))
        sig = "spec+" + ",".join(sorted({move_kind(b, u) for u in spec_only})) + " kernel+" + ",".join(sorted({move_kind(b, u) for u in kern_only}))
        out.append(("python", "legal", sig, f"spec only {spec_only}, kernel only {kern_only}"))
    n_next = 0
    for m in b.legal_moves:
        u = m.uci()
        if u not in r.legal:
            continue
        ns = GC.next_state(st, u, r.mover, prog)
        c = b.copy(stack=False)
        c.push(m)
        ks = GC.state_of(c)
        n_next += 1
        cover_move(cov, b, m, ns)
        if ns != ks:
            out.append(("python", "next", "+".join(_fluents(ns ^ ks)) + ":" + move_kind(b, u),
                        f"after {u}: spec only {sorted(GC.fact_text(f) for f in ns - ks)}, kernel only {sorted(GC.fact_text(f) for f in ks - ns)}"))
    k_reasons = GC.kernel_reasons(b)
    if r.reasons != k_reasons:
        out.append(("python", "terminal", f"spec {'+'.join(r.reasons) or '-'} kernel {'+'.join(k_reasons) or '-'}", f"spec {r.reasons}, kernel {k_reasons}"))
    kind = CHESS.terminal(b)
    spec_kind = next((x for x in GC.KERNEL_ORDER if x in r.reasons), None)
    if (GC.KERNEL_KIND.get(kind) if kind else None) != spec_kind or r.terminal != (kind is not None):
        out.append(("python", "verdict", f"spec {spec_kind} kernel {kind}", f"spec terminal {r.terminal} {spec_kind}, Game.terminal {kind}"))
    if r.goals != GC.goals_of(kind, r.mover):
        out.append(("python", "goal", f"spec {r.goals} kernel {GC.goals_of(kind, r.mover)}", ""))
    for x in r.reasons:
        cov["reason " + x] += 1
    if r.check:
        cov["check"] += 1
    if rust:
        out += _native_against_spec(b, st, r, prog, cov)
    return {"n_moves": len(k_legal), "n_next": n_next, "coverage": cov, "disagree": out}


def _native_against_spec(b: chess.Board, st, r, prog, cov) -> list:
    out = []
    try:
        p = AC.Pos(b.fen())
    except ValueError as x:
        return [("rust", "unparsable", "", str(x))]
    rl = sorted(p.legal_ucis())
    if rl != r.legal:
        spec_only, kern_only = sorted(set(r.legal) - set(rl)), sorted(set(rl) - set(r.legal))
        sig = "spec+" + ",".join(sorted({move_kind(b, u) for u in spec_only})) + " kernel+" + ",".join(sorted({move_kind(b, u) for u in kern_only}))
        out.append(("rust", "legal", sig, f"spec only {spec_only}, kernel only {kern_only}"))
    for u in rl:
        if u not in r.legal:
            continue
        ns = GC.next_state(st, u, r.mover, prog)
        ks = _rust_state(p.push(u))
        cov["rust next"] += 1
        if ns != ks:
            out.append(("rust", "next", "+".join(_fluents(ns ^ ks)) + ":" + move_kind(b, u),
                        f"after {u}: spec only {sorted(GC.fact_text(f) for f in ns - ks)}, kernel only {sorted(GC.fact_text(f) for f in ks - ns)}"))
    rm, rs = p.is_checkmate(), p.is_stalemate()
    if rm != ("mate" in r.reasons) or rs != ("stalemate" in r.reasons):
        out.append(("rust", "terminal", f"spec {'+'.join(x for x in r.reasons if x in ('mate', 'stalemate')) or '-'} kernel "
                    f"{'+'.join(x for x, y in (('mate', rm), ('stalemate', rs)) if y) or '-'}", ""))
    rdraw = dict(AC.game_draw_check(b.fen(), []))          # the native core's draws (game.rs `drawn`), read at the successors (no history)
    for u in rl:
        if u not in r.legal:
            continue
        ns = GC.next_state(st, u, r.mover, prog)
        exp = GC.expected_rust_draw(ns, prog)
        got = rdraw.get(u)
        if got == "stalemate" and exp is None:
            continue                                      # (a stalemate after the move: compared at the position itself)
        if got != exp:
            out.append(("rust", "draw", f"spec {exp} kernel {got}: {GC.material_signature(ns)}", f"after {u}: spec {exp}, Rust game_draw_check {got}"))
        cov["rust draw read"] += 1
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ the claim readers

def _reader_rows(b: chess.Board, stake: int, readers: list) -> list:
    """The claims binding on `b` (its side to move) by the native readers named (the definition's own call, no move read past the node's)."""
    from catspace.incubator.analyst.discovery import graph as G
    P = G.pos_of(b)
    if P is None:
        return None
    return AC.definition_rows_pos(P, stake, b.turn, readers, [])[0]


def terminal_reader_against_spec(fen: str, prog=None) -> list:
    """The "terminal" reader (a mate in one) against the spec: some legal move of the spec whose next state the spec reads as mate. -> [(class, signature, detail)]."""
    prog = prog or GC.program()
    b = chess.Board(fen)
    if b.is_game_over():
        return []
    rows = _reader_rows(b, 1, ["terminal"])
    if rows is None:
        return []
    fast = any(r == "terminal" for r, _sq, _v in rows)
    st = GC.state_of(b)
    rd = GC.Reading(st, prog)
    spec = next((u for u in rd.legal if "mate" in GC.Reading(GC.next_state(st, u, rd.mover, prog), prog).reasons), None)
    if fast != (spec is not None):
        return [("terminal", f"reader {fast} spec {spec is not None}", f"spec's mating move {spec}")]
    return []


def exchange_reader_against_regression(fen: str, stakes=(1, 3)) -> list:
    """The "exchange" reader against the regression on the spec (the win goal regressed on each unit of theirs, `regress._claims_binding_py`), at each stake."""
    from catspace.incubator.analyst.discovery import regress as R
    b = chess.Board(fen)
    if b.is_game_over():
        return []
    out = []
    word = dict(R._claim_readers()[0]).get("exchange")
    for stake in stakes:
        rows = _reader_rows(b, stake, ["exchange"])
        if rows is None:
            return []
        fast = {sq: v for r, sq, v in rows if r == "exchange"}
        oracle = {sq: v for (k, sq), v in R._claims_binding_py(b, b.turn, stake).items() if k == word}
        if fast != oracle:
            out.append(("exchange", f"stake {stake}", f"reader {sorted(fast.items())}, regression {sorted(oracle.items())}"))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ the lemmas' declared domains

def domains_against_lemmas(fen: str) -> list:
    """Each promotion lemma asked WITHOUT its gate where the board falls outside its declared domain (`needs`): it must not fire there. Both sides to move are read."""
    from catspace.incubator.analyst.discovery import regress as R
    out = []
    b0 = chess.Board(fen)
    for turn in (chess.WHITE, chess.BLACK):
        b = b0.copy(stack=False)
        b.turn = turn
        b.ep_square = None
        if not b.is_valid() or b.is_check():
            continue
        k = R._kinds(b, not turn)
        for name, mask, _key, read in R._promotion_lemmas():
            if k & ~mask and read(b):
                out.append((name, "fires outside its domain", f"{b.fen()}: their kinds {bin(k)}, domain {bin(mask)}"))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ the corpora

def puzzle_fens(step: int = PUZZLE_STEP) -> list:
    """Every `step`-th distinct position of the puzzle sets (each set's file, its order)."""
    out, seen = [], set()
    for f in sorted(paths.derived_dir().glob("ledger_puzzles_*.json")):
        d = json.load(open(f))
        for p in (d["results"] if isinstance(d, dict) else d):
            fen = p.get("fen") if isinstance(p, dict) else None
            if fen and fen not in seen:
                seen.add(fen)
                out.append(fen)
    return out[::step]


def playout_fens(games: int = GAMES, seed: int = SEED) -> list:
    """Every position of `games` uniformly random games from the initial position (seeded), each to the kernel's terminal."""
    rng = random.Random(seed)
    out = []
    for _g in range(games):
        b = chess.Board()
        while True:
            out.append(b.fen())
            if CHESS.terminal(b) is not None:
                break
            b.push(rng.choice(list(b.legal_moves)))
    return out


def endgame_fens(n: int = ENDGAMES, seed: int = SEED) -> list:
    """`n` random legal endgames (seeded): both kings, two to four pawns, and up to two other units of the kernel's kinds (the lemmas' domains are read on both sides of them)."""
    rng = random.Random(seed)
    others = [k for k in chess.PIECE_TYPES if k not in (chess.PAWN, chess.KING)]
    out = []
    while len(out) < n:
        b = chess.Board(None)
        sq = rng.sample(range(64), 8)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
        for s in sq[2:2 + rng.randint(2, 4)]:
            if 8 <= s < 56:
                b.set_piece_at(s, chess.Piece(chess.PAWN, rng.random() < 0.5))
        for s in sq[6:6 + rng.randint(0, 2)]:
            b.set_piece_at(s, chess.Piece(rng.choice(others), rng.random() < 0.5))
        b.turn = rng.random() < 0.5
        if b.is_valid() and not b.is_game_over():
            out.append(b.fen())
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ the bindings and their certificates

def _src(*names) -> list:
    return [str(p.relative_to(ROOT)) for p in names]


BINDINGS = {
    # name: (what it ingests, its oracle, the files its answers depend on, the corpus, the check)
    "kernel python-chess": ("the rules: legal moves, the next state, the terminal reasons and verdict, the goals", "rules/chess.gdl read by discovery/gdl.py",
                            _src(GC.SPEC, *SPEC_READER) + ["python-chess " + chess.__version__], "playouts+corners+puzzles",
                            lambda fen, prog: [(k, c, s, d) for k, c, s, d in kernels_against_spec(fen, rust=False, prog=prog)["disagree"]]),
    "kernel analyst_core": ("the rules: legal moves, the next state, checkmate, stalemate, the draws", "rules/chess.gdl read by discovery/gdl.py",
                            _src(GC.SPEC, *SPEC_READER, CORE / "pos.rs", CORE / "game.rs", CORE / "rules.rs", LOCK), "playouts+corners+puzzles",
                            lambda fen, prog: [x for x in kernels_against_spec(fen, rust=True, prog=prog)["disagree"] if x[0] == "rust"]),
    "reader terminal": ("the terminal goal's claim: a mate in one (the definition of forcing's reader)", "the spec: its legal moves, next states and terminal reasons",
                        _src(GC.SPEC, *SPEC_READER, CORE / "regress.rs", CORE / "pos.rs", LOCK, *GOALS), "playouts+puzzles",
                        lambda fen, prog: terminal_reader_against_spec(fen, prog)),
    "reader exchange": ("the gain goal's claim: the exchange law on each unit of theirs we attack (the definition of forcing's reader)",
                        "the regression on the spec (regression.py: the win goal, regress._claims_binding_py)",
                        _src(GC.SPEC, *SPEC_READER, DISC / "regression.py", CORE / "regress.rs", CORE / "pos.rs", CORE / "vocab.rs", CORE / "rules.rs", CORE / "valuation.rs", LOCK, *GOALS),
                        "playouts+puzzles",
                        lambda fen, prog: exchange_reader_against_regression(fen)),
    "domains of the promotion lemmas": ("each lemma's declared domain (its store entry's `needs`)", "the lemma itself, asked without the gate",
                                        _src(LS.DIR / "unstoppablepasser2_sound.json", LS.DIR / "ideas-race-race_sound.json",
                                             ROOT / "catspace" / "incubator" / "analyst" / "lean_passer_v2_proto.py",
                                             ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "race.py"), "endgames+puzzles",
                                        lambda fen, prog: domains_against_lemmas(fen)),
}


# plan 78.1: THE VARIANT KERNELS (rust/engine/game, shakmaty's variants behind the Game trait) against their GDL texts, BY CO-PLAY (cs_game::certify: seeded random
# games played side by side from the start, the mover, the legal move names, the terminal verdict and the outcome compared at every state). A variant's text is
# rules/chess.gdl itself, or a DELTA on it in rules/variants/ (cs_gdl::compose). A variant with no text yet plays on its kernel UNCERTIFIED (`uncertified()`).
ENGINE = ROOT / "rust" / "engine"
COPLAY_GAMES = 20         # the co-play's seeded games per variant (each to the kernel's terminal)
VARIANT_TEXTS = {"chess": None, "kingofthehill": ROOT / "rules" / "variants" / "kingofthehill.gdl", "3check": ROOT / "rules" / "variants" / "3check.gdl",
                 # plan 79.35: two games with another start (chess960, horde) and two with other rules (atomic, antichess)
                 "chess960": ROOT / "rules" / "variants" / "chess960.gdl", "horde": ROOT / "rules" / "variants" / "horde.gdl",
                 "atomic": ROOT / "rules" / "variants" / "atomic.gdl", "antichess": ROOT / "rules" / "variants" / "antichess.gdl"}
# plan 79.35: besides its own start, a game may be played from a list of positions (a data file under rules/variants, a position text a line): chess960 from several of its 960 starts and from corner positions of the
# two-unit move of the leader; the others from the corner positions of their rules. The description is given the same start as the kernel (rules::restart_delta).
VARIANT_STARTS = {v: ROOT / "rules" / "variants" / f"{v}.starts" for v in ("chess960", "horde", "atomic", "antichess")}
COPLAY_START_GAMES = 4    # the co-play's seeded games from each position of a game's list


def start_positions(variant: str) -> list:
    p = VARIANT_STARTS.get(variant)
    return [] if p is None else [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def coplay_check(variant: str) -> tuple:
    """The variant's kernel against its GDL text by co-play, from its own start and from each position of its list -> (states read, [disagreements])."""
    import cs_engine as CE
    base = GC.SPEC.read_text()
    p = VARIANT_TEXTS[variant]
    text = base if p is None else p.read_text()
    g, plies, states, bad = CE.certify_coplay(variant, text, COPLAY_GAMES, SEED, None if p is None else base)
    out = [("coplay", variant, d[:300]) for d in bad]
    for fen in start_positions(variant):
        g, plies, n, bad = CE.certify_coplay_from(variant, text, COPLAY_START_GAMES, SEED, None if p is None else base, fen)
        states += n
        out += [("coplay", f"{variant} from {fen}", d[:300]) for d in bad]
    return states, out


for _v, _p in VARIANT_TEXTS.items():
    BINDINGS[f"kernel {_v} (cs_game)"] = (f"the rules of {_v}: the mover, the legal moves, the next state (by its consequences), the terminal verdict, the outcome, the state's facts (the fact layer's base, plan 78.13)",
                                         "the GDL text " + ("rules/chess.gdl" if _p is None else f"rules/chess.gdl with the delta {_p.relative_to(ROOT)}") + ", read by cs_gdl",
                                         _src(GC.SPEC, *([] if _p is None else [_p]), *([VARIANT_STARTS[_v]] if _v in VARIANT_STARTS else []), ENGINE / "gdl" / "src" / "lib.rs", ENGINE / "game" / "src" / "kernel.rs",
                                              ENGINE / "game" / "src" / "gdl_game.rs", ENGINE / "game" / "src" / "certify.rs", ENGINE / "Cargo.lock"),
                                         f"coplay:{_v}", None)


def uncertified() -> list:
    """The variants the kernels play with no GDL text yet: they play, uncertified (plan 78.1)."""
    import cs_engine as CE
    return [v for v in CE.variants() if v not in VARIANT_TEXTS]


def key_of(name: str) -> str:
    """The hash of the files a binding's answers depend on (a version string is hashed as itself)."""
    h = hashlib.sha256()
    for s in BINDINGS[name][2]:
        p = ROOT / s
        h.update(s.encode())
        h.update(p.read_bytes() if p.is_file() else b"")
    return h.hexdigest()[:16]


def corpus(spec: str) -> tuple:
    """The corpus named by `spec` -> (its positions, how many invalid ones were left out: the kernels and the spec's readings assume a legal position -- python-chess
    `is_valid`, as scripts/crossplay.py left them out)."""
    parts = {"playouts": playout_fens, "corners": lambda: list(RULE_CORNERS), "puzzles": puzzle_fens, "endgames": endgame_fens}
    out, seen, invalid = [], set(), 0
    for p in spec.split("+"):
        for f in parts[p]():
            if f in seen:
                continue
            seen.add(f)
            if not chess.Board(f).is_valid():
                invalid += 1
                continue
            out.append(f)
    return out, invalid


def _check_chunk(args) -> tuple:
    name, fens = args
    prog = GC.program()
    check = BINDINGS[name][4]
    bad = []
    for f in fens:
        for d in check(f, prog):
            bad.append((f, *d))
    return len(fens), bad


def certify(name: str, jobs: int = 1) -> dict:
    """Check one binding on its corpus -> its certificate (not written)."""
    what, oracle, _srcs, corp, _check = BINDINGS[name]
    if corp.startswith("coplay:"):
        t = time.time()
        variant = corp.split(":", 1)[1]
        states, bad = coplay_check(variant)
        listed = len(start_positions(variant))
        what_played = f"{COPLAY_GAMES} seeded co-play games (seed {SEED})" + (f" and {COPLAY_START_GAMES} from each of {listed} listed positions" if listed else "")
        return {"binding": name, "ingests": what, "oracle": oracle, "corpus": what_played, "positions": states,
                "invalid_left_out": 0, "disagreements": len(bad), "examples": [list(map(str, x)) for x in bad[:EXAMPLES]], "key": key_of(name),
                "sources": BINDINGS[name][2], "date": datetime.date.today().isoformat(), "seconds": round(time.time() - t, 1)}
    fens, invalid = corpus(corp)
    t = time.time()
    chunks = [(name, fens[i::max(jobs, 1)]) for i in range(max(jobs, 1))]
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            res = pool.map(_check_chunk, chunks)
    else:
        res = [_check_chunk(c) for c in chunks]
    bad = [x for _n, b in res for x in b]
    return {"binding": name, "ingests": what, "oracle": oracle, "corpus": corp, "positions": sum(n for n, _b in res), "invalid_left_out": invalid, "disagreements": len(bad),
            "examples": [list(map(str, x)) for x in bad[:EXAMPLES]], "key": key_of(name), "sources": BINDINGS[name][2],
            "date": datetime.date.today().isoformat(), "seconds": round(time.time() - t, 1)}


def path_of(name: str):
    return DIR / (LS.slug(name) + ".json")


def write(cert: dict) -> None:
    DIR.mkdir(exist_ok=True)
    path_of(cert["binding"]).write_text(json.dumps(cert, indent=1, ensure_ascii=True) + "\n")


def load(name: str) -> "dict | None":
    p = path_of(name)
    return json.loads(p.read_text()) if p.exists() else None


def status() -> dict:
    """{binding: 'fresh' | 'stale' | 'failing' | 'missing'} -- a certificate is FRESH when it exists, its key is the current files' and it found no disagreement."""
    out = {}
    for name in BINDINGS:
        c = load(name)
        out[name] = "missing" if c is None else "failing" if c["disagreements"] else "stale" if c["key"] != key_of(name) else "fresh"
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ online: the search keeps checking

SPENT = {"reading": 0.0, "checks": 0.0, "positions": 0}
FOUND: list = []            # the disagreements found online: (binding, fen, detail)
DEMOTED: set = set()        # the readers demoted for the process (the definition reads their claims through the oracle)


def reset() -> None:
    SPENT.update(reading=0.0, checks=0.0, positions=0)
    FOUND.clear()
    DEMOTED.clear()


def online_due() -> bool:
    """A check is due while the checks have taken less than ONLINE_SHARE of the reading time they guard."""
    return SPENT["checks"] <= ONLINE_SHARE * SPENT["reading"]


def online_check(board: chess.Board, before_rows: list, stake: int) -> None:
    """The cheap readers checked on one node the definition read (its claims binding now, by the native readers): the exchange reader against the regression; the lemmas'
    domains. A disagreement is recorded and demotes the reader."""
    from catspace.incubator.analyst.discovery import regress as R
    t = time.perf_counter()
    try:
        word = dict(R._claim_readers()[0]).get("exchange")
        if word is not None and "exchange" not in DEMOTED:
            fast = {sq: v for r, sq, v in before_rows if r == "exchange"}
            oracle = {sq: v for (k, sq), v in R._claims_binding_py(board, board.turn, stake).items() if k == word}
            if fast != oracle:
                FOUND.append(("reader exchange", board.fen(), f"stake {stake}: reader {sorted(fast.items())}, regression {sorted(oracle.items())}"))
                DEMOTED.add("exchange")
        for name, _kind, detail in domains_against_lemmas(board.fen()):
            FOUND.append(("domain " + name, board.fen(), detail))
            DEMOTED.add("domain " + name)
    finally:
        SPENT["checks"] += time.perf_counter() - t
        SPENT["positions"] += 1


# ------------------------------------------------------------------------------------------------------------------------------------------ the command

def main(argv: list) -> int:
    if not argv or argv[0] not in ("run", "status"):
        print(__doc__)
        return 2
    if argv[0] == "status":
        for n, s in status().items():
            print(f"OBS certificate {n}: {s}")
        print(f"OBS uncertified variants (no GDL text yet: they play on their kernel): {', '.join(uncertified())}")
        return 0
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
    names = [a for a in argv[1:] if a in BINDINGS] or list(BINDINGS)
    bad = 0
    for n in names:
        c = certify(n, jobs)
        write(c)
        bad += c["disagreements"]
        print(f"OBS certificate {n}: positions {c['positions']} ({c['corpus']}; invalid left out {c['invalid_left_out']}), disagreements {c['disagreements']}, {c['seconds']} s; ingests {c['ingests']}; oracle {c['oracle']}")
        for e in c["examples"]:
            print("   FINDING", " | ".join(e)[:300])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
