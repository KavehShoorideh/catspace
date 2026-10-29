"""plan 76.2 THE RULE SHELF: the generated spec (rules/chess.gdl -> lean/ideas/Ideas/Spec/Chess.lean) is a certified model of the rule kernel for moves, states,
terminals and goals -- `legal` proved equal to the kernel's `legalMove` (Ideas/Spec/LegalEquiv.lean, with NextEquiv's next state); what is not proved stands as a
CONJECTURE entry of the store with its obstacle. (b) THE GAME LAYER READS THROUGH THE CERTIFIED KERNEL: every Game method that reads a rule (game.SPEC) is run
against the GDL interpreter as the ORACLE (gdl_chess.game_oracle) on the corners, every filed surprise and a seeded random play-out; the second copies the audit found
(the judge's check / capture read off the SAN, the exchange's principal lines in Python, the mate in one by python-chess) are routed to one implementation and the
replaced code is kept HERE as the test oracle."""
from __future__ import annotations

import importlib.util
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import chess
import pytest

import analyst_core as AC
from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import game as GM
from catspace.incubator.analyst.discovery import lemma_store as LS

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "ideas"
SEED, GAMES = 762, 1          # THE RANDOM SAMPLE (named): every position of GAMES uniformly random games from the initial position, seeded, each played to the
                              # kernel's terminal with its game record (no ply cap: the rules end every game) -- beside the corners and the filed surprises


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"plan762_{name}", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sample() -> list:
    """(fen, the game record before it or None): the corners, the filed surprises, the seeded play-outs."""
    cp = _script("crossplay")
    out = [(f, None) for f in cp.corner_fens()] + [(c["fen"], None) for c in GC.load_surprises()]
    rng = random.Random(SEED)
    for _g in range(GAMES):
        b, line = chess.Board(), []
        while True:
            before = GM.Record.of_line(GM.CHESS, [chess.Board(f) for f in line])
            out.append((b.fen(), before))
            if GM.CHESS.terminal(b, before) is not None:
                break
            line.append(b.fen())
            b.push(rng.choice(list(b.legal_moves)))
    return [(f, r) for f, r in out if chess.Board(f).is_valid()]

# the theorems of 76.2 (a): the moves, the end of the game
EQUIVALENCES = ("Ideas.LegalEquiv.legal_iff", "Ideas.LegalEquiv.legal_iff_reachable", "Ideas.LegalEquiv.legal_moveOf", "Ideas.LegalEquiv.attackBy_iff",
                "Ideas.LegalEquiv.check_iff", "Ideas.LegalEquiv.hasMove_iff", "Ideas.LegalEquiv.mate_iff", "Ideas.LegalEquiv.stalemate_iff",
                "Ideas.LegalEquiv.mate_iff_mated", "Ideas.LegalEquiv.stalemate_iff_stalemated", "Ideas.LegalEquiv.drawByFiftyMoves_iff",
                "Ideas.LegalEquiv.goal_win_iff", "Ideas.LegalEquiv.goal_loss_iff", "Ideas.LegalEquiv.terminal_sound", "Ideas.LegalEquiv.terminal_complete")


def _lake() -> str:
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake).exists():
        pytest.skip("no lake")
    return lake


def test_the_equivalences_are_indexed_theorems():
    """Every theorem 76.2 (a) states is a proved theorem of docs/LEAN_INDEX.md (the index rebuilt after the file)."""
    idx = LS.lean_index()
    missing = [t for t in EQUIVALENCES if idx.get(t) not in ("theorem", "lemma")]
    assert not missing, missing


def test_the_open_clause_is_a_conjecture():
    """What Lean does not prove of the terminal (the criterion's knight clause as a dead position) is a CONJECTURE entry naming its obstacle and the theorem it
    leaves open."""
    e = LS.load(force=True)["dead position (the knight clause)"]
    assert e["status"] == "CONJECTURE" and e["lean"] == []
    assert "lean/ideas/Ideas/Spec/LegalEquiv.lean:terminal_sound" in e["implementation"]
    assert "obstacle" in e["note"]


@pytest.mark.slow
def test_legal_equiv_builds_by_reasoning():
    """SLOW (a Lean build, ~15 s with the library built): LegalEquiv builds; no sorry, no native_decide, no decide tactic in it; the main theorems rest on
    the standard axioms only (no sorryAx, no Lean.ofReduceBool)."""
    lake = _lake()
    text = (LEAN / "Ideas" / "Spec" / "LegalEquiv.lean").read_text()
    body = text.split("-/", 1)[1]                                              # (the module docstring names the banned tactics)
    for banned in ("sorry", "native_decide", "by decide", "<;> decide", "interval_cases"):
        assert banned not in body, banned
    r = subprocess.run([lake, "build", "Ideas.Spec.LegalEquiv"], cwd=LEAN, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as fh:
        fh.write("import Ideas.Spec.LegalEquiv\n" + "".join(f"#print axioms {t}\n" for t in EQUIVALENCES))
    r = subprocess.run([lake, "env", "lean", fh.name], cwd=LEAN, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    lines = [ln for ln in r.stdout.splitlines() if "depends on axioms" in ln]
    assert len(lines) == len(EQUIVALENCES), r.stdout
    for ln in lines:
        assert "sorryAx" not in ln and "ofReduceBool" not in ln, ln


# ---------------------------------------------------------------------------------------------------------------- (b) the game layer through the certified kernel

def test_every_method_is_a_rule_or_says_why_not():
    """Every method of the Game protocol is either bound to the spec's relation it implements (game.SPEC, an ontology shelf name) or listed as reading no rule, with why."""
    protocol = {n for n in vars(GM.Game) if callable(getattr(GM.Game, n)) and (not n.startswith("_") or n == "pass_")}
    assert protocol == set(GM.SPEC) | set(GM.NOT_RULES), sorted(protocol ^ (set(GM.SPEC) | set(GM.NOT_RULES)))
    assert not set(GM.SPEC) & set(GM.NOT_RULES)
    from catspace.incubator.analyst.discovery import ontology as ON
    shelf = ON.load()
    assert all(rel in shelf for rel in GM.SPEC.values()), [r for r in GM.SPEC.values() if r not in shelf]


def test_the_surface_is_the_spec_on_the_sample():
    """THE ORACLE: on the corners, every filed surprise and the seeded play-out (each position with its game record), every Game method of game.SPEC answers as
    rules/chess.gdl does -- the legal moves, legality, the moves between two squares, the next state after every legal move, the side to move, the terminal, the
    goals, the repetition, check, the clock and its events, the units, the attacks, the king's step, the capture, what a move is."""
    sample = _sample()
    bad = []
    for fen, before in sample:
        bad += [(fen, m, d) for m, d in GC.game_oracle(GM.CHESS, chess.Board(fen), before)]
    print(f"OBS 76.2 (b) oracle test: positions {len(sample)} (the corners, the surprises, the play-out seed {SEED}, {GAMES} game(s)); disagreements {len(bad)}")
    assert not bad, bad[:5]


class _WrongSurface(GM.ChessGame):
    """THE NEGATIVE CONTROL: a surface that reads what a move is off the notation's marks (the mate's '#' forgotten) and loses the en-passant capture's event."""

    def reply_kind(self, s, m):
        san = s.san(m)
        return "check" if san.endswith("+") else "capture" if "x" in san else GM.QUIET

    def clock_events(self, s, m):
        return frozenset(e for e in super().clock_events(s, m) if not (e == "capture" and s.is_en_passant(m)))


def test_the_oracle_fires_on_a_wrong_surface():
    mate = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert "reply_kind" in {m for m, _d in GC.game_oracle(_WrongSurface(), mate)}                # Ra8# read as a quiet move
    ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    assert "clock_events" in {m for m, _d in GC.game_oracle(_WrongSurface(), ep)}               # exd6 e.p.: its capture event lost
    assert not GC.game_oracle(GM.CHESS, mate) and not GC.game_oracle(GM.CHESS, ep)


def test_every_filed_surprise_agrees_with_the_surface():
    """Every case of rules/surprises/ is read by the oracle of every Game method: none disagrees (a case is filed when found and kept forever)."""
    for case in GC.load_surprises():
        assert not GC.game_oracle(GM.CHESS, chess.Board(case["fen"])), case["file"]


# the second copies, routed to one: the replaced code is the oracle

def _py_principal_lines(P2) -> list:
    """THE PYTHON TWIN of the exchange's principal lines (agenda.py `_principal_captures` before plan 76.2 (b)): -> [(square, [(SAN, FEN after, last, credit)])]."""
    from catspace.incubator.analyst.discovery import names as NM
    from catspace.incubator.analyst.discovery import rboard as RB
    from catspace.incubator.analyst.discovery.words import CHESS_WORDS as W
    b_rb = RB.RBoard(P2)
    me = b_rb.turn
    best, sqs = 0, []
    for x in chess.SQUARES:
        p = b_rb.piece_at(x)
        if p is None or p.color == me or NM.royal(p) or not b_rb.is_attacked_by(me, x):
            continue
        v = W.base.see(P2, x)
        if v > best:
            best, sqs = v, [x]
        elif v == best and v > 0:
            sqs.append(x)
    lines = []
    for sq in sqs:
        c_rb, gains, sans, boards = b_rb, [], [], []
        while len(sans) < 12:
            caps = [m for m in c_rb.legal_moves if m.to_square == sq]
            if not caps or c_rb.piece_at(sq) is None:
                break
            m = min(caps, key=lambda m: (W.base.worth(c_rb.piece_at(m.from_square)), m.uci()))
            gains.append(W.base.worth(c_rb.piece_at(sq)))
            sans.append(c_rb.san(m))
            c_rb = c_rb.push(m)
            boards.append(c_rb)
        n = len(sans)
        f = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            f[i] = max(0, gains[i] - f[i + 1])
        taken = 0
        while taken < n and gains[taken] - f[taken + 1] > 0:
            taken += 1
        if taken:
            lines.append((sq, [(sans[i], boards[i].fen(), i == taken - 1, f[i + 1]) for i in range(taken) if i % 2 == 0]))
    return lines


def _twin_positions() -> list:
    """The guard's case positions, the corners, and every position one legal move after them."""
    ti = _script("trace_identity")
    roots = [fen for _pid, fen, _us in ti._cases()] + list(ti.CORNERS)
    out = []
    for f in roots:
        b = chess.Board(f)
        if not b.is_valid():
            continue
        out.append(b.fen())
        for m in b.legal_moves:
            c = b.copy(stack=False)
            c.push(m)
            if c.is_valid():
                out.append(c.fen())
    return list(dict.fromkeys(out))


def test_the_routed_second_copies_equal_their_oracles():
    """The exchange's principal lines are the Rust core's (`Pos.principal_lines`), the mate in one is the Rust word (`Pos.mates`), what a move is is the Game surface's
    (`Game.reply_kind`) -- each equal, on the guard's positions and one move after them, to the code it replaced: the Python swap list, python-chess's
    `graph.mates`, the SAN's marks."""
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    from catspace.incubator.analyst.discovery import graph as G
    from catspace.incubator.analyst.discovery.agenda import Agenda
    BV.install_terms()
    n = lines = mates = moves = 0
    for fen in _twin_positions():
        b = chess.Board(fen)
        P = AC.Pos(b.fen())
        n += 1
        rs = [(sq, [(san, rb.fen(), last, cr) for san, rb, last, cr in caps]) for sq, caps in Agenda._principal_captures(None, P)]
        py = _py_principal_lines(P)
        assert rs == py, fen
        lines += len(py)
        rm = sorted(m.uci() for m in Agenda._mating_moves(b))
        assert rm == sorted(m.uci() for m in b.legal_moves if b.gives_check(m) and G.mates(b, m)), fen
        mates += len(rm)
        for m in b.legal_moves:
            san = b.san(m)
            assert GM.CHESS.reply_kind(b, m) == ("check" if san.endswith(("+", "#")) else "capture" if "x" in san else GM.QUIET), (fen, san)
            moves += 1
    print(f"OBS 76.2 (b) twins test: positions {n}, principal lines {lines}, mating moves {mates}, moves {moves}: every one equal to its oracle")
    assert lines and mates and moves


# ---------------------------------------------------------------------------------------------------------------- (c) the rule shelf in the store

def test_the_rule_shelf_is_in_the_store():
    """Every RULE entry names its generated Spec definition (in the index), its spec relation and the Game methods implementing it (rule methods of game.SPEC); a
    PROVED one cites only proved theorems of the index; the report prints the shelf -- the RULE entries and the names table's words."""
    from catspace.incubator.analyst.discovery import names as NM
    store = LS.load(force=True)
    rules = {n: e for n, e in store.items() if e["kind"] == "RULE"}
    assert {"the rule: legal move", "the rule: next state", "the rule: checkmate", "the rule: stalemate", "the rule: the fifty-move draw",
            "the rule: threefold repetition", "the rule: the clock's events", "the rule: the end of the game", "the rule: win and loss"} <= set(rules)
    assert not LS.rule_problems(store) and not LS.citation_problems(store)
    assert rules["the rule: legal move"]["status"] == "PROVED" and "Ideas.LegalEquiv.legal_iff" in rules["the rule: legal move"]["lean"]
    assert rules["the rule: the end of the game"]["status"] == "CONJECTURE" and "obstacle" in rules["the rule: the end of the game"]["note"]
    lines = LS.rule_shelf_lines(store)
    print(lines[0])
    assert lines[0].startswith(f"OBS lemma store RULE shelf: {len(rules) + len(NM.NAMES)} rules")
    assert sum(ln.startswith("OBS rule ") and "(name)" not in ln for ln in lines) == len(rules)


def test_the_rust_terminal_is_the_surfaces():
    """The Rust core's one definition of the rules' ends (game.rs `terminal`, the readings `claimable_draw` and `drawn` beside it) gives Game.terminal's kind on the
    oracle's sample, the game record passed as the history."""
    n = 0
    for fen, before in _sample():
        b = chess.Board(fen)
        hist = [] if before is None else [f for f, c in _record_fens(before) for _ in range(c)]
        assert AC.game_terminal(b.fen(), hist) == GM.CHESS.terminal(b, before), fen
        n += 1
    assert n > 400


def _record_fens(before: "GM.Record") -> list:
    """A record's identities as FENs the Rust parser reads (an EPD with the clocks appended), with their counts."""
    return [(k + " 0 1", c) for k, c in before.counts.items()]


def test_the_technique_reads_the_draws_through_the_surface():
    """technique_question's draw (its second copy of the rules, plan 76.2 (c)) reads Game.clock along the line: a capture on the line resets the clock (the kernel's
    count), where the copy added the plies to the root's clock."""
    from catspace.incubator.analyst.discovery import technique_question as TQ
    s = TQ.Search(None, chess.WHITE, 0, frozenset(), 99)
    b = chess.Board("8/8/8/8/8/2k5/r7/R3K3 w - - 99 80")
    assert s.drawn(b, 0) is False
    b.push(chess.Move.from_uci("a1a2"))                                                      # Rxa2: a capture resets the clock
    assert GM.CHESS.clock(b) == 0 and s.drawn(b, 1) is False                                 # (the copy read 99 + 1 >= 100: a draw)
    b2 = chess.Board("8/8/8/8/8/2k5/8/R3K3 w - - 99 80")
    b2.push(chess.Move.from_uci("a1a3"))                                                     # a quiet move: the clock reaches 100
    assert s.drawn(b2, 1) is True
