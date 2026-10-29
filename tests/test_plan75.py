"""PLAN 75.2 THE MOVE (docs/ANALYST_PLAN.md 75, Kaveh 2026-10-27: "the goal is actually very generic: first of all, don't lose, and then win ... some chess vocab has to be
provided by us ... everything else should be derived from the basic rules"). (a) THE GOAL LADDER AND KINDS FROM THE STORE: the GOAL entries (mate, promotion, win, save) and
the material VALUATION load; the stake of mate is read from the store at run time (a patched store moves the agenda's stake); material has one definition (Game.measure,
BaseWords.worth, selfmine's table and the Rust kernel's balance agree with the entry); the outcome ladder (don't lose, then win) is the worklist's generic arithmetic and the
agenda's status reads it. Count budgets only."""
import ast
import copy
import json
import re
from pathlib import Path

import chess
import pytest

import analyst_core as AC
from catspace.incubator.analyst.discovery import game as GM
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import names as NM
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import selfmine as SM
from catspace.incubator.analyst.discovery import words as WD
from catspace.incubator.analyst.discovery import worklist as WL
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State

RA8 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"                 # Ra8# : a mate root on a8
PROMO = "8/4P3/8/8/8/8/k7/6K1 w - - 0 1"                  # e8=Q : a promote root one push away
POSITIONS = (RA8, PROMO, "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "8/8/8/K2pP2r/8/8/8/7k w - d6 0 2",
             "2r2rk1/R5pp/5p2/2q1p3/3pQ3/8/2P2PPP/R5K1 w - - 4 30")


CHESS_WORDS = r"\b(chess|analyst_core|python-chess|king|queen|rook|bishop|knight|pawn|piece|square|board|mate|check|capture|castl\w*|promot\w*|uci|san|fen|file|rank|" \
              r"KING|QUEEN|ROOK|BISHOP|KNIGHT|PAWN|[a-h][1-8])\b"          # (test_plan74's: the generic modules name no game object)


def _code(p: Path) -> str:
    """The module's code without its docstrings and comments (as tests/test_plan73_5b.py reads it)."""
    t = re.sub(r'"""(?:.|\n)*?"""', "", p.read_text())
    return "\n".join(ln.split("#")[0] for ln in t.splitlines())


def _entry(name: str) -> dict:
    return json.loads((LS.DIR / (LS.slug(name) + ".json")).read_text())


def _agenda(fen: str) -> Agenda:
    return Agenda(State(fen=fen, us="w" if chess.Board(fen).turn else "b", pieces=[]))


# ---------------------------------------------------------------------------------------------------------------- (a) the goal kinds and the valuation

def test_75a_goal_entries_load_with_their_roles_and_the_schema():
    store = LS.load(force=True)
    gk = LS.goal_kinds()
    print("OBS 75a goal kinds: " + ", ".join(f"{g.name} ({g.question}: {g.basis}, stake {g.stake})" for g in gk.by_question.values()) + f"; valuation {gk.valuation.name} {gk.valuation.worth}, unit {gk.valuation.unit}")
    assert {n for n, e in store.items() if e["kind"] == "GOAL"} >= {"mate", "promotion", "win", "save"}
    assert store["material"]["kind"] == "VALUATION"
    assert (gk.terminal.name, gk.promotion.name, gk.gain.name, gk.keep.name) == ("mate", "promotion", "win", "save")
    assert gk.terminal.basis == gk.promotion.basis == "RULE" and gk.gain.basis == gk.keep.basis == "VALUATION"
    assert gk.terminal.rule == "Game.terminal" and gk.promotion.rule == "Game.promotes"
    assert gk.terminal.ends_game and not gk.terminal.keep and gk.keep.keep and not gk.gain.keep
    assert gk.terminal.stake == _entry("mate")["params"]["stake"]                       # the stake is the entry's, read
    v = _entry("material")["params"]["worth"]
    assert gk.promotion.stake == v["q"] - v["p"] and gk.promotion.places == "q"          # the promotion's stake: the valuation's queen less its pawn
    assert gk.of("fork") is LS.NO_GOAL and not LS.NO_GOAL.ends_game and not LS.NO_GOAL.keep
    for n in ("mate", "promotion", "win", "save", "material"):
        assert LS.check_entry(store[n]) == [], n
    bad = copy.deepcopy(store["mate"])
    del bad["params"]["stake"]
    assert LS.check_entry(bad)                                                          # a GOAL without its stake breaks the schema
    bad = copy.deepcopy(store["material"])
    bad["params"]["royal"] = "x"
    assert LS.check_entry(bad)
    two = dict(store)
    two["mate twice"] = dict(copy.deepcopy(store["mate"]), name="mate twice")
    two["mate twice"]["params"]["question"] = "mate again"
    with pytest.raises(LS.LemmaStoreError):
        LS.GoalKinds(two)                                                               # exactly one goal ends the game


def test_75a_the_stake_of_mate_is_read_from_the_store(monkeypatch):
    """The agenda reads the stake at run time: a patched store (mate's stake 123, the queen worth 7) moves the agenda's terminal stake, its mate roots and its promotion stake."""
    ag = _agenda(RA8)
    mates = [q for q in ag.all if q.goal == "mate"]
    assert mates and all(q.stake == LS.goal_kinds().terminal.stake for q in mates) and ag.end_stake == R.STAKE_MATE == BD.MATE_VALUE == GM.MATE_VALUE
    patched = copy.deepcopy(LS.load())
    patched["mate"]["params"]["stake"] = 123
    patched["material"]["params"]["worth"]["q"] = 7
    monkeypatch.setattr(LS, "_STORE", [patched])                                       # (the loaded store: every read goes through it)
    ag2 = _agenda(RA8)
    assert ag2.end_stake == 123 and all(q.stake == 123 for q in ag2.all if q.goal == "mate")
    ag3 = _agenda(PROMO)
    assert ag3.goals.promotion.stake == 6 and [q.stake for q in ag3.all if q.goal == "promote"] == [6]
    print(f"OBS 75a patched store: the terminal stake {ag2.end_stake}, the promotion stake {ag3.goals.promotion.stake} (queen 7 less pawn 1)")


def test_75a_material_has_one_definition():
    v = LS.goal_kinds().valuation
    assert SM.PIECE_WORTH == v.worth == G.WORTH
    base = WD.CHESS_WORDS.base
    for sym in "pnbrqk":
        for colour in (chess.WHITE, chess.BLACK):
            pc = chess.Piece.from_symbol(sym if colour == chess.BLACK else sym.upper())
            assert base.worth(pc) == v.worth_of(pc) == G.worth(pc) == _entry("material")["params"]["worth"][sym]
    for fen in POSITIONS:
        b = chess.Board(fen)
        for side in (chess.WHITE, chess.BLACK):
            own = sum(v.worth_of(p) * (1 if p.color == side else -1) for p in b.piece_map().values())
            assert GM.CHESS.measure(b, side) == own
            assert AC.Pos(b.fen()).balance(side) - 64 == own                                 # the Rust kernel's copy of the worths agrees (a FINDING: rules.rs holds its own table)


def test_75a_the_outcome_ladder_is_the_worklists_and_the_status_reads_it():
    assert WL.ladder(0, 0, 1) == dict(balance=0, threat=0, s0=0, word=WL.LEVEL, line=0)
    assert WL.ladder(9, 0, 1)["word"] == WL.WINNING and WL.ladder(9, 0, 1)["line"] == 1
    assert WL.ladder(-1, 9, 1)["word"] == WL.LOSING and WL.ladder(-1, 9, 1)["s0"] == -10
    st = WL.ladder(-9, 0, 1)
    assert not WL.kept(st, None) and not WL.kept(st, 8) and WL.kept(st, 9) and WL.Worklist.need(st) == 9
    assert WL.raises(st, 9, 1) and not WL.raises(WL.ladder(9, 0, 1), -8, 1)              # level reached, not yet ahead: the line may rise
    up = WL.raised(WL.ladder(0, 0, 1), 1)
    assert up["line"] == 1 and up["raised"] and WL.line_name(up) == "stay ahead" and WL.line_name(WL.ladder(0, 0, 1)) == "reach level"
    assert WL.rung(-1, 1) == WL.LOSING and WL.rung(0, 1) == WL.LEVEL and WL.rung(1, 1) == WL.WINNING
    ag = _agenda("4k3/8/8/8/8/8/8/3qK3 w - - 0 1")
    st = ag.status()
    assert {k: st[k] for k in ("balance", "threat", "s0", "word", "line")} == WL.ladder(st["balance"], st["threat"], ag.unit)


# ---------------------------------------------------------------------------------------------------------------- (b) the stops

STOPS = {"root": ("budget",), "chase": ("chase fuse", "the chase clock", "a second check of theirs in a row", "tempo arithmetic", "the walk of checks", "the chase's follow list"),
         "judge": ("cut by the judge",)}


def _patched(monkeypatch, name: str, **kw) -> None:
    store = copy.deepcopy(LS.load())
    store[name]["params"].update(kw)
    monkeypatch.setattr(LS, "_STORE", [store])


def test_75b_the_stops_are_store_entries_with_their_shapes():
    store = LS.load(force=True)
    for ctx, names in STOPS.items():
        got = LS.stops_for(ctx)
        for n in names:
            assert store[n]["kind"] == "STOP" and store[n]["status"] == "HEURISTIC" and n in got, n
    print("OBS 75b stops: " + "; ".join(f"{n} {dict((k, v) for k, v in store[n]['params'].items() if k != 'context')}" for names in STOPS.values() for n in names))
    bad = copy.deepcopy(store["budget"])
    bad["params"]["context"] = "nowhere"
    assert LS.check_entry(bad)                                                          # a stop says where it applies


def test_75b_a_stops_parameter_changed_moves_the_stop(monkeypatch):
    """The agenda reads the stops' shapes at run time: the question budget per unit, the chase clock's cap from its tier, the follow list's power below the root."""
    ag = _agenda(RA8)
    units, caps = 2, sum(1 for m in ag.board.legal_moves if ag.board.is_capture(m))
    assert ag.budget == units * 8 + caps
    deep = Agenda(State(fen=RA8, us="w", pieces=[]), tier=3)
    assert deep._clock() == min(4, 4)
    win = next(q for q in ag.all if q.goal == "win")
    ag.chase_depth = 2
    assert not ag._can_chase(win)                                                       # a win question's power stops below the first level
    _patched(monkeypatch, "budget", per_unit=3)
    assert _agenda(RA8).budget == units * 3 + caps
    _patched(monkeypatch, "the chase clock", at_most=2)
    assert Agenda(State(fen=RA8, us="w", pieces=[]), tier=3)._clock() == 2
    _patched(monkeypatch, "the chase's follow list", power_depth={"promote": 1})
    ag2 = _agenda(RA8)
    ag2.chase_depth = 2
    assert ag2._can_chase(next(q for q in ag2.all if q.goal == "win"))
    print(f"OBS 75b patched stops: budget {ag.budget} -> {units * 3 + caps} (per unit 8 -> 3), the tier-3 clock 4 -> 2, a win's power at depth 2 off -> on")


# ---------------------------------------------------------------------------------------------------------------- (c) the names table and the words' fields

def test_75c_every_name_is_bound_to_the_kernel_and_a_present_lean_twin():
    idx = LS.lean_index()
    assert NM.problems(idx) == []
    supplied = {"pawn", "knight", "bishop", "rook", "queen", "king", "square", "rank", "file", "attack", "move", "slide", "capture", "check", "mate", "stalemate", "draw", "win", "lose",
                "tempo"}
    assert supplied <= set(NM.NAMES)
    for n in NM.NAMES.values():
        assert n.lean is not None and idx.get(n.lean) is not None, n.word                # every twin is present in docs/LEAN_INDEX.md
    assert set(LS.BASE) == set(NM.READS) and all(w in NM.NAMES for w in NM.READS.values())      # the store's reads are names of the table
    for e in LS.load().values():
        assert set(e["reads"]) <= set(NM.READS), e["name"]
    assert all(NM.kind(k) == v for k, v in NM.KIND.items()) and NM.royal(chess.Piece.from_symbol("k")) and NM.pawn(chess.Piece.from_symbol("P"))
    assert set(LS.goal_kinds().valuation.worth) == set(NM.KIND)                          # the valuation prices exactly the named unit kinds
    print(f"OBS 75c names: {len(NM.NAMES)} names, every one with a kernel binding and a Lean twin present in the index")


def test_75c_the_words_read_their_entries_fields():
    forced = LS.forced_words()
    assert {"check", "capture", "threat of mate"} <= set(forced)
    for w in ("check", "check that also threatens +3", "capture 5", "threat of mate", "threat of mate in two"):
        assert w.startswith(forced), w
    for w in ("threat 3", "delayed threat: after every reply ...", "no safe move: ...", "push: their king is no stopper", "evade"):
        assert not w.startswith(forced), w
    assert [LS.forcing_rank(w) for w in ("check", "capture 3", "check that mates", "threat 5")] == [0, 1, 2, 2]
    assert LS.walked("check") and not LS.walked("check that mates") and not LS.walked("capture 3")
    assert LS.head_says("chase: step_after+step_least over delayed_balance_alive_d1, mate_in_two", "reads_every_reply") and LS.head_says("mate_in_one", "reads_every_reply")
    assert not LS.head_says("unstoppablePasser2_sound", "reads_every_reply") and LS.head_says("delayed_root_alive_d1", "leaf_read") and not LS.head_says("delayed_root_alive_d2", "leaf_read")
    assert LS.head_says("delayed_balance_alive_d2", "window") and LS.says("chase: step_after+step_least over repetition", "draw")


# ---------------------------------------------------------------------------------------------------------------- (d) the rest: kernel reads, the store's fields

def test_75d_the_castling_rook_is_read_off_the_kernel():
    """A castling mate root places the rook where the kernel's castling puts it (75.2 (d): no rook squares written in the agenda)."""
    for fen, want in (("5k2/8/8/8/8/8/8/4K2R w K - 0 1", ("R", chess.F1, chess.H1)), ("4k2r/8/8/8/8/8/8/5K2 b k - 0 1", ("r", chess.F8, chess.H8))):
        ag = _agenda(fen)
        mores = [q.pattern["more"] for q in ag.all if q.goal == "mate" and q.pattern.get("more")]
        assert mores and all(tuple(m[0]) == want for m in mores), (fen, mores)


def test_75d_the_words_and_shapes_of_the_rest_are_the_entries(monkeypatch):
    """The kill words that refute, the words aimed at their king, the shape key's order, the measure's units and the stopper arithmetic's move in hand are store fields; a patched
    field moves the agenda's reading."""
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    assert Agenda._refuted("after Rxd5 they mate us") and Agenda._refuted("stalemate") and Agenda._refuted("route Qd5: lost a piece (-3)")
    assert not Agenda._refuted("the promotion is not claimed") and not Agenda._refuted("no claim of ours and no material gain (+0)")
    b = chess.Board(RA8)
    m = chess.Move.from_uci("a1a8")
    for why in ("check", "threat of mate", "threat of mate in two", "capture 3", "threat 3"):
        th = R.threat_of(b, m, why)
        assert LS.at_king(why) == (th.kind in ("mate", "mate2", "check")), why          # the king's zone: exactly the words whose threat is on their royal unit
    sh = dict(mate=False, loss=0, hanging=["e4"], checks=[("Qh5+", ["e4"]), ("Bb5+", [])], guarded=3, poking=2)
    assert PA.shape_key(sh) == (0, 1, 1, -3, -2) and PA.shape_key(dict(sh, mate=True)) == (-1,)
    assert LS.params_of("reasoned: the measure they hold")["units"] == ["R"] and LS.params_of("no stopper in time")["ahead"] == 1
    store = copy.deepcopy(LS.load())
    store["the shape key"]["params"]["order"] = [["poking", -1], ["hanging", 1]]
    store["they mate us"]["params"]["refutes"] = False
    monkeypatch.setattr(LS, "_STORE", [store])
    assert PA.shape_key(sh) == (-2, 1)
    assert not Agenda._refuted("after Rxd5 they mate us") and Agenda._refuted("stalemate")
    bad = copy.deepcopy(store["the shape key"])
    bad["params"]["order"] = [["poking", 2]]
    assert LS.check_entry(bad)                                                          # a component's sign is 1 or -1
    print("OBS 75d store fields: refutes, at_king, the shape key's order, the measure's units, the stopper's move in hand -- a patched field moves the reading")


AGENDA_CORE = {                                                                           # 75.2 (d)'s generic test: the agenda's functions that must name nothing outside names.py and the store
    "selection": ("_push", "_wl_push", "open_questions", "_wl_status", "_wl_price", "_wl_said", "_graph_order", "_ask_loop"),
    "the status": ("status", "keeps", "status_line", "_keep_goal", "_target"),
    "chase expansion": ("_chase_routes", "_local_next", "_inner"),
    "the judge": ("_judge_on", "_spn", "_judge_stake", "_whole_share", "_decides", "_price", "_route_need", "_reply_kind", "_kind_spn", "_low_rate", "_priced", "_judge_price",
                  "_kind_tally", "_competitor", "_judge_begin", "_judge_resume", "_judge_end", "_judge_node"),
}
KNOWLEDGE_NAMES = ("STAKE_MATE", "STAKE_PROMOTE", "MATE_VALUE", "PIECE_WORTH", "PROMOTE_MAX_PUSHES", "FORCED_WORDS", "RULE2_FROM", "CHESS_WORDS")


def _named_outside(fn: "ast.FunctionDef", words: set) -> list:
    """What a function names outside names.py and the store: a kernel constant spelled here (chess.KING, chess.BB_*), a module constant holding a stake, a worth or a word list,
    the base word `worth`, or a game word (a goal, a unit kind, a forcing word) as a string the code compares, matches or builds a question or a claim with."""
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "chess" and n.attr.isupper():
            out.append(f"chess.{n.attr}")
        elif isinstance(n, (ast.Attribute, ast.Name)) and (n.attr if isinstance(n, ast.Attribute) else n.id) in KNOWLEDGE_NAMES:
            out.append(n.attr if isinstance(n, ast.Attribute) else n.id)
        elif isinstance(n, ast.Attribute) and n.attr == "worth" and isinstance(n.value, ast.Attribute) and n.value.attr == "base":
            out.append("W.base.worth")
        elif isinstance(n, ast.Compare):
            for c in [n.left] + list(n.comparators):
                for k in (c.elts if isinstance(c, (ast.Tuple, ast.List, ast.Set)) else [c]):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value in words:
                        out.append(repr(k.value))
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in ("startswith", "endswith", "new_question", "Claim"):
                for a in n.args[:1]:
                    for k in (a.elts if isinstance(a, ast.Tuple) else [a]):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str) and (k.value in words or name in ("startswith", "endswith") and any(k.value.startswith(w) for w in words)):
                            out.append(f"{name}({k.value!r})")
    return out


def _game_words() -> set:
    gk = LS.goal_kinds()
    return set(NM.NAMES) | {g.question for g in gk.by_question.values()} | set(LS.forced_words()) | {"fork", "technique", "draw", "save", "promote", "checkers", "checker", "promoter"}


# the core functions the generic test covers after 75.2 (c): no game name in their code and no CHESS FACT row in the recount (docs/AGENDA_AUDIT.md's 75.2 recount lists the rest
# with their rows: _push, _ask_loop, status, _keep_goal, _target, _chase_routes, _inner, _competitor, _judge_resume)
COVERED = {"_wl_push", "open_questions", "_wl_status", "_wl_price", "_wl_said", "_graph_order", "keeps", "status_line", "_local_next", "_judge_on", "_spn", "_judge_stake",
           "_whole_share", "_decides", "_price", "_route_need", "_reply_kind", "_kind_spn", "_low_rate", "_priced", "_judge_price", "_kind_tally", "_judge_begin", "_judge_end",
           "_judge_node"}


def test_75d_the_generic_modules_and_the_agendas_core_name_nothing_outside_the_names_and_the_store():
    """75.2 (d) THE GENERIC TEST: worklist.py and reasoning.py name no game object (test_plan74's words), and the agenda's selection, status, chase-expansion and judge functions name
    nothing outside names.py and the store. The functions not yet coverable are printed with what they still name."""
    from catspace.incubator.analyst.discovery import agenda as AG
    root = Path(AG.__file__).parent
    for f in ("worklist.py", "reasoning.py"):
        code = _code(root / f)
        hit = re.search(CHESS_WORDS, code, flags=re.I)
        assert hit is None, (f, hit.group(0) if hit else None)
    tree = ast.parse((root / "agenda.py").read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    words = _game_words()
    covered, open_ = [], {}
    for group, names in AGENDA_CORE.items():
        for nm in names:
            got = _named_outside(fns[nm], words)
            (open_.__setitem__(nm, got) if got else covered.append(nm))
    print(f"OBS 75d generic test: covered {len(covered)} of {sum(len(v) for v in AGENDA_CORE.values())} core functions: {', '.join(covered)}")
    for nm, got in open_.items():
        print(f"OBS 75d not yet coverable {nm}: {len(got)} -- {', '.join(sorted(set(got)))}")
    assert set(covered) >= COVERED, sorted(COVERED - set(covered))

