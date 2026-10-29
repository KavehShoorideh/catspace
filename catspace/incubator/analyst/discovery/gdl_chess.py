"""plan 76.0 (b) THE CHESS BINDING: the fast kernel's state as the spec's facts, the spec's moves as the kernel's notation -- the one place where the GDL interpreter
(`gdl.py`, which knows no game) meets python-chess. Everything here is a reading of rules/chess.gdl's state vocabulary:

  (cell ?f ?r ?c ?k)          python-chess `Board.piece_map()` (files a..h, ranks 1..8, colours white / black, kinds by `chess.piece_name`)
  (control ?c)                `Board.turn`
  (castling_right ?c ?side)   `Board.clean_castling_rights()` (the rook squares h1 / a1 / h8 / a8 -> kingside / queenside)
  (ep_square ?f ?r)           `Board.ep_square` -- the RAW square (set after every double push, legal capture or not: what the spec's `next` sets)
  (halfmove ?n)               min(`Board.halfmove_clock`, 100) -- the spec's clock saturates at its limit (the rule reads only "100 or more")
  (record ?n)                 plan 76.1 (b) (5), an INPUT: min(how often the game record holds the position's identity before it, 2) -- game.py `Record.count` of
                              `Game.position_key` (the kernel's identity: a LEGAL e.p. square only); 0 with no record (a puzzle)

and the moves: (move f1 r1 f2 r2) / (promote f1 r1 f2 r2 k) are the UCI text (castling the king's two-file move, en passant the pawn's diagonal move). The fullmove
number is no rule: it is not in the state.
"""
from __future__ import annotations

from functools import lru_cache

import chess

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.io import paths

SPEC = paths.REPO_ROOT / "rules" / "chess.gdl"          # the spec (a tracked file of the repo, not data)
CLOCK_LIMIT = 100
COLOUR = {chess.WHITE: "white", chess.BLACK: "black"}
RIGHT = {chess.H1: ("white", "kingside"), chess.A1: ("white", "queenside"), chess.H8: ("black", "kingside"), chess.A8: ("black", "queenside")}
PROMO = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}
PROMO_LETTER = {v: k for k, v in PROMO.items()}


@lru_cache(maxsize=1)
def program() -> GDL.Program:
    """rules/chess.gdl, loaded once (its static relations evaluated)."""
    return GDL.load(SPEC)


def sq(s: int) -> tuple:
    return (chess.FILE_NAMES[chess.square_file(s)], str(chess.square_rank(s) + 1))


def state_of(board: chess.Board, ep: "int | None | str" = "raw") -> frozenset:
    """The spec's state for a python-chess board. `ep`: 'raw' -- `board.ep_square`; or an explicit square / None (another kernel's raw square)."""
    facts = set()
    for s, p in board.piece_map().items():
        f, r = sq(s)
        facts.add(("true.cell", (f, r, COLOUR[p.color], chess.piece_name(p.piece_type))))
    facts.add(("true.control", (COLOUR[board.turn],)))
    for s in chess.SquareSet(board.clean_castling_rights()):
        c, side = RIGHT[s]
        facts.add(("true.castling_right", (c, side)))
    e = board.ep_square if ep == "raw" else ep
    if e is not None:
        facts.add(("true.ep_square", sq(e)))
    facts.add(("true.halfmove", (str(min(board.halfmove_clock, CLOCK_LIMIT)),)))
    return frozenset(facts)


def move_term(uci: str) -> tuple:
    f1, r1, f2, r2 = uci[0], uci[1], uci[2], uci[3]
    if len(uci) == 5:
        return ("promote", f1, r1, f2, r2, PROMO[uci[4]])
    return ("move", f1, r1, f2, r2)


def uci_of(term) -> str:
    if term[0] == "promote":
        return "".join(term[1:5]) + PROMO_LETTER[term[5]]
    if term[0] == "move":
        return "".join(term[1:5])
    raise ValueError(f"not a move term: {term}")


def fact_text(f) -> str:
    rel, args = f
    name = rel.split(".", 1)[1] if rel.startswith("true.") else rel
    return "(" + " ".join((name,) + tuple(GDL.term_text(a) for a in args)) + ")"


REPETITION_LIMIT = 2                                  # the record's count saturates where the rule stops reading it (the third occurrence: twice before)


def inputs_of(board: "chess.Board | None" = None, before=None) -> tuple:
    """plan 76.1 (b) (5): the INPUT facts a turn supplies besides `does` -- `(record ?n)`: how often the game record `before` (game.py `Record`) holds the board's
    identity, saturating at REPETITION_LIMIT; 0 with no record."""
    n = 0
    if before is not None and board is not None:
        from catspace.incubator.analyst.discovery.game import CHESS
        n = before.count(CHESS.position_key(board))
    return (("record", (str(min(n, REPETITION_LIMIT)),)),)


class Reading:
    """The spec's answers at one state: the mover's legal moves (UCI), the terminal verdict, the terminal reasons that hold, the goal values. `inputs`: the
    turn's input facts (`inputs_of`: the game record's reading; none given -- no record)."""

    REASONS = ("mate", "stalemate", "draw", "draw_by_repetition", "draw_by_fifty_moves")

    def __init__(self, state: frozenset, prog: "GDL.Program | None" = None, inputs: "tuple | None" = None):
        prog = prog or program()
        self.state = state
        self.db = prog.run(state, inputs=inputs if inputs is not None else inputs_of())
        mover = next(a[0] for rel, a in state if rel == "true.control")
        self.mover = mover
        self.legal = sorted(uci_of(m) for role, m in self.db.facts("legal") if role == mover and m != "noop")
        self.terminal = prog.holds(self.db, "terminal")
        self.reasons = tuple(r for r in self.REASONS if prog.holds(self.db, r))
        self.goals = {role: sorted(vs) for role, vs in prog.goals(self.db).items()}
        self.check = (mover,) in self.db.facts("check")


def next_state(state: frozenset, uci: str, mover: str, prog: "GDL.Program | None" = None) -> frozenset:
    prog = prog or program()
    other = "black" if mover == "white" else "white"
    return prog.next_state(state, [(mover, move_term(uci)), (other, "noop")])


# the kernel's terminal kinds (Game.terminal) and the spec's reasons, by name
KERNEL_KIND = {"mate": "mate", "stalemate": "stalemate", "draw": "draw", "repetition": "draw_by_repetition", "fifty-move": "draw_by_fifty_moves"}
KERNEL_ORDER = ("mate", "stalemate", "draw", "draw_by_repetition", "draw_by_fifty_moves")      # Game.terminal's order (Lean `Chess.gameStatus`: a mate first)


def kernel_reasons(board: chess.Board, before=None) -> tuple:
    """python-chess's terminal predicates, each read on its own (not only the first): the reasons that hold. `before` (plan 76.1 (b) (5)): the game record --
    the repetition, Game.repetition."""
    out = []
    if board.is_checkmate():
        out.append("mate")
    if board.is_stalemate():
        out.append("stalemate")
    if board.is_insufficient_material():
        out.append("draw")
    if before is not None:
        from catspace.incubator.analyst.discovery.game import CHESS
        if CHESS.repetition(board, before):
            out.append("draw_by_repetition")
    if board.is_fifty_moves():
        out.append("draw_by_fifty_moves")
    return tuple(out)


def goals_of(kind: "str | None", mover: str) -> dict:
    """The goal values of a kernel verdict (the spec's: the mated side 0, the other 100; a draw 50 each; not terminal: none)."""
    other = "black" if mover == "white" else "white"
    if kind is None:
        return {}
    if kind == "mate":
        return {mover: ["0"], other: ["100"]}
    return {mover: ["50"], other: ["50"]}


# ------------------------------------------------------------------------------------------------------------------------------------------ the surprise store
# (plan 76.0 (b) wrote these in scripts/crossplay.py; plan 76.1 (a) moved them here, the one place both the cross-play and the ontology shelf's agreement read)

SURPRISES = paths.REPO_ROOT / "rules" / "surprises"


def expected_rust_draw(ns, prog=None) -> "str | None":
    """The Rust core's draw reason after a move, as the SPEC's facts give it in the Rust core's own order (game.rs `drawn`): the clock at 100 and no mate on the move ->
    'the fifty-move rule'; else the spec's insufficient material -> 'insufficient material'; else None (a stalemate is compared at the position)."""
    prog = prog or program()
    clock = next((int(a[0]) for rel, a in ns if rel == "true.halfmove"), 0)
    if clock >= CLOCK_LIMIT:
        db = prog.run(ns, wanted=("mate",))
        if not prog.holds(db, "mate"):
            return "the fifty-move rule"
        return None
    db = prog.run(ns, wanted=("draw",))
    if prog.holds(db, "draw"):
        return "insufficient material"
    return None


def material_signature(state) -> str:
    """A state's material by colour (the signature of a draw disagreement): e.g. 'KB(l) v KB(l)'."""
    sides = {"white": [], "black": []}
    for rel, a in state:
        if rel == "true.cell":
            f, rk, c, k = a
            letter = "N" if k == "knight" else k[0].upper()
            if k == "bishop":
                letter += "(d)" if (ord(f) - 97 + int(rk) - 1) % 2 == 0 else "(l)"
            sides[c].append(letter)
    order = "KQRB(d)B(l)NP"
    key = lambda x: order.index(x[0]) if x[0] in "KQRNP" else order.index(x)
    return " v ".join("".join(sorted(sides[c], key=key)) for c in ("white", "black"))


def load_surprises() -> list:
    """Every filed case of rules/surprises/ (with its file name)."""
    import json
    return [dict(json.loads(p.read_text()), file=p.name) for p in sorted(SURPRISES.glob("*.json"))]


def file_surprise(key: tuple, fen: str, shrunk: str, detail: str):
    """A new case under rules/surprises/ (status 'open': the reading of the rules, the side that was wrong and the change are written after reading them)."""
    import json
    SURPRISES.mkdir(parents=True, exist_ok=True)
    n = len(list(SURPRISES.glob("*.json")))
    p = SURPRISES / f"s{n + 1:03d}_{key[0]}_{key[1].replace(' ', '_')}.json"
    case = {"fen": shrunk, "found_at": fen, "kernel": key[0], "class": key[1], "signature": key[2],
            "spec_vs_kernel": detail, "status": "open", "wrong": None, "fide": None, "change": None, "expect": "agree"}
    p.write_text(json.dumps(case, indent=1) + "\n")
    return p


# ------------------------------------------------------------------------------------------------------------------------------------------ plan 76.2 (b): THE ORACLE
# The GDL interpreter read against EVERY Game method that reads a rule (game.SPEC: the method -> the spec's relation): the same state, the spec's answer and the
# surface's, compared. One reading of the state (the legal moves, the terminal reasons, the goals, check, the cells, the attacks, the captures), one next state and
# one clock-event reading per legal move, one check reading per next state.

def _squares(game, board) -> dict:
    return {s: sq(s) for s in game.locations()}


def game_oracle(game, board: chess.Board, before=None, prog: "GDL.Program | None" = None) -> list:
    """-> [(method, detail)]: every Game method of game.SPEC whose answer at `board` (with the game record `before`) differs from the spec's. Empty: the surface
    agrees with rules/chess.gdl at this state, on every legal move."""
    from catspace.incubator.analyst.discovery import game as GM
    prog = prog or program()
    out = []
    state = state_of(board)
    mover = COLOUR[board.turn]
    other = "black" if mover == "white" else "white"
    r = Reading(state, prog, inputs_of(board, before))
    db = r.db
    name = _squares(game, board)
    legal = list(game.moves(board))
    ucis = sorted(m.uci() for m in legal)
    if ucis != r.legal:
        out.append(("moves", f"surface {sorted(set(ucis) - set(r.legal))} not the spec's; spec {sorted(set(r.legal) - set(ucis))} not the surface's"))
    for m in list(board.generate_pseudo_legal_moves()) + legal:
        if game.legal(board, m) != (m.uci() in r.legal):
            out.append(("legal", f"{m.uci()}: surface {game.legal(board, m)}"))
    for m in legal:
        frm, to = m.from_square, m.to_square
        got = sorted(x.uci() for x in game.moves_to(board, frm, to))
        exp = sorted(u for u in r.legal if u[:4] == m.uci()[:4])
        if got != exp:
            out.append(("moves_to", f"{m.uci()[:4]}: surface {got}, spec {exp}"))
    if COLOUR[game.turn(board)] not in {a[0] for rel, a in state if rel == "true.control"}:
        out.append(("turn", f"surface {game.turn(board)}"))
    kind = game.terminal(board, before)
    spec_kind = next((k for k in KERNEL_ORDER if k in r.reasons), None)
    if (KERNEL_KIND.get(kind) if kind is not None else None) != spec_kind:
        out.append(("terminal", f"surface {kind}, spec reasons {r.reasons}"))
    for side, role in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        g = game.goal(board, side, before)
        exp = r.goals.get(role, [])
        if (sorted([str(g)]) if g is not None else []) != exp:
            out.append(("goal", f"{role}: surface {g}, spec {exp}"))
    if before is not None and game.repetition(board, before) != ("draw_by_repetition" in r.reasons):
        out.append(("repetition", f"surface {game.repetition(board, before)}"))
    if game.in_check(board) != r.check:
        out.append(("in_check", f"surface {game.in_check(board)}, spec {r.check}"))
    clock = next(int(a[0]) for rel, a in state if rel == "true.halfmove")
    if min(game.clock(board), CLOCK_LIMIT) != clock:
        out.append(("clock", f"surface {game.clock(board)}, spec {clock}"))
    cells = {(a[0], a[1]): (a[2], a[3]) for rel, a in state if rel == "true.cell"}
    for s in game.locations():
        u = game.unit_at(board, s)
        exp = cells.get(name[s])
        got = (COLOUR[u.owner], chess.piece_name(u.kind)) if u is not None else None
        if got != exp:
            out.append(("unit_at", f"{chess.square_name(s)}: surface {got}, spec {exp}"))
    units = sorted((name[u.loc], COLOUR[u.owner], chess.piece_name(u.kind)) for u in game.units(board))
    if units != sorted((fr, ck[0], ck[1]) for fr, ck in cells.items()):
        out.append(("units", "the units differ"))
    for side, role in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        k = game.royal(board, side)
        exp = [fr for fr, ck in cells.items() if ck == (role, "king")]
        if ([name[k]] if k is not None else []) != exp[:1] or len(exp) > 1:
            out.append(("royal", f"{role}: surface {k}, spec {exp}"))
    attacks = db.facts("attack")
    for s in game.locations():
        for side, role in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            got = sorted(name[x] for x in game.attackers(board, side, s))
            exp = sorted((a[1], a[2]) for a in attacks if a[0] == role and (a[3], a[4]) == name[s])
            if got != exp:
                out.append(("attackers", f"{role} on {chess.square_name(s)}: surface {got}, spec {exp}"))
        u = game.unit_at(board, s)
        if u is not None:
            got = sorted(name[x] for x in game.attacks_from(board, s))
            exp = sorted((a[3], a[4]) for a in attacks if a[0] == COLOUR[u.owner] and (a[1], a[2]) == name[s])
            if got != exp:
                out.append(("attacks_from", f"{chess.square_name(s)}: surface {got}, spec {exp}"))
        got = sorted(name[x] for x in game.adjacent(s))
        exp = sorted((a[2], a[3]) for a in prog.static["king_step"].facts if (a[0], a[1]) == name[s])
        if got != exp:
            out.append(("adjacent", f"{chess.square_name(s)}: surface {got}, spec {exp}"))
    captures = {(a[2], a[3], a[4], a[5]) for a in db.facts("capture") if a[1] == mover}
    nexts = tuple(rel for rel in prog.nodes if rel.startswith("next."))
    for m in legal:
        u = m.uci()
        term = move_term(u)
        dbn = prog.run(state, [(mover, term), (other, "noop")], wanted=nexts + ("clock_event",))      # (one run: the next state and the move's clock events)
        ns = frozenset(("true." + rel[5:], t) for rel in nexts for t in dbn.facts(rel))
        if state_of(game.apply(board, m)) != ns:
            out.append(("apply", f"{u}: the next states differ"))
        chk = (other,) in prog.run(ns, wanted=("check",)).facts("check")
        if game.gives_check(board, m) != chk:
            out.append(("gives_check", f"{u}: surface {game.gives_check(board, m)}, spec {chk}"))
        cap = (sq(m.from_square) + sq(m.to_square)) in captures
        if game.is_capture(board, m) != cap:
            out.append(("is_capture", f"{u}: surface {game.is_capture(board, m)}, spec {cap}"))
        rk = "check" if chk else "capture" if cap else GM.QUIET
        if game.reply_kind(board, m) != rk:
            out.append(("reply_kind", f"{u}: surface {game.reply_kind(board, m)}, spec {rk}"))
        ev = {a[0] for a in dbn.facts("clock_event")}
        if set(game.clock_events(board, m)) != ev:
            out.append(("clock_events", f"{u}: surface {sorted(game.clock_events(board, m))}, spec {sorted(ev)}"))
        if game.promotes(m) != (term[0] == "promote"):
            out.append(("promotes", f"{u}: surface {game.promotes(m)}"))
    return out
