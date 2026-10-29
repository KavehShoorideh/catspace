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


# ------------------------------------------------------------------------------------------------------------------------------------------ plan 76.4a: THE REGRESSION'S BINDING
# What discovery/regression.py (generic: it names no game object) needs of a game, for chess on python-chess boards: the spec's state of a board (`state_of`), the
# CERTIFIED relations read through the kernel -- `legal` (the store's RULE "the rule: legal move", PROVED: Ideas.LegalEquiv.legal_iff; Game.moves) -- the certified next
# state (RULE "the rule: next state", PROVED: Ideas.NextEquiv.applyMoveSpec_eq_applyMove; Game.apply), the turn passed (Game.pass_: the null move), a board given to a
# side, the exchange law's value (the Rust `see`, the mirror of Lean Ideas.ExchangeLaw; None where the kernel holds no position: the side not to move in check), and the
# reading of the spec's terms as squares (a location is a (file, rank) pair of the spec's sorts).

CERTIFIED = ("legal", "attack_by")                      # the spec relations the regression reads through the kernel (their RULE entries are PROVED): legal moves
LOCATION_SORTS = ("file", "rank")                       # (Ideas.LegalEquiv.legal_iff), a unit's attack (Ideas.LegalEquiv.attackBy_iff: unconditional)
_KIND = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
_SQ = {(chess.FILE_NAMES[f], str(r + 1)): chess.square(f, r) for f in range(8) for r in range(8)}


class Binding:
    """The chess binding of the regression (plan 76.4a): python-chess boards as the states, the kernel's legal moves and next state as the certified relations."""

    def __init__(self, prog: "GDL.Program | None" = None):
        self.prog = prog or program()
        self._facts: dict = {}
        self._slots: dict = {}

    @staticmethod
    def key(b: chess.Board) -> tuple:
        return (b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings, b.occupied_co[True], b.occupied_co[False], b.turn, b.clean_castling_rights(),
                b.ep_square, min(b.halfmove_clock, CLOCK_LIMIT))

    def facts(self, b: chess.Board) -> frozenset:
        k = self.key(b)
        got = self._facts.get(k)
        if got is None:
            if len(self._facts) > 200_000:
                self._facts.clear()
            got = self._facts[k] = state_of(b) | frozenset(inputs_of())
        return got

    @staticmethod
    def inputs(b: chess.Board):
        """The spec's state of `b` read where it is asked: (rel, pattern) -> the matching rows of `true.cell` (the unit on a square), `true.control`,
        `true.castling_right`, `true.ep_square`, `true.halfmove` and the input `record` (no game record: 0) -- `state_of`'s facts, by pattern."""
        def rows(rel: str, pat: tuple):
            if rel == "true.cell":
                f, r, c, k = pat
                if f is not None and r is not None:
                    s = _SQ[(f, r)]
                    p = b.piece_at(s)
                    if p is None:
                        return ()
                    row = (f, r, COLOUR[p.color], _KIND[p.piece_type])
                    return (row,) if (c is None or c == row[2]) and (k is None or k == row[3]) else ()
                out = []
                for s, p in b.piece_map().items():
                    row = sq(s) + (COLOUR[p.color], _KIND[p.piece_type])
                    if (f is None or f == row[0]) and (r is None or r == row[1]) and (c is None or c == row[2]) and (k is None or k == row[3]):
                        out.append(row)
                return out
            if rel == "true.control":
                row = (COLOUR[b.turn],)
                return (row,) if pat[0] is None or pat[0] == row[0] else ()
            if rel == "true.castling_right":
                out = [RIGHT[s] for s in chess.SquareSet(b.clean_castling_rights())]
                return [t for t in out if all(v is None or v == t[i] for i, v in enumerate(pat))]
            if rel == "true.ep_square":
                if b.ep_square is None:
                    return ()
                row = sq(b.ep_square)
                return (row,) if all(v is None or v == row[i] for i, v in enumerate(pat)) else ()
            if rel == "true.halfmove":
                row = (str(min(b.halfmove_clock, CLOCK_LIMIT)),)
                return (row,) if pat[0] is None or pat[0] == row[0] else ()
            if rel == "record":
                return (("0",),) if pat[0] in (None, "0") else ()
            return ()
        return rows

    def certified(self, b: chess.Board):
        """The rows of the certified relations on `b` matching a pattern (None where free): `legal` (the kernel's legal moves of the side to move, in its order; the
        other side's noop), `attack_by` (each unit's kind, colour, square and a square it bears on: python-chess's attacks, the unit's geometry with the line stopped
        at the first unit)."""
        mover = COLOUR[b.turn]
        other = COLOUR[not b.turn]
        memo: list = []

        def rows(rel: str, pat: tuple) -> list:
            if rel == "legal":
                if not memo:
                    memo.append([(mover, move_term(m.uci())) for m in b.legal_moves] + [(other, "noop")])
                return [t for t in memo[0] if (pat[0] is None or pat[0] == t[0]) and (pat[1] is None or pat[1] == t[1])]
            if rel == "attack_by":
                k, c, f1, r1, f2, r2 = pat
                out = []
                if f2 is not None and r2 is not None:
                    t = _SQ[(f2, r2)]
                    for colour in ((c == "white",) if c is not None else (True, False)):
                        for s in b.attackers(colour, t):
                            kind = _KIND[b.piece_type_at(s)]
                            fs, rs = sq(s)
                            if (k is None or k == kind) and (f1 is None or f1 == fs) and (r1 is None or r1 == rs):
                                out.append((kind, COLOUR[colour], fs, rs, f2, r2))
                    return out
                srcs = [_SQ[(f1, r1)]] if f1 is not None and r1 is not None else list(chess.SquareSet(b.occupied))
                for s in srcs:
                    p = b.piece_at(s)
                    if p is None or (c is not None and COLOUR[p.color] != c) or (k is not None and _KIND[p.piece_type] != k):
                        continue
                    fs, rs = sq(s)
                    for t in b.attacks(s):
                        ft, rt = sq(t)
                        if (f2 is None or f2 == ft) and (r2 is None or r2 == rt):
                            out.append((_KIND[p.piece_type], COLOUR[p.color], fs, rs, ft, rt))
                return out
            raise KeyError(f"{rel} is not a certified relation of the chess binding")
        return rows

    @staticmethod
    def action(args: tuple):
        role, term = args
        return None if term == "noop" else (role, term)

    @staticmethod
    def apply(b: chess.Board, action) -> chess.Board:
        c = b.copy(stack=False)
        c.push(chess.Move.from_uci(uci_of(action[1])))
        return c

    @staticmethod
    def pass_(b: chess.Board) -> chess.Board:
        c = b.copy(stack=False)
        c.turn = not b.turn
        c.ep_square = None
        return c

    @staticmethod
    def side(role: str) -> bool:
        return role == "white"

    def to_move(self, b: chess.Board, role: str) -> chess.Board:
        colour = self.side(role)
        if b.turn == colour:
            return b
        c = b.copy(stack=False)
        c.turn = colour
        c.ep_square = None
        return c

    def value(self, b: chess.Board, law: str, args: tuple):
        """The valuation laws the store's goals read: 'exchange' (args: the target's square, the capturing side) -- what the side wins by the exchange on the square
        (the Rust see, the mirror of Lean Ideas.ExchangeLaw); 'exchange from' (args: the target's square, the capturer's square, the side; plan 76.4a (d)) -- the
        exchange begun by that capture: the target's worth (the store's material valuation) less what the other side then wins back on the square (their exchange,
        at least 0). None where the kernel holds no position to read it on."""
        from catspace.incubator.analyst.discovery import graph as G
        if law == "exchange":
            f, r, role = args
            p = G.pos_of(self.to_move(b, role))
            return None if p is None else p.see(chess.square(chess.FILE_NAMES.index(f), int(r) - 1))
        if law == "exchange from":
            tf, tr, ff, fr, role = args
            t, s = _SQ[(tf, tr)], _SQ[(ff, fr)]
            x = self.to_move(b, role)
            m = chess.Move(s, t)
            if x.piece_at(t) is None or x.piece_at(s) is None or not x.is_pseudo_legal(m):
                return None
            from catspace.incubator.analyst.discovery import lemma_store as LS
            worth = LS.goal_kinds().valuation.worth
            gain = worth.get(x.piece_at(t).symbol().lower(), 0)
            after = x.copy(stack=False)
            after.push(m)
            p = G.pos_of(after)
            return None if p is None else gain - max(0, p.see(t))
        raise KeyError(f"no valuation law {law!r} in the chess binding")

    @staticmethod
    def touched(b: chess.Board, action) -> tuple:
        """The squares an action's next state changes on `b` (the spec's `lifted` and `landed`): its two squares, the castling rook's two, the pawn taken en passant."""
        m = chess.Move.from_uci(uci_of(action[1]))
        out = [m.from_square, m.to_square]
        if b.is_castling(m):
            rank = chess.square_rank(m.from_square)
            ks = chess.square_file(m.to_square) > chess.square_file(m.from_square)
            out += [chess.square(7 if ks else 0, rank), chess.square(5 if ks else 3, rank)]
        elif b.is_en_passant(m):
            out.append(chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)))
        return tuple(out)

    @staticmethod
    def same_at(b0: chess.Board, b1: chess.Board, locs) -> bool:
        """THE CONTRACT the regression's validation reads (plan 76.4a): `b0` and `b1` hold the same unit on every location of `locs` and the same side to move, castling
        rights and e.p. square -- then every leaf the regression reads at those locations (a unit on a square, a square occupied, the side to move) has the same truth in
        the next state of any action whose touched squares are among them."""
        if b0.turn != b1.turn or b0.castling_rights != b1.castling_rights or b0.ep_square != b1.ep_square:
            return False
        return all(b0.piece_at(s) == b1.piece_at(s) for s in locs if s is not None)

    @staticmethod
    def ends(action) -> tuple:
        t = action[1]
        return (chess.square(chess.FILE_NAMES.index(t[1]), int(t[2]) - 1), chess.square(chess.FILE_NAMES.index(t[3]), int(t[4]) - 1))

    def loc(self, atom) -> "int | None":
        """The first ground location (a file and a rank slot of the relation's signature) of an atom; None when it has none."""
        rel, args = atom
        slots = self._slots.get(rel)
        if slots is None:
            sig = self.prog.signature(rel) or ()
            slots = self._slots[rel] = tuple(i for i in range(len(sig) - 1) if sig[i] == LOCATION_SORTS[0] and sig[i + 1] == LOCATION_SORTS[1])
        for i in slots:
            if i + 1 < len(args) and args[i] is not None and args[i + 1] is not None:
                return chess.square(chess.FILE_NAMES.index(args[i]), int(args[i + 1]) - 1)
        return None

    @staticmethod
    def terms(sq: int) -> tuple:
        return sq_terms(sq)

    @staticmethod
    def role(side: bool) -> str:
        return COLOUR[side]

    @staticmethod
    def other(role: str) -> str:
        return "black" if role == "white" else "white"

    @staticmethod
    def loc_name(sq) -> str:
        return chess.square_name(sq) if sq is not None else ""

    @staticmethod
    def order_key(sq) -> int:
        return -1 if sq is None else sq

    @staticmethod
    def unit_letter(b: chess.Board, sq: int) -> str:
        p = b.piece_at(sq)
        return p.symbol().upper() if p is not None else "?"

    # -------------------------------------------------------------------------------- plan 76.4b: THE NATIVE CORE (rust/analyst_core/src/regression.rs)
    # The regression read in Rust (the spec loaded there from rules/chess.gdl's text, the binding mirrored there: gdl_chess.rs reads the board as python-chess
    # holds it, pyboard.rs); the Python reading of regression.py is its TEST ORACLE (`regression.PyRegressor`). What crosses: the board's parts, a goal's id (the
    # GOAL entry's `gdl` substituted, registered once), the store's DEFECT_FIX bindings (registered once); back: the regression's fields, or the names' records
    # (the template's fields -- the TEXT is formatted here, RUST_BOUNDARY risk 3), or per move the fix test's verdict. No Python callback per candidate.

    @staticmethod
    def future(b: chess.Board, fut) -> chess.Board:
        """The goal's future of a state (regress.future_board, read here for the oracle): None -- the state itself; else (placements [(symbol, at, lifted from)], white
        to move) -- each unit placed (graph.with_piece: whatever stood there gone, lifted from its square), then that side to move, the e.p. square dropped."""
        if fut is None:
            return b
        pl, turn = fut
        c = b
        for sym, at, frm in pl:
            c = c.copy(stack=False)
            p = chess.Piece.from_symbol(sym)
            if frm is not None:
                c._remove_piece_at(frm)
            c._set_piece_at(at, p.piece_type, p.color)
        c = c.copy(stack=False)
        c.turn = turn
        c.ep_square = None
        return c

    def native_goal(self, goal) -> int:
        """The goal's id in the native core (registered once per process: the head with its variables None, the role on move, the value)."""
        gid = _GIDS.get(goal)
        if gid is None:
            A = native()
            rel, args = goal.head
            v = goal.value
            val = None if v is None else (v[0], list(v[1]), int(v[2]), v[3][0], [None if GDL.is_var(a) else a for a in v[3][1]])
            gid = _GIDS[goal] = A.spec_goal(goal.name, rel, [None if GDL.is_var(a) else a for a in args], goal.on_move, val)
        return gid

    @staticmethod
    def native_entries(entries: list) -> None:
        """The store's DEFECT_FIX bindings handed to the native core (once per list: the store's `defect_names`), the templates' keys with them; the texts stay here."""
        if _ENTRIES[0] is entries:
            return
        A = native()
        rows, keys = [], []
        for e in entries:
            g = e["gdl"]
            lf, un = g.get("leaf"), g.get("unless")
            rows.append((e["name"], g["goal"], g.get("for", "ours") == "theirs", g["defect"], g["order"], None if lf is None else tuple(lf), None if un is None else tuple(un),
                         g.get("sq", ""), g.get("by", ""), g.get("group", ""), g.get("sort") == "sq", None if g.get("clause") is None else (g["clause"], g["lit"])))
            keys.append(sorted(g["text"]) if isinstance(g["text"], dict) else None)
        A.spec_names(rows, keys)
        _ENTRIES[0] = entries

    def native_regress(self, b: chess.Board, goal):
        """THE REGRESSION read by the native core, as regression.py's Regression (its dataclasses): the derivation, the defects with their witnesses and the leaves
        they flip, the read set, the head's rows, the value read, the leaves."""
        from catspace.incubator.analyst.discovery import regression as RG
        holds, clauses, defects, reads, rows, value, leaves = native().regress_bb(parts(b), self.native_goal(goal))

        def leaf(t):
            return RG.Leaf(t[0], t[1], t[2])
        ds = []
        for kind, clause, lit, atom, deep, w, val in defects:
            wit = None if w is None else RG.Witness(w[0], tuple(w[1]), w[2], frozenset(leaf(f) for f in w[3]))
            ds.append(RG.Derived(kind, clause, lit, atom, deep if deep is not None else (), wit, None if val is None else (val[0], val[1], tuple(sorted(val[2])))))
        return RG.Regression(goal, holds, [(i, list(ls)) for i, ls in clauses], ds, frozenset(reads), rows=list(rows), value=value,
                             leaves=frozenset(leaf(t) for t in leaves), state=b)

    def native_names(self, b: chess.Board, goal, entries: list, theirs: bool = False, fut=None) -> list:
        """THE NAMES of the goal's defects on the state's future, read by the native core (cached there per (key, goal)): the records' fields filled into the entries'
        templates here (regression.name's texts)."""
        from catspace.incubator.analyst.discovery import regression as RG
        self.native_entries(entries)
        out = []
        for idx, s, by, frm, to, unit, value, rel in native().names_bb(parts(b), self.native_goal(goal), theirs, fut):
            e = entries[idx]
            g = e["gdl"]
            d = g["defect"]
            name = chess.square_name(s) if s is not None else ""
            if d == "witness":
                why = g["text"].format(sq=name, frm=chess.square_name(frm), to=chess.square_name(to), unit=unit, n=len(by))
            elif d == "false":
                tmpl = g["text"].get(rel) if isinstance(g["text"], dict) else g["text"]
                why = tmpl.format(sq=name)
            elif d == "leaf":
                why = g["text"].format(sq=name, unit=unit)
            else:
                why = g["text"].format(sq=name, n=len(by), value=value)
            out.append(RG.Named(e["name"], s, list(by), why, []))
        return out

    def native_fixes(self, b: chess.Board, goal, key: tuple, actions: list, fut, entries: list, before=None) -> list:
        """THE FIX TEST of the named defect `key` for each action (a binding action (role, term) or the kernel's move), read by the native core in one crossing:
        [(action, mended, new)] as regression.fixes."""
        self.native_entries(entries)
        moves = []
        for a in actions:
            if isinstance(a, chess.Move):
                moves.append((a.from_square, a.to_square, a.promotion or 0))
            else:
                m = chess.Move.from_uci(uci_of(a[1]))
                moves.append((m.from_square, m.to_square, m.promotion or 0))
        got = native().fixes_bb(parts(b), self.native_goal(goal), (key[0], key[1]), moves, fut, None if before is None else list(before))
        return [(a, mended, [tuple(k) for k in new]) for a, (mended, new) in zip(actions, got)]

    @staticmethod
    def native_rows(b: chess.Board, rel: str, pattern: tuple) -> list:
        """The rows of a spec relation on the board (plan 76.4b (d)), by the native core's prover with the certified relations read through the binding."""
        return native().spec_ans_bb(parts(b), rel, list(pattern), True)

    @staticmethod
    def native_clear() -> None:
        if _NATIVE[0] is not None:
            _NATIVE[0].spec_cache_clear()

    @staticmethod
    def native_stats() -> tuple:
        return _NATIVE[0].spec_stats() if _NATIVE[0] is not None else (0, 0, 0)


_NATIVE: list = [None]               # plan 76.4b: the native core (analyst_core with the spec loaded), once per process
_GIDS: dict = {}                     # Goal -> its id in the native core
_ENTRIES: list = [None]              # the DEFECT_FIX list registered there


def parts(b: chess.Board) -> tuple:
    """The board's fields as the native core reads a python-chess board: the eight bitboards, white to move, the RAW castling rights, the RAW e.p. square, the clocks."""
    return ([b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings, b.occupied_co[True], b.occupied_co[False]], b.turn, b.castling_rights, b.ep_square,
            (b.halfmove_clock, b.fullmove_number))


def native_bindings(prog: "GDL.Program | None" = None) -> list:
    """THE BINDING TABLE handed to the native core (plan 76.4b (a)): the ontology shelf's entry of every input relation of the spec (a `true` fluent, `does`, `record`)
    and of every certified relation -- (relation, the entry's `rust` binding); an object kind's slot entry is no relation's binding."""
    from catspace.incubator.analyst.discovery import ontology as ON
    prog = prog or program()
    want = {r for r in prog.nodes if GDL._is_input(r)} | set(CERTIFIED)
    out = []
    for e in ON.load().values():
        g = e.get("gdl") or {}
        if "slot" in g or not e.get("rust"):
            continue
        rel = ("true." + g["fluent"]) if "fluent" in g else g.get("relation")
        if rel in want:
            out.append((rel, e["rust"]))
    return sorted(out)


def native():
    """The native core with rules/chess.gdl loaded (its TEXT read here, the file the source; the shelf's binding table; the certified relations), once per process.
    The 'exchange from' law's worths are the core's valuation (plan 76.5: the store's entry, set once in the core -- lemma_store.install_valuation)."""
    if _NATIVE[0] is None:
        import analyst_core as A
        A.spec_load(SPEC.read_text(), native_bindings(), list(CERTIFIED), LOCATION_SORTS, CLOCK_LIMIT)
        _NATIVE[0] = A
    return _NATIVE[0]


def sq_terms(s: int) -> tuple:
    return sq(s)


# ------------------------------------------------------------------------------------------------------------------------------------------ plan 76.4c: the term language's reader

class TermReader:
    """PLAN 76.4c (a): the chess reading of the term language (terms.py, which names no game object): a state is a python-chess board; `rows` -- every row of a
    relation of the spec, read by the native core's prover with the certified relations through the binding (the regression's reading, 76.4b: equal to the GDL
    interpreter on every relation of the guard positions, scripts/plan76_4a.py crosscheck); `lift` -- the board with the units on the given locations taken off
    (the side to move, the rights, the clocks kept: the counterfactual the regression's boards already are)."""

    def __init__(self, prog: "GDL.Program | None" = None):
        self.prog = prog or program()

    @staticmethod
    def key(b: chess.Board):
        return Binding.key(b)

    def rows(self, b: chess.Board, rel: str):
        sig = self.prog.signature(rel)
        return Binding.native_rows(b, rel, (None,) * len(sig))

    @staticmethod
    def lift(b: chess.Board, locs, dom) -> chess.Board:
        c = b.copy(stack=False)
        for i in locs:
            c.remove_piece_at(_SQ[tuple(dom.ungroup(i))])
        return c


def valuation_weights(valuation) -> dict:
    """PLAN 76.4c: a VALUATION entry's worths (by the kernel's unit letters) as weights of the spec's unit sort (its members by name) -- the binding's mapping."""
    from catspace.incubator.analyst.discovery import names as NM
    return {_KIND[NM.kind(letter)]: w for letter, w in valuation.worth.items()}


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
