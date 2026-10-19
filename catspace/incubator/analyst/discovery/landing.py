"""52.18 (1): THE LANDING TABLE -- targets and threats (Lean: Ideas.Landing; words in GLOSSARY.md).

A LANDING is (side, piece, from, to): the piece lifted from `from` and put on `to`, the OTHER side to move (the counterfactual the roots already use, Lean `land`). It is a TARGET of
the side when a claim of that side binds in it with no defect left -- a mate in one, the exchange winning (the move is forcing by 52.16 (3): safe on its square, or the threat
survives the capture of the piece), a fork (52.18 (4)); the same word read from the other side is a THREAT. The table is NOT maintained: it is the memo of landing answers (the claim, or
the defect list, with the READ SET of the squares and pieces read), keyed by (side, piece, from, to) and valid on a board as long as the 52.13 diff (`regress.diff_squares`) of the board it
was asked on and this one misses the read set (Lean `land_agree` is why the key is enough). The INVERTED READ INDEX answers "what is known about their queen on d6": every landing
whose read set contains d6. No per-square control table: the exchange word orders attackers by value on the fly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import regress as R


@dataclass
class Landing:
    side: bool
    sym: str
    frm: int
    to: int
    kind: str = "none"                       # 'target' (a claim of `side` binds) | 'none'
    claim: str = ""                          # 'mate in one' | 'exchange' | 'fork' | ''
    value: int = 0
    via: str = ""                            # the words of the claim, for the trace
    defects: list = field(default_factory=list)       # the defect kinds that stand (mate landing that is not yet a mate)
    defect_objs: list = field(default_factory=list)   # the Defect objects (the agenda's mate roots read them back)
    reads: frozenset = frozenset()

    def key(self) -> tuple:
        return (self.side, self.sym, self.frm, self.to)

    def text(self, who: str = "") -> str:
        name = G.name
        head = f"{self.sym.upper()} {name(self.frm)}->{name(self.to)}"
        if self.kind == "target":
            return f"{head}: {'threat' if who == 'theirs' else 'target'} ({self.via}), reads {{{', '.join(name(s) for s in sorted(self.reads))}}}"
        return f"{head}: no claim binds" + (f" ({', '.join(self.defects)} stand)" if self.defects else "") + f", reads {{{', '.join(name(s) for s in sorted(self.reads))}}}"


def exchange_reads(board: chess.Board, sq: int) -> set:
    """The squares an exchange on `sq` reads: the square, every piece that attacks it directly or through pieces (x-ray, either colour), the pieces in front of them and the lines between."""
    out = {sq}
    for colour in (chess.WHITE, chess.BLACK):
        out.update(board.attackers(colour, sq))
        for ps, blockers in G.xrays_to(board, sq, colour):
            out.add(ps)
            out.update(blockers)
            out.update(chess.SquareSet(chess.between(ps, sq)))
    return out


def evaluate(board: chess.Board, side: bool, frm: int, to: int) -> Landing:
    """The answer to the landing question, computed (no table)."""
    pc = board.piece_at(frm)
    lnd = Landing(side, pc.symbol().lower() if pc is not None else "?", frm, to)
    lnd.reads = frozenset({frm, to})
    if pc is None or pc.color != side:
        return lnd
    occ = board.piece_at(to)
    if occ is not None and (occ.color == side or occ.piece_type == chess.KING):
        return lnd
    cf = G.with_turn(G.with_piece(board, pc, to, frm), not side)
    reads = {frm, to}
    ksq = cf.king(not side)
    if ksq is not None and ksq in cf.attacks(to):                       # the landing gives check: is it a mate?
        defects = R.mate_defects_on(cf, side)
        reads.update({ksq, *chess.SquareSet(chess.BB_KING_ATTACKS[ksq]), *chess.SquareSet(chess.between(to, ksq))})
        for r in chess.SQUARES:                                         # a pin lies on a line through their king: the lines are read (a taker pinned against the king cannot take the checker)
            if r != ksq and (chess.square_file(r) == chess.square_file(ksq) or chess.square_rank(r) == chess.square_rank(ksq)
                             or abs(chess.square_file(r) - chess.square_file(ksq)) == abs(chess.square_rank(r) - chess.square_rank(ksq))):
                reads.add(r)
        for d in defects:
            reads.update(d.by)
            if d.sq is not None:
                reads.add(d.sq)
        if not defects:
            lnd.kind, lnd.claim, lnd.value, lnd.via = "target", "mate in one", R.STAKE_MATE, "mate in one"
        else:
            lnd.defect_objs = defects
            lnd.defects = sorted({d.kind for d in defects})
    if lnd.kind != "target":
        m = chess.Move(frm, to, promotion=chess.QUEEN if pc.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7) else None)
        mover = G.with_turn(board, side)
        if m in mover.legal_moves and not mover.gives_check(m):
            why = R.forcing_why(mover, m, 1, side)
            if why and (why.startswith("threat") or why.startswith("capture")):
                import re
                v = int(re.search(r"\d+", why).group(0))
                c2 = mover.copy(stack=False)
                c2.push(m)
                best = (0, None)
                after = G.with_turn(c2, side)                           # our exchange after a pass: where the claim lives
                for s in chess.SQUARES:
                    p2 = after.piece_at(s)
                    if p2 is not None and p2.color != side and p2.piece_type != chess.KING and after.is_attacked_by(side, s):
                        w = G.see(after, s)
                        if w > best[0]:
                            best = (w, s)
                lnd.kind, lnd.claim, lnd.value = "target", "exchange", v
                lnd.via = f"win {v} by the exchange: {why}" + (f"; {R.exchange_read(after)}" if best[1] is not None else "")
                if best[1] is not None:
                    reads.update(exchange_reads(after, best[1]))
                reads.update(exchange_reads(c2, to))
                for cap in [x for x in c2.legal_moves if x.to_square == to]:         # the threat that SURVIVES the capture of the piece reads its own lines (Bxf6, then Qxa8 through the vacated e4)
                    d = c2.copy(stack=False)
                    d.push(cap)
                    dd = G.with_turn(d, side)
                    for s2 in chess.SQUARES:
                        p3 = dd.piece_at(s2)
                        if p3 is not None and p3.color != side and p3.piece_type != chess.KING and dd.is_attacked_by(side, s2):
                            reads.update(exchange_reads(dd, s2))
    lnd.reads = frozenset(reads)
    return lnd


class LandingTable:
    """The memo of landing answers with the inverted read index. `ask` returns (landing, served_from_table)."""

    def __init__(self):
        self.memo: dict = {}                 # key -> (Landing, the board it was asked on)
        self.asked = 0
        self.served = 0

    def lookup(self, board: chess.Board, key: tuple) -> "Landing | None":
        hit = self.memo.get(key)
        if hit is None:
            return None
        lnd, old = hit
        if lnd.reads & R.diff_squares(old, board):
            return None
        return lnd

    def put(self, board: chess.Board, lnd: Landing) -> None:
        self.memo[lnd.key()] = (lnd, board.copy(stack=False))

    def ask(self, board: chess.Board, side: bool, frm: int, to: int) -> tuple:
        pc = board.piece_at(frm)
        key = (side, pc.symbol().lower() if pc is not None else "?", frm, to)
        self.asked += 1
        hit = self.lookup(board, key)
        if hit is not None:
            self.served += 1
            return hit, True
        lnd = evaluate(board, side, frm, to)
        self.put(board, lnd)
        return lnd, False

    def known_about(self, sq: int) -> list:
        """THE INVERTED READ INDEX: every landing of the table whose read set contains `sq`, targets first."""
        return sorted((l for l, _b in self.memo.values() if sq in l.reads), key=lambda l: (l.kind != "target", -l.value, l.key()))
