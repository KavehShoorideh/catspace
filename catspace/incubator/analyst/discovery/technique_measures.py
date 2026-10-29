"""PLAN 73.5 (3) THE MEASURES AND THE MINER: a technique is MINED, not written. A small library of quantities DERIVED FROM THE BASE -- each a function of (board,
winning side) built only from MOVEMENT (a king's steps; the squares a unit moves along), ATTACK (the squares a side bears on), MATERIAL (a unit's worth) and TEMPO
(the moves played) -- is read over a corpus of won games of one class (technique_corpus.py); the quantity that never rises on a winning move, drops in every game and
reaches a floor where the existing mate words bind is the class's TECHNIQUE CANDIDATE. Nothing here names an ending, a piece kind or a pattern: the class is the
signature given on the command line, the same code runs on every class.

  python -m catspace.incubator.analyst.discovery.technique_measures KRvK [--corpus <file>] [--diagnose]

THE TERMS are printed in the base vocabulary's s-expressions (base_vocabulary.py: `W` our king, `L` their king, `free` the squares we do not attack -- their king
lifted, so a line through it runs on --, `reach X S` the squares X can walk to by king steps over S, `edge`, `corner`, `count`, `and`, `not`, `dist`, `sum`,
`material_us`, `material_them`, `halfmoves`), with the library's three readings the vocabulary does not have as words: `(dist X S)` the king-step distance from X
to the nearest square of the set S (`dist` of two squares, taken over a set); `(area S)` the squares of the smallest rectangle of files and ranks that holds S;
`(product A B)`. A term's BASE READS are `lemma_store.term_reads` of its s-expression (the vocabulary's own table of which primitive reads which base word): the
library cannot hold a term reading anything else. Where the vocabulary says the same number (`bv`), the tests read it through the Rust evaluator (graph.word,
the BaseWords of words.py) and compare: the Python reading here is python-chess's move generation on the same definitions, chosen for speed (one flood fill per
position serves three atoms).

PLAN 76.1 (c) THE TERMS ARE THE RELATIONS: every symbol of the vocabulary is bound to the ontology shelf's relations (ontology.py, the entries' `vocabulary`), so an
atom's and a candidate's term reads named relations of the rule spec -- `Atom.rels` / `Term.rels` (ontology.term_relations): the region reads `cell`, `king_step`
and `attack`, the material atoms `material`, the tempo `halfmove`, whose move `tempo`. A reference only: the miner reads nothing new.

THE ATOMS (nine; the plan's list): their king's REGION `(reach L free)` -- the squares it walks to without stepping onto a square we attack, our units held still --
its size, its part off the edge, its rectangle's area; their king's distance to the nearest edge square and to the nearest corner; the kings' distance; our
material and theirs; the tempo count (the fifty-move clock: the moves played since the last capture or pawn move). THE EDGE and THE CORNER are read from movement:
the squares from which a king has fewer than eight steps, and the squares from which it has three (they are the vocabulary's `edge` and `corner`). ONE LEVEL OF
COMPOSITION: the sum and the product of every two atoms. 9 + 36 + 36 = 81 candidates (printed).

THE MINER (`mine`): over the corpus, for every candidate, at the WINNING SIDE's turns of each game: the value now and at its next turn (after our move and the
reply the table played; after the mating move, the mated position) -- a VIOLATION is a winning move after which the value ROSE; the games in which it never
strictly dropped; the FLOOR of a game (the value at the last position before the mate) and the floor's spread over the games; the entry's floor is the largest
game floor, and the FLOOR POSITIONS are the corpus positions (winning side to move) at or below it -- at each the existing mate words are asked (`mate_in_one`:
the handle's reading; `mate_in_two`: selfmine.bind_mate2, the binder's schema). A TECHNIQUE CANDIDATE has 0 violations, drops in every game, and a mate the words
find at every floor position. THE CHOICE: among the candidates, the smallest floor spread, then the fewest base reads (then the fewest atoms, the shorter term).
If there is no technique candidate nothing is relaxed: the zero-violation terms (if any) and the best by violation rate are printed, with one violation each,
and the miner stops -- that is the finding; no entry is written.

THE PER-AXIS ATOMS (plan 73.6 (2), `AXIS_ATOMS`; read by `atom_values` beside the nine, used by the 73.6 library of technique_miner.py -- 73.5's library above is
the record and unchanged): for every unit U of a three-man class (W our king, L their king, P our other piece, the vocabulary's names) its RANK DISTANCE TO AN EDGE
`(dist U (or (rank a1) (rank a8)))` and its FILE DISTANCE TO AN EDGE `(dist U (or (file a1) (file h1)))` (the king steps to the nearest square of the first or the
last rank / file: movement); for every two units U, V the RANK DISTANCE and the FILE DISTANCE between them, `(dist U (rank V))` and `(dist U (file V))` (the king
steps from U to V's rank / file: movement); and WHOSE MOVE, `turn` (0 when the winning side is to move, 1 when the other side is: TEMPO, the base's "whose move";
a reading of the library like `area`). No formula is named after an ending: the same per-axis readings for every unit and every pair, the class's units by the
vocabulary's names. Each that the vocabulary also says carries its `bv` (rankof / fileof / min / diff), read through the Rust evaluator in the tests.

THE DIAGNOSTIC (`diagnose`, `--diagnose`; NOT the plan's criterion -- it chooses nothing and writes nothing): the Lean obligation's own form read on the same
positions -- at each corpus position (winning side to move) above a measured floor, does SOME move drop the measure against EVERY reply (depth one; and, where
none does, within two of our moves: depth two), refusing a move that stalemates or leaves a unit of ours that the exchange law loses (graph.see_for); the floor
is the largest value at or below which every corpus position has a mate the words find. Read for every candidate and for every ordered LEXICOGRAPHIC pair of
atoms (drop the first, or keep it and drop the second: a composition the plan's library does not have).
"""
from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass

import chess

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.io import paths

# ---------------------------------------------------------------------------------------------------------------- the readings (movement, attack, material, tempo)

EDGE = sum(chess.BB_SQUARES[s] for s in chess.SQUARES if chess.popcount(chess.BB_KING_ATTACKS[s]) < 8)       # movement: fewer than eight king steps
CORNER = sum(chess.BB_SQUARES[s] for s in chess.SQUARES if chess.popcount(chess.BB_KING_ATTACKS[s]) == 3)    # movement: three king steps


def _attacks_from(board: chess.Board, s: int, occ: int) -> int:
    """ATTACK: the squares the unit on `s` bears on over the occupancy `occ` (python-chess's attack tables)."""
    pt = board.piece_type_at(s)
    if pt == chess.PAWN:
        return chess.BB_PAWN_ATTACKS[board.color_at(s)][s]
    if pt == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[s]
    if pt == chess.KING:
        return chess.BB_KING_ATTACKS[s]
    m = 0
    if pt in (chess.BISHOP, chess.QUEEN):
        m |= chess.BB_DIAG_ATTACKS[s][chess.BB_DIAG_MASKS[s] & occ]
    if pt in (chess.ROOK, chess.QUEEN):
        m |= chess.BB_RANK_ATTACKS[s][chess.BB_RANK_MASKS[s] & occ] | chess.BB_FILE_ATTACKS[s][chess.BB_FILE_MASKS[s] & occ]
    return m


def attacked_by(board: chess.Board, side: bool) -> int:
    """The squares `side` attacks, the other side's king lifted off the board (a line through it runs on): the vocabulary's `attacked` (vocab.rs Term::Attacked)."""
    k = board.king(not side)
    occ = board.occupied & ~(chess.BB_SQUARES[k] if k is not None else 0)
    m = 0
    for s in chess.scan_forward(board.occupied_co[side]):
        m |= _attacks_from(board, s, occ)
    return m


def region(board: chess.Board, side: bool) -> int:
    """THEIR KING'S REGION, `(reach L free)`: the squares their king walks to by king steps (movement) without stepping onto a square `side` attacks (attack), our
    units held still; their king's own square included (vocab.rs Term::Reach)."""
    k = board.king(not side)
    if k is None:
        return 0
    free = ~attacked_by(board, side) & chess.BB_ALL
    seen = chess.BB_SQUARES[k]
    frontier = [k]
    while frontier:
        x = frontier.pop()
        new = chess.BB_KING_ATTACKS[x] & free & ~seen
        seen |= new
        frontier.extend(chess.scan_forward(new))
    return seen


def dist_to(s: "int | None", target: int) -> int:
    """MOVEMENT: the king-step distance from `s` to the nearest square of the set `target` (a walk on the empty board); 0 on it."""
    if s is None or not target:
        return 0
    seen, ring, d = chess.BB_SQUARES[s], chess.BB_SQUARES[s], 0
    while not ring & target:
        nxt = 0
        for x in chess.scan_forward(ring):
            nxt |= chess.BB_KING_ATTACKS[x]
        ring = nxt & ~seen
        seen |= ring
        d += 1
    return d


def area(m: int) -> int:
    """The squares of the smallest rectangle of files and ranks that holds the set (0 for no square)."""
    if not m:
        return 0
    fs = [chess.square_file(s) for s in chess.scan_forward(m)]
    rs = [chess.square_rank(s) for s in chess.scan_forward(m)]
    return (max(fs) - min(fs) + 1) * (max(rs) - min(rs) + 1)


def material(board: chess.Board, side: bool) -> int:
    """MATERIAL: the declared worth of a side's units (graph.worth, the BaseWords' `worth`; the king 0)."""
    return sum(G.worth(board.piece_at(s)) for s in chess.scan_forward(board.occupied_co[side]))


# ---------------------------------------------------------------------------------------------------------------- the atoms and the library

@dataclass(frozen=True)
class Atom:
    key: str            # a short name (the trace's)
    sexpr: str          # the term
    gloss: str          # the term in plain words
    bv: "str | None"    # the vocabulary s-expression of the same number (read by the Rust evaluator in the tests), when the vocabulary has one

    @property
    def rels(self) -> tuple:
        """plan 76.1 (c) THE TERMS ARE THE RELATIONS: the ontology shelf's relations the atom's term reads (each symbol's binding, ontology.term_relations) --
        e.g. the region `(count (reach L free))` reads `cell` (where their king stands), `king_step` (its walk) and `attack` (the squares we bear on)."""
        from catspace.incubator.analyst.discovery import ontology as ON
        return tuple(ON.term_relations(self.sexpr))


ATOMS = (
    Atom("region", "(count (reach L free))", "the size of their king's region (the squares it walks to without stepping onto a square we attack)",
         "(count (reach L free))"),
    Atom("region_off_edge", "(count (and (reach L free) (not edge)))", "the squares of their king's region off the edge",
         "(count (and (reach L free) (not edge)))"),
    Atom("region_area", "(area (reach L free))", "the area of the smallest rectangle of files and ranks holding their king's region", None),
    Atom("edge_dist", "(dist L edge)", "their king's distance to the nearest edge square",
         "(min (min (fileof L) (diff (fileof L) 7)) (min (rankof L) (diff (rankof L) 7)))"),
    Atom("corner_dist", "(dist L corner)", "their king's distance to the nearest corner",
         "(min (min (dist L a1) (dist L h1)) (min (dist L a8) (dist L h8)))"),
    Atom("kings_dist", "(dist W L)", "the kings' distance", "(dist W L)"),
    Atom("material_us", "material_us", "our material", "material_us"),
    Atom("material_them", "material_them", "their material", "material_them"),
    Atom("tempo", "halfmoves", "the moves played since the last capture or pawn move", "halfmoves"),
)
UNITS = (("W", "our king"), ("L", "their king"), ("P", "our other piece"))      # the vocabulary's names of a three-man class's units (base_vocabulary.PRIMITIVES)


def _axis_atoms() -> tuple:
    """THE PER-AXIS ATOMS (73.6 (2); see the docstring): every unit's rank and file distance to an edge, every two units' rank and file distance, whose move."""
    out = []
    for u, g in UNITS:
        out.append(Atom(f"{u}_rank_edge", f"(dist {u} (or (rank a1) (rank a8)))", f"{g}'s rank distance to the nearest edge rank (the first or the last)",
                        f"(min (rankof {u}) (diff (rankof {u}) 7))"))
        out.append(Atom(f"{u}_file_edge", f"(dist {u} (or (file a1) (file h1)))", f"{g}'s file distance to the nearest edge file (the a-file or the h-file)",
                        f"(min (fileof {u}) (diff (fileof {u}) 7))"))
    for (u, g), (v, h) in itertools.combinations(UNITS, 2):
        out.append(Atom(f"{u}{v}_rank", f"(dist {u} (rank {v}))", f"the rank distance between {g} and {h} (the king steps from {g} to {h}'s rank)",
                        f"(diff (rankof {u}) (rankof {v}))"))
        out.append(Atom(f"{u}{v}_file", f"(dist {u} (file {v}))", f"the file distance between {g} and {h} (the king steps from {g} to {h}'s file)",
                        f"(diff (fileof {u}) (fileof {v}))"))
    out.append(Atom("turn", "turn", "whose move: 0 when the winning side is to move, 1 when the other side is", None))
    return tuple(out)


AXIS_ATOMS = _axis_atoms()
ATOM_BY_KEY = {a.key: a for a in ATOMS + AXIS_ATOMS}

# PLAN 76.4c (a) THE ATOMS AS TERMS OF THE ONE TERM LANGUAGE (terms.py: typed combinators over the ontology shelf's relations, read through the spec): every atom
# above re-expressed, each value equal to the python-chess reading here on the guard positions and the surprise shelf (tests/test_plan76_4c.py; the OBS of
# scripts/plan76_4c.py atoms). The units by the spec's placement: W (cell ? us king), L (cell ? them king), P the first of our units with our king lifted (the
# unit sort's declared order, then the location's); THE REGION reaches by the king's step only onto the squares we do not attack with their king lifted (a line
# through it runs on); THE EDGE the squares with fewer king steps than the most, THE CORNER those with the fewest; material the wcount of a VALUATION's worths.
ATOM_DEFS = (
    ("W", "(cell ? us king)"),
    ("L", "(cell ? them king)"),
    ("P", "(first (without W (cell ? us ?)))"),
    ("free", "(not (without L (attacked us ?)))"),
    ("region_set", "(reach (restrict (king_step ? ?) free) L)"),
    ("edge", "(not (extreme max all (king_step ? ?)))"),
    ("corner", "(extreme min all (king_step ? ?))"),
)
ATOM_TERMS = {
    "region": "(count region_set)",
    "region_off_edge": "(count (and region_set (not edge)))",
    "region_area": "(area region_set)",
    "edge_dist": "(dist (king_step ? ?) L edge)",
    "corner_dist": "(dist (king_step ? ?) L corner)",
    "kings_dist": "(dist (king_step ? ?) W L)",
    "material_us": "(wcount 1 material (cell ? us ?))",
    "material_them": "(wcount 1 material (cell ? them ?))",
    "tempo": "(value (halfmove ?))",
    **{f"{u}_rank_edge": f"(axisdist rank {u} (axisends rank))" for u, _g in UNITS},
    **{f"{u}_file_edge": f"(axisdist file {u} (axisends file))" for u, _g in UNITS},
    **{f"{u}{v}_rank": f"(axisdist rank {u} {v})" for (u, _g), (v, _h) in itertools.combinations(UNITS, 2)},
    **{f"{u}{v}_file": f"(axisdist file {u} {v})" for (u, _g), (v, _h) in itertools.combinations(UNITS, 2)},
    "turn": "(ind (notb (eq mover us)))",
}


def term_language():
    """(the domain of rules/chess.gdl and the shelf, the chess reader, the atoms' defined names, the atoms' terms by key) -- plan 76.4c (a)."""
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import ontology as ON
    from catspace.incubator.analyst.discovery import terms as TL
    dom = TL.Domain(GC.program(), ON.load())
    vals = {"material": GC.valuation_weights(LS.goal_kinds().valuation)}
    defs = TL.Defs()
    for name, body in ATOM_DEFS:
        defs.define(name, TL.parse(body, dom, defs, vals))
    atoms = {k: TL.parse(s, dom, defs, vals) for k, s in ATOM_TERMS.items()}
    return dom, GC.TermReader(), defs, atoms


def term_atom_values(board: chess.Board, side: bool, lang=None) -> dict:
    """plan 76.4c (a): every atom's value read through the term language (the same keys as `atom_values`)."""
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import terms as TL
    dom, reader, defs, atoms = lang or term_language()
    ctx = TL.Ctx(dom, reader, board, GC.COLOUR[side], defs)
    return {k: ctx.ev(t) for k, t in atoms.items()}
_KIND_ORDER = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)


def other_unit(board: chess.Board, side: bool) -> "int | None":
    """`P`, our other piece (the vocabulary's first-ranked non-king piece of `side`: the strongest kind, a pawn the most advanced, then the lower square), or None."""
    for k in _KIND_ORDER:
        sq = list(chess.scan_forward(board.pieces_mask(k, side)))
        if sq:
            if k == chess.PAWN:
                return max(sq, key=lambda s: ((chess.square_rank(s) if side else 7 - chess.square_rank(s)), -s))
            return min(sq)
    return None


def axis_values(board: chess.Board, side: bool) -> dict:
    """THE PER-AXIS ATOMS' values (73.6 (2)): the king steps to the nearest square of a rank / file set are the rank / file differences (a king steps one rank and
    one file at once), read here as the arithmetic (the tests compare `dist_to` and the vocabulary's evaluator); a unit not on the board reads 0 (as `onum` does)."""
    sq = dict(W=board.king(side), L=board.king(not side), P=other_unit(board, side))
    out = {}
    for u, _g in UNITS:
        s = sq[u]
        out[f"{u}_rank_edge"] = 0 if s is None else min(chess.square_rank(s), 7 - chess.square_rank(s))
        out[f"{u}_file_edge"] = 0 if s is None else min(chess.square_file(s), 7 - chess.square_file(s))
    for (u, _g), (v, _h) in itertools.combinations(UNITS, 2):
        s, t = sq[u], sq[v]
        out[f"{u}{v}_rank"] = 0 if s is None or t is None else abs(chess.square_rank(s) - chess.square_rank(t))
        out[f"{u}{v}_file"] = 0 if s is None or t is None else abs(chess.square_file(s) - chess.square_file(t))
    out["turn"] = 0 if board.turn == side else 1
    return out


def atom_values(board: chess.Board, side: bool) -> dict:
    """Every atom's value on `board` read for the winning side `side` (one flood fill serves the three region atoms); the per-axis atoms of 73.6 too."""
    reg = region(board, side)
    lk, wk = board.king(not side), board.king(side)
    return dict(region=chess.popcount(reg), region_off_edge=chess.popcount(reg & ~EDGE), region_area=area(reg), edge_dist=dist_to(lk, EDGE),
                corner_dist=dist_to(lk, CORNER), kings_dist=chess.square_distance(wk, lk) if lk is not None and wk is not None else 0,
                material_us=material(board, side), material_them=material(board, not side), tempo=board.halfmove_clock, **axis_values(board, side))


@dataclass(frozen=True)
class Term:
    sexpr: str
    op: str             # 'atom' | 'sum' | 'product'
    parts: tuple        # atom keys
    gloss: str
    bv: "str | None"    # the vocabulary s-expression of the whole number (an atom's, or the sum of two that have one), else None

    @property
    def reads(self) -> frozenset:
        """THE BASE READS: the base words the term's s-expression reads, by the vocabulary's own table (lemma_store.term_reads)."""
        return frozenset(LS.term_reads(self.sexpr))

    @property
    def rels(self) -> tuple:
        """plan 76.1 (c): the ontology shelf's relations the term reads (ontology.term_relations of its s-expression)."""
        from catspace.incubator.analyst.discovery import ontology as ON
        return tuple(ON.term_relations(self.sexpr))

    def value(self, vals: dict) -> int:
        if self.op == "atom":
            return vals[self.parts[0]]
        a, b = vals[self.parts[0]], vals[self.parts[1]]
        return a + b if self.op == "sum" else a * b

    def simplicity(self) -> tuple:
        """(how many base words it reads, how many atoms, the term's length): the fewest base reads first."""
        return (len(self.reads), len(self.parts), len(self.sexpr))


def library() -> list:
    """THE CANDIDATES: every atom, and the sum and the product of every two atoms (one level of composition)."""
    out = [Term(a.sexpr, "atom", (a.key,), a.gloss, a.bv) for a in ATOMS]
    for a, b in itertools.combinations(ATOMS, 2):
        out.append(Term(f"(sum {a.sexpr} {b.sexpr})", "sum", (a.key, b.key), f"{a.gloss} plus {b.gloss}",
                        f"(sum {a.bv} {b.bv})" if a.bv and b.bv else None))
    for a, b in itertools.combinations(ATOMS, 2):
        out.append(Term(f"(product {a.sexpr} {b.sexpr})", "product", (a.key, b.key), f"{a.gloss} times {b.gloss}", None))
    return out


_BY_SEXPR: dict = {}


def term(sexpr: str) -> "Term | None":
    """The library's term printed as `sexpr`, or None when the library has no such term."""
    if not _BY_SEXPR:
        _BY_SEXPR.update({t.sexpr: t for t in library()})
    return _BY_SEXPR.get(sexpr)


def measure(t: Term, board: chess.Board, side: bool) -> int:
    return t.value(atom_values(board, side))


# ---------------------------------------------------------------------------------------------------------------- the mate words at the floor

def mate_words(board: chess.Board) -> "str | None":
    """The existing mate words read on `board` (the side to move): 'mate_in_one' (the handle's reading), else 'mate_in_two' (selfmine.bind_mate2: the binder's
    schema, every reply met by a mate), else None."""
    from catspace.incubator.analyst.discovery import selfmine as SM
    p = G.pos_of(board)
    if p is None:
        return None
    if p.mate_in_one():
        return "mate_in_one"
    if SM.bind_mate2(None, "w" if board.turn else "b", pos=p):
        return "mate_in_two"
    return None


# ---------------------------------------------------------------------------------------------------------------- the miner

def _sequences(games: list) -> list:
    """Per game: (the winning side, [(fen, our move's SAN, atom values) at each winning-side turn], the atom values of the final position, the result)."""
    out = []
    for rows in games:
        winner = rows[0]["winner"] == "w"
        turns = [(r["fen"], r["san"], atom_values(chess.Board(r["fen"]), winner)) for r in rows if r["move"] is not None and (r["side"] == "w") == winner]
        last = rows[-1]
        out.append((winner, turns, atom_values(chess.Board(last["fen"]), winner), last["result"]))
    return out


def _mates(seqs: list) -> dict:
    """fen -> the mate word read there (or None), for every winning-side position of the corpus."""
    mates = {}
    for _w, turns, _f, _r in seqs:
        for f, _s, _v in turns:
            if f not in mates:
                mates[f] = mate_words(chess.Board(f))
                G.clear_caches()
    return mates


def score(t: Term, seqs: list, mates: dict) -> dict:
    """One candidate over the corpus (see the docstring); `example`: its first violation (game, fen, our move, the value before, at our next turn)."""
    moves = viol = never = 0
    floors, values, example = [], [], None
    for g, (winner, turns, final, result) in enumerate(seqs):
        vs = [t.value(v) for _f, _s, v in turns] + [t.value(final)]
        moves += len(turns)
        for i, (a, b) in enumerate(zip(vs, vs[1:])):
            if b > a:
                viol += 1
                if example is None:
                    example = (g, turns[i][0], turns[i][1], a, b)
        never += not any(b < a for a, b in zip(vs, vs[1:]))
        if result == "mate" and turns:
            floors.append(vs[-2])
        values += [(f, x) for (f, _s, _v), x in zip(turns, vs)]
    floor = max(floors) if floors else None
    at_floor = [f for f, x in values if floor is not None and x <= floor]
    found = sum(1 for f in at_floor if mates[f] is not None)
    return dict(term=t, games=len(seqs), moves=moves, violations=viol, never=never, floor=floor, floor_min=min(floors) if floors else None,
                spread=(max(floors) - min(floors)) if floors else None, floor_positions=len(at_floor), found=found, example=example,
                words=sorted({mates[f] for f in at_floor if mates[f] is not None}),
                candidate=viol == 0 and never == 0 and floor is not None and found == len(at_floor))


def _line(sig: str, rank: int, s: dict) -> str:
    t = s["term"]
    rate = s["violations"] / s["moves"] if s["moves"] else 0.0
    return (f"OBS miner {sig} #{rank} {t.sexpr}: games {s['games']}, winning moves {s['moves']}, violations {s['violations']} ({rate:.3f}), games never dropped {s['never']}, "
            f"floor {s['floor']} (game floors {s['floor_min']}..{s['floor']}, spread {s['spread']}), floor positions {s['floor_positions']}, mate found {s['found']}"
            f"{' (' + ', '.join(s['words']) + ')' if s['words'] else ''}; reads {', '.join(sorted(t.reads))}" + ("; TECHNIQUE CANDIDATE" if s["candidate"] else ""))


def _example(sig: str, s: dict) -> str:
    g, fen, san, a, b = s["example"]
    return f"OBS miner {sig} violation of {s['term'].sexpr}: game {g}, {fen}: after {san} and the table's reply it reads {b} (was {a})"


def mine(sig: str, corpus: str, top: int = 20, log=print) -> dict:
    """THE MINER over a corpus file -> {sig, corpus, scores (ranked), choice (a score or None), finding (a string or None), candidates}; prints the OBS lines."""
    t0 = time.monotonic()
    TC.parse_signature(sig)
    seqs = _sequences(TC.read_corpus(corpus))
    lib = library()
    for t in lib:
        bad = t.reads - set(LS.BASE)
        assert t.reads and not bad, f"{t.sexpr} reads {sorted(t.reads)}: a measure is built from the base only"
    mates = _mates(seqs)
    turns = [f for _w, ts, _f, _r in seqs for f, _s, _v in ts]
    log(f"OBS miner {sig} library: candidates {len(lib)} (atoms {sum(t.op == 'atom' for t in lib)}, sums {sum(t.op == 'sum' for t in lib)}, products "
        f"{sum(t.op == 'product' for t in lib)}); base reads of the atoms: " + "; ".join(f"{a.sexpr} {', '.join(sorted(LS.term_reads(a.sexpr)))}" for a in ATOMS))
    log(f"OBS miner {sig} corpus {corpus}: games {len(seqs)}, winning-side positions {len(turns)}, with a mate the words find {sum(mates[f] is not None for f in turns)} "
        f"(mate_in_one {sum(mates[f] == 'mate_in_one' for f in turns)}, mate_in_two {sum(mates[f] == 'mate_in_two' for f in turns)})")
    scores = [score(t, seqs, mates) for t in lib]
    scores.sort(key=lambda s: (s["violations"], s["never"], s["floor_positions"] - s["found"], s["spread"] if s["spread"] is not None else 99, s["term"].simplicity()))
    for i, s in enumerate(scores[:top], 1):
        log(_line(sig, i, s))
    zero = [s for s in scores if s["violations"] == 0]
    cands = [s for s in zero if s["candidate"]]
    choice, finding = None, None
    if cands:
        choice = min(cands, key=lambda s: (s["spread"], s["term"].simplicity()))
        t = choice["term"]
        log(f"OBS miner {sig} choice: {t.sexpr} -- {t.gloss}; floor {choice['floor']} (spread {choice['spread']}), terminal {' / '.join(choice['words'])}; "
            f"{len(cands)} technique candidate(s); reads {', '.join(sorted(t.reads))}; {time.monotonic() - t0:.1f} s")
    else:
        if zero:
            log(f"OBS miner {sig} zero violations: {len(zero)} term(s), none a technique candidate: " + "; ".join(
                f"{s['term'].sexpr} (games never dropped {s['never']}, floor positions {s['floor_positions']}, mate found {s['found']})" for s in zero))
        moving = sorted((s for s in scores if s["never"] == 0), key=lambda s: (s["violations"] / max(1, s["moves"]), s["term"].simplicity()))[:5]
        log(f"OBS miner {sig} best by violation rate of the terms that drop in every game: "
            + "; ".join(f"{s['term'].sexpr} {s['violations']}/{s['moves']} ({s['violations'] / max(1, s['moves']):.3f})" for s in moving))
        for s in moving[:3]:
            if s["example"] is not None:
                log(_example(sig, s))
        finding = (f"no technique candidate among the {len(lib)}: " + (f"the {len(zero)} with 0 violations never drop in a game (constants of the class); " if zero else
                   "none has 0 violations; ") + (f"every term that drops in every game rises on a winning move of the table's play (the best {moving[0]['term'].sexpr}: "
                   f"{moving[0]['violations']} of {moving[0]['moves']})" if moving else "no term drops in every game"))
        log(f"OBS miner {sig}: NO TECHNIQUE CANDIDATE -- {finding}; nothing is relaxed and no entry is written; {time.monotonic() - t0:.1f} s")
    return dict(sig=sig, corpus=corpus, scores=scores, choice=choice, finding=finding, candidates=len(cands))


# ---------------------------------------------------------------------------------------------------------------- the diagnostic (not the plan's criterion)

def _children(board: chess.Board, side: bool, key) -> list:
    """Our legal moves on `board` under the never-rules, each with the keys after every reply: [(move, 'mate' | [key after each reply])]; a move that stalemates
    or leaves a unit of ours the exchange law loses (graph.see_for for the side to move after it) is not listed."""
    out = []
    for m in list(board.legal_moves):
        board.push(m)
        try:
            if board.is_checkmate():
                out.append((m, "mate"))
            elif board.is_stalemate() or any(G.see_for(board, s, board.turn) > 0 for s in chess.scan_forward(board.occupied_co[side] & ~board.kings)):
                continue
            else:
                reps = []
                for r in list(board.legal_moves):
                    board.push(r)
                    reps.append(key(board))
                    board.pop()
                out.append((m, reps))
        finally:
            board.pop()
    return out


def _drops(kids: list, now) -> bool:
    return any(r == "mate" or max(r) < now for _m, r in kids)


def _drops2(board: chess.Board, side: bool, key, now) -> bool:
    """Depth two: a move (never-rules) after which, for every reply, the key is already below `now` or a depth-one drop below `now` exists (or a mate)."""
    for m in list(board.legal_moves):
        board.push(m)
        try:
            if board.is_checkmate():
                return True
            if board.is_stalemate() or any(G.see_for(board, s, board.turn) > 0 for s in chess.scan_forward(board.occupied_co[side] & ~board.kings)):
                continue
            ok = True
            for r in list(board.legal_moves):
                board.push(r)
                try:
                    if not (key(board) < now or _drops(_children(board, side, key), now)):
                        ok = False
                finally:
                    board.pop()
                if not ok:
                    break
            if ok:
                return True
        finally:
            board.pop()
    return False


def diagnose(sig: str, corpus: str, top: int = 5, depth2: int = 3, log=print) -> dict:
    """THE DIAGNOSTIC (see the docstring): -> {singles, pairs}: per measure (positions above its floor, how many have no depth-one drop, the floor); prints OBS."""
    t0 = time.monotonic()
    seqs = _sequences(TC.read_corpus(corpus))
    mates = _mates(seqs)
    vals = lambda b, w: atom_values(b, w)
    pos = []                                                       # (fen, winner, atom values, [(move, 'mate' | [atom values after each reply])])
    for winner, turns, _f, _r in seqs:
        for f, _s, v in turns:
            pos.append((f, winner, v, _children(chess.Board(f), winner, lambda b, w=winner: vals(b, w))))
    lib = library()
    pairs = [(a, b) for a, b in itertools.permutations([x.key for x in ATOMS], 2)]

    def read(key) -> tuple:
        unbound = [key(v) for f, _w, v, _k in pos if mates[f] is None]
        floor = min(unbound) if unbound else None                   # the positions strictly below the smallest unbound value are at the floor
        above = [(f, w, v, k) for f, w, v, k in pos if floor is None or not key(v) < floor]
        fails = [(f, w) for f, w, v, k in above if not any(r == "mate" or max(key(x) for x in r) < key(v) for _m, r in k)]
        return len(above), fails, floor
    rows = []
    for t in lib:
        n, fails, floor = read(lambda v, t=t: t.value(v))
        rows.append(("single", t.sexpr, n, fails, floor, lambda v, t=t: t.value(v), t.simplicity()))
    for a, b in pairs:
        n, fails, floor = read(lambda v, a=a, b=b: (v[a], v[b]))
        rows.append(("pair", f"({ATOM_BY_KEY[a].sexpr}, then {ATOM_BY_KEY[b].sexpr})", n, fails, floor, lambda v, a=a, b=b: (v[a], v[b]), (0,)))
    out = {}
    for kind in ("single", "pair"):
        rs = sorted((r for r in rows if r[0] == kind), key=lambda r: (len(r[3]), r[6], r[1]))
        out[kind] = [(r[1], r[2], len(r[3]), r[4]) for r in rs]
        for r in rs[:top]:
            log(f"OBS miner {sig} DIAGNOSTIC depth one, {'one term' if kind == 'single' else 'lexicographic pair'} {r[1]}: positions above the floor {r[2]}, "
                f"with no measure-dropping move {len(r[3])} (floor: below {r[4]})")
        for r in rs[:depth2]:
            key = r[5]
            still = [f for f, w in r[3] if not _drops2(chess.Board(f), w, lambda bb, w=w, key=key: key(atom_values(bb, w)), key(atom_values(chess.Board(f), w)))]
            G.clear_caches()
            log(f"OBS miner {sig} DIAGNOSTIC depth two, {r[1]}: of the {len(r[3])} positions with no depth-one drop, {len(r[3]) - len(still)} drop within two of our moves, "
                f"{len(still)} do not" + (f" (e.g. {still[0]})" if still else ""))
    log(f"OBS miner {sig} DIAGNOSTIC: the Lean obligation's form (a measure-dropping move against every reply) read on {len(pos)} corpus positions, "
        f"{len(lib)} terms and {len(pairs)} lexicographic pairs; NOT the plan's criterion: nothing is chosen or written; {time.monotonic() - t0:.1f} s")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("signature")
    ap.add_argument("--corpus", default=None, help="the corpus file (default: artifacts/experiments/technique_corpus/<sig>_seed1.jsonl)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--diagnose", action="store_true", help="also read the diagnostic (not the plan's criterion)")
    a = ap.parse_args(argv)
    corpus = a.corpus or paths.experiment("technique_corpus", f"{a.signature}_seed1.jsonl")
    mine(a.signature, corpus, a.top)
    if a.diagnose:
        diagnose(a.signature, corpus)


if __name__ == "__main__":
    main()
