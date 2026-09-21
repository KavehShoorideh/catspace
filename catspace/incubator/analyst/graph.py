"""graph.py -- THE INTERACTION GRAPH: one explicit object per position (Kaveh 2026-09-09: "start with the graph of
the interactions, and from the rules calculate who is stuck where").

Nodes are pieces. Every node carries ANSWERS to named questions, and every ordered pair carries named RELATIONS;
the position carries named INDICATORS. All of it is computed by the four generic operators the layers already run
(the exchange on a square, counterfactual removal, the free move, shortest paths on the move graph) plus the pawn
facts from the rules -- nothing in here knows what a pin or a fork is. The names in NODE_QUESTIONS / EDGE_RELATIONS
/ POSITION_QUESTIONS are the vocabulary a learned model is allowed to see (one learned embedding per answer, one
attention bias per relation), so what a model does with them can be read back by name.

Zones are the sub-games (components of the lemma graph); a zone's TEMPERATURE is what a free move there is worth
to whichever side profits most (the hottest zone is where the action is).

    python -m catspace.incubator.analyst.graph --fen "<fen>" [--piece e4] [--time data/lichess/latest.pgn]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import chess
import chess.pgn

from . import rules as R
from .facts import value
from .nodes import tick
from .memo import Memo, changed_squares
from .defense import defense, exposure_candidates, SLIDERS
from .subgames import static_lemmas, mate_lemmas, holdings, components, cost_of_move, mate_in_one
from .regime import cone, race, pawn_path
from .threats import free_move_threats

MATE = 1000.0

NODE_QUESTIONS = (
    "value", "ours",
    "attackers", "defenders", "attackers_count", "defenders_count", "exchange", "falls", "pinned",
    "guards", "shields", "holds", "holds_value", "keeps_off_mate",
    "legal_moves", "safe_moves", "free_moves", "stuck", "trapped", "cone1", "cone2",
    "king_distance", "own_king_distance", "king_zone_hits", "attacks_up", "attacks_falling",
    "threat", "threatened",
    "pawn_rank", "passed", "to_go", "race_margin", "blockaded", "isolated", "doubled", "supported",
    "king_attackers", "king_defenders", "king_shelter", "king_open_files",
    "zone_size", "zone_temperature", "in_hottest",
)

EDGE_RELATIONS = (
    "attacks", "defends", "attack_counts", "defend_counts",
    "through", "through_value", "through_royal", "blocks",
    "holds", "wins_with", "co_attack", "co_defend",
    "reach1", "reach2", "distance", "aligned", "same_zone",
)

POSITION_QUESTIONS = (
    "in_check", "castle_ours", "castle_theirs",
    "pawns_ours", "knights_ours", "bishops_ours", "rooks_ours", "queens_ours",
    "pawns_theirs", "knights_theirs", "bishops_theirs", "rooks_theirs", "queens_theirs",
    "material_diff", "phase", "pieces",
    "queens_on", "rooks_connected_ours", "rooks_connected_theirs", "bishop_pair_ours", "bishop_pair_theirs",
    "passed_ours", "passed_theirs", "runner_ours", "runner_theirs", "race_ours", "race_theirs",
    "falls_ours", "falls_theirs", "threat_ours", "threat_theirs", "mate_ours", "mate_theirs",
    "king_attackers_ours", "king_attackers_theirs", "open_files",
    "zones", "hottest", "idle_pieces",
)


@dataclass
class Node:
    square: int
    piece: chess.Piece
    answers: dict = field(default_factory=dict)

    def vector(self) -> list[float]:
        return [float(self.answers.get(q, 0.0)) for q in NODE_QUESTIONS]


@dataclass
class Graph:
    fen: str
    side: chess.Color
    nodes: list[Node]
    edges: dict                               # (i, j) -> {relation: value}, sparse over node indices
    position: dict
    zones: list[set[int]]                     # squares per zone, hottest first
    temperatures: list[float]
    footprint: int = 0
    seconds: float = 0.0

    def index(self) -> dict[int, int]:
        return {n.square: i for i, n in enumerate(self.nodes)}

    def node_matrix(self):
        import numpy as np
        return np.array([n.vector() for n in self.nodes], dtype=np.float32).reshape(len(self.nodes), len(NODE_QUESTIONS))

    def edge_tensor(self):
        import numpy as np
        n = len(self.nodes)
        t = np.zeros((n, n, len(EDGE_RELATIONS)), dtype=np.float32)
        col = {r: k for k, r in enumerate(EDGE_RELATIONS)}
        for (i, j), rel in self.edges.items():
            for r, v in rel.items():
                t[i, j, col[r]] = v
        return t

    def position_vector(self):
        import numpy as np
        return np.array([float(self.position.get(q, 0.0)) for q in POSITION_QUESTIONS], dtype=np.float32)


# ---------------------------------------------------------------- the change list (what a move can invalidate)
def stale_entries(memo: Memo, mask: int) -> tuple[int, int]:
    """How many stored facts a change on `mask` invalidates, against how many are stored."""
    stale = total = 0
    for entries in memo.table.values():
        for e in entries:
            total += 1
            if e.mask & mask:
                stale += 1
    return stale, total


# ---------------------------------------------------------------- helpers
def _king_zone(board: chess.Board, color: chess.Color) -> int:
    k = R.royal_square(board, color)
    return 0 if k is None else (chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k])


def _attacks_from(board: chess.Board, sq: int, to: int) -> int:
    """The squares the piece on sq would attack standing on `to` (the board otherwise as it is)."""
    p = board.piece_at(sq)
    b = board.copy(stack=False); b.remove_piece_at(sq); b.set_piece_at(to, p)
    return int(b.attacks(to))


def _through(board: chess.Board, a: int) -> list[tuple[int, int]]:
    """For a slider on a: (blocker, beyond) pairs where exactly one piece stands between a and `beyond` on a's ray,
    the blocker and `beyond` being of one colour (the line relation behind pins, skewers, x-rays, shields)."""
    p = board.piece_at(a)
    if p is None or p.piece_type not in SLIDERS:
        return []
    out = []
    for t, tp in board.piece_map().items():
        if t == a or tp.color == p.color:
            continue
        ray = chess.ray(a, t)
        if not ray:
            continue
        straight = chess.square_file(a) == chess.square_file(t) or chess.square_rank(a) == chess.square_rank(t)
        if (p.piece_type == chess.ROOK and not straight) or (p.piece_type == chess.BISHOP and straight):
            continue
        between = chess.SquareSet.between(a, t) & chess.SquareSet(board.occupied)
        if len(between) != 1:
            continue
        blocker = next(iter(between))
        if board.piece_at(blocker).color == tp.color:
            out.append((blocker, t))
    return out


# ---------------------------------------------------------------- the graph
def build(board: chess.Board, memo: Memo | None = None, hops: int = 2, top_threats: int = 8) -> Graph:
    tick("graph")
    memo = memo if memo is not None else Memo()
    t0 = time.perf_counter()
    with memo.recording() as acc:
        g = _build(board, memo, hops, top_threats)
    g.footprint = acc[0]
    g.seconds = time.perf_counter() - t0
    return g


def _build(board: chess.Board, memo: Memo, hops: int, top_threats: int) -> Graph:
    us = board.turn
    squares = sorted(chess.scan_forward(board.occupied))
    idx = {sq: i for i, sq in enumerate(squares)}
    nodes = [Node(sq, board.piece_at(sq)) for sq in squares]
    # ---- the four operators
    defs = {sq: defense(board, sq, memo=memo) for sq in squares if not R.is_royal(board, sq)}
    lemmas = static_lemmas(board, memo) + mate_lemmas(board)
    held = holdings(board, memo, [l for l in lemmas if l.status != "mate"])
    cones = {sq: cone(board, sq, max_hops=hops, memo=memo) for sq in squares}
    threats = {c: free_move_threats(board, c, memo, top=top_threats) for c in (chess.WHITE, chess.BLACK)}
    mates = {c: mate_in_one(board, c) for c in (chess.WHITE, chess.BLACK)}
    races = {}
    for sq in squares:
        p = board.piece_at(sq)
        if p.piece_type == chess.PAWN and pawn_path(board, sq):
            r = race(board, sq, memo=memo)
            if r is not None:
                races[sq] = r
    # ---- zones: components of the lemma graph, temperature from the free-move ledgers
    comps = components(lemmas)
    zones = [set().union(*(l.participants for l in c)) for c in comps]
    zone_of = {}
    for zi, z in enumerate(zones):
        for sq in z:
            zone_of.setdefault(sq, zi)
    temps = [0.0] * len(zones)
    for c, ts in threats.items():
        for m, gain, why in ts:
            zi = zone_of.get(m.from_square, zone_of.get(m.to_square))
            if zi is not None:
                temps[zi] = max(temps[zi], gain)
    order = sorted(range(len(zones)), key=lambda i: (-temps[i], -len(zones[i])))
    zones = [zones[i] for i in order]; temps = [temps[i] for i in order]
    zone_of = {}
    for zi, z in enumerate(zones):
        for sq in z:
            zone_of.setdefault(sq, zi)
    # ---- per-piece threat potential and exposure from the ledgers
    threat_by = {}; threatened = {}
    for c, ts in threats.items():
        for m, gain, why in ts:
            threat_by[m.from_square] = max(threat_by.get(m.from_square, 0.0), gain)
            if board.piece_at(m.to_square) is not None:
                threatened[m.to_square] = max(threatened.get(m.to_square, 0.0), gain)
    # ---- node answers
    kz = {c: _king_zone(board, not c) for c in (chess.WHITE, chess.BLACK)}      # per colour: the ENEMY king's zone
    for n in nodes:
        sq, p, c = n.square, n.piece, n.piece.color
        a = n.answers
        d = defs.get(sq)
        a["value"] = value(board, sq); a["ours"] = float(c == us)
        a["attackers"] = len(d.attackers) if d else len(board.attackers(not c, sq))
        a["defenders"] = len(d.defenders) if d else len(board.attackers(c, sq))
        a["attackers_count"] = len(d.effective_attackers) if d else 0
        a["defenders_count"] = len(d.effective) if d else a["defenders"]
        a["exchange"] = d.see if d else 0.0
        a["falls"] = float(bool(d and d.falls))
        a["pinned"] = float(board.is_pinned(c, sq))
        own_guarded = [t for t in board.attacks(sq) if board.piece_at(t) and board.piece_at(t).color == c]
        a["guards"] = len(own_guarded)
        a["shields"] = len(exposure_candidates(board, sq, c)) - len(own_guarded)
        hs = held.get(sq, [])
        a["holds"] = len(hs); a["holds_value"] = sum(min(h.loss, MATE) for h in hs)
        a["keeps_off_mate"] = float(any(h.effect == "mated" for h in hs))
        b = board.copy(stack=False); b.turn = c
        legal = [m for m in b.legal_moves if m.from_square == sq] if b.is_valid() else []
        a["legal_moves"] = len(legal)
        cn = cones.get(sq, {})
        a["cone1"] = sum(1 for h in cn.values() if h == 1); a["cone2"] = sum(1 for h in cn.values() if 1 <= h <= 2)
        safe = [m for m in legal if not b.is_capture(m) and m.to_square in cn and cn[m.to_square] == 1]
        a["safe_moves"] = len(safe)
        free = [m for m in safe if not cost_of_move(board, m, held, memo)]
        a["free_moves"] = len(free)
        a["stuck"] = float(len(free) == 0)
        a["trapped"] = float(bool(d and d.falls) and len(safe) == 0)
        ek, ok = R.royal_square(board, not c), R.royal_square(board, c)
        a["king_distance"] = chess.square_distance(sq, ek) if ek is not None else 0
        a["own_king_distance"] = chess.square_distance(sq, ok) if ok is not None and ok != sq else 0
        a["king_zone_hits"] = chess.popcount(int(board.attacks(sq)) & kz[c])
        enemies = [t for t in board.attacks(sq) if board.piece_at(t) and board.piece_at(t).color != c]
        a["attacks_up"] = sum(1 for t in enemies if value(board, t) > a["value"] or R.is_royal(board, t))
        a["attacks_falling"] = sum(1 for t in enemies if defs.get(t) is not None and defs[t].falls)
        a["threat"] = min(threat_by.get(sq, 0.0), MATE); a["threatened"] = min(threatened.get(sq, 0.0), MATE)
        # pawns
        if p.piece_type == chess.PAWN:
            r = chess.square_rank(sq); a["pawn_rank"] = r if c == chess.WHITE else 7 - r
            f = chess.square_file(sq)
            ahead = [t for t, tp in board.piece_map().items() if tp.color != c and tp.piece_type == chess.PAWN and abs(chess.square_file(t) - f) <= 1
                     and ((chess.square_rank(t) > r) if c == chess.WHITE else (chess.square_rank(t) < r))]
            a["passed"] = float(not ahead)
            a["to_go"] = (7 - r) if c == chess.WHITE else r
            rc = races.get(sq)
            a["race_margin"] = (min(rc.catch_hops, 9) - rc.steps) if rc else 0
            stop = sq + (8 if c == chess.WHITE else -8)
            a["blockaded"] = float(0 <= stop < 64 and board.piece_at(stop) is not None)
            own_files = {chess.square_file(t) for t in board.pieces(chess.PAWN, c)}
            a["isolated"] = float(not ({f - 1, f + 1} & own_files))
            a["doubled"] = float(sum(1 for t in board.pieces(chess.PAWN, c) if chess.square_file(t) == f) >= 2)
            a["supported"] = float(any(board.piece_at(t) and board.piece_at(t).piece_type == chess.PAWN and board.piece_at(t).color == c for t in board.attackers(c, sq)))
        else:
            a["pawn_rank"] = -1
        # kings
        if R.is_royal(board, sq):
            zone = _king_zone(board, c)
            a["king_attackers"] = sum(1 for t, tp in board.piece_map().items() if tp.color != c and int(board.attacks(t)) & zone)
            a["king_defenders"] = sum(1 for t, tp in board.piece_map().items() if tp.color == c and t != sq and int(board.attacks(t)) & zone)
            f, r = chess.square_file(sq), chess.square_rank(sq)
            fwd = 1 if c == chess.WHITE else -1
            a["king_shelter"] = sum(1 for t in board.pieces(chess.PAWN, c) if abs(chess.square_file(t) - f) <= 1 and 1 <= (chess.square_rank(t) - r) * fwd <= 2)
            a["king_open_files"] = sum(1 for ff in range(max(0, f - 1), min(7, f + 1) + 1) if not any(chess.square_file(t) == ff for t in board.pieces(chess.PAWN, c)))
        zi = zone_of.get(sq)
        a["zone_size"] = len(zones[zi]) if zi is not None else 0
        a["zone_temperature"] = min(temps[zi], MATE) if zi is not None else 0.0
        a["in_hottest"] = float(zi == 0 and bool(zones))
    # ---- edges
    edges: dict = {}
    def rel(i, j, name, v=1.0):
        edges.setdefault((i, j), {})[name] = v
    for n in nodes:
        sq, p, c = n.square, n.piece, n.piece.color
        i = idx[sq]
        for t in board.attacks(sq):
            tp = board.piece_at(t)
            if tp is None:
                continue
            j = idx[t]
            rel(i, j, "defends" if tp.color == c else "attacks")
            d = defs.get(t)
            if d is not None:
                if sq in d.effective_attackers:
                    rel(i, j, "attack_counts")
                if sq in d.effective:
                    rel(i, j, "defend_counts")
        for blocker, beyond in _through(board, sq):
            rel(i, idx[blocker], "through"); rel(i, idx[blocker], "through_value", value(board, beyond))
            if R.is_royal(board, beyond):
                rel(i, idx[blocker], "through_royal")
        if p.piece_type in SLIDERS:                                       # the first piece on each of its rays blocks it
            for t in board.attacks(sq):
                if board.piece_at(t) is not None:
                    rel(idx[t], i, "blocks")
        for h in held.get(sq, []):
            tgt = h.lemma.target
            if tgt in idx:
                rel(i, idx[tgt], "holds" if h.effect in ("exposed", "mated") else "wins_with")
        cn = cones.get(sq, {})
        for s, hcount in cn.items():
            if s == sq or hcount > 2:
                continue
            for t in chess.scan_forward(_attacks_from(board, sq, s) & board.occupied):
                if t == sq:
                    continue
                key = "reach1" if hcount == 1 else "reach2"
                cur = edges.get((i, idx[t]), {})
                if key == "reach2" and "reach1" in cur:
                    continue
                rel(i, idx[t], key)
    attackers_of: dict[int, set[int]] = {}
    for (i, j), r in list(edges.items()):
        if "attacks" in r:
            attackers_of.setdefault(j, set()).add(i)
    defenders_of: dict[int, set[int]] = {}
    for (i, j), r in list(edges.items()):
        if "defends" in r:
            defenders_of.setdefault(j, set()).add(i)
    for grp, name in ((attackers_of, "co_attack"), (defenders_of, "co_defend")):
        for j, s in grp.items():
            for i1 in s:
                for i2 in s:
                    if i1 != i2:
                        rel(i1, i2, name)
    for i, n in enumerate(nodes):
        for j, m in enumerate(nodes):
            if i == j:
                continue
            dist = chess.square_distance(n.square, m.square)
            aligned = bool(chess.ray(n.square, m.square))
            zi, zj = zone_of.get(n.square), zone_of.get(m.square)
            if dist <= 2 or aligned or (i, j) in edges or (zi is not None and zi == zj):
                rel(i, j, "distance", dist)
                if aligned:
                    rel(i, j, "aligned")
                if zi is not None and zi == zj:
                    rel(i, j, "same_zone")
    # ---- the position
    pos = {}
    pos["in_check"] = float(board.is_check())
    pos["castle_ours"] = float(board.has_castling_rights(us)); pos["castle_theirs"] = float(board.has_castling_rights(not us))
    for c, tag in ((us, "ours"), (not us, "theirs")):
        for pt, nm in ((chess.PAWN, "pawns"), (chess.KNIGHT, "knights"), (chess.BISHOP, "bishops"), (chess.ROOK, "rooks"), (chess.QUEEN, "queens")):
            pos[f"{nm}_{tag}"] = len(board.pieces(pt, c))
        rooks = sorted(board.pieces(chess.ROOK, c))
        pos[f"rooks_connected_{tag}"] = float(len(rooks) >= 2 and any(rooks[b] in board.attacks(rooks[a]) for a in range(len(rooks)) for b in range(len(rooks)) if a != b))
        bs = list(board.pieces(chess.BISHOP, c))
        pos[f"bishop_pair_{tag}"] = float(len(bs) >= 2 and len({(chess.square_file(x) + chess.square_rank(x)) % 2 for x in bs}) == 2)
        passed = [n for n in nodes if n.piece.color == c and n.piece.piece_type == chess.PAWN and n.answers.get("passed")]
        pos[f"passed_{tag}"] = len(passed)
        pos[f"runner_{tag}"] = min((n.answers["to_go"] for n in passed), default=0)
        pos[f"race_{tag}"] = max((n.answers["race_margin"] for n in passed if races.get(n.square)), default=0)
        pos[f"falls_{tag}"] = sum(1 for n in nodes if n.piece.color == c and n.answers.get("falls"))
        pos[f"threat_{tag}"] = min(max((g for _, g, _ in threats[c]), default=0.0), MATE)
        pos[f"mate_{tag}"] = float(bool(mates[c]))
        k = R.royal_square(board, c)
        pos[f"king_attackers_{tag}"] = nodes[idx[k]].answers.get("king_attackers", 0) if k is not None else 0
    pos["material_diff"] = sum(value(board, s) for s in squares if board.piece_at(s).color == us and not R.is_royal(board, s)) - \
        sum(value(board, s) for s in squares if board.piece_at(s).color != us and not R.is_royal(board, s))
    pos["phase"] = sum(value(board, s) for s in squares if board.piece_at(s).piece_type not in (chess.PAWN, chess.KING))
    pos["pieces"] = len(squares)
    pos["queens_on"] = pos["queens_ours"] + pos["queens_theirs"]
    pos["open_files"] = sum(1 for f in range(8) if not any(chess.square_file(t) == f for t in board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)))
    pos["zones"] = len(zones); pos["hottest"] = min(temps[0], MATE) if temps else 0.0
    pos["idle_pieces"] = sum(1 for sq in squares if sq not in zone_of)
    return Graph(board.fen(), us, nodes, edges, pos, zones, temps)


# ---------------------------------------------------------------- prose (the product reads the same answers)
def _pc(board: chess.Board, sq: int) -> str:
    from .verbalize import pc
    return pc(board.piece_at(sq).symbol(), sq)


def node_text(g: Graph, n: Node) -> str:
    board = chess.Board(g.fen)
    a = n.answers
    bits = [f"attacked {a['attackers']:.0f} ({a['attackers_count']:.0f} count), defended {a['defenders']:.0f} ({a['defenders_count']:.0f} count)"]
    if a["falls"]:
        bits.append("falls")
    if a["pinned"]:
        bits.append("pinned")
    if a["holds"]:
        bits.append(f"holds {a['holds']:.0f} worth {a['holds_value']:g}")
    if a["keeps_off_mate"]:
        bits.append("keeps off a mate")
    bits.append(f"{a['legal_moves']:.0f} moves, {a['safe_moves']:.0f} safe, {a['free_moves']:.0f} free" + (" -- stuck" if a["stuck"] else ""))
    if a["trapped"]:
        bits.append("trapped")
    if a["threat"]:
        bits.append(f"threatens {'mate' if a['threat'] >= MATE else format(a['threat'], 'g')}")
    if a["threatened"]:
        bits.append(f"is a target worth {'mate' if a['threatened'] >= MATE else format(a['threatened'], 'g')}")
    if n.piece.piece_type == chess.PAWN and a.get("passed"):
        bits.append(f"passed, {a['to_go']:.0f} to go" + (f", race margin {a['race_margin']:+.0f}" if a["race_margin"] else ""))
    if a["zone_size"]:
        bits.append(f"zone of {a['zone_size']:.0f} at temperature {a['zone_temperature']:g}" + (" (the hottest)" if a["in_hottest"] else ""))
    else:
        bits.append("in no sub-game")
    return f"{_pc(board, n.square)}: " + "; ".join(bits)


def lines(g: Graph, top: int = 6) -> list[str]:
    board = chess.Board(g.fen)
    out = [f"{len(g.nodes)} pieces, {len(g.edges)} pairs related, {len(g.zones)} zones (temperatures {', '.join(f'{t:g}' for t in g.temperatures[:4])}), "
           f"{g.position['idle_pieces']:.0f} idle; {g.seconds:.2f}s, footprint {chess.popcount(g.footprint)} squares"]
    for zi, z in enumerate(g.zones[:3]):
        out.append(f"zone {zi + 1} ({g.temperatures[zi]:g}): " + ", ".join(_pc(board, s) for s in sorted(z)))
    ranked = sorted(g.nodes, key=lambda n: -(n.answers["holds_value"] + n.answers["threat"] + n.answers["threatened"] + 5 * n.answers["falls"]))
    for n in ranked[:top]:
        out.append(node_text(g, n))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default=""); ap.add_argument("--piece", default=""); ap.add_argument("--time", default="")
    ap.add_argument("--positions", type=int, default=40)
    a = ap.parse_args(argv)
    if a.fen:
        g = build(chess.Board(a.fen))
        for l in lines(g):
            print(l)
        if a.piece:
            sq = chess.parse_square(a.piece); n = next(n for n in g.nodes if n.square == sq)
            print(node_text(g, n))
            i = g.index()[sq]
            for (x, y), r in sorted(g.edges.items()):
                if x == i:
                    print(f"  -> {chess.square_name(g.nodes[y].square)}: {r}")
    if a.time:
        with open(a.time) as f:
            game = chess.pgn.read_game(f)
        board = game.board(); memo = Memo(); cold = Memo()
        rows = []
        for ply, mv in enumerate(list(game.mainline_moves())[:a.positions]):
            g = build(board, memo)
            h0 = memo.stats["hit"]; m0 = memo.stats["miss"]
            g2 = build(board, cold)
            stale, total = stale_entries(memo, changed_squares(board, mv))
            rows.append((ply, g.seconds, g2.seconds, stale, total, len(g.zones), g.temperatures[0] if g.temperatures else 0))
            board.push(mv)
        print(f"[graph] {len(rows)} positions: warm {sum(r[1] for r in rows) / len(rows):.2f}s each, cold {sum(r[2] for r in rows) / len(rows):.2f}s each; "
              f"facts invalidated by the move played {sum(r[3] for r in rows) / max(1, sum(r[4] for r in rows)):.0%} of those stored; {memo.report()}")
        print("ply  warm  cold  stale/stored zones hottest")
        for r in rows[::5]:
            print(f"{r[0]:>3} {r[1]:>5.2f} {r[2]:>5.2f} {r[3]:>5}/{r[4]:<6} {r[5]:>4} {r[6]:>7g}")


if __name__ == "__main__":
    main()
