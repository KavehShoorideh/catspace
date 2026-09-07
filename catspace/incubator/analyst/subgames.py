"""subgames.py -- sub-games as lemmas (Kaveh 2026-09-07): "define sub-games as an interplay of several pieces over a
few moves, and an opportunity cost for a piece as it participates in different sub-games".

A LEMMA is a goal (win the piece on a square, for one side) with its status (falls / joinable / undermine / held,
or proven / refuted / given up once the goal search has run), its value, the PIECES that take part (the target's
attackers and defenders, what a disqualified piece would expose, the joiner, the piece to remove; after a search,
every piece that moves inside the proof), and the read FOOTPRINT (the squares the computation depended on, from
the concept store's recording). Two lemmas that share a piece are coupled; a piece in several lemmas has an
OPPORTUNITY COST: what the other lemmas lose when it leaves (opportunity.py builds on this).

The combinatorial-game view (Elkies 1996, Muller 1999) is exact only when the lemmas are independent -- a sum of
games. `measure` tests that assumption on real games: how many components the interaction graph splits into,
how often the piece that captures sits in more than one lemma, what a capture does to the components.

    python -m catspace.incubator.analyst.subgames measure data/lichess/latest.pgn [--proofs --budget 120 --every 2]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass, field

import chess
import chess.pgn

from . import rules as R
from .facts import value
from .memo import Memo
from .reach import progress, Target


@dataclass
class Lemma:
    side: chess.Color
    target: int
    piece: str                       # symbol of the target piece
    status: str                      # falls | joinable | undermine | held | proven | refuted | given_up
    value: float
    participants: set[int]           # squares (at the lemma's board) of the pieces that take part, both colours
    footprint: int = 0               # read footprint (soundness), when recorded
    line: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def goal(self) -> str:
        return f"{'White' if self.side else 'Black'} wins the {chess.piece_name(chess.Piece.from_symbol(self.piece).piece_type)} on {chess.square_name(self.target)}"


# ---------------------------------------------------------------- static lemmas from the progress map
def participants(board: chess.Board, t: Target) -> set[int]:
    """The pieces the target's ledger and reach name: attackers, defenders, what a disqualified piece exposes, the
    joiner, the defender to remove and ITS ledger. The interaction neighbourhood, one to two levels deep."""
    out = {t.square}
    d = t.ledger
    if d is not None:
        out |= set(d.attackers) | set(d.defenders)
        for x in list(d.disqualified) + list(d.attackers_out):
            if x.exposed is not None:
                out.add(x.exposed)
    for p in ([t.joiner] if t.joiner else []) + list(t.joiners):
        if p is not None:
            out.add(p.piece)
    if t.remove is not None:
        out.add(t.remove)
        from .defense import defense
        dd = defense(board, t.remove)
        if dd is not None:
            out |= set(dd.attackers) | set(dd.defenders)
    return {s for s in out if board.piece_at(s) is not None}


def static_lemmas(board: chess.Board, memo: Memo | None = None, sides=(chess.WHITE, chess.BLACK)) -> list[Lemma]:
    """One lemma per target that has an interaction (anything but unreachable), for each side."""
    out = []
    for side in sides:
        for t in progress(board, side, max_hops=3, top=64, memo=memo):
            if t.status == "unreachable":
                continue
            out.append(Lemma(side, t.square, t.piece, t.status, value(board, t.square), participants(board, t), t.footprint))
    return out


# ---------------------------------------------------------------- lemmas from proofs
def proof_lemma(board: chess.Board, side: chess.Color, target: int, memo, engine, budget: int = 150, horizon: int = 6,
                engine_depth: int = 8) -> Lemma:
    """Run the goal search for one target and read the lemma off the proof: the pieces that move inside the proof
    (mapped back to their root squares) plus the target's root ledger; the footprint from the store's recording."""
    from .goal_search import GoalSearch, result_text
    from .defense import defense
    gs = GoalSearch(board, side, target, memo=memo, engine=engine, horizon=horizon, budget=budget, engine_depth=engine_depth)
    if memo is not None:
        with memo.recording() as acc:
            r = gs.run()
        fp = acc[0]
    else:
        r = gs.run(); fp = 0
    parts = {target}
    d = defense(board, target, memo=memo)
    if d is not None:
        parts |= set(d.attackers) | set(d.defenders)
    root = gs.root
    if root is not None:
        parts |= _movers(root, {sq: sq for sq in chess.scan_forward(board.occupied)}, proven_only=r.proven)
    for sq in list(parts):
        fp |= chess.BB_SQUARES[sq]
    status = "proven" if r.proven else ("refuted" if r.disproven else "given_up")
    p = board.piece_at(target)
    return Lemma(side, target, p.symbol() if p else "?", status, value(board, target), {s for s in parts if board.piece_at(s)}, fp,
                 r.line if r.proven else r.refutation, result_text(r))


def _movers(node, origin: dict[int, int], proven_only: bool) -> set[int]:
    """Root squares of every piece that moves in the (proven part of the) tree under node."""
    out = set()
    for c in node.children or []:
        if proven_only and c.pn != 0 and not node.is_or:      # inside a proof every reply counts; at OR nodes only the proven move
            continue
        if proven_only and node.is_or and c.pn != 0:
            continue
        m = c.move
        if m is None:
            continue
        if m.from_square in origin:
            out.add(origin[m.from_square])
        o2 = dict(origin)
        o2[m.to_square] = origin.get(m.from_square, m.from_square)
        o2.pop(m.from_square, None)
        out |= _movers(c, o2, proven_only)
    return out


# ---------------------------------------------------------------- coupling
def components(lemmas: list[Lemma]) -> list[list[Lemma]]:
    """Union-find over lemmas that share a participant piece: the sub-games that cannot be valued independently."""
    parent = list(range(len(lemmas)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    by_piece: dict[int, list[int]] = {}
    for i, l in enumerate(lemmas):
        for s in l.participants:
            by_piece.setdefault(s, []).append(i)
    for idx in by_piece.values():
        for j in idx[1:]:
            a, b = find(idx[0]), find(j)
            if a != b:
                parent[b] = a
    groups: dict[int, list[Lemma]] = {}
    for i, l in enumerate(lemmas):
        groups.setdefault(find(i), []).append(l)
    return sorted(groups.values(), key=lambda g: -len(g))


def multiplicity(lemmas: list[Lemma]) -> Counter:
    """piece square -> number of lemmas it takes part in."""
    c = Counter()
    for l in lemmas:
        for s in l.participants:
            c[s] += 1
    return c


# ---------------------------------------------------------------- the measurement
def measure(pgn_paths: list[str], proofs: bool = False, budget: int = 150, every: int = 1, engine_depth: int = 8, max_positions: int = 0):
    from .engine import Engine
    eng = Engine(depth=engine_depth) if proofs else None
    memo = Memo()
    t0 = time.perf_counter()
    rows = []                                    # one per position
    for path in pgn_paths:
        with open(path) as f:
            game = chess.pgn.read_game(f)
        board = game.board()
        moves = list(game.mainline_moves())
        prev = None
        for ply, mv in enumerate(moves):
            if ply % every == 0:
                lem = static_lemmas(board, memo)
                comps = components(lem)
                mult = multiplicity(lem)
                n_pieces = chess.popcount(board.occupied) - 2
                phase = sum(value(board, s) for s in chess.scan_forward(board.occupied) if not R.is_royal(board, s))
                capt = board.is_capture(mv)
                row = dict(game=path.split("/")[-1], ply=ply, pieces=n_pieces, material=phase, lemmas=len(lem),
                           components=len(comps), biggest=len(comps[0]) if comps else 0,
                           mover_mult=mult.get(mv.from_square, 0), capture=capt,
                           shared=sum(1 for s, k in mult.items() if k >= 2), pieces_in=len(mult))
                if proofs:
                    side = board.turn
                    tops = [t for t in progress(board, side, top=64, memo=memo) if t.status in ("falls", "joinable", "undermine")][:2]
                    pl = [proof_lemma(board, side, t.square, memo, eng, budget=budget, engine_depth=engine_depth) for t in tops]
                    row["proofs"] = [(l.status, len(l.participants), chess.popcount(l.footprint)) for l in pl]
                    pm = multiplicity(pl + lem)
                    row["proof_mover_mult"] = max([pm.get(mv.from_square, 0)] + [0])
                    row["proof_shared"] = sum(1 for a in pl for b in pl if a is not b and a.participants & b.participants) // 2
                rows.append(row)
                if prev is not None and prev["game"] == row["game"]:
                    row["d_components"] = row["components"] - prev["components"]
                prev = row
            board.push(mv)
            if max_positions and len(rows) >= max_positions:
                break
    dt = time.perf_counter() - t0
    if eng:
        eng.close()
    return rows, dt, memo


def report(rows, dt, memo) -> str:
    out = []
    def mean(xs):
        xs = list(xs); return sum(xs) / len(xs) if xs else 0.0
    games = sorted({r["game"] for r in rows})
    out.append(f"{len(rows)} positions from {len(games)} games in {dt:.1f}s | {memo.report()}")
    out.append(f"{'game':<16}{'pos':>4}{'lemmas':>8}{'comps':>7}{'biggest':>9}{'pieces>=2':>11}{'of':>4}{'capt':>6}{'mover>=2':>10}")
    for g in games:
        rs = [r for r in rows if r["game"] == g]
        caps = [r for r in rs if r["capture"]]
        out.append(f"{g[:16]:<16}{len(rs):>4}{mean(r['lemmas'] for r in rs):>8.1f}{mean(r['components'] for r in rs):>7.2f}"
                   f"{mean(r['biggest'] for r in rs):>9.1f}{mean(r['shared'] for r in rs):>11.1f}{mean(r['pieces_in'] for r in rs):>4.0f}"
                   f"{len(caps):>6}{(sum(1 for r in caps if r['mover_mult'] >= 2) / len(caps) if caps else 0):>10.0%}")
    # by phase (material on board)
    out.append("by material on the board (both sides, pawns=1):")
    bins = [(60, 99, "opening 60+"), (40, 60, "middlegame 40-59"), (20, 40, "late 20-39"), (0, 20, "ending <20")]
    for lo, hi, name in bins:
        rs = [r for r in rows if lo <= r["material"] < hi]
        if not rs:
            continue
        caps = [r for r in rs if r["capture"]]
        out.append(f"  {name:<18} n={len(rs):<4} components {mean(r['components'] for r in rs):.2f}  biggest {mean(r['biggest'] for r in rs):.1f} of {mean(r['lemmas'] for r in rs):.1f} lemmas"
                   f"  pieces in >=2 lemmas {mean(r['shared'] for r in rs):.1f}/{mean(r['pieces_in'] for r in rs):.0f}"
                   f"  captures whose piece is in >=2 lemmas {(sum(1 for r in caps if r['mover_mult'] >= 2) / len(caps) if caps else 0):.0%} ({len(caps)})")
    caps = [r for r in rows if r["capture"] and "d_components" in r]
    if caps:
        out.append(f"components after a capture vs before (next sampled position): mean change {mean(r['d_components'] for r in caps):+.2f}; "
                   f"split {sum(1 for r in caps if r['d_components'] > 0)}, merged {sum(1 for r in caps if r['d_components'] < 0)}, same {sum(1 for r in caps if r['d_components'] == 0)}")
    pr = [r for r in rows if "proofs" in r]
    if pr:
        st = Counter(s for r in pr for s, _, _ in r["proofs"])
        out.append(f"proof lemmas: {sum(st.values())} searches: {dict(st)}; participants per proof {mean(n for r in pr for _, n, _ in r['proofs']):.1f}, "
                   f"footprint {mean(f for r in pr for _, _, f in r['proofs']):.0f} squares; positions where the two top proofs share a piece: "
                   f"{sum(1 for r in pr if r.get('proof_shared', 0) > 0)}/{len(pr)}; captures whose piece is in a proof lemma: "
                   f"{sum(1 for r in pr if r['capture'] and r['proof_mover_mult'] >= 2)}/{sum(1 for r in pr if r['capture'])}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure"); m.add_argument("pgns", nargs="+"); m.add_argument("--proofs", action="store_true")
    m.add_argument("--budget", type=int, default=150); m.add_argument("--every", type=int, default=1); m.add_argument("--depth", type=int, default=8)
    m.add_argument("--max-positions", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        rows, dt, memo = measure(a.pgns, a.proofs, a.budget, a.every, a.depth, a.max_positions)
        print(report(rows, dt, memo))


if __name__ == "__main__":
    main()
