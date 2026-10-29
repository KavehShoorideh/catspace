"""THE PROVED BACK-RANK LEMMA IN A PUZZLE SEARCH (Kaveh 2026-10-04: "see if the theorem is useful in action ... a
puzzle ... apply it in the puzzle and cut down on search correctly ... get it end to end for one lemma first").

A plain mate search (AND/OR, full width, exact: "can the side to move force mate within n moves") on Lichess puzzles
tagged backRankMate, in three arms:
- BASE: checks first, then captures, then the rest; at the last move only checks are tried (a mate is a check).
- CLOSE: the same, but at every attacker node the Lean detector is asked first; a candidate whose hypotheses all hold
  is mate by `backRankMate_sound` -- the node is closed without expanding it.
- TARGET: CLOSE, and when exactly one hypothesis of a candidate fails, that candidate's own move and then the moves that
  touch its target squares (take the defender, move the blocker, cover the flight square, attack the interposer) are
  tried first within each group of the base order.
The search is exact in all three arms (closing uses a proved theorem; ordering never removes a move), so every arm must
find the same verdict; what changes is the work. Every closed node is audited with python-chess (the audit is not
counted as search). The bridge mode checks that python-chess and the Lean kernel agree on the legal moves of the same
positions before any number is trusted.

    python -m catspace.incubator.analyst.lean_backrank_puzzles --bridge 300 --offset 5
    python -m catspace.incubator.analyst.lean_backrank_puzzles --mate 2 --n 300 --offset 17
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import chess

from catspace.io import paths

from .lean_ideas import LeanIdeas, rank_check_possible

from catspace.incubator.analyst.discovery import lemma_store as LS  # noqa: E402

VALUE = LS.valuation().by_kind()                 # plan 76.5: the store's valuation, not a table of its own
ARMS = ("base", "close", "target")


def puzzle_rows(n: int, stride: int, offset: int, mate_in: int | None, theme: str = "backRankMate") -> list:
    """Rows i % stride == offset of the Lichess puzzle file (the repo's held-out rows excluded, as in
    frame_puzzles.fresh_rows) with `theme` and, if given, mateIn{mate_in}; the first n."""
    import csv, io, zstandard
    want = f"mateIn{mate_in}" if mate_in else None
    out = []
    with open(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst", "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % stride != offset or i % 997 == 0 or i % 97 in (49, 73):
                continue
            th = row["Themes"].split()
            if theme in th and (want is None or want in th):
                row["_row"] = i
                out.append(row)
                if len(out) >= n:
                    break
    return out


def start_position(row: dict) -> tuple[chess.Board, list[chess.Move]]:
    """The solver's position (after the opponent's first move) and the solution moves."""
    b = chess.Board(row["FEN"])
    mv = row["Moves"].split()
    b.push_uci(mv[0])
    return b, [chess.Move.from_uci(u) for u in mv[1:]]


# --- the bridge: python-chess and the Lean kernel on the same positions -------------------------------------------

def bridge(rows: list, li: LeanIdeas, log=print) -> dict:
    """Legal-move sets and checkmate verdicts compared along every puzzle line; lemma MATE claims audited."""
    pos = agree = 0
    same = total = 0
    mate_claims = mate_true = 0
    bad = []
    for row in rows:
        b = chess.Board(row["FEN"])
        for u in row["Moves"].split():
            py = {m.uci() for m in b.legal_moves}
            ln = li.legal(b, fen=True)
            lb = li.legal(b)                                     # the bitboard input must give the same kernel answer
            pos += 1
            if py == ln == lb:
                agree += 1
            elif len(bad) < 5:
                bad.append((b.fen(), sorted(py - ln), sorted(ln - py), sorted(lb ^ ln)))
            fen_verdicts = {i.move.uci(): i.status for i in li.backrank(b, fen=True)}
            for inst in li.backrank(b):
                same += fen_verdicts.get(inst.move.uci()) == inst.status
                total += 1
                if inst.status == "MATE":
                    mate_claims += 1
                    nb = b.copy(stack=False)
                    nb.push(inst.move)
                    mate_true += nb.is_checkmate()
            b.push_uci(u)
    log(f"OBS bridge positions {pos}: legal-move sets agree (python-chess, Lean from FEN, Lean from bitboards) {agree}/{pos}; "
        f"verdicts equal from FEN and bitboards {same}/{total}; lemma MATE claims {mate_claims}, "
        f"checkmate by python-chess {mate_true}/{mate_claims}")
    for fen, only_py, only_lean, bb_diff in bad:
        log(f"  DISAGREE {fen}  only python-chess {only_py}  only Lean {only_lean}  bitboards vs FEN {bb_diff}")
    return dict(positions=pos, agree=agree, mate_claims=mate_claims, mate_true=mate_true)


# --- the search -------------------------------------------------------------------------------------------------

@dataclass
class Stats:
    nodes: int = 0            # positions pushed by the search (the work the lemma should cut)
    calls: int = 0            # detector queries
    closed: int = 0           # attacker nodes closed by a proved MATE instance
    audit_bad: int = 0        # closed nodes python-chess does not confirm as checkmate (must stay 0)
    order_pushes: int = 0     # pushes spent computing target-touching moves (ordering work, not search)
    target_nodes: int = 0     # attacker nodes where a one-away target reordered the moves
    t_detector: float = 0.0
    extra: dict = field(default_factory=dict)


def _gate(b: chess.Board) -> bool:
    """The detector can only fire when the defender's king is on its home rank and we have a rook or queen."""
    k = b.king(not b.turn)
    home = 7 if b.turn == chess.WHITE else 0
    return k is not None and chess.square_rank(k) == home and bool(b.pieces(chess.ROOK, b.turn) | b.pieces(chess.QUEEN, b.turn))


def _base_order(b: chess.Board, moves: list) -> list:
    def key(m):
        chk = b.gives_check(m)
        cap = b.piece_type_at(m.to_square)
        return (0 if chk else 1, -(VALUE.get(cap, 0) if cap else -1))
    return sorted(moves, key=key)


def _touches(b: chess.Board, m: chess.Move, targets: set, st: Stats) -> bool:
    if m.to_square in targets or m.from_square in targets:
        return True
    b.push(m)
    st.order_pushes += 1
    hit = bool(chess.SquareSet(b.attacks_mask(m.to_square)) & targets)
    b.pop()
    return hit


def _reply_order(b: chess.Board, moves: list) -> list:
    def key(m):
        cap = b.piece_type_at(m.to_square)
        return (0 if cap else (1 if b.piece_type_at(m.from_square) == chess.KING else 2), -(VALUE.get(cap, 0) if cap else 0))
    return sorted(moves, key=key)


class Search:
    def __init__(self, arm: str, li: LeanIdeas | None, node_cap: int = 2_000_000):
        self.arm, self.li, self.cap = arm, li, node_cap
        self.st = Stats()

    def attack(self, b: chess.Board, n: int) -> chess.Move | None:
        """A move that forces mate within n of our moves, or None."""
        st = self.st
        targets = set()
        near = set()                     # the one-away instances' own moves: the strike that would mate if one more thing held
        # CLOSE needs only MATE instances (a rank check must exist); TARGET also reads the one-away instances
        if (self.arm == "close" and rank_check_possible(b)) or (self.arm == "target" and _gate(b)):
            t0 = time.perf_counter()
            insts = self.li.backrank(b)
            st.t_detector += time.perf_counter() - t0
            st.calls += 1
            for inst in insts:
                if inst.status == "MATE":
                    st.closed += 1
                    nb = b.copy(stack=False)
                    nb.push(inst.move)
                    if not nb.is_checkmate():
                        st.audit_bad += 1
                        st.extra.setdefault("audit_fens", []).append((b.fen(), inst.move.uci()))
                    return inst.move
            if self.arm == "target":
                for inst in insts:
                    if inst.status == "ONEAWAY":
                        targets.update(inst.targets)
                        near.add(inst.move)
        moves = list(b.legal_moves)
        if n == 1:
            moves = [m for m in moves if b.gives_check(m)]
        moves = _base_order(b, moves)
        if targets or near:
            # a tie-break inside the base order's groups (checks, captures, quiet): within each group the one-away moves
            # themselves come first (the strike that is one condition short), then the moves that touch a one-away target
            # (take it, move it, attack it); the groups keep their order
            touch = {m for m in moves if m not in near and _touches(b, m, targets, st)}
            if touch or near & set(moves):
                st.target_nodes += 1
                group = lambda m: 0 if b.gives_check(m) else (1 if b.piece_type_at(m.to_square) else 2)
                moves = sorted(moves, key=lambda m: (group(m), 0 if m in near else (1 if m in touch else 2)))
        for m in moves:
            if st.nodes >= self.cap:
                return None
            b.push(m)
            st.nodes += 1
            ok = b.is_checkmate() or (n > 1 and not b.is_stalemate() and self.defend(b, n - 1))
            b.pop()
            if ok:
                return m
        return None

    def defend(self, b: chess.Board, n: int) -> bool:
        """Every reply allows mate within n of our moves."""
        for r in _reply_order(b, list(b.legal_moves)):
            if self.st.nodes >= self.cap:
                return False
            b.push(r)
            self.st.nodes += 1
            ok = self.attack(b, n) is not None
            b.pop()
            if not ok:
                return False
        return True


def run(rows: list, mate: int, li: LeanIdeas, node_cap: int, log=print) -> list:
    out = []
    for i, row in enumerate(rows):
        b, sol = start_position(row)
        rec = dict(id=row["PuzzleId"], row=row["_row"], rating=int(row["Rating"]), solution=sol[0].uci())
        for arm in ARMS:
            s = Search(arm, li if arm != "base" else None, node_cap)
            t0 = time.perf_counter()
            mv = s.attack(b.copy(stack=False), mate)
            rec[arm] = dict(move=mv.uci() if mv else None, nodes=s.st.nodes, calls=s.st.calls, closed=s.st.closed,
                            audit_bad=s.st.audit_bad, order_pushes=s.st.order_pushes, target_nodes=s.st.target_nodes,
                            t=time.perf_counter() - t0, t_detector=s.st.t_detector, capped=s.st.nodes >= node_cap)
        out.append(rec)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(rows)} puzzles")
    return out


def summarize(res: list, mate: int, log=print) -> None:
    n = len(res)
    if not n:
        log("no puzzles"); return
    base_nodes = sum(r["base"]["nodes"] for r in res)
    for arm in ARMS:
        found = sum(r[arm]["move"] is not None for r in res)
        sol = sum(r[arm]["move"] == r["solution"] for r in res)
        same = sum(r[arm]["move"] == r["base"]["move"] for r in res)
        nodes = sum(r[arm]["nodes"] for r in res)
        t = sum(r[arm]["t"] for r in res)
        td = sum(r[arm]["t_detector"] for r in res)
        calls = sum(r[arm]["calls"] for r in res)
        closed = sum(r[arm]["closed"] for r in res)
        bad = sum(r[arm]["audit_bad"] for r in res)
        op = sum(r[arm]["order_pushes"] for r in res)
        capped = sum(r[arm]["capped"] for r in res)
        fewer = sum(r[arm]["nodes"] < r["base"]["nodes"] for r in res)
        more = sum(r[arm]["nodes"] > r["base"]["nodes"] for r in res)
        log(f"OBS mateIn{mate} {arm:6s} puzzles {n}: mate found {found}, = Lichess solution {sol}, same move as base {same}, "
            f"capped {capped}; nodes {nodes} ({nodes / max(base_nodes, 1):.3f} of base), fewer/more than base {fewer}/{more}; "
            f"detector calls {calls}, closed {closed}, audit failures {bad}, ordering pushes {op}; "
            f"time {t:.1f}s (detector {td:.1f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", type=int, default=0, help="check python-chess vs the Lean kernel on this many puzzles")
    ap.add_argument("--mate", type=int, default=2)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stride", type=int, default=97)
    ap.add_argument("--offset", type=int, default=5, help="rows i %% stride == offset; 5 = development, 17 = test")
    ap.add_argument("--node-cap", type=int, default=2_000_000)
    ap.add_argument("--dump", default=None, help="write per-puzzle results to data/derived/<name>.json")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    with LeanIdeas() as li:
        if a.bridge:
            rows = puzzle_rows(a.bridge, a.stride, a.offset, None)
            print(f"loaded {len(rows)} backRankMate puzzles (stride {a.stride}, offset {a.offset}) in {time.perf_counter() - t0:.0f}s")
            bridge(rows, li)
            return
        rows = puzzle_rows(a.n, a.stride, a.offset, a.mate)
        print(f"loaded {len(rows)} backRankMate mateIn{a.mate} puzzles (stride {a.stride}, offset {a.offset}) "
              f"in {time.perf_counter() - t0:.0f}s")
        res = run(rows, a.mate, li, a.node_cap)
    summarize(res, a.mate)
    if a.dump:
        with open(paths.derived(f"{a.dump}.json"), "w") as fh:
            json.dump(dict(args=vars(a), results=res), fh)
        print(f"wrote {paths.derived(a.dump + '.json')}")


if __name__ == "__main__":
    main()
