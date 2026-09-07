"""regime.py -- the board as a SUM OF SUB-GAMES, when it earns it (Kaveh 2026-09-07: "the rook is out of commission,
leave that side as it is; on the other side I am a pawn up and I know I win; the king cannot leave because the
pawn would queen -- every piece has a job in its graph neighbourhood").

The default is one game. A decomposition is an inference: lemmas (subgames.py) are coupled when they share a piece
that must MOVE for one of them, or when a participant's LIGHT CONE -- the squares it can reach safely within the
horizon, shrunk by what it holds and by the races it must win (a king stays inside the square of the passed pawn)
-- meets the other's footprint. Components of that coupling graph are the sub-games. A component is FROZEN when no
move inside it by either side gains (a local fixpoint over its own pieces; the null moves are its tempo reservoir,
Elkies' numbers); a component with five men or fewer once the frozen pieces are removed is SOLVED by the syzygy
tables; otherwise it is the live game and the existing machinery runs on it. The claim is stamped with its horizon
and checked twice: a proof that leaks outside its component refutes the split, and the engine judges the whole.

    python -m catspace.incubator.analyst.regime --fen "<fen>" [--engine]
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field

import chess
import chess.syzygy as syzygy

from . import rules as R
from .defense import defense
from .facts import value, material
from .memo import Memo
from .subgames import Lemma, static_lemmas, holdings, cost_of_move, Holding, MATE_VALUE

PROMOTION_VALUE = 8.0            # a queen for a pawn


# ---------------------------------------------------------------- light cones
def _quiet_moves(board: chess.Board, sq: int) -> list[chess.Move]:
    b = board.copy(stack=False); b.turn = board.piece_at(sq).color
    try:
        return [m for m in b.legal_moves if m.from_square == sq and not b.is_capture(m) and m.promotion in (None, chess.QUEEN)]
    except Exception:
        return []


def cone(board: chess.Board, sq: int, max_hops: int = 3, memo=None, allowed=None) -> dict[int, int]:
    """Squares the piece on sq can reach safely in at most max_hops of its own quiet moves, with the hop count; the
    piece's own square at 0. Safe = the defense operator says it does not fall there (a king: the move is legal).
    `allowed(board_after, frm, to) -> bool` prunes squares the piece cannot afford (its holdings, its races)."""
    p = board.piece_at(sq)
    if p is None:
        return {}
    out = {sq: 0}
    q = deque([(board.copy(stack=False), sq, 0)])
    royal = R.is_royal(board, sq)
    while q:
        b, cur, h = q.popleft()
        if h >= max_hops:
            continue
        for m in _quiet_moves(b, cur):
            to = m.to_square
            if to in out:
                continue
            nb = b.copy(stack=False); nb.turn = p.color; nb.push(m)
            if not royal:
                d = defense(nb, to, memo=memo)
                if d is not None and d.falls:
                    continue
            if allowed is not None and not allowed(nb, cur, to):
                continue
            out[to] = h + 1
            q.append((nb, to, h + 1))
    return out


# ---------------------------------------------------------------- races: the passed pawn and who catches it
@dataclass
class Race:
    pawn: int
    path: tuple[int, ...]            # squares to the promotion square
    steps: int                       # own moves to promote
    catcher: int | None              # the fastest enemy piece
    catch_hops: int                  # its hops to a square on the path (or attacking the promotion square)
    promotes: bool                   # with the side to move as it stands


def pawn_path(board: chess.Board, sq: int) -> tuple[int, ...] | None:
    p = board.piece_at(sq)
    if p is None or p.piece_type != chess.PAWN:
        return None
    step = 8 if p.color == chess.WHITE else -8
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for esq, ep in board.piece_map().items():                  # passed: no enemy pawn ahead on its file or the neighbours
        if ep.color != p.color and ep.piece_type == chess.PAWN and abs(chess.square_file(esq) - f) <= 1:
            if (chess.square_rank(esq) > r) if p.color == chess.WHITE else (chess.square_rank(esq) < r):
                return None
    path = []
    s = sq + step
    while 0 <= s < 64:
        if board.piece_at(s) is not None:
            return None                                        # blocked: not a runner (a blockade is a different lemma)
        path.append(s)
        if chess.square_rank(s) in (0, 7):
            break
        s += step
    return tuple(path)


def race(board: chess.Board, pawn_sq: int, memo=None, max_hops: int = 7) -> Race | None:
    """The rule of the square by computation: the pawn's steps to promote against the fastest enemy piece that can
    reach its path (or attack the promotion square) safely, with the tempo of the side to move."""
    path = pawn_path(board, pawn_sq)
    if not path:
        return None
    p = board.piece_at(pawn_sq)
    steps = len(path)
    start_rank = 1 if p.color == chess.WHITE else 6
    if chess.square_rank(pawn_sq) == start_rank and len(path) >= 2:
        steps -= 1                                             # the double step
    targets = set(path)
    best, best_sq = None, None
    for esq, ep in board.piece_map().items():
        if ep.color == p.color:
            continue
        c = cone(board, esq, max_hops=min(max_hops, steps + 1), memo=memo)
        for s, h in c.items():
            hits = (s in targets) or (path[-1] in board.attacks(s) if board.piece_at(s) is not None else False)
            if not hits:
                b2 = board.copy(stack=False); b2.set_piece_at(s, ep)
                if s != esq:
                    b2.remove_piece_at(esq)
                hits = bool(b2.attacks(s) & (targets | {path[-1]}))
            if hits and (best is None or h < best):
                best, best_sq = h, esq
    if best is None:
        return Race(pawn_sq, path, steps, None, 99, True)
    pawn_to_move = board.turn == p.color
    promotes = (best >= steps) if pawn_to_move else (best > steps)
    return Race(pawn_sq, path, steps, best_sq, best, promotes)


def promotion_lemmas(board: chess.Board, memo=None) -> list[Lemma]:
    """One lemma per passed runner: for its owner, worth a queen less a pawn; participants = the pawn, the pieces that
    can catch it in time (their job is to stay in the square), and the owner's pieces whose attacks cover the path."""
    out = []
    for sq, p in board.piece_map().items():
        if p.piece_type != chess.PAWN:
            continue
        r = race(board, sq, memo)
        if r is None:
            continue
        parts = {sq}
        if r.catcher is not None:
            parts.add(r.catcher)
        for esq, ep in board.piece_map().items():
            if ep.color != p.color and esq != r.catcher:
                c = cone(board, esq, max_hops=min(6, r.steps + 1), memo=memo)
                if any(s in c for s in r.path):
                    parts.add(esq)                              # it can also get there in time: it shares the job
            elif ep.color == p.color and esq != sq and board.attacks(esq) & set(r.path):
                parts.add(esq)                                  # it covers the path
        status = "falls" if r.promotes else "held"
        l = Lemma(p.color, sq, p.symbol(), status, PROMOTION_VALUE, parts, 0, [chess.square_name(s) for s in r.path], "promotion")
        l.kind = "promotion"; l.race = r
        out.append(l)
    return out


# ---------------------------------------------------------------- binding: what a piece cannot afford
def bound_cone(board: chess.Board, sq: int, held: dict[int, list[Holding]], races: list[Race], max_hops: int = 3, memo=None) -> dict[int, int]:
    """The piece's cone minus the squares it cannot afford: a first step that breaks a holding worth a pawn or more,
    and for a catcher, any square from which it no longer catches the runner (it stays inside the square)."""
    mine = [r for r in races if r.catcher == sq]
    bad_first = set()
    if held.get(sq):
        for m in _quiet_moves(board, sq):
            if sum(h.loss for h in cost_of_move(board, m, held, memo)) >= 1.0:
                bad_first.add(m.to_square)
    def allowed(nb: chess.Board, frm: int, to: int) -> bool:
        if frm == sq and to in bad_first:
            return False
        for r in mine:                                          # still in time from there?
            if nb.piece_at(r.pawn) is None:
                continue
            rr = race(nb, r.pawn, memo)
            if rr is not None and rr.promotes and not r.promotes:
                return False
        return True
    return cone(board, sq, max_hops, memo, allowed)


# ---------------------------------------------------------------- the coupling graph and its components
@dataclass
class SubGame:
    lemmas: list[Lemma]
    pieces: set[int]                                            # participants of its lemmas
    frozen: bool = False
    reservoir: dict = field(default_factory=dict)               # null moves per side when frozen: {"w": n, "b": n}
    losing: dict = field(default_factory=dict)                  # per side: the moves inside that lose material (SAN)
    solved: str = ""                                            # tablebase verdict for the side to move ("win" / "draw" / "loss"), when probed
    men: int = 0
    shared: set[int] = field(default_factory=set)               # pieces that serve another sub-game too (static jobs in both)
    text: str = ""


def movers(board: chess.Board, l: Lemma) -> set[int]:
    """The participants that would have to MOVE for the lemma to be cashed: the capturers of a falling target, the
    joiner, the runner. Holders do not couple."""
    out = set()
    if getattr(l, "kind", "") == "promotion":
        return {l.target}
    if l.status == "falls":
        out |= {a for a in board.attackers(l.side, l.target)}
    if l.status == "joinable":
        out |= {s for s in l.participants if board.piece_at(s) and board.piece_at(s).color == l.side and s != l.target}
    return out


def coupling(board: chess.Board, lemmas: list[Lemma], cones: dict[int, dict[int, int]]) -> list[list[int]]:
    """Components over lemma indices: coupled when they share a piece that moves for one of them, or when a mover
    of one can reach (its bound cone) a participant's square or attack a participant of the other."""
    n = len(lemmas)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    mv = [movers(board, l) for l in lemmas]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = lemmas[i], lemmas[j]
            shared = a.participants & b.participants
            coupled = any(s in mv[i] or s in mv[j] for s in shared) or a.target in b.participants or b.target in a.participants
            if not coupled:
                for s in mv[i]:
                    c = cones.get(s, {})
                    if any(t in c for t in b.participants if t != s):
                        coupled = True; break
                    for sq in c:
                        if sq == s:
                            continue
                        b2 = board.copy(stack=False); b2.remove_piece_at(s); b2.set_piece_at(sq, board.piece_at(s))
                        if b2.attacks(sq) & b.participants:
                            coupled = True; break
                    if coupled:
                        break
            if not coupled:
                for s in mv[j]:
                    c = cones.get(s, {})
                    if any(t in c for t in a.participants if t != s):
                        coupled = True; break
            if coupled:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ---------------------------------------------------------------- frozen: the local fixpoint
def region_moves(board: chess.Board, pieces: set[int], side: chess.Color) -> list[chess.Move]:
    b = board.copy(stack=False); b.turn = side
    if b.is_check():
        return []
    return [m for m in b.legal_moves if m.from_square in pieces and (not b.is_capture(m) or m.to_square in pieces)]


def _best_capture(board: chess.Board, side: chess.Color, pieces: set[int], memo=None) -> float:
    """The best profitable capture `side` has on the region's pieces right now, by the ledger (the attackers that
    count only), or 0."""
    best = 0.0
    b = board.copy(stack=False); b.turn = side
    if b.is_check():
        return 0.0
    for s in pieces:
        p = board.piece_at(s)
        if p is None or p.color == side or R.is_royal(board, s):
            continue
        d = defense(b, s, memo=memo)
        if d is not None and d.falls and d.see > best:
            best = d.see
    return best


def capture_gain(board: chess.Board, m: chess.Move, side: chess.Color, memo=None) -> float:
    """What the capture m wins by the ledger when THIS piece starts the exchange: a capturer that does not count
    (its departure hangs something, or the capture is illegal) loses that instead; otherwise the swap list from it."""
    from .facts import see
    b = board.copy(stack=False); b.turn = side
    d = defense(b, m.to_square, memo=memo)
    if d is None:
        return value(b, m.to_square)
    for x in d.attackers_out:
        if x.square == m.from_square:
            return value(b, m.to_square) - (x.cost if x.reason == "exposes" else value(b, m.from_square))
    others = {a for a in d.attackers if a != m.from_square}
    return see(b, m.to_square, side, exclude=others | {x.square for x in d.disqualified})


def _gain(board: chess.Board, m: chess.Move, side: chess.Color, pieces: set[int], memo=None) -> float:
    """What the move wins inside the region by the ledger, one reply deep: the capture's exchange value (a capturer
    that does not count -- its departure hangs something -- loses instead); then, with the other side to move, their
    best profitable capture is subtracted (they take it); if they have none, the enemy region pieces that newly fall
    count, less the most valuable one (they save it)."""
    b = board.copy(stack=False); b.turn = side
    g = 0.0
    if b.is_capture(m):
        g += capture_gain(b, m, side, memo)
    nb = b.copy(stack=False); nb.push(m)
    pcs = {(m.to_square if s == m.from_square else s) for s in pieces} | {m.to_square}
    theirs = _best_capture(nb, not side, pcs, memo)
    if theirs > 0:
        return g - theirs
    new_falls = []
    for s in pcs:
        if s == m.to_square or nb.piece_at(s) is None or nb.piece_at(s).color == side:
            continue
        d0 = defense(b, s, memo=memo) if b.piece_at(s) else None
        d1 = defense(nb, s, memo=memo)
        if (d1 is not None and d1.falls) and not (d0 is not None and d0.falls):
            new_falls.append(value(nb, s))
    if new_falls:
        g += sum(new_falls) - max(new_falls)
    return g


def frozen(board: chess.Board, pieces: set[int], depth: int = 2, memo=None) -> tuple[bool, dict, dict]:
    """No move by either side inside the region gains, to `depth` null moves: the region is a constant. Returns
    (frozen, null moves per side at the root = the tempo reservoir, losing moves per side)."""
    reservoir = {"w": 0, "b": 0}; losing = {"w": [], "b": []}
    def check(b: chess.Board, pcs: set[int], d: int, top: bool, sides) -> bool:
        for side in sides:                                     # at the root either side may be to move; below, only the other one
            key = "w" if side else "b"
            for m in region_moves(b, pcs, side):
                g = _gain(b, m, side, pcs, memo)
                if g > 0.5:
                    return False
                nb = b.copy(stack=False); nb.turn = side; nb.push(m)
                if g < -0.5:
                    if top:
                        losing[key].append(b.copy(stack=False).san(m) if b.turn == side else _san(b, side, m))
                    continue
                # a null move: does it hand the OTHER side a gain next?
                pcs2 = {(m.to_square if s == m.from_square else s) for s in pcs if nb.piece_at(m.to_square if s == m.from_square else s) is not None}
                other_gain = max([_gain(nb, m2, not side, pcs2, memo) for m2 in region_moves(nb, pcs2, not side)] + [0.0])
                if other_gain > 0.5:
                    if top:
                        losing[key].append(_san(b, side, m))
                    continue
                if top:
                    reservoir[key] += 1
                if d > 1 and not check(nb, pcs2, d - 1, False, (not side,)):
                    return False
        return True
    ok = check(board, set(pieces), depth, True, (chess.WHITE, chess.BLACK))
    return ok, reservoir, losing


def _san(board: chess.Board, side: chess.Color, m: chess.Move) -> str:
    b = board.copy(stack=False); b.turn = side
    return b.san(m)


# ---------------------------------------------------------------- solved: the tablebase on the abstracted remainder
def tablebase_verdict(board: chess.Board) -> str | None:
    """Syzygy WDL for the side to move, when the tables cover the position (five men or fewer here)."""
    if chess.popcount(board.occupied) > 5 or not board.is_valid():
        return None
    try:
        from ...io import paths
        tb = syzygy.open_tablebase(str(paths.syzygy_dir()))
    except Exception:
        return None
    try:
        w = tb.probe_wdl(board)
    except Exception:
        return None
    finally:
        tb.close()
    return {2: "win", 1: "win (cursed)", 0: "draw", -1: "loss (blessed)", -2: "loss"}.get(w)


def abstracted(board: chess.Board, keep: set[int]) -> chess.Board:
    """The board with only `keep` (plus both kings), castling rights cleaned, the same side to move."""
    b = chess.Board(None)
    b.turn = board.turn
    for sq, p in board.piece_map().items():
        if sq in keep or p.piece_type == chess.KING:
            b.set_piece_at(sq, p)
    b.castling_rights = b.clean_castling_rights()
    return b


# ---------------------------------------------------------------- the decomposition
@dataclass
class Decomposition:
    subgames: list[SubGame]
    lemmas: list[Lemma]
    cones: dict[int, dict[int, int]]
    races: list[Race]
    horizon: int
    verdict: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def split(self) -> bool:
        return len([g for g in self.subgames if g.lemmas]) > 1


def decompose(board: chess.Board, memo: Memo | None = None, horizon: int = 3, engine=None) -> Decomposition:
    memo = memo if memo is not None else Memo()
    lemmas = [l for l in static_lemmas(board, memo) if l.status != "quiet"]
    proms = promotion_lemmas(board, memo)
    lemmas += proms
    races = [l.race for l in proms]
    held = holdings(board, memo, lemmas)
    # trapped pieces: no safe square within the horizon and no safe capture -> a lemma of their own (out of play)
    cones = {}
    for sq, p in board.piece_map().items():
        if R.is_royal(board, sq) and p.piece_type == chess.KING:
            cones[sq] = bound_cone(board, sq, held, races, horizon, memo)
        elif not R.is_royal(board, sq):
            cones[sq] = bound_cone(board, sq, held, races, horizon, memo)
    for sq, p in board.piece_map().items():
        if p.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) and len(cones.get(sq, {})) <= 1:
            b = board.copy(stack=False); b.turn = p.color
            caps = [m for m in b.legal_moves if m.from_square == sq and b.is_capture(m)]
            if all(capture_gain(b, m, p.color, memo) <= 0 for m in caps):
                l = Lemma(not p.color, sq, p.symbol(), "trapped", value(board, sq), {sq} | {a for a in board.attackers(not p.color, sq)}
                          | {s for m in caps for s in (m.to_square,)}, 0, [], "out of play")
                l.kind = "trapped"
                lemmas.append(l)
    comps = coupling(board, lemmas, cones)
    subgames = []
    for idx in comps:
        ls = [lemmas[i] for i in idx]
        pieces = set().union(*[l.participants for l in ls])
        g = SubGame(ls, pieces)
        live = any(getattr(l, "kind", "") == "promotion" for l in ls) or any(board.piece_at(s) and board.piece_at(s).piece_type == chess.KING for s in pieces)
        if not live:                                           # a runner or a king inside: never a constant
            g.frozen, g.reservoir, g.losing = frozen(board, pieces, depth=2, memo=memo)
        subgames.append(g)
    # kings join the sub-game whose pieces their bound cone touches (adjacent counts: a king next to a pawn guards it)
    for ksq in (board.king(chess.WHITE), board.king(chess.BLACK)):
        if ksq is None:
            continue
        c = set(cones.get(ksq, {ksq: 0}))
        home = next((g for g in subgames if ksq in g.pieces), None)
        if home is None:
            for g in subgames:
                if any(chess.square_distance(s, q) <= 1 for s in g.pieces for q in c):
                    home = g; break
        if home is not None:
            home.pieces.add(ksq)
            home.frozen = False                                # a king inside: the live game
    for g in subgames:
        g.shared = {s for s in g.pieces if any(s in h.pieces for h in subgames if h is not g)}
    # the live remainder: its own participants (shared pieces included: they serve here too), the frozen rest set aside
    for g in subgames:
        if g.frozen:
            continue
        ab = abstracted(board, set(g.pieces))
        g.men = chess.popcount(ab.occupied)
        if any(h.frozen for h in subgames):                   # a sum earns the abstraction...
            v = tablebase_verdict(ab)
        else:                                                  # ...one game is judged as it stands (exact when the tables cover it)
            v = tablebase_verdict(board)
            g.men = chess.popcount(board.occupied)
        if v:
            g.solved = v
    dec = Decomposition(subgames, lemmas, cones, races, horizon)
    live = [g for g in subgames if not g.frozen and g.lemmas]
    if len(live) == 1 and live[0].solved:
        dec.verdict = (f"{live[0].solved} for {'White' if board.turn else 'Black'} (the live game by tablebase, the locked side set aside)" if dec.split
                       else f"{live[0].solved} for {'White' if board.turn else 'Black'} (tablebase)")
    if engine is not None:
        try:
            l = engine.best(board, depth=12)
            if l is not None:
                dec.verdict += ("; " if dec.verdict else "") + f"engine {l.cp / 100:+.1f} for {'White' if board.turn else 'Black'} on the whole board"
        except Exception:
            pass
    dec.lines = narrate(board, dec)
    return dec


# ---------------------------------------------------------------- prose
def _pc(board: chess.Board, sq: int) -> str:
    from .verbalize import pc
    p = board.piece_at(sq)
    return pc(p.symbol(), sq) if p else chess.square_name(sq)


def _files(board: chess.Board, pieces: set[int]) -> str:
    fs = sorted(chess.square_file(s) for s in pieces if board.piece_at(s) and board.piece_at(s).piece_type != chess.KING) or \
         sorted(chess.square_file(s) for s in pieces)
    if not fs:
        return "the board"
    mid = fs[len(fs) // 2]
    if fs[-1] <= 3:
        return "the queenside"
    if fs[0] >= 4:
        return "the kingside"
    return "the queenside" if mid <= 3 else "the kingside"


def narrate(board: chess.Board, dec: Decomposition) -> list[str]:
    out = []
    side = board.turn
    who = "White" if side else "Black"
    if not dec.split and not any(g.solved for g in dec.subgames):
        return []                                              # one game, nothing set aside: nothing to say here
    for g in dec.subgames:
        if not g.lemmas:
            continue
        region = _files(board, g.pieces)
        bits = []
        for l in g.lemmas:
            k = getattr(l, "kind", "")
            if k == "trapped":
                bits.append(f"{_pc(board, l.target)} is out of play (no safe move)")
            elif k == "promotion":
                r = l.race
                if r.promotes:
                    bits.append(f"{_pc(board, l.target)} promotes: {r.steps} steps, the fastest catcher needs {r.catch_hops}")
                elif r.catcher is not None:
                    how = "already covers its path" if r.catch_hops == 0 else f"reaches its path in {r.catch_hops} against {r.steps} steps"
                    bits.append(f"{_pc(board, l.target)} ({r.steps} steps to queen) is stopped only by {_pc(board, r.catcher)}, which {how}: it cannot leave")
        if g.frozen:
            res = ", ".join(f"{'White' if k == 'w' else 'Black'} {n}" for k, n in g.reservoir.items())
            lose = "; ".join(f"{'White' if k == 'w' else 'Black'}: {', '.join(v)} loses" for k, v in g.losing.items() if v)
            out.append(f"{region[0].upper() + region[1:]}: locked" + (f" ({'; '.join(bits)})" if bits else "") + f". Waiting moves: {res}." + (f" {lose}." if lose else "") + " Leave it as it is.")
        else:
            head = f"{region[0].upper() + region[1:]}: the live game"
            if g.shared:
                head += f" ({', '.join(_pc(board, s) for s in sorted(g.shared))} serves both sides)"
            if g.solved:
                head += (f" -- with the locked side set aside it is {g.men} men, tablebase: {g.solved} for {who}" if dec.split
                         else f" -- {g.men} men, tablebase: {g.solved} for {who}")
            out.append(head + (f" ({'; '.join(bits)})" if bits else "") + ".")
    if dec.split:
        out.insert(0, f"The position splits into {len([g for g in dec.subgames if g.lemmas])} sub-games within {dec.horizon} moves.")
    if dec.verdict:
        out.append("Verdict: " + dec.verdict + ".")
    return out


def regime_lines(board: chess.Board, memo=None, horizon: int = 3, engine=None) -> list[str]:
    """The decomposition as prose, or nothing when the board is one game with nothing to set aside."""
    try:
        return decompose(board, memo, horizon, engine).lines
    except Exception:
        return []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--engine", action="store_true"); ap.add_argument("--horizon", type=int, default=3)
    a = ap.parse_args(argv)
    b = chess.Board(a.fen)
    eng = None
    if a.engine:
        from .engine import Engine
        eng = Engine(depth=12)
    dec = decompose(b, Memo(), a.horizon, eng)
    for l in dec.lines:
        print(l)
    for g in dec.subgames:
        print("  sub-game:", sorted(chess.square_name(s) for s in g.pieces), "frozen" if g.frozen else "live", g.solved, [(l.goal if not getattr(l, 'kind', '') else l.kind, l.status) for l in g.lemmas])
    for sq, c in dec.cones.items():
        if board_piece := b.piece_at(sq):
            if board_piece.piece_type == chess.KING or len(c) <= 1:
                print("  cone", _pc(b, sq), sorted(chess.square_name(s) for s in c))
    if dec.verdict:
        print("  " + dec.verdict)
    if eng:
        eng.close()


if __name__ == "__main__":
    main()
