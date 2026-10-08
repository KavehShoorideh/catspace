"""MINING THE EXCHANGE FRAME'S SURPRISES FOR LEMMAS (Kaveh 2026-10-08: "learn some lemmas and then bring your lemmas into
the process ... when using all of your lemmas you're not needing any more lemmas, your lemmas are mostly complete").

On the exchanges of real games where the exchange frame missed the actual material gain by 2+ pawns and a wider frame
(the joined frame: captures anywhere; or the forcing frame: captures and checks) explains it within a pawn, the wider
frame's principal line is read and its FIRST MOVE OFF THE SQUARE is classified by how it relates to the exchange:
  guard removed     the victim off the square was defended, before the exchange, by a piece that has since left (it
                    took part on the square): the overload
  discovered        the capturer attacks the victim only because a piece that left opened its line
  landing attack    the capturer is the piece standing on the square (it captured there and now attacks elsewhere)
  already loose     the victim was already winnable by the static exchange before the exchange began: an
                    independent frame
  a check           the move is a check (in the forcing frame)
  other             none of these
The counts say which lemma to build next; examples are printed for reading.

    python -m catspace.incubator.analyst.exchange_mine --tag c300
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import chess

from .forcing_frame import ForcingFrame, forcing_lemmas


def principal_line(b: chess.Board, s: int, checks: bool, plies: int = 8, n_checks: int = 2) -> list:
    """The forcing (or joined) frame's best line after the side to move's best capture on s: [moves]."""
    from .frame_search import FrameSearch
    fs = FrameSearch(ForcingFrame(checks), forcing_lemmas()); fr = fs.frame
    st = (b, plies, n_checks)
    first = max((m for m in b.legal_moves if m.to_square == s and b.is_capture(m)),
                key=lambda m: fr.gain(st, m) - fs.value(fr.next(st, m)), default=None)
    if first is None:
        return []
    line = [first]; st = fr.next(st, first)
    while True:
        v = fs.value(st)
        best = None
        for m in fr.moves(st):
            if fr.gain(st, m) - fs.value(fr.next(st, m)) == v:
                best = m; break
        if best is None or (fr.exit(st) == v and not st[0].is_check()):
            return line
        line.append(best); st = fr.next(st, best)


def classify(b0: chess.Board, s: int, line: list) -> tuple:
    """-> (kind, the off-square move as SAN-ish text) for the first move of `line` that is not on s."""
    from .exchange_law import see
    left = set()
    bb = b0.copy(stack=False)
    for m in line:
        if m.to_square != s:
            mover = bb.piece_at(m.from_square)
            if bb.gives_check(m) and not bb.is_capture(m):
                return "a check", bb.san(m)
            if not bb.is_capture(m):
                return "other", bb.san(m)
            t = m.to_square; X = bb.turn
            if m.from_square == s:
                return "landing attack", bb.san(m)
            victim_side = not X
            guards0 = set(b0.attackers(victim_side, t))
            if guards0 & left:
                return "guard removed", bb.san(m)
            if m.from_square not in b0.attackers(X, t) and b0.piece_at(m.from_square) is not None and b0.piece_at(m.from_square).color == X:
                return "discovered", bb.san(m)
            b0x = b0.copy(stack=False); b0x.turn = X; b0x.ep_square = None
            if b0x.piece_at(t) is not None and b0x.attackers(X, t) and see(b0x, t)[0] > 0:
                return "already loose", bb.san(m)
            return "other", bb.san(m)
        left.add(m.from_square)
        bb.push(m)
    return "none off the square", ""


def _game(plies: dict) -> list:
    from .exchange_frame import Search, frame_of, root_values
    from .exchange_predict import MATE, _pawns, _quiet
    from .forcing_frame import capture_value
    from .tactic_coverage import game_moves
    moves = game_moves(plies); out = []; stats = Counter()
    for t in sorted(plies):
        if t not in moves:
            continue
        r = plies[t]; b = chess.Board(r["fen"]); m = moves[t]
        if not b.is_capture(m) or b.piece_at(m.to_square) is None:
            continue
        s = m.to_square
        st = frame_of(b, s)
        if not st[1] or abs(_pawns(r["eval"])) >= MATE:
            continue
        v = max(root_values(Search(True), *st).values())
        u = t; bb = b.copy(stack=False)                              # follow the played exchange to its first quiet move
        while u in moves and u + 1 in plies and not _quiet(bb, moves[u]):
            bb.push(moves[u]); u += 1
        if u not in moves:                                           # the game ended inside the exchange (resignation, draw,
            stats["cut by the game's end"] += 1                      # mate): the material is not settled (bug 2026-10-08:
            continue                                                 # Qxc6+ with Kxc6 never played counted +9)
        ev = [(_pawns(plies[k]["eval"]) if plies[k]["side"] == r["side"] else -_pawns(plies[k]["eval"])) for k in range(t, u + 1) if k in plies]
        if any(abs(x) >= MATE for x in ev):
            continue
        # THE TRUTH (review 2026-10-08): the material actually won, graded only where both sides played the exchange
        # accurately -- Stockfish's evaluation never moves by a pawn or more from one ply to the next inside it (the
        # played line's material otherwise carries a blunder; Stockfish's own value carries the position's positional worth)
        if any(abs(ev[i + 1] - ev[i]) >= 1.0 for i in range(len(ev) - 1)):
            stats["blunder inside"] += 1
            continue
        end = plies[u]
        gain = (end["material"] if end["side"] == r["side"] else -end["material"]) - r["material"]
        stats["graded"] += 1
        if abs(gain - v) < 2:
            continue
        vj = capture_value(b, s, use_checks=False)[0]; vf = capture_value(b, s, use_checks=True)[0]
        which = "joined" if abs(vj - gain) < 1 else ("forcing" if abs(vf - gain) < 1 else None)
        if which is None:
            out.append(dict(kind="beyond forcing play", which=None, fen=r["fen"], sq=chess.square_name(s), v=v, gain=round(gain, 1), move="", vf=vf))
            continue
        line = principal_line(b, s, checks=(which == "forcing"))
        kind, mv = classify(b, s, line)
        out.append(dict(kind=kind, which=which, fen=r["fen"], sq=chess.square_name(s), v=v, gain=round(gain, 1), move=mv,
                        line=[x.uci() for x in line]))
    return out, stats


def run(tag: str) -> None:
    from multiprocessing import Pool

    from .tactic_census_measure import load
    by = defaultdict(dict)
    for r in load(tag):
        by[r["game"]][r["ply"]] = r
    recs = []; stats = Counter()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for o, st in pool.imap_unordered(_game, [by[g] for g in sorted(by)], chunksize=2):
            recs += o; stats.update(st)
    print(f"OBS captures taken and graded (accurate exchanges: Stockfish moves less than a pawn per ply inside) {stats['graded']}; "
          f"set aside with a blunder inside {stats['blunder inside']}, cut by the game's end {stats['cut by the game\'s end']}")
    kinds = Counter(r["kind"] for r in recs)
    print(f"OBS the exchange frame's surprises in {tag} (|frame - the material won in an accurate exchange| >= 2 pawns): {len(recs)}; by the first move off the square in the explaining frame's line:")
    print("OBS   " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    for k, _ in kinds.most_common():
        ex = [r for r in recs if r["kind"] == k][:4]
        for r in ex:
            print(f"OBS   {k}: {r['fen']} on {r['sq']}: frame {r['v']:+d}, won {r['gain']:+.0f}{(', forcing ' + format(r['vf'], '+.0f')) if 'vf' in r else ''}; off the square: {r['move']}; line {' '.join(r.get('line', []))}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="surprises", choices=["surprises", "blunders"])
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=None)
    a = ap.parse_args(argv)
    (run(a.tag) if a.cmd == "surprises" else blunder_test(a.tag, a.games))



# ---------------------------------------------------------------------------------------------------------------------
# BLUNDERS AS A TEST OF THE LEMMAS (Kaveh 2026-10-08: "you can use blundered exchanges to test your lemmas, because that
# should indicate a violated or overlooked lemma by the player who blundered"). Inside the exchanges of real games, a
# BLUNDER is a move after which Stockfish's evaluation for its mover drops by a pawn or more. Our lemma sets FLAG it when
# they predict the move costs at least a pawn against the best forcing alternative:
#   hanging     after the move, the opponent has a capture that wins by the plain count (the static exchange)
#   joined      the joined frame (captures anywhere, both sides answering)
#   forcing     the forcing frame (captures and checks)
# Unflagged blunders are what our lemmas still miss.

def _frame_value(b: chess.Board, checks: bool) -> float:
    """The side to move's value in the forcing (or joined) frame at the root: the best of standing pat and its moves."""
    from .frame_search import FrameSearch
    fs = FrameSearch(ForcingFrame(checks), forcing_lemmas())
    return fs.value((b, 8, 2))


def _move_cost(b: chess.Board, m: chess.Move, checks: bool) -> float:
    """What playing m costs its mover against the frame's best, in pawns (>= 0 up to the frame's horizon)."""
    fr = ForcingFrame(checks)
    g = fr.gain((b, 8, 2), m)
    nb = b.copy(stack=False); nb.push(m)
    return _frame_value(b, checks) - (g - _frame_value(nb, checks))


def _hanging(b: chess.Board, m: chess.Move) -> int:
    """After m, the opponent's best capture by the plain count (0 when none wins)."""
    from .exchange_law import see
    nb = b.copy(stack=False); nb.push(m)
    best = 0
    for q, pc in nb.piece_map().items():
        if pc.color != nb.turn and pc.piece_type != chess.KING and nb.attackers(nb.turn, q):
            best = max(best, see(nb, q)[0])
    return best


def _blunders(plies: dict) -> list:
    from .exchange_predict import MATE, _pawns, _quiet
    from .tactic_coverage import game_moves
    moves = game_moves(plies); out = []; done = set()
    for t in sorted(plies):
        if t not in moves:
            continue
        b = chess.Board(plies[t]["fen"])
        if not b.is_capture(moves[t]) or t in done:
            continue
        u = t; bb = b.copy(stack=False)
        while u in moves and u + 1 in plies and not _quiet(bb, moves[u]):
            bb.push(moves[u]); u += 1
        for k in range(t, u + 1):                                   # every move of the exchange, the closing quiet one too
            if k in done or k not in moves or k + 1 not in plies:
                continue
            done.add(k)
            e0, e1 = _pawns(plies[k]["eval"]), -_pawns(plies[k + 1]["eval"])      # both from the mover's side
            if abs(e0) >= MATE or abs(e1) >= MATE or e0 - e1 < 1.0:
                continue
            bk = chess.Board(plies[k]["fen"]); m = moves[k]
            out.append(dict(fen=plies[k]["fen"], move=bk.san(m), drop=round(e0 - e1, 1), elo=plies[k]["elo"],
                            hanging=_hanging(bk, m) >= 1, joined=_move_cost(bk, m, False) >= 1, forcing=_move_cost(bk, m, True) >= 1))
    return out


def blunder_test(tag: str, games: int | None = None) -> None:
    from multiprocessing import Pool

    from .tactic_census_measure import load
    by = defaultdict(dict)
    for r in load(tag):
        by[r["game"]][r["ply"]] = r
    recs = []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for o in pool.imap_unordered(_blunders, [by[g] for g in sorted(by)][:games], chunksize=2):
            recs += o
    n = len(recs)
    f = lambda k: sum(r[k] for r in recs)
    anyf = sum(r["hanging"] or r["joined"] or r["forcing"] for r in recs)
    print(f"OBS blunders inside exchanges in {tag} (Stockfish drops a pawn or more for the mover): {n}; flagged by our lemmas -- hanging piece "
          f"{f('hanging')} ({100 * f('hanging') / max(n, 1):.0f}%), joined frame {f('joined')} ({100 * f('joined') / max(n, 1):.0f}%), forcing frame "
          f"{f('forcing')} ({100 * f('forcing') / max(n, 1):.0f}%), any {anyf} ({100 * anyf / max(n, 1):.0f}%)")
    big = [r for r in recs if r["drop"] >= 3]
    print(f"OBS   the large ones (a drop of 3+ pawns): {len(big)}; flagged by the forcing frame {sum(r['forcing'] for r in big)}, by any "
          f"{sum(r['hanging'] or r['joined'] or r['forcing'] for r in big)}")
    miss = sorted((r for r in recs if not (r["hanging"] or r["joined"] or r["forcing"])), key=lambda r: -r["drop"])
    print("OBS   unflagged, the largest first (what our lemmas still miss):")
    for r in miss[:12]:
        print(f"OBS     {r['fen']} {r['move']}: Stockfish drop {r['drop']:.1f}")



if __name__ == "__main__":
    main()
