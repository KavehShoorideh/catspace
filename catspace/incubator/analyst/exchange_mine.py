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
            from .chess_domain import passed                       # the safe null move, not a bare turn flip (review 2026-10-08)
            b0x = b0 if b0.turn == X else passed(b0)
            if b0x is not None and b0x.piece_at(t) is not None and b0x.attackers(X, t) and see(b0x, t)[0] > 0:
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
    ap.add_argument("--plies", type=int, default=8)
    ap.add_argument("--checks", type=int, default=2)
    a = ap.parse_args(argv)
    (run(a.tag) if a.cmd == "surprises" else blunder_test(a.tag, a.games, [a.plies, a.checks]))



# ---------------------------------------------------------------------------------------------------------------------
# BLUNDERS AS A TEST OF THE LEMMAS (Kaveh 2026-10-08: "you can use blundered exchanges to test your lemmas, because that
# should indicate a violated or overlooked lemma by the player who blundered"). Inside the exchanges of real games, a
# BLUNDER is a move after which Stockfish's evaluation for its mover drops by a pawn or more. Our lemma sets FLAG it when,
# valued by that set, the move is at least a pawn worse than the BEST LEGAL ALTERNATIVE:
#   hanging     after the move, the opponent has a capture that wins by the plain count (the static exchange)
#   joined      the joined frame (captures anywhere, both sides answering): a move is worth its gain less the opponent's value
#   forcing     the forcing frame (captures and checks), the same way
#   ledger      the frame player's one-ply ledger (fates.ledger_move_value: the earlier work)
# Unflagged blunders are what our lemmas still miss.

HORIZON = [8, 2]                                                  # plies, checks: the forcing frame's horizon (settable)


def _frame_value(b: chess.Board, checks: bool, threats: bool = False) -> float:
    """The side to move's value in the forcing (or joined) frame at the root: the best of standing pat and its moves."""
    from .frame_search import FrameSearch
    fs = FrameSearch(ForcingFrame(checks, threats), forcing_lemmas(threats))
    return fs.value((b, HORIZON[0], HORIZON[1]))


def _move_cost(b: chess.Board, m: chess.Move, checks: bool) -> float:
    """What playing m costs its mover against the frame's best, in pawns (>= 0 up to the frame's horizon)."""
    fr = ForcingFrame(checks)
    g = fr.gain((b, HORIZON[0], HORIZON[1]), m)
    nb = b.copy(stack=False); nb.push(m)
    return _frame_value(b, checks) - (g - _frame_value(nb, checks))


def _hanging(b: chess.Board, m: chess.Move) -> int:
    """After m, the opponent's best capture by the plain count, LESS what m itself captured (bug 2026-10-08: dxe5 Nxe5
    counted Black's pawn and not White's), over LEGAL captures only (bug: after a4+ it counted a capture White, in check,
    cannot make). 0 when none wins."""
    from .exchange_law import see
    nb = b.copy(stack=False); nb.push(m)
    own = VALS.get(b.piece_at(m.to_square).piece_type, 0) if b.piece_at(m.to_square) else (1 if b.is_en_passant(m) else 0)
    best = 0
    for x in nb.legal_moves:
        if nb.is_capture(x) and nb.piece_at(x.to_square) is not None and nb.piece_at(x.to_square).piece_type != chess.KING:
            best = max(best, see(nb, x.to_square)[0])
    return best - own


VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _ledger_cost(b: chess.Board, m: chess.Move) -> float:
    """The frame player's own judgment (the earlier work, fates.py): the move's one-ply ledger -- what it wins, minus the
    opponent's best exchange anywhere after it, plus theirs if we had passed (frames by `square_frame`) -- against the best
    we could do by our frames resolved jointly (`joint_threats`; 0 = standing still)."""
    from .fates import joint_threats, ledger_move_value, square_frame
    v = ledger_move_value(b, m, square_frame)[0]
    best = max(0.0, joint_threats(b, square_frame)[0])
    return best - v


def _gap_to_best(b: chess.Board, m: chess.Move, valuer) -> float:
    """How much worse m is than the best legal move, every move valued by the same `valuer(b, move)` (review 2026-10-08:
    against passing, a threat that already stood is netted out and ignoring it is never flagged; against standing pat,
    a quiet defensive alternative is invisible -- the fair test compares with the best alternative)."""
    vals = {x: valuer(b, x) for x in b.legal_moves}
    return max(vals.values()) - vals[m]


def _forcing_valuer(checks: bool, threats: bool = False):
    fr = ForcingFrame(checks, threats)

    def v(b, x):
        nb = b.copy(stack=False); nb.push(x)
        if nb.is_checkmate():
            return 1000.0
        return fr.gain((b, HORIZON[0], HORIZON[1]), x) - _frame_value(nb, checks, threats)
    return v


def _ledger_valuer(b0: chess.Board):
    from .chess_domain import passed
    from .fates import ledger_move_value, square_frame, threats
    stand = passed(b0)
    before = threats(stand, square_frame) if stand is not None else (0.0, None, [])
    return lambda b, x: ledger_move_value(b, x, square_frame, before=before)[0]


def _blunders(plies: dict, horizon=None) -> list:
    if horizon:
        HORIZON[:] = horizon
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
            if abs(e0) >= MATE or abs(e1) >= MATE:
                continue
            good = e0 - e1 < 0.3 and k % 3 == 0                     # a sample of GOOD moves: the detectors' false alarms
            if e0 - e1 < 1.0 and not good:
                continue
            bk = chess.Board(plies[k]["fen"]); m = moves[k]
            out.append(dict(fen=plies[k]["fen"], move=bk.san(m), drop=round(e0 - e1, 1), elo=plies[k]["elo"], good=good,
                            hanging=_hanging(bk, m) >= 1, joined=_gap_to_best(bk, m, _forcing_valuer(False)) >= 1,
                            forcing=_gap_to_best(bk, m, _forcing_valuer(True)) >= 1, ledger=_gap_to_best(bk, m, _ledger_valuer(bk)) >= 1,
                            threat=_gap_to_best(bk, m, _forcing_valuer(True, True)) >= 1,
                            parry=_gap_to_best(bk, m, _forcing_valuer(True, "parry")) >= 1))
    return out


_SF = None


def _sf_init():
    global _SF
    import chess.engine
    _SF = chess.engine.SimpleEngine.popen_uci("stockfish")


def _reply_kind(r: dict) -> tuple:
    """Stockfish's best reply to the blunder, by kind: a pawn ending; a capture; a check; a quiet move."""
    import chess.engine
    b = chess.Board(r["fen"]); b.push(b.parse_san(r["move"]))
    if b.is_game_over():
        return r, "game over", "-"
    best = _SF.play(b, chess.engine.Limit(depth=12)).move
    pieces = {p.piece_type for p in b.piece_map().values()}
    if pieces <= {chess.KING, chess.PAWN}:
        kind = "a pawn ending"
    elif b.is_capture(best):
        kind = "a capture"
    elif b.gives_check(best):
        kind = "a quiet check"
    else:                                                   # what the quiet reply creates against the blunderer
        from .chess_domain import passed
        from .exchange_law import see
        from .forcing_frame import double_threat
        a = b.copy(stack=False); a.push(best)
        p = passed(a)
        mate = p is not None and any((p.push(x), p.is_checkmate(), p.pop())[1] for x in list(p.legal_moves))
        top = 0
        if p is not None:
            top = max([see(p, q)[0] for q, pc in p.piece_map().items() if pc.color == a.turn and pc.piece_type != chess.KING and p.attackers(p.turn, q)] + [0])
        kind = ("a quiet move: a mate threat" if mate else "a quiet move: a double threat" if double_threat(a) >= 1
                else "a quiet move: a single threat" if top >= 1 else "a quiet move: nothing material")
    return r, kind, b.san(best)


def blunder_test(tag: str, games: int | None = None, horizon=None) -> None:
    from multiprocessing import Pool

    from .tactic_census_measure import load
    by = defaultdict(dict)
    for r in load(tag):
        by[r["game"]][r["ply"]] = r
    recs = []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        from functools import partial
        for o in pool.imap_unordered(partial(_blunders, horizon=horizon), [by[g] for g in sorted(by)][:games], chunksize=2):
            recs += o
    goods = [r for r in recs if r["good"]]; recs = [r for r in recs if not r["good"]]
    keys = ("hanging", "joined", "forcing", "ledger", "threat", "parry")
    print(f"OBS FALSE ALARMS on {len(goods)} good moves inside the same exchanges (Stockfish moves less than 0.3 pawns): " +
          ", ".join(f"{k} {sum(g[k] for g in goods)} ({100 * sum(g[k] for g in goods) / max(len(goods), 1):.1f}%)" for k in keys))
    n = len(recs)
    f = lambda k: sum(r[k] for r in recs)
    anyf = sum(r["hanging"] or r["joined"] or r["forcing"] or r["ledger"] or r["threat"] or r["parry"] for r in recs)
    print(f"OBS   the forcing frame WITH the parry lemma flags {f('parry')} ({100 * f('parry') / max(n, 1):.0f}%); newly flagged against the double-threat "
          f"frame {sum(r['parry'] and not r['threat'] for r in recs)}, lost {sum(r['threat'] and not r['parry'] for r in recs)}")
    print(f"OBS   the forcing frame WITH the double-threat exit lemma flags {f('threat')} ({100 * f('threat') / max(n, 1):.0f}%); newly flagged against the "
          f"plain forcing frame {sum(r['threat'] and not r['forcing'] for r in recs)}, lost {sum(r['forcing'] and not r['threat'] for r in recs)}")
    print(f"OBS   the frame player's ledger (fates.ledger_move_value + joint_threats, the earlier work) flags {f('ledger')} ({100 * f('ledger') / max(n, 1):.0f}%); "
          f"flagged by the ledger but not the forcing frame {sum(r['ledger'] and not r['forcing'] for r in recs)}, by the forcing frame but not the ledger "
          f"{sum(r['forcing'] and not r['ledger'] for r in recs)}")
    print(f"OBS blunders inside exchanges in {tag} (Stockfish drops a pawn or more for the mover; the forcing frame's horizon {horizon or HORIZON}): {n}; flagged by our lemmas -- hanging piece "
          f"{f('hanging')} ({100 * f('hanging') / max(n, 1):.0f}%), joined frame {f('joined')} ({100 * f('joined') / max(n, 1):.0f}%), forcing frame "
          f"{f('forcing')} ({100 * f('forcing') / max(n, 1):.0f}%), any {anyf} ({100 * anyf / max(n, 1):.0f}%)")
    big = [r for r in recs if r["drop"] >= 3]
    print(f"OBS   the large ones (a drop of 3+ pawns): {len(big)}; flagged by the forcing frame {sum(r['forcing'] for r in big)}, by any "
          f"{sum(r['hanging'] or r['joined'] or r['forcing'] or r['ledger'] or r['threat'] for r in big)}, by the ledger {sum(r['ledger'] for r in big)}, with the double threat {sum(r['threat'] for r in big)}")
    miss = sorted((r for r in recs if not (r["hanging"] or r["joined"] or r["forcing"] or r["ledger"] or r["threat"] or r["parry"])), key=lambda r: -r["drop"])
    # WHAT THE UNFLAGGED ARE: Stockfish's best reply to the blunder (depth 12), by kind
    kinds = Counter(); ex = defaultdict(list)
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_sf_init) as pool:
        for r, kind, reply in pool.imap_unordered(_reply_kind, miss, chunksize=4):
            kinds[kind] += 1
            if len(ex[kind]) < 3:
                ex[kind].append((r["fen"], r["move"], reply, r["drop"]))
    print(f"OBS   the {len(miss)} unflagged, by Stockfish's best reply to the blunder: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    for k, _ in kinds.most_common():
        for fen, mv, reply, drop in ex[k]:
            print(f"OBS     {k}: {fen} {mv}, Stockfish answers {reply} (drop {drop:.1f})")
    print("OBS   unflagged, the largest first (what our lemmas still miss):")
    for r in miss[:12]:
        print(f"OBS     {r['fen']} {r['move']}: Stockfish drop {r['drop']:.1f}")



if __name__ == "__main__":
    main()
