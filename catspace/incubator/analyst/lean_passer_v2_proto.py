"""THE PASSER LEMMA, SECOND FORM, PROTOTYPED BEFORE IT IS PROVED (short runs before big runs, applied to proofs).

The first form needs a bare enemy king and never fires on real positions (0 of 4,700). The second form allows enemy
PAWNS that provably cannot interfere, and keeps our side without sliders. Its hypotheses, evaluated here in Python
exactly as they would be in Lean, and then every PROVED claim checked by the exhaustive race search: a false claim
means the hypothesis set is unsound and must change before a proof is attempted.

Hypotheses (us = side to move, pawn on `sq`, `n` pushes to promotion, the enemy gets `n - 1` moves before it):
  ownPawn, pathClear, noSliders (ours), notInCheck, kingOutside     -- as in the first form
  enemyKingPawns   -- the enemy has only its king and pawns
  pawnsFar         -- every enemy pawn needs more than n - 1 moves (a lower bound: max(|file diff|, rank diff minus
                      one for a possible double step)) to reach any BAD square: the path and our pawn's square, the
                      squares from which it would check our king, and its promotion rank -- or (the cone form,
                      2026-10-06) the bad square is off the pawn's file and the pawn cannot leave its file in time:
                      it is more than n - 1 moves from every square it could capture on beside it (our pieces, the
                      path; nothing else of ours moves during the race)
  kingRoom         -- for every square the enemy king can reach in n - 1 moves, the number of its neighbours that
                      hold no piece of ours, are attacked by none of our pieces (our pawn's whole future attack set
                      counted), and are not on the path exceeds the number of enemy pawns: the enemy can never run
                      out of king moves (no self-stalemate)

    python -m catspace.incubator.analyst.lean_passer_v2_proto --theme pawnEndgame --n 500 --offset 17
    python -m catspace.incubator.analyst.lean_passer_v2_proto --random 2000
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter

import chess

from .lean_backrank_puzzles import puzzle_rows
from .lean_passer_check import promotes_within

SLIDERS = chess.ROOK, chess.BISHOP, chess.QUEEN


def kdist(a: int, b: int) -> int:
    return chess.square_distance(a, b)


def passer_v2(b: chess.Board, sq: int) -> dict:
    """The hypotheses of the second form for our pawn on `sq`. -> {name: bool}, plus 'n'."""
    us = b.turn
    them = not us
    fwd = 1 if us == chess.WHITE else -1
    f, r = chess.square_file(sq), chess.square_rank(sq)
    promo_rank = 7 if us == chess.WHITE else 0
    n = abs(promo_rank - r)
    path = [chess.square(f, r + fwd * i) for i in range(1, n + 1)]
    h = {"n": n}
    pc = b.piece_at(sq)
    h["ownPawn"] = pc is not None and pc.color == us and pc.piece_type == chess.PAWN and r not in (0, 7)
    h["pathClear"] = all(b.piece_at(t) is None for t in path)
    h["noSliders"] = not ((b.rooks | b.bishops | b.queens) & b.occupied_co[us])
    h["enemyKingPawns"] = not ((b.rooks | b.bishops | b.queens | b.knights) & b.occupied_co[them])
    ok = b.king(us)
    ek = b.king(them)
    h["notInCheck"] = ok is not None and ek is not None and not b.is_check() and not b.attackers_mask(us, ek)
    h["kingOutside"] = ek is not None and n < kdist(ek, chess.square(f, promo_rank))
    # enemy pawns: lower bound on the moves to reach a bad square
    their_fwd = -fwd
    their_start = 6 if them == chess.BLACK else 1
    their_promo = 0 if them == chess.BLACK else 7
    bad = set(path) | {chess.square(x, their_promo) for x in range(8)}        # our pawn's own square: we push first
    if ok is not None:                                 # squares from which their pawn gives check
        for df in (-1, 1):
            cf, cr = chess.square_file(ok) + df, chess.square_rank(ok) - their_fwd
            if 0 <= cf < 8 and 0 <= cr < 8:
                bad.add(chess.square(cf, cr))

    def pawn_dist(e: int, t: int) -> float:
        ef, er = chess.square_file(e), chess.square_rank(e)
        tf, tr = chess.square_file(t), chess.square_rank(t)
        ahead = (tr - er) * their_fwd                    # how many ranks t is ahead of the pawn (0: its own rank)
        if ahead < 0 or abs(tf - ef) > ahead:
            return float("inf")
        return max(abs(tf - ef), ahead - (1 if er == their_start else 0))   # 0 on its own square (the Lean pdist)

    # the cone form (Lean `pawnFarB`, 2026-10-06): a bad square off the pawn's file is also safe when the pawn cannot
    # leave its file in time -- a pawn changes file only by capturing, our pieces never move but the passer along
    # its path, and the pawn is more than n - 1 moves from every capture square on the files beside it
    capturable = set(chess.SquareSet(b.occupied_co[us])) | set(path)

    def stuck(e: int) -> bool:
        return all(pawn_dist(e, d) > n - 1 for d in capturable if abs(chess.square_file(d) - chess.square_file(e)) == 1)

    h["pawnsFar"] = all(pawn_dist(e, t) > n - 1 or (chess.square_file(t) != chess.square_file(e) and stuck(e))
                        for e in chess.SquareSet(b.pawns & b.occupied_co[them]) for t in bad)
    # king room: our attacks with the passer's whole future attack set, every square the enemy king can reach
    our_attacks = 0
    for s in chess.SquareSet(b.occupied_co[us]):
        if s != sq:
            our_attacks |= b.attacks_mask(s)
    for i in range(0, n):                                # the pawn on r + i*fwd (now, and after each push) attacks
        rr = r + fwd * (i + 1)
        if 0 <= rr < 8:
            for df in (-1, 1):
                if 0 <= f + df < 8:
                    our_attacks |= chess.BB_SQUARES[chess.square(f + df, rr)]
    forbidden = our_attacks | b.occupied_co[us]
    for t in path:
        forbidden |= chess.BB_SQUARES[t]
    n_their = len(chess.SquareSet(b.pawns & b.occupied_co[them]))
    # the squares the enemy king can never stand on: next to our king, within one of the promotion square (the
    # Lean `neverKingB`; skipping every forbidden square, as the first prototype did, was unsound for squares
    # holding our capturable pieces, 2026-10-05)
    promo_sq = chess.square(f, promo_rank)
    room = True
    if ek is not None and ok is not None:
        for x in range(64):
            if kdist(x, ek) <= n - 1 and not (chess.BB_KING_ATTACKS[ok] & chess.BB_SQUARES[x]) and kdist(x, promo_sq) > 1:
                free = sum(1 for d in chess.SquareSet(chess.BB_KING_ATTACKS[x]) if not (chess.BB_SQUARES[d] & forbidden))
                if free <= n_their:
                    room = False
                    break
    h["kingRoom"] = room
    return h


HYPS = ("ownPawn", "pathClear", "noSliders", "enemyKingPawns", "notInCheck", "kingOutside", "pawnsFar", "kingRoom")

QUEEN_GAIN = 8.0        # a queen for a pawn, the value the zone search already uses for a promotion


def passer_gate(b: chess.Board) -> bool:
    """Where the second form can fire at all: we have pawns and no sliders, the enemy has only king and pawns."""
    us = b.turn
    return bool(b.pawns & b.occupied_co[us]) and not ((b.rooks | b.bishops | b.queens) & b.occupied_co[us]) \
        and not ((b.rooks | b.bishops | b.queens | b.knights) & b.occupied_co[not us])


USE_LEAN = True         # the Lean detector (the proved definitions); False: this Python prototype of them


def proved_passers(b: chess.Board) -> list[tuple[int, int]]:
    """Our pawns whose promotion the second form proves: [(square, plies)]. By the Lean detector (`passer2`,
    theorem unstoppablePasser2_sound) unless USE_LEAN is off."""
    if not passer_gate(b):
        return []
    if USE_LEAN:
        from .lean_ideas import detector
        return [(i.square, i.plies) for i in detector().passer2(b) if i.status == "PROVED"]
    out = []
    for sq in chess.SquareSet(b.pawns & b.occupied_co[b.turn]):
        h = passer_v2(b, sq)
        if all(h[k] for k in HYPS):
            out.append((sq, 2 * h["n"] - 1))
    return out


VALUE = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}


def proved_passer_gains(b: chess.Board) -> dict:
    """-> {square: (gain, plies)} for each proved passer of the side to move whose credit is positive.
    A conservative material credit for a proved passer of the side to move at the search's horizon: the queen it
    brings, minus the value of our pieces the enemy king (king distance) or pawns (pawn distance) can reach during the
    race, and only when every enemy pawn needs at least two more moves after our promotion to promote itself (else the
    race is level and the credit would be a horizon artefact: puzzle 0bmcR, 2026-10-05). 0 when none."""
    ps = proved_passers(b)
    if not ps:
        return {}
    us, them = b.turn, not b.turn
    their_fwd = -1 if them == chess.BLACK else 1
    their_start = 6 if them == chess.BLACK else 1
    their_promo = 0 if them == chess.BLACK else 7
    ek = b.king(them)
    gains: dict = {}
    for sq, plies in ps:
        n = (plies + 1) // 2
        # their pawns: at least two more moves after our promotion
        slow = True
        for e in chess.SquareSet(b.pawns & b.occupied_co[them]):
            d = abs(their_promo - chess.square_rank(e)) - (1 if chess.square_rank(e) == their_start else 0)
            if d < n + 2:
                slow = False
                break
        if not slow:
            continue
        # our pieces the enemy can reach in n - 1 moves
        reachable = 0.0
        for s in chess.SquareSet(b.occupied_co[us]):
            pt = b.piece_type_at(s)
            if pt == chess.KING or s == sq:
                continue
            if ek is not None and chess.square_distance(ek, s) <= n - 1:
                reachable += VALUE[pt]
                continue
            for e in chess.SquareSet(b.pawns & b.occupied_co[them]):
                ahead = (chess.square_rank(s) - chess.square_rank(e)) * their_fwd
                df = abs(chess.square_file(s) - chess.square_file(e))
                if 0 < ahead and df <= ahead and max(df, ahead - (1 if chess.square_rank(e) == their_start else 0)) <= n - 1:
                    reachable += VALUE[pt]
                    break
        if QUEEN_GAIN - reachable > 0:
            gains[sq] = (QUEEN_GAIN - reachable, plies)
    return gains


def proved_passer_gain(b: chess.Board) -> float:
    """The best credit among proved_passer_gains (the zone search's leaf credit; 0 when none)."""
    return max((g for g, _ in proved_passer_gains(b).values()), default=0.0)


def random_pawn_ending(rng: random.Random) -> chess.Board | None:
    b = chess.Board(None)
    squares = rng.sample(range(64), 2 + rng.randint(1, 3) + rng.randint(0, 3))
    k, ek, *rest = squares
    if kdist(k, ek) <= 1:
        return None
    b.set_piece_at(k, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(ek, chess.Piece(chess.KING, chess.BLACK))
    nw = rng.randint(1, len(rest))
    for i, s in enumerate(rest):
        if chess.square_rank(s) in (0, 7):
            return None
        b.set_piece_at(s, chess.Piece(chess.PAWN, chess.WHITE if i < nw else chess.BLACK))
    b.turn = chess.WHITE
    if not b.is_valid() or b.is_check() or b.attackers_mask(chess.WHITE, ek):
        return None
    return b


def evaluate(boards, log=print, check_cap: int = 400, max_plies: int = 11):
    status = Counter()
    failing = Counter()
    proved = []
    for b in boards:
        for sq in chess.SquareSet(b.pawns & b.occupied_co[b.turn]):
            h = passer_v2(b, sq)
            fails = [k for k in HYPS if not h[k]]
            if not fails:
                status["PROVED"] += 1
                proved.append((b.copy(), sq, h["n"]))
            elif len(fails) == 1:
                status["ONEAWAY"] += 1
                failing[fails[0]] += 1
            else:
                status["FAIL"] += 1
    checked = confirmed = 0
    wrong = []
    plies_hist = Counter()
    for b, sq, n in proved:
        plies = 2 * n - 1
        plies_hist[plies] += 1
        if checked >= check_cap or plies > max_plies:
            continue
        checked += 1
        if promotes_within(b.copy(), chess.square_file(sq), plies):
            confirmed += 1
        elif len(wrong) < 5:
            wrong.append((b.fen(), chess.square_name(sq)))
    log(f"OBS passer v2 prototype: {len(boards)} positions; instances {dict(status)}; one-away failing {dict(failing)}; "
        f"PROVED by plies {dict(sorted(plies_hist.items()))}; race search confirmed {confirmed}/{checked} "
        f"(cap {check_cap}, plies <= {max_plies}); WRONG {len(wrong)}")
    for fen, s in wrong:
        log(f"  WRONG {fen} pawn {s}")
    return status, confirmed, checked, wrong


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="pawnEndgame")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--offset", type=int, default=17)
    ap.add_argument("--random", type=int, default=0, help="instead: this many random pawn endings with enemy pawns")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-plies", type=int, default=11)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    boards = []
    if a.random:
        rng = random.Random(a.seed)
        while len(boards) < a.random:
            b = random_pawn_ending(rng)
            if b is not None:
                boards.append(b)
    else:
        for row in puzzle_rows(a.n, 97, a.offset, None, theme=a.theme):
            b = chess.Board(row["FEN"])
            mv = row["Moves"].split()
            b.push_uci(mv[0])
            for i in range(1, len(mv)):
                boards.append(b.copy())
                b.push_uci(mv[i])
    print(f"{len(boards)} positions in {time.perf_counter() - t0:.0f}s")
    evaluate(boards, max_plies=a.max_plies)
    print(f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
