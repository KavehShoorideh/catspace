"""THE EXCHANGE LAW, prototyped before it is proved (Kaveh 2026-10-07: "the tactics are basically exceptions to the exchange
theorem"; the standing method: prototype -> bridge -> prove).

THE LAW. In position p with side c to move, a square s holds an enemy piece; A are c's pieces attacking s, D the enemy's.
The static exchange (the classical SEE: each side recaptures with its least valuable piece or stops) gives a value
see(p, s). The claim: if the frame is HONEST -- the hypotheses below hold -- then after c captures on s with its least
valuable attacker, c's material against every legal reply is at least see(p, s) within the sequence's length.

THE HYPOTHESES (each a computable check on the position, named; a tactic is the failure of one of them):
  target        s holds an enemy piece that is not the king, and c attacks it
  noKing        no king among the attackers or defenders of s (a king cannot capture into an attack)
  noPin         no attacker or defender is pinned to its king (every capture on s is legal when its turn comes)
  noXray        no slider of either side stands behind a single piece on a line to s (no second-row recapture)
  noPromotion   no pawn participant stands on its seventh rank; no en passant on s
  noOtherCapture  the enemy can capture nothing of ours but on s (our pieces off s are unattacked), so stopping the
                exchange banks the material -- the "overloaded / loose piece" exceptions
  noCheck       the enemy has no checking move now, our king shares no line or knight relation with s, and no enemy
                slider shares a line with our king -- no in-between check (the "zwischenzug" exception)
  noInterpose   every attacker of ours is a knight, a pawn, or stands next to s -- the enemy cannot block our recapture
  noArrival     no enemy piece off the frame can move to a square attacking s in one move -- the defenders are fixed
THE BRIDGE: an exact alpha-beta over every legal move to the sequence's depth, with a captures-only quiescence at the
leaves (either side may stand pat), which must never fall below see(p, s) when the hypotheses hold; where they fail,
the gap is the tactic.

    python -m catspace.incubator.analyst.exchange_law census --tag c300 [--limit 3000]
Prints OBS lines: how often the law fires, how often the bridge agrees (must be always), and which hypothesis fails most.
"""
from __future__ import annotations

import argparse
from collections import Counter

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)
HYPS = ("target", "noKing", "noPin", "noXray", "noPromotion", "noOtherCapture", "noCheck", "noInterpose", "noArrival")


def material(b: chess.Board, side: chess.Color) -> int:
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values() if p.piece_type != chess.KING)


def participants(b: chess.Board, s: int) -> tuple:
    """(attackers of the side to move, defenders of the other side), as sorted lists of squares, cheapest first."""
    c = b.turn
    key = lambda sq: (VAL[b.piece_at(sq).piece_type], sq)
    return sorted(b.attackers(c, s), key=key), sorted(b.attackers(not c, s), key=key)


def see(b: chess.Board, s: int) -> tuple:
    """The static exchange on s for the side to move: (value, number of captures in the sequence). The classical gain
    list, each side capturing with its cheapest piece and free to stop; no x-rays (the hypothesis excludes them)."""
    A, D = participants(b, s)
    target = b.piece_at(s)
    gain = [VAL[target.piece_type]]
    on_s = VAL[b.piece_at(A[0]).piece_type]                      # our cheapest attacker lands on s
    att, dfd = list(A[1:]), list(D)
    side_lists = [dfd, att]                                       # the enemy recaptures first
    i = 0
    while side_lists[i % 2]:
        sq = side_lists[i % 2].pop(0)
        gain.append(on_s - gain[-1])                              # speculative: the piece on s is taken
        on_s = VAL[b.piece_at(sq).piece_type]
        i += 1
    while len(gain) > 1:                                          # fold back: each side may stop
        last = gain.pop()
        gain[-1] = -max(-gain[-1], last)
    return gain[0], i + 1


def _rays_to(s: int, sq: int) -> list:
    return [t for t in chess.SQUARES if chess.BB_SQUARES[t] & chess.between(sq, s)]


def hypotheses(b: chess.Board, s: int) -> dict:
    """The law's hypotheses on (b, s), each named -> bool."""
    c = b.turn
    h = {k: False for k in HYPS}
    p = b.piece_at(s)
    A, D = participants(b, s)
    h["target"] = p is not None and p.color != c and p.piece_type != chess.KING and bool(A) \
        and any(m.from_square == A[0] and m.to_square == s for m in b.legal_moves)        # our cheapest capture is legal (not in check, not pinned)
    if not h["target"]:
        return h
    parts = A + D
    h["noKing"] = all(b.piece_at(q).piece_type != chess.KING for q in parts)
    h["noPin"] = all(not b.is_pinned(b.piece_at(q).color, q) for q in parts)
    # x-rays: a slider of either side that does not attack s now but would if one piece were removed
    xray = False
    for q, pc in b.piece_map().items():
        if pc.piece_type in SLIDERS and q not in parts and q != s and chess.BB_SQUARES[s] & chess.BB_RAYS[q][s] if chess.BB_RAYS[q][s] else False:
            kind_ok = pc.piece_type == chess.QUEEN or (pc.piece_type == chess.ROOK and (chess.square_file(q) == chess.square_file(s) or chess.square_rank(q) == chess.square_rank(s))) \
                or (pc.piece_type == chess.BISHOP and chess.square_file(q) != chess.square_file(s) and chess.square_rank(q) != chess.square_rank(s))
            if kind_ok:
                blockers = [t for t in _rays_to(s, q) if b.piece_at(t) is not None]
                if len(blockers) == 1:
                    xray = True
    h["noXray"] = not xray
    seventh = lambda q: chess.square_rank(q) == (6 if b.piece_at(q).color == chess.WHITE else 1)     # its own seventh rank
    h["noPromotion"] = all(not (b.piece_at(q).piece_type == chess.PAWN and seventh(q)) for q in parts) and b.ep_square != s
    # captures elsewhere, read in the position after our first capture (the target is gone): no enemy capture off s pays by
    # the static exchange from their side, and no participant of ours defends a piece of ours they attack (an overloaded
    # defender would leave it loose when it recaptures on s)
    b1 = b.copy(stack=False); b1.push(next(m for m in b.legal_moves if m.from_square == A[0] and m.to_square == s))
    ok = True
    for q, pc in b1.piece_map().items():
        if pc.color != c or pc.piece_type == chess.KING or q == s:
            continue
        if not b1.attackers(not c, q):
            continue
        if set(A) & set(b1.attackers(c, q)):                     # a participant of ours is among its defenders: overload
            ok = False; break
        if see(b1, q)[0] > 0:                                     # the enemy, to move in b1, would win material there
            ok = False; break
    h["noOtherCapture"] = ok
    k = b.king(c)
    nb = b.copy(stack=False); nb.turn = not c
    enemy_checks = any(nb.gives_check(m) for m in nb.legal_moves) if k is not None and not nb.is_check() else True
    king_rel = k is not None and (chess.BB_SQUARES[k] & (chess.BB_RAYS[s][k] if chess.BB_RAYS[s][k] else 0) or chess.square_distance(s, k) <= 2)
    slider_line = k is not None and any(pc.color != c and pc.piece_type in SLIDERS and chess.BB_RAYS[q][k] for q, pc in b.piece_map().items())
    h["noCheck"] = not enemy_checks and not king_rel and not slider_line
    h["noInterpose"] = all(b.piece_at(q).piece_type in (chess.KNIGHT, chess.PAWN) or chess.square_distance(q, s) == 1 for q in A)
    arrives = False
    for m in nb.legal_moves:                                      # the enemy's moves now (a null move): can a non-participant come to attack s?
        if m.from_square in D or m.from_square == s or m.to_square == s:                   # the target cannot move: we capture first
            continue
        nb2 = nb.copy(stack=False); nb2.push(m)
        if m.to_square in nb2.attackers(not c, s):
            arrives = True; break
    h["noArrival"] = not arrives
    return h


def law(b: chess.Board, s: int) -> dict:
    """-> dict(holds, value, captures, failing): the law's verdict on capturing s."""
    h = hypotheses(b, s)
    failing = [k for k in HYPS if not h[k]]
    out = dict(holds=not failing, failing=failing, hyps=h, value=None, captures=0)
    if h["target"]:
        out["value"], out["captures"] = see(b, s)
    return out


def bridge(b: chess.Board, s: int, depth: int) -> float:
    """The exact material outcome for the side to move after capturing on s with its cheapest attacker: minimax over every
    legal move to `depth` plies (material at the leaves, from the mover's view, relative to now); checkmate +-1000."""
    c = b.turn
    A, _ = participants(b, s)
    m0 = next(m for m in b.legal_moves if m.from_square == A[0] and m.to_square == s)
    base = material(b, c)
    def quiesce(bb, alpha, beta, us, qd=8):
        """Captures only at the horizon, either side free to stand pat (review 2026-10-07: a leaf after our capture
        left the opponent's pending recapture uncounted and flattered the attacker)."""
        if bb.is_checkmate():
            return -1000.0 if bb.turn == us else 1000.0
        stand = material(bb, us) - base
        if qd == 0 or bb.is_stalemate():
            return stand
        mover = bb.turn == us
        best = stand
        caps = sorted((m for m in bb.legal_moves if bb.is_capture(m)), key=lambda m: -VAL.get((bb.piece_at(m.to_square) or chess.Piece(chess.PAWN, not bb.turn)).piece_type, 1))
        for m in caps:
            bb.push(m); v = quiesce(bb, alpha, beta, us, qd - 1); bb.pop()
            if mover:
                best = max(best, v); alpha = max(alpha, v)
            else:
                best = min(best, v); beta = min(beta, v)
            if alpha >= beta:
                break
        return best

    def ab(bb, d, alpha, beta, us):
        if bb.is_checkmate():
            return -1000.0 if bb.turn == us else 1000.0
        if bb.is_stalemate():
            return material(bb, us) - base
        if d == 0:
            return quiesce(bb, alpha, beta, us)
        moves = sorted(bb.legal_moves, key=lambda m: not bb.is_capture(m))
        if bb.turn == us:
            v = -1e9
            for m in moves:
                bb.push(m); v = max(v, ab(bb, d - 1, alpha, beta, us)); bb.pop()
                alpha = max(alpha, v)
                if alpha >= beta: break
            return v
        v = 1e9
        for m in moves:
            bb.push(m); v = min(v, ab(bb, d - 1, alpha, beta, us)); bb.pop()
            beta = min(beta, v)
            if alpha >= beta: break
        return v
    bb = b.copy(stack=False); bb.push(m0)
    return ab(bb, depth - 1, -1e9, 1e9, c)


def census(tag: str, limit: int = 3000, max_depth: int = 5) -> None:
    from .tactic_census_measure import load
    rows = load(tag)[:limit]
    fired = agree = wrong = long = 0; fails = Counter(); frames = 0; examples = []
    for r in rows:
        b = chess.Board(r["fen"]); c = b.turn
        for s, pc in b.piece_map().items():
            if pc.color == c or pc.piece_type == chess.KING or not b.attackers(c, s):
                continue
            frames += 1
            v = law(b, s)
            if not v["holds"]:
                fails[v["failing"][0]] += 1
                continue
            fired += 1
            depth = 2 * v["captures"] - 1
            if depth > max_depth:
                long += 1; continue
            got = bridge(b, s, depth)
            if got >= v["value"] - 1e-9:
                agree += 1
            else:
                wrong += 1
                if len(examples) < 5:
                    examples.append((r["fen"], chess.square_name(s), v["value"], got, depth))
    print(f"OBS exchange law on {len(rows)} positions of {tag}: attacked enemy pieces {frames}; the law fires {fired} ({100 * fired / max(frames, 1):.1f}%); "
          f"bridge-checked {agree + wrong} (sequences within {max_depth} plies; {long} longer): agree {agree}, WRONG {wrong}")
    print("OBS   first failing hypothesis where it does not fire: " + ", ".join(f"{k} {v}" for k, v in fails.most_common()))
    for e in examples:
        print(f"OBS   wrong: {e[0]} on {e[1]}: law {e[2]:+d}, exact {e[3]:+.0f} at depth {e[4]}")


def bite(tag: str, limit: int = 1000, max_depth: int = 5) -> None:
    """How often each exception BITES: over every frame where the static exchange says the capture wins (see > 0), the
    exact search's outcome against see, grouped by the first failing hypothesis (or 'honest' when all hold). A bite is an
    exact outcome below the static one -- the tactic the defender had. Also the reverse: frames where see says the
    capture loses but the exact search wins material -- the attacker's tactic."""
    from .tactic_census_measure import load
    rows = load(tag)[:limit]
    by = {}; rev = Counter(); rev_n = Counter()
    for r in rows:
        b = chess.Board(r["fen"]); c = b.turn
        for s, pc in b.piece_map().items():
            if pc.color == c or pc.piece_type == chess.KING or not b.attackers(c, s):
                continue
            v = law(b, s)
            if v["value"] is None:
                continue
            depth = min(2 * v["captures"] - 1, max_depth)
            key = "honest" if v["holds"] else v["failing"][0]
            if v["value"] > 0:
                got = bridge(b, s, depth)
                cell = by.setdefault(key, [0, 0, 0.0]); cell[0] += 1; cell[1] += got < v["value"] - 1e-9; cell[2] += min(0.0, got - v["value"])
            elif v["value"] <= 0 and not v["holds"]:
                got = bridge(b, s, depth)
                rev_n[key] += 1; rev[key] += got > 0
    print(f"OBS exception bite rates on {len(rows)} positions of {tag} (frames where the static exchange says the capture WINS):")
    for key, (n, bites, short) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"OBS   {key:15s} n {n:5d}: the exact outcome falls short in {bites:4d} = {100 * bites / n:5.1f}%; mean shortfall when it does {short / max(bites, 1):+.2f} pawns")
    print("OBS the reverse (the static exchange says the capture loses; the exact search wins material anyway): " +
          ", ".join(f"{k} {rev[k]}/{rev_n[k]}" for k, _ in sorted(rev_n.items(), key=lambda kv: -kv[1])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "bite"])
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args(argv)
    (census if a.cmd == "census" else bite)(a.tag, a.limit)


if __name__ == "__main__":
    main()
