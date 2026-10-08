"""THE EXCHANGE LAW, prototyped before it is proved (Kaveh 2026-10-07: "the tactics are basically exceptions to the exchange
theorem"; the standing method: prototype -> bridge -> prove).

THE LAW. In position p with side c to move, a square s holds an enemy piece; A are c's pieces attacking s, D the enemy's.
The static exchange (the classical SEE: each side recaptures with its least valuable piece or stops) gives a value
see(p, s). The claim: if the frame is HONEST -- the hypotheses below hold -- then after c captures on s with its least
valuable attacker, c's material against every legal reply is at least see(p, s) within the sequence's length.

THE HYPOTHESES (each a computable check on the position, named; a tactic is the failure of one of them):
  target        s holds an enemy piece that is not the king, and c attacks it
  noKing        no king among the attackers or defenders of s (a king cannot capture into an attack)
  noPin         every attacker's capture on s is legal now and every defender's recapture is legal after ours
  noXray        no slider of either side is hidden behind pieces that will all capture on s in sequence (the nearest a
                participant, each behind it a slider with geometry to s); a blocker that never moves hides nothing
  noPromotion   no pawn participant stands on its seventh rank; no en passant on s; no enemy pawn could promote inside
                the claim's horizon
  noOtherCapture  after our first capture, no enemy capture off s is even or better by the static exchange, no attacker
                of ours defends a piece of ours they attack (the overload), none of our remaining attackers is attacked
                (trading one off changes the frame: the zwischenzug)
  noCheck       after our first capture the enemy has no checking move; our king shares no line or knight relation
                with s; no enemy slider shares a line with our king -- no in-between check
  noInterpose   every attacker of ours is a knight, a pawn, or stands next to s -- the enemy cannot block our recapture
  noArrival     after our first capture no move of the enemy off the frame adds an attacker of s, arrived or discovered
                -- the defenders are fixed
Every clause is the kernel's (Ideas/ExchangeLaw.lean `hypotheses`), including the tie order among equal attackers.
THE BRIDGE: an exact alpha-beta over every legal move to the sequence's depth, with a captures-only quiescence at the
leaves (either side may stand pat), which must never fall below see(p, s) when the hypotheses hold; where they fail,
the gap is the tactic.

    python -m catspace.incubator.analyst.exchange_law census --tag c300 [--limit 3000]    # fire rate, bridge agreement
    python -m catspace.incubator.analyst.exchange_law bite --tag c300                     # each exception's bite rate
    python -m catspace.incubator.analyst.exchange_law kernel --tag c300 --limit 1500      # the kernel vs this file vs the bridge
Prints OBS lines: how often the law fires, how often the bridge agrees (must be always), and which hypothesis fails most.
"""
from __future__ import annotations

import argparse
from collections import Counter

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)
HYPS = ("target", "noKing", "noPin", "noXray", "noPromotion", "noOtherCapture", "noOverload", "noCheck", "noInterpose", "noArrival")


def material(b: chess.Board, side: chess.Color) -> int:
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values() if p.piece_type != chess.KING)


def _kernel_order(sq: int) -> int:
    """The kernel's square order (`allSquares`: file-major, a1 a2 ... a8 b1 ...) for the tie among equal-value attackers."""
    return chess.square_file(sq) * 8 + chess.square_rank(sq)


def participants(b: chess.Board, s: int) -> tuple:
    """(attackers of the side to move, defenders of the other side), as lists of squares, cheapest first; equal values in
    the kernel's square order."""
    c = b.turn
    key = lambda sq: (VAL[b.piece_at(sq).piece_type], _kernel_order(sq))
    return sorted(b.attackers(c, s), key=key), sorted(b.attackers(not c, s), key=key)


def cheapest_capture(b: chess.Board, s: int, A: list) -> chess.Move | None:
    """Our capture on s by the cheapest attacker whose capture is legal (the kernel's `cheapestCapture`), or None."""
    for q in A:
        for m in b.legal_moves:
            if m.from_square == q and m.to_square == s:
                return m
    return None


def see(b: chess.Board, s: int) -> tuple:
    """The static exchange on s for the side to move: (value, number of captures in the sequence). The classical gain
    list, each side capturing with its cheapest piece and free to stop; no x-rays (the hypothesis excludes them). The
    first capture is by the cheapest attacker by VALUE whether or not it is legal (the kernel's `seeB`); the value is the
    capture's own, possibly negative."""
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


def see_x(b: chess.Board, s: int) -> tuple:
    """The static exchange WITH X-RAYS (Kaveh 2026-10-08: the battery is the x-ray that matters, recursively, for all
    pieces): when a piece captures on s and leaves its square, the next piece behind it on the same ray, if a slider with
    geometry to s, joins ITS side's attackers (either colour), cheapest first as before. -> (value, captures)."""
    c = b.turn
    A, D = participants(b, s); A, D = list(A), list(D)
    removed = set()
    key = lambda u: (VAL[b.piece_at(u).piece_type], _kernel_order(u))

    def reveal(q):
        if not chess.BB_RAYS[q][s]:
            return
        df = (chess.square_file(q) > chess.square_file(s)) - (chess.square_file(q) < chess.square_file(s))
        dr = (chess.square_rank(q) > chess.square_rank(s)) - (chess.square_rank(q) < chess.square_rank(s))
        f, r = chess.square_file(q) + df, chess.square_rank(q) + dr
        while 0 <= f < 8 and 0 <= r < 8:
            t = chess.square(f, r); p = b.piece_at(t)
            if p is not None and t not in removed:
                if p.piece_type in SLIDERS and _geometry(p.piece_type, t, s) and t not in A and t not in D:
                    lst = A if p.color == c else D
                    lst.append(t); lst.sort(key=key)
                return
            f += df; r += dr

    gain = [VAL[b.piece_at(s).piece_type]]
    q = A.pop(0); on_s = VAL[b.piece_at(q).piece_type]; removed.add(q); reveal(q)
    side = [D, A]; i = 0
    while side[i % 2]:
        q = side[i % 2].pop(0)
        gain.append(on_s - gain[-1])
        on_s = VAL[b.piece_at(q).piece_type]; removed.add(q); reveal(q); i += 1
    while len(gain) > 1:
        last = gain.pop()
        gain[-1] = -max(-gain[-1], last)
    return gain[0], i + 1


def _rays_to(s: int, sq: int) -> list:
    return [t for t in chess.SQUARES if chess.BB_SQUARES[t] & chess.between(sq, s)]


def _geometry(kind: int, q: int, s: int) -> bool:
    """A slider of `kind` on q would attack s were the line empty."""
    if not chess.BB_RAYS[q][s]:
        return False
    straight = chess.square_file(q) == chess.square_file(s) or chess.square_rank(q) == chess.square_rank(s)
    return kind == chess.QUEEN or (kind == chess.ROOK and straight) or (kind == chess.BISHOP and not straight)


def _knight_relation(a: int, b_: int) -> bool:
    return {abs(chess.square_file(a) - chess.square_file(b_)), abs(chess.square_rank(a) - chess.square_rank(b_))} == {1, 2}


def hypotheses(b: chess.Board, s: int) -> dict:
    """The law's hypotheses on (b, s), each named -> bool: the kernel's reading (Ideas/ExchangeLaw.lean `hypotheses`);
    whatever looks past our first capture is read in the position after it (b1, the opponent to move)."""
    c = b.turn
    h = {k: False for k in HYPS}
    p = b.piece_at(s)
    A, D = participants(b, s)
    m0 = cheapest_capture(b, s, A)
    h["target"] = p is not None and p.color != c and p.piece_type != chess.KING and m0 is not None
    if not h["target"]:
        return h
    parts = A + D
    n = len(parts)                                                # the claim's horizon in plies
    h["noKing"] = all(b.piece_at(q).piece_type != chess.KING for q in parts)
    b1 = b.copy(stack=False); b1.push(m0)
    to_s = lambda bb, q: any(m.from_square == q and m.to_square == s for m in bb.legal_moves)
    h["noPin"] = all(to_s(b, q) for q in A) and all(to_s(b1, q) for q in D)    # every capture on s legal when its turn comes
    # x-rays (Kaveh 2026-10-08: "recursively check if the piece will participate in sequence, for all pieces"): a slider
    # of either side off the frame, on a line to s, is HIDDEN when every piece between it and s will capture on s in turn
    # -- the nearest a participant now, each one behind it a slider with geometry to s (Alekhine's gun, pawn-bishop-queen)
    xray = False
    for q, pc in b.piece_map().items():
        if pc.piece_type in SLIDERS and q not in parts and q != s and _geometry(pc.piece_type, q, s):
            between = sorted((t for t in _rays_to(s, q) if b.piece_at(t) is not None), key=lambda t: chess.square_distance(t, s))
            if between and between[0] in parts and all(b.piece_at(t).piece_type in SLIDERS and _geometry(b.piece_at(t).piece_type, t, s) for t in between[1:]):
                xray = True
    h["noXray"] = not xray
    seventh = lambda q: chess.square_rank(q) == (6 if b.piece_at(q).color == chess.WHITE else 1)     # its own seventh rank
    steps = lambda q, pc: (7 - chess.square_rank(q)) if pc.color == chess.WHITE else chess.square_rank(q)   # to promotion
    pawn_near = any(pc.color != c and pc.piece_type == chess.PAWN and steps(q, pc) <= (n + 1) // 2        # an enemy pawn that
                    for q, pc in b1.piece_map().items())                                                   # could promote inside the horizon
    h["noPromotion"] = all(not (b.piece_at(q).piece_type == chess.PAWN and seventh(q)) for q in parts) and b.ep_square != s and not pawn_near
    # captures elsewhere, read after our first capture: no enemy capture off s is even or better by the static exchange
    # from their side (an even trade of a non-participant would still cost the horizon), no participant of ours defends a
    # piece of ours they attack (an overloaded defender), and none of our remaining attackers is attacked at all (trading
    # one off changes the frame: the zwischenzug)
    other = overload = True
    for q, pc in b1.piece_map().items():
        if pc.color != c or pc.piece_type == chess.KING or q == s or not b1.attackers(not c, q):
            continue
        if see(b1, q)[0] > 0:
            other = False
        if set(A) & set(b1.attackers(c, q)):
            overload = False
    h["noOtherCapture"] = other
    h["noOverload"] = overload and all(b1.piece_at(a) is None or not b1.attackers(not c, a) for a in A)
    k = b.king(c)
    enemy_checks = any(b1.gives_check(m) for m in b1.legal_moves)                      # a check in between
    king_rel = k is not None and (bool(chess.BB_RAYS[s][k]) or _knight_relation(s, k))  # our king on a line or a knight's jump from s
    slider_line = k is not None and any(pc.color != c and pc.piece_type in SLIDERS and chess.BB_RAYS[q][k] for q, pc in b.piece_map().items())
    h["noCheck"] = k is not None and not enemy_checks and not king_rel and not slider_line
    h["_checkReply"], h["_kingLine"], h["_sliderOnKing"] = enemy_checks, king_rel, slider_line   # the parts, for the names
    h["noInterpose"] = all(b.piece_at(q).piece_type in (chess.KNIGHT, chess.PAWN) or chess.square_distance(q, s) == 1 for q in A)
    arrives = False
    for m in b1.legal_moves:                                      # the enemy's moves after our capture: can a non-participant come to attack s?
        if m.from_square in D or m.to_square == s:
            continue
        nb2 = b1.copy(stack=False); nb2.push(m)
        if set(nb2.attackers(not c, s)) - set(D):                 # a new attacker of s: arrived or discovered
            arrives = True; break
    h["noArrival"] = not arrives
    return h


VARIANTS = {                                                       # Kaveh 2026-10-08: keep the names even where a tier
    "noKing": "king exchange",                                     # computes the variant with no extra code -- "this is not
    "noPin": "pinned exchange",                                    # a regular exchange, it is an intermezzo exchange, a pawn
    "noXray": "x-ray exchange",                                    # promotion exchange, an exchange where the king is a
    "noPromotion": "promotion exchange",                           # participant"
    "noOtherCapture": "intermezzo exchange",
    "noOverload": "overloaded exchange",
    "noCheck": "exchange with a check in between",
    "noInterpose": "interposition exchange",
    "noArrival": "exchange with an arriving defender",
}


def variants(b: chess.Board, s: int, h: dict | None = None) -> list:
    """The names of the exchange on s: 'plain exchange' when every hypothesis holds, else one name per failing
    hypothesis -- and the x-ray and king names whenever a hidden piece or a king takes part, whatever the hypotheses."""
    h = h or hypotheses(b, s)
    if not h["target"]:
        return []
    names = [VARIANTS[k] for k in HYPS if k != "target" and not h[k]]
    A, D = participants(b, s)
    if any(b.piece_at(q).piece_type == chess.KING for q in A + D) and "king exchange" not in names:
        names.append("king exchange")
    if see_x(b, s) != see(b, s) and "x-ray exchange" not in names:
        names.append("x-ray exchange")
    return names or ["plain exchange"]


def intermezzo(b: chess.Board, s: int) -> int:
    """The opponent's best static capture elsewhere after our first capture on s (0 when none pays): the intermezzo's
    price, which subtracts in the composed tier."""
    A, _ = participants(b, s)
    m0 = cheapest_capture(b, s, A)
    if m0 is None:
        return 0
    c = b.turn
    b1 = b.copy(stack=False); b1.push(m0)
    return max([see(b1, q)[0] for q, pc in b1.piece_map().items()
                if pc.color == c and pc.piece_type != chess.KING and q != s and b1.attackers(not c, q)] + [0])


def law(b: chess.Board, s: int) -> dict:
    """-> dict(holds, value, captures, failing, composed, composed_holds): the law's verdict on capturing s, strict (all
    hypotheses) and composed (noOtherCapture dropped, the intermezzo's price subtracted)."""
    h = hypotheses(b, s)
    failing = [k for k in HYPS if not h[k]]
    out = dict(holds=not failing, failing=failing, hyps=h, value=None, captures=0, composed=None, xray=None, xray_captures=0,
               composed_holds=h["target"] and all(h[k] for k in HYPS if k != "noOtherCapture"),
               xray_holds=h["target"] and all(h[k] for k in HYPS if k not in ("noOtherCapture", "noXray")))
    if h["target"]:
        out["value"], out["captures"] = see(b, s)
        g = intermezzo(b, s)
        out["composed"] = out["value"] - g
        vx, out["xray_captures"] = see_x(b, s)
        out["xray"] = vx - g
    out["variants"] = variants(b, s, h)
    return out


def bridge(b: chess.Board, s: int, depth: int) -> float:
    """The exact material outcome for the side to move after capturing on s with its cheapest attacker: minimax over every
    legal move to `depth` plies (material at the leaves, from the mover's view, relative to now); checkmate +-1000."""
    c = b.turn
    A, _ = participants(b, s)
    m0 = cheapest_capture(b, s, A)
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


def _workers() -> int:
    import os
    return max(1, (os.cpu_count() or 2) - 1)                      # Kaveh 2026-10-08: use the whole laptop, not 1 of 11 cores


def _frames(b: chess.Board):
    c = b.turn
    for s, pc in b.piece_map().items():
        if pc.color == c or pc.piece_type == chess.KING or not b.attackers(c, s):
            continue
        yield s


def _census_row(args):
    fen, max_depth = args
    b = chess.Board(fen); out = []
    for s in _frames(b):
        v = law(b, s); rec = dict(sq=chess.square_name(s), holds=v["holds"], failing=v["failing"], composed_holds=v["composed_holds"], xray_holds=v["xray_holds"],
                                  value=v["value"], composed=v["composed"], xray=v["xray"], got=None, depth=None, long=False)
        if v["holds"]:
            depth = 2 * v["captures"] - 1
            if depth > max_depth:
                rec["long"] = True
            else:
                rec["got"], rec["depth"] = bridge(b, s, depth), depth
        elif v["composed_holds"]:
            rec["got"] = bridge(b, s, min(2 * v["captures"] + 1, max_depth))
        elif v["xray_holds"]:
            rec["got"] = bridge(b, s, min(2 * v["xray_captures"] + 1, max_depth))
        out.append(rec)
    return fen, out


def census(tag: str, limit: int = 3000, max_depth: int = 5) -> None:
    from multiprocessing import Pool

    from .tactic_census_measure import load
    rows = load(tag)[:limit]
    fired = agree = wrong = long = 0; fails = Counter(); frames = 0; examples = []; cfired = cagree = cwrong = 0; xfired = xagree = xwrong = 0
    with Pool(_workers()) as pool:
        for fen, recs in pool.imap_unordered(_census_row, [(r["fen"], max_depth) for r in rows], chunksize=8):
            for rec in recs:
                frames += 1
                if not rec["holds"]:
                    fails[rec["failing"][0]] += 1
                    if rec["composed_holds"]:                              # the composed tier fires where only noOtherCapture failed
                        cfired += 1
                        if rec["got"] >= rec["composed"] - 1e-9:
                            cagree += 1
                        else:
                            cwrong += 1
                            if len(examples) < 5:
                                examples.append((fen, rec["sq"], rec["composed"], rec["got"], "composed"))
                    elif rec["xray_holds"]:                                # the x-ray tier: hidden pieces in the game, the intermezzo priced
                        xfired += 1
                        if rec["got"] >= rec["xray"] - 1e-9:
                            xagree += 1
                        else:
                            xwrong += 1
                            if len(examples) < 5:
                                examples.append((fen, rec["sq"], rec["xray"], rec["got"], "x-ray"))
                    continue
                fired += 1
                if rec["long"]:
                    long += 1; continue
                if rec["got"] >= rec["value"] - 1e-9:
                    agree += 1
                else:
                    wrong += 1
                    if len(examples) < 5:
                        examples.append((fen, rec["sq"], rec["value"], rec["got"], f"depth {rec['depth']}"))
    print(f"OBS exchange law on {len(rows)} positions of {tag}: attacked enemy pieces {frames}; the law fires {fired} ({100 * fired / max(frames, 1):.1f}%); "
          f"bridge-checked {agree + wrong} (sequences within {max_depth} plies; {long} longer): agree {agree}, WRONG {wrong}")
    print("OBS   first failing hypothesis where it does not fire: " + ", ".join(f"{k} {v}" for k, v in fails.most_common()))
    print(f"OBS   the composed tier (noOtherCapture dropped, the intermezzo's price subtracted) fires on {cfired} more frames: agree {cagree}, WRONG {cwrong}")
    print(f"OBS   the x-ray tier (noXray dropped too, the hidden pieces in the capture game) fires on {xfired} more frames: agree {xagree}, WRONG {xwrong}")
    for e in examples:
        print(f"OBS   wrong: {e[0]} on {e[1]}: law {e[2]:+d}, exact {e[3]:+.0f} at {e[4]}")


def _bite_row(args):
    fen, max_depth = args
    b = chess.Board(fen); out = []
    for s in _frames(b):
        v = law(b, s)
        if v["value"] is None:
            continue
        depth = min(2 * v["captures"] - 1, max_depth)
        key = "honest" if v["holds"] else v["failing"][0]
        if v["value"] > 0 or not v["holds"]:
            out.append((key, v["value"], bridge(b, s, depth)))
    return out


def bite(tag: str, limit: int = 1000, max_depth: int = 5) -> None:
    """How often each exception BITES: over every frame where the static exchange says the capture wins (see > 0), the
    exact search's outcome against see, grouped by the first failing hypothesis (or 'honest' when all hold). A bite is an
    exact outcome below the static one -- the tactic the defender had. Also the reverse: frames where see says the
    capture loses but the exact search wins material -- the attacker's tactic."""
    from multiprocessing import Pool

    from .tactic_census_measure import load
    rows = load(tag)[:limit]
    by = {}; rev = Counter(); rev_n = Counter()
    with Pool(_workers()) as pool:
        for recs in pool.imap_unordered(_bite_row, [(r["fen"], max_depth) for r in rows], chunksize=8):
            for key, value, got in recs:
                if value > 0:
                    cell = by.setdefault(key, [0, 0, 0.0]); cell[0] += 1; cell[1] += got < value - 1e-9; cell[2] += min(0.0, got - value)
                else:
                    rev_n[key] += 1; rev[key] += got > 0
    print(f"OBS exception bite rates on {len(rows)} positions of {tag} (frames where the static exchange says the capture WINS):")
    for key, (n, bites, short) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"OBS   {key:15s} n {n:5d}: the exact outcome falls short in {bites:4d} = {100 * bites / n:5.1f}%; mean shortfall when it does {short / max(bites, 1):+.2f} pawns")
    print("OBS the reverse (the static exchange says the capture loses; the exact search wins material anyway): " +
          ", ".join(f"{k} {rev[k]}/{rev_n[k]}" for k, _ in sorted(rev_n.items(), key=lambda kv: -kv[1])))


def _kernel_row(args):
    fen, max_depth = args
    from .lean_ideas import detector
    li = detector()
    b = chess.Board(fen); out = []
    for s in _frames(b):
        v = law(b, s)
        kh, kv, khonest = li.exchange(b, s)
        dis = [name for name in HYPS if bool(kh.get(name)) != bool(v["hyps"][name]) and (name == "target" or (kh.get("target") and v["hyps"]["target"]))]
        kx, kxh, kseex = li.exchange_xray(b, s)
        rec = dict(sq=chess.square_name(s), dis=dis, khonest=khonest, phonest=v["holds"], kv=kv, vdiff=v["value"] is not None and kv != v["value"],
                   xdiff=v["value"] is not None and (kseex != see_x(b, s)[0] or kxh != v["xray_holds"]), verdict=None, n=None, got=None)
        if khonest:
            A, D = participants(b, s); rec["n"] = len(A) + len(D)
            if rec["n"] > max_depth:
                rec["verdict"] = "skipped (large frame)"
            else:
                try:
                    rec["verdict"] = "SETTLED" if li.exchange_cert(b, s, timeout=30) else "NOT"
                except TimeoutError:
                    rec["verdict"] = "timeout"
            rec["got"] = bridge(b, s, min(rec["n"], max_depth))
        out.append(rec)
    return fen, out


def kernel(tag: str, limit: int = 1500, max_depth: int = 5) -> None:
    """The kernel's reading (Ideas/ExchangeLaw.lean through the binary) against the prototype's and against the exact
    search: on every frame the nine hypotheses by both (disagreements by name, which must be 0 once the prototype is the
    kernel's mirror); on every frame the KERNEL calls honest, its value against the prototype's, its own certificate
    (`settledWithin`) and the exact bridge, which must never fall below the value."""
    from multiprocessing import Pool

    from .tactic_census_measure import load
    rows = load(tag)[:limit]
    frames = k_honest = p_honest = both = vdiff = agree = wrong = xdiff = 0; dis = Counter(); cert = Counter(); values = Counter(); examples = []; not_settled = []
    with Pool(_workers()) as pool:
        for fen, recs in pool.imap_unordered(_kernel_row, [(r["fen"], max_depth) for r in rows], chunksize=8):
            for rec in recs:
                frames += 1
                dis.update(rec["dis"]); xdiff += rec["xdiff"]
                k_honest += rec["khonest"]; p_honest += rec["phonest"]; both += rec["khonest"] and rec["phonest"]
                if rec["khonest"]:
                    values[rec["kv"]] += 1; vdiff += rec["vdiff"]; cert[rec["verdict"]] += 1
                    if rec["verdict"] != "SETTLED":
                        not_settled.append((fen, rec["sq"], rec["kv"], rec["n"], rec["verdict"]))
                    if rec["got"] >= rec["kv"] - 1e-9:
                        agree += 1
                    else:
                        wrong += 1
                        if len(examples) < 5:
                            examples.append((fen, rec["sq"], rec["kv"], rec["got"]))
    print(f"OBS kernel vs prototype on {len(rows)} positions of {tag}, {frames} frames: honest by the kernel {k_honest}, by the prototype {p_honest}, both {both}; "
          f"hypothesis disagreements: " + (", ".join(f"{k} {n}" for k, n in dis.most_common()) or "none") + f"; x-ray value or tier disagreements {xdiff}")
    print(f"OBS   on the kernel's honest frames: value differs from the prototype's {vdiff}; certificate " + " ".join(f"{k} {n}" for k, n in cert.items()) + "; "
          f"exact bridge agrees {agree}, WRONG {wrong}; values " + ", ".join(f"{k:+d}: {n}" for k, n in sorted(values.items())))
    for e in examples:
        print(f"OBS   wrong: {e[0]} on {e[1]}: kernel {e[2]:+d}, exact {e[3]:+.0f}")
    for e in not_settled:
        print(f"OBS   not certified: {e[0]} on {e[1]}: value {e[2]:+d}, frame {e[3]} participants, {e[4]}")


def _xray_row(args):
    fen, max_depth = args
    b = chess.Board(fen); out = []
    for s in _frames(b):
        v = law(b, s)
        if not v["hyps"]["target"] or "noXray" not in v["failing"] or set(v["failing"]) - {"noXray", "noOtherCapture", "noArrival", "noInterpose"}:
            continue
        A, D = participants(b, s)
        vx, caps = see_x(b, s)
        pred = vx - intermezzo(b, s)
        out.append((v["value"], pred, bridge(b, s, min(2 * caps + 1, max_depth)), fen, chess.square_name(s)))
    return out


def xray(tag: str, limit: int = 8000, max_depth: int = 5) -> None:
    """Measured before built: on frames failing noXray (alone, or with noOtherCapture priced as the intermezzo, noArrival,
    noInterpose), is the exact outcome at least the static exchange WITH x-rays minus the intermezzo's price?"""
    from multiprocessing import Pool

    from .tactic_census_measure import load
    rows = load(tag)[:limit]
    n = ok = ok_naive = 0; gap = Counter(); misses = []
    with Pool(_workers()) as pool:
        for recs in pool.imap_unordered(_xray_row, [(r["fen"], max_depth) for r in rows], chunksize=8):
            for v, pred, got, fen, sq in recs:
                n += 1; ok += got >= pred - 1e-9; ok_naive += got >= v - 1e-9; gap[int(round(got - pred))] += 1
                if got < pred - 1e-9 and len(misses) < 12:
                    misses.append((fen, sq, v, pred, got))
    print(f"OBS x-ray frames in {len(rows)} positions of {tag}: {n}; exact >= plain static {ok_naive} ({100 * ok_naive / max(n, 1):.0f}%); "
          f"exact >= static WITH x-rays minus the intermezzo {ok} ({100 * ok / max(n, 1):.0f}%)")
    print("OBS   exact minus predicted: " + ", ".join(f"{k:+d}: {c}" for k, c in sorted(gap.items())))
    for m in misses:
        print(f"OBS   miss: {m[0]} on {m[1]}: plain {m[2]:+d}, predicted {m[3]:+d}, exact {m[4]:+.0f}")


def _king_row(args):
    fen, max_depth = args
    b = chess.Board(fen); out = []
    for s in _frames(b):
        v = law(b, s); h = v["hyps"]
        if not h["target"] or h["noKing"]:
            continue
        A, D = participants(b, s)
        ours = any(b.piece_at(q).piece_type == chess.KING for q in A)
        # the frame apart from the king: the hypotheses the tiers already price, and noCheck failing ONLY because our own
        # participating king stands next to the square (the king line clause), not by a checking reply or a slider on it
        allowed = {"noKing", "noOtherCapture", "noXray", "noArrival", "noInterpose"}
        check_only_king = not h["noCheck"] and not h["_checkReply"] and not h["_sliderOnKing"] and ours
        rest = set(v["failing"]) - allowed - ({"noCheck"} if check_only_king else set())
        if rest:
            continue
        vx, caps = see_x(b, s)
        pred = vx - intermezzo(b, s)
        out.append(("ours" if ours else "theirs", v["value"], pred, bridge(b, s, min(2 * caps + 1, max_depth)), fen, chess.square_name(s)))
    return out


def king(tag: str, limit: int = 8000, max_depth: int = 5) -> None:
    """Measured before built: on frames where a king takes part (ours or theirs; the king goes last -- at 100 it never
    captures into a defended square, so the exchange's fold-back stops before it), with the priced hypotheses allowed to
    fail, is the exact outcome at least the static exchange with x-rays minus the intermezzo's price?"""
    from multiprocessing import Pool

    from .tactic_census_measure import load
    rows = load(tag)[:limit]
    n = Counter(); ok = Counter(); ok_naive = Counter(); gap = Counter(); misses = []
    with Pool(_workers()) as pool:
        for recs in pool.imap_unordered(_king_row, [(r["fen"], max_depth) for r in rows], chunksize=8):
            for who, v, pred, got, fen, sq in recs:
                n[who] += 1; ok[who] += got >= pred - 1e-9; ok_naive[who] += got >= v - 1e-9; gap[int(round(got - pred))] += 1
                if got < pred - 1e-9 and len(misses) < 12:
                    misses.append((who, fen, sq, v, pred, got))
    for who in ("ours", "theirs"):
        print(f"OBS king exchange ({who} king takes part) in {len(rows)} positions of {tag}: {n[who]} frames; exact >= plain static {ok_naive[who]} "
              f"({100 * ok_naive[who] / max(n[who], 1):.0f}%); exact >= x-ray static minus the intermezzo {ok[who]} ({100 * ok[who] / max(n[who], 1):.0f}%)")
    print("OBS   exact minus predicted: " + ", ".join(f"{k:+d}: {c}" for k, c in sorted(gap.items())))
    for m in misses:
        print(f"OBS   miss ({m[0]}): {m[1]} on {m[2]}: plain {m[3]:+d}, predicted {m[4]:+d}, exact {m[5]:+.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "bite", "kernel", "xray", "king"])
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args(argv)
    dict(census=census, bite=bite, kernel=kernel, xray=xray, king=king)[a.cmd](a.tag, a.limit)


if __name__ == "__main__":
    main()
