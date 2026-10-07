"""THE COVERAGE GAP of the tactical vocabulary (Kaveh 2026-10-07): a TACTIC is "a piece that is lost despite being visibly
defended -- at first glance it is defended, but not really -- or not immediately under threat, but it will be lost in the
near future, there is no way around it"; a pawn counts. On the tactic census:
  a FIRED TACTIC against V, started at a position where A is to move: a piece of V (not the king) that at that position
  is VISIBLY SAFE -- not attacked by A, or attacked but defended by V with no attacker cheaper than itself -- and is
  captured within the next 6 plies (followed by identity if it moves) with V's material down 1+ pawn at the end of the
  window, and FORCED: the engine (Stockfish, depth 10, from the census) shows V a pawon or more worse at its next turn
  than at its previous one (so V's reply could not save it; a later error by V is not a tactic). Plain hangs (attacked
  and undefended, or attacked by something cheaper) are counted apart, for context.
  What saw it at the start (A's tokens):
    motif, attributed     a MOTIF family (back rank, passer, pin, discovery) at k = 0 whose own strike is the move played
    motif, co-occurring   a motif family at k = 0 on the board, but the move played is not its strike
    search only           no motif, but the short exact searches saw a win (double_threat: the 3-ply local check;
                          material_now: the one-ply read)
    nearly                a motif family at k = 1
    gap                   nothing
Checked 2026-10-07 (Kaveh: "don't gate on a fake measurement"): the first version counted the double threat as a named
family, read losses one ply late, took co-occurrence for attribution, dropped the last plies of a game, and used a
2-pawn swing for "a tactic"; this version uses Kaveh's definition.

    python -m catspace.incubator.analyst.tactic_coverage --tag c300
Prints OBS lines."""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

import chess

MOTIF = ("back_rank", "passer", "pin", "discovery")
WINDOW = 6
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def strike_moves(b: chess.Board, family: str) -> set:
    """The family's own strikes in `b` (UCI), by its detector."""
    from . import discovery, pin
    from .lean_ideas import detector
    if family == "pin":
        return {m.uci() for m, _, _ in pin.strikes(b)}
    if family == "discovery":
        return {m.uci() for m, _, _ in discovery.strikes(b)}
    if family == "back_rank":
        return {i.move.uci() for i in detector().backrank(b) if i.status == "MATE"}
    if family == "passer":
        fwd = 8 if b.turn == chess.WHITE else -8
        return {chess.Move(i.square, i.square + fwd).uci() for i in detector().passer2(b) if i.status == "PROVED"}
    return set()


def classify(r: dict, played: str | None) -> str:
    toks = r["stm"] or []
    motifs0 = {t[0] for t in toks if t[1] == 0 and t[0] in MOTIF}
    if motifs0:
        if played is not None and any(played in strike_moves(chess.Board(r["fen"]), f) for f in motifs0):
            return "motif, attributed"
        return "motif, co-occurring"
    if any(t[1] == 0 for t in toks if t[0] in ("double_threat", "material_now")):
        return "search only"
    if any(t[1] == 1 for t in toks if t[0] in MOTIF):
        return "nearly"
    return "gap"


def visibly_safe(b: chess.Board, sq: int) -> str:
    """'safe' (not attacked), 'defended' (attacked, defended, no cheaper attacker), or 'hanging'."""
    p = b.piece_at(sq); v = b.turn                                  # b: the attacker to move; the piece is the victim's
    att = b.attackers(v, sq)
    if not att:
        return "safe"
    cheaper = any(VAL[b.piece_at(a).piece_type] < VAL[p.piece_type] for a in att)
    if b.attackers(not v, sq) and not cheaper:
        return "defended"
    return "hanging"


def follow(moves: list, b: chess.Board, sq: int) -> tuple:
    """Follow the piece on `sq` through `moves` (played from `b`): -> ('captured', ply index) or ('kept', None)."""
    cur = sq; b = b.copy(stack=False)
    for i, m in enumerate(moves):
        if m.to_square == cur and b.piece_at(cur) is not None and b.piece_at(cur).color != b.turn:
            return "captured", i
        if b.is_en_passant(m) and cur == m.to_square + (-8 if b.turn == chess.WHITE else 8):
            return "captured", i
        if m.from_square == cur:
            cur = m.to_square
        elif b.is_castling(m) and b.piece_at(cur) == chess.Piece(chess.ROOK, b.turn):
            rank = chess.square_rank(m.from_square)
            if cur == chess.square(7 if b.is_kingside_castling(m) else 0, rank):
                cur = chess.square(5 if b.is_kingside_castling(m) else 3, rank)
        b.push(m)
    return "kept", None


def game_moves(plies: dict) -> dict:
    """{ply: the move played from that ply's position}, recovered from consecutive FENs."""
    out = {}
    for t in sorted(plies):
        nxt = plies.get(t + 1)
        if nxt is None:
            continue
        b = chess.Board(plies[t]["fen"]); target = chess.Board(nxt["fen"]).board_fen()
        for m in b.legal_moves:
            b.push(m); same = b.board_fen() == target; b.pop()
            if same:
                out[t] = m; break
    return out


def events_of(plies: dict) -> list:
    """The fired tactics of one game: [(t, square, piece, kind, forced, net, class)]."""
    moves = game_moves(plies); last = max(plies); out = []
    for t in sorted(plies):
        if t not in moves:
            continue
        r = plies[t]; b = chess.Board(r["fen"]); A = b.turn; V = not A
        seq = [moves[k] for k in range(t, min(t + WINDOW, last)) if k in moves]
        end = plies[min(t + WINDOW, last)]
        net = (end["material"] if end["side"] == r["side"] else -end["material"]) - r["material"]      # from A's view
        prev, nxt = plies.get(t - 1), plies.get(t + 1)
        forced = None if prev is None or nxt is None else (nxt["eval"] - prev["eval"] <= -100)        # V's evals, V's turns
        for sq, p in b.piece_map().items():
            if p.color != V or p.piece_type == chess.KING:
                continue
            kind = visibly_safe(b, sq)
            got, at = follow(seq, b, sq)
            if got != "captured" or net < 1:
                continue
            out.append((t, sq, p.piece_type, kind, forced, net, classify(r, moves[t].uci()) if kind != "hanging" else None))
    return out


def main(tag: str = "c300") -> None:
    from .tactic_census_measure import load
    rows = load(tag)
    by = defaultdict(dict)
    for r in rows:
        by[r["game"]][r["ply"]] = r
    kinds = Counter(); cls = Counter(); pieces = Counter(); fams = Counter(); n_forced = 0
    for g, plies in by.items():
        for t, sq, pt, kind, forced, net, c in events_of(plies):
            kinds[(kind, forced)] += 1
            if kind != "hanging" and forced:
                n_forced += 1; cls[c] += 1; pieces[(c, "pawn" if pt == chess.PAWN else "piece")] += 1
                for f in {x[0] for x in plies[t]["stm"] if x[1] == 0 and x[0] in MOTIF}:
                    fams[f] += 1
    lost = sum(kinds.values())
    print(f"OBS pieces lost within {WINDOW} plies with the loser's material down 1+ ({lost} in {len(by)} games): "
          f"hanging {sum(v for (k, f), v in kinds.items() if k == 'hanging')}; visibly safe or defended but lost: "
          f"forced {sum(v for (k, f), v in kinds.items() if k != 'hanging' and f)}, not forced (a later error) "
          f"{sum(v for (k, f), v in kinds.items() if k != 'hanging' and f is False)}, unknown {sum(v for (k, f), v in kinds.items() if k != 'hanging' and f is None)}")
    print(f"OBS   of the forced: not attacked at the start {sum(v for (k, f), v in kinds.items() if k == 'safe' and f)}, "
          f"visibly defended {sum(v for (k, f), v in kinds.items() if k == 'defended' and f)}")
    print(f"OBS coverage of the {n_forced} FIRED TACTICS (Kaveh's definition) at their start, by what saw them:")
    for c in ("motif, attributed", "motif, co-occurring", "search only", "nearly", "gap"):
        v = cls[c]
        print(f"OBS    {c:22s} {v:5d} = {100 * v / max(n_forced, 1):5.1f}%  (pawns {pieces[(c, 'pawn')]}, pieces {pieces[(c, 'piece')]})")
    print("OBS    motif families at k = 0 at a tactic's start: " + ", ".join(f"{f} {v}" for f, v in fams.most_common()))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "c300")
