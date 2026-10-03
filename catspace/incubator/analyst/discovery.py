"""THE DISCOVERED CHECK AS AN ARMED LEMMA (Kaveh 2026-10-03, his own game 6IDKdXUd: "I saw the discovered check a few
moves before ... as long as a piece wasn't protected, I could win it, and a pawn for a bishop. So I kicked the other
bishop away, took with my knight, setting up a baby trap, then I struck when the opponent took back with their queen.
I want lemmas like this discovery check and the cost calculation to be precalculated from a different game,
remembered, imagined, set as a high level goal, and when the tactic presented itself I could execute without delay";
his pick: one lemma end to end).

The lemma, in four stages:
- BATTERY: a slider of ours (the REAR: bishop, rook, queen) whose line runs through one of our own pieces (the FRONT)
  and on past it (the LINE: the squares beyond the front, up to the first piece there).
- ARMED: the front has a move that GIVES CHECK (Bxh7+): the opponent must answer the check, so whatever stands on the
  line cannot simply step away. Its COST is known in advance: what the front gives up (its value if it can be taken
  where it lands) less what it takes on the way (Bxh7+ Kxh7: a bishop for a pawn, 2).
- STRIKE: an enemy piece worth more than the cost stands on the line, and the front's checking move wins it (verified
  by the zone search on that one move). Played at once -- no ordinary search needed.
- LURE (the high-level goal while armed): a move of ours after which the natural recapture puts an enemy piece worth
  more than the cost ON the line -- 17.Nxd4: the knight takes on d4, the queen takes back on d4, and the trap is sprung.

    python -m catspace.incubator.analyst.discovery --game          # replays Kaveh's game, move by move
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
DIRS = {chess.BISHOP: [(1, 1), (1, -1), (-1, 1), (-1, -1)], chess.ROOK: [(1, 0), (-1, 0), (0, 1), (0, -1)]}
DIRS[chess.QUEEN] = DIRS[chess.BISHOP] + DIRS[chess.ROOK]
KAVEH_GAME = ("1. d4 Nf6 2. Nc3 b6 3. e4 Bb7 4. Bd3 e6 5. a3 Be7 6. f3 O-O 7. Nge2 d6 8. Be3 Nbd7 9. O-O c5 10. e5 Nd5 "
              "11. Nxd5 Bxd5 12. exd6 Bxd6 13. dxc5 Nxc5 14. Bxc5 Bxc5+ 15. Kh1 Bd4 16. c4 Bb7 17. Nxd4 Qxd4 18. Bxh7+")


@dataclass
class Battery:
    rear: int                       # our slider
    front: int                      # our piece in front of it
    line: list                      # the squares beyond the front, up to and including the first piece there
    target: int | None              # that first piece if it is the enemy's
    departures: list = field(default_factory=list)   # [(move, cost)]: the front's checking moves off the line

    def text(self, b: chess.Board) -> str:
        n = lambda s: f"{b.piece_at(s).symbol()}{chess.square_name(s)}" if b.piece_at(s) else chess.square_name(s)
        deps = ", ".join(f"{b.san(m)} (cost {c:g})" for m, c in self.departures) or "none"
        return (f"{n(self.rear)} behind {n(self.front)}, line {chess.square_name(self.line[0])}-{chess.square_name(self.line[-1])}"
                f"{' onto ' + n(self.target) if self.target is not None else ''}; checking departures: {deps}")


def _ray(sq: int, df: int, dr: int):
    f, r = chess.square_file(sq) + df, chess.square_rank(sq) + dr
    while 0 <= f < 8 and 0 <= r < 8:
        yield chess.square(f, r)
        f += df; r += dr


def _cost(b: chess.Board, m: chess.Move) -> float:
    """What the front's move gives up: its value if the enemy can take it where it lands (after the check, as the
    reply), less what it captures."""
    pc = b.piece_at(m.from_square)
    nb = b.copy(stack=False); nb.push(m)
    lost = VAL[pc.piece_type] if nb.is_attacked_by(nb.turn, m.to_square) else 0
    taken = VAL[b.piece_type_at(m.to_square)] if b.piece_at(m.to_square) else 0
    return float(lost - taken)


def batteries(b: chess.Board, side: chess.Color | None = None) -> list:
    """Every battery of `side` (default: the side to move) with its checking departures and their cost."""
    side = b.turn if side is None else side
    out = []
    for rear in chess.SquareSet(b.occupied_co[side]):
        pt = b.piece_type_at(rear)
        if pt not in DIRS:
            continue
        for df, dr in DIRS[pt]:
            front = None; line = []
            for s in _ray(rear, df, dr):
                p = b.piece_at(s)
                if front is None:
                    if p is None:
                        continue
                    if p.color != side or p.piece_type == chess.KING:
                        break
                    front = s
                    continue
                line.append(s)
                if p is not None:
                    break
            if front is None or not line:
                continue
            last = line[-1]
            target = last if b.piece_at(last) is not None and b.piece_at(last).color != side else None
            bat = Battery(rear, front, line, target)
            if b.turn == side:
                ray = set(_ray(rear, df, dr))
                for m in b.legal_moves:
                    if m.from_square == front and m.to_square not in ray and b.gives_check(m):
                        bat.departures.append((m, _cost(b, m)))
            out.append(bat)
    return out


def armed(b: chess.Board) -> list:
    """The side to move's batteries with a checking departure: the lemma is armed (whether or not a target stands on
    the line yet)."""
    return [x for x in batteries(b) if x.departures]


def _verify(b: chess.Board, m: chess.Move, depth: int = 4) -> float:
    """The zone search's value of move m (frames, our own search; no Stockfish)."""
    from .fates import clear_caches
    from .zone_search import TacticalSpec, ZoneSearch
    clear_caches()
    zs = ZoneSearch(spec=TacticalSpec(depths=(depth,), budget=20000))
    return float(zs.rank(b, [m])[m][0])


def strikes(b: chess.Board) -> list:
    """Armed batteries whose line holds an enemy piece worth more than the departure's cost, and the departure wins
    material by the zone search: [(move, gain by the search, battery)] -- play at once."""
    out = []
    for bat in armed(b):
        if bat.target is None:
            continue
        tv = VAL[b.piece_type_at(bat.target)]
        for m, cost in bat.departures:
            if tv > cost:
                g = _verify(b, m)
                if g >= 1:
                    out.append((m, g, bat))
    return sorted(out, key=lambda x: -x[1])


def lures(b: chess.Board) -> list:
    """While armed: our captures ON the battery line whose natural recapture puts an enemy piece worth more than the
    cost on the line -- after which the strike works. [(move, the recapturing piece, gain of the strike then)]."""
    out = []
    for bat in armed(b):
        line = set(bat.line)
        for m in b.legal_moves:
            if m.to_square not in line or not b.is_capture(m) or m.from_square in (bat.front, bat.rear):
                continue
            nb = b.copy(stack=False); nb.push(m)
            for r in nb.legal_moves:
                if r.to_square != m.to_square or not nb.is_capture(r):
                    continue                                          # the natural reply: taking back on that square
                rb = nb.copy(stack=False); rb.push(r)
                for s, g, _ in strikes(rb):
                    out.append((m, f"{nb.piece_at(r.from_square).symbol()}{chess.square_name(r.from_square)}", g, s))
    return out


def replay(pgn: str, side: chess.Color, start: int = 10, log=print) -> None:
    """Kaveh's game move by move from `side`'s chair: what is armed, the lures, the strikes."""
    import io
    import chess.pgn
    g = chess.pgn.read_game(io.StringIO(pgn)); b = g.board()
    for m in g.mainline_moves():
        if b.turn == side and b.fullmove_number >= start:
            arm = armed(b); st = strikes(b); lu = lures(b)
            log(f"move {b.fullmove_number} ({b.san(m)} played): armed {len(arm)}"
                + "".join(f"\n     armed: {x.text(b)}" for x in arm)
                + "".join(f"\n     LURE: {b.san(mv)} -- the {rec} takes back onto the line; then {sm.uci()} wins {g:+g}" for mv, rec, g, sm in lu)
                + "".join(f"\n     STRIKE: {b.san(sm)} wins {g:+g} (zone search)" for sm, g, _ in st))
        b.push(m)


def _game_rows(pgn: str) -> list:
    """Every position of one game: armed? the best strike and whether it was played; a lure and whether it was played;
    after a played lure, did the opponent take back onto the line, and was the strike played next?"""
    import io
    import chess.pgn
    g = chess.pgn.read_game(io.StringIO(pgn)); b = g.board(); moves = list(g.mainline_moves()); rows = []
    for i, m in enumerate(moves):
        arm = armed(b)
        if arm:
            st = strikes(b); lu = lures(b)
            mates = b.copy(stack=False); mates.push(m)
            played_value = (1e3 if mates.is_checkmate() else _verify(b, m)) if st and m != st[0][0] else None
            row = dict(game=g.headers.get("Site", ""), ply=i, fen=b.fen(), played=b.san(m),
                       strike=(b.san(st[0][0]), st[0][1]) if st else None, struck=bool(st) and m == st[0][0],
                       missed=bool(st) and m != st[0][0] and played_value < st[0][1] - 1,   # not a mate, and clearly worse by the
                                                                                               # same search (2026-10-03: mates read as misses)
                       lure=[(b.san(x[0]), x[1], x[2]) for x in lu][:3], lured=any(m == x[0] for x in lu))
            if row["lured"] and i + 2 < len(moves):                    # the trap: did they take back onto the line, and was it sprung?
                after = b.copy(); after.push(m)
                reply = moves[i + 1]
                row["took_back"] = reply.to_square == m.to_square and after.is_capture(reply)
                after.push(reply)
                row["sprung"] = any(moves[i + 2] == s for s, _, _ in strikes(after)) if row["took_back"] else False
            rows.append(row)
        b.push(m)
    return rows


def games(n: int = 2000, lo: int = 1100, hi: int = 1700, workers: int = 10, log=print) -> None:
    """The lemma on real games: rated blitz (3-10 minutes), both players rated lo-hi (Kaveh's game: 1334-1321)."""
    import io
    import json
    import multiprocessing as mp
    import time
    import zstandard
    from catspace.io import paths
    t0 = time.perf_counter(); pgns = []; buf = []
    fh = open(paths.assets_dir() / "corpora" / "raw" / "lichess_db_standard_rated_2019-01.pgn.zst", "rb")
    rd = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
    head = {}
    for line in rd:
        if line.startswith("[") and not buf:
            k, _, v = line[1:].partition(" "); head[k] = v.strip().rstrip("]").strip('"')
            continue
        if line.startswith("1.") or buf:
            buf.append(line)
            if line.strip() == "" or line.rstrip().endswith(("1-0", "0-1", "1/2-1/2", "*")):
                tc = head.get("TimeControl", "-")
                try:
                    base = int(tc.split("+")[0]); we = int(head.get("WhiteElo", 0)); be = int(head.get("BlackElo", 0))
                except ValueError:
                    base = we = be = 0
                if 180 <= base <= 600 and lo <= we <= hi and lo <= be <= hi and head.get("Termination") == "Normal":
                    pgns.append(f'[Site "{head.get("Site", "")}"]\n\n' + "".join(buf))
                buf = []; head = {}
                if len(pgns) >= n:
                    break
    log(f"[discovery] {len(pgns)} games read ({time.perf_counter() - t0:.0f}s)")
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for rs in pool.imap_unordered(_game_rows, pgns, chunksize=4) for r in rs]
    json.dump(rows, open(paths.derived("discovery_games.json"), "w"))
    st = [r for r in rows if r["strike"]]; big = [r for r in st if r["strike"][1] >= 3]
    lu = [r for r in rows if r["lure"]]; played = [r for r in lu if r["lured"]]
    log(f"OBS {len(pgns)} games ({time.perf_counter() - t0:.0f}s): armed positions {len(rows)} in {len({r['game'] for r in rows})} games; "
        f"a strike available in {len(st)} positions (winning 3+ in {len(big)}), struck {sum(r['struck'] for r in st)} "
        f"(of the 3+: {sum(r['struck'] for r in big)}), missed -- the move played clearly worse by the same search -- {sum(r['missed'] for r in st)} "
        f"(of the 3+: {sum(r['missed'] for r in big)}); a lure available in {len(lu)}, played {len(played)}, the opponent took back onto "
        f"the line {sum(r.get('took_back', False) for r in played)}, and the strike followed {sum(r.get('sprung', False) for r in played)}")
    for r in [r for r in big if r["missed"]][:5]:
        log(f"   MISSED {r['game']} ply {r['ply']}: played {r['played']}, the strike {r['strike'][0]} wins {r['strike'][1]:+g}  {r['fen']}")
    for r in [r for r in played if r.get("sprung")][:5]:
        log(f"   TRAP SPRUNG {r['game']} ply {r['ply']}: lure {r['played']}, then the strike")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--game", action="store_true"); ap.add_argument("--games", type=int, default=0)
    a = ap.parse_args(argv)
    if a.game:
        replay(KAVEH_GAME, chess.WHITE)
    if a.games:
        games(a.games)


if __name__ == "__main__":
    main()
