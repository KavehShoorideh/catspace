"""L1 OBSERVABLES (ANALYST_PLAN 41.2, 41.7): the event stream of a game through a domain adapter.

  python -m catspace.incubator.analyst.discovery.observables --pgn-index N

Prints OBS counts per kind. Everything chess-specific lives in ChessAdapter (the kinds, the settled material change as ledger.material_events computes it);
a consumer (windows.py) reads only the Observable record and the adapter's methods.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import chess

from . import ledger as LG

PVAL = LG.PVAL


@dataclass
class Observable:
    game: str
    ply: int                 # index of the move that realises it (0-based; fen_before is the position it is played from)
    kind: str
    side: str                # who made it happen
    victim: str | None       # who loses by it (None when nobody does)
    delta: int               # material, in pawns, the victim loses (0 for non-material kinds)
    fen_before: str
    fen_after: str
    source: str = "watched"  # watched | played | surprise

    def asdict(self) -> dict:
        return asdict(self)


class ChessDomain:
    """The chess DOMAIN ADAPTER: the observable kinds, positions as FEN strings and moves as UCI strings, the victim's loss, the settled end of a teacher line,
    the board picture. windows.py and any teacher see only this surface."""
    kinds = ("capture", "promotion", "check", "castle", "mate", "resignation", "shift")
    material_kinds = ("capture", "promotion", "mate")
    important_kinds = ("shift", "mate")      # only these get windows (Kaveh 2026-10-15: the windows were too many)

    @staticmethod
    def material(board: chess.Board, color: bool) -> int:
        """own material minus the other side's, in pawns."""
        return sum(PVAL.get(p.piece_type, 0) * (1 if p.color == color else -1) for p in board.piece_map().values())

    def game_id(self, game) -> str:
        return (game.headers.get("Site") or "game").rstrip("/").split("/")[-1]

    def moves_boards(self, game) -> tuple:
        b = game.board()
        moves, boards = [], []
        for m in game.mainline_moves():
            boards.append(b.copy(stack=False)); moves.append(m); b.push(m)
        boards.append(b.copy(stack=False))
        return moves, boards


    # ---- the surface windows.py uses (no python-chess outside this class) ----
    MATE_SCORE = 9000          # a teacher score at or beyond this (for the mover) means a forced mate

    def load_game(self, index: int):
        return LG.find_game(index)[0]

    def band_of(self, game) -> str:
        try:
            e = (int(game.headers["WhiteElo"]) + int(game.headers["BlackElo"])) // 2
        except (KeyError, ValueError):
            return "unknown"
        return str(e // 200 * 200)

    def describe(self, game) -> dict:
        h = game.headers
        return dict(white=h.get("White"), black=h.get("Black"), elo=[h.get("WhiteElo"), h.get("BlackElo")], site=h.get("Site"))

    def positions(self, game) -> tuple:
        """-> (moves as uci, fens before each move + the last)."""
        moves, boards = self.moves_boards(game)
        return [m.uci() for m in moves], [b.fen() for b in boards]

    def is_capture(self, fen: str, move: str) -> bool:
        b = chess.Board(fen); m = chess.Move.from_uci(move)
        return b.is_capture(m) or m.promotion is not None

    def side_to_move(self, fen: str) -> str:
        return chess.COLOR_NAMES[chess.Board(fen).turn]

    def apply(self, fen: str, move: str) -> str:
        b = chess.Board(fen); b.push_uci(move)
        return b.fen()

    def san(self, fen: str, moves: list) -> list:
        b, out = chess.Board(fen), []
        for u in moves:
            m = chess.Move.from_uci(u); out.append(b.san(m)); b.push(m)
        return out

    def loss(self, fen: str, victim: str) -> int:
        """the victim's material standing in this position (own minus the other's); its change is the victim's loss."""
        return self.material(chess.Board(fen), chess.WHITE if victim == "white" else chess.BLACK)

    def outcome(self, fen: str, move: str, line: list, score: float, obs: Observable, plies: int) -> dict:
        """classify a branch from the teacher's line after `move` (line[0] == move): SAME (the victim still loses >= delta, or is mated), AVOIDED, OTHER."""
        victim = chess.WHITE if obs.victim == "white" else chess.BLACK
        b = chess.Board(fen)
        mover = b.turn
        e = b.copy(stack=False)
        mvs = [chess.Move.from_uci(u) for u in line]
        n = 0
        while n < len(mvs) and (n < plies or e.is_capture(mvs[n])):          # the first `plies`, then on through captures to a quiet end
            e.push(mvs[n]); n += 1
        mated = None
        if abs(score) >= self.MATE_SCORE:
            mated = (not mover) if score > 0 else mover
        if e.is_checkmate():
            mated = e.turn
        swing = self.material(e, victim) - self.material(b, victim)
        same = (mated == victim) if obs.kind == "mate" else (mated == victim or swing <= -abs(obs.delta))
        other = None
        if same:
            cls = "SAME"
        elif mated is not None or swing != 0:
            cls = "OTHER"
            other = "mate against " + chess.COLOR_NAMES[mated] if mated is not None else ("victim loses" if swing < 0 else "victim gains") + f" {abs(swing)}"
        else:
            cls = "AVOIDED"
        return dict(cls=cls, other=other, swing=swing, line=self.san(fen, line[:n]), end_fen=e.fen(), end_bal=self.material(e, victim),
                    mated=None if mated is None else chess.COLOR_NAMES[mated])

    def board_svg(self, fen: str, move: str | None = None) -> str:
        import chess.svg
        return chess.svg.board(chess.Board(fen), lastmove=chess.Move.from_uci(move) if move else None, size=360, coordinates=True)

    def balance(self, fen: str, side: str) -> int:
        """material balance in `side`'s view (own minus the other's), in pawns."""
        return self.material(chess.Board(fen), chess.WHITE if side == "white" else chess.BLACK)

    def shifts(self, moves, boards) -> list:
        """(ply, side, gain) for every turn whose END balance, in its own view, differs from the end of its previous turn (the first turn compares with the start).
        The end of a turn that captured is the SETTLED end: through the capture chain that follows it (a recapture belongs to the exchange), so an even trade
        is no shift and a two-ply exchange is one. Only the gaining side emits (the loser sees the same event as a loss)."""
        out = []
        last = {c: self.material(boards[0], c) for c in (chess.WHITE, chess.BLACK)}
        for i, m in enumerate(moves):
            s = boards[i].turn
            j = i + 1
            if boards[i].is_capture(m) or m.promotion:
                while j < len(moves) and (boards[j].is_capture(moves[j]) or moves[j].promotion):
                    j += 1
            bal = self.material(boards[j], s)
            if bal > last[s]:
                out.append((i, s, bal - last[s]))
            last[s] = bal
        return out

    def stream(self, game, source: str = "watched") -> list:
        gid = self.game_id(game)
        moves, boards, events = LG.material_events(game)
        out = []

        def mk(ply, kind, side, victim, delta):
            return Observable(gid, ply, kind, chess.COLOR_NAMES[side], None if victim is None else chess.COLOR_NAMES[victim], delta,
                              boards[ply].fen(), boards[ply + 1].fen(), source)
        for e in events:
            winner = not e["loser"]
            promo = any(moves[j].promotion and boards[j].turn == winner for j in range(e["c0"], e["c1"] + 1))
            out.append(mk(e["win_first"], "promotion" if promo else "capture", winner, e["loser"], e["delta"]))
        for i, s, gain in self.shifts(moves, boards):
            out.append(mk(i, "shift", s, not s, gain))
        for i, m in enumerate(moves):
            b, a = boards[i], boards[i + 1]
            if b.is_castling(m):
                out.append(mk(i, "castle", b.turn, None, 0))
            if a.is_checkmate():
                out.append(mk(i, "mate", b.turn, a.turn, 0))
            elif a.is_check():
                out.append(mk(i, "check", b.turn, a.turn, 0))
        res = game.headers.get("Result")
        if moves and res in ("1-0", "0-1") and not boards[-1].is_checkmate() and "forfeit" not in game.headers.get("Termination", "").lower():
            w = chess.WHITE if res == "1-0" else chess.BLACK
            out.append(mk(len(moves) - 1, "resignation", w, not w, 0))
        return sorted(out, key=lambda o: (o.ply, self.kinds.index(o.kind)))


ChessAdapter = ChessDomain
ADAPTER = DOMAIN = ChessDomain()


def stream(game, source: str = "watched", adapter=DOMAIN) -> list:
    return adapter.stream(game, source)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn-index", type=int, required=True)
    a = ap.parse_args(argv)
    game, site = LG.find_game(a.pgn_index)
    obs = stream(game)
    counts = {k: sum(o.kind == k for o in obs) for k in ADAPTER.kinds}
    print(f"OBS observables {site}: {game.headers.get('WhiteElo')} v {game.headers.get('BlackElo')}, {len(obs)} in all, " + ", ".join(f"{k} {n}" for k, n in counts.items()))


if __name__ == "__main__":
    main()
