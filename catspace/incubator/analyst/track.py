"""track.py -- the state across plies (Kaveh 2026-09-09: "the ledger that keeps track of all the threats has to keep
track of where the action is, which areas have changed and which have not").

Per ply, from the position before the move: both sides' free-move threats, the zones (sub-games) with their
temperatures, and what the move did to them -- EXECUTED a threat of ours, CREATED new ones, PARRIED threats of theirs
(gone after the move without being executed), IGNORED the ones that stayed; and where the action is: the zone the
move touched, matched to the previous ply's zone by piece overlap (the mover carries its membership), so 'the
same fight continues' or 'the action moved' is read off, with the zone named by its pieces, never by a wing.

    python -m catspace.incubator.analyst.track data/lichess/latest.pgn [--plies 30]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import chess
import chess.pgn

from .memo import Memo
from .threats import free_move_threats
from .subgames import static_lemmas, mate_lemmas, components
from .gauge_focus import touched, carry, same_zone


@dataclass
class Ply:
    ply: int
    san: str
    side: chess.Color
    ours_before: dict            # san -> gain: our threats before the move
    theirs_before: dict          # their threats before the move
    theirs_after: dict           # their threats after the move (our move has been played)
    ours_after: dict = field(default_factory=dict)   # our threats after the move (with a free move next)
    executed: list = field(default_factory=list)
    created: list = field(default_factory=list)      # our threats that exist after the move and did not before
    parried: list = field(default_factory=list)      # their threats gone after the move, not by being executed
    ignored: list = field(default_factory=list)      # their threats still standing after the move
    zone: set = field(default_factory=set)           # the zone the move touched (squares, after the move)
    zone_name: str = ""
    same_fight: bool | None = None                   # the move touched the previous ply's zone
    hottest: float = 0.0
    n_zones: int = 0


def zones_of(board: chess.Board, memo: Memo, threats: dict) -> tuple[list[set], list[float]]:
    lemmas = static_lemmas(board, memo) + mate_lemmas(board)
    zones = [set().union(*(l.participants for l in c)) for c in components(lemmas)]
    temps = [0.0] * len(zones)
    for side, ts in threats.items():
        for m, g, _ in ts:
            for zi, z in enumerate(zones):
                if m.from_square in z or m.to_square in z:
                    temps[zi] = max(temps[zi], g); break
    order = sorted(range(len(zones)), key=lambda i: (-temps[i], -len(zones[i])))
    return [zones[i] for i in order], [temps[i] for i in order]


def _name(board: chess.Board, zone: set) -> str:
    from .verbalize import pc
    from .facts import value
    pieces = sorted((s for s in zone if board.piece_at(s)), key=lambda s: -value(board, s))[:3]
    return "the fight around " + ", ".join(pc(board.piece_at(s).symbol(), s) for s in pieces) if pieces else "nothing"


def track(game: chess.pgn.Game, memo: Memo | None = None, plies: int = 0) -> list[Ply]:
    memo = memo if memo is not None else Memo()
    board = game.board(); out = []
    moves = list(game.mainline_moves())
    prev_zone = None
    for i, mv in enumerate(moves):
        if plies and i >= plies:
            break
        if board.is_game_over():
            break
        us = board.turn
        def led(b, side):
            b2 = b.copy(stack=False)
            return {b2.copy(stack=False).san(m) if (b2.turn == side and m in b2.legal_moves) else _san(b2, side, m): (g, m) for m, g, _ in free_move_threats(b, side, memo, top=8)}
        ours = led(board, us); theirs = led(board, not us)
        zones, temps = zones_of(board, memo, {us: [(m, g, "") for _, (g, m) in ours.items()], not us: [(m, g, "") for _, (g, m) in theirs.items()]})
        tz = touched(board, zones and _Z(zones) or _Z([]), mv) if zones else []
        san = board.san(mv)
        after = board.copy(stack=False); after.push(mv)
        theirs_after = led(after, not us) if not after.is_game_over() else {}
        ours_after = led(after, us) if not after.is_game_over() else {}
        p = Ply(i, san, us, {k: g for k, (g, _) in ours.items()}, {k: g for k, (g, _) in theirs.items()}, {k: g for k, (g, _) in theirs_after.items()},
                {k: g for k, (g, _) in ours_after.items()})
        p.executed = [san] if san in ours else []
        p.created = [k for k in ours_after if k not in ours]
        p.parried = [k for k in theirs if k not in theirs_after]
        p.ignored = [k for k in theirs if k in theirs_after]
        p.hottest = temps[0] if temps else 0.0; p.n_zones = len(zones)
        if tz:
            z = zones[tz[0]]; p.zone = carry(z, mv); p.zone_name = _name(after, p.zone)
            p.same_fight = same_zone(prev_zone, z) if prev_zone is not None else None
            prev_zone = p.zone
        else:
            p.same_fight = False if prev_zone is not None else None
        out.append(p)
        board.push(mv)
    return out


class _Z:
    """A stand-in with .zones for gauge_focus.touched."""
    def __init__(self, zones):
        self.zones = zones


def _san(board: chess.Board, side: chess.Color, m: chess.Move) -> str:
    b = board.copy(stack=False); b.turn = side
    try:
        return b.san(m)
    except Exception:
        return m.uci()


def lines(plies: list[Ply]) -> list[str]:
    out = []
    def fmt(k, g):
        return f"{k} ({'mate' if g >= 1000 else f'+{g:g}'})"
    for p in plies:
        who = "White" if p.side else "Black"
        bits = []
        if p.executed:
            bits.append("executes " + ", ".join(fmt(k, p.ours_before[k]) for k in p.executed))
        if p.created:
            bits.append("creates " + ", ".join(fmt(k, p.ours_after[k]) for k in sorted(p.created, key=lambda k: -p.ours_after[k])[:3]))
        if p.parried:
            bits.append("parries " + ", ".join(fmt(k, p.theirs_before[k]) for k in p.parried[:3]))
        if p.ignored:
            bits.append("leaves " + ", ".join(fmt(k, p.theirs_before[k]) for k in p.ignored[:3]))
        where = ""
        if p.zone:
            where = f" -- {'the same fight' if p.same_fight else 'the action moves to ' + p.zone_name if p.same_fight is False else p.zone_name}"
        elif p.same_fight is False:
            where = " -- outside every sub-game"
        out.append(f"{p.ply // 2 + 1}{'.' if p.side else '...'} {p.san}: " + ("; ".join(bits) if bits else "quiet") + where + (f" [{p.n_zones} zones, hottest {p.hottest:g}]" if p.n_zones else ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("pgn"); ap.add_argument("--plies", type=int, default=30)
    a = ap.parse_args(argv)
    with open(a.pgn) as f:
        game = chess.pgn.read_game(f)
    ps = track(game, plies=a.plies)
    for l in lines(ps):
        print(l)
    n = [p for p in ps if p.same_fight is not None]
    print(f"[track] {len(ps)} plies; the move stayed in the previous fight {sum(1 for p in n if p.same_fight)}/{len(n)}; executed {sum(1 for p in ps if p.executed)}, parried {sum(1 for p in ps if p.parried)}, ignored {sum(1 for p in ps if p.ignored)}")


if __name__ == "__main__":
    main()
