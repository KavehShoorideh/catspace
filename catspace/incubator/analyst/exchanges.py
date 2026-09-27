"""EXCHANGES AS FRAMES (Kaveh 2026-09-27: "exchange is the fundamental unit ... each exchange, each attack, or each
capture should define an instance of a frame"). A frame is not chosen by enumerating piece sets; it is read off what
happened in a game. An EXCHANGE is a maximal run of consecutive captures (a single capture is one too); its CONTESTED
squares are where the captures land; its FRAME is every piece that captured or was captured, and every piece that
attacked or defended a contested square on any board of the run -- the x-ray behind a slider joins when the piece in
front leaves. The frame's links are a STAR around the contested piece (each piece attacks or defends it), not every
pair: "the addition of the sixth piece ... just interacts with the square".

Pieces are named by their square at the start of the exchange; the position before the first capture is where the
frame stands. Games are split by identity (game id % 5 == 0 is held out).

    python -m catspace.incubator.analyst.exchanges            # label every game, print the statistics and examples
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import chess

from catspace.io import paths

from .chess_domain import VAL


@dataclass
class Exchange:
    game: int
    start_ply: int
    end_ply: int
    fen: str                                        # the position before the first capture
    squares: list                                   # the contested squares, in the order captured on
    captures: list                                  # [(ply, uci, captured symbol)]
    pieces: dict = field(default_factory=dict)      # start square -> {symbol, role, link, direct}
    won: dict = field(default_factory=dict)         # 'white' / 'black' -> material captured by that side

    @property
    def size(self) -> int:
        return len(self.pieces)

    @property
    def held_out(self) -> bool:
        return self.game % 5 == 0


def games(meta: list) -> dict:
    """game id -> [(ply, fen, played uci)] in ply order (the knowledge corpus: every ply of each game)."""
    g = defaultdict(list)
    for r in meta:
        g[r["game"]].append((r["ply"], r["fen"], r["played"]))
    return {k: sorted(v) for k, v in g.items()}


def victim_square(b: chess.Board, m: chess.Move) -> int:
    return chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)) if b.is_en_passant(m) else m.to_square


GAP = 3                                                          # a capture on the same square within this many plies continues the exchange (room for an intermezzo)


def moved(where: dict, b: chess.Board, m: chess.Move) -> dict:
    """The identity map (current square -> start square) after move m on board b: captures remove the victim, castling
    moves the rook too, a promotion keeps the pawn's identity."""
    w = dict(where); start = w.pop(m.from_square)
    w.pop(victim_square(b, m), None)
    w[m.to_square] = start
    if b.is_castling(m):
        rank = chess.square_rank(m.from_square); kside = chess.square_file(m.to_square) > chess.square_file(m.from_square)
        rf, rt = (chess.square(7, rank), chess.square(5, rank)) if kside else (chess.square(0, rank), chess.square(3, rank))
        if b.piece_at(m.to_square) is not None and b.piece_at(m.to_square).piece_type == chess.ROOK:   # chess960-style notation: the king 'captures' its rook
            rf = m.to_square
        if rf in w:
            w[rt] = w.pop(rf)
    return w


def label_chain(game: int, span: list, sq: int) -> Exchange:
    """span: [(ply, fen, uci)] from the first capture on `sq` to the last; the plies between that do not capture on sq
    are intermezzi (passes, from the frame's side). The frame: whoever captured or was captured on sq, and whoever
    attacked or defended sq on any board of the span (x-rays join when the piece in front leaves)."""
    b = chess.Board(span[0][1]); b0 = b.copy(stack=False)
    where = {x: x for x in b.piece_map()}
    boards = []; captures = []; won = {"white": 0.0, "black": 0.0}; roles = {}; intermezzi = []
    for ply, fen, uci in span:
        m = chess.Move.from_uci(uci)
        boards.append((b.copy(stack=False), where))
        if b.is_capture(m) and victim_square(b, m) == sq:
            victim = b.piece_at(sq); mover = b.piece_at(m.from_square)
            captures.append((ply, uci, victim.symbol()))
            won["white" if mover.color else "black"] += VAL[victim.piece_type]
            roles.setdefault(where[m.from_square], "capturer"); roles[where[sq]] = "captured"
        else:
            intermezzi.append((ply, uci))
        where = moved(where, b, m); b.push(m)
    boards.append((b.copy(stack=False), where))
    occ = b0.piece_at(sq)
    pieces = {sq: dict(direct=True)}
    for k, (bb, wh) in enumerate(boards):
        for color in (chess.WHITE, chess.BLACK):
            for a in bb.attackers(color, sq):
                st = wh.get(a)
                if st is not None and st not in pieces:
                    pieces[st] = dict(direct=(k == 0))
    for st, role in roles.items():
        pieces.setdefault(st, dict(direct=False))["role"] = role
    for st, d in pieces.items():
        pc = b0.piece_at(st)
        d["symbol"] = pc.symbol()
        d["link"] = "self" if st == sq else ("defends" if pc.color == occ.color else "attacks")
    e = Exchange(game, span[0][0], span[-1][0], span[0][1], [sq], captures, {chess.square_name(k): v for k, v in pieces.items()}, won)
    e.intermezzi = intermezzi
    return e


def exchanges(meta: list) -> list:
    """Per game, chains of captures on one square, each capture within GAP plies of the previous one there; captures
    on other squares in between are other exchanges (and intermezzi for this one)."""
    out = []
    for gid, plies in games(meta).items():
        open_: dict = {}                                         # square -> [indices of its captures]
        def close(sq):
            idx = open_.pop(sq)
            out.append(label_chain(gid, plies[idx[0]: idx[-1] + 1], sq))
        for i, (ply, fen, uci) in enumerate(plies):
            for sq in [s for s, idx in open_.items() if plies[idx[-1]][0] < ply - GAP]:
                close(sq)
            b = chess.Board(fen); m = chess.Move.from_uci(uci)
            if m not in b.legal_moves or not b.is_capture(m):
                continue
            vs = victim_square(b, m)
            open_.setdefault(vs, []).append(i)
        for sq in list(open_):
            close(sq)
    return out


def show(e: Exchange, log=print):
    b = chess.Board(e.fen)
    log(f"game {e.game} plies {e.start_ply}-{e.end_ply}: {' '.join(f'{u}(x{s})' for _, u, s in e.captures)}"
        + (f" with intermezzi {' '.join(u for _, u in e.intermezzi)}" if getattr(e, 'intermezzi', None) else "") + "; contested "
        f"{', '.join(chess.square_name(s) for s in e.squares)}; size {e.size}; won white {e.won['white']:.0f} black {e.won['black']:.0f}")
    log(b.unicode(borders=False, empty_square='·'))
    for sq, d in sorted(e.pieces.items(), key=lambda kv: (kv[1].get('link') != 'self', kv[0])):
        log(f"   {d['symbol']}{sq:3s} {d.get('link', ''):8s} {'direct' if d.get('direct') else 'x-ray/later':11s} {d.get('role', '')}")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args(argv)
    from .dataset import load
    meta = load(paths.derived("knowledge"), meta_only=True)["meta"]
    ex = exchanges(meta)
    out = paths.derived("exchanges.jsonl")
    with open(out, "w") as f:
        for e in ex:
            f.write(json.dumps(dict(asdict(e), size=e.size, intermezzi=getattr(e, "intermezzi", []))) + "\n")
    sizes = Counter(e.size for e in ex); runs = Counter(len(e.captures) for e in ex)
    big = [e for e in ex if len(e.captures) >= 2]
    print(f"[exchanges] {len(ex)} exchanges in {len(games(meta))} games ({sum(e.held_out for e in ex)} in held-out games) -> {out}")
    print(f"OBS captures per exchange: {dict(sorted(runs.items()))}")
    print(f"OBS frame size (pieces involved): {dict(sorted(sizes.items()))}")
    print(f"OBS exchanges of 2+ captures: {len(big)}; their size median {sorted(e.size for e in big)[len(big) // 2] if big else 0}")
    for e in sorted(big, key=lambda e: -e.size)[: a.examples] + big[: a.examples]:
        show(e)


if __name__ == "__main__":
    main()
