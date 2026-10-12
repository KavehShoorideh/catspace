"""THE KING-AND-QUEEN MATE, MEASURED (2026-10-12): the planner (a PlanState: the game context -- repetitions, the clock)
plays king and queen against Stockfish's lone king from random legal starts; every move of the defender is recorded in
the game's history. Per game: mate (and in how many moves), stalemate, a claimable draw (threefold, fifty moves), or
not finished. A MEASUREMENT of the tactic, not a proof (the proved part: mate only on the edge, Lean kq_mate_on_edge).

  python -m catspace.incubator.analyst.kq_playout --games 40
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import Counter


def _start(rng: random.Random, piece: str = "Q"):
    import chess
    kind = {"Q": chess.QUEEN, "R": chess.ROOK}[piece]
    while True:
        sq = rng.sample(range(64), 3)
        b = chess.Board(None)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(kind, chess.WHITE))
        b.set_piece_at(sq[2], chess.Piece(chess.KING, chess.BLACK))
        b.turn = chess.WHITE
        if b.is_valid() and not b.is_game_over() and chess.square_distance(sq[0], sq[2]) > 1:
            return b


def _game(args):
    seed, flags, piece = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import chess
    import chess.engine
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core
    for kv in filter(None, flags.split(",")):                  # any switch: name=0|1 (set_<name>)
        k, v = kv.split("=")
        getattr(analyst_core, f"set_{k}")(v == "1")
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = _start(random.Random(seed), piece)
    start = b.fen()
    st = analyst_core.PlanState()
    moves = 0
    none = 0
    ms = 0.0
    sure = 0                                          # moves whose measure falls against every reply (the certificate)
    tempo = 0                                         # of which within two moves (a tempo move)
    unc = []                                          # the uncertified moves
    while True:
        if b.is_checkmate():
            out = "mate"; break
        if b.is_stalemate():
            out = "stalemate"; break
        # (the DEFENDER claims: at its turn any claim it can make; at ours, a position already on the board a third time --
        # python-chess's can_claim_threefold_repetition at our turn counts OUR possible claims, which we never make)
        if (b.turn == chess.BLACK and b.can_claim_threefold_repetition()) or b.is_repetition(3):
            out = "threefold"; break
        if b.halfmove_clock >= 100:
            out = "fifty moves"; break
        if moves >= 80:
            out = "not finished"; break
        if b.turn == chess.WHITE:
            tr, dec, _, _, t, _, _ = st.observe(b.fen())
            ms += t
            cert = [l for l in tr if l.startswith(("king and queen", "king and rook")) and "against every reply:" in l]
            # (a mate by checks from the planner's pre-pass is a proof in itself: every reply searched)
            proven_mate = any(l.startswith("mate:") and "proven" in l for l in tr)
            ok = proven_mate or (bool(cert) and not any(x in cert[0] for x in ("no progress", "no component moves")) or (bool(cert) and "a tempo move" in cert[0]))
            sure += ok
            if not ok:
                unc.append((b.fen(), dec[1] if dec else None, cert[0].split("against every reply: ")[-1] if cert else "-"))
            tempo += bool(cert) and "a tempo move" in cert[0]
            mv = None if not dec else (dec[1] if dec[0] in ("", "defend", "cheapest") else dec[0])
            if mv is None:
                none += 1; mv = next(iter(b.legal_moves)).uci()
            b.push_uci(mv); moves += 1
        else:
            st.record(b.fen())
            b.push(eng.play(b, chess.engine.Limit(depth=12)).move)
    eng.quit()
    return dict(seed=seed, start=start, out=out, moves=moves, none=none, ms=ms / max(moves, 1), sure=sure, tempo=tempo, unc=unc)


def run(games: int, workers: int, flags: str = "", piece: str = "Q") -> None:
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers) as p:
        res = list(p.imap_unordered(_game, [(g, flags, piece) for g in range(games)], chunksize=1))
    by = Counter(r["out"] for r in res)
    mates = [r["moves"] for r in res if r["out"] == "mate"]
    print(f"OBS kq_playout (K+{piece} v K): {games} games ({time.time() - t0:.0f} s on {workers} processes): {dict(by)}; moves to mate "
          f"mean {sum(mates) / max(len(mates), 1):.1f}, max {max(mates, default=0)}; no decision {sum(r['none'] for r in res)} times; "
          f"{sum(r['ms'] for r in res) / len(res):.1f} ms a move; progress guaranteed against every reply on "
          f"{100 * sum(r['sure'] for r in res) / max(sum(r['moves'] for r in res), 1):.1f}% of our moves "
          f"({sum(r["tempo"] for r in res)} of them tempo moves, progress within two or three); uncertified "
          f"{sum(r['moves'] for r in res) - sum(r['sure'] for r in res)}")
    for r in res:
        for f, m, w in r.get("unc", []):
            print(f"OBS   uncertified: {m} in {f} -- {w}")
    for r in res:
        if r["out"] != "mate":
            print(f"OBS   {r['out']} after {r['moves']} moves from {r['start']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--flags", default="", help="name=0|1,... (set_<name>), e.g. kq_every_reply=1")
    ap.add_argument("--piece", default="Q", help="Q or R")
    a = ap.parse_args(argv)
    run(a.games, a.workers, a.flags, a.piece)


if __name__ == "__main__":
    main()
