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


def _start(rng: random.Random):
    import chess
    while True:
        sq = rng.sample(range(64), 3)
        b = chess.Board(None)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.QUEEN, chess.WHITE))
        b.set_piece_at(sq[2], chess.Piece(chess.KING, chess.BLACK))
        b.turn = chess.WHITE
        if b.is_valid() and not b.is_game_over() and chess.square_distance(sq[0], sq[2]) > 1:
            return b


def _game(seed: int):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import chess
    import chess.engine
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = _start(random.Random(seed))
    start = b.fen()
    st = analyst_core.PlanState()
    moves = 0
    none = 0
    ms = 0.0
    sure = 0                                          # moves whose measure falls against every reply (the certificate)
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
        if moves >= 60:
            out = "not finished"; break
        if b.turn == chess.WHITE:
            tr, dec, _, _, t, _, _ = st.observe(b.fen())
            ms += t
            cert = [l for l in tr if l.startswith("king and queen") and "against every reply:" in l]
            sure += bool(cert) and not any(x in cert[0] for x in ("no progress", "no component moves"))
            mv = None if not dec else (dec[1] if dec[0] in ("", "defend", "cheapest") else dec[0])
            if mv is None:
                none += 1; mv = next(iter(b.legal_moves)).uci()
            b.push_uci(mv); moves += 1
        else:
            st.record(b.fen())
            b.push(eng.play(b, chess.engine.Limit(depth=12)).move)
    eng.quit()
    return dict(seed=seed, start=start, out=out, moves=moves, none=none, ms=ms / max(moves, 1), sure=sure)


def run(games: int, workers: int) -> None:
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers) as p:
        res = list(p.imap_unordered(_game, range(games), chunksize=1))
    by = Counter(r["out"] for r in res)
    mates = [r["moves"] for r in res if r["out"] == "mate"]
    print(f"OBS kq_playout: {games} games ({time.time() - t0:.0f} s on {workers} processes): {dict(by)}; moves to mate "
          f"mean {sum(mates) / max(len(mates), 1):.1f}, max {max(mates, default=0)}; no decision {sum(r['none'] for r in res)} times; "
          f"{sum(r['ms'] for r in res) / len(res):.1f} ms a move; progress guaranteed against every reply on "
          f"{100 * sum(r['sure'] for r in res) / max(sum(r['moves'] for r in res), 1):.0f}% of our moves")
    for r in res:
        if r["out"] != "mate":
            print(f"OBS   {r['out']} after {r['moves']} moves from {r['start']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    run(a.games, a.workers)


if __name__ == "__main__":
    main()
