"""PLAYING OUT A PROVEN MATE, checked (Kaveh 2026-10-10: once a mate is proven, play it out from the proof, no search).
Census positions where the planner proves a mate through its king target (not the mate-by-checks pre-pass): from
each, our planned move, Stockfish's best defence, our next move from the game's PlanState -- repeated to the end.
Counts: mates delivered, moves played from the stored solution with zero proofs, solutions dropped, failures."""
import os
import random
from multiprocessing import Pool

import chess
import chess.engine


def one(fen):
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    tr = analyst_core.plan_trace(fen)[0]
    if not any("is a proven mate" in l for l in tr):
        return None
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    st = analyst_core.PlanState()
    b = chess.Board(fen)
    played_out = searched = 0
    log = []
    for _ in range(12):
        t, dec, proofs, nodes, ms, imm, _ = st.observe(b.fen())
        if not dec:
            eng.quit(); return dict(fen=fen, result="no decision", log=log)
        mv = chess.Move.from_uci(dec[1])
        if "played out" in dec[3]:
            played_out += 1
            assert proofs == 0
        else:
            searched += 1
        log.append(b.san(mv)); b.push(mv)
        if not b.is_game_over():
            st.observe(b.fen())                                   # their turn, as in a game (after our move)
        if b.is_checkmate():
            eng.quit(); return dict(fen=fen, result="mate", played_out=played_out, searched=searched, log=log)
        if b.is_game_over():
            eng.quit(); return dict(fen=fen, result="game over without mate", log=log)
        reply = eng.play(b, chess.engine.Limit(depth=14)).move
        log.append(b.san(reply)); b.push(reply)
    eng.quit()
    return dict(fen=fen, result="no mate in 12", log=log)


if __name__ == "__main__":
    from catspace.incubator.analyst.tactic_census_measure import load
    rows = random.Random(4).sample(load("c300"), 6000)
    with Pool(os.cpu_count()) as p:
        rs = [r for r in p.map(one, [r["fen"] for r in rows], chunksize=16) if r]
    from collections import Counter
    print("OBS positions with a king-target mate proof:", len(rs), dict(Counter(r["result"] for r in rs)))
    m = [r for r in rs if r["result"] == "mate"]
    print(f"OBS in the mates: moves played from the stored solution {sum(r['played_out'] for r in m)}, moves searched {sum(r['searched'] for r in m)}")
    for r in rs[:6]:
        print("  ", r["result"], r["fen"], " ".join(r["log"]))
