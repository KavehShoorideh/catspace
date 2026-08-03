#!/usr/bin/env python
"""experiments/controlfield_gates.py -- Phase 2.3 validation gates for the control
field / ascent cone (docs/CONTROL-FIELD-SPEC.md). Per the spec: do not proceed to
Phase 3 until these pass; report honestly, do not tune weights to force gate 3.

Gate 1 (sanity): cone_size should correlate positively with a shallow SF eval
  favoring the mover, on real game positions. Spearman >= 0.15.
Gate 2 (known tactics): on Lichess puzzles tagged mateIn2/kingsideAttack, the
  puzzle's solution move should lie in K(s) (king_zone targets) >= 60% of the time.
Gate 3 (gambit case study): on hand-picked accepted-vs-declined gambit lines,
  cone_size/best_gain should be higher for the sacrificing side in the accepted
  line than the declined line, immediately after the sacrifice.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.controlfield.derivative import ascent_cone, ConeConfig   # noqa: E402


def gate1_sanity(n, engine, depth, seed=0):
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    t = pq.read_table("data/records/lichess_2019-01/records_00000.parquet").to_pylist()
    rng.shuffle(t)
    cone_sizes, evals = [], []
    tried = 0
    for g in t:
        if len(cone_sizes) >= n:
            break
        moves = g["moves"].split()
        if len(moves) < 20:
            continue
        b = chess.Board()
        ply = int(rng.integers(10, len(moves) - 2))
        try:
            for mv in moves[:ply]:
                b.push(chess.Move.from_uci(mv))
        except Exception:
            continue
        if b.is_game_over():
            continue
        tried += 1
        out = ascent_cone(b)
        info = engine.analyse(b, chess.engine.Limit(depth=depth))
        score = info["score"].pov(b.turn).score(mate_score=3200)
        if score is None:
            continue
        cone_sizes.append(out["cone_size"])
        evals.append(score)
    from scipy.stats import spearmanr
    rho, p = spearmanr(cone_sizes, evals)
    return dict(n=len(cone_sizes), tried=tried, rho=float(rho), p=float(p))


def gate2_puzzles(n, theme_filter, seed=0):
    import pandas as pd
    import zstandard as zstd
    with open("data/lichess_db_puzzle.csv.zst", "rb") as f:
        data = zstd.ZstdDecompressor().stream_reader(f).read()
    df = pd.read_csv(io.BytesIO(data))
    mask = df["Themes"].fillna("").apply(
        lambda s: any(t in s.split() for t in theme_filter))
    pool = df[mask]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), min(n, len(pool)), replace=False)
    rows = pool.iloc[idx]
    hits, tried, skipped = 0, 0, 0
    for _, row in rows.iterrows():
        try:
            b = chess.Board(row["FEN"])
            moves = row["Moves"].split()
            if len(moves) < 2:
                skipped += 1
                continue
            b.push(chess.Move.from_uci(moves[0]))       # setup move (opponent's blunder)
            solution = chess.Move.from_uci(moves[1])      # first solving move
            if solution not in b.legal_moves or b.is_game_over():
                skipped += 1
                continue
            out = ascent_cone(b, cone_cfg=ConeConfig(target_mode="king_zone"))
            tried += 1
            if solution in out["moves"]:
                sol_idx = out["moves"].index(solution)
                if out["in_cone"][sol_idx]:
                    hits += 1
        except Exception:
            skipped += 1
            continue
    rate = hits / tried if tried else float("nan")
    return dict(tried=tried, skipped=skipped, hits=hits, rate=rate)


# Hand-picked gambit lines: (name, accepted_uci_moves, declined_uci_moves).
# Both lines share the same opening moves up to the sacrifice point; "accepted"
# has the defender take the offered pawn, "declined" has them decline it with a
# standard alternative. Sacrificer = the side that offered the gambit (White in
# all three below). cone_size/best_gain compared for White immediately after.
GAMBITS = [
    ("Danish Gambit",
     "e2e4 e7e5 d2d4 e5d4 c2c3 d4c3 f1c4".split(),          # accepted: ...dxc3, Bc4 (down 2 pawns, full compensation)
     "e2e4 e7e5 d2d4 e5d4 c2c3 d7d5".split()),                # declined: 3...d5 (returns the pawn immediately)
    ("Evans Gambit",
     "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4 c5b4".split(),      # accepted: 4...Bxb4
     "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4 c5b6".split()),      # declined: 4...Bb6
    ("Benko Gambit",
     "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5 c4b5 a7a6".split(),      # accepted: 4.cxb5 a6
     "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5 g1f3 g7g6".split()),      # declined: 4.Nf3 (ignores b5)
]


def gate3_gambits(weights=None):
    rows = []
    for name, acc, dec in GAMBITS:
        b_acc = chess.Board()
        for mv in acc:
            b_acc.push(chess.Move.from_uci(mv))
        b_dec = chess.Board()
        for mv in dec:
            b_dec.push(chess.Move.from_uci(mv))
        # cone/gain computed for the SACRIFICER (White) -- need White to move in
        # both; if the last move made it Black's turn, step back conceptually by
        # evaluating from White's last position isn't right either -- spec says
        # "immediately after the sacrifice", i.e. the position as given (whoever
        # is to move), scored from White's (the sacrificer's) perspective. Since
        # ascent_cone always scores the side TO MOVE, and both lines above end
        # with Black to move (even move count), cone_size/best_gain as computed
        # are already White's opponent's cone -- to get White's own cone/gain we
        # evaluate one ply later is wrong (changes the position); instead compute
        # the cone as SEEN correctly by asking for the position as-is: the mover
        # POV values ARE for whoever is to move, so if it's Black to move, invert
        # interpretation is not straightforward for gain (target squares differ
        # by mover). Report both sides' cone/gain rather than assume; the White-
        # perspective quantity we actually want is best obtained by comparing the
        # control field C_a itself (mover-POV-independent White-raw), not cone_size.
        from catspace.controlfield.control import weighted_attacker_field
        c_acc = weighted_attacker_field(b_acc, weights)   # White POV raw
        c_dec = weighted_attacker_field(b_dec, weights)
        out_acc = ascent_cone(b_acc, weights=weights)
        out_dec = ascent_cone(b_dec, weights=weights)
        rows.append(dict(
            name=name,
            white_control_sum_accepted=float(c_acc.sum()),
            white_control_sum_declined=float(c_dec.sum()),
            mover_cone_size_accepted=out_acc["cone_size"],
            mover_cone_size_declined=out_dec["cone_size"],
            mover_best_gain_accepted=out_acc["best_gain"],
            mover_best_gain_declined=out_dec["best_gain"],
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1-n", type=int, default=2000)
    ap.add_argument("--gate1-depth", type=int, default=6)
    ap.add_argument("--gate2-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    import shutil
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")

    print("=== Gate 1: sanity (cone_size vs shallow SF eval) ===", flush=True)
    g1 = gate1_sanity(args.gate1_n, eng, args.gate1_depth, args.seed)
    print(f"VERDICT gate1-sanity: n={g1['n']} (tried {g1['tried']}) | Spearman rho "
          f"{g1['rho']:+.3f} (p={g1['p']:.2e}) | gate >=0.15 | "
          f"{'PASS' if g1['rho'] >= 0.15 else 'FAIL'}", flush=True)

    print("=== Gate 2: known tactics (puzzle solution in K(s)) ===", flush=True)
    g2 = gate2_puzzles(args.gate2_n, {"mateIn2", "kingsideAttack"}, args.seed)
    print(f"VERDICT gate2-tactics: tried={g2['tried']} skipped={g2['skipped']} "
          f"hits={g2['hits']} rate={g2['rate']:.1%} | gate >=60% | "
          f"{'PASS' if g2['rate'] >= 0.60 else 'FAIL'}", flush=True)

    print("=== Gate 3: gambit case study (report honestly, no tuning) ===", flush=True)
    g3 = gate3_gambits()
    n_pass = 0
    for r in g3:
        acc_higher_control = r["white_control_sum_accepted"] > r["white_control_sum_declined"]
        print(f"  {r['name']}: White raw control sum accepted={r['white_control_sum_accepted']:+.2f} "
              f"declined={r['white_control_sum_declined']:+.2f} | "
              f"mover cone_size accepted(mover={'?'})={r['mover_cone_size_accepted']:.3f} "
              f"declined={r['mover_cone_size_declined']:.3f} | "
              f"{'accepted-higher-control' if acc_higher_control else 'declined-higher-or-equal'}")
        n_pass += int(acc_higher_control)
    print(f"VERDICT gate3-gambits: {n_pass}/{len(g3)} gambits show higher White "
          f"raw control-sum in the accepted line vs declined | "
          f"{'REPORTED (no pass/fail threshold given by spec)' }", flush=True)

    eng.quit()
    print(f"DONE controlfield_gates [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
