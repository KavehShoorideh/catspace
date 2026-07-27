#!/usr/bin/env python
"""experiments/blunder_model.py -- LAYER 1 (stopgap, open-source): predict WHERE an opponent of a
given strength is most likely to BLUNDER (cross a basin boundary against themselves). Kaveh: use an
open-source blunder calculator for now -> Maia (rating-conditioned human move model) + Stockfish
(near-perfect value oracle to detect the basin crossing).

For position s with the opponent (rating r) to move:
  c(s)      = Stockfish white-POV P(win)  (the perfect committor)
  P(m|s,r)  = Maia-r policy over the opponent's legal moves
  blunder magnitude of move m (mover-POV) = how much m moves c AGAINST the mover
  B(s,r)    = sum_m P(m|s,r) * max(0, mover-POV committor loss of m)   [expected self-blunder]
B(s,r) IS the transition predictor T(s, z=rating): the opponent's error map. Validated here against
the ACTUAL move played in real games at that rating -> does high B predict real blunders?
"""
from __future__ import annotations

import argparse, glob, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MAIA = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]


def nearest_maia(elo):
    return min(MAIA, key=lambda r: abs(r - elo))


class MaiaEngine:
    """Persistent lc0+maia subprocess giving the full per-move POLICY (VerboseMoveStats)."""
    def __init__(self, rating):
        self.p = subprocess.Popen(["lc0", f"--weights=data/engines/maia/maia-{rating}.pb.gz", "--backend=eigen"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        self._send("uci"); self._send("setoption name VerboseMoveStats value true"); self._send("isready")
        while self.p.stdout.readline().strip() != "readyok":
            pass

    def _send(self, c):
        self.p.stdin.write(c + "\n"); self.p.stdin.flush()

    def policy(self, fen):
        self._send(f"position fen {fen}"); self._send("go nodes 1")
        pol = {}
        while True:
            line = self.p.stdout.readline()
            if not line or line.startswith("bestmove"):
                break
            if "(P:" in line:
                m = re.search(r"([a-h][1-8][a-h][1-8][qrbn]?)\b", line); pp = re.search(r"P:\s*([\d.]+)%", line)
                if m and pp:
                    pol[m.group(1)] = float(pp.group(1)) / 100.0
        return pol

    def close(self):
        try:
            self._send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main():
    import chess, chess.engine
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/records/lichess_2019-01")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--topk", type=int, default=6); ap.add_argument("--blunder-thresh", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    import pyarrow.parquet as pq
    from lczerolens import LczeroBoard  # noqa (kept for parity; not needed here)

    sf = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    try:
        sf.configure({"UCI_ShowWDL": True})
    except Exception:
        pass

    def committor(board):                                   # white-POV P(win)
        info = sf.analyse(board, chess.engine.Limit(depth=args.depth))
        w = info["wdl"].white(); tot = max(1, w.wins + w.draws + w.losses)
        return w.wins / tot

    # sample human positions: (board, mover elo, actual next move)
    files = sorted(glob.glob(str(Path(args.records) / "*.parquet")))[:3]
    samples = []
    for f in files:
        t = pq.read_table(f, columns=["moves", "white_elo", "black_elo"]).to_pydict()
        for r in rng.permutation(len(t["moves"])):
            if len(samples) >= args.n:
                break
            ucis = t["moves"][r].split()
            if len(ucis) < 20:
                continue
            ply = int(rng.integers(10, min(len(ucis) - 1, 70)))
            b = chess.Board(); ok = True
            for u in ucis[:ply]:
                try:
                    b.push(chess.Move.from_uci(u))
                except Exception:
                    ok = False; break
            if not ok or b.is_game_over():
                continue
            mover_elo = t["white_elo"][r] if b.turn == chess.WHITE else t["black_elo"][r]
            try:
                actual = chess.Move.from_uci(ucis[ply])
                if actual not in b.legal_moves:
                    continue
            except Exception:
                continue
            samples.append((b, int(mover_elo), actual))
        if len(samples) >= args.n:
            break
    print(f"[blunder-model] {len(samples)} positions | SF depth {args.depth} | Maia top-{args.topk}", flush=True)

    maia_cache = {}
    rows = []
    for i, (b, elo, actual) in enumerate(samples):
        r = nearest_maia(elo)
        if r not in maia_cache:
            maia_cache[r] = MaiaEngine(r)
        pol = maia_cache[r].policy(b.fen())
        if not pol:
            continue
        c0 = committor(b)
        white_to_move = (b.turn == chess.WHITE)
        # mover-POV loss of a move = c moves against mover. white mover: loss = c0 - c(after); black: c(after) - c0
        def mover_loss(mv):
            b.push(mv); c1 = committor(b); b.pop()
            return (c0 - c1) if white_to_move else (c1 - c0)
        top = sorted(pol.items(), key=lambda kv: -kv[1])[:args.topk]
        B = 0.0
        for uci, pr in top:
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                continue
            if mv in b.legal_moves:
                B += pr * max(0.0, mover_loss(mv))
        act_loss = max(0.0, mover_loss(actual))
        rows.append((r, elo, c0, B, act_loss, act_loss >= args.blunder_thresh))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)}", flush=True)
    for e in maia_cache.values():
        e.close()
    sf.quit()

    R = np.array([(x[3], x[4], x[5], x[0]) for x in rows])   # B, act_loss, is_blunder, rating
    B = R[:, 0]; act = R[:, 1]; isbl = R[:, 2].astype(bool); rat = R[:, 3]
    from scipy.stats import spearmanr
    print(f"\nVALIDATION (n={len(R)}):")
    print(f"  predicted B(s,r) vs ACTUAL self-committor-loss: Spearman {spearmanr(B, act).correlation:+.3f}")
    # calibration: actual blunder rate by predicted-B quartile
    q = np.quantile(B, [0, .25, .5, .75, 1.0])
    print("  actual blunder-rate by predicted-B quartile (does high predicted B -> more real blunders?):")
    for k in range(4):
        m = (B >= q[k]) & (B <= q[k+1]) if k == 3 else (B >= q[k]) & (B < q[k+1])
        if m.sum():
            print(f"    B in [{q[k]:.3f},{q[k+1]:.3f}]: predicted meanB {B[m].mean():.3f} | ACTUAL blunder-rate {isbl[m].mean():.1%} (n={int(m.sum())})")
    print("  mean predicted B by rating band (weaker -> higher blunder prob?):")
    for lo, hi in [(1000, 1400), (1400, 1700), (1700, 2100)]:
        m = (rat >= lo) & (rat < hi)
        if m.sum():
            print(f"    {lo}-{hi}: mean B {B[m].mean():.3f} | actual blunder-rate {isbl[m].mean():.1%} (n={int(m.sum())})")
    print("DONE blunder_model", flush=True)


if __name__ == "__main__":
    main()
