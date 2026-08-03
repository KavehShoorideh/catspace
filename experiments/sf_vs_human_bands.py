#!/usr/bin/env python
"""experiments/sf_vs_human_bands.py -- the 3-basin BANDS + basin transition MATRIX, side by side for
LICHESS (human-play committor = field WDL) vs PERFECT (Stockfish WDL), on the SAME positions. Kaveh's
control: under perfect play the transition matrix should be near-DIAGONAL (infinite barriers, basins
don't leak) while human play has big off-diagonals (win<->loss ~0.25). Same positions -> the whole
difference is the play measure.
"""
from __future__ import annotations

import argparse, glob, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BASINS = ["Win", "Draw", "Loss"]; BCOL = {0: "#3b6fb0", 1: "#7a7a7a", 2: "#c04040"}


def sf_worker(task):
    fens, depth = task
    import chess, chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass
    out = []
    for fen in fens:
        try:
            info = eng.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))
            w = info["wdl"].white(); tot = max(1, w.wins + w.draws + w.losses)
            out.append((w.wins / tot, w.draws / tot, w.losses / tot))
        except Exception:
            out.append((np.nan, np.nan, np.nan))
    eng.quit(); return out


def bands_matrix(dist):
    dist = dist / dist.sum(1, keepdims=True)
    basin = dist.argmax(1); leak = 1 - dist[np.arange(len(dist)), basin]
    M = np.zeros((3, 3))
    for i in range(3):
        m = basin == i
        if m.any():
            M[i] = dist[m].mean(0)
        M[i, i] = 0.0
    return basin, leak, M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/records/lichess_2019-01")
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/sf_vs_human_bands.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)

    import chess, torch, pyarrow.parquet as pq
    from lczerolens import LczeroBoard
    files = sorted(glob.glob(str(Path(args.records) / "*.parquet")))[:3]
    fens, planes = [], []
    for f in files:
        mc = pq.read_table(f, columns=["moves"]).column("moves").to_pylist()
        for r in rng.permutation(len(mc)):
            if len(fens) >= args.n:
                break
            ucis = mc[r].split()
            if len(ucis) < 16:
                continue
            b = LczeroBoard(); take_ply = int(rng.integers(8, min(len(ucis), 80)))
            for ply, u in enumerate(ucis[:take_ply + 1]):
                try:
                    b.push(chess.Move.from_uci(u))
                except Exception:
                    b = None; break
            if b is None or b.is_game_over():
                continue
            fens.append(b.fen()); planes.append(b.to_input_tensor().to(dtype=torch.uint8).numpy())
        if len(fens) >= args.n:
            break
    planes = np.stack(planes)
    print(f"[sf-vs-human-bands] {len(fens)} positions", flush=True)

    from experiments.train_clock_field import ClockField
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    import torch.nn.functional as F
    dev = resolve_device("auto")
    p = torch.load(args.ckpt, map_location=dev, weights_only=False); cfg = p["cfg"]
    net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
    net.load_state_dict(p["state_dict"]); net.eval()
    fd = []
    for i in range(0, len(planes), 4096):
        x = torch.from_numpy(planes[i:i+4096].astype(np.float32)).to(dev)
        with torch.no_grad():
            pe = F.softmax(net.d_mate_and_end(x)[1], 1).cpu().numpy()
        fd.append(np.stack([pe[:, 0], pe[:, 1:5].sum(1), pe[:, 5]], 1))
    field_dist = np.concatenate(fd)

    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"  Stockfish d{args.depth} x {W} workers...", flush=True)
    chunks = np.array_split(np.array(fens, dtype=object), W)
    sf = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for c in ex.map(sf_worker, [(list(ch), args.depth) for ch in chunks]):
            sf.extend(c)
    sf_dist = np.array(sf); ok = ~np.isnan(sf_dist).any(1)
    field_dist = field_dist[ok]; sf_dist = sf_dist[ok]
    print(f"  SF done, {ok.sum()} valid [{time.time()-t0:.0f}s]", flush=True)

    hb, hl, hM = bands_matrix(field_dist)
    sb, sl, sM = bands_matrix(sf_dist)
    for name, M in [("LICHESS (human)", hM), ("PERFECT (Stockfish)", sM)]:
        print(f"\n{name} basin->basin transition matrix:")
        for i in range(3):
            print(f"  {BASINS[i]:5} " + "  ".join(f"{M[i,j]:.3f}" for j in range(3)))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 11), height_ratios=[2.3, 1])
    for col, (basin, leak, M, name) in enumerate([(hb, hl, hM, "LICHESS 1400-1800 (human play)"),
                                                  (sb, sl, sM, f"PERFECT (Stockfish d{args.depth})")]):
        a = ax[0, col]
        for b in range(3):
            m = basin == b
            x = b + (rng.random(m.sum()) - 0.5) * 0.8
            sc = a.scatter(x, leak[m], c=leak[m], cmap="viridis", s=6, alpha=0.55, vmin=0, vmax=0.66)
        a.set_xticks([0, 1, 2]); a.set_xticklabels(BASINS); a.set_ylim(-0.02, 1.0)
        a.set_ylabel("transition prob (leak = 1 - p_own)"); a.set_title(name)
        am = ax[1, col]
        im = am.imshow(M, cmap="magma", vmin=0, vmax=0.5)
        am.set_xticks(range(3)); am.set_xticklabels(BASINS); am.set_yticks(range(3)); am.set_yticklabels(BASINS)
        am.set_xlabel("to basin"); am.set_ylabel("from basin")
        for i in range(3):
            for j in range(3):
                am.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.3 else "black", fontsize=11)
        am.set_title(f"{name.split('(')[0]}transition matrix")
    fig.colorbar(sc, ax=ax[0, 1], label="leak", fraction=0.04)
    fig.colorbar(im, ax=ax[1, 1], fraction=0.04)
    fig.suptitle("3 basins + transition matrix: HUMAN vs PERFECT play (same positions)", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nVERDICT -> {args.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
