#!/usr/bin/env python
"""experiments/sf_wdl_by_material.py -- perfect-vs-human committor by MATERIAL, on the SAME positions.
For a material-stratified sample of real human positions, compute TWO committors:
  * FIELD committor  = P(win under HUMAN play)  (our trained field's ending head)
  * STOCKFISH WDL    = P(win under PERFECT play) (SF depth-D white-POV win fraction) -- the true value
and stack each as a committor ridgeline by material class. Kaveh's control: the perfect committor
should be BIMODAL at ~all material (the value is already decided at 30 pieces), while the human-play
committor is a unimodal jumble until ~20 pieces -- i.e. HUMAN ERROR is what keeps the midgame open.
"""
from __future__ import annotations

import argparse, glob, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKETS = [(27, 33), (23, 27), (19, 23), (15, 19), (11, 15), (8, 11), (5, 8), (3, 5)]


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
        b = chess.Board(fen)
        try:
            info = eng.analyse(b, chess.engine.Limit(depth=depth))
            wdl = info.get("wdl")
            if wdl is not None:
                w = wdl.white(); tot = w.wins + w.draws + w.losses
                out.append(w.wins / tot if tot else 0.5)                # white-POV P(win)
            else:
                cp = info["score"].white().score(mate_score=10000)
                out.append(1.0 / (1.0 + 10 ** (-cp / 400.0)))           # fallback: cp -> ~win prob
        except Exception:
            out.append(np.nan)
    eng.quit()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/records/lichess_2019-01")
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--per-bucket", type=int, default=700)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=8000)
    ap.add_argument("--out", default="artifacts/experiments/sf_wdl_by_material.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)

    # 1. material-stratified sample of positions from records -> (fen, planes, pieces)
    import chess, torch, pyarrow.parquet as pq
    from lczerolens import LczeroBoard
    files = sorted(glob.glob(str(Path(args.records) / "*.parquet")))[:4]
    caps = {b: args.per_bucket for b in BUCKETS}
    fens, planes, pieces = [], [], []
    def bucket(np_):
        for lo, hi in BUCKETS:
            if lo <= np_ < hi:
                return (lo, hi)
        return None
    games_used = 0
    for f in files:
        moves_col = pq.read_table(f, columns=["moves"]).column("moves").to_pylist()
        order = rng.permutation(len(moves_col))
        for r in order:
            if games_used >= args.max_games or all(v <= 0 for v in caps.values()):
                break
            games_used += 1
            ucis = moves_col[r].split()
            if len(ucis) < 12:
                continue
            b = LczeroBoard()
            for ply, u in enumerate(ucis):
                try:
                    b.push(chess.Move.from_uci(u))
                except Exception:
                    break
                if ply < 8 or ply % 4 != 0:
                    continue
                npc = chess.popcount(b.occupied); bk = bucket(npc)
                if bk is None or caps[bk] <= 0:
                    continue
                if b.is_game_over():
                    continue
                caps[bk] -= 1
                fens.append(b.fen()); pieces.append(npc)
                planes.append(b.to_input_tensor().to(dtype=torch.uint8).numpy())
        if all(v <= 0 for v in caps.values()):
            break
    pieces = np.array(pieces); planes = np.stack(planes)
    print(f"[sf-wdl] {len(fens)} positions from {games_used} games | per-bucket target {args.per_bucket}", flush=True)

    # 2. FIELD committor (P win under human play)
    from experiments.train_clock_field import ClockField
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    import torch.nn.functional as F
    dev = resolve_device("auto")
    p = torch.load(args.ckpt, map_location=dev, weights_only=False); cfg = p["cfg"]
    net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
    net.load_state_dict(p["state_dict"]); net.eval()
    field_c = []
    for i in range(0, len(planes), 4096):
        x = torch.from_numpy(planes[i:i+4096].astype(np.float32)).to(dev)
        with torch.no_grad():
            field_c.append(F.softmax(net.d_mate_and_end(x)[1], 1)[:, 0].cpu().numpy())
    field_c = np.concatenate(field_c)

    # 3. STOCKFISH WDL (P win under perfect play), parallel
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    chunks = np.array_split(np.array(fens, dtype=object), W)
    print(f"  Stockfish depth {args.depth} x {W} workers on {len(fens)} positions...", flush=True)
    sf_c = np.empty(len(fens))
    with ProcessPoolExecutor(max_workers=W) as ex:
        res = list(ex.map(sf_worker, [(list(c), args.depth) for c in chunks]))
    k = 0
    for c in res:
        sf_c[k:k+len(c)] = c; k += len(c)
    ok = ~np.isnan(sf_c)
    print(f"  SF done, {ok.sum()} valid [{time.time()-t0:.0f}s]", flush=True)

    # 4. ridgelines side by side
    from scipy.stats import gaussian_kde, skew, kurtosis
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    def bimod(c):
        return (skew(c) ** 2 + 1) / (kurtosis(c, fisher=True) + 3) if len(c) > 8 else np.nan
    for ax, cvals, name, cmapn in [(axes[0], field_c, "HUMAN-play committor (field)", "viridis"),
                                    (axes[1], sf_c, f"PERFECT committor (Stockfish d{args.depth} WDL)", "plasma")]:
        print(f"\n{name}: bimodality by bucket")
        for k, (lo, hi) in enumerate(BUCKETS):
            m = (pieces >= lo) & (pieces < hi) & ok if name.startswith("PERFECT") else (pieces >= lo) & (pieces < hi)
            y0 = len(BUCKETS) - 1 - k
            c = np.clip(cvals[m], 1e-3, 1 - 1e-3)
            if len(c) < 20:
                continue
            dens = gaussian_kde(c)(xs); dens = dens / dens.max() * 0.9
            ax.fill_between(xs, y0, y0 + dens, alpha=0.7, color=plt.get_cmap(cmapn)(k / len(BUCKETS)))
            ax.plot(xs, y0 + dens, color="k", lw=0.6, alpha=0.5)
            ax.text(-0.02, y0 + 0.15, f"{lo}-{hi-1}p", ha="right", va="bottom", fontsize=9)
            print(f"  {lo}-{hi-1}p: bimodality {bimod(c):.3f} (n={len(c)})")
        ax.set_yticks([]); ax.set_xlabel("committor c = P(win)"); ax.set_title(name); ax.set_xlim(-0.1, 1.02)
    fig.suptitle("Same human positions, two committors: HUMAN-play vs PERFECT (Stockfish) -- stacked by material", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nVERDICT -> {args.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
