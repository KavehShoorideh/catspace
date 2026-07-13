#!/usr/bin/env python
"""
experiments/qm_fitness_probe.py — structural fitness diagnostics for the
learned (quasi)metric, per the 2026-07-12 literature survey (JOURNAL.md):
what the quasimetric-RL literature reports when arguing a learned distance
is structurally sound, adapted to chess -- where, uniquely, Syzygy
tablebases give EXACT ground-truth distances for a subspace of positions.

Five instruments (build-first ranking from the survey):

  syzygy    calibration of d(F(s), zMATE) against tablebase DTZ on
            tablebase-covered winning positions: Spearman rho, per-DTZ-bin
            mean predicted distance, and (with the ply-gap-calibrated
            scale, d*ply_gap_scale ~ plies) stratified absolute error.
            [gridworld ground-truth protocol of PQE/IQE/QRL, with real
            ground truth instead of a toy]
  retrieval horizon-stratified retrieval accuracy on held-out games: is the
            true k-plies-later position scored closer than in-batch
            negatives, for k in {1,2,5,10,20,50}? [InfoNCE critic accuracy,
            stratified by horizon -- shows WHERE along the horizon the
            embedding stops discriminating]
  asymmetry capture-boundary audit: for real game pairs (s earlier, g
            later, material strictly decreased in between), compare
            forward score(F(s),B(g)) vs reverse score(F(g),B(s)) -- chess
            cannot un-capture, so reverse should read strictly farther.
            Also the infinity column: pairs where the "goal" has MORE
            pieces than the state (truly unreachable) must score far.
            [PQE one-way-door, quantified; IQE's infinite-distance column]
  triangle  PQE Defn 4.2 violation ratio over sampled triples --
            architectural sanity on d alone (must be exactly quasimetric),
            plus the informative version on the full score r-d, which is
            NOT guaranteed and tells us how non-metric the readout is.
  spread    degeneracy panel: spread ratio (mean d over random cross-game
            pairs / mean d over 1-ply pairs; ~1 = collapse), effective rank
            (entropy of singular values) of F and B, embedding norms.

Non-quasimetric checkpoints run everything except the d-only triangle
check (score IS the dot product; distance-like = -score still supports
the comparative instruments).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.io.paths import derived_dir
from catspace.nn.features import feature_planes, omega_ids

HOLDOUT_MOD = 50
HORIZONS = (1, 2, 5, 10, 20, 50)


# --------------------------------------------------------------- embedding
def embed_boards(fb, boards: list, device, elo: int = 1800, clock: float = 300.0):
    import torch
    packed = np.stack([encode_packed(b) for b in boards])
    meta = np.stack([encode_meta(b) for b in boards])
    planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
    om = torch.from_numpy(omega_ids(np.full(len(boards), elo), np.full(len(boards), elo),
                                    np.full(len(boards), clock))).to(device)
    with torch.no_grad():
        F = fb.embed_F(planes, om).cpu().numpy()
        B = fb.embed_B(planes).cpu().numpy()
    return F, B


def embed_rows(fb, packed: np.ndarray, meta: np.ndarray, device,
               elo_w: np.ndarray, elo_b: np.ndarray, clock: np.ndarray, near: bool = False):
    import torch
    planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
    om = torch.from_numpy(omega_ids(elo_w, elo_b, clock)).to(device)
    embF = fb.embed_F_near if near else fb.embed_F
    embB = fb.embed_B_near if near else fb.embed_B
    with torch.no_grad():
        F = embF(planes, om).cpu().numpy()
        B = embB(planes).cpu().numpy()
    return F, B


def dist_like(fb, F: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance-like quantity per (row-aligned) pair: the metric d itself in
    quasimetric mode, else -score (= -dot). (n,d),(n,d) -> (n,)."""
    import torch
    if fb.quasimetric:
        with torch.no_grad():
            return fb.distance_matrix(torch.from_numpy(F), torch.from_numpy(B)).diagonal().numpy()
    return -np.einsum("nd,nd->n", F, B)


def score_pairs_rowwise(fb, F: np.ndarray, B: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        f, b = torch.from_numpy(F), torch.from_numpy(B)
        if fb.quasimetric:
            return (torch.einsum("nd,de,ne->n", f, fb.W.cpu(), b)
                    - fb.distance_matrix(f, b).diagonal()).numpy()
        return np.einsum("nd,nd->n", F, B)


# ------------------------------------------------------------ shard pairs
def load_holdout_games(shard_dir: Path, n_games: int, seed: int, min_plies: int = 12):
    """Contiguous per-game rows from holdout games: list of dicts with
    packed/meta/elo/clock arrays, one entry per game."""
    games = []
    for path in sorted(shard_dir.glob("shard_*.npz")):
        npz = np.load(path)
        data = {k: npz[k] for k in npz.files}
        gid = data["game_id"]
        held = np.unique(gid[gid % HOLDOUT_MOD == 0])
        for g in held:
            lo, hi = np.searchsorted(gid, [g, g + 1])
            if hi - lo < min_plies:
                continue
            games.append({k: data[k][lo:hi] for k in
                          ("packed", "meta", "ply", "white_elo", "black_elo", "clock", "result")})
            if len(games) >= n_games:
                return games
    return games


# ------------------------------------------------------------- instruments
def random_krvk(rng: np.random.Generator, max_tries: int = 200) -> chess.Board | None:
    """Random legal White K+R vs K position, White to move, not already over.
    Pawnless 3-piece: the only zeroing move is the rook capture/mate line, so
    tablebase DTZ here IS the true distance-to-mate (up to 50-move rule) --
    a far better calibration target than KRRvKBP's capture-compressed DTZ."""
    for _ in range(max_tries):
        squares = rng.choice(64, size=3, replace=False)
        board = chess.Board(None)
        board.set_piece_at(int(squares[0]), chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(int(squares[1]), chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(int(squares[2]), chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        if not board.is_valid() or board.is_game_over():
            continue
        return board
    return None


def syzygy_calibration_krvk(fb, zgoals, device, syzygy_dir: Path, n_positions: int,
                            seed: int) -> dict:
    """d(F(s), zMATE_W) vs DTZ on KRvK winning positions -- pawnless, so
    DTZ = plies-to-mate under optimal play, spread over 1..~32."""
    rng = np.random.default_rng(seed)
    tb = chess.syzygy.open_tablebase(str(syzygy_dir))
    boards, dtzs = [], []
    tries = 0
    while len(boards) < n_positions and tries < n_positions * 60:
        tries += 1
        b = random_krvk(rng)
        if b is None:
            continue
        try:
            if tb.probe_wdl(b) != 2:
                continue
            dtz = abs(int(tb.probe_dtz(b)))
        except (KeyError, chess.syzygy.MissingTableError):
            continue
        boards.append(b)
        dtzs.append(dtz)
    tb.close()
    if len(boards) < 20:
        return dict(error=f"only {len(boards)} probeable KRvK positions found")

    F, _ = embed_boards(fb, boards, device)
    z = zgoals["MATE_W"].cpu().numpy()
    Z = np.tile(z, (len(F), 1)).astype(np.float32)
    d = dist_like(fb, F, Z)
    dtz_arr = np.array(dtzs, dtype=np.float64)

    # nearest-exemplar variant (2026-07-13 finding: centroids are flat but
    # min-distance over genuine same-material mates correlates -- this is
    # the calibration number that actually tracks embedding progress)
    edge = [s for s in range(64)
            if chess.square_rank(s) in (0, 7) or chess.square_file(s) in (0, 7)]
    mate_boards = []
    for bk in edge:
        for wk in range(64):
            if chess.square_distance(bk, wk) < 2:
                continue
            for r_sq in range(64):
                if r_sq in (bk, wk):
                    continue
                b = chess.Board(None)
                b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                b.set_piece_at(r_sq, chess.Piece(chess.ROOK, chess.WHITE))
                b.turn = chess.BLACK
                if b.is_valid() and b.is_checkmate():
                    mate_boards.append(b)
    _, Bm = embed_boards(fb, mate_boards, device)
    import torch
    if fb.quasimetric:
        with torch.no_grad():
            D = fb.distance_matrix(torch.from_numpy(F), torch.from_numpy(Bm)).numpy()
        d_nearest = D.min(axis=1)
    else:
        d_nearest = (-(F @ Bm.T)).min(axis=1)

    from scipy.stats import spearmanr
    rho = spearmanr(d, dtz_arr).statistic
    rho_nearest = spearmanr(d_nearest, dtz_arr).statistic
    bins = {}
    for lo, hi in ((1, 4), (5, 8), (9, 14), (15, 22), (23, 40)):
        m = (dtz_arr >= lo) & (dtz_arr <= hi)
        if m.sum() >= 5:
            bins[f"dtz_{lo}_{hi}"] = dict(n=int(m.sum()), mean_d=float(d[m].mean()),
                                          std_d=float(d[m].std()))
    return dict(n=len(boards), spearman_rho_d_vs_dtz=float(rho),
               spearman_rho_nearest_exemplar=float(rho_nearest),
               n_mate_exemplars=len(mate_boards), per_dtz_bins=bins,
               dtz_range=[int(dtz_arr.min()), int(dtz_arr.max())],
               note="KRvK: DTZ == plies-to-mate (pawnless, no interposed zeroing "
                    "moves). rho > 0 wanted; rho_nearest_exemplar is the number "
                    "that tracks real embedding progress (centroids stay flat).")


def syzygy_calibration(fb, zgoals, device, syzygy_dir: Path, n_positions: int,
                       seed: int) -> dict:
    """d(F(s), zMATE_of_winning_side) vs tablebase DTZ on KRRvKBP-family
    winning positions (winning side to move). Uses the SAME generator as
    the diagnostic set so material/colors match the tables we have."""
    from catspace.diagnostic_krrkbp import random_krrkbp
    rng = np.random.default_rng(seed)
    tb = chess.syzygy.open_tablebase(str(syzygy_dir))
    boards, dtzs = [], []
    tries = 0
    while len(boards) < n_positions and tries < n_positions * 50:
        tries += 1
        b = random_krrkbp(rng)
        try:
            wdl = tb.probe_wdl(b)
            if wdl != 2:
                continue
            dtz = tb.probe_dtz(b)
        except (KeyError, chess.syzygy.MissingTableError):
            continue
        boards.append(b)
        dtzs.append(abs(int(dtz)))
    tb.close()
    if len(boards) < 20:
        return dict(error=f"only {len(boards)} probeable positions found")

    F, _ = embed_boards(fb, boards, device)
    z = zgoals["MATE_W"].cpu().numpy()
    Z = np.tile(z, (len(F), 1)).astype(np.float32)
    d = dist_like(fb, F, Z)
    dtz_arr = np.array(dtzs, dtype=np.float64)

    from scipy.stats import spearmanr
    rho = spearmanr(d, dtz_arr).statistic
    bins = {}
    for lo, hi in ((0, 2), (3, 5), (6, 10), (11, 20), (21, 100)):
        m = (dtz_arr >= lo) & (dtz_arr <= hi)
        if m.sum() >= 3:
            bins[f"dtz_{lo}_{hi}"] = dict(n=int(m.sum()), mean_d=float(d[m].mean()),
                                          std_d=float(d[m].std()))
    return dict(n=len(boards), spearman_rho_d_vs_dtz=float(rho), per_dtz_bins=bins,
               note="rho > 0 wanted: farther-from-mate (higher DTZ) should read as larger d")


def horizon_retrieval(fb, games: list, device, n_queries: int, seed: int,
                      near: bool = False) -> dict:
    """For each horizon k: does the true k-later position outscore 63
    cross-game negatives? Accuracy per k. near=True scores with the
    two-horizon NEAR head (cosine) instead of the far head -- the near
    head's short-k accuracy (k=1) is the sharpness the gate must preserve."""
    import torch
    rng = np.random.default_rng(seed)
    Bneg = []
    for g in games[: min(len(games), 200)]:
        i = int(rng.integers(len(g["packed"])))
        _, B = embed_rows(fb, g["packed"][i:i + 1], g["meta"][i:i + 1], device,
                          g["white_elo"][i:i + 1], g["black_elo"][i:i + 1],
                          g["clock"][i:i + 1], near=near)
        Bneg.append(B[0])
    Bneg = np.stack(Bneg)

    def sim(F, cands):
        if near:                                  # cosine (near head is not quasimetric)
            return (F @ cands.T)[0]
        with torch.no_grad():
            return fb.score_matrix(torch.from_numpy(F), torch.from_numpy(cands)).numpy()[0]

    out = {}
    for k in HORIZONS:
        correct = total = 0
        for g in games:
            n = len(g["packed"])
            if n < k + 2:
                continue
            i = int(rng.integers(0, n - k - 1))
            F, _ = embed_rows(fb, g["packed"][i:i + 1], g["meta"][i:i + 1], device,
                              g["white_elo"][i:i + 1], g["black_elo"][i:i + 1],
                              g["clock"][i:i + 1], near=near)
            _, Bpos = embed_rows(fb, g["packed"][i + k:i + k + 1], g["meta"][i + k:i + k + 1],
                                 device, g["white_elo"][i + k:i + k + 1],
                                 g["black_elo"][i + k:i + k + 1], g["clock"][i + k:i + k + 1],
                                 near=near)
            n_negs = min(63, len(Bneg) - 1)
            negs = Bneg[rng.choice(len(Bneg), size=n_negs, replace=False)]
            cands = np.concatenate([Bpos, negs])
            correct += int(np.argmax(sim(F, cands)) == 0)
            total += 1
            if total >= n_queries:
                break
        out[f"k={k}"] = dict(acc=float(correct / max(1, total)), n=total,
                             chance=1.0 / (1 + n_negs))
    return out


def asymmetry_audit(fb, games: list, device, n_pairs: int, seed: int) -> dict:
    """Forward (s -> later g, material decreased) vs reverse (g -> s):
    reverse should read farther. Plus the infinity column: goals with MORE
    pieces than the state are unreachable and must score far."""
    rng = np.random.default_rng(seed)

    def n_pieces(packed_row) -> int:
        return int(sum(bin(int(m)).count("1") for m in packed_row))

    fwd_d, rev_d = [], []
    for g in games:
        n = len(g["packed"])
        if n < 10:
            continue
        i = int(rng.integers(0, n - 8))
        j = int(rng.integers(i + 4, n))
        pi, pj = n_pieces(g["packed"][i]), n_pieces(g["packed"][j])
        if pj >= pi:                    # need a capture in between
            continue
        Fi, Bi = embed_rows(fb, g["packed"][i:i + 1], g["meta"][i:i + 1], device,
                            g["white_elo"][i:i + 1], g["black_elo"][i:i + 1], g["clock"][i:i + 1])
        Fj, Bj = embed_rows(fb, g["packed"][j:j + 1], g["meta"][j:j + 1], device,
                            g["white_elo"][j:j + 1], g["black_elo"][j:j + 1], g["clock"][j:j + 1])
        # forward s -> g is feasible real play; reverse g -> s would require
        # un-capturing (piece count strictly increases) -- the infinity
        # direction, IQE's infinite-distance column and PQE's one-way door
        # rolled into one real-data probe
        fwd_d.append(float(dist_like(fb, Fi, Bj)[0]))
        rev_d.append(float(dist_like(fb, Fj, Bi)[0]))
        if len(fwd_d) >= n_pairs:
            break

    fwd, rev = np.array(fwd_d), np.array(rev_d)
    if len(fwd) < 10:
        return dict(error="too few capture-boundary pairs found")
    return dict(n=len(fwd),
               mean_forward_d=float(fwd.mean()), mean_reverse_d=float(rev.mean()),
               frac_reverse_leq_forward=float((rev <= fwd).mean()),
               mean_gap=float((rev - fwd).mean()),
               note="frac_reverse_leq_forward ~0 wanted: un-capturing is impossible, "
                    "reverse must read farther; ~0.5 means the metric is blind to the "
                    "arrow of material")


def triangle_violation(fb, games: list, device, n_triples: int, seed: int) -> dict:
    """PQE Defn 4.2: vio = d(x,z) / (d(x,y) + d(y,z)) over sampled triples.
    On the metric d (quasimetric mode): must be <= 1 everywhere
    (architecture guarantee; this is the regression test against the
    IQE-documented MRN non-negativity bug class). On the full score's
    distance-like -(r - d): NOT guaranteed -- the interesting number."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in games[: min(len(games), 300)]:
        i = int(rng.integers(len(g["packed"])))
        rows.append((g, i))
    F_all, B_all = [], []
    for g, i in rows:
        F, B = embed_rows(fb, g["packed"][i:i + 1], g["meta"][i:i + 1], device,
                          g["white_elo"][i:i + 1], g["black_elo"][i:i + 1], g["clock"][i:i + 1])
        F_all.append(F[0]); B_all.append(B[0])
    F_all, B_all = np.stack(F_all), np.stack(B_all)

    import torch
    n = len(F_all)
    idx = rng.integers(0, n, size=(n_triples, 3))
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]

    out = {}
    with torch.no_grad():
        if fb.quasimetric:
            tF, tB = torch.from_numpy(F_all), torch.from_numpy(B_all)
            D = fb.distance_matrix(tF, tB).numpy()
            dxz, dxy, dyz = D[x, z], D[x, y], D[y, z]
            vio = dxz / np.maximum(dxy + dyz, 1e-9)
            out["d_only"] = dict(max_vio=float(vio.max()), frac_vio_gt_1=float((vio > 1 + 1e-4).mean()),
                                 note="architectural guarantee: max_vio must be <= 1 + eps")
        S = fb.score_matrix(torch.from_numpy(F_all), torch.from_numpy(B_all)).numpy()
    negS = -S                                     # distance-like readout
    shift = max(0.0, float(-negS.min())) + 1e-6   # make nonnegative for the ratio
    negS = negS + shift
    dxz, dxy, dyz = negS[x, z], negS[x, y], negS[y, z]
    vio = dxz / np.maximum(dxy + dyz, 1e-9)
    out["full_score"] = dict(max_vio=float(vio.max()),
                             frac_vio_gt_1=float((vio > 1 + 1e-4).mean()),
                             mean_vio=float(vio.mean()), shift_applied=float(shift),
                             note="r - d readout: violations expected; tracks HOW non-metric "
                                  "the actual planning signal is")
    return out


def degeneracy_panel(fb, games: list, device, n_pairs: int, seed: int) -> dict:
    """Spread ratio + effective rank + norms. spread_ratio ~ 1 = collapse."""
    rng = np.random.default_rng(seed)
    one_ply, cross = [], []
    F_bank, B_bank = [], []
    for g in games[: min(len(games), 300)]:
        n = len(g["packed"])
        i = int(rng.integers(0, n - 1))
        F, B = embed_rows(fb, g["packed"][i:i + 2], g["meta"][i:i + 2], device,
                          g["white_elo"][i:i + 2], g["black_elo"][i:i + 2], g["clock"][i:i + 2])
        one_ply.append(float(dist_like(fb, F[:1], B[1:2])[0]))
        F_bank.append(F[0]); B_bank.append(B[0])
    F_bank, B_bank = np.stack(F_bank), np.stack(B_bank)
    m = len(F_bank)
    pick = rng.integers(0, m, size=(min(n_pairs, m * 4), 2))
    ok = pick[:, 0] != pick[:, 1]
    cross = dist_like(fb, F_bank[pick[ok, 0]], B_bank[pick[ok, 1]])

    def eff_rank(X):
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        p = s / s.sum()
        return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

    return dict(mean_d_one_ply=float(np.mean(one_ply)), mean_d_cross_game=float(np.mean(cross)),
               spread_ratio=float(np.mean(cross) / max(np.mean(one_ply), 1e-9)),
               eff_rank_F=eff_rank(F_bank), eff_rank_B=eff_rank(B_bank), d_dim=fb.d,
               note="spread_ratio ~1 = distance collapse; want cross-game >> one-ply")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--n-games", type=int, default=300)
    ap.add_argument("--n-syzygy", type=int, default=300)
    ap.add_argument("--n-queries", type=int, default=150)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--n-triples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: probe batches are tiny (1-2 rows), and the "
                         "distance/score analysis tensors are built CPU-side -- also keeps "
                         "the probe from contending with MPS training/eval runs")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    zgoals = payload.get("zgoals", {})
    print(f"ckpt={args.ckpt}  quasimetric={fb.quasimetric}  device={device}", flush=True)

    games = load_holdout_games(Path(args.shards), args.n_games, args.seed)
    print(f"{len(games)} holdout games loaded", flush=True)

    report = dict(ckpt=str(args.ckpt), quasimetric=bool(fb.quasimetric), seed=args.seed)
    report["syzygy_calibration"] = syzygy_calibration(fb, zgoals, device, Path(args.syzygy_dir),
                                                      args.n_syzygy, args.seed)
    report["syzygy_calibration_krvk"] = syzygy_calibration_krvk(fb, zgoals, device,
                                                                Path(args.syzygy_dir),
                                                                args.n_syzygy, args.seed)
    print("syzygy done", flush=True)
    report["horizon_retrieval"] = horizon_retrieval(fb, games, device, args.n_queries, args.seed)
    if getattr(fb, "two_horizon", False):
        # the far metrics above use embed_F/embed_B (the far head); add the
        # NEAR head's retrieval -- its k=1 is the short-horizon sharpness the
        # two-horizon gate must keep at ~0.97 (TWO_HORIZON_DESIGN.md)
        report["horizon_retrieval_near"] = horizon_retrieval(
            fb, games, device, args.n_queries, args.seed, near=True)
    print("retrieval done", flush=True)
    report["asymmetry"] = asymmetry_audit(fb, games, device, args.n_pairs, args.seed)
    print("asymmetry done", flush=True)
    report["triangle"] = triangle_violation(fb, games, device, args.n_triples, args.seed)
    print("triangle done", flush=True)
    report["degeneracy"] = degeneracy_panel(fb, games, device, args.n_pairs, args.seed)
    print("degeneracy done", flush=True)

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
