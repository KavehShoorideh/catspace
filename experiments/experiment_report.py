#!/usr/bin/env python
"""
experiments/experiment_report.py — the A/B experimentation harness: audits a
candidate TorchFB checkpoint for Stockfish-oracle leakage (catspace.audit,
HARD gate -- refuses to proceed if either the static code-path check or the
checkpoint's own provenance stamp comes back dirty), then runs the same
metric suite the viz builders compute (reach/diff slopes, the M1.5
decomposer's FRAC_IMPROVED/MEAN_GAIN, an arena run vs a fixed opponent with
catspace.abtest.EValueTest's anytime-valid e-value verdict), and -- if
--baseline is given -- a direct candidate-vs-baseline head-to-head with the
SAME e-value machinery (experiments.arena_real.run_arena, generalized to
take a color-specific opponent). Writes ONE structured JSON record per run
to artifacts/experiments/ (tracked in git, unlike the regenerable viz HTML)
so a sequence of "slowly improve the model" runs can be read back, diffed,
and ranked without opening anything -- see experiment_leaderboard.py.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

# experiments/ (unlike catspace/) isn't an installed package -- only on
# sys.path when the repo root happens to be there (python -m, or a `-c`
# invocation from repo root). Importing experiments.arena_real below (to
# reuse run_arena/make_opponent instead of duplicating them) needs this
# regardless of how this script itself was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from catspace.audit import audit_checkpoint
from catspace.data.encode import board_from_packed
from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, experiments_dir, newest_shard_dir
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.policy_fb import FBBoardPolicy, make_search_policy
from catspace.planner.decompose import WaypointPool, decompose, hop_reach
from catspace.viz.payload import json_default as _numpy_json_default  # reuse the same numpy-JSON fix

PLANNER_ELO, PLANNER_CLOCK = 1800, 300.0


def file_sha256(path: Path, n_bytes: int = 1 << 20) -> str:
    """Hash just the first megabyte -- identity, not integrity (checkpoints
    are large; this is enough to tell two saves apart, not to detect bitrot)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


# ------------------------------------------------------------- reach slopes
# duplicated from experiments/train_lichess_fb.py::reach_slope rather than
# imported: that module's reach_slope is a private-ish helper next to a CLI
# `main()`, and importing it would pull in argparse-time behavior on import
# for no benefit -- the ~15 lines are cheap and this keeps the report driver
# independent of the training script's internals.
def reach_slope(shard_dir: Path, fb, z, device, want_result: int, n_games: int = 200):
    from scipy.stats import spearmanr
    rhos = []
    for path in sorted(shard_dir.glob("shard_*.npz")):
        npz = np.load(path)
        data = {k: npz[k] for k in npz.files}
        gid = data["game_id"]
        held = (gid % 50 == 0) & (data["result"] == want_result)
        for g in np.unique(gid[held]):
            lo, hi = np.searchsorted(gid, [g, g + 1])
            rows = np.arange(lo, hi)
            if len(rows) < 10:
                continue
            planes = torch.from_numpy(feature_planes(data["packed"][rows], data["meta"][rows])).to(device)
            om = torch.from_numpy(omega_ids(data["white_elo"][rows], data["black_elo"][rows],
                                            data["clock"][rows])).to(device)
            with torch.no_grad():
                reach = fb.score(fb.embed_F(planes, om), z.to(device)).cpu().numpy()
            rho = spearmanr(data["ply"][rows], reach).statistic
            if np.isfinite(rho):
                rhos.append(rho)
            if len(rhos) >= n_games:
                return float(np.mean(rhos)), len(rhos)
    return float(np.mean(rhos)) if rhos else float("nan"), len(rhos)


def compute_reach_slopes(shard_dir: Path, fb, zgoals: dict, device) -> dict:
    slope_w, nw = reach_slope(shard_dir, fb, zgoals["MATE_W"], device, want_result=1)
    slope_l, nl = reach_slope(shard_dir, fb, zgoals["MATE_W"], device, want_result=-1)
    dslope_w, _ = reach_slope(shard_dir, fb, zgoals["MATE_DIFF"], device, want_result=1)
    dslope_l, _ = reach_slope(shard_dir, fb, zgoals["MATE_DIFF"], device, want_result=-1)
    return dict(reach_slope_won=slope_w, reach_slope_lost=slope_l, n_won=nw, n_lost=nl,
               diff_slope_won=dslope_w, diff_slope_lost=dslope_l)


# ------------------------------------------------------------- decompose metrics
# duplicated (not imported) from experiments/viz/build_decompose_viewer.py
# for the same reason -- this repo's convention is small self-contained
# scripts, see e.g. decompose_demo.py vs build_decompose_viewer.py.
def load_rows(shard_dir: Path, picks: list) -> dict:
    by_file: dict = {}
    for name, row in picks:
        by_file.setdefault(name, []).append(row)
    cols = ("packed", "meta", "ply", "result", "game_id")
    out: dict = {k: [] for k in cols}
    for name, rows in sorted(by_file.items()):
        npz = np.load(shard_dir / name)
        idx = np.array(sorted(rows))
        for k in cols:
            out[k].append(npz[k][idx])
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def embed_rows(fb, data, device, batch=2048):
    Fs, Bs = [], []
    om_row = omega_ids(np.array([PLANNER_ELO]), np.array([PLANNER_ELO]), np.array([PLANNER_CLOCK]))[0]
    n = len(data["packed"])
    for i in range(0, n, batch):
        planes = torch.from_numpy(feature_planes(
            data["packed"][i:i + batch], data["meta"][i:i + batch])).to(device)
        om = torch.from_numpy(np.tile(om_row, (len(planes), 1))).to(device)
        Fs.append(fb.embed_F(planes, om).cpu().numpy())
        Bs.append(fb.embed_B(planes).cpu().numpy())
    return np.concatenate(Fs), np.concatenate(Bs)


def compute_decompose_metrics(shard_dir: Path, fb, zgoals: dict, device, n_pool: int, n_starts: int,
                              start_ply_lo: int = 20, start_ply_hi: int = 40, max_depth: int = 3,
                              dry_gain: float = 0.02, seed: int = 0) -> dict:
    zg = zgoals["MATE_W"].numpy().astype(np.float32)
    z_goal = zg / np.linalg.norm(zg)

    picks = sample_shard_rows(shard_dir, n_pool + 4 * n_starts, seed=seed, holdout_only=True)
    data = load_rows(shard_dir, picks)
    is_start = (data["ply"] >= start_ply_lo) & (data["ply"] <= start_ply_hi)
    start_idx = np.flatnonzero(is_start)[:n_starts]
    pool_idx = np.setdiff1d(np.arange(len(data["ply"])), start_idx)[:n_pool]

    F_all, B_all = embed_rows(fb, data, device)
    pool = WaypointPool(F=F_all[pool_idx], B=B_all[pool_idx], labels=[int(i) for i in pool_idx])

    npz = np.load(sorted(shard_dir.glob("shard_*.npz"))[0])
    gid, ply, res = npz["game_id"], npz["ply"], npz["result"]
    last_ply = np.zeros(gid.max() + 1, dtype=ply.dtype)
    np.maximum.at(last_ply, gid, ply)
    nw_rows = np.flatnonzero((res == 1) & (ply >= last_ply[gid] - 10) & (gid % 50 == 0))
    rng = np.random.default_rng(seed)
    nw_rows = rng.choice(nw_rows, size=min(2000, len(nw_rows)), replace=False)
    nw_data = {k: npz[k][np.sort(nw_rows)] for k in ("packed", "meta")}
    F_nw, _ = embed_rows(fb, nw_data, device)
    # score through fb.np_score_matrix everywhere: exactly the dot product on
    # non-quasimetric checkpoints (bit-identical to the old F @ z), and the
    # only correctly-calibrated score on quasimetric ones (2026-07-12 review)
    sp = fb.np_score_matrix
    tau_exec = float(np.median(sp(F_nw, z_goal[None, :])[:, 0]))
    tau_floor = float(np.quantile(sp(F_all[start_idx], z_goal[None, :])[:, 0], 0.10))

    results = []
    for si in start_idx:
        dec = decompose(F_all[si], z_goal, pool, tau_exec=tau_exec, tau_floor=tau_floor,
                        dry_gain=dry_gain, max_depth=max_depth, score_pairs=sp)
        results.append((int(si), dec))

    direct = np.array([hop_reach(F_all[si], z_goal, sp) for si, _ in results])
    bottle = np.array([dec.plan_bottleneck for _, dec in results])
    gain = bottle - direct
    execf = np.array([dec.executable for _, dec in results])
    n_way = np.array([len(dec.waypoints) for _, dec in results])
    return dict(frac_improved=float((gain > 0).mean()), mean_gain=float(gain.mean()),
               frac_executable=float(execf.mean()), mean_waypoints=float(n_way.mean()),
               tau_exec=tau_exec, tau_floor=tau_floor, n_starts=len(results), n_pool=len(pool))


def load_and_audit(ckpt_path: Path, device: str, label: str) -> tuple:
    fb, payload = load_ckpt(ckpt_path, device)
    fb.eval()
    if "MATE_W" not in payload.get("zgoals", {}):
        raise SystemExit(f"{label} checkpoint {ckpt_path} has no zgoals -- "
                         f"finish a train_lichess_fb.py run first")
    audit = audit_checkpoint(payload)
    if not audit["clean"]:
        raise SystemExit(f"LEAKAGE AUDIT FAILED for {label} checkpoint {ckpt_path}:\n"
                         f"{json.dumps(audit, indent=2, default=_numpy_json_default)}\n"
                         f"Refusing to run -- this checkpoint (or the current code) may have "
                         f"seen Stockfish-derived signal.")
    return fb, payload, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="candidate checkpoint (default: data/derived/lichess_fb.pt)")
    ap.add_argument("--baseline", default=None, help="optional previous checkpoint for direct head-to-head")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--opponent", default="random", help="random | sf:<elo> | sf:skill=<k>")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--baseline-games", type=int, default=40)
    ap.add_argument("--depth", type=int, default=2, choices=(1, 2),
                    help="FBBoardPolicy readout depth (ignored if --search-depth is given)")
    ap.add_argument("--search-nodes", type=int, default=None,
                    help="use FBSearchPolicy (beam-limited multi-ply minimax over the SAME "
                         "F(s)@z value, no retraining) instead of FBBoardPolicy for the "
                         "candidate's side, with this fixed node budget per move (depth is "
                         "derived from the real branching factor to spend it -- see "
                         "catspace.nn.policy_fb.FBSearchPolicy)")
    ap.add_argument("--search-beam", type=int, default=4,
                    help="branching cap per ply beyond the root, only used with --search-nodes")
    ap.add_argument("--search", choices=("beam", "mcts"), default="beam",
                    help="readout used with --search-nodes: beam minimax (default) or PUCT "
                         "MCTS at the same node budget (catspace/nn/mcts.py)")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--elo-cond", type=int, default=1800)
    ap.add_argument("--skip-decompose", action="store_true")
    ap.add_argument("--n-pool", type=int, default=20_000)
    ap.add_argument("--n-decompose-starts", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tag", default=None, help="free-text label for this run (e.g. 'lr=3e-4,steps=40000')")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from experiments.arena_real import make_opponent, run_arena
    from catspace.uci import UCIBoardPolicy

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    device = pick_device(args.device)

    report = dict(timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                 tag=args.tag, shards=shard_dir.name, device=device,
                 args=vars(args))

    t0 = time.time()
    fb, payload, audit = load_and_audit(ckpt_path, device, "candidate")
    report["candidate"] = dict(path=str(ckpt_path), sha256=file_sha256(ckpt_path),
                               step=payload.get("step", "?"), leakage_audit=audit)
    print(f"[{time.time() - t0:.1f}s] candidate {ckpt_path.name} step={payload.get('step')} "
          f"AUDIT={'CLEAN' if audit['clean'] else 'DIRTY'}")

    t0 = time.time()
    report["candidate"]["reach_slopes"] = compute_reach_slopes(shard_dir, fb, payload["zgoals"], device)
    print(f"[{time.time() - t0:.1f}s] reach slopes: {report['candidate']['reach_slopes']}")

    if not args.skip_decompose:
        t0 = time.time()
        report["candidate"]["decompose"] = compute_decompose_metrics(
            shard_dir, fb, payload["zgoals"], device, args.n_pool, args.n_decompose_starts, seed=args.seed)
        print(f"[{time.time() - t0:.1f}s] decompose: {report['candidate']['decompose']}")

    def make_candidate_policy(fb_, z):
        if args.search_nodes is not None:
            return make_search_policy(args.search, fb_, z, max_nodes=args.search_nodes,
                                      beam=args.search_beam, c_puct=args.c_puct,
                                      elo=args.elo_cond, device=device)
        return FBBoardPolicy(fb_, z, depth=args.depth, elo=args.elo_cond, device=device)

    fb_white = make_candidate_policy(fb, payload["zgoals"]["MATE_W"])
    fb_black = make_candidate_policy(fb, payload["zgoals"]["MATE_B"])
    readout = (f"{args.search}(max_nodes={args.search_nodes}, beam={args.search_beam})"
              if args.search_nodes is not None else f"FBBoardPolicy(depth={args.depth})")
    print(f"readout: {readout}")

    opponent, opp_name = make_opponent(args.opponent)
    t0 = time.time()

    def run_vs_opponent():
        return run_arena(fb_white, fb_black, opponent, args.games, args.opening_plies,
                         args.max_plies, args.seed, alpha=args.alpha, verbose=False)

    if isinstance(opponent, UCIBoardPolicy):
        with opponent:
            arena = run_vs_opponent()
    else:
        arena = run_vs_opponent()
    arena.pop("records")   # BoardGameRecord objects aren't JSON-safe; the summary is what's stored
    arena["opponent"] = opp_name
    report["candidate"]["arena_vs_opponent"] = arena
    print(f"[{time.time() - t0:.1f}s] arena vs {opp_name}: +{arena['wins']} ={arena['draws']} "
          f"-{arena['losses']}  score={arena['score_mean']:.3f}  e={arena['e_value']:.2f}  "
          f"{'REJECT' if arena['reject_at_alpha'] else 'no-reject'}")

    report["baseline"] = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        t0 = time.time()
        fb_b, payload_b, audit_b = load_and_audit(baseline_path, device, "baseline")
        report["baseline"] = dict(path=str(baseline_path), sha256=file_sha256(baseline_path),
                                  step=payload_b.get("step", "?"), leakage_audit=audit_b)
        print(f"[{time.time() - t0:.1f}s] baseline {baseline_path.name} step={payload_b.get('step')} "
              f"AUDIT={'CLEAN' if audit_b['clean'] else 'DIRTY'}")

        base_white = make_candidate_policy(fb_b, payload_b["zgoals"]["MATE_W"])
        base_black = make_candidate_policy(fb_b, payload_b["zgoals"]["MATE_B"])
        t0 = time.time()
        h2h = run_arena(fb_white, fb_black, (base_white, base_black), args.baseline_games,
                        args.opening_plies, args.max_plies, args.seed, alpha=args.alpha, verbose=False)
        h2h.pop("records")
        report["head_to_head"] = h2h
        print(f"[{time.time() - t0:.1f}s] candidate vs baseline: +{h2h['wins']} ={h2h['draws']} "
              f"-{h2h['losses']}  score={h2h['score_mean']:.3f}  e={h2h['e_value']:.2f}  "
              f"{'REJECT(candidate!=baseline)' if h2h['reject_at_alpha'] else 'no-reject'}")

    stamp = report["timestamp"].replace(":", "").replace("-", "")
    out = Path(args.out) if args.out else \
        experiments_dir() / f"{stamp}__step{report['candidate']['step']}__{report['candidate']['sha256']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=_numpy_json_default))
    print(f"\nwrote {out}")

    dslope = report["candidate"]["reach_slopes"]
    print(f"VERDICT step={report['candidate']['step']} AUDIT={'CLEAN' if audit['clean'] else 'DIRTY'} "
          f"DIFF_SLOPE_WON={dslope['diff_slope_won']:+.3f} DIFF_SLOPE_LOST={dslope['diff_slope_lost']:+.3f} "
          f"ARENA_vs_{opp_name}={arena['score_mean']:.3f}(e={arena['e_value']:.2f})"
          + (f" H2H_vs_baseline={report['head_to_head']['score_mean']:.3f}"
             f"(e={report['head_to_head']['e_value']:.2f})" if report.get("head_to_head") else ""))


if __name__ == "__main__":
    main()
