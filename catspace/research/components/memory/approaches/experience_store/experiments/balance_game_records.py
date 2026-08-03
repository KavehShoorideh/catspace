#!/usr/bin/env python
"""catspace/research/components/memory/approaches/experience_store/experiments/balance_game_records.py -- STAGE B of the identity-preserving data pipeline.

Reads Stage-A parquet game records (build_game_records.py) and does two jobs:

 1. EVENNESS RE-CHECK (the game-level successor to data_distribution_check.py, which read the old
    position shards). Reports, per GAME: OUTCOME balance (W/D/L), STRENGTH distribution (min-Elo
    band + normalized-entropy evenness), PHASE (game length), and -- new, gates the z-encoder --
    the PER-PLAYER game-count distribution (how many identities clear the >=20-game training bar vs
    the 5-10-game online-inference regime).

 2. STRATIFIED BALANCER. Resamples records to a target OUTCOME x STRENGTH-BAND distribution
    (default: uniform over occupied strata) with a bounded oversample factor (no pathological
    replication), and writes the balanced record set + a before/after evenness JSON. This fixes the
    draw-starvation (4.1%) and strength-skew (evenness 0.79) that the original check caught -- as
    far as lichess-only supply allows; the residual gap (what only engine/CCRL data can fill) is
    reported explicitly, never silently.

Balancing is a SAMPLING step over records (reproducible via --seed), not baked into the scan --
so it is tunable and testable independently, and engine records ingest into the same strata.
"""
from __future__ import annotations

import argparse, glob, json, sys, time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catspace.io import paths


ELO_BANDS = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 4000]
OUTCOMES = {1: "White win", 0: "draw", -1: "Black win"}


def _bar(frac, width=30):
    n = int(round(frac * width))
    return "#" * n + "-" * (width - n)


def _evenness(counts):
    """normalized entropy of a count vector (1.0 = perfectly even over occupied bins)."""
    p = np.asarray(counts, float); p = p[p > 0]; p = p / p.sum()
    k = len(counts)
    return float(-(p * np.log(p)).sum() / np.log(k)) if k > 1 and len(p) > 1 else 0.0


def load_records(rec_dir):
    files = sorted(glob.glob(str(Path(rec_dir) / "*.parquet")))
    if not files:
        sys.exit(f"no parquet records under {rec_dir}")
    tbl = pa.concat_tables([pq.read_table(f) for f in files])
    return tbl


def band_index(min_elo):
    return np.clip(np.searchsorted(ELO_BANDS, min_elo, side="right") - 1, 0, len(ELO_BANDS) - 2)


def report_evenness(d, tag, out=None):
    n = len(d["result"]); res = np.asarray(d["result"]); we = np.asarray(d["white_elo"]); be = np.asarray(d["black_elo"])
    nply = np.asarray(d["n_plies"])
    rep = {"tag": tag, "n_games": int(n)}
    print(f"\n===== EVENNESS [{tag}] : {n:,} games =====")

    # OUTCOME
    print("OUTCOME (per game):")
    oc = {}
    for v in (1, 0, -1):
        c = int((res == v).sum()); oc[OUTCOMES[v]] = c
        print(f"  {OUTCOMES[v]:10} {c/n:6.1%} {_bar(c/n)} ({c:,})")
    draw_frac = oc["draw"] / n
    rep["outcome"] = oc; rep["draw_frac"] = draw_frac
    rep["outcome_evenness"] = _evenness([oc[OUTCOMES[v]] for v in (1, 0, -1)])
    print(f"  outcome EVENNESS {rep['outcome_evenness']:.2f} | draws {draw_frac:.1%} "
          f"{'<-- STARVED (need engine/high-rated draws)' if draw_frac < 0.15 else 'ok'}")

    # STRENGTH (min of the two Elos)
    band = np.minimum(we, be); bi = band_index(band)
    h = np.bincount(bi, minlength=len(ELO_BANDS) - 1)
    print("STRENGTH (per game, min(W,B) Elo band):")
    for i in range(len(ELO_BANDS) - 1):
        f = h[i] / n; flag = "  <-- SPARSE" if f < 0.03 else ""
        print(f"  {ELO_BANDS[i]:>4}-{ELO_BANDS[i+1]:<4} {f:6.1%} {_bar(f)} ({int(h[i]):,}){flag}")
    rep["strength_hist"] = {f"{ELO_BANDS[i]}-{ELO_BANDS[i+1]}": int(h[i]) for i in range(len(h))}
    rep["strength_evenness"] = _evenness(h)
    print(f"  strength EVENNESS {rep['strength_evenness']:.2f} "
          f"({'skewed' if rep['strength_evenness'] < 0.8 else 'reasonably even'})")

    # PHASE (game length)
    print("PHASE (n_plies per game):")
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 120), (120, 10_000)]:
        f = float(((nply >= lo) & (nply < hi)).mean())
        print(f"  plies {lo:>3}-{hi:<4} {f:6.1%} {_bar(f)}")

    # PER-PLAYER game counts (z-encoder trainability)
    ids = np.concatenate([np.asarray(d["white_id"]), np.asarray(d["black_id"])])
    _, cnts = np.unique(ids, return_counts=True)
    n_players = len(cnts)
    for thr in (5, 20, 50):
        rep[f"players_ge_{thr}"] = int((cnts >= thr).sum())
    rep["n_players"] = n_players
    print(f"PLAYERS: {n_players:,} unique | games/player med {int(np.median(cnts))} max {int(cnts.max())}")
    print(f"  z-TRAINABLE >=20 games: {rep['players_ge_20']:,} ({rep['players_ge_20']/n_players:.1%}) | "
          f">=50: {rep['players_ge_50']:,} | >=5 (online-inference regime): {rep['players_ge_5']:,}")
    if out is not None:
        out.append(rep)
    return rep


def stratified_resample(d, rng, target="uniform", max_oversample=3.0, cap_per_stratum=None):
    """Resample record indices toward an even OUTCOME x STRENGTH-BAND distribution. Oversamples
    sparse strata (with replacement) up to max_oversample x their supply; subsamples dense ones.
    Returns (indices, info)."""
    res = np.asarray(d["result"]); band = np.minimum(np.asarray(d["white_elo"]), np.asarray(d["black_elo"]))
    bi = band_index(band)
    strata = {}
    for oc in (1, 0, -1):
        for bidx in range(len(ELO_BANDS) - 1):
            idx = np.where((res == oc) & (bi == bidx))[0]
            if len(idx):
                strata[(oc, bidx)] = idx
    supplies = {k: len(v) for k, v in strata.items()}
    # target count per stratum: even -> the median supply (robust to the tiny + huge strata)
    med = int(np.median(list(supplies.values())))
    tgt = cap_per_stratum or med
    out_idx = []
    info = {"n_strata": len(strata), "target_per_stratum": int(tgt), "strata": {}}
    for k, idx in strata.items():
        supply = len(idx)
        take = min(int(tgt), int(round(supply * max_oversample)))       # cap oversampling
        replace = take > supply
        chosen = rng.choice(idx, size=take, replace=replace)
        out_idx.append(chosen)
        info["strata"][f"o{k[0]}_b{k[1]}"] = {"supply": supply, "taken": int(take), "oversampled": bool(replace)}
    out_idx = np.concatenate(out_idx); rng.shuffle(out_idx)
    return out_idx, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=paths.records("smoke_lichess"))
    ap.add_argument("--balance", type=int, default=1, help="1=write a stratified-balanced record set")
    ap.add_argument("--out", default="", help="balanced output dir (default: <records>_balanced)")
    ap.add_argument("--max-oversample", type=float, default=3.0)
    ap.add_argument("--cap-per-stratum", type=int, default=0, help="0 = use median supply")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-json", default=paths.experiment("records_evenness.json"))
    args = ap.parse_args()
    t0 = time.time()

    tbl = load_records(args.records)
    d = tbl.to_pydict()
    reports = []
    report_evenness(d, "RAW " + args.records, reports)

    if args.balance:
        rng = np.random.default_rng(args.seed)
        idx, info = stratified_resample(d, rng, max_oversample=args.max_oversample,
                                        cap_per_stratum=(args.cap_per_stratum or None))
        bal = tbl.take(pa.array(idx))
        report_evenness(bal.to_pydict(), "BALANCED", reports)
        out_dir = Path(args.out or (str(args.records).rstrip("/") + "_balanced"))
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(bal, out_dir / "records_00000.parquet", compression="zstd")
        (out_dir / "balance_info.json").write_text(json.dumps(info, indent=2))
        print(f"\n[balanced] {len(idx):,} rows -> {out_dir} (from {tbl.num_rows:,}; {info['n_strata']} strata, "
              f"target {info['target_per_stratum']}/stratum)")

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(reports, indent=2))
    print(f"\nreport -> {args.report_json} [{time.time()-t0:.0f}s]")
    print("DONE balance_game_records", flush=True)


if __name__ == "__main__":
    main()
