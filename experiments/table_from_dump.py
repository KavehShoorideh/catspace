#!/usr/bin/env python
"""
experiments/table_from_dump.py — build a certainty table from a rollout dump.

The dump (certainty_rollouts --dump-rollouts) is append-only jsonl, one line
per rollout -- so this works MID-RUN on a partial file, and builds NESTED
tables for the data-scaling curve (--max-rollouts K uses only each start's
first K rollouts: growing K = growing data on the same distribution).

Quality report (the mid-run health check):
  coverage   starts seen, rollouts, unique states, states >= min-visits
  spread     P-hat mean / frac 0 / frac 1 / frac mid (gate: mid >= 0.3)
  depth      visits per kept state (median / p90)
  truth      discrimination vs the tablebase on a sample of kept states:
             mean P-hat on tb-WON vs tb-not-won states and their gap --
             P-hat measures conversion under NOISY play, so it should sit
             well below 1 on won states, but the GAP must be positive and
             clear or the table is teaching the wrong field.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def outcome_class(rec):
    """Boundary the rollout hit (committor targets). The WIN boundary is a
    UNION of surfaces -- mate and opponent flag-fall (Kaveh 2026-07-15: "win
    outcomes should include mate and timeout"); they're recorded as distinct
    sub-classes because they're reached by different plans (mate by
    conversion, timeout by survive-under-clock), while the d_W committor
    target is P(any WIN_*) -- touchdown semantics over the union. Untimed
    generators never emit TIMEOUT; timed harnesses must record
    reason="TIMEOUT" with `won` set from who flagged. Old dumps (no
    `reason`) degrade to WIN_MATE / OTHER."""
    reason = rec.get("reason")
    if rec.get("won"):
        return {"TIMEOUT": "WIN_TIMEOUT",    # opponent's flag fell
                "RESIGNATION": "WIN_RESIGN", # opponent resigned
                "ABANDONED": "WIN_ABANDON",
                }.get(reason, "WIN_MATE")
    if reason is None:
        return "OTHER"
    # Human-game boundaries (RESIGNATION / DRAW_AGREE / ABANDONED) are
    # BELIEF-ACTIONS, not rule surfaces: a resignation is the human's own
    # committor reading P(win)~0, an agreed draw is both players co-signing
    # P~draw -- weak human labels of the field, biased by the humans' own
    # fallibility. Untimed toy generators never emit them; the Lichess
    # shard-side classifier does (Termination tag + final-board inspection:
    # decisive-without-mate = resignation or flag).
    return {"CHECKMATE": "LOSS_MATE",        # not won + checkmate = White got mated
            "TIMEOUT": "LOSS_TIMEOUT",       # White's own flag fell
            "RESIGNATION": "LOSS_RESIGN",
            "ABANDONED": "LOSS_ABANDON",
            "AGREEMENT": "DRAW_AGREE",
            "STALEMATE": "DRAW_STALE",
            "INSUFFICIENT_MATERIAL": "DRAW_INSUF",
            "THREEFOLD_REPETITION": "DRAW_3FOLD",
            "FIVEFOLD_REPETITION": "DRAW_3FOLD",
            "FIFTY_MOVES": "DRAW_50",
            "SEVENTYFIVE_MOVES": "DRAW_50",
            "MAX_PLIES": "CAP"}.get(reason, "OTHER")


def build(dump_paths, max_rollouts=None, min_visits=4):
    """Aggregate one or more dump files (parallel workers write separate
    dumps; si is global, so merging is exact). Traj entries are [fen, ply]
    (old dumps) or [fen, ply, rep] (v2, repetition count at visit time)."""
    if isinstance(dump_paths, (str, Path)):
        dump_paths = [dump_paths]
    stats = defaultdict(lambda: [0, 0, [], defaultdict(int), 0])
    starts = set()
    rollouts = 0
    for dump_path in dump_paths:
        with open(dump_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                   # mid-write tail line: skip
                if max_rollouts is not None and rec["r"] >= max_rollouts:
                    continue
                starts.add(rec["si"])
                rollouts += 1
                oc = outcome_class(rec)
                for entry in rec["traj"]:
                    fen, ply, rep = (entry if len(entry) == 3 else (*entry, 0))
                    s = stats[fen]
                    s[0] += 1
                    if rec["won"]:
                        s[1] += 1
                        s[2].append(rec["end_ply"] - ply)
                    s[3][oc] += 1
                    s[4] = max(s[4], rep)
    rows = [dict(fen=f, n=v[0], p_hat=v[1] / v[0],
                 plies=(float(np.mean(v[2])) if v[2] else None),
                 outcomes=dict(v[3]), rep_max=v[4])
            for f, v in stats.items() if v[0] >= min_visits]
    return rows, len(starts), rollouts, len(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", nargs="+",
                    default=["artifacts/experiments/rollout_dump_fixedstart.jsonl"],
                    help="one or more dump files (parallel workers write separate dumps; "
                         "si is global so they merge exactly)")
    ap.add_argument("--max-rollouts", type=int, default=None,
                    help="nested-table size knob: first K rollouts per start")
    ap.add_argument("--min-visits", type=int, default=4)
    ap.add_argument("--tb-sample", type=int, default=500,
                    help="kept states to check against the tablebase (0 = skip)")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the table json (omit = report only)")
    args = ap.parse_args()

    rows, n_starts, n_rollouts, n_unique = build(args.dump, args.max_rollouts,
                                                 args.min_visits)
    p = np.array([r["p_hat"] for r in rows])
    n = np.array([r["n"] for r in rows])
    mid = float(((p > 0) & (p < 1)).mean()) if len(p) else float("nan")
    print(f"coverage: {n_starts} starts, {n_rollouts} rollouts, {n_unique} unique "
          f"states, {len(rows)} kept (>= {args.min_visits} visits)")
    if len(p):
        print(f"spread:   P-hat mean {p.mean():.2f}  frac0 {(p == 0).mean():.2f}  "
              f"frac1 {(p == 1).mean():.2f}  fracMID {mid:.2f} "
              f"[gate >= 0.30: {'PASS' if mid >= 0.30 else 'FAIL'}]")
        print(f"depth:    visits median {np.median(n):.0f}  p90 {np.percentile(n, 90):.0f}")
        oc = defaultdict(int)
        for r in rows:
            for k, v in r.get("outcomes", {}).items():
                oc[k] += v
        tot = sum(oc.values())
        if tot:
            print("boundaries: " + "  ".join(f"{k} {v/tot:.2f}" for k, v in
                                             sorted(oc.items(), key=lambda kv: -kv[1])))
    if args.tb_sample and len(rows):
        # kept (>=min-visits) states are ~all tb-WON by construction: Black is
        # tb-optimal (won stays won unless White throws it) and each thrown
        # line diverges uniquely, never reaching the visit floor. So the
        # truth check is WITHIN-won validity: P-hat should FALL as the win
        # gets longer to convert (|dtz| proxy) -- more plies = more chances
        # to blunder under eps-noise. Positive rho = table teaches the
        # certainty gradient; ~0/negative = noise.
        from experiments.certainty_distill import spearman_ci
        from experiments.value_fixed_point import TB, white_pov_value
        tb = TB(args.syzygy_dir)
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(rows), size=min(args.tb_sample, len(rows)), replace=False)
        ph, dtz, nonwon = [], [], 0
        for i in pick:
            b = chess.Board(rows[int(i)]["fen"])
            v = white_pov_value(b, tb)
            if v is None:
                continue
            if v != 1.0:
                nonwon += 1
                continue
            _, d = tb.wdl_dtz(b)
            if d is not None:
                ph.append(rows[int(i)]["p_hat"])
                dtz.append(abs(d))
        tb.close()
        if len(ph) >= 20:
            r, lo, hi = spearman_ci(np.array(ph), -np.array(dtz))
            print(f"truth:    within-won gradient Spearman(P-hat, -|dtz|) = {r:+.3f} "
                  f"CI[{lo:+.3f},{hi:+.3f}] (n={len(ph)} won, {nonwon} non-won sampled) "
                  f"[{'HEALTHY' if lo > 0 else 'SUSPECT'}]")
    if args.out and len(rows):
        Path(args.out).write_text(json.dumps(dict(source=str(args.dump),
                                                  max_rollouts=args.max_rollouts,
                                                  rows=rows)))
        print(f"-> {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
