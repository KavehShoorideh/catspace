#!/usr/bin/env python
"""plot_reach_ladder.py -- the LEARNING CURVE for the strata question: effect vs training step.

The single control that decides this experiment is not any one number, it is the TREND. The
previous attempt on the frozen lc0 trunk failed exactly here: paired ratchet 0.585/0.572/0.585/0.570
across its ladder -- a respectable-looking 0.57 that never moved, which is what a property of the
INPUT looks like rather than something the objective learned. So every headline metric is plotted
against step, with the random-init null drawn as a horizontal band rather than quoted in prose.

WHAT IS PLOTTED, and why each panel is there:

  capture - quiet   THE strata claim. Both groups are unobserved reversals given byte-identical
                    repulsion and matched on ply gap, so a uniform training signal cannot separate
                    them -- only the data can. Bootstrap CI per rung; the null band is the same
                    quantity on the random-init checkpoint, and the null is NOT zero (measured
                    -0.057, CI [-0.078,-0.039]), so the effect is the trained-vs-null DIFFERENCE,
                    not "the trained CI excludes 0".

  paired ratchet    The pre-registered cross-game readout, both arms. A source-blind model scores
                    EXACTLY 0.500 here by construction, so 0.500 is the meaningful floor, not 0.

  sep_auc           Does it learn reachability at all? A sanity channel: if this is flat the run
                    is broken and the other panels are noise about nothing.

  confounded        The retracted matched-|delta| metric, kept as a NEGATIVE control. It should sit
                    flat and read the same for a trained and an untrained model; if it ever tracks
                    the real effect, the real effect is suspect too.

READ THE CAVEAT PRINTED ON THE FIGURE. The per-rung CIs are bootstraps over PAIRS, not over
training runs, so they say nothing about seed variance -- one seed cannot distinguish "the
objective learns this" from "this seed learned this". That sentence is drawn on the figure itself
rather than left to a caption nobody carries around with the PNG.

    .venv/bin/python .../plot_reach_ladder.py            # reads artifacts/experiments/*ladder*.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402

from catspace.io import paths                                              # noqa: E402

FIELDS = ("paired_ratchet_region", "paired_ratchet_iqe", "capture_minus_quiet",
          "diff_capture", "diff_quiet", "diff_reversible", "confounded_ratchet",
          "sep_auc", "step")


def parse(path):
    """-> dict of the VERDICT line's key=value pairs, plus the CI if the file carries one."""
    txt = open(path).read()
    m = re.search(r"^VERDICT REACH-STRATA-VIT (.+)$", txt, re.M)
    if not m:
        return None
    out = {}
    for k, v in re.findall(r"(\w+)=([-+\d.naN]+)", m.group(1)):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    ci = re.search(r"ci=\[([-+\d.]+),([-+\d.]+)\]", m.group(1))
    if ci:
        out["ci_lo"], out["ci_hi"] = float(ci.group(1)), float(ci.group(2))
    return out


def collect(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = parse(f)
        if d and "step" in d:
            d["_file"] = os.path.basename(f)
            rows.append(d)
    # a rung rescored later (e.g. 2500 without CIs, then with) keeps the RICHER record
    best = {}
    for d in rows:
        s = int(d["step"])
        if s not in best or ("ci_lo" in d and "ci_lo" not in best[s]):
            best[s] = d
    return [best[s] for s in sorted(best)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=str(paths.experiment("reach_vit_v1_ladder*.txt")))
    ap.add_argument("--out", default=paths.figure("reach_vit_v1_ladder.png"))
    args = ap.parse_args()

    rows = collect(args.glob)
    if not rows:
        raise SystemExit(f"no VERDICT lines under {args.glob}")
    null = next((r for r in rows if int(r["step"]) == 0), None)
    tr = [r for r in rows if int(r["step"]) > 0]
    if not tr:
        raise SystemExit("no trained rungs yet -- nothing to plot")
    step = np.array([r["step"] for r in tr])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("reach_vit_v1 -- does the material ratchet EMERGE with training?", fontsize=13)

    # ---- 1. the differential: THE claim -------------------------------------------------------
    a = ax[0][0]
    y = np.array([r.get("capture_minus_quiet", np.nan) for r in tr])
    lo = np.array([r.get("ci_lo", np.nan) for r in tr])
    hi = np.array([r.get("ci_hi", np.nan) for r in tr])
    a.plot(step, y, "o-", color="#1a7f5a", lw=2, label="trained")
    ok = ~np.isnan(lo)
    if ok.any():
        a.fill_between(step[ok], lo[ok], hi[ok], color="#1a7f5a", alpha=0.2, label="95% CI (pairs)")
    if null:
        a.axhline(null["capture_minus_quiet"], color="#b03a2e", ls="--", label="random-init null")
        if "ci_lo" in null:
            a.axhspan(null["ci_lo"], null["ci_hi"], color="#b03a2e", alpha=0.15)
    a.axhline(0, color="k", lw=0.6)
    a.set_title("capture - quiet  (uniform repulsion CANNOT produce this)")
    a.set_xlabel("training step"); a.set_ylabel("d(rev)/d(fwd) difference"); a.legend(fontsize=8)

    # ---- 2. paired ratchet, both arms ---------------------------------------------------------
    a = ax[0][1]
    for key, c, lab in (("paired_ratchet_region", "#2e5f9e", "region arm"),
                        ("paired_ratchet_iqe", "#7d3c98", "IQE arm")):
        a.plot(step, [r.get(key, np.nan) for r in tr], "o-", color=c, label=lab)
        if null:
            a.axhline(null.get(key, np.nan), color=c, ls=":", alpha=0.7)
    a.axhline(0.5, color="k", lw=0.8, label="0.500 = source-blind floor")
    a.set_title("paired ratchet  (dotted = that arm's null)")
    a.set_xlabel("training step"); a.set_ylabel("P(plausible > impossible)"); a.legend(fontsize=8)

    # ---- 3. does it learn reachability at all -------------------------------------------------
    a = ax[1][0]
    a.plot(step, [r.get("sep_auc", np.nan) for r in tr], "o-", color="#1a7f5a")
    if null:
        a.axhline(null.get("sep_auc", np.nan), color="#b03a2e", ls="--", label="null")
    a.axhline(0.5, color="k", lw=0.6)
    a.set_title("observed-reachable vs cross-game AUC  (sanity channel)")
    a.set_xlabel("training step"); a.set_ylabel("AUC"); a.legend(fontsize=8)

    # ---- 4. the NEGATIVE control ---------------------------------------------------------------
    a = ax[1][1]
    a.plot(step, [r.get("confounded_ratchet", np.nan) for r in tr], "o-", color="#888")
    if null:
        a.axhline(null.get("confounded_ratchet", np.nan), color="#b03a2e", ls="--", label="null")
    a.set_title("CONFOUNDED matched-|delta| (retracted; must stay flat)")
    a.set_xlabel("training step"); a.set_ylabel("AUC"); a.legend(fontsize=8)

    fig.text(0.5, 0.005,
             "SINGLE SEED. CIs are bootstraps over PAIRS, not over training runs -- they do not "
             "measure seed variance. Null is NOT zero, so the effect is trained MINUS null.",
             ha="center", fontsize=9, color="#b03a2e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(args.out, dpi=140)
    print(f"[ladder] {len(tr)} trained rungs + {'null' if null else 'NO NULL'} -> {args.out}")
    print(f"  {'step':>7} {'cap-quiet':>10} {'CI':>20} {'ratchet_rgn':>12} {'ratchet_iqe':>12} {'sep_auc':>8}")
    if null:
        print(f"  {'NULL':>7} {null.get('capture_minus_quiet', float('nan')):>10.3f} "
              f"{'[%+.3f,%+.3f]' % (null['ci_lo'], null['ci_hi']) if 'ci_lo' in null else '':>20} "
              f"{null.get('paired_ratchet_region', float('nan')):>12.4f} "
              f"{null.get('paired_ratchet_iqe', float('nan')):>12.4f} "
              f"{null.get('sep_auc', float('nan')):>8.3f}")
    for r in tr:
        ci = "[%+.3f,%+.3f]" % (r["ci_lo"], r["ci_hi"]) if "ci_lo" in r else ""
        print(f"  {int(r['step']):>7} {r.get('capture_minus_quiet', float('nan')):>10.3f} {ci:>20} "
              f"{r.get('paired_ratchet_region', float('nan')):>12.4f} "
              f"{r.get('paired_ratchet_iqe', float('nan')):>12.4f} "
              f"{r.get('sep_auc', float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
