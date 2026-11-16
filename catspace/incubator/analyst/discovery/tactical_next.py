"""PLAN 78.62 F: THE SIMPLEST SEQUENCE MODEL, and what it adds over C. The next state from the present state and the parts: on the rows of stage C, the same two targets (a unit of theirs lost by the estimate
with us to move within 8 plies; a unit of ours with them to move) from the parts of C (K+S+E+R) PLUS the present tactical state of the row (per side, ours and theirs: units read, forced safe, safe by the estimate,
threatened, trapped, immobile, resting on a bound, lost by the estimate), the same fit and the same test games. The simplest first: one step of state, no history, a linear head. The gain over C with a 90 pct
band over games; direction only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.tactical_next --state-dir D
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import tactical_chances as TC
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.incubator.analyst.discovery import tactical_tables as TT
from catspace.io import paths

FIELDS = ("units_read", "safe_firm", "safe_est", "threat", "trapped", "immobile", "lost_bound", "lost_est")


def present_state(d: dict, t: dict) -> np.ndarray:
    tb, st = d["tb"], d["st"]
    i = t["state_row"]
    c = st["mover"].astype(int)[i]
    cols = []
    for f in FIELDS:
        cols.append(tb[f][i, c].astype(float))
        cols.append(tb[f][i, 1 - c].astype(float))
    return np.stack(cols, axis=1)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-dir", default=paths.experiment("tactical_78_62", "state"))
    ap.add_argument("--heads-dir", default=paths.experiment(*TS.HEADS_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--draws", type=int, default=TC.DRAWS)
    a = ap.parse_args(argv)
    out = Path(a.out_dir or paths.experiment("tactical_78_62", "next"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS tactical next: {msg}")
        print(lines[-1], flush=True)

    d = TT.load_all(Path(a.state_dir))
    t = TC.target_rows(d, Path(a.heads_dir))
    fit, test = np.flatnonzero(t["fit"]), np.flatnonzero(~t["fit"])
    base_x = TC.sets_of(t)["K+S+E+R"]
    ps = present_state(d, t)
    say(f"rows {len(t['game'])}: fit {len(fit)}, test {len(test)}; present-state columns {ps.shape[1]} (" + ", ".join(FIELDS) + ", each ours then theirs)")
    for tag, key in (("us", "y_us"), ("them", "y_them")):
        y = t[key].astype(float)
        res = {}
        for nm, x in (("C: K+S+E+R", base_x), ("C + present state", np.hstack([base_x, ps])), ("present state alone", ps)):
            m = TC.fit_logit(x[fit], y[fit], t["game"][fit])
            res[nm] = TC.row_loss(TC.predict_logit(m, x[test]), y[test])
            v, lo, hi = TC.mean_band(t["game"][test], res[nm], a.draws, 6265)
            say(f"[{tag}] test log loss, {nm}: {v:.5f} [{lo:.5f}, {hi:.5f}]; columns kept {int((m['coef'] != 0).sum())} of {len(m['coef'])}")
        g, lo, hi = TC.paired_gain(t["game"][test], res["C: K+S+E+R"], res["C + present state"], a.draws, 6265)
        say(f"[{tag}] test log loss gain of adding the present state to C: {g:+.5f} [{lo:+.5f}, {hi:+.5f}] (positive: with it is better)")
    (out / "next_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
