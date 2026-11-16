"""PLAN 78.58 B (amendment): THE ESTIMATOR RECORD AS NAMED HEADS over the same inputs. Reads the record of estimator_fit.py and writes `estimator_heads.json`:
  head "result": the probability of win, draw and loss for the side to move (kinds model; and kinds + concepts, with how each kind's term moved against the baseline);
  head "kind_baseline": the fitted terms of the kinds alone (the order of kinds), kept as its own head so a later fit can be read as a correction to it;
  head "arrival": the 78.49 target at h = 8 (a CONFIRMED threat move of ours arrives within h plies, less theirs: forcing_arrival_measure.arrival_targets), fitted on the kinds (ours less theirs) at the quiet rows of the
  forcing reads of the 500 games (the rows where the reads exist), a linear fit (subgoal_fit_measure.fit_sparse), fit half = even game ids, test half = odd. The concepts are NOT inputs of this head (the count on those rows was not made): stated in the head.
Interface: record -> head name -> inputs -> a probability or an expected value, with source, data and date per head (`head_predict`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.estimator_heads --record R --out-dir O
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import estimator_fit as EF
from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SF
from catspace.io import paths

H = 8


def head_predict(rec: dict, head: str, x: np.ndarray) -> np.ndarray:
    """record -> head name -> inputs (ours less theirs by kind, in the record's kind order) -> a probability (3 columns) or an expected value (1 column)."""
    h = rec["heads"][head]
    if h["type"] == "probability":
        return EF.predict_record(h["model"], x)
    if h["type"] == "expected_value":
        if h.get("model") is None:
            raise ValueError(f"head {head} is not fitted: {h.get('note')}")
        m = h["model"]
        return (((x - np.asarray(m["mean"])) / np.asarray(m["scale"])) @ np.asarray(m["coef"]) + m["intercept"])[:, None]
    if h["type"] == "terms":
        return np.asarray(h["terms"])
    raise ValueError(head)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    rec = json.loads(Path(a.record).read_text())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS estimator heads: {m}")
        print(lines[-1], flush=True)

    kinds = rec["kinds"]
    km = rec["kinds_model"]
    base_terms = EF.kind_terms(km["classes"], np.asarray(km["coef"]))
    heads = {
        "result": {"type": "probability", "inputs": f"ours less theirs by kind {kinds}", "model": km, "source": rec["source"], "data": rec["data"], "date": rec["date"],
                   "concepts_model": rec["concepts_model"], "concepts_note": "the model with the top distinct concepts of 78.54 added (ours less theirs, columns scaled by concept_scale), fitted on every 4th quiet row"},
        "kind_baseline": {"type": "terms", "inputs": "none (a term per kind)", "kinds": kinds, "terms": [float(v) for v in base_terms], "order_of_kinds": rec["order_of_kinds"], "source": rec["source"], "data": rec["data"], "date": rec["date"]},
    }
    cm = rec["concepts_model"]
    if cm:
        ck = EF.kind_terms(cm["classes"], np.asarray(cm["coef"])[:, :len(kinds)])
        for k, name in enumerate(kinds):
            say(f"result head with concepts added: term of {name} {ck[k]:+.4f} against the baseline {base_terms[k]:+.4f} (moved {ck[k] - base_terms[k]:+.4f})")
        heads["result"]["kind_terms_with_concepts"] = [float(v) for v in ck]
    # ---- the arrival head
    t1 = time.perf_counter()
    gdir = Path(paths.experiment("opportunity_78_44", "games_s5016_n500"))
    reads_path = Path(paths.experiment("forcing_opportunity_78_48", "games_s5016_n500", "reads_fuse5000.jsonl"))
    records = FO.read_games(str(gdir))
    reads = {}
    with open(reads_path) as f:
        for line in f:
            g = json.loads(line)
            reads[g["game"]] = g["plies"]
    model = GL.model(EF.GAME, EF.VARIANT)
    goal = rec["goal_kind"]
    names, raw = model.position_table([(r.get("start"), list(r["moves"])) for r in records], a.threads)
    nk = len(names)
    flat = np.frombuffer(raw, dtype=np.int16).astype(int)
    w = 2 + 2 * nk
    use = [names.index(k) for k in kinds]
    xs, ys, gid, off = [], [], [], 0
    hi = FA.HORIZONS.index(H)
    for r in records:
        n = len(r["moves"]) + 1
        tab = flat[off:off + n * w].reshape(n, w)
        off += n * w
        pl = reads[r["game"]]
        own, oth = FA.counts_of_game(pl)
        for t in FA.quiet_indices(pl, own, oth):
            tg = FA.arrival_targets(own, len(pl), t)[hi]
            if tg is None or tab[t, 0] != 1:
                continue
            xs.append([tab[t, 2 + k] - tab[t, 2 + nk + k] for k in use]); ys.append(tg[0] - tg[1]); gid.append(r["game"])
    x, y, gid = np.array(xs, float), np.array(ys, float), np.array(gid)
    fit = gid % 2 == 0
    say(f"arrival head h={H}: rows with the forcing reads (quiet) {len(y)} of {len(records)} games; fit half (even ids) {int(fit.sum())}, test half {int((~fit).sum())}; mean target {y.mean():+.4f}")
    m = SF.fit_sparse(x[fit], y[fit], gid[fit])
    pred = SF.predict(m, x[~fit])
    ss = ((y[~fit] - y[~fit].mean()) ** 2).sum()
    r2 = 1 - ((y[~fit] - pred) ** 2).sum() / ss if ss > 0 else float("nan")
    corr = float(np.corrcoef(pred, y[~fit])[0, 1]) if pred.std() > 0 else float("nan")
    say(f"arrival head h={H}: fitted on kinds only (the concepts were not counted on these rows), penalty {m['alpha']:.5f}; test half R squared {r2:+.4f}, correlation {corr:+.4f} ({time.perf_counter() - t1:.1f} s)")
    heads["arrival"] = {"type": "expected_value", "definition": "plan 78.49 target at h=8: a CONFIRMED threat move of ours arrives within 8 plies, less theirs (forcing_arrival_measure.arrival_targets)",
                        "inputs": f"ours less theirs by kind {kinds} (the concepts are not inputs of this head: not counted on these rows)",
                        "model": {"mean": m["mean"].tolist(), "scale": m["scale"].tolist(), "coef": m["coef"].tolist(), "intercept": m["intercept"], "alpha": m["alpha"]},
                        "source": "self-play only (the 500 games of opportunity_78_44 and their forcing reads of forcing_opportunity_78_48)", "data": {"rows": int(len(y)), "fit_rows": int(fit.sum()), "test_rows": int((~fit).sum()), "split": "even/odd game id"},
                        "test_r2": float(r2), "date": datetime.date.today().isoformat()}
    rec["heads"] = heads
    (out / "estimator_heads.json").write_text(json.dumps(rec, indent=1))
    say(f"record with heads {list(heads)} written to {out / 'estimator_heads.json'} ({time.perf_counter() - t0:.1f} s)")
    (out / "heads_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
