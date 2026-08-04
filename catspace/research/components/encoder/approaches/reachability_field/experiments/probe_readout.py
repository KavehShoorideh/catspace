#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/probe_readout.py -- shared linear + MLP(+control-task) readout
evaluation, used by hanging_piece_probe.py and hanging_piece_probe_lc0.py.

The MLP half follows Hewitt & Liang 2019 ("Designing and Interpreting Probes with
Control Tasks", control tasks aclanthology.org/D19-1275): a high-capacity nonlinear
probe can memorize spurious feature-to-label mappings, so a positive MLP result on
its own is not trustworthy -- it must be checked against a CONTROL TASK where labels
are independently randomized (breaking any true relationship) but the probe is given
the same capacity and features. If the MLP also beats chance on the control task, its
capacity alone is producing spurious lift and the real-task result can't be trusted;
"selectivity" = auc_real - auc_control is the number that matters, not auc_real alone.
McGrath et al. 2022 (PNAS, AlphaZero concept probing) used sparse LINEAR probes
specifically to sidestep this issue -- linear is reported alongside MLP here for that
reason, not as a lesser alternative.
"""
from __future__ import annotations

import numpy as np


def _boot_ci_auc(y, p, rng, n_boot=2000):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    idx_pool = np.arange(len(y)); vals = []
    from sklearn.metrics import roc_auc_score
    for _ in range(n_boot):
        idx = rng.choice(idx_pool, len(idx_pool), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def linear_readout(feats, labels, tr, te, rng, n_boot=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(feats[tr], labels[tr])
    p_te = clf.predict_proba(feats[te])[:, 1]
    acc = clf.score(feats[te], labels[te])
    auc = roc_auc_score(labels[te], p_te) if len(np.unique(labels[te])) > 1 else float("nan")
    lo, hi = _boot_ci_auc(labels[te], p_te, rng, n_boot)
    return {"clf": clf, "acc": acc, "auc": auc, "lo": lo, "hi": hi, "p_te": p_te}


def mlp_readout_with_control(feats, labels, tr, te, rng, hidden=(128, 32), n_boot=2000,
                              seed=0, alpha=1e-3, max_iter=5000):
    """Real-task MLP AUC + a Hewitt&Liang control-task AUC (independently randomized
    labels, same features/capacity/split). selectivity = auc_real - auc_control;
    only trust auc_real if selectivity is clearly positive (control stays near 0.5).
    `hidden` may be an int (one layer) or a tuple (multi-layer)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    hidden_sizes = (hidden,) if isinstance(hidden, int) else tuple(hidden)

    def fit_eval(y_tr, y_te, rs):
        clf = MLPClassifier(hidden_layer_sizes=hidden_sizes, max_iter=max_iter, random_state=rs,
                             early_stopping=True, alpha=alpha)
        clf.fit(feats[tr], y_tr)
        p = clf.predict_proba(feats[te])[:, 1]
        auc = roc_auc_score(y_te, p) if len(np.unique(y_te)) > 1 else float("nan")
        return auc, p

    auc_real, p_real = fit_eval(labels[tr], labels[te], seed)
    lo_real, hi_real = _boot_ci_auc(labels[te], p_real, rng, n_boot)

    # control task: independently randomized labels (breaks any true mapping),
    # same marginal class balance, same features, same split, same probe capacity.
    p1 = labels[tr].mean()
    rand_tr = (rng.random(len(labels[tr])) < p1).astype(int)
    p1_te = labels[te].mean()
    rand_te = (rng.random(len(labels[te])) < p1_te).astype(int)
    auc_ctrl, p_ctrl = fit_eval(rand_tr, rand_te, seed + 1)
    lo_ctrl, hi_ctrl = _boot_ci_auc(rand_te, p_ctrl, rng, n_boot)

    selectivity = auc_real - auc_ctrl if not (np.isnan(auc_real) or np.isnan(auc_ctrl)) else float("nan")
    return {"auc_real": auc_real, "lo_real": lo_real, "hi_real": hi_real,
            "auc_ctrl": auc_ctrl, "lo_ctrl": lo_ctrl, "hi_ctrl": hi_ctrl,
            "selectivity": selectivity}
