"""PLAN 78.65 B, C, D, E, F: THE EVENT HEADS OF 78.64 GIVEN THE FRAMES. A measurement in a sandbox; nothing is wired into any player, nothing is written to rules/learned or data/derived. Game-free: the
events, sides, kinds, classes and bases are words read at run time (events_rules.py, the state rows of frames_state_rows.py); none is named here. No engine evaluation, no labeller, no external model: what
happened on the board and the result of the games. No assumed worth: the only order of kinds is the human record of 78.59 inside the frame read (said in frames_state_rows).

B. THE INPUTS FROM THE FRAMES, per side AS SEEN BY THE MOVER (`us` is the side to move at the row, `them` the other; a row with the sides swapped has the two blocks swapped): the counts of the units the frame
read holds, by class and by what the class rests on; the counts by trap word; the counts of the units that stand lost (by the estimate) by kind; the count of frames, the units read and their size per frame.
The fate of a unit within the old window is NEVER an input (it is the row's future). A column that is nonzero on fewer than MIN_ROWS rows of the FIT games is dropped, and the counts are printed.
C. THE LINEAR HEADS of 78.64 (events_hazard: a discrete hazard, no horizon) with the inputs added in STEPS, a learning curve at each step (an eighth, a quarter, a half, all of the fit games), the gain over the
base hazard fitted on the same games, with a 90 pct band over test games. On the thinned rows (those that have the 64 concepts) the concepts are a step of their own.
D. A HEAD THAT CAN USE PAIRS: gradient-boosted trees (scikit-learn, HistGradientBoostingClassifier, imported) on the SAME hazard records (one record per (row, distance bin, flag) with the count of its steps
as the weight, and the event's record with weight 1), with its learning curve, read on the SAME test rows as the linear head by the same likelihood. The pairs it leans on (Friedman's H on the strongest inputs)
are printed in words.
E. THE READING THAT MATTERS FOR A LEAF, at settled rows (every row of the hazard is one): the chance that each side next wins and keeps a unit (a derived reading of the rules' events), who first, the
calibration, by band of rating.
F. THE ENDGAME FILL of 78.64 in the fit, with and without, for the test rows with few units and for all test rows; the fill games' frame rows are made too (frames_state_rows fill).

  .venv/bin/python -m catspace.incubator.analyst.discovery.frames_heads run  --state-dir D --out-dir O [--tree-heads LIST] [--no-curves]
  .venv/bin/python -m catspace.incubator.analyst.discovery.frames_heads fill --state-dir D --out-dir O
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import events_fill as EF
from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.incubator.analyst.discovery import events_rules as ER
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.io import paths

MIN_ROWS = 30                                                       # a column nonzero on fewer fit rows than this is dropped (the same count as the thin sets of 78.64)
STATE = ("frames_heads_78_65", "state_rows", "human")
MAX_ITER = 200                                                      # the most boosting rounds tried; the count used is chosen on a fifth of the fit games held out inside the fit games
TREE = dict(learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0)
NB, SEED = EH.NB, EH.SEED
EPS = 1e-9


# ------------------------------------------------------------------------------------------------------------------------------ B: the inputs from the frames

def frame_blocks(st: dict, nkind: int, kinds: list) -> "dict[str, tuple[np.ndarray, list]]":
    """The blocks of frame inputs of the rows of a state dict (rows by game and ply; units: owner, kind, class, basis, trap). `us` is the owner flag 1 (the side to move at the row), `them` is 0."""
    n = len(st["ply"])
    rid = np.repeat(np.arange(n), np.diff(st["uoff"]))
    u = st["units"].astype(np.int64)
    side = 1 - u[:, 0]                                              # 0: us (the mover's units), 1: them
    ncls, nbas, ntrap = len(TS.CLASS_WORDS), len(TS.BASIS_WORDS), len(TS.TRAP_WORDS)
    sides = ("us", "them")

    def counts(width: int, col: np.ndarray, pick: np.ndarray) -> np.ndarray:
        a = np.zeros((n, 2, width), np.float32)
        np.add.at(a, (rid[pick], side[pick], col[pick]), 1.0)
        return a
    allp = np.ones(len(u), bool)
    basis = np.where(u[:, 3] < nbas, u[:, 3], nbas)                 # a class that rests on nothing (the engine's 255) is the last basis column
    cb = counts(ncls * (nbas + 1), u[:, 2] * (nbas + 1) + basis, allp)
    tr = counts(ntrap, u[:, 4], u[:, 4] > 0)                        # trap word 0 is `free`: the count of free units is the units read less these
    lost = ((u[:, 2] == TS.CLASS_WORDS.index("lost")) | (u[:, 2] == TS.CLASS_WORDS.index("both"))) & (u[:, 3] == TS.BASIS_WORDS.index("estimate"))
    lk = counts(nkind, u[:, 1], lost)
    read = counts(1, np.zeros(len(u), np.int64), allp)[:, :, 0]
    bw = TS.BASIS_WORDS + ["nothing"]
    nm_cb = [f"{s}: {TS.CLASS_WORDS[c]} resting on {bw[b]}" for s in sides for c in range(ncls) for b in range(nbas + 1)]
    nm_tr = [f"{s}: trap word {TS.TRAP_WORDS[t]}" for s in sides for t in range(ntrap)]
    nm_lk = [f"{s}: lost by the estimate, kind {kinds[k]}" for s in sides for k in range(nkind)]
    frames = st["frames"].astype(np.float32)
    size = (read.sum(axis=1) / np.maximum(frames, 1.0)).astype(np.float32)
    fr = np.column_stack([frames, st["unread"].astype(np.float32), read[:, 0], read[:, 1], size])
    nm_fr = ["frames at the row", "lines unread", "us: units read", "them: units read", "units read per frame"]
    return {"classes": (cb.reshape(n, -1), nm_cb), "trap": (tr.reshape(n, -1), nm_tr), "lost kinds": (lk.reshape(n, -1), nm_lk), "frames": (fr, nm_fr)}


def drop_sparse(X: np.ndarray, names: list, fit_mask: np.ndarray, min_rows: int = MIN_ROWS) -> "tuple[np.ndarray, list, list]":
    """Keep the columns that are nonzero on at least min_rows of the FIT rows; returns the kept matrix, the kept names and the dropped (name, rows) pairs."""
    cnt = (X[fit_mask] != 0).sum(axis=0)
    keep = cnt >= min_rows
    return X[:, keep], [n for n, k in zip(names, keep) if k], [(n, int(c)) for n, c, k in zip(names, cnt, keep) if not k]


def align_state(rows: dict, sample_ids: list, st: dict, state_ids: list) -> np.ndarray:
    """For each hazard row the index of its row in the state dict (a row of the game `sample_ids[g]` at the ply `ply`); every row must have one."""
    slot = {i: k for k, i in enumerate(state_ids)}
    want = np.array([slot[sample_ids[int(g)]] for g in rows["g"]], np.int64) * 1000 + rows["ply"].astype(np.int64)
    have = st["game"].astype(np.int64) * 1000 + st["ply"].astype(np.int64)
    order = np.argsort(have, kind="stable")
    pos = np.searchsorted(have[order], want)
    pos = np.minimum(pos, len(have) - 1)
    ix = order[pos]
    assert (have[ix] == want).all(), "a hazard row has no state row"
    return ix


# ------------------------------------------------------------------------------------------------------------------------------ the likelihood from a table of chances

def table_linear(m: "dict | None", X: "np.ndarray | None", sel: np.ndarray, base_b: "np.ndarray | None" = None) -> np.ndarray:
    """P[row, bin, flag]: the chance of the event at a step of that bin and flag, of the linear head `m` (None: the base hazard `base_b`) on the rows sel."""
    if m is None:
        eta0, b = np.zeros(len(sel)), base_b.astype(np.float64)
    else:
        eta0, b = (((X[sel] - m["mean"]) / m["scale"]) @ m["w"]).astype(np.float64), m["b"].astype(np.float64)
    return 1.0 / (1.0 + np.exp(-(eta0[:, None, None] + b[None])))


def nll_table(h: EH.Head, P: np.ndarray, sel: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """(per row nll, per row and bin nll) of the chances P (rows of sel) on the head's records: the non-event steps of each bin and flag in the risk set, and the event's step."""
    P = np.clip(P, EPS, 1 - EPS)
    non = h.Nf[sel].astype(np.float64) * (-np.log1p(-P))
    per = non.sum(axis=2)
    i = np.arange(len(sel))
    w = h.has[sel]
    evP = P[i, h.evbin[sel], h.evflag[sel]]
    per[i, h.evbin[sel]] += np.where(w, -np.log(evP), 0.0)
    return per.sum(axis=1), per


# ------------------------------------------------------------------------------------------------------------------------------ D: the tree head on the same records

def records(h: EH.Head, X: np.ndarray, sel: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The head's hazard records on the rows sel: features (the row's inputs, the distance bin, the flag), label, weight (the steps of the bin and flag; the event's record has weight one)."""
    Nf = h.Nf[sel]
    r, j, f = np.nonzero(Nf > 0)
    w0 = Nf[r, j, f]
    ev = np.flatnonzero(h.has[sel])
    rr = np.concatenate([r, ev])
    jj = np.concatenate([j, h.evbin[sel][ev]])
    ff = np.concatenate([f, h.evflag[sel][ev]])
    y = np.concatenate([np.zeros(len(r)), np.ones(len(ev))])
    w = np.concatenate([w0, np.ones(len(ev))])
    Xs = X[sel]
    feats = np.empty((len(rr), Xs.shape[1] + 2), np.float32)               # filled in place: the records are millions of rows at the full sample
    feats[:, :-2] = Xs[rr]
    feats[:, -2], feats[:, -1] = jj, ff
    return feats, y.astype(np.float32), w.astype(np.float32)


def weighted_loss(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log1p(-p))).sum())


def fit_tree(h: EH.Head, X: np.ndarray, rows: dict, fit_sel: np.ndarray, n_iter: "int | None", max_iter: int = MAX_ITER, additive_distance: bool = False) -> dict:
    """Boosted trees on the records of fit_sel. The number of rounds is chosen on a fifth of the fit games held out inside the fit games (unless given), then the trees are fitted on all of fit_sel.
    `additive_distance` keeps the distance bin and the flag out of every split's company with the inputs (interaction constraints: two groups), so the trees can still make pairs OF INPUTS but the
    distance acts on the log odds additively, as it does in the linear head."""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    kw = dict(TREE)
    if additive_distance:
        nf = X.shape[1]
        kw["interaction_cst"] = [list(range(nf)), [nf, nf + 1]]
    if n_iter is None:
        gs = np.unique(rows["g"][fit_sel])
        rng = np.random.default_rng(SEED)
        hold = set(rng.choice(gs, max(1, len(gs) // 5), replace=False).tolist())
        is_hold = np.array([x in hold for x in rows["g"][fit_sel]])
        tr, ho = fit_sel[~is_hold], fit_sel[is_hold]
        Ft, yt, wt = records(h, X, tr)
        Fh, yh, wh = records(h, X, ho)
        clf = HGB(max_iter=max_iter, early_stopping=False, random_state=0, **kw)
        clf.fit(Ft, yt, sample_weight=wt)
        losses = [weighted_loss(p[:, 1], yh, wh) for p in clf.staged_predict_proba(Fh)]
        n_iter = int(np.argmin(losses)) + 1
    Fa, ya, wa = records(h, X, fit_sel)
    clf = HGB(max_iter=n_iter, early_stopping=False, random_state=0, **kw)
    clf.fit(Fa, ya, sample_weight=wa)
    return {"clf": clf, "n_iter": n_iter, "capped": n_iter >= max_iter}


def table_tree(m: dict, X: np.ndarray, sel: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """P[row, bin, flag] of the tree head on the rows sel (every bin and flag, so the race and the chance can be read to any distance)."""
    out = np.zeros((len(sel), NB, 2), np.float64)
    jj, ff = np.meshgrid(np.arange(NB), np.arange(2), indexing="ij")
    jj, ff = jj.ravel().astype(np.float32), ff.ravel().astype(np.float32)
    for a in range(0, len(sel), chunk):
        s = sel[a:a + chunk]
        x = X[s]
        F = np.hstack([np.repeat(x, NB * 2, axis=0), np.tile(jj, len(s))[:, None], np.tile(ff, len(s))[:, None]]).astype(np.float32)
        out[a:a + len(s)] = m["clf"].predict_proba(F)[:, 1].reshape(len(s), NB, 2)
    return out


def curve_tree(h: EH.Head, name: str, sname: str, X: np.ndarray, rows: dict, fit_sel: np.ndarray, test_sel: np.ndarray, perm: np.ndarray, say, draws: int, curves: bool):
    """The learning curve of the tree head, as EH.curve does for the linear one: fits on an eighth, a quarter, a half and all of the fit games (a fixed random order), the rounds chosen on all and kept for
    the smaller fits, each read on the test rows as the gain in log loss per step over the base hazard fitted on the same games, with a 90 pct band over test games. Returns the full fit and its test numbers."""
    den = h.steps()[test_sel].astype(np.float64)
    full, n_iter, said = None, None, []
    for fr in (reversed(EH.FRACTIONS) if curves else (1.0,)):
        k = max(1, int(round(len(perm) * fr)))
        chosen = set(perm[:k].tolist())
        sub = fit_sel[np.array([x in chosen for x in rows["g"][fit_sel]])]
        m = fit_tree(h, X, rows, sub, n_iter if fr < 1.0 else None)
        if fr == 1.0:
            n_iter = m["n_iter"]
        P = table_tree(m, X, test_sel)
        tot_h, per_h = nll_table(h, P, test_sel)
        tot_bs, per_bs = nll_table(h, table_linear(None, None, test_sel, h.base_logit(sub)), test_sel)
        g, glo, ghi = EH.boot(rows["g"][test_sel], tot_bs - tot_h, den, draws, SEED)
        said.append(f"HEAD {name} set {sname} [TREES]: fit games {k} (fit rows {len(sub)}): test log loss gain over the base hazard (same fit games) per step {g:+.5f} [{glo:+.5f}, {ghi:+.5f}]; rounds {m['n_iter']}")
        if fr == 1.0:
            full = (m, tot_h, per_h, tot_bs, per_bs, P)
    for line in reversed(said):
        say(line)
    return full


def friedman_pairs(m: dict, F: np.ndarray, names: list, top: int, n_pts: int = 80, seed: int = 0) -> list:
    """Friedman's H statistic for the pairs among the `top` inputs the trees lean on most (permutation loss on a sample of the records is not needed: the inputs are ranked by the spread of the model's
    partial dependence on each), read on a sample of the test records. Returns [(H squared, name a, name b)] largest first."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(F), min(n_pts, len(F)), replace=False)
    bg = F[rng.choice(len(F), min(n_pts, len(F)), replace=False)]
    ev = F[pick]
    clf = m["clf"]

    def f(Z):
        return clf.decision_function(Z)

    def pd1(i):
        out = np.zeros(len(ev))
        for a, x in enumerate(ev):
            Z = bg.copy()
            Z[:, i] = x[i]
            out[a] = f(Z).mean()
        return out
    single = {i: pd1(i) for i in range(F.shape[1])}
    cols = [i for _s, i in sorted(((float(np.std(v)), i) for i, v in single.items()), reverse=True)[:top]]
    pd_single = {i: single[i] - single[i].mean() for i in cols}
    out = []
    for ai, i in enumerate(cols):
        for j in cols[ai + 1:]:
            pdij = np.zeros(len(ev))
            for a, x in enumerate(ev):
                Z = bg.copy()
                Z[:, i], Z[:, j] = x[i], x[j]
                pdij[a] = f(Z).mean()
            pdij = pdij - pdij.mean()
            den = float((pdij ** 2).sum())
            out.append((float(((pdij - pd_single[i] - pd_single[j]) ** 2).sum() / den) if den > 0 else 0.0, names[i], names[j]))
    return sorted(out, reverse=True)


# ------------------------------------------------------------------------------------------------------------------------------ E: the reading for a leaf

def side_chance(h: EH.Head, P: np.ndarray, remaining: np.ndarray) -> np.ndarray:
    """The chance that the side has an event of the class in the steps still left in the record (one minus the product of the survivals over its active flags)."""
    surv = np.ones(len(P))
    for d in range(1, int(remaining.max()) + 1):
        live = remaining >= d
        f = 0 if (d % 2 == 1) == (h.rel == "us") else 1
        if h.active[f]:
            surv = np.where(live, surv * (1 - np.clip(P[:, int(EH.bin_of_d(d)), f], EPS, 1 - EPS)), surv)
    return 1 - surv


def race_table(hu: EH.Head, ht: EH.Head, Pu: np.ndarray, Pt: np.ndarray, remaining: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """(P(mover first), P(other first)) from the two heads' chance tables; the odd steps are the mover's hazard (flag 0 of `us`), the even steps the other's (flag 0 of `them`)."""
    surv = np.ones(len(Pu))
    pu, pt = np.zeros(len(Pu)), np.zeros(len(Pu))
    for d in range(1, int(remaining.max()) + 1):
        live = remaining >= d
        j = int(EH.bin_of_d(d))
        if d % 2 == 1:
            if not hu.active[0]:
                continue
            hz = np.clip(Pu[:, j, 0], EPS, 1 - EPS)
            pu += np.where(live, surv * hz, 0.0)
        else:
            if not ht.active[0]:
                continue
            hz = np.clip(Pt[:, j, 0], EPS, 1 - EPS)
            pt += np.where(live, surv * hz, 0.0)
        surv = np.where(live, surv * (1 - hz), surv)
    return pu, pt


def calib_line(p: np.ndarray, y: np.ndarray, bins: int = 10) -> str:
    order = np.argsort(p, kind="stable")
    return " ".join(f"{p[b].mean():.3f}>{y[b].mean():.3f}" for b in np.array_split(order, bins) if len(b))


def leaf_reading(core: tuple, hs: dict, Ps: dict, base_P: dict, rows: dict, test: np.ndarray, label: str, say, draws: int) -> None:
    """Stage E for one class with both sides: on the test rows where the record is not cut for the class, (1) the chance of each side's event within the record against what happened (log loss gain over the base
    hazard's chance, calibration), (2) who first (accuracy, gain, calibration), (3) both by band of rating. `Ps` maps rel to the table of chances on `test`, `base_P` the base hazard's."""
    hu, ht = hs[(core, "us")], hs[(core, "them")]
    remaining = (rows["T"] - rows["ply"])[test]
    cutany = hu.cut[test] | ht.cut[test]
    ok = ~cutany
    name = " | ".join(core)
    for rel, hd in (("us", hu), ("them", ht)):
        p = side_chance(hd, Ps[rel], remaining)
        p0 = side_chance(hd, base_P[rel], remaining)
        y = hd.has[test].astype(float)
        s = np.flatnonzero(ok)
        one = np.ones(len(s))

        def ll(q):
            q = np.clip(q, 1e-6, 1 - 1e-6)
            return -(y[s] * np.log(q[s]) + (1 - y[s]) * np.log(1 - q[s]))
        g, lo, hi = EH.boot(rows["g"][test][s], ll(p0) - ll(p), one, draws, SEED)
        say(f"LEAF [{label}] {name} [{rel}]: chance that the side has the event before the record ends, {len(s)} test rows (not cut; observed {y[s].mean():.4f}, predicted {p[s].mean():.4f}); log loss gain over the base hazard's chance {g:+.5f} [{lo:+.5f}, {hi:+.5f}]; calibration (predicted>observed) {calib_line(p[s], y[s])}")
    tu, tt = np.where(hu.has[test], hu.de[test], EH.INF), np.where(ht.has[test], ht.de[test], EH.INF)
    occ = ((tu < EH.INF) | (tt < EH.INF)) & ok
    s = np.flatnonzero(occ)
    if len(s) < 100:
        return
    y = (tu < tt)[s].astype(float)
    pu, pt = race_table(hu, ht, Ps["us"][s], Ps["them"][s], remaining[s])
    pk = pu / np.maximum(pu + pt, 1e-12)
    pu0, pt0 = race_table(hu, ht, base_P["us"][s], base_P["them"][s], remaining[s])
    bp = pu0 / np.maximum(pu0 + pt0, 1e-12)
    g_ = rows["g"][test][s]
    one = np.ones(len(s))

    def ll2(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))
    ah, alo, ahi = EH.boot(g_, ((pk > 0.5) == (y > 0.5)).astype(float), one, draws, SEED)
    ab, blo, bhi = EH.boot(g_, ((bp > 0.5) == (y > 0.5)).astype(float), one, draws, SEED)
    gg, glo, ghi = EH.boot(g_, ll2(bp) - ll2(pk), one, draws, SEED)
    say(f"LEAF [{label}] {name}: WHO FIRST, {len(s)} test rows where the class occurred before the record ended (the mover first in {y.mean():.4f}); accuracy head {ah:.4f} [{alo:.4f}, {ahi:.4f}], base hazard {ab:.4f} [{blo:.4f}, {bhi:.4f}]; log loss gain over the base hazard's chance {gg:+.5f} [{glo:+.5f}, {ghi:+.5f}]; calibration {calib_line(pk, y)}")
    band = rows["band"][test][s]
    for b in np.unique(band):
        t = np.flatnonzero(band == b)
        if len(t) < 100:
            continue
        acc_head = ((pk[t] > 0.5) == (y[t] > 0.5)).mean()
        acc_base = ((bp[t] > 0.5) == (y[t] > 0.5)).mean()
        gb, lob, hib = EH.boot(g_[t], ll2(bp)[t] - ll2(pk)[t], np.ones(len(t)), draws, SEED)
        say(f"LEAF [{label}] {name}: WHO FIRST by rating band {int(b)}: {len(t)} rows; accuracy head {acc_head:.4f}, base hazard {acc_base:.4f}; log loss gain {gb:+.5f} [{lob:+.5f}, {hib:+.5f}]; mean predicted {pk[t].mean():.3f}, observed {y[t].mean():.3f}")


# ------------------------------------------------------------------------------------------------------------------------------ the run

def load_state(state_dir: Path) -> "tuple[dict, list, list]":
    z = np.load(state_dir / "state_rows.npz")
    st = {k: z[k] for k in z.files}
    meta = json.loads((state_dir / "meta.json").read_text())
    return st, meta["ids"], meta["kinds"]


def build(state_dir: Path, say) -> dict:
    """The hazard rows of the games that have state rows, with the frame blocks (B) and the pruned concept columns."""
    st, state_ids, kinds = load_state(state_dir)
    full = EH.load_sample(Path(paths.experiment("events_78_64", "full12000")))
    pos = {i: k for k, i in enumerate(full["ids"])}
    sample = {"ids": list(state_ids), "res": [full["res"][pos[i]] for i in state_ids]}
    heads_dir = Path(paths.experiment(*ER.HEADS_DIR))
    allg = {json.loads(x)["id"]: json.loads(x) for x in (heads_dir / "games.jsonl").read_text().splitlines() if x.strip()}
    spec, _v = ER.spec_and_vocab()
    roles = ER.role_words(ER.rule_clauses(spec))
    EH.end_events(sample["res"], sample["ids"], allg, roles)
    rows = EH.build_rows(sample, heads_dir, None)
    ix = align_state(rows, sample["ids"], st, state_ids)
    assert (st["mover"][ix] == rows["mover"]).all(), "the mover of a state row and of a hazard row differ"
    sub = {k: st[k][ix] for k in ("ply", "mover", "frames", "unread", "game")}
    # the unit records of the picked rows, re-cut
    chunks = [st["units"][st["uoff"][j]:st["uoff"][j + 1]] for j in ix]
    sub["units"] = np.concatenate(chunks)
    sub["uoff"] = np.concatenate([[0], np.cumsum([len(c) for c in chunks])]).astype(np.int64)
    blocks = frame_blocks(sub, len(kinds), kinds)
    fit = rows["fit"]
    out = {"sample": sample, "rows": rows, "roles": roles, "allg": allg, "blocks": {}, "kinds": kinds}
    for g, (X, names) in blocks.items():
        X2, n2, dropped = drop_sparse(X, names, fit)
        out["blocks"][g] = (X2, n2)
        some = [(n, c) for n, c in dropped if c > 0]
        cnt_all = (X[fit] != 0).sum(axis=0)
        kept_c = sorted(int(c) for c in cnt_all if c >= MIN_ROWS)
        say(f"INPUTS [{g}]: {len(names)} columns, {len(n2)} kept (rows with a nonzero value in the fit rows: least {kept_c[0] if kept_c else 0}, most {kept_c[-1] if kept_c else 0}); dropped under {MIN_ROWS} fit rows: {len(dropped)}, of which {len(dropped) - len(some)} never nonzero" + ("; the others: " + "; ".join(f"{n} ({c} rows)" for n, c in some) if some else ""))
    thin = rows["cmask"]
    Cn = [str(c) for c in rows["meta"]["concepts"]]
    cnt = (rows["C"][thin & fit] != 0).sum(axis=0)
    keep = cnt >= MIN_ROWS
    out["C"] = (rows["C"][:, keep], [n for n, k in zip(Cn, keep) if k])
    say(f"INPUTS [concepts]: {len(Cn)} columns on {int(thin.sum())} thinned rows ({int((thin & fit).sum())} fit); {int(keep.sum())} kept with at least {MIN_ROWS} nonzero fit rows; rows each kept column has (min, median, max): {int(cnt[keep].min()) if keep.any() else 0}, {int(np.median(cnt[keep])) if keep.any() else 0}, {int(cnt[keep].max()) if keep.any() else 0}; dropped {int((~keep).sum())} columns with counts {sorted(cnt[~keep].tolist())}")
    return out


def steps_of(d: dict) -> "list[tuple[str, np.ndarray, list]]":
    """The input steps, each the base K+S+E+R of 78.64 with one more group (cumulative): (name, matrix, column names)."""
    rows, b = d["rows"], d["blocks"]
    base = EH.sets_of(rows)["K+S+E+R"]
    names = [f"kind {k}" for k in rows["meta"]["kinds"]] + [f"statement {s}" for s in rows["meta"]["statements"]] + [f"edge distances of {k}" for k in rows["meta"]["edge_kinds"]] + ["mean rating", "rating difference", "ply"]
    out = [("base K+S+E+R", base, names)]
    X, nm = base, names
    for g in ("classes", "trap", "lost kinds", "frames"):
        X, nm = np.hstack([X, b[g][0]]), nm + b[g][1]
        out.append((f"+{g}", X, list(nm)))
    return out


def group_importance(h: EH.Head, m: dict, X: np.ndarray, groups: dict, test: np.ndarray, rows: dict, say, name: str, tree: bool, draws: int) -> None:
    """The loss that comes back when one group of columns is shuffled across the test rows (seeded): the groups that carry the gain."""
    den = h.steps()[test].astype(np.float64)
    P0 = table_tree(m, X, test) if tree else table_linear(m, X, test)
    tot0, _p = nll_table(h, P0, test)
    rng = np.random.default_rng(SEED + 9)
    parts = []
    for gname, cols in groups.items():
        sh = rng.permutation(len(test))
        saved = X[np.ix_(test, cols)].copy()
        X[np.ix_(test, cols)] = X[np.ix_(test[sh], cols)]                     # shuffled in place, restored below
        try:
            P1 = table_tree(m, X, test) if tree else table_linear(m, X, test)
        finally:
            X[np.ix_(test, cols)] = saved
        tot1, _p = nll_table(h, P1, test)
        v, lo, hi = EH.boot(rows["g"][test], tot1 - tot0, den, draws, SEED)
        parts.append(f"{gname} {v:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    say(f"IMPORTANCE {name} [{'TREES' if tree else 'LINEAR'}]: loss per step that returns when the group is shuffled across the test rows (larger: the group carries more): " + "; ".join(parts))


def main_run(a) -> int:
    t0 = time.perf_counter()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS frames heads: {m}")
        print(lines[-1], flush=True)
        (out / "frames_obs.txt").write_text("\n".join(lines) + "\n")
    d = build(Path(a.state_dir), say)
    rows, sample, roles, allg = d["rows"], d["sample"], d["roles"], d["allg"]
    res = sample["res"]
    fit_rows = rows["fit"]
    fit_g = np.array([allg[i]["fit"] for i in sample["ids"]])
    say(f"rows: {len(rows['g'])} settled rows of {len(res)} games; fit rows {int(fit_rows.sum())} ({len(np.unique(rows['g'][fit_rows]))} games), test rows {int((~fit_rows).sum())} ({len(np.unique(rows['g'][~fit_rows]))} games); thinned rows with concepts {int(rows['cmask'].sum())}; every row has a frame-read row")
    for lab, pick in (("fit games", fit_g), ("test games", ~fit_g)):
        idx = np.flatnonzero(pick)
        EF.coverage([res[i] for i in idx], [allg[sample["ids"][i]] for i in idx], say, lab)
    cores = EH.cores_of(res, roles, fit_g)
    heads: dict = {}
    for core, n in sorted(cores.items(), key=lambda kv: -kv[1]):
        for rel in ("us", "them"):
            h = EH.Head(core, rel, *EH.targets(core, rel, roles, rows, res))
            h.activate(fit_rows)
            nev = int((h.has & fit_rows).sum())
            if nev < ER.MIN_EVENTS:
                say(f"head {' | '.join(core)} [{rel}]: {nev} first events in the fit rows, under {ER.MIN_EVENTS}: too rare, left out")
                continue
            heads[(core, rel)] = h
    say(f"heads: {len(heads)}")
    steps = steps_of(d)
    fit_all, test = np.flatnonzero(fit_rows), np.flatnonzero(~fit_rows)
    perm = np.random.default_rng(SEED + 1).permutation(np.unique(rows["g"][fit_all]))
    tree_names = [s.strip() for s in a.tree_heads.split(",") if s.strip()] if a.tree_heads else []
    full: dict = {}
    for (core, rel), h in heads.items():
        name = f"{' | '.join(core)} [{rel}]"
        nev_fit = int((h.has & fit_rows).sum())
        say(f"HEAD {name}: first events in rows: fit {nev_fit}, test {int((h.has & ~fit_rows).sum())}; cut rows {int(h.cut.sum())}; risk steps in the fit rows {int(h.steps()[fit_all].sum())}; flags in the risk set {[('own' if f == 0 else 'other', bool(x)) for f, x in enumerate(h.active)]}")
        tot_b, _p = EH.eval_head(h, None, None, test, h.base_logit(fit_all))
        v, lo, hi = EH.boot(rows["g"][test], tot_b, h.steps()[test].astype(np.float64), a.draws, SEED)
        say(f"HEAD {name}: base hazard, test log loss per step {v:.5f} [{lo:.5f}, {hi:.5f}]")
        for sname, X, _nm in steps:
            full[(core, rel, sname, "linear")] = EH.curve(h, name, sname, X, rows, fit_all, test, perm, a, say, not a.no_curves)
        # the thinned rows: the concepts as a step of their own
        thin = np.flatnonzero(rows["cmask"])
        fit_thin, test_thin = np.intersect1d(thin, fit_all), np.intersect1d(thin, test)
        if len(fit_thin) > 200 and len(test_thin) > 100:
            pt = perm[np.array([x in set(np.unique(rows["g"][fit_thin]).tolist()) for x in perm])]
            allf = steps[-1][1]
            for sname, X in (("base K+S+E+R", steps[0][1]), ("base +concepts", np.hstack([steps[0][1], d["C"][0]])), ("+all frames", allf), ("+all frames +concepts", np.hstack([allf, d["C"][0]]))):
                EH.curve(h, name + " [thinned rows]", sname, X, rows, fit_thin, test_thin, pt, a, say, not a.no_curves)
        if " | ".join(core) in tree_names:
            sname, X, _nm = steps[-1]
            r = curve_tree(h, name, sname, X, rows, fit_all, test, perm, say, a.draws, not a.no_curves)
            full[(core, rel, sname, "trees")] = r
            lin = full[(core, rel, sname, "linear")]
            den = h.steps()[test].astype(np.float64)
            g, lo, hi = EH.boot(rows["g"][test], lin[1] - r[1], den, a.draws, SEED)
            say(f"HEAD {name} set {sname}: SAME {len(test)} TEST ROWS, log loss per step linear {lin[1].sum() / den.sum():.5f}, trees {r[1].sum() / den.sum():.5f}; linear less trees {g:+.5f} [{lo:+.5f}, {hi:+.5f}] (positive: the trees are better)")
            groups = {"ratings and ply": list(range(steps[0][1].shape[1] - 3, steps[0][1].shape[1]))}
            ofs = steps[0][1].shape[1]
            for gname in ("classes", "trap", "lost kinds", "frames"):
                w = d["blocks"][gname][0].shape[1]
                groups[gname] = list(range(ofs, ofs + w))
                ofs += w
            groups["kinds, statements, edges"] = list(range(0, steps[0][1].shape[1] - 3))
            group_importance(h, lin[0], X, groups, test, rows, say, name, False, a.draws)
            group_importance(h, r[0], X, groups, test, rows, say, name, True, a.draws)
            nm_all = steps[-1][2] + ["distance bin", "flag"]
            F, _y, _w = records(h, X, test[:min(len(test), 4000)])
            pairs = friedman_pairs(r[0], F, nm_all, top=10)
            say(f"PAIRS {name} [TREES, Friedman's H squared on the ten inputs with the widest partial dependence, 80 test records]: " + "; ".join(f"{p:.3f} {x} WITH {y}" for p, x, y in pairs[:8]))
    # the reading for a leaf (stage E) on the keeps class, linear and trees where fitted
    last = steps[-1][0]
    for core in sorted({c for (c, _r) in heads}):
        if (core, "us") not in heads or (core, "them") not in heads or " | ".join(core) not in (tree_names + [a.leaf_core]):
            continue
        hs = {(core, r): heads[(core, r)] for r in ("us", "them")}
        base_P = {r: table_linear(None, None, test, hs[(core, r)].base_logit(fit_all)) for r in ("us", "them")}
        Pl = {r: table_linear(full[(core, r, last, "linear")][0], steps[-1][1], test) for r in ("us", "them")}
        leaf_reading(core, hs, Pl, base_P, rows, test, "linear, all frames", say, a.draws)
        Pb = {r: table_linear(full[(core, r, steps[0][0], "linear")][0], steps[0][1], test) for r in ("us", "them")}
        leaf_reading(core, hs, Pb, base_P, rows, test, "linear, base inputs only", say, a.draws)
        if all((core, r, last, "trees") in full for r in ("us", "them")):
            Pt = {r: full[(core, r, last, "trees")][5] for r in ("us", "them")}
            leaf_reading(core, hs, Pt, base_P, rows, test, "trees, all frames", say, a.draws)
    say(f"wall {time.perf_counter() - t0:.1f} s; disk of the outputs {sum(f.stat().st_size for f in out.glob('*')) / 1e6:.2f} MB")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ F: the fill, with and without

def main_fill(a) -> int:
    t0 = time.perf_counter()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS frames fill: {m}")
        print(lines[-1], flush=True)
        (out / "fill_obs.txt").write_text("\n".join(lines) + "\n")
    d = build(Path(a.state_dir), say)
    rows, sample, roles, allg = d["rows"], d["sample"], d["roles"], d["allg"]
    nh = len(sample["res"])
    fill_dir = Path(a.fill_dir)
    fz, frows = EF.fill_sample(fill_dir, a.threads, say)
    fidx = {g["id"]: i for i, g in enumerate(fz["games"])}
    fill_rows = EH.build_rows({"ids": fz["ids"], "res": fz["res"]}, None, None, rows_src=frows, index=fidx)
    fst = {k: v for k, v in np.load(Path(a.fill_state_dir) / "state_rows.npz").items()}
    # the fill state rows are keyed by the game's number in fill_games.json, the same order as fz["ids"]
    have = fst["game"].astype(np.int64) * 1000 + fst["ply"].astype(np.int64)
    want = fill_rows["g"].astype(np.int64) * 1000 + fill_rows["ply"].astype(np.int64)
    order = np.argsort(have, kind="stable")
    fix = order[np.minimum(np.searchsorted(have[order], want), len(have) - 1)]
    assert (have[fix] == want).all(), "a fill row has no frame-read row"
    sub = {k: fst[k][fix] for k in ("ply", "mover", "frames", "unread", "game")}
    chunks = [fst["units"][fst["uoff"][j]:fst["uoff"][j + 1]] for j in fix]
    sub["units"], sub["uoff"] = np.concatenate(chunks), np.concatenate([[0], np.cumsum([len(c) for c in chunks])]).astype(np.int64)
    assert (fst["mover"][fix] == fill_rows["mover"]).all()
    fblocks = frame_blocks(sub, len(d["kinds"]), d["kinds"])
    # the same kept columns as the human rows
    kept = {g: [fblocks[g][1].index(n) for n in d["blocks"][g][1]] for g in d["blocks"]}
    F_h = np.hstack([d["blocks"][g][0] for g in ("classes", "trap", "lost kinds", "frames")])
    F_f = np.hstack([fblocks[g][0][:, kept[g]] for g in ("classes", "trap", "lost kinds", "frames")])
    res = sample["res"] + fz["res"]
    rows_all = EF.concat(rows, fill_rows, nh)
    X_se = np.hstack([rows_all["K"], rows_all["S"], rows_all["E"]])
    X_fr = np.hstack([X_se, np.vstack([F_h, F_f])])
    men = np.array([r["census"][p].sum() for r, p in zip((res[int(g)] for g in rows_all["g"]), rows_all["ply"])])
    is_fill = rows_all["g"] >= nh
    fit_h = np.flatnonzero(rows_all["fit"] & ~is_fill)
    test_all = np.flatnonzero(~rows_all["fit"] & ~is_fill)
    test_end = test_all[men[test_all] <= EF.ENDGAME_MEN]
    say(f"rows: human {int((~is_fill).sum())} (fit {len(fit_h)}, test {len(test_all)}), fill {int(is_fill.sum())} from {len(fz['res'])} games, every one with a frame-read row; test rows with at most {EF.ENDGAME_MEN} units: {len(test_end)}; fill rows with at most {EF.ENDGAME_MEN} units: {int((is_fill & (men <= EF.ENDGAME_MEN)).sum())}")
    fill_games = np.unique(rows_all["g"][is_fill])
    perm = np.random.default_rng(SEED + 5).permutation(fill_games)
    fit_flags = np.array([allg[i]["fit"] for i in sample["ids"]])
    cores = [c for c in sorted(EH.cores_of(sample["res"], roles, fit_flags)) if c[0] == EF.TAKING or c == (EF.KEPT,)]
    tree_names = [s.strip() for s in a.tree_heads.split(",") if s.strip()] if a.tree_heads else []
    say("comparison heads: " + ", ".join(" | ".join(c) for c in cores) + f"; inputs K+S+E and the frame inputs (no ratings, no concepts: the fill has none); trees on: {tree_names}")
    for core in cores:
        for rel in ("us", "them"):
            hd = EH.Head(core, rel, *EH.targets(core, rel, roles, rows_all, res))
            hd.activate(rows_all["fit"])
            name = f"{' | '.join(core)} [{rel}]"
            if int((hd.has & rows_all["fit"] & ~is_fill).sum()) < ER.MIN_EVENTS:
                continue
            for model in ("linear", "trees"):
                if model == "trees" and " | ".join(core) not in tree_names:
                    continue
                lam, out_tot = None, {}
                for share in (1.0, 0.0, 0.5):
                    use = perm[:int(round(len(perm) * share))]
                    fit_sel = np.concatenate([fit_h, np.flatnonzero(np.isin(rows_all["g"], use) & is_fill)]) if share else fit_h
                    if model == "linear":
                        m = EH.fit_best(hd, X_fr, rows_all, fit_sel, lam)
                        lam = m["lam"] if lam is None else lam
                    else:
                        m = fit_tree(hd, X_fr, rows_all, fit_sel, lam)
                        lam = m["n_iter"] if lam is None else lam
                    bb = hd.base_logit(fit_sel)
                    for lab, sel_test in (("all human test rows", test_all), (f"test rows with at most {EF.ENDGAME_MEN} units", test_end)):
                        P = table_linear(m, X_fr, sel_test) if model == "linear" else table_tree(m, X_fr, sel_test)
                        tot_h, _p = nll_table(hd, P, sel_test)
                        tot_b, _p2 = nll_table(hd, table_linear(None, None, sel_test, bb), sel_test)
                        den = hd.steps()[sel_test].astype(np.float64)
                        out_tot[(share, lab)] = tot_h
                        g, lo, hi = EH.boot(rows_all["g"][sel_test], tot_b - tot_h, den, a.draws, SEED)
                        say(f"HEAD {name} [{model}] on {lab} ({len(sel_test)} rows, {int(hd.has[sel_test].sum())} events): fill share {share} ({len(use)} fill games, {len(fit_sel)} fit rows): log loss per step {tot_h.sum() / den.sum():.5f}; gain over its base hazard {g:+.5f} [{lo:+.5f}, {hi:+.5f}]")
                for lab, sel_test in (("all human test rows", test_all), (f"test rows with at most {EF.ENDGAME_MEN} units", test_end)):
                    den = hd.steps()[sel_test].astype(np.float64)
                    g, lo, hi = EH.boot(rows_all["g"][sel_test], out_tot[(0.0, lab)] - out_tot[(1.0, lab)], den, a.draws, SEED)
                    say(f"HEAD {name} [{model}] on {lab}: log loss per step WITHOUT the fill less WITH all of it {g:+.5f} [{lo:+.5f}, {hi:+.5f}] (positive: the fill helps)")
    say(f"wall {time.perf_counter() - t0:.1f} s")
    return 0


GAIN = re.compile(r"HEAD (?P<head>.+?) set (?P<set>.+?)(?P<trees> \[TREES\])?: fit games (?P<k>\d+) \(fit rows (?P<r>\d+)\): .*? per step (?P<g>[+-][\d.]+) \[(?P<lo>[+-][\d.]+), (?P<hi>[+-][\d.]+)\]")
BASE = re.compile(r"HEAD (?P<head>.+?): base hazard, test log loss per step (?P<v>[\d.]+)")
SAME = re.compile(r"HEAD (?P<head>.+?) set (?P<set>.+?): SAME (?P<n>\d+) TEST ROWS, log loss per step linear (?P<a>[\d.]+), trees (?P<b>[\d.]+); linear less trees (?P<g>[+-][\d.]+) \[(?P<lo>[+-][\d.]+), (?P<hi>[+-][\d.]+)\]")


def summarise(lines: list) -> list:
    """Tables read off the OBS lines a run printed (nothing is computed here but the arrangement): per head the gains at all fit games by input step with bands, the trees beside the linear head on the same
    rows, and the learning curves."""
    gains: dict = {}
    base: dict = {}
    same: dict = {}
    for ln in lines:
        m = GAIN.search(ln)
        if m:
            key = (m["head"], m["set"], bool(m["trees"]))
            gains.setdefault(key, {})[int(m["k"])] = (m["g"], m["lo"], m["hi"])
            continue
        m = BASE.search(ln)
        if m:
            base[m["head"]] = m["v"]
            continue
        m = SAME.search(ln)
        if m:
            same[m["head"]] = (m["n"], m["a"], m["b"], m["g"], m["lo"], m["hi"])
    out = []
    heads = []
    for (h, _s, _t) in gains:
        if h not in heads and "[thinned rows]" not in h:
            heads.append(h)
    steps = ["base K+S+E+R", "+classes", "+trap", "+lost kinds", "+frames"]
    out.append("TABLE A: per head, the gain over its base hazard at ALL the fit games, per step of the head's risk set, with the 90 pct band over test games; columns: base hazard's log loss per step; the steps of inputs, each added to the one before (the first is the K+S+E+R of 78.64); the trees on the last step; the trees' loss less the linear head's on the same test rows (rows, linear loss, trees loss, linear less trees)")
    for h in heads:
        cells = []
        for s in steps:
            g = gains.get((h, s, False), {})
            cells.append(f"{s} {g[max(g)][0]} [{g[max(g)][1]}, {g[max(g)][2]}]" if g else f"{s} -")
        t = gains.get((h, "+frames", True), {})
        cells.append(f"TREES {t[max(t)][0]} [{t[max(t)][1]}, {t[max(t)][2]}]" if t else "TREES -")
        sm = same.get(h)
        cells.append(f"same rows {sm[0]}: linear {sm[1]}, trees {sm[2]}, linear less trees {sm[3]} [{sm[4]}, {sm[5]}]" if sm else "same rows -")
        out.append(f"TABLE A | {h} | base loss {base.get(h, '-')} | " + " | ".join(cells))
    out.append("TABLE B: learning curves, the gain over the base hazard on an eighth, a quarter, a half and all of the fit games (games: gain [band]); linear by step, then the trees")
    for (h, s, t), g in gains.items():
        if "[thinned rows]" in h:
            continue
        out.append(f"TABLE B | {h} | {s}{' [TREES]' if t else ''} | " + " | ".join(f"{k} games: {v[0]} [{v[1]}, {v[2]}]" for k, v in sorted(g.items())))
    out.append("TABLE C: the thinned rows (the rows that have the 64 concepts; 'concepts' a step of its own), the gain at ALL the fit games on those rows")
    for (h, s, t), g in gains.items():
        if "[thinned rows]" in h:
            out.append(f"TABLE C | {h} | {s} | {g[max(g)][0]} [{g[max(g)][1]}, {g[max(g)][2]}] ({max(g)} fit games)")
    return out


def cmd_summary(a) -> int:
    lines = Path(a.obs).read_text().splitlines()
    out = ["OBS frames table: " + x for x in summarise(lines)]
    for x in out:
        print(x)
    Path(a.obs).with_name("frames_table.txt").write_text("\n".join(out) + "\n")
    return 0


def cmd_ablate(a) -> int:
    """What the trees add beyond the linear head, split in two: (1) more boosting rounds than the cap of the run (the cap was reached by some heads), (2) the distance acting on the log odds additively
    (as in the linear head) against the distance interacting with the inputs. Same test rows, the same likelihood, one fit each at all the fit games."""
    t0 = time.perf_counter()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS frames ablate: {m}")
        print(lines[-1], flush=True)
        (out / "ablate_obs.txt").write_text("\n".join(lines) + "\n")
    d = build(Path(a.state_dir), say)
    rows, sample, roles, allg = d["rows"], d["sample"], d["roles"], d["allg"]
    res = sample["res"]
    fit_g = np.array([allg[i]["fit"] for i in sample["ids"]])
    cores = EH.cores_of(res, roles, fit_g)
    steps = steps_of(d)
    X = steps[-1][1]
    fit_all, test = np.flatnonzero(rows["fit"]), np.flatnonzero(~rows["fit"])
    for core in sorted(cores, key=lambda c: -cores[c]):
        if " | ".join(core) not in [s.strip() for s in a.heads.split(",")]:
            continue
        for rel in a.rels.split(","):
            h = EH.Head(core, rel, *EH.targets(core, rel, roles, rows, res))
            h.activate(rows["fit"])
            name = f"{' | '.join(core)} [{rel}]"
            den = h.steps()[test].astype(np.float64)
            lin = EH.fit_best(h, X, rows, fit_all, None)
            tl, _ = nll_table(h, table_linear(lin, X, test), test)
            fits = {}
            for tag, kw in (("trees, run settings (at most %d rounds)" % MAX_ITER, {}), ("trees, at most 500 rounds", {"max_iter": 500}), ("trees, at most 500 rounds, distance additive", {"max_iter": 500, "additive_distance": True})):
                m = fit_tree(h, X, rows, fit_all, None, **kw)
                tt, _ = nll_table(h, table_tree(m, X, test), test)
                fits[tag] = (m, tt)
                say(f"HEAD {name} [{tag}]: rounds {m['n_iter']}{' (the cap)' if m['capped'] else ''}; test log loss per step {tt.sum() / den.sum():.5f}; linear head {tl.sum() / den.sum():.5f}")
            tags = list(fits)
            for (la, ta), (lb, tb) in ((("linear", tl), (tags[0], fits[tags[0]][1])), ((tags[0], fits[tags[0]][1]), (tags[1], fits[tags[1]][1])), ((tags[1], fits[tags[1]][1]), (tags[2], fits[tags[2]][1])), (("linear", tl), (tags[2], fits[tags[2]][1]))):
                g, lo, hi = EH.boot(rows["g"][test], ta - tb, den, EH.DRAWS, SEED)
                say(f"HEAD {name}: [{la}] less [{lb}] on the same {len(test)} test rows {g:+.5f} [{lo:+.5f}, {hi:+.5f}] (positive: the second is better)")
    say(f"wall {time.perf_counter() - t0:.1f} s")
    return 0


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ab = sub.add_parser("ablate")
    ab.add_argument("--heads", default="taken,keeps")
    ab.add_argument("--rels", default="us")
    ab.add_argument("--state-dir", default=paths.experiment(*STATE))
    ab.add_argument("--out-dir", required=True)
    sm = sub.add_parser("summary")
    sm.add_argument("--obs", required=True)
    r = sub.add_parser("run")
    r.add_argument("--no-curves", action="store_true")
    r.add_argument("--tree-heads", default="taken,keeps,end")
    r.add_argument("--leaf-core", default="keeps")
    f = sub.add_parser("fill")
    f.add_argument("--fill-dir", default=paths.experiment("events_78_64", "fill"))
    f.add_argument("--fill-state-dir", default=paths.experiment("frames_heads_78_65", "state_rows", "fill"))
    f.add_argument("--tree-heads", default="taken,keeps")
    f.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    for p in (r, f):
        p.add_argument("--state-dir", default=paths.experiment(*STATE))
        p.add_argument("--out-dir", required=True)
        p.add_argument("--draws", type=int, default=EH.DRAWS)
    a = ap.parse_args(argv)
    return {"run": main_run, "fill": main_fill, "summary": cmd_summary, "ablate": cmd_ablate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
