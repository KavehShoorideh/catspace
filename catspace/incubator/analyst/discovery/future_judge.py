"""PLAN 78.94 STEP ONE, THE FUTURE JUDGE (SYSTEM_SPEC 7 "the future imagination layer, how it is built" and "the prior learns the tactical situations of the opening"; 6 the estimator's heads; 9 the oracle's labels, the wall). A PRIOR: it orders and chooses among
equal moves and never scores a leaf (AGENTS.md rule 7). Game-free: a kind, a statement or a move is a word or a number read from the engine at run time. The judge never calls the oracle and reads none of its numbers: it is fitted on the CLASS and the goals that the
labelling stage stored (the raw score sits beside them in the label file and no function here reads it: tests/test_plan78_94_future.py), on FIND games only.

THE ENCODER IS THE ONE THE ESTIMATOR'S HEADS READ (plan 78.59's rows): the layer-1 table's counts of units by kind for each side and the engine's statements for each side (human_heads_data.build_rows' inputs), as the mover sees them, in the side's own order (the side asked
first). The statements that are read in a worth (the engine's table of worths) are left out: no worth is assumed here, and the judge learns its own. THE HEADS are of the two forms the estimator's heads have: a linear head (L2 logistic, the penalty chosen on FIND by games in
folds) and boosted trees (HistGradientBoostingClassifier, the rounds chosen on a fifth of the FIND games held out inside FIND, the held-out loss printed round by round), one more head of each on the same inputs; nothing new is invented.

INTERFACE (written before the bodies).
  rows_from_labels(recs, ks) -> Rows       one row per (game, sampled ply, side): the side asked (the mover or not), the targets G_k (a goal of that side holds at a sampled ply of the same game within k plies after), W_k (a winning point of that side: the oracle's class
                                           for it), L_k (a losing point), GW_k (either), -1 where the window leaves the game; `now_*` the same flags at the row itself. The class and the goals come from the label records; the raw score is never read.
  Inputs(layer1, statements)               inputs_of(wires, plies) -> the unit counts and statements of the positions (mover's block first, then the other's), as arrays.
  side_matrix(inp, side_is_mover) -> X     the side's own order: [its kinds, the other's, its statements, the other's, 1 when it is the mover].
  FutureJudge.fit(rows, X)                 one linear head and one tree head per target, on FIND rows only (raises on any other part); the head used for a target is the one with the lower held-out loss on FIND.
  FutureJudge.chance(X, k, what) -> probability     what in {"goal", "win", "lose", "either"}; X the side matrices (side_matrix) of one or more positions. The wake's call (never the oracle's).
  FutureJudge.chance_of(X, target, k) -> probability the pairwise form: `target` a description in facts (the kind words of a goal's two threats ordered by the judge's own worth order and whether it is open or a fork); fitted by fit_pair on pair_rows.
  Called by: future_measure.py (fit, score, explicit, human-move test), the match player of stage five if it runs, tests.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

KS = (4, 8, 16)                                               # plies after the row: the windows the judge is asked about (stated; the label stride is two plies)
WHATS = ("goal", "win", "lose", "either")
EPS = 1e-9


# ------------------------------------------------------------------------------------------------------------------------------ the targets

def side_flags(q: dict, side_is_mover: bool) -> "tuple[bool, bool, bool]":
    """(a goal of the side holds, the side is at a winning point, at a losing point) at a labelled position `q`, for the side that is the mover at it or the other."""
    g = (q["mover_g"] if side_is_mover else q["other_g"])["goal"]
    c = q["cls"]
    win = (c == 1) if side_is_mover else (c == -1)
    lose = (c == -1) if side_is_mover else (c == 1)
    return bool(g), bool(win), bool(lose)


def result_of(y_first: int, ply: int, side_is_mover: bool, mover_at: int) -> int:
    """The game's result for the side (0 win, 1 draw, 2 loss), from the first mover's result `y_first`: the first mover moves at even plies, so the mover at `ply` is the first mover when `ply` is even."""
    mover_is_first = (mover_at == 0)
    side_is_first = mover_is_first if side_is_mover else not mover_is_first
    return y_first if side_is_first else 2 - y_first


class Rows:
    """The label table: arrays over the rows, in game order (a row is a game, a position, a side)."""

    def __init__(self, cols: dict):
        self.c = cols

    def __len__(self) -> int:
        return len(self.c["game"])

    def __getitem__(self, k: str) -> np.ndarray:
        return self.c[k]

    def select(self, mask: np.ndarray) -> "Rows":
        return Rows({k: v[mask] for k, v in self.c.items()})


def rows_from_labels(recs: list, ks: tuple = KS) -> Rows:
    """See the module doc. A window that leaves the game (the row's ply + k beyond the last move less one) is -1. The raw score of a label is never read."""
    cols: dict = {n: [] for n in ("game", "ord", "ply", "mover_side", "units", "n_s", "n_o", "part", "y", "mover_at", "n_moves")}
    for k in ks:
        for w in ("G", "W", "L", "GW"):
            cols[f"{w}{k}"] = []
    for w in ("G", "W", "L"):
        cols[f"now_{w}"] = []
    for gi, r in enumerate(recs):
        pos = r["positions"]
        for i, q in enumerate(pos):
            for side_is_mover in (True, False):
                g, win, lose = side_flags(q, side_is_mover)
                cols["game"].append(gi)
                cols["ord"].append(r["ord"])
                cols["ply"].append(q["ply"])
                cols["mover_side"].append(int(side_is_mover))
                cols["units"].append(q["units"])
                cols["n_s"].append(q["n_us"] if side_is_mover else q["units"] - q["n_us"])
                cols["n_o"].append(q["units"] - cols["n_s"][-1])
                cols["part"].append(0 if r["part"] == "find" else 1)
                cols["y"].append(result_of(r["y"], q["ply"], side_is_mover, q["mover"]))
                cols["mover_at"].append(q["mover"])
                cols["n_moves"].append(r["n_moves"])
                cols["now_G"].append(int(g)); cols["now_W"].append(int(win)); cols["now_L"].append(int(lose))
                for k in ks:
                    ahead = [side_flags(x, side_is_mover) for x in pos[i + 1:] if x["ply"] <= q["ply"] + k]
                    seen = q["ply"] + k <= r["n_moves"] - 1
                    for name, idx in (("G", 0), ("W", 1), ("L", 2)):
                        cols[f"{name}{k}"].append(int(any(a[idx] for a in ahead)) if seen else -1)
                    cols[f"GW{k}"].append(int(any(a[0] or a[1] for a in ahead)) if seen else -1)
    return Rows({k: np.array(v) for k, v in cols.items()})


# ------------------------------------------------------------------------------------------------------------------------------ the inputs

class Inputs:
    """The estimator's inputs for a set of positions: see the module doc. `layer1` the engine's graph model (graph_layer1.model), `statements` the function human_heads_data.statements_chunked."""

    def __init__(self, layer1, statements):
        self.layer1, self.statements = layer1, statements
        self.kind_names = None
        self.keep = None

    def inputs_of(self, wires: list, plies: list, threads: int = 1) -> dict:
        """`wires` [(start, moves)], `plies[i]` the plies of game i wanted; arrays over the wanted positions in order: ku, ko (units by kind, mover's then the other's), su, so (statements, worth-free ones only)."""
        names, raw = self.layer1.position_table([(s, list(m)) for s, m in wires], threads)
        nk = len(names)
        self.kind_names = list(names)
        flat = np.frombuffer(raw, dtype=np.int16)
        w = 2 + 2 * nk
        sn, stats = self.statements([{"start": s, "moves": list(m)} for s, m in wires], threads)
        if self.keep is None:
            self.keep = [j for j, (_n, d) in enumerate(sn) if "worth" not in d]
            self.stat_names = [sn[j][0] for j in self.keep]
        ns = len(sn)
        out = {"ku": [], "ko": [], "su": [], "so": []}
        off = 0
        for gi, (s, m) in enumerate(wires):
            n = len(m) + 1
            t = flat[off:off + n * w].reshape(n, w).astype(np.float32)
            off += n * w
            st = stats[gi]
            for p in plies[gi]:
                out["ku"].append(t[p, 2:2 + nk]); out["ko"].append(t[p, 2 + nk:2 + 2 * nk])
                row = np.nan_to_num(st[p]) if p < len(st) else np.zeros(2 * ns, np.float32)
                out["su"].append(row[:ns][self.keep]); out["so"].append(row[ns:][self.keep])
        return {k: np.array(v, np.float32) for k, v in out.items()}


def side_matrix(inp: dict, side_is_mover: "np.ndarray | bool") -> np.ndarray:
    """X for rows: the side's own kinds, the other's, the side's statements, the other's, and 1 when the side is the mover. `side_is_mover` is a flag per row or one flag."""
    n = len(inp["ku"])
    f = np.broadcast_to(np.asarray(side_is_mover, bool), (n,))[:, None]
    ku, ko, su, so = inp["ku"], inp["ko"], inp["su"], inp["so"]
    return np.hstack([np.where(f, ku, ko), np.where(f, ko, ku), np.where(f, su, so), np.where(f, so, su), f.astype(np.float32)])


def unit_matrix(inp: dict, side_is_mover: "np.ndarray | bool") -> np.ndarray:
    """THE FIRST BASELINE: the count of units of the side and of the other, nothing else."""
    n = len(inp["ku"])
    f = np.broadcast_to(np.asarray(side_is_mover, bool), (n,))
    a, b = inp["ku"].sum(axis=1), inp["ko"].sum(axis=1)
    return np.stack([np.where(f, a, b), np.where(f, b, a)], axis=1).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------------------------ the heads

def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean())


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """The area under the ranking curve (ties counted half): the chance that a random positive is ranked above a random negative; nan when one class is missing."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ps = p[order]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 5) -> list:
    """[(mean predicted, observed share, rows)] over bins of equal count of the predictions."""
    o = np.argsort(p)
    out = []
    for part in np.array_split(o, bins):
        if len(part):
            out.append((float(np.mean(p[part])), float(np.mean(y[part])), int(len(part))))
    return out


def effective_rank(X: np.ndarray) -> float:
    """exp of the entropy of the normalised singular values of the standardised inputs: how many directions the encoder's output really uses (1 is a collapse)."""
    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def fit_linear(X: np.ndarray, y: np.ndarray, groups: np.ndarray, grid: tuple = (0.003, 0.03, 0.3, 3.0), folds: int = 5) -> dict:
    """L2 logistic on standardised inputs; the penalty chosen by the held-out log loss with the FIND games cut into folds (GroupKFold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    best, curve = None, []
    for c in grid:
        losses = []
        for tr, te in GroupKFold(min(folds, len(np.unique(groups)))).split(Z, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = LogisticRegression(C=c, max_iter=2000).fit(Z[tr], y[tr])
            losses.append(logloss(m.predict_proba(Z[te])[:, 1], y[te]))
        curve.append((c, float(np.mean(losses)) if losses else float("inf")))
    c = min(curve, key=lambda t: t[1])[0]
    m = LogisticRegression(C=c, max_iter=2000).fit(Z, y)
    return {"kind": "linear", "m": m, "mu": mu, "sd": sd, "C": c, "curve": curve}


def fit_tree(X: np.ndarray, y: np.ndarray, groups: np.ndarray, say=None, max_iter: int = 200, seed: int = 0) -> dict:
    """Boosted trees; the rounds chosen on a fifth of the FIND games held out inside FIND (the held-out loss printed every 25 rounds), then fitted on all."""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    kw = dict(learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0)
    gs = np.unique(groups)
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(gs, max(1, len(gs) // 5), replace=False).tolist())
    ho = np.array([g in hold for g in groups])
    if len(np.unique(y[~ho])) < 2 or len(np.unique(y[ho])) < 2:
        return {"kind": "tree", "m": None, "n_iter": 0, "curve": []}
    clf = HGB(max_iter=max_iter, early_stopping=False, random_state=0, **kw).fit(X[~ho], y[~ho])
    losses = [logloss(p[:, 1], y[ho]) for p in clf.staged_predict_proba(X[ho])]
    n_iter = int(np.argmin(losses)) + 1
    if say:
        say("held-out log loss by round: " + ", ".join(f"{i + 1}: {losses[i]:.4f}" for i in range(0, len(losses), 25)) + f"; best round {n_iter}")
    m = HGB(max_iter=n_iter, early_stopping=False, random_state=0, **kw).fit(X, y)
    return {"kind": "tree", "m": m, "n_iter": n_iter, "curve": [float(v) for v in losses[::25]], "held_loss": float(losses[n_iter - 1])}


def predict(head: dict, X: np.ndarray) -> np.ndarray:
    if head["kind"] == "linear":
        return head["m"].predict_proba((X - head["mu"]) / head["sd"])[:, 1]
    return head["m"].predict_proba(X)[:, 1]


class FutureJudge:
    """See the module doc."""

    def __init__(self):
        self.heads: dict = {}
        self.pair: dict = {}
        self.info: dict = {}

    def fit(self, rows: Rows, X: np.ndarray, targets: "list | None" = None, say=None, min_pos: int = 25) -> None:
        if (rows["part"] != 0).any():
            raise ValueError("the future judge is fitted on FIND games only")
        names = targets or [f"{w}{k}" for k in KS for w in ("G", "W", "L", "GW")]
        for name in names:
            now = {"G": "now_G", "W": "now_W", "L": "now_L", "GW": None}[name.rstrip("0123456789")]
            use = rows[name] >= 0
            if now is not None:
                use &= rows[now] == 0
            else:
                use &= (rows["now_G"] == 0) & (rows["now_W"] == 0)
            y, g = rows[name][use], rows["game"][use]
            if int(y.sum()) < min_pos or int((1 - y).sum()) < min_pos:
                self.info[name] = {"rows": int(use.sum()), "positives": int(y.sum()), "fitted": False}
                continue
            lin = fit_linear(X[use], y, g)
            if say:
                say(f"head {name}: rows {int(use.sum())}, positives {int(y.sum())}; linear penalty C {lin['C']} (held-out folds {[(c, round(v, 4)) for c, v in lin['curve']]})")
            tr = fit_tree(X[use], y, g, (lambda m: say(f"head {name}: tree " + m)) if say else None)
            pl = logloss(predict(lin, X[use]), y)
            self.heads[name] = {"linear": lin, "tree": tr}
            self.info[name] = {"rows": int(use.sum()), "positives": int(y.sum()), "fitted": True, "tree_held": tr.get("held_loss"), "linear_cv": min(v for _c, v in lin["curve"]), "use": "tree" if tr["m"] is not None and tr.get("held_loss", 9) < min(v for _c, v in lin["curve"]) else "linear"}

    def head(self, name: str) -> "dict | None":
        h = self.heads.get(name)
        if h is None:
            return None
        return h["tree"] if self.info[name]["use"] == "tree" else h["linear"]

    def chance(self, X: np.ndarray, k: int, what: str) -> np.ndarray:
        """The chance, for rows of side matrices X, that a goal / winning point / losing point (or either) of the side holds within k plies later in the same game."""
        name = {"goal": "G", "win": "W", "lose": "L", "either": "GW"}[what] + str(k)
        h = self.head(name)
        if h is None:
            raise KeyError(f"head {name} is not fitted")
        return predict(h, X)


    def fit_pair(self, rows: Rows, X: np.ndarray, pairs: dict, vocab: list, k: int, say=None) -> None:
        """THE PAIRWISE FORM on FIND rows: boosted trees on [the row's inputs, the target's one-hot]; `pairs` from pair_rows."""
        if (rows["part"][pairs["row"]] != 0).any():
            raise ValueError("the future judge is fitted on FIND games only")
        F = self.pair_features(X[pairs["row"]], pairs["target"], len(vocab))
        y, g = pairs["label"], rows["game"][pairs["row"]]
        self.pair[k] = {"tree": fit_tree(F, y, g, (lambda m: say("pairwise k " + str(k) + ": " + m)) if say else None), "n": len(vocab), "vocab": vocab}

    @staticmethod
    def pair_features(X: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
        hot = np.zeros((len(target), n), np.float32)
        hot[np.arange(len(target)), target] = 1.0
        return np.hstack([X, hot])

    def chance_of(self, X: np.ndarray, target: tuple, k: int) -> np.ndarray:
        """The pairwise chance, for rows of side matrices X, that the target description (a goal's two kind words and open or fork) comes true for the side within k plies later."""
        p = self.pair[k]
        if target not in p["vocab"]:
            raise KeyError(f"target {target} is not in the vocabulary of the labels")
        t = np.full(len(X), p["vocab"].index(target))
        return p["tree"]["m"].predict_proba(self.pair_features(X, t, p["n"]))[:, 1]


def description_of(goal: dict) -> tuple:
    """A target in facts: the kind words of the goal's two threats (the larger worth first, by the order the caller gives) and whether it is open or a fork."""
    return (tuple(goal["shape"]), goal["kind"])


def target_vocabulary(recs: list) -> list:
    """The descriptions that occur in the labels (a goal of either side), by how often, the most common first."""
    c: Counter = Counter()
    for r in recs:
        for q in r["positions"]:
            for g in (q["mover_g"], q["other_g"]):
                if g["goal"]:
                    c[(tuple(g["shape"]), g["kind"])] += 1
    return [d for d, _n in c.most_common()]


def pair_rows(recs: list, rows: Rows, vocab: list, k: int, seed: int = 0, negatives: int = 2) -> dict:
    """THE PAIRWISE FORM: rows (row index, target index, label). For each row of the table and each description that held for the side in the window (label 1) and a few descriptions drawn from other games and from the same game's past (label 0 unless
    the description does hold in the window, in which case 1): `label` = the description holds for the side at a sampled ply within k plies after the row. Only rows whose window is inside the game."""
    rng = np.random.default_rng(seed)
    index = {d: i for i, d in enumerate(vocab)}
    out = {"row": [], "target": [], "label": []}
    for i in range(len(rows)):
        gi = int(rows["game"][i])
        r = recs[gi]
        side_is_mover = bool(rows["mover_side"][i])
        ply = int(rows["ply"][i])
        if rows[f"G{k}"][i] < 0:
            continue
        window = set()
        past = []
        for q in r["positions"]:
            g = (q["mover_g"] if side_is_mover else q["other_g"])
            if g["goal"] and (tuple(g["shape"]), g["kind"]) in index:
                d = (tuple(g["shape"]), g["kind"])
                if ply < q["ply"] <= ply + k:
                    window.add(d)
                elif q["ply"] < ply:
                    past.append(d)
        for d in window:
            out["row"].append(i); out["target"].append(index[d]); out["label"].append(1)
        pool = [index[d] for d in past] + [int(t) for t in rng.integers(0, len(vocab), negatives)]
        for t in pool[:negatives + len(past)]:
            out["row"].append(i); out["target"].append(t); out["label"].append(int(vocab[t] in window))
    return {k2: np.array(v) for k2, v in out.items()}
