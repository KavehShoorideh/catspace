"""Fact-space forward operator probe: can a LINEAR map on the engine's own fact counts predict a position several plies on better than copying it?

A position is read through the engine's fact layer (from the mover's view: sides named us / them) and turned into a fixed-length vector of counts: per relation, and per
relation split by each small-cardinality constant of an argument position. Pairs (x_t, x_{t+h}) from self-play games are split by game; models fitted on the train games only:
COPY, DIRECT_h (ridge), ROLLED (the horizon-2 operator applied h/2 times), and a one-move conditioned model. Then three event readouts and the sibling-move ranking.
Game-free: the game, the vocabulary and the names of the unit / threat relations are parameters. Every number is printed on a line starting with `OBS `.
"""
import argparse
import json
import os
import time

import numpy as np

SELF, OTHER = "us", "them"
MAX_VALUES = 12


# ----------------------------------------------------------------------------------------------------------------- games

def make_games(variant, games=120, seconds=0.05, random_plies=4, max_plies=120, seed=1, out=None, threads=0):
    """Self-play games (wake.play_self) saved as JSON Lines at `out`/games.jsonl; a saved file is reused. -> list of {game, start, moves, outcome}."""
    path = os.path.join(out, "games.jsonl") if out else None
    if path and os.path.isfile(path):
        return [json.loads(l) for l in open(path) if l.strip()]
    from catspace.incubator.analyst.discovery import wake
    rows = wake.play_self(variant, games, seconds=seconds, random_plies=random_plies, max_plies=max_plies, seed=seed, threads=threads)
    rows = [{"game": r["game"], "start": r["start"], "moves": r["moves"], "outcome": r["outcome"]} for r in rows]
    if path:
        os.makedirs(out, exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows


def default_dir(variant, games, seconds, seed):
    from catspace.io import paths
    return os.path.dirname(paths.derived(f"fact_forward_{variant}_{games}g_{seconds}s_s{seed}", "games.jsonl"))


# ----------------------------------------------------------------------------------------------------------------- facts and observables

def read_facts(variant, spec, vocab_json, start, moves):
    """The facts of the position after `moves` from `start`, in the mover's view: [(relation, arg, ...)]."""
    import cs_engine as CE
    return [tuple(f) for f in CE.facts(variant, spec, vocab_json, fen=start, moves=list(moves), perspective=True)]


def read_game(variant, spec, vocab_json, game):
    """Facts of every position of a game, ply 0 .. last; stops early at a position the fact layer cannot read. -> list of fact lists."""
    out = []
    for t in range(len(game["moves"]) + 1):
        try:
            out.append(read_facts(variant, spec, vocab_json, game["start"], game["moves"][:t]))
        except Exception:
            break
    return out


class Schema:
    """A fixed list of named columns over facts. Column kinds: ("total", rel), ("split", rel, pos, value), ("material",), ("const",)."""

    def __init__(self, cols, unit_relation="cell", side_pos=-2, kind_pos=-1):
        self.cols = list(cols)
        self.unit_relation, self.side_pos, self.kind_pos = unit_relation, side_pos, kind_pos
        self.names = [self.name_of(c) for c in self.cols]
        self.index = {}
        for i, c in enumerate(self.cols):
            if c[0] == "total":
                self.index[(c[1], None, None)] = i
            elif c[0] == "split":
                self.index[(c[1], c[2], c[3])] = i
        self.material = next((i for i, c in enumerate(self.cols) if c[0] == "material"), None)

    @staticmethod
    def name_of(c):
        if c[0] == "total":
            return f"count:{c[1]}"
        if c[0] == "split":
            return f"count:{c[1]}[arg{c[2]}={c[3]}]"
        if c[0] == "material":
            return "material: units us less them"
        return "const"

    @property
    def dim(self):
        return len(self.cols)

    def encode(self, facts):
        x = np.zeros(self.dim)
        ix = self.index
        for f in facts:
            rel, args = f[0], f[1:]
            j = ix.get((rel, None, None))
            if j is not None:
                x[j] += 1
            for p, a in enumerate(args):
                j = ix.get((rel, p, a))
                if j is not None:
                    x[j] += 1
        if self.material is not None:
            x[self.material] = material_of(facts, self.unit_relation, self.side_pos)
        x[-1] = 1.0
        return x


def material_of(facts, unit_relation="cell", side_pos=-2):
    """Units of the mover's side less the other side's, read from the unit relation."""
    n = 0
    for f in facts:
        if f[0] == unit_relation:
            s = f[1:][side_pos]
            n += 1 if s == SELF else (-1 if s == OTHER else 0)
    return n


def build_schema(corpus, max_dim=200, unit_relation="cell", side_pos=-2, kind_pos=-1):
    """Discover the columns from `corpus` (a list of fact lists): one total per relation seen; for each relation and argument position with 2..12 distinct values over the
    corpus, one count per value; then keep the highest-variance split columns so the whole stays within `max_dim` (totals, the material column and the constant always stay)."""
    arity, values, rels = {}, {}, set()
    for facts in corpus:
        for f in facts:
            rels.add(f[0])
            for p, a in enumerate(f[1:]):
                values.setdefault((f[0], p), set()).add(a)
    cand = [("total", r) for r in sorted(rels)]
    splits = [("split", r, p, v) for (r, p), vs in sorted(values.items()) if 2 <= len(vs) <= MAX_VALUES for v in sorted(vs)]
    fixed = cand + [("material",)]
    room = max(0, max_dim - len(fixed) - 1)
    if len(splits) > room:
        tmp = Schema(fixed + splits + [("const",)], unit_relation, side_pos, kind_pos)
        X = np.array([tmp.encode(f) for f in corpus])
        var = X.var(axis=0)[len(fixed):len(fixed) + len(splits)]
        keep = sorted(np.argsort(-var, kind="stable")[:room])
        splits = [splits[i] for i in keep]
    return Schema(fixed + splits + [("const",)], unit_relation, side_pos, kind_pos)


def kind_values(corpus, unit_relation="cell", kind_pos=-1):
    """The sorted constants seen in the kind position of the unit relation."""
    return sorted({f[1:][kind_pos] for facts in corpus for f in facts if f[0] == unit_relation})


def moved_kind(before, after, kinds, unit_relation="cell", side_pos=-2, kind_pos=-1):
    """One-hot (over `kinds`) of the kind of the mover's unit that is new after the move: the mover is `us` before and `them` after; read from the facts alone."""
    def owned(facts, side):
        rest = lambda args: tuple(a for i, a in enumerate(args) if i not in (side_pos % len(args), kind_pos % len(args)))
        return {(rest(f[1:]), f[1:][kind_pos]) for f in facts if f[0] == unit_relation and f[1:][side_pos] == side}
    new = owned(after, OTHER) - owned(before, SELF)
    v = np.zeros(len(kinds))
    for _, k in sorted(new):
        if k in kinds:
            v[kinds.index(k)] += 1
    return v


# ----------------------------------------------------------------------------------------------------------------- pairs, split, scoring

def build_pairs(seqs, h):
    """seqs: {game id: [x_0, x_1, ...]} -> (X, Y, game ids) of pairs (x_t, x_{t+h}) within each game."""
    X, Y, G = [], [], []
    for g, xs in seqs.items():
        for t in range(len(xs) - h):
            X.append(xs[t]); Y.append(xs[t + h]); G.append(g)
    return np.array(X), np.array(Y), np.array(G)


def split_games(ids, held_frac=0.25, seed=0):
    """Split game ids into (train, held) lists, by game."""
    ids = sorted(ids)
    rng = np.random.RandomState(seed)
    perm = list(rng.permutation(len(ids)))
    n_held = max(1, int(round(len(ids) * held_frac)))
    held = sorted(ids[i] for i in perm[:n_held])
    train = sorted(ids[i] for i in perm[n_held:])
    return train, held


def ridge_fit(X, Y, lam):
    """Closed-form ridge, the last column of X taken as the constant and left unpenalised. -> W with Y ~ X @ W."""
    d = X.shape[1]
    P = lam * np.eye(d)
    P[-1, -1] = 0.0
    return np.linalg.solve(X.T @ X + P + 1e-9 * np.eye(d), X.T @ Y)


def ridge_pick(X, Y, groups, lams=(0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6), frac=0.2):
    """Pick the ridge strength on the last `frac` of the train games (by game), then refit on all of X. -> (W, lam)."""
    gs = sorted(set(groups.tolist()))
    cut = set(gs[int(len(gs) * (1 - frac)):])
    va = np.array([g in cut for g in groups])
    if va.all() or not va.any():
        return ridge_fit(X, Y, lams[len(lams) // 2]), lams[len(lams) // 2]
    best = min(lams, key=lambda l: ((X[va] @ ridge_fit(X[~va], Y[~va], l) - Y[va]) ** 2).sum())
    return ridge_fit(X, Y, best), best


def r2_columns(Y, P):
    """Per-column R2 of predictions P against Y (mean of Y as the reference); nan where Y does not vary."""
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    sse = ((Y - P) ** 2).sum(0)
    with np.errstate(all="ignore"):
        return np.where(sst > 1e-12, 1 - sse / sst, np.nan)


def rank_average(a):
    """Ranks (1-based) with ties given their average."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auc(score, label):
    """Area under the ROC curve (Mann-Whitney, ties half); nan if one class is missing."""
    y = np.asarray(label).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank_average(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    """Spearman correlation (ranks with ties averaged); nan when either side is constant."""
    ra, rb = rank_average(a), rank_average(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def logistic_fit(X, y, steps=300, lr=0.5, l2=1e-3):
    """Plain gradient descent logistic regression on standardised X (constant column kept). -> predict function."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    mu[-1], sd[-1] = 0.0, 1.0
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
    return lambda Xn: ((Xn - mu) / sd) @ w


def event_labels(material, checks, hang_us, hang_them, t, far=8, near=4):
    """The three event labels at ply t of one game (None when the future is too short). `material` is the mover-relative unit difference per ply, `checks` the count of
    facts of the goal-precondition relation per ply, `hang_us` / `hang_them` the counts of the mover-at-t's and the other side's undefended-and-attacked units per ply
    (in the view of that ply's mover, so the mover-at-t is `us` on even offsets and `them` on odd ones).
    (a) material is lower `far` plies on (same side to move); (b) the precondition relation holds at some ply in t+1..t+far; (c) the mover's hanging count rises within `near`."""
    n = len(material)
    if t + far >= n:
        return None
    a = material[t + far] < material[t]
    b = any(checks[t + k] > 0 for k in range(1, far + 1))
    mine = lambda k: hang_us[t + k] if k % 2 == 0 else hang_them[t + k]
    c = any(mine(k) > mine(0) for k in range(1, near + 1))
    return bool(a), bool(b), bool(c)


def hanging_counts(facts, relation="hanging", side_pos=0):
    """(count with the mover's side, count with the other side's) for the threat relation."""
    u = sum(1 for f in facts if f[0] == relation and f[1:][side_pos] == SELF)
    o = sum(1 for f in facts if f[0] == relation and f[1:][side_pos] == OTHER)
    return u, o


# ----------------------------------------------------------------------------------------------------------------- the probe

def _p(*a):
    print("OBS", *a, flush=True)


def _fmt(x):
    return round(float(x), 4)


def run(game, games=120, seconds=0.05, seed=1, vocabulary="relational", out=None, max_dim=200, variant=None, check_relation="check", hanging_relation="hanging",
        unit_relation="cell", side_pos=-2, kind_pos=-1, siblings=300, sibling_seconds=0.05, threads=0):
    from catspace.incubator.analyst.discovery import game_store as GS
    variant = variant or game
    spec = GS.spec_texts(game, vocabulary)[variant]
    vocab_json = json.dumps(GS.vocabulary(game, vocabulary))
    out = out or default_dir(variant, games, seconds, seed)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rows = make_games(variant, games, seconds, seed=seed, out=out, threads=threads)
    _p("games", len(rows), "plies", sum(len(r["moves"]) for r in rows), "dir", out, "seconds", _fmt(time.time() - t0))
    t0 = time.time()
    raw = {r["game"]: read_game(variant, spec, vocab_json, r) for r in rows}
    npos = sum(len(v) for v in raw.values())
    _p("positions", npos, "fact_read_ms_per_position", _fmt(1000 * (time.time() - t0) / max(npos, 1)))
    train_ids, held_ids = split_games(list(raw), 0.25, seed)
    _p("split games train", len(train_ids), "held", len(held_ids))
    schema = build_schema([f for g in train_ids for f in raw[g]], max_dim, unit_relation, side_pos, kind_pos)
    t0 = time.time()
    seqs = {g: [schema.encode(f) for f in fl] for g, fl in raw.items()}
    _p("observables dim", schema.dim, "encode_ms_per_position", _fmt(1000 * (time.time() - t0) / max(npos, 1)))
    names = schema.names
    varying = np.array([c[0] != "const" for c in schema.cols])

    def pairs(h, ids):
        return build_pairs({g: seqs[g] for g in ids}, h)

    def section(label, X, Y, models):
        """models: {name: predictions}. Prints mean R2 over varying columns, share beating COPY by .02, best and worst ten."""
        copy_r2 = r2_columns(Y, models["COPY"])
        ok = varying & ~np.isnan(copy_r2)
        for m, P in models.items():
            r2 = r2_columns(Y, P)
            ok_m = ok & ~np.isnan(r2)
            gain = r2 - copy_r2
            _p(label, m, "mean_R2", _fmt(np.nanmean(r2[ok_m])), "columns", int(ok_m.sum()), "share_beating_COPY_by_0.02", _fmt((gain[ok_m] > 0.02).mean()) if m != "COPY" else "-")
            if m != "COPY":
                idx = np.where(ok_m)[0]
                order = idx[np.argsort(-gain[idx])]
                _p(label, m, "helps_most", "; ".join(f"{names[i]} {gain[i]:+.3f}" for i in order[:10]))
                _p(label, m, "helps_least", "; ".join(f"{names[i]} {gain[i]:+.3f}" for i in order[::-1][:10]))
                kinds = {}
                for i in idx:
                    k = schema.cols[i][0] + (":" + schema.cols[i][1] if schema.cols[i][0] != "material" else "")
                    kinds.setdefault(schema.cols[i][0], []).append(gain[i])
                _p(label, m, "mean_gain_by_column_kind", "; ".join(f"{k} {np.mean(v):+.3f} (n={len(v)})" for k, v in sorted(kinds.items())))

    A2 = None
    for h in (2, 4, 8):
        Xtr, Ytr, Gtr = pairs(h, train_ids)
        Xte, Yte, _ = pairs(h, held_ids)
        W, lam = ridge_pick(Xtr, Ytr, Gtr)
        if h == 2:
            A2 = W
        P_direct = Xte @ W
        P_roll = Xte.copy()
        for _ in range(h // 2):
            P_roll = P_roll @ A2
        _p(f"h{h}", "pairs train", len(Xtr), "held", len(Xte), "ridge", lam)
        section(f"h{h}", Xte, Yte, {"COPY": Xte, "DIRECT": P_direct, "ROLLED": P_roll})
    # the operator's own numbers
    B = A2[:-1, :-1].T                                  # x' = B x  (constant column removed)
    rho = float(np.max(np.abs(np.linalg.eigvals(B))))
    Xh, _, _ = pairs(2, held_ids)
    z = Xh[:, :-1]
    Zroll, growth = z, []
    for k in range(4):
        Zroll = Zroll @ A2[:-1, :-1] + A2[-1, :-1]
        growth.append(float(np.linalg.norm(Zroll, axis=1).mean() / np.linalg.norm(z, axis=1).mean()))
    X8, Y8, _ = pairs(8, held_ids)
    _p("operator h2 spectral_radius", _fmt(rho), "mean_norm_ratio_after_1_2_3_4_rolls", " ".join(str(_fmt(g)) for g in growth),
       "true_norm_ratio_h8", _fmt(np.linalg.norm(Y8[:, :-1], axis=1).mean() / np.linalg.norm(X8[:, :-1], axis=1).mean()))
    # one move, conditioned on the kind of the moved unit
    kinds = kind_values([f for g in train_ids for f in raw[g]], unit_relation, kind_pos)
    cond = {g: [np.concatenate([schema.encode(raw[g][t]), moved_kind(raw[g][t], raw[g][t + 1], kinds, unit_relation, side_pos, kind_pos)]) for t in range(len(raw[g]) - 1)] for g in raw}

    def cpairs(ids):
        X = np.array([c for g in ids for c in cond[g]])
        Y = np.array([seqs[g][t + 1] for g in ids for t in range(len(cond[g]))])
        G = np.array([g for g in ids for _ in cond[g]])
        return X, Y, G
    Xc, Yc, Gc = cpairs(train_ids)
    Xct, Yct, _ = cpairs(held_ids)
    d = schema.dim
    # layout of the conditioned input: [x (with its constant last), kind one-hot]; keep the constant last for the ridge by moving it
    mv = lambda M: np.hstack([M[:, :d - 1], M[:, d:], M[:, d - 1:d]])
    Wc, lc = ridge_pick(mv(Xc), Yc, Gc)
    Wd, ld = ridge_pick(Xc[:, :d], Yc, Gc)
    # COPY at one ply: the sides swap between the two views, so "copy" is x_t as it stands (same relation names, other side)
    _p("h1", "pairs train", len(Xc), "held", len(Xct), "ridge conditioned", lc, "direct", ld, "kinds", ",".join(kinds))
    section("h1", Xct[:, :d], Yct, {"COPY": Xct[:, :d], "DIRECT": Xct[:, :d] @ Wd, "MOVE_CONDITIONED": mv(Xct) @ Wc})
    # events
    _events(raw, seqs, schema, train_ids, held_ids, check_relation, hanging_relation, unit_relation, side_pos)
    # siblings
    if siblings:
        _siblings(variant, spec, vocab_json, rows, held_ids, schema, A2, siblings, sibling_seconds, out, threads, unit_relation, side_pos)
    return 0


def _events(raw, seqs, schema, train_ids, held_ids, check_relation, hanging_relation, unit_relation, side_pos):
    lab, feat = {}, {}
    for g, fl in raw.items():
        mat = [material_of(f, unit_relation, side_pos) for f in fl]
        chk = [sum(1 for a in f if a[0] == check_relation) for f in fl]
        hh = [hanging_counts(f, hanging_relation) for f in fl]
        hu, ho = [a for a, _ in hh], [b for _, b in hh]
        for t in range(len(fl)):
            e = event_labels(mat, chk, hu, ho, t)
            if e is not None:
                lab.setdefault(g, []).append(e); feat.setdefault(g, []).append(seqs[g][t])
    names = ("a_material_lower_in_8", "b_" + check_relation + "_in_next_8", "c_" + hanging_relation + "_rises_in_4")
    tr = lambda ids: (np.array([x for g in ids for x in feat.get(g, [])]), np.array([e for g in ids for e in lab.get(g, [])]))
    Xtr, Ltr = tr(train_ids)
    Xte, Lte = tr(held_ids)
    for k, nm in enumerate(names):
        ytr, yte = Ltr[:, k].astype(float), Lte[:, k].astype(bool)
        base = yte.mean()
        pred = logistic_fit(Xtr, ytr)
        a_log = auc(pred(Xte), yte)
        cols = [j for j in range(schema.dim - 1)]
        tr_auc = [auc(Xtr[:, j], ytr) for j in cols]
        tr_dir = [(0 if np.isnan(a) else abs(a - 0.5)) for a in tr_auc]
        jb = cols[int(np.argmax(tr_dir))]
        sign = 1 if tr_auc[jb] >= 0.5 else -1
        a_single = auc(sign * Xte[:, jb], yte)
        te_auc = [auc(Xte[:, j], yte) for j in cols]
        te_best = max(cols, key=lambda j: 0 if np.isnan(te_auc[j]) else abs(te_auc[j] - 0.5))
        _p("event", nm, "held_n", len(yte), "base_rate", _fmt(base), "AUC_logistic_readout", _fmt(a_log), "AUC_best_single_column_chosen_on_train", _fmt(a_single),
           "column", schema.names[jb], "AUC_best_single_column_chosen_on_held_out", _fmt(max(te_auc[te_best], 1 - te_auc[te_best])), "column", schema.names[te_best])


def _siblings(variant, spec, vocab_json, rows, held_ids, schema, A2, n_pos, seconds, out, threads, unit_relation, side_pos):
    """At held-out positions: every legal move; score = predicted material two plies after the move (the position after one move is read in the OTHER side's view, so the
    horizon-2 operator is applied to it as it stands and the material column read there is negated to be the original mover's). Also the predicted CHANGE (prediction less the after-move reading), because the ledger value counts only what the search line gains from the position after the move on, not the move's own gain. Exact = the engine's ledger value of the
    position after the move, negated (the value is in the view of the side to move there)."""
    import cs_engine as CE
    rng = np.random.RandomState(7)
    held = set(held_ids)
    pool = [(r["game"], t) for r in rows if r["game"] in held for t in range(len(r["moves"]))]
    byid = {r["game"]: r for r in rows}
    sel = [pool[i] for i in rng.permutation(len(pool))[:n_pos]]
    t0 = time.time()
    items, owner, scores_op, scores_copy, scores_delta = [], [], [], [], []
    mi = schema.material
    for pi, (g, t) in enumerate(sel):
        r = byid[g]
        mv = r["moves"][:t]
        try:
            legal = CE.state(variant, r["start"], mv)[2]
        except Exception:
            continue
        for m in legal:
            try:
                fa = read_facts(variant, spec, vocab_json, r["start"], mv + [m])
            except Exception:
                continue
            x = schema.encode(fa)
            p = x @ A2
            scores_op.append(-p[mi]); scores_copy.append(-x[mi]); scores_delta.append(-(p[mi] - x[mi]))
            items.append((variant, r["start"], mv + [m])); owner.append((pi, m))
    _p("siblings positions", len(sel), "moves", len(items), "prepare_seconds", _fmt(time.time() - t0))
    trace = os.path.join(out, "sibling_trace.jsonl")
    if os.path.exists(trace):
        os.remove(trace)
    t0 = time.time()
    CE.solve_batch(items, seconds, trace_path=trace, threads=threads, labels=[f"{i}" for i in range(len(items))])
    exact = {}
    for line in open(trace):
        d = json.loads(line)
        if d.get("kind") != "position" or d.get("error"):
            continue
        for s in d.get("steps", []):
            if s.get("operator") == "ledger" and s.get("value") is not None:
                exact[int(d["label"])] = -float(s["value"])
    _p("siblings exact_seconds", _fmt(time.time() - t0), "moves_with_value", len(exact), "of", len(items))
    per_op, per_copy, per_delta = [], [], []
    for pi in range(len(sel)):
        ix = [i for i, (q, _) in enumerate(owner) if q == pi and i in exact]
        if len(ix) < 3:
            continue
        e = [exact[i] for i in ix]
        a, b = spearman([scores_op[i] for i in ix], e), spearman([scores_copy[i] for i in ix], e)
        c = spearman([scores_delta[i] for i in ix], e)
        if not np.isnan(c):
            per_delta.append(c)
        if not np.isnan(a):
            per_op.append(a)
        if not np.isnan(b):
            per_copy.append(b)
    _p("siblings positions_scored", len(per_op), "mean_spearman_operator_h2", _fmt(np.mean(per_op)) if per_op else "nan",
       "mean_spearman_copy_after_move", _fmt(np.mean(per_copy)) if per_copy else "nan",
       "mean_spearman_operator_predicted_change_only", _fmt(np.mean(per_delta)) if per_delta else "nan", "positions_scored_change", len(per_delta))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vocabulary", default="relational")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-dim", type=int, default=200)
    ap.add_argument("--siblings", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--check-relation", default="check")
    ap.add_argument("--hanging-relation", default="hanging")
    a = ap.parse_args(argv)
    return run(a.game, a.games, a.seconds, a.seed, a.vocabulary, a.out, a.max_dim, a.variant, a.check_relation, a.hanging_relation, siblings=a.siblings, threads=a.threads)


if __name__ == "__main__":
    raise SystemExit(main())
