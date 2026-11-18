"""PLAN 78.80 (SYSTEM_SPEC 7 "subgoals as symbols that go with winning", 9 the score): SYMBOLS, THE FIRST KIND, FACT PATTERNS. A symbol is an instance of a small pattern of the layer-1 graph (plan 78.53's
patterns, matched and counted in Rust) bound to its units. What a move does to symbols: for each side, the patterns whose count of instances is HIGHER on the position after the move than before (made true) and
those whose count is LOWER (made false). A symbol is a hint that orders the moves a search holds equal; it is never a value and never a bin of a head's output (AGENTS.md rule 7). Game-free: this module names
no object of any game; a pattern is its id and the engine's own words; the game's model and goal kind are handed in.

INTERFACE (what each piece takes, what it returns, who calls it). Written before the bodies.
  Counter(model, keys, goal_kind, max_size=2, levels=(0, 1, 2, 3), threads=1)
      The engine's pattern counts for a fixed list of pattern keys. `count(wires, rows) -> (us, them)` two sparse matrices (rows x keys), the counts of the side to move of the position and of the other side;
      `wires` the engine's (start, moves) games, `rows[i]` the plies wanted of game i, the rows numbered in the order of `rows` across games. Called by deltas() and by the measure.
  mine(model, wires, rows, goal_kind, floor, sizes, levels, threads) -> the patterns met on at least `floor` of the wanted rows (the keys for a Counter), patterns with a `shut` atom left out (that relation exists
      for the side to move only, so its count before and after a move is not comparable). Called by symbols_measure on the find part's rows only.
  event            (column, side, direction): column = the index of the pattern in the counter's keys; side "mover" (the side that plays the tied move) or "other"; direction "up" (made true: its count is higher after
                   the move) or "down" (made false: lower). The id of a pattern is `pattern_id(key)`, the key in hexadecimal.
  deltas_matrix(positions, tied_lists, counter) -> a sparse 0/1 matrix, one row per tied move (positions in order, each one's tied moves in order), 4 x n_keys columns (event_index): what the measure keeps at scale.
  deltas(positions, tied_lists, counter) -> [ {move: frozenset of events} ] one dict per position; `positions` are {"moves": [...]} (the game so far), `tied_lists[i]` the moves held equal there.
      The mover's count before is the position's count for the side to move; after the move it is the count of the OTHER side of the position after it (the side to move there is the other side).
      `deltas_of(position, tied, counter)` is the same for one position. Called by symbols_measure (stages b, f) and ladder_print.
  rate_table(E, game, played) -> test two for every event column of the move-by-event matrix E (event_matrix builds it); term_table(...) -> test one for pattern columns.
  find(two, one, n_cols) -> (picked, tried)   the rule below on the find part's tables;   confirm(picked, two, one) -> the picked events with the confirm part's numbers and "passes" (two over the picked events only).
  symbol_pick(position, tied, patterns, counter=None, table=None) -> a move INSIDE `tied`: the move with the highest sum, over the passing events it carries, of the event's weight; a tie (all equal, or no
      tied move carries one: the pick is silent) goes to the search's first move, which is tied[0]. `patterns` = [{"column", "side", "dir", "weight"}]. Called by symbols_measure (stage f). It reads no answer.

THE TWO TESTS (stated before they were run; SYSTEM_SPEC 7).
  TEST ONE (goes with the result beyond the count of units): on the find part's head rows, a head of the result of the side to move on its inputs (rating terms, ply, the count of units by kind, ours less theirs: plan 78.59's
  R+K, fitted once and held), then the same with the pattern's count for each side added: an offset fit (the head's own scores held; the win and loss classes get a free shift and a term for each side's count,
  Newton steps on all the head rows, a small ridge on the terms), the term of a side being the log-odds of a win against a loss per extra instance, for the result of the side to move. Its band: 90 pct over games resampled.
  (The head's own coefficients are not refitted with each pattern: that is the one way this is cheaper than adding a column and refitting everything.)
  TEST TWO (the move played reaches it): per tied move of the sample positions, the share of the PLAYED moves that make the event true against the share of the OTHER tied moves that do (smoothed by one half), the
  log of their ratio with a 90 pct interval from the games as clusters (the delta method with the covariance of the two shares inside a game).
  THE RULE (fixed before the run): an event is PICKED on the find part when (i) it occurs on at least FLOOR_MOVES tied moves of which at least FLOOR_EACH are played moves and at least FLOOR_EACH are other moves, and its
  pattern occurs on at least FLOOR_ROWS head rows; (ii) the interval of test two excludes equality; (iii) test one's band for the side excludes zero; (iv) the signs agree: sign(term) x direction (+1 up, -1 down)
  equals the sign of the log ratio. It PASSES when, read once on the confirm part, (ii) and (iii) hold with the signs the find part had. Its weight is the find part's log ratio.

PLAN 78.81 (SYSTEM_SPEC 7 "subgoals as symbols" and "reachable has two layers", 10 the endings risk, 9 the score): FEWER, CLEANER REASONS, THEN A FIRST READING OF LAYER ONE. INTERFACE, written before the bodies.
  Choice(E, C, control_names, pos, played, game, part, source)
      The rows of a choice fit: one row per tied move, positions contiguous (`pos` the position of the row, `played` True on its one played move), E the events (0/1, rows x events), C the controls (rows x control
      names, floats), `game` the cluster of each row, `part` "find" or "confirm", `source` "human" or "generated". Built by symbols_reasons_measure; read by fit_choice, drop_duplicates and score_choice.
  require_find(rows)
      Raises ValueError unless every row is of the FIND part and of a human game: the fit and the duplicate drop read no CONFIRM row and no generated row (the mover of a generated game is our own player).
      Called first by both.
  drop_duplicates(rows, order, threshold) -> (kept columns, info)
      Events whose carrier sets (the tied-move rows that carry them) equal another's, or are near it (Jaccard of the two sets at or above `threshold`), are dropped, the one with the smaller pattern kept:
      `order[j]` is the sort key of column j (size, level, key), the smaller kept; an event with no carrier is dropped. info: {"in", "empty", "exact", "near", "kept"}. Reads rows.E and the part and source of the rows only.
  fit_choice(rows, controls=(), events=None, folds=FOLDS, seed=0) -> fit
      A CONDITIONAL CHOICE MODEL: a softmax over the tied moves of one position, the played move the one chosen. The score of a tied move is the sum of the weights of the events it carries plus the weights of the
      controls named in `controls` (columns of rows.C); `events` the event columns used (None: all of rows.E; []: none). The events' weights carry an L1 penalty (sparsity), the controls a small ridge only. The penalty
      is chosen on FIND alone: the games are cut into `folds` parts, the path of penalties is fitted on all but one part and read on that one, and the largest penalty whose held-out log likelihood is within one
      standard error of the best is kept; the final fit is on all the rows. torch (a standard framework, already in the environment) does the arithmetic: FISTA, a proximal gradient with momentum, a small body tested
      against planted weights (no library in the environment fits a conditional logit with an L1 penalty).
      -> {"lambda", "events" (the columns with a nonzero weight), "weights" (for those), "controls" (name -> weight), "cv" [(lambda, mean held-out log likelihood a position, standard error)], "n_events", "positions"}.
  score_choice(rows, fit, controls=False) -> per position the index (inside the position) of the tied move with the highest score, the first on a tie: a hint that orders moves in the tie set, never a value.
  Reach(model, variant, goal_kind, kinds)   what ahead() reads: the engine's graph model of the game and its variant (the legal moves come from the engine's own state), the goal kind and the kind words.
  reach_rows(env, positions, counter) -> per position: the legal moves and, per key of the counter, the moves that make the event "mover up" true there.
  one_short(items, units, pattern, kinds, side) -> (short, full)   the keys (atom number, the units of that atom's unit variables) of the instances of a pattern of size 2 that are ONE FACT SHORT (one of the two atoms
      holds on these units and no second atom completes the pattern on them) and of those that are complete. Read for the side `side` (0: the side to move, 1: the other side).
  nearer(env, text, move, pattern) -> bool   the move makes some instance one fact short that was not one before (the units followed through the move), for the mover.
  ahead(position, reason, limit, env) -> count or None
      `position` {"moves": the game up to the move that makes the reason true}; `reason` {"key", "side", "dir", "bytes"}. Walks back along the same game to the mover's own earlier positions (k = 1, 2, ... own moves
      earlier, at most `limit`, while the position is in the record) and returns the length of the unbroken run of earlier own positions at which the pattern was reachable in exactly one own move (some legal move makes
      it true); 0 when it was not at k = 1; None for a reason this layer does not read (a "down" reason or the other side's: no walk back is defined for them) or when the record has no earlier own position.
      `limit` is a cost guard: a count equal to it may be longer (symbols_reasons_measure prints how often it is hit). Called by symbols_reasons_measure (stage ahead) and the tests.
  Called by: symbols_reasons_measure.py; ladder_print.py reads its outputs for the cases.

PLAN 78.88 (SYSTEM_SPEC 7 "subgoals as symbols", the form of the model "a combination, mixture of experts kind of thing"): ONE REASON ONE PURPOSE, AND WEIGHTS BY PART OF THE GAME. INTERFACE, written before the bodies.
  THE GROUPING RULE (fixed before it was run; no number in it was tuned): two reasons are TWINS when their carrier sets on the FIND rows (the tied moves, in the human games' FIND positions, that carry the event) are alike. The groups are
  the clusters of AGGLOMERATIVE CLUSTERING with AVERAGE LINKAGE on the Jaccard index of the carrier sets: start with every reason alone; repeatedly merge the two clusters with the highest average Jaccard index over all pairs of
  their members, but only two clusters whose members all have the same side, the same direction (up or down) and the same sign of the weight (78.81's weight; for a cell of `split_borne` its parent's); stop when the best pair is
  below GROUP_SIM = 0.5 (the one fixed threshold). A reason with no carrier on the rows stays alone. One group is one feature: true on a tied move when ANY member fires; it gets one weight in the fit.
  groups_of(reasons, rows) -> [[reason index, ...]]    `reasons` the list of {"side", "dir", "weight"}, `rows` a Choice of FIND rows whose column i of E is reason i (require_find is called first). Groups in the order of their first member.
  group_matrix(E, groups) -> a sparse 0/1 matrix, rows x groups: the OR of the members' columns.
  split_borne(reasons, kinds) -> reasons    each reason of the form "a unit of the mover's bears on a unit of theirs" (bear_form: one atom, the first unit the mover's, the second the other side's, event side "mover") is
      replaced by four, one for each CELL, a cell being two facts the layer's graph already holds about the instance's units: whether the unit of theirs is GUARDED (a unit of its own side bears on it) and whether the bearing unit
      is BORNE ON (a unit of the other side bears on it); cell = 2 x guarded + borne. The four copy the parent and carry "cell" and "parent" (the index in the input); every other reason is copied with "cell" None.
      Why Python and not patterns: a count of instances of a size-2 pattern (the two bears onto one unit) counts a unit once for each guard, and the language has no negation, so "loose" and "not borne on" are not
      patterns; the matcher's instances (graph_layer2_match.match_binds) carry the units, the graph's own `bear` items carry the two facts, and no Rust change is needed.
  cell_counts(items, units, pattern, kinds) -> (first, second)  the instance counts of a bear_form pattern by cell, read for the side to move (first) and for the other side (second); each a list of four.
  cell_events(before, after) -> (up cells, down cells)   the mover's count before is the first of the position, after the move the second of the position after it (the side to move there is the other side).
  Choice.units (optional)   the units on the board at the position of each row (the caller's count; this module counts nothing of any game); fit_choice(rows, ..., band=(low, high)) reads the positions whose units lie in the band
      only; score_choice_by_band(rows, fits, bands, controls) -> per position the index of the highest-scoring tied move under the fit of the position's band (the first on a tie); `fits` {band name: fit}, `bands` [(name, low, high)].
PLAN 78.89 (SYSTEM_SPEC 7, 9, 10 the endings risk and the fill): Choice.fill_ok (default False) -- the builder of the rows says that rows marked source "generated" (our own self-play games, FIND only, the mover's own move the one
  "played") may be read by fit_choice; require_find then lets them pass. Without it nothing changes: a generated row is refused as before. The CONFIRM part is never generated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from catspace.incubator.analyst.discovery import graph_layer2 as L2

Z90 = 1.645
FLOOR_MOVES = 30                      # an event must occur on at least this many tied moves of the find part ...
FLOOR_EACH = 10                       # ... of which at least this many are played moves and this many are other tied moves
FLOOR_ROWS = 200                      # the pattern's count must be above zero on this many head rows of the find part (either side)
SIDES = ("mover", "other")
DIRS = ("up", "down")
PRIOR = 0.5                           # the smoothing of a share: (k + PRIOR) / (n + 2 PRIOR)
RIDGE = 1.0                           # the ridge of the added terms (on the sum of the log likelihood)


def pattern_id(key: int) -> str:
    return f"{int(key):016x}"


def has_shut(raw: bytes) -> bool:
    """True for a pattern with a `shut` atom: those relations exist for the side to move only, so their counts before and after a move are not comparable and they are left out."""
    return any(a[0] == "shut" for a in L2.decode(raw).atoms)


# ------------------------------------------------------------------------------------------------------------------------------ the counts

class Counter:
    """The engine's pattern counts for a fixed list of keys. See the module doc."""

    def __init__(self, model, keys: list, goal_kind, max_size: int = 2, levels: tuple = (0, 1, 2, 3), threads: int = 1):
        self.model, self.keys, self.goal_kind = model, [int(k) for k in keys], goal_kind
        self.max_size, self.levels, self.threads = max_size, list(levels), threads

    def count(self, wires: list, rows: list):
        n = sum(len(r) for r in rows)
        raw = self.model.pattern_count(wires, rows, self.keys, self.max_size, self.levels, self.goal_kind, self.threads)
        return L2.counts_matrix(raw, n, len(self.keys))


def mine(model, wires: list, rows: list, goal_kind, floor: int, sizes: tuple = (1, 2), levels: tuple = (0, 1, 2, 3), threads: int = 1) -> list:
    """The patterns of the given sizes that occur on at least `floor` of the wanted rows, grown as plan 78.53 grows them (a candidate needs its parts kept), shut patterns left out.
    -> [{"key", "id", "size", "level", "support", "total", "bytes"}] sorted by size, level, key. Called on the find part's rows only."""
    out: list = []
    allowed = None
    for size in sizes:
        _f, rec = model.pattern_freq(wires, rows, size, list(levels), allowed, goal_kind, threads, 10 ** 9)
        kept = [(h, lv, sup, tot, bytes(b)) for h, lv, sup, tot, b, _parts, _fp in rec if sup >= floor]
        allowed = [h for h, *_r in kept]
        out += [{"key": h, "id": pattern_id(h), "size": size, "level": lv, "support": sup, "total": tot, "bytes": b} for h, lv, sup, tot, b in kept if not has_shut(b)]
    return sorted(out, key=lambda r: (r["size"], r["level"], r["key"]))


# ------------------------------------------------------------------------------------------------------------------------------ what a move does to symbols

def deltas_matrix(positions: list, tied_lists: list, counter: Counter):
    """The events of every tied move as a sparse 0/1 matrix: one row per tied move (the positions in order, each one's tied moves in order), 4 x n_keys columns laid out by event_index (mover up, mover down, other up,
    other down, each over the keys). One engine call for the positions and the positions after every tied move."""
    from scipy import sparse
    wires, rows, before, after = [], [], [], []
    for pos, tied in zip(positions, tied_lists):
        n = len(pos["moves"])
        wires.append((None, list(pos["moves"])))
        rows.append([n])
        b = len(wires) - 1
        for m in tied:
            wires.append((None, list(pos["moves"]) + [m]))
            rows.append([n + 1])
            before.append(b)
            after.append(len(wires) - 1)
    us, them = counter.count(wires, rows)
    us, them = us.tocsr(), them.tocsr()
    if not after:
        return sparse.csr_matrix((0, 4 * len(counter.keys)), dtype=np.float32)
    ub, tb, ua, ta = us[before], them[before], us[after], them[after]
    mover = ta - ub                                  # the mover's count: before = us (the side to move), after = them (the side to move there is the other side)
    other = ua - tb
    blocks = [(mover > 0), (mover < 0), (other > 0), (other < 0)]
    return sparse.hstack([b.astype(np.float32) for b in blocks], format="csr")


def events_of_row(M, row: int, n_cols: int) -> frozenset:
    """The events of one row of deltas_matrix as (column, side, direction)."""
    return frozenset(event_of(int(i), n_cols) for i in M.indices[M.indptr[row]:M.indptr[row + 1]])


def deltas(positions: list, tied_lists: list, counter: Counter) -> list:
    """See the module doc: deltas_matrix read back as one dict {move: frozenset of events} a position."""
    M = deltas_matrix(positions, tied_lists, counter).tocsr()
    n_cols = len(counter.keys)
    out, r = [], 0
    for tied in tied_lists:
        d = {}
        for m in tied:
            d[m] = events_of_row(M, r, n_cols)
            r += 1
        out.append(d)
    return out


def deltas_of(position: dict, tied: list, counter: Counter) -> dict:
    return deltas([position], [tied], counter)[0]


def symbol_pick(position: dict, tied: list, patterns: list, counter: "Counter | None" = None, table: "dict | None" = None) -> str:
    """See the module doc. `table` is deltas_of's output when it was read already. Without patterns, or when nothing separates the moves, the search's first move."""
    if not patterns or len(tied) < 2:
        return tied[0]
    if table is None:
        table = deltas_of(position, tied, counter)
    w = {(p["column"], p["side"], p["dir"]): float(p["weight"]) for p in patterns}
    score = {m: sum(w.get(e, 0.0) for e in table.get(m, ())) for m in tied}
    top = max(score.values())
    best = [m for m in tied if score[m] == top]
    return tied[0] if tied[0] in best else best[0]


# ------------------------------------------------------------------------------------------------------------------------------ test two

def event_index(col: int, side: str, d: str, n_cols: int) -> int:
    return (SIDES.index(side) * 2 + DIRS.index(d)) * n_cols + col


def event_of(index: int, n_cols: int) -> tuple:
    blk, col = divmod(index, n_cols)
    return col, SIDES[blk // 2], DIRS[blk % 2]


def event_matrix(tables: list, n_cols: int):
    """Rows: the tied moves of the positions in order; columns: 4 x n_cols events (event_index). `tables[i]` = (moves in order, {move: events})."""
    from scipy import sparse
    r, c = [], []
    k = 0
    for moves, tab in tables:
        for m in moves:
            for col, s, d in tab.get(m, ()):
                r.append(k)
                c.append(event_index(col, s, d, n_cols))
            k += 1
    return sparse.csr_matrix((np.ones(len(r), np.float64), (r, c)), shape=(k, 4 * n_cols))


def rate_table(E, game: np.ndarray, played: np.ndarray) -> dict:
    """TEST TWO for every event column of E (moves x events, 0/1). `game` the cluster (an integer per move row), `played` True on the played move of each position. -> arrays per event: kp (played moves with the
    event), ko (other tied moves with it), np and no (their totals), the smoothed shares sp and so, the log ratio lr, its standard error se (clusters = games), lo and hi (90 pct)."""
    from scipy import sparse
    E = E.tocsr().astype(np.float64)
    _u, g = np.unique(game, return_inverse=True)
    G = len(_u)
    pl = np.asarray(played, bool)
    Ap = sparse.csr_matrix((np.ones(int(pl.sum())), (g[pl], np.flatnonzero(pl))), shape=(G, len(pl)))
    Ao = sparse.csr_matrix((np.ones(int((~pl).sum())), (g[~pl], np.flatnonzero(~pl))), shape=(G, len(pl)))
    Kp, Ko = (Ap @ E).tocsc(), (Ao @ E).tocsc()
    Np = np.asarray(Ap.sum(axis=1)).ravel()
    No = np.asarray(Ao.sum(axis=1)).ravel()
    k1, k2 = np.asarray(Kp.sum(axis=0)).ravel(), np.asarray(Ko.sum(axis=0)).ravel()
    n1, n2 = float(Np.sum()), float(No.sum())
    s1, s2 = (k1 + PRIOR) / (n1 + 2 * PRIOR), (k2 + PRIOR) / (n2 + 2 * PRIOR)
    sq1 = np.asarray(Kp.multiply(Kp).sum(axis=0)).ravel()
    sq2 = np.asarray(Ko.multiply(Ko).sum(axis=0)).ravel()
    x12 = np.asarray(Kp.multiply(Ko).sum(axis=0)).ravel()
    kn1, kn2 = np.asarray(Kp.T @ Np).ravel(), np.asarray(Ko.T @ No).ravel()          # sum over games of K N, per event
    kn12, kn21 = np.asarray(Kp.T @ No).ravel(), np.asarray(Ko.T @ Np).ravel()
    nn1, nn2, nn12 = float(Np @ Np), float(No @ No), float(Np @ No)
    r1 = sq1 - 2 * s1 * kn1 + s1 ** 2 * nn1                                          # sum of squared cluster residuals
    r2 = sq2 - 2 * s2 * kn2 + s2 ** 2 * nn2
    r12 = x12 - s2 * kn12 - s1 * kn21 + s1 * s2 * nn12
    v1, v2, c12 = r1 / max(n1, 1.0) ** 2, r2 / max(n2, 1.0) ** 2, r12 / max(n1 * n2, 1.0)
    floor = (1.0 / (k1 + PRIOR) - 1.0 / (n1 + 2 * PRIOR)) + (1.0 / (k2 + PRIOR) - 1.0 / (n2 + 2 * PRIOR))         # the variance of two independent binomial shares' log ratio: the clusters may add to it, never take it below
    var = np.maximum(v1 / s1 ** 2 + v2 / s2 ** 2 - 2 * c12 / (s1 * s2), floor)
    se = np.sqrt(var)
    lr = np.log(s1 / s2)
    return {"kp": k1, "np": n1, "ko": k2, "no": n2, "sp": s1, "so": s2, "lr": lr, "se": se, "lo": lr - Z90 * se, "hi": lr + Z90 * se}


# ------------------------------------------------------------------------------------------------------------------------------ test one

def _fit_addon(off: np.ndarray, y: np.ndarray, f: np.ndarray, w: np.ndarray, iters: int = 8, free: "np.ndarray | None" = None) -> np.ndarray:
    """The added terms on rows with weights `w`: for the win and the loss class (the draw class is the reference) a shift of its score and a term per side's count, theta = [win: shift, mover's count, other's count,
    loss: shift, mover's count, other's count], by Newton steps on the log likelihood of the softmax of (the head's scores `off` + the added terms) at the observed class; a ridge on the count terms only (the shifts are
    free, so the head's calibration on these rows does not leak into a term). `off` rows x 3 (win, draw, loss), `f` rows x 2 the counts. PLAN 78.81: `free` (rows x e) are further columns with their own free coefficients in
    each class (a shift for the generated rows, whose head scores are not those of human games), laid out after the three; the step is damped (halved until the objective does not fall), as an undamped Newton step
    diverged on rows whose head scores are far from their results."""
    X = np.column_stack([np.ones(len(f)), f] + ([] if free is None else [free]))
    k0 = X.shape[1]
    th = np.zeros(2 * k0)
    ks = (0, 2)
    ind = np.stack([(y == 0), (y == 2)], axis=1).astype(float)
    ridge = np.array(([1e-6, RIDGE, RIDGE] + [1e-6] * (k0 - 3)) * 2)
    for _ in range(iters):
        z = off.copy()
        for a, k in enumerate(ks):
            z[:, k] += X @ th[k0 * a:k0 * (a + 1)]
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad = np.zeros(2 * k0)
        H = np.zeros((2 * k0, 2 * k0))
        for a, k in enumerate(ks):
            grad[k0 * a:k0 * (a + 1)] = ((ind[:, a] - p[:, k]) * w) @ X
            for b, k2 in enumerate(ks):
                wk = w * p[:, k] * ((1.0 if k == k2 else 0.0) - p[:, k2])
                H[k0 * a:k0 * (a + 1), k0 * b:k0 * (b + 1)] = -(X * wk[:, None]).T @ X
        grad -= ridge * th
        H -= np.diag(ridge)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        cur = _addon_objective(th, off, ind, X, w, ridge, ks, k0)
        scale = 1.0
        while scale > 1e-4 and _addon_objective(th - scale * step, off, ind, X, w, ridge, ks, k0) < cur - 1e-12:
            scale /= 2.0
        th = th - scale * step
        if np.max(np.abs(scale * step)) < 1e-6:
            break
    return th


def _addon_objective(th: np.ndarray, off: np.ndarray, ind: np.ndarray, X: np.ndarray, w: np.ndarray, ridge: np.ndarray, ks: tuple, k0: int) -> float:
    """The weighted log likelihood of the added terms less the ridge (what _fit_addon climbs)."""
    z = off.copy()
    for a, k in enumerate(ks):
        z[:, k] += X @ th[k0 * a:k0 * (a + 1)]
    m = z.max(axis=1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))
    at = z[:, 0] * ind[:, 0] + z[:, 2] * ind[:, 1] + z[:, 1] * (1 - ind[:, 0] - ind[:, 1])
    return float(((at - lse) * w).sum() - 0.5 * (ridge * th * th).sum())


def term_table(off: np.ndarray, y: np.ndarray, game: np.ndarray, us, them, columns: list, draws: int, seed: int, free: "np.ndarray | None" = None) -> dict:
    """TEST ONE for the pattern columns `columns` (indices into the sparse matrices `us` and `them`, head rows x patterns; `us` the count of the side to move, `them` of the other). `off` the held head's scores for the rows
    (rows x 3), `y` the result of the side to move (0 win, 1 draw, 2 loss), `game` the game of each row. -> {column: {"term": [mover's, other's], "lo": [..], "hi": [..], "rows": rows with the pattern}}, the
    term of a side the log-odds of a win against a loss per extra instance (win class less loss class), the band 90 pct over `draws` resamples of the games. `free`: see _fit_addon."""
    us, them = us.tocsc(), them.tocsc()
    _u, g = np.unique(game, return_inverse=True)
    G = len(_u)
    rng = np.random.default_rng(seed)
    mult = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(draws)]).astype(float)        # draws x games
    out = {}
    for c in columns:
        u = us[:, c].toarray().ravel()
        v = them[:, c].toarray().ravel()
        res = {"rows": int(((u > 0) | (v > 0)).sum())}
        if res["rows"] == 0:
            out[c] = dict(res, term=[0.0, 0.0], lo=[0.0, 0.0], hi=[0.0, 0.0])
            continue
        f = np.stack([u, v], axis=1).astype(float)
        k0 = 3 + (0 if free is None else free.shape[1])
        th = _fit_addon(off, y, f, np.ones(len(y)), free=free)
        t = np.array([th[1] - th[k0 + 1], th[2] - th[k0 + 2]])
        bs = np.array([_fit_addon(off, y, f, mult[d][g], free=free) for d in range(draws)])
        tb = np.stack([bs[:, 1] - bs[:, k0 + 1], bs[:, 2] - bs[:, k0 + 2]], axis=1)
        lo, hi = np.percentile(tb, [5, 95], axis=0)
        out[c] = dict(res, term=[float(t[0]), float(t[1])], lo=[float(lo[0]), float(lo[1])], hi=[float(hi[0]), float(hi[1])])
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the rule

def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def occurs_enough(two: dict, i: int, rows_with: int) -> bool:
    return bool(two["kp"][i] + two["ko"][i] >= FLOOR_MOVES and two["kp"][i] >= FLOOR_EACH and two["ko"][i] >= FLOOR_EACH and rows_with >= FLOOR_ROWS)


def events_over_floor(two: dict, rows_with: dict, n_cols: int) -> list:
    """The event indices that meet the floors of the rule (i): the events TRIED. `rows_with[column]` the head rows where the pattern occurs."""
    return [i for i in range(4 * n_cols) if occurs_enough(two, i, rows_with.get(i % n_cols, 0))]


def find(two: dict, one: dict, n_cols: int) -> "tuple[list, dict]":
    """THE RULE on the find part. `two`: rate_table's output over all 4 x n_cols events; `one`: term_table's output by pattern column (it holds the columns of the events over the floors). -> (picked, info):
    `picked` = [{"column", "side", "dir", "weight", ...}] and info["tried"] the count of events that met the floors. Reads nothing of the confirm part."""
    picked, tried = [], 0
    for i in range(4 * n_cols):
        col, side, d = event_of(i, n_cols)
        t = one.get(col)
        if t is None or not occurs_enough(two, i, t["rows"]):
            continue
        tried += 1
        s = SIDES.index(side)
        if not (two["lo"][i] > 0 or two["hi"][i] < 0):
            continue
        if not (t["lo"][s] > 0 or t["hi"][s] < 0):
            continue
        sd = 1 if d == "up" else -1
        if _sign(t["term"][s]) * sd != _sign(two["lr"][i]):
            continue
        picked.append({"column": col, "side": side, "dir": d, "weight": float(two["lr"][i]), "sign_two": _sign(two["lr"][i]), "sign_one": _sign(t["term"][s]),
                       "f_lr": float(two["lr"][i]), "f_lo": float(two["lo"][i]), "f_hi": float(two["hi"][i]), "f_kp": float(two["kp"][i]), "f_ko": float(two["ko"][i]),
                       "f_sp": float(two["sp"][i]), "f_so": float(two["so"][i]), "f_term": float(t["term"][s]), "f_tlo": float(t["lo"][s]), "f_thi": float(t["hi"][s])})
    return picked, {"tried": tried}


def confirm(picked: list, two: dict, one: dict) -> list:
    """Read once on the confirm part, for the picked events only: `two` is rate_table's output over the picked events IN ORDER (the move-by-event matrix holds only those columns), `one` term_table's output by pattern
    column for the picked patterns. The event passes when test two's interval excludes equality with the sign it had on the find part and test one's band for the side excludes zero with the sign it had.
    Returns the picked records with the confirm numbers added and "passes"."""
    out = []
    for j, p in enumerate(picked):
        s = SIDES.index(p["side"])
        t = one.get(p["column"], {"term": [0, 0], "lo": [0, 0], "hi": [0, 0], "rows": 0})
        ok2 = bool(two["lo"][j] > 0) if p["sign_two"] > 0 else bool(two["hi"][j] < 0)
        ok1 = bool(t["lo"][s] > 0) if p["sign_one"] > 0 else bool(t["hi"][s] < 0)
        out.append(dict(p, c_lr=float(two["lr"][j]), c_lo=float(two["lo"][j]), c_hi=float(two["hi"][j]), c_kp=float(two["kp"][j]), c_ko=float(two["ko"][j]), c_sp=float(two["sp"][j]), c_so=float(two["so"][j]),
                        c_term=float(t["term"][s]), c_tlo=float(t["lo"][s]), c_thi=float(t["hi"][s]), c_rows=int(t["rows"]), passes=bool(ok2 and ok1)))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ PLAN 78.81: one weight per reason

FOLDS = 5                              # the parts of the games inside FIND the penalty is chosen on
DUP_THRESHOLD = 0.9                    # two events whose carrier sets have a Jaccard index at or above this are duplicates (stated before the run)
LAMBDA_STEPS = 10                      # the path of penalties: from the smallest that gives no event weight down to a thousandth of it
RIDGE_CONTROLS = 1e-3                  # the controls' ridge (on the mean log likelihood a position)


@dataclass
class Choice:
    """The rows of a choice fit (see the module doc)."""
    E: object
    C: np.ndarray
    control_names: list
    pos: np.ndarray
    played: np.ndarray
    game: np.ndarray
    part: np.ndarray
    source: np.ndarray
    units: "np.ndarray | None" = None                      # PLAN 78.88: the units on the board at the row's position (None where the caller did not say)
    fill_ok: bool = False                                  # PLAN 78.89: the builder says generated rows (our own self-play games, marked source "generated", FIND only) may be read by the fit

    def n_positions(self) -> int:
        return int(self.pos[-1]) + 1 if len(self.pos) else 0

    def subset(self, positions: np.ndarray) -> "Choice":
        """The rows of the positions in `positions` (a boolean over positions or indices), renumbered from zero."""
        positions = np.asarray(positions)
        keep = positions if positions.dtype == bool else np.isin(np.arange(self.n_positions()), positions)
        rows = keep[self.pos]
        new = np.cumsum(keep) - 1
        return Choice(self.E.tocsr()[rows], self.C[rows], self.control_names, new[self.pos[rows]], self.played[rows], self.game[rows], self.part[rows], self.source[rows],
                      None if self.units is None else self.units[rows], self.fill_ok)


def require_find(rows: Choice) -> None:
    """See the module doc: the fit reads FIND rows of human games only."""
    if len(rows.part) and not bool(np.all(np.asarray(rows.part) == "find")):
        raise ValueError("a row that is not of the FIND part was offered to the fit")
    if len(rows.source) and not rows.fill_ok and not bool(np.all(np.asarray(rows.source) == "human")):
        raise ValueError("a row of a generated game was offered to the fit")


def drop_duplicates(rows: Choice, order: list, threshold: float = DUP_THRESHOLD) -> "tuple[list, dict]":
    """See the module doc. Greedy over the columns in the order of `order` (the smaller pattern first): a column is kept when its carrier set's Jaccard index with every kept column is below `threshold`."""
    require_find(rows)
    E = rows.E.tocsc().astype(np.float64)
    n = E.shape[1]
    carriers = np.asarray(E.sum(axis=0)).ravel()
    G = (E.T @ E).tocsr()
    kept: list = []
    seen: set = set()
    info = {"in": n, "empty": int((carriers == 0).sum()), "exact": 0, "near": 0}
    for j in sorted(range(n), key=lambda j: order[j]):
        if carriers[j] == 0:
            continue
        row = G.getrow(j)
        dup = 0
        for i, g in zip(row.indices, row.data):
            if i in seen and i != j:
                jac = g / (carriers[i] + carriers[j] - g)
                if jac >= threshold:
                    dup = 2 if (g == carriers[i] and g == carriers[j]) else 1
                    break
        if dup == 2:
            info["exact"] += 1
        elif dup == 1:
            info["near"] += 1
        else:
            kept.append(j)
            seen.add(j)
    info["kept"] = len(kept)
    return sorted(kept), info


def _groups(pos: np.ndarray, played: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """(row index [P, T] padded with 0, mask [P, T], the place of the played move inside its position [P]) for contiguous positions."""
    P = int(pos[-1]) + 1
    sizes = np.bincount(pos, minlength=P)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    T = int(sizes.max())
    idx = np.zeros((P, T), np.int64)
    mask = np.zeros((P, T), bool)
    for t in range(T):
        has = sizes > t
        idx[has, t] = start[has] + t
        mask[has, t] = True
    tgt = np.zeros(P, np.int64)
    pr = np.flatnonzero(played)
    tgt[pos[pr]] = pr - start[pos[pr]]
    return idx, mask, tgt


class _Problem:
    """One conditional logit problem on dense float32 features, the arithmetic in torch."""

    def __init__(self, X: np.ndarray, pos: np.ndarray, played: np.ndarray):
        import torch
        self.torch = torch
        self.X = torch.from_numpy(np.ascontiguousarray(X, np.float32))
        idx, mask, tgt = _groups(pos, played)
        self.idx, self.mask, self.tgt = torch.from_numpy(idx), torch.from_numpy(mask), torch.from_numpy(tgt)
        self.P = len(tgt)

    def scores(self, w):
        s = self.X @ w
        sg = s[self.idx]
        return sg.masked_fill(~self.mask, float("-inf"))

    def nll(self, w):
        sg = self.scores(w)
        lse = self.torch.logsumexp(sg, dim=1)
        return (lse - sg.gather(1, self.tgt[:, None])[:, 0]).mean()

    def loglik(self, w) -> float:
        with self.torch.no_grad():
            return -float(self.nll(w))

    def grad(self, w):
        w = w.detach().requires_grad_(True)
        v = self.nll(w)
        (g,) = self.torch.autograd.grad(v, w)
        return g

    def lipschitz(self) -> float:
        torch = self.torch
        v = torch.randn(self.X.shape[1], generator=torch.Generator().manual_seed(0))
        for _ in range(40):
            v = self.X.T @ (self.X @ v)
            v = v / (v.norm() + 1e-12)
        sigma2 = float((self.X @ v).norm() ** 2)
        return 0.5 * sigma2 / max(self.P, 1) + 1e-9                  # diag(p) - p p^T has no eigenvalue above 1/2

    def fit(self, lam: float, pen, w0, ridge: float, lip: float, iters: int = 1500, tol: float = 1e-5):
        """FISTA (with the gradient restart) on mean negative log likelihood + lam |w_pen|_1 + ridge/2 |w_rest|^2. `pen` a 0/1 float vector (1: L1-penalised). Stops when the proximal gradient mapping
        (the step's move over the step size) is below `tol` in every coordinate."""
        torch = self.torch
        w, y, t = w0.clone(), w0.clone(), 1.0
        step = 1.0 / (lip + ridge)
        for _ in range(iters):
            g = self.grad(y) + ridge * y * (1 - pen)
            z = y - step * g
            z = torch.where(pen > 0, torch.sign(z) * torch.clamp(z.abs() - step * lam, min=0.0), z)
            if float((y - z).abs().max()) / step < tol:
                w = z
                break
            if float(((y - z) * (z - w)).sum()) > 0:                                  # the momentum points back uphill: restart it
                t = 1.0
            t2 = (1 + (1 + 4 * t * t) ** 0.5) / 2
            y = z + ((t - 1) / t2) * (z - w)
            w, t = z, t2
        return w


def _features(rows: Choice, controls: tuple, events: "list | None") -> "tuple[np.ndarray, np.ndarray, list, list]":
    cols = list(range(rows.E.shape[1])) if events is None else [int(c) for c in events]
    parts = []
    if cols:
        parts.append(rows.E.tocsc()[:, cols].toarray().astype(np.float32))
    cidx = [rows.control_names.index(c) for c in controls]
    if cidx:
        parts.append(rows.C[:, cidx].astype(np.float32))
    X = np.hstack(parts) if parts else np.zeros((len(rows.pos), 0), np.float32)
    pen = np.concatenate([np.ones(len(cols)), np.zeros(len(cidx))]).astype(np.float32)
    return X, pen, cols, list(controls)


def position_units(rows: Choice) -> np.ndarray:
    """PLAN 78.88: the units on the board at each position (the value on the position's first row)."""
    if rows.units is None:
        raise ValueError("the rows carry no unit counts")
    P = rows.n_positions()
    first = np.concatenate([[0], np.flatnonzero(np.diff(rows.pos)) + 1]) if len(rows.pos) else np.zeros(0, int)
    out = np.zeros(P, int)
    out[rows.pos[first]] = np.asarray(rows.units)[first]
    return out


def fit_choice(rows: Choice, controls: tuple = (), events: "list | None" = None, folds: int = FOLDS, seed: int = 0, band: "tuple | None" = None) -> dict:
    """See the module doc. PLAN 78.88: `band` = (low, high) fits on the positions whose units on the board lie in it (inclusive) and nothing else; the folds are then cut from the games of those positions."""
    require_find(rows)
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    if band is not None:
        pu = position_units(rows)
        rows = rows.subset((pu >= band[0]) & (pu <= band[1]))
        folds = max(2, min(folds, len(np.unique(rows.game)))) if len(rows.pos) else folds
        if rows.n_positions() == 0:
            return {"lambda": 0.0, "events": [], "weights": [], "controls": {}, "cv": [], "n_events": 0, "positions": 0, "band": list(band)}
    X, pen_np, cols, cnames = _features(rows, tuple(controls), events)
    P = rows.n_positions()
    out = {"lambda": 0.0, "events": [], "weights": [], "controls": {}, "cv": [], "n_events": len(cols), "positions": P}
    if X.shape[1] == 0:
        return dict(out, loglik_uniform=True)
    pen = torch.from_numpy(pen_np)
    prob = _Problem(X, rows.pos, rows.played)
    lip = prob.lipschitz()
    w0 = torch.zeros(X.shape[1])
    if len(cols) == 0:                                                       # controls alone: no penalty, one fit
        w = prob.fit(0.0, pen, w0, RIDGE_CONTROLS, lip)
        return dict(out, controls={c: float(w[i]) for i, c in enumerate(cnames)}, loglik=prob.loglik(w))
    base = w0
    if cnames:                                                               # the controls alone first, so no event enters the starting point
        pc = _Problem(X[:, len(cols):], rows.pos, rows.played)
        wc = pc.fit(0.0, torch.zeros(len(cnames)), torch.zeros(len(cnames)), RIDGE_CONTROLS, pc.lipschitz())
        base = torch.cat([torch.zeros(len(cols)), wc])
    g0 = prob.grad(base)
    lam_max = float((g0.abs() * pen).max()) + 1e-12
    grid = [lam_max * 10.0 ** (-3.0 * i / (LAMBDA_STEPS - 1)) for i in range(LAMBDA_STEPS)]
    # the penalty by folds of games inside the rows
    games = np.unique(rows.game)
    rng = np.random.default_rng(seed)
    fold_of_game = dict(zip(games.tolist(), rng.permutation(len(games)) % folds))
    fold_pos = np.zeros(P, int)
    fold_pos[rows.pos] = [fold_of_game[int(g)] for g in rows.game]
    held = np.zeros((folds, len(grid)))
    n_held = np.zeros(folds)
    for k in range(folds):
        tr = fold_pos != k
        te = ~tr
        if te.sum() == 0 or tr.sum() == 0:
            held[k] = np.nan
            continue
        a, b = rows.subset(tr), rows.subset(te)
        Xa, _p, _c, _n = _features(a, tuple(controls), events)
        Xb, _p2, _c2, _n2 = _features(b, tuple(controls), events)
        pa, pb = _Problem(Xa, a.pos, a.played), _Problem(Xb, b.pos, b.played)
        la = pa.lipschitz()
        w = base.clone()
        for i, lam in enumerate(grid):
            w = pa.fit(lam, pen, w, RIDGE_CONTROLS, la)
            held[k, i] = pb.loglik(w)
        n_held[k] = te.sum()
    mean = np.nanmean(held, axis=0)
    se = np.nanstd(held, axis=0, ddof=1) / np.sqrt(max(1, np.sum(~np.isnan(held[:, 0])))) if folds > 1 else np.zeros(len(grid))
    best = int(np.nanargmax(mean))
    ok = [i for i in range(len(grid)) if mean[i] >= mean[best] - se[best]]
    pick = min(ok)                                                           # the largest penalty (the grid falls) within one standard error of the best
    w = base.clone()
    for i in range(pick + 1):
        w = prob.fit(grid[i], pen, w, RIDGE_CONTROLS, lip)
    wn = w.numpy()
    nz = [i for i in range(len(cols)) if wn[i] != 0.0]
    return dict(out, **{"lambda": float(grid[pick]), "events": [cols[i] for i in nz], "weights": [float(wn[i]) for i in nz],
                        "controls": {c: float(wn[len(cols) + j]) for j, c in enumerate(cnames)},
                        "cv": [(float(grid[i]), float(mean[i]), float(se[i])) for i in range(len(grid))], "best": best, "pick": pick, "loglik": prob.loglik(w)})


def choice_scores(rows: Choice, fit: dict, controls: bool = False) -> np.ndarray:
    """The score of every row: the sum of the weights of the events the move carries (plus the controls' weights when asked)."""
    s = np.zeros(len(rows.pos))
    if fit["events"]:
        s += rows.E.tocsc()[:, fit["events"]] @ np.asarray(fit["weights"], float)
    if controls:
        for name, w in fit["controls"].items():
            s += rows.C[:, rows.control_names.index(name)] * w
    return s


def score_choice(rows: Choice, fit: dict, controls: bool = False) -> np.ndarray:
    """See the module doc: per position the index inside the position of the highest-scoring tied move; the first one on a tie (the search's first move), so silence is the search's own move."""
    s = choice_scores(rows, fit, controls)
    P = rows.n_positions()
    sizes = np.bincount(rows.pos, minlength=P)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = np.zeros(P, int)
    for p in range(P):
        seg = s[start[p]:start[p] + sizes[p]]
        out[p] = int(np.argmax(seg)) if seg.max() > seg[0] else 0
    return out


def score_choice_by_band(rows: Choice, fits: dict, bands: list, controls: bool = False) -> np.ndarray:
    """PLAN 78.88, see the module doc: the mixture gated by the band of the position's units. A band with no fit (a key missing from `fits`) is silent: the search's first move."""
    P = rows.n_positions()
    pu = position_units(rows)
    sizes = np.bincount(rows.pos, minlength=P)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = np.zeros(P, int)
    for name, lo, hi in bands:
        if name not in fits:
            continue
        s = choice_scores(rows, fits[name], controls)
        for p in np.flatnonzero((pu >= lo) & (pu <= hi)):
            seg = s[start[p]:start[p] + sizes[p]]
            out[p] = int(np.argmax(seg)) if seg.max() > seg[0] else 0
    return out


# ------------------------------------------------------------------------------------------------------------------------------ PLAN 78.88: twins grouped, the unit borne on

GROUP_SIM = 0.5                        # the one fixed threshold of the grouping rule (module doc): not tuned


def groups_of(reasons: list, rows: Choice, threshold: float = GROUP_SIM) -> list:
    """See the module doc. Average linkage on the Jaccard index of the carrier sets over `rows` (FIND only), merging only reasons of one side, one direction and one sign of weight."""
    require_find(rows)
    E = rows.E.tocsc().astype(np.float64)
    n = E.shape[1]
    if n != len(reasons):
        raise ValueError("one column of rows.E for each reason")
    car = np.asarray(E.sum(axis=0)).ravel()
    G = (E.T @ E).toarray()
    J = G / np.maximum(car[:, None] + car[None, :] - G, 1.0)
    J[(car[:, None] == 0) | (car[None, :] == 0)] = 0.0
    key = [(r["side"], r["dir"], 1 if float(r["weight"]) > 0 else -1) for r in reasons]
    clusters = [[i] for i in range(n)]
    while True:
        best, pair = threshold, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if key[clusters[a][0]] != key[clusters[b][0]]:
                    continue
                s = float(J[np.ix_(clusters[a], clusters[b])].mean())
                if s >= best and (pair is None or s > best):
                    best, pair = s, (a, b)
        if pair is None:
            break
        a, b = pair
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
    return sorted(clusters, key=lambda c: c[0])


def group_matrix(E, groups: list):
    """See the module doc: one column for each group, 1 where any member's column is 1."""
    from scipy import sparse
    E = E.tocsc()
    cols = []
    for g in groups:
        s = E[:, g].sum(axis=1)
        cols.append(sparse.csc_matrix((np.asarray(s).ravel() > 0).astype(np.float32)[:, None]))
    return sparse.hstack(cols, format="csr") if cols else sparse.csr_matrix((E.shape[0], 0), dtype=np.float32)


CELL_WORDS = ("the unit of theirs is not guarded and the bearing unit is not borne on", "the unit of theirs is not guarded and the bearing unit is borne on by a unit of theirs",
              "the unit of theirs is guarded and the bearing unit is not borne on", "the unit of theirs is guarded and the bearing unit is borne on by a unit of theirs")


def bear_form(p: L2.Pattern) -> bool:
    """One `bear` atom from the first unit to the second, the first the side's own and the second the other side's (a unit of the mover's bears on a unit of theirs, for the mover's events)."""
    return p.size == 1 and p.atoms[0][0] == "bear" and p.atoms[0][2] == "u" and p.atoms[0][1] == 0 and p.atoms[0][3] == 1 and list(p.sides) == [True, False]


def split_borne(reasons: list, kinds: list) -> list:
    """See the module doc. `reasons` carry "bytes" (the pattern's encoding in hexadecimal) and "side"; `kinds` the kind words (the decoder needs none for a split decision, kept for the layer-3 words)."""
    out = []
    for i, r in enumerate(reasons):
        if r["side"] == "mover" and bear_form(L2.decode(bytes.fromhex(r["bytes"]))):
            for c in range(4):
                out.append(dict(r, cell=c, parent=i, words=r["words"] + "; " + CELL_WORDS[c]))
        else:
            out.append(dict(r, cell=None, parent=i))
    return out


def cell_counts(items: list, units: dict, p: L2.Pattern, kinds: list) -> "tuple[list, list]":
    """See the module doc. The instances are the matcher's; the two facts are read from the graph's own `bear` items between units."""
    from catspace.incubator.analyst.discovery import graph_layer2_match as GM
    bears = [it for it in items if it[0] == "bear" and it[2][0] == "u"]
    guarded, borne = set(), set()
    for _ty, actor, (_s, tgt), _ln, _b in bears:
        if actor == tgt:
            continue
        if units[actor]["side"] == units[tgt]["side"]:
            guarded.add(tgt)
        else:
            borne.add(tgt)
    s0, s1 = GM.match_binds(bears, units, p, kinds)
    out = []
    for found in (s0, s1):
        c = [0, 0, 0, 0]
        for fs in found:
            u0, u1 = sorted(fs)[0]
            c[2 * int(u1 in guarded) + int(u0 in borne)] += 1
        out.append(c)
    return out[0], out[1]


def cell_events(before: list, after: list) -> "tuple[list, list]":
    """The cells whose count rose and those whose count fell for the mover: `before` the position's first, `after` the second of the position after the move."""
    return [c for c in range(4) if after[c] > before[c]], [c for c in range(4) if after[c] < before[c]]


# ------------------------------------------------------------------------------------------------------------------------------ PLAN 78.81: layer one, a first reading

class Reach:
    """What ahead() reads: the graph model of a game and its variant, the goal kind and the kind words (all handed in)."""

    def __init__(self, model, variant: str, goal_kind, kinds: list, facts=None):
        self.model, self.variant, self.goal_kind, self.kinds = model, variant, goal_kind, list(kinds)
        self.static = model.static_part()
        self.facts = facts

    def legal(self, moves: list) -> list:
        import cs_engine as CE
        st = CE.state(self.variant, None, list(moves))
        return list(st[2]) if st[1] is not None and st[3] is None else []

    def text(self, moves: list) -> str:
        import cs_engine as CE
        return CE.state(self.variant, None, list(moves))[0]

    def items(self, text: str):
        from catspace.incubator.analyst.discovery import graph_layer1 as GL
        from catspace.incubator.analyst.discovery import graph_layer2_match as GM
        return GM.graph_items(GL.graph(self.model, text, self.static), self.goal_kind)


def reach_rows(env: Reach, positions: list, counter: Counter) -> list:
    """See the module doc: [{"legal": [moves], "makes": {key: [moves whose event "mover up" holds]}}] per position, the keys of the counter."""
    legal = [env.legal(p["moves"]) for p in positions]
    M = deltas_matrix(positions, legal, counter).tocsc()
    n = len(counter.keys)
    out, r = [], 0
    for lg in legal:
        up = M[r:r + len(lg), :n].tocsr() if lg else None                  # the first block: mover up
        makes = {}
        for j, key in enumerate(counter.keys):
            col = up[:, j].toarray().ravel() if up is not None else np.zeros(0)
            makes[key] = [lg[i] for i in np.flatnonzero(col > 0)]
        out.append({"legal": lg, "makes": makes})
        r += len(lg)
    return out


def _unit_ok(un: dict, u: int, p: L2.Pattern, kinds: list, side: int) -> bool:
    mine = un["mine"] if side == 0 else not un["mine"]
    return not (mine != p.sides[u] or (p.level >= 1 and un["goal"] != p.goal[u]) or (p.level == 2 and un["edge"] != p.edge[u]) or (p.level == 3 and kinds[p.kind[u]] != un["kind"]))


def one_short(items: list, units: dict, p: L2.Pattern, kinds: list, side: int) -> "tuple[set, set]":
    """See the module doc. For a pattern of size 2 with atoms (A, B): an item that matches atom A, with the unit variables of A bound to units that meet the pattern's attributes, is a PARTIAL instance (key: the atom number
    and the units of that atom's unit variables); it is COMPLETE when some other item matches the other atom so that the two bind consistently (the same objects for the shared variables, injectively) and every unit meets
    its attributes, and ONE FACT SHORT when no such item exists."""
    from catspace.incubator.analyst.discovery import graph_layer2_match as GM
    if p.size != 2:
        return set(), set()
    seen, full = set(), set()
    for a in (0, 1):
        atom_a = p.atoms[a]
        uvars = sorted({v for v in (atom_a[1], atom_a[3] if atom_a[2] == "u" else None, atom_a[5]) if v is not None})
        for it in items:
            bound = GM._bind([atom_a], (it,))
            if not bound:
                continue
            ub = {v[1]: o for v, o in bound.items() if v[0] == "u"}
            if not all(_unit_ok(units[ub[u]], u, p, kinds, side) for u in uvars if u in ub):
                continue
            key = (a, tuple((u, ub[u]) for u in uvars))
            done = False
            for jt in items:
                if jt is it:
                    continue
                order = (it, jt) if a == 0 else (jt, it)
                bd = GM._bind(p.atoms, order)
                if not bd:
                    continue
                uv = {v[1]: o for v, o in bd.items() if v[0] == "u"}
                if all(u in uv and _unit_ok(units[uv[u]], u, p, kinds, side) for u in range(p.n_units())):
                    done = True
                    break
            seen.add(key)
            if done:
                full.add(key)
    return seen - full, full


def _follow(keys: set, frm: str, to: str) -> set:
    """The keys with the unit that stood on `frm` now standing on `to`."""
    def f(uid: str) -> str:
        return uid.rsplit(":", 1)[0] + ":" + to if uid.endswith(":" + frm) else uid
    return {(a, tuple((u, f(o)) for u, o in us)) for a, us in keys}


def nearer(env: Reach, text: str, move: str, pattern: L2.Pattern, before: "tuple | None" = None) -> "bool | None":
    """See the module doc. `before` = the (short, full) keys at `text` for the side to move when already read (the same position, many moves). None where the move's units cannot be followed."""
    import cs_engine as CE
    if env.facts is None:
        return None
    if before is None:
        items, units = env.items(text)
        before = one_short(items, units, pattern, env.kinds, 0)
    pl = env.facts.replay(text, [move])
    if not pl or not pl[0]["from"] or not pl[0]["to"]:
        return None
    after_text = CE.state(env.variant, text, [move])[0]
    items2, units2 = env.items(after_text)
    short2, full2 = one_short(items2, units2, pattern, env.kinds, 1)         # after the move the mover is the other side of the position
    old = _follow(before[0] | before[1], pl[0]["from"], pl[0]["to"])
    return bool(short2 - old)


def ahead(position: dict, reason: dict, limit: int, env: Reach, rows: "list | None" = None) -> "int | None":
    """See the module doc. `rows` (optional) = reach_rows' results for the earlier positions already read, by k (1-based), so a batch need not ask again."""
    if reason.get("side") != "mover" or reason.get("dir") != "up":
        return None
    moves = list(position["moves"])
    n = len(moves)
    if n - 2 < 0:
        return None
    run = 0
    for k in range(1, limit + 1):
        at = n - 2 * k
        if at < 0:
            return run if run else None
        if rows is not None:
            makes = rows[k - 1]["makes"].get(reason["key"], [])
        else:
            counter = Counter(env.model, [reason["key"]], env.goal_kind, 2, (0, 1, 2, 3), 1)
            makes = reach_rows(env, [{"moves": moves[:at]}], counter)[0]["makes"].get(reason["key"], [])
        if not makes:
            break
        run += 1
    return run
