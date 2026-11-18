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
"""
from __future__ import annotations

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

def _fit_addon(off: np.ndarray, y: np.ndarray, f: np.ndarray, w: np.ndarray, iters: int = 8) -> np.ndarray:
    """The added terms on rows with weights `w`: for the win and the loss class (the draw class is the reference) a shift of its score and a term per side's count, theta = [win: shift, mover's count, other's count,
    loss: shift, mover's count, other's count], by Newton steps on the log likelihood of the softmax of (the head's scores `off` + the added terms) at the observed class; a ridge on the count terms only (the shifts are
    free, so the head's calibration on these rows does not leak into a term). `off` rows x 3 (win, draw, loss), `f` rows x 2 the counts."""
    X = np.column_stack([np.ones(len(f)), f])
    th = np.zeros(6)
    ks = (0, 2)
    ind = np.stack([(y == 0), (y == 2)], axis=1).astype(float)
    ridge = np.array([1e-6, RIDGE, RIDGE] * 2)
    for _ in range(iters):
        z = off.copy()
        for a, k in enumerate(ks):
            z[:, k] += X @ th[3 * a:3 * a + 3]
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad = np.zeros(6)
        H = np.zeros((6, 6))
        for a, k in enumerate(ks):
            grad[3 * a:3 * a + 3] = ((ind[:, a] - p[:, k]) * w) @ X
            for b, k2 in enumerate(ks):
                wk = w * p[:, k] * ((1.0 if k == k2 else 0.0) - p[:, k2])
                H[3 * a:3 * a + 3, 3 * b:3 * b + 3] = -(X * wk[:, None]).T @ X
        grad -= ridge * th
        H -= np.diag(ridge)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        th = th - step
        if np.max(np.abs(step)) < 1e-6:
            break
    return th


def term_table(off: np.ndarray, y: np.ndarray, game: np.ndarray, us, them, columns: list, draws: int, seed: int) -> dict:
    """TEST ONE for the pattern columns `columns` (indices into the sparse matrices `us` and `them`, head rows x patterns; `us` the count of the side to move, `them` of the other). `off` the held head's scores for the rows
    (rows x 3), `y` the result of the side to move (0 win, 1 draw, 2 loss), `game` the game of each row. -> {column: {"term": [mover's, other's], "lo": [..], "hi": [..], "rows": rows with the pattern}}, the
    term of a side the log-odds of a win against a loss per extra instance (win class less loss class), the band 90 pct over `draws` resamples of the games."""
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
        th = _fit_addon(off, y, f, np.ones(len(y)))
        t = np.array([th[1] - th[4], th[2] - th[5]])
        bs = np.array([_fit_addon(off, y, f, mult[d][g]) for d in range(draws)])
        tb = np.stack([bs[:, 1] - bs[:, 4], bs[:, 2] - bs[:, 5]], axis=1)
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
