"""PLAN 78.67 A: THE NUMBER OF A MOVE, from the frame read of the position after it and the event heads of 78.65. A sandbox: nothing is wired into any player of the wake, nothing is written to rules/learned or
data/derived. Game-free: a kind, a role or a place is a word read from the engine at run time. No engine evaluation, no labeller, no external model: the heads are fitted on human games and their results.

THE NUMBER of a root move: in the position after it, the chance that the side that played it next wins and keeps a unit BEFORE the other side does, less the chance of the other side first (the race of
78.65 stage E, read from the two heads of the class "keeps", one for each side, with the tempo of the steps). `trees`: the boosted-trees heads; `linear`: the linear hazard heads of 78.64.

THE INPUTS of a position are those of the heads' fit, read by the same code (the kinds and statements of the layer-1 table and the statement sheet, the edge distances of the units, the frame read's
class counts); the ratings are not known at play time: the mean rating of the fit rows and a rating difference of nothing stand in (a stated limit), and the steps left in the record are the plies
left to the game's own ply cap.

  .venv/bin/python -m catspace.incubator.analyst.discovery.tiebreak_number fit [--out-dir D]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.incubator.analyst.discovery import frames_heads as FH
from catspace.incubator.analyst.discovery import frames_measure as OLD
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.incubator.analyst.discovery import tactical_tables as TT
from catspace.io import paths

OUT = ("tiebreak_78_67", "heads")
CORE = ("keeps",)
KEEP = [0, 1, 2, 3, 4]                                      # the unit record's columns the heads read (the fate column is the row's future and is never read)
BLOCKS = ("classes", "trap", "lost kinds", "frames")


def heads_file(out_dir: "str | Path | None" = None) -> Path:
    return Path(out_dir or paths.experiment(*OUT)) / "heads.pkl"


# ------------------------------------------------------------------------------------------------------------------------------ the fit (once; the 78.65 run kept no fitted head)

def fit(out_dir: "str | Path | None" = None, say=print) -> Path:
    """Refit the two heads of the class `keeps` (each side) on the 9000 games of 78.65 with the code of 78.65 (linear: EH.fit_best; trees: FH.fit_tree) and save them, the names of the columns they read
    and the constants the play-time inputs need."""
    t0 = time.perf_counter()
    d = FH.build(Path(paths.experiment(*FH.STATE)), say)
    rows, sample, roles, allg = d["rows"], d["sample"], d["roles"], d["allg"]
    res = sample["res"]
    fit_rows = rows["fit"]
    steps = FH.steps_of(d)
    _name, X, col_names = steps[-1]
    fit_all, test = np.flatnonzero(fit_rows), np.flatnonzero(~fit_rows)
    out = {"core": CORE, "columns": col_names, "blocks": {g: d["blocks"][g][1] for g in BLOCKS}, "meta": rows["meta"], "state_kinds": d["kinds"], "roles": roles,
           "mean_rating": float(rows["R"][fit_rows][:, 0].mean()), "linear": {}, "trees": {}, "active": {}}
    for rel in ("us", "them"):
        h = EH.Head(CORE, rel, *EH.targets(CORE, rel, roles, rows, res))
        h.activate(fit_rows)
        out["active"][rel] = [bool(x) for x in h.active]
        t1 = time.perf_counter()
        lin = EH.fit_best(h, X, rows, fit_all, None)
        out["linear"][rel] = lin
        tr = FH.fit_tree(h, X, rows, fit_all, None)
        out["trees"][rel] = tr
        den = h.steps()[test].astype(np.float64)
        tl, _p = FH.nll_table(h, FH.table_linear(lin, X, test), test)
        tt, _p = FH.nll_table(h, FH.table_tree(tr, X, test), test)
        tb, _p = EH.eval_head(h, None, None, test, h.base_logit(fit_all))
        say(f"FIT keeps [{rel}]: fit rows {len(fit_all)}, test rows {len(test)}; trees: {tr['n_iter']} rounds (capped {tr['capped']}); test log loss per step: base hazard {tb.sum() / den.sum():.5f}, linear {tl.sum() / den.sum():.5f}, trees {tt.sum() / den.sum():.5f}; wall {time.perf_counter() - t1:.0f} s")
    p = heads_file(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(out, f)
    say(f"FIT saved {p} ({p.stat().st_size / 1e6:.1f} MB); wall {time.perf_counter() - t0:.0f} s")
    return p


# ------------------------------------------------------------------------------------------------------------------------------ the inputs of a position at play time

def unit_rows(res: dict, base: int = 0) -> dict:
    """The state dict (as frames_state_rows.read_rows makes it) of a `questions_state` result, plus per row the board's units as (role, kind, place)."""
    rs = res["rows"]
    uoff, units = [0], []
    for r in rs:
        units += r["units"]
        uoff.append(len(units))
    u = np.array(units, np.uint8).reshape(-1, 6)[:, KEEP]
    return {"game": np.array([r["game"] + base for r in rs], np.int32), "ply": np.array([r["ply"] for r in rs], np.int16), "mover": np.array([r["mover"] for r in rs], np.int8),
            "frames": np.array([r["frames"] for r in rs], np.int16), "unread": np.array([r["unread"] for r in rs], np.int32), "units": u, "uoff": np.array(uoff, np.int64),
            "board": [r["board"] for r in rs]}


def edge_difference(st: dict, res: dict) -> np.ndarray:
    """Per row: the sum of the edge distances of the mover's units less the other side's, by kind (the same input as the fit's E)."""
    ed = TT.edge_distances(res["places"])
    nk, nr = len(res["kinds"]), len(res["roles"])
    out = np.zeros((len(st["ply"]), nk), np.float32)
    for i, b in enumerate(st["board"]):
        e = np.zeros((nr, nk), np.float32)
        for role, kind, place in b:
            e[role, kind] += ed[place]
        m = int(st["mover"][i])
        out[i] = e[m] - e[1 - m]
    return out


def statements_children(sheet, n: int, moves: list, probe: str) -> "dict[int, np.ndarray]":
    """{legal index: the statements of the position after it, the side to move there less the other side} for the position after `moves`, from one reading of its children per statement (a one-hot
    weight; the engine returns the weighted statement of the other side less that of the side to move, so the sign is turned). `probe`: any legal move there (it makes the position a played one)."""
    out: dict = {}
    for j in range(n):
        w = [0.0] * n
        w[j] = 1.0
        rows = sheet.subgoal_children([(None, list(moves) + [probe])], 1, [[len(moves)]], w)[0]
        if not rows:
            return {}
        for ix, v in rows[0][2]:
            out.setdefault(int(ix), np.zeros(n, np.float32))[j] = -v
    return out



class Reader:
    """Reads the number of positions. One per process (the models are built once). `heads` is the pickle `fit` wrote."""

    def __init__(self, heads: "str | Path | None" = None, threads: int = 1):
        self.threads = threads
        f = Path(heads) if heads is not None and Path(heads).is_file() else heads_file(heads)
        self.h = pickle.loads(f.read_bytes())
        from catspace.incubator.analyst.discovery import question_sheet as QS
        from catspace.incubator.analyst.discovery import sheet_rust as SR
        self.layer1 = GL.model(HD.GAME, HD.VARIANT)
        self.goal = G2M.goal_kind_of(HD.VARIANT)
        self.sheet = SR.model(QS.Env(HD.VARIANT), {})   # the fit rows were read with an empty worth: no assumed worth
        self.n_statements = len(self.sheet.subgoal_names())
        rec = TS.human_kind_record()
        sample = TS.load_sample(Path(paths.experiment(*TS.HEADS_DIR)), TS.PER_SHARD, 0)
        self.frame_model = TS.make_model(sample, rec, 0)
        picked = OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), TS.SEEDS)
        self.keys = [k for _dc, k, _d, _z in picked]
        self.heads_us = SimpleNamespace(active=self.h["active"]["us"])
        self.heads_them = SimpleNamespace(active=self.h["active"]["them"])
        self.cost: dict = {"frames": 0.0, "kinds": 0.0, "statements": 0.0, "heads": 0.0, "rows": 0}

    # -- the parts
    def kinds_part(self, positions: list) -> "tuple[np.ndarray, np.ndarray]":
        """(kinds ours less theirs without the goal kind, quiet flag) at the last position of each (start, moves) game."""
        names, raw = self.layer1.position_table(positions, self.threads)
        nk = len(names)
        w = 2 + 2 * nk
        flat = np.frombuffer(raw, dtype=np.int16)
        use = [k for k, nm in enumerate(names) if nm != self.goal]
        out, quiet, off = [], [], 0
        for _s, mv in positions:
            n = len(mv) + 1
            t = flat[off:off + n * w].reshape(n, w).astype(np.int32)
            off += n * w
            last = t[-1]
            out.append((last[2:2 + nk] - last[2 + nk:2 + 2 * nk])[use])
            quiet.append(bool(last[0] == 1 and last[1] == 1))
        return np.array(out, np.float32), np.array(quiet)

    def statements_children(self, moves: list, probe: str) -> "dict[int, np.ndarray]":
        return statements_children(self.sheet, self.n_statements, moves, probe)

    def frames_part(self, positions: list) -> "tuple[dict, dict]":
        """(state dict, the raw result) of the frame read at the last position of each (start, moves) game, one thread."""
        res = self.frame_model.questions_state([(s, list(m)) for s, m in positions], [[len(m)] for _s, m in positions], self.keys, self.goal, TS.FUSE, TS.ROW_FUSE, OLD.MAX_SEED_SIZE, True,
                                               TS.TRAP_FUSE, TS.HORIZON, self.threads)
        return unit_rows(res), res

    # -- the matrix
    def matrix(self, positions: list, statements: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """(inputs of the heads for the positions that have a frame read row, the index of those positions): `statements` one row per position (the statements, side to move less the other side)."""
        t0 = time.perf_counter()
        st, res = self.frames_part(positions)
        t1 = time.perf_counter()
        kd, _quiet = self.kinds_part(positions)
        t2 = time.perf_counter()
        keep = st["game"].astype(int)
        blocks = FH.frame_blocks({k: st[k] for k in ("ply", "units", "uoff", "frames", "unread")}, len(self.h["state_kinds"]), self.h["state_kinds"])
        cols = [blocks[g][0][:, [blocks[g][1].index(nm) for nm in self.h["blocks"][g]]] for g in BLOCKS]
        ed = edge_difference(st, res)
        ply = np.array([len(positions[i][1]) for i in keep], np.float32)
        rating = np.stack([np.full(len(keep), self.h["mean_rating"], np.float32), np.zeros(len(keep), np.float32), ply], axis=1)
        X = np.hstack([kd[keep], statements[keep], ed, rating] + cols).astype(np.float32)
        assert X.shape[1] == len(self.h["columns"]), (X.shape, len(self.h["columns"]))
        self.cost["frames"] += t1 - t0
        self.cost["kinds"] += t2 - t1
        self.cost["rows"] += len(keep)
        return X, keep

    def numbers_of_matrix(self, X: np.ndarray, remaining: int) -> "dict[str, np.ndarray]":
        """{head: the mover's first-chance less the other side's, at each row} for rows of positions, by `trees` and `linear`. The side that moved INTO the position is the `them` head's side."""
        t0 = time.perf_counter()
        sel = np.arange(len(X))
        rem = np.full(len(X), max(1, int(remaining)))
        out = {}
        for kind in ("trees", "linear"):
            if kind == "trees":
                Pu, Pt = (FH.table_tree(self.h["trees"][r], X, sel) for r in ("us", "them"))
            else:
                Pu, Pt = (FH.table_linear(self.h["linear"][r], X, sel) for r in ("us", "them"))
            pu, pt = FH.race_table(self.heads_us, self.heads_them, Pu, Pt, rem)
            out[kind] = first_difference(pu, pt)
        self.cost["heads"] += time.perf_counter() - t0
        return out

    def numbers(self, moves: list, legal: list, names: list, cap: int) -> dict:
        """{move name: {"trees": x, "linear": y, "quiet": flag}} for the moves `names` (some of `legal`) at the position after `moves`: the number of the root mover's move, which is the position's
        side that has just moved. A position the frame read has no row for (the game is over) is left out."""
        t0 = time.perf_counter()
        probe = legal[0]
        stm = self.statements_children(moves, probe)
        self.cost["statements"] += time.perf_counter() - t0
        idx = {nm: legal.index(nm) for nm in names}
        positions = [(None, list(moves) + [nm]) for nm in names]
        sts = np.array([stm.get(idx[nm], np.zeros(self.n_statements, np.float32)) for nm in names], np.float32)
        X, keep = self.matrix(positions, sts)
        if not len(keep):
            return {}
        nums = self.numbers_of_matrix(X, cap - (len(moves) + 1))
        _kd, quiet = self.kinds_part([positions[i] for i in keep])
        return {names[int(i)]: {"trees": float(-nums["trees"][k]), "linear": float(-nums["linear"][k]), "quiet": bool(quiet[k])} for k, i in enumerate(keep)}


def first_difference(first_to_move: np.ndarray, first_other: np.ndarray) -> np.ndarray:
    """At a position: the chance that the side to move there is first, less the other side's (the second is the side that moved into it). Swapping the two arguments negates it."""
    return np.asarray(first_to_move) - np.asarray(first_other)


def verify(n: int, say, seed: int = 6767) -> None:
    """For `n` rows of the fit (settled positions of human games) rebuild the inputs from the moves alone with this module's reading and compare them with the fit's matrix, by group of columns."""
    d = FH.build(Path(paths.experiment(*FH.STATE)), say)
    rows, sample, allg = d["rows"], d["sample"], d["allg"]
    X = FH.steps_of(d)[-1][1]
    r = Reader(threads=1)
    meta = rows["meta"]
    nk, ns, ne = len(meta["kinds"]), len(meta["statements"]), len(meta["edge_kinds"])
    groups = {"kinds": (0, nk), "statements": (nk, nk + ns), "edge distances": (nk + ns, nk + ns + ne), "ply": (nk + ns + ne + 2, nk + ns + ne + 3)}
    ofs = nk + ns + ne + 3
    for g in BLOCKS:
        w = len(d["blocks"][g][1])
        groups[g] = (ofs, ofs + w)
        ofs += w
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows["g"]), n, replace=False)
    same = {k: 0 for k in groups}
    worst = {k: 0.0 for k in groups}
    for i in pick:
        moves = allg[sample["ids"][int(rows["g"][i])]]["moves"][:int(rows["ply"][i])]
        sts = np.array(r.sheet.subgoal_games([(None, moves)], 1)[0][-1], np.float32)
        sd = sts[:ns] - sts[ns:]
        mine, _keep = r.matrix([(None, moves)], sd[None, :])
        for k, (a, b) in groups.items():
            diff = float(np.abs(mine[0, a:b] - X[i, a:b]).max()) if b > a else 0.0
            worst[k] = max(worst[k], diff)
            same[k] += diff < 1e-4
    for k in groups:
        say(f"VERIFY inputs rebuilt from the moves against the fit's matrix, {n} rows: {k}: identical in {same[k]} rows, largest difference {worst[k]:.5f}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out-dir", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--n", type=int, default=60)
    v.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir or paths.experiment(*OUT))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS tiebreak heads: {m}")
        print(lines[-1], flush=True)
        (out / f"{a.cmd}_obs.txt").write_text("\n".join(lines) + "\n")
    if a.cmd == "fit":
        fit(out, say)
    else:
        verify(a.n, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
