"""PLAN 78.96 (SYSTEM_SPEC 7 the prior and "the future imagination layer", 9 "the oracle labels positions and frames in sleep, for the prior", 11 the judge against the board): THE MEASURE STAGES OF "WHERE THE CLASS IS LOST, THE SEARCH OR THE CHOICE" AND OF THE PRIOR FITTED ON WHICH
EQUAL MOVE KEEPS IT. A sandbox: no player is changed; nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/prior_78_96/<tag>/. Game-free: the game is named only through the pieces that exist (prior_equal for the labels,
future_play's Picker for the picks of 78.94 and 78.95, symbols_play's reader for 78.82's reasons); the board and the words of a case are printed by ladder_print only. Every number is an OBS line (also in obs.txt). One run each: directions, not verdicts. This file never calls the oracle:
it reads the labels prior_equal.py made.

THE QUESTION. At a position the search holds several moves equal; the oracle's class of the position after each move says whether a move KEEPS the class of the position (its class is not lower than the class before it). Where the equal moves differ in that (the ROOM of a choice) a pick can
keep the class or lose it; where no equal move keeps it though the oracle's own move does, the loss is the SEARCH'S, not the choice's.

STAGES (each a command; the labels first, by prior_equal.py `label`).
  picks    for every labelled position with two or more equal moves: the picks that exist (the search's first move, the pattern pick of 78.82, the judge of 78.94, the joining of the two of 78.95; a random pick is its expected share) and the candidate matrices of part two (prior_pick.Features); picks.json, feats.npz.
  room     part one: by band and by class of the position, FIND and CONFIRM kept apart, 90 pct bands by games: the share where the equal moves differ in the class kept, where they differ in class, where no equal move keeps the class though the oracle's move does; the share where each pick keeps the class.
  fit      part two: the prior's two heads (linear, boosted trees) on FIND positions where the equal moves differ in keeping the class; one chosen on FIND by the held-out share of positions where its pick keeps the class (games cut into folds); prior.pkl.
  confirm  part two on CONFIRM, read once: the share of positions where each pick keeps the class, by band, with 90 pct bands by games, paired differences against the pattern pick; THE CONDITION OF PART THREE, written before any CONFIRM number was looked at: the prior is AHEAD of the pattern pick when the 90 pct band of the
           paired difference (prior less pattern pick, share of positions that keep the class) over ALL CONFIRM positions with two or more equal moves lies wholly above zero.
  cases    twenty positions for ladder_print --prior96: ten where the prior's pick keeps the class and the pattern pick does not, ten the other way (CONFIRM, equal moves that differ in keeping it; the order of a hash).

  .venv/bin/python -m catspace.incubator.analyst.discovery.prior_measure picks|room|fit|confirm|cases --tag short|full
Called by: the command line only; ladder_print.py reads cases96.json.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import pickle
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import future_judge as FJ
from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_pick as PP
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.io import paths

BANDS = LM.BANDS
SEED = 9696
PICKS = ("first", "pattern", "judge", "joined")


def band_index(units: np.ndarray) -> np.ndarray:
    return np.array([next(i for i, (_n, lo, hi) in enumerate(BANDS) if lo <= u <= hi) for u in units])


def share(vals: np.ndarray, game: np.ndarray, seed: int = SEED) -> "tuple[float, float, float]":
    """The mean of `vals` with a 90 pct band, the games resampled whole."""
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    _u, g = np.unique(game, return_inverse=True)
    G = len(_u)
    sums = np.bincount(g, weights=np.asarray(vals, float), minlength=G)
    cnt = np.bincount(g, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    d = rng.integers(0, G, (400, G))
    est = sums[d].sum(axis=1) / cnt[d].sum(axis=1)
    lo, hi = np.percentile(est, [5, 95])
    return float(np.mean(vals)), float(lo), float(hi)


def fmt(t: tuple, scale: float = 100.0, plus: bool = False) -> str:
    v, lo, hi = t
    if v != v:
        return "n/a"
    s = "+" if plus else ""
    return f"{scale * v:{s}.1f} [{scale * lo:{s}.1f}, {scale * hi:{s}.1f}]"


def keepvec(r: dict) -> np.ndarray:
    return np.array([PE.keeps(r["cls"], c) for c in r["cm"]], bool)


# ------------------------------------------------------------------------------------------------------------------------------ picks

_W: dict = {}


def _init_picks() -> None:
    from catspace.incubator.analyst.discovery import future_play as FP
    from catspace.incubator.analyst.discovery import symbols_play as SP
    pk = FP.picker()
    _W["picker"], _W["reader"] = pk, SP.reader82()
    _W["features"] = PP.Features(_W["reader"], pk.inputs, 1)


def _picks_job(chunk: list) -> list:
    pk, rd, ft = _W["picker"], _W["reader"], _W["features"]
    items = [(moves, tied) for _id, moves, tied in chunk]
    rows = ft.rows(items)
    out = []
    for (pid, moves, tied), r in zip(chunk, rows):
        pattern, _d = rd.pick(list(moves), list(tied), "reasons")
        j = pk.old_.choose(list(moves), list(tied))
        judge = j if j in tied else tied[0]
        joined = judge if bool(r["takes"].any()) else pattern
        out.append({"id": pid, "pattern": pattern, "judge": judge, "joined": joined, "x": r["x"]})
    return out


def cmd_picks(root: Path, say, workers: int = 0) -> None:
    recs = [r for r in PE.load(root) if len(r["tied"]) >= 2]
    games = {g["ord"]: g for g in FL.universe()}
    chunks, cur = [], []
    for r in recs:
        cur.append((r["id"], games[r["ord"]]["moves"][:r["ply"]], r["tied"]))
        if len(cur) == 40:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    t0 = time.perf_counter()
    res: dict = {}
    with multiprocessing.get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 2) - 1), initializer=_init_picks) as pool:
        for k, part in enumerate(pool.imap_unordered(_picks_job, chunks, chunksize=1), 1):
            for o in part:
                res[o["id"]] = o
            if k % 25 == 0:
                say(f"picks: {k} of {len(chunks)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    ids = [r["id"] for r in recs]
    X = np.concatenate([res[i]["x"] for i in ids]) if ids else np.zeros((0, 0), np.float32)
    off = np.cumsum([0] + [len(r["tied"]) for r in recs])
    np.savez_compressed(root / "feats.npz", x=X, off=off, ids=np.array(ids))
    (root / "picks.json").write_text(json.dumps({i: {k: res[i][k] for k in ("pattern", "judge", "joined")} for i in ids}))
    say(f"picks: {len(ids)} positions with two or more equal moves, {len(X)} candidate rows of {X.shape[1] if len(X) else 0} columns, in {time.perf_counter() - t0:.0f} s")


class Data:
    """The labelled equal moves with the picks and the candidate matrices, arrays over the positions with two or more equal moves."""

    def __init__(self, root: Path):
        self.root = root
        self.all = PE.load(root)
        self.recs = [r for r in self.all if len(r["tied"]) >= 2]
        z = np.load(root / "feats.npz")
        self.X, self.off = z["x"], z["off"]
        assert list(z["ids"]) == [r["id"] for r in self.recs]
        self.picks = json.loads((root / "picks.json").read_text())
        self.keep = [keepvec(r) for r in self.recs]
        self.game = np.array([r["ord"] for r in self.recs])
        self.units = np.array([r["units"] for r in self.recs])
        self.band = band_index(self.units)
        self.cls = np.array([r["cls"] for r in self.recs])
        self.part = np.array([r["part"] for r in self.recs])
        self.room = np.array([bool(k.any() and not k.all()) for k in self.keep])

    def rows_of(self, i: int) -> np.ndarray:
        return self.X[self.off[i]:self.off[i + 1]]

    def hit(self, name: str) -> np.ndarray:
        """1 where the pick of that name keeps the class (the search's first move: the first tied move)."""
        out = np.zeros(len(self.recs))
        for i, r in enumerate(self.recs):
            m = r["tied"][0] if name == "first" else self.picks[r["id"]][name]
            out[i] = float(self.keep[i][r["tied"].index(m)])
        return out

    def random_hit(self) -> np.ndarray:
        return np.array([k.mean() for k in self.keep])

    def any_hit(self) -> np.ndarray:
        return np.array([float(k.any()) for k in self.keep])


# ------------------------------------------------------------------------------------------------------------------------------ part one: the room

def cmd_room(root: Path, say) -> None:
    allr = PE.load(root)
    say(f"ROOM: labelled positions {len(allr)} (the stratified sample of prior_equal: the same number a part, class and band where the cell holds it; the shares are those of the sample, not of games); "
        f"the oracle's class of the position after each equal move at depth {PE.LO.FUTURE_DEPTH}; a move KEEPS the class when its class is not lower than the class of the position")
    for part in ("find", "confirm"):
        sub = [r for r in allr if r["part"] == part]
        game = np.array([r["ord"] for r in sub])
        units = np.array([r["units"] for r in sub])
        bi = band_index(units)
        cls = np.array([r["cls"] for r in sub])
        ge2 = np.array([len(r["tied"]) >= 2 for r in sub], float)
        nk = np.array([len(r["tied"]) for r in sub], float)
        kp = [keepvec(r) for r in sub]
        room = np.array([float(k.any() and not k.all()) for k in kp])
        dcls = np.array([float(len(set(r["cm"])) > 1) for r in sub])
        anyk = np.array([float(k.any()) for k in kp])
        firstk = np.array([float(k[0]) for k in kp])
        bestk = np.array([float(r["best_cls"] is not None and PE.keeps(r["cls"], r["best_cls"])) for r in sub])
        searchs = np.array([float((not k.any()) and b == 1.0) for k, b in zip(kp, bestk)])
        lost_first = np.array([float((not k[0]) and b == 1.0) for k, b in zip(kp, bestk)])
        groups = [("all", np.ones(len(sub), bool))] + [(n, bi == i) for i, (n, _l, _h) in enumerate(BANDS)] + [(f"class {c}", cls == c) for c in (1, 0, -1)]
        groups += [(f"{n} class {c}", (bi == i) & (cls == c)) for i, (n, _l, _h) in enumerate(BANDS) for c in (1, 0, -1)]
        for name, m in groups:
            if int(m.sum()) < 30:
                say(f"ROOM {part} {name}: {int(m.sum())} positions: too few")
                continue
            g = game[m]
            say(f"ROOM {part} {name}: positions {int(m.sum())}; two or more equal moves {fmt(share(ge2[m], g))} pct (mean equal moves {nk[m].mean():.2f}); the equal moves differ in the class kept (the room) {fmt(share(room[m], g))} pct, differ in class {fmt(share(dcls[m], g))} pct; "
                f"the oracle's own move keeps the class {fmt(share(bestk[m], g))} pct; some equal move keeps it {fmt(share(anyk[m], g))} pct; the SEARCH'S PART (no equal move keeps it, the oracle's move does) {fmt(share(searchs[m], g))} pct; "
                f"the search's first move loses a class the oracle's move keeps {fmt(share(lost_first[m], g))} pct (90 pct bands, games resampled)")
    d = Data(root)
    hits = {n: d.hit(n) for n in PICKS}
    hits["random"] = d.random_hit()
    hits["best of the equal"] = d.any_hit()
    for part in ("find", "confirm"):
        pm = d.part == part
        for name, m in [("all", pm)] + [(n, pm & (d.band == i)) for i, (n, _l, _h) in enumerate(BANDS)] + [(f"{n} room", pm & (d.band == i) & d.room) for i, (n, _l, _h) in enumerate(BANDS)] + [("room", pm & d.room)]:
            if int(m.sum()) < 30:
                say(f"PICKS {part} {name}: {int(m.sum())} positions: too few")
                continue
            say(f"PICKS {part} {name} (positions with two or more equal moves {int(m.sum())}): share where the pick keeps the class: " + "; ".join(f"{n} {fmt(share(hits[n][m], d.game[m]))}" for n in list(PICKS) + ["random", "best of the equal"]) + " (pct, 90 pct bands, games resampled)")


# ------------------------------------------------------------------------------------------------------------------------------ part two: the fit

def cv_hit(d: Data, ix: np.ndarray, kind: str, say) -> float:
    """The held-out share of positions where the head's pick keeps the class: FIND room positions cut into folds by game."""
    from sklearn.model_selection import GroupKFold
    hits = []
    for tr, te in GroupKFold(5).split(ix, groups=d.game[ix]):
        head = fit_head(d, ix[tr], kind)
        hits += [head_hit(d, head, i) for i in ix[te]]
    return float(np.mean(hits))


def rows_of_positions(d: Data, ix: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    X = np.concatenate([PP.centred(d.rows_of(i)) for i in ix])
    y = np.concatenate([d.keep[i].astype(int) for i in ix])
    g = np.concatenate([[d.game[i]] * len(d.keep[i]) for i in ix])
    return X, y, g


def fit_head(d: Data, ix: np.ndarray, kind: str, say=None) -> dict:
    X, y, g = rows_of_positions(d, ix)
    return FJ.fit_linear(X, y, g) if kind == "linear" else FJ.fit_tree(X, y, g, say)


def head_hit(d: Data, head: dict, i: int) -> float:
    tied = d.recs[i]["tied"]
    sc = FJ.predict(head, PP.centred(d.rows_of(i))) if head.get("m") is not None else np.zeros(len(tied))
    pick = PP.Prior.pick_of(tied, sc)
    return float(d.keep[i][tied.index(pick if pick else tied[0])])


def cmd_fit(root: Path, say) -> None:
    d = Data(root)
    ix = np.flatnonzero((d.part == "find") & d.room)
    say(f"FIT: FIND positions where the equal moves differ in keeping the class: {len(ix)} of {int(((d.part == 'find')).sum())} with two or more equal moves; candidate rows {int(sum(len(d.keep[i]) for i in ix))}; columns {d.X.shape[1]}; "
        f"the target of a row is whether the move keeps the class; inputs: pattern changes, the estimator's inputs after the move, whether it takes a unit, each read against the position's mean; no oracle number is an input")
    first = float(np.mean([d.keep[i][0] for i in ix]))
    pat = float(np.mean(d.hit("pattern")[ix]))
    say(f"FIT FIND room: share where the pick keeps the class: the search's first move {100 * first:.2f}, the pattern pick {100 * pat:.2f}, the judge {100 * np.mean(d.hit('judge')[ix]):.2f}, the joined {100 * np.mean(d.hit('joined')[ix]):.2f}, random {100 * np.mean(d.random_hit()[ix]):.2f}")
    cv = {}
    for kind in ("linear", "tree"):
        t0 = time.perf_counter()
        cv[kind] = cv_hit(d, ix, kind, say)
        say(f"FIT FIND {kind} head: held-out share where its pick keeps the class (five folds cut by game inside FIND) {100 * cv[kind]:.2f} pct; {time.perf_counter() - t0:.0f} s")
    chosen = "linear" if cv["linear"] >= cv["tree"] else "tree"
    heads = {k: fit_head(d, ix, k, say) for k in ("linear", "tree")}
    info = {k: ({"C": h.get("C"), "curve": h.get("curve")} if k == "linear" else {"n_iter": h.get("n_iter"), "held_loss": h.get("held_loss")}) for k, h in heads.items()}
    say(f"FIT: the head chosen on FIND: {chosen}; fitted on all FIND room positions; settings {json.dumps(info, default=float)}")
    pickle.dump({"heads": heads, "chosen": chosen, "cv": cv}, open(root / "prior.pkl", "wb"))


# ------------------------------------------------------------------------------------------------------------------------------ the CONFIRM table

def confirm_hits(d: Data, saved: dict) -> dict:
    hits = {n: d.hit(n) for n in PICKS}
    hits["random"] = d.random_hit()
    for k, h in saved["heads"].items():
        hits[f"prior {k}"] = np.array([head_hit(d, h, i) for i in range(len(d.recs))])
    hits["prior"] = hits[f"prior {saved['chosen']}"]
    hits["best of the equal"] = d.any_hit()
    return hits


def cmd_confirm(root: Path, say) -> None:
    d = Data(root)
    saved = pickle.load(open(root / "prior.pkl", "rb"))
    hits = confirm_hits(d, saved)
    pm = d.part == "confirm"
    names = list(PICKS) + ["prior linear", "prior tree", "prior", "random", "best of the equal"]
    say(f"CONFIRM (read once; the head chosen on FIND is {saved['chosen']}): positions with two or more equal moves {int(pm.sum())}, of them with room {int((pm & d.room).sum())}")
    cover = ", ".join(f"{n} {int((pm & (d.band == i)).sum())} (room {int((pm & (d.band == i) & d.room).sum())})" for i, (n, _l, _h) in enumerate(BANDS))
    say(f"CONFIRM coverage per band of units (few 0-8, mid 9-16, many 17 and more), positions with two or more equal moves: {cover}")
    groups = [("all", pm)] + [(n, pm & (d.band == i)) for i, (n, _l, _h) in enumerate(BANDS)] + [("room", pm & d.room)] + [(f"{n} room", pm & (d.band == i) & d.room) for i, (n, _l, _h) in enumerate(BANDS)]
    groups += [(f"class {c}", pm & (d.cls == c)) for c in (1, 0, -1)]
    for name, m in groups:
        if int(m.sum()) < 30:
            say(f"CONFIRM {name}: {int(m.sum())} positions: too few")
            continue
        say(f"CONFIRM {name} (positions {int(m.sum())}): share where the pick keeps the class: " + "; ".join(f"{n} {fmt(share(hits[n][m], d.game[m]))}" for n in names) + " (pct, 90 pct bands, games resampled)")
    for name, m in groups:
        if int(m.sum()) < 30:
            continue
        out = []
        for k, ref in enumerate(("pattern", "first", "joined", "judge", "random")):
            t = SM.paired(hits["prior"][m], hits[ref][m], d.game[m], SEED + k)
            out.append(f"prior less {ref} {fmt(t, plus=True)}")
        say(f"CONFIRM paired {name}: " + "; ".join(out) + " (pct points, 90 pct bands, games resampled)")
    t = SM.paired(hits["prior"][pm], hits["pattern"][pm], d.game[pm], SEED)
    ahead = t[1] > 0
    say(f"CONFIRM THE CONDITION OF PART THREE (prior less pattern pick, all CONFIRM positions with two or more equal moves, 90 pct band wholly above zero): {100 * t[0]:+.2f} [{100 * t[1]:+.2f}, {100 * t[2]:+.2f}] pct points: {'MET' if ahead else 'NOT MET'}")
    json.dump({"ahead": bool(ahead), "diff": t}, open(root / "condition.json", "w"))


def cmd_cases(root: Path, say) -> None:
    d = Data(root)
    saved = pickle.load(open(root / "prior.pkl", "rb"))
    hits = confirm_hits(d, saved)
    games = {g["ord"]: g for g in FL.universe()}
    pm = (d.part == "confirm") & d.room
    order = sorted(np.flatnonzero(pm), key=lambda i: PE.order_key(d.recs[i]["ord"], d.recs[i]["ply"]))
    good = [i for i in order if hits["prior"][i] == 1 and hits["pattern"][i] == 0][:10]
    bad = [i for i in order if hits["prior"][i] == 0 and hits["pattern"][i] == 1][:10]
    head = saved["heads"][saved["chosen"]]
    out = []
    for grp, ids in (("prior keeps, pattern pick does not", good), ("pattern pick keeps, prior does not", bad)):
        for i in ids:
            r = d.recs[i]
            sc = FJ.predict(head, PP.centred(d.rows_of(i))) if head.get("m") is not None else np.zeros(len(r["tied"]))
            out.append({"group": grp, "id": r["id"], "moves": games[r["ord"]]["moves"][:r["ply"]], "tied": r["tied"], "cm": r["cm"], "cls": r["cls"], "units": r["units"], "band": r["band"], "best": r["best"], "best_cls": r["best_cls"],
                        "pattern": d.picks[r["id"]]["pattern"], "judge": d.picks[r["id"]]["judge"], "joined": d.picks[r["id"]]["joined"], "first": r["tied"][0], "prior": PP.Prior.pick_of(r["tied"], sc) or r["tied"][0], "scores": [float(x) for x in sc],
                        "takes": [float(v) for v in d.rows_of(i)[:, -1]]})
    (root / "cases96.json").write_text(json.dumps(out))
    say(f"CASES: {len(good)} positions where the prior keeps the class and the pattern pick does not, {len(bad)} the other way (CONFIRM, with room, in the order of a hash); written to cases96.json for ladder_print --prior96")


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS prior {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["picks", "room", "fit", "confirm", "cases"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    root = PE.out_root(a.tag)
    say = Say(root, a.stage)
    if a.stage == "picks":
        cmd_picks(root, say, a.workers)
    else:
        {"room": cmd_room, "fit": cmd_fit, "confirm": cmd_confirm, "cases": cmd_cases}[a.stage](root, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
