"""PLAN 78.6 THE CENSUS OF TACTICS BY THEIR DEFECTS, MEASURED (Kaveh 2026-11-04: "are we able to identify all the tactics in the position? How are we doing with the
puzzles?"; "look at the density of tactics in games ... human versus human"). The census is cs_engine.census (Rust: cs_tactics::census -- a tactic's defects counted in
tempo: the free moves of ours it needs; read by the ledger search under the store's valuation record in use).

  puzzles   per lichess puzzle: is the ANSWER among the 0-defect tactics (the census's recall of the solution)? the tactics at each defect level; the cost.
  games     along lichess games (records_00000: human against human), every STEP-th position from ply 10: the census of BOTH sides; the tactic counts by defect level,
            our side less theirs, against the game's outcome for the side to move (win / draw / loss rates by bucket; a logistic fit of the outcome on the counts,
            beside the unit-count difference alone).

  .venv/bin/python scripts/plan78_census.py puzzles [--max-defects 2] [--depth 3] [--judge record]
  .venv/bin/python scripts/plan78_census.py games [--games 200] [--step 6] [--max-defects 2] [--depth 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cs_engine as CE
from catspace.io import paths

FUSE = 200_000            # nodes per reading (every position the same budget)


def judge(name: str) -> dict:
    from catspace.incubator.analyst.discovery import lemma_store as LS
    return LS.judge_of("chess", None if name == "record" else name)


def _puzzle(job):
    p, k, depth, jd, fuse = job
    t = time.time()
    tac, st = CE.census("chess", p["fen"], [], k, depth, fuse, "mover", **jd)
    by = [sum(1 for x in tac if x["defects"] == d) for d in range(k + 1)]
    return {"id": p["id"], "answer": p["answer"], "found0": any(x["defects"] == 0 and x["line"][:1] == [p["answer"]] for x in tac),
            "found_any": any(x["line"][:1] == [p["answer"]] or x["free"][:1] == [p["answer"]] for x in tac), "by": by, "cut": st["cut"], "readings": st["readings"],
            "sec": time.time() - t, "goal": any(x["wins"] == ["goal"] and x["defects"] == 0 for x in tac)}


def puzzles(a) -> None:
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    ps = [p for s in [f"rand{i}_10" for i in range(1, 22)] for p in PA.load_set(s)]
    jd = judge(a.judge)
    t0 = time.time()
    with Pool(a.jobs) as pool:
        res = pool.map(_puzzle, [(p, a.max_defects, a.depth, jd, a.fuse) for p in ps], chunksize=1)
    n = len(res)
    by = np.array([r["by"] for r in res])
    print(f"OBS 78.6 census on {n} puzzles (judge {jd.get('valuation', 'count')}; defects <= {a.max_defects}, depth {a.depth}, fuse {a.fuse}; wall {time.time() - t0:.0f} s): "
          f"the ANSWER among the 0-defect tactics {sum(r['found0'] for r in res)} / {n}; as the first move of any tactic or free move {sum(r['found_any'] for r in res)}; "
          f"tactics per position by defects (mean) {', '.join(f'{d}: {by[:, d].mean():.1f}' for d in range(by.shape[1]))}; positions with no tactic at all "
          f"{int((by.sum(1) == 0).sum())}; readings cut {sum(r['cut'] for r in res)} of {sum(r['readings'] for r in res)}; {np.mean([r['sec'] for r in res]):.2f} s a position")
    Path(paths.experiment(f"plan78_census_puzzles_{a.tag}.json")).write_text(json.dumps(res, indent=1))


def _game(job):
    gid, moves, result, step, k, depth, jd = job
    rows = []
    for ply in range(10, len(moves), step):
        st = CE.state("chess", None, moves[:ply])
        if st[3] is not None:
            break
        mover = st[1]
        ours, _s1 = CE.census("chess", None, moves[:ply], k, depth, FUSE, "mover", **jd)
        theirs, _s2 = CE.census("chess", None, moves[:ply], k, depth, FUSE, "other", **jd)
        units = Counter((p.isupper()) for p in st[0].split()[0] if p.isalpha() and p.lower() != "k")
        me_white = mover == "white"
        udiff = (units[True] - units[False]) * (1 if me_white else -1)
        out = result if me_white else -result                     # (+1 the side to move won, 0 drawn, -1 lost)
        rows.append({"game": gid, "ply": ply, "ours": [sum(1 for x in ours if x["defects"] == d) for d in range(k + 1)],
                     "theirs": [sum(1 for x in theirs if x["defects"] == d) for d in range(k + 1)], "units": udiff, "outcome": out})
    return rows


def _fit(X: np.ndarray, y: np.ndarray, games: np.ndarray, iters: int = 400) -> tuple:
    """A binary logistic of 'the side to move wins' (draws left out) on X: (weights, mean log loss, its standard error with the games as clusters)."""
    keep = y != 0
    X, t, g = X[keep], (y[keep] > 0).astype(float), games[keep]
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(H, Xb.T @ (p - t))
    p = np.clip(1 / (1 + np.exp(-Xb @ w)), 1e-12, 1 - 1e-12)
    loss = -(t * np.log(p) + (1 - t) * np.log(1 - p))
    m = loss.mean()
    per = defaultdict(lambda: [0.0, 0])
    for l, gg in zip(loss, g):
        per[gg][0] += l
        per[gg][1] += 1
    se = np.sqrt(sum((s - c * m) ** 2 for s, c in per.values())) / len(loss)
    return w, m, se, loss


def games(a) -> None:
    import pyarrow.parquet as pq
    f = Path(paths.data_dir()) / "records" / "lichess_2019-01" / "records_00000.parquet"
    t = pq.read_table(f, columns=["game_id", "result", "moves"]).slice(0, a.games).to_pylist()
    jd = judge(a.judge)
    jobs = [(int(r["game_id"]), r["moves"].split(), int(r["result"]), a.step, a.max_defects, a.depth, jd) for r in t]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        rows = [x for part in pool.map(_game, jobs, chunksize=1) for x in part]
    Path(paths.experiment(f"plan78_census_games_{a.tag}.json")).write_text(json.dumps(rows))
    k = a.max_defects
    O = np.array([r["ours"] for r in rows], dtype=float)
    T = np.array([r["theirs"] for r in rows], dtype=float)
    U = np.array([r["units"] for r in rows], dtype=float)
    y = np.array([r["outcome"] for r in rows])
    gm = np.array([r["game"] for r in rows])
    print(f"OBS 78.6 census along {a.games} lichess games (records_00000, human against human; every {a.step}th position from ply 10; judge {jd.get('valuation')}; "
          f"defects <= {k}, depth {a.depth}): positions {len(rows)}, wall {time.time() - t0:.0f} s; tactics per position (mean) -- the side to move "
          + ", ".join(f"{d}: {O[:, d].mean():.2f}" for d in range(k + 1)) + "; the other side " + ", ".join(f"{d}: {T[:, d].mean():.2f}" for d in range(k + 1)))
    D = O - T
    for d in range(k + 1):
        line = []
        for lo, hi, lab in ((-99, -2, "<= -2"), (-1, -1, "-1"), (0, 0, "0"), (1, 1, "+1"), (2, 99, ">= +2")):
            sel = (D[:, d] >= lo) & (D[:, d] <= hi)
            if sel.sum():
                line.append(f"{lab}: n {int(sel.sum())} win {np.mean(y[sel] > 0):.2f} draw {np.mean(y[sel] == 0):.2f} loss {np.mean(y[sel] < 0):.2f}")
        print(f"   {d}-defect tactics, ours less theirs -- " + "; ".join(line))
    _w0, m0, se0, l0 = _fit(U[:, None], y, gm)
    w1, m1, se1, l1 = _fit(np.hstack([U[:, None], D]), y, gm)
    keep = y != 0
    diff = l1 - l0
    per = defaultdict(lambda: [0.0, 0])
    for v, gg in zip(diff, gm[keep]):
        per[gg][0] += v
        per[gg][1] += 1
    md = diff.mean()
    sed = np.sqrt(sum((s - c * md) ** 2 for s, c in per.values())) / len(diff)
    print(f"   the outcome (decisive games, the side to move wins?) fit on the unit-count difference alone: log loss {m0:.4f} (se {se0:.4f}); with the tactic counts by "
          f"defect level added: {m1:.4f} (se {se1:.4f}) -- the paired difference {md:+.4f} (se {sed:.4f}, games as clusters); weights: units {w1[0]:+.3f}, "
          + ", ".join(f"{d}-defect {w1[1 + d]:+.3f}" for d in range(k + 1)))


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["puzzles", "games"])
    ap.add_argument("--max-defects", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--judge", default="record")
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--fuse", type=int, default=FUSE)
    ap.add_argument("--tag", default="v0")
    a = ap.parse_args(argv)
    (puzzles if a.mode == "puzzles" else games)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
