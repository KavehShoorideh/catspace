"""THE NODES-TO-VALUE CURVE (plan §37, phase one). The baseline every lemma is judged against: how much WORK our
decision costs, and what it buys, at a ladder of budgets -- with the family lemmas on and off.

On tablebase positions (endgames_v2: every legal move's result is known exactly, and DTZ gives the POTENTIAL) we
run the rules ranking at each budget and report, per budget and per mode:
    decision   the chosen move keeps the result (the only thing that matters; beside a random legal move)
    progress   the potential's change the chosen move buys: the tablebase's distance to conversion, signed so
               that + is progress for the mover (a won position converting sooner, a lost one held longer)
    work       the node counter's total, and its parts
    per node   progress per unit of work, the plan's headline ratio
Nothing here learns; it is the ruler.

    python -m catspace.incubator.analyst.gauge_nodes --n 200 --budgets 0,40,120,400 --materials KRPvKR,KPvKP,KQPvKQ
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

from . import rank as R
from .endgames import load as load_rows, split_of
from .memo import Memo
from .nodes import count, total

OUT = Path("data/derived/nodes")
TB = "assets/tablebases/syzygy"
HELD = {"KRPvKP", "KQvKRP", "KBNvK"}


def potential(tb, board: chess.Board) -> float | None:
    """The mover's potential: + when closer to converting a win, + when further from losing. Scaled to [-1, 1]
    by the tablebase's 100-ply zeroing rule.
    NOTE (2026-09-21): Syzygy's DTZ is the distance to the next ZEROING (a pawn move or a capture), not to mate,
    so a good manoeuvring move often raises it. Raw potential DIFFERENCES are therefore noisy ply to ply; the
    gauge reports REGRET against the best legal move's own potential, which is exact and is what distance-optimal
    play means. The potential itself is kept for the value it carries (won / drawn / lost, and how far)."""
    if board.is_game_over():
        return 1.0 if board.is_checkmate() else 0.0        # the mover is mated: for the SIDE TO MOVE it is the worst
    try:
        wdl = tb.probe_wdl(board); dtz = tb.probe_dtz(board)
    except Exception:
        return None
    if wdl == 0:
        return 0.0
    d = min(abs(int(dtz)), 100) / 100.0
    return (1.0 - d) if wdl > 0 else (-1.0 + d)            # a win is better the sooner; a loss is better the later


def best_progress(tb, board: chess.Board) -> tuple[float, dict]:
    """The best potential change available here, and every legal move's, in the mover's view."""
    per = {}
    for m in board.legal_moves:
        nb = board.copy(stack=False); nb.push(m)
        phi1 = potential(tb, nb)
        if phi1 is not None:
            per[m.uci()] = -phi1
    return (max(per.values()) if per else 0.0), per


def run(n: int, budgets: list[int], materials: list[str] | None, seed: int, split: str, log=print):
    rows = load_rows(Path("data/derived/endgames_v2"))
    ids = [i for i, r in enumerate(rows) if split_of(r, HELD) == split and (not materials or r["material"] in materials)]
    random.Random(seed).shuffle(ids); ids = ids[:n]
    tb = chess.syzygy.open_tablebase(TB)
    out = defaultdict(lambda: defaultdict(list))
    t0 = time.perf_counter()
    for k, i in enumerate(ids):
        r = rows[i]; b = chess.Board(r["fen"])
        best = max(np.sign(v) for v in r["moves"].values())
        keepers = {u for u, v in r["moves"].items() if np.sign(v) == best}
        phi0 = potential(tb, b)
        best_phi, per_move = best_progress(tb, b)
        for lem in (True, False):
            R.FAMILY_LEMMAS = lem
            for bud in budgets:
                memo = Memo()
                with count() as c:
                    sc = R.score_moves(b, memo, None, budget=bud, mate_budget=max(1, bud * 10), plan_needs=False)
                mv = max(sc, key=lambda m: sc[m][0])
                nb = b.copy(stack=False); nb.push(mv)
                phi1 = potential(tb, nb)
                key = ("lemmas" if lem else "plain", bud)
                out[key]["keeps"].append(float(mv.uci() in keepers))
                out[key]["work"].append(total(c))
                out[key]["ledger"].append(c.get("ledger", 0)); out[key]["covered"].append(c.get("covered", 0))
                if phi0 is not None and phi1 is not None:
                    out[key]["progress"].append(-phi1 - phi0)      # the child's potential is the OPPONENT's: negate
                    out[key]["regret"].append(best_phi - (-phi1))  # 0 = a distance-optimal move; the tablebase's own ruler
                    out[key]["optimal"].append(float(abs(best_phi - (-phi1)) < 1e-9))
                out[key]["random"].append(len(keepers) / max(1, len(r["moves"])))
                if per_move:
                    out[key]["random_regret"].append(best_phi - float(np.mean(list(per_move.values()))))
        if (k + 1) % 20 == 0:
            log(f"[nodes] {k + 1}/{len(ids)} positions, {time.perf_counter() - t0:.0f}s", flush=True)
    tb.close()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {f"{m}@{b_}": {k: float(np.mean(v)) for k, v in d.items()} for (m, b_), d in out.items()}
    for (m, b_), d in out.items():
        res[f"{m}@{b_}"]["n"] = len(d["keeps"])
    json.dump(res, open(OUT / f"curve_{split}.json", "w"), indent=1)
    report(res, budgets, log)
    return res


def report(res: dict, budgets: list[int], log=print):
    log("[nodes] the nodes-to-value curve: the chosen move keeps the result, the progress it buys, and the work it cost")
    log(f"    {'budget':>7} | {'mode':>7} | {'keeps':>6} {'rand':>5} | {'regret':>7} {'rand':>5} {'optimal':>7} | {'work':>7} {'saved':>5} | {'regret/1k work':>14}")
    for bud in budgets:
        for m in ("plain", "lemmas"):
            d = res.get(f"{m}@{bud}")
            if not d:
                continue
            per = 1000.0 * d.get("regret", 0.0) / max(1.0, d["work"])
            log(f"    {bud:>7} | {m:>7} | {d['keeps']:>6.3f} {d['random']:>5.3f} | {d.get('regret', float('nan')):>7.3f} {d.get('random_regret', float('nan')):>5.3f} {d.get('optimal', float('nan')):>7.3f} | {d['work']:>7.0f} {d['covered']:>5.0f} | {per:>14.3f}")


# ---------------------------------------------------------------- openings and middlegames: the engine as the oracle
def run_games(n: int, budgets: list[int], seed: int, depth: int, lo: int, hi: int, tag: str, log=print):
    """The same curve where no tablebase reaches: positions from the game corpus, the oracle a deeper Stockfish.
    Regret is in centipawns against its best move; 'keeps' = within 50 cp of it. The oracle's own engine calls are
    made OUTSIDE the counter, so the work reported is our search's alone."""
    from .dataset import load as load_ds
    from .engine import Engine
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    rows = [m for m in meta if lo <= m["ply"] <= hi]
    random.Random(seed).shuffle(rows); rows = rows[:n]
    eng = Engine(depth=depth, cache_path=str(OUT / f"oracle_d{depth}.sqlite"))
    out = defaultdict(lambda: defaultdict(list))
    t0 = time.perf_counter()
    for k, m in enumerate(rows):
        b = chess.Board(m["fen"])
        if b.is_game_over():
            continue
        line = eng.best(b, depth=depth)
        if line is None:
            continue
        best_cp = float(line.cp)
        rnd = [chess.Move.from_uci(u) for u in random.Random(k).sample([x.uci() for x in b.legal_moves], min(2, b.legal_moves.count()))]
        r_regrets = []
        for mv in rnd:
            e = eng.eval_move(b, mv, depth=depth)
            if e is not None:
                r_regrets.append(min(300.0, max(0.0, best_cp - float(e.cp))))
        for lem in (True, False):
            R.FAMILY_LEMMAS = lem
            for bud in budgets:
                memo = Memo()
                with count() as c:
                    sc = R.score_moves(b, memo, None, budget=bud, mate_budget=max(1, bud * 10), plan_needs=False)
                mv = max(sc, key=lambda x: sc[x][0])
                e = eng.eval_move(b, mv, depth=depth)                      # outside the counter: the oracle is not our cost
                if e is None:
                    continue
                reg = min(300.0, max(0.0, best_cp - float(e.cp)))
                key = ("lemmas" if lem else "plain", bud)
                out[key]["keeps"].append(float(reg <= 50)); out[key]["tight"].append(float(reg <= 20))
                out[key]["top1"].append(float(mv.uci() == line.move))
                out[key]["regret"].append(reg); out[key]["work"].append(total(c))
                out[key]["covered"].append(c.get("covered", 0)); out[key]["lemma_pass"].append(c.get("lemma", 0))
                if r_regrets:
                    out[key]["random_regret"].append(float(np.mean(r_regrets)))
                    out[key]["random_keeps"].append(float(np.mean([x <= 50 for x in r_regrets])))
        if (k + 1) % 10 == 0:
            log(f"[nodes/{tag}] {k + 1}/{len(rows)} positions, {time.perf_counter() - t0:.0f}s, engine calls {eng.calls} hits {eng.hits}", flush=True)
    eng.close()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {f"{mo}@{bu}": {kk: float(np.mean(vv)) for kk, vv in d.items()} | {"n": len(d["regret"])} for (mo, bu), d in out.items()}
    json.dump(res, open(OUT / f"curve_{tag}.json", "w"), indent=1)
    report_games(res, budgets, tag, log)
    return res


def report_games(res: dict, budgets: list[int], tag: str, log=print):
    log(f"[nodes/{tag}] the curve against Stockfish: within 50 cp of its best move, the centipawns lost, and the work")
    log(f"    {'budget':>7} | {'mode':>7} | {'<=50cp':>7} {'rand':>5} | {'<=20cp':>7} {'top-1':>6} | {'cp lost':>8} {'rand':>6} | {'work':>7} {'saved':>5} | {'cp/1k work':>10}")
    for bud in budgets:
        for mo in ("plain", "lemmas"):
            d = res.get(f"{mo}@{bud}")
            if not d:
                continue
            per = 1000.0 * d["regret"] / max(1.0, d["work"])
            log(f"    {bud:>7} | {mo:>7} | {d['keeps']:>7.3f} {d.get('random_keeps', float('nan')):>5.3f} | {d['tight']:>7.3f} {d['top1']:>6.3f} | {d['regret']:>8.1f} {d.get('random_regret', float('nan')):>6.1f} | {d['work']:>7.0f} {d['covered']:>5.1f} | {per:>10.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--budgets", default="0,40,120,400")
    ap.add_argument("--materials", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--split", default="test")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--corpus", default="endgames", choices=["endgames", "opening", "middlegame"])
    ap.add_argument("--depth", type=int, default=16)
    a = ap.parse_args(argv)
    buds = [int(x) for x in a.budgets.split(",")]
    if a.report:
        if a.corpus == "endgames":
            report(json.load(open(OUT / f"curve_{a.split}.json")), buds)
        else:
            report_games(json.load(open(OUT / f"curve_{a.corpus}.json")), buds, a.corpus)
    elif a.corpus == "endgames":
        run(a.n, buds, [m for m in a.materials.split(",") if m], a.seed, a.split)
    else:
        lo, hi = (6, 20) if a.corpus == "opening" else (21, 70)
        run_games(a.n, buds, a.seed, a.depth, lo, hi, a.corpus)


if __name__ == "__main__":
    main()
