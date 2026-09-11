"""gauge_sts.py -- the Strategic Test Suite (Corbit & Swaminathan; 1500 positions, 15 positional themes, every
candidate move scored 0-10) for every expert we have (Kaveh 2026-09-11: "what I care about is that the engine
solves quiet positional puzzles").

Experts: C the rules ranking (rank.score_moves, no learning), B the knowledge network's policy head, A the
board-only network's, M Maia 1800, and the fixed blend with the weights fit on game positions (experts.py). Per
expert: STS points (the score of its top move, 0 when unlisted), out of 10 per position, by theme; the standard
suite's own scale is points out of 15000.

    python -m catspace.incubator.analyst.gauge_sts --experts b,m,a [--shard 0/6] ; ... --experts c --shard i/6
    python -m catspace.incubator.analyst.gauge_sts --report
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import chess

from .memo import Memo

OUT = Path("data/derived/sts")
BLEND = {"a": 0.12, "b": 0.13, "c": 0.33, "m": 0.42}          # experts.py, 3200 rows (2026-09-10)


def load(path: str = "assets/suites/STS.epd"):
    items = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        board, ops = chess.Board.from_epd(line)
        pid = ops.get("id", ""); theme = re.sub(r"^STS\(v[\d.]+\)\s*", "", pid).rsplit(".", 1)[0]
        theme = {"AKPC": "Advancement of f/g/h pawns", "AT": "Avoid Pointless Exchanges", "Knight Outposts/Centralization/Repositioning": "Knight Outposts/Repositioning/Centralization"}.get(theme, theme)
        scores = {}
        for m, s in zip(ops.get("c9", "").split(), ops.get("c8", "").split()):
            scores[chess.Move.from_uci(m)] = int(s)
        items.append(dict(id=pid, theme=theme, board=board, scores=scores))
    return items


def run(items, experts: tuple, shard: str = "0/1", ckpt_b: str = "data/derived/checkpoints/kn_future4/best.pt", ckpt_a: str = "data/derived/checkpoints/kn_board/best.pt", log=print):
    i, k = (int(x) for x in shard.split("/"))
    items = items[i::k]
    OUT.mkdir(parents=True, exist_ok=True)
    A = B = maia = eng = None; memo = Memo()
    if "a" in experts or "b" in experts:
        from .experts import NetScorer
        from .graph import build
        A = NetScorer(ckpt_a) if "a" in experts else None; B = NetScorer(ckpt_b) if "b" in experts else None
    if "m" in experts:
        from .maia import Maia
        from .contrast import HUMAN_LEVEL
        maia = Maia(HUMAN_LEVEL)
    if "c" in experts:
        from .engine import Engine
        from .rank import score_moves
        eng = Engine(depth=12, cache_path=str(OUT / f"engine_{i}.sqlite"))
    rows = []; t0 = time.perf_counter()
    for n, it in enumerate(items, 1):
        b = it["board"]; legal = list(b.legal_moves)
        row = dict(id=it["id"], theme=it["theme"], legal=[m.uci() for m in legal], scores={m.uci(): s for m, s in it["scores"].items()})
        try:
            if A is not None or B is not None:
                from .graph import build
                g = build(b, memo)
                if A is not None:
                    sa = A.policy(b, g); row["a"] = [sa[m] for m in legal]
                if B is not None:
                    sb = B.policy(b, g); row["b"] = [sb[m] for m in legal]
            if maia is not None:
                import math
                pol = maia.policy(b); row["m"] = [math.log(max(pol.get(m.uci(), 0.0), 1e-6)) for m in legal]
            if eng is not None:
                sc = score_moves(b, memo, eng, budget=120, engine_depth=6)
                row["c"] = [float(sc.get(m, (0.0, ""))[0]) for m in legal]; row["c_why"] = [sc.get(m, (0.0, ""))[1] for m in legal]
        except Exception as ex:
            log(f"[sts] {it['id']} error {ex}", flush=True); continue
        rows.append(row)
        if n % 25 == 0:
            log(f"[sts {shard} {''.join(experts)}] {n}/{len(items)}, {time.perf_counter() - t0:.0f}s", flush=True)
    if eng is not None:
        eng.close()
    if maia is not None:
        maia.close()
    with open(OUT / f"rows_{''.join(experts)}_{i}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"[sts {shard} {''.join(experts)}] done {len(rows)} in {time.perf_counter() - t0:.0f}s", flush=True)


def report() -> str:
    import numpy as np
    by_id = {}
    for f in sorted(OUT.glob("rows_*.jsonl")):
        for l in open(f):
            r = json.loads(l); by_id.setdefault(r["id"], {}).update(r)
    rows = list(by_id.values())
    experts = [e for e in ("c", "b", "a", "m") if all(e in r for r in rows)]
    def pick(r, e):
        v = np.array(r[e]); return r["legal"][int(v.argmax())]
    def blend_pick(r):
        tot = np.zeros(len(r["legal"]))
        for e, w in BLEND.items():
            if e not in r:
                continue
            v = np.array(r[e], dtype=float)
            if e == "c":
                v = v / (np.abs(v).max() + 1.0)
            tot += w * v
        return r["legal"][int(tot.argmax())]
    cols = experts + (["blend"] if len(experts) >= 2 else [])
    out = [f"[sts] {len(rows)} positions; experts {cols}; points = the suite's score of the chosen move (10 = the key), by theme:"]
    out.append(f"   {'theme':<34}{'n':>5}" + "".join(f"{c:>9}" for c in cols))
    themes = defaultdict(list)
    for r in rows:
        themes[r["theme"]].append(r)
    totals = {c: 0.0 for c in cols}; solved = {c: 0 for c in cols}
    for th, rs in sorted(themes.items()):
        line = f"   {th:<34}{len(rs):>5}"
        for c in cols:
            pts = [r["scores"].get(blend_pick(r) if c == "blend" else pick(r, c), 0) for r in rs]
            totals[c] += sum(pts); solved[c] += sum(1 for p in pts if p == 10)
            line += f"{np.mean(pts):>9.2f}"
        out.append(line)
    n = len(rows)
    out.append(f"   {'ALL (mean points of 10)':<34}{n:>5}" + "".join(f"{totals[c] / n:>9.2f}" for c in cols))
    out.append(f"   {'key move found (top-1 = 10)':<34}{n:>5}" + "".join(f"{solved[c] / n:>9.1%}" for c in cols))
    out.append(f"   {'suite points (of ' + str(10 * n) + ')':<34}{'':>5}" + "".join(f"{int(totals[c]):>9}" for c in cols))
    if "c" in experts:
        silent = sum(1 for r in rows if max(r["c"]) <= 0)
        out.append(f"   the rules ranking is silent (no positive score) on {silent}/{n}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--experts", default="b,m"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
    else:
        run(load(), tuple(a.experts.split(",")), a.shard)


if __name__ == "__main__":
    main()
