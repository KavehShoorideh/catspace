"""THE VALUE OF A PIECE LEAVING, FROM OUTCOMES (Kaveh 2026-09-18: "trade my inactive bishop; get rid of their
active knight" -- whether that pays is the planner's value, learned from results, never a rule).

For every piece in the knowledge corpus (every ply of Kaveh's games; train split only), the cell = whose piece x
its type x its activity now (free moves: none / 1-2 / 3-5 / 6+) x whether it hits the enemy king zone x the
engine's evaluation band (the mover's view). The contrast: the mover's result when that piece was GONE within 20
plies with the material even over the window (|change| <= 1: a trade, not a loss) minus when it was KEPT with the
material even. n rows and n games per arm; a game-clustered standard error (rows of one game are not
independent). Baselines: zero, and the position-level trade value (transitions.json, "phase" in the same band).

The neighbourhood ("remembering"): the same contrast restricted to the k nearest positions in the recall
memory's geometry, for pieces of the same side, type and activity."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import chess

OUT = Path("data/derived/piece_values.json")
DATA = "data/derived/knowledge"
TYPE = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
ACT = (("no free moves", 0, 0), ("1-2 free moves", 1, 2), ("3-5 free moves", 3, 5), ("6+ free moves", 6, 999))


def act_bin(free_moves: float) -> str:
    f = int(round(free_moves))
    return next(n for n, lo, hi in ACT if lo <= f <= hi)


def eval_band(cp) -> str:
    if cp is None:
        return "unknown"
    return "losing" if cp <= -200 else "worse" if cp <= -60 else "level" if cp < 60 else "better" if cp < 200 else "winning"


def cells_of(side: str, ptype: int, free_moves: float, kz: float, cp) -> list[str]:
    """Finest first; the lookup falls back to the coarser ones when n is thin."""
    t = TYPE[abs(ptype)]; a = act_bin(free_moves); k = "hits the king zone" if kz > 0 else "away from the king"; e = eval_band(cp)
    return [f"{side} {t} | {a} | {k} | {e}", f"{side} {t} | {a} | {e}", f"{side} {t} | {a}", f"{side} {t}"]


def _rows(data_dir: str = DATA, split: str = "train"):
    """Per piece-row: (row, node, cells, gone_even, kept_even, result, game)."""
    from .knowledge_net import Data
    from .dataset import load, load_piece_futures
    from .graph import NODE_QUESTIONS, POSITION_QUESTIONS
    D = Data(data_dir, meta_only=True, log=lambda *a, **k: None)
    d = load(data_dir); pf = load_piece_futures(data_dir)
    meta = d["meta"]; nodes = d["nodes"]; pos = d["position"]
    FM, KZ = NODE_QUESTIONS.index("free_moves"), NODE_QUESTIONS.index("king_zone_hits")
    MAT = POSITION_QUESTIONS.index("material_diff")
    fate = pf["fate_20"]
    out = []
    for r, m in enumerate(meta):
        if split and D.split[r] != split:
            continue
        if m["result"] is None:
            continue
        f = D.index.get((m["game"], m["ply"] + 20))
        if f is None:
            continue
        even = abs(float(pos[f, MAT]) - float(pos[r, MAT])) <= 1.0
        if not even:
            continue
        for j in range(m["n_pieces"]):
            t = m["types"][j]
            if t == 0 or abs(t) == 6:
                continue
            fj = int(fate[r, j])
            if fj < 0:
                continue
            side = "our" if t > 0 else "their"
            out.append((r, j, cells_of(side, t, nodes[r, j, FM], nodes[r, j, KZ], m["best_cp"]), fj in (1, 2), fj == 0, float(m["result"]), m["game"]))
    return out


def _contrast(items) -> dict | None:
    """items: (gone, kept, result, game). The diff with a game-clustered SE."""
    g = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for gone, kept, res, game in items:
        if gone:
            g[game][0] += res; g[game][1] += 1
        elif kept:
            g[game][2] += res; g[game][3] += 1
    gone_m = [v[0] / v[1] for v in g.values() if v[1]]; kept_m = [v[2] / v[3] for v in g.values() if v[3]]
    n_gone = sum(v[1] for v in g.values()); n_kept = sum(v[3] for v in g.values())
    if len(gone_m) < 2 or len(kept_m) < 2:
        return None
    # the mean over rows, the SE over games
    mg = sum(v[0] for v in g.values()) / max(1, n_gone); mk = sum(v[2] for v in g.values()) / max(1, n_kept)
    se = math.sqrt(np.var(gone_m) / len(gone_m) + np.var(kept_m) / len(kept_m))
    return dict(diff=round(mg - mk, 4), se=round(se, 4), gone=round(mg, 4), kept=round(mk, 4), n_gone=n_gone, n_kept=n_kept, games_gone=len(gone_m), games_kept=len(kept_m))


def fit(data_dir: str = DATA, out: Path = OUT, log=print) -> dict:
    rows = _rows(data_dir, "train")
    by = defaultdict(list)
    for r, j, cells, gone, kept, res, game in rows:
        for c in cells:
            by[c].append((gone, kept, res, game))
    table = {}
    for c, items in by.items():
        v = _contrast(items)
        if v is not None:
            table[c] = v
    json.dump(dict(table=table, note="the mover's result when the piece was gone within 20 plies with the material even minus when kept (even); train split of Kaveh's games"), open(out, "w"), indent=1)
    log(f"[piece-values] {len(rows)} piece-rows (train, material even over 20 plies); {len(table)} cells -> {out}")
    return table


def lookup(table: dict, cells: list[str], min_games: int = 30) -> tuple[str, dict] | None:
    for c in cells:
        v = table.get(c)
        if v is not None and min(v["games_gone"], v["games_kept"]) >= min_games:
            return c, v
    return None


def load_table(path: Path = OUT) -> dict:
    return json.load(open(path))["table"] if path.exists() else {}


def neighbourhood(rows_near: list[int], side: str, ptype: int, free_moves: float, data_dir: str = DATA, _cache={}) -> dict | None:
    """The contrast among the given knowledge rows (the nearest positions in the geometry) for pieces of the same
    side, type and activity bin."""
    if "rows" not in _cache:
        _cache["rows"] = defaultdict(list)
        for r, j, cells, gone, kept, res, game in _rows(data_dir, "train"):
            _cache["rows"][r].append((cells[2], gone, kept, res, game))
    want = cells_of(side, ptype, free_moves, 0, None)[2]
    items = [(gone, kept, res, game) for r in rows_near for (c, gone, kept, res, game) in _cache["rows"].get(r, []) if c == want]
    return _contrast(items)


def report(path: Path = OUT, min_games: int = 40, log=print):
    t = load_table(path)
    tr = {}
    try:
        tr = json.load(open("data/derived/transitions.json"))["table"]
    except Exception:
        pass
    log("[piece-values] the mover's result when the piece was traded off within 20 plies (material even) minus when it stayed; "
        "+ = trading it paid. se over games; the position-level 'phase' trade value in the same band beside it")
    for side in ("our", "their"):
        for t_ in ("knight", "bishop", "rook", "queen", "pawn"):
            for a, _, _ in ACT:
                c = f"{side} {t_} | {a}"
                v = t.get(c)
                if v is None or min(v["games_gone"], v["games_kept"]) < min_games:
                    continue
                log(f"  {c:<34} {v['diff']:+.3f} (se {v['se']:.3f}; gone {v['n_gone']} rows / {v['games_gone']} games, kept {v['n_kept']} / {v['games_kept']})")
    log("[piece-values] by evaluation band (bishops and knights, activity x band):")
    for side in ("our", "their"):
        for t_ in ("knight", "bishop"):
            for a, _, _ in ACT:
                bits = []
                for e in ("losing", "worse", "level", "better", "winning"):
                    v = t.get(f"{side} {t_} | {a} | {e}")
                    if v is not None and min(v["games_gone"], v["games_kept"]) >= min_games:
                        ph = tr.get(f"kaveh {e} | even", {}).get("phase", {}).get("diff")
                        bits.append(f"{e} {v['diff']:+.3f}±{v['se']:.3f}" + (f" (trades {ph:+.3f})" if ph is not None else ""))
                if bits:
                    log(f"  {side} {t_} | {a:<15} " + "; ".join(bits))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "report"]); ap.add_argument("--data", default=DATA)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit(a.data); report()
    else:
        report()


if __name__ == "__main__":
    main()
