"""THE COMPOSITION GAUGE (Kaveh 2026-09-23: lemmas self-contained in their world, invoked as concepts in bigger
frames; growth driven by what the small worlds got wrong). On real positions, for every legal move: the COMPOSED
prediction from the lemmas embedded in the current bodies, level by level of the body search, against the
TRUTH on the full board (the resolved exchange, relative to the null move). The residual by level says whether
bringing in bodies makes the composed world converge on the real one. Beside it, the USED lemmas: those whose
verdict on the engine's best move differs from their verdict on the alternatives it beat (premise selection by
use, no leave-one-out), weighted by the stake.

    python -m catspace.incubator.analyst.gauge_compose --n 200
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess

from .lemma_apply import Applier, load_library, compose

OUT = Path("data/derived/lemmas")


def truth(dom, b: chess.Board, mv: chess.Move) -> float | None:
    from .chess_domain import State, resolved_reply_scalar, resolved_null_reply_scalar
    st = State(b.copy(stack=False), [mv.from_square])
    base = resolved_null_reply_scalar(st, mv.from_square)
    if base is None:
        return None
    after = dom.transition(st, (st.board, mv))
    d = resolved_reply_scalar(after, after.ids[0]) - base
    return max(-10.0, min(10.0, d))                                          # material units, as the lemma values are (a mate would own the mean error)


def run(n: int, lo: int, hi: int, library: Path, proved: Path, max_bodies: int = 5, seed: int = 0, log=print, ranker=None, tag: str = ""):
    from .body_search import body_search
    from .chess_domain import domain
    from .dataset import load as load_ds
    dom = domain(); app = Applier(dom, load_library(library, proved))
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    rows = [m for m in meta if lo <= m["ply"] <= hi and m.get("best")]
    rng = random.Random(seed); rng.shuffle(rows)
    t0 = time.perf_counter()
    err = defaultdict(list); spoken = Counter(); n_pos = 0; used = Counter(); used_stake = Counter(); best_pred_rank = []
    rho = defaultdict(list); best_rank_lvl = defaultdict(list)
    for m_ in rows:
        if n_pos >= n:
            break
        b = chess.Board(m_["fen"])
        if b.is_game_over():
            continue
        try:
            best = chess.Move.from_uci(m_["best"])
        except Exception:
            continue
        legal = list(b.legal_moves)
        if best not in legal:
            continue
        n_pos += 1
        preds_final = {}; verdicts_final = {}; truths = {}; preds_lvl = defaultdict(dict)
        for mv in legal:
            t = truth(dom, b, mv)
            if t is None:
                continue
            truths[mv] = t
            r = body_search(app, b, mv.from_square, mv, max_bodies=max_bodies, ranker=ranker)
            for lvl, level in enumerate(r["trace"]):
                vs = [(app.lemmas_by_idx[idx], p, fired, None) for idx, text, p, fired in level["used"]]
                c = compose(vs)
                if c is not None:
                    err[lvl + 1].append(abs(c - t)); spoken[lvl + 1] += 1; preds_lvl[lvl + 1][mv] = c
        # per position, per level: rank correlation between composed prediction and truth over the moves spoken about,
        # and the best move's rank (units differ between a lemma's world and the board; order is what composes)
        from scipy.stats import spearmanr
        for lvl, pm in preds_lvl.items():
            common = [m for m in pm if m in truths]
            if len(common) >= 4:
                r_, _ = spearmanr([pm[m] for m in common], [truths[m] for m in common])
                if r_ == r_:
                    rho[lvl].append(float(r_))
            if best in pm:
                ranked = sorted(pm.values(), reverse=True)
                best_rank_lvl[lvl].append((ranked.index(pm[best]) + 1) / len(pm))
            last = next((l for l in reversed(r["trace"]) if l["used"]), None)
            if last:
                vs = [(app.lemmas_by_idx[idx], p, fired, None) for idx, text, p, fired in last["used"]]
                preds_final[mv] = compose(vs); verdicts_final[mv] = {idx: p for idx, text, p, fired in last["used"]}
        # premise selection by use: a lemma is USED when its verdict on the best move differs from its verdict on an alternative
        if best in verdicts_final:
            vb = verdicts_final[best]
            for mv, vm in verdicts_final.items():
                if mv == best:
                    continue
                for idx, p in vm.items():
                    if idx in vb and vb[idx] != p:
                        used[app.lemmas_by_idx[idx].text] += 1
                        used_stake[app.lemmas_by_idx[idx].text] += abs(truth(dom, b, mv) or 0.0)
        if preds_final and best in preds_final and preds_final[best] is not None:
            ranked = sorted((v for v in preds_final.values() if v is not None), reverse=True)
            best_pred_rank.append(ranked.index(preds_final[best]) + 1)
        if n_pos % 25 == 0:
            log(f"[compose] {n_pos}/{n} ({time.perf_counter() - t0:.0f}s)")
    base_err = None
    log(f"[compose] {n_pos} positions, {time.perf_counter() - t0:.0f}s; lemmas {len(app.lemmas)}")
    log("  |composed prediction - truth| by number of bodies (mean pawns; moves spoken about): " + ", ".join(f"L{k}: {sum(v) / len(v):.2f} ({spoken[k]})" for k, v in sorted(err.items())))
    log("  Spearman(composed, truth) per position by bodies: " + ", ".join(f"L{k}: {sum(v) / len(v):+.3f} ({len(v)})" for k, v in sorted(rho.items())) + "   (0 = no order)")
    log("  the best move's rank by composed value, as a share of the spoken moves (0.5 = random): " + ", ".join(f"L{k}: {sum(v) / len(v):.2f}" for k, v in sorted(best_rank_lvl.items())))
    if best_pred_rank:
        log(f"  the engine's best move's rank by composed prediction: mean {sum(best_pred_rank) / len(best_pred_rank):.2f} over {len(best_pred_rank)} positions (random baseline ~ half the legal moves)")
    log("  most USED lemmas (verdict differs between the best move and an alternative), by stake:")
    for t, k in used_stake.most_common(8):
        log(f"    stake {k:7.1f}  used {used[t]:4d}  {t[:150]}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(positions=n_pos, err_by_level={k: sum(v) / len(v) for k, v in err.items()}, spoken=dict(spoken), best_rank=best_pred_rank, used=used.most_common(200), used_stake=used_stake.most_common(200)), open(OUT / f"gauge_compose{tag}.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--lo", type=int, default=10); ap.add_argument("--hi", type=int, default=80); ap.add_argument("--max-bodies", type=int, default=5)
    ap.add_argument("--library", default=str(OUT / "library_v2_size3.json")); ap.add_argument("--proved", default=str(OUT / "proved_v2_size3.json"))
    ap.add_argument("--guide", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--compose", default="mean", choices=["mean", "specific", "sum"])
    a = ap.parse_args(argv)
    import catspace.incubator.analyst.lemma_apply as la
    la.COMPOSE_RULE = a.compose
    ranker = None
    if a.guide == "tf":
        from .body_guide_tf import GraphRanker
        from .chess_domain import domain
        ranker = GraphRanker(domain())
    run(a.n, a.lo, a.hi, Path(a.library), Path(a.proved), a.max_bodies, ranker=ranker, tag=a.tag)


if __name__ == "__main__":
    main()
