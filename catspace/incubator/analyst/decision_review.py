"""THE PLANNER'S DECISIONS UNDER REVIEW (Kaveh 2026-10-10: "let's look at a few examples ... put them in the lemma
explorer"): census positions where the planner decided, its decision move against Stockfish (depth 12): the decisions
losing 150+ cp to Stockfish's best (the misses, worst first), a few within 20 cp for comparison; per position the
planner's reason, both lines, the loss. Added as a second review to the lemma reviewer's page (lemma_reviewer.py page)."""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter

import chess
import chess.engine

from catspace.io import paths

_ENG = None
KIND = {"": "an immediate win", "defend": "a defence", "cheapest": "the cheapest loss"}


def _one(fen):
    global _ENG
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    if os.environ.get("CATSPACE_HARASS") == "0":
        analyst_core.set_harass(False)
    tr, dec, proofs, nodes, ms, imm = analyst_core.plan_trace(fen)
    if not dec:
        return None
    mv = dec[1] if dec[0] in KIND else dec[0]
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = chess.Board(fen)
    i0 = _ENG.analyse(b, chess.engine.Limit(depth=12))
    a = b.copy(); a.push_uci(mv)
    i1 = _ENG.analyse(a, chess.engine.Limit(depth=12))
    s0 = i0["score"].pov(b.turn); s1 = i1["score"].pov(b.turn)
    best = s0.score(mate_score=3000); got = s1.score(mate_score=3000)
    reasons = [l.strip() for l in tr if l.startswith(("their best immediate win", "their threat if we pass", "what changed", "after:"))
               or "stops it" in l or "the best net" in l or "cheapest way" in l or "arms" in l][:8]
    return dict(fen=fen, move=mv, kind=KIND.get(dec[0], "a prepared threat"), summary=dec[3], loss=best - got, cp=got,
                mate=s1.is_mate() and (s1.mate() or 0) > 0, best=best, best_move=i0["pv"][0].uci() if i0.get("pv") else None,
                line=[m.uci() for m in [chess.Move.from_uci(mv)] + i1.get("pv", [])[:6]],
                best_line=[m.uci() for m in i0.get("pv", [])[:7]], reasons=reasons, proofs=proofs)


def collect(n: int, seed: int = 5) -> list:
    from multiprocessing import Pool
    from .waiting_frames import _games
    fens = random.Random(seed).sample([f for _, fs in _games("c300", 300) for f in fs], n)
    with Pool(os.cpu_count()) as p:
        return [r for r in p.map(_one, fens, chunksize=8) if r]


def _sans(fen, ucis):
    b = chess.Board(fen); out = []
    for u in ucis:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            break
        out.append(dict(san=b.san(m), uci=u)); b.push(m); out[-1]["fen"] = b.fen()
    return out


def review(rows: list, misses: int = 30, good: int = 5, seed: int = 0) -> dict:
    """The rows as a review for the explorer page (the same case shape as lemma_reviewer)."""
    from .tactic_census_measure import load
    census = load("c300")
    where = {r["fen"]: (r["game"], r["ply"]) for r in census}
    at = {(r["game"], r["ply"]): r["fen"] for r in census}
    from .lemma_reviewer import _between
    bad = sorted([r for r in rows if r["loss"] >= 150], key=lambda r: -r["loss"])
    ok = random.Random(seed).sample([r for r in rows if r["loss"] <= 20], min(good, sum(r["loss"] <= 20 for r in rows)))
    cases = []
    for r, wrong in [(r, True) for r in bad[:misses]] + [(r, False) for r in ok]:
        b = chess.Board(r["fen"]); g, ply = where.get(r["fen"], (None, None))
        last = _between(at.get((g, ply - 1)), r["fen"]) if g is not None else None
        played = _between(r["fen"], at.get((g, ply + 1))) if g is not None else None
        san = b.san(chess.Move.from_uci(r["move"]))
        best_san = b.san(chess.Move.from_uci(r["best_move"])) if r["best_move"] else None
        cases.append(dict(fen=r["fen"], game=g, ply=ply, stratum="decision", move=r["move"], san=san, targets=[], target_sq=None,
                          wrong=wrong, cp=r["cp"], mate=r["mate"], gain=None, line=_sans(r["fen"], r["line"]),
                          best=r["best"], best_move=r["best_move"], best_san=best_san, best_line=_sans(r["fen"], r["best_line"]),
                          sf_tactic=False, last=last, played=(b.san(chess.Move.from_uci(played)) if played else None),
                          played_uci=played, facts={},
                          headline=f"The planner plays {san}: {r['kind']}. Its reason: {r['summary']}.",
                          verdict=(f"Stockfish: {best_san} is better by {r['loss'] / 100:.2f} pawns." if wrong
                                   else f"Stockfish: within {max(r['loss'], 0) / 100:.2f} pawns of its best ({best_san})."),
                          reasons=r["reasons"]))
    kinds_bad = Counter(r["kind"] for r in bad); kinds_all = Counter(r["kind"] for r in rows)
    facts = {f"the decision is {k}": dict(wrong=kinds_bad[k], n_wrong=len(bad), right=kinds_all[k] - kinds_bad[k], n_right=len(rows) - len(bad))
             for k in sorted(kinds_all)}
    n = len(rows)
    return dict(lemma=dict(name="The planner's decisions",
                           statement="The planner plays the move with the best proved net: what it wins now minus the opponent's "
                                     "best proved win after it (immediate wins, defences ranked by net, prepared threats, the cheapest loss).",
                           unless=["it reasons about material, promotion, trapped pieces and mate by checks -- nothing positional",
                                   "it looks one move of theirs ahead of each decision, through its proofs"],
                           source="plan.rs, 2026-10-10 (the net-based planner, commit e7d06b44)"),
                groups=dict(surprise=dict(name="Stockfish disagrees", note="The planner's move loses 150+ cp against Stockfish's best "
                                          "(depth 12), worst first. What is the planner not reasoning about here?"),
                            refused=dict(name="Agrees", note="A few decisions within 20 cp of Stockfish's best, for comparison."),
                            passed=dict(name="-", note="")),
                open_text=f"{len(bad)} of {n} decisions ({100 * len(bad) / max(n, 1):.1f}%) lose 150+ cp against Stockfish's best. "
                          "By the kind of decision (misses against the rest):",
                fact_cols=["in the misses", "in the rest"],
                counts=[[n, "decisions sampled"], [len(bad), "lose 150+ cp"], [f"{100 * len(bad) / max(n, 1):.1f}%", "of decisions"]],
                summary=dict(facts=facts), cases=cases, n_positions=n,
                foot=f"Data: {n} planner decisions on a random sample of census positions; Stockfish depth 12. One run: directional. "
                     f"Built {time.strftime('%Y-%m-%d %H:%M')}. Pieces: Lichess cburnett.")


def page(rv: dict) -> str:
    """Fill the explorer page (lemma_reviewer.html, Lichess's cburnett pieces from CATSPACE_PIECES) with a review."""
    import base64
    from pathlib import Path
    pieces_dir = Path(os.environ["CATSPACE_PIECES"])
    pieces = {p: "data:image/svg+xml;base64," + base64.b64encode((pieces_dir / f"{p}.svg").read_bytes()).decode()
              for p in ("wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP")}
    tpl = (Path(__file__).with_name("lemma_reviewer.html")).read_text()
    out = Path(paths.experiment("decision_review.html"))
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(rv)).replace("/*__PIECES__*/null", json.dumps(pieces)))
    return str(out)


if __name__ == "__main__":
    import sys
    t0 = time.time()
    store = paths.experiment("decision_review.json")
    if "--page-only" in sys.argv:
        rows = json.load(open(store))
    else:
        rows = collect(4000)
        json.dump(rows, open(store, "w"))
    if os.environ.get("CATSPACE_PIECES"):
        print("page written to", page(review(rows)))
    bad = [r for r in rows if r["loss"] >= 150]
    print(f"TIME {time.time() - t0:.0f} s; OBS {len(rows)} decisions, losing 150+ cp {len(bad)} ({100 * len(bad) / len(rows):.1f}%); by kind {dict(Counter(r['kind'] for r in bad))}")
