"""THE EXPLAINER -- a product prototype (Kaveh 2026-10-02: "the final product is important: to name why the normal
human-looking move is not good, and why a different move that humans don't see is good (whether tactically or
strategically)"; "it's ok to use you as a final LLM layer to manually hand-label and name things that are
chess-specific").

For a position and a human move:
- WHY THE HUMAN MOVE IS NOT GOOD: after it, Stockfish's line for the opponent; if the evaluation swings by two pawns
  or more, the opponent's first move is read by the high-value lemma detectors (lemma_detect, our own relations, no
  oracle) and named by its family (tactic_families.json, the LLM-labelled names): what the tactic is and why the move
  that allowed it fails, with the pieces it involves.
- WHY THE MOVE HUMANS DON'T SEE IS GOOD: Stockfish's best move in the same position. If the detectors read it as a
  tactic of ours (P(any family) of a half or more), its family's name; else, if it denies the opponent the tactic the
  human move allowed, that; else 'strategic (not yet named)' with Stockfish's margin.
On Lichess puzzles the human move is the blunder that created the puzzle (the first move of its solution), real and
human. CHECK: the family named for the opponent's tactic against the puzzle's own Lichess themes (each family's typical
themes from its training tactics); the control: always naming the most common family.

    python -m catspace.incubator.analyst.explain --n 200 --offset 100      # fresh puzzles: the check + examples
    python -m catspace.incubator.analyst.explain --fen "<fen>" --move e2e4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import chess
import chess.engine

from catspace.io import paths

DEPTH = 14
SWING = 200                                                          # centipawns: the human move lets a tactic in
GENERIC = {"short", "long", "veryLong", "middlegame", "endgame", "opening", "crushing", "advantage", "master",
           "masterVsMaster", "superGM", "oneMove", "mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5"}
_ENG = {}
_FAM = {}


def _engine():
    if "e" not in _ENG:
        import shutil
        e = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
        e.configure({"Threads": 1, "Hash": 64})
        _ENG["e"] = e
    return _ENG["e"]


def _families() -> dict:
    if not _FAM:
        _FAM.update(json.load(open(paths.derived("tactic_families.json")))["families"])
    return _FAM


def _sf(b: chess.Board, mover: chess.Color, raw: bool = False):
    """(evaluation for `mover` in centipawns, principal variation[, the engine's score])."""
    if b.is_checkmate():
        r = ((-10000 if b.turn == mover else 10000), [])
        return r + (None,) if raw else r
    if b.is_game_over():
        return (0, [], None) if raw else (0, [])
    info = _engine().analyse(b, chess.engine.Limit(depth=DEPTH, time=10.0))
    r = (info["score"].pov(mover).score(mate_score=10000), info.get("pv", []))
    return r + (info["score"],) if raw else r


PREVENTS = {                                                          # the LLM naming layer: a defending move's relation -> words
    "defend:pawn_next_to_king_moves": "gives the king luft",
    "defend:unpins": "unpins",
    "defend:king_moves": "steps the king out of the line of fire",
    "defend:lands_between_key_pieces": "blocks the line",
    "defend:adds_defender_to_key_piece": "adds a defender",
    "defend:captures_key_piece": "captures a piece the tactic needs",
    "defend:a_key_piece_moves": "moves a piece the tactic needs out of the way",
}


def _how_it_prevents(b: chess.Board, bm: chess.Move, nb: chess.Board, opp_first: chess.Move) -> str:
    """How the better move bm (on b) heads off the opponent's tactic that starts with opp_first (on nb, after the
    human move): its relations to that tactic's key squares, in words."""
    from .tactic_reasons import describe_defence
    me = b.turn; them = not me
    after = nb.copy(stack=False); after.push(opp_first)
    hits = {t for t in chess.SquareSet(after.attacks_mask(opp_first.to_square)) if after.piece_at(t) is not None and after.piece_at(t).color == me}
    key = dict(owner=them, victim=me, first=opp_first, won=set(), pieces={opp_first.from_square, opp_first.to_square} | hits, king=b.king(me))
    feats = describe_defence(b, bm, key)
    if "defend:king_moves" in feats:
        feats = [f for f in feats if f != "defend:lands_between_key_pieces"]   # a king step is 'out of the line', not a block
    words = [PREVENTS[f] for f in PREVENTS if f in feats]
    return ", ".join(words[:2])


def _san_line(b: chess.Board, pv: list, n: int = 6) -> str:
    bb = b.copy(); out = []
    for m in pv[:n]:
        out.append(bb.san(m)); bb.push(m)
    return " ".join(out)


def _family_of(b: chess.Board, m: chess.Move) -> tuple:
    """The detectors' reading of move m on board b (m's side to move): (P(any family), family id or None)."""
    from .lemma_detect import values
    v = values(b).get(m.uci())
    return (0.0, None) if v is None else (v[1], v[2])


def _family_by_preventers(nb: chess.Board, line: list) -> int | None:
    """Name the tactic on nb (its side to move has it, `line` cashes it in) the way the families were DEFINED: its
    preventers (tactic_reasons, Stockfish), described by relations only, assigned to the nearest family centroid. The
    product may use Stockfish (it explains on top of it); the frame player cannot, and uses the detectors."""
    import numpy as np
    from collections import Counter
    from .tactic_reasons import _reason_only, preventers_of, stake
    s0 = stake(nb, nb.turn)
    if s0 is None or s0 < 200:
        return None
    pre, _ = preventers_of(nb, line, s0)
    if not pre:
        return None
    fam = json.load(open(paths.derived("tactic_families.json")))
    c = _reason_only(Counter(f for p in pre for f in p["feats"] + [f"{p['kind']}:any"]))
    v = np.log1p(np.array([c.get(f, 0) for f in fam["vocab"]], dtype=float))
    if not v.any():
        return None
    v = v / np.linalg.norm(v)
    d = ((np.array(fam["centroids"]) - v) ** 2).sum(1)
    return int(np.argmin(d))


def explain(b: chess.Board, human: chess.Move) -> dict:
    """The two explanations for the side to move on b and its (human) move."""
    me = b.turn
    best_ev, best_pv, best_score = _sf(b, me, raw=True)
    nb = b.copy(); nb.push(human)
    after_ev, opp_pv = _sf(nb, me)
    out = dict(fen=b.fen(), human=b.san(human), eval_best=best_ev, eval_after_human=after_ev, swing=best_ev - after_ev)
    fams = _families()
    if best_ev - after_ev >= SWING and opp_pv:
        p, f = _family_of(nb, opp_pv[0])                               # the detectors (fallback)
        fp = _family_by_preventers(nb, opp_pv)                         # the families' own definition (Stockfish preventers)
        if fp is not None:
            f = fp
        fam = fams.get(str(f), {}) if f is not None else {}
        out["why_not"] = dict(opponent_line=_san_line(nb, opp_pv), family=f, p_any=round(p, 2), name=fam.get("name"),
                              text=(f"{b.san(human)} allows {fam.get('name', 'a tactic')}: {fam.get('why_it_works', '')} -- "
                                    f"{fam.get('why_the_move_allowing_it_fails', '')}. "
                                    f"After {b.san(human)}, {_san_line(nb, opp_pv, 4)}."))
    else:
        out["why_not"] = dict(text=f"{b.san(human)} costs {best_ev - after_ev} centipawns: no tactic, a positional concession")
    if best_pv and best_pv[0] != human:
        from .sf_depth import payoff_ply
        bm = best_pv[0]
        p, f = _family_of(b, bm)
        fam = fams.get(str(f), {}) if f is not None else {}
        line = _san_line(b, best_pv)
        pay = payoff_ply(b, best_pv, best_score) if best_score is not None else None   # Stockfish's line wins material or mates
        if pay is not None and pay <= 8 and fam:                                      # soon: a tactic of ours (the detectors name it)
            text = f"{b.san(bm)} starts {fam['name']}: {fam['why_it_works']}. Line: {line}."
        elif "family" in out["why_not"]:
            how = _how_it_prevents(b, bm, nb, opp_pv[0])
            text = (f"{b.san(bm)}{(' ' + how) if how else ''}, which keeps the opponent from {out['why_not'].get('name') or 'the tactic'} "
                    f"(Stockfish {best_ev:+d}, against {after_ev:+d} after {b.san(human)}). Line: {line}.")
        else:
            text = f"{b.san(bm)} is better by {best_ev - after_ev} centipawns -- strategic (not yet named). Line: {line}."
        out["why_better"] = dict(move=b.san(bm), family=f, p_any=round(p, 2), tactic=pay is not None and pay <= 8, text=text)
    return out


def _themes_of_family() -> dict:
    """Each family's typical specific themes (by lift in its training tactics)."""
    from collections import Counter
    fam = json.load(open(paths.derived("tactic_families.json")))
    recs = {json.loads(l)["pid"]: json.loads(l) for l in open(paths.derived("tactic_reasons_3000_200.jsonl"))}
    allc = Counter(t for r in recs.values() if "skipped" not in r for t in r["themes"])
    N = sum(1 for r in recs.values() if "skipped" not in r)
    out = {}
    for g in fam["families"]:
        pids = [p for p, l in fam["label"].items() if str(l) == g]
        c = Counter(t for p in pids for t in recs[p]["themes"])
        lifts = sorted(((c[t] / len(pids)) / (allc[t] / N), t) for t in c if t not in GENERIC and c[t] >= 4)[::-1]
        out[g] = [t for l, t in lifts[:3] if l >= 1.5]
    return out


def _puzzle(row: dict) -> dict | None:
    b = chess.Board(row["FEN"]); human = chess.Move.from_uci(row["Moves"].split()[0])
    try:
        e = explain(b, human)
    except Exception as ex:                                            # a bad position: reported, not hidden
        return dict(pid=row["PuzzleId"], error=repr(ex))
    e.update(pid=row["PuzzleId"], themes=row["Themes"].split(), rating=int(row.get("Rating") or 0))
    return e


def check(n: int, offset: int, workers: int = 10, log=print) -> None:
    import multiprocessing as mp
    import random
    from collections import Counter
    from .frame_puzzles import fresh_rows
    t0 = time.perf_counter()
    rows = fresh_rows(n, 997, offset)
    with mp.get_context("fork").Pool(workers) as pool:
        res = [r for r in pool.map(_puzzle, rows, chunksize=2) if r is not None]
    json.dump(res, open(paths.derived(f"explain_{n}_{offset}.json"), "w"), indent=1)
    errs = [r for r in res if "error" in r]; res = [r for r in res if "error" not in r]
    named = [r for r in res if r["why_not"].get("family") is not None]
    tf = _themes_of_family()
    hit = [r for r in named if set(tf.get(str(r["why_not"]["family"]), [])) & set(r["themes"])]
    common = Counter(r["why_not"]["family"] for r in named).most_common(1)[0][0] if named else None
    base = [r for r in named if set(tf.get(str(common), [])) & set(r["themes"])]
    with_tf = [r for r in named if tf.get(str(r["why_not"]["family"]))]
    log(f"OBS explain on {len(rows)} fresh puzzles (rows i % 997 == {offset}; {time.perf_counter() - t0:.0f}s; errors {len(errs)}): the blunder "
        f"swings 2+ pawns by Stockfish depth {DEPTH} in {sum(1 for r in res if r['swing'] >= SWING)}; a family named for the opponent's tactic "
        f"in {len(named)}; the family's typical themes among the puzzle's own Lichess themes in {len(hit)} of the {len(with_tf)} named with "
        f"typical themes (always naming the most common family, {common}: {len(base)}); the better move named as a tactic of ours in "
        f"{sum(1 for r in res if r.get('why_better', {}).get('tactic'))}")
    by = Counter((str(r["why_not"]["family"]), bool(set(tf.get(str(r["why_not"]["family"]), [])) & set(r["themes"]))) for r in named)
    fams = _families()
    for g in sorted({k for k, _ in by}, key=lambda g: -(by[(g, True)] + by[(g, False)])):
        tot = by[(g, True)] + by[(g, False)]
        log(f"   {fams[g]['name'][:44]:44s} named {tot:3d}; its themes ({', '.join(tf.get(g, [])) or 'none'}) in the puzzle's in {by[(g, True)]}")
    random.seed(0)
    for r in random.sample(named, min(6, len(named))):
        log(f"\n   {r['pid']} ({', '.join(t for t in r['themes'] if t not in GENERIC)}) {r['fen']}")
        log(f"   WHY NOT: {r['why_not']['text']}")
        if r.get("why_better"):
            log(f"   BETTER:  {r['why_better']['text']}")


def main(argv=None):
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--offset", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--fen", default=None); ap.add_argument("--move", default=None)
    a = ap.parse_args(argv)
    try:
        if a.fen:
            b = chess.Board(a.fen); e = explain(b, chess.Move.from_uci(a.move))
            print("WHY NOT:", e["why_not"]["text"]); print("BETTER: ", e.get("why_better", {}).get("text", "(the move was best)"))
        else:
            check(a.n, a.offset, a.workers)
    finally:
        for e in _ENG.values():
            e.quit()
        from . import tactic_reasons as _tr
        for e in _tr._ENG.values():
            e.quit()


if __name__ == "__main__":
    main()
