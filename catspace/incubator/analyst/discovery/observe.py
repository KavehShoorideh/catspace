"""OBSERVE GAMES, LEARN FROM SURPRISE (Kaveh 2026-10-12: "start observing games, and whenever you are surprised, see if
you can understand your surprise ... and figure out if you can build a proof or a frame or a lemma using this prior to
address it and not be surprised anymore").

  OBSERVE   the planner (a PlanState: the game context) plays its frames' material against Stockfish
  SURPRISE  a move whose measure did not fall against every reply (uncertified), a position with no plan, a game not
            ending in mate -- each with its explanation (the certificate's: which component failed to fall)
  ADDRESS   the surprise boards become a HOLE whose base is the frame's current measure; the pipeline (proposers ->
            filter -> verifier) proposes a term; a verified improvement goes to the GENERATED track with its evidence
  AGAIN     observe again: did the surprise rate fall?

  python -m catspace.incubator.analyst.discovery.observe --piece R --games 300
  python -m catspace.incubator.analyst.discovery.observe --piece R --flags kr_generated=0,kr_chase=0   the derived frame
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from catspace.io import paths

# the frames' current measures as verifier specs (named components or terms over the base)
FRAME_SPECS = {"R": ["fn:wall", "fn:kr_generated"], "Q": ["fn:kq_offedge", "fn:kq_approach", "fn:kq_region"]}


def frame_spec(piece: str, flags: str) -> tuple:
    """(the hole's name suffix, the verifier spec) of the frame the switches select (mating.rs frame_for)."""
    f = dict(kv.split("=") for kv in filter(None, flags.split(",")))
    if piece == "R" and f.get("kr_generated") == "0":
        return ("chase", ["fn:wall", "fn:chase"]) if f.get("kr_chase", "1") == "1" else ("derived", ["fn:wall", "fn:opposition"])
    return ("", FRAME_SPECS[piece])


def _game(args):
    import os
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    seed, piece, flags, term = args
    import chess
    import chess.engine
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core
    for kv in filter(None, flags.split(",")):                  # any switch: name=0|1 (set_<name>)
        k, v = kv.split("=")
        getattr(analyst_core, f"set_{k}")(v == "1")
    if term:                                                    # a proposal in play: (schema, sexpr) on the derived frame
        analyst_core.set_kr_term(term[1], term[0])
    from catspace.incubator.analyst.kq_playout import _start
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = _start(random.Random(seed), piece)
    st = analyst_core.PlanState()
    surprises, moves = [], 0
    out = "not finished"
    while True:
        if b.is_checkmate():
            out = "mate"; break
        if b.is_stalemate() or (b.turn == chess.BLACK and b.can_claim_threefold_repetition()) or b.is_repetition(3) or b.halfmove_clock >= 100 or moves >= 80:
            out = "no mate"; break
        if b.turn == chess.WHITE:
            tr, dec, *_ = st.observe(b.fen())
            if not dec:
                surprises.append(dict(kind="no plan", fen=b.fen(), why="no frame decided"))
                break
            cert = [l for l in tr if "against every reply:" in l]
            proven = any(l.startswith("mate:") and "proven" in l for l in tr)
            why = cert[0].split("against every reply: ")[-1] if cert else ""
            if not proven and (not cert or why.startswith(("no progress", "no component"))):  # an "entry:" move is certified by Reach.bind
                surprises.append(dict(kind="uncertified", fen=b.fen(), move=dec[1], why=why or "no certificate line"))
            b.push_uci(dec[1]); moves += 1
        else:
            st.record(b.fen())
            b.push(eng.play(b, chess.engine.Limit(depth=10)).move)
    eng.quit()
    if out != "mate":
        surprises.append(dict(kind="no mate", fen=b.fen(), why=out))
    return dict(seed=seed, out=out, moves=moves, surprises=surprises)


def observe(piece: str, games: int, workers: int = 6, flags: str = "", term: tuple | None = None) -> list:
    from multiprocessing import Pool
    with Pool(workers) as p:
        res = list(p.imap_unordered(_game, [(g, piece, flags, term) for g in range(games)], chunksize=1))
    sur = [s for r in res for s in r["surprises"]]
    kinds = Counter(s["kind"] for s in sur)
    moves = sum(r["moves"] for r in res)
    mated = [r["moves"] for r in res if r["out"] == "mate"]
    print(f"OBS observed {games} K+{piece} v K games against Stockfish ({flags or 'the default frame'}{f'; term {term}' if term else ''}): "
          f"{Counter(r['out'] for r in res)}, {sum(mated) / max(len(mated), 1):.1f} moves per mate; {moves} of our moves; "
          f"surprises {dict(kinds)} ({100 * len(sur) / max(moves, 1):.2f}% of our moves)")
    whys = Counter(s["why"].split(" (")[0][:70] for s in sur)
    for w, n in whys.most_common(5):
        print(f"OBS   {n:4d}  {w}")
    return sur


def show(surprises: list, lines: int = 30) -> None:
    """UNDERSTAND the surprise: each board, the move played, and the planner's own trace (its certificate's account)."""
    import chess
    import analyst_core
    for s in surprises:
        print(f"OBS surprise ({s['kind']}): {s['fen']} played {s.get('move')} -- {s['why']}")
        tr, dec, *_ = analyst_core.PlanState().observe(s["fen"])
        for l in tr[:lines]:
            print(f"      {l[:220]}")
        print(chess.Board(s["fen"]))


def address(piece: str, surprises: list, budget: int, flags: str = "") -> dict | None:
    """The surprise boards as a hole on the frame's current measure; the pipeline proposes; the verifier decides."""
    from . import holes as H
    boards = sorted({s["fen"] for s in surprises if s["kind"] == "uncertified"})
    if len(boards) < 5:
        print(f"OBS   {len(boards)} uncertified boards -- too few to address (no hole made)")
        return None
    suffix, spec = frame_spec(piece, flags)
    name = f"surprise_k{piece.lower()}" + (f"_{suffix}" if suffix else "")
    hole = dict(name=name, material=f"K{piece}", base=spec, k=H.K, stuck=boards, sampled=len(boards))
    json.dump(hole, open(paths.derived(f"hole_{name}.json"), "w"))
    print(f"OBS   hole {name}: {len(boards)} surprise boards, base {hole['base']} -> the pipeline")
    from .pipeline import run
    res = run(name, budget, hole=hole)
    base, best = res[0], max(res[1:], key=lambda r: r["score"]) if len(res) > 1 else None
    if best and best["score"] > base["score"] + 0.05:
        print(f"OBS   ADDRESSED on the surprise boards: {best['term']} [{best['schema']}] scores {best['score']:.3f} against the frame's own "
              f"{base['score']:.3f} -- next, whole games with it in play")
        return best
    print(f"OBS   not addressed: no proposal beats the frame's own measure ({base['score']:.3f}) by 0.05 on these boards")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", default="R")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--show", action="store_true", help="print each surprise board with the planner's trace")
    ap.add_argument("--flags", default="", help="name=0|1,... (set_<name>): which frame plays, e.g. kr_generated=0,kr_chase=0")
    a = ap.parse_args(argv)
    sur = observe(a.piece, a.games, flags=a.flags)
    if a.show:
        show(sur)
    best = address(a.piece, sur, a.budget, a.flags)
    if best and a.piece == "R" and "kr_generated=0" in a.flags and "kr_chase=0" in a.flags:
        # AGAIN: the same games with the proposal in play (set_kr_term joins it to the derived frame by its schema)
        observe(a.piece, a.games, flags=a.flags, term=(best["schema"], best["term"]))


if __name__ == "__main__":
    main()
