"""THE FRAME PLAYER ON TACTICAL PUZZLES (Kaveh 2026-09-28: "let's try solving some tactical puzzles and see where do we
make mistakes in tactical puzzles and then use that to improve").

Lichess puzzles: the solver's position is the FEN after the puzzle's first move; at each of the solver's steps the frame
player chooses (`frame_player.choose`), and the step counts when the solution's move is its single top move ('unique')
or among its tied top moves ('tied'); any mating move counts where the solution mates. The solution's own moves are then
played and the next step is asked. WAC (Win At Chess): the suite's best move, one try. Misses are printed with the
solution line, what the frames chose and why, and the solution move's value -- for the dig.

    python -m catspace.incubator.analyst.frame_puzzles --n 300                 # Lichess
    python -m catspace.incubator.analyst.frame_puzzles --source wac --n 300
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import chess

from catspace.io import paths


def _solve_lichess(row: dict) -> dict:
    from .frame_player import choose
    from .fates import clear_caches
    clear_caches()
    b = chess.Board(row["FEN"]); moves = row["Moves"].split()
    b.push_uci(moves[0])
    start = b.fen(); sol = moves[1:]
    steps = []
    for i in range(0, len(sol), 2):
        c = choose(b)
        u = sol[i]
        mate_ok = False
        if u not in c["best"]:
            nb = b.copy(); nb.push_uci(u)
            if nb.is_checkmate():                                 # any mate is a solution
                mate_ok = any(_mates(b, x) for x in c["best"])
        ok_tied = u in c["best"] or mate_ok
        ok_unique = (c["best"] == [u]) or (mate_ok and len(c["best"]) == 1)
        steps.append(dict(fen=b.fen(), sol=u, tied=ok_tied, unique=ok_unique, n_top=len(c["best"]), best=c["best"][:6],
                          v_sol=c["values"].get(u), v_top=max(c["values"].values())))
        if not ok_tied:
            break
        b.push_uci(u)
        if i + 1 < len(sol):
            b.push_uci(sol[i + 1])
    return dict(id=row.get("PuzzleId"), rating=int(row.get("Rating", 0) or 0), themes=row.get("Themes", "").split(), start=start,
                solution=sol, steps=steps, first_tied=steps[0]["tied"], first_unique=steps[0]["unique"],
                solved=all(s["tied"] for s in steps) and len(steps) == (len(sol) + 1) // 2,
                solved_unique=all(s["unique"] for s in steps) and len(steps) == (len(sol) + 1) // 2)


def _mates(b: chess.Board, u: str) -> bool:
    nb = b.copy(); nb.push_uci(u); return nb.is_checkmate()


def _solve_epd(item) -> dict:
    from .frame_player import choose
    from .fates import clear_caches
    clear_caches()
    pid, fen, bms = item
    b = chess.Board(fen); c = choose(b)
    us = [m.uci() for m in bms]
    tied = any(u in c["best"] for u in us); unique = len(c["best"]) == 1 and c["best"][0] in us
    return dict(id=pid, rating=0, themes=[], start=fen, solution=us[:1],
                steps=[dict(fen=fen, sol=us[0], tied=tied, unique=unique, n_top=len(c["best"]), best=c["best"][:6],
                            v_sol=c["values"].get(us[0]), v_top=max(c["values"].values()))],
                first_tied=tied, first_unique=unique, solved=tied, solved_unique=unique)


def _line(fen: str, ucis: list) -> str:
    b = chess.Board(fen); out = []
    for u in ucis:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            break
        out.append(b.san(m)); b.push(m)
    return " ".join(out)


def compare(before: str, after: str) -> None:
    """Two runs on the same puzzles: solved counts, puzzles gained and lost, and whether each miss's solution move was
    ever a candidate (the reply ply's candidate set) or was one and was valued too low."""
    import json
    x = {r["id"]: r for r in json.load(open(paths.derived(before)))}
    y = {r["id"]: r for r in json.load(open(paths.derived(after)))}
    common = [k for k in x if k in y]
    for name, d in ((before, x), (after, y)):
        print(f"OBS {name}: whole solution among the tied top moves {sum(d[k]['solved'] for k in common)}/{len(common)}; "
              f"single top move {sum(d[k]['solved_unique'] for k in common)}/{len(common)}")
    gained = [k for k in common if y[k]["solved"] and not x[k]["solved"]]
    lost = [k for k in common if x[k]["solved"] and not y[k]["solved"]]
    print(f"OBS {after} against {before}: gained {len(gained)}, lost {len(lost)} ({', '.join(lost)})")
    for name, d in ((before, x), (after, y)):
        miss = [k for k in common if not d[k]["solved"]]
        never = [k for k in miss if next((s for s in d[k]["steps"] if not s["tied"]), {}).get("v_sol", 0) is None]
        print(f"OBS {name}: misses {len(miss)}; the solution move was never a candidate in {len(never)} ({', '.join(never)}), "
              f"a candidate valued below the top in {len(miss) - len(never)}")


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="lichess", choices=["lichess", "lichess-quiet", "wac", "bk"]); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stride", type=int, default=997); ap.add_argument("--workers", type=int, default=10); ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--ledger", default="two_ply_candidates", choices=["one_ply", "two_ply", "two_ply_candidates", "two_ply_instinct", "zone_candidates"])   # tactics need the reply ply (2026-09-28)
    ap.add_argument("--propose", type=int, default=0, help="zone_candidates: the moves realizing the what-if proposer's top-k edits join the candidates")
    ap.add_argument("--quiet-offset", type=int, default=73, help="lichess-quiet: rows i %% 97 == offset with a quiet theme (the proposer trained on 49)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="two runs' dumps (data/derived names): what changed, no solving")
    a = ap.parse_args(argv)
    if a.compare:
        return compare(*a.compare)
    from . import frame_player as FP
    FP.LEDGER = a.ledger; FP.PROPOSE = a.propose
    tag = a.ledger + (f"_propose{a.propose}" if a.propose else "")
    t0 = time.perf_counter()
    import multiprocessing as mp
    if a.source in ("lichess", "lichess-quiet"):
        if a.source == "lichess":
            from .gauge_puzzles import sample
            rows = sample(str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"), a.n, a.stride)
        else:                                                     # quiet-theme puzzles held out from the proposer's training rows
            from .whatif_proposer import training_rows
            rows = training_rows(a.n, a.quiet_offset, stride=97)
            tag += f"_quiet{a.quiet_offset}"
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(_solve_lichess, rows, chunksize=2)
    else:
        from .gauge_epd import load
        items = load(str(paths.assets_dir() / "suites" / ("WAC.epd" if a.source == "wac" else "BK.epd")))[:a.n]
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(_solve_epd, [(pid, b.fen(), bms) for pid, b, bms in items], chunksize=2)
    n = len(res)
    print(f"[frame_puzzles] {a.source}: {n} puzzles, ledger {tag}, {time.perf_counter() - t0:.0f}s")
    print(f"OBS first move: the frames' single top move {sum(r['first_unique'] for r in res)}/{n}; among the tied top moves {sum(r['first_tied'] for r in res)}/{n} "
          f"(top set size, median {sorted(r['steps'][0]['n_top'] for r in res)[n // 2]})")
    if a.source in ("lichess", "lichess-quiet"):
        print(f"OBS the whole solution (every step of the solver): single top move each time {sum(r['solved_unique'] for r in res)}/{n}; "
              f"among the tied top moves each time {sum(r['solved'] for r in res)}/{n}")
        bands = [(0, 1200), (1200, 1600), (1600, 2000), (2000, 4000)]
        print("OBS by rating, whole solution among the tied top moves: " + "; ".join(
            f"{lo}-{hi}: {sum(r['solved'] for r in res if lo <= r['rating'] < hi)}/{sum(1 for r in res if lo <= r['rating'] < hi)}" for lo, hi in bands))
        th = Counter(t for r in res for t in r["themes"])
        rows = []
        for t, k in th.most_common(40):
            if k < 12 or t in ("short", "long", "veryLong", "middlegame", "endgame", "opening", "master", "masterVsMaster", "superGM", "crushing", "advantage", "one Move"):
                continue
            rows.append(f"{t} {sum(r['solved'] for r in res if t in r['themes'])}/{k}")
        print("OBS by theme, whole solution among the tied top moves: " + "; ".join(rows[:16]))
    import json
    json.dump(res, open(paths.derived(f"frame_puzzles_{a.source}_{tag}.json"), "w"))   # every puzzle's steps, for the dig
    misses = [r for r in res if not r["solved"]]
    for r in misses[:a.show]:
        s = next(x for x in r["steps"] if not x["tied"])
        b = chess.Board(s["fen"])
        print(f"\nMISS {r['id']} (rating {r['rating']}; {' '.join(r['themes'][:5])}) {s['fen']}  {'White' if b.turn else 'Black'} to move")
        print(f"  solution from here: {_line(s['fen'], r['solution'][2 * r['steps'].index(s):][:8])}")
        print(f"  the frames' top ({s['n_top']} tied at {s['v_top']:+g}): {', '.join(b.san(chess.Move.from_uci(u)) for u in s['best'])}; "
              f"the solution move {b.san(chess.Move.from_uci(s['sol']))} " + (f"scores {s['v_sol']:+g}" if s['v_sol'] is not None else "was not a candidate for the reply ply"))


if __name__ == "__main__":
    main()
