"""HOW DEEP DOES THE ANSWER LIE? (Kaveh 2026-09-30: "before deciding how deep to go, maybe you can learn from
Stockfish: how deep does Stockfish go past the point it should prune while it achieves quiescence? that's the same
order of plies we have to go").

For every solver step of the 300 test puzzles, Stockfish (one thread) analyses to depth 24 and every depth is kept:
its best move, `seldepth` (the deepest ply any line reached: extensions and quiescence past the nominal depth) and
its score. Printed:
- CONVERGENCE depth: from which nominal depth its best move no longer changes; SOLUTION depth: from which it is the
  puzzle's move and stays; seldepth there, and seldepth - depth (how far past the nominal depth it looked).
- the PAYOFF PLY along its final principal variation: the ply from which the side to move's running material has
  secured the line's gain (capped at a minor piece) and keeps it (a mate: the mate's ply) -- the horizon a search must reach to SEE
  the point, comparable to the zone search's plies (Stockfish's nominal depth is pruned, so not comparable itself).

    python -m catspace.incubator.analyst.sf_depth --n 300
"""
from __future__ import annotations

import argparse
import json
import shutil

import chess
import chess.engine

from catspace.io import paths

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_ENG = {}


def _engine():
    if "e" not in _ENG:
        e = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
        e.configure({"Threads": 1, "Hash": 64})
        _ENG["e"] = e
    return _ENG["e"]


def payoff_ply(b: chess.Board, pv: list, score) -> int | None:
    """The ply of the principal variation from which the mover's running material stays within half a pawn of the
    line's final material (a mate: the mating ply)."""
    if score is not None and score.relative.is_mate() and score.relative.mate() > 0:
        return 2 * score.relative.mate() - 1
    me = b.turn; bb = b.copy(); run = [0.0]; bal = 0.0
    for m in pv:
        cap = bb.piece_at(m.to_square)
        if cap is None and bb.is_en_passant(m):
            cap = chess.Piece(chess.PAWN, not bb.turn)
        g = (VAL[cap.piece_type] if cap else 0) + ((VAL[m.promotion] - 1) if m.promotion else 0)
        bal += g if bb.turn == me else -g
        run.append(bal); bb.push(m)
    final = run[-1]
    if final < 0.5:
        return None                                                   # the line wins no material (a positional or defensive point)
    T = min(final, 3.0)                                               # the gain SECURED, capped at a minor piece: equal trades later in a
    for k in range(len(run)):                                         # long line must not push the payoff out (a first version measured
        if run[k] >= T and all(x >= T - 0.5 for x in run[k:]):       # where the line's END material settled: 90th percentile 24 plies)
            return k
    return None


def _analyse(args) -> dict:
    pid, fen, sol, dmax = args
    b = chess.Board(fen)
    rows = []
    with _engine().analysis(b, chess.engine.Limit(depth=dmax, time=20.0)) as an:   # (a time guard: a full run hung once)
        for info in an:
            if "depth" in info and "pv" in info and info.get("multipv", 1) == 1 and "seldepth" in info:
                rows.append((info["depth"], info["seldepth"], info["pv"][0].uci(), info.get("score"), info["pv"]))
    by = {}
    for d, sd, mv, sc, pv in rows:
        by[d] = (sd, mv, sc, pv)
    ds = sorted(by)
    if not ds:
        return {}
    final_mv = by[ds[-1]][1]
    conv = next((d for d in ds if all(by[x][1] == final_mv for x in ds if x >= d)), None)
    sold = next((d for d in ds if all(by[x][1] == sol for x in ds if x >= d)), None)
    sd_final, _, sc_final, pv_final = by[ds[-1]]
    return dict(pid=pid, fen=fen, sol=sol, final=final_mv, final_is_sol=final_mv == sol, conv=conv, sold=sold,
                seldepth_at_conv=by[conv][0] if conv in by else None, seldepth_at_sol=by[sold][0] if sold in by else None,
                seldepth_final=sd_final, payoff=payoff_ply(b, pv_final, sc_final), pv_len=len(pv_final))


def main(argv=None):
    import multiprocessing as mp
    import statistics as st
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--misses", default="frame_puzzles_lichess_zone_candidates.json")
    a = ap.parse_args(argv)
    from .gauge_puzzles import sample
    rows = sample(str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"), a.n, 997)
    items = []
    for r in rows:
        b = chess.Board(r["FEN"]); mv = r["Moves"].split(); b.push_uci(mv[0]); sol = mv[1:]
        for i in range(0, len(sol), 2):
            items.append((f"{r['PuzzleId']}:{i // 2}", b.fen(), sol[i], a.depth))
            b.push_uci(sol[i])
            if i + 1 < len(sol):
                b.push_uci(sol[i + 1])
    out = paths.derived(f"sf_depth_{a.n}.jsonl")
    res = []
    with mp.get_context("fork").Pool(a.workers) as pool, open(out, "w") as fh:   # line by line: a stall loses nothing
        for x in pool.imap_unordered(_analyse, items, chunksize=1):
            if x:
                x = {k: v for k, v in x.items()}
                fh.write(json.dumps(x) + "\n"); fh.flush(); res.append(x)
    miss = set()
    try:
        for r in json.load(open(paths.derived(a.misses))):
            if not r["solved"]:
                k = next(i for i, s in enumerate(r["steps"]) if not s["tied"])
                miss.add(f"{r['id']}:{k}")
    except FileNotFoundError:
        pass

    def summary(name, xs):
        if not xs:
            return
        q = lambda v: (f"median {st.median(v):g}, 90th {sorted(v)[int(0.9 * (len(v) - 1))]:g}, max {max(v):g}" if v else "none")
        conv = [x["conv"] for x in xs if x["conv"] is not None]
        sold = [x["sold"] for x in xs if x["sold"] is not None]
        exc = [x["seldepth_at_sol"] - x["sold"] for x in xs if x["sold"] is not None and x["seldepth_at_sol"] is not None]
        pay = [x["payoff"] for x in xs if x["payoff"] is not None]
        print(f"OBS {name} ({len(xs)} positions; Stockfish depth {a.depth}, final move = the solution in {sum(x['final_is_sol'] for x in xs)}):")
        print(f"   convergence depth {q(conv)}; solution depth {q(sold)} (never settles on it: {len(xs) - len(sold)})")
        print(f"   seldepth past the nominal depth at the solution depth: {q(exc)}")
        print(f"   payoff ply along the final line (material realised and kept; a mate's ply): {q(pay)}; beyond 8 plies "
              f"{sum(1 for p in pay if p > 8)} of {len(pay)}; beyond 12 {sum(1 for p in pay if p > 12)}; no material won {len(xs) - len(pay)}")
    summary("all solver steps", res)
    summary("first steps", [x for x in res if x["pid"].endswith(":0")])
    summary("the zone ledger's failing steps", [x for x in res if x["pid"] in miss])
    for e in _ENG.values():
        e.quit()


if __name__ == "__main__":
    main()
