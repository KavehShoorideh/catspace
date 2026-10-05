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


MATCH_NODES: dict = {}


def _solve_lichess(row: dict) -> dict:
    from .frame_player import choose
    from .fates import clear_caches, LEMMA_MISMATCHES
    clear_caches()
    seen = len(LEMMA_MISMATCHES)                                  # verify mode (ANALYST_VERIFY_FRAMES=1): this task's lemma mismatches
    b = chess.Board(row["FEN"]); moves = row["Moves"].split()
    b.push_uci(moves[0])
    start = b.fen(); sol = moves[1:]
    steps = []
    for i in range(0, len(sol), 2):
        if MATCH_NODES:                                           # equal compute: this step's node total from another run's dump
            from . import frame_player as _FPm
            _FPm.TOTAL_CAP[0] = MATCH_NODES.get(b.fen())
        c = choose(b)
        u = sol[i]
        mate_ok = False
        if u not in c["best"]:
            nb = b.copy(); nb.push_uci(u)
            if nb.is_checkmate():                                 # any mate is a solution
                mate_ok = any(_mates(b, x) for x in c["best"])
        ok_tied = u in c["best"] or mate_ok
        ok_unique = (c["best"] == [u]) or (mate_ok and len(c["best"]) == 1)
        from . import frame_player as _FP
        steps.append(dict(fen=b.fen(), sol=u, tied=ok_tied, unique=ok_unique, n_top=len(c["best"]), best=c["best"],
                          v_sol=c["values"].get(u), v_top=max(c["values"].values()), nodes=_FP.NODES[0], depth=_FP.DEPTH[0],
                          proved=_FP.PROVED[0], passer_leaves=_FP.PASSER_LEAVES[0], order=_FP.LOOP_ORDER_FIRED[0]))
        if not ok_tied:
            break
        b.push_uci(u)
        if i + 1 < len(sol):
            b.push_uci(sol[i + 1])
    return dict(id=row.get("PuzzleId"), rating=int(row.get("Rating", 0) or 0), themes=row.get("Themes", "").split(), start=start,
                lemma_mismatches=[list(x[:2]) + [x[2], x[3]] for x in LEMMA_MISMATCHES[seen:]][:5], n_lemma_mismatches=len(LEMMA_MISMATCHES) - seen,
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
                steps=[dict(fen=fen, sol=us[0], tied=tied, unique=unique, n_top=len(c["best"]), best=c["best"],
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


def fresh_rows(n: int, stride: int, offset: int, theme: str | None = None) -> list:
    """Puzzles from rows i % stride == offset of the 6.1M-puzzle Lichess file, excluding the rows already used: the test
    sample (i % 997 == 0) and the proposer's and the quiet set's rows (i % 97 in 49, 73); with `theme`, only puzzles
    carrying it."""
    import csv, io, zstandard
    out = []
    with open(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst", "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % stride == offset and i % 997 != 0 and i % 97 not in (49, 73) and (theme is None or theme in row["Themes"].split()):
                out.append(row)
                if len(out) >= n:
                    break
    return out


def compare(before: str, after: str) -> None:
    """Two runs on the same puzzles: solved counts, puzzles gained and lost, and whether each miss's solution move was
    ever a candidate (the reply ply's candidate set) or was one and was valued too low."""
    import json
    x = {r["id"]: r for r in json.load(open(paths.derived(before)))}
    y = {r["id"]: r for r in json.load(open(paths.derived(after)))}
    common = [k for k in x if k in y]
    def random_pick(r):                                           # a player must pick ONE move: with a random pick among the tied
        if not r["solved"]:                                       # top moves, the chance it solves every step (the fair control
            return 0.0                                            # for any tie-break, 2026-10-01)
        p = 1.0
        for st_ in r["steps"]:
            p *= 1.0 / max(1, st_["n_top"])
        return p
    for name, d in ((before, x), (after, y)):
        print(f"OBS {name}: whole solution among the tied top moves {sum(d[k]['solved'] for k in common)}/{len(common)}; "
              f"single top move {sum(d[k]['solved_unique'] for k in common)}/{len(common)}; "
              f"a random pick among ties, expected {sum(random_pick(d[k]) for k in common):.1f}")
    gained = [k for k in common if y[k]["solved"] and not x[k]["solved"]]
    lost = [k for k in common if x[k]["solved"] and not y[k]["solved"]]
    print(f"OBS {after} against {before}: gained {len(gained)}, lost {len(lost)} ({', '.join(lost)})")
    pairs = [(sx["nodes"], sy["nodes"]) for k in common for sx in x[k]["steps"] for sy in y[k]["steps"]
             if sx["fen"] == sy["fen"] and sx.get("nodes") is not None and sy.get("nodes") is not None]
    if pairs:                                                     # the search's work on the same positions (paired steps)
        nx = sum(a for a, _ in pairs); ny = sum(c for _, c in pairs)
        print(f"OBS zone-search nodes on the {len(pairs)} steps both runs searched: {nx} -> {ny} ({ny / max(nx, 1):.3f}); "
              f"fewer {sum(c < a for a, c in pairs)}, more {sum(c > a for a, c in pairs)}, equal {sum(c == a for a, c in pairs)}")
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
    ap.add_argument("--ledger", default="zone_candidates", choices=["one_ply", "two_ply", "two_ply_candidates", "two_ply_instinct", "zone_candidates"])   # tactics need the reply ply (2026-09-28); the zone verifier (2026-09-29)
    ap.add_argument("--propose", type=int, default=0, help="zone_candidates: the moves realizing the what-if proposer's top-k edits join the candidates")
    ap.add_argument("--quiet-offset", type=int, default=73, help="lichess-quiet: rows i %% 97 == offset with a quiet theme (the proposer trained on 49)")
    ap.add_argument("--check-ext", type=int, default=None, help="zone search: checks that do not use up a ply, per line (zone_search.CHECK_EXT)")
    ap.add_argument("--offset", type=int, default=0, help="lichess: fresh rows i %% stride == offset (never the test sample's, the proposer's or the quiet set's rows)")
    ap.add_argument("--match-nodes", default=None, help="a dump (data/derived name) whose per-step nodes cap this run's steps: equal compute")
    ap.add_argument("--budget-policy", default="fixed", choices=["fixed", "watch", "learned", "decider"], help="zone search budget: equal shares, the king watches' rule, the learned policy, or the learned depth decider")
    ap.add_argument("--decider-threshold", type=float, default=0.5, help="--budget-policy decider: go on past a rung when the chance it changes the decision is at least this")
    ap.add_argument("--decider-model", default=None, help="--budget-policy decider: the model's data/derived name (default depth_decider.pkl)")
    ap.add_argument("--decider-price", type=float, default=None, help="--budget-policy decider: a price on search instead (go on when the chance >= price x the next rung's nodes)")
    ap.add_argument("--armed", nargs="*", default=[], choices=["discovery", "pin", "backrank", "fork", "overload"], help="armed lemmas: strikes at once, lures as candidates")
    ap.add_argument("--lemmas", action="store_true", help="the learned high-value lemma detectors as the tie-break (lemma_detect)")
    ap.add_argument("--imagine", type=int, default=0, help="the top-k imagined frames: their moves join the candidates, their squares the zone")
    ap.add_argument("--leaf-mates", action="store_true", help="zone search: a mate in one for the side to move is seen at the horizon")
    ap.add_argument("--zugzwang-guard", action="store_true", help="zone search: a king-and-pawns side may not pass inside the tree")
    ap.add_argument("--proved-mates", type=int, default=0, choices=[0, 1, 2], help="zone search: a back-rank mate proved in Lean closes the node "
                    "(1 inside the tree, 2 also at the horizon)")
    ap.add_argument("--theme", default=None, help="lichess --offset: only puzzles with this theme (e.g. backRankMate)")
    ap.add_argument("--proved-passer", action="store_true", help="the passer lemma (second form, prototype): its queen at the horizon, its push a candidate")
    ap.add_argument("--tag", default="", help="a suffix for the dump's name (keeps an earlier run's dump from being overwritten)")
    ap.add_argument("--loop-active", default=None, help="the idea loop's wiring (loop/use stage active.json): proved closures and mined tie-breaks")
    ap.add_argument("--frame-ext", type=int, default=0, help="zone search: replies on the same square do not use up a ply (up to N a line)")
    ap.add_argument("--pawn-moves", action="store_true", help="every pawn move is a candidate and is searched in the zone search")
    ap.add_argument("--only-unsolved-in", default=None, help="lichess --offset: only the puzzles this earlier dump (data/derived name) missed")
    ap.add_argument("--depths", type=int, nargs="+", default=None, help="the zone search's rungs (the planner's; default TacticalSpec.depths)")
    ap.add_argument("--watch-zone", type=int, default=0, help="zone search: the king watches' top-N pursued members join the root zone")
    ap.add_argument("--king-watch", action="store_true", help="zone_candidates: the standing king frames' moves join the candidates")
    ap.add_argument("--jobs", action="store_true", help="the job ledger: its knowledge orders the zone search, its moves join the candidates")
    ap.add_argument("--prune-jobs", type=int, default=0, help="with --jobs: drop moves that abandon a job near the horizon (depth <= N)")
    ap.add_argument("--trades", action="store_true", help="the planner's material-count trade preference (a tie-break; measured)")
    ap.add_argument("--no-runners", action="store_true", help="zone search: no passed pawns in the root zone (ablation)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="two runs' dumps (data/derived names): what changed, no solving")
    a = ap.parse_args(argv)
    if a.compare:
        return compare(*a.compare)
    from . import frame_player as FP
    FP.LEDGER = a.ledger; FP.PROPOSE = a.propose
    tag = a.ledger + (f"_propose{a.propose}" if a.propose else "")
    if a.match_nodes:
        import json as _json
        for r in _json.load(open(paths.derived(a.match_nodes))):
            for s in r["steps"]:
                if s.get("nodes"):
                    MATCH_NODES[s["fen"]] = s["nodes"]
        tag += "_matched"
    if a.budget_policy != "fixed":
        FP.BUDGET_POLICY = a.budget_policy; tag += f"_budget{a.budget_policy}"
        if a.budget_policy == "decider":
            FP.DECIDER_THRESHOLD[0] = a.decider_threshold; FP.DECIDER_PRICE[0] = a.decider_price
            tag += f"_price{a.decider_price:g}" if a.decider_price is not None else f"{a.decider_threshold:g}"
            if a.decider_model:
                from . import depth_decider as _DD
                _DD.MODEL_NAME = a.decider_model; tag += "_" + a.decider_model.rsplit(".", 1)[0]
    planner_kw = {}                                               # every search limit and goal is the PLANNER's (frame_planner)
    if a.watch_zone:
        planner_kw["watch_zone"] = a.watch_zone; tag += f"_watchzone{a.watch_zone}"
    if a.king_watch:
        FP.KING_WATCH = True; tag += "_kingwatch"
    if a.no_runners:
        planner_kw["runner_steps"] = None; tag += "_norunners"
    if a.trades:
        planner_kw["trades"] = True; tag += "_trades"
    if a.imagine:
        planner_kw["imagine"] = a.imagine; FP.IMAGINE = a.imagine; tag += f"_imagine{a.imagine}"
    if a.lemmas:
        planner_kw["lemmas"] = True; tag += "_lemmas"
    for name in a.armed:                                          # the armed lemmas (discovery, pin, backrank)
        FP.ARMED.add(name); tag += f"_{name}"
    if a.jobs:
        planner_kw["jobs"] = True; FP.JOBS = True; tag += "_jobs"
    spec_kw = {}                                                  # the tactical spec's limits, all the planner's
    if a.prune_jobs:
        spec_kw["prune_jobs"] = a.prune_jobs; tag += f"_prunejobs{a.prune_jobs}"
    if a.check_ext is not None:
        spec_kw["check_ext"] = a.check_ext; tag += f"_ext{a.check_ext}"
    if a.depths:
        spec_kw["depths"] = tuple(sorted(a.depths)); tag += "_d" + "-".join(map(str, sorted(a.depths)))
    if a.pawn_moves:
        spec_kw["pawn_moves"] = True; FP.PAWN_MOVES = True; tag += "_pawnmoves"
    if a.frame_ext:
        spec_kw["frame_ext"] = a.frame_ext; tag += f"_frameext{a.frame_ext}"
    if a.leaf_mates:
        spec_kw["leaf_mates"] = True; tag += "_leafmates"
    if a.zugzwang_guard:
        spec_kw["zugzwang_guard"] = True; tag += "_zzguard"
    if a.proved_mates:
        spec_kw["proved_mates"] = a.proved_mates; tag += f"_proved{a.proved_mates}"
    if a.proved_passer:
        spec_kw["proved_passer"] = True; FP.PASSER = True; tag += "_passer"
    if a.loop_active:                                             # the idea loop's wiring for a round (loop/wiring.py)
        import json as _jl
        from .loop.wiring import activate
        tag += activate(_jl.load(open(a.loop_active)), FP, spec_kw)
    if spec_kw:
        from .zone_search import TacticalSpec
        planner_kw["spec"] = TacticalSpec(**spec_kw)
    if planner_kw:
        from .frame_planner import MaterialPlanner
        FP.PLANNER = MaterialPlanner(**planner_kw)
    t0 = time.perf_counter()
    import multiprocessing as mp
    if a.source in ("lichess", "lichess-quiet"):
        if a.source == "lichess" and a.offset:
            rows = fresh_rows(a.n, a.stride, a.offset, a.theme); tag += f"_off{a.offset}" + (f"_{a.theme}" if a.theme else "")
            if a.only_unsolved_in:                                  # the misses of an earlier run, replayed (a targeted gauge)
                import json as _jm                                  # (main imports json later: a local name here)
                miss = {r["id"] for r in _jm.load(open(paths.derived(a.only_unsolved_in))) if not r["solved"]}
                rows = [r for r in rows if r["PuzzleId"] in miss]; tag += "_onlymisses"
        elif a.source == "lichess":
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
    ns = [s.get("nodes", 0) for r in res for s in r["steps"]]
    if any(ns):
        print(f"OBS zone search nodes per solver step: mean {sum(ns) / len(ns):.0f} over {len(ns)} steps")
        pv = [s.get("proved", 0) for r in res for s in r["steps"]]
        if any(pv):
            print(f"OBS proved back-rank mates closing zone-search nodes: {sum(pv)} in {sum(1 for x in pv if x)}/{len(pv)} steps")
        pl = [s.get("passer_leaves", 0) for r in res for s in r["steps"]]
        if any(pl):
            print(f"OBS leaves credited with a proved passer: {sum(pl)} in {sum(1 for x in pl if x)}/{len(pl)} steps")
        po = [s.get("order", 0) for r in res for s in r["steps"]]
        if any(po):
            print(f"OBS the loop's order use broke a tie in {sum(po)}/{len(po)} steps")
        ds = [s.get("depth", 0) for r in res for s in r["steps"]]
        print("OBS the deepest rung completed per solver step: " + ", ".join(f"{d}: {ds.count(d)}" for d in sorted(set(ds))))
    from .fates import VERIFY_FRAMES
    if VERIFY_FRAMES:                                             # the lemma library served a value its fresh resolution disagrees with
        bad = [r for r in res if r.get("n_lemma_mismatches")]
        print(f"OBS verify mode: lemma mismatches {sum(r.get('n_lemma_mismatches', 0) for r in res)} in {len(bad)}/{n} puzzles"
              + "".join(f"\n   {r['id']}: {m[0]} squares {[chess.square_name(s) for s in m[1]]} lemma {m[2]:+g} fresh {m[3]:+g}" for r in bad[:6] for m in r["lemma_mismatches"][:1]))
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
    tag += f"_{a.tag}" if a.tag else ""
    json.dump(res, open(paths.derived(f"frame_puzzles_{a.source}_{tag}.json"), "w"))   # every puzzle's steps, for the dig
    print(f"wrote {paths.derived(f'frame_puzzles_{a.source}_{tag}.json')}")
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
