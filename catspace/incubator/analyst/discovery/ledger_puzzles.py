"""THE LEDGER'S DECISIONS ON PUZZLES (Kaveh 2026-10-14; ANALYST_PLAN 39.4 item 3): the reasoning ledger (reasoning.rs `Ledger.decide`) chooses the
move; the same decision is made by the exact solver alone (forced_moves over every legal move at the puzzle's value, the smallest depth at which it
finds a winning move, cap 6, nodes counted) and by the existing planner. Lichess puzzles: the opponent's Moves[0] is pushed, the answer is Moves[1]
(correct = equal to it, or for a mate puzzle any move that mates as fast).

  DEV   rows i % 997 == 5 and i % 200 != 0 (clear of both held-out samples) -- for building
  TEST  gauge_puzzles.sample(n=300, stride 200, offset 0) -- run once, at the end

  python -m catspace.incubator.analyst.discovery.ledger_puzzles --set dev --n 20
  python -m catspace.incubator.analyst.discovery.ledger_puzzles --set dev --n 300 --misses 10
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing
import os
import re
import time
from collections import defaultdict

import chess

from catspace.io import paths
from catspace.incubator.analyst.discovery import lemma_store as LS

VAL = LS.valuation().by_kind(royal=False)        # plan 76.5: the store's valuation (the royal unit aside), not a table of its own
CAP_DEPTH = 6
BUDGET = 1_000_000
NODE_CAP = 30_000_000           # the solver alone stops deepening past this many nodes (not found)


def _csv_path() -> str:
    return str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst")


def dev_rows(n: int, offset: int = 5) -> list:
    """offset 5: the DEV set; offset 11: the FRESH sample (untouched, disjoint from dev and from the test)."""
    import zstandard
    out = []
    with open(_csv_path(), "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % 997 == offset and i % 200 != 0:
                out.append(row)
                if len(out) >= n:
                    break
    return out


def test_rows(n: int) -> list:
    from catspace.incubator.analyst import gauge_puzzles
    return gauge_puzzles.sample(_csv_path(), n=n)


def _balance(b: chess.Board, color: bool) -> int:
    return sum(VAL.get(p.piece_type, 0) * (1 if p.color == color else -1) for p in b.piece_map().values())


def _setup(row: dict):
    b = chess.Board(row["FEN"])
    mv = row["Moves"].split()
    b.push_uci(mv[0])
    return b, mv[1:], row["Themes"].split()


def _mate_n(themes: list):
    for t in themes:
        m = re.fullmatch(r"mateIn(\d)", t)
        if m:
            return int(m.group(1))
    return None


def _value(b: chess.Board, sol: list, themes: list) -> int:
    if _mate_n(themes):
        return 100
    c = b.turn
    x = b.copy()
    base = _balance(x, c)
    for u in sol:
        x.push_uci(u)
    return max(1, _balance(x, c) - base)


def _correct(fen: str, move: str, answer: str, themes: list, A) -> bool:
    if move == answer:
        return True
    n = _mate_n(themes)
    if n and move:
        return any(r[0] == move and r[1] == "forced" and r[2] <= n for r in A.forced_moves(fen, 100, n, BUDGET))
    return False


LEDGER_BUDGET = 200


def one(row: dict) -> dict:
    only = os.environ.get("LEDGER_ONLY_IDS")
    if only and row["PuzzleId"] not in only.split(","):
        return None
    return _one(row, int(os.environ.get("LEDGER_BUDGET", "200")))


def _one(row: dict, budget: int) -> dict:
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import catspace.incubator.analyst.discovery  # noqa: F401  (installs the chess terms)
    import analyst_core as A
    from catspace.incubator.analyst.discovery import explorer
    b, sol, themes = _setup(row)
    fen, answer = b.fen(), sol[0]
    out = dict(id=row["PuzzleId"], fen=fen, answer=answer, themes=themes, legal=b.legal_moves.count())
    if os.environ.get("LEDGER_LOOP", "script") == "event":
        try:
            from catspace.incubator.analyst.discovery import ledger as LG
            t0 = time.perf_counter()
            L = LG.new(fen, 5000)
            LG.load_transitions(L)
            mode = os.environ.get("LEDGER_LEMMAS", "off")
            if mode in ("learn", "on"):
                text = open(os.environ["LEDGER_LEMMA_FILE"]).read() if mode == "on" and os.path.exists(os.environ["LEDGER_LEMMA_FILE"]) else ""
                L.set_lemmas(text, int(os.environ.get("LEDGER_STAKE_KNOB", "3")), 5, mode == "learn")
            L.set_time_limit(float(os.environ.get("LEDGER_TIME_LIMIT", "300")))             # a decision is cut off after this many seconds (the check of a puzzle that would not end)
            prop = json.loads(os.environ.get("LEDGER_PROPOSAL", "null") or "null")
            if prop:
                from catspace.incubator.analyst.discovery import proposer as PR
                L.set_proposal(prop.get("tag_bonus") or {}, PR.rule_moves(fen, prop.get("rules") or []))
            d = L.decide_chain({}, 120, 20)
            out["ledger"] = dict(move=d["move"], status=d["status"].split(" ")[0], reason="the chain", interval=d["interval"], questions=d["questions"], nodes=d["questions"], facts=d["facts"],
                                 candidates=len(d["candidates"]), legal=d["legal"], refined=0, zero_stake=0, seconds=round(time.perf_counter() - t0, 3), confirm=None, why="",
                                 all=[[c["uci"], [], 0, [c["lo"], c["hi"]]] for c in d["candidates"]], generated=d["generated"], unanswered=[], chain=[], table=[], lines=d["lines"],
                                 constraints=len(d["constraints"]), opportunities=len(d["opportunities"]), lemma=d.get("lemma"),
                                 rows=d["candidates"], tags=d.get("tags", {}), plies=len(sol), cut=bool(d.get("cut")))
            out["ledger"]["correct"] = _correct(fen, d["move"], answer, themes, A)
            out["ledger"]["alt_move"] = d["move"]; out["ledger"]["alt_correct"] = out["ledger"]["correct"]
        except BaseException as e:                           # noqa: BLE001
            out["ledger"] = dict(error=repr(e), correct=False, status="ERROR", questions=0, nodes=0, facts=0, candidates=0, legal=out["legal"], seconds=0)
        return _baselines(out, row, b, sol, themes, fen, answer, A, explorer)
    try:
        t0 = time.perf_counter()
        d = A.Ledger(fen, budget).decide(os.environ.get("LEDGER_FORCING", "threats"))
        out["ledger"] = dict(move=d["move"], status=d["status"], reason=d["reason"], interval=d["interval"], questions=d["questions"], nodes=d["nodes"], facts=d["facts"],
                             candidates=d["candidates"], legal=d["legal"], refined=d["refined"], zero_stake=d["zero_stake"], seconds=round(time.perf_counter() - t0, 3),
                             confirm=d["confirm"], why=d["why"], all=d["all"], unanswered=d["unanswered"], chain=d["chain"], table=d["table"])
        out["ledger"]["correct"] = _correct(fen, d["move"], answer, themes, A)
        # THE VARIANT chosen on the dev set: an ESTIMATED decision takes the best LOWER bound (ties: the upper, then forcing moves) instead of the midpoint
        alt = d["move"]
        if d["status"] == "ESTIMATED":
            forcing = lambda c: any(w in ("capture", "check", "promotion") for w in c[1])
            alt = max(d["all"], key=lambda c: (c[3][0], c[3][1], forcing(c)))[0]
        out["ledger"]["alt_move"] = alt
        out["ledger"]["alt_correct"] = _correct(fen, alt, answer, themes, A)
    except BaseException as e:                               # noqa: BLE001
        out["ledger"] = dict(error=repr(e), correct=False, status="ERROR", questions=0, nodes=0, facts=0, candidates=0, legal=out["legal"], seconds=0)
    return _baselines(out, row, b, sol, themes, fen, answer, A, explorer)


def _baselines(out, row, b, sol, themes, fen, answer, A, explorer):
    if os.environ.get("LEDGER_NO_BASELINES") == "1":
        out["solver"] = dict(move=None, depth=None, nodes=0, seconds=0, correct=False)
        out["planner"] = dict(move=None, correct=False)
        out["value"] = _value(b, sol, themes)
        return out
    # the solver alone: every legal move at the puzzle's value, deeper until a move is found
    V = _value(b, sol, themes)
    out["value"] = V
    A.solver_nodes_reset()
    t0 = time.perf_counter()
    sm, depth = None, None
    for d_ in range(1, CAP_DEPTH + 1):
        res = A.forced_moves(fen, V, d_, BUDGET)
        if A.solver_nodes() > NODE_CAP and not any(r[1] == "forced" for r in res):
            break
        won = sorted([r for r in res if r[1] == "forced"], key=lambda r: r[2])
        if won:
            sm, depth = won[0][0], d_
            break
    out["solver"] = dict(move=sm, depth=depth, nodes=A.solver_nodes(), seconds=round(time.perf_counter() - t0, 3), correct=bool(sm) and _correct(fen, sm, answer, themes, A))
    try:
        pm = explorer.planner_move(fen)[0]
    except BaseException as e:                               # noqa: BLE001
        pm = None
    out["planner"] = dict(move=pm, correct=bool(pm) and _correct(fen, pm, answer, themes, A))
    return out


def summarize(res: list, label: str) -> dict:
    def block(rs: list) -> dict:
        n = len(rs)
        led = [r["ledger"] for r in rs]
        pr = [l for l in led if l.get("status") == "PROVEN"]
        cf = [l for l in led if l.get("confirm")]
        return dict(n=n, ledger_correct=sum(l["correct"] for l in led) / n, alt_correct=sum(l.get("alt_correct", False) for l in led) / n, proven=len(pr) / n,
                    proven_correct=(sum(l["correct"] for l in pr) / len(pr)) if pr else None,
                    questions=sum(l["questions"] for l in led) / n, nodes_mean=sum(l.get("nodes", 0) for l in led) / n,
                    nodes_med=sorted(l.get("nodes", 0) for l in led)[n // 2], nodes_p90=sorted(l.get("nodes", 0) for l in led)[min(n - 1, int(0.9 * n))], facts=sum(l["facts"] for l in led) / n,
                    candidates=sum(l["candidates"] for l in led) / n, legal=sum(l["legal"] for l in led) / n,
                    seconds=sum(l["seconds"] for l in led) / n, confirmed=sum(1 for l in cf if l["confirm"]["result"] == "CONFIRMED"), claims=len(cf),
                    solver_correct=sum(r["solver"]["correct"] for r in rs) / n, solver_nodes=sum(r["solver"]["nodes"] for r in rs) / n,
                    solver_found=sum(1 for r in rs if r["solver"]["move"]) / n, planner_correct=sum(r["planner"]["correct"] for r in rs) / n)
    by = defaultdict(list)
    for r in res:
        for t in r["themes"]:
            by[t].append(r)
    tot = block(res)
    themes = {t: block(rs) for t, rs in by.items() if len(rs) >= 3}
    fmt = lambda x: "-" if x is None else f"{100 * x:.0f}%"
    hdr = f"{'theme':22s} {'n':>4s} {'ledger':>7s} {'lo-first':>8s} {'proven':>7s} {'prov.ok':>7s} {'quest':>6s} {'pos.med':>8s} {'pos.p90':>8s} {'solver':>7s} {'sol.nodes':>10s} {'planner':>8s}"
    print(f"OBS {label}\nOBS " + hdr)
    for name, x in [("ALL", tot)] + sorted(themes.items(), key=lambda kv: -kv[1]["n"]):
        print(f"OBS {name:22s} {x['n']:4d} {fmt(x['ledger_correct']):>7s} {fmt(x['alt_correct']):>8s} {fmt(x['proven']):>7s} {fmt(x['proven_correct']):>7s} {x['questions']:6.1f} {x['nodes_med']:8d} {x['nodes_p90']:8d} "
              f"{fmt(x['solver_correct']):>7s} {x['solver_nodes']:10.0f} {fmt(x['planner_correct']):>8s}")
    print(f"OBS ledger: {tot['candidates']:.1f} candidates of {tot['legal']:.1f} legal moves per decision, {tot['facts']:.0f} facts, {tot['seconds']:.2f} s; "
          f"claims confirmed by the solver {tot['confirmed']} of {tot['claims']}; solver found a move at depth <= {CAP_DEPTH} in {fmt(tot['solver_found'])}")
    return dict(total=tot, themes=themes)


def classify_event(r: dict) -> str:
    """ANALYST_PLAN 40.5's four classes, from the loop's own records: which event was missing (no condition matched: the answer was never a candidate), which question
    was never opened (the candidate cap / the budget cut it), which answer was wrong (it was evaluated and ranked below), which shape has no lemma (phase 3: every chain is run from scratch)."""
    l = r["ledger"]
    if l.get("status") == "ERROR":
        return "ERROR"
    n = _mate_n(r["themes"])
    ev = {a[0] for a in l["all"]}
    gen = set(l.get("generated", []))
    if r["answer"] not in gen and not n:
        return "event missing: no condition made the answer a candidate"
    if r["answer"] not in ev and not n:
        return "question never opened: the candidate cap cut it"
    if n and n >= 2:
        return "needs a mate search beyond three moves (no word for it)"
    return "answer wrong: evaluated, ranked below the chosen move"


def classify(r: dict) -> str:
    l = r["ledger"]
    if l.get("status") == "ERROR":
        return "ERROR"
    gen = {a[0] for a in l["all"]}
    if r["answer"] not in gen and not _mate_n(r["themes"]):
        return "candidate never generated"
    if l["status"] == "ESTIMATED" and "budget" in l["reason"]:
        return "budget"
    if l["status"] == "ESTIMATED":
        return "wrong estimate (overlap left, midpoints)"
    if l.get("confirm") and l["confirm"]["result"] == "REFUTED":
        return "surprise: claim refuted by the solver (a bound or a read is wrong)"
    n = _mate_n(r["themes"])
    if n and n >= 2:
        return "needs a mate search (not a ledger question)"
    return "proven but not the puzzle's move (bounds)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", choices=("dev", "test", "fresh", "new17"), default="dev")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--misses", type=int, default=0, help="print the because-chain and a classification for this many misses")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--forcing", choices=("quiescence", "threats"), default="threats", help="what drives the forcing line: every check / capture / promotion, or the highest-value threat")
    ap.add_argument("--chains", type=int, default=0, help="print the because-chain (the line and the findings at its end) for this many decisions of different themes")
    ap.add_argument("--loop", choices=("event", "script"), default="event", help="the ledger's decision: read off the event loop's chain (ANALYST_PLAN 40.4, the default) or the older fixed script + B* stop (comparison only)")
    ap.add_argument("--lemmas", choices=("off", "learn", "on"), default="off", help="chain lemmas (ANALYST_PLAN 40.3): learn = record the shapes (no lemma used) and merge them into the store, on = promoted lemmas answer their questions")
    ap.add_argument("--promote", type=int, default=20, help="the promotion threshold: the count a shape must settle the same way (never below 5)")
    ap.add_argument("--stake-knob", type=int, default=3, help="a question with a stake above this is still run when a lemma matches")
    ap.add_argument("--lemma-file", default=None)
    ap.add_argument("--propose", action="store_true", help="one round of the proposer (ANALYST_PLAN 40.5): surprises on dev --n, proposals, each checked on the fresh sample of the same size; see proposer.py")
    ap.add_argument("--no-baselines", action="store_true", help="skip the solver-alone and planner columns (they are the same on both loops)")
    ap.add_argument("--budget", type=int, default=200, help="questions per decision (the plan's 200; other values only as a sensitivity check)")
    a = ap.parse_args(argv)
    if a.propose:
        from catspace.incubator.analyst.discovery import proposer
        return proposer.main(a.n, a.promote)
    global LEDGER_BUDGET
    LEDGER_BUDGET = a.budget
    os.environ["LEDGER_BUDGET"] = str(a.budget)
    os.environ["LEDGER_FORCING"] = a.forcing
    os.environ["LEDGER_LOOP"] = a.loop
    from catspace.incubator.analyst.discovery import chain_lemmas as CL
    lemma_file = a.lemma_file or CL.default_path()
    os.environ["LEDGER_LEMMAS"] = a.lemmas; os.environ["LEDGER_LEMMA_FILE"] = lemma_file; os.environ["LEDGER_STAKE_KNOB"] = str(a.stake_knob)
    if a.lemmas != "off":
        print(f"OBS lemmas {a.lemmas}: promotion threshold {max(CL.MIN_THRESHOLD, a.promote)} (flag --promote {a.promote}), stake knob {a.stake_knob}, store {lemma_file}; every lemma is a {CL.STATUS}")
    os.environ["LEDGER_NO_BASELINES"] = "1" if a.no_baselines else "0"
    t0 = time.perf_counter()
    rows = dev_rows(a.n) if a.set == "dev" else dev_rows(a.n, 11) if a.set == "fresh" else dev_rows(a.n, 17) if a.set == "new17" else test_rows(a.n)
    print(f"OBS {a.set}: {len(rows)} puzzles loaded in {time.perf_counter() - t0:.1f} s")
    ctx = multiprocessing.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(one, rows, chunksize=1)
    print(f"OBS {len(res)} puzzles decided in {time.perf_counter() - t0:.1f} s")
    for r in res:
        l = r["ledger"]
        if a.verbose or True:
            c = l.get("confirm")
            print(f"OBS {r['id']} {','.join(r['themes'][:3])}: ledger {l.get('move')} {'ok' if l['correct'] else 'MISS'} {l['status']} q={l['questions']} facts={l['facts']} "
                  f"cand={l['candidates']}/{l['legal']} {l['seconds']}s confirm={c['result'] if c else '-'} | solver {r['solver']['move']} d={r['solver']['depth']} "
                  f"nodes={r['solver']['nodes']} {'ok' if r['solver']['correct'] else 'MISS'} | planner {r['planner']['move']} {'ok' if r['planner']['correct'] else 'MISS'} (answer {r['answer']})")
    summ = summarize(res, f"{a.set} set, {len(res)} puzzles")
    if a.lemmas != "off":
        reps = [r["ledger"].get("lemma") for r in res if r["ledger"].get("lemma")]
        qs = sum(r["ledger"]["questions"] for r in res) / len(res)
        secs = sum(r["ledger"]["seconds"] for r in res) / len(res)
        print(f"OBS lemmas {a.lemmas}: {sum(r['ledger']['correct'] for r in res)} of {len(res)} correct; mean questions per decision {qs:.1f}; mean seconds per decision {secs:.2f}")
        if a.lemmas == "learn":
            print("OBS lemmas learn: " + json.dumps(CL.merge(reps, lemma_file, a.promote)))
        else:
            hits = sum(x["hits"] for x in reps); conf = sum(x["confirms"] for x in reps); con = sum(len(x["contradictions"]) for x in reps)
            print(f"OBS lemmas on: {hits} questions answered by a lemma, {conf} held-out checks confirmed, {con} contradictions (surprises, the lemma demoted); {sum(x['promoted_loaded'] for x in reps) // max(1, len(reps))} promoted lemmas loaded")
            if con:
                print("OBS lemmas on: store after the demotions: " + json.dumps(CL.merge(reps, lemma_file, a.promote)))
    cls = defaultdict(int)
    misses = [r for r in res if not r["ledger"]["correct"]]
    for r in misses:
        cls[(classify_event if os.environ.get("LEDGER_LOOP") == "event" else classify)(r)] += 1
    print(f"OBS misses {len(misses)}: " + "; ".join(f"{k}: {v}" for k, v in sorted(cls.items(), key=lambda kv: -kv[1])))
    for r in misses[:a.misses]:
        l = r["ledger"]
        print(f"OBS MISS {r['id']} {r['fen']} answer {r['answer']} ledger {l.get('move')} [{classify(r)}] themes {r['themes']}")
        print(f"OBS   interval {l.get('interval')} {l.get('reason')} why {l.get('why')}; table " + "; ".join(f"{t['move']} {t['interval']}{'*' if t['refined'] else ''}" for t in l.get("table", [])[:5]))
        ans = [x for x in l.get("all", []) if x[0] == r["answer"]]
        print(f"OBS   the answer as a candidate: {ans}")
        for f in l.get("chain", [])[:6]:
            print(f"OBS   [{f['status']}] {f['claim'][:180]} [{f['lo']},{f['hi']}]")
    seen_t = set()
    shown = 0
    for r in res:
        if shown >= a.chains: break
        if a.loop == "event" and not r["ledger"].get("lines"): continue
        th = [t for t in r["themes"] if t in ("hangingPiece", "pin", "fork", "discoveredAttack", "deflection", "sacrifice", "mateIn2", "skewer", "attraction", "intermezzo", "defensiveMove", "quietMove")]
        if not th or th[0] in seen_t or not (r["ledger"].get("chain") or r["ledger"].get("lines")): continue
        seen_t.add(th[0]); shown += 1
        l = r["ledger"]
        print(f"OBS CHAIN {r['id']} [{th[0]}] {r['fen']} answer {r['answer']} ledger {l['move']} {'ok' if l['correct'] else 'MISS'} {l['status']} positions {l.get('nodes')}")
        for f in l["chain"][:10]:
            print(f"OBS    [{f['status']}] {f['claim'][:200]}")
        for line in l.get("lines", []):
            print(f"OBS    {line[:260]}")
    p = paths.derived(f"ledger_puzzles_{a.set}_{len(res)}_{a.forcing}_{a.loop}" + (f"_b{a.budget}" if a.budget != 200 else "") + ".json")
    json.dump(dict(summary=summ, results=[{k: v for k, v in r.items() if k != "ledger"} | {"ledger": {k: v for k, v in r["ledger"].items() if k != "chain"}} for r in res]), open(p, "w"))
    print(f"OBS saved {p}")


if __name__ == "__main__":
    main()
