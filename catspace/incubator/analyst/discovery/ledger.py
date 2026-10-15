"""THE REASONING LEDGER, from Python (Kaveh 2026-10-14; docs/ANALYST_PLAN.md section 39; the engine is rust/analyst_core/src/reasoning.rs).

  python -m catspace.incubator.analyst.discovery.ledger --fen "<fen>" [--line "Qc7 Rbd1 Rc6 h5 exd4"]

Prints the because-chain (the dependency path from the decision down to the facts), the counts by status, the questions asked against the
budget, the conjectures in the back pocket (CONDITIONAL findings and their premises), and with --line the roll forward: per ply how many findings
were retracted, carried over and re-asked, and what became of the pin on the e-pawn. `--mirror` mirrors the board first (the plan's wording; the
ledger reads either side, so it is not needed).
"""
from __future__ import annotations

import argparse
import json
import time

import chess

KAVEH = "5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R b - - 0 1"


def _a():
    import catspace.incubator.analyst.discovery  # noqa: F401  (installs the chess terms: job, pinned ...)
    import analyst_core as A
    return A


def analyse(fen: str, budget: int = 200) -> dict:
    return _a().ledger(fen, budget)


def new(fen: str, budget: int = 200):
    return _a().Ledger(fen, budget)


def by_id(report: dict) -> dict:
    return {f["id"]: f for f in report["findings"]}


def chain(report: dict, root: int, depth: int = 0, seen: set | None = None, quiet_zero: bool = True) -> list:
    """The because-chain from a finding down: its claim, its status and interval, then what it cites (a finding cited twice is shown once; a
    job that is worth nothing is not a reason, only counted)."""
    fs = by_id(report)
    seen = seen if seen is not None else set()
    f = fs[root]
    pad = "  " * depth
    out = [f"{pad}[{f['status']}] {f['claim']}  value [{f['lo']}, {f['hi']}]  ({f['type']}#{f['id']})"]
    if root in seen:
        out[0] += "  (above)"
        return out
    seen.add(root)
    zero = 0
    for c in f["cites"]:
        g = fs.get(c)
        if g is None:
            continue
        if quiet_zero and g["type"] in ("JOB", "RECAP") and g["lo"] == 0:
            zero += 1
            continue
        out += chain(report, c, depth + 1, seen, quiet_zero)
    if zero:
        out.append(f"{pad}  ({zero} other guard jobs asked, worth nothing)")
    return out


def pins(report: dict, square: str | None = None) -> list:
    return [f for f in report["findings"] if f["type"] == "PIN" and (square is None or f["key"].split("|")[2].endswith(square))]


def show(report: dict, root: int | None = None) -> None:
    t = report["questions_asked"]
    print(f"OBS ledger: {t} questions asked of a budget of {report['budget']} ({len(report['skipped'])} skipped), {report['facts_evaluated']} facts evaluated, "
          f"{report['solver_calls']} solver call(s)")
    print(f"OBS ledger: live findings by status {report['counts']}")
    top = sorted(pins(report), key=lambda f: -f["lo"])
    if root is None and top:
        root = top[0]["id"]
    if root is not None:
        print("OBS the because-chain:")
        for line in chain(report, root):
            print("OBS   " + line)
    for c in report["confirmations"]:
        print(f"OBS solver on finding #{c['finding']}: " + ("CONFIRMS" if c["confirmed"] else "does NOT confirm") +
              f" -- their gain of >= {c['value']} stays forced after " + "; ".join(f"{m['move']} ({m['status']}: {m['line']})" for m in c["moves"]))
    print(f"OBS the agenda (by stake), first ten: {report['agenda'][:10]}")
    for f in report["back_pocket"]:
        print(f"OBS back pocket: {f['claim']}  [premise: {f['premise']}]  value {f['value']}")


def roll(fen: str, san_line: list, budget: int = 200, square: str = "e5", verbose: bool = True) -> list:
    """Roll the ledger through a line (SAN, from `fen`), return the per-ply records; prints the pin on `square` at each ply."""
    L = new(fen, budget)
    rep = L.run()
    b = chess.Board(fen)
    recs = [dict(ply=0, move=None, retracted=0, lookups=0, re_asked=rep["questions_asked"], pins=[dict(claim=f["claim"], lo=f["lo"], hi=f["hi"]) for f in pins(rep, square)])]
    if verbose:
        print(f"OBS ply 0: {rep['questions_asked']} questions asked; pin on {square}: " + "; ".join(f"{p['claim']} [{p['lo']}, {p['hi']}]" for p in recs[0]["pins"]))
    for san in san_line:
        m = b.parse_san(san)
        mover = chess.COLOR_NAMES[b.turn]
        b.push(m)
        r = L.push(m.uci())
        rep = L.report_dict()
        ps = [f for f in pins(rep, square)]
        r["pin_on_square"] = [dict(claim=f["claim"], lo=f["lo"], hi=f["hi"], status=f["status"]) for f in ps]
        recs.append(r)
        if verbose:
            before = sum(1 for f in rep["findings"] if f["ply"] < r["ply"])
            print(f"OBS ply {r['ply']} {mover} {san}: retracted {r['retracted']} (touched {','.join(r['touched'])}), carried over {before} findings "
                  f"(looked up {r['lookups']} times), re-asked {r['re_asked']}, premise listeners woken {r['premise_listeners_woken']}")
            for p, f in zip(r["pin_on_square"], ps):
                print(f"OBS    pin on {square}: [{p['status']}] {p['claim']}  [{p['lo']}, {p['hi']}]")
                for mv in json.loads(f["note"].split(" | ")[0] or "[]"):
                    print(f"OBS      {mv['move']}: gains {mv['gain']}, check {mv['check']}, costs {mv['cost']}")
            if not r["pin_on_square"]:
                print(f"OBS    no pin finding on {square} now")
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fen", default=KAVEH)
    ap.add_argument("--line", default=None, help='SAN moves from the position, e.g. "Qc7 Rbd1 Rc6 h5 exd4"')
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--square", default="e5", help="the piece whose pin is followed along the line")
    ap.add_argument("--trace", action="store_true", help="print the reasoning on this position step by step, in plain chess words")
    ap.add_argument("--depth", type=int, default=4, help="plies of the line (trace)")
    ap.add_argument("--mirror", action="store_true", help="mirror the board first (the ledger reads either side; not needed)")
    a = ap.parse_args(argv)
    fen = chess.Board(a.fen).mirror().fen() if a.mirror else a.fen
    if a.trace:
        t0 = time.time()
        r = new(fen, max(a.budget, 1000)).trace(a.depth)
        for line in r["trace"]:
            print(line)
        print(f"OBS trace: {r['positions']} positions examined, {r['questions']} questions, {r['facts']} facts, {round(time.time() - t0, 2)} s; line "
              + " ".join(x.replace('|', ' [') + ']' for x in r["line"]))
        return
    t0 = time.time()
    rep = analyse(fen, a.budget)
    print(f"OBS ledger on {fen}: {round(time.time() - t0, 2)} s")
    show(rep)
    if a.line:
        t0 = time.time()
        roll(fen, a.line.split(), a.budget, a.square)
        print(f"OBS roll forward: {round(time.time() - t0, 2)} s")


if __name__ == "__main__":
    main()
