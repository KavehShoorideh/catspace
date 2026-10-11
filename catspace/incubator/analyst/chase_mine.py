"""THE CHASE MINING (Kaveh 2026-10-11: "if you chased and never got anything in return, then maybe you don't need to
chase -- can you prove that the chase is bad? ... mine it"). Every one-target proof question the planner asks on
census positions (rust/analyst_core/src/waiting.rs prove_t, lemma.rs MINE), with the one-move threat lemma's stopper
verdict (lemma.rs threat_lemma: target / noAttack / exchange / escape / check / pin / promotion / defence /
intermezzo / none), whether the search held anyway, at which tier, its nodes and its proving line. Printed: per
stopper, questions, held, nodes spent on the failures; the held cases where the static exchange said nothing is won
(the "exchange" stopper) are saved with SAN lines for reading -- the exceptions of a no-chase lemma.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from catspace.io import paths


def _san_line(fen: str, line: list) -> str:
    import chess
    b = chess.Board(fen)
    out = []
    for u in line:
        if u == "pass":
            out.append("(pass)")
            b.push(chess.Move.null())
            continue
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            out.append(u)
            break
        out.append(b.san(m))
        b.push(m)
    return " ".join(out)


def _game(args):
    gid, fens = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    analyst_core.set_mine(True)
    analyst_core.chase_why()
    rows = []
    rows_why = []
    for ply, fen in enumerate(fens):
        analyst_core.mine_cases()
        analyst_core.plan_trace(fen)
        why = analyst_core.chase_why()
        rows_why.append(why)
        for c in analyst_core.mine_cases():
            rows.append(dict(game=gid, ply=ply, fen=c[0], uci=c[1], target=c[2], goal=c[3], stopper=c[4], holds=c[5],
                             tier=c[6], nodes=c[7], line=c[8], chase=c[9]))
    w = [sum(x[i] for x in rows_why) for i in range(10)] if rows_why else [0] * 10
    if rows:
        rows[0]["why"] = w
    return rows


def _onoff(args):
    gid, fens = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    out = []
    for fen in fens:
        analyst_core.set_chase(False); r0 = analyst_core.plan_trace(fen)
        analyst_core.set_chase(True); r1 = analyst_core.plan_trace(fen)
        analyst_core.set_chase(False)
        out.append(dict(fen=fen, d0=r0[1], d1=r1[1], i0=r0[5], i1=r1[5], ms0=r0[4], ms1=r1[4], n0=r0[3], n1=r1[3]))
    return out


def onoff(tag: str, games: int, skip: int, workers: int) -> None:
    """The planner with the no-chase lemma on and off on the same positions: decisions, immediate wins, time."""
    from multiprocessing import Pool
    from .waiting_frames import _games
    t0 = time.time()
    with Pool(workers) as p:
        rows = [r for rs in p.imap_unordered(_onoff, _games(tag, skip + games)[skip:], chunksize=1) for r in rs]
    n = len(rows)
    same = sum(1 for r in rows if r["d0"] == r["d1"] and r["i0"] == r["i1"])
    print(f"OBS ON/OFF games {skip + 1}-{skip + games}, {n} positions ({time.time() - t0:.0f} s): decisions and immediate wins identical {same} "
          f"({100 * same / n:.2f}%); ms a position off {sum(r['ms0'] for r in rows) / n:.1f} -> on {sum(r['ms1'] for r in rows) / n:.1f}; "
          f"nodes off {sum(r['n0'] for r in rows) / n:,.0f} -> on {sum(r['n1'] for r in rows) / n:,.0f}")
    for r in [r for r in rows if not (r["d0"] == r["d1"] and r["i0"] == r["i1"])][:12]:
        print(f"OBS   off {r['i0'][3] if r['i0'] else None} / {r['d0'][3] if r['d0'] else None}; on {r['i1'][3] if r['i1'] else None} / {r['d1'][3] if r['d1'] else None}  {r['fen']}")


def run(tag: str, games: int, workers: int, skip: int = 0) -> None:
    from multiprocessing import Pool
    from .waiting_frames import _games
    t0 = time.time()
    with Pool(workers) as p:
        rows = [r for rs in p.imap_unordered(_game, _games(tag, skip + games)[skip:], chunksize=1) for r in rs]
    w = [sum(r.get("why", [0] * 10)[i] for r in rows) for i in range(10)]
    print("OBS   NO-CHASE stopped at: " + ", ".join(f"{n} {v}" for n, v in zip(["m forcing/pawn", "banked>=goal", "king/no target", "no quiet escape", "(a) collectable", "(b) our check", "(c) renewed attack", "(c) second threat", "DECIDED", "(forcing searches unknown, per harbour)"], w)))
    print(f"OBS {games} games, {len(rows)} one-target questions ({time.time() - t0:.0f} s on {workers} processes)")
    by = defaultdict(list)
    for r in rows:
        by[r["stopper"]].append(r)
    tot_fail_nodes = sum(r["nodes"] for r in rows if not r["holds"])
    for k, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        held = [r for r in rs if r["holds"]]
        fail_nodes = sum(r["nodes"] for r in rs if not r["holds"])
        tiers = defaultdict(int)
        for r in held:
            tiers[r["tier"]] += 1
        print(f"OBS   {k:11s} questions {len(rs):6d}, held {len(held):5d} ({100 * len(held) / len(rs):5.2f}%; tier {dict(tiers)}), "
              f"nodes on failures {fail_nodes:,} ({100 * fail_nodes / max(tot_fail_nodes, 1):.0f}% of all failed nodes)")
    # THE NO-CHASE LEMMA's verdicts against the search: level 1 skips the 2-ply tier, level 2 both tiers
    for lvl, name in ((2, "harbour (skip both)"),):
        rs = [r for r in rows if r["chase"] == lvl]
        wrong = [r for r in rs if r["holds"] and (lvl == 2 or r["tier"] == 1)]
        later = [r for r in rs if r["holds"] and lvl == 1 and r["tier"] == 2]
        saved = sum(r["nodes"] for r in rs if not r["holds"])
        print(f"OBS   NO-CHASE level {lvl} ({name}): {len(rs)} questions, would wrongly skip {len(wrong)} held "
              f"({100 * len(wrong) / max(len(rs), 1):.2f}%)" + (f", held later at 4 plies {len(later)}" if lvl == 1 else "") +
              f"; nodes on its failures {saved:,} ({100 * saved / max(tot_fail_nodes, 1):.0f}% of all failed nodes)")
        for r in wrong:
            r["san"] = _san_line(r["fen"], r["line"])
        with open(paths.experiment(f"chase_mine_wrong_{lvl}.json"), "w") as fh:
            json.dump(wrong, fh, indent=1)
    cases = [r for r in by.get("exchange", []) if r["holds"]]
    for r in cases:
        r["san"] = _san_line(r["fen"], r["line"])
    out = paths.experiment("chase_mine_exchange_held.json")
    with open(out, "w") as fh:
        json.dump(cases, fh, indent=1)
    print(f"OBS   {len(cases)} held cases with the exchange stopper -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip", type=int, default=0, help="start after this many games (held-out runs)")
    ap.add_argument("--onoff", action="store_true", help="the planner with the lemma on vs off")
    a = ap.parse_args(argv)
    if a.onoff:
        onoff(a.tag, a.games, a.skip, a.workers)
    else:
        run(a.tag, a.games, a.workers, a.skip)


if __name__ == "__main__":
    main()
