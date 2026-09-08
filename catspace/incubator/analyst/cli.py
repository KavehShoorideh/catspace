"""analyst CLI (S0). Local Stockfish, no network, no LLM.
    python -m catspace.incubator.analyst.cli analyze --pgn game.pgn [--depth 16 --multipv 5]
    python -m catspace.incubator.analyst.cli analyze --fen <fen> [--moves "d4 Be6"]   (the moves define
        the position; the fen before them is the 'previous' position for killed ideas)
    python -m catspace.incubator.analyst.cli threats --fen <fen>"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass

import chess

from .engine import Engine
from .ideas import analyse_game, analyse_position
from .verbalize import render, effect_phrase
from .dynamics import threats, zugzwang


def _json(o):
    if is_dataclass(o):
        return asdict(o)
    raise TypeError


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--pgn"); a.add_argument("--fen"); a.add_argument("--moves", default="")
    a.add_argument("--depth", type=int, default=16); a.add_argument("--multipv", type=int, default=5)
    a.add_argument("--json", action="store_true"); a.add_argument("--quiet", action="store_true")
    a.add_argument("--fresh-cache", action="store_true", help="ignore the on-disk cache (timing runs)")
    a.add_argument("--long", action="store_true", help="every notable fact per ply (the S0 form)")
    t = sub.add_parser("threats"); t.add_argument("--fen", required=True); t.add_argument("--depth", type=int, default=16)
    h = sub.add_parser("horizon"); h.add_argument("--fen", required=True); h.add_argument("--depth", type=int, default=16); h.add_argument("--k", type=int, default=6)
    h.add_argument("--maia", default="1100,1500,1900", help="rating ladder whose top moves join the candidates ('' to skip)")
    h.add_argument("--include", default="", help="SAN moves to force into the candidate set, e.g. the move played")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    if args.cmd == "horizon":
        from .horizon import trajectories, find_traps
        from .dynamics import line_effects
        from .verbalize import best_fact, cp_str
        b = chess.Board(args.fen); w = b.turn
        with Engine(depth=8) as shallow:
            cands = [chess.Move.from_uci(l.move) for l in shallow.analyse(b, multipv=args.k, depth=8)]
            path = shallow.path
        ladder = {}                                   # uci -> {rating: prob}
        if args.maia:
            from .maia import Maia
            for r in [int(x) for x in args.maia.split(",") if x]:
                mm = Maia(r)
                for u, pr in mm.top(b, 2):
                    ladder.setdefault(u, {})[r] = pr
                    m = chess.Move.from_uci(u)
                    if m not in cands:
                        cands.append(m)
                mm.quit()
        for san in args.include.split():
            m = b.parse_san(san)
            if m not in cands:
                cands.append(m)
        trajs = trajectories(b, path, cands, depth=args.depth)
        ratings = sorted({r for d in ladder.values() for r in d})
        head = "        " + " ".join(f"{'d'+str(d):>6s}" for d in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20) if d <= args.depth) + "   " + " ".join(f"M{r}" for r in ratings)
        print(head)
        for t in trajs:
            row = " ".join(f"{cp_str(t.scores[d], w):>6s}" if d in t.scores else "      " for d in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20) if d <= args.depth)
            lad = " ".join(f"{ladder.get(t.move, {}).get(r, 0)*100:4.0f}%" for r in ratings)
            print(f"{t.san:7s} {row}   {lad}")
        for tr in find_traps(trajs, args.depth):
            eff = line_effects(b.copy(), [tr.traj.move] + tr.refutation[1:], 4)
            why = "; ".join(f"{e.san}: {best_fact(e)}" for e in eff[1:3] if best_fact(e))
            tb = b.copy(); line = []
            for u in ([tr.traj.move] + tr.refutation[1:])[:6]:
                m = chess.Move.from_uci(u); line.append(tb.san(m)); tb.push(m)
            print(f"TRAP {tr.traj.san}: looks {cp_str(tr.peak_cp, w)} at depth {tr.peak_depth}, collapses at depth {tr.collapse_depth} to "
                  f"{cp_str(tr.final_cp, w)} (best {tr.best_san} {cp_str(tr.best_final_cp, w)}). Refutation: {' '.join(line)}" + (f" -- {why}" if why else ""))
        print(f"[horizon] {len(trajs)} candidates in {time.perf_counter()-t0:.1f}s", file=sys.stderr)
        return
    import tempfile, os
    cache = os.path.join(tempfile.mkdtemp(), "c.sqlite") if getattr(args, "fresh_cache", False) else None
    with Engine(depth=args.depth, cache_path=cache) as eng:
        if args.cmd == "threats":
            b = chess.Board(args.fen)
            for th in threats(b, eng):
                print(f"{'REAL ' if th.real else '     '}{th.gain_cp:+5d}  {th.san_line(b)}  {effect_phrase(th.first_effects) if th.first_effects else ''}")
            z = zugzwang(b, eng)
            if z: print(f"zugzwang: {z.flagged} (best {z.best_cp:+d}, if-pass {z.pass_cp:+d})")
            return
        if args.pgn:
            recs, game = analyse_game(open(args.pgn).read(), eng, multipv=args.multipv)
            hdr = game.headers
            print(f"{hdr.get('White','?')} vs {hdr.get('Black','?')} ({hdr.get('Date','?')})")
        else:
            prev = chess.Board(args.fen); b = prev.copy(); sans = []
            for s in args.moves.split():
                sans.append(s); b.push_san(s)
            recs = [analyse_position(b, eng, prev=prev if len(sans) == 2 else None, killed_by=sans[-1] if sans else None, multipv=args.multipv)]
        if args.json:
            print(json.dumps([asdict(r) for r in recs], default=_json, indent=1))
        else:
            standing = []
            for r in recs:
                print(render(r, long=args.long, alternatives=not args.quiet))
                if args.long and r.played:                                    # the standing threat: what the move threatens, and how it met the last one
                    from .threats import threats_after, judge_reply
                    b0 = chess.Board(r.fen); mv = chess.Move.from_uci(r.played.uci)
                    try:
                        if standing:
                            for l in judge_reply(standing, b0, mv):
                                print(f"  Against the standing threat: {l}")
                        standing = threats_after(b0, mv, None, eng, budget=120)
                        for t in standing:
                            print(f"  {r.played.san} threatens {t.text}: they must " + ", ".join(list(t.defences)[:5]) + (" ..." if len(t.defences) > 5 else ""))
                    except Exception:
                        standing = []
        n = len(recs); dt = time.perf_counter() - t0
        print(f"[analyst] {n} positions in {dt:.1f}s ({dt/max(n,1):.1f}s/pos) | engine calls {eng.calls}, cache hits {eng.hits}", file=sys.stderr)


if __name__ == "__main__":
    main()
