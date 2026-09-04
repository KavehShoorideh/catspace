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
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
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
            for r in recs:
                print(render(r, long=args.long, alternatives=not args.quiet))
        n = len(recs); dt = time.perf_counter() - t0
        print(f"[analyst] {n} positions in {dt:.1f}s ({dt/max(n,1):.1f}s/pos) | engine calls {eng.calls}, cache hits {eng.hits}", file=sys.stderr)


if __name__ == "__main__":
    main()
