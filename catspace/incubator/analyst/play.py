"""play.py -- the analyst as a player's eyes. Each of OUR turns: analyse the position (idea,
threats, what no longer works), play the idea. Each of THEIR turns: ask an opponent engine
(Maia via lc0 at nodes=1) and let the analyst say what that move did. Writes a transcript + PGN.
    python -m catspace.incubator.analyst.play --maia assets/engines/maia/maia-1100.pb.gz \
        --color white --depth 14 --plies 120 --out tmp/analyst_vs_maia.txt"""
from __future__ import annotations

import argparse
import json
import time

import chess
import chess.engine
import chess.pgn

from catspace.io import paths
from .engine import Engine
from .ideas import analyse_position
from .verbalize import render, COLOR


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--maia", default=paths.engine("maia/maia-1100.pb.gz"))
    ap.add_argument("--color", default="white"); ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=4); ap.add_argument("--plies", type=int, default=120)
    ap.add_argument("--out", default="tmp/analyst_vs_maia.txt")
    ap.add_argument("--records", default="", help="append one JSON line per opponent move (for the miner)")
    ap.add_argument("--temperature", type=float, default=1.0, help="lc0 policy temperature (1.0 = Maia's human-move protocol)")
    ap.add_argument("--game-id", default="")
    a = ap.parse_args(argv)
    us = a.color == "white"
    t0 = time.perf_counter()
    import random
    from .maia import Maia
    maia = Maia(weights=a.maia)
    rng = random.Random(a.game_id or None)
    from .relations import relations
    FAMILY = {"pin": ("pin", "disabled_attack", "disabled_defense", "pinned_and_attacked", "hanging_through_pin"),
              "fork": ("fork",), "discovered": ("discovered_attack", "battery"), "skewer": ("skewer",),
              "hanging": ("hanging", "loose", "attacked_by_lower_value"), "overload": ("overloaded",)}
    def families(names):
        return sorted(f for f, ns in FAMILY.items() if any(n in ns for n in names))
    recs_out = open(a.records, "a") if a.records else None
    lines = [f"analyst ({a.color}, Stockfish depth {a.depth}, multipv {a.multipv}) vs Maia-1100 (lc0 nodes=1)", ""]
    board = chess.Board(); boards, sans = [board.copy()], []
    with Engine(depth=a.depth) as eng:
        while not board.is_game_over(claim_draw=True) and board.ply() < a.plies:
            prev = boards[-3] if len(boards) >= 3 else None
            kb = sans[-1] if sans else None
            if board.turn == us:
                rec = analyse_position(board, eng, prev=prev, killed_by=kb, multipv=a.multipv)
                move = chess.Move.from_uci(rec.idea.uci)
                lines.append(render(rec, alternatives=False)); lines.append(f"  >> we play {board.san(move)}"); lines.append("")
            else:
                move = maia.sample(board, a.temperature, rng)
                rec = analyse_position(board, eng, prev=prev, killed_by=kb, played=move, multipv=a.multipv)
                tag = f" ({rec.verdict})" if rec.verdict else ""
                if recs_out:
                    r = rec.played.refutation if rec.played else None
                    mech = []
                    if r and r.effects:
                        e0 = r.effects[0]
                        mech = families([x.name for x in e0.created] + [x.name for x in e0.uses] + [x.name for x in e0.resolved])
                        if e0.captured: mech.append("capture")
                        if r.san.endswith("#") or (r.mate is not None): mech.append("mate")
                    present = families([x.name for x in relations(board) if (x.owner == us) or (x.name in FAMILY["hanging"] and x.owner != us)])
                    recs_out.write(json.dumps({"game": a.game_id, "ply": board.ply(), "maia": "black" if us else "white",
                        "san": board.san(move), "verdict": rec.verdict, "drop": (rec.eval_cp - rec.played.cp) if rec.played else 0,
                        "eval_before": rec.eval_cp, "phase_nonpawn": rec.facts.phase_nonpawn,
                        "mechanisms": sorted(set(mech)), "present": present,
                        "ignored_threat": bool(r and any(t.line.move == r.uci for t in rec.threats)),
                        "threat_existed": bool(rec.threats)}) + "\n")
                lines.append(f"[{board.fullmove_number}{'.' if board.turn else '...'}] Maia plays {board.san(move)}{tag}")
                if rec.verdict in ("blunder", "mistake", "inaccuracy"):
                    lines.append(render(rec, alternatives=False).split("\n", 1)[1].split("\n")[0])   # the verdict line only
                lines.append("")
            sans.append(board.san(move)); board.push(move); boards.append(board.copy())
        res = board.result(claim_draw=True)
        lines.append(f"RESULT {res} after {board.ply()} plies | {time.perf_counter()-t0:.0f}s | engine calls {eng.calls}, cache hits {eng.hits}")
    maia.quit()
    if recs_out:
        recs_out.close()
    game = chess.pgn.Game.from_board(board); game.headers["White"] = "analyst" if us else "Maia-1100"; game.headers["Black"] = "Maia-1100" if us else "analyst"; game.headers["Result"] = res
    lines.append(""); lines.append(str(game))
    open(a.out, "w").write("\n".join(lines))
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
