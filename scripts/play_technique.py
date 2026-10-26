"""PLAN 73.5 (5) THE PLAY-OUTS: the agenda as a PLAYER in one material class, against the table's slowest defence -- the 2026-10-23 K+R v K experiment's driver
(scratchpad krk/play_krk.py) generalised to a signature and copied into the repo.

  .venv/bin/python scripts/play_technique.py KRvK --seed 1 --floor 20 --table off

The winning side (White) is the AGENDA, moved by the play server's routine reproduced line for line (`AgendaPlayer.move` without the server: an Agenda per move
at the COUNT budget, `seconds=None`; the game's history for the CYCLE claim; the fact store carried from move to move; the one-ply fallback when no decision);
the lone side is the TABLE's slowest line (technique_corpus.defender_move). The start is technique_corpus.random_start (the same draw as play_krk.py: KRvK,
seed 1, floor 20 is 4K3/R7/8/8/2k5/8/8/8 w). `--table off` sets table.TABLE = False (plan 62.3's table claim off: the experiment); `on` leaves it as the
engine has it. The game ends by the game's rules (`is_game_over(claim_draw=True)`: mate, stalemate, repetition, the fifty-move rule).

Printed: one line per move of ours (the decision, its source, the questions asked, the inner agendas, the DTZ before and after) into the log, and the OBS line.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import chess

from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import table as TBMOD
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.play_server import AgendaPlayer, game_history
from catspace.incubator.analyst.discovery.state import state_of
from catspace.io import paths


def play(sig: str, seed: int, floor: "int | None", table: bool, out: str, log=print, technique: bool = True) -> dict:
    TBMOD.TABLE = table
    AG.TECHNIQUE = technique                                     # plan 73.5b: the technique question on (the store's mined entry of the class is asked) or off (the 73.5 baseline)
    tb = TC.tablebase()
    if floor is None:
        floor = TC.dtz_floor(sig)[0]
    board = TC.random_start(random.Random(seed), sig, floor)
    start, dtz0 = board.fen(), tb.probe_dtz(board)
    me = board.turn
    player = AgendaPlayer(fallback="one_ply")
    carry = None
    rows = []
    os.makedirs(out, exist_ok=True)
    tag = f"{sig}_seed{seed}_table{'on' if table else 'off'}_technique{'on' if technique else 'off'}"
    f = open(os.path.join(out, tag + ".log"), "w")
    print(f"START {start} dtz {dtz0} (floor {floor}, table {'on' if table else 'off'})", file=f, flush=True)
    t_all = time.monotonic()
    while not board.is_game_over(claim_draw=True):
        if board.turn == me:
            fen, us = board.fen(), "w" if board.turn else "b"
            d0 = tb.probe_dtz(board)
            t0 = time.monotonic()
            if carry is not None:
                carry.facts.advance(board)
            ag = Agenda(state_of(fen, us), seconds=None, carry=carry)
            ag.ctx.history = game_history(board)
            ag.ask()
            d = ag.decision()
            if carry is None:
                ag.ctx.facts.advance(board)
            carry = ag.ctx
            asked, inner = ag.asked + ag.ctx.asked, ag.ctx.agendas
            mv, src = None, "agenda"
            if d is not None and d.move:
                cand = chess.Move.from_uci(d.move)
                if cand in board.legal_moves:
                    mv = cand
            if mv is None:
                mv, src = player._fallback_move(board), "fallback:one_ply"
            san = board.san(mv)
            board.push(mv)
            d1 = tb.probe_dtz(board) if not board.is_game_over() else 0
            tech = next((ln for ln in ag.lines if ln.startswith("TECHNIQUE ")), None)
            row = dict(ply=len(board.move_stack), san=san, src=src, asked=asked, inner=inner, secs=round(time.monotonic() - t0, 1), dtz_before=d0, dtz_after=d1,
                       decision=(d.what, d.kind, d.value, d.proved, d.lemma) if d else None, technique=tech)
            rows.append(row)
            print(f"MOVE {row['ply']:3d} {san:6s} {src:16s} asked {asked:5d} inner {inner:4d} {row['secs']:6.1f} s  dtz {d0} -> {d1}  decision {row['decision']}", file=f, flush=True)
            if tech:
                print(f"         {tech}", file=f, flush=True)
                for ln in ag.lines[ag.lines.index(tech) + 1:ag.lines.index(tech) + 2]:
                    print(f"         {ln.strip()}", file=f, flush=True)
        else:
            mv = TC.defender_move(board)
            san = board.san(mv)
            board.push(mv)
            print(f"      ... {san}", file=f, flush=True)
    res = board.result(claim_draw=True)
    why = TC._result(board)
    n_ag = sum(1 for r in rows if r["src"] == "agenda")
    shorter = sum(1 for r in rows if r["dtz_after"] is not None and r["dtz_before"] is not None and abs(r["dtz_after"]) < abs(r["dtz_before"]))
    tot = sum(r["asked"] for r in rows)
    lem: dict = {}
    for r in rows:
        if r["decision"]:
            lem[r["decision"][4]] = lem.get(r["decision"][4], 0) + 1
    line = (f"OBS play {sig} seed {seed} table {'on' if table else 'off'} technique {'on' if technique else 'off'}: start {start}; dtz at start {dtz0}; result {res} ({why}) "
            f"after {len(board.move_stack)} plies; our moves {len(rows)}, by the agenda {n_ag}, by the fallback {len(rows) - n_ag}; questions asked {tot} "
            f"({tot / max(1, len(rows)):.0f} per move); inner agendas {sum(r['inner'] for r in rows)}; moves that shortened the table's distance {shorter} of {len(rows)}; "
            f"decision lemmas {', '.join(f'{k} {v}' for k, v in sorted(lem.items()))}; technique questions asked {sum(1 for r in rows if r['technique'])}; "
            f"wall {time.monotonic() - t_all:.0f} s")
    print(line, file=f, flush=True)
    f.close()
    log(line)
    return dict(result=res, why=why, plies=len(board.move_stack), moves=len(rows), asked=tot, rows=rows, start=start)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("signature")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--floor", type=int, default=None, help="the start's DTZ floor (default: the class's median, technique_corpus.dtz_floor)")
    ap.add_argument("--table", choices=("on", "off"), default="off")
    ap.add_argument("--technique", choices=("on", "off"), default="on", help="plan 73.5b: the technique question (the store's mined entry of the class) on, or off (the baseline)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    play(a.signature, a.seed, a.floor, a.table == "on", a.out or paths.experiment("technique_play"), technique=a.technique == "on")


if __name__ == "__main__":
    main()
