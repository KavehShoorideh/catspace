#!/usr/bin/env python
"""experiments/uci_engine.py -- UCI shell around the catspace planner engine (Kaveh:
'use the existing frameworks where chess bots compete'). Speaks enough UCI for
cutechess-cli / fastchess / python-chess: uci, isready, ucinewgame, position, go
(movetime / wtime+winc / nodes), quit.

Time management: budget = movetime, else remaining/28 + 0.7*inc. The search runs in
64-eval chunks with tree reuse until the budget is spent -- a faster field buys MORE
NODES at the same clock, which is how fixed-TC play converts speed into strength
(the strength-per-node frontier, operationalized). Mates harvested into the shared
banks as always (banks are facts).

Engine options (setoption):  Field <ckpt>   -- override the field checkpoint
                             Nodes <n>      -- hard node cap per move (0 = clock only)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr      # engine-module diagnostics must NOT pollute the UCI stream


def say(s):
    _REAL_STDOUT.write(s + "\n")
    _REAL_STDOUT.flush()


def main():
    import chess
    board = chess.Board()
    opts = {"Field": "", "Nodes": 0}
    E = {}

    def ensure():
        if E:
            return
        import numpy as np
        from catspace.engine.fields import FieldModel
        from catspace.nn.mcts import MCTS
        from experiments.bootstrap_mate_engine import (OnlineMateBank, harvest,
                                                       make_batched_energy_prior,
                                                       make_boot_value, make_planner)
        field = opts["Field"]
        if not field:
            ptr = Path("data/derived/sep/self_field_current.txt")
            field = ptr.read_text().strip() if ptr.exists() \
                else "data/derived/sep/lichess_mc2.pt"
        fm = FieldModel(field, device="mps")
        pfx = "artifacts/experiments/assistant"        # shared fact banks
        bank = OnlineMateBank(fm, Path(pfx + "_bank.fens"))
        loss = OnlineMateBank(fm, Path(pfx + "_lossbank.fens"))
        draw = OnlineMateBank(fm, Path(pfx + "_drawbank.fens"))
        for bk in (bank, loss, draw):
            bk.sync()
        ctx = {"plan": "direct", "hist": {}}
        times = {}
        vfn = make_boot_value(fm, bank, times, loss, draw_bank=draw, game_ctx=ctx)
        pfn, pfnb = make_batched_energy_prior(
            (Path("data/derived/sep/opponent_energy_current.txt").read_text().strip()
             if Path("data/derived/sep/opponent_energy_current.txt").exists()
             else "data/derived/sep/opponent_energy_v1.pt"), game_ctx=ctx)
        E.update(np=np, MCTS=MCTS, harvest=harvest, fm=fm, bank=bank, loss=loss,
                 draw=draw, ctx=ctx, vfn=vfn, pfn=pfn, pfnb=pfnb,
                 planner=make_planner(fm, bank))

    def reset_game():
        from collections import Counter
        E["ctx"]["hist"] = Counter({board.epd(): 1})
        E["ctx"]["plan"] = "direct"; E["ctx"]["target_pt"] = None

    def go(budget_ms, node_cap):
        np, MCTS = E["np"], E["MCTS"]
        if hasattr(E["vfn"], "set_anchor"):
            E["vfn"].set_anchor(board)
        ps = E["planner"](board, len(board.move_stack))
        E["ctx"]["plan"] = ps["plan"]; E["ctx"]["target_pt"] = ps.get("target_pt")
        m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=64, mate_stop=True,
                 pw_c=1.5, root_min_visits=10, value_fn=E["vfn"], policy_fn=E["pfn"],
                 policy_batch_fn=E["pfnb"], batch_leaves=32)
        t0 = time.time()
        root, used = None, 0
        while (time.time() - t0) * 1000 < budget_ms and (not node_cap or used < node_cap):
            root = m.run(board.copy(stack=True), reuse_root=root)
            used += int(m.evals_used)
            if m.evals_used == 0:              # certified mate in hand
                break
        if root is None or not root.children:
            mv = next(iter(board.legal_moves))
        else:
            best = max(root.children,
                       key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None
                                            else c.Q)))
            mv = best.move
            w, l, s = E["harvest"](root)
            E["bank"].add(w); E["loss"].add(l); E["draw"].add(s)
        return mv, used

    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "uci":
            say("id name catspace-planner")
            say("id author kaveh+claude")
            say("option name Field type string default ")
            say("option name Nodes type spin default 0 min 0 max 100000")
            say("uciok")
        elif cmd == "setoption" and len(parts) >= 5 and parts[1] == "name":
            name = parts[2]
            val = " ".join(parts[parts.index("value") + 1:]) if "value" in parts else ""
            if name in opts:
                opts[name] = int(val) if name == "Nodes" else val
        elif cmd == "isready":
            ensure()
            say("readyok")
        elif cmd == "ucinewgame":
            ensure()
            board.reset()
            reset_game()
        elif cmd == "position":
            ensure()
            if parts[1] == "startpos":
                board.reset()
                mvs = parts[3:] if len(parts) > 2 and parts[2] == "moves" else []
            else:
                fi = parts.index("fen") + 1
                mi = parts.index("moves") if "moves" in parts else len(parts)
                board.set_fen(" ".join(parts[fi:mi]))
                mvs = parts[mi + 1:] if mi < len(parts) else []
            reset_game()
            for u in mvs:
                board.push_uci(u)
                E["ctx"]["hist"][board.epd()] += 1
        elif cmd == "go":
            ensure()
            kv = dict(zip(parts[1::2], parts[2::2]))
            if "movetime" in kv:
                budget = int(kv["movetime"])
            elif "wtime" in kv or "btime" in kv:
                rem = int(kv.get("wtime" if board.turn else "btime", 60000))
                inc = int(kv.get("winc" if board.turn else "binc", 0))
                budget = max(200, rem / 28 + 0.7 * inc)
            else:
                budget = 5000
            cap = int(kv.get("nodes", opts["Nodes"] or 0))
            mv, used = go(budget, cap)
            say(f"info nodes {used}")
            say(f"bestmove {mv.uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
