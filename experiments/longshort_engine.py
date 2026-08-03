#!/usr/bin/env python
"""experiments/longshort_engine.py -- the long/short planner engine (Kaveh 2026-07-21 UI spec): the FIELD does
long-range planning by picking a SUBGOAL (a reachable B-cluster on the way to mate); a UNIFORM-PRIOR MCTS
searches locally to reach it; when we get within --handoff-pieces of the goal OR the planner STALLS (no longer
getting closer to mate), a plain mate-seeking MCTS takes over. Exposes move(board) -> {move, phase, subgoal,
plan}, where plan is the MCTS principal variation (the hops), so the UI can draw both.

Design honesty (JOURNAL 2026-07-21): the field is a collapsed ~1-D distance-to-mate predictor that navigates
the coarse phase efficiently (reach_efficiency: 97% vs pure 77% @400n) but fails as a fine move-prior -- so
this uses UNIFORM priors (which beat the field prior) with the field only as the coarse VALUE / subgoal
selector, and hands off to search near the goal. That is exactly the coarse-navigator / fine-executor split
the experiments support.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS
from experiments.value_fixed_point import TB, tb_best_move


class LongShortEngine:
    def __init__(self, field, bank_pk, bank_mt, dev, nodes=400, n_basins=40, lam=1.0,
                 replan=3, handoff_pieces=5, stall_patience=3):
        self.fb, self.dev, self.lam = load_ckpt(Path(field), dev)[0], dev, lam
        self.fb.eval()
        self.zW = torch.load(field, map_location="cpu", weights_only=False)["zgoals"]["MATE_W"]
        self.zW = (self.zW.detach().float() if torch.is_tensor(self.zW) else torch.tensor(np.asarray(self.zW, np.float32))).to(dev)[None, :]
        self.om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
        self.nodes, self.replan, self.handoff_pieces, self.stall_patience = nodes, replan, handoff_pieces, stall_patience
        with torch.no_grad():
            Bb = self._embB(bank_pk, bank_mt); Fb = self._embF(bank_pk, bank_mt)
            dmate = self.fb.distance_matrix(Fb, self.zW)[:, 0].cpu().numpy()   # each waypoint's own dist-to-mate
        lab = KMeans(n_clusters=n_basins, n_init=4, random_state=0).fit_predict(Bb.cpu().numpy())
        self.basins = [(Bb[torch.from_numpy(np.flatnonzero(lab == c)).to(dev)],
                        float(np.quantile(dmate[lab == c], 0.25))) for c in range(n_basins)
                       if (lab == c).any()]
        self.basin_dmate = torch.tensor([b[1] for b in self.basins], device=dev)
        # per-game state (call reset() before a new game)
        self.reset()

    def reset(self):
        self.subgoal = None; self.since = 10 ** 9; self._dmate_hist = []; self._stall = 0

    def _embF(self, pk, mt):
        with torch.no_grad():
            return self.fb.embed_F(torch.from_numpy(feature_planes(pk, mt)).to(self.dev),
                                   torch.from_numpy(np.tile(self.om, (len(pk), 1))).to(self.dev))

    def _embB(self, pk, mt):
        with torch.no_grad():
            return self.fb.embed_B(torch.from_numpy(feature_planes(pk, mt)).to(self.dev))

    def _F1(self, board):
        return self._embF(encode_packed(board)[None], encode_meta(board)[None])

    def _dmate(self, board):                                    # coarse distance-to-mate (the stall signal)
        with torch.no_grad():
            return float(self.fb.distance_matrix(self._F1(board), self.zW)[0, 0])

    def _select_subgoal(self, board):
        with torch.no_grad():
            f = self._F1(board)
            reach = torch.stack([self.fb.distance_matrix(f, Bm)[0].min() for Bm, _ in self.basins])
            c = int((reach + self.lam * self.basin_dmate).argmin())      # reachable AND leads-to-mate
        self.subgoal = self.basins[c][0]; self.since = 0

    def _mcts(self, board, target, mate_stop):
        """uniform-prior MCTS whose leaf value is -d(F, target). target=None => target is the mate region zW."""
        tgt = self.zW if target is None else target

        def value_fn(boards):
            with torch.no_grad():
                F = self._embF(np.stack([encode_packed(b) for b in boards]), np.stack([encode_meta(b) for b in boards]))
                d = self.fb.distance_matrix(F, tgt).min(1).values.cpu().numpy()
            return np.tanh((6.0 - d) / 6.0) * (1 if board.turn == chess.WHITE else 1)   # white-POV-ish squashed reach

        def policy_fn(b):
            lm = list(b.legal_moves); return {m: 1.0 / len(lm) for m in lm}              # UNIFORM prior

        m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=self.nodes, mate_stop=mate_stop,
                 pw_c=1.5, root_min_visits=10, policy_fn=policy_fn, value_fn=value_fn)
        return m.run(board), m

    def move(self, board, rng=None):
        """Return {uci, phase, subgoal_dmate, plan:[san...], dmate}."""
        npieces = len(board.piece_map())
        dm = self._dmate(board); self._dmate_hist.append(dm)
        # stall = coarse distance-to-mate not improving over the last stall_patience White moves
        if len(self._dmate_hist) > self.stall_patience and dm >= min(self._dmate_hist[-self.stall_patience - 1:-1]) - 1e-3:
            self._stall += 1
        else:
            self._stall = 0
        near = npieces <= self.handoff_pieces
        stalled = self._stall >= self.stall_patience
        phase = "execute" if (near or stalled) else "plan"
        if phase == "plan":
            if self.subgoal is None or self.since >= self.replan:
                self._select_subgoal(board)          # <-- RL SEAM: a learned PlanSelector replaces this heuristic
            self.since += 1
            # spec: field plans a SUBGOAL, uniform-MCTS navigates to it. (Heuristic subgoal selection is weak on
            # the collapsed field -- JOURNAL; the eventual RL planner is what makes this strong. Wired as-is.)
            root, _ = self._mcts(board, self.subgoal, mate_stop=True)
        else:
            # EXECUTE: PURE MCTS + mate_stop -- the field value HURTS near mate (pure search beats it,
            # reach_efficiency/field-vs-pure), so the local finisher uses NO field, just search.
            m = MCTS(lambda bs: np.zeros(len(bs), dtype=float), max_nodes=self.nodes, mate_stop=True,
                     pw_c=1.5, root_min_visits=10)
            root = m.run(board)
        white = board.turn == chess.WHITE
        best = max(root.children, key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q) * (1 if white else -1)))
        # plan = the PV hops from the search
        plan, node = [], best
        b = board.copy(stack=False)
        for _ in range(6):
            plan.append(b.san(node.move)); b.push(node.move)
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.N)
        return dict(uci=best.move.uci(), phase=phase, plan=plan, dmate=round(dm, 2),
                    subgoal_dmate=(round(float(self.basin_dmate[int((torch.stack([self.fb.distance_matrix(self._F1(board), Bm)[0].min() for Bm, _ in self.basins]) + self.lam * self.basin_dmate).argmin())]), 2) if phase == "plan" else None))


def _smoke():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="data/derived/sep/nucleus_distilled.pt")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--nodes", type=int, default=400)
    args = ap.parse_args()
    dev = pick_device("cpu"); rng = np.random.default_rng(0); tb = TB("data/syzygy")
    dz = np.load("data/derived/dtm_endgame.npz"); idx = rng.permutation(len(dz["packed"]))[:1500]
    eng = LongShortEngine(args.field, dz["packed"][idx], dz["meta"][idx], dev, nodes=args.nodes)
    import json
    starts = [chess.Board(f) for f in json.loads(Path("artifacts/experiments/krrkbp_test_n200.json").read_text())["fens"][:args.n]]
    mated = 0
    for s in starts:
        eng.reset(); b = s.copy(stack=False); phases = []
        for _ in range(80):
            if b.is_game_over(claim_draw=True):
                break
            if b.turn == chess.WHITE:
                mv = eng.move(b); phases.append(mv["phase"][0]); b.push(chess.Move.from_uci(mv["uci"]))
            else:
                b.push(tb_best_move(b, tb))
        out = b.outcome(claim_draw=True); win = out and out.winner == chess.WHITE; mated += int(bool(win))
        print(f"  start: mated={bool(win)}  phase-trace={''.join(phases)[:40]}")
    tb.close()
    print(f"VERDICT LONGSHORT_ENGINE_SMOKE field={Path(args.field).stem} n={len(starts)} mate_rate={mated/len(starts):.2f}")


if __name__ == "__main__":
    t0 = time.time(); _smoke(); print(f"[done] {time.time()-t0:.0f}s")
