#!/usr/bin/env python
"""experiments/basin_mate_engine.py -- MATING BY SIMULATION (Kaveh 2026-07-23: "that's the
one I want to be used to get the mate"). At each position: roll a batch of value-net-guided
playouts (both sides, epsilon-noised, rules-adjudicated); the WINNING rollouts define the
basin online; play the move the winners favor (success- and shortness-weighted).
Tablebase-free, hand-code-free: the basin emerges from the simulations' outcomes.

Batched lockstep rollouts: one net call per ply across all simulations.
"""
from __future__ import annotations
import sys
from pathlib import Path
import chess
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes


class BasinRolloutEngine:
    def __init__(self, value_ckpt, rollouts=48, max_len=44, eps=0.15, device="cpu", seed=0):
        import torch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
        from experiments.train_dtm_cnn import DTMNet
        self.torch = torch
        self.dev = pick_device(device)
        st = torch.load(value_ckpt, map_location="cpu", weights_only=False)
        self.net = DTMNet(c=st["c"]).to(self.dev); self.net.load_state_dict(st["state"]); self.net.eval()
        self.scale = st.get("scale", 20.0)
        self.R, self.max_len, self.eps = rollouts, max_len, eps
        self.rng = np.random.default_rng(seed)

    def _score_children(self, boards_children):
        """boards_children: flat list of child boards -> net 'plies-to-mate' preds (lower=better for White)."""
        pk = np.stack([encode_packed(b) for b in boards_children])
        mt = np.stack([encode_meta(b) for b in boards_children])
        with self.torch.no_grad():
            return (self.net(self.torch.from_numpy(feature_planes(pk, mt)).to(self.dev))
                    .cpu().numpy() * self.scale)

    def move(self, board: chess.Board):
        root_moves = list(board.legal_moves)
        if len(root_moves) == 1:
            return root_moves[0], {}
        # immediate mate short-circuit (rules, not a concept)
        for m in root_moves:
            c = board.copy(stack=False); c.push(m)
            if c.is_checkmate():
                return m, {"mate1": True}
        sims = []
        for i in range(self.R):
            m0 = root_moves[i % len(root_moves)]        # every root move gets rollouts
            b = board.copy(stack=False); b.push(m0)
            sims.append({"first": m0, "b": b, "done": b.is_game_over(claim_draw=True),
                         "won": b.is_checkmate(), "len": 1})
        for ply in range(self.max_len):
            live = [s for s in sims if not s["done"]]
            if not live:
                break
            # gather all children of all live sims -> ONE net call
            packs, owners = [], []
            for s in live:
                ms = list(s["b"].legal_moves)
                s["_ms"] = ms
                for m in ms:
                    c = s["b"].copy(stack=False); c.push(m)
                    packs.append(c); owners.append(s)
            if not packs:
                break
            preds = self._score_children(packs)
            idx = 0
            for s in live:
                ms = s["_ms"]; n = len(ms)
                pr = preds[idx: idx + n]; idx += n
                white_to_move = s["b"].turn == chess.WHITE
                # White minimizes predicted plies-to-mate; Black maximizes (both from the SAME net)
                order = np.argsort(pr if white_to_move else -pr)
                pick = order[0] if self.rng.random() > self.eps else order[int(self.rng.integers(n))]
                s["b"].push(ms[int(pick)])
                s["len"] += 1
                if s["b"].is_game_over(claim_draw=True):
                    s["done"] = True
                    s["won"] = s["b"].is_checkmate() and s["b"].turn == chess.BLACK
        # the basin vote: success- and shortness-weighted first moves
        score = {}
        for s in sims:
            w = (1.0 / s["len"]) if s["won"] else 0.0
            score[s["first"]] = score.get(s["first"], 0.0) + w
        best = max(root_moves, key=lambda m: score.get(m, 0.0))
        n_won = sum(s["won"] for s in sims)
        if score.get(best, 0.0) == 0.0:
            # no rollout mated: fall back to the net's 1-ply argmax (value guidance)
            children = []
            for m in root_moves:
                c = board.copy(stack=False); c.push(m); children.append(c)
            preds = self._score_children(children)
            best = root_moves[int(np.argmin(preds))]
        return best, {"rollouts_won": n_won, "R": self.R}
