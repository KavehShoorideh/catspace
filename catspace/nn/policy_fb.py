"""
nn/policy_fb.py — FBBoardPolicy: greedy readout of the TorchFB cone on real
boards. depth=1 scores own successors by F(s')@z; depth=2 applies the MIN
over opponent replies (README lesson 3: the readout's opponent model matters
as much as the field's) with every grandchild encoded in ONE batched forward.

Terminal ordering (README lesson 5, terminal scoring is load-bearing):
  I deliver mate  +1e9   >   any reach value   >   draw  -1e9   >   I get mated  -2e9
Draws score BAD because the policy is hunting its mate goal -- the toy
convention (draw at the 0.1% reach quantile) carried over.

Works for either color: pass the z that matches the side this policy plays
(zMATE_W when playing white, zMATE_B when playing black).
"""
from __future__ import annotations

import chess
import numpy as np
import torch

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids

MATE_SCORE = 1e9
DRAW_SCORE = -1e9
MATED_SCORE = -2e9


class FBBoardPolicy:
    def __init__(self, fb, z, depth: int = 2, elo: int = 1800, clock: float = 300.0,
                 device: str = "cpu"):
        assert depth in (1, 2)
        self.fb = fb.to(device).eval()
        self.z = torch.as_tensor(z, dtype=torch.float32, device=device)
        self.depth = depth
        self.device = device
        self._omega_row = omega_ids(np.array([elo]), np.array([elo]), np.array([clock]))[0]

    @torch.no_grad()
    def _reach(self, boards: list[chess.Board]) -> np.ndarray:
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        planes = torch.from_numpy(feature_planes(packed, meta)).to(self.device)
        om = torch.from_numpy(np.tile(self._omega_row, (len(boards), 1))).to(self.device)
        return (self.fb.embed_F(planes, om) @ self.z).cpu().numpy()

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        moves = list(board.legal_moves)
        scores = np.full(len(moves), -np.inf)
        pending: list[tuple[int, chess.Board]] = []      # depth-1 leaves
        pending2: list[tuple[int, chess.Board]] = []     # depth-2 leaves (i = my move idx)

        for i, m in enumerate(moves):
            child = board.copy(stack=False)
            child.push(m)
            if child.is_checkmate():
                return m                                   # my mate: nothing beats it
            if child.is_game_over(claim_draw=True):
                scores[i] = DRAW_SCORE
                continue
            if self.depth == 1:
                pending.append((i, child))
                continue
            worst = np.inf
            replies = []
            for r in child.legal_moves:
                grand = child.copy(stack=False)
                grand.push(r)
                if grand.is_checkmate():
                    worst = MATED_SCORE                    # opponent mates me
                    break
                if grand.is_game_over(claim_draw=True):
                    worst = min(worst, DRAW_SCORE)
                    continue
                replies.append(grand)
            if worst <= MATED_SCORE:
                scores[i] = MATED_SCORE
            elif not replies:
                scores[i] = worst if np.isfinite(worst) else DRAW_SCORE
            else:
                pending2.extend((i, g) for g in replies)
                scores[i] = worst                          # min over terminal replies so far

        if pending:
            reach = self._reach([b for _, b in pending])
            for (i, _), v in zip(pending, reach):
                scores[i] = v
        if pending2:
            reach = self._reach([b for _, b in pending2])
            for (i, _), v in zip(pending2, reach):
                scores[i] = min(scores[i], v) if np.isfinite(scores[i]) else v

        best = int(np.argmax(scores))
        return moves[best]
