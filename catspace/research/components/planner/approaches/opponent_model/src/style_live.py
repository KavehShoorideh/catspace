"""catspace/style/live.py -- LIVE opponent-state wiring: observed moves -> causal ẑ_opp(t).

Wraps the M2c OpponentEstimator with the play-time featurization it needs (Maia-2 candidates in
the mover frame + frozen-field phi), matching the TRAINING conditioning exactly:
  * features = the m2b_cache recipe (top-16 ∪ played, mover-frame indices, strict vocab mask;
    observations with an unscorable played move are skipped -- the v2/v3 sweep guard);
  * ẑ recomputed on the DOUBLING schedule (1,2,4,8,16,32,64) and held between -- the exact
    conditioning the v3 field was trained under (train == play, no skew);
  * cold start = z=0 (population at Elo), n_obs=0 -- the field's verified fallback.

Per-game use (M4): fresh instance per game. Cross-game warm start (M6): persist/load the
posterior via PlanStore.opponents -- deliberately NOT automatic here.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from catspace.research.components.planner.approaches.opponent_model.src.style_estimator import OpponentEstimator
from catspace.research.components.planner.approaches.opponent_model.src.style_model import VOCAB, StyleResidual
from catspace.io import paths

RECOMPUTE_AT = (1, 2, 4, 8, 16, 32, 64)
K = 17


class LiveOpponent:
    def __init__(self, rf, m2, m2_inf, prepared, style_ckpt: str, train_cache: str,
                 elo_known: float, device: str = "cpu"):
        self.rf = rf; self.m2 = m2; self.inf = m2_inf
        self.all_moves, self.elo_dict, _ = prepared
        self.mirror = m2_inf.mirror_move if hasattr(m2_inf, "mirror_move") else None
        if self.mirror is None:
            from maia2.inference import mirror_move
            self.mirror = mirror_move
        ck = torch.load(style_ckpt, map_location=device, weights_only=False)
        self.model = StyleResidual(n_individual=ck["n_individual"], d_z=ck["d_z"],
                                   lam_prior=ck["lam"], learn_mu=ck.get("learn_mu", False)).to(device)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()
        z_train = self.model.delta.weight.detach()
        tc = np.load(train_cache, allow_pickle=True)
        tp, te = tc["pidx"], tc["elo_self"]
        train_elo = np.zeros(ck["n_individual"], np.float32)
        for p in range(ck["n_individual"]):
            train_elo[p] = te[tp == p].mean()
        self.est = OpponentEstimator(self.model, z_train, train_elo, elo_known=float(elo_known),
                                     device=device)
        self.device = device
        self.z = np.zeros(ck["d_z"], np.float32)
        self.n_obs = 0
        self._next_rc = 0
        self._elo = float(elo_known)

    def observe_move(self, lcboard_before, uci_played: str, mover_white: bool,
                     opp_elo_frame: float):
        """Feed one observed opponent move (board BEFORE their move). Updates z on the
        doubling schedule. Skips unscorable moves (vocab overflow) -- counted, not fatal."""
        import pandas as pd
        fen = lcboard_before.fen()
        df = pd.DataFrame({"fen": [fen], "move": ["0000"],
                           "elo_self": [int(self._elo)], "elo_oppo": [int(opp_elo_frame)]})
        try:
            df, _ = self.inf.inference_batch(df, self.m2, verbose=False, batch_size=8,
                                             num_workers=0)
        except Exception:
            return False                                     # maia2 poison position: skip obs
        probs = df["move_probs"][0]
        top = [m for m, _ in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:K - 1]]
        cand = top + ([uci_played] if uci_played not in top else [])
        cand = cand[:K]
        if uci_played not in cand:
            cand[-1] = uci_played
        idx = np.full(K, VOCAB, np.int64); logp = np.full(K, -30.0, np.float32)
        for c_i, m in enumerate(cand):
            key = m if mover_white else self.mirror(m)
            idx[c_i] = self.all_moves.get(key, VOCAB)
            logp[c_i] = math.log(max(probs.get(m, 1e-6), 1e-6))
        overflow = idx >= VOCAB
        played_slot = cand.index(uci_played)
        if overflow[played_slot]:
            return False                                     # unscorable played move: skip
        phi = self.rf.phi([lcboard_before]).cpu().float()
        feats = {"phi": phi,
                 "cand_idx": torch.from_numpy(np.where(overflow, VOCAB, idx))[None, :],
                 "cand_logp": torch.from_numpy(logp)[None, :],
                 "cand_mask": torch.from_numpy(~overflow)[None, :],
                 "rank": (torch.arange(K).float() / (K - 1))[None, :],
                 "played_slot": torch.tensor([played_slot]),
                 "elo": torch.tensor([self._elo])}
        self.est.observe(feats)
        n = self.est.n_observed
        if n > self.n_obs:
            self.n_obs = n
            if self._next_rc < len(RECOMPUTE_AT) and n >= RECOMPUTE_AT[self._next_rc]:
                with torch.no_grad():
                    self.z = self.est.z().cpu().numpy().astype(np.float32)
                while (self._next_rc < len(RECOMPUTE_AT)
                       and n >= RECOMPUTE_AT[self._next_rc]):
                    self._next_rc += 1
        return True


def _tests():
    """CPU-only self-test with the real frozen pieces (no games, no MPS)."""
    import chess
    from lczerolens import LczeroBoard
    from maia2 import model as maia_model, inference
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    prepared = inference.prepare()
    m2 = maia_model.from_pretrained(type="rapid", device="cpu")
    rf = ReachabilityField(device="cpu")
    lo = LiveOpponent(rf, m2, inference, prepared,
                      paths.experiment("m2b_style_3k.pt"),
                      paths.derived("m2b/cache_3k.npz"), elo_known=1100, device="cpu")
    assert lo.n_obs == 0 and np.allclose(lo.z, 0), "cold start = population prior"
    b = LczeroBoard()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f6e4"]
    n_ok = 0
    for i, u in enumerate(moves):
        if i % 2 == 1:                                       # observe BLACK as the opponent
            n_ok += lo.observe_move(b, u, mover_white=False, opp_elo_frame=1800)
        b.push(chess.Move.from_uci(u))
    assert lo.n_obs == n_ok == 4, f"4 observations expected, got {lo.n_obs}/{n_ok}"
    assert np.linalg.norm(lo.z) > 0, "z must move off the prior after recomputes"
    print(f"LIVE-OPPONENT TESTS PASSED: n_obs={lo.n_obs} |z|={np.linalg.norm(lo.z):.4f} "
          f"(cold->informed transition on the doubling schedule)")


if __name__ == "__main__":
    _tests()
