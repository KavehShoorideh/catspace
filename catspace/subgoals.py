"""catspace/subgoals.py -- M3 subgoal API: (s, context) -> ranked subgoal regions.

score(region) = P_reach(g | s, z_self, z_opp, elos)  x  net_flux(g | bands)  x  quality(g | band)
    P_reach : the v2 two-z first-hit field (retrieval-factored -- goal tower precomputed ONCE).
    net_flux: empirical SF-refereed crossing rate of the region at the OPPONENT's band minus at
              OURS (their error zone, not ours -- THESIS §5). Region x band table built offline
              from the M2a labeled set (experiments/m3_build_region_table.py); the learned
              T(s, ctx) upgrade replaces the table lookup in M3.1 (recorded follow-up).
    quality : mean SF committor (mover POV, our band) of the region -- how winnable it is for us.

Components are returned unreduced so the planner (optionality portfolio) can re-weigh; `score`
is the default product with net_flux floored at 0 (a region where WE cross more than they do is
not a subgoal). Latency is measured by experiments/m3_subgoal_gates.py and recorded in JOURNAL.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SubgoalRanker:
    def __init__(self, field_ckpt: str, reach_npz: str, region_table: str,
                 flux_t: str = "", device: str = "cpu"):
        from experiments.train_reach_head import ReachHead
        self.dev = torch.device(device)
        ck = torch.load(field_ckpt, map_location=self.dev, weights_only=False)
        d_opp = 17 if any(k.startswith("state.0") and v.shape[1] > 84
                          for k, v in ck["state_dict"].items()) else 0
        self.model = ReachHead(d_phi=64, d_z=16, d_opp=d_opp).to(self.dev)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()
        t = np.load(region_table, allow_pickle=True)
        if "regions" in t.files:
            self.bank = torch.as_tensor(t["regions"], dtype=torch.float32, device=self.dev)
        else:
            z = np.load(reach_npz, allow_pickle=True)
            self.bank = torch.as_tensor(z["bank"], dtype=torch.float32, device=self.dev)
        self.flux = t["crossing_rate"]          # (G, B) SF-refereed crossing rate per mover band
        self.quality = t["committor_mean"]      # (G, B) mover-POV committor
        self.counts = t["count"]                # (G, B)
        self.band_edges = t["band_edges"]       # e.g. [1500]
        self.pcond = t["pcond"] if "pcond" in t.files else None   # (G, CB, B) composite factor
        self.n_cband = self.pcond.shape[1] if self.pcond is not None else 1
        # augmented-codebook assignment (fired eyes-gate): regions_assign lives in
        # [zs(phi) ⊕ w·zs(human_feats)] space; queries assign there, tower stays on phi
        self.assign_bank = t["regions_assign"] if "regions_assign" in t.files else None
        if self.assign_bank is not None:
            self.aug = {k: t[k] for k in ("aug_mu_p", "aug_sd_p", "aug_mu_f", "aug_sd_f", "aug_w")}

        with torch.no_grad():                   # goal tower ONCE (retrieval-factored)
            self.gh, self.gt = self.model.goal_embs(self.bank)
        self.T = None
        if flux_t:                              # learned continuous flux scorer (M3 design path)
            from experiments.train_transition_estimator import T as TNet
            tc = torch.load(flux_t, map_location=self.dev, weights_only=False)
            self.T = TNet(tc["d_phi"], tc["d_ctx"]).to(self.dev)
            self.T.load_state_dict(tc["state_dict"]); self.T.eval()
            cb = torch.as_tensor(tc["cb_mean"], device=self.dev)
            self._t_pos = torch.stack([1 - (2 * cb - 1).abs(), cb,
                                       torch.as_tensor(tc["ply_mean"], device=self.dev) / 100.0], 1)

    @torch.no_grad()
    def t_flux(self, elo_mover: float, elo_opp: float):
        """expected mover committor-loss per region at these ratings (T at centroids)."""
        G = len(self.bank)
        rb = torch.tensor([[(elo_mover - 1500) / 400, (elo_opp - 1500) / 400]],
                          dtype=torch.float32, device=self.dev).expand(G, -1)
        return self.T(self.bank, torch.cat([self._t_pos, rb], 1)).cpu().numpy()

    @torch.no_grad()
    def p_composite(self, phis, elo_self: float, elo_oppo: float, z_self=None, z_opp=None,
                    n_obs: int = 0):
        """Batched composite reach probabilities for the PLANNER: phis (B,64) -> (B, G*CB).
        Same context for all rows (one decision point, many successor boards)."""
        B = len(phis)
        f = torch.as_tensor(np.asarray(phis, np.float32), device=self.dev)
        zs = (torch.zeros(B, 16, device=self.dev) if z_self is None else
              torch.as_tensor(np.asarray(z_self, np.float32), device=self.dev).view(1, -1).expand(B, -1))
        ctx = [torch.tensor([[(elo_self - 1500) / 400, (elo_oppo - 1500) / 400, 1.0, 1.0]],
                            dtype=torch.float32, device=self.dev).expand(B, -1)]
        if self.model.state[0].in_features > 84:
            zo = (torch.zeros(B, 16, device=self.dev) if z_opp is None else
                  torch.as_tensor(np.asarray(z_opp, np.float32), device=self.dev).view(1, -1).expand(B, -1))
            nn_ = torch.full((B, 1), float(np.log1p(n_obs) / np.log1p(64.0)),
                             dtype=torch.float32, device=self.dev)
            ctx.append(torch.cat([zo, nn_], 1))
        sh, _ = self.model.state_embs(f, zs, torch.cat(ctx, 1))
        pr = torch.sigmoid(sh @ self.gh.T * self.model.scale + self.model.b_hit).cpu().numpy()
        if self.pcond is not None:
            pr = (pr[:, :, None] * self.pcond[None, :, :, self.band(elo_self)]).reshape(B, -1)
        return pr

    def band(self, elo: float) -> int:
        return int(np.searchsorted(self.band_edges, elo, side="right"))

    def assign(self, phis, feats=None):
        """(B,64) phi [+ (B,4) production-recipe human feats] -> region ids. Augmented tables
        assign in [zs(phi) ⊕ w·zs(feats)] space; plain tables in phi space."""
        phis = np.asarray(phis, np.float32)
        if self.assign_bank is None:
            bank = self.bank.cpu().numpy()
            d2 = (phis*phis).sum(1)[:, None] + (bank*bank).sum(1)[None, :] - 2.0*phis@bank.T
            return d2.argmin(1)
        assert feats is not None, "augmented table needs human feats for assignment"
        A = np.concatenate([(phis - self.aug["aug_mu_p"]) / self.aug["aug_sd_p"],
                            float(self.aug["aug_w"]) * (np.asarray(feats, np.float32)
                                                        - self.aug["aug_mu_f"])
                            / self.aug["aug_sd_f"]], 1).astype(np.float32)
        ab = self.assign_bank
        d2 = (A*A).sum(1)[:, None] + (ab*ab).sum(1)[None, :] - 2.0*A@ab.T
        return d2.argmin(1)

    @torch.no_grad()
    def rank(self, phi_s, elo_self: float, elo_oppo: float, z_self=None, z_opp=None,
             n_obs: int = 0, top: int = 8):
        """phi_s: (64,) frozen-trunk embedding of s. Returns dict of per-region components +
        the ranked top-`top` region ids by the default product score."""
        f = torch.as_tensor(np.asarray(phi_s, np.float32), device=self.dev).view(1, -1)
        zs = torch.zeros(1, 16, device=self.dev) if z_self is None else \
            torch.as_tensor(np.asarray(z_self, np.float32), device=self.dev).view(1, -1)
        ctx = [torch.tensor([[(elo_self - 1500) / 400, (elo_oppo - 1500) / 400, 1.0, 1.0]],
                            dtype=torch.float32, device=self.dev)]
        if self.model.state[0].in_features > 84:            # two-z field
            zo = torch.zeros(1, 16, device=self.dev) if z_opp is None else \
                torch.as_tensor(np.asarray(z_opp, np.float32), device=self.dev).view(1, -1)
            nn_ = torch.tensor([[float(np.log1p(n_obs) / np.log1p(64.0))]],
                               dtype=torch.float32, device=self.dev)
            ctx.append(torch.cat([zo, nn_], 1))
        sh, st = self.model.state_embs(f, zs, torch.cat(ctx, 1))
        p_reach = torch.sigmoid(sh @ self.gh.T * self.model.scale + self.model.b_hit)[0]
        plies = torch.expm1((st @ self.gt.T * self.model.scale + self.model.b_time).clamp(0, 8))[0]
        b_opp, b_self = self.band(elo_oppo), self.band(elo_self)
        pr = p_reach.cpu().numpy()
        if self.pcond is not None:
            # FACTORIZED COMPOSITE (2026-07-29 decision): P(reach phi-cell AND committor-band)
            # ~= P(reach cell) * P(cband | cell) -- independence approximation, validated by the
            # even->odd 2.96x/2.95x enrichment; field-v3 (composite goals) removes it (queued).
            pr = (pr[:, None] * self.pcond[:, :, b_self]).reshape(-1)    # (G*CB,)
        if self.T is not None:                  # continuous: their risk at g minus ours
            net_flux = self.t_flux(elo_oppo, elo_self) - self.t_flux(elo_self, elo_oppo)
        else:
            net_flux = self.flux[:, b_opp] - self.flux[:, b_self]
        quality = self.quality[:, b_self]
        score = pr * np.maximum(net_flux, 0.0) * quality
        # AVOID list (Kaveh 2026-07-29): regions we are LIKELY to pass through where the error
        # asymmetry runs AGAINST us -- the steer-away half. Same tables, roles swapped.
        score_avoid = pr * np.maximum(-net_flux, 0.0)
        order = np.argsort(-score)[:top]        # composite ids: region*CB + cband
        return {"score": score, "p_reach": pr, "net_flux": net_flux,
                "quality": quality, "exp_plies": plies.cpu().numpy(), "top": order,
                "score_avoid": score_avoid, "avoid": np.argsort(-score_avoid)[:top]}
