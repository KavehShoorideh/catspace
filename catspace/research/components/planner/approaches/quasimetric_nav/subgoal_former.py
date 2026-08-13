#!/usr/bin/env python
"""subgoal_former.py -- the M4 planner core (Kaveh 2026-08-12, docs/SUBGOALFORMER.md).

Three cooperating pieces, segregated by training regime (gradient boundaries are the design):

  GeoQuery        frozen field + jqt sidecar -> live geometry for any token set: distances to
                  anchors from BOTH points of view (null-move twin gives the opponent-to-move
                  state), pairwise anchor relations, activation probabilities. NO gradients
                  flow into the field from here (the consumer never trains its instrument).
  SubgoalFormer   GeoAttention layers over subgoal tokens + a one-FC revised-P head.
                  certificate() emits the legible plan artifact: committed subgoal, revised
                  p-hat, and the COUNTERFACTUAL worry/opportunity vector (mask token b,
                  recompute p-hat: exact attribution; attention is the consistency check,
                  never the attribution itself). Trained SUPERVISED on reach-events only --
                  RL must never shape p-hat (a policy-trained certificate is propaganda).
  alert_set()     certificate DIFF across a move -> worries AND opportunities in one object
                  (sign is the only difference; a removed blocker = armed tactics M7 for
                  free). Top-K salience tokens are the RL observation; the RL acts by
                  POINTING at one (pursue/deny/hold) + a search budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from catspace.research.components.planner.approaches.quasimetric_nav.geo_attention import (
    GeoAttention, N_GEO)


class GeoQuery:
    """live geometry for (head, code) subgoal tokens against a position. Everything detached."""

    def __init__(self, eng, jqt_path, leverage_path=None, device="mps"):
        from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
            JQTModule)
        self.eng = eng                              # KittyChess: field access (net, dA/dB, poles)
        pay = torch.load(jqt_path, map_location=device, weights_only=False)
        self.jqt = JQTModule(d_model=pay["d_in"], heads=pay["heads"], codes=pay["codes"],
                             d=pay["d"], square_codes=pay.get("square_codes", 0),
                             piece_codes=pay.get("piece_codes", 0)).to(device)
        self.jqt.load_state_dict(pay["state_dict"], strict=False)
        self.jqt.eval()
        self.H, self.C = pay["heads"], pay["codes"]
        self.device = device
        self.lev = None
        if leverage_path is not None:
            z = np.load(leverage_path)
            self.lev = np.zeros((self.H, self.C), np.float32)
            for sw, hh, cc in zip(z["swing"], z["head"], z["code"]):
                self.lev[int(hh), int(cc)] = float(sw)

    @torch.no_grad()
    def state_embed(self, board):
        """(z_us, z_opp): arm-B embeddings of the position and its null-move twin (the
        opponent-to-move state Kaveh asked to query from). Twin falls back to z_us when the
        null-move is illegal (mover in check)."""
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
        tk, gl = tokenize(board)
        b2 = board.copy(stack=False)
        b2.turn = not b2.turn
        b2.ep_square = None
        rows_t, rows_g = [np.asarray(tk)], [np.asarray(gl)]
        if b2.is_valid():
            tk2, gl2 = tokenize(b2)
            rows_t.append(np.asarray(tk2)); rows_g.append(np.asarray(gl2))
        z = self.eng._embed(rows_t, rows_g).float()
        return z[0], (z[1] if len(z) > 1 else z[0])

    @torch.no_grad()
    def geometry(self, board, hc):
        """hc (T,2) int64 [head, code] -> (G (T,T,N_GEO), feats (T,F_TOK)) all torch, detached.
        feats = [dA(s->g), dA(s_opp->g), P(act g), leverage, log-anchor-norm] per token."""
        hc = torch.as_tensor(hc, dtype=torch.long, device=self.device)
        T = len(hc)
        A = self.jqt.anchors_for(hc).float()                       # (T, d) z_B-space anchors
        z_us, z_opp = self.state_embed(board)
        net = self.eng.net
        dA_s = net.dA(z_us.expand(T, -1), A).float()
        dA_o = net.dA(z_opp.expand(T, -1), A).float()
        dB_s = net.dB(z_us.expand(T, -1), A).float()
        p_act = torch.sigmoid(self.jqt.activation_logit(dB_s)).float()
        ii = torch.arange(T, device=self.device).repeat_interleave(T)
        jj = torch.arange(T, device=self.device).repeat(T)
        pA = torch.log1p(net.dA(A[ii], A[jj]).clamp(min=0)).view(T, T).float()
        pB = torch.log1p(net.dB(A[ii], A[jj]).clamp(min=0)).view(T, T).float()
        race = (torch.log1p(dA_s.clamp(min=0))[:, None]
                - torch.log1p(dA_o.clamp(min=0))[None, :])         # our g vs their g'
        G = torch.stack([pA, pA.T, pB, pB.T,
                         torch.log1p(dA_s.clamp(min=0))[None, :].expand(T, T),
                         p_act[None, :].expand(T, T), race], -1)
        lev = torch.zeros(T, device=self.device)
        if self.lev is not None:
            lev = torch.as_tensor(self.lev[hc[:, 0].cpu(), hc[:, 1].cpu()],
                                  device=self.device)
        feats = torch.stack([torch.log1p(dA_s.clamp(min=0)),
                             torch.log1p(dA_o.clamp(min=0)),
                             p_act, lev, A.norm(dim=-1).log1p()], -1)
        return G.cpu(), feats.cpu()

    @torch.no_grad()
    def candidates_live(self, board, k=12, k_lev=4):
        """POSITION-CONDITIONED candidates (2026-08-13): whole vocabulary ranked by live
        P(activate) from THIS position + a few leverage staples. One batched pass."""
        allhc = torch.tensor([(h, c) for h in range(self.H) for c in range(self.C)],
                             dtype=torch.long, device=self.device)
        A = self.jqt.anchors_for(allhc).float()
        z_us, _ = self.state_embed(board)
        dB = self.eng.net.dB(z_us.expand(len(A), -1), A)
        p = torch.sigmoid(self.jqt.activation_logit(dB))
        top = p.argsort(descending=True)[:k].cpu().numpy()
        hcs = [(int(i // self.C), int(i % self.C)) for i in top]
        if self.lev is not None:
            fl = np.argsort(-np.abs(self.lev).ravel())[:k_lev]
            hcs += [(int(i // self.C), int(i % self.C)) for i in fl]
        seen, out = set(), []
        for h, c in hcs:
            if (h, c) not in seen:
                seen.add((h, c)); out.append((h, c))
        return np.array(out, np.int64)

    @torch.no_grad()
    def move_toward(self, board, goal, minimize=False):
        """PREMOVE-tier move choice: NO search -- one batched readout of every child's
        P(activate goal); argmax (pursue) or argmin (deny). The b=0 action."""
        import chess as _ch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            tokenize as _tk)
        moves = list(board.legal_moves)
        if not moves:
            return None, 0
        rows_t, rows_g = [], []
        for mv in moves:
            board.push(mv)
            tk, gl = _tk(board)
            rows_t.append(np.asarray(tk)); rows_g.append(np.asarray(gl))
            board.pop()
        z = self.eng._embed(rows_t, rows_g).float()
        A = self.jqt.anchors_for(torch.tensor([goal], dtype=torch.long,
                                              device=self.device)).float()
        dB = self.eng.net.dB(z, A.expand(len(z), -1))
        pg = torch.sigmoid(self.jqt.activation_logit(dB))
        # disaster veto: among the top half by mover E, pick best goal score
        probs = []
        P3 = self.eng.poles[[self.eng.pi[k] for k in ("WIN", "DRAW", "LOSS")]].to(self.device)
        DBp = torch.stack([self.eng.net.dB(z, P3[[k]].expand(len(z), -1)) for k in range(3)], 1)
        pr = torch.softmax(-DBp / 5.0, 1)
        e_w = (pr[:, 0] + 0.5 * pr[:, 1]).cpu().numpy()
        e_m = e_w if board.turn == _ch.WHITE else 1.0 - e_w
        ok = e_m >= np.quantile(e_m, 0.5)
        score = pg.cpu().numpy() * (1 if not minimize else -1)
        score[~ok] = -1e9
        return moves[int(np.argmax(score))], len(moves)

    def candidates(self, k_lev=12, extra=()):
        """default token set: top-|leverage| codes each way + tempo/check slots + extras."""
        hcs = list(extra)
        if self.lev is not None:
            fl = np.argsort(-np.abs(self.lev).ravel())[:k_lev]
            hcs += [(int(i // self.C), int(i % self.C)) for i in fl]
        seen, out = set(), []
        for h, c in hcs:
            if (h, c) not in seen:
                seen.add((h, c)); out.append((h, c))
        return np.array(out, np.int64)


class SubgoalFormer(nn.Module):
    """GeoAttention stack + revised-P head. Supervised on reach-events ONLY (see module doc)."""

    F_TOK = 5

    def __init__(self, n_head=8, n_code=64, d_tok=64, heads=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_head * n_code, d_tok)
        self.side = nn.Embedding(2, d_tok)                 # ours / theirs
        self.f_in = nn.Linear(self.F_TOK, d_tok)
        self.n_code = n_code
        self.layers = nn.ModuleList(
            [GeoAttention(d_tok=d_tok, heads=heads) for _ in range(layers)])
        self.p_head = nn.Linear(d_tok, 1)                  # ONE FC: the interactions already
                                                           # happened in the attention

    def forward(self, hc, sides, feats, G, return_attn=False):
        E = (self.emb(hc[:, 0] * self.n_code + hc[:, 1])
             + self.side(sides) + self.f_in(feats))
        attns = []
        for lyr in self.layers:
            out = lyr(E, G, return_attn=return_attn)
            E, a = out if return_attn else (out, None)
            if a is not None:
                attns.append(a)
        p = torch.sigmoid(self.p_head(E)).squeeze(-1)
        return (p, E, attns) if return_attn else (p, E)

    @torch.no_grad()
    def certificate(self, hc, sides, feats, G, committed_idx):
        """the legible plan artifact. Worry/opportunity attribution is COUNTERFACTUAL:
        drop token b -> recompute p-hat(committed). Positive delta = b suppresses the plan
        (worry); negative = b supports it. Attention rides along as the consistency check."""
        p, _, attns = self.forward(hc, sides, feats, G, return_attn=True)
        base = float(p[committed_idx])
        T = len(hc)
        deltas = np.zeros(T, np.float32)
        for b in range(T):
            if b == committed_idx:
                continue
            keep = [i for i in range(T) if i != b]
            kt = torch.tensor(keep)
            pb, _ = self.forward(hc[kt], sides[kt], feats[kt], G[kt][:, kt])
            ci = keep.index(committed_idx)
            deltas[b] = float(pb[ci]) - base       # >0: removing b RAISES p-hat => b is a worry
        attn_row = attns[-1].mean(0)[committed_idx].cpu().numpy()
        return Certificate(hc=hc.cpu().numpy(), sides=sides.cpu().numpy(),
                           committed=int(committed_idx), p_hat=base,
                           p_all=p.cpu().numpy(), worry=deltas, attn=attn_row)


@dataclass
class Certificate:
    hc: np.ndarray            # (T,2) [head, code]
    sides: np.ndarray         # (T,) 0 ours / 1 theirs
    committed: int
    p_hat: float
    p_all: np.ndarray         # revised P per token
    worry: np.ndarray         # counterfactual delta per token (>0 = suppresses the plan)
    attn: np.ndarray          # last-layer attention row of the committed token (consistency)

    def premove_safe(self, p_min=0.97, worry_max=0.02):
        return self.p_hat >= p_min and float(self.worry.max(initial=0.0)) <= worry_max

    def render(self, names=None):
        nm = (lambda h, c: names.get((h, c), f"h{h}/c{c}")) if names else \
             (lambda h, c: f"h{h}/c{c}")
        h, c = self.hc[self.committed]
        lines = [f"committed: {nm(h, c)}  p={self.p_hat:.2f}  "
                 f"premove={'SAFE' if self.premove_safe() else 'no'}"]
        for i in np.argsort(-self.worry)[:4]:
            if self.worry[i] > 0.005:
                lines.append(f"  worry: {nm(*self.hc[i])} "
                             f"d_p={self.worry[i]:+.3f} attn={self.attn[i]:.2f}")
        return "\n".join(lines)


@dataclass
class Alert:
    hc: tuple
    side: int
    kind: str                 # "opportunity" | "worry"
    salience: float
    d_p: float
    feats: np.ndarray = field(default=None)


def alert_set(cert_prev, cert_now, feats_now, k=16, lev=None):
    """certificate DIFF -> top-K worries AND opportunities (one object, sign apart).
    A blunder is a diff: their protective concept dropping / our p-hat jumping surfaces
    here with no dedicated machinery (M7 armed-tactics is this diff's special case)."""
    out = []
    prev = {tuple(cert_prev.hc[i]): cert_prev.p_all[i] for i in range(len(cert_prev.hc))}
    for i in range(len(cert_now.hc)):
        key = tuple(cert_now.hc[i])
        d_p = float(cert_now.p_all[i] - prev.get(key, cert_now.p_all[i]))
        w = float(abs(lev[key[0], key[1]])) if lev is not None else 1.0
        sal = abs(d_p) * (0.05 + w)
        if sal <= 0:
            continue
        ours = int(cert_now.sides[i]) == 0
        kind = "opportunity" if (d_p > 0) == ours else "worry"
        out.append(Alert(hc=key, side=int(cert_now.sides[i]), kind=kind,
                         salience=sal, d_p=d_p,
                         feats=feats_now[i].cpu().numpy() if hasattr(feats_now[i], "cpu")
                         else np.asarray(feats_now[i])))
    out.sort(key=lambda a: -a.salience)
    return out[:k]
