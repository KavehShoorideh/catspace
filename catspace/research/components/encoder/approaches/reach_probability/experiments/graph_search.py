#!/usr/bin/env python
"""graph_search.py -- basic depth-controlled search over the gdv graph policy.

Kaveh 2026-08-20: "take the best one that we have and put on top of it some basic search,
allow me to control the search depth." The best one is the gdv prior (+88=12-0 vs random,
+81=19-0 vs material-greedy); this wraps it in a beam negamax whose VALUE is the path-sum
of dV -- the per-move advantage head, in game points, gauge dV(best)~0. Because V(s) terms
telescope out when comparing root moves at equal depth, minimaxing the alternating
advantage sum ranks root moves without ever needing an absolute evaluator:

    q(m at s, depth d) = terminal(m) ? +-10 / 0 : dV(m) - max_reply q(reply, d-1)

Terminal values dominate the dV scale (+-10 vs +-1ish), so found mates are always taken.
The beam is prior-ordered (the trained plausibility orderer) with the top-2 dV moves
unioned in as a tactical safety net. One graph forward scores ALL legal moves at a node,
so cost = nodes visited, ~8-15 ms each on CPU.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

MATE = 10.0


class GdvSearch:
    def __init__(self, ckpt, device="cpu", cache_size=50000):
        import chess  # noqa: F401  (kept close: callers pass python-chess boards)
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .graph_delta_value import GraphDeltaValue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
        self.net = GraphDeltaValue(d=A["d_model"], layers=A["layers"],
                                   heads=A["attn_heads"]).to(device)
        self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.dev = device
        self.tag = f"{ckpt.split('/')[-1]}"
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self.nodes = 0

    @torch.no_grad()
    def node_eval(self, b):
        """-> (moves, prior, dv) for ALL legal moves (<=48), fen-cached."""
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .graph_delta_value import batch_of
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        key = b.fen()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        g, _ = parent_graph(b)
        D = {k: torch.from_numpy(np.asarray(g[k])[None])
             for k in ("sq", "mv", "mut", "vac", "dsc", "att")}
        D["gl"] = torch.from_numpy(glob_of_board(b)[None])
        D["n_mv"] = torch.tensor([g["n_mv"]], dtype=torch.uint8)
        D["n_att"] = torch.tensor([g["n_att"]], dtype=torch.uint16)
        sqf, mvf, edges, mask = batch_of(D, torch.arange(1), self.dev)
        prior, dv = self.net(sqf, mvf, edges, mask)
        n = int(g["n_mv"])
        mv = list(b.legal_moves)[:n]
        out = (mv, prior[0, :n].cpu().numpy().astype(np.float64),
               dv[0, :n].cpu().numpy().astype(np.float64))
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        self.nodes += 1
        return out

    def _terminal(self, b):
        """value from the perspective of the side to move at b, or None."""
        out = b.outcome(claim_draw=True)
        if out is None:
            return None
        if out.winner is None:
            return 0.0
        return -MATE          # side to move at a decided terminal is the mated side

    def _candidates(self, mv, prior, dv, beam):
        order_p = np.argsort(-prior)[:beam]
        order_d = np.argsort(-dv)[:2]
        seen, cand = set(), []
        for i in list(order_p) + list(order_d):
            if i not in seen:
                seen.add(i); cand.append(int(i))
        return cand

    def _rec(self, b, depth, beam):
        """best alternating advantage sum for the side to move; -> (q, pv_moves)."""
        t = self._terminal(b)
        if t is not None:
            return t, []
        if depth == 0:
            return 0.0, []
        mv, prior, dv = self.node_eval(b)
        best, best_pv = -1e9, []
        for i in self._candidates(mv, prior, dv, beam):
            b.push(mv[i])
            t2 = self._terminal(b)
            if t2 is not None:
                q, pv = dv[i] - t2, []
            else:
                q_r, pv = self._rec(b, depth - 1, beam)
                q = dv[i] - q_r
            b.pop()
            if q > best:
                best, best_pv = q, [mv[i]] + pv
        return best, best_pv

    def analyze(self, b, depth=2, beam=6, topk=8):
        """-> dict with per-move rows (searched root candidates first, then shallow rest)."""
        import chess
        self.nodes = 0
        mv, prior, dv = self.node_eval(b)
        p = np.exp(prior - prior.max()); p = p / p.sum()
        cand = self._candidates(mv, prior, dv, beam)
        rows = []
        for i in range(len(mv)):
            row = {"uci": mv[i].uci(), "san": b.san(mv[i]), "prior_p": float(p[i]),
                   "dv": float(dv[i]), "q": float(dv[i]), "depth": 1, "pv": []}
            if i in cand and depth >= 1:
                b.push(mv[i])
                t2 = self._terminal(b)
                if t2 is not None:
                    row["q"], pv = float(dv[i] - t2), []
                else:
                    q_r, pv = self._rec(b, depth - 1, beam)
                    row["q"] = float(dv[i] - q_r)
                row["depth"] = depth
                sans, bb = [], b.copy()
                for m2 in pv:
                    sans.append(bb.san(m2)); bb.push(m2)
                row["pv"] = [row["san"]] + sans
                b.pop()
            rows.append(row)
        rows.sort(key=lambda r: (-r["depth"], -r["q"]))
        return {"rows": rows[:max(topk, 1)], "nodes": self.nodes,
                "model": self.tag, "depth": depth, "beam": beam,
                "n_legal": len(mv), "capped": len(mv) == 48}
