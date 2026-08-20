#!/usr/bin/env python
"""concept_engine.py -- ConceptAlphaBeta: gcf3-native negamax for KittyChess.

Kaveh 2026-08-20: "set the search to be minimax with alpha beta pruning ranked by our
concepts." Both halves of the search ARE the concept model:

    ordering   the gcf3 MOVE HEAD -- moves scored through the quantized concept state
               (zq + move-node), so the alpha-beta visits children in concept-plausibility
               order; good ordering is what makes the pruning bite
    leaf eval  the gcf3 RESOLUTION head's W/D/L expectation, mover-POV margin
               E = P(win) - P(loss); mate/stalemate detected exactly at any depth

Duck-typed to the surfaces kittychess_server actually calls: search / negamax / margins /
wdl / cascade_rank / device / concepts (the new per-position panel payload).
"""
from __future__ import annotations

import numpy as np
import torch

import chess

MATE = 100.0

# head labels from the 2026-08-20 alignment eval on gcf3 (eta^2 evidence; heads whose top
# feature is weak stay honestly unnamed)
HEAD_LABELS = {
    0: "phase / global state", 1: "pawn structure", 2: "mobility & tension",
    3: "(mixed slow)", 4: "knights", 5: "rooks & majors", 6: "material balance",
    7: "queens", 8: "(fast 8)", 9: "pawn count (fast)", 10: "(fast 10)",
    11: "(fast 11)", 12: "phase shift (fast)", 13: "(fast 13)", 14: "(fast 14)",
    15: "material swing (fast)",
}


class ConceptAlphaBeta:
    def __init__(self, ckpt, device="cpu", width=8,
                 stats_path="data/derived/concept_stats_gcf3.npz"):
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .graph_concept_field import GraphConceptField
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .concept_capture_field import factor_vocab
        from catspace.research.components.encoder.approaches.reach_probability.src import (
            trajectories as T)
        from catspace.io import paths
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
        tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                     max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
        lz = np.load(paths.derived(f"tb_entry_5-4-3men_{len(tr.tok)}.npz"), allow_pickle=True)
        vocab = [str(x) for x in lz["vocab"]]
        cr, ce, co, egs = factor_vocab(vocab)
        self.net = GraphConceptField(
            d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"],
            n_res=len(vocab), n_eg=len(egs), cls_reach=cr, cls_eg=ce, cls_out=co,
            edge_codes=A["edge_codes"], move_codes=A["move_codes"],
            gps_parallel=bool(A["gps_parallel"]), attn_bias=bool(A["attn_bias"]),
            agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
            file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
            threat_head=bool(A["threat_head"])).to(device)
        self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.device = device
        self.width = width
        self.tag = ckpt.split("/")[-1]
        try:
            sz = np.load(stats_path, allow_pickle=True)
            self.stat_mean, self.stat_cnt = sz["mean_wpov"], sz["count"]
            self.stat_ex2 = sz["ex2"] if "ex2" in sz.files else None
        except Exception:
            self.stat_mean = self.stat_cnt = self.stat_ex2 = None
        self._cache = {}

    class SearchStop(Exception):
        pass

    # ---- gcf3 forward on a batch of boards ------------------------------------
    @torch.no_grad()
    def _forward(self, boards):
        """-> list of dicts (moves, order, margin mover-POV, wdl White-POV, g_ids, te)."""
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .graph_move_policy import featurize, MV_CAP
        misses = [b for b in boards if b.fen() not in self._cache]
        if misses:
            gs = [parent_graph(b)[0] for b in misses]
            D = {k: torch.from_numpy(np.stack([g[k] for g in gs]))
                 for k in ("sq", "mv", "mut", "vac", "dsc", "att")}
            D["gl"] = torch.from_numpy(np.stack([glob_of_board(b) for b in misses]))
            D["n_mv"] = torch.tensor([g["n_mv"] for g in gs], dtype=torch.uint8)
            D["n_att"] = torch.tensor([g["n_att"] for g in gs], dtype=torch.uint16)
            dev = self.device
            bb = {k: D[k].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
            sqf, mvf, edges, mask = featurize(bb, dev, n_nodes=self.net.n_nodes)
            off = (torch.arange(len(misses), device=dev) * self.net.n_nodes)[:, None]
            jj = torch.arange(MV_CAP, device=dev)[None, :]
            for name in ("vac", "dsc"):
                arr = D[name].to(dev).long()
                val = (arr != 255) & mask[:, :, None]
                edges[name] = (((jj + 64 + off)[:, :, None].expand_as(arr))[val],
                               (arr + off[:, :, None])[val])
            zq, g_ids, sq_ids, _, h_mv, pre_g = self.net.encode(sqf, mvf, edges, mask)
            lg = self.net.move_logits(zq, h_mv, mask)
            _, _, out_l = self.net.predict_resolution(zq)
            pwdl = torch.softmax(out_l, -1).cpu().numpy()      # White-POV [W, D, L]
            te = (torch.sigmoid(self.net.te_head(self.net.last_h_sq).squeeze(-1)).cpu().numpy()
                  if self.net.threat_head else None)
            lgn = lg.cpu().numpy(); gin = g_ids.cpu().numpy()
            for i, b in enumerate(misses):
                n = int(D["n_mv"][i])
                mvs = list(b.legal_moves)[:n]
                sc = lgn[i, :n]
                order = list(np.argsort(-sc))
                w, d, l = float(pwdl[i, 0]), float(pwdl[i, 1]), float(pwdl[i, 2])
                margin = (w - l) if b.turn == chess.WHITE else (l - w)
                self._cache[b.fen()] = dict(
                    moves=mvs, order=order, logits=sc, margin=margin, wdl=(w, d, l),
                    g_ids=gin[i], te=(te[i] if te is not None else None))
                if len(self._cache) > 200000:
                    self._cache.clear()
        return [self._cache[b.fen()] for b in boards]

    def _terminal_value(self, board):
        out = board.outcome(claim_draw=True)
        if out is None:
            return None
        if out.winner is None:
            return 0.0
        return -MATE          # side to move at a decided terminal is mated

    def margins(self, boards):
        """mover-POV margins, batched (used by the server's forcingness readout)."""
        return [f["margin"] for f in self._forward(boards)]

    def wdl(self, board):
        f = self._forward([board])[0]
        return list(f["wdl"]), None

    def cascade_rank(self, board):
        """concept-move-head ranking (the 'cascade' slot in the server's display logic)."""
        f = self._forward([board])[0]
        return f["moves"], f["order"], None

    # ---- the search: negamax alpha-beta, concept-ordered, concept-evaluated ----
    def negamax(self, board, depth, alpha, beta, stop=None):
        if stop is not None and stop():
            raise ConceptAlphaBeta.SearchStop
        tv = self._terminal_value(board)
        if tv is not None:
            return tv, []
        f = self._forward([board])[0]
        if depth <= 0:
            return f["margin"], []
        best_v, best_pv = -float("inf"), []
        for i in f["order"][:self.width]:
            m = f["moves"][i]
            board.push(m)
            try:
                v, pv = self.negamax(board, depth - 1, -beta, -alpha, stop)
            finally:
                board.pop()
            v = -v
            if v > best_v:
                best_v, best_pv = v, [m] + pv
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        if not best_pv and f["moves"]:
            m = f["moves"][f["order"][0]]
            best_v, best_pv = f["margin"], [m]
        return best_v, best_pv

    def search(self, board, depth=3, top=3, stop=None, progress=None, **kw):
        """root MultiPV in the KittyChess rows contract: [{'mv','pv','value'}...]."""
        tv = self._terminal_value(board)
        if tv is not None:
            return []
        f = self._forward([board])[0]
        sc = f["logits"] - f["logits"].max()
        pr = np.exp(sc); pr = pr / pr.sum()
        rows = []
        try:
            for i in f["order"][:max(top, self.width)]:
                m = f["moves"][i]
                board.push(m)
                try:
                    v, pv = self.negamax(board, depth - 1, -MATE, MATE, stop)
                finally:
                    board.pop()
                rows.append({"mv": m, "pv": [m] + pv, "value": -v,
                             "prior": round(float(pr[i]), 4)})
                rows.sort(key=lambda r: -r["value"])
                if progress is not None:
                    progress(rows)
        except ConceptAlphaBeta.SearchStop:
            pass
        return rows

    # ---- server-surface compatibility shims -------------------------------------
    def search_coherent(self, board, budget=1.5, **kw):
        """the server's one search entry point: budget (s) -> alpha-beta depth."""
        depth = 2 if budget < 1.0 else 3 if budget < 3.0 else 4
        return self.search(board, depth=depth, top=6)

    def rank_by_child_E(self, board, rows, *a, **kw):
        return rows                      # rows already sorted by searched value

    def bellman_residual(self, board):
        return None                      # no quasimetric field here; adaptive depth off

    # ---- the concepts panel -----------------------------------------------------
    def concepts(self, board, human_white=True):
        """per-position panel payload: active code per head, its label, and the corpus
        lean (mean White-POV outcome of positions sharing the code, with count)."""
        f = self._forward([board])[0]
        rows = []
        for h in range(len(f["g_ids"])):
            c = int(f["g_ids"][h])
            r = {"head": h, "label": HEAD_LABELS.get(h, f"head {h}"), "code": c,
                 "slow": h < 8}
            if self.stat_mean is not None:
                n = int(self.stat_cnt[h, c]); mw = float(self.stat_mean[h, c])
                # 95 percent CI on the mean: exact when the sweep stored E[x^2], else the
                # conservative bound var <= 1 - mu^2 (outcomes live in [-1, 1]). The chip
                # only claims a side when |mean| clears the CI (Kaveh 2026-08-20: "our
                # scores and deltas need CIs").
                if n > 1:
                    var = (float(self.stat_ex2[h, c]) - mw * mw) if self.stat_ex2 is not None \
                        else (1.0 - mw * mw)
                    ci = 1.96 * (max(var, 0.0) / n) ** 0.5
                else:
                    ci = 2.0
                r["n"] = n
                r["wpov"] = round(mw, 3)
                r["ci"] = round(ci, 3)
                r["helps"] = ("you" if mw - ci > 0 else "opp" if mw + ci < 0
                              else "even") if n > 30 else "?"
                if not human_white and r["helps"] in ("you", "opp"):
                    r["helps"] = "opp" if r["helps"] == "you" else "you"
            rows.append(r)
        w, d, l = f["wdl"]
        te_top = []
        if f["te"] is not None:
            idx = np.argsort(-f["te"])[:3]
            te_top = [{"sq": chess.SQUARE_NAMES[int(q)], "p": round(float(f["te"][q]), 2)}
                      for q in idx if f["te"][q] > 0.25 and board.piece_at(int(q))]
        return {"heads": rows, "wdl": [round(w, 3), round(d, 3), round(l, 3)],
                "phase_bucket": None, "exec_hot": te_top, "model": self.tag}
