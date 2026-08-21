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
# labels from the 2026-08-20 alignment eval, WITH their eta^2 share so a weak label
# discloses itself (Kaveh: a 33-percent-material head must not masquerade as "material
# balance" when its code flips for the other 67 percent of reasons)
HEAD_LABELS_GCF4H = {   # human model, 32 shallow heads (2026-08-20 alignment + elo evals)
    0: "phase+kingsafety 96%", 2: "elo-tier s115", 5: "elo-tier s149",
    8: "pawn-isolated (weak)", 9: "elo-tier s130", 10: "elo-tier s110",
    11: "tension+mobility 33%", 12: "elo-tier s112", 15: "elo-tier s111",
    18: "elo-tier f130", 20: "elo-tier f135", 27: "in-check f24", 28: "checks f9",
}
HEAD_LABELS = {
    0: "phase+kingsafety 95%", 1: "pawn counts 17%", 2: "mobility/tension 17%",
    3: "mixed-slow 10%", 4: "knights 11%", 5: "rooks+majors 23%",
    6: "material+minors 33%", 7: "queens 18%", 8: "fast-8 ?", 9: "pawn-count 39%",
    10: "fast-10 ?", 11: "fast-11 ?", 12: "phase-shift 43%", 13: "fast-13 ?",
    14: "fast-14 ?", 15: "material-swing 58%",
}


class ConceptAlphaBeta:
    def __init__(self, ckpt, device="cpu", width=8,
                 stats_path="data/derived/concept_stats_gcf3.npz", view_elo=1500.0):
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .graph_concept_field import GraphConceptField
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .concept_capture_field import factor_vocab
        from catspace.research.components.encoder.approaches.reach_probability.src import (
            trajectories as T)
        from catspace.io import paths
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
        if A.get("res_mode", "tb") == "tb":
            tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                         max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
            lz = np.load(paths.derived(f"tb_entry_5-4-3men_{len(tr.tok)}.npz"),
                         allow_pickle=True)
            vocab = [str(x) for x in lz["vocab"]]
            cr, ce, co, egs = factor_vocab(vocab)
        else:                     # outcome-mode checkpoint: no corpus/TB needed at all
            vocab, egs = ["W", "D", "L"], ["-"]
            cr = np.zeros(3, np.int64); ce = np.zeros(3, np.int64)
            co = np.arange(3, dtype=np.int64)
        if A.get("res_mode", "tb") != "tb":
            vocab, egs = ["W", "D", "L"], ["-"]
            cr = np.zeros(3, np.int64); ce = np.zeros(3, np.int64)
            co = np.arange(3, dtype=np.int64)
        self.net = GraphConceptField(
            d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"],
            n_res=len(vocab), n_eg=len(egs), cls_reach=cr, cls_eg=ce, cls_out=co,
            edge_codes=A["edge_codes"], move_codes=A["move_codes"],
            gps_parallel=bool(A["gps_parallel"]), attn_bias=bool(A["attn_bias"]),
            agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
            file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
            threat_head=bool(A["threat_head"]), elo_cond=bool(A.get("elo", 0)),
            g_heads=A.get("g_heads", 16), g_codes=A.get("g_codes", 32),
            d_code=A.get("d_code", 32)).to(device)
        self.labels = HEAD_LABELS_GCF4H if A.get("n_human", 0) else HEAD_LABELS
        self.elo_default = float(view_elo)
        self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.device = device
        self.width = width
        self.lock = None          # injected by the server: every net touch self-locks at
                                  # FORWARD granularity, so UI requests interleave between
                                  # node evaluations instead of queueing behind whole waves
        self.tag = ckpt.split("/")[-1]
        self._visited = 0          # positions touched (cache hits included), for the UI
        try:
            sz = np.load(stats_path, allow_pickle=True)
            self.stat_mean, self.stat_cnt = sz["mean_wpov"], sz["count"]
            self.stat_ex2 = sz["ex2"] if "ex2" in sz.files else None
            self.stat_ci = sz["ci_cluster"] if "ci_cluster" in sz.files else None
        except Exception:
            self.stat_mean = self.stat_cnt = self.stat_ex2 = self.stat_ci = None
        self._cache = {}

    class SearchStop(Exception):
        pass

    def _elo_in(self, n, dev):
        if not getattr(self.net, "elo_cond", False):
            return None
        from catspace.research.components.encoder.approaches.reach_probability \
            .src.trajectories import normalise_elo
        ev = float(normalise_elo(np.float32(self.elo_default)))
        return torch.full((n, 2), ev, device=dev)

    # ---- gcf3 forward on a batch of boards ------------------------------------
    @torch.no_grad()
    def _forward(self, boards):
        """-> list of dicts (moves, order, margin mover-POV, wdl White-POV, g_ids, te)."""
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .graph_move_policy import featurize, MV_CAP
        from contextlib import nullcontext
        self._visited += len(boards)
        misses = [b for b in boards if b.fen() not in self._cache]
        if misses:
          with (self.lock if self.lock is not None else nullcontext()):
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
              elo_in = self._elo_in(len(misses), dev)
              if self.net.edge_codes:
                  zq, g_ids, sq_ids, _, h_mv, pre_g, _hist = self.net.encode(
                      sqf, mvf, edges, mask, want_edge_ids=True, elo=elo_in)
                  e_ids = self.net.last_edge_ids.cpu().numpy()
                  e_s = (self.net.last_edge_sd[0] % self.net.n_nodes).cpu().numpy()
                  e_d = (self.net.last_edge_sd[1] % self.net.n_nodes).cpu().numpy()
                  e_samp = (self.net.last_edge_sd[1] // self.net.n_nodes).cpu().numpy()
              else:
                  zq, g_ids, sq_ids, _, h_mv, pre_g = self.net.encode(sqf, mvf, edges,
                                                                      mask, elo=elo_in)
                  e_ids = None
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
                  edges_i = None
                  if e_ids is not None:
                      m_i = e_samp == i
                      ei, es, ed = e_ids[m_i], e_s[m_i], e_d[m_i]
                      keep = (es < 64) & (ed < 64)
                      ei, es, ed = ei[keep], es[keep], ed[keep]
                      edges_i = {}
                      if len(ei):
                          for c2 in np.bincount(ei).argsort()[::-1][:8]:
                              j2 = np.flatnonzero(ei == c2)
                              if len(j2):
                                  exs = [(int(es[j3]), int(ed[j3])) for j3 in j2[:2]]
                                  edges_i[int(c2)] = (int(len(j2)), exs)
                  self._cache[b.fen()] = dict(
                      moves=mvs, order=order, logits=sc, margin=margin, wdl=(w, d, l),
                      g_ids=gin[i], te=(te[i] if te is not None else None),
                      sq_ids=sq_ids[i].cpu().numpy(), edge_top=edges_i)
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
        if depth == 1:
            # LEAF LEVEL, BATCHED: all top-width children in ONE forward (2026-08-20
            # "why is it so slow" -- unbatched leaves dominated the node count)
            cand = f["order"][:self.width]
            vals, kids = [], []
            for i in cand:
                board.push(f["moves"][i])
                tv2 = self._terminal_value(board)
                if tv2 is not None:
                    vals.append((-tv2, i))
                else:
                    kids.append((i, board.copy(stack=False)))
                board.pop()
            if kids:
                for (i, _b2), fc in zip(kids, self._forward([b2 for _, b2 in kids])):
                    vals.append((-fc["margin"], i))
            if not vals:
                return f["margin"], []
            v, bi = max(vals)
            return v, [f["moves"][bi]]
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
                             "prior": round(float(pr[i]), 4), "depth_used": depth})
                rows.sort(key=lambda r: -r["value"])
                if progress is not None:
                    progress(rows)
        except ConceptAlphaBeta.SearchStop:
            pass
        return rows

    def warm_openings(self, plies=3, chunk=256, lock=None, log=print):
        """prewarm the fen cache with the opening tree (Kaveh 2026-08-20: "cache the
        embeddings of the start position and all first few legal moves"). Runs in a
        background thread at startup; takes `lock` per batch so live requests interleave
        instead of queueing behind the whole warmup."""
        import time as _t
        import chess as _c
        t0 = _t.time()
        seen, frontier = {_c.Board().fen()}, [_c.Board()]
        boards = [_c.Board()]
        for _ in range(plies):
            nxt = []
            for b in frontier:
                for m in b.legal_moves:
                    b.push(m)
                    f = b.fen()
                    if f not in seen:
                        seen.add(f); c = b.copy(stack=False)
                        nxt.append(c); boards.append(c)
                    b.pop()
            frontier = nxt
        done = 0
        for a in range(0, len(boards), chunk):
            batch = boards[a:a + chunk]
            if lock is not None:
                with lock:
                    self._forward(batch)
            else:
                self._forward(batch)
            done += len(batch)
        log(f"[concept-engine] opening cache warm: {done:,} positions "
            f"({plies} plies) in {_t.time()-t0:.0f}s; cache {len(self._cache):,}")

    @torch.no_grad()
    def move_concept_deltas(self, board):
        """Kaveh 2026-08-20: "which concepts are likely to change with that move". The
        dynamics head predicts the CHILD's global codes from (concept state, move); heads
        whose predicted code differs from the current one are the expected concept
        changes, each scored by the corpus-lean delta (White-POV) with propagated
        cluster-CIs. One encode + one tiny MLP batch for ALL legal moves; fen-cached."""
        from contextlib import nullcontext
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .graph_move_policy import featurize, MV_CAP
        key = board.fen()
        if not hasattr(self, "_cd_cache"):
            self._cd_cache = {}
        if key in self._cd_cache:
            return self._cd_cache[key]
        with (self.lock if self.lock is not None else nullcontext()):
            g, _ = parent_graph(board)
            D = {k: torch.from_numpy(np.asarray(g[k])[None])
                 for k in ("sq", "mv", "mut", "vac", "dsc", "att")}
            D["gl"] = torch.from_numpy(glob_of_board(board)[None])
            D["n_mv"] = torch.tensor([g["n_mv"]], dtype=torch.uint8)
            D["n_att"] = torch.tensor([g["n_att"]], dtype=torch.uint16)
            dev = self.device
            bb = {k: D[k].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
            sqf, mvf, edges, mask = featurize(bb, dev, n_nodes=self.net.n_nodes)
            off = (torch.arange(1, device=dev) * self.net.n_nodes)[:, None]
            jj = torch.arange(MV_CAP, device=dev)[None, :]
            for name in ("vac", "dsc"):
                arr = D[name].to(dev).long()
                val = (arr != 255) & mask[:, :, None]
                edges[name] = (((jj + 64 + off)[:, :, None].expand_as(arr))[val],
                               (arr + off[:, :, None])[val])
            enc = self.net.encode(sqf, mvf, edges, mask, elo=self._elo_in(1, dev))
            zq, g_ids = enc[0], enc[1]
            n = int(g["n_mv"])
            emb = self.net.mv_emb(mvf[0, :n])
            dy = self.net.dyn(torch.cat([zq.expand(n, -1), emb], -1))
            Hh, C = self.net.g_heads, 32
            pred = dy[:, :Hh * C].reshape(n, Hh, C).argmax(-1).cpu().numpy()
        cur = g_ids[0].cpu().numpy()
        mvs = list(board.legal_moves)[:n]
        out = {}
        for j, m in enumerate(mvs):
            ch = []
            for h in range(Hh):
                new = int(pred[j, h])
                if new == int(cur[h]) or self.stat_mean is None:
                    continue
                dw = float(self.stat_mean[h, new] - self.stat_mean[h, int(cur[h])])
                if getattr(self, "stat_ci", None) is not None:
                    ci = float((self.stat_ci[h, new] ** 2
                                + self.stat_ci[h, int(cur[h])] ** 2) ** 0.5)
                else:
                    ci = 1.0
                ch.append({"label": self.labels.get(h, f"h{h}"),
                           "fr": int(cur[h]), "to": new,
                           "dw": round(dw, 3), "ci": round(ci, 3),
                           "sig": abs(dw) > ci})
            ch.sort(key=lambda x: (not x["sig"], -abs(x["dw"])))
            out[m.uci()] = ch[:3]
        if len(self._cd_cache) > 2000:
            self._cd_cache.clear()
        self._cd_cache[key] = out
        return out

    def watch(self, board, hist_fens=None, human_white=True):
        """Kaveh 2026-08-20: "which concepts are likely to change and become bad for me
        ... what restricts my move choice, and just changed recently." Three readouts,
        all in concept-flip language (changes, not states):

          recent    flips that actually happened in the last plies of the game
          threats   null-move view: the opponent's prior-weighted likely flips vs me
          restrict  bad flips present in MOST of my moves -- the constraints -- with
                    the escape moves that avoid each

        White-POV lean deltas throughout; 'bad for me' resolved by human_white."""
        import chess as _c
        sgn = 1.0 if human_white else -1.0
        out = {"recent": [], "threats": [], "restrict": []}
        if self.stat_mean is None:
            return out

        def flip_dw(h, fr, to):
            dw = float(self.stat_mean[h, to] - self.stat_mean[h, fr])
            ci = float((self.stat_ci[h, to] ** 2 + self.stat_ci[h, fr] ** 2) ** 0.5) \
                if getattr(self, "stat_ci", None) is not None else 1.0
            return dw, ci

        # ---- recent: diff the game's code history ---------------------------------
        if hist_fens:
            boards = [_c.Board(f) for f in hist_fens[-5:]] + [board]
            fs = self._forward(boards)
            for k in range(len(fs) - 1):
                ago = len(fs) - 1 - k
                a, b2 = fs[k]["g_ids"], fs[k + 1]["g_ids"]
                for h in range(len(a)):
                    if int(a[h]) != int(b2[h]):
                        dw, ci = flip_dw(h, int(a[h]), int(b2[h]))
                        if abs(dw) > ci:
                            out["recent"].append(
                                {"label": self.labels.get(h, f"h{h}"), "ago": ago,
                                 "fr": int(a[h]), "to": int(b2[h]),
                                 "dw": round(dw, 3), "ci": round(ci, 3)})
            out["recent"].sort(key=lambda x: (x["ago"], -abs(x["dw"])))
            out["recent"] = out["recent"][:4]

        # ---- threats: what the opponent's plausible moves do to the concepts ------
        bn = board.copy(stack=False)
        try:
            bn.push(_c.Move.null())
            cd_opp = self.move_concept_deltas(bn)
            f_opp = self._forward([bn])[0]
            pr = np.exp(f_opp["logits"] - f_opp["logits"].max())
            pr = pr / pr.sum()
            agg = {}
            for j, m in enumerate(f_opp["moves"]):
                for cdx in cd_opp.get(m.uci(), []):
                    if cdx["dw"] * sgn < 0 and cdx["sig"]:       # bad for the human
                        k2 = (cdx["label"], cdx["to"])
                        e = agg.setdefault(k2, {"label": cdx["label"], "to": cdx["to"],
                                                "dw": cdx["dw"], "p": 0.0, "ex": m})
                        e["p"] += float(pr[j])
            th = sorted(agg.values(), key=lambda x: -x["p"] * abs(x["dw"]))[:3]
            out["threats"] = [{"label": t["label"], "dw": round(t["dw"], 3),
                              "p": round(t["p"], 2), "ex": board.san(t["ex"])
                              if t["ex"] in board.legal_moves else t["ex"].uci()}
                             for t in th]
        except Exception:
            pass

        # ---- restrictions: bad flips hanging over MOST of my moves ----------------
        cd_me = self.move_concept_deltas(board)
        f_me = self._forward([board])[0]
        n_mv = len(f_me["moves"])
        if n_mv:
            per_move = [(m, {(c["label"], c["to"]) for c in cd_me.get(m.uci(), [])
                             if c["dw"] * sgn < 0 and c["sig"]})
                        for m in f_me["moves"]]
            cnt = {}
            for _m, fl in per_move:
                for k2 in fl:
                    cnt[k2] = cnt.get(k2, 0) + 1
            esc = {k2: [m for m, fl in per_move if k2 not in fl] for k2 in cnt}
            for k2, c2 in sorted(cnt.items(), key=lambda kv: -kv[1]):
                frac = c2 / n_mv
                if frac >= 0.5:
                    escapes = esc.get(k2, [])
                    pr_me = np.exp(f_me["logits"] - f_me["logits"].max())
                    pr_me = pr_me / pr_me.sum()
                    escapes.sort(key=lambda m: -pr_me[f_me["moves"].index(m)])
                    out["restrict"].append(
                        {"label": k2[0], "frac": round(frac, 2),
                         "esc": [board.san(m) for m in escapes[:4]]})
            out["restrict"] = out["restrict"][:3]
        return out

    @torch.no_grad()
    def concept_attribution(self, board, head):
        """Kaveh 2026-08-20: "tie concepts to pieces or squares". Exact by architecture:
        pre_g[h] = W_h . MEAN_q(h_sq[q]), so square q's vote for the ACTIVE code over its
        nearest rival is  a_q = (W_h h_sq[q]/64) . (e_active - e_runnerup)  -- closed form,
        one forward. Plus the causal tie: pieces whose REMOVAL flips this head's code
        (reuses the batched removal forwards)."""
        from contextlib import nullcontext
        import chess as _c
        h = int(head)
        f_now = self._forward([board])[0]
        cur = int(f_now["g_ids"][h])
        with (self.lock if self.lock is not None else nullcontext()):
            # fresh single forward for the h_sq stash (cache bypass is fine: cheap)
            from catspace.research.tools.training_infra.gen_move_graphs import (
                parent_graph, glob_of_board)
            from catspace.research.components.encoder.approaches.reach_probability.experiments \
                .graph_move_policy import featurize, MV_CAP
            g, _ = parent_graph(board)
            D = {k: torch.from_numpy(np.asarray(g[k])[None])
                 for k in ("sq", "mv", "mut", "vac", "dsc", "att")}
            D["gl"] = torch.from_numpy(glob_of_board(board)[None])
            D["n_mv"] = torch.tensor([g["n_mv"]], dtype=torch.uint8)
            D["n_att"] = torch.tensor([g["n_att"]], dtype=torch.uint16)
            dev = self.device
            bb = {k: D[k].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
            sqf, mvf, edges, mask = featurize(bb, dev, n_nodes=self.net.n_nodes)
            off = (torch.arange(1, device=dev) * self.net.n_nodes)[:, None]
            jj = torch.arange(MV_CAP, device=dev)[None, :]
            for name in ("vac", "dsc"):
                arr = D[name].to(dev).long()
                val = (arr != 255) & mask[:, :, None]
                edges[name] = (((jj + 64 + off)[:, :, None].expand_as(arr))[val],
                               (arr + off[:, :, None])[val])
            enc = self.net.encode(sqf, mvf, edges, mask, elo=self._elo_in(1, dev))
            pre_g = enc[5]
            h_sq = self.net.last_h_sq[0]                       # (64, d)
            dc = self.net.d_code
            W_h = self.net.g_proj.weight[h * dc:(h + 1) * dc]  # (d_code, d)
            emb = self.net.vq[h]._codebook.embed[0]            # (32, d_code)
            dist = ((pre_g[0, h][None] - emb) ** 2).sum(-1)
            order = dist.argsort()
            c1 = int(order[0]); c2 = int(order[1])
            direction = (emb[c1] - emb[c2])                    # decision direction
            contrib = (h_sq @ W_h.T) / 64.0                    # (64, d_code)
            a = (contrib @ direction).cpu().numpy()            # per-square vote
        sal = [{"sq": _c.SQUARE_NAMES[q], "a": round(float(a[q]), 4),
                "pc": (board.piece_at(q).unicode_symbol() if board.piece_at(q) else None)}
               for q in np.argsort(-np.abs(a))[:10] if abs(a[q]) > 1e-6]
        # causal: removal flips (batched, shares the why-cache)
        occ = [q for q in range(64)
               if board.piece_at(q) and board.piece_at(q).piece_type != _c.KING]
        boards2 = []
        for q in occ:
            b2 = board.copy(stack=False)
            b2.remove_piece_at(q); b2.ep_square = None
            boards2.append(b2)
        held = []
        if boards2:
            for q, fr in zip(occ, self._forward(boards2)):
                new = int(fr["g_ids"][h])
                if new != cur:
                    dw, ci = (float(self.stat_mean[h, new] - self.stat_mean[h, cur]),
                              0.0) if self.stat_mean is not None else (0.0, 0.0)
                    held.append({"sq": _c.SQUARE_NAMES[q],
                                 "pc": board.piece_at(q).unicode_symbol(),
                                 "to": new, "dw": round(dw, 3)})
        held.sort(key=lambda x: -abs(x["dw"]))
        return {"head": h, "code": cur, "runner_up": c2, "sal": sal, "held": held[:6]}

    # ---- server-surface compatibility shims -------------------------------------
    def search_coherent(self, board, budget=1.5, stop=None, progress=None, **kw):
        """the server's one search entry point: budget (s) -> alpha-beta depth. stop and
        progress stream through to the root loop: lines publish per candidate, and a new
        move aborts mid-wave."""
        depth = 2 if budget < 1.0 else 3 if budget < 3.5 else 4
        return self.search(board, depth=depth, top=6, stop=stop, progress=progress)

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
            r = {"head": h, "label": self.labels.get(h, f"head {h}"), "code": c,
                 "slow": h < 8}
            if self.stat_mean is not None:
                n = int(self.stat_cnt[h, c]); mw = float(self.stat_mean[h, c])
                # 95 percent CI on the mean: exact when the sweep stored E[x^2], else the
                # conservative bound var <= 1 - mu^2 (outcomes live in [-1, 1]). The chip
                # only claims a side when |mean| clears the CI (Kaveh 2026-08-20: "our
                # scores and deltas need CIs").
                if getattr(self, "stat_ci", None) is not None:
                    ci = float(self.stat_ci[h, c])   # cluster-robust by GAME (honest n)
                elif n > 1:
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
        sq_out = None
        if f.get("sq_ids") is not None:
            sq_out = []
            for q in range(64):
                pc = board.piece_at(q)
                imp = 0.0
                if f["te"] is not None and pc is not None:
                    # capture-risk map, White-POV sign: green = a BLACK piece likely
                    # falls (helps White), red = a White piece in danger
                    imp = float(f["te"][q]) * (1.0 if not pc.color else -1.0)
                sq_out.append({"sq": chess.SQUARE_NAMES[q], "code": int(f["sq_ids"][q]),
                               "imp": round(imp, 3),
                               "pc": (pc.unicode_symbol() if pc else None)})
        edge_out = None
        if f.get("edge_top"):
            edge_out = [{"code": c2, "n": n2,
                         "ex": [f"{chess.SQUARE_NAMES[s2]}\u2192{chess.SQUARE_NAMES[d2]}"
                                for (s2, d2) in exs]}
                        for c2, (n2, exs) in sorted(f["edge_top"].items(),
                                                    key=lambda kv: -kv[1][0])]
        return {"heads": rows, "wdl": [round(w, 3), round(d, 3), round(l, 3)],
                "phase_bucket": None, "exec_hot": te_top, "model": self.tag,
                "sq": sq_out, "edges": edge_out}
