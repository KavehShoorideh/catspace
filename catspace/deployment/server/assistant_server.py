#!/usr/bin/env python
"""catspace/deployment/server/assistant_server.py -- THE PLANNER AS CO-ANALYST (Kaveh 2026-07-25):
play against a weak maia (or engine of choice) in the browser while OUR planner assists:
it prompts 'let's calculate here' (probe-triggered), searches while you think, then shows
the top moves and -- when a plan is active -- the most likely LEAVES you'll end up in.
Every suggested idea carries a pencil-editable tag; edited names persist to
artifacts/experiments/concept_tags.jsonl as HUMAN LABELS for field regions/plans
(concept-extraction meets the planner).

Run:  .venv/bin/python -m catspace.deployment.server.assistant_server --port 8777
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.engine
import numpy as np

from catspace.approaches.bootstrap_mate import (OnlineMateBank, harvest, make_batched_energy_prior,
                                                make_boot_value, make_planner, mat_sig)
from catspace.approaches.bootstrap_mate import config as bootstrap_mate_config
from catspace.fields import FieldModel
from catspace.introspection import ProbeKit
from catspace.io import paths
from catspace.io.paths import REPO_ROOT, experiments_dir
from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS

ROOT = REPO_ROOT
ASSETS = REPO_ROOT / "catspace/research/tools/viz/viz/assets"
TAGS = experiments_dir() / "concept_tags.jsonl"


SHARED: dict = {}      # heavies built once: fm, banks, compute lock, atlas cache


class Session:
    """PER-USER game session (Kaveh: 'persistent chess game sessions per user' via a
    sid cookie). Heavy state (field model, banks, compute lock, atlas) is SHARED across
    sessions; each session owns its board/ctx/value-closures/opponent-engine/calc."""

    def __init__(self, args, sid="default"):
        self.args = args
        self.sid = sid
        self.last_seen = time.time()
        if not SHARED:
            import threading as _thr
            fm = FieldModel(args.field, device=args.device)
            pfx = args.banks_prefix
            bank = OnlineMateBank(fm, Path(pfx + "_bank.fens"))
            loss = OnlineMateBank(fm, Path(pfx + "_lossbank.fens"))
            draw = OnlineMateBank(fm, Path(pfx + "_drawbank.fens"))
            for bk in (bank, loss, draw):
                bk.sync()
            # ThreadingHTTPServer runs requests in threads, but torch-MPS and UMAP's
            # numba are NOT threadsafe -- one lock serializes all heavy compute.
            SHARED.update(fm=fm, bank=bank, loss=loss, draw=draw, compute=_thr.Lock())
        self.fm = SHARED["fm"]
        self.bank, self.loss, self.draw = SHARED["bank"], SHARED["loss"], SHARED["draw"]
        self.compute = SHARED["compute"]
        self.ctx: dict = {"plan": "direct", "hist": {}}
        self.times: dict = {}
        self.pinned = bool(getattr(args, "pin_model", ""))
        lm = args.pin_model if self.pinned else args.last_mile
        if not self.pinned:
            import glob as _g
            import re as _re
            cands = _g.glob(str(ROOT / "data/derived/sep/dtm_tok_r*.pt"))
            if cands:
                lm = max(cands, key=lambda p: int(_re.search(r"r(\d+)\.pt", p).group(1)))
        self.vfn = make_boot_value(self.fm, self.bank, self.times, self.loss,
                                   dtm_ckpt=lm or None,
                                   draw_bank=self.draw, game_ctx=self.ctx)
        self.pfn, self.pfnb = make_batched_energy_prior(args.energy, game_ctx=self.ctx)
        self.planner = make_planner(self.fm, self.bank)
        try:                            # 'seen N× before' from the experience store
            import sqlite3
            _edb = sqlite3.connect("data/derived/experience.sqlite",
                                   check_same_thread=False)
        except Exception:
            _edb = None
        self.probes = ProbeKit(self.fm, self.bank, self.loss, self.draw,
                               exp_db=_edb, game_ctx=self.ctx, prior_fn=self.pfn)
        self.opp = None
        self._lm = lm
        self.version = self._version_of(self._lm)
        if self.pinned:
            self.version["pinned"] = True
        self.new_game(args.opponent)

    _CUM = {0: 300, 1: 1000, 2: 3000, 3: 6000, 4: 12000}    # per-class cumulative quotas

    def _version_of(self, ckpt: str) -> dict:
        name = Path(ckpt).stem if ckpt else "none"
        pct = None
        if name.startswith("dtm_tok_r"):
            try:
                k = int(name.rsplit("r", 1)[1])
                pct = round(100 * self._CUM.get(k, 12000) / 12000, 1)
            except ValueError:
                pass
        return {"model": name, "data_pct": pct}

    def new_game(self, weights):
        if self.opp is not None:
            try:
                self.opp.quit()
            except Exception:
                pass
        self.opp = None
        try:
            self.opp = chess.engine.SimpleEngine.popen_uci(["lc0", f"--weights={weights}"])
        except Exception:
            try:                                    # container fallback: weak stockfish
                self.opp = chess.engine.SimpleEngine.popen_uci(["stockfish"])
                self.opp.configure({"Skill Level": 1})
                print("[assistant] lc0/maia unavailable -> stockfish skill 1", flush=True)
            except Exception:
                print("[assistant] no opponent engine -> analysis-only mode", flush=True)
        self.board = chess.Board()
        from collections import Counter
        self.ctx["hist"] = Counter({self.board.epd(): 1})
        self.ctx["plan"] = "direct"; self.ctx["target_pt"] = None
        self.idea_seq = 0

    def _prompt(self):
        """probe-triggered 'let's calculate here' with the reason."""
        s = self.probes.summary(self.board)
        try:                    # CALIBRATED belief (raw d_loss is meaningless OOD --
            if hasattr(self.vfn, "diag"):       # an opening 'reads' d~5 to an endgame
                s.update(self.vfn.diag(self.board))
        except Exception:
            pass
        reasons = []
        if s.get("child_dwin_margin", 1) < 0.02 and s.get("n_win", 0) > 0:
            reasons.append("the field is flat here -- intuition alone won't rank these moves")
        if s.get("prior_entropy", 0) > 2.6:
            reasons.append("many plausible candidates (high prior entropy)")
        if s.get("p_l", 0) > s.get("p_w", 0) and s.get("v", 0) < -0.1:
            reasons.append(f"engine reads danger: P(loss) {s['p_l']} > P(win) {s['p_w']}")
        if s.get("seen_across_games", 0) == 0:
            reasons.append("never seen this position before")
        if s.get("clock_headroom", 100) < 30:
            reasons.append("the fifty-move clock is squeezing us")
        return {"suggest": bool(reasons), "reasons": reasons[:3], "probe": s}

    def _tops_leaves(self, b, root):
        kids = sorted(root.children, key=lambda c: -c.N)[:5]
        top = [{"uci": c.move.uci(), "san": b.san(c.move), "visits": int(c.N),
                "q": round(float(c.W / max(c.N, 1)), 3)} for c in kids]
        # most likely LEAVES under the plan: follow max-N children from each top move
        leaves = []
        for c in kids[:3]:
            node, line = c, [b.san(c.move)]
            depth = 0
            bb = b.copy(stack=False); bb.push(c.move)
            fens = [bb.fen()]
            while node.children and depth < 8:
                nxt = max(node.children, key=lambda x: x.N)
                if nxt.N < 1:      # expanded but never simulated: beyond the search's
                    break          # evidence -- showing it would be prior, not search
                node = nxt
                line.append(bb.san(node.move)); bb.push(node.move)
                fens.append(bb.fen())
                depth += 1
            leaves.append({"line": " ".join(line), "fen": bb.fen(), "fens": fens,
                           "visits": int(node.N),
                           "v": round(float(node.W / max(node.N, 1)), 3)})
        return top, leaves

    # ---- ATLAS (Kaveh: 'a tab with a UMAP embedding of known B's -- mates, losses,
    # draws -- and a highlight of subgoal clusters for the selected sequence'). The
    # banks store B-side embeddings (goal identities); trajectories are projected with
    # embed_B too so states and goals live in ONE map. Feasibility is NEVER read off
    # this view (that's the directed field's job) -- the atlas is for seeing basins.
    def atlas_data(self, n_win=3000, n_other=1500):
        """Sampled + DISK-CACHED projection (Kaveh: 'too slow; sample a set of points and
        UMAP those; cache the umap and reuse if it hasn't changed'). Cache key = field ckpt
        + bank sizes bucketed to the nearest 1000, so ordinary bank growth reuses the fit
        and only a real shift (new field / +1000 mates) rebuilds."""
        if getattr(self, "_atlas", None) is None:
            import pickle
            cache_p = ROOT / "artifacts/experiments/atlas_cache.pkl"
            meta = dict(v=2, field=str(self.args.field), n_win=n_win, n_other=n_other,
                        nw=len(self.bank) // 1000, nl=len(self.loss) // 1000,
                        nd=len(self.draw) // 1000)
            if cache_p.exists():
                try:
                    c = pickle.load(open(cache_p, "rb"))
                    if c.get("meta") == meta:
                        self._atlas = c["atlas"]
                        print("[atlas] cache hit", flush=True)
                except Exception:
                    pass
            if getattr(self, "_atlas", None) is None:
                from umap import UMAP
                rng = np.random.default_rng(0)
                embs, kinds, sigs, epds = [], [], [], []
                for kind, bk, cap in (("win", self.bank, n_win),
                                      ("loss", self.loss, n_other),
                                      ("draw", self.draw, n_other)):
                    E = bk.embs
                    if E is None or len(E) == 0:
                        continue
                    # bank file order (dedup-first) == emb append order
                    seen, ordered = set(), []
                    for line in bk.bank_file.read_text().splitlines():
                        e = line.strip()
                        if e and e not in seen:
                            seen.add(e); ordered.append(e)
                    idx = rng.permutation(len(E))[:min(len(E), cap)]
                    embs.append(np.asarray(E)[idx]); kinds += [kind] * len(idx)
                    bs = bk.sigs
                    sigs += [bs[i] if kind == "win" and i < len(bs) else "" for i in idx]
                    epds += [ordered[i] if i < len(ordered) else "" for i in idx]
                X = np.concatenate(embs, 0).astype(np.float32)
                um = UMAP(n_components=2, n_neighbors=25, min_dist=0.15, random_state=0)
                xy = um.fit_transform(X)
                self._atlas = dict(um=um, xy=xy, kinds=kinds, X=X, sigs=sigs, epds=epds)
                try:
                    pickle.dump({"meta": meta, "atlas": self._atlas}, open(cache_p, "wb"))
                except Exception:
                    pass
        a = self._atlas
        return {"pts": [[round(float(x), 3), round(float(y), 3), k]
                        for (x, y), k in zip(a["xy"], a["kinds"])]}

    def atlas_plan(self):
        """Plan -> SUBGOAL rendering (Kaveh: 'arrows through subgoals when I click the
        plan; see the subgoals somehow'). The plan's goal is a material CLASS; its
        subgoal cluster = the atlas's win points of that class (the density prior made
        visible). Arrow: current position -> subgoal centroid. Direct plans fall back to
        the nearest-mates cluster."""
        if getattr(self, "_atlas", None) is None:
            return {"err": "atlas not built yet"}
        a = self._atlas
        E = np.asarray(self.fm.embed_B_boards([self.board]), dtype=np.float32)
        cur = a["um"].transform(E)[0]
        cs = getattr(self, "calc", {}) or {}
        plan, goal = cs.get("plan") or "direct", cs.get("goal")
        mem = [i for i, s in enumerate(a["sigs"]) if goal and s == goal]
        label = f"{plan} → {goal}" if mem else f"{plan} → nearest mate basin"
        if not mem:      # direct / unknown class: k nearest banked mates in B-space
            win_idx = np.array([i for i, k in enumerate(a["kinds"]) if k == "win"])
            d = ((a["X"][win_idx] - E[0]) ** 2).sum(1)
            mem = [int(i) for i in win_idx[np.argsort(d)[:40]]]
        cx, cy = a["xy"][mem].mean(0)
        return {"from": [round(float(cur[0]), 3), round(float(cur[1]), 3)],
                "centroid": [round(float(cx), 3), round(float(cy), 3)],
                "members": mem[:500], "label": label}

    def atlas_select(self, fens, k=24):
        if getattr(self, "_atlas", None) is None:
            return {"err": "atlas not built yet -- open the atlas tab first"}
        boards = [chess.Board(f) for f in fens]
        E = np.asarray(self.fm.embed_B_boards(boards), dtype=np.float32)
        xy = self._atlas["um"].transform(E)
        # subgoal cluster highlight: nearest banked WIN points to the line's END
        X, kinds = self._atlas["X"], self._atlas["kinds"]
        win_idx = np.array([i for i, kk in enumerate(kinds) if kk == "win"])
        d = ((X[win_idx] - E[-1]) ** 2).sum(1)
        near = win_idx[np.argsort(d)[:k]]
        self._atlas_sel = {"path": [[round(float(x), 3), round(float(y), 3)] for x, y in xy],
                           "near": [int(i) for i in near]}
        return {"ok": True, **self._atlas_sel}

    # ---- directed field helpers (the quasimetric is the substrate for cone/rivers)
    def _d_to_win(self, boards):
        """d(x -> nearest banked mate): forward cost-to-go over the win basin."""
        import torch
        if self.bank.embs is None or len(self.bank.embs) == 0:
            return np.full(len(boards), np.nan)
        return self.fm.d_to_bank(self.fm.embed_F_boards(boards), self.bank.embs)

    def _d_to_loss(self, boards):
        """d(x -> nearest banked LOSS): the OPPONENT'S goal region -- their plan is
        movement toward it, so prevention is measurable on the same field."""
        if self.loss.embs is None or len(self.loss.embs) == 0:
            return np.full(len(boards), np.nan)
        return self.fm.d_to_bank(self.fm.embed_F_boards(boards), self.loss.embs)

    def _threat(self, b, M_l, k_opp=5):
        """Opponent's PLAN at position b (null-move if it's our turn): their
        probability-weighted best CALIBRATED step toward the loss basin -- deltas in
        p_l = exp(-d_loss/M_l) space, never raw distances (raw d is meaningless
        off-support; the running-median temperature is what the value itself uses).
        Returns (threat, best_move, p_l_now, p_l_after_best)."""
        import math
        import chess as _c
        bb = b.copy(stack=False)
        if bb.turn == _c.WHITE:                 # what could they do if we PASSED?
            bb.push(_c.Move.null())
        if bb.is_game_over() or not M_l:
            return 0.0, None, None, None
        pr = self.pfn(bb)
        if not pr:
            return 0.0, None, None, None
        top = sorted(pr.items(), key=lambda kv: -kv[1])[:k_opp]
        kids = []
        for mv, _p in top:
            c = bb.copy(stack=False); c.push(mv)
            kids.append(c)
        pl_now = math.exp(-float(self._d_to_loss([bb])[0]) / M_l)
        d_after = self._d_to_loss(kids)
        best_i, best_t = None, 0.0
        for i, (mv, p) in enumerate(top):
            gain = p * max(0.0, math.exp(-float(d_after[i]) / M_l) - pl_now)
            if gain > best_t:
                best_t, best_i = gain, i
        if best_i is None:
            return 0.0, None, pl_now, None
        return best_t, top[best_i][0], pl_now, math.exp(-float(d_after[best_i]) / M_l)

    def cone(self, depth=2, width=6):
        """FORWARD REACHABILITY CONE (Kaveh: 'basins/rivers, focused on the cone in
        front of White'). BFS White's legal tree to `depth` full moves; each node's
        HEIGHT = d(node -> nearest mate). Downhill (height decreasing) = the 'river'
        toward the goal; a node where every child rises = a chute/dead-end. Feasibility
        is the directed field itself, not a symmetric projection (per the doctrine)."""
        import chess as _c
        root = self.board
        nodes = [{"id": 0, "fen": root.fen(), "san": "(now)", "depth": 0, "parent": -1}]
        frontier = [(0, root)]
        for d in range(depth):
            nxt = []
            for pid, b in frontier:
                if b.is_game_over():
                    continue
                # our move (White): rank children by field, keep the best `width`
                kids = []
                for mv in b.legal_moves:
                    c = b.copy(stack=False); c.push(mv)
                    kids.append((mv, c))
                hs = self._d_to_win([c for _, c in kids])
                order = np.argsort(hs)[:width]
                for oi in order:
                    mv, c = kids[oi]
                    nid = len(nodes)
                    nodes.append({"id": nid, "fen": c.fen(), "san": b.san(mv),
                                  "h": round(float(hs[oi]), 2), "depth": d + 1, "parent": pid})
                    # opponent reply (their single best under the energy model), so the
                    # cone follows the actual game tree, not just our wishes
                    if d + 1 < depth and not c.is_game_over():
                        pr = self.pfn(c)          # dict keyed by chess.Move
                        if pr:
                            best = max(pr, key=pr.get)
                            c2 = c.copy(stack=False); c2.push(best)
                            nxt.append((nid, c2))
            frontier = nxt
        # the 'river': greedy min-height chain from the root
        river, cur = [], 0
        while True:
            ch = [n for n in nodes if n["parent"] == cur]
            if not ch:
                break
            nb = min(ch, key=lambda n: n.get("h", 9e9))
            river.append(nb["id"]); cur = nb["id"]
        return {"nodes": nodes, "river": river}

    def energy(self):
        """-log P(move) under the SELECTED opponent model (Kaveh: 'visualize -log(P) for
        the selected engine'). Low -logP = the human-likely move; the engine's surprises
        are the high-(-logP) moves it still rates highly. Returned for the current position
        (per legal move) and, if a calc ran, along the top engine line."""
        import math
        pr = self.pfn(self.board)                 # dict keyed by chess.Move
        rows = sorted(([self.board.san(m), m.uci(), p,
                        round(-math.log(max(p, 1e-9)), 2)] for m, p in pr.items()),
                      key=lambda r: r[2], reverse=True)
        line = []
        cs = getattr(self, "calc", {}) or {}
        if cs.get("leaves"):
            # CHAINED -logP along the top line (Kaveh: 'the chaining of -logP, and the
            # chance of going astray'). -logP adds where P multiplies, so the running sum
            # IS the line's improbability -- the probability-product of the foveated
            # doctrine. 'Astray' tracks only OPPONENT plies: 1 - prod P(their moves) =
            # chance they deviate from this line somewhere before ply k. Our own plies
            # carry style info (inhuman/only-moves), not risk -- we choose our moves.
            b = self.board.copy(stack=False)
            we_move = b.turn
            p_stay = 1.0
            for san in cs["leaves"][0]["line"].split():
                try:
                    mv = b.parse_san(san)
                except Exception:
                    break
                p = self.pfn(b).get(mv, 1e-9)
                ours = b.turn == we_move
                if not ours:
                    p_stay *= max(p, 1e-9)
                line.append([san, round(-math.log(max(p, 1e-9)), 2),
                             round(100 * (1 - p_stay), 1), int(ours)])
                b.push(mv)
        return {"cohort": "maia/self mix", "moves": rows[:12], "line": line}

    def explore(self, ucis, k=6):
        """VARIATION EXPLORER (Kaveh: options -> opponent reply probabilities -> my
        replies, board advancing on hover, good-AND-surprising ranked higher).
        For the position after `ucis` from the current game position:
          - our move rows score = v + 0.12*min(-logP_opp, 4): value first, boosted when
            the opponent model finds the move unlikely (they won't have prepared it);
          - opponent rows rank by THEIR reply probability (that is the question the
            popup answers), with White-POV v shown so likely-AND-strong replies read
            as the real dangers."""
        import math
        b = self.board.copy(stack=True)
        for u in ucis:
            b.push(chess.Move.from_uci(u))
        if b.is_game_over(claim_draw=True):
            return {"moves": [], "over": True, "fen": b.fen(),
                    "result": b.result(claim_draw=True)}
        rk = self.board.epd()          # anchor once per GAME position: hovered nodes sit
        if hasattr(self.vfn, "set_anchor") and getattr(self, "_ex_anchor", None) != rk:
            self.vfn.set_anchor(self.board)     # within the anchor's +-ply carry
            self._ex_anchor = rk
        pr = self.pfn(b)
        moves = list(b.legal_moves)
        kids = []
        for mv in moves:
            c = b.copy(stack=False); c.push(mv)
            kids.append(c)
        vs = self.vfn(kids)
        rows = []
        for mv, c, v in zip(moves, kids, vs):
            p = float(pr.get(mv, 0.0))
            mlp = round(-math.log(max(p, 1e-9)), 2)
            rows.append(dict(san=b.san(mv), uci=mv.uci(), p=round(p, 3), mlp=mlp,
                             v=round(float(v), 3), fen=c.fen()))
        # SEARCH OVERLAY (Kaveh: 'my own planner's moves are horrible' -- they were raw
        # 1-ply field reads; the calculated tree has stress-tested rankings): walk the
        # kept tree along `ucis`; where it covers this node, its visit counts and Q
        # replace the raw value and dominate the ordering.
        searched = False
        rs = getattr(self, "_calc_resume", None)
        if rs and rs["epd"] == self.board.epd() and rs.get("root") is not None:
            node = rs["root"]
            for u in ucis:
                node = next((c for c in node.children
                             if c.move.uci() == u), None) if node else None
            if node is not None and node.children:
                stats = {c.move.uci(): (int(c.N), round(float(c.W / max(c.N, 1)), 3))
                         for c in node.children}
                for r in rows:
                    if r["uci"] in stats and stats[r["uci"]][0] > 0:
                        r["visits"], r["v"] = stats[r["uci"]]
                        searched = True
        ours = b.turn == chess.WHITE
        if ours:
            for r in rows:
                r["score"] = round(r["v"] + 0.12 * min(r["mlp"], 4.0), 3)
            rows.sort(key=lambda r: (-(r.get("visits", 0)), -r["score"]))
            rows = rows[:k]
            # PROPHYLAXIS (Kaveh: 'two types of moves: advancing my plan, preventing
            # theirs, or both -- name the prevention'). Advance = our d_win progress;
            # Threat = their P-weighted best step toward the LOSS BANK (their goal
            # region) if we pass; Prevention = how much our move removes that.
            try:
                import math
                dg = self.vfn.diag(b) if hasattr(self.vfn, "diag") else {}
                M, M_l = dg.get("M"), dg.get("M_l")
                base_T, base_mv, dl0, dl1 = self._threat(b, M_l)
                pw0 = math.exp(-float(self._d_to_win([b])[0]) / M) if M else 0.0
                for r in rows:
                    c = b.copy(stack=False); c.push(chess.Move.from_uci(r["uci"]))
                    adv = (math.exp(-float(self._d_to_win([c])[0]) / M) - pw0) \
                        if M else 0.0
                    T_after, _, _, _ = self._threat(c, M_l)
                    prev = base_T - T_after
                    r["adv"] = round(adv, 3); r["prev"] = round(prev, 3)
                    heavy_p = prev > 0.03 and prev > 1.5 * max(adv, 0)
                    heavy_a = adv > 0.03 and adv > 1.5 * max(prev, 0)
                    if heavy_p and base_mv is not None:
                        # name the prevented plan via the nearest loss exemplar's class
                        sig = ""
                        try:
                            cb = b.copy(stack=False)
                            cb.push(chess.Move.null()); cb.push(base_mv)
                            e = self.fm.embed_F_boards([cb])
                            import numpy as _np
                            d = ((_np.asarray(self.loss.embs) - e[0]) ** 2).sum(1)
                            sig = self.loss.sigs[int(_np.argmin(d))]
                        except Exception:
                            pass
                        nb = b.copy(stack=False); nb.push(chess.Move.null())
                        base_san = nb.san(base_mv) if base_mv in nb.legal_moves \
                            else base_mv.uci()
                        r["role"] = "prevents"
                        r["why"] = (f"stops {base_san} — their best plan heads toward"
                                    f"{' ' + sig if sig else ''} losses"
                                    f" (P(loss) {dl0:.2f}→{dl1:.2f})")
                    elif heavy_a:
                        r["role"] = "advances"
                    elif prev > 0.03 and adv > 0.03:
                        r["role"] = "both"
            except Exception:
                pass
        else:
            rows.sort(key=lambda r: -r["p"])
            rows = rows[:k]
        return {"moves": rows, "turn": "w" if ours else "b", "fen": b.fen(),
                "searched": searched}

    def calculate_start(self, nodes, chunk=64, resume=False):
        """STREAMING calculation (Kaveh: 'a way for calculations to stream in as it's
        calculating'): chunked MCTS on a thread, tree reused across chunks; /calc_state
        serves the running snapshot after every chunk. Stop keeps the TREE; resume=True
        continues it (same position only -- a moved board starts fresh)."""
        if getattr(self, "_calc_busy", False):
            return {"ok": False, "busy": True}
        self._calc_busy = True
        self._calc_cancel = False
        import threading
        rs = getattr(self, "_calc_resume", None)
        if resume and rs and rs["epd"] == self.board.epd():
            # resume = fresh budget ON TOP of the kept tree (works after Stop AND after
            # a natural finish -- 'Calculate more')
            target = rs["used"] + int(nodes)
            rs["nodes"] = target
            self.calc.update(done=False, stopped=False, target=target)
            threading.Thread(target=self._calc_work, args=(target, chunk),
                             kwargs={"resume_state": rs}, daemon=True).start()
            return {"ok": True, "target": target, "resumed": True}
        self._calc_resume = None
        self.calc = {"done": False, "evals": 0, "target": int(nodes),
                     "top": [], "leaves": [], "ideas": [], "plan": None, "goal": None}
        threading.Thread(target=self._calc_work, args=(int(nodes), chunk),
                         daemon=True).start()
        return {"ok": True, "target": int(nodes)}

    def _calc_work(self, nodes, chunk, resume_state=None):
        try:
            self.compute.acquire()                  # serialize field/MPS work (threadsafe)
            if resume_state is not None:            # STOP kept the tree; keep growing it
                b = resume_state["b"]; ps = resume_state["ps"]
                m = resume_state["m"]; root = resume_state["root"]
                used = resume_state["used"]
                self.calc["stage"] = "search"
            else:
                b = self.board.copy(stack=True)
                self.calc["stage"] = "planner"
                if hasattr(self.vfn, "set_anchor"):  # tri-anchor prune: without this the
                    self.vfn.set_anchor(b)           # 86k seeded bank is scanned per eval
                ps = self.planner(b, len(b.move_stack))
                self.ctx["plan"] = ps["plan"]; self.ctx["target_pt"] = ps.get("target_pt")
                self.calc.update(plan=ps["plan"], goal=ps.get("goal"), stage="search")
                m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=chunk, mate_stop=True,
                         pw_c=1.5, root_min_visits=10, value_fn=self.vfn,
                         policy_fn=self.pfn, policy_batch_fn=self.pfnb, batch_leaves=32)
                root, used = None, 0
            snap = dict(self.times); t_run = time.time()
            self._calc_live = m          # /calc_state reads sub-chunk progress off this
            while used < nodes:
                if getattr(self, "_calc_cancel", False):
                    # STOP pressed or human moved: yield at the chunk boundary
                    break
                root = m.run(b.copy(stack=True), reuse_root=root)
                used += int(m.evals_used)
                top, leaves = self._tops_leaves(b, root)
                self.calc.update(evals=used, top=top, leaves=leaves)
                if m.evals_used == 0:        # certified mate in hand -- nothing to add
                    break
            stopped = bool(getattr(self, "_calc_cancel", False)) and used < nodes
            if root is not None:             # keep the tree: Resume (after stop) and
                self._calc_resume = dict(b=b, ps=ps, m=m, root=root, used=used,
                                         epd=b.epd(), nodes=nodes)
            else:                            # 'Calculate more' (after finish) reuse it
                self._calc_resume = None
            try:
                from catspace.research.tools.stats_eval.metrics import observe
                tot = time.time() - t_run
                acc = 0.0
                for st, key in (("prior", "prior_s"), ("embF", "embedF_s"),
                                ("dbank", "dbank_s"), ("dtm", "dtm_s")):
                    v = self.times.get(key, 0) - snap.get(key, 0)
                    observe(st, v); acc += v
                observe("tree", max(tot - acc, 0)); observe("move_total", tot)
            except Exception:
                pass
            if root is not None:
                self.calc.update(self._finish_calc(b, root, ps, used))
            self.calc["stopped"] = stopped
            self.calc["done"] = True
        except Exception as e:                              # noqa: BLE001
            self.calc.update(done=True, err=str(e))
        finally:
            if self.compute.locked():
                self.compute.release()
            self._calc_busy = False

    def calculate(self, nodes):
        """Blocking wrapper (A/B harness + backwards compat): start + wait."""
        r = self.calculate_start(nodes)
        if not r.get("ok"):
            return {"err": "busy"}
        while not self.calc.get("done"):
            time.sleep(0.2)
        return self.calc

    def _finish_calc(self, b, root, ps, evals_used):
        wins, losses, stales = harvest(root)
        self.bank.add(wins); self.loss.add(losses); self.draw.add(stales)
        top, leaves = self._tops_leaves(b, root)
        ideas = []
        def idea(kind, tag, detail):
            self.idea_seq += 1
            ideas.append({"id": f"i{int(time.time())}_{self.idea_seq}", "kind": kind,
                          "tag": tag, "detail": detail})
        if ps["plan"] != "direct":
            idea("plan", ps["plan"] + (f" -> {ps['goal']}" if ps.get("goal") else ""),
                 "planner's active plan (edit the name to what YOU would call it)")
        if top:
            idea("candidate", f"main idea: {top[0]['san']}",
                 f"most-visited move ({top[0]['visits']} visits, v={top[0]['q']})")
        pr = self._prompt()
        for r in pr["reasons"]:
            idea("sense", r, "why the assistant wanted to calculate here")
        return {"top": top, "leaves": leaves, "plan": ps["plan"],
                "goal": ps.get("goal"), "ideas": ideas, "probe": pr["probe"],
                "evals": int(evals_used)}


ARGS = None
STORE: dict = {}


def resolve(handler):
    """sid cookie -> Session (Kaveh: persistent per-user sessions). Cookie-less
    clients (harnesses, curl) share the 'default' session so existing tooling keeps
    working; browsers get a fresh sid on the page load."""
    import uuid
    sid = None
    for part in (handler.headers.get("Cookie", "") or "").split(";"):
        part = part.strip()
        if part.startswith("sid="):
            sid = part[4:]
    new_cookie = None
    if not sid:
        if handler.path == "/":
            sid = uuid.uuid4().hex[:16]
            new_cookie = sid
        else:
            sid = "default"
    if sid not in STORE:
        STORE[sid] = Session(ARGS, sid=sid)
    S = STORE[sid]
    S.last_seen = time.time()
    return S, new_cookie


def global_reloader():
    """One tick for all sessions: shared bank sync, dtm auto-swap per session,
    idle-session eviction (>3h, never 'default')."""
    import glob as _g
    import re as _re
    while True:
        time.sleep(45)
        try:
            if SHARED:
                for bk in (SHARED["bank"], SHARED["loss"], SHARED["draw"]):
                    bk.sync()
        except Exception:
            pass
        try:
            cands = _g.glob(str(ROOT / "data/derived/sep/dtm_tok_r*.pt"))
            best = max(cands, key=lambda q: int(_re.search(r"r(\d+)\.pt", q).group(1))) \
                if cands else None
            now = time.time()
            for sid in list(STORE):
                S = STORE[sid]
                if sid != "default" and now - S.last_seen > 3 * 3600:
                    STORE.pop(sid, None)
                    try:
                        if S.opp is not None:
                            S.opp.quit()
                    except Exception:
                        pass
                    print(f"[assistant] session {sid} evicted (idle)", flush=True)
                    continue
                if best and not S.pinned and best != S._lm:
                    S.vfn = make_boot_value(S.fm, S.bank, S.times, S.loss,
                                            dtm_ckpt=best, draw_bank=S.draw,
                                            game_ctx=S.ctx)
                    S._lm = best
                    S.version = S._version_of(best)
                    print(f"[assistant] {sid} MODEL SWAPPED -> {S.version}", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[assistant] reloader: {e}", flush=True)
PAGE = (ROOT / "catspace/viz/templates/assistant.html")


USAGE = ROOT / "artifacts/experiments/usage.jsonl"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _usage(self, path, code, ms):
        try:
            with open(USAGE, "a") as f:
                f.write(json.dumps({"ts": time.time(), "path": path, "code": code,
                                    "ms": round(ms, 1)}) + "\n")
            from catspace.research.tools.stats_eval.metrics import count, observe
            count(path); observe("http", ms / 1000.0)
        except Exception:
            pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        if getattr(self, "_newck", None):
            self.send_header("Set-Cookie",
                             f"sid={self._newck}; Path=/; Max-Age=2592000; SameSite=Lax")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.S, self._newck = resolve(self)
        t = time.time()
        try:
            self._get()
        finally:
            self._usage(self.path, 200, (time.time() - t) * 1000)

    def _get(self):
        if self.path == "/metrics":
            from catspace.research.tools.stats_eval.metrics import latest
            self._send(200, latest(), "text/plain; version=0.0.4")
        elif self.path == "/health":
            self._send(200, {"ok": True, "banks": {"win": len(self.S.bank),
                                                   "loss": len(self.S.loss),
                                                   "draw": len(self.S.draw)}})
        elif self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.read_bytes(), "text/html")
        elif self.path.startswith("/assets/"):
            f = ASSETS / self.path[len("/assets/"):]
            if f.exists():
                ct = "text/css" if f.suffix == ".css" else "application/javascript"
                self._send(200, f.read_bytes(), ct)
            else:
                self._send(404, {"err": "no asset"})
        elif self.path == "/calc_state":
            st = dict(getattr(self.S, "calc", {}) or {})
            if st and not st.get("done"):
                live = getattr(self.S, "_calc_live", None)
                st["evals"] = min(st.get("evals", 0) + int(getattr(live, "evals_used", 0) or 0),
                                  st.get("target", 10**9))
            self._send(200, st)
        elif self.path == "/ab":
            f = ROOT / "catspace/viz/templates/ab.html"
            self._send(200, f.read_bytes(), "text/html")
        elif self.path == "/atlas":
            f = ROOT / "catspace/viz/templates/atlas.html"
            self._send(200, f.read_bytes(), "text/html")
        elif self.path == "/atlas_data":
            with self.S.compute:
                data = self.S.atlas_data()
            self._send(200, data)
        elif self.path == "/cone":
            try:
                with self.S.compute:
                    data = self.S.cone()
                self._send(200, data)
            except Exception as e:                          # noqa: BLE001
                self._send(200, {"err": str(e), "nodes": [], "river": []})
        elif self.path == "/energy":
            try:
                with self.S.compute:
                    data = self.S.energy()
                self._send(200, data)
            except Exception as e:                          # noqa: BLE001
                self._send(200, {"err": str(e), "moves": [], "line": []})
        elif self.path == "/atlas_selected":
            self._send(200, getattr(self.S, "_atlas_sel", None) or {})
        elif self.path == "/atlas_plan":
            try:
                with self.S.compute:
                    data = self.S.atlas_plan()
                self._send(200, data)
            except Exception as e:                          # noqa: BLE001
                self._send(200, {"err": str(e)})
        elif self.path == "/ab_state":
            f = ROOT / "artifacts/experiments/ab_live.json"
            self._send(200, f.read_bytes() if f.exists() else b"{}", "application/json")
        elif self.path == "/state":
            b = self.S.board
            dests = {}
            for mv in b.legal_moves:
                dests.setdefault(chess.square_name(mv.from_square), []).append(
                    chess.square_name(mv.to_square))
            self._send(200, {"fen": b.fen(), "turn": "w" if b.turn else "b",
                             "dests": dests, "over": b.is_game_over(claim_draw=True),
                             "result": b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else None,
                             "version": self.S.version})
        else:
            self._send(404, {"err": "?"})

    def do_POST(self):
        self.S, self._newck = resolve(self)
        t = time.time()
        try:
            self._post()
        finally:
            self._usage(self.path, 200, (time.time() - t) * 1000)

    def _post(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/new":
            w = req.get("opponent", self.S.args.opponent)
            self.S.new_game(w)
            self._send(200, {"ok": True, "fen": self.S.board.fen()})
        elif self.path == "/human_move":
            try:
                self.S._calc_cancel = True     # any running calc is stale + steals compute
                mv = chess.Move.from_uci(req["uci"])
                if mv not in self.S.board.legal_moves:
                    raise ValueError("illegal")
                self.S.board.push(mv)
                self.S.ctx["hist"][self.S.board.epd()] += 1
                reply = None
                if not self.S.board.is_game_over(claim_draw=True) and self.S.opp is not None:
                    r = self.S.opp.play(self.S.board, chess.engine.Limit(nodes=1))
                    reply = self.S.board.san(r.move)
                    self.S.board.push(r.move)
                    self.S.ctx["hist"][self.S.board.epd()] += 1
                self._send(200, {"ok": True, "fen": self.S.board.fen(), "reply": reply,
                                 "assistant": self.S._prompt()})
            except Exception as e:                          # noqa: BLE001
                self._send(400, {"err": str(e)})
        elif self.path == "/load":          # paste FEN or PGN (PGN keeps move HISTORY --
            try:                            # repetition/clock state survive the load)
                text = req.get("text", "").strip()
                self.S._calc_cancel = True
                b = None
                try:
                    b = chess.Board(text)
                except Exception:
                    import io
                    import chess.pgn as _pgn
                    g = _pgn.read_game(io.StringIO(text))
                    if g is None or g.errors:
                        raise ValueError("neither valid FEN nor PGN")
                    b = g.board()
                    for mv in g.mainline_moves():
                        b.push(mv)
                from collections import Counter as _C
                hist = _C()
                bb = b.copy(stack=True)
                keys = [bb.epd()]
                while bb.move_stack:
                    bb.pop(); keys.append(bb.epd())
                for k2 in keys:
                    hist[k2] += 1
                self.S.board = b
                self.S.ctx["hist"] = hist
                self.S.ctx["plan"] = "direct"; self.S.ctx["target_pt"] = None
                reply = None
                if b.turn == chess.BLACK and not b.is_game_over(claim_draw=True) \
                        and self.S.opp is not None:
                    r = self.S.opp.play(b, chess.engine.Limit(nodes=1))
                    reply = b.san(r.move)
                    b.push(r.move)
                    self.S.ctx["hist"][b.epd()] += 1
                self._send(200, {"ok": True, "fen": b.fen(),
                                 "plies": len(b.move_stack), "reply": reply})
            except Exception as e:                          # noqa: BLE001
                self._send(400, {"err": str(e)})
        elif self.path == "/set_fen":       # A/B harness: probe an arbitrary position
            try:
                from collections import Counter as _C
                self.S.board = chess.Board(req["fen"])
                self.S.ctx["hist"] = _C({self.S.board.epd(): 1})
                self.S.ctx["plan"] = "direct"
                self._send(200, {"ok": True, "fen": self.S.board.fen()})
            except Exception as e:                          # noqa: BLE001
                self._send(400, {"err": str(e)})
        elif self.path == "/calculate":
            self._send(200, self.S.calculate(int(req.get("nodes", 1500))))
        elif self.path == "/calculate_start":
            self._send(200, self.S.calculate_start(int(req.get("nodes", 1500)),
                                                resume=bool(req.get("resume"))))
        elif self.path == "/calculate_stop":
            self.S._calc_cancel = True
            self._send(200, {"ok": True})
        elif self.path == "/explore":
            try:
                with self.S.compute:
                    data = self.S.explore(req.get("ucis", []),
                                       k=min(int(req.get("k", 6)), 24))
                self._send(200, data)
            except Exception as e:                          # noqa: BLE001
                self._send(200, {"err": str(e), "moves": []})
        elif self.path == "/atlas_fens":    # hover-peek: positions behind atlas indices
            try:
                a = getattr(self.S, "_atlas", None) or {}
                epds, kinds = a.get("epds", []), a.get("kinds", [])
                out = [[epds[i], kinds[i]] for i in req.get("idx", [])[:9]
                       if 0 <= i < len(epds) and epds[i]]
                self._send(200, {"pos": out})
            except Exception as e:                          # noqa: BLE001
                self._send(200, {"pos": [], "err": str(e)})
        elif self.path == "/atlas_select":
            try:
                with self.S.compute:
                    data = self.S.atlas_select(req["fens"])
                self._send(200, data)
            except Exception as e:                          # noqa: BLE001
                self._send(400, {"err": str(e)})
        elif self.path == "/tag":
            with open(TAGS, "a") as f:
                f.write(json.dumps({"id": req.get("id"), "tag": req.get("tag"),
                                    "kind": req.get("kind"), "fen": self.S.board.fen(),
                                    "ts": time.time()}) + "\n")
            self._send(200, {"ok": True})
        else:
            self._send(404, {"err": "?"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--host", default="127.0.0.1")
    # Defaults resolve through the path registry, so the server behaves the same whatever
    # directory it is launched from (and inside the container). The field/energy defaults
    # follow the bootstrap_mate pointer files when a training run has published one.
    ap.add_argument("--field", default=bootstrap_mate_config.field_checkpoint())
    ap.add_argument("--energy", default=bootstrap_mate_config.opponent_energy_checkpoint())
    ap.add_argument("--last-mile", default=str(paths.sep_dir() / "dtm_cnn_v2.pt"))
    ap.add_argument("--banks-prefix", default=str(paths.experiments_dir() / "assistant"))
    ap.add_argument("--opponent",
                    default=str(paths.engines_dir() / "maia" / "maia-1200.pb.gz"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pin-model", default="",
                    help="A/B endpoint mode: serve exactly this dtm ckpt, never auto-swap "
                         "(bank sync stays live). Run a second instance on another --port "
                         "as the challenger.")
    args = ap.parse_args()
    global ARGS
    ARGS = args
    STORE["default"] = Session(args, sid="default")   # warm the shared heavies at boot
    import threading
    threading.Thread(target=global_reloader, daemon=True).start()
    print(f"assistant on http://localhost:{args.port}  opponent={args.opponent} "
          f"(per-user sessions via sid cookie)", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
