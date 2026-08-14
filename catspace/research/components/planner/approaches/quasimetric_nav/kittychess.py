#!/usr/bin/env python
"""kittychess.py -- THE most basic quasimetric-navigation engine (Kaveh 2026-08-08).

Move choice is THREAT-FIRST navigation on the outcome field, exactly as specified:
  'it's about holding back the draw and holding back the loss and going towards the win.'

For every legal move, embed the child and read d(child -> WIN/DRAW/LOSS poles), POV-flipped
(the child's mover is the opponent: our win is their loss). With d_bad = min(d_draw, d_loss):

  primary   maximise  d_bad - d_win      (push the nearest threat out AND pull the win in;
                                          a move that delays the draw but delays our win MORE
                                          worsens the margin and is rejected)
  tie-break maximise  d_bad              (among margin-equal moves, buy time vs the threat)
  <=5 pieces: Syzygy lookup outright     (the hybrid oracle -- no field consulted)

No search. One field readout per legal move. This is deliberately the dumbest possible planner:
its match results are the honest floor of what the geometry alone currently buys.
"""
from __future__ import annotations

import argparse
import random

import chess
import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize, move_ids
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB


class KittyChess:
    def __init__(self, ckpt, device="mps", cond_elo=None, use_tb=True, nav="cascade",
                 gate=0.20, head_order=False, half=False):
        self.nav = nav          # "db" threat-first | "ab" A-steer+B-gate | "cascade" (2026-08-11)
        self.qext = True                          # quiescence at the batched horizon (#3)
        self.head_order = head_order            # move-head orders internal search nodes
        self.gate = gate    # cascade gate CALIBRATED 2026-08-11: accuracy-vs-gap is a
                            # VALLEY (89% below 0.01, 77-80% in 0.01-0.2, 89% above 0.2) --
                            # the old 0.05 trusted the head exactly where it is wrongest
        self.net, pay = load_net(ckpt, device)
        self.cfg = pay["cfg"]
        self.device = device
        self.cond_elo = cond_elo
        pn = self.cfg["pole_names"]
        self.poles = self.net.poles.poles.detach().float()
        self.pi = {n: pn.index(n) for n in ("WIN", "DRAW", "LOSS")}
        # basin-pov 'white' (2026-08-08): poles are colour-fixed (WIN = white wins), so
        # readouts select the pole by colour, not by whose turn it is at the queried node.
        self.white_pov = (self.cfg.get("train_args") or {}).get("basin_pov") == "white"
        # terminal-exemplar committor (2026-08-08): contrastive-mode ckpts train NO poles --
        # net.poles holds untouched init buffers (the frozen 45/65/139 readout). The validated
        # readout is median distance to real terminal boards (export_exemplars.py sidecar,
        # white-POV classes W/D/L). When the sidecar exists it REPLACES the pole readout.
        if half and device == "mps":
            try:
                self.net = self.net.half()
                self.half = True
            except Exception:
                self.half = False
        else:
            self.half = False
        exp = (ckpt[:-3] if ckpt.endswith(".pt") else ckpt) + "_exemplars.pt"
        self.ex = None
        import os as _os
        if _os.path.exists(exp):
            pay_ex = torch.load(exp, map_location=device, weights_only=False)
            self.ex = {k: pay_ex[k].to(device) for k in ("W", "D", "L")}
        self.tb = TB() if use_tb else None
        self._ckpt_base = ckpt[:-3] if ckpt.endswith(".pt") else ckpt
        self._mcache = {}                        # fen -> leaf value cache
        # EMBEDDING CACHE (Kaveh 2026-08-13: "cache all of the embeddings and evaluations
        # so we don't have to keep doing it again"). Scoring a move's direction embeds the
        # CHILD -- which is exactly the node search expands next, so with this cache the
        # directional readout is PREFETCH, not overhead. Keyed like the value cache
        # (zobrist under FB, fen otherwise). Bounded; z rows live on device.
        self._zcache = {}
        # CONCEPT-MEDIATED EVALUATION (Kaveh 2026-08-11: 'I don't want it based on some
        # internal margin'): values flow through the concept bottleneck -- quantize phi,
        # DECODE the codes back to the six outputs, score from those. Faithful by
        # construction: every value is a function of the named concept profile.
        self.cvq = None
        try:
            from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
                ConceptVQ)
            _pv = torch.load((ckpt[:-3] if ckpt.endswith(".pt") else ckpt) + "_vq.pt",
                             map_location=device, weights_only=False)
            _m = ConceptVQ(d_in=_pv["d_in"], heads=_pv["heads"], codes=_pv["codes"]).to(device)
            _m.load_state_dict(_pv["state_dict"]); _m.eval()
            self.cvq = _m
            self.cvq_mu = _pv["mu"].to(device)
            self.cvq_sd = _pv["sd"].to(device)
        except Exception:
            # JQT-NATIVE concept evaluator (2026-08-12): a jointly-trained run carries its
            # quantizer in the _jqt sidecar; consume THAT instead of requiring a post-hoc
            # ConceptVQ refit (always the most complete artifact available).
            try:
                from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
                    JQTModule)
                import re as _re, os as _osj
                _cands = [self._ckpt_base + "_jqt.pt",
                          _re.sub(r"_(latest|step\d+)$", "", self._ckpt_base) + "_jqt.pt"]
                _jp = next(c for c in _cands if _osj.path.exists(c))
                _pj = torch.load(_jp, map_location=device, weights_only=False)
                _jm = JQTModule(d_model=_pj["d_in"], heads=_pj["heads"],
                                codes=_pj["codes"], d=_pj["d"],
                                square_codes=_pj.get("square_codes", 0),
                                piece_codes=_pj.get("piece_codes", 0)).to(device)
                _jm.load_state_dict(_pj["state_dict"], strict=False); _jm.eval()

                class _JQTAsCVQ(torch.nn.Module):
                    def __init__(self, jm):
                        super().__init__(); self.jm = jm
                    def forward(self, phi):
                        _h, zq, ids, _vl = self.jm.quantize(phi)
                        return self.jm.dec(zq), ids, 0.0

                self.cvq = _JQTAsCVQ(_jm)
                self.cvq_mu = _pj["y_mu"].to(device)
                self.cvq_sd = _pj["y_sd"].to(device)
            except Exception:
                pass
        if getattr(self.net, "split_head", False):
            self.dist = self.net.dB
        elif getattr(self.net, "dual", False):
            self.dist = self.net.qhead.d_base
        else:
            self.dist = self.net.iqe

    def _exd(self, z):
        """(len(z), 3) median distance to the W/D/L exemplar sets -- white-POV columns."""
        cols = []
        for k in ("W", "D", "L"):
            E = self.ex[k]
            dd = torch.stack([self.dist(z, E[e].expand(len(z), -1)) for e in range(len(E))], 1)
            cols.append(dd.median(1).values)
        return torch.stack(cols, 1)

    def _embed(self, toks, globs):
        tok_t = torch.from_numpy(np.array(toks).astype(np.int64)).to(self.device)
        glob_t = torch.from_numpy(np.array(globs).astype(np.float32)).to(self.device)
        if getattr(self, "half", False):
            glob_t = glob_t.half()
        if self.cond_elo is not None and getattr(self.net, "dual", False):
            cval = (self.cond_elo - 1500.0) / 500.0
            cond = torch.full((len(toks), self.net.qhead.proj_delta.in_features
                               - self.net.qhead.proj_base.in_features), cval, device=self.device)
            return self.net.encode_dual(tok_t, glob_t, cond)[1]
        return self.net.encode_q(tok_t, glob_t)

    def _tb_move(self, board):
        best, key = None, None
        for mv in board.legal_moves:
            board.push(mv)
            try:
                w, dz = self.tb.wdl_dtz(board)
            except Exception:
                w = None
            board.pop()
            if w is None:
                return None                      # any unprobeable child -> fall back to field
            # opponent POV: lower w better for us; among our wins prefer fast (small |dtz|),
            # among our losses prefer slow
            k = (w, (abs(dz) if w < 0 else -abs(dz)) if dz is not None else 0)
            if key is None or k < key:
                key, best = k, mv
        return best

    MATE = 1e6

    @staticmethod
    def margin_e(dW, dD, dL, E, tau_c=0.03):
        """E-CONDITIONAL margin (Kaveh 2026-08-12: 'the margin should differ if you're
        ahead or behind'). Win always better, loss always worse, the DRAW switches from
        threat (ahead) to salvation (behind) -- as a CONTINUOUS blend, no dead zone:
        the first version zeroed the draw term exactly in the opening band (E~0.5-0.55)
        and the engine bongclouded (the draw distance was the signal shaping quiet play).
            s = sigmoid((E - 0.5)/tau_c)
            margin = s*[min(dD,dL) - dW] + (1-s)*[dL - min(dD,dW)]"""
        # PIECEWISE (second iteration: the symmetric blend leaked salvation weight into
        # the opening band E~0.52 and the engine lunged wing pawns): AT OR ABOVE equality
        # the classic threat form applies UNCHANGED; the draw-as-salvation blend engages
        # only when genuinely behind, ramping in over ~tau_c below 0.5.
        import math
        if E >= 0.5:
            return min(dD, dL) - dW
        sig = 1.0 / (1.0 + math.exp(-(0.5 - E) / tau_c))    # 0.5 at E=0.5 -> 1 when far behind
        w_salv = 2.0 * (sig - 0.5)
        return (1.0 - w_salv) * (min(dD, dL) - dW) + w_salv * (dL - min(dD, dW))

    def rank_by_child_E(self, board, rows, top=10):
        """RANK BY E (Kaveh 2026-08-12: 'results should be ranked by E'; the deep-backup
        probe showed minimax over near-flat noisy leaves INVERTS the one-ply truth -- Ng8
        ranked above e6 while the calibrated one-ply E ordered them correctly). Mate/TB
        proofs keep absolute rank; the rest re-rank by the CHILD'S calibrated E (one batched
        readout), mover POV. The searched value stays as the tiebreak."""
        import chess as _ch
        if not rows:
            return rows
        head = [r for r in rows[:top]]
        rest = rows[top:]
        boards = []
        for r in head:
            b2 = board.copy(stack=False)
            b2.push(r["mv"])
            boards.append(b2)
        probs = []
        for b2 in boards:
            pr, _ = self.wdl(b2)
            probs.append(pr)
        wtm = board.turn == _ch.WHITE
        for r, pr in zip(head, probs):
            e_w = pr[0] + 0.5 * pr[1]
            r["child_E"] = e_w if wtm else 1.0 - e_w
        head.sort(key=lambda r: (0, -r["value"], 0) if abs(r["value"]) >= 5e5
                  else (1, -r["child_E"], -r["value"]))
        return head + rest

    def exit_readout(self, board, tau=5.0):
        """THE position readout contract (Kaveh 2026-08-12): 'every position has 3
        probabilities and 3 distances. expected score should combine the probability,
        chosen exit should show draw win or lose, and distance to it in plies.'
        -> {"E": white-POV expected points, "exit": "white"|"draw"|"black",
            "plies": dA to the chosen exit on the LENGTH ruler (calibrated in plies),
            "probs": [pW,pD,pB], "dA": [3], "dB": [3]}  (exact at terminals/TB)."""
        import chess as _ch
        probs, dB = self.wdl(board, tau=tau)
        out = {"probs": probs, "E": float(probs[0] + 0.5 * probs[1]),
               "exit": ("white", "draw", "black")[int(np.argmax(probs))],
               "dB": dB, "dA": None, "plies": None}
        if dB is None:                                  # terminal or TB-exact
            if self.tb is not None and not board.is_game_over(claim_draw=True)                     and len(board.piece_map()) <= 5:
                try:
                    _w, dz = self.tb.wdl_dtz(board)
                    out["plies"] = abs(dz) if dz is not None else 0
                except Exception:
                    pass
            else:
                out["plies"] = 0
            return out
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            tokenize as _tkz)
        tk, gl = _tkz(board)
        with torch.no_grad():
            z = self._embed([tk], [gl])
            if getattr(self.net, "split_head", False):
                P3 = self.poles[[self.pi[k] for k in ("WIN", "DRAW", "LOSS")]].to(self.device)
                dA = [float(self.net.dA(z, P3[[k]].expand(1, -1)).float().cpu())
                      for k in range(3)]
                if not self.white_pov:
                    dA = dA[::-1] if not (board.turn == _ch.WHITE) else dA
                out["dA"] = dA
                out["plies"] = round(dA[int(np.argmax(probs))], 1)
        return out

    def margins(self, boards):
        """mover-POV threat-first margin for each board, one batched forward."""
        import numpy as np, torch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize, move_ids
        toks, globs = zip(*(tokenize(b) for b in boards))
        with torch.no_grad():
            z = self._embed(list(toks), list(globs))
            if self.ex is not None:
                M = self._exd(z).float().cpu().numpy()   # white-POV cols: W / D / L
                import numpy as _np
                pr = _np.exp(-M / 5.0)
                pr = pr / pr.sum(1, keepdims=True)
                out = []
                for i, b in enumerate(boards):
                    mw, ml = (0, 2) if b.turn else (2, 0)
                    Em = float(pr[i, mw] + 0.5 * pr[i, 1])
                    out.append(float(self.margin_e(M[i, mw], M[i, 1], M[i, ml], Em)))
                return out
            D = {n: self.dist(z, self.poles[[k]].expand(len(z), -1).to(self.device))
                 .float().cpu().numpy() for n, k in self.pi.items()}
        if self.white_pov:
            out = []
            for i, b in enumerate(boards):
                mw, ml = ("WIN", "LOSS") if b.turn else ("LOSS", "WIN")
                out.append(float(min(D["DRAW"][i], D[ml][i]) - D[mw][i]))
            return out
        # mover-POV poles: my win = d->LOSS pole, my loss = d->WIN, draw = d->DRAW
        return [float(min(D["DRAW"][i], D["WIN"][i]) - D["LOSS"][i]) for i in range(len(boards))]

    def wdl(self, board, tau=5.0):
        """WHITE-POV [P(white wins), P(draw), P(black wins)] -- softmax over the three
        pole distances.

        This IS the eval bar (Kaveh 2026-08-08): no scalar, three distances softmaxed.
        Exact at terminals and in tablebase range; field softmax elsewhere. tau matches
        the basin-CE training temperature. Returns (probs, dists) with dists=None when
        the answer is exact.

        KNOWN LIMIT (2026-08-08 diagnostic): every current checkpoint trained the basin
        CE at w=10 where the force audit demanded ~1000x, so d(z->pole) is position-
        invariant (~0.1 over startpos..bare-kings) and the field part of this bar is
        flat until a re-based checkpoint lands."""
        wtm = board.turn == chess.WHITE
        if board.is_checkmate():                        # mover is mated
            return ([0.0, 0.0, 1.0] if wtm else [1.0, 0.0, 0.0]), None
        if board.is_game_over(claim_draw=True):
            return [0.0, 1.0, 0.0], None
        if self.tb is not None and len(board.piece_map()) <= 5:
            try:
                w, _ = self.tb.wdl_dtz(board)           # mover POV
                if w is not None:
                    if w == 0:
                        return [0.0, 1.0, 0.0], None
                    mover_wins = w > 0
                    return ([1.0, 0.0, 0.0] if mover_wins == wtm
                            else [0.0, 0.0, 1.0]), None
            except Exception:
                pass
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize, move_ids
        toks, globs = tokenize(board)
        with torch.no_grad():
            z = self._embed([toks], [globs])
            if self.ex is not None:                     # committor from geometry: white-POV cols
                d = self._exd(z)[0].float().cpu().numpy()
            else:
                D = {n: float(self.dist(z, self.poles[[k]].to(self.device)).float().cpu())
                     for n, k in self.pi.items()}
                if self.white_pov:                      # colour-fixed poles: direct readout
                    d = np.array([D["WIN"], D["DRAW"], D["LOSS"]])
                else:                                   # mover-POV reachability reading, then flip
                    d = np.array([D["LOSS"], D["DRAW"], D["WIN"]])
                    if not wtm:
                        d = d[::-1].copy()
        e = np.exp(-(d - d.min()) / tau)
        return [float(x) for x in (e / e.sum())], [float(x) for x in d]

    def _terminal_value(self, board):
        """exact value at game end / tablebase, mover POV, on the MATE scale."""
        if board.is_checkmate():
            return -self.MATE                       # mover is mated
        if board.is_game_over(claim_draw=True):
            return 0.0                              # stalemate/draw rules
        if self.tb is not None and len(board.piece_map()) <= 5:
            try:
                w, dz = self.tb.wdl_dtz(board)
                if w is not None:
                    if w == 0:
                        return 0.0
                    sign = 1 if w > 0 else -1
                    return sign * (self.MATE / 2 - abs(dz or 0))   # prefer faster wins
            except Exception:
                pass
        return None

    def _tb_probe(self, fen):
        """tablebase probe by fen for the fast (Rust-board) path; None = not covered."""
        if self.tb is None:
            return None
        try:
            w, dz = self.tb.wdl_dtz(chess.Board(fen))
            if w is None:
                return None
            if w == 0:
                return 0.0
            return (1 if w > 0 else -1) * (self.MATE / 2 - abs(dz or 0))
        except Exception:
            return None

    def turbulence(self, board):
        """QUIESCENCE FROM OUR OWN EVALS (Kaveh 2026-08-11): tau = |E_static(s) - best
        child E (mover-POV)|. Quiet <=> the evaluation agrees with itself one ply deeper;
        the Nxe5 mid-exchange reads tau ~ 0.35. One batched forward. Returns (tau, E_static,
        E_resolved) white-POV, or None at terminals."""
        if board.is_game_over(claim_draw=True) or not self.white_pov:
            return None
        moves = list(board.legal_moves)
        if not moves:
            return None
        toks, globs = [tokenize(board)], []
        globs = [toks[0][1]]; toks = [toks[0][0]]
        for mv in moves:
            board.push(mv)
            tk, gl = tokenize(board)
            toks.append(tk); globs.append(gl)
            board.pop()
        P3 = self.poles[[self.pi["WIN"], self.pi["DRAW"], self.pi["LOSS"]]].to(self.device)
        with torch.no_grad():
            z = self._embed(toks, globs)
            DBc = torch.stack([self.dist(z, P3[[k]].expand(len(z), -1))
                               for k in range(3)], 1)
            pr = torch.softmax(-DBc / 5.0, 1).float().cpu().numpy()
        E = pr[:, 0] + 0.5 * pr[:, 1]                    # white-POV, row 0 = the position
        Em = E[1:] if board.turn else 1.0 - E[1:]        # children, mover-POV
        e_res = float(E[1:][int(np.argmax(Em))])         # best child's white-POV E
        return abs(e_res - float(E[0])), float(E[0]), e_res

    def bellman_residual(self, board):
        """|d(s -> mover's win pole) - 1 - min over moves d(child -> same pole)| in the field.

        The field's local self-consistency at this position. ~0 = the geometry's gradient is a
        valid plan here (trust it, search shallow); large = the map contradicts itself here
        (deliberate). White-POV pole selection; None for terminals or non-white-pov ckpts."""
        if not self.white_pov or board.is_game_over(claim_draw=True):
            return None
        moves = list(board.legal_moves)
        if not moves:
            return None
        toks, globs = [tokenize(board)], []
        globs = [toks[0][1]]; toks = [toks[0][0]]
        for mv in moves:
            board.push(mv)
            tk, gl = tokenize(board)
            toks.append(tk); globs.append(gl)
            board.pop()
        pole = self.poles[[self.pi["WIN" if board.turn else "LOSS"]]].to(self.device)
        with torch.no_grad():
            z = self._embed(toks, globs)
            d = self.dist(z, pole.expand(len(z), -1)).float().cpu().numpy()
        return float(d[0] - 1.0 - d[1:].min())

    def cascade_rank(self, board):
        """full CASCADE ordering over legal moves (Kaveh 2026-08-11: 'the ranking should be
        based on the cascade'): returns (moves, order_indices, E_list). Gate passes -> order
        by expected points; else standing-aware length ordering."""
        moves = list(board.legal_moves)
        if not moves:
            return [], [], []
        toks, globs = [], []
        for mv in moves:
            board.push(mv)
            tk, gl = tokenize(board)
            toks.append(tk); globs.append(gl)
            board.pop()
        P3 = self.poles[[self.pi["WIN"], self.pi["DRAW"], self.pi["LOSS"]]].to(self.device)
        with torch.no_grad():
            z = self._embed(toks, globs)
            DBc = torch.stack([self.dist(z, P3[[k]].expand(len(z), -1)) for k in range(3)], 1)
            pr = torch.softmax(-DBc / 5.0, 1).float().cpu().numpy()
            DAc = torch.stack([self.net.dA(z, P3[[k]].expand(len(z), -1))
                               for k in range(3)], 1).float().cpu().numpy()
        E = pr[:, 0] + 0.5 * pr[:, 1]
        Em = E if board.turn else 1.0 - E                    # mover-POV
        oE = np.argsort(-Em)
        if len(Em) > 1 and Em[oE[0]] - Em[oE[1]] >= self.gate:
            return moves, list(oE), list(E)
        aw, ad, al = ((DAc[:, 0], DAc[:, 1], DAc[:, 2]) if board.turn
                      else (DAc[:, 2], DAc[:, 1], DAc[:, 0]))
        if float(Em.mean()) <= 0.45:
            tgt = aw if aw.min() <= ad.min() else ad
            return moves, list(np.argsort(tgt)), list(E)
        return moves, list(np.argsort(-(np.minimum(ad, al) - aw))), list(E)

    def line_coherence(self, board, pv, max_plies=6):
        """FORCINGNESS along a line (Kaveh 2026-08-11): walk the PV, read dA to the favored
        side's pole at every position; forced = the distance steps down ~1 ply per ply.
        Returns (per_ply_drop, monotone_frac, favored 'w'/'b') or None."""
        seq = [board.copy()]
        b = board.copy()
        for mv in pv[:max_plies]:
            b.push(mv)
            seq.append(b.copy())
            if b.is_game_over(claim_draw=True):
                break
        if len(seq) < 3:
            return None
        toks, globs = zip(*(tokenize(x) for x in seq))
        with torch.no_grad():
            z = self._embed(list(toks), list(globs))
            dW = self.net.dA(z, self.poles[[self.pi["WIN"]]].expand(len(z), -1)
                             .to(self.device)).float().cpu().numpy()
            dB_ = self.net.dA(z, self.poles[[self.pi["LOSS"]]].expand(len(z), -1)
                              .to(self.device)).float().cpu().numpy()
        fav = "w" if dW[0] <= dB_[0] else "b"
        d = dW if fav == "w" else dB_
        steps = np.diff(d)
        drop = float(-steps.mean())
        mono = float((steps < 0).mean())
        return drop, mono, fav

    def search_batched(self, board, depth=3, stop=None):
        """LEVEL-BATCHED fixed-depth minimax (2026-08-11 horizon autopsy: the engine loses on
        depth; the recursive search paid one tiny MPS forward PER NODE. Batching all frontier
        leaves into large forwards buys depth 3 inside the old depth-2 budget). Full-width --
        at these sizes one big batch beats alpha-beta. Returns rows like search()."""
        root_moves = list(board.legal_moves)
        if not root_moves:
            return []
        # ---- expansion: levels[lv] = list of (board, parent_index, move); kids bookkeeping
        levels = [[(board.copy(stack=False), -1, None)]]
        kids = []                                        # kids[lv][parent_i] -> child indices
        for d in range(depth):
            frontier = levels[-1]
            nxt = []
            k = [[] for _ in frontier]
            for pi, (b, _, _) in enumerate(frontier):
                if b.is_game_over(claim_draw=True):
                    continue
                for mv in b.legal_moves:
                    b.push(mv)
                    k[pi].append(len(nxt))
                    nxt.append((b.copy(stack=False), pi, mv))
                    b.pop()
                if stop is not None and stop():
                    raise KittyChess.SearchStop
            if not nxt:
                break
            kids.append(k)
            levels.append(nxt)
        leaf_lv = len(levels) - 1
        # ---- evaluate the frontier in large chunks (terminals exact)
        leaves = levels[leaf_lv]
        lv_vals = [None] * len(leaves)
        todo, toks, globs = [], [], []
        qx_parent, qx_boards = [], []            # speedup #3: capture-resolution frontier
        for i, (b, _, _) in enumerate(leaves):
            tv = self._terminal_value(b)
            if tv is not None:
                lv_vals[i] = tv
                continue
            fen = b.fen()
            hit = self._mcache.get(fen)
            if hit is not None:
                lv_vals[i] = hit
            else:
                todo.append(i)
                tk, gl = tokenize(b)
                toks.append(tk); globs.append(gl)
            if self.qext:
                for mv in b.legal_moves:
                    if b.is_capture(mv):
                        b.push(mv)
                        qx_parent.append(i); qx_boards.append(b.copy(stack=False))
                        b.pop()
        P3 = self.poles[[self.pi["WIN"], self.pi["DRAW"], self.pi["LOSS"]]].to(self.device)
        for a in range(0, len(todo), 4096):
            if stop is not None and stop():
                raise KittyChess.SearchStop
            with torch.no_grad():
                z = self._embed(toks[a:a+4096], globs[a:a+4096])
                D = [self.dist(z, P3[[k2]].expand(len(z), -1)).float().cpu().numpy()
                     for k2 in range(3)]
                DA = ([self.net.dA(z, P3[[k2]].expand(len(z), -1)).float().cpu().numpy()
                       for k2 in (0, 2)] if getattr(self.net, "split_head", False) else None)
            for j in range(len(D[0])):
                b = leaves[todo[a + j]][0]
                if self.white_pov:
                    iw, il = (0, 2) if b.turn else (2, 0)
                else:
                    iw, il = 2, 0
                import numpy as _np9
                _e = _np9.exp(-_np9.array([D[0][j], D[1][j], D[2][j]]) / 5.0)
                _e = _e / _e.sum()
                _Em = float(_e[iw] + 0.5 * _e[1])
                # P primary, DISTANCE secondary (Kaveh: 'maximizing probability then by
                # distance'): plies to the MOVER'S OWN WIN on the length ruler, tiny
                # weight (<=4 units) -- only decides inside genuine P-ties, and always
                # prefers progress over hopping the pony back home (3...Ng8).
                m = 1000.0 * _Em - (0.1 * min(float(DA[0 if iw == 0 else 1][j]), 40.0)
                                    if DA is not None else 0.0)
                lv_vals[todo[a + j]] = float(m)
                self._mcache[b.fen()] = float(m)
        if len(self._mcache) > 400_000:
            self._mcache.clear()
        # ---- speedup #3: quiescence at the horizon -- resolve captures one level, leaf
        # value = best of stand-pat and the capture continuation (mover chooses)
        if self.qext and qx_boards:
            q_vals = [None] * len(qx_boards)
            q_todo, q_toks, q_globs = [], [], []
            for qi, qb in enumerate(qx_boards):
                tv = self._terminal_value(qb)
                if tv is not None:
                    q_vals[qi] = tv
                    continue
                hit = self._mcache.get(qb.fen())
                if hit is not None:
                    q_vals[qi] = hit
                else:
                    q_todo.append(qi)
                    tk, gl = tokenize(qb)
                    q_toks.append(tk); q_globs.append(gl)
            for a in range(0, len(q_todo), 4096):
                if stop is not None and stop():
                    raise KittyChess.SearchStop
                with torch.no_grad():
                    z = self._embed(q_toks[a:a+4096], q_globs[a:a+4096])
                    D = [self.dist(z, P3[[k2]].expand(len(z), -1)).float().cpu().numpy()
                         for k2 in range(3)]
                    DA = ([self.net.dA(z, P3[[k2]].expand(len(z), -1)).float().cpu().numpy()
                           for k2 in (0, 2)] if getattr(self.net, "split_head", False) else None)
                for j in range(len(D[0])):
                    qb = qx_boards[q_todo[a + j]]
                    if self.white_pov:
                        iw, il = (0, 2) if qb.turn else (2, 0)
                    else:
                        iw, il = 2, 0
                    import numpy as _np9
                    _e = _np9.exp(-_np9.array([D[0][j], D[1][j], D[2][j]]) / 5.0)
                    _e = _e / _e.sum()
                    _Em = float(_e[iw] + 0.5 * _e[1])
                    m = 1000.0 * _Em - (0.1 * min(float(DA[0 if iw == 0 else 1][j]), 40.0)
                                        if DA is not None else 0.0)
                    q_vals[q_todo[a + j]] = float(m)
                    self._mcache[qb.fen()] = float(m)
            for qi, pi_ in enumerate(qx_parent):
                if q_vals[qi] is None or lv_vals[pi_] is None:
                    continue
                cont = -q_vals[qi]                # capture continuation, leaf-mover POV
                if cont > lv_vals[pi_]:
                    lv_vals[pi_] = cont
        # ---- minimax rollup with stored per-level values
        vals = [None] * len(levels)
        vals[leaf_lv] = lv_vals
        for lv in range(leaf_lv - 1, -1, -1):
            out = [None] * len(levels[lv])
            for pi, (b, _, _) in enumerate(levels[lv]):
                ks = kids[lv][pi] if pi < len(kids[lv]) else []
                cand = [-vals[lv + 1][ci] for ci in ks if vals[lv + 1][ci] is not None]
                if cand:
                    out[pi] = max(cand)
                else:
                    tv = self._terminal_value(b)
                    out[pi] = tv if tv is not None else 0.0
            vals[lv] = out
        # ---- rows + PV by argmax walk
        rows = []
        for i, (b, pi, mv) in enumerate(levels[1]):
            pv = [mv]
            idx, lv = i, 1
            while lv < leaf_lv:
                ks = kids[lv][idx] if idx < len(kids[lv]) else []
                ks = [ci for ci in ks if vals[lv + 1][ci] is not None]
                if not ks:
                    break
                best = max(ks, key=lambda ci: -vals[lv + 1][ci])
                pv.append(levels[lv + 1][best][2])
                idx, lv = best, lv + 1
            # vals[1][i] is the CHILD-mover's value; the root row is its negation
            rows.append({"mv": mv, "value": float(-vals[1][i]), "pv": pv,
                         "resid": None, "depth_used": depth})
        rows.sort(key=lambda r: r["value"], reverse=True)
        return rows

    def search_wave(self, board, budget=1.5, waves_cap=64, wave_size=384, explore=1.5):
        """SELECTIVE WAVE SEARCH (2026-08-11, after full-width batching lost to alpha-beta):
        batching AND pruning. Batched MCTS-shaped minimax: each wave runs `wave_size` tree
        descents (negamax-best child + optimism bonus for unexpanded nodes, virtual-loss so
        descents diverge), collects distinct unexpanded nodes, expands and evaluates ALL their
        children in one large forward, backs values up. Promising lines run deep; dull ones
        stay shallow; nothing is hard-pruned -- neglected branches revive when siblings sour
        (the soft-pruning property the tactical-memory design wants). Returns rows like
        search()."""
        import heapq, time as _time
        deadline = _time.time() + budget
        N = {"b": [board.copy(stack=False)], "par": [-1], "mv": [None],
             "kids": [None], "stat": [0.0], "val": [0.0], "term": [None], "vl": [0]}
        if board.is_game_over(claim_draw=True) or not list(board.legal_moves):
            return []                       # TB-covered roots still get searched
        N["term"][0] = None

        def backup(idx):
            while idx != -1:
                ks = N["kids"][idx]
                if ks:
                    N["val"][idx] = max(-N["val"][c] for c in ks)
                idx = N["par"][idx]

        def descend():
            idx = 0
            while True:
                ks = N["kids"][idx]
                if ks is None:                       # unexpanded -> select it
                    return idx
                if not ks:                           # terminal
                    return None
                best, bestv = None, -1e18
                for c in ks:
                    v = -N["val"][c] + (explore if N["kids"][c] is None else 0.0) \
                        - 0.35 * N["vl"][c]
                    if v > bestv:
                        bestv, best = v, c
                N["vl"][best] += 1
                idx = best

        expanded_root = False
        for _w in range(waves_cap):
            if _time.time() > deadline and expanded_root:
                break
            targets = []
            seen = set()
            for _ in range(wave_size):
                t = descend()
                if t is None:
                    continue
                if t not in seen:
                    seen.add(t); targets.append(t)
            if not targets:
                break
            new_nodes, toks, globs = [], [], []
            for t in targets:
                b = N["b"][t]
                kid_ids = []
                for mv in b.legal_moves:
                    b.push(mv)
                    cb = b.copy(stack=False)
                    b.pop()
                    ci = len(N["b"])
                    for k2, v2 in (("b", cb), ("par", t), ("mv", mv), ("kids", None),
                                   ("stat", 0.0), ("val", 0.0), ("term", None), ("vl", 0)):
                        N[k2].append(v2)
                    kid_ids.append(ci)
                    tvc = self._terminal_value(cb)
                    if tvc is not None:
                        N["term"][ci] = tvc
                        N["stat"][ci] = tvc
                        N["val"][ci] = tvc
                        N["kids"][ci] = []
                    else:
                        cached = self._mcache.get(cb.fen())
                        if cached is not None:
                            N["stat"][ci] = cached; N["val"][ci] = cached
                        else:
                            new_nodes.append(ci)
                            tk, gl = tokenize(cb)
                            toks.append(tk); globs.append(gl)
                N["kids"][t] = kid_ids
            if new_nodes:
                P3 = self.poles[[self.pi["WIN"], self.pi["DRAW"],
                                 self.pi["LOSS"]]].to(self.device)
                for a in range(0, len(new_nodes), 4096):
                    with torch.no_grad():
                        z = self._embed(toks[a:a+4096], globs[a:a+4096])
                        D = [self.dist(z, P3[[k2]].expand(len(z), -1)).float().cpu().numpy()
                             for k2 in range(3)]
                        DA = ([self.net.dA(z, P3[[k2]].expand(len(z), -1)).float().cpu().numpy()
                               for k2 in (0, 2)] if getattr(self.net, "split_head", False)
                              else None)
                    for j in range(len(D[0])):
                        ci = new_nodes[a + j]
                        cb = N["b"][ci]
                        if self.white_pov:
                            iw, il = (0, 2) if cb.turn else (2, 0)
                        else:
                            iw, il = 2, 0
                        import numpy as _np9
                        _e = _np9.exp(-_np9.array([D[0][j], D[1][j], D[2][j]]) / 5.0)
                        _e = _e / _e.sum()
                        _Em = float(_e[iw] + 0.5 * _e[1])
                        m = 1000.0 * _Em - (0.1 * min(float(DA[0 if iw == 0 else 1][j]), 40.0)
                                            if DA is not None else 0.0)
                        N["stat"][ci] = float(m); N["val"][ci] = float(m)
                        self._mcache[cb.fen()] = float(m)
            for t in targets:
                backup(t)
            for i in range(len(N["vl"])):
                N["vl"][i] = 0
            expanded_root = True
        rows = []
        for c in (N["kids"][0] or []):
            pv, idx = [N["mv"][c]], c
            while N["kids"][idx]:
                nxt = max(N["kids"][idx], key=lambda x: -N["val"][x])
                pv.append(N["mv"][nxt]); idx = nxt
            rows.append({"mv": N["mv"][c], "value": float(-N["val"][c]), "pv": pv,
                         "resid": None, "depth_used": len(pv)})
        rows.sort(key=lambda r: r["value"], reverse=True)
        return rows

    def _jqt_mod(self):
        """the concept sidecar (lazy, shared by the goal bias and the directional readout)."""
        if getattr(self, "_jqt", None) is None:
            from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
                JQTModule)
            import os as _os, re as _re
            base = getattr(self, "_ckpt_base", None)
            _stem = _re.sub(r"_(latest|step\d+)$", "", base or "")
            _p = next((c + "_jqt.pt" for c in (base, _stem)
                       if c and _os.path.exists(c + "_jqt.pt")), None)
            if _p is None:
                raise FileNotFoundError(f"no _jqt.pt beside {base}")
            pay = torch.load(_p, map_location=self.device, weights_only=False)
            self._jqt = JQTModule(d_model=pay["d_in"], heads=pay["heads"], codes=pay["codes"],
                                  d=pay["d"], square_codes=pay.get("square_codes", 0),
                                  piece_codes=pay.get("piece_codes", 0)).to(self.device)
            self._jqt.load_state_dict(pay["state_dict"], strict=False)
            self._jqt.eval()
        return self._jqt

    def embed_cached(self, keys, toks, globs):
        """embed only the MISSES; returns (z (N,d), n_new). Rows come back in input order.
        The cache is shared with everything else that embeds positions, so a position
        scored for direction is free when the search reaches it."""
        miss = [i for i, k in enumerate(keys) if k not in self._zcache]
        if miss:
            with torch.no_grad():
                zm = self._embed([toks[i] for i in miss], [globs[i] for i in miss]).float()
            for j, i in enumerate(miss):
                self._zcache[keys[i]] = zm[j]
            if len(self._zcache) > 200_000:
                self._zcache.clear()
        return torch.stack([self._zcache[k] for k in keys]), len(miss)

    def goal_anchor(self, goal, gtype="global"):
        """(key1, key2) -> the z_B-space anchor for a concept goal, any vocabulary."""
        jm = self._jqt_mod()
        k1 = torch.tensor([int(goal[0])], dtype=torch.long, device=self.device)
        k2 = torch.tensor([int(goal[1])], dtype=torch.long, device=self.device)
        with torch.no_grad():
            if gtype == "square":
                return jm.anchors_for_sq(k1, k2).float()
            if gtype == "piece":
                return jm.anchors_for_pc(k1, k2).float()
            return jm.anchors_for(torch.stack([k1, k2], 1)).float()

    def goal_directions(self, board, goal, gtype="global", k=None):
        """DIRECTIONS IN SPACE (Kaveh 2026-08-13: "which move gets me closer to my goal").
        One batched readout over EVERY legal move -- no search -- returning per move:
            p_act    P(activate the goal) from the resulting position (probability ruler)
            dp       change vs the current position (+ = this move helps)
            dA       log1p distance to the goal anchor (length ruler)
            dd       parent dA - child dA (+ = moved closer, in plies-ish units)
            E        mover-POV expected points at the child (the disaster veto input)
            dE       change in E (what the direction COSTS)
        Sorted by p_act then dA (Kaveh's move-selection rule: probability first, then
        distance). Also returns `spread` = max-min p_act: when the field is flat this is
        near zero and the ranking is a SHORTLIST for search, not an answer.
        Embeddings and values are cached, so the search that follows pays nothing again."""
        import numpy as _np
        moves = list(board.legal_moves)
        if not moves:
            return {"rows": [], "spread": 0.0, "n_embedded": 0}
        fast = bool(getattr(self, "fastgen", True))
        try:
            from catspace.research.components.planner.approaches.quasimetric_nav.fastboard import FB
            rb = FB.from_chess(board) if fast else None
        except Exception:
            rb, fast = None, False
        keys, toks, globs, kids = [], [], [], []
        if fast and rb is not None:
            pairs = rb.children()
            _mv = {}
            for u, cb in pairs:
                keys.append(cb.key())
                tk, gl = cb.tok_glob()
                toks.append(tk); globs.append(gl); kids.append(cb)
                _mv[len(keys) - 1] = chess.Move.from_uci(u)
            moves = [_mv[i] for i in range(len(keys))]
        else:
            for mv in moves:
                board.push(mv)
                keys.append(board.fen())
                tk, gl = tokenize(board)
                toks.append(tk); globs.append(gl); kids.append(None)
                board.pop()
        pk = rb.key() if (fast and rb is not None) else board.fen()
        ptk, pgl = (rb.tok_glob() if (fast and rb is not None) else tokenize(board))
        z_all, n_new = self.embed_cached([pk] + keys, [ptk] + toks, [pgl] + globs)
        A = self.goal_anchor(goal, gtype)
        P3 = self.poles[[self.pi["WIN"], self.pi["DRAW"], self.pi["LOSS"]]].to(self.device)
        jm = self._jqt_mod()
        with torch.no_grad():
            Ae = A.expand(len(z_all), -1)
            dB = self.net.dB(z_all, Ae)
            p_act = torch.sigmoid(jm.activation_logit(dB)).cpu().numpy()
            dA = torch.log1p(self.net.dA(z_all, Ae).clamp(min=0)).cpu().numpy()
            DBp = torch.stack([self.net.dB(z_all, P3[[j]].expand(len(z_all), -1))
                               for j in range(3)], 1)
            pr = torch.softmax(-DBp / 5.0, 1).cpu().numpy()
        E_w = pr[:, 0] + 0.5 * pr[:, 1]
        # parent row is index 0; children follow. E is the MOVER's (the side to move NOW).
        mover_white = board.turn == chess.WHITE
        E_m = E_w if mover_white else 1.0 - E_w
        p0, d0, e0 = float(p_act[0]), float(dA[0]), float(E_m[0])
        rows = []
        for i, mv in enumerate(moves, start=1):
            rows.append({"uci": mv.uci(), "san": board.san(mv),
                         "p_act": round(float(p_act[i]), 4),
                         "dp": round(float(p_act[i]) - p0, 4),
                         "dA": round(float(dA[i]), 3),
                         "dd": round(d0 - float(dA[i]), 3),
                         "E": round(float(E_m[i]), 4),
                         "dE": round(float(E_m[i]) - e0, 4)})
        rows.sort(key=lambda r: (-r["p_act"], r["dA"]))
        sp = float(p_act[1:].max() - p_act[1:].min()) if len(moves) else 0.0
        return {"rows": rows[:k] if k else rows, "spread": round(sp, 4),
                "p_now": round(p0, 4), "dA_now": round(d0, 3), "E_now": round(e0, 4),
                "n_embedded": n_new, "n_moves": len(moves)}

    def goal_candidates(self, board, goal, gtype="global", k=6, min_E=None):
        """the BEST-SHOT shortlist (Kaveh 2026-08-13: "we wanna start from our best shot"):
        top-k moves by goal progress, disaster-vetoed. Flatness is fine here -- these are
        candidates to SEARCH, and the field differentiates as the search descends."""
        out = self.goal_directions(board, goal, gtype)
        rows = out["rows"]
        if min_E is not None:
            keep = [r for r in rows if r["E"] >= min_E]
            rows = keep or rows[:k]
        return [chess.Move.from_uci(r["uci"]) for r in rows[:k]], out

    def _goal_bias(self, toks, globs, goal):
        """GOAL-CONDITIONED leaf bias (Kaveh 2026-08-12: 'search is towards a subgoal').
        P(activate goal) on the probability ruler, scaled to the tie-epsilon band of the
        concept-value scale. The OUTCOME VETO is applied by the caller: mission-ranked,
        disaster-vetoed."""
        self._jqt_mod()
        with torch.no_grad():
            hc = torch.tensor([goal], dtype=torch.long, device=self.device)
            A = self._jqt.anchors_for(hc).float()
            z = self._embed(toks, globs).float()
            dB = self.net.dB(z, A.expand(len(z), -1))
            return torch.sigmoid(self._jqt.activation_logit(dB)).float().cpu().numpy()

    def search_coherent(self, board, budget=1.5, mass_floor=0.01, tau_m=0.35,
                        batch_cap=384, goal=None, w_goal=25.0):
        # NOTE: terminal values (MATE scale 1e6) dominate either evaluation scale, so
        # mates/TB stay decisive under concept evaluation too.
        """COHERENCE-BOUNDED SEARCH (Kaveh 2026-08-11: 'depth has to go as far as the
        coherence allows'). Each node carries P(reach) = product of move probabilities along
        its path, where a mover's move distribution = softmax(their child values / tau_m) --
        the probability head's worldview as a policy prior. The frontier is expanded in order
        of P(reach); expansion STOPS where the mass frays below `mass_floor`. Forced sequences
        (replies near probability 1) run 8-12 plies deep; quiet positions stay shallow -- the
        tree's shape IS the position's forcingness. Values back up as minimax for the move
        choice; returns rows like search() plus row['preach'] and row['depth_used']."""
        import heapq, time as _time
        deadline = _time.time() + budget
        if board.is_game_over(claim_draw=True) or not list(board.legal_moves):
            return []                       # TB-covered roots still get searched (sanity
                                            # battery: wave/coherent returned NO move at
                                            # <=5 pieces and silently drew ladder games)
        # RUST MOVEGEN (Kaveh 2026-08-12 "eng fix with a rust chess framework"): cozy-chess
        # via fastboard.FB -- 13.9x movegen+apply, 3.1x tokenize, zobrist cache keys. Interior
        # nodes are FB; moves are standard-uci STRINGS converted to chess.Move at the rows.
        self.last_evals = 0                     # per-call leaf-eval count (the effort ruler)
        fast = bool(getattr(self, "fastgen", True))
        if fast:
            try:
                from catspace.research.components.planner.approaches.quasimetric_nav.fastboard import FB
            except Exception:
                fast = False
        if fast:
            try:                       # cozy PANICS on illegal roots (analysis-board setups);
                root = FB.from_chess(board)   # python-chess path tolerates them
                root.legal_count()
            except BaseException:
                fast = False
        root = root if fast else board.copy(stack=False)
        _ck0 = (lambda x: x.key()) if fast else (lambda x: x.fen())
        ckey = (lambda x: (_ck0(x), goal)) if goal is not None else _ck0
        N = {"b": [root], "par": [-1], "mv": [None], "kids": [None],
             "val": [0.0], "preach": [1.0]}
        heap = [(-1.0, 0)]
        P3 = self.poles[[self.pi["WIN"], self.pi["DRAW"], self.pi["LOSS"]]].to(self.device)

        def backup(idx):
            while idx != -1:
                ks = N["kids"][idx]
                if ks:
                    N["val"][idx] = max(-N["val"][c] for c in ks)
                idx = N["par"][idx]

        while heap and _time.time() < deadline:
            # pop a batch of the most-reachable unexpanded nodes above the floor
            targets = []
            while heap and len(targets) < batch_cap:
                pr, idx = heapq.heappop(heap)      # floor is enforced at PUSH time; the old
                if N["kids"][idx] is None:         # pop-side re-check starved waves to 1 node
                    targets.append(idx)
            if not targets:
                break
            new_eval, toks, globs = [], [], []
            for t in targets:
                b = N["b"][t]
                kid_ids = []
                if fast:
                    pairs = b.children()
                else:
                    pairs = []
                    for mv in b.legal_moves:
                        b.push(mv)
                        pairs.append((mv, b.copy(stack=False)))
                        b.pop()
                for mv, cb in pairs:
                    ci = len(N["b"])
                    N["b"].append(cb); N["par"].append(t); N["mv"].append(mv)
                    N["kids"].append(None); N["val"].append(0.0); N["preach"].append(0.0)
                    kid_ids.append(ci)
                    tv = (cb.terminal_value(self.MATE, self._tb_probe) if fast
                          else self._terminal_value(cb))
                    if tv is not None:
                        N["val"][ci] = tv
                        # TB values are BOUNDS, not mate proofs: DTZ cannot rank equal-dtz
                        # mates (the ladder case). TB-DECISIVE nodes stay EXPANDABLE at any
                        # depth so the search can prove the real mate (1e6 > 5e5-scale)
                        # inside the budget; true game-end terminals and TB draws close.
                        if abs(tv) < self.MATE and tv != 0.0:
                            pass                      # leave kids=None -> searchable
                        else:
                            N["kids"][ci] = []
                    else:
                        cached = self._mcache.get(ckey(cb))
                        if cached is not None:
                            N["val"][ci] = cached
                        else:
                            new_eval.append(ci)
                            tk, gl = cb.tok_glob() if fast else tokenize(cb)
                            toks.append(tk); globs.append(gl)
                N["kids"][t] = kid_ids
            self.last_evals += len(new_eval)
            if self.cvq is not None and getattr(self, "concept_eval", True):
                for a in range(0, len(new_eval), 4096):
                    turns = [N["b"][ci].turn for ci in new_eval[a:a+4096]]
                    vals = self.concept_values(toks[a:a+4096], globs[a:a+4096], turns)
                    if goal is not None and len(vals):
                        pg = self._goal_bias(toks[a:a+4096], globs[a:a+4096], goal)
                        # mission-ranked, disaster-vetoed: bias only where the mover is
                        # not already losing outright (Em > 0.35 on the 1000-scale)
                        vals = [v + w_goal * float(pg[j]) * (1.0 if v > 350.0 else 0.0)
                                for j, v in enumerate(vals)]
                    for j, ci in enumerate(new_eval[a:a+4096]):
                        N["val"][ci] = vals[j]
                        self._mcache[ckey(N["b"][ci])] = vals[j]
            else:
                for a in range(0, len(new_eval), 4096):
                    with torch.no_grad():
                        z = self._embed(toks[a:a+4096], globs[a:a+4096])
                        D = [self.dist(z, P3[[k2]].expand(len(z), -1)).float().cpu().numpy()
                             for k2 in range(3)]
                        DA = ([self.net.dA(z, P3[[k2]].expand(len(z), -1)).float().cpu().numpy()
                               for k2 in (0, 2)] if getattr(self.net, "split_head", False)
                              else None)
                    for j in range(len(D[0])):
                        ci = new_eval[a + j]
                        cb = N["b"][ci]
                        if self.white_pov:
                            iw, il = (0, 2) if cb.turn else (2, 0)
                        else:
                            iw, il = 2, 0
                        import numpy as _np9
                        _e = _np9.exp(-_np9.array([D[0][j], D[1][j], D[2][j]]) / 5.0)
                        _e = _e / _e.sum()
                        _Em = float(_e[iw] + 0.5 * _e[1])
                        m = 1000.0 * _Em - (0.1 * min(float(DA[0 if iw == 0 else 1][j]), 40.0)
                                            if DA is not None else 0.0)
                        N["val"][ci] = float(m)
                        self._mcache[ckey(cb)] = float(m)
            # priors: mover picks among children ~ softmax(-child_val / tau) (child vals are
            # the CHILD-mover's POV, so the parent's preference is the negation)
            for t in targets:
                ks = N["kids"][t]
                if not ks:
                    continue
                import numpy as _np
                sc = _np.array([-N["val"][c] for c in ks]) / tau_m
                sc -= sc.max()
                pr = _np.exp(sc); pr /= pr.sum()
                depth_t = 0
                cur = t
                while cur != -1:
                    depth_t += 1; cur = N["par"][cur]
                for c, pc in zip(ks, pr):
                    N["preach"][c] = N["preach"][t] * float(pc)
                    # coherence floor beyond a 2-ply minimum (tactics need >= 2 plies even
                    # when the prior is flat -- the prior's flatness is the sibling problem)
                    if N["kids"][c] is None and (depth_t <= 2
                                                 or N["preach"][c] >= mass_floor):
                        heapq.heappush(heap, (-N["preach"][c], c))
                backup(t)
        import math as _math
        import numpy as _np
        _mvo = (lambda u: chess.Move.from_uci(u)) if fast else (lambda u: u)
        rows = []
        for c in (N["kids"][0] or []):
            pv, idx = [_mvo(N["mv"][c])], c
            while N["kids"][idx]:
                nxt = max(N["kids"][idx], key=lambda x: -N["val"][x])
                pv.append(_mvo(N["mv"][nxt])); idx = nxt
            # FORCING METRICS (Kaveh 2026-08-12, the two-queens game: "choose forcing moves
            # over moves that fray ... where we can premove"). Both come free from the tree:
            # force_h = entropy of the OPPONENT's plausible-reply distribution at this child
            # (low = constraining: checks, captures, only-moves). premove = the max-min value
            # of ONE move of ours that answers ALL their expanded replies (the pawn-push
            # property: our next move is chosen before seeing theirs).
            ks = N["kids"][c]
            if ks:
                sc = _np.array([-N["val"][k] for k in ks]) / tau_m
                sc -= sc.max()
                pr = _np.exp(sc); pr /= pr.sum()
                force_h = float(-(pr * _np.log(pr + 1e-12)).sum())
                by_uci = {}
                for r in ks:
                    if N["kids"][r]:
                        for m in N["kids"][r]:
                            _u = N["mv"][m] if fast else N["mv"][m].uci()
                            by_uci.setdefault(_u, []).append((r, -N["val"][m]))
                n_rep = sum(1 for r in ks if N["kids"][r])
                premove = max((min(v for _r, v in lst) for lst in by_uci.values()
                               if len(lst) == n_rep), default=None) if n_rep else None
            else:
                force_h = _math.log(max(1, N["b"][c].legal_count() if fast
                                        else len(list(N["b"][c].legal_moves))))
                premove = None
            rows.append({"mv": _mvo(N["mv"][c]), "value": float(-N["val"][c]), "pv": pv,
                         "preach": float(N["preach"][c]), "resid": None,
                         "depth_used": len(pv), "force_h": force_h, "premove": premove})
        rows.sort(key=lambda r: r["value"], reverse=True)
        # FORCING PREFERENCE: proven mates keep absolute rank; among the rest, moves within
        # eps of the best value re-rank by (constrain the opponent, then premove-ability) --
        # the cheap-to-execute win over the merely-shortest win. Flag-gated.
        # ONLY WHEN WINNING (Kaveh 2026-08-12, the knight-shuffle game: Nh3/Ng5/Nxh7 came
        # from breaking QUIET-position ties by forcingness -- the principle is 'among equal
        # WINS prefer the premove-able one', never 'prefer lunges when equal'): the band
        # re-ranks only if the top move already reads clearly winning for the mover.
        _win_thr = 650.0 if self.cvq is not None else 10.0
        if getattr(self, "forcing_pref", True) and len(rows) > 1 \
                and rows[0]["value"] >= min(_win_thr, 5e5):
            eps = getattr(self, "forcing_eps", 15.0)
            top = rows[0]["value"]
            band = [r for r in rows if r["value"] > 5e5 or top - r["value"] <= eps]
            rest = [r for r in rows if r not in band]
            band.sort(key=lambda r: (0, -r["value"], 0.0) if r["value"] > 5e5
                      else (1, r["force_h"],
                            -(r["premove"] if r["premove"] is not None else -1e9)))
            rows = band + rest
        return rows

    def concept_values(self, toks, globs, turns):
        """batched concept-bottleneck evaluation: returns mover-POV scalars. Scalar =
        1000 * mover expected points (decoded probabilities, the cascade's primary) +
        decoded decisive-distance margin as the tie epsilon. Every value decomposes into
        the position's 8 concept codes by construction."""
        with torch.no_grad():
            tok_t = torch.from_numpy(np.array(toks).astype(np.int64)).to(self.device)
            glob_t = torch.from_numpy(np.array(globs).astype(np.float32)).to(self.device)
            if getattr(self, "half", False):
                glob_t = glob_t.half()
            phi = self.net.backbone(tok_t, glob_t)
            y, _, _ = self.cvq(phi)
            y = (y * self.cvq_sd + self.cvq_mu).float().cpu().numpy()
        out = []
        for j, wtm in enumerate(turns):
            daW, daD, daL, pW, pD, pB = y[j]
            pW, pD, pB = max(pW, 0.0), max(pD, 0.0), max(pB, 0.0)
            zs = pW + pD + pB
            if zs > 1e-6:
                pW, pD, pB = pW / zs, pD / zs, pB / zs
            E = pW + 0.5 * pD
            Em = E if wtm else 1.0 - E
            out.append(1000.0 * Em)
        return out

    def mc_tiebreak(self, board, rows, R=24, band=1.0, cap=6, ply_cap=60):
        """CPU rollout tiebreak (Kaveh 2026-08-11 worst case, MEASURED better than the field
        at sibling resolution in winning positions: 82.0% vs 74.7% keeps-the-win over 150):
        among search rows within `band` of the top value, rank by mean random-playout result.
        Pure CPU -- runs while the GPU idles; no priors, no network."""
        import random as _rnd
        if not rows:
            return rows
        top_v = rows[0]["value"]
        tied = [r for r in rows if top_v - r["value"] <= band][:cap]
        if len(tied) < 2:
            return rows
        rr = _rnd.Random(0xC0FFEE)
        def playout(bb):
            n = 0
            while not bb.is_game_over(claim_draw=True) and n < ply_cap:
                bb.push(rr.choice(list(bb.legal_moves)))
                n += 1
            o = bb.outcome(claim_draw=True)
            if o is None or o.winner is None:
                return 0.5
            return 1.0 if o.winner else 0.0
        mover_white = board.turn
        scored = []
        for r in tied:
            board.push(r["mv"])
            w = sum(playout(board.copy(stack=False)) for _ in range(R)) / R
            board.pop()
            scored.append((w if mover_white else 1.0 - w, r))
        scored.sort(key=lambda t: -t[0])
        reordered = [r for _, r in scored] + rows[len(tied):]
        for (w, r) in scored:
            r["mc"] = round(w, 3)
        return reordered

    class SearchStop(Exception):
        """raised mid-search when the caller's stop() turns true (navigation cancelled it)."""

    def negamax(self, board, depth, alpha, beta, stop=None):
        """alpha-beta minimax where the EVALUATION IS THE MARGIN (Kaveh 2026-08-08:
        'minimax alpha beta pruning but instead of evaluate we do it from the margin').
        Children are scored in ONE batched forward per node for move ordering, which is also
        what makes the pruning bite. Returns (value mover-POV, pv list of Moves)."""
        if stop is not None and stop():
            raise KittyChess.SearchStop
        tv = self._terminal_value(board)
        if tv is not None:
            return tv, []
        moves = list(board.legal_moves)
        if not moves:
            return 0.0, []
        children = []
        for m in moves:
            board.push(m); children.append(board.copy(stack=False)); board.pop()
        cm = self.margins(children)                 # child margins, OPPONENT'S POV
        order = sorted(range(len(moves)), key=lambda i: cm[i])   # their worst first
        if depth <= 1:
            best = order[0]
            return -cm[best], [moves[best]]
        best_v, best_pv = -float("inf"), []
        for i in order:
            v, pv = self.negamax(children[i], depth - 1, -beta, -alpha, stop)
            v = -v
            if v > best_v:
                best_v, best_pv = v, [moves[i]] + pv
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best_v, best_pv

    def search(self, board, depth=3, top=3, stop=None, progress=None,
               adaptive=False, max_extra=1, resid_scale=2.0):
        """root search: rows sorted by searched value, each with its PV. `stop` (callable) makes
        the search ABORTABLE lichess-style: checked at every node; on stop the rows finished so
        far are returned (partial MultiPV beats a frozen UI). `progress(rows_sorted)` fires after
        every completed root move for live streaming.

        `adaptive` (Kaveh 2026-08-08, both Bellman uses): the field's own Bellman residual at
        the root gates the depth -- consistent field, base depth; contradictory field, up to
        `max_extra` deeper (one extra ply per `resid_scale` of |residual|). The residual is
        attached to every returned row as row['resid']; search budget falls automatically as
        the field's consistency improves ([[adaptive_search_budget]])."""
        resid = None
        if adaptive:
            resid = self.bellman_residual(board)
            if resid is not None:
                depth += min(max_extra, int(abs(resid) / resid_scale))
        tv = self._terminal_value(board)
        moves = list(board.legal_moves)
        if not moves:
            return []
        if self.head_order and getattr(self.net, "move_head", None) is not None \
                and len(moves) > 12:
            # HEAD AS ROOT PRIOR (2026-08-11): search only the head's top-10 candidates --
            # the policy narrows, the search verifies. Qg4 (head rank #27) is never searched.
            tk, gl = tokenize(board)
            with torch.no_grad():
                phi = self.net.backbone(
                    torch.from_numpy(np.asarray([tk], dtype=np.int64)).to(self.device),
                    torch.from_numpy(np.asarray([gl], dtype=np.float32)).to(self.device))
                mids = torch.from_numpy(np.array([move_ids(m) for m in moves],
                                                 dtype=np.int64)).to(self.device)
                delta = self.net.move_head(phi.expand(len(moves), -1),
                                           mids).float().cpu().numpy()
            mw = 0 if board.turn else 2
            keep = list(np.argsort(delta[:, mw])[:10])
            moves = [moves[i] for i in keep]
        rows = []
        for m in moves:
            board.push(m)
            try:
                v, pv = self.negamax(board, depth - 1, -float("inf"), float("inf"), stop)
            except KittyChess.SearchStop:
                board.pop()
                break
            board.pop()
            rows.append({"mv": m, "value": -v, "pv": [m] + pv,
                         "resid": resid, "depth_used": depth})
            if progress is not None:
                progress(sorted(rows, key=lambda r: r["value"], reverse=True))
        rows.sort(key=lambda r: r["value"], reverse=True)
        return rows

    def choose(self, board):
        moves = list(board.legal_moves)
        if not moves:
            return None
        if self.tb is not None and len(board.piece_map()) <= 5:
            mv = self._tb_move(board)
            if mv is not None:
                return mv
        toks, globs = [], []
        for mv in moves:
            board.push(mv)
            tk, gl = tokenize(board)
            toks.append(tk); globs.append(gl)
            board.pop()
        with torch.no_grad():
            z = self._embed(toks, globs)
            if self.ex is not None:
                M = self._exd(z).float().cpu().numpy()   # white-POV cols
                ow, ot = (0, 2) if board.turn else (2, 0)
                d_win, d_draw, d_loss = M[:, ow], M[:, 1], M[:, ot]
                D = None
            else:
                D = {n: self.dist(z, self.poles[[k]].expand(len(z), -1).to(self.device))
                     .float().cpu().numpy() for n, k in self.pi.items()}
        if D is None:
            pass
        elif self.white_pov:
            # colour-fixed poles: our win pole is by OUR colour, same at every child
            ours, theirs = ("WIN", "LOSS") if board.turn else ("LOSS", "WIN")
            d_win, d_draw, d_loss = D[ours], D["DRAW"], D[theirs]
        else:
            # POV flip at the child: our win = child-mover's LOSS
            d_win, d_draw, d_loss = D["LOSS"], D["DRAW"], D["WIN"]
        d_bad = np.minimum(d_draw, d_loss)
        margin = d_bad - d_win
        if self.nav == "cascade" and self.white_pov:
            # KAVEH'S CASCADE (2026-08-11): probability decides when it clearly isolates a
            # move; otherwise standing-aware LENGTH navigation. Gate = expected-points gap.
            P3 = self.poles[[self.pi["WIN"], self.pi["DRAW"], self.pi["LOSS"]]].to(self.device)
            with torch.no_grad():
                DBc = torch.stack([self.dist(z, P3[[k]].expand(len(z), -1))
                                   for k in range(3)], 1)
                pr = torch.softmax(-DBc / 5.0, 1).float().cpu().numpy()   # white-POV W/D/L
                DAc = torch.stack([self.net.dA(z, P3[[k]].expand(len(z), -1))
                                   for k in range(3)], 1).float().cpu().numpy()
            E = pr[:, 0] + 0.5 * pr[:, 1]
            if not board.turn:
                E = 1.0 - E                                  # mover-POV expected points
            order_E = np.argsort(-E)
            if len(E) > 1 and E[order_E[0]] - E[order_E[1]] >= self.gate:
                return moves[int(order_E[0])]                # probability isolates the move
            # length fallback, standing-aware; columns mover-POV
            aw, ad, al = ((DAc[:, 0], DAc[:, 1], DAc[:, 2]) if board.turn
                          else (DAc[:, 2], DAc[:, 1], DAc[:, 0]))
            standing = float(E.mean())
            if standing <= 0.45:
                # losing: chase the nearer salvage (win or draw), per Kaveh's branch 3
                tgt = aw if aw.min() <= ad.min() else ad
                return moves[int(np.argmin(tgt))]
            # winning/balanced: descend own win, flee draw AND loss (progress + no shuffling)
            sc = np.minimum(ad, al) - aw
            return moves[int(np.argmax(sc))]
        if self.nav == "ab" and self.white_pov:
            # A-STEER + B-GATE (Kaveh 2026-08-08 "try both ways, arena them"): progress along
            # the LENGTH ruler toward our absorbing pole (descends -- the odometer), safety
            # from the PROBABILITY ruler (the committor threat margin). Each z-scored within
            # the move set so neither ruler's native scale dominates.
            ow, ot = ("WIN", "LOSS") if board.turn else ("LOSS", "WIN")
            with torch.no_grad():
                daw = self.net.dA(z, self.poles[[self.pi[ow]]].expand(len(z), -1)
                                  .to(self.device)).float().cpu().numpy()
                dal = self.net.dA(z, self.poles[[self.pi[ot]]].expand(len(z), -1)
                                  .to(self.device)).float().cpu().numpy()
            prog = dal - daw                     # higher = closer to our end than theirs
            def _z(v):
                s = v.std()
                return (v - v.mean()) / (s if s > 1e-9 else 1.0)
            margin = _z(prog) + _z(margin)
        order = np.lexsort((-d_bad, -margin))    # primary margin desc, then d_bad desc
        return moves[int(order[0])]


def play(engine, opponent, engine_white, max_plies=300, start_fen=None):
    b = chess.Board(start_fen) if start_fen else chess.Board()
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        mv = engine.choose(b) if b.turn == engine_white else opponent(b)
        if mv is None:
            break
        b.push(mv)
    out = b.outcome(claim_draw=True)
    if out is None or out.winner is None:
        return 0.5
    return 1.0 if out.winner == engine_white else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--opponent", default="random", choices=["random"])
    ap.add_argument("--cond-elo", type=float, default=None)
    ap.add_argument("--start", default="balanced", choices=["balanced", "piece-up"],
                    help="piece-up: opponent missing a knight -- the conversion test")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    eng = KittyChess(args.ckpt, args.device, args.cond_elo)
    rng = random.Random(0)

    def rand_opp(b):
        ms = list(b.legal_moves)
        return rng.choice(ms) if ms else None

    def start_fen(white):
        if args.start == "balanced":
            return None
        b = chess.Board()
        squares = [sq for sq, pc in b.piece_map().items()
                   if pc.piece_type == chess.KNIGHT and pc.color != white]
        b.remove_piece_at(rng.choice(squares))
        return b.fen()

    score, n = 0.0, 0
    for g in range(args.games):
        white = g % 2 == 0
        score += play(eng, rand_opp, white, start_fen=start_fen(white))
        n += 1
        if n % 10 == 0:
            print(f"[kitty] {n}/{args.games}  score {score/n:.2f}", flush=True)
    print(f"[kitty] FINAL vs {args.opponent} ({args.start}): {score/n:.3f} over {n} games "
          f"(0.5 = parity; random-vs-random ~0.5)")


if __name__ == "__main__":
    main()
