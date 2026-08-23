#!/usr/bin/env python
"""koopman_search.py -- alpha-beta search in operator space (Kaveh 2026-08-22 mandate).

The search economics thesis as an engine: a node's children are evaluated by ONE
batched linear operator application on the augmented concept state (60-300us/child),
priced by the phi-only value head; full board re-encodes are spent ONLY on nodes that
survive into the principal variation (the trusted-anchor budget). Move ordering at
rolled states uses the rolled action-value head with position-free move tokens, so
grandchildren are ranked without any re-encode.

    .venv/bin/python -m ...koopman_search --ckpt artifacts/experiments/gko2i_latest.pt \
        --selftest
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


class KoopmanEngine:
    def __init__(self, ckpt, field_args="artifacts/experiments/gcf4h_latest.pt",
                 device="cpu", reencode_budget=8, blend=0.75):
        import chess
        from catspace.research.tools.training_infra.hl_gauss import HLGauss
        from catspace.research.tools.training_infra.koopman_ops import (
            BasisOps, KoopmanOps, TransformerLift)
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_concept_field import GraphConceptField
        self.chess = chess
        self.dev = device
        ckf = torch.load(field_args, map_location="cpu", weights_only=False)
        A = dict(ckf["args"])
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        RA = ck.get("args") or {}
        self.gen5 = bool(RA.get("gen5"))
        for k in ("g_heads", "g_codes", "d_code", "slow_heads", "sq_codes"):
            if RA.get(k):
                A[k] = RA[k]
        self.field = GraphConceptField(
            d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3,
            n_eg=1, cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
            cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
            move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
            attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]),
            kin=bool(A["kin"]), file_nodes=bool(A["file_nodes"]),
            phase_head=bool(A["phase_head"]), threat_head=bool(A["threat_head"]),
            elo_cond=True, g_heads=A["g_heads"], g_codes=A["g_codes"],
            d_code=A["d_code"], sq_codes=A.get("sq_codes", 256),
            mh_vq=bool(RA.get("g_heads")), gen5=self.gen5).eval()
        sd = ck["state_dict"]
        self.field.load_state_dict({k[6:]: v for k, v in sd.items()
                                    if k.startswith("field.")})
        self.lift = TransformerLift(self.field.d_state, g_heads=A["g_heads"],
                                    d_code=A["d_code"],
                                    d_sq=(16 if self.gen5 else 32), d_phi=256,
                                    n_layers=int(RA.get("lift_layers", 3)),
                                    n_piece=(32 if self.gen5 else 0)).eval()
        self.lift.load_state_dict({k[5:]: v for k, v in sd.items()
                                   if k.startswith("lift.")})
        if self.gen5:
            self.ops = BasisOps(self.lift.d_aug, n_basis=64,
                                rank=int(RA.get("op_rank", 16))).eval()
        else:
            self.ops = KoopmanOps(self.lift.d_aug, n_codes=A["move_codes"],
                                  rank=int(RA.get("op_rank", 16))).eval()
        self.ops.load_state_dict({k[4:]: v for k, v in sd.items()
                                  if k.startswith("ops.")})
        import torch.nn as nn
        kb = sd["sf_av.2.bias"].shape[0]
        self.sf_av = nn.Sequential(nn.Linear(self.field.d_state + A["d_model"], 512),
                                   nn.GELU(), nn.Linear(512, kb))
        self.sf_av.load_state_dict({k[6:]: v for k, v in sd.items()
                                    if k.startswith("sf_av.")})
        if any(k.startswith("sf_v_full.") for k in sd):
            # search reads the FULL head when trained (material-sighted); phi-only
            # remains the training anchor
            vb = sd["sf_v_full.2.bias"].shape[0]
            self.sf_v = nn.Sequential(nn.Linear(self.lift.d_aug, 256), nn.GELU(),
                                      nn.Linear(256, vb))
            self.sf_v.load_state_dict({k[10:]: v for k, v in sd.items()
                                       if k.startswith("sf_v_full.")})
            self._v_slice = slice(None)
            self._wdl = (vb == 3)
        else:
            vb = sd["sf_v.2.bias"].shape[0]
            self.sf_v = nn.Sequential(nn.Linear(256, 256), nn.GELU(),
                                      nn.Linear(256, vb))
            self.sf_v.load_state_dict({k[5:]: v for k, v in sd.items()
                                       if k.startswith("sf_v.")})
            self._v_slice = slice(self.lift.d_state, None)
            self._wdl = (vb == 3)
        self.av = nn.Sequential(nn.Linear(self.lift.d_aug + 64, 512), nn.GELU(),
                                nn.Linear(512, kb))
        self.av.load_state_dict({k[8:]: v for k, v in sd.items()
                                 if k.startswith("av_roll.")})
        self.hl = HLGauss(k=kb)
        for m in (self.field, self.lift, self.ops, self.sf_v, self.av):
            for p in m.parameters():
                p.requires_grad_(False)
        self.reencode_budget = reencode_budget
        self.blend = blend
        self.stats = {"encodes": 0, "op_applies": 0, "t_encode": 0.0, "t_op": 0.0}

    def _root_av(self, zq, h_mv, n):
        """Per-move action-values at an ENCODED node -- the sharp head (SF-spearman
        ~0.5); rolled values are the cheap wide net, this is the aimed rifle."""
        with torch.no_grad():
            x = torch.cat([zq[:, None, :].expand(-1, h_mv.shape[1], -1), h_mv], -1)
            wp = self.hl.decode(self.sf_av(x))[0, :n]
        return wp

    def _encode(self, board, elo=2000.0):
        """Full root/anchor encode -> (z_aug, per-move codes, move list)."""
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import featurize, MV_CAP
        from catspace.research.components.encoder.approaches.reach_probability.src \
            .trajectories import normalise_elo
        t0 = time.perf_counter()
        self._last = None
        g, _ = parent_graph(board)
        b = {k: torch.from_numpy(np.asarray(g[k])[None]) for k in
             ("sq", "mv", "mut", "att", "n_mv", "n_att")}
        psl = (torch.from_numpy(np.asarray(g["pslot"])[None]).long()
               if self.gen5 else None)
        b["gl"] = torch.from_numpy(glob_of_board(board)[None])
        sqf, mvf, edges, mask = featurize(b, self.dev, n_nodes=self.field.n_nodes)
        off = torch.zeros(1, 1, dtype=torch.long)
        jj = torch.arange(MV_CAP)[None, :]
        for name in ("vac", "dsc"):
            a2 = torch.from_numpy(np.asarray(g[name])[None]).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        el = torch.from_numpy(normalise_elo(np.full((1, 2), elo, np.float32)))
        with torch.no_grad():
            zq, _, _, _, h_mv, _ = self.field.encode(sqf, mvf, edges, mask, elo=el,
                                                     pslot=psl)[:6]
            q_mv, idx_mv, _ = self.field.mv_vq(self.field.mv_vq_proj(h_mv))
            za = self.lift(zq, q_mv * mask[..., None])
        moves = list(board.legal_moves)[:int(g["n_mv"])]
        self._last = (zq, h_mv, len(moves))
        self.stats["encodes"] += 1
        self.stats["t_encode"] += time.perf_counter() - t0
        if self.gen5:
            # bare move identity (from, to, promo-class): raw mv cols 1/2/5 --
            # the BasisOps conditioning contract (state carries all context)
            cond = torch.from_numpy(
                np.asarray(g["mv"])[:len(moves), [1, 2, 5]].astype(np.int64))
        else:
            cond = idx_mv[0, :len(moves)]
        return za, cond, moves, mvf[0, :len(moves)]

    def _child_values(self, za, codes):
        """Batched operator roll of ALL children + phi-value pricing.
        Returns win-prob FOR THE MOVER at the parent (value of child is for the
        child's mover = opponent, so flip)."""
        t0 = time.perf_counter()
        with torch.no_grad():
            if self.gen5:
                zac = self.ops(za.expand(len(codes), -1),
                               codes[:, 0], codes[:, 1], codes[:, 2])
            else:
                zac = self.ops(za.expand(len(codes), -1), codes)
            logits = self.sf_v(zac[:, self._v_slice])
            if self._wdl:
                pr = torch.softmax(logits, -1)
                wp_child = pr[:, 0] + 0.5 * pr[:, 1]
            else:
                wp_child = self.hl.decode(logits)
        self.stats["op_applies"] += len(codes)
        self.stats["t_op"] += time.perf_counter() - t0
        return 1.0 - wp_child, zac

    def _order_moves(self, zac, mvf_child):
        """Rolled action-value ordering for a CHILD state's replies (no re-encode):
        av head on [rolled z_aug ; mv_emb(token)] -- used to sort expansion."""
        with torch.no_grad():
            emb = self.field.mv_emb(mvf_child[None])
            x = torch.cat([zac[None, None, :].expand(1, emb.shape[1], -1), emb], -1)
            wp = self.hl.decode(self.av(x))[0]
        return wp

    def search(self, board, depth=3, elo=2000.0):
        """Alpha-beta with operator-rolled leaf evaluation and PV re-encoding."""
        self.stats = {"encodes": 0, "op_applies": 0, "t_encode": 0.0, "t_op": 0.0}
        budget = [self.reencode_budget]

        def ab(bd, d, alpha, beta, can_encode):
            if bd.is_checkmate():
                return 0.0
            if bd.is_stalemate() or bd.is_insufficient_material():
                return 0.5
            za, codes, moves, mvf = self._encode(bd, elo)
            if not moves:
                return 0.5
            vals, zac = self._child_values(za, codes)
            # SHARP-HEAD BLEND at encoded nodes: per-move action-values dominate the
            # ranking (sibling resolution); rolled values remain as the tiebreak/prune
            av = self._root_av(*self._last)
            vals = self.blend * av + (1.0 - self.blend) * vals
            order = torch.argsort(vals, descending=True)
            if d <= 1:
                # CAPTURE QUIESCENCE (stockfish-style): never trust a hot leaf --
                # if the best reply is a capture or check, resolve one ply further
                # (rolled values misprice hanging pieces; quiet states they price well)
                bi = order[0].item()
                if budget[0] > 0 and (bd.is_capture(moves[bi])
                                      or bd.gives_check(moves[bi])):
                    budget[0] -= 1
                    bd.push(moves[bi])
                    v = 1.0 - ab(bd, 1, 1.0 - beta, 1.0 - alpha, False)
                    bd.pop()
                    # stand-pat: the best QUIET alternative's static value; the hot
                    # move counts only at its RESOLVED value
                    quiet = [o.item() for o in order
                             if not bd.is_capture(moves[o.item()])
                             and not bd.gives_check(moves[o.item()])]
                    sp = float(vals[quiet[0]]) if quiet else 0.0
                    return max(sp, v)
                return float(vals[order[0]])
            best = alpha
            for rank, oi in enumerate(order.tolist()):
                mv = moves[oi]
                if rank == 0 and can_encode and budget[0] > 0 and d > 1:
                    budget[0] -= 1
                    bd.push(mv)
                    v = 1.0 - ab(bd, d - 1, 1.0 - beta, 1.0 - best, True)
                    bd.pop()
                else:
                    v = float(vals[oi])
                if v > best:
                    best = v
                if best >= beta:
                    break
            return best

        za, codes, moves, mvf = self._encode(board, elo)
        vals, zac = self._child_values(za, codes)
        av = self._root_av(*self._last)
        vals = self.blend * av + (1.0 - self.blend) * vals
        order = torch.argsort(vals, descending=True)
        scored = []
        for oi in order.tolist()[:8]:
            board.push(moves[oi])
            v = 1.0 - ab(board, depth - 1, 0.0, 1.0, True)
            board.pop()
            scored.append((moves[oi], v))
        scored.sort(key=lambda t: -t[1])
        return scored


def _selftest(ckpt):
    import chess
    eng = KoopmanEngine(ckpt)
    fails = []

    def gate(name, ok, detail):
        print(f"  {name:<6} {'PASS' if ok else 'FAIL':<5} {detail}")
        if not ok:
            fails.append(name)

    b = chess.Board()
    scored = eng.search(b, depth=2)
    legal = all(m in b.legal_moves for m, _ in scored)
    probs = all(0.0 <= v <= 1.0 for _, v in scored)
    gate("S1", len(scored) > 0 and legal and probs,
         f"root search returns {len(scored)} legal scored moves, best "
         f"{scored[0][0].uci()} @ {scored[0][1]:.3f}")
    s2 = eng.search(b, depth=2)
    gate("S2", [ (m.uci(), round(v,6)) for m,v in scored ] ==
               [ (m.uci(), round(v,6)) for m,v in s2 ],
         "deterministic across repeated searches")
    st = eng.stats
    per_op = st["t_op"] / max(st["op_applies"], 1) * 1e6
    per_enc = st["t_encode"] / max(st["encodes"], 1) * 1e3
    gate("S3", st["op_applies"] > st["encodes"],
         f"economics: {st['encodes']} encodes ({per_enc:.1f}ms) vs "
         f"{st['op_applies']} op-applies ({per_op:.0f}us) -> "
         f"{per_enc*1000/max(per_op,1e-9):.0f}x per node")
    mate = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    sm = eng.search(mate, depth=2)
    gate("S4", sm[0][1] > 0.5, f"winning position evaluated {sm[0][1]:.3f} > 0.5 "
         f"(best {sm[0][0].uci()})")
    print(f"\n  {'ALL SEARCH GATES PASS' if not fails else 'FAILED: ' + ','.join(fails)}")
    import sys
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gko2i_latest.pt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(a.ckpt)
