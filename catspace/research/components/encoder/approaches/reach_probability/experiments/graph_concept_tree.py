#!/usr/bin/env python
"""graph_concept_tree.py -- gct: the shallow search breathed into training (v2c realized).

Kaveh 2026-08-20: "I want to predict stockfish not just based on the one or two moves but
based on a shallow search... how do we bring the shallow search into the training so that
it's available as a tool that the weights build around?"

Three mechanisms, per the locked design (JOURNAL 2026-08-20, literature distilled first):

  A. TreeQN-style DIFFERENTIABLE TREE, fully latent past the root, in CONCEPT space:
     depth-1 children for ALL legal moves from the trained dynamics head (expected-code
     embeddings, one batch); deeper plies expand B learned REPLY SLOTS via residual
     transitions z' = unitnorm(z + tanh(W[z ++ e_slot])) (probabilistic plies); the
     committor values every node; NEGAMAX-parity soft backup sum(x*softmax(x)) with
     -b(-x) on opponent plies and TD(lambda) mixing of intermediate values. The move
     loss applies to the SEARCHED logits (prior + kappa*Q_backed) -- gradients flow
     through the backup into transition, value, and prior.
  B. GUMBEL SEARCH TARGETS (mctx form): a real-board beam alpha-beta with the CURRENT
     weights on a small per-step subset; prior head trained toward
     softmax(prior + sigma*normQ_completed).
  C. TD-LEAF: huber(V_static(root), stopgrad(V_backed(root))).

Fine-tunes a gcf4h checkpoint (--resume-field); the field's own losses keep running at
their weights so the vocabulary does not drift from under the tree.

    .venv/bin/python -u -m ...graph_concept_tree --graphs ..._hum_....npz \
        --resume-field artifacts/experiments/gcf4h_latest.pt --n-human 30000 --games 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import MV_CAP


class ConceptTree(nn.Module):
    """the differentiable shallow search over a GraphConceptField's concept space."""

    def __init__(self, field, beam=3, depth=3, lam=0.8, kappa=2.0):
        super().__init__()
        self.field = field
        self.beam, self.depth, self.lam, self.kappa = beam, depth, lam, kappa
        d_state = field.d_state
        # reply-slot transitions: B abstract continuations per latent node
        self.slot_emb = nn.Parameter(torch.randn(beam, 64) * 0.05)
        self.trans = nn.Sequential(nn.Linear(d_state + 64, d_state), nn.Tanh())
        self.slot_w = nn.Sequential(nn.Linear(d_state, 64), nn.GELU(),
                                    nn.Linear(64, beam))

    @staticmethod
    def _unit(z):
        return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _value(self, z, white_pov):
        """committor -> mover-POV margin at a node whose side-to-move is given."""
        _, _, out_l = self.field.predict_resolution(z)
        p = torch.softmax(out_l, -1)                      # White-POV [W, D, L]
        vw = p[..., 0] - p[..., 2]
        return torch.where(white_pov, vw, -vw)

    def _expand_slots(self, z):
        """(N, d) -> (N, B, d): B latent continuations + their mix weights (N, B)."""
        N = z.shape[0]
        e = self.slot_emb[None].expand(N, -1, -1)
        zin = torch.cat([z[:, None, :].expand(-1, self.beam, -1), e], -1)
        z2 = self._unit(z[:, None, :] + self.trans(zin))
        w = torch.softmax(self.slot_w(z), -1)
        return z2, w

    def _backup(self, v):
        """soft-greedy backup over the last dim: sum x*softmax(x) (TreeQN eq. 9)."""
        return (v * torch.softmax(v, -1)).sum(-1)

    def _recurse(self, z, depth, white_pov):
        """-> mover-POV backed value of latent nodes z (N, d); side to move = white_pov."""
        v_here = self._value(z, white_pov)
        if depth == 0:
            return v_here
        z2, w = self._expand_slots(z)                      # (N,B,d), (N,B)
        flat = z2.reshape(-1, z2.shape[-1])
        wp = white_pov if torch.is_tensor(white_pov) else torch.full(
            (z.shape[0],), bool(white_pov), device=z.device, dtype=torch.bool)
        v_child = self._recurse(flat, depth - 1, (~wp).repeat_interleave(self.beam))
        v_child = v_child.reshape(z.shape[0], self.beam)
        # child values are OPPONENT-POV; negamax: my value of a reply set = -softbackup(-(-v))
        # children maximize their own POV, I see the negation, weighted by slot plausibility
        v_opp = self._backup(v_child + torch.log(w.clamp(min=1e-8)))   # their soft-best
        v_search = -v_opp
        # TD(lambda) mixing (TreeQN eq. 4): blend this node's static value in
        return (1 - self.lam) * v_here + self.lam * v_search

    def searched_logits(self, zq, g_ids, h_mv, mv_mask, dyn_children, white_root):
        """-> (B, MV) searched logits = prior + kappa * Q_backed per root move.
        dyn_children: (B, MV, d_state) depth-1 latent children (expected-code embeddings),
        white_root: (B,) bool side to move at root."""
        prior = self.field.move_logits(zq, h_mv, mv_mask)              # (B, MV)
        Bb, MV, D = dyn_children.shape
        top = min(self.beam, MV)
        idx = prior.masked_fill(~mv_mask, -1e9).topk(top, dim=-1).indices   # (B, top)
        kids = dyn_children.gather(1, idx[:, :, None].expand(-1, -1, D))
        flat = self._unit(kids.reshape(-1, D))
        wp_child = (~white_root)[:, None].expand(-1, top).reshape(-1)
        v_kid = self._recurse(flat, self.depth - 1, wp_child)          # child-mover POV
        q = (-v_kid).reshape(Bb, top)                                  # root-mover POV
        searched = prior.clone()
        searched = searched.scatter(1, idx, prior.gather(1, idx) + self.kappa * q)
        return searched.masked_fill(~mv_mask, float("-inf")), q, idx


def build_expected_children(field, zq, mvf, mv_mask, g_codes):
    """(B, MV, d_state) differentiable depth-1 children for ALL legal moves: the dyn
    head's code DISTRIBUTIONS -> expected codebook embeddings; square/edge blocks carry
    the parent's (one ply moves them little; the global heads are what the tree plans
    over). Concept-space children by construction."""
    B, MV = mv_mask.shape
    Hh, dc = field.g_heads, field.d_code
    e = field.mv_emb(mvf)                                   # (B, MV, 64)
    dy = field.dyn(torch.cat([zq[:, None, :].expand(-1, MV, -1), e], -1))
    lg = dy[:, :, :Hh * g_codes].reshape(B, MV, Hh, g_codes)
    probs = torch.softmax(lg, -1)
    books = torch.stack([field.vq[h]._codebook.embed[0] for h in range(Hh)])  # (H,C,dc)
    gpart = torch.einsum("bmhc,hcd->bmhd", probs, books).reshape(B, MV, Hh * dc)
    # square + edge blocks: keep the parent's (slow-moving relative to one ply, and the
    # global heads are what the tree searches over); zero-pad would distort the committor
    rest = zq[:, None, Hh * dc:].expand(-1, MV, -1)
    kids = torch.cat([gpart, rest], -1)
    return kids * mv_mask[:, :, None].float()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--resume-field", required=True, help="gcf4h checkpoint to fine-tune")
    ap.add_argument("--n-human", type=int, default=30000)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--n-piecedown", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--w-tree", type=float, default=2.0)
    ap.add_argument("--w-prior", type=float, default=0.5)
    ap.add_argument("--w-tdleaf", type=float, default=0.5)
    ap.add_argument("--w-gumbel", type=float, default=1.0)
    ap.add_argument("--gumbel-n", type=int, default=8, help="positions/step given REAL "
                    "depth-1 search targets (0 disables)")
    ap.add_argument("--gumbel-width", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--fixed-elo", type=float, default=None,
                    help="condition EVERY row at this elo (SF corpus has none; use the "
                         "human corpus max ~2750 -- 'predict SF through the search' with "
                         "the trunk dialed to its strongest-seen rating)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/gct1")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    dev = args.device

    import chess
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.tools.training_infra.concepts_reference import board_from_row
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph, glob_of_board)
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)

    t0 = time.time()
    z = np.load(args.graphs)
    D = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in z.files}
    N = len(D["p_row"])
    tr = T.build(n_human=args.n_human, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_te = test_g[gor[D["p_row"].numpy()]]
    tr_i = np.flatnonzero(~is_te); te_i = np.flatnonzero(is_te)
    p_np = D["p_row"].numpy()

    ck = torch.load(args.resume_field, map_location="cpu", weights_only=False)
    A = ck["args"]
    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"],
        n_res=3, n_eg=1, cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64),
        edge_codes=A["edge_codes"], move_codes=A["move_codes"],
        gps_parallel=bool(A["gps_parallel"]), attn_bias=bool(A["attn_bias"]),
        agg_max=bool(A["agg_max"]), kin=bool(A["kin"]), file_nodes=bool(A["file_nodes"]),
        phase_head=bool(A["phase_head"]), threat_head=bool(A["threat_head"]),
        elo_cond=bool(A.get("elo", 0)), g_heads=A["g_heads"], g_codes=A["g_codes"],
        d_code=A["d_code"]).to(dev)
    field.load_state_dict(ck["state_dict"])
    G_CODES = A["g_codes"]
    tree = ConceptTree(field, beam=args.beam, depth=args.depth, lam=args.lam,
                       kappa=args.kappa).to(dev)
    print(f"[gct] field resumed from {args.resume_field} (step {ck.get('step')}) | "
          f"tree beam {args.beam} depth {args.depth} | "
          f"{sum(p2.numel() for p2 in tree.parameters()) - sum(p2.numel() for p2 in field.parameters()):,} new params",
          flush=True)

    EL = None
    if A.get("elo", 0) and args.fixed_elo is not None:
        from catspace.research.components.encoder.approaches.reach_probability.src \
            .trajectories import normalise_elo
        e = float(normalise_elo(np.float32(args.fixed_elo)))
        EL = torch.full((N, 2), e)
        print(f"[gct] fixed elo conditioning: {args.fixed_elo:.0f} (normalized {e:.2f})",
              flush=True)
    elif A.get("elo", 0):
        el_g = np.asarray(tr.elo, np.float32)
        el_rows = el_g[gor[p_np]]
        stm_rows = D["gl"][:, 0].numpy().astype(bool)
        mover = np.where(stm_rows, el_rows[:, 0], el_rows[:, 1])
        opp = np.where(stm_rows, el_rows[:, 1], el_rows[:, 0])
        from catspace.research.components.encoder.approaches.reach_probability.src \
            .trajectories import normalise_elo
        EL = torch.from_numpy(np.stack([normalise_elo(mover), normalise_elo(opp)], 1))

    opt = torch.optim.Adam(tree.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    probe = te_i[rng.choice(len(te_i), min(2500, len(te_i)), replace=False)]
    WHITE_ROOT = torch.from_numpy(D["gl"][:, 0].numpy().astype(bool))

    def parent_batch(idx):
        b = {k: D[k][idx].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev, n_nodes=field.n_nodes)
        off = (torch.arange(len(idx), device=dev) * field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a = D[name][idx].to(dev).long()
            val = (a != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a))[val],
                           (a + off[:, :, None])[val])
        return sqf, mvf, edges, mask

    @torch.no_grad()
    def gumbel_targets(idx):
        """REAL depth-1 Q for top-width moves; mctx-form target on all legal."""
        rows = p_np[idx.numpy()]
        outs = []
        for r in rows:
            b = board_from_row(tr.tok[r], tr.glob[r])
            g0, _ = parent_graph(b)
            n_mv = int(g0["n_mv"])
            mvs = list(b.legal_moves)[:n_mv]
            kids = []
            for m in mvs:
                b.push(m); kids.append((b.copy(stack=False))); b.pop()
            outs.append((b, mvs, kids))
        return outs

    def encode_boards(boards, elo_rows):
        gs = [parent_graph(b)[0] for b in boards]
        Db = {k: torch.from_numpy(np.stack([g[k] for g in gs])) for k in
              ("sq", "mv", "mut", "vac", "dsc", "att")}
        Db["gl"] = torch.from_numpy(np.stack([glob_of_board(b) for b in boards]))
        Db["n_mv"] = torch.tensor([g["n_mv"] for g in gs], dtype=torch.uint8)
        Db["n_att"] = torch.tensor([g["n_att"] for g in gs], dtype=torch.uint16)
        bb = {k: Db[k].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(bb, dev, n_nodes=field.n_nodes)
        off = (torch.arange(len(boards), device=dev) * field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a = Db[name].to(dev).long()
            val = (a != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a))[val],
                           (a + off[:, :, None])[val])
        enc = field.encode(sqf, mvf, edges, mask, elo=elo_rows)
        return enc[0]

    def searched_forward(idx, want_probe=False):
        sqf, mvf, edges, mask = parent_batch(idx)
        elo_b = EL[idx].to(dev) if EL is not None else None
        zq, g_ids, sq_ids, vl, h_mv, pre_g = field.encode(sqf, mvf, edges, mask, elo=elo_b)
        kids = build_expected_children(field, zq, mvf, mask, G_CODES)
        wr = WHITE_ROOT[idx].to(dev)
        searched, q, top_idx = tree.searched_logits(zq, g_ids, h_mv, mask, kids, wr)
        return searched, q, top_idx, zq, h_mv, mask, vl, wr, mvf

    def step_fn(model, step):
        field.train(); tree.train()
        idx = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
        searched, q, top_idx, zq, h_mv, mask, vl, wr, mvf = searched_forward(idx)
        played = D["played"][idx].long().to(dev)
        l_tree = F.cross_entropy(searched, played)
        prior = field.move_logits(zq, h_mv, mask)
        l_prior = F.cross_entropy(prior, played)
        v_static = tree._value(zq, wr)
        v_backed = q.max(dim=1).values
        l_tdleaf = F.huber_loss(v_static, v_backed.detach())

        l_gum = torch.zeros((), device=dev)
        if args.gumbel_n > 0 and step % 2 == 0:
            gi = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.gumbel_n)])
            rows = p_np[gi.numpy()]
            boards, all_kids, kid_of = [], [], []
            prior_g = None
            with torch.no_grad():
                sqf, mvf2, edges, mask2 = parent_batch(gi)
                elo_g = EL[gi].to(dev) if EL is not None else None
                zq2, _, _, _, hmv2, _ = field.encode(sqf, mvf2, edges, mask2, elo=elo_g)
                prior_g = field.move_logits(zq2, hmv2, mask2)
                topw = prior_g.masked_fill(~mask2, -1e9).topk(
                    min(args.gumbel_width, mask2.shape[1]), -1).indices
                for j, r in enumerate(rows):
                    b = board_from_row(tr.tok[r], tr.glob[r])
                    mvs = list(b.legal_moves)[:int(D["n_mv"][gi[j]])]
                    for t in topw[j].tolist():
                        if t < len(mvs):
                            b.push(mvs[t]); all_kids.append(b.copy(stack=False)); b.pop()
                            kid_of.append((j, t))
                if all_kids:
                    elo_k = (elo_g[[j for j, _ in kid_of]] if elo_g is not None else None)
                    zk = encode_boards(all_kids, elo_k)
                    wr_k = ~WHITE_ROOT[gi][[j for j, _ in kid_of]].to(dev)
                    vk = tree._value(zk, wr_k)                     # child-mover POV
                    qreal = torch.full_like(prior_g, float("nan"))
                    for (j, t), v in zip(kid_of, (-vk).tolist()):
                        qreal[j, t] = v
                    root_v = tree._value(zq2, WHITE_ROOT[gi].to(dev))
                    filled = torch.where(torch.isnan(qreal),
                                         root_v[:, None].expand_as(qreal), qreal)
                    # mctx qtransform: min-max normalize Q across legal moves per position
                    # so sigma spans the full range regardless of margin compression
                    big = filled.masked_fill(~mask2, float("inf"))
                    qmin = big.min(dim=1, keepdim=True).values
                    qmax = filled.masked_fill(~mask2, float("-inf")).max(
                        dim=1, keepdim=True).values
                    qn = (filled - qmin) / (qmax - qmin).clamp(min=1e-6)
                    tgt = torch.softmax(
                        (prior_g + args.sigma * qn).masked_fill(~mask2, -1e9), -1)
            prior_live = None
            sqf, mvf2, edges, mask2 = parent_batch(gi)
            zq3, _, _, _, hmv3, _ = field.encode(sqf, mvf2, edges, mask2, elo=elo_g)
            prior_live = field.move_logits(zq3, hmv3, mask2)
            l_gum = F.kl_div(F.log_softmax(prior_live.masked_fill(~mask2, -1e9), -1),
                             tgt, reduction="batchmean")

        loss = (args.w_tree * l_tree + args.w_prior * l_prior
                + args.w_tdleaf * l_tdleaf + args.w_gumbel * l_gum + vl)
        opt.zero_grad(); loss.backward(); opt.step()

        m = {}
        if step % args.eval_every == 0 or step == 1:
            field.eval(); tree.eval()
            s1 = p1 = tot = 0
            with torch.no_grad():
                for a in range(0, len(probe), 256):
                    ii = torch.from_numpy(probe[a:a + 256])
                    searched2, _, _, zq4, hmv4, mask4, _, _, _ = searched_forward(ii)
                    prior2 = field.move_logits(zq4, hmv4, mask4)
                    pl2 = D["played"][ii].long().to(dev)
                    s1 += int((searched2.argmax(-1) == pl2).sum())
                    p1 += int((prior2.argmax(-1) == pl2).sum())
                    tot += len(ii)
            el = time.time() - t0
            print(f"[gct] step {step:>5}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | tree {float(l_tree.detach()):.3f} "
                  f"prior {float(l_prior.detach()):.3f} tdleaf {float(l_tdleaf.detach()):.4f} "
                  f"gum {float(l_gum.detach()):.6f} | HELD-OUT searched-top1 {s1/tot:.3f} "
                  f"prior-top1 {p1/tot:.3f} ({(s1-p1)/tot:+.3f} from the tree)", flush=True)
            m = {"searched_top1": s1 / tot, "prior_top1": p1 / tot}
            field.train(); tree.train()
        return {"loss": float(loss.detach()), **m}

    class Both(nn.Module):
        def __init__(self):
            super().__init__()
            self.field, self.tree = field, tree
    both = Both()
    opt = torch.optim.Adam(both.parameters(), lr=args.lr)
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      start_step=0, eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=dev)
    standard_train(step_fn, both, cfg, args)


if __name__ == "__main__":
    main()
