"""Subgoal planning by search, over the learned concept space.

Kaveh 2026-08-26: "let's allow discrete codes to do their thing, and then see if we can
plan based on it... I don't need them to be interpretable yet. Let's just get the plan,
because the plan is the crux of the interpretability more than anything else."

WHY THIS SHAPE. The latent forward model does not work -- 2-ply concept prediction scores
a lift of ~0.00 against a copy baseline across every run -- so subgoals cannot be
PREDICTED. They can still be FOUND: search a few plies with real re-encoding, look at the
concept configurations the leaves actually land in, and cluster them. Each cluster is a
region of concept space reachable from here; the plan is the decision to steer into one.

WHY THIS IS LEGIBLE WITHOUT NAMED CODES. The concept layout is known even though the codes
are anonymous:

    dims    0..  96   board / global
    dims   96..1632   64 SQUARES x 24 bits   -> bit d belongs to square (d-96)//24
    dims 1632..2400   32 PIECE SLOTS x 24    -> bit d belongs to slot  (d-1632)//24
    dims 2400..2528   edge histogram

So "this subgoal flips 14 bits, 9 of them on squares d8/e8/f6" is a statement about the
board that a human can check, with no per-code semantics required. That is the sense in
which the plan, not the code, carries the interpretability.

MEASURED, not asserted:
  * does steering toward the chosen subgoal pick a DIFFERENT move than greedy?
  * PLAN PERSISTENCE -- replan one ply later; does the same subgoal survive? A "plan" that
    changes every move is not a plan, and this is the metric that says so.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch


def build(ckpt_path, arm, dev, field_only=False):
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.tools.training_infra.qat_model import QATReadout
    base = torch.load("artifacts/experiments/gcf4h_latest.pt", map_location="cpu",
                      weights_only=False)
    A = dict(base["args"])
    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=True, g_heads=32, g_codes=8,
        d_code=8, sq_codes=A.get("sq_codes", 256), mh_vq=True, gen5=True, gen6=True,
        binary=True, mv_concepts=True).to(dev).eval()
    model = QATReadout(field, arm=arm).to(dev).eval()
    if field_only:
        return model, field, 0, 0
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    miss = model.load_state_dict(ck["state_dict"], strict=False)
    return model, field, len(ck["state_dict"]), len(miss.missing_keys)


class Planner:
    def __init__(self, model, field, dev, depth=3, beam=4, n_subgoals=4, density_gamma=0.5):
        self.m, self.f, self.dev = model, field, dev
        self.depth, self.beam, self.k = depth, beam, n_subgoals
        self.gamma = density_gamma
        self.encodes = 0                      # north-star cost: evals per move

    @torch.no_grad()
    def _encode(self, boards):
        import chess
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import featurize, MV_CAP
        from catspace.research.components.encoder.approaches.reach_probability.src \
            .trajectories import normalise_elo
        acc = {k: [] for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att",
                               "vac", "dsc", "pslot")}
        for b in boards:
            g, _ = parent_graph(b)
            for k in ("sq", "mv", "mut", "att", "vac", "dsc", "pslot"):
                acc[k].append(g[k])
            acc["n_mv"].append(np.uint8(g["n_mv"])); acc["n_att"].append(np.uint8(g["n_att"]))
            acc["gl"].append(glob_of_board(b))
        R = {k: torch.from_numpy(np.stack(v)) for k, v in acc.items()}
        n = len(boards)
        bb = {k: R[k].to(self.dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(bb, self.dev, n_nodes=self.f.n_nodes)
        off = (torch.arange(n, device=self.dev) * self.f.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=self.dev)[None, :]
        for nm in ("vac", "dsc"):
            a2 = R[nm].to(self.dev).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[nm] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                         (a2 + off[:, :, None])[val])
        elo = torch.from_numpy(normalise_elo(np.full(n, 2750., np.float32)))[:, None] \
            .repeat(1, 2).to(self.dev)
        zq, _, _, _, h_mv, _ = self.f.encode(sqf, mvf, edges, mask, elo=elo,
                                             pslot=R["pslot"].to(self.dev))[:6]
        self.encodes += n
        return zq, h_mv

    @torch.no_grad()
    def score_children(self, board):
        """-> (legal moves, per-move value W-L). ONE pass scores every child."""
        import chess
        z, h = self._encode([board])
        lg = self.m(z, h)[0][0]
        p = torch.softmax(lg, -1)
        val = (p[:, 0] - p[:, 2]).cpu().numpy()
        mv = list(board.legal_moves)
        return mv, val[:len(mv)], z[0].cpu().numpy()

    @torch.no_grad()
    def search(self, board):
        """Beam search to self.depth. -> list of (move_path, leaf_board, leaf_z, value)."""
        mv, val, z0 = self.score_children(board)
        if not mv:
            return [], z0
        order = np.argsort(-val)[:self.beam]
        frontier = []
        for i in order:
            b = board.copy(); b.push(mv[i])
            frontier.append(([mv[i]], b, float(val[i])))
        for _ in range(self.depth - 1):
            nxt = []
            for path, b, v in frontier:
                if b.is_game_over():
                    nxt.append((path, b, v)); continue
                m2, v2, _ = self.score_children(b)
                if not m2:
                    nxt.append((path, b, v)); continue
                # the child is scored from the OPPONENT's side to move, so their gain is
                # our loss -- negate, exactly as a minimax would
                for j in np.argsort(v2)[:self.beam]:
                    bb = b.copy(); bb.push(m2[j])
                    nxt.append((path + [m2[j]], bb, -float(v2[j])))
            frontier = sorted(nxt, key=lambda t: -t[2])[:self.beam * self.beam]
        leaves = []
        for path, b, v in frontier:
            z, _ = self._encode([b])
            leaves.append((path, b, z[0].cpu().numpy(), v))
        return leaves, z0

    def subgoals(self, board):
        """Cluster the reachable leaf configurations. Each cluster is a candidate subgoal."""
        leaves, z0 = self.search(board)
        if len(leaves) < 2:
            return None, [], z0
        Z = np.stack([l[2] for l in leaves])
        B = (Z > 0).astype(np.float32)
        k = min(self.k, len(leaves))
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(B)
        out = []
        for c in range(k):
            idx = np.nonzero(lab == c)[0]
            if not len(idx):
                continue
            vals = np.array([leaves[i][3] for i in idx])
            cen = B[idx].mean(0)
            # density prior: a subgoal in a well-populated region is more trustworthy
            score = float(vals.mean()) * (len(idx) / len(leaves)) ** self.gamma
            out.append({"cluster": c, "n": len(idx), "value": float(vals.mean()),
                        "score": score, "centroid": cen,
                        "paths": [leaves[i][0] for i in idx],
                        "best_path": leaves[int(idx[np.argmax(vals)])][0]})
        out.sort(key=lambda d: -d["score"])
        return out[0] if out else None, out, z0


def describe(centroid, z0):
    """Which SQUARES and PIECE SLOTS does this subgoal change? Legible without code names."""
    cur = (z0 > 0).astype(np.float32)
    diff = np.abs(centroid - cur)
    flip = np.nonzero(diff > 0.5)[0]
    import chess
    sq, pc, other = {}, {}, 0
    for d in flip:
        if 96 <= d < 1632:
            sq[(d - 96) // 24] = sq.get((d - 96) // 24, 0) + 1
        elif 1632 <= d < 2400:
            pc[(d - 1632) // 24] = pc.get((d - 1632) // 24, 0) + 1
        else:
            other += 1
    top_sq = sorted(sq.items(), key=lambda kv: -kv[1])[:6]
    return {"n_bits": int(len(flip)),
            "squares": [(chess.square_name(int(s)), int(n)) for s, n in top_sq],
            "sq_set": set(int(s) for s in sq),          # ALL changed squares, for overlap
            "n_squares": len(sq), "n_pieces": len(pc), "other_bits": other}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/qat2_latest.pt")
    ap.add_argument("--arm", default="qat")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--subgoals", type=int, default=4)
    ap.add_argument("--positions", type=int, default=12)
    args = ap.parse_args()
    import chess
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    dev = resolve_device("auto")
    model, field, nt, nm = build(args.ckpt, args.arm, dev)
    print(f"  loaded {args.ckpt}: {nt} tensors, {nm} missing\n")

    import glob as _g
    from catspace.research.tools.npz_io import load_npz
    fens = []
    for f in sorted(_g.glob("data/derived/gko1_shards/shard_*.npz"))[:4]:
        z = load_npz(f)
        if "wdl" in z.files and (z["kind"] == 0).any():
            fens += [str(x) for x in np.asarray(z["fen"])[:60]]
        if len(fens) >= args.positions * 3:
            break

    P = Planner(model, field, dev, args.depth, args.beam, args.subgoals)
    agree = same_goal = n = 0
    lifts = []
    for fen in fens:
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        if b.is_game_over() or not list(b.legal_moves):
            continue
        e0 = P.encodes
        best, allg, z0 = P.subgoals(b)
        if best is None:
            continue
        d = describe(best["centroid"], z0)
        mv, val, _ = P.score_children(b)
        greedy = mv[int(np.argmax(val))]
        planned = best["best_path"][0]
        agree += int(greedy == planned)
        # PLAN PERSISTENCE: play the planned move, replan, is the subgoal the same region?
        b2 = b.copy(); b2.push(planned)
        keep = "--"
        if not b2.is_game_over():
            n2, _, z2 = P.subgoals(b2)
            if n2 is not None:
                d2 = describe(n2["centroid"], z0)
                # PERSISTENCE MUST BE SCORED AGAINST CHANCE. Raw overlap of two square
                # sets rises with their size, so a subgoal that touches 60 of 64 squares
                # "persists" trivially. Measured: raw overlap correlated +0.595 with how
                # DIFFUSE the subgoal was -- i.e. it was scoring size, not persistence.
                A, Bs = d["sq_set"], d2["sq_set"]
                exp = len(A) * len(Bs) / 64.0                  # overlap expected by chance
                obs = len(A & Bs)
                lift = obs / exp if exp > 1e-9 else 0.0
                keep = f"{lift:.2f}x"
                same_goal += int(lift > 1.5)                   # beats chance by half again
                lifts.append(lift)
        n += 1
        print(f"  {fen[:36]:38s} plan {planned.uci():6s} greedy {greedy.uci():6s} "
              f"| subgoal v={best['value']:+.3f} n={best['n']:2d} "
              f"flips {d['n_bits']:4d} bits over {d['n_squares']:2d} sq "
              f"{[s for s,_ in d['squares'][:3]]} | persist {keep} "
              f"| {P.encodes-e0:3d} encodes")
    if n:
        print(f"\n  positions planned            : {n}")
        print(f"  plan move == greedy move     : {100*agree/n:.0f}%  "
              f"(if 100%, the subgoal layer changes nothing)")
        import numpy as _np
        L = _np.array(lifts) if lifts else _np.zeros(1)
        print(f"  subgoal overlap vs CHANCE    : {L.mean():.2f}x  (1.00 = the next plan's "
              f"squares are no more related than random)")
        print(f"  positions beating chance 1.5x: {100*same_goal/n:.0f}%")
        print(f"  encodes per move             : {P.encodes/max(n,1):.1f}")


if __name__ == "__main__":
    main()
