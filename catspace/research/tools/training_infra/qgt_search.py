"""qgt_search.py -- search over the LATENT DYNAMICS, and its arena (Kaveh
2026-08-28: "put this to the test in the arena. search should be via the dynamics
operators. build appropriate abstract classes to step dynamics in the latent space
for a search that can take in any algorithm... we also want to time it in the
arena").

Doctrine embodied here: V is a function of POSITION (concepts); a move's worth is
V(step(s, m)) -- derived, never attached to the move. The board supplies the RULES
(legal moves, game termination); the latents supply the FUTURE (candidate
evaluation). One `step` = one MOVE unit: the mover's move with the reply
marginalised, landing when it is the mover's turn again -- exactly the trained
semantics.

    LatentDynamics    the abstract dynamics: encode / root_moves / step / value
    QGTDynamics       a qgt checkpoint (discrete or graph arm) behind that ABC
    SearchAlgorithm   choose(dyn, board, rng) -> move; ANY algorithm plugs in
    Greedy1           depth-1 latent search: argmax_m V(step(s, m)), mover-relative
    latent_arena      games vs Maia (lc0 at nodes=1 -- the calibrated benchmark),
                      alternating colors, mirrored random openings, claim-draw rules,
                      per-move WALL-CLOCK timing (under time control, speed is Elo)

    .venv/bin/python -m catspace.research.tools.training_infra.qgt_search \
        --ckpt artifacts/experiments/qgt7_step8000.pt --games 100
"""
from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import chess
import numpy as np
import torch


# ---------------------------------------------------------------------------------
class LatentDynamics(ABC):
    """Step dynamics in latent space. The board appears ONLY in encode/root_moves."""

    @abstractmethod
    def encode(self, board: chess.Board):
        """board -> (state, ctx): state is the latent the search rolls; ctx carries
        whatever root_moves needs (move-node features)."""

    @abstractmethod
    def root_moves(self, board: chess.Board, ctx):
        """-> (moves list, codes, frm, to): the legal moves the dynamics can
        condition on, with their latent move codes and injection squares."""

    @abstractmethod
    def step(self, state, codes, frm, to):
        """Batched: one MOVE unit per row. state (n,...) -> next state (n,...)."""

    @abstractmethod
    def value(self, state, stm):
        """-> (n,3) White-POV W/D/L probabilities read from the state."""


class SearchAlgorithm(ABC):
    @abstractmethod
    def choose(self, dyn: LatentDynamics, board: chess.Board,
               rng: np.random.Generator):
        """-> (move, info dict). info must include 'evals' (positions evaluated)."""


# ---------------------------------------------------------------------------------
class QGTDynamics(LatentDynamics):
    """A qgt checkpoint (discrete or graph arm) as a LatentDynamics."""

    def __init__(self, ckpt_path: str, device: str = "auto"):
        from catspace.research.tools.training_infra.train.scaffold import (
            resolve_device)
        from catspace.research.tools.training_infra.qgt_field import (
            QGTField, BitTransition, GraphBitTransition)
        self.dev = resolve_device(device)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        A = ck["args"]
        assert A.get("dynamics") in ("discrete", "graph"), \
            f"{ckpt_path}: no bit transition (dynamics={A.get('dynamics')})"
        self.graph = A["dynamics"] == "graph"
        m = QGTField(d=A["d"], layers=A["layers"], sq_bits=A["sq_bits"],
                     rvq_levels=A["rvq_levels"], mv_bits=A["mv_bits"],
                     qa_levels=A["qa_levels"], n_q=A["n_q"],
                     qa_heads=A["attn_heads"], top_k=A["top_k"],
                     out_bits=A["out_bits"])
        cls = GraphBitTransition if self.graph else BitTransition
        m.bit_trans = cls(sq_bits=A["sq_bits"], mv_bits=A["mv_bits"], n_q=A["n_q"],
                          heads=A["attn_heads"], top_k=A["top_k"],
                          out_bits=A["out_bits"],
                          w_id_init=A.get("w_id_init", 0.5))
        missing = m.load_state_dict(ck["state_dict"], strict=False)
        assert not missing.missing_keys, f"missing: {missing.missing_keys[:4]}"
        self.model = m.to(self.dev).eval()
        self.step_count = int(ck.get("step", 0))
        self.ckpt = ckpt_path

    @torch.no_grad()
    def _feats(self, board: chess.Board):
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import featurize, MV_CAP
        g, _ = parent_graph(board)
        R = {k: torch.from_numpy(np.stack([g[k]])) for k in
             ("sq", "mv", "mut", "att", "vac", "dsc")}
        R["gl"] = torch.from_numpy(np.stack([glob_of_board(board)]))
        R["n_mv"] = torch.tensor([np.uint8(g["n_mv"])])
        R["n_att"] = torch.tensor([int(g["n_att"])])
        b = {k: R[k].to(self.dev) for k in ("sq", "gl", "mv", "mut", "att",
                                            "n_mv", "n_att")}
        n_nodes = 64 + MV_CAP
        sqf, mvf, edges, mask = featurize(b, self.dev, n_nodes=n_nodes)
        off = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        jj = torch.arange(MV_CAP, device=self.dev)[None, :]
        for name in ("vac", "dsc"):
            a2 = R[name].to(self.dev).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        return sqf, mvf, edges, mask

    @torch.no_grad()
    def encode(self, board: chess.Board):
        sqf, mvf, edges, mask = self._feats(board)
        c, h_mv, _ = self.model.carrier(sqf, mvf, edges, mask)
        bits = self.model.rvq_bits(c)[0]                   # (1,64,sq_bits)
        return bits, h_mv

    @torch.no_grad()
    def prior_logits(self, board: chess.Board):
        """The AMORTIZED head: the model's own per-move logits (position read +
        move bits), exactly the trainer's eval semantics. The arena's ceiling
        control -- NOT the canonical value pathway."""
        sqf, mvf, edges, mask = self._feats(board)
        stm = torch.tensor([1 if board.turn else 0], device=self.dev)
        logit = self.model(sqf, mvf, edges, mask, stm=stm)[0]
        return logit[0]                                    # (MV_CAP, 3)

    @torch.no_grad()
    def root_moves(self, board: chess.Board, ctx):
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import MV_CAP
        h_mv = ctx
        legal = list(board.legal_moves)[:MV_CAP]           # move-node order
        dq = self.model.dyn_mv_lfq(self.model.dyn_mv_proj(h_mv))[0][0]  # (MV,32)
        n = len(legal)
        codes = dq[:n]
        frm = torch.tensor([m.from_square for m in legal], device=self.dev)
        to = torch.tensor([m.to_square for m in legal], device=self.dev)
        return legal, codes, frm, to

    @torch.no_grad()
    def step(self, state, codes, frm, to):
        bt = self.model.bit_trans
        if self.graph:
            lg, _, _ = bt(state, codes, frm, to)
        else:
            lg, _ = bt(state, codes)
        zb = torch.sign(lg)
        return torch.where(zb == 0, torch.ones_like(zb), zb)

    @torch.no_grad()
    def value(self, state, stm: int):
        z = self.model.levels_from_bits(state)[0]
        W = self.model.readout.weight
        pv = torch.nn.functional.linear(z, W[:, :self.model.d_pos],
                                        self.model.readout.bias)
        pv = pv + self.model.turn[torch.full((len(z),), int(stm), dtype=torch.long,
                                             device=self.dev)]
        return torch.softmax(pv, -1)


class Greedy1(SearchAlgorithm):
    """Depth-1 latent search: one MOVE-unit step per candidate, argmax of the
    mover-relative read. The minimal honest test of the dynamics operators."""

    def choose(self, dyn: LatentDynamics, board: chess.Board,
               rng: np.random.Generator):
        state, ctx = dyn.encode(board)
        legal, codes, frm, to = dyn.root_moves(board, ctx)
        if not legal:
            raise ValueError("no legal moves")
        n = len(legal)
        base = state.expand(n, -1, -1)
        nxt = dyn.step(base, codes, frm, to)
        stm = 1 if board.turn else 0                       # landing stm == source stm
        pv = dyn.value(nxt, stm).cpu().numpy()
        sgn = 1.0 if board.turn else -1.0
        sc = sgn * (pv[:, 0] - pv[:, 2]) + rng.random(n) * 1e-9
        j = int(sc.argmax())
        return legal[j], {"evals": n, "score": float(sc[j])}


class PriorPolicy(SearchAlgorithm):
    """The amortized readout head at the root, no rolling -- the ceiling control.
    Offline instruments: top1 0.324 / regret 0.107 on the qgt7 checkpoint."""

    def choose(self, dyn, board: chess.Board, rng: np.random.Generator):
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import MV_CAP
        legal = list(board.legal_moves)[:MV_CAP]
        lg = dyn.prior_logits(board).cpu().numpy()
        sgn = 1.0 if board.turn else -1.0
        sc = sgn * (lg[:len(legal), 0] - lg[:len(legal), 2]) \
            + rng.random(len(legal)) * 1e-9
        j = int(sc.argmax())
        return legal[j], {"evals": 1, "score": float(sc[j])}


class RandomPolicy(SearchAlgorithm):
    """The floor control."""

    def choose(self, dyn, board, rng):
        legal = list(board.legal_moves)
        return legal[int(rng.integers(len(legal)))], {"evals": 0}


# ---------------------------------------------------------------------------------
@dataclass
class ArenaSummary:
    games: int = 0
    w: int = 0
    d: int = 0
    l: int = 0
    ms: list = field(default_factory=list)
    evals: int = 0
    our_moves: int = 0

    def score(self):
        return (self.w + 0.5 * self.d) / max(self.games, 1)

    def wilson(self):
        n, p = self.games, self.score()
        if n == 0:
            return (0.0, 1.0)
        z = 1.96
        den = 1 + z * z / n
        mid = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return (mid - half, mid + half)

    def line(self, name):
        lo, hi = self.wilson()
        ms = np.array(self.ms) if self.ms else np.array([np.nan])
        return (f"{name}: {self.score():.3f} [{lo:.3f},{hi:.3f}] "
                f"(+{self.w}={self.d}-{self.l}/{self.games}) | "
                f"our move: mean {np.mean(ms):.0f}ms median "
                f"{np.median(ms):.0f}ms p90 {np.percentile(ms, 90):.0f}ms | "
                f"{self.evals / max(self.our_moves, 1):.1f} step-evals/move")


def latent_arena(algo: SearchAlgorithm, dyn, maia: str, games: int,
                 opening_plies: int = 2, cap: int = 300, seed: int = 0,
                 tag: str = "latent"):
    """Alternating colors; game pair 2i/2i+1 shares a mirrored random opening.
    Maia at nodes=1 (its calibrated rating). Claim-draw rules on; ply cap = draw."""
    from catspace.research.tools.chess_specific.uci import UCIBoardPolicy
    s = ArenaSummary()
    rng = np.random.default_rng(seed)
    openings = {}
    with UCIBoardPolicy(cmd=["lc0", f"--weights={maia}"], movetime=None,
                        nodes=1) as opp:
        for gi in range(games):
            pair, our_color = gi // 2, (gi % 2 == 0)      # True = we are White
            if pair not in openings:
                ob = chess.Board()
                orng = np.random.default_rng(10_000 + pair)
                for _ in range(opening_plies):
                    ms = list(ob.legal_moves)
                    ob.push(ms[int(orng.integers(len(ms)))])
                openings[pair] = [m.uci() for m in ob.move_stack]
            board = chess.Board()
            for u in openings[pair]:
                board.push_uci(u)
            while (not board.is_game_over(claim_draw=True)
                   and board.ply() < cap):
                if board.turn == our_color:
                    t0 = time.perf_counter()
                    mv, info = algo.choose(dyn, board, rng)
                    s.ms.append(1000 * (time.perf_counter() - t0))
                    s.evals += info.get("evals", 0)
                    s.our_moves += 1
                else:
                    mv = opp.move(board, rng)
                board.push(mv)
            res = board.result(claim_draw=True)
            if res == "*":
                res = "1/2-1/2"                            # cap adjudication
            s.games += 1
            if res == "1/2-1/2":
                s.d += 1
            elif (res == "1-0") == our_color:
                s.w += 1
            else:
                s.l += 1
            if (gi + 1) % 10 == 0:
                print(f"[arena:{tag}] {s.line(tag)}", flush=True)
    print(f"[arena:{tag}] FINAL {s.line(tag)}", flush=True)
    return s


# ---------------------------------------------------------------------------------
def _units(args):
    dyn = QGTDynamics(args.ckpt)
    b = chess.Board()
    state, ctx = dyn.encode(b)
    assert state.shape[1:] == (64, dyn.model.sq_bits)
    u = set(torch.unique(state).tolist())
    assert u <= {-1.0, 1.0}, "encoded state not bits"
    legal, codes, frm, to = dyn.root_moves(b, ctx)
    assert len(legal) == 20 and codes.shape == (20, dyn.model.mv_bits)
    assert int(frm.max()) < 64 and int(to.max()) < 64
    # batched step == singleton steps, exactly
    nxt = dyn.step(state.expand(20, -1, -1), codes, frm, to)
    for j in (0, 7, 19):
        one = dyn.step(state, codes[j:j + 1], frm[j:j + 1], to[j:j + 1])
        assert bool((one[0] == nxt[j]).all()), f"batched != singleton at {j}"
    assert set(torch.unique(nxt).tolist()) <= {-1.0, 1.0}
    pv = dyn.value(nxt, 1)
    assert pv.shape == (20, 3) and torch.isfinite(pv).all()
    assert torch.allclose(pv.sum(-1), torch.ones(20, device=pv.device), atol=1e-5)
    # a full game against the random floor completes with timing recorded
    g1 = Greedy1()
    rng = np.random.default_rng(0)
    board = chess.Board()
    t0 = time.perf_counter()
    n_mv = 0
    while not board.is_game_over(claim_draw=True) and board.ply() < 60:
        if board.turn:
            mv, info = g1.choose(dyn, board, rng)
            n_mv += 1
        else:
            ms = list(board.legal_moves)
            mv = ms[int(rng.integers(len(ms)))]
        board.push(mv)
    dt = (time.perf_counter() - t0) / max(n_mv, 1) * 1000
    print(f"  S1 PASS encode/root_moves/step/value shapes + exact batch=singleton; "
          f"bits stay bits; 30-move game vs random OK at ~{dt:.0f}ms per our move")
    print("  ALL QGT SEARCH GATES PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/qgt7_step8000.pt")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--control-games", type=int, default=20,
                    help="random-policy floor games (0 to skip)")
    ap.add_argument("--maia", default="")
    ap.add_argument("--opening-plies", type=int, default=2)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--algo", choices=("greedy1", "prior"), default="greedy1")
    ap.add_argument("--units", action="store_true")
    args = ap.parse_args()
    if args.units:
        return _units(args)
    if not args.maia:
        from catspace.io import paths
        args.maia = paths.engine("maia/maia-1100.pb.gz")
    dyn = QGTDynamics(args.ckpt)
    print(f"[arena] dynamics: {args.ckpt} (step {dyn.step_count}, "
          f"{'graph' if dyn.graph else 'discrete'} arm) vs {args.maia} @ nodes=1",
          flush=True)
    out = {}
    if args.control_games > 0:
        c = latent_arena(RandomPolicy(), dyn, args.maia, args.control_games,
                         args.opening_plies, args.cap, args.seed + 1, tag="random")
        out["random"] = {"score": c.score(), "wilson": c.wilson(), "n": c.games}
    algo = Greedy1() if args.algo == "greedy1" else PriorPolicy()
    s = latent_arena(algo, dyn, args.maia, args.games, args.opening_plies,
                     args.cap, args.seed, tag=args.algo)
    ms = np.array(s.ms)
    out[args.algo] = {"score": s.score(), "wilson": s.wilson(), "n": s.games,
                      "w": s.w, "d": s.d, "l": s.l,
                      "ms_mean": float(ms.mean()), "ms_median": float(np.median(ms)),
                      "ms_p90": float(np.percentile(ms, 90)),
                      "evals_per_move": s.evals / max(s.our_moves, 1),
                      "ckpt": args.ckpt, "maia": args.maia, "seed": args.seed}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[arena] results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
