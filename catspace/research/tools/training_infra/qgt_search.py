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
    def step(self, state, codes, frm, to, frm2=None, to2=None):
        """Batched: one MOVE unit per row. state (n,...) -> next state (n,...).
        frm2/to2 (n,) long: the REPLY's squares for T2-conditional checkpoints
        (Kaveh 8/29) -- search supplies the reply, the transition never guesses."""

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
        assert A.get("dynamics") in ("discrete", "graph", "canon"), \
            f"{ckpt_path}: no bit transition (dynamics={A.get('dynamics')})"
        self.graph = A["dynamics"] in ("graph", "canon")
        m = QGTField(d=A["d"], layers=A["layers"], sq_bits=A["sq_bits"],
                     rvq_levels=A["rvq_levels"], mv_bits=A["mv_bits"],
                     qa_levels=A["qa_levels"], n_q=A["n_q"],
                     qa_heads=A["attn_heads"], top_k=A["top_k"],
                     out_bits=A["out_bits"])
        cls = GraphBitTransition if self.graph else BitTransition
        kw = dict(sq_bits=A["sq_bits"], mv_bits=A["mv_bits"], n_q=A["n_q"],
                  heads=A["attn_heads"], top_k=A["top_k"],
                  out_bits=A["out_bits"],
                  w_id_init=A.get("w_id_init", 0.5))
        if self.graph:
            kw.update(d=A.get("dyn_d", 64), layers=A.get("dyn_layers", 2),
                      qa_continuous=A.get("dynamics") == "canon",
                      edge_mode=A.get("edge_mode", "question"))
        m.bit_trans = cls(**kw)
        # a T2-conditional checkpoint carries reply markers; legacy ones don't
        # (their random-init markers are never injected: step() gets no frm2)
        self.canon = A.get("dynamics") == "canon"
        self.conditional = ("bit_trans.rep_from" in ck["state_dict"]
                            and not self.canon)   # canon never injects replies
        if self.canon:
            from catspace.research.tools.training_infra.qgt_canon import (
                MIRROR_PERM)
            self.PERM = torch.from_numpy(MIRROR_PERM).to(self.dev)
        if "mat_head.weight" in ck["state_dict"]:
            m.mat_head = torch.nn.Linear(m.d_pos, 1)
        if "value_head.weight" in ck["state_dict"]:
            m.value_head = torch.nn.Linear(m.d_pos, 3)
        if "sib_scorer.weight" in ck["state_dict"]:
            m.sib_scorer = torch.nn.Linear(
                ck["state_dict"]["sib_scorer.weight"].shape[1], 1)
        missing = m.load_state_dict(ck["state_dict"], strict=False)
        real_missing = [k for k in missing.missing_keys
                        if not k.startswith("bit_trans.rep_")]
        assert not real_missing, f"missing: {real_missing[:4]}"
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
        """Canon ckpts: encode the CANONICAL board (mover always White); ctx
        carries (h_mv, flipped) so root_moves can map moves back."""
        if self.canon:
            from catspace.research.tools.training_infra.qgt_canon import canon
            cb, fl = canon(board)
            sqf, mvf, edges, mask = self._feats(cb)
            c, h_mv, _ = self.model.carrier(sqf, mvf, edges, mask)
            return self.model.rvq_bits(c)[0], (h_mv, fl)
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
    def node_info(self, board: chess.Board):
        """ONE featurize+carrier pass -> (bits, legal, codes, frm, to, prior).
        What _mknode needs; encode()+prior_logits() would pay the pass twice."""
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import MV_CAP
        sqf, mvf, edges, mask = self._feats(board)
        stm = torch.tensor([1 if board.turn else 0], device=self.dev)
        logit, _, _, aux = self.model(sqf, mvf, edges, mask, stm=stm)
        bits = self.model.rvq_bits(aux["carrier"])[0]
        legal = list(board.legal_moves)[:MV_CAP]
        dq = self.model.dyn_mv_lfq(self.model.dyn_mv_proj(aux["h_mv"]))[0][0]
        n = len(legal)
        frm = torch.tensor([m.from_square for m in legal], device=self.dev)
        to = torch.tensor([m.to_square for m in legal], device=self.dev)
        sg = 1.0 if board.turn else -1.0
        pri = torch.softmax(sg * (logit[0, :n, 0] - logit[0, :n, 2]), 0)
        return bits, legal, dq[:n], frm, to, pri.cpu().numpy()

    @torch.no_grad()
    def root_moves(self, board: chess.Board, ctx):
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import MV_CAP
        if self.canon:
            from catspace.research.tools.training_infra.qgt_canon import (
                canon, from_canon)
            h_mv, fl = ctx
            cb, _ = canon(board)
            legal_c = list(cb.legal_moves)[:MV_CAP]       # canonical order
            dq = self.model.dyn_mv_lfq(self.model.dyn_mv_proj(h_mv))[0][0]
            n = len(legal_c)
            frm = torch.tensor([m.from_square for m in legal_c], device=self.dev)
            to = torch.tensor([m.to_square for m in legal_c], device=self.dev)
            legal = [from_canon(m, fl) for m in legal_c]  # REAL moves, same order
            return legal, dq[:n], frm, to
        h_mv = ctx
        legal = list(board.legal_moves)[:MV_CAP]           # move-node order
        dq = self.model.dyn_mv_lfq(self.model.dyn_mv_proj(h_mv))[0][0]  # (MV,32)
        n = len(legal)
        codes = dq[:n]
        frm = torch.tensor([m.from_square for m in legal], device=self.dev)
        to = torch.tensor([m.to_square for m in legal], device=self.dev)
        return legal, codes, frm, to

    @torch.no_grad()
    def step(self, state, codes, frm, to, frm2=None, to2=None):
        bt = self.model.bit_trans
        if self.graph:
            lg, _, _ = bt(state, codes, frm, to, frm2=frm2, to2=to2)
        else:
            lg, _ = bt(state, codes)
        zb = torch.sign(lg)
        zb = torch.where(zb == 0, torch.ones_like(zb), zb)
        if self.canon:                     # ONE ply; the frame mirrors after it
            zb = zb[:, self.PERM]
        return zb

    @torch.no_grad()
    def material(self, state):
        """White-POV material estimate (pawns) from mat_head; None if the
        checkpoint predates material grounding (qgt12)."""
        if not hasattr(self.model, "mat_head"):
            return None
        z = self.model.levels_from_bits(state)[0]
        return self.model.mat_head(z).squeeze(-1) * 10.0

    @torch.no_grad()
    def value(self, state, stm: int):
        z = self.model.levels_from_bits(state)[0]
        if hasattr(self.model, "value_head"):     # the DEDICATED head (qgt16+)
            pv = self.model.value_head(z)
        else:
            W = self.model.readout.weight
            pv = torch.nn.functional.linear(z, W[:, :self.model.d_pos],
                                            self.model.readout.bias)
        pv = pv + self.model.turn[torch.full((len(z),), int(stm), dtype=torch.long,
                                             device=self.dev)]
        return torch.softmax(pv, -1)


class Greedy1(SearchAlgorithm):
    """Depth-1 latent search: one MOVE-unit step per candidate, argmax of the
    mover-relative read. The minimal honest test of the dynamics operators.

    T2-CONDITIONAL checkpoints (Kaveh 8/29): the move unit needs the reply as an
    INPUT, so each candidate is scored as the MIN of the read over every legal
    reply -- the refutation is enumerated and backed up, never marginalized. The
    real board supplies reply LEGALITY only; every evaluation stays latent."""

    def __init__(self, mat_lambda=0.5):
        self.lam = mat_lambda

    def choose(self, dyn: LatentDynamics, board: chess.Board,
               rng: np.random.Generator):
        state, ctx = dyn.encode(board)
        legal, codes, frm, to = dyn.root_moves(board, ctx)
        if not legal:
            raise ValueError("no legal moves")
        n = len(legal)
        stm = 1 if board.turn else 0                       # landing stm == source stm
        sgn = 1.0 if board.turn else -1.0
        if getattr(dyn, "canon", False):
            # depth-2 negamax: my ply, then every legal reply as ITS OWN 1-ply
            # step in the child's canonical frame; after two plies the frame is
            # mine again, the read is mover-POV directly. Board = legality only.
            sc = np.full(n, np.nan)
            evals = 0
            for j, mv in enumerate(legal):
                z1 = dyn.step(state, codes[j:j + 1], frm[j:j + 1], to[j:j + 1])
                b1 = board.copy(stack=False)
                b1.push(mv)
                reps1, cd1, f1, t1 = dyn.root_moves(b1, dyn.encode(b1)[1])
                if not reps1:
                    sc[j] = 2.0 if b1.is_checkmate() else 0.0
                    continue
                nr = len(reps1)
                z2 = dyn.step(z1.expand(nr, -1, -1), cd1, f1, t1)
                evals += nr
                pv = dyn.value(z2, 1).cpu().numpy()        # my POV again
                es = pv[:, 0] - pv[:, 2]
                mat = dyn.material(z2)
                if mat is not None and self.lam > 0:
                    es = es + self.lam * np.tanh(mat.cpu().numpy() / 5.0)
                sc[j] = float(es.min())                    # opponent's best shot
            sc = sc + rng.random(n) * 1e-9
            j = int(np.nanargmax(sc))
            return legal[j], {"evals": evals, "score": float(sc[j])}
        if not getattr(dyn, "conditional", False):
            base = state.expand(n, -1, -1)
            nxt = dyn.step(base, codes, frm, to)
            pv = dyn.value(nxt, stm).cpu().numpy()
            sc = sgn * (pv[:, 0] - pv[:, 2]) + rng.random(n) * 1e-9
            j = int(sc.argmax())
            return legal[j], {"evals": n, "score": float(sc[j])}
        rows_j, f2, t2, term = [], [], [], np.full(n, np.nan)
        for j, mv in enumerate(legal):
            b1 = board.copy(stack=False); b1.push(mv)
            reps = list(b1.legal_moves)
            if not reps:                       # mate delivered > any read; stalemate = 0
                term[j] = 2.0 if b1.is_checkmate() else 0.0
                continue
            for r in reps:
                rows_j.append(j); f2.append(r.from_square); t2.append(r.to_square)
        sc = term.copy()
        if rows_j:
            ridx = torch.tensor(rows_j, device=frm.device)
            nxt = dyn.step(state.expand(len(rows_j), -1, -1), codes[ridx],
                           frm[ridx], to[ridx],
                           frm2=torch.tensor(f2, device=frm.device),
                           to2=torch.tensor(t2, device=frm.device))
            pv = dyn.value(nxt, stm).cpu().numpy()
            wp = sgn * (pv[:, 0] - pv[:, 2])
            rj = np.asarray(rows_j)
            for j in range(n):
                mine = wp[rj == j]
                if len(mine):
                    sc[j] = float(mine.min())              # the opponent's best shot
        sc = sc + rng.random(n) * 1e-9
        j = int(np.nanargmax(sc))
        return legal[j], {"evals": len(rows_j), "score": float(sc[j])}


class MCTS(SearchAlgorithm):
    """PUCT over the learned dynamics (roadmap step 2, Kaveh 8/29: "I can just do
    regular MCTS over this learned model to discover tactics").

    Tree alternates my-move and reply levels; a T2-conditional step fires when
    both are chosen, so every stored state is a my-turn state rolled purely in
    LATENT space. The real board supplies legality, terminals, and move features
    for codes (the contract Greedy1 already uses) -- never the state. Edge
    expansion picks the reply minimizing the mover's read (greedy opponent inside
    the edge). Leaf score = White-POV (P(W)-P(L)) + mat_lambda*tanh(material/5)
    when the checkpoint carries a material head (qgt12+)."""

    def __init__(self, nodes=128, c_puct=1.5, mat_lambda=0.5):
        self.N, self.c, self.lam = nodes, c_puct, mat_lambda

    def _score(self, dyn, state, stm):
        return float(self._score_batch(dyn, state, stm)[0])

    def _score_batch(self, dyn, states, stm):
        """(n,...) states -> (n,) White-POV scalars, one batched read."""
        pv = dyn.value(states, stm)
        s = (pv[:, 0] - pv[:, 2])
        mat = dyn.material(states)
        if mat is not None and self.lam > 0:
            s = s + self.lam * torch.tanh(mat / 5.0)
        return s.cpu().numpy()

    def _mknode(self, dyn, z, board):
        if not any(True for _ in board.legal_moves):
            v = (1.0 if board.is_checkmate() and not board.turn else
                 -1.0 if board.is_checkmate() else 0.0)
            return dict(term=v)
        _, lg, cd, fb, tb, pri = dyn.node_info(board)         # one pass
        return dict(z=z, board=board, legal=lg, codes=cd, frm=fb, to=tb,
                    pri=pri, n=np.zeros(len(lg)), w=np.zeros(len(lg)),
                    kids={}, term=None)

    def choose(self, dyn: LatentDynamics, board: chess.Board,
               rng: np.random.Generator):
        import math
        state, _ = dyn.encode(board)
        root = self._mknode(dyn, state, board)
        if root["term"] is not None:
            raise ValueError("no legal moves")
        evals = 0
        for _ in range(self.N):
            path, node, val = [], root, 0.0
            while True:
                if node["term"] is not None:
                    val = node["term"]
                    break
                nsum = node["n"].sum() + 1.0
                sg = 1.0 if node["board"].turn else -1.0
                q = np.where(node["n"] > 0,
                             node["w"] / np.maximum(node["n"], 1), 0.0) * sg
                u = self.c * node["pri"] * math.sqrt(nsum) / (1.0 + node["n"])
                a = int((q + u + rng.random(len(q)) * 1e-9).argmax())
                path.append((node, a))
                if a in node["kids"]:
                    node = node["kids"][a]
                    continue
                mv = node["legal"][a]                         # EXPAND edge a
                b1 = node["board"].copy(stack=False)
                b1.push(mv)
                reps = list(b1.legal_moves)
                if not reps:
                    val = (1.0 if b1.is_checkmate() and not b1.turn else
                           -1.0 if b1.is_checkmate() else 0.0)
                    node["kids"][a] = dict(term=val)
                    break
                if len(reps) > 20:              # loud replies + 8 quiet: the min
                    loud = [r for r in reps     # is almost always a refutation
                            if b1.is_capture(r) or b1.gives_check(r)]
                    quiet = [r for r in reps if r not in loud]
                    reps = (loud + quiet[:max(8, 20 - len(loud))])[:24]
                nr = len(reps)
                f2 = torch.tensor([r.from_square for r in reps],
                                  device=node["frm"].device)
                t2 = torch.tensor([r.to_square for r in reps],
                                  device=node["frm"].device)
                stm1 = 1 if node["board"].turn else 0
                if getattr(dyn, "conditional", False):
                    nx = dyn.step(node["z"].expand(nr, -1, -1),
                                  node["codes"][a].expand(nr, -1),
                                  node["frm"][a].expand(nr),
                                  node["to"][a].expand(nr), frm2=f2, to2=t2)
                    evals += nr
                    sc = self._score_batch(dyn, nx, stm1)
                    ri = int(sc.argmin() if node["board"].turn else sc.argmax())
                    zc, val = nx[ri:ri + 1], float(sc[ri])
                    b2 = b1.copy(stack=False)
                    b2.push(reps[ri])
                else:
                    zc = dyn.step(node["z"], node["codes"][a:a + 1],
                                  node["frm"][a:a + 1], node["to"][a:a + 1])
                    evals += 1
                    val = self._score(dyn, zc, stm1)
                    b2 = b1
                node["kids"][a] = self._mknode(dyn, zc, b2)
                break
            for pn, pa in path:                               # BACKUP White-POV
                pn["n"][pa] += 1
                pn["w"][pa] += val
        j = int(root["n"].argmax())
        return root["legal"][j], {"evals": evals,
                                  "score": float(root["w"][j] /
                                                 max(root["n"][j], 1))}


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
WP_K = 0.00368208          # oracle_service's logistic: wp = 1/(1+exp(-k*cp))


def _wp(score, turn_pov: bool):
    """chess.engine score -> mover win-prob via the oracle's logistic."""
    s = score.pov(chess.WHITE if turn_pov else chess.BLACK)
    if s.is_mate():
        return 1.0 if s.mate() > 0 else 0.0
    return 1.0 / (1.0 + np.exp(-WP_K * s.score()))


def sf_regret(engine, board: chess.Board, our_move: chess.Move,
              movetime: float = 0.08):
    """Regret of our_move in win-prob units, measured by Stockfish with the SAME
    instrument on any position (corpus or in-play): wp(SF best) - wp(our move)."""
    info = engine.analyse(board, chess.engine.Limit(time=movetime))
    wp_best = _wp(info["score"], board.turn)
    b2 = board.copy(); b2.push(our_move)
    if b2.is_game_over(claim_draw=True):
        r = b2.result(claim_draw=True)
        if r == "1-0":
            wp_ours = 1.0 if board.turn else 0.0
        elif r == "0-1":
            wp_ours = 0.0 if board.turn else 1.0
        else:
            wp_ours = 0.5
    else:
        info2 = engine.analyse(b2, chess.engine.Limit(time=movetime))
        wp_ours = _wp(info2["score"], board.turn)
    return max(0.0, wp_best - wp_ours)


def _reg_stats(name, rs):
    r = np.array(rs)
    return (f"{name}: n={len(r)} mean {r.mean():.3f} median {np.median(r):.3f} "
            f"p90 {np.percentile(r, 90):.3f} p99 {np.percentile(r, 99):.3f} | "
            f"blunder(>0.25) {(r > 0.25).mean():.1%} catastrophic(>0.5) "
            f"{(r > 0.5).mean():.1%}")


def regret_probe(dyn, algo, maia: str, games: int = 24, corpus_n: int = 300,
                 movetime: float = 0.08, seed: int = 0):
    """Separate TAILS from TRANSFER: the same SF instrument measures our chooser's
    regret (a) on corpus positions and (b) on the positions our own games produce.
    Mean balloons in-play -> transfer; mean holds but tail fat -> tails."""
    import glob as _glob
    from catspace.research.tools.npz_io import load_npz
    from catspace.research.tools.chess_specific.uci import UCIBoardPolicy
    rng = np.random.default_rng(seed)
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Threads": 2})
    # (a) corpus arm, same instrument
    corpus_r = []
    for f in sorted(_glob.glob("data/derived/gko1_shards/shard_*.npz"))[:8]:
        z = load_npz(f)
        if "wdl" not in z.files:
            continue
        for i in np.flatnonzero(z["kind"] == 0)[:60]:
            try:
                b = chess.Board(str(z["fen"][i]))
            except Exception:
                continue
            mv, _ = algo.choose(dyn, b, rng)
            corpus_r.append(sf_regret(sf, b, mv, movetime))
            if len(corpus_r) >= corpus_n:
                break
        if len(corpus_r) >= corpus_n:
            break
    print("[probe] " + _reg_stats("corpus ", corpus_r), flush=True)
    # (b) in-play arm: real games vs maia, same instrument on OUR moves
    play_r, ply_of = [], []
    with UCIBoardPolicy(cmd=["lc0", f"--weights={maia}"], movetime=None,
                        nodes=1) as opp:
        for gi in range(games):
            our_color = gi % 2 == 0
            board = chess.Board()
            orng = np.random.default_rng(20_000 + gi // 2)
            for _ in range(2):
                ms = list(board.legal_moves)
                board.push(ms[int(orng.integers(len(ms)))])
            while (not board.is_game_over(claim_draw=True) and board.ply() < 300):
                if board.turn == our_color:
                    mv, _ = algo.choose(dyn, board, rng)
                    play_r.append(sf_regret(sf, board, mv, movetime))
                    ply_of.append(board.ply())
                else:
                    mv = opp.move(board, rng)
                board.push(mv)
            if (gi + 1) % 6 == 0:
                print("[probe] " + _reg_stats(f"in-play (g{gi+1})", play_r),
                      flush=True)
    sf.quit()
    print("[probe] " + _reg_stats("corpus ", corpus_r), flush=True)
    print("[probe] " + _reg_stats("in-play", play_r), flush=True)
    pr = np.array(play_r); po = np.array(ply_of)
    for lo, hi in ((0, 20), (20, 60), (60, 400)):
        m = (po >= lo) & (po < hi)
        if m.any():
            print(f"[probe]   in-play plies {lo}-{hi}: mean "
                  f"{pr[m].mean():.3f} p90 {np.percentile(pr[m], 90):.3f} "
                  f"(n={int(m.sum())})", flush=True)
    return corpus_r, play_r


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
    if dyn.conditional:
        # T2-conditional: reply injection is live and batched == singleton with it
        f2 = torch.tensor([52] * 20, device=frm.device)
        t2 = torch.tensor([36] * 20, device=frm.device)
        nxt2 = dyn.step(state.expand(20, -1, -1), codes, frm, to, frm2=f2, to2=t2)
        one2 = dyn.step(state, codes[3:4], frm[3:4], to[3:4],
                        frm2=f2[3:4], to2=t2[3:4])
        assert bool((one2[0] == nxt2[3]).all()), "conditional batched != singleton"
        print(f"[units] conditional ckpt: reply changes "
              f"{float((nxt2 != nxt).float().mean()):.3f} of bits at this state")
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
    ap.add_argument("--algo", choices=("greedy1", "prior", "mcts"),
                    default="greedy1")
    ap.add_argument("--mcts-nodes", type=int, default=128)
    ap.add_argument("--mat-lambda", type=float, default=0.5)
    ap.add_argument("--regret-probe", action="store_true",
                    help="measure our chooser's SF-regret on corpus vs in-play "
                         "positions with one instrument (tails-vs-transfer)")
    ap.add_argument("--units", action="store_true")
    args = ap.parse_args()
    if args.units:
        return _units(args)
    if not args.maia:
        from catspace.io import paths
        args.maia = paths.engine("maia/maia-1100.pb.gz")
    dyn = QGTDynamics(args.ckpt)
    if args.regret_probe:
        algo = (Greedy1(args.mat_lambda) if args.algo == "greedy1" else
            MCTS(args.mcts_nodes, mat_lambda=args.mat_lambda)
            if args.algo == "mcts" else PriorPolicy())
        regret_probe(dyn, algo, args.maia or "", games=24, seed=args.seed)
        return
    print(f"[arena] dynamics: {args.ckpt} (step {dyn.step_count}, "
          f"{'graph' if dyn.graph else 'discrete'} arm) vs {args.maia} @ nodes=1",
          flush=True)
    out = {}
    if args.control_games > 0:
        c = latent_arena(RandomPolicy(), dyn, args.maia, args.control_games,
                         args.opening_plies, args.cap, args.seed + 1, tag="random")
        out["random"] = {"score": c.score(), "wilson": c.wilson(), "n": c.games}
    algo = (Greedy1(args.mat_lambda) if args.algo == "greedy1" else
            MCTS(args.mcts_nodes, mat_lambda=args.mat_lambda)
            if args.algo == "mcts" else PriorPolicy())
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
