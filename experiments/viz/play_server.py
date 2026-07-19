#!/usr/bin/env python
"""
experiments/viz/play_server.py — COMPONENT B of the play-atlas app.

A tiny stdlib HTTP server that loads the INCUMBENT model on CPU (the training
GPU stays free) and exposes the play + projection API the frontend calls. Run
the precompute (build_play_atlas.py) first so artifacts/generated/play_atlas/
exists, then:

  .venv/bin/python experiments/viz/play_server.py --port 8000
  # open http://localhost:8000

Endpoints (see the frozen contract): GET / /atlas /toy /region_sample
/legal_moves ; POST /apply_move /engine_move /project /set_board. The model is
single-threaded behind a lock; searches use a committor-MCTS at --nodes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catspace.nn.features import feature_planes, omega_ids  # noqa: E402
from catspace.data.encode import encode_meta, encode_packed  # noqa: E402

TEMPLATE = ROOT / "catspace/viz/templates/play_atlas.html"
ATLAS_DIR = ROOT / "artifacts/generated/play_atlas"
ASSETS_DIR = ROOT / "catspace/viz/assets"      # vendored chessground (JS+CSS)
# THE canonical KRRvKBP toy START (experiments/selfplay_generate.KRRKBP_FIXED_START,
# syzygy wdl=2): Ra1 Ke1 Rh1 vs Bc8 Ke8 pd7. Every train/eval position derives
# from THIS by 2-10 random legal moves — so the test-set FENs look "scattered";
# the Toy button must be the start itself, not a random-derived position (Kaveh).
TOY_FALLBACK = "4kb2/4p3/8/8/8/8/8/R3K2R w - - 0 1"
_MIME = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
         ".json": "application/json", ".html": "text/html; charset=utf-8"}


class Engine:
    """Model + committor-MCTS + fitted t-SNE projection + atlas, all on CPU."""

    def __init__(self, ckpt: str, phead: str, nodes: int, c_puct: float = 1.5,
                 prior_tau: float = 0.5, pw_c: float = 0.0, memory_dir: str | None = None,
                 tactical_prior: float = 0.0, root_min_visits: int = 0,
                 policy_path: str | None = None, value_mode: str = "committor"):
        import torch
        from catspace.nn.eval_head import EvalHead
        from catspace.nn.fb import load_ckpt
        from catspace.nn.policy_fb import make_search_policy
        self.torch = torch
        self.lock = threading.Lock()
        self.dev = "cpu"
        self._ckpt, self._phead = ckpt, phead   # kept so /rebuild_atlas can re-run the builder
        fb, pay = load_ckpt(Path(ckpt), self.dev)
        fb.eval()
        hp = torch.load(phead, map_location=self.dev, weights_only=False)
        ph = EvalHead(d_in=hp["d_in"]).to(self.dev)
        ph.load_state_dict(hp["state"])
        ph.eval()
        self.fb, self.phead = fb, ph
        self.omega_row = omega_ids(np.array([1800]), np.array([1800]),
                                   np.array([300.0]))[0]

        class Committor(torch.nn.Module):
            def forward(self, f):
                p = torch.softmax(ph(f), dim=1)
                return -torch.log(p[:, 0].clamp_min(1e-6)).unsqueeze(-1)

        # quasimetric-distance value (Kaveh 2026-07-19: "use the quasimetric
        # distance"). reach = -head = -d(f -> MATE_W): navigate DOWN the metric's
        # gradient toward mate. Toy A/B: 0.525 vs the committor's 0.425. (The
        # centroid MATE_W beats a region soft-min here -- the field's per-exemplar
        # d is too noisy, so averaging smooths it.)
        zW = pay["zgoals"]["MATE_W"].to(self.dev).float()

        class DistanceHead(torch.nn.Module):
            def forward(self, f):
                return self.torch_fb.distance_matrix(f, zW[None, :])
        DistanceHead.torch_fb = fb
        value_head = DistanceHead() if value_mode == "distance" else Committor()

        # AZ-style cheap expansion when a policy head is present (F-only): child
        # priors from policy(F(node)) + node value from committor(F(node)) in ~1
        # eval, so a simulation costs ~1 eval instead of branching-many.
        pol_fn = val_fn = None
        if value_mode == "committor" and policy_path and Path(policy_path).exists():
            from catspace.nn.policy_head import PolicyHead, legal_priors
            pp = torch.load(policy_path, map_location=self.dev, weights_only=False)
            self._policy_head = PolicyHead(d_in=pp["d_in"], hidden=pp.get("hidden", 256)).to(self.dev)
            self._policy_head.load_state_dict(pp["state"]); self._policy_head.eval()

            def pol_fn(board):
                f = self._embed_F(board)
                with self.torch.no_grad():
                    lg = self._policy_head(self.torch.from_numpy(f[None]).to(self.dev)).cpu().numpy()[0]
                return legal_priors(lg, board)

            def val_fn(boards):
                F = self._embed_F_batch(boards)
                with self.torch.no_grad():
                    p = self.torch.softmax(ph(self.torch.from_numpy(F).to(self.dev)), dim=1).cpu().numpy()
                return p[:, 0] - p[:, 2]                 # white-POV W - L in [-1, 1]

            print(f"AZ cheap-expansion ON: policy head {Path(policy_path).name}", flush=True)

        self.pol = make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"],
                                      max_nodes=nodes, device=self.dev, c_puct=c_puct,
                                      prior_tau=prior_tau, pw_c=pw_c,
                                      tactical_prior=tactical_prior,
                                      root_min_visits=root_min_visits,
                                      policy_fn=pol_fn, value_fn=val_fn,
                                      committor_head=value_head, mate_stop=True)
        self.rng = np.random.default_rng(0)
        self._atlas = None
        self._proj = None
        # position MEMORY (vector DB of seen positions + outcomes). Lazy-loaded;
        # append hooks: finished UI games (add_game) + proven MCTS terminals
        # (_harvest_tree). NOTE queries/appends embed with the server's fixed
        # omega (1800/300) while the shard build used each row's true omega --
        # a ~0.006-cosine wobble on the omega-conditioned incumbent, gone once
        # the field is omega-free.
        self._memory_dir = memory_dir
        self._memory = None
        self._rebuild_proc = None               # running t-SNE rebuild subprocess (stoppable)
        self._ckpt_tag = f"{Path(ckpt).name}@{pay.get('step', '?')}"
        # stateful Analyze: keep the last-analyzed tree so repeated Analyze on
        # the SAME position EXTENDS the search (+budget) instead of restarting
        self._an_root = None
        self._an_fen = None
        self._an_total = 0

    # -- lazy atlas + projection ------------------------------------------
    @property
    def atlas(self):
        if self._atlas is None:
            p = ATLAS_DIR / "atlas.json"
            if not p.exists():
                raise FileNotFoundError(
                    "atlas.json missing — run build_play_atlas.py first")
            self._atlas = json.loads(p.read_text())
        return self._atlas

    @property
    def proj(self):
        if self._proj is None:
            from catspace.viz.projection import Normalizer
            from catspace.viz.manifold import load_projector
            d = ATLAS_DIR / "tsne_map"
            nz = np.load(d / "normalizer.npz")
            projector = load_projector(d)           # tsne / umap / vae (per manifest)
            norm = Normalizer(mu=nz["mu"], sd=nz["sd"])

            class _MapProj:                         # normalize -> out-of-sample transform
                def transform(self, F):
                    return projector.transform(norm.apply(F))
            self._proj = _MapProj()
        return self._proj

    # -- position memory ---------------------------------------------------
    @property
    def memory(self):
        if self._memory is None and self._memory_dir is not None:
            from catspace.memory.store import PositionMemory
            p = Path(self._memory_dir)
            if (p / "meta.json").exists():
                self._memory = PositionMemory.load(p, expect_ckpt_tag=self._ckpt_tag)
                self._mem_seen: set = set()
                print(f"position memory loaded: {len(self._memory)} entries", flush=True)
        return self._memory

    def _embed_F_batch(self, boards: list) -> np.ndarray:
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        planes = self.torch.from_numpy(feature_planes(packed, meta)).to(self.dev)
        om = self.torch.from_numpy(np.tile(self.omega_row, (len(boards), 1))).to(self.dev)
        with self.torch.no_grad():
            return self.fb.embed_F(planes, om).cpu().numpy()

    def neighbors(self, board: chess.Board, k: int = 8) -> dict:
        mem = self.memory
        if mem is None:
            return dict(neighbors=[], n=0)
        nb = mem.query(self._embed_F(board), k=int(k))
        return dict(neighbors=nb, n=len(mem))

    def _mem_append(self, boards: list, results: list, certified: list, source: str):
        """Embed + append boards the server has SEEN (dedup by fen; autosave)."""
        mem = self.memory
        if mem is None or not boards:
            return 0
        fresh = [(b, r, c) for b, r, c in zip(boards, results, certified)
                 if b.fen() not in self._mem_seen]
        if not fresh:
            return 0
        boards, results, certified = map(list, zip(*fresh))
        fens = [b.fen() for b in boards]
        self._mem_seen.update(fens)
        if len(self._mem_seen) > 200_000:
            self._mem_seen.clear()                       # crude bound
        mem.add(self._embed_F_batch(boards), fens, results, certified, source,
                plies=[b.ply() for b in boards])
        if mem._dirty >= 1000:
            mem.save(Path(self._memory_dir))
        return len(fens)

    def add_game(self, fens: list, result: str) -> dict:
        """A game COMPLETED in the UI: every position gets the final outcome.
        chess.js enforced the rules client-side, so the outcome is certified."""
        res = {"white": 1, "black": -1, "draw": 0}.get(result)
        if res is None:
            return dict(added=0)
        boards = []
        for f in fens:
            try:
                boards.append(chess.Board(f))
            except ValueError:
                pass
        n = self._mem_append(boards, [res] * len(boards), [True] * len(boards),
                             source="play_ui")
        return dict(added=n)

    def _harvest_tree(self, root) -> int:
        """Append search lines that reached a RULES-certified terminal ("every
        monte carlo simulation we carry to completion"): each node on the path
        root->terminal gets the terminal's outcome as its MC outcome sample.
        Recognizer-planted terminals (cert_planted) are NOT game truth -> skipped."""
        if self.memory is None:
            return 0
        from catspace.nn.mcts import game_truth
        boards, results = [], []
        stack = [(root, [root.board])]
        while stack and len(boards) < 200:               # cap per search
            node, path = stack.pop()
            if game_truth(node):
                out = 1 if node.terminal_v > 0.5 else (-1 if node.terminal_v < -0.5 else 0)
                for b in path:
                    boards.append(b)
                    results.append(out)
            for c in node.children:
                if c.N > 0 or c.terminal_v is not None:
                    stack.append((c, path + [c.board]))
        return self._mem_append(boards, results, [True] * len(boards), source="mcts_sim")

    # -- embedding helpers -------------------------------------------------
    def _embed_F(self, board: chess.Board) -> np.ndarray:
        packed = encode_packed(board)[None]
        meta = encode_meta(board)[None]
        planes = self.torch.from_numpy(feature_planes(packed, meta)).to(self.dev)
        om = self.torch.from_numpy(np.tile(self.omega_row, (1, 1))).to(self.dev)
        with self.torch.no_grad():
            return self.fb.embed_F(planes, om).cpu().numpy()[0]

    def winp(self, board: chess.Board) -> float:
        f = self.torch.from_numpy(self._embed_F(board)[None]).to(self.dev)
        with self.torch.no_grad():
            return float(self.torch.softmax(self.phead(f), dim=1)[0, 0])

    def _xy(self, board: chess.Board) -> tuple:
        xy = self.proj.transform(self._embed_F(board)[None])[0]
        return round(float(xy[0]), 3), round(float(xy[1]), 3)

    def _pv(self, board: chess.Board, node, maxlen: int = 6) -> list:
        """Principal variation under `node` as [(san, fen, white, board)], with
        NO projection (batched later). Descends max-visit children."""
        b = board.copy(stack=False)
        out, cur = [], node
        while cur is not None and cur.move is not None and len(out) < maxlen:
            try:
                san = b.san(cur.move)
            except (ValueError, AssertionError):
                break
            white = b.turn == chess.WHITE            # side making THIS move
            b.push(cur.move)
            out.append((san, b.fen(), white, b.copy(stack=False)))
            cur = max(cur.children, key=lambda c: c.N, default=None) if cur.children else None
        return out

    def _project_boards(self, boards: list, project: bool = True):
        """Embed F, read winp, and (when `project`) t-SNE-project to (x,y) for
        a LIST of boards in ONE batched pass each. openTSNE .transform() and the
        model forwards amortize over the batch, so this is ~O(1) transform-setup
        for the whole analyze -- replacing the per-hop single-point transforms
        that dominated latency (dozens per Analyze/engine move -> one).
        project=False skips the transform entirely (~0.7s) -- used by STREAMING
        analyze chunks, where only visits/values/winp change frame-to-frame."""
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        planes = self.torch.from_numpy(feature_planes(packed, meta)).to(self.dev)
        om = self.torch.from_numpy(np.tile(self.omega_row, (len(boards), 1))).to(self.dev)
        with self.torch.no_grad():
            f = self.fb.embed_F(planes, om)
            winp = self.torch.softmax(self.phead(f), dim=1)[:, 0].cpu().numpy()
        xy = self.proj.transform(f.cpu().numpy()) if project else None
        return xy, winp

    def _candidates(self, board: chess.Board, nodes, maxlen: int = 6,
                    project: bool = True):
        """Build candidate dicts (uci/san/visits/value/winp/x,y/hops) for the
        given child nodes, projecting root + every endpoint + every PV hop in a
        SINGLE batched transform (x/y are None when project=False). Returns
        ((root_x, root_y), root_winp, cands)."""
        pvs = []
        for c in nodes:
            child = board.copy(stack=False); child.push(c.move)
            pvs.append((c, child, self._pv(board, c, maxlen)))
        boards = [board]
        for _c, child, pv in pvs:
            boards.append(child)
            boards.extend(h[3] for h in pv)
        xy, winp = self._project_boards(boards, project=project)

        def _xy_i(i):
            if xy is None:
                return None, None
            return round(float(xy[i][0]), 3), round(float(xy[i][1]), 3)

        cands, i = [], 1
        for c, child, pv in pvs:
            (cx, cy), cwin = _xy_i(i), round(float(winp[i]), 3)
            i += 1
            hops = []
            for h in pv:
                hx, hy = _xy_i(i)
                hops.append(dict(san=h[0], fen=h[1], white=h[2], x=hx, y=hy))
                i += 1
            cands.append(dict(uci=c.move.uci(), san=board.san(c.move), visits=int(c.N),
                              value=round(float(c.terminal_v if c.terminal_v is not None else c.Q), 3),
                              ci=round(float(0.0 if c.terminal_v is not None else c.value_ci()[1]), 3),
                              winp=cwin, x=cx, y=cy, hops=hops))
        return _xy_i(0), round(float(winp[0]), 3), cands

    def project(self, board: chess.Board) -> dict:
        f = self._embed_F(board)
        xy = self.proj.transform(f[None])[0]
        zw = self.pol.z.cpu().numpy() if hasattr(self.pol, "z") else None
        reach = float(f @ (zw / (np.linalg.norm(zw) + 1e-12))) if zw is not None else 0.0
        return dict(x=round(float(xy[0]), 3), y=round(float(xy[1]), 3),
                    reach=round(reach, 4), winp=round(self.winp(board), 4))

    def analyze(self, board: chess.Board, topk: int = 3, nodes: int | None = None,
                extend: bool = False, project: bool = True) -> dict:
        """Top-k candidate moves for the side to move, each with the PROJECTED
        (x,y) of the resulting position AND its principal-variation LINE (SAN).
        `nodes` sets this call's budget. `extend`=True CONTINUES the previous
        Analyze tree for the SAME position (adds `nodes` more evals on top —
        the tree, visit counts and lines accumulate) instead of restarting;
        a different position (or extend=False) starts fresh. Returns `nodes` =
        the CUMULATIVE budget spent on this position."""
        over, res = self._outcome(board)
        if over:
            self._an_root = self._an_fen = None
            self._an_total = 0
            px, py = self._xy(board)
            return dict(game_over=True, result=res, winp=round(self.winp(board), 4),
                        x=px, y=py, candidates=[], pv=[], nodes=0)
        fen = board.fen()
        with self.lock:
            reuse = (self._an_root if extend and self._an_fen == fen
                     and self._an_root is not None else None)
            if nodes == 0:
                # READOUT ONLY: no new search. Re-serve the cached tree (project
                # the final streaming frame, or expand topk to show more tried
                # moves). None if this position was never searched.
                root, total = reuse, (self._an_total if reuse is not None else 0)
            else:
                old = self.pol.mcts.max_nodes
                self.pol.mcts.max_nodes = int(nodes or old)
                try:
                    root = self.pol.mcts.run(board, reuse_root=reuse)
                finally:
                    self.pol.mcts.max_nodes = old
                self._an_root, self._an_fen = root, fen
                self._an_total = (self._an_total if reuse is not None else 0) + int(nodes or old)
                total = self._an_total
        if root is None:                      # nodes==0 on an un-searched position
            rx, ry = self._xy(board) if project else (None, None)
            return dict(game_over=False, result=None, winp=round(self.winp(board), 4),
                        x=rx, y=ry, candidates=[], pv=[], nodes=0)
        if nodes != 0:
            self._harvest_tree(root)          # memory: completed-simulation lines
        cand = sorted(root.children, key=lambda c: c.N, reverse=True)[:topk]
        (px, py), root_winp, out = self._candidates(board, cand, project=project)
        pv = [h["san"] for h in out[0]["hops"]] if out else []
        return dict(game_over=False, result=None, winp=root_winp,
                    x=px, y=py, candidates=out, pv=pv, nodes=total)

    def navigate(self, board: chess.Board, target_fen: str, nodes: int = 400,
                 topk: int = 3) -> dict:
        """Adversarial goal-conditioned search: the side to move at the ROOT
        steers toward the clicked TARGET position; the opponent RESISTS
        (Kaveh 2026-07-19: "adversarial of course"). Reach = score(F(s), B(target)),
        so the search maximizes CLOSENESS to the target region. FIELD-NATIVE
        directional prior only -- NO policy head (Kaveh: "i don't need the policy
        head if i have the directions properly"): the distance gradient IS the
        move ordering. Returns the top-k approach lines (with projected hops) and
        the target's own map position. Only as good as the field's directions --
        sharp on an aligned field, mushy on the incumbent."""
        from catspace.nn.mcts import MCTS
        over, res = self._outcome(board)
        if over:
            px, py = self._xy(board)
            return dict(game_over=True, result=res, x=px, y=py, candidates=[], target=None)
        tgt = chess.Board(target_fen)
        with self.lock:
            tp = self.torch.from_numpy(feature_planes(
                encode_packed(tgt)[None], encode_meta(tgt)[None])).to(self.dev)
            with self.torch.no_grad():
                zt = self.fb.embed_B(tp)[0]                 # target goal embedding (no omega)
            # sign makes the ROOT MOVER the approach-MAXIMIZER: the MCTS backs up
            # White-POV and flips at Black nodes, so for a Black root we negate
            # reach (White-POV maximizer then flees the target => Black approaches).
            sign = 1.0 if board.turn == chess.WHITE else -1.0

            def reach_fn(boards):
                F = self.torch.from_numpy(self._embed_F_batch(boards)).to(self.dev)
                with self.torch.no_grad():
                    s = self.fb.score(F, zt).cpu().numpy()   # reach toward target (higher=closer)
                return sign * np.asarray(s, dtype=float)

            m = self.pol.mcts
            nav = MCTS(reach_fn, max_nodes=int(nodes),
                       c_puct=getattr(m, "c_puct", 1.0), prior_tau=getattr(m, "prior_tau", 0.5),
                       pw_c=getattr(m, "pw_c", 1.5), root_min_visits=getattr(m, "root_min_visits", 10),
                       tactical_prior=0.0, mate_stop=False, detect_threefold=True)
            root = nav.run(board)
        cand = sorted(root.children, key=lambda c: c.N, reverse=True)[:topk]
        (px, py), _rw, out = self._candidates(board, cand, project=True)
        txy = self.proj.transform(self._embed_F(tgt)[None])[0]
        return dict(game_over=False, x=px, y=py, candidates=out,
                    target=dict(x=round(float(txy[0]), 3), y=round(float(txy[1]), 3),
                                fen=target_fen))

    # -- play --------------------------------------------------------------
    @staticmethod
    def _outcome(board: chess.Board):
        out = board.outcome(claim_draw=True)
        if out is None:
            return False, None
        res = ("white" if out.winner is True else
               "black" if out.winner is False else "draw")
        return True, res

    def engine_move(self, board: chess.Board, nodes: int | None = None) -> dict:
        from catspace.nn.mcts import game_truth
        over, res = self._outcome(board)
        if over:
            return dict(move=None, san=None, fen=board.fen(), game_over=True,
                        result=res, winp=round(self.winp(board), 4),
                        pv=[], candidates=[])
        with self.lock:
            old = self.pol.mcts.max_nodes
            if nodes:                         # match the move budget to the UI depth
                self.pol.mcts.max_nodes = int(nodes)
            try:
                root = self.pol.mcts.run(board)
            finally:
                self.pol.mcts.max_nodes = old
        self._harvest_tree(root)          # memory: completed-simulation lines
        white = board.turn == chess.WHITE
        kids = list(root.children)
        best = None
        for c in kids:
            if game_truth(c) and (c.terminal_v > 0.5 if white else c.terminal_v < -0.5):
                best = c
                break
        if best is None:
            best = max(kids, key=lambda c: (c.N, (c.terminal_v if c.terminal_v
                       is not None else c.Q) * (1 if white else -1)))
        # candidates: top 6 by visits + the chosen best, projected in ONE batch
        cand = sorted(kids, key=lambda c: c.N, reverse=True)[:6]
        nodes = cand if best in cand else [best] + cand
        _root, _rw, projected = self._candidates(board, nodes)
        by_uci = {d["uci"]: d for d in projected}
        px, py = _root
        candidates = [by_uci[c.move.uci()] for c in cand]
        pv = [h["san"] for h in by_uci[best.move.uci()]["hops"]]   # best move's hop-line
        after = board.copy(stack=False)
        san = board.san(best.move)
        after.push(best.move)
        over2, res2 = self._outcome(after)
        return dict(move=best.move.uci(), san=san, fen=after.fen(),
                    game_over=over2, result=res2, x=px, y=py,
                    winp=round(self.winp(after), 4), pv=pv, candidates=candidates)

    def rebuild_atlas(self, n=6000, algo="tsne", params=None):
        """Re-run build_play_atlas.py (CPU) with the selected ALGO (t-SNE / UMAP /
        VAE) + its params, and hot-reload the map: null the cached atlas/projection
        so the next /atlas and /project pick up the freshly-written artifacts. Runs
        as a SUBPROCESS (own model load) so a build failure can't corrupt the live
        server; build_play_atlas writes atlas.json atomically, so a concurrent
        /atlas read never sees a half-written file. n is bounded because this
        competes with any training job for CPU."""
        import subprocess
        import sys as _sys
        from catspace.viz.manifold import clean_params
        params = params or {}
        algo = algo if algo in ("tsne", "umap", "vae") else "tsne"
        n = max(500, min(int(n), 40000))
        p = clean_params(algo, params)          # only valid keys for this algo, cast+defaulted
        cmd = [_sys.executable, str(ROOT / "experiments/viz/build_play_atlas.py"),
               "--ckpt", self._ckpt, "--phead", self._phead,
               "--n", str(n), "--algo", algo]
        flag = {  # param-name -> CLI flag, per algo
            "tsne": {"perplexity": "--perplexity", "exaggeration": "--exaggeration",
                     "n_iter": "--tsne-iter"},
            "umap": {"n_neighbors": "--umap-neighbors", "min_dist": "--umap-min-dist",
                     "n_epochs": "--umap-epochs"},
            "vae":  {"epochs": "--vae-epochs", "hidden": "--vae-hidden", "beta": "--vae-beta"},
        }[algo]
        for k, f in flag.items():
            cmd += [f, str(p[k])]
        # Popen (not run) so /rebuild_atlas_stop can kill it mid-fit. atlas.json
        # is written atomically, so a kill never leaves a half-written map.
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        self._rebuild_proc = proc
        out, _ = proc.communicate()
        self._rebuild_proc = None
        if proc.returncode != 0:
            raise RuntimeError((out or "atlas build stopped/failed").strip()[-400:])
        self._atlas = None                      # force lazy reload of the new artifacts
        self._proj = None
        return dict(n=n, algo=algo, params=p)

    def rebuild_atlas_stop(self) -> dict:
        p = getattr(self, "_rebuild_proc", None)
        if p is not None and p.poll() is None:
            p.terminate()
            return dict(stopped=True)
        return dict(stopped=False)


ENGINE: Engine | None = None


def toy_fen() -> str:
    return TOY_FALLBACK


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    @staticmethod
    def _board(fen: str) -> chess.Board:
        return chess.Board(fen)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, TEMPLATE.read_text(), "text/html; charset=utf-8")
            elif u.path.startswith("/assets/"):
                # serve vendored chessground JS/CSS (same-origin static files);
                # resolve() + prefix guard prevents path traversal
                rel = u.path[len("/assets/"):]
                fp = (ASSETS_DIR / rel).resolve()
                if not str(fp).startswith(str(ASSETS_DIR.resolve())) or not fp.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                self._send(200, fp.read_bytes(), _MIME.get(fp.suffix, "application/octet-stream"))
            elif u.path == "/atlas":
                self._send(200, (ATLAS_DIR / "atlas.json").read_bytes())
            elif u.path == "/toy":
                self._json({"fen": toy_fen()})
            elif u.path == "/legal_moves":
                b = self._board(q["fen"][0])
                self._json({"moves": [m.uci() for m in b.legal_moves]})
            elif u.path == "/region_sample":
                cid = int(q["cluster"][0])
                i = int(q.get("i", ["0"])[0])
                cl = next(c for c in ENGINE.atlas["clusters"] if c["id"] == cid)
                pool = cl["fens"] or [toy_fen()]
                self._json({"fen": pool[i % len(pool)], "i": i % len(pool),
                            "count": len(pool)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/apply_move":
                b = self._board(body["fen"])
                mv = chess.Move.from_uci(body["uci"])
                if mv not in b.legal_moves:
                    self._json({"ok": False, "fen": b.fen(), "san": None,
                                "game_over": False, "result": None})
                    return
                san = b.san(mv)
                b.push(mv)
                over, res = Engine._outcome(b)
                self._json({"ok": True, "fen": b.fen(), "san": san,
                            "game_over": over, "result": res})
            elif u.path == "/engine_move":
                self._json(ENGINE.engine_move(self._board(body["fen"]), body.get("nodes")))
            elif u.path == "/analyze":
                self._json({"ok": True, **ENGINE.analyze(self._board(body["fen"]),
                            int(body.get("topk", 3)), body.get("nodes"),
                            bool(body.get("extend", False)),
                            bool(body.get("project", True)))})
            elif u.path == "/project":
                self._json({"ok": True, **ENGINE.project(self._board(body["fen"]))})
            elif u.path == "/neighbors":
                self._json({"ok": True, **ENGINE.neighbors(self._board(body["fen"]),
                                                           int(body.get("k", 8)))})
            elif u.path == "/navigate":
                self._json({"ok": True, **ENGINE.navigate(
                    self._board(body["fen"]), body["target_fen"],
                    nodes=int(body.get("nodes", 400)))})
            elif u.path == "/memory_add_game":
                self._json({"ok": True, **ENGINE.add_game(body.get("fens", []),
                                                          body.get("result", ""))})
            elif u.path == "/rebuild_atlas":
                self._json({"ok": True, **ENGINE.rebuild_atlas(
                    n=body.get("n", 6000), algo=body.get("algo", "tsne"),
                    params=body.get("params", {}))})
            elif u.path == "/rebuild_atlas_stop":
                self._json({"ok": True, **ENGINE.rebuild_atlas_stop()})
            elif u.path == "/set_board":
                fen = body["fen"]
                try:
                    b = self._board(fen)
                    valid = b.is_valid()
                except Exception:
                    self._json({"ok": True, "valid": False})
                    return
                over, res = Engine._outcome(b)
                self._json({"ok": True, "valid": bool(valid),
                            **ENGINE.project(b),
                            "legal_moves": [m.uci() for m in b.legal_moves],
                            "game_over": over, "result": res})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--c-puct", type=float, default=1.0,
                    help="MCTS exploration constant. LOWER => visits concentrate on the "
                         "best moves instead of spreading across all legal ones (the "
                         "'too wide / low visits' lever). 1.5 is the training default; "
                         "0.75-1.0 reads better interactively.")
    ap.add_argument("--prior-tau", type=float, default=0.5,
                    help="softmax temperature on the field-value move priors; lower => "
                         "sharper priors (more concentrated expansion).")
    ap.add_argument("--memory", default="data/derived/position_memory",
                    help="position-memory dir (vector DB of seen positions + outcomes; "
                         "build with experiments/build_position_memory.py). Serves "
                         "/neighbors and accumulates play_ui/mcts_sim entries online. "
                         "'' disables.")
    ap.add_argument("--pw-c", type=float, default=1.5,
                    help="progressive-widening constant: PUCT descends only into the "
                         "top-K(N)=max(4, ceil(pw_c*N^0.5)) children by field value, so "
                         "the budget deepens instead of spreading over every legal move. "
                         "0 disables (the pre-2026-07-19 full-width behavior).")
    ap.add_argument("--tactical-prior", type=float, default=0.25,
                    help="blend weight w for the rule-derived tactical move prior "
                         "(checks/captures/promotions/material threats): P=(1-w)*field "
                         "+ w*uniform(tactical), and tactical moves always stay in the "
                         "widening window. Ordering only -- values untouched. 0 disables.")
    ap.add_argument("--value", default="committor", choices=["committor", "distance"],
                    help="MCTS leaf value: 'committor' P(win) (flat plateau) or 'distance' "
                         "-d(s->MATE_W), navigating the quasimetric gradient toward mate "
                         "(toy A/B 0.525 vs 0.425). 'distance' is value-only (no AZ policy).")
    ap.add_argument("--policy", default=None,
                    help="policy-head checkpoint (F-only move priors) enabling AZ-style "
                         "cheap expansion (~1 eval/sim). Default: auto-load <ckpt>_policy.pt "
                         "if it exists. '' disables.")
    ap.add_argument("--root-min-visits", type=int, default=10,
                    help="CI-driven root exploration: every non-terminal root move gets "
                         ">= this many visits, then budget goes by UCB/LCB best-arm ID "
                         "(keep sampling moves that could still be best; stop once ~95%% "
                         "confidently worse). Per-move CIs are shown in the panel. 0 = "
                         "plain PUCT root.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    global ENGINE
    print(f"loading incumbent on CPU ({args.ckpt}) ...", flush=True)
    policy_path = args.policy
    if policy_path is None:                              # auto-load <ckpt>_policy.pt
        cand = Path(args.ckpt).with_name(Path(args.ckpt).stem + "_policy.pt")
        policy_path = str(cand) if cand.exists() else None
    ENGINE = Engine(args.ckpt, args.phead, args.nodes, c_puct=args.c_puct,
                    prior_tau=args.prior_tau, pw_c=args.pw_c,
                    memory_dir=args.memory or None, tactical_prior=args.tactical_prior,
                    root_min_visits=args.root_min_visits, policy_path=policy_path or None,
                    value_mode=args.value)
    if not (ATLAS_DIR / "atlas.json").exists():
        print("WARNING: atlas.json missing — run experiments/viz/build_play_atlas.py "
              "(the map will 500 until then)", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"play-atlas server on http://{args.host}:{args.port}  "
          f"(mcts nodes={args.nodes}, model on cpu)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
