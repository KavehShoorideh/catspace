#!/usr/bin/env python
"""experiments/m5_mcts_probe.py -- M5 MVP: chain-plan navigation MCTS vs Maia (the prober).

Kaveh 2026-07-29 (iter-2, threshold-free): one-subgoal routing gives no lift -- the plan is a
CHAIN of CHUTES  here -> g1 -> g2 -> ... : each subgoal is the EDGE of a chute (a region where
the OPPONENT's committor-crossing rate is high), we park there hoping they fall through; if
they don't, we move to the next chute. Candidate plans ranked by value, best one navigated:

    V(g) = max( q(g),  fall_opp(g) + (1 - fall_opp(g) - fall_us(g)) * max_g' P(g->g') V(g') )
    plan(a) = argmax_g  P(reach g | a, z, elos) * V(g)

  * interior edges P(g->g'): the v3 first-hit field evaluated at region CENTROIDS
    (off-manifold approximation, one batched forward at load -> static directed G x G graph);
  * fall rates: the region table's SF-refereed committor-crossing rates at THEIR band vs OURS
    (the M3 "their error zone" columns) -- symmetric risk: our own crossings count against;
  * q(g) = region committor (count-weighted over c-bands) SHRUNK toward the population mean
    with pseudo-count = median table support (a prior, not a cutoff);
  * V by value iteration on the absorbing walk (max-product Bellman, converges); per our-move
    the plan is a field sweep + argmax -- chain recovered for the ledger via next_hop. NO
    hand-coded gates: unreachable targets and dead chutes lose the argmax on their own.
  * navigation: MCTS toward the FIRST HOP; node signal = P(reach g1 | s), White-POV signed;
    rule-exact mate/draw terminals stay (game truth, not WDL leaves -- MILESTONES §M5).
    Replan every our-move (selection is O(G)); no hysteresis, no death threshold -- switches
    are INSTRUMENTED, not suppressed.

Known approximation (instrumented, not knob-patched): chain likelihood composes game-remainder
first-hit probabilities as if Markov in region space with a full game per hop -- optimistic for
long chains. The plans verdict (predicted vs realized) is the check.

VERDICT lines (journal rule: numbers only from printed script output):
  M5 score  : W/D/L + score vs maia-<elo>  (context: 0.125 = shallow committor baseline)
  M5 plans  : plan spells / first-hop hit rate / pred vs realized plies / switch rate / depth
  M5 budget : mean fresh evals per our-move (node-budget accounting for the equal-budget A/B)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.field import ReachabilityField                            # noqa: E402
from catspace.subgoals import SubgoalRanker                             # noqa: E402
from catspace.nn.mcts import MCTS                                       # noqa: E402
from catspace.train.scaffold import resolve_device                      # noqa: E402


class ChainPlanner:
    """Most-likely-chain plan selection + reach-guided MCTS navigation. Threshold-free."""

    def __init__(self, rk: SubgoalRanker, rf: ReachabilityField, elo_self: float,
                 elo_oppo: float, nodes: int, c_puct: float = 1.5, prior_tau: float = 0.5):
        self.rk = rk; self.rf = rf
        self.elo_self = elo_self; self.elo_oppo = elo_oppo
        self.nodes = nodes; self.c_puct = c_puct; self.prior_tau = prior_tau
        self.b_self = rk.band(elo_self)

        # -- region-level quality with SHRINKAGE (prior, not cutoff) --------------------
        G, CB = len(rk.bank), rk.n_cband
        q = rk.quality[:, self.b_self].reshape(G, CB)
        c = rk.counts[:, self.b_self].reshape(G, CB).astype(float)
        n = c.sum(1)
        q_raw = (q * c).sum(1) / np.maximum(n, 1.0)
        q_bar = float((q * c).sum() / max(c.sum(), 1.0))         # population mean committor
        n0 = float(np.median(n))                                  # pseudo-count = median support
        self.q_region = (n * q_raw + n0 * q_bar) / (n + n0)

        # -- CHUTE fall rates (Kaveh 2026-07-29: subgoal = edge of a chute) --------------
        # crossing_rate[g, band] = SF-refereed committor-crossing rate of the region at that
        # mover band (composite-granular -> count-weighted to region level, like quality).
        b_opp = rk.band(elo_oppo)
        fl_op = rk.flux[:, b_opp].reshape(G, CB)
        fl_us = rk.flux[:, self.b_self].reshape(G, CB)
        c_op = rk.counts[:, b_opp].reshape(G, CB).astype(float)
        self.fall_opp = (fl_op * c_op).sum(1) / np.maximum(c_op.sum(1), 1.0)
        self.fall_us = (fl_us * c).sum(1) / np.maximum(n, 1.0)

        # -- static region graph + absorbing-walk VALUE ITERATION ------------------------
        # V(g) = max( q(g),                                    stop-and-convert here
        #             fall_opp(g)                              they fall through the chute
        #             + (1 - fall_opp(g) - fall_us(g))         neither falls ->
        #               * max_g' P(g -> g') V(g') )            move to the NEXT chute
        p_cc, _ = self._region_heads(rk.bank.cpu().numpy())       # (G, G) centroid edges
        np.fill_diagonal(p_cc, 0.0)
        stay = np.clip(1.0 - self.fall_opp - self.fall_us, 0.0, 1.0)
        V = self.q_region.copy()
        for _ in range(200):
            cont = (p_cc * V[None, :]).max(1)
            V_new = np.maximum(self.q_region, self.fall_opp + stay * cont)
            if np.abs(V_new - V).max() < 1e-9:
                V = V_new
                break
            V = V_new
        cont = p_cc * V[None, :]
        self.next_hop = np.where(self.fall_opp + stay * cont.max(1) > self.q_region,
                                 cont.argmax(1), -1)              # -1 = convert here
        self.V = V

        self.cache: dict = {}            # (region, fen) -> reach; outlives moves
        self.evals = []                  # fresh evals per our-move
        self.new_game()

    def new_game(self):
        self.active = None               # current first-hop target region
        self.spell = None                # instrument row for the current target spell
        self.spells: list[dict] = []

    def chain(self, g: int) -> list[int]:
        out = [g]
        while self.next_hop[out[-1]] >= 0 and len(out) < 32:
            out.append(int(self.next_hop[out[-1]]))
        return out

    # -- field readout (region level) ---------------------------------------------------
    @torch.no_grad()
    def _region_heads(self, phis):
        """(B,64) phi -> (p_hit (B,G), plies (B,G)) under the population-z context."""
        B = len(phis)
        m = self.rk.model
        f = torch.as_tensor(np.asarray(phis, np.float32), device=self.rk.dev)
        zs = torch.zeros(B, 16, device=self.rk.dev)
        ctx = [torch.tensor([[(self.elo_self - 1500) / 400, (self.elo_oppo - 1500) / 400,
                              1.0, 1.0]], dtype=torch.float32, device=self.rk.dev).expand(B, -1)]
        if m.state[0].in_features > 84:                     # two-z field: z_opp cold start
            ctx.append(torch.cat([torch.zeros(B, 16, device=self.rk.dev),
                                  torch.zeros(B, 1, device=self.rk.dev)], 1))
        sh, st = m.state_embs(f, zs, torch.cat(ctx, 1))
        p = torch.sigmoid(sh @ self.rk.gh.T * m.scale + m.b_hit)
        plies = torch.expm1((st @ self.rk.gt.T * m.scale + m.b_time).clamp(0, 8))
        return p.cpu().numpy(), plies.cpu().numpy()

    def _phi_region(self, phi):
        bank = self.rk.bank.cpu().numpy()
        return int(((bank - phi) ** 2).sum(1).argmin())

    # -- the move -----------------------------------------------------------------------
    def select(self, board, ply: int):
        phi_now = self.rf.phi([board]).cpu().numpy()[0]
        cur = self._phi_region(phi_now)
        p, plies = self._region_heads(phi_now[None]); p, plies = p[0], plies[0]

        # best plan from HERE: log P(reach g | a) + log V(g); own region excluded (no-op plan)
        score = np.log(np.maximum(p, 1e-12)) + np.log(np.maximum(self.V, 1e-12))
        score[cur] = -np.inf
        g1 = int(np.argmax(score))

        if self.spell is not None:                               # close out the running spell
            inc = self.spell["target"]
            if cur == inc:
                self.spell.update(outcome="hit", actual=ply - self.spell["ply"])
                self.spell = None
            elif g1 != inc:
                # flapping vs honest abandonment: margin = challenger - incumbent NOW;
                # decay = how much the incumbent's own score fell since adoption
                self.spell.update(outcome="switch", actual=ply - self.spell["ply"],
                                  margin=float(score[g1] - score[inc]),
                                  decay=float(self.spell["loglik"] - score[inc]))
                self.spell = None
        if self.spell is None:
            ch = self.chain(g1)
            self.spell = dict(ply=ply, target=g1, chain=len(ch),
                              loglik=float(score[g1]), pred_plies=float(plies[g1]),
                              outcome="open", actual=None)
            self.spells.append(self.spell)

        tid, our_white = g1, board.turn == chess.WHITE

        def reach_fn(boards):
            phis = self.rf.phi(boards).cpu().numpy()
            pr = self._region_heads(phis)[0][:, tid]
            return pr if our_white else -pr                      # White-POV sign

        mcts = MCTS(reach_fn, max_nodes=self.nodes, c_puct=self.c_puct,
                    prior_tau=self.prior_tau, cache=self.cache,
                    cache_key_fn=lambda b: f"{tid}|{b.fen()}")
        mv = mcts.best_move(board)
        self.evals.append(mcts.evals_used)
        return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="artifacts/experiments/reach_v3_full_latest.pt")
    ap.add_argument("--reach", default="data/derived/reach/reach_v3.npz")
    ap.add_argument("--table", default="data/derived/reach/region_table_v3.npz")
    ap.add_argument("--maia-elo", type=int, default=1100)
    ap.add_argument("--our-elo", type=float, default=1800.0, help="our rating frame (as m4)")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--nodes", type=int, default=200, help="fresh-eval budget per our-move")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="m5b")
    ap.add_argument("--save-pgn", default="artifacts/experiments/m5_probe.pgn")
    args = ap.parse_args()
    dev = resolve_device("auto"); rng = np.random.default_rng(args.seed); t0 = time.time()

    rf = ReachabilityField(device=str(dev))
    rk = SubgoalRanker(args.field, args.reach, args.table, device=str(dev))
    nav = ChainPlanner(rk, rf, args.our_elo, float(args.maia_elo), args.nodes)
    chutes = int((nav.next_hop >= 0).sum())
    dep = np.array([len(nav.chain(g)) for g in range(len(nav.V))])
    print(f"  graph: V median {np.median(nav.V):.3f} p90 {np.percentile(nav.V, 90):.3f} | "
          f"{chutes}/{len(nav.V)} regions continue to a next chute (rest convert in place) | "
          f"chain depth median {np.median(dep):.0f} p90 {np.percentile(dep, 90):.0f} "
          f"max {dep.max()} | fall_opp-fall_us median "
          f"{np.median(nav.fall_opp - nav.fall_us):+.4f}", flush=True)
    maia = chess.engine.SimpleEngine.popen_uci(
        ["lc0", f"--weights=data/engines/maia/maia-{args.maia_elo}.pb.gz", "--backend=eigen"])

    W = D = L = 0; pgns = []; all_spells = []
    for g in range(args.games):
        from lczerolens import LczeroBoard
        board = LczeroBoard(); our_white = (g % 2 == 0)
        nav.new_game()
        for _ in range(args.opening_plies):
            ms = list(board.legal_moves)
            if not ms:
                break
            board.push(ms[rng.integers(0, len(ms))])
        ply = board.ply()
        while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
            we_move = board.turn == (chess.WHITE if our_white else chess.BLACK)
            mv = nav.select(board, ply) if we_move else \
                maia.play(board, chess.engine.Limit(nodes=1)).move
            if mv is None:
                break
            board.push(mv); ply += 1
        res = board.result(claim_draw=True)
        s = 0.5 if res == "1/2-1/2" else (1.0 if (res == "1-0") == our_white else 0.0)
        W += s == 1.0; D += s == 0.5; L += s == 0.0
        if nav.spell is not None:
            nav.spell.update(outcome="end", actual=ply - nav.spell["ply"])
        all_spells.extend(nav.spells)
        gp = chess.pgn.Game.from_board(board)
        gp.headers["White"] = ("catspace-m5" if our_white else f"maia-{args.maia_elo}")
        gp.headers["Black"] = (f"maia-{args.maia_elo}" if our_white else "catspace-m5")
        gp.headers["Result"] = res
        pgns.append(str(gp))
        hits = sum(sp["outcome"] == "hit" for sp in nav.spells)
        print(f"  game {g+1}/{args.games} -> {res} (us {s}) | spells {len(nav.spells)} "
              f"hit {hits} | {time.time()-t0:.0f}s", flush=True)
    maia.quit()
    Path(args.save_pgn).write_text("\n\n".join(pgns))

    n = args.games
    print(f"VERDICT M5 score: {(W + 0.5 * D) / n:.3f} (W{W} D{D} L{L} of {n}) "
          f"vs maia-{args.maia_elo} [nodes={args.nodes}] "
          f"(shallow committor baseline context: 0.125)")
    if all_spells:
        hits = [sp for sp in all_spells if sp["outcome"] == "hit"]
        sw = sum(sp["outcome"] == "switch" for sp in all_spells)
        pp = np.array([sp["pred_plies"] for sp in hits], float)
        aa = np.array([sp["actual"] for sp in hits], float)
        cal = (f"plies pred median {np.median(pp):.1f} vs realized {np.median(aa):.1f}"
               if hits else "no hits")
        print(f"VERDICT M5 plans: {len(all_spells)} spells | hit {len(hits)} "
              f"({len(hits)/len(all_spells):.0%}) switch {sw} "
              f"({sw/len(all_spells):.0%}) | chain depth median "
              f"{np.median([sp['chain'] for sp in all_spells]):.0f} | {cal}")
        sws = [sp for sp in all_spells if sp["outcome"] == "switch"]
        if sws:
            mg = np.array([sp["margin"] for sp in sws])
            dc = np.array([sp["decay"] for sp in sws])
            print(f"VERDICT M5 switches: margin (challenger-incumbent, nats) median "
                  f"{np.median(mg):.3f} p90 {np.percentile(mg, 90):.3f} | incumbent decay "
                  f"since adoption median {np.median(dc):.3f} "
                  f"({np.mean(dc > np.median(mg)):.0%} decay-dominated = honest abandonment)")
    ev = np.array(nav.evals, float)
    print(f"VERDICT M5 budget: {ev.mean():.0f} fresh evals/our-move "
          f"(median {np.median(ev):.0f}, n={len(ev)}) | cache {len(nav.cache)} entries")


if __name__ == "__main__":
    main()
