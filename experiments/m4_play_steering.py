#!/usr/bin/env python
"""experiments/m4_play_steering.py -- M4 first play harness: planner-ON vs planner-OFF vs Maia.

Both arms share the SAME value policy (the v3 committor field, 1-ply, as in play_vs_maia.py).
The ON arm adds the portfolio shaping term on top of value:
    score(m) = value(m) + eta * [ gain_me(m) - lam * gain_opp(m) - mu * self_blunder(m) ]
with gains = soft_reach deltas over the M3 subgoal lists (approach + avoid), distances
d = -log P(reach) from the two-z field, active plan held with hysteresis, intent logged to the
PlanStore every one of our plies.

VERDICTs printed per MILESTONES M4 gates (this harness = the STEERING instrument; the parity
SPRT at scale runs on top of it):
  score        : W/D/L + score per arm
  steering     : mean predicted net-flux of positions actually VISITED, ON vs OFF, game-
                 clustered bootstrap CI (the DoD's "steering demonstrated" readout)
  plan ledger  : plans logged / reached-rate / mean plies-to-reach (from plan_outcomes)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.play_vs_maia import CommittorGreedy                    # noqa: E402
from catspace.field import ReachabilityField                           # noqa: E402
from catspace.subgoals import SubgoalRanker                            # noqa: E402
from catspace.memory.plan_store import PlanStore                       # noqa: E402
from catspace.planner.subgoal_gen import SubgoalGenerator              # noqa: E402
from catspace.planner.optionality import (ShapeWeights, board_self_blunder,  # noqa: E402
                                          move_scores)
from catspace.train.scaffold import resolve_device                     # noqa: E402
from catspace.stats import paired_nll_ci                               # noqa: E402


class PlannerPolicy(CommittorGreedy):
    """value (committor-greedy 1-ply) + optionality shaping over the M3 subgoal lists."""

    def __init__(self, ckpt, device, gen: SubgoalGenerator | None, rf: ReachabilityField,
                 eta: float, weights: ShapeWeights, elo_self: float, elo_oppo: float,
                 opp_tau: float = 0.15, nu: float = 0.5, sub_ratio: float = 0.25,
                 delta: float = 0.02):
        super().__init__(ckpt, device, opp_tau=opp_tau)
        self.gen = gen; self.rf = rf; self.eta = eta; self.w = weights
        self.nu = nu; self.sub_ratio = sub_ratio; self.delta = delta
        self.elo_self = elo_self; self.elo_oppo = elo_oppo
        self.game_key = ""; self.visited_phis = []

    def _values_1ply(self, lcboard, moves):
        my_white = (lcboard.turn == chess.WHITE)
        planes, term = [], {}
        for i, m in enumerate(moves):
            lcboard.push(m)
            if lcboard.is_game_over(claim_draw=True):
                term[i] = self._term_myval(lcboard, my_white)
            else:
                term[i] = ("leaf", len(planes))
                planes.append(lcboard.to_input_tensor().to("cpu").float().numpy())
            lcboard.pop()
        c = self._committor(planes)
        return np.array([t if not isinstance(t, tuple) else
                         (c[t[1]] if my_white else 1 - c[t[1]]) for t in term.values()])

    def _values_2ply(self, lcboard, moves):
        """root values under 2-ply EXPECTIMAX vs a fallible opponent (opp_tau softmax) --
        the configuration of the historic 0.125 baseline vs Maia-1100."""
        my_white = (lcboard.turn == chess.WHITE)
        leaves, move_reply = [], []
        for m in moves:
            lcboard.push(m)
            if lcboard.is_game_over(claim_draw=True):
                move_reply.append([("term", self._term_myval(lcboard, my_white))])
                lcboard.pop(); continue
            rr = []
            for r_ in lcboard.legal_moves:
                lcboard.push(r_)
                if lcboard.is_game_over(claim_draw=True):
                    rr.append(("term", self._term_myval(lcboard, my_white)))
                else:
                    rr.append(("leaf", len(leaves)))
                    leaves.append(lcboard.to_input_tensor().float().numpy())
                lcboard.pop()
            move_reply.append(rr); lcboard.pop()
        c = self._committor(leaves)

        def leafval(t):
            return t[1] if t[0] == "term" else (c[t[1]] if my_white else 1 - c[t[1]])

        vals = []
        for rr in move_reply:
            v = np.array([leafval(t) for t in rr]) if rr else np.array([0.5])
            if self.opp_tau <= 0:
                vals.append(float(v.min()))
            else:
                w = np.exp(-(v - v.min()) / self.opp_tau); w /= w.sum()
                vals.append(float((w * v).sum()))
        return np.array(vals)

    def select(self, lcboard, rng, depth=2, ply=0):
        moves = list(lcboard.legal_moves)
        if not moves:
            return None, 0.5
        vals = self._values_2ply(lcboard, moves) if depth >= 2 else             self._values_1ply(lcboard, moves)
        if self.gen is None or len(moves) == 1:              # planner OFF -> pure value
            i = int(np.argmax(vals))
            return moves[i], float(vals[i])
        # --- planner ON (iter-3 shaping, 2026-07-29 bundle): PROBABILITY-space gains +
        # successor NET-FLUX term, auto-SUBORDINATED to the value signal (the iter-2 diagnosis:
        # log-space shaping at small p overrode the engine with amplified noise) ---
        phi_now = self.rf.phi([lcboard]).cpu().numpy()
        pc = self.gen.plan(phi_now[0], self.game_key, ply, "w" if lcboard.turn else "b",
                           self.elo_self, self.elo_oppo)
        succ = []
        for m in moves:
            lcboard.push(m); succ.append(lcboard.copy(stack=False)); lcboard.pop()
        phis_after = self.rf.phi(succ).cpu().numpy()
        wme = pc.w_me / pc.w_me.sum(); wop = pc.w_opp / pc.w_opp.sum()
        p_me_b = self.gen.reach_p(phi_now, pc.cells_me, self.elo_self, self.elo_oppo)[0]
        p_op_b = self.gen.reach_p(phi_now, pc.cells_opp, self.elo_self, self.elo_oppo)[0]
        p_me_a = self.gen.reach_p(phis_after, pc.cells_me, self.elo_self, self.elo_oppo)
        p_op_a = self.gen.reach_p(phis_after, pc.cells_opp, self.elo_self, self.elo_oppo)
        gain_me = (p_me_a - p_me_b) @ wme
        gain_opp = (p_op_a - p_op_b) @ wop
        # steer INTO their-error territory now: net flux of each successor's composite cell
        bank = self.gen.rk.bank.cpu().numpy()
        d2 = ((phis_after * phis_after).sum(1)[:, None] + (bank * bank).sum(1)[None, :]
              - 2.0 * phis_after @ bank.T)
        cellp = d2.argmin(1) * self.gen.rk.n_cband + np.digitize(vals, [0.35, 0.65])
        nf = (self.gen.rk.flux[:, self.gen.rk.band(self.elo_oppo)]
              - self.gen.rk.flux[:, self.gen.rk.band(self.elo_self)])[cellp]
        blun = np.array([board_self_blunder(b) for b in succ])
        prior_raw = gain_me - self.w.lam * gain_opp + self.nu * nf - self.w.mu * blun
        # ITER-4 (2026-07-30): TIE-BREAK, not additive. Additive shaping was either noise (log
        # space, iter-2) or inert (subordinated, iter-3): one move's honest effect on reaching a
        # horizon-scale region is tiny. Plans express where chess offers near-equal choices:
        # among moves within DELTA of the best value, take the most plan-advancing one --
        # never pay more than delta (committor units) for the plan.
        best = vals.max()
        cand = np.flatnonzero(vals >= best - self.delta)
        i = int(cand[np.argmax(prior_raw[cand])])
        return moves[i], float(vals[i])


def flux_of_positions(rk, phis, cbs, elo_self, elo_oppo):
    """predicted net-flux of visited positions: composite cell of each -> table lookup."""
    bank = rk.bank.cpu().numpy()
    d2 = (phis * phis).sum(1)[:, None] + (bank * bank).sum(1)[None, :] - 2.0 * phis @ bank.T
    reg = d2.argmin(1)
    cband = np.digitize(cbs, [0.35, 0.65])
    cell = reg * rk.n_cband + cband
    nf = rk.flux[:, rk.band(elo_oppo)] - rk.flux[:, rk.band(elo_self)]
    return nf[cell]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--field", default="artifacts/experiments/reach_v2_full_latest.pt")
    ap.add_argument("--reach", default="data/derived/reach/reach_v2.npz")
    ap.add_argument("--table", default="data/derived/reach/region_table_v2.npz")
    ap.add_argument("--store", default="data/derived/engine_memory.sqlite")
    ap.add_argument("--maia-elo", type=int, default=1100)
    ap.add_argument("--games", type=int, default=20, help="per arm")
    ap.add_argument("--eta", type=float, default=0.5, help="(iter-2 legacy; unused in iter-3)")
    ap.add_argument("--nu", type=float, default=0.5, help="successor net-flux weight in the prior mix")
    ap.add_argument("--sub-ratio", type=float, default=0.25,
                    help="(iter-3 legacy; unused in iter-4 tie-break)")
    ap.add_argument("--delta", type=float, default=0.02,
                    help="value tolerance (committor units) inside which the plan picks the move")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--opp-tau", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--reach-horizon", type=int, default=40)
    ap.add_argument("--our-elo", type=float, default=1800.0,
                    help="context Elo for OUR side fed to the field (engine-as-strong-club-player)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="m4a")
    ap.add_argument("--save-pgn", default="artifacts/experiments/m4_steering.pgn")
    args = ap.parse_args()
    dev = resolve_device("auto"); rng = np.random.default_rng(args.seed); t0 = time.time()

    rf = ReachabilityField(device=str(dev))
    rk = SubgoalRanker(args.field, args.reach, args.table, device=str(dev))
    store = PlanStore(args.store)
    weights = ShapeWeights(beta=args.beta, lam=args.lam, mu=args.mu)
    maia = chess.engine.SimpleEngine.popen_uci(
        ["lc0", f"--weights=data/engines/maia/maia-{args.maia_elo}.pb.gz", "--backend=eigen"])

    results = {}
    for arm in ("off", "on"):
        gen = SubgoalGenerator(rk, store, top_k=args.top_k) if arm == "on" else None
        pol = PlannerPolicy(args.ckpt, dev, gen, rf, args.eta, weights,
                            args.our_elo, float(args.maia_elo), opp_tau=args.opp_tau,
                            nu=args.nu, sub_ratio=args.sub_ratio, delta=args.delta)
        W = D = L = 0; flux_means = []; flux_all = []; pgns = []
        for g in range(args.games):
            from lczerolens import LczeroBoard
            board = LczeroBoard(); our_white = (g % 2 == 0)
            pol.game_key = f"{args.tag}_{arm}_{g}"
            for _ in range(args.opening_plies):
                ms = list(board.legal_moves)
                if not ms:
                    break
                board.push(ms[rng.integers(0, len(ms))])
            phis_seen, planes_seen, ours_mask = [], [], []
            ply = board.ply()
            while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
                we_move = board.turn == (chess.WHITE if our_white else chess.BLACK)
                if we_move:
                    mv, _ = pol.select(board, rng, depth=args.depth, ply=ply)
                else:
                    mv = maia.play(board, chess.engine.Limit(nodes=1)).move
                if mv is None:
                    break
                board.push(mv); ply += 1
                if not board.is_game_over(claim_draw=True):
                    phis_seen.append(rf.phi([board]).cpu().numpy()[0])
                    planes_seen.append(board.to_input_tensor().float().numpy())
                    ours_mask.append(we_move)
            res = board.result(claim_draw=True)
            s = 0.5 if res == "1/2-1/2" else (1.0 if (res == "1-0") == our_white else 0.0)
            W += s == 1.0; D += s == 0.5; L += s == 0.0
            # steering readout: predicted net-flux of positions actually visited
            cbs = np.zeros(0)
            if phis_seen:
                cbs = pol._committor(planes_seen)
                cbs = cbs if our_white else 1 - cbs          # mover-POV-ish committor coordinate
                fx = flux_of_positions(rk, np.stack(phis_seen), np.asarray(cbs),
                                       args.our_elo, float(args.maia_elo))
                om = np.asarray(ours_mask, bool)
                flux_means.append(float(fx[om].mean()) if om.any() else float(fx.mean()))
                flux_all.append(float(fx.mean()))
            # fill plan outcomes (ledger): did the game enter the active cell later?
            if arm == "on":
                P = np.stack(phis_seen) if phis_seen else np.zeros((0, 64))
                cells_seen = None
                if len(P):
                    bank = rk.bank.cpu().numpy()
                    d2 = (P * P).sum(1)[:, None] + (bank * bank).sum(1)[None, :] - 2 * P @ bank.T
                    cells_seen = d2.argmin(1) * rk.n_cband + np.digitize(
                        np.asarray(cbs), [0.35, 0.65])
                for pid, pply, cell in store.pending(pol.game_key):
                    hit = np.flatnonzero(cells_seen == cell) if cells_seen is not None else []
                    store.log_outcome(pid, len(hit) > 0,
                                      int(hit[0]) if len(hit) else None, False, 0.0)
            gp = chess.pgn.Game.from_board(board)
            gp.headers["White"] = ("catspace" if our_white else f"maia-{args.maia_elo}")
            gp.headers["Black"] = (f"maia-{args.maia_elo}" if our_white else "catspace")
            gp.headers["Result"] = res
            pgns.append(str(gp))
            print(f"  [{arm}] game {g+1}/{args.games} -> {res} (us {s}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        n = args.games
        results[arm] = dict(score=(W + 0.5 * D) / n, W=int(W), D=int(D), L=int(L),
                            flux=flux_means, flux_all=flux_all)
        Path(args.save_pgn.replace(".pgn", f"_{arm}.pgn")).write_text("\n\n".join(pgns))
    maia.quit()

    for arm in ("off", "on"):
        r = results[arm]
        print(f"VERDICT M4 score [{arm}]: {r['score']:.3f} (W{r['W']} D{r['D']} L{r['L']} "
              f"of {args.games}) vs maia-{args.maia_elo}")
    fon, foff = np.array(results["on"]["flux"]), np.array(results["off"]["flux"])
    diff = fon.mean() - foff.mean()
    boots = [np.random.default_rng(i).choice(fon, len(fon)).mean()
             - np.random.default_rng(i + 9999).choice(foff, len(foff)).mean() for i in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"VERDICT M4 steering (OUR-move positions): ON {fon.mean():+.5f} vs OFF "
          f"{foff.mean():+.5f} | diff {diff:+.5f} CI[{lo:+.5f},{hi:+.5f}] "
          f"{'PASS' if lo > 0 else 'not yet'} (game-bootstrap)")
    fa_on, fa_off = np.array(results["on"]["flux_all"]), np.array(results["off"]["flux_all"])
    print(f"VERDICT M4 steering (all positions):    ON {fa_on.mean():+.5f} vs OFF "
          f"{fa_off.mean():+.5f} | diff {fa_on.mean()-fa_off.mean():+.5f}")
    rows = store.intent_vs_realization([f"{args.tag}_on_{g}" for g in range(args.games)])
    if rows:
        reached = np.array([r[4] for r in rows])
        print(f"VERDICT M4 ledger: {len(rows)} plans | reached {reached.mean():.1%}")


if __name__ == "__main__":
    main()
