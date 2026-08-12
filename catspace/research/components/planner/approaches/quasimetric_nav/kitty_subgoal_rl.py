#!/usr/bin/env python
"""kitty_subgoal_rl.py -- the SUBGOAL-CHOOSING reinforcement planner (Kaveh 2026-08-11:
"the RL planner should learn [fast vs slow concepts] itself... this won't work without it").

Loop, per decision point (every K plies or on achievement/failure):
    selector pi(g | phi(s)) picks a TARGET CONCEPT g = (vq-head, code) among codes NOT
    currently active; moves are then chosen by the concept-dynamics head -- score(m) =
    log P(child activates g) + lam * z(cascade margin)  (the blend keeps pursuit from
    hanging pieces; lam=0 tests pure pursuit).

Rewards, per Kaveh's dense-credit design:
    achievement -- did g activate within H plies of commitment? (dense, per decision;
                   masked to inactive codes so "pick what's already true" cannot score)
    outcome     -- the game result from the decider's POV (sparse, teaches WORTH)
    REINFORCE with separate running baselines; R = a*ach + b*outcome.

Legibility outputs: per-concept choice counts, achievement rates, and mean plies-to-achieve
-- the fast/slow spectrum DISCOVERED, not curated.

    .venv/bin/python -m ...kitty_subgoal_rl --ckpt <field.pt> [--iters 40] [--games-per 16]
"""
from __future__ import annotations

import argparse
import os
import random

import chess
import numpy as np
import torch
import torch.nn as nn

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_dynamics import (
    ConceptDynamics)
from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
    ConceptVQ)
from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize, move_ids


class Selector(nn.Module):
    def __init__(self, d_in=128, heads=8, codes=64, hidden=256, deny=False):
        super().__init__()
        self.deny = deny
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, heads * codes * (2 if deny else 1)))

    def forward(self, phi):
        return self.net(phi)


class Planner:
    def __init__(self, eng, vq, dyn, selector, K=4, H=6, lam=0.5, device="mps"):
        self.eng, self.vq, self.dyn, self.sel = eng, vq, dyn, selector
        self.K, self.H, self.lam, self.device = K, H, lam, device
        self.heads, self.codes = vq.heads, dyn.codes

    def phi_codes(self, board):
        tk, gl = tokenize(board)
        with torch.no_grad():
            phi = self.eng.net.backbone(
                torch.from_numpy(np.asarray([tk], dtype=np.int64)).to(self.device),
                torch.from_numpy(np.asarray([gl], dtype=np.float32)).to(self.device))
            _, ids, _ = self.vq(phi)
        return phi, ids[0].cpu().numpy()

    def pick_subgoal(self, phi, codes, sample=True, armed=None):
        K = self.heads * self.codes
        deny = getattr(self.sel, "deny", False)
        logits = self.sel(phi)[0].view(-1).clone()
        for h in range(self.heads):                       # mask already-active codes (both modes)
            logits[h * self.codes + codes[h]] = -1e9
            if deny:
                logits[K + h * self.codes + codes[h]] = -1e9
        if deny and armed is not None:                    # denial only against ARMED threats
            logits[K:][~armed.view(-1)] = -1e9
        if sample:
            g = int(torch.distributions.Categorical(logits=logits).sample())
        else:
            g = int(logits.argmax())
        logp = torch.log_softmax(logits, 0)[g]
        mode = "deny" if (deny and g >= K) else "pursue"
        g = g % K
        return g // self.codes, g % self.codes, logp, mode

    def move_for(self, board, phi, gh, gc, deny=False):
        moves = list(board.legal_moves)
        if not moves:
            return None
        mids = torch.from_numpy(np.array([move_ids(m) for m in moves],
                                         dtype=np.int64)).to(self.device)
        with torch.no_grad():
            out = self.dyn(phi.expand(len(moves), -1), mids)
            if isinstance(out, tuple):
                lg_c, lg_r = out
                src = lg_r if deny else lg_c              # denial steers the REPLY landscape
            else:
                src = out
            act = torch.log_softmax(src[:, gh], -1)[:, gc].float().cpu().numpy()
        if deny:
            act = -act                                    # minimize the threat's activation
        score = act.copy()
        if self.lam > 0:                                  # safety blend: standing z-score,
            _, _, E = self.eng.cascade_rank(board)        # scaled to the activation spread
            Em = np.array(E) if board.turn else 1.0 - np.array(E)
            sd = Em.std() if Em.std() > 1e-9 else 1.0
            score = act + self.lam * ((Em - Em.mean()) / sd) * max(1.0, float(act.std()))
        return moves[int(np.argmax(score))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games-per", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--deny", action="store_true",
                    help="v2 action space: PURSUE a concept or DENY one (keep an opponent-"
                         "achievable concept OFF for the horizon). Requires the reply-head "
                         "dynamics (_dyn2.pt). Denial only scores against ARMED threats: at "
                         "commitment the reply head must rate the concept opponent-achievable "
                         "(P >= deny-thr) -- denying the impossible cannot farm reward.")
    ap.add_argument("--deny-thr", type=float, default=0.05)
    ap.add_argument("--cascade-frac", type=float, default=0.35,
                    help="per ply, probability of playing the CASCADE move instead of pure "
                         "pursuit -- games must END for the outcome channel to teach "
                         "(v1: decisive near zero, outcome silent)")
    ap.add_argument("--a-ach", type=float, default=1.0)
    ap.add_argument("--b-out", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    pv = torch.load(base + "_vq.pt", map_location=args.device, weights_only=False)
    vq = ConceptVQ(d_in=pv["d_in"], heads=pv["heads"], codes=pv["codes"]).to(args.device)
    vq.load_state_dict(pv["state_dict"]); vq.eval()
    dyn_path = base + ("_dyn2.pt" if args.deny else "_dyn.pt")
    pd_ = torch.load(dyn_path, map_location=args.device, weights_only=False)
    dyn = ConceptDynamics(d_in=pd_["d_in"], heads=pd_["heads"], codes=pd_["codes"],
                          reply=pd_.get("reply", False)).to(args.device)
    dyn.load_state_dict(pd_["state_dict"]); dyn.eval()
    eng = KittyChess(args.ckpt, args.device)
    sel = Selector(d_in=pv["d_in"], heads=pv["heads"], codes=pv["codes"],
                   deny=args.deny).to(args.device)
    sel_path = base + "_selector.pt"
    if os.path.exists(sel_path):
        sel.load_state_dict(torch.load(sel_path, map_location=args.device))
        print("[srl] resumed selector")
    opt = torch.optim.Adam(sel.parameters(), lr=args.lr)
    pl = Planner(eng, vq, dyn, sel, K=args.K, H=args.H, lam=args.lam, device=args.device)
    rng = random.Random(args.seed)
    fens = []
    for tsv in ("piecedown_sfsf_all_v2.tsv",):
        fp = paths.derived(tsv)
        if os.path.exists(fp):
            fens += [l.split("\t")[2] for l in open(fp) if l.count("\t") >= 3][:30000]
    print(f"[srl] {len(fens):,} start fens; subgoal space {pv['heads']}x{pv['codes']}")

    stats = {}                                            # (h,c) -> [chosen, achieved, plies]
    ach_base, out_base = 0.5, 0.0
    for it in range(args.iters):
        decisions = []                                    # (logp, achieved, outcome_signed)
        n_moves = 0
        for g in range(args.games_per):
            b = chess.Board(rng.choice(fens)) if fens and g % 3 else chess.Board()
            pending = []                                  # (logp, target, deadline, white_side)
            while not b.is_game_over(claim_draw=True) and b.ply() < 140:
                phi, codes = pl.phi_codes(b)
                # resolve pending achievements at every ply
                for pdg in pending[:]:
                    logp, tgt3, dl, ws = pdg
                    gh, gc, mode = tgt3
                    hit = codes[gh] == gc
                    if mode == "pursue" and hit:          # pursued concept arrived: success
                        decisions.append([logp, 1.0, None, ws, tgt3, b.ply()])
                        stats.setdefault((mode, gh, gc), [0, 0, 0.0])[1] += 1
                        pending.remove(pdg)
                    elif mode == "deny" and hit:          # denied concept slipped in: failure
                        decisions.append([logp, 0.0, None, ws, tgt3, None])
                        pending.remove(pdg)
                    elif b.ply() >= dl:                   # horizon: pursue fails, deny succeeds
                        ok = 1.0 if mode == "deny" else 0.0
                        if mode == "deny":
                            stats.setdefault((mode, gh, gc), [0, 0, 0.0])[1] += 1
                        decisions.append([logp, ok, None, ws, tgt3, None])
                        pending.remove(pdg)
                # COMMITMENT PROTOCOL (smoke bug: K-cadence re-picks silently discarded
                # unresolved subgoals -- 6 decisions logged where ~140 occurred, all biased
                # toward instant successes): hold ONE subgoal until it RESOLVES.
                if not pending:
                    armed = None
                    if args.deny:
                        with torch.no_grad():             # opponent-achievable set (reply head)
                            nullm = torch.zeros((1, 3), dtype=torch.long, device=args.device)
                            _, lg_r = dyn(phi, nullm)
                            armed = (torch.softmax(lg_r[0], -1) >= args.deny_thr)
                    gh, gc, logp, mode = pl.pick_subgoal(phi, codes, armed=armed)
                    stats.setdefault((mode, gh, gc), [0, 0, 0.0])[0] += 1
                    pending = [(logp, (gh, gc, mode), b.ply() + pl.H, b.turn)]
                gh, gc, mode = pending[0][1]
                if rng.random() < args.cascade_frac:
                    mv = eng.choose(b)
                else:
                    mv = pl.move_for(b, phi, gh, gc, deny=(mode == "deny"))
                if mv is None:
                    break
                b.push(mv); n_moves += 1
            out = b.outcome(claim_draw=True)
            res = 0.0 if out is None or out.winner is None else (1.0 if out.winner else -1.0)
            for dch in decisions:
                if dch[2] is None:
                    dch[2] = res if dch[3] else -res
        if not decisions:
            continue
        ach = np.array([d[1] for d in decisions]); outs = np.array([d[2] for d in decisions])
        ach_base = 0.9 * ach_base + 0.1 * float(ach.mean())
        out_base = 0.9 * out_base + 0.1 * float(outs.mean())
        loss = torch.zeros((), device=args.device)
        for logp, a, o, *_ in decisions:
            R = args.a_ach * (a - ach_base) + args.b_out * (o - out_base)
            loss = loss - logp * float(R)
        (loss / len(decisions)).backward()
        opt.step(); opt.zero_grad()
        dec = float(np.mean(np.abs([d[2] for d in decisions])))
        print(f"[srl] iter {it+1}/{args.iters}: {len(decisions)} decisions, "
              f"achievement {ach.mean():.1%} (base {ach_base:.1%}), decisive {dec:.0%}",
              flush=True)
        if (it + 1) % 10 == 0:
            torch.save(sel.state_dict(), sel_path)
            top = sorted(stats.items(), key=lambda kv: -kv[1][0])[:8]
            print("[srl] most-chosen subgoals (chosen/achieved):")
            for key, (n, a2, _) in top:
                lbl = (f"{key[0]} h{key[1]}/c{key[2]}" if len(key) == 3
                       else f"h{key[0]}/c{key[1]}")
                print(f"    {lbl}: {n} chosen, {a2/max(n,1):.0%} achieved", flush=True)
    torch.save(sel.state_dict(), sel_path)
    import json as _json
    _json.dump({(f"{k[0]}:{k[1]}/{k[2]}" if len(k) == 3 else f"{k[0]}/{k[1]}"):
                {"chosen": v[0], "achieved": v[1]} for k, v in stats.items()},
               open(sel_path.replace(".pt", "_stats.json"), "w"))
    print(f"[srl] saved {sel_path} + stats")


if __name__ == "__main__":
    main()
