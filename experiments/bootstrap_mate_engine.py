#!/usr/bin/env python
"""experiments/bootstrap_mate_engine.py -- BOOTSTRAP MATE ENGINE (Kaveh 2026-07-24): NO
external mate bank. The engine starts knowing nothing; MCTS (energy prior + mate_stop) probes
the reachability field, every checkmate LEAF the search touches is harvested into an online
bank (own experience only), and the value becomes distance-to-DISCOVERED-mates. One knob:
search budget (--nodes). Question: at what budget does KRRvK-central hit 100%?

Fast path: priors cached by position (stable net), F-embeddings cached by position (stable
towers), only the bank-min-distance recomputed as the bank grows (cheap). Field net on MPS.
Games parallelized across workers sharing discoveries via an append-only FEN file.

Run (launcher spawns workers):   --nodes 5000 --n 48 --j 4
Single worker (internal):        --nodes 5000 --n 48 --j 4 --worker 0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.engine.fields import FieldModel
from catspace.nn.mcts import MCTS
from catspace.tb import TB, tb_best_move
from experiments.mate_ladder_eval import make_energy_prior, sample_scenarios


class OnlineMateBank:
    """Episodic memory of mates the ENGINE found (rules-certified terminal states). Shared
    across workers via an append-only FEN file; embeddings computed locally, deduped by EPD."""

    def __init__(self, fm: FieldModel, bank_file: Path):
        self.fm = fm; self.bank_file = bank_file
        self.keys: set[str] = set(); self.embs: np.ndarray | None = None

    def _embed_add(self, boards):
        E = self.fm.embed_B_boards(boards)
        self.embs = E if self.embs is None else np.concatenate([self.embs, E])

    def sync(self):
        """pick up other workers' discoveries (eventually-consistent shared memory)."""
        if not self.bank_file.exists():
            return
        new = []
        for line in self.bank_file.read_text().splitlines():
            epd = line.strip()
            if epd and epd not in self.keys:
                self.keys.add(epd); new.append(chess.Board(epd))
        if new:
            self._embed_add(new)

    def add(self, boards) -> int:
        fresh = []
        for b in boards:
            epd = b.epd()
            if epd not in self.keys:
                self.keys.add(epd); fresh.append(b)
        if fresh:
            self._embed_add(fresh)
            with open(self.bank_file, "a") as f:
                f.writelines(b.epd() + "\n" for b in fresh)
        return len(fresh)

    def __len__(self):
        return len(self.keys)


def harvest(root) -> list:
    """all checkmate leaves the search TOUCHED (Black to move & mated = White wins)."""
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        if n.board is not None and n.board.turn == chess.BLACK and n.board.is_checkmate():
            out.append(n.board)
        stack.extend(n.children)
    return out


def make_boot_value(fm: FieldModel, bank: OnlineMateBank):
    """value = tanh((M - dmin)/M), M = running median of observed dmin (self-calibrating:
    no field-scale constant; ordering is what MCTS needs). Bank empty -> 0 (prior-only)."""
    emb_cache: dict[str, np.ndarray] = {}
    recent = deque(maxlen=512)

    def value_fn(boards):
        if len(bank) == 0:
            return np.zeros(len(boards))
        miss = [b for b in boards if b.epd() not in emb_cache]
        if miss:
            E = fm.embed_F_boards(miss)
            for b, e in zip(miss, E):
                emb_cache[b.epd()] = e
        F = np.stack([emb_cache[b.epd()] for b in boards])
        d = fm.d_to_bank(F, bank.embs)
        recent.extend(d.tolist())
        M = max(float(np.median(recent)), 1e-6)
        return np.tanh((M - d) / M)
    return value_fn


def cached_prior(pfn):
    cache: dict[str, dict] = {}

    def policy_fn(b):
        k = b.epd()
        if k not in cache:
            cache[k] = pfn(b)
        return cache[k]
    return policy_fn


def worker(args):
    t0 = time.time(); tb = TB()
    starts = dict(sample_scenarios(np.random.default_rng(args.seed), args.n))[args.scenario]
    fm = FieldModel(args.field, device=args.device)
    bank = OnlineMateBank(fm, Path(args.bank_file))
    vfn = make_boot_value(fm, bank)
    pfn = cached_prior(make_energy_prior(ckpt=args.energy_ckpt, device="cpu"))

    my_games = list(range(args.worker, len(starts), args.j))
    results = []
    for gi in my_games:
        bank.sync()
        b = starts[gi].copy(stack=False)
        plies = 0; nodes_spent = 0; tmoves = []; found_this_game = 0
        while plies < args.max_plies and not b.is_game_over(claim_draw=True):
            if b.turn == chess.WHITE:
                tm = time.time()
                m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=args.nodes, mate_stop=True,
                         pw_c=1.5, root_min_visits=10, value_fn=vfn, policy_fn=pfn,
                         batch_leaves=32)
                root = m.run(b)
                found_this_game += bank.add(harvest(root))
                best = max(root.children,
                           key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)))
                nodes_spent += m.evals_used; tmoves.append(time.time() - tm)
                b.push(best.move)
            else:
                b.push(tb_best_move(b, tb))
            plies += 1
        out = b.outcome(claim_draw=True)
        mated = bool(out and out.winner == chess.WHITE)
        results.append((gi, mated, plies, nodes_spent, sum(tmoves), len(tmoves)))
        print(f"  g{gi:03d} {'mate' if mated else 'FAIL'} plies={plies} bank={len(bank)}(+{found_this_game}) "
              f"t/move={np.median(tmoves):.1f}s t/game={sum(tmoves):.0f}s "
              f"nodes/s={nodes_spent/max(sum(tmoves),1e-9):.0f} [{time.time()-t0:.0f}s]", flush=True)
    tb.close()
    m_ = [r for r in results if r[1]]
    print(f"[worker {args.worker}] {len(m_)}/{len(results)} mate  "
          f"med t/move={np.median([t/max(k,1) for _, _, _, _, t, k in results]):.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--j", type=int, default=4)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--scenario", default="KRRvK-central")
    ap.add_argument("--field", default="data/derived/sep/lichess_mc2.pt")
    ap.add_argument("--energy-ckpt", default="data/derived/sep/opponent_energy_v1.pt")
    ap.add_argument("--bank-file", default=None)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.bank_file is None:
        args.bank_file = f"artifacts/experiments/boot_bank_n{args.nodes}.fens"

    if args.worker is not None:
        worker(args); return

    Path(args.bank_file).unlink(missing_ok=True)
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--worker", str(w)])
             for w in range(args.j)]
    for p in procs:
        p.wait()
    # aggregate: replay worker stdout is interleaved above; final bank + verdict from file
    n_bank = len(set(Path(args.bank_file).read_text().splitlines())) if Path(args.bank_file).exists() else 0
    print(f"VERDICT BOOTSTRAP_MATE scenario={args.scenario} nodes={args.nodes} n={args.n} "
          f"bank_final={n_bank}  [{time.time()-t0:.0f}s] -- per-game lines above; "
          f"grep ' g' | mate rate = share of 'mate' lines", flush=True)


if __name__ == "__main__":
    main()
