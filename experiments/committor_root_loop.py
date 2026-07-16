#!/usr/bin/env python
"""
experiments/committor_root_loop.py — the closed loop from THE root (Kaveh GO,
2026-07-15): start every rollout at the canonical KRRvKBP position, balance
exploration/exploitation with epsilon, and train the field on the data as it
arrives. Round-based:

  round r: generate N eps-rollouts from the root (White = current field + eps,
           Black = tb-optimal; 5 seed-split workers, v2 dumps)
        -> CUMULATIVE per-boundary table (all loop rounds so far; earlier
           rounds' rollouts come from weaker policies -- accepted
           non-stationarity, the data is still real play)
        -> distill d_W (+d_D) into the CURRENT lineage checkpoint
        -> gates: held-out rho / rim rho must not fall more than --gate-slack
           below the reigning champion's; conversion-from-root probe (eps play,
           the root's own P-hat) tracked as a trajectory metric
        -> ADVANCE the lineage only if gates pass (ratchet; a failed round
           still contributes its DATA to later rounds).

All numbers are parsed from the child scripts' printed VERDICT lines and
appended to artifacts/experiments/committor_loop_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.selfplay_generate import KRRKBP_FIXED_START
from experiments.value_fixed_point import TB, tb_best_move

PY = sys.executable
EXP = Path("artifacts/experiments")


def run(cmd, log):
    print(f"+ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(log.read_text() + r.stdout + r.stderr if log.exists()
                   else r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[1]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def root_probe(ckpt, whead_path, root_fen, n_games, nodes, eps, seed, device):
    """Conversion from THE root under eps-play (the root's own P-hat)."""
    import torch
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.mcts import FBMCTSPolicy
    dev = pick_device(device)
    fb, pay = load_ckpt(Path(ckpt), dev)
    hp = torch.load(whead_path, map_location=dev, weights_only=False)
    whead = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
    whead.load_state_dict(hp["state"]); whead.eval()
    tb = TB("data/syzygy")
    wins = 0
    for g in range(n_games):
        pol = FBMCTSPolicy(fb, pay["zgoals"]["MATE_W"], max_nodes=nodes, device=dev,
                           committor_head=whead)
        rng = np.random.default_rng([seed, g])
        b = chess.Board(root_fen)
        seen = set()
        for _ in range(120):
            if b.is_game_over(claim_draw=True):
                break
            if b.turn == chess.WHITE:
                if rng.random() < eps:
                    ms = list(b.legal_moves)
                    m = ms[int(rng.integers(len(ms)))]
                else:
                    m = pol.move(b, rng)
            else:
                m = tb_best_move(b, tb, seen)
                seen.add(b.board_fen())
            if m is None:
                break
            b.push(m)
        out = b.outcome(claim_draw=True)
        wins += int(bool(out and out.winner == chess.WHITE))
    tb.close()
    return wins / n_games


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-in", default="data/derived/sep/committor_joint.pt")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--rollouts-per-worker", type=int, default=400)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--distill-steps", type=int, default=4000)
    ap.add_argument("--probe-games", type=int, default=32)
    ap.add_argument("--probe-nodes", type=int, default=200)
    ap.add_argument("--gate-slack", type=float, default=0.02)
    ap.add_argument("--root-fen", default=KRRKBP_FIXED_START)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="rootloop")
    args = ap.parse_args()

    loop_log = EXP / "committor_loop_log.jsonl"
    champ = args.ckpt_in                     # reigning lineage checkpoint
    champ_whead = args.ckpt_in.replace(".pt", "_whead.pt")
    best_rho, best_rim = -np.inf, -np.inf
    dumps = []

    for r in range(1, args.rounds + 1):
        print(f"===== ROUND {r} (champion: {champ}) =====", flush=True)
        # 1. generate from THE root, seed-split workers
        procs = []
        for w in range(args.workers):
            dump = EXP / f"rollout_dump_{args.tag}_r{r}_w{w}.jsonl"
            dumps.append(str(dump))
            starts = EXP / f"{args.tag}_root.json"
            starts.write_text(json.dumps(dict(fens=[args.root_fen])))
            cmd = [PY, "-u", "experiments/certainty_rollouts.py",
                   "--ckpt", champ, "--committor-head", champ_whead,
                   "--starts", str(starts), "--n-starts", "1",
                   "--rollouts", str(args.rollouts_per_worker),
                   "--epsilon", str(args.epsilon), "--nodes", str(args.nodes),
                   "--search", "mcts", "--max-plies", "100", "--min-visits", "4",
                   "--seed", str(args.seed + 1000 * r + w),
                   "--out", f"/tmp/{args.tag}_r{r}_w{w}.json",
                   "--dump-rollouts", str(dump), "--device", args.device]
            procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL))
        for p in procs:
            if p.wait() != 0:
                raise RuntimeError(f"round {r}: generation worker failed")

        # 2. cumulative table
        table = EXP / f"certainty_table_{args.tag}.json"
        out = run([PY, "-u", "experiments/table_from_dump.py", "--dump", *dumps,
                   "--min-visits", "4", "--tb-sample", "300",
                   "--out", str(table)], Path(f"/tmp/{args.tag}_table_r{r}.log"))
        kept = int(re.search(r"(\d+) kept", out).group(1))
        grad = re.search(r"gradient Spearman\(P-hat, -\|dtz\|\) = ([+-]\d+\.\d+)", out)

        # 3. distill into the champion lineage
        cand = f"data/derived/sep/{args.tag}_r{r}.pt"
        out = run([PY, "-u", "experiments/committor_distill.py", "--loss", "mse",
                   "--table", str(table), "--ckpt-in", champ, "--ckpt-out", cand,
                   "--steps", str(args.distill_steps), "--eval-every", "500",
                   "--patience", "4", "--device", args.device],
                  Path(f"/tmp/{args.tag}_distill_r{r}.log"))
        rho = float(re.search(r"-> head ([+-]\d+\.\d+)", out).group(1))
        rim_m = re.search(r"RIM_RESOLUTION.*-> head ([+-]\d+\.\d+)", out)
        rim = float(rim_m.group(1)) if rim_m else float("nan")

        # 4. gates + probe
        advanced = (rho >= best_rho - args.gate_slack
                    and (np.isnan(rim) or rim >= best_rim - args.gate_slack))
        cand_whead = cand.replace(".pt", "_whead.pt")
        conv = root_probe(cand if advanced else champ,
                          cand_whead if advanced else champ_whead,
                          args.root_fen, args.probe_games, args.probe_nodes,
                          args.epsilon, args.seed + r, args.device)
        if advanced:
            champ, champ_whead = cand, cand_whead
            best_rho, best_rim = max(best_rho, rho), (max(best_rim, rim)
                                                      if not np.isnan(rim) else best_rim)
        rec = dict(round=r, kept_states=kept,
                   table_gradient=(float(grad.group(1)) if grad else None),
                   rho=rho, rim=rim, advanced=bool(advanced), root_conv=conv,
                   champion=champ)
        with open(loop_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"VERDICT LOOP_ROUND {json.dumps(rec)}", flush=True)

    print(f"LOOP_DONE champion={champ}")


if __name__ == "__main__":
    main()
