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


def paired_rho(ckpt_a, whead_a, ckpt_b, whead_b, table_path, rim_plies=8.0, seed=0,
               device="auto"):
    """Champion (a) and candidate (b) scored on the SAME holdout rows of the
    CURRENT table -- the gate must be a paired difference, never a comparison
    of rhos measured on different tables (holdout target noise attenuates rho
    as the cumulative table grows; measured rounds 5-6)."""
    import torch
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.data.encode import encode_meta, encode_packed
    from catspace.nn.features import feature_planes, omega_ids
    from experiments.certainty_distill import spearman_ci
    dev = pick_device(device)
    rows = json.loads(Path(table_path).read_text())["rows"]
    rng = np.random.default_rng(seed)
    hold = [rows[i] for i in rng.permutation(len(rows))[:max(500, len(rows) // 5)]]
    t = np.array([-np.log(max(r["p_hat"], 1.0 / (r["n"] + 2))) for r in hold])
    rim = np.array([r["plies"] is not None and r["plies"] <= rim_plies for r in hold])

    def head_d(ckpt, whead_path):
        fb, _ = load_ckpt(Path(ckpt), dev)
        hp = torch.load(whead_path, map_location=dev, weights_only=False)
        head = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                                   torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
        head.load_state_dict(hp["state"]); head.eval(); fb.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(hold), 512):
                ch = hold[i:i + 512]
                boards = [chess.Board(r["fen"]) for r in ch]
                packed = np.stack([encode_packed(b) for b in boards])
                meta = np.stack([encode_meta(b) for b in boards])
                om = omega_ids(np.full(len(ch), 1800), np.full(len(ch), 1800),
                               np.full(len(ch), np.nan))
                f = fb.embed_F(torch.from_numpy(feature_planes(packed, meta)).to(dev),
                               torch.from_numpy(om).to(dev))
                out.append(head(f).squeeze(-1).cpu().numpy())
        return np.concatenate(out)

    da, db = head_d(ckpt_a, whead_a), head_d(ckpt_b, whead_b)
    ra = spearman_ci(da, t)[0]; rb = spearman_ci(db, t)[0]
    rim_a = spearman_ci(da[rim], t[rim])[0] if rim.sum() >= 30 else float("nan")
    rim_b = spearman_ci(db[rim], t[rim])[0] if rim.sum() >= 30 else float("nan")
    return ra, rb, rim_a, rim_b


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
    ap.add_argument("--play-slack", type=float, default=0.125,
                    help="play gate: candidate root-conv may trail the champion's "
                         "(same seed set, same round) by at most this (4/32 games). "
                         "PLAY IS THE ARBITER -- round 7 advanced a field-better/"
                         "play-worse candidate before this gate existed")
    ap.add_argument("--gate-slack", type=float, default=0.02)
    ap.add_argument("--rim-slack", type=float, default=0.12,
                    help="separate (wider) slack for the rim gate: its held-out "
                         "subset is tens of rows, so round-to-round swings of "
                         "~0.1 are sampling noise (measured r1-r3: +.26/-.12/+.01)")
    ap.add_argument("--resume", action="store_true",
                    help="glob this tag's existing round dumps so a restarted "
                         "loop keeps the cumulative table")
    ap.add_argument("--init-best-rho", type=float, default=None,
                    help="seed the ratchet with the pre-restart champion's rho")
    ap.add_argument("--init-best-rim", type=float, default=None)
    ap.add_argument("--start-round", type=int, default=1,
                    help="first round number (restart bookkeeping)")
    ap.add_argument("--root-fen", default=KRRKBP_FIXED_START)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="rootloop")
    args = ap.parse_args()

    loop_log = EXP / f"committor_loop_log_{args.tag}.jsonl"
    champ = args.ckpt_in                     # reigning lineage checkpoint
    champ_whead = args.ckpt_in.replace(".pt", "_whead.pt")
    best_rho = args.init_best_rho if args.init_best_rho is not None else -np.inf
    best_rim = args.init_best_rim if args.init_best_rim is not None else -np.inf
    dumps = []
    if args.resume:
        dumps = sorted(str(p) for p in EXP.glob(f"rollout_dump_{args.tag}_r*_w*.jsonl"))
        print(f"resume: {len(dumps)} existing dumps for tag {args.tag}")

    for r in range(args.start_round, args.start_round + args.rounds):
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
                   "--head-init", champ_whead,
                   "--steps", str(args.distill_steps), "--eval-every", "500",
                   "--patience", "4", "--device", args.device],
                  Path(f"/tmp/{args.tag}_distill_r{r}.log"))
        rho = float(re.search(r"-> head ([+-]\d+\.\d+)", out).group(1))

        # 4. PAIRED gate (same holdout rows, same round) + probe
        cand_whead = cand.replace(".pt", "_whead.pt")
        champ_rho, cand_rho, champ_rim, cand_rim = paired_rho(
            champ, champ_whead, cand, cand_whead, table,
            seed=args.seed + 10_000 + r, device=args.device)
        rim = cand_rim
        field_ok = (cand_rho >= champ_rho - args.gate_slack
                    and (np.isnan(cand_rim) or np.isnan(champ_rim)
                         or cand_rim >= champ_rim - args.rim_slack))
        # PLAY GATE: both arms probed on the same seed set, every round
        champ_conv = root_probe(champ, champ_whead, args.root_fen, args.probe_games,
                                args.probe_nodes, args.epsilon, args.seed + r, args.device)
        cand_conv = root_probe(cand, cand_whead, args.root_fen, args.probe_games,
                               args.probe_nodes, args.epsilon, args.seed + r, args.device)
        advanced = field_ok and (cand_conv >= champ_conv - args.play_slack)
        conv = cand_conv if advanced else champ_conv
        if advanced:
            champ, champ_whead = cand, cand_whead
        rec = dict(round=r, kept_states=kept,
                   table_gradient=(float(grad.group(1)) if grad else None),
                   rho=rho, champ_rho=champ_rho, cand_rho=cand_rho,
                   champ_rim=(None if np.isnan(champ_rim) else champ_rim),
                   cand_rim=(None if np.isnan(cand_rim) else cand_rim),
                   champ_conv=champ_conv, cand_conv=cand_conv,
                   advanced=bool(advanced), root_conv=conv, champion=champ)
        with open(loop_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"VERDICT LOOP_ROUND {json.dumps(rec)}", flush=True)

    print(f"LOOP_DONE champion={champ}")


if __name__ == "__main__":
    main()
