"""qgt_rule_mine.py -- first pass of the LOGICAL-REASONING LAYER (Kaveh 8/29:
"I wanna have logical reasoning over past concept sets that are discrete").

Mines implications between ORACLE PREDICATES (python-chess ground truth about a
move: capture, check, to-square defended, leaves a piece hanging) and MODEL
EVENTS (which concept bits flip at which squares under the learned transition).
Two directions:

  rule:      predicate -> bit-flip      "captures flip bit 7 at the to-square"
  semantics: bit-flip -> predicate      "bit 7 flipping MEANS a capture, p=0.9"

High-lift, high-support pairs are the raw material for the rules layer: named
tactical motifs written in the model's own concept alphabet, each independently
checkable against the oracle. Association mining only (support/confidence/lift)
-- no ILP machinery until the alphabet proves stable.

    .venv/bin/python -m catspace.research.tools.training_infra.qgt_rule_mine \
        --ckpt artifacts/experiments/qgt11_step8000.pt --n 300
"""
from __future__ import annotations

import argparse
import glob
import json

import chess
import numpy as np
import torch

PREDS = ("capture", "check", "to_defended", "leaves_hang", "quiet")


def predicates(b0: chess.Board, mv: chess.Move):
    """Oracle predicates of (position, move) -- exact, from python-chess."""
    cap = b0.is_capture(mv)
    b1 = b0.copy(stack=False)
    b1.push(mv)
    chk = b1.is_check()
    tod = bool(b0.attackers(not b0.turn, mv.to_square))
    hang = False
    for s in chess.SquareSet(b1.occupied_co[not b1.turn]):   # mover's pieces now
        if b1.piece_type_at(s) == chess.KING:
            continue
        if b1.attackers(b1.turn, s) and not b1.attackers(not b1.turn, s):
            hang = True
            break
    return {"capture": cap, "check": chk, "to_defended": tod,
            "leaves_hang": hang, "quiet": not (cap or chk)}, b1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/qgt11_step8000.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--moves-per", type=int, default=2)
    ap.add_argument("--out", default="artifacts/experiments/rules_qgt11.json")
    args = ap.parse_args()
    from catspace.research.tools.npz_io import load_npz
    from catspace.research.tools.training_infra.qgt_search import QGTDynamics
    import chess.engine
    dyn = QGTDynamics(args.ckpt)
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Threads": 1})
    rng = np.random.default_rng(0)
    fens = []
    for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:10]:
        z = load_npz(f)
        if "wdl" not in z.files:
            continue
        for i in np.flatnonzero(z["kind"] == 0):
            fens.append(str(z["fen"][i]))
            if len(fens) >= args.n:
                break
        if len(fens) >= args.n:
            break

    B = dyn.model.sq_bits
    # counts[loc][pred or "_all"] -> (B,) flip counts; loc: to/from/hangsq
    LOCS = ("to", "frm", "hangsq")
    cnt = {L: {p: np.zeros(B) for p in PREDS + ("_all",)} for L in LOCS}
    nn_ = {p: 0 for p in PREDS + ("_all",)}
    n_hang = 0
    with torch.no_grad():
        for fen in fens:
            try:
                b0 = chess.Board(fen)
            except Exception:
                continue
            state, ctx = dyn.encode(b0)
            moves, codes, frm, to = dyn.root_moves(b0, ctx)
            if not moves:
                continue
            sel = rng.choice(len(moves), min(args.moves_per, len(moves)),
                             replace=False)
            for j in sel:
                mv = moves[j]
                pr, b1 = predicates(b0, mv)
                reps = list(b1.legal_moves)
                if not reps:
                    continue
                rm = sf.play(b1, chess.engine.Limit(depth=8)).move
                if rm is None or rm not in reps:
                    continue
                nxt = dyn.step(state, codes[j:j + 1], frm[j:j + 1], to[j:j + 1],
                               frm2=torch.tensor([rm.from_square],
                                                 device=frm.device),
                               to2=torch.tensor([rm.to_square],
                                                device=frm.device))
                flips = (nxt[0] != state[0]).cpu().numpy()       # (64, B)
                where = {"to": mv.to_square, "frm": mv.from_square,
                         "hangsq": None}
                if pr["leaves_hang"]:
                    b2 = b1.copy(stack=False)
                    b2.push(rm)
                    for s in chess.SquareSet(b2.occupied_co[b2.turn]):
                        if b1.piece_type_at(s) and not b2.piece_type_at(s):
                            where["hangsq"] = s                  # it got eaten
                            break
                    n_hang += 1
                nn_["_all"] += 1
                for p in PREDS:
                    if pr[p]:
                        nn_[p] += 1
                for L in LOCS:
                    s = where[L]
                    if s is None:
                        continue
                    cnt[L]["_all"] += flips[s]
                    for p in PREDS:
                        if pr[p]:
                            cnt[L][p] += flips[s]
    sf.quit()
    print(f"[mine] {nn_['_all']} (position, move, SF-reply) transitions | "
          f"predicate counts: " +
          " ".join(f"{p}={nn_[p]}" for p in PREDS) + f" | hang-sq located {n_hang}")
    rules = []
    for L in LOCS:
        base = cnt[L]["_all"] / max(nn_["_all"], 1)              # P(flip_k at L)
        for p in PREDS:
            if nn_[p] < 8:
                continue
            conf = cnt[L][p] / nn_[p]                            # P(flip | pred)
            for k in range(B):
                if conf[k] > 0.25 and nn_[p] >= 8:
                    lift = conf[k] / max(base[k], 1e-4)
                    if lift > 2.0 or conf[k] > 0.6:
                        rules.append(dict(loc=L, pred=p, bit=k,
                                          conf=round(float(conf[k]), 3),
                                          base=round(float(base[k]), 3),
                                          lift=round(float(lift), 1),
                                          n=int(nn_[p])))
    rules.sort(key=lambda r: -r["conf"] * min(r["lift"], 10))
    print(f"[mine] {len(rules)} rules above threshold (conf>0.25 & lift>2, "
          f"or conf>0.6). Top 20:")
    for r in rules[:20]:
        print(f"  {r['pred']:>12} -> bit{r['bit']:>2}@{r['loc']:<6} "
              f"conf {r['conf']:.2f} (base {r['base']:.2f}, lift {r['lift']}x, "
              f"n={r['n']})")
    with open(args.out, "w") as f:
        json.dump(rules, f, indent=1)
    print(f"[mine] full set -> {args.out}")


if __name__ == "__main__":
    main()
