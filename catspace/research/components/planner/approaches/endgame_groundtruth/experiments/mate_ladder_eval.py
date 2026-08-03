#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/mate_ladder_eval.py -- THE MATE MISSION EXAM (Kaveh 2026-07-23: "get the
current checkpoint to reasonably mate progressively harder toy scenarios without relying
on tablebases"). Graded difficulty ladder; the ENGINE uses only learned components +
search (tablebase-FREE at play); the tablebase referees (optimal defense = the exam).

Scenarios (progressively harder):
  KRvK-easy      3-piece, dtm <= 20
  KRRvK-central  the two-rook ladder, central king (the 0.12-pure / 0.75-oracle benchmark)
  KRRvKB         bishop interference, dtm <= 30
  KRRvKP         pawn racing, dtm <= 30
  KRRvKBP        the full toy, dtm <= 40

Configs: pure (no value) · dtm (learned DTM CNN) · escape (learned constraint net) ·
blend (escape + dtm). VERDICT per (scenario, config): mate rate, median plies, search/mate.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.ladder_mate import make_dtm_value, play_out, random_krrvk
from catspace.io import paths

VALUE_C = 8.0


def make_escape_value(ckpt, device="cpu"):
    import torch
    from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
    from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.train_dtm_cnn import DTMNet
    dev = pick_device(device)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = DTMNet(c=st["c"]).to(dev); net.load_state_dict(st["state"]); net.eval()
    scale = st.get("scale", 8.0)

    def value_fn(boards):
        pk = np.stack([encode_packed(b) for b in boards])
        mt = np.stack([encode_meta(b) for b in boards])
        with torch.no_grad():
            pred = net(torch.from_numpy(feature_planes(pk, mt)).to(dev)).cpu().numpy() * scale
        return np.tanh((VALUE_C - pred) / VALUE_C)      # smaller box = better (white POV)
    return value_fn


def make_blend_value(escape_fn, dtm_fn, w=0.5):
    def value_fn(boards):
        return w * escape_fn(boards) + (1 - w) * dtm_fn(boards)
    return value_fn


def sample_scenarios(rng, n):
    dz = np.load(paths.derived("dtm_endgame.npz"))
    P, M, dtm = np.asarray(dz["packed"]), np.asarray(dz["meta"]), np.asarray(dz["dtm"])
    mk = np.array(["".join(sorted(p.symbol() for p in board_from_packed(P[i], M[i]).piece_map().values()))
                   for i in range(len(P))])

    def pick(sig, lo, hi):
        idx = np.flatnonzero((mk == sig) & (dtm >= lo) & (dtm <= hi))
        rng.shuffle(idx)
        out = []
        for i in idx:
            b = board_from_packed(P[i], M[i])
            if b.turn == chess.WHITE and not b.is_game_over():
                out.append(b)
            if len(out) >= n:
                break
        return out

    def synth(extra_piece, lo, hi, tb):
        """5-piece scenarios direct-generated (the pool barely passes through them):
        KRR + k + one black minor/pawn; tb used only to certify the EXAM (won, in band)."""
        from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import rollout_dtm
        out = []
        for _ in range(n * 300):
            if len(out) >= n:
                break
            sqs = rng.choice(64, size=5, replace=False)
            b = chess.Board(None)
            b.set_piece_at(int(sqs[0]), chess.Piece(chess.KING, chess.WHITE))
            b.set_piece_at(int(sqs[1]), chess.Piece(chess.ROOK, chess.WHITE))
            b.set_piece_at(int(sqs[2]), chess.Piece(chess.ROOK, chess.WHITE))
            b.set_piece_at(int(sqs[3]), chess.Piece(chess.KING, chess.BLACK))
            pt = chess.BISHOP if extra_piece == "b" else chess.PAWN
            if pt == chess.PAWN and chess.square_rank(int(sqs[4])) in (0, 7):
                continue
            b.set_piece_at(int(sqs[4]), chess.Piece(pt, chess.BLACK))
            b.turn = chess.WHITE
            if not b.is_valid() or b.is_game_over():
                continue
            w, _d = tb.wdl_dtz(b)
            if w != 2:
                continue
            d0 = rollout_dtm(b, tb)
            if d0 is not None and lo <= d0 <= hi:
                out.append(b)
        return out

    tb_exam = TB()
    scen = [                                   # difficulty order (measured, Kaveh-corrected):
        ("KRRvK-central", [b for b in (random_krrvk(rng, central=True) for _ in range(n * 2)) if b][:n]),
        ("KRRvKB", pick("KRRbk", 8, 30) or synth("b", 8, 30, tb_exam)),
        ("KRRvKP", pick("KRRkp", 8, 30) or synth("p", 8, 30, tb_exam)),
        ("KRRvKBP", pick("KRRbkp", 8, 40)),
        ("KRvK-technique", pick("KRk", 4, 20)),   # one rook = the TECHNIQUE exam (0.00 baselines)
    ]
    tb_exam.close()
    return scen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="pure,dtm",
                    help="comma list: pure,dtm,escape,blend")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--nodes", type=int, default=600)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--dtm-ckpt", default=paths.sep("dtm_cnn.pt"))
    ap.add_argument("--escape-ckpt", default=paths.sep("escape_net_v1.pt"))
    ap.add_argument("--blend-w", type=float, default=0.5)
    ap.add_argument("--energy-ckpt", default=paths.sep("opponent_energy_v1_step8000.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed); tb = TB()

    # configs -> (value_fn, policy_fn)
    values = {"pure": (None, None)}
    if any(c in args.configs for c in ("dtm", "blend")):
        values["dtm"] = (make_dtm_value(args.dtm_ckpt), None)
    if "escape" in args.configs:
        values["escape"] = (make_escape_value(args.escape_ckpt), None)
    if any(c in args.configs for c in ("field", "fieldenergy")):
        fv = make_field_value()
        values["field"] = (fv, None)
    if any(c in args.configs for c in ("energy", "fieldenergy")):
        ep = make_energy_prior(args.energy_ckpt)
        values["energy"] = (None, ep)
    if "fieldenergy" in args.configs:
        values["fieldenergy"] = (fv, ep)

    scenarios = sample_scenarios(rng, args.n)
    print(f"[exam] scenarios: {[(s, len(b)) for s, b in scenarios]}  nodes={args.nodes}", flush=True)
    for cfg in args.configs.split(","):
        vfn, pfn = values[cfg]
        for name, starts in scenarios:
            if not starts:
                print(f"VERDICT MATE_LADDER cfg={cfg:7s} {name:14s} SKIPPED (no starts)", flush=True)
                continue
            res = [play_out(s, tb, args.nodes, args.max_plies, value_fn=vfn, policy_fn=pfn) for s in starts]
            mates = [p for m, p, _k, _n in res if m]
            sn = [nn for m, _p, _k, nn in res if m]
            rate = len(mates) / len(res)
            print(f"VERDICT MATE_LADDER cfg={cfg:7s} {name:14s} mate={rate:.2f} "
                  f"({len(mates)}/{len(res)})  med_plies={np.median(mates) if mates else float('nan'):.0f}  "
                  f"search/mate={np.median(sn) if sn else 0:,.0f}  [{time.time()-t0:.0f}s]", flush=True)
    tb.close()


# ---- Kaveh 2026-07-23: THE mating configuration = IQE field + flavored-energy prior ----
def make_field_value(field_ckpt=paths.sep("iqe_geom_field.pt"), device="cpu", c=6.0):
    """Value = field distance to the MATE BANK (goal-as-region: nearest exemplar, never a
    centroid). Re-test of the shelved field-value verdict, now on the HEALTHY field."""
    import torch
    from catspace.fields import FieldModel
    from catspace.research.tools.viz.viz_b_mate_clusters import harvest_mates
    fm = FieldModel(field_ckpt, device=device)
    rng = np.random.default_rng(7)
    mates, _labels = harvest_mates(paths.derived("dtm_endgame.npz"), 400, rng)
    bank = fm.embed_B_boards(mates)

    def value_fn(boards):
        d = fm.d_boards_to_bank(boards, bank)
        return np.tanh((c - d) / c)
    return value_fn


def make_energy_prior(ckpt=paths.sep("opponent_energy_v1.pt"), cohort=11, device="cpu"):
    """Move prior = the flavored-energy opponent model's policy for a STRONG cohort
    (11 = sf_full): 'what would strong play consider' as search ordering."""
    import torch
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.opponent import OpponentModel
    from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
    dev = pick_device(device)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = OpponentModel(**st["config"]).to(dev)
    net.load_state_dict(st["state"]); net.eval()
    L = st["config"].get("max_moves", 80)

    def policy_fn(b):
        moves = list(b.legal_moves)[:L]
        f = np.zeros((1, L), np.int64); t = np.zeros((1, L), np.int64)
        pc = np.zeros((1, L), np.int64); ct = np.zeros((1, L), np.int64)
        for j, m in enumerate(moves):
            f[0, j], t[0, j] = m.from_square, m.to_square
            pc[0, j] = b.piece_type_at(m.from_square) or 0
            cap = b.piece_type_at(m.to_square)
            ct[0, j] = cap or (1 if b.is_en_passant(m) else 0)
        pl = torch.from_numpy(feature_planes(encode_packed(b)[None], encode_meta(b)[None])).to(dev)
        with torch.no_grad():
            lg = net(pl, torch.from_numpy(f).to(dev), torch.from_numpy(t).to(dev),
                     torch.from_numpy(pc).to(dev), torch.from_numpy(ct).to(dev),
                     torch.tensor([len(moves)]).to(dev), torch.tensor([cohort]).to(dev))
        p = torch.softmax(lg[0, :len(moves)], 0).cpu().numpy()
        return {m: float(p[j]) for j, m in enumerate(moves)}
    return policy_fn


if __name__ == "__main__":
    main()
