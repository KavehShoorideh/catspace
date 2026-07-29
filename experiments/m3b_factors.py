#!/usr/bin/env python
"""experiments/m3b_factors.py -- M3b concept mining: matched case-control on SF-labeled positions.

Cases = realized crossings (mover_loss >= 0.2); controls = clear non-crossings (< 0.05);
ambiguous middle excluded. MATCHED via stratification on the confounders the M0/M2a analyses
exposed: committor_before bins (sharpness -- the dominant crossing driver), piece-count bins
(phase), and Elo band. Within each stratum, effect = mean(factor | case) - mean(factor | control);
combined across strata by case-count weights; CI = game-clustered bootstrap (TESTING §3 -- never
resample positions).

Direction: positive effect (more present before crossings) = ATTACKING/risk factor;
negative = PROTECTIVE factor. Gate (MILESTONES M3b): >= 5 significant each way.
Hand-coded extractors first (this file); SAE/CAV stack later (concept_extraction_stack memory).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FACTORS = [
    "hanging_own", "hanging_opp", "underdefended_own", "pins_own", "opp_checks",
    "tension", "king_ring_pressure", "king_open_files", "in_check", "queens_on",
    "mobility_low", "back_rank_weak", "all_defended", "king_shield", "king_escape",
    "no_pins", "low_tension", "opp_no_checks",
    # LATENT-threat block (2026-07-29 iteration: round 1 showed VISIBLE danger is protective --
    # crossings live where danger is hidden; these extract hidden affordances)
    "mate_threat", "fork_avail", "xray_royal", "discovery_avail", "overloaded_def",
    "barely_defended", "q_harass", "opp_adv_passer",
    # round-3 block: classic hidden affordances + visible-pressure candidates
    "loose_pieces", "opp_rook_7th", "king_uncastled_late", "opp_minor_near_king",
    "opp_pawn_storm", "opp_battery_king", "opp_visible_pressure", "we_checks",
]


def extract_chunk(fens):
    import chess
    out = np.zeros((len(fens), len(FACTORS)), np.float32)
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        us = b.turn; them = not us
        occ_us = [(sq, p) for sq, p in b.piece_map().items() if p.color == us and p.piece_type != chess.KING]
        hang_own = underdef = 0
        for sq, p in occ_us:
            att = len(b.attackers(them, sq)); dfd = len(b.attackers(us, sq))
            if att > 0 and dfd == 0:
                hang_own += 1
            elif att > dfd:
                underdef += 1
        hang_opp = sum(1 for sq, p in b.piece_map().items()
                       if p.color == them and p.piece_type != chess.KING
                       and len(b.attackers(us, sq)) > 0 and len(b.attackers(them, sq)) == 0)
        pins_own = sum(1 for sq, p in occ_us if b.is_pinned(us, sq))
        # opponent latent resources via the null-move view (skipped when in check: null illegal)
        opp_checks = mate_threat = fork_avail = q_harass = 0
        _qs = list(b.pieces(chess.QUEEN, us))
        qsq = _qs[0] if _qs else None
        if not b.is_check():
            bb = b.copy(stack=False)
            bb.push(chess.Move.null())
            HI = {chess.KING, chess.QUEEN, chess.ROOK}
            for m in bb.legal_moves:
                if bb.gives_check(m):
                    opp_checks += 1
                pc = bb.piece_at(m.from_square)
                bb.push(m)
                if bb.is_checkmate():
                    mate_threat = 1
                att = bb.attacks(m.to_square)
                hi_hits = sum(1 for sq in att
                              if (q := bb.piece_at(sq)) and q.color == us and q.piece_type in HI)
                if pc.piece_type == chess.KNIGHT and hi_hits >= 2:
                    fork_avail = 1
                if qsq is not None and pc.piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP) \
                        and qsq in att:
                    q_harass = 1
                bb.pop()
        tension = sum(1 for m in b.legal_moves if b.is_capture(m))
        ksq = b.king(us)
        ring = chess.SquareSet(chess.BB_KING_ATTACKS[ksq])
        ring_pressure = sum(len(b.attackers(them, sq)) for sq in ring)
        kfile = chess.square_file(ksq)
        open_files = sum(1 for f in {max(0, kfile - 1), kfile, min(7, kfile + 1)}
                         if not any(b.piece_at(chess.square(f, r)) and
                                    b.piece_at(chess.square(f, r)).piece_type == chess.PAWN and
                                    b.piece_at(chess.square(f, r)).color == us for r in range(8)))
        shield = sum(1 for sq in ring if b.piece_at(sq) and b.piece_at(sq).color == us
                     and b.piece_at(sq).piece_type == chess.PAWN)
        escape = sum(1 for sq in ring if not b.piece_at(sq) and not b.attackers(them, sq))
        mobility = b.legal_moves.count()
        back = chess.BB_RANK_1 if us == chess.WHITE else chess.BB_RANK_8
        back_weak = 1.0 if (chess.BB_SQUARES[ksq] & back) and escape == 0 else 0.0
        # x-ray / discovery: opponent slider aligned with mover K/Q through exactly one blocker
        xray = disc = 0
        royals = [b.king(us)] + list(b.pieces(chess.QUEEN, us))
        for sq, p in b.piece_map().items():
            if p.color != them or p.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                continue
            for r in royals:
                if r is None:
                    continue
                ray = chess.SquareSet.ray(sq, r)
                if not ray or r == sq:
                    continue
                between = chess.SquareSet.between(sq, r)
                if p.piece_type == chess.ROOK and chess.square_file(sq) != chess.square_file(r) \
                        and chess.square_rank(sq) != chess.square_rank(r):
                    continue
                if p.piece_type == chess.BISHOP and (chess.square_file(sq) == chess.square_file(r)
                                                     or chess.square_rank(sq) == chess.square_rank(r)):
                    continue
                blockers = [bs for bs in between if b.piece_at(bs)]
                if len(blockers) == 1:
                    xray = 1
                    if b.piece_at(blockers[0]).color == them:
                        disc = 1
        # overloaded defender: a mover piece that is the SOLE defender of >=2 attacked mover pieces
        defended_by = {}
        for sq, p in occ_us:
            if len(b.attackers(them, sq)) > 0:
                dfs = list(b.attackers(us, sq))
                if len(dfs) == 1:
                    defended_by[dfs[0]] = defended_by.get(dfs[0], 0) + 1
        overloaded = sum(1 for v in defended_by.values() if v >= 2)
        barely = sum(1 for sq, p in occ_us
                     if len(b.attackers(them, sq)) == 1 and len(b.attackers(us, sq)) == 1)
        opp_passer = 0
        for sq in b.pieces(chess.PAWN, them):
            rel = chess.square_rank(sq) if them == chess.WHITE else 7 - chess.square_rank(sq)
            if rel >= 5:
                f = chess.square_file(sq)
                ahead = [chess.square(ff, rr) for ff in {max(0,f-1), f, min(7,f+1)}
                         for rr in (range(chess.square_rank(sq)+1, 8) if them == chess.WHITE
                                    else range(0, chess.square_rank(sq)))]
                if not any((q := b.piece_at(a)) and q.color == us and q.piece_type == chess.PAWN
                           for a in ahead):
                    opp_passer = 1
        # round-3 extractors
        loose = sum(1 for sq, p2 in occ_us if len(b.attackers(us, sq)) == 0)   # LPDO
        r7 = our2 = chess.BB_RANK_2 if us == chess.WHITE else chess.BB_RANK_7
        opp_r7 = sum(1 for sq in b.pieces(chess.ROOK, them) | b.pieces(chess.QUEEN, them)
                     if chess.BB_SQUARES[sq] & r7)
        start_k = chess.E1 if us == chess.WHITE else chess.E8
        uncastled = float(b.fullmove_number > 12 and ksq in
                          (start_k, start_k + (1 if us == chess.WHITE else -1) * 0) and
                          bool(b.pieces(chess.QUEEN, them)))
        near_k = sum(1 for sq in (b.pieces(chess.KNIGHT, them) | b.pieces(chess.BISHOP, them))
                     if chess.square_distance(sq, ksq) <= 2)
        storm = sum(1 for sq in b.pieces(chess.PAWN, them)
                    if abs(chess.square_file(sq) - kfile) <= 1 and
                    (chess.square_rank(sq) <= 3 if us == chess.WHITE else chess.square_rank(sq) >= 4))
        batt = sum(len(b.attackers(them, sq)) >= 2 for sq in ring)
        vis_pressure = sum(1 for sq, p2 in occ_us if len(b.attackers(them, sq)) > 0)
        we_checks = sum(1 for m in b.legal_moves if b.gives_check(m))
        out[i] = [
            hang_own, hang_opp, underdef, pins_own, opp_checks,
            tension, ring_pressure, open_files, float(b.is_check()),
            float(len(b.pieces(chess.QUEEN, us)) + len(b.pieces(chess.QUEEN, them)) > 0),
            float(mobility <= 10), back_weak,
            float(hang_own == 0 and underdef == 0), float(shield >= 2), float(escape >= 2),
            float(pins_own == 0), float(tension <= 2), float(opp_checks == 0),
            mate_threat, fork_avail, xray, disc, overloaded,
            barely, q_harass, opp_passer,
            loose, float(opp_r7 > 0), uncastled, float(near_k > 0),
            float(storm >= 2), float(batt >= 1), vis_pressure, we_checks,
        ]
    return out


def stratified_effect(f, m_case, strat_):
    """case-control effect within strata, case-count weighted (the M3b matched harness)."""
    eff, wsum = 0.0, 0.0
    for st in np.unique(strat_):
        m = strat_ == st
        nc, nk = (m & m_case).sum(), (m & ~m_case).sum()
        if nc >= 20 and nk >= 20:
            eff += nc * (f[m & m_case].mean() - f[m & ~m_case].mean()); wsum += nc
    return eff / max(wsum, 1)


def build_strata(cb, pieces, elo):
    s_cb = np.digitize(cb, [0.2, 0.35, 0.65, 0.8])
    s_pc = np.digitize(pieces, [10, 20, 27])
    return s_cb * 100 + s_pc * 10 + (np.asarray(elo) >= 1500).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--case-thr", type=float, default=0.2)
    ap.add_argument("--ctrl-thr", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    d = dict(np.load(args.labeled, allow_pickle=True))
    ok = ~np.isnan(d["mover_loss"])
    fen = d["fen"][ok]; y = d["mover_loss"][ok]; cb = d["committor_before"][ok]
    elo = d["elo_mover"][ok]; game = d["game"][ok].astype(np.int64)
    case = y >= args.case_thr; ctrl = y < args.ctrl_thr
    keep = case | ctrl
    print(f"{ok.sum():,} labeled | cases {case.sum():,} controls {ctrl.sum():,} "
          f"(ambiguous excluded {(~keep).sum():,})")

    chunks = np.array_split(np.flatnonzero(keep), args.workers * 8)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        feats = np.concatenate(list(ex.map(extract_chunk,
                                           [fen[c].tolist() for c in chunks])))
    idx = np.concatenate(chunks)
    ycase = case[idx]; cbk = cb[idx]; elok = elo[idx]; gk = game[idx]
    import chess
    pieces = np.array([sum(ch.isalpha() for ch in f.split()[0]) for f in fen[idx]])
    print(f"extracted {feats.shape} in {time.time()-t0:.0f}s")

    # strata: sharpness x phase x band (the confounder set)
    strat = build_strata(cbk, pieces, elok)

    rng = np.random.default_rng(0)
    games = np.unique(gk)
    gidx = {g: np.flatnonzero(gk == g) for g in games}
    results = []
    for j, name in enumerate(FACTORS):
        obs = stratified_effect(feats[:, j], ycase, strat)
        boots = np.empty(args.n_boot)
        for bi in range(args.n_boot):
            picks = rng.integers(0, len(games), len(games))
            rows = np.concatenate([gidx[games[p]] for p in picks])
            boots[bi] = stratified_effect(feats[rows, j], ycase[rows], strat[rows])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = (lo > 0) or (hi < 0)
        results.append((name, obs, lo, hi, sig))

    atk = [(n, e, l, h) for n, e, l, h, s in results if s and e > 0]
    prot = [(n, e, l, h) for n, e, l, h, s in results if s and e < 0]
    print("\nVERDICT M3b factors (stratified case-control, game-clustered 95% CI):")
    for n, e, l, h in sorted(atk, key=lambda r: -r[1]):
        print(f"  ATTACKING  {n:<18} +{e:.4f} [{l:+.4f},{h:+.4f}]")
    for n, e, l, h in sorted(prot, key=lambda r: r[1]):
        print(f"  PROTECTIVE {n:<18} {e:+.4f} [{l:+.4f},{h:+.4f}]")
    ns = [n for n, *_, s in [(r[0], r[4]) for r in results] if not s]
    ns = [r[0] for r in results if not r[4]]
    print(f"  not significant: {', '.join(ns) if ns else 'none'}")
    print(f"VERDICT M3b GATE: {len(atk)} attacking + {len(prot)} protective significant "
          f"(need >=5 + >=5) -- {'PASS' if len(atk) >= 5 and len(prot) >= 5 else 'FAIL'}")


if __name__ == "__main__":
    main()
