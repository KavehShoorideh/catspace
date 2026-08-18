#!/usr/bin/env python
"""gen_event_labels.py -- the next N INTERESTING EVENTS and the material they leave behind.

"let's change the objective functions so that we predict the next five interesting events,
whether they're captures or promotions, and we predict the piece count or type of pieces that
are available at that stage of the game." (Kaveh 2026-08-17)

WHY THIS REPLACES gen_capture_labels. The captures-only labels were diagnosed to fail at naming
the ending for two measured reasons, and this fixes both at the source:

  1 PROMOTIONS ARE INVISIBLE. Material = current minus victims reproduces the true ending
    signature 0.0% of the time when a promotion happens (13.2% of the relevant rows). A pawn
    becoming a queen changes the material with nothing leaving the board, so a capture-only event
    stream literally cannot represent it. Promotions are events here.

  2 DERIVING MATERIAL IS A CONJUNCTION. Reconstructing the ending by chaining victims needs EVERY
    victim right at once: per-capture victim accuracy runs 0.436 / 0.412 / 0.389 / 0.348 / 0.352,
    and the joint lands at 10.4%. So the material after each event is emitted as its OWN label --
    a direct prediction rather than a product of five fragile ones.

Emitted per position row, for each of the next N events:
    ev_type    int8   1 capture, 2 promotion, 3 capture-with-promotion, -1 none
    ev_sq      int8   destination square 0-63
    ev_mover   int8   piece id 1-12 AFTER the move (a promotion shows the NEW piece)
    ev_victim  int8   piece id 1-12 captured, 0 if none
    ev_ply     int16  plies from this row until the event
    ev_counts  int8   (N, 12) piece counts AFTER the event, jepa order PNBRQK pnbrqk
    ev_men     int8   total men after the event
    ev_sig     int16  index into a signature vocabulary ('KRP|KR'), -1 if unseen

    .venv/bin/python -m ...gen_event_labels --n-events 5 --save
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

_LETTER = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
_ORDER = "KQRBNP"


def sig_of_counts(c):
    """c = 12 counts in jepa order -> 'KRP|KR', white left of the bar (same convention as
    gen_tb_entry_labels.signature, so the two label sets speak the same language)."""
    w = {_LETTER[i + 1]: int(c[i]) for i in range(6)}
    b = {_LETTER[i + 1]: int(c[i + 6]) for i in range(6)}
    f = lambda d: "".join(p * d[p] for p in _ORDER if d.get(p, 0) > 0)
    return f"{f(w)}|{f(b)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-events", type=int, default=5)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    tok, n = tr.tok, len(tr.tok)
    gz = np.load(paths.derived(f"game_transitions_FULL_{n}.npz"))
    par, mid = gz["par"], gz["mid"]

    # per-row piece counts, once -- everything below is derived from these
    counts = np.zeros((n, 12), np.int8)
    for t in range(1, 13):
        counts[:, t - 1] = (tok == t).sum(1)
    men = counts.sum(1).astype(np.int8)

    # ---- find the events -------------------------------------------------------------
    # A CAPTURE removes a man. A PROMOTION converts a pawn into a non-pawn of the same colour,
    # leaving the man count alone unless it is also a capture. Diffing the counts detects both,
    # and detects the two happening on the same move, which a to-square test would miss.
    c_par, c_kid = counts[par].astype(np.int16), counts[par + 1].astype(np.int16)
    d = c_kid - c_par
    is_cap = men[par + 1] < men[par]
    w_promo = (d[:, 0] < 0) & (d[:, 1:6].sum(1) > 0)
    b_promo = (d[:, 6] < 0) & (d[:, 7:12].sum(1) > 0)
    is_promo = w_promo | b_promo
    ev_mask = is_cap | is_promo
    ev_rows = par[ev_mask]
    ev_kind = np.where(is_cap[ev_mask] & is_promo[ev_mask], 3,
                       np.where(is_cap[ev_mask], 1, 2)).astype(np.int8)
    to_sq = mid[ev_mask, 1].astype(np.int64)
    mover = tok[ev_rows + 1, to_sq].astype(np.int8)
    victim = tok[ev_rows, to_sq].astype(np.int8)
    # EN PASSANT: the target square was empty yet a man vanished; the victim is the pawn beside
    # it, of the opposite colour to the capturer. Reconstructed rather than dropped -- silently
    # mislabelling ~1% of captures would poison the pawn class.
    ep = is_cap[ev_mask] & (victim == 0)
    victim[ep] = np.where(mover[ep] <= 6, 7, 1)
    victim[~is_cap[ev_mask]] = 0

    print(f"[ev] {len(ev_rows):,} events in {len(par):,} plies ({100*len(ev_rows)/len(par):.1f}%)")
    kc = Counter(ev_kind.tolist())
    print(f"[ev]   captures {kc.get(1,0):,} | promotions {kc.get(2,0):,} | "
          f"capture+promotion {kc.get(3,0):,}")
    print(f"[ev]   captures-only labels would MISS {kc.get(2,0):,} pure promotions "
          f"({100*kc.get(2,0)/max(1,len(ev_rows)):.2f}% of events)")

    # ---- per row, the next N events ---------------------------------------------------
    N = args.n_events
    ev_type = np.full((n, N), -1, np.int8)
    ev_sq = np.full((n, N), -1, np.int8)
    ev_mv = np.full((n, N), -1, np.int8)
    ev_vic = np.full((n, N), -1, np.int8)
    ev_ply = np.full((n, N), -1, np.int16)
    ev_cnt = np.full((n, N, 12), -1, np.int8)
    ev_men = np.full((n, N), -1, np.int8)

    order = np.argsort(ev_rows)
    er, ek, es, em, evi = (ev_rows[order], ev_kind[order], to_sq[order].astype(np.int8),
                           mover[order], victim[order])
    for gi in range(len(tr.start)):
        s0, ln = int(tr.start[gi]), int(tr.length[gi])
        lo = np.searchsorted(er, s0); hi = np.searchsorted(er, s0 + ln)
        if hi <= lo:
            continue
        ge, gk, gs, gm, gv = er[lo:hi], ek[lo:hi], es[lo:hi], em[lo:hi], evi[lo:hi]
        j = 0
        for r in range(s0, s0 + ln):
            while j < len(ge) and ge[j] < r:
                j += 1
            k = min(N, len(ge) - j)
            if k <= 0:
                continue
            sl = slice(j, j + k)
            ev_type[r, :k] = gk[sl]; ev_sq[r, :k] = gs[sl]
            ev_mv[r, :k] = gm[sl]; ev_vic[r, :k] = gv[sl]
            ev_ply[r, :k] = (ge[sl] - r).astype(np.int16)
            # the material AFTER each event is just the child position's counts
            ev_cnt[r, :k] = counts[ge[sl] + 1]
            ev_men[r, :k] = men[ge[sl] + 1]

    have = ev_type[:, N - 1] >= 0
    print(f"[ev] positions with all {N} events ahead: {int(have.sum()):,} "
          f"({100*have.mean():.1f}%)")
    dd = ev_ply[have, N - 1]
    print(f"[ev] plies to the {N}th event: median {np.median(dd):.0f} "
          f"p25 {np.percentile(dd,25):.0f} p75 {np.percentile(dd,75):.0f}")

    # ---- signature vocabulary over the post-event material ---------------------------
    # The signature index is kept for ANALYSIS only. The model predicts the 12 COUNTS, because
    # the intermediate stages have tens of thousands of distinct signatures -- a softmax over
    # those is hopeless, while 12 small integers compose into exactly the same thing and each is
    # individually easy. int32 because the vocabulary overflows int16.
    vocab = {}
    uniq, inv = np.unique(ev_cnt.reshape(-1, 12), axis=0, return_inverse=True)
    names = [sig_of_counts(u) if (u >= 0).all() else "" for u in uniq]
    for nm in names:
        if nm:
            vocab.setdefault(nm, len(vocab))
    remap = np.array([vocab.get(nm, -1) if nm else -1 for nm in names], np.int32)
    sig_idx = remap[inv].reshape(n, N)
    print(f"[ev] {len(vocab):,} distinct post-event material signatures "
          f"(analysis only -- the model predicts the 12 counts)")
    mx = int(ev_cnt.max())
    print(f"[ev] max count in any single piece class: {mx}  -> {mx+1}-way heads x 12")

    # ---- THE GO/NO-GO: do 5 events reach a 5-man tablebase? --------------------------
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    tv = [str(x) for x in lz["vocab"]]
    cls, plies, m0 = lz["tb_class"].astype(np.int64), lz["tb_plies"], lz["men"]
    is_eg = np.array([not v.startswith("WDL:") and "=" in v for v in tv])
    band = (cls >= 0) & (plies >= 18) & (plies <= 22) & (m0 > 5) & is_eg[cls.clip(0)]
    rows = np.flatnonzero(band)
    reach = np.zeros(len(rows), bool); at_k = np.full(len(rows), -1)
    for k in range(N):
        hit = (ev_men[rows, k] == 5) & (~reach)
        at_k[hit] = k; reach |= hit
    print(f"\n[ev] GO/NO-GO on the {len(rows):,} endgame_20 rows "
          f"(18-22 plies before a 5-man entrance):")
    print(f"[ev]   an event brings the board to exactly 5 men within {N} events: "
          f"{100*reach.mean():.1f}%")
    for k in range(N):
        print(f"[ev]     by event {k+1}: {100*(at_k[(at_k>=0)&(at_k<=k)].shape[0]/len(rows)):.1f}%")
    ok = reach & (sig_idx[rows, np.clip(at_k, 0, N - 1)] >= 0)
    print(f"[ev]   ...and its signature is in the vocabulary: {100*ok.mean():.1f}%")

    if args.save:
        p = paths.derived(f"event_labels_{N}_{n}.npz")
        np.savez_compressed(p, ev_type=ev_type, ev_sq=ev_sq, ev_mover=ev_mv, ev_victim=ev_vic,
                            ev_ply=ev_ply, ev_counts=ev_cnt, ev_men=ev_men, ev_sig=sig_idx,
                            sig_vocab=np.array(list(vocab.keys()), dtype=object))
        print(f"\n[ev] wrote {p}")


if __name__ == "__main__":
    main()
