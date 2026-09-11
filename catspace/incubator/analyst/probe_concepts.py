"""probe_concepts.py -- WHAT HAS THE KNOWLEDGE NETWORK LEARNED, asked in the rules' vocabulary (Kaveh 2026-09-11:
"did we get 'take the centre'? use the rules engine to probe the concepts it has learned").

Four probes, all readable because the network's representation is a sum of named senses (Backpack):
  atlas    the linear heads (the evaluation now, the sharpness ahead, the future codes) dotted with every sense
           vector: which (piece, question, answer) pushes the value up or down, in words
  squares  the empirical worth of a piece type on each square: the mean contribution of the piece's tokens to the
           evaluation, over held-out positions, squares mirrored so ranks read from the mover's side
  probes   linear probes from phi for concepts the RULES define but the inputs do not name -- central pawns,
           centre control, development, space, castled, outposts, rooks on open files, pawn islands -- against the
           same probe on the pooled input features (is the concept computed by the network, or merely passed in?)
  moves    the network's rating of quiet moves by the rules' kind (a pawn to the centre, another pawn move, a minor
           piece developing, castling, a rook to an open file, a king move...): the policy head's preference, and
           the value head's change on the child position for pawn moves to the centre against other pawn moves

    python -m catspace.incubator.analyst.probe_concepts all [--ckpt ...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .graph import NODE_QUESTIONS, POSITION_QUESTIONS, build
from .memo import Memo
from . import knowledge_net as K
from .planner import DEFAULT_CKPT
from .experts import net_sample

CENTRE = {chess.D4, chess.E4, chess.D5, chess.E5}
BIG_CENTRE = CENTRE | {chess.C4, chess.F4, chess.C5, chess.F5, chess.C3, chess.D3, chess.E3, chess.F3, chess.C6, chess.D6, chess.E6, chess.F6}
TYPE_NAMES = {0: "pad", 1: "our pawn", 2: "our knight", 3: "our bishop", 4: "our rook", 5: "our queen", 6: "our king",
              7: "their pawn", 8: "their knight", 9: "their bishop", 10: "their rook", 11: "their queen", 12: "their king"}


def load(ckpt):
    model, cfg, device = K.load_model(ckpt)
    meta = json.load(open(Path(ckpt).parent / "meta.json"))
    buckets = K.Buckets.from_state(cfg.buckets, meta["buckets"]); pos_buckets = K.Buckets.from_state(cfg.buckets, meta["pos_buckets"])
    return model, cfg, device, buckets, pos_buckets


def bucket_text(buckets: K.Buckets, q: int, k: int) -> str | None:
    """The answer range of bucket k of question q; None for a bucket slot the quantiles never use (untrained)."""
    e = buckets.edges[q]
    if k > len(e):
        return None
    lo = e[k - 1] if k >= 1 else None; hi = e[k] if k < len(e) else None
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"< {hi:g}"
    if hi is None:
        return f">= {lo:g}"
    return f"{lo:g}" if abs(hi - lo) < 1e-6 else f"[{lo:g}, {hi:g})"


# ---------------------------------------------------------------- atlas
@torch.no_grad()
def atlas(model, cfg, buckets, data: "K.Data | None" = None, top: int = 16, min_count: int = 300) -> list[str]:
    """Per (piece type, question): how the head's reading changes from the LOWEST answer that occurs for that type
    to the HIGHEST (both with at least `min_count` occurrences in the training positions), so 'our pawn: cone2 up
    -> value up' is one line. Senses for answers a type never gives are skipped (they are untrained)."""
    S = model.sense.weight.detach().cpu().numpy()                                       # (13*Q*B, d)
    Q, B = K.Q, cfg.buckets
    counts = np.zeros((13, Q, B), np.int64)
    if data is not None:
        tr = data.ids("train")
        t = data.types[tr]; nb = data.node_b[tr]                                        # (N, 32), (N, 32, Q)
        m = t > 0
        for q in range(Q):
            np.add.at(counts[:, q, :], (t[m], nb[m][:, q]), 1)
    heads = {"the evaluation now (cp head)": model.cp_head.weight[0].detach().cpu().numpy(),
             "sharpness ahead (a big threat within 10 plies)": model.sharp_head.weight[0].detach().cpu().numpy()}
    out = []
    for name, w in heads.items():
        eff = (S @ w).reshape(13, Q, B)
        rows = []
        for t in range(1, 13):
            for q in range(Q):
                ks = [k for k in range(B) if counts[t, q, k] >= min_count and bucket_text(buckets, q, k) is not None]
                if len(ks) < 2:
                    continue
                lo, hi = ks[0], ks[-1]
                rows.append((float(eff[t, q, hi] - eff[t, q, lo]), t, q, lo, hi, int(counts[t, q, lo]), int(counts[t, q, hi])))
        rows.sort()
        out.append(f"[{name}] from the lowest answer that occurs to the highest, per (piece, question): the {top} most NEGATIVE and most POSITIVE slopes")
        for e, t, q, lo, hi, c0, c1 in rows[:top]:
            out.append(f"   {e:+.3f}  {TYPE_NAMES[t]}: {NODE_QUESTIONS[q]} {bucket_text(buckets, q, lo)} -> {bucket_text(buckets, q, hi)}   (n {c0} / {c1})")
        out.append("   ...")
        for e, t, q, lo, hi, c0, c1 in rows[-top:][::-1]:
            out.append(f"   {e:+.3f}  {TYPE_NAMES[t]}: {NODE_QUESTIONS[q]} {bucket_text(buckets, q, lo)} -> {bucket_text(buckets, q, hi)}   (n {c0} / {c1})")
        by_q = defaultdict(float)
        for e, t, q, *_ in rows:
            by_q[q] += abs(e)
        out.append("   the questions the head reads most (sum of |slope| over piece types): " + ", ".join(f"{NODE_QUESTIONS[q]} {v:.1f}" for q, v in sorted(by_q.items(), key=lambda kv: -kv[1])[:12]))
    return out


# ---------------------------------------------------------------- phi and per-piece contributions over the test split
@torch.no_grad()
def contributions(model, cfg, data: K.Data, ids, device, batch: int = 256):
    """Per position: phi, and per piece its total contribution to the cp head (all its senses, identity and square),
    plus its (type, mirrored square)."""
    w_cp = model.cp_head.weight[0]
    phis, rows = [], []
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]
        b = data.batch(chunk, device)
        o = model(b)
        a = o["alpha"]; senses = model.sense(model.sense_index(b)); ts, ss = model.type_sense(b["types"]), model.square_sense(b["squares"])
        contrib = (a[..., :K.Q, None] * senses).sum(2) + a[..., K.Q:K.Q + 1] * ts + a[..., K.Q + 1:K.Q + 2] * ss          # (B, 32, d)
        per_piece = (contrib @ w_cp) / o["norm"]                                                                      # (B, 32)
        phis.append(o["phi"].cpu().numpy())
        pp = per_piece.cpu().numpy(); types = b["types"].cpu().numpy(); sqs = b["squares"].cpu().numpy(); mask = b["mask"].cpu().numpy()
        for j, i in enumerate(chunk):
            side = data.meta[int(i)]["side"]
            for p in range(K.MAX_PIECES):
                if mask[j, p]:
                    sq = int(sqs[j, p]); sq_m = sq if side else (sq ^ 56)
                    rows.append((int(types[j, p]), sq_m, float(pp[j, p])))
    return np.concatenate(phis), rows


def squares_map(rows, top_types=(1, 2, 3)) -> list[str]:
    out = []
    for t in top_types:
        grid = defaultdict(list)
        for tt, sq, v in rows:
            if tt == t:
                grid[sq].append(v)
        out.append(f"[{TYPE_NAMES[t]}] mean contribution to the evaluation by square (x100, ranks from our side, n in brackets where < 20):")
        for r in range(7, -1, -1):
            cells = []
            for f in range(8):
                sq = chess.square(f, r); v = grid.get(sq, [])
                cells.append(f"{np.mean(v) * 100:+5.1f}" + (f"({len(v)})" if 0 < len(v) < 20 else "     ") if v else "   .      ")
            out.append(f"   {r + 1}  " + " ".join(cells))
        out.append("      " + "   ".join(f"   {chess.FILE_NAMES[f]}   " for f in range(8)))
        centre = [v for tt, sq, v in rows if tt == t and sq in CENTRE]; other = [v for tt, sq, v in rows if tt == t and sq not in CENTRE]
        out.append(f"   centre squares (d4 e4 d5 e5): {np.mean(centre) * 100:+.2f} (n {len(centre)}) against elsewhere {np.mean(other) * 100:+.2f} (n {len(other)})")
    return out


# ---------------------------------------------------------------- rules-defined concepts and linear probes
def concepts_of(fen: str, side: bool) -> dict:
    b = chess.Board(fen); us = side
    m = (lambda sq: sq) if us else (lambda sq: sq ^ 56)
    pawns = [m(s) for s in b.pieces(chess.PAWN, us)]
    out = {}
    out["central pawns (d4 e4 d5 e5)"] = sum(1 for s in pawns if s in CENTRE)
    out["pawns in the big centre"] = sum(1 for s in pawns if s in BIG_CENTRE)
    out["centre control (our attacks on d4 e4 d5 e5)"] = sum(len(b.attackers(us, sq if us else sq ^ 56)) for sq in CENTRE)
    minors = [m(s) for s in b.pieces(chess.KNIGHT, us) | b.pieces(chess.BISHOP, us)]
    out["development (minor pieces off the back rank)"] = sum(1 for s in minors if chess.square_rank(s) > 0)
    out["space (our pawns beyond the 4th rank)"] = sum(1 for s in pawns if chess.square_rank(s) >= 4)
    k = b.king(us); km = m(k) if k is not None else None
    out["castled (king on g1/c1, no rights)"] = int(km in (chess.G1, chess.C1, chess.B1, chess.H1) and not b.has_castling_rights(us)) if km is not None else 0
    files = sorted({chess.square_file(s) for s in pawns})
    out["pawn islands"] = sum(1 for i, f in enumerate(files) if i == 0 or files[i - 1] != f - 1)
    from .structure import piece_facts, file_facts
    rels = piece_facts(b) + file_facts(b)
    out["outposts"] = sum(1 for r in rels if r.name == "outpost" and r.owner == us)
    out["rooks on open or half-open files"] = sum(1 for r in rels if r.name in ("rook_on_open_file", "rook_on_half_open_file") and r.owner == us)
    out["their king's open files (rules: king_open_file)"] = 0
    from .structure import king_facts
    out["their king's open files (rules: king_open_file)"] = sum(1 for r in king_facts(b) if r.name == "king_open_file" and r.owner == (not us))
    return out


def probes(model, cfg, data: K.Data, device, n_train: int = 20000, n_test: int = 4000, seed: int = 0) -> list[str]:
    from sklearn.linear_model import Ridge
    from .trajectory_check import features as pooled_features
    rng = np.random.default_rng(seed)
    tr = data.ids("train"); te = data.ids("test")
    tr = rng.choice(tr, min(n_train, len(tr)), replace=False); te = te[:n_test]
    t0 = time.perf_counter()
    phi_tr, _ = contributions(model, cfg, data, tr, device); phi_te, _ = contributions(model, cfg, data, te, device)
    Xin = pooled_features(dict(nodes=data.nodes, mask=data.mask, position=data.position))
    xin_tr, xin_te = Xin[tr], Xin[te]
    mu, sd = xin_tr.mean(0), xin_tr.std(0) + 1e-6; xin_tr = (xin_tr - mu) / sd; xin_te = (xin_te - mu) / sd
    C_tr = [concepts_of(data.meta[int(i)]["fen"], bool(data.meta[int(i)]["side"])) for i in tr]
    C_te = [concepts_of(data.meta[int(i)]["fen"], bool(data.meta[int(i)]["side"])) for i in te]
    out = [f"[probes] {len(tr)} training and {len(te)} held-out positions; phi and concepts in {time.perf_counter() - t0:.0f}s",
           f"   {'concept':<52}{'from phi':>10}{'from inputs':>13}{'(R2; > inputs = computed by the network)':>20}"]
    for name in C_tr[0]:
        y_tr = np.array([c[name] for c in C_tr], np.float32); y_te = np.array([c[name] for c in C_te], np.float32)
        if y_te.var() < 1e-6:
            continue
        r_phi = Ridge(alpha=1.0).fit(phi_tr, y_tr); r_in = Ridge(alpha=10.0).fit(xin_tr, y_tr)
        r2 = lambda p: 1 - ((p - y_te) ** 2).mean() / y_te.var()
        out.append(f"   {name:<52}{r2(r_phi.predict(phi_te)):>10.3f}{r2(r_in.predict(xin_te)):>13.3f}")
    return out


# ---------------------------------------------------------------- moves by the rules' kind
def move_kind_rules(board: chess.Board, m: chess.Move) -> str:
    us = board.turn; mm = (lambda sq: sq) if us else (lambda sq: sq ^ 56)
    p = board.piece_at(m.from_square)
    if board.is_capture(m) or board.gives_check(m) or m.promotion:
        return "forcing"
    if board.is_castling(m):
        return "castles"
    if p.piece_type == chess.PAWN:
        return "pawn to the centre" if mm(m.to_square) in CENTRE else ("pawn to the big centre" if mm(m.to_square) in BIG_CENTRE else "pawn elsewhere")
    if p.piece_type in (chess.KNIGHT, chess.BISHOP):
        if chess.square_rank(mm(m.from_square)) == 0:
            return "minor piece develops"
        return "minor piece to the centre" if mm(m.to_square) in BIG_CENTRE else "minor piece elsewhere"
    if p.piece_type == chess.ROOK:
        f = chess.square_file(m.to_square)
        own = any(chess.square_file(s) == f for s in board.pieces(chess.PAWN, us))
        return "rook to an open or half-open file" if not own else "rook elsewhere"
    if p.piece_type == chess.QUEEN:
        return "queen move"
    return "king move"


@torch.no_grad()
def moves(model, cfg, data: K.Data, device, buckets, pos_buckets, n: int = 1500, n_child: int = 120, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    te = [int(i) for i in data.ids("test")]; rng.shuffle(te)
    pref = defaultdict(list); best_kind = defaultdict(int); n_best = 0
    for i in te[:n]:
        m = data.meta[i]; board = chess.Board(m["fen"])
        b = data.batch([i], device); o = model(b)
        pl = model.policy_logits(o["h"], b)[0]
        lp = F.log_softmax(pl.view(-1), 0).view(K.MAX_PIECES, 64)
        idx = {s: j for j, s in enumerate(m["squares"]) if s < 64}
        legal = list(board.legal_moves)
        scores = {mv: float(lp[idx[mv.from_square], mv.to_square]) for mv in legal}
        order = sorted(legal, key=lambda mv: -scores[mv])
        for mv in legal:
            kind = move_kind_rules(board, mv)
            pref[kind].append((scores[mv], order.index(mv) + 1, len(legal)))
        if m["best"]:
            best_kind[move_kind_rules(board, chess.Move.from_uci(m["best"]))] += 1; n_best += 1
    out = [f"[moves] the policy head over {n} held-out positions, quiet moves by the rules' kind:",
           f"   {'kind':<36}{'n':>7}{'mean log p':>11}{'mean rank':>11}{'top-3 share':>13}{'engine best share':>19}"]
    for kind, rows in sorted(pref.items(), key=lambda kv: -np.mean([r[0] for r in kv[1]])):
        out.append(f"   {kind:<36}{len(rows):>7}{np.mean([r[0] for r in rows]):>11.2f}{np.mean([r[1] for r in rows]):>11.1f}{np.mean([r[1] <= 3 for r in rows]):>13.1%}{best_kind[kind] / max(1, n_best):>19.1%}")
    # the value head on the CHILD: a pawn move to the centre against the other pawn moves of the same position
    memo = Memo(); deltas = defaultdict(list); done = 0
    for i in te:
        if done >= n_child:
            break
        m = data.meta[i]; board = chess.Board(m["fen"])
        pawn_moves = [mv for mv in board.legal_moves if board.piece_at(mv.from_square).piece_type == chess.PAWN and not board.is_capture(mv) and not board.gives_check(mv) and not mv.promotion]
        kinds = {mv: move_kind_rules(board, mv) for mv in pawn_moves}
        if "pawn to the centre" not in kinds.values() or "pawn elsewhere" not in kinds.values():
            continue
        try:
            g0 = build(board, memo); s0, _ = net_sample(g0, board, buckets, pos_buckets); v0 = float(model.cp_head(model(K.collate([s0], device))["phi"])[0, 0])
            for mv, kind in kinds.items():
                nb = board.copy(stack=False); nb.push(mv)
                if nb.is_game_over():
                    continue
                g1 = build(nb, memo); s1, _ = net_sample(g1, nb, buckets, pos_buckets); o1 = model(K.collate([s1], device))
                v1 = -float(model.cp_head(o1["phi"])[0, 0])                                         # from OUR side: the child is the opponent's to move
                sh = float(torch.sigmoid(model.sharp_head(o1["phi"])[0, 0]))
                deltas[kind].append((v1 - v0, sh))
        except Exception:
            continue
        done += 1
    out.append(f"[moves] the value head on the child, {done} positions that have both a pawn move to the centre and a pawn move elsewhere:")
    for kind in ("pawn to the centre", "pawn to the big centre", "pawn elsewhere"):
        rows = deltas.get(kind, [])
        if rows:
            out.append(f"   {kind:<28} n={len(rows):<5} mean change of the evaluation (tanh units x100) {np.mean([r[0] for r in rows]) * 100:+.2f}; predicted sharpness after it {np.mean([r[1] for r in rows]):.2f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["atlas", "squares", "probes", "moves", "all"]); ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--data", default="data/derived/knowledge")
    a = ap.parse_args(argv)
    model, cfg, device, buckets, pos_buckets = load(a.ckpt)
    data = K.Data(a.data, buckets=buckets, n_buckets=cfg.buckets, n_codes=cfg.n_codes, log=lambda *x, **k: None)
    if a.what in ("atlas", "all"):
        print("\n".join(atlas(model, cfg, buckets, data)), flush=True)
    if a.what in ("squares", "probes", "moves", "all"):
        if a.what in ("squares", "all"):
            _, rows = contributions(model, cfg, data, data.ids("test")[:4000], device)
            print("\n".join(squares_map(rows)), flush=True)
        if a.what in ("probes", "all"):
            print("\n".join(probes(model, cfg, data, device)), flush=True)
        if a.what in ("moves", "all"):
            print("\n".join(moves(model, cfg, data, device, buckets, pos_buckets)), flush=True)


if __name__ == "__main__":
    main()
