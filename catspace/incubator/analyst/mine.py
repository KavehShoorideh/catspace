"""THE MINER (plan §37 phase three), first experiment: counterexample-driven refinement on the PIN.

Mathematicians' fourth mechanism: state the cheap over-general claim, collect its counterexamples EXHAUSTIVELY
(the tablebase can, Stockfish cannot), and read the minimal guard off them. The pin is the test because we know
the answer in advance: "a pinned piece does not move" is over-general, and the guard is "unless its move gives
check, or takes the pinner, or the pin is only relative and the loss behind it is worth less than the move wins".
If the miner recovers that guard from the corpus without it being written, the miner can be trusted where we do
not know the answer.

The claim is a CONSTRAINT lemma, stated with its frame so it transfers out of tablebase territory:
    frame: an enemy slider S, our piece A on the line between S and our piece B, value(B) >= value(A), nothing
           else on the segments S..A and A..B.
    claim: a move of A that leaves the line loses the result.
The oracle is the corpus's exact partition of moves (endgames_v2: every legal move's result).

    python -m catspace.incubator.analyst.mine pin [--max-rows 100000]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import chess
import numpy as np

from .endgames import load as load_rows
from .facts import value as piece_value

OUT = Path("data/derived/mine")


def pins_of(board: chess.Board, side: chess.Color) -> list[tuple[int, int, int, bool]]:
    """(A, S, B, absolute): our piece A shielded from the enemy slider S by nothing, with our B behind it.
    B is the king for an absolute pin. Frame: the segments S..A and A..B hold nothing else."""
    out = []
    for a, pa in board.piece_map().items():
        if pa.color != side:
            continue
        for s, ps in board.piece_map().items():
            if ps.color == side or ps.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
                continue
            df, dr = chess.square_file(a) - chess.square_file(s), chess.square_rank(a) - chess.square_rank(s)
            straight = (df == 0) != (dr == 0); diag = abs(df) == abs(dr) and df != 0
            if not ((straight and ps.piece_type in (chess.QUEEN, chess.ROOK)) or (diag and ps.piece_type in (chess.QUEEN, chess.BISHOP))):
                continue
            if any(board.piece_at(t) for t in chess.SquareSet.between(s, a)):
                continue                                                   # the frame: S sees A
            step = ((df > 0) - (df < 0), (dr > 0) - (dr < 0))
            f, r = chess.square_file(a) + step[0], chess.square_rank(a) + step[1]
            b_sq = None
            while 0 <= f < 8 and 0 <= r < 8:
                t = chess.square(f, r)
                if board.piece_at(t) is not None:
                    b_sq = t; break
                f += step[0]; r += step[1]
            if b_sq is None:
                continue
            pb = board.piece_at(b_sq)
            if pb.color != side:
                continue
            out.append((a, s, b_sq, pb.piece_type == chess.KING))
    return out


def features(board: chess.Board, a: int, s: int, b_sq: int, m: chess.Move) -> dict:
    """The candidate guards, all from the rules: what the move of A does."""
    nb = board.copy(stack=False); nb.push(m)
    return dict(behind_is_dearer=piece_value(board, b_sq) >= piece_value(board, a),     # the condition that MAKES it a pin: now a candidate guard, not a hypothesis
                behind_is_cheaper=piece_value(board, b_sq) < piece_value(board, a),
                gives_check=nb.is_check(),
                takes_the_pinner=m.to_square == s,
                stays_on_the_line=m.to_square in chess.SquareSet.between(s, b_sq) or m.to_square == s,
                captures=board.is_capture(m),
                absolute=board.piece_at(b_sq).piece_type == chess.KING,
                b_defended=bool(board.attackers(board.turn, b_sq) - {a}),
                gain_ge_loss=(piece_value(board, m.to_square) if board.is_capture(m) else 0.0) >= piece_value(board, b_sq) - 0.5)


def pin(max_rows: int, log=print):
    rows = load_rows(Path("data/derived/endgames_v2"))[:max_rows]
    n_pos = n_pins = 0
    tally = Counter()                       # (keeps, guard-tuple) -> count
    by_guard = defaultdict(lambda: [0, 0])  # guard name -> [keeps, total] among off-line moves
    off_line_total = off_line_keeps = 0
    examples = defaultdict(list)
    for r in rows:
        b = chess.Board(r["fen"]); moves = r["moves"]
        if not moves:
            continue
        best = max(np.sign(v) for v in moves.values())
        ps = pins_of(b, b.turn)
        if not ps:
            continue
        n_pos += 1
        for a, s, b_sq, absolute in ps:
            n_pins += 1
            for m in b.legal_moves:
                if m.from_square != a or m.uci() not in moves:
                    continue
                f = features(b, a, s, b_sq, m)
                if f["stays_on_the_line"] and not f["takes_the_pinner"]:
                    continue                                    # not a move that breaks the pin
                keeps = np.sign(moves[m.uci()]) == best
                off_line_total += 1; off_line_keeps += keeps
                for g in ("gives_check", "takes_the_pinner", "captures", "absolute", "b_defended", "gain_ge_loss", "behind_is_dearer", "behind_is_cheaper"):
                    if f[g]:
                        by_guard[g][0] += keeps; by_guard[g][1] += 1
                key = tuple(sorted(k for k in ("gives_check", "takes_the_pinner", "captures", "behind_is_dearer", "gain_ge_loss") if f[k]))
                tally[(bool(keeps), key)] += 1
                if keeps and len(examples[key]) < 2:
                    examples[key].append((r["fen"], b.san(m)))
    log(f"[mine/pin] {len(rows)} positions scanned; {n_pos} with a pin in the frame ({n_pins} pins); {off_line_total} moves of the pinned piece that leave the line")
    log(f"[mine/pin] THE NAIVE CLAIM 'a pinned piece's move off the line loses the result': it is WRONG in {off_line_keeps}/{off_line_total} = {100 * off_line_keeps / max(1, off_line_total):.1f}% of cases")
    log("[mine/pin] candidate guards, each: among the off-line moves that satisfy it, how many keep the result")
    for g, (k, t) in sorted(by_guard.items(), key=lambda kv: -(kv[1][0] / max(1, kv[1][1]))):
        log(f"    {g:<18} keeps {k}/{t} = {100 * k / max(1, t):.1f}%   (the base rate is {100 * off_line_keeps / max(1, off_line_total):.1f}%)")
    log("[mine/pin] the conjunctions that survive, biggest first (keeps / total):")
    combos = defaultdict(lambda: [0, 0])
    for (keeps, key), c in tally.items():
        combos[key][0] += c * keeps; combos[key][1] += c
    for key, (k, t) in sorted(combos.items(), key=lambda kv: -kv[1][1])[:10]:
        name = " AND ".join(key) if key else "(no guard: a plain move off the line)"
        log(f"    {name:<60} {k}/{t} = {100 * k / max(1, t):.1f}%" + (f"   e.g. {examples[key][0][1]} in {examples[key][0][0]}" if examples[key] else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(off_line_total=int(off_line_total), off_line_keeps=int(off_line_keeps),
                   guards={g: [int(x) for x in v] for g, v in by_guard.items()},
                   combos={" AND ".join(k) or "(none)": [int(x) for x in v] for k, v in combos.items()}),
              open(OUT / "pin.json", "w"), indent=1)


def pin_games(n: int, depth: int, lo: int, hi: int, within: int = 50, log=print):
    """The same experiment where the pin actually lives: middlegame positions, the oracle a search (Stockfish at
    a fixed depth). 'Keeps' = the move is within `within` centipawns of the engine's best. Only the pinned piece's
    off-line moves are searched, so the cost is a handful of calls a position."""
    from .dataset import load as load_ds
    from .engine import Engine
    import random as _r
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    rows = [m for m in meta if lo <= m["ply"] <= hi]
    _r.Random(0).shuffle(rows)
    eng = Engine(depth=depth, cache_path=str(OUT / f"pin_oracle_d{depth}.sqlite"))
    by_guard = defaultdict(lambda: [0, 0]); combos = defaultdict(lambda: [0, 0]); examples = defaultdict(list)
    off_total = off_keeps = n_pos = 0
    import time as _t; t0 = _t.perf_counter()
    for m_ in rows:
        if n_pos >= n:
            break
        b = chess.Board(m_["fen"])
        if b.is_game_over():
            continue
        ps = pins_of(b, b.turn)
        if not ps:
            continue
        cand = []
        for a_, s_, b_sq, absolute in ps:
            for mv in b.legal_moves:
                if mv.from_square != a_:
                    continue
                f = features(b, a_, s_, b_sq, mv)
                if f["stays_on_the_line"] and not f["takes_the_pinner"]:
                    continue
                cand.append((mv, f))
        if not cand:
            continue
        line = eng.best(b, depth=depth)
        if line is None:
            continue
        n_pos += 1
        for mv, f in cand:
            e = eng.eval_move(b, mv, depth=depth)
            if e is None:
                continue
            keeps = (float(line.cp) - float(e.cp)) <= within
            off_total += 1; off_keeps += keeps
            for g in ("gives_check", "takes_the_pinner", "captures", "absolute", "b_defended", "gain_ge_loss", "behind_is_dearer", "behind_is_cheaper"):
                if f[g]:
                    by_guard[g][0] += keeps; by_guard[g][1] += 1
            key = tuple(sorted(k for k in ("gives_check", "takes_the_pinner", "captures", "behind_is_dearer", "gain_ge_loss") if f[k]))
            combos[key][0] += keeps; combos[key][1] += 1
            if keeps and len(examples[key]) < 2:
                examples[key].append((m_["fen"], b.san(mv)))
        if n_pos % 25 == 0:
            log(f"[mine/pin-games] {n_pos}/{n} positions with a pin, {off_total} moves judged, {_t.perf_counter() - t0:.0f}s, engine calls {eng.calls} hits {eng.hits}", flush=True)
    eng.close()
    log(f"[mine/pin-games] {n_pos} middlegame positions with a pin in the frame; {off_total} moves of the pinned piece off the line, judged by Stockfish depth {depth} (within {within} cp of its best)")
    log(f"[mine/pin-games] THE NAIVE CLAIM 'the pinned piece does not move': WRONG in {off_keeps}/{off_total} = {100 * off_keeps / max(1, off_total):.1f}% of cases")
    for g, (k, t) in sorted(by_guard.items(), key=lambda kv: -(kv[1][0] / max(1, kv[1][1]))):
        log(f"    {g:<18} keeps {k}/{t} = {100 * k / max(1, t):.1f}%   (the base rate is {100 * off_keeps / max(1, off_total):.1f}%)")
    log("[mine/pin-games] the conjunctions, biggest first:")
    for key, (k, t) in sorted(combos.items(), key=lambda kv: -kv[1][1])[:8]:
        name = " AND ".join(key) if key else "(no guard: a plain move off the line)"
        log(f"    {name:<58} {k}/{t} = {100 * k / max(1, t):.1f}%" + (f"   e.g. {examples[key][0][1]}" if examples[key] else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(positions=n_pos, off_total=off_total, off_keeps=int(off_keeps), depth=depth, within=within,
                   guards={g: [int(x) for x in v] for g, v in by_guard.items()},
                   combos={" AND ".join(k) or "(none)": [int(x) for x in v] for k, v in combos.items()}),
              open(OUT / "pin_games.json", "w"), indent=1)


# ---------------------------------------------------------------- the loop's missing connector: mine the MODEL'S residual
def residual(ckpt: str, split: str = "test", min_support: int = 25, top: int = 12, log=print):
    """Point the counterexample machinery at the model's own mistakes, over the registry's whole vocabulary
    instead of a hand-written claim. For every held-out position: did the concept network's chosen move keep the
    result? The features are the registry's concepts present in the position (and the chosen move's consequences).
    A candidate COMPOUND concept is a conjunction whose presence raises the chance of a mistake -- ranked by lift
    against the base rate, with support and two examples so the next round has something concrete to read.

        python -m catspace.incubator.analyst.mine residual --ckpt data/derived/checkpoints/concept_net_c4/last.pt
    """
    import torch
    from . import concept_net as CN
    from .concepts import Registry
    model, cfg, device = CN.load_model(ckpt)
    data = CN.Data(log=lambda *a, **k: None)
    reg = Registry(version=CN.REGISTRY_VERSION); names = [c.name for c in reg.concepts]
    ids = data.ids(split)
    F = len(names) + len(CN.MOVE_FEATURES)
    X = np.zeros((len(ids), F), bool); y = np.zeros(len(ids), bool); rows = []
    k = 0
    with torch.no_grad():
        for s0 in range(0, len(ids), 256):
            chunk = ids[s0:s0 + 256]
            b = data.batch(chunk, device); o = model(b)
            keep = b["keep"] & b["lm"]; has = keep.any(-1); pick = o["score"].argmax(-1)
            hit = keep.gather(1, pick[:, None]).squeeze(1)
            ic = b["ic"].cpu().numpy(); im = b["im"].cpu().numpy()
            mvf = b["mvf"].cpu().numpy() if "mvf" in b else None
            pk = pick.cpu().numpy()
            for i in range(len(chunk)):
                if not has[i]:
                    continue
                present = np.zeros(len(names), bool)
                present[np.unique(ic[i][im[i]])] = True
                X[k, :len(names)] = present
                if mvf is not None:
                    X[k, len(names):] = mvf[i, pk[i]] > 0
                y[k] = not bool(hit[i]); rows.append(int(chunk[i])); k += 1
    X, y = X[:k], y[:k]
    feat = names + [f"move:{m}" for m in CN.MOVE_FEATURES]
    base = float(y.mean())
    log(f"[mine/residual] {ckpt} on {split}: {k} positions, {int(y.sum())} wrong ({100 * base:.1f}%); {len(feat)} features")
    def lift(mask):
        n = int(mask.sum())
        if n < min_support:
            return None
        p = float(y[mask].mean())
        return (p / base if base > 0 else 0.0), p, n
    singles = []
    for j, f in enumerate(feat):
        r = lift(X[:, j])
        if r:
            singles.append((r[0], r[1], r[2], (j,), f))
    singles.sort(key=lambda t: -t[0])
    log(f"[mine/residual] single features whose presence raises the chance of a mistake (base {100 * base:.1f}%):")
    for l, p, n, _, f in singles[:top]:
        log(f"    {f:<46} wrong {100 * p:5.1f}% of {n:5d}   lift x{l:.2f}")
    # pairs, over the leading singles (the compound candidates)
    lead = [t[3][0] for t in singles[:40]]
    pairs = []
    for a in range(len(lead)):
        for b_ in range(a + 1, len(lead)):
            m = X[:, lead[a]] & X[:, lead[b_]]
            r = lift(m)
            if r and r[0] > 1.0:
                pairs.append((r[0], r[1], r[2], (lead[a], lead[b_]), f"{feat[lead[a]]} AND {feat[lead[b_]]}"))
    pairs.sort(key=lambda t: -t[0])
    log("[mine/residual] the compound candidates (conjunctions), by lift:")
    seen_fens = {}
    for l, p, n, js, f in pairs[:top]:
        m = np.ones(len(y), bool)
        for j in js:
            m &= X[:, j]
        ex = [data.rows[rows[i]]["fen"] for i in np.nonzero(m & y)[0][:2]]
        log(f"    {f:<62} wrong {100 * p:5.1f}% of {n:5d}   lift x{l:.2f}" + (f"   e.g. {ex[0]}" if ex else ""))
        seen_fens[f] = ex
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ckpt=ckpt, split=split, n=k, wrong=int(y.sum()), base=base,
                   singles=[dict(feature=f, wrong=p, n=n, lift=l) for l, p, n, _, f in singles[:40]],
                   pairs=[dict(feature=f, wrong=p, n=n, lift=l, examples=seen_fens.get(f, [])) for l, p, n, _, f in pairs[:40]]),
              open(OUT / f"residual_{split}.json", "w"), indent=1)
    return singles, pairs


# ---------------------------------------------------------------- minimal pairs: the same move, one edit, the verdict flips
CONTRAST_FEATURES = ("attackers_of_target", "defenders_of_target", "attackers_minus_defenders", "target_is_defended",
                     "value_of_target", "value_of_mover", "mover_cheaper_than_target", "exchange_see",
                     "mover_falls_after", "gives_check", "target_guarded_by_king", "our_king_near_target", "their_king_near_target")


def contrast_features(board: chess.Board, m: chess.Move) -> dict:
    """The candidate tokens for a capture, computed straight from the rules (the same quantities the graph's node
    answers carry -- attackers_count, defenders_count, exchange -- without paying for a graph build)."""
    from .defense import defense
    us = board.turn; t = m.to_square
    att = len(board.attackers(us, t)); dfd = len(board.attackers(not us, t))
    vt = piece_value(board, t) if board.piece_at(t) else 0.0
    vm = piece_value(board, m.from_square)
    nb = board.copy(stack=False); nb.push(m)
    d = defense(nb, t, memo=None)
    ek, ok_ = board.king(not us), board.king(us)
    return {"attackers_of_target": att, "defenders_of_target": dfd, "attackers_minus_defenders": att - dfd,
            "target_is_defended": float(dfd > 0), "value_of_target": vt, "value_of_mover": vm,
            "mover_cheaper_than_target": float(vm < vt), "exchange_see": (d.see if d is not None else 0.0),
            "mover_falls_after": float(bool(d is not None and d.falls)), "gives_check": float(nb.is_check()),
            "target_guarded_by_king": float(ek is not None and t in chess.SquareSet(chess.BB_KING_ATTACKS[ek])),
            "our_king_near_target": float(ok_ is not None and chess.square_distance(ok_, t) <= 2),
            "their_king_near_target": float(ek is not None and chess.square_distance(ek, t) <= 2)}


def _verdict(tb, board: chess.Board, m: chess.Move) -> int | None:
    """THE MOVE'S OWN OUTCOME in the mover's view: +1 it wins, 0 it draws, -1 it loses (exactly, by the tablebase).
    NOT 'is it the best move' (2026-09-22: that version flipped when a BETTER ALTERNATIVE appeared -- the same
    capture's own value unchanged -- and the tokens then explained the alternative, not the capture)."""
    nb = board.copy(stack=False); nb.push(m)
    if nb.is_game_over():
        return 1 if nb.is_checkmate() else 0
    try:
        v = -tb.probe_wdl(nb)
    except Exception:
        return None
    return (v > 0) - (v < 0)


def contrast(n_seeds: int, log=print, max_pairs: int = 400, seed: int = 0):
    """Kaveh 2026-09-21: 'I take a protected pawn and the eval drops; in a similar position with another attacker
    it does not. What is the difference?' Build MINIMAL PAIRS mechanically: take a capture whose verdict is known,
    perturb the position by moving ONE piece one square, and keep the perturbation only when the SAME capture's
    verdict flips. Then find the smallest set of tokens whose difference is consistent across every pair."""
    import chess.syzygy
    rows = load_rows(Path("data/derived/endgames_v2"))
    _r = __import__("random").Random(seed); _r.shuffle(rows)
    tb = chess.syzygy.open_tablebase("assets/tablebases/syzygy")
    pairs = []
    seeds = 0
    for r in rows:
        if len(pairs) >= max_pairs or seeds >= n_seeds:
            break
        b = chess.Board(r["fen"])
        caps = [m for m in b.legal_moves if b.is_capture(m) and not b.is_en_passant(m)]
        if not caps:
            continue
        seeds += 1
        for m in caps[:3]:
            v0 = _verdict(tb, b, m)
            if v0 is None:
                continue
            for sq, pc in list(b.piece_map().items()):
                if sq in (m.from_square, m.to_square):
                    continue
                for to in chess.SquareSet(chess.BB_KING_ATTACKS[sq]):
                    if b.piece_at(to) is not None:
                        continue
                    nb = b.copy(stack=False); nb.remove_piece_at(sq); nb.set_piece_at(to, pc)
                    if not nb.is_valid() or nb.is_game_over() or m not in nb.legal_moves:
                        continue
                    v1 = _verdict(tb, nb, m)
                    if v1 is None or v1 == v0:
                        continue
                    good, bad = (b, nb) if v0 > v1 else (nb, b)
                    edited = chess.square_name(sq) + chess.square_name(to)
                    pairs.append((good.fen(), bad.fen(), m.uci(), edited, pc.symbol(), max(v0, v1), min(v0, v1)))
                    break
                if len(pairs) >= max_pairs:
                    break
    tb.close()
    log(f"[mine/contrast] {seeds} seed positions scanned; {len(pairs)} minimal pairs (one piece moved one square, the same capture flips verdict)")
    if not pairs:
        return
    # the smallest set of tokens that separates the good member from the bad one
    diffs = {f: [0, 0, 0] for f in CONTRAST_FEATURES}      # feature -> [good>bad, good<bad, equal]
    rowsf = []
    for gf, bf, u, *_ in pairs:
        gb, bb = chess.Board(gf), chess.Board(bf); mv = chess.Move.from_uci(u)
        fg, fb = contrast_features(gb, mv), contrast_features(bb, mv)
        rowsf.append((fg, fb))
        for f in CONTRAST_FEATURES:
            d = fg[f] - fb[f]
            diffs[f][0 if d > 0 else 1 if d < 0 else 2] += 1
    log("[mine/contrast] single tokens, by how often their difference points the same way across the pairs:")
    ranked = []
    for f, (up, dn, eq) in diffs.items():
        n = up + dn
        acc = max(up, dn) / max(1, len(pairs))                     # a pair where the token is equal cannot be explained by it
        ranked.append((acc, f, up, dn, eq))
    ranked.sort(reverse=True)
    for acc, f, up, dn, eq in ranked:
        log(f"    {f:<28} separates {100 * acc:5.1f}% of pairs (higher in the good one {up}, lower {dn}, equal {eq})")
    # greedy minimal set: cover the pairs one token at a time
    covered = np.zeros(len(pairs), bool); chosen = []
    for _ in range(4):
        best = None
        for acc, f, up, dn, eq in ranked:
            sign = 1 if up >= dn else -1
            ok = np.array([ (fg[f] - fb[f]) * sign > 0 for fg, fb in rowsf])
            gain = int((ok & ~covered).sum())
            if best is None or gain > best[0]:
                best = (gain, f, sign, ok)
        gain, f, sign, ok = best
        if gain <= 0:
            break
        chosen.append((f, "higher" if sign > 0 else "lower")); covered |= ok
        log(f"[mine/contrast] the minimal set so far: " + " ; ".join(f"{a} {b} in the good position" for a, b in chosen) + f"  -> explains {100 * covered.mean():.1f}% of the pairs")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(pairs=len(pairs), ranked=[dict(feature=f, separates=a, up=u, down=d, equal=e) for a, f, u, d, e in ranked],
                   minimal=[list(x) for x in chosen], explained=float(covered.mean()),
                   examples=[dict(good=g, bad=b_, move=u, edit=ed, piece=pcs, good_value=int(gv), bad_value=int(bv)) for g, b_, u, ed, pcs, gv, bv in pairs[:40]]), open(OUT / "contrast.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pin", "pin-games", "residual", "contrast"])
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--ckpt", default="data/derived/checkpoints/concept_net_c4/last.pt"); ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--within", type=int, default=50); ap.add_argument("--plies", default="21,70")
    a = ap.parse_args(argv)
    if a.cmd == "contrast":
        contrast(a.seeds, max_pairs=a.pairs)
    elif a.cmd == "residual":
        residual(a.ckpt, a.split)
    elif a.cmd == "pin":
        pin(a.max_rows)
    else:
        lo, hi = (int(x) for x in a.plies.split(","))
        pin_games(a.n, a.depth, lo, hi, a.within)


if __name__ == "__main__":
    main()
