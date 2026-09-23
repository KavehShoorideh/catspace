"""BODY-ORDER DECOMPOSITION OF A MOVE'S OUTCOME, imported from shapiq (Muschalik et al., NeurIPS 2024) rather than
hand-rolled (Kaveh 2026-09-23: "use their frameworks as written").

The cooperative game: players = the non-king pieces on the board; v(S) = the outcome of the MOVE when only the pieces
in S (plus both kings, plus the mover) are on the board -- exactly the sampled-frame semantics, now evaluated on a
real position's own pieces. shapiq's exact Möbius transform then says how much of the outcome is carried by each
subset: the empty set (the mover alone), singletons, pairs, triples ... This is the ground truth for "which bodies
does this move's verdict depend on", and the label the neural guide (AlphaGeometry's role: which body to bring
in next) is trained on.

    python -m catspace.incubator.analyst.shap_bodies --n 100 --max-players 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess
import numpy as np

OUT = Path("data/derived/lemmas")


def outcome_on_subset(dom, board: chess.Board, move: chess.Move, keep: set, relative: bool) -> float:
    """The move's outcome (material after the opponent's best capture/check reply, from the mover's side) on the
    board restricted to `keep` (squares) plus both kings and the mover; None if the restricted position is illegal."""
    b = chess.Board(None); b.turn = board.turn
    b.castling_rights = 0; b.ep_square = None
    for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING or sq == move.from_square or sq in keep:
            b.set_piece_at(sq, p)
    if not b.is_valid() or move not in b.legal_moves:
        return None
    from .chess_domain import State
    st = State(b, [move.from_square])
    before = dom.scalar(st, move.from_square)
    after = dom.transition(st, (b, move))
    d = dom.reply_scalar(after, after.ids[0]) - before
    if relative:
        base = dom.null_reply_scalar(st, move.from_square)
        if base is None:
            return None
        d -= base - before
    return d


def decompose(dom, board: chess.Board, move: chess.Move, max_players: int, relative: bool, rng: random.Random):
    from shapiq import ExactComputer
    pieces = [sq for sq, p in board.piece_map().items() if p.piece_type != chess.KING and sq != move.from_square]
    if len(pieces) > max_players:
        # keep the pieces nearest the move (start and landing squares): the far ones are the least likely to matter
        pieces.sort(key=lambda s: min(chess.square_distance(s, move.from_square), chess.square_distance(s, move.to_square)))
        pieces = pieces[:max_players]
    n = len(pieces)
    if n == 0:
        return None
    cache = {}
    def game(coalitions):
        out = []
        for row in coalitions:
            keep = frozenset(pieces[i] for i in range(n) if row[i])
            if keep not in cache:
                v = outcome_on_subset(dom, board, move, set(keep), relative)
                cache[keep] = 0.0 if v is None else v
            out.append(cache[keep])
        return np.array(out, dtype=float)
    ec = ExactComputer(game, n_players=n)
    mob = ec.moebius_transform            # a cached property in shapiq 1.7
    return dict(pieces=pieces, n=n, moebius=mob, game_calls=len(cache), full=cache.get(frozenset(pieces)))


def run(n: int, lo: int, hi: int, max_players: int, relative: bool, seed: int = 0, log=print, all_records: bool = False, with_random: bool = False, tag: str = ""):
    from .chess_domain import domain
    from .dataset import load as load_ds
    dom = domain()
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    rows = [m for m in meta if lo <= m["ply"] <= hi and m.get("best")]
    rng = random.Random(seed); rng.shuffle(rows)
    t0 = time.perf_counter()
    mass_by_order = Counter(); n_pos = 0; relevant_sizes = Counter(); top_orders = Counter(); records = []
    zero_moves = 0; max_order_seen = Counter()
    for m_ in rows:
        if n_pos >= n:
            break
        b = chess.Board(m_["fen"])
        if b.is_game_over():
            continue
        try:
            mv = chess.Move.from_uci(m_["best"])
        except Exception:
            continue
        if mv not in b.legal_moves:
            continue
        moves = [("best", mv)] + ([("random", rng.choice(list(b.legal_moves)))] if with_random else [])
        n_pos += 1
        for kind, mv in moves:
          r = decompose(dom, b, mv, max_players, relative, rng)
          if r is None:
            continue
          mob = r["moebius"]
          # interactions by order (|coefficient| mass), excluding the empty set (the mover alone)
          by_order = defaultdict(float); coeffs = []
          for subset, v in zip(mob.interaction_lookup.keys(), mob.values):
              k = len(subset)
              if abs(v) < 1e-9:
                  continue
              by_order[k] += abs(v); coeffs.append((subset, float(v)))
          total = sum(by_order.values())
          if total < 1e-9:
              zero_moves += 1                                                   # the outcome does not depend on any other piece
              relevant_sizes[0] += 1
              continue
          for k, v in by_order.items():
              mass_by_order[k] += v / total
          coeffs.sort(key=lambda sv: -abs(sv[1]))
          relevant = set()
          acc = 0.0
          for subset, v in coeffs:
              relevant |= set(subset); acc += abs(v)
              if acc >= 0.9 * total:
                  break
          relevant_sizes[len(relevant)] += 1
          top_orders[len(coeffs[0][0])] += 1
          max_order_seen[max(len(s) for s, _ in coeffs)] += 1
          if all_records or len(records) < 300:
              share = defaultdict(float)                                        # per piece: |mass| of the subsets it belongs to, over the total
              for subset, v in coeffs:
                  for i in subset:
                      share[i] += abs(v) / total
              records.append(dict(fen=b.fen(), move=mv.uci(), kind=kind, n=r["n"], full=r["full"], relevant=[chess.square_name(r["pieces"][i]) for i in sorted(relevant)],
                                  share={chess.square_name(r["pieces"][i]): round(x, 3) for i, x in share.items()},
                                  top=[([chess.square_name(r["pieces"][i]) for i in s], round(v, 2)) for s, v in coeffs[:6]]))
        if n_pos % 25 == 0:
            log(f"[shap_bodies] {n_pos}/{n} ({time.perf_counter() - t0:.0f}s)")
    N = max(1, n_pos)
    log(f"[shap_bodies] {n_pos} positions (best move), players <= {max_players}, outcome {'RELATIVE' if relative else 'ABSOLUTE'}, {time.perf_counter() - t0:.0f}s")
    log(f"  moves whose outcome depends on NO other piece: {zero_moves}/{n_pos} = {100 * zero_moves / N:.1f}%")
    dep = max(1, n_pos - zero_moves)
    log("  share of |Möbius| mass by interaction order (mean over dependent moves): " + ", ".join(f"order {k}: {100 * v / dep:.1f}%" for k, v in sorted(mass_by_order.items())))
    log("  highest order present per move: " + ", ".join(f"{k}: {v}" for k, v in sorted(max_order_seen.items())))
    log("  order of the single largest term: " + ", ".join(f"{k}: {v}" for k, v in sorted(top_orders.items())))
    log("  pieces needed for 90% of the mass (the relevant set): " + ", ".join(f"{k}: {v} ({100 * v / N:.0f}%)" for k, v in sorted(relevant_sizes.items())))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(positions=n_pos, relative=relative, zero=zero_moves, mass_by_order={k: v / dep for k, v in mass_by_order.items()}, relevant_sizes=dict(relevant_sizes), records=records),
              open(OUT / f"shap_bodies{'_relative' if relative else ''}{tag}.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--lo", type=int, default=10); ap.add_argument("--hi", type=int, default=80)
    ap.add_argument("--max-players", type=int, default=10); ap.add_argument("--relative", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-records", action="store_true"); ap.add_argument("--with-random", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    run(a.n, a.lo, a.hi, a.max_players, a.relative, a.seed, all_records=a.all_records, with_random=a.with_random, tag=a.tag)


if __name__ == "__main__":
    main()
