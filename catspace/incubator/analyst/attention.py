"""WHERE IS THE ACTION? (Kaveh 2026-10-09: "when I see the board, I can immediately see where the action is by looking
at who is attacking where.")

The attention map is read off the attack map alone -- no search:
  contested   a piece (either side, not the king) attacked by the enemy and undefended, attacked by something cheaper,
              or attacked more often than defended
  loose       a piece (not the king, not a pawn) nothing defends
  king        the king's zone (its square and neighbours) when the enemy attacks at least `king_k` of its squares
  changed     the squares whose attackers changed with the last move
The HOT set is a union of these, dilated by `d` king steps. A frame is IN THE ACTION when its focus or its seed's
destination lies in the hot set. The test, on the frames recorded by waiting_frames.py: of the tactics that turned
active, how many were in the action (recall), and how much of the frame work the filter skips.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import chess

from catspace.io import paths

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def _attackers(b: chess.Board):
    return {(c, s): b.attackers(c, s) for c in (chess.WHITE, chess.BLACK) for s in chess.SQUARES}


def components(b: chess.Board, prev: chess.Board | None = None, king_k: int = 2) -> dict:
    att = _attackers(b)
    contested, loose, king, changed = set(), set(), set(), set()
    for s, pc in b.piece_map().items():
        if pc.piece_type == chess.KING:
            zone = chess.SquareSet(chess.BB_KING_ATTACKS[s]) | chess.SquareSet.from_square(s)
            hit = sum(1 for z in zone if att[(not pc.color, z)])
            if hit >= king_k:
                king |= set(zone)
            continue
        a = att[(not pc.color, s)]; d = att[(pc.color, s)]
        if a and (not d or min(VAL[b.piece_type_at(x)] for x in a) < VAL[pc.piece_type] or len(a) > len(d)):
            contested.add(s)
        if not d and pc.piece_type != chess.PAWN:
            loose.add(s)
    if prev is not None:
        pa = _attackers(prev)
        changed = {s for s in chess.SQUARES if any(pa[(c, s)] != att[(c, s)] for c in (chess.WHITE, chess.BLACK))}
    return dict(contested=contested, loose=loose, king=king, changed=changed)


def dilate(sqs: set, d: int) -> set:
    if d == 0:
        return set(sqs)
    return {t for s in sqs for t in chess.SQUARES if chess.square_distance(s, t) <= d}


VARIANTS = {
    "contested": ("contested",),
    "contested+loose": ("contested", "loose"),
    "contested+loose+king": ("contested", "loose", "king"),
    "all four (+changed)": ("contested", "loose", "king", "changed"),
}


def _activations(hist, fens):
    """(t, key) for frames newly active for the side to move -- the definition of waiting_frames.report."""
    out = []
    for t in range(len(hist)):
        stm = "w" if chess.Board(fens[t]).turn else "b"
        for key, (st, *_) in hist[t].items():
            if st == "active" and key[0] == stm:
                prev2 = hist[t - 2].get(key) if t > 1 else None
                if prev2 is not None and prev2[0] == "active":
                    continue
                out.append((t, key))
    return out


def measure(path: str) -> None:
    runs = [json.loads(l) for l in open(path)]
    runs = [r for r in runs if r["exact"]] or runs
    rows = []                                                          # (variant, d, kind, in_action)
    tally = Counter(); totals = Counter(); scores = []
    for r in runs:
        hist, fens = r["hist"], r["fens"]
        acts = set(_activations(hist, fens))
        comps = [components(chess.Board(fens[t]), chess.Board(fens[t - 1]) if t else None) for t in range(len(fens))]
        for t in range(len(fens)):
            for key, (st, alts, born, name, val) in hist[t].items():
                owner, uci, focus = key.split("|")
                sqs = {chess.parse_square(x) for x in focus.split(",") if x} | {chess.Move.from_uci(uci).to_square}
                kind = "activation" if (t, key) in acts else st
                totals[kind] += 1
                # THE GRADED SCORE: every tension signal near the frame's squares, nearer counting more
                sc = sum(w for part in ("contested", "loose", "king", "changed") for d, w in ((0, 1.0), (1, 0.5), (2, 0.25))
                         if sqs & dilate(comps[t][part], d))
                scores.append((sc, kind == "activation", st == "far"))
                for vname, parts in VARIANTS.items():
                    base = set().union(*(comps[t][p] for p in parts))
                    for d in (0, 1, 2):
                        if sqs & dilate(base, d):
                            tally[(vname, d, kind)] += 1
    print(f"OBS {len(runs)} games: {totals['activation']} tactics turned active; frames evaluated: far {totals['far']}, "
          f"dormant {totals['dormant']}, active {totals['active'] + totals['activation']}")
    print(f"OBS {'hot set':24s} {'dilate':6s} {'recall':>8s} {'far kept':>9s} {'dormant kept':>13s} {'all kept':>9s}")
    for vname in VARIANTS:
        for d in (0, 1, 2):
            rec = tally[(vname, d, 'activation')] / max(totals['activation'], 1)
            far = tally[(vname, d, 'far')] / max(totals['far'], 1)
            dor = tally[(vname, d, 'dormant')] / max(totals['dormant'], 1)
            allk = sum(tally[(vname, d, k)] for k in totals) / max(sum(totals.values()), 1)
            print(f"OBS {vname:24s} {d:6d} {100 * rec:7.0f}% {100 * far:8.0f}% {100 * dor:12.0f}% {100 * allk:8.0f}%")
    print(f"OBS graded score: AUC tactics vs every other frame {_auc([(s, y) for s, y, _ in scores]):.3f}; "
          f"tactics vs far frames {_auc([(s, y) for s, y, f in scores if y or f]):.3f} (0.5 = no better than chance)")
    srt = sorted(scores, key=lambda x: -x[0]); n = len(srt); pos = sum(y for _, y, _ in srt)
    for frac in (0.25, 0.5, 0.75):
        top = srt[:int(frac * n)]
        print(f"OBS   opening only the top {int(100 * frac)}% of frames by score keeps {100 * sum(y for _, y, _ in top) / max(pos, 1):.0f}% of the tactics")


def _auc(pairs) -> float:
    """P(score of a random positive > a random negative), ties half."""
    pos = sorted(s for s, y in pairs if y); neg = sorted(s for s, y in pairs if not y)
    if not pos or not neg:
        return float("nan")
    import bisect
    win = sum(bisect.bisect_left(neg, s) + 0.5 * (bisect.bisect_right(neg, s) - bisect.bisect_left(neg, s)) for s in pos)
    return win / (len(pos) * len(neg))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="waiting_frames_c300_60.jsonl")
    a = ap.parse_args(argv)
    measure(str(paths.experiment(a.file)))


if __name__ == "__main__":
    main()
