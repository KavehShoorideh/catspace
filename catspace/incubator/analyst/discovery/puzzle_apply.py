"""PUZZLES ARE APPLY-ONLY (Kaveh 2026-10-16): the window is given by the puzzle, so nothing is mined, drilled or sent to Lean here. The machinery
applies what it has learnt, in Kaveh's order, and prints the EVENT STREAM as it goes:

  EV 1 position           the puzzle, our side, the target (the puzzle's value is the window's delta; 100 = mate)
  EV 2 threats to us      what they would take if we passed, each attack with its dead-end verdict (the proved schema `dead_end`, bound)
  EV 3 root               every capture of ours read in the root's words (worth taken, our pieces hit after, k, whether `hanging` applies)
  EV 4 bind               the proved schemas bound at this position, in the ladder's order (root, delayed_root, delayed_root_max, delayed_balance; depth 1, 2)
  EV 5 decision           the first binding in that order, or none
  EV 6 check              the exact solver on the decision, and the puzzle's own answer

Run: .venv/bin/python -m catspace.incubator.analyst.discovery.puzzle_apply --set dev_20 --index 0
"""
from __future__ import annotations

import argparse
import json
import time

import chess

from catspace.io import paths
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import selfmine as SM

ORDER = ("delayed_root", "delayed_root_max", "delayed_balance")


def load_set(name: str) -> list:
    d = json.load(open(paths.derived(f"ledger_puzzles_{name}.json")))
    return d["results"] if isinstance(d, dict) else d


def passed(fen: str) -> str:
    """The position if we passed (a null move): their turn, same board. Only for READING threats; never used as a premise."""
    b = chess.Board(fen)
    b.push(chess.Move.null())
    return b.fen()


def depth2_candidates(fen: str) -> list:
    """Pairs of our moves for the depth-2 schemas: m1, then (after the null reply) m2 -- a cheap candidate set; the schema's own tail quantifies over every reply."""
    b = chess.Board(fen)
    out = []
    for m1 in b.legal_moves:
        if b.piece_at(m1.from_square).piece_type == chess.KING or m1.promotion:
            continue
        c = b.copy()
        c.push(m1)
        if c.is_game_over():
            continue
        c.push(chess.Move.null())
        if c.is_valid() is False:
            continue
        for m2 in c.legal_moves:
            if c.piece_at(m2.from_square).piece_type == chess.KING or m2.promotion:
                continue
            out.append([(chess.square_name(m1.from_square), chess.square_name(m1.to_square)),
                        (chess.square_name(m2.from_square), chess.square_name(m2.to_square))])
    return out


def apply(p: dict, log=print, deep: bool = True) -> dict:
    BV.install_terms()
    fen, need = p["fen"], int(p["value"])
    b = chess.Board(fen)
    us = "w" if b.turn else "b"
    t0 = time.time()
    log(f"EV 1 position {p['id']}: {fen}  {'white' if us == 'w' else 'black'} to move; the puzzle's window: gain {need}" + (" (mate)" if need >= 100 else "")
        + f"; themes {p.get('themes')}")
    # 2. threats to us, biggest first
    if b.is_check():
        checkers = [chess.square_name(s) for s in b.checkers()]
        log(f"EV 2 threats to us: we are IN CHECK from {', '.join(checkers)}: the biggest threat is the king itself; passing is not a position, so the threat "
            f"reading is the check and every move must answer it ({b.legal_moves.count()} legal moves)")
        v = dict(items=[])
    else:
        v = SM.verdicts(passed(fen), us, log=lambda *a: None)
    atk = [it for it in v["items"] if it["kind"] != "quiet"]
    atk.sort(key=lambda it: -SM.PIECE_WORTH[b.piece_at(chess.parse_square(it["t"])).symbol().lower()])
    if not atk:
        log("EV 2 threats to us: none (if we passed, no capture of theirs)")
    for it in atk:
        w = SM.PIECE_WORTH[b.piece_at(chess.parse_square(it["t"])).symbol().lower()]
        log(f"EV 2 threat {it['san']} takes our {b.piece_at(chess.parse_square(it['t'])).symbol()} on {it['t']} (worth {w}): {it['kind']}"
            + (f" (we take back with {it['z']})" if it["kind"] == "dead end" else (f" {it.get('k', '')}" if it["kind"] == "costs" else "")))
    biggest = next((it for it in atk if it["kind"] == "costs"), None)
    if biggest:
        log(f"EV 2 biggest threat to us: {biggest['san']} costs {biggest.get('k')}; a move must answer it or outweigh it")
    # 3. the root
    reads = SM.root_readings(fen, us, need)
    reads.sort(key=lambda r: -r["k"])
    if not reads:
        log("EV 3 root: no capture available")
    for r in reads:
        log(f"EV 3 root {r['san']}: takes {r['worth']}, ours hit after {r['attacked_after']} (most worth {r['w']}), k = {r['k']}; "
            + ("`hanging` applies" if r["applies"] else "the theorem does not apply"))
    # 4. bind: each proved schema read ONCE per candidate (its tail is a number; the target only sets the floor), the largest guaranteed gain wins,
    #    ties in the ladder's order; no floor on the balance: the tail reads our material too (the game is alive when it is 4 or more), so a gain from
    #    a deficit or a level position is an instance as well
    found = []
    c2 = depth2_candidates(fen) if deep else []
    for r in reads:
        if r["applies"]:
            found.append(("hanging", r["san"], f"{r['x']}->{r['y']} takes {r['worth']}, nothing of ours is hit after", r["worth"]))
            log(f"EV 4 bind hanging: {r['san']} guarantees {r['worth']} ({found[-1][2]})")
    for fam in ORDER:
        for depth in (1, 2):
            if depth == 2 and not deep:
                continue
            t1 = time.time()
            best = best_bind(depth, fen, us, c2 if depth == 2 else None, fam)
            name = f"{fam}_d{depth}"
            if best is None:
                log(f"EV 4 bind {name}: no candidate's legs hold [{round(time.time() - t1, 1)} s]")
                continue
            san, B, T, n = best
            if T > B:
                found.append((name, san, f"B={B} tail={T}", T - B))
                log(f"EV 4 bind {name}: best of {n} candidates {san}: the tail reads {T} against {B} now, guaranteed gain {T - B} [{round(time.time() - t1, 1)} s]")
            else:
                log(f"EV 4 bind {name}: best of {n} candidates {san} reads {T} against {B} now: no gain [{round(time.time() - t1, 1)} s]")
    found.sort(key=lambda f: -f[3])
    # 5. decision
    if found:
        name, san, why, k = found[0]
        log(f"EV 5 decision: {san.split()[0]} by {name}, guaranteed gain {k} ({why})" + (f"; the puzzle's window asks {need}: {'reached' if k >= need else 'NOT reached'}"))
    else:
        log("EV 5 decision: none -- no proved lemma binds here at any gain" + (" (mate: no lemma reads mate yet)" if need >= 100 else ""))
    # 6. check
    ans = b.san(chess.Move.from_uci(p["answer"]))
    lab = SM._label((fen, need, 1)) if need < 100 else dict(status="skipped (mate)")
    ours = found[0][1].split()[0] if found else None
    log(f"EV 6 check: the puzzle's answer is {ans}; our decision {ours or 'none'}: {'MATCH' if ours == ans else 'MISS'}; "
        f"the solver on a gain >= {need}: {lab['status']}" + (f", line {' '.join(lab['line'])}" if lab.get("status") == "forced" else "") + f"  [{round(time.time() - t0, 1)} s]")
    return dict(id=p["id"], decision=ours, answer=ans, match=ours == ans, by=found[0][0] if found else None)


def best_bind(depth: int, fen: str, mover: str, candidates: list | None, family: str):
    """Every candidate whose legs hold, its tail read once (the leaf number with a large constant C, so the min is the plain leaf); the schema then needs the
    tail with C = T - 4 (13 for the weakened family) to read T as well: our material alive at the leaf. A candidate whose material is short is printed and
    skipped. -> (san, B, best tail, candidates tried) or None."""
    B = SM.val_at(fen, "balance", mover)
    if candidates is None:
        b = chess.Board(fen)
        candidates = []
        for m in b.legal_moves:
            if b.piece_at(m.from_square).piece_type == chess.KING or m.promotion:
                continue
            candidates.append([(chess.square_name(m.from_square), chess.square_name(m.to_square))])
    best, n = None, 0
    for pairs in candidates:
        if len(pairs) != depth:
            continue
        with SM.using_mover(mover):
            tail = SM.delayed_tail(pairs, mover, family, "1000")
            prem = [SM.atom_of("balance", "<=", B), SM.atom_of(tail, ">=", 0)]
        if not SM.premise_truth(prem, [dict(fen=fen, mover=mover)])[0]:
            continue
        T = SM.val_at(fen, tail, mover)
        n += 1
        if T > B:                              # a gain: the schema's tail (C = T - 4, 13 for the weakened family) must read T as well -- our material alive at every leaf
            need = SM.ALIVE_NEED[family]
            if T < need:
                print(f"OBS bind {family}_d{depth}: {' '.join(_san_pairs(fen, pairs))} reads {T}, below the alive material {need}: the schema cannot apply")
                continue
            with SM.using_mover(mover):
                alive = [SM.atom_of("balance", "<=", B), SM.atom_of(SM.delayed_tail(pairs, mover, family, SM.alive_const(T, family)), ">=", T)]
            if not SM.premise_truth(alive, [dict(fen=fen, mover=mover)])[0]:
                print(f"OBS bind {family}_d{depth}: {' '.join(_san_pairs(fen, pairs))} reads {T} but our material is below {need} at a leaf: the schema cannot apply")
                continue
        if best is None or T > best[2]:
            best = (" ".join(_san_pairs(fen, pairs)), B, T, 0)
    return None if best is None else (best[0], best[1], best[2], n)


def _san_pairs(fen: str, pairs: list) -> list:
    b = chess.Board(fen)
    out = []
    for i, (x, y) in enumerate(pairs):
        m = chess.Move.from_uci(x + y)
        out.append(b.san(m))
        b.push(m)
        if i + 1 < len(pairs):
            b.push(chess.Move.null())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev_20")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--id")
    ap.add_argument("--shallow", action="store_true", help="depth-1 schemas only")
    a = ap.parse_args()
    ps = load_set(a.set)
    p = next(x for x in ps if x["id"] == a.id) if a.id else ps[a.index]
    apply(p, deep=not a.shallow)


if __name__ == "__main__":
    main()
