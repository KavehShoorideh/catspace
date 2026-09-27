"""THE VERDICT GAUGE (Kaveh 2026-09-26, "do it"): does a lemma library's verdict on a move predict what Stockfish says
the move does, better than a one-line rule? On held-out Lichess positions (the re-test split: no library here was
fitted on them) every legal move of the side to move gets a TRUTH from Stockfish -- the move against standing still
(the lemmas' own relative outcome; primary) and against the best move (what a player needs; secondary), 'loses' when
it is worse by LOSS_CP or more -- and four predictions:

  majority    always 'keeps' (the degenerate baseline: balanced accuracy 0.5)
  undefended  the moved piece lands attacked and undefended
  hangs       it lands attacked, and undefended or attacked by a cheaper piece
  library     the most specific lemmas that speak on the move (the largest frames; their majority; a tie keeps);
              'keeps' when none speaks

Reported: balanced accuracy and the precision and recall of 'loses', on all moves and on the moves the library speaks
about (the rules scored on the same subset), and how often the library says what 'hangs' says.

    python -m catspace.incubator.analyst.gauge_verdicts --n 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import chess

from catspace.io import paths

from .chess_domain import VAL
from .engine import Engine

LOSS_CP = 150


def positions(n: int, seed: int = 0) -> list:
    """Held-out positions: the Lichess re-test split (the generation's hash split), shuffled, without positions that
    are over or in check (standing still is illegal there, so the relative outcome is undefined)."""
    fens = []
    for line in open(paths.derived("chessbench_positions.jsonl")):
        fen = json.loads(line)["fen"]
        if int(hashlib.sha1(fen.encode()).hexdigest()[:8], 16) % 5 == 0:
            fens.append(fen)
    random.Random(seed).shuffle(fens)
    out = []
    for f in fens:
        b = chess.Board(f)
        if not b.is_game_over() and not b.is_check():
            out.append(b)
        if len(out) >= n:
            break
    return out


def truths(eng: Engine, b: chess.Board, depth: int) -> dict | None:
    """uci -> (cp against standing still, cp against the best move), from the mover's side."""
    lines = eng.analyse(b, multipv=b.legal_moves.count(), depth=depth)
    probe = eng.null_move_probe(b, multipv=1, depth=depth)
    if not lines or not probe:
        return None
    cp = {l.move: l.cp for l in lines}
    stand = -probe[0].cp                                                  # the opponent's best after our pass, from our side
    best = max(cp.values())
    return {m: (c - stand, c - best) for m, c in cp.items()}


def rules(b: chess.Board, m: chess.Move) -> tuple:
    """(undefended, hangs) for the moved piece on its destination, after the move."""
    us = b.turn
    nb = b.copy(stack=False); nb.push(m)
    p = nb.piece_at(m.to_square)
    att = nb.attackers(not us, m.to_square)
    if p is None or not att:
        return False, False
    undefended = not nb.attackers(us, m.to_square)
    cheaper = any(VAL[nb.piece_at(a).piece_type] < VAL[p.piece_type] for a in att)
    return undefended, undefended or cheaper


def library_verdicts(app, b: chess.Board) -> dict:
    """uci -> (says 'loses', number of lemmas at the top size, tie) for the moves the library speaks about."""
    res = app.apply(b)
    out = {}
    for (e, key), vs in res["verdicts"].items():
        top = max(c.frame.size for c, _, _, _ in vs)
        preds = [p for c, p, _, _ in vs if c.frame.size == top]
        n_lose = sum(1 for p in preds if p == "loses")
        out[key] = (n_lose * 2 > len(preds), len(preds), n_lose * 2 == len(preds))
    return out


def scores(pairs: list) -> dict:
    """pairs of (truth loses, predicted loses) -> balanced accuracy, precision and recall of 'loses', n, base rate."""
    tp = sum(1 for t, p in pairs if t and p); fn = sum(1 for t, p in pairs if t and not p)
    fp = sum(1 for t, p in pairs if not t and p); tn = sum(1 for t, p in pairs if not t and not p)
    rec = tp / (tp + fn) if tp + fn else float("nan"); spec = tn / (tn + fp) if tn + fp else float("nan")
    return dict(n=len(pairs), base=(tp + fn) / max(1, len(pairs)), bacc=(rec + spec) / 2,
                prec=tp / (tp + fp) if tp + fp else float("nan"), rec=rec)


def bootstrap_diff(rows: list, truth_col: int, pred_a, pred_b, n_boot: int = 1000, seed: int = 0) -> tuple:
    """95% interval of balanced accuracy A minus B, resampling POSITIONS (a position's moves move together)."""
    import numpy as np
    pos = np.array([r[0] for r in rows]); t = np.array([r[truth_col] <= -LOSS_CP for r in rows])
    a = np.array([pred_a(r) for r in rows]); b = np.array([pred_b(r) for r in rows])
    uniq, inv = np.unique(pos, return_inverse=True)
    rng = np.random.default_rng(seed); diffs = []
    def bacc(w, p):
        tp = (w * (t & p)).sum(); fn = (w * (t & ~p)).sum(); fp = (w * (~t & p)).sum(); tn = (w * (~t & ~p)).sum()
        return 0.5 * (tp / max(tp + fn, 1e-9) + tn / max(tn + fp, 1e-9))
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float)
        diffs.append(bacc(w, a) - bacc(w, b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def library_spec(spec: str):
    """name=library.json[:proved.json]; a checkpoint file is read for its library (a run stopped mid-size)."""
    name, rest = spec.split("=", 1)
    lib, _, prov = rest.partition(":")
    lp = Path(lib)
    if lp.name.startswith("checkpoint_"):
        out = lp.with_name(lp.name.replace("checkpoint_", "library_").replace(".json", "_ckpt.json"))
        json.dump(json.load(open(lp))["library"], open(out, "w"))
        lp = out
    return name, lp, (Path(prov) if prov else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--library", action="append", default=None, help="name=library.json[:proved.json] (repeatable; a checkpoint_*.json gives its library)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    L = paths.derived("lemmas")
    specs = a.library or [f"refined905={L}/library_games_frame6k.json:{L}/proved_games_frame6k.json",
                          f"full3_partial={L}/checkpoint_chess_games_frame_full3.json",
                          f"cheap_partial={L}/checkpoint_chess_games_frame_contests_counter2.json"]
    from .chains import domain_of
    from .lemma_apply import Applier, load_library
    t0 = time.perf_counter()
    boards = positions(a.n, a.seed)
    eng = Engine(threads=a.threads, depth=a.depth)
    rows = []                                                             # (board index, uci, rel cp, best cp, undefended, hangs)
    for i, b in enumerate(boards):
        t = truths(eng, b, a.depth)
        if t is None:
            continue
        for m in b.legal_moves:
            if m.uci() in t:
                rows.append((i, m.uci(), *t[m.uci()], *rules(b, m)))
    eng.close()
    print(f"[verdicts] {len(boards)} held-out positions, {len(rows)} moves with a Stockfish truth (depth {a.depth}; {time.perf_counter() - t0:.0f}s)")
    report = dict(n_positions=len(boards), n_moves=len(rows), depth=a.depth, loss_cp=LOSS_CP, libraries={})
    for truth_name, col in (("vs standing still", 2), ("vs the best move", 3)):
        base = [(r[col] <= -LOSS_CP) for r in rows]
        print(f"OBS truth {truth_name}: 'loses' in {sum(base)}/{len(rows)} moves ({100 * sum(base) / max(1, len(rows)):.1f}%)")
        for pname, f in (("majority", lambda r: False), ("undefended", lambda r: r[4]), ("hangs", lambda r: r[5])):
            s = scores([(r[col] <= -LOSS_CP, f(r)) for r in rows])
            print(f"OBS   {pname:10s} balanced acc {s['bacc']:.3f}  precision {s['prec']:.3f}  recall {s['rec']:.3f}")
            report.setdefault("rules", {})[f"{truth_name}|{pname}"] = s
    for spec in specs:
        name, lp, prov = library_spec(spec)
        t1 = time.perf_counter()
        app = Applier(domain_of(lp), load_library(lp, prov))
        says = {}
        for i, b in enumerate(boards):
            for key, v in library_verdicts(app, b).items():
                says[(i, key)] = v
        spoken = [r for r in rows if (r[0], r[1]) in says]
        ties = sum(1 for r in spoken if says[(r[0], r[1])][2])
        agree = sum(1 for r in spoken if says[(r[0], r[1])][0] == r[5])
        print(f"OBS library {name}: {len(app.lemmas)} lemmas; speaks on {len(spoken)}/{len(rows)} moves ({100 * len(spoken) / max(1, len(rows)):.1f}%), "
              f"top-size ties {ties}; says what 'hangs' says on {agree}/{len(spoken)} ({time.perf_counter() - t1:.0f}s)")
        rep = dict(lemmas=len(app.lemmas), spoken=len(spoken), ties=ties, agree_hangs=agree)
        for truth_name, col in (("vs standing still", 2), ("vs the best move", 3)):
            s_all = scores([(r[col] <= -LOSS_CP, says.get((r[0], r[1]), (False,))[0]) for r in rows])
            s_lib = scores([(r[col] <= -LOSS_CP, says[(r[0], r[1])][0]) for r in spoken])
            s_hang = scores([(r[col] <= -LOSS_CP, r[5]) for r in spoken])
            s_und = scores([(r[col] <= -LOSS_CP, r[4]) for r in spoken])
            print(f"OBS   {truth_name}: all moves (silent = keeps) balanced acc {s_all['bacc']:.3f} precision {s_all['prec']:.3f} recall {s_all['rec']:.3f}")
            print(f"OBS   {truth_name}: where it speaks ('loses' {100 * s_lib['base']:.1f}%): library {s_lib['bacc']:.3f} (p {s_lib['prec']:.3f} r {s_lib['rec']:.3f}) | "
                  f"hangs {s_hang['bacc']:.3f} (p {s_hang['prec']:.3f} r {s_hang['rec']:.3f}) | undefended {s_und['bacc']:.3f}")
            lo, hi = bootstrap_diff(rows, col, lambda r: says.get((r[0], r[1]), (False,))[0], lambda r: r[5])
            print(f"OBS   {truth_name}: library minus 'hangs', balanced accuracy on all moves: {s_all['bacc'] - scores([(r[col] <= -LOSS_CP, r[5]) for r in rows])['bacc']:+.3f} "
                  f"(95% interval over positions {lo:+.3f} .. {hi:+.3f})")
            rep[truth_name] = dict(all=s_all, spoken=s_lib, spoken_hangs=s_hang, spoken_undefended=s_und, minus_hangs_ci=[lo, hi])
        report["libraries"][name] = rep
    out = Path(paths.derived("lemmas", f"gauge_verdicts{('_' + a.tag) if a.tag else ''}.json"))
    json.dump(report, open(out, "w"), indent=1)
    print(f"[verdicts] -> {out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
