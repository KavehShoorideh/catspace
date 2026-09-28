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


def decide(vs: list, max_size: int = 99):
    """The most specific lemmas (the largest frames up to max_size) decide by majority; a tie keeps. -> (says 'loses',
    how many decided, tie, top size, the deciders [(lemma, prediction, fired term)]) or None when none speaks."""
    vs = [v for v in vs if v[0].frame.size <= max_size]
    if not vs:
        return None
    top = max(c.frame.size for c, _, _, _ in vs)
    dec = [(c, p, f) for c, p, f, _ in vs if c.frame.size == top]
    n_lose = sum(1 for _, p, _ in dec if p == "loses")
    return n_lose * 2 > len(dec), len(dec), n_lose * 2 == len(dec), top, dec


def size_of(c, pred: str, fired) -> float | None:
    """A verdict's SIZE (Kaveh 2026-09-27: "give claims a size"): the mean outcome of the rows where the fired guard
    term fires, else the lemma's mean over its exceptions (a term fired) or over the rows where it held."""
    v = c.value or {}
    if fired is not None and fired in (v.get("terms") or {}):
        return v["terms"][fired]
    if not v:
        return None
    return v.get("exception") if fired is not None else v.get("holds")


def library_verdicts(app, b: chess.Board, raw: bool = False) -> dict:
    """uci -> (says 'loses', number of lemmas at the top size, tie, the deciders' mean size or None) for the moves the
    library speaks about; with raw, uci -> every verdict [(lemma, prediction, fired term, ids)]."""
    res = app.apply(b)
    if raw:
        return {key: vs for (e, key), vs in res["verdicts"].items()}
    out = {}
    for (e, key), vs in res["verdicts"].items():
        d = decide(vs)
        sizes = [x for x in (size_of(c, p, f) for c, p, f in d[4]) if x is not None]
        out[key] = (*d[:3], sum(sizes) / len(sizes) if sizes else None)
    return out


_APP = None


def _verdicts_job(args):
    i, fen = args
    return i, library_verdicts(_APP, chess.Board(fen))


def _raw_job(args):
    i, fen = args
    return i, library_verdicts(_APP, chess.Board(fen), raw=True)


def raw_all(app, boards: list, workers: int) -> dict:
    """position index -> every verdict [(lemma, prediction, fired, ids)] per move, over a fork pool."""
    global _APP
    _APP = app
    if workers <= 1:
        return {i: library_verdicts(app, b, raw=True) for i, b in enumerate(boards)}
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        return dict(pool.imap_unordered(_raw_job, [(i, b.fen()) for i, b in enumerate(boards)], chunksize=2))


def verdicts_all(app, boards: list, workers: int) -> dict:
    """position index -> the library's verdicts, over a fork pool (the applier is inherited, not pickled)."""
    global _APP
    _APP = app
    if workers <= 1:
        return {i: library_verdicts(app, b) for i, b in enumerate(boards)}
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        return dict(pool.imap_unordered(_verdicts_job, [(i, b.fen()) for i, b in enumerate(boards)], chunksize=4))


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


def diagnose(boards, rows, name, app, n_examples: int = 3, log=print, workers: int = 1) -> dict:
    """Where the library loses to 'hangs' (truth: against standing still): accuracy by the largest frame allowed to
    decide, the disagreements with 'hangs' and who is right, the lemmas that decide the moves it gets wrong where
    'hangs' is right, and worked examples of the most frequent ones."""
    col = 2
    raw = {(i, key): vs for i, per in raw_all(app, boards, workers).items() for key, vs in per.items()}
    t = lambda r: r[col] <= -LOSS_CP
    out = dict(by_cap={})
    for k in (1, 2, 3, 4):
        pairs = []
        for r in rows:
            d = decide(raw.get((r[0], r[1]), []), k)
            pairs.append((t(r), bool(d and d[0])))
        s = scores(pairs); out["by_cap"][k] = s
        log(f"OBS {name}: frames up to size {k} decide -> balanced acc {s['bacc']:.3f} (p {s['prec']:.3f} r {s['rec']:.3f})")
    top_n = {}; wrong_where_hangs_right = []
    cells = {("L", "K", True): 0, ("L", "K", False): 0, ("K", "L", True): 0, ("K", "L", False): 0}
    by_top = {}
    for r in rows:
        d = decide(raw.get((r[0], r[1]), []))
        lib = bool(d and d[0]); top = d[3] if d else 0
        by_top.setdefault(top, []).append((t(r), lib, r[5]))
        if lib != r[5]:
            cells[("L" if lib else "K", "L" if r[5] else "K", t(r))] += 1
            if lib != t(r):
                wrong_where_hangs_right.append((r, d))
    for top, trip in sorted(by_top.items()):
        sl = scores([(a, b) for a, b, _ in trip]); sh = scores([(a, c) for a, _, c in trip])
        log(f"OBS {name}: moves decided at size {top}: {len(trip)} -- library {sl['bacc']:.3f} vs hangs {sh['bacc']:.3f} on the same moves")
    log(f"OBS {name}: disagreements with 'hangs' -- library says loses, hangs keeps: Stockfish loses {cells[('L', 'K', True)]} / keeps {cells[('L', 'K', False)]}; "
        f"library keeps, hangs loses: Stockfish loses {cells[('K', 'L', True)]} / keeps {cells[('K', 'L', False)]}")
    from collections import Counter
    cul = Counter(); ex = {}
    for r, d in wrong_where_hangs_right:
        seen = set()
        for c, p, f in d[4]:
            if (p == "loses") == d[0] and c.text not in seen:              # voted with the (wrong) decision; one count a move, however many embeddings
                seen.add(c.text); cul[c.text] += 1; ex.setdefault(c.text, []).append((r, c, p, f))
    log(f"OBS {name}: {len(wrong_where_hangs_right)} moves the library gets wrong where 'hangs' is right; the lemmas deciding them, most frequent first:")
    for text, k in cul.most_common(8):
        log(f"   {k:5d}  {text[:170]}")
    out["cells"] = {"|".join(map(str, k)): v for k, v in cells.items()}; out["culprits"] = cul.most_common(20)
    shown = set(); printed = 0
    def show(title, r, c, p, f):
        b = boards[r[0]]; m = chess.Move.from_uci(r[1])
        log(f"\nEXAMPLE ({name}, {title}): {b.fen()}  {'white' if b.turn else 'black'} to move: {b.san(m)}")
        log(b.unicode(borders=False, empty_square='·'))
        log(f"  Stockfish: {r[2]:+d} cp against standing still, {r[3]:+d} against the best move -> {'LOSES' if t(r) else 'keeps'}")
        nb = b.copy(stack=False); nb.push(m)
        nm = lambda sqs: ", ".join(f"{nb.piece_at(x).symbol()}{chess.square_name(x)}" for x in sqs) or "none"
        log(f"  after {b.san(m)}, {chess.square_name(m.to_square)} is attacked by {nm(nb.attackers(not b.turn, m.to_square))}; defended by {nm(nb.attackers(b.turn, m.to_square))}"
            f" -> hangs says {'loses' if r[5] else 'keeps'}")
        log(f"  library says {p}: {c.text[:320]}")
        log(f"  the guard term that fired: {f}" if f else "  no guard term fired (the claim itself decided)")
    for text, k in cul.most_common():
        if printed >= n_examples:
            break
        e = next((x for x in ex[text] if x[0][0] not in shown), None)       # a position not shown yet (the top lemmas are one lemma in several slot orders)
        if e is None:
            continue
        shown.add(e[0][0]); printed += 1
        show(f"a wrong verdict where 'hangs' is right; this lemma decides {k} such moves", *e)
    # the library as a VETO on 'hangs': loses only when both say so (its 'loses' overrides were mostly wrong, its
    # 'keeps' overrides mostly right)
    for truth_name, c2 in (("vs standing still", 2), ("vs the best move", 3)):
        comb = lambda r: r[5] and bool((decide(raw.get((r[0], r[1]), [])) or (False,))[0])
        sc = scores([(r[c2] <= -LOSS_CP, comb(r)) for r in rows]); sh = scores([(r[c2] <= -LOSS_CP, r[5]) for r in rows])
        lo, hi = bootstrap_diff(rows, c2, comb, lambda r: r[5])
        log(f"OBS {name}: {truth_name}: 'hangs' unless the library says keeps -> balanced acc {sc['bacc']:.3f} (p {sc['prec']:.3f} r {sc['rec']:.3f}) "
            f"vs hangs {sh['bacc']:.3f}; difference {sc['bacc'] - sh['bacc']:+.3f} (95% over positions {lo:+.3f} .. {hi:+.3f})")
        out.setdefault("veto", {})[truth_name] = dict(combined=sc, hangs=sh, ci=[lo, hi])
    right = [(r, decide(raw.get((r[0], r[1]), []))) for r in rows if r[5] and not t(r)]
    right = [(r, d) for r, d in right if d and not d[0]]
    for r, d in right[:2 * n_examples]:
        if r[0] in shown:
            continue
        shown.add(r[0]); c, p, f = next((x for x in d[4] if x[1] == "keeps"), d[4][0])
        show("a correct override: 'hangs' says loses, the library says keeps, Stockfish agrees with the library", r, c, p, f)
        if sum(1 for _ in shown) >= 2 * n_examples:
            break
    return out


def frame_gauge(boards, rows, n_examples: int = 3, log=print) -> dict:
    """Moves judged in EXCHANGE FRAMES (Kaveh 2026-09-27): each move's value against standing still is the exchange
    at its destination minus the exchange standing still allows on its origin, inside the frame a chooser picks
    around each square: the learned prior, the square's direct attackers and defenders, the old four-piece contest.
    'loses' when the value is at most -LOSS_CP/100 pawns."""
    from .fates import exchange_move_value
    from .frame_prior import Prior, rule_frames
    prior = Prior.load()
    choosers = dict(prior=prior.frame, direct=lambda b, s: rule_frames(b, s)["direct"], contest4=lambda b, s: rule_frames(b, s)["contest4"])
    out = {}
    for name, fo in choosers.items():
        t0 = time.perf_counter(); vals = {}
        for r in rows:
            b = boards[r[0]]; vals[(r[0], r[1])] = exchange_move_value(b, chess.Move.from_uci(r[1]), fo)
        pred = lambda r: vals[(r[0], r[1])][0] <= -LOSS_CP / 100
        rep_ = {}
        for truth_name, col in (("vs standing still", 2), ("vs the best move", 3)):
            sc = scores([(r[col] <= -LOSS_CP, pred(r)) for r in rows]); sh = scores([(r[col] <= -LOSS_CP, r[5]) for r in rows])
            lo, hi = bootstrap_diff(rows, col, pred, lambda r: r[5])
            log(f"OBS exchange frames by {name:8s} {truth_name}: balanced acc {sc['bacc']:.3f} (p {sc['prec']:.3f} r {sc['rec']:.3f}) vs hangs {sh['bacc']:.3f}; "
                f"difference {sc['bacc'] - sh['bacc']:+.3f} (95% over positions {lo:+.3f} .. {hi:+.3f})  [{time.perf_counter() - t0:.0f}s]")
            rep_[truth_name] = dict(frames=sc, hangs=sh, ci=[lo, hi])
        out[name] = rep_
        if name == "prior":
            wrong = [r for r in rows if pred(r) != (r[2] <= -LOSS_CP) and r[5] == (r[2] <= -LOSS_CP)]
            log(f"OBS prior frames wrong where 'hangs' is right: {len(wrong)} of {len(rows)} moves "
                f"(says loses: {sum(1 for r in wrong if pred(r))}, says keeps: {sum(1 for r in wrong if not pred(r))})")
            shown = set()
            for r in wrong:
                if len(shown) >= n_examples or r[0] in shown:
                    continue
                shown.add(r[0]); b = boards[r[0]]; m = chess.Move.from_uci(r[1]); v, dl, ol = vals[(r[0], r[1])]
                nb = b.copy(stack=False); nb.push(m)
                fr = prior.frame(nb, m.to_square)
                log(f"\nEXAMPLE (prior frames wrong, 'hangs' right): {b.fen()}  {'white' if b.turn else 'black'} to move: {b.san(m)}")
                log(b.unicode(borders=False, empty_square='·'))
                log(f"  Stockfish: {r[2]:+d} cp against standing still, {r[3]:+d} against the best move -> {'LOSES' if r[2] <= -LOSS_CP else 'keeps'}")
                log(f"  frame around {chess.square_name(m.to_square)} after the move: {' '.join(nb.piece_at(q).symbol() + chess.square_name(q) for q in fr)}")
                log(f"  exchange value {v:+.1f}: destination line {dl or '(nobody captures)'}; standing still would lose on the origin: {ol or 'nothing'}; hangs says {'loses' if r[5] else 'keeps'}")
    return out


def ledger_gauge(boards, rows, n_examples: int = 3, log=print) -> dict:
    """Moves judged in exchange frames, two ways and against both truths (Kaveh 2026-09-27: the frame's interface is
    the higher-value intermezzo elsewhere): SINGLE = the destination and origin frames (the prior's frames); LEDGER =
    what the move wins, minus the opponent's best exchange anywhere after it, plus their best if we had passed (the
    direct frames). 'vs standing still': loses when the value is at most -1.5; 'vs the best move': loses when it is
    1.5 or more below the position's best value by the same measure -- a comparison between moves that a one-move
    rule cannot make."""
    from .fates import direct_frame, exchange_move_value, ledger_move_value
    from .frame_prior import Prior
    prior = Prior.load(); thr = LOSS_CP / 100
    t0 = time.perf_counter(); single = {}; ledger = {}
    for r in rows:
        b = boards[r[0]]; m = chess.Move.from_uci(r[1])
        single[(r[0], r[1])] = exchange_move_value(b, m, prior.frame)[0]
        ledger[(r[0], r[1])] = ledger_move_value(b, m, lambda bb, s: direct_frame(bb, s))
    best_s = {}; best_l = {}
    for r in rows:
        best_s[r[0]] = max(best_s.get(r[0], -1e9), single[(r[0], r[1])]); best_l[r[0]] = max(best_l.get(r[0], -1e9), ledger[(r[0], r[1])][0])
    preds = {"single vs standing": lambda r: single[(r[0], r[1])] <= -thr, "single vs best": lambda r: single[(r[0], r[1])] <= best_s[r[0]] - thr,
             "ledger vs standing": lambda r: ledger[(r[0], r[1])][0] <= -thr, "ledger vs best": lambda r: ledger[(r[0], r[1])][0] <= best_l[r[0]] - thr}
    log(f"[verdicts] single and ledger values for {len(rows)} moves ({time.perf_counter() - t0:.0f}s)")
    out = {}
    for truth_name, col in (("vs standing still", 2), ("vs the best move", 3)):
        sh = scores([(r[col] <= -LOSS_CP, r[5]) for r in rows])
        for name, f in preds.items():
            sc = scores([(r[col] <= -LOSS_CP, f(r)) for r in rows]); lo, hi = bootstrap_diff(rows, col, f, lambda r: r[5])
            log(f"OBS truth {truth_name}: {name:18s} balanced acc {sc['bacc']:.3f} (p {sc['prec']:.3f} r {sc['rec']:.3f}) vs hangs {sh['bacc']:.3f}; "
                f"difference {sc['bacc'] - sh['bacc']:+.3f} ({lo:+.3f} .. {hi:+.3f})")
            out[f"{truth_name}|{name}"] = dict(scores=sc, hangs=sh, ci=[lo, hi])
    f = preds["ledger vs best"]; col = 3
    wrong = [r for r in rows if f(r) != (r[col] <= -LOSS_CP) and r[5] == (r[col] <= -LOSS_CP)]
    log(f"OBS ledger vs best wrong where 'hangs' is right: {len(wrong)} (says loses {sum(1 for r in wrong if f(r))}, keeps {sum(1 for r in wrong if not f(r))})")
    shown = set()
    for r in wrong:
        if len(shown) >= n_examples or r[0] in shown:
            continue
        shown.add(r[0]); b = boards[r[0]]; m = chess.Move.from_uci(r[1]); v, after, before = ledger[(r[0], r[1])]
        bm = max((x for x in rows if x[0] == r[0]), key=lambda x: ledger[(x[0], x[1])][0])
        nm = lambda t: f"{t[0]:+.0f} on {chess.square_name(t[1]) if t[1] is not None else '-'} {t[2]}"
        log(f"\nEXAMPLE (ledger vs best wrong, 'hangs' right): {b.fen()}  {'white' if b.turn else 'black'} to move: {b.san(m)}")
        log(b.unicode(borders=False, empty_square='·'))
        log(f"  Stockfish: {r[2]:+d} against standing still, {r[3]:+d} against the best move -> {'LOSES' if r[3] <= -LOSS_CP else 'keeps'}; hangs says {'loses' if r[5] else 'keeps'}")
        log(f"  ledger: {v:+.1f} (their best after: {nm(after)}; if we pass: {nm(before)}); the ledger's best move {b.san(chess.Move.from_uci(bm[1]))} {ledger[(bm[0], bm[1])][0]:+.1f}")
    return out


COMBINE_SIZES = range(1, 9)


def move_features(vs: list) -> list:
    """A LEARNED COMBINATION's inputs (Kaveh 2026-09-27: "better lemma combination"): from every lemma that speaks on
    a move, per frame size k = 1..8 -- the mean and the worst predicted size, the share saying 'loses', and whether any
    speaks -- then the overall mean and worst size. Sizes are the fired term's mean outcome (size_of)."""
    by = {k: [] for k in COMBINE_SIZES}; allv = []
    for c, p, f, _ in vs:
        v = size_of(c, p, f)
        v = (-1.0 if p == "loses" else 0.0) if v is None else v
        k = min(c.frame.size, max(COMBINE_SIZES))
        by[k].append((v, p == "loses")); allv.append(v)
    out = []
    for k in COMBINE_SIZES:
        xs = by[k]
        out += [sum(v for v, _ in xs) / len(xs) if xs else 0.0, min((v for v, _ in xs), default=0.0),
                sum(1 for _, l in xs if l) / len(xs) if xs else 0.0, 1.0 if xs else 0.0]
    out += [sum(allv) / len(allv) if allv else 0.0, min(allv, default=0.0)]
    return out


FEATURE_NAMES = [f"size{k}_{n}" for k in COMBINE_SIZES for n in ("mean", "worst", "loses_share", "speaks")] + ["all_mean", "all_worst"]


def combine_gauge(a, eng_depth: int, lib_spec: str, n_train: int, log=print) -> dict:
    """Train the combination on moves of n_train further held-out positions (Stockfish's truth), apply it to the
    gauge's 1,000 positions, against 'hangs' -- one logistic regression per truth, its weights printed; for the truth
    against the best move a move's mean size relative to the position's best is added (a comparison between moves)."""
    from sklearn.linear_model import LogisticRegression
    from .chains import domain_of
    from .lemma_apply import Applier, load_library
    n_eval = a.n
    boards = positions(n_eval + n_train, a.seed)
    ev, tr = boards[:n_eval], boards[n_eval:]
    eng = Engine(threads=a.threads, depth=eng_depth)
    def rows_of(bs, offset):
        out = []
        for i, b in enumerate(bs):
            t = truths(eng, b, eng_depth)
            if t is None:
                continue
            for m in b.legal_moves:
                if m.uci() in t:
                    out.append((offset + i, m.uci(), *t[m.uci()], *rules(b, m)))
        return out
    t0 = time.perf_counter()
    r_ev = rows_of(ev, 0); r_tr = rows_of(tr, n_eval); eng.close()
    log(f"[combine] truths: {len(r_ev)} evaluation moves, {len(r_tr)} training moves ({time.perf_counter() - t0:.0f}s)")
    name, lp, prov = library_spec(lib_spec)
    app = Applier(domain_of(lp), load_library(lp, prov))
    raw = {(i, key): vs for i, per in raw_all(app, boards, a.workers).items() for key, vs in per.items()}
    log(f"[combine] library {name}: {len(app.lemmas)} lemmas applied to {len(boards)} positions ({time.perf_counter() - t0:.0f}s)")
    def feats(rows):
        F = [move_features(raw.get((r[0], r[1]), [])) for r in rows]
        best = {}
        for r, f in zip(rows, F):
            best[r[0]] = max(best.get(r[0], -1e9), f[-2])
        return np.array([f + [f[-2] - best[r[0]]] for r, f in zip(rows, F)])
    import numpy as np
    X_tr, X_ev = feats(r_tr), feats(r_ev)
    out = {}
    for truth_name, col in (("vs standing still", 2), ("vs the best move", 3)):
        cols = slice(0, len(FEATURE_NAMES) + (1 if col == 3 else 0))
        y_tr = np.array([r[col] <= -LOSS_CP for r in r_tr])
        model = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(X_tr[:, cols], y_tr)
        pred = dict(zip(((r[0], r[1]) for r in r_ev), model.predict(X_ev[:, cols])))
        f = lambda r: bool(pred[(r[0], r[1])])
        sc = scores([(r[col] <= -LOSS_CP, f(r)) for r in r_ev]); sh = scores([(r[col] <= -LOSS_CP, r[5]) for r in r_ev])
        lo, hi = bootstrap_diff(r_ev, col, f, lambda r: r[5])
        log(f"OBS {truth_name}: the learned combination {sc['bacc']:.3f} (p {sc['prec']:.3f} r {sc['rec']:.3f}) vs hangs {sh['bacc']:.3f}; "
            f"difference {sc['bacc'] - sh['bacc']:+.3f} ({lo:+.3f} .. {hi:+.3f})")
        names = FEATURE_NAMES + (["relative_to_best"] if col == 3 else [])
        log("OBS   weights: " + ", ".join(f"{n} {w:+.2f}" for n, w in sorted(zip(names, model.coef_[0]), key=lambda t: -abs(t[1]))[:10]))
        out[truth_name] = dict(scores=sc, hangs=sh, ci=[lo, hi], weights=dict(zip(names, model.coef_[0].tolist())))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 1))
    ap.add_argument("--library", action="append", default=None, help="name=library.json[:proved.json] (repeatable; a checkpoint_*.json gives its library)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--diagnose", action="store_true", help="where each library loses to 'hangs', with worked examples")
    ap.add_argument("--combine", default="", help="name=library.json[:proved]: a learned combination of the library's verdicts, trained on --train further positions")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--ledger", action="store_true", help="single frames and the cross-frame ledger, against standing still and against the best move")
    ap.add_argument("--frames", action="store_true", help="moves judged in exchange frames picked by the prior / the direct rule / the old contest, instead of libraries")
    a = ap.parse_args(argv)
    L = paths.derived("lemmas")
    specs = a.library or [f"refined905={L}/library_games_frame6k.json:{L}/proved_games_frame6k.json",
                          f"full3_partial={L}/checkpoint_chess_games_frame_full3.json",
                          f"cheap_partial={L}/checkpoint_chess_games_frame_contests_counter2.json"]
    from .chains import domain_of
    from .lemma_apply import Applier, load_library
    t0 = time.perf_counter()
    if a.combine:
        rep_ = combine_gauge(a, a.depth, a.combine, a.train)
        out = Path(paths.derived("lemmas", f"gauge_verdicts_combine{('_' + a.tag) if a.tag else ''}.json"))
        json.dump(rep_, open(out, "w"), indent=1, default=str)
        print(f"[verdicts] -> {out} ({time.perf_counter() - t0:.0f}s)"); return
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
    if a.ledger:
        report["ledger"] = ledger_gauge(boards, rows)
        out = Path(paths.derived("lemmas", f"gauge_verdicts_ledger{('_' + a.tag) if a.tag else ''}.json"))
        json.dump(report, open(out, "w"), indent=1, default=str)
        print(f"[verdicts] -> {out} ({time.perf_counter() - t0:.0f}s)"); return
    if a.frames:
        report["frames"] = frame_gauge(boards, rows)
        out = Path(paths.derived("lemmas", f"gauge_verdicts_frames{('_' + a.tag) if a.tag else ''}.json"))
        json.dump(report, open(out, "w"), indent=1, default=str)
        print(f"[verdicts] -> {out} ({time.perf_counter() - t0:.0f}s)"); return
    if a.diagnose:
        report["diagnosis"] = {}
        for spec in specs:
            name, lp, prov = library_spec(spec)
            app = Applier(domain_of(lp), load_library(lp, prov))
            report["diagnosis"][name] = diagnose(boards, rows, name, app, workers=a.workers)
        out = Path(paths.derived("lemmas", f"gauge_verdicts_diagnosis{('_' + a.tag) if a.tag else ''}.json"))
        json.dump(report, open(out, "w"), indent=1, default=str)
        print(f"[verdicts] -> {out} ({time.perf_counter() - t0:.0f}s)"); return
    for spec in specs:
        name, lp, prov = library_spec(spec)
        t1 = time.perf_counter()
        app = Applier(domain_of(lp), load_library(lp, prov))
        says = {(i, key): v for i, vs in verdicts_all(app, boards, a.workers).items() for key, v in vs.items()}
        spoken = [r for r in rows if (r[0], r[1]) in says]
        ties = sum(1 for r in spoken if says[(r[0], r[1])][2])
        agree = sum(1 for r in spoken if says[(r[0], r[1])][0] == r[5])
        print(f"OBS library {name}: {len(app.lemmas)} lemmas; speaks on {len(spoken)}/{len(rows)} moves ({100 * len(spoken) / max(1, len(rows)):.1f}%), "
              f"top-size ties {ties}; says what 'hangs' says on {agree}/{len(spoken)} ({time.perf_counter() - t1:.0f}s)")
        rep = dict(lemmas=len(app.lemmas), spoken=len(spoken), ties=ties, agree_hangs=agree)
        for truth_name, col in (("vs standing still", 2), ("vs the best move", 3)):
            sized = lambda r: (lambda v: v is not None and v[3] is not None and v[3] <= -LOSS_CP / 100)(says.get((r[0], r[1])))
            s_size = scores([(r[col] <= -LOSS_CP, sized(r)) for r in rows]); lo_s, hi_s = bootstrap_diff(rows, col, sized, lambda r: r[5])
            print(f"OBS   {truth_name}: by SIZE (the deciders' mean outcome <= -{LOSS_CP / 100:.1f}): balanced acc {s_size['bacc']:.3f} (p {s_size['prec']:.3f} r {s_size['rec']:.3f}); "
                  f"minus 'hangs' {s_size['bacc'] - scores([(r[col] <= -LOSS_CP, r[5]) for r in rows])['bacc']:+.3f} ({lo_s:+.3f} .. {hi_s:+.3f})")
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
