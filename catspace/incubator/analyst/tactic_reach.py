"""TACTIC REACHABILITY (Kaveh 2026-10-09: "we also need a tactic reachability model -- once it's fast, label some tactics
based on Stockfish eval and try to predict it from the reasonings we carry at previous points").

LABEL (Stockfish, two best moves, side to move's view): a TACTIC is available at ply t when the best move beats the
second best by at least GAP centipawns and leaves the mover at least WIN up -- an "only move that wins", the puzzle
definition. A position with one legal move is not a tactic.

PREDICT the label at t from what we carried BEFORE t, never from the board at t:
  frames   the waiting-frames pool (waiting_frames.py) at t-1 (the opponent to move; our frames on the null move) and at
           t-2 (our previous turn): for each side, active / dormant / far frames by family, the conditions our dormant
           frames wait on, the breakers on our active ones
  board    the attack map at t-1 (attention.py: contested, loose pieces, king pressure per side), material, phase, the
           number of legal moves -- what a reader of the board at t-1 sees without the frames
  eval     Stockfish's evaluation at t-1 -- a reference, not a competitor (it is a search)
Grouped cross-validation by game (a game never sits in both train and test). Reported: ranking quality (AUC) and
average precision against the base rate, for each feature set and their union.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import chess

from catspace.io import paths

GAP, WIN = 150, 150
GAIN = 2            # pawns of material the best line must leave the mover up, against the position BEFORE the opponent's last move
ALT_WINS = 300      # if the second-best move already wins this much, the best move is not an "only move"
PV_PLIES = 8        # how far along Stockfish's line the material is read (then the captures are resolved)

_T0 = time.time()


def _tick(what: str, t: float) -> float:
    now = time.time()
    print(f"TIME {what}: {now - t:.1f} s (total {now - _T0:.0f} s)", flush=True)
    return now
DEPTH = 12
FAMILIES = ("check", "fork", "single threat", "cornered piece", "trapped piece")
STATUSES = ("active", "dormant", "far")


def _cp(score) -> int:
    return max(-1500, min(1500, score.score(mate_score=10000)))


_ENGINE = None


def _label_one(args):
    global _ENGINE
    game, ply, fen = args
    import chess.engine
    if _ENGINE is None:
        _ENGINE = chess.engine.SimpleEngine.popen_uci("stockfish")
        _ENGINE.configure({"Threads": 1, "Hash": 32})
        # (the pool's workers are terminated, not shut down, so no atexit runs; the orphaned engine exits on stdin EOF)
    b = chess.Board(fen)
    n = b.legal_moves.count()
    if n == 0:
        return dict(game=game, ply=ply, best=None, second=None, move=None)
    info = _ENGINE.analyse(b, chess.engine.Limit(depth=DEPTH), multipv=min(2, n))
    best = _cp(info[0]["score"].pov(b.turn))
    second = _cp(info[1]["score"].pov(b.turn)) if n > 1 else None
    pv = [m.uci() for m in info[0].get("pv", [])[:PV_PLIES]]
    return dict(game=game, ply=ply, best=best, second=second, move=pv[0] if pv else None, pv=pv,
                mate=info[0]["score"].pov(b.turn).is_mate())


def label(tag: str, workers: int) -> dict:
    from multiprocessing import Pool
    from .tactic_census_measure import load
    ck = paths.experiment(f"tactic_labels_{tag}_d{DEPTH}_pv.jsonl")
    done = {}
    if os.path.exists(ck):
        with open(ck) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(r["game"], r["ply"])] = r
    todo = [(r["game"], r["ply"], r["fen"]) for r in load(tag) if (r["game"], r["ply"]) not in done]
    print(f"labels: {len(done)} checkpointed, {len(todo)} to go", flush=True)
    if todo:
        t0 = time.time()
        print(f"  Stockfish depth {DEPTH}, two best moves, {workers} processes", flush=True)
        with Pool(workers) as p, open(ck, "a") as fh:
            for i, r in enumerate(p.imap_unordered(_label_one, todo, chunksize=16)):
                fh.write(json.dumps(r) + "\n"); fh.flush(); done[(r["game"], r["ply"])] = r   # per-item checkpoint
                if (i + 1) % 2000 == 0:
                    print(f"  {i + 1}/{len(todo)} labelled, {time.time() - t0:.0f} s", flush=True)
    return done


_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _material(b: chess.Board, color: bool) -> int:
    return sum(v * (len(b.pieces(p, color)) - len(b.pieces(p, not color))) for p, v in _VAL.items())


def material_gain(fen_before: str, fen: str, pv: list) -> int:
    """The mover's material at the end of Stockfish's line (its first PV_PLIES moves, then every capture still hanging
    resolved by the exchange law's static exchange -- cheap and enough for a label) minus the mover's material BEFORE
    the opponent's last move. A recapture restores what was there: gain 0. (Spot check 2026-10-09: four of eight
    sampled "tactics" were recaptures -- Nxd5 after Bxd5, Rxf1 after Nxf1, Rxc6, gxh6 -- the only move only because
    not recapturing loses.)"""
    b = chess.Board(fen); us = b.turn
    before = _material(chess.Board(fen_before), us)
    for u in pv:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            break
        b.push(m)
    return _material(b, us) - before


def is_tactic(r, fen_before: str | None = None, fen: str | None = None) -> bool:
    """An only move that wins: best beats second by GAP and leaves the mover WIN up, the second best does not already
    win ALT_WINS (a faster mate in a won ending is not a tactic), and -- given the boards -- the line wins GAIN pawns of
    material (or mates) against the position before the opponent's last move."""
    if r["best"] is None or r["second"] is None:
        return False
    if not (r["best"] - r["second"] >= GAP and r["best"] >= WIN and r["second"] < ALT_WINS):
        return False
    if fen_before is None or fen is None:
        return True
    return bool(r.get("mate")) or material_gain(fen_before, fen, r.get("pv") or []) >= GAIN


def _frame_feats(snap: dict, us: str, prefix: str) -> dict:
    f = defaultdict(float)
    for key, v in snap.items():
        st, alts, born, name, val = v[:5]
        brk = v[5] if len(v) > 5 else []
        who = "us" if key[0] == us else "them"
        f[f"{prefix}{who}_{st}"] += 1
        f[f"{prefix}{who}_{name}_{st}"] += 1
        if st == "dormant":
            f[f"{prefix}{who}_conditions"] += len(alts)
        if st == "active":
            f[f"{prefix}{who}_breakers"] += len(brk)
    return f


def _board_feats(fen: str, us_color: bool) -> dict:
    from .attention import components
    b = chess.Board(fen)
    c = components(b)
    f = {}
    for part in ("contested", "loose"):
        f[f"board_{part}_us"] = sum(1 for s in c[part] if b.color_at(s) == us_color)
        f[f"board_{part}_them"] = sum(1 for s in c[part] if b.color_at(s) != us_color)
    ks = {b.king(chess.WHITE), b.king(chess.BLACK)}
    f["board_king_hot_them"] = float(b.king(not us_color) in c["king"])
    f["board_king_hot_us"] = float(b.king(us_color) in c["king"])
    f["board_legal"] = b.legal_moves.count()
    val = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    f["board_material"] = sum(v * (len(b.pieces(p, us_color)) - len(b.pieces(p, not us_color))) for p, v in val.items())
    f["board_pieces"] = len(b.piece_map())
    return f


def dataset(tag: str, frames_file: str, labels: dict):
    from .tactic_census_measure import load
    t = time.time()
    census = load(tag)
    ev = {(r["game"], r["ply"]): r["eval"] for r in census}
    plies_of = defaultdict(list)
    for (g, pl) in labels:
        plies_of[g].append(pl)
    t = _tick("dataset: census + label index", t)
    rows = []; t_feat = t_frames = 0.0; t_json = time.time()
    lines = open(paths.experiment(frames_file)).readlines()
    t = _tick(f"dataset: read the frames file ({len(lines)} games)", t)
    for line in lines:
        r = json.loads(line)
        if not r.get("proofs", True):
            continue
        fens, hist = r["fens"], r["hist"]
        plies = sorted(plies_of[r["game"]])
        if len(plies) != len(fens):
            continue                                                   # the frames replay and the census disagree
        for i in range(2, len(fens)):
            lab = labels.get((r["game"], plies[i]))
            if lab is None or lab["best"] is None:
                continue
            us_color = chess.Board(fens[i]).turn
            us = "w" if us_color else "b"
            x = {}
            a = time.time()
            x.update(_frame_feats(hist[i - 1], us, "f1_"))
            x.update(_frame_feats(hist[i - 2], us, "f2_"))
            b_ = time.time(); t_frames += b_ - a
            x.update(_board_feats(fens[i - 1], us_color))
            t_feat += time.time() - b_
            e = ev.get((r["game"], plies[i - 1]))
            x["eval_prev_for_us"] = max(-1500, min(1500, -e)) if e is not None else 0.0   # the eval at t-1 is the opponent's view
            rows.append((r["game"], x, is_tactic(lab, fens[i - 1], fens[i])))
    print(f"TIME dataset: frame features {t_frames:.1f} s, board features {t_feat:.1f} s over {len(rows)} positions", flush=True)
    _tick("dataset: total", t)
    return rows


def evaluate(rows) -> None:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    names = sorted({k for _, x, _ in rows for k in x})
    X = np.array([[x.get(k, 0.0) for k in names] for _, x, _ in rows])
    y = np.array([lab for _, _, lab in rows], dtype=int)
    groups = np.array([g for g, _, _ in rows])
    print(f"OBS {len(rows)} positions from {len(set(groups))} games; tactics (Stockfish: best beats second by >= {GAP} cp and "
          f"wins >= {WIN} cp) {y.sum()} ({100 * y.mean():.1f}%, the base rate = the average precision of a constant model)")
    sets = {
        "frames carried (t-1, t-2)": [i for i, n in enumerate(names) if n.startswith(("f1_", "f2_"))],
        "frames at t-1 only": [i for i, n in enumerate(names) if n.startswith("f1_")],
        "board at t-1 (attack map, material)": [i for i, n in enumerate(names) if n.startswith("board_")],
        "frames + board": [i for i, n in enumerate(names) if n.startswith(("f1_", "f2_", "board_"))],
        "Stockfish eval at t-1 (reference)": [names.index("eval_prev_for_us")],
        "frames + board + eval": list(range(len(names))),
    }
    cv = GroupKFold(n_splits=5)
    t = time.time()
    for sname, cols in sets.items():
        for mname, make in (("logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
                            ("boosted trees", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05))):
            p = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                m = make().fit(X[tr][:, cols], y[tr])
                p[te] = m.predict_proba(X[te][:, cols])[:, 1]
            print(f"OBS   {sname:38s} {mname:14s} AUC {roc_auc_score(y, p):.3f}  average precision {average_precision_score(y, p):.3f}", flush=True)
            t = _tick(f"model {sname} / {mname} (5 folds)", t)
    # which carried facts carry the signal (logistic on frames + board, standardized coefficients)
    cols = sets["frames + board"]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[:, cols], y)
    coef = sorted(zip(m[-1].coef_[0], [names[i] for i in cols]), key=lambda t: -abs(t[0]))[:12]
    print("OBS   strongest features (logistic, standardized):", ", ".join(f"{n} {c:+.2f}" for c, n in coef))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--frames", default="waiting_frames_v2_c300_300.jsonl")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--label-only", action="store_true")
    a = ap.parse_args(argv)
    t = time.time()
    labels = label(a.tag, a.workers)
    t = _tick("labels (Stockfish, or the checkpoint)", t)
    if a.label_only:
        return
    rows = dataset(a.tag, a.frames, labels)
    evaluate(rows)
    _tick("all", _T0)


if __name__ == "__main__":
    main()
