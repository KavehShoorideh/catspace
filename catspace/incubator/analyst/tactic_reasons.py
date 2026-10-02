"""WHAT WOULD HAVE PREVENTED IT -- tactics classified by their preventers (Kaveh 2026-10-01: "looking at games where,
before material is lost, the evaluation suddenly swings ... there is a tactic available ... that is the frame. Predict
the situation that creates the frame, the frame itself, the pieces involved ... finding the similarity depends on
finding the reasoning ... we start from a seed and ask what would have prevented this. And all of those where the same
thing would have prevented this, like unpin, those are all a pinning situation.")

The seeds: Lichess puzzles -- each one is a game position right after a blunder, where a tactic became available (the
evaluation swung before any material was lost), with the line that cashes it in and the site's THEME tags (pin, fork,
backRankMate, ...), which are used only to check the result, never to build it.

For the position where the tactic is available, the oracle is Stockfish (depth 12, one thread per worker): the
tactic's STAKE = its evaluation for the side that has the tactic minus the material balance (so an edit that changes
material is not mistaken for one that changes the tactic). Two kinds of edits, each PREVENTS the tactic when the stake
falls below a third of what it was (and below a pawn):
- REMOVE: one non-king piece taken off the board -- the pieces the tactic needs (or that it needs to be there: the
  victim's own pawn boxing in its king).
- DEFEND: the victim given one free move (the side with the tactic passes) -- the victim's defences: unpin, luft,
  interpose, defend, step away.
Each preventer is described by generic relations to the tactic's KEY squares -- the first solution move's squares, the
pieces the line wins, the victim's king: its side and piece type, whether it is the victim's king or adjacent to it,
on a line between two key pieces, attacks or defends a key piece, is pinned; for a defending move, whether it moves
the king, is a pawn move next to the king, lands between two key pieces, captures a piece of the line, adds a defender
to a key piece. The puzzle's SIGNATURE: which of these its preventers show.
Printed: per theme, the preventer features over-represented in it (lift); and how well the signature predicts each
theme on held-out puzzles (AUC; a constant is 0.5).

    python -m catspace.incubator.analyst.tactic_reasons collect --n 100 --offset 200     # a pilot
    python -m catspace.incubator.analyst.tactic_reasons report
"""
from __future__ import annotations

import argparse
import json
import os
import time

import chess
import chess.engine

from catspace.io import paths

DEPTH = 12
MATE_CP = 2000
VAL = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
THEMES = ("fork", "pin", "skewer", "backRankMate", "discoveredAttack", "deflection", "attraction", "hangingPiece",
          "doubleCheck", "smotheredMate", "trappedPiece", "interference", "clearance", "xRayAttack", "exposedKing",
          "kingsideAttack", "sacrifice", "mate", "advancedPawn", "quietMove", "defensiveMove", "capturingDefender")
_ENG = {}


def _engine():
    if "e" not in _ENG:
        import shutil
        e = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
        e.configure({"Threads": 1, "Hash": 32})
        _ENG["e"] = e
    return _ENG["e"]


def _material(b: chess.Board, side: chess.Color) -> int:
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values())


def stake(b: chess.Board, owner: chess.Color) -> float | None:
    """Stockfish's evaluation for `owner` minus the material balance, in centipawns; None for an invalid position."""
    if not b.is_valid():
        return None
    if b.is_checkmate():
        ev = -MATE_CP if b.turn == owner else MATE_CP
    elif b.is_game_over():
        ev = 0
    else:
        s = _engine().analyse(b, chess.engine.Limit(depth=DEPTH, time=5.0))["score"].pov(owner)
        ev = s.score(mate_score=MATE_CP)
        ev = max(-MATE_CP, min(MATE_CP, ev))
    return ev - _material(b, owner)


def _between(a: int, c: int, s: int) -> bool:
    return s in chess.SquareSet(chess.between(a, c))


def _key(b: chess.Board, line: list) -> dict:
    """The tactic's key squares: the first solution move's squares, the pieces the line wins (where they stood), the
    victim's king."""
    owner = b.turn; victim = not owner
    bb = b.copy(); won = set()
    for i, m in enumerate(line):
        if bb.is_capture(m) and bb.turn == owner:
            sq = m.to_square
            if bb.is_en_passant(m):
                sq = chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))
            won.add(sq)
        bb.push(m)
    first = line[0]
    pieces = {first.from_square} | {s for s in won if b.piece_at(s) is not None}
    if not won and first.to_square is not None:
        pieces |= {t for t in chess.SquareSet(_after(b, first).attacks_mask(first.to_square))
                   if b.piece_at(t) is not None and b.piece_at(t).color == victim}   # a quiet first move: what it hits
    return dict(owner=owner, victim=victim, first=first, won=won, pieces=pieces, king=b.king(victim))


def _after(b: chess.Board, m: chess.Move) -> chess.Board:
    nb = b.copy(stack=False); nb.push(m)
    return nb


def describe_piece(b: chess.Board, sq: int, key: dict) -> list:
    """Generic relations of the piece on sq to the key squares."""
    pc = b.piece_at(sq); out = []
    side = "owner" if pc.color == key["owner"] else "victim"
    out.append(f"{side}:{chess.piece_name(pc.piece_type)}")
    k = key["king"]
    if k is not None and chess.square_distance(sq, k) == 1:
        out.append(f"{side}:next_to_victim_king")
    keys = [s for s in key["pieces"] | ({k} if k is not None else set()) if s != sq]
    if any(_between(a, c, sq) for a in keys for c in keys if a != c):
        out.append(f"{side}:between_key_pieces")
    att = chess.SquareSet(b.attacks_mask(sq))
    if any(s in att and b.piece_at(s).color != pc.color for s in keys if b.piece_at(s)):
        out.append(f"{side}:attacks_key_piece")
    if any(s in att and b.piece_at(s).color == pc.color for s in keys if b.piece_at(s)):
        out.append(f"{side}:defends_key_piece")
    if b.is_pinned(pc.color, sq):
        out.append(f"{side}:pinned")
    if sq == key["first"].from_square:
        out.append("owner:the_first_mover")
    if sq in key["won"]:
        out.append("victim:won_by_the_line")
    return out


def describe_defence(b: chess.Board, r: chess.Move, key: dict) -> list:
    """Generic relations of a victim's defending move to the key squares (b: the victim to move)."""
    pc = b.piece_at(r.from_square); out = [f"defend:{chess.piece_name(pc.piece_type)}"]
    k = key["king"]
    if pc.piece_type == chess.KING:
        out.append("defend:king_moves")
    if pc.piece_type == chess.PAWN and k is not None and chess.square_distance(r.from_square, k) == 1:
        out.append("defend:pawn_next_to_king_moves")             # luft, or a pawn shield moved
    keys = [s for s in key["pieces"] | ({k} if k is not None else set())]
    if any(_between(a, c, r.to_square) for a in keys for c in keys if a != c):
        out.append("defend:lands_between_key_pieces")           # an interposition
    if b.is_capture(r):
        out.append("defend:captures" + ("_key_piece" if r.to_square in key["pieces"] else ""))
    nb = _after(b, r)
    if any(s in chess.SquareSet(nb.attacks_mask(r.to_square)) and nb.piece_at(s) and nb.piece_at(s).color == pc.color
           for s in keys if s != r.to_square):
        out.append("defend:adds_defender_to_key_piece")
    if r.from_square in key["pieces"] or r.from_square == k:
        out.append("defend:a_key_piece_moves")
    if b.is_pinned(pc.color, r.from_square) is False and any(b.is_pinned(pc.color, s) for s in chess.SquareSet(b.occupied_co[pc.color])) \
            and not any(nb.is_pinned(pc.color, s) for s in chess.SquareSet(nb.occupied_co[pc.color])):
        out.append("defend:unpins")
    return out


def preventers_of(b: chess.Board, line: list, s0: float) -> tuple:
    """The tactic on b (its side to move has it; `line` cashes it in; `s0` its stake) and what would have prevented it:
    piece removals and the victim's free moves that collapse the stake. -> (preventers, the victim's move count)."""
    owner = b.turn
    key = _key(b, line)
    th = s0 / 3
    pre = []
    for sq, pc in b.piece_map().items():                              # REMOVE one piece
        if pc.piece_type == chess.KING:
            continue
        bb = b.copy(stack=False); bb.remove_piece_at(sq)
        v = stake(bb, owner)
        if v is not None and v < max(th, 100):
            pre.append(dict(kind="remove", square=chess.square_name(sq), stake=v, feats=describe_piece(b, sq, key)))
    pb = b.copy(stack=False); pb.turn = key["victim"]; pb.ep_square = None
    if pb.is_valid():                                                  # DEFEND: the victim's free move
        for r in list(pb.legal_moves):
            bb = _after(pb, r)
            v = stake(bb, owner)
            if v is not None and v < max(th, 100):
                pre.append(dict(kind="defend", move=pb.san(r), stake=v, feats=describe_defence(pb, r, key)))
    return pre, (pb.legal_moves.count() if pb.is_valid() else 0)


def puzzle(row: dict) -> dict | None:
    b = chess.Board(row["FEN"]); mv = row["Moves"].split()
    b.push_uci(mv[0])                                                  # the blunder: now the tactic is available
    line = [chess.Move.from_uci(u) for u in mv[1:]]
    if not line or line[0] not in b.legal_moves:
        return None
    owner = b.turn
    s0 = stake(b, owner)
    if s0 is None or s0 < 200:
        return dict(pid=row["PuzzleId"], skipped=f"stake {s0}")
    pre, n_def = preventers_of(b, line, s0)
    return dict(pid=row["PuzzleId"], themes=row["Themes"].split(), rating=int(row.get("Rating") or 0), fen=b.fen(),
                line=mv[1:], stake=s0, preventers=pre, n_pieces=len(b.piece_map()), n_defences=n_def)


def collect(n: int, offset: int, workers: int = 10, log=print) -> str:
    import multiprocessing as mp
    from .frame_puzzles import fresh_rows
    t0 = time.perf_counter()
    rows = fresh_rows(n, 997, offset)
    out = paths.derived(f"tactic_reasons_{n}_{offset}.jsonl")
    k = 0
    with mp.get_context("fork").Pool(workers) as pool, open(out, "w") as fh:
        for r in pool.imap_unordered(puzzle, rows, chunksize=1):
            if r is not None:
                fh.write(json.dumps(r) + "\n"); fh.flush(); k += 1
    log(f"[tactic_reasons] {len(rows)} puzzles, {k} written -> {out} ({time.perf_counter() - t0:.0f}s)")
    return str(out)


def control_signature(r: dict) -> list:
    """The CONTROL: the same relations over EVERY non-king piece and EVERY victim move -- no 'would have prevented it'
    filter. If the preventer signature beats this, the counterfactual (what would have prevented it) carries the
    reasoning, not the description of the position."""
    b = chess.Board(r["fen"]); line = [chess.Move.from_uci(u) for u in r["line"]]
    key = _key(b, line)
    out = set()
    for sq, pc in b.piece_map().items():
        if pc.piece_type != chess.KING:
            out |= set(describe_piece(b, sq, key)); out.add("remove:any")
    pb = b.copy(stack=False); pb.turn = key["victim"]; pb.ep_square = None
    if pb.is_valid():
        for m in pb.legal_moves:
            out |= set(describe_defence(pb, m, key)); out.add("defend:any")
    return sorted(out)


def _counts(r: dict, control: bool) -> dict:
    """Per feature, how many preventers (or, for the control, pieces and moves) show it -- counts, not just presence."""
    from collections import Counter
    if not control:
        return Counter(f for p in r["preventers"] for f in p["feats"] + [f"{p['kind']}:any"])
    b = chess.Board(r["fen"]); line = [chess.Move.from_uci(u) for u in r["line"]]
    key = _key(b, line); c = Counter()
    for sq, pc in b.piece_map().items():
        if pc.piece_type != chess.KING:
            c.update(describe_piece(b, sq, key) + ["remove:any"])
    pb = b.copy(stack=False); pb.turn = key["victim"]; pb.ep_square = None
    if pb.is_valid():
        for m in pb.legal_moves:
            c.update(describe_defence(pb, m, key) + ["defend:any"])
    return c


def report(path: str, log=print) -> None:
    import hashlib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    recs = [json.loads(l) for l in open(path)]
    skipped = [r for r in recs if "skipped" in r]; recs = [r for r in recs if "skipped" not in r]
    import statistics as st
    log(f"OBS {path.rsplit('/', 1)[-1]}: {len(recs)} tactics (skipped {len(skipped)}: stake under 2 pawns or invalid); "
        f"preventers a tactic: median {st.median(len(r['preventers']) for r in recs):g} (remove {st.median(sum(p['kind'] == 'remove' for p in r['preventers']) for r in recs):g}, "
        f"defend {st.median(sum(p['kind'] == 'defend' for p in r['preventers']) for r in recs):g}); no preventer at all in {sum(1 for r in recs if not r['preventers'])}")
    sig = [sorted({f for p in r["preventers"] for f in p["feats"]} | {f"{p['kind']}:any" for p in r["preventers"]}) for r in recs]
    vocab = sorted({f for s in sig for f in s})
    X = np.array([[f in s for f in vocab] for s in sig], dtype=float)
    folds = np.array([int(hashlib.sha1(("tr" + r["pid"]).encode()).hexdigest()[:8], 16) % 5 for r in recs])
    from collections import Counter
    vc = sorted({f for r in recs for f in _counts(r, True)} | set(vocab))
    Xc = np.array([[c.get(f, 0) for f in vc] for c in (_counts(r, True) for r in recs)], dtype=float)
    Xp = np.array([[c.get(f, 0) for f in vc] for c in (_counts(r, False) for r in recs)], dtype=float)
    Xc = np.log1p(Xc); Xp = np.log1p(Xp)

    def heldout(Xm, y):
        p = np.zeros(len(recs))
        for k in range(5):
            tr, te = folds != k, folds == k
            if y[tr].sum() and (~y[tr]).sum():
                p[te] = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(Xm[tr], y[tr]).predict_proba(Xm[te])[:, 1]
        return roc_auc_score(y, p)
    for t in THEMES:
        y = np.array([t in r["themes"] for r in recs])
        if y.sum() < 8:
            continue
        log(f"OBS {t}: held-out AUC with feature COUNTS -- the preventers {heldout(Xp, y):.3f}; the control (every piece and victim move, no "
            f"prevention filter) {heldout(Xc, y):.3f}; both together {heldout(np.hstack([Xp, Xc]), y):.3f}")
        lifts = []
        for j, f in enumerate(vocab):
            a = X[y, j].mean(); b_ = X[~y, j].mean()
            if a >= 0.25 and b_ > 0:
                lifts.append((a / b_, f, a, b_))
        lifts.sort(reverse=True)
        p = np.zeros(len(recs))
        for k in range(5):
            tr, te = folds != k, folds == k
            if y[tr].sum() and (~y[tr]).sum():
                p[te] = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
        log(f"OBS {t} ({int(y.sum())} of {len(recs)}): held-out AUC of the preventer signature {auc:.3f}; most over-represented: "
            + "; ".join(f"{f} {a:.2f} vs {b_:.2f}" for _, f, a, b_ in lifts[:3]))


PIECE_NAMES = tuple(chess.piece_name(t) for t in chess.PIECE_TYPES)


def _reason_only(c):
    """Drop the piece-type features (who prevents) and keep the relations (how it prevents): the reasoning."""
    from collections import Counter
    return Counter({f: n for f, n in c.items() if f.split(":", 1)[1] not in PIECE_NAMES})


def groups(path: str, ks=(10, 16, 24), detail_k: int = 16, reasons_only: bool = False, log=print) -> None:
    """THE TACTICS GROUPED WITHOUT THE THEMES (Kaveh 2026-10-01: "all of those where the same thing would have
    prevented this ... are all a pinning situation"): k-means on each tactic's preventer counts (log, L2-normalized
    rows), the same on the control's (every piece and victim move, no prevention filter), and random groups of the
    preventer groups' sizes. Each group is described by its most over-represented preventers -- its candidate LEMMA --
    and checked against the themes afterwards: per theme, the best-matching group's F1."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if "skipped" not in r]
    pc = [_counts(r, False) for r in recs]; cc = [_counts(r, True) for r in recs]
    if reasons_only:                                                  # (2026-10-01: with piece types the groups split by WHICH piece
        pc = [_reason_only(c) for c in pc]; cc = [_reason_only(c) for c in cc]   # defends -- a description, not a reason)
        log("[tactic_reasons] groups on the RELATIONS only (no piece types)")
    vp = sorted({f for c in pc for f in c}); vc = sorted({f for c in cc for f in c})
    Xp = normalize(np.log1p(np.array([[c.get(f, 0) for f in vp] for c in pc], dtype=float)))
    Xc = normalize(np.log1p(np.array([[c.get(f, 0) for f in vc] for c in cc], dtype=float)))
    themes = [t for t in THEMES if sum(t in r["themes"] for r in recs) >= 15]
    Y = {t: np.array([t in r["themes"] for r in recs]) for t in themes}
    rng = np.random.default_rng(0)

    def best_f1(lab, t):
        y = Y[t]; best = (0.0, -1, 0.0, 0.0)
        for g in set(lab):
            m = lab == g; tp = (m & y).sum()
            if tp == 0:
                continue
            prec = tp / m.sum(); rec = tp / y.sum(); f1 = 2 * prec * rec / (prec + rec)
            best = max(best, (f1, g, prec, rec))
        return best
    for k in ks:
        lp = KMeans(k, n_init=10, random_state=0).fit_predict(Xp)
        lc = KMeans(k, n_init=10, random_state=0).fit_predict(Xc)
        lr = rng.permutation(lp)
        log(f"OBS k={k}: per theme, the best-matching group's F1 -- preventer groups / control groups / random groups (the preventer groups' sizes)")
        log("   " + "; ".join(f"{t} {best_f1(lp, t)[0]:.2f} / {best_f1(lc, t)[0]:.2f} / {best_f1(lr, t)[0]:.2f}" for t in themes))
        if k == detail_k:
            Xraw = np.log1p(np.array([[c.get(f, 0) for f in vp] for c in pc], dtype=float))
            overall = Xraw.mean(0) + 1e-9
            for g in sorted(set(lp), key=lambda g: -(lp == g).sum()):
                m = lp == g
                lift = Xraw[m].mean(0) / overall
                share = (Xraw[m] > 0).mean(0)
                top = [i for i in np.argsort(-lift) if share[i] >= 0.4][:4]
                th = sorted(((Y[t][m].mean() / max(Y[t].mean(), 1e-9), t, Y[t][m].mean()) for t in themes), reverse=True)[:3]
                ex = next(r["pid"] for r, x in zip(recs, m) if x)
                log(f"   group {g:2d} ({int(m.sum()):4d}): prevented by " + ", ".join(f"{vp[i]} ({share[i]:.2f})" for i in top)
                    + " | themes " + ", ".join(f"{t} {sh:.2f} (x{l:.1f})" for l, t, sh in th) + f" | e.g. {ex}")


def main(argv=None):
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--n", type=int, default=100); c.add_argument("--offset", type=int, default=200)
    c.add_argument("--workers", type=int, default=10)
    r = sub.add_parser("report"); r.add_argument("--path", default=None); r.add_argument("--n", type=int, default=100); r.add_argument("--offset", type=int, default=200)
    g = sub.add_parser("groups"); g.add_argument("--path", default=None); g.add_argument("--n", type=int, default=3000); g.add_argument("--offset", type=int, default=200)
    g.add_argument("--reasons-only", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "collect":
            report(collect(a.n, a.offset, a.workers))
        elif a.cmd == "groups":
            groups(a.path or str(paths.derived(f"tactic_reasons_{a.n}_{a.offset}.jsonl")), reasons_only=a.reasons_only)
        else:
            report(a.path or str(paths.derived(f"tactic_reasons_{a.n}_{a.offset}.jsonl")))
    finally:
        for e in _ENG.values():
            e.quit()


if __name__ == "__main__":
    main()
