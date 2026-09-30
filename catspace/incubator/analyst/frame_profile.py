"""FRAMES AS THE CONCEPTS (Kaveh 2026-09-29: "one set of concepts could be all the frames that are present or important
right now ... is there a future in which there is a checkmate frame like this, or similar to this? ... the frames are
features; we track the trajectory based on those features"). His call: measure first (do frame profiles predict the
frames that come, better than material and the raw board?), and "a frame like this" = the same RELATIONAL TYPE,
wherever it stands on the board.

THE PROFILE of a position: every frame of both sides' frame books (the side to move's; the other side's as if we
passed), each with its TYPE -- fates.lemma_key(strict=False): the relations of its pieces with the squares abstracted
away, the joined kings, a check, a pawn near promotion and a capture that gives check as part of the description --
its owner and its resolved value. A MATE FRAME: a frame whose resolution mates (value >= 500).

    python -m catspace.incubator.analyst.frame_profile collect --games 60           # profiles along real games
    python -m catspace.incubator.analyst.frame_profile gauge                        # vocabulary, persistence, prediction
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time

import chess

from catspace.io import paths

from . import fates as F

MATE_AT = 500.0


_KIND = " pnbrqk"


def type_ids(b: chess.Board, squares, kings, members) -> tuple:
    """-> (the exact relational type, a COARSE type: which pieces are in the frame -- kind, side, target or not, pinned,
    near promotion -- and the flags, without the pairwise relations, the frame's RELATION ATOMS)."""
    k = F.lemma_key(b, squares, kings, members, strict=False)
    if k is None:
        return "big", "big", ["big"]
    h = lambda x: hashlib.sha1(repr(x).encode()).hexdigest()[:12]
    return h(k[0]), h((tuple(sorted(k[0][1])), k[0][3:])), atoms(k[0])


def atoms(key: tuple) -> list:
    """THE FRAME'S PARTS (Kaveh 2026-09-29: frames decomposed into their relations, which recur where whole frames do
    not): for every ordered pair of its pieces that bear on each other -- attacker side and kind > target side and kind
    (sides relative to the frame's mover: m, o), direct or x-ray, the capture legal, giving check, the target's
    defenders inside the frame, the target the frame's target square, pinned -- and the frame's flags (a king joined,
    in check, en passant, a pawn near promotion, a capturer from outside)."""
    _, info, rels = key[0], key[1], key[2]
    flags = key[3:]
    n = len(info)
    rel = {}
    it = iter(rels)
    for x in range(n):
        for y in range(n):
            if x != y:
                rel[(x, y)] = next(it)
    side = lambda s: "m" if info[s][0] else "o"
    out = []
    for (x, y), r in rel.items():
        att, xr, legal = r[0], r[1], r[2]
        chk = r[4] if len(r) > 4 else False
        if not att and xr < 0:
            continue
        defenders = sum(1 for z in range(n) if z != y and info[z][0] == info[y][0] and (rel[(z, y)][0] or rel[(z, y)][1] >= 0))
        out.append(f"{side(x)}{_KIND[info[x][1]]}>{side(y)}{_KIND[info[y][1]]}|{'a' if att else 'x' + str(xr)}"
                   f"|{'L' if legal else ''}{'+' if chk else ''}|d{min(defenders, 3)}|{'T' if info[y][2] else ''}{'p' if info[y][3] else ''}")
    kings = flags[0] if flags else ()
    out.append(f"frame|k{''.join('m' if c else 'o' for c in kings)}|{'chk' if len(flags) > 1 and flags[1] else ''}"
               f"|{'promo' if len(flags) > 3 and flags[3] else ''}|{'out' if len(flags) > 4 and flags[4] else ''}|n{min(n, 8)}")
    return out


def profile(b: chess.Board) -> list:
    """-> [(owner colour, kind 'sq'|'king', exact type id, value, coarse type id, relation atoms)] for both sides' frame books."""
    out = []
    boards = [b]
    pb = F.passed(b)
    if pb is not None:
        boards.append(pb)
    for t in boards:
        for key, (fr, _, v, _ln) in F.frame_book(t).items():
            fine, coarse, ats = type_ids(t, fr[0], fr[1], fr[2])
            out.append((t.turn, key[0], fine, float(v), coarse, ats))
    return out


def _game_rows(args) -> list:
    gid, pgn_text = args
    import chess.pgn
    F.clear_caches()
    g = chess.pgn.read_game(io.StringIO(pgn_text))
    rows = []
    if g is None:
        return rows
    b = g.board()
    for ply, mv in enumerate(g.mainline_moves()):
        b.push(mv)
        if b.is_game_over():
            break
        try:
            prof = profile(b)
        except Exception:
            continue
        bag = {}
        for o, k, tid, v, ct, ats in prof:
            for a in ats:
                bag[(int(o), f"{k}|{a}")] = bag.get((int(o), f"{k}|{a}"), 0) + 1
        rows.append(dict(game=gid, ply=ply + 1, fen=b.fen(), frames=[[int(o), k, tid, v, ct] for o, k, tid, v, ct, _ in prof],
                         atoms=[[o, a, c] for (o, a), c in bag.items()]))
        if ply % 10 == 0:
            F.clear_caches()
    return rows


def collect(games: int = 60, skip: int = 0, workers: int = 10, log=print) -> str:
    """Profiles of every position of `games` games of lichess_2015-06 (after `skip`), in parallel over games."""
    import multiprocessing as mp
    import zstandard
    t0 = time.perf_counter()
    texts = []
    with open(paths.data_dir() / "external" / "lichess" / "lichess_2015-06.pgn.zst", "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        buf = []; n = 0
        for line in text:
            if line.startswith("[Event ") and buf:
                if n >= skip:
                    texts.append((n, "".join(buf)))
                n += 1; buf = []
                if len(texts) >= games:
                    break
            buf.append(line)
    out = paths.derived(f"frame_profiles_{games}_{skip}.jsonl.gz")
    with mp.get_context("fork").Pool(workers) as pool, gzip.open(out, "wt") as fh:
        npos = 0
        for rows in pool.imap_unordered(_game_rows, texts, chunksize=2):
            for r in rows:
                fh.write(json.dumps(r) + "\n"); npos += 1
    log(f"[frame_profile] {len(texts)} games, {npos} positions -> {out} ({time.perf_counter() - t0:.0f}s)")
    return str(out)


def _load(path: str) -> list:
    with gzip.open(path, "rt") as fh:
        return [json.loads(l) for l in fh]


def gauge(path: str, ks=(4, 8), log=print) -> None:
    """(1) the vocabulary of frame types; (2) persistence: the share of a position's (owner, type) frames present again
    L plies later (L = 1 and 2 -- the side to move flips at odd lags, so both are shown); (3) does the profile PREDICT a
    NEW mate frame (or a frame winning 3+) of the side to move within k plies -- one it does not have now -- against
    material and the raw board (logistic regression, held out by game)."""
    import numpy as np
    from collections import Counter
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.sparse import hstack, csr_matrix
    rows = _load(path)
    rows.sort(key=lambda r: (r["game"], r["ply"]))
    n = len(rows)
    log(f"[frame_profile] {n} positions from {len({r['game'] for r in rows})} games; {sum(len(r['frames']) for r in rows)} frames")
    by_game = {}
    for i, r in enumerate(rows):
        by_game.setdefault(r["game"], []).append(i)
    for label, col in (("exact", 2), ("coarse", 4)):
        tc = Counter(f[col] for r in rows for f in r["frames"])
        tot = sum(tc.values())
        top = [c for _, c in tc.most_common()]
        log(f"OBS {label} type vocabulary: {len(tc)} distinct types; the top 50 cover {sum(top[:50]) / tot:.3f} of frames, the top 500 "
            f"{sum(top[:500]) / tot:.3f}; types seen once {sum(1 for c in top if c == 1)} ({sum(1 for c in top if c == 1) / len(tc):.3f} of types)")
        sets = [set((f[0], f[col]) for f in r["frames"]) for r in rows]
        for L in (1, 2):
            vals = []
            for idx in by_game.values():
                for a, bi in zip(idx, idx[L:]):
                    if sets[a]:
                        vals.append(len(sets[a] & sets[bi]) / len(sets[a]))
            log(f"OBS {label} type persistence at lag {L}: a position's (owner, type) frames present {L} ply later: mean {np.mean(vals):.3f} over {len(vals)} positions")
    ac = Counter(a[1] for r in rows for a in r["atoms"])
    atot = sum(ac.values()); atop = [c for _, c in ac.most_common()]
    log(f"OBS relation atoms: {len(ac)} distinct; the top 50 cover {sum(atop[:50]) / atot:.3f} of atom occurrences, the top 500 "
        f"{sum(atop[:500]) / atot:.3f}; atoms seen once {sum(1 for c in atop if c == 1)} ({sum(1 for c in atop if c == 1) / len(ac):.3f})")
    asets = [set((a[0], a[1]) for a in r["atoms"]) for r in rows]
    for L in (1, 2):
        vals = []
        for idx in by_game.values():
            for a, bi in zip(idx, idx[L:]):
                if asets[a]:
                    vals.append(len(asets[a] & asets[bi]) / len(asets[a]))
        log(f"OBS atom persistence at lag {L}: a position's (owner, atom) present {L} ply later: mean {np.mean(vals):.3f}")
    # targets: a NEW frame of the side to move within k plies
    def owned(r, colour, pred):
        return any(f[0] == colour and pred(f[3]) for f in r["frames"])
    is_mate = lambda v: v >= MATE_AT
    is_gain = lambda v: 3.0 <= v < MATE_AT
    feats_frames, feats_mat, feats_board, feats_atoms, groups = [], [], [], [], []
    for r in rows:
        b = chess.Board(r["fen"]); me = b.turn
        feats_frames.append([f"{'us' if f[0] == me else 'them'}:{f[1]}:{f[2]}" for f in r["frames"]]
                            + [f"{'us' if f[0] == me else 'them'}:{f[1]}:c{f[4]}" for f in r["frames"]]
                            + [f"{'us' if f[0] == me else 'them'}:v{min(9, int(max(f[3], -9)))}" for f in r["frames"] if f[3] < MATE_AT]
                            + [f"{'us' if f[0] == me else 'them'}:mate" for f in r["frames"] if f[3] >= MATE_AT])
        fa = {}
        for o, a, c in r["atoms"]:
            fa[f"{'us' if o == me else 'them'}:{a}"] = float(c)
        feats_atoms.append(fa)
        mat = []
        for c in (me, not me):
            for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                mat.append(len(b.pieces(pt, c)))
        feats_mat.append(mat + [sum(mat[:5][i] * w for i, w in enumerate((1, 3, 3, 5, 9))) - sum(mat[5:][i] * w for i, w in enumerate((1, 3, 3, 5, 9)))])
        oh = []
        for s, p in b.piece_map().items():
            ss = s if me == chess.WHITE else chess.square_mirror(s)
            oh.append(f"{'us' if p.color == me else 'them'}{p.piece_type}@{ss}")
        feats_board.append(oh)
        groups.append(int(hashlib.sha1(str(r["game"]).encode()).hexdigest()[:8], 16) % 5 == 0)
    hasher = FeatureHasher(n_features=2 ** 18, input_type="string")
    Xf = hasher.transform(feats_frames); Xb = hasher.transform(feats_board)
    Xm = csr_matrix(np.array(feats_mat, dtype=float) / 10.0)
    from sklearn.feature_extraction import DictVectorizer
    Xa = DictVectorizer().fit_transform(feats_atoms)
    Xa.data = np.log1p(Xa.data)
    test = np.array(groups); train = ~test
    for name, pred in (("mate frame", is_mate), ("frame winning 3+", is_gain)):
        for k in ks:
            y = np.zeros(n, dtype=bool); keep = np.zeros(n, dtype=bool)
            for idx in by_game.values():
                for j, i in enumerate(idx):
                    me = chess.Board(rows[i]["fen"]).turn
                    if owned(rows[i], me, pred) or j + 1 >= len(idx):
                        continue                                  # it has one now, or the game ends here
                    keep[i] = True
                    y[i] = any(owned(rows[x], me, pred) for x in idx[j + 1:j + 1 + k])
            tr, te = train & keep, test & keep
            if y[te].sum() == 0 or y[tr].sum() == 0:
                log(f"OBS {name} within {k}: too few positives (train {int(y[tr].sum())}, test {int(y[te].sum())})")
                continue
            res = []
            from sklearn.preprocessing import MaxAbsScaler
            gv = np.array([int(hashlib.sha1(("val" + str(r["game"])).encode()).hexdigest()[:8], 16) % 4 == 0 for r in rows])
            fit, val = tr & ~gv, tr & gv                              # C chosen on held-in games (bug 2026-09-29: one fixed C for
            for fname, X in (("material", Xm), ("raw board", Xb), ("frames", Xf), ("frames+material", hstack([Xf, Xm]).tocsr()),
                             ("board+material", hstack([Xb, Xm]).tocsr()), ("board+frames+material", hstack([Xb, Xf, Xm]).tocsr()),
                             ("atoms", Xa), ("atoms+material", hstack([Xa, Xm]).tocsr()), ("board+atoms+material", hstack([Xb, Xa, Xm]).tocsr())):
                X = MaxAbsScaler().fit(X[tr]).transform(X)
                bestC, bestA = None, -1.0
                for C in (0.01, 0.03, 0.1, 0.3, 1.0):
                    mm = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(X[fit], y[fit])
                    av = roc_auc_score(y[val], mm.predict_proba(X[val])[:, 1])
                    if av > bestA:
                        bestC, bestA = C, av
                m = LogisticRegression(C=bestC, max_iter=3000, solver="liblinear").fit(X[tr], y[tr])
                res.append(f"{fname} {roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]):.3f} (C {bestC:g})")
            log(f"OBS a NEW {name} of the side to move within {k} plies: base rate {y[te].mean():.3f} (test {int(te.sum())} positions, "
                f"{int(y[te].sum())} positive); held-out AUC: " + ", ".join(res))


def latent_lines(b: chess.Board, by: chess.Color) -> list:
    """`by`'s sliders in line with the other king with 1-3 pieces between (Kaveh: "a king and the opponent's rook on the
    same file ... a few pieces in between ... suddenly three pieces are gone and I find myself in check"):
    -> [(slider square, blocker squares)]."""
    k = b.king(not by)
    out = []
    if k is None:
        return out
    for s in b.pieces(chess.BISHOP, by) | b.pieces(chess.ROOK, by) | b.pieces(chess.QUEEN, by):
        if not chess.BB_RAYS[s][k]:
            continue
        orth = chess.square_file(s) == chess.square_file(k) or chess.square_rank(s) == chess.square_rank(k)
        pt = b.piece_type_at(s)
        if (pt == chess.ROOK and not orth) or (pt == chess.BISHOP and orth):
            continue
        between = [x for x in chess.SquareSet(chess.between(s, k)) if b.piece_at(x) is not None]
        if 1 <= len(between) <= 3:
            out.append((s, between))
    return out


def mate_misses(path: str, k: int = 4, show: int = 6, log=print) -> None:
    """Kaveh: "find the situations where we cannot predict mate from the frames". On held-out games, the positions
    where a NEW mate frame of the side to move appears within k plies: the atoms+material model's scores; the quarter
    it scores lowest (MISSED) against the quarter it scores highest (CAUGHT), and what was there at the start -- the
    mated king's frame in the profile and its value, the mating side's latent lines on that king (sliders in line
    with 1-3 pieces between), its pieces within two king steps, the king's flight squares -- and how the mate came:
    the first check of the mate frame's line, and whether it is a slider whose line to the king was blocked at the
    start (a latent line that opened)."""
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler
    from scipy.sparse import hstack, csr_matrix
    rows = _load(path); rows.sort(key=lambda r: (r["game"], r["ply"]))
    by_game = {}
    for i, r in enumerate(rows):
        by_game.setdefault(r["game"], []).append(i)
    n = len(rows)
    y = np.zeros(n, dtype=bool); keep = np.zeros(n, dtype=bool); when = {}
    turn = [chess.Board(r["fen"]).turn for r in rows]
    owned = lambda i, c: any(f[0] == c and f[3] >= MATE_AT for f in rows[i]["frames"])
    for idx in by_game.values():
        for j, i in enumerate(idx):
            if owned(i, turn[i]) or j + 1 >= len(idx):
                continue
            keep[i] = True
            for x in idx[j + 1:j + 1 + k]:
                if owned(x, turn[i]):
                    y[i] = True; when[i] = x; break
    fa, fm = [], []
    for i, r in enumerate(rows):
        me = turn[i]; b = chess.Board(r["fen"])
        fa.append({f"{'us' if o == me else 'them'}:{a}": float(c) for o, a, c in r["atoms"]})
        m = [len(b.pieces(pt, c)) for c in (me, not me) for pt in (1, 2, 3, 4, 5)]
        fm.append(m)
    Xa = DictVectorizer().fit_transform(fa); Xa.data = np.log1p(Xa.data)
    X = hstack([Xa, csr_matrix(np.array(fm, dtype=float) / 10.0)]).tocsr()
    test = np.array([int(hashlib.sha1(str(r["game"]).encode()).hexdigest()[:8], 16) % 5 == 0 for r in rows])
    tr, te = ~test & keep, test & keep
    X = MaxAbsScaler().fit(X[tr]).transform(X)
    model = LogisticRegression(C=0.1, max_iter=3000, solver="liblinear").fit(X[tr], y[tr])
    pos = np.where(te & y)[0]
    sc = model.predict_proba(X[pos])[:, 1]
    order = pos[np.argsort(sc)]
    q = max(1, len(order) // 4)
    groups = {"MISSED (lowest quarter)": order[:q], "CAUGHT (highest quarter)": order[-q:]}
    log(f"[mate_misses] held-out positions where a new mate frame of the side to move appears within {k} plies: {len(pos)}")
    for name, ids in groups.items():
        stats = dict(king_frame=0, king_value_pos=0, latent=0, latent_opened=0, near=0, flights=0, check_first=0, n=0)
        ex = []
        for i in ids:
            b = chess.Board(rows[i]["fen"]); me = turn[i]; kk = b.king(not me)
            stats["n"] += 1
            kf = [f for f in rows[i]["frames"] if f[0] == me and f[1] == "king"]
            stats["king_frame"] += bool(kf); stats["king_value_pos"] += any(f[3] > 0 for f in kf)
            lat = latent_lines(b, me)
            stats["latent"] += bool(lat)
            stats["near"] += sum(1 for s, p in b.piece_map().items() if p.color == me and chess.square_distance(s, kk) <= 2)
            stats["flights"] += sum(1 for s in chess.SquareSet(chess.BB_KING_ATTACKS[kk]) if not b.is_attacked_by(me, s) and (b.piece_at(s) is None or b.piece_at(s).color == me))
            later = chess.Board(rows[when[i]]["fen"])
            t2 = later if later.turn == me else F.passed(later)
            line = []
            if t2 is not None:
                book = F.frame_book(t2)
                mf = [(v, ln) for key, (_fr, _, v, ln) in book.items() if v >= MATE_AT]
                if mf:
                    line = max(mf)[1]
            first_check = None
            if line and t2 is not None:
                bb = t2.copy()
                for u in line:
                    mv = chess.Move.from_uci(u)
                    if bb.turn == me and mv in bb.legal_moves and bb.gives_check(mv):
                        first_check = (bb.piece_type_at(mv.from_square), mv.from_square); break
                    if mv not in bb.legal_moves:
                        break
                    bb.push(mv)
            if first_check:
                stats["check_first"] += 1
                if first_check[0] in (chess.BISHOP, chess.ROOK, chess.QUEEN) and any(s == first_check[1] for s, _ in lat):
                    stats["latent_opened"] += 1
                # HOW THE MATE CAME: follow the checking piece back to the start position through the game's moves; did it
                # have to ARRIVE (piece reach from where it stood), and did its check line have to OPEN (pieces on it at
                # the start, gone at the check)
                path_idx = list(range(i, when[i]))
                sq = first_check[1]
                for x in reversed(path_idx):                      # the game's moves between the start and the mate frame
                    b0, b1 = chess.Board(rows[x]["fen"]), chess.Board(rows[x + 1]["fen"])
                    mv = next((mm for mm in b0.legal_moves if (lambda c: (c.push(mm), c.board_fen() == b1.board_fen())[1])(b0.copy(stack=False))), None)
                    if mv is not None and mv.to_square == sq:
                        sq = mv.from_square
                start_sq = sq
                pc0 = b.piece_at(start_sq)
                chk_move = next((chess.Move.from_uci(u) for u in line if chess.Move.from_uci(u).from_square == first_check[1]), None)
                arrived = opened = None
                if pc0 is not None and pc0.color == me and chk_move is not None:
                    target_sq = chk_move.to_square
                    from .piece_reach import piece_reach
                    rc = piece_reach(b, start_sq, k=4)
                    if b.piece_at(target_sq) is None:                 # moves to stand on the checking square; a capture there:
                        arrive_moves = rc.stand.get(target_sq)        # moves to attack it, plus the capture (bug 2026-09-29:
                    else:                                             # 'stand' holds empty squares only, so capture checks read
                        d_ = rc.attack.get(target_sq)                 # 'not reachable')
                        arrive_moves = None if d_ is None else d_ + 1
                    arrived = arrive_moves is None or arrive_moves >= 2   # (bug 2026-09-29: every check moves its piece -- 'arrived'
                    blockers = [x for x in chess.SquareSet(chess.between(target_sq, kk)) if b.piece_at(x) is not None and x != start_sq] if kk is not None else []
                    opened = bool(blockers) and first_check[0] in (chess.BISHOP, chess.ROOK, chess.QUEEN)   # means more than the check itself)
                    kind = ("arrived and line opened" if arrived and opened else "arrived" if arrived else "line opened" if opened
                            else "the check was one move away at the start")
                    stats.setdefault("how", {}).setdefault(kind, 0); stats["how"][kind] += 1
                    if arrived:
                        stats.setdefault("arrive", []).append(arrive_moves if arrive_moves is not None else 9)
            if len(ex) < show:
                ex.append(f"   {rows[i]['fen']}  -> mate frame at +{rows[when[i]]['ply'] - rows[i]['ply']} ply: {' '.join(line[:6])}; "
                          f"latent lines {[(chess.square_name(s), [chess.square_name(x) for x in bl]) for s, bl in lat]}")
        m = max(stats['n'], 1)
        log(f"OBS {name} ({stats['n']} positions): the mated king's frame in the profile {stats['king_frame']} (its value > 0: {stats['king_value_pos']}); "
            f"a latent line on that king {stats['latent']}; the mate's first check comes down a latent line {stats['latent_opened']} "
            f"(of {stats['check_first']} with a check read); mating side's pieces within 2 of the king {stats['near'] / m:.2f} a position; "
            f"the king's flight squares {stats['flights'] / m:.2f}")
        how = stats.get("how", {}); arr = stats.get("arrive", [])
        log(f"   how the mate's first check came: " + ", ".join(f"{k2} {v}" for k2, v in sorted(how.items(), key=lambda kv: -kv[1]))
            + (f"; arriving pieces needed (piece reach from the start) 1: {arr.count(1)}, 2: {arr.count(2)}, 3+: {sum(1 for a in arr if 3 <= a < 9)}, "
               f"not within 4 own moves: {arr.count(9)}" if arr else ""))
        for e in ex:
            log(e)


def check(games: int = 40, skip: int = 5000, log=print) -> None:
    """Bug checks for the gauge's negative result (docs/BUGS.md classes): (1) silent skips -- positions whose profile
    raised; (2) 'big' frames that collapse to one type; (3) perspective -- a position and its colour mirror must give
    the same types with the owners swapped; (4) the target -- in games ending in checkmate, does the position before
    the mate show a mate frame for the side that mates; (5) the frame store in verify mode along game positions."""
    import chess.pgn
    import zstandard
    from collections import Counter
    texts = []
    with open(paths.data_dir() / "external" / "lichess" / "lichess_2015-06.pgn.zst", "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        n = 0
        while len(texts) < games:
            g = chess.pgn.read_game(text)
            if g is None:
                break
            n += 1
            if n > skip:
                texts.append(g)
    old = F.VERIFY_FRAMES
    F.VERIFY_FRAMES = True
    errors = Counter(); positions = frames = big = 0; mirror_bad = mirror_n = 0; mates = mate_seen = 0; leaks = 0
    try:
        for g in texts:
            F.clear_caches()
            b = g.board(); mv = list(g.mainline_moves())
            for i, m in enumerate(mv):
                b.push(m)
                if b.is_game_over():
                    break
                positions += 1
                try:
                    prof = profile(b)
                except AssertionError:
                    leaks += 1; continue
                except Exception as e:
                    errors[type(e).__name__] += 1; continue
                frames += len(prof); big += sum(1 for f in prof if f[2] == "big")
                if positions % 7 == 0:
                    mb = b.mirror()
                    try:
                        pm = profile(mb)
                    except Exception:
                        pm = None
                    rel = lambda pr, bb: Counter((f[0] == bb.turn, f[1], f[2], round(f[3], 3)) for f in pr)
                    mirror_n += 1; mirror_bad += pm is None or rel(prof, b) != rel(pm, mb)
                if i == len(mv) - 2:
                    last = b.copy(); last.push(mv[-1])
                    if last.is_checkmate():
                        mates += 1
                        mate_seen += any(f[0] == b.turn and f[3] >= MATE_AT for f in prof)
    finally:
        F.VERIFY_FRAMES = old
    log(f"[frame_profile check] {len(texts)} games, {positions} positions, {frames} frames")
    log(f"OBS check 1, silent skips: {sum(errors.values())} positions raised ({dict(errors)}); frame-store leaks in verify mode {leaks}")
    log(f"OBS check 2, 'big' frames (over 16 pieces, one shared type): {big} of {frames} ({big / max(frames, 1):.3f})")
    log(f"OBS check 3, colour mirror gives different types or values: {mirror_bad} of {mirror_n} positions")
    log(f"OBS check 4, games ending in mate: {mates}; the position before the mate shows a mate frame for the mating side in {mate_seen}")


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--games", type=int, default=60); c.add_argument("--skip", type=int, default=0)
    c.add_argument("--workers", type=int, default=10)
    g = sub.add_parser("gauge"); g.add_argument("--path", default=None); g.add_argument("--games", type=int, default=60); g.add_argument("--skip", type=int, default=0)
    k = sub.add_parser("check"); k.add_argument("--games", type=int, default=40); k.add_argument("--skip", type=int, default=5000)
    mm = sub.add_parser("mate_misses"); mm.add_argument("--games", type=int, default=1000); mm.add_argument("--k", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "mate_misses":
        return mate_misses(str(paths.derived(f"frame_profiles_{a.games}_0.jsonl.gz")), a.k)
    if a.cmd == "check":
        return check(a.games, a.skip)
    if a.cmd == "collect":
        collect(a.games, a.skip, a.workers)
    else:
        gauge(a.path or str(paths.derived(f"frame_profiles_{a.games}_{a.skip}.jsonl.gz")))


if __name__ == "__main__":
    main()
