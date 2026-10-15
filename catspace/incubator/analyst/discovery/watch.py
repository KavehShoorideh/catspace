"""LEARNING BY WATCHING, THEN BY PLAYING (Kaveh 2026-10-13: "We have to start learning by watching, then by playing").

Before the from-scratch run plays its first game it WATCHES a teacher play the material from the same curriculum starts:

  THE TEACHER   the tablebase's own play for five men or fewer (the winner: the fewest plies to the next capture or pawn
                move -- in these endings the mate; the loser: the longest resistance), Stockfish beyond
  LOOPS         the teacher's games read by the loop-frame proposer (patterns.derive_loops): arrangements seen again with
                their king shifted, checked by our own exact search, refined, held out
  SURPRISES     the positions where the teacher's move is proved good and our planner's own move proved bad
                (explorer.locate_watched) -- explored, separated, repaired, through every guard (held out, a second
                surprise, fresh games)

The teacher only shows where something exists; every lesson is checked by our own search before it is kept. Then the
rounds of play begin from what was learned (surprise_loop --watch N).
"""
from __future__ import annotations

import json
import os
import random
import time

import chess

from catspace.io import paths

TB = ENG = None


def _init():
    global TB, ENG
    import chess.engine
    import chess.syzygy
    TB = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    ENG.configure({"SyzygyPath": str(paths.syzygy_dir()), "Threads": 1})


def _tb_move(b: chess.Board) -> chess.Move:
    best = None
    for m in b.legal_moves:
        x = b.copy(stack=False); x.push(m)
        if x.is_checkmate():
            return m
        d = TB.probe_dtz(x)
        key = (abs(d) if TB.probe_wdl(x) == -2 else 10 ** 6) if b.turn == chess.WHITE else -d
        if best is None or key < best[0]:
            best = (key, m)
    return best[1]


def _teach(args) -> dict:
    """One teacher's game from a curriculum start: rows of the teacher's positions (the loop's format)."""
    import chess.engine
    from catspace.incubator.analyst.material_walk import _material, _start, engine_start
    seed, material, band = args
    men = len(material.replace("v", ""))
    b = _start(random.Random(seed), material, TB, band) if men <= 5 else engine_start(random.Random(seed), material, ENG)
    rows = []
    for _ in range(300):
        if b.is_game_over(claim_draw=True):
            break
        m = _tb_move(b) if men <= 5 and len(b.piece_map()) <= 5 else ENG.play(b, chess.engine.Limit(depth=14)).move
        if b.turn == chess.WHITE:
            rows.append(dict(fen=b.fen(), material=_material(b), san=b.san(m), why="the teacher"))
        b.push(m)
    return dict(seed=seed, out="win" if b.is_checkmate() and b.turn == chess.BLACK else "draw", moves=len(rows), rows=rows)


def teacher_games(material: str, n: int, band, seed0: int = 50000, workers: int | None = None) -> list:
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 8) - 2), initializer=_init) as p:
        return p.map(_teach, [(seed0 + i, material, band) for i in range(n)])


def _after(fen: str, u: str) -> str:
    """The position after our move, read from our side (the side to move set back to us -- a static reading)."""
    b = chess.Board(fen)
    b.push(chess.Move.from_uci(u))
    b.turn = not b.turn
    b.ep_square = None
    return b.fen()


def _lex(S, prefix: list):
    """Signs (rows x words) under the frame's earlier terms: a row the earlier terms already order keeps their sign (the
    frame compares vectors, the earlier terms first); only the rows they tie are left to the word."""
    import numpy as np
    c = np.zeros(len(S), dtype=S.dtype)
    for p in prefix:
        c = np.where(c != 0, c, S[:, p])
    return np.where(c[:, None] != 0, c[:, None], S)


def _rank_score(args):
    """RANKING at one watched position (learning heuristics from plans by ranking: Yoon, Fern & Givan 2008; Garrett,
    Kaelbling & Lozano-Perez 2016): for every word, the frame's earlier terms in front of it -- each other move of ours
    counts +1 when the vector puts it strictly above the teacher's choice (worse: the measure should fall), -1 when
    strictly below, 0 when tied; averaged over the other moves. A word that ties every move scores 0: it says nothing."""
    import numpy as np
    from . import explorer as E
    fen, t, prefix = args
    b = chess.Board(fen)
    moves = [m.uci() for m in b.legal_moves]
    if t not in moves or len(moves) < 2:
        return None
    M = E.matrix([_after(fen, u) for u in moves]).astype(np.int32)
    i = moves.index(t)
    S = _lex(np.sign(M - M[i]).astype(np.int8), prefix)
    return np.delete(S, i, axis=0).astype(np.float32).mean(axis=0)


def _falls(mats: list, prefix: list, k: int):
    """For every word, the frame's earlier terms in front of it: how often the vector is strictly lower within k of the
    teacher's own moves along its line (the last position, the mate, counts as fallen)."""
    import numpy as np
    ok, n = 0, 0
    for M in mats:
        for i in range(len(M)):
            ahead = M[i + 1: i + 1 + k]
            ok = ok + ((_lex(np.sign(ahead - M[i]).astype(np.int8), prefix) < 0).any(axis=0) if len(ahead) else 1)
            n += 1
    return ok / n


def _judge(args):
    """A candidate frame on the watched positions: certified (some move makes it fall within k against every reply) and
    imitated (following it, our planner chooses the teacher's move)."""
    import analyst_core as A
    ts, k, pos = args
    fens = [f for f, _ in pos]
    cert = A.verify_count(fens, ts[:-1], ts[-1], "append", k, "*") / len(fens)
    agree = sum((A.frame_choose(f, ts + [f"horizon:{k}"], None, "none", "*", []) or ("",))[0] == u for f, u in pos) / len(pos)
    return cert, agree


def trajectory_frame(games: list, material: str, ks=(1, 2, 3, 4), top: int = 16, most: int = 3, sample: int = 600,
                     log=print, pool=None, probe=()) -> list | None:
    """FRAMES FROM THE TEACHER'S TRAJECTORIES (Kaveh 2026-10-13: "track the frame across several moves"): every word on
    every watched position, scored two ways -- how often it is strictly lower within k of the teacher's own moves (the
    line), and how the teacher's choice ranks against every other move by it (the choice, ties counting nothing); the best
    CHECKED by our exact search (does it fall within k against EVERY reply, on the watched positions) and IMITATED
    (following it, does our planner choose the teacher's move). Then the next term the same way, scored only where the
    terms before it tie (a frame compares its terms in order), kept while the check and the imitation improve; up to
    `most` terms. -> [terms..., "horizon:k"], or None. `probe`: words to report beside the chosen ones."""
    import numpy as np
    from . import base_vocabulary as V
    from . import explorer as E
    lines = [[r for r in g["rows"] if r["material"] == material] for g in games if g["out"] == "win"]
    lines = [l for l in lines if len(l) >= 2]
    if not lines:
        return None
    pos = [(r["fen"], chess.Board(r["fen"]).parse_san(r["san"]).uci()) for l in lines for r in l]
    jpos = pos if len(pos) <= sample else random.Random(0).sample(pos, sample)
    T = E.terms()
    idx = {t: j for j, t in enumerate(T)}
    size = lambda j: V.size(V.parse(T[j]))
    allow = E.layer_mask(0, set())                     # (no named square: a frame for anywhere)
    mats = [E.matrix([r["fen"] for r in l]).astype(np.int32) for l in lines]
    mp = (lambda f, xs: pool.imap(f, xs, chunksize=8)) if pool else map
    frame, k, have = [], None, -1.0
    for _ in range(most):
        pre = [idx[t] for t in frame]
        tot, n = 0, 0
        for x in mp(_rank_score, [(f, u, pre) for f, u in pos]):
            if x is not None:
                tot, n = tot + x, n + 1
        rank = tot / n
        tried, probes = [], []
        for kk in (ks if not frame else (k,)):
            fall = _falls(mats, pre, kk)
            score = np.where(allow, fall + rank, -9.0)
            score[pre] = -9.0
            cands = sorted(range(len(T)), key=lambda j: (-score[j], size(j)))[:top]
            tried += [(kk, j, fall[j], rank[j]) for j in cands]
            for t in (probe if not frame else ()):
                if t in idx:
                    j = idx[t]
                    log(f"OBS   probe {t} at horizon {kk}: falls {fall[j]:.2f}, ranks {rank[j]:+.2f}, place "
                        f"{int((score > score[j]).sum()) + 1} of {int(allow.sum())}")
                    probes.append((kk, j))
        res = list(mp(_judge, [(frame + [T[j]], kk, jpos) for kk, j, _, _ in tried] + [([T[j]], kk, jpos) for kk, j in probes]))
        for (kk, j), (c, a) in zip(probes, res[len(tried):]):
            log(f"OBS   probe {T[j]} at horizon {kk}: certified {c:.2f}, our planner chooses the teacher's move {a:.2f}")
        (kk, j, f, r), (c, a) = max(zip(tried, res), key=lambda x: (x[1][0] + x[1][1], -size(x[0][1])))
        if c + a <= have:
            log(f"OBS   term {len(frame) + 1}: nothing improves the check and the imitation ({have:.2f}) -- the frame stops")
            break
        frame.append(T[j]); k, have = kk, c + a
        log(f"OBS   term {len(frame)}: {T[j]} = {V.gloss(V.parse(T[j]))[:110]}; horizon {k} -- falls along the lines "
            f"{f:.2f}, ranks the teacher's choice {r:+.2f}, certified {c:.2f}, our planner chooses the teacher's move {a:.2f}")
    log(f"OBS WATCHING: the frame from {len(lines)} watched lines ({len(pos)} positions; checked on {len(jpos)}): "
        f"{frame}, horizon {k}")
    return frame + [f"horizon:{k}"]


def disagreements(games: list, frames: dict, back: int = 3, log=print) -> list:
    """Where our planner, with the frames, would leave the teacher's line for something worse (the tablebase's distance for
    five men or fewer) -- each with the positions before it on the line (the context): -> the drill's starts."""
    import chess.syzygy
    from . import explorer as E
    E._install(frames)
    out = []
    with chess.syzygy.open_tablebase(str(paths.syzygy_dir())) as tb:
        for g in games:
            rows = g["rows"]
            for i, r in enumerate(rows):
                b = chess.Board(r["fen"])
                if len(b.piece_map()) > 5:
                    continue
                ours = E.planner_move(r["fen"])[0]
                t = b.parse_san(r["san"]).uci()
                if not ours or ours == t:
                    continue
                dist = lambda u: (lambda x: (abs(tb.probe_dtz(x)) if tb.probe_wdl(x) == -2 else 10 ** 6) if not x.is_checkmate() else 0)(
                    (lambda y: (y.push(chess.Move.from_uci(u)), y)[1])(b.copy()))
                if dist(ours) > dist(t) + 1:
                    out += [x["fen"] for x in rows[max(0, i - back): i + 1]]
                    break                                 # (the first a game: the rest of its line follows from it)
    fens = list(dict.fromkeys(out))
    log(f"OBS WATCHING: {len(fens)} positions where our planner would leave the teacher's line for worse, with the {back} "
        f"before each")
    return fens


def watch(material: str, n: int, band, frames: dict, tag: str, surprises: int = 12, utility_n: int = 200, log=print,
          drill_rounds: int = 4) -> dict:
    """THE WATCHING PHASE -> the frames to start playing with."""
    from . import explorer as E
    from . import patterns as Pt
    from . import scratch
    t0 = time.time()
    games = teacher_games(material, n, band)
    path = paths.experiment(f"watch_{material}_{tag}.json")
    json.dump(games, open(path, "w"))
    won = [g for g in games if g["out"] == "win"]
    log(f"OBS WATCHING: the teacher played {len(games)} games of {material} from the curriculum's starts (band {band}): mated "
        f"{len(won)}, {sum(g['moves'] for g in won) / max(len(won), 1):.1f} moves a mate ({time.time() - t0:.0f} s)")
    mats = {r["material"] for g in games for r in g["rows"]}
    frames = dict(frames)
    # 1 THE TRAJECTORIES: a frame for each material the teacher played, read off its lines
    for m in sorted(mats):
        tf = trajectory_frame(games, m, log=log)
        if tf and not frames.get(m):
            frames[m] = tf
    # 2 THE LOOPS in the teacher's games
    loops = Pt.derive_loops([str(path)], mats, log=log)
    frames.update(loops)
    log(f"OBS WATCHING: {len(loops)} loop frame(s) from the teacher's games")
    # 3 THE DRILL: where we would leave the teacher's line, with the moves before -- played until no longer surprised
    from .surprise_loop import drill
    fens = disagreements(games, frames, log=log)
    if fens:
        frames = drill(fens, frames, material, tag, iterations=drill_rounds, log=log)
    log(f"OBS WATCHING done ({time.time() - t0:.0f} s): the frames to play with {frames}")
    return frames


# ---------------------------------------------------------------------------------------------------------------------
# WATCHING FULL GAMES (Kaveh 2026-10-13: "we can try watching full games here. We don't need to play. We can just download
# the game and watch it. Basically, go through its moves in sequence"): the Lichess database already on disk
# (assets/corpora/raw/lichess_db_standard_rated_2019-01.pgn.zst, ~33 million rated games), streamed once; the games finished
# by mate that reach a target ending, kept from the moment they reach it, the winner the teacher -- mirrored when Black
# won, so the winner is White as in every game of the loop.
# ---------------------------------------------------------------------------------------------------------------------
LICHESS = "assets/corpora/raw/lichess_db_standard_rated_2019-01.pgn.zst"


def _key(b: chess.Board, c: bool) -> str:
    side = lambda s: "K" + "".join(chess.piece_symbol(p).upper() * len(b.pieces(p, s)) for p in (5, 4, 3, 2, 1))
    return side(c) + "v" + side(not c)


def _watch_one(args):
    """One game's text -> the winner's moves from the first target ending on (mirrored to White), or None."""
    import io
    import chess.pgn
    text, targets, idx = args
    g = chess.pgn.read_game(io.StringIO(text))
    if g is None:
        return None
    res = g.headers.get("Result")
    if res not in ("1-0", "0-1"):
        return None
    winner = chess.WHITE if res == "1-0" else chess.BLACK
    b = g.board()
    rows, started, key = [], False, None
    for m in g.mainline_moves():
        if not started and _key(b, winner) in targets:
            started, key = True, _key(b, winner)
        if started and b.turn == winner:
            w = b if winner == chess.WHITE else b.mirror()
            mm = m if winner == chess.WHITE else chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
            rows.append(dict(fen=w.fen(), material=_key(b, winner), san=w.san(mm), why="the teacher (Lichess)"))
        b.push(m)
    if not started or not b.is_checkmate() or len(rows) < 2:
        return None
    return dict(seed=idx, out="win", moves=len(rows), rows=rows, ending=key,
                elo=g.headers.get("WhiteElo" if winner == chess.WHITE else "BlackElo"),
                site=g.headers.get("Site"))


def extract(targets: set, limit_games: int | None = None, workers: int = 10, log=print) -> dict:
    """Stream the database once: {ending: [watched games]} -- the games finished by mate that reach a target ending."""
    import subprocess
    from multiprocessing import get_context
    t0 = time.time()
    proc = subprocess.Popen(["zstd", "-dc", LICHESS], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    out = {t: [] for t in targets}
    seen = mates = 0

    def games():
        nonlocal seen, mates
        buf = []
        for line in proc.stdout:
            if line.startswith("[Event ") and buf:
                seen += 1
                text = "".join(buf)
                if '[Termination "Normal"]' in text and "#" in text:          # finished by mate (or resignation: '#' rules it in)
                    mates += 1
                    yield (text, targets, seen)
                buf = []
                if limit_games and seen >= limit_games:
                    return
            buf.append(line)

    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_watch_one, games(), chunksize=256):
            if r:
                out[r["ending"]].append(r)
    proc.kill()
    log(f"OBS WATCHING LICHESS: {seen} games read, {mates} ended by mate, kept "
        + ", ".join(f"{t} {len(v)}" for t, v in out.items()) + f" ({time.time() - t0:.0f} s)")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------
# WATCHING FOR THE THREAT (Kaveh 2026-10-14: "I want it to learn what a threat is" -- by watching Stockfish, not by a rule we write). The one given
# notion is the PASS TEST: a threat against side S in position P (S to move, not in check) is a move m that the other side O could make if S
# passed, sized by what S then loses with S's best defence. pass(P) = the same board with O to move. The teacher labels every O move on pass(P).

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
THREAT_DEPTH = 10
WALK_PLIES = 8
MATE_SIZE = 100


def _bal(b: chess.Board, color: bool) -> int:
    """Material balance from `color`'s side (P/N/B/R/Q = 1/3/3/5/9)."""
    return sum(VAL.get(p.piece_type, 0) * (1 if p.color == color else -1) for p in b.piece_map().values())


def pass_board(fen: str):
    """pass(P): the same board, the other side to move; None when S is in check (no pass) or the board is then illegal."""
    b = chess.Board(fen)
    if b.is_check():
        return None
    q = b.copy(stack=False)
    q.turn = not q.turn
    q.ep_square = None
    return q if q.is_valid() else None


def walk_size(p0: chess.Board, m: chess.Move, pv: list, mate_for_o: bool, s: bool) -> int:
    """S's net material loss from P (before m) to the first quiet point of the line m, pv...: stop when ply >= 2 and the next move of the line is
    not a capture or promotion, or after WALK_PLIES; a mate against S on the line is MATE_SIZE."""
    b = p0.copy(stack=False)
    b.push(m)
    base = _bal(p0, s)
    if b.is_checkmate():
        return MATE_SIZE
    k = 0
    while k < len(pv) and k < WALK_PLIES:
        if k >= 2 and not (b.is_capture(pv[k]) or pv[k].promotion):
            break
        b.push(pv[k]); k += 1
        if b.is_checkmate():
            return MATE_SIZE if b.turn == s else base - _bal(b, s)
    if mate_for_o and k >= len(pv):
        return MATE_SIZE
    return base - _bal(b, s)


def threat_labels(fen: str, eng, depth: int = THREAT_DEPTH) -> dict | None:
    """The teacher's labels for one position: for every legal O move m on pass(P), the size of the threat (S's net loss with Stockfish defending)
    and the centipawn drop (a second field, for reporting only)."""
    import chess.engine
    pb = pass_board(fen)
    if pb is None:
        return None
    s = chess.Board(fen).turn
    lim = chess.engine.Limit(depth=depth)
    cp0 = eng.analyse(chess.Board(fen), lim)["score"].pov(s).score(mate_score=10000)
    out = []
    for m in pb.legal_moves:
        q = pb.copy(stack=False); q.push(m)
        if q.is_game_over():
            out.append(dict(uci=m.uci(), size=MATE_SIZE if q.is_checkmate() else 0, cp_drop=cp0 + 10000 if q.is_checkmate() else cp0))
            continue
        info = eng.analyse(q, lim)
        sc = info["score"].pov(s)
        size = walk_size(pb, m, info.get("pv", []), sc.is_mate() and sc.mate() < 0, s)
        out.append(dict(uci=m.uci(), size=size, cp_drop=cp0 - sc.score(mate_score=10000)))
    return dict(fen=fen, pass_fen=pb.fen(), cp=cp0, moves=out)


def threat_rows(offset: int, n: int) -> list:
    """Lichess puzzle rows i % 997 == offset, i % 200 != 0 (clear of dev 5, fresh 11, test stride 200, new sample 17 for held-out use)."""
    import csv
    import io
    import zstandard
    from .ledger_puzzles import _csv_path
    out = []
    with open(_csv_path(), "rb") as fh:
        rd = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(rd):
            if i % 997 == offset and i % 200 != 0:
                out.append(row)
                if len(out) >= n:
                    break
    return out


TRACE_FENS = ["5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R b - - 0 1", "r5k1/2p1q2p/2Qp2p1/1rpPpp2/2P1P3/P4P2/6PP/4R1K1 w - - 0 27",
              "1r2r3/1kp1q2p/4Q3/p2P4/1P1P4/P1p2P2/2P3PP/2KR3R b - - 0 30"]


def _key4(fen: str) -> str:
    return " ".join(fen.split()[:4])


def threat_positions(rows: list) -> list:
    """From each row two positions: the FEN as given (an ordinary game position) and the position after Moves[0]; S in check skipped."""
    out = []
    for r in rows:
        b = chess.Board(r["FEN"])
        mv = r["Moves"].split()
        for src, fen in (("given", b.fen()),):
            out.append((r["PuzzleId"], src, fen))
        b.push_uci(mv[0])
        out.append((r["PuzzleId"], "after", b.fen()))
    keep = [x for x in out if pass_board(x[2]) is not None]
    bad = {_key4(f) for f in TRACE_FENS} | {_key4(pass_board(f).fen()) for f in TRACE_FENS if pass_board(f)}
    assert not any(_key4(x[2]) in bad for x in keep), "a trace position is in the training or held-out set"
    return keep


def _label_init():
    global ENG
    import chess.engine
    ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    ENG.configure({"Threads": 1})


def _label_one(args):
    pid, src, fen = args
    r = threat_labels(fen, ENG)
    if r is not None:
        r.update(id=pid, source=src)
    return r


def label_set(name: str, offset: int, n: int, log=print) -> dict:
    import multiprocessing
    t0 = time.time()
    pos = threat_positions(threat_rows(offset, n))
    with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_label_init) as pool:
        recs = [r for r in pool.imap(_label_one, pos, chunksize=2) if r]
    os.makedirs(paths.derived("watch"), exist_ok=True)
    p = paths.derived(f"watch/threat_labels_{name}.json")
    json.dump(recs, open(p, "w"))
    sizes = [m["size"] for r in recs for m in r["moves"]]
    withthr = sum(any(m["size"] >= 1 for m in r["moves"]) for r in recs)
    hist = {}
    for z in sizes:
        k = "<=0" if z <= 0 else "1" if z == 1 else "2" if z == 2 else "3-4" if z <= 4 else "5-8" if z <= 8 else "9-99" if z < 100 else "mate"
        hist[k] = hist.get(k, 0) + 1
    log(f"OBS THREAT LABELS {name}: {len(recs)} positions (of {len(pos)} candidates), {len(sizes)} moves, positions with a threat >= 1: {withthr / max(1, len(recs)):.0%}, "
        f"size histogram {hist}, {time.time() - t0:.0f} s -> {p}")
    return dict(positions=len(recs), moves=len(sizes))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# LEARNING THE THREAT: the definition in our words. Rows = (P, m); every word of explorer.terms() is read on the board after m (S to move: the words
# read from the side to move, so S -- the same side whose position P is) and on P itself (S to move, the "before"); the null move ("S passed and
# nothing happened") is an alternative of size 0 read on P. Two families of words: the value after m ("raw") and its change from P ("delta"; 0 for the
# null). Strict pairwise ranking within each position, pairs whose sizes differ by >= 1 (the null among them): a word scores +1 when it orders the pair
# like the teacher, -1 reversed, 0 on a tie; both orientations (the sign is recorded). Ranked terms: the next term scored only where the earlier ones
# tie (the frame compares vectors); up to three; stop when the training score stops improving.

THREAT_TERMS = 3
MIN_GAIN = 0.01                  # a further term must raise the strict pairwise score by this share of the pairs


# THE THREAT LEARNER'S TERM SET (Kaveh 2026-10-14): a threat is a property of the BOARD. Words that read the game's history (the fifty-move clock, repetition,
# time) are out -- the clock resets on every capture, so it only marks "the move was a capture" -- and so are the forcing words (they ARE the pass test by our
# exact search: the checker, not a word to learn from). The base words each name ONE piece; "the most we lose on any piece" needs a PIECE-QUANTIFIER LIFT: for
# every word with one piece slot f(X), the max and the sum of f over our named pieces (P, P2 .. P5) and over theirs (O, O2 .. O5), as plain nested max / sum
# compositions of existing words (the evaluator and the Lean renderer unfold them unchanged).
HISTORY = ("halfmoves", "seen", "(ago ", "forcing_", "(delta", "(gained", "(lost")
OUR_PIECES = ("P", "P2", "P3", "P4", "P5")
THEIR_PIECES = ("O", "O2", "O3", "O4", "O5")
# (template with X, gloss of f(X), who it is for: "ours" / "theirs" / "both")
LIFT_TEMPLATES = [
    ("(see X)", "what the other side wins by the exchange on", "both"),
    ("(attackers X)", "how many of their pieces attack", "both"),
    ("(defenders X)", "how many of their pieces defend", "both"),
    ("(min (see X) 1)", "whether the exchange on it can be won (1 or 0):", "both"),
    ("(count (and (moves X) unhit))", "how many squares it can go to that the other side does not hit:", "both"),
    ("(eq (count (and (moves X) unhit)) 0)", "whether it has no square to go to that the other side does not hit (1 or 0):", "both"),
    ("(pinned_ours X)", "whether it is pinned to our king (1 or 0):", "ours"),
    ("(pinned_theirs X)", "whether it is pinned to their king (1 or 0):", "theirs"),
    ("(min (min (see X) 1) (eq (count (and (moves X) unhit)) 0))", "whether it can be won AND has no square to go to that the other side does not hit (1 or 0):", "both"),
    ("(targets_ours X)", "how many of their pieces it attacks:", "ours"),
    ("(targets_theirs X)", "how many of our pieces it attacks:", "theirs"),
]
_TT = None


def _nest(op: str, items: list) -> str:
    t = items[-1]
    for x in reversed(items[:-1]):
        t = f"({op} {x} {t})"
    return t


def lift_terms() -> tuple:
    """-> (the lifted terms, {term: gloss})"""
    out, gl = [], {}
    for tpl, g, who in LIFT_TEMPLATES:
        for side, names, word in (("ours", OUR_PIECES, "our"), ("theirs", THEIR_PIECES, "their")):
            if who not in ("both", side):
                continue
            fs = [tpl.replace("X", n) for n in names]
            for op, quant in (("max", "the most"), ("sum", "the total")):
                t = _nest(op, fs)
                out.append(t)
                gl[t] = f"{quant} over {word} pieces of ({g} the piece)"
    # THE BALANCE once each side has taken its biggest exchange, we first (signed, on an offset of 64: the vocabulary's numbers are unsigned, so the
    # difference is built with `if` / `diff` -- no new word): our material + the most we win by an exchange, against their material + the most they win
    for op, quant in (("max", "biggest"), ("sum", "all")):
        a = f"(sum material_us {_nest(op, [f'(see {n})' for n in THEIR_PIECES])})"
        b = f"(sum material_them {_nest(op, [f'(see {n})' for n in OUR_PIECES])})"
        t = f"(if (eq (max {a} {b}) {a}) (sum 64 (diff {a} {b})) (diff 64 (diff {a} {b})))"
        out.append(t)
        gl[t] = f"the material balance for us (on 64) once each side has taken {quant} of its exchanges, we first"
    return out, gl


def threat_terms() -> tuple:
    global _TT
    if _TT is None:
        from . import explorer as E
        base = [t for t in E.terms() if not any(h in t for h in HISTORY)]
        lifted, gl = lift_terms()
        _TT = (base + lifted, gl)
    return _TT


def term_gloss(t: str) -> str:
    from . import base_vocabulary as V
    gl = threat_terms()[1]
    return gl[t] if t in gl else V.gloss(V.parse(t))


def _tmatrix(fens: list, us_white: bool | None = None):
    """Every word on every board; `us_white`: "us" is NAMED (a true board read from one side's point of view while the other side is to move)."""
    import numpy as np
    import analyst_core as A
    terms = threat_terms()[0]
    vals = A.component_values(fens, terms) if us_white is None else A.component_values_us(fens, terms, us_white)
    return np.asarray(vals, dtype=np.int64).reshape(len(fens), len(terms))


def _after_fen(pass_fen: str, uci: str) -> str:
    b = chess.Board(pass_fen)
    b.push(chess.Move.from_uci(uci))
    return b.fen()


def _position_matrices(rec: dict):
    """-> (sizes (n+1,), raw (n+1 x T), delta (n+1 x T)); the last row is the null move."""
    import numpy as np
    fens = [_after_fen(rec["pass_fen"], m["uci"]) for m in rec["moves"]] + [rec["fen"]]
    M = _tmatrix(fens).astype(np.int32)
    base = M[-1]
    delta = M - base
    delta[-1] = 0
    # THE SIZE OF A THREAT IS max(0, S's net loss): in a puzzle position S often still has its own tactic, so most O moves have a negative net -- those are no
    # threats, all alike (0), tied with each other and with the null; no pair among non-threats is ever ranked
    sizes = np.maximum(0, np.array([m["size"] for m in rec["moves"]] + [0]))
    return sizes, M, delta


def _threat_chunk(args):
    """In a worker: the scores of every word on the pairs still tied under the prefix; the pairs the prefix already orders (correct, wrong)."""
    import numpy as np
    recs, prefix = args
    T = len(threat_terms()[0])
    raw_s = np.zeros(T, np.int64); del_s = np.zeros(T, np.int64)
    ok = bad = tied = 0
    for rec in recs:
        sizes, R, D = _position_matrices(rec)
        I, J = np.nonzero(sizes[:, None] - sizes[None, :] >= 1)
        if not len(I):
            continue
        undecided = np.ones(len(I), bool)
        for (col, sign, fam) in prefix:
            X = (R if fam == "raw" else D)[:, col] * sign
            d = np.sign(X[I] - X[J])
            ok += int(((d > 0) & undecided).sum()); bad += int(((d < 0) & undecided).sum())
            undecided &= d == 0
        tied += int(undecided.sum())
        if undecided.any():
            raw_s += np.sign(R[I[undecided]] - R[J[undecided]]).sum(axis=0)
            del_s += np.sign(D[I[undecided]] - D[J[undecided]]).sum(axis=0)
    return raw_s, del_s, ok, bad, tied


def threat_frame(labels_path: str, workers: int | None = None, log=print) -> dict:
    import multiprocessing
    import numpy as np
    from . import base_vocabulary as V
    from . import explorer as E
    t0 = time.time()
    recs = json.load(open(labels_path))
    terms = threat_terms()[0]
    log(f"OBS THREAT TERM SET: {len(terms)} words ({len(lift_terms()[0])} piece-quantifier lifts; history and forcing words excluded)")
    sizes_ = [V.size(V.parse(t)) for t in terms]
    chunks = [recs[i::40] for i in range(40)]
    prefix, hist = [], []
    total_pairs = None
    with multiprocessing.get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        for rank in range(THREAT_TERMS):
            res = pool.map(_threat_chunk, [(c, prefix) for c in chunks])
            raw = sum(r[0] for r in res); dl = sum(r[1] for r in res)
            ok = sum(r[2] for r in res); bad = sum(r[3] for r in res); tied = sum(r[4] for r in res)
            if total_pairs is None:
                total_pairs = ok + bad + tied
            allsc = np.concatenate([raw, dl])
            best = max(range(len(allsc)), key=lambda i: (abs(int(allsc[i])), -sizes_[i % len(terms)], -i))
            sc = int(allsc[best]); col, fam = best % len(terms), "raw" if best < len(terms) else "delta"
            sign = 1 if sc >= 0 else -1
            before = (ok - bad) / total_pairs
            after = (ok - bad + abs(sc)) / total_pairs
            log(f"OBS THREAT FRAME term {rank + 1}: {fam} of {terms[col]} (sign {sign:+d}), tied pairs {tied} of {total_pairs}; strict pairwise score {before:.3f} -> {after:.3f}")
            if rank > 0 and after - before < MIN_GAIN:
                log("OBS THREAT FRAME: the training score stopped improving; no further term")
                break
            prefix.append((col, sign, fam))
            hist.append(dict(term=terms[col], family=fam, sign=sign, gloss=term_gloss(terms[col]), score_after=after))
    frame = dict(terms=hist, pairs=total_pairs, positions=len(recs), seconds=round(time.time() - t0, 1), labels=labels_path)
    json.dump(frame, open(paths.derived("threat_frame.json"), "w"), indent=1)
    for h in hist:
        log(f"OBS THREAT WORD: {'the CHANGE in' if h['family'] == 'delta' else ''} {h['gloss']}  [{h['term']}], {'higher' if h['sign'] > 0 else 'lower'} = bigger threat")
    log(f"OBS threat frame learned in {frame['seconds']} s -> {paths.derived('threat_frame.json')}")
    return frame


def frame_ranking(frame: dict, rec: dict):
    """The frame's order of the O moves of a labelled position against the null: -> [(uci, tuple)] above the null, biggest first; and the null's tuple."""
    import numpy as np
    terms = threat_terms()[0]
    cols = [terms.index(h["term"]) for h in frame["terms"]]
    fens = [_after_fen(rec["pass_fen"], m["uci"]) for m in rec["moves"]] + [rec["fen"]]
    M = _tmatrix(fens)[:, cols]
    D = M - M[-1]
    D[-1] = 0
    def tup(i):
        return tuple(int(h_sign * (M[i, k] if h["family"] == "raw" else D[i, k])) for k, (h, h_sign) in enumerate(zip(frame["terms"], [h["sign"] for h in frame["terms"]])))
    null = tup(len(fens) - 1)
    ranked = sorted(((tup(i), rec["moves"][i]["uci"]) for i in range(len(rec["moves"]))), reverse=True)
    return [(u, t) for t, u in ranked if t > null], null


def _check_chunk(args):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core as A
    frame, recs = args
    out = []
    for rec in recs:
        above, null = frame_ranking(frame, rec)
        top = above[0][0] if above else None
        conf = None
        if top:
            res = A.forced_moves(rec["pass_fen"], 1, 4, 300_000)
            conf = any(r[0] == top and r[1] == "forced" for r in res)
        sizes = {m["uci"]: m["size"] for m in rec["moves"]}
        mx = max(sizes.values()) if sizes else 0
        best = {u for u, z in sizes.items() if z == mx and mx >= 1}
        out.append(dict(top=top, confirmed=conf, teacher_max=mx, top1=bool(top and top in best), top3=any(u in best for u, _ in above[:3]),
                        frame_has=bool(above), teacher_has=mx >= 1, top_size=sizes.get(top, None)))
    return out


def check_frame(frame_path: str, labels_path: str, log=print) -> dict:
    """The learned definition checked on the HELD positions (once): our exact solver (not Lean) confirms the frame's top O move wins >= 1 by force
    within 4; agreement with the teacher's biggest threat; agreement on 'no threat'."""
    import multiprocessing
    t0 = time.time()
    frame = json.load(open(frame_path))
    recs = json.load(open(labels_path))
    chunks = [recs[i::20] for i in range(20)]
    with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = [r for part in pool.map(_check_chunk, [(frame, c) for c in chunks]) for r in part]
    n = len(rows)
    has = [r for r in rows if r["teacher_has"]]
    tops = [r for r in rows if r["top"]]
    out = dict(positions=n, teacher_with_threat=len(has), frame_with_threat=len(tops),
               solver_confirmed=sum(bool(r["confirmed"]) for r in tops) / max(1, len(tops)),
               top1=sum(r["top1"] for r in has) / max(1, len(has)), top3=sum(r["top3"] for r in has) / max(1, len(has)),
               no_threat_agreement=sum(r["frame_has"] == r["teacher_has"] for r in rows) / n,
               top_size_ge1=sum((r["top_size"] or 0) >= 1 for r in tops) / max(1, len(tops)), seconds=round(time.time() - t0, 1))
    log(f"OBS THREAT FRAME CHECK (held, {n} positions; checked by our exact solver, not proved in Lean): teacher sees a threat in {len(has)}, the frame in {len(tops)}; "
        f"the frame's top move confirmed by the solver as a forced gain >= 1 within 4 in {out['solver_confirmed']:.0%}; the teacher also sizes it >= 1 in {out['top_size_ge1']:.0%}; "
        f"top-1 agreement with the teacher's biggest {out['top1']:.0%}; teacher's biggest within the frame's top 3 {out['top3']:.0%}; agreement on 'threat / no threat' {out['no_threat_agreement']:.0%}; {out['seconds']} s")
    json.dump(out, open(paths.derived("threat_frame_check.json"), "w"), indent=1)
    return out


def _sizer_chunk(args):
    """In a worker: the ledger's sizer on each held position (every move of the other side sized), against the teacher's labels."""
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core as A
    recs = args
    out = []
    for rec in recs:
        t1 = time.time()
        L = A.Ledger(rec["fen"], 1000)
        if os.environ.get("SIZE_ALL_ROOT"):
            L.set_prefilter_root(False)
        rep = L.threat_report(True)
        th = [t for t in rep["threats"] if t["size"] >= 1]
        sizes = {m["uci"]: max(0, m["size"]) for m in rec["moves"]}
        mx = max(sizes.values()) if sizes else 0
        best = {u for u, z in sizes.items() if z == mx and mx >= 1}
        top = th[0]["uci"] if th else None
        conf = None
        if top:
            conf = any(r[0] == top and r[1] == "forced" for r in A.forced_moves(rec["pass_fen"], 1, 4, 300_000))
        by = {t["uci"]: t["size"] for t in rep["threats"]}
        out.append(dict(fen=rec["fen"], top=top, top_san=th[0]["san"] if th else None, top_size=th[0]["size"] if th else 0, confirmed=conf,
                        teacher_max=mx, teacher_moves=sorted(best), teacher_san=[(chess.Board(rec["pass_fen"]).san(chess.Move.from_uci(u)), sizes[u]) for u in sorted(best)][:3],
                        top1=bool(top and top in best), top3=any(t["uci"] in best for t in th[:3]), ledger_has=bool(th), teacher_has=mx >= 1,
                        ledger_on_teachers_best=[by.get(u, 0) for u in sorted(best)][:1], detail=th[0]["detail"] if th else "",
                        positions=rep["positions"], looks=rep["looks"], questions=rep["questions"], seconds=round(time.time() - t1, 2)))
    return out


def check_sizer(labels_path: str, log=print) -> dict:
    import multiprocessing
    t0 = time.time()
    recs = json.load(open(labels_path))
    chunks = [recs[i::20] for i in range(20)]
    with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = [r for part in pool.map(_sizer_chunk, chunks) for r in part]
    n = len(rows)
    has = [r for r in rows if r["teacher_has"]]
    tops = [r for r in rows if r["top"]]
    out = dict(positions=n, teacher_with_threat=len(has), ledger_with_threat=len(tops),
               top1=sum(r["top1"] for r in has) / max(1, len(has)), top3=sum(r["top3"] for r in has) / max(1, len(has)),
               no_threat_agreement=sum(r["ledger_has"] == r["teacher_has"] for r in rows) / n,
               solver_confirmed=sum(bool(r["confirmed"]) for r in tops) / max(1, len(tops)),
               mean_positions=sum(r["positions"] for r in rows) / n, mean_looks=sum(r["looks"] for r in rows) / n, mean_questions=sum(r["questions"] for r in rows) / n,
               mean_seconds=sum(r["seconds"] for r in rows) / n, seconds=round(time.time() - t0, 1))
    log(f"OBS THREAT SIZER CHECK (held, {n} positions; the ledger's sizing of every move of the other side vs Stockfish; checked by our exact solver, not Lean): "
        f"teacher sees a threat in {len(has)}, the ledger in {len(tops)}; top-1 {out['top1']:.0%}, teacher's biggest within the ledger's top 3 {out['top3']:.0%}; "
        f"agreement on threat / no threat {out['no_threat_agreement']:.0%}; the ledger's biggest confirmed by the solver as a forced gain >= 1 within 4: {out['solver_confirmed']:.0%}; "
        f"per position {out['mean_positions']:.0f} positions, {out['mean_looks']:.0f} static looks, {out['mean_questions']:.0f} questions, {out['mean_seconds']:.1f} s; total {out['seconds']} s")
    dis = [r for r in rows if (r["teacher_has"] and not r["top1"]) or (r["ledger_has"] != r["teacher_has"])]
    import random as _r
    _r.Random(0).shuffle(dis)
    for r in dis[:10]:
        if r["teacher_has"] and r["ledger_has"]:
            lb = r["ledger_on_teachers_best"][0] if r["ledger_on_teachers_best"] else 0
            why = f"ledger's biggest {r['top_san']} ({r['top_size']}); it sizes the teacher's move at {lb}" + (" -- the static sizing sees a defence or counter-chance the teacher's deeper line does not allow" if lb == 0 else " -- ranked below another threat")
        elif r["teacher_has"]:
            why = "the ledger sizes none >= 1: S's static value there says it can save the piece / counter, or the loss is beyond the forcing lines"
        else:
            why = f"the teacher sees no threat; the ledger sizes {r['top_san']} at {r['top_size']} ({r['detail'][:90]}): a counter-chance of S the static sizing misses"
        log(f"OBS DISAGREEMENT {r['fen']} | teacher: {r['teacher_san']} | ledger: {r['top_san']} {r['top_size']} | {why}")
    json.dump(dict(summary=out, rows=rows), open(paths.derived("threat_sizer_check.json"), "w"))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------
# LEARNING HOW NOT TO LOSE MATERIAL (Kaveh 2026-10-14: "Define it. Let the machinery learn how to not lose material."). The threat is DEFINED (material lost,
# or mate); what is learned from Stockfish is the DEFENCE: in positions where the teacher sees a threat against S (the pass test, threat_labels), S to move
# in the real position P, every legal move a of S is labelled by Stockfish (depth 10, the PV walked to the first quiet point): loss(a) = S's net material
# loss from P (>= 0; a mate against S = 100). The words are read on the board AFTER a with the turn handed back to S (a static reading, like _after): so
# "our" is S's side and "see" / the lifts mean what the other side can win on S's pieces; the "before" is P itself, read the same way.

DEFENCE_CAP = 20               # a loss beyond this (a mate) is ranked as this much


def defence_labels(fen: str, eng, depth: int = THREAT_DEPTH) -> dict | None:
    import chess.engine
    b = chess.Board(fen)
    if b.is_check() or b.is_game_over():
        return None
    s = b.turn
    lim = chess.engine.Limit(depth=depth)
    out = []
    for a in b.legal_moves:
        q = b.copy(stack=False); q.push(a)
        if q.is_game_over():
            out.append(dict(uci=a.uci(), loss=0)); continue
        info = eng.analyse(q, lim)
        sc = info["score"].pov(s)
        out.append(dict(uci=a.uci(), loss=max(0, walk_size(b, a, info.get("pv", []), sc.is_mate() and sc.mate() < 0, s))))
    return dict(fen=fen, moves=out)


def _defence_one(args):
    rec = args
    r = defence_labels(rec["fen"], ENG)
    if r is not None:
        r.update(id=rec.get("id"), source=rec.get("source"))
    return r


def label_defence(name: str, log=print) -> dict:
    import multiprocessing
    t0 = time.time()
    src = json.load(open(paths.derived(f"watch/threat_labels_{name}.json")))
    recs = [r for r in src if any(m["size"] >= 1 for m in r["moves"])]
    with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_label_init) as pool:
        out = [r for r in pool.imap(_defence_one, recs, chunksize=2) if r]
    p = paths.derived(f"watch/defence_labels_{name}.json")
    json.dump(out, open(p, "w"))
    nz = sum(any(m["loss"] == 0 for m in r["moves"]) for r in out)
    hist = {}
    for r in out:
        for m in r["moves"]:
            z = m["loss"]
            k = "0" if z <= 0 else "1" if z == 1 else "2" if z == 2 else "3-4" if z <= 4 else "5-8" if z <= 8 else "9-99" if z < 100 else "mate"
            hist[k] = hist.get(k, 0) + 1
    log(f"OBS DEFENCE LABELS {name}: {len(out)} positions (of {len(src)} with a teacher threat >= 1), {sum(len(r['moves']) for r in out)} moves, positions where some move "
        f"loses nothing: {nz / max(1, len(out)):.0%}, loss histogram {hist}, {time.time() - t0:.0f} s -> {p}")
    return dict(positions=len(out))


def _true_after(fen: str, uci: str) -> str:
    """The TRUE board after S's move (the other side to move) -- read with us = S, so a move that gives check is read like any other (the first version handed the turn
    back to S, which made every checking move a position the evaluator refused: all zeros, and "lower is better" gave checks a free pass)."""
    b = chess.Board(fen)
    b.push(chess.Move.from_uci(uci))
    return b.fen()


def sf_move_loss(fen: str, uci: str, eng, depth: int = 14) -> int:
    """The teacher's loss for S's move a at P (S to move): S's net material loss with Stockfish defending, the PV walked to the first quiet point; mate against S = 100."""
    import chess.engine
    b = chess.Board(fen)
    a = chess.Move.from_uci(uci)
    s = b.turn
    q = b.copy(stack=False); q.push(a)
    if q.is_game_over():
        return 0
    info = eng.analyse(q, chess.engine.Limit(depth=depth))
    sc = info["score"].pov(s)
    return max(0, walk_size(b, a, info.get("pv", []), sc.is_mate() and sc.mate() < 0, s))


def sf_capture_size(fen: str, uci: str, loser: bool, eng, depth: int = 14) -> int:
    """The teacher's size of a move m of the winner at B (the winner to move): the loser's net loss with Stockfish defending."""
    import chess.engine
    b = chess.Board(fen)
    m = chess.Move.from_uci(uci)
    q = b.copy(stack=False); q.push(m)
    if q.is_checkmate():
        return MATE_SIZE
    if q.is_game_over():
        return 0
    info = eng.analyse(q, chess.engine.Limit(depth=depth))
    sc = info["score"].pov(loser)
    return walk_size(b, m, info.get("pv", []), sc.is_mate() and sc.mate() < 0, loser)


def _flip_after(fen: str, uci: str) -> str:
    """The board after S's move a with the turn handed back to S (a static reading: the words read from S's side)."""
    b = chess.Board(fen)
    b.push(chess.Move.from_uci(uci))
    b.turn = not b.turn
    b.ep_square = None
    return b.fen()


def _defence_matrices(rec: dict):
    import numpy as np
    fens = [_true_after(rec["fen"], m["uci"]) for m in rec["moves"]] + [rec["fen"]]
    M = _tmatrix(fens, chess.Board(rec["fen"]).turn).astype(np.int32)
    base = M[-1]
    D = M[:-1] - base
    sizes = -np.minimum(np.array([m["loss"] for m in rec["moves"]]), DEFENCE_CAP)       # larger = better = less loss
    return sizes, M[:-1], D


def _defence_chunk(args):
    import numpy as np
    recs, prefix = args
    T = len(threat_terms()[0])
    raw_s = np.zeros(T, np.int64); del_s = np.zeros(T, np.int64)
    ok = bad = tied = 0
    for rec in recs:
        sizes, R, D = _defence_matrices(rec)
        I, J = np.nonzero(sizes[:, None] - sizes[None, :] >= 1)
        if not len(I):
            continue
        undecided = np.ones(len(I), bool)
        for (col, sign, fam) in prefix:
            X = (R if fam == "raw" else D)[:, col] * sign
            d = np.sign(X[I] - X[J])
            ok += int(((d > 0) & undecided).sum()); bad += int(((d < 0) & undecided).sum())
            undecided &= d == 0
        tied += int(undecided.sum())
        if undecided.any():
            raw_s += np.sign(R[I[undecided]] - R[J[undecided]]).sum(axis=0)
            del_s += np.sign(D[I[undecided]] - D[J[undecided]]).sum(axis=0)
    return raw_s, del_s, ok, bad, tied


def defence_frame(labels_path: str, log=print) -> dict:
    import multiprocessing
    import numpy as np
    from . import base_vocabulary as V
    t0 = time.time()
    recs = json.load(open(labels_path))
    terms = threat_terms()[0]
    sizes_ = [V.size(V.parse(t)) for t in terms]
    chunks = [recs[i::40] for i in range(40)]
    prefix, hist, total = [], [], None
    log(f"OBS DEFENCE LEARNER: {len(recs)} positions; the words are read on the board after S's move with the turn handed back to S, so 'our' is S's side and "
        f"see / the lifts mean what the other side can win on S's pieces; {len(terms)} words (history and forcing words out, piece-quantifier lifts in)")
    with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for rank in range(THREAT_TERMS):
            res = pool.map(_defence_chunk, [(c, prefix) for c in chunks])
            raw = sum(r[0] for r in res); dl = sum(r[1] for r in res)
            ok = sum(r[2] for r in res); bad = sum(r[3] for r in res); tied = sum(r[4] for r in res)
            if total is None:
                total = ok + bad + tied
            allsc = np.concatenate([raw, dl])
            # a word that names a square (a1 .. h8) is a position, not a notion: skipped
            import re
            sq = re.compile(r"(?<![a-z])[a-h][1-8](?![0-9])")
            order = sorted(range(len(allsc)), key=lambda i: (-abs(int(allsc[i])), sizes_[i % len(terms)], i))
            best = next(i for i in order if not sq.search(terms[i % len(terms)]))
            sc = int(allsc[best]); col, fam = best % len(terms), "raw" if best < len(terms) else "delta"
            sign = 1 if sc >= 0 else -1
            before = (ok - bad) / total
            after = (ok - bad + abs(sc)) / total
            log(f"OBS DEFENCE FRAME term {rank + 1}: {fam} of {terms[col][:160]} (sign {sign:+d}), tied pairs {tied} of {total}; strict pairwise {before:.3f} -> {after:.3f}")
            if rank > 0 and after - before < MIN_GAIN:
                log("OBS DEFENCE FRAME: the training score stopped improving; no further term")
                break
            prefix.append((col, sign, fam))
            hist.append(dict(term=terms[col], family=fam, sign=sign, gloss=term_gloss(terms[col]), score_after=after))
    frame = dict(terms=hist, pairs=total, positions=len(recs), seconds=round(time.time() - t0, 1), labels=labels_path, reads="the TRUE board after S's move (the other side to move), us = S")
    json.dump(frame, open(paths.derived("defence_frame.json"), "w"), indent=1)
    for h in hist:
        log(f"OBS DEFENCE WORD: {'the CHANGE in ' if h['family'] == 'delta' else ''}{h['gloss']}  [{h['term'][:200]}], {'higher' if h['sign'] > 0 else 'lower'} = a better defence")
    log(f"OBS defence frame learned in {frame['seconds']} s -> {paths.derived('defence_frame.json')}")
    return frame


def defence_ranking(frame: dict, rec: dict) -> list:
    """The frame's order of S's moves at a labelled position, best first: [(uci, tuple)]."""
    import numpy as np
    terms = threat_terms()[0]
    cols = [terms.index(h["term"]) for h in frame["terms"]]
    fens = [_true_after(rec["fen"], m["uci"]) for m in rec["moves"]] + [rec["fen"]]
    M = _tmatrix(fens, chess.Board(rec["fen"]).turn)[:, cols]
    D = M[:-1] - M[-1]
    tups = [tuple(int(h["sign"] * (M[i, k] if h["family"] == "raw" else D[i, k])) for k, h in enumerate(frame["terms"])) for i in range(len(rec["moves"]))]
    return sorted(((t, rec["moves"][i]["uci"]) for i, t in enumerate(tups)), reverse=True)


def _defence_check_chunk(args):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core as A
    frame, recs = args
    out = []
    for rec in recs:
        rk = defence_ranking(frame, rec)
        loss = {m["uci"]: m["loss"] for m in rec["moves"]}
        free = {u for u, z in loss.items() if z == 0}
        top = rk[0][1]
        b = chess.Board(rec["fen"]); b.push_uci(top)
        conf = not any(r[1] == "forced" for r in A.forced_moves(b.fen(), 1, 4, 300_000)) if not b.is_game_over() else True
        out.append(dict(fen=rec["fen"], top=top, top_loss=loss[top], some_free=bool(free), top_free=top in free, top3_free=any(u in free for _, u in rk[:3]), confirmed=conf,
                        n_moves=len(loss), best_loss=min(loss.values())))
    return out


def check_defence(frame_path: str, labels_path: str, log=print) -> dict:
    """Held out, once: does the frame's top move lose nothing (by the teacher)? does our exact solver confirm that after it the other side cannot force a gain
    >= 1 within depth 4? (checked by our exact solver, not proved in Lean)"""
    import multiprocessing
    t0 = time.time()
    frame = json.load(open(frame_path))
    recs = json.load(open(labels_path))
    chunks = [recs[i::20] for i in range(20)]
    with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = [r for part in pool.map(_defence_check_chunk, [(frame, c) for c in chunks]) for r in part]
    n = len(rows)
    sf = [r for r in rows if r["some_free"]]
    out = dict(positions=n, positions_with_a_free_move=len(sf),
               top_loses_nothing_of_those=sum(r["top_free"] for r in sf) / max(1, len(sf)), top3_has_one_of_those=sum(r["top3_free"] for r in sf) / max(1, len(sf)),
               top_loses_nothing_all=sum(r["top_loss"] == 0 for r in rows) / n, mean_loss_top=sum(min(r["top_loss"], DEFENCE_CAP) for r in rows) / n,
               mean_best_loss=sum(min(r["best_loss"], DEFENCE_CAP) for r in rows) / n,
               solver_confirms_top=sum(r["confirmed"] for r in rows) / n, solver_confirms_top_when_teacher_says_free=sum(r["confirmed"] for r in sf if r["top_free"]) / max(1, sum(r["top_free"] for r in sf)),
               seconds=round(time.time() - t0, 1))
    log(f"OBS DEFENCE FRAME CHECK (held, {n} positions with a teacher threat; checked by our exact solver, not proved in Lean): some move loses nothing in {len(sf)}; "
        f"in those the frame's top move loses nothing {out['top_loses_nothing_of_those']:.0%}, a zero-loss move is in its top 3 {out['top3_has_one_of_those']:.0%}; "
        f"all positions: the top move loses nothing {out['top_loses_nothing_all']:.0%} (mean loss {out['mean_loss_top']:.2f} against the best move's {out['mean_best_loss']:.2f}); "
        f"the solver confirms the top move (O cannot force a gain >= 1 within 4) in {out['solver_confirms_top']:.0%} of all and {out['solver_confirms_top_when_teacher_says_free']:.0%} of those the teacher calls free; {out['seconds']} s")
    json.dump(out, open(paths.derived("defence_frame_check.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("extract", nargs="?", help="'extract' (Lichess games) or 'threats' (the teacher's threat labels)")
    ap.add_argument("--set", choices=("train", "held"), default="train")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--targets", default="KRvK,KQvK,KRRvK,KQvKR,KRPvKR,KPvK")
    ap.add_argument("--limit", type=int, default=None, help="games to read at most (default: the whole month)")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.extract == "learn-threats":
        threat_frame(paths.derived("watch/threat_labels_train.json"))
        raise SystemExit
    if a.extract == "defence-labels":
        label_defence(a.set)
        raise SystemExit
    if a.extract == "learn-defence":
        defence_frame(paths.derived("watch/defence_labels_train.json"))
        raise SystemExit
    if a.extract == "check-defence":
        check_defence(paths.derived("defence_frame.json"), paths.derived("watch/defence_labels_held.json"))
        raise SystemExit
    if a.extract == "check-sizer":
        check_sizer(paths.derived("watch/threat_labels_held.json"))
        raise SystemExit
    if a.extract == "check-threats":
        check_frame(paths.derived("threat_frame.json"), paths.derived("watch/threat_labels_held.json"))
        raise SystemExit
    if a.extract == "threats":
        label_set(a.set, 17 if a.set == "train" else 23, a.n or (400 if a.set == "train" else 100))
        raise SystemExit
    got = extract(set(a.targets.split(",")), a.limit, a.workers)
    os.makedirs(paths.derived("watch"), exist_ok=True)
    for t, gs in got.items():
        json.dump(gs, open(paths.derived(f"watch/lichess_2019-01_{t}.json"), "w"))
