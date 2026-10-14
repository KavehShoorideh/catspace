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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("extract", nargs="?")
    ap.add_argument("--targets", default="KRvK,KQvK,KRRvK,KQvKR,KRPvKR,KPvK")
    ap.add_argument("--limit", type=int, default=None, help="games to read at most (default: the whole month)")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    got = extract(set(a.targets.split(",")), a.limit, a.workers)
    os.makedirs(paths.derived("watch"), exist_ok=True)
    for t, gs in got.items():
        json.dump(gs, open(paths.derived(f"watch/lichess_2019-01_{t}.json"), "w"))
