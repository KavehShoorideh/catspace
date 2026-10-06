"""THE CASE DATABASE (Kaveh 2026-10-05): "when it's a blunder, the learning is that this is a case where an attack
didn't materialize. Why is interesting if we could mine it. We'd need a memory of similar positions where the attack
succeeded. We'd need an indexed database."

EVENTS are indexed from the Lichess records on disk (any players; the watcher's own stream, game_id % 97 == 7, is left
out so nothing mined here can leak the games it is scored on). The first event is a SACRIFICE: a capture whose
static exchange, the capturer going first with the piece it used, loses at least 2 pawns (35.Qxd3 in game 11453:
queen takes a bishop, a pawn takes the queen, -6). Each event keeps named attributes (the pieces, the investment,
check, distance to the enemy king, the attacker's pieces bearing on the king's squares, the king's pawn shield,
material, phase, ratings) and its OUTCOME (Kaveh: success = the material comes back within N plies after the
capture, or the attacker mates).

Retrieval (Kaveh: "same event, nearby attributes"): the cases with the same capturing and captured pieces, ranked by
distance over the attributes. Mining: the condition that separates successes from failures among them, by a shallow
decision tree over the named attributes (scikit-learn), read as a rule and its exception.

    python -m catspace.incubator.analyst.watch.cases build --shards 0 1 [--games N] [--workers 11]
    python -m catspace.incubator.analyst.watch.cases similar --game 11453 --ply 69 [--k 60]
    python -m catspace.incubator.analyst.watch.cases mine --capturer Q [--victim B]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

import chess

from catspace.io import paths

from .frames import VAL, legal_attackers, material, see

WATCH_STRIDE, WATCH_RESIDUE = 97, 7
N_PLIES = 10                                                                 # success: the material back within this many plies
LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
FEATURES = ["investment", "gives_check", "king_dist", "king_zone_attackers", "king_shield", "material", "phase", "elo", "elo_diff"]


def db_path() -> Path:
    return Path(os.environ.get("CATSPACE_WATCH_DATA") or paths.derived("watch")) / "cases.sqlite"


def exchange_for(b: chess.Board, m: chess.Move) -> int:
    """The capturer's exchange when it takes with THIS piece first and both sides then take with their cheapest legal
    attacker, each free to stop: what it takes, less what the other side then wins on the square."""
    victim = b.piece_at(m.to_square)
    if b.is_en_passant(m):
        victim = chess.Piece(chess.PAWN, not b.turn)
    if victim is None:
        return 0
    after = b.copy(stack=False)
    after.push(m)
    return VAL[victim.piece_type] - see(after, m.to_square, after.turn)


def king_features(b: chess.Board, attacker: bool) -> dict:
    """Around the DEFENDER's king (the side that is not `attacker`): the attacker's pieces bearing on its squares, the
    defender's own pawns next to it."""
    k = b.king(not attacker)
    if k is None:
        return dict(king_zone_attackers=0, king_shield=0)
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[k]) | chess.SquareSet.from_square(k)
    bearing = set()
    for sq in zone:
        bearing |= set(b.attackers(attacker, sq))
    shield = sum(1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[k]) if (p := b.piece_at(sq)) and p.color != attacker and p.piece_type == chess.PAWN)
    return dict(king_zone_attackers=len(bearing), king_shield=shield)


def outcome(boards: list[chess.Board], t: int, side: bool) -> dict:
    """After the capture at ply t (boards[t] -> boards[t+1]): whether the side's material came back to where it was
    before the capture within N_PLIES plies after the reply (or it mated), when, and whether the piece was taken back."""
    base = material(boards[t], side)
    end = min(len(boards) - 1, t + 1 + N_PLIES)
    for i in range(t + 2, end + 1):                                          # from the position after the reply
        if boards[i].is_checkmate() and boards[i].turn != side:
            return dict(success=1, back_in=i - t, mate=1)
        stands = i + 1 >= len(boards) or material(boards[i + 1], side) >= base   # the gain must survive the next reply:
        if material(boards[i], side) >= base and stands:                     # a peak mid-exchange is not "back" (2026-10-05)
            return dict(success=1, back_in=i - t, mate=0)
    return dict(success=0, back_in=None, mate=0)


def scan_game(row: dict) -> list[dict]:
    """Every sacrifice of one game, with its attributes and outcome."""
    moves = row["moves"].split()
    boards = [chess.Board()]
    try:
        for u in moves:
            nb = boards[-1].copy(stack=False)
            nb.push_uci(u)
            boards.append(nb)
    except ValueError:
        return []
    out = []
    for t, u in enumerate(moves):
        b = boards[t]
        m = chess.Move.from_uci(u)
        if not b.is_capture(m):
            continue
        x = exchange_for(b, m)
        if x > -2:
            continue
        side = b.turn
        after = boards[t + 1]
        k = after.king(not side)
        recap = legal_attackers(after, not side, m.to_square)
        recapturer = min((after.piece_at(s).piece_type for s in recap), key=lambda p: VAL[p] if p != chess.KING else 100) if recap else None
        me, them = (row["white_elo"], row["black_elo"]) if side == chess.WHITE else (row["black_elo"], row["white_elo"])
        nonpawn = sum({1: 0, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}[p.piece_type] for p in b.piece_map().values())
        ev = dict(game=row["game_id"], ply=t + 1, fen=b.fen(), move=u, kind="sacrifice",
                  capturer=LETTER[b.piece_at(m.from_square).piece_type],
                  victim=LETTER[(b.piece_at(m.to_square) or chess.Piece(chess.PAWN, not side)).piece_type],
                  recapturer=LETTER[recapturer] if recapturer else "-",
                  investment=-x, gives_check=int(b.gives_check(m)),
                  king_dist=chess.square_distance(m.to_square, k) if k is not None else 8,
                  material=material(b, side), phase=nonpawn, elo=me or 0, elo_diff=(me or 0) - (them or 0),
                  result=(0 if row["result"] == 0 else (1 if (row["result"] == 1) == (side == chess.WHITE) else -1)),
                  term=row["termination"], **king_features(after, side))
        ev.update(outcome(boards, t, side))
        out.append(ev)
    return out


def _scan_shard(args) -> list[dict]:
    shard, limit = args
    import pyarrow.parquet as pq
    t = pq.read_table(paths.REPO_ROOT / "data" / "records" / "lichess_2019-01" / shard,
                      columns=["game_id", "white_elo", "black_elo", "result", "termination", "moves"])
    out, n = [], 0
    for row in t.to_pylist():
        if row["game_id"] % WATCH_STRIDE == WATCH_RESIDUE:
            continue
        n += 1
        if limit and n > limit:
            break
        out += scan_game(row)
    return out


COLS = ["game", "ply", "fen", "move", "kind", "capturer", "victim", "recapturer", "investment", "gives_check", "king_dist",
        "king_zone_attackers", "king_shield", "material", "phase", "elo", "elo_diff", "result", "term", "success", "back_in", "mate"]


def build(shards: list[int], workers: int = 11, limit: int = 0) -> dict:
    """Scan the shards in parallel (one shard per worker) and write each shard's events as it arrives, so memory holds
    one shard at a time; prints a line per shard."""
    from multiprocessing import Pool
    t0 = time.perf_counter()
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute(f"CREATE TABLE IF NOT EXISTS events ({', '.join(COLS)}, PRIMARY KEY (game, ply))")
    con.execute("CREATE INDEX IF NOT EXISTS by_kind ON events (kind, capturer, victim)")
    jobs = [(f"records_{s:05d}.parquet", limit) for s in shards]
    done = added = 0
    with Pool(min(workers, len(jobs))) as pool:
        for part in pool.imap_unordered(_scan_shard, jobs):
            con.executemany(f"INSERT OR REPLACE INTO events VALUES ({', '.join('?' * len(COLS))})", [[e.get(c) for c in COLS] for e in part])
            con.commit()
            done += 1
            added += len(part)
            print(f"OBS cases: shard {done}/{len(jobs)} written, {len(part)} sacrifices; {added} so far; {time.perf_counter() - t0:.0f}s", flush=True)
    n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    con.close()
    secs = time.perf_counter() - t0
    print(f"OBS cases: {added} sacrifices from {len(shards)} shards ({'up to ' + str(limit) + ' games each' if limit else 'all games'}) "
          f"in {secs:.0f}s; the database holds {n}")
    return dict(events=added, seconds=secs, total=n)


def rows(where: str = "", params=()) -> list[dict]:
    con = sqlite3.connect(db_path())
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(f"SELECT * FROM events {where}", params)]
    con.close()
    return out


def similar(query: dict, k: int = 60) -> list[dict]:
    """The k cases of the same event with the same capturing and captured pieces, nearest over the attributes (each
    scaled by its spread among them)."""
    same = rows("WHERE kind = ? AND capturer = ? AND victim = ?", (query["kind"], query["capturer"], query["victim"]))
    same = [c for c in same if not (c["game"] == query["game"] and c["ply"] == query["ply"])]
    if not same:
        return []
    import statistics as st
    scale = {f: (st.pstdev([c[f] for c in same]) or 1.0) for f in FEATURES}
    key = lambda c: sum(((c[f] - query[f]) / scale[f]) ** 2 for f in FEATURES)
    return sorted(same, key=key)[:k]


def mine(cases: list[dict], depth: int = 2, min_leaf: int = 8) -> list[str]:
    """The conditions that separate successes from failures among cases: a shallow decision tree over the named
    attributes, read leaf by leaf as rules."""
    from sklearn.tree import DecisionTreeClassifier
    X = [[c[f] for f in FEATURES] for c in cases]
    y = [c["success"] for c in cases]
    if len(set(y)) < 2:
        return [f"all {len(cases)} cases {'succeeded' if y and y[0] else 'failed'}"]
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=0).fit(X, y)
    t = tree.tree_
    out = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            n = int(t.n_node_samples[node])
            v = t.value[node][0]
            rate = v[1] / v.sum() if v.sum() else 0
            out.append(f"{' and '.join(conds) or 'all'}: {n} cases, attack succeeded in {100 * rate:.0f}%")
            return
        f, thr = FEATURES[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.1f}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.1f}"])
    walk(0, [])
    return out


def query_from_game(game_id: int, ply: int) -> dict:
    """The sacrifice played at `ply` (the move that reached position ply) of a game, as a query case."""
    from .watcher import games_by_id
    g = games_by_id(game_id)
    for e in scan_game(g):
        if e["ply"] == ply:
            return e
    raise SystemExit(f"no sacrifice at ply {ply} of game {game_id}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "similar", "mine"])
    ap.add_argument("--shards", type=int, nargs="*", default=[0])
    ap.add_argument("--games", type=int, default=0, help="build: at most this many games per shard (0: all)")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--game", type=int, default=None)
    ap.add_argument("--ply", type=int, default=None)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--capturer", default=None)
    ap.add_argument("--victim", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.shards, a.workers, a.games)
    elif a.cmd == "similar":
        q = query_from_game(a.game, a.ply)
        near = similar(q, a.k)
        ok = sum(c["success"] for c in near)
        print(f"OBS query: game {q['game']} ply {q['ply']} {q['capturer']}x{q['victim']} (recaptured by {q['recapturer']}), invests {q['investment']}, "
              f"check {q['gives_check']}, king {q['king_dist']} away, {q['king_zone_attackers']} attackers on the king's squares, shield "
              f"{q['king_shield']}: success {q['success']}")
        print(f"OBS its {len(near)} nearest cases: the attack succeeded in {ok} ({100 * ok / max(len(near), 1):.0f}%)")
        for line in mine(near):
            print("OBS   " + line)
    else:
        where, params = "WHERE kind = 'sacrifice'", []
        if a.capturer:
            where += " AND capturer = ?"
            params.append(a.capturer)
        if a.victim:
            where += " AND victim = ?"
            params.append(a.victim)
        cs = rows(where, params)
        ok = sum(c["success"] for c in cs)
        print(f"OBS {len(cs)} sacrifices ({a.capturer or 'any'} x {a.victim or 'any'}): the attack succeeded in {ok} ({100 * ok / max(len(cs), 1):.0f}%)")
        for line in mine(cs, depth=3, min_leaf=30):
            print("OBS   " + line)


if __name__ == "__main__":
    main()
