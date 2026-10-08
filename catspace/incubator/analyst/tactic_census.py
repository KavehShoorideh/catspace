"""THE TACTIC CENSUS (Kaveh 2026-10-07: "find a way to quantify the world in terms of the existence of these tactics in a
statistical way ... the existence of these tactics in prior times with these kinds of conditions will then lead to
likely these kinds of tactics with these kinds of conditions"; the planner then navigates toward tactical states that
favour us). A MEASUREMENT before any planner: over ordinary Lichess games, every position's tactics -- fired or latent --
as tokens with their BARRIERS (how many conditions the tactic still misses, and which), for the side to move and, through
a null move, for the opponent; a few named structure features; the material; the Stockfish evaluation.

A token is (family, k, barrier) -- k = 0: the tactic is on now; k >= 1: it misses k conditions, named in `barrier`.
Families, each read from an existing detector:
    back_rank      Lean `backrank` (Ideas.backRankMate_sound), the best instance: k = 0 the lemma holds (MATE); 1 one
                   hypothesis fails (ONEAWAY, named); 2 = two or more fail (the detector stops at the second failure)
    passer         Lean `passer2` (Ideas.Passer2.unstoppablePasser2_sound): the same, over our pawns
    pin            pin.strikes -> 0 (a pin or skewer that wins now); pin.pins without a strike -> 1 ("no strike")
    discovery      discovery.strikes -> 0; a battery aimed at an enemy piece with a checking departure -> 1 ("no
                   strike"); aimed but without one -> 2 ("no departure")
    double_threat  tactic_rules.fork_claims (a local check wins >= 1) -> 0; double_threats without a winning one -> 1
                   ("answered")
    material_now   the one-ply ledger's best frame wins >= 1 now (fates.joint_threats) -> 0
Structure (python-chess): open files, our half-open files, our passed / isolated pawns, pawn islands, the pawns in front
of our king (three files, two ranks).

    python -m catspace.incubator.analyst.tactic_census build --games 300 --shard 50 --workers 8 --tag c300
    python -m catspace.incubator.analyst.tactic_census build --bands 1000-1400,1400-1700,1700-2000,2000-2500 --games 150 --tag wide
    python -m catspace.incubator.analyst.tactic_census measure --tag c300
Rows go to data/derived/tactic_census/census_<tag>.jsonl (one per position); the job is CHECKPOINTED per game (a
.done list beside the file) and resumes with the same tag (Kaveh 2026-10-07)."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import chess

from catspace.io import paths

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
FAMILIES = ("back_rank", "passer", "pin", "discovery", "double_threat", "material_now")


def _lean_tokens(insts, family: str) -> list:
    """The best Lean instance as one token; none when the detector has no candidate. The detector evaluates the lemma's
    hypotheses in order and stops at the SECOND failure, so k is 0 (the lemma holds), 1 (exactly one hypothesis fails,
    named) or 2 meaning "two or more" (the first two failing, named)."""
    best = None
    for i in insts:
        if i.status in ("MATE", "PROVED"):
            return [(family, 0, "")]
        if i.status == "ONEAWAY":
            cand = (1, i.failing or "+".join(k for k, v in (i.hyps or {}).items() if v is False))
        else:
            cand = (2, "+".join(k for k, v in (i.hyps or {}).items() if v is False) or "unknown")
        if best is None or cand[0] < best[0]:
            best = cand
    return [(family, best[0], best[1])] if best else []


def tokens(b: chess.Board) -> list:
    """The side to move's tactic tokens in `b`: [(family, k, barrier)]. Deterministic; the board is not changed."""
    from . import discovery, pin, tactic_rules
    from .fates import joint_threats, square_frame
    from .lean_ideas import detector
    out = []
    li = detector()
    out += _lean_tokens(li.backrank(b), "back_rank")
    out += _lean_tokens(li.passer2(b), "passer")
    if pin.strikes(b):
        out.append(("pin", 0, ""))
    elif pin.pins(b, b.turn):
        out.append(("pin", 1, "no strike"))
    if discovery.strikes(b):
        out.append(("discovery", 0, ""))
    else:
        aimed = [x for x in discovery.batteries(b, b.turn) if x.target is not None]
        if any(x.departures for x in aimed):
            out.append(("discovery", 1, "no strike"))
        elif aimed:
            out.append(("discovery", 2, "no departure"))
    if tactic_rules.fork_claims(b, min_gain=1.0, families=("double threat",)):
        out.append(("double_threat", 0, ""))
    elif tactic_rules.double_threats(b):
        out.append(("double_threat", 1, "answered"))
    if joint_threats(b, square_frame)[0] >= 1:                          # under check too: a capture of the checker that wins is a win
        out.append(("material_now", 0, ""))                               # (review 2026-10-07: the check gate dropped 21% of real wins)
    return out


def opponent_tokens(b: chess.Board) -> list | None:
    """The opponent's tokens: what they would have if it were their move (a null move); None when the side to move is
    in check (no null move)."""
    if b.is_check():
        return None
    nb = b.copy(stack=False)
    nb.push(chess.Move.null())
    return tokens(nb)


def material(b: chess.Board, side: chess.Color) -> int:
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values())


def structure(b: chess.Board, side: chess.Color) -> dict:
    """Named pawn-structure features for `side`."""
    ours, theirs = b.pieces(chess.PAWN, side), b.pieces(chess.PAWN, not side)
    files_ours = {chess.square_file(s) for s in ours}
    files_theirs = {chess.square_file(s) for s in theirs}
    fwd = 1 if side == chess.WHITE else -1
    passed = 0
    for s in ours:
        f, r = chess.square_file(s), chess.square_rank(s)
        if not any(abs(chess.square_file(t) - f) <= 1 and (chess.square_rank(t) - r) * fwd > 0 for t in theirs):
            passed += 1
    isolated = sum(1 for s in ours if not ({chess.square_file(s) - 1, chess.square_file(s) + 1} & files_ours))
    islands, prev = 0, False
    for f in range(8):
        on = f in files_ours
        islands += on and not prev
        prev = on
    k = b.king(side)
    shelter = 0
    if k is not None:
        kf, kr = chess.square_file(k), chess.square_rank(k)
        shelter = sum(1 for s in ours if abs(chess.square_file(s) - kf) <= 1 and 1 <= (chess.square_rank(s) - kr) * fwd <= 2)
    return dict(open_files=sum(1 for f in range(8) if f not in files_ours and f not in files_theirs),
                half_open=sum(1 for f in range(8) if f not in files_ours and f in files_theirs),
                passed=passed, isolated=isolated, islands=islands, shelter=shelter)


def _game_rows(row: dict, engine=None, depth: int = 10) -> list:
    b = chess.Board()
    result_white = int(row["result"])                                  # +1 white won, -1 black won, 0 draw
    out = []
    moves = row["moves"].split()
    for ply in range(len(moves) + 1):
        side = b.turn
        if not b.is_game_over():
            ev = None
            if engine is not None:
                sc = engine.analyse(b, chess.engine.Limit(depth=depth))["score"].pov(side)
                ev = (10000 if sc.mate() > 0 else -10000) if sc.is_mate() else sc.score()
            opp = opponent_tokens(b)
            out.append(dict(game=row["game_id"], ply=ply, side="w" if side == chess.WHITE else "b",
                            elo=int(row["white_elo"] if side == chess.WHITE else row["black_elo"]),
                            elo_opp=int(row["black_elo"] if side == chess.WHITE else row["white_elo"]),
                            fen=b.fen(), stm=tokens(b), opp=opp, material=material(b, side),
                            phase=sum(VAL[p.piece_type] for p in b.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING)),
                            structure=structure(b, side), eval=ev,
                            result=result_white if side == chess.WHITE else -result_white))
        if ply < len(moves):
            b.push(chess.Move.from_uci(moves[ply]))
    return out


_ENGINE = None


def _init_worker(depth: int) -> None:
    """One Stockfish per worker process, created after the fork."""
    global _ENGINE, _DEPTH
    import chess.engine
    _ENGINE = chess.engine.SimpleEngine.popen_uci("stockfish")
    _ENGINE.configure({"Threads": 1, "Hash": 32})
    _DEPTH = depth


def _work_game(row: dict) -> tuple:
    """One game: -> (game_id, rows). The unit of checkpointing (Kaveh 2026-10-07: never lose finished work)."""
    return row["game_id"], _game_rows(row, _ENGINE, _DEPTH)


def select_games(shard: int, n: int, lo: int = 1500, hi: int = 2200, min_base: int = 180, min_plies: int = 40) -> list:
    """The first n games of the shard, by game id, with both players rated in [lo, hi], a base time of at least
    min_base seconds, a normal termination and at least min_plies plies."""
    import pyarrow.parquet as pq
    t = pq.read_table(paths.REPO_ROOT / "data" / "records" / "lichess_2019-01" / f"records_{shard:05d}.parquet",
                      columns=["game_id", "white_elo", "black_elo", "result", "n_plies", "time_control", "termination", "moves"])
    out = []
    for r in sorted(t.to_pylist(), key=lambda r: int(r["game_id"])):
        try:
            we, be, base = int(r["white_elo"]), int(r["black_elo"]), int(str(r["time_control"]).split("+")[0])
        except ValueError:
            continue
        if lo <= we <= hi and lo <= be <= hi and base >= min_base and r["termination"] == "Normal" and int(r["n_plies"]) >= min_plies:
            out.append(r)
            if len(out) >= n:
                break
    return out


def select_bands(shard: int, bands: list, per_band: int) -> list:
    """The first per_band games of each rating band (both players inside it), concatenated."""
    out = []
    for lo, hi in bands:
        out += select_games(shard, per_band, lo, hi)
    return out


def census_path(tag: str) -> Path:
    return Path(paths.derived("tactic_census", f"census_{tag}.jsonl"))


def reconcile(out: Path, done_path: Path) -> set:
    """The games finished so far: the .done list; rows of a game cut off before its done-mark are dropped from the rows
    file. A rows file with NO .done beside it predates checkpointing and is complete: its games are taken as done and
    the list written (review 2026-10-07: it was reconciled against an empty list and wiped)."""
    if not done_path.exists() and out.exists() and out.stat().st_size > 0:
        legacy = sorted({str(json.loads(line)["game"]) for line in open(out)})
        done_path.write_text("".join(f"{g}\n" for g in legacy))
        print(f"[census] no .done beside {out.name}: its {len(legacy)} games taken as finished and listed", flush=True)
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    if out.exists():
        kept, dropped = [], set()
        for line in open(out):
            gid = str(json.loads(line)["game"])
            if gid in done:
                kept.append(line)
            else:
                dropped.add(gid)
        if dropped:
            open(out, "w").writelines(kept)
            print(f"[census] reconciled: dropped the rows of {len(dropped)} unfinished game(s) {sorted(dropped)[:5]}", flush=True)
    return done


def build(games: list, workers: int, tag: str, depth: int = 10, label: str = "") -> None:
    """Checkpointed: every finished game's rows are appended to the output at once and its id to the .done list; a
    restart with the same tag skips the done games (no loss, no duplicates). Progress lines carry the clock."""
    t0 = time.time()
    out = census_path(tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_path = out.with_suffix(".done")
    done = reconcile(out, done_path)
    seen = set(); todo = []
    for g in games:                                                    # deduplicated: a game on a shared band edge was selected twice
        gid = str(g["game_id"])
        if gid not in done and gid not in seen:
            seen.add(gid); todo.append(g)
    print(f"[census] {tag}: {len(games)} games selected, {len(done)} done before, {len(todo)} to do; {workers} workers", flush=True)
    finished = len(done); rows = 0
    with Pool(workers, initializer=_init_worker, initargs=(depth,)) as pool, open(out, "a") as fh, open(done_path, "a") as dh:
        for gid, part in pool.imap_unordered(_work_game, todo, chunksize=1):
            for r in part:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            dh.write(f"{gid}\n"); dh.flush()
            finished += 1; rows += len(part)
            if finished % 10 == 0 or finished == len(games):
                print(f"[census] {time.strftime('%H:%M:%S')} {finished}/{len(games)} games, {rows} new positions, {time.time() - t0:.0f}s", flush=True)
    total = sum(1 for _ in open(out))
    print(f"OBS census {tag}: {len(games)} games ({label or 'both rated 1500-2200'}, base >= 180 s, normal end, >= 40 plies), "
          f"{total} positions in the file; this run {rows} positions in {time.time() - t0:.0f}s -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "measure"])
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--shard", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--bands", default=None, help="rating bands 'lo-hi,lo-hi,...' with --games per band (both players inside)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if a.bands:
            bands = [tuple(int(x) for x in b.split("-")) for b in a.bands.split(",")]
            build(select_bands(a.shard, bands, a.games), a.workers, a.tag, a.depth, label=f"{a.games} games per band {a.bands}")
        else:
            build(select_games(a.shard, a.games), a.workers, a.tag, a.depth)
    else:
        from .tactic_census_measure import measure
        measure(a.tag)


if __name__ == "__main__":
    main()
