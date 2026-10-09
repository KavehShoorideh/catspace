"""THE LEMMAS AGAINST ANNOTATED DATA (Kaveh 2026-10-08: "use annotated chess data sources for advantage"): the Lichess
puzzle database tags every puzzle with its themes (fork, pin, trappedPiece, mateIn2, discoveredAttack, deflection,
intermezzo, quietMove, ...) and gives the winning line -- named ground truth for the lemma registry.

For each theme, up to N puzzles (one pass over the 6.1M-puzzle file, rows disjoint from the earlier benchmarks: the
test sample i % 997 == 0 and the proposer's / quiet sets i % 97 in 49, 73 are skipped). On the puzzle's position (after
the opponent's first move), the SOLUTION's first move m:
  claims     what the rule layer (analyst_core.threat_claims: Ascent rules + the mirrored certificate) says m created
  certified  some claim on m is certified by the kernel's localCheck (proved sound: a forced gain three plies on)
  mates      m is checkmate (a mate needs no claim)
  found      the frame with all our lemmas (forcing + double threat + certified threat moves) values m within half a
             pawn of its best legal move: our lemmas would play the puzzle's move
Per theme: the coverage, the names our registry gives, and examples of the misses -- the themes least covered are
where the next lemmas are needed.

    python -m catspace.incubator.analyst.lemma_puzzles --per-theme 150
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import chess

THEMES = ("fork", "pin", "skewer", "discoveredAttack", "doubleCheck", "trappedPiece", "hangingPiece", "mateIn1", "mateIn2",
          "mateIn3", "backRankMate", "smotheredMate", "deflection", "attraction", "interference", "intermezzo", "quietMove",
          "defensiveMove", "clearance", "xRayAttack", "capturingDefender", "exposedKing", "promotion", "advancedPawn",
          "zugzwang", "sacrifice")


def collect(per_theme: int, stride: int = 7, offset: int = 3) -> dict:
    """{theme: [rows]} in one pass over the puzzle file."""
    import csv, io, zstandard
    from catspace.io import paths
    out = {t: [] for t in THEMES}
    with open(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst", "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % stride != offset or i % 997 == 0 or i % 97 in (49, 73):
                continue
            for t in row["Themes"].split():
                if t in out and len(out[t]) < per_theme:
                    out[t].append(row)
            if all(len(v) >= per_theme for v in out.values()):
                break
    return out


def _one(args) -> dict:
    theme, row = args
    import analyst_core
    from .exchange_mine import _lemma_valuer
    b = chess.Board(row["FEN"]); mv = row["Moves"].split()
    b.push_uci(mv[0])
    m = chess.Move.from_uci(mv[1])
    nb = b.copy(stack=False); nb.push(m)
    cl = [c for c in analyst_core.threat_claims(b.fen()) if c[0] == m.uci()]
    val = _lemma_valuer()
    vals = {x: val(b, x) for x in b.legal_moves}
    best = max(vals.values()); top = [x for x, v in vals.items() if v >= best - 0.5]
    return dict(theme=theme, id=row["PuzzleId"], fen=b.fen(), move=b.san(m), rating=int(row["Rating"]),
                kind="capture" if b.is_capture(m) else "check" if b.gives_check(m) else "quiet",
                names=sorted({c[1] for c in cl}), certified=any(c[3] for c in cl), framed=any(c[5] for c in cl),
                cert_names=sorted({c[1] for c in cl if c[3] or c[5]}), mates=nb.is_checkmate(),
                found=m in top and len(top) <= 2, top=len(top))   # review 2026-10-08: "within 0.5 of the best" counted every
                                                                # move when the frames see nothing (all tie at 0)


def run(per_theme: int, only: str | None = None) -> None:
    from multiprocessing import Pool
    rows = collect(per_theme)
    jobs = [(t, r) for t, rs in rows.items() for r in rs if only is None or t == only]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        recs = list(pool.imap_unordered(_one, jobs, chunksize=8))
    by = defaultdict(list)
    for r in recs:
        by[r["theme"]].append(r)
    print(f"OBS lemma coverage on {len(recs)} Lichess puzzles ({per_theme} per theme where available; the solution's first move): "
          f"theme, n, kind of move (capture/check/quiet), our lemmas FIND it (in a top set of at most 2 moves within 0.5 of the best), a claim CERTIFIED on it (three plies), "
          f"or by the FRAME-SIZED certificate, "
          f"only MATCHED, it mates; the names our registry gives")
    for t in sorted(by, key=lambda t: sum(r["found"] for r in by[t]) / len(by[t])):
        g = by[t]; n = len(g)
        kinds = Counter(r["kind"] for r in g)
        f = sum(r["found"] for r in g); c = sum(r["certified"] for r in g); mt = sum(bool(r["names"]) and not r["framed"] for r in g)
        fr = sum(r["framed"] for r in g)
        import statistics
        tops = statistics.median(r["top"] for r in g)
        mate = sum(r["mates"] for r in g)
        names = Counter(nm for r in g for nm in r["cert_names"])
        print(f"OBS   {t:18s} n {n:4d} ({kinds['capture']}/{kinds['check']}/{kinds['quiet']}): found {100 * f / n:3.0f}% (top set median {tops:.0f}), "
              f"certified {100 * c / n:3.0f}%, frame-sized {100 * fr / n:3.0f}%, matched only {100 * mt / n:3.0f}%, mates {100 * mate / n:3.0f}%; as "
              + (", ".join(f"{k} {v}" for k, v in names.most_common(3)) or "-"))
    print("OBS   misses (not found, not mating), two per theme for reading:")
    for t in THEMES:
        for r in [x for x in by.get(t, []) if not x["found"] and not x["mates"]][:2]:
            print(f"OBS     {t}: {r['fen']} {r['move']} (rating {r['rating']}; {r['kind']}; names {r['names']})")




# ---------------------------------------------------------------------------------------------------------------------
# DIGGING THE FAILURES ONE BY ONE (Kaveh 2026-10-08: "follow along with Stockfish's line and at every point guess the
# next move from our lemmas ... follow it down a few lines and see how it ends up; follow your own line a few lines
# down ... then you understand the difference -- and that is the lemma you need"). For every puzzle of a theme our
# lemmas do not find: the solution move against our pick (values, claims, Stockfish after each); the solution line
# played out; OUR line (both sides playing our lemmas' top move) for six plies; Stockfish at the end of each.

_ENG = None


def _eng_init():
    global _ENG
    import chess.engine
    _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")


def _sf(b: chess.Board, pov: chess.Color) -> float:
    import chess.engine
    if b.is_checkmate():
        return -20.0 if b.turn == pov else 20.0
    s = _ENG.analyse(b, chess.engine.Limit(depth=14))["score"].pov(pov)
    return max(-20.0, min(20.0, s.score(mate_score=2000) / 100))


def _mat(b: chess.Board, side: chess.Color) -> int:
    V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    return sum(V[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values())


def _dig_one(args) -> str:
    theme, row = args
    import analyst_core
    from .exchange_mine import _lemma_valuer
    val = _lemma_valuer()
    b = chess.Board(row["FEN"]); mv = row["Moves"].split()
    b.push_uci(mv[0])
    side = b.turn
    sol = [chess.Move.from_uci(u) for u in mv[1:]]
    vals = {x: val(b, x) for x in b.legal_moves}
    ours = max(vals, key=vals.get)
    claims = lambda bb, m: [f"{c[1]} +{c[2]}{' CERT' if c[3] else ' framed' if c[5] else ''}" for c in analyst_core.threat_claims(bb.fen()) if c[0] == m.uci()]
    out = [f"=== {row['PuzzleId']} ({theme}, rating {row['Rating']}) {b.fen()}  {'White' if side else 'Black'} to move"]
    for tag, m in (("solution", sol[0]), ("ours", ours)):
        nb = b.copy(); nb.push(m)
        out.append(f"  {tag:8s} {b.san(m):7s} our value {vals[m]:+.0f}; claims {claims(b, m) or '-'}; Stockfish after {_sf(nb, side):+.1f}")
    top = sorted(vals.items(), key=lambda kv: -kv[1])[:5]
    out.append("  our top five: " + ", ".join(f"{b.san(x)} {v:+.0f}" for x, v in top))
    bb = b.copy(); line = []
    for m in sol:
        line.append(f"{bb.san(m)}[{_mat(bb, side):+d}]"); bb.push(m)
    out.append(f"  SOLUTION line: {' '.join(line)} -> material {_mat(bb, side):+d}, Stockfish {_sf(bb, side):+.1f}")
    bb = b.copy(); line = []
    for i in range(6):
        if bb.is_game_over():
            break
        v2 = _lemma_valuer()
        vs = {x: v2(bb, x) for x in bb.legal_moves}
        x = max(vs, key=vs.get)
        line.append(f"{bb.san(x)}[{vs[x]:+.0f}]"); bb.push(x)
    out.append(f"  OUR line:      {' '.join(line)} -> material {_mat(bb, side):+d}, Stockfish {_sf(bb, side):+.1f}")
    return "\n".join(out)


def dig(theme: str, per_theme: int = 150, path: str | None = None) -> None:
    from multiprocessing import Pool
    rows = collect(per_theme)[theme]
    recs_needed = []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        recs = list(pool.imap(_one, [(theme, r) for r in rows], chunksize=4))
    for r, row in zip(recs, rows):
        if not r["found"] and not r["mates"]:
            recs_needed.append(row)
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_eng_init) as pool:
        texts = list(pool.imap(_dig_one, [(theme, r) for r in recs_needed], chunksize=1))
    body = "\n".join(texts)
    if path:
        open(path, "w").write(body + "\n")
    print(f"OBS {theme}: {len(rows)} puzzles, {len(recs_needed)} not found by our lemmas; reports -> {path}")


def _reasoned(args) -> dict:
    theme, row = args
    from .target_reasoner import reason
    b = chess.Board(row["FEN"]); mv = row["Moves"].split(); b.push_uci(mv[0])
    r = reason(b)
    return dict(theme=theme, id=row["PuzzleId"], solved=r["move"] is not None and r["move"].uci() == mv[1],
                answered=r["move"] is not None, ms=r["ms"], target=r["target"], gain=r["gain"],
                defended=bool(r.get("defend_first")), move=b.san(r["move"]) if r["move"] else None, sol=b.san(chess.Move.from_uci(mv[1])),
                fen=b.fen())


def reasoner_coverage(per_theme: int, only: str | None, offset: int = 3) -> None:
    """The target-first reasoner as the move chooser: its move equals the puzzle's first move (solved), it gave a
    certified move at all (answered), and its time per puzzle."""
    import statistics
    from multiprocessing import Pool
    rows = collect(per_theme, offset=offset)
    keep = set(only.split(",")) if only else None
    jobs = [(t, r) for t, rs in rows.items() for r in rs if keep is None or t in keep]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        recs = list(pool.imap_unordered(_reasoned, jobs, chunksize=2))
    by = defaultdict(list)
    for r in recs:
        by[r["theme"]].append(r)
    import json
    json.dump(recs, open(os.environ.get("CATSPACE_REASONER_DUMP", "/tmp/reasoner_last.json"), "w"))
    for t, g in sorted(by.items()):
        ms = sorted(r["ms"] for r in g)
        print(f"OBS reasoner {t:18s} n {len(g)}: solved {100 * sum(r['solved'] for r in g) / len(g):.0f}%, answered {100 * sum(r['answered'] for r in g) / len(g):.0f}%; "
              f"time median {ms[len(ms) // 2]} ms, 90th {ms[int(0.9 * len(ms))]} ms, max {ms[-1]} ms")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="coverage", choices=["coverage", "dig", "reasoner"])
    ap.add_argument("--per-theme", type=int, default=150)
    ap.add_argument("--theme", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--offset", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "reasoner":
        reasoner_coverage(a.per_theme, a.theme, a.offset)
    elif a.cmd == "dig":
        dig(a.theme or "trappedPiece", a.per_theme, a.out)
    else:
        run(a.per_theme, a.theme)


if __name__ == "__main__":
    main()
