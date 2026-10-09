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


def run(per_theme: int) -> None:
    from multiprocessing import Pool
    rows = collect(per_theme)
    jobs = [(t, r) for t, rs in rows.items() for r in rs]
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-theme", type=int, default=150)
    a = ap.parse_args(argv)
    run(a.per_theme)


if __name__ == "__main__":
    main()
