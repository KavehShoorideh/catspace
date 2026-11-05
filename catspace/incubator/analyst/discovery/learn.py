"""PLAN 78.4 THE LEARNING COMMAND: game knowledge mined from instances of the game, the loops in Rust (cs_learn through cs_engine). Prints what it finds; writes a
store entry ONLY with --write (Kaveh 2026-11-04: the product is chess; the capability to learn another game's knowledge is built, not exercised).

  .venv/bin/python -m catspace.incubator.analyst.discovery.learn valuation chess --lichess 4000          # the lichess records 76.4c read (games 0..N-1)
  .venv/bin/python -m catspace.incubator.analyst.discovery.learn valuation <game> --instances [name]     # data/derived/instances/<game>/ (watched or played)
  .venv/bin/python -m catspace.incubator.analyst.discovery.learn watch <game> --games 100 [--name x]     # instances by watching an outside engine (Fairy-Stockfish)
  ... add --write to file the mined valuation as a MINED VALUATION entry scoped to the game
"""
from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

import cs_engine as CE
from catspace.incubator.analyst.discovery import instance as IN
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.io import paths

FROM_PLY = 10           # the first plies of a record are left out (76.4c's filter: the records' `skip_first_plies`)
FUSE = 64               # the price the integer search gives up at (the board's locations)
TEACHER = "/opt/homebrew/bin/fairy-stockfish"


def lichess_records(n: int) -> tuple:
    import pyarrow.parquet as pq
    f = Path(paths.data_dir()) / "records" / "lichess_2019-01" / "records_00000.parquet"
    t = pq.read_table(f, columns=["result", "moves"]).slice(0, n).to_pylist()
    return [("", r["moves"].split(), {1: [100, 0], -1: [0, 100], 0: [50, 50]}[int(r["result"])]) for r in t], f"lichess 2019-01 records_00000 games 0..{n - 1}"


def mine(game: str, records: list, what: str) -> dict:
    t = time.time()
    f = CE.mine_valuation(game, records, FROM_PLY, FUSE)
    f["seconds"] = round(time.time() - t, 1)
    f["what"] = what
    w = dict(zip(f["kinds"], f["worth"] or []))
    print(f"OBS valuation of {game} mined from {what}: positions {f['positions']}, games {f['games']}; continuous " +
          ", ".join(f"{k} {b:.4f}" for k, b in zip(f["kinds"], f["beta"])) + f" (log loss {f['logloss_best']:.5f}); "
          + (f"worth {w} (price {f['price']}, log loss {f['logloss']:.5f}, + {f['excess']:.5f} within the paired error {f['se']:.5f})" if f["worth"] else "no integer worths within one paired error by the fuse")
          + f"; constant (no weight) {f['constant']}; {f['seconds']} s")
    return f


def entry_of(game: str, f: dict) -> dict:
    """The MINED VALUATION entry scoped to `game` (the royal: the one constant kind, if any)."""
    w = dict(zip(f["kinds"], f["worth"]))
    royal = f["constant"][0] if len(f["constant"]) == 1 else None
    if royal is not None:
        w[royal] = 0
    unit = min(abs(x) for x in w.values() if x) if any(w.values()) else 1
    name = f"material ({game}, mined)"
    return dict(name=name, kind="VALUATION", reads=["material"], terms=[], status="HEURISTIC", lean=[], origin="MINED", plan=["78.4"], journal=datetime.date.today().isoformat(),
                measure=None, evidence={"uses": {}, "confirmed": [f"plan 78.4: fit only to outcomes -- {f['what']}: {f['positions']} positions of {f['games']} games; price {f['price']}, "
                                                                 f"log loss {f['logloss']:.5f} (+{f['excess']:.5f} within {f['se']:.5f})"]},
                implementation=["rust/engine/learn/src/valuation.rs:search"], note=f"The valuation of {game} mined from outcomes alone (cs_learn).", game=game,
                params=dict(worth=w, unit=unit, royal=royal if royal is not None else min(w), term=f"(sub (wcount 1 {name} (cell ? us ?)) (wcount 1 {name} (cell ? them ?)))",
                            obligation="stated, not proved: the exchange law restated with these worths (as material (mined)'s obligation)"))


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd, game = argv[0], argv[1]
    if cmd == "watch":
        n = int(argv[argv.index("--games") + 1]) if "--games" in argv else 100
        name = argv[argv.index("--name") + 1] if "--name" in argv else "watched"
        t = time.time()
        recs = CE.watch(game, TEACHER, n, 6, 20, 78, 11)
        insts = [IN.Instance(game=game, start=s, moves=m, outcome=o, players={"white": "fairy-stockfish", "black": "fairy-stockfish"}, source="watch") for s, m, o in recs]
        f = IN.save(insts, game, name)
        print(f"OBS watched {len(insts)} games of {game} (finished {sum(1 for i in insts if i.outcome)}), {time.time() - t:.0f} s -> {f}")
        return 0
    if cmd == "valuation":
        if "--lichess" in argv:
            recs, what = lichess_records(int(argv[argv.index("--lichess") + 1]))
        else:
            nm = argv[argv.index("--instances") + 1] if "--instances" in argv and argv.index("--instances") + 1 < len(argv) and not argv[argv.index("--instances") + 1].startswith("--") else None
            recs = [i.record() for i in IN.load(game, nm)]
            what = f"data/derived/instances/{game}/{nm or '*'}.jsonl"
        f = mine(game, recs, what)
        if "--write" in argv and f["worth"]:
            e = entry_of(game, f)
            p = LS.DIR / (LS.slug(e["name"]) + ".json")
            p.write_text(json.dumps(e, indent=1, ensure_ascii=True) + "\n")
            print(f"OBS written {p}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
