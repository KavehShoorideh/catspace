"""plan 76.4c THE HARNESS: mining objects and material (docs/ANALYST_PLAN.md 76.4 and 76's borrowings (4) and (5)).

  .venv/bin/python scripts/plan76_4c.py atoms [--control]
        (a) THE ATOMS AS TERMS: on the guard positions (ten + wide, the case lists of scripts/trace_identity.py) and the surprise shelf, every technique atom read by
        python-chess (technique_measures.atom_values) against its term of the one term language (terms.py, technique_measures.ATOM_TERMS) read through the chess
        reader, both sides as the side read for; and the chess reader against the GDL oracle reader (gdl.py's interpreter) on the same terms. `--control`: a
        deliberately wrong build (the free squares read off THEIR attack) -- the comparison must fire; `--unlifted`: their king not lifted (a wrong build
        the guard positions cannot see: none has the side read for's opponent in check along a line with a free square behind its king).
Outputs under artifacts/experiments/plan76_4c/ (untracked)."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]


def _trace_identity():
    """The guard's case lists (scripts/trace_identity.py is a script, not a module of the package: loaded by its path)."""
    spec = importlib.util.spec_from_file_location("trace_identity", ROOT / "scripts" / "trace_identity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def guard_positions() -> list:
    """(set, id, fen, us) of the ten, the wide set and the surprise shelf."""
    TI = _trace_identity()
    out = []
    for sets in ("ten", "wide", "surprises"):
        for pid, fen, us in TI._cases(sets):
            out.append((sets, pid, fen, us))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ (a)

def atoms(control: bool = False) -> int:
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import technique_measures as TM
    from catspace.incubator.analyst.discovery import terms as TL
    t0 = time.monotonic()
    lang = TM.term_language()
    dom, reader, defs, terms = lang
    if control == "unlifted":                       # a control the guard positions cannot see: their king not lifted from the board our attack is read on
        defs.body["free"] = TL.parse("(not (attacked us ?))", dom, defs)
    elif control:                                   # NEGATIVE CONTROL: the free squares read off the other side's attack
        defs.body["free"] = TL.parse("(not (without L (attacked them ?)))", dom, defs)
    oracle = TL.GDLReader(GC.program(), GC.inputs_of())
    pos = guard_positions()
    by = {k: 0 for k in terms}
    bad = []
    n = 0
    for sets, pid, fen, us in pos:
        b = chess.Board(fen)
        for side in (chess.WHITE, chess.BLACK):
            if b.king(side) is None:
                continue
            n += 1
            hand = TM.atom_values(b, side)
            got = TM.term_atom_values(b, side, lang)
            for k in terms:
                if hand[k] == got[k]:
                    by[k] += 1
                else:
                    bad.append((sets, pid, fen, side, k, hand[k], got[k]))
    print(f"OBS 76.4c (a) atoms as terms: positions {len(pos)} (ten, wide, surprise shelf), readings {n} (each side as the side read for), atoms {len(terms)}; "
          f"equal to python-chess's reading: " + ", ".join(f"{k} {v}/{n}" for k, v in by.items()) + f"; differences {len(bad)}")
    for x in bad[:40]:
        print(f"  DIFF {x[0]} {x[1]} {x[2]} side {'w' if x[3] else 'b'}: {x[4]} hand {x[5]} term {x[6]}")
    # the chess reader against the GDL oracle (a slower reading: the interpreter), the same terms
    ok = agree = 0
    for sets, pid, fen, us in pos:
        b = chess.Board(fen)
        st = GC.state_of(b)
        for side in (chess.WHITE, chess.BLACK):
            if b.king(side) is None:
                continue
            c1 = TL.Ctx(dom, reader, b, GC.COLOUR[side], defs)
            c2 = TL.Ctx(dom, oracle, st, GC.COLOUR[side], defs)
            same = all(c1.ev(t) == c2.ev(t) for t in terms.values())
            ok += 1
            agree += same
    print(f"OBS 76.4c (a) readers: the chess reader (the native core's prover) against the GDL oracle (gdl.py) on the same {len(terms)} terms: {agree} of {ok} readings "
          f"agree on every term; sizes " + ", ".join(f"{k} {t.size}" for k, t in terms.items()) + f"; {time.monotonic() - t0:.1f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "atoms":
        sys.exit(atoms(control=("unlifted" if "--unlifted" in a else "--control" in a)))
    sys.exit(__doc__)
