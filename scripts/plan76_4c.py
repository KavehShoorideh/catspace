"""plan 76.4c THE HARNESS: mining objects and material (docs/ANALYST_PLAN.md 76.4 and 76's borrowings (4) and (5)).

  .venv/bin/python scripts/plan76_4c.py atoms [--control]
        (a) THE ATOMS AS TERMS: on the guard positions (ten + wide, the case lists of scripts/trace_identity.py) and the surprise shelf, every technique atom read by
        python-chess (technique_measures.atom_values) against its term of the one term language (terms.py, technique_measures.ATOM_TERMS) read through the chess
        reader, both sides as the side read for; and the chess reader against the GDL oracle reader (gdl.py's interpreter) on the same terms. `--control`: a
        deliberately wrong build (the free squares read off THEIR attack) -- the comparison must fire; `--unlifted`: their king not lifted (a wrong build
        the guard positions cannot see: none has the side read for's opponent in check along a line with a free square behind its king).
  .venv/bin/python scripts/plan76_4c.py objects [--levels 4]
        (b) MINING OBJECTS: the regression's readings on the guard positions, the obligations' cited sets, the generator's levels (object_miner.py); the three test
        cases (LINE, BLOCKED, REGION) against the accepted relations -> artifacts/experiments/plan76_4c/objects.json
  .venv/bin/python scripts/plan76_4c.py entries <objects.json> <date>      (b) the accepted relations as ontology entries (origin MINED)
  .venv/bin/python scripts/plan76_4c.py class-miner <objects.json>          (b) (i) the KRvK whole-class miner with the mined atoms
  .venv/bin/python scripts/plan76_4c.py material [--games N] [--per-sig N]  (c) the valuation miner on the game records and the table-judged endings
  .venv/bin/python scripts/plan76_4c.py valuation-entry <material.json> <date>   (c) the mined VALUATION entry
  .venv/bin/python scripts/plan76_4c.py swap mined|hand                     (c)'s one experiment: the GOAL entries count in the mined valuation (never committed)
  .venv/bin/python scripts/plan76_4c.py decisions <parent dir> <after dir>  (c) the guard's DECISION lines that differ
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


# ------------------------------------------------------------------------------------------------------------------------------------------ (b)

def class_positions(sig: str, n: int, seed: int) -> list:
    """`n` random legal placements of a material class (technique_corpus.random_start's draw, no table: every legal placement with the winning side -- White -- to
    move, not in check, not over)."""
    import random
    from catspace.incubator.analyst.discovery import technique_corpus as TC
    ours, theirs = TC.parse_signature(sig)
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        sq = rng.sample(range(64), len(ours) + len(theirs))
        b = chess.Board(None)
        for s, ch in zip(sq[:len(ours)], ours):
            b.set_piece_at(s, chess.Piece(TC.UNITS[ch], chess.WHITE))
        for s, ch in zip(sq[len(ours):], theirs):
            b.set_piece_at(s, chess.Piece(TC.UNITS[ch], chess.BLACK))
        b.turn = chess.WHITE
        if b.is_valid() and not b.is_check() and not b.is_game_over():
            out.append(b)
    return out


def _cases_for_harvest(positions: list) -> list:
    """(state, side, goals) per guard position and per its turn passed: the store's mate goal on the state as it is, the win goal of the side to move on every
    unit of the other side but the royal one (regression.goal_of over the store's GOAL params, the binding's terms)."""
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import lemma_store as LS
    from catspace.incubator.analyst.discovery import regression as RG
    B = GC.Binding()
    gg = LS.gdl_goals()
    unit = LS.goal_kinds().valuation.unit
    out = []
    for b in positions:
        for x in (b, B.pass_(b)):
            me = B.role(x.turn)
            goals = [RG.goal_of("mate", gg["mate"], B, me)]
            for s in chess.scan_forward(x.occupied_co[not x.turn] & ~x.kings):
                goals.append(RG.goal_of("win", gg["win"], B, me, target=s, unit=unit))
            out.append((x, me, goals))
    return out


def spec_relation(name: str) -> "str | None":
    """The spec relation a shelf entry binds (a fluent as `true.<fluent>`), or None."""
    from catspace.incubator.analyst.discovery import ontology as ON
    g = ON.get(name).get("gdl") or {}
    if "slot" in g or "constant" in g or "args" in g:
        return None
    return g.get("relation") or (("true." + g["fluent"]) if "fluent" in g else None)


def objects(fuse: "int | None" = None, levels: int = 4, control: bool = False):
    """(b) MINING OBJECTS: the regression's readings on the guard positions, the obligations' cited sets on the class's positions, the generator's levels."""
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import lemma_store as LS
    from catspace.incubator.analyst.discovery import object_miner as OM
    from catspace.incubator.analyst.discovery import ontology as ON
    from catspace.incubator.analyst.discovery import regression as RG
    from catspace.incubator.analyst.discovery import technique_measures as TM
    from catspace.incubator.analyst.discovery import terms as TL
    t0 = time.monotonic()
    prog = GC.program()
    dom = TL.Domain(prog, ON.load())
    reader = GC.TermReader()
    pos = [chess.Board(f) for _s, _p, f, _u in guard_positions()]
    R = RG.PyRegressor(RG.Spec(prog, certified=GC.CERTIFIED), GC.Binding())
    readings = OM.harvest(R, dom, _cases_for_harvest(pos))
    # the obligations' cited sets: every TECHNIQUE entry's measure, its set sub-terms the technique library reads (technique_measures.SET_READINGS), on 200 random
    # positions of its class and on the guard positions (each side read for: the vocabulary's term means the same off the class); the term's reads through the
    # shelf (ontology.term_relations) are the test's alphabet
    cited = {}
    for name, e in sorted(LS.load().items()):
        m = e.get("measure") or {}
        if not m.get("obligation") or not m.get("class"):
            continue
        cls = [(b, chess.WHITE) for b in class_positions(m["class"], 200, 76)]
        cls += [(b, side) for b in pos for side in (chess.WHITE, chess.BLACK) if b.king(not side) is not None]      # the term's meaning off the class too
        for sub, fn in TM.SET_READINGS.items():
            if sub in m["quantity"]:
                rels = [spec_relation(r) for r in ON.term_relations(sub)]
                cited[f"{name}: {sub}"] = ([r for r in rels if r], [(b, GC.COLOUR[side]) for b, side in cls], [set(chess.scan_forward(fn(b, side))) for b, side in cls])
    tests = OM.tests_of(dom, readings, cited)
    for t in tests:
        print(f"OBS objects test {t.name}: alphabet {sorted(t.alphabet)}")
    krvk = class_positions("KRvK", 6, 77)
    sample = [(b, GC.COLOUR[b.turn]) for b in pos[:10]] + [(b, "white") for b in krvk] + [readings.states[f[1]] for f in readings.flips[:6]]
    print(f"OBS objects the levels: at most {levels} (the run's cost: a fifth level's search of the still open tests to the fuse ran past 15 minutes at size 6 "
          f"on 2026-10-29 -- the tests it leaves open are GAPs, not refutations)")
    fz = fuse or OM.body_fuse(prog)
    print(f"OBS objects the fuse: {fz} (the largest clause body of rules/chess.gdl{'' if fuse is None else ', overridden'}); sample {len(sample)} states; tests {len(tests)}")
    got = OM.mine(dom, reader, sample, readings, tests, fz, max_levels=levels)
    print(f"OBS objects search: candidates tried {got['tried']}, citation tests read {got['tested']}, pruned by (4) {got['pruned']}, dropped {got['dropped']}, "
          f"accepted {len(got['accepted'])}; tests explained {len(got['explained'])} of {len(tests)} (open: {', '.join(got['open']) or 'none'}); {time.monotonic() - t0:.1f} s")
    return got, dom, reader, readings


# THE THREE TEST CASES (the oracle of (b): read only here, after the search -- the generator never reads them), python-chess's readings of what we mean:
#   LINE     (a, b, c): c strictly between two squares a and b on one rank, file or diagonal (chess.between: the squares a sliding unit passes on an empty board)
#   BLOCKED  (a, b): a and b on one line, an occupied square strictly between them
#   REGION   the squares their king reaches by king steps without stepping onto a square we attack (technique_measures.region: the 73.5 atom, their king lifted)
REFERENCES = ("line", "blocked", "region")


def _ref_line() -> frozenset:
    return frozenset((a, b, c) for a in chess.SQUARES for b in chess.SQUARES if a != b and chess.BB_RAYS[a][b] for c in chess.scan_forward(chess.between(a, b)))


def _ref_blocked(board: chess.Board) -> frozenset:
    return frozenset((a, b) for a in chess.SQUARES for b in chess.SQUARES if a != b and chess.BB_RAYS[a][b] and chess.between(a, b) & board.occupied)


def rediscover(got: dict, dom, reader, log=print) -> dict:
    """Each accepted relation against the three references, by its sort, on every guard position (ten, wide, the surprise shelf), each side read for: -> {reference:
    [accepted names equivalent to it]}; prints OBS."""
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import technique_measures as TM
    from catspace.incubator.analyst.discovery import terms as TL
    pos = [chess.Board(f) for _s, _p, f, _u in guard_positions()]
    line = _ref_line()
    found = {r: [] for r in REFERENCES}
    for a in got["accepted"]:
        t = a["term"]
        sort = t.sort
        nloc = sum(1 for s in sort if s == TL.LOC) if isinstance(sort, tuple) else 0
        ref = {3: "line", 2: "blocked"}.get(nloc) if sort != TL.LOCS else ("region" if any("(reach L free)" in x for x in a["tests"]) else None)
        if ref is None:
            continue                                              # (a set cited for its own obligation term -- free, corner -- equals it by the test itself)
        same = n = 0
        first = None
        for b in pos:
            for side in (chess.WHITE, chess.BLACK):
                if b.king(not side) is None:
                    continue
                ctx = TL.Ctx(dom, reader, b, GC.COLOUR[side], got["defs"])
                v = ctx.ev(t)
                if ref == "line":
                    mine, want = frozenset((r[-3], r[-2], r[-1]) for r in v), line
                elif ref == "blocked":
                    mine, want = frozenset((r[-2], r[-1]) for r in v), _ref_blocked(b)
                else:
                    mine, want = frozenset(r[0] for r in v), frozenset(chess.scan_forward(TM.region(b, side)))
                n += 1
                same += mine == want
                if mine != want and first is None:
                    first = (b.fen(), "w" if side else "b", sorted(mine - want)[:4], sorted(want - mine)[:4])
        a["vs"] = (ref, same, n, first)
        log(f"OBS objects test case {ref.upper()} against {a['name']} := {t.sexpr()}: equal on {same} of {n} readings (the guard positions, each side read for)"
            + (f"; first difference {first[0]} {first[1]}: the mined only {first[2]}, the reference only {first[3]}" if first else ""))
        if same == n:
            found[ref].append(a["name"])
    log("OBS objects the three test cases: " + "; ".join(f"{r.upper()} {'FOUND as ' + ', '.join(found[r]) if found[r] else 'NOT FOUND'}" for r in REFERENCES))
    return found


def class_miner(region_term: str, corner_term: "str | None", defs_terms: list, sig: str = "KRvK", log=print) -> dict:
    """(b) (i) THE TECHNIQUE MINER RE-RUN WITH THE DERIVED RELATIONS ADMITTED AS ATOMS: technique_miner over EVERY position of the class (73.5b's whole-class
    criterion, its library), the region atoms (count, off the edge, area) and the corner distance read from the mined terms through the term language in place of
    the hand readings -> the miner's result (its first zero-failure level, every measure tied there) and how many of those cite the derived atoms."""
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import ontology as ON
    from catspace.incubator.analyst.discovery import technique_measures as TM
    from catspace.incubator.analyst.discovery import technique_miner as TMI
    from catspace.incubator.analyst.discovery import technique_sample as TS
    from catspace.incubator.analyst.discovery import terms as TL
    t0 = time.monotonic()
    dom = TL.Domain(GC.program(), ON.load())
    reader = GC.TermReader()
    defs = TL.Defs()
    for name, body in defs_terms:
        defs.define(name, TL.parse(body, dom, defs))
    reg = TL.parse(region_term, dom, defs)
    cor = TL.parse(corner_term, dom, defs) if corner_term else None
    orig = TM.atom_values
    calls = [0, 0]

    def atom_values(board, side, _orig=orig):
        v = _orig(board, side)
        ctx = TL.Ctx(dom, reader, board, GC.COLOUR[side], defs)
        m = 0
        for r in ctx.ev(reg):
            m |= 1 << r[0]
        v["region"], v["region_off_edge"], v["region_area"] = chess.popcount(m), chess.popcount(m & ~TM.EDGE), TM.area(m)
        if cor is not None:
            c = 0
            for r in ctx.ev(cor):
                c |= 1 << r[0]
            v["corner_dist"] = TM.dist_to(board.king(not side), c)
        calls[0] += 1
        if calls[0] % 3000 == 0:
            GC.Binding.native_clear()
        return v
    TM.atom_values = atom_values
    try:
        rows = TS.whole_class(sig, log=log)
        got = TMI.mine(sig, rows, log=log)
    finally:
        TM.atom_values = orig
    c = got["choice"]
    zero = c["zero"] if c else []
    cites = [m for m in zero if "reach L free" in m or "corner" in m]
    log(f"OBS objects (i) the {sig} miner re-run with the derived atoms (the region's count, off-edge count and area from {region_term}; the corner distance from "
        f"{corner_term}): " + (f"the first zero-failure level d={c['d']} ranks={c['ranks']} n={c['n']} on {got['positions']} positions; tied {c['tied']}: "
                               f"{', '.join(zero)}; citing a derived atom {len(cites)} of {len(zero)}" if c else "no zero-failure measure to the fuse")
        + f"; atoms read through the term language {calls[0]}; {time.monotonic() - t0:.1f} s")
    return dict(choice=c, cites=len(cites), zero=zero, positions=got["positions"], secs=round(time.monotonic() - t0, 1))


# ------------------------------------------------------------------------------------------------------------------------------------------ (c)

LICHESS = "records/lichess_2019-01"         # under LCP_DATA: the game records (UCI moves, the result from White's view: 1 / 0 / -1)
FROM_PLY = 10                               # the records' own filter: the first ten plies are skipped (manifest.json `skip_first_plies`)


def _atoms_worker(job):
    """One chunk of positions -> [(game id, the atom vector, the outcome from the side to move's view)]: the count atoms read through the term language."""
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import ontology as ON
    from catspace.incubator.analyst.discovery import terms as TL
    from catspace.incubator.analyst.discovery import valuation_miner as VM
    kind, items = job
    dom = TL.Domain(GC.program(), ON.load())
    reader = GC.TermReader()
    atoms = VM.count_terms(dom)
    kinds = list(atoms)
    out = []
    n = 0
    for gid, payload, res in items:
        if kind == "game":
            b = chess.Board()
            for ply, u in enumerate(payload.split()):
                if ply >= FROM_PLY and not b.is_game_over():
                    side = GC.COLOUR[b.turn]
                    ctx = TL.Ctx(dom, reader, b, side)
                    o = 0.5 if res == 0 else (1.0 if (res == 1) == b.turn else 0.0)
                    out.append((gid, tuple(ctx.ev(atoms[k]) for k in kinds), o))
                    n += 1
                    if n % 2000 == 0:
                        GC.Binding.native_clear()
                try:
                    b.push_uci(u)
                except ValueError:
                    break
        else:
            b = chess.Board(payload)
            ctx = TL.Ctx(dom, reader, b, GC.COLOUR[b.turn])
            out.append((gid, tuple(ctx.ev(atoms[k]) for k in kinds), res))
    return kinds, out


def lichess_jobs(n_games: int, chunk: int = 200) -> tuple:
    """The first `n_games` records of the first shard -> (jobs, what was read)."""
    import pyarrow.parquet as pq
    from catspace.io import paths
    f = Path(paths.data_dir()) / LICHESS / "records_00000.parquet"
    t = pq.read_table(f, columns=["game_id", "result", "moves"]).slice(0, n_games).to_pylist()
    games = [(int(r["game_id"]), r["moves"], int(r["result"])) for r in t]
    return [("game", games[i:i + chunk]) for i in range(0, len(games), chunk)], f"the game records data/{LICHESS}/records_00000.parquet (games 0..{n_games - 1})"


def table_jobs(per_sig: int, seed: int = 764, chunk: int = 500) -> tuple:
    """Random legal placements of every material signature the Syzygy table holds (3 to 5 units), the side to move drawn, judged by the table: a win (WDL 2) 1, a
    loss (-2) 0, every other verdict a draw (a cursed win or a blessed loss is a draw under the fifty-move rule) -> (jobs, what was read)."""
    import random
    from catspace.incubator.analyst.discovery import technique_corpus as TC
    from catspace.io import paths
    tb = TC.tablebase()
    rng = random.Random(seed)
    sigs = sorted({p.stem for p in Path(paths.syzygy_dir()).glob("*.rtbw")})
    items = []
    for sig in sigs:
        w, bl = sig.split("v")
        units = [(ch, True) for ch in w] + [(ch, False) for ch in bl]
        got = tries = 0
        while got < per_sig and tries < per_sig * 50:
            tries += 1
            sq = rng.sample(range(64), len(units))
            b = chess.Board(None)
            for s, (ch, col) in zip(sq, units):
                pt = chess.PIECE_SYMBOLS.index(ch.lower())
                if pt == chess.PAWN and chess.square_rank(s) in (0, 7):
                    break
                b.set_piece_at(s, chess.Piece(pt, col))
            else:
                b.turn = rng.random() < 0.5
                if not b.is_valid() or b.is_game_over():
                    continue
                wdl = tb.probe_wdl(b)
                items.append((len(items), b.fen(), 1.0 if wdl == 2 else 0.0 if wdl == -2 else 0.5))
                got += 1
    return [("fen", items[i:i + chunk]) for i in range(0, len(items), chunk)], f"the Syzygy tables' {len(sigs)} signatures (assets/tablebases/syzygy) x {per_sig} random positions (seed {seed})"


def material(n_games: int = 4000, per_sig: int = 200, jobs: int = 10) -> dict:
    """(c) MATERIAL AS THE FIRST MINED VALUATION: two outcome corpora (the game records; the table-judged endings), the count atoms through the term language, the
    priced search, the hand worths' fit on the same corpus."""
    from multiprocessing import Pool
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import lemma_store as LS
    from catspace.incubator.analyst.discovery import valuation_miner as VM
    out = {}
    hand = GC.valuation_weights(LS.goal_kinds().valuation)
    for name, (jb, what) in (("games", lichess_jobs(n_games)), ("table", table_jobs(per_sig))):
        t0 = time.monotonic()
        with Pool(jobs) as pool:
            parts = pool.map(_atoms_worker, jb)
        kinds = parts[0][0]
        rows = [r for _k, rs in parts for r in rs]
        C = VM.Corpus(kinds, rows)
        print(f"OBS valuation corpus {name}: {what}; positions {C.positions} (from ply {FROM_PLY} on for the games), games {C.games}, outcomes for the side to move "
              f"win {int(C.n[:, 2].sum())} draw {int(C.n[:, 1].sum())} loss {int(C.n[:, 0].sum())}; atoms read through the term language "
              f"{time.monotonic() - t0:.1f} s")
        got = VM.search(C)
        h = VM.score_of(C, hand, got["ref"])
        ch = got["choice"]
        if ch is not None:
            print(f"OBS valuation corpus {name} MINED worths {ch['worth']} (price {ch['price']}, {ch['tried']} integer vectors read): log loss {ch['logloss']:.5f}, Brier "
                  f"{ch['brier']:.5f} (s {ch['s']:.4f}, c {ch['c']:.4f}; the continuous fit's + {ch['excess']:.5f}, the paired error {ch['se']:.5f}); "
                  f"HAND worths { {k: hand[k] for k in C.kinds} }: log loss {h['logloss']:.5f}, Brier {h['brier']:.5f} (s {h['s']:.4f}, c {h['c']:.4f}; the continuous "
                  f"fit's + {h['excess']:.5f}, paired {h['paired']:.5f}); {got['secs']} s")
        else:
            print(f"OBS valuation corpus {name}: no integer vector within one standard error by the fuse; the best {got['best']}")
        got.pop("ref", None)
        out[name] = dict(search=got, hand=h, positions=C.positions, games=C.games, what=what, kinds=C.kinds, constant=C.constant)
    import json
    from catspace.io import paths
    path = paths.experiment("plan76_4c", "material.json")
    json.dump(out, open(path, "w"), indent=1, default=str)
    print(f"(saved {path})")
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ (b) the entries

MINED_VERSION = 3          # graph_policy's feature-set versions 0, 1, 2 are taken (the board, the survey, the arrangement): a mined kind's node enters at 3


def _name_of(acc: dict, found: dict) -> str:
    """An accepted relation's shelf name: our word when the rediscovery read it equivalent to a test case, or the obligation's word it was cited for (its
    sub-term's last word: `free`, `corner`; the region for `(reach L free)`), else the miner's own name."""
    for ref, names in found.items():
        if acc["name"] in names:
            return ref
    for t in acc["tests"]:
        if t.startswith("obligation ") and ": " in t:
            sub = t.split(": ", 1)[1]
            return "region" if sub.startswith("(reach") else sub.strip("()").split()[-1]
    return acc["name"]


def write_entries(saved: str, journal: str) -> list:
    """(b) AN ACCEPTED RELATION BECOMES AN ONTOLOGY ENTRY: kind OBJECT, origin MINED, its term (the miner's names replaced by the shelf's), its sort, its reads, its
    evidence (the tests that cite it, how many readings; the rediscovery's reading), its node (feature-set version MINED_VERSION) -> the files written."""
    import json
    import re
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import ontology as ON
    from catspace.incubator.analyst.discovery import terms as TL
    got = json.load(open(saved))
    store = ON.load(force=True)
    names = {a["name"]: _name_of(a, got["found"]) for a in got["accepted"]}
    order = max(e["node"]["order"] for e in store.values() if e["node"]) + 1
    dom = TL.Domain(GC.program(), store)
    defs = TL.Defs()
    out = []
    for a in got["accepted"]:
        term = re.sub(r"\bmined_\d+\b", lambda m: names[m.group(0)], a["term"])
        t = TL.parse(term, dom, defs)
        defs.define(names[a["name"]], t)
        sort = []
        for s in t.sort:
            sort += list(dom.comps) if s == TL.LOC else [dom.spec_sort(s)]
        vs = a.get("vs")
        conf = [f"plan 76.4c (b) {journal}: accepted at level {a['level']}, size {a['size']}, cited by " + ", ".join(a["tests"])]
        if vs:
            conf.append(f"the rediscovery: equal to the test case {vs[0].upper()} on {vs[1]} of {vs[2]} guard readings (ten, wide, the surprise shelf, each side)")
        e = dict(name=names[a["name"]], kind="OBJECT", origin="MINED", sort=sort, reads=TL.base_reads(t, dom, store, defs), gdl=None, rust=None, lean=None,
                 python="catspace.incubator.analyst.discovery.terms.evaluate", bind=None, node=dict(kind=names[a["name"]], order=order, since=MINED_VERSION), edge=None,
                 vocabulary=[], names=False,
                 means=f"a mined object: the relation {term} of the one term language over the spec's relations, accepted by its uses ({', '.join(a['tests'])})",
                 why=dict(gdl="a mined object is a term over the spec's relations, not a relation the spec states",
                          rust="the term is read by the term language (terms.py) through the binding's reader, no relation of the Rust core of its own",
                          lean="no Lean definition generated for a mined term yet (the translation of a term is the prover plan's)"),
                 plan=["76.4"], journal=journal, term=term,
                 evidence=dict(uses={x: {"count": int(a["uses"][x]), "date": journal} for x in a["tests"]}, confirmed=conf, refuted=[]))
        bad = ON.check_entry(e)
        assert not bad, bad
        p = ON.DIR / f"{ON.slug(e['name'])}.json"
        p.write_text(json.dumps(e, indent=1) + "\n")
        out.append(str(p))
        order += 1
    ON.load(force=True)
    print(f"OBS objects entries written: {len(out)} -- " + "; ".join(f"{names[a['name']]} := {re.sub(r'mined_[0-9]+', lambda m: names[m.group(0)], a['term'])}"
                                                            for a in got["accepted"]))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ (c) the experiment's store

MINED_VALUATION = "material (mined)"


def write_valuation(saved: str, journal: str) -> str:
    """(c) THE MINED VALUATION ENTRY (beside the hand `material`, which stays and which the GOAL entries still count in): the game records' choice; its term; the
    exchange law restated over it as the Lean obligation (stated, not proved); the evidence -- both corpora's fits and the hand worths' on the same corpus."""
    import json
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    from catspace.incubator.analyst.discovery import lemma_store as LS
    from catspace.incubator.analyst.discovery import names as NM
    from catspace.incubator.analyst.discovery import ontology as ON
    from catspace.incubator.analyst.discovery import terms as TL
    from catspace.incubator.analyst.discovery import valuation_miner as VM
    got = json.load(open(saved))
    g = got["games"]
    ch = g["search"]["choice"]
    letter = {GC._KIND[t]: k for k, t in NM.KIND.items()}
    hand = LS.goal_kinds().valuation
    worth = {letter[k]: int(v) for k, v in ch["worth"].items()}
    for k in g["constant"]:
        worth[letter[k]] = 0                                          # the constant kind (the royal unit: one each): no outcome, no weight
    unit = min(v for v in worth.values() if v > 0)
    dom = TL.Domain(GC.program(), ON.load())
    term = VM.material_term(dom, LS.slug(MINED_VALUATION), {k: int(v) for k, v in ch["worth"].items()}).sexpr()
    w = ", ".join(f"{k} {worth[k]}" for k in "qrbnpk")
    obligation = (f"THE EXCHANGE LAW OVER THE MINED TERM: Ideas.ExchangeLaw's static exchange (seeB, seeXB, settledWithinAt) restated with the unit values of the mined "
                  f"valuation in place of valueAt's hand values ({w}; the king's capturing value as valueAt has it) -- OBLIGATION: for every position p and square s, "
                  f"the static exchange read with these worths is a lower bound of what the side capturing first on s keeps of the term {term} when the exchange on s "
                  f"is played out with each side free to stop; stated, NOT proved (no theorem of docs/LEAN_INDEX.md states it; the prover plan's).")
    t = got["table"]
    e = dict(name=MINED_VALUATION, kind="VALUATION", reads=["material"], terms=[], status="HEURISTIC", lean=[], origin="MINED", plan=["76.4"], journal=journal,
             measure=None,
             evidence=dict(uses={}, confirmed=[
                 f"plan 76.4c (c) {journal}: fit only to outcomes -- {g['what']}: {g['positions']} positions of {g['games']} games (from ply {FROM_PLY}); the priced "
                 f"search's first price within one paired error of the best weights: {ch['price']} ({ch['tried']} vectors read), log loss {ch['logloss']:.5f} nats "
                 f"per position, Brier {ch['brier']:.5f}; the continuous fit's worths relative to the smallest " + ", ".join(
                     f"{k} {b / min(x for x in g['search']['beta'].values() if x > 0):.3f}" for k, b in g["search"]["beta"].items())
                 + f"; the hand worths on the same corpus: log loss {g['hand']['logloss']:.5f}, Brier {g['hand']['brier']:.5f} (the continuous fit's + "
                 f"{g['hand']['excess']:.5f}, paired error {g['hand']['paired']:.5f})",
                 f"plan 76.4c (c) {journal}, the second corpus (not the entry's): {t['what']}, {t['positions']} table-judged positions: the continuous fit's worths "
                 f"relative to the smallest " + ", ".join(f"{k} {b / min(x for x in t['search']['beta'].values() if x > 0):.3f}" for k, b in t["search"]["beta"].items())
                 + "; no integer vector within one paired error by the fuse; the hand worths' log loss " + f"{t['hand']['logloss']:.5f}"], refuted=[]),
             implementation=["catspace/incubator/analyst/discovery/valuation_miner.py:search", "scripts/plan76_4c.py:write_valuation"],
             note=(f"The material valuation MINED from outcomes alone (plan 76.4c (c)): unit worths {w} fit to the game records' results by the priced search; "
                   f"the agenda still reads the HAND entry `material` (its GOAL entries count in it) -- this entry stands beside it."),
             params=dict(worth=worth, unit=unit, royal=hand.royal if hasattr(hand, "royal") else "k", term=term, obligation=obligation))
    bad = LS.check_entry(e)
    assert not bad, bad
    p = LS.DIR / f"{LS.slug(e['name'])}.json"
    p.write_text(json.dumps(e, indent=1) + "\n")
    LS.load(force=True)
    print(f"OBS valuation entry written: {p.name} -- worth {worth}, unit {unit}, MINED, HEURISTIC")
    return str(p)


GOAL_FILES = ("exchange", "mate", "promotion", "win", "save")


def swap_valuation(mined: bool) -> None:
    """(c)'s ONE EXPERIMENT (never committed): the store's GOAL entries count in the MINED valuation (`mined`) or back in the hand one -- each GOAL file's
    `params.valuation` rewritten, nothing else."""
    import json
    from catspace.incubator.analyst.discovery import lemma_store as LS
    want, was = ("material (mined)", "material") if mined else ("material", "material (mined)")
    for n in GOAL_FILES:
        p = LS.DIR / f"{n}.json"
        e = json.loads(p.read_text())
        assert e["params"]["valuation"] == was, (n, e["params"]["valuation"])
        e["params"]["valuation"] = want
        p.write_text(json.dumps(e, indent=1) + "\n")
    print(f"OBS valuation experiment: the GOAL entries {', '.join(GOAL_FILES)} count in {want!r}")


def decisions_diff(parent: str, after: str) -> list:
    """(c)'s experiment: every guard case's DECISION line in two snapshots (parent, the mined valuation's) -> [(set, case, fen, before, after)] where they differ."""
    import os
    out = []
    fens = {(s, p): f for s, p, f, _u in guard_positions()}
    for sub in ("ten_t0", "ten_t2", "wide_t0", "wide_t2"):
        d0, d1 = os.path.join(parent, sub), os.path.join(after, sub)
        if not os.path.isdir(d1):
            continue
        n = 0
        for f in sorted(os.listdir(d1)):
            if not f.endswith(".txt") or f.count(".") > 1:
                continue
            pick = lambda p: next((ln for ln in open(p).read().splitlines() if ln.startswith("DECISION ")), "DECISION ?")
            x, y = pick(os.path.join(d0, f)), pick(os.path.join(d1, f))
            n += 1
            if x != y:
                pid = f[:-4]
                out.append((sub, pid, next((v for (s, p), v in fens.items() if p == pid), "?"), x[9:], y[9:]))
        print(f"OBS valuation experiment {sub}: DECISION lines differ in {sum(1 for o in out if o[0] == sub)} of {n}")
    for sub, pid, fen, x, y in out:
        print(f"  DECISION {sub} {pid} {fen}: hand {x} | mined {y}")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "decisions":
        decisions_diff(a[1], a[2])
        sys.exit(0)
    if a and a[0] == "valuation-entry":
        write_valuation(a[1], a[2])
        sys.exit(0)
    if a and a[0] == "entries":
        write_entries(a[1], a[2])
        sys.exit(0)
    if a and a[0] == "class-miner":
        import json
        got = json.load(open(a[1]))
        names = {x["name"]: x for x in got["accepted"]}
        reg = next(x for x in got["accepted"] if x["name"] in got["found"].get("region", []) or any("(reach L free)" in t for t in x["tests"]))
        cor = next((x for x in got["accepted"] if any(t.endswith(": corner") for t in x["tests"])), None)
        defs = [(x["name"], x["term"]) for x in got["accepted"] if x["level"] < reg["level"]]
        class_miner(reg["term"], cor["term"] if cor else None, defs)
        sys.exit(0)
    if a and a[0] == "swap":
        swap_valuation(a[1] == "mined")
        sys.exit(0)
    if a and a[0] == "material":
        material(n_games=int(a[a.index("--games") + 1]) if "--games" in a else 4000, per_sig=int(a[a.index("--per-sig") + 1]) if "--per-sig" in a else 200)
        sys.exit(0)
    if a and a[0] == "atoms":
        sys.exit(atoms(control=("unlifted" if "--unlifted" in a else "--control" in a)))
    if a and a[0] == "objects":
        import json
        from catspace.io import paths
        got, dom, reader, readings = objects(fuse=int(a[a.index("--fuse") + 1]) if "--fuse" in a else None, levels=int(a[a.index("--levels") + 1]) if "--levels" in a else 4)
        found = rediscover(got, dom, reader)
        out = paths.experiment("plan76_4c", "objects.json")
        json.dump(dict(accepted=[dict(name=x["name"], term=x["term"].sexpr(), size=x["size"], tests=x["tests"], uses=x["uses"], level=x["level"], vs=x.get("vs"))
                                 for x in got["accepted"]],
                       found=found, levels=got["levels"], tried=got["tried"], tested=got["tested"], pruned=got["pruned"], dropped=got["dropped"], open=got["open"],
                       secs=got["secs"]), open(out, "w"), indent=1, default=str)
        print(f"(saved {out})")
        sys.exit(0)
    sys.exit(__doc__)
