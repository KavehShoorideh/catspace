"""PLAN 79.44 PART B: THE CERTIFICATE OF A LEMMA (SYSTEM_SPEC 11 "One lemma machinery, used twice"; pieces: rules, and the long-term memory's statements; ledger rows 49, 44).

THIS TEXT IS THE INTERFACE, written before the body. Game-free: the module reads a lemma through `rule_lemma.py` and the description through the reader (discovery/gdl.py); it names
no game and no relation of any description.

WHAT IT DOES. A lemma says that two relations of a description (with the lemma's delta composed on) have the same rows (SAME) or that one's rows are among the other's (IMPLIES),
under a condition. The certificate asks the reader's own evaluator (the oracle: the description is the source of the rules) at every state of a corpus: it evaluates both relations
and the condition, and where the condition holds it compares their rows. Nothing is weakened to make a lemma pass: a disagreement is kept with its state and its rows, and a lemma with
one is not certified.
THE CORPUS (the fixtures and the seeded games the kernel certificates use, by the description's own moves):
  the FIXTURES  rules/lemmas/<game>/states.txt, one state a line (the format is in rule_lemma.py);
  the GAMES     `GAMES` seeded games, seed `SEED` (the seed of discovery/certify.py), each played from the description's initial state by choosing, with a seeded generator, one
                legal move of each role at each turn (the description's `legal` and `next`), to the description's `terminal` or to the fuse `PLY_FUSE` (a named fuse: random play
                is not promised to end).
WHAT IS COUNTED AND PRINTED (OBS lines): states read; states where the condition held; rows of each relation on those states; rows compared (SAME: the union; IMPLIES: the left
rows); disagreements (states, and rows: SAME the rows of one and not the other, IMPLIES the left rows not on the right) with the first `EXAMPLES` states and rows. A lemma whose
condition never held, or whose left relation never had a row where it held, is VACUOUS and is not certified.
THE CERTIFICATE (rules/lemmas/<game>/<name>.certificate.json, the record of discovery/certify.py: binding, ingests, oracle, corpus, positions, disagreements, examples, key, sources,
date, seconds, and for a lemma the counts above): its KEY is a hash of what the answer depends on: the composed description text, the delta, the condition, the form and the two
relations, the reader's source (discovery/gdl.py), the fixtures, the corpus parameters, and the sources of this module and of rule_lemma.py. A change to any of them makes the
certificate STALE until `run` is repeated. The certificates of the old bindings (discovery/certify.py BINDINGS, their twelve) are not touched and not counted here.
INTERFACE (who calls it: `rule_lemma.standing_of` through `status`; the tests; `python -m catspace.incubator.analyst.discovery.rule_lemma_certify run <game> [<name>] [--jobs N]`):
  `key_of(lemma) -> str`                        the 16 hex digits of the key
  `corpus_states(lemma, games=GAMES, seed=SEED) -> iterator of (label, state)`   (for tests; `certify` reads it by chunks)
  `read_state(lemma, prog, state) -> dict`      {"held": bool, "left": set(rows), "right": set(rows)} at one state
  `certify(lemma, jobs=1, games=GAMES, fixtures=None) -> dict`   the certificate (not written); `fixtures`: a list of state texts that replaces the folder's (a test's own)
  `write(lemma, cert)`, `load(lemma) -> dict | None`, `path_of(lemma) -> Path`
  `status(lemma) -> "fresh" | "stale" | "failing" | "vacuous" | "missing"`   fresh: it exists, its key is the current one, it found no disagreement and was not vacuous
  `standing(lemma) -> (word, detail)`           `rule_lemma.standing_of` with the freshness read here
  `run(game, names=None, jobs=1, games=GAMES) -> int`   certify and write each lemma, print the OBS lines; the exit status 1 when one is not certified
  `adopt(lemma, cert)`                          set the document's standing to CERTIFIED with the certificate's key (only for a certificate that is clean and not vacuous)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import rule_lemma as RL

ROOT = RL.ROOT
GAMES = 20                # the seeded games of the corpus (the co-play of discovery/certify.py plays as many)
SEED = 77                 # discovery/certify.py's seed
PLY_FUSE = 300            # a game is cut here: random play is not promised to end (a named fuse, not tuned to a result)
EXAMPLES = 5              # the disagreements kept whole with a certificate (the count is kept whole)
SOURCES = [GDL.__file__, __file__, RL.__file__]          # the reader and the two modules of the machinery: their bytes are in the key


# ------------------------------------------------------------------------------------------------------------------------------------------ the corpus

def fixtures_path(lemma: RL.Lemma) -> Path:
    return RL.lemma_dir(lemma.game) / "states.txt"


def fixture_lines(lemma: RL.Lemma) -> list:
    p = fixtures_path(lemma)
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith(";")] if p.is_file() else []


def state_of_text(text: str) -> frozenset:
    """A fixture line: the state's fluents as written in the description without `true`: `(f a b) (g c)`."""
    out = set()
    for form in GDL.parse(text):
        rel, args = GDL.to_atom(["true", form])
        out.add((rel, args))
    return frozenset(out)


def state_text(state) -> str:
    return " ".join(sorted("(" + " ".join([rel.split(".", 1)[1]] + [GDL.term_text(a) for a in args]) + ")" for rel, args in state))


def wanted_of(lemma: RL.Lemma, prog: GDL.Program) -> tuple:
    return tuple(sorted({lemma.left["relation"], lemma.right["relation"], *RL.condition_relations(lemma, prog)}))


def play(prog: GDL.Program, wanted: tuple, gi: int, seed: int = SEED, fuse: int = PLY_FUSE):
    """One seeded game from the description's initial state: yields (label, state, evaluated state) at each turn, one legal move of each role chosen by the seeded generator."""
    rng = random.Random(seed * 1000 + gi)
    state = prog.init()
    roles = prog.roles()
    for ply in range(fuse + 1):
        db = prog.run(state, wanted=wanted + ("legal", "terminal"))
        yield f"game {gi} ply {ply}", state, db
        if prog.holds(db, "terminal"):
            return
        legal = prog.legal(state, db)
        does = [(r, rng.choice(sorted(legal[r], key=str))) for r in roles if legal.get(r)]
        if not does:
            return
        state = prog.next_state(state, does)


def corpus_states(lemma: RL.Lemma, games: int = GAMES, seed: int = SEED):
    """The fixtures, then the seeded games: (label, state) in that order."""
    prog = RL.program(lemma)
    for i, ln in enumerate(fixture_lines(lemma)):
        yield f"fixture {i}", state_of_text(ln)
    for gi in range(games):
        for label, state, _db in play(prog, wanted_of(lemma, prog), gi, seed):
            yield label, state


# ------------------------------------------------------------------------------------------------------------------------------------------ one state

def read_state(lemma: RL.Lemma, prog: GDL.Program, state, db: "GDL.DB | None" = None) -> dict:
    db = db or prog.run(state, wanted=wanted_of(lemma, prog))
    held = bool(RL.condition_rows(lemma, db, prog))
    return {"held": held, "left": set(db.facts(lemma.left["relation"])), "right": set(db.facts(lemma.right["relation"]))}


def differ_rows(lemma: RL.Lemma, left: set, right: set) -> tuple:
    """(the rows that disagree, rows compared): SAME the rows of one and not the other; IMPLIES the left rows not on the right."""
    if lemma.form == "SAME":
        return left ^ right, len(left | right)
    return left - right, len(left)


def _row_text(row) -> str:
    return "(" + " ".join(GDL.term_text(a) for a in row) + ")"


def _read_items(args) -> dict:
    """One worker's share: the items ("fixture", index, text) and ("game", index) it reads -> counts and the first disagreements."""
    doc, items, games_seed, fuse = args
    lemma = RL.parse(doc)
    prog = RL.program(lemma)
    wanted = wanted_of(lemma, prog)
    out = {"states": 0, "held": 0, "left": 0, "right": 0, "compared": 0, "bad_states": 0, "bad_rows": 0, "examples": [], "plies": 0}

    def take(label, state, db):
        r = read_state(lemma, prog, state, db)
        out["states"] += 1
        if not r["held"]:
            return
        out["held"] += 1
        out["left"] += len(r["left"])
        out["right"] += len(r["right"])
        bad, n = differ_rows(lemma, r["left"], r["right"])
        out["compared"] += n
        if bad:
            out["bad_states"] += 1
            out["bad_rows"] += len(bad)
            if len(out["examples"]) < EXAMPLES:
                out["examples"].append({"at": label, "state": state_text(state), "rows": sorted(_row_text(b) for b in bad)[:6],
                                        "only_left": len(r["left"] - r["right"]), "only_right": len(r["right"] - r["left"])})

    for it in items:
        if it[0] == "fixture":
            st = state_of_text(it[2])
            take(f"fixture {it[1]}", st, prog.run(st, wanted=wanted))
        else:
            for label, st, db in play(prog, wanted, it[1], games_seed, fuse):
                out["plies"] += 1
                take(label, st, db)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ the key and the certificate

def key_of(lemma: RL.Lemma, fixtures: "list | None" = None, games: int = GAMES) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([RL.composed_text(lemma), lemma.form, lemma.left, lemma.right, lemma.condition, lemma.delta, games, SEED, PLY_FUSE], sort_keys=True).encode())
    for p in SOURCES:
        h.update(Path(p).name.encode())
        h.update(Path(p).read_bytes())
    h.update("\n".join(fixtures if fixtures is not None else fixture_lines(lemma)).encode())
    return h.hexdigest()[:16]


def path_of(lemma: RL.Lemma) -> Path:
    return RL.lemma_dir(lemma.game) / f"{lemma.name}.certificate.json"


def write(lemma: RL.Lemma, cert: dict) -> Path:
    p = path_of(lemma)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cert, indent=1, ensure_ascii=True) + "\n")
    return p


def load(lemma: RL.Lemma) -> "dict | None":
    p = path_of(lemma)
    return json.loads(p.read_text()) if p.is_file() else None


def certify(lemma: RL.Lemma, jobs: int = 1, games: int = GAMES, fixtures: "list | None" = None) -> dict:
    """The lemma on its corpus -> its certificate (not written)."""
    t = time.time()
    fx = fixtures if fixtures is not None else fixture_lines(lemma)
    items = [("fixture", i, ln) for i, ln in enumerate(fx)] + [("game", gi) for gi in range(games)]
    n = max(jobs, 1)
    chunks = [(lemma.doc(), items[i::n], SEED, PLY_FUSE) for i in range(n)]
    if n > 1:
        from multiprocessing import Pool
        with Pool(n) as pool:
            res = pool.map(_read_items, chunks)
    else:
        res = [_read_items(c) for c in chunks]
    tot = {k: sum(r[k] for r in res) for k in ("states", "held", "left", "right", "compared", "bad_states", "bad_rows", "plies")}
    examples = [e for r in res for e in r["examples"]][:EXAMPLES]
    vacuous = tot["held"] == 0 or tot["left"] == 0
    return {"binding": f"lemma {lemma.game} {lemma.name}", "origin": lemma.origin, "form": lemma.form, "left": lemma.left, "right": lemma.right,
            "ingests": f"the lemma {lemma.name} of {lemma.game}: under its condition {lemma.left['relation']} {'has the same rows as' if lemma.form == 'SAME' else 'has its rows among those of'} {lemma.right['relation']}",
            "oracle": "the description with the lemma's delta, evaluated by discovery/gdl.py",
            "corpus": f"{len(fx)} fixture states and {games} seeded games (seed {SEED}, fuse {PLY_FUSE} plies), {tot['plies']} game states",
            "positions": tot["states"], "states_held": tot["held"], "rows_left": tot["left"], "rows_right": tot["right"], "rows_compared": tot["compared"],
            "disagreeing_states": tot["bad_states"], "disagreeing_rows": tot["bad_rows"], "disagreements": tot["bad_states"], "vacuous": vacuous,
            "examples": examples, "key": key_of(lemma, fx, games), "sources": ["the description " + lemma.game + "/" + lemma.variant, "the delta of " + lemma.name,
                                                                              "discovery/gdl.py", "discovery/rule_lemma.py", "discovery/rule_lemma_certify.py", "the fixtures"],
            "date": datetime.date.today().isoformat(), "seconds": round(time.time() - t, 1)}


def status(lemma: RL.Lemma) -> str:
    """'fresh' | 'stale' | 'failing' | 'vacuous' | 'missing' (fresh: it exists, its key is the current one, it found no disagreement and was not vacuous)."""
    c = load(lemma)
    if c is None:
        return "missing"
    if c["disagreements"]:
        return "failing"
    if c["key"] != key_of(lemma, None, _games_of(c)):
        return "stale"
    return "vacuous" if c.get("vacuous") else "fresh"


def _games_of(cert: dict) -> int:
    """The number of games the certificate was made with (read back from its corpus text)."""
    import re
    m = re.search(r"and (\d+) seeded games", cert.get("corpus", ""))
    return int(m.group(1)) if m else GAMES


def standing(lemma: RL.Lemma) -> tuple:
    c = load(lemma)
    return RL.standing_of(lemma, status(lemma) == "fresh", c["key"] if c else None)


def adopt(lemma: RL.Lemma, cert: dict) -> "Path | None":
    """Write the document with the standing CERTIFIED and the certificate's key -- only for a clean, not vacuous certificate."""
    if cert["disagreements"] or cert["vacuous"]:
        return None
    lemma.standing = {"is": "CERTIFIED", "key": cert["key"]}
    return RL.write(lemma)


# ------------------------------------------------------------------------------------------------------------------------------------------ the command

def obs(cert: dict) -> str:
    return (f"OBS certificate of the {cert['binding']} ({cert['form']}, origin {cert['origin']}): states {cert['positions']}, condition held {cert['states_held']}, rows left {cert['rows_left']} "
            f"right {cert['rows_right']}, rows compared {cert['rows_compared']}, disagreeing states {cert['disagreeing_states']} rows {cert['disagreeing_rows']}"
            f"{', VACUOUS' if cert['vacuous'] else ''}, {cert['seconds']} s; corpus {cert['corpus']}; key {cert['key']}")


def run(game: str, names: "list | None" = None, jobs: int = 1, games: int = GAMES, adopt_standing: bool = False) -> int:
    bad = 0
    for lemma in RL.load_all(game):
        if names and lemma.name not in names:
            continue
        cert = certify(lemma, jobs, games)
        write(lemma, cert)
        print(obs(cert))
        for e in cert["examples"]:
            print(f"   FINDING {e['at']}: {e['state']}\n           rows {e['rows']} (left only {e['only_left']}, right only {e['only_right']})")
        if adopt_standing:
            adopt(lemma, cert)
        bad += int(bool(cert["disagreements"] or cert["vacuous"]))
    return 1 if bad else 0


def main(argv: list) -> int:
    if len(argv) < 2 or argv[0] not in ("run", "status"):
        print(__doc__)
        return 2
    game = argv[1]
    if argv[0] == "status":
        for lemma in RL.load_all(game):
            w, d = standing(lemma)
            print(f"OBS lemma {game} {lemma.name} ({lemma.form}, origin {lemma.origin}): certificate {status(lemma)}, standing {w} ({d})")
        return 0
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
    names = [a for a in argv[2:] if not a.startswith("--") and not a.isdigit()]
    return run(game, names or None, jobs, adopt_standing="--adopt" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
