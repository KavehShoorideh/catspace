"""THE SELF-MINING LOOP on one window (ANALYST_PLAN 41.8, 41.9; phase B stage 1).

  python -m catspace.incubator.analyst.discovery.selfmine --game sHa6yUKv --window 3

From the words only (base_vocabulary, evaluated by analyst_core's vocab_values), the exact solver (goals.forced_moves) and Lean:
  1. TARGET   at the window's start the played move loses >= delta (SAME); the avoiding moves do not (AVOIDED). Examples = the positions after the SAME
              moves, counter-examples = the positions after the AVOIDED moves, OTHER set aside.
  2. PROPOSE  atoms = a vocabulary number over the window's OBJECTS (squares that move, capture or are captured in its lines, the kings) compared with a
              threshold read off the examples; premises = conjunctions of 1-3 atoms true on every example and false somewhere on every counter-example's
              side (every counter-example falsifies one atom), minimal; ranked by size, then by plies covered.
  3. SOLVER   for the examples, the other side forces a gain >= delta (forced_moves), else the premise is dropped.
  4. LEAN     each surviving premise becomes a theorem  premise -> forcingOf p c delta != 0  over Ideas.Base.Vocabulary, a proof attempt, run with
              `lake env lean`: TYPECHECKS / PROVES / FAILS with the error. Written to lean/ideas/Ideas/Mined/<game>_<window>.lean.
  5. DIFFERENTLY  on a Lean failure the next ranked premises; on exhaustion of the cap, the atoms are widened once (more objects, 4 atoms).
  6. OBS lines.
NOT used: the chain (ledger decide / size_kick / the defence frame / chain_lemmas.json), the chunked store, any hand-written frame or learned priority.
The words are read from the mover's side ("us" = the side to move in the position after the bad move); the premise's square names are read from that side
too (as the vocabulary names them), so a premise is stated once for either colour. The canonical form of a premise under colour and the left-right mirror
is chain_lemmas' form only (a sorted tuple of atoms, squares named from the mover's side), not its store.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import time
from pathlib import Path

from catspace.io import paths

from . import base_vocabulary as BV

SQUARES = BV.SQUARE_NAMES
BUDGET = 2_000_000
LEAN_DIR = Path(__file__).resolve().parents[4] / "lean" / "ideas"


# ---------------------------------------------------------------------------------------------------------------- the window

def view(sq: str, mover: str) -> str:
    """An absolute square named from the mover's side (black reads the ranks upside down)."""
    return sq if mover == "w" else sq[0] + str(9 - int(sq[1]))


def mirror_file(sq: str) -> str:
    return chr(ord("h") - (ord(sq[0]) - ord("a"))) + sq[1:]


def render(t, mover: str, fn=view):
    """A term over absolute squares, with its squares named from `mover`'s side."""
    if isinstance(t, str):
        return fn(t, mover) if t in SQUARES else t
    return (t[0], *[render(a, mover, fn) for a in t[1:]])


def load_window(game: str, index: int) -> dict:
    d = json.loads(Path(paths.derived("windows", f"{game}.json")).read_text())
    return d["windows"][index]


def positions(win: dict) -> list:
    """Every branch of every ply as the position after its move: {k, cls, played, san, uci, fen, mover}."""
    import chess
    out = []
    for ply in win["plies"]:
        for b in ply["branches"]:
            bd = chess.Board(ply["fen"])
            bd.push_uci(b["uci"])
            out.append(dict(k=ply["k"], cls=b["cls"], played=b["played"], san=b["san"], uci=b["uci"], fen=bd.fen(), mover="w" if bd.turn else "b",
                            start=ply["fen"], line=b.get("line", [])))
    return out


def objects(win: dict, wide: bool = False):
    """The squares of the window's objects, and the (from, to) pairs of its moves. -> (squares sorted, pairs sorted)."""
    import chess
    sqs, pairs = set(), set()
    first_to = set()
    for ply in win["plies"]:
        bd0 = chess.Board(ply["fen"])
        for c in (chess.WHITE, chess.BLACK):
            ks = bd0.king(c)
            if ks is not None:
                sqs.add(chess.square_name(ks))
        for b in ply["branches"]:
            bd = chess.Board(ply["fen"])
            for i, san in enumerate(b.get("line", [])):
                try:
                    m = bd.parse_san(san)
                except ValueError:
                    break
                f, t = chess.square_name(m.from_square), chess.square_name(m.to_square)
                sqs.update((f, t))
                pairs.add((f, t))
                if i == 0:
                    first_to.add(t)
                bd.push(m)
    if wide:
        bd = chess.Board(win["plies"][0]["fen"] if win["plies"] else win["start_fen"])
        occ = [chess.square_name(s) for s in chess.SQUARES if bd.piece_at(s)]
        start = chess.Board(win["start_fen"])
        occ += [chess.square_name(s) for s in chess.SQUARES if start.piece_at(s)]
        occ = sorted(set(occ))
        sqs.update(occ)
        pairs.update((f, t) for f in occ for t in first_to if f != t)
    return sorted(sqs), sorted(pairs)


# ---------------------------------------------------------------------------------------------------------------- propose

def legal(f: str, t: str):
    """The piece on f can legally move to t. `after` reads a relaxed board (Lean afterMove asks no legality), so every premise that reads
    the board after a move carries this conjunct for that move."""
    return ("count", ("and", ("moves", f), ("at", t)))


def make_terms(objs: list, pairs: list) -> list:
    T = []
    kings = ["W", "L"]                 # the kings by name, so a premise about a check survives the king standing elsewhere
    for x in objs:
        T += [("see", x), ("attackers", x), ("defenders", x), ("declined", x), ("count", ("mobility", x)), ("attackable", x, "1"),
              ("dist", "W", x), ("dist", "L", x)]
    for x in kings:
        T += [("attackers", x), ("defenders", x), ("count", ("mobility", x)), ("attackable", x, "1")]
    for f, t in pairs:
        T += [legal(f, t), ("count", ("and", ("between", f, t), ("or", "ours", "theirs")))]
        for x in objs + kings:         # the board after the move, read at every object: the check, the piece newly attacked, the exchange there
            T += [("after", f, t, ("see", x)), ("after", f, t, ("attackers", x)), ("after", f, t, ("defenders", x))]
    seen, out = set(), []
    for t in T:
        s = BV.sexpr(t)
        if s not in seen:
            seen.add(s)
            out.append(t)
    return out


def evaluate(terms: list, poss: list) -> dict:
    """values[sexpr(term)] = the term's number on every position (read from that position's mover; squares named from the mover's side)."""
    import analyst_core as A
    BV.install_terms()
    vals = {}
    for t in terms:
        row = [0] * len(poss)
        for mover in ("w", "b"):
            idx = [i for i, p in enumerate(poss) if p["mover"] == mover]
            if not idx:
                continue
            r = A.vocab_values([poss[i]["fen"] for i in idx], BV.sexpr(render(t, mover)))
            if len(r) != len(idx):
                raise ValueError(f"the evaluator does not read {BV.sexpr(t)}")
            for i, v in zip(idx, r):
                row[i] = v
        vals[BV.sexpr(t)] = row
    return vals


class Atom:
    def __init__(self, term, op, v, truth):
        self.term, self.op, self.v, self.truth = term, op, v, truth
        self.size = BV.size(term)

    def key(self, mover):
        return (BV.sexpr(render(self.term, mover)), self.op, self.v)


def propose(terms, vals, poss, ex, cx, max_atoms=3, mover="w", cap_atoms=200):
    """Atoms true on every example and false on some counter-example; synonyms (the same truth on every window position) merged; premises
    = minimal conjunctions of up to `max_atoms` atoms that falsify every counter-example. -> (stats, premises)."""
    atoms, n_atoms_true = [], 0
    for t in terms:
        row = vals[BV.sexpr(t)]
        ev = [row[i] for i in ex]
        pair = (t[1], t[2]) if t[0] == "after" else None
        if pair is not None and any(vals[BV.sexpr(legal(*pair))][i] < 1 for i in ex):
            continue                                                    # the move is not even legal on an example
        for op, v in ((">=", min(ev)), ("<=", max(ev))):
            truth = [(x >= v) if op == ">=" else (x <= v) for x in row]
            if pair is not None:
                lg = vals[BV.sexpr(legal(*pair))]
                truth = [a and lg[i] >= 1 for i, a in enumerate(truth)]
            n_atoms_true += 1
            a = Atom(t, op, v, truth)
            a.pair = pair
            atoms.append(a)
    full = (1 << len(cx)) - 1
    kept, seen_sig = [], {}
    for a in sorted(atoms, key=lambda a: (a.size, BV.sexpr(a.term), a.op)):
        a.kill = sum(1 << j for j, i in enumerate(cx) if not a.truth[i])
        if a.kill == 0:                       # kills nothing: no conjunction can use it
            continue
        sig = tuple(a.truth)
        if sig in seen_sig:
            seen_sig[sig].syn.append((BV.sexpr(a.term), a.op, a.v))      # the same truth on every position of the window: a synonym here
            continue
        a.syn = [(BV.sexpr(a.term), a.op, a.v)]
        seen_sig[sig] = a
        kept.append(a)
    stats = dict(terms=len(terms), atoms=len(atoms), useful=len(kept), synonyms=sum(len(a.syn) for a in kept) - len(kept))
    prem, enumerated = [], 0
    sep = [a for a in kept if a.kill == full]
    rest = sorted((a for a in kept if a.kill != full), key=lambda a: (-bin(a.kill).count("1"), a.size))[:cap_atoms]   # the strongest killers first
    enumerated += len(kept)
    prem += [(a,) for a in sep]
    for n in range(2, max_atoms + 1):
        for combo in itertools.combinations(rest, n):
            enumerated += 1
            k = 0
            for a in combo:
                k |= a.kill
            if k == full:
                prem.append(combo)
    stats["enumerated"] = enumerated
    stats["separating"] = len(prem)
    return stats, prem


def canon(prem, mover) -> tuple:
    """The premise's canonical form under colour (squares named from the mover's side) and the left-right mirror: the smaller of the two."""
    a = tuple(sorted(a.key(mover) for a in prem))
    b = tuple(sorted((BV.sexpr(render(x.term, mover, lambda s, m: mirror_file(view(s, m)))), x.op, x.v) for x in prem))
    return min(a, b)


def coverage(prem, poss, win_ks):
    """The plies k at which the premise holds on every SAME position of that ply."""
    cov = set()
    for k in win_ks:
        idx = [i for i, p in enumerate(poss) if p["k"] == k and p["cls"] == "SAME"]
        if idx and all(all(a.truth[i] for a in prem) for i in idx):
            cov.add(k)
    return sorted(cov)


# ---------------------------------------------------------------------------------------------------------------- the solver

VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}


def balance(fen: str, white: bool) -> int:
    """Material (pawn 1, minor 3, rook 5, queen 9) of one side minus the other's."""
    b = fen.split()[0]
    w = sum(VALUE[c.lower()] for c in b if c.lower() in VALUE and c.isupper())
    k = sum(VALUE[c] for c in b if c in VALUE)
    return (w - k) if white else (k - w)


def solver_check(fen: str, delta: int, n_opp: int, start_fen: str) -> dict:
    """The side to move in `fen` forces a gain, over its balance at the START of the window (before the bad move, so a recapture of what the bad
    move took is no gain), of >= delta: the fewest of its moves, trying the window's own count of replies and then one more."""
    import analyst_core as A
    white = fen.split()[1] == "w"
    delta = max(1, delta + balance(start_fen, white) - balance(fen, white))
    status = "stopped"
    for d in sorted({max(1, n_opp), max(1, n_opp) + 1}):
        r = A.forced_moves(fen, delta, d, BUDGET)
        f = [x for x in r if x[1] == "forced"]
        if f:
            b = min(f, key=lambda x: x[2])
            return dict(status="forced", depth=d, moves=b[2], first=b[0], line=b[3])
        if any(x[1] == "unknown" for x in r):
            status = "unknown"
    return dict(status=status)


# ---------------------------------------------------------------------------------------------------------------- the drill

def perturb(fen: str) -> list:
    """Every legal one-piece perturbation of the position: each piece (kings included) removed (kings excepted) or moved to each empty square.
    -> [{fen, what}]; the side to move and the castling rights (dropped when a king or rook moves) are kept."""
    import chess
    bd0 = chess.Board(fen)
    out, seen = [], {bd0.fen()}
    empties = [s for s in chess.SQUARES if bd0.piece_at(s) is None]
    for s in chess.SQUARES:
        pc = bd0.piece_at(s)
        if pc is None:
            continue
        name = pc.symbol() + chess.square_name(s)
        cands = [(None, f"remove {name}")] if pc.piece_type != chess.KING else []
        cands += [(t, f"{name} to {chess.square_name(t)}") for t in empties]
        for t, what in cands:
            if pc.piece_type == chess.PAWN and t is not None and chess.square_rank(t) in (0, 7):
                continue
            bd = bd0.copy()
            bd.remove_piece_at(s)
            if t is not None:
                bd.set_piece_at(t, pc)
            bd.castling_rights = bd.clean_castling_rights()
            bd.ep_square = None
            if not bd.is_valid():
                continue
            f = bd.fen()
            if f in seen:
                continue
            seen.add(f)
            out.append(dict(fen=f, what=what))
    return out


def _label(args):
    fen, need, n_opp = args
    import analyst_core as A
    status = "stopped"
    for d in sorted({max(1, n_opp), max(1, n_opp) + 1}):
        r = A.forced_moves(fen, need, d, BUDGET)
        f = [x for x in r if x[1] == "forced"]
        if f:
            b = min(f, key=lambda x: x[2])
            return dict(status="forced", moves=b[2], line=b[3])
        if any(x[1] == "unknown" for x in r):
            status = "unknown"
    return dict(status=status)


def drill(base: dict, need: int, n_opp: int, log=print) -> list:
    """THE PERTURBATION DRILL: the example position varied one piece at a time, each variant labelled by the solver with the SAME requirement
    (the side to move forces a gain >= need in the window's own count of moves): forced -> SAME, not forced -> AVOIDED, unknown -> dropped.
    No teacher. -> positions in the shape of positions()."""
    from multiprocessing import Pool
    import os
    vs = perturb(base["fen"])
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        labs = pool.map(_label, [(v["fen"], need, n_opp) for v in vs], chunksize=4)
    out = []
    for v, lab in zip(vs, labs):
        if lab["status"] == "unknown":
            continue
        out.append(dict(k=base["k"], cls="SAME" if lab["status"] == "forced" else "AVOIDED", played=False, san=base["san"] + " | " + v["what"],
                        uci=base["uci"], fen=v["fen"], mover=base["mover"], start=base["start"], line=lab.get("line", []), drill=True))
    n_same = sum(p["cls"] == "SAME" for p in out)
    log(f"OBS drill: {len(vs)} one-piece variants of the position after {base['san']}, {len(labs) - len(out)} unknown; the solver labels {n_same} still "
        f"forced (SAME) and {len(out) - n_same} not (AVOIDED); {round(time.time() - t0, 1)} s on {os.cpu_count()} cores")
    return out


# ---------------------------------------------------------------------------------------------------------------- Lean

LEAN_HEAD = """import Ideas.Base.Vocabulary

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py ({tag}): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c {delta} != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of {delta} or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.{ns}

open Chess hiding between
open Ideas.Vocabulary
"""

TACTIC = ("intro h\n  first\n    | omega\n    | (simp_all; done)\n"
          "    | (simp_all [Ideas.Vocabulary.seeOf, Ideas.ExchangeLocal.seeXB_le_value]; done)\n    | aesop")


def used_names(t) -> set:
    s = BV.sexpr(t)
    return {n for n in ("W", "L") if re.search(rf"(?<![A-Za-z0-9_]){n}(?![A-Za-z0-9_])", s)}


def with_legality(prem) -> list:
    """The premise's atoms preceded by one legality atom per move it reads the board after. -> [(term, op, v)]"""
    pairs = sorted({a.pair for a in prem if getattr(a, "pair", None)})
    return [(legal(f, t), ">=", 1) for f, t in pairs] + [(a.term, a.op, a.v) for a in prem]


def statement(name: str, prem, mover: str, delta: int, doc: str, proof: str) -> str:
    atoms = []
    names = set()
    for term, op, v in with_legality(prem):
        t = render(term, mover)
        names |= used_names(t)
        atoms.append(f"({BV.lean(t)} {'≥' if op == '>=' else '≤'} {v})")
    binders = "(p : Position) (c : Color) " + "".join(f"({n} : Square) " for n in sorted(names)) + "(hc : p.turn = c)"
    for n in sorted(names):
        binders += f" (h{n} : kingSq p {'c' if n == 'W' else '(Color.opposite c)'} = some {n})"
    return f"/-- {doc} -/\ntheorem {name} {binders} :\n    {' ∧ '.join(atoms)} → forcingOf p c {delta} ≠ 0 := by\n  {proof}\n"


def words(prem, mover) -> str:
    return " AND ".join(f"{BV.gloss(render(t, mover))} {'>=' if op == '>=' else '<='} {v}" for t, op, v in with_legality(prem))


_ERR = re.compile(r"^(\S+?\.lean):(\d+):(\d+): (error|warning): (.*)$")


def run_lean(path: Path) -> list:
    r = subprocess.run(["lake", "env", "lean", str(path.relative_to(LEAN_DIR))], cwd=LEAN_DIR, capture_output=True, text=True, timeout=1800)
    errs, cur = [], None
    for ln in (r.stdout + "\n" + r.stderr).splitlines():
        m = _ERR.match(ln)
        if m:
            cur = dict(line=int(m.group(2)), kind=m.group(4), msg=[m.group(5)])
            errs.append(cur)
        elif cur is not None and ln.strip():
            cur["msg"].append(ln.strip())
    return [dict(line=e["line"], kind=e["kind"], msg=" ".join(e["msg"])[:300]) for e in errs if e["kind"] == "error"]


def lean_round(path: Path, head: str, items: list, delta: int) -> list:
    """items: {name, prem, mover, doc}. Fills item['typechecks'], ['proved'], ['error']. Two runs: the statements with `sorry`, then the proof attempts."""
    for stage in ("statement", "proof"):
        live = [it for it in items if stage == "statement" or it["typechecks"]]
        text, spans = head, {}
        for it in live:
            s = statement(it["name"], it["prem"], it["mover"], delta, it["doc"], "sorry" if stage == "statement" else TACTIC)
            a = text.count("\n") + 1
            text += "\n" + s
            spans[it["name"]] = (a + 1, text.count("\n") + 1)
        path.write_text(text)
        errs = run_lean(path)
        for it in live:
            a, b = spans[it["name"]]
            es = [e for e in errs if a <= e["line"] <= b]
            if stage == "statement":
                it["typechecks"] = not es
                it["error"] = es[0]["msg"] if es else None
            else:
                it["proved"] = not es
                it["error"] = es[0]["msg"] if es else None
    return items


def write_final(path: Path, head: str, items: list, delta: int) -> None:
    text = head
    for it in items:
        if not it["typechecks"]:
            text += f"\n/- DOES NOT TYPECHECK: {it['error']}\n-/\n"
        elif it["proved"]:
            text += "\n" + statement(it["name"], it["prem"], it["mover"], delta, it["doc"], TACTIC)
        else:
            text += f"\n-- PROOF FAILED: {it['error']}\n" + statement(it["name"], it["prem"], it["mover"], delta, it["doc"], "sorry")
    path.write_text(text + f"\nend Ideas.Mined.{path.stem}\n")


# ---------------------------------------------------------------------------------------------------------------- the loop

DEFS = ("seeOf", "attackerCount", "afterMove", "attackableOf", "declinedOf", "mobilityOf", "between", "dist", "kingSq")


def reason(it: dict) -> str:
    """One line: why the proof failed -- the goal Lean was left with, and the local theorem that is missing."""
    e = re.sub(r"\s+", " ", it["error"] or "")
    used = sorted({d for d in DEFS if d in statement("x", it["prem"], it["mover"], 0, "", "sorry")})
    stuck = e.split("⊢")[-1].strip()[:40] if "⊢" in e else e[:80]
    return (f"{e.split(' ', 2)[0]} {e.split(' ', 2)[1] if ' ' in e else ''} (left: {stuck}); missing: a theorem from {'/'.join(used)} to forcingOf "
            f"(forcingOf is a bounded search over all replies, no lemma unfolds it)")


def mine(win: dict, tag: str, top: int = 10, lean_cap: int = 20, do_lean: bool = True, out: Path | None = None, log=print, do_drill: bool = False) -> dict:
    delta = int(win["observable"]["delta"])
    poss = positions(win)
    ks = sorted({p["k"] for p in poss})
    tk = max(k for k in ks if {"SAME", "AVOIDED"} <= {p["cls"] for p in poss if p["k"] == k})
    ex = [i for i, p in enumerate(poss) if p["k"] == tk and p["cls"] == "SAME"]
    cx = [i for i, p in enumerate(poss) if p["k"] == tk and p["cls"] == "AVOIDED"]
    ox = [i for i, p in enumerate(poss) if p["k"] == tk and p["cls"] == "OTHER"]
    mover = poss[ex[0]]["mover"]
    n_opp = max(1, len({k for k in ks if k < tk and any(p["k"] == k and p["mover"] != mover for p in poss)}))
    log(f"OBS target {tag}: start ply k={tk}, delta {delta}; examples (SAME) {[poss[i]['san'] for i in ex]}, counter-examples (AVOIDED) "
        f"{[poss[i]['san'] for i in cx]}, set aside (OTHER) {[poss[i]['san'] for i in ox]}; the side to move after the move is {'white' if mover == 'w' else 'black'}")
    # solver on the examples (and, as a check only, on the counter-examples)
    sol = {i: solver_check(poss[i]["fen"], delta, n_opp, poss[i]["start"]) for i in ex}
    for i in ex:
        log(f"OBS solver example {poss[i]['san']}: {sol[i]['status']}" + (f" in {sol[i]['moves']} moves (depth {sol[i]['depth']} for {n_opp} replies in the window), "
            f"first {sol[i]['first']}, line {' '.join(sol[i]['line'])}" if sol[i]["status"] == "forced" else ""))
    for i in cx:
        r = solver_check(poss[i]["fen"], delta, n_opp, poss[i]["start"])
        log(f"OBS solver counter-example {poss[i]['san']}: {r['status']}")
    solver_ok = all(sol[i]["status"] == "forced" for i in ex)
    dx = []
    if do_drill and solver_ok:
        base = poss[ex[0]]
        need = max(1, delta + balance(base["start"], base["mover"] == "w") - balance(base["fen"], base["mover"] == "w"))
        drilled = drill(base, need, n_opp, log)
        n0 = len(poss)
        poss += drilled
        dx = [i for i in range(n0, len(poss)) if poss[i]["cls"] == "SAME"]
        cx += [i for i in range(n0, len(poss)) if poss[i]["cls"] == "AVOIDED"]

    tried, rounds, proved_any, final_items = set(), [], False, []
    stages = [("atoms over the window's objects, up to 3 atoms", False, 3), ("widened: all pieces of the start position, up to 4 atoms", True, 4)]
    out = out or (LEAN_DIR / "Ideas" / "Mined" / f"{tag}.lean")
    out.parent.mkdir(parents=True, exist_ok=True)
    head = LEAN_HEAD.format(tag=tag, delta=delta, ns=tag)
    n_lean = 0
    for si, (label, wide, max_atoms) in enumerate(stages):
        objs, pairs = objects(win, wide)
        terms = make_terms(objs, pairs)
        t0 = time.time()
        vals = evaluate(terms, poss)
        stats, prem = propose(terms, vals, poss, ex, cx, max_atoms, mover)
        covs = {id(pr): coverage(pr, poss, ks) for pr in prem}
        gen = {id(pr): sum(all(a.truth[i] for a in pr) for i in dx) for pr in prem}       # how many drilled forced variants the premise still covers
        prem.sort(key=lambda pr: (len(pr), -gen[id(pr)], -len(covs[id(pr)]), sum(a.size for a in pr), BV.sexpr(pr[0].term)))
        seen_c, uniq = set(), []
        for pr in prem:
            c = canon(pr, mover)
            if c not in seen_c:
                seen_c.add(c)
                uniq.append(pr)
        log(f"OBS stage {si + 1} ({label}): objects {len(objs)} squares, {len(pairs)} moves; atoms: {stats['terms']} terms, {stats['atoms']} atoms true on the examples, "
            f"{stats['useful']} that falsify a counter-example after merging {stats['synonyms']} synonyms; premises enumerated {stats['enumerated']}; "
            f"kept after separation {len(uniq)} ({len(prem) - len(uniq)} merged as canonical duplicates); {round(time.time() - t0, 1)} s")
        kept_solver = uniq if solver_ok else []
        if si == 0:
            first_stage = kept_solver
        log(f"OBS stage {si + 1}: kept after the solver {len(kept_solver)} (the solver confirms a gain >= {delta} on {sum(sol[i]['status'] == 'forced' for i in ex)}/{len(ex)} examples)")
        for r, pr in enumerate(kept_solver[:top], 1):
            log(f"OBS top {r} (size {len(pr)}, covers plies k={covs[id(pr)]}" + (f", {gen[id(pr)]}/{len(dx)} drilled forced variants" if dx else "") + f"): {words(pr, mover)}   [{' & '.join(a.op + str(a.v) + ' ' + BV.sexpr(render(a.term, mover)) for a in pr)}; absolute squares: {' & '.join(a.op + str(a.v) + ' ' + BV.sexpr(a.term) for a in pr)}]")
        queue = [pr for pr in kept_solver if canon(pr, mover) not in tried]
        while queue and do_lean and n_lean < lean_cap * (si + 1) and not proved_any:
            batch = queue[:min(top, lean_cap * (si + 1) - n_lean)]
            queue = queue[len(batch):]
            items = []
            for pr in batch:
                tried.add(canon(pr, mover))
                n = len(final_items) + len(items) + 1
                items.append(dict(name=f"premise_{n:02d}", prem=pr, mover=mover, doc=f"{words(pr, mover)} (squares named from the mover's side; covers plies k={covs[id(pr)]})"))
            n_lean += len(items)
            t0 = time.time()
            lean_round(out, head, final_items + items, delta)
            final_items += items
            for it in items:
                state = "PROVES" if it["proved"] else ("TYPECHECKS (proof fails)" if it["typechecks"] else "FAILS to typecheck")
                log(f"OBS lean {it['name']}: {state}" + (f": {reason(it)}" if it["error"] else "") + f"   [{words(it['prem'], mover)}]")
            proved_any = any(it["proved"] for it in final_items)
            log(f"OBS lean round: {len(items)} statements in {round(time.time() - t0, 1)} s")
        if proved_any or not do_lean:
            break
        if si == 0:
            log("OBS no premise proved within the cap: widening the atoms once")
    if do_lean and final_items:
        write_final(out, head, final_items, delta)
        rep = run_lean(out)
        log(f"OBS lean file {out.relative_to(LEAN_DIR.parent.parent)}: {len(final_items)} theorems, final check {'clean' if not rep else str(len(rep)) + ' errors'}")
    tc = sum(it["typechecks"] for it in final_items)
    pv = sum(it["proved"] for it in final_items)
    log(f"OBS lean: attempted {len(final_items)}, typechecked {tc}, proved {pv}")
    return dict(items=final_items, typechecked=tc, proved=pv, solver_ok=solver_ok, premises=first_stage, solver=sol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sHa6yUKv")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--lean-cap", type=int, default=20, help="statements sent to Lean per stage")
    ap.add_argument("--no-lean", action="store_true")
    ap.add_argument("--drill", action="store_true", help="the perturbation drill: one-piece variants of the example, labelled by the solver, as counter-examples")
    a = ap.parse_args()
    win = load_window(a.game, a.window)
    t0 = time.time()
    mine(win, f"{a.game}_{a.window}" + ("_drill" if a.drill else ""), a.top, a.lean_cap, not a.no_lean, do_drill=a.drill)
    print(f"OBS total {round(time.time() - t0, 1)} s")


if __name__ == "__main__":
    main()
