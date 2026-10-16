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

def ours_at(f: str):
    """A piece of ours (not the king) stands on f: what Lean's step_after reads to know whose move `after f t` is."""
    return ("count", ("and", "ours", ("at", f)))


_CTX = {"mover": "w"}


def set_mover(mover: str) -> None:
    """The side whose premise is being built (the side to move at the example). The legs name the rank before promotion from ITS side, so they
    need it; the default is white (hand-made tests)."""
    _CTX["mover"] = mover


def promo_from(mover: str | None = None) -> str:
    """The absolute square (file a) on the rank from which OUR pawn promotes: our seventh rank, read from the mover's side."""
    return view("a7", mover or _CTX["mover"])


def no_pawn_at(f: str):
    """No pawn of ours on f that is about to promote, read as <= 0 (ANALYST_PLAN 44 item 0). `after` promotes only to a queen, so a pawn on the
    rank before promotion cannot be the piece of a named move; any other pawn may (its capture, its push, an en passant: the kernel's own move)."""
    return ("count", ("and", "ourpawns", ("and", ("at", f), ("rank", promo_from()))))


def no_pawn_here(f: str):
    """No pawn of ours on f at all (the root `hanging` is proved for a piece that is not a pawn): read as <= 0."""
    return ("count", ("and", "ourpawns", ("at", f)))


def _is_no_pawn(g) -> bool:
    while isinstance(g, tuple) and g[0] in ("after", "least", "most"):
        g = g[-1]
    return isinstance(g, tuple) and g[0] == "count" and g[1][0] == "and" and g[1][1] == "ourpawns"


def leg_cond(g):
    """How a leg is read: the no-pawn leg as <= 0, every other as >= 1."""
    return ("<=", 0) if _is_no_pawn(g) else (">=", 1)


def leg_holds(g):
    op, v = leg_cond(g)
    return (lambda x: x <= v) if op == "<=" else (lambda x: x >= v)


def legal(f: str, t: str):
    """The piece on f can legally move to t. `after` reads a relaxed board (Lean afterMove asks no legality), so every premise that reads
    the board after a move carries this conjunct for that move."""
    return ("count", ("and", ("moves", f), ("at", t)))


def _wrap(prefix, body):
    """`body` read after the moves of `prefix` (nested `after`s)."""
    for f, t in reversed(prefix):
        body = ("after", f, t, body)
    return body


def legs_of(t, rebuild=lambda b, pol: b) -> list:
    """The legality terms of every `after` a term reads, each read at its own depth and under the same quantifiers:
    (after m1 (least (count (and (moves f2) (at t2))))) -- after m1, whatever they reply, m2 is legal. A leg read as "at most" (no pawn
    of ours on f) goes under `most` instead, so that it too holds for EVERY reply. `least` / `most` ask the legal replies themselves,
    so they add none of their own."""
    if not isinstance(t, tuple):
        return []
    if t[0] == "after":
        f, tt, body = t[1], t[2], t[3]
        return ([rebuild(legal(f, tt), "ge"), rebuild(ours_at(f), "ge"), rebuild(no_pawn_at(f), "le")]
                + legs_of(body, lambda x, pol: rebuild(("after", f, tt, x), pol)))
    if t[0] in ("least", "most"):
        return legs_of(t[1], lambda x, pol: rebuild(("least" if pol == "ge" else "most", x), pol))
    out = []
    for x in t[1:]:
        out += legs_of(x, rebuild)
    return out


def line_pairs_of(fen: str, line: list) -> list:
    """The solver's line (SAN, from `fen`) as (from, to) pairs: our move, their reply, our move, ..."""
    import chess
    b, out = chess.Board(fen), []
    for san in line:
        m = b.parse_san(san)
        out.append((chess.square_name(m.from_square), chess.square_name(m.to_square)))
        b.push(m)
    return out


def wrap_tree(ours: list, body):
    """`body` read after OUR moves of the line, every reply of theirs in between quantified: (after m1 (least (after m2 (least ... body)))).
    A reply is never named: `forcedWithinB` quantifies over every reply, so a premise naming one could not be proved."""
    for j, (f, t) in enumerate(reversed(ours)):
        body = ("after", f, t, body) if j == 0 else ("after", f, t, ("least", body))
    return body


def make_tree_terms(objs: list, line_pairs: list) -> list:
    """Premises read ALONG THE LINE as a small tree (Kaveh 2026-10-15): after our j-th move of the solver's line, over every reply, the
    least X (whatever they answer, X holds) and the most X (the best they can do); their replies quantified, ours named. The cheap words
    at every depth; the three-ply forcing words at depth one only. THE MINIMAX TAIL: after our named moves, "some move of ours, then
    whatever they reply" over the balance -- (most (least balance)) -- so a line whose next move depends on their reply (the recapture)
    is still one statement, and the leaf is ONE number a single witness move bounds (the word `balance`)."""
    kings = ["W", "L"]
    ours = line_pairs[0::2]
    bodies = [(w, x) for x in objs + kings for w in ("see", "attackers", "defenders")] + ["material_us", "material_them", "balance"]
    T = []
    for j in range(len(ours)):
        bs = bodies + (["forcing_them_1", "forcing_them_3", "forcing_them_5"] if j == 0 else [])
        if j >= 2:
            bs = ["material_us", "material_them", "balance"] + [("see", x) for x in objs]     # depth three: 30^3 boards per position; the leaf words only
        for b in bs:
            T.append(wrap_tree(ours[:j + 1], ("least", b)))
            T.append(wrap_tree(ours[:j + 1], ("most", b)))
        for b in ("balance", "material_us"):                                                   # the minimax tail from this depth
            T.append(wrap_tree(ours[:j + 1], ("least", ("most", ("least", b)))))
    return T


def make_terms(objs: list, pairs: list, line_pairs: list | None = None) -> list:
    T = []
    kings = ["W", "L"]                 # the kings by name, so a premise about a check survives the king standing elsewhere
    for x in objs:
        T += [("see", x), ("attackers", x), ("defenders", x), ("declined", x), ("count", ("mobility", x)), ("attackable", x, "1"),
              ("dist", "W", x), ("dist", "L", x)]
    for x in kings:
        T += [("attackers", x), ("defenders", x), ("count", ("mobility", x)), ("attackable", x, "1")]
    T += ["balance", "material_us", "material_them"]       # the root's balance: a gain is read against it (the minimax tail reads the absolute balance)
    for f, t in pairs:
        T += [legal(f, t), ("count", ("and", ("between", f, t), ("or", "ours", "theirs")))]
        for x in objs + kings:         # the board after the move, read at every object: the check, the piece newly attacked, the exchange there
            T += [("after", f, t, ("see", x)), ("after", f, t, ("attackers", x)), ("after", f, t, ("defenders", x))]
    if line_pairs:
        T += make_tree_terms(objs, line_pairs)
    for t in list(T):
        T += legs_of(t)                                        # every after's legs (legal, ours, no pawn), read once: the atoms are gated by them
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


def covers(kept, full, max_atoms, limit=200_000):
    """Every minimal set of up to `max_atoms` atoms whose kills together cover all counter-examples: branch on the hardest uncovered
    counter-example (the fewest atoms kill it), prune when the rest cannot be covered in the picks left. No cap on the atoms."""
    n = full.bit_length()
    killers = [[i for i, a in enumerate(kept) if a.kill >> j & 1] for j in range(n)]
    kills = [a.kill for a in kept]
    found, work = set(), [0]

    def rec(chosen, k):
        work[0] += 1
        if work[0] > limit:
            return
        unc = full & ~k
        if not unc:
            found.add(frozenset(chosen))
            return
        r = max_atoms - len(chosen)
        if r == 0:
            return
        marg = [(m & unc).bit_count() for m in kills]
        if sum(sorted(marg, reverse=True)[:r]) < unc.bit_count():
            return                                           # the best r atoms together cannot cover what is left
        j = min((j for j in range(n) if unc >> j & 1), key=lambda j: len(killers[j]))
        for i in sorted(killers[j], key=lambda i: -marg[i]):
            if i not in chosen:
                rec(chosen + [i], k | kills[i])

    rec([], 0)
    minimal = [c for c in found if not any(d < c for d in found)]
    return [tuple(kept[i] for i in sorted(c)) for c in minimal], work[0]


def propose(terms, vals, poss, ex, cx, max_atoms=3, mover="w", cap_atoms=200, exact=False):
    """Atoms true on every example and false on some counter-example; synonyms (the same truth on every window position) merged; premises
    = minimal conjunctions of up to `max_atoms` atoms that falsify every counter-example. -> (stats, premises)."""
    atoms, n_atoms_true = [], 0
    for t in terms:
        row = vals[BV.sexpr(t)]
        ev = [row[i] for i in ex]
        pair = (t[1], t[2]) if t[0] == "after" else None
        legs = [(vals[BV.sexpr(g)], leg_holds(g)) for g in dict.fromkeys(legs_of(t))]
        if any(not ok(g[i]) for g, ok in legs for i in ex):
            continue                                                    # a move it reads the board after is not even legal (or not ours) on an example
        for op, v in ((">=", min(ev)), ("<=", max(ev))):
            truth = [(x >= v) if op == ">=" else (x <= v) for x in row]
            for lg, ok in legs:
                truth = [a and ok(lg[i]) for i, a in enumerate(truth)]
            n_atoms_true += 1
            a = Atom(t, op, v, truth)
            a.pair = pair
            a.legs = tuple(dict.fromkeys(legs_of(t)))
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
    if exact:                                  # all kept atoms, branching on the hardest counter-example (no cap, no combinations of the strongest)
        more, work = covers(kept, full, max_atoms)
        prem += [c for c in more if len(c) > 1]
        enumerated += work
    for n in range(2, 1 if exact else max_atoms + 1):
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


# ---------------------------------------------------------------------------------------------------------------- the proof generator

def _split(t):
    """A term as (path, body): the path of steps from the root -- ("after", f, t) our named move, ("least",) their turn, ("most",) a turn
    quantified existentially -- and the body read at its end."""
    path = []
    while isinstance(t, tuple) and t[0] in ("after", "least", "most"):
        if t[0] == "after":
            path.append(("after", t[1], t[2]))
            t = t[3]
        else:
            path.append((t[0],))
            t = t[1]
    return tuple(path), t


def _same(pa, sp) -> bool:
    """Two paths to the same node: equal step by step, a `least` and a `most` at the same depth being the same (their) turn."""
    return len(pa) == len(sp) and all(x == y or (x[0] in ("least", "most") and y[0] in ("least", "most")) for x, y in zip(pa, sp))


def n_replies(win: dict, poss: list, ks: list, tk: int, mover: str) -> int:
    """how many of the winner's moves the solver may need from the example: the opponent's plies back in the tree (the old count) or, when the window
    records its quiescence ply (43.2), the winner's moves of the game line from the example to it, whichever is more."""
    n = max(1, len({k for k in ks if k < tk and any(p["k"] == k and p["mover"] != mover for p in poss)}))
    q = win.get("quiescence_ply")
    ply = next((p.get("ply") for p in win["plies"] if p["k"] == tk), None)
    if q is not None and ply is not None and q - 1 >= ply + 1:
        n = max(n, (q - 1 - (ply + 1)) // 2 + 1)
    return n


def _walk(hyps: dict, spine: tuple, n_plies: int, finish) -> list | None:
    """The steps of a proof along a spine of `after` / `least` / `most` nodes, from the hypotheses `hyps` (name -> (path, body, op, v)): step_after_np at
    our named moves (its legs are the ownership, legality and not-about-to-promote atoms read at that node), step_least at theirs (least_floor carries
    every `least` atom one reply down, le_mostOver every `most` one), and at a `most` node `finish(depth, position, turn_proof, hyps_at_node)` closes
    the proof. None when a hypothesis the steps need is missing."""
    lines = []
    fresh = [0]

    def nm(prefix):
        fresh[0] += 1
        return f"{prefix}{fresh[0]}"

    def turn(d, P):
        if d == 0:
            return "hc"
        return (f"(by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : ({P}).turn = c)" if d % 2 == 0
                else f"(by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : ({P}).turn ≠ c)")

    def pos_ok(v):
        return v >= 1 if not isinstance(v, str) else True          # (a symbolic threshold is at least 1 by the schema's own side condition)

    pos = "p"
    for depth, step in enumerate(spine):
        remaining = n_plies - depth
        at_node = {n: h for n, h in hyps.items() if _same(h[0][:depth], spine[:depth])}      # read at this node or below it
        if step[0] == "after":
            f, t = step[1], step[2]

            def find(body, op):
                return next((n for n, (pa, b, o, v) in at_node.items() if _same(pa, spine[:depth]) and o == op and b == body), None)
            hown = find(("count", ("and", "ours", ("at", f))), ">=")
            hnp = find(no_pawn_at(f), "<=")
            hmv = find(("count", ("and", ("moves", f), ("at", t))), ">=")
            if not (hown and hnp and hmv):
                return None
            m, hm, haf = nm("m"), nm("hm"), nm("haf")
            lines.append(f"refine Ideas.TreeStep.step_after_np c _ ({pos}) {remaining - 1} {turn(depth, pos)} _ _ {hown} {hnp} {hmv} ?_")
            lines.append(f"intro {m} {hm} {haf}")
            nxt = {}
            for n, (pa, b, o, v) in at_node.items():
                if len(pa) > depth and pa[depth] == step:
                    lines += [f"rw [{haf}] at {n}", f"beta_reduce at {n}"]
                    nxt[n] = (pa, b, o, v)
            hyps, pos = nxt, f"applyMove ({pos}) {m}"
        elif step[0] == "least":
            hne = next(((n, v) for n, (pa, b, o, v) in at_node.items() if len(pa) > depth and pa[depth] == ("least",) and o == ">=" and pos_ok(v)), None)
            if hne is None:
                return None
            r, hr = nm("r"), nm("hr")
            lines.append(f"refine Ideas.TreeStep.step_least c _ ({pos}) {remaining - 1} {turn(depth, pos)} _ {hne[1]} (by omega) {hne[0]} ?_")
            lines.append(f"intro {r} {hr}")
            nxt = {}
            for n, (pa, b, o, v) in at_node.items():
                if len(pa) > depth and pa[depth] == ("least",) and o == ">=":
                    lines += [f"have {n}' := Ideas.TreeStep.least_floor _ _ {v} {n} {r} {hr}", f"beta_reduce at {n}'"]
                    nxt[n + "'"] = (pa, b, o, v)
                elif len(pa) > depth and pa[depth] == ("most",) and o == "<=":
                    lines += [f"have {n}' := le_trans (Ideas.Vocabulary.le_mostOver _ _ {r} {hr}) {n}", f"beta_reduce at {n}'"]
                    nxt[n + "'"] = (pa, b, o, v)
            hyps, pos = nxt, f"applyMove ({pos}) {r}"
        elif step[0] == "most":
            tail = finish(depth, pos, turn(depth, pos), {n: h for n, h in hyps.items() if _same(h[0][:depth], spine[:depth])})
            if tail is None:
                return None
            lines += tail
            break
    return lines


def tree_proof(atoms: list, delta: int) -> str | None:
    """THE COMPOSING STEP: a Lean proof script for a premise whose atoms lie along one spine -- our named move, every reply, ..., then
    `(most (least balance))` -- built from Ideas.TreeStep: step_after_np at our named moves (its legs are the ownership and legality
    atoms read at that node), step_least at theirs (least_floor carries every `least` atom one reply down), tail_two at the end
    (gainGoal_of_balance, with the root's `balance` atom bounding the base). None when the premise has no such spine. `atoms` are
    (term, op, v) as the statement lists them, in order."""
    spine, k_leaf = None, 0
    for term, op, v in atoms:
        path, body = _split(term)
        if body == "balance" and op == ">=" and len(path) >= 2 and path[-2:] == (("most",), ("least",)):
            spine, k_leaf = path, v - 64
            break
    if spine is None or k_leaf < 4:
        return None
    root_b = [v for term, op, v in atoms if term == "balance" and op == "<="]
    if not root_b or k_leaf < min(root_b) - 64 + delta:    # base <= root balance - 64, so k_leaf >= base + delta needs this
        return None
    n_plies = len(spine)
    if n_plies % 2 or n_plies > 6:
        return None
    hyps = {f"h{i}": (*_split(term), op, v) for i, (term, op, v) in enumerate(atoms)}   # name -> (path, body, op, v), at the node
    lines = ["intro hall", f"obtain ⟨{', '.join(f'h{i}' for i in range(len(atoms)))}⟩ := hall",
             f"refine Ideas.TreeStep.forcingOf_ne_zero p c {delta} hc {n_plies // 2} (by simp) ?_",
             f"show Ideas.forcedWithinB c _ p {n_plies} = true"]
    hb = next(n for n, (pa, b, op, v) in hyps.items() if b == "balance" and op == "<=" and pa == ())
    lines.append(f"have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ {min(root_b) - 64} := by unfold balanceOf at {hb}; omega")

    def finish(depth, pos, turn_proof, at_node):
        if spine[depth:] != (("most",), ("least",)):
            return None
        return [f"exact Ideas.TreeStep.tail_two c _ {delta} ({pos}) {turn_proof} {k_leaf} (by omega) (by omega) (by omega)"]
    body = _walk(hyps, spine, n_plies, finish)
    return None if body is None else "\n  ".join(lines + body)


def _lean_sq(sq: str, mover: str) -> str:
    """An absolute square as Lean's `named c (file, rank)` from the mover's side."""
    r = view(sq, mover)
    return f"(named c (({ord(r[0]) - 97} : Fin 8), ({int(r[1]) - 1} : Fin 8)))"


def root_proof(atoms: list, delta: int, mover: str = "w") -> str | None:
    """THE ROOT'S DISCHARGE (plan 42.3 item 4, 42.5): a premise holding the root's atoms for one capture x -> y -- a piece of theirs on y, its worth
    at least delta, nothing of ours attacked after the capture, no pawn of theirs on the rank before promotion, a legal reply, material 4 or more,
    and the ownership / no-pawn / legality legs of x -- is an instance of `Ideas.Root.hanging`. None when the atoms are not those. `atoms` are
    (term, op, v) in the statement's order; the mover only names the squares for Lean (the same absolute squares the statement renders)."""
    names = {f"h{i}": a for i, a in enumerate(atoms)}

    def find(pred):
        return next((n for n, (t, op, v) in names.items() if pred(t, op, v)), None)
    hit = find(lambda t, op, v: isinstance(t, tuple) and t[0] == "after" and t[3] == ("count", ("and", "ours", "hit")) and op == "<=" and v == 0)
    if hit is None:
        return None
    x, y = names[hit][0][1], names[hit][0][2]
    th = find(lambda t, op, v: t == ("count", ("and", "theirs", ("at", y))) and op == ">=" and v >= 1)
    w = find(lambda t, op, v: t == ("worth", y) and op == ">=" and v >= delta)
    own = find(lambda t, op, v: t == ("count", ("and", "ours", ("at", x))) and op == ">=" and v >= 1)
    npw = find(lambda t, op, v: t == ("count", ("and", "ourpawns", ("at", x))) and op == "<=" and v <= 0)
    mv = find(lambda t, op, v: t == ("count", ("and", ("moves", x), ("at", y))) and op == ">=" and v >= 1)
    pr = find(lambda t, op, v: isinstance(t, tuple) and t[0] == "after" and t[1:3] == (x, y) and isinstance(t[3], tuple) and t[3][0] == "count"
              and t[3][1][:2] == ("and", "theirpawns") and t[3][1][2][0] == "rank" and op == "<=" and v == 0)
    mat = find(lambda t, op, v: t == "material_us" and op == ">=" and v >= 4)
    mov = find(lambda t, op, v: t == ("after", x, y, ("least", "1")) and op == ">=" and v >= 1)
    if None in (th, w, own, npw, mv, pr, mat, mov):
        return None
    return "\n  ".join(["intro hall", f"obtain ⟨{', '.join(names)}⟩ := hall",
                       f"exact Ideas.Root.hanging p c _ W hc hW {_lean_sq(x, mover)} {_lean_sq(y, mover)} {own} {npw} {mv} {th} "
                       f"(by simp only [onum] at {w}; omega) {hit} {pr} (by omega) {mov}"])


# ---------------------------------------------------------------------------------------------------------------- schemas (ANALYST_PLAN 44)

SCHEMAS_FILE = LEAN_DIR / "Ideas" / "Mined" / "Schemas.lean"
DUMMY = {"x1": "b1", "y1": "c1", "x2": "d1", "y2": "e1"}          # the schema's square variables, as squares (file a is the promotion / en-passant rank's)
SCHEMA_VARS = ("x1", "y1", "x2", "y2")


class using_mover:
    """`with using_mover("w"):` -- the legs and the guard are built for that side, and the previous side comes back after."""
    def __init__(self, mover):
        self.mover = mover

    def __enter__(self):
        self.old = _CTX["mover"]
        set_mover(self.mover)

    def __exit__(self, *a):
        set_mover(self.old)


FAMILIES = ("delayed_root", "delayed_root_max", "delayed_balance")      # the ladder's order: the smallest first, the most general (and least explanatory) last


def tail_guard(mover: str, hit: bool = True):
    """THE GUARD of a delayed root (44.2, with the two soundness conjuncts): nothing of ours is attacked (the board's own king lifted; left out when `hit` is
    False, the weakened family), they have no pawn on our second rank (no promotion) nor on our fourth (no en passant), and they have a legal reply.
    Zero when all hold."""
    ours_hit = ("count", ("and", "ours", "hit"))
    promo = ("count", ("and", "theirpawns", ("rank", view(PROMO_SQ, mover))))
    ep = ("count", ("and", "theirpawns", ("rank", view("a4", mover))))
    no_reply = ("eq", ("least", "1"), "0")
    rest = ("sum", promo, ep)
    return ("eq", ("sum", ("sum", ("sum", ours_hit, promo), ep), no_reply), "0") if hit else ("eq", ("sum", rest, no_reply), "0")


MAX_ATTACKED = ("maxworth", ("and", "ours", "hit"))


def tail_body(mover: str, family: str = "delayed_root"):
    """After the last named move and every reply of theirs: some move of ours, after which the guard holds and the balance is read, else 0 (delayed_root);
    the same without the 'nothing attacked' conjunct and with the balance less the most valuable attacked piece (delayed_root_max: they take back at most one
    piece); or just the minimax tail `(most (least balance))` (delayed_balance: the 41.9 tree lemma with its squares made variables)."""
    if family == "delayed_balance":
        return ("least", ("most", ("least", "balance")))
    if family == "delayed_root_max":
        return ("least", ("most", ("if", tail_guard(mover, hit=False), ("diff", "balance", MAX_ATTACKED), "0")))
    return ("least", ("most", ("if", tail_guard(mover), "balance", "0")))


def delayed_tail(pairs: list, mover: str, family: str = "delayed_root"):
    return wrap_tree(pairs, tail_body(mover, family))


def schema_name(depth: int, family: str = "delayed_root") -> str:
    return f"{family}_d{depth}"


def schema_hyps(depth: int, squares: list, B, T, mover: str, family: str = "delayed_root") -> list:
    """The hypotheses of the delayed-root schema of this depth bound to `squares` (absolute, x1 y1 [x2 y2]) and the numbers B (the balance now is at most
    it) and T (the tail reads at least it): [(name, term, op, v)] in the order of with_legality (legs, then the balance, then the tail)."""
    pairs = [(squares[2 * i], squares[2 * i + 1]) for i in range(depth)]
    with using_mover(mover):
        prem = [atom_of("balance", "<=", B), atom_of(delayed_tail(pairs, mover, family), ">=", T)]
        atoms = with_legality(prem)
    return [(f"h{i}", t, op, v) for i, (t, op, v) in enumerate(atoms)]


def tail_family(term):
    """Which family's tail this atom's term is (None when none): `... (most (if G balance 0))` delayed_root, `... (most (if G (diff balance (maxworth ..)) 0))`
    delayed_root_max, `... (most (least balance))` delayed_balance."""
    path, body = _split(term)
    if not path:
        return None
    if path[-1] == ("most",) and isinstance(body, tuple) and body[0] == "if":
        return "delayed_root_max" if isinstance(body[2], tuple) and body[2][0] == "diff" else "delayed_root"
    if path[-2:] == (("most",), ("least",)) and body == "balance":
        return "delayed_balance"
    return None


def _is_tail(term, family: str = "delayed_root") -> bool:
    return tail_family(term) == family


def _hyps_dict(hl: list) -> dict:
    return {n: (*_split(t), op, v) for n, t, op, v in hl}


def schema_proof(depth: int, family: str = "delayed_root") -> str | None:
    """The proof of the delayed-root schema of this depth: forcingOf_ne_zero at horizon depth + 1, step_after_np at each named move, step_least at each
    reply, and at the end `Ideas.Root.root_tail` (the one hand-written step; the tail of a delayed root). One `root_tail` per schema."""
    sq = [DUMMY[v] for v in SCHEMA_VARS[:2 * depth]]
    hl = schema_hyps(depth, sq, "B", "T", "w", family)
    hyps = _hyps_dict(hl)
    tail = next(n for n, t, op, v in hl if _is_tail(t, family))
    hb = next(n for n, t, op, v in hl if t == "balance")
    spine = hyps[tail][0]
    n_plies = 2 * (depth + 1)
    last = {"delayed_root": "Ideas.Root.root_tail", "delayed_root_max": "Ideas.Root.root_tail_max", "delayed_balance": "Ideas.TreeStep.tail_two"}[family]
    lines = [f"refine Ideas.TreeStep.forcingOf_ne_zero p c v hc {depth + 1} (by simp) ?_",
             f"show Ideas.forcedWithinB c _ p {n_plies} = true",
             f"have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at {hb}; omega"]

    def finish(d, pos, turn_proof, at_node):
        if spine[d:] != ((("most",), ("least",)) if family == "delayed_balance" else (("most",),)):
            return None
        return [f"exact {last} c _ v ({pos}) {turn_proof} (T - 64) (by omega) (by omega) (by omega)"]
    with using_mover("w"):                      # (the legs the walk looks for name the rank before promotion from white's side, as the schema was built)
        body = _walk(hyps, spine, n_plies, finish)
    return None if body is None else "\n  ".join(lines + body)


def _var_squares(text: str) -> str:
    """The dummy squares of a rendered schema, as Lean variables."""
    for var, sq in DUMMY.items():
        text = text.replace(f"(some (named c ((({ord(sq[0]) - 97} : Fin 8)), (({int(sq[1]) - 1} : Fin 8)))))", f"(some {var})")
    return text


FAMILY_DOC = {
    "delayed_root": ("THE DELAYED ROOT", "some move of ours leaves a position in which nothing of ours is attacked, they have no pawn about to promote or to take "
                     "en passant, they have a reply, and the balance is at least T"),
    "delayed_root_max": ("THE DELAYED ROOT, WEAKENED (they take back at most one piece)", "some move of ours leaves a position in which they have no pawn about to "
                         "promote or to take en passant, they have a reply, and the balance less the worth of the most valuable piece of ours that is attacked is at least T"),
    "delayed_balance": ("THE DELAYED BALANCE (the minimax tail)", "some move of ours, then whatever they reply, leaves the balance at least T"),
}


def schema_theorem(depth: int, proof: str | None = None, family: str = "delayed_root") -> str:
    sq = [DUMMY[v] for v in SCHEMA_VARS[:2 * depth]]
    hl = schema_hyps(depth, sq, "B", "T", "w", family)
    vs = " ".join(SCHEMA_VARS[:2 * depth])
    hy = "".join(f"\n    ({n} : {_var_squares(BV.lean(render(t, 'w')))} {'≥' if op == '>=' else '≤'} {v})" for n, t, op, v in hl)
    title, tail_words = FAMILY_DOC[family]
    doc = (f"{title}, depth {depth} (plan 44.2, the operators DELAY / RECAPTURE). Our moves {' and '.join(f'x{i + 1} -> y{i + 1}' for i in range(depth))} are "
           f"named (legal, a piece of ours that is not about to promote); whatever they reply to the last, {tail_words}; the balance now is at "
           f"most B, and T is at least B + v and at least 68: then the gain v is forced. The squares are variables: bind them to any position.")
    return (f"/-- {doc} -/\ntheorem {schema_name(depth, family)} (p : Position) (c : Color){'' if family == 'delayed_balance' else ' (W : Square)'} (hc : p.turn = c){'' if family == 'delayed_balance' else ' (hW : kingSq p c = some W)'}\n"
            f"    ({vs} : Square) (B T : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hT : T ≥ 68){hy} :\n    forcingOf p c v ≠ 0 := by\n  "
            f"{proof if proof is not None else schema_proof(depth, family)}\n")


SCHEMA_HEAD = """import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root

/-! SCHEMAS found by the loop (ANALYST_PLAN 44), written by catspace/incubator/analyst/discovery/selfmine.py (`ensure_schema`): each theorem is a premise
over square VARIABLES, proved once from the step theorems (`step_after_np`, `step_least`) and the one hand-written tail `Ideas.Root.root_tail`. A mined window
instance is `exact` of one of them with its squares and numbers. A schema is appended only when Lean proves it. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 400000

namespace Ideas.Mined.Schemas

open Chess hiding between
open Ideas.Vocabulary
"""
SCHEMA_TAIL = "\nend Ideas.Mined.Schemas\n"


def schemas_text() -> str:
    return SCHEMAS_FILE.read_text() if SCHEMAS_FILE.exists() else SCHEMA_HEAD + SCHEMA_TAIL


def proved_schemas() -> list:
    """The names of the schemas in Schemas.lean (a schema is appended only when proved)."""
    return re.findall(r"^theorem (delayed_[a-z_]+_d\d+|dead_end)", schemas_text(), flags=re.M)


def _append_theorem(name: str, thm: str, log=print) -> dict:
    """Append a generated theorem to Schemas.lean when Lean proves the whole file with it, else leave the file as it was. -> dict(proved, new, error)."""
    old = schemas_text()
    body = old[: old.rindex("end Ideas.Mined.Schemas")].rstrip("\n") + "\n\n" + thm + SCHEMA_TAIL
    trial = SCHEMAS_FILE.parent / "_SchemaTrial.lean"
    t0 = time.time()
    try:
        trial.write_text(body)
        errs = run_lean(trial)
    finally:
        trial.unlink(missing_ok=True)
    if errs:
        log(f"OBS lean schema {name}: FAILS: {errs[0]['msg'][:200]} (line {errs[0]['line']})")
        return dict(proved=False, new=True, error=errs[0]["msg"])
    SCHEMAS_FILE.write_text(body)
    build_schemas()
    log(f"OBS lean schema {name}: PROVES ({round(time.time() - t0, 1)} s); appended to {SCHEMAS_FILE.relative_to(LEAN_DIR.parent.parent)}")
    return dict(proved=True, new=True, error=None)


def ensure_schema(depth: int, log=print, do_lean: bool = True, family: str = "delayed_root") -> dict:
    """The schema of this family and depth is in Schemas.lean (proved), or is generated, checked by Lean and appended on success. -> dict(proved, new, error)."""
    name = schema_name(depth, family)
    if name in proved_schemas():
        return dict(proved=True, new=False, error=None)
    if not do_lean:
        return dict(proved=False, new=False, error="--no-lean")
    return _append_theorem(name, schema_theorem(depth, None, family), log)


def build_schemas() -> None:
    """Compile Ideas.Mined.Schemas so that window files can import it."""
    if not SCHEMAS_FILE.exists():
        SCHEMAS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCHEMAS_FILE.write_text(SCHEMA_HEAD + SCHEMA_TAIL)
    subprocess.run(["lake", "build", "Ideas.Mined.Schemas"], cwd=LEAN_DIR, capture_output=True, text=True, timeout=1800)


# ---------------------------------------------------------------------------------------------------------------- dead ends (ANALYST_PLAN 45)

DEAD_VARS = {"s": "b1", "t": "c1", "z": "d1"}        # the dead-end schema's square variables, as squares


def dead_end_atoms(s: str, t: str, z: str, W, mover: str = "w") -> list:
    """THE ONE-STEP EXCHANGE as a premise, read from the CAPTURER's side (the side to move at the leaf; the words' `ours` is the capturer, as the Rust
    evaluator reads a position from its mover): the capturer's piece on s takes the piece of theirs on t, worth at least what it takes (W between the two worths),
    and after s -> t a piece of theirs on z, not about to promote, can go to t. Squares absolute. -> [(name, term, op, v)], the legs of s -> t first."""
    pr = view(PROMO_SQ, mover)
    leg = [("hmv", legal(s, t), ">=", 1), ("hown", ours_at(s), ">=", 1), ("hnp", no_pawn_at(s), "<=", 0)]
    return leg + [("hth", ("count", ("and", "theirs", ("at", t))), ">=", 1), ("hw1", ("worth", s), ">=", W), ("hw2", ("worth", t), "<=", W),
                  ("hzm", ("after", s, t, ("count", ("and", ("moves", z), ("at", t)))), ">=", 1),
                  ("hzo", ("after", s, t, ("count", ("and", "theirs", ("at", z)))), ">=", 1),
                  ("hznp", ("after", s, t, ("count", ("and", "theirpawns", ("and", ("at", z), ("rank", pr))))), "<=", 0)]


def dead_end_conclusion(s: str, t: str):
    """The claim: after the capture s -> t the capturer's balance, whatever they answer, is at most what it is now (the capture gains nothing)."""
    return ("after", s, t, ("least", "balance")), "<=", "balance"


def dead_end_theorem(proof: str | None = None) -> str:
    sq = [DEAD_VARS[v] for v in "stz"]
    with using_mover("w"):
        atoms = dead_end_atoms(*sq, "W")
        lhs, _, rhs = dead_end_conclusion(sq[0], sq[1])
    hy = "".join(f"\n    ({n} : {_dead_squares(BV.lean(render(t, 'w')))} {'≥' if op == '>=' else '≤'} {v})" for n, t, op, v in atoms)
    concl = f"{_dead_squares(BV.lean(render(lhs, 'w')))} ≤ {BV.lean(rhs)}"
    doc = ("THE DEAD END, the one-step exchange from the capturer's side (plan 45.2). The side to move (c) has a piece on s, not about to promote, that "
           "can take a piece of theirs on t, and the piece on s is worth at least W while what stands on t is worth at most W; after that capture a piece of "
           "theirs on z, not about to promote, can take back on t. Then whatever they reply after the capture, the capturer's balance is at most what it is now: "
           "the capture gains nothing, an attack that is a dead end for the attacker. The squares are variables: bind them to any position.")
    return (f"/-- {doc} -/\ntheorem dead_end (p : Position) (c : Color) (hc : p.turn = c) (s t z : Square) (W : Nat){hy} :\n    {concl} := by\n  "
            f"{proof if proof is not None else dead_end_proof()}\n")


def _dead_squares(text: str) -> str:
    for var, sq in DEAD_VARS.items():
        text = text.replace(f"(some (named c ((({ord(sq[0]) - 97} : Fin 8)), (({int(sq[1]) - 1} : Fin 8)))))", f"(some {var})")
    return text


def dead_end_proof() -> str:
    """By reasoning, from the step theorems: the capture is the kernel's normal move (`after_capture_normal`), the recapture is a witness for `least` and
    `dead_end_core` adds the two captures (`capture_gains` twice) and `leastOver_le`."""
    return "\n  ".join([
        "obtain ⟨q0, hs, hq0c, -⟩ := Ideas.TreeStep.ours_at p c s hown",
        "obtain ⟨r, ht, hrc, -⟩ := Ideas.TreeStep.ours_at p (Color.opposite c) t hth",
        "have hnp' := Ideas.TreeStep.not_promoting_at p c s hc hnp q0 hs hq0c",
        "obtain ⟨hm, haf⟩ := Ideas.Root.after_capture_normal p s t q0 r hs (by rw [hq0c, hc]) hnp' ht (Ideas.TreeStep.moves_of_count p s t hmv)",
        "rw [haf] at hzm hzo hznp ⊢",
        "beta_reduce at hzm hzo hznp ⊢",
        "obtain ⟨q1, hz, hc1, -⟩ := Ideas.TreeStep.ours_at _ (Color.opposite c) z hzo",
        "have hnp1 := Ideas.Root.not_promoting_at_opp _ c z (by rw [applyMove_turn, hc]) hznp q1 hz hc1",
        "exact Ideas.Root.dead_end_core p c hc s t z q0 r hs hq0c ht hrc W hw1 hw2 q1 hz hc1 hnp1 (Ideas.TreeStep.moves_of_count _ z t hzm)"])


def ensure_dead_end(log=print, do_lean: bool = True) -> dict:
    """The dead-end schema is in Schemas.lean (proved), or is generated, checked by Lean and appended on success."""
    if "dead_end" in proved_schemas():
        return dict(proved=True, new=False, error=None)
    if not do_lean:
        return dict(proved=False, new=False, error="--no-lean")
    return _append_theorem("dead_end", dead_end_theorem(), log)


def schema_instance_proof(atoms: list, delta: int, mover: str = "w") -> str | None:
    """A premise that holds a delayed-root schema's atoms (its legs, `balance <= B`, the tail `>= T`) for named squares is an instance of the schema theorem:
    `exact Ideas.Mined.Schemas.delayed_root_d<n> p c W hc hW <squares> B T v ...`. Extra atoms are unused hypotheses. None when the atoms are not those."""
    tail = None
    for term, op, v in atoms:
        fam = tail_family(term)
        if op == ">=" and fam is not None:
            tail = (term, v, _split(term)[0], fam)
            break
    if tail is None:
        return None
    term, T, path, family = tail
    afters = [st for st in path if st[0] == "after"]
    depth = len(afters)
    shape = ["after", "least"] * depth + (["most", "least"] if family == "delayed_balance" else ["most"])
    if depth not in (1, 2) or [st[0] for st in path] != shape or schema_name(depth, family) not in proved_schemas():
        return None
    bals = [v for t, op, v in atoms if t == "balance" and op == "<="]
    if not bals:
        return None
    B = min(bals)
    if T < 68 or T < B + delta:
        return None
    squares = [x for st in afters for x in st[1:3]]
    hl = schema_hyps(depth, squares, B, T, mover, family)
    key = {(BV.sexpr(t), op, v): i for i, (t, op, v) in enumerate(atoms)}
    args = []
    for n, t, op, v in hl:
        i = key.get((BV.sexpr(t), op, v))
        if i is None:
            return None
        args.append(f"h{i}")
    sq = " ".join(_lean_sq(x, mover) for x in squares)
    return "\n  ".join(["intro hall", f"obtain ⟨{', '.join(f'h{i}' for i in range(len(atoms)))}⟩ := hall",
                       f"exact Ideas.Mined.Schemas.{schema_name(depth, family)} p c {'' if family == 'delayed_balance' else 'W hc hW'}{' hc' if family == 'delayed_balance' else ''} {sq} {B} {T} {delta} (by omega) (by omega) {' '.join(args)}"])


def prove(atoms: list, delta: int, mover: str = "w") -> str:
    """The proof script for a statement: an instance of a PROVED schema (the reuse), else the tree composition, else the root's instance, else the plain tactic."""
    return schema_instance_proof(atoms, delta, mover) or tree_proof(atoms, delta) or root_proof(atoms, delta, mover) or TACTIC


# ---------------------------------------------------------------------------------------------------------------- Lean

LEAN_HEAD = """import Ideas.Base.Vocabulary
import Ideas.TreeStep

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
    out = {n for n in ("W", "L") if re.search(rf"(?<![A-Za-z0-9_]){n}(?![A-Za-z0-9_])", s)}
    for word, n in (("hit", "W"), ("unhit", "W"), ("attacked", "L"), ("free", "L")):     # the sets whose Lean form names a king
        if re.search(rf"(?<![A-Za-z0-9_]){word}(?![A-Za-z0-9_])", s):
            out.add(n)
    return out


def with_legality(prem) -> list:
    """The premise's atoms preceded by one legality atom per move it reads the board after. -> [(term, op, v)]"""
    legs = sorted({g for a in prem for g in (a.legs if hasattr(a, "legs") else legs_of(a.term))}, key=BV.sexpr)
    return [(g, *leg_cond(g)) for g in legs] + [(a.term, a.op, a.v) for a in prem]


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


_ERR = re.compile(r"^(\S+?\.lean):(\d+):(\d+): (error|warning)(?:\([^)]*\))?: (.*)$")


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
            s = statement(it["name"], it["prem"], it["mover"], delta, it["doc"],
                          "sorry" if stage == "statement" else prove(with_legality(it["prem"]), delta, it["mover"]))
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
            text += "\n" + statement(it["name"], it["prem"], it["mover"], delta, it["doc"], prove(with_legality(it["prem"]), delta, it["mover"]))
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


def mine(win: dict, tag: str, top: int = 10, lean_cap: int = 20, do_lean: bool = True, out: Path | None = None, log=print, do_drill: bool = False, tree: bool = True) -> dict:
    delta = int(win["observable"]["delta"])
    poss = positions(win)
    ks = sorted({p["k"] for p in poss})
    both = [k for k in ks if {"SAME", "AVOIDED"} <= {p["cls"] for p in poss if p["k"] == k}]
    if not both:
        log(f"OBS target {tag}: no ply of the window has both a SAME and an AVOIDED branch (nothing to separate); the tree search is skipped")
        return dict(items=[], typechecked=0, proved=0, solver_ok=False, premises=[], solver={})
    tk = max(both)
    ex = [i for i, p in enumerate(poss) if p["k"] == tk and p["cls"] == "SAME"]
    cx = [i for i, p in enumerate(poss) if p["k"] == tk and p["cls"] == "AVOIDED"]
    ox = [i for i, p in enumerate(poss) if p["k"] == tk and p["cls"] == "OTHER"]
    mover = poss[ex[0]]["mover"]
    set_mover(mover)
    n_opp = n_replies(win, poss, ks, tk, mover)
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

    line_pairs = line_pairs_of(poss[ex[0]]["fen"], sol[ex[0]]["line"]) if (tree and solver_ok) else []
    tree = tree and bool(line_pairs)
    if tree:
        log(f"OBS tree: the example's solver line as pairs {line_pairs}")
    tried, rounds, proved_any, final_items = set(), [], False, []
    stages = [("atoms over the window's objects, up to 3 atoms", False, 3, False),
              ("the same atoms, exact cover by branching, up to 4 atoms", False, 4, True),
              ("widened: all pieces of the start position, exact cover, up to 4 atoms", True, 4, True)]
    memo = {}
    ns = "g_" + tag if tag[0].isdigit() else tag                     # (a Lean namespace cannot start with a digit)
    out = out or (LEAN_DIR / "Ideas" / "Mined" / f"{ns}.lean")
    out.parent.mkdir(parents=True, exist_ok=True)
    head = LEAN_HEAD.format(tag=tag, delta=delta, ns=ns)
    n_lean = 0
    for si, (label, wide, max_atoms, exact) in enumerate(stages):
        objs, pairs = objects(win, wide)
        t0 = time.time()
        if wide not in memo:                                   # (the terms of a stage are read once)
            terms = make_terms(objs, pairs, line_pairs if tree else None)
            memo[wide] = (terms, evaluate(terms, poss))
        terms, vals = memo[wide]
        if tree:
            log(f"OBS stage {si + 1}: {len(terms)} terms, {sum('least' in BV.sexpr(t) or 'most' in BV.sexpr(t) for t in terms)} of them read along the line; evaluated on {len(poss)} positions in {round(time.time() - t0, 1)} s")
        stats, prem = propose(terms, vals, poss, ex, cx, max_atoms, mover, exact=exact)
        covs = {id(pr): coverage(pr, poss, ks) for pr in prem}
        gen = {id(pr): sum(all(a.truth[i] for a in pr) for i in dx) for pr in prem}       # how many drilled forced variants the premise still covers
        prem.sort(key=lambda pr: (len(pr), tree_proof(with_legality(pr), delta) is None, -gen[id(pr)], -len(covs[id(pr)]), sum(a.size for a in pr),
                                  BV.sexpr(pr[0].term)))                      # a premise whose proof the generator can write goes to Lean first
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
        if si == 0 or not first_stage:
            first_stage = kept_solver
        if tree:
            n_tree = [pr for pr in kept_solver if any(re.search(r"\((least|most) ", BV.sexpr(a.term)) for a in pr)]
            log(f"OBS stage {si + 1}: premises that exclude all {len(cx)} counter-examples: {len(kept_solver)} (up to {max_atoms} atoms); with an atom read along the line {len(n_tree)}"
                + (f"; by size {dict(sorted({n: sum(len(p) == n for p in kept_solver) for n in {len(p) for p in kept_solver}}.items()))}" if kept_solver else ""))
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
        if proved_any or (not do_lean and kept_solver):
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


# ---------------------------------------------------------------------------------------------------------------- the ladder (plan 42.5)

LAYERS_HEAD = LEAN_HEAD.replace("import Ideas.TreeStep", "import Ideas.TreeStep\nimport Ideas.Root\nimport Ideas.Mined.Schemas")
PROMO_SQ = "a2"          # their pawn on the rank before promotion, named from the mover's side


def val_at(fen: str, term, mover: str) -> int:
    """A word read on `fen` (squares absolute, named from the mover's side as the evaluator reads them)."""
    import analyst_core as A
    BV.install_terms()
    return A.vocab_eval(fen, BV.sexpr(render(term, mover)))


def root_atoms(x: str, y: str, mover: str, delta: int) -> list:
    """The root's atoms for the capture x -> y, (term, op, v), absolute squares; the other legs of x are added by with_legality (the root asks the
    stronger "no pawn of ours on x at all": `hanging` is proved for a piece that is not a pawn)."""
    pr = view(PROMO_SQ, mover)
    return [(("count", ("and", "theirs", ("at", y))), ">=", 1), (("worth", y), ">=", delta), (no_pawn_here(x), "<=", 0),
            (("after", x, y, ("count", ("and", "ours", "hit"))), "<=", 0),
            (("after", x, y, ("count", ("and", "theirpawns", ("rank", pr)))), "<=", 0),
            ("material_us", ">=", 4), (("after", x, y, ("least", "1")), ">=", 1)]


def atom_of(term, op, v):
    a = Atom(term, op, v, None)
    a.legs = tuple(dict.fromkeys(legs_of(term)))
    return a


def root_readings(fen: str, mover: str, delta: int) -> list:
    """STEP 1 OF THE LADDER: every capture the side to move can make, read in the root's words. Per capture: the worth taken, the pieces of ours
    attacked after it (and the most worth among them, w), a pawn of theirs about to promote, a legal reply, our material; the root's claim
    k = worth - w (- 8 for a promotion threat), and whether the proved theorem `hanging` applies (w = 0, no promotion, a reply, material 4+)."""
    import chess
    bd = chess.Board(fen)
    out = []
    for m in bd.legal_moves:
        if not bd.is_capture(m) or bd.is_en_passant(m):
            continue
        x, y = chess.square_name(m.from_square), chess.square_name(m.to_square)
        pc = bd.piece_at(m.from_square)
        wy = val_at(fen, ("worth", y), mover)
        hitn = val_at(fen, ("after", x, y, ("count", ("and", "ours", "hit"))), mover)
        w = 0
        if hitn:
            for sq in SQUARES:
                if val_at(fen, ("after", x, y, ("count", ("and", ("and", "ours", "hit"), ("at", sq)))), mover):
                    w = max(w, val_at(fen, ("after", x, y, ("worth", sq)), mover))
        promo = val_at(fen, ("after", x, y, ("count", ("and", "theirpawns", ("rank", view(PROMO_SQ, mover))))), mover)
        mov = val_at(fen, ("after", x, y, ("least", "1")), mover)
        mat = val_at(fen, "material_us", mover)
        k = wy - w - (8 if promo else 0)
        pawn = pc.piece_type == chess.PAWN
        out.append(dict(san=bd.san(m), x=x, y=y, worth=wy, attacked_after=hitn, w=w, promo=promo, reply=mov, material=mat, k=k, pawn=pawn,
                        applies=(not pawn and hitn == 0 and promo == 0 and mov >= 1 and mat >= 4)))
    return out


def lean_root_instance(r: dict, mover: str, delta: int, out: Path, head: str, do_lean: bool = True, log=print) -> dict:
    """The root's atoms for the capture `r` as a Lean statement, proved by `Ideas.Root.hanging` (root_proof). -> the item (proved / typechecks / error)."""
    prem = [atom_of(t, op, v) for t, op, v in root_atoms(r["x"], r["y"], mover, delta)]
    item = dict(name="hanging_instance", prem=prem, mover=mover, doc=f"THE ROOT on {r['san']}: {words(prem, mover)}")
    if do_lean:
        lean_round(out, head, [item], delta)
        write_final(out, head, [item], delta)
        rep = run_lean(out)
        state = "PROVES" if item["proved"] else ("TYPECHECKS (proof fails)" if item["typechecks"] else "FAILS to typecheck")
        log(f"OBS lean hanging_instance: {state}" + (f": {item['error']}" if item["error"] else "") + f"   [{words(prem, mover)}]")
        log(f"OBS lean file {out.relative_to(LEAN_DIR.parent.parent)}: 1 theorem, final check {'clean' if not rep else str(len(rep)) + ' errors'}")
    return item


def seeded_premises(terms, vals, poss, ex, cx, seed, extra=2, cap_atoms=300):
    """STEP 3: the deviation. A premise = the root's seed atoms [(term, op)] plus at most `extra` more atoms that, together with the seed, falsify
    every counter-example (and hold on every example). -> [tuple of Atom], the fewest extra atoms first."""
    def build(t, ops):
        row = vals[BV.sexpr(t)]
        ev = [row[i] for i in ex]
        legs = [(vals[BV.sexpr(g)], leg_holds(g)) for g in dict.fromkeys(legs_of(t))]
        if any(not ok(g[i]) for g, ok in legs for i in ex):
            return []
        res = []
        for op, v in ((">=", min(ev)), ("<=", max(ev))):
            if ops is not None and op not in ops:
                continue
            truth = [(x >= v) if op == ">=" else (x <= v) for x in row]
            for lg, ok in legs:
                truth = [a and ok(lg[i]) for i, a in enumerate(truth)]
            a = Atom(t, op, v, truth)
            a.legs = tuple(dict.fromkeys(legs_of(t)))
            res.append(a)
        return res
    full = (1 << len(cx)) - 1
    seed_atoms = [a for t, ops in seed for a in build(t, ops)]
    for a in seed_atoms:
        a.kill = sum(1 << j for j, i in enumerate(cx) if not a.truth[i])
    k0 = 0
    for a in seed_atoms:
        k0 |= a.kill
    seed_keys = {BV.sexpr(t) for t, _ in seed}
    cand, seen = [], set()
    for t in terms:
        if BV.sexpr(t) in seed_keys:
            continue
        for a in build(t, None):
            a.kill = sum(1 << j for j, i in enumerate(cx) if not a.truth[i]) & ~k0
            if not a.kill or tuple(a.truth) in seen:
                continue
            seen.add(tuple(a.truth))
            cand.append(a)
    cand.sort(key=lambda a: (-a.kill.bit_count(), a.size, BV.sexpr(a.term)))
    cand = cand[:cap_atoms]
    unc = full & ~k0
    prem = []
    if not unc:
        prem.append(tuple(seed_atoms))
    for n in range(1, extra + 1):
        for combo in itertools.combinations(cand, n):
            k = 0
            for a in combo:
                k |= a.kill
            if k & unc == unc:
                prem.append(tuple(seed_atoms) + combo)
        if prem:
            break
    return prem


SCORE_FILE = ("lemmas", "score.json")


def premise_truth(prem, poss: list) -> list:
    """For each position, does the whole premise (its atoms and their legs) hold? Read in the words, from each position's mover."""
    atoms = with_legality(prem)
    terms = list({BV.sexpr(t): t for t, _, _ in atoms}.values())
    vals = evaluate(terms, poss)
    return [all((vals[BV.sexpr(t)][i] >= v) if op == ">=" else (vals[BV.sexpr(t)][i] <= v) for t, op, v in atoms) for i in range(len(poss))]


def drill_report(name: str, prem, poss: list, dx: list, cx: list, proved: bool, log=print) -> dict:
    """42.6: over the drilled variants, where the premise holds and the solver DISAGREES (a variant it labels not forced). Must be 0 for a proved lemma:
    a non-zero count is a bug in the evaluator or the solver, and the run stops."""
    truth = premise_truth(prem, poss)
    cov = sum(truth[i] for i in dx)
    dis = [i for i in cx if truth[i] and poss[i].get("drill")]
    log(f"OBS drill {name}: the premise holds on {cov}/{len(dx)} drilled variants the solver forces and on {len(dis)} of the {sum(1 for i in cx if poss[i].get('drill'))} it does not "
        f"(disagreements: {len(dis)}{'' if not dis else ', ' + ', '.join(poss[i]['san'] for i in dis[:5])})")
    if proved and dis:
        raise SystemExit(f"OBS BUG: the proved lemma {name} holds on {len(dis)} drilled variants the solver does not force: an evaluator or solver bug; stopping")
    return dict(covered=cov, variants=len(dx), disagreements=len(dis))


def update_score(tag: str, lemmas: list, log=print) -> dict:
    """42.7: data/derived/lemmas/score.json -- per lemma: size (atoms), the windows it explains, the drilled variants covered per window; reuse = the number of
    distinct windows it explained (44.2; 42.7 counted those beyond the first)."""
    path = Path(paths.derived(*SCORE_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    table = json.loads(path.read_text()) if path.exists() else {}
    for lm in lemmas:
        e = table.setdefault(lm["name"], dict(size=lm["size"], windows={}))
        e["size"] = lm["size"]
        e["windows"][tag] = dict(explained=bool(lm["explained"]), variants=lm.get("covered", 0), of=lm.get("variants", 0), proved=bool(lm.get("proved")))
    path.write_text(json.dumps(table, indent=1, sort_keys=True))
    for name, e in sorted(table.items()):
        won = [w for w, d in e["windows"].items() if d["explained"]]
        reuse = len(won)                      # 44.2: the number of distinct windows a schema (or the root) explained
        cov = sum(d["variants"] for d in e["windows"].values())
        log(f"OBS score {name}: size {e['size']}, reuse {reuse}, coverage {cov} (explains {won or 'no window'}; per window {({w: str(d['variants']) + '/' + str(d['of']) for w, d in e['windows'].items()})})")
    return table


# ---------------------------------------------------------------------------------------------------------------- diagnose, repair, bind (ANALYST_PLAN 44)

ROOT_HYPS = ("hth", "hv", "hhit", "hpr", "hmov", "hmat", "hown", "hnp", "hmv")


def diagnose(fen: str, mover: str, need: int, x: str, y: str) -> dict:
    """DIAGNOSE: every hypothesis of the root `Ideas.Root.hanging` read at the move x -> y (a bool per name, True: holds). hth: a piece of theirs on y;
    hv: its worth is at least the gain; hhit: nothing of ours is attacked after; hpr: no pawn of theirs about to promote; hmov: they have a reply;
    hmat: our material is 4 or more; hown / hnp / hmv: the piece on x is ours, not a pawn, and can go to y. Squares absolute."""
    pr = view(PROMO_SQ, mover)
    tests = {"hth": (("count", ("and", "theirs", ("at", y))), ">=", 1),
             "hv": (("worth", y), ">=", need),
             "hhit": (("after", x, y, ("count", ("and", "ours", "hit"))), "<=", 0),
             "hpr": (("after", x, y, ("count", ("and", "theirpawns", ("rank", pr)))), "<=", 0),
             "hmov": (("after", x, y, ("least", "1")), ">=", 1),
             "hmat": ("material_us", ">=", 4),
             "hown": (ours_at(x), ">=", 1), "hnp": (no_pawn_here(x), "<=", 0), "hmv": (legal(x, y), ">=", 1)}
    out = {}
    for name in ROOT_HYPS:
        t, op, v = tests[name]
        got = val_at(fen, t, mover)
        out[name] = (got >= v) if op == ">=" else (got <= v)
    return out


def failing(diag: dict) -> list:
    return [n for n in ROOT_HYPS if not diag[n]]


CLAUSE = {"hth": "there is no piece of theirs on {y} to take", "hv": "what stands on {y} is worth less than the {need} to gain",
          "hhit": "something of ours would be attacked after the move", "hpr": "they would have a pawn about to promote",
          "hmov": "they would have no legal reply", "hmat": "we have less than 4 of material", "hown": "{x} holds no piece of ours",
          "hnp": "the piece on {x} is a pawn", "hmv": "{x} cannot legally go to {y}"}


def repair(diag: dict) -> list:
    """REPAIR OPERATORS by the failing hypothesis (44.2): hv / hth -> DELAY (the gain is not now: it is one move later under every reply); hhit / hpr ->
    RECAPTURE (the capturing piece is attacked after: the same schema, the diagnosis names the deviation); hmat / hmov -> EXCLUDE (an extra atom). EXCLUDE also
    follows a DELAY that does not prove. -> [(operator, failing names that call it)], in the order they are tried."""
    f = failing(diag)
    ops = []
    d = [n for n in f if n in ("hv", "hth")]
    r = [n for n in f if n in ("hhit", "hpr")]
    if d:
        ops.append(("DELAY", d + r))
    elif r:
        ops.append(("RECAPTURE", r))
    elif "hnp" in f:                       # (the root is proved for a piece that is not a pawn: nothing else fails, the delayed root takes a pawn too)
        ops.append(("DELAY", ["hnp"]))
    ex = [n for n in f if n in ("hmat", "hmov")]
    ops.append(("EXCLUDE", ex or ["a failed delay"]))
    return ops


def explanation(diag: dict, op: str, depth: int, pairs: list, need: int, family: str = "delayed_root") -> str:
    """The explanation line from the diagnosis: "normally ..., but because ...". `pairs` are the schema's named moves (absolute squares)."""
    (x, y) = pairs[0]
    because = " and ".join(CLAUSE[n].format(x=x, y=y, need=need) for n in failing(diag)) or "nothing of the root fails"
    moves = " then ".join(f"{a}->{b}" for a, b in pairs)
    if family == "delayed_root_max":
        return (f"normally nothing of ours may be attacked after the move, but because they can take back at most one piece, the balance minus the most valuable "
                f"attacked piece is what must clear {need}: the root fails at {x}->{y} on {{{', '.join(failing(diag))}}} ({because}); after {moves}, whatever they reply, "
                f"we have a move after which no pawn of theirs can promote or take en passant, they have a reply, and our balance less the worth of our most "
                f"valuable attacked piece has risen by {need}, so the gain comes {depth + 1} moves of ours later ({op})")
    if family == "delayed_balance":
        return (f"normally the hanging piece is taken now, but because {because}, no named piece or square explains the gain beyond the balance itself: after {moves}, "
                f"whatever they reply, we have a move and then whatever they reply the balance has risen by {need} (the minimax tail, {depth + 1} moves of ours; "
                f"the most general and the least explanatory schema)")
    return (f"normally the hanging piece is taken now (a piece of theirs on {y} worth the gain, nothing of ours attacked after), but because {because}, "
            f"the gain comes {depth + 1} moves of ours later ({op}): after {moves}, whatever they reply, we have a move after which nothing of ours is "
            f"attacked, no pawn of theirs can promote or take en passant, they have a reply, and our material is up by {need}")


def bind(depth: int, fen: str, mover: str, need: int, candidates: list | None = None, family: str = "delayed_root") -> list:
    """BIND the delayed-root schema of this depth to a position: for each candidate (our named moves [(x1, y1), ...], absolute squares; by default every legal
    move of a piece of ours that is not a king nor a pawn about to promote, for depth one) substitute the squares, evaluate the premise in Rust (the legs,
    `balance <= B`, the tail `>= T` with B the balance now and T = max(B + need, 68)), keep the bindings where every atom holds.
    -> [dict(depth, pairs, B, T, prem, atoms)]."""
    import chess
    B = val_at(fen, "balance", mover)
    T = max(B + need, 68)
    if candidates is None:
        bd = chess.Board(fen)
        seen, candidates = set(), []
        for m in bd.legal_moves:
            pc = bd.piece_at(m.from_square)
            if pc.piece_type == chess.KING or m.promotion:
                continue
            pr = (chess.square_name(m.from_square), chess.square_name(m.to_square))
            if pr not in seen:
                seen.add(pr)
                candidates.append([pr])
    out = []
    for pairs in candidates:
        if len(pairs) != depth:
            continue
        with using_mover(mover):
            prem = [atom_of("balance", "<=", B), atom_of(delayed_tail(pairs, mover, family), ">=", T)]
        if premise_truth(prem, [dict(fen=fen, mover=mover)])[0]:
            out.append(dict(depth=depth, pairs=pairs, B=B, T=T, prem=prem, atoms=with_legality(prem), family=family))
    return out


def schema_sentence(depth: int, family: str = "delayed_root") -> str:
    """The schema of this family and depth as a sentence over its square variables x1 y1 [x2 y2]."""
    legs = [f"{'after x1->y1, whatever they reply, ' if i else ''}x{i + 1} holds a piece of ours that is not a king and not a pawn about to promote and can legally go to y{i + 1}"
            for i in range(depth)]
    first = "; ".join(legs)
    moves = " then ".join(f"x{i + 1}->y{i + 1}" for i in range(depth))
    tails = {"delayed_root": "we have a move after which nothing of ours is attacked (the board's own king lifted), they have no pawn on our second or fourth rank, "
                             "they have a legal reply, and our balance is at least T",
             "delayed_root_max": "we have a move after which they have no pawn on our second or fourth rank, they have a legal reply, and our balance less the worth "
                                 "of our most valuable attacked piece (the board's own king lifted) is at least T",
             "delayed_balance": "we have a move and then, whatever they reply, our balance is at least T"}
    return (f"SCHEMA {schema_name(depth, family)}(x1 y1{' x2 y2' if depth == 2 else ''}, B, T, v): {first}; our balance now is at most B; after {moves}, whatever they reply, "
            f"{tails[family]}; with T >= B + v and T >= 68 we force a gain of v within {depth + 1} moves of ours against every defence")


def played_next(win: dict, fen: str):
    """The game's own next move from the example position, when the window's played line passes through it: (san, x, y), absolute squares; else None."""
    import chess
    if not win.get("played_uci") or not win.get("start_fen"):
        return None
    b = chess.Board(win["start_fen"])
    key = " ".join(fen.split()[:2])
    for j, u in enumerate(win["played_uci"]):
        if " ".join(b.fen().split()[:2]) == key:
            m = chess.Move.from_uci(u)
            return b.san(m), chess.square_name(m.from_square), chess.square_name(m.to_square)
        b.push_uci(u)
    return None


def later_root(win: dict, base: dict, mover: str, delta: int, ref: str, n_opp: int, log=print):
    """43.2: the victim's later moves of the played line inside the window (after the example, up to the quiescence ply): the first position where the root's
    capture applies, takes at least what is still to be won, and the solver confirms it in one move of ours. -> (reading, base-like dict, need, label) or None."""
    import chess
    q = win.get("quiescence_ply")
    if q is None or not win.get("played_uci"):
        return None
    start = win["start_ply"]
    b = chess.Board(win["start_fen"])
    for j, u in enumerate(win["played_uci"]):
        ply = start + j
        san = b.san(chess.Move.from_uci(u))
        b.push_uci(u)
        if ply <= base_ply(win, base) or ply >= q - 1 or ("w" if b.turn else "b") != mover:
            continue
        fen2 = b.fen()
        need = max(1, delta + balance(ref, mover == "w") - balance(fen2, mover == "w"))
        reads = root_readings(fen2, mover, need)
        cands = [r for r in reads if r["applies"] and r["worth"] >= need]
        lab = _label((fen2, need, 1))
        log(f"OBS root at the played move {san} (game ply {ply + 1}, inside the window): needs {need}, {len(cands)} capture(s) the theorem covers, solver {lab['status']} in one move")
        if cands and lab["status"] == "forced":
            return cands[0], dict(k=base["k"], cls="SAME", played=True, san=san, uci=u, fen=fen2, mover=mover, start=base["start"], line=lab["line"]), need, lab
    return None


def base_ply(win: dict, base: dict) -> int:
    return next(p["ply"] for p in win["plies"] if p["k"] == base["k"])


def layers(win: dict, tag: str, top: int = 10, lean_cap: int = 20, do_lean: bool = True, do_drill: bool = False, log=print,
           out: Path | None = None, fallback: bool = True) -> dict:
    """THE LADDER (plan 42.5): the root first, then the deviation seeded by the root, then (when nothing is proved) the 41.9 tree lemma as the fallback."""
    delta = int(win["observable"]["delta"])
    poss = positions(win)
    ks = sorted({p["k"] for p in poss})
    # the EXAMPLE is the window's START (43.2): the deepest ply where the victim to move has a SAME branch, i.e. where the trap was laid, not the last ply with both classes
    victim = win["observable"]["victim"]
    side = {"white": "w", "black": "b"}[victim] if victim else None
    starts = [k for k in ks if k >= 1 and any(p["k"] == k and p["cls"] == "SAME" and p["mover"] != side and side for p in poss)]    # a victim's move leaves the winner to move
    if starts:
        tk = max(starts)
    else:                                                                       # no such ply: the old choice
        both = [k for k in ks if {"SAME", "AVOIDED"} <= {p["cls"] for p in poss if p["k"] == k}]
        tk = max(both) if both else min(k for k in ks if k >= 1 and any(p["k"] == k and p["cls"] == "SAME" for p in poss))
    log(f"OBS target {tag}: the example is the window's start, ply k={tk} of {max(ks)} (game ply {next(p for p in win['plies'] if p['k'] == tk)['ply'] + 1}), "
        f"{'a victim move' if starts else 'no victim ply with a SAME branch: the last ply with both classes'}")
    ex = [i for i, p in enumerate(poss) if p["k"] == tk and p["cls"] == "SAME"]
    cx = [i for i, p in enumerate(poss) if p["k"] == tk and p["cls"] == "AVOIDED"]
    ex.sort(key=lambda i: not poss[i]["played"])                  # the move that was played first: the example is the game's own position
    mover = poss[ex[0]]["mover"]
    set_mover(mover)
    n_opp = n_replies(win, poss, ks, tk, mover)
    base = poss[ex[0]]
    fen = base["fen"]
    log(f"OBS layers {tag}: example {base['san']} (position after the move, {'white' if mover == 'w' else 'black'} to move), delta {delta}, "
        f"{len(ex)} examples, {len(cx)} counter-examples, {n_opp} reply/replies in the window")
    # the balance the gain is counted from: the position the sequence starts at when the window has a quiescence point (43.2), else the start of the window
    ref = win["observable"]["fen_before"] if win["observable"].get("settle_ply") is not None else base["start"]
    sol = solver_check(fen, delta, n_opp, ref)
    log(f"OBS solver example {base['san']}: {sol['status']}" + (f" in {sol['moves']} moves, line {' '.join(sol['line'])}" if sol["status"] == "forced" else ""))
    # 1. ROOT FIRST
    reads = root_readings(fen, mover, delta)
    for r in reads:
        log(f"OBS root capture {r['san']} ({r['x']}->{r['y']}): worth taken {r['worth']}, our pieces attacked after {r['attacked_after']} (most worth {r['w']}), "
            f"pawn about to promote {r['promo']}, replies {r['reply']}, material {r['material']}; root claims k = {r['k']}; "
            + ("the theorem `hanging` applies" if r["applies"] else "pawn move: the root theorem does not apply" if r["pawn"] else "the theorem `hanging` does not apply (w > 0, a promotion, no reply or material < 4)"))
    if not reads:
        log("OBS root: no capture available; the root claims nothing")
    # 2. CHECK against the solver, one move of ours and one reply
    lab = _label((fen, delta, 1))
    log(f"OBS root check: the solver {'confirms' if lab['status'] == 'forced' else 'does not confirm (' + lab['status'] + ')'} a forced gain >= {delta} within 1 move of ours and 1 reply"
        + (f", line {' '.join(lab['line'])}" if lab["status"] == "forced" else ""))
    dx = []
    solver_ok = sol["status"] == "forced"
    if do_drill and (solver_ok or lab["status"] == "forced"):
        need = max(1, delta + balance(ref, mover == "w") - balance(fen, mover == "w"))
        drilled = drill(base, need, n_opp, log)
        n0 = len(poss)
        poss += drilled
        dx = [i for i in range(n0, len(poss)) if poss[i]["cls"] == "SAME"]
        cx += [i for i in range(n0, len(poss)) if poss[i]["cls"] == "AVOIDED"]
    cands = [r for r in reads if r["applies"] and r["worth"] >= delta]
    lean_tag = ("g_" + tag if tag[0].isdigit() else tag) + "_layers"          # (a Lean namespace cannot start with a digit)
    out = out or (LEAN_DIR / "Ideas" / "Mined" / f"{lean_tag}.lean")
    out.parent.mkdir(parents=True, exist_ok=True)
    head = LAYERS_HEAD.format(tag=lean_tag, delta=delta, ns=lean_tag)
    if do_lean:
        build_schemas()
    result = dict(tag=tag, delta=delta, explained=False, lemmas=[], items=[])
    if cands and lab["status"] == "forced":
        r = cands[0]
        item = lean_root_instance(r, mover, delta, out, head, do_lean, log)
        prem = item["prem"]
        if (not do_lean) or item["proved"]:
            log(f"OBS EXPLAINED BY THE ROOT: {tag}: the capture {r['san']} takes {r['worth']} (>= {delta}), nothing of ours is attacked after it, no pawn of theirs promotes, "
                f"and the solver confirms; the instance of `hanging` is " + ("proved in Lean" if do_lean else "not run (--no-lean)"))
            result.update(explained=True, capture=r["san"], items=[item])
            lm = dict(name="hanging", size=len(prem), proved=bool(do_lean and item["proved"]), explained=True)
            if dx:
                lm.update(drill_report("hanging", prem, poss, dx, cx, lm["proved"], log))
            result["lemmas"] = [lm]
            update_score(tag, result["lemmas"], log)
            return result
    # the root at the victim's LATER moves inside the window (43.2: the sequence may begin before the hanging piece does -- a check that forks, then the piece is left)
    later = later_root(win, base, mover, delta, ref, n_opp, log) if (not result["explained"]) else None
    if later:
        r, base2, need2, lab2 = later
        item = lean_root_instance(r, mover, need2, out, LAYERS_HEAD.format(tag=lean_tag, delta=need2, ns=lean_tag), do_lean, log)
        if (not do_lean) or item["proved"]:
            log(f"OBS EXPLAINED BY THE ROOT: {tag}: at the played move {base2['san']} (inside the window, after the example) the capture {r['san']} takes {r['worth']} (>= {need2}), "
                f"nothing of ours is attacked after it, no pawn of theirs promotes, and the solver confirms; the instance of `hanging` is " + ("proved in Lean" if do_lean else "not run (--no-lean)"))
            result.update(explained=True, capture=r["san"], items=[item])
            lm = dict(name="hanging", size=len(item["prem"]), proved=bool(do_lean and item["proved"]), explained=True)
            if do_drill:
                dr = drill(base2, need2, n_opp, log)
                poss2 = [base2] + dr
                dx2 = [i for i in range(1, len(poss2)) if poss2[i]["cls"] == "SAME"]
                cx2 = [i for i in range(1, len(poss2)) if poss2[i]["cls"] == "AVOIDED"]
                lm.update(drill_report("hanging", item["prem"], poss2, dx2, cx2, lm["proved"], log))
            result["lemmas"] = [lm]
            update_score(tag, result["lemmas"], log)
            return result
    # root does not explain
    best = max(reads, key=lambda r: r["k"]) if reads else None
    if (lab["status"] == "forced" or sol["status"] == "forced") and (best is None or best["k"] < delta):
        log(f"OBS root UNDER-PREDICTS: the solver forces a gain >= {delta} (in {lab['moves'] if lab['status'] == 'forced' else sol['moves']} moves) but the best capture reads k = {best['k'] if best else 0} < {delta}"
            + (f" ({best['san']})" if best else "") + ": a follow-up or a larger gain than the hanging piece; the deviation is searched next")
    elif lab["status"] != "forced" and best is not None and best["k"] >= delta:
        log(f"OBS root OVER-PREDICTS: capture {best['san']} reads k = {best['k']} >= {delta} but the solver does not confirm ({lab['status']}); the deviation is searched next")
    else:
        log(f"OBS root does not explain the window (solver {lab['status']}, best k {best['k'] if best else 0}); the deviation is searched next")
    # 44. THE SCHEMAS: diagnose the root at the solver's first move, bind the proved schemas, else repair by an operator and prove the new schema once
    schema_line = sol["line"] if sol["status"] == "forced" else (lab["line"] if lab["status"] == "forced" else None)
    if not schema_line:                                       # the solver did not decide: say where the root fails at the game's own move (no schema claims without the solver)
        pn = played_next(win, fen)
        if pn:
            need_ = max(1, delta + balance(ref, mover == "w") - balance(fen, mover == "w"))
            d_ = diagnose(fen, mover, need_, pn[1], pn[2])
            f_ = failing(d_)
            log(f"OBS diagnose {tag}: (the solver did not decide: {sol['status']}) root at the game's own move {pn[0]} ({pn[1]}->{pn[2]}) fails {{{', '.join(f_)}}}"
                + (" (" + "; ".join(CLAUSE[n].format(x=pn[1], y=pn[2], need=need_) for n in f_) + ")" if f_ else "; the root applies")
                + f"; operators called: {[op for op, _ in repair(d_)]}")
    if schema_line:
        n_ours = sol["moves"] if sol["status"] == "forced" else lab["moves"]
        need = max(1, delta + balance(ref, mover == "w") - balance(fen, mover == "w"))
        ours = line_pairs_of(fen, schema_line)[0::2]
        x0, y0 = ours[0]
        diag = diagnose(fen, mover, need, x0, y0)
        fails = failing(diag)
        log(f"OBS diagnose {tag}: root at {schema_line[0]} ({x0}->{y0}) fails {{{', '.join(fails)}}}"
            + (" (" + "; ".join(CLAUSE[n].format(x=x0, y=y0, need=need) for n in fails) + ")" if fails else "; the root applies"))
        maxd = max(1, min(2, n_ours - 1))
        known = proved_schemas()

        def try_schema(depth: int, how: str, family: str = "delayed_root"):
            """bind the schema of this family and depth; if an instance holds: drill, prove the schema (once), prove the instance by `exact`. -> True when it explains."""
            cands = [ours[:depth]] if depth >= 2 else None
            if depth >= 2 and len(ours) < depth:
                return False
            t0 = time.time()
            insts = bind(depth, fen, mover, need, cands, family)
            insts.sort(key=lambda i: (i["pairs"][0] != ours[0], i["pairs"]))
            name = schema_name(depth, family)
            log(f"OBS bind {name} ({how}): {len(insts)} binding(s) of {schema_sentence(depth, family).split(':')[0]} hold at the example"
                + (f", first {dict(zip(SCHEMA_VARS, [q for pr in insts[0]['pairs'] for q in pr]))} with B={insts[0]['B']}, T={insts[0]['T']}" if insts else "") + f" ({round(time.time() - t0, 1)} s)")
            if not insts:
                return False
            inst = insts[0]
            if dx:
                dr = drill_report(name, inst["prem"], poss, dx, cx, False, log)
                if dr["disagreements"]:
                    log(f"OBS {name}: the drill disagrees with the instance on {dr['disagreements']} variant(s): the operator EXCLUDE would add an atom; not explained by this schema")
                    return False
            es = ensure_schema(depth, log, do_lean, family)
            log(f"OBS schema {name} in words: {schema_sentence(depth, family)}")
            if not es["proved"]:
                return False
            item = dict(name=f"{name}_instance", prem=inst["prem"], mover=mover,
                        doc=f"INSTANCE of {name} {dict(zip(SCHEMA_VARS, [q for pr in inst['pairs'] for q in pr]))}, B={inst['B']}, T={inst['T']}, v={need}")
            lean_round(out, head, [item], need)
            write_final(out, head, [item], need)
            rep = run_lean(out)
            state = "PROVES" if item["proved"] else ("TYPECHECKS (proof fails)" if item["typechecks"] else "FAILS to typecheck")
            log(f"OBS lean {item['name']}: {state}" + (f": {item['error']}" if item["error"] else "") + f" (by `exact Ideas.Mined.Schemas.{name}`)")
            log(f"OBS lean file {out.relative_to(LEAN_DIR.parent.parent)}: 1 theorem, final check {'clean' if not rep else str(len(rep)) + ' errors'}")
            if not item["proved"]:
                return False
            op = repair(diag)[0][0]
            log(f"OBS EXPLAINED BY SCHEMA {name} bound to " + ", ".join(f"{v}={q}" for v, q in zip(SCHEMA_VARS, [q for pr in inst["pairs"] for q in pr]))
                + f": {explanation(diag, op, depth, inst['pairs'], need, family)}; the instance is proved in Lean by `exact` of the schema theorem; the solver confirms a forced gain >= {need}")
            lm = dict(name=name, size=len(inst["atoms"]), proved=True, explained=True)
            if dx:
                lm.update(drill_report(name, inst["prem"], poss, dx, cx, True, log))
            result.update(explained=True, schema=name, items=[item], lemmas=[lm])
            update_score(tag, result["lemmas"], log)
            return True
        for family in FAMILIES:                               # (1) the schemas already proved, the smallest family first, the most general last
            for depth in range(1, maxd + 1):
                if schema_name(depth, family) in known and try_schema(depth, "proved in Schemas.lean", family):
                    return result
        for op, why in repair(diag):                          # (2) repair by the operator the diagnosis calls, prove the schema once
            if op == "EXCLUDE":
                break
            log(f"OBS repair {op} (for {{{', '.join(why)}}}): the gain is one move later under every reply; depths 1..{maxd} (the solver needs {n_ours} moves of ours)")
            for family in FAMILIES:
                for depth in range(1, maxd + 1):
                    if schema_name(depth, family) not in known and try_schema(depth, op, family):
                        return result
        log("OBS no delayed-root schema explains the window; the operator EXCLUDE (an extra atom on the root's seed) is tried next")
    # 3. EXCLUDE (44.2), the root's atoms for the solver's first move when it captures, else for the best capture, plus one more atom: SEEDED by the root's atoms for the solver's first move when it captures, else for the best capture
    first = sol["line"][0] if sol["status"] == "forced" else None
    chosen = None
    if first and sol["status"] == "forced":
        fx, fy = line_pairs_of(fen, sol["line"])[0]
        chosen = next((r for r in reads if (r["x"], r["y"]) == (fx, fy)), None)
    chosen = chosen or best
    if chosen is None:
        log("OBS deviation: no capture to seed the search; falling back")
    items = []
    if chosen is not None and solver_ok:
        x, y = chosen["x"], chosen["y"]
        line_pairs = line_pairs_of(fen, sol["line"])
        objs, pairs = objects(win, False)
        seed = [(("count", ("and", "theirs", ("at", y))), {">="}), (("worth", y), {">="}), (("after", x, y, ("count", ("and", "ours", "hit"))), {"<="})]
        terms = make_terms(objs, pairs, line_pairs)
        have = {BV.sexpr(t) for t in terms}
        terms += [t for t, _ in seed if BV.sexpr(t) not in have]
        for t in list(terms):
            for g in legs_of(t):
                if BV.sexpr(g) not in have and BV.sexpr(g) not in {BV.sexpr(u) for u in terms}:
                    terms.append(g)
        t0 = time.time()
        vals = evaluate(terms, poss)
        prem = seeded_premises(terms, vals, poss, ex, cx, seed, extra=1)
        covs = {id(pr): coverage(pr, poss, ks) for pr in prem}
        gen = {id(pr): sum(all(a.truth[i] for a in pr) for i in dx) for pr in prem}
        prem.sort(key=lambda pr: (len(pr), -gen[id(pr)], -len(covs[id(pr)]), sum(a.size for a in pr), BV.sexpr(pr[-1].term)))
        seen_c, uniq = set(), []
        for pr in prem:
            c = canon(pr, mover)
            if c not in seen_c:
                seen_c.add(c)
                uniq.append(pr)
        log(f"OBS deviation: seed = the root's atoms for {chosen['san']} ({x}->{y}: a piece of theirs there, its worth, nothing of ours attacked after); {len(terms)} terms on "
            f"{len(poss)} positions in {round(time.time() - t0, 1)} s; {len(uniq)} premises = seed + at most 1 more atom exclude all {len(cx)} counter-examples")
        for r_, pr in enumerate(uniq[:top], 1):
            log(f"OBS deviation top {r_} (size {len(pr)}, {len(pr) - 3} beyond the seed, covers plies k={covs[id(pr)]}" + (f", {gen[id(pr)]}/{len(dx)} drilled forced variants" if dx else "")
                + f"): {words(pr, mover)}")
        for r_, pr in enumerate(uniq[:min(top, lean_cap)], 1):
            items.append(dict(name=f"conjecture_{r_:02d}", prem=pr, mover=mover, doc=f"root + deviation: {words(pr, mover)} (covers plies k={covs[id(pr)]})"))
        if do_lean and items:
            t0 = time.time()
            lean_round(out, head, items, delta)
            write_final(out, head, items, delta)
            for it in items:
                state = "PROVES" if it["proved"] else ("TYPECHECKS (proof fails)" if it["typechecks"] else "FAILS to typecheck")
                log(f"OBS lean {it['name']}: {state}" + (f": {reason(it)}" if it["error"] else ""))
            log(f"OBS lean round: {len(items)} statements in {round(time.time() - t0, 1)} s")
        for it in items:
            lm = dict(name=f"{tag}:{it['name']}", size=len(it["prem"]), proved=bool(it.get("proved")), explained=False)
            if dx:
                lm.update(drill_report(lm["name"], it["prem"], poss, dx, cx, lm["proved"], log))
            result["lemmas"].append(lm)
        result["items"] = items
    # the sound fallback: the 41.9 tree lemma
    if fallback and do_lean and not any(it.get("proved") for it in items):
        log("OBS fallback: no root+deviation conjecture is proved; the 41.9 tree search (premises read along the solver's line) runs for the sound statement")
        fb = mine(win, tag + "_fb", top, lean_cap, True, do_drill=do_drill, tree=True, log=log)
        result["fallback"] = dict(proved=fb["proved"], typechecked=fb["typechecked"])
        for it in fb["items"]:
            if not it["proved"]:
                continue
            log(f"OBS EXPLAINED BY TREE LEMMA {it['name']} (squares named, so no reuse; proved in Lean): {words(it['prem'], mover)}")
            lm = dict(name=f"{tag}:fb_{it['name']}", size=len(it["prem"]), proved=True, explained=True)
            if dx:
                lm.update(drill_report(lm["name"], it["prem"], poss, dx, cx, True, log))
            result["lemmas"].append(lm)
    update_score(tag, result["lemmas"], log)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sHa6yUKv")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--lean-cap", type=int, default=20, help="statements sent to Lean per stage")
    ap.add_argument("--no-lean", action="store_true")
    ap.add_argument("--no-tree", action="store_true", help="skip the premises read along the line (least / most over the replies)")
    ap.add_argument("--layers", action="store_true", help="the ladder (plan 42.5): the root first, then the deviation seeded by the root, then the tree fallback")
    ap.add_argument("--drill", action="store_true", help="the perturbation drill: one-piece variants of the example, labelled by the solver, as counter-examples")
    a = ap.parse_args()
    win = load_window(a.game, a.window)
    t0 = time.time()
    if a.layers:
        layers(win, f"{a.game}_{a.window}" + ("_drill" if a.drill else ""), a.top, a.lean_cap, not a.no_lean, do_drill=a.drill)
    else:
        mine(win, f"{a.game}_{a.window}" + ("_drill" if a.drill else ""), a.top, a.lean_cap, not a.no_lean, do_drill=a.drill, tree=not a.no_tree)
    print(f"OBS total {round(time.time() - t0, 1)} s")


if __name__ == "__main__":
    main()
