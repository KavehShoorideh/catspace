"""THE TOWER (Kaveh 2026-10-14: "start from a king, and then ask, how can I mate a king? ... if I'm unable, I bring my
strongest piece in. If I'm successful, I bring their pieces in ... a binary search in the space to find the areas where
it's edgy ... those are the sharp positions ... the frames are not really absolute ... relational, like a sliding piece
attacking this from four squares away"). docs/ANALYST_PLAN.md section 38.

The surprise loop learns in whatever order surprise brings; the tower learns in the order the rules make easy, side by side
with it (nothing in the loop changes):

  LADDER    from K v K: draw positions of a material (us to move) and ask the exact solver (goals.rs forced_goal) for a mate
            within k of our moves. None forced: bring our strongest piece in. All forced: bring one of theirs. Some forced,
            some not: the material is SHARP at k -- the band where the conditions live
  FLOOR     floor 0 is the mate; floor n is the positions from which some lower floor is FORCED within k of our moves
            against every reply. Its CONDITION is found in the base vocabulary (no named square first: the relative words --
            dfile, drank, between, nearer, xray, attackers -- are "a sliding piece attacking from four squares away"): true
            on the most forced positions and on no unforced one, refined on the leaks (CEGIS), then tried on fresh draws
  TOWER     the floors of one material, as a frame dict in the loop's own format: "<material>:if:<condition>" -> [measure,
            "horizon:k"]. The claim "the measure falls within k against every reply, or the goal" is Ideas.Forced.
            forced_by_measure (lean/ideas/Ideas/ForcedByMeasure.lean): the chooser already plays frames
  STATUS    <= 3 men: every placement inside the condition is checked by the solver -- all forced is a LEMMA (a finite
            exact check, not a Lean proof); the failures become unforced rows and the condition is refined once; what
            still fails, and everything sampled, is a CONJECTURE
  DEFENCES  bring THEIR piece in at the floor: the positions that stop being forced, and the reply that stops them (capture,
            interpose, flee, counter-threat, stalemate); the word that names the blocker, or a VOCABULARY GAP (printed,
            never patched -- the vocabulary is Kaveh's)
  ABLATION  Kaveh's metric, lemmas found and what each changed: the same starts played with the tower, with none, with the
            tower minus each floor, the tablebase defending

  python -m catspace.incubator.analyst.discovery.tower --material KQvK --horizons 1,2,3
  python -m catspace.incubator.analyst.discovery.tower --ladder --horizons 1,2,3
  python -m catspace.incubator.analyst.discovery.tower --material KRvK --horizons 2,3 --ablate
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import random
import time

import chess

from catspace.io import paths

BUDGET = 2_000_000
MATE = [("(count edge)", 1, 0)]          # a literal that never holds: the goal "a mate" (a mate counts whatever the literals say)
BIG = 4_294_967_295                      # u32::MAX, the upper bound of a ">=" literal
MIN_RIGHT = 20                           # forced positions wanted before a floor's condition is searched for
MAX_LITS = 4
HELD_OUT = 300
ENUMERATE_MEN = 3                        # <= this many men: every placement inside the condition is checked
SAMPLE, SAMPLE_HELD = 600, 300           # beyond: a sample and a held-out set (a conjecture)
THEIRS = "PNBRQ"                         # their pieces, weakest first
OURS = "QRBNP"                           # ours, strongest first
FLOORS = 6


def workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)           # all cores but one (Kaveh 2026-10-08)


def make_pool(n: int | None = None):
    """A spawn pool (never from a `python -` heredoc: workers cannot re-import <stdin>; always an -m entry point)."""
    return multiprocessing.get_context("spawn").Pool(n or workers())


def _a():
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    return A


def obs(*a) -> None:
    print("OBS", *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- words

def cond_text(conds: list) -> str:
    return "&&".join(f"{c['term']}{c['op']}{c['theta']}" for c in conds)


def lits_of(conds: list) -> list:
    """A condition as the solver's literals (term, lo, hi); a bound below zero never holds (or always does)."""
    out = []
    for c in conds:
        th = c["theta"]
        if c["op"] == ">=":
            out.append((c["term"], max(th, 0), BIG))
        else:
            out.append((c["term"], 1, 0) if th < 0 else (c["term"], 0, th))
    return out


def in_words(conds: list) -> str:
    from . import base_vocabulary as V
    return " and ".join(f"{V.gloss(V.parse(c['term']))} {'at least' if c['op'] == '>=' else 'at most'} {c['theta']}" for c in conds) or "always"


def men(key: str) -> int:
    return len(key.replace("v", ""))


def material_key(ours: str, theirs: str) -> str:
    return "K" + "".join(sorted(ours, key=OURS.index)) + "v" + "K" + "".join(sorted(theirs, key=OURS.index))


def split_key(key: str) -> tuple:
    a, b = key.split("v")
    return a[1:], b[1:]


# ------------------------------------------------------------------------------------------------------------ canonical

def canonical(fen: str, translate: bool = True) -> str:
    """One arrangement counted once: the position, its left-right mirror (Ideas/Mirror.lean, forcedWithin_mirror -- exact) and,
    with `translate`, the pattern shifted across the board (explorer.translations: the same relative arrangement elsewhere).
    The smallest FEN of the orbit."""
    from . import explorer as E
    orbit = [fen, E.mirror_fen(fen)]
    if translate:
        orbit += E.translations(fen, 7) + E.translations(orbit[1], 7)
    return min(orbit)


def _dedupe(fens: list) -> list:
    """One of each position and its mirror (the mirror's answers are the same by the lemma)."""
    from . import explorer as E
    seen, out = set(), []
    for f in fens:
        k = min(f, E.mirror_fen(f))
        if k not in seen:
            seen.add(k); out.append(f)
    return out


# --------------------------------------------------------------------------------------------------------------- workers

def _label_chunk(args) -> list:
    """In a worker: for each position, "forced" (some goal is forced within k, the fewest moves) / "stopped" (every goal
    proved not within k) / "unknown" (the budget ran out and none was forced). A goal is a list of literals."""
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    fens, goals, k, budget = args
    A = _a()
    out = []
    for f in fens:
        best, unknown = None, False
        for lits in goals:
            st, n, _ = A.forced_goal(f, lits, k, budget)
            if st == "forced" and (best is None or n < best):
                best = n
            unknown |= st == "unknown"
        out.append(("forced", best) if best is not None else ("unknown", 0) if unknown else ("stopped", 0))
    return out


def _mate_children_chunk(fens: list) -> list:
    """In a worker: the children of each position by a move of ours -- the mated ones, the checks that are not mate, and a
    quiet one -- read as they stand, the side to move (the mated side) being "us" to the vocabulary (the engine reads no position
    with the side not to move in check: a mate is described from the side that suffers it)."""
    out = []
    for f in fens:
        b = chess.Board(f)
        mates, checks, quiet = [], [], []
        for m in list(b.legal_moves):
            b.push(m)
            (mates if b.is_checkmate() else checks if b.is_check() else quiet).append(b.fen())
            b.pop()
        out.append((mates, checks[:3], quiet[:2]))
    return out


def _chunks(xs: list, n: int) -> list:
    size = max(1, -(-len(xs) // (n * 4)))
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def label(pool, fens: list, goals: list, k: int, budget: int = BUDGET) -> list:
    """-> [(status, fewest moves)] per position, in order."""
    if not fens:
        return []
    ch = _chunks(fens, pool._processes)
    return [x for part in pool.map(_label_chunk, [(c, goals, k, budget) for c in ch]) for x in part]


# --------------------------------------------------------------------------------------------------------------- draws

def draw(key: str, n: int, rng, lower: list, pool=None, seen: set | None = None) -> list:
    """n random positions of the material, us to move, none of them in a lower floor already, none twice (a position and its
    mirror once)."""
    from . import explorer as E
    seen = seen if seen is not None else set()
    out = []
    for f in E.random_positions(key, n, rng):
        m = min(f, E.mirror_fen(f))
        if m not in seen:
            seen.add(m); out.append(f)
    for conds in lower:
        if out:
            out = [f for f, h in zip(out, E._holds(out, conds)) if not h]
    return out


# ------------------------------------------------------------------------------------------------------ the separator

LAMBDAS = (1.0, 2.0, 4.0, 16.0)           # how much one wrong position costs against one right one (tried in turn)


def find_condition(right: list, wrong: list, mask=None, max_lits: int = MAX_LITS, ops=(">=", "<=")) -> dict:
    """A conjunction of the base vocabulary's conditions "term >= theta" / "term <= theta", true on the most RIGHT positions and
    false on every WRONG one (the explorer's separator, `separate`, for a set of positions instead of one surprise):
    greedily, the literal that keeps the most right for the wrong it lets through (a wrong costing LAMBDA rights, several
    LAMBDAs tried: a strict one reaches no wrong at all with a sliver of the right -- the first cut, 2026-10-14, covered 0 of 35).
    The result with no wrong left in and the most right kept; else the fewest wrong left. -> the conditions, how many right it
    keeps, which wrong it leaves in (the fens; empty is the goal)."""
    import numpy as np
    from . import base_vocabulary as V
    from . import explorer as E
    T = E.terms()
    sizes = np.array([V.size(V.parse(t)) for t in T])
    allow = mask if mask is not None else np.ones(len(T), bool)
    R = E.matrix(right).astype(np.int16) if right else np.zeros((0, len(T)), np.int16)
    W = E.matrix(wrong).astype(np.int16) if wrong else np.zeros((0, len(T)), np.int16)
    nv = int(max(R.max(initial=0), W.max(initial=0))) + 2

    def hist(X, idx):
        if len(idx) == 0:
            return np.zeros((nv, len(T)), np.int64)
        flat = (X[idx].astype(np.int64) * len(T) + np.arange(len(T))).ravel()
        return np.bincount(flat, minlength=nv * len(T)).reshape(nv, len(T))

    def greedy(lam):
        conds = []
        ra, wa = np.arange(len(right)), np.arange(len(wrong))
        while len(conds) < max_lits and len(wa) > 0 and len(ra) > 0:
            hr, hw = hist(R, ra), hist(W, wa)
            ge_r, ge_w = hr[::-1].cumsum(0)[::-1], hw[::-1].cumsum(0)[::-1]           # >= th
            le_r, le_w = hr.cumsum(0), hw.cumsum(0)                                   # <= th
            best = None
            for op, kr, kw in (("<=", le_r, le_w), (">=", ge_r, ge_w)):
                if op not in ops:
                    continue
                score = kr / len(ra) - lam * kw / len(wa)
                score = np.where((kw < len(wa)) & (kr > 0) & allow[None, :], score, -99.0)
                th, j = np.unravel_index(np.argmax(score - 1e-6 * sizes[None, :]), score.shape)
                cand = (float(score[th, j]), -int(sizes[j]), op, int(th), int(j))
                if best is None or cand[:2] > best[:2]:
                    best = cand
            if best is None or best[0] <= -99.0:
                break
            _, _, op, th, j = best
            conds.append(dict(term=T[j], op=op, theta=th))
            keep_r = (R[ra, j] >= th) if op == ">=" else (R[ra, j] <= th)
            keep_w = (W[wa, j] >= th) if op == ">=" else (W[wa, j] <= th)
            ra, wa = ra[keep_r], wa[keep_w]
        return dict(conds=conds, kept=int(len(ra)), of=len(right), leaks=[wrong[i] for i in wa])
    tries = [greedy(lam) for lam in LAMBDAS]
    return min(tries, key=lambda r: (len(r["leaks"]), -r["kept"], len(r["conds"])))


def condition_for(right: list, wrong: list, rounds: int = 4) -> dict:
    """The separator with CEGIS (as the explorer's check does): the layer without a named square first, then with them;
    every wrong position the condition lets through is added to the sample and the search repeated."""
    import numpy as np
    from . import explorer as E
    names = set(E.named_squares(" ".join(E.terms())))
    masks = [("layer 0: no named square", E.layer_mask(0, set())), ("layer 1: named squares", E.layer_mask(1, names))]
    res = None
    for name, mask in masks:
        samp_w = wrong[:500]
        for rnd in range(rounds):
            res = find_condition(right[:400], samp_w, mask)
            leaks = [f for f, h in zip(wrong, E._holds(wrong, res["conds"])) if h] if res["conds"] else list(wrong)
            if not leaks or rnd == rounds - 1:
                break
            samp_w = samp_w + [f for f in leaks if f not in samp_w][:100]
        res["layer"] = name
        res["leaks"] = leaks
        if res["conds"] and not leaks:
            return res
    return res


def measure_for(low: list, high: list) -> str:
    """The floor's measure: a word that is small on the lower floors and larger on this one (the best "<=" literal) -- it falls
    as the floor number does. The mate has no lower floor of positions to read: the floor-1 frame has no measure of its own."""
    if not low or not high:
        return "(count free)"
    r = find_condition(low[:400], high[:500], None, 1, ops=("<=",))
    return r["conds"][0]["term"] if r["conds"] else "(count free)"


# ------------------------------------------------------------------------------------------------------------ floor 0

def floor_zero(key: str, pool, rng, n: int = 4000) -> dict:
    """The mate in words: positions after our mating move (right) against positions after a check that is not mate and a
    quiet move (wrong), read from the side to move (the mated side: "our king is attacked and ..."). The condition that holds
    on mated positions only."""
    fens = _dedupe(draw(key, n, rng, [], pool))
    ch = _chunks(fens, pool._processes)
    right, wrong = [], []
    for part in pool.map(_mate_children_chunk, ch):
        for m, c, q in part:
            right += m; wrong += c + q
    right = list(dict.fromkeys(right)); wrong = list(dict.fromkeys(wrong))
    res = condition_for(right, wrong) if right and wrong else dict(conds=[], kept=0, of=len(right), leaks=wrong, layer="-")
    holds_r = sum(_holds_all(right, res["conds"])) if res["conds"] else 0
    holds_w = sum(_holds_all(wrong, res["conds"])) if res["conds"] else 0
    return dict(n=0, k=0, _right=right[:400], conds=res["conds"], text=cond_text(res["conds"]), words=in_words(res["conds"]) if res["conds"] else "-",
                layer=res.get("layer"), mated=len(right), not_mated=len(wrong), mated_held=holds_r, not_mated_held=holds_w,
                examples=right[:3], status="CONDITION" if res["conds"] and holds_w == 0 else "NONE")


def _holds_all(fens: list, conds: list) -> list:
    from . import explorer as E
    return E._holds(fens, conds) if fens and conds else [False] * len(fens)


# ------------------------------------------------------------------------------------------------------------- the floors

def _goals(floors: list) -> list:
    return [MATE] + [lits_of(f["conds"]) for f in floors]


def floor(key: str, k: int, floors: list, pool, rng, n: int = 2000, max_draws: int | None = None, min_right: int = MIN_RIGHT) -> dict | None:
    """The next floor of the tower of `key` at horizon k, given the floors below: draw positions of the material outside them,
    label each exactly (a lower floor forced within k of our moves, or not), find the condition in words, try it on fresh draws.
    None when no position of the material is forced (the tower ends)."""
    t0 = time.time()
    goals = _goals(floors)
    lower = [f["conds"] for f in floors]
    seen = set()
    rows, drawn, max_draws = [], 0, max_draws or n * 25
    while drawn < max_draws:
        fens = draw(key, n, rng, lower, pool, seen)
        drawn += n
        rows += list(zip(fens, label(pool, fens, goals, k)))
        if sum(s == "forced" for _, (s, _) in rows) >= min_right:
            break
    right = [f for f, (s, _) in rows if s == "forced"]
    wrong = [f for f, (s, _) in rows if s == "stopped"]
    unknown = sum(s == "unknown" for _, (s, _) in rows)
    nth = len(floors) + 1
    obs(f"{key} k={k} floor {nth}: drew {drawn}, kept {len(rows)}: forced {len(right)}, stopped {len(wrong)}, unknown {unknown}")
    out = dict(n=nth, k=k, material=key, drawn=drawn, forced=len(right), stopped=len(wrong), unknown=unknown,
               arrangements=len({canonical(f) for f in right[:200]}), _right=right[:400], _wrong=wrong[:500])
    if len(right) < 3 or not wrong:
        out.update(status="NO FLOOR", conds=[], text="", seconds=round(time.time() - t0, 1))
        return None if not right else out
    res = condition_for(right, wrong)
    conds = res["conds"]
    out.update(conds=conds, text=cond_text(conds), words=in_words(conds) if conds else "-", layer=res.get("layer"),
               kept=res["kept"], of=res["of"], leaks=len(res["leaks"]), examples=right[:3])
    if not conds or res["leaks"]:
        gap = res["leaks"] or wrong
        obs(f"{key} k={k} floor {nth}: VOCABULARY GAP -- no word separates; {len(res['leaks'])} of {len(wrong)} unforced positions stay inside "
            f"the best condition ({cond_text(conds) or 'none'}); ten of them:")
        for f in gap[:10]:
            obs("   ", f)
        out.update(status="VOCABULARY GAP", gap=gap[:10], seconds=round(time.time() - t0, 1))
        return out
    inside = _holds_all(right, conds)
    out["covers"] = f"{sum(inside)}/{len(right)}"
    # fresh draws, never seen: the condition's precision and recall (held out)
    h = draw(key, HELD_OUT, rng, lower, pool, seen)
    hl = label(pool, h, goals, k)
    hin = _holds_all(h, conds)
    out["held_out"] = dict(drawn=len(h), inside=sum(hin), inside_forced=sum(i and s == "forced" for i, (s, _) in zip(hin, hl)),
                           inside_stopped=sum(i and s == "stopped" for i, (s, _) in zip(hin, hl)),
                           forced_outside=sum((not i) and s == "forced" for i, (s, _) in zip(hin, hl)))
    obs(f"{key} k={k} floor {nth}: {cond_text(conds)}  [{out['words']}]  covers {out['covers']} forced; held out: {out['held_out']}")
    out["seconds"] = round(time.time() - t0, 1)
    return out


# -------------------------------------------------------------------------------------------------------------- status

def enumerate_inside(key: str, conds: list, lower: list, pool) -> list:
    """Every legal placement of the material (us to move) inside `conds` and outside the lower floors (<= 3 men)."""
    from catspace.incubator.analyst.material_walk import _pieces
    ours, theirs = _pieces(key)
    kinds = [(p, True) for p in ours] + [(p, False) for p in theirs]
    out, batch = [], []

    def flush():
        nonlocal batch
        ok = _holds_all(batch, conds)
        batch = [f for f, o in zip(batch, ok) if o]
        for c in lower:
            if batch:
                batch = [f for f, h in zip(batch, _holds_all(batch, c)) if not h]
        out.extend(batch); batch = []
    for sq in itertools.permutations(range(64), 2 + len(kinds)):
        b = chess.Board(None)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, True)); b.set_piece_at(sq[1], chess.Piece(chess.KING, False))
        bad = False
        for s_, (p, col) in zip(sq[2:], kinds):
            if p == chess.PAWN and chess.square_rank(s_) in (0, 7):
                bad = True; break
            b.set_piece_at(s_, chess.Piece(p, col))
        if bad or not b.is_valid() or b.is_game_over():
            continue
        batch.append(b.fen())
        if len(batch) >= 20000:
            flush()
    flush()
    return out


def status(f: dict, floors: list, pool, rng, ladder_goal=None) -> dict:
    """LEMMA: the claim checked on every placement inside the condition (<= 3 men; the failures are made wrong rows and the
    condition refined once); CONJECTURE: sampled inside the condition and held out. Sets f["status"]."""
    key, k = f["material"], f["k"]
    goals = _goals(floors)
    lower = [x["conds"] for x in floors]
    t0 = time.time()
    if f.get("status") in ("VOCABULARY GAP", "NO FLOOR"):
        return f
    for attempt in range(2):
        if men(key) <= ENUMERATE_MEN:
            fens = enumerate_inside(key, f["conds"], lower, pool)
            exhaustive = True
        else:
            fens = _dedupe(draw_inside(key, SAMPLE + SAMPLE_HELD, rng, f["conds"], lower))
            exhaustive = False
        st = label(pool, fens, goals, k)
        fail = [x for x, (s, _) in zip(fens, st) if s != "forced"]
        unknown = sum(s == "unknown" for s, _ in st)
        obs(f"{key} k={k} floor {f['n']}: {'every placement' if exhaustive else 'sampled'} inside {f['text']}: {len(fens)} checked, "
            f"{len(fens) - len(fail)} forced, {len(fail)} not ({unknown} unknown)")
        f["check"] = dict(exhaustive=exhaustive, checked=len(fens), failures=len(fail), unknown=unknown, attempt=attempt, examples=fail[:5])
        if not fail:
            f["status"] = "LEMMA" if exhaustive else "CONJECTURE"
            break
        f["status"] = "CONJECTURE"
        if attempt == 0 and fail:                                   # refine once: the failures are unforced rows now
            right = [x for x, (s, _) in zip(fens, st) if s == "forced"]
            seen_right = [r for r in f.get("_right", [])] + right
            res = condition_for(seen_right or right, fail + f.get("_wrong", []))
            if res["conds"] and not res["leaks"]:
                f["conds"] = res["conds"]; f["text"] = cond_text(res["conds"]); f["words"] = in_words(res["conds"])
                f["refined"] = True
                f["held_out_before_refinement"] = f.pop("held_out", None)
                f["covers"] = f"{sum(_holds_all(f['_right'], res['conds']))}/{len(f['_right'])}"
                continue
            break
    f["status_seconds"] = round(time.time() - t0, 1)
    return f


def draw_inside(key: str, n: int, rng, conds: list, lower: list) -> list:
    from . import explorer as E
    out = E.random_positions(key, n, rng, conds)
    for c in lower:
        if out:
            out = [f for f, h in zip(out, E._holds(out, c)) if not h]
    return out



# ------------------------------------------------------------------------------------------------------------ defences

def _pairs(floors: list) -> list:
    """The goals of the next floor, with their conditions: the mate (None), and every floor below."""
    return [(MATE, None)] + [(lits_of(f["conds"]), f["conds"]) for f in floors]


def _forced_from(d: chess.Board, pairs: list, kk: int, budget: int) -> bool:
    """Our turn at d with kk moves of ours left: a goal holds already, or is forced within kk."""
    from . import explorer as E
    fen = d.fen()
    A = _a()
    for lits, conds in pairs:
        if conds is not None and E._holds([fen], conds)[0]:
            return True
        if kk >= 1 and A.forced_goal(fen, lits, kk, budget)[0] == "forced":
            return True
    return False


def refuters(c: chess.Board, pairs: list, k: int, budget: int) -> list:
    """Their replies at c (after our move) that leave no goal forced within k-1 more moves of ours: [(move, kind)]; empty when
    the move forces (a mate counts). The kind of a reply: capture / interpose / flee / counter-threat / stalemate / quiet."""
    if c.is_checkmate():
        return []
    if c.is_stalemate():
        return [(None, "stalemate")]
    out = []
    for r in list(c.legal_moves):
        d = c.copy(stack=False); d.push(r)
        if not _forced_from(d, pairs, k - 1, budget):
            out.append((r, _reply_kind(c, r)))
    return out


def _reply_kind(c: chess.Board, r: chess.Move) -> str:
    """What kind of defence a reply is (c: the position before it, their move) -- the names of the plan: capture, interpose
    (a check met by a piece between), flee (the king moves), counter-threat (the reply gives check), stalemate."""
    if c.is_capture(r):
        return "capture"
    if c.piece_type_at(r.from_square) == chess.KING:
        return "flee"
    if c.is_check():
        return "interpose"
    d = c.copy(stack=False); d.push(r)
    if d.is_check():
        return "counter-threat"
    if d.is_stalemate():
        return "stalemate"
    return "quiet"


def _defence_chunk(args) -> list:
    """In a worker: a piece of theirs added at a random empty square of each position; -> [(fen, with the piece, kind)]: "still"
    when a goal is still forced, else the kind of defence that most often refutes our forcing first moves."""
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    fens, piece, pairs, k, budget, seed = args
    rng = random.Random(seed)
    out = []
    for f in fens:
        base = chess.Board(f)
        for _ in range(8):
            g = base.copy(stack=False)
            sq = rng.choice([s for s in chess.SQUARES if g.piece_at(s) is None])
            if piece == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            g.set_piece_at(sq, chess.Piece(piece, chess.BLACK))
            if not g.is_valid() or g.is_game_over():
                continue
            first = [m for m in base.legal_moves if not refuters(_after(base, m), pairs, k, budget)]
            if not first:
                break                                                       # (the floor's row is not forced by a move: skip)
            kinds, holds = [], False
            for m in first:
                if m not in g.legal_moves:
                    kinds.append("blocked")                                 # (the piece stands in the way of the move)
                    continue
                refs = refuters(_after(g, m), pairs, k, budget)
                holds |= not refs
                kinds += [kd for _, kd in refs]
            out.append((f, g.fen(), "still" if holds else max(set(kinds), key=kinds.count)))
            break
    return out


def _after(b: chess.Board, m: chess.Move) -> chess.Board:
    c = b.copy(stack=False); c.push(m)
    return c


def defences(key: str, k: int, f: dict, floors: list, pool, rng, budget: int = 200_000, rows: int = 150) -> list:
    """Bring THEIR piece in at floor `f` (weakest first; 5 men at most): the positions of the floor with a piece of theirs
    added that stop being forced, and the reply that stops them. The blocker named by a word (a condition true on every
    stopped position, false on every still-forced one) -- or a VOCABULARY GAP: ten stopped positions printed, the branch
    stopped, no word added (the vocabulary is Kaveh's)."""
    out = []
    if men(key) >= 5 or not f.get("_right"):
        return out
    pairs = _pairs(floors[:f["n"] - 1])
    ours, theirs = split_key(key)
    for pc in THEIRS:
        sub = f["_right"][:rows]
        res = [x for part in pool.map(_defence_chunk, [(c, chess.PIECE_SYMBOLS.index(pc.lower()), pairs, k, budget, rng.randrange(10 ** 9))
                                                      for c in _chunks(sub, pool._processes)]) for x in part]
        stopped = [g for _, g, kind in res if kind != "still"]
        still = [g for _, g, kind in res if kind == "still"]
        kinds = {}
        for _, _, kind in res:
            if kind != "still":
                kinds[kind] = kinds.get(kind, 0) + 1
        rec = dict(piece=pc, material=material_key(ours, theirs + pc), tried=len(res), stopped=len(stopped), still=len(still), kinds=kinds)
        obs(f"{key} k={k} floor {f['n']} + their {chess.piece_name(chess.PIECE_SYMBOLS.index(pc.lower()))}: {len(res)} tried, "
            f"{len(stopped)} no longer forced, {len(still)} still; stopped by {kinds}")
        if len(stopped) >= 5 and still:
            r = condition_for(stopped, still)
            if r["conds"] and not r["leaks"]:
                rec.update(blocker=cond_text(r["conds"]), blocker_words=in_words(r["conds"]))
                obs(f"   the blocker in words: {rec['blocker_words']}")
            else:
                rec.update(blocker=None, gap=stopped[:10])
                obs(f"{key} k={k} floor {f['n']} + {pc}: VOCABULARY GAP -- no word separates the stopped from the still-forced; ten stopped:")
                for g in stopped[:10]:
                    obs("   ", g)
        elif len(stopped) >= 5:
            rec.update(blocker="nothing stays forced with this piece")
        out.append(rec)
    return out


# -------------------------------------------------------------------------------------------------------------- the tower

def tower(key: str, k: int, pool, rng, n: int = 2000, floors: int = FLOORS, with_defences: bool = True) -> dict:
    """The tower of one material at one horizon: floor 0 (the mate in words), the floors 1.. each with its condition, evidence
    and status; the frames dict in the loop's format, the measure of a floor a word small on the floors below and larger
    on it (the floor number, as far as the vocabulary can say it)."""
    t0 = time.time()
    z = floor_zero(key, pool, rng, n=max(n, 2000))
    obs(f"{key} floor 0 (the mate): {z['text'] or 'none'}  [{z['words']}]; mated {z['mated']} (condition holds on {z['mated_held']}), "
        f"not mated {z['not_mated']} (condition holds on {z['not_mated_held']})")
    built = []
    while len(built) < floors:
        f = floor(key, k, built, pool, rng, n)
        if f is None:
            break
        built.append(f)
        if f.get("status") in ("VOCABULARY GAP", "NO FLOOR"):
            break
    for i, f in enumerate(built):
        if f.get("conds") and f.get("status") not in ("VOCABULARY GAP", "NO FLOOR"):
            status(f, built[:i], pool, rng)
            obs(f"{key} k={k} floor {f['n']}: {f['status']}")
    frames = {}
    for i, f in enumerate(built):
        if not f.get("conds") or f.get("status") in ("VOCABULARY GAP", "NO FLOOR"):
            continue
        if i == 0:                         # (floor 1's goal is the mate itself, which the chooser's own proof reaches: no frame)
            f["measure"] = None
            continue
        low = [x for g in built[:i] for x in g.get("_right", [])]
        f["measure"] = measure_for(low, f.get("_right", []))
        f["frame"] = f"{key}:if:{f['text']}"
        frames[f["frame"]] = [f["measure"], f"horizon:{k}"]
        obs(f"{key} k={k} floor {f['n']}: frame {f['frame']} -> [{f['measure']}]")
    out = dict(material=key, k=k, floor0=z, floors=built, frames=frames)
    if with_defences:
        out["defences"] = {}
        for f in built:
            if f.get("status") in ("LEMMA", "CONJECTURE"):
                out["defences"][str(f["n"])] = defences(key, k, f, built, pool, rng)
    out["seconds"] = round(time.time() - t0, 1)
    obs(f"{key} k={k}: {len([f for f in built if f.get('frame')])} floors, {sum(f.get('status') == 'LEMMA' for f in built)} LEMMA, "
        f"{sum(f.get('status') == 'CONJECTURE' for f in built)} CONJECTURE, {out['seconds']} s")
    return out


def _clean(x):
    """The JSON of a tower: the positions kept as examples only."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def save(key: str, towers: list) -> str:
    p = paths.derived(f"tower_{key}.json")
    frames = {}
    for t in towers:
        frames.update(t["frames"])
    with open(p, "w") as fh:
        json.dump(_clean(dict(material=key, frames=frames, towers=towers)), fh, indent=1)
    return p


# ---------------------------------------------------------------------------------------------------------------- ladder

def ladder(start: str = "KvK", pool=None, horizons=(1, 2, 3), n: int = 200, seed: int = 0, max_men: int = 5, budget: int = 200_000) -> list:
    """The binary search over material (Kaveh: "if I'm unable, I bring my strongest piece in. If I'm successful, I bring their
    pieces in"): at each material and horizon the share of drawn positions from which a mate is forced. Mixed anywhere:
    SHARP (and their pieces are brought in, weakest first, to see where the band goes). All forced at the longest horizon:
    their weakest unvisited piece in. None: our strongest unvisited piece in. At most `max_men` men (the tablebases end there)."""
    from . import explorer as E
    rng = random.Random(seed)
    own = pool is None
    pool = pool or make_pool()
    try:
        seen, queue, rows = set(), [start], []
        while queue:
            key = queue.pop(0)
            if key in seen:
                continue
            seen.add(key)
            fens = _dedupe(E.random_positions(key, n, rng))
            per = {}
            for k in horizons:
                st = label(pool, fens, [MATE], k, budget)
                per[k] = dict(forced=sum(s == "forced" for s, _ in st), stopped=sum(s == "stopped" for s, _ in st),
                              unknown=sum(s == "unknown" for s, _ in st))
            mixed = [k for k, v in per.items() if v["forced"] and v["stopped"]]
            top = per[max(horizons)]
            verdict = "sharp" if mixed else "all forced" if top["forced"] and not top["stopped"] else "none forced" if not top["forced"] else "mixed"
            obs(f"LADDER {key}: " + "; ".join(f"k={k} forced {v['forced']}/{len(fens)} unknown {v['unknown']}" for k, v in per.items()) +
                f" -> {verdict}" + (f" (sharp at k={','.join(map(str, mixed))})" if mixed else ""))
            rows.append(dict(material=key, drawn=len(fens), per_horizon=per, verdict=verdict, sharp_at=mixed))
            ours, theirs = split_key(key)
            kids = []
            if verdict == "sharp" or verdict == "all forced":
                kids += [material_key(ours, theirs + c) for c in (THEIRS if verdict == "sharp" else THEIRS[:1])]
                if verdict == "all forced":                                 # (the weakest not yet visited)
                    kids = [material_key(ours, theirs + c) for c in THEIRS if material_key(ours, theirs + c) not in seen][:1]
            elif verdict == "none forced":
                kids += [material_key(ours + c, theirs) for c in OURS if material_key(ours + c, theirs) not in seen][:1]
            queue += [c for c in kids if men(c) <= max_men and c not in seen]
        return rows
    finally:
        if own:
            pool.close(); pool.join()


# ------------------------------------------------------------------------------------------------------------- ablation

SEEDS = range(5000, 5100)
BAND = (1, 256)


def _outcome(g: dict) -> tuple:
    return (g["out"] == "win", g["moves"])


def ablate(key: str, frames: dict, seeds=SEEDS, band=BAND, opponent: str = "tablebase", only: list | None = None) -> dict:
    """The same starts (seeds, band), the tablebase defending, the hand-derived mating frames off (CATSPACE_SCRATCH): played with
    no frames, with the tower, and with the tower minus each floor. -> per variant mated / moves to mate, and better / worse
    than no frames paired by seed (better: mated where none did not, or in fewer moves; worse: the reverse)."""
    from . import surprise_loop as S
    os.environ.setdefault("CATSPACE_SCRATCH", "tower")
    variants = {"none": {}, "tower": frames}
    for name in (only or list(frames)):
        variants[f"minus {name}"] = {k: v for k, v in frames.items() if k != name}
    res, out = {}, {}
    for name, fr in variants.items():
        t0 = time.time()
        games = S.play(key, len(seeds), fr, seeds[0], opponent=opponent, band=band)
        res[name] = {g["seed"]: g for g in games}
        mated = [g for g in games if g["out"] == "win"]
        out[name] = dict(mated=len(mated), games=len(games), moves=round(sum(g["moves"] for g in mated) / max(1, len(mated)), 2),
                         seconds=round(time.time() - t0, 1))
        base = res["none"]
        better = worse = 0
        for s, g in res[name].items():
            a, b = _outcome(base[s]), _outcome(g)
            if a == b:
                continue
            better += (b[0] and not a[0]) or (b[0] and a[0] and b[1] < a[1])
            worse += (a[0] and not b[0]) or (b[0] and a[0] and b[1] > a[1])
        out[name].update(better=int(better), worse=int(worse))
        obs(f"ABLATION {key} {name}: mated {out[name]['mated']}/{out[name]['games']}, moves to mate {out[name]['moves']}, "
            f"better {int(better)} worse {int(worse)} vs none ({out[name]['seconds']} s)")
    return out


# ------------------------------------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--material", default=None, help="e.g. KQvK")
    ap.add_argument("--horizons", default="1,2,3")
    ap.add_argument("--ladder", action="store_true", help="the search over material from K v K")
    ap.add_argument("--draws", type=int, default=2000, help="positions drawn per batch (a floor draws up to 25 batches until it has 20 forced)")
    ap.add_argument("--floors", type=int, default=FLOORS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", action="store_true", help="play the tower against none and against the tower minus each floor")
    ap.add_argument("--ablate-saved", action="store_true", help="the ablation on the frames already saved for --material (no new tower)")
    ap.add_argument("--no-defences", action="store_true")
    ap.add_argument("--max-men", type=int, default=5)
    a = ap.parse_args(argv)
    os.environ.setdefault("CATSPACE_SCRATCH", "tower")        # (the from-scratch run: the hand-derived mating frames off, every worker)
    ks = [int(x) for x in a.horizons.split(",")]
    t0 = time.time()
    with make_pool() as pool:
        if a.ladder:
            rows = ladder("KvK", pool, ks, n=min(a.draws, 400), seed=a.seed, max_men=a.max_men)
            p = paths.derived("tower_ladder.json")
            json.dump(rows, open(p, "w"), indent=1)
            obs(f"ladder: {len(rows)} materials, {sum(r['verdict'] == 'sharp' for r in rows)} sharp, {round(time.time() - t0, 1)} s -> {p}")
        if a.material and a.ablate_saved:
            p = paths.derived(f"tower_{a.material}.json")
            d = json.load(open(p))
            d["ablation"] = ablate(a.material, d["frames"])
            json.dump(d, open(p, "w"), indent=1)
        elif a.material:
            rng = random.Random(a.seed)
            towers = [tower(a.material, k, pool, rng, a.draws, a.floors, not a.no_defences) for k in ks]
            p = save(a.material, towers)
            obs(f"{a.material}: {sum(len(t['floors']) for t in towers)} floors in {round(time.time() - t0, 1)} s -> {p}")
            if a.ablate:
                frames = {}
                for t in towers:
                    frames.update(t["frames"])
                ab = ablate(a.material, frames)
                d = json.load(open(p))
                d["ablation"] = ab
                json.dump(d, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
