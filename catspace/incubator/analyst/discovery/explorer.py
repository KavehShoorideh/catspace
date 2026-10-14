"""THE SURPRISE EXPLORER (Kaveh 2026-10-13: "why are we mining statistically? Why are we not reasoning? ... when I see one
surprise, I can ... explore the different lines, and then understand dilemmas, or understand what to do when this happens,
or what to do when that happens ... there must be something that I misunderstood, that I didn't look at").

One surprise -- a game of ours -- and no statistics:

  LOCATE    walk the game: the first decision of ours where our own search proves a gain forced and the move played
            throws it away -- every legal move labelled by goals.rs forced_moves: FORCED within n moves of ours against
            every defence (with the line), STOPPED (proved: not within n), UNKNOWN (the budget ran out)
  EXPLORE   the space around it: the decision and every position one piece one square away, and one more step; each
            labelled the same way, exact; the left-right mirror of each labelled by the mirror lemma
            (lean/ideas/Ideas/Mirror.lean, forcedWithin_mirror) instead of searched
  SEPARATE  where the planner's own move keeps the forced gain (RIGHT) and where it throws it away (WRONG): the base
            vocabulary's conditions "term >= theta" / "term <= theta" true on every WRONG position and on no RIGHT one --
            what was not looked at, with no error allowed on the explored space
  REPAIR    inside that condition, a measure under which the planner's move is forced on every explored position there:
            the stage "<material>:if:<condition>" -> [measure]
  STATE     the lemma and its evidence (positions explored, each checked; the line on the surprise) -- a conjecture,
            exact on the explored space; the next surprise tests it, Lean proves it later

  python -m catspace.incubator.analyst.discovery.explorer --games <results.json> --material KPvK
"""
from __future__ import annotations

import argparse
import json
import os
import time

import chess

from catspace.io import paths

DEPTH = 12                    # the horizon: moves of ours (the defence mode: of theirs, --depth)
MODE = "win"                  # "win": does our move keep a forced gain; "defence": does it stop their forced gain
BUDGET = 2_000_000            # nodes per first move
RADIUS = 2                    # one piece one square, twice
CAP = 600                     # explored positions at most (the decision first, then by distance)
_FRAMES = None


def _install(frames: dict, mode: str | None = None, depth: int | None = None) -> None:
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    global _FRAMES, MODE, DEPTH
    MODE = mode or MODE
    DEPTH = depth or DEPTH
    A.set_generic_frame(True)
    for m, terms in frames.items():
        A.set_material_frame(m, terms)
    _FRAMES = frames


def _value(fen: str, side: str = "us") -> int:
    """The goal: the smallest gain that matters at this balance to that side (the outcome curve), held after the reply --
    or a mate."""
    import analyst_core as A
    b = chess.Board(fen)
    val = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
    lead = sum(val[p.piece_type] * (1 if p.color == b.turn else -1) for p in b.piece_map().values())
    return A.gain_floor_of(lead if side == "us" else -lead)


def labels(fen: str, depth: int | None = None, budget: int = BUDGET) -> dict:
    """{uci: (status, moves, line)} for every legal move of ours -- exact or unknown. "win": FORCED (it keeps a gain of
    ours forced within the horizon) / STOPPED; "defence": STOPPED (it stops every gain of theirs that matters to them
    within the horizon -- their imagined future avoided) / FORCED (not avoided)."""
    import analyst_core as A
    d = depth or DEPTH
    if MODE == "defence":
        return {u: (st, n, line) for u, st, n, line in A.defence_moves(fen, _value(fen, "them"), d, budget)}
    return {u: (st, n, line) for u, st, n, line in A.forced_moves(fen, _value(fen), d, budget)}


def good(st: str) -> bool:
    return st == ("stopped" if MODE == "defence" else "forced")


def bad(st: str) -> bool:
    return st == ("forced" if MODE == "defence" else "stopped")


def planner_move(fen: str) -> tuple:
    """(uci, why) -- what the planner plays here, the frames installed (a fresh game: no history)."""
    import analyst_core as A
    st = A.PlanState()
    tr, dec, *_ = st.observe(fen)
    b = chess.Board(fen)
    for u in ((dec[0], dec[1]) if dec else ()):
        try:
            m = chess.Move.from_uci(u)
        except (ValueError, TypeError):
            continue
        if m in b.legal_moves:
            return u, dec[3]
    m = next(iter(b.legal_moves), None)
    return (m.uci() if m else None), "no move"


def _label_one(fen: str) -> dict:
    return dict(fen=fen, labels=labels(fen), move=planner_move(fen))


def material(b: chess.Board) -> str:
    from catspace.incubator.analyst.material_walk import _material
    return _material(b)


def locate(game: dict) -> dict | None:
    """The first decision of ours whose move threw away a gain our search proves forced (None: no such decision)."""
    for i, r in enumerate(game["rows"]):
        b = chess.Board(r["fen"])
        u = b.parse_san(r["san"]).uci()
        lab = labels(r["fen"])
        if any(good(x[0]) for x in lab.values()) and bad(lab.get(u, ("unknown",))[0]):
            return dict(index=i, fen=r["fen"], played=u, why=r["why"], labels=lab)
    return None


def neighbours(fen: str, radius: int = RADIUS, cap: int = CAP) -> list:
    """Positions one piece one square away (a king step for any piece; pawns never onto the first or last rank), the
    same material, the same side to move, legal and not over -- breadth first to `radius`, at most `cap`."""
    root = chess.Board(fen)
    key = material(root)
    seen, out, frontier = {root.board_fen()}, [fen], [root]
    steps = [(df, dr) for df in (-1, 0, 1) for dr in (-1, 0, 1) if (df, dr) != (0, 0)]
    for _ in range(radius):
        nxt = []
        for b in frontier:
            for sq, p in b.piece_map().items():
                f, r = chess.square_file(sq), chess.square_rank(sq)
                for df, dr in steps:
                    if not (0 <= f + df < 8 and 0 <= r + dr < 8):
                        continue
                    to = chess.square(f + df, r + dr)
                    if b.piece_at(to) or (p.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7)):
                        continue
                    x = b.copy(stack=False)
                    x.remove_piece_at(sq); x.set_piece_at(to, p)
                    x.castling_rights = 0; x.ep_square = None
                    if x.board_fen() in seen or not x.is_valid() or x.is_game_over() or material(x) != key:
                        continue
                    seen.add(x.board_fen())
                    out.append(x.fen()); nxt.append(x)
                    if len(out) >= cap:
                        return out
        frontier = nxt
    return out


def mirror_fen(fen: str) -> str:
    return chess.Board(fen).transform(chess.flip_horizontal).fen()


def mirror_labels(lab: dict) -> dict:
    """The mirror lemma (forcedWithin_mirror): the reflected move in the reflected position has the same label."""
    flip = lambda u: chess.square_name(chess.parse_square(u[:2]) ^ 7) + chess.square_name(chess.parse_square(u[2:4]) ^ 7) + u[4:]
    return {flip(u): (st, n, line) for u, (st, n, line) in lab.items()}


def shape(fen: str, u: str | None) -> str | None:
    """What kind of decision a move is: the kind of piece it moves (the rule that fired, in the planner's terms)."""
    if not u:
        return None
    p = chess.Board(fen).piece_at(chess.parse_square(u[:2]))
    return chess.piece_name(p.piece_type) if p else None


def classify(rows: list) -> tuple:
    """RIGHT: the planner's move keeps a forced gain; WRONG: a gain was forced and its move throws it away."""
    right, wrong = [], []
    for r in rows:
        lab, (u, _) = r["labels"], r["move"]
        if not any(good(x[0]) for x in lab.values()) or u not in lab:
            continue
        if good(lab[u][0]):
            right.append(r)
        elif bad(lab[u][0]):
            wrong.append(r)
    return right, wrong


def separate(right: list, wrong: list, at: str, top: int = 8) -> list:
    """What distinguishes the surprise from every case where the planner was right (Ripple-Down Rules: the new case
    against the cornerstones): the conditions of the base vocabulary true at the surprise `at`, false on every RIGHT
    position -- the threshold as loose as that allows, so it covers the most WRONG positions; most covered, then simplest."""
    import analyst_core as A
    from . import base_vocabulary as V
    fens = [at] + [r["fen"] for r in wrong] + [r["fen"] for r in right]
    nw = len(wrong)
    out = []
    for t in V.enumerate_numbers():
        s = V.sexpr(t)
        v = A.vocab_values(fens, s)
        x, w, rt = v[0], v[1:1 + nw], v[1 + nw:]
        if not rt:
            continue
        hi, lo = max(rt), min(rt)
        if x > hi:                                            # "term >= hi + 1": the surprise in, every right out
            th = hi + 1
            out.append(dict(term=s, op=">=", theta=th, covers=sum(y >= th for y in w), margin=x - hi, size=V.size(t)))
        if x < lo:
            th = lo - 1
            out.append(dict(term=s, op="<=", theta=th, covers=sum(y <= th for y in w), margin=lo - x, size=V.size(t)))
    out.sort(key=lambda c: (-c["covers"], c["size"], -c["margin"], c["term"]))
    return out[:top] if top else out


def _after_us(fen: str, u: str) -> str:
    """The position after our move, read from our side (the side to move set back to us: a static reading of the
    measure, as the frame reads it after its move)."""
    b = chess.Board(fen)
    b.push(chess.Move.from_uci(u))
    b.turn = not b.turn
    b.ep_square = None
    return b.fen()


_TERMS = None


def terms() -> list:
    """The base vocabulary's number terms, smallest first (S-expressions)."""
    global _TERMS
    if _TERMS is None:
        from . import base_vocabulary as V
        _TERMS = [V.sexpr(t) for t in V.enumerate_numbers()]
    return _TERMS


def matrix(fens: list):
    """Every term on every position, one call (rows: positions, columns: terms)."""
    import numpy as np
    import analyst_core as A
    return np.asarray(A.component_values(fens, terms()), dtype=np.int64).reshape(len(fens), len(terms()))


# THE PROOF GATES THE WORDS (explanation-based learning: Mitchell, Keller & Kedar-Cabelli 1986; Minton 1984 -- the
# condition comes from what the proof touches): layer 0, the words without a named square (the small, symmetric family
# first -- Quinlan & Cameron-Jones 1995 on oversearching); layer 1, named squares only those the surprise's own proof
# visits (its lines' squares and its pieces' squares) -- "e7" is allowed only if the proof goes through e7
_SQ = None


def named_squares(term: str) -> set:
    import re
    global _SQ
    _SQ = _SQ or re.compile(r"(?<![a-z])[a-h][1-8](?![0-9])")
    return set(_SQ.findall(term))


def proof_squares(at: dict) -> set:
    """The squares the surprise's proof touches, named from our side: its pieces, and every move of the lines that show
    the good moves (in the defence mode also their line after the move played)."""
    b0 = chess.Board(at["fen"])
    name = (lambda sq: chess.square_name(sq)) if b0.turn == chess.WHITE else (lambda sq: chess.square_name(sq ^ 56))
    out = {name(sq) for sq in b0.piece_map()}
    lines = [x[2] for x in at["labels"].values() if good(x[0])]
    if MODE == "defence":
        lines.append([chess.Board(at["fen"]).san(chess.Move.from_uci(at["played"]))] + list(at["labels"][at["played"]][2]))
    for line in lines:
        b = b0.copy()
        for san in line:
            try:
                m = b.parse_san(san)
            except ValueError:
                break
            out |= {name(m.from_square), name(m.to_square)}
            b.push(m)
    return out


def layer_mask(layer: int, proof: set):
    """Which terms a layer allows."""
    import numpy as np
    return np.array([(not named_squares(t)) if layer == 0 else named_squares(t) <= proof for t in terms()])


def _cond_text(conds: list) -> str:
    return "&&".join(f"{c['term']}{c['op']}{c['theta']}" for c in conds)


def _holds(fens: list, conds: list) -> list:
    import analyst_core as A
    ok = [True] * len(fens)
    for c in conds:
        v = A.vocab_values(fens, c["term"])
        ok = [o and ((x >= c["theta"]) if c["op"] == ">=" else (x <= c["theta"])) for o, x in zip(ok, v)]
    return ok


def _second(at: str, out: list, keep: list, mask=None) -> dict | None:
    """A condition true at the surprise, false on every position of `out`, true on the most of `keep`; simplest next."""
    import numpy as np
    from . import base_vocabulary as V
    M = matrix([at] + out + keep)
    x, o, k = M[0], M[1:1 + len(out)], M[1 + len(out):]
    hi = o.max(axis=0) if len(out) else np.full(M.shape[1], -1)
    lo = o.min(axis=0) if len(out) else np.full(M.shape[1], 10 ** 6)
    best = None
    allow = mask if mask is not None else np.ones(M.shape[1], bool)
    for op, th, ok in ((">=", hi + 1, (x >= hi + 1) & allow), ("<=", lo - 1, (x <= lo - 1) & allow)):
        cov = ((k >= th) if op == ">=" else (k <= th)).sum(axis=0) if len(keep) else np.zeros(M.shape[1], dtype=int)
        for j in np.flatnonzero(ok):
            c = (int(cov[j]), -V.size(V.parse(terms()[j])) if cov[j] >= (best or {}).get("covers", -1) else 0)
            if best is None or c > (best["covers"], -best["size"]):
                best = dict(term=terms()[j], op=op, theta=int(th[j]), covers=int(cov[j]), size=-c[1] if c[1] else V.size(V.parse(terms()[j])))
    return best


ALWAYS = dict(term="(count edge)", op=">=", theta=0)        # a condition that always holds (the edge has 28 squares)


def _trial(args) -> dict | None:
    """In a worker: the stage "<key>:if:<conds>" -> [measure] installed, the planner's move at the surprise and on every
    explored position with a good move -- which it now gets right that it had wrong (FIXED), wrong that it had right
    (BROKE). None when the surprise is still played wrong."""
    import analyst_core as A
    key, conds, s, rows, at = args
    ok = lambda r, u: good(r["labels"].get(u, ("unknown",))[0])
    inside = [r for r, h in zip(rows, _holds([r["fen"] for r in rows], conds)) if h]
    stage = f"{key}:if:{_cond_text(conds)}"
    A.set_material_frame(stage, [s])
    try:
        if not good(at["labels"].get(planner_move(at["fen"])[0], ("unknown",))[0]):
            return None
        now = {r["fen"]: ok(r, planner_move(r["fen"])[0]) for r in inside}
    finally:
        A.set_material_frame(stage, [])
    was = {r["fen"]: ok(r, r["move"][0]) for r in inside}
    return dict(stage=stage, measure=s, conds=conds, inside=len(inside),
                broke=[f for f in was if was[f] and not now[f]], fixed=[f for f in was if now[f] and not was[f]],
                wrong_before=sum(not x for x in was.values()), right_before=sum(was.values()))


def lemma(key: str, rows: list, at: dict, pool, mask=None, top: int = 40, refine: int = 8, keep: int = 8) -> list:
    """THE CANDIDATES, measure first: the measures (of the allowed words) under which the planner plays a good move at the
    surprise, each tried everywhere in the real planner (what it fixes, what it breaks); a measure that breaks nothing is
    a candidate as it is; one that breaks some gets the CONDITION keeping the surprise in and everything it broke out
    (Ripple-Down: true at the surprise, false on every broken case, true on the most fixed ones), re-checked. -> the
    committee: every candidate that breaks nothing, the most fixed first, then the simplest (at most `keep`)."""
    import numpy as np
    from . import base_vocabulary as V
    rows = [r for r in rows if any(good(x[0]) for x in r["labels"].values())]
    goods = [u for u, x in at["labels"].items() if good(x[0])]
    after = [_after_us(at["fen"], u) for u in goods] + [_after_us(at["fen"], at["played"])]
    M = matrix(after)
    allow = mask if mask is not None else np.ones(M.shape[1], bool)
    cands = sorted((V.size(V.parse(terms()[j])), terms()[j]) for j in range(M.shape[1]) if allow[j] and M[:-1, j].min() < M[-1, j])
    slim = [dict(fen=r["fen"], labels=r["labels"], move=r["move"]) for r in rows]
    tried = [x for x in pool.map(_trial, [(key, [ALWAYS], sx, slim, at) for _, sx in cands[:top]]) if x]
    out = [x for x in tried if not x["broke"]]
    for x in sorted([x for x in tried if x["broke"]], key=lambda x: (len(x["broke"]) - len(x["fixed"]), len(x["broke"])))[:refine]:
        c = _second(at["fen"], x["broke"], x["fixed"], mask)
        if c is None:
            continue
        y = _trial((key, [c], x["measure"], slim, at))
        if y and not y["broke"]:
            y["refined_from"] = dict(broke=len(x["broke"]), fixed=len(x["fixed"]))
            out.append(y)
    size = lambda y: V.size(V.parse(y["measure"])) + sum(V.size(V.parse(c["term"])) for c in y["conds"] if c != ALWAYS)
    out.sort(key=lambda y: (-len(y["fixed"]), size(y)))
    return out[:keep]


def _moves_under(args) -> list:
    """In a worker: the planner's moves on `fens` with the stage installed."""
    import analyst_core as A
    key, conds, measure, fens = args
    stage = f"{key}:if:{_cond_text(conds)}"
    A.set_material_frame(stage, [measure])
    try:
        return [planner_move(f)[0] for f in fens]
    finally:
        A.set_material_frame(stage, [])


def committee(key: str, cands: list, at: dict, pool, rng, rounds: int = 4, sample: int = 600) -> tuple:
    """QUERY BY COMMITTEE (Seung, Opper & Sompolinsky 1992; Mitchell's version space): fresh positions of the material where
    the candidates would play differently are labelled exactly; a candidate that breaks a right decision there is out.
    -> (the survivors, the labelled rows, the log)"""
    rows_all, log = [], []
    for rnd in range(rounds):
        if len(cands) <= 1:
            break
        fens = random_positions(key, sample, rng)
        chunks = [fens[i:i + 50] for i in range(0, len(fens), 50)]
        per = []
        for c in cands:
            ms = pool.map(_moves_under, [(key, c["conds"], c["measure"], ch) for ch in chunks])
            per.append([m for ch in ms for m in ch])
        split = [i for i in range(len(fens)) if len({p[i] for p in per}) > 1]
        if not split:
            log.append(dict(step=f"committee {rnd + 1}", candidates=len(cands), sampled=len(fens), disagree=0)); break
        rows = pool.map(_label_one, [fens[i] for i in split])
        rows_all += rows
        alive = []
        for c, p in zip(cands, per):
            broke = sum(good(r["labels"].get(r["move"][0], ("unknown",))[0]) and bad(r["labels"].get(p[i], ("unknown",))[0])
                        for i, r in zip(split, rows))
            fixed = sum(bad(r["labels"].get(r["move"][0], ("unknown",))[0]) and good(r["labels"].get(p[i], ("unknown",))[0])
                        for i, r in zip(split, rows))
            c.setdefault("committee", []).append(dict(broke=broke, fixed=fixed))
            if broke == 0:
                alive.append(c)
        log.append(dict(step=f"committee {rnd + 1}", candidates=len(cands), sampled=len(fens), disagree=len(split), survived=len(alive)))
        cands = alive
    return cands, rows_all, log


def fixes_decision(key: str, L: dict, d: dict) -> bool:
    """With the lemma installed, does the planner play a good move at a surprise's decision (`d`: fen, labels)?"""
    import analyst_core as A
    stage = f"{key}:if:{_cond_text(L['conds'])}"
    A.set_material_frame(stage, [L["measure"]])
    try:
        return good(d["labels"].get(planner_move(d["fen"])[0], ("unknown",))[0])
    finally:
        A.set_material_frame(stage, [])


# ---------------------------------------------------------------------------------------------------------------------
# THE PRACTICES (Kaveh 2026-10-13: "do all of the best practices from this kind of field and literature"), after three
# lemmas each exact on their neighbourhood and worth nothing net beyond it:
#   CORNERSTONES FROM EVERY CASE (Ripple-Down Rules, Compton & Jansen): every position ever labelled for the material is
#     kept (the case base); every right decision there is a cornerstone a new exception must not break
#   THE SAME PATTERN ELSEWHERE: the surprise shifted across the board is explored too -- a rule tied to one square fails
#   COUNTEREXAMPLES SOUGHT (CEGIS, Solar-Lezama): positions where the candidate's condition holds, drawn fresh, labelled,
#     the candidate re-checked; what it breaks joins the cases and the lemma is learned again
#   MINIMUM COVERAGE AND THE SIMPLEST (Occam, MDL): an exception must fix more than its own surprise; ties go to the
#     shortest description
#   HELD OUT: fresh positions inside the condition, never seen while learning -- it must break none and fix some
#   UTILITY (Minton's utility problem): fresh starts played with and without it -- kept only if it wins games net
# ---------------------------------------------------------------------------------------------------------------------
MIN_FIX = 3                   # an exception fixes this many cases besides its surprise
CEGIS_ROUNDS = 3
CEGIS_SAMPLE = 300
HELD_OUT = 300


def _case_path(key: str) -> str:
    return paths.derived(f"explorer/cases/{key}_{MODE}.jsonl")


def load_cases(key: str) -> list:
    p = _case_path(key)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def save_cases(key: str, rows: list) -> int:
    """Append the newly labelled positions (by board) to the material's case base; -> how many new."""
    p = _case_path(key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    have = {json.loads(l)["fen"].split(" ")[0] for l in open(p)} if os.path.exists(p) else set()
    n = 0
    with open(p, "a") as fh:
        for r in rows:
            b = r["fen"].split(" ")[0]
            if b in have:
                continue
            have.add(b); n += 1
            fh.write(json.dumps(dict(fen=r["fen"], labels=r["labels"])) + "\n")
    return n


def translations(fen: str, reach: int = 3) -> list:
    """The position shifted whole across the board (files and ranks), wherever it stays legal -- the same pattern elsewhere."""
    b = chess.Board(fen)
    key = material(b)
    out = []
    for df in range(-reach, reach + 1):
        for dr in range(-reach, reach + 1):
            if (df, dr) == (0, 0):
                continue
            x = chess.Board(None); x.turn = b.turn
            ok = True
            for sq, p in b.piece_map().items():
                f, r = chess.square_file(sq) + df, chess.square_rank(sq) + dr
                if not (0 <= f < 8 and 0 <= r < 8) or (p.piece_type == chess.PAWN and r in (0, 7)):
                    ok = False; break
                x.set_piece_at(chess.square(f, r), p)
            if ok and x.is_valid() and not x.is_game_over() and material(x) == key:
                out.append(x.fen())
    return out


def random_positions(key: str, n: int, rng, conds: list | None = None, tries: int = 200) -> list:
    """Random legal positions of the material, us to move -- inside the conditions when given (at most `tries` x n draws)."""
    from catspace.incubator.analyst.material_walk import _pieces
    ours, theirs = _pieces(key)
    out, draws = [], 0
    batch = []
    while len(out) < n and draws < tries * n:
        draws += 1
        sq = rng.sample(range(64), 2 + len(ours) + len(theirs))
        b = chess.Board(None)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE)); b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
        bad_ = False
        for i, (s_, kind) in enumerate(zip(sq[2:], ours + theirs)):
            if kind == chess.PAWN and chess.square_rank(s_) in (0, 7):
                bad_ = True; break
            b.set_piece_at(s_, chess.Piece(kind, chess.WHITE if i < len(ours) else chess.BLACK))
        b.turn = chess.WHITE
        if bad_ or not b.is_valid() or b.is_game_over():
            continue
        batch.append(b.fen())
        if len(batch) >= 512 or draws >= tries * n:
            keep = _holds(batch, conds) if conds else [True] * len(batch)
            out += [f for f, k in zip(batch, keep) if k]
            batch = []
    if batch:
        keep = _holds(batch, conds) if conds else [True] * len(batch)
        out += [f for f, k in zip(batch, keep) if k]
    return out[:n]


def _with_move(r: dict) -> dict:
    """A stored case with the planner's move now (the frames change as lemmas are kept)."""
    return dict(fen=r["fen"], labels=r["labels"], move=planner_move(r["fen"]))


def check(key: str, L: dict, at: dict, pool, fens: list) -> tuple:
    """Label `fens` exactly and try the lemma there: -> (the labelled rows, the trial: what it fixes and breaks)."""
    rows = pool.map(_label_one, fens)
    slim = [dict(fen=r["fen"], labels=r["labels"], move=r["move"]) for r in rows if any(good(x[0]) for x in r["labels"].values())]
    t = _trial((key, L["conds"] or [ALWAYS], L["measure"], slim, at)) if slim else None
    return rows, t


def explore(game: dict, frames: dict, workers: int = 10, seed: int = 0) -> dict:
    """One surprise, end to end -- with the practices above."""
    import random
    from multiprocessing import get_context
    t0 = time.time()
    _install(frames)
    mode, depth = MODE, DEPTH
    at = locate(game)
    if at is None:
        return dict(seed=game.get("seed"), found=False, why=("no decision let a gain of theirs through that a move of ours stops"
                    if MODE == "defence" else "no decision threw away a gain our search proves forced"))
    key = material(chess.Board(at["fen"]))
    # (an earlier exception may already handle it: Ripple-Down -- the case is met by the rules as they stand now)
    if good(at["labels"].get(planner_move(at["fen"])[0], ("unknown",))[0]):
        return dict(seed=game.get("seed"), found=False, why="handled: with the lemmas kept so far the planner plays it right",
                    decision=dict(fen=at["fen"], labels=at["labels"]), material=material(chess.Board(at["fen"])))
    rng = random.Random(f"{game.get('seed')}-{seed}")
    fens = neighbours(at["fen"]) + translations(at["fen"])
    log = []
    # (spawned, not forked: a fork inherits the Rust thread pool's state without its threads -- 2026-10-13, the workers hung in
    # vocab_values)
    with get_context("spawn").Pool(workers, initializer=_install, initargs=(frames, mode, depth)) as pool:
        rows = pool.map(_label_one, fens)
        mir = [(mirror_fen(r["fen"]), mirror_labels(r["labels"])) for r in rows]
        have = {chess.Board(r["fen"]).board_fen() for r in rows}
        mir = [(f, lab) for f, lab in mir if chess.Board(f).board_fen() not in have]
        moves = pool.map(planner_move, [f for f, _ in mir])
        rows += [dict(fen=f, labels=lab, move=mv, mirrored=True) for (f, lab), mv in zip(mir, moves)]
        right, wrong = classify(rows)
        unknown = sum(1 for r in rows if r["move"][0] in r["labels"] and r["labels"][r["move"][0]][0] == "unknown")
        kind = shape(at["fen"], at["played"])
        # THE CORNERSTONES: every case labelled before for this material, with the planner's move as it stands now
        base = [b for b in load_cases(key) if b["fen"].split(" ")[0] not in have]
        base = pool.map(_with_move, base)
        train = rows + base
        # THE WORDS IN LAYERS, GATED BY THE PROOF
        proof = proof_squares(at)
        cands, layer = [], None
        for lay in (0, 1):
            cands = lemma(key, train, at, pool, layer_mask(lay, proof))
            log.append(dict(step=f"layer {lay}", words="no named square" if lay == 0 else f"named squares of the proof ({len(proof)})",
                            cases=len(train), candidates=len(cands)))
            if cands:
                layer = lay; break
        # THE COMMITTEE: fresh positions where the candidates disagree, labelled; a candidate breaking one is out
        cands, crows, clog = committee(key, cands, at, pool, rng)
        log += clog
        sought = list(crows)
        # COUNTEREXAMPLES inside each survivor's own condition (the committee's draws rarely land there); the first clean one
        L = None
        for c in cands:
            if len(c["fixed"]) < MIN_FIX:
                log.append(dict(step="too narrow", measure=c["measure"], fixed=len(c["fixed"]))); continue
            ok = True
            for rnd in range(CEGIS_ROUNDS):
                cx_rows, t = check(key, c, at, pool, random_positions(key, CEGIS_SAMPLE, rng, c["conds"]))
                sought += cx_rows
                broke = len(t["broke"]) if t else 0
                log.append(dict(step=f"counterexamples {rnd + 1}", measure=c["measure"], broke=broke, fixed=len(t["fixed"]) if t else 0))
                if broke:
                    ok = False; break
                if t is None or len(cx_rows) < CEGIS_SAMPLE // 2:
                    break
            if ok:
                L = c; break
        # HELD OUT: fresh positions inside the condition, never used while learning
        hold = None
        if L:
            hold_fens = random_positions(key, HELD_OUT, random.Random(f"held-{game.get('seed')}-{seed}"), L["conds"])
            hold_rows, t = check(key, L, at, pool, hold_fens)
            hold = dict(sampled=len(hold_fens), cases=sum(any(good(x[0]) for x in r["labels"].values()) for r in hold_rows),
                        broke=len(t["broke"]) if t else 0, fixed=len(t["fixed"]) if t else 0)
            sought += hold_rows
        else:
            L = dict(stage=key, measure=None, rejected="no candidate survived the committee and the counterexamples")
        added = save_cases(key, rows + sought)
    return dict(seed=game.get("seed"), found=True, mode=mode, depth=depth, material=key,
                decision=dict(fen=at["fen"], played=at["played"], why=at["why"],
                forced=[(u, x[1], x[2]) for u, x in at["labels"].items() if good(x[0])],
                theirs=at["labels"].get(at["played"], (None, 0, []))[2] if mode == "defence" else None),
                explored=len(rows), mirrored=len(mir), right=len(right), wrong=len(wrong), unknown=unknown, kind=kind,
                cornerstones=len(base), log=log, held_out=hold, cases_added=added, layer=layer,
                lemma=L, seconds=round(time.time() - t0, 1))


def utility(key: str, frames: dict, L: dict, seeds: range, workers: int = 10) -> dict:
    """Fresh starts of the material played with the frames and with the frames plus the lemma, paired by game."""
    from multiprocessing import get_context
    from .surprise_loop import _game
    rank = {"loss": 0, "draw": 1, "not finished": 1, "win": 2}
    with_ = dict(frames); with_[L["stage"]] = [L["measure"]]
    with get_context("spawn").Pool(workers) as p:
        a = {g["seed"]: g["out"] for g in p.map(_game, [(s, key, frames, "tablebase", None) for s in seeds])}
        b = {g["seed"]: g["out"] for g in p.map(_game, [(s, key, with_, "tablebase", None) for s in seeds])}
    from .defenders import sign_e
    better = sum(rank[b[s]] > rank[a[s]] for s in a)
    worse = sum(rank[b[s]] < rank[a[s]] for s in a)
    return dict(games=len(a), better=better, worse=worse, net=better - worse, e=round(sign_e(better, worse), 2))


def report(x: dict) -> None:
    from . import base_vocabulary as V
    if not x["found"]:
        print(f"OBS seed {x['seed']}: {x['why']}")
        return
    d = x["decision"]
    b = chess.Board(d["fen"])
    if x.get("mode") == "defence":
        print(f"OBS seed {x['seed']} ({x['material']}, defence, their window {x['depth']}): at {d['fen']} the planner played "
              f"{b.san(chess.Move.from_uci(d['played']))} -- then their gain is forced: {' '.join(d['theirs'][:8])}; "
              f"stopped by: " + ", ".join(b.san(chess.Move.from_uci(u)) for u, n, l in d["forced"]))
    else:
        print(f"OBS seed {x['seed']} ({x['material']}): at {d['fen']} the planner played {b.san(chess.Move.from_uci(d['played']))}; "
              f"forced instead: " + "; ".join(f"{b.san(chess.Move.from_uci(u))} in {n} ({' '.join(l[:6])})" for u, n, l in d["forced"]))
    print(f"OBS   explored {x['explored']} positions ({x['mirrored']} by the mirror lemma) + {x['cornerstones']} earlier cases "
          f"of {x['material']}; the planner right on {x['right']}, wrong on {x['wrong']} of the explored ({x['seconds']} s)")
    for e in x["log"]:
        print("OBS     " + ", ".join(f"{k} {v}" for k, v in e.items()))
    L = x["lemma"]
    if L and L.get("measure"):
        cond = " and ".join(f"{V.gloss(V.parse(c['term']))} {c['op']} {c['theta']}" for c in L["conds"] if c is not ALWAYS and c != ALWAYS) or "always"
        print(f"OBS   LEMMA: {L['stage']} -> [{L['measure']}]")
        print(f"OBS     in words: while {cond}, follow {V.gloss(V.parse(L['measure']))}")
        print(f"OBS     learning cases: fixes {len(L['fixed'])} of the planner's {L['wrong_before']} wrong decisions inside, breaks "
              f"none of its {L['right_before']} right ones")
        h = x.get("held_out")
        if h:
            print(f"OBS     HELD OUT ({h['sampled']} fresh positions inside the condition, {h['cases']} with a good move): "
                  f"fixes {h['fixed']}, breaks {h['broke']}")
    elif L:
        print(f"OBS   no lemma: {L.get('rejected') or 'no measure that plays the surprise right keeps the right decisions'}")
    if x.get("utility"):
        u = x["utility"]
        print(f"OBS   UTILITY ({u['games']} fresh starts, paired): better in {u['better']} games, worse in {u['worse']} -- net {u['net']:+d}, "
              f"e {u['e']} -> {x['verdict']}")
    elif x.get("verdict"):
        print(f"OBS   -> {x['verdict']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="a results JSON of games (surprise_loop._game rows)")
    ap.add_argument("--material", default=None, help="only games starting in this material")
    ap.add_argument("--n", type=int, default=1, help="how many surprises (in order)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="win", choices=("win", "defence"))
    ap.add_argument("--depth", type=int, default=None, help="the horizon (moves of the attacker); default 12 win, 6 defence")
    ap.add_argument("--outcome", default="nonwin", choices=("nonwin", "loss"), help="which games are surprises")
    ap.add_argument("--utility", type=int, default=200, help="fresh starts for the utility check (0: none)")
    ap.add_argument("--utility-seed", type=int, default=20000)
    a = ap.parse_args(argv)
    global MODE, DEPTH
    MODE, DEPTH = a.mode, a.depth or (12 if a.mode == "win" else 6)
    from . import registry as R
    frames = R.load_frames()
    games = [g for g in json.load(open(a.games)) if (g["out"] != "win" if a.outcome == "nonwin" else g["out"] == "loss")
             and (a.material is None or g["rows"][0]["material"] == a.material)]
    from .defenders import E_PROMOTE
    found, kept, pending, decisions = [], [], [], []
    store = paths.derived("explorer/lemmas.json")
    os.makedirs(os.path.dirname(store), exist_ok=True)
    ledger = json.load(open(store)) if os.path.exists(store) else []
    for i, g in enumerate(games[: a.n]):
        # RIPPLE-DOWN: each surprise meets the lemmas kept so far (its decision may already be handled)
        x = explore(g, frames, seed=i)
        d = x.get("decision")
        if d:
            decisions.append(dict(seed=x["seed"], key=x["material"], fen=d["fen"], labels=d.get("labels") or labels(d["fen"])))
        L = x.get("lemma") if x["found"] else None
        h = x.get("held_out")
        if L and L.get("measure"):
            if h and h["broke"] == 0 and h["fixed"] > 0:
                pending.append(dict(lemma=L, key=x["material"], seed=x["seed"], x=x))
                x["verdict"] = "pending (needs a second surprise)"
            else:
                x["verdict"] = "dropped (held out)"
        # SUPPORT FROM DIFFERENT GAMES (the brief: at least two surprises): each pending lemma tried on every decision so far
        _install(frames)
        for p in pending:
            p["support"] = sorted({dd["seed"] for dd in decisions if dd["key"] == p["key"] and fixes_decision(p["key"], p["lemma"], dd)})
        for p in [p for p in pending if len(p["support"]) >= 2]:
            pending.remove(p)
            u = utility(p["key"], frames, p["lemma"], range(a.utility_seed + 1000 * i, a.utility_seed + 1000 * i + a.utility)) if a.utility else None
            p["x"]["utility"] = u
            if u and u["net"] > 0 and u["e"] >= E_PROMOTE:
                frames = dict(frames); frames[p["lemma"]["stage"]] = [p["lemma"]["measure"]]
                kept.append(p["lemma"]["stage"]); p["x"]["verdict"] = f"KEPT (support {p['support']})"
            else:
                p["x"]["verdict"] = f"dropped (utility e {u['e'] if u else '-'} < {E_PROMOTE} or net <= 0; support {p['support']})"
            ledger.append(dict(stage=p["lemma"]["stage"], measure=p["lemma"]["measure"], material=p["key"], seed=p["seed"],
                               support=p["support"], utility=u, verdict=p["x"]["verdict"], time=time.strftime("%Y-%m-%d %H:%M")))
            print(f"OBS   the lemma from seed {p['seed']} ({p['lemma']['stage']} -> [{p['lemma']['measure']}]): fixes the decisions of "
                  f"games {p['support']}; UTILITY {u} -> {p['x']['verdict']}")
        report(x)
        found.append(x)
    for p in pending:
        ledger.append(dict(stage=p["lemma"]["stage"], measure=p["lemma"]["measure"], material=p["key"], seed=p["seed"],
                           support=p.get("support"), verdict="pending at the end (one surprise only)", time=time.strftime("%Y-%m-%d %H:%M")))
    json.dump(ledger, open(store, "w"), indent=1)
    out = a.out or paths.derived(f"explorer/explored_{int(time.time())}.json")
    json.dump(found, open(out, "w"), indent=1, default=str)
    print(f"OBS {len(kept)} lemma(s) kept of {sum(1 for x in found if x.get('lemma') and x['lemma'].get('measure'))} learned "
          f"from {len(found)} surprises ({len(pending)} still pending on one surprise): {kept} -> {store}")


if __name__ == "__main__":
    main()
