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

DEPTH = 12                    # the horizon: moves of ours
BUDGET = 2_000_000            # nodes per first move
RADIUS = 2                    # one piece one square, twice
CAP = 600                     # explored positions at most (the decision first, then by distance)
_FRAMES = None


def _install(frames: dict) -> None:
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    global _FRAMES
    A.set_generic_frame(True)
    for m, terms in frames.items():
        A.set_material_frame(m, terms)
    _FRAMES = frames


def _value(fen: str) -> int:
    """The goal: the smallest gain that matters at this balance (the outcome curve), held after their reply -- or a mate."""
    import analyst_core as A
    b = chess.Board(fen)
    val = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
    lead = sum(val[p.piece_type] * (1 if p.color == b.turn else -1) for p in b.piece_map().values())
    return A.gain_floor_of(lead)


def labels(fen: str, depth: int = DEPTH, budget: int = BUDGET) -> dict:
    """{uci: (status, our moves, line)} for every legal move -- exact (forced / stopped) or unknown."""
    import analyst_core as A
    return {u: (st, n, line) for u, st, n, line in A.forced_moves(fen, _value(fen), depth, budget)}


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
        if any(x[0] == "forced" for x in lab.values()) and lab.get(u, ("unknown",))[0] == "stopped":
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
        if not any(x[0] == "forced" for x in lab.values()) or u not in lab:
            continue
        if lab[u][0] == "forced":
            right.append(r)
        elif lab[u][0] == "stopped":
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


def _cond_text(conds: list) -> str:
    return "&&".join(f"{c['term']}{c['op']}{c['theta']}" for c in conds)


def _holds(fens: list, conds: list) -> list:
    import analyst_core as A
    ok = [True] * len(fens)
    for c in conds:
        v = A.vocab_values(fens, c["term"])
        ok = [o and ((x >= c["theta"]) if c["op"] == ">=" else (x <= c["theta"])) for o, x in zip(ok, v)]
    return ok


def _second(at: str, out: list, keep: list) -> dict | None:
    """A condition true at the surprise, false on every position of `out`, true on the most of `keep`; simplest next."""
    import analyst_core as A
    from . import base_vocabulary as V
    fens = [at] + out + keep
    best = None
    for t in V.enumerate_numbers():
        s = V.sexpr(t)
        v = A.vocab_values(fens, s)
        x, o, k = v[0], v[1:1 + len(out)], v[1 + len(out):]
        for op, th in ((">=", max(o) + 1 if o else 0), ("<=", min(o) - 1 if o else 99)):
            if (op == ">=" and x >= th) or (op == "<=" and x <= th):
                cov = sum((y >= th) if op == ">=" else (y <= th) for y in k)
                c = dict(term=s, op=op, theta=th, covers=cov, size=V.size(t))
                if best is None or (cov, -c["size"]) > (best["covers"], -best["size"]):
                    best = c
    return best


def repair(key: str, cond: dict, rows: list, at: dict, top: int = 200, refine: int = 5) -> dict | None:
    """The exception's measure: with the stage "<key>:if:<condition>" installed, the planner must play a forced move at
    the surprise and break no explored position it had right; among those, the measure fixing the most positions it had
    wrong, then the simplest. A measure that fixes the surprise but breaks some is REFINED (Ripple-Down): a second
    condition true at the surprise and false on every position it broke, re-checked in the planner. Candidates: the
    measures reading lower after a forced move than after the move played at the surprise, simplest first."""
    import analyst_core as A
    from . import base_vocabulary as V
    good = lambda r, u: r["labels"].get(u, ("unknown",))[0] == "forced"
    forced = [u for u, x in at["labels"].items() if x[0] == "forced"]
    after = [_after_us(at["fen"], u) for u in forced] + [_after_us(at["fen"], at["played"])]
    cands = []
    for t in V.enumerate_numbers():
        s = V.sexpr(t)
        v = A.vocab_values(after, s)
        if min(v[:-1]) < v[-1]:
            cands.append((V.size(t), s))
    cands.sort()

    def trial(conds: list, s: str) -> dict | None:
        rs = [r for r, h in zip(rows, _holds([r["fen"] for r in rows], conds)) if h and any(x[0] == "forced" for x in r["labels"].values())]
        stage = f"{key}:if:{_cond_text(conds)}"
        A.set_material_frame(stage, [s])
        try:
            if planner_move(at["fen"])[0] not in forced:
                return None
            now = {r["fen"]: good(r, planner_move(r["fen"])[0]) for r in rs}
        finally:
            A.set_material_frame(stage, [])
        was = {r["fen"]: good(r, r["move"][0]) for r in rs}
        return dict(stage=stage, measure=s, conds=conds, inside=len(rs),
                    broke=[f for f in was if was[f] and not now[f]], fixed=[f for f in was if now[f] and not was[f]],
                    wrong_before=sum(not x for x in was.values()), right_before=sum(was.values()))

    tried = [x for x in (trial([cond], s) for _, s in cands[:top]) if x]
    clean = [x for x in tried if not x["broke"]]
    if clean:
        return max(clean, key=lambda x: (len(x["fixed"]), -V.size(V.parse(x["measure"]))))
    for x in sorted(tried, key=lambda x: (len(x["broke"]), -len(x["fixed"])))[:refine]:
        c2 = _second(at["fen"], x["broke"], x["fixed"])
        if c2 is None:
            continue
        y = trial([cond, c2], x["measure"])
        if y and not y["broke"]:
            y["refined_from"] = dict(broke=len(x["broke"]), fixed=len(x["fixed"]))
            return y
    return dict(stage=f"{key}:if:{_cond_text([cond])}", measure=None, tried=len(tried), inside=0)


def explore(game: dict, frames: dict, workers: int = 10) -> dict:
    """One surprise, end to end."""
    from multiprocessing import get_context
    t0 = time.time()
    _install(frames)
    at = locate(game)
    if at is None:
        return dict(seed=game.get("seed"), found=False, why="no decision threw away a gain our search proves forced")
    key = material(chess.Board(at["fen"]))
    fens = neighbours(at["fen"])
    with get_context("fork").Pool(workers, initializer=_install, initargs=(frames,)) as pool:
        rows = pool.map(_label_one, fens)
        # the mirror of each, labelled by the mirror lemma -- the planner's move there is asked, not assumed
        mir = [(mirror_fen(r["fen"]), mirror_labels(r["labels"])) for r in rows]
        have = {chess.Board(r["fen"]).board_fen() for r in rows}
        mir = [(f, lab) for f, lab in mir if chess.Board(f).board_fen() not in have]
        moves = pool.map(planner_move, [f for f, _ in mir])
    rows += [dict(fen=f, labels=lab, move=mv, mirrored=True) for (f, lab), mv in zip(mir, moves)]
    right, wrong = classify(rows)
    unknown = sum(1 for r in rows if r["move"][0] in r["labels"] and r["labels"][r["move"][0]][0] == "unknown")
    # THE CORNERSTONES (Ripple-Down Rules): the cases where the same kind of decision -- the same kind of piece moved --
    # was right; the exception is what tells the surprise apart from them
    kind = shape(at["fen"], at["played"])
    corner = [r for r in right if shape(r["fen"], r["move"][0]) == kind]
    same = [r for r in wrong if shape(r["fen"], r["move"][0]) == kind]
    seps = separate(corner, same, at["fen"])
    fixed = repair(key, seps[0], rows, at) if seps else None
    return dict(seed=game.get("seed"), found=True, material=key, decision=dict(fen=at["fen"], played=at["played"], why=at["why"],
                forced=[(u, x[1], x[2]) for u, x in at["labels"].items() if x[0] == "forced"]),
                explored=len(rows), mirrored=len(mir), right=len(right), wrong=len(wrong), unknown=unknown, kind=kind,
                cornerstones=len(corner), same_wrong=len(same),
                separators=seps, lemma=fixed, seconds=round(time.time() - t0, 1))


def report(x: dict) -> None:
    from . import base_vocabulary as V
    if not x["found"]:
        print(f"OBS seed {x['seed']}: {x['why']}")
        return
    d = x["decision"]
    b = chess.Board(d["fen"])
    print(f"OBS seed {x['seed']} ({x['material']}): at {d['fen']} the planner played {b.san(chess.Move.from_uci(d['played']))} "
          f"({d['why'][:80]}); forced instead: " + "; ".join(f"{b.san(chess.Move.from_uci(u))} in {n} ({' '.join(l[:6])})" for u, n, l in d["forced"]))
    print(f"OBS   explored {x['explored']} positions ({x['mirrored']} by the mirror lemma): the planner right on {x['right']}, "
          f"wrong on {x['wrong']}, unknown {x['unknown']} ({x['seconds']} s)")
    print(f"OBS   the decision that fired: a {x['kind']} move -- right on {x['cornerstones']} explored positions (the cornerstones), "
          f"wrong on {x['same_wrong']}")
    for c in x["separators"][:3]:
        print(f"OBS   what was not looked at: {c['term']} {c['op']} {c['theta']} -- {V.gloss(V.parse(c['term']))} "
              f"(true at the surprise and on {c['covers']} of the {x['same_wrong']} wrong, on none of the {x['cornerstones']} cornerstones)")
    if not x["separators"]:
        print("OBS   no condition of the vocabulary separates them -- a gap")
    L = x["lemma"]
    if L and L.get("measure"):
        cond = " and ".join(f"{V.gloss(V.parse(c['term']))} {'>=' if c['op'] == '>=' else '<='} {c['theta']}" for c in L["conds"])
        print(f"OBS   LEMMA (an exception; a conjecture, exact on the {L['inside']} explored positions inside its condition): "
              f"{L['stage']} -> [{L['measure']}]")
        print(f"OBS     in words: while {cond}, follow {V.gloss(V.parse(L['measure']))}")
        print(f"OBS     the surprise then played right; of the planner's {L['wrong_before']} wrong decisions inside it fixes "
              f"{len(L['fixed'])}, of its {L['right_before']} right ones it breaks none"
              + (f" (refined: unrefined it broke {L['refined_from']['broke']})" if L.get("refined_from") else ""))
    elif L:
        print(f"OBS   no measure of the {L.get('tried')} that play the surprise right avoids breaking a right decision, refined or not")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="a results JSON of games (surprise_loop._game rows)")
    ap.add_argument("--material", default=None, help="only games starting in this material")
    ap.add_argument("--n", type=int, default=1, help="how many surprises (non-won games, in order)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from . import registry as R
    frames = R.load_frames()
    games = [g for g in json.load(open(a.games)) if g["out"] != "win"
             and (a.material is None or g["rows"][0]["material"] == a.material)]
    found = []
    for g in games[: a.n]:
        x = explore(g, frames)
        report(x)
        found.append(x)
    out = a.out or paths.derived(f"explorer/explored_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(found, open(out, "w"), indent=1)
    print(f"OBS -> {out}")


if __name__ == "__main__":
    main()
