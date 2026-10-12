"""THE VERIFIER'S OPPONENT, A SWITCH (Kaveh 2026-10-12: "up until now we've been playing against our own estimate of what
is best? ... we should play against Stockfish, not our own best"; "we should have this as a thing that could be
changed"). Stage 2 of the verifier plays the candidate measure from the hole's boards; who defends:

  self         the measure's adversary (Rust verify_play): material first, then our measure as high as it goes -- OUR OWN
               estimate of their best (the default until 2026-10-12; it never harasses with checks)
  tablebase    the Syzygy tablebases: the perfect defender -- the worst outcome for us, then the longest way to it (5 pieces
               or fewer; beyond the tables, Stockfish)
  stockfish:D  Stockfish at depth D, with the tablebases

OUR SIDE, a switch too: "frame" -- the candidate measure alone (Rust frame_choose); "planner" -- the whole player with the
candidate installed as its material's frame (the tactics first, the frame where nothing is proved: the lemma judged where
it is used; 2026-10-12: the frame alone won 28 / 30 boards against "self", 0 / 30 against the tablebase and Stockfish --
no measure wins a rook by itself against perfect defence). The rules of the loop are verify_play's: a mate, or (the
material's frame) a capture we keep, succeeds; ground lost after our move, a stalemate, a third occurrence, or no goal
within the limit fails.
"""
from __future__ import annotations

import os

from catspace.io import paths

DEFENDERS = ("self", "tablebase", "stockfish")
_TB = None
_ENG = None


def parse(defender: str) -> tuple:
    kind, _, arg = defender.partition(":")
    if kind not in DEFENDERS:
        raise ValueError(f"defender {defender!r}: one of self, tablebase, stockfish:<depth>")
    return kind, int(arg) if arg else 10


def _init(defender: str) -> None:
    global _TB, _ENG
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import chess.engine
    import chess.syzygy
    _TB = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    _ENG.configure({"SyzygyPath": str(paths.syzygy_dir())})


def reply(b, defender: str):
    """The defender's move (they to move on `b`)."""
    import chess.engine
    kind, depth = parse(defender)
    if kind == "tablebase" and len(b.piece_map()) <= 5:
        best = None
        for r in b.legal_moves:
            b.push(r)
            if b.is_checkmate():
                key = (-3, 0)                                  # they mate us
            else:
                try:
                    w, d = _TB.probe_wdl(b), _TB.probe_dtz(b)  # our side to move: lower is better for them
                except KeyError:
                    w, d = 0, 0
                key = (w, -d)                                  # the worst outcome for us, then the longest way to it
            b.pop()
            if best is None or key < best[0]:
                best = (key, r)
        return best[1]
    return _ENG.play(b, chess.engine.Limit(depth=depth)).move


def _material(b, white: bool) -> int:
    v = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
    return sum(v.get(p.piece_type, 0) for p in b.piece_map().values() if p.color == white)


def compose(base: list, term: str | None, schema: str) -> list:
    if not term:
        return list(base)
    if schema == "prepend":
        return [term] + base
    if schema == "append" or not base:
        return base + [term]
    if schema == "sum":
        return base[:-1] + [f"(sum {base[-1]} {term})"]
    return base[:-1] + [term]


def sym_key(b) -> str:
    """The position up to the board's symmetries (Kaveh 2026-10-12: "play ... until you're confident you're back to the
    same position or a mirror/rotation of it"): no pawns and no castling -- the 8 rotations and reflections; pawns -- the
    left-right mirror only; castling rights -- none. The side to move kept."""
    import chess
    bb = chess.BaseBoard(b.board_fen())
    if b.castling_rights:
        group = [lambda x: x]
    elif b.pawns:
        group = [lambda x: x, chess.flip_horizontal]
    else:
        fv, fh, fd, fa = chess.flip_vertical, chess.flip_horizontal, chess.flip_diagonal, chess.flip_anti_diagonal
        group = [lambda x: x, fv, fh, fd, fa, lambda x: fh(fv(x)), lambda x: fv(fd(x)), lambda x: fh(fd(x))]
    return min(bb.transform(f).board_fen() for f in group) + (" w" if b.turn else " b")


GUARD = 500                     # a safety cap only (logged when hit): every game ends at the goal, ground lost or a cycle


def _play_one(args):
    """One board, played to its NATURAL END (no move limit -- Kaveh: "principled, not numbered limits"): the goal (mate, or
    a capture we keep), ground lost (material, mated, a draw while ahead -- STOP AND SAY WHY: Kaveh, "when we lose ground
    we should stop and look at why"), or a CYCLE -- a position equivalent to an earlier one under the board's symmetries,
    the same side to move: no progress, the defender can repeat it (a draw: ground lost when ahead). -> dict: won, lost,
    why (the end's kind and its moves), our moves, the positions of ours where the FRAME moved (its claims) and every
    position of ours (to judge them)."""
    import chess
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    fen, base, cand, schema, piece, limit, defender, player, material = args
    b = chess.Board(fen)
    us = b.turn
    theirs0, lead0 = _material(b, not us), _material(b, us) - _material(b, not us)
    history, ours, framed, seen = [], [], [], {sym_key(b)}
    out = dict(won=False, lost=False, moves=0, why=None, states=0)
    # THE LEMMA-STATE LOOP (Kaveh: "moving to a new place where new lemmas are applying is progress ... cycling back to a
    # place where the same lemmas apply is a loop"): at our turn, the facts that hold (Rust lemma_state) with the
    # material; a state never seen is progress, a state seen again with the frame's measure LOWER than at its last
    # visit is progress (we stayed and squeezed), a state seen again without it is a LOOP -- no progress: the game stops
    terms = compose(base, cand, schema)
    best_at: dict = {}
    st = None
    if player == "planner":                       # the whole player, the candidate as its material's frame
        A.set_generic_frame(True)
        A.set_material_frame(material, compose(base, cand, schema))
        st = A.PlanState()

    def lose(kind, **kw):
        out["lost"] = True
        out["why"] = dict(kind=kind, **kw)

    def drawn(kind, **kw):                         # a draw: ground lost when ahead
        out["lost"] = lead0 > 0
        out["why"] = dict(kind=kind, **kw)

    for n in range(1, GUARD + 1):
        history.append(b.fen())
        ours.append((b.fen(), _material(b, not us), _material(b, us) - _material(b, not us)))
        before = b.fen()
        state = (A.lemma_state([before])[0], _material(b, us), _material(b, not us))
        meas = A.component_values([before], terms)[0] if terms else []
        if state in best_at and not meas < best_at[state]:
            drawn("loop", fen=before, facts=[f for f, v in A.lemma_facts_of(before) if v]); break
        if state not in best_at:
            out["states"] += 1
        best_at[state] = meas
        if st is not None:
            tr, dec, *_ = st.observe(b.fen())
            mv = None
            for u in ((dec[0], dec[1]) if dec else ()):
                try:
                    m = chess.Move.from_uci(u)
                except (ValueError, TypeError):
                    continue
                if m in b.legal_moves:
                    mv = m; break
            if mv is None:
                drawn("no move", fen=before); break
            if any(l.startswith("frame:") for l in tr):
                framed.append(len(ours) - 1)
        else:
            pick = A.frame_choose(b.fen(), base, cand, schema, piece, history)
            if pick is None:
                drawn("no move", fen=before); break
            mv = chess.Move.from_uci(pick[0])
            framed.append(len(ours) - 1)
        ours_san = b.san(mv)
        b.push(mv)
        out["moves"] = n
        if b.is_checkmate():
            out["won"] = True; break
        if piece == "*":
            if _material(b, us) - _material(b, not us) < lead0:
                lose("material", fen=before, ours=ours_san); break       # (our move left material behind)
            if _material(b, not us) < theirs0:
                out["won"] = True; break                                 # the goal: the next frame's material
        if b.is_stalemate():
            drawn("stalemate", fen=before, ours=ours_san); break
        k = sym_key(b)
        if k in seen:
            drawn("cycle", fen=before, ours=ours_san); break
        seen.add(k)
        history.append(b.fen())
        if st is not None:
            st.record(b.fen())
        r = reply(b, defender)
        theirs_san, check = b.san(r), b.gives_check(r)
        b.push(r)
        if b.is_checkmate():
            lose("mated", fen=before, ours=ours_san, theirs=theirs_san); break
        if piece != "*" and not any(p.color == us and p.piece_type != chess.KING for p in b.piece_map().values()):
            lose("material", fen=before, ours=ours_san, theirs=theirs_san); break
        if b.halfmove_clock >= 100:
            drawn("fifty moves", fen=before, ours=ours_san, theirs=theirs_san); break
        if b.is_stalemate():
            drawn("stalemate", fen=before, ours=ours_san, theirs=theirs_san); break
        k = sym_key(b)
        if k in seen:
            drawn("cycle (their checks)" if check else "cycle", fen=before, ours=ours_san, theirs=theirs_san); break
        seen.add(k)
    else:
        out["why"] = dict(kind="guard hit", fen=b.fen())   # never expected: logged
    if out["why"] and out["why"]["kind"] == "material":
        out["why"]["after"] = b.fen()
    out.update(ours=ours, framed=framed, end_won=out["won"])
    return out


def claims_kept(res: list, terms: list, k: int = 2) -> tuple:
    """The frame's claims along the games -- its measure falls within k moves of ours, or the goal comes -- judged against
    what the defender did: -> (kept, claims)."""
    import analyst_core as A
    kept = claims = 0
    for r in res:
        ours = r["ours"]
        vals = A.component_values([o[0] for o in ours], terms) if terms else [[] for _ in ours]
        for i in r["framed"]:
            claims += 1
            win = range(i + 1, min(i + 1 + k, len(ours)))
            if any(ours[j][1] < ours[i][1] and ours[j][2] >= ours[i][2] for j in win) or (r["won"] and len(ours) - 1 - i < k):
                kept += 1
            elif terms and any(vals[j] < vals[i] for j in win):
                kept += 1
    return kept, claims


WORKERS = max(1, (os.cpu_count() or 8) - 2)    # the cores, two left for the proposers' background training


def closed_loop(boards: list, base: list, cand: str | None, schema: str, piece: str, limit: int, defender: str,
                workers: int = WORKERS, player: str = "frame", material: str = "") -> tuple:
    """-> (games won, total moves of the won games, our moves played, our moves certified), as Rust verify_play."""
    import analyst_core as A
    if parse(defender)[0] == "self":
        if player != "frame":
            raise ValueError("the 'self' defender (the measure's adversary) lives in Rust and plays the frame alone")
        return tuple(A.verify_play(boards, base, cand, schema, piece, limit))
    from multiprocessing import Pool
    with Pool(workers, initializer=_init, initargs=(defender,)) as p:
        res = p.map(_play_one, [(f, base, cand, schema, piece, limit, defender, player, material) for f in boards], chunksize=2)
    kept, claims = claims_kept(res, compose(base, cand, schema))
    from collections import Counter
    whys = [r["why"] for r in res if r["lost"] and r["why"]]
    return dict(n=len(res), won=sum(r["won"] for r in res), lost=sum(r["lost"] for r in res),
                moves=sum(r["moves"] for r in res if r["won"]), played=sum(len(r["ours"]) for r in res), kept=kept, claims=claims,
                states=sum(r["states"] for r in res), why=dict(Counter(w["kind"] for w in whys)), examples=whys[:3])


def ranked(r: dict) -> tuple:
    """Kaveh's ranks: first don't lose (the boards held), then win, then PROGRESS (the lemma-states reached before a loop
    or the goal, a board's mean), then the claims kept."""
    n = max(r["n"], 1)
    return (1 - r["lost"] / n, r["won"] / n, r.get("states", 0) / n, r["kept"] / max(r["claims"], 1))


def score(r: dict) -> float:
    """The ranks as one number that keeps their order (a board held outweighs every win, a win all progress, progress
    every claim)."""
    h, w, p, c = ranked(r)
    return h + 1e-2 * w + 1e-4 * min(p, 99) / 100 + 1e-7 * c


def better(a: dict, b: dict) -> bool:
    """a beats b: a board more held, or (as many held) a win more. The claims kept ORDER the candidates but never promote
    one alone -- a measure we lower ourselves keeps its claims by construction (2026-10-12, the "ends" run: "their king's
    distance to our rook" and then a progress term were promoted on claims alone, holding 2 / 30 and 0 / 30 like the
    frames before them; the frame then held nothing)."""
    n = max(a["n"], 1)
    ha, wa, pa, _ = ranked(a); hb, wb, pb, _ = ranked(b)
    if abs(ha - hb) * n >= 0.5:
        return ha > hb
    if abs(wa - wb) * n >= 0.5:
        return wa > wb
    # then PROGRESS before the loop: a fifth more lemma-states reached, at least one more a board (the facts do not
    # depend on the measure, so a measure we lower ourselves cannot fake new states -- it only delays its loop)
    return pa >= 1.2 * pb and pa >= pb + 1


def main(argv=None):
    """Compare the defenders on a hole's boards: python -m catspace.incubator.analyst.discovery.defenders --hole NAME."""
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--hole", required=True)
    ap.add_argument("--boards", type=int, default=30)
    ap.add_argument("--defenders", default="self,tablebase,stockfish:8")
    ap.add_argument("--term", default=None)
    ap.add_argument("--schema", default="none")
    ap.add_argument("--player", default="frame,planner")   # (self only with the frame)
    a = ap.parse_args(argv)
    h = json.load(open(paths.derived(f"hole_{a.hole}.json")))
    boards = h["stuck"][:a.boards]
    for pl in a.player.split(","):
      for d in a.defenders.split(","):
        if pl == "planner" and d == "self":
            continue
        t0 = time.time()
        r = closed_loop(boards, h["base"], a.term, a.schema, h.get("piece") or h["material"][1], 60, d, player=pl, material=h["material"])
        if isinstance(r, tuple):
            print(f"OBS {a.hole} {a.term or ''} -- {pl} against {d}: won {r[0]}/{len(boards)}, certified {r[3]}/{r[2]} -- {time.time() - t0:.1f} s")
        else:
            print(f"OBS {a.hole} {a.term or ''} -- {pl} against {d}: held {r['n'] - r['lost']}/{r['n']}, won {r['won']}, "
                  f"lemma-states before the loop {r['states'] / max(r['n'], 1):.1f} a board, claims kept {r['kept']}/{r['claims']} "
                  f"(ranks {tuple(round(x, 3) for x in ranked(r))}) -- {time.time() - t0:.1f} s")
            print(f"OBS   ground lost, why: {r['why']}")
            for e in r["examples"]:
                print(f"OBS     {e}")


if __name__ == "__main__":
    main()
