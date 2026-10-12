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


def _play_one(args):
    """One board: (Some(moves to the goal) or None, our moves played, how many carried a certificate)."""
    import chess
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    fen, base, cand, schema, piece, limit, defender, player, material = args
    b = chess.Board(fen)
    us = b.turn
    theirs0, lead0 = _material(b, not us), _material(b, us) - _material(b, not us)
    history, played, cert = [], 0, 0
    st = None
    if player == "planner":                       # the whole player, the candidate as its material's frame
        A.set_generic_frame(True)
        A.set_material_frame(material, compose(base, cand, schema))
        st = A.PlanState()
    for n in range(1, limit + 1):
        history.append(b.fen())
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
                return None, played, cert
            pick = (mv.uci(), any(l.startswith("frame:") and "certified" in l and "NOT certified" not in l for l in tr) or not any(l.startswith("frame:") for l in tr))
        else:
            pick = A.frame_choose(b.fen(), base, cand, schema, piece, history)
        if pick is None:
            return None, played, cert
        played += 1; cert += int(pick[1])
        b.push(chess.Move.from_uci(pick[0]))
        if b.is_checkmate():
            return n, played, cert
        if piece == "*":
            if _material(b, us) - _material(b, not us) < lead0:
                return None, played, cert                      # first don't lose
            if _material(b, not us) < theirs0:
                return n, played, cert                         # the goal: the next frame's material
        if not any(b.legal_moves) or b.is_repetition(3):
            return None, played, cert
        history.append(b.fen())
        if st is not None:
            st.record(b.fen())
        b.push(reply(b, defender))
        if piece != "*" and not any(p.color == us and p.piece_type != chess.KING for p in b.piece_map().values()):
            return None, played, cert                          # our piece taken
        if b.is_repetition(3) or b.halfmove_clock >= 100:
            return None, played, cert
    return None, played, cert


def closed_loop(boards: list, base: list, cand: str | None, schema: str, piece: str, limit: int, defender: str,
                workers: int = 6, player: str = "frame", material: str = "") -> tuple:
    """-> (games won, total moves of the won games, our moves played, our moves certified), as Rust verify_play."""
    import analyst_core as A
    if parse(defender)[0] == "self":
        if player != "frame":
            raise ValueError("the 'self' defender (the measure's adversary) lives in Rust and plays the frame alone")
        return tuple(A.verify_play(boards, base, cand, schema, piece, limit))
    from multiprocessing import Pool
    with Pool(workers, initializer=_init, initargs=(defender,)) as p:
        res = p.map(_play_one, [(f, base, cand, schema, piece, limit, defender, player, material) for f in boards], chunksize=2)
    won = [r[0] for r in res if r[0] is not None]
    return len(won), sum(won), sum(r[1] for r in res), sum(r[2] for r in res)


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
        won, moves, played, cert = closed_loop(boards, h["base"], a.term, a.schema, h.get("piece") or h["material"][1], 60, d,
                                               player=pl, material=h["material"])
        print(f"OBS {a.hole} base {h['base']} {a.term or ''} -- {pl} against {d}: won {won}/{len(boards)}, certified {cert}/{played} "
              f"(score {cert / max(played, 1) * won / len(boards):.3f}) -- {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
