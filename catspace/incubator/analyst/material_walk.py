"""A MATERIAL THE PLANNER HAS NEVER SEEN, WALKED AGAINST A PERFECT DEFENDER (Kaveh 2026-10-12: "let's see how it performs
against two rooks versus one rook ... I want to trace its thinking ... a case it has never seen before").

  START     random legal positions of the material, us to move, that the tablebase says we WIN
  OURS      the planner (PlanState.observe: the game context carried) -- its move when it has one; when it has NO PLAN,
            Stockfish's move is played and marked FALLBACK, so the game goes on and its thinking is traced throughout
  THEIRS    Stockfish with the Syzygy tablebases (a perfect defender)
  REFEREE   the tablebase on every move of ours: did it keep the win -- WITH the game's 50-move clock (the tablebase's
            own verdict assumes a fresh clock: distance to zeroing plus the clock must stay within 100 plies); did the
            distance to zeroing fall (or zero)
Every position's trace is saved (data/experiments/material_walk_<material>.json); --show prints one game in full.

  python -m catspace.incubator.analyst.material_walk --material KRRvKR --games 30 --show 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from catspace.io import paths

LIMIT = 100                                     # moves of ours per game


def _pieces(material: str):
    import chess
    ours, theirs = material.split("v")
    kind = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}
    return [kind[c] for c in ours[1:]], [kind[c] for c in theirs[1:]]


def _start(rng: random.Random, material: str, tb):
    """A random legal position of the material, White (us) to move, won for us by the tablebase."""
    import chess
    ours, theirs = _pieces(material)
    while True:
        sq = rng.sample(range(64), 2 + len(ours) + len(theirs))
        b = chess.Board(None)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
        for s, p in zip(sq[2:], ours + theirs):
            if p == chess.PAWN and chess.square_rank(s) in (0, 7):
                break
            b.set_piece_at(s, chess.Piece(p, chess.WHITE if len(b.pieces(p, chess.WHITE)) < ours.count(p) else chess.BLACK))
        else:
            b.turn = chess.WHITE
            if b.is_valid() and not b.is_game_over() and tb.probe_wdl(b) == 2:
                return b


def _move_of(b, dec):
    """The planner's move: the preparing move when its first slot is one, else the seed."""
    import chess
    for u in (dec[0], dec[1]):
        try:
            m = chess.Move.from_uci(u)
        except (ValueError, TypeError):
            continue
        if m in b.legal_moves:
            return m
    return None


def _game(args):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    seed, material = args
    import chess
    import chess.engine
    import chess.syzygy
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core
    tb = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    eng.configure({"SyzygyPath": str(paths.syzygy_dir())})
    b = _start(random.Random(seed), material, tb)
    start = b.fen()
    st = analyst_core.PlanState()
    rows, out, ours = [], None, 0
    while out is None:
        if b.is_checkmate():
            out = "mate" if b.turn == chess.BLACK else "lost"; break
        if b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100 or b.is_repetition(3):
            out = "draw"; break
        if ours >= LIMIT:
            out = "not finished"; break
        if b.turn == chess.WHITE:
            wdl0, dtz0 = tb.probe_wdl(b), tb.probe_dtz(b)
            tr, dec, proofs, nodes, ms, *_ = st.observe(b.fen())
            m = _move_of(b, dec) if dec else None
            source = "planner" if m else "FALLBACK (no plan)"
            if m is None:
                m = eng.play(b, chess.engine.Limit(depth=10)).move
            san = b.san(m)
            mat = _material(b)
            zeroing = b.is_zeroing(m)
            b.push(m)
            mate = b.is_checkmate()
            wdl1 = 2 if mate else -tb.probe_wdl(b)
            dtz1 = 0 if mate else -tb.probe_dtz(b)
            rows.append(dict(ply=len(rows), fen=rows and rows[-1].get("after") or start, before=None, san=san, source=source, material=mat,
                             why=dec[3] if dec else "", proofs=proofs, nodes=nodes, ms=ms, wdl=(wdl0, wdl1), dtz=(dtz0, dtz1),
                             kept=wdl1 == 2 and (mate or dtz1 + b.halfmove_clock <= 100), progress=mate or zeroing or (wdl1 == 2 and 0 < dtz1 < dtz0), trace=tr))
            ours += 1
        else:
            st.record(b.fen())
            b.push(eng.play(b, chess.engine.Limit(depth=10)).move)
            rows[-1]["reply"] = b.peek().uci()
            rows[-1]["after"] = b.fen()
    eng.quit()
    for i, r in enumerate(rows):                       # the position each move was chosen in
        r["fen"] = start if i == 0 else rows[i - 1]["after"]
        r.pop("before", None)
    return dict(seed=seed, start=start, out=out, moves=ours, rows=rows)


def _material(b) -> str:
    import chess
    side = lambda c: "K" + "".join(chess.piece_symbol(p).upper() * len(b.pieces(p, c)) for p in (5, 4, 3, 2, 1))
    return side(chess.WHITE) + "v" + side(chess.BLACK)


def walk(material: str, games: int, workers: int = 6) -> list:
    from multiprocessing import Pool
    with Pool(workers) as p:
        res = sorted(p.imap_unordered(_game, [(g, material) for g in range(games)], chunksize=1), key=lambda r: r["seed"])
    rows = [r for g in res for r in g["rows"]]
    mine = [r for r in rows if r["source"] == "planner"]
    mated = [g["moves"] for g in res if g["out"] == "mate"]
    print(f"OBS {games} {material} games from tablebase wins, the planner against Stockfish + Syzygy: {dict(Counter(g['out'] for g in res))}"
          f"{f', {sum(mated) / len(mated):.1f} moves per mate' if mated else ''}")
    print(f"OBS   our moves {len(rows)}: the planner had a plan in {len(mine)} ({100 * len(mine) / max(len(rows), 1):.0f}%), "
          f"no plan (Stockfish fallback) in {len(rows) - len(mine)}")
    for mat in sorted({r["material"] for r in rows}, key=lambda m: -sum(r["material"] == m for r in rows)):
        here = [r for r in rows if r["material"] == mat]
        print(f"OBS   with {mat} on the board ({len(here)} moves):")
        for name, rs in (("the planner", [r for r in here if r["source"] == "planner"]), ("no plan, the fallback", [r for r in here if r["source"] != "planner"])):
            if rs:
                print(f"OBS     {name} {len(rs)}: kept the win {sum(r['kept'] for r in rs)}, progress (mate, a capture or "
                      f"the distance to zeroing down) {sum(r['progress'] for r in rs)}")
        kinds = Counter(_kind(r["why"]) for r in here if r["source"] == "planner")
        if kinds:
            print("OBS     the planner's reasons: " + "; ".join(f"{k} {n}" for k, n in kinds.most_common(8)))
    lost = [(g["seed"], r) for g in res for r in g["rows"] if not r["kept"]]
    for s, r in lost[:5]:
        print(f"OBS   threw the win: game {s} {r['fen']} {r['san']} [{r['source']}] {r['why'][:120]}")
    json.dump(res, open(paths.experiment(f"material_walk_{material}.json"), "w"))
    return res


def _kind(why: str) -> str:
    """The family of a reason (its wording after the move's name)."""
    if why.startswith(("king and rook", "king and queen")):
        return "the frame (" + why.split(":")[0] + ")"
    w = why.split(" ", 1)[1] if " " in why else why
    for key in ("wins", "makes two threats", "proven mate", "mate", "stops their", "answers their", "arms", "avoids the blunder",
                "cheapest", "exploits", "passer"):
        if key in w:
            return key
    return w[:40]


def show(res: list, seed: int, lines: int = 14) -> None:
    import chess
    g = next(g for g in res if g["seed"] == seed)
    print(f"OBS game {seed}: {g['start']} -> {g['out']} in {g['moves']} moves")
    print(chess.Board(g["start"]))
    for r in g["rows"]:
        tag = "" if r["kept"] else "  <-- THREW THE WIN"
        print(f"\n{r['ply'] + 1}. {r['san']} [{r['source']}] tablebase {r['wdl'][0]:+d}->{r['wdl'][1]:+d}, dtz {r['dtz'][0]}->{r['dtz'][1]}"
              f" ({r['proofs']} proofs, {r['nodes']:,} nodes, {r['ms']:.0f} ms) {r['why'][:160]}{tag}  ... {r.get('reply', '')}")
        for l in r["trace"][:lines]:
            print(f"      {l[:200]}")
        if len(r["trace"]) > lines:
            print(f"      ... {len(r['trace']) - lines} more lines")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="KRRvKR")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--show", type=int, default=None, help="print this game (by seed) in full")
    a = ap.parse_args(argv)
    res = walk(a.material, a.games)
    if a.show is not None:
        show(res, a.show)


if __name__ == "__main__":
    main()
