"""PLAN 73.5b (1) THE SAMPLE: the won positions of a material class the amended generator's criterion is checked on, and THE FLOOR the engine's own mate search
reads on them. The table is the ORACLE here and only here (it judges "won" and writes the 73.5 corpus); it answers nothing inside the agenda.

  python -m catspace.incubator.analyst.discovery.technique_sample KRvK [--corpus <file>] [--uniform 3000] [--seed 2]

THE SAMPLE = the corpus positions of 73.5 (2) (the winning side's turns of the table's games, technique_corpus.py; one row per position) AND UNIFORM won positions of
the class: random legal placements (technique_corpus.random_start's draw: the units on distinct random squares, the winning side -- White -- to move, not in check,
not over), kept when the table judges them WON (WDL +2, DTZ >= 1). The corpus holds the hard half of the class as the table plays it; the uniform part holds what
the corpus never visits (our king far away, a unit attacked, a mate in one ...).

UNIFORM = 3000, by the RULE OF THREE: a measure that fails on none of N independent uniform draws has a per-position failure rate below 3 / N with 95% confidence --
N = 3000 bounds it below 1 in 1000, so a play-out over the fifty-move rule's horizon (50 moves of ours) meets a failing position with probability at most about 5%.
The seed of the uniform draw (2) is apart from the corpus games' (1).

THE FLOOR (73.5b): a position is AT THE FLOOR n when the engine's mate search finds a mate within n moves -- n = 2: the MATE WORDS (technique_measures.mate_words:
mate_in_one, the handle's reading; mate_in_two, the binder's schema); n = 3: mating.rs's NET (`analyst_core.net_mate_of`, the binding added for 73.5b to the existing
`mating::net_mate`: a forced mate in at most n moves against every reply, every move of ours tried, never through a stalemate or a claimable draw) -- no new mate
solver. The words and the net are compared at n = 2 on every sample position (printed).
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import Counter

import chess

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery import technique_measures as TM
from catspace.io import paths

UNIFORM = 3000              # the rule of three (see the docstring)
SAMPLE_SEED = 2             # the uniform draw's own seed (the corpus games are seed 1)
MAX_PLIES = 6               # the longest strategy the amended search reads (d = 3, the fuse): the sample's fifty-move clocks must stay below 100 - MAX_PLIES


def net(board: chess.Board, n: int) -> "tuple | None":
    """mating.rs's net on `board` (the side to move mates): -> (the first move's uci, the moves to mate) or None."""
    import analyst_core as A
    return A.net_mate_of(board.fen(), n)


def floor_word(board: chess.Board, n: int) -> "str | None":
    """THE FLOOR on `board` (the winning side to move): the word of the engine's mate search that finds a mate within `n` moves, or None (see the docstring)."""
    if n <= 2:
        return TM.mate_words(board)
    got = net(board, n)
    return f"net_mate_in_{got[1]}" if got is not None else None


def floor_move(board: chess.Board, n: int) -> "tuple | None":
    """The first move of the mate the floor's search finds -> (uci, the word), or None: n = 2 -- a mating move (mate_in_one), else the binder's mate_in_two binding;
    n = 3 -- the net's first move."""
    w = floor_word(board, n)
    if w is None:
        return None
    if w == "mate_in_one":
        for m in board.legal_moves:
            board.push(m)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return m.uci(), w
        return None
    if w == "mate_in_two":
        from catspace.incubator.analyst.discovery import selfmine as SM
        got = SM.bind_mate2(None, "w" if board.turn else "b", pos=G.pos_of(board))
        if not got:
            return None
        (x, y), = got[0]["pairs"]                                  # (the binding's squares, by name)
        sq = lambda s: chess.parse_square(s) if isinstance(s, str) else s
        return chess.Move(sq(x), sq(y)).uci(), w
    got = net(board, n)
    return (got[0], w) if got is not None else None


def uniform(sig: str, n: int, seed: int) -> tuple:
    """`n` uniform won positions of the class -> (their FENs, Counter(valid = the placements drawn that are legal with the winning side to move, not in check and
    not over; won = of which the table judges won)). The draw is technique_corpus.random_start's, the table's verdict counted."""
    ours, theirs = TC.parse_signature(sig)
    rng = random.Random(seed)
    tb = TC.tablebase()
    out, drawn = [], Counter()
    k = len(ours) + len(theirs)
    while len(out) < n:
        sq = rng.sample(range(64), k)
        b = chess.Board(None)
        for s, ch in zip(sq[:len(ours)], ours):
            b.set_piece_at(s, chess.Piece(TC.UNITS[ch], chess.WHITE))
        for s, ch in zip(sq[len(ours):], theirs):
            b.set_piece_at(s, chess.Piece(TC.UNITS[ch], chess.BLACK))
        b.turn = chess.WHITE
        if not b.is_valid() or b.is_check() or b.is_game_over():
            continue
        drawn["valid"] += 1
        if tb.probe_wdl(b) == 2 and tb.probe_dtz(b) >= 1:
            drawn["won"] += 1
            out.append(b.fen())
    return out, drawn


def _key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def sample(sig: str, corpus: str, n_uniform: int = UNIFORM, seed: int = SAMPLE_SEED, log=print, floors: bool = True) -> list:
    """THE SAMPLE -> [dict(fen, source, dtz, floor2, floor3)]: the corpus positions (the winning side's turns of every game, one row per position) then the uniform
    won positions not already in it; `floor2` / `floor3`: the word of the floor's mate search (None: not at that floor). Prints `OBS sample <sig>: ...` and, with
    `floors`, `OBS sample <sig> floor: ...`."""
    t0 = time.monotonic()
    TC.parse_signature(sig)
    rows, seen, turns = [], set(), 0
    for game in TC.read_corpus(corpus):
        for r in game:
            if r["move"] is not None and r["side"] == r["winner"]:
                turns += 1
                if _key(r["fen"]) not in seen:
                    seen.add(_key(r["fen"]))
                    rows.append(dict(fen=r["fen"], source="corpus"))
    nc = len(rows)
    fens, drawn = uniform(sig, n_uniform, seed)
    dup = 0
    for f in fens:
        if _key(f) in seen:
            dup += 1
            continue
        seen.add(_key(f))
        rows.append(dict(fen=f, source="uniform"))
    tb = TC.tablebase()
    for r in rows:
        r["dtz"] = tb.probe_dtz(chess.Board(r["fen"]))
    clocks = [chess.Board(r["fen"]).halfmove_clock for r in rows]
    assert max(clocks) + MAX_PLIES < 100, "a sample clock near the fifty-move rule: the miner's search would need it"
    log(f"OBS sample {sig}: positions {len(rows)} -- corpus {nc} (the distinct positions of {turns} winning-side turns of {os.path.relpath(corpus, str(paths.REPO_ROOT))}), uniform {len(rows) - nc} "
        f"({len(fens)} drawn, seed {seed}, {dup} already in the corpus; of {drawn['valid']} valid placements drawn the table judged {drawn['won']} won); "
        f"DTZ {min(r['dtz'] for r in rows)}..{max(r['dtz'] for r in rows)}; largest fifty-move clock {max(clocks)} (below 100 - {MAX_PLIES}: no claimable draw within d = 3)")
    if floors:
        read_floors(rows, log, sig)
    return rows


def read_floors(rows: list, log=print, sig: str = "") -> None:
    """The floor words of every row (`floor2`, `floor3`), in place; prints `OBS sample <sig> floor: ...`."""
    t0 = time.monotonic()
    words, agree = Counter(), 0
    for j, r in enumerate(rows):
        b = chess.Board(r["fen"])
        r["floor2"], w3 = floor_word(b, 2), floor_word(b, 3)
        r["floor3"] = r["floor2"] or w3                       # (the floor at n = 3 holds the floor at n = 2)
        words.update(w for w in (r["floor2"], w3) if w)
        agree += (r["floor2"] is not None) == (net(b, 2) is not None)
        if j % 500 == 0:
            G.clear_caches()
    log(f"OBS sample {sig} floor: at n = 2 (the mate words) {sum(r['floor2'] is not None for r in rows)}, at n = 3 (the net) {sum(r['floor3'] is not None for r in rows)} "
        f"of {len(rows)}; " + ", ".join(f"{k} {v}" for k, v in sorted(words.items())) + f"; the mate words and the net agree at n = 2 on {agree} of {len(rows)}; "
        f"{time.monotonic() - t0:.1f} s")


def whole_class(sig: str, log=print) -> list:
    """THE CLASS AS THE SAMPLE (73.5b, after the first K+R play-out refuted the sampled entry -- the journal of 2026-10-26): EVERY position of the class, the
    winning side to move (every legal placement, not in check, not over, judged won by the table), with its floors. For the three-man classes the miner's own
    d = 3 graph from the sample already holds all but a few hundred of these positions (printed in the journal), so checking every one costs only its floor;
    a larger class is sampled."""
    import itertools
    t0 = time.monotonic()
    ours, theirs = TC.parse_signature(sig)
    tb = TC.tablebase()
    rows, valid = [], 0
    for sq in itertools.permutations(range(64), len(ours) + len(theirs)):
        b = chess.Board(None)
        for s, ch in zip(sq[:len(ours)], ours):
            b.set_piece_at(s, chess.Piece(TC.UNITS[ch], chess.WHITE))
        for s, ch in zip(sq[len(ours):], theirs):
            b.set_piece_at(s, chess.Piece(TC.UNITS[ch], chess.BLACK))
        b.turn = chess.WHITE
        if not b.is_valid() or b.is_check() or b.is_game_over():
            continue
        valid += 1
        if tb.probe_wdl(b) == 2:
            rows.append(dict(fen=b.fen(), source="class", dtz=tb.probe_dtz(b)))
    log(f"OBS sample {sig} THE WHOLE CLASS: positions {len(rows)} (every legal placement with the winning side to move, not in check, not over: {valid}; won by the table: "
        f"{len(rows)}); DTZ {min(r['dtz'] for r in rows)}..{max(r['dtz'] for r in rows)}; {time.monotonic() - t0:.1f} s")
    read_floors(rows, log, sig)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("signature")
    ap.add_argument("--corpus", default=None, help="the 73.5 corpus (default: artifacts/experiments/technique_corpus/<sig>_seed1.jsonl)")
    ap.add_argument("--uniform", type=int, default=UNIFORM)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    a = ap.parse_args(argv)
    sample(a.signature, a.corpus or paths.experiment("technique_corpus", f"{a.signature}_seed1.jsonl"), a.uniform, a.seed)


if __name__ == "__main__":
    main()
