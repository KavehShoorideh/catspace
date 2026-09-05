"""cards.py -- LINE CARDS: walk a line from a player's own games node by node. At every node:
the side to move's NATURAL moves (the Maia ladder's probabilities per level), each move's loss in
win-probability and horizon kind (blind spot / horizon trap with collapse depth), the ONLY-MOVE
flag (the best move is rare and every other natural move loses), the TRAP YIELD per level
(sum over natural moves of P(move) x loss), the analyst's idea for the best move, and the
historical record from the games (who went where from here, and the player's score).
A trap = a node with high yield; it dies at the level where the setup stops being played OR the
defence starts being found -- the card shows which (Kaveh's Jobava question, 2026-09-04).
    python -m catspace.incubator.analyst.cards --pgn games.pgn --player NAME --color white \
        --line "d4 d5 Nc3" --plies 6 --min-games 5 --depth 14 --out tmp/cards_x.txt"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import chess
import chess.pgn

from .engine import Engine
from .ideas import analyse_position
from .verbalize import cp_str, terse_line
from .horizon import winp
from .maia import Maia

NATURAL = 0.10


def build_tree(games, player, color, prefix, plies):
    """node key = tuple of SAN; value = (count, score_for_player, Counter(next SAN))"""
    tree = {}
    for g in games:
        if (g.headers.get("White", "").lower() == player.lower()) != (color == chess.WHITE):
            continue
        r = g.headers.get("Result", "*")
        sc = 0.5 if r == "1/2-1/2" else 1.0 if (r == "1-0") == (color == chess.WHITE) else 0.0 if r in ("1-0", "0-1") else None
        b = g.board(); sans = []
        for m in list(g.mainline_moves())[:len(prefix) + plies]:
            sans.append(b.san(m)); b.push(m)
        if tuple(sans[:len(prefix)]) != tuple(prefix):
            continue
        for k in range(len(prefix), len(sans) + 1):
            key = tuple(sans[:k])
            node = tree.setdefault(key, [0, 0.0, 0, Counter()])
            node[0] += 1
            if sc is not None:
                node[1] += sc; node[2] += 1
            if k < len(sans):
                node[3][sans[k]] += 1
    return tree


def board_of(sans):
    b = chess.Board()
    for s in sans:
        b.push_san(s)
    return b


def node_card(sans, tree, eng, maias, ratings, depth, player_color, child_yield=None, labels=("you", "them")):
    """child_yield: {san: {rating: yield}} -- the yield the side to move faces after each of its moves and the
    player's usual reply (propagated from deeper cards), so a setup move shows its delayed cost."""
    child_yield = child_yield or {}
    b = board_of(sans); to_move = b.turn
    cnt, sc, n_sc, nxt = tree[tuple(sans)]
    who = labels[0] if to_move == player_color else labels[1]
    num = b.ply() // 2 + 1
    head = f"[{num}{'.' if to_move else '...'}] {who} to move -- {cnt} games reached this; " + (f"{labels[0]}'s score from here {sc/max(n_sc,1):.2f}" if n_sc else "")
    lines = [head]
    lad = {r: maias[r].policy(b) for r in ratings}
    best = eng.best(b)
    if best is None:
        return lines, {r: 0.0 for r in ratings}
    ev_best = best.cp
    cands = {best.move}
    for r in ratings:
        for u, p in lad[r].items():
            if p >= NATURAL:
                cands.add(u)
    for s, k in nxt.most_common(4):                      # what actually got played from here
        cands.add(b.parse_san(s).uci())
    rows = []
    for u in cands:
        m = chess.Move.from_uci(u)
        line, scores = eng.trajectory(b, m, depth=depth)
        cp = line.cp if line else (scores[max(scores)] if scores else ev_best)
        loss = winp(ev_best) - winp(cp)
        kind = ""
        if loss >= 10 and scores:
            peak_d = max((d for d in scores if d >= 2), key=lambda d: winp(scores[d]), default=None)
            if peak_d is not None and winp(scores[peak_d]) - winp(cp) >= 10:
                collapse = next((d for d in sorted(scores) if d > peak_d and all(winp(scores[peak_d]) - winp(scores[dd]) >= 10 for dd in scores if dd >= d)), max(scores))
                kind = f"horizon trap, collapses at depth {collapse}"
            else:
                kind = "blind spot"
        rows.append((b.san(m), cp, loss, {r: lad[r].get(u, 0.0) for r in ratings}, nxt.get(b.san(m), 0), kind))
    rows.sort(key=lambda x: -max(x[3].values()) if x[3] else 0)
    lines.append("    move     eval    loss   " + " ".join(f"M{r:<5d}" for r in ratings) + "  played-here  kind")
    for san, cp, loss, ps, played, kind in rows:
        lines.append(f"    {san:8s} {cp_str(cp, to_move):>6s}  {loss:5.0f}   " + " ".join(f"{ps[r]*100:4.0f}% " for r in ratings) + f"  {played:5d}        {kind}")
    # direct yield + delayed yield through the child nodes (weighted by how natural each move is)
    yields = {r: sum(ps[r] * (max(loss, 0) + child_yield.get(san, {}).get(r, 0.0)) for san, _, loss, ps, _, _ in rows) for r in ratings}
    good = [(san, ps) for san, cp, loss, ps, _, _ in rows if loss < 5]
    only = len(good) == 1 and any(loss >= 10 and max(ps.values()) >= NATURAL for _, _, loss, ps, _, _ in rows)
    if to_move != player_color:
        lines.append(f"    trap yield for {labels[1]} (expected win% loss incl. what follows): " + " ".join(f"M{r}:{yields[r]:.1f}" for r in ratings)
                     + (f"   ONLY MOVE {good[0][0]}: found by " + " ".join(f"M{r}:{good[0][1][r]*100:.0f}%" for r in ratings) if only else ""))
    rec = analyse_position(b, eng, multipv=2)
    if rec.idea:
        tl = terse_line(rec.idea, 2)
        lines.append(f"    idea for {who}: {rec.idea.san} -- {rec.idea.line_san}" + (f". {tl}" if tl else ""))
    for t in rec.threats[:1]:
        lines.append(f"    standing threat: {t.san_line(b, 4)}" + (f" (already: {t.structure.name.replace('_',' ')})" if t.structure else ""))
    return lines, yields


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--player", required=True); ap.add_argument("--color", default="white")
    ap.add_argument("--line", default=""); ap.add_argument("--plies", type=int, default=6); ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--depth", type=int, default=14); ap.add_argument("--ladder", default="1100,1300,1500,1700,1900"); ap.add_argument("--out", default="")
    ap.add_argument("--prep", action="store_true", help="labels for prepping AGAINST --player: the player by name, the reader as 'you'")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    color = chess.WHITE if a.color == "white" else chess.BLACK
    prefix = a.line.split()
    games = []
    with open(a.pgn) as f:
        while (g := chess.pgn.read_game(f)):
            games.append(g)
    tree = build_tree(games, a.player, color, prefix, a.plies)
    ratings = [int(x) for x in a.ladder.split(",")]
    maias = {r: Maia(r) for r in ratings}
    out = [f"LINE CARDS: {a.player} as {a.color}, line '{a.line or '(root)'}', {tree[tuple(prefix)][0] if tuple(prefix) in tree else 0} games, nodes with >= {a.min_games} games", ""]
    with Engine(depth=a.depth) as eng:
        keys = sorted((k for k in tree if tree[k][0] >= a.min_games and len(k) >= len(prefix)), key=lambda k: (len(k), -tree[k][0]))
        cards, yields = {}, {}
        for k in reversed(keys):                            # deepest first so yields propagate upward
            b = board_of(list(k))
            cy = {}
            if b.turn != color:                             # their node: what follows each of their moves, via our usual reply
                for san in tree[k][3]:
                    k1 = k + (san,)
                    if k1 in tree and tree[k1][3]:
                        reply = tree[k1][3].most_common(1)[0][0]
                        k2 = k1 + (reply,)
                        if k2 in yields:
                            cy[san] = yields[k2]
            cards[k], yields[k] = node_card(list(k), tree, eng, maias, ratings, a.depth, color, cy, (a.player, "you") if a.prep else ("you", "them"))
        for k in keys:
            out.append("line: " + " ".join(f"{i//2+1}.{s}" if i % 2 == 0 else s for i, s in enumerate(k)))
            out += cards[k]
            out.append("")
    for m in maias.values():
        m.quit()
    out.append(f"[cards] {len(keys)} nodes in {time.perf_counter()-t0:.0f}s")
    text = "\n".join(out)
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
