"""game_report.py -- the full review of ONE game, both sides, as one markdown file (Kaveh 2026-09-05).
    python -m catspace.incubator.analyst.game_report --pgn data/lichess/X.pgn --user huvaxshtra [--depth 18 --level 1700]
Runs the per-player review for both players (headlines, ladder, horizon) and the per-position analysis
(what every move does), then merges: every move gets its eval and one line; errors get the review's headline,
the standing threat the engine saw before the move, and who plays it."""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys

import chess
import chess.pgn

from .engine import Engine
from .ideas import analyse_game
from .verbalize import cp_str, best_fact, delta_str, motif_phrase
from .structure import top_changes

FACT_KEYS = ("takes", "hanging", "forks", "pins", "skewers", "checkmate", "discovering", "attacks", "with check",
             "overloaded", "two jobs", "lured", "deflect", "guard", "cut")
ERR = ("inaccuracy", "mistake", "blunder")


def _review(pgn: str, user: str, depth: int, level: int, out: str, threads: int) -> None:
    cmd = [sys.executable, "-m", "catspace.incubator.analyst.review", pgn, "--user", user, "--max-games", "1",
           "--depth", str(depth), "--multipv", "3", "--level", str(level), "--why", "--no-filter", "--threads", str(threads), "--out", out]
    subprocess.run(cmd, check=True, capture_output=True)


def _parse_review(path: str) -> tuple[dict, dict, dict, str]:
    heads, ladder, horizon, cur = {}, {}, {}, None
    text = open(path).read()
    for line in text.splitlines():
        m = re.match(r"^\[(\d+\.(?:\.\.)?)\] (?:White|Black) to move", line)
        if m:
            cur = m.group(1); continue
        m = re.match(r"^\[(\d+\.(?:\.\.)?)\] (.*)$", line)
        if m:
            heads[m.group(1)] = m.group(2); continue
        m = re.match(r"^\s+ladder plays your move: (.*) \| best move: (.*)$", line)
        if m and cur:
            ladder[cur] = (m.group(1), m.group(2)); continue
        m = re.match(r"^\s+horizon: (.*)$", line)
        if m and cur:
            horizon[cur] = m.group(1)
    return heads, ladder, horizon, text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--user", required=True)
    ap.add_argument("--depth", type=int, default=18); ap.add_argument("--level", type=int, default=1700)
    ap.add_argument("--opp-level", type=int, default=1300); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    pgn_text = open(a.pgn).read()
    game = chess.pgn.read_game(io.StringIO(pgn_text)); h = game.headers
    me_white = h["White"] == a.user
    opp = h["Black"] if me_white else h["White"]
    gid = h.get("GameId", "game")
    _review(a.pgn, a.user, a.depth, a.level, f"tmp/review_{gid}_{a.user}.txt", a.threads)
    _review(a.pgn, opp, a.depth, a.opp_level, f"tmp/review_{gid}_{opp}.txt", a.threads)
    mine = _parse_review(f"tmp/review_{gid}_{a.user}.txt"); theirs = _parse_review(f"tmp/review_{gid}_{opp}.txt")
    heads = {**theirs[0], **mine[0]}; ladder = {**theirs[1], **mine[1]}; horizon = {**theirs[2], **mine[2]}
    with Engine(depth=a.depth) as eng:
        recs, _ = analyse_game(pgn_text, eng, multipv=3)

    out = [f"# {h['White']} ({h.get('WhiteElo','?')}) vs {h['Black']} ({h.get('BlackElo','?')}) — {h.get('Event','').replace('rated ','')} {h.get('TimeControl','')}, {h.get('Date','')} — {h.get('Result','')}",
           f"{h.get('Opening','')} (ECO {h.get('ECO','')}). Stockfish depth {a.depth}, 3 lines per position; Maia 1100–1900 for 'who plays this'. "
           f"Evals are from White's side; each error leads with its cost against the best move (pawns, win-probability points).\n"]
    v = re.search(r"verdicts: (\{[^}]*\}) \| error rate (\S+)", mine[3]); eff = re.search(r"effective rating[^:]*: (\d+)", mine[3])
    causes = re.search(r"OBS cause of errors: (.*)", mine[3]); ov = re.search(r"verdicts: (\{[^}]*\}) \| error rate (\S+)", theirs[3])
    side = "White" if me_white else "Black"
    out.append(f"## Summary ({a.user}, {side})\n")
    out.append(f"- Your moves: {v.group(1).strip('{}').replace(chr(39), '')} — error rate {v.group(2)}.")
    out.append(f"- Effective rating this game (the Maia level whose policy best predicts your moves): {eff.group(1)}.")
    if causes: out.append(f"- Why the errors: {causes.group(1)}.")
    out.append(f"- Opponent {opp}: {ov.group(1).strip('{}').replace(chr(39), '')} — error rate {ov.group(2)}.\n")
    out.append("## Move by move\n")
    out.append("Format: move (eval after it, White's side) — verdict: the tool's headline. For mistakes and blunders, two extra lines: "
               "the standing threat the engine saw before the move (what the opponent does if you pass), and who plays it "
               "(Maia's probability of the move played and of the best move at 1100–1900, plus whether the move looks fine at shallow depth and collapses later).\n")
    board = game.board()
    for i, (rec, mv) in enumerate(zip(recs, game.mainline_moves())):
        label = f"{i//2+1}." if i % 2 == 0 else f"{i//2+1}..."
        san = board.san(mv); pl = rec.played
        ev_after = cp_str(pl.cp, board.turn, pl.mate) if pl else "?"
        line = f"**{label} {san}** ({ev_after})"
        if rec.verdict in ERR and label in heads:
            txt = heads[label]
            txt = txt[len(san) + 1:] if txt.startswith(san + " ") else txt
            line += f" — {rec.verdict.upper()}: {txt}"
        else:
            e0 = pl.effects[0] if pl and pl.effects else None
            fact = best_fact(e0, strict=True) if e0 else None
            fact = fact if fact and any(k in fact for k in FACT_KEYS) else None
            mp = motif_phrase(board, pl.pv) if pl and pl.pv else ""
            if not (mp or fact) and e0 is not None:                       # quiet move: its structural change, if any
                sc = top_changes(e0.structure_created, e0.structure_ended, k=1)
                fact = sc[0] if sc else None
            if rec.verdict == "best":
                line += " — best" + (f". {mp or fact}" if (mp or fact) else "")
            else:
                line += (f" — ok ({delta_str(rec)}); {rec.idea.san} was a touch better" if rec.idea and rec.idea.san != san else " — ok")
                if fact: line += f". {san}: {fact}"
        if rec.verdict in ERR and "ignored a standing threat" not in line:
            th = [t for t in rec.threats if t.real]
            if th: line += f"  \n  · standing threat before the move: {th[0].san_line(board, 4)}"
        if rec.verdict in ("mistake", "blunder") and label in ladder:
            line += f"  \n  · who plays it: {ladder[label][0]} · best move: {ladder[label][1]}" + (f" · {horizon[label]}" if label in horizon else "")
        out.append(line)
        board.push(mv)
    path = a.out or f"tmp/game_{gid}_full.md"
    open(path, "w").write("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
