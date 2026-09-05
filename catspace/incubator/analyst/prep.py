"""prep.py -- a one-page PREP SHEET on an opponent from their games: repertoire by colour (what
they actually play, with their score in each line), error profile by phase, effective rating by
phase, what punishes them (with presence baselines), what they walk into, shared blind spots at
their level, and the handful of positions that show it. The last block turns the numbers into
plain advice, each line citing the number it comes from. Same instrument as the self-review,
pointed the other way.
    python -m catspace.incubator.analyst.prep --pgn their_games.pgn --opponent NAME [--workers 8 --depth 10]"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import chess
import chess.pgn

from .horizon import winp
from . import parallel
from .verbalize import ladder_tag

ERR = ("blunder", "mistake", "inaccuracy")
FAMS = ("pin", "fork", "discovered", "skewer", "hanging", "attacked_by_lower", "overload")


def repertoire(games, opp, color: bool, plies: int = 8, top: int = 3):
    """Tree of the opponent's most played lines as `color`, with their score; plus opening names."""
    lines, names = [], Counter()
    for g in games:
        is_col = (g.headers.get("White", "").lower() == opp.lower()) == color
        if not is_col:
            continue
        r = g.headers.get("Result", "*")
        score = 0.5 if r == "1/2-1/2" else 1.0 if (r == "1-0") == color else 0.0 if r in ("1-0", "0-1") else None
        b = g.board(); sans = []
        for m in list(g.mainline_moves())[:plies]:
            sans.append(b.san(m)); b.push(m)
        lines.append((tuple(sans), score))
        names[g.headers.get("Opening", "?")] += 1
    out = [f"  as {'White' if color else 'Black'}: {len(lines)} games; openings: " + ", ".join(f"{n} ({k})" for n, k in names.most_common(4))]

    def walk(prefix, depth, indent):
        node = [(l, s) for l, s in lines if l[:len(prefix)] == prefix and len(l) > len(prefix)]
        if not node or depth == 0:
            return
        nxt = Counter(l[len(prefix)] for l, s in node)
        for mv, n in nxt.most_common(top if indent == 0 else 2):
            sub = [(l, s) for l, s in node if l[len(prefix)] == mv]
            sc = [s for l, s in sub if s is not None]
            ply = len(prefix)
            label = f"{ply // 2 + 1}.{'' if ply % 2 == 0 else '..'}{mv}"
            out.append("    " + "  " * indent + f"{label:9s} {n:3d} games  their score {sum(sc)/max(len(sc),1):.2f}")
            if n >= 3:
                walk(prefix + (mv,), depth - 1, indent + 1)
    walk((), 4, 0)
    return out


def eff_rating(rows, ratings):
    if not rows:
        return None
    ll = {r: sum(math.log(max(row["p_played"].get(r, 1e-3), 1e-3)) for row in rows) / len(rows) for r in ratings}
    return max(ll, key=ll.get)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--opponent", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--rows", default="", help="reuse a review .jsonl instead of re-analysing")
    a = ap.parse_args(argv)
    out_txt = a.out or f"tmp/prep_{a.opponent}.txt"
    games = []
    with open(a.pgn) as f:
        while (g := chess.pgn.read_game(f)):
            games.append(g)
    if a.rows:
        rows = [json.loads(l) for l in open(a.rows) if l.strip()]
        for r in rows:
            r["p_played"] = {int(k): v for k, v in r["p_played"].items()}; r["p_best"] = {int(k): v for k, v in r.get("p_best", {}).items()}
    else:
        rev = out_txt.replace(".txt", "_review.txt")
        parallel.main([a.pgn, "--user", a.opponent, "--workers", str(a.workers), "--out", rev, "--depth", str(a.depth), "--multipv", "3", "--level", "1500"])
        rows = [json.loads(l) for l in open(rev.replace(".txt", ".jsonl")) if l.strip()]
        for r in rows:
            r["p_played"] = {int(k): v for k, v in r["p_played"].items()}; r["p_best"] = {int(k): v for k, v in r.get("p_best", {}).items()}
    ratings = sorted({k for r in rows for k in r["p_played"]})
    errs = [r for r in rows if r["verdict"] in ERR]
    level = eff_rating(rows, ratings)
    S = [f"PREP SHEET: {a.opponent} -- {len(games)} games, {len(rows)} of their moves (one sample; directional)", ""]
    S.append("REPERTOIRE (what they actually play, with THEIR score in each line)")
    S += repertoire(games, a.opponent, chess.WHITE); S += repertoire(games, a.opponent, chess.BLACK); S.append("")
    S.append(f"LEVEL: plays like a {level} overall" + "".join(
        f"; {name} {eff_rating([r for r in rows if sel(r)], ratings)}" for name, sel in
        (("opening", lambda r: r["ply"] <= 20), ("middlegame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] > 12), ("endgame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] <= 12))))
    for name, sel in (("opening", lambda r: r["ply"] <= 20), ("middlegame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] > 12), ("endgame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] <= 12)):
        rs = [r for r in rows if sel(r)]; es = [r for r in rs if r["verdict"] in ERR]
        S.append(f"  errors in the {name}: {len(es)}/{len(rs)} = {len(es)/max(len(rs),1):.2f}" + (f"; blunders {sum(r['verdict']=='blunder' for r in es)}" if es else ""))
    S.append("")
    S.append("WHAT PUNISHES THEM (mechanism in the refutation of their errors vs how often it was merely present)")
    lifts = {}
    for f in FAMS:
        m = sum(f in r["mechanisms"] for r in errs); p = sum(f in r["present"] for r in rows)
        rm, rp = m / max(len(errs), 1), p / max(len(rows), 1); lifts[f] = (rm, rp, rm / rp if rp else 0)
        S.append(f"  {f:17s} {rm:.2f} vs {rp:.2f}  x{lifts[f][2]:.1f}")
    cz = Counter(r["cause"]["cause"] for r in errs if r.get("cause")); kz = Counter(r["cause"]["kind"] for r in errs if r.get("cause"))
    S.append("  cause: " + ", ".join(f"{k} {v/max(len(errs),1):.0%}" for k, v in cz.most_common()))
    S.append("  kind:  " + ", ".join(f"{k} {v/max(len(errs),1):.0%}" for k, v in kz.most_common()))
    wi = Counter(f for r in errs if r.get("cause") for f in r["cause"]["walked_into_hit"])
    S.append("  they walk into: " + ", ".join(f"{k} {v}" for k, v in wi.most_common(5))); S.append("")
    shared = [r for r in errs if r["verdict"] in ("blunder", "mistake") and r["p_played"].get(level, 0) >= 0.3]
    S.append(f"SHARED BLIND SPOTS (errors a {level} also plays >=30% of the time): {len(shared)} of {sum(r['verdict'] in ('blunder','mistake') for r in errs)} blunders+mistakes")
    def useful(r):
        d = winp(r["eval_cp"]) - winp(r["eval_cp"] - r["drop"]); tk = r.get("trap") or {}
        return d * (1.5 if tk.get("kind") == "blind_spot" or (tk.get("kind") == "horizon" and tk.get("collapse_depth", 99) <= 8) else 1.0)
    for r in sorted(shared or errs, key=useful, reverse=True)[:5]:
        S.append(f"  game {r['game']} ply {r['ply']}: {r.get('headline') or r['san']}{ladder_tag(r['p_played'], r.get('p_best', {}), level, r['san'], r.get('best'))}")
    S.append("")
    S.append("HOW TO PLAY THEM (each line derived from a number above)")
    adv = []
    counts = {f: sum(f in r["mechanisms"] for r in errs) for f in FAMS}
    strong = [f for f in FAMS if counts[f] >= 20 and lifts[f][2] >= 1.5]
    if strong:
        top = max(strong, key=lambda f: lifts[f][2])
        AIM = {"pin": "pins", "fork": "forks", "discovered": "discovered attacks", "skewer": "skewers", "hanging": "loose pieces (they leave pieces undefended)",
               "attacked_by_lower": "pawn and minor-piece attacks on their bigger pieces", "overload": "overloaded defenders"}
        adv.append(f"- Aim for {AIM[top]}: they fall to them {lifts[top][2]:.1f}x more often than the structure appears ({counts[top]} cases, rate {lifts[top][0]:.2f}).")
    if cz and cz.most_common(1)[0][0] == "ignored_threat": adv.append(f"- Leave threats standing: {cz['ignored_threat']/max(len(errs),1):.0%} of their errors ignore a threat that was already on the board.")
    if cz.get("walked_into", 0) / max(len(errs), 1) >= 0.15: adv.append(f"- They self-inflict: {cz['walked_into']/max(len(errs),1):.0%} of errors create the very weakness that gets punished -- keep pieces flexible and wait.")
    ph = {name: eff_rating([r for r in rows if sel(r)], ratings) for name, sel in (("opening", lambda r: r["ply"] <= 20), ("middlegame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] > 12), ("endgame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] <= 12))}
    if ph.get("endgame") and ph.get("opening") and ph["endgame"] < ph["opening"]: adv.append(f"- Trade down: they play like a {ph['opening']} in the opening and a {ph['endgame']} in the endgame.")
    if kz.get("compound", 0) > kz.get("simple", 0): adv.append(f"- Two-piece patterns beat plain attacks against them: {kz['compound']/max(len(errs),1):.0%} of punishments were compound.")
    S += adv or ["- (no strong signal; play normally)"]
    open(out_txt, "w").write("\n".join(S))
    print("\n".join(S))


if __name__ == "__main__":
    main()
