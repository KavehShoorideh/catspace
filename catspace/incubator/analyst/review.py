"""review.py -- review a PLAYER's games from their side of the board.
For each of the player's moves: verdict + terse explanation (analyst), the mechanism of the
punishment (miner families), and the Maia LADDER's probability of the move at each rating (which
level plays like this). For errors: the HORIZON classification (blind spot / horizon trap with
its collapse depth / computer move). Aggregates: error rates, effective rating (the ladder
level whose policy best predicts the player's moves) overall and by phase, tendency table with
presence baselines, and the errors ranked by what is immediately useful to fix.
    python -m catspace.incubator.analyst.review data/lichess/huvaxshtra_games.pgn --user huvaxshtra \
        --max-games 20 --depth 12 --out tmp/review_huvaxshtra.txt"""
from __future__ import annotations

import argparse
import json
import sys
import math
import time
from collections import Counter, defaultdict

import chess
import chess.pgn

from .engine import Engine
from .ideas import analyse_position
from .verbalize import render, cp_str, best_fact, headline, ladder_tag
from .relations import relations
from .dynamics import significant, line_effects
from .horizon import trajectories, find_traps, winp
from .maia import Maia
from .causes import classify
STAGE1_DEPTH = 6
STAGE1_FLAG_CP = 30          # a shallow drop this big (or a Lichess-eval drop) sends the move to the full analysis

FAMILY = {"pin": ("pin", "disabled_attack", "disabled_defense", "pinned_and_attacked", "hanging_through_pin"),
          "fork": ("fork",), "discovered": ("discovered_attack", "battery"), "skewer": ("skewer",),
          "hanging": ("hanging",), "attacked_by_lower": ("attacked_by_lower_value",), "overload": ("overloaded",)}
ERR = ("blunder", "mistake", "inaccuracy")


def families(names):
    return sorted(f for f, ns in FAMILY.items() if any(n in ns for n in names))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn"); ap.add_argument("--user", required=True); ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--depth", type=int, default=12); ap.add_argument("--multipv", type=int, default=3)
    ap.add_argument("--ladder", default="1100,1300,1500,1700,1900"); ap.add_argument("--out", default="tmp/review.txt")
    ap.add_argument("--horizon-depth", type=int, default=14)
    ap.add_argument("--since", default="", help="only games dated >= YYYY.MM.DD")
    ap.add_argument("--perf", default="", help="comma list of TimeControl prefixes to keep, e.g. '600,900' (rapid)")
    ap.add_argument("--ladder-only", action="store_true", help="no engine: effective rating / phase profile only")
    ap.add_argument("--flush-every", type=int, default=5, help="rewrite the report every N games")
    ap.add_argument("--no-filter", action="store_true", help="full analysis of every move (the S0 behaviour)")
    ap.add_argument("--cache", default="", help="sqlite cache path (engine lines + Maia policies); default = the shared analyst cache")
    ap.add_argument("--threads", type=int, default=2, help="Stockfish threads (use 1 under the parallel driver)")
    ap.add_argument("--level", type=int, default=1700, help="the player's level for the ladder tag (shared vs slip)")
    ap.add_argument("--why", action="store_true", help="print the full explanation under each headline")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    T = Counter()                                                  # where the time goes
    ratings = [int(x) for x in a.ladder.split(",") if x]
    maias = {r: Maia(r, cache_path=a.cache or None) for r in ratings}   # policies cached in the analyst sqlite
    rows, texts = [], []
    games = []
    with open(a.pgn) as f:
        while (g := chess.pgn.read_game(f)) and len(games) < a.max_games:
            if a.since and g.headers.get("Date", "") < a.since:
                continue
            if a.perf and not any(g.headers.get("TimeControl", "").startswith(x) for x in a.perf.split(",")):
                continue
            games.append(g)
    def flush(final=False):
        write_report(a, games, rows, texts, ratings, t0, final)
    eng = None if a.ladder_only else Engine(depth=a.depth, cache_path=a.cache or None, threads=a.threads)
    try:
        for gi, g in enumerate(games):
            if gi and gi % a.flush_every == 0:
                flush()
            me_white = g.headers.get("White", "").lower() == a.user.lower()
            us = chess.WHITE if me_white else chess.BLACK
            hdr = f"### game {gi+1}: {g.headers.get('White')} vs {g.headers.get('Black')} {g.headers.get('Result')} ({g.headers.get('Date')}, {g.headers.get('TimeControl')}) -- you are {'White' if me_white else 'Black'}"
            texts.append(hdr)
            board = g.board(); boards, sans = [board.copy()], []
            nodes = list(g.mainline())                      # node i carries the eval AFTER move i (Lichess convention)
            for i, node in enumerate(nodes):
                mv = node.move
                if board.turn == us and not board.is_game_over():
                    prev = boards[-3] if len(boards) >= 3 else None
                    _t = time.perf_counter()
                    lad = {r: maias[r].policy(board) for r in ratings}
                    p_played = {r: lad[r].get(mv.uci(), 0.0) for r in ratings}
                    T["ladder"] += time.perf_counter() - _t
                    # ---------------- STAGE 1: is this move even a candidate for error? ----------------
                    flagged, s1 = True, None
                    if not a.ladder_only and not a.no_filter:
                        ev_before = nodes[i - 1].eval() if i > 0 else None
                        ev_after = node.eval()
                        if ev_before is not None and ev_after is not None:          # Lichess evals: free stage 1
                            cb = ev_before.pov(us).score(mate_score=10000); ca = ev_after.pov(us).score(mate_score=10000)
                            s1 = ("lichess", cb, cb - ca)
                        else:
                            _t = time.perf_counter()
                            b6 = eng.analyse(board, multipv=1, depth=STAGE1_DEPTH)
                            if b6:
                                p6 = b6[0] if b6[0].move == mv.uci() else (eng.eval_move(board, mv, depth=STAGE1_DEPTH) or b6[0])
                                s1 = ("sf6", b6[0].cp, b6[0].cp - p6.cp)
                            T["stage1"] += time.perf_counter() - _t
                        if s1 is not None:
                            flagged = s1[2] >= STAGE1_FLAG_CP
                    if not flagged:
                        rows.append({"game": gi + 1, "ply": board.ply(), "fen": board.fen(), "san": board.san(mv),
                                     "verdict": "best" if s1[2] <= 10 else "ok", "drop": s1[2], "eval_cp": s1[1],
                                     "phase_nonpawn": sum(1 for pc in board.piece_map().values() if pc.piece_type not in (1, 6)) * 3,
                                     "best": None, "p_played": p_played, "p_best": {}, "mechanisms": [], "present": [],
                                     "ignored_threat": False, "threat_existed": False, "trap": None, "stage": s1[0],
                                     "date": g.headers.get("Date", ""), "tc": g.headers.get("TimeControl", "")})
                        sans.append(board.san(mv)); board.push(mv); boards.append(board.copy())
                        continue
                    if a.ladder_only:
                        rows.append({"game": gi + 1, "ply": board.ply(), "fen": board.fen(), "san": board.san(mv), "verdict": None, "drop": 0,
                                     "eval_cp": 0, "phase_nonpawn": sum(1 for _ in board.piece_map()), "best": None, "p_played": p_played,
                                     "p_best": {}, "mechanisms": [], "present": [], "ignored_threat": False, "threat_existed": False, "trap": None,
                                     "date": g.headers.get("Date", ""), "tc": g.headers.get("TimeControl", "")})
                        sans.append(board.san(mv)); board.push(mv); boards.append(board.copy())
                        continue
                    _t = time.perf_counter()
                    rec = analyse_position(board, eng, prev=prev, killed_by=sans[-1] if sans else None, played=mv, multipv=a.multipv)
                    T["full"] += time.perf_counter() - _t
                    p_best = {r: lad[r].get(rec.idea.uci, 0.0) for r in ratings} if rec.idea else {}
                    row = {"game": gi + 1, "ply": board.ply(), "fen": board.fen(), "san": board.san(mv), "verdict": rec.verdict,
                           "drop": (rec.eval_cp - rec.played.cp) if rec.played else 0, "eval_cp": rec.eval_cp,
                           "phase_nonpawn": rec.facts.phase_nonpawn, "best": rec.idea.san if rec.idea else None,
                           "p_played": p_played, "p_best": p_best, "mechanisms": [], "present": families(
                               [x.name for x in relations(board) if significant(x) and ((x.owner != us) or (x.name in ("hanging",) and x.owner == us))]),
                           "ignored_threat": False, "threat_existed": bool(rec.threats), "trap": None, "stage": "full",
                           "date": g.headers.get("Date", ""), "tc": g.headers.get("TimeControl", "")}
                    if rec.verdict in ERR:
                        r = rec.played.refutation
                        if r and r.effects:
                            e0 = r.effects[0]
                            row["mechanisms"] = families([x.name for x in e0.created] + [x.name for x in e0.uses] + [x.name for x in e0.resolved])
                            if e0.captured: row["mechanisms"].append("capture")
                            if r.mate is not None or r.san.endswith("#"): row["mechanisms"].append("mate")
                            row["ignored_threat"] = any(t.line.move == r.uci for t in rec.threats)
                        avail = None
                        if r and not row["ignored_threat"] and not board.is_check():
                            nb = board.copy(stack=False); nb.push(chess.Move.null()); rm = chess.Move.from_uci(r.uci)
                            if rm in nb.legal_moves:
                                l0 = eng.eval_move(nb, rm, depth=max(6, a.depth - 4))
                                avail = bool(l0 and (l0.cp + rec.eval_cp - 100) >= 30)
                        row["available_before"] = avail
                        row["cause"] = classify(board, mv, row["mechanisms"], row["ignored_threat"], avail)
                        if rec.played_scores and rec.idea:
                            sc = rec.played_scores; dmax = max(sc)
                            final = sc[dmax]; peak_d = max((d for d in sc if d >= 2), key=lambda d: winp(sc[d]), default=None)
                            if peak_d is not None and winp(sc[peak_d]) - winp(final) >= 10 and winp(rec.eval_cp) - winp(final) >= 5:
                                collapse = next((d for d in sorted(sc) if d > peak_d and all(winp(sc[peak_d]) - winp(sc[dd]) >= 10 for dd in sc if dd >= d)), dmax)
                                row["trap"] = {"kind": "horizon", "peak_depth": peak_d, "collapse_depth": collapse}
                            else:
                                d2 = sc.get(2, sc[min(sc)])
                                row["trap"] = {"kind": "blind_spot" if winp(d2) - winp(final) < 10 else "unclear"}
                        _t = time.perf_counter()
                        hl, computer = headline(rec, row.get("cause"), row.get("trap"), p_played, p_best, a.level)
                        row["headline"], row["computer_move"] = hl, computer
                        text = render(rec, alternatives=False)
                        T["render"] += time.perf_counter() - _t
                        lad_line = "   ladder plays your move: " + " ".join(f"M{r}:{p_played[r]*100:.0f}%" for r in ratings) + \
                                   " | best move: " + " ".join(f"M{r}:{p_best.get(r,0)*100:.0f}%" for r in ratings)
                        tk = row["trap"]
                        tk_line = ("   horizon: " + (f"trap -- looks fine to depth {tk['peak_depth']}, collapses at depth {tk['collapse_depth']}" if tk and tk["kind"] == "horizon"
                                   else "blind spot -- bad at every depth" if tk and tk["kind"] == "blind_spot" else "")) if tk else ""
                        num = board.ply() // 2 + 1
                        texts.append(f"[{num}{'.' if board.turn else '...'}] " + ("(computer move) " if computer else "") + hl + ladder_tag(p_played, p_best, a.level, board.san(mv), rec.idea.san if rec.idea else None))
                        if a.why:
                            texts.append(text); texts.append(lad_line)
                            if tk_line.strip(): texts.append(tk_line)
                        texts.append("")
                    rows.append(row)
                sans.append(board.san(mv)); board.push(mv); boards.append(board.copy())
            texts.append("")
    finally:
        if eng: eng.close()
        for m in maias.values():
            m.quit()
    flush(final=True)
    tot = time.perf_counter() - t0
    print("[timing] " + ", ".join(f"{k} {v:.0f}s ({v/tot:.0%})" for k, v in T.most_common()) + f", total {tot:.0f}s", file=sys.stderr)


def write_report(a, games, rows, texts, ratings, t0, final):
    out = [f"REVIEW of {a.user}: {len(games)} games{'' if final else ' (IN PROGRESS)'}, {len(rows)} of your moves, depth {a.depth}, {time.perf_counter()-t0:.0f}s", ""]
    if not rows:
        open(a.out, "w").write("\n".join(out)); return
    vc = Counter(r["verdict"] for r in rows); errs = [r for r in rows if r["verdict"] in ERR]
    out.append(f"verdicts: {dict(vc)} | error rate {len(errs)}/{len(rows)} = {len(errs)/max(len(rows),1):.2f}")
    def eff_rating(rs):
        if not rs: return None, {}
        ll = {r: sum(math.log(max(row['p_played'][r], 1e-3)) for row in rs) / len(rs) for r in ratings}
        return max(ll, key=ll.get), ll
    er, ll = eff_rating(rows)
    out.append("effective rating (ladder level whose policy best predicts your moves): " + f"{er}  [" + " ".join(f"M{r}:{v:.2f}" for r, v in ll.items()) + "] (mean log-prob)")
    for name, sel in (("opening (ply<=20)", lambda r: r["ply"] <= 20), ("middlegame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] > 12), ("endgame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] <= 12)):
        rs = [r for r in rows if sel(r)]; e, _ = eff_rating(rs); es = [r for r in rs if r["verdict"] in ERR]
        out.append(f"OBS {name:18s} moves {len(rs):3d}  errors {len(es):3d} ({len(es)/max(len(rs),1):.2f})  effective rating {e}")
    out.append(""); out.append(f"{'mechanism':16s} {'in refutations of your errors':>30s}   {'present in your positions':>26s}   lift")
    for f in list(FAMILY) + ["capture", "mate"]:
        m = sum(f in r["mechanisms"] for r in errs); p = sum(f in r["present"] for r in rows)
        rm, rp = m / max(len(errs), 1), p / max(len(rows), 1)
        base = f"{rp:.2f}" if f in FAMILY else "--"; lift = f"x{rm/rp:.1f}" if f in FAMILY and rp > 0 else ""
        out.append(f"OBS {f:16s} {rm:.2f} ({m}/{len(errs)})".ljust(48) + f"{base:>26s}   {lift}")
    ign = sum(r["ignored_threat"] for r in errs); had = sum(r["threat_existed"] for r in rows)
    out.append(f"OBS ignored a standing threat: {ign}/{len(errs)} errors; a real threat existed in {had}/{len(rows)} of your positions ({had/max(len(rows),1):.2f})")
    st = Counter(r.get("stage") for r in rows)
    out.append(f"OBS stage-1 source: {dict(st)} (full analysis only where flagged)")
    cz = Counter(r["cause"]["cause"] for r in errs if r.get("cause")); kz = Counter(r["cause"]["kind"] for r in errs if r.get("cause"))
    if cz:
        out.append("OBS cause of errors: " + ", ".join(f"{k} {v} ({v/max(len(errs),1):.0%})" for k, v in cz.most_common()))
        out.append("OBS kind of punishment: " + ", ".join(f"{k} {v} ({v/max(len(errs),1):.0%})" for k, v in kz.most_common()))
        wi = Counter(f for r in errs if r.get("cause") for f in r["cause"]["walked_into_hit"])
        out.append("OBS walked into: " + ", ".join(f"{k} {v}" for k, v in wi.most_common()))
    kinds = Counter((r["trap"] or {}).get("kind") for r in errs if r["verdict"] in ("blunder", "mistake"))
    out.append(f"OBS blunders+mistakes by horizon kind: {dict(kinds)}")
    coll = [r["trap"]["collapse_depth"] for r in errs if r["trap"] and r["trap"]["kind"] == "horizon"]
    if coll: out.append(f"OBS horizon traps you fell for: collapse depths {sorted(coll)} (median {sorted(coll)[len(coll)//2]})")
    # immediately useful: biggest win-% drops among errors that a level <= yours plays and that are blind spots or shallow traps
    def useful(r):
        d = winp(r["eval_cp"]) - winp(r["eval_cp"] - r["drop"]); tk = r["trap"] or {}
        shallow = tk.get("kind") == "blind_spot" or (tk.get("kind") == "horizon" and tk.get("collapse_depth", 99) <= 8)
        return d * (1.5 if shallow else 1.0)
    top = sorted((r for r in errs if not r.get("computer_move")), key=useful, reverse=True)[:8]
    out.append(""); out.append("MOST USEFUL TO FIX (biggest win-probability losses, shallow ones weighted up):")
    for r in top:
        tag = ladder_tag(r["p_played"], r.get("p_best", {}), er, r["san"], r.get("best")) if er else ""
        out.append(f"  game {r['game']} ply {r['ply']} (-{winp(r['eval_cp']) - winp(r['eval_cp']-r['drop']):.0f} win%): {r.get('headline') or r['san']}{tag}")
    cm = sum(1 for r in errs if r.get("computer_move"))
    if cm: out.append(f"  ({cm} inaccuracies are computer moves -- every level prefers your move; hidden by default)")
    out.append(""); out.append("=" * 80); out.append("")
    # by month: effective rating and error rate (are you improving?)
    months = sorted({r["date"][:7] for r in rows if r.get("date")})
    for mo in months:
        rs = [r for r in rows if r.get("date", "").startswith(mo)]; e, _ = eff_rating(rs); es = [r for r in rs if r["verdict"] in ERR]
        out.append(f"OBS month {mo}: moves {len(rs):4d}  effective rating {e}" + (f"  error rate {len(es)/max(len(rs),1):.2f}" if not a.ladder_only else ""))
    open(a.out, "w").write("\n".join(out + [""] + texts))
    open(a.out.replace(".txt", ".jsonl"), "w").write("\n".join(json.dumps(r) for r in rows))
    if final:
        print("\n".join(out))


if __name__ == "__main__":
    main()
