"""THE LEMMA REVIEWER (Kaveh 2026-10-10: "a lemma reviewer, which states the lemma we're working with and what's missing,
the positions that it does catch and the positions that it doesn't catch -- and then we can see what's going on").

One lemma per page. The lemma's FIRINGS are collected from the planner's trace (rust/analyst_core/src/plan.rs) on census
positions: every move it refused to prove. Each firing is judged by Stockfish (depth 12) on the position after the move:
the lemma claims the move wins nothing, and it is WRONG (judge()) when the move is the labelled only-move tactic, or it
wins by the label's standard -- 1.5 pawns up by Stockfish's score, GAIN pawns of material along its line (or a mate) --
within 50 cp of the best; material read as in tactic_reach.material_gain, against BOTH the position before the
opponent's last move and the current one (the smaller: a recapture is no win, a piece they just hung is not the
attack's doing).
Two strata: every Stockfish only-move tactic position (where the misses are) and a random census sample (the rates).
Candidate facts -- Claude's first guesses at what the lemma lacks, NOT reviewed -- are counted on right and wrong
firings. The page (lemma_reviewer.html, Lichess's cburnett pieces) is filled with the data and written next to it.

    python -m catspace.incubator.analyst.lemma_reviewer --sample 3000
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import chess

from catspace.io import paths

LEMMAS = {
    "lone_attack": dict(
        name="The lone-attack lemma",
        statement="A move that attacks a piece which can step away to a safe square wins nothing, so it gets no proof.",
        unless=["the move gives check", "the move captures", "the move attacks a second piece (two new threats)",
                "the move opens a new path onto their king (a check next move)", "the attacked piece is the king"],
        source="plan.rs, 2026-10-09: the castling position went from 4 proofs to 10 on attacks the piece just steps away from",
        pattern=re.compile(r"^\s+(\S+): a lone attack on a piece that can step away"),
    ),
}
HEADER = re.compile(r"^(\w+) on ([a-h][1-8]) \(goal")
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
NEAR = 50          # a surprise must be within this many cp of Stockfish's best (bug check 2026-10-10: material won along the
                   # line flagged blunders -- Rc6 at -6.48 when bxa6 wins the queen -- as the lemma's mistakes)
PIECES = Path(os.environ["CATSPACE_PIECES"]) if os.environ.get("CATSPACE_PIECES") else None       # a directory of Lichess's cburnett SVGs (wK.svg ... bP.svg)


def _fire(args):
    """The planner on one position -> its firings of the lemma: (san, target piece, target square) and its trace."""
    fen, key = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    trace, _, _, _, _, imm = analyst_core.plan_trace(fen)
    pat = LEMMAS[key]["pattern"]
    out, tgt = [], None
    for l in trace:
        h = HEADER.match(l)
        if h:
            tgt = (h.group(1), h.group(2))
        m = pat.match(l)
        if m and tgt:
            out.append((m.group(1), tgt[0], tgt[1]))
    threat = next((int(x) for l in trace for x in re.findall(r"^their threat if we pass: worth (\d+)", l)), 0)
    stops = sorted({l.split()[0] for l in trace if l.startswith("    ") and l.rstrip().endswith("stops it")})
    return fen, out, threat, stops, (imm[1] if imm else None)


_ENG = None


def _judge(args):
    """Stockfish on the position after the skipped move: the mover's score, the line, the material it wins."""
    global _ENG
    import chess.engine
    from .tactic_reach import material_gain, DEPTH, PV_PLIES
    fen, prev, uci = args
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = chess.Board(fen); m = chess.Move.from_uci(uci)
    a = b.copy(stack=False); a.push(m)
    info = _ENG.analyse(a, chess.engine.Limit(depth=DEPTH))
    sc = info["score"].pov(b.turn)
    mate = sc.is_mate() and (sc.mate() or 0) > 0
    cp = sc.score(mate_score=100000)
    pv = [x.uci() for x in info.get("pv", [])[:PV_PLIES - 1]]
    g_now = material_gain(fen, fen, [uci] + pv)
    g_prev = material_gain(prev, fen, [uci] + pv) if prev else g_now
    return fen, uci, cp, mate, pv, min(g_now, g_prev)


def judge(c: dict) -> bool:
    """The lemma was WRONG on a firing: the refused move is the labelled only-move tactic, or it WINS by the tactic
    label's own standard (tactic_reach: the mover WIN cp up and GAIN pawns of material along the line, or a mate) and
    is within NEAR of Stockfish's best -- an only move or not. (Bug check 2026-10-10: material alone flagged a +0.70
    position mid-exchange.)"""
    from .tactic_reach import GAIN, WIN
    if c["sf_tactic"]:
        return True
    if c["best"] is None or c["best"] - c["cp"] > NEAR:
        return False
    return c["mate"] or (c["cp"] >= WIN and c["gain"] >= GAIN)


def _facts(b: chess.Board, m: chess.Move, t: int, threat: int, stops: list) -> dict:
    """Claude's candidate facts about a firing (unreviewed): what a lone attack might have that the lemma does not see."""
    us = b.turn
    a = b.copy(stack=False); a.push(m)
    tp = a.piece_type_at(t)
    safe = False
    for e in a.legal_moves:                                           # the target's escapes (they are to move)
        if e.from_square != t:
            continue
        att = [VAL[a.piece_type_at(s)] for s in a.attackers(us, e.to_square)]
        dfd = [s for s in a.attackers(not us, e.to_square) if s != t]
        if not att or (min(att) >= VAL[tp] and dfd):
            safe = True
            break
    ours_attacked = {s for s, p in a.piece_map().items() if p.color != us and s != t and a.attackers(us, s)}
    guards = any(s in ours_attacked for s in a.attacks(t))
    to = m.to_square
    mv_val = VAL[a.piece_type_at(to)]
    theirs = [VAL[a.piece_type_at(s)] for s in a.attackers(not us, to)]
    moved_safe = not theirs or (bool(a.attackers(us, to)) and min(theirs) >= mv_val)
    return {
        "the attacked piece has no safe square": not safe,
        "the attacked piece guards another of their pieces we attack": guards,
        "the move answers their threat": threat > 0 and b.san(m) in stops,
        "the moving piece lands safe": moved_safe,
    }


def collect(key: str, sample: int, seed: int, workers: int) -> dict:
    from multiprocessing import Pool
    from .tactic_census_measure import load
    from .tactic_reach import is_tactic, GAIN
    t0 = time.time()
    census = load("c300")
    at = {(r["game"], r["ply"]): r["fen"] for r in census}
    labels = {(r["game"], r["ply"]): r for r in map(json.loads, open(paths.experiment("tactic_labels_c300_d12_pv.jsonl")))}
    tactic = {}
    for r in census:
        k = (r["game"], r["ply"]); lab = labels.get(k); pf = at.get((r["game"], r["ply"] - 1))
        if lab and pf and lab.get("move") and is_tactic(lab, pf, r["fen"]):
            tactic[r["fen"]] = lab["move"]
    rng = random.Random(seed)
    pool_rows = [r for r in census if r["fen"] not in tactic]
    sampled = {r["fen"] for r in rng.sample(pool_rows, min(sample, len(pool_rows)))}
    where = {r["fen"]: (r["game"], r["ply"]) for r in census}
    fens = list(tactic) + sorted(sampled)
    print(f"TIME load: {time.time() - t0:.0f} s -- {len(tactic)} Stockfish-tactic positions + {len(sampled)} sampled", flush=True)
    t1 = time.time()
    with Pool(workers) as p:
        fired = list(p.imap_unordered(_fire, [(f, key) for f in fens], chunksize=2))
    n_fire = sum(len(o) for _, o, _, _, _ in fired)
    print(f"TIME planner on {len(fens)} positions: {time.time() - t1:.0f} s -- {n_fire} firings of the lemma", flush=True)
    jobs, meta = [], {}
    passed = []
    for fen, out, threat, stops, imm in fired:
        b = chess.Board(fen)
        g, ply = where[fen]
        # THE LEMMA LET IT THROUGH, CORRECTLY: a quiet Stockfish only-move tactic (no capture, no check -- the lemma's
        # ground) that the planner then won with
        sol = tactic.get(fen)
        if sol and imm == sol and not any(b.parse_san(x).uci() == sol for x, _, _ in out):
            m = chess.Move.from_uci(sol)
            if not b.is_capture(m) and not b.gives_check(m):
                passed.append(fen)
        for san, tname, tsq in out:
            m = b.parse_san(san)
            if (fen, m.uci()) in meta:
                meta[(fen, m.uci())]["targets"].append(f"{tname} on {tsq}")
                continue
            meta[(fen, m.uci())] = dict(targets=[f"{tname} on {tsq}"], tsq=chess.parse_square(tsq), threat=threat, stops=stops)
            jobs.append((fen, at.get((g, ply - 1)), m.uci()))
    t2 = time.time()
    with Pool(workers) as p:
        judged = list(p.imap_unordered(_judge, jobs, chunksize=4))
    print(f"TIME Stockfish on {len(jobs)} firings: {time.time() - t2:.0f} s", flush=True)
    cases = []
    for fen, uci, cp, mate, pv, gain in judged:
        md = meta[(fen, uci)]
        g, ply = where[fen]
        b = chess.Board(fen); m = chess.Move.from_uci(uci)
        lab = labels.get((g, ply), {})
        prev = at.get((g, ply - 1)); nxt = at.get((g, ply + 1))
        last = _between(prev, fen); played = _between(fen, nxt)
        line = _sans(fen, [uci] + pv)
        best_line = _sans(fen, lab.get("pv", [])[:7]) if lab.get("pv") else []
        cases.append(dict(
            fen=fen, game=g, ply=ply, stratum="tactic" if fen in tactic else "sample",
            move=uci, san=b.san(m), targets=md["targets"], target_sq=chess.square_name(md["tsq"]),
            # wrong: judge() below (bug check 2026-10-10: the judge's 8-ply line stopped short of 11 labelled wins -- Bc5
            # before a8=Q -- so the labelled only-move tactic counts on its own)
            wrong=None, cp=cp, mate=mate, gain=gain, line=line,
            best=lab.get("best"), best_move=lab.get("move"), best_san=(b.san(chess.Move.from_uci(lab["move"])) if lab.get("move") else None),
            best_line=best_line, sf_tactic=(tactic.get(fen) == uci),
            last=last, played=(chess.Board(fen).san(chess.Move.from_uci(played)) if played else None), played_uci=played,
            facts=_facts(b, m, md["tsq"], md["threat"], md["stops"])))
    for fen in passed:
        g, ply = where[fen]; lab = labels[(g, ply)]; b = chess.Board(fen); m = chess.Move.from_uci(lab["move"])
        last = _between(at.get((g, ply - 1)), fen); played = _between(fen, at.get((g, ply + 1)))
        cases.append(dict(fen=fen, game=g, ply=ply, stratum="passed", move=m.uci(), san=b.san(m), targets=[], target_sq=None,
                          wrong=False, cp=lab["best"], mate=lab.get("mate", False), gain=None, line=_sans(fen, lab["pv"][:7]),
                          best=lab["best"], best_move=m.uci(), best_san=b.san(m), best_line=_sans(fen, lab["pv"][:7]), sf_tactic=True,
                          last=last, played=(b.san(chess.Move.from_uci(played)) if played else None), played_uci=played, facts={}))
    print(f"TIME total: {time.time() - t0:.0f} s -- {len(passed)} quiet tactics the lemma let through and the planner won", flush=True)
    return dict(key=key, cases=cases, n_positions=len(fens), n_tactic=len(tactic), n_sample=len(sampled),
                tactic_missed_by_lemma=sum(1 for c in cases if c["sf_tactic"] and c["stratum"] == "tactic"))


def _between(a: str | None, b: str | None) -> str | None:
    """The move that took board a to board b (a game's consecutive census plies)."""
    if not a or not b:
        return None
    x = chess.Board(a); target = chess.Board(b)
    for m in x.legal_moves:
        x.push(m)
        if x.board_fen() == target.board_fen() and x.turn == target.turn:
            return m.uci()
        x.pop()
    return None


def _sans(fen: str, ucis: list) -> list:
    b = chess.Board(fen); out = []
    for u in ucis:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            break
        out.append(dict(san=b.san(m), uci=u)); b.push(m)
    return out


def summarize(data: dict) -> dict:
    cs = data["cases"]
    out = {"passed": dict(firings=0, wrong=0, positions=sum(c["stratum"] == "passed" for c in cs))}
    for st in ("sample", "tactic"):
        g = [c for c in cs if c["stratum"] == st]
        out[st] = dict(firings=len(g), wrong=sum(c["wrong"] for c in g), positions=len({c["fen"] for c in g}))
    facts = {}
    fired = [c for c in cs if c["stratum"] != "passed"]
    for name in (fired[0]["facts"] if fired else {}):
        r = [c for c in fired if not c["wrong"]]; w = [c for c in fired if c["wrong"]]
        facts[name] = dict(right=sum(c["facts"][name] for c in r), n_right=len(r), wrong=sum(c["facts"][name] for c in w), n_wrong=len(w))
    return dict(strata=out, facts=facts)


def page(data: dict, keep_right: int, seed: int) -> Path:
    """Fill the page template: the lemma, the summary, every wrong firing and a sample of the right ones."""
    lem = {k: v for k, v in LEMMAS[data["key"]].items() if k != "pattern"}
    cs = data["cases"]
    rng = random.Random(seed)
    wrong = sorted([c for c in cs if c["wrong"]], key=lambda c: (not c["sf_tactic"], -c["gain"]))
    # for comparison, plausible refusals: within 100 cp of the best and winning nothing (a blunder teaches nothing)
    right = [c for c in cs if not c["wrong"] and c["stratum"] != "passed" and c["best"] is not None
             and c["best"] - c["cp"] <= 100 and c["gain"] <= 0]
    right = rng.sample(right, min(keep_right, len(right)))
    passed = [c for c in cs if c["stratum"] == "passed"]
    passed = rng.sample(passed, min(keep_right, len(passed)))
    for c in wrong + right + passed:                                    # the board after each step of both lines
        for ln in (c["line"], c["best_line"]):
            b = chess.Board(c["fen"])
            for st in ln:
                b.push(chess.Move.from_uci(st["uci"])); st["fen"] = b.fen()
    pieces = {p: "data:image/svg+xml;base64," + base64.b64encode((PIECES / f"{p}.svg").read_bytes()).decode()
              for p in ("wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP")}
    payload = dict(lemma=lem, summary=summarize(data), n_positions=data["n_positions"], n_tactic=data["n_tactic"],
                   n_sample=data["n_sample"], cases=wrong + right + passed, built=time.strftime("%Y-%m-%d %H:%M"))
    tpl = (Path(__file__).with_name("lemma_reviewer.html")).read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload)).replace("/*__PIECES__*/null", json.dumps(pieces))
    out = Path(paths.experiment(f"lemma_reviewer_{data['key']}.html"))
    out.write_text(html)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemma", default="lone_attack")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep-right", type=int, default=4, help="how many of each unsurprising kind the page shows")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--page-only", action="store_true", help="rebuild the page from the saved firings")
    a = ap.parse_args(argv)
    store = Path(paths.experiment(f"lemma_reviewer_{a.lemma}.json"))
    if a.page_only:
        data = json.loads(store.read_text())
    else:
        data = collect(a.lemma, a.sample, a.seed, a.workers)
        store.write_text(json.dumps(data))
    for c in data["cases"]:
        if c["stratum"] != "passed":
            c["wrong"] = judge(c)
    s = summarize(data)
    for st, v in s["strata"].items():
        if st == "passed":
            print(f"OBS passed : {v['positions']} quiet Stockfish tactics the lemma let through and the planner won")
            continue
        print(f"OBS {st:7s}: {v['firings']} firings on {v['positions']} positions, the lemma wrong on {v['wrong']} "
              f"({100 * v['wrong'] / max(v['firings'], 1):.1f}%)")
    print(f"OBS Stockfish only-move tactics the lemma skipped: {data['tactic_missed_by_lemma']} of {data['n_tactic']} tactic positions")
    for name, v in s["facts"].items():
        print(f"OBS fact '{name}': wrong {v['wrong']}/{v['n_wrong']} ({100 * v['wrong'] / max(v['n_wrong'], 1):.0f}%), "
              f"right {v['right']}/{v['n_right']} ({100 * v['right'] / max(v['n_right'], 1):.0f}%)")
    if PIECES and PIECES.is_dir():
        print(f"page written to {page(data, a.keep_right, a.seed)}")


if __name__ == "__main__":
    main()
