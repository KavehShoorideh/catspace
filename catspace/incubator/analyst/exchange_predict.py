"""PREDICT EVERY EXCHANGE IN REAL GAMES, AND BE SURPRISED (Kaveh 2026-10-08: "take a game, start from the beginning, and
at every point predict how every possible exchange will go, and keep it until something changes about that exchange ...
whether the player will take it ... if taken, the material right after the exchange -- including intermezzos -- against
the Stockfish evaluation; where it doesn't match, we're surprised, and we look for the next lemma to build").

THE REPLAY: every game of a census, every ply; side A to move. For every enemy piece A attacks, the exchange frame
(exchange_frame.py) gives v, the value of capturing there (the frame search, cheapest-first line first), and v_i = v minus
the intermezzo's price (the opponent's best paying capture elsewhere after A's capture; exchange_law.intermezzo). A frame
whose state is unchanged since an earlier ply keeps its prediction (counted: reused vs computed).

THE TAKE: the prediction is that A captures on the square of the best v_i when it is > 0. Outcomes: taken (the played
move captures there), another exchange taken, or none.

THE OUTCOME, when an exchange is taken: the played moves are followed until it resolves -- the first QUIET move (not a
capture, not a check, the mover not in check), so in-between captures and checks stay inside. The truth is Stockfish's
evaluation there, from A's side, in pawns (mates at +-20). The predictions graded against it:
  model           material now + v_i (the frame, the intermezzo priced)
  frame only      material now + v
  plain count     material now + the classical static exchange
  no exchange     material now                      (the degenerate baseline)
  Stockfish now   Stockfish's evaluation before the capture (what a full engine already knew: the ceiling, not a rival)
The errors are grouped by the exchange's variant NAMES (exchange_law.variants); the name with the largest error is where
the next lemma is needed, and the worst cases are listed.

    python -m catspace.incubator.analyst.exchange_predict --tag c300
"""
from __future__ import annotations

import argparse
import os
import statistics
from collections import Counter, defaultdict

import chess

MATE = 20.0


def _pawns(cp: int) -> float:
    return max(-MATE, min(MATE, cp / 100.0))


def _quiet(b: chess.Board, m: chess.Move) -> bool:
    return not b.is_capture(m) and not b.gives_check(m) and not b.is_check()


def _game(plies: dict) -> dict:
    """One game's predictions and outcomes."""
    from .exchange_frame import Search, frame_of, root_values
    from .exchange_law import intermezzo, see, variants
    from .tactic_coverage import game_moves
    moves = game_moves(plies)
    last = max(plies)
    seen = {}                                                       # frame state -> v (the prediction kept until it changes)
    out = dict(computed=0, reused=0, take=Counter(), graded=[])
    for t in sorted(plies):
        if t not in moves:
            continue
        r = plies[t]; b = chess.Board(r["fen"]); c = b.turn
        preds = {}
        for s, pc in b.piece_map().items():
            if pc.color == c or pc.piece_type == chess.KING or not b.attackers(c, s):
                continue
            st = frame_of(b, s)
            key = (s, st)
            if key in seen:
                v = seen[key]; out["reused"] += 1
            else:
                v = max(root_values(Search(True), *st).values()); seen[key] = v; out["computed"] += 1
            preds[s] = (v, v - intermezzo(b, s))
        m = moves[t]
        best = max(preds, key=lambda q: preds[q][1], default=None)
        want = best is not None and preds[best][1] > 0
        took = m.to_square if (b.is_capture(m) and m.to_square in preds) else None
        bucket = "predicted take" if want else ("predicted even" if best is not None and preds[best][1] == 0 else "predicted none")
        out["take"][(bucket, "took it" if took is not None and took == best else "took another" if took is not None else "took none")] += 1
        if took is None:
            continue
        u = t                                                       # follow the exchange to its first quiet move
        bb = b.copy(stack=False); events = set()
        while u in moves and u + 1 in plies and not _quiet(bb, moves[u]):
            x = moves[u]                                            # what actually happened inside the exchange
            if bb.gives_check(x) and x.to_square != took:
                events.add("a check in between")
            if bb.is_capture(x) and x.to_square != took:
                events.add("a capture elsewhere")
            if bb.piece_at(x.from_square).piece_type == chess.KING and x.to_square == took:
                events.add("the king took part")
            if x.promotion:
                events.add("a promotion")
            bb.push(x); u += 1
        end = plies[u]
        truth = _pawns(end["eval"] if end["side"] == r["side"] else -end["eval"])
        mat = r["material"]
        v, vi = preds[took]
        out["graded"].append(dict(game=r["game"], ply=t, fen=r["fen"], sq=chess.square_name(took), plies=u - t, truth=truth,
                                  mat_end=(end["material"] if end["side"] == r["side"] else -end["material"]),
                                  events=sorted(events) or ["a plain exchange"], mate=abs(truth) >= MATE or abs(_pawns(r["eval"])) >= MATE,
                                  model=mat + vi, frame=mat + v, plain=mat + see(b, took)[0], none=mat, sf_now=_pawns(r["eval"]),
                                  names=variants(b, took), elo=r["elo"]))
    return out


def run(tag: str, limit_games: int | None = None) -> None:
    from multiprocessing import Pool

    from .tactic_census_measure import load
    by = defaultdict(dict)
    for r in load(tag):
        by[r["game"]][r["ply"]] = r
    games = [by[g] for g in sorted(by)][:limit_games]
    computed = reused = 0; take = Counter(); graded = []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for o in pool.imap_unordered(_game, games, chunksize=2):
            computed += o["computed"]; reused += o["reused"]; take.update(o["take"]); graded += o["graded"]
    n_pos = sum(take.values())
    print(f"OBS exchange prediction on {len(games)} games of {tag} ({n_pos} positions): frame predictions computed {computed}, kept from an "
          f"earlier ply {reused} ({100 * reused / max(computed + reused, 1):.0f}%)")
    for want in ("predicted take", "predicted even", "predicted none"):
        row = {k[1]: v for k, v in take.items() if k[0] == want}; tot = sum(row.values())
        print(f"OBS   {want:15s} {tot:5d} positions: took it {row.get('took it', 0)} ({100 * row.get('took it', 0) / max(tot, 1):.0f}%), took another "
              f"{row.get('took another', 0)}, took none {row.get('took none', 0)} ({100 * row.get('took none', 0) / max(tot, 1):.0f}%)")
    mates = [g for g in graded if g["mate"]]; graded = [g for g in graded if not g["mate"]]
    keys = ("model", "frame", "plain", "none", "sf_now")
    err = lambda g, k, truth="truth": abs(g[k] - g[truth])
    print(f"OBS exchanges taken: {len(graded) + len(mates)}; set aside where Stockfish sees a mate (now or at the end; a mate frame, not the "
          f"exchange's) {len(mates)}; graded {len(graded)}, resolved in a median of {statistics.median([g['plies'] for g in graded]) if graded else 0} plies")
    for truth, label in (("mat_end", "the material actually left after the played exchange"), ("truth", "Stockfish at the end")):
        ks = keys if truth == "truth" else keys[:4]
        print(f"OBS   mean |prediction - {label}|: " + ", ".join(f"{k} {statistics.mean(err(g, k, truth) for g in graded):.2f}" for k in ks)
              + "; median: " + ", ".join(f"{k} {statistics.median(err(g, k, truth) for g in graded):.2f}" for k in ks))
    # OUR JUDGMENT vs WHAT HAPPENED (Kaveh 2026-10-08: predicting "not worth taking" is right when they take it and lose;
    # what is graded is our value against the actual gain -- and a capture we called bad that gained is a frame to learn)
    sign = lambda x: "+" if x > 0 else ("-" if x < 0 else "0")
    conf = Counter(); err_by = defaultdict(list)
    for g in graded:
        v = g["frame"] - g["none"]; d = g["mat_end"] - g["none"]
        conf[(sign(v), sign(d))] += 1; err_by[sign(v)].append(abs(v - d))
    print("OBS our judgment of the capture taken (rows: the frame's value; columns: the material actually gained; a player taking a capture we "
          "called losing and losing is OUR correct call, their blunder):")
    for p in "+0-":
        tot = sum(conf[(p, q)] for q in "+0-")
        print(f"OBS     frame {p}  n {tot:5d}: gained + {conf[(p, '+')]:5d}, even {conf[(p, '0')]:5d}, lost - {conf[(p, '-')]:5d}; "
              f"mean |our value - the actual gain| {statistics.mean(err_by[p]) if err_by[p] else 0:.2f}")
    agree = sum(conf[(p, p)] for p in "+0-")
    print(f"OBS   the frame's sign right on {agree}/{len(graded)} ({100 * agree / max(len(graded), 1):.0f}%); the player took a capture the frame called "
          f"losing {sum(conf[('-', q)] for q in '+0-')} times: lost as predicted {conf[('-', '-')]} (their blunder, our call), gained {conf[('-', '+')]} "
          f"(they saw something the frame did not: frames to learn)")
    learn = sorted((g for g in graded if g["frame"] - g["none"] < 0 and g["mat_end"] - g["none"] > 0), key=lambda g: -(g["mat_end"] - g["frame"]))
    from .exchange_law import HYPS, hypotheses
    why = Counter()
    for g in learn:                                         # which hypothesis of the plain count fails on them
        h = hypotheses(chess.Board(g["fen"]), chess.parse_square(g["sq"]))
        why.update(k for k in HYPS if k != "target" and not h[k])
        why["(all hold)"] += all(h[k] for k in HYPS)
    print(f"OBS   on the {len(learn)} frames to learn, the hypotheses that fail (several per frame): " + ", ".join(f"{k} {v}" for k, v in why.most_common()))
    print("OBS   frames to learn (called losing, gained), the largest gaps first:")
    for g in learn[:12]:
        print(f"OBS     {g['fen']} on {g['sq']}: frame {g['frame'] - g['none']:+.0f}, gained {g['mat_end'] - g['none']:+d}; happened: {', '.join(g['events'])}")
    by_ev = defaultdict(list)
    for g in graded:
        for nm in g["events"]:
            by_ev[nm].append(g)
    tot_err = sum(err(g, "model", "mat_end") for g in graded) or 1
    print("OBS   by what happened inside the exchange (n; mean |model - material after|, |no exchange - material after|; share of the model's error):")
    for nm, gs in sorted(by_ev.items(), key=lambda kv: -sum(err(g, "model", "mat_end") for g in kv[1])):
        print(f"OBS     {nm:22s} n {len(gs):5d}: {statistics.mean(err(g, 'model', 'mat_end') for g in gs):5.2f} / {statistics.mean(err(g, 'none', 'mat_end') for g in gs):5.2f}; "
              f"share {100 * sum(err(g, 'model', 'mat_end') for g in gs) / tot_err:4.1f}%")
    worst = sorted(graded, key=lambda g: -err(g, "model", "mat_end"))[:12]
    print("OBS   the biggest surprises against the material after the played exchange:")
    for g in worst:
        print(f"OBS     {g['fen']} on {g['sq']}: model {g['model']:+.0f}, material after {g['mat_end']:+d}, Stockfish at the end {g['truth']:+.1f}; "
              f"happened: {', '.join(g['events'])}; names: {', '.join(g['names'])}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.tag, a.games)


if __name__ == "__main__":
    main()
