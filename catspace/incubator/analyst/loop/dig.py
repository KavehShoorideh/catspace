"""DIG: what a round's ORDER use did, puzzle by puzzle -- the gate's look under the ledger (Kaveh 2026-10-05: "human in
the loop testing that decides whether the loop is going off the rails").

For one held-out set, the round's arm is paired with the previous kept round's arm (the player before the round's
ideas; the ledger's 'previous'). A puzzle went UP or DOWN by its expected score (measure.expected_score). At the first
step where the two arms' top moves differ, the mined ideas' verdicts are read again with the code the player ran
(wiring.MinedOrder over the round's active.json) and the puzzle is classed:
- DOWN (read at the first step that demoted the solution), a rule calls the solution a loss: some verdict on it is 'loses';
- DOWN, unanimous 'keeps': every verdict on the solution and on the moves kept is 'keeps', and the composed preference
  (lemma_apply.compose: the mean, over the rules that speak, of each rule's outcome when it holds) ranked the kept higher;
- DOWN, unspoken: no mined idea speaks about the solution (it scores 0);
- UP (read at the first step whose tie shrank), the same classes read on the moves the tie-break demoted.
For DOWN by a loss, the solution's piece and its landing square (attacked or not after the move, defended or not).
    python -m catspace.incubator.analyst.loop.dig --round 1 --set pawnendings [--show 3]
Read-only; prints OBS lines."""
from __future__ import annotations

import argparse
from collections import Counter

import chess

from .round import previous_measure
from .stages.measure import expected_score, load_dump
from .store import read_json, record_dir, round_dir


def _split(prev: dict, now: dict, kind: str):
    """The step that moved the puzzle. A tie-break only shrinks tied sets, so a step's chance can only rise (the
    solution kept in a smaller tie) or drop to 0 (the solution demoted): DOWN is the first step that demoted the
    solution, UP the first step whose tie shrank around it."""
    for sp, sn in zip(prev["steps"], now["steps"]):
        if sp["fen"] != sn["fen"]:
            return None
        if kind == "down" and sp["tied"] and not sn["tied"]:
            return sp, sn
        if kind == "up" and sn["tied"] and sn["n_top"] < sp["n_top"]:          # n_top is exact; 'best' keeps only 6
            return sp, sn
    return None


def dig(n: int, name: str, show: int = 0) -> dict:
    from .wiring import MinedOrder
    meas = read_json(record_dir(n) / "measure.json") or {}
    arm = next(m["dump"] for m in meas["measurements"] if m["set"] == name and m["arm"] == "round")
    prev_meas = previous_measure(n) or {}
    prev = next(m["dump"] for m in prev_meas["measurements"] if m["set"] == name and m["arm"] == "round")
    active = read_json(round_dir(n) / "active.json") or {}
    mo = MinedOrder(active.get("order", []))
    P = {r["id"]: r for r in load_dump(prev)}
    N = {r["id"]: r for r in load_dump(arm)}
    common = [k for k in P if k in N]
    up = [k for k in common if expected_score(N[k]) > expected_score(P[k]) + 1e-12]
    down = [k for k in common if expected_score(N[k]) < expected_score(P[k]) - 1e-12]
    print(f"OBS dig round {n} {name} against {prev}: expected {sum(expected_score(P[k]) for k in common):.1f} -> "
          f"{sum(expected_score(N[k]) for k in common):.1f}; puzzles up {len(up)}, down {len(down)}")
    classes = dict(up=Counter(), down=Counter())
    piece = Counter()
    rules = Counter()
    shown = 0
    partial = 0
    delta: Counter = Counter()                                               # the expected score each class moved
    for kind, ids in (("down", down), ("up", up)):
        for k in ids:
            sp = _split(P[k], N[k], kind)
            if sp is None:
                classes[kind]["no such step at the same positions (the lines differ)"] += 1
                continue
            sp, sn = sp
            partial += sp["n_top"] > len(sp["best"]) or sn["n_top"] > len(sn["best"])
            b = chess.Board(sn["fen"])
            res = mo.app.apply(b) if mo.app else dict(verdicts={})

            def preds(u):
                m = chess.Move.from_uci(u)
                return res["verdicts"].get((m.from_square, u), [])
            sol = sn["sol"]
            kept = sn["best"]
            demoted = sorted(set(sp["best"]) - set(sn["best"]))
            if kind == "down":
                vs = preds(sol)
                if not vs:
                    c = "unspoken: no mined idea speaks about the solution"
                elif any(p == "loses" for _, p, _, _ in vs):
                    c = "a rule calls the solution a loss"
                    for lem, p, fired, _ in vs:
                        if p == "loses":
                            rules[f"{lem.id} ({fired or 'its claim'})"] += 1
                    mv = chess.Move.from_uci(sol)
                    after = b.copy(stack=False)
                    after.push(mv)
                    att = bool(after.attackers(after.turn, mv.to_square))
                    dfd = bool(after.attackers(not after.turn, mv.to_square))
                    piece[(chess.piece_name(b.piece_at(mv.from_square).piece_type), "lands attacked" if att else "lands safe",
                           "defended" if dfd else "undefended", "capture" if b.is_capture(mv) else "quiet")] += 1
                elif all(p == "keeps" for u in [sol, *kept] for _, p, _, _ in preds(u)) and all(preds(u) for u in kept):
                    c = "unanimous 'keeps': the composed preference ranked the kept moves higher"
                else:
                    c = "other"
            else:
                dv = [preds(u) for u in demoted]
                if any(p == "loses" for vs in dv for _, p, _, _ in vs):
                    c = "a rule calls a demoted rival a loss"
                elif dv and all(vs for vs in dv) and all(p == "keeps" for vs in dv + [preds(sol)] for _, p, _, _ in vs):
                    c = "unanimous 'keeps': the composed preference ranked the solution higher"
                elif any(not vs for vs in dv):
                    c = "a demoted rival is unspoken (scores 0)"
                else:
                    c = "other"
            classes[kind][c] += 1
            delta[(kind, c)] += expected_score(N[k]) - expected_score(P[k])
            if shown < show:
                shown += 1
                print(f"  {kind} {k} {sn['fen']} solution {sol}; kept {kept}; demoted {demoted}")
                for u in [sol, *demoted, *kept][:6]:
                    print(f"    {u}: " + ", ".join(f"{lem.id} {p}" + (f" [{fired}]" if fired else "") for lem, p, fired, _ in preds(u)))
    for kind in ("down", "up"):
        for c, v in classes[kind].most_common():
            print(f"OBS {name} {kind} {v:3d} (expected {delta[(kind, c)]:+.1f}): {c}")
    print(f"OBS {name}: {partial} of the classified puzzles had a tie longer than the 6 top moves a dump stores; "
          "their classes read the stored 6 (frame_puzzles keeps best[:6]; the tie size n_top is exact)")
    for k, v in piece.most_common(8):
        print(f"OBS {name} down by a loss, the solution: {v:3d} {' '.join(k)}")
    for k, v in rules.most_common(6):
        print(f"OBS {name} down by a loss, the rule: {v:3d} {k}")
    return dict(up=len(up), down=len(down), classes={k: dict(v) for k, v in classes.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args(argv)
    dig(a.round, a.set, a.show)


if __name__ == "__main__":
    main()
