"""THE GENERIC THREAT, MEASURED (Kaveh 2026-10-10: "a threat should have a value ... one threat is losing a piece, the
other is becoming mate -- cornering a king or losing a king ... protecting against the threat should be generic and not
specifically a piece"). A piece's JOB is parrying threats; the lone-attack lemma refuses an attack on a piece that can
step away, but stepping away gives up its job. Test, the 'pinned to what' lift generalized: after the refused move, LIFT
the attacked piece off the board, give us the move, run the planner (its immediate win: material or mate) -- against
the same with the piece still there (the opponent passing). A job = the lifted board wins more. Surprises vs 1000 random
correct refusals, from lemma_reviewer's saved firings."""
import json, os, random, statistics, time
from multiprocessing import Pool

import chess

from catspace.io import paths
from catspace.incubator.analyst.lemma_reviewer import judge


def _imm(fen):
    import analyst_core
    r = analyst_core.plan_trace(fen)
    return (r[5][2], r[5][3]) if r[5] else (0, "")


def one(c):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    t = time.time()
    b = chess.Board(c["fen"]); b.push_uci(c["move"])
    tsq = chess.parse_square(c["target_sq"])
    stay = b.copy(); stay.push(chess.Move.null())                     # they pass: the piece still there
    lift = b.copy(); lift.remove_piece_at(tsq); lift.push(chess.Move.null())
    if lift.was_into_check() or not lift.is_valid():                 # lifting it exposes their king: a mate-level job
        return c, None, "lifting it exposes their king", (time.time() - t) * 1000
    import analyst_core
    # with the piece there, winning IT is the attack itself: the baseline is the best win on any OTHER target
    # (bug 2026-10-10: comparing against the whole immediate win counted the attacked queen and hid every job)
    wins, _ = analyst_core.threats_after(c["fen"], c["move"])
    g_stay = max([g for _, t_, g in wins if t_ != c["target_sq"]] + [0])
    g_lift, how = _imm(lift.fen())
    return c, (g_stay, g_lift), how, (time.time() - t) * 1000


if __name__ == "__main__":
    d = json.load(open(paths.experiment("lemma_reviewer_lone_attack.json")))
    fired = [c for c in d["cases"] if c["stratum"] != "passed"]
    sur = sorted([c for c in fired if judge(c)], key=lambda c: (not c["sf_tactic"], -c["gain"]))
    ok = random.Random(1).sample([c for c in fired if not judge(c)], 1000)
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rs = p.map(one, sur + ok, chunksize=4)
    print(f"TIME {time.time() - t0:.0f} s for {len(rs)} firings")
    out = []
    for name, g in (("surprises", rs[:len(sur)]), ("correct refusals", rs[len(sur):])):
        job = 0; mate = 0; either = 0
        for c, gs, how, ms in g:
            j = gs is None or gs[1] > gs[0]                                # the lift wins more than the other targets did
            job += j; mate += (gs is not None and gs[1] >= 500) or gs is None
            either += j or (gs is not None and gs[0] > 0)                  # a job given up, or a second threat
            if name == "surprises":
                out.append(dict(fen=c["fen"], move=c["move"], job=j, stay=gs[0] if gs else None, lift=gs[1] if gs else None, how=how))
        ms = [x[3] for x in g]
        print(f"OBS {name:17s} n {len(g):4d}: lifting the attacked piece wins more (it has a job) {job:4d} ({100 * job / len(g):.0f}%), "
              f"of which a mate or an exposed king {mate}; a job OR a second threat {either} ({100 * either / len(g):.0f}%); ms median {statistics.median(ms):.0f}")
    json.dump(out, open(paths.experiment("lone_attack_jobs.json"), "w"))
    for i, r in enumerate(out[:12]):
        print(f"  surprise {i + 1:2d}: job {'yes' if r['job'] else 'no '}  stay {r['stay']}  lift {r['lift']}  {r['how'][:90]}")
