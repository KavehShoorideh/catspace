"""Does the reasoning we have already see a second threat after the lone-attack lemma's refused moves? (Kaveh 2026-10-10:
"maybe it's not the lone attack lemma that's lacking, but bringing in the other lemmas".) analyst_core.threats_after: the
move, the opponent passes, the planner's immediate phase with the stop rule off; a proved win on a target other than the
attacked piece = a second threat. Surprises vs 1000 random correct refusals, from lemma_reviewer's saved firings."""
import json, os, random, statistics, chess
from multiprocessing import Pool
from catspace.io import paths
from catspace.incubator.analyst.lemma_reviewer import judge

def one(c):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    wins, ms = analyst_core.threats_after(c["fen"], c["move"])
    others = sorted({(t, g, s) for s, t, g in wins if t != c["target_sq"]}, key=lambda x: -x[1])
    return c, wins, others, ms

if __name__ == "__main__":
    d = json.load(open(paths.experiment("lemma_reviewer_lone_attack.json")))
    fired = [c for c in d["cases"] if c["stratum"] != "passed"]
    sur = sorted([c for c in fired if judge(c)], key=lambda c: (not c["sf_tactic"], -c["gain"]))
    ok = random.Random(1).sample([c for c in fired if not judge(c)], 1000)
    with Pool(os.cpu_count()) as p:
        rs = p.map(one, sur + ok, chunksize=4)
    rs_s, rs_o = rs[:len(sur)], rs[len(sur):]
    for name, g in (("surprises", rs_s), ("correct refusals", rs_o)):
        sec = sum(bool(o) for _, _, o, _ in g)
        ms = [m for *_, m in g]
        print(f"OBS {name:17s} n {len(g):4d}: a proved win on a second target {sec:4d} ({100*sec/len(g):.0f}%); "
              f"ms median {statistics.median(ms):.1f}, p90 {sorted(ms)[int(.9*len(ms))]:.1f}")
    print("first surprises (the page's order):")
    for i, (c, wins, others, ms) in enumerate(rs_s[:12]):
        b = chess.Board(c["fen"])
        sec = ", ".join(f"{chess.Board(c['fen']).san(chess.Move.from_uci(c['move']))}..{s}: {g} on {t}" for t, g, s in others[:3]) or "none"
        print(f"  {i+1:2d}. {b.san(chess.Move.from_uci(c['move'])):6s} lone target {c['target_sq']}  second-target wins: {sec}")
    json.dump([dict(fen=c["fen"], move=c["move"], wins=w, ms=m) for c, w, _, m in rs_s], open(paths.experiment("lone_attack_second_threat.json"), "w"))
