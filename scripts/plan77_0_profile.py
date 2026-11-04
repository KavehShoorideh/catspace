"""77.0 (1) THE DERIVATION SHARE: where a search's wall time goes, by category (self time, cProfile)."""
import cProfile, pstats, sys, time, os, json
from collections import defaultdict
sys.setrecursionlimit(20000)
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import state_of
from catspace.incubator.analyst.discovery import puzzle_apply as PA

def cat(fn, name):
    f = fn.replace("\\", "/")
    if f == "~":
        if "analyst_core" in name or "Pos" in name or "Ledger" in name:
            n = name.lower()
            if any(k in n for k in ("regress", "fixes", "gdl", "spec", "derive", "relation")):
                return "rust: regression / spec"
            return "rust: words, handle, reply table"
        return "python builtins"
    if "/chess/" in f:
        return "python-chess"
    if "/discovery/" in f:
        m = f.rsplit("/", 1)[1]
        if m in ("regression.py", "regress.py", "graph.py", "gdl_chess.py", "gdl.py", "words.py", "race.py", "landing.py", "ontology.py", "terms.py"):
            return "python: derivation glue (" + m + ")"
        if m in ("facts.py", "lemma_store.py", "state.py"):
            return "python: store (" + m + ")"
        return "python: agenda / questions (" + m + ")"
    return "python: other (" + f.rsplit("/", 1)[-1] + ")"

def run(cases, label):
    pr = cProfile.Profile()
    t0 = time.monotonic(); qs = inner = 0; dec = []
    pr.enable()
    for fen, us, secs in cases:
        ag = Agenda(state_of(fen, us), seconds=secs); ag.ask(); d = ag.decision()
        qs += ag.asked + ag.ctx.asked; inner += ag.ctx.agendas; dec.append(d.what if d else None)
    pr.disable()
    wall = time.monotonic() - t0
    st = pstats.Stats(pr)
    agg = defaultdict(float)
    for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
        agg[cat(fn, name)] += tt
    tot = sum(agg.values())
    print(f"OBS 77.0 (1) {label}: wall {wall:.1f} s (profiled), self time {tot:.1f} s, questions {qs}, inner agendas {inner}, decisions {dec[:6]}")
    groups = defaultdict(float)
    for k, v in agg.items():
        g = k.split(" (")[0]
        groups[g] += v
    for g, v in sorted(groups.items(), key=lambda t: -t[1]):
        print(f"   {g:42s} {v:7.2f} s  {100*v/tot:5.1f} %")
    top = sorted(agg.items(), key=lambda t: -t[1])[:8]
    print("   top modules:", ", ".join(f"{k.split('(')[-1].rstrip(')')} {100*v/tot:.1f}%" for k, v in top))

def fen_of(setname, pid):
    p = [x for x in PA.load_set(setname) if x["id"] == pid][0]
    return p["fen"], ("w" if p["fen"].split()[1] == "w" else "b")

which = sys.argv[1]
if which == "w0vfo":
    f, u = fen_of("rand8_10", "W0VFO"); run([(f, u, 10)], "W0VFO 10 s")
elif which == "disc":
    run([("r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1", "w", 30)], "discovery 30 s")
elif which == "loop":
    cs = []
    for s, pid in (("rand14_10", "9QsbL"), ("rand15_10", "2CEDb"), ("rand16_10", "L2luA"), ("rand17_10", "r3oph"), ("rand14_10", "KSMKJ")):
        f, u = fen_of(s, pid); cs.append((f, u, 15))
    run(cs, "five loop-2 puzzles 15 s")
elif which == "ten":
    import importlib.util
    spec = importlib.util.spec_from_file_location("ti", "scripts/trace_identity.py"); ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(ti)
    cs = [(fen, us, None) for _pid, fen, us in ti._cases()]
    run(cs, "ten guard cases, count budget")
