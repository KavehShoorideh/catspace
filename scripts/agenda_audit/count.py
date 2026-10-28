"""Plan 75.1 audit, step 3: per CHESS FACT family, the guard cases whose trace or word list passes through it (its signature), at tier 0 and tier 2,
and the DECISION lines that cite it; the store names checked against the lemma store. Prints OBS lines; writes counts.json. Read-only."""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, "/tmp/audit751")
from facts import FACTS  # noqa: E402

SNAP = sys.argv[1]
STORE = sys.argv[2]

names = set()
for f in glob.glob(os.path.join(STORE, "*.json")):
    names.add(json.load(open(f))["name"])
bad = sorted({n for fa in FACTS.values() for n in fa["store"] if n not in names})
print(f"OBS audit store names checked: {sum(len(fa['store']) for fa in FACTS.values())} citations of {len(names)} entries; not in the store {len(bad)}" + (f": {bad}" if bad else ""))

cases = {}
for tier in ("t0", "t2"):
    for d in (f"ten_{tier}", f"wide_{tier}"):
        for f in glob.glob(os.path.join(SNAP, d, "*.txt")):
            b = os.path.basename(f)
            if b.endswith((".reads.txt", ".leaf.txt", ".why.txt")):
                continue
            cid = b.split(".")[0]
            cases.setdefault((tier, cid), []).append(open(f, errors="replace").read().splitlines())
ids = {t: sorted({c for (tt, c) in cases if tt == t}) for t in ("t0", "t2")}
print(f"OBS audit guard snapshots read: tier 0 cases {len(ids['t0'])}, tier 2 cases {len(ids['t2'])} (ten + wide, the count budget, PYTHONHASHSEED=0; main ba587d27's agenda.py = this commit's)")

rows = [json.loads(l) for l in open("/tmp/audit751/rows.jsonl")]
sites = collections.Counter(r["fact"] for r in rows if r["cls"] == "CHESS FACT")
out = {}
for fid, fa in FACTS.items():
    res = dict(t0=None, t2=None, dec=None, sites=sites.get(fid, 0))
    if fa["sig"]:
        rx = re.compile(fa["sig"])
        drx = re.compile(fa.get("dec") or fa["sig"])
        dec = 0
        for t in ("t0", "t2"):
            n = 0
            for cid in ids[t]:
                hit = False
                for lines in cases[(t, cid)]:
                    for line in lines:
                        if line.startswith("DECISION") and drx.search(line[9:]):
                            dec += 1
                        if not hit and rx.search(line):
                            hit = True
                n += hit
            res[t] = n
        res["dec"] = dec
    out[fid] = res
json.dump(out, open("/tmp/audit751/counts.json", "w"), indent=1)
for fid, r in sorted(out.items(), key=lambda kv: (-((kv[1]["t0"] or 0) + (kv[1]["t2"] or 0)), -(kv[1]["dec"] or 0), -kv[1]["sites"])):
    print(f"OBS audit fact {fid}: cases t0 {r['t0'] if r['t0'] is not None else '-'} / t2 {r['t2'] if r['t2'] is not None else '-'} of 109, DECISION lines citing {r['dec'] if r['dec'] is not None else '-'} of 218, rows {r['sites']}")
