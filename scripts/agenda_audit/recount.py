"""Plan 75.2: RECOUNT the audit's CHESS FACT rows on the current agenda.py / ponder.py (the number to drive to zero). Read-only on the engine.

  .venv/bin/python scripts/agenda_audit/recount.py <scratch dir> [--snap <guard snapshot dir>] [--before <a previous recount's summary.json>]

The audit's scripts (extract / classify / count / facts) are copied into the scratch dir with their scratch path pointed there, and run on the files of the
working tree. The classifier's hand overrides and hand-read items are keyed by the lines of agenda.py / ponder.py at the audit's commit (BASE): this script
maps every line difflib finds UNCHANGED from BASE to its line now (linemap.json); an override on a line that changed is dropped and printed (its hand reading
was of the old text: the new line is classified by the content rules alone -- read the list). Prints OBS lines: the rows by class, the CHESS FACT rows, the
rows per family; with --before, the families whose rows moved; with --snap, count.py's per-family case counts too. Writes <scratch>/summary.json."""
import collections
import difflib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = "65024fbf"                      # the commit the audit read (docs/AGENDA_AUDIT.md: agenda.py as of 635990e9 = 65024fbf)
MID = "9e4e5673"                       # main after plan 74 (75.2's parent): BASE -> MID by difflib and the plan 74 agent's hand carries, then MID -> the working tree
FILES = {"agenda.py": "catspace/incubator/analyst/discovery/agenda.py", "ponder.py": "catspace/incubator/analyst/discovery/ponder.py"}
# the plan 74 agent's recount (2026-10-27, its copy of the classifier at /tmp/audit751_plan74) carried three readings by hand across lines plan 74 rewrote with
# the same fact (the heap became the worklist): BASE line -> MID line
CARRY = {"agenda.py": {1575: 1734, 4236: 4433, 4283: 4480}}
# plan 75.2: the hand readings that STILL APPLY on a line 75.2 edited (MID line -> the override regexes, None = every atom): the edit moved a store value of the line
# (a goal word, a stake, the unit) but the family's own fact stays written there; each is carried to the edited line (difflib's most similar line of the changed block)
# and printed. Every other reading on an edited line is dropped and printed: its fact left the line.
KEEP_ON_EDIT = {"agenda.py": {
    1042: [None],                                      # D14: their promote claim answered by 'stop their pawn' (the test now reads the GOAL entry's role; the root kind stays D14's)
    1919: ["goal|chase_depth"],                        # B6: the follow list's depth limit (stage b)
    2311: ["checkers"], 2322: ["reach"], 2325: ["promoter"], 2328: ["threat"],      # D21 / D1e / D14: the save and mate sub-kinds by their pattern keys (stage d)
    2763: ["goal"],                                    # D4: the smallest loss is for the keep goal only
    2900: ["goal|checkers"], 2937: ["goal|checkers"],  # B2: the checks evaded in a row (stage b)
    3267: [None],                                      # D8c: a promote route's reply answered by the material won back
    3273: ["dbal"], 3275: ['dbal >= target|claim\\"\\] is None'], 3303: ["kind|>= 0"],      # D2: the static leaf (stage d)
    954: ["table"],                                    # D20: a table root matches no arrangement
    4690: [None],                                      # D21: in check, the captures of the checker are the routes
    4184: ["len\\(path\\)"], 4897: ["n == 1"],         # RULE: the push promotes on its last step
    4281: ["^evading$", "_keep_goal", "goal|chase_depth|CHASE_SAVES|DEEP_WIN", "min_stake"],      # D21 / A10 / B6 / A7: the chase's guard (stage b)
    3602: ["stake"],                                   # D3: the exchange must reach the goal's stake
    4105: ["move|reach"],                              # B5: the goal's own move or a reach step is a route of the walk of checks (stage b)
    4472: ["goal"],                                    # D10: the refuter fix is for the gain goal only
    4498: ["material"],                                # D8c: a promote route answered by material is a win claim
    4504: ["kind|repetition|balance"],                 # A8: a repetition when behind is a draw (stage a's draw: not moved)
    4737: ["MATE_VALUE|in tg|m.from_square"], 4796: [None],      # D6: the parry set of a threat; a mate threat is parried only by the threatener's capture or a mate
}}
SCRIPTS = ("extract.py", "classify.py", "count.py", "facts.py", "findings_text.py")


def linemap(old: list, new: list, replaced: "dict | None" = None) -> dict:
    """{old line: new line} for the lines difflib finds equal; `replaced` (a dict) receives, for each line of a changed block, the most similar line of the block that
    replaced it (ratio >= 0.6), the edited line."""
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    out = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[str(i1 + k + 1)] = j1 + k + 1
        elif tag == "replace" and replaced is not None:
            for i in range(i1, i2):
                best = max(range(j1, j2), key=lambda j: difflib.SequenceMatcher(None, old[i], new[j], autojunk=False).ratio())
                if difflib.SequenceMatcher(None, old[i], new[best], autojunk=False).ratio() >= 0.4:
                    replaced[str(i + 1)] = best + 1
    return out


def main(argv: list) -> int:
    out = os.path.abspath(argv[0])
    snap = argv[argv.index("--snap") + 1] if "--snap" in argv else None
    before = argv[argv.index("--before") + 1] if "--before" in argv else None
    os.makedirs(out, exist_ok=True)
    for f in SCRIPTS:
        src = open(os.path.join(HERE, f)).read().replace("/tmp/audit751", out)
        open(os.path.join(out, f), "w").write(src)
    lm, olds, news, carry = {}, {}, {}, {}
    for short, path in FILES.items():
        olds[short] = subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:{path}"], capture_output=True, text=True, check=True).stdout.splitlines()
        mid = subprocess.run(["git", "-C", ROOT, "show", f"{MID}:{path}"], capture_output=True, text=True, check=True).stdout.splitlines()
        news[short] = open(os.path.join(ROOT, path)).read().splitlines()
        m1 = linemap(olds[short], mid)
        m1.update({str(a): b for a, b in CARRY.get(short, {}).items()})
        rep: dict = {}
        m2 = linemap(mid, news[short], rep)
        lm[short] = {a: m2[str(b)] for a, b in m1.items() if str(b) in m2}
        keep = KEEP_ON_EDIT.get(short, {})
        carry[short] = {a: dict(to=rep[str(b)], rx=keep[b], mid=b) for a, b in m1.items() if str(b) not in m2 and b in keep and str(b) in rep}
        missing = [b for b in keep if str(b) in m2 or str(b) not in rep]
        if missing:
            print(f"OBS audit recount: KEEP_ON_EDIT lines of {short} not edited or not found: {missing}")
    json.dump(lm, open(os.path.join(out, "linemap.json"), "w"))
    json.dump(carry, open(os.path.join(out, "carry.json"), "w"))
    py = sys.executable
    for short, path in FILES.items():
        got = subprocess.run([py, os.path.join(out, "extract.py"), os.path.join(ROOT, path)], capture_output=True, text=True, check=True).stdout
        open(os.path.join(out, f"items_{short.split('.')[0]}.jsonl"), "w").write(got)
    r = subprocess.run([py, os.path.join(out, "classify.py")], capture_output=True, text=True, cwd=out)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        return 1
    rows = [json.loads(x) for x in open(os.path.join(out, "rows.jsonl"))]
    classes = ["LOGIC", "RULE", "NAME", "CHESS FACT", "HEURISTIC"]
    cc = collections.Counter(x["cls"] for x in rows)
    print(f"OBS audit rows total {len(rows)}: " + ", ".join(f"{c} {cc[c]}" for c in classes))
    for fl in FILES:
        c2 = collections.Counter(x["cls"] for x in rows if x["file"] == fl)
        print(f"OBS audit rows {fl} {sum(c2.values())}: " + ", ".join(f"{c} {c2[c]}" for c in classes))
    fam = collections.Counter(x["fact"] for x in rows if x["cls"] == "CHESS FACT")
    print(f"OBS audit CHESS FACT rows {sum(fam.values())} (families {len(fam)})")
    for f, n in sorted(fam.items(), key=lambda t: (-t[1], t[0])):
        print(f"OBS audit fact {f}: rows {n}")
    dropped = json.load(open(os.path.join(out, "dropped.json")))
    carried = json.load(open(os.path.join(out, "carried.json"))) if os.path.exists(os.path.join(out, "carried.json")) else []
    print(f"OBS audit hand readings carried onto an edited line (the family's fact stays there): {len(carried)}")
    for d in carried:
        print(f"   {d['file']}:{d['line']} -> {d['to']} {d['what'][:120]}  | now: {news[d['file']][d['to'] - 1].strip()[:140]}")
    print(f"OBS audit hand readings dropped (their line changed since {BASE}): {len(dropped)}")
    for d in dropped:
        print(f"   {d['file']}:{d['line']} {d['what'][:150]}  | was: {olds[d['file']][d['line'] - 1].strip()[:140]}")
    summ = dict(rows=len(rows), classes=dict(cc), fact_rows=sum(fam.values()), families=dict(fam), dropped=len(dropped))
    if before:
        b = json.load(open(before))
        print(f"OBS audit recount CHESS FACT rows {b['fact_rows']} -> {summ['fact_rows']}")
        for f in sorted(set(b["families"]) | set(fam)):
            x, y = b["families"].get(f, 0), fam.get(f, 0)
            if x != y:
                print(f"OBS audit recount fact {f}: rows {x} -> {y}")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    if snap:
        store = os.path.join(ROOT, "catspace/incubator/analyst/discovery/lemmas")
        r = subprocess.run([py, os.path.join(out, "count.py"), snap, store], capture_output=True, text=True, cwd=out)
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
