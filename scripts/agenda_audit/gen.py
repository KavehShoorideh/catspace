"""Plan 75.1 audit, step 4: write docs/AGENDA_AUDIT.md from rows.jsonl, counts.json and the fact catalogue; print the OBS lines it writes."""
import collections
import json
import sys

sys.path.insert(0, "/tmp/audit751")
from facts import FACTS  # noqa: E402
from classify import EXTRA  # noqa: E402  (re-runs the classifier: rows.jsonl is rewritten identically)

OUT = sys.argv[1]
rows = [json.loads(l) for l in open("/tmp/audit751/rows.jsonl")]
counts = json.load(open("/tmp/audit751/counts.json"))
items = [json.loads(l) for f in ("items_agenda.jsonl", "items_ponder.jsonl") for l in open("/tmp/audit751/" + f)]
CLASSES = ["LOGIC", "RULE", "NAME", "CHESS FACT", "HEURISTIC"]

obs = []
ak = collections.Counter((i["file"], i["kind"]) for i in items)
for fl in ("agenda.py", "ponder.py"):
    n = sum(v for (f, _k), v in ak.items() if f == fl) + sum(1 for e in EXTRA if e[0] == fl)
    obs.append(f"OBS audit atoms {fl}: {n} (" + ", ".join(f"{k} {ak[(fl, k)]}" for k in ("CONST", "BRANCH", "COMPARE", "ORDER", "LITERAL", "KIND")) + f", read by hand {sum(1 for e in EXTRA if e[0] == fl)})")
cc = collections.Counter(r["cls"] for r in rows)
obs.append(f"OBS audit rows total {len(rows)}: " + ", ".join(f"{c} {cc[c]}" for c in CLASSES))
for fl in ("agenda.py", "ponder.py"):
    c2 = collections.Counter(r["cls"] for r in rows if r["file"] == fl)
    obs.append(f"OBS audit rows {fl} {sum(c2.values())}: " + ", ".join(f"{c} {c2[c]}" for c in CLASSES))
cf = [r for r in rows if r["cls"] == "CHESS FACT"]
ex = [r for r in cf if FACTS[r["fact"]]["proposed"] is None]
fams = sorted({r["fact"] for r in cf})
fex = [f for f in fams if FACTS[f]["proposed"] is None]
obs.append(f"OBS audit CHESS FACT rows {len(cf)}: with an existing store entry {len(ex)}, without (an entry proposed) {len(cf) - len(ex)}; families {len(fams)} (existing {len(fex)}, proposed {len(fams) - len(fex)})")
obs.append(f"OBS audit HEURISTIC rows {cc['HEURISTIC']}")
to = sum(1 for r in cf if r["text_only"])
obs.append(f"OBS audit CHESS FACT rows that only shape trace text {to} (counted: the knowledge is still written in agenda.py)")
obs += [l.strip() for l in open("/tmp/audit751/count.log") if l.startswith(("OBS audit store", "OBS audit guard"))]
for o in obs:
    print(o)


def ranked(fids):
    def key(f):
        c = counts[f]
        has = c["t0"] is not None
        return (0 if has else 1, -((c["t0"] or 0) + (c["t2"] or 0)), -(c["dec"] or 0), -c["sites"], f)
    return sorted(fids, key=key)


def store_of(fid, short=True):
    fa = FACTS[fid]
    if fa["proposed"] is None:
        ns = fa["store"]
        s = "; ".join((n if len(n) <= 48 else n[:45] + "...") for n in ns[:2]) + (f" (+{len(ns) - 2})" if len(ns) > 2 else "")
        return f"[{fid}] existing: {s}"
    name, kind, reads = fa["proposed"]
    ex = (" (beside existing: " + "; ".join(n if len(n) <= 40 else n[:37] + "..." for n in fa["store"][:2]) + ")") if fa["store"] else ""
    return f"[{fid}] PROPOSED: \"{name}\" ({kind}; reads {reads}){ex}"


def dec_of(fid):
    c = counts[fid]
    if c["t0"] is None:
        return f"no printed signature: {c['sites']} rows"
    return f"cases {c['t0']}/{c['t2']}, DECISION {c['dec']}"


def esc(s):
    return s.replace("|", "\\|").replace("\n", " ")


L = []
A = L.append
A("# AGENDA_AUDIT.md -- plan 75.1, the agenda audit")
A("")
A("Read-only classification of `catspace/incubator/analyst/discovery/agenda.py` (the agenda, the chase, the tiers driver `_ask_tiers`, the pool driver `_ask_pool`, the judge "
  "`judge_next` / `_decides` / `_judge_*` / `_competitor` -- they live there; no other module defines them) and `ponder.py`, as of main 65024fbf (agenda.py last changed by "
  "635990e9, ponder.py earlier). No engine file, test or other doc was changed; no gate, no snapshot, no timed run. Every number below is printed by the audit's own scripts "
  "(OBS lines, section 2). The extractor and classifier are not committed (scratch scripts); the classification rules they apply are stated in section 1.")
A("")
A("## 1. The principle and the five classes")
A("")
A("THE PRINCIPLE (plan 75, Kaveh 2026-10-27). The goal is the OUTCOME: don't lose, then win. We supply NAMES -- attack, move, slide, capture, check, mate, stalemate, draw, win, lose, "
  "the unit kinds, rank / file, tempo -- each bound to a definition of the rule kernel, never to a derived fact. Everything else -- words, valuations (material!), techniques, "
  "stops -- is DERIVED and lives in the lemma store, not in agenda.py.")
A("")
A("THE FIVE CLASSES, as applied (each item exactly one; the first that fits):")
A("")
A("* **LOGIC** -- goal regression, AND-OR over replies, frames and read sets, the heap and its stake order, memoisation and the fact store, the budgets' MECHANICS (a count or a clock "
  "against what is left), the tier / pool plumbing, the switches (a flag's test; what it gates is classed where it acts), trace plumbing (a branch or a print cap that only shapes "
  "a line). A comparison of two machinery quantities (a value with a stake, a price with the time left).")
A("* **RULE** -- a fact of the rules read through the kernel: legal moves, check, mate, stalemate, a capture, castling, promotion, attack, a square's occupant and owner, the side to "
  "move -- even where it gates a chess decision. Rules re-implemented in Python are RULE rows (listed in the findings).")
A("* **NAME** -- a supplied name used as a label: the colours, the square set, square / uci notation, the side letters, a goal or defect named in a trace line.")
A("* **CHESS FACT** -- derived knowledge as code: a goal-kind test or literal that routes the agenda; a stake (mate 100, promotion 8, a unit's worth); the status ladder's thresholds, "
  "line and need; the gain unit (value >= 1); a piece-kind or defect-kind test that decides; the forcing words; a stop's shape (the clock, tempo arithmetic, the second check, the "
  "fuse, the walk of checks); a tie-break keyed by chess content (worth, the shape key, checks first). Each row names its FAMILY (section 4) and its store entry, existing or "
  "proposed (name, kind, reads).")
A("* **HEURISTIC** -- a tuned number or rule with no proof obligation: budget and fuse values, time margins and shares, the tier composition, the policy's 1/n bar, generic "
  "tie-breaks (the uci, the price).")
A("")
A("## 2. OBS lines")
A("")
A("```")
for o in obs:
    A(o)
A("```")
A("")
A("The per-family counts (section 4) are the `OBS audit fact <id>` lines of the count script, printed beside these.")
A("")
A("## 3. The table")
A("")
A("WHAT AN ITEM IS (mechanical, from the AST): a module or class CONSTANT; each operand of an `and` / `or` / `not` in an `if` / `elif` / `while` / `assert` / conditional "
  "expression / comprehension filter, and in a short-circuit outside a test (a BRANCH atom); a COMPARISON outside a test; an ORDER (`sorted` / `min` / `max` / `.sort` with a key, "
  "`heappush`); a LITERAL (a number other than 0 and 1, a `chess.*` or another module's UPPERCASE constant); a KIND literal (the goal / claim / defect kind a question, claim or "
  "defect is built with). Docstrings and f-string text are not items. The literal 1 as the unit of material, a few formulas outside any test and two label dicts were read by "
  "hand ('read by hand' in section 2). ONE ROW = the atoms of one line, one class and one CHESS FACT family: a line that mixes classes gives one row per class.")
A("")
A("Columns: file:line | the item's text (its atoms joined by ' ; ', trimmed; 'in:' gives a literal's statement) | class ('text only': a CHESS FACT that only shapes a trace line) | "
  "store entry ([family] existing name, or PROPOSED name, kind, reads) | decisions depending ('cases t0/t2' = cases of 109 at tier 0 / tier 2 passing through the family, "
  "'DECISION' = DECISION lines of 218 citing it -- the method heads section 4 -- or the row count where nothing is printed). The last two columns are for CHESS FACT rows only.")
A("")
for fl in ("agenda.py", "ponder.py"):
    A(f"### {fl}")
    A("")
    A("| file:line | text | class | store entry | decisions depending |")
    A("|---|---|---|---|---|")
    for r in rows:
        if r["file"] != fl:
            continue
        t = " ; ".join(r["texts"])
        t = t if len(t) <= 120 else t[:117] + "..."
        tag = " (text only)" if r["text_only"] and r["cls"] == "CHESS FACT" else ""
        st = store_of(r["fact"]) if r["cls"] == "CHESS FACT" else "--"
        dd = dec_of(r["fact"]) if r["cls"] == "CHESS FACT" else "--"
        A(f"| {fl}:{r['line']} | `{esc(t).replace('`', chr(39))}` | {r['cls']}{tag} | {esc(st)} | {dd} |")
    A("")
A("## 4. The move order")
A("")
A("The CHESS FACT families in the order they should leave agenda.py for the store, grouped into plan 75.2's stages: (a) the goal ladder and the goal kinds, (b) the stops, (c) the "
  "names, (d) the rest. Inside a stage: most cases passing through first (tier 0 + tier 2), then DECISION lines citing, then rows; families with no printed signature after, by "
  "rows. 'rows' = this audit's table rows of the family (its call sites in agenda.py / ponder.py).")
A("")
A("DECISIONS DEPENDING (the method). The trace guard's snapshots of the 73.5b acceptance (`guard_after/` in session b30a4c85's scratchpad: ten + wide = 109 cases, tier 0 and tier "
  "2, the count budget, PYTHONHASHSEED=0; agenda.py unchanged since) were read, not regenerated. A case PASSES THROUGH a family when its trace or its word list (the claim "
  "lemmas, kill and GAP reasons of every agenda of the run, `<id>.words.txt`) matches the family's printed signature (the regex in the last column: question heads, stop texts, "
  "kill and GAP reasons); a DECISION line CITES it when the signature (or the family's lemma) is on the case's `DECISION` line. A family with no printed signature is ranked by "
  "its rows, and says so. Limits: the count-budget snapshots never run the status driver, the pool or the judge (no clock), so the ladder's and the judge's counts are lower "
  "bounds; an inner agenda's chase notes are not printed in the root trace (the walk of checks shows 0); 'passes through' is an upper bound of 'depends on'. A DECISION line "
  "names only the deciding lemma (the chase, a schema, the unfinished line, 'none'), so most families are cited by none.")
A("")
STAGES = [("a", "(a) the goal ladder and the goal kinds"), ("b", "(b) the stops"), ("c", "(c) the names (bind the name to the kernel; the knowledge goes to the valuation or the store)"), ("d", "(d) the rest")]
overall = []
for sid, title in STAGES:
    A(f"### {title}")
    A("")
    A("| # | family | what is written as code | store entry | cases t0 / t2 (of 109) | DECISION lines (of 218) | rows | signature |")
    A("|---|---|---|---|---|---|---|---|")
    fids = ranked([f for f, fa in FACTS.items() if fa["stage"] == sid and counts.get(f, {}).get("sites", 0) > 0])
    for i, f in enumerate(fids, 1):
        fa, c = FACTS[f], counts[f]
        if fa["proposed"] is None:
            st = "existing: " + "; ".join(fa["store"])
        else:
            n, k, rd = fa["proposed"]
            st = f"PROPOSED \"{n}\" ({k}; reads {rd})" + (("; beside existing: " + "; ".join(fa["store"])) if fa["store"] else "")
        sig = f"`{esc(fa['sig'])}`" if fa["sig"] else "none (ranked by rows)"
        t02 = f"{c['t0']} / {c['t2']}" if c["t0"] is not None else "--"
        A(f"| {i} | {f} | {esc(fa['title'])} | {esc(st)} | {t02} | {c['dec'] if c['dec'] is not None else '--'} | {c['sites']} | {sig} |")
        overall.append(f)
    A("")
A("THE TOP TEN over all stages (by cases passing through, tier 0 + tier 2; ties by DECISION lines, then rows):")
A("")
top = ranked([f for f in FACTS if counts.get(f, {}).get("sites", 0) > 0])[:10]
for i, f in enumerate(top, 1):
    c = counts[f]
    A(f"{i}. {f} -- {FACTS[f]['title'].split(':')[0]}: cases {c['t0']} / {c['t2']}, DECISION lines {c['dec']}, rows {c['sites']}")
A("")
from findings_text import FINDINGS  # noqa: E402
A(FINDINGS)
open(OUT, "w").write("\n".join(L) + "\n")
print("TOP10", [(f, counts[f]["t0"], counts[f]["t2"], counts[f]["dec"], counts[f]["sites"]) for f in top])
