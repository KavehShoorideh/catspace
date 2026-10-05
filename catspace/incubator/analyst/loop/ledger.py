"""THE LEDGER: one page per round, for Kaveh's gate. Every number is read from the stage outputs of the round (never
typed). Sections: inputs, mined, promoted, fired, changed, controls, proposed next, decision. The controls:
- NOTHING FIRES: no instance of any newly promoted idea at the solver's positions of the held-out sets;
- RESULTS FALL: puzzles lost exceed puzzles gained on a held-out set (flagged for judgment, not a verdict);
- UNSOUND (red): an idea with a Lean statement whose axioms or bridge failed; a red round cannot be gated continue;
- WALL TIME: seconds per solver step, the round's player against the baseline's."""
from __future__ import annotations

import time


def controls(out: dict, inputs: dict) -> dict:
    prove = out.get("prove") or {}
    measure = out.get("measure") or {}
    before = set(inputs.get("basis_before_ids", []))
    promoted = prove.get("promoted", [])
    new_ids = [p["candidate"]["id"] for p in promoted if p["candidate"]["id"] not in before and not p["unsound"]]
    fire = measure.get("fire", {})
    inst = {i: sum((fire.get(s, {}).get(i) or {}).get("instances", 0) for s in fire) for i in new_ids}
    nothing = bool(new_ids) and sum(inst.values()) == 0
    falls = []
    for m in measure.get("measurements", []):
        if m["arm"] == "round" and len(m["lost"]) > len(m["gained"]):
            falls.append(dict(set=m["set"], gained=len(m["gained"]), lost=len(m["lost"]), lost_ids=m["lost"][:10]))
    unsound = [dict(id=p["candidate"]["id"], reason=p["reason"]) for p in promoted if p["unsound"]]
    wall = []
    base = {m["set"]: m for m in measure.get("measurements", []) if m["arm"] == "baseline"}
    for m in measure.get("measurements", []):
        if m["arm"] != "round":
            continue
        b = base.get(m["set"], {})
        wall.append(dict(set=m["set"], round=(m["wall_s"] / m["steps"]) if m.get("wall_s") and m["steps"] else None,
                         baseline=(b["wall_s"] / b["steps"]) if b.get("wall_s") and b.get("steps") else None))
    return dict(new_ids=new_ids, instances=inst, nothing_fires=nothing, results_fall=falls, unsound=unsound, wall=wall,
                red=bool(unsound))


def _fmt_wall(w):
    if w["round"] is None and w["baseline"] is None:
        return f"{w['set']}: not timed"
    r = "n/a" if w["round"] is None else f"{w['round']:.2f}s"
    bl = "n/a" if w["baseline"] is None else f"{w['baseline']:.2f}s"
    ratio = f" ({w['round'] / w['baseline']:.2f}x)" if w["round"] and w["baseline"] else ""
    return f"{w['set']}: {r} a step against {bl}{ratio}"


def render(n: int, out: dict, inputs: dict, ctl: dict, timings: dict, heads: dict) -> str:
    L = []
    w = L.append
    w(f"# Idea loop — round {n:03d} ({time.strftime('%Y-%m-%d %H:%M %Z')})")
    w("")
    w(f"Code {inputs.get('code')}; basis before the round {len(inputs.get('basis_before_ids', []))} ideas (head {heads.get('before', '')[:12]}), "
      f"after {heads.get('after_count', '?')} ideas (head {heads.get('after', '')[:12]}). Stages: "
      + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()) + f"; total {sum(timings.values()) / 60:.1f} min.")
    w("")
    w("## Inputs")
    fr = out.get("frame") or {}
    if fr.get("skipped"):
        w("- No framing this round (baseline round).")
    else:
        c = fr.get("counts", {})
        w(f"- Framed: {c.get('puzzle', 0)} puzzle misses on development rows (i % 97 == 5) and {c.get('game', 0)} game positions where the frames "
          f"rated the move safe and the game took material within the next plies (development games). Ids frozen in inputs_framed.json.")
    w("")
    w("## Mined")
    mi = out.get("mine") or {}
    guards = mi.get("guards", [])
    if mi.get("skipped"):
        w("- Nothing mined (baseline round).")
    else:
        from collections import Counter
        st = Counter(g["held_out"].get("status") for g in guards)
        w(f"- {len(guards)} lemmas from lemma_gen on the framed positions; held-out re-test: " + ", ".join(f"{k} {v}" for k, v in st.items()) + ".")
        top = sorted(guards, key=lambda g: -(g["support"] * (g["held_out"].get("purity") or 0)))[:5]
        for g in top:
            w(f"  - {g['text'][:140]} (support {g['support']}, held-out {g['held_out'].get('status')}, purity {g['held_out'].get('purity') or 0:.2f})")
    w("")
    w("## Promoted")
    pr = out.get("prove") or {}
    promoted = pr.get("promoted", [])
    proved = [p for p in promoted if p["tier"] in ("unconditional", "forcing")]
    conj = [p for p in promoted if p["tier"] == "conjecture"]
    for p in proved:
        br = p["evidence"].get("bridge") or {}
        w(f"- **{p['candidate']['id']}** — {p['tier']}; {p['evidence'].get('theorem')}; axioms {', '.join(p['evidence'].get('axioms') or [])}; "
          f"bridge {br.get('checked')} claims checked, {br.get('wrong')} wrong.")
    if conj:
        w(f"- Conjecture tier: {len(conj)} ideas ({sum(p['unsound'] for p in conj)} unsound). Allowed uses: order, sentence.")
        for p in conj[:3]:
            w(f"  - {p['candidate']['id']}: {p['candidate']['sentence'][:150]}")
    w(f"- Not promoted: {len(pr.get('not_promoted', []))}.")
    us = out.get("use") or {}
    refused = [u for u in us.get("uses", []) if u["refused"]]
    if refused:
        w(f"- Refused uses: " + "; ".join(f"{u['mode']} for {u['idea']} ({u['reason']})" for u in refused[:5]))
    w("")
    w("## Fired (static scan at the solver's positions of the held-out sets)")
    me = out.get("measure") or {}
    fire = me.get("fire", {})
    sets = list(fire)
    if sets:
        w("| idea | " + " | ".join(sets) + " |")
        w("|---|" + "---|" * len(sets))
        ids = sorted({i for s in sets for i in fire[s]}, key=lambda i: (not i.startswith("lean:"), -sum((fire[s].get(i) or {}).get("instances", 0) for s in sets)))
        for i in ids[:12]:
            cells = []
            for s in sets:
                r = fire[s].get(i)
                if not r:
                    cells.append("-")
                    continue
                extra = f", proved {r['proved']}, one-away {r['oneaway']}" if "proved" in r else f", speaks {r.get('speaks', 0)}"
                cells.append(f"{r['instances']}/{r['positions']}{extra}")
            w(f"| {i} | " + " | ".join(cells) + " |")
        if len(ids) > 12:
            w(f"| ... {len(ids) - 12} more | " + " | ".join("" for _ in sets) + " |")
    w("")
    w("## Changed (the player with this round's wiring, paired by position against the frozen plain baselines)")
    rows = me.get("measurements", [])
    base = {m["set"]: m for m in rows if m["arm"] == "baseline"}
    w("| set | baseline tied / single | round tied / single | gained | lost | nodes on the same positions | wall / step |")
    w("|---|---|---|---|---|---|---|")
    for m in rows:
        if m["arm"] != "round":
            continue
        b = base[m["set"]]
        p = m["paired"]
        wall = f"{m['wall_s'] / m['steps']:.2f}s" if m.get("wall_s") else "not timed"
        w(f"| {m['set']} | {b['tied']}/{b['puzzles']} / {b['single']} | {m['tied']}/{m['puzzles']} / {m['single']} | {len(m['gained'])} | {len(m['lost'])} | "
          f"{p['before']:,} → {p['after']:,} ({p['ratio']:.3f}; fewer {p['fewer']}, more {p['more']}) | {wall} |")
    w("")
    w("## Controls")
    w(f"- NOTHING FIRES: " + ("**FLAG** — no newly promoted idea has an instance in the held-out sets" if ctl["nothing_fires"]
                               else f"ok — newly promoted ideas with instances: {sum(1 for v in ctl['instances'].values() if v)} of {len(ctl['new_ids'])}"))
    w(f"- RESULTS FALL: " + ("**FLAG** for judgment — " + "; ".join(f"{f['set']}: lost {f['lost']} > gained {f['gained']}" for f in ctl["results_fall"])
                              if ctl["results_fall"] else "ok — no held-out set lost more puzzles than it gained"))
    w(f"- UNSOUND: " + ("**RED** — " + "; ".join(f"{u['id']}: {u['reason']}" for u in ctl["unsound"]) + ". This round cannot be gated continue."
                        if ctl["unsound"] else "ok — every Lean statement has standard axioms and a clean bridge"))
    w(f"- WALL TIME: " + ("; ".join(_fmt_wall(x) for x in ctl["wall"]) or "not measured"))
    w("")
    w("## Proposed next")
    props = []
    if ctl["unsound"]:
        props.append("Investigate the unsound ideas before anything else: a wrong claim means the detector, the kernel or the bridge disagree.")
    if ctl["nothing_fires"]:
        props.append("Nothing new fires on the held-out sets: frame from the sets' own themes, or swap the miner.")
    if conj and not proved:
        props.append("All new ideas are conjecture tier: slot R2 (the guard-to-Lean translator) and R3 (automatic proofs) decide whether any can close search.")
    if ctl["results_fall"]:
        props.append("A set lost puzzles: dig the lost ids (MISS anatomy) before keeping the round's order uses.")
    w("\n".join(f"- {x}" for x in props) or "- Continue: no control flagged.")
    w("")
    w("## Decision")
    w(f"- pending: `python -m catspace.incubator.analyst.loop.gate --round {n} --decision continue|rollback|swap [--stage NAME] --note \"...\"`")
    return "\n".join(L) + "\n"
