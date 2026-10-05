"""THE LEDGER: one page per round, for Kaveh's gate. Every number is read from the stage outputs of the round (never
typed). Sections: inputs, mined, promoted, fired, changed, controls, proposed next, decision. The controls:
- NOTHING FIRES: no newly promoted idea FIRES at the solver's positions of the held-out sets -- a Lean idea fires where
  it is proved there (an instance is only a candidate: every pawn is a passer candidate), a mined idea where it gives
  a verdict on a move;
- RESULTS FALL: a held-out set's expected score (the puzzles a player picking at random among its tied top moves
  solves; measure.expected_score) fell against the plain baseline or against the previous kept round (flagged for
  judgment, not a verdict). Not 'lost > gained' in tied-set membership: a tie-break only shrinks tied sets, so that
  count falls whenever one acts, right or wrong (round 1, 2026-10-05);
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

    def fires(rec):
        rec = rec or {}
        return rec.get("proved", 0) if "proved" in rec else rec.get("speaks", 0)
    inst = {i: sum(fires(fire.get(s, {}).get(i)) for s in fire) for i in new_ids}
    nothing = bool(new_ids) and sum(inst.values()) == 0
    falls = []
    for m in measure.get("measurements", []):
        if m["arm"] != "round":
            continue
        p, v = m.get("paired") or {}, m.get("vs_previous") or {}
        if m.get("expected") is None or p.get("expected_before") is None:   # no dumps to score: the old count, labelled
            if len(m["lost"]) > len(m["gained"]):
                falls.append(dict(set=m["set"], why=[f"lost {len(m['lost'])} > gained {len(m['gained'])} in tied-set membership (no expected score)"],
                                  down=m["lost"][:10]))
            continue
        why = []
        if m["expected"] < p["expected_before"] - 1e-9:
            why.append(f"expected {m['expected']:.1f} < plain {p['expected_before']:.1f}")
        if v and v["expected_after"] < v["expected_before"] - 1e-9:
            why.append(f"expected {v['expected_after']:.1f} < previous round {v['expected_before']:.1f}")
        if why:
            falls.append(dict(set=m["set"], why=why, down=(v.get("down") if v else p.get("down", []))[:10]))
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
    w(f"Code {inputs.get('code')} (ledger rendered by {heads.get('render_code', inputs.get('code'))}); "
      f"basis before the round {len(inputs.get('basis_before_ids', []))} ideas (head {heads.get('before', '')[:12]}), "
      f"after {heads.get('after_count', '?')} ideas (head {heads.get('after', '')[:12]}). Stages: "
      + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()) + f"; total {sum(timings.values()) / 60:.1f} min.")
    codes = sorted({c for c in (heads.get("stage_codes") or {}).values() if c and c != inputs.get("code")})
    if codes:
        w("Stages ran under code " + ", ".join(f"{st} {c}" for st, c in (heads.get("stage_codes") or {}).items() if c in codes) + ".")
    if heads.get("render_only"):
        w("Rendered from the stage outputs on disk (--render-only)" + (f"; the code of {', '.join(heads['stale'])} changed since they ran "
          "(or predates the source digest)." if heads.get("stale") else "; the analyst and Lean source are unchanged since the stages ran."))
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
    w("## Changed (the player with this round's wiring, paired by position: plain = the frozen baseline, previous = the previous kept round's arm)")
    rows = me.get("measurements", [])
    base = {m["set"]: m for m in rows if m["arm"] == "baseline"}

    def num(x, f="{:.1f}"):
        return "—" if x is None else f.format(x)
    w("| set | expected score: plain / previous / round | single top move: plain / previous / round | puzzles up / down vs previous | solution among tied (context): plain / previous / round | nodes vs plain on the same positions | wall / step |")
    w("|---|---|---|---|---|---|---|")
    for m in rows:
        if m["arm"] != "round":
            continue
        b = base[m["set"]]
        p = m["paired"]
        v = m.get("vs_previous") or {}
        ud = f"{len(v['up'])} / {len(v['down'])}" if v else (f"{len(p['up'])} / {len(p['down'])} (vs plain)" if "up" in p else "—")
        wall = f"{m['wall_s'] / m['steps']:.2f}s" if m.get("wall_s") else "not timed"
        w(f"| {m['set']} | {num(b.get('expected'))} / {num(v.get('expected_before'))} / {num(m.get('expected'))} | "
          f"{b['single']} / {num(v.get('single_before'), '{}')} / {m['single']} | {ud} | "
          f"{b['tied']} / {num(v.get('tied_before'), '{}')} / {m['tied']} of {m['puzzles']} | "
          f"{p['before']:,} → {p['after']:,} ({p['ratio']:.3f}; fewer {p['fewer']}, more {p['more']}) | {wall} |")
    w("")
    w("Expected score: the puzzles a player picking at random among its tied top moves solves. A tie-break can only remove "
      "moves from the tied set, so 'solution among tied' can only fall and 'single top move' only rise when one acts; "
      "the expected score moves either way and is what RESULTS FALL reads.")
    w("")
    w("## Controls")
    w(f"- NOTHING FIRES: " + ("**FLAG** — no newly promoted idea fires in the held-out sets (a Lean idea fires where it is proved, a mined one where it gives a verdict on a move)" if ctl["nothing_fires"]
                               else f"ok — newly promoted ideas that fire: {sum(1 for v in ctl['instances'].values() if v)} of {len(ctl['new_ids'])}"))
    w(f"- RESULTS FALL: " + ("**FLAG** for judgment — " + "; ".join(f"{f['set']}: " + ", ".join(f["why"]) for f in ctl["results_fall"])
                              if ctl["results_fall"] else "ok — no held-out set's expected score fell against the plain baseline or the previous round"))
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
    new_ids = set(ctl["new_ids"])                                            # the registered ideas are re-checked every round
    if any(p["candidate"]["id"] in new_ids for p in conj) and not any(p["candidate"]["id"] in new_ids for p in proved):
        props.append("All new ideas are conjecture tier: slot R2 (the guard-to-Lean translator) and R3 (automatic proofs) decide whether any can close search.")
    if ctl["results_fall"]:
        props.append("A set's expected score fell: dig the puzzles that went down (MISS anatomy) before keeping the round's uses.")
    lopsided = [m["set"] for m in rows if m["arm"] == "round" and m.get("vs_previous")
                and len(m["vs_previous"]["down"]) > len(m["vs_previous"]["up"]) and m["set"] not in {f["set"] for f in ctl["results_fall"]}]
    if lopsided:
        props.append(f"More puzzles went down than up against the previous round on {', '.join(lopsided)}, though the expected score did not fall: "
                     "the gain sits in fewer puzzles; dig the down ids (MISS anatomy) before relying on the round's uses.")
    slow = [x["set"] for x in ctl["wall"] if x["round"] and x["baseline"] and x["round"] / x["baseline"] > 1.5]
    if slow:
        props.append(f"Time per step above 1.5x the baseline on {', '.join(slow)}: profile the round's uses before keeping them.")
    w("\n".join(f"- {x}" for x in props) or "- Continue: no control flagged.")
    w("")
    w("## Decision")
    w(f"- pending: `python -m catspace.incubator.analyst.loop.gate --round {n} --decision continue|rollback|swap [--stage NAME] --note \"...\"`")
    return "\n".join(L) + "\n"
