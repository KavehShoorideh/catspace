"""PROVE: a tier for every candidate, and the evidence for it. Three paths:
- a Lean theorem (registered, or produced by a translator): its axioms checked by lean/ideas/scripts/check_axioms.sh
  (anything beyond propext, Classical.choice, Quot.sound fails) and its bridges run (loop.bridges: the detector's
  claims checked by exhaustive search in python-chess). Axioms standard + bridges clean + at least one claim checked
  -> the theorem's tier (unconditional or forcing). A failed axiom check or any wrong claim -> UNSOUND (conjecture,
  and the round's unsound control turns red);
- lemma_gen's held-out re-test for a mined guard: survived -> conjecture tier; did not -> not promoted;
- research slot R3 (automatic proof attempts) plugs in here.
Registered ideas already in the basis are re-checked every round (config recheck_registered): a detector or kernel
change that breaks a bridge is caught the round it happens."""
from __future__ import annotations

import subprocess

from catspace.io import paths

from .. import registry
from ..bridges import BRIDGES
from ..contracts import STANDARD_AXIOMS, Candidate, Evidence, Promoted, from_dict, to_dict

SCRIPT = paths.REPO_ROOT / "lean" / "ideas" / "scripts" / "check_axioms.sh"


def axioms(theorems: list[str]) -> tuple[dict, str]:
    """-> ({theorem: [axioms] or None if missing}, the script's output)."""
    if not theorems:
        return {}, ""
    r = subprocess.run([str(SCRIPT), *theorems], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        if line.startswith("AXIOMS "):
            _, thm, ax = line.split(" ", 2)
            out[thm] = None if ax == "?MISSING" else ([] if ax == "-" else ax.split(","))
    return out, r.stdout + r.stderr


def judge_lean(c: Candidate, ax: list | None, parts: dict) -> tuple[str, bool, str, Evidence]:
    """The tier rule for an idea with a Lean statement. -> (tier, unsound, reason, evidence)."""
    checked = sum(p.get("checked", 0) for p in parts.values())
    wrong = sum(p.get("wrong", 0) for p in parts.values())
    ax_ok = ax is not None and not (set(ax) - set(STANDARD_AXIOMS))
    ev = Evidence(theorem=c.lean["theorem"], axioms=ax, axioms_ok=ax_ok, bridge=dict(checked=checked, wrong=wrong, parts=parts))
    if not ax_ok:
        return "conjecture", True, f"axioms not standard: {ax}", ev
    if wrong:
        return "conjecture", True, f"the bridge found {wrong} wrong claims of {checked}", ev
    if not checked:
        return "conjecture", False, "the bridge checked no claim (unverified, not unsound)", ev
    return c.lean.get("tier", "unconditional"), False, f"Lean theorem with standard axioms; bridge {checked} claims, 0 wrong", ev


def run(ctx) -> dict:
    from ...lean_ideas import LeanIdeas
    cfg = ctx.cfg("prove")
    cands = [from_dict(Candidate, c) for c in (ctx.load("translate") or {}).get("candidates", [])]
    guards = {g["id"]: g for g in (ctx.load("mine") or {}).get("guards", [])}
    leans = [c for c in cands if c.lean]
    ax, ax_out = axioms(sorted({c.lean["theorem"] for c in leans}))
    for line in ax_out.splitlines():
        if line.startswith("AXIOMS"):
            ctx.log(line)
    promoted, not_promoted = [], []
    cache: dict = {}
    extra = cfg.get("extra_bridges") or {}                                    # tests inject bridges by name
    with LeanIdeas() as li:
        for c in leans:
            parts = {}
            for b in c.lean.get("bridges", []):
                if b not in cache:
                    fn = extra.get(b) or BRIDGES[b]
                    cache[b] = fn(li)
                    ctx.log(f"OBS bridge {b}: {cache[b]['checked']} claims checked, {cache[b]['wrong']} wrong -- {cache[b]['what']} ({cache[b].get('seconds', 0):.0f}s)")
                parts[b] = {k: v for k, v in cache[b].items() if k != "examples"} | {"examples": cache[b].get("examples", [])[:3]}
            tier, unsound, reason, ev = judge_lean(c, ax.get(c.lean["theorem"]), parts)
            promoted.append(Promoted(c, tier, ev, unsound, reason, ctx.prov("prove", [c.id])))
    for c in cands:
        if c.lean:
            continue
        g = guards.get(c.guard) or {}
        held = g.get("held_out") or {}
        if held.get("status") in ("PROVED", "THEOREM"):
            promoted.append(Promoted(c, "conjecture", Evidence(held_out=held), False,
                                     f"survived lemma_gen's held-out re-test ({held.get('status')}; no Lean statement)", ctx.prov("prove", [c.id])))
        else:
            not_promoted.append(dict(id=c.id, reason=f"held-out re-test: {held.get('status') or 'not run'}", held_out=held))
    ctx.log(f"OBS prove: {len(promoted)} promoted ({sum(p.tier != 'conjecture' for p in promoted)} proved, "
            f"{sum(p.unsound for p in promoted)} unsound), {len(not_promoted)} not promoted")
    return dict(promoted=[to_dict(p) for p in promoted], not_promoted=not_promoted, axioms=ax, axioms_output=ax_out)
