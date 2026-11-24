"""PLAN 79.49 PART A: THE PATH FROM A RULE LEMMA'S OBLIGATION TO THE PROVER HARNESS AND BACK (SYSTEM_SPEC 11 "One lemma machinery, used twice"; AGENTS.md rule 11: Lean is the
source of truth; piece: the long-term memory's statements; ledger rows 49, 33).

THIS TEXT IS THE INTERFACE, written before the body. Game-free: it reads a lemma through `rule_lemma.py`, writes its obligation through `rule_lemma_lean.py`, and asks the harness
(`prover.py`, plan 73.6) for the proof and for the acceptance check; it names no game and no relation. NOBODY WRITES THE PROOF (Kaveh 2026-10-27): a proof text enters a document only
as the text the harness's tools produced, and only after the harness's own acceptance check (`prover.accept`: no error, no `sorry`, the three standard axioms, no enumeration) passed it
inside the whole generated file. A text Lean rejects leaves the document, the obligation and the standing as they were.

WHAT THE HARNESS NEEDED to take an obligation of this kind (all in prover.py, general): (1) `prove` takes the path of a .lean file as well as a module of Ideas/Mined (a scratch copy of an
obligation, read and never written); (2) the model is shown the DEFINITIONS the statement rests on, read from the library modules the file imports (`spec_context`: the obligation names
the description's relations and holds none of them); (3) a hint can be NAMES of library theorems the index holds (`--hints`): their statements go to the model and the automation is handed
the names (`simp_all [..]`, `grind [..]`, `aesop (add unsafe apply [..])`); (4) a tool whose scratch project is not installed (Duper, LeanCopilot) is reported not installed and skipped.

INTERFACE (who calls it: the tests; `python -m catspace.incubator.analyst.discovery.rule_lemma_prove ...`):
  `open_copy(lemma, dest) -> Path`                       the obligation WITHOUT its proof, written at `dest` (a scratch file: what `prover prove <dest>` reads)
  `candidate(lemma, proof_lines, uses) -> RL.Lemma`      the lemma with that proof and the PROVED standing claimed (in memory)
  `uses_of(proof_text) -> list`                          the theorems of the index (not OPEN) the proof text names
  `try_proof(lemma, proof_text, uses=None) -> dict`      {accepted, why, uses, text, secs}: the whole generated file checked by the harness; nothing written
  `adopt(lemma, proof_text, run="", tool="", uses=None) -> dict`
        try_proof; when accepted, the document gets `lean.proof`, `lean.uses` and the standing PROVED, the obligation is written again from it, the index is rebuilt, and the standing is
        computed again by `rule_lemma_certify.standing`; when that is not PROVED, or the check rejected, the document and the obligation are put back as they were.
        -> {adopted, accepted, why, standing, uses}
  `adopt_summary(game, name, summary_path) -> dict`      `adopt` with the proof a prover run's summary.json holds for the lemma's goal (a run with helper lemmas is refused: the
        document holds a proof, not helpers)
  `python -m ...rule_lemma_prove open <game> <name> <dest>` | `adopt <game> <name> <summary.json>` | `status <game>`
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from catspace.incubator.analyst.discovery import prover as PR
from catspace.incubator.analyst.discovery import rule_lemma as RL
from catspace.incubator.analyst.discovery import rule_lemma_lean as RLL
from catspace.io import paths

ROOT = RL.ROOT
IDENT = re.compile(r"[A-Za-z_][\w'.!?]*")


def open_copy(lemma: RL.Lemma, dest: "str | Path") -> Path:
    """The obligation as the document would write it with NO proof (`sorry`), at `dest`: the file the harness is run on (the repo's own obligation is never the run's file)."""
    open_lemma = candidate(lemma, [], [])
    open_lemma.standing = {"is": "CONJECTURE"}
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(RLL.obligation(open_lemma))
    return dest


def candidate(lemma: RL.Lemma, proof_lines: list, uses: list) -> RL.Lemma:
    """A copy of the lemma carrying the proof, its uses and the PROVED standing claimed (the theorem the obligation states)."""
    import copy
    c = copy.deepcopy(lemma)
    c.lean = dict(c.lean)
    c.lean.pop("proof", None)
    c.lean.pop("uses", None)
    if proof_lines:
        c.lean["proof"] = list(proof_lines)
        c.lean["uses"] = list(uses)
    c.standing = {"is": "PROVED", "theorem": RL.generated_theorem(c)}
    return c


def _index_theorems() -> dict:
    """{full name: decl} of the theorems of the index that are not OPEN (the derived json of lean_index; built when absent)."""
    from catspace.incubator.analyst import lean_index as LI
    p = Path(paths.derived("lean_index.json"))
    decls = json.load(open(p)) if p.exists() else LI.build()
    return {d["full"]: d for d in decls if d["kind"] in ("theorem", "lemma") and not d["open"]}


def uses_of(proof_text: str) -> list:
    """The theorems of the index the proof text names: a name written in full, or by its tail where exactly one theorem of the index ends so (the namespaces the file opens
    are not resolved: a tail that two theorems share is not claimed). The obligation's own theorem is never one."""
    thms = _index_theorems()
    tails: dict = {}
    for full in thms:
        if full.startswith("Ideas.Mined.rule_lemma_"):
            continue
        tails.setdefault(full.split(".")[-1], []).append(full)
    out = []
    clean = re.sub(r"--[^\n]*", " ", proof_text)
    for tok in IDENT.findall(clean):
        tok = tok.rstrip(".")
        if tok in thms and tok not in out:
            out.append(tok)
            continue
        hits = [f for f in thms if f.endswith("." + tok) and not f.startswith("Ideas.Mined.rule_lemma_")]
        if len(hits) == 1 and hits[0] not in out:
            out.append(hits[0])
    return out


def proof_lines_of(proof_text: str) -> list:
    return textwrap.dedent(proof_text).strip("\n").splitlines()


def try_proof(lemma: RL.Lemma, proof_text: str, uses: "list | None" = None) -> dict:
    """The harness's acceptance check on the WHOLE generated file with this proof (compiled in a scratch copy, nothing written)."""
    uses = uses_of(proof_text) if uses is None else list(uses)
    cand = candidate(lemma, proof_lines_of(proof_text), uses)
    text = RLL.obligation(cand)
    name = RL.generated_theorem(cand)
    chk = PR.lean_check(text, (name,), "ideas", tag=f"lemma_{lemma.name}")
    ok, why = PR.accept(chk, (name,), (proof_text,))
    return dict(accepted=ok, why=why, uses=uses, text=text, secs=chk.secs, errors=chk.errors[:3])


def rebuild_index() -> int:
    r = subprocess.run([sys.executable, "-m", "catspace.incubator.analyst.lean_index", "build"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode


def standing_now(game: str, name: str) -> tuple:
    from catspace.incubator.analyst.discovery import rule_lemma_certify as RC
    return RC.standing(RL.load(game, name))


def adopt(lemma: RL.Lemma, proof_text: str, run: str = "", tool: str = "", uses: "list | None" = None) -> dict:
    """See the module text: the document is changed only for a proof the harness accepts, and only kept when the standing computed again reads PROVED."""
    t = try_proof(lemma, proof_text, uses)
    res = dict(adopted=False, accepted=t["accepted"], why=t["why"], uses=t["uses"], standing=None, errors=t["errors"])
    if not t["accepted"]:
        res["standing"] = standing_now(lemma.game, lemma.name)
        return res
    doc_path = RL.lemma_dir(lemma.game) / f"{lemma.name}.json"
    ob_path = RL.obligation_path(lemma)
    before = (doc_path.read_text(), ob_path.read_text() if ob_path.is_file() else None)
    cand = candidate(lemma, proof_lines_of(proof_text), t["uses"])
    RL.write(cand)
    RLL.write(cand)
    rebuild_index()
    word = standing_now(lemma.game, lemma.name)
    res["standing"] = word
    if word[0] == "PROVED":
        res["adopted"] = True
        return res
    doc_path.write_text(before[0])
    if before[1] is not None:
        ob_path.write_text(before[1])
    rebuild_index()
    res["why"] = [f"the standing computed again reads {word[0]}: {word[1]}"]
    res["standing"] = standing_now(lemma.game, lemma.name)
    return res


def adopt_summary(game: str, name: str, summary_path: str) -> dict:
    s = json.load(open(summary_path))
    lemma = RL.load(game, name)
    g = next((v for k, v in s["goals"].items() if k.endswith(f"{lemma.name}_{lemma.form.lower()}")), None)
    if g is None or not g.get("closed_by"):
        return dict(adopted=False, accepted=False, why=["the run did not close the lemma's goal"], standing=standing_now(game, name), uses=[])
    if (g.get("lemmas") or "").strip():
        return dict(adopted=False, accepted=False, why=["the accepted proof carries helper lemmas: the document holds a proof, not helpers (reported, not adopted)"],
                    standing=standing_now(game, name), uses=[])
    return adopt(lemma, g["proof"], run=s.get("run", ""), tool=g["closed_by"])


def main(argv: list) -> int:
    if len(argv) >= 4 and argv[0] == "open":
        print(f"OBS lemma open copy {argv[1]} {argv[2]}: {open_copy(RL.load(argv[1], argv[2]), argv[3])}")
        return 0
    if len(argv) == 4 and argv[0] == "adopt":
        r = adopt_summary(argv[1], argv[2], argv[3])
        print(f"OBS lemma adopt {argv[1]} {argv[2]}: " + ("ADOPTED" if r["adopted"] else "NOT adopted") + f"; accepted {r['accepted']}; why {r['why']}; standing {r['standing']}; uses {r['uses']}")
        return 0 if r["adopted"] else 1
    if len(argv) == 2 and argv[0] == "status":
        for lemma in RL.load_all(argv[1]):
            w = standing_now(argv[1], lemma.name)
            print(f"OBS lemma standing {argv[1]} {lemma.name}: {w[0]} ({w[1]})")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
