"""PLAN 73.6 (3) THE PROVER HARNESS: off-the-shelf tools SEARCH for the proof of a statement the engine emitted (technique_lean.py); Lean checks every candidate;
nobody writes the proof by hand. The tools, cheapest first, on every `sorry` (a GOAL), each candidate compiled in a SCRATCH copy of the file (never the file):

  (a) LEAN'S OWN AUTOMATION -- `aesop`, `simp_all`, `omega`, `grind`, `tauto`, `exact?`, each as it is and after `intros`; and DUPER (leanprover-community/duper
      v4.32.0, with lean-auto at the commit it pins: a release for Lean 4.32 whose manifest pins the very batteries revision of lean/ideas, so nothing of Mathlib
      is rebuilt) -- `duper [*]`, in the scratch tool project (artifacts/experiments/prover/lean_tools, its own .lake, the shared packages linked read-only);
  (b) LEANCOPILOT v4.32.0 (lean-dojo; its batteries and aesop revisions are lean/ideas's) -- `search_proof` (the ByT5 tactic generator ct2-leandojo-lean4-tacgen-
      byt5-small, best-first search with aesop), in the same tool project;
  (c) AN OPEN-WEIGHT PROVER run locally through ollama -- Goedel-Prover-V2-8B (Goedel-LM, 2025; the GGUF of mradermacher, Q8_0, 8.7 GB) -- in an AGENTIC
      LOOP: WHOLE-PROOF attempts in the model's own prompt format, Lean's errors fed back (a chain ends when Lean's first error repeats: the model no longer
      moves), and SUBGOAL DECOMPOSITION (the DeepSeek-Prover-V2 pattern): the model writes a SKETCH of `have` steps closed by `sorry`; Lean's `extract_goal`
      (Mathlib) turns each `sorry` into a standalone goal with its context; each is attacked by (a)-(c) recursively; a proved sub-lemma is kept, offered to
      every later attempt and reused; when every goal of a sketch is proved the parent is assembled from them and checked;
EVERYTHING IS LOCAL (Kaveh 2026-10-27): no network service is ever called for proving -- the model runs in the local ollama server, the tools in Lean on
this machine (the plan's Aristotle path is not built).

THE FUSE (named, per top obligation per run, wall-clock): FUSE_SECONDS = 90 minutes. Why: the time this plan has for each of its two runs (blind, hinted) in one
sitting beside the re-mine and the gate, on a machine where the model writes about nine tokens a second (the probes of 2026-10-27, journal: a sample of 4243
tokens in 464.4 s, one of 1923 in 241.0 s) -- about a dozen whole samples, the feedback rounds and one or two decompositions. Sub-goals spend the top
obligation's fuse (each its share of what is left); nothing is retried past it. TOKEN_CAP = 10240 tokens per sample; an answer cut by it ends the step it
was asked for (the loop moves on: a whole-proof chain to the sketch, a sketch to the next round). THE MODEL'S REASONING MODE IS OFF (`--think` turns it on):
the model's chat template always opens the answer with `<think>` and ollama's `think: false` only moves the text to another field (it does not change the
prompt: journal 2026-10-27), so OFF is the template itself sent raw with an EMPTY think block (RAW_NO_THINK). With the reasoning on, samples reached the cap
in 1178.0 s and 1155.2 s still reasoning -- the model card's 32k cap would make one sample over an hour here.

THE ACCEPTANCE CHECK of any proof (`accept`): the scratch copy compiles with no error and no `sorry` warning; `#print axioms` of the theorem (and of the file's
corollary) shows only propext, Classical.choice, Quot.sound (no Lean.ofReduceBool, no sorryAx, no new axiom); a syntactic scan of the proof text (everything
the tools wrote) rejects `native_decide`, `decide` (any use: the decision procedure is the enumeration the owner banned -- by reasoning, never brute force),
`fin_cases` / `interval_cases` (case splits over squares and coordinates), `Fintype` / `Finset.univ` / `List.finRange` (enumerations), `sorry` / `admit`,
`axiom`, `ofReduceBool`, `implemented_by`, `extern`, `unsafe`. A syntactic scan cannot see every enumeration (a hand-written split of 64 cases would pass it):
every accepted proof is logged in full for a reader. THE HINT (the hinted run): Kaveh's natural-language proof of 2026-10-27, quoted VERBATIM from JOURNAL.md
by code (`hint_text`), handed to the model as a sketch -- nothing is added to it.

Every attempt is logged (artifacts/experiments/prover/<run>/attempts.jsonl: the tool, the goal, the prompt, the model's output and reasoning, the candidate,
Lean's errors, the verdict, the seconds); the run's summary is <run>/summary.json and its OBS lines.

  python -m catspace.incubator.analyst.discovery.prover prove mined_technique_krvk --mode blind|hinted [--fuse 5400] [--tools automation,copilot,llm]
  python -m catspace.incubator.analyst.discovery.prover accept <scratch .lean> <theorem> ...      # the acceptance check of a file as it stands
  python -m catspace.incubator.analyst.discovery.prover setup                                     # the scratch tool project (Duper, LeanCopilot)
  python -m catspace.incubator.analyst.discovery.prover record <run>/summary.json                 # (4) the store from a run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from catspace.io import paths

LEAN_DIR = Path(__file__).resolve().parents[4] / "lean" / "ideas"
MINED_DIR = LEAN_DIR / "Ideas" / "Mined"
TOOLS_DIR = Path(paths.experiment("prover", "lean_tools"))
COPILOT_LIB = TOOLS_DIR / ".lake" / "packages" / "LeanCopilot" / ".lake" / "build" / "lib"
MODEL = "hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q8_0"
OLLAMA = "http://localhost:11434/api/generate"
FUSE_SECONDS = 90 * 60          # see the docstring: the fuse, named, per top obligation per run
TOKEN_CAP = 10240               # per sample (reasoning + code)
CONTEXT = 24576                 # the model's window for one sample (the prompt -- the file, the errors, the hint -- and the cap)
CHECK_TIMEOUT = 900             # one Lean check, seconds (a runaway tactic is Lean's heartbeats' to stop; this is the process's guard)
ALLOWED_AXIOMS = ("propext", "Classical.choice", "Quot.sound")
BANNED = ("native_decide", "decide", "fin_cases", "interval_cases", "Fintype", "Finset.univ", "List.finRange", "sorry", "admit", "axiom", "ofReduceBool",
          "implemented_by", "extern", "unsafe")
AUTO_TACTICS = ("aesop", "simp_all", "omega", "grind", "tauto", "exact?")
TOOL_TACTICS = {"automation": ("duper [*]",), "copilot": ("search_proof",)}
DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*(theorem|lemma|def|abbrev|instance|structure|inductive|example)\s+([^\s:(\[{]*)",
                  re.M)
TOP = re.compile(r"^(?:theorem|lemma|def|abbrev|instance|structure|inductive|example|end|namespace|section|open|set_option|#|@\[|noncomputable|private|protected|"
                 r"/-|--|variable|universe|import|attribute)", re.M)


# ---------------------------------------------------------------------------------------------------------------- goals and files

@dataclass
class Goal:
    """A `sorry` to fill: theorem `name` whose statement is `stmt` (the text from `theorem` to `:= by`), checked inside `prefix` (the file before it: the
    definitions; the sub-lemmas proved so far are added at every check, never kept here) and `suffix` (the file after its proof); `full` its full name;
    `depth` 0 for the file's own sorry."""
    name: str
    stmt: str
    prefix: str
    suffix: str
    full: str
    depth: int = 0
    parent: "str | None" = None
    extra_axioms: tuple = ()           # the theorems after it whose axioms are checked too (the file's corollary)

    def text(self, proof: str, lemmas: str = "") -> str:
        return self.prefix + lemmas + self.stmt + "\n" + indent(proof) + "\n" + self.suffix


def indent(proof: str, k: int = 2) -> str:
    """technique_lean.indent_proof: the check and the file place a proof the same way."""
    from catspace.incubator.analyst.discovery import technique_lean as TL
    return TL.indent_proof(proof, k)


def namespace_of(text: str) -> str:
    m = re.search(r"^namespace\s+(\S+)", text, re.M)
    return m.group(1) if m else ""


def goals_of(path: Path) -> list:
    """The file's goals: every theorem whose proof is `by sorry` (the emitted statement holds exactly one, `hprog`). Its docstring travels with it (a lemma put
    before the theorem goes before the docstring)."""
    text = path.read_text()
    ns = namespace_of(text)
    out, pos = [], 0
    chunks = []
    for kind, name, chunk in split_decls(text):
        chunks.append((kind, name, pos, chunk))
        pos += len(chunk)
    lead = len(text) - len("".join(c for _k, _n, _p, c in chunks))          # (text before the first top-level command)
    for i, (kind, name, at, chunk) in enumerate(chunks):
        at += lead
        m = re.search(r":=\s*by\s*\n\s*sorry\s*\n?\s*$", chunk)
        if kind not in ("theorem", "lemma") or m is None:
            continue
        start = at
        if i > 0 and chunks[i - 1][3].lstrip().startswith("/--"):
            start = chunks[i - 1][2] + lead
        later = [n for k, n, _p, _c in chunks[i + 1:] if k in ("theorem", "lemma") and n]
        stmt = text[start:at] + chunk[:m.start()].rstrip() + " := by"
        out.append(Goal(name=name, stmt=stmt, prefix=text[:start], suffix="\n" + text[at + len(chunk):].lstrip("\n"),
                        full=f"{ns}.{name}" if ns else name, extra_axioms=tuple(f"{ns}.{n}" for n in later)))
    return out


# ---------------------------------------------------------------------------------------------------------------- the checker

@dataclass
class Check:
    ok: bool                 # compiled with no error
    errors: list
    sorries: int
    axioms: dict             # theorem -> [axioms]
    output: str
    secs: float
    project: str
    infos: list = field(default_factory=list)


AXIOMS = re.compile(r"'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")
SORRY_WARNING = re.compile(r"declaration uses [`']sorry[`']")


MSG = re.compile(r"^(?:(error|warning|info)(?:\([\w.]+\))?: \S+\.lean:\d+:\d+: |\S+\.lean:\d+:\d+: (error|warning|info)(?:\([\w.]+\))?: )")
# (lake build's form | lake env lean's form; Lean 4.32 may name the message's code: `error(lean.unknownIdentifier): `)
STATUS = re.compile(r"^(?:✔|⚠|ℹ|✖|trace:|Some required|- |Build completed|All targets|error: Lean exited|error: build failed|error: \S+: |'[^']+' (?:depends on axioms|does not depend))")


def _errors(out: str) -> list:
    """The error messages of Lean's output (both the `lake env lean` and the `lake build` forms), each with its continuation lines."""
    msgs, cur = [], None
    for ln in out.splitlines():
        m = MSG.match(ln)
        if m:
            if cur:
                msgs.append("\n".join(cur))
            cur = [ln] if (m.group(1) or m.group(2)) == "error" else None
        elif STATUS.match(ln):
            if cur:
                msgs.append("\n".join(cur))
            cur = None
            if ln.startswith("error: ") and not re.match(r"^error: (?:Lean exited|build failed)", ln):
                msgs.append(ln)                                     # (a build error with no position: a library that does not load, a timeout)
        elif cur is not None:
            cur.append(ln)
        elif ln.startswith("error: the check ran past"):
            msgs.append(ln)
    if cur:
        msgs.append("\n".join(cur))
    return msgs


def lean_check(text: str, names: tuple, project: str = "ideas", tag: str = "attempt", imports: tuple = ()) -> Check:
    """Compile `text` (a scratch copy) with `#print axioms` of `names` appended -> Check. project 'ideas': `lake env lean` in lean/ideas; 'tools': a module of the
    scratch tool project built by `lake build` (Duper / LeanCopilot are precompiled libraries; LeanCopilot's ctranslate2 is found through DYLD_LIBRARY_PATH)."""
    body = "".join(f"import {i}\n" for i in imports) + text + "\n" + "".join(f"#print axioms {n}\n" for n in names)
    tag = f"{tag}_p{os.getpid()}"                                       # (one scratch file per process: two runs never share one)
    t0 = time.monotonic()
    if project == "ideas":
        scratch = Path(paths.experiment("prover", "scratch", f"{tag}.lean"))
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_text(body)
        try:
            r = subprocess.run(["lake", "env", "lean", str(scratch)], cwd=LEAN_DIR, capture_output=True, text=True, timeout=CHECK_TIMEOUT)
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            out = f"error: the check ran past {CHECK_TIMEOUT} s (killed)"
    else:
        mod = re.sub(r"\W", "_", tag)
        src = TOOLS_DIR / "ProverTools" / f"{mod}.lean"
        src.write_text(body)
        env = dict(os.environ, DYLD_LIBRARY_PATH=str(COPILOT_LIB))
        try:
            r = subprocess.run(["lake", "build", f"ProverTools.{mod}"], cwd=TOOLS_DIR, capture_output=True, text=True, timeout=CHECK_TIMEOUT, env=env)
            out = r.stdout + r.stderr
            k = out.find(f"ProverTools.{mod}")                    # (the replayed logs of the tool packages' own builds come first: dropped)
            out = out[k:] if k >= 0 else out
        except subprocess.TimeoutExpired:
            out = f"error: the check ran past {CHECK_TIMEOUT} s (killed)"
    errs = _errors(out)
    ax = {m.group(1): [a.strip() for a in (m.group(2) or "").split(",") if a.strip()] for m in AXIOMS.finditer(out)}
    infos = [m.group(0) for m in re.finditer(r"theorem \S*?extracted_\w+[\s\S]*?:= sorry", out)]
    return Check(ok=not errs, errors=errs, sorries=len(SORRY_WARNING.findall(out)), axioms=ax, output=out[-20000:], secs=round(time.monotonic() - t0, 1),
                 project=project, infos=infos)


def scan(proof: str) -> list:
    """THE SYNTACTIC SCAN of a proof text (see the docstring) -> the banned words found."""
    clean = re.sub(r"--[^\n]*|/-[\s\S]*?-/", " ", proof)
    return sorted({w for w in BANNED if re.search(r"(?<![\w.])" + re.escape(w) + r"(?![\w])", clean)})


def accept(chk: Check, names: tuple, proof_texts: tuple) -> tuple:
    """THE ACCEPTANCE CHECK (see the docstring) -> (accepted, reasons)."""
    why = []
    if not chk.ok:
        why.append(f"Lean errors: {len(chk.errors)}")
    if chk.sorries:
        why.append(f"sorry warnings: {chk.sorries}")
    for n in names:
        if n not in chk.axioms:
            why.append(f"no axioms line for {n}")
            continue
        bad = [a for a in chk.axioms[n] if a not in ALLOWED_AXIOMS]
        if bad:
            why.append(f"{n} depends on {', '.join(bad)}")
    found = sorted({w for p in proof_texts for w in scan(p)})
    if found:
        why.append("banned: " + ", ".join(found))
    return (not why, why)


# ---------------------------------------------------------------------------------------------------------------- the log

class Log:
    """The run's attempts, one JSON line each; `on_write(row)` is called after every attempt (the demonstration's live transcript, prover_demo.py)."""

    def __init__(self, run_dir: Path, on_write=None):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "attempts.jsonl"
        self.n = 0
        self.on_write = on_write

    def write(self, **row) -> None:
        self.n += 1
        row = dict(i=self.n, t=time.strftime("%H:%M:%S"), **row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if self.on_write is not None:
            self.on_write(dict(row, run=self.dir.name))


def first_error(chk: Check) -> str:
    e = chk.errors[0] if chk.errors else ""
    e = re.sub(r"^\S+\.lean:\d+:\d+: ", "", e)
    return e[:600]


# ---------------------------------------------------------------------------------------------------------------- (a), (b): tactics

def tools_ready() -> bool:
    """The scratch tool project (`setup`) exists: Duper and LeanCopilot can be asked. Where it does not, those two are reported as not installed and skipped (never
    installed here by a run: `setup` is its own command); Lean's own automation and the model do not need it."""
    return (TOOLS_DIR / "lakefile.toml").is_file() and (TOOLS_DIR / "ProverTools").is_dir()


def hint_tactics(hints: tuple) -> tuple:
    """The tactics that take the NAMES of hinted theorems (the index's own theorems, found by `lean_index search`): Lean's automation is handed the names; nothing is
    written by hand."""
    if not hints:
        return ()
    names = ", ".join(hints)
    return (f"simp_all [{names}]", f"grind [{names}]", f"aesop (add unsafe apply [{names}])")


def try_tactics(goal: Goal, tool: str, log: Log, deadline: float, lemmas: str = "", last: "list | None" = None, hints: tuple = ()) -> "dict | None":
    """(a) / (b): each tactic of the tool on the goal, as it is and after `intros` -> the first accepted {tool, proof, check}, else None (every try logged)."""
    if tool == "automation":
        tacs = [(t, "ideas") for t in AUTO_TACTICS + hint_tactics(hints)] + [(t, "tools") for t in TOOL_TACTICS["automation"]]
    else:
        tacs = [(t, "tools") for t in TOOL_TACTICS[tool]]
    names = (goal.full,) + goal.extra_axioms
    last = last if last is not None else [None]
    for t, proj in tacs:
        if proj == "tools" and not tools_ready():
            log.write(tool=tool, tactic=t, goal=goal.name, depth=goal.depth, accepted=False, why=["not installed: the scratch tool project (`prover setup`) does not exist"],
                      secs=0, project=proj)
            last[0] = last[0] or f"{tool} {t!r}: not installed"
            continue
        for proof in (t, f"intros\n{t}"):
            if time.monotonic() > deadline:
                return None
            imports = ("Duper",) if t.startswith("duper") else ("LeanCopilot",) if t.startswith("search_proof") else ()
            chk = lean_check(goal.text(proof, lemmas), names, proj, tag=f"{tool}_{goal.name}", imports=imports)
            ok, why = accept(chk, names, (proof,))
            log.write(tool=tool, tactic=t, goal=goal.name, depth=goal.depth, proof=proof, accepted=ok, why=why, errors=chk.errors[:3], secs=chk.secs, project=proj)
            if ok:
                return dict(tool=f"{tool}: {t}", proof=proof, check=chk)
            last[0] = f"{tool} {t!r}: " + (first_error(chk) or "; ".join(why))
    return None


# ---------------------------------------------------------------------------------------------------------------- (c): the open-weight prover

RAW_NO_THINK = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"     # the model's own chat template (ollama show --template)


def generate(prompt: str, seed: int, think: bool = False) -> dict:
    """One sample of the local model (ollama) -> {response, thinking, tokens, secs, prompt_tokens}."""
    opts = dict(num_ctx=CONTEXT, num_predict=TOKEN_CAP, temperature=0.6, top_p=0.95, seed=seed)
    if think:
        body = dict(model=MODEL, prompt=prompt, stream=False, think=True, options=opts)
    else:                                      # the model's template always opens the answer with <think>; `think: false` only moves the text (journal
        body = dict(model=MODEL, prompt=RAW_NO_THINK.format(prompt=prompt), raw=True, stream=False, options=opts)   # 2026-10-27): an empty think block, raw
    t0 = time.monotonic()
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    except Exception as x:                                      # (the server's failure is the attempt's result, logged)
        return dict(response="", thinking="", tokens=0, secs=round(time.monotonic() - t0, 1), prompt_tokens=0, error=str(x))
    return dict(response=r.get("response") or "", thinking=r.get("thinking") or "", tokens=r.get("eval_count", 0), prompt_tokens=r.get("prompt_eval_count", 0),
                secs=round(time.monotonic() - t0, 1), done_reason=r.get("done_reason"))


def code_block(text: str) -> "str | None":
    """The last Lean code block of a model's answer."""
    blocks = re.findall(r"```(?:lean4|lean)?[ \t]*\n([\s\S]*?)```", text)
    return blocks[-1] if blocks else None


def split_decls(code: str) -> list:
    """Top-level declarations of a code text -> [(kind, name, text)] (each to the next top-level command)."""
    starts = [m.start() for m in TOP.finditer(code)] + [len(code)]
    out = []
    for a, b in zip(starts, starts[1:]):
        chunk = code[a:b]
        m = DECL.match(chunk)
        out.append((m.group(1), m.group(2), chunk) if m else ("other", "", chunk))
    return out


def take_proof(code: str, goal: Goal, known: set) -> "tuple | None":
    """From the model's code: (its proof of the goal -- the body after the goal's `:= by` --, its NEW declarations before it, as text). The statement and the
    definitions are always OURS (the model's copy of them is dropped); a new declaration whose name is ours is dropped too."""
    lemmas, proof = [], None
    for kind, name, chunk in split_decls(code):
        short = name.split(".")[-1]
        if kind in ("theorem", "lemma") and short == goal.name:
            m = re.search(r":=\s*by\b", chunk)
            if m is None:
                m2 = re.search(r":=", chunk)
                if m2 is None:
                    return None
                proof = "exact (" + chunk[m2.end():].strip() + ")"
            else:
                proof = chunk[m.end():].rstrip()
            break
        if kind in ("theorem", "lemma", "def", "abbrev") and short not in known:
            lemmas.append(re.sub(r"^lemma\b", "theorem", chunk.rstrip()) + "\n\n")     # (`lemma` is Mathlib's command: written as `theorem`)
    if proof is None or not proof.strip():
        return None
    return proof, "".join(lemmas)


WHOLE = ("{context}Complete the following Lean 4 code:\n\n```lean4\n{code}```\n\n"
         "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\n"
         "The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.\n\n"
         "{rules}{hint}{known}")
RULES = ("Rules for the proof: replace the `sorry` of `{name}` with a complete proof; keep every definition and statement exactly as given; you may add helper "
         "lemmas before `{name}`. The proof must REASON: do not use native_decide, decide, fin_cases, interval_cases, Fintype or Finset.univ enumeration, "
         "sorry, admit or new axioms -- a proof that checks positions or squares one by one is rejected.\n\n")
REVISE = ("{whole}\n\nA previous attempt:\n```lean4\n{attempt}```\n\nLean 4 rejected it with these errors:\n```\n{errors}\n```\n\n"
          "Revise the proof. Answer with the complete corrected Lean 4 code in one ```lean4 block.")
SKETCH = ("{context}Here is a Lean 4 file whose theorem `{name}` is not yet proved:\n\n```lean4\n{code}```\n\n{hint}{known}"
          "Do not prove it in one go. Write a PROOF SKETCH of `{name}`: a structured Lean 4 proof that states the key intermediate facts as `have` steps, EACH "
          "closed by `sorry`, and then closes the goal from those facts with no further `sorry`. Name every hypothesis you introduce. Each `have` should be a "
          "meaningful lemma about the definitions above that a later proof can attack on its own. Answer with the complete Lean 4 code of `{name}` in one "
          "```lean4 block.")
SKETCH_REVISE = ("{sketch}\n\nA previous sketch:\n```lean4\n{attempt}```\n\nLean 4 rejected it (apart from the `sorry`s, which are expected) with:\n```\n{errors}\n```\n\n"
                 "Revise the sketch. Answer with the complete Lean 4 code of `{name}` in one ```lean4 block.")


def context_of(path: Path) -> tuple:
    """The DEFINITIONS a file imports from other mined modules (`import Ideas.Mined.<m>`: the text of <m> before its first theorem), shown to the model as
    context, and their names (a model's copy of one is dropped) -> (the context block, the names)."""
    blocks, names = [], set()
    for m in re.findall(r"^import Ideas\.Mined\.(\S+)", path.read_text(), re.M):
        src = (MINED_DIR / f"{m}.lean").read_text()
        chunks = split_decls(src)
        lead = len(src) - len("".join(c for _k, _n, c in chunks))
        keep = []
        for k, _n, c in chunks:
            if k in ("theorem", "lemma"):
                if keep and keep[-1].lstrip().startswith("/--"):
                    keep.pop()                                         # (the theorem's docstring goes with it)
                break
            keep.append(c)
        defs = src[:lead] + "".join(keep)
        names |= {n for k, n, _c in split_decls(defs) if n}
        blocks.append(f"The definitions it uses, from the imported module Ideas.Mined.{m}:\n\n```lean4\n{defs.rstrip()}\n```\n\n")
    return "".join(blocks), names


def hint_text() -> str:
    """THE HINT: Kaveh's natural-language proof, quoted verbatim from JOURNAL.md's entry of 2026-10-27 (the passage after 'Kaveh, verbatim:', through the
    measure he named) -- extracted by code, nothing added."""
    j = (Path(paths.REPO_ROOT) / "JOURNAL.md").read_text()
    i = j.index("## 2026-10-27 -- Kaveh on the mined K+R technique")
    seg = j[i:]
    a = seg.index('Kaveh, verbatim: "') + len('Kaveh, verbatim: ')
    b = seg.index('Then: "i don\'t want you to write this')
    text = " ".join(seg[a:b].split())
    return text.rstrip()


SPEC_BUDGET = 20000     # characters of a spec module's definitions shown to the model (about a third of its prompt window)
NAME = re.compile(r"[A-Za-z_][\w']*")


def spec_context(path: Path) -> str:
    """The DEFINITIONS a file's statement rests on, from the library modules it imports (`import Ideas.<m>` other than a mined one): the declarations (inductive, def,
    structure, abbrev) whose names the theorem's statement uses, then those THEIR text uses, breadth first, until SPEC_BUDGET characters -- shown to the model as context
    (a rule lemma's obligation names the description's relations and holds none of them). Game-free: it reads names, not a game."""
    text = path.read_text()
    mods = [m for m in re.findall(r"^import (Ideas\.(?!Mined\.)\S+)", text, re.M)]
    decls: dict = {}
    for m in mods:
        f = LEAN_DIR / (m.replace(".", "/") + ".lean")
        if not f.is_file():
            continue
        src = f.read_text()
        chunks = split_decls(src)
        for i, (k, n, c) in enumerate(chunks):
            if k in ("inductive", "def", "abbrev", "structure") and n and n.split(".")[-1] not in decls:
                doc = chunks[i - 1][2] if i > 0 and chunks[i - 1][2].lstrip().startswith("/--") else ""
                decls[n.split(".")[-1]] = (m, doc + c)
    goals = [g for g in goals_of(path)]
    seed = " ".join(g.stmt for g in goals) or text
    out, seen, queue, used = [], set(), [w for w in NAME.findall(seed) if w in decls], 0
    queue = list(dict.fromkeys(queue))
    while queue:
        n = queue.pop(0)
        if n in seen:
            continue
        seen.add(n)
        mod, body = decls[n]
        if used + len(body) > SPEC_BUDGET:
            continue
        used += len(body)
        out.append(body.rstrip())
        queue += [w for w in NAME.findall(body) if w in decls and w not in seen]
    if not out:
        return ""
    return "The definitions its statement rests on, from the imported library modules (in the order they are reached):\n\n```lean4\n" + "\n\n".join(out) + "\n```\n\n"


def theorem_statement(full: str) -> str:
    """The statement (docstring and signature, to `:=`) of the library theorem `full`, read from the file the index names (docs/LEAN_INDEX.md's derived json)."""
    from catspace.incubator.analyst import lean_index as LI
    p = Path(paths.derived("lean_index.json"))
    decls = json.load(open(p)) if p.exists() else LI.build()
    d = next((x for x in decls if x["full"] == full and x["kind"] in ("theorem", "lemma")), None)
    if d is None:
        raise SystemExit(f"hint {full}: not a theorem of the index (python -m catspace.incubator.analyst.lean_index search <words>)")
    lines = (Path(paths.REPO_ROOT) / d["file"]).read_text().splitlines()
    i = d["line"] - 1
    j = i
    while j < len(lines) and ":=" not in lines[j]:
        j += 1
    sig = "\n".join(lines[i:j + 1]).split(":=")[0].strip()
    return f"-- {full}\n" + (f"/-- {d['doc'][:600]} -/\n" if d["doc"] else "") + sig


def _hint_block(mode: str, hints: tuple = ()) -> str:
    if mode != "hinted":
        return ""
    if hints:                     # the hinted run of a rule lemma: names of theorems the index holds, with their statements; nothing else
        return ("These theorems of the library are proved and may be used by name (their full names and statements):\n```lean4\n"
                + "\n\n".join(theorem_statement(h) for h in hints) + "\n```\n\n")
    return ("A natural-language proof by a human (a proof sketch to follow; the formal statement above is what must be proved):\n\"\"\"\n" + hint_text()
            + "\n\"\"\"\n\n")


def _known_block(proved: dict) -> str:
    if not proved:
        return ""
    return ("These lemmas are already proved and may be used (they will be placed before the theorem):\n```lean4\n" + "".join(v["text"] for v in proved.values())
            + "```\n\n")


class Prover:
    """The agentic loop of (c) with (a), (b) inside it, over one top goal, under the fuse."""

    def __init__(self, path: Path, mode: str, tools: tuple, log: Log, fuse: float, think: bool = False, hints: tuple = ()):
        self.path, self.mode, self.tools, self.log, self.think, self.hints = path, mode, tools, log, think, tuple(hints)
        self.t0 = time.monotonic()
        self.deadline = self.t0 + fuse
        self.proved: dict = {}           # sub-lemma name -> {text, statement, proof, tool, parent}
        self.open: dict = {}             # sub-goal name -> {statement, last}
        self.proposed: list = []         # every sub-goal proposed: (name, parent, statement)
        self.samples = 0
        self.seed = 1
        self.nsub = 0                    # sub-goals named so far
        self.top = ""                    # the top goal's name (it prefixes the sub-goals' names)
        self.checked = 0                 # the model's candidates and sketches Lean has checked (a round adding none ends the goal's loop)
        self.capped = 0                  # the samples cut by the token cap (unfinished answers: they do not end the loop)
        self.context, imported = context_of(path)
        self.context += spec_context(path)
        self.known = {n for _k, n, _c in split_decls(path.read_text()) if n} | imported
        self.results: dict = {}          # goal name -> {closed_by, last}
        self.share: list = []            # the deadlines of the sub-goals being attacked (each its share of what was left)

    def left(self) -> float:
        return min([self.deadline] + self.share) - time.monotonic()

    def lemmas(self) -> str:
        return "".join(v["text"] for v in self.proved.values())

    # -- a goal, every tool in order
    def attack(self, goal: Goal) -> "dict | None":
        res = None
        last = [None]
        for tool in ("automation", "copilot"):
            if tool in self.tools and self.left() > 0:
                res = try_tactics(goal, tool, self.log, time.monotonic() + self.left(), self.lemmas(), last, self.hints)
                if res:
                    break
        if res is None and last[0]:
            self.results.setdefault(goal.name, {})["last"] = last[0]
        if res is None and "llm" in self.tools:
            res = self.llm(goal)
        self.results[goal.name] = dict(closed_by=res["tool"] if res else None, last=self.results.get(goal.name, {}).get("last"))
        return res

    def check(self, goal: Goal, proof: str, lemmas: str, tool: str, extra: dict) -> tuple:
        names = (goal.full,) + goal.extra_axioms
        self.checked += 1
        chk = lean_check(goal.text(proof, self.lemmas() + lemmas), names, "ideas", tag=f"llm_{goal.name}")
        ok, why = accept(chk, names, (proof, lemmas))
        self.log.write(tool=tool, goal=goal.name, depth=goal.depth, proof=proof, lemmas=lemmas, accepted=ok, why=why, errors=chk.errors[:4], secs=chk.secs, **extra)
        if not ok:
            self.results.setdefault(goal.name, {})["last"] = first_error(chk) or "; ".join(why)
        return ok, chk, why

    def sample(self, prompt: str) -> dict:
        self.samples += 1
        self.seed += 1
        out = generate(prompt, self.seed, self.think)
        self.capped += out.get("done_reason") == "length"
        return out

    # -- (c) whole-proof chains, then decomposition, until the fuse
    def llm(self, goal: Goal) -> "dict | None":
        code = goal.prefix + self.lemmas() + goal.stmt + "\n  sorry\n" + goal.suffix
        whole = WHOLE.format(context=self.context, code=code, rules=RULES.format(name=goal.name), hint=_hint_block(self.mode, self.hints), known=_known_block(self.proved))
        while self.left() > 0:
            c0, k0 = self.checked, self.capped
            res = self.chain(goal, whole)
            if res:
                return res
            if self.left() <= 0:
                break
            res = self.decompose(goal, code)
            if res:
                return res
            if self.checked == c0 and self.capped == k0:              # (a round gave Lean nothing to check and no answer was cut: the model does not address it)
                self.results.setdefault(goal.name, {})["last"] = (self.results.get(goal.name, {}).get("last") or "") + " [no candidate in a whole round: stopped]"
                return None
        return None

    def chain(self, goal: Goal, whole: str) -> "dict | None":
        """Whole-proof attempts with Lean's errors fed back; the chain ends when Lean's first error repeats (the model no longer moves) or the fuse."""
        prompt, last_err = whole, None
        while self.left() > 0:
            out = self.sample(prompt)
            got = take_proof(code_block(out["response"]) or "", goal, self.known | set(self.proved))
            extra = dict(prompt=prompt, response=out["response"], thinking=out["thinking"], tokens=out["tokens"], gen_secs=out["secs"], mode=self.mode,
                         done_reason=out.get("done_reason"))
            if got is None:
                self.log.write(tool="llm whole", goal=goal.name, depth=goal.depth, accepted=False, why=["no proof of the goal in the answer"], **extra)
                self.results.setdefault(goal.name, {})["last"] = "the model's answer held no proof of the goal (" + (out.get("done_reason") or "?") + ")"
                return None                                        # (no proof, or cut by the token cap: the loop moves on to the sketch)
            proof, lemmas = got
            ok, chk, why = self.check(goal, proof, lemmas, "llm whole", extra)
            if ok:
                self.keep_lemmas(lemmas, goal)
                return dict(tool="llm whole-proof (Goedel-Prover-V2-8B)", proof=proof, lemmas=lemmas, check=chk)
            err = first_error(chk) or "; ".join(why)
            if err == last_err:
                return None
            last_err = err
            prompt = REVISE.format(whole=whole, attempt=lemmas + goal.stmt + "\n" + indent(proof) + "\n", errors="\n\n".join(chk.errors[:4])[:6000] or "; ".join(why))
        return None

    def keep_lemmas(self, lemmas: str, goal: Goal) -> None:
        for kind, name, chunk in split_decls(lemmas):
            if kind in ("theorem", "lemma") and name:
                self.proved[name.split(".")[-1]] = dict(text=chunk.rstrip() + "\n\n", statement=chunk.split(":= by")[0].split(":=")[0].strip(), proof=chunk,
                                                        tool="llm (helper lemma of an accepted proof)", parent=goal.name)

    def decompose(self, goal: Goal, code: str) -> "dict | None":
        """SUBGOAL DECOMPOSITION: a sketch of `have` steps closed by sorry; Lean's extract_goal states each; each attacked recursively; the parent assembled."""
        base = SKETCH.format(context=self.context, name=goal.name, code=code, hint=_hint_block(self.mode, self.hints), known=_known_block(self.proved))
        prompt, last_err = base, None
        while self.left() > 0:
            out = self.sample(prompt)
            got = take_proof(code_block(out["response"]) or "", goal, self.known | set(self.proved))
            extra = dict(prompt=prompt, response=out["response"], thinking=out["thinking"], tokens=out["tokens"], gen_secs=out["secs"], mode=self.mode,
                         done_reason=out.get("done_reason"))
            if got is None:
                self.log.write(tool="llm sketch", goal=goal.name, depth=goal.depth, accepted=False, why=["no sketch of the goal in the answer"], **extra)
                self.results.setdefault(goal.name, {})["last"] = "the model's answer held no sketch of the goal (" + (out.get("done_reason") or "?") + ")"
                return None
            sketch, lemmas = got
            if not re.search(r"\bsorry\b", sketch):                  # (a whole proof after all: checked as one)
                ok, chk, why = self.check(goal, sketch, lemmas, "llm sketch (no sorry: a whole proof)", extra)
                if ok:
                    return dict(tool="llm whole-proof (Goedel-Prover-V2-8B)", proof=sketch, lemmas=lemmas, check=chk)
                return None
            probe = fill_sorries(sketch, lambda k, form: {"by": "by extract_goal; sorry", "tactic": "extract_goal; sorry"}.get(form, "(by extract_goal; sorry)"))
            names = (goal.full,)
            self.checked += 1
            chk = lean_check("import Mathlib.Tactic.ExtractGoal\n" + goal.text(probe, self.lemmas() + lemmas), names, "ideas", tag=f"sketch_{goal.name}")
            self.log.write(tool="llm sketch", goal=goal.name, depth=goal.depth, proof=sketch, lemmas=lemmas, accepted=False, errors=chk.errors[:4],
                           subgoals=len(chk.infos), secs=chk.secs, **extra)
            if chk.errors or not chk.infos:
                err = first_error(chk) or "the sketch stated no sub-goal"
                self.results.setdefault(goal.name, {})["last"] = "sketch: " + err
                if err == last_err:
                    return None
                last_err = err
                prompt = SKETCH_REVISE.format(sketch=base, attempt=lemmas + goal.stmt + "\n" + indent(sketch) + "\n", errors="\n\n".join(chk.errors[:4])[:6000],
                                              name=goal.name)
                continue
            again = self.restating(goal, lemmas, chk.infos)
            if again:
                err = f"the sketch's sub-goal {again} restates the goal (the goal follows from it by `apply` and the goal's own hypotheses): no progress"
                self.results.setdefault(goal.name, {})["last"] = "sketch: " + err
                self.log.write(tool="llm sketch refused", goal=goal.name, depth=goal.depth, accepted=False, why=[err])
                if err == last_err:
                    return None
                last_err = err
                prompt = SKETCH_REVISE.format(sketch=base, attempt=lemmas + goal.stmt + "\n" + indent(sketch) + "\n", errors=err, name=goal.name)
                continue
            return self.solve_sketch(goal, sketch, lemmas, chk.infos)
        return None

    def restating(self, goal: Goal, lemmas: str, infos: list) -> "str | None":
        """NO PROGRESS (Lean decides): the first sub-goal of a sketch from which the goal follows by `apply` and the goal's own hypotheses -- one no easier
        than the goal (the model wrote the goal again as a `have`) -- or None."""
        target = closed_prop(goal.stmt)
        for k, info in enumerate(infos, 1):
            sub = closed_prop(re.sub(r":= sorry$", ":= by", info.strip()).replace("✝", "_d"))
            if not sub or not target:
                continue
            ns = namespace_of(goal.prefix)                              # (the goal's own suffix may name the goal -- the corollary does: only close the namespace)
            test = (goal.prefix + self.lemmas() + lemmas + f"theorem restating_test (hk : {sub}) : {target} := by\n  intros\n  apply hk <;> assumption\n"
                    + (f"\nend {ns}\n" if ns else "\n"))
            chk = lean_check(test, (), "ideas", tag=f"restating_{goal.name}")
            self.log.write(tool="no-progress check", goal=goal.name, depth=goal.depth, sub=k, accepted=False, restates=chk.ok, secs=chk.secs,
                           errors=chk.errors[:1])
            if chk.ok:
                return f"#{k}"
        return None

    def solve_sketch(self, goal: Goal, sketch: str, lemmas: str, infos: list) -> "dict | None":
        subs = []
        ns = namespace_of(goal.prefix)
        for k, info in enumerate(infos, 1):
            self.nsub += 1
            name = (f"{self.top}_" if self.top else "") + f"{self.mode}_s{self.nsub}"   # (unique in the run, across runs and tops; its parent recorded)
            stmt = re.sub(r"^theorem \S*?extracted_\w+", f"theorem {name}", info.strip())
            stmt = re.sub(r":= sorry$", ":= by", stmt).replace("✝", "_d")
            subs.append(Goal(name=name, stmt=stmt, prefix=goal.prefix + lemmas, suffix=f"\nend {ns}\n" if ns else "\n",
                             full=f"{ns}.{name}" if ns else name, depth=goal.depth + 1, parent=goal.name))
            self.proposed.append((name, goal.name, stmt))
        done = {}
        for j, s in enumerate(subs):
            if self.left() <= 0:
                self.open[s.name] = dict(statement=s.stmt, last="not attacked: the fuse ran out", parent=goal.name)
                continue
            self.share.append(time.monotonic() + self.left() / (len(subs) - j))       # (each sub-goal its share of what is left: none starves the others)
            try:
                r = self.attack(s)
            finally:
                self.share.pop()
            if r:
                text = (r.get("lemmas") or "") + s.stmt + "\n" + indent(r["proof"]) + "\n\n"
                self.proved[s.name] = dict(text=text, statement=s.stmt, proof=r["proof"], tool=r["tool"], parent=goal.name)
                done[s.name] = r
            else:
                self.open[s.name] = dict(statement=s.stmt, last=self.results.get(s.name, {}).get("last"), parent=goal.name)
        if len(done) < len(subs) or not subs:
            return None
        # every goal of the sketch proved: the parent from them
        def use(k: int, form: str) -> str:
            s = subs[k]
            call = " ".join([s.name] + binder_names(s.stmt))
            return {"by": f"by exact {call}", "tactic": f"exact {call}"}.get(form, f"({call})")
        assembled = fill_sorries(sketch, use)
        ok, chk, why = self.check(goal, assembled, lemmas, "llm sketch assembled", dict(mode=self.mode))
        if ok:
            return dict(tool="decomposition (Goedel-Prover-V2-8B sketch; sub-goals by " + ", ".join(sorted({d['tool'] for d in done.values()})) + ")",
                        proof=assembled, lemmas=lemmas, check=chk)
        return None


def fill_sorries(proof: str, fill) -> str:
    """Each `sorry` of a sketch, in order, replaced by fill(k, form): form 'by' when it stands as `by sorry` (the whole `by sorry` is replaced by a `by ...`),
    'tactic' when it is a tactic line of its own (replaced by tactics), 'term' otherwise (replaced by a term)."""
    k = [0]

    def sub(m):
        form = "by" if m.group(1) else "tactic" if m.group(2) is not None else "term"
        out = (m.group(2) or "") + fill(k[0], form)
        k[0] += 1
        return out
    return re.sub(r"(\bby\s+)sorry\b|(^[ \t]*)sorry\b|\bsorry\b", sub, proof, flags=re.M)


def closed_prop(stmt: str) -> str:
    """A theorem's statement `theorem X binders : concl := by` (a docstring before it allowed) as one closed proposition `∀ binders, concl`."""
    m = re.search(r"(?:^|\n)theorem \S+", stmt)
    if m is None:
        return ""
    head = stmt[m.end():].split(":= by")[0]
    depth = 0
    for i, ch in enumerate(head):
        if ch in "([{⦃":
            depth += 1
        elif ch in ")]}⦄":
            depth -= 1
        elif ch == ":" and depth == 0:
            binders, concl = head[:i].strip(), head[i + 1:].strip()
            return f"∀ {binders}, {concl}" if binders else concl
    return ""


def binder_names(stmt: str) -> list:
    """The explicit binder names of a `theorem X (a b : T) (h : P) ... : goal` statement, in order (instance and implicit binders are inferred)."""
    head = stmt.split(":= by")[0]
    out, depth, i, start = [], 0, 0, None
    s = head[head.find(" ", len("theorem ")):]
    while i < len(s):
        ch = s[i]
        if ch in "([{⦃":
            if depth == 0:
                start = (ch, i)
            depth += 1
        elif ch in ")]}⦄":
            depth -= 1
            if depth == 0 and start and start[0] == "(":
                inner = s[start[1] + 1:i]
                if ":" in inner:
                    out += inner.split(":")[0].split()
        elif ch == ":" and depth == 0:
            break
        i += 1
    return out


# ---------------------------------------------------------------------------------------------------------------- the run

def prove(module: str, mode: str = "blind", tools: tuple = ("automation", "copilot", "llm"), fuse: float = FUSE_SECONDS, log_print=print,
          think: bool = False, tag: str = "", on_write=None, hints: tuple = ()) -> dict:
    """THE RUN on the emitted file's goals -> the summary (also <run>/summary.json); prints the OBS lines. `on_write`: called after every attempt. `module`: a module of
    lean/ideas/Ideas/Mined, or the path of a .lean file (a scratch copy of an obligation: the file is read, never written). `hints`: full names of library theorems the
    index holds (the hinted run of a lemma whose hint is theorems, not Kaveh's text)."""
    path = Path(module) if module.endswith(".lean") else MINED_DIR / f"{module}.lean"
    module = path.stem
    run = time.strftime("%Y%m%d_%H%M%S") + f"_{module}_{mode}" + (f"_{tag}" if tag else "")
    log = Log(Path(paths.experiment("prover", run)), on_write)
    goals = goals_of(path)
    summary = dict(module=module, mode=mode, tools=list(tools), fuse=fuse, model=MODEL, think=think, token_cap=TOKEN_CAP, run=run, goals={},
                   started=time.strftime("%Y-%m-%d %H:%M:%S"), hint=(list(hints) if hints else hint_text()) if mode == "hinted" else None,
                   hints=list(hints), file=str(path))
    for g in goals:
        P = Prover(path, mode, tuple(tools), log, fuse, think, hints)
        P.top = g.name.removeprefix("conjecture_")
        res = P.attack(g)
        secs = round(time.monotonic() - P.t0, 1)
        summary["goals"][g.name] = dict(
            closed_by=res["tool"] if res else None, proof=res["proof"] if res else None, lemmas=res.get("lemmas", "") if res else "",
            last=P.results.get(g.name, {}).get("last"), secs=secs, samples=P.samples,
            proposed=[dict(name=n, parent=p, statement=s) for n, p, s in P.proposed],
            proved={n: dict(statement=v["statement"], proof=v["proof"], tool=v["tool"], parent=v["parent"], text=v["text"]) for n, v in P.proved.items()},
            open={n: v for n, v in P.open.items() if n not in P.proved},
            subresults={n: v for n, v in P.results.items() if n != g.name})
        gs = summary["goals"][g.name]
        log_print(f"OBS prover {module} {mode} goal {g.name}: " + (f"CLOSED by {gs['closed_by']}" if res else f"OPEN after the fuse; last: {str(gs['last'])[:300]}")
                  + f"; samples {P.samples}; sub-goals proposed {len(P.proposed)}, proved {len(P.proved)}, open {len(gs['open'])}; wall {secs} s")
    summary["attempts"] = log.n
    summary["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (log.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    tally: dict = {}
    for ln in log.path.read_text().splitlines() if log.path.exists() else []:
        r = json.loads(ln)
        k = r.get("tool", "?") + (f" {r['tactic']}" if r.get("tactic") else "")
        a = tally.setdefault(k, [0, 0])
        a[0] += 1
        a[1] += bool(r.get("accepted"))
    log_print(f"OBS prover {module} {mode} tally (tried, accepted): " + "; ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(tally.items()))
              + f"; attempts logged {log.n} in {os.path.relpath(log.path, str(paths.REPO_ROOT))}")
    return summary


# ---------------------------------------------------------------------------------------------------------------- the tool project

TOOL_PINS = (   # (name, url, the tag's commit, inputRev, config): Duper and LeanCopilot at their Lean 4.32 releases; lean-auto at the commit Duper pins
    ("Duper", "https://github.com/leanprover-community/duper", "0992bfade8392ca97ef0f3c547aec414b3b42075", "v4.32.0", "lakefile.lean"),
    ("auto", "https://github.com/leanprover-community/lean-auto.git", "fcbce0f216e71516e88b784944636da4d28ee780", "fcbce0f216e71516e88b784944636da4d28ee780",
     "lakefile.lean"),
    ("LeanCopilot", "https://github.com/lean-dojo/LeanCopilot", "2a5cd0ea68268bab49dc96d4d480bfe4fa22a38e", "v4.32.0", "lakefile.lean"),
)


def setup_tools(log=print) -> None:
    """THE SCRATCH TOOL PROJECT (artifacts/experiments/prover/lean_tools): its own .lake; lean/ideas's packages LINKED (never updated -- the manifest copies
    lean/ideas's revisions exactly, so Lake resolves nothing and rebuilds no Mathlib module; never run `lake update` there); Ideas as a path dependency;
    Duper / lean-auto / LeanCopilot cloned into it at their pins and built (Duper's precompiled modules add Batteries' native objects to the shared Batteries
    build directory, no .olean; LeanCopilot's source build of ct2.cpp fails when its copy of the ctranslate2 headers is nested twice -- the nested copy is
    removed and the build repeated); LeanCopilot's tactic model is fetched by its own `download` (into ~/.cache/lean_copilot; its Git LFS step fails without
    git-lfs, the tactic generator's weights arrive regardless -- checked by size). Checks there run with DYLD_LIBRARY_PATH at LeanCopilot's lib (its FFI
    library looks for libctranslate2 in the toolchain's directory only)."""
    shared = (LEAN_DIR / ".lake" / "packages").resolve()
    (TOOLS_DIR / ".lake" / "packages").mkdir(parents=True, exist_ok=True)
    (TOOLS_DIR / "ProverTools").mkdir(exist_ok=True)
    base = json.loads((LEAN_DIR / "lake-manifest.json").read_text())
    pk = [{"type": "path", "scope": "", "name": "Ideas", "manifestFile": "lake-manifest.json", "inherited": False, "dir": str(LEAN_DIR), "configFile": "lakefile.toml"}]
    for e in base["packages"]:
        link = TOOLS_DIR / ".lake" / "packages" / e["name"]
        if not link.exists():
            os.symlink(shared / e["name"], link)
        pk.append(dict(e, inherited=True))
    for name, url, rev, inp, cfg in TOOL_PINS:
        pk.append({"url": url, "type": "git", "subDir": None, "scope": "", "rev": rev, "name": name, "manifestFile": "lake-manifest.json", "inputRev": inp,
                   "inherited": name == "auto", "configFile": cfg})
    (TOOLS_DIR / "lake-manifest.json").write_text(json.dumps({"version": "1.2.0", "packagesDir": ".lake/packages", "packages": pk, "name": "ProverTools",
                                                             "lakeDir": ".lake", "fixedToolchain": False}, indent=1) + "\n")
    (TOOLS_DIR / "lean-toolchain").write_text((LEAN_DIR / "lean-toolchain").read_text())
    (TOOLS_DIR / "lakefile.toml").write_text(f'name = "ProverTools"\n\n[[require]]\nname = "Ideas"\npath = "{LEAN_DIR}"\n\n[[require]]\nname = "Duper"\n'
                                            f'git = "{TOOL_PINS[0][1]}"\nrev = "v4.32.0"\n\n[[require]]\nname = "LeanCopilot"\ngit = "{TOOL_PINS[2][1]}"\nrev = "v4.32.0"\n\n'
                                            f'[[lean_lib]]\nname = "ProverTools"\n')
    (TOOLS_DIR / "ProverTools.lean").write_text("-- the scratch library of the prover harness (plan 73.6 (3)); every attempt is a module of it\n")
    chk = subprocess.run(["lake", "build", "--no-build", "Ideas.Base.Vocabulary", "Ideas.ForcedByMeasure"], cwd=TOOLS_DIR, capture_output=True, text=True)
    log(f"OBS prover setup: the shared build seen from the tool project: {(chk.stdout + chk.stderr).strip().splitlines()[-1][:160]}")
    if chk.returncode != 0:
        raise SystemExit("the tool project would rebuild the shared packages: stopped (nothing built)")
    for target in ("Duper", "LeanCopilot"):
        r = subprocess.run(["lake", "build", target], cwd=TOOLS_DIR, capture_output=True, text=True)
        nested = TOOLS_DIR / ".lake" / "packages" / "LeanCopilot" / ".lake" / "build" / "include" / "ctranslate2" / "ctranslate2"
        if r.returncode != 0 and target == "LeanCopilot" and nested.exists():
            import shutil
            shutil.rmtree(nested)
            r = subprocess.run(["lake", "build", target], cwd=TOOLS_DIR, capture_output=True, text=True)
        log(f"OBS prover setup: lake build {target}: exit {r.returncode}; {(r.stdout + r.stderr).strip().splitlines()[-1][:160]}")
    r = subprocess.run(["lake", "exe", "LeanCopilot/download"], cwd=TOOLS_DIR, capture_output=True, text=True)
    log(f"OBS prover setup: LeanCopilot/download exit {r.returncode}: {(r.stdout + r.stderr).strip().splitlines()[-1][:160]}")


# ---------------------------------------------------------------------------------------------------------------- (4) THE STORE

ROLE_READS = {"movement": ("legalMoves", "applyMove", "ours", "theirs", "Reach", "stalemate", "checkmate", "mate", "reach", "dist", "kingSqD"),
              "tempo": ("legalMoves", "applyMove", "ours", "theirs", "Reach"),
              "attack": ("inCheckB", "attackedBy", "isAttackedB", "unitLost", "checkmate", "stalemate", "seeOf"),
              "material": ("seeOf", "unitLost")}


def conclusion(stmt: str) -> str:
    """The conclusion of `theorem X binders : concl := by` (after the binders' top-level colon)."""
    head = stmt.split(":= by")[0]
    depth = 0
    for i, ch in enumerate(head):
        if ch in "([{⦃":
            depth += 1
        elif ch in ")]}⦄":
            depth -= 1
        elif ch == ":" and depth == 0 and i > len("theorem "):
            return head[i + 1:].strip()
    return head


def role(stmt: str) -> str:
    """THE KIND BY ROLE of a sub-lemma (plan 73.6 (4)): it concludes a forcing (`Reach`) -> FORCING; a mate -> CLOSING; anything else (a never-rule kept, a move
    that exists, a measure's bound) -> DEFECT_FIX (a condition the technique keeps or a defect it rules out)."""
    c = conclusion(stmt)
    if re.search(r"\bReach\b", c):
        return "FORCING"
    if re.search(r"\b(checkmate|mate)\b", c):
        return "CLOSING"
    return "DEFECT_FIX"


def reads_of(stmt: str, measure_reads: tuple) -> list:
    """The base words a sub-lemma's statement reads: the kernel's words it names (moves: movement and tempo; attacks, checks: attack; the exchange: material
    and attack) and the measure's own reads when it names the measure."""
    words = set(re.findall(r"[A-Za-z_][\w']*", stmt))
    out = {b for b, ws in ROLE_READS.items() if words & set(ws)}
    if words & {"mu"} or any(w.startswith("atom") for w in words):
        out |= set(measure_reads)
    return sorted(out or {"movement"})


def _entry_of_module(module: str) -> "tuple":
    from catspace.incubator.analyst.discovery import lemma_store as LS
    from catspace.incubator.analyst.discovery import technique_lean as TL
    for e in LS.load(force=True).values():
        if e["kind"] == "TECHNIQUE" and isinstance(e.get("measure"), dict) and "class" in e["measure"] and TL.module_name(e) == module:
            return e, TL
    raise SystemExit(f"no TECHNIQUE entry of the store emits {module}")


def no_word(module: str, ns: str, stmt: str, ancestors: list) -> "str | None":
    """NOT A WORD (Lean decides, in the built module's context): a sub-lemma whose conclusion is one of its own hypotheses (`intros; assumption` closes it)
    states nothing; one from which an ancestor goal follows by `apply` and the ancestor's own hypotheses restates the ancestor. -> the reason, or None. Such a
    sub-goal is shown in the run's record and never entered in the store (a word must say something its parents do not)."""
    head = f"import Ideas.Mined.{module}\n\nnamespace {ns}\n\nopen Chess hiding between\nopen Ideas.Vocabulary\n\n"
    prop = closed_prop(stmt)
    if not prop:
        return None
    t = lean_check(head + f"theorem no_word_tautology : {prop} := by\n  intros\n  assumption\n\nend {ns}\n", (), "ideas", tag="no_word")
    if t.ok:
        return "a tautology: its conclusion is one of its hypotheses (`intros; assumption` closes it)"
    for a in ancestors:
        ap = closed_prop(a)
        if not ap:
            continue
        r = lean_check(head + f"theorem no_word_restates (hk : {prop}) : {ap} := by\n  intros\n  apply hk <;> assumption\n\nend {ns}\n", (), "ideas",
                       tag="no_word")
        if r.ok:
            return "a restatement of an ancestor goal (the ancestor follows from it by `apply` and its own hypotheses)"
    return None


def ancestors_of(name: str, proposed: list, top_stmt: str) -> list:
    """The statements of a sub-goal's ancestors, its parent first, the top goal last."""
    by = {d["name"]: d for d in proposed}
    out, cur = [], by.get(name, {}).get("parent")
    while cur in by:
        out.append(by[cur]["statement"])
        cur = by[cur].get("parent")
    return out + [top_stmt]


def words_of(module: str, ns: str, g: dict, top_stmt: str, log=print) -> tuple:
    """The run's sub-goals that are WORDS (proved, open) and the ones that are not ({name: reason}), by `no_word`."""
    proved = dict(g.get("proved") or {})
    open_ = {n: v for n, v in (g.get("open") or {}).items() if n not in proved}
    dropped = {}
    for pool in (proved, open_):
        for n in list(pool):
            why = no_word(module, ns, pool[n]["statement"], ancestors_of(n, g.get("proposed") or [], top_stmt))
            if why:
                dropped[n] = why
                del pool[n]
                log(f"OBS prover record: {n} is not a word -- {why}")
    return proved, open_, dropped


def drop_run_entries(sub: str) -> list:
    """Re-recording a run replaces its own outputs: the store entries implemented in the run's file and the file itself -> the names removed."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    gone = []
    for e in LS.load(force=True).values():
        if any(i.startswith(f"lean/ideas/Ideas/Mined/{sub}.lean:") for i in e["implementation"]):
            (LS.DIR / f"{LS.slug(e['name'])}.json").unlink()
            gone.append(e["name"])
    f = MINED_DIR / f"{sub}.lean"
    if f.exists():
        f.unlink()
    LS.load(force=True)
    return gone


def _axioms_of(module: str, names: tuple, proofs: tuple, tag: str) -> tuple:
    """The acceptance check of theorems of a BUILT module (imported, not re-elaborated): `#print axioms` and the scan of the proof texts."""
    chk = lean_check(f"import Ideas.Mined.{module}\n", names, "ideas", tag=tag)
    return accept(chk, names, proofs)


def sub_file_text(module: str, ns: str, run: str, mode: str, proved: dict, open_: dict) -> str:
    """THE RUN'S LEAN FILE of sub-goals (Ideas/Mined/<module>_<mode>.lean): the PROVED ones with the proofs the tools found, the OPEN ones with `sorry` and the
    prover's last state above it; it imports the emitted module (the definitions) and shares its namespace."""
    lines = [f"import Ideas.Mined.{module}", "",
             f"/-! AUTO-WRITTEN by catspace/incubator/analyst/discovery/prover.py (record) from prover run {run} ({mode}) on Ideas.Mined.{module} -- do not edit by",
             "hand. The SUB-GOALS the tools proposed on the way (plan 73.6 (4)): each PROVED one with the proof the tools found (accepted on its own: no sorry,",
             "the standard axioms only, no enumeration); each OPEN one with `sorry` and the prover's last state above it. Nobody wrote these proofs. -/", "",
             "set_option linter.unusedVariables false", "", f"namespace {ns}", "", "open Chess hiding between", "open Ideas.Vocabulary", ""]
    for n, v in open_.items():                                          # (the OPEN ones first: lean_index reads a theorem's body to the next declaration)
        last = re.sub(r"-/", "- /", str(v.get("last") or "?"))[:900]
        lines += [f"/-- OPEN sub-goal of {v.get('parent')} (run {run}); the prover's last state: {last} -/", open_name(v["statement"]).rstrip(), "  sorry", ""]
    for n, v in proved.items():
        lines += [f"/-- PROVED by {v['tool']} (sub-goal of {v['parent']}, run {run}). -/", v["text"].rstrip(), ""]
    lines += [f"end {ns}", ""]
    return "\n".join(lines)


def open_name(stmt: str) -> str:
    """An open statement under the repo's name for one (`conjecture_<name>`: tests/test_lean_index.py)."""
    return re.sub(r"^theorem (?!conjecture_)(\S+)", r"theorem conjecture_\1", stmt.strip())


def sub_entry(n: str, v: dict, ns: str, sub: str, run: str, mode: str, entry_name: str, mreads: tuple, day: str, accepted: "bool | None",
              why: tuple = ()) -> dict:
    """A sub-goal's STORE ENTRY: origin MINED, kind by role, reads by its statement, implementation its theorem in the run's file; PROVED (lean = its name) only
    when `accepted` (the acceptance check on the built file), else CONJECTURE -- a proved one the check rejected says why, an open one (`accepted` None)
    carries the prover's last state."""
    n = n if accepted is not None else ("" if n.startswith("conjecture_") else "conjecture_") + n      # (an open one is named as the file names it)
    base = dict(name=f"mined lemma {ns}.{n}", kind=role(v["statement"]), reads=reads_of(v["statement"], mreads), terms=[], origin="MINED", plan=["73.6"],
                journal=day, measure=None, implementation=[f"lean/ideas/Ideas/Mined/{sub}.lean:{n}"])
    if accepted is None:
        return dict(base, status="CONJECTURE", lean=[], evidence=dict(uses={}, confirmed=[], refuted=[]),
                    note=f"A sub-goal the prover proposed for {v.get('parent')} and did not prove (run {run}, {mode}); the prover's last state: "
                         f"{str(v.get('last') or '?')[:400]}")
    return dict(base, status="PROVED" if accepted else "CONJECTURE", lean=[f"{ns}.{n}"] if accepted else [],
                evidence=dict(uses={}, confirmed=[f"proved by {v['tool']} in prover run {run} ({mode}), a sub-goal of {v['parent']}" + ("" if accepted else
                              f"; the acceptance check REJECTED it: {'; '.join(why)}")], refuted=[]),
                note=f"A sub-lemma the tools proposed and proved on the way to {entry_name}'s obligation (plan 73.6): {conclusion(v['statement'])[:300]}")


def record(summary_path: str, log=print) -> dict:
    """(4) THE STORE from a run's summary: an ACCEPTED top proof -> written into the emitted file by code (technique_lean.emit), `lake build` of it (no sorry),
    the acceptance check again on the built module, the entry PROVED (lean = hprog and the corollary); every PROVED sub-lemma (accepted on its own) -> the run's
    Lean file Ideas/Mined/<module>_<mode>.lean and its own entry (origin MINED, kind by role, implementation the file); every OPEN sub-goal -> the same file,
    `sorry` and the prover's last state above it, and a CONJECTURE entry; then docs/LEAN_INDEX.md rebuilt and the store validated. Nothing is upgraded without
    the check."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    from catspace.incubator.analyst.discovery import technique_miner as MN
    s = json.load(open(summary_path))
    module, mode, run = s["module"], s["mode"], s["run"]
    entry, TL = _entry_of_module(module)
    ns = TL.namespace(entry)
    mreads = tuple(sorted(MN.measure_of(entry["measure"]["quantity"]).reads))
    out = dict(proved_top=False, entries=[], file=None)
    g = next(iter((s.get("goals") or {}).values()), {})               # (the run's one top goal: hprog, or conjecture_hprog while it is open)
    day = time.strftime("%Y-%m-%d")
    if g.get("closed_by"):
        TL.emit(entry, proof=dict(proof=g["proof"], lemmas=g.get("lemmas") or "", tool=g["closed_by"], run=run))
        b = TL.build(module)
        ok, why = (False, ["lake build: " + "; ".join(b["errors"][:3])]) if not b["ok"] or b["sorries"] else _axioms_of(
            module, (f"{ns}.hprog", f"{ns}.forced_mate"), (g["proof"], g.get("lemmas") or ""), f"record_{module}")
        log(f"OBS prover record {module} {mode}: the top proof written and built: ok {b['ok']}, sorry warnings {b['sorries']}; the acceptance check "
            + ("ACCEPTED" if ok else "REJECTED: " + "; ".join(why)))
        if ok:
            e = dict(entry, status="PROVED", lean=[f"{ns}.hprog", f"{ns}.forced_mate"])
            e["evidence"] = dict(entry["evidence"], confirmed=entry["evidence"]["confirmed"] + [
                f"PROVED in Lean ({day}, plan 73.6): hprog found by {g['closed_by']} (prover run {run}, mode {mode}); accepted: lake build, no sorry, the standard axioms "
                f"only, no enumeration; lean/ideas/Ideas/Mined/{module}.lean"])
            e["note"] = entry["note"] + f" PROVED by plan 73.6's prover ({g['closed_by']}, run {run}): Ideas.Mined.{module}.hprog and forced_mate."
            MN.write_entry(e)
            out["proved_top"] = True
            out["entries"].append(e["name"])
        else:
            TL.emit(entry)                                            # (not accepted: the statement as it was, the sorry back)
    drop_run_entries(f"{module}_{mode}")
    top_stmt = goals_of(MINED_DIR / f"{module}.lean")[0].stmt if goals_of(MINED_DIR / f"{module}.lean") else ""
    proved, open_, dropped = words_of(module, ns, g, top_stmt, log)
    out["not_words"] = dropped
    if proved or open_:
        sub = f"{module}_{mode}"
        path = MINED_DIR / f"{sub}.lean"
        path.write_text(sub_file_text(module, ns, run, mode, proved, open_))
        b = TL.build(sub)
        log(f"OBS prover record {sub}: written, lake build ok {b['ok']}, sorry warnings {b['sorries']} (the open sub-goals: {len(open_)})"
            + ("".join("\n  " + x for x in b["errors"][:6]) if b["errors"] else ""))
        out["file"] = str(path)
        if b["ok"]:
            for n, v in proved.items():
                ok, why = _axioms_of(sub, (f"{ns}.{n}",), (v["proof"],), f"record_{sub}_{n}")
                e = sub_entry(n, v, ns, sub, run, mode, entry["name"], mreads, day, accepted=ok, why=why)
                MN.write_entry(e)
                out["entries"].append(e["name"])
            for n, v in open_.items():
                e = sub_entry(n, v, ns, sub, run, mode, entry["name"], mreads, day, accepted=None)
                MN.write_entry(e)
                out["entries"].append(e["name"])
    if out["entries"] or out["file"]:
        r = subprocess.run([str(Path(paths.REPO_ROOT) / ".venv" / "bin" / "python"), "-m", "catspace.incubator.analyst.lean_index", "build"], cwd=paths.REPO_ROOT,
                           capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(paths.REPO_ROOT)))
        log(f"OBS prover record: lean_index build exit {r.returncode}: {(r.stdout.strip().splitlines() or [''])[-1][:300]}")
        rc = LS.validate(log=log, run=False)
        out["validate"] = rc
    log(f"OBS prover record {module} {mode}: the top obligation {'PROVED' if out['proved_top'] else 'not proved'}; store entries written {len(out['entries'])}"
        + (": " + ", ".join(out["entries"]) if out["entries"] else ""))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prove", help="run the tools on the emitted file's sorry")
    p.add_argument("module", help="the module under lean/ideas/Ideas/Mined/, e.g. mined_technique_krvk, or the path of a .lean file (a scratch obligation)")
    p.add_argument("--mode", choices=("blind", "hinted"), default="blind")
    p.add_argument("--fuse", type=float, default=FUSE_SECONDS, help="seconds of wall-clock for the top obligation (default: the named fuse)")
    p.add_argument("--tools", default="automation,copilot,llm")
    p.add_argument("--think", action="store_true", help="the model's reasoning mode on (off by default: see the docstring's fuse)")
    p.add_argument("--hints", default="", help="hinted mode: comma-separated full names of library theorems of the index (their statements go to the model; the automation is handed the names)")
    p.add_argument("--tag", default="", help="a tag for the run's directory (e.g. probe)")
    c = sub.add_parser("accept", help="the acceptance check of a Lean file as it stands (compiled in lean/ideas)")
    c.add_argument("file")
    c.add_argument("theorems", nargs="+")
    r = sub.add_parser("record", help="(4) the store from a run's summary.json")
    r.add_argument("summary")
    sub.add_parser("setup", help="the scratch tool project: Duper and LeanCopilot against the shared build (setup_tools)")
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        return setup_tools()
    if a.cmd == "prove":
        prove(a.module, a.mode, tuple(a.tools.split(",")), a.fuse, think=a.think, tag=a.tag, hints=tuple(h for h in a.hints.split(",") if h))
    elif a.cmd == "record":
        record(a.summary)
    else:
        text = Path(a.file).read_text()
        chk = lean_check(text, tuple(a.theorems), "ideas", tag="accept_" + Path(a.file).stem)
        ok, why = accept(chk, tuple(a.theorems), (text,))
        print(f"OBS prover accept {a.file}: {'ACCEPTED' if ok else 'REJECTED'}" + ("" if ok else ": " + "; ".join(why)) + f"; axioms {chk.axioms}; {chk.secs} s")


if __name__ == "__main__":
    main()
