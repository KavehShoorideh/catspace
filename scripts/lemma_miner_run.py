"""PLAN 79.59 (THE LEMMA MINER OF LAYER 2; SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules; ledger rows 59, 58, 57, 49): THE SLEEP SCRIPT. It names no game: the games come from scripts/gdl_to_rust_make.py (which may name them).
  python scripts/lemma_miner_run.py propose            PROPOSER ONE (the null change) on the composed description of every game: the proposal, its reads, and the reason where the clauses refuse it; each proposal is written as a document of the form FILTER, standing PROPOSED, to rules/lemmas/<game>/mined/.
  python scripts/lemma_miner_run.py make               the module WITH each proposed lemma (pass 29 `filter`), kept as rust/engine/game/tests/generated_mined/<game>_mined.rs, and the spec file the judge reads (paths.experiment("lemma_miner_spec.txt")).
  python scripts/lemma_miner_run.py control [<games> [<plies>]]   THE NEGATIVE CONTROL: a read struck out of the hits of the first game's lemma; the gate must fire; the document is kept as REJECTED; the true module is made again.
  python scripts/lemma_miner_run.py judge [<games> [<plies>]]   THE JUDGE (rust/engine/game/tests/translated_judge.rs, all cores for the cargo build, one run): the gate at zero tolerance, then tests for each allowed row, then the clock, on SEGMENTS of consecutive states (the games of a file for the first game, seeded walks for the others); the standing of each document is then
        written: TESTED (and `paying` or `not used for speed` by the clock) or REJECTED with the reason and the segments it failed on; PROVED where the theorem of its kind compiles. Prints the OBS lines of the judge and the readable statements.
        The run is a sleep job: the machine should be quiet (the load is printed in the clock lines); CS_FACTS_N (games, default 30) and CS_FACTS_MAX (plies, default 40) size the segments; JUDGE_ALL=1 compiles every game's modules (the default is the first game only).
Part D's rule: a lemma that passes the gate and loses the clock is kept as TESTED and marked `not used for speed`; the translator is given only lemmas marked paying (`make --kept`)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from catspace.incubator.analyst.discovery import gdl_to_rust as T
from catspace.incubator.analyst.discovery import lemma_miner as M
from catspace.incubator.analyst.discovery import rule_lemma as RL
from catspace.incubator.analyst.discovery import rule_lemma_filter as RLF
from catspace.io import paths


def _load_make():
    spec = importlib.util.spec_from_file_location("gdl_to_rust_make", ROOT / "scripts" / "gdl_to_rust_make.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MK = _load_make()
OUT_MINED = ROOT / "rust" / "engine" / "game" / "tests" / "generated_mined"
SPEC = lambda: Path(paths.experiment("lemma_miner_spec.txt"))
CARGO = ["cargo", "test", "--release", "--offline", "-j", "8", "--manifest-path", str(ROOT / "rust" / "engine" / "Cargo.toml"), "-p", "cs_game", "--test", "translated_judge", "--", "--ignored", "--nocapture"]


def description_text(game: str) -> str:
    srcs = MK.GAMES[game]
    return T.compose_text(srcs) if len(srcs) > 1 else srcs[0].read_text()


def proposals(games=None) -> list:
    """-> [(game, text, proposal or None, why)] for every test the finder found in the description of each game and a question reads. Prints the OBS lines of the first proposer (tag B)."""
    out = []
    for game in (games or MK.GAMES):
        text = description_text(game)
        d, f = M.shapes(text)
        wanted = [x for x in T.QUESTION_RELS if x in d.prog.nodes] + [f"next.{x}" for x in d.fluents if f"next.{x}" in d.prog.nodes]
        attain = d.cone(wanted)
        if not f.found:
            print(f"OBS 79.59 B {game}: no test of the shape found; refused: " + "; ".join(f"{k}: {v[:90]}" for k, v in sorted(f.refused.items())[:4]))
            out.append((game, text, None, "no test of the shape (a test of a key on a hypothetical relation made from a state relation) is found in the clauses"))
        for s in f.found:
            if s["R"] not in attain:
                print(f"OBS 79.59 B {game}: the test {s['R']} is found but no question reads it: nothing to narrow")
                out.append((game, text, None, f"{s['R']}: no question reads it"))
                continue
            try:
                pr = M.null_change(text, s, d, f)
            except M.Refused as e:
                print(f"OBS 79.59 B {game}: the test {s['R']} (hypothesis {s['H']}, state relation {s['B']}) is REFUSED by the null change: {e}")
                out.append((game, text, None, str(e)))
                continue
            print(f"OBS 79.59 B {game}: the test {pr.test} of a key of {pr.places} places on {pr.hypothetical} (made from {pr.state_relation}); the null test {pr.null_test}; guard {pr.guard}; {len(pr.reads)} reads, {len(pr.clauses)} clauses and {len(pr.signatures)} signatures of delta")
            for x in M.reads_text(pr):
                print(f"OBS 79.59 B {game}: {x}")
            out.append((game, text, pr, ""))
    return out


def write_proposals(props: list) -> list:
    """Each proposal as a document of the form FILTER, read back by `rule_lemma` (the document is refused whole when it has a problem), written to rules/lemmas/<game>/mined/."""
    out = []
    for game, text, pr, why in props:
        if pr is None:
            continue
        doc = M.document(pr, game, game)
        lem = RLF.parse(doc, base_text=text)
        p = RLF.write_mined(lem)
        print(f"OBS 79.59 C {game}: document {p.relative_to(ROOT)}, form FILTER, origin MINED, standing PROPOSED")
        out.append((game, text, pr, lem))
    return out


def module_of(game: str, text: str, lem: "RLF.FilterLemma") -> str:
    """The module WITH the lemma: the description with the lemma's delta composed on, every pass but carry and the pass filter, the lemma given as data."""
    full = RL.compose(text, lem.delta)
    doc = {"name": lem.name, "form": "FILTER", "left": lem.left, "right": lem.right, "filter": lem.filter, "standing": lem.standing}
    return T.translate(full, MK.rust_name(game) + "_mined", MK.PASSES_NOCARRY + ["filter"], [doc], ways=MK.WAYS.get(game))


def make(written: list, only_paying: bool = False) -> list:
    OUT_MINED.mkdir(parents=True, exist_ok=True)
    spec_lines, made = [], []
    for game, text, pr, lem in written:
        if only_paying and (lem.evidence or {}).get("paying") is not True:
            continue
        src = module_of(game, text, lem)
        (OUT_MINED / f"{game}_mined.rs").write_text(src)
        head = [x[3:] for x in src.split("\n", 12)[:12] if x.startswith("// lemma")]
        print(f"OBS 79.59 E translate {game}_mined (module {MK.rust_name(game)}_mined): Rust {len(src)} bytes {src.count(chr(10))} lines; " + " | ".join(head))
        spec_lines.append(f"game {game} {pr.test} {pr.answer} {pr.hits} {pr.guard_relation} {pr.null_test}")
        spec_lines += [f"hit {r.relation}" for r in pr.reads]
        spec_lines += [f"read {r.literal} of {r.clause}" for r in pr.reads]
        spec_lines.append("cone " + " ".join([pr.test] + sorted(set(pr.cone) - {pr.test})))
        spec_lines.append("dviews " + " ".join(pr.diff_views))
        made.append((game, text, pr, lem))
    SPEC().write_text("\n".join(spec_lines) + "\n")
    return made


def parse_lines(stdout: str) -> dict:
    """The judge's lines by game: {game: {"result": {...}, "gate": line, "narrow": line, "clock": line, "reads": [(i, relation, keys a state, share of hits)], "states": [lines]}}."""
    res: dict = {}
    for ln in stdout.splitlines():
        if ln.startswith("MINED RESULT "):
            f = ln.split()
            g = f[2]
            res.setdefault(g, {})["result"] = {f[i]: float(f[i + 1]) for i in range(3, len(f) - 1, 2)}
        elif ln.startswith("MINED READ "):
            f = ln.split()
            res.setdefault(f[2], {}).setdefault("reads", []).append((int(f[3]), f[4], float(f[5]), float(f[6])))
        elif ln.startswith("OBS 79.59 D "):
            g = ln.split()[3]
            d = res.setdefault(g, {})
            for key in ("gate", "narrowness", "clock"):
                if f" {key}:" in ln:
                    d[key] = ln
        elif ln.startswith("MINED "):
            res.setdefault("_states", {}).setdefault("lines", []).append(ln)
    return res


def judge(made: list, games_file: Path, n: int, plies: int) -> None:
    env = {**os.environ, "CS_MINED_SPEC": str(SPEC()), "CS_FACTS_GAMES": str(games_file), "CS_FACTS_N": str(n), "CS_FACTS_MAX": str(plies), "CS_MINED_KEEP_GOING": "1"}
    if os.environ.get("JUDGE_ALL"):
        env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " --cfg judge_all").strip()
    r = subprocess.run(CARGO + ["judge_all"], env=env, capture_output=True, text=True)
    if r.returncode and "OBS 79.59 D" not in r.stdout:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(r.returncode)
    for ln in r.stdout.splitlines():
        if ln.startswith(("OBS 79.59 D", "MINED STATE", "MINED ALLOWED", "MINED REFUSED")):
            print(ln)
    got = parse_lines(r.stdout)
    for game, text, pr, lem in made:
        g = got.get(game)
        if not g or "result" not in g:
            print(f"OBS 79.59 D {game}: not judged (no module with the lemma was compiled into the judge for it); the document stays PROPOSED")
            continue
        res = g["result"]
        gate_held = res["differs"] == 0
        paying = gate_held and res["ns_with"] < res["ns_without"]
        evidence = {"gate": "held" if gate_held else "failed", "field": g.get("gate", ""), "narrowness": g.get("narrowness", ""), "clock": g.get("clock", ""), "paying": bool(paying),
                    "use": "paying" if paying else "not used for speed", "tests_per_allowed_row_without": res["tests_without"], "tests_per_allowed_row_with": res["tests_with"], "ns_without": res["ns_without"], "ns_with": res["ns_with"]}
        if gate_held:
            standing = {"is": "TESTED", "detail": "the gate held at every state of the segments (evidence.field); " + ("paying: the clock of the module with the lemma is under the module without it" if paying else "not used for speed: the clock of the module with the lemma is not under the module without it")}
        else:
            standing = {"is": "REJECTED", "detail": "the gate failed (evidence.field names the states and the first failing segments); conditional on the field it was tested on, not a verdict beyond it"}
        lem2 = RLF.FilterLemma(**{**lem.__dict__, "standing": standing, "evidence": evidence})
        if gate_held:
            lem3 = RLF.FilterLemma(**{**lem2.__dict__, "standing": {"is": "PROVED", "detail": "TESTED and the theorem of its kind compiles"}})
            word, detail = RLF.filter_standing(lem3)
            if word == "PROVED":
                use = "paying: the clock of the module with the lemma is under the module without it" if paying else "not used for speed: the clock of the module with the lemma is not under the module without it"
                lem2 = RLF.FilterLemma(**{**lem3.__dict__, "standing": {"is": "PROVED", "detail": f"{detail}; the gate held at every state of the segments (evidence.field); {use}"}})
        RLF.write_mined(lem2)
        counts = {x[1]: (x[2], x[3]) for x in g.get("reads", [])}
        print(f"OBS 79.59 D {game} standing: {lem2.standing['is']} ({lem2.standing['detail'][:160]})")
        for ln in M.statement(pr, counts):
            print(f"MINED STATEMENT {game}: {ln}")


def control(games_file: Path, n: int, plies: int) -> None:
    """THE NEGATIVE CONTROL (never kept as a lemma in use): the first game's proposal with every read but the first struck out of the hits (the clauses that make the hits from the keys those reads meet are left out), the module made from it, the judge run on it. The gate must fire; the document is kept as REJECTED with the segments it failed on
    (a rejection is kept, conditional on the field it was tested on); the true module is made again afterwards."""
    written = write_proposals(proposals([next(iter(MK.GAMES))]))
    game, text, pr, lem = written[0]
    victims = [r.relation for r in pr.reads[1:]]                # every read but the first (the first alone leaves the keys that read a line out of the hits)
    kept = [c for c in pr.clauses if not (c.startswith(f"(<= ({pr.hits} ") and any(f"({v} " in c for v in victims))]
    assert len(kept) == len(pr.clauses) - len(victims), "the clauses to strike out were not found"
    bad = M.Proposal(**{**pr.__dict__, "clauses": kept, "name": pr.name + "_planted_fault"})
    doc = M.document(bad, game, game)
    lem_bad = RLF.parse(doc, base_text=text)
    made = make([(game, text, bad, lem_bad)])
    print(f"OBS 79.59 D control: the planted fault strikes the reads {victims} out of the hits of {game}")
    judge_control(made, games_file, n, plies, written)


def judge_control(made, games_file, n, plies, written) -> None:
    env = {**os.environ, "CS_MINED_SPEC": str(SPEC()), "CS_FACTS_GAMES": str(games_file), "CS_FACTS_N": str(n), "CS_FACTS_MAX": str(plies), "CS_MINED_KEEP_GOING": "1", "CS_MINED_SHOW": "0", "CS_MINED_REPS": "1"}
    r = subprocess.run(CARGO + ["judge_all"], env=env, capture_output=True, text=True)
    got = parse_lines(r.stdout)
    game, text, bad, lem_bad = made[0]
    res = got.get(game, {}).get("result")
    gate = got.get(game, {}).get("gate", "")
    print(gate)
    caught = bool(res and res["differs"] > 0)
    print(f"OBS 79.59 D control: the gate {'FIRES' if caught else 'DID NOT FIRE'} on the planted fault ({int(res['differs']) if res else '?'} states differ of {int(res['states']) if res else '?'})")
    if caught:
        standing = {"is": "REJECTED", "detail": "the planted fault of the negative control: a read struck out of the hits; the gate failed (evidence.field); conditional on the field it was tested on"}
        RLF.write_mined(RLF.FilterLemma(**{**lem_bad.__dict__, "standing": standing, "evidence": {"gate": "failed", "field": gate, "control": True}}))
    make(written)                                              # the true module again
    print("OBS 79.59 D control: the module of the true lemma is made again")


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    if cmd == "control":
        n = int(argv[1]) if len(argv) > 1 else int(os.environ.get("CS_FACTS_N", "30"))
        plies = int(argv[2]) if len(argv) > 2 else int(os.environ.get("CS_FACTS_MAX", "40"))
        out = Path(paths.experiment("facts_moved_games.txt"))
        subprocess.run([sys.executable, str(ROOT / "scripts" / "ply_profile_print.py"), "games", "lichess_2019-01", str(out), str(n)], check=True)
        control(out, n, plies)
        return 0
    if cmd == "propose":
        write_proposals(proposals())
        return 0
    if cmd == "make":
        written = write_proposals(proposals())
        make(written, only_paying="--kept" in argv)
        return 0
    if cmd == "judge":
        n = int(argv[1]) if len(argv) > 1 else int(os.environ.get("CS_FACTS_N", "30"))
        plies = int(argv[2]) if len(argv) > 2 else int(os.environ.get("CS_FACTS_MAX", "40"))
        written = write_proposals(proposals())
        made = make(written)
        sys.path  # (no path is added: the games file is made by the printing script of the per-ply measure)
        out = Path(paths.experiment("facts_moved_games.txt"))
        subprocess.run([sys.executable, str(ROOT / "scripts" / "ply_profile_print.py"), "games", "lichess_2019-01", str(out), str(n)], check=True)
        judge(made, out, n, plies)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
