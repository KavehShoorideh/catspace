"""PLAN 79.45 PART D (SYSTEM_SPEC 11, "One lemma machinery, used twice"; piece: rules; ledger rows 49, 45): THE RECORD OF A LEMMA'S CERTIFICATE BY TRANSLATED CODE.
The certificate of discovery/rule_lemma_certify.py reads 208 fixtures and 20 seeded games with the reader's evaluator. The test rust/engine/game/tests/translated_lemma.rs reads a lemma again with the module the translator made from the composed description, at as many states as the wall allows; this
script holds what is not Rust: it writes the states of the certificate's own games (so that the small run can be held to the certificate's counts), and it writes the record of the large run beside the lemma, as `rules/lemmas/<game>/<name>.translated.certificate.json` (a name that
rule_lemma.load_all does not take for a lemma), naming the key of the lemma's certificate, the kept module it was made with (its hash) and the translator's hash.
  python scripts/rule_lemma_translated.py corpus <game> <name> <out>      the states of the certificate's seeded games, one a line (the fixtures are read by the test from the lemma's folder)
  python scripts/rule_lemma_translated.py record <game> <name> <raw.json>   the test's raw record (LEMMA_RECORD) completed and written beside the lemma
  python scripts/rule_lemma_translated.py status <game> <name>              fresh | stale | missing | failing
This script names no game (a game is an argument)."""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from catspace.incubator.analyst.discovery import gdl_to_rust as T
from catspace.incubator.analyst.discovery import rule_lemma as RL
from catspace.incubator.analyst.discovery import rule_lemma_certify as RC

GEN = ROOT / "rust" / "engine" / "game" / "tests" / "generated"


def record_path(lemma: RL.Lemma) -> Path:
    return RL.lemma_dir(lemma.game) / f"{lemma.name}.translated.certificate.json"


def module_path(lemma: RL.Lemma) -> Path:
    return GEN / f"{lemma.game}_lemma.rs"


def corpus_lines(lemma: RL.Lemma) -> list:
    """The states of the seeded games of the lemma's certificate, as its fixtures are written (the games are those `rule_lemma_certify.certify` plays: same seed, same player)."""
    prog = RL.program(lemma)
    wanted = RC.wanted_of(lemma, prog)
    out = []
    for gi in range(RC.GAMES):
        for _label, state, _db in RC.play(prog, wanted, gi, RC.SEED, RC.PLY_FUSE):
            out.append(RC.state_text(state))
    return out


def current(lemma: RL.Lemma) -> dict:
    """What a record is held to now: the key of the lemma's certificate, the hash of the kept module and of the translator."""
    c = RC.load(lemma)
    mod = module_path(lemma)
    return {"certificate_key": c["key"] if c else None, "module_sha256": hashlib.sha256(mod.read_bytes()).hexdigest() if mod.is_file() else None, "translator_sha256": T.translator_hash()}


def status(lemma: RL.Lemma) -> str:
    p = record_path(lemma)
    if not p.is_file():
        return "missing"
    r = json.loads(p.read_text())
    if r["disagreeing_states"] or r["flag_disagreements"]:
        return "failing"
    now = current(lemma)
    return "fresh" if all(r.get(k) == v for k, v in now.items()) else "stale"


def write_record(lemma: RL.Lemma, raw: dict) -> Path:
    rec = {**raw, "lemma_certificate": f"{lemma.name}.certificate.json", "certificate_counts": None, **current(lemma), "date": datetime.date.today().isoformat()}
    c = RC.load(lemma)
    if c:
        rec["certificate_counts"] = {k: c[k] for k in ("positions", "states_held", "rows_left", "rows_right", "rows_compared", "disagreements")}
    p = record_path(lemma)
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=True) + "\n")
    return p


def main(argv: list) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, game, name = argv[0], argv[1], argv[2]
    lemma = RL.load(game, name)
    if cmd == "corpus":
        lines = corpus_lines(lemma)
        Path(argv[3]).write_text("\n".join(lines) + "\n")
        print(f"OBS 79.45 D corpus of the certificate of {game}/{name}: {len(lines)} states of {RC.GAMES} seeded games (seed {RC.SEED}, fuse {RC.PLY_FUSE}) written to {argv[3]}")
        return 0
    if cmd == "record":
        raw = json.loads(Path(argv[3]).read_text())
        p = write_record(lemma, raw)
        print(f"OBS 79.45 D record of {game}/{name} by translated code written to {p.relative_to(ROOT)}: {raw['states']} states, held {raw['states_held']}, rows compared {raw['rows_compared']}, disagreeing states {raw['disagreeing_states']}")
        return 0
    if cmd == "status":
        s = status(lemma)
        print(f"OBS 79.45 D record of {game}/{name}: {s}")
        return 0 if s == "fresh" else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
