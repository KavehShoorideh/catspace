"""PLAN 78.14 THE GAME STORE: what the engine has LEARNED about a game, kept where a person can read it, and published as immutable snapshots the Rust core reads.

  rules/learned/<game>/vocabulary.json      the game's FACT VOCABULARY (rules ingestion's second half): its description (and, per variant, the delta that composes it),
                                            the derived relations that belong to the fact layer, the relations a move's effect is read in, how a state holds its units.
  rules/learned/<game>/vocabulary.<name>.json  (plan 78.24) a NAMED vocabulary: the same keys plus `concepts` (rules-delta path(s) composed onto the description); a lemma records
                                            the vocabulary it was written in (`meta.vocabulary`) and a snapshot of that vocabulary holds only those.
  rules/learned/<game>/lemmas/<name>.json   one learned lemma per file (the source of truth, committed): its body and response in the lemma language
                                            (rust/engine/memory/src/lemma.rs), what verification measured, where it came from.
  data/derived/snapshots/<game>/<id>.json   a SNAPSHOT: vocabulary + judge + lemmas + parent, immutable, named by the first 16 hex digits of the SHA-256 of its bytes
                                            (the Rust core refuses a file that does not hash to its name). `HEAD` holds the id of the latest.

The VALUATION stays the legacy store's entry (lemma_store.judge_of); a snapshot copies it in as its judge, nothing here replaces it. A learned lemma is a PRIOR:
the engine reads the moves it proposes first, and the search's values are untouched (AGENTS.md rule 7: subgoals enter through the prior, never the value). Its
status is SAMPLED -- verified exactly on a sample of positions, not proved; Lean stays the source of truth for what is proved (rule 11).

  .venv/bin/python -m catspace.incubator.analyst.discovery.game_store publish chess      # publish the next snapshot of the game's lemmas
  .venv/bin/python -m catspace.incubator.analyst.discovery.game_store status chess
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import board_structure as BS
from catspace.io import paths

ROOT = paths.REPO_ROOT
RULES = ROOT / "rules"


# Where the learned sources and the snapshots live; a test points them at a temporary directory so it never touches the real store.
LEARNED_ROOT: "Path | None" = None
SNAPSHOT_ROOT: "Path | None" = None


def learned_dir(game: str) -> Path:
    return (LEARNED_ROOT or RULES / "learned") / game


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "-", name.lower()).strip("-")


def vocabulary_file(game: str, name: "str | None" = None) -> dict:
    """The vocabulary file of the game: `vocabulary.json` (name None, the default) or `vocabulary.<name>.json` (plan 78.24: a NAMED lemma vocabulary, the same keys plus an optional
    `concepts`: a rules-delta path, or a list of them, relative to rules/, composed in order onto the description)."""
    p = learned_dir(game) / ("vocabulary.json" if name is None else f"vocabulary.{name}.json")
    if not p.is_file():
        raise FileNotFoundError(f"{p}: the game {game} has no fact vocabulary" + ("" if name is None else f" named {name}") + f" ({p.name} under rules/learned/<game>/)")
    return json.loads(p.read_text())


def vocabulary(game: str, name: "str | None" = None) -> dict:
    """The vocabulary as the Rust core reads it: {derived, delta, units}."""
    v = vocabulary_file(game, name)
    return {"derived": list(v.get("derived", [])) + [r for r in BS.relation_names(v) if r not in v.get("derived", [])], "delta": list(v.get("delta", [])), "units": v.get("units")}


_vocabulary = vocabulary          # (`build` has a parameter of the same name)


def concept_deltas(v: dict) -> list:
    """The rules-delta paths a vocabulary's `concepts` names, in order (none, one path, or a list)."""
    c = v.get("concepts")
    return [] if not c else ([c] if isinstance(c, str) else list(c))


def spec_texts(game: str, name: "str | None" = None) -> dict:
    """{variant: description text} for the variants of the game that have a description: the base text, composed with the variant's delta where it has one, then with the
    named vocabulary's concept deltas in order (none for the default vocabulary unless its file says so)."""
    v = vocabulary_file(game, name)
    base = (RULES / v["spec"]).read_text()
    out = {}
    for variant, delta in v["variants"].items():
        text = base if delta is None else CE.compose(base, (RULES / delta).read_text())
        for c in concept_deltas(v):
            text = CE.compose(text, (RULES / c).read_text())
        if v.get("axes"):                                    # plan 78.33: the board's structure (edge and goal distances of each axis), generated from the description
            text = CE.compose(text, BS.structure_delta(text, v))
        out[variant] = text
    return out


def graph_vocabulary(game: str) -> dict:
    """The game's GRAPH VOCABULARY (plan 78.21; rules/learned/<game>/graph_vocabulary.json): what a graph reader sees -- state, derived and static relations, the
    product sorts it fuses into objects, the numeric scales -- and the concepts delta that carries the human vocabulary."""
    p = learned_dir(game) / "graph_vocabulary.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p}: the game {game} has no graph vocabulary (rules/learned/<game>/graph_vocabulary.json)")
    return json.loads(p.read_text())


def graph_spec_texts(game: str) -> dict:
    """{variant: description text} as `spec_texts`, each then composed with the graph vocabulary's `concepts` delta when it names one."""
    texts = spec_texts(game)
    concepts = graph_vocabulary(game).get("concepts")
    if not concepts:
        return texts
    delta = (RULES / concepts).read_text()
    return {variant: CE.compose(text, delta) for variant, text in texts.items()}


# ---------------------------------------------------------------------------------------------------------------- lemma sources

def lemma_dir(game: str) -> Path:
    return learned_dir(game) / "lemmas"


def check_lemma(lemma: dict) -> None:
    """The engine's own reading of the lemma (raises ValueError if it is not well formed)."""
    CE.check_lemma(json.dumps(lemma))


def write_lemma(game: str, lemma: dict) -> Path:
    check_lemma(lemma)
    d = lemma_dir(game)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{slug(lemma['name'])}.json"
    p.write_text(json.dumps(lemma, indent=1, sort_keys=True) + "\n")
    return p


ANY = object()          # `read_lemmas(vocabulary=ANY)`: every lemma of the game, whatever vocabulary it was written in


def read_lemmas(game: str, vocabulary: "str | None | object" = None) -> list:
    """The game's lemma sources. `vocabulary` selects the vocabulary they were written in (`meta.vocabulary`; absent or None = the default one): None (the default argument)
    returns the lemmas of the DEFAULT vocabulary, which for a store with only default lemmas is all of them, as before; a name returns that vocabulary's; `ANY` returns all."""
    d = lemma_dir(game)
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        e = json.loads(p.read_text())
        if p.stem != slug(e["name"]):
            raise ValueError(f"{p.name}: the file is named {slug(e['name'])}.json by the lemma's name")
        check_lemma(e)
        if vocabulary is ANY or (e.get("meta") or {}).get("vocabulary") == vocabulary:
            out.append(e)
    return out


# ---------------------------------------------------------------------------------------------------------------- snapshots

def snapshot_dir(game: str) -> Path:
    if SNAPSHOT_ROOT is not None:
        d = SNAPSHOT_ROOT / game
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(paths.derived("snapshots", game, ".keep")).parent


def judge_of(game: str, name: "str | None" = None) -> "dict | None":
    """The legacy store's valuation record in use for `game` -- or the record `name` (a comparison, or the judge a sleep learns by) -- as the snapshot's judge
    (None: the engine's built-in count). Values come from the store's records only, never from code."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    j = LS.judge_of(game, name)
    return None if not j else {"name": j["valuation"], "worth": {k: int(v) for k, v in j["worth"].items()}, "ours_first": bool(j["ours_first"])}


def build(game: str, *, parent: "str | None" = None, teacher_used: bool = False, judge: "dict | None | str" = "store", vocabulary: "str | None" = None) -> bytes:
    """The canonical bytes of the game's next snapshot: its vocabulary, judge and lemma sources. Canonical: sorted keys, no spaces, so the same content is the same
    bytes and the same id. `vocabulary` (plan 78.24) names the lemma vocabulary: the snapshot carries it and only the lemmas written in it (a lemma of another vocabulary would
    never fire); None is the default vocabulary, byte for byte as before."""
    lemmas = read_lemmas(game, vocabulary)
    snap = {"format": 1, "game": game, "parent": parent, "vocabulary": _vocabulary(game, vocabulary),
            "judge": judge_of(game) if judge == "store" else judge, "teacher_used": bool(teacher_used) or any((l.get("meta") or {}).get("teacher") for l in lemmas),
            "lemmas": lemmas}
    return json.dumps(snap, sort_keys=True, separators=(",", ":")).encode()


def head(game: str) -> "str | None":
    p = snapshot_dir(game) / "HEAD"
    return p.read_text().strip() if p.is_file() else None


def path_of(game: str, snapshot_id: str) -> Path:
    return snapshot_dir(game) / f"{snapshot_id}.json"


def head_judge(game: str) -> "dict | None | str":
    """The valuation record the game's HEAD snapshot carries (the record the wake now plays by), for a snapshot built on top of it: a rule added at sleep must be tried and published
    under the same record the wake reads positions by. "store" where there is no HEAD yet (the store's record in use)."""
    cur = head(game)
    if not cur or not path_of(game, cur).is_file():
        return "store"
    return json.loads(path_of(game, cur).read_bytes()).get("judge")


def publish(game: str, **kw) -> str:
    """Publish the next snapshot of `game` (its parent the current HEAD) and make it HEAD -> its id. The same content publishes to the same id (nothing new is written);
    the file is read back the way the engine reads it, and refused if it does not hash to its name."""
    data = build(game, parent=head(game), **kw)
    sid = CE.content_id(data)
    p = path_of(game, sid)
    # the same lemmas and judge under another parent are the same state of the store: do not grow the chain for nothing
    cur = head(game)
    if cur and path_of(game, cur).is_file():
        old = json.loads(path_of(game, cur).read_bytes())
        new = json.loads(data)
        if {k: v for k, v in old.items() if k != "parent"} == {k: v for k, v in new.items() if k != "parent"}:
            return cur
    if not p.is_file():
        p.write_bytes(data)
    info = CE.snapshot_info(str(p))
    assert info["id"] == sid, (info, sid)
    (snapshot_dir(game) / "HEAD").write_text(sid + "\n")
    return sid


def info(game: str, snapshot_id: "str | None" = None) -> dict:
    sid = snapshot_id or head(game)
    if sid is None:
        raise FileNotFoundError(f"{game}: no snapshot published")
    return CE.snapshot_info(str(path_of(game, sid)))


def main(argv: list) -> int:
    if len(argv) != 2 or argv[0] not in ("publish", "status"):
        print(__doc__)
        return 2
    game = argv[1]
    if argv[0] == "publish":
        sid = publish(game)
        print(f"OBS snapshot {game} {sid}: {len(info(game, sid)['lemmas'])} lemma(s), parent {info(game, sid)['parent']}")
    else:
        i = info(game)
        print(f"OBS snapshot {game} HEAD {i['id']}: {len(i['lemmas'])} lemma(s) {i['lemmas']}, parent {i['parent']}, teacher used {i['teacher_used']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
