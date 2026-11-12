"""PLAN 78.32 THE EXACT CORPUS AND THE FULL DRILL. At the sizes an exact labeller covers (exact.py) the result class of every position is known and so is, for every legal move, whether it KEEPS the
class. A CORPUS is a large set of distinct positions drawn from generic sources, each with its legal moves labelled keep / lose, a GROUP (the start it came from) and a development / held-out split
BY GROUP. A rule (a lemma in the store's format) is run over the whole corpus by the engine's own matcher on cards (`CardSet.eval` with `all_named`): the FULL DRILL. TRAINING DATA ONLY: the wake, the
snapshot writer and the Rust sources never read it (tests/test_plan78_32_exam_drill_induce.py).

Game-free: the game is the variant string; the starts come from curriculum.py, the labels from an ExactLabeller object, the facts from the store's vocabulary; no game object is named here.

  build_corpus(game, size, *, variant, n, seed, sources, labeller, ...) -> Corpus        (distinct positions, cached in a file under paths.derived)
  Corpus.rows / .part(name) / .cards(spec, vocab_json) / .excluding(keys) / .save(path) / load_corpus(path)
  drill(rule, corpus, *, part) -> {binds, names_a_move, right, wrong, trivial, wrong_examples}       (drill_many / drill_cases: many rules, with the card indices)
     RIGHT   the rule's body binds at a non-trivial position and EVERY move it names keeps the class      WRONG   some move it names loses the class
     TRIVIAL every legal move keeps the class (a position where no rule can be wrong: not counted as right or wrong)      names_a_move: non-trivial positions where it names at least one move
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import curriculum as CU
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.io import paths

FORMAT = "exact-corpus-1"
SOURCES = ("starts", "random", "playouts", "logs")


class Corpus:
    """Distinct labelled positions. `rows[i]` = {start, moves, group, before (the class for the mover), keep {move: bool}}. `groups` = {group: "dev" | "held"}. The cards of the rows are built ONCE per
    (spec, vocabulary) and kept in memory (the engine's object is not serialised): `cards(spec, vocab_json)` -> the CardSet valued 1 for a keeping move and 0 for a losing one; `dev_idx` / `held_idx` /
    `all_idx` are card indices; `row_of[card index]` the row."""

    def __init__(self, rows: list, groups: dict, *, game: str = "", variant: str = "", size: int = 0, seed: int = 0, meta: "dict | None" = None):
        self.rows, self.groups, self.game, self.variant, self.size, self.seed, self.meta = rows, groups, game, variant, size, seed, dict(meta or {})
        self._cards = None
        self._ctx = None
        self._build = None
        self._pos_keys = None
        self.card_of: dict = {}
        self.row_of: list = []

    def __len__(self) -> int:
        return len(self.rows)

    def non_trivial(self, i: int) -> bool:
        k = self.rows[i]["keep"].values()
        return any(k) and not all(k)

    def part_of(self, i: int) -> str:
        return self.groups[self.rows[i]["group"]]

    def part(self, name: str) -> list:
        """Row indices of the part: "dev", "held" or "all"."""
        if name == "all":
            return list(range(len(self.rows)))
        if name not in ("dev", "held"):
            raise ValueError(f"part is 'dev', 'held' or 'all', not {name!r}")
        return [i for i in range(len(self.rows)) if self.part_of(i) == name]

    def excluding(self, keys: set) -> "Corpus":
        """The corpus without the positions whose key ((start, tuple(moves))) is in `keys` (the exam's, so that nothing induced or admitted ever sees them)."""
        rows = [r for r in self.rows if (r["start"], tuple(r["moves"])) not in keys]
        return Corpus(rows, dict(self.groups), game=self.game, variant=self.variant, size=self.size, seed=self.seed, meta={**self.meta, "excluded": len(self.rows) - len(rows)})

    # the cards, once
    def cards(self, spec: dict, vocab_json: str, depth: int = 1, threads: int = 0):
        ctx = (hashlib.sha1((json.dumps(spec, sort_keys=True) + vocab_json).encode()).hexdigest(), depth)
        if self._cards is not None and self._ctx == ctx:
            return self._cards
        items = [(self.variant, r["start"], list(r["moves"]), f"c{i}", r["group"]) for i, r in enumerate(self.rows)]
        cs = CE.build_cards(items, spec, vocab_json, depth, 2_000_000, threads, {}, False)
        ids = [int(json.loads(cs.card(i))["id"][1:]) for i in range(len(cs))]
        vals = [{m: (1 if k else 0) for m, k in self.rows[ri]["keep"].items()} for ri in ids]
        self._cards = cs.revalued(vals) if len(cs) else cs
        self._ctx = ctx
        self._build = (spec, vocab_json, depth, threads)
        self.row_of = ids
        self.card_of = {ri: ci for ci, ri in enumerate(ids)}
        return self._cards

    def position_keys(self) -> set:
        """The kernel's position keys of the rows (computed once, kept up to date by `extend`): a position reached by another move order is the same position."""
        if self._pos_keys is None:
            self._pos_keys = {CE.state(self.variant, r["start"], r["moves"])[4] for r in self.rows}
        return self._pos_keys

    def extend(self, rows: list) -> int:
        """PLAN 78.34: append labelled rows (each {start, moves, group, before, keep}; every group must already be in `groups`). When the cards are built, only the NEW rows' cards are built (valued 1 / 0 as
        `cards`) and merged after the old ones, so the card indices of the old rows stay and nothing is rebuilt (building all 30000 cards takes about a minute). -> the number of rows added."""
        missing = sorted({r["group"] for r in rows} - set(self.groups))
        if missing:
            raise ValueError(f"extend: group(s) {missing} are not in the corpus's groups; add them (with their part) first")
        if not rows:
            return 0
        base = len(self.rows)
        keys = self.position_keys()
        self.rows.extend(rows)
        keys.update(CE.state(self.variant, r["start"], r["moves"])[4] for r in rows)
        if self._cards is None:
            return len(rows)
        spec, vocab_json, depth, threads = self._build
        items = [(self.variant, r["start"], list(r["moves"]), f"c{base + k}", r["group"]) for k, r in enumerate(rows)]
        cs = CE.build_cards(items, spec, vocab_json, depth, 2_000_000, threads, {}, False)
        if len(cs):
            ids = [int(json.loads(cs.card(i))["id"][1:]) for i in range(len(cs))]
            new = cs.revalued([{m: (1 if k else 0) for m, k in self.rows[ri]["keep"].items()} for ri in ids])
            self._cards = CE.CardSet.merge([self._cards, new]) if len(self._cards) else new
            n0 = len(self.row_of)
            self.row_of = self.row_of + ids
            self.card_of.update({ri: n0 + k for k, ri in enumerate(ids)})
        return len(rows)

    def idx(self, part: str) -> list:
        """Card indices of the part (the cards must have been built: `cards`)."""
        want = None if part == "all" else part
        return [ci for ci, ri in enumerate(self.row_of) if want is None or self.part_of(ri) == want]

    # the file
    def save(self, path) -> str:
        starts, sidx = [], {}
        for r in self.rows:
            if r["start"] not in sidx:
                sidx[r["start"]] = len(starts)
                starts.append(r["start"])
        rows = [{"s": sidx[r["start"]], "m": r["moves"], "g": r["group"], "b": r["before"], "k": [m for m, k in r["keep"].items() if k], "l": [m for m, k in r["keep"].items() if not k]} for r in self.rows]
        d = {"format": FORMAT, "game": self.game, "variant": self.variant, "size": self.size, "seed": self.seed, "meta": self.meta, "starts": starts, "groups": self.groups, "rows": rows}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, p)
        return str(p)


def load_corpus(path) -> Corpus:
    d = json.loads(Path(path).read_text())
    assert d.get("format") == FORMAT, f"{path}: not a corpus file"
    rows = [{"start": d["starts"][r["s"]], "moves": r["m"], "group": r["g"], "before": r["b"], "keep": {**{m: True for m in r["k"]}, **{m: False for m in r["l"]}}} for r in d["rows"]]
    return Corpus(rows, d["groups"], game=d["game"], variant=d["variant"], size=d["size"], seed=d["seed"], meta=d.get("meta"))


def default_path(game: str, size: int, n: int, seed: int) -> str:
    return paths.derived(f"exact_corpus_{game}_{size}_{n}_{seed}.json")


def split_groups(groups: list, seed: int, dev_share: float = 0.5) -> dict:
    """{group: "dev" | "held"}: the groups sorted, shuffled by `seed`, the first `dev_share` of them development (at least one of each when there are two or more)."""
    gs = sorted(groups)
    random.Random(seed).shuffle(gs)
    k = int(round(len(gs) * dev_share))
    if len(gs) >= 2:
        k = min(max(k, 1), len(gs) - 1)
    return {g: ("dev" if i < k else "held") for i, g in enumerate(gs)}


def class_counts(rows: list) -> dict:
    return dict(sorted(Counter(r["before"] for r in rows).items()))


# ------------------------------------------------------------------------------------------------------------------------------------ building

def build_corpus(game: str, size: int, *, variant: "str | None" = None, n: int = 30000, seed: int = 1, labeller, sources: tuple = ("starts", "random", "playouts"), logs=(), starts: "list | None" = None,
                 max_starts: int = 400, walk_plies: int = 24, playout_plies: int = 40, dev_share: float = 0.5, path: "str | None" = None, say=print) -> Corpus:
    """Distinct positions (by the kernel's position key) of `size` units from generic sources, each labelled by `labeller` (positions it does not cover are skipped), up to `n`:
      starts     the curriculum's start positions (curriculum.starts)         random    random legal play from a start (every position along the walk, at most `walk_plies` plies)
      playouts   careful playouts (cs_engine.playout_game) from a short random prefix, every position along them          logs      every position of the games of the wake log(s) `logs`
    The groups are the STARTS the positions came from; the dev / held-out split is by group. The sources are drawn round-robin over the starts so every group is represented. `starts` overrides the
    curriculum's. With `path`, the corpus is written there. Prints `OBS corpus: ...`."""
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise ValueError(f"unknown corpus source(s) {unknown}; the sources are {SOURCES}")
    t0 = time.perf_counter()
    variant = variant or game
    starts = list(starts) if starts is not None else CU.starts(game, size, variant=variant, limit=max_starts, seed=seed)
    group_of = {s: f"start{i}" for i, s in enumerate(starts)}
    rng = random.Random(seed)
    rows: list = []
    seen: set = set()

    def add(start, moves: list, group: str) -> bool:
        _f, mover, legal, outcome, key = CE.state(variant, start, moves)
        if mover is None or outcome is not None or not legal or key in seen:
            return False
        seen.add(key)                      # (a position seen and not covered is not asked again either)
        keep = labeller.moves_keep(variant, start, moves)
        if keep is None:
            return False
        before = labeller.result(variant, start, moves)
        if before is None:
            return False
        rows.append({"start": start, "moves": list(moves), "group": group, "before": before, "keep": keep})
        return True

    def walk(start, group: str, plies: int, prefix: "list | None" = None) -> list:
        moves = list(prefix or [])
        for _ in range(plies):
            _f, mover, legal, outcome, _k = CE.state(variant, start, moves)
            if mover is None or outcome is not None or not legal:
                break
            moves.append(rng.choice(sorted(legal)))
            add(start, moves, group)
        return moves

    if "starts" in sources:
        for s in starts:
            add(s, [], group_of[s])
    if "logs" in sources:
        for lg in ([logs] if isinstance(logs, (str, Path)) else list(logs)):
            _d, games = SL.read_log(lg)
            for (_run, _g), row in sorted(games.items()):
                st = row.get("start")
                grp = group_of.setdefault(st, f"start{len(group_of)}")
                for ply in range(len(row["moves"])):
                    add(st, row["moves"][:ply], grp)
    stale, rounds = 0, 0
    active = [s for s in ("random", "playouts") if s in sources]
    while active and len(rows) < n and stale < 60:
        rounds += 1
        before_n = len(rows)
        for s in starts:
            g = group_of[s]
            if "random" in active:
                walk(s, g, rng.randint(2, walk_plies))
            if "playouts" in active:
                pre = walk(s, g, rng.randint(0, 6))
                try:
                    pg = CE.playout_game(variant, s, pre, seed=rng.randrange(1 << 30), max_plies=playout_plies, rollout="careful", store=None, spec=None, worth={}, valuation="")
                except Exception:
                    continue
                for i in range(1, len(pg["moves"]) + 1):
                    add(s, pre + pg["moves"][:i], g)
            if len(rows) >= n:
                break
        stale = stale + 1 if len(rows) == before_n else 0
    rows = rows[:n] if len(rows) > n else rows
    groups = split_groups(sorted({r["group"] for r in rows}), seed, dev_share)
    c = Corpus(rows, groups, game=game, variant=variant, size=size, seed=seed, meta={"n_target": n, "sources": list(sources), "labeller": getattr(labeller, "source", "exact"), "rounds": rounds})
    nt = sum(1 for i in range(len(rows)) if c.non_trivial(i))
    secs = time.perf_counter() - t0
    say(f"OBS corpus: positions {len(rows)}, groups {len(groups)}, non-trivial {nt} (some move keeps and some loses), by result class {class_counts(rows)}, {secs:.0f} seconds"
        f" (dev groups {sum(1 for v in groups.values() if v == 'dev')}, held-out groups {sum(1 for v in groups.values() if v == 'held')})")
    if path:
        c.save(path)
        getattr(labeller, "save", lambda: None)()
    return c


def ensure_corpus(game: str, size: int, *, variant: "str | None" = None, n: int = 30000, seed: int = 1, labeller, path: "str | None" = None, say=print, **kw) -> Corpus:
    """The corpus file at `path` (default: default_path) when it exists, else built and written there."""
    path = path or default_path(game, size, n, seed)
    if Path(path).exists():
        c = load_corpus(path)
        nt = sum(1 for i in range(len(c)) if c.non_trivial(i))
        say(f"OBS corpus: positions {len(c)}, groups {len(c.groups)}, non-trivial {nt} (some move keeps and some loses), by result class {class_counts(c.rows)}, 0 seconds (read from {path})")
        return c
    return build_corpus(game, size, variant=variant, n=n, seed=seed, labeller=labeller, path=path, say=say, **kw)


# ------------------------------------------------------------------------------------------------------------------------------------ the full drill

def _tally(ev: dict, corpus: Corpus) -> dict:
    binds = ev["fires"] + ev["trivial"]
    right_at, wrong_at, names = [], [], 0
    for ci, holds, k in ev["cases"]:
        if k == 0:
            continue
        names += 1
        (right_at if holds else wrong_at).append(ci)
    return {"binds": binds, "names_a_move": names, "right": len(right_at), "wrong": len(wrong_at), "trivial": ev["trivial"], "truncated": ev["truncated"], "right_at": right_at, "wrong_at": wrong_at}


def drill_many(rules: list, corpus: Corpus, *, part: str = "dev", indices: "list | None" = None, max_bindings: int = 256) -> list:
    """`rules` (lemma dicts) over the corpus part (or only the card `indices`) by the engine's matcher with all_named. One dict per rule: {binds, names_a_move, right, wrong, trivial, truncated,
    right_at, wrong_at} (card indices). `binds` = positions where the body binds, trivial ones included; the others are counted on the non-trivial ones. The corpus's cards must be built
    (`corpus.cards(spec, vocab_json)`)."""
    if corpus._cards is None:
        raise ValueError("build the corpus's cards first: corpus.cards(spec, vocab_json)")
    idx = indices if indices is not None else corpus.idx(part)
    if not rules:
        return []
    evs = corpus._cards.eval_many([json.dumps(r) for r in rules], idx, max_bindings, None, True)
    return [_tally(e, corpus) for e in evs]


def drill_cases(rule: dict, corpus: Corpus, *, part: str = "dev", max_bindings: int = 256) -> dict:
    return drill_many([rule], corpus, part=part, max_bindings=max_bindings)[0]


def drill(rule: dict, corpus: Corpus, *, part: str = "dev", examples: int = 5, max_bindings: int = 256) -> dict:
    """The full drill of one rule on the corpus part -> {binds, names_a_move, right, wrong, trivial, wrong_examples: [{start, moves, group, losing}] (at most `examples`; `losing` = the moves of the
    position that lose the class), truncated}."""
    t = drill_cases(rule, corpus, part=part, max_bindings=max_bindings)
    ex = []
    for ci in t["wrong_at"][:examples]:
        r = corpus.rows[corpus.row_of[ci]]
        ex.append({"start": r["start"], "moves": list(r["moves"]), "group": r["group"], "losing": sorted(m for m, k in r["keep"].items() if not k)})
    return {k: t[k] for k in ("binds", "names_a_move", "right", "wrong", "trivial", "truncated")} | {"wrong_examples": ex}
