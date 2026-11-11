"""PLAN 78.23 A RANK SOURCE AS THE JUDGE OF THE CARDS. A lemma claims "an optimal move of this kind exists"; optimal is by a judge. When the store's material table cannot tell moves
apart (in quiet endgames every move ties), the oracle itself judges: each card's moves are revalued by a rank source's own scores, graded into integers, and the rest of the sleep
(confirmation, climb, verification) reads the revalued cards unchanged. The source stays in the training kit (the lemma is marked teacher-judged; nothing here runs at inference).

    grade(score) -> int     how a source's scores become card values (an optional method of the source; `default_grade` otherwise): moves the source cannot tell apart share a value
    revalue(...)            -> (a CardSet with the source's values, the report)
    regret_seeds(...)       the regret read DIRECTLY on the revalued card: the played move is strictly worse than the best; the saving moves are the optimal ones, the source's top first
    consequence(...)        a source's line replayed through the kernel: where it ends and where the units first change (the evidence of WHY a move is bad)
    describe(source)        the source's own parameters that matter, recorded on every lemma it judged
    Cached(source, path)    any rank source remembered on disk: the same position asked twice gets the same answer (an outside engine on a clock is not repeatable otherwise)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cs_engine as CE
from catspace.incubator.analyst.discovery import curriculum as CU
from catspace.incubator.analyst.discovery import detectors as D


def default_grade(score) -> int:
    """The grade of a source with no `grade`: an int as it is, a float rounded."""
    return score if isinstance(score, int) and not isinstance(score, bool) else int(round(score))


def grader(source):
    """The source's own `grade` when it declares one, else `default_grade`."""
    g = getattr(source, "grade", None)
    return g if callable(g) else default_grade


def describe(source) -> dict:
    """What of the source matters to the claim it judged: its name, whether it is training-only, and the parameters it declares (movetime, gap, options, path)."""
    out = {"source": getattr(source, "source", "teacher"), "training_only": bool(getattr(source, "training_only", True))}
    for k in ("movetime", "gap", "mate_bucket", "grade_mode", "decisive", "options", "seconds", "path"):
        v = getattr(source, k, None)
        if v is not None:
            out[k] = v
    return out


class Cached:
    """A rank source remembered on disk (plan 78.24): `rank` and `lines` of the wrapped source, answered from a JSON file when the same (variant, start, moves, cap) was asked
    before, so a sleep, its audit and a re-run all read ONE verdict per position (an engine on a clock answers differently run to run). Everything else (`source`,
    `training_only`, `grade`, `gap`, `mate_bucket`, ...) is the wrapped source's own, so the grading can change without asking again. `lines` exists only when the source has it;
    a miss asks the source for both in one go (a source that shares one analysis between them, like the UCI one, pays once). `save()` writes the file (atomically); `close()`
    saves and closes the source. `asked` and `hits` count the questions and the ones answered from the file."""

    def __init__(self, source, path, every: int = 200):
        self.inner, self.cache_path, self.every = source, Path(path), every
        self.memory = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.asked = self.hits = self.unsaved = 0
        if callable(getattr(source, "lines", None)):
            self.lines = self._lines

    def __getattr__(self, name):                              # (only what this object does not define: the wrapped source's own attributes)
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _entry(self, variant: str, moves: list, cap, start) -> dict:
        key = json.dumps([variant, start, list(moves), cap])
        self.asked += 1
        e = self.memory.get(key)
        if e is not None:
            self.hits += 1
            return e
        lines = getattr(self.inner, "lines", None)
        e = {"rank": [[m, s] for m, s in self.inner.rank(variant, moves, cap=cap, start=start)],
             "lines": lines(variant, moves, cap=cap, start=start) if callable(lines) else None}
        self.memory[key] = e
        self.unsaved += 1
        if self.every and self.unsaved >= self.every:
            self.save()
        return e

    def rank(self, variant: str, moves: list, cap: "int | None" = None, start: "str | None" = None) -> list:
        return [(m, s) for m, s in self._entry(variant, moves, cap, start)["rank"]]

    def _lines(self, variant: str, moves: list, cap: "int | None" = None, start: "str | None" = None) -> dict:
        return dict(self._entry(variant, moves, cap, start)["lines"] or {})

    def save(self) -> None:
        if not self.unsaved:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.memory))
        os.replace(tmp, self.cache_path)
        self.unsaved = 0

    def close(self) -> None:
        self.save()
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def consequence(variant: str, start: "str | None", moves: list, line: list, codec=None) -> dict:
    """Replay `line` from the position (`start`, `moves`) through the kernel, stopping at the first illegal move, at the end of the game or at the line's end. The units are the
    codec's (`curriculum.codec_of(variant)` by default): their count, and, when the codec offers `kinds`, the multiset of their kinds. -> {plies: moves replayed, outcome: the
    game's result if it ended (else None), stopped: "line end" | "game over" | "illegal", units_before, units_end, first_change: {ply (0 = the line's first move), move,
    units} where the units first change (count, or kinds when the codec has them) or None}."""
    codec = codec or CU.codec_of(variant)

    def seen(position: str):
        kinds = getattr(codec, "kinds", None)
        return (sorted(kinds(position)) if kinds else None), len(codec.units(position))

    pos, _mover, legal, outcome, _key = CE.state(variant, start, list(moves))
    before = now = seen(pos)
    out = {"plies": 0, "outcome": outcome, "stopped": "line end" if outcome is None else "game over", "units_before": before[1], "units_end": before[1], "first_change": None}
    for k, move in enumerate(line):
        if outcome is not None:
            out["stopped"] = "game over"
            break
        if move not in legal:
            out["stopped"] = "illegal"
            break
        pos, _mover, legal, outcome, _key = CE.state(variant, start, list(moves) + list(line[:k + 1]))
        now = seen(pos)
        out["plies"] = k + 1
        out["units_end"] = now[1]
        if out["first_change"] is None and now != before:
            out["first_change"] = {"ply": k, "move": move, "units": now[1]}
        out["outcome"] = outcome
    else:
        out["stopped"] = "game over" if outcome is not None else "line end"
    return out


def say_consequence(c: dict) -> str:
    """A few words for a `consequence` (for the OBS lines)."""
    bits = [f"{c['plies']} plies"]
    fc = c["first_change"]
    if fc:
        bits.append(f"units {c['units_before']}->{fc['units']} at ply {fc['ply']} ({fc['move']})")
    if c["outcome"] is not None:
        bits.append(f"game over {c['outcome']}")
    if c["stopped"] == "illegal":
        bits.append("line illegal")
    return ", ".join(bits)


def revalue(cards, positions: dict, source, variant: str, indices: "list | None" = None, *, cap: "int | None" = None, say=None, context: "D.Context | None" = None) -> tuple:
    """Ask `source` once per card of `indices` (all when None), grade each move's score, and return (`cards.revalued(...)`, report). `positions` {card id: (start, moves)}.
    A card the source cannot rank completely (a legal move left out, or no position known) is cut; a card already cut by the store's reading is left as it is (not asked).
    `context` (a detectors.Context) shares its `rankings` cache, so a detector and the judge never ask the same position twice. Cards outside `indices` are unchanged.
    report: {source, asked, revalued, cut, skipped_cut, seconds, optimal_share (mean over revalued cards of |optimal| / |moves|: the chance figure), moves_mean}."""
    t0 = time.perf_counter()
    grade = grader(source)
    name = getattr(source, "source", "teacher")
    todo = list(range(len(cards))) if indices is None else list(indices)
    values: list = [None] * len(cards)
    rep = {"source": name, "asked": 0, "revalued": 0, "cut": 0, "skipped_cut": 0, "cards": len(todo)}
    shares, sizes = [], []
    for i in todo:
        card = json.loads(cards.card(i))
        if card["cut"]:
            rep["skipped_cut"] += 1
            continue
        cid = card["id"]
        ranking = None
        if cid in positions:
            key = (id(source), cid)
            cache = context.rankings if context is not None else {}
            if key not in cache:
                start, moves = positions[cid]
                cache[key] = D.ask(source, variant, moves, start, cap=cap)
                rep["asked"] += 1
            ranking = cache[key]
            if context is not None and callable(getattr(source, "lines", None)) and key not in context.continuations:
                start, moves = positions[cid]
                context.continuations[key] = source.lines(variant, list(moves), **({"start": start} if start is not None else {}))          # (a source answers from the analysis `rank` just ran)
        legal = [m["name"] for m in card["moves"]]
        score = {m: sc for m, sc in (ranking or [])}
        if not legal or any(m not in score for m in legal):
            values[i] = {}                                               # cannot be judged completely: Rust cuts the card (a move without a value is never guessed)
            rep["cut"] += 1
            continue
        values[i] = {m: int(grade(score[m])) for m in legal}
        best = max(values[i].values())
        shares.append(sum(1 for v in values[i].values() if v == best) / len(legal))
        sizes.append(len(legal))
        rep["revalued"] += 1
    rep["seconds"] = round(time.perf_counter() - t0, 3)
    rep["optimal_share"] = round(sum(shares) / len(shares), 4) if shares else None
    rep["moves_mean"] = round(sum(sizes) / len(sizes), 2) if sizes else None
    if say is not None:
        say(f"OBS sleep judge {name}: asked {rep['asked']}, revalued {rep['revalued']}, cut {rep['cut']}, skipped (already cut) {rep['skipped_cut']}, optimal share of the moves "
            f"{rep['optimal_share']} (chance), moves per card {rep['moves_mean']}, {rep['seconds']} s")
    return cards.revalued(values), rep


def _why(ctx: "D.Context", source, sur: dict) -> "dict | None":
    """The evidence of a confirmed regret when the source offers `lines`: {played, refutation (the source's line after the played move), refutation_end (its `consequence`),
    best, best_line, best_end}. None when the source has no `lines` or no line for either move."""
    if not callable(getattr(source, "lines", None)):
        return None
    key = (id(source), sur["id"])
    if key not in ctx.continuations:
        ctx.continuations[key] = source.lines(ctx.variant, list(sur["moves_before"]), **({"start": sur["start"]} if sur["start"] is not None else {}))
    lines = ctx.continuations[key]
    played, best = sur["played"], sur["source_top"]
    if played not in lines or best not in lines:
        return None
    return {"played": played, "refutation": list(lines[played]), "refutation_end": consequence(ctx.variant, sur["start"], sur["moves_before"], lines[played]),
            "best": best, "best_line": list(lines[best]), "best_end": consequence(ctx.variant, sur["start"], sur["moves_before"], lines[best])}


def regret_seeds(ctx: "D.Context", source, rep: dict) -> list:
    """THE REGRET READ DIRECTLY on the revalued cards of `ctx.cards` (the source is the judge: no deeper card). The surprises are `detectors.Regret`'s (the played move worse than
    the source's best by its own `gap`, by the shared ranking cache); one is CONFIRMED when the played move's value is strictly below the card's best, TIED when it is not
    (the grading put them together), UNCONFIRMED when the played move is not on the card, CUT when the card is. A confirmed one seeds from the card's optimal moves, the source's
    top first, then by fewer units lost, then name (at most `max_saving`). Fills rep["regret"]. -> seeds [(x, card, move, source's top, played, None)]."""
    det = D.Regret(source)
    found = det.detect(ctx)
    r = rep["regret"]
    k = ctx.settings.max_saving
    seeds = []
    for sur in found:
        card = ctx.card(sur["card"])
        if card["cut"]:
            r["cut"].append(sur["id"])
            continue
        played = next((m for m in card["moves"] if m["name"] == sur["played"]), None)
        if played is None:
            r["unconfirmed"].append(sur["id"])
            continue
        best = max(m["value"] for m in card["moves"])
        if played["value"] >= best:
            ev = sur["regret"]
            r["tied"].append(sur["id"])
            r["open"].append({"id": sur["id"], "played": sur["played"], "source_top": sur["source_top"], "played_score": ev["played_score"], "top_score": ev["top_score"],
                              "gap": ev["gap"], "rule": ev["rule"], "source": sur["source"]})
            continue
        r["confirmed"].append(sur["id"])
        top = sur["source_top"]
        why = _why(ctx, source, sur)
        optimal = sorted((m for m in card["moves"] if m["value"] == best), key=lambda m: (m["name"] != top, m["lost"], m["name"]))
        x = {"id": sur["id"], "group": sur["group"], "played": sur["played"], "reply": top, "predicted": 0, "seen": 0, "kind": "regret", "regret": sur["regret"], **({"forced": True} if sur["id"] in ctx.force_ids else {}), **({"why": why} if why else {})}
        if why:
            r.setdefault("confirmed_rows", []).append({"id": sur["id"], **why})
        seeds.extend((x, card, m, top, played, None) for m in optimal[:k])
    return seeds
