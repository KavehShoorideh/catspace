"""PLAN 78.24 B THE AUDIT OF APPLICATION: lemma by lemma, how a lemma was MINED (`mined`, `render`) and how it was APPLIED in the games played with the snapshot that holds it (`audit`).

How a lemma acts in a wake decision (read in rust/engine/memory/src/op.rs and checked by running): `play_self(..., store=<snapshot>, specs=<spec_texts>)` is enough; the lemma
operator then matches every lemma on the position, and the ledger prefers the moves they propose (`prior` / `ties` only add the ledger's ties). The decision's log row records
the `lemma` step (candidates = how many moves were proposed in all), a `fired` fact `[name, bindings, moves proposed]` per lemma that bound, and the evidence line
`lemma <name>: <n> binding(s), <m> move(s) proposed` in `trace`. It does NOT record WHICH moves: the audit recovers the names with the ENGINE's own matcher: one card of the
position (`CE.build_cards`, the snapshot's vocabulary), revalued one-hot per move (`CardSet.revalued`), then `CardSet.eval` of the lemma: its single case says `holds` exactly when
the lemma's response matches that move (holds = a matched move is optimal, and the one-hot makes only that move optimal). No Python re-implementation of the matcher.
The recomputed count is compared with the row's own `moves proposed` (`mismatch` in the report).

A rank source (`rank(variant, moves, cap=None, start=None)`, graded by `judges.grader`) says which moves are optimal: the best grade. Training-side check only.

  python -m catspace.incubator.analyst.discovery.apply_audit chess --log <wake.jsonl> [--snapshot <path>] [--teacher <uci> --teacher-ms 100 --teacher-option NAME=VALUE] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import sys

import cs_engine as CE

from catspace.incubator.analyst.discovery import detectors as D
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import judges as JG
from catspace.incubator.analyst.discovery import sleep as SL


# ---------------------------------------------------------------------------------------------------------------- how a lemma was mined

def literal(lit: dict) -> str:
    return ("not " if lit.get("neg") else "") + f"{lit['relation']}({', '.join(lit['args'])})"


def render(lemma: dict) -> str:
    """A readable lemma: `IF <body> THEN a best move removes [...] adds [...]`, its stats (dev / held fires, precision, excess), its judge, and its `meta.why` when present."""
    resp = lemma["response"]
    out = [f"IF {' AND '.join(literal(l) for l in lemma['body']) or 'true'} THEN a best move removes [{', '.join(literal(l) for l in resp.get('removed', []))}] "
           f"adds [{', '.join(literal(l) for l in resp.get('added', []))}]"]
    st = lemma.get("stats") or {}
    bits = []
    for part in ("dev", "held"):
        e = st.get(part)
        if e:
            bits.append(f"{part}: {e.get('fires')} fires, {e.get('holds')} held, precision {e.get('precision')}, excess {e.get('excess')}")
    if bits:
        out.append("    stats " + "; ".join(bits))
    if st.get("judge"):
        out.append(f"    judge {st['judge']}")
    why = (lemma.get("meta") or {}).get("why")
    if why:
        out.append(f"    why: played {why['played']}; the refutation {' '.join(why['refutation'][:8])} ({JG.say_consequence(why['refutation_end'])}); "
                   f"best {why['best']}: {' '.join(why['best_line'][:8])} ({JG.say_consequence(why['best_end'])})")
    return "\n".join(out)


def mined(game: str, vocabulary: "str | None" = None) -> list:
    """Every lemma of `vocabulary` in the store, rendered (the store is whatever `game_store` points at; None is the default vocabulary, a name that vocabulary's lemmas)."""
    return [f"{l['name']}: {render(l)}" for l in GS.read_lemmas(game, vocabulary)]


# ---------------------------------------------------------------------------------------------------------------- how it was applied

def proposals(card_set, lemma: dict) -> list:
    """The moves of the ONE card in `card_set` that `lemma`'s response matches, by the engine's matcher: one-hot revalue per move, `eval`'s case says `holds` iff that move matched."""
    card = json.loads(card_set.card(0))
    names = [m["name"] for m in card["moves"]]
    text = json.dumps(lemma)
    out = []
    for n in names:
        ev = card_set.revalued([{m: (1 if m == n else 0) for m in names}]).eval(text, None, 64)
        if any(holds for _ci, holds, _k in ev["cases"]):
            out.append(n)
    return out


def _snapshot(cache: dict, game: str, row_id: str, snapshot: "str | None") -> dict:
    key = snapshot or row_id
    if key not in cache:
        path = snapshot or str(GS.path_of(game, row_id))
        cache[key] = json.loads(open(path, "rb").read())
    return cache[key]


def optimal_of(source, variant: str, start, moves: list) -> "tuple | None":
    """(the set of optimal moves, {move: grade}) by the source's graded ranking; None when it ranks nothing."""
    ranking = D.ask(source, variant, moves, start)
    if not ranking:
        return None
    grade = JG.grader(source)
    g = {m: int(grade(s)) for m, s in ranking}
    best = max(g.values())
    return {m for m, v in g.items() if v == best}, g


def audit(game: str, log: str, *, snapshot: "str | None" = None, variant: "str | None" = None, source=None, vocabulary: "str | None" = None, limit: "int | None" = None,
          say=print) -> dict:
    """The audit of a wake log played with a snapshot that holds lemmas -> {overall, lemmas {name: summary}, decisions [..], examples}. See the module."""
    variant = variant or game
    decisions, games = SL.read_log(log)
    specs = GS.spec_texts(game, vocabulary)
    cache: dict = {}
    src_name = getattr(source, "source", "teacher") if source is not None else None
    rows, per, examples, mismatch, touched = [], {}, {}, 0, set()
    for (run, g, ply), d in sorted(decisions.items()):
        fired = [f for f in (d["row"].get("facts") or []) if f.get("relation") == "fired"]
        if not fired:
            continue
        if limit is not None and len(rows) >= limit:
            break
        game_row = games.get((run, g))
        if game_row is None:
            continue
        touched.add((run, g))
        snap = _snapshot(cache, game, d["row"]["snapshot"], snapshot)
        by_name = {l["name"]: l for l in snap["lemmas"]}
        start, moves = game_row.get("start"), list(game_row["moves"][:ply])
        cid = f"{run}:{g}:{ply}"
        cs = CE.build_cards([(variant, start, moves, cid, f"{run}:{g}")], specs, json.dumps(snap["vocabulary"]), 1)
        legal = [m["name"] for m in json.loads(cs.card(0))["moves"]]
        played = d["move"]
        opt = optimal_of(source, variant, start, moves) if source is not None else None
        entry = {"id": cid, "fen": CE.state(variant, start, moves)[0], "start": start, "moves": moves, "played": played, "legal": len(legal),
                 "source_best": sorted(opt[0]) if opt else None, "played_optimal": (played in opt[0]) if opt else None, "lemmas": []}
        for f in fired:
            name, bindings, count = f["args"][0], int(f["args"][1]), int(f["args"][2])
            prop = proposals(cs, by_name[name])
            if len(prop) != count:
                mismatch += 1
            item = {"name": name, "bindings": bindings, "engine_count": count, "proposed": prop, "followed": played in prop,
                    "proposal_optimal": (bool(set(prop) & opt[0]) if opt else None), "proposed_optimal": ([m for m in prop if m in opt[0]] if opt else None)}
            entry["lemmas"].append(item)
            s = per.setdefault(name, {"fired": 0, "games": set(), "proposed": 0, "followed": 0, "scored": 0, "proposal_optimal": 0, "followed_n": 0, "followed_optimal": 0,
                                      "not_followed_n": 0, "not_followed_optimal": 0})
            s["fired"] += 1
            s["games"].add((run, g))
            s["proposed"] += len(prop)
            s["followed"] += item["followed"]
            if opt:
                s["scored"] += 1
                s["proposal_optimal"] += bool(item["proposal_optimal"])
                k = "followed" if item["followed"] else "not_followed"
                s[k + "_n"] += 1
                s[k + "_optimal"] += bool(entry["played_optimal"])
            ex = examples.setdefault(name, [])
            if len(ex) < 3:
                ex.append({"id": cid, "fen": entry["fen"], "start": start, "moves": moves, "proposed": prop, "played": played, "source_best": entry["source_best"]})
        rows.append(entry)
    lemmas = {n: {**{k: v for k, v in s.items() if k != "games"}, "games": len(s["games"])} for n, s in sorted(per.items())}
    overall = {"decisions": len(decisions), "with_lemma": len(rows), "games": len(games), "games_touched": len(touched),
               "share_games": round(len(touched) / len(games), 4) if games else None, "mismatch": mismatch, "source": src_name}
    rep = {"game": game, "log": log, "overall": overall, "lemmas": lemmas, "decisions": rows, "examples": examples}
    ids = {d["row"].get("snapshot") for d in decisions.values()} - {None, ""}
    n_lemmas = 0
    for sid in sorted(ids):
        try:
            n_lemmas += len(_snapshot(cache, game, sid, snapshot)["lemmas"])
        except (FileNotFoundError, KeyError):
            pass                                              # (the log's run label where no store was used)
    rep["overall"]["lemmas_in_snapshots"] = n_lemmas
    if n_lemmas == 0:
        say(f"OBS audit {game}: no lemma in the store ({len(decisions)} decisions in {len(games)} games, log {log})")
        return rep
    say(f"OBS audit {game}: {len(decisions)} decisions, {len(rows)} with a lemma firing, {len(touched)} of {len(games)} games touched ({overall['share_games']}), "
        f"recomputed proposal counts differing from the engine's {mismatch}, rank source {src_name}")
    for n, s in lemmas.items():
        claim = f", claim held {s['proposal_optimal']} of {s['scored']} by {src_name}, played optimal {s['followed_optimal']}/{s['followed_n']} when followed vs {s['not_followed_optimal']}/{s['not_followed_n']} when not" \
            if source is not None else ""
        say(f"OBS audit lemma {n}: fired {s['fired']} in {s['games']} games, proposed {s['proposed']} moves, followed {s['followed']}{claim}")
        for ex in examples[n]:
            pos = f"start {ex['start'] or 'initial'} moves {' '.join(ex['moves']) or '-'}; fen {ex['fen']}"
            say(f"OBS audit example {n} {ex['id']}: {pos}: proposed {ex['proposed']}, played {ex['played']}, {src_name or 'source'} best {ex['source_best']}")
    return rep


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--log", required=True)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--teacher", default=None, help="a UCI engine as the rank source (training only)")
    ap.add_argument("--teacher-ms", type=int, default=100)
    ap.add_argument("--teacher-option", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--vocabulary", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    teacher = SL.UciTeacher(a.teacher, a.teacher_ms, dict(o.partition("=")[::2] for o in a.teacher_option)) if a.teacher else None
    try:
        lines = mined(a.game, a.vocabulary)
        for l in lines:
            print("OBS mined " + l)
        if not lines:
            print(f"OBS mined {a.game}: no lemma in the store")
        rep = audit(a.game, a.log, snapshot=a.snapshot, variant=a.variant, source=teacher, vocabulary=a.vocabulary, limit=a.limit)
    finally:
        if teacher:
            teacher.close()
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, default=str)
        print(f"OBS audit report -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
