"""PLAN 79.24 PART C: THE MAKER OF LAYER 3'S FIXTURES (SYSTEM_SPEC 11 "layer 3 holds two implementations: value and the measure of progress", 10 data rules, 15 standing limits; ledger rows 35, 36; piece: value and measure of progress).
Made once, by a program that may play games and may ask the endgame table as a MEASURE (through the one oracle module, measure_oracle_79_17); the layer's test (rust/engine/value/tests/layer3.rs) reads the files it makes and calls no search and no outside program.
This is the one place that names the game's objects (a maker script named `*_print.py`). No player's name or id is read, kept or printed; no number of an outside engine is in any fit; no worth is assumed.

TWO FIXTURES, in rust/engine/value/tests/fixtures/ (tab separated lines, the first field the tag):
  value.fixture
    game   <name> <declaration file under rules/>
    model  kind  <json>   the value: the stored result head of the kinds alone, fitted on human games only (estimator_78_59, `result_human_K`: a multinomial logistic of the side to move's result on the count of each kind, ours less theirs; fit games the
                          hash 7 in 10 of the games of the shards, quiet rows from ply 10); classes win, draw, loss. NOT refitted here.
    model  count <json>   the plain count of material the value was fitted beside: the same form on one feature, the count of units ours less theirs, fitted HERE on the SAME fit rows (the hash split of the human_heads_data sample), the same penalty (C = 1.0)
    sample <name> <kind> <games> <positions> <endgame positions> <endgame share>      the samples: human games of the test split in four rating bands (of the mean of the two ratings), and generated games that fill the thin endgames
    pos    <sample> <game id (a number made here, never a player's id)> <ply> <units left> <result of the side to move: 0 win, 1 draw, 2 loss> <position text>
  ending.fixture
    game   <name> <declaration file>
    measure <class> <term> <within d> <sample the term was mined on>    the measures mined in plan 79.17 (artifacts whole_79_17/full/measures.json), read as they are
    line   <set> <class of the measure> <id> <attacker role> <won 0/1> <position:distance;...>   the table's own best play from a won start (the attacker the table's shortest, the defender the table's longest), the positions with the attacker to move, each with the table's distance there
HUMAN DATA: data/records/lichess_2019-01 only (the one directory allowed), the test split of human_heads_data, the games of its shards; the share of endgame positions (at most ENDGAME_MEN units on the board, both sides, all kinds) is printed for every human sample; a sample under
THIN_SHARE of endgame positions is filled with generated games (the generation player of plan 78.61 from endgame positions of the human test games; its judge is the odds record, results of games; no outside engine). The generated games are a sample of their own.

  .venv/bin/python -m catspace.incubator.analyst.discovery.layer3_fixtures_print OUT_DIR [--workers K]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import measure_oracle_79_17 as MO
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import sleep_loop as SL
from catspace.io import paths

GAME = "chess"
DECL = "learned/chess/graph_vocabulary.json"
HEAD = ("estimator_78_59", "full", "estimator_heads_human.json")
MEASURES = ("whole_79_17", "full", "measures.json")
ODDS_RECORD = ("odds_78_61", "full", "gen2", "record.json")
GAMES_PER_BAND = 40                         # human test games per rating band
STEP = 4                                    # every STEP-th quiet row of a game from ply OPEN
OPEN = HD.OPEN_PLIES
ENDGAME_MEN = 8                             # units on the board in all (the convention of events_fill) at or under which a position is an endgame position
THIN_SHARE = 0.10                           # a human sample with a smaller share of endgame positions is thin and is filled
GEN_STARTS = 50                             # endgame starts (positions of the human test games) of the generated games, two games each
GEN_GAMES = 2
GEN_OPENING = 2
GEN_SECONDS = 0.05
GEN_PLIES = 120
ENDING_SETS = {"one": ("one", MO.table_games), "two": ("two", MO.table_games), "twofar": ("two", MO.far_table_games)}      # the set of lines: (the class whose measure it is read by, how its lines are made)
ENDINGS = ("one", "two")
LINES_PER_ENDING = 30
LINE_SEED = 24
FIT_C = 1.0


def h(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def rows_of(model, games: list, threads: int, starts: "list | None" = None) -> list:
    """Per game the quiet rows from ply OPEN on (as human_heads_data.build_rows reads them): (ply, kinds ours less theirs as a vector, units on the board, the side to move's result)."""
    names, raw = model.position_table([(None if starts is None else starts[i], g["moves"]) for i, g in enumerate(games)], threads)
    nk = len(names)
    flat = np.frombuffer(raw, dtype=np.int16)
    w = 2 + 2 * nk
    goal = "king"
    use = [k for k, nm in enumerate(names) if nm != goal]
    out, off = [], 0
    for i, g in enumerate(games):
        n = len(g["moves"]) + 1
        t = flat[off:off + n * w].reshape(n, w).astype(np.int32)
        off += n * w
        rows = []
        for p in range(OPEN, n):
            if t[p, 0] != 1 or t[p, 1] != 1:
                continue
            first = p % 2 == 0
            y = g["y"] if first else 2 - g["y"]
            rows.append((p, (t[p, 2:2 + nk] - t[p, 2 + nk:2 + 2 * nk])[use], int(t[p, 2:2 + 2 * nk].sum()), y))
        out.append(rows)
    return out


def head_model() -> dict:
    rec = json.loads(Path(paths.experiment(*HEAD)).read_text())
    hd = rec["heads"]["result_human_K"]
    words = OM.kernel_names()                                # the stored head names a kind as the graph does; the game's own units are named as the engine's captures name them
    kinds = [words[k.split(":", 1)[1]] for k in hd["inputs"]]
    m = hd["model"]
    assert m["classes"] == [0, 1, 2] and all(x == 0.0 for x in hd["transform"]["centre"]) and all(x == 1.0 for x in hd["transform"]["scale"])
    return {"kinds": kinds, "intercept": m["intercept"], "coef": m["coef"], "count": hd["data"]["fit_rows"],
            "source": f"the stored kinds-only result head of plan 78.59 (human games, hash split 7 in 10 fit, quiet rows from ply {hd['data']['from_ply']}, penalty C {m['C']}); not refitted"}


def fit_count(fit_games: list, rows: list, say) -> dict:
    from sklearn.linear_model import LogisticRegression
    x, y = [], []
    for g, rs in zip(fit_games, rows):
        for p, kd, units, yy in rs:
            x.append([float(kd.sum())])
            y.append(yy)
    x, y = np.array(x), np.array(y)
    m = LogisticRegression(C=FIT_C, max_iter=1000).fit(x, y)
    assert list(m.classes_) == [0, 1, 2]
    say(f"OBS 79.24 count model: fitted on {len(y)} fit rows of {len(fit_games)} fit games; intercept {[round(float(v), 4) for v in m.intercept_]}, coefficient {[round(float(v[0]), 4) for v in m.coef_]}")
    return {"intercept": [float(v) for v in m.intercept_], "coef": [float(v[0]) for v in m.coef_], "count": int(len(y)),
            "source": f"the same fit rows as the kinds' head: the human games of the hash split (fit side), quiet rows from ply {OPEN}, penalty C {FIT_C}, one feature: the units ours less theirs"}


def texts_of(moves: list, plies: list, start: "str | None") -> dict:
    return {p: CE.state(GAME, start, moves[:p])[0] for p in plies}


def human_sample(say) -> tuple:
    model = GL.model(GAME, GAME)
    allg = []
    for i in HD.SHARDS:
        games, c = HD.sample_shard(i, HD.PER_SHARD, 0)
        games, refused = HD.drop_illegal(model, games)
        say(f"OBS 79.24 human shard {i}: sampled {len(games) + refused}, refused by the rules {refused}, kept {len(games)}")
        allg += games
    return model, allg


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--short", action="store_true", help="a small run to try the maker")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines_obs: list = []

    def say(m: str) -> None:
        lines_obs.append(m)
        print(m, flush=True)
    t0 = time.perf_counter()
    model, allg = human_sample(say)
    fit_games = [g for g in allg if g["fit"]]
    test_games = [g for g in allg if not g["fit"]]
    say(f"OBS 79.24 human sample: {len(allg)} games of the shards, fit {len(fit_games)}, test {len(test_games)}; the bands of the mean rating: " + ", ".join(f"{nm} {sum(g['band'] == b for g in allg)}" for b, nm in enumerate(HD.BAND_NAMES)))
    fit_rows = rows_of(model, fit_games, a.workers)
    count_model = fit_count(fit_games, fit_rows, say)
    kind_model = head_model()
    say(f"OBS 79.24 the stored head was fitted on {kind_model['count']} rows; the fit rows read here: {sum(len(r) for r in fit_rows)}")
    test_rows = rows_of(model, test_games, a.workers)
    # the human samples: per band the first GAMES_PER_BAND test games in a hash order that have quiet rows
    per_band = 8 if a.short else GAMES_PER_BAND
    samples, pos = [], []
    gid = 0
    for b, nm in enumerate(HD.BAND_NAMES):
        cand = sorted((i for i, g in enumerate(test_games) if g["band"] == b and test_rows[i]), key=lambda i: h(f"layer3:{test_games[i]['id']}"))[:per_band]
        all_rows = [r for i, g in enumerate(test_games) if g["band"] == b for r in test_rows[i]]
        end_all = sum(1 for r in all_rows if r[2] <= ENDGAME_MEN)
        rows = []
        for i in cand:
            g = test_games[i]
            gid += 1
            sel = test_rows[i][::STEP]
            tx = texts_of(g["moves"], [r[0] for r in sel], None)
            for p, kd, units, y in sel:
                rows.append(("human_" + nm.split()[0], gid, p, units, y, tx[p]))
        end = sum(1 for r in rows if r[3] <= ENDGAME_MEN)
        say(f"OBS 79.24 sample human_{nm.split()[0]} ({nm}): {len(cand)} games, {len(rows)} positions, endgame positions (at most {ENDGAME_MEN} units on the board) {end} = {end / max(len(rows), 1):.4f} of the sample; of all the band's quiet test rows {end_all} of {len(all_rows)} = {end_all / max(len(all_rows), 1):.4f}")
        samples.append(("human_" + nm.split()[0], "human", len(cand), rows, end))
        pos += rows
    thin = [s for s in samples if s[4] / max(len(s[3]), 1) < THIN_SHARE]
    say(f"OBS 79.24 human samples under {THIN_SHARE} of endgame positions (thin): {[s[0] for s in thin]}")
    # the fill: generated games from endgame positions of the human test games
    starts = []
    for i in sorted(range(len(test_games)), key=lambda i: h(f"layer3gen:{test_games[i]['id']}")):
        end_rows = [r for r in test_rows[i] if r[2] <= ENDGAME_MEN]
        if end_rows:
            p = end_rows[0][0]
            text = CE.state(GAME, None, test_games[i]["moves"][:p])[0]
            if OM.playable(text):
                starts.append(text)
        if len(starts) >= (4 if a.short else GEN_STARTS):
            break
    rec = json.loads(Path(paths.experiment(*ODDS_RECORD)).read_text())
    player = SL.player_from_values(rec["heads"]["odds_effect"]["kind_values"])
    specs = []
    for j, text in enumerate(starts):
        for k in range(GEN_GAMES):
            specs.append({"gid": f"fill:{j}:{k}", "start": text, "text": text, "opening": OM.random_opening(text, 7400 + 10 * j + k, GEN_OPENING), "players": [player, player],
                          "seconds": GEN_SECONDS, "max_plies": GEN_PLIES, "scratch": str(out / "scratch")})
    played = OM.run_games(specs, a.workers, out / "played.jsonl", say=say)
    gen = []
    for g in played:
        if g["outcome"] is None or g["unfinished"]:
            continue
        first = CE.state(GAME, g["start"], [])[1]
        first_white = first == "white"
        first_wins = (g["outcome"] == 100 and first_white) or (g["outcome"] == 0 and not first_white)
        y = 0 if first_wins else (1 if g["outcome"] == 50 else 2)
        gen.append({"id": g["gid"], "start": g["start"], "moves": g["moves"], "y": y})
    say(f"OBS 79.24 generated games: {len(specs)} played from {len(starts)} endgame starts, {len(gen)} finished (the ply cap and the unfinished are dropped)")
    grows = rows_of(model, gen, a.workers, starts=[g["start"] for g in gen])
    rows = []
    for i, g in enumerate(gen):
        gid += 1
        sel = [r for r in grows[i] if r[0] >= GEN_OPENING][:: 3]
        tx = texts_of(g["moves"], [r[0] for r in sel], g["start"])
        for p, kd, units, y in sel:
            rows.append(("generated", gid, p, units, y, tx[p]))
    end = sum(1 for r in rows if r[3] <= ENDGAME_MEN)
    say(f"OBS 79.24 sample generated: {len(gen)} games, {len(rows)} positions, endgame positions {end} = {end / max(len(rows), 1):.4f}")
    samples.append(("generated", "generated", len(gen), rows, end))
    pos += rows
    allpos = len(pos)
    allend = sum(1 for r in pos if r[3] <= ENDGAME_MEN)
    say(f"OBS 79.24 the value fixture: {allpos} positions, endgame positions {allend} = {allend / max(allpos, 1):.4f}; results of the side to move: " + ", ".join(f"{nm} {sum(1 for r in pos if r[4] == k)}" for k, nm in enumerate(("win", "draw", "loss"))))
    with open(out / "value.fixture", "w") as f:
        f.write(f"game\t{GAME}\t{DECL}\n")
        f.write("model\tkind\t" + json.dumps(kind_model, sort_keys=True) + "\n")
        f.write("model\tcount\t" + json.dumps(count_model, sort_keys=True) + "\n")
        for nm, kind, ng, rows, end in samples:
            f.write(f"sample\t{nm}\t{kind}\t{ng}\t{len(rows)}\t{end}\t{end / max(len(rows), 1):.4f}\n")
        for sname, g, p, units, y, text in pos:
            f.write(f"pos\t{sname}\t{g}\t{p}\t{units}\t{y}\t{text}\n")
    # the endings: the table's own best play
    measures = json.loads(Path(paths.experiment(*MEASURES)).read_text())
    with open(out / "ending.fixture", "w") as f:
        f.write(f"game\t{GAME}\t{DECL}\n")
        for cls in ENDINGS:
            m = measures[cls]
            f.write(f"measure\t{cls}\t{m['term']}\t{m['d']}\tplan 79.17, artifacts whole_79_17/full/measures.json\n")
        for cls, (mclass, make) in ENDING_SETS.items():
            games = make(mclass, 4 if a.short else LINES_PER_ENDING, LINE_SEED)
            for k, g in enumerate(games):
                fens = MO.fens_of(g["text"], g["moves"])
                mine = fens[0::2]
                won = MO.attacker_won(g["text"], g["moves"], g["attacker_white"])
                f.write(f"line\t{cls}\t{mclass}\t{k}\t{MO.side_of(g['attacker_white'])}\t{int(won)}\t" + ";".join(f"{x}|{MO.distance(x)}" for x in mine) + "\n")
            say(f"OBS 79.24 ending {cls}: {len(games)} table lines, {sum(len(MO.fens_of(g['text'], g['moves'])[0::2]) for g in games)} positions with the attacker to move; coverage by the table's distance at the start {MO.coverage([{'dtz': MO.distance(g['text'])} for g in games])}")
    say(f"OBS 79.24 maker wall {time.perf_counter() - t0:.1f} s")
    (out / "maker_obs.txt").write_text("\n".join(lines_obs) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
