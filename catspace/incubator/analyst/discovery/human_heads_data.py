"""PLAN 78.59 A AND B: THE SAMPLE OF HUMAN GAMES AND THE INPUT ROWS AT QUIET POSITIONS. A measurement in a sandbox; nothing is wired into any player, nothing is written to rules/learned or data/derived.
Game-free: a kind is a word read from the graph at run time; no unit, place or kind of any game is named here. No engine evaluation of any kind enters: only the result of each game and the two ratings.

THE DATA: parquet shards of the month of human games under data/records (the one directory the owner allowed on 2026-11-15; the other game directory is not read). The columns loaded are the game number, the
two ratings, the result, the plies, the way the game ended and the moves: no player id is loaded, kept or printed. Every game of the chosen shards has its `moves` field checked token by token: a token that is
not a bare move word (an annotation, a comment, an evaluation or a clock) makes the game ANNOTATED, and it is dropped unread (counted).
KEPT: ended "Normal" (on the board, by resignation or agreement; a game ended by the clock or by an infraction is another thing, dropped and counted), a result, 20 to 200 plies, every move accepted by the engine.
THE SAMPLE: per shard the kept games ordered by a hash of (shard, game number), the first PER_SHARD; the first ARRIVAL_PER_SHARD of that order are the arrival sample. THE SPLIT: by game, by a second hash:
SPLIT_FIT in 10 fit, the rest test. Bands of the mean rating: BAND_CUTS.
ROWS: every position from ply OPEN_PLIES on that is quiet (`position_table`: no unit of either side borne on by the other and not by its own), with the kinds (ours less theirs), the 26 statements (ours less
theirs), the ratings and the result of the side to move; the distinct kept concepts of 78.54 (ours less theirs) on every CONCEPT_EVERY-th quiet row of a game.

  .venv/bin/python -m catspace.incubator.analyst.discovery.human_heads_data --out-dir O [--short N] [--threads K]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from catspace.io import paths

SHARDS = (3, 15, 27, 39, 51, 63, 75, 87)
SHARD_DIR = "lichess_2019-01"                                # under the registry's records directory
PER_SHARD = 1500
ARRIVAL_PER_SHARD = 250
SPLIT_FIT = 7
BAND_CUTS = (1350, 1550, 1750)
BAND_NAMES = ("A <1350", "B 1350-1549", "C 1550-1749", "D 1750+")
MIN_PLIES, MAX_PLIES = 20, 200
OPEN_PLIES = 10
CONCEPT_EVERY = 8
CONCEPTS = 64
VARIANT = "chess"
GAME = "chess"
MOVE_WORD = re.compile(r"^([a-h][1-8][a-h][1-8][qrbn]? ?)+$")
COLUMNS = ["game_id", "white_elo", "black_elo", "result", "n_plies", "termination", "moves"]


def shard_path(i: int) -> Path:
    return paths.records_dir() / SHARD_DIR / f"records_{i:05d}.parquet"


def band_of(mean_rating: float) -> int:
    return int(sum(mean_rating >= c for c in BAND_CUTS))


def key_of(shard: int, gid: int, salt: str) -> int:
    return int(hashlib.md5(f"{shard}:{gid}:{salt}".encode()).hexdigest(), 16)


def sample_shard(i: int, per_shard: int, arrival: int) -> "tuple[list, dict]":
    """(games, counts) of one shard. A game is a dict: id, moves (list of words), welo, belo, y (the first mover's result: 0 win, 1 draw, 2 loss), band, fit (bool), arrival (bool)."""
    import pyarrow.parquet as pq
    df = pq.read_table(shard_path(i), columns=COLUMNS).to_pandas()
    n = len(df)
    annotated = ~df["moves"].str.match(MOVE_WORD)
    ended = df["termination"] != "Normal"
    nores = ~df["result"].isin([-1, 0, 1])
    size = (df["n_plies"] < MIN_PLIES) | (df["n_plies"] > MAX_PLIES)
    keep = ~annotated & ~ended & ~nores & ~size
    counts = {"shard": i, "games": n, "annotated": int(annotated.sum()), "not_normal": int((ended & ~annotated).sum()), "no_result": int(nores.sum()), "out_of_size": int(size.sum()),
              "qualifying": int(keep.sum())}
    sub = df[keep]
    order = sorted(range(len(sub)), key=lambda k: key_of(i, int(sub["game_id"].iloc[k]), "order"))[:per_shard]
    games = []
    for rank, k in enumerate(order):
        r = sub.iloc[k]
        gid = int(r["game_id"])
        w, b = int(r["white_elo"]), int(r["black_elo"])
        games.append({"id": f"{i}:{gid}", "moves": r["moves"].split(), "welo": w, "belo": b, "y": {1: 0, 0: 1, -1: 2}[int(r["result"])], "band": band_of((w + b) / 2),
                      "fit": key_of(i, gid, "split") % 10 < SPLIT_FIT, "arrival": rank < arrival})
    return games, counts


def drop_illegal(model, games: list) -> "tuple[list, int]":
    """The games whose every move the engine accepts, and the count it refused (a batch is tried whole, then game by game)."""
    ok, bad = [], 0
    for g in games:
        try:
            model.position_table([(None, g["moves"])], 1)
            ok.append(g)
        except ValueError:
            bad += 1
    return ok, bad


def sample(short: int, say) -> list:
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    model = GL.model(GAME, VARIANT)
    allg, tot = [], {"games": 0, "annotated": 0, "not_normal": 0, "no_result": 0, "out_of_size": 0, "qualifying": 0, "refused": 0}
    for i in SHARDS if not short else SHARDS[:1]:
        t0 = time.perf_counter()
        games, c = sample_shard(i, short or PER_SHARD, min(short or ARRIVAL_PER_SHARD, ARRIVAL_PER_SHARD))
        games, refused = drop_illegal(model, games)
        c["refused"] = refused
        say(f"A shard {i}: games {c['games']}, carrying an annotation (dropped unread) {c['annotated']}, not ended Normal (dropped) {c['not_normal']}, no result {c['no_result']}, outside {MIN_PLIES}..{MAX_PLIES} plies {c['out_of_size']}, "
            f"qualifying {c['qualifying']}; sampled {len(games) + refused}, refused by the engine's rules (dropped) {refused}, kept {len(games)} ({time.perf_counter() - t0:.1f} s)")
        for k in tot:
            tot[k] += c.get(k, 0)
        allg += games
    say(f"A total over the shards read: games {tot['games']}, annotated and dropped {tot['annotated']}, not Normal {tot['not_normal']}, no result {tot['no_result']}, out of size {tot['out_of_size']}, qualifying {tot['qualifying']}, "
        f"refused {tot['refused']}; the sample {len(allg)} games ({sum(len(g['moves']) for g in allg)} plies), arrival sample {sum(g['arrival'] for g in allg)}")
    fit = sum(g["fit"] for g in allg)
    say(f"A split by game: fit {fit}, test {len(allg) - fit}; by band of the mean rating (games): " + ", ".join(f"{nm} {sum(g['band'] == b for g in allg)}" for b, nm in enumerate(BAND_NAMES)))
    res = np.bincount([g["y"] for g in allg], minlength=3)
    say(f"A the first mover's result over the sample: win {res[0] / len(allg):.4f}, draw {res[1] / len(allg):.4f}, loss {res[2] / len(allg):.4f}")
    return allg


def statements_chunked(games: list, threads: int, chunk: int = 400) -> "tuple[list, list]":
    """(names, per game an array (positions, 2 x statements) as float32, NaN where over): the engine's statements, in chunks across threads."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    model = SR.model(QS.Env(VARIANT), QS.store_worth(VARIANT))
    names = model.subgoal_names()
    n = len(names)
    out = []
    for i in range(0, len(games), chunk):
        res = model.subgoal_games([(None, g["moves"]) for g in games[i:i + chunk]], threads)
        for g in res:
            out.append(np.array([x if x is not None else [np.nan] * (2 * n) for x in g], dtype=np.float32).reshape(len(g), 2 * n))
    return names, out


def build_rows(games: list, threads: int, say, concepts: bool = True) -> dict:
    """The quiet rows of all games as arrays: game, ply, y (side to move), kd (kinds ours less theirs), sd (statements ours less theirs), the mover's and the other's rating, band, fit, arrival,
    and for the thinned rows the concept columns (`cmask` marks the rows that carry them)."""
    from catspace.incubator.analyst.discovery import frames_measure as OLD
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2 as LAYER2
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    model = GL.model(GAME, VARIANT)
    goal = G2M.goal_kind_of(VARIANT)
    t0 = time.perf_counter()
    names, raw = model.position_table([(None, g["moves"]) for g in games], threads)
    nk = len(names)
    flat = np.frombuffer(raw, dtype=np.int16)
    w = 2 + 2 * nk
    use = [k for k, nm in enumerate(names) if nm != goal]
    kind_names = [names[k] for k in use]
    t1 = time.perf_counter()
    snames, stats = statements_chunked(games, threads)
    ns = len(snames)
    say(f"B kinds {kind_names} (the goal kind {goal} left out); position table {t1 - t0:.1f} s; statements {ns} per side, read for every position of {len(games)} games in {time.perf_counter() - t1:.1f} s")
    gi, ply, y, kd, sd, rm, ro, band, fit, arr = [], [], [], [], [], [], [], [], [], []
    off = 0
    for i, g in enumerate(games):
        n = len(g["moves"]) + 1
        t = flat[off:off + n * w].reshape(n, w).astype(np.int32)
        off += n * w
        st = stats[i]
        for p in range(OPEN_PLIES, n):
            if t[p, 0] != 1 or t[p, 1] != 1 or p >= len(st) or np.isnan(st[p]).any():
                continue
            first = p % 2 == 0
            gi.append(i); ply.append(p)
            y.append(g["y"] if first else 2 - g["y"])
            kd.append((t[p, 2:2 + nk] - t[p, 2 + nk:2 + 2 * nk])[use])
            sd.append(st[p, :ns] - st[p, ns:])
            rm.append(g["welo"] if first else g["belo"]); ro.append(g["belo"] if first else g["welo"])
            band.append(g["band"]); fit.append(g["fit"]); arr.append(g["arrival"])
    rows = {"game": np.array(gi, np.int32), "ply": np.array(ply, np.int16), "y": np.array(y, np.int8), "kd": np.array(kd, np.int16), "sd": np.array(sd, np.float32).reshape(len(gi), ns),
            "rm": np.array(rm, np.int16), "ro": np.array(ro, np.int16), "band": np.array(band, np.int8), "fit": np.array(fit, bool), "arrival": np.array(arr, bool)}
    say(f"B quiet rows from ply {OPEN_PLIES} on: {len(gi)} of {len(games)} games; fit {int(rows['fit'].sum())}, test {int((~rows['fit']).sum())}; result of the side to move over the rows: "
        + ", ".join(f"{nm} {float((rows['y'] == k).mean()):.4f}" for k, nm in enumerate(("win", "draw", "loss"))))
    first_of = np.searchsorted(rows["game"], rows["game"])
    cmask = ((np.arange(len(gi)) - first_of) % CONCEPT_EVERY) == 0
    rows["cmask"] = cmask
    cnames: list = []
    rows["cx"] = np.zeros((int(cmask.sum()), 0), np.float32)
    if concepts:
        t2 = time.perf_counter()
        picked = OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), CONCEPTS)
        sel = np.flatnonzero(cmask)
        gidx = np.unique(rows["game"][sel])
        wires, rls = [], []
        order = []
        for gx in gidx:
            s = sel[rows["game"][sel] == gx]
            order.append(s)
            wires.append((None, games[int(gx)]["moves"]))
            rls.append([int(v) for v in rows["ply"][s]])
        order = np.concatenate(order)
        raw_c = model.pattern_count(wires, rls, [k for _dc, k, _d, _z in picked], 3, list(LAYER2.LEVELS), goal, threads)
        us, them = LAYER2.counts_matrix(raw_c, len(order), len(picked))
        diff = (us - them).toarray().astype(np.float32)
        pos = np.searchsorted(sel, order)
        cx = np.zeros_like(diff)
        cx[pos] = diff
        rows["cx"] = cx
        cnames = [f"{dc}: {d}" for dc, _k, d, _z in picked]
        say(f"B concepts: the top {len(cnames)} distinct kept concepts of 78.54 by |z| on its half A, ours less theirs, on every {CONCEPT_EVERY}th quiet row of each game, {len(sel)} rows ({time.perf_counter() - t2:.1f} s)")
    rows["meta"] = {"kinds": kind_names, "statements": [n if isinstance(n, str) else n[0] for n in snames], "concepts": cnames, "goal": goal}
    return rows


def save_rows(rows: dict, p: Path) -> None:
    arrays = {k: v for k, v in rows.items() if k != "meta"}
    np.savez_compressed(p, **arrays)
    p.with_suffix(".json").write_text(json.dumps(rows["meta"]))


def load_rows(p: Path) -> dict:
    z = np.load(p)
    rows = {k: z[k] for k in z.files}
    rows["meta"] = json.loads(p.with_suffix(".json").read_text())
    return rows


def free_gb() -> float:
    st = os.statvfs(".")
    return st.f_bavail * st.f_frsize / 1e9


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--short", type=int, default=0, help="one shard, this many games")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-concepts", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir or paths.experiment("estimator_78_59", "short" if a.short else "full"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS human heads: {msg}")
        print(lines[-1], flush=True)

    say(f"disk free before: {free_gb():.2f} GB")
    games = sample(a.short, say)
    (out / "games.jsonl").write_text("\n".join(json.dumps({k: v for k, v in g.items()}) for g in games) + "\n")
    rows = build_rows(games, a.threads, say, concepts=not a.no_concepts)
    save_rows(rows, out / "rows.npz")
    say(f"rows written to {out / 'rows.npz'}; disk free after: {free_gb():.2f} GB; wall {time.perf_counter() - t0:.1f} s")
    (out / "data_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
