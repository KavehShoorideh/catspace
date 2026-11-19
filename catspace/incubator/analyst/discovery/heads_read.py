"""PLAN 79.04 PART ONE, THE HEADS WE ALREADY HAVE READ AT THE EQUAL MOVES (SYSTEM_SPEC 5 and 6 the value fitted on results and its heads, 4 the leaf, 9 the oracle as labeller in sleep; piece: the estimator's heads, a MEASURE). No fit, no new head, no engine change, no match, nothing written to
rules/learned or data/derived, no oracle call (the labels are 79.00's and 79.01's, stored); outputs under artifacts/experiments/heads_79_04/<tag>/.

THE HEADS (plans 78.58 and 78.59, the record `estimator_78_59/full/estimator_heads_human.json`, never refitted). RESULT heads (win, draw, loss of the side to move; a multinomial logistic fit on results alone): `result_sp_K` (self-play results, the kinds), `result_human_K` (human games' results, the kinds: the head of 79.01),
`result_human_RK` (with ratings), `result_human_RKS` (with the 26 statements), the same two on the thinned rows, and `result_human_RKC` / `result_human_RKSC` (thinned rows, with the 64 kept concepts). ARRIVAL heads (an expected value: the chance that a confirmed threat of the side to move arrives within 8 plies, less the other side's; a sparse linear fit): kinds, statements, kinds and statements, each with and
without ratings, and on the thinned rows with concepts. Inputs are the kinds, the statements and the concepts ours less theirs for the side to move (ratings R: mean, difference, ply). None has oracle numbers in it.

HOW A HEAD IS READ AT A POSITION. The statements and kinds and concepts are made by the SAME calls that made the fit's rows (`human_heads_data.build_rows`: `position_table` for the kinds and the flags, `statements_chunked` for the 26 statements, `pattern_count` and `graph_layer2.counts_matrix` for the concepts), over a wire that STARTS at the root position (the engine's own position text) and plays only the tail: the move, or the line
to its settled end. That is the same read as the whole game from the start (checked on a sample, printed). Ratings are not known at a leaf: the mean rating is the head's own centre (an average pair of players) and the rating difference is 0, the ply is the ROOT position's ply (the same for every equal move, so it cannot order them). Not-quiet leaves are read as they are (the heads were fitted on quiet rows; the share is printed).

BOTH SIDES' VIEW (so that the side to move's bonus cancels). A head's value is for the side to move at the position. At a leaf the side to move is the root mover or the other side depending on the parity of the line, and a head carries a bonus for whoever is to move, so the raw value orders two equal moves by the parity of their lines (79.01). The reading used here: the value for the side to move at the leaf, from the leaf's own inputs
X (ours less theirs), is averaged with the value the OTHER side would get as the side to move on the swapped inputs (-X): result heads 0.5 (v(X) + 1 - v(-X)), arrival heads 0.5 (v(X) - v(-X)); then taken for the root mover (one less it for a probability, minus it for an arrival value, when the other side is to move at the leaf). Two lines of different parity ending in the same position (the same statements, seen from the two sides) get the same
value; tested. What it does not cancel: anything the statements say about whose turn it is inside the position itself.

INTERFACE (written before the bodies).
  SPEC                                            the heads read: (name, record key, kind "prob" | "value", the letters of the input blocks in the head's own column order).
  Record()                                        per process: the record. `.heads` the names; `.stm(name, blocks, ply)` the value for the side to move from input blocks {"K","S","C"} (n rows); `.view(name, blocks, ply, is_root, sym)` the value for the ROOT mover (pure given the record; tested).
  Reader(threads)                                 per process: the engine calls above. `.read(wires) -> {"K","S","C","quiet","ok"}` of the LAST position of each (start text, tail) wire.
  tails_of(rec, line_rows) ...                    see `positions`.
  run(tag, limit, threads, say)                   reads, for every equal move of every firm-room position of 79.01 (leaf_line.items_of; the lines of 79.01's lines.jsonl; a line longer than the 14 names stored is read again by leaf_line.Lines), the blocks one move after the root and at the end of the line; blocks.jsonl.
  load(root)                                      -> [position blocks].
  Called by: heads_measure (the tables, the count, the judged positions), ladder_print --heads04.

  .venv/bin/python -m catspace.incubator.analyst.discovery.heads_read blocks --tag short --limit 40 | --tag full [--threads K]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import cs_engine as CE
from catspace.incubator.analyst.discovery import estimator_heads as EH
from catspace.incubator.analyst.discovery import frames_measure as OLD
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as LAYER2
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import human_heads_result as HR
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import leaf_line as LL
from catspace.io import paths

OUT = ("heads_79_04",)
SRC_TAG = "full"                                                  # 79.01's run: its labels, firm room and lines are read only
WIDTH = {"K": 5, "S": 26, "C": 64}
CHUNK = 12
CHECK_WIRES = 25
SPEC = (
    ("result_sp_K", "result", "prob", "K"),
    ("result_human_K", "result_human_K", "prob", "K"),
    ("result_human_RK", "result_human_RK", "prob", "RK"),
    ("result_human_RKS", "result_human_RKS", "prob", "RKS"),
    ("result_human_RK_thinned", "result_human_RK_thinned", "prob", "RK"),
    ("result_human_RKS_thinned", "result_human_RKS_thinned", "prob", "RKS"),
    ("result_human_RKC_thinned", "result_human_RKC_thinned", "prob", "RKC"),
    ("result_human_RKSC_thinned", "result_human_RKSC_thinned", "prob", "RKSC"),
    ("arrival_human_K", "arrival_human_K", "value", "K"),
    ("arrival_human_K_R", "arrival_human_K_R", "value", "KR"),
    ("arrival_human_S", "arrival_human_S", "value", "S"),
    ("arrival_human_S_R", "arrival_human_S_R", "value", "SR"),
    ("arrival_human_KS", "arrival_human_KS", "value", "KS"),
    ("arrival_human_KS_R", "arrival_human_KS_R", "value", "KSR"),
    ("arrival_human_thin_S", "arrival_human_thin_S", "value", "S"),
    ("arrival_human_thin_S_R", "arrival_human_thin_S_R", "value", "SR"),
    ("arrival_human_thin_SC", "arrival_human_thin_SC", "value", "SC"),
    ("arrival_human_thin_SC_R", "arrival_human_thin_SC_R", "value", "SCR"),
    ("arrival_human_thin_KSC", "arrival_human_thin_KSC", "value", "KSC"),
    ("arrival_human_thin_KSC_R", "arrival_human_thin_KSC_R", "value", "KSCR"),
)
NAMES = tuple(s[0] for s in SPEC)
BASELINE = "result_human_K"                                       # the kinds head of 79.01


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------------------------------------------ the record and the view (pure)

def swapped(blocks: dict) -> dict:
    """The same position seen by the other side: every block is ours less theirs, so it changes sign."""
    return {k: -np.asarray(v, float) for k, v in blocks.items()}


class Record:
    """See the module doc."""

    def __init__(self, path: "Path | None" = None):
        self.rec = json.loads(Path(path or paths.experiment("estimator_78_59", "full", "estimator_heads_human.json")).read_text())
        self.spec = {s[0]: s for s in SPEC}
        self.heads = list(NAMES)

    def head(self, name: str) -> dict:
        return self.rec["heads"][self.spec[name][1]]

    def kind(self, name: str) -> str:
        return self.spec[name][2]

    def letters(self, name: str) -> str:
        return self.spec[name][3]

    def offset(self, name: str, ch: str) -> int:
        return sum(WIDTH[c] if c != "R" else 3 for c in self.letters(name)[:self.letters(name).index(ch)])

    def mean_rating(self, name: str) -> float:
        """An average pair of players: the head's own centre of the mean-rating column."""
        h = self.head(name)
        if self.kind(name) == "prob":
            return float(h["transform"]["centre"][self.offset(name, "R")]) if "R" in self.letters(name) else 0.0
        return float(h["model"]["mean"][self.offset(name, "R")]) if "R" in self.letters(name) else 0.0

    def columns(self, name: str, blocks: dict, ply: float) -> np.ndarray:
        n = len(next(iter(blocks.values())))
        cols = []
        for ch in self.letters(name):
            cols.append(np.column_stack([np.full(n, self.mean_rating(name)), np.zeros(n), np.full(n, float(ply))]) if ch == "R" else np.asarray(blocks[ch], float).reshape(n, WIDTH[ch]))
        return np.hstack(cols)

    def stm(self, name: str, blocks: dict, ply: float) -> np.ndarray:
        """The head's value for the side to move at the position the blocks describe (a probability head: the win share plus half the draw share)."""
        x = self.columns(name, blocks, ply)
        h = self.head(name)
        if self.kind(name) == "prob":
            p = HR.head_probs(h, x) if h.get("transform") else EH.head_predict(self.rec, self.spec[name][1], x)
            return p[:, 0] + 0.5 * p[:, 1]
        return EH.head_predict(self.rec, self.spec[name][1], x)[:, 0]

    def view(self, name: str, blocks: dict, ply: float, is_root: np.ndarray, sym: bool = True) -> np.ndarray:
        """The value for the ROOT mover; `is_root` per row: the root mover is the side to move at the leaf. sym: both sides' view (module doc)."""
        v = self.stm(name, blocks, ply)
        if self.kind(name) == "prob":
            if sym:
                v = 0.5 * (v + 1.0 - self.stm(name, swapped(blocks), ply))
            return np.where(is_root, v, 1.0 - v)
        if sym:
            v = 0.5 * (v - self.stm(name, swapped(blocks), ply))
        return np.where(is_root, v, -v)

    def weights(self, name: str) -> "tuple[list, np.ndarray]":
        """(the column names, the linear weight of each INPUT on the head's odd part: for a probability head the log-odds of a win against a loss per raw unit of the column, for an arrival head the value per raw unit); rating columns are 0 (they are held fixed at a leaf)."""
        h = self.head(name)
        names, w = [], []
        if self.kind(name) == "prob":
            m = h["model"]
            coef = np.zeros((3, len(h["transform"]["scale"]) if h.get("transform") else np.asarray(m["coef"]).shape[1]))
            coef[list(m["classes"])] = np.asarray(m["coef"])
            ws = coef[0] - coef[2]
            sc = np.asarray(h["transform"]["scale"]) if h.get("transform") else np.ones(len(ws))
            ws = ws / sc
        else:
            m = h["model"]
            ws = np.asarray(m["coef"]) / np.asarray(m["scale"])
        j = 0
        for ch in self.letters(name):
            if ch == "R":
                names += ["rating mean", "rating difference", "ply"]
                w += [0.0, 0.0, 0.0]
                j += 3
            else:
                names += [f"{ch}:{i}" for i in range(WIDTH[ch])]
                w += [float(v) for v in ws[j:j + WIDTH[ch]]]
                j += WIDTH[ch]
        return names, np.array(w)


# ------------------------------------------------------------------------------------------------------------------------------ reading blocks at wires (the engine)

class Reader:
    """See the module doc."""

    def __init__(self, threads: int = 1, concepts: bool = True):
        self.threads = threads
        self.model = GL.model(HD.GAME, HD.VARIANT)
        self.goal = G2M.goal_kind_of(HD.VARIANT)
        self.concepts = concepts
        self.keys = [k for _dc, k, _d, _z in OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), HD.CONCEPTS)] if concepts else []

    def read(self, wires: list) -> dict:
        names, raw = self.model.position_table(wires, self.threads)
        nk = len(names)
        w = 2 + 2 * nk
        flat = np.frombuffer(raw, dtype=np.int16).astype(np.int32)
        use = [k for k, nm in enumerate(names) if nm != self.goal]
        _sn, stats = HD.statements_chunked([{"start": s, "moves": mv} for s, mv in wires], self.threads)
        off = 0
        K, S, Q, OK = [], [], [], []
        for (_s, mv), st in zip(wires, stats):
            n = len(mv) + 1
            last = flat[off:off + n * w].reshape(n, w)[-1]
            off += n * w
            K.append((last[2:2 + nk] - last[2 + nk:2 + 2 * nk])[use])
            row = st[n - 1] if len(st) >= n else np.full(2 * WIDTH["S"], np.nan)
            good = not bool(np.isnan(row).any())
            S.append(row[:WIDTH["S"]] - row[WIDTH["S"]:2 * WIDTH["S"]] if good else np.full(WIDTH["S"], np.nan))
            Q.append(bool(last[0] == 1 and last[1] == 1))
            OK.append(good)
        out = {"K": np.array(K, float), "S": np.array(S, float), "quiet": np.array(Q), "ok": np.array(OK)}
        if self.concepts:
            raw_c = self.model.pattern_count(wires, [[len(mv)] for _s, mv in wires], self.keys, 3, list(LAYER2.LEVELS), self.goal, self.threads)
            us, them = LAYER2.counts_matrix(raw_c, len(wires), len(self.keys))
            out["C"] = (us - them).toarray().astype(float)
        return out


def root_text(moves: list) -> str:
    """The engine's own text of the position after `moves` (the start of the short wires)."""
    return CE.state(HD.VARIANT, None, list(moves))[0]


def block_row(blk: dict, i: int) -> dict:
    return {"K": [int(v) for v in blk["K"][i]], "S": [float(v) for v in blk["S"][i]], "C": [int(v) for v in blk["C"][i]] if "C" in blk else None, "quiet": bool(blk["quiet"][i]), "ok": bool(blk["ok"][i])}


def tails_of(rec: dict, rows: list, longer: "callable") -> "tuple[list, list]":
    """(the tails after the root at the end of each equal move's line, the number of lines read again): the line starts with the move and its first `settled` names are the line to its settled end. A line longer than the names kept by 79.01 is asked of `longer(move) -> names`."""
    tails, again = [], 0
    for m, r in zip(rec["tied"], rows):
        names = list(r["names"])
        if len(names) < r["settled"]:
            names = list(longer(m))
            again += 1
        tails.append(names[:r["settled"]])
    return tails, again


def read_position(rd: Reader, rec: dict, moves: list, rows: list, longer: "callable") -> dict:
    start = root_text(moves)
    tails, again = tails_of(rec, rows, longer)
    wires = [(start, [m]) for m in rec["tied"]] + [(start, t) for t in tails]
    blk = rd.read(wires)
    n = len(rec["tied"])
    return {"id": rec["id"], "ply": len(moves), "again": again,
            "rows": [{"move": m, "settled": r["settled"], "starts_with_move": bool(tails[j] and tails[j][0] == m), "end_is_root": bool(r["end_is_root"]), "one": block_row(blk, j), "end": block_row(blk, n + j)} for j, (m, r) in enumerate(zip(rec["tied"], rows))]}


def same_read_whole_game(rd: Reader, items: list, lines: dict, say) -> None:
    """The short wire against the whole game from the start, on a sample of end wires: statements, kinds, concepts must be equal."""
    rng = np.random.default_rng(7904)
    picks = []
    for k in rng.permutation(len(items))[:CHECK_WIRES]:
        rec, moves = items[int(k)]
        ln = lines.get(rec["id"])
        if not ln or "rows" not in ln or not ln["rows"][0]["ok"]:
            continue
        r = ln["rows"][0]
        picks.append((moves, r["names"][:r["settled"]]))
    short = rd.read([(root_text(mv), t) for mv, t in picks])
    whole = rd.read([(None, list(mv) + t) for mv, t in picks])
    ds = np.nanmax(np.abs(short["S"] - whole["S"]))
    dk = np.abs(short["K"] - whole["K"]).max()
    dc = np.abs(short["C"] - whole["C"]).max() if "C" in short else 0.0
    say(f"SAME READ: the short wire against the whole game from the start, {len(picks)} end-of-line wires: largest difference in the statements {ds:g}, in the kinds {dk:g}, in the concepts {dc:g}; quiet flags equal {bool((short['quiet'] == whole['quiet']).all())}")


def run(tag: str, limit: int, threads: int, say) -> None:
    root = out_root(tag)
    (root / "blocks").mkdir(exist_ok=True)
    items = LL.items_of(SRC_TAG)
    lines = {r["id"]: r for r in LL.load(LF.out_root(SRC_TAG))}
    if limit:
        items = items[:limit]
    items = [(rec, mv) for rec, mv in items if rec["id"] in lines and "rows" in lines[rec["id"]] and all(x["ok"] for x in lines[rec["id"]]["rows"])]
    say(f"firm-room positions to read {len(items)} (of 79.01's firm room, {'the first ' + str(limit) if limit else 'all'}; positions whose lines were all read); engine threads {threads}; load average {os.getloadavg()[0]:.2f}")
    rd = Reader(threads)
    same_read_whole_game(rd, items, lines, say)
    lz = {"x": None}

    def longer_of(rec, moves):
        def f(m):
            if lz["x"] is None:
                lz["x"] = LL.Lines()
            return lz["x"].line(moves, m, rec["depth"])["names"]
        return f
    t0 = time.perf_counter()
    for a in range(0, len(items), CHUNK):
        f = root / "blocks" / f"c{a // CHUNK:05d}.json"
        if f.exists():
            continue
        out = [read_position(rd, rec, mv, lines[rec["id"]]["rows"], longer_of(rec, mv)) for rec, mv in items[a:a + CHUNK]]
        f.write_text(json.dumps(out))
        say(f"progress: {min(a + CHUNK, len(items))} of {len(items)} positions; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    allp = []
    for f in sorted((root / "blocks").glob("c*.json")):
        allp += json.loads(f.read_text())
    (root / "blocks.jsonl").write_text("\n".join(json.dumps(p) for p in allp) + "\n")
    nrows = sum(len(p["rows"]) for p in allp)
    bad = sum(1 for p in allp for r in p["rows"] if not (r["one"]["ok"] and r["end"]["ok"]))
    quiet_end = sum(1 for p in allp for r in p["rows"] if r["end"]["quiet"])
    quiet_one = sum(1 for p in allp for r in p["rows"] if r["one"]["quiet"])
    mism = sum(1 for p in allp for r in p["rows"] if not r["starts_with_move"])
    say(f"read {len(allp)} positions, {nrows} equal moves in {time.perf_counter() - t0:.0f} s wall; lines read again for their length ({LL.NAMES_SHOWN} names kept): {sum(p['again'] for p in allp)}; moves whose statements are not defined at one end (the game is over there): {bad}; line not starting with its move: {mism}; quiet one move after the root {100.0 * quiet_one / max(1, nrows):.1f} pct, quiet at the end of the line {100.0 * quiet_end / max(1, nrows):.1f} pct")


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "blocks.jsonl").read_text().splitlines() if x.strip()]


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["blocks"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    root = out_root(a.tag)

    def say(msg: str) -> None:
        line = f"OBS heads read: {msg}"
        print(line, flush=True)
        with open(root / "obs.txt", "a") as f:
            f.write(line + "\n")
    run(a.tag, a.limit, a.threads, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
