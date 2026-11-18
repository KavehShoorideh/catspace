"""PLAN 78.84 PART ONE: WHY THE SEARCH HOLDS A MOVE THAT GIVES UP A UNIT EQUAL TO A SOUND ONE. SYSTEM_SPEC 5 (the tactical reasoner decides; the hint only orders what it holds equal), 7, 8. A measurement with pieces that
exist: no new method, no player changed, nothing written to rules/learned or data/derived. Game-free: the game is named only through the pieces that exist; the printing of the cases is ladder_print's (`--cut84`).

INTERFACE (what each stage takes, what it returns, who calls it; written before the bodies).
  collect                  takes the match files of 78.82 and 78.83 (read only; every `*_v_control_*` match of a reasons-side player: its decision log `dlog` names the ply of each tie decision) and returns decisions.json:
                           the tie decisions that can be rebuilt, by a rule fixed here before any was looked at (below).
  read                     takes decisions.json; for each decision plays the search again at the match's own settings (0.2 s, ledger share TM.LEDGER_SHARE, the record the matches judge by) on the position, reads the line of
                           every tied move with symbols_play.LineCut (the line reading of threat_concrete, the settings of ladder_measure.Tools.line), and for the decisions where the tie set holds a move whose line gives up a unit for nothing
                           AND a move whose line does not, asks the search for the numbers of the moves themselves (below). Writes reads.jsonl (one row a decision; a rerun skips the rows done).
  cutstats                 (part two) reads the match files of the cut players and prints how often the cut read a tie set, dropped a move, dropped the search's first move, and what it cost, by band.
  report                   takes reads.jsonl and prints, per band of units on the board (LM.BANDS), the counts and the causes as OBS lines; writes cases84.json (twenty flagged decisions by a fixed rule) for ladder_print `--cut84`.
Called by: the command line only.

WHAT CAN BE REBUILT. The match files keep, per tie decision of the tested side, its ply, the size of the tie set and the cost, and the full tie set only where the pick differed from the search's first move. The search's own numbers (depth, value,
bounds) are not kept. So the tie decisions are rebuilt as POSITIONS (the moves played to the decision, from the stored game) and the search is PLAYED AGAIN on each at the match's settings: the tie set read again can differ from the stored one (a clock
of 0.2 s, a loaded machine); `report` prints how often the stored set (where there is one) is the set found again.

THE RULE FOR THE DECISIONS (fixed before any was read): every tie decision logged by the tested side in the match files of 78.82 (reasons, quiet, withctl, and the charged reasons, each v control, sets A and B) and of 78.83 (the four windows v
control, sets A and B); a position (the moves played) counts once; each is put in its band by its count of units (LM.BANDS); the positions of a band are ordered by the hash of their moves (sha1 of the moves joined by a space) and the first N_BAND
are taken (every one when a band holds fewer).

WHAT THE SEARCH REPORTS ABOUT A MOVE. `solve_equals_batch` hands back, for the position, the depth the ledger finished (D) and its value (v), whether the proof or the clock stopped the equal reading, and the moves it holds equal: a root move is
asked "at least the best at depth D?" by a null window, so equality is a BOUND, not a value; it reports no value per move. For the numbers of the two moves we read each as the search would: the position after the move, searched (1.0 s) with the
ledger deepening capped at D - 1; the move's value at depth D is then what the move itself takes (its own gain by the record) less the child's value (the child's mover is the other side; the judge is zero-sum), and the same for the
search's own move, which must equal v (the CONSISTENCY of this reading, printed). The bad move is also read at D + 1 (child at D) to see whether the search itself would see the loss with two more plies.
The line reading of a move (mode 9) is a fresh ledger search of the position after the move to a depth of TC.DEPTH + 1 plies after the move, its value `net_search` (the move's own gain included, for the mover) is the search's own number for the move at 8 plies from
the root; its events (what is lost and won) are read off the line the search printed (the principal line), and `verdict_of` judges those events; the two can disagree (`pv_disagrees`).
CAUSES (counted for each flagged decision; the first that holds): DECIDED (the value is a won or lost game: the equal reading is skipped by the search), UNREAD (the child search did not finish depth D - 1
in its second), BELOW BEST (the move's value at depth D is under v: the search reports it equal to a best it is not at: a fault of the reading), then, the move's value at D being v or more: LINE VALUE NOT BEHIND (the line search's own value of the
bad move is v or more while the events on its printed line show a loss: the loss is not what the line search itself counts), BEYOND DEPTH (the line's value is under v and the line reaches deeper than D: held equal rightly at depth D; the loss lies beyond D;
SEEN DEEPER when the bad move at D + 1 falls under v, else NOT SEEN), SEARCH DEEPER THAN LINE (the line's value is under v although the search reached as deep as the line or deeper: the search sees farther than the line, which
the loss does not survive, or two searches give different numbers for the one move).

  .venv/bin/python -m catspace.incubator.analyst.discovery.tie_cut_measure collect | read | report | cutstats [--tag full]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.io import paths

OUT = ("symbols_play_78_84",)
SOURCES = (("symbols_play_78_82", "full"), ("symbols_play_78_83", "full"))        # read only
N_BAND = 1200                                                     # decisions asked of each band (at least a thousand)
CHILD_SECONDS = 1.0                                               # the clock of a child search for the numbers of a move
CHILD_SHARE = 0.9
WIN = 1 << 40                                                     # the engine's value of a won game (rust/engine/tactics ledger)
CASES_SPREAD = {"few": 7, "mid": 7, "many": 6}                    # the twenty cases: the first of each band's flagged decisions in hash order
CAUSES = ("DECIDED", "UNREAD", "BELOW BEST", "LINE VALUE NOT BEHIND", "BEYOND DEPTH, SEEN DEEPER", "BEYOND DEPTH, NOT SEEN", "SEARCH DEEPER THAN LINE")


def out_root(tag: str = "full") -> Path:
    return Path(paths.experiment(*OUT, tag))


def key_of(moves: list) -> str:
    return hashlib.sha1(" ".join(moves).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ collect

def stored_games() -> list:
    out = []
    for src in SOURCES:
        for p in sorted(Path(paths.experiment(*src, "matches")).glob("*_v_control_*.jsonl")):
            out.append((src[0], p.name, p))
    return out


def collect(n_band: int, say) -> list:
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    pool: dict = {}                                               # key of the moves -> decision
    for exp, name, p in stored_games():
        for g in TM.read_games(p):
            st = g["stats"][TM.seat_of(g, 0)]
            if not st.get("dlog"):
                continue
            counts = LM.unit_counts(layer1, g["moves"])
            stored = {tuple(c["moves"]): c["tied"] for c in st["cases"]}
            for ply, n_tied, _ch, _ms in st["dlog"]:
                moves = list(g["moves"][:ply])
                k = key_of(moves)
                if k in pool:
                    continue
                pool[k] = {"key": k, "exp": exp, "file": name, "gid": g["gid"], "ply": ply, "moves": moves, "units": counts[ply], "band": LM.band_of(counts[ply]), "stored_n": n_tied,
                           "stored_tied": stored.get(tuple(moves))}
    rows = []
    for band, _lo, _hi in LM.BANDS:
        mine = sorted((d for d in pool.values() if d["band"] == band), key=lambda d: d["key"])
        say(f"COLLECT {band}: {len(mine)} distinct tie decisions in the stored matches; the first {min(n_band, len(mine))} by hash order are read")
        rows += mine[:n_band]
    say(f"COLLECT {len(rows)} decisions from {len(stored_games())} match files")
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ read

_W: dict = {}


def _tools() -> dict:
    if not _W:
        base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
        _W["worth"] = dict(base["worth"])
        _W["facts"] = SG.Facts(HD.VARIANT)
    return _W


def search_row(moves: list, seconds: float, share: float, depth_cap: "int | None" = None) -> dict:
    kw = {} if depth_cap is None else {"ledger_depth": int(depth_cap)}
    return PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], seconds, worth=dict(_tools()["worth"]), ours_first=False, valuation="odds record", threads=1, ledger_share=share, **kw)[0])


def own_gain(moves: list, move: str) -> int:
    """What the move itself adds for the mover, by the record, as the ledger counts it: the worth of the unit it takes plus, when the unit that moves changes kind, the worth of the new kind less the old one's (0 when neither)."""
    t = _tools()
    word = OM.kernel_names()                                       # the replay names a kind in the game's words, the record by its letter
    text = CE.state(HD.VARIANT, None, list(moves))[0]
    pl = t["facts"].replay(text, [move])
    if not pl:
        return 0
    gain = t["worth"].get(word.get(pl[0]["took"][2], ""), 0) if pl[0]["took"] else 0
    old = (pl[0]["uid"] or "::").split(":")[2]
    if pl[0]["kind"] and old and old != pl[0]["kind"]:
        gain += t["worth"].get(word.get(pl[0]["kind"], ""), 0) - t["worth"].get(word.get(old, ""), 0)
    return gain


def child_number(moves: list, move: str, depth: int, seconds: float = CHILD_SECONDS) -> dict:
    """The move's value at `depth` as the search reads it: its own gain less the value of the position after it searched with the ledger capped at depth - 1 (the other side to move; the judge is zero-sum)."""
    gain = own_gain(moves, move)
    try:
        r = search_row(list(moves) + [move], seconds, CHILD_SHARE, max(depth - 1, 1))
    except Exception as e:                                        # the position after the move is over
        return {"gain": gain, "error": str(e)[:80]}
    if r["value"] is None:
        return {"gain": gain, "child_depth": r["depth"], "proved": r["proved"], "value": None}
    return {"gain": gain, "child_depth": r["depth"], "child_value": r["value"], "value": gain - r["value"], "proved": r["proved"]}


def compact(v: "dict | None") -> "dict | None":
    if v is None:
        return None
    return {"verdict": v["verdict"], "lost": v["left_lost"], "won": v["left_won"], "net": v["net"], "res": v["resolution"], "end": v["end"], "depth": v["depth"], "nodes": v["nodes"],
            "names": v["full_names"][:9], "ev_lost": v["taker_lost"], "ev_won": v["taker_won"], "give": SP.gives_up(v), "net_search": v["net_search"], "pv_dis": v["pv_disagrees"]}


def read_decision(d: dict) -> dict:
    moves = d["moves"]
    t0 = time.perf_counter()
    row = search_row(moves, TM.SECONDS, TM.LEDGER_SHARE)
    t1 = time.perf_counter()
    out = {"key": d["key"], "band": d["band"], "units": d["units"], "ply": d["ply"], "search": {k: row[k] for k in ("move", "proved", "nodes", "seconds", "depth", "value", "ties_seconds", "legal", "cut")},
           "equals": row["equals"], "search_wall": t1 - t0}
    mv, eq = row["move"], row["equals"]
    if mv is None or len(eq) < 2:
        out["tied"] = None
        return out
    tied = [mv] + sorted(m for m in eq if m != mv)
    out["tied"] = tied
    c0 = time.perf_counter()
    ver = SP.cutter().reads(moves, tied)
    out["lines_wall"] = time.perf_counter() - c0
    out["lines"] = {m: compact(ver[m]) for m in tied}
    bad = [m for m in tied if SP.gives_up(ver[m])]
    good = [m for m in tied if m not in bad]
    out["bad"], out["good"] = bad, good
    if bad and good:
        b, g = bad[0], good[0]
        D = row["depth"] or 1
        out["pair"] = {"bad": b, "good": g}
        out["numbers"] = {"bad": child_number(moves, b, D), "good": child_number(moves, g, D)}
        if mv not in (b, g):
            out["numbers"]["search"] = child_number(moves, mv, D)
        else:
            out["numbers"]["search"] = out["numbers"]["bad" if mv == b else "good"]
        out["numbers"]["bad_deeper"] = child_number(moves, b, D + 2, 2.0)
    return out


def read_chunk(chunk: list) -> list:
    return [read_decision(d) for d in chunk]


def run_reads(rows: list, workers: int, out_file: Path, say) -> None:
    done = set()
    if out_file.exists():
        done = {json.loads(x)["key"] for x in out_file.read_text().splitlines() if x.strip()}
    todo = [d for d in rows if d["key"] not in done]
    chunks = [todo[i:i + 6] for i in range(0, len(todo), 6)]
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    n = len(done)
    with open(out_file, "a") as f, ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(read_chunk, chunks):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += len(res)
            if (n // 6) % 25 == 0:
                say(f"READ {n} of {len(rows)} decisions after {time.perf_counter() - t0:.0f} s")
    say(f"READ done: {len(rows)} decisions in {time.perf_counter() - t0:.0f} s, {workers} workers")


# ------------------------------------------------------------------------------------------------------------------------------ report

def cause_of(r: dict) -> str:
    """The first cause that holds (the clock cutting the equal reading is counted apart: a move in a cut list was held at the best before the clock stopped, so the cut does not say why it is there)."""
    s = r["search"]
    D, v = s["depth"] or 1, s["value"]
    nb = r["numbers"]["bad"]
    if v is not None and abs(v) >= WIN - 1000:
        return "DECIDED"
    if nb.get("value") is None or nb.get("child_depth") is None or nb["child_depth"] < D - 1:
        return "UNREAD"
    if v is not None and nb["value"] < v:
        return "BELOW BEST"
    lb = r["lines"][r["pair"]["bad"]]
    if v is not None and lb["net_search"] >= v:
        return "LINE VALUE NOT BEHIND"
    if D >= 1 + lb["depth"]:
        return "SEARCH DEEPER THAN LINE"
    deep = r["numbers"]["bad_deeper"]
    if v is not None and deep.get("value") is not None and deep.get("child_depth", 0) >= D + 1 and deep["value"] < v:
        return "BEYOND DEPTH, SEEN DEEPER"
    return "BEYOND DEPTH, NOT SEEN"


def _agree(c: "dict | None") -> "dict | None":
    """A stored line as the verdict dict `symbols_play.gives_up` reads, with its loss counted only where the line's own value agrees (SP.agrees)."""
    if c is None:
        return None
    return {"verdict": c["verdict"], "end": c["end"], "left_lost": c["lost"], "left_won": c["won"], "net_search": c["net_search"], "resolution": c["res"]}


def report(tag: str, say) -> None:
    root = out_root(tag)
    dec = {d["key"]: d for d in json.loads((root / "decisions.json").read_text())}
    rows = [json.loads(x) for x in (root / "reads.jsonl").read_text().splitlines() if x.strip()]
    say(f"REPORT {len(rows)} decisions read of {len(dec)} asked")
    cases = []
    for band, _lo, _hi in LM.BANDS:
        rs = [r for r in rows if r["band"] == band]
        tied = [r for r in rs if r["tied"]]
        n_moves = sum(len(r["tied"]) for r in tied)
        some_bad = [r for r in tied if r["bad"]]
        flagged = [r for r in tied if r["bad"] and r["good"]]
        all_bad = [r for r in tied if r["bad"] and not r["good"]]
        bad_moves = sum(len(r["bad"]) for r in tied)
        dropped = sum(len(r["bad"]) for r in flagged)
        lost_game = sum(1 for r in tied for m in r["tied"] if (r["lines"][m] or {}).get("end") == 2)
        unread = sum(1 for r in tied for m in r["tied"] if r["lines"][m] and r["lines"][m]["verdict"] == "UNREAD")
        no_line = sum(1 for r in tied for m in r["tied"] if r["lines"][m] is None)
        same = [r for r in tied if dec[r["key"]]["stored_tied"] is not None]
        eq = sum(1 for r in same if set(r["tied"]) == set(dec[r["key"]]["stored_tied"]))
        sizes = [len(r["tied"]) for r in tied]
        ms = [1000 * r["lines_wall"] for r in tied]
        say(f"BAND {band}: decisions read {len(rs)}; tie set found again in {len(tied)} ({100 * len(tied) / max(len(rs), 1):.1f} pct), mean {np.mean(sizes):.1f} tied moves ({n_moves} moves); "
            f"stored tie set (kept where the pick differed) is the set found again in {eq} of {len(same)}; "
            f"tie sets with a move that gives up a unit for nothing: {len(some_bad)} ({100 * len(some_bad) / max(len(tied), 1):.1f} pct); of them FLAGGED (another tied move does not): {len(flagged)} "
            f"({100 * len(flagged) / max(len(tied), 1):.1f} pct of tie sets), all tied moves give up: {len(all_bad)}; moves that give up {bad_moves} of {n_moves} ({100 * bad_moves / max(n_moves, 1):.1f} pct); "
            f"the cut would drop {dropped} moves; lines ending in the game lost for the mover {lost_game}, lines cut by the fuse {unread}, moves with no line read {no_line}; "
            f"cost of reading the lines of a tie set mean {np.mean(ms):.0f} ms median {np.median(ms):.0f} ms")
        both = [r for r in tied if SP.cut_of({m: _agree(r["lines"][m]) for m in r["tied"]}, r["tied"], True) != r["tied"]]
        n_both = sum(len(r["tied"]) - len(SP.cut_of({m: _agree(r["lines"][m]) for m in r["tied"]}, r["tied"], True)) for r in both)
        say(f"BAND {band} the cut when the line's own value agrees with its events (the loss also shows in the line search's value: value under minus the resolution): tie sets cut {len(both)} ({100 * len(both) / max(len(tied), 1):.1f} pct), "
            f"moves dropped {n_both}; the cut by the events alone drops {dropped}")
        causes = [cause_of(r) for r in flagged]
        say(f"BAND {band} CAUSES of the {len(flagged)} flagged decisions: " + "; ".join(f"{c} {causes.count(c)}" for c in CAUSES))
        if flagged:
            cons = [r for r in flagged if r["numbers"]["search"].get("value") is not None and r["numbers"]["search"].get("child_depth", 0) >= (r["search"]["depth"] or 1) - 1 and r["search"]["value"] is not None]
            ok = sum(1 for r in cons if r["numbers"]["search"]["value"] == r["search"]["value"])
            say(f"BAND {band} CONSISTENCY of the number of a move: the search's own move read this way equals the search's value in {ok} of {len(cons)} flagged decisions where its child finished depth D - 1")
            deep = [r for r in flagged if r["numbers"]["bad_deeper"].get("value") is not None and r["numbers"]["bad_deeper"].get("child_depth", 0) >= (r["search"]["depth"] or 1) + 1 and r["search"]["value"] is not None]
            say(f"BAND {band} the bad move at D + 1 finished in {len(deep)} of {len(flagged)}; its value fell under the search's value in {sum(1 for r in deep if r['numbers']['bad_deeper']['value'] < r['search']['value'])}")
            say(f"BAND {band} the bad move's printed line: events disagree with the line's own value in {sum(1 for r in flagged if r['lines'][r['pair']['bad']]['pv_dis'])} of {len(flagged)}; "
                f"the line's own value is under the search's value in {sum(1 for r in flagged if r['search']['value'] is not None and r['lines'][r['pair']['bad']]['net_search'] < r['search']['value'])}; "
                f"the good move's line is above the search's value in {sum(1 for r in flagged if r['search']['value'] is not None and r['lines'][r['pair']['good']]['net_search'] > r['search']['value'])} "
                f"(the line is shallower than the search and misses what the search sees)")
            ds = [r["search"]["depth"] for r in flagged if r["search"]["depth"] is not None]
            say(f"BAND {band} depth the search reached on the flagged decisions: mean {np.mean(ds):.1f}, min {min(ds)}, max {max(ds)}; ties cut by the clock {sum(1 for r in flagged if r['search']['cut'])}; "
                f"proved {sum(1 for r in flagged if r['search']['proved'])}; bad line depth mean {np.mean([r['lines'][r['pair']['bad']]['depth'] for r in flagged]):.1f} plies")
        for r in sorted(flagged, key=lambda r: r["key"])[:CASES_SPREAD[band]]:
            cases.append(dict(r, cause=cause_of(r), moves=dec[r["key"]]["moves"], gid=dec[r["key"]]["gid"]))
    (root / "cases84.json").write_text(json.dumps(cases))
    say(f"CASES84 {len(cases)} flagged decisions saved to {root / 'cases84.json'} (the first of each band's flagged decisions in hash order: " + ", ".join(f"{b} {CASES_SPREAD[b]}" for b in CASES_SPREAD) + ")")


# ------------------------------------------------------------------------------------------------------------------------------ part two: what the cut did in the matches

def cut_stats(tag: str, say) -> None:
    """From the match files of the cut players (symbols_play `match`): per player and set, how often the cut read a tie set, how often it dropped a move, how often it dropped the search's first move, and its cost; by band of units."""
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    for p in sorted((out_root(tag) / "matches").glob("*_v_control_*.jsonl")):
        stem = p.stem.removesuffix("_charged")
        a, rest = stem.split("_v_")
        if "sy_" + a not in SP.CUT_KINDS:
            continue
        games = TM.read_games(p)
        rows = []                                                 # (band, tied, dropped, first dropped, ms)
        n_dec = n_ties = 0
        for g in games:
            st = g["stats"][TM.seat_of(g, 0)]
            n_dec += st["decisions"]
            n_ties += st["ties"]
            counts = LM.unit_counts(layer1, g["moves"]) if st["cut_log"] else []
            for ply, n_tied, dropped, first, ms in st["cut_log"]:
                rows.append((LM.band_of(counts[ply]), n_tied, dropped, first, ms))
        if not rows:
            say(f"CUTSTATS {p.stem}: no tie decision was cut-read")
            continue

        def line(label: str, sel: list) -> str:
            if not sel:
                return f"CUTSTATS {p.stem} {label}: none"
            d = [r for r in sel if r[2] > 0]
            return (f"CUTSTATS {p.stem} {label}: {len(sel)} tie sets read; the cut dropped a move in {len(d)} ({100 * len(d) / len(sel):.1f} pct), {sum(r[2] for r in sel)} moves of {sum(r[1] for r in sel)} "
                    f"({100 * sum(r[2] for r in sel) / sum(r[1] for r in sel):.1f} pct); the search's first move was dropped in {sum(r[3] for r in sel)} ({100 * sum(r[3] for r in sel) / len(sel):.1f} pct of the tie sets); "
                    f"cost of the cut a tie set: mean {np.mean([r[4] for r in sel]):.0f} ms, median {np.median([r[4] for r in sel]):.0f} ms")
        say(line("all bands", rows) + f"; of {n_dec} decisions {n_ties} had two or more equal moves")
        for band, _lo, _hi in LM.BANDS:
            say(line(f"band {band}", [r for r in rows if r[0] == band]))


# ------------------------------------------------------------------------------------------------------------------------------ the run

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("collect", "read", "report", "cutstats"))
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n", type=int, default=N_BAND)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"

    def say(line: str) -> None:
        TM.emit([f"OBS tie_cut {line}"], obs)
    if a.stage == "collect":
        rows = collect(a.n, say)
        (root / "decisions.json").write_text(json.dumps(rows))
    elif a.stage == "read":
        rows = json.loads((root / "decisions.json").read_text())
        load, busy = TM.wait_quiet(lambda s: TM.emit([s], obs))
        say(f"READ start: load average {load:.2f}, cores busy {busy:.2f}")
        run_reads(rows, a.workers, root / "reads.jsonl", say)
    elif a.stage == "cutstats":
        cut_stats(a.tag, say)
    else:
        report(a.tag, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
