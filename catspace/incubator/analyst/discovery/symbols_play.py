"""PLAN 78.82: THE REASONS OF 78.81 AS THE HINT IN PLAY, in the match harness of 78.67 and 78.75. SYSTEM_SPEC 7 (the hint among the moves the search holds equal), 8 (the wake), 9 (the wake's match answers one question
only: whether surprises resolved turn into better play). A sandbox: nothing is wired into any player of the wake, nothing is written to rules/learned or data/derived; every output goes under
artifacts/experiments/symbols_play_82/ (symbols_play_78_82). Game-free: the game is named only through the pieces that exist (odds_measure, human_heads_data, graph_layer1, subgoal_goals, ladder_measure's unit count); the
printing of the cases is ladder_print's.

THE HINT IS NEVER A VALUE: it orders the moves INSIDE the tie set the search hands back; the search's value, its depth and its first move are the control's.

INTERFACE (what each piece takes, what it returns, who calls it). Written before the bodies.
  ReasonsReader(root=None)        per process (use reader82()): loads ONCE the reasons of 78.81 (symbols_78_81/full/reasons.json, the weights of the joint fit on the FIND part; refitted nothing) and the weights of the two
                                  controls of that fit (fits.pkl, key "both"), builds the engine's pattern counter for the reasons' keys and the replay reader of the game.
      .pick(moves, tied, mode) -> (move, detail)
                                  `moves` the game so far, `tied` the moves the search holds equal with the search's first move FIRST. `mode`: "reasons" (the sum of the weights of the reasons a move carries; symbols.symbol_pick's
                                  rule), "quiet" (the reasons' pick only where NO tied move takes a unit, else the search's first move), "withctl" (the reasons' weights plus the two controls' weights: takes a unit, the kind of unit that
                                  moves; whether the controls may enter the hint is open question 6 of the spec, so they are a separate player). The move returned is ALWAYS in `tied`; a tie of scores, or no reason that separates the
                                  moves, gives the search's first move (tied[0]). One engine call counts the reasons for the position and every tied move. detail: {"scores": {move: score}, "carried": {move: [[reason index, weight]]},
                                  "takes": {move: bool} or None, "kinds": {move: kind word} or None}.
      (the agreement of the pick with symbols_reasons_measure's pick is the stage `agreement`.)
  MoverReasons(TM.Mover67)        the movers of 78.67 plus the kinds `sy_reasons`, `sy_quiet`, `sy_withctl`: the SAME search call as every player of 78.67 (`solve_equals_batch`, the equal moves handed back, the search's own
      .pick(position, tied) -> move   move first); then, for a position with two or more equal moves, `pick` ({"moves": [...]}, the tied moves) chooses one of them. `--charge`: the pick's reading time (the running mean of the
                                  picks so far) is taken out of the search's clock, as 78.75 did. Every tie decision is logged in stats["dlog"]; a decision where the pick differed from the search's first move is kept in full in stats["cases"].
  play82(spec) -> game            one game with these movers (a module-level function: the Pool spawns workers that import it). Called by tiebreak_measure.run_games(play=play82).
  stages (command line)           agreement | match | report | cases. agreement: on 300 positions of the 78.80 CONFIRM sample the player's pick against the pick symbols_reasons_measure computes for the same tie set. match: one match, the load
                                  wait of 78.67 first; report: every match found, per set and pooled, the shares and the cost; cases: thirty decisions by a rule fixed here (below) for ladder_print `--play82`; cases83 (78.83): thirty decisions, ten a band, from 78.82's set B games, for ladder_print `--play83`.
PLAN 78.83 (SYSTEM_SPEC 7, 8, 9): WHERE IN THE GAME THE GAIN COMES FROM. Interface added, written before the bodies:
  MoverReasons.window = (low, high) | None     the count of units on the board (inclusive) inside which the hint is used; None (the default) is every position. A tie decision whose position lies OUTSIDE the window is
                                  played as the control plays it: the search's first move, no pick, no reading cost. The window comes from the player's kind (WINDOW: the kinds `sy_many`, `sy_mid`, `sy_few`, `sy_midfew`, the bands of
                                  ladder_measure.BANDS: few 0-8, mid 9-16, many 17 and up; all four pick in mode "reasons") or from spec["window"]. The unit count is read from the carried state of the position.
  ReasonsReader.state_of(moves) -> text        the state of the game after `moves`, kept between decisions: the next call advances the kept state by the moves played since (the engine replays the game only the first time).
  ReasonsReader.units_of(moves) -> int         the count of units on the board in that state (the layer-1 table of one position).
  The kept state serves the unit count and the controls (the replay reading). The counts of the reasons (symbols.deltas_of) still read the game from its first position: handing the engine the kept state there was tried and
  cost the same (the cost is the pattern count, not the replay: a throwaway timing of 181 decisions, OBS COST lines in the 78.83 obs.txt), so it was not kept in the code.
  stats: "ties" every tie decision; "ties_in" those inside the window (the ones that were picked and logged in dlog/cases). The match stage writes matches/<file>.meta.json (the load seen, cores busy, wall) which `report` prints.
Called by: the command line only. The harness is tiebreak_measure.py (the openings, the clock, the load wait, the match loop and its lines), reused as it is.

THE RULE FOR THE THIRTY DECISIONS (fixed before any game was read): in the games of reasons v control on opening set A, the games in the order of their ids; in each game the first decision of the reasons' side where the pick
differed from the search's first move; the first thirty games that have one.

  .venv/bin/python -m catspace.incubator.analyst.discovery.symbols_play agreement | match --a reasons --b control --set A --pairs 10 --tag short | report --tag T | cases --tag T
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.io import paths

OUT = ("symbols_play_78_83",)                                     # this run's outputs (78.82's run lives in symbols_play_78_82/full, read only)
OUT82 = ("symbols_play_78_82", "full")
SRC = ("symbols_78_81", "full")                                   # the reasons and their weights: 78.81's run, read, never refitted
KIND = dict(TM.KIND, reasons="sy_reasons", quiet="sy_quiet", withctl="sy_withctl", many="sy_many", mid="sy_mid", few="sy_few", midfew="sy_midfew")
MODE = {"sy_reasons": "reasons", "sy_quiet": "quiet", "sy_withctl": "withctl", "sy_many": "reasons", "sy_mid": "reasons", "sy_few": "reasons", "sy_midfew": "reasons"}
_BAND = {n: (lo, hi) for n, lo, hi in LM.BANDS}
WINDOW = {"sy_many": _BAND["many"], "sy_mid": _BAND["mid"], "sy_few": _BAND["few"], "sy_midfew": (_BAND["few"][0], _BAND["mid"][1])}
PLAYERS = tuple(KIND)
MODE_PLAYERS = tuple(p for p in KIND if KIND[p] in MODE)
TIE_DIGITS = 9                                                    # two scores that agree to this many places are equal (sums of the same weights in another order differ by rounding only)
AGREE_N = 300                                                     # positions of the CONFIRM sample the agreement reads (evenly spaced in the sample's order)
CASES_N = 30
_READER = None


def out_root(tag: str = "full") -> Path:
    return Path(paths.experiment(*OUT, tag))


def src_root() -> Path:
    return Path(paths.experiment(*SRC))


# ------------------------------------------------------------------------------------------------------------------------------ the reader of the reasons

class ReasonsReader:
    """See the module doc. One per process."""

    def __init__(self, root: "Path | None" = None):
        root = root or src_root()
        self.reasons = json.loads((root / "reasons.json").read_text())
        fits = pickle.loads((root / "fits.pkl").read_bytes())
        self.ctl_weights = dict(fits["both"]["controls"])         # {"takes": w, "moves:<kind>": w, ...}: the controls of the joint fit
        keys = sorted({int(r["key"]) for r in self.reasons})
        col = {k: j for j, k in enumerate(keys)}
        self.weight = {(col[int(r["key"])], r["side"], r["dir"]): float(r["weight"]) for r in self.reasons}
        self.index = {(col[int(r["key"])], r["side"], r["dir"]): i for i, r in enumerate(self.reasons)}
        self.order = sorted(self.weight, key=lambda e: self.index[e])      # the reasons' own order: the sums are always added in it
        self.counter = SY.Counter(GL.model(HD.GAME, HD.VARIANT), keys, G2M.goal_kind_of(HD.VARIANT), 2, (0, 1, 2, 3), 1)
        self.facts = None
        self._state = None                                        # (the moves of the game so far, the state text after them): kept between decisions of one game

    def state_of(self, moves: list) -> str:
        """The state text after `moves`; the kept state is advanced by the moves played since the last call (a game replays once), read again from the first position when `moves` does not extend it."""
        kept = getattr(self, "_state", None)
        n = len(moves)
        if kept and n >= len(kept[0]) and list(moves[:len(kept[0])]) == kept[0]:
            text = kept[1] if n == len(kept[0]) else CE.state(HD.VARIANT, kept[1], list(moves[len(kept[0]):]))[0]
        else:
            text = CE.state(HD.VARIANT, None, list(moves))[0]
        self._state = (list(moves), text)
        return text

    def units_of(self, moves: list) -> int:
        """The count of units on the board after `moves` (the layer-1 table of the one position, read from the kept state)."""
        names, raw = self.counter.model.position_table([(self.state_of(moves), [])], 1)
        return int(np.frombuffer(raw, dtype=np.int16).reshape(1, 2 + 2 * len(names)).astype(np.int32)[0, 2:].sum())

    def controls_of(self, moves: list, tied: list) -> "tuple[dict, dict]":
        """For each tied move: whether it takes a unit and the kind word of the unit that moves, from the board's own difference across the move (subgoal_goals.Facts.replay, the same reading 78.81's controls used)."""
        if self.facts is None:
            self.facts = SG.Facts(HD.VARIANT)
        text = self.state_of(moves)
        takes, kinds = {}, {}
        for m in tied:
            pl = self.facts.replay(text, [m])
            takes[m] = bool(pl) and pl[0]["took"] is not None
            kinds[m] = (pl[0]["kind"] or "") if pl else ""
        return takes, kinds

    def pick(self, moves: list, tied: list, mode: str) -> "tuple[str, dict]":
        assert mode in ("reasons", "quiet", "withctl"), mode
        if len(tied) < 2:
            return tied[0], {"scores": {}, "carried": {}, "takes": None, "kinds": None}
        table = SY.deltas_of({"moves": list(moves)}, tied, self.counter)
        carried = {m: [e for e in self.order if e in table.get(m, ())] for m in tied}
        score = {m: sum(self.weight[e] for e in carried[m]) for m in tied}
        takes = kinds = None
        if mode in ("quiet", "withctl"):
            takes, kinds = self.controls_of(moves, tied)
        if mode == "withctl":
            for m in tied:
                score[m] += self.ctl_weights.get("takes", 0.0) * float(takes[m]) + self.ctl_weights.get(f"moves:{kinds[m] or 'none'}", 0.0)
        detail = {"scores": score, "carried": {m: [[self.index[e], self.weight[e]] for e in carried[m]] for m in tied}, "takes": takes, "kinds": kinds}
        if mode == "quiet" and any(takes.values()):
            return tied[0], detail
        s = {m: round(v, TIE_DIGITS) for m, v in score.items()}
        top = max(s.values())
        best = [m for m in tied if s[m] == top]
        return (tied[0] if tied[0] in best else best[0]), detail


def reader82() -> ReasonsReader:
    global _READER
    if _READER is None:
        _READER = ReasonsReader()
    return _READER


# ------------------------------------------------------------------------------------------------------------------------------ the players

class MoverReasons(TM.Mover67):
    """See the module doc."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.detail: dict = {}
        self.window = tuple(spec["window"]) if spec.get("window") else WINDOW.get(self.kind)
        self.stats["ties_in"] = 0
        self.stats["dlog"] = []
        self.stats["cases"] = []

    def search_clock(self) -> float:
        if self.charge and self.kind in MODE:
            return TM.reading_clock(self.seconds, self.costs)
        return super().search_clock()

    def pick(self, position: dict, tied: list) -> str:
        move, self.detail = reader82().pick(position["moves"], tied, MODE[self.kind])
        assert move in tied
        return move

    def in_window(self, moves: list) -> bool:
        if self.window is None:
            return True
        low, high = self.window
        return low <= reader82().units_of(moves) <= high

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        if self.kind not in MODE:
            return super().choose(text, moves, legal)
        t0 = time.perf_counter()
        row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, text, list(moves))], self.search_clock(), worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1,
                                                 ledger_share=TM.LEDGER_SHARE)[0])
        t1 = time.perf_counter()
        mv, eq = row["move"], row["equals"]
        pick, st = mv, self.stats
        st["decisions"] += 1
        if mv is not None and len(eq) >= 2:
            st["ties"] += 1
        if mv is not None and len(eq) >= 2 and self.in_window(moves):
            st["ties_in"] += 1
            tied = [mv] + sorted(m for m in eq if m != mv)
            pick = self.pick({"moves": list(moves)}, tied)
            cost = time.perf_counter() - t1
            self.costs.append(cost)
            st["changed"] += pick != mv
            st["dlog"].append([len(moves), len(tied), int(pick != mv), round(1000 * cost, 3)])
            if pick != mv:
                d = self.detail
                st["cases"].append({"ply": len(moves), "moves": list(moves), "tied": tied, "first": mv, "pick": pick, "carried_pick": d["carried"][pick], "carried_first": d["carried"][mv],
                                    "scores": d["scores"], "takes": d["takes"], "kinds": d["kinds"]})
        st["log"].append((len(moves), len(eq), int(pick != mv)))
        t2 = time.perf_counter()
        st["search"] += t1 - t0
        st["read"] += t2 - t1
        st["wall"] += t2 - t0
        return pick, row["depth"]


def play82(spec: dict) -> dict:
    OM.Mover = MoverReasons
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g


# ------------------------------------------------------------------------------------------------------------------------------ the matches and what they printed

def games_file(tag: str, a: str, b: str, set_name: str, charge: bool = False) -> Path:
    return out_root(tag) / "matches" / f"{a}_v_{b}_{set_name}{'_charged' if charge else ''}.jsonl"


def _plain(games: list) -> list:
    return [dict(g, stats=[{k: v for k, v in s.items() if k not in ("dlog", "cases")} for s in g["stats"]]) for g in games]


def lines_of(a: str, b: str, set_name: str, games: list, load: float, busy: float, wall: float, seconds: float, workers: int, charge: bool) -> list:
    """78.67's match lines (W D L, the score and its 90 pct band, the ties, the cost), the player logs taken off the stats."""
    out = TM.match_lines(a, b, set_name, _plain(games), load, busy, wall, seconds, workers)
    return [x.replace("OBS tiebreak match", "OBS symbols_play match" + (" CHARGED" if charge else " uncharged"), 1) for x in out]


def pooled_lines(a: str, b: str, sets: dict, charge: bool) -> list:
    """The match pooled over the opening sets found: W D L, the score with its band by bootstrap over opening pairs (the pairs of both sets)."""
    games = [g for s in sorted(sets) for g in sets[s]]
    sc = np.array([OM.white_score(g) if g["colour"] == 0 else 1 - OM.white_score(g) for g in games])
    s, lo, hi = OM.score_band(OM.pair_scores(games))
    return [f"OBS symbols_play pooled{' CHARGED' if charge else ''} {a} v {b} [sets {''.join(sorted(sets))}]: games {len(games)}: {a} W {int((sc == 1).sum())} D {int((sc == 0.5).sum())} L {int((sc == 0).sum())}; "
            f"score {s:.4f} [{lo:.4f}, {hi:.4f}] 90 pct band by bootstrap over opening pairs"]


def units_after(layer1, moves: list, ply: int) -> int:
    return LM.unit_counts(layer1, moves[:ply])[ply]


def decision_lines(a: str, b: str, set_name: str, games: list, layer1) -> list:
    """From the decision logs of the side that plays `a` (a reasons player): the share of decisions with two or more equal moves, the share of those where the pick differed from the search's first move, the mean cost of the
    pick a decision with a tie, and the same by phase (units on the board, LM.BANDS)."""
    rows = []                                                      # (units band, differed, cost ms)
    n_dec = 0
    for g in games:
        seat = TM.seat_of(g, 0)
        st = g["stats"][seat]
        n_dec += st["decisions"]
        counts = LM.unit_counts(layer1, g["moves"]) if st["dlog"] else []
        for ply, _n, ch, ms in st["dlog"]:
            rows.append((LM.band_of(counts[ply]), ch, ms))
    tag = f"{a} v {b} [set {set_name}]"
    out = []

    def line(label: str, sel: list, dec: "int | None") -> str:
        n = len(sel)
        if not n:
            return f"OBS symbols_play decisions {tag}: {label}: no decision with two or more equal moves"
        ch = sum(r[1] for r in sel)
        share = f"{100 * n / dec:.1f} pct of {dec} decisions have two or more equal moves; " if dec else f"{n} decisions with two or more equal moves; "
        return f"OBS symbols_play decisions {tag}: {label}: {share}the pick differed from the search's first move in {ch} ({100 * ch / n:.1f} pct of the ties); cost of the pick a tie decision: mean {np.mean([r[2] for r in sel]):.1f} ms, median {np.median([r[2] for r in sel]):.1f}"
    out.append(line("all phases", rows, n_dec))
    for band, lo, hi in LM.BANDS:
        out.append(line(f"units {lo}-{hi} ({band})", [r for r in rows if r[0] == band], None))
    return out


CASES_PER_BAND = 10


def cases_by_band(games: list, a: str, layer1) -> list:
    """PLAN 78.83: THE THIRTY by the rule fixed before any game was read: in the games of `a` v control on set B (the full reasons player of 78.82), the games in the order of their ids; in each game and each band of units on the
    board (LM.BANDS, the count at the decision's position) the first decision of side `a` where the pick differed from the search's first move; per band the first ten games that have one. -> the cases, band by band, each with
    gid, band, units."""
    out = {b: [] for b, _lo, _hi in LM.BANDS}
    for g in sorted(games, key=lambda x: x["gid"]):
        cs = g["stats"][TM.seat_of(g, 0)]["cases"]
        if not cs:
            continue
        counts = LM.unit_counts(layer1, g["moves"])
        seen = set()
        for c in cs:
            u = counts[c["ply"]]
            b = LM.band_of(u)
            if b not in seen:
                seen.add(b)
                if len(out[b]) < CASES_PER_BAND:
                    out[b].append(dict(c, gid=g["gid"], band=b, units=u))
    return [c for b, _lo, _hi in LM.BANDS for c in out[b]]


def cases_of(games: list, a: str) -> list:
    """THE THIRTY by the rule in the module doc: the games in the order of their ids, the first decision of side `a` where the pick differed from the search's first move, the first thirty games that have one."""
    out = []
    for g in sorted(games, key=lambda x: x["gid"]):
        st = g["stats"][TM.seat_of(g, 0)]
        if st["cases"]:
            out.append(dict(st["cases"][0], gid=g["gid"]))
        if len(out) >= CASES_N:
            break
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the agreement with 78.81's pick

def agreement(n: int, say) -> dict:
    """On `n` positions of the 78.80 CONFIRM sample (evenly spaced in the sample's order), the player's pick (ReasonsReader.pick, the tie set of the sample) against the pick that symbols_reasons_measure computes for the same tie
    set (SY.score_choice over the choice rows of the CONFIRM part, the weights of the reasons, the controls of the sample's controls file). The tie sets are the same by construction (the sample's); `controls` agreement: the
    player's replay reading against controls.npz."""
    from catspace.incubator.analyst.discovery import symbols_reasons_measure as SR
    rd = reader82()
    inp = SR.load_inputs()
    ctl = SR.load_controls(src_root())
    cols = [r["event"] for r in rd.reasons]
    rows = SR.choice_rows(inp["sample"], inp["M"], cols, ctl, "confirm")
    fit_ev = {"events": list(range(len(cols))), "weights": [r["weight"] for r in rd.reasons], "controls": {}}
    fit_ct = dict(fit_ev, controls=rd.ctl_weights)
    sample_c = [r for r in inp["sample"] if r["part"] == "confirm"]
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    pick_ev, pick_ct = SY.score_choice(rows, fit_ev, False), SY.score_choice(rows, fit_ct, True)
    takes_row = rows.C[:, 0]
    chosen = np.unique(np.linspace(0, len(sample_c) - 1, n).astype(int))
    agree = {"reasons": 0, "quiet": 0, "withctl": 0}
    ctl_ok = ctl_all = 0
    differ = {"reasons": 0, "quiet": 0, "withctl": 0}
    bad = []
    for p in chosen:
        r = sample_c[p]
        tied = r["tied"]
        ref = {"reasons": tied[pick_ev[p]], "withctl": tied[pick_ct[p]]}
        some = bool(takes_row[start[p]:start[p] + sizes[p]].any())
        ref["quiet"] = tied[0] if some else ref["reasons"]
        t0 = time.perf_counter()
        for mode in agree:
            got, det = rd.pick(r["moves"], tied, mode)
            ok = got == ref[mode]
            agree[mode] += ok
            differ[mode] += got != tied[0]
            if not ok:
                bad.append((r["id"], mode, got, ref[mode]))
            if mode == "withctl":
                for j, m in enumerate(tied):
                    ctl_all += 1
                    ctl_ok += bool(det["takes"][m]) == bool(takes_row[start[p] + j])
    say(f"AGREEMENT the player's pick against symbols_reasons_measure's pick for the same tie set, {len(chosen)} positions of the 78.80 CONFIRM sample (evenly spaced), tie sets equal by construction: "
        + "; ".join(f"{m} {agree[m]} of {len(chosen)} ({100 * agree[m] / len(chosen):.1f} pct), the pick differs from the first move in {differ[m]}" for m in agree)
        + f"; the player's 'takes a unit' equals controls.npz on {ctl_ok} of {ctl_all} tied moves")
    for b in bad[:10]:
        say(f"AGREEMENT disagreement {b}")
    return {"n": len(chosen), "agree": agree, "bad": bad}


# ------------------------------------------------------------------------------------------------------------------------------ the run

def emit(lines: list, out_file: "Path | None") -> None:
    TM.emit(lines, out_file)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("agreement", "match", "report", "cases", "cases83"))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--a", default="reasons", choices=PLAYERS)
    ap.add_argument("--b", default="control", choices=PLAYERS)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--n", type=int, default=AGREE_N)
    ap.add_argument("--charge", action="store_true", help="take the pick's reading time out of the clock of the search")
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"

    def say(line: str) -> None:
        emit([f"OBS symbols_play {line}"], obs)
    if a.stage == "agreement":
        agreement(a.n, say)
    elif a.stage == "match":
        load, busy = TM.wait_quiet(lambda s: emit([s], obs))
        t0 = time.perf_counter()
        gf = games_file(a.tag, a.a, a.b, a.set, a.charge)
        games = TM.run_games(TM.match_specs(a.a, a.b, a.set, a.pairs, a.seconds, root / "scratch", a.charge, KIND), a.workers, gf, lambda s: emit([s], obs), play=play82)
        wall = time.perf_counter() - t0
        gf.with_suffix(".meta.json").write_text(json.dumps({"load": load, "busy": busy, "wall": wall, "workers": a.workers, "seconds": a.seconds}))
        emit(lines_of(a.a, a.b, a.set, games, load, busy, wall, a.seconds, a.workers, a.charge), obs)
    elif a.stage == "report":
        layer1 = GL.model(HD.GAME, HD.VARIANT)
        found: dict = {}
        for p in sorted((root / "matches").glob("*.jsonl")):
            stem = p.stem
            charge = stem.endswith("_charged")
            x, rest = stem.removesuffix("_charged").split("_v_")
            y, s = rest.rsplit("_", 1)
            found.setdefault((x, y, charge), {})[s] = TM.read_games(p)
        for (x, y, charge), sets in sorted(found.items()):
            for s, games in sorted(sets.items()):
                mf = games_file(a.tag, x, y, s, charge).with_suffix(".meta.json")
                meta = json.loads(mf.read_text()) if mf.exists() else {}
                emit(lines_of(x, y, s, games, meta.get("load", float("nan")), meta.get("busy", float("nan")), meta.get("wall", float("nan")), a.seconds, a.workers, charge), None)
                if x in MODE_PLAYERS:
                    emit(decision_lines(x, y, s, games, layer1), None)
            if len(sets) > 1:
                emit(pooled_lines(x, y, sets, charge), None)
    elif a.stage == "cases83":
        src = Path(paths.experiment(*OUT82, "matches", "reasons_v_control_B.jsonl"))
        cs = cases_by_band(TM.read_games(src), "reasons", GL.model(HD.GAME, HD.VARIANT))
        (root / "cases83.json").write_text(json.dumps(cs))
        say(f"CASES83 {len(cs)} decisions saved to {root / 'cases83.json'}: " + ", ".join(f"{b} {sum(c['band'] == b for c in cs)}" for b, _lo, _hi in LM.BANDS) + f" (read from {src})")
    else:
        games = TM.read_games(games_file(a.tag, "reasons", "control", "A"))
        cs = cases_of(games, "reasons")
        (root / "cases82.json").write_text(json.dumps(cs))
        say(f"CASES {len(cs)} decisions saved to {root / 'cases82.json'} (the rule in the module doc)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
