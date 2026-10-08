"""THE TACTIC MAP (Kaveh 2026-10-07): "each tactic has a likelihood of an out conditioned on Elo and a gain conditioned on
Elo, so we can calculate the expectation value of each tactic conditioned on Elo ... an expectation value of gain overall
given some sort of reachability ... navigate to the place where we have the highest likelihood of winning."

TABLES, fitted on a tactic census (tactic_census.py) by family and rating band of the actor:
  hold[f, band]     the holder's mean material swing over 4 plies when family f is available (k = 0) to it -- execution,
                    the opponent's outs and the gain in one empirical number; a game ending in mate inside the window
                    counts MATE_VALUE pawns against the mated side (the census has no row after the mate);
  exposed[f, band]  the victim's mean swing when the opponent's f is available against it (negative);
  step[f, band]     P(the family goes from k = 1 to k = 0 at the holder's next move) -- the ladder's one-move chance.
THE VALUE of a position for the side to move, given both ratings:
  V = sum over f of  hold[f, us] * [our f at k=0] + step[f, us] * hold[f, us] * [our f at k=1]
      + exposed[f, us] * [their f at k=0] + step[f, them] * exposed[f, us] * [their f at k=1]
  -- our tactics by what they are worth to a player of our strength, theirs by what they cost one of ours.
THE NAVIGATOR: a conjecture-tier rule (frame_player.RULE_BOOK "tactic_map"): among the moves the one-ply ledger ties, the
token state AFTER each move is computed exactly (tactic_census.tokens on the resulting position: their tactics; the
null move: ours) and the move with the highest V is preferred. Reachability one move out is exact; beyond, the step
table. Subgoals enter through the prior, never the value (AGENTS rule 7).

    python -m catspace.incubator.analyst.tactic_map fit --tag wide
Tables go to data/derived/tactic_census/map_<tag>.json."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import chess

from catspace.io import paths

FAMILIES = ("material_now", "double_threat", "pin", "discovery", "back_rank", "passer")
DEFAULT_BANDS = ((1000, 1400), (1400, 1700), (1700, 2000), (2000, 2500))
MIN_N = 20
MATE_VALUE = 20.0          # a game ending in mate inside the window, in pawns, from the mated side's view (a choice: more
                           # than a queen, so a mating family outweighs a material one at the same chance)


def map_path(tag: str) -> Path:
    return Path(paths.derived("tactic_census", f"map_{tag}.json"))


def band_name(elo: int, bands) -> str:
    for lo, hi in bands:
        if lo <= elo < hi:
            return f"{lo}-{hi}"
    return f"{bands[0][0]}-{bands[0][1]}" if elo < bands[0][0] else f"{bands[-1][0]}-{bands[-1][1]}"


def fit(tag: str, bands=DEFAULT_BANDS, window: int = 4) -> dict:
    """The tables from a census; cells with fewer than MIN_N observations fall back to the family's all-band value."""
    from .tactic_census_measure import load
    rows = load(tag)
    by = defaultdict(dict)
    for r in rows:
        by[r["game"]][r["ply"]] = r
    hold = defaultdict(lambda: [0, 0.0]); exposed = defaultdict(lambda: [0, 0.0]); step = defaultdict(lambda: [0, 0])
    for g, plies in by.items():
        last = max(plies)
        lb = chess.Board(plies[last]["fen"])
        mated_side = None                                                  # the game ended in mate: the last row's mover won
        if any((lb.push(m), lb.is_checkmate(), lb.pop())[1] for m in lb.legal_moves):
            mated_side = "b" if plies[last]["side"] == "w" else "w"
        elif plies[last]["result"] != 0:                                   # a resignation: the loser's last window ends in the loss
            mated_side = plies[last]["side"] if plies[last]["result"] < 0 else ("b" if plies[last]["side"] == "w" else "w")
        for t, r in plies.items():
            b = band_name(r["elo"], bands)
            later = plies.get(t + window)
            swing = None
            if later is not None:
                swing = (later["material"] if later["side"] == r["side"] else -later["material"]) - r["material"]
            elif mated_side is not None and t + window > last:             # the window runs into the mate: a loss of the game
                swing = -MATE_VALUE if r["side"] == mated_side else MATE_VALUE
            if swing is not None:
                for f in FAMILIES:
                    if any(x[0] == f and x[1] == 0 for x in r["stm"]):
                        c = hold[(f, b)]; c[0] += 1; c[1] += swing
                        c = hold[(f, "all")]; c[0] += 1; c[1] += swing
                    if any(x[0] == f and x[1] == 0 for x in (r["opp"] or [])):
                        c = exposed[(f, b)]; c[0] += 1; c[1] += swing
                        c = exposed[(f, "all")]; c[0] += 1; c[1] += swing
            nxt = plies.get(t + 2)                                         # the holder's next move: k = 1 -> k = 0?
            if nxt is not None:
                for f in FAMILIES:
                    if any(x[0] == f and x[1] == 1 for x in r["stm"]):
                        fired = any(x[0] == f and x[1] == 0 for x in nxt["stm"])
                        c = step[(f, b)]; c[0] += 1; c[1] += fired
                        c = step[(f, "all")]; c[0] += 1; c[1] += fired
    names = [f"{lo}-{hi}" for lo, hi in bands]
    tables = dict(tag=tag, bands=[list(b) for b in bands], window=window, hold={}, exposed={}, step={}, n={})
    for f in FAMILIES:
        for b in names + ["all"]:
            for name, acc in (("hold", hold), ("exposed", exposed), ("step", step)):
                n, s = acc[(f, b)]
                src = (f, b) if n >= MIN_N else (f, "all")
                n2, s2 = acc[src]
                tables[name][f"{f}|{b}"] = (s2 / n2) if n2 else 0.0
                tables["n"][f"{name}|{f}|{b}"] = n
    out = map_path(tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(tables, open(out, "w"), indent=1)
    for f in FAMILIES:
        print(f"OBS map {tag} {f:13s} " + "  ".join(f"{b}: hold {tables['hold'][f'{f}|{b}']:+.2f} exposed {tables['exposed'][f'{f}|{b}']:+.2f} step {100 * tables['step'][f'{f}|{b}']:.0f}% (n {tables['n'][f'hold|{f}|{b}']}/{tables['n'][f'exposed|{f}|{b}']}/{tables['n'][f'step|{f}|{b}']})" for b in names))
    return tables


class TacticMap:
    """The value of a position from its tactic tokens, by the tables of a census."""

    def __init__(self, tag: str = "wide", elo_us: int = 1800, elo_them: int = 1800):
        p = map_path(tag)
        if not p.exists():
            p = map_path("c300")
        self.t = json.load(open(p))
        self.bands = [tuple(b) for b in self.t["bands"]]
        self.elo_us, self.elo_them = elo_us, elo_them

    def value_for_side_to_move(self, stm, opp, band_stm: str, band_opp: str) -> float:
        """V for the side to move, from its own tables (both tables are fitted on rows where the actor is to move): its
        tactics at hold, the opponent's at exposed; k = 1 tokens by the ladder's one-move chance of the side that would
        fire them."""
        v = 0.0
        for f in FAMILIES:
            ks = min((x[1] for x in (stm or []) if x[0] == f), default=None)
            ko = min((x[1] for x in (opp or []) if x[0] == f), default=None)
            if ks == 0:
                v += self.t["hold"][f"{f}|{band_stm}"]
            elif ks == 1:
                v += self.t["step"][f"{f}|{band_stm}"] * self.t["hold"][f"{f}|{band_stm}"]
            if ko == 0:
                v += self.t["exposed"][f"{f}|{band_stm}"]
            elif ko == 1:
                v += self.t["step"][f"{f}|{band_opp}"] * self.t["exposed"][f"{f}|{band_stm}"]
        return v

    def value_of_tokens(self, ours, theirs) -> float:
        """V for us, to move, with our tactics `ours` and theirs `theirs` (the null-move reading)."""
        return self.value_for_side_to_move(ours, theirs, band_name(self.elo_us, self.bands), band_name(self.elo_them, self.bands))

    def value(self, b: chess.Board) -> float:
        """V for the side to move of `b` (taken as us)."""
        from .tactic_census import opponent_tokens, tokens
        return self.value_of_tokens(tokens(b), opponent_tokens(b))

    def value_after(self, b: chess.Board, m: chess.Move) -> float:
        """V for the mover, read in the position after the move, where the OPPONENT is to move: the opponent's value by
        its own tables (its tactics at hold -- it executes first; ours at exposed -- it answers first), negated (review
        2026-10-07: the first version valued our tactics at hold and theirs at exposed after our move, about 2x too
        much for ours and 3x too little for theirs)."""
        from .tactic_census import opponent_tokens, tokens
        nb = b.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            return 100.0
        if nb.is_stalemate():                                        # the game ends: nothing is gained or lost from here
            return 0.0
        theirs = [t for t in tokens(nb) if t[0] != "material_now"]   # their winning capture is the ledger's own charge (review
        ours = opponent_tokens(nb)                                    # 2026-10-07: priced twice, every even trade lost its tie)
        if ours is None:                                              # the move gives check: no null move; our tactics read before it
            ours = tokens(b)
        return -self.value_for_side_to_move(theirs, ours, band_name(self.elo_them, self.bands), band_name(self.elo_us, self.bands))


_MAP = None


def _map():
    """The map in play: the tables named by CATSPACE_TACTIC_MAP (default: wide, else c300), both ratings 1800 unless
    CATSPACE_ELO_US / CATSPACE_ELO_THEM say otherwise."""
    global _MAP
    if _MAP is None:
        import os, sys
        tag = os.environ.get("CATSPACE_TACTIC_MAP", "wide")
        if not map_path(tag).exists():
            print(f"[tactic_map] no tables for {tag!r}; falling back to c300", file=sys.stderr)
        _MAP = TacticMap(tag, int(os.environ.get("CATSPACE_ELO_US", 1800)), int(os.environ.get("CATSPACE_ELO_THEM", 1800)))
    return _MAP


def passer_credit() -> float:
    """The census's price of a proved passer for a player of our rating: the `hold` of the passer family (the material
    swing over the window when a proved passer is available at one's turn; +4.4 pawns in both censuses)."""
    try:
        tm = _map()
    except FileNotFoundError:                                        # no census tables at all: the passer's long-run price
        import sys
        print("[tactic_map] no census tables: a proved passer credited +4.0", file=sys.stderr)
        return 4.0
    return float(tm.t["hold"][f"passer|{band_name(tm.elo_us, tm.bands)}"])


def map_fire(b: chess.Board, tied: list | None = None) -> dict:
    """The rule: {uci: (score, why)} over the moves (the ledger's ties when given, else every legal move); score = V after
    the move shifted to be positive (a conjecture's claim orders only when positive)."""
    tm = _map()
    moves = [chess.Move.from_uci(u) for u in tied] if tied else list(b.legal_moves)
    vals = {m.uci(): tm.value_after(b, m) for m in moves}
    lo = min(vals.values(), default=0.0)
    return {u: (v - lo + 1.0, f"tactic map: expected gain {v:+.2f} pawns after the move (ours held minus theirs exposed, by rating)") for u, v in vals.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--tag", default="wide")
    a = ap.parse_args(argv)
    fit(a.tag)


if __name__ == "__main__":
    main()
