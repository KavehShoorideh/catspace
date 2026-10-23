"""plan 71.3 THE WORDS INTERFACE (layer 3 of the three): what a domain's VOCABULARY supplies to the machinery, in two parts --

  BaseWords  -- the primitive predicates and measures over states, supplied per domain, hand-written, few, each Lean-defined (chess: graph.py's words --
                attacks, guards, pushes, distances, worth -- the piece values and the terminals);
  Words      -- the PRODUCED vocabulary built from the base words and the axioms (Game): the root goals (questions), the defects of a goal and their fixes,
                the forcing words of a move, threats and parries, the rule closes at a leaf, the binder's lemma families, the endgame words. For chess these are
                handwritten today (regress.py, race.py, plan67.py, binder.py, table.py) and stand as the SEED and the BENCHMARK the mined vocabulary must reproduce.

The agenda (layer 1) imports `Words`; it never imports regress / race / plan67 / binder / table directly once 71.3 is complete. `ChessWords` delegates, one line
each, to the existing modules: nothing is rewritten, the interface names what exists (the inventory of 2026-10-22: R.forcing_why 8 calls, R.promote_defects 4,
R.mate_defects 4, R.material_defects 2, R.promote_fixes 2, R.fixes_of 2, R.exchange_read 2, R.drives_off 2, R.pawn_path 3, R.threat_of 2, R.tempo_ok 1,
R.slide_region 1, R.attack_lines 1, R.opponent_defects 1, RC.race_claim 2, PV.proved_passers 2, PV.passer_v / HYPS, TB.table_claim / covered, PL.read /
support_fixes / trace, BD.verify_tried / verify_keep / schema / generate). The machinery's own types (Question, Route, Defect, Claim, the ids) stay in the
machinery; `regress.py` is split later (71.3 b) into the generic question algebra and the chess words."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import plan67 as PL
from catspace.incubator.analyst.discovery import race as RC
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import table as TB
from catspace.incubator.analyst import lean_passer_v2_proto as PV  # the proved passer lemma family (Ideas.Passer2)


@runtime_checkable
class BaseWords(Protocol):
    """The primitive words a domain supplies (chess: graph.py). Each has a Lean definition; none is learned."""

    def word(self, s, key, side) -> int: ...                       # a named measure of the state for a side ("pushes", "dist", ...)
    def worth(self, unit) -> int: ...
    def name(self, loc) -> str: ...
    def see(self, s, loc): ...                                     # the exchange count on a location (the side to move captures)
    def see_for(self, s, loc, capturer): ...                        # ... for a named capturing side


class ChessBaseWords:
    def word(self, s, key, side):
        return G.word(s, key, side)

    def worth(self, unit):
        return G.worth(unit)

    def name(self, loc):
        return G.name(loc)

    def see(self, s, loc):
        return G.see(s, loc)

    def see_for(self, s, loc, capturer):
        return G.see_for(s, loc, capturer)


@runtime_checkable
class Words(Protocol):
    """The produced vocabulary of a domain: what the agenda asks of the words. Every method returns the machinery's own types (Question, Route, Defect, Claim)."""

    # the forcing words of a move
    def forcing_why(self, s, m, stake, us, **kw) -> str | None: ...
    def threat_of(self, s, m, why, us=None): ...
    def tempo_ok(self, *a, **kw): ...
    # goals, defects, fixes
    def mate_defects(self, s, us, placement) -> list: ...
    def material_defects(self, s, us, target) -> list: ...
    def promote_defects(self, s, sq, lift=False) -> tuple: ...
    def opponent_defects(self, *a, **kw): ...
    def fixes_of(self, defect, q, s, us) -> list: ...
    def promote_fixes(self, s, sq, defect, q): ...
    def drives_off(self, *a, **kw): ...
    def slide_region(self, *a, **kw): ...
    def attack_lines(self, *a, **kw): ...
    def pawn_path(self, s, sq, n) -> list: ...
    # the rule closes at a leaf
    def exchange_read(self, s) -> str: ...
    def table_claim(self, s): ...
    def table_covered(self, s) -> bool: ...
    # the endgame words (62, 67)
    def race_claim(self, s, sq, **kw): ...
    def proved_passers(self, s, **kw): ...
    def passer_hyps(self, s, sq): ...
    def plan_read(self, s, sq): ...
    def plan_support(self, s, sq): ...
    def plan_trace(self, s, plan) -> str: ...
    # the binder's lemma families
    def verify_tried(self, *a, **kw): ...
    def verify_keep(self, *a, **kw): ...
    def binder_schema(self, *a, **kw): ...
    def binder_generate(self, *a, **kw): ...
    # the pricing words (the cost of a route / node, 61 C1)
    def route_cost(self, *a, **kw): ...
    def node_cost(self, *a, **kw): ...
    # the skeleton goals (72.8): arrangements of the fast units against the fixed board, candidates only
    def skeleton_report(self, s, us, **kw) -> dict: ...
    def skeleton_goals(self, s, us, **kw) -> list: ...
    def arrangement_line(self, a, goal_text=None) -> str: ...
    def arrangement_items(self, a) -> str: ...
    def arrangement_steps(self, s, a) -> list: ...
    def arrangement_fix(self, s, us, a, q, text=None): ...


class ChessWords:
    """Chess's vocabulary as it stands: handwritten, each method a one-line delegation (the seed and the benchmark for the mined vocabulary)."""

    base = ChessBaseWords()

    def forcing_why(self, s, m, stake, us, **kw):
        return R.forcing_why(s, m, stake, us, **kw)

    def threat_of(self, s, m, why, us=None):
        return R.threat_of(s, m, why, us)

    def tempo_ok(self, *a, **kw):
        return R.tempo_ok(*a, **kw)

    def mate_defects(self, s, us, placement):
        return R.mate_defects(s, us, placement)

    def material_defects(self, s, us, target):
        return R.material_defects(s, us, target)

    def promote_defects(self, s, sq, lift=False):
        return R.promote_defects(s, sq, lift=lift)

    def opponent_defects(self, *a, **kw):
        return R.opponent_defects(*a, **kw)

    def fixes_of(self, defect, q, s, us):
        return R.fixes_of(defect, q, s, us)

    def promote_fixes(self, s, sq, defect, q):
        return R.promote_fixes(s, sq, defect, q)

    def drives_off(self, *a, **kw):
        return R.drives_off(*a, **kw)

    def slide_region(self, *a, **kw):
        return R.slide_region(*a, **kw)

    def attack_lines(self, *a, **kw):
        return R.attack_lines(*a, **kw)

    def pawn_path(self, s, sq, n):
        return R.pawn_path(s, sq, n)

    def exchange_read(self, s):
        return R.exchange_read(s)

    def table_claim(self, s):
        return TB.table_claim(s)

    def table_covered(self, s):
        return TB.covered(s)

    def race_claim(self, s, sq, **kw):
        return RC.race_claim(s, sq, **kw)

    def proved_passers(self, s, **kw):
        return PV.proved_passers(s, **kw)

    def passer_hyps(self, s, sq):
        return PV.passer_v2(s, sq)

    def plan_read(self, s, sq):
        return PL.read(s, sq)

    def plan_support(self, s, sq):
        return PL.support_fixes(s, sq)

    def plan_trace(self, s, plan):
        return PL.trace(s, plan)

    def verify_tried(self, *a, **kw):
        return BD.verify_tried(*a, **kw)

    def verify_keep(self, *a, **kw):
        return BD.verify_keep(*a, **kw)

    def binder_schema(self, *a, **kw):
        return BD.schema(*a, **kw)

    def binder_generate(self, *a, **kw):
        return BD.generate(*a, **kw)

    def route_cost(self, *a, **kw):
        return R.route_cost(*a, **kw)

    def node_cost(self, *a, **kw):
        return R.node_cost(*a, **kw)

    def skeleton_report(self, s, us, **kw):
        return R.skeleton_report(s, us, **kw)

    def skeleton_goals(self, s, us, **kw):
        return R.skeleton_goals(s, us, **kw)

    def arrangement_line(self, a, goal_text=None):
        return R.arrangement_line(a, goal_text)

    def arrangement_items(self, a):
        return R.arrangement_items(a)

    def arrangement_steps(self, s, a):
        return R.arrangement_steps(s, a)

    def arrangement_fix(self, s, us, a, q, text=None):
        return R.arrangement_fix(s, us, a, q, text)


CHESS_WORDS = ChessWords()
assert isinstance(CHESS_WORDS, Words) and isinstance(CHESS_WORDS.base, BaseWords)
