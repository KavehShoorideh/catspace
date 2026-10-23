"""THE AGENDA (ANALYST_PLAN 52.1, 52.2): inference is GOAL REGRESSION on QUESTIONS, never a search of moves. Root questions are the goals in order of STAKE: mate
their king (100), win their queen 9 / rook 5 / minor 3 / pawn 1, save ours (the biggest of ours that hangs; in check the king is the stake). Each is asked as a
counterfactual (regress.py), its defects generate sub-questions, tempo arithmetic prunes (and says why), a ROUTE (a forcing move supplying a fix) is VERIFIED
by the generic binder over every reply (binder.py) and, if it binds, becomes a CLAIM on the state with the proved lemma behind it.

STOP (principled, no cap): (a) the best claim's value is at least the stake of every open question (a mate ends it); (b) every question is closed; (c) the
budget (questions asked <= our pieces x 8 squares around their king + our captures) is hit: the open questions become GAPS, never a guess.
Among equal values the OPPONENT'S VIEW breaks the tie (a safe check of theirs that forks something of ours is a defect of that future). Memory is not used here.

THE REGRESSION FROM THEIR SIDE (52.4). The same Agenda, run for the opponent (side_depth 0: it never runs ours on its routes), in two places:
(a) THREATS: their agenda on the current board as if we passed (turn flipped, en passant cleared; not when we are in check): each claim they get is a THREAT and
a `save` root of ours with the claim's value as stake; (b) DEFECTS OF OUR FUTURES: their agenda on the board after each of our routes (their turn, no pass): their
best claim is subtracted from the route's bound value, and a route whose future gives them a claim at least our stake (and nets nothing) is killed with the reason.
A claim of theirs is only what a proved schema binds from THEIR side. `opponent_defects` (regress.py) is kept only as a fallback label / tie-break when no
claim of theirs binds.

THE CHASE (52.12). The fix often exists only AFTER THEIR FORCED REPLY, so a question that is killed (no forcing fix, or no route binds) is asked once more as a CHASE:
our forcing moves on the board are the routes; for every reply of theirs the WHOLE agenda is asked again on the position (an inner Agenda, same side, memoised by
FEN across the run); the route's value is the least over the replies of (balance after the route and the reply less the balance now) + the inner best claim; a
reply with no inner claim and no material gain kills the route at once; a route claims at value >= 1 (inner mates: 100). The budget (the root's question budget)
is shared with the inner agendas; when it is hit the open chase becomes a GAP, never a kill. The guarantee is step_after (our named move) + step_least (every reply)
over the inner claims, each a bound schema (the composed statement is a TODO in Schemas.lean). Their agendas never chase. A gain that stands only as a static
balance with us to move is a result only once THEIR agenda has read it with the move passed to them (52.13 (3): a claim of theirs of t makes the reply worth the gain less t;
0AgL8's Nxg2 died on this; read on the ROOT agenda's chase only -- at the inner levels "if we pass" refuted 0VSUT's true Qxe5 line, where we have a move to parry). The reach (52.12 (4)): a mate
root at distance two (one prep move) whose fixes are the first steps; a step is forcing when it checks or leaves a mate in one (a mate threat).
"""
from __future__ import annotations

import dataclasses
import heapq
import re
import time

import analyst_core as AC
import chess

from catspace.incubator.analyst import lean_passer_v2_proto as PV
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import facts as FS
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import landing as LD
from catspace.incubator.analyst.discovery import region as RG
from catspace.incubator.analyst.discovery import race as RC
from catspace.incubator.analyst.discovery import plan67 as PL
from catspace.incubator.analyst.discovery import table as TB
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import question_prior as QP
from catspace.incubator.analyst.discovery import reasoned as RS
from catspace.incubator.analyst.discovery import selfmine as SM
from catspace.incubator.analyst.discovery.state import Claim, State


DEPTH_FUSE = 30                                                 # 52.13 (2b): ... and a chase this many inner levels deep is cut the same way (python's stack: ~9 frames per level; the 10x fuse alone reached the recursion limit)
RULE2_FROM = {"mate": 2, "win": 1, "promote": 1}              # 52.17 (2): the chase depth from which a check must address a defect of its goal (the root's chase, depth 0, keeps every forcing move; a mate goal's checks are the first level of an inner agenda, its walk begins one level below)
FUSE = 10                                                       # 52.13 (2b): the chase's budget is a fuse at this multiple of the old question count
NEED_AWARE_CLOSE = True                                         # 52.23 (2): ON at every tier (plan 64.2, 2026-10-21: "off is not built"; it was OFF because 07Z0P's Bb3 no longer finished in its share)
EVASION_CHASE = True                                            # 52.23 (3): ON at every tier (plan 64.2)
NEED_PROPAGATION = True                                         # 52.23 (7): ON at every tier (plan 64.2) -- the need propagates to the inner agendas, a gaining reply carries the inner claim, the chain of two meets the need, a bound below the need chases first
GOAL_JUDGE = True                                               # 52.23 (6): ON at every tier (plan 64.2; its stop is a GAP since 64.1)
CLOCK_PAWN = True                                              # plan 62.1c / 66.5: the chase clock also resets on a PAWN move of ours (an irreversible move, the fifty-move rule's other event); CHASE_CLOCK stays the parameter
EVADED_RESET = False                                            # plan 66.5 (b), separated 2026-10-22 (Kav, eSJqH trace): a CAPTURING check does not count as "a second check in a row". OFF: with it the Qa5 read followed capturing checks past Qc6 and ran out of its slice (tier 2, 6.5 s), while without it Qa5 is PROVED in tier 2 at 3.9 s; the principled end of a check sequence is repetition (CYCLE) or the clock, not an exemption -- open
CLOCK_IRREVERSIBLE = True                                       # plan 66.5 (Kaveh 2026-10-22): the chase clock resets on an irreversible event -- a CAPTURE by either side that is not bad for us, or a PAWN move of ours (and the checks evaded in a row: a capturing check is no perpetual); `analyse --no-irreversible` (then: our capture not bad for us, our pawn move)
CHASE_CLOCK = 4                                                 # 52.23 (8): ON at every tier (plan 64.2; its stop is a GAP since 64.1); 99 = off; 4 was Kaveh's starting value
                                                # 52.23 (8): our moves on a chain without a capture that is not bad for us (Kaveh 2026-10-20: "four or five ... a hyperparameter, tune it later")
DEEP_WIN_CHASE = True                                            # 52.23 (5): ON at every tier (plan 64.2)
PRIOR = False                                                   # plan 53 (2): the learned prior orders the root agenda's questions (analyse --prior); it never decides
SCHEDULER = False                                               # plan 56 (1): the chase as an explicit TASK GRAPH on a priority queue with one worker loop (scheduler.py), not the recursion below; the same decisions (scripts/compare_scheduler.py)
CLOSE_AFTER_CAPTURE = True                                      # 2026-10-20 (kcQil): a reply closes by the exchange only if their agenda after our capture finds nothing either
RAISE_TO_WIN = True                                             # plan 59 (4): under a time budget, a reached 'not losing' that is not 'winning' raises the line while time remains
PARRIES = True                                                  # plan 61 A2 (ON since plan 64.4): the chase names the threat of a route (regress.threat_of) and labels a closed reply that does NOT parry it "closed by the threat" (regress.parries; Lean Ideas.Forcing); same values
DELAYED_THREAT = True                                           # plan 61 A3 (ON since plan 63.3: no RIGHT lost on the 90, with the check-that-is-a-threat follow-up): a quiet move after which EVERY reply allows a forcing move of ours is a route (tier 1+, root chase only; regress.delayed_threat)
NO_SAFE_MOVE = True                                             # plan 61 A4 (ON since plan 64.4): a quiet move after which EVERY reply loses at least the stake by rule is a route (tier 1+, root chase only; regress.no_safe_move)
ESTIMATE = True                                                 # plan 61 B2 (ON since plan 64.5: inert on the 90, proved beats estimated structurally): a GAP route whose open replies are QUIET leaves closes as an ESTIMATED claim (status = material - their pass threat; confidence from data/derived/estimate_calibration.json); the decision consults it only when no proved claim keeps the status
COST_JUDGE = True                                               # plan 61 C2, ON since plan 64.3 (2026-10-21): under a TIME budget the cost judge (measured cost x seconds per node <= the route's time left) decides what is read on and REPLACES the count fuse; the cut is a GAP
CYCLE = True                                                    # plan 61 C4 (ON: graded on the 90, no decision changed, nothing lost): a position that repeats one of the line (or of ctx.history) closes as a PROVED claim kind 'draw' (value = minus the balance) when we are behind; `analyse --no-cycle` turns it off
JUDGE_PRICE_CAP = 120                                           # plan 64.3: the pricing of a route stops at this cost while no rate is known (pricing 50-parry routes was 7.2 of 15 s on 07Z0P)
JUDGE_MIN_AGENDAS = 8                                           # plan 61 C2: the judge reads on until this many cost units have been timed (no rate, no cut)
TIERS_UNDER_TIME = True                                         # plan 59 default (2026-10-20, graded on 85 random puzzles): ON whenever the budget is TIME -- the tiers share the seconds, so the cost is bounded
                                                                # (s10: RIGHT 59 -> 61, WRONG 4 -> 4; hqGOF right again); under the COUNT fuse they stay OFF unless TIERS (fuse: +2 RIGHT for ~650 s extra on 25 widened puzzles)
TIERS = False                                                   # plan 59: STATUS-DRIVEN WIDENING at the root -- tier 0 (today's moves), then wider tiers (settings of the root's ChaseCtx) only while no claim keeps our status; analyse --tiers; the play server honours it
TIER_COUNT = 4                                                  # tiers 0..3
COMPETE = True                                                  # plan 65.1: the NEXT tier's cheapest route competes with this tier's root routes (Lean `Ideas.Forcing.judgeNext`): a NEW route (not yet open in this tier) priced above it is a GAP, read in the next tier (deferred once); `analyse --no-compete`
UNPROVED = True                                                 # plan 65.3: no claim at all -> the best UNFINISHED root route (a GAP) decides, marked unproved; `analyse --no-unproved`
DEEP_PASS = True                                                # plan 65.4: the status reads THEIR pass with delayed_balance_deep when the d1 family finds no claim and a piece of ours is attacked; `analyse --no-deep-pass`
STOP_AFFORD = True                                              # plan 65.2: the widening (and the RAISE_TO_WIN continuation) ends when no open route of any tier is priced within the time left; `analyse --no-stop-afford`
RESUME = True                                                   # plan 66.4: with time left after the tiers, the root's CHAIN-STOPPED GAP routes are resumed in price order, one more level of the chain allowance a pass, ONLY while the line is still forcing (every reply a parry: Lean `Ideas.Forcing.stillForcing`); `analyse --no-resume`
DEFER = True                                                    # plan 68 (Kaveh 2026-10-22, eSJqH): RELEVANCE TO THE STATUS orders the root questions across the tiers -- a root question whose stake is below the status's need (line - balance; a save is always relevant) is DEFERRED behind the next tier's relevant questions and asked in a last pass with the time left; Lean `Ideas.Forcing.irrelevant_keeps_nothing`; `analyse --no-defer`
CARRY = True                                                    # plan 66.1 / 66.2: a PROOF CARRIES across the tiers (Lean `Ideas.Forcing.claim_mono`): a wider tier serves the proved claims of the root's questions and the proved replies of every route read before, and re-reads only kills, GAPs and what could beat a floor; `analyse --no-carry`
NO_FUSE = 1 << 40                                               # 52.23 (1): the fuse when the budget is time: never reached (the deadline is the only limit)


def judge_next(competitor: "float | None", price: float) -> bool:
    """plan 65.1 (Lean `Ideas.Forcing.judgeNext`): a route of this tier is read while it is priced no higher than the cheapest route the next tier would read (None: nothing competes). Lean prices in Nat
    (milliseconds); the comparison is the same on any ordered price."""
    return competitor is None or price <= competitor


def _aligned(piece_type: int, sq: int, ksq: int) -> bool:
    """Whether a piece of this type on `sq` would attack `ksq` on an EMPTY board (every attack on a real board is one of these)."""
    if piece_type == chess.KNIGHT:
        return bool(chess.BB_KNIGHT_ATTACKS[sq] & chess.BB_SQUARES[ksq])
    hit = 0
    if piece_type in (chess.BISHOP, chess.QUEEN):
        hit |= chess.BB_DIAG_ATTACKS[sq][0]
    if piece_type in (chess.ROOK, chess.QUEEN):
        hit |= chess.BB_RANK_ATTACKS[sq][0] | chess.BB_FILE_ATTACKS[sq][0]
    return bool(hit & chess.BB_SQUARES[ksq])


def _pos_key(fen: str) -> str:
    """The position without its move counters (board, turn, castling, en passant): the chase's memo key. A check chain that comes back to a position only the move number
    apart is a CYCLE (0DoFB: the same board at moves 32, 34, 36 ... each re-asked, 1033 inner questions)."""
    return " ".join(fen.split()[:4])


class _TierLines(list):
    """plan 59: the trace of a wider tier's root agenda: every line is passed on to the root's trace (and its stream) with the tier named, and kept here."""

    def __init__(self, sink, prefix: str):
        super().__init__()
        self.sink, self.prefix = sink, prefix

    def append(self, x):
        super().append(x)
        self.sink.append(self.prefix + str(x))

    def extend(self, xs):
        for x in xs:
            self.append(x)

    def __iadd__(self, xs):
        self.extend(xs)
        return self


def _wider(owner: "Agenda", ag: "Agenda") -> bool:
    """plan 59: does the best claim of the wider tier `ag` beat the one held (by `owner`)? `Agenda._better`'s order: greater value / fewer opponent defects, then the claim's own tie-break key."""
    new, old = ag.best, owner.best
    if (new[0], new[1]) != (old[0], old[1]):
        return (new[0], new[1]) > (old[0], old[1])
    return ag._claim_key(new[2]) > owner._claim_key(old[2])


class ChaseCtx:
    """The state shared by the root agenda and its inner agendas (52.12): the FEN memo, the route memo, the shared question budget and the counters."""
    def __init__(self, facts: "FS.FactStore | None" = None):
        self.facts = facts if facts is not None else FS.FactStore()     # 52.19 (1): every claim and kill derived in the run, with its justifications; shared by the root agenda and every inner agenda
        self.their_facts: dict = {}               # 52.19: key of their agenda's position -> the fact id of its answer
        self.gap_facts: dict = {}                 # 52.19 (3): route key -> the id of the open gap fact (resolved when the route is followed to its end)
        self.memo: dict = {}                      # fen -> {value, claim, asked}: the inner agenda's best claim, asked once for the whole run
        self.routes: dict = {}                    # (fen, uci) -> the evaluated route (memoised: the same position is chased by many questions)
        self.active: set = set()                  # the FENs being asked (a cycle ends here)
        self.tier = 0                             # plan 59: the WIDENING TIER of this root's chase (0 = today's moves; 1 quiet threats below the root; 2 + mate-in-two threats; 3 + the 52.23 chain items): a setting of the context, read through it, never a process-wide flag
        self.left = 0                             # 52.13 (2b): the FUSE of the root agenda: 10x its question budget, set once, spent by every chase and inner agenda under it
        self.depth_cuts = 0                       # ... of the fuse hits, the ones from the depth cap (the stack), not the question count
        self.sched = None                         # plan 56: the Scheduler running this context's root agenda (None: the recursive path)
        self.stats = None                         # ... and the task counts of its last run
        self.theirs: dict = {}                    # 52.13 (3): fen (their move) -> (value, claim) of THEIR agenda there, asked once for the whole run
        self.their_ags: dict = {}                 # 52.14: ... and that agenda itself (its claims are the save roots of the inner agenda on the static leaf)
        self.static_read = 0                      # 52.14: static leaves (a reply after which the gain stands only as a balance) read from their side
        self.static_parried = 0                   # ... of which every threat of theirs had a verified parry (the gain kept) / ...
        self.static_unparried = 0                 # ... at least one threat had no parry (killed, or a gap)
        self.static_lines: list = []              # 52.14: one line per static leaf whose threats were asked as save roots (chase depth, reply, verdict); the root agenda prints them
        self.promote_stands = 0                   # 52.16 (4): promote routes ended at a reply that leaves the question's stopper defect standing with no new fix
        self.standing = 0                         # 52.13 (4): routes ended by a standing kill (no inner agenda)
        self.closed = 0                           # ... replies closed by a rule (mate in one / exchange, nothing of theirs)
        self.reply_kills: dict = {}               # reply uci -> [(position left, dbal, reads of the inner agenda, route san, the target it was made for)] of the routes it killed with no claim of ours
        self.logged = 0                           # rows written to the chase log by the root agenda
        self.tag = None                           # the id the chase log rows carry (None: no log)
        self.rows: list = []                      # the chase log rows, written once by the root agenda
        self.gap_reused = 0                       # 52.13 (2a): GAP routes seen again with no more fuse than when they were cut, not re-run
        self.agendas = 0                          # inner agendas run
        self.asked = 0                            # questions they asked
        self.budget_hits = 0
        self.chain_hits = 0                       # 64.2: chain stops (clock, tempo allowance, second check) -- counted apart from the budget cuts: they do not depend on the budget
        self.chain_gap = None                     # 64.1: the reason of the chain stop (clock, tempo allowance) that cut the route in hand, until `_cut` names it
        self.inherited = 0                        # 52.12 (3): questions of the outer agenda inherited, not asked again
        self.inherited_claims = 0                 # 52.13 (1): ... of which claims (the rest are kills)
        self.reopened = 0                         # ... and the ones that met the diff and were asked
        self.fresh = 0                            # 52.13 (1): questions NEW on the inner board (a key the outer agenda did not have, or did not settle), asked
        self.reasoned = 0                         # 52.17 (1): mate questions killed by reasoning (the wall, the measure they hold), no chase
        self.deadline = None                      # 52.17 (5): THE BUDGET IS TIME. time.monotonic() value past which no inner agenda is opened (the current slice of the final deadline)
        self.final = None                         # ... the final deadline of the root agenda (seconds given); None: today's behaviour
        self.t0 = None
        self.seconds = None
        self.resume: list = []                    # 52.17 (5): the resume points of every "out of time" GAP: {question key, text, position, route, reply, mark, cut_left}; nothing cut by the clock is dropped
        self.landings = LD.LandingTable()         # 52.18 (1): the memo of landing answers with the inverted read index, shared by the root agenda and its inner agendas
        self.landing_lines: list = []             # the first served-from-the-table lines of the inner agendas, printed by the root
        self.landing_served = 0                   # ... mate roots whose defect list was served from the table (nothing its reads touch had changed)
        self.time_outs = 0                        # ... root agendas that met their deadline
        self.hit = False
        self.route_depth = 0                      # 52.21 (3): the routes being chased right now (nested: a route's reply opens an inner agenda that chases routes)
        self.route_names: list = []
        self.hand: "int | None" = None            # ... the route IN HAND when the deadline was first seen: the innermost one (its depth). It runs to its end; no other route is opened
        self.abort = False                        # ... the human moved (the background tier): nothing is finished, every test of the clock stops
        self.finish_lines: list = []              # ... "out of time: finishing <route>, no new routes", printed by the root
        self.roots_served = 0                     # 52.21 (1): root lists (the reach roots) served from the fact store, not rescanned
        self.finished = 0                         # 52.21 (3): routes that were in hand when the deadline passed and were run to their end
        self.chase_served = 0                     # 52.21 (2): chase claims served from the store on a board whose cells matched the claim's read set
        self.refuter_fixes = 0                    # 52.22 (3): killed routes saved by the refuter fix (the refuting piece exchanged with tempo, the route asked again after it)
        self.mate_levels: list = []               # 52.23 (6): the prep moves the mate chain in hand may still spend (tempo arithmetic: a mate pattern with d defects needs at most d prep moves)
        self.slice_len = 0.0                      # 52.23 (11): the time slice the root route in hand was given
        self.clock_stops = 0                      # 52.23 (8): chases not started by the chase clock
        self.mate_walks = 0                       # ... mate chases not started because the chain had spent them (the walk of checks)
        self.parry_seen = 0                       # plan 61 A2: closed replies checked against the route's threat / ...
        self.parry_closed = 0                     # ... of which the reply does not parry it (closed by the threat)
        self.route_threats: dict = {}             # (position key, uci) -> the Threat of the route (None: no object)
        self.delayed_routes = 0                   # plan 61 A3 / A4: routes admitted by the delayed threat / by no safe move
        self.nosafe_routes = 0
        self.est_rows: list = []                  # plan 61 B2: the estimated leaves of this root (written to estimates_log.jsonl by `decision()`)
        self.est_set = ""                         # ... the set name the rows carry
        self.judge_t = 0.0                        # plan 61 C2: seconds spent in the root routes the judge let finish (killed or closed, not cut by a clock) / ...
        self.judge_n = 0                          # ... the inner agendas those routes ran: seconds_per_node = judge_t / judge_n
        self.root_end = 0.0                       # ... the end of the root's share of the time for the route being judged (time.monotonic())
        self.route_costs: dict = {}               # 64.3: (position, route, stake) -> R.route_cost, asked once per run
        self.route_secs: dict = {}                # 64.3: (position, root route, stake) -> (seconds it took, did it finish): a route READ before carries its measured cost into the later tiers / questions
        self.rate_t = 0.0                         # ... the rate measured by the EARLIER tiers of this root (seconds / inner agendas): the run's rate, not the tier's
        self.rate_n = 0
        self.held = None                          # plan 65.1: () -> does the best claim so far already keep the status (set by the tier driver; None: no tiers, no competition)
        self.competitors: dict = {}               # ... (position, stake, tier) -> [(move, san, why)]: the routes the NEXT tier would read at the root of a question (priced on demand, `_price`)
        self.root_gaps: dict = {}                 # plan 65.2: (position, route, stake) -> (cost, san): the root routes of this tier that ended as GAPs and are still open
        self.compete_cuts = 0                     # ... routes cut because the next tier's cheapest route was cheaper
        self.opened: set = set()                  # ... the root routes this tier has opened (read at least once): a route already open is not a NEW route, the competition does not cut it
        self.deferred: set = set()                # ... their keys (position, route, stake): read in the NEXT tier, which does not defer them again
        self.exempt: frozenset = frozenset()      # ... the routes the previous tier deferred to this one: read here, whatever tier k+2's cheapest route costs
        self.carry_q: dict = {}                   # plan 66.1: (goal, text, stake) of a ROOT question -> {net, second, claim, tier, dep}: its proved claim in an earlier tier (handed on by the tier driver)
        self.carry_status: "dict | None" = None   # ... the root's status when this tier opened (a carried claim that keeps it ends the question)
        self.carried_q = 0                        # ... root questions served a carried claim / ...
        self.carried_q_kept = 0                   # ... of which not re-asked (the claim keeps the status, or no chase made it: a schema binding is the same in every tier)
        self.carried_q_reasked = 0                # ... re-asked for a better claim above the carried floor
        self.carry_r: dict = {}                   # plan 66.2: (position key, route key, reply uci) -> {g, part, lemma, closed, tier}: a reply PROVED in an earlier tier (closed by rule, or an inner claim): a proof over the same board
        self.proofs: dict = {}                    # ... the replies this tier proved (the tier driver hands them on, with carry_r, to the next tier)
        self.carry_served = 0                     # ... replies served from carry_r / ...
        self.carry_reread = 0                     # ... carried replies read again: their proof is below what this read needs (the agenda's target, the floor of 66.1, the goal's stake for a close)
        self.carry_routes = 0                     # ... root routes resumed with at least one carried reply
        self.line: list = []                      # plan 66.4: the line being read, root first: (position before our move, our move, their reply, the stake) per step
        self.chain_line: list = []                # ... the line at the last chain stop (copied by `_chain_gap`)
        self.extend = 0                           # ... the extra levels of the chain allowance a resumed read is given (0: none); applied at a stop only while the line there is still forcing
        self.forcing_steps: dict = {}             # ... (position, move, reply, stake) -> (still forcing?, why): one step judged once
        self.chain_gaps: list = []                # ... the ROOT routes this tier ended as chain-stop GAPs: {key, route, uci, stake, line, gap}
        self.resume_set: "dict | None" = None     # ... a resume pass: carry key of a root question -> the route ucis to read again (None: not a resume pass)
        self.ext_used = 0                         # ... chain stops passed by the extension (the line was still forcing) / ...
        self.ext_quiet = 0                        # ... not passed: the line had gone quiet
        self.resume_passes = 0                    # ... resume passes run by the root
        self.need: "int | None" = None            # plan 68: what a claim must bring to keep the status (line - balance); None: no tiers (nothing deferred)
        self.deferred_qs: list = []               # ... the root questions deferred this tier (stake below the need, not a save)
        self.resume_ext: dict = {}                # ... (carry key of a root question, route uci) -> {ext, moves}: the root routes the previous tier stopped by the chain while the line was still forcing; read here with the allowance +ext
        self.resumed = 0                          # ... root routes read with the extension
        self.judge_reads = 0                      # ... nodes the judge let be read / cut (a GAP)
        self.judge_cuts = 0
        self.judge_routes = 0                     # ... root routes judged
        self.judged = False                       # ... the last cut of `_inner` was the judge's (the GAP text says so)
        self.c_lines: list = []                   # plan 61 C2 / C4: trace lines of the judge's cuts inside a route and of the cycle claims (the root prints them after the route)
        self.history: set = set()                 # plan 61 C4: position keys (_pos_key) of the game so far, us to move: a chase position that repeats one is a repetition (the play server may fill it)
        self.root_pk: "str | None" = None         # ... the root agenda's position (the line's first)
        self.active_label: dict = {}              # ... position key -> "the position after <route> <reply>" of the inner agendas on the line
        self.cycles = 0                           # ... cycle claims made
        self.est_routes = 0                       # ... gap routes that closed as an estimated claim
        self.est_why: dict = {}                   # ... and, for the gap routes that did not, the reason (a counter for the OBS line)
        self.overload_opens = 0                   # 2026-10-20: replies not closed by the exchange because our capture left a duty (their claim after it)
        self.unmet = 0                            # 52.22 (1): replies NOT closed by the exchange because the chased goal (win the piece) is not met by it and a forcing fix of its defect remains
        self.open_lines: list = []                # ... and the trace line of each (printed by the root)
        self.unaddressed = 0                      # 52.17 (2): forcing moves NOT followed by a chase because they fix no defect of the goal and make no claim of their own


    def carry(self) -> "ChaseCtx":
        """52.19 (4): a fresh context for the next root agenda that keeps what outlives one agenda -- the fact store, the FEN memos, the route memo, the landing table (all exact or
        read-set valid) -- with the counters, the fuse and the deadline new."""
        c = ChaseCtx(self.facts)
        c.memo, c.routes, c.theirs, c.their_ags, c.landings = self.memo, self.routes, self.theirs, self.their_ags, self.landings
        c.their_facts, c.gap_facts = self.their_facts, self.gap_facts
        return c

    def out_of_time(self) -> bool:
        """52.17 (5): the deadline (or the slice of it this route may use) has passed. Never true when no seconds were given. 52.21 (3): the first time it is seen, the route being chased at that
        moment (the innermost) is IN HAND: `finishing` says it may run to its end."""
        if self.deadline is None:
            return False
        if time.monotonic() >= self.deadline:
            self.hit = True
            if self.hand is None:
                sliced = self.final is not None and self.final - self.deadline > 1e-6          # a SLICE of the final deadline (the root shares the time fairly between its routes): the route is cut where it is, as in 52.17; only the final deadline lets the route in hand finish
                self.hand = 0 if sliced else self.route_depth
                if self.hand and not self.abort:
                    self.finish_lines.append(f"out of time: finishing {self.route_names[-1]} (the route in hand, depth {self.route_depth}), no new routes")
            return True
        return False

    def finishing(self, frame: int) -> bool:
        """52.21 (3): whether work belonging to the route at nesting depth `frame` may go on past the deadline: only the route in hand (and what it opens to read its replies), unless the human moved."""
        return self.hand is not None and not self.abort and frame > 0 and frame == self.hand


class Agenda:
    def __init__(self, state: State, side_depth: int = 1, memory: list | None = None, leaf_all: bool = False, chase_depth: int = 0, ctx: "ChaseCtx | None" = None, min_stake: int = 0, inherit: dict | None = None, chase_log: str | None = None, their: "Agenda | None" = None, keep: int = 0, seconds: "float | None" = None, facts: "FS.FactStore | None" = None, carry: "ChaseCtx | None" = None, evaded: bool = False, since: int = 0, free_level: bool = False, tier: int = 0):
        R.reset_ids()
        if chase_depth == 0 and ctx is None and carry is None:
            G.clear_caches()                                    # 52.20 (1): the word / FEN caches live for one root question
        self.their = their                                      # 52.14: THEIR agenda on this board with the move passed to them: its claims are asked as save roots (an inner agenda on a static leaf)
        self.keep = keep                                        # 52.14: the gain (a balance, us to move) the static leaf stands on: a parry of a threat is verified when it keeps at least 1 of it
        self._extra_reads: frozenset = frozenset()               # 52.22 (3): squares the deciding route read beyond its facts (the refuter's)
        self._chase_q = None                                    # 52.22 (1): the question being chased (a win question's stake is what the exchange must reach to close a reply by rule)
        self._chasing = None                                    # 52.16 (4): the promote question being chased (the inner agendas inherit it as a kill where nothing it read has changed)
        self.threat_qs: list = []                               # 52.14: (question, their claim) of every threat root pushed from `their`
        self.inherit = inherit or {}                            # 52.12 (3), 52.13 (1): key -> {kind: kill | claim, reads, diff, ...} for the outer agenda's SETTLED questions (a kill, or a claim with its reads)
        self.inherited = 0
        self.min_stake = min_stake                              # 52.12: an inner agenda asks only the roots whose stake can still rescue the route (a root below it is not pushed)
        self.free_level = free_level                            # 52.23 (6): this agenda follows a recapture on our route's square (an exchange): the chain allowance is not spent by it
        self.since_capture = since                              # 52.23 (8): THE CHASE CLOCK (Kaveh 2026-10-20) -- our moves on this chain since the last capture that was not bad for us
        self._route_move = None                                 # ... the route being read (set by _chase_route)
        self._reply_san = None                                  # plan 61 C4: the reply being read (the label of the position it leads to)
        self.evaded = evaded                                    # 52.23 (3): this agenda was opened by an EVASION route (the check before it was their intermezzo)
        self.chase_depth = chase_depth                          # 52.12: 0 for the root agenda, +1 per inner agenda of a chase
        self.ctx = ctx if ctx is not None else ((carry.carry() if carry is not None else ChaseCtx(facts)) if side_depth else None)       # their agendas (side_depth 0, no ctx) never chase
        if self.ctx is not None and chase_log is not None:
            self.ctx.tag = chase_log                            # 52.13 (4): the id the chase log rows carry
        self._root = ctx is None and self.ctx is not None       # plan 59: a ROOT agenda (the analyser's, the server's): not an inner agenda, not their agenda, not one of ponder.py's
        self.memory = memory or []                              # 52.9: [(sim, record)] recalled for this state; it only REORDERS equal stakes, never adds or removes
        self.asked_at_claim = 0                                 # questions asked when the deciding claim was set
        self.side_depth = side_depth                            # 1: run THEIR agenda (see `their_view`); 0: do not (this is their agenda)
        self.their_runs = 0                                     # how many of their agendas this one ran (all levels)
        self.state = state
        self.board = chess.Board(state.fen)
        self.me = self.board.turn
        self.us = "w" if self.me else "b"
        self.lines: list = []
        self.asked = 0
        self.killed_tempo = 0
        self.kills: list = []                                   # (question id, reason) for every kill
        self.all: list = []
        self.heap: list = []
        self.order: list = []                                   # plan 53 (1): (question, its features) in the order asked
        self.prior_moves = 0                                    # plan 53 (2): questions the prior asked ahead of the stake order
        self.best: tuple | None = None                          # (value, -opp defects, claim): PROVED claims only -- the search flow (stops, tiers, kills) reads this and nothing estimated
        self.best_est: tuple | None = None                      # plan 61 B2: (value, 0, claim) of the best ESTIMATED claim (proved=False); the decision consults it only when no proved claim keeps the status
        self._decision = None                                   # ... `decision()` is read once: (claim, estimated?)
        self.budget = sum(1 for s in chess.SQUARES if self.board.piece_at(s) and self.board.piece_at(s).color == self.me) * 8 \
            + sum(1 for m in self.board.legal_moves if self.board.is_capture(m))
        self.notes: list = []
        if ctx is None and self.ctx is not None:
            self.ctx.tier = tier                                # plan 59: the tier is a setting of the root's context (the inner agendas share it)
            self.ctx.left = self.budget * FUSE * (1 << tier)    # the root agenda's FUSE (10x the question budget; plan 59: x 2^tier), set ONCE: every chase of every question and the inner agendas spend it
            if seconds is not None and not (TIERS or TIERS_UNDER_TIME):       # (64.3: tried: the judge in place of the count fuse under the tiers lets tier 0 take the whole budget -- 4pC83 8.5 s, 5JrXD 4.7 s -- and the wider tiers never run; the fuse stays the tier's cap)       # 52.17 (5): THE BUDGET IS TIME. 52.23 (1): then there is NO question-count fuse (Kaveh 2026-10-20: unlimited time
                self.ctx.left = NO_FUSE                         # must mean the chase runs to its end); the time slices share the budget between the routes, the count never
            if seconds is not None:                             # (plan 59: with TIERS on the count fuse stays under a time budget -- the time is a cap, not a replacement -- and the tiers share the seconds)
                self.ctx.seconds, self.ctx.t0 = seconds, time.monotonic()
                self.ctx.deadline = self.ctx.final = self.ctx.t0 + seconds
        self._threats_done = not side_depth and their is None
        self._pass_done, self._pass_best = False, None          # plan 59: their best claim with the move passed to them (set by `_threat_roots`; read by `status`)
        self._pass_deep: "tuple | None" = None                  # plan 65.4: their pass read with delayed_balance_deep: (value, their move SAN, candidates read), asked once
        self._q_best: dict = {}                                 # plan 66.1: question id -> (net, second, claim): the best claim each question offered (the tier driver carries the proved ones)
        self._floor: "int | None" = None                        # ... the carried floor of the root question being re-asked (a better claim must beat it)
        self._carried_route = 0                                 # plan 66.2: the replies the deciding route served from an earlier tier (a chase claim standing on them is filed for this position only)
        self.root_reads: dict = {}                              # plan 65.3: root route (uci) -> how its last read ended: {status: killed | gap, san, why | gains}; a kill is final, a gap keeps the replies read so far
        self._hang: list = []
        self.threat_views: list = []                            # (claim, their agenda) for every threat root
        self._seq = 0
        self.promote_stake = R.STAKE_PROMOTE                                  # 52.15 (2): the stake of the promote question being chased (a reply that gave up that much needs no claim)
        self.leaf_obs: list = []                                # 52.10: (our san, claimed, leaf line, their claim or None, w, verdict) per leaf read, for the OBS lines
        self._leaf: dict = {}
        self.leaf_all = leaf_all                                # 52.10: read the leaf of EVERY reply of theirs (4x the seconds on dev_20 / fresh_60, no claim downgraded more), not only the minimising ones
        self.leaf_stats = dict(read=0, down=0, untrusted=0)       # claims read at the leaf / downgraded / not trusted (the decomposition missed the word)
        self._tv: dict = {}                                     # their view of a route, asked once (the verify and the smallest-loss decision share it)
        self.pk = _pos_key(state.fen)
        if self._root and self.ctx is not None:
            self.ctx.root_pk = self.pk                          # plan 61 C4: the line's first position
        self.fact_of: dict = {}                                 # 52.19: question id -> the id of the fact it derived or was served
        self.claim_fid: dict = {}                               # id(claim) -> the fact id that holds it
        self.premises: list = []                                # ... the ids of every fact this agenda derived or was served (an inner answer stands on them)
        self._cur: "R.Question | None" = None                   # the question being asked (the stake of the facts derived under it)
        self._qdeps: list = []                                  # the premises of the fact the current question will derive (a chase fills them)
        self._rf: "list | None" = None                          # the facts the route being chased read (inner answers, their agendas)
        self._frame: list = []                                  # the facts the listener flagged for the reply being read (restored when the reply is done)
        self._cell = FS.board_cell(self.board)
        self._piece = FS.board_piece(self.board)
        self._fs0 = self.ctx.facts.snapshot() if self.ctx is not None else None      # the store's counters when this agenda began (the deltas are its own)
        self._pre: dict = {}                                    # 52.17 (4): route key -> the replies the cheap pass closed by rule
        self.frame = self.ctx.route_depth if self.ctx is not None else 0       # 52.21 (3): the route (its depth) this agenda was opened to read the reply of
        self.late = self.ctx is not None and self.ctx.deadline is not None and time.monotonic() >= self.ctx.deadline       # ... opened after the deadline (by the route in hand): it reads its roots; one opened before finishes its current question only
        self._roots()
        for i, q in enumerate(self.all, 1):                      # numbered in the order they are asked
            q.id = i
        if getattr(self, "_tq", None):
            self._tq[1].qid = self._tq[0].id
        R._ids[0] = len(self.all)

    def _t(self, flag: bool, k: int) -> bool:
        """plan 59: a 52.23 chain item is ON when its module flag forces it (at every tier) or this root's tier is at least `k`. Read through the ctx, so the play server's ponder thread cannot race."""
        return flag or (self.ctx is not None and self.ctx.tier >= k)

    def _clock(self) -> int:
        """The chase clock (52.23 (8)): CHASE_CLOCK, and 4 (Kaveh's starting value) from tier 3."""
        return min(CHASE_CLOCK, 4) if self.ctx is not None and self.ctx.tier >= 3 else CHASE_CLOCK

    @property
    def questions(self) -> list:
        return sorted(self.all, key=lambda q: (-q.stake, q.depth, q.id))

    # ------------------------------------------------------------------------------------------------------------ roots

    def _push(self, q: R.Question, force: bool = False) -> R.Question:
        if q.stake + (self._top_other(q) if self._t(NEED_PROPAGATION, 3) else 0) < self.min_stake and not force:      # 52.23 (7): the need may be met by a CHAIN -- this question's gain and one more claim after it (zBouR: Qh1+ Rf1 Qxf1+ Rxf1 Rxf1+ Kd2 Rf2+:
            return q                                                         # the rook, 5, then the queen by the skewer, 9, meet a need of 10 that no single stake does)
        if self.ctx is not None and self.ctx.deadline is not None and self.ctx.out_of_time() and not (self.late and self.ctx.finishing(self.frame)):      # 52.21 (3): past the deadline no new question is pushed
            self.state.gaps.append(f"{q.text} (stake {q.stake}): out of time, not asked")
            return q
        self.all.append(q)
        self._seq += 1
        heapq.heappush(self.heap, (-q.stake, -self._recall_score(q)[0], q.depth, self._seq, q))
        return q

    def _recall_score(self, q: R.Question) -> tuple:
        """52.9: (summed similarity of the recalled records whose goal kind is this question's goal kind, the best such record or None)."""
        hit = [(sim, r) for sim, r in self.memory if r.get("goal") == q.goal]
        return (round(sum(s for s, _ in hit), 3), max(hit, key=lambda t: t[0])[1] if hit else None)

    def _fix_kinds(self, q: R.Question, move: str) -> tuple:
        """The kinds of the fix sub-questions on the path of the claim's first move."""
        out = []

        def walk(f):
            for g in f.fixes:
                if any(r.move.uci() == move for r in R.all_routes(g)):
                    out.append(g.goal)
                    walk(g)
        walk(q)
        return tuple(out)

    def _better(self, value, second, claim: Claim) -> bool:
        """53 (3): is (value, second, claim) a better best than the one held? Greater wins; at EQUAL value the tie is broken by the claim itself, never by the order asked (the prior
        orders, it must not decide: r3oph's Rxd1+ and Nxe5 both read 1 and the first asked won): the question's stake, then the shape key the smallest-loss rule uses (fewest
        hanging, most guarded, most poking), then the move's uci (a fixed, if arbitrary, last resort)."""
        if claim.proved and not claim.word and any(x.id == claim.qid for x in self.all):      # plan 66.1: each question's own best proved claim (the tier driver hands them to the next tier)
            old = self._q_best.get(claim.qid)
            if old is None or (value, second) > old[:2]:
                self._q_best[claim.qid] = (value, second, claim)
        if self.best is None:
            return True
        if (value, second) != self.best[:2]:
            return (value, second) > self.best[:2]
        return self._claim_key(claim) > self._claim_key(self.best[2])

    def _claim_key(self, c: Claim) -> tuple:
        """The tie-break key of a claim at equal value (see `_better`)."""
        stake = next((q.stake for q in self.all if q.id == c.qid), 0)
        mv = c.move or ""
        try:
            sh = PA.shape_key(PA.shape(self.board.fen(), [(mv[:2], mv[2:4])], self.us, threats=False)) if len(mv) >= 4 else ()
        except Exception:
            sh = ()
        return (0 if c.proved_by == "table" else 1, stake, tuple(-x for x in sh) if sh else (), mv)         # plan 62.3: at an equal value a proved word beats a table claim

    def _stamp(self, claim: Claim, q: R.Question) -> Claim:
        claim.goal = q.goal
        claim.fixes = self._fix_kinds(q, claim.move)
        claim.qid = q.id                                        # plan 53: the question log says which question decided
        return claim

    def _roots(self) -> None:
        b, me = self.board, self.me
        ksq = b.king(not me)
        if b.is_check():
            ck = list(b.checkers())
            self._push(R.new_question("save", R.STAKE_MATE, dict(checkers=ck), text="save our king (in check)"))
            for c in ck:                                         # 52.15 (1): the checker is a win target when some legal move takes it (the king's included); the routes are those captures
                if any(m.to_square == c for m in b.legal_moves):
                    pc = b.piece_at(c)
                    self._push(R.new_question("win", G.worth(pc), dict(target=c, checker=True), text=f"win their {pc.symbol().upper()} on {G.name(c)} (it gives check)"))
            self.notes.append("in check: every legal move answers it; the roots are the save-king question and the win of the checker (mate and material goals are read on their routes)")
            return
        marks = []
        for m in b.legal_moves:
            pc = b.piece_at(m.from_square)
            if ksq is None or (pc.piece_type == chess.KING and not b.is_castling(m)):
                continue
            placed = chess.Piece(m.promotion, pc.color) if m.promotion else pc        # a promotion places the NEW piece (Kaveh 2026-10-17)
            more = []
            if b.is_castling(m):                                                       # castling places the king AND the rook; the rook is the checker
                rf, rt = (chess.H1 if m.to_square > m.from_square else chess.A1), (chess.F1 if m.to_square > m.from_square else chess.D1)
                if pc.color == chess.BLACK:
                    rf, rt = rf + 56, rt + 56
                more = [(b.piece_at(rf).symbol(), rt, rf)]
            cf = G.with_piece(b, placed, m.to_square, m.from_square)
            for sym, at, frm in more:
                cf = G.with_piece(cf, chess.Piece.from_symbol(sym), at, frm)
            checker = more[0][1] if more else m.to_square
            if ksq in cf.attacks(checker):
                marks.append((m, pc, placed, more, checker))
        scored = []
        for m, pc, placed, more, checker in marks:
            q = R.new_question("mate", R.STAKE_MATE, dict(sym=placed.symbol(), at=m.to_square, frm=m.from_square, king=ksq, more=more, move=m.uci()))
            q.defects = R.mate_defects(b, self.us, R.placement_of(q))
            how = b.san(m) if (m.promotion or more) else f"{pc.symbol().upper()} {G.name(m.from_square)}->{G.name(m.to_square)}"
            q.text = f"mate on {G.name(checker)} ({how}, their king on {G.name(ksq)})"
            scored.append((len(q.defects), G.worth(placed), q.id, q))                 # fewest defects, then the CHEAPER piece (the heavy one stays back as a guard)
        for _, _, _, q in sorted(scored, key=lambda t: t[:3]):
            self._push(q)
        self._reach_roots(ksq)
        hang = RS.hanging_of_ours(b, self.us)
        if hang:
            big = max(hang, key=lambda s: (G.worth(b.piece_at(s)), -s))
            self._push(R.new_question("save", G.worth(b.piece_at(big)), dict(target=big), text=f"save our {b.piece_at(big).symbol().upper()} on {G.name(big)} (it hangs)"))
        self._hang = hang                                        # the threat roots (52.4a) are asked for lazily: see ask()
        self._promote_roots()
        theirs = [s for s in chess.SQUARES if b.piece_at(s) and b.piece_at(s).color != me and b.piece_at(s).piece_type != chess.KING]
        for s in sorted(theirs, key=lambda s: (-G.worth(b.piece_at(s)), s)):
            self._push(R.new_question("win", G.worth(b.piece_at(s)), dict(target=s),
                                      text=f"win their {b.piece_at(s).symbol().upper()} on {G.name(s)} (worth {G.worth(b.piece_at(s))})"))
        self._fork_roots()
        self._table_roots()

    def _table_roots(self) -> None:
        """plan 62.3: a position whose whole board is at most five men closes by the TABLE (regress-free, exact): a claim labelled `proved_by = "table"` (a win: the material of their last pieces, a bare king left
        is a mate; a draw when we are behind: minus the balance). It competes like any claim (value first; at an equal value a proved word beats it, `_claim_key`)."""
        b = self.board
        if not TB.TABLE or b.turn != self.me or not TB.covered(b):
            return
        tc = TB.table_claim(b)
        if tc is None:
            return
        m = chess.Move.from_uci(tc["move"])
        san = b.san(m)
        q = R.new_question("win", tc["value"], dict(table=tc["code"]), text=f"{tc['text']}")
        self.all.append(q)
        claim = Claim(tc["kind"], san, tc["value"], "syzygy", f"{san} ({tc['text']})", move=tc["move"], goal=q.goal, proved_by="table")
        self._tq = (q, claim)                                   # (its qid is set once the questions are numbered)
        self.state.claims.append(claim)
        q.status = "verified"
        q.claimed = (0, claim, frozenset())
        self.lines.append(f"  -> CLAIM {tc['kind']} {san} (value {tc['value']}), {tc['text']}; label: table (exact by exhaustion, unexplained)")
        if self._better(tc["value"], 0, claim):
            self.best = (tc["value"], 0, claim)
            self.asked_at_claim = self.asked

    def _top_other(self, q: R.Question) -> int:
        """52.23 (7): the largest worth among their pieces other than this question's target (what one more claim on this chain could add); 0 at the root (no need there)."""
        if not self.min_stake:
            return 0
        b, t = self.board, q.pattern.get("target")
        if t is None or not b.is_attacked_by(self.me, t):               # only a question whose target we can TAKE now chains (the capture, then one more claim)
            return 0
        return max([G.worth(p) for s, p in b.piece_map().items() if p.color != self.me and p.piece_type != chess.KING and s != t] or [0])

    def _fork_roots(self) -> None:
        """52.18 (4): THE FORK SQUARE AS A ROOT -- `fork on s with our <piece>` (stake = the smaller of the two values it attacks; the king and one piece for a check), for the root
        agenda only (an inner agenda reads the forks its own chase finds). The knight-fork colour lemma kills the pairs a knight cannot fork, with the read set."""
        b = self.board
        if self.chase_depth or self.ctx is None or b.is_check() or self.min_stake:
            return
        for a, c2, reads in LD.knight_fork_kills(b, self.me):
            self.notes.append(f"fork of the {b.piece_at(a).symbol().upper()} on {G.name(a)} and the {b.piece_at(c2).symbol().upper()} on {G.name(c2)} by a knight: killed by reasoning -- "
                              f"a knight attacks only squares of one colour (Ideas.GeoLemmas.knight_fork_same_colour), these are of two; reads {{{', '.join(G.name(x) for x in sorted(reads))}}}")
        for f in LD.fork_candidates(b, self.me):
            m = f["move"]
            names = ", ".join(f"the {b.piece_at(t).symbol().upper()} on {G.name(t)} (wins {w})" for t, w in f["targets"])
            self._push(R.new_question("fork", f["bound"], dict(move=m.uci(), san=f["san"], frm=m.from_square, at=m.to_square, targets=f["targets"], check=f["check"]),
                                      text=f"fork on {G.name(m.to_square)} with our {f['piece'].symbol().upper()} ({f['san']}): " + ("check and " if f["check"] else "") + f"attacks {names}"))

    def _reach_roots(self, ksq: int | None) -> None:
        """52.12 (4): the mate roots at distance two, only when a mate is the stake (one prep move): a piece of ours that would check their king from a square it needs
        two moves to reach, the pattern being a MATE otherwise (no other defect). The defect `reach` names the first steps; the fix routes are those steps.
        52.21 (1): THE ROOT LIST IS A FACT. The roots of one piece are a pure function of the cells its scan read (`_scan_reach` records them); filed in the store under (their king, the piece, our king),
        they are served to every agenda whose board has those cells as they were -- a sibling, a transposition, the next reply -- without rescanning. A knight reads a few squares and is served
        often; a queen reads the lines of their king and seldom is."""
        b = self.board
        if ksq is None:
            return
        fs = self.ctx.facts if self.ctx is not None else None            # their agendas (no ctx) scan as before
        for frm in sorted(chess.SquareSet(b.occupied_co[self.me] & ~b.kings & ~b.pawns)):
            pc = b.piece_at(frm)
            key = ("roots:reach", ksq, pc.symbol(), frm, b.king(self.me))
            f = fs.lookup(key, self.me, self._cell, piece=self._piece)[0] if fs is not None else None
            if f is not None:
                specs = f.payload["specs"]
                self.ctx.roots_served += 1
                self.lines.append(f"roots served from the store (fact #{f.id}, the reach roots of the {pc.symbol().upper()} on {G.name(frm)}, derived under {f.under})")
            else:
                specs, reads, occ = self._scan_reach(frm, ksq)
                if fs is not None:
                    occ = occ - reads
                    fs.add("roots", key, self.me, f"reach roots of the {pc.symbol().upper()} on {G.name(frm)}, their king on {G.name(ksq)}", reads=reads, occ=occ,
                           sig=FS.signature(self._cell, reads) + FS.occ_signature(self._piece, occ), stake=R.STAKE_MATE, depth=self.chase_depth, under=f"depth {self.chase_depth}", pk=self.pk, payload=dict(specs=specs))
            for sym, frm_, s, steps in specs:
                q = R.new_question("mate", R.STAKE_MATE, dict(sym=sym, at=s, frm=frm_, king=ksq, more=[], move=None, reach=list(steps)))
                via = ", ".join(sorted({G.name(chess.Move.from_uci(u).to_square) for u in steps}))
                q.defects = [R.Defect("reach", s, [frm_], f"the {pc.symbol().upper()} needs 2 moves (via {via})")]
                q.text = f"mate on {G.name(s)} ({pc.symbol().upper()} {G.name(frm_)}->{G.name(s)} in 2 moves via {via}, their king on {G.name(ksq)})"
                self._push(q)
        if any(q.pattern.get("reach") for q in self.all):
            self.notes.append("reach roots (mate at distance 2) only when a mate is the stake: one prep move, the pattern otherwise a mate")

    def _scan_reach(self, frm: int, ksq: int) -> tuple:
        """The reach roots of the piece of ours on `frm` by scanning, with the cells the scan READ (52.21 (1)). -> ([(piece symbol, from, target, the first steps as UCI)], the squares read with
        their attackers (the question's own reads: the mate defects), the squares read for their occupant only (rays, lines, legal moves)). A square is read when a change on it could add or remove a
        root: their king, our king and the lines a pin comes along, the piece's square and, for every square from which this piece type would attack their king on an empty board, that square and
        the squares between it and the king (the check can be blocked); for a landing square that passed, what `reach_steps` read and what the mate defects read (the question's read set)."""
        b, me = self.board, self.me
        reads, occ = set(), {ksq, frm}
        if b.king(me) is not None:
            occ.add(b.king(me))
        specs = []
        pc = b.piece_at(frm)
        table = None
        for s in chess.SQUARES:
            if s == frm or not _aligned(pc.piece_type, s, ksq):
                continue
            occ.add(s)
            occ.update(chess.SquareSet(chess.between(s, ksq)))
            o = b.piece_at(s)
            if o is not None and o.color == me:
                continue
            cf = G.with_piece(b, pc, s, frm)
            if ksq not in cf.attacks(s):
                continue
            if table is None:
                table = R.ReachTable(b, self.us, frm)
            occ.add(s)
            steps = table.steps(s)
            if not steps:
                continue
            q = R.new_question("mate", R.STAKE_MATE, dict(sym=pc.symbol(), at=s, frm=frm, king=ksq, more=[], move=None, reach=[m.uci() for m in steps]))
            q.defects = R.mate_defects(b, self.us, R.placement_of(q))
            qreads = R.reads_of(q, b)
            reads |= qreads
            occ |= R.attack_lines(R.future_board(b, R.placement_of(q), not me), qreads)      # the defects are read on the counterfactual (the piece moved): what could attack these squares THERE
            if q.defects:
                continue                                         # another defect as well: two prep moves, the tempo arithmetic kills it before it is asked
            specs.append((pc.symbol(), frm, s, tuple(m.uci() for m in steps)))
        if table is not None:
            occ |= table.reads
        return specs, frozenset(reads), frozenset(occ)

    def _promote_roots(self) -> None:
        """52.6: `promote our P on <sq>` (stake 8) for every pawn of the mover with `pushes` <= 3 (farther pawns: the mate / material stakes dominate)."""
        b = self.board
        near = []
        for s in sorted(chess.SquareSet(b.pawns & b.occupied_co[self.me])):
            n = G.word(b, ("pushes", G.name(s)), self.me)
            if 0 < n <= R.PROMOTE_MAX_PUSHES:
                near.append((n, s))
        for n, s in sorted(near):
            self._push(R.new_question("promote", R.STAKE_PROMOTE, dict(pawn=s, pushes=n), text=f"promote our P on {G.name(s)} ({n} push{'es' if n > 1 else ''} to {G.name(R.pawn_path(b, s, n)[-1])})"))
        if near:
            self.notes.append(f"promote roots only for pawns with pushes <= {R.PROMOTE_MAX_PUSHES} (the mate / material stakes dominate farther pawns)")

    # ------------------------------------------------------------------------------------------------------------ their side (52.4)

    def _their_agenda(self, board: chess.Board) -> "Agenda":
        """Their agenda on `board` (their turn, as given), side_depth 0, asked to its end."""
        saved = R._ids[0]
        ag = Agenda(State(fen=board.fen(), us="w" if board.turn else "b", pieces=[]), side_depth=0)
        ag.ask()
        R._ids[0] = saved
        self.their_runs += 1 + ag.their_runs
        return ag

    def their_view(self, route: chess.Move) -> tuple:
        """Their agenda on the board after our route: -> (their best claim or None, the agenda or None). None, None when they have no move (mate / stalemate)."""
        if route in self._tv:
            return self._tv[route]
        c = self.board.copy(stack=False)
        c.push(route)
        if not any(True for _ in c.legal_moves):
            self._tv[route] = (None, None)
            return self._tv[route]
        ag = self._their_agenda(c)
        best = max(ag.state.claims, key=lambda x: x.value) if ag.state.claims else None
        self._tv[route] = (best, ag)
        return self._tv[route]

    def _threat_roots(self, hang: list) -> None:
        """(a) Their agenda on the current board as if we passed: every claim of theirs is a threat and a save root of ours (stake = its value). Not when we
        are in check (then there is nothing to pass). A claim that is just the capture of a piece that hangs now is the one-move case `hang` already roots."""
        b = self.board
        if b.is_check():
            return
        given = self.their is not None                           # 52.14: an inner agenda on a static leaf is handed their agenda; its threat roots are asked whatever `min_stake` says
        ag = self.their if given else self._their_agenda(G.with_turn(b, not self.me))
        if not given:
            self._pass_done, self._pass_best = True, (max(ag.state.claims, key=lambda x: x.value) if ag.state.claims else None)      # plan 59: the status reads this (no second run of their agenda)
        seen = set()
        for cl in ag.state.claims:
            m = chess.Move.from_uci(cl.move) if cl.move else None
            if m is None or cl.move in seen or (not given and m.to_square in hang and b.piece_at(m.to_square)):       # (given: the hang root is below `min_stake` here, so the capture of a piece that hangs is a threat root too)
                continue
            seen.add(cl.move)
            if cl.kind == "promote":
                pw = m.from_square
                pb = ag.board
                ppath = R.pawn_path(pb, pw, G.word(pb, ("pushes", G.name(pw)), not self.me))
                if not ppath:
                    continue                                     # (their claim's move is not a pawn's push with a path: nothing to stop by this root)
                dest = ppath[-1]
                q = self._push(R.new_question("save", cl.value, dict(promoter=pw, push=cl.move, value=cl.value, lemma=cl.lemma, line=cl.line, what=cl.what, inner_threat=given),
                                             text=f"stop their P on {G.name(pw)} (promotes on {G.name(dest)}): their claim {cl.what} wins {cl.value} (lemma {cl.lemma})"), force=given)
                self.threat_qs.append((q, cl))
                self.threat_views.append((cl, ag))
                continue
            tgt = self._targets_of(m, cl)
            what = "our king" if cl.value >= BD.MATE_VALUE else (" and ".join(f"our {b.piece_at(t).symbol().upper()} on {G.name(t)}" for t in tgt) or "something of ours")
            q = self._push(R.new_question("save", cl.value, dict(threat=cl.move, targets=tgt, value=cl.value, lemma=cl.lemma, line=cl.line, what=cl.what, inner_threat=given),
                                          text=f"save {what}: their threat {cl.what} wins {cl.value} (their claim, lemma {cl.lemma})"), force=given)
            self.threat_qs.append((q, cl))
            self.threat_views.append((cl, ag))

    def _targets_of(self, m: chess.Move, cl: Claim) -> list:
        """Our pieces (not the king) that their threatening move newly attacks (the ones it would win); [] for a mate."""
        if cl.value >= BD.MATE_VALUE:
            return []
        b = G.with_turn(self.board, not self.me)
        if b.is_capture(m):                                      # a capture threatens the piece it takes
            return [m.to_square]
        before = set(b.attacks(m.from_square))
        b.push(m)
        out = [t for t in b.attacks(m.to_square) if t not in before and b.piece_at(t) and b.piece_at(t).color == self.me and b.piece_at(t).piece_type != chess.KING]
        return sorted(out, key=lambda t: (-G.worth(b.piece_at(t)), t))

    # ------------------------------------------------------------------------------------------------------------ the loop

    def ask(self) -> None:
        if (TIERS or (TIERS_UNDER_TIME and self.ctx is not None and self.ctx.seconds is not None)) and self._root and self.ctx.tier == 0:          # plan 59 (3): the root is driven through the widening tiers (tier 0 is this agenda; wider tiers run on fresh contexts)
            self._ask_tiers()
        else:
            self._ask_once()

    def _ask_once(self) -> None:
        """One run of this root agenda to its end (what `ask` was before plan 59)."""
        if SCHEDULER and self.ctx is not None and self.chase_depth == 0 and self.ctx.sched is None:       # plan 56 (1): the root agenda is run by the worker loop (the inner agendas are tasks of it); ponder.py's agendas are not roots and stay recursive
            from catspace.incubator.analyst.discovery import scheduler as SC
            SC.run_root(self)
        else:
            self._ask_loop()
            self._ask_finish()

    # ------------------------------------------------------------------------------------------------------------ plan 59: the status

    def status(self) -> dict:
        """plan 59 (1): THE STATUS of the root. s0 = our material balance minus the value of THEIR best claim with the move passed to them (their agenda on this board, the threat
        `_threat_roots` reads; 100 for a mate threat; 0 when none or when we are in check: there is no pass). The word: winning (s0 >= 1), level (0), losing (< 0). The LINE we must
        stay on: 1 ('winning: stay ahead') when s0 >= 1, else 0 ('not losing: reach level')."""
        b = self.board
        bal = AC.Pos(b.fen()).balance(self.me) - 64              # (the word adds 64: the owner's material less the other's, plus 64)
        if b.is_check():
            cl = None
        else:
            if not self._pass_done:                              # (not asked yet: a mate of ours ended the agenda before the threat roots, or the roots were not read)
                ag = self._their_agenda(G.with_turn(b, not self.me))
                self._pass_done, self._pass_best = True, (max(ag.state.claims, key=lambda x: x.value) if ag.state.claims else None)
            cl = self._pass_best
        threat = min(cl.value, BD.MATE_VALUE) if cl is not None else 0
        deep = None
        if DEEP_PASS and cl is None and not b.is_check() and self._piece_attacked():
            dv, dsan, dn = self._deep_pass()                     # plan 65.4: the d1 family found no claim of theirs and a piece of ours is attacked: "cannot prove" is not "no threat" -- the deep family reads the pass
            if dv > 0:
                threat, deep = dv, f"{dsan}, {dn} candidates read"
        s0 = bal - threat
        return dict(balance=bal, threat=threat, s0=s0, word="winning" if s0 >= 1 else "level" if s0 == 0 else "losing", line=1 if s0 >= 1 else 0, deep=deep)

    def keeps(self, st: dict, best: "tuple | None" = None) -> tuple:
        """plan 59 (2): KEEPS THE STATUS -- the best claim (its net value, as `self.best[0]` ranks it; a save in check by the smallest loss included) keeps it when balance + value >= the line.
        No claim: not kept. -> (kept, the value or None)."""
        best = self.best if best is None else best
        if best is None:
            return False, None
        return st["balance"] + best[0] >= st["line"], best[0]

    def status_line(self, tier: int, st: "dict | None" = None, best: "tuple | None" = None, opened: str = "") -> str:
        """The trace line of plan 59: 'STATUS: losing (balance -3, their threat 0): the line is reach level (0); tier 0's best claim d3 +1 gives -2: not kept; tier 1 opened (...)'."""
        best = self.best if best is None else best
        if best is not None and best[0] >= BD.MATE_VALUE and st is None:
            return f"STATUS: tier {tier}'s best claim {best[2].what} is a mate: the status is kept"      # (a mate ends the game: the status needs no reading)
        st = st or self.status()
        kept, v = self.keeps(st, best)
        head = f"STATUS: {st['word']} (balance {st['balance']:+d}, their threat {st['threat']}{' (deep: ' + st['deep'] + ')' if st.get('deep') else ''}): the line is '{'stay ahead' if st['line'] else 'reach level'}' ({st['line']}); " if tier == 0 else ""
        claim = "no claim" if v is None else f"best claim {best[2].what} {v:+d} gives {st['balance'] + v:+d}"
        return f"{head}tier {tier}'s {claim}: {'kept' if kept else 'not kept'}" + (f"; {opened}" if opened else "")

    TIER_WHY = ("today's moves", "quiet threats worth the stake followed below the root", "mate-in-two threats", "the fuse x8: the 52.23 chain items run at every tier since plan 64.2, so this tier is the widest fuse")

    def _ask_tiers(self) -> None:
        """plan 59 (3): THE DRIVER. Run tier 0 (this agenda); if the best claim does not keep the status, run the next tier on a FRESH context (a kill under a narrower move set is not a kill under a
        wider one: facts are not carried across tiers; the play server's carried store stays tier 0's); stop at the first tier that keeps the status or after the last. The DECISION is the best claim
        over all tiers run (`_better`'s order: claims are verified, a wider tier only finds more). Under a time budget every tier keeps its count fuse (the time is a cap) and the tiers share the
        seconds: a tier starts only while time remains, and a tier cut by the deadline still contributes its verified claims."""
        total = self.ctx.seconds
        t0 = time.monotonic()
        self.tier_info: list = []
        self.status_info = None
        owner, ag, decided, k = self, self, 0, 0
        if DEFER:                                                # plan 68: the status is a fact of the position: read first, so tier 0 already asks only the questions relevant to it
            self.status_info = self.status()
            self.ctx.need = self.status_info["line"] - self.status_info["balance"]

        def held() -> bool:                                      # plan 65.1: does the best claim so far (this tier's or an earlier one's) END the widening? it does when it keeps the status and no raise is coming
            cands = [x for x in (self.best, ag.best) if x is not None]
            if not cands:
                return False
            top = max(cands, key=lambda x: (x[0], x[1]))
            if top[0] >= BD.MATE_VALUE:
                return True
            st = self.status_info if self.status_info is not None else self.status()
            if not self.keeps(st, top)[0]:
                return False
            return not (RAISE_TO_WIN and st["line"] == 0 and st["balance"] + top[0] < 1)       # (kept at 'not losing' with time left: the line rises to 'stay ahead' and the next tier opens anyway -- 59 (4))
        while True:
            ag.ctx.held = held
            t1 = time.monotonic()
            ag._ask_once()
            secs = time.monotonic() - t1
            if ag is not self:                                   # fold the wider tier into this root: its claims and gaps, its decision when it beats the one held
                if self._fold_tier(ag, k, owner):
                    owner, decided = ag, k
            overall = self.best
            mate = overall is not None and overall[0] >= BD.MATE_VALUE
            if self.status_info is None and not mate:
                self.status_info = self.status()                 # (read once, at tier 0's end: their pass is the position's, not the tier's)
            kept = mate or self.keeps(self.status_info, overall)[0]
            self.tier_info.append(dict(tier=k, secs=round(secs, 2), asked=ag.asked + ag.ctx.asked, kept=kept, hit=bool(ag.ctx.hit), claims=len(ag.state.claims),
                                       best=None if ag.best is None else (ag.best[2].what, ag.best[0])))
            st = self.status_info
            if kept and k > 0:
                self.lines.append(f"tier {k} keeps the status: {overall[2].what}" + ("" if mate else f" ({st['balance'] + overall[0]:+d} against the line {st['line']})"))
            left = None if total is None else total - (time.monotonic() - t0)
            afford = None
            widens = not kept or (RAISE_TO_WIN and not mate and st["line"] == 0 and overall is not None and st["balance"] + overall[0] < 1)       # (the widening goes on: no claim keeps the status, or 'not losing' is reached and 'winning' is asked)
            if STOP_AFFORD and widens and left is not None and left > 0.05 and k < TIER_COUNT - 1 and not self.ctx.abort:
                afford = self._affordable(ag, k, left, t0)           # plan 65.2: nothing affordable is left in ANY tier: the widening (and the raise) ends here
            if afford is not None and not afford[0]:
                self.lines.append(afford[1])
                self.stop_line = afford[1]
            if RAISE_TO_WIN and (afford is None or afford[0]) and kept and not mate and left is not None and left > 0.05 and k < TIER_COUNT - 1 and not self.ctx.abort \
                    and st["line"] == 0 and overall is not None and st["balance"] + overall[0] < 1:
                # plan 59 (4), Kaveh 2026-10-20 ("if we have time, yes; we'll figure out time management later"): 'not losing' is reached but not 'winning', and time remains: the
                # line rises to 'stay ahead' and the widening goes on (kcQil: Qxf3 reaches level, Qc4 wins). Under a count budget (no time) nothing changes.
                self.lines.append(f"STATUS kept at 'not losing' ({overall[2].what}: {st['balance'] + overall[0]:+d}) with {left:.1f} s left: the line rises to 'stay ahead' (1)")
                self.status_info = st = dict(st, line=1, raised=True)
                kept = False
            stop = kept or k >= TIER_COUNT - 1 or self.ctx.abort
            if not stop and left is not None and left <= 0.05:           # (`ctx.hit` alone is a SLICE of the budget met, not the budget)
                stop = True
            if not stop and afford is not None and not afford[0]:
                stop = True
            if k == 0 or not kept:
                why = "" if stop and kept else ("nothing affordable is left (see TIERS)" if stop and afford is not None and not afford[0] else f"no time left for tier {k + 1}" if stop and not kept and left is not None and k < TIER_COUNT - 1 else
                                                "all tiers run" if stop and not kept else f"tier {k + 1} opened ({self.TIER_WHY[k + 1]})")
                if k == 0:
                    self.lines.append(self.status_line(0, st, overall, why))
                else:
                    self.lines.append(f"tier {k}'s " + ("no claim" if ag.best is None else f"best claim {ag.best[2].what} {ag.best[0]:+d} gives {st['balance'] + ag.best[0]:+d}") + f": not kept" + (f"; {why}" if why else ""))
            if stop:
                break
            k += 1
            nxt = self._next_agenda(ag, k, left, st, k - 1)
            nxt.lines = _TierLines(self.lines, f"[tier {k}] ")
            ag = nxt
        if not kept and not self.ctx.abort and (afford is None or afford[0]):
            self.lines.append(self._open_line(ag, time.monotonic() - t0))      # plan 66.3: why the widening did not end before the deadline (what was still open and affordable)
        if RESUME and not kept and not self.ctx.abort and total is not None:
            owner, decided = self._resume_chain_stops(ag, total, t0, owner, decided)      # plan 66.4
        if DEFER and not kept and not self.ctx.abort and total is not None and ag.ctx.deferred_qs:      # plan 68: the questions below the need, with the time the relevant ones left
            left = total - (time.monotonic() - t0)
            names = ", ".join(q.text for q in ag.ctx.deferred_qs[:4]) + (", ..." if len(ag.ctx.deferred_qs) > 4 else "")
            if left > 0.05:
                da = self._next_agenda(ag, ag.ctx.tier, left, self.status_info, "deferred")
                da.ctx.need = 0
                da.ctx.held = None
                da.lines = _TierLines(self.lines, "[deferred] ")
                self.lines.append(f"DEFERRED PASS: {len(ag.ctx.deferred_qs)} question(s) below the need {ag.ctx.need} ({names}) asked with {left:.1f} s left (the relevant ones are carried)")
                t1 = time.monotonic()
                da._ask_once()
                secs = time.monotonic() - t1
                if self._fold_tier(da, ag.ctx.tier, owner):
                    owner, decided = da, ag.ctx.tier
                self.tier_info.append(dict(tier="deferred", secs=round(secs, 2), asked=da.asked + da.ctx.asked, kept=False, hit=bool(da.ctx.hit), claims=len(da.state.claims),
                                           best=None if da.best is None else (da.best[2].what, da.best[0])))
            else:
                self.lines.append(f"DEFERRED: {len(ag.ctx.deferred_qs)} question(s) below the need {ag.ctx.need} ({names}) not asked: no time left")
        self.tier_decided = decided
        self.tier_secs = [t["secs"] for t in self.tier_info]
        self.lines.append(f"TIERS: ran {len(self.tier_info)} (0..{len(self.tier_info) - 1}); the decision is " + (f"{self.best[2].what} ({self.best[0]:+d}) from tier {decided}" if self.best is not None else "none") + f"; seconds per tier {self.tier_secs}")

    def _fold_tier(self, ag: "Agenda", k, owner: "Agenda") -> bool:
        """plan 59 (3): fold a wider tier (or a resume pass) into this root: its claims and gaps, its counters; -> True when its best claim beats the one held (it decides)."""
        for u, d in ag.root_reads.items():               # plan 65.3: what the wider tier learned of each root route (a kill is final, the longer read of a GAP stands)
            old = self.root_reads.get(u)
            if old is None or (old["status"] != "killed" and (d["status"] == "killed" or len(d["gains"]) >= len(old["gains"]))):
                self.root_reads[u] = d
        self.their_runs += ag.their_runs
        self.ctx.est_rows += ag.ctx.est_rows                 # plan 61 B2: the wider tier's estimated leaves and claims are the root's
        self.ctx.est_routes += ag.ctx.est_routes
        self.state.estimated += [dataclasses.replace(c, qid=-1) for c in ag.state.estimated]
        if ag.best_est is not None and (self.best_est is None or (ag.best_est[0], ag.best_est[2].confidence) > (self.best_est[0], self.best_est[2].confidence)):
            self.best_est = (ag.best_est[0], 0, dataclasses.replace(ag.best_est[2], qid=-1))
        for f in ("parry_seen", "parry_closed", "delayed_routes", "nosafe_routes", "judge_routes", "judge_reads", "judge_cuts", "judge_t", "judge_n", "cycles", "carried_q", "carried_q_kept", "carried_q_reasked", "carry_served", "carry_reread", "carry_routes", "ext_used", "ext_quiet", "resumed"):
            setattr(self.ctx, f, getattr(self.ctx, f) + getattr(ag.ctx, f))      # plan 61: the tiers' counters are the root's
        mark = {id(c): dataclasses.replace(c, qid=-1) for c in ag.state.claims}      # (the question ids are the tier's own: the claim answers no question of tier 0)
        if ag.best is not None and id(ag.best[2]) not in mark:
            mark[id(ag.best[2])] = dataclasses.replace(ag.best[2], qid=-1)
        self.state.claims += list(mark.values())
        self.state.gaps += [f"[tier {k}] {g}" for g in ag.state.gaps]
        if ag.best is not None and (owner.best is None or _wider(owner, ag)):
            self.best = (ag.best[0], ag.best[1], mark[id(ag.best[2])])
            self.asked_at_claim = 0
            return True
        return False

    def _next_agenda(self, ag: "Agenda", tier: int, left: float, st: dict, label) -> "Agenda":
        """plan 59 / 65 / 66: the next root agenda after `ag` (a wider tier, or a resume pass at the same tier), on a FRESH context with what carries: the route costs and measured seconds, their
        agendas, the rate, the deferred routes, and since 66.1 / 66.2 the proved claims of the root's questions and the proved replies of every route."""
        k = tier
        nxt = Agenda(State(fen=self.state.fen, us=self.state.us, pieces=list(self.state.pieces)), memory=self.memory, leaf_all=self.leaf_all, seconds=left, tier=k)
        nxt.ctx.route_costs = ag.ctx.route_costs
        nxt.ctx.need = (st["line"] - st["balance"]) if DEFER and st is not None else None      # plan 68
        nxt.ctx.exempt = frozenset(ag.ctx.deferred)              # 65.1: the routes this tier deferred are read in the next one, not deferred again
        nxt.ctx.route_secs = ag.ctx.route_secs                  # 64.3: the measured cost of the routes read so far (the judge's price for a route read before)
        if CARRY:                                                # plan 66.3: a route CUT after t needed more than t only while the next read started from nothing; with its proved replies carried (66.2) the
            nxt.ctx.route_secs = self._carried_secs(ag.ctx.route_secs)      # resumed read does not redo them, so t is no longer a lower bound of what is left (an estimate is never a bound): it is priced as a route not yet read
        nxt.ctx.theirs, nxt.ctx.their_ags = ag.ctx.theirs, ag.ctx.their_ags      # 64.0 (speed only): THEIR agenda at a position is a function of the position, not of the tier: kept across the tiers
        nxt.ctx.rate_t, nxt.ctx.rate_n = ag.ctx.judge_t + ag.ctx.rate_t, ag.ctx.judge_n + ag.ctx.rate_n      # plan 61 C2: seconds per node is measured in the RUN so far
        if CARRY:                                                # plan 66.1: the proved claims of the root's questions carry (a refutation does not: kills and GAPs are read again)
            carried = dict(ag.ctx.carry_q)
            for key, d in ag._carried_out().items():
                old = carried.get(key)
                if old is None or (d["net"], d["second"]) > (old["net"], old["second"]):
                    carried[key] = dict(d, tier=label)
                elif old is not None:
                    carried[key] = dict(old, dep=old["dep"] or d["dep"])
            nxt.ctx.carry_q = carried
            nxt.ctx.carry_status = dict(st)
            proofs = dict(ag.ctx.carry_r)                        # plan 66.2: the replies proved in every earlier tier (the larger proof of a reply wins)
            for kk, d in ag.ctx.proofs.items():
                if kk not in proofs or d["g"] > proofs[kk]["g"]:
                    proofs[kk] = d
            nxt.ctx.carry_r = proofs
        if RESUME:                                               # plan 66.4: the root routes this read stopped by the chain are read with one more level of the allowance, ONLY while their line is still forcing
            seen = set()
            for it in reversed(ag.ctx.chain_gaps):
                kk = (it["key"], it["uci"])
                if kk in seen:
                    continue
                seen.add(kk)
                ok, why = ag._still_forcing(it["line"])
                if ok:
                    nxt.ctx.resume_ext[kk] = dict(ext=ag.ctx.resume_ext.get(kk, {}).get("ext", 0) + 1, moves=len(it["line"]) or 1)
                else:
                    line = f"RESUME: not resumed {it['route'].san}: the line went quiet ({why}; {it['gap']})"
                    if line not in self.lines:
                        self.lines.append(line)
        return nxt

    def _resume_chain_stops(self, ag: "Agenda", total: float, t0: float, owner: "Agenda", decided):
        """plan 66.4 RESUME A CHAIN-STOPPED LINE WHILE IT IS STILL FORCING. With time left after the tiers, the ROOT routes the last read ended as chain-stop GAPs (the clock, the tempo arithmetic, a second
        check of theirs: the GAP facts of the background queue that belong to the root) are read again, in price order (the judge's first-level cost x the measured rate), in a RESUME PASS at the widest tier
        read: a fresh context with every proof carried (66.1 / 66.2) and the chain allowance one level longer per pass -- applied at a stop ONLY while the line to it is still forcing (every move of ours named a
        threat and every reply of theirs parried it: Lean `Ideas.Forcing.stillForcing`, `quiet_breaks`). A line that went quiet is not resumed and says why. The passes end at a natural end: no forcing
        chain stop is left, a claim keeps the status, or the time. -> (owner, decided)."""
        prev, p = ag, 0
        st = self.status_info
        while True:
            left = total - (time.monotonic() - t0)
            items, seen = [], set()
            for it in reversed(prev.ctx.chain_gaps):
                if (it["key"], it["uci"]) not in seen:
                    seen.add((it["key"], it["uci"]))
                    items.append(it)
            if not items:
                break
            if left <= 0.05:
                self.lines.append(f"RESUME: {len(items)} chain-stopped root route(s) left, no time to resume them")
                break
            spn = prev._spn()
            go = []
            for it in items:
                ok, why = prev._still_forcing(it["line"])
                if not ok:
                    self.lines.append(f"RESUME: not resumed {it['route'].san}: the line went quiet ({why}; {it['gap']})")
                    continue
                cost = prev._price(it["route"].move, it["stake"], JUDGE_PRICE_CAP)[0]
                go.append((cost * spn if spn else float(cost), cost, it))
            if not go:
                break
            go.sort(key=lambda t: (t[0], t[2]["uci"]))
            p += 1
            ra = self._next_agenda(prev, prev.ctx.tier, left, st, f"resume {p - 1}" if p > 1 else prev.ctx.tier)
            ra.ctx.extend = p
            ra.ctx.held = None
            want: dict = {}
            for price, cost, it in go:
                want.setdefault(it["key"], []).append(it["uci"])
                moves = sum(1 for _ in it["line"]) or 1
                self.lines.append(f"RESUME pass {p}: resumed {it['route'].san} (chain stop at move {moves}; every reply a parry; priced {cost}" + (f" x {spn:.4f} = {price:.2f} s" if spn else "") + f"; {left:.1f} s left; the allowance +{p})")
            ra.ctx.resume_set = want
            ra.lines = _TierLines(self.lines, f"[resume {p}] ")
            t1 = time.monotonic()
            ra._ask_once()
            secs = time.monotonic() - t1
            if self._fold_tier(ra, f"resume {p}", owner):
                owner, decided = ra, f"resume {p}"
            self.ctx.resume_passes = p
            overall = self.best
            kept = overall is not None and (overall[0] >= BD.MATE_VALUE or self.keeps(st, overall)[0])
            self.tier_info.append(dict(tier=f"resume {p}", secs=round(secs, 2), asked=ra.asked + ra.ctx.asked, kept=kept, hit=bool(ra.ctx.hit), claims=len(ra.state.claims),
                                       best=None if ra.best is None else (ra.best[2].what, ra.best[0])))
            self.lines.append(f"RESUME pass {p}: " + ("no claim" if ra.best is None else f"best claim {ra.best[2].what} {ra.best[0]:+d}") + (": the status is kept" if kept else ""))
            if kept or self.ctx.abort:
                break
            prev = ra
        return owner, decided

    @staticmethod
    def _refuted(why: str) -> bool:
        """plan 65.3: does a route's kill PROVE a loss? A material kill reads '... (-n)' (a reply of theirs after which we are n down with nothing to win back), or the route walks into a mate / stalemate. A kill of
        a question's own goal (the promotion is not claimed, the blocked defect stands) or one that ends level ('(+0)') proves no loss: that route is not refuted."""
        m = re.search(r"\((-\d+)\)\s*$", why)
        return bool(m) or "they mate us" in why or "stalemate" in why or "no parry" in why

    def _record_roots(self, routes: list, results: dict) -> None:
        """plan 65.3: what the ROOT's chase learned of each route: REFUTED (a kill that proves a loss: final), or UNFINISHED (a GAP, or a kill that proves no loss: the replies read so far are `gains`). The decision
        reads this when no claim exists at all. Inner agendas record nothing."""
        if self.chase_depth != 0 or self.ctx is None:
            return
        for i, r in enumerate(routes):
            res = results.get(i)
            u = r.move.uci()
            if res is None or self.root_reads.get(u, {}).get("status") == "killed":
                continue
            gains = list(res.get("gains") or [])
            if res.get("killed") and self._refuted(res["killed"]):
                self.root_reads[u] = dict(status="killed", san=r.san, why=res["killed"], gains=gains)
            elif res.get("killed") or res.get("gap"):
                why = res.get("gap") or f"not refuted: {res['killed']}"
                old = self.root_reads.get(u)
                if old is None or len(gains) >= len(old["gains"]):
                    self.root_reads[u] = dict(status="gap", san=r.san, gap=why, gains=gains)

    def _unfinished_claim(self) -> "Claim | None":
        """plan 65.3 THE UNPROVED DECISION: no claim of any kind exists. An UNFINISHED route (a GAP: nothing is proved against it) is preferred to a refuted one (a KILL is a proved loss). Among the GAPs: the best
        partial floor first, then the judge's price (the cheaper first). The partial floor is the least gain over the replies read so far (the route body reads them in a fixed order, so the replies read are the first
        `n` of that order); a reply not read yet counts by its static material change (what the route would cost with no further reading), so a route whose one reply read is a mate does not outrank a route read
        to the end. -> a Claim marked `proved=False`, word 'unfinished line', or None."""
        ctx = self.ctx
        gaps = [(u, d) for u, d in self.root_reads.items() if d["status"] == "gap"]
        if not gaps:
            return None
        b = self.board
        P0 = AC.Pos(b.fen())
        rows = []
        for u, d in gaps:
            P1 = P0.push(u)
            shift = P1.balance(self.me) - P0.balance(self.me)
            reps = [(shift + dbal, uc, chk, cap, rec) for uc, dbal, chk, cap, rec, _m1 in P1.replies()]
            reps.sort(key=lambda t: (0 if t[4] else 1 if t[2] else 2 if t[3] else 3, t[0], t[1]))       # (the order `_chase_route_body` reads them in)
            n = len(d["gains"])
            read = min(d["gains"]) if d["gains"] else None
            rest = min((t[0] for t in reps[n:]), default=None)
            floor = min(x for x in (read, rest) if x is not None) if (read is not None or rest is not None) else None
            rows.append(((floor is None, -(floor if floor is not None else 0)), u, d, floor, read, rest, n, len(reps)))
        top = min(r[0] for r in rows)
        tied = [r for r in rows if r[0] == top]                       # the best floor; the judge's price (first-level cost, `_price`: priced as far as the cheapest so far) orders the tie
        best, limit = None, 1 << 12
        for r in sorted(tied, key=lambda t: t[1]):
            cost, _kids, lower = self._price(chess.Move.from_uci(r[1]), 1, limit)
            if best is None or cost < best[0]:
                best, limit = (cost, r), cost
        _key, u, d, floor, read, rest, n, total = best[1]
        refuted = [x["san"] for x in self.root_reads.values() if x["status"] == "killed"][:3]
        label = f"unfinished: {n} of {total} replies read"
        st = getattr(self, "status_info", None) or self.status()
        if floor is not None and floor < -st["threat"]:                   # 2026-10-22 (Kav): an unfinished line is played only if its read part beats the pass: it may lose no more than their threat does
            self.lines.append(f"no proved claim: the best unfinished line {d['san']} (floor {floor:+d}, {label}) loses more than their threat ({st['threat']}) on the replies read: no decision"
                              + f" ({len(gaps)} unfinished line(s) compared; " + (", ".join(refuted) + " refuted)" if refuted else "nothing is refuted)"))
            return None
        self.lines.append(f"no proved claim: {d['san']} is played as the best unfinished line (floor " + (f"{floor:+d}" if floor is not None else "unknown") + f", {label}"
                          + (f"; least gain read {read:+d}" if read is not None else "") + (f", the {total - n} unread by their static balance {rest:+d}" if rest is not None else "") + "; "
                          + (", ".join(refuted) + (" is" if len(refuted) == 1 else " are") + " refuted" if refuted else "nothing is refuted") + f"; {len(gaps)} unfinished line(s) compared, {len(tied)} tied at the floor, ordered by the judge's price {best[0]})")
        return Claim("win", d["san"], floor if floor is not None else 0, "unfinished line", f"{d['san']} (floor " + (f"{floor:+d}" if floor is not None else "unknown") + f", {label}; {d['gap']})", move=u, goal="win", proved=False, confidence=0.0, estimator="", word="unfinished line")

    def _piece_attacked(self) -> bool:
        """Is a piece of ours (not the king) attacked by the other side on this board?"""
        b = self.board
        return any(pc.color == self.me and pc.piece_type != chess.KING and b.is_attacked_by(not self.me, sq) for sq, pc in b.piece_map().items())

    def _deep_pass(self) -> tuple:
        """plan 65.4: THEIR PASS read with the proved DEEP delayed balance (Lean `Ideas.TreeStep.tail_four_alive`, family delayed_balance_deep): a move of theirs on the pass board, then whatever we reply, a move of
        theirs, whatever we reply, a move of theirs, whatever we reply: the minimax tail at the leaves against the balance now. -> (the best value T - B > 0, or 0; their move SAN or None; candidates read). Asked
        once; a candidate is not started after the final deadline (the candidates read so far stand)."""
        if self._pass_deep is not None:
            return self._pass_deep
        import contextlib
        import io
        p = G.with_turn(self.board, not self.me)
        their = "w" if p.turn else "b"
        cands = [[pr] for pr in BD.generate(p)]
        best, done = None, 0
        fin = self.ctx.final if self.ctx is not None else None
        with contextlib.redirect_stdout(io.StringIO()):
            for c in cands:
                if fin is not None and time.monotonic() >= fin:
                    break
                done += 1
                got = PA.best_bind(1, p.fen(), their, [c], "delayed_balance_deep")
                if got is not None and got[2] > got[1] and (best is None or got[2] - got[1] > best[0]):
                    best = (got[2] - got[1], got[0])
        self._pass_deep = (best[0], best[1], done) if best else (0, None, done)
        return self._pass_deep

    def _affordable(self, ag: "Agenda", k: int, left: float, t0: float) -> tuple:

        """plan 65.2 (Lean `Ideas.Forcing.none_affordable_of_cheapest`): is any OPEN route priced within the time left? The open routes after tier k: the root routes that ended tier k as GAPs (cut by the judge, the
        competition, the fuse or the clock; a route cut after t seconds needs more than t) and the routes tier k+1's wider vocabulary ADDS at the root of each question tier k chased (forcing with tier k+1's flags,
        not with tier k's). A route that tier k read to its end (a claim, a kill) is not open. The price of a route is the judge's (`_route_need`: first-level cost x the measured seconds per node, or its measured
        seconds). -> (True, '') if one fits (or none can be priced: no rate yet, nothing to say), else (False, the TIERS line). Nothing open at all, with a vocabulary still to come, is not 'nothing affordable':
        the wider vocabulary re-reads the closed routes (a kill under a narrower move set is not a kill under a wider one)."""
        ctx = ag.ctx
        spn = ag._spn()
        if spn is None or spn <= 0:
            return True, ""
        limit = int(left / spn) + 1
        cands = []
        for mkey, (cost, san) in ctx.root_gaps.items():
            seen = ctx.route_secs.get(mkey)
            try:
                san = self.board.san(chess.Move.from_uci(san)) if len(san) in (4, 5) and san[0] in "abcdefgh" and san[1] in "12345678" else san
            except (ValueError, AssertionError):
                pass
            cands.append((self._route_need(cost, seen, spn), cost, san, k))
        b = self.board
        dl, ns = DELAYED_THREAT, NO_SAFE_MOVE
        for (pk, stake, kk), news in list(ctx.competitors.items()):
            if kk != k:
                continue
            for m, san, _why in news:
                if R.forcing_why(b, m, stake, self.us, mate_threats=True, mate2=k >= 2, delayed=dl and k >= 1, nosafe=ns and k >= 1):
                    continue                                         # forcing in tier k already: not new
                cost, _kids, lower = ag._price(m, stake, limit)
                cands.append((self._route_need(cost, ctx.route_secs.get((pk, m.uci(), stake)), spn), cost, san, k + 1))
        if not cands:
            return True, ""
        need, cost, san, tier = min(cands)
        if need <= left:
            return True, ""
        return False, f"TIERS: stopped at {time.monotonic() - t0:.1f} s: the cheapest unread route (tier {tier}, {san}) is priced {cost} x {spn:.4f} = {need:.1f} s > the {left:.1f} s left"

    @staticmethod
    def _carried_secs(route_secs: dict) -> dict:
        """plan 66.3: the measured seconds the next tier may price a route by when proofs carry: a FINISHED read (its seconds; carrying only removes work); never a CUT one's 'more than t'."""
        return {k: v for k, v in route_secs.items() if v[1]}

    def _open_line(self, ag: "Agenda", at: float) -> str:
        """plan 66.3: the root routes still OPEN (GAPs) when the widening ended without the stop of 65.2 (the deadline, or the last tier): how many, and the cheapest by the judge's price. The trace says why the
        decision was not made earlier: open work that was affordable all along, not a re-read."""
        ctx = ag.ctx
        spn = ag._spn()
        rows = []
        for mkey, (cost, san) in ctx.root_gaps.items():
            try:
                san = self.board.san(chess.Move.from_uci(san)) if len(san) in (4, 5) and san[0] in "abcdefgh" and san[1] in "12345678" else san
            except (ValueError, AssertionError):
                pass
            rows.append(((self._route_need(cost, ctx.route_secs.get(mkey), spn) if spn else None), cost, san))
        if not rows:
            return f"TIERS: ended at {at:.1f} s with no root route open (every route read to a claim or a kill)"
        rows.sort(key=lambda r: (r[0] is None, r[0] or 0, r[1]))
        p, c, san = rows[0]
        return f"TIERS: ended at {at:.1f} s with {len(rows)} root route(s) still open (GAPs); the cheapest, {san}, priced {c}" + (f" x {spn:.4f} = {p:.2f} s" if p is not None else " (no rate)") + ": affordable work was left, the stop of 65.2 did not apply"

    def _ask_finish(self) -> None:
        """What `ask` adds to the trace once the loop has ended (the root agenda only); the scheduler's node task calls it too."""
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.landings.asked:        # 52.18 (1): the landing table and what it knows about the deciding piece
            tb = self.ctx.landings
            self.lines.append(f"landing table: {tb.asked} landings asked, {tb.served} served from the table" + ("; e.g." if self.ctx.landing_lines else ""))
            self.lines += ["  " + x for x in self.ctx.landing_lines]
        if self.ctx is not None and self.chase_depth == 0 and self.best is not None and self.best[2].move:
            mv = chess.Move.from_uci(self.best[2].move)
            pc = self.board.piece_at(mv.from_square)
            if pc is not None:
                known = [l for l in self.ctx.landings.known_about(mv.from_square) + self.ctx.landings.known_about(mv.to_square) if l.frm == mv.from_square or (l.to == mv.to_square and l.sym == pc.symbol().lower())]
                seen = []
                for l in known:
                    if l.key() not in seen:
                        seen.append(l.key())
                        if len(seen) <= 5:
                            self.lines.append(f"known about the {pc.symbol().upper()} on {G.name(mv.from_square)}: {l.text()}")
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.finish_lines:         # 52.21 (3)
            self.lines += self.ctx.finish_lines
            self.ctx.finish_lines = []
        if self.ctx is not None and self.chase_depth == 0:
            self.lines += self.facts_lines()
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.final is not None:       # 52.17 (5)
            used = time.monotonic() - self.ctx.t0
            self.ctx.time_outs += 1 if self.ctx.hit else 0
            self.lines.append(f"budget: {self.ctx.seconds:g} s, used {used:.2f} s, {self.asked + self.ctx.asked} questions" + (" -- OUT OF TIME: the open routes are GAPs, the best bound claim decides" if self.ctx.hit else ""))
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.static_lines:       # 52.14: the static leaves read from their side, and the parries
            self.lines.append(f"static leaves read from their side, threats asked as save roots ({len(self.ctx.static_lines)}):")
            self.lines += ["  " + x for x in self.ctx.static_lines]
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.open_lines:       # 52.22 (1): the replies the exchange did not close because the goal was not met
            self.lines.append(f"replies not closed by rule, the goal not met ({self.ctx.unmet}):")
            self.lines += ["  " + x for x in self.ctx.open_lines[:12]]
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.rows:       # 52.13 (4): the chase log of this root, appended once (per item)
            import json
            from catspace.io import paths
            import os
            data = "".join(json.dumps(row) + "\n" for row in self.ctx.rows).encode()      # ONE write on an O_APPEND descriptor: puzzles run in parallel processes (2026-10-20)
            fd = os.open(paths.experiment("chase_log.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                while data:
                    data = data[os.write(fd, data):]
            finally:
                os.close(fd)
            self.ctx.logged = len(self.ctx.rows)
            self.ctx.rows = []

    def _ask_loop(self) -> None:
        while True:
            if not self._threats_done and not (self.best and self.best[0] >= R.STAKE_MATE) and (not self.heap or self.heap[0][4].stake < R.STAKE_MATE):
                self._threats_done = True                        # a mate of ours ends the game, so their threats are asked only once no mate root is left to ask
                self._threat_roots(self._hang)
            if not self.heap:
                break
            nstake = self.heap[0][4].stake
            if self.best and (self.best[0] > nstake or self.best[0] >= BD.MATE_VALUE):      # 53 (3): STRICTLY above the next stake (a question of equal stake may still claim as much with a better shape; the order asked must not decide), a mate ends everything
                self.lines.append(f"STOP (a): the best claim ({self.best[2].what}, {self.best[0]}) is above the stake of every open question (next: {nstake})")
                return
            if self.asked >= self.budget:
                left = [t[4] for t in self.heap]
                self.lines.append(f"STOP (c): budget {self.budget} questions asked; {len(left)} open questions become gaps")
                self.state.gaps += [f"budget: {q.text} (stake {q.stake}) not asked" for q in left]
                return
            if self.chase_depth and self.ctx.out_of_time() and not (self.late and self.ctx.finishing(self.frame)):       # 52.17 (5): an inner agenda that meets the deadline stops where it is (its answer is cut, never memoised, never read as "no claim"); 52.21 (3): the ones opened by the route in hand read on
                self.ctx.budget_hits += 1
                self.lines.append("STOP (t): out of time")
                return
            plain = min(self.heap, key=lambda t: (t[0], t[2], t[3]))                # the stake-only order
            if PRIOR and not self.chase_depth and QP.load() is not None:          # plan 53 (2): THE PRIOR ORDERS the root agenda's pool: the open question with the highest score is asked next
                scored = [(QP.score(QP.features(t[4], self.board, self.us, self.chase_depth)), -t[0], -t[3], t) for t in self.heap]        # (it orders, it never decides: the stops stay by stakes and claims)
                best = max(scored)[3]
                self.heap.remove(best)
                heapq.heapify(self.heap)
                q = best[4]
                if plain[4] is not q:
                    self.lines.append(f"prior: Q{q.id} asked first (score {max(scored)[0]:+.2f}; the stake order would ask Q{plain[4].id})")
                    self.prior_moves += 1
            else:
                _, _, _, _, q = heapq.heappop(self.heap)
                if plain[4] is not q:
                    sc, rec = self._recall_score(q)
                    self.lines.append(f"memory: Q{q.id} moved up by {sc} (seen {rec['id']}: {rec['goal']} by {rec['lemma']})")
            if DEFER and self._root and self.chase_depth == 0 and self.ctx is not None and self.ctx.need is not None and self.ctx.need > 0 and q.goal != "save" and q.stake < self.ctx.need:
                self.ctx.deferred_qs.append(q)                     # plan 68: a claim below the need keeps no status (Ideas.Forcing.irrelevant_keeps_nothing): asked after the tiers' relevant questions
                self.lines.append(f"deferred: Q{q.id} {q.text} (stake {q.stake} < the status's need {self.ctx.need}): asked after the tiers' relevant questions")
                continue
            self.order.append(q)                                   # plan 53 (1): the order asked (the features are read at log time: the root board does not change while its agenda runs)
            if self.ctx is not None and self.ctx.resume_set is not None and self.chase_depth == 0 and self._root and self._carry_key(q) not in self.ctx.resume_set:
                if self.ctx.carry_q.get(self._carry_key(q)) is not None:      # plan 66.4: a resume pass reads only the chain-stopped routes; every other root question keeps its carried claim
                    self._carry_question(q, settle=True)
                continue
            if self.ctx is not None and self.ctx.carry_q and self.chase_depth == 0 and self._root and self._carry_question(q):
                continue
            if self._inherit(q):
                continue
            if self.chase_depth:                                 # an inner agenda pays the shared budget as it asks (a chain of checks meets it on the way down)
                self.ctx.left -= 1
                before = self.asked
            self._cur, self._qdeps, n0 = q, [], len(self.state.claims)
            self._floor = getattr(q, "_floor", None) if self.chase_depth == 0 else None      # plan 66.1: the carried floor of a re-asked root question
            self._ask(q)
            self._floor = None
            if self.ctx is not None:
                self._file(q, self.state.claims[n0:])
            if self.chase_depth:
                self.ctx.left -= max(self.asked - before - 1, 0)
        self.lines.append("STOP (b): every question is closed")

    @staticmethod
    def _carry_key(q: R.Question) -> tuple:
        """plan 66.1: a ROOT question across the tiers of one position: its goal, its text and its stake (the root questions of every tier are generated the same way from the same board)."""
        return (q.goal, q.text, q.stake)

    def _carried_out(self) -> dict:
        """plan 66.1: what this tier's root PROVED, question by question: (goal, text, stake) -> {net, second, claim, dep}. `dep`: the answer read the chase (the tier's vocabulary); a question answered
        without it (a schema binding, the smallest loss in check with no chase, a race word, the table) is answered the same way in every tier. Estimated and unfinished claims never carry."""
        out = {}
        for q in self.all:
            hit = self._q_best.get(q.id)
            if hit is None or q.status != "verified":
                continue
            out[self._carry_key(q)] = dict(net=hit[0], second=hit[1], claim=hit[2], dep=bool(getattr(q, "_tier_dep", False)))
        return out

    def _carry_question(self, q: R.Question, settle: bool = False) -> bool:
        """plan 66.1 PROVED ROOT CLAIMS CARRY (Lean `Ideas.Forcing.claim_mono`: a claim forced with our route set R is forced with any R' >= R; the wider tier widens OUR vocabulary only). The question's
        claim from an earlier tier is SERVED ("carried from tier k: ..."). It is not asked again when the claim keeps the status, or when no chase made it (a schema binding reads no vocabulary of the tier:
        the wider tier would answer it the same way). Otherwise it is RE-ASKED for a better claim, with the carried claim as its FLOOR (the best already holds it; 66.2: only what could beat it is re-read).
        -> True when the question is settled (not asked again)."""
        hit = self.ctx.carry_q.get(self._carry_key(q))
        if hit is None:
            return False
        old = hit["claim"]
        cl = dataclasses.replace(old, qid=q.id, line=f"carried from tier {hit['tier']}: {old.line}")
        net = hit["net"]
        self.state.claims.append(cl)
        q.status = "verified"
        self.ctx.carried_q += 1
        if self._better(net, hit["second"], cl):
            self.best = (net, hit["second"], cl)
            self.asked_at_claim = self.asked
        st = self.ctx.carry_status
        keeps = net >= BD.MATE_VALUE or (st is not None and st["balance"] + net >= st["line"])
        head = f"Q{q.id} {q.text}: carried from tier {hit['tier']}: {cl.what} {net:+d} (proved, lemma {cl.lemma})"
        if settle:                                                 # (a resume pass: the claim is served, the question is not read)
            self.ctx.carried_q_kept += 1
            self.lines.append(head + "; not resumed: served")
            return True
        if keeps or not hit["dep"]:
            self.ctx.carried_q_kept += 1
            self.lines.append(head + ("; it keeps the status: not asked again" if keeps else "; no chase made it (the answer reads no vocabulary of a tier): not asked again"))
            return True
        self.ctx.carried_q_reasked += 1
        q._floor = net
        self.lines.append(head + f"; it does not keep the status: re-asked for a better claim above the floor {net:+d}")
        return False

    def _inherit(self, q: R.Question) -> bool:
        """52.12 (3), 52.13 (1): THE DIFF DECIDES WHAT IS ASKED. An inner agenda asks (a) the question that sent it there (it is not settled in the outer agenda, so it is
        not in `inherit`), (b) every outer question whose reads meet the diff of the route and the reply, (c) every question NEW on this board. The rest is INHERITED
        with its status: a kill as it was, a claim re-stamped on this board (the move is legal here) with the line 'inherited: nothing changed for ...'."""
        key = R.question_key(q)
        if not self.inherit and self.ctx is not None and key is not None:
            return self._serve(q, key)                           # 52.19 (1): the fact store replaces the parent-only walk (the dict form below is what the tests of 52.12 / 52.13 pass in)
        hit = self.inherit.get(key) if key is not None else None
        if hit is None:
            if key is not None and self.inherit:
                self.ctx.fresh += 1
            return False
        if hit["reads"] & hit["diff"]:
            self.ctx.reopened += 1
            return False
        b = self.board
        sq = q.pattern.get("at") if q.goal == "mate" else q.pattern.get("target")
        pc = b.piece_at(sq) if sq is not None else None
        what = f"the {pc.symbol().upper() if pc else 'piece'} on {G.name(sq) if sq is not None else '?'}"
        if hit["kind"] == "claim":
            old = hit["claim"]
            mv = chess.Move.from_uci(old.move)
            if mv not in b.legal_moves:
                self.ctx.reopened += 1
                return False
            cl = dataclasses.replace(old, what=b.san(mv), line=f"inherited: nothing changed for {what} ({old.line})")
            self.state.claims.append(cl)
            q.status, q.claimed, q.reads = "verified", (hit["net"], cl, hit["reads"]), hit["reads"]
            self.ctx.inherited_claims += 1
            self.lines.append(f"Q{q.id} {q.text}: inherited: nothing changed for {what}: CLAIM {cl.kind} {cl.what} (value {cl.value}, net {hit['net']}), lemma {cl.lemma}")
            if self._better(hit["net"], 0, cl):
                self.best = (hit["net"], 0, cl)
                self.asked_at_claim = self.asked
        else:
            q.status, q.reason, q.reads = "killed", "inherited: " + hit["why"], hit["reads"]
            self.kills.append((q.id, q.reason))
            self.lines.append(f"Q{q.id} {q.text}: inherited: nothing changed for {what} ({hit['why']})")
        self.inherited += 1
        self.ctx.inherited += 1
        return True

    def _can_chase(self, q: R.Question) -> bool:
        """Whether this agenda would chase the question if it came to be killed (the static guards of `_chase`)."""
        return self.ctx is not None and q.goal in ("mate", "win", "promote") and self.min_stake < BD.MATE_VALUE and not (self.chase_depth >= 2 and q.goal != "mate")

    def _power(self, q: R.Question) -> int:
        """How hard this agenda chases the question if it comes to be killed: 0 not at all, 1 as an inner agenda (checks, captures, mate threats), 2 as the root (every forcing move, the stopper routes). A kill served to
        an agenda must have been chased at least as hard as this one would chase."""
        return 0 if not self._can_chase(q) else (2 if self.chase_depth == 0 else 1)

    def _fkey(self, q: R.Question):
        key = R.question_key(q)
        return None if key is None else (key, self.board.is_check())

    def _serve(self, q: R.Question, key) -> bool:
        """52.19 (1): THE STORE IS CONSULTED BEFORE A QUESTION IS ASKED. A fresh fact of the same question whose read cells on this board are what they were (a signature, not a FEN) is served:
        a kill as it was, a claim re-stamped on this board (its move is legal here). Siblings, transpositions and later root agendas share facts; the parent-only walk is a special case."""
        fs, b = self.ctx.facts, self.board
        f, cand, stale_miss = fs.lookup(self._fkey(q), self.me, self._cell, power=self._power(q))
        if f is not None and f.kind == "claim":
            mv = chess.Move.from_uci(f.payload["claim"].move)
            if mv not in b.legal_moves:
                f, stale_miss = None, False
        if f is None:
            if stale_miss:
                fs.recomputed += 1                               # a stale fact of this very question no longer matches and the live question needs it: recomputed NOW (foreground), the rest stays queued
            if self.chase_depth:
                if cand:
                    self.ctx.reopened += 1
                else:
                    self.ctx.fresh += 1
            return False
        sq = q.pattern.get("at") if q.goal == "mate" else q.pattern.get("target")
        pc = b.piece_at(sq) if sq is not None else None
        what = f"the {pc.symbol().upper() if pc else 'piece'} on {G.name(sq) if sq is not None else '?'}"
        src = f"served from the store (fact #{f.id}, derived under {f.under})"
        if f.kind == "claim":
            old = f.payload["claim"]
            self.ctx.chase_served += 1 if f.payload.get("chase") else 0
            cl = dataclasses.replace(old, what=b.san(chess.Move.from_uci(old.move)), line=f"inherited: nothing changed for {what} ({old.line})")
            self.state.claims.append(cl)
            self.claim_fid[id(cl)] = f.id
            q.status, q.claimed, q.reads = "verified", (f.payload["net"], cl, f.reads), f.reads
            self.ctx.inherited_claims += 1
            self.lines.append(f"Q{q.id} {q.text}: inherited: nothing changed for {what}: CLAIM {cl.kind} {cl.what} (value {cl.value}, net {f.payload['net']}), lemma {cl.lemma} -- {src}")
            if self._better(f.payload["net"], 0, cl):
                self.best = (f.payload["net"], 0, cl)
                self.asked_at_claim = self.asked
        else:
            q.status, q.reason, q.reads = "killed", "inherited: " + f.payload.get("why", ""), f.reads
            self.kills.append((q.id, q.reason))
            self.lines.append(f"Q{q.id} {q.text}: inherited: nothing changed for {what} ({f.payload.get('why', '')}) -- {src}")
        self.fact_of[q.id] = f.id
        self.premises.append(f.id)
        self.inherited += 1
        self.ctx.inherited += 1
        return True

    def _file(self, q: R.Question, new_claims: list) -> None:
        """52.19 (1): what the question just derived is FILED as a fact: its read set and signature on this board, the facts it was derived from (the route that decided a chase claim, every route
        for a chase kill, the inner answers), the stake and the position it was asked on."""
        if q.status not in ("killed", "verified", "gap"):      # 52.23 (9): a chain-cut kill is contextual, not filed
            return
        key = self._fkey(q)
        fs, kind = self.ctx.facts, {"killed": "kill", "verified": "claim", "gap": "gap"}[q.status]
        reads, value, payload, portable = q.reads, 0, {}, key is not None
        claim = None
        if kind == "claim":
            if q.claimed is not None:
                value, claim, reads = q.claimed[0], q.claimed[1], q.claimed[2]
                payload = dict(claim=claim, net=q.claimed[0])
                portable = portable and not claim.lemma.startswith(("chase", "delayed", "mate_in"))      # a chase claim reads every reply of the board it was made on: it is not transferable (as in 52.13) -- unless `_chase_reads` below finds it a read set (52.21 (2));
                # 52.23 (10) / 53 (3): a SCHEMA claim reads every reply too (all their moves): served by the target's signature it was unsound (2VDtX: SF -697), and with the move's squares and their king added it still was (OF8nj: bxc3+ valued 6 by a claim from another position where the chase on this board reads 1). Not portable.
            else:
                claim = new_claims[-1] if new_claims else None
                value = claim.value if claim is not None else 0
                portable = False
            if claim is not None and claim.lemma.startswith("chase") and key is not None:      # 52.21 (2): a chase claim is portable when the facts it stands on have read sets
                creads, why = self._chase_reads(q, claim)
                if creads is None:
                    self.lines.append(f"  (chase claim filed for this position only: {why})")
                    portable = False
                else:
                    reads, portable, payload = creads, True, dict(claim=claim, net=value, chase=True)
        elif kind == "kill":
            payload = dict(why=q.reason.removeprefix("inherited: ").split(";")[0][:80])
        text = f"{claim.kind} {claim.what}: {q.text}" if claim is not None else q.text
        sig = FS.signature(self._cell, reads) if reads is not None else ()
        f = fs.add(kind, key if key is not None else ("q", q.goal, q.text, self.pk), self.me, text, value=value, reads=reads, sig=sig, just=self._qdeps, stake=q.stake, depth=self.chase_depth,
                   under=f"Q{q.id} at depth {self.chase_depth}", pk=self.pk, portable=portable, power=(2 if getattr(q, "_reasoned", False) or (kind == "claim" and value >= min(q.stake, BD.MATE_VALUE)) else self._power(q) if getattr(q, "_chased", False) and not (kind == "kill" and getattr(q, "_chain_stopped", False)) else 0), payload=payload)
        # (plan 66.5: a CLAIM at the question's stake is FINAL whatever chase made it -- no chase of any power reads past the stake (52.13 (2c)), and a proof stands under any clock: the chase that found it may
        #  have been cut elsewhere (the irreversible-move clock lets a chain run on until the fuse cuts another route), which weakens what a KILL says, never what a claim at the stake says)
        self.fact_of[q.id] = f.id
        self.premises.append(f.id)
        if claim is not None:
            self.claim_fid[id(claim)] = f.id

    def _resumed_line(self, r: R.Route, res: dict) -> None:
        """plan 66.2: the trace of a ROOT route that served replies from an earlier tier: 'resumed Rg8+: 3 of 5 replies carried from tier 0, 2 read'."""
        if not res.get("carried") or self.chase_depth != 0 or res.get("reused"):
            return
        self.ctx.carry_routes += 1
        n = len(res.get("gains") or [])
        P1 = AC.Pos(self.board.fen()).push(r.move.uci())
        self.lines.append(f"    resumed {r.san}: {res['carried']} of {P1.legal_count()} replies carried from tier {res.get('carried_from', 0)}, {n - res['carried']} read"
                          + (" (then a GAP)" if res.get("gap") else " (then killed)" if res.get("killed") else ""))

    def _chase_reads(self, q: R.Question, claim: Claim) -> tuple:
        """52.21 (2): THE READ SET OF A CHASE CLAIM, from its justifications. The claim stands on the facts of the route that decided it (the inner answers of its replies, their agendas): the union of what
        they read, the question's own reads, the route's squares, and what the replies themselves depend on -- the pieces of both sides after the route move and what they attack (that decides which
        replies exist, which the rules close, and what a mate in one needs), with the lines from both kings (pins). A fact that read its whole position (an agenda there, their agenda with the move
        passed) makes the claim positional: -> (None, why). -> (the squares, None) otherwise."""
        fs, b = self.ctx.facts, self.board
        if self._carried_route:                                  # plan 66.2: replies served from an earlier tier stand on that tier's facts, not this store's: the claim's read set is unknown here
            return None, f"its route served {self._carried_route} replies carried from an earlier tier (their facts are that tier's)"
        reads = set(R.reads_of(q, b))
        for j in self._qdeps:
            f = fs.facts[j]
            fr = f.reads or f.payload.get("reads")
            if not fr:
                return None, f"it stands on fact #{j} ({f.text}), which read its whole position"
            reads |= fr
        m = chess.Move.from_uci(claim.move)
        reads |= {m.from_square, m.to_square}
        reads |= self._extra_reads
        c = b.copy(stack=False)
        c.push(m)
        for sq in chess.SquareSet(c.occupied):
            reads.add(sq)
            reads.update(chess.SquareSet(c.attacks_mask(sq)))
        for colour in (chess.WHITE, chess.BLACK):
            k = c.king(colour)
            if k is not None:
                reads.update(R.slide_region(k, c.occupied))
        return frozenset(reads), None

    def _inherit_for(self, diff: frozenset) -> dict:
        """What an inner agenda inherits from this one: the settled questions (killed, or claimed with their reads) by key, with the diff. A chase claim is not
        inherited (its tail reads every reply of the board it was made on), and neither is a question this agenda has not settled (the one being asked is among them)."""
        out = {}
        for q in self.all:
            key = R.question_key(q)
            if key is None:
                continue
            if q.status == "killed" and q.reads is not None:
                out[key] = dict(kind="kill", reads=q.reads, diff=diff, why=q.reason.removeprefix("inherited: ").split(";")[0][:80])
            elif q.status == "verified" and q.claimed is not None and not q.claimed[1].lemma.startswith("chase"):
                out[key] = dict(kind="claim", reads=q.claimed[2], diff=diff, net=q.claimed[0], claim=q.claimed[1])
        return out

    def _count(self, q: R.Question) -> int:
        return 1 + sum(self._count(f) for f in q.fixes)

    def _ask(self, q: R.Question) -> None:
        b, us = self.board, self.us
        self.asked += 1
        head = f"Q{q.id} {q.text} (stake {q.stake})"
        if q.goal == "save" and "checkers" in q.pattern:
            self._ask_save_king(q, head)
            return
        if q.goal == "promote":
            self._ask_promote(q, head)
            return
        if q.goal == "fork":
            self._ask_fork(q, head)
            return
        if q.goal == "mate" and self._reasoned_kill(q, head):
            return
        if q.goal == "mate" and q.pattern.get("reach"):
            self._ask_reach(q, head)
            return
        if q.goal == "save" and "promoter" in q.pattern:
            self._ask_stop_pawn(q, head)
            return
        if q.goal == "save" and "threat" in q.pattern:
            self._ask_save_threat(q, head)
            return
        if q.goal == "save":
            self._ask_save_piece(q, head)
            return
        if q.pattern.get("checker"):                             # 52.15 (1): the checker taken by any legal capture, read by the schemas as a win root's routes are
            self.lines.append(head + ": the checker is taken by a legal capture (no defect to fix)")
            self._verify(q, self._direct_routes(q), mates_only=False)
            return
        if q.goal == "mate":
            q.defects = self._landing_defects(q)
        else:
            q.defects = R.material_defects(b, us, q.pattern["target"])
        self.lines.append(head + (": " + "; ".join(f"defect {d.kind} ({d.why})" for d in q.defects) if q.defects else ": no defect -- the pattern already stands"))
        if not q.defects:
            routes = self._direct_routes(q)
        else:
            for d in q.defects:
                R.fixes_of(d, q, b, us)
            self.asked += sum(self._count(f) for f in q.fixes)
            for f in q.fixes:
                self._trace_fix(f, 1)
            if not R.tempo_ok(q):
                if self._chase(q):
                    return
                self._kill(q, q.reason, tempo=True)
                return
            self.lines.append(f"  tempo: {len(q.defects)} defect(s), each has a forcing fix")
            routes = self._merge_routes(q)
        self._verify(q, routes, mates_only=q.goal == "mate")

    def _landing_defects(self, q: R.Question) -> list:
        """52.18 (1): a mate root is a LANDING that binds (or lacks a claim by its defects). The defect list is the landing's answer, served from the table when nothing its read set
        touches has changed since it was asked (an inner board, the same landing), else computed and entered. Castling and promotion roots are read directly."""
        p, b = q.pattern, self.board
        frm = p.get("frm")
        if self.ctx is None or p.get("more") or frm is None or b.piece_at(frm) is None or b.piece_at(frm).symbol() != p.get("sym"):
            return R.mate_defects(b, self.us, R.placement_of(q))
        tbl = self.ctx.landings
        key = (self.me, p["sym"].lower(), frm, p["at"])
        hit = tbl.lookup(b, key)
        tbl.asked += 1
        if hit is not None:
            tbl.served += 1
            self.ctx.landing_served += 1
            line = f"landing Q{q.id} on {G.name(p['at'])} (depth {self.chase_depth}): {hit.text().split(': ', 1)[1]}, served from the table"
            self.lines.append(line)
            if len(self.ctx.landing_lines) < 3:
                self.ctx.landing_lines.append(line)
            return list(hit.defect_objs)
        lnd = LD.evaluate(b, self.me, frm, p["at"])
        tbl.put(b, lnd)
        return list(lnd.defect_objs)

    def _ask_fork(self, q: R.Question, head: str) -> None:
        """52.18 (4): the fork root is VERIFIED by the rules reply by reply: after the landing, for every legal reply of theirs the balance we then hold plus the best static exchange of ours
        (the Rust `Pos.exchange`, the mirror of Lean `Ideas.ExchangeLaw`) is read; the claim is the least of them, when at least 1. A reply that saves both is a DEFECT of the fork and is named:
        the capture of the forking piece, a move of one attacked piece that guards or takes the other, a check of theirs. Lean `Ideas.ForkSquare.one_move_touches_one`: any other single
        move touches at most one of the two squares."""
        b, p = self.board, q.pattern
        m = chess.Move.from_uci(p["move"])
        self.lines.append(head)
        P0 = AC.Pos(b.fen())
        P1 = P0.push(p["move"])
        shift = P1.balance(self.me) - P0.balance(self.me)
        worst = None
        c = b.copy(stack=False)
        c.push(m)
        for uci, dbal, chk, cap, rec, mate1 in P1.replies():
            P2 = P1.push(uci)
            if P2.legal_count() == 0:
                net = -R.STAKE_MATE if P2.is_check() else 0
            elif mate1 or P2.mate_in_one():
                net = R.STAKE_MATE
            else:
                ex = 0 if P2.is_check() else P2.exchange()
                if ex > 0 and self.ctx is not None and CLOSE_AFTER_CAPTURE and self._their_after_capture(P2)[0]:
                    ex = 0                                           # 64.0: their free move after our capture(s) has a claim: the exchange is not PROVED to hold, so it is not counted
                net = shift + dbal + ex
            if worst is None or net < worst[0]:
                worst = (net, uci)
        reads = {m.from_square, m.to_square, *(t for t, _w in p["targets"])}
        for t, _w in p["targets"]:
            reads.update(LD.exchange_reads(c, t))
        q.reads = frozenset(reads)
        if worst is None or worst[0] < 1:
            r = chess.Move.from_uci(worst[1]) if worst is not None else None
            how = ""
            if r is not None:
                if r.to_square == m.to_square:
                    how = "captures the forking piece"
                elif c.gives_check(r):
                    how = "a check of theirs"
                elif r.from_square in [t for t, _w in p["targets"]]:
                    how = "a move of one attacked piece that answers the other"
                else:
                    how = "a reply that keeps what it has"
            self._kill(q, f"after {c.san(r) if r is not None else '?'} ({how}) we win {worst[0] if worst else 0}, less than 1: one move answers both threats", tempo=False)
            return
        what = f"{p['san']}"
        claim = Claim("win", what, worst[0], "fork: two threats, one move (Ideas.ForkSquare.one_move_touches_one); every reply read by the exchange rule (Ideas.ExchangeLaw)",
                      f"{what}: " + ("check and " if p["check"] else "") + "attacks " + " and ".join(f"{b.piece_at(t).symbol().upper() if b.piece_at(t) else '?'}{G.name(t)}" for t, _w in p["targets"])
                      + f"; the least over their replies of balance + exchange is {worst[0]} (worst reply {c.san(chess.Move.from_uci(worst[1]))})", move=p["move"])
        self._stamp(claim, q)
        self.state.claims.append(claim)
        q.status = "verified"
        q.claimed = (worst[0], claim, q.reads)
        self.lines.append(f"  -> CLAIM win {what} (value {worst[0]}), lemma {claim.lemma}, line {claim.line}")
        if self._better(worst[0], 0, claim):
            self.best = (worst[0], 0, claim)
            self.asked_at_claim = self.asked

    def _their_passer(self, rook: int) -> "tuple | None":
        """52.17 (1): THEIR passer if we pass: a pawn of theirs that promotes in `n` pushes unless the rook is spent on it -- its promote defects (the Rust words, from its owner's
        side) are none, or stoppers whose only piece is our rook (the king and every other piece of ours cannot reach in time). -> (n, pawn square, the stopper text) of the
        quickest such pawn, or None."""
        b, best = self.board, None
        for pw in chess.SquareSet(b.pawns & b.occupied_co[not self.me]):
            n = G.word(b, ("pushes", G.name(pw)), not self.me)
            if n < 1:
                continue
            defects, _facts = R.promote_defects(b, pw)
            if any(d.kind != "stopper" or set(d.by) - {rook} for d in defects):
                continue
            if best is None or n < best[0]:
                best = (n, pw, "; ".join(d.why for d in defects) or "no stopper of ours in time")
        return best

    def _reasoned_kill(self, q: R.Question, head: str) -> bool:
        """52.17 (1): a mate goal DIES BY REASONING, before any chase: (a) THE WALL (region_kill: the covers it still needs are out of reach of our king's region -- a held
        line is a wall for a king -- and no pawn or other piece can supply them); (b) THE MEASURE THEY HOLD (held_measure: a king and one rook against a king with room, every
        first move answered by a step into the open that keeps the distance or by the capture of the rook, our rook also the only stopper of their passer). Both print as
        "killed by reasoning -- ..." with the read set; a chase of the goal is warranted only when neither fires. -> True when the question was killed."""
        if self.ctx is None:
            return False
        b, us = self.board, self.us
        ksq = b.king(not self.me)
        if ksq is None:
            return False
        defects = q.defects or R.mate_defects(b, us, R.placement_of(q))
        flights = [d.sq for d in defects if d.kind == "flight"]
        frm = q.pattern.get("frm")
        why, reads, which = None, None, None
        if flights and frm is not None:
            k = RG.region_kill(b, self.me, frm, flights)
            if k is not None:
                why, reads, which = k["why"], k["reads"], "the wall"
        if why is None:
            heavy = [s for s in chess.SQUARES if b.piece_at(s) is not None and b.piece_at(s).color == self.me and b.piece_at(s).piece_type not in (chess.KING, chess.PAWN)]
            if len(heavy) == 1 and b.piece_at(heavy[0]).piece_type == chess.ROOK and not b.is_check():
                steps = [chess.Move.from_uci(u) for u in q.pattern["reach"]] if q.pattern.get("reach") else ([chess.Move.from_uci(q.pattern["move"])] if q.pattern.get("move") else [])
                steps = [m for m in steps if R.forcing_why(b, m, q.stake, us, mate_threats=True)]
                passer = self._their_passer(heavy[0]) if steps else None
                if passer is not None:
                    k = RG.held_measure(b, self.me, steps, passer[0])
                    if k is not None:
                        pw = passer[1]
                        bound = [f"our {b.piece_at(g).symbol().upper()} on {G.name(g)} is bound to the {b.piece_at(y).symbol().upper()} on {G.name(y)}" for g, y in RG.bound_pieces(b, self.me) if b.piece_at(g).piece_type == chess.KING][:2]
                        why = k["why"] + f" (their pawn on {G.name(pw)} promotes in {passer[0]}: {passer[2]})" + ("; " + "; ".join(bound) if bound else "")
                        reads, which = k["reads"] | {pw}, "the measure they hold"
        if why is None:
            return False
        q.reads = R.reads_of(q, b) | frozenset(reads)
        self.ctx.reasoned += 1
        q._reasoned = True                                       # a "do not chase" lemma: the kill holds for every asker
        self.lines.append(head + f": killed by reasoning ({which}) -- {why}")
        self._kill(q, f"reasoned: {which}", tempo=False)
        return True

    @staticmethod
    def _merge_routes(q: R.Question) -> list:
        """The routes of a question's fixes, one per move; a move that is also a trap keeps the trap square (52.10: the binder may then read it at depth two)."""
        by, routes = {}, []
        for f in q.fixes:
            for r in R.all_routes(f):
                if r.move not in by:
                    by[r.move] = r
                    routes.append(r)
                else:
                    if r.trap is not None and by[r.move].trap is None:
                        by[r.move].trap = r.trap
                    if r.fork is not None and by[r.move].fork is None:
                        by[r.move].fork = r.fork
        return routes

    def _trace_fix(self, f: R.Question, lvl: int) -> None:
        pad = "  " * lvl
        if f.status == "gap":
            self.lines.append(f"{pad}-> Q{f.id} {f.text}: GAP, {f.reason}")
            self.state.gaps.append(f"Q{f.id} {f.text}: {f.reason}")
        else:
            rs = ", ".join(f"{r.san} ({r.why})" for r in f.routes)
            tail = f"{'routes' if f.text.startswith(('take its squares', 'fork')) else 'forcing'}: {rs}" if rs else "no forcing move" + (f" (quiet fixes: {', '.join(f.unforced[:4])})" if f.unforced else "")
            self.lines.append(f"{pad}-> Q{f.id} {f.text}" + ("" if f.fixes and not f.routes else f": {tail}"))
        for w in f.why:
            self.lines.append(f"{pad}     {w}")
        for g in f.fixes:
            self._trace_fix(g, lvl + 1)

    def _direct_routes(self, q: R.Question) -> list:
        b = self.board
        if q.goal == "mate":
            m = chess.Move.from_uci(q.pattern["move"]) if q.pattern.get("move") else chess.Move(q.pattern["frm"], q.pattern["at"])
            return [R.Route(m, b.san(m), "the placement move")]
        t = q.pattern["target"]
        return [R.Route(m, b.san(m), f"capture {G.worth(b.piece_at(t))}") for m in b.legal_moves if m.to_square == t and not m.promotion]

    # ------------------------------------------------------------------------------------------------------------ verify and claim

    def _verify(self, q: R.Question, routes: list, mates_only: bool) -> None:
        b, us = self.board, self.us
        if not routes:
            if self._chase(q):
                return
            self._kill(q, "no route: the fixes supply no move", tempo=False)
            return
        hits, why = [], []
        for r in routes:
            bind, tried = BD.verify_tried(r.move, b, us, mates_only, trap=r.trap, fork=r.fork)
            asked = self._asked(tried)
            if bind is None:
                self.lines.append(f"  verify {r.san}: nothing binds ({asked})")
                why.append(self._escape_note(r, tried))
                continue
            head = f"  verify {r.san}: binds {bind.schema} ({len(bind.replies)} replies{': ' + ' / '.join(bind.replies[:4]) if len(bind.replies) <= 4 else ''}), value {bind.value}"
            opp, theirs, net = [], None, bind.value
            if bind.value >= BD.MATE_VALUE or not self.side_depth:
                self.lines.append(head + ("" if bind.value >= BD.MATE_VALUE else "; their view: not asked (this is their agenda)"))
            else:
                theirs, _ = self.their_view(r.move)
                if theirs is not None:
                    net = bind.value - theirs.value
                    self.lines.append(head + f"; their view: their {theirs.what} wins {theirs.value} (their claim, lemma {theirs.lemma}; {theirs.line}) -> net {net}")
                else:
                    opp = R.opponent_defects(b, r.move, us)
                    self.lines.append(head + "; their view: no claim of theirs binds" + (f" (fallback label: {'; '.join(opp)})" if opp else ""))
            if (self.side_depth or (self.chase_depth and bind.value >= self.min_stake)) and bind.schema.startswith("delayed_") and bind.schema.endswith("_d1"):         # 64.0: the root families too, and in the chase's inner agendas (their free move comes after our LAST move, which the tail does not read)
                ref = self._leaf_read(bind, r)
                if ref is not None:
                    gap = f"{r.san}: refuted at the leaf: their {ref['claim'].what} wins {ref['claim'].value} (lemma {ref['claim'].lemma}) after {ref['line']}"
                    self.state.gaps.append(f"Q{q.id} {q.text}: {gap}")
                    self.lines.append(f"  {r.san} DOWNGRADED to a gap: {gap}; our claimed gain {bind.value} is not decided by")
                    why.append(f"{r.san}: {gap}")
                    continue
            if theirs is not None and theirs.value >= q.stake and net <= 0:
                why.append(f"{r.san}: their {theirs.what} wins {theirs.value} (lemma {theirs.lemma}) -- the future loses it")
                self.lines.append(f"  {r.san} KILLED as a route: their {theirs.what} wins {theirs.value} (lemma {theirs.lemma}) -- the future gives them a claim of at least our stake {q.stake} and nets {net}")
                continue
            hits.append((bind, opp, r, net, theirs))
        if not hits:
            if self._chase(q):
                return
            self._kill(q, f"verify: no route binds ({', '.join(r.san for r in routes[:6])}{' ...' if len(routes) > 6 else ''})"
                       + ("; " + "; ".join(w for w in why if w and "no mate in one" in w)[:300] if mates_only else "")
                       + ("; " + "; ".join(w for w in why if "the future loses it" in w)[:300]), tempo=False)
            return
        hits.sort(key=lambda h: (-h[3], len(h[1])))               # net value first; the heuristic label only breaks a tie where no claim of theirs bound
        if self._t(NEED_PROPAGATION, 3) and self.chase_depth and hits[0][3] < self.min_stake and self._chase(q):      # 52.23 (7): a bound below what this agenda must reach rescues nothing: the question chases first (zBouR: Rxe3+ bound 1, 6 needed, Qh1+ drives the king onto the skewer line)
            return
        bind, opp, r, net, theirs = hits[0]
        for other, oopp, orr, onet, otheirs in hits[1:]:
            if onet == net:
                if oopp or opp:
                    if len(oopp) > len(opp):
                        self.lines.append(f"  {orr.san} ties at net {onet} but no claim of theirs binds and its future has a defect of theirs (fallback label: {'; '.join(oopp)}): {r.san} is preferred")
            elif otheirs is not None and net > onet:
                self.lines.append(f"  {orr.san} demoted: bound value {other.value} less their {otheirs.what} ({otheirs.value}, lemma {otheirs.lemma}) = net {onet}, against {r.san} net {net}")
        kind = "mate" if bind.value >= BD.MATE_VALUE else ("save" if q.goal == "save" else "win")
        claim = Claim(kind, bind.san, bind.value, bind.theorem, self._line_of(bind), move=bind.assignment["x"] + bind.assignment["y"])
        self._stamp(claim, q)
        self.state.claims.append(claim)
        q.status = "verified"
        self._record_claim(q, net, claim, bind)
        window = " [a gain WITHIN THE WINDOW: the leaf is not read to quiescence -- 0UqOv's queen was trapped two plies beyond it]" if "balance" in str(claim.lemma) else ""
        self.lines.append(f"  -> CLAIM {kind} {claim.what} (value {claim.value}{'' if net == claim.value else f', net {net} of their best claim'}), lemma {claim.lemma}, line {claim.line}{window}")
        key = (net, -len(opp))
        if q.pattern.get("inner_threat"):                        # 52.14: the parry of a threat on a static leaf keeps the gain; it is not a gain of its own (the reply's value stays dbal)
            return
        if self._better(key[0], key[1], claim):
            self.best = (net, -len(opp), claim)
            self.asked_at_claim = self.asked

    def _record_claim(self, q: R.Question, net: int, claim: Claim, bind: "BD.Binding | None") -> None:
        """52.13 (1): a claim's reads = the question's reads plus the squares named in the atoms of its bound schema (our named moves x y x2 y2)."""
        reads = set(q.reads if q.reads is not None else R.reads_of(q, self.board))
        if bind is not None:
            for k in ("x", "y", "x2", "y2"):
                if bind.assignment.get(k):
                    reads.add(chess.parse_square(bind.assignment[k]))
        q.claimed = (net, claim, frozenset(reads))

    def _leaf_read(self, bind: BD.Binding, r: R.Route) -> dict | None:
        """52.10: THE LEAF READ BY THEIR AGENDA. A balance claim of ours (d1: our move x->y; the least over their replies of the most over our moves of the
        least over their replies of the balance) is followed to its leaves, the argmin / argmax recomputed over python-chess legal moves by reading the SAME tail
        terms of the vocabulary on each child (`SM.tail_body`; `least` over theirs, `most` over ours, as the word is defined), and THEIR agenda (side_depth 0)
        is run at the leaf (their turn). A leaf is REFUTED when their best claim there, w, is at least the gain we have AT the leaf before their move (the balance there less B, the
        balance now): their claim is measured from the leaf, while the word's number V already holds their best one-move reply, so comparing w with the claimed
        gain V - B would count their recapture twice (it killed 000Sa's Bxc2 at the queen trade: reported). The claim is refuted WITHIN ONE VIEW when
        a reply leaves every maximising move of ours refuted -> {claim, line} (evidence of a refutation, not a guarantee). Else None.
        The replies read are the MINIMISING ones (the spec); with `leaf_all` every reply, the minimising first (a claim "whatever they reply" is false if it fails
        after any reply; 0UqOv's trapped queen, fxe6 Qxa8 Qc7, is not on the minimising line -- but their agenda finds nothing there either). Ties:
        our `most` is an existence (one maximising move whose leaf they do not refute keeps the reply). Memoised per route."""
        key = r.move
        if key in self._leaf:
            return self._leaf[key]
        self._leaf[key] = None
        self.leaf_stats['read'] += 1
        fam = BD.schema(bind.schema).family
        if fam not in PA.ORDER:
            return None
        b, us, x, y = self.board, self.us, bind.assignment["x"], bind.assignment["y"]
        mv = chess.Move.from_uci(x + y)
        body = SM.tail_body(us, fam, "1000")                              # (least (most (least (min balance alive)))): the first least is over their replies
        inner1, inner2 = body[1], body[1][1]                              # (most (least ..)) after their reply; (least (min ..)) after our move
        t_root = SM.val_at(b.fen(), SM.delayed_tail([(x, y)], us, fam, "1000"), us)
        c1 = b.copy(stack=False)
        c1.push(mv)
        reps = list(c1.legal_moves)
        if not reps:
            return None
        fens = []
        for rep in reps:
            d = c1.copy(stack=False)
            d.push(rep)
            fens.append(d.fen())
        v1 = SM.evaluate([inner1], [dict(fen=f, mover=us) for f in fens])[BV.sexpr(inner1)]
        lo = min(v1)
        if lo != t_root:                                                   # the decomposition must read what the word reads, else the leaf is not trusted
            self.lines.append(f"  leaf read of {r.san}: the argmin decomposition reads {lo}, the word reads {t_root}: not trusted, claim kept")
            self.leaf_stats['untrusted'] += 1
            return None
        base = lo - bind.value                                             # B: the balance now, in the word's units
        san_x = b.san(mv)
        order = sorted((i for i in range(len(reps)) if self.leaf_all or v1[i] == lo), key=lambda i: (v1[i], i))
        refuted, shown = None, None
        for i in order:
            rep, v = reps[i], v1[i]
            d = c1.copy(stack=False)
            san_r = c1.san(rep)
            d.push(rep)
            # the word reads from the side to move at the position it is given (balance = the mover's material less the other's), so a position after OUR move
            # is read from the position before it, with `after x y`, exactly as the tail is built (castling and promotions have no `after`: left out, and a top
            # that then misses the word's is reported, never trusted)
            leaves = []
            for m in list(d.legal_moves):
                if m.promotion or d.is_castling(m):
                    continue
                e = d.copy(stack=False)
                e.push(m)
                leaves.append((m, e))
            terms = [("after", chess.square_name(m.from_square), chess.square_name(m.to_square), inner2) for m, _ in leaves]
            vals = SM.evaluate(terms, [dict(fen=d.fen(), mover=us)]) if terms else {}
            v2 = [vals[BV.sexpr(t)][0] for t in terms]
            top = max(v2) if v2 else None
            if top != v:
                self.lines.append(f"  leaf read of {r.san}: after {san_r} the argmax decomposition reads {top}, the word reads {v}: not trusted, claim kept")
                self.leaf_stats['untrusted'] += 1
                return None
            tied = [(m, e) for (m, e), w in zip(leaves, v2) if w == top]
            tied.sort(key=lambda t: (not d.is_capture(t[0]), not d.gives_check(t[0]), t[0].uci()))          # the natural moves first; the order never changes the verdict
            bterms = [("after", chess.square_name(m.from_square), chess.square_name(m.to_square), "balance") for m, _ in tied]
            bvals = SM.evaluate(bterms, [dict(fen=d.fen(), mover=us)]) if bterms else {}
            gains = [bvals[BV.sexpr(t)][0] - base for t in bterms]         # our gain AT the leaf, before their move: their claim is measured from here
            first_ref, held = None, None
            for (m, e), gain_here in zip(tied, gains):
                line = f"{san_x} {san_r} {d.san(m)}"
                theirs = None
                if any(True for _ in e.legal_moves):
                    lk = ("leaf", _pos_key(e.fen()))                       # 64.0: memoised by position in the run's `theirs` table (the leaf read now runs in the inner agendas too)
                    if self.ctx is not None and lk in self.ctx.theirs:
                        theirs = self.ctx.theirs[lk]
                    else:
                        ag = self._their_agenda(e)
                        theirs = max(ag.state.claims, key=lambda cl: cl.value) if ag.state.claims else None
                        if self.ctx is not None:
                            self.ctx.theirs[lk] = theirs
                if theirs is not None and theirs.value >= gain_here:
                    first_ref = first_ref or dict(claim=theirs, line=line, w=theirs.value, gain=gain_here)
                    continue
                held = (line, theirs)
                break
            if held is None and first_ref is not None:
                refuted = first_ref                                        # every maximising move of ours is refuted after THIS reply
                break
            if shown is None:
                shown = held
        if refuted is not None:
            c = refuted["claim"]
            self.leaf_obs.append(f"our {san_x} claimed {bind.value}; at the leaf {refuted['line']} (our gain there {refuted['gain']}) their {c.what} {c.value} (lemma {c.lemma}) -> downgraded")
        elif shown is not None:
            line, theirs = shown
            self.leaf_obs.append(f"our {san_x} claimed {bind.value}; at the leaf {line} (the minimising reply) " + (f"their {theirs.what} {theirs.value} (lemma {theirs.lemma})" if theirs else "no claim of theirs") + " -> kept")
        self.leaf_stats['down'] += 1 if refuted is not None else 0
        self._leaf[key] = refuted
        return refuted

    def _decide_by_smallest_loss(self, q: R.Question) -> bool:
        """52.8: a `save` root whose parries bind no gain DECIDES the parry after which they have the least: net = our bound value (0 when nothing binds) less
        their best claim on the future (their agenda, `their_view`); ties by puzzle_apply's shape key (fewest hanging, most guarded, most poking). The claim is
        `save <san>` with value = net (<= 0 here), lemma none, and it ranks BELOW any gain claim (the best-key compares the net). Never for a win root; never in
        their agenda (side_depth 0: no view of the view)."""
        if not self.side_depth or q.goal != "save":
            return False
        if not self.board.is_check():                      # ONLY in check (2026-10-17 audit): elsewhere "no claim of theirs binds" means UNKNOWN, not 0 -- 10 of 11
            self.lines.append("    nothing gains and we are not in check: no decision by the smallest loss (their silence is unknown, not zero)")
            return False                                   # such decisions were materially wrong by Stockfish (06J6T Bf3, 07Z0P a4, 0DoFB c5 ...); in check we must move
        b, us = self.board, self.us
        seen, routes = set(), []
        for f in q.fixes:
            for r in f.routes:
                if r.move not in seen:
                    seen.add(r.move)
                    routes.append(r)
        if not routes:
            return False
        rows = []
        for r in routes:
            bind, _ = BD.verify_tried(r.move, b, us, False)
            gain = bind.value if bind is not None else 0
            theirs, _ag = self.their_view(r.move)
            tv = theirs.value if theirs is not None else 0
            sh = PA.shape(b.fen(), [(chess.square_name(r.move.from_square), chess.square_name(r.move.to_square))], us, threats=False)
            rows.append((gain - tv, PA.shape_key(sh), r, theirs, gain))
        rows.sort(key=lambda t: (-t[0], t[1], t[2].move.uci()))
        net, key, r, theirs, gain = rows[0]
        tied = [t for t in rows[1:] if t[0] == net]
        worse = [t for t in rows[1:] if t[0] < net]
        self.lines.append(f"  nothing gains: deciding by the smallest loss, each parry read from the opponent's view ({len(rows)} parries)")
        for n_, k_, rr, th_, g_ in rows[:8]:
            self.lines.append(f"    {rr.san}: " + (f"their {th_.what} wins {th_.value} (lemma {th_.lemma})" if th_ is not None else "no claim of theirs binds")
                              + (f", our bound gain {g_}" if g_ else "") + f" -> net {n_}")
        if len(rows) > 8:
            self.lines.append(f"    ... and {len(rows) - 8} more")
        why = (f"their best claim after it is {-net + gain}" if theirs is not None else "no claim of theirs binds after it")
        if worse:
            wt = worse[0]
            why += f" (after {wt[2].san}: " + (f"their {wt[3].what} wins {wt[3].value}" if wt[3] is not None else "net " + str(wt[0])) + ")"
        if tied:
            why += f"; tied at net {net} with {', '.join(t[2].san for t in tied[:5])}: the shape key (fewest hanging, most guarded, most poking) prefers {r.san}"
        self.lines.append(f"  decision {r.san}: {why}")
        claim = Claim("save", r.san, net, "none", f"{r.san}: the opponent's best claim is {-net + gain}" if theirs is not None else f"{r.san}: no claim of theirs binds", move=r.move.uci())
        self._stamp(claim, q)
        self.state.claims.append(claim)
        q.status = "verified"
        self.lines.append(f"  -> CLAIM save {r.san} (net {net}), lemma none: the opponent's best claim is {-net + gain}")
        if self._better(net, 0, claim):
            self.best = (net, 0, claim)
            self.asked_at_claim = self.asked
        return True

    @staticmethod
    def _asked(tried: list) -> str:
        """The schemas asked, smallest first, each with what it read: '-' (premise false), 'T/B' (the tail read T against the balance B now), or its value."""
        out = []
        for n, v, note in tried:
            m = re.search(r"reads (\d+) against (\d+)", note or "")
            out.append(f"{n} {v if v is not None else (m.group(1) + '/' + m.group(2) if m else '-')}")
        return ", ".join(out)

    def _line_of(self, bind: BD.Binding) -> str:
        if bind.schema != "mate_in_two":
            return f"{bind.san} (every reply: {len(bind.replies)} replies)" + (f" ({bind.note})" if bind.note else "")
        b = self.board.copy(stack=False)
        b.push(chess.Move.from_uci(bind.assignment["x"] + bind.assignment["y"]))
        parts = []
        for r in list(b.legal_moves):
            c = b.copy(stack=False)
            c.push(r)
            got = SM.bind_mate(c.fen(), self.us)
            mv = c.san(chess.Move.from_uci("".join(got[0]["pairs"][0]))) if got else "?"
            parts.append(f"{b.san(r)} {mv}")
        return f"{bind.san} " + " / ".join(parts)

    @staticmethod
    def _mates(c: chess.Board, mv: chess.Move) -> bool:
        return G.mates(c, mv)

    def _escape_note(self, r: R.Route, tried: list) -> str:
        """Why a mate route did not bind: the replies after which we have no mate in one."""
        notes = [f"{n}: {note}" for n, v, note in tried if note and v is None]
        if not self.side_depth:                                          # their agenda and the leaf read never print the note: no reply walk there
            return f"{r.san} fails" + (f" ({'; '.join(notes)})" if notes else "")
        b = self.board.copy(stack=False)
        b.push(r.move)
        esc = []
        for rep in list(b.legal_moves):
            c = b.copy(stack=False)
            c.push(rep)
            if any(self._mates(c, mv) for mv in c.legal_moves):         # 52.11: mate in one is a rules fact (the binder's mirror of it is tested in test_regress)
                continue
            esc.append(b.san(rep))
        return f"{r.san} fails" + (f" ({'; '.join(notes)})" if notes else "") + (f", no mate in one after {', '.join(esc[:4])}" if esc and 0 < len(esc) <= 40 else "")

    def _ask_reach(self, q: R.Question, head: str) -> None:
        """52.12 (4): the root whose only defect is the reach. The fix routes are the first steps; a step is FORCING when it is a check or after it a pass leaves us a
        mate in one (a mate threat: their every reply must meet it). The binder reads each over every reply; when none binds the chase regresses through their replies."""
        b, us = self.board, self.us
        d = q.defects[0]
        self.lines.append(head + f": defect reach ({d.why})")
        steps = [chess.Move.from_uci(u) for u in q.pattern["reach"]]
        f = R.new_question("cover", q.stake, dict(q.pattern), q, defect_key=d.key(), text=f"step toward {G.name(q.pattern['at'])} with tempo")
        forcing, quiet = [], []
        for m in steps:
            why = R.forcing_why(b, m, q.stake, us, mate_threats=True)
            (forcing if why else quiet).append(R.Route(m, b.san(m), why) if why else b.san(m))
        f.routes, f.unforced = forcing, quiet
        q.fixes.append(f)
        self.asked += 1
        self.lines.append(f"  -> Q{f.id} {f.text}: " + (", ".join(f"{r.san} ({r.why})" for r in forcing) if forcing else "no forcing step" + (f" (quiet steps: {', '.join(quiet[:4])})" if quiet else "")))
        q.reads = R.reads_of(q, b)
        if not forcing:
            if self._chase(q):
                return
            self._kill(q, "tempo: reach: only quiet steps, the opponent gets a free reply" if quiet else "tempo: reach: no step", tempo=True)
            return
        self._verify(q, forcing, mates_only=True)

    # ------------------------------------------------------------------------------------------------------------ the chase (52.12)

    def _inner(self, fen: str, need: int, diff: "frozenset | None" = None, their: "Agenda | None" = None, keep: int = 0, since: int | None = None, exchange: bool = False, reset: bool = False) -> "dict | None":
        """The WHOLE agenda asked again on `fen` (our move), only the roots of stake >= `need` (a smaller claim cannot rescue the route): -> {value, claim, asked, need};
        None when the shared budget is spent. Memoised by FEN (an answer asked with a smaller `need` serves a larger one); a FEN being asked (a cycle) answers no claim.
        52.14: with `their` (their agenda there, the move passed to them) their claims are also asked as SAVE roots; the answer lists them (`threats`: (their claim, our parry claim or None)).
        The memo is keyed by (position, with threats): an answer asked WITH threats serves a later ask without them, not the reverse."""
        ctx = self.ctx
        pk = _pos_key(fen)
        wt = their is not None
        evading = self._chase_q is not None and self._chase_q.goal == "save" and "checkers" in self._chase_q.pattern
        hit = None
        if hit is None:
            hit = ctx.memo.get((pk, True, keep)) if wt else (ctx.memo.get((pk, False, 0)) or next((h for k_, h in ctx.memo.items() if len(k_) == 3 and k_[0] == pk and k_[1]), None))     # (position, with threats, the gain kept): an answer WITH threats serves an ask without, not the reverse
        if hit is not None and hit["need"] <= need:
            self._touch(hit.get("fact"))
            return hit
        if pk in ctx.active or (CYCLE and (pk == ctx.root_pk or pk in ctx.history)):
            if CYCLE and not wt:
                cyc = self._cycle_claim(fen, pk, need)
                if cyc is not None:
                    return cyc
            return dict(value=0, claim=None, asked=0, need=need, threats=[(cl, None) for cl in their.state.claims] if wt else [])     # a cycle parries nothing
        if ctx.left <= 0 or self.chase_depth >= DEPTH_FUSE or (ctx.out_of_time() and not ctx.finishing(ctx.route_depth)):      # 52.21 (3): only the route in hand opens an agenda past the deadline
            if ctx.left > 0:
                ctx.depth_cuts += 1
            return None
        if COST_JUDGE and self._judge_node(fen):                # plan 61 C2: the judge re-asked at this node of the route with the remaining cost of the chain
            return None
        ctx.active.add(pk)
        ctx.active_label[pk] = f"{self.board.san(self._route_move)} {self._reply_san}" if self._route_move is not None and self._reply_san else "the line"
        hits0, chain0 = ctx.budget_hits, ctx.chain_hits
        saved = R._ids[0]
        evading = self._chase_q is not None and self._chase_q.goal == "save" and "checkers" in self._chase_q.pattern
        ag = Agenda(State(fen=fen, us=self.us, pieces=[]), side_depth=0, chase_depth=self.chase_depth + 1, ctx=ctx, min_stake=need, their=their, keep=keep, evaded=(int(self.evaded) + 1) if evading and not reset else 0,
                    since=self.since_capture + 1 if since is None else since, free_level=exchange)      # (52.19: what it inherits comes from the fact store, not from this agenda alone)
        ag.ask()
        R._ids[0] = saved
        ctx.active.discard(pk)
        ctx.active_label.pop(pk, None)
        ctx.agendas += 1
        ctx.asked += ag.asked
        best = ag.best[2] if ag.best and ag.best[0] > 0 else None
        reads = None if any(q.reads is None for q in ag.all) else frozenset().union(*(q.reads for q in ag.all))     # 52.13 (4): everything this agenda read (None: some question did not record it)
        threats = [(cl, q.claimed[1] if q.status == "verified" and q.claimed is not None else None) for q, cl in ag.threat_qs]
        out = dict(value=ag.best[0] if best else 0, claim=best, asked=ag.asked, need=need, cut=ctx.budget_hits > hits0, chain=ctx.chain_hits > chain0, reads=reads, threats=threats,
                   promote=next((c for c in ag.state.claims if c.kind == "promote"), None))      # 52.15 (2): the promote claim of this position, whether or not it is the best claim
        cfid = ag.claim_fid.get(id(best)) if best is not None else None
        f = ctx.facts.add("gap" if out["cut"] else ("claim" if best is not None else "kill"), ("inner", pk, wt, keep), self.me, f"the agenda on {pk.split()[0]}" + (f": {best.what}" if best is not None else ": no claim"),
                          value=out["value"], reads=reads, just=[cfid] if cfid is not None else ag.premises, stake=self._cur.stake if self._cur is not None else need, depth=self.chase_depth + 1,
                          under=f"Q{self._cur.id} at depth {self.chase_depth}" if self._cur is not None else f"depth {self.chase_depth}", pk=pk, portable=False)
        out["fact"] = f.id
        self._touch(f.id)
        if not out["cut"] and not out["chain"]:                 # an answer the budget cut short (or a chain stop, which depends on the chain's state) is not memoised and is never read as "no claim"
            ctx.memo[(pk, wt, keep if wt else 0)] = out
        return out

    def _touch(self, fid) -> None:
        """52.19: the route being chased read this fact (an inner answer, their agenda): the claim or kill of the route stands on it."""
        if fid is not None and self._rf is not None and fid not in self._rf:
            self._rf.append(fid)

    def _carried_reply(self, key: tuple, uci: str, gstake: int = 0) -> "dict | None":
        """plan 66.2 CLOSED REPLIES CARRY INSIDE A ROUTE (Lean `Ideas.Forcing.claim_mono`, `routeLow_mono`): a reply of this route PROVED in an earlier tier (closed by rule or answered by an inner claim) is a
        proof over the same board and the same reply of theirs: it is served, not read again. Served only when it already meets what THIS read needs -- the agenda's target (the need it propagates), the floor of a
        re-asked root question (66.1: a reply at or below the floor is the only place the route could rise above it, so it is read again), and for a close by the exchange the goal's stake (52.22 (1): a close below
        it would not have been made) -- else the reply is read as before. A served value is a floor of what the wider tier would read (never more). Kills and GAPs are never carried."""
        if not CARRY or not self.ctx.carry_r or key[1].endswith(":refuter"):
            return None
        hit = self.ctx.carry_r.get((key[0], key[1], uci))
        if hit is None:
            return None
        need = self._target()
        if self._floor is not None and self.chase_depth == 0:
            need = max(need, self._floor + 1)
        if hit["g"] < need or (hit["closed"] and gstake and hit["g"] < gstake):
            self.ctx.carry_reread += 1
            return None
        self.ctx.carry_served += 1
        return hit

    def _proved_reply(self, key: tuple, uci: str, g: int, part: str, lemma: "str | None" = None, closed: bool = False, material: bool = False) -> None:
        """plan 66.2: file a reply this route PROVED (its value g against this reply of theirs) for the next tier. The refuter fix's replies are its own (not filed)."""
        if not CARRY or key[1].endswith(":refuter"):
            return
        k = (key[0], key[1], uci)
        old = self.ctx.proofs.get(k)
        if old is None or g > old["g"]:
            self.ctx.proofs[k] = dict(g=g, part=part, lemma=lemma, closed=closed, material=material, tier=self.ctx.tier)      # (material: a promote route's reply answered by winning the material back, not by the promotion)

    @staticmethod
    def _serve_carried(cr: dict, res: dict, gains: list) -> None:
        """plan 66.2: a carried reply enters the route like a read one, said so."""
        gains.append(cr["g"])
        res["parts"].append(f"{cr['part']} [carried from tier {cr['tier']}]")
        if cr["lemma"]:
            res["lemmas"].append(cr["lemma"])
        if cr.get("material"):
            res["material"] = True                               # (a promote route answered by material is a win claim, as when read)
        res["carried"] = res.get("carried", 0) + 1
        res["carried_from"] = min(res.get("carried_from", cr["tier"]), cr["tier"])

    def _chase_route(self, m: chess.Move, promote: bool = False, cheap: bool = False, override: "dict | None" = None) -> dict:
        """52.19: the route of `_chase_route_body`, with the facts it read (inner answers, their agendas) kept on the result (`facts`) for the claim or kill that stands on it, and the listener's
        frame for the reply being read restored on the way out."""
        saved_rf, saved_frame = self._rf, self._frame
        self._rf, self._frame = [], []
        self._route_move = m
        if self.chase_depth == 0 and self.ctx.deadline is not None:
            self.ctx.slice_len = max(self.ctx.deadline - time.monotonic(), 0)      # 52.23 (11): the slice this root route was given
        ctx = self.ctx
        ctx.route_depth += 1
        ctx.route_names.append(self.board.san(m))
        try:
            res, tok = self._judge_begin(m, promote, cheap, override) if COST_JUDGE else (None, None)
            if res is None:
                res = self._chase_route_body(m, promote, cheap, override)
            if COST_JUDGE or CYCLE:
                self._judge_end(tok, res)
            if ESTIMATE and not cheap and not promote and res.get("gap") is not None:
                self._estimate_route(m, res)                         # plan 61 B2: the quiet leaves of a cut route
            res["facts"] = list(dict.fromkeys(res.get("facts", []) + self._rf))
            if not cheap:
                self._gap_fact(m, promote, res)
            return res
        finally:
            if ctx.hand is not None and not ctx.abort and ctx.route_depth == ctx.hand and not cheap:
                ctx.finished += 1                                    # 52.21 (3): the route in hand ran to its end (or to a GAP of its own)
            ctx.route_depth -= 1
            ctx.route_names.pop()
            self.ctx.facts.restore(self._frame)
            self._rf, self._frame = saved_rf, saved_frame

    def _gap_fact(self, m: chess.Move, promote: bool, res: dict) -> None:
        """52.19 (3): a route the fuse or the clock cut is a GAP FACT, kept with where to resume (the position, the route, the reply it stopped at, the time mark) and the stake of the question
        that chased it; it waits in the background queue by chain stake and is resolved when the route is later followed to its end."""
        ctx, fs = self.ctx, self.ctx.facts
        key = (self.pk, m.uci() + (":promote" if promote else ""))
        if res.get("gap") is None or res.get("reused"):
            fid = ctx.gap_facts.pop(key, None) if res.get("gap") is None else None
            if fid is not None:
                fs.facts[fid].resolved = True
            return
        q = self._cur
        if key in ctx.gap_facts:
            fs.facts[ctx.gap_facts[key]].superseded = True        # the route was cut again: the new gap replaces the old
        f = fs.add("gap", ("route",) + key, self.me, f"{self.board.san(m)}: {res['gap']}", value=0, stake=q.stake if q is not None else 0, depth=self.chase_depth,
                   under=f"Q{q.id} at depth {self.chase_depth}" if q is not None else f"depth {self.chase_depth}", pk=self.pk, portable=False, just=res["facts"],
                   resume=dict(fen=self.board.fen(), uci=m.uci(), promote=promote, reply=res.get("reply"), timed=bool(res.get("timed")), mark=res.get("mark"), question=q.text if q is not None else ""))
        ctx.gap_facts[key] = f.id

    def _enter_reply(self, diff) -> None:
        """52.19 (2): THE LISTENER. A reply is about to be read: the squares its route and itself changed flag every fact that read them (and, through the justifications, every fact built on them);
        nothing is recomputed. The flags of the previous reply of this route are restored first (the chase is a tree: a sibling does not inherit them)."""
        self.ctx.facts.restore(self._frame)
        self._frame = self.ctx.facts.invalidate(diff)

    def _chase_route_body(self, m: chess.Move, promote: bool = False, cheap: bool = False, override: "dict | None" = None) -> dict:
        """One forcing route of the chase: every reply of theirs, the whole agenda on each resulting position. -> {value, killed, gap, parts, lemmas}.
        52.17 (4) `cheap`: CHEAPEST REPLY FIRST -- only what the rules close is read (standing kills, a reply that mates or stalemates us, a mate in one, the exchange with nothing of
        theirs, a promotion that stands); a reply that needs an inner agenda leaves the route PENDING (not memoised) and the closed replies are kept for the second pass.
        52.15 (2) `promote`: the question is a promotion; a reply is answered when the inner agenda claims the promotion (or a mate), or the reply itself gave up at least the
        stake in material; the roots asked there are those of stake >= the promote stake, and the route's value is the promote stake (a mate: 100)."""
        b, ctx = self.board, self.ctx
        key = (_pos_key(b.fen()), m.uci() + (":promote" if promote else "") + (":refuter" if override else ""))      # 52.22 (3) `override`: reply uci -> (value, part, facts) read by the refuter fix, not by an inner agenda
        pre = self._pre.pop(key, None) if not cheap else None      # the replies the cheap pass closed by rule (value, why)
        cq = self._chase_q
        gstake = cq.stake if cq is not None and cq.goal == "win" and cq.pattern.get("target") is not None else 0        # 52.22 (1): what the exchange must reach to close a reply by rule
        if key in ctx.routes:
            hit = ctx.routes[key]
            stale = hit["gap"] is None and hit["killed"] is None and gstake and hit.get("closed_min", gstake) < gstake and hit.get("stake_checked", 0) < gstake       # a value closed by a short exchange under a lesser goal is not this goal's
            stale = stale or (hit["gap"] is not None and hit["gap"].startswith("chain stop") and hit.get("chain_state") != self._chain_state())      # 66.4 (mh1q8): a chain stop under another chain state is not this one's
            if hit["gap"] is None and not stale:
                return hit
            slice_now = max(ctx.deadline - time.monotonic(), 0) if ctx.deadline is not None else None
            if not stale and (slice_now <= hit.get("cut_slice", 0) if slice_now is not None else ctx.left <= hit["cut_left"]):      # 52.13 (2a): a GAP stays a GAP while the fuse left is no more than when it was cut; 52.23 (11): under a time budget, while its slice is no longer than the one it was cut on
                ctx.gap_reused += 1
                return dict(hit, reused=True)
        san = b.san(m)
        c = b.copy(stack=False)
        c.push(m)
        res = dict(value=0, killed=None, gap=None, parts=[], lemmas=[])
        replies = list(c.legal_moves)
        if not replies:
            if c.is_checkmate():
                res.update(value=BD.MATE_VALUE, parts=[f"{san} mates"])
            else:
                res["killed"] = "stalemate"
            ctx.routes[key] = res
            return res
        # 52.13 (5): the replies are read through the Rust `Pos` (no FEN, no board copy per reply); a python-chess board is built only for a reply that is read further
        P0 = AC.Pos(b.fen())
        P1 = P0.push(m.uci())
        shift = P1.balance(self.me) - P0.balance(self.me)
        rows = [(shift + dbal, uci, chk, cap, rec, mate1) for uci, dbal, chk, cap, rec, mate1 in P1.replies()]
        pos_of: dict = {}

        def P2_of(uci: str) -> "AC.Pos":
            if uci not in pos_of:
                pos_of[uci] = P1.push(uci)
            return pos_of[uci]

        def san_of(uci: str) -> str:
            return c.san(chess.Move.from_uci(uci))
        # 52.13 (4): the SHORTLIST. A standing kill (an earlier route's reply, the same reply move, nothing its reads touch has changed in the position it leaves) ends the
        # route at once and is counted; otherwise the replies are read recaptures on the route's square, then checks, then captures, then the rest, the worst balance first inside a class.
        kill_uci = None                                          # 52.22 (3): the reply that killed the route: its piece is the REFUTER (Ideas.Refuter.Refutes)
        for dbal, uci, chk, cap, rec, mate1 in rows:
            sk = None if promote or (override and uci in override) else self._standing_kill(uci, P2_of(uci), dbal, self._target())
            if sk is not None:
                kill_uci = uci
                res["killed"] = f"after {san_of(uci)}: {sk[0]} (standing kill: nothing the inner agenda read after {sk[1]} {san_of(uci)} has changed)"
                ctx.standing += 1
                self._log(key, uci, "standing", dbal, None, sk[2], res["killed"])
                break
        if res["killed"] is not None:
            res["refuter"] = kill_uci
            ctx.routes[key] = res
            return res
        rows.sort(key=lambda t: (0 if t[4] else 1 if t[2] else 2 if t[3] else 3, t[0], t[1]))
        gains = []
        closed_now = {}
        closed_min = BD.MATE_VALUE                               # the least value an exchange closed a reply at (the memo entry keeps it: another goal's stake may ask for more)
        for dbal, uci, chk, cap, rec, mate1 in rows:
            P2 = P2_of(uci)
            san_r = san_of(uci)
            kill_uci = uci
            if override and uci in override:                     # 52.22 (3): the reply the refuter fix answered
                val, part, ofacts = override[uci]
                gains.append(val)
                res["parts"].append(part)
                self._rf.extend(f for f in ofacts if f not in self._rf)
                continue
            if pre is not None and uci in pre:                   # closed by rule in the cheap pass: counted and logged there
                val, why = pre[uci]
                if val < BD.MATE_VALUE:
                    closed_min = min(closed_min, val)
                gains.append(val)
                res["parts"].append(f"after {san_r}: closed {self._closed_tag(why)}: {why} -> {val}")
                continue
            cr = self._carried_reply(key, uci, gstake) if not override else None      # plan 66.2: a reply proved in an earlier tier
            if cr is not None:
                if cr["closed"] and cr["g"] < BD.MATE_VALUE:
                    closed_min = min(closed_min, cr["g"])
                self._serve_carried(cr, res, gains)
                continue
            if P2.legal_count() == 0:
                res["killed"] = f"after {san_r} " + ("they mate us" if P2.is_check() else "stalemate")
                self._log(key, uci, "killed", dbal, None, None, res["killed"])
                break
            closed = ((BD.MATE_VALUE, "mate in one") if (P2.mate_in_one() if mate1 is None else mate1) else None) if promote else self._closed_by_rule(P2, dbal, mate1)
            if closed is not None and not promote:
                closed = self._parry_label(m, uci, closed)
            if closed is not None:
                val, why = closed
                if val < BD.MATE_VALUE:
                    closed_min = min(closed_min, val)
                gains.append(val)
                res["parts"].append(f"after {san_r}: closed {self._closed_tag(why)}: {why} -> {val}")
                self._proved_reply(key, uci, val, res["parts"][-1], closed=True)
                ctx.closed += 1
                closed_now[uci] = closed
                self._log(key, uci, "closed", dbal, val, None, why)
                continue
            if promote and dbal < self.promote_stake:
                why = self._promote_stands(m, P2)
                if why is not None:
                    res["killed"] = f"after {san_r}: {why}"
                    ctx.promote_stands += 1
                    self._log(key, uci, "killed", dbal, None, None, res["killed"])
                    break
            if cheap:                                            # the reply needs an inner agenda: the route stays open for the second pass
                res["pending"] = True
                break                                            # (the replies after it stay for the second pass: reading them all by rule cost 20x their agendas)
            target = max(1, self.min_stake) if self._t(NEED_PROPAGATION, 3) else 1      # 52.23 (7): THE NEED PROPAGATES -- the route must reach what this agenda serves (its min_stake), not 1 (zBouR: Qh1+ Rf1 read to 5 when 6 was needed)
            need = min(R.STAKE_PROMOTE, 1 - dbal) if promote and dbal <= 0 else R.STAKE_PROMOTE if promote else target - dbal if dbal < target else BD.MATE_VALUE          # a reply that already gains the need is read for a mate only; else its static gain must survive their threats (below)
            self._enter_reply(frozenset(P0.diff(P2)))
            self._reply_san = san_r
            ctx.line.append((b.fen(), m.uci(), uci, self._judge_stake()))      # plan 66.4: the line being read
            inn = self._inner(P2.fen(), need, frozenset(P0.diff(P2)), since=self._since_after(uci, dbal), reset=self._resets(uci),
                              exchange=self.board.is_capture(self._route_move) and chess.Move.from_uci(uci).to_square == self._route_move.to_square)      # 52.23 (6): a recapture on the route's square: the exchange is ONE event of the chain
            ctx.line.pop()
            if inn is None:
                res["gap"] = self._cut(san, san_r)
                res["reply"] = uci
                ctx.budget_hits += 1
                self._log(key, uci, "gap", dbal, None, None, res["gap"])
                break
            if promote:                                          # 52.15 (2): the inner agenda asked the promote root (and the mates) again on this reply
                pc = inn.get("promote")
                if inn["value"] >= BD.MATE_VALUE:
                    g, part = BD.MATE_VALUE, f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value})"
                elif pc is not None:
                    g, part = R.STAKE_PROMOTE, f"{pc.what} ({pc.line}; value {pc.value}, lemma {pc.lemma})"
                elif inn["claim"] is not None and inn["value"] >= self.promote_stake:     # a claim of at least the stake (the agenda stops at it: the promote root itself is then not asked)
                    g, part = R.STAKE_PROMOTE, f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value}, at least the stake {self.promote_stake})"
                elif dbal >= self.promote_stake:
                    g, part = dbal, f"material {dbal:+d} (the reply gave up at least the stake {self.promote_stake})"
                elif dbal <= 0 and inn["claim"] is not None and inn["value"] >= 1 - dbal:       # no promotion, but the material the reply cost is won back and more (the ordinary chase's reading)
                    g, part = dbal + inn["value"], f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value}) wins the material back: {dbal:+d} + {inn['value']}"
                    res["material"] = True
                elif (inn.get("cut") or inn.get("chain")):
                    res["gap"] = self._cut(san, san_r, True, inn)
                    res["reply"] = uci
                    self._mark_chain(res)
                    ctx.chain_hits += 0 if inn.get("cut") else 1      # 64.2: a chain stop is counted apart from a budget cut
                    ctx.budget_hits += 1 if inn.get("cut") else 0
                    self._log(key, uci, "gap", dbal, inn["value"], None, res["gap"])
                    break
                else:
                    res["killed"] = f"after {san_r}: the promotion is not claimed (no promote claim of ours; the reply took {-dbal})" if dbal <= 0 else f"after {san_r}: the promotion is not claimed ({dbal:+d}, below the stake)"
                    self._log(key, uci, "killed", dbal, inn["value"], inn.get("reads"), res["killed"])
                    break
                gains.append(g)
                res["parts"].append(f"after {san_r}: {part} -> {g}")
                self._proved_reply(key, uci, g, res["parts"][-1], pc.lemma if pc is not None else None, material=g == dbal + inn["value"] and res.get("material", False) and pc is None and inn["value"] < BD.MATE_VALUE)
                self._log(key, uci, "claim", dbal, inn["value"], inn.get("reads"), res["parts"][-1])
                if pc is not None:
                    res["lemmas"].append(pc.lemma)
                continue
            static = dbal >= 1 and inn["value"] < BD.MATE_VALUE
            t_their, claim_their, unparried, parry_note = 0, None, 0, None
            if static and not P2.is_check():                    # 52.13 (3), 52.14: a gain that stands only as a static balance (us to move) is a result only once THEIR agenda has read it, with the move passed to them, at every chase depth
                t_their, claim_their = self._their_pass(P2.fen())
                if claim_their is not None:
                    ctx.static_read += 1
                if claim_their is not None and dbal - t_their < 1:
                    # 52.14: their claim is a THREAT and a SAVE root of the inner agenda on this position (we are to move: a parry, a counter, taking the attacker); the reply is
                    # killed only by the threats no parry of ours is verified for
                    ctx.line.append((b.fen(), m.uci(), uci, self._judge_stake()))
                    inn2 = self._inner(P2.fen(), need, frozenset(P0.diff(P2)), their=ctx.their_ags.get(_pos_key(G.with_turn(chess.Board(P2.fen()), not self.me).fen())), keep=dbal)
                    ctx.line.pop()
                    if inn2 is None:
                        res["gap"] = self._cut(san, san_r)
                        res["reply"] = uci
                        ctx.budget_hits += 1
                        self._log(key, uci, "gap", dbal, None, None, res["gap"])
                        break
                    inn = dict(inn, cut=inn.get("cut") or inn2.get("cut"), chain=inn.get("chain") or inn2.get("chain"))
                    held = [(cl, pc) for cl, pc in inn2["threats"] if pc is not None]
                    lost = [cl for cl, pc in inn2["threats"] if pc is None]
                    unparried = sum(cl.value for cl in lost)
                    parry_note = "; ".join(f"their {cl.what} wins {cl.value} if we pass; parry: {pc.what} (lemma {pc.lemma}) keeps {pc.value:+d}" for cl, pc in held)
                    if lost:
                        ctx.static_unparried += 1
                    else:
                        ctx.static_parried += 1
                    self._static_line(san, san_r, dbal, claim_their, parry_note, lost)
                    if dbal - unparried < 1:
                        worst = max(lost, key=lambda cl: cl.value) if lost else claim_their
                        res["killed"] = f"after {san_r}: {dbal:+d} stands only as a balance, their {worst.what} wins {worst.value} (lemma {worst.lemma}), no parry"
                        self._log(key, uci, "killed", dbal, inn["value"], None, res["killed"])
                        break
                    claim_their = None
                else:
                    unparried = t_their
            if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE and (inn.get("cut") or inn.get("chain")):
                res["gap"] = self._cut(san, san_r, True, inn)
                res["reply"] = uci
                self._mark_chain(res)
                ctx.chain_hits += 0 if inn.get("cut") else 1      # 64.2: a chain stop is counted apart from a budget cut
                ctx.budget_hits += 1 if inn.get("cut") else 0
                self._log(key, uci, "gap", dbal, inn["value"], None, res["gap"])
                break
            if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE:
                res["killed"] = f"after {san_r}: " + (f"our best claim {inn['claim'].what} wins {inn['value']}, the reply took {-dbal}" if inn["claim"] else "no claim of ours and no material gain") + f" ({dbal:+d})"
                if inn["claim"] is None and not inn.get("cut") and not inn.get("chain"):          # a standing kill: the later routes' same reply is skipped while the reads stay untouched (with the need it was made for)
                    ctx.reply_kills.setdefault(uci, []).append((P2, dbal, inn.get("reads"), san, target))
                self._log(key, uci, "killed", dbal, inn["value"], inn.get("reads"), res["killed"])
                break
            cl = inn["claim"] if inn["value"] >= need else None
            gains.append(BD.MATE_VALUE if inn["value"] >= BD.MATE_VALUE else max(dbal - unparried, dbal + inn["value"] if cl and self._t(NEED_PROPAGATION, 3) else dbal - unparried) if static else dbal + (inn["value"] if cl else 0))      # 52.23 (7): a reply that already gains carries the inner claim too (it answers every reply of theirs): zBouR's Rxf1+ Kd2 Rf2+, +5 then the queen
            res["parts"].append(f"after {san_r}: " + (f"{cl.what} ({cl.line}; value {cl.value})" if cl else f"material {dbal:+d}" + (f", their {claim_their.what} wins {t_their} if we pass" if claim_their else "") + (f", {parry_note}" if parry_note else "")) + (f" -> {gains[-1]}" if cl or claim_their or parry_note else ""))
            self._proved_reply(key, uci, gains[-1], res["parts"][-1], cl.lemma if cl else None)
            self._log(key, uci, "claim" if cl else "material", dbal, inn["value"], inn.get("reads"), res["parts"][-1])
            if cl:
                res["lemmas"].append(cl.lemma)
        if cheap and res.get("pending") and res["killed"] is None:
            self._pre[key] = closed_now                          # not memoised: the second pass finishes it
            return res
        res["closed_min"], res["stake_checked"] = closed_min, gstake
        res["gains"] = list(gains)                               # plan 61 B2: the replies read before a cut (the estimate continues from here)
        if res["killed"] is not None and kill_uci is not None:
            res["refuter"] = kill_uci
        if res["killed"] is None and res["gap"] is None:
            res["value"] = min(gains) if gains else 0
        if res["gap"] is not None:
            res["cut_left"] = ctx.left
            res["cut_slice"] = ctx.slice_len                     # 52.23 (11): the slice this route had when it was cut (a later, longer slice re-runs it)
            if res["gap"].startswith("out of time"):                # 52.17 (5): the deadline mark on the GAP memo entry beside cut_left: the background tier resumes exactly here
                res["timed"], res["mark"] = True, round(time.monotonic() - ctx.t0, 3) if ctx.t0 is not None else None
        ctx.routes[key] = res
        return res

    def _refuter_fixes(self, q: R.Question, routes: list, results: dict, done: list, pool: int) -> None:
        """52.22 (3): THE REFUTER FIX (Lean `Ideas.Refuter.Refutes`). A route killed by a reply names the piece that made it, the REFUTER ("Bb3 fails to Qxb3"). The fix is the one the agenda applies to a
        defender or a blocker: take the refuter with tempo -- a route of this chase that captures it where it stands and is itself killed only by the RECAPTURE, by a piece that cannot make the
        refuting move -- and ask the killed route again on the board after the exchange (a new agenda, one level in; the goal there is the piece now on the target square). The capturing route is
        then read again with that one reply answered by the re-asked route (every other reply is read as before); a route the exchange saves is a claim of its own, filed with the refuter's squares in
        its read set."""
        b, ctx = self.board, self.ctx
        P0 = AC.Pos(b.fen())
        share = max(pool // max(len(routes), 1), 1)                # the fix is the continuation of a route: charged at most an equal share of the fuse the chase started with
        for i, r_k in enumerate(routes):
            res_k = results.get(i)
            if res_k is None or res_k["killed"] is None or not res_k.get("refuter") or res_k.get("gap"):
                continue
            ref = chess.Move.from_uci(res_k["refuter"])
            for j, r2 in enumerate(routes):
                res2 = results.get(j)
                if j == i or res2 is None or res2["killed"] is None or res2.get("gap") or res2.get("fixed_by") or not res2.get("refuter") or not b.is_capture(r2.move) \
                        or r2.move.to_square != ref.from_square or r_k.move.to_square == ref.from_square:
                    continue
                saved = ctx.left
                if saved < pool:
                    ctx.left = pool                              # (read on the whole pool the chase started with, charged at most its share: the finishing of a route is free, as in 52.16 (2))
                before = ctx.left
                try:
                    got = self._refuter_fix(q, r_k, ref, r2, res2["refuter"], P0)
                    new = self._chase_route(r2.move, override={res2["refuter"]: (got["value"], got["part"], got["facts"])}) if got is not None else None
                finally:
                    ctx.left = saved - min(max(before - ctx.left, 0), share) if saved > 0 else saved
                if new is None:
                    continue
                if new["killed"] is not None or new["gap"] is not None or new["value"] < 1:
                    continue
                new.update(extra_reads=got["reads"], fixed_by=r_k.san)
                results[j] = new
                self.lines.append(f"    {r2.san}: refuter fix -- {got['text']} => value {new['value']}")
                ctx.refuter_fixes += 1
                done.append((new["value"], r2, new))

    def _refuter_fix(self, q: R.Question, r_k: R.Route, ref: chess.Move, r2: R.Route, rep_uci: str, P0: "AC.Pos") -> "dict | None":
        """52.22 (3): the exchange `r2`, `rep_uci` (the recapture that killed r2), then `r_k` asked again: -> {value (the exchange's balance + the route's), part, facts, reads, text} or None
        (the reply is not a recapture on the square, the recapturing piece can make the refuting move, the route is no longer legal, or it is not worth a gain there)."""
        b, me, ctx = self.board, self.me, self.ctx
        c = b.copy(stack=False)
        pc_ref = c.piece_at(ref.from_square)
        c.push(r2.move)
        x = chess.Move.from_uci(rep_uci)
        if x not in c.legal_moves or not c.is_capture(x) or x.to_square != r2.move.to_square:
            return None
        san_x = c.san(x)
        san_ref_c = b.copy(stack=False)
        san_ref_c.push(r_k.move)
        san_ref = san_ref_c.san(ref)
        c2 = c.copy(stack=False)
        c2.push(x)
        if r_k.move not in c2.legal_moves:
            return None
        c3 = c2.copy(stack=False)
        c3.push(r_k.move)
        if any(mv.from_square == x.to_square and mv.to_square == ref.to_square for mv in c3.legal_moves):
            return None                                          # the piece that recaptures can make the refuting move
        hits = [s for s in chess.SquareSet(c3.attacks_mask(r_k.move.to_square)) if c3.piece_at(s) is not None and c3.piece_at(s).color != me and c3.piece_at(s).piece_type != chess.KING]
        tgt = max(hits, key=lambda s: (G.worth(c3.piece_at(s)), -s)) if hits else None         # the goal asked again: the best piece the route's piece attacks once the refuter is gone
        pc_t = c2.piece_at(tgt) if tgt is not None else None
        dbal = AC.Pos(b.fen()).push(r2.move.uci()).push(x.uci()).balance(me) - P0.balance(me)
        saved = R._ids[0]
        q2 = R.new_question("win", G.worth(pc_t), dict(target=tgt), text=f"win their {pc_t.symbol().upper()} on {G.name(tgt)}") if pc_t is not None else None
        R._ids[0] = saved
        ag2 = Agenda(State(fen=c2.fen(), us=self.us, pieces=[]), side_depth=0, chase_depth=self.chase_depth + 1, ctx=ctx, min_stake=1)
        ag2._chase_q = q2
        res3 = ag2._chase_route(r_k.move)
        if res3["gap"] is not None or res3["killed"] is not None or res3["value"] < 1 - dbal:
            return None
        v = dbal + res3["value"]
        reads = frozenset({ref.from_square, ref.to_square, r2.move.to_square, r_k.move.from_square, r_k.move.to_square})
        part = f"after {san_x}: {r_k.san} asked again on the board after the exchange ({'; '.join(res3['parts'][:2])}) -> {v}"
        text = (f"{r_k.san} fails to {san_ref} (the refuter: the {pc_ref.symbol().upper()} on {G.name(ref.from_square)}); exchange it first -- {r2.san} {san_x}: the "
                f"{c2.piece_at(x.to_square).symbol().upper()} on {G.name(x.to_square)} cannot make that move; {r_k.san} is asked again after it")
        return dict(value=v, part=part, facts=list(res3.get("facts", [])), reads=reads, text=text)

    def _note_resume(self, q: R.Question, r: "R.Route | None", res: dict) -> None:
        """52.17 (5): record where the clock cut: the question key, the position, the route and the reply it stopped at, the time mark and the fuse left (the 52.13 GAP memo keeps cut_left too)."""
        self.ctx.resume.append(dict(question=R.question_key(q), text=q.text, fen=_pos_key(self.board.fen()), route=r.move.uci() if r is not None else None,
                                    san=r.san if r is not None else None, reply=res.get("reply"), mark=round(time.monotonic() - self.ctx.t0, 3) if self.ctx.t0 is not None else None,
                                    cut_left=self.ctx.left, depth=self.chase_depth))

    def _chain_gap(self, q: R.Question, why: str) -> None:
        """64.1: the chase of `q` stops by the CHAIN'S STATE (the clock, the tempo allowance): not a fact of the position, so it is never a kill. The route that asked is a GAP (the budget_hits
        count the agenda as cut, `_cut` names the reason), the question is not 'chased'."""
        self.ctx.chain_hits += 1
        self.ctx.chain_gap = why
        self.ctx.chain_line = list(self.ctx.line)                # plan 66.4: the line at the stop (a resume reads it again, one more level, only while it is still forcing)
        self.lines.append(f"  chase: not started -- {why} (a GAP, never a kill)")
        self.state.gaps.append(f"Q{q.id} {q.text}: chain stop: {why}")
        q._chased = False

    def _mark_chain(self, res: dict) -> None:
        """plan 66.4: a route GAP by a chain stop keeps the line at the stop (the resume reads it) and the chain state it was cut under (its memo serves only that state)."""
        if res["gap"].startswith("chain stop"):
            res["stop_line"] = list(self.ctx.chain_line)
            res["chain_state"] = self._chain_state()

    def _cut(self, san: str, san_r: str, inner: bool = False, inn: "dict | None" = None) -> str:
        """The GAP text of a route cut at a reply: the chain's stop (64.1: the inner answer had a chain stop and no budget cut), the TIME when the deadline passed (52.17 (5)), else the fuse."""
        if inn is not None and inn.get("chain") and not inn.get("cut"):
            return f"chain stop: {san} not followed past {san_r} ({self.ctx.chain_gap})"
        if self.ctx.out_of_time():
            return f"out of time: {san} not followed past {san_r}"
        if self.ctx.judged:                                      # plan 61 C2: the judge cut it (a GAP like the clock's: timed, with a resume point)
            self.ctx.judged = False
            return f"out of time: {san} cut by the judge past {san_r}"
        return f"chase fuse: {san} not followed past {san_r}" + (" (the inner agenda was cut by the fuse)" if inner else "")

    def _promote_stands(self, m: chess.Move, P2: "AC.Pos") -> str | None:
        """52.16 (4): a promote route ends AT ONCE at a reply that leaves the question's stopper defect standing and brings no new fix: the same stopper (or blocker) of the
        promotion chased, on the same square, still reaches, and no forcing fix of its defects exists there that the question did not already have. The pawn is still on its
        square (the route is not its own push). Cut by the defects and the fixes, not by a count; the inner agenda (every root again) is not run. -> why or None."""
        q = self._chasing
        pawn = q.pattern.get("pawn") if q is not None else None
        if pawn is None or m.from_square == pawn:
            return None
        b2 = chess.Board(P2.fen())
        pc = b2.piece_at(pawn)
        if pc is None or pc.piece_type != chess.PAWN or pc.color != self.me:
            return None
        if RC.RACE and b2.turn == self.me and RC.race_claim(b2, pawn) is not None:      # plan 62.1: the race word binds on the reply's board: the stopper defect that stands is not a stop
            return None
        defects, _facts = R.promote_defects(b2, pawn)
        old = {d.key() for d in q.defects if d.kind in ("stopper", "blocked")}
        standing = [d for d in defects if d.key() in old]
        if not standing:
            return None
        saved = R._ids[0]
        dummy = R.new_question("promote", q.stake, dict(q.pattern))
        have = {r.move.uci() for f in q.fixes for r in R.all_routes(f)}
        new = set()
        for d in defects:
            f = R.promote_fixes(b2, pawn, d, dummy)
            new |= {r.move.uci() for r in R.all_routes(f)} - have
        R._ids[0] = saved
        if new:
            return None
        d0 = standing[0]
        return f"the {d0.kind} defect stands ({d0.why}) and no fix is new there"

    def _static_line(self, san: str, san_r: str, dbal: int, claim_their: Claim, parry_note: str, lost: list) -> None:
        """52.14: the trace line of a static leaf whose threats were asked as save roots of the inner agenda (printed by the root agenda)."""
        line = f"depth {self.chase_depth}: {san} after {san_r}: {dbal:+d} stands as a balance; " + ("; ".join(x for x in (parry_note, "no parry for " + ", ".join(f"their {cl.what} wins {cl.value}" for cl in lost) if lost else "") if x))
        if line not in self.ctx.static_lines:
            self.ctx.static_lines.append(line)

    def _target(self) -> int:
        """52.23 (7): what a route of this agenda must reach: 1, or with NEED_PROPAGATION the agenda's own need (`min_stake`)."""
        return max(1, self.min_stake) if self._t(NEED_PROPAGATION, 3) else 1

    def _standing_kill(self, uci: str, P2: "AC.Pos", dbal: int, target: int = 1) -> tuple | None:
        """52.13 (4): a reply of theirs killed an earlier route of this root with no claim of ours; the kill stands for this route when the position this reply leaves
        differs from that one only where the inner agenda's reads were not (the reads are recorded; unknown reads never transfer) and the reply took no less.
        -> (why, route san, reads)."""
        for P_old, dbal0, reads, san0, target0 in self.ctx.reply_kills.get(uci, ()):
            if reads is None or dbal >= 1 or dbal0 < dbal or target0 > target:          # the inner agenda read the roots of stake >= target0 - dbal0: enough for any reply that took at least as much, for a route that needs at least as much
                continue                                            # 64.2: AND that needs at least as much (W0VFO: a kill made for the need of several was served to Qb7 Qxa7, which needs 1)
            if not (reads & frozenset(P_old.diff(P2))):
                return ("no claim of ours and no material gain", san0, sorted(reads))
        return None

    def _closed_by_rule(self, P2: "AC.Pos", dbal: int, mate1: bool | None = None) -> tuple | None:
        """52.13 (4): a reply that the RULES close, no inner agenda: (a) we have a mate in one (some legal move of ours checkmates); (b) a capture of ours wins by the
        exchange at least what the route still needs (`need` = 1 - dbal) and THEIR agenda with the move passed to them finds no claim. `P2` = the position after the reply
        (our move). -> (route value for this reply, why) or None."""
        if (P2.mate_in_one() if mate1 is None else mate1):
            return BD.MATE_VALUE, "mate in one"
        if dbal >= 1 or P2.is_check():
            return None
        need = 1 - dbal
        best = P2.exchange()
        if best < need:
            return None
        if self._t(NEED_AWARE_CLOSE, 3) and dbal + best < self.min_stake:       # 52.23 (2): a rule closes a reply only at a value that can still rescue the route this agenda serves (its need,
            return None                                             # `min_stake`); a lower bound below it decides nothing, so the reply is ASKED (zBouR: Qh1+ Kd2 closed at the rook, 5, when 6 was needed; Rxf2+ Kc1 Rxc2+ wins the queen)
        q = self._chase_q
        if q is not None and q.goal == "win" and q.pattern.get("target") is not None and dbal + best < q.stake:
            fix = self._unmet_fix(P2, q)                            # 52.22 (1): the exchange reaches the route's need but not what the goal asks
            if fix:
                self.ctx.unmet += 1
                line = f"not closed by rule: the goal ({q.text}, stake {q.stake}) is not met by the exchange ({dbal + best:+d}); a forcing fix remains ({fix})"
                if line not in self.ctx.open_lines:
                    self.ctx.open_lines.append(line)
                return None
        t, _cl = self._their_pass(P2.fen())
        if t:
            return None
        if CLOSE_AFTER_CAPTURE:                                   # 2026-10-20 (kcQil, Kaveh: "Qxf3 is losing, you're still down a pawn"): the capture can LEAVE a duty -- Rxh6 "wins 3"
            t2, cl2 = self._their_after_capture(P2)               # but the h4 rook was the only guard of the d4 bishop, and Rxd4 wins it back. Their claims are read AFTER our
            if t2:                                                # capture too (their recaptures on the exchange square excluded: the exchange already counts them): their best
                self.ctx.overload_opens += 1                      # claim there means the rule cannot PROVE the close: the reply is asked, not closed. (Subtracting their claim is
                line = f"not closed by rule: after {cl2[0]} their {cl2[1].what} wins {t2} (the capturing piece left a duty)"      # unsound: their claim is THEIR lower bound, so
                if line not in self.ctx.open_lines:               # our remainder would be an upper bound -- kcQil came back at +1.)
                    self.ctx.open_lines.append(line)
                return None
        return dbal + best, f"exchange +{best} ({R.exchange_read(chess.Board(P2.fen()))}), nothing of theirs"

    # ------------------------------------------------------------------------------------------------------------ plan 61 C: the cost judge and the cycle terminal

    def _judge_on(self) -> bool:
        ctx = self.ctx
        return COST_JUDGE and ctx is not None and ctx.final is not None and ctx.deadline is not None

    def _spn(self) -> "float | None":
        """plan 61 C2: SECONDS PER NODE as measured in this run: the time spent in the root routes the judge let be read over their COST (so the cost model's error is absorbed). None until JUDGE_MIN_AGENDAS cost units were timed."""
        ctx = self.ctx
        n = ctx.judge_n + ctx.rate_n
        return (ctx.judge_t + ctx.rate_t) / n if n >= JUDGE_MIN_AGENDAS else None

    def _judge_stake(self) -> int:
        cq = self._chase_q
        return cq.stake if cq is not None else 1

    def _price(self, m: chess.Move, stake: int, limit: "int | None") -> tuple:
        """The first-level cost of the route `m` at the stake (64.3, speed only: a function of the position, the move and the stake, asked once per run; pricing stops at the limit, then the cost is a LOWER bound).
        -> (cost, [(parry SAN, child cost)], is it a lower bound)."""
        ctx = self.ctx
        ckey = (_pos_key(self.board.fen()), m.uci(), stake)
        hit = ctx.route_costs.get(ckey)
        if hit is None or (hit[2] and (limit is None or limit > hit[0])):
            cost_, kids_ = R.route_cost(self.board, m, stake, self.us, limit=limit)
            hit = ctx.route_costs[ckey] = (cost_, kids_, limit is not None and cost_ > limit)
        return hit

    @staticmethod
    def _route_need(cost: int, seen: "tuple | None", spn: float) -> float:
        """plan 64.3 / 65.1: THE PRICE OF A ROUTE IN SECONDS: first-level cost x the measured rate for a route never read; the measured seconds for one that finished before; at least the measured seconds
        (and the cost x rate) for one that was cut after t (it needs more than t)."""
        if seen is None:
            return cost * spn
        return seen[0] if seen[1] else max(cost * spn, seen[0])

    def _competitor(self, need: float, spn: float) -> "tuple | None":
        """plan 65.1: THE CHEAPEST ROUTE THE NEXT TIER WOULD READ at the root of the question being chased: tier k+1's route set is tier k's routes (re-read: the inner vocabulary is wider, the fuse doubles, nothing
        is carried) PLUS the routes the wider vocabulary adds (tier 1: the delayed threat, no safe move; tier 2: the threat of mate in two) -- every legal move that is forcing with tier k+1's flags -- each priced
        like the judge prices a root route (`_route_need`: the first-level cost x the measured rate, or the seconds it was measured to take). Only a price STRICTLY below `need` (the route under test) matters, so
        a candidate is priced only as far as that. -> (price in seconds, cost, san, why) of the cheapest such candidate, or None (no tier driver, the last tier, no rate yet: nothing is priced, or no
        candidate is cheaper). The candidates are found once per (position, stake, tier)."""
        ctx, q = self.ctx, self._chase_q
        k = ctx.tier
        if spn <= 0 or q is None or ctx.held is None or self.chase_depth != 0 or k >= TIER_COUNT - 1 or (q.goal == "save" and "checkers" in q.pattern):
            return None
        key = (self.pk, q.stake, k)
        cands = ctx.competitors.get(key)
        if cands is None:
            b = self.board
            dl, ns = DELAYED_THREAT, NO_SAFE_MOVE
            cands = []
            for m in R._moves(b, self.us):
                why = R.forcing_why(b, m, q.stake, self.us, mate_threats=True, mate2=k + 1 >= 2, delayed=dl and k + 1 >= 1, nosafe=ns and k + 1 >= 1)
                if why:
                    cands.append((m, b.san(m), why))
            ctx.competitors[key] = cands
        best = None
        top = need                                                     # the price a candidate must stay strictly below
        for m, san, why in cands:
            seen = ctx.route_secs.get((self.pk, m.uci(), q.stake))
            if seen is not None and seen[1]:
                price, cost = seen[0], None                            # finished before: its measured seconds, no pricing
            elif seen is not None and seen[0] >= top:
                continue                                               # cut after t >= the bar: it needs more than the bar
            else:
                limit = int(top / spn)                                 # cost x rate must stay below the bar: price the cost only up to it
                if limit < 1:
                    continue
                cost, _kids, lower = self._price(m, q.stake, limit)
                if lower:
                    continue
                price = self._route_need(cost, seen, spn)
            if price < top:
                best, top = (price, cost, san, why), price
        return best

    def _judge_begin(self, m: chess.Move, promote: bool, cheap: bool, override) -> tuple:
        """plan 61 C2 at a ROOT route: the cost of the route (1 + the cost of the unread child of each parry), printed; read on while cost x seconds_per_node <= the time left of this route's slice,
        else the route is a GAP (value 0, never a kill). -> (the GAP result or None, the token `_judge_end` closes)."""
        ctx = self.ctx
        if cheap or override or self.chase_depth != 0 or not self._judge_on():
            return None, None
        san = self.board.san(m)
        spn0 = self._spn()
        now0 = time.monotonic()
        limit = JUDGE_PRICE_CAP                                          # no rate yet: price up to this many cost units (a lower bound above it: such a route is the biggest of the root's, read on or cut later by the measured rate); the cost above which the judge cuts the route whatever it is: need > pool/2 and need > equal (see below)
        if spn0 is not None and spn0 > 0:
            limit = int(max(max((ctx.root_end or ctx.final) - now0, 0.0) / 2, max(ctx.deadline - now0, 0.0)) / spn0) + 1
        hit = self._price(m, self._judge_stake(), limit)
        cost, kids = hit[0], hit[1]
        ctx.judge_routes += 1
        shown = ", ".join(f"{k} {c}" for k, c in kids[:6]) + (", ..." if len(kids) > 6 else "")
        what = f"cost {cost} ({len(kids)} parries: {shown})"
        now = time.monotonic()
        spn = self._spn()
        pool = max((ctx.root_end or ctx.final) - now, 0.0)             # what the root's share has left for ALL its open routes
        equal = max(ctx.deadline - now, 0.0)                             # ... the equal slice plan 52.21 gives this route
        mkey = (_pos_key(self.board.fen()), m.uci(), self._judge_stake())
        seen = ctx.route_secs.get(mkey)                                  # 64.3: MEASURED cost -- a route read before (an earlier tier or question) carries its seconds
        measured = None if seen is None else seen[0]                       # ... finished in t: it needs t again; cut after t: it needs MORE than t (a lower bound: the slice below is twice the need, as iterative deepening)
        if spn is None:                                                  # no rate yet: read on, with a slice long enough to measure one
            slice_ = max(equal, min(pool / 4, 0.5))
            need = measured
            if need is not None:
                slice_ = max(equal, min(2 * need, pool / 2))
        else:                                                            # the route is read iff it fits in half of what is left (the rest stays for the other routes); it is given the time it needs (twice the estimate), at least its equal slice
            need = self._route_need(cost, seen, spn)                       # (the prediction is first-level cost x the rate ONLY for a route never read)
            slice_ = max(equal, min(2 * need, pool / 2))
        mkey_c = (_pos_key(self.board.fen()), m.uci(), self._judge_stake())
        comp = self._competitor(need, spn) if COMPETE and spn is not None and need is not None and mkey_c not in ctx.exempt and mkey_c not in ctx.opened else None       # (tier k stops opening NEW routes: one already open is read on; a route the previous tier deferred is read now: deferred once, never twice)
        if comp is not None and not judge_next(comp[0], need) and not ctx.held():       # plan 65.1: the next tier's cheapest route is priced below this one (and the status is not kept: the next tier opens)
            ctx.compete_cuts += 1
            ctx.judge_cuts += 1
            ctx.budget_hits += 1
            ccost = f"cost {comp[1]}, " if comp[1] is not None else "measured, "
            self.lines.append(f"    judge: {san} cut ({what}, {need:.2f} s): tier {ctx.tier + 1}'s {comp[2]} ({comp[3]}; {ccost}{comp[0]:.2f} s) is cheaper -> GAP, read in tier {ctx.tier + 1}, in price order")
            res = dict(value=0, killed=None, gap=f"out of time: {san} cut by the judge (tier {ctx.tier + 1}'s {comp[2]} is cheaper: {comp[0]:.2f} s against {need:.2f} s)", parts=[], lemmas=[])
            ctx.root_gaps[(_pos_key(self.board.fen()), m.uci(), self._judge_stake())] = (cost, san)
            ctx.deferred.add(mkey_c)
            return res, None
        if need is not None and need > pool / 2 and need > equal:
            self.lines.append(f"    judge: {san} cut ({what}, {need:.1f} s needed, {pool:.1f} s left of the root's share) -> GAP")
            ctx.judge_cuts += 1
            ctx.budget_hits += 1
            res = dict(value=0, killed=None, gap=f"out of time: {san} cut by the judge ({what}, {need:.1f} s needed, {pool:.1f} s left)", parts=[], lemmas=[])
            ctx.root_gaps[(_pos_key(self.board.fen()), m.uci(), self._judge_stake())] = (cost, san)
            return res, None
        ctx.opened.add(mkey_c)
        ctx.deadline = min(now + slice_, ctx.final)
        self.lines.append(f"    judge: {san} read on ({what}, " + (f"{need:.1f} s needed, " if need is not None else "no rate yet, ") + f"{ctx.deadline - now:.1f} s for it, {pool:.1f} s left of the root's share)")
        return None, (time.monotonic(), ctx.agendas, cost, mkey)

    def _judge_end(self, tok, res: dict) -> None:
        """plan 61 C2: the route read to its end (a kill or a value, not a GAP) adds its time and its inner agendas to the rate; the cuts the judge made inside it are printed."""
        ctx = self.ctx
        if self.chase_depth == 0 and ctx.c_lines:
            self.lines += ctx.c_lines
            ctx.c_lines = []
        if tok is None:
            return
        t0, _a0, cost, mkey = tok
        if res.get("gap") is None:
            ctx.root_gaps.pop(mkey, None)                                # plan 65.2: read to its end: no longer open
        else:
            ctx.root_gaps[mkey] = (cost, mkey[1])
        el = time.monotonic() - t0
        prev = ctx.route_secs.get(mkey)
        ctx.route_secs[mkey] = (el, True) if res.get("gap") is None else (max(el, prev[0]) if prev is not None and not prev[1] else el, False)      # 64.3: what this route cost (finished) or at least cost (cut)
        ctx.judge_t += time.monotonic() - t0                             # (a route cut by the clock counts too, with the time it was given: else a slice too short to finish a route never learns the rate)
        ctx.judge_n += cost

    def _judge_node(self, fen: str) -> bool:
        """plan 61 C2: the judge asked at a node of the route (the position before an inner agenda opens) with the REMAINING cost of the chain: the number of parries of our best forcing move there
        (a leaf closed by rule or with nothing forcing costs 0). True = cut (the route becomes a GAP, never a kill)."""
        ctx = self.ctx
        ctx.judged = False
        if not self._judge_on() or ctx.hand is not None:
            return False
        spn = self._spn()
        if spn is None:
            return False
        cost, mv = R.node_cost(chess.Board(fen), self._judge_stake(), self.us)
        left = max(ctx.deadline - time.monotonic(), 0.0)
        if cost * spn <= left:
            ctx.judge_reads += 1
            return False
        ctx.judged = True
        ctx.judge_cuts += 1
        ctx.c_lines.append(f"    judge: {self.board.san(self._route_move) if self._route_move is not None else '?'} cut after {self._reply_san} (remaining cost {cost}" + (f", best forcing {mv}" if mv else "") + f", {cost * spn:.1f} s needed, {left:.1f} s left) -> GAP")
        return True

    def _cycle_claim(self, fen: str, pk: str, need: int) -> "dict | None":
        """plan 61 C4: THE CYCLE TERMINAL. `fen` (our move) repeats a position of this line (an inner agenda still open, the root) or of the game history: when we are BEHIND (balance < 0) the repetition
        is a PROVED claim of kind 'draw' worth minus the balance (it brings the result to level; value + balance = 0, Lean `Ideas.Forcing.repetition_level`). Sound because the route that reached it
        read EVERY reply at each step of the cycle (the AND of `_chase_route_body`) and the repeating move is ours to make: the position is ours to move there, as it was the first time. Ahead or level:
        no claim, as before. -> the inner answer or None."""
        ctx = self.ctx
        bal = AC.Pos(fen).balance(self.me) - 64
        if bal >= 0:
            return None
        label = "the root position" if pk == ctx.root_pk and pk not in ctx.active_label else ctx.active_label.get(pk, "the game history" if pk in ctx.history else "the line")
        now = f"{self.board.san(self._route_move)} {self._reply_san}" if self._route_move is not None and self._reply_san else "the line"
        line = f"the position after {now} repeats {label if label.startswith('the root') or label.startswith('the game') else 'the one after ' + label}"
        cl = Claim("draw", "repetition", -bal, "repetition", line, goal="draw")
        ctx.cycles += 1
        ctx.c_lines.append(f"    closed: repetition ({line}) -> draw, value {-bal:+d} (minus the balance {bal:+d})")
        return dict(value=-bal, claim=cl, asked=0, need=need, cut=False, reads=None, threats=[], promote=None)

    def _parry_label(self, m: chess.Move, uci: str, closed: tuple) -> tuple:
        """plan 61 A2 (Lean `Ideas.Forcing.closed_binds`): a reply the rules CLOSED, checked against the THREAT of the route (`regress.threat_of` from the route's reason): when the reply does not
        parry it (the threat still binds on the reply's own board and the reply is no counter-threat) the line says so -- "the threat ... still binds (+v)". Same value, same decision:
        only the reason text changes. Off unless PARRIES."""
        if not PARRIES or closed is None:
            return closed
        ctx, b = self.ctx, self.board
        cq = self._chase_q
        stake = cq.stake if cq is not None else 1
        key = (_pos_key(b.fen()), m.uci(), stake)
        if key not in ctx.route_threats:
            why = R.forcing_why(b, m, stake, self.us, mate_threats=True, mate2=ctx.tier >= 2)
            ctx.route_threats[key] = R.threat_of(b, m, why, self.us)
        th = ctx.route_threats[key]
        if th is None or th.kind not in ("mate", "mate2", "threat", "pass-claim"):
            return closed
        ctx.parry_seen += 1
        after = b.copy(stack=False)
        after.push(m)
        r = chess.Move.from_uci(uci)
        if r not in after.legal_moves:
            return closed
        d = after.copy(stack=False)
        d.push(r)
        v = 0 if d.is_check() else R._binds(d, th, stake, self.me)
        if v and not R._counter(after, r, stake, after.turn):
            ctx.parry_closed += 1
            return closed[0], f"the threat ({th.why}) still binds (+{v}); {closed[1]}"
        return closed

    @staticmethod
    def _closed_tag(why: str) -> str:
        return "by the threat" if why.startswith("the threat (") else "by rule"

    def _principal_captures(self, P2: "AC.Pos") -> list:
        """64.0: the PRINCIPAL LINES of the best static exchange on P2 (our move): one per square whose static exchange is the best (a tie is a choice of ours) -> [(square, [(our capture's SAN, the
        board after it, does the line END with it, the exchange's CREDIT to them there)])]. A line is the legal captures on the square, the cheapest piece first, each side going on while going on
        gains (the swap-list's optimal stop). The credit is f[i+1], what the swap-list already gives them from that board on (0 after the capture that ends the line)."""
        b = chess.Board(P2.fen())
        me = b.turn
        best, sqs = 0, []
        for x in chess.SQUARES:
            p = b.piece_at(x)
            if p is None or p.color == me or p.piece_type == chess.KING or not b.is_attacked_by(me, x):
                continue
            v = G.see(b, x)
            if v > best:
                best, sqs = v, [x]
            elif v == best and v > 0:
                sqs.append(x)
        lines = []
        for sq in sqs:
            c, gains, sans, boards = b.copy(stack=False), [], [], []
            while len(sans) < 12:
                caps = [m for m in c.legal_moves if m.to_square == sq]
                if not caps or c.piece_at(sq) is None:
                    break
                m = min(caps, key=lambda m: (G.worth(c.piece_at(m.from_square)), m.uci()))
                gains.append(G.worth(c.piece_at(sq)))
                sans.append(c.san(m))
                c.push(m)
                boards.append(c.copy(stack=False))
            n = len(sans)
            f = [0] * (n + 1)                                        # f[i] = what the side making capture i gets from i on (it may stop, so >= 0)
            for i in range(n - 1, -1, -1):
                f[i] = max(0, gains[i] - f[i + 1])
            taken = 0
            while taken < n and gains[taken] - f[taken + 1] > 0:
                taken += 1
            if taken:
                lines.append((sq, [(sans[i], boards[i], i == taken - 1, f[i + 1]) for i in range(taken) if i % 2 == 0]))      # "last" = the line ENDS with this capture of ours (no recapture follows: their free move is next)
        return lines

    def _their_after_capture(self, P2: "AC.Pos") -> tuple:
        """64.0: THEIR agenda after EACH capture of ours on a principal line of the best static exchange on P2 (our move): -> (the value of their best claim, (our capture's SAN, that claim)) or (0, None).
        A claim of theirs landing on the exchange square is excluded only when it gains no more than the exchange's credit to them there (the swap-list already priced that recapture); a landing that
        gains MORE is counted (the discovery position: after Qxa8, Qxa8 Bxa8 gxf6 wins the knight back -- a true claim that lands on a8). After the capture that ends the line the credit is 0, so every
        positive claim counts. What is read is their FREE move after each of our captures, the last one included (pos2: Nxf4's close read their agenda after Bxf4 only, and 3.Qxf4 Bg5 pins the queen to the king). Equal best exchanges are a choice of ours: the line with the LEAST claim of theirs
        is the one that proves the close (FORK2: Ne3 Rc2, Nxf1 allows Rxc4+ but Nxc2 does not). Memoised by position in the run's `theirs` table."""
        lines = self._principal_captures(P2)
        if not lines:
            return 0, None
        least = None
        for sq, caps in lines:
            out = (0, None)
            for san, b, last, credit in caps:
                if not any(True for _ in b.legal_moves):
                    continue
                key = ("after-capture", _pos_key(b.fen()), chess.square_name(sq), credit)
                hit = self.ctx.theirs.get(key)
                if hit is None:
                    ag = self._their_agenda(b)
                    cls = [c for c in (ag.state.claims if ag is not None else []) if not (c.move and c.move[2:4] == chess.square_name(sq) and c.value <= credit)]      # (a recapture worth no more than the exchange's credit IS the exchange: the swap-list priced it)
                    cl = max(cls, key=lambda c: c.value) if cls else None
                    hit = self.ctx.theirs[key] = (cl.value if cl is not None and cl.value > 0 else 0, cl)
                if hit[0] > out[0]:
                    out = (hit[0], (san, hit[1]))
            if least is None or out[0] < least[0]:
                least = out
            if least[0] == 0:
                break
        return least

    def _unmet_fix(self, P2: "AC.Pos", q: R.Question) -> "str | None":
        """52.22 (1): the goal's piece is still on its square on P2 (our move), it is not won by the exchange, and `fixes_of` finds a forcing fix of its defect there
        (remove the defender with tempo, attack it with tempo, trap it, discover). -> the fix moves, or None (then the reply closes by rule as before)."""
        t = q.pattern["target"]
        pc0 = self.board.piece_at(t)
        b2 = chess.Board(P2.fen())
        pc2 = b2.piece_at(t)
        if pc0 is None or pc2 is None or pc2.color == self.me or pc2.piece_type != pc0.piece_type:
            return None
        defects = R.material_defects(b2, self.us, t)
        if not defects:
            return None
        saved = R._ids[0]
        dummy = R.new_question("win", q.stake, dict(q.pattern))
        sans = []
        for d in defects:
            for f in R.fixes_of(d, dummy, b2, self.us):
                sans += [r.san for r in R.all_routes(f) if r.san not in sans]
        R._ids[0] = saved
        return ", ".join(sans[:3]) if sans else None

    def _log(self, key: tuple, rep: str, outcome: str, dbal: int, inner: int | None, reads, reason: str | None) -> None:
        """52.13 (4): one row per reply read, for the plan 53 ordering prior (the prior itself is not built). Buffered; written by the root agenda when it ends."""
        if self.ctx.tag is None:
            return
        self.ctx.rows.append(dict(id=self.ctx.tag, depth=self.chase_depth, fen=key[0], route=key[1], reply=rep, outcome=outcome, dbal=dbal, inner=inner,
                                  reads=sorted(reads) if reads is not None else None, reason=reason))

    def _their_pass(self, fen: str) -> tuple:
        """52.13 (3): THEIR agenda on the position `fen` (our move) with the move passed to them, memoised by FEN across the run: -> (the value of their best claim, the claim) or (0, None)."""
        c = G.with_turn(chess.Board(fen), not self.me)
        key = _pos_key(c.fen())
        hit = self.ctx.theirs.get(key)
        if self.ctx.sched is not None:
            self.ctx.sched.leaf("their_pass", self, key, hit is not None)          # plan 56: their agenda is a leaf task of the node that asked (it chases nothing)
        if hit is None:
            ag = self._their_agenda(c) if any(True for _ in c.legal_moves) else None
            cl = max(ag.state.claims, key=lambda x: x.value) if ag is not None and ag.state.claims else None
            hit = self.ctx.theirs[key] = (cl.value if cl else 0, cl)
            if ag is not None:
                self.ctx.their_ags[key] = ag
            f = self.ctx.facts.add("claim" if cl else "kill", ("theirs", key), not self.me, f"their agenda with the move passed: " + (f"{cl.what} wins {cl.value}" if cl else "no claim"), value=hit[0], reads=None,
                                   stake=self._cur.stake if self._cur is not None else 0, depth=self.chase_depth, under=f"Q{self._cur.id} at depth {self.chase_depth}" if self._cur is not None else f"depth {self.chase_depth}", pk=key, portable=False,
                                   payload=dict(reads=None if ag is None or not getattr(ag, "all", None) or any(q.reads is None for q in ag.all) else frozenset().union(*(q.reads for q in ag.all))))      # 52.21 (2): what their agenda read, kept apart from `reads` (the listener's index) for the claims that stand on it
            self.ctx.their_facts[key] = f.id
        self._touch(self.ctx.their_facts.get(key))
        return hit

    def _addresses(self, q: R.Question, m: chess.Move, why: str) -> tuple:
        """52.17 (2): A CHASE ROUTE MUST ADDRESS A DEFECT OF ITS OWN GOAL. A forcing move is followed only when it is (a) forcing by a CLAIM of its own (a threat of mate, a capture or
        a threat worth the stake: 52.16 (3)), (b) the goal's own move or a step of its reach, or (c) a CHECK that fixes a defect of the goal: covers the flight, takes the piece that
        takes the checker, attacks the target (a fork), deflects the defender (every reply is the defender's), takes the stopper / blocker, or drives the king off (item 3).
        A check that fixes nothing of the goal is the walk of checks, not a route. -> (accepted, the note naming the reason).
        The note is the trace line: "Rh8+: not a route of mate h8 (fixes no defect: d7 stays a flight)"."""
        b, me = self.board, self.me
        san = b.san(m)
        if self.chase_depth < RULE2_FROM.get(q.goal, 99):      # the ROOT's own chase keeps every forcing move (its routes are the first level, 52.12); the WALK of checks is below it
            return True, ""
        if why.startswith("fix:") or why != "check":
            return True, ""
        goal = {"mate": f"mate {G.name(q.pattern['at'])}" if q.pattern.get("at") is not None else "mate", "win": f"win {G.name(q.pattern['target'])}" if q.pattern.get("target") is not None else "win", "promote": f"promote {G.name(q.pattern['pawn'])}" if q.pattern.get("pawn") is not None else "promote"}.get(q.goal, q.goal)
        if q.goal == "mate" and (q.pattern.get("move") == m.uci() or m.uci() in (q.pattern.get("reach") or ())):
            return True, ""
        hit = self.ctx.routes.get((_pos_key(b.fen()), m.uci()))
        if hit is not None and hit["gap"] is not None:             # a GAP is never a kill: the route the fuse cut under an earlier goal stays open for the later goals (0BHbB's Qb8+, 04Qx2's Be8+)
            return True, ""
        c = b.copy(stack=False)
        c.push(m)
        if any(ch != m.to_square for ch in c.checkers()):          # a DISCOVERED check (or a double check): the move opens a line of another piece onto the king -- a new threat from the vacated line (Kaveh's source 2; the word comes in 52.18 (2))
            return True, ""
        replies = list(c.legal_moves)
        left = []
        for d in q.defects:
            if d.kind == "flight" and d.sq is not None and c.is_attacked_by(me, d.sq):
                return True, ""
            if d.kind == "checker takeable" and m.to_square in d.by:
                return True, ""
            if d.kind in ("defended", "checker takeable") and replies and all(r.from_square in d.by for r in replies):
                return True, ""
            if d.kind == "not attacked" and m.to_square in c.attackers(me, d.sq):
                return True, ""
            if q.goal == "win" and q.pattern.get("target") in c.attacks(m.to_square) and d.kind in ("not attacked", "defended"):
                return True, ""
            if q.goal == "win" and d.kind in ("not attacked", "defended") and G.skewer_after(c, me, q.pattern.get("target")) == m.to_square:     # 52.23 (4): the skewer
                return True, f"{san}: a route of {goal}: a check through their king, the line runs on to {G.name(q.pattern['target'])} (Ideas.Skewer.skewer_attack)"
            if q.goal == "promote":
                why_fix = self._fixes_promote_defect(q, d, m, c)
                if why_fix:
                    return True, f"{san}: a route of {goal}: {why_fix}"
            left.append(d)
        stays = "; ".join(f"{d.sq and G.name(d.sq) + ' ' or ''}{ {'flight': 'stays a flight', 'checker takeable': 'stays takeable', 'interposition': 'stays open', 'not attacked': 'stays unattacked', 'defended': 'stays defended', 'blocked': 'stays blocked', 'stopper': 'stays a stopper', 'reach': 'stays out of reach'}.get(d.kind, 'stands') }" for d in left[:3]) or "no defect"
        return False, f"{san}: not a route of {goal} (a check that fixes no defect: {stays})"

    def _fixes_promote_defect(self, q: R.Question, d: R.Defect, m: chess.Move, c: chess.Board) -> "str | None":
        """52.17 (2), (3): a check that fixes a defect of the promotion: takes the blocker / stopper / attacker it names, DRIVES THEIR KING OFF the square the defect is about (a check
        after which every king reply leaves it unattacked by the king), or is a STEP of that fix: whatever the king replies, a check that drives it off exists there (Rh8+ Kd7, then
        Rd8+). -> the reason, or None."""
        b, me = self.board, self.me
        if m.to_square in d.by:
            return f"it takes the {b.piece_at(m.to_square).symbol().upper()} on {G.name(m.to_square)}"
        ek = b.king(not me)
        if ek is None or not ((d.kind == "blocked" and d.sq == ek) or (d.kind in ("path attacked", "promo square attacked", "stopper") and ek in d.by)):
            return None
        x = d.sq
        if R.drives_off(b, m, x, ek):
            return f"it drives their king off {G.name(x)}"
        reps = list(c.legal_moves)
        if not reps:
            return None
        steps = []
        for r in reps:
            c2 = c.copy(stack=False)
            san_r = c2.san(r)
            c2.push(r)
            k2 = c2.king(not me)
            nxt = [m2 for m2 in R._moves(c2, self.us) if k2 is not None and R.drives_off(c2, m2, x, k2)]
            if not nxt:
                return None
            steps.append(f"after {san_r}, {c2.san(nxt[0])}")
        return f"a step of driving their king off {G.name(x)} ({'; '.join(steps[:3])} drives it off)"

    def _relabel_lifted(self, q: R.Question, lifted: list) -> None:
        """plan 62.1c (Kaveh 2026-10-22): the king-only verdict is PROVED and said the king is no stopper; the unlifted read may still end the question as "stopped: ..." (a static estimate of another piece).
        That kill would say something false, so the question stays a GAP in the trace and the state: 'the chase did not close it: unfinished'. The decision treats it as any GAP."""
        if q.status != "killed" or not str(q.reason).startswith("stopped:"):
            return
        why = "lifted: their king cannot stop it (race.king_verdict); the chase did not close it: unfinished (the unlifted read said " + q.reason + ")"
        if self.kills and self.kills[-1][0] == q.id:
            self.kills.pop()
        for k in range(len(self.lines) - 1, -1, -1):
            if self.lines[k].startswith(f"  -> KILLED Q{q.id}: stopped:"):
                self.lines[k] = f"  -> GAP Q{q.id}: {why}"
                break
        q.status, q.reason = "gap", why
        self.state.gaps.append(f"Q{q.id} {q.text}: {why}")

    def _lift_routes(self, b: chess.Board, sq: int, facts: dict) -> list:
        """plan 62.1c: the pawn's push as a route of a promote question whose kill was lifted (their king is no stopper by the proved analysis)."""
        path = facts["path"]
        push = chess.Move(sq, path[0], promotion=chess.QUEEN if len(path) == 1 else None)
        if push not in b.legal_moves:
            return []
        return [R.Route(push, b.san(push), "push: their king is no stopper")]

    def _irreversible(self, uci: str) -> bool:
        """plan 66.5: an event that never comes back (the fifty-move rule's) on the chain: a capture by EITHER side (the route in hand or their reply `uci`), or a pawn move OF OURS (the plan: "a capture or a
        pawn move of ours"; their pawn push is a reply like any other -- resetting on it let their pawn checks extend our chain: eSJqH's Qa5 e4+ ran out of its slice)."""
        b, m = self.board, self._route_move
        if b.is_capture(m) or b.piece_type_at(m.from_square) == chess.PAWN:
            return True
        after = b.copy(stack=False)
        after.push(m)
        return after.is_capture(chess.Move.from_uci(uci))

    def _since_after(self, uci: str, dbal: int) -> int:
        """The chase clock of the agenda after the route in hand and their reply `uci`: 0 after an irreversible event that is not bad for us (66.5: a capture by either side with dbal >= 0, or our pawn
        move), else one more move of ours. Without CLOCK_IRREVERSIBLE: 0 after OUR capture that is not bad for us or our pawn move (62.1c). The checks evaded in a row reset on ANY capture (`_resets`)."""
        m = self._route_move
        if CLOCK_IRREVERSIBLE:                                   # a capture by EITHER side that is not bad for us (the balance after the route and the reply not below where it was: dbal >= 0), or a pawn move of ours;
            return 0 if (self._irreversible(uci) and (dbal >= 0 or self._pawn_move(m))) else self.since_capture + 1      # (their capture that wins material is THEIR progress: eSJqH and xhSWo ran out of time reading on after it)
        return 0 if (self.board.is_capture(m) and dbal >= 0) or self._pawn_move(m) else self.since_capture + 1

    def _resets(self, uci: str) -> bool:
        """plan 66.5: the checks evaded in a row start again after an irreversible event (a capturing check, our capture or pawn move): the stop exists for perpetual-check loops, which capture nothing."""
        return EVADED_RESET and self._irreversible(uci)

    def _pawn_move(self, m: chess.Move) -> bool:
        """plan 62.1c: a pawn move of ours resets the chase clock (irreversible, like a capture)."""
        return CLOCK_PAWN and self.board.piece_type_at(m.from_square) == chess.PAWN

    def _step_forcing(self, step: tuple) -> tuple:
        """plan 66.4 (Lean `Ideas.Forcing.Step` / `stillForcing`): one step of the line -- our move named a threat (`regress.threat_of` of its forcing reason) and their reply PARRIED it (the threat no longer
        binds on the reply's board, or the reply is itself a counter-threat: `regress._binds`, `regress._counter`, the reading of `regress.parries`). -> (forcing?, why)."""
        hit = self.ctx.forcing_steps.get(step)
        if hit is not None:
            return hit
        fen, mu, ru, stake = step
        b = chess.Board(fen)
        m, r = chess.Move.from_uci(mu), chess.Move.from_uci(ru)
        san = b.san(m)
        why = R.forcing_why(b, m, stake, self.us, mate_threats=True, mate2=self.ctx.tier >= 2)
        th = R.threat_of(b, m, why, self.us)
        if th is None:
            out = (False, f"{san} names no threat")
        else:
            after = b.copy(stack=False)
            after.push(m)
            d = after.copy(stack=False)
            d.push(r)
            v = 0 if d.is_check() else R._binds(d, th, stake, self.me)
            out = (True, "") if not v or R._counter(after, r, stake, after.turn) else (False, f"{after.san(r)} does not parry {san} ({th.why})")
        self.ctx.forcing_steps[step] = out
        return out

    def _still_forcing(self, line: list) -> tuple:
        """plan 66.4: every step of the line forcing (`_step_forcing`). -> (True, '') or (False, why at the first quiet step)."""
        for k, st in enumerate(line, 1):
            ok, why = self._step_forcing(st)
            if not ok:
                return False, f"move {k}: {why}"
        return True, ""

    def _extension(self) -> int:
        """plan 66.4: the extra levels a chain stop here is given: the resume pass's extension while the line to this position is still forcing, else 0."""
        if self.ctx is None or not self.ctx.extend:
            return 0
        ok, _why = self._still_forcing(self.ctx.line)
        if ok:
            self.ctx.ext_used += 1
            return self.ctx.extend
        self.ctx.ext_quiet += 1
        return 0

    def _resume_begin(self, q: R.Question, r: R.Route) -> int:
        """plan 66.4: before a ROOT route is read, the extension it is given (a route the previous tier stopped by the chain, the line still forcing): the trace says 'resumed Rxf2+ (chain stop at move 4;
        every reply a parry; t s left; the allowance +1)'. -> the extension in force before (restored after the route)."""
        ext0 = self.ctx.extend
        if self.chase_depth != 0 or not self.ctx.resume_ext:
            return ext0
        e = self.ctx.resume_ext.get((self._carry_key(q), r.move.uci()))
        if e is None:
            return ext0
        self.ctx.extend = max(ext0, e["ext"])
        self.ctx.resumed += 1
        left = max((self.ctx.final or time.monotonic()) - time.monotonic(), 0.0)
        self.lines.append(f"    resumed {r.san} (chain stop at move {e['moves']}; every reply a parry; {left:.1f} s left; the allowance +{self.ctx.extend})")
        return ext0

    def _chain_state(self) -> tuple:
        """plan 66.4 (mh1q8): what a chain stop depends on -- the clock, the checks evaded in a row, the exchange level, the allowance of the goal in hand, the extension. A chain-stop GAP memoised under one
        chain state is not the answer under another (a mate goal's allowance is not the queen goal's)."""
        return (self.since_capture, int(self.evaded), self.free_level, tuple(self.ctx.mate_levels[-1:]) if self.ctx is not None else (), self.ctx.extend if self.ctx is not None else 0)

    def _chase(self, q: R.Question) -> bool:
        """52.12: the chase of a question that would be killed. -> True when a route CLAIMS (the question is verified)."""
        evading = self._t(EVASION_CHASE, 3) and q.goal == "save" and "checkers" in q.pattern          # 52.23 (3): the save-king question of an agenda in check: its routes are the evasions, all forced (`_verify` calls the chase too: the switch is read HERE)
        if self.ctx is None or (q.goal not in ("mate", "win", "promote") and not evading) or self.min_stake >= BD.MATE_VALUE or (self.chase_depth >= 2 and (q.goal == "promote" or (q.goal == "win" and not self._t(DEEP_WIN_CHASE, 3)))):      # a mate-only read (the reply already gained, or lost too much for anything but a mate) does not chase on
            return False                                               # 52.23 (5): a win question chases at every depth; its checks must address a defect (52.17 (2)) -- the depth-two gate was a numbered limit
        if self.chase_depth and self.since_capture >= self._clock() and not evading and self.since_capture >= self._clock() + self._extension():      # 52.23 (8): the chase clock -- a capture that is not bad for us is one-way progress (material never comes back); without one for
            self.ctx.clock_stops += 1                                   # CHASE_CLOCK of our moves the chain goes nowhere and ends here (a hyperparameter to tune, Kaveh 2026-10-20)
            self._chain_gap(q, f"the chase clock: {self.since_capture} moves of ours without a capture that is not bad for us")      # 64.1: a hyperparameter's stop is a GAP, never a kill
            return False
        if evading and self.evaded and self.evaded > self._extension():      # 52.23 (3): ONE intermezzo check is followed (66.4: one more per level of a resumed, still-forcing line); a second check of theirs against our evasion is not read (their chain, not our line) --
            self._chain_gap(q, "a second check of theirs in a row (not followed)")      # 64.1: a GAP, never the kill "no claim of ours" it was (2zB89: 1.Qxa5 Qxa1+ 2.Rf1 Qd4+). Following the chain
            return False                                               # costs lK9WA's Nxc6 at 10 s (the king dance after Rxb6+): the cost judge (64.3) is what may follow it
        if self.best is not None and self.best[0] >= q.stake:      # 52.13 (2c): a question is chased only when its stake exceeds the best claim so far (stop (a) would end the agenda anyway)
            self.lines.append(f"  chase: not asked, the best claim ({self.best[2].what}, {self.best[0]}) is at least the stake {q.stake}")
            return False
        q._tier_dep = True                                         # plan 66.1: the chase reads the tier's vocabulary: this question's answer may differ in a wider tier
        if self.ctx.out_of_time():                                 # 52.17 (5): past the deadline a question is asked only for what closes at the static leaf: no chase
            self.lines.append("  chase: out of time -- not started (a GAP, never a kill)")
            self.ctx.budget_hits += 1                                # the agenda this question belongs to is cut: its route is a GAP, never refuted by what was not chased
            self._note_resume(q, None, dict(gap="chase not started"))
            self.state.gaps.append(f"Q{q.id} {q.text}: out of time: chase not started")
            return False
        allow = None
        if q.goal in ("mate", "win") and self._t(GOAL_JUDGE, 3):             # 52.23 (6): TEMPO ARITHMETIC ENDS THE CHAIN (the goal-progress judge, Kaveh 2026-10-20: "mating the king is like mating the bishop").
            fresh = q.goal == "win" and self.chase_depth > 0 and self.since_capture == 0      # A pattern with d defects needs at most d prep moves of ours, each fixing one (regress.py's tempo arithmetic); a fix that
            top = self.ctx.mate_levels[-1] if self.ctx.mate_levels else None                  # itself needs two moves (drive the king onto the skewer line, then skewer) says so (`steps`). A chain that has spent
            parent = None if fresh or top is None or q.stake > top[1] else top[0]             # (a BIGGER goal found on the chain is a new goal with its own distance: zBouR's queen under the f2 pawn's chase)
            dist = max([1, len(q.defects)] + [f.pattern.get("steps", 1) for f in q.fixes])      # them with the goal unmet is the walk. For a WIN goal a capture that is not bad for us is one-way progress (Kaveh
            allow = dist if parent is None else min(parent - (0 if self.free_level else 1), dist)      # 2026-10-20): the allowance starts again (zBouR: Qh1+ Rf1 Qxf1+ Kd2 Rxf2+); a mate's distance is its defects alone.
            if allow <= 0 and allow + self._extension() <= 0:          # 66.4: a resumed line that is still forcing gets one more level of the allowance per pass
                self.ctx.mate_walks += 1
                self._chain_gap(q, f"tempo arithmetic: the chain has spent the prep moves the goal's defects allow ({len(q.defects)} defect(s), {parent} level(s) left under the {top[1]}-stake goal)")      # 64.1: a heuristic's stop is a GAP unless it is a proof (the judge is not a lemma)
                return False
        b = self.board
        routes, notes = [], []
        for m in R._moves(b, self.us):
            if evading:                                          # 52.23 (3): in check every legal move is a route (it is forced: the check must be answered), none is quiet
                pc = b.piece_at(m.to_square)
                routes.append(R.Route(m, b.san(m), f"capture {G.worth(pc)}" if pc is not None else "evade"))
                continue
            why = R.forcing_why(b, m, q.stake, self.us, mate_threats=True, notes=notes, mate2=self.ctx.tier >= 2,      # plan 59 tier 2: a mate-in-two threat is forcing too
                                delayed=DELAYED_THREAT and self.ctx.tier >= 1 and not self.chase_depth, nosafe=NO_SAFE_MOVE and self.ctx.tier >= 1 and not self.chase_depth)      # plan 61 A3 / A4: tier 1+, the root's chase only
            if why:
                if why.startswith("delayed threat"):
                    self.ctx.delayed_routes += 1
                elif why.startswith("no safe move"):
                    self.ctx.nosafe_routes += 1
                ok, note = self._addresses(q, m, why)
                if ok:
                    routes.append(R.Route(m, b.san(m), why))
                    if note:
                        notes.append(note)
                else:
                    self.ctx.unaddressed += 1
                    notes.append(note)
        if q.goal == "promote" or (q.goal == "win" and self._t(DEEP_WIN_CHASE, 3)):      # 52.15 (2): the fix routes of the defects (take the stopper or the blocker, push with check) are routes too, at every depth; 52.23 (4): a win's named fixes too (the skewer, the drive onto its line)
            seen = {r.move for r in routes}
            for f in q.fixes:
                for r in R.all_routes(f):
                    if r.move not in seen:
                        seen.add(r.move)
                        routes.append(R.Route(r.move, r.san, f"fix: {r.why}"))
        if q.goal == "promote" and getattr(q, "_lift_routes", None):      # plan 62.1c: the pawn's push when their king is no stopper (their other pieces are replies)
            seen = {r.move for r in routes}
            routes += [r for r in q._lift_routes if r.move not in seen]
        if q.goal == "promote" and getattr(q, "_support_routes", None):   # plan 67.4: a support placement after which no weak moment of ours is reachable (the rook behind the pawn)
            seen = {r.move for r in routes}
            routes += [r for r in q._support_routes if r.move not in seen]
        if q.goal == "promote" and not self.chase_depth:        # 52.15 (2): plus a move that attacks a named STOPPER of the question ("threat on the stopper"), or offers the pawn to it
            routes += self._stopper_routes(q, [r.move for r in routes], notes)
        if self.chase_depth and not evading:                    # inner levels follow the most forcing moves only: a check, a capture, a mate threat (the quiet "threat n" moves are the root's)
            follow = ("check", "threat of mate", "capture") + (("threat",) if self.ctx.tier >= 1 else ()) + (("push",) if q.goal == "promote" else ())      # 62.1c: a lifted promote question follows the pawn's push      # plan 59 tier 1 lifts the cut for the quiet threats worth the stake (and the tier-2 mate-in-two threat)
            routes = [r for r in routes if r.why.removeprefix("fix: ").startswith(follow)]
        if self.chase_depth >= 2 and q.goal == "win":           # 52.23 (5): below the first inner level a win question follows only the NAMED FIXES of its defects that CHECK (the skewer, the fork-check:
            fix_moves = {r.move for f in q.fixes for r in R.all_routes(f)}     # the reply is forced); every other forcing move is the root's and the first level's (the named fixes that do not check still cost 0LRTF 4x)
            routes = [r for r in routes if r.move in fix_moves and b.gives_check(r.move)]
        routes.sort(key=lambda r: (0 if r.why.removeprefix("fix: ") == "check" else 1 if r.why.removeprefix("fix: ").startswith("capture") else 2))      # the most forcing first: a check, a capture, a threat (stable inside)
        if self.chase_depth == 0 and self.ctx.resume_set is not None:      # plan 66.4: a resume pass reads only the chain-stopped routes of this question, in price order
            want = self.ctx.resume_set.get(self._carry_key(q), [])
            routes = [r for u in want for r in routes if r.move.uci() == u]
        self.promote_stake = q.stake if q.goal == "promote" else 0
        if not routes:
            self.lines.append("  chase: no forcing move on this board")
            q._chased = True                                     # nothing to follow: the chase ran to its end
            return False
        for n in notes[:6]:
            self.lines.append("  chase: " + n)
        self.lines.append(f"  chase: make them move -- {len(routes)} forcing route(s): " + ", ".join(f"{r.san} ({r.why})" for r in routes[:12]) + (" ..." if len(routes) > 12 else ""))
        done, cut, route_facts = [], False, []
        results: dict = {}
        pool = self.ctx.left
        self._chasing = q if q.goal == "promote" else None
        self._chase_q = q
        if allow is not None:
            self.ctx.mate_levels.append((allow, q.stake))
        try:
            return self._chase_routes(q, routes, cut, done, route_facts, results, pool)
        finally:
            if allow is not None:
                self.ctx.mate_levels.pop()

    def _chase_routes(self, q: R.Question, routes: list, cut: bool, done: list, route_facts: list, results: dict, pool: int) -> bool:
        """52.23 (6): the body of `_chase` from the routes on (split so the mate allowance is pushed and popped around it)."""
        # 52.17 (4): CHEAPEST REPLY FIRST. Before any route opens an inner agenda, every route's replies are read at the static leaf and what the rules close is closed (a standing
        # kill, a mate or stalemate against us, a mate in one, the exchange with nothing of theirs): a route killed there never draws a share of the fuse, and the routes still open
        # share it in order.
        cheap_res = {}
        stopped = False                                          # 64.2: a route ended by a chain stop (clock, tempo allowance, second check)
        root_final = None                                        # 52.23 (11): the root's share of the time (set at the first open route)
        for i, r in enumerate(routes):
            got = self._chase_route(r.move, promote=q.goal == "promote", cheap=True)
            if not got.get("pending"):
                cheap_res[i] = got
        for i, r in enumerate(routes):
            saved = self.ctx.left
            open_left = sum(1 for j in range(i, len(routes)) if j not in cheap_res)
            share = max(saved // max(open_left, 1), 1) if saved > 0 else 0
            if self.chase_depth == 0 and saved > 0 and i not in cheap_res and self.ctx.final is None:             # the root's fuse is shared FAIRLY (52.23 (1): by TIME when seconds are given) by the routes of this chase: each gets an equal share of what is left (a chain of checks must not spend what Qd5 needs: 0DoFB)
                self.ctx.left = share
            before = self.ctx.left
            if i in cheap_res:
                res = cheap_res[i]
            elif self.ctx.out_of_time():                           # 52.17 (5): the open routes end as GAPs
                res = dict(value=0, killed=None, gap=f"out of time: {r.san} not started", parts=[], lemmas=[])
                self.ctx.budget_hits += 1
                if ESTIMATE and q.goal != "promote":
                    self._estimate_route(r.move, res)                # plan 61 B2
            else:
                if self.chase_depth == 0 and self.ctx.final is not None:       # ... and the time is shared FAIRLY like the fuse: each open route gets an equal slice of what is left
                    now = time.monotonic()
                    if root_final is None:                                     # 52.23 (11): and, like the fuse, the ROOT gets at most half of what is left while another root can beat the best claim (BEKFf: the first mate root ate the whole minute)
                        cap = any(t[4].goal in ("mate", "win", "promote") and (self.best is None or self.best[0] < t[4].stake) for t in self.heap)
                        root_final = now + (self.ctx.final - now) / 2 if cap else self.ctx.final
                    self.ctx.deadline = now + max(min(root_final, self.ctx.final) - now, 0) / max(open_left, 1)
                    self.ctx.root_end = min(root_final, self.ctx.final)               # plan 61 C2: the end of the root's share of the time (the judge's pool)
                    self.ctx.hand = None                           # (a new slice: the route in hand is the one this slice sees)
                ext0 = self._resume_begin(q, r)                       # plan 66.4: a route the last tier stopped by the chain, still forcing: one more level of the allowance
                res = self._chase_route(r.move, promote=q.goal == "promote")
                self.ctx.extend = ext0
                if self.chase_depth == 0 and self.ctx.final is not None:
                    self.ctx.deadline = self.ctx.final
                    self.ctx.hand = None
            if self.chase_depth == 0 and i not in cheap_res:
                # 52.16 (2): a route is CHARGED at most its share. The frames open when the fuse runs out finish their questions (that is how 08pSj's Rf1+ and 0VSUT's Qxe5 are decided),
                # but that finishing is free: charged in full it took the share of every later route (0LRTF: 239 questions on a share of 120, then Rd7 got 1).
                self.ctx.left = saved - min(max(before - self.ctx.left, 0), share) if saved > 0 else self.ctx.left
            route_facts += res.get("facts", [])
            results[i] = res
            if res["gap"] and res["gap"].startswith("out of time"):
                self._note_resume(q, r, res)
            if res["gap"]:
                if res["gap"].startswith("chain stop"):
                    if self.chase_depth == 0 and not res.get("reused") and RESUME:      # plan 66.4: a root route stopped by the chain: the resume pass may read it again
                        self.ctx.chain_gaps.append(dict(key=self._carry_key(q), route=r, uci=r.move.uci(), stake=q.stake, line=list(res.get("stop_line") or []), gap=res["gap"], mkey=(self.pk, r.move.uci(), q.stake)))
                    stopped = True                                   # 64.2: a chain stop is deterministic (the same chain state stops at the same place whatever the budget): it does not make the chase a CUT one
                else:
                    cut = True
                self.lines.append(f"    {r.san}: GAP, {res['gap']}" + (" (seen before with no more fuse: not re-run)" if res.get("reused") else ""))
                self.state.gaps.append(f"Q{q.id} {q.text}: {res['gap']}")
            elif res["killed"]:
                self.lines.append(f"    {r.san}: killed -- {res['killed']}")
            elif res["value"] >= 1:
                self.lines.append(f"    {r.san}: " + " / ".join(res["parts"][:6]) + (f" / ... {len(res['parts'])} replies" if len(res["parts"]) > 6 else "") + f" => value {res['value']}")
                done.append((res["value"], r, res))
            else:
                self.lines.append(f"    {r.san}: value {res['value']} (no gain on the worst reply)")
            self._resumed_line(r, res)
        if q.goal == "win" and not self.ctx.out_of_time():
            self._refuter_fixes(q, routes, results, done, pool)
        self._record_roots(routes, results)
        self._chasing = None
        self._chase_q = None
        q._chased = not cut                                      # 52.19: a kill is "chased" only when no route was cut (a GAP); a cut chase is as weak as no chase
        q._chain_stopped = stopped                               # 64.2: ... and a KILL is not filed as chased to its end when a route was a chain stop (a claim is: it is true whatever the other routes did)
        self._qdeps = list(dict.fromkeys(route_facts))           # a kill stands on every route's facts ...
        if self.chase_depth == 0 and pool > 1 and any(t[4].goal in ("mate", "win", "promote") and (self.best is None or self.best[0] < t[4].stake) for t in self.heap):
            # ... and a ROOT question is charged at most half of the pool it started with, while another root that can still beat the best claim is open: by stake order the
            # question that matters most gets the larger share, never all of it (0GJec: the promotion's chase took 618 of 570 and every later root was read with nothing)
            charged = pool - max(self.ctx.left, pool - pool // 2)
            self.ctx.left = max(self.ctx.left, pool - pool // 2)
            self.lines.append(f"    fuse: this root is charged {charged} of the pool {pool} (routes at most their shares, the root at most half while another root can beat the best claim)")
        self.lines.append(f"    (chase: {self.ctx.agendas} inner agendas, {self.ctx.asked} inner questions, {self.ctx.budget_hits} fuse hit(s), {self.ctx.gap_reused} gap route(s) reused, fuse left {self.ctx.left})")
        self.lines.append("    chase ended: " + ("the FUSE was hit on at least one route (a GAP, never a kill)" if cut else "a chain stop on at least one route (a GAP, never a kill)" if stopped else "every route closed (claim on every reply or a refuting reply)"))
        if ESTIMATE:
            self._est_claims(q, routes, results, done)
        if not done:
            return False
        done.sort(key=lambda t: (-t[0], t[1].move.uci()))
        value, r, res = done[0]
        self._qdeps = list(res.get("facts", []))                 # ... a claim only on the facts of the route that decided it
        self._extra_reads = frozenset(res.get("extra_reads", ()))     # 52.22 (3): and on the squares of the refuter it exchanged
        self._carried_route = res.get("carried", 0)              # plan 66.2: ... and on the replies it served from an earlier tier
        net = value
        kind = "mate" if value >= BD.MATE_VALUE else ("promote" if q.goal == "promote" and not res.get("material") else "save" if q.goal == "save" else "win")
        if CYCLE and kind != "mate" and any("repetition" in x for x in res["lemmas"]) and value <= -(AC.Pos(self.board.fen()).balance(self.me) - 64):
            kind = "draw"                                        # plan 61 C4: the route brings the result to level by a repetition (proved: every reply of the cycle was read)
        parts = " / ".join(res["parts"][:4]) + (" / ..." if len(res["parts"]) > 4 else "")
        lemma = "chase: step_after+step_least over " + ", ".join(sorted(set(res["lemmas"]))[:3]) if res["lemmas"] else "chase: step_after+step_least"
        claim = Claim(kind, r.san, value, lemma, f"{r.san}; {parts}", move=r.move.uci())
        self._stamp(claim, q)
        claim.fixes = claim.fixes + ("chase",)
        self.state.claims.append(claim)
        q.status = "verified"
        self.lines.append(f"  -> CLAIM {kind} {r.san} (value {value}), lemma {lemma}, line {claim.line}")
        if self._better(net, 0, claim):
            self.best = (net, 0, claim)
            self.asked_at_claim = self.asked
        return True

    # ------------------------------------------------------------------------------------------------------------ kills, saves

    # ------------------------------------------------------------------------------------------------------------ plan 61 B: the proved / estimated label

    def _quiet_leaf(self, fen: str) -> bool:
        """plan 61 B2: is the leaf QUIET -- not in check, no forcing move of ours by `forcing_why` at the stake of the question being chased, and their pass threat 0 (`_their_pass`: their agenda
        with the move passed to them finds no claim)?"""
        b = chess.Board(fen)
        if b.is_check():
            return False
        cq = self._chase_q
        stake = cq.stake if cq is not None else 1
        if any(R.forcing_why(b, m, stake, self.us, mate_threats=True) for m in R._moves(b, self.us)):
            return False
        t, _cl = self._their_pass(fen)
        return t == 0

    def _est_no(self, why: str) -> None:
        self.ctx.est_why[why] = self.ctx.est_why.get(why, 0) + 1

    def _estimate_route(self, m: chess.Move, res: dict) -> None:
        """plan 61 B2 THE QUIET-STATUS ESTIMATE. A route that ended as a GAP (the fuse, the clock) keeps its flow exactly as before (a gap, counted, never a kill); in ADDITION, when every reply
        from the cut on is either closed by rule or a QUIET leaf, the route has an ESTIMATED value: the least of the replies read before the cut and, for each quiet leaf, its static status (material
        balance minus their pass threat = the balance change, their threat being 0 there). Only a value of at least 1 is kept (an estimate below the stake claims nothing). The label: proved False,
        confidence the PRODUCT of the leaves' calibrated confidences (`question_prior.estimate_confidence`), estimator 'quiet-status'. Reads nothing past the cut with an inner agenda."""
        if res.get("est_done") or res.get("gap") is None or res.get("killed") or self.chase_depth != 0:       # (the root's routes only: an inner agenda's cut shows up as the root route's gap, read from the reply on)
            return
        res["est_done"] = True
        b, me = self.board, self.me
        P0 = AC.Pos(b.fen())
        P1 = P0.push(m.uci())
        shift = P1.balance(me) - P0.balance(me)
        rows = [(shift + dbal, uci, chk, cap, rec, mate1) for uci, dbal, chk, cap, rec, mate1 in P1.replies()]
        rows.sort(key=lambda t: (0 if t[4] else 1 if t[2] else 2 if t[3] else 3, t[0], t[1]))
        before = res.get("gains", [])
        reply = res.get("reply")
        idx = 0 if reply is None else next((i for i, t in enumerate(rows) if t[1] == reply), -1)
        if idx < 0 or len(before) != idx:
            return self._est_no("replies read before the cut not recoverable")
        gains, confs, leaves = list(before), [], []
        for dbal, uci, chk, cap, rec, mate1 in rows[idx:]:
            P2 = P1.push(uci)
            if P2.legal_count() == 0:
                return self._est_no("a reply mates us / stalemate")  # a refutation for the rules to read, not a leaf to estimate
            closed = self._closed_by_rule(P2, dbal, mate1)
            if closed is not None:
                gains.append(closed[0])
                confs.append((True, 1.0))
                continue
            fen = P2.fen()
            if not self._quiet_leaf(fen):
                return self._est_no("a reply leaves a non-quiet leaf")      # a non-quiet gap stays a gap
            cls = "quiet:us" if chess.Board(fen).turn == me else "quiet:them"
            conf = QP.estimate_confidence(cls)
            gains.append(dbal)                                      # (material balance minus their pass threat, which is 0 on a quiet leaf)
            confs.append((False, conf))
            leaves.append(dict(fen=fen, value=dbal, status=P2.balance(me) - 64, cls=cls, confidence=conf, estimator="quiet-status", reply=uci, us=self.us))
        value = min(gains)
        if not leaves or value < 1:
            return self._est_no("estimate below the stake" if leaves else "no quiet leaf")
        from catspace.incubator.analyst.discovery.state import combine
        proved, conf = combine(confs)
        res["est"] = dict(value=value, proved=proved, confidence=conf, estimator="quiet-status", leaves=leaves, reply_parts=len(rows))
        self.ctx.est_routes += 1
        for lf in leaves:
            self.ctx.est_rows.append(dict(lf, route=m.uci(), route_san=b.san(m), root_fen=b.fen(), depth=self.chase_depth, leaned=False))

    def _est_claims(self, q: R.Question, routes: list, results: dict, done: list) -> None:
        """plan 61 B1/B2: the ESTIMATED claim of a question: its best route with an estimate, when it is worth more than the best PROVED route (a proved route at least as good needs no estimate).
        Filed in `state.estimated` and `best_est`, never in `claims` or `best`: the search reads the proved ones only."""
        cands = [(results[i]["est"]["value"], r, results[i]) for i, r in enumerate(routes) if results.get(i) and results[i].get("est")]
        if not cands:
            return
        cands.sort(key=lambda t: (-t[0], t[1].move.uci()))
        value, r, res = cands[0]
        if done and max(d[0] for d in done) >= value:
            return
        est = res["est"]
        kind = "save" if q.goal == "save" else "win"
        lemma = "chase: quiet-status estimate"
        parts = " / ".join(res["parts"][:4]) + (" / ..." if len(res["parts"]) > 4 else "")
        claim = Claim(kind, r.san, value, lemma, f"{r.san}; {parts}" + (" / " if parts else "") + f"{len(est['leaves'])} quiet leaf(s) read by their status", move=r.move.uci(),
                      proved=False, confidence=est["confidence"], estimator=est["estimator"])
        self._stamp(claim, q)
        claim.fixes = claim.fixes + ("chase",)
        self.state.estimated.append(claim)
        self.lines.append(f"  -> CLAIM ESTIMATED {kind} {r.san} (value {value}, confidence {claim.confidence:.2f} (product) by {claim.estimator}): not proved, {len(est['leaves'])} quiet leaf(s) closed by their status")
        if self.best_est is None or (value, 0) > self.best_est[:2] or ((value, 0) == self.best_est[:2] and claim.confidence > self.best_est[2].confidence):
            self.best_est = (value, 0, claim)

    def decision(self) -> Claim | None:
        """The claim that decides. A PROVED claim that keeps the status decides, and nothing estimated is looked at. Only when ESTIMATE found an estimated claim and no proved claim keeps the status
        (plan 59's rule: `status()` / `keeps()`), the best estimated claim that keeps it decides, and the trace says so: 'DECISION ESTIMATED: ...'. Read once."""
        if self._decision is not None:
            return self._decision[0]
        dec, est = (self.best[2] if self.best else None), False
        if ESTIMATE and self.best_est is not None and self.chase_depth == 0 and not (self.best is not None and self.best[0] >= BD.MATE_VALUE):
            st = getattr(self, "status_info", None) or self.status()
            if not self.keeps(st, self.best)[0] and self.keeps(st, self.best_est)[0]:
                dec, est = self.best_est[2], True
                self.lines.append(f"DECISION ESTIMATED: {dec.what} {dec.value:+d}, confidence {dec.confidence:.2f} by {dec.estimator}; no proved claim keeps the status")
        if dec is None and UNPROVED and self._root and self.chase_depth == 0 and self.ctx is not None and self.best_est is None:
            dec = self._unfinished_claim()                       # plan 65.3: no claim of any kind: the best unfinished line, marked unproved, never silently (the trace line above)
        self._decision = (dec, est)
        self._flush_estimates(dec if est else None)
        return dec

    def decision_estimated(self) -> bool:
        """plan 61 B1: did the decision lean on an estimated claim (the trace line 'DECISION ESTIMATED' was printed)?"""
        self.decision()
        return self._decision[1]

    def _flush_estimates(self, leaned: "Claim | None") -> None:
        """plan 61 B2: every estimated leaf of this root appended (ONE write) to artifacts/experiments/estimates_log.jsonl with whether the decision leaned on its route."""
        if not ESTIMATE or self.chase_depth != 0 or self.ctx is None or not self.ctx.est_rows:
            return
        import json
        root = self.board.fen()
        rows = []
        for r in self.ctx.est_rows:
            rows.append(json.dumps(dict(r, id=self.ctx.tag, set=self.ctx.est_set, leaned=bool(leaned is not None and r["root_fen"] == root and r["route"] == leaned.move and r["depth"] == 0))) + "\n")
        self.ctx.est_rows = []
        from catspace.io import paths
        from catspace.incubator.analyst.discovery import analyse as AN
        AN._append_once(paths.experiment(QP.EST_LOG), "".join(rows))

    def _stopper_routes(self, q: R.Question, have: list, notes: list) -> list:
        """52.15 (2): the extra routes of a promote question: a quiet move after which the moved piece attacks a STOPPER of the question (the defect's piece, a blocker of theirs too, not the king: a
        check is already a route) and is not lost by the exchange on its own square (the 52.13 (2d) rule) -- "threat on the stopper"; and the push of the promoting pawn
        itself onto a square a stopper (not the king) attacks, the pawn guarded so that the exchange there loses them the stopper -- "offers the pawn to the stopper"."""
        b, me = self.board, self.me
        stoppers = sorted({s for d in q.defects if d.kind in ("stopper", "blocked") for s in d.by if s != b.king(not me) and b.piece_at(s) is not None and b.piece_at(s).color != me})      # (the blocker of theirs is a stopper too)
        out = []
        for m in R._moves(b, self.us):
            if m in have or b.is_castling(m):
                continue
            c = b.copy(stack=False)
            c.push(m)
            hit = [s for s in stoppers if c.piece_at(s) is not None and c.is_attacked_by(me, s) and m.to_square in chess.SquareSet(c.attackers(me, s))]
            if hit and G.see_for(c, m.to_square, not me) <= 0:
                pc = c.piece_at(hit[0])
                out.append(R.Route(m, b.san(m), f"threat on the stopper: the {pc.symbol().upper()} on {G.name(hit[0])}"))
                continue
            if m.from_square == q.pattern["pawn"]:
                by = [s for s in stoppers if c.piece_at(s) is not None and s in c.attackers(not me, m.to_square)]
                if by and G.see_for(c, m.to_square, not me) <= 0:
                    pc = c.piece_at(by[0])
                    out.append(R.Route(m, b.san(m), f"offers the pawn to the stopper: the {pc.symbol().upper()} on {G.name(by[0])} takes, and the exchange there loses it"))
        return out

    def _kill(self, q: R.Question, reason: str, tempo: bool) -> None:
        q.status, q.reason = "killed", reason
        if q.reads is None and q.goal in ("mate", "win"):
            q.reads = R.reads_of(q, self.board)
        self.killed_tempo += 1 if tempo else 0
        self.kills.append((q.id, reason))
        self.lines.append(f"  -> KILLED Q{q.id}: {reason}")

    def _ask_save_king(self, q: R.Question, head: str) -> None:
        b = self.board
        ck = q.pattern["checkers"]
        q.defects = [R.Defect("in check", c, [], f"{b.piece_at(c).symbol().upper()} on {G.name(c)} gives check") for c in ck]
        self.lines.append(head + ": " + "; ".join(f"defect {d.kind} ({d.why})" for d in q.defects))
        takes = [R.Route(m, b.san(m), f"capture {G.worth(b.piece_at(m.to_square))}") for m in b.legal_moves if m.to_square in ck]
        others = [R.Route(m, b.san(m), "evade") for m in b.legal_moves if m.to_square not in ck]
        f1 = R.new_question("remove", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="take the checker")
        f1.routes = takes
        self.asked += 1
        self.lines.append(f"  -> Q{f1.id} {f1.text}: " + (", ".join(f"{r.san} ({r.why})" for r in takes) or "no capture of the checker"))
        q.fixes.append(f1)
        before = len(self.state.claims)
        if takes:
            self._verify(q, takes, mates_only=False)
        if len(self.state.claims) == before and others:
            f2 = R.new_question("cover", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="step away or interpose")
            f2.routes = others
            self.asked += 1
            self.lines.append(f"  -> Q{f2.id} {f2.text}: " + ", ".join(r.san for r in others))
            q.fixes.append(f2)
            self._verify(q, others, mates_only=False)
        if len(self.state.claims) == before and not self._decide_by_smallest_loss(q) and self._t(EVASION_CHASE, 3):
            self._chase(q)                                       # 52.23 (3): an INNER agenda in check chases its evasions (every legal move answers the check, so each is a route): zBouR's Rhxf2 cxb7+ Kb8 Rxf2 Qh1+

    def _ask_save_piece(self, q: R.Question, head: str) -> None:
        b = self.board
        sq = q.pattern["target"]
        parries = RS._threat_moves(b, self.us, [sq])
        q.defects = [R.Defect("hangs", sq, sorted(b.attackers(not self.me, sq)), f"they win {G.see(b, sq)} by the exchange on {G.name(sq)}")]
        self.lines.append(head + f": defect hangs ({q.defects[0].why})")
        f = R.new_question("remove", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="parry it, or check, or take as much")
        f.routes = [R.Route(m, b.san(m), "parry") for m in parries]
        self.asked += 1
        self.lines.append(f"  -> Q{f.id} {f.text}: {len(f.routes)} moves")
        q.fixes.append(f)
        before = len(self.state.claims)
        self._verify(q, f.routes, mates_only=False)
        if len(self.state.claims) == before and not self._decide_by_smallest_loss(q):
            self.state.gaps.append(f"Q{q.id} {q.text}: no parry binds a guaranteed gain and none could be read from their view")

    def _ask_save_threat(self, q: R.Question, head: str) -> None:
        """A root from THEIR agenda on the pass board: parries are our moves of the threatened pieces, checks, captures of the threatening piece, and moves that add
        an attacker of a threatened piece (all legal moves for a mate threat); each is verified by the binder and viewed by their agenda on the board after it."""
        b = self.board
        p = q.pattern
        m = chess.Move.from_uci(p["threat"])
        q.defects = [R.Defect("threatened", m.to_square, [m.from_square], f"their {p['what']} wins {p['value']} if we pass")]
        self.lines.append(head + f": defect threatened (their view: {p['what']} wins {p['value']}, lemma {p['lemma']}, {p['line']})")
        tg = p["targets"]
        parries = []
        for mv in b.legal_moves:
            if p["value"] >= BD.MATE_VALUE or mv.from_square in tg or b.gives_check(mv) or (b.is_capture(mv) and mv.to_square == m.from_square):
                parries.append(mv)
                continue
            c = b.copy(stack=False)
            c.push(mv)
            if any(len(c.attackers(self.me, t)) > len(b.attackers(self.me, t)) for t in tg):
                parries.append(mv)
        f = R.new_question("remove", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="parry the threat (move the target, check, take the threatener, add a guard)")
        f.routes = [R.Route(mv, b.san(mv), "parry") for mv in parries]
        self.asked += 1
        self.lines.append(f"  -> Q{f.id} {f.text}: {len(f.routes)} moves")
        q.fixes.append(f)
        if self.their is not None:                               # 52.14: a threat on a static leaf is parried by KEEPING the gain (the binder reads a gain of at least 1 from a balance <= 0, and we are ahead here)
            self._parry_keeps(q, f.routes)
            return
        before = len(self.state.claims)
        self._verify(q, f.routes, mates_only=False)
        if len(self.state.claims) == before and not self._decide_by_smallest_loss(q):
            self.state.gaps.append(f"Q{q.id} {q.text}: no parry binds a guaranteed gain and none could be read from their view")

    def _parry_keeps(self, q: R.Question, routes: list) -> None:
        """52.15 (3): a parry of a threat on a static leaf is VERIFIED by the binder's KEEP family (binder.verify_keep): the Lean theorem `delayed_balance_alive_d1` with the
        balance bound B = the balance now and the gain v = 0 -- after the named move and every reply of theirs, some move of ours and every reply, the balance is still at
        least B and our material alive. The claim's lemma is that theorem's name; the gain it keeps is `keep`, a balance (us to move) of at least 1. (52.14 verified the parry by
        material after it less THEIR best claim; that label, `their_view`, is gone.) A MATE threat is parried only by taking the threatening piece or by mating."""
        b = self.board
        P0 = AC.Pos(b.fen())
        if q.pattern.get("value", 0) >= BD.MATE_VALUE and q.pattern.get("threat"):
            # a MATE threat is not parried by a held balance (the tail does not see a deep mate: 0AgL8's Kc8 against Qa7+, then Bc6+ mates in 4): only taking the threatening
            # piece, or mating first, is a parry here
            src = chess.Move.from_uci(q.pattern["threat"]).from_square
            keep_routes = []
            for r in routes:
                c = b.copy(stack=False)
                c.push(r.move)
                if r.move.to_square == src or c.is_checkmate():
                    keep_routes.append(r)
            routes = keep_routes
        base = P0.balance(self.me)
        order = sorted(routes, key=lambda r: (-(P0.push(r.move.uci()).balance(self.me) - base), r.move.uci()))      # the parry that wins most now first; the first that binds is the claim
        notes = []
        if self.keep >= 1:
            for r in order:
                bind, note = BD.verify_keep(r.move, b, self.us)
                if bind is None:
                    notes.append(f"{r.san}: {note}")
                    continue
                claim = Claim("save", r.san, self.keep, bind.theorem, f"{r.san} (every reply: {len(bind.replies)} replies; the balance is held)", move=r.move.uci())
                self._stamp(claim, q)
                self.state.claims.append(claim)
                q.status = "verified"
                q.claimed = (0, claim, frozenset())
                self.lines.append(f"  -> PARRY {r.san} keeps {self.keep:+d} (binds {bind.schema}, {len(bind.replies)} replies), lemma {bind.theorem}")
                return
        self._kill(q, f"no parry keeps the gain ({len(order)} parr{'y' if len(order) == 1 else 'ies'} tried by the keep family" + (f": {'; '.join(notes[:3])}" if notes else "") + f"; the gain to keep is {self.keep:+d})" if order else "no parry move", tempo=False)

    # ------------------------------------------------------------------------------------------------------------ promotion (52.6)

    def _ask_promote(self, q: R.Question, head: str, lift: bool = True) -> None:
        """The defects of the pawn's path are read with the words; a stopper in time kills the question WITH THE DISTANCE REASON; no defect = unstoppable by the
        tempo arithmetic, and then the claim is made ONLY when the proved passer (Ideas.Passer2.unstoppablePasser2_sound, via proved_passers) fires; if it does
        not, the question ends as a GAP naming the hypothesis that fails."""
        b = self.board
        sq = q.pattern["pawn"]
        q.defects, facts = R.promote_defects(b, sq, lift=lift and self.chase_depth == 0)
        if facts.get("lifted") is not None and q.defects:           # plan 62.1c: the king holds (or a defect stands): the old verdict, every stopper, stands as it was
            q.defects, facts = R.promote_defects(b, sq)
        n, promo = facts["pushes"], facts["promo"]
        q.reads = R.reads_of(q, b) | frozenset(facts["path"]) | {sq}            # 52.16 (4): the pawn, its path (the promotion square included), the defects' squares and their pieces: what the question read
        kd = facts["king_dist"]
        self.lines.append(head + (": " + "; ".join(f"defect {d.kind} ({d.why})" for d in q.defects) if q.defects else
                                  ": no defect -- unstoppable by the tempo arithmetic" + (f" (their K needs {kd} to {G.name(promo)}, the pawn {n})" if kd is not None else "")))
        lifted = facts.get("lifted")
        if lifted is not None and lifted:
            self.lines.append("  lifted: " + "; ".join(lifted) + " (their other pieces are replies the chase reads; the kill is a GAP until it closes)")
        if lifted is not None and not q.defects:
            q._lift_routes = self._lift_routes(b, sq, facts)
        if self._plan_claim(q, sq):                               # plan 67: the plan's own weaknesses: closed by arithmetic (no weak moment reachable), or the support placements become routes
            return
        if q.defects:
            if self._race_claim(q, sq):                           # plan 62.1: the race word binds although a defect stands (the stopper the king cannot be: denied, overloaded)
                return
            for d in q.defects:
                f = R.promote_fixes(b, sq, d, q)
                rs = ", ".join(f"{r.san} ({r.why})" for r in f.routes)
                self.lines.append(f"  -> Q{f.id} {f.text}: " + (f"forcing: {rs}" if rs else "no forcing move"))
            self.asked += len(q.fixes)
            stop = [d for d in q.defects if d.kind == "stopper"]
            if stop:
                reason = "stopped: " + "; ".join(d.why for d in stop)
            else:
                reason = "tempo: " + "; ".join(f"{d.kind} {G.name(d.sq)}: {d.why}" for d in q.defects)
            if self._chase(q):                                   # 52.15 (2): a promotion that the defects stop is chased: make them move, ask the promote root again on each reply
                return
            self._kill(q, reason, tempo=True)
            return
        ps = dict(PV.proved_passers(b, bishops=True))
        if sq in ps:
            push = chess.Move(sq, facts["path"][0], promotion=chess.QUEEN if n == 1 else None)
            san = b.san(push)
            rest = [G.name(t) for t in facts["path"][1:-1]]
            line = f"{san}, then {', '.join(rest)}{', ' if rest else ''}{G.name(promo)}=Q within {ps[sq]} plies whatever they do ({n} pushes)" if n > 1 else f"{san} within {ps[sq]} ply"
            claim = Claim("promote", san, R.STAKE_PROMOTE, "unstoppablePasser2_sound", line, move=push.uci())
            self._stamp(claim, q)
            self.state.claims.append(claim)
            q.status = "verified"
            self.lines.append(f"  -> CLAIM promote {san} (value {claim.value}), lemma {claim.lemma}, line {claim.line}")
            if self._better(claim.value, 0, claim):
                self.best = (claim.value, 0, claim)
                self.asked_at_claim = self.asked
            return
        if self._race_claim(q, sq):                               # plan 62.1: no defect, the lemma does not fire (our sliders, say): the race word
            return
        ex = self._promotion_exchange(q, sq, facts)
        if ex is not None:
            return
        if lifted is not None:                                    # plan 62.1c: the kill is lifted: the chase proves it, or the verdict that stood before the lift is read again (never worse than without the lift)
            late = self.ctx is not None and self.ctx.deadline is not None and time.monotonic() >= self.ctx.deadline       # read without `out_of_time`, which marks the route in hand: past the deadline the lifted chase is not started
            if not late and self._chase(q):
                return
            q._lift_routes = None
            self.lines.append("  lifted kill: not closed by the chase; the unlifted verdict is read")
            self._ask_promote(q, head, lift=False)
            self._relabel_lifted(q, lifted)
            return
        h = PV.passer_v2(b, sq)
        bad = [k for k in PV.HYPS if not h[k]]
        why = []
        for k in bad:
            if k == "enemyPieces":
                why.append("enemyPieces: " + ", ".join(f"their {b.piece_at(s).symbol().upper()} on {G.name(s)}" for s in chess.SquareSet((b.rooks | b.queens | b.knights) & b.occupied_co[not self.me])))
            elif k == "bishopsFar":
                why.append("bishopsFar: " + ", ".join(f"their B on {G.name(s)}" for s in chess.SquareSet(b.bishops & b.occupied_co[not self.me])))
            elif k == "noSliders":
                why.append("noSliders: " + ", ".join(f"our {b.piece_at(s).symbol().upper()} on {G.name(s)}" for s in chess.SquareSet((b.rooks | b.bishops | b.queens) & b.occupied_co[self.me])))
            else:
                why.append(k)
        msg = (f"Q{q.id} {q.text}: the defects say unstoppable but the lemma unstoppablePasser2_sound does not fire -- failing hypothesis "
               + ("; ".join(why) if why else "none in the Python mirror (the Lean detector declined)") + " (the next lemma to prove)")
        q.status = "gap"
        q.reason = msg
        self.state.gaps.append(msg)
        self.lines.append(f"  -> GAP: {msg}")

    def _plan_claim(self, q: R.Question, sq: int) -> bool:
        """plan 67 (Lean `Ideas.Plan.plan_sound`): roll the pawn's plan forward without the opponent, list the weak moments it creates (capture, blockade, check
        squares with deadlines), and test every piece of theirs against them with the proved lower bounds. No weak moment reachable and no stalemate: the
        promotion is CLAIMED by arithmetic. Otherwise the trace names the reachable weaknesses; at the root (and one level in) the support placements after
        which no weakness of OURS is reachable (67.4: the rook behind the pawn) become quiet routes of the question (`_support_routes`), read by the chase."""
        if not PL.PLAN:
            return False
        b = self.board
        if not q.defects and sq in dict(PV.proved_passers(b, bishops=True)):
            return False                                         # the full-rules lemma (Ideas.Passer2.unstoppablePasser2_sound) fires below: it is the stronger word, it claims first
        pl = PL.read(b, sq)
        if not pl.moves:
            return False
        self.lines.append("  " + PL.trace(b, pl))
        if pl.closed and pl.moves[0].from_square == sq:         # (a plan that opens with a blocker's step-aside is not claimed as a promote move: a promote claim's move is the pawn's push)
            push = pl.moves[0]
            san = b.san(push)
            claim = Claim("promote", san, R.STAKE_PROMOTE, "plan_sound", f"{pl.line(b)}: {len(pl.weak)} weak moment(s), none reachable by any piece of theirs in time", move=push.uci())
            self._stamp(claim, q)
            self.state.claims.append(claim)
            q.status = "verified"
            self.lines.append(f"  -> CLAIM promote {san} (value {claim.value}), lemma {claim.lemma}, line {claim.line}")
            if self._better(claim.value, 0, claim):
                self.best = (claim.value, 0, claim)
                self.asked_at_claim = self.asked
            return True
        if self._root and self.chase_depth == 0 and not pl.immune:      # (the support search is the ROOT's: inside their agendas and the chase it cost 2 s of eSJqH's 15, 2026-10-22 profile)
            fixes = PL.support_fixes(b, sq)
            if fixes:
                q._support_routes = [R.Route(m, san, f"support: after it no weak moment of ours is reachable ({p2.n} moves; {len(p2.reach)} check square(s) left to the chase)") for m, san, p2 in fixes]
                self.lines.append("  support (plan 67.4): " + ", ".join(san for _m, san, _p in fixes))
        return False

    def _race_claim(self, q: R.Question, sq: int) -> bool:
        """plan 62.1 / 62.2: THE RACE WORD (Lean Ideas.Race.race_sound; regress.race_claim): when it binds for the pawn on `sq` the promotion is a PROVED claim (value 8) whose line names both counts and
        whose trace line prints the FRAME -- the pieces read, the pieces excluded and why (62.2, Ideas.Race.frame_sound_pass). Served only on a board where the pawn's owner is to move. -> True if the claim was made."""
        b = self.board
        if not RC.RACE or b.turn != self.me:
            return False
        rc = RC.race_claim(b, sq)
        if rc is None:
            return False
        san = b.san(rc.first)
        line = f"{san} -- {rc.text}"
        claim = Claim("promote", san, R.STAKE_PROMOTE, "Ideas.Race.race_sound", line, move=rc.first.uci())
        self._stamp(claim, q)
        self.state.claims.append(claim)
        q.status = "verified"
        self.lines.append(f"  -> CLAIM promote {san} (value {claim.value}), lemma {claim.lemma}, word race ({rc.text})")
        self.lines.append(f"     {rc.frame}")
        if self._better(claim.value, 0, claim):
            self.best = (claim.value, 0, claim)
            self.asked_at_claim = self.asked
        return True

    def _promotion_exchange(self, q: R.Question, sq: int, facts: dict) -> "Claim | None":
        """52.17 (3): THE PROMOTION THAT IS AN EXCHANGE. No static defect, the push promotes now (one push), and the passer lemma does not fire because a rook or a queen of
        theirs stands behind the pawn on its file (it attacks the promotion square once the pawn has left the line: an X-RAY through the pawn). The push is played (a queen) and
        the exchange on the promotion square is READ with their move (the x-ray static exchange of `Ideas.ExchangeLaw`, the mirror `G.see`; only when some LEGAL reply of theirs
        captures there): the promotion is worth the queen less the pawn, less what they win on the new queen -- 0LRTF: c8=Q+ Rxc8 Rxc8, 8 - (9 - 5) = 4, a rook for a pawn. A claim
        of kind `win` with that value (a promotion nets less than the passer lemma's 8 there); the composed statement (push, then the exchange) is a TODO in Schemas.lean like the
        chase claims. -> the claim, or None."""
        b, me = self.board, self.me
        if facts["pushes"] != 1:
            return None
        push = chess.Move(sq, facts["promo"], promotion=chess.QUEEN)
        if push not in b.legal_moves:
            return None
        san = b.san(push)
        c = b.copy(stack=False)
        c.push(push)
        if c.is_checkmate() or not any(m.to_square == facts["promo"] for m in c.legal_moves):
            return None
        lost = G.see_for(c, facts["promo"], not me)
        net = R.STAKE_PROMOTE - lost
        if lost <= 0 or net < 1:
            return None
        read = R.exchange_read(G.with_turn(c, not me))
        claim = Claim("win", san, net, "promotion_exchange: the push, then the x-ray exchange on the promotion square (Ideas.ExchangeLaw seeXB, mirrored by the Rust see)",
                      f"{san}; {read}: the promotion is worth {R.STAKE_PROMOTE}, less the {lost} they win on the new queen = {net}", move=push.uci())
        self._stamp(claim, q)
        self.state.claims.append(claim)
        q.status = "verified"
        q.claimed = (net, claim, frozenset(q.reads or ()) | {sq, facts["promo"]})
        self.lines.append(f"  -> CLAIM win {san} (value {net}), lemma {claim.lemma}, line {claim.line}")
        if self._better(net, 0, claim):
            self.best = (net, 0, claim)
            self.asked_at_claim = self.asked
        return claim

    def _ask_stop_pawn(self, q: R.Question, head: str) -> None:
        """THEIR passer, a claim of their agenda, as OUR save root: the routes are the stoppers' moves (our king runs toward the promotion square, a capture of
        the pawn); the binder verifies them as any save root; the tempo arithmetic says `forced` when our king's distance equals the pawn's pushes."""
        b = self.board
        p = q.pattern
        pw = p["promoter"]
        them = not self.me
        n = G.word(b, ("pushes", G.name(pw)), them)
        promo = R.pawn_path(b, pw, n)[-1]
        kd = G.word(b, ("dist", "L", ("promo", G.name(pw))), them)               # their pawn's words, read from its owner's side: L = our king
        q.defects = [R.Defect("promotes", promo, [pw], f"their {p['what']} promotes on {G.name(promo)} in {n} ({p['value']}, lemma {p['lemma']})")]
        self.lines.append(head + f": defect promotes (their view: {p['what']}, {p['line']})")
        arith = f"our K reaches {G.name(promo)} in {kd}, the pawn needs {n}"
        ksq = b.king(self.me)
        runs = []
        for m in b.legal_moves:
            if m.to_square == pw:
                runs.append(R.Route(m, b.san(m), f"capture {G.worth(b.piece_at(pw))}"))
            elif m.from_square == ksq:
                c = b.copy(stack=False)
                c.push(m)
                if G.word(c, ("dist", "L", ("promo", G.name(pw))), them) < kd:
                    runs.append(R.Route(m, b.san(m), "K runs back"))
        f = R.new_question("remove", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="stop it: our king runs toward the promotion square, or takes the pawn")
        f.routes = runs
        self.asked += 1
        self.lines.append(f"  -> Q{f.id} {f.text}: {arith}; " + (", ".join(f"{r.san} ({r.why})" for r in runs) or "no such move"))
        q.fixes.append(f)
        # WE move first on the real board and the pawn is promoted by its nth push, so we make n moves before it; the king stops it when it is next to the
        # promotion square then (it takes the new queen), i.e. distance <= n + 1 (the same inequality as the lemma's kingOutside, n < dist, for the defender
        # with n - 1 moves). Distance = n + 1 is exact: every move of ours is spent.
        if kd == n + 1:
            self.lines.append(f"     forced: every move of ours is spent stopping the pawn (distance {kd} = pushes {n} + 1: next to {G.name(promo)} after our {n} moves)")
        elif kd > n + 1:
            self.lines.append(f"     no stopper in time: distance {kd} > pushes {n} + 1")
        else:
            self.lines.append(f"     slack: distance {kd} <= pushes {n}: {n - kd + 1} spare move(s)")
        before = len(self.state.claims)
        if self.their is not None:                               # 52.14: on a static leaf the stopper keeps the gain or not
            self._parry_keeps(q, runs)
            return
        if runs:
            self._verify(q, runs, mates_only=False)
        if len(self.state.claims) == before:
            self.lines.append(f"  no route binds a guaranteed gain against the passer (the binder has no promotion tail); {arith}")
            if kd > n + 1:
                self.state.gaps.append(f"Q{q.id} {q.text}: no stopper in time ({arith}) and no capture binds")

    # ------------------------------------------------------------------------------------------------------------ outputs

    def facts_delta(self) -> dict:
        """52.19: this agenda's own counters of the store (derived, served, revalidated, foreground recomputations) and the background queue as it stands."""
        fs = self.ctx.facts
        now = fs.snapshot()
        out = {k: now[k] - self._fs0[k] for k in now}
        q = fs.queue()
        out["queue"], out["top"] = len(q), q[0].chain_stake if q else 0
        return out

    def facts_lines(self) -> list:
        """The trace of the two tiers and the justification graph of the deciding claim (52.19 (3)): both tiers' counts, then the chain of facts the decision stands on."""
        d = self.facts_delta()
        out = [f"facts: derived {d['derived']}, served from the store {d['served']} (flagged stale and revalidated {d['revalidated']}); foreground: recomputed {d['recomputed']} stale fact(s) of the live questions; "
               f"background queue: {d['queue']} (top stake {d['top']})"]
        fid = self.claim_fid.get(id(self.best[2])) if self.best is not None else None
        if fid is not None:
            out.append(f"justification graph of the decision ({self.best[2].what}):")
            out += ["  " + x for x in self.ctx.facts.graph_lines(fid)]
        return out

    def trace(self) -> str:
        return "\n".join(self.lines)

