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
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import reasoned as RS
from catspace.incubator.analyst.discovery import selfmine as SM
from catspace.incubator.analyst.discovery.state import Claim, State


DEPTH_FUSE = 30                                                 # 52.13 (2b): ... and a chase this many inner levels deep is cut the same way (python's stack: ~9 frames per level; the 10x fuse alone reached the recursion limit)
RULE2_FROM = {"mate": 2, "win": 1, "promote": 1}              # 52.17 (2): the chase depth from which a check must address a defect of its goal (the root's chase, depth 0, keeps every forcing move; a mate goal's checks are the first level of an inner agenda, its walk begins one level below)
FUSE = 10                                                       # 52.13 (2b): the chase's budget is a fuse at this multiple of the old question count


def _pos_key(fen: str) -> str:
    """The position without its move counters (board, turn, castling, en passant): the chase's memo key. A check chain that comes back to a position only the move number
    apart is a CYCLE (0DoFB: the same board at moves 32, 34, 36 ... each re-asked, 1033 inner questions)."""
    return " ".join(fen.split()[:4])


class ChaseCtx:
    """The state shared by the root agenda and its inner agendas (52.12): the FEN memo, the route memo, the shared question budget and the counters."""
    def __init__(self, facts: "FS.FactStore | None" = None):
        self.facts = facts if facts is not None else FS.FactStore()     # 52.19 (1): every claim and kill derived in the run, with its justifications; shared by the root agenda and every inner agenda
        self.their_facts: dict = {}               # 52.19: key of their agenda's position -> the fact id of its answer
        self.gap_facts: dict = {}                 # 52.19 (3): route key -> the id of the open gap fact (resolved when the route is followed to its end)
        self.memo: dict = {}                      # fen -> {value, claim, asked}: the inner agenda's best claim, asked once for the whole run
        self.routes: dict = {}                    # (fen, uci) -> the evaluated route (memoised: the same position is chased by many questions)
        self.active: set = set()                  # the FENs being asked (a cycle ends here)
        self.left = 0                             # 52.13 (2b): the FUSE of the root agenda: 10x its question budget, set once, spent by every chase and inner agenda under it
        self.depth_cuts = 0                       # ... of the fuse hits, the ones from the depth cap (the stack), not the question count
        self.theirs: dict = {}                    # 52.13 (3): fen (their move) -> (value, claim) of THEIR agenda there, asked once for the whole run
        self.their_ags: dict = {}                 # 52.14: ... and that agenda itself (its claims are the save roots of the inner agenda on the static leaf)
        self.static_read = 0                      # 52.14: static leaves (a reply after which the gain stands only as a balance) read from their side
        self.static_parried = 0                   # ... of which every threat of theirs had a verified parry (the gain kept) / ...
        self.static_unparried = 0                 # ... at least one threat had no parry (killed, or a gap)
        self.static_lines: list = []              # 52.14: one line per static leaf whose threats were asked as save roots (chase depth, reply, verdict); the root agenda prints them
        self.promote_stands = 0                   # 52.16 (4): promote routes ended at a reply that leaves the question's stopper defect standing with no new fix
        self.standing = 0                         # 52.13 (4): routes ended by a standing kill (no inner agenda)
        self.closed = 0                           # ... replies closed by a rule (mate in one / exchange, nothing of theirs)
        self.reply_kills: dict = {}               # reply uci -> [(position left, dbal, reads of the inner agenda, route san)] of the routes it killed with no claim of ours
        self.logged = 0                           # rows written to the chase log by the root agenda
        self.tag = None                           # the id the chase log rows carry (None: no log)
        self.rows: list = []                      # the chase log rows, written once by the root agenda
        self.gap_reused = 0                       # 52.13 (2a): GAP routes seen again with no more fuse than when they were cut, not re-run
        self.agendas = 0                          # inner agendas run
        self.asked = 0                            # questions they asked
        self.budget_hits = 0
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
        self.unaddressed = 0                      # 52.17 (2): forcing moves NOT followed by a chase because they fix no defect of the goal and make no claim of their own


    def out_of_time(self) -> bool:
        """52.17 (5): the deadline (or the slice of it this route may use) has passed. Never true when no seconds were given."""
        if self.deadline is None:
            return False
        if time.monotonic() >= self.deadline:
            self.hit = True
            return True
        return False


class Agenda:
    def __init__(self, state: State, side_depth: int = 1, memory: list | None = None, leaf_all: bool = False, chase_depth: int = 0, ctx: "ChaseCtx | None" = None, min_stake: int = 0, inherit: dict | None = None, chase_log: str | None = None, their: "Agenda | None" = None, keep: int = 0, seconds: "float | None" = None, facts: "FS.FactStore | None" = None):
        R.reset_ids()
        self.their = their                                      # 52.14: THEIR agenda on this board with the move passed to them: its claims are asked as save roots (an inner agenda on a static leaf)
        self.keep = keep                                        # 52.14: the gain (a balance, us to move) the static leaf stands on: a parry of a threat is verified when it keeps at least 1 of it
        self._chasing = None                                    # 52.16 (4): the promote question being chased (the inner agendas inherit it as a kill where nothing it read has changed)
        self.threat_qs: list = []                               # 52.14: (question, their claim) of every threat root pushed from `their`
        self.inherit = inherit or {}                            # 52.12 (3), 52.13 (1): key -> {kind: kill | claim, reads, diff, ...} for the outer agenda's SETTLED questions (a kill, or a claim with its reads)
        self.inherited = 0
        self.min_stake = min_stake                              # 52.12: an inner agenda asks only the roots whose stake can still rescue the route (a root below it is not pushed)
        self.chase_depth = chase_depth                          # 52.12: 0 for the root agenda, +1 per inner agenda of a chase
        self.ctx = ctx if ctx is not None else (ChaseCtx(facts) if side_depth else None)       # their agendas (side_depth 0, no ctx) never chase
        if self.ctx is not None and chase_log is not None:
            self.ctx.tag = chase_log                            # 52.13 (4): the id the chase log rows carry
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
        self.best: tuple | None = None                          # (value, -opp defects, claim)
        self.budget = sum(1 for s in chess.SQUARES if self.board.piece_at(s) and self.board.piece_at(s).color == self.me) * 8 \
            + sum(1 for m in self.board.legal_moves if self.board.is_capture(m))
        self.notes: list = []
        if ctx is None and self.ctx is not None:
            self.ctx.left = self.budget * FUSE                  # the root agenda's FUSE (10x the question budget), set ONCE: every chase of every question and the inner agendas spend it
            if seconds is not None:                             # 52.17 (5): THE BUDGET IS TIME; the question-count fuse stays only as a safety (20x larger)
                self.ctx.left = self.budget * FUSE * 20
                self.ctx.seconds, self.ctx.t0 = seconds, time.monotonic()
                self.ctx.deadline = self.ctx.final = self.ctx.t0 + seconds
        self._threats_done = not side_depth and their is None
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
        self.fact_of: dict = {}                                 # 52.19: question id -> the id of the fact it derived or was served
        self.claim_fid: dict = {}                               # id(claim) -> the fact id that holds it
        self.premises: list = []                                # ... the ids of every fact this agenda derived or was served (an inner answer stands on them)
        self._cur: "R.Question | None" = None                   # the question being asked (the stake of the facts derived under it)
        self._qdeps: list = []                                  # the premises of the fact the current question will derive (a chase fills them)
        self._rf: "list | None" = None                          # the facts the route being chased read (inner answers, their agendas)
        self._frame: list = []                                  # the facts the listener flagged for the reply being read (restored when the reply is done)
        self._cell = FS.board_cell(self.board)
        self._fs0 = self.ctx.facts.snapshot() if self.ctx is not None else None      # the store's counters when this agenda began (the deltas are its own)
        self._pre: dict = {}                                    # 52.17 (4): route key -> the replies the cheap pass closed by rule
        self._roots()
        for i, q in enumerate(self.all, 1):                      # numbered in the order they are asked
            q.id = i
        R._ids[0] = len(self.all)

    @property
    def questions(self) -> list:
        return sorted(self.all, key=lambda q: (-q.stake, q.depth, q.id))

    # ------------------------------------------------------------------------------------------------------------ roots

    def _push(self, q: R.Question, force: bool = False) -> R.Question:
        if q.stake < self.min_stake and not force:
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

    def _stamp(self, claim: Claim, q: R.Question) -> Claim:
        claim.goal = q.goal
        claim.fixes = self._fix_kinds(q, claim.move)
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
        two moves to reach, the pattern being a MATE otherwise (no other defect). The defect `reach` names the first steps; the fix routes are those steps."""
        b = self.board
        if ksq is None:
            return
        for frm in sorted(chess.SquareSet(b.occupied_co[self.me] & ~b.kings & ~b.pawns)):
            pc = b.piece_at(frm)
            for s in chess.SQUARES:
                o = b.piece_at(s)
                if s == frm or (o is not None and o.color == self.me):
                    continue
                cf = G.with_piece(b, pc, s, frm)
                if ksq not in cf.attacks(s):
                    continue
                steps = R.reach_steps(b, self.us, frm, s)
                if not steps:
                    continue
                q = R.new_question("mate", R.STAKE_MATE, dict(sym=pc.symbol(), at=s, frm=frm, king=ksq, more=[], move=None, reach=[m.uci() for m in steps]))
                if R.mate_defects(b, self.us, R.placement_of(q)):
                    continue                                     # another defect as well: two prep moves, the tempo arithmetic kills it before it is asked
                via = ", ".join(sorted({G.name(m.to_square) for m in steps}))
                q.defects = [R.Defect("reach", s, [frm], f"the {pc.symbol().upper()} needs 2 moves (via {via})")]
                q.text = f"mate on {G.name(s)} ({pc.symbol().upper()} {G.name(frm)}->{G.name(s)} in 2 moves via {via}, their king on {G.name(ksq)})"
                self._push(q)
        if any(q.pattern.get("reach") for q in self.all):
            self.notes.append("reach roots (mate at distance 2) only when a mate is the stake: one prep move, the pattern otherwise a mate")

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
        seen = set()
        for cl in ag.state.claims:
            m = chess.Move.from_uci(cl.move) if cl.move else None
            if m is None or cl.move in seen or (not given and m.to_square in hang and b.piece_at(m.to_square)):       # (given: the hang root is below `min_stake` here, so the capture of a piece that hangs is a threat root too)
                continue
            seen.add(cl.move)
            if cl.kind == "promote":
                pw = m.from_square
                pb = ag.board
                dest = R.pawn_path(pb, pw, G.word(pb, ("pushes", G.name(pw)), not self.me))[-1]
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
        self._ask_loop()
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
        if self.ctx is not None and self.chase_depth == 0:
            self.lines += self.facts_lines()
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.final is not None:       # 52.17 (5)
            used = time.monotonic() - self.ctx.t0
            self.ctx.time_outs += 1 if self.ctx.hit else 0
            self.lines.append(f"budget: {self.ctx.seconds:g} s, used {used:.2f} s, {self.asked + self.ctx.asked} questions" + (" -- OUT OF TIME: the open routes are GAPs, the best bound claim decides" if self.ctx.hit else ""))
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.static_lines:       # 52.14: the static leaves read from their side, and the parries
            self.lines.append(f"static leaves read from their side, threats asked as save roots ({len(self.ctx.static_lines)}):")
            self.lines += ["  " + x for x in self.ctx.static_lines]
        if self.ctx is not None and self.chase_depth == 0 and self.ctx.rows:       # 52.13 (4): the chase log of this root, appended once (per item)
            import json
            from catspace.io import paths
            with open(paths.experiment("chase_log.jsonl"), "a") as f:
                for row in self.ctx.rows:
                    f.write(json.dumps(row) + "\n")
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
            if self.best and self.best[0] >= nstake:
                self.lines.append(f"STOP (a): the best claim ({self.best[2].what}, {self.best[0]}) is at least the stake of every open question (next: {nstake})")
                return
            if self.asked >= self.budget:
                left = [t[4] for t in self.heap]
                self.lines.append(f"STOP (c): budget {self.budget} questions asked; {len(left)} open questions become gaps")
                self.state.gaps += [f"budget: {q.text} (stake {q.stake}) not asked" for q in left]
                return
            if self.chase_depth and self.ctx.out_of_time():       # 52.17 (5): an inner agenda that meets the deadline stops where it is (its answer is cut, never memoised, never read as "no claim")
                self.ctx.budget_hits += 1
                self.lines.append("STOP (t): out of time")
                return
            plain = min(self.heap, key=lambda t: (t[0], t[2], t[3]))                # the stake-only order
            _, _, _, _, q = heapq.heappop(self.heap)
            if plain[4] is not q:
                sc, rec = self._recall_score(q)
                self.lines.append(f"memory: Q{q.id} moved up by {sc} (seen {rec['id']}: {rec['goal']} by {rec['lemma']})")
            if self._inherit(q):
                continue
            if self.chase_depth:                                 # an inner agenda pays the shared budget as it asks (a chain of checks meets it on the way down)
                self.ctx.left -= 1
                before = self.asked
            self._cur, self._qdeps, n0 = q, [], len(self.state.claims)
            self._ask(q)
            if self.ctx is not None:
                self._file(q, self.state.claims[n0:])
            if self.chase_depth:
                self.ctx.left -= max(self.asked - before - 1, 0)
        self.lines.append("STOP (b): every question is closed")

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
            if self.best is None or (hit["net"], 0) > self.best[:2]:
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
            cl = dataclasses.replace(old, what=b.san(chess.Move.from_uci(old.move)), line=f"inherited: nothing changed for {what} ({old.line})")
            self.state.claims.append(cl)
            self.claim_fid[id(cl)] = f.id
            q.status, q.claimed, q.reads = "verified", (f.payload["net"], cl, f.reads), f.reads
            self.ctx.inherited_claims += 1
            self.lines.append(f"Q{q.id} {q.text}: inherited: nothing changed for {what}: CLAIM {cl.kind} {cl.what} (value {cl.value}, net {f.payload['net']}), lemma {cl.lemma} -- {src}")
            if self.best is None or (f.payload["net"], 0) > self.best[:2]:
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
        if q.status not in ("killed", "verified", "gap"):
            return
        key = self._fkey(q)
        fs, kind = self.ctx.facts, {"killed": "kill", "verified": "claim", "gap": "gap"}[q.status]
        reads, value, payload, portable = q.reads, 0, {}, key is not None
        claim = None
        if kind == "claim":
            if q.claimed is not None:
                value, claim, reads = q.claimed[0], q.claimed[1], q.claimed[2]
                payload = dict(claim=claim, net=q.claimed[0])
                portable = portable and not claim.lemma.startswith("chase")      # a chase claim reads every reply of the board it was made on: it is not transferable (as in 52.13)
            else:
                claim = new_claims[-1] if new_claims else None
                value = claim.value if claim is not None else 0
                portable = False
        elif kind == "kill":
            payload = dict(why=q.reason.removeprefix("inherited: ").split(";")[0][:80])
        text = f"{claim.kind} {claim.what}: {q.text}" if claim is not None else q.text
        sig = FS.signature(self._cell, reads) if reads is not None else ()
        f = fs.add(kind, key if key is not None else ("q", q.goal, q.text, self.pk), self.me, text, value=value, reads=reads, sig=sig, just=self._qdeps, stake=q.stake, depth=self.chase_depth,
                   under=f"Q{q.id} at depth {self.chase_depth}", pk=self.pk, portable=portable, power=(2 if getattr(q, "_reasoned", False) else self._power(q) if getattr(q, "_chased", False) else 0), payload=payload)
        self.fact_of[q.id] = f.id
        self.premises.append(f.id)
        if claim is not None:
            self.claim_fid[id(claim)] = f.id

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
                net = shift + dbal + (0 if P2.is_check() else P2.exchange())
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
        if self.best is None or (worst[0], 0) > self.best[:2]:
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
            if self.side_depth and bind.schema.startswith("delayed_balance") and bind.schema.endswith("_d1"):
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
        if self.best is None or key > self.best[:2]:
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
        if fam != "delayed_balance":
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
                    ag = self._their_agenda(e)
                    theirs = max(ag.state.claims, key=lambda cl: cl.value) if ag.state.claims else None
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
        if self.best is None or (net, 0) > self.best[:2]:
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
        c.push(mv)
        try:
            return c.is_checkmate()
        finally:
            c.pop()

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

    def _inner(self, fen: str, need: int, diff: "frozenset | None" = None, their: "Agenda | None" = None, keep: int = 0) -> "dict | None":
        """The WHOLE agenda asked again on `fen` (our move), only the roots of stake >= `need` (a smaller claim cannot rescue the route): -> {value, claim, asked, need};
        None when the shared budget is spent. Memoised by FEN (an answer asked with a smaller `need` serves a larger one); a FEN being asked (a cycle) answers no claim.
        52.14: with `their` (their agenda there, the move passed to them) their claims are also asked as SAVE roots; the answer lists them (`threats`: (their claim, our parry claim or None)).
        The memo is keyed by (position, with threats): an answer asked WITH threats serves a later ask without them, not the reverse."""
        ctx = self.ctx
        pk = _pos_key(fen)
        wt = their is not None
        hit = ctx.memo.get((pk, True, keep)) if wt else (ctx.memo.get((pk, False, 0)) or next((h for (k, w, _kp), h in ctx.memo.items() if k == pk and w), None))     # (position, with threats, the gain kept): an answer WITH threats serves an ask without, not the reverse
        if hit is not None and hit["need"] <= need:
            self._touch(hit.get("fact"))
            return hit
        if pk in ctx.active:
            return dict(value=0, claim=None, asked=0, need=need, threats=[(cl, None) for cl in their.state.claims] if wt else [])     # a cycle parries nothing
        if ctx.left <= 0 or self.chase_depth >= DEPTH_FUSE or ctx.out_of_time():
            if ctx.left > 0:
                ctx.depth_cuts += 1
            return None
        ctx.active.add(pk)
        hits0 = ctx.budget_hits
        saved = R._ids[0]
        ag = Agenda(State(fen=fen, us=self.us, pieces=[]), side_depth=0, chase_depth=self.chase_depth + 1, ctx=ctx, min_stake=need, their=their, keep=keep)      # (52.19: what it inherits comes from the fact store, not from this agenda alone)
        ag.ask()
        R._ids[0] = saved
        ctx.active.discard(pk)
        ctx.agendas += 1
        ctx.asked += ag.asked
        best = ag.best[2] if ag.best and ag.best[0] > 0 else None
        reads = None if any(q.reads is None for q in ag.all) else frozenset().union(*(q.reads for q in ag.all))     # 52.13 (4): everything this agenda read (None: some question did not record it)
        threats = [(cl, q.claimed[1] if q.status == "verified" and q.claimed is not None else None) for q, cl in ag.threat_qs]
        out = dict(value=ag.best[0] if best else 0, claim=best, asked=ag.asked, need=need, cut=ctx.budget_hits > hits0, reads=reads, threats=threats,
                   promote=next((c for c in ag.state.claims if c.kind == "promote"), None))      # 52.15 (2): the promote claim of this position, whether or not it is the best claim
        cfid = ag.claim_fid.get(id(best)) if best is not None else None
        f = ctx.facts.add("gap" if out["cut"] else ("claim" if best is not None else "kill"), ("inner", pk, wt, keep), self.me, f"the agenda on {pk.split()[0]}" + (f": {best.what}" if best is not None else ": no claim"),
                          value=out["value"], reads=reads, just=[cfid] if cfid is not None else ag.premises, stake=self._cur.stake if self._cur is not None else need, depth=self.chase_depth + 1,
                          under=f"Q{self._cur.id} at depth {self.chase_depth}" if self._cur is not None else f"depth {self.chase_depth}", pk=pk, portable=False)
        out["fact"] = f.id
        self._touch(f.id)
        if not out["cut"]:                                      # an answer the budget cut short is not memoised and is never read as "no claim"
            ctx.memo[(pk, wt, keep if wt else 0)] = out
        return out

    def _touch(self, fid) -> None:
        """52.19: the route being chased read this fact (an inner answer, their agenda): the claim or kill of the route stands on it."""
        if fid is not None and self._rf is not None and fid not in self._rf:
            self._rf.append(fid)

    def _chase_route(self, m: chess.Move, promote: bool = False, cheap: bool = False) -> dict:
        """52.19: the route of `_chase_route_body`, with the facts it read (inner answers, their agendas) kept on the result (`facts`) for the claim or kill that stands on it, and the listener's
        frame for the reply being read restored on the way out."""
        saved_rf, saved_frame = self._rf, self._frame
        self._rf, self._frame = [], []
        try:
            res = self._chase_route_body(m, promote, cheap)
            res["facts"] = list(dict.fromkeys(res.get("facts", []) + self._rf))
            if not cheap:
                self._gap_fact(m, promote, res)
            return res
        finally:
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

    def _chase_route_body(self, m: chess.Move, promote: bool = False, cheap: bool = False) -> dict:
        """One forcing route of the chase: every reply of theirs, the whole agenda on each resulting position. -> {value, killed, gap, parts, lemmas}.
        52.17 (4) `cheap`: CHEAPEST REPLY FIRST -- only what the rules close is read (standing kills, a reply that mates or stalemates us, a mate in one, the exchange with nothing of
        theirs, a promotion that stands); a reply that needs an inner agenda leaves the route PENDING (not memoised) and the closed replies are kept for the second pass.
        52.15 (2) `promote`: the question is a promotion; a reply is answered when the inner agenda claims the promotion (or a mate), or the reply itself gave up at least the
        stake in material; the roots asked there are those of stake >= the promote stake, and the route's value is the promote stake (a mate: 100)."""
        b, ctx = self.board, self.ctx
        key = (_pos_key(b.fen()), m.uci() + (":promote" if promote else ""))
        pre = self._pre.pop(key, None) if not cheap else None      # the replies the cheap pass closed by rule (value, why)
        if key in ctx.routes:
            hit = ctx.routes[key]
            if hit["gap"] is None:
                return hit
            if ctx.left <= hit["cut_left"]:                      # 52.13 (2a): a GAP stays a GAP while the fuse left is no more than when it was cut
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
        for dbal, uci, chk, cap, rec, mate1 in rows:
            sk = None if promote else self._standing_kill(uci, P2_of(uci), dbal)
            if sk is not None:
                res["killed"] = f"after {san_of(uci)}: {sk[0]} (standing kill: nothing the inner agenda read after {sk[1]} {san_of(uci)} has changed)"
                ctx.standing += 1
                self._log(key, uci, "standing", dbal, None, sk[2], res["killed"])
                break
        if res["killed"] is not None:
            ctx.routes[key] = res
            return res
        rows.sort(key=lambda t: (0 if t[4] else 1 if t[2] else 2 if t[3] else 3, t[0], t[1]))
        gains = []
        closed_now = {}
        for dbal, uci, chk, cap, rec, mate1 in rows:
            P2 = P2_of(uci)
            san_r = san_of(uci)
            if pre is not None and uci in pre:                   # closed by rule in the cheap pass: counted and logged there
                val, why = pre[uci]
                gains.append(val)
                res["parts"].append(f"after {san_r}: closed by rule: {why} -> {val}")
                continue
            if P2.legal_count() == 0:
                res["killed"] = f"after {san_r} " + ("they mate us" if P2.is_check() else "stalemate")
                self._log(key, uci, "killed", dbal, None, None, res["killed"])
                break
            closed = ((BD.MATE_VALUE, "mate in one") if (P2.mate_in_one() if mate1 is None else mate1) else None) if promote else self._closed_by_rule(P2, dbal, mate1)
            if closed is not None:
                val, why = closed
                gains.append(val)
                res["parts"].append(f"after {san_r}: closed by rule: {why} -> {val}")
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
            need = min(R.STAKE_PROMOTE, 1 - dbal) if promote and dbal <= 0 else R.STAKE_PROMOTE if promote else 1 - dbal if dbal <= 0 else BD.MATE_VALUE          # a reply that already gains is read for a mate only; else its static gain must survive their threats (below)
            self._enter_reply(frozenset(P0.diff(P2)))
            inn = self._inner(P2.fen(), need, frozenset(P0.diff(P2)))
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
                elif inn.get("cut"):
                    res["gap"] = self._cut(san, san_r, True)
                    res["reply"] = uci
                    ctx.budget_hits += 1
                    self._log(key, uci, "gap", dbal, inn["value"], None, res["gap"])
                    break
                else:
                    res["killed"] = f"after {san_r}: the promotion is not claimed (no promote claim of ours; the reply took {-dbal})" if dbal <= 0 else f"after {san_r}: the promotion is not claimed ({dbal:+d}, below the stake)"
                    self._log(key, uci, "killed", dbal, inn["value"], inn.get("reads"), res["killed"])
                    break
                gains.append(g)
                res["parts"].append(f"after {san_r}: {part} -> {g}")
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
                    inn2 = self._inner(P2.fen(), need, frozenset(P0.diff(P2)), their=ctx.their_ags.get(_pos_key(G.with_turn(chess.Board(P2.fen()), not self.me).fen())), keep=dbal)
                    if inn2 is None:
                        res["gap"] = self._cut(san, san_r)
                        res["reply"] = uci
                        ctx.budget_hits += 1
                        self._log(key, uci, "gap", dbal, None, None, res["gap"])
                        break
                    inn = dict(inn, cut=inn.get("cut") or inn2.get("cut"))
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
            if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE and inn.get("cut"):
                res["gap"] = self._cut(san, san_r, True)
                res["reply"] = uci
                ctx.budget_hits += 1
                self._log(key, uci, "gap", dbal, inn["value"], None, res["gap"])
                break
            if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE:
                res["killed"] = f"after {san_r}: " + (f"our best claim {inn['claim'].what} wins {inn['value']}, the reply took {-dbal}" if inn["claim"] else "no claim of ours and no material gain") + f" ({dbal:+d})"
                if inn["claim"] is None and not inn.get("cut"):          # a standing kill: the later routes' same reply is skipped while the reads stay untouched
                    ctx.reply_kills.setdefault(uci, []).append((P2, dbal, inn.get("reads"), san))
                self._log(key, uci, "killed", dbal, inn["value"], inn.get("reads"), res["killed"])
                break
            cl = inn["claim"] if inn["value"] >= need else None
            gains.append(BD.MATE_VALUE if inn["value"] >= BD.MATE_VALUE else dbal - unparried if static else dbal + (inn["value"] if cl else 0))
            res["parts"].append(f"after {san_r}: " + (f"{cl.what} ({cl.line}; value {cl.value})" if cl else f"material {dbal:+d}" + (f", their {claim_their.what} wins {t_their} if we pass" if claim_their else "") + (f", {parry_note}" if parry_note else "")) + (f" -> {gains[-1]}" if cl or claim_their or parry_note else ""))
            self._log(key, uci, "claim" if cl else "material", dbal, inn["value"], inn.get("reads"), res["parts"][-1])
            if cl:
                res["lemmas"].append(cl.lemma)
        if cheap and res.get("pending") and res["killed"] is None:
            self._pre[key] = closed_now                          # not memoised: the second pass finishes it
            return res
        if res["killed"] is None and res["gap"] is None:
            res["value"] = min(gains) if gains else 0
        if res["gap"] is not None:
            res["cut_left"] = ctx.left
            if res["gap"].startswith("out of time"):                # 52.17 (5): the deadline mark on the GAP memo entry beside cut_left: the background tier resumes exactly here
                res["timed"], res["mark"] = True, round(time.monotonic() - ctx.t0, 3) if ctx.t0 is not None else None
        ctx.routes[key] = res
        return res

    def _note_resume(self, q: R.Question, r: "R.Route | None", res: dict) -> None:
        """52.17 (5): record where the clock cut: the question key, the position, the route and the reply it stopped at, the time mark and the fuse left (the 52.13 GAP memo keeps cut_left too)."""
        self.ctx.resume.append(dict(question=R.question_key(q), text=q.text, fen=_pos_key(self.board.fen()), route=r.move.uci() if r is not None else None,
                                    san=r.san if r is not None else None, reply=res.get("reply"), mark=round(time.monotonic() - self.ctx.t0, 3) if self.ctx.t0 is not None else None,
                                    cut_left=self.ctx.left, depth=self.chase_depth))

    def _cut(self, san: str, san_r: str, inner: bool = False) -> str:
        """The GAP text of a route cut at a reply: the TIME when the deadline passed (52.17 (5)), else the fuse."""
        if self.ctx.out_of_time():
            return f"out of time: {san} not followed past {san_r}"
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

    def _standing_kill(self, uci: str, P2: "AC.Pos", dbal: int) -> tuple | None:
        """52.13 (4): a reply of theirs killed an earlier route of this root with no claim of ours; the kill stands for this route when the position this reply leaves
        differs from that one only where the inner agenda's reads were not (the reads are recorded; unknown reads never transfer) and the reply took no less.
        -> (why, route san, reads)."""
        for P_old, dbal0, reads, san0 in self.ctx.reply_kills.get(uci, ()):
            if reads is None or dbal >= 1 or dbal0 < dbal:          # the inner agenda read the roots of stake >= 1 - dbal0: enough for any reply that took at least as much
                continue
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
        t, _cl = self._their_pass(P2.fen())
        if t:
            return None
        return dbal + best, f"exchange +{best} ({R.exchange_read(chess.Board(P2.fen()))}), nothing of theirs"

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
        if hit is None:
            ag = self._their_agenda(c) if any(True for _ in c.legal_moves) else None
            cl = max(ag.state.claims, key=lambda x: x.value) if ag is not None and ag.state.claims else None
            hit = self.ctx.theirs[key] = (cl.value if cl else 0, cl)
            if ag is not None:
                self.ctx.their_ags[key] = ag
            f = self.ctx.facts.add("claim" if cl else "kill", ("theirs", key), not self.me, f"their agenda with the move passed: " + (f"{cl.what} wins {cl.value}" if cl else "no claim"), value=hit[0], reads=None,
                                   stake=self._cur.stake if self._cur is not None else 0, depth=self.chase_depth, under=f"Q{self._cur.id} at depth {self.chase_depth}" if self._cur is not None else f"depth {self.chase_depth}", pk=key, portable=False)
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

    def _chase(self, q: R.Question) -> bool:
        """52.12: the chase of a question that would be killed. -> True when a route CLAIMS (the question is verified)."""
        if self.ctx is None or q.goal not in ("mate", "win", "promote") or self.min_stake >= BD.MATE_VALUE or (self.chase_depth >= 2 and q.goal != "mate"):      # (below the first inner level only the mate questions chase) # a mate-only read (the reply already gained, or lost too much for anything but a mate) does not chase on
            return False
        if self.best is not None and self.best[0] >= q.stake:      # 52.13 (2c): a question is chased only when its stake exceeds the best claim so far (stop (a) would end the agenda anyway)
            self.lines.append(f"  chase: not asked, the best claim ({self.best[2].what}, {self.best[0]}) is at least the stake {q.stake}")
            return False
        if self.ctx.out_of_time():                                 # 52.17 (5): past the deadline a question is asked only for what closes at the static leaf: no chase
            self.lines.append("  chase: out of time -- not started (a GAP, never a kill)")
            self._note_resume(q, None, dict(gap="chase not started"))
            self.state.gaps.append(f"Q{q.id} {q.text}: out of time: chase not started")
            return False
        b = self.board
        routes, notes = [], []
        for m in R._moves(b, self.us):
            why = R.forcing_why(b, m, q.stake, self.us, mate_threats=True, notes=notes)
            if why:
                ok, note = self._addresses(q, m, why)
                if ok:
                    routes.append(R.Route(m, b.san(m), why))
                    if note:
                        notes.append(note)
                else:
                    self.ctx.unaddressed += 1
                    notes.append(note)
        if q.goal == "promote":                                  # 52.15 (2): the fix routes of the defects (take the stopper or the blocker, push with check) are routes too, at every depth
            seen = {r.move for r in routes}
            for f in q.fixes:
                for r in R.all_routes(f):
                    if r.move not in seen:
                        seen.add(r.move)
                        routes.append(R.Route(r.move, r.san, f"fix: {r.why}"))
        if q.goal == "promote" and not self.chase_depth:        # 52.15 (2): plus a move that attacks a named STOPPER of the question ("threat on the stopper"), or offers the pawn to it
            routes += self._stopper_routes(q, [r.move for r in routes], notes)
        if self.chase_depth:                                    # inner levels follow the most forcing moves only: a check, a capture, a mate threat (the quiet "threat n" moves are the root's)
            routes = [r for r in routes if r.why.removeprefix("fix: ") in ("check", "threat of mate") or r.why.removeprefix("fix: ").startswith("capture")]
        routes.sort(key=lambda r: (0 if r.why.removeprefix("fix: ") == "check" else 1 if r.why.removeprefix("fix: ").startswith("capture") else 2))      # the most forcing first: a check, a capture, a threat (stable inside)
        self.promote_stake = q.stake if q.goal == "promote" else 0
        if not routes:
            self.lines.append("  chase: no forcing move on this board")
            q._chased = True                                     # nothing to follow: the chase ran to its end
            return False
        for n in notes[:6]:
            self.lines.append("  chase: " + n)
        self.lines.append(f"  chase: make them move -- {len(routes)} forcing route(s): " + ", ".join(f"{r.san} ({r.why})" for r in routes[:12]) + (" ..." if len(routes) > 12 else ""))
        done, cut, route_facts = [], False, []
        pool = self.ctx.left
        self._chasing = q if q.goal == "promote" else None
        # 52.17 (4): CHEAPEST REPLY FIRST. Before any route opens an inner agenda, every route's replies are read at the static leaf and what the rules close is closed (a standing
        # kill, a mate or stalemate against us, a mate in one, the exchange with nothing of theirs): a route killed there never draws a share of the fuse, and the routes still open
        # share it in order.
        cheap_res = {}
        for i, r in enumerate(routes):
            got = self._chase_route(r.move, promote=q.goal == "promote", cheap=True)
            if not got.get("pending"):
                cheap_res[i] = got
        for i, r in enumerate(routes):
            saved = self.ctx.left
            open_left = sum(1 for j in range(i, len(routes)) if j not in cheap_res)
            share = max(saved // max(open_left, 1), 1) if saved > 0 else 0
            if self.chase_depth == 0 and saved > 0 and i not in cheap_res:             # the root's fuse is shared FAIRLY by the routes of this chase: each gets an equal share of what is left (a chain of checks must not spend what Qd5 needs: 0DoFB)
                self.ctx.left = share
            before = self.ctx.left
            if i in cheap_res:
                res = cheap_res[i]
            elif self.ctx.out_of_time():                           # 52.17 (5): the open routes end as GAPs
                res = dict(value=0, killed=None, gap=f"out of time: {r.san} not started", parts=[], lemmas=[])
                self.ctx.budget_hits += 1
            else:
                if self.chase_depth == 0 and self.ctx.final is not None:       # ... and the time is shared FAIRLY like the fuse: each open route gets an equal slice of what is left
                    now = time.monotonic()
                    self.ctx.deadline = now + max(self.ctx.final - now, 0) / max(open_left, 1)
                res = self._chase_route(r.move, promote=q.goal == "promote")
                if self.chase_depth == 0 and self.ctx.final is not None:
                    self.ctx.deadline = self.ctx.final
            if self.chase_depth == 0 and i not in cheap_res:
                # 52.16 (2): a route is CHARGED at most its share. The frames open when the fuse runs out finish their questions (that is how 08pSj's Rf1+ and 0VSUT's Qxe5 are decided),
                # but that finishing is free: charged in full it took the share of every later route (0LRTF: 239 questions on a share of 120, then Rd7 got 1).
                self.ctx.left = saved - min(max(before - self.ctx.left, 0), share) if saved > 0 else self.ctx.left
            route_facts += res.get("facts", [])
            if res["gap"] and res["gap"].startswith("out of time"):
                self._note_resume(q, r, res)
            if res["gap"]:
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
        self._chasing = None
        q._chased = not cut                                      # 52.19: a kill is "chased" only when no route was cut (a GAP); a cut chase is as weak as no chase
        self._qdeps = list(dict.fromkeys(route_facts))           # a kill stands on every route's facts ...
        if self.chase_depth == 0 and pool > 1 and any(t[4].goal in ("mate", "win", "promote") and (self.best is None or self.best[0] < t[4].stake) for t in self.heap):
            # ... and a ROOT question is charged at most half of the pool it started with, while another root that can still beat the best claim is open: by stake order the
            # question that matters most gets the larger share, never all of it (0GJec: the promotion's chase took 618 of 570 and every later root was read with nothing)
            charged = pool - max(self.ctx.left, pool - pool // 2)
            self.ctx.left = max(self.ctx.left, pool - pool // 2)
            self.lines.append(f"    fuse: this root is charged {charged} of the pool {pool} (routes at most their shares, the root at most half while another root can beat the best claim)")
        self.lines.append(f"    (chase: {self.ctx.agendas} inner agendas, {self.ctx.asked} inner questions, {self.ctx.budget_hits} fuse hit(s), {self.ctx.gap_reused} gap route(s) reused, fuse left {self.ctx.left})")
        self.lines.append("    chase ended: " + ("the FUSE was hit on at least one route (a GAP, never a kill)" if cut else "every route closed (claim on every reply or a refuting reply)"))
        if not done:
            return False
        done.sort(key=lambda t: (-t[0], t[1].move.uci()))
        value, r, res = done[0]
        self._qdeps = list(res.get("facts", []))                 # ... a claim only on the facts of the route that decided it
        net = value
        kind = "mate" if value >= BD.MATE_VALUE else ("promote" if q.goal == "promote" and not res.get("material") else "win")
        parts = " / ".join(res["parts"][:4]) + (" / ..." if len(res["parts"]) > 4 else "")
        lemma = "chase: step_after+step_least over " + ", ".join(sorted(set(res["lemmas"]))[:3]) if res["lemmas"] else "chase: step_after+step_least"
        claim = Claim(kind, r.san, value, lemma, f"{r.san}; {parts}", move=r.move.uci())
        self._stamp(claim, q)
        claim.fixes = claim.fixes + ("chase",)
        self.state.claims.append(claim)
        q.status = "verified"
        self.lines.append(f"  -> CLAIM {kind} {r.san} (value {value}), lemma {lemma}, line {claim.line}")
        if self.best is None or (net, 0) > self.best[:2]:
            self.best = (net, 0, claim)
            self.asked_at_claim = self.asked
        return True

    # ------------------------------------------------------------------------------------------------------------ kills, saves

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
        if len(self.state.claims) == before:
            self._decide_by_smallest_loss(q)

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

    def _ask_promote(self, q: R.Question, head: str) -> None:
        """The defects of the pawn's path are read with the words; a stopper in time kills the question WITH THE DISTANCE REASON; no defect = unstoppable by the
        tempo arithmetic, and then the claim is made ONLY when the proved passer (Ideas.Passer2.unstoppablePasser2_sound, via proved_passers) fires; if it does
        not, the question ends as a GAP naming the hypothesis that fails."""
        b = self.board
        sq = q.pattern["pawn"]
        q.defects, facts = R.promote_defects(b, sq)
        n, promo = facts["pushes"], facts["promo"]
        q.reads = R.reads_of(q, b) | frozenset(facts["path"]) | {sq}            # 52.16 (4): the pawn, its path (the promotion square included), the defects' squares and their pieces: what the question read
        kd = facts["king_dist"]
        self.lines.append(head + (": " + "; ".join(f"defect {d.kind} ({d.why})" for d in q.defects) if q.defects else
                                  ": no defect -- unstoppable by the tempo arithmetic" + (f" (their K needs {kd} to {G.name(promo)}, the pawn {n})" if kd is not None else "")))
        if q.defects:
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
            if self.best is None or (claim.value, 0) > self.best[:2]:
                self.best = (claim.value, 0, claim)
                self.asked_at_claim = self.asked
            return
        ex = self._promotion_exchange(q, sq, facts)
        if ex is not None:
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
        if self.best is None or (net, 0) > self.best[:2]:
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

    def decision(self) -> Claim | None:
        return self.best[2] if self.best else None
