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
balance with us to move is checked on the chosen route by THEIR agenda (a claim of theirs at least the gain refutes it; 0AgL8's Nxg2). The reach (52.12 (4)): a mate
root at distance two (one prep move) whose fixes are the first steps; a step is forcing when it checks or leaves a mate in one (a mate threat).
"""
from __future__ import annotations

import dataclasses
import heapq
import re

import chess

from catspace.incubator.analyst import lean_passer_v2_proto as PV
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import reasoned as RS
from catspace.incubator.analyst.discovery import selfmine as SM
from catspace.incubator.analyst.discovery.state import Claim, State


class ChaseCtx:
    """The state shared by the root agenda and its inner agendas (52.12): the FEN memo, the route memo, the shared question budget and the counters."""
    def __init__(self):
        self.memo: dict = {}                      # fen -> {value, claim, asked}: the inner agenda's best claim, asked once for the whole run
        self.routes: dict = {}                    # (fen, uci) -> the evaluated route (memoised: the same position is chased by many questions)
        self.active: set = set()                  # the FENs being asked (a cycle ends here)
        self.left = 0                             # the shared question budget of the root question being chased
        self.agendas = 0                          # inner agendas run
        self.asked = 0                            # questions they asked
        self.budget_hits = 0
        self.inherited = 0                        # 52.12 (3): questions of the outer agenda inherited, not asked again
        self.inherited_claims = 0                 # 52.13 (1): ... of which claims (the rest are kills)
        self.reopened = 0                         # ... and the ones that met the diff and were asked
        self.fresh = 0                            # 52.13 (1): questions NEW on the inner board (a key the outer agenda did not have, or did not settle), asked


class Agenda:
    def __init__(self, state: State, side_depth: int = 1, memory: list | None = None, leaf_all: bool = False, chase_depth: int = 0, ctx: "ChaseCtx | None" = None, min_stake: int = 0, inherit: dict | None = None):
        R.reset_ids()
        self.inherit = inherit or {}                            # 52.12 (3), 52.13 (1): key -> {kind: kill | claim, reads, diff, ...} for the outer agenda's SETTLED questions (a kill, or a claim with its reads)
        self.inherited = 0
        self.min_stake = min_stake                              # 52.12: an inner agenda asks only the roots whose stake can still rescue the route (a root below it is not pushed)
        self.chase_depth = chase_depth                          # 52.12: 0 for the root agenda, +1 per inner agenda of a chase
        self.ctx = ctx if ctx is not None else (ChaseCtx() if side_depth else None)       # their agendas (side_depth 0, no ctx) never chase
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
            self.ctx.left = self.budget                         # the root agenda's question budget, shared by every chase of every question and the inner agendas
        self._threats_done = not side_depth
        self._hang: list = []
        self.threat_views: list = []                            # (claim, their agenda) for every threat root
        self._seq = 0
        self.leaf_obs: list = []                                # 52.10: (our san, claimed, leaf line, their claim or None, w, verdict) per leaf read, for the OBS lines
        self._leaf: dict = {}
        self.leaf_all = leaf_all                                # 52.10: read the leaf of EVERY reply of theirs (4x the seconds on dev_20 / fresh_60, no claim downgraded more), not only the minimising ones
        self.leaf_stats = dict(read=0, down=0, untrusted=0)       # claims read at the leaf / downgraded / not trusted (the decomposition missed the word)
        self._tv: dict = {}                                     # their view of a route, asked once (the verify and the smallest-loss decision share it)
        self._roots()
        for i, q in enumerate(self.all, 1):                      # numbered in the order they are asked
            q.id = i
        R._ids[0] = len(self.all)

    @property
    def questions(self) -> list:
        return sorted(self.all, key=lambda q: (-q.stake, q.depth, q.id))

    # ------------------------------------------------------------------------------------------------------------ roots

    def _push(self, q: R.Question) -> R.Question:
        if q.stake < self.min_stake:
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
            self.notes.append("in check: every legal move answers it, so the save-king question is the only root (mate and material goals are read on its routes)")
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
        ag = self._their_agenda(G.with_turn(b, not self.me))
        seen = set()
        for cl in ag.state.claims:
            m = chess.Move.from_uci(cl.move) if cl.move else None
            if m is None or cl.move in seen or (m.to_square in hang and b.piece_at(m.to_square)):
                continue
            seen.add(cl.move)
            if cl.kind == "promote":
                pw = m.from_square
                pb = ag.board
                dest = R.pawn_path(pb, pw, G.word(pb, ("pushes", G.name(pw)), not self.me))[-1]
                self._push(R.new_question("save", cl.value, dict(promoter=pw, push=cl.move, value=cl.value, lemma=cl.lemma, line=cl.line, what=cl.what),
                                          text=f"stop their P on {G.name(pw)} (promotes on {G.name(dest)}): their claim {cl.what} wins {cl.value} (lemma {cl.lemma})"))
                self.threat_views.append((cl, ag))
                continue
            tgt = self._targets_of(m, cl)
            what = "our king" if cl.value >= BD.MATE_VALUE else (" and ".join(f"our {b.piece_at(t).symbol().upper()} on {G.name(t)}" for t in tgt) or "something of ours")
            self._push(R.new_question("save", cl.value, dict(threat=cl.move, targets=tgt, value=cl.value, lemma=cl.lemma, line=cl.line, what=cl.what),
                                      text=f"save {what}: their threat {cl.what} wins {cl.value} (their claim, lemma {cl.lemma})"))
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
            self._ask(q)
            if self.chase_depth:
                self.ctx.left -= max(self.asked - before - 1, 0)
        self.lines.append("STOP (b): every question is closed")

    def _inherit(self, q: R.Question) -> bool:
        """52.12 (3), 52.13 (1): THE DIFF DECIDES WHAT IS ASKED. An inner agenda asks (a) the question that sent it there (it is not settled in the outer agenda, so it is
        not in `inherit`), (b) every outer question whose reads meet the diff of the route and the reply, (c) every question NEW on this board. The rest is INHERITED
        with its status: a kill as it was, a claim re-stamped on this board (the move is legal here) with the line 'inherited: nothing changed for ...'."""
        key = R.question_key(q)
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
        if q.goal == "mate":
            q.defects = R.mate_defects(b, us, R.placement_of(q))
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

    def _inner(self, fen: str, need: int, after: chess.Board | None = None) -> "dict | None":
        """The WHOLE agenda asked again on `fen` (our move), only the roots of stake >= `need` (a smaller claim cannot rescue the route): -> {value, claim, asked, need};
        None when the shared budget is spent. Memoised by FEN (an answer asked with a smaller `need` serves a larger one); a FEN being asked (a cycle) answers no claim."""
        ctx = self.ctx
        hit = ctx.memo.get(fen)
        if hit is not None and hit["need"] <= need:
            return hit
        if fen in ctx.active:
            return dict(value=0, claim=None, asked=0, need=need)
        if ctx.left <= 0:
            return None
        ctx.active.add(fen)
        hits0 = ctx.budget_hits
        saved = R._ids[0]
        inherit = self._inherit_for(R.diff_squares(self.board, after)) if after is not None and any(q.status in ("killed", "verified") for q in self.all) else None
        ag = Agenda(State(fen=fen, us=self.us, pieces=[]), side_depth=0, chase_depth=self.chase_depth + 1, ctx=ctx, min_stake=need, inherit=inherit)
        ag.ask()
        R._ids[0] = saved
        ctx.active.discard(fen)
        ctx.agendas += 1
        ctx.asked += ag.asked
        best = ag.best[2] if ag.best and ag.best[0] > 0 else None
        out = dict(value=ag.best[0] if best else 0, claim=best, asked=ag.asked, need=need, cut=ctx.budget_hits > hits0)
        if not out["cut"]:                                      # an answer the budget cut short is not memoised and is never read as "no claim"
            ctx.memo[fen] = out
        return out

    def _chase_route(self, m: chess.Move) -> dict:
        """One forcing route of the chase: every reply of theirs, the whole agenda on each resulting position. -> {value, killed, gap, parts, lemmas}."""
        b, ctx = self.board, self.ctx
        key = (b.fen(), m.uci())
        if key in ctx.routes:
            return ctx.routes[key]
        san = b.san(m)
        c = b.copy(stack=False)
        c.push(m)
        res = dict(value=0, killed=None, gap=None, parts=[], lemmas=[], static=[])
        replies = list(c.legal_moves)
        if not replies:
            if c.is_checkmate():
                res.update(value=BD.MATE_VALUE, parts=[f"{san} mates"])
            else:
                res["killed"] = "stalemate"
            ctx.routes[key] = res
            return res
        bal0 = G.balance(b, self.me)
        rows = []
        for r in replies:
            d = c.copy(stack=False)
            d.push(r)
            rows.append((G.balance(d, self.me) - bal0, r, d))
        rows.sort(key=lambda t: (t[0], t[1].uci()))                  # the worst reply first: a route that fails dies at its first
        gains = []
        for dbal, rep, d in rows:
            san_r = c.san(rep)
            if not any(True for _ in d.legal_moves):
                res["killed"] = f"after {san_r} " + ("they mate us" if d.is_check() else "stalemate")
                break
            need = 1 - dbal if dbal <= 0 else BD.MATE_VALUE          # a reply that already gains is read for a mate only; else its static gain must survive their threats (below)
            inn = self._inner(d.fen(), need, d)
            if inn is None:
                res["gap"] = f"chase budget: {san} not followed past {san_r}"
                ctx.budget_hits += 1
                break
            if dbal >= 1 and inn["value"] < BD.MATE_VALUE:         # a gain that stands only as a static balance with us to move: kept or not is read by THEIR agenda on the chosen route (see _chase)
                res["static"].append((san_r, d.fen(), dbal))
            if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE and inn.get("cut"):
                res["gap"] = f"chase budget: {san} not followed past {san_r} (the inner agenda was cut by the budget)"
                ctx.budget_hits += 1
                break
            if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE:
                res["killed"] = f"after {san_r}: " + (f"our best claim {inn['claim'].what} wins {inn['value']}, the reply took {-dbal}" if inn["claim"] else "no claim of ours and no material gain") + f" ({dbal:+d})"
                break
            cl = inn["claim"] if inn["value"] >= need else None
            gains.append(BD.MATE_VALUE if inn["value"] >= BD.MATE_VALUE else dbal + (inn["value"] if cl else 0))
            res["parts"].append(f"after {san_r}: " + (f"{cl.what} ({cl.line}; value {cl.value})" if cl else f"material {dbal:+d}") + (f" -> {gains[-1]}" if cl else ""))
            if cl:
                res["lemmas"].append(cl.lemma)
        if res["killed"] is None and res["gap"] is None:
            res["value"] = min(gains) if gains else 0
        if res["gap"] is None:
            ctx.routes[key] = res
        return res

    def _static_refuted(self, r: R.Route, res: dict) -> str | None:
        """For each reply after which the route's gain is only the static balance (us to move), their agenda with the move passed to them: a claim of theirs at least the
        gain means the gain is not kept (we cannot show a parry) -> the reason, else None."""
        for san_r, fen, dbal in res["static"]:
            d = chess.Board(fen)
            if d.is_check():
                continue
            ag = self._their_agenda(G.with_turn(d, not self.me))
            theirs = max(ag.state.claims, key=lambda cl: cl.value) if ag.state.claims else None
            if theirs is not None and theirs.value >= dbal:
                return f"after {san_r} the gain {dbal:+d} stands only as a balance and their {theirs.what} wins {theirs.value} (lemma {theirs.lemma}) if we pass"
        return None

    def _chase(self, q: R.Question) -> bool:
        """52.12: the chase of a question that would be killed. -> True when a route CLAIMS (the question is verified)."""
        if self.ctx is None or q.goal not in ("mate", "win") or self.min_stake >= BD.MATE_VALUE or (self.chase_depth >= 2 and q.goal != "mate"):      # (below the first inner level only the mate questions chase) # a mate-only read (the reply already gained, or lost too much for anything but a mate) does not chase on
            return False
        b = self.board
        routes = []
        for m in R._moves(b, self.us):
            why = R.forcing_why(b, m, q.stake, self.us, mate_threats=True)
            if why:
                routes.append(R.Route(m, b.san(m), why))
        if self.chase_depth:                                    # inner levels follow the most forcing moves only: a check, a capture, a mate threat (the quiet "threat n" moves are the root's)
            routes = [r for r in routes if r.why == "check" or r.why.startswith("capture") or r.why == "threat of mate"]
        routes.sort(key=lambda r: (0 if r.why == "check" else 1 if r.why.startswith("capture") else 2))      # the most forcing first: a check, a capture, a threat (stable inside)
        if not routes:
            self.lines.append("  chase: no forcing move on this board")
            return False
        if self.chase_depth == 0:
            self.ctx.left = self.budget                         # the root question's budget, shared by this chase and every inner agenda under it, spent route by route
        self.lines.append(f"  chase: make them move -- {len(routes)} forcing route(s): " + ", ".join(f"{r.san} ({r.why})" for r in routes[:12]) + (" ..." if len(routes) > 12 else ""))
        done = []
        for r in routes:
            res = self._chase_route(r.move)
            if res["gap"]:
                self.lines.append(f"    {r.san}: GAP, {res['gap']}")
                self.state.gaps.append(f"Q{q.id} {q.text}: {res['gap']}")
            elif res["killed"]:
                self.lines.append(f"    {r.san}: killed -- {res['killed']}")
            elif res["value"] >= 1:
                self.lines.append(f"    {r.san}: " + " / ".join(res["parts"][:6]) + (f" / ... {len(res['parts'])} replies" if len(res["parts"]) > 6 else "") + f" => value {res['value']}")
                done.append((res["value"], r, res))
            else:
                self.lines.append(f"    {r.san}: value {res['value']} (no gain on the worst reply)")
        self.lines.append(f"    (chase: {self.ctx.agendas} inner agendas, {self.ctx.asked} inner questions, {self.ctx.budget_hits} budget hit(s), budget left {self.ctx.left})")
        if not done:
            return False
        done.sort(key=lambda t: (-t[0], t[1].move.uci()))
        value, r, res = None, None, None
        for value_, r_, res_ in done:                           # the chosen route is checked by THEIR agenda on every reply that gains only as a static balance (0AgL8's Nxg2: a mating attack)
            bad = self._static_refuted(r_, res_) if self.side_depth else None
            if bad is None:
                value, r, res = value_, r_, res_
                break
            self.lines.append(f"  {r_.san} KILLED as a chase route: {bad}")
        if r is None:
            return False
        net = value
        kind = "mate" if value >= BD.MATE_VALUE else "win"
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
        before = len(self.state.claims)
        self._verify(q, f.routes, mates_only=False)
        if len(self.state.claims) == before and not self._decide_by_smallest_loss(q):
            self.state.gaps.append(f"Q{q.id} {q.text}: no parry binds a guaranteed gain and none could be read from their view")

    # ------------------------------------------------------------------------------------------------------------ promotion (52.6)

    def _ask_promote(self, q: R.Question, head: str) -> None:
        """The defects of the pawn's path are read with the words; a stopper in time kills the question WITH THE DISTANCE REASON; no defect = unstoppable by the
        tempo arithmetic, and then the claim is made ONLY when the proved passer (Ideas.Passer2.unstoppablePasser2_sound, via proved_passers) fires; if it does
        not, the question ends as a GAP naming the hypothesis that fails."""
        b = self.board
        sq = q.pattern["pawn"]
        q.defects, facts = R.promote_defects(b, sq)
        n, promo = facts["pushes"], facts["promo"]
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
            self._kill(q, reason, tempo=True)
            return
        ps = dict(PV.proved_passers(b))
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
        h = PV.passer_v2(b, sq)
        bad = [k for k in PV.HYPS if not h[k]]
        why = []
        for k in bad:
            if k == "enemyKingPawns":
                why.append("enemyKingPawns: " + ", ".join(f"their {b.piece_at(s).symbol().upper()} on {G.name(s)}" for s in chess.SquareSet((b.rooks | b.bishops | b.queens | b.knights) & b.occupied_co[not self.me])))
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
        if runs:
            self._verify(q, runs, mates_only=False)
        if len(self.state.claims) == before:
            self.lines.append(f"  no route binds a guaranteed gain against the passer (the binder has no promotion tail); {arith}")
            if kd > n + 1:
                self.state.gaps.append(f"Q{q.id} {q.text}: no stopper in time ({arith}) and no capture binds")

    # ------------------------------------------------------------------------------------------------------------ outputs

    def trace(self) -> str:
        return "\n".join(self.lines)

    def decision(self) -> Claim | None:
        return self.best[2] if self.best else None
