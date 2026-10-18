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
"""
from __future__ import annotations

import heapq
import re

import chess

from catspace.incubator.analyst import lean_passer_v2_proto as PV
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import reasoned as RS
from catspace.incubator.analyst.discovery import selfmine as SM
from catspace.incubator.analyst.discovery.state import Claim, State


class Agenda:
    def __init__(self, state: State, side_depth: int = 1, memory: list | None = None):
        R.reset_ids()
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
        self._threats_done = not side_depth
        self._hang: list = []
        self.threat_views: list = []                            # (claim, their agenda) for every threat root
        self._seq = 0
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
            self._ask(q)
        self.lines.append("STOP (b): every question is closed")

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
                self._kill(q, q.reason, tempo=True)
                return
            self.lines.append(f"  tempo: {len(q.defects)} defect(s), each has a forcing fix")
            seen, routes = set(), []
            for f in q.fixes:
                for r in R.all_routes(f):
                    if r.move not in seen:
                        seen.add(r.move)
                        routes.append(r)
        self._verify(q, routes, mates_only=q.goal == "mate")

    def _trace_fix(self, f: R.Question, lvl: int) -> None:
        pad = "  " * lvl
        if f.status == "gap":
            self.lines.append(f"{pad}-> Q{f.id} {f.text}: GAP, {f.reason}")
            self.state.gaps.append(f"Q{f.id} {f.text}: {f.reason}")
        else:
            rs = ", ".join(f"{r.san} ({r.why})" for r in f.routes)
            tail = f"forcing: {rs}" if rs else "no forcing move" + (f" (quiet fixes: {', '.join(f.unforced[:4])})" if f.unforced else "")
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
            self._kill(q, "no route: the fixes supply no move", tempo=False)
            return
        hits, why = [], []
        for r in routes:
            bind, tried = BD.verify_tried(r.move, b, us, mates_only)
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
            if theirs is not None and theirs.value >= q.stake and net <= 0:
                why.append(f"{r.san}: their {theirs.what} wins {theirs.value} (lemma {theirs.lemma}) -- the future loses it")
                self.lines.append(f"  {r.san} KILLED as a route: their {theirs.what} wins {theirs.value} (lemma {theirs.lemma}) -- the future gives them a claim of at least our stake {q.stake} and nets {net}")
                continue
            hits.append((bind, opp, r, net, theirs))
        if not hits:
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
        window = " [a gain WITHIN THE WINDOW: the leaf is not read to quiescence -- 0UqOv's queen was trapped two plies beyond it]" if "balance" in str(claim.lemma) else ""
        self.lines.append(f"  -> CLAIM {kind} {claim.what} (value {claim.value}{'' if net == claim.value else f', net {net} of their best claim'}), lemma {claim.lemma}, line {claim.line}{window}")
        key = (net, -len(opp))
        if self.best is None or key > self.best[:2]:
            self.best = (net, -len(opp), claim)
            self.asked_at_claim = self.asked

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
            return f"{bind.san} (every reply: {len(bind.replies)} replies)"
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

    def _escape_note(self, r: R.Route, tried: list) -> str:
        """Why a mate route did not bind: the replies after which we have no mate in one."""
        b = self.board.copy(stack=False)
        b.push(r.move)
        esc = []
        for rep in list(b.legal_moves):
            c = b.copy(stack=False)
            c.push(rep)
            if not SM.bind_mate(c.fen(), self.us):
                esc.append(b.san(rep))
        notes = [f"{n}: {note}" for n, v, note in tried if note and v is None]
        return f"{r.san} fails" + (f" ({'; '.join(notes)})" if notes else "") + (f", no mate in one after {', '.join(esc[:4])}" if esc and 0 < len(esc) <= 40 else "")

    # ------------------------------------------------------------------------------------------------------------ kills, saves

    def _kill(self, q: R.Question, reason: str, tempo: bool) -> None:
        q.status, q.reason = "killed", reason
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
