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

from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import reasoned as RS
from catspace.incubator.analyst.discovery import selfmine as SM
from catspace.incubator.analyst.discovery.state import Claim, State


class Agenda:
    def __init__(self, state: State, side_depth: int = 1):
        R.reset_ids()
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
        heapq.heappush(self.heap, (-q.stake, q.depth, self._seq, q))
        return q

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
        theirs = [s for s in chess.SQUARES if b.piece_at(s) and b.piece_at(s).color != me and b.piece_at(s).piece_type != chess.KING]
        for s in sorted(theirs, key=lambda s: (-G.worth(b.piece_at(s)), s)):
            self._push(R.new_question("win", G.worth(b.piece_at(s)), dict(target=s),
                                      text=f"win their {b.piece_at(s).symbol().upper()} on {G.name(s)} (worth {G.worth(b.piece_at(s))})"))

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
        c = self.board.copy(stack=False)
        c.push(route)
        if not any(True for _ in c.legal_moves):
            return None, None
        ag = self._their_agenda(c)
        best = max(ag.state.claims, key=lambda x: x.value) if ag.state.claims else None
        return best, ag

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
            if not self._threats_done and not (self.best and self.best[0] >= R.STAKE_MATE) and (not self.heap or self.heap[0][3].stake < R.STAKE_MATE):
                self._threats_done = True                        # a mate of ours ends the game, so their threats are asked only once no mate root is left to ask
                self._threat_roots(self._hang)
            if not self.heap:
                break
            nstake = self.heap[0][3].stake
            if self.best and self.best[0] >= nstake:
                self.lines.append(f"STOP (a): the best claim ({self.best[2].what}, {self.best[0]}) is at least the stake of every open question (next: {nstake})")
                return
            if self.asked >= self.budget:
                left = [t[3] for t in self.heap]
                self.lines.append(f"STOP (c): budget {self.budget} questions asked; {len(left)} open questions become gaps")
                self.state.gaps += [f"budget: {q.text} (stake {q.stake}) not asked" for q in left]
                return
            _, _, _, q = heapq.heappop(self.heap)
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
        self.state.claims.append(claim)
        q.status = "verified"
        self.lines.append(f"  -> CLAIM {kind} {claim.what} (value {claim.value}{'' if net == claim.value else f', net {net} of their best claim'}), lemma {claim.lemma}, line {claim.line}")
        key = (net, -len(opp))
        if self.best is None or key > self.best[:2]:
            self.best = (net, -len(opp), claim)

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
        if len(self.state.claims) == before:
            self.state.gaps.append(f"Q{q.id} {q.text}: no parry binds a guaranteed gain (deciding by a parry that merely keeps the balance is not in this slice)")

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
        if len(self.state.claims) == before:
            self.state.gaps.append(f"Q{q.id} {q.text}: no parry binds a guaranteed gain without a worse claim of theirs (deciding by a parry that merely keeps the balance is not in this slice)")

    # ------------------------------------------------------------------------------------------------------------ outputs

    def trace(self) -> str:
        return "\n".join(self.lines)

    def decision(self) -> Claim | None:
        return self.best[2] if self.best else None
