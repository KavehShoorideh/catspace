"""THE AGENDA (ANALYST_PLAN 52.1, 52.2): inference is GOAL REGRESSION on QUESTIONS, never a search of moves. Root questions are the goals in order of STAKE: mate
their king (100), win their queen 9 / rook 5 / minor 3 / pawn 1, save ours (the biggest of ours that hangs; in check the king is the stake). Each is asked as a
counterfactual (regress.py), its defects generate sub-questions, tempo arithmetic prunes (and says why), a ROUTE (a forcing move supplying a fix) is VERIFIED
by the generic binder over every reply (binder.py) and, if it binds, becomes a CLAIM on the state with the proved lemma behind it.

STOP (principled, no cap): (a) the best claim's value is at least the stake of every open question (a mate ends it); (b) every question is closed; (c) the
budget (questions asked <= our pieces x 8 squares around their king + our captures) is hit: the open questions become GAPS, never a guess.
Among equal values the OPPONENT'S VIEW breaks the tie (a safe check of theirs that forks something of ours is a defect of that future). Memory is not used here.
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
    def __init__(self, state: State):
        R.reset_ids()
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
            if m.promotion or b.is_castling(m) or pc.piece_type == chess.KING or ksq is None:
                continue
            cf = G.with_piece(b, pc, m.to_square, m.from_square)
            if ksq in cf.attacks(m.to_square):
                marks.append((m, pc))
        scored = []
        for m, pc in marks:
            q = R.new_question("mate", R.STAKE_MATE, dict(sym=pc.symbol(), at=m.to_square, frm=m.from_square, king=ksq))
            q.defects = R.mate_defects(b, self.us, R.placement_of(q))
            q.text = f"mate on {G.name(m.to_square)} ({pc.symbol().upper()} {G.name(m.from_square)}->{G.name(m.to_square)}, their king on {G.name(ksq)})"
            scored.append((len(q.defects), -G.worth(pc), q.id, q))
        for _, _, _, q in sorted(scored, key=lambda t: t[:3]):
            self._push(q)
        hang = RS.hanging_of_ours(b, self.us)
        if hang:
            big = max(hang, key=lambda s: (G.worth(b.piece_at(s)), -s))
            self._push(R.new_question("save", G.worth(b.piece_at(big)), dict(target=big), text=f"save our {b.piece_at(big).symbol().upper()} on {G.name(big)} (it hangs)"))
        theirs = [s for s in chess.SQUARES if b.piece_at(s) and b.piece_at(s).color != me and b.piece_at(s).piece_type != chess.KING]
        for s in sorted(theirs, key=lambda s: (-G.worth(b.piece_at(s)), s)):
            self._push(R.new_question("win", G.worth(b.piece_at(s)), dict(target=s),
                                      text=f"win their {b.piece_at(s).symbol().upper()} on {G.name(s)} (worth {G.worth(b.piece_at(s))})"))

    # ------------------------------------------------------------------------------------------------------------ the loop

    def ask(self) -> None:
        while self.heap:
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
            m = chess.Move(q.pattern["frm"], q.pattern["at"])
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
            opp = R.opponent_defects(b, r.move, us)
            self.lines.append(f"  verify {r.san}: binds {bind.schema} ({len(bind.replies)} replies{': ' + ' / '.join(bind.replies[:4]) if len(bind.replies) <= 4 else ''}),"
                              f" value {bind.value}" + (f"; their view: {'; '.join(opp)}" if opp else "; their view: no safe fork"))
            hits.append((bind, opp, r))
        if not hits:
            self._kill(q, f"verify: no route binds ({', '.join(r.san for r in routes[:6])}{' ...' if len(routes) > 6 else ''})"
                       + ("; " + "; ".join(w for w in why if w and "no mate in one" in w)[:300] if mates_only else ""), tempo=False)
            return
        hits.sort(key=lambda h: (-h[0].value, len(h[1])))
        bind, opp, r = hits[0]
        for other, oopp, orr in hits[1:]:
            if other.value == bind.value and len(oopp) > len(opp):
                self.lines.append(f"  {orr.san} ties at {other.value} but its future has a defect of theirs ({'; '.join(oopp)}): {r.san} is preferred")
        kind = "mate" if bind.value >= BD.MATE_VALUE else ("save" if q.goal == "save" else "win")
        claim = Claim(kind, bind.san, bind.value, bind.theorem, self._line_of(bind))
        self.state.claims.append(claim)
        q.status = "verified"
        self.lines.append(f"  -> CLAIM {kind} {claim.what} (value {claim.value}), lemma {claim.lemma}, line {claim.line}")
        key = (claim.value, -len(opp))
        if self.best is None or key > self.best[:2]:
            self.best = (claim.value, -len(opp), claim)

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

    # ------------------------------------------------------------------------------------------------------------ outputs

    def trace(self) -> str:
        return "\n".join(self.lines)

    def decision(self) -> Claim | None:
        return self.best[2] if self.best else None
