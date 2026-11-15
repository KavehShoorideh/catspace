"""PLAN 78.55: the readable Python mirror of ONE frame (rust/engine/tactics/src/sheet/frames.rs), made independently of the Rust code so the two can be tested against each other. Game-free: it reads a game
only through an ADAPTER (a small object, below), names no unit, place or kind, and makes no number of an outcome: outcomes are compared by inclusion of the units lost, and what a side can force is a SET.

THE ADAPTER (what a game must say; the tests supply one for the game they play, the module holds no game):
    adapter.cells(state)                  sorted list of units, each (owner word, kind word, place text)
    adapter.mover(state)                  the owner word of the side to move, None when the game is over
    adapter.moves(state)                  every legal move of the side to move as (from place, to place, next state)
    adapter.pass_(state)                  the same position with the other side to move, nothing played; None where the game refuses
    adapter.key(state)                    a hashable key of the position
    adapter.bearers(state, place, lifted) the units that bear on `place` once the units standing on the places in `lifted` are taken off the board (the October `square_frame` lifting)

THE FRAME (SYSTEM_SPEC 12, plan 78.55 A). The seed's places P (its own and its units' places); the members are the units on P and every unit that bears on a place of P once the members that do not stand on P
are lifted off the board, to a fixed point (the units behind a unit of the frame appear when it leaves: the October rule). The Rust closure uses the layer-1 `through` relation instead, which sees one unit
behind one unit; the tests compare the two and say where they differ.

THE READING. The side `reading` can force a SET of outcomes; an outcome is (units the reading side loses, units the other side loses); one outcome is at least as good as another when it loses no unit the
other keeps and the other side loses every unit it loses there. At a node of the reading side: the unbeaten of the options' guarantees; at the other side's node: the unbeaten of the meets, one guarantee
from each reply (loses: union; wins: intersection). Options of the side to move: a move of a member onto a place of P ("inside"); a PASS (the game's pass, and realisable: some move of a unit that is not a
member leaves the frame's projection as it was) while the side has a pass left in its budget; for the side that is NOT the one read, a LEAVE (a member to a place outside P, or a non-member move that
changes the projection) which ends the line as it stands; a side with no option takes a forced pass if it can, else the line is stuck (the worst outcome for the side read, as it stands for the other).
The depth is one ply per member plus one; the fuse counts positions read; a line the fuse ends is the worst for the side read and is counted unread.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import unit_fates as UF

ALL = 255
MAX_PAIRS = 128


def at_least(a: tuple, b: tuple) -> bool:
    """Outcome a is at least as good as b for the reading side: loses no unit b keeps, the other side loses every unit it loses in b."""
    return a[0] <= b[0] and b[1] <= a[1]


def meet(a: tuple, b: tuple) -> tuple:
    return (a[0] | b[0], a[1] & b[1])


def unbeaten(pairs) -> list:
    ps = list(set(pairs))
    return [a for a in ps if not any(b != a and at_least(b, a) for b in ps)]


class Frame:
    def __init__(self, places: frozenset, members: list):
        self.places = places
        self.members = sorted(members)


def uid(cell: tuple) -> str:
    return f"unit:{cell[0]}:{cell[1]}:{cell[2]}"


def members_by_lifting(adapter, state, places: frozenset, seed_units=()) -> list:
    """The members: the units on P, the units that bear on a place of P with the members that are not on P lifted off the board, to a fixed point."""
    cells = adapter.cells(state)
    members = {c for c in cells if c[2] in places} | set(seed_units)
    while True:
        lifted = frozenset(c[2] for c in members if c[2] not in places)
        grown = set(members)
        for p in places:
            grown |= set(adapter.bearers(state, p, lifted))
        if grown == members:
            return sorted(members)
        members = grown


def frame_of(adapter, state, seed_units, seed_places) -> Frame:
    places = frozenset(set(seed_places) | {c[2] for c in seed_units})
    return Frame(places, members_by_lifting(adapter, state, places, seed_units))


class Reader:
    def __init__(self, adapter, frame: Frame, fuse: int = 20000):
        self.a, self.f, self.fuse = adapter, frame, fuse
        self.ids = {c: i for i, c in enumerate(frame.members)}
        self.depth = len(frame.members) + 1
        self.unread = 0
        self.nodes = 0

    def projection(self, state) -> list:
        return members_by_lifting(self.a, state, self.f.places)

    def expand(self, state, tracked: dict):
        """-> (mover, inside [(next state, tracked, lost ids, taken [(victim id, captor id)])], pass state or None, leaves)."""
        a = self.a
        mover = a.mover(state)
        if mover is None:
            return None, [], None, False
        prev = a.cells(state)
        inside, leaves, outsiders = [], False, []
        for (frm, to, nxt) in a.moves(state):
            cell = next((c for c in tracked if c[0] == mover and c[2] == frm), None)
            if cell is None:
                outsiders.append((frm, to, nxt))
            elif to in self.f.places:
                t2, lost, taken = self.after(prev, a.cells(nxt), tracked)
                inside.append((nxt, t2, lost, taken, (frm, to)))
            else:
                leaves = True
        here = self.projection(state)
        neutral, changing = False, leaves
        for (frm, to, nxt) in outsiders:
            if to in self.f.places or frm in self.f.places:
                ok = False
            else:
                ok = self.projection(nxt) == here
            neutral, changing = (True, changing) if ok else (neutral, True)
        return mover, inside, (a.pass_(state) if neutral else None), changing

    def after(self, prev, nxt, tracked):
        steps, _ = UF.steps_between(prev, nxt)
        out, lost, victims = {}, set(), []
        for cell, uid_ in tracked.items():
            st = steps[prev.index(cell)]
            if st[0] == "alive":
                out[nxt[st[1]]] = uid_
            elif st[0] == "taken":
                lost.add(uid_)
                victims.append((uid_, cell[2]))
        taken = [(v, next((i for c, i in out.items() if c[2] == at), None)) for v, at in victims]
        return out, frozenset(lost), taken

    def solve(self, state, tracked, reading, bud, depth, memo):
        if self.nodes >= self.fuse:
            self.unread += 1
            return [(frozenset(self.f.members), frozenset())]
        key = (self.a.key(state), tuple(sorted(tracked.items())), reading, bud, depth)
        if key in memo:
            return memo[key]
        self.nodes += 1
        before = self.unread
        mover, inside, pas, leaves = self.expand(state, tracked)
        if mover is None:
            return [(frozenset(), frozenset())]
        if depth == 0:
            return [(frozenset(), frozenset())]
        owner = {i: c[0] for c, i in self.ids.items()}
        sets = []
        for (nxt, t2, lost, taken, _ends) in inside:
            sub = self.solve(nxt, t2, reading, bud, depth - 1, memo)
            lx = frozenset(self.f.members[i] for i in lost if self.f.members[i][0] == reading)
            lo = frozenset(self.f.members[i] for i in lost if self.f.members[i][0] != reading)
            sets.append([(p[0] | lx, p[1] | lo) if p != (frozenset(self.f.members), frozenset()) else p for p in sub])
        side = 0 if mover == reading else 1
        if bud[side] > 0 and pas is not None:
            spent = list(bud)
            if spent[side] != ALL:
                spent[side] -= 1
            sets.append(self.solve(pas, tracked, reading, tuple(spent), depth - 1, memo))
        if mover != reading and leaves:
            sets.append([(frozenset(), frozenset())])
        if not sets and pas is not None:
            sets.append(self.solve(pas, tracked, reading, bud, depth - 1, memo))
        if not sets:
            result = [(frozenset(self.f.members), frozenset())] if mover == reading else [(frozenset(), frozenset())]
        elif mover == reading:
            result = unbeaten([p for s in sets for p in s])
        else:
            acc = sets[0]
            for s in sets[1:]:
                acc = unbeaten([meet(x, y) for x in acc for y in s])
            result = acc
        if self.unread == before:
            memo[key] = result
        return result

    def forceable(self, state, reading, flipped=False, bud=None) -> "list | None":
        """The unbeaten guaranteed outcomes of `reading`, as [(frozenset of unit ids it loses, frozenset of unit ids the other loses)]; None when the flipped root cannot be made."""
        bud = bud if bud is not None else sound_budget()
        tracked = {c: i for i, c in enumerate(self.f.members)}
        root = state
        if flipped:
            mover, _inside, pas, _l = self.expand(state, tracked)
            if pas is None:
                return None
            root = pas
        self.unread, self.nodes = 0, 0
        pairs = self.solve(root, tracked, reading, bud, self.depth, {})
        return sorted((tuple(sorted(uid(c) for c in p[0])), tuple(sorted(uid(c) for c in p[1]))) for p in pairs)


def sound_budget() -> tuple:
    """(passes of the reading side, passes of the other side): the reading side spends none and the other may pass at every ply, the form of the Lean lemma `frame_sound_pass`."""
    return (0, ALL)


def holds(pairs, kind: str, unit: str) -> bool:
    return any((kind == "take_free" and unit in lo and not lx) or (kind == "take_any" and unit in lo) or (kind == "safe" and unit not in lx) for lx, lo in pairs)
