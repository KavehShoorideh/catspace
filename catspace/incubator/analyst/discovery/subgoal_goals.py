"""PLAN 78.71 A, B: GOALS AS FACTS, THE FIX AS THE SUBGOAL. SYSTEM_SPEC section 7 "Choosing a subgoal, and getting there" (and section 1, the second half of the goal); the piece is the OPPORTUNITY POLICY'S PROPOSER.
A sandbox: nothing calls it in play (the wake will call it later), no player is changed, nothing is written to rules/learned or data/derived. It READS the concrete threat list (threat_concrete.py: concrete_lists, Lines.ask,
verdict_of, response_of; threat_list.py) and the graph of layer 1 (graph_layer1.py: the bear, may_move and shut relations the engine derives from the rules); it duplicates none of them. Game-free: a kind, a role, a place is a
word read from the engine at run time; no unit, place or kind is named here. No engine evaluation, no external model, no assumed worth (the judge of a line is the measured record, as in threat_concrete). A subgoal
orders; it never adds to the value (AGENTS.md rule 7, spec 15). No name of a pattern: a goal is the set of facts of the graph that would make a threat of ours work.

INTERFACE (what this module takes, what it returns, who calls it).
  Facts(variant)                 one per process: the graph of a position as dicts. `of(text)` -> PositionFacts (units, who stands where, bear, may_move, shut with the units between); `replay(text, names)` -> the plies
                                 of a line (who moved, from where to where, what was taken), identities carried from the position the line starts at.
  goals(item, side, c, facts, editor=None, record=None)
                                 THE PROPOSER. `item` = {"start", "moves", "raw", "names", "list"} as for concrete_lists; `side` = the absolute side (0 or 1) that wants the goals; `c` = that item's concrete list (the output
                                 of threat_concrete.concrete_lists for the item, which now also keeps the dropped claims as `dropped_rows`); `facts` a Facts; `editor` (optional, supplied by the printing script, the one
                                 place the game's own text is spelled) an object with `relocate(text, a, b)` -> text or None, used only to find where a unit that is missing could stand. Returns the goals of `side`:
                                 one per threat of ours that does not work yet. A GOAL = {"target": (side, kind, place), "source": why it is a goal (see SOURCES), "row": the row of the concrete list it came from,
                                 "facts": [fact, ...], "unmet": [fact names], "fixes": [fix, ...], "key": a word for what stops it (the chances are tabled by it), "stake", "line": the line that was read,
                                 "two_moves": whether a quiet first move is needed, "lead": that move}. A FACT = {"name", "vars": the units as variables, "status": met | unmet | open | unread, "cause": None or
                                 {"type", "stoppers": [unit], "victim": unit or None, "words"}, "fix": None or {"edits", "words"}}. A UNIT = (side, kind, place) in the names the engine gives. An EDIT is data:
                                 ("remove", place), ("relocate", place, place) or ("move", the move's name); a game's own spelling is given to them only by the printing script.
  works(item, side, c)           the threats of ours that already work (CONFIRMED, no quiet first move): not goals; kept apart and counted.
  subgoal_claims(goal)           THE FOUR CLAIMS of spec 7, each with the referee that can test it by our own reasoning: worth it (if the facts held we would be better off: the concrete line read on the position
                                 with the facts made true, and the stake of the kinds reading), the right fix (with the fixes alone the goal works, and without them it does not: the same read twice, edited and not),
                                 reachable with a chance (the chance that the unmet facts come true within the time, from our own record of play: a table by what stops the goal), the best use of the moves (no other
                                 goal of the position has more chance times stake). In play only our own reasoning and our own estimate referee; the oracle is not read anywhere in this plan.
  Called by: subgoal_measure.py (the checks B and C) and subgoal_print.py (the print) now; the wake's choice of move and the sleep's ladder later. Nothing calls it in play.

PLAN 78.74 A to C (SYSTEM_SPEC 7, the proposer; the interface above is UNCHANGED, a goal only gains fields): a goal for EVERY unit of the other side (subgoal_units.every_unit adds to `c` a row, read by the existing line reader, for each unit the threat
list had none for; goals() then builds from the rows as before) and for the GOAL UNIT (goal_unit_goal: source "goal_unit"; its facts borne_on, covered(place) for each place the rules let it move to, forced_end). A fact is read from the LINE where one was
read (a taking works or fails by the line, never by counting the units that bear) and says so in cause["basis"] ("line", or "graph" where no line was read). Fields added to a goal: `stoppers` (every unit that stops an unmet fact: a set), `fix_set` (each stopper or
brought unit with `left` = the facts still unmet if only it is dealt with, ordered by the fewest left; `removable` False for a unit of the goal kind, which the rules do not let be taken out), `set_edits` (the edits of the whole set), `steps` (where nothing
of ours bears: the unit brought to bear, its route and by how many moves of the graph, 1, 2 or None, then the rest), `n_unmet`, `every` (the row was not in the threat list), and for the goal unit `escapes`. The key names a fact stopped by two units or more with a
suffix (the table of chances is by the key). The default order of goals() is the weakest target first: the fewest unmet facts, then the larger stake; the measure puts the chance of the key between the two.

THE FACTS OF A GOAL (the same four for every threat of ours; units are variables A, T, R, V, B, X):
  bears(A, T)        a unit of ours bears on the target (the graph's `bear`). Unmet: the line to it is shut by units between (named: they are the stoppers; the graph's `shut`), or no unit of ours has a line to it.
  may_take(A, T)     a bearer may take (the graph's `may_move`). Unmet: the bearer is tied (it stands alone between one of their units and one of ours: its leaving uncovers a line onto a unit of ours, the
                     named cause `uncovered_line` of plan 78.60) or our goal-kind unit is borne on and the taking does not answer it.
  unguarded(T)       no unit of theirs bears on the place of the target. Unmet: the units that bear (the retaker named by the line that was read first).
  kept(A, ...)       nothing of ours is taken elsewhere in the line that was read (the `only_guard` and `uncovered_line` causes of plan 78.60, or a unit of ours that was already borne on: `loose`, or a unit that
                     comes into the line: `arrives`). Read from the line; `open` while no line was read.
THE CAUSE comes from the line that was read (what the other side did that ended the line behind or even), and from the graph for the facts the line does not reach. THE FIX of an unmet fact is the stopper taken out of
play (a unit of theirs removed, pinned, blocked or drawn away: one edit, `remove`, stands for all four; the far end is a variable), a unit of ours that stands between moved off the line (the same edit), or, where no
unit has a line to the target, a unit of ours brought to bear (`relocate`: the far end of where it comes from is a variable). A threat that needs only a quiet first move has that move as its fix.
SOURCES of goals, from the concrete list: a claim REFUTED (the taker ends behind), a claim where a unit bears and no taking stands, a claim nothing bears on where no move makes a taking stand, an EXCHANGE row, an
UNREAD row (the line was cut: the cause is unknown and there is no fix: said), a threat that needs a quiet first move (the fix is that move).
"""
from __future__ import annotations

import cs_engine as CE

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import threat_concrete as TC

SOURCES = ("refuted", "exchange", "unread", "no_taking_legal", "no_bearer_no_move", "two_move_not_ahead", "two_moves", "goal_unit")
FACT_NAMES = ("bears", "may_take", "unguarded", "cornered", "kept", "borne_on", "covered", "forced_end")


# ------------------------------------------------------------------------------------------------------------------------------ the graph as dicts

class PositionFacts:
    """One position's graph: units (id -> (side, kind, place)), who stands where, and the relations bear, may_move (the mover's units only) and shut (with the units between)."""

    def __init__(self, text: str, mover: str, units: dict, at: dict, bear: dict, may: dict, shut: dict):
        self.text, self.mover, self.units, self.at, self.bear, self.may, self.shut = text, mover, units, at, bear, may, shut

    def side_of(self, uid: str) -> str:
        return self.units[uid][0]

    def bearers(self, place: str, side: str) -> list:
        """The units of `side` that bear on `place`, in the order of their ids."""
        return sorted(u for u, (s, _k, _p) in self.units.items() if s == side and place in self.bear.get(u, ()))

    def unit(self, uid: str) -> tuple:
        return self.units[uid]


class Facts:
    """One per process: the engine's graph model of the game and what it needs to read a position's facts and to replay a line."""

    def __init__(self, variant: str):
        self.variant = variant
        self.model = GL.model(variant)
        self.static = self.model.static_part()
        self.dist: "dict | None" = None
        self.goal_name: "str | None" = None
        self.trace: tuple = ([], [])
        self.cost = {"positions": 0, "wall": 0.0}

    # ---- a position
    def of(self, text: str) -> PositionFacts:
        pos = self.model.position(text)
        off = pos["offset"]
        nodes = pos["nodes"]
        units, at, bear, may, shut, shut_of = {}, {}, {}, {}, {}, {}
        for k, (t, nid) in enumerate(nodes):
            if t == "unit":
                _tag, side, kind, place = nid.split(":")
                units[nid] = (side, kind, place)
                at[place] = nid
            elif t in ("bear", "may_move", "shut"):
                inner = nid[nid.index("(") + 1:-1]
                u, p = inner.split(",")
                p = p.split(":", 1)[1]
                if t == "bear":
                    bear.setdefault(u, set()).add(p)
                elif t == "may_move":
                    may.setdefault(u, set()).add(p)
                else:
                    shut_of[off + k] = (u, p)
                    shut[(u, p)] = []
        for (t, a, b, _via, _v) in pos["links"]:
            if t == "between" and a in shut_of:
                shut[shut_of[a]].append(nodes[b - off][1])
        self.cost["positions"] += 1
        return PositionFacts(text, pos["mover"], units, at, bear, may, shut)

    def passed_text(self, text: str) -> "str | None":
        r = CE.passed(self.variant, text, [])
        return None if r is None else r[0]

    def facts_for(self, text: str, side_name: str) -> "PositionFacts | None":
        """The facts as the side `side_name` sees them to move (a side that is not to move: after the pass of the game; None where the game gives none)."""
        f = self.of(text)
        if f.mover == side_name:
            return f
        t2 = self.passed_text(text)
        return None if t2 is None else self.of(t2)

    # ---- a line
    def board(self, text: str) -> set:
        return {(o, k, p) for o, k, p, _b in self.model.bearings(text)}

    def replay(self, text: str, names: list) -> list:
        """The plies of a line from the position `text`: per ply {"side", "name", "uid" (the mover's id at the start of the line), "kind", "from", "to", "took": None or (uid, side, kind, place)}. Identities are carried: a unit
        keeps the id it had at the start. Read from the difference of two boards (what disappeared from a place, what appeared): no move is parsed."""
        cur = text
        before = self.board(cur)
        who = {p: f"unit:{o}:{k}:{p}" for o, k, p in before}
        out = []
        self.trace = ([dict(who)], [cur])                          # after every ply: who stands where (the ids carried), and the position's text (for a long line, read once)
        for nm in names:
            st0 = CE.state(self.variant, cur, [])
            mover = st0[1]
            st1 = CE.state(self.variant, cur, [nm])
            nxt = st1[0]
            after = self.board(nxt)
            gone, came = before - after, after - before
            m_out = sorted(x for x in gone if x[0] == mover)
            m_in = sorted(x for x in came if x[0] == mover)
            taken = sorted(x for x in gone if x[0] != mover)
            mv_from = mv_to = mv_kind = uid = None
            if m_out and m_in:
                best = None
                for a in m_out:                                    # pair a unit that left a place with one that arrived: the same kind first (a move of two units pairs them by kind; the first pair is the unit whose id sorts first)
                    for b in m_in:
                        sc = (a[1] != b[1], a[2], b[2])
                        if best is None or sc < best[0]:
                            best = (sc, a, b)
                _sc, a, b = best
                mv_from, mv_to, mv_kind = a[2], b[2], b[1]
                uid = who.get(mv_from)
            took = None
            if taken:
                x = taken[0]
                took = (who.get(x[2]), x[0], x[1], x[2])
            moved = {}                                             # carry the ids: every unit that left a place stands on the place it came to, a unit taken is gone

            for a in m_out:
                cands = [b for b in m_in if b[1] == a[1]] or list(m_in)
                if cands:
                    b = sorted(cands)[0]
                    moved[a[2]] = b[2]
                    m_in = [y for y in m_in if y != b]
            newwho = {p: u for p, u in who.items() if p not in {a[2] for a in m_out} and p not in {x[2] for x in taken}}
            for src, dst in moved.items():
                newwho[dst] = who.get(src)
            who = newwho
            self.trace[0].append(dict(who))
            self.trace[1].append(nxt)
            out.append({"side": mover, "name": nm, "uid": uid, "kind": mv_kind, "from": mv_from, "to": mv_to, "took": took, "outcome": st1[3]})
            cur, before = nxt, after
        return out

    # ---- steps between places (the static `next` relation), for a unit brought to bear
    def steps(self) -> dict:
        if self.dist is None:
            nm = [n for _t, n in self.static["nodes"]]
            adj: dict = {}
            for (t, a, b, _via, _v) in self.static["links"]:
                if t == "next":
                    adj.setdefault(nm[a], set()).add(nm[b])
                    adj.setdefault(nm[b], set()).add(nm[a])
            dist = {}
            for s in adj:
                d = {s: 0}
                q = [s]
                for x in q:
                    for y in adj[x]:
                        if y not in d:
                            d[y] = d[x] + 1
                            q.append(y)
                dist[s] = d
            self.dist = dist
        return self.dist


# ------------------------------------------------------------------------------------------------------------------------------ the causes

def _named(uid: "str | None", f: PositionFacts) -> "tuple | None":
    return None if uid is None or uid not in f.units else f.units[uid]


def line_story(f: PositionFacts, taker: str, t: str, plies: list) -> dict:
    """What the other side did in the line that stopped it: the first unit of theirs that took a unit of ours, and where; every loss of ours with its named cause (plan 78.60: only_guard, uncovered_line; or loose: the
    unit was already borne on; or arrives: the unit takes after coming into the line). `f` = the facts at the start of the line, `t` the target's place, `plies` the replayed line."""
    taker_unit = None
    for p in plies:
        if p["took"] and p["took"][3] == t and p["side"] == taker:
            taker_unit = p["uid"]
            break
    losses = []
    for p in plies:
        if p["took"] is None or p["side"] == taker:
            continue
        x, v_uid, v_place = p["uid"], p["took"][0], p["took"][3]
        if x is None:
            continue
        if v_place == t:
            ctype = "guard" if (x in f.units and t in f.bear.get(x, ())) else "arrives"
        elif v_uid in f.units and f.units[v_uid][2] == v_place and x in f.units:
            if f.shut.get((x, v_place)) == [taker_unit] and taker_unit is not None:
                ctype = "uncovered_line"
            elif v_place in f.bear.get(x, ()):
                others = [g for g in f.bearers(v_place, taker) if g != taker_unit]
                ctype = "loose" if others else "only_guard"
            else:
                ctype = "arrives"
        else:
            ctype = "arrives"
        i = plies.index(p)
        if v_place != t and any(q["side"] == taker and q["took"] and q["took"][3] == v_place for q in plies[i + 1:]):
            continue                                              # a unit of ours taken and taken back at once is a trade, not a loss (the ledger of the line has it)
        losses.append({"type": ctype, "stopper": x, "victim": v_uid, "at": v_place, "ply": i})
    return {"taker_unit": taker_unit, "losses": losses, "ends": bool(plies and plies[-1]["outcome"] is not None)}


def _words(ctype: str) -> str:
    return {
        "guard": "a unit of theirs bears on the target's place and takes back",
        "only_guard": "the taking unit was the only guard of a unit of ours that they now bear on (named cause only_guard)",
        "uncovered_line": "the taking unit leaves a line that was shut: a unit of theirs now bears on a unit of ours (named cause uncovered_line)",
        "loose": "a unit of ours that they already bear on is taken in the line",
        "arrives": "a unit of theirs comes into the line and takes",
        "blocked": "the line from a unit of ours to the target is shut by the units between",
        "no_bearer": "no unit of ours has a line to the target",
        "tied": "the bearer stands alone between a unit of theirs and a unit of ours: its leaving uncovers a line (named cause uncovered_line)",
        "goal_unit_attacked": "our goal-kind unit is borne on and a taking does not answer it",
        "needs_quiet_move": "a quiet move of ours is needed first to bring a unit to bear",
        "escapes": "the target itself moves away in the line (nothing of ours covers where it goes)",
        "out_of_budget": "the line was cut by the fuse: what stops it is not known",
        "ends_game": "the line ends with the game lost for the taker",
        "not_ahead": "the line is read to its end and does not end ahead for the taker (the units of theirs that take in it are named)",
        "no_forced_end": "the forcing read finds no forced end of the game within its reach",
    }.get(ctype, ctype)


def _fix_remove(f: PositionFacts, uid: str, words: str) -> dict:
    return {"edits": [("remove", f.units[uid][2])], "words": words, "units": [f.units[uid]]}


def bring_table(f: PositionFacts, facts: Facts, side: str, editor) -> dict:
    """PLAN 78.74 A: where each unit of `side` could stand to bear on a place, asked of the engine ONCE per position (the bear relation of the position with that one unit moved: not a table of how a kind moves) and kept on the position's facts:
    {(unit id, place): {"bears": the places it then bears on, "en_prise": a unit of the other side then bears on that place, "one": the rules let it move there now (the graph's may_move)}}. Places: the empty places the rules let it move to now,
    and the empty places within two steps of the place graph."""
    tab = getattr(f, "bring", None)
    if tab is not None:
        return tab
    tab = {}
    f.bring = tab
    if editor is None:
        return tab
    dist = facts.steps()
    other = next((x for x in {v[0] for v in f.units.values()} if x != side), None)
    for uid, (s, _k, a) in sorted(f.units.items()):
        if s != side:
            continue
        reach = {q: d for q, d in dist.get(a, {}).items() if 0 < d <= 2}
        for q in f.may.get(uid, ()):
            reach.setdefault(q, dist.get(a, {}).get(q, 9))
        for q in sorted(reach):
            if q in f.at:
                continue
            text2 = editor.relocate(f.text, a, q)
            if text2 is None:
                continue
            g = facts.of(text2)
            u2 = g.at.get(q)
            if u2 is None:
                continue
            tab[(uid, q)] = {"bears": frozenset(g.bear.get(u2, ())), "en_prise": bool(g.bearers(q, other)), "one": q in f.may.get(uid, ()), "d": reach[q]}
    return tab


def route_of(f: PositionFacts, facts: Facts, editor, uid: str, q: str) -> "tuple[int | None, list]":
    """By which moves of the graph a unit comes to a place: (1, [from, to]) when the rules let it move there now; (2, [from, via, to]) when it moves to a place the rules let it reach and from there the rules let it move to the place
    (the second move asked of the graph of the position with the first move made: the side to move is the same); (None, [from, to]) when neither is so (the place is within two steps of the place graph and more moves are needed)."""
    a = f.units[uid][2]
    if q in f.may.get(uid, ()):
        return 1, [a, q]
    for q1 in sorted(f.may.get(uid, ())):
        if q1 in f.at:
            continue
        t1 = editor.relocate(f.text, a, q1)
        if t1 is None:
            continue
        g = facts.of(t1)
        u1 = g.at.get(q1)
        if u1 is not None and q in g.may.get(u1, ()):
            return 2, [a, q1, q]
    return None, [a, q]


def bring_fix(f: PositionFacts, facts: Facts, side: str, t: str, editor) -> "dict | None":
    """THE MISSING BEARER: a unit of ours brought to bear on the place `t`, the far end (where it comes from) a variable. Candidates: a unit of ours moved to an empty place within two steps of where it stands (or to which the rules let it move now) from which
    it bears on `t` (asked of the engine on the edited text, once per position: `bring_table`); the rules' own move first (one move), then a place the other side does not bear on, then the nearest by the steps of the place graph. PLAN 78.74 A: the fix says by which moves
    of the graph the unit comes (`moves`: 1, 2 or None, and the `route`)."""
    if editor is None:
        return None
    best = None
    for (uid, q), e in sorted(bring_table(f, facts, side, editor).items()):
        if t not in e["bears"]:
            continue
        s, k, a = f.units[uid]
        sc = (not e["one"], e["en_prise"], e["d"], k, a, q)
        if best is None or sc < best[0]:
            best = (sc, uid, a, q)
    if best is None:
        return None
    _sc, uid, a, q = best
    moves, route = route_of(f, facts, editor, uid, q)
    return {"edits": [("relocate", a, q)], "words": "a unit of ours comes to bear (the far end of where it comes from is a variable)", "units": [f.units[uid]], "moves": moves, "route": route}


# ------------------------------------------------------------------------------------------------------------------------------ the goals

def _fact(name: str, vars_: dict, status: str, cause: "dict | None" = None, fix: "dict | None" = None) -> dict:
    return {"name": name, "vars": vars_, "status": status, "cause": cause, "fix": fix}


def _cause(ctype: str, stoppers: list, victim, basis: str = "line", victims: "list | None" = None) -> dict:
    """A cause: the type, ALL the units that stop it (a set, PLAN 78.74 B), the victim, and the basis of the claim: `line` (a line read by the engine said so) or `graph` (a relation of the position: no line was read for it)."""
    return {"type": ctype, "stoppers": stoppers, "victim": victim, "victims": victims or ([victim] if victim else []), "words": _words(ctype), "basis": basis}


def _units_of(f: PositionFacts, uids) -> list:
    out, seen = [], set()
    for u in uids:
        if u in f.units and u not in seen:
            seen.add(u)
            out.append(f.units[u])
    return out


def fix_set_of(facts_out: list, goal_kind: "str | None") -> list:
    """PLAN 78.74 B: THE FIX AS A SET. Every unit that stops a fact (the stoppers of every unmet fact) and every unit to bring, each with the facts that stay unmet if only it is dealt with (a fact stays unmet while any stopper of it is left; a fact that names
    no stopper stays), ordered by which single removal leaves the fewest facts unmet, then the fewest stoppers left in all. An element: {"unit", "edit", "removable" (a unit of the goal kind cannot be taken out: the rules forbid it), "kind": remove | bring,
    "resolves": the names of the facts it alone makes met, "left": the count of facts left unmet, "left_stoppers"}."""
    unmet = [x for x in facts_out if x["status"] in ("unmet", "unread")]
    sets = []
    for x in unmet:
        c = x["cause"] or {}
        sets.append({u for u in c.get("stoppers", ())})
    brings = [x for x in unmet if x["fix"] and x["fix"]["edits"] and x["fix"]["edits"][0][0] == "relocate"]
    elems, seen = [], set()
    for x in unmet:
        c = x["cause"] or {}
        for u in c.get("stoppers", ()):
            if u in seen:
                continue
            seen.add(u)
            left = [i for i, st in enumerate(sets) if not st or (st - {u})]
            tot = sum(len(st - {u}) for st in sets)
            elems.append({"unit": u, "edit": ("remove", u[2]), "kind": "remove", "removable": u[1] != goal_kind,
                          "resolves": [unmet[i]["name"] for i in range(len(unmet)) if i not in left], "left": len(left), "left_stoppers": tot})
    for x in brings:
        ed = x["fix"]["edits"][0]
        left = [i for i, y in enumerate(unmet) if y is not x]
        elems.append({"unit": (x["fix"]["units"] or [None])[0], "edit": ed, "kind": "bring", "removable": True, "resolves": [x["name"]], "left": len(left), "left_stoppers": sum(len(st) for st in sets)})
    elems.sort(key=lambda e: (e["left"], e["left_stoppers"], not e["removable"], str(e["edit"])))
    return elems


def build_goal(item: dict, side: int, row: dict, source: str, facts: Facts, editor, text: str, taker: str, owner: str, names: dict) -> "dict | None":
    """One goal from one row of the concrete list (see the module doc for the facts and the causes). `text` is the position's text, `taker` and `owner` the roles' names.
    PLAN 78.74 A: a fact is read from the LINE where a line was read (a taking is said to work or fail by the line, never by counting the units that bear: a unit that bears on the place and does not take back in the line is no stopper); where no line was
    read the fact is a relation of the graph and says so (`basis`). A goal gains fields only: `stoppers` (every unit that stops an unmet fact), `fix_set` (see `fix_set_of`), `set_edits` (the edits of the whole set, units of the goal kind left out),
    `steps` (where nothing of ours bears: bring a unit to bear, by which moves of the graph, then the rest)."""
    t = row["place"] if isinstance(row["place"], str) else names["places"][row["place_ix"]]
    other = owner
    f = facts.facts_for(text, taker)
    if f is None:
        return None
    tgt_uid = f.at.get(t)
    if tgt_uid is None or f.units[tgt_uid][0] != owner:
        return None
    target = f.units[tgt_uid]
    two = bool(row.get("two_moves"))
    line = list(row.get("line") or [])
    plies = []
    if line and source != "unread":
        try:
            plies = facts.replay(f.text, line)
        except Exception:                                          # a line the engine will not replay is read as no line (counted by the caller through the empty plies)
            plies = []
    story = line_story(f, taker, t, plies) if plies else {"taker_unit": None, "losses": [], "ends": False}
    vars_ = {"T": target}
    # ---- bears(A, T)
    bearers = f.bearers(t, taker)
    fixes: list = []
    facts_out: list = []
    steps: list = []
    if bearers:
        a_uid = story["taker_unit"] if story["taker_unit"] in bearers else bearers[0]
        facts_out.append(_fact("bears", {"A": f.units[a_uid], "T": target}, "met"))
    else:
        a_uid = story["taker_unit"]
        shut = sorted(((len(b), u, b) for (u, p), b in f.shut.items() if p == t and f.units[u][0] == taker and b))
        if two:
            lead = line[0] if line else None
            mover_uid = plies[0]["uid"] if plies and plies[0]["side"] == taker else None
            cause = _cause("needs_quiet_move", [], None)
            fix = {"edits": [("move", lead)], "words": "the quiet first move that brings a unit to bear", "units": _units_of(f, [mover_uid])}
            facts_out.append(_fact("bears", {"A": f.units.get(a_uid) if a_uid in f.units else None, "T": target}, "unmet", cause, fix))
            fixes.append(fix)
            if plies and plies[0]["side"] == taker:
                steps.append({"step": "bring", "unit": f.units.get(mover_uid), "route": [plies[0]["from"], plies[0]["to"]], "moves": 1, "by": "the line read (a quiet first move, then the taking)", "move": lead})
        elif shut:
            n, u, btw = shut[0]
            cause = _cause("blocked", [f.units[b] for b in btw], None, "graph")
            fix = {"edits": [("remove", f.units[b][2]) for b in btw], "words": "the units between are moved off the line, taken or drawn away (the far end a variable)", "units": [f.units[b] for b in btw]}
            facts_out.append(_fact("bears", {"A": f.units[u], "T": target}, "unmet", cause, fix))
            fixes.append(fix)
            steps.append({"step": "free the line", "unit": f.units[u], "route": [f.units[u][2], t], "moves": 1, "by": "the graph (the units between are named)"})
        else:
            cause = _cause("no_bearer", [], None, "graph")
            fix = bring_fix(f, facts, taker, t, editor)
            facts_out.append(_fact("bears", {"A": None, "T": target}, "unmet", cause, fix))
            if fix:
                fixes.append(fix)
                steps.append({"step": "bring", "unit": fix["units"][0], "route": fix["route"], "moves": fix["moves"], "by": "the graph (the unit's own moves; the bear relation of the position with it moved)"})
    # ---- may_take(A, T)
    if bearers:
        if f.mover != taker:
            facts_out.append(_fact("may_take", {"A": f.units[a_uid], "T": target}, "open"))
        else:
            legal = [u for u in bearers if t in f.may.get(u, ())]
            if legal:
                facts_out.append(_fact("may_take", {"A": f.units[legal[0]], "T": target}, "met"))
            else:
                a0 = bearers[0]
                tied, victim = [], None
                for u in bearers:
                    for (x, p), btw in sorted(f.shut.items()):
                        if f.units[x][0] == other and btw == [u] and p in f.at and f.units[f.at[p]][0] == taker:
                            tied.append(x)
                            if victim is None:
                                victim, a0 = f.units[f.at[p]], u
                if tied:
                    cause = _cause("tied", _units_of(f, tied), victim, "graph")
                    fix = {"edits": [("remove", f.units[x][2]) for x in dict.fromkeys(tied)], "words": "the unit of theirs that holds the bearer is taken, blocked or drawn away (the far end a variable)", "units": _units_of(f, tied)}
                else:
                    mine = [u for u, (s2, k2, _p2) in sorted(f.units.items()) if s2 == taker and k2 == facts.goal_name]
                    att = sorted({x for u in mine for x in f.bearers(f.units[u][2], other)})
                    cause = _cause("goal_unit_attacked", _units_of(f, att), f.units[mine[0]] if mine else None, "graph")
                    fix = {"edits": [("remove", f.units[x][2]) for x in att], "words": "the unit of theirs that attacks our goal-kind unit is taken, blocked or drawn away (the far end a variable)", "units": _units_of(f, att)} if att else None
                facts_out.append(_fact("may_take", {"A": f.units[a0], "T": target}, "unmet", cause, fix))
                if fix:
                    fixes.append(fix)
    else:
        facts_out.append(_fact("may_take", {"A": None, "T": target}, "open"))
    # ---- unguarded(T): by the line where one was read (the unit of theirs that took back at the place of the target); by the graph where none was (the units that bear on the place)
    guards = f.bearers(t, owner)
    first_guard = next((l for l in story["losses"] if l["type"] in ("guard", "arrives") and l["at"] == t), None)
    if plies:
        if first_guard is None:
            facts_out.append(_fact("unguarded", {"T": target}, "met"))                 # the line shows no taking back at the place: the units that bear are no stopper (pinned, or the taking is not what ends the line)
        else:
            r_uid = first_guard["stopper"]
            stoppers = _units_of(f, [r_uid] + [g for g in guards if g != r_uid])
            if r_uid in f.units:
                cause = _cause("guard", stoppers, None, "line")
                fix = _fix_remove(f, r_uid, "the unit that takes back is taken, pinned, blocked or drawn away (the far end a variable)")
                facts_out.append(_fact("unguarded", {"T": target, "R": f.units[r_uid]}, "unmet", cause, fix))
                fixes.append(fix)
    elif guards:
        cause = _cause("guard", _units_of(f, guards), None, "graph")
        fix = _fix_remove(f, guards[0], "the unit that bears on the place is taken, pinned, blocked or drawn away (the far end a variable)")
        facts_out.append(_fact("unguarded", {"T": target, "R": f.units[guards[0]]}, "unmet", cause, fix))
        fixes.append(fix)
    else:
        facts_out.append(_fact("unguarded", {"T": target}, "met"))
    # ---- cornered(T): in a line that begins with a quiet move of ours, the other side's first reply is the target itself moving away
    if (two or source == "two_move_not_ahead") and len(plies) >= 2:
        rep1 = next((p for p in plies[1:] if p["side"] == owner), None)
        if rep1 is not None and rep1["uid"] == tgt_uid and rep1["took"] is None:
            facts_out.append(_fact("cornered", {"T": target}, "unmet", _cause("escapes", [], None), None))
    # ---- kept(A)
    elsewhere = [l for l in story["losses"] if l["at"] != t]
    if source == "unread":
        facts_out.append(_fact("kept", {}, "unread", _cause("out_of_budget", [], None), None))
    elif not plies:
        facts_out.append(_fact("kept", {}, "open"))
    elif not elsewhere:
        facts_out.append(_fact("kept", {"A": f.units.get(story["taker_unit"])}, "met"))
    else:
        l = elsewhere[0]
        x = l["stopper"]
        v = f.units.get(l["victim"]) if l["victim"] else None
        sts = _units_of(f, [y["stopper"] for y in elsewhere])
        vics = _units_of(f, [y["victim"] for y in elsewhere if y["victim"]])
        if x in f.units:
            cause = _cause(l["type"], sts, v, "line", vics)
            fix = _fix_remove(f, x, "the unit of theirs that takes is taken, pinned, blocked or drawn away (the far end a variable)")
        else:
            cause = _cause(l["type"], [], v, "line", vics)
            fix = None
        facts_out.append(_fact("kept", {"A": f.units.get(story["taker_unit"]), "V": v, "X": f.units.get(x)}, "unmet", cause, fix))
        if fix:
            fixes.append(fix)
    if source != "unread" and plies and not any(x["status"] == "unmet" and x["name"] != "bears" for x in facts_out) and row.get("verdict") != "CONFIRMED":
        # the line did not end ahead but no named fact of the line is unmet (a taking that is not made, or a line that ends level): the line itself ends not ahead for the taker. Say so, and name the units of theirs that take in the line (the stoppers)
        took_by = [p["uid"] for p in plies if p["side"] == owner and p["took"] is not None and p["uid"] is not None and p["uid"] != tgt_uid]       # the target itself taking is not something to take out: it is the goal
        sts = _units_of(f, took_by)
        if story["ends"] and not sts:
            facts_out.append(_fact("kept", {}, "unmet", _cause("ends_game", [], None, "line"), None))
        else:
            fix = {"edits": [("remove", u[2]) for u in sts], "words": "the units of theirs that take in the line are taken, pinned, blocked or drawn away (the far end a variable)", "units": sts} if sts else None
            facts_out.append(_fact("kept", {"A": f.units.get(story["taker_unit"]), "X": sts[0] if sts else None}, "unmet", _cause("not_ahead", sts, None, "line"), fix))
            if fix:
                fixes.append(fix)
    if source != "unread" and not any(x["status"] == "unmet" for x in facts_out) and row.get("verdict") != "CONFIRMED":
        facts_out.append(_fact("kept", {}, "unmet", _cause("ends_game" if story["ends"] else "arrives", [], None), None))       # as in plan 78.71: no line to read and nothing named
    return _finish(item, source, row, facts, target, facts_out, fixes, steps, line, two, taker, owner, plies)


def layer_one(facts_c: list, steps_c: list, limit: int = 2) -> "int | None":
    """PLAN 78.74 (spec 7, reachable, layer one: OUR OWN MOVES ONLY, the other side passing): the number of our moves that make the unmet facts true, or None. Computed from the unmet facts and the steps (tuples (name, status, cause type) and (step, moves)):
    no unmet fact: 0; a fact of bearing (bears, borne_on, covered) that needs our unit brought or a line freed counts the moves of its step (1 or 2, by the rules' own moves of the graph; a step with no known count: None); any other unmet fact (a unit of theirs
    that takes back or takes elsewhere, a line that ends level, the forced end) cannot be made true by our moves alone with the other side passing, as far as this build computes it: None. A total above `limit` moves: None."""
    unmet = [f for f in facts_c if f[1] in ("unmet", "unread")]
    if not unmet:
        return 0
    if any(f[0] not in ("bears", "borne_on", "covered") or f[2] not in ("no_bearer", "blocked", "needs_quiet_move") for f in unmet):
        return None
    mv = [m for _s, m in steps_c]
    if len(mv) < len(unmet) or any(m is None for m in mv):
        return None
    tot = sum(mv)
    return tot if tot <= limit else None


def _finish(item: dict, source: str, row: dict, facts: Facts, target: tuple, facts_out: list, fixes: list, steps: list, line: list, two: bool, taker: str, owner: str, plies: list) -> dict:
    """The goal dict from its facts: the unmet names, the key, the fixes (as in plan 78.71) and, new in plan 78.74, the stoppers, the fix as a set and its edits."""
    unmet = [x["name"] for x in facts_out if x["status"] in ("unmet", "unread")]
    key = "|".join(sorted(f"{x['name']}:{x['cause']['type']}" + (f"*{min(3, len(x['cause']['stoppers']))}" if len(x["cause"]["stoppers"]) > 1 else "") for x in facts_out if x["status"] in ("unmet", "unread") and x["cause"]))   # PLAN 78.74 C: a fact stopped by two units or more is a different chance from one stopped by one (the table is by this key)
    seen, uniq = set(), []
    for fx in fixes:                                               # the same edit once among the fixes' edits
        for e in fx["edits"]:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
    stoppers = []
    for x in facts_out:
        if x["status"] in ("unmet", "unread") and x["cause"]:
            for u in x["cause"]["stoppers"]:
                if u not in stoppers:
                    stoppers.append(u)
    fset = fix_set_of(facts_out, facts.goal_name)
    set_edits, used = [], set()
    for e in fset:                                                 # the set's edits: a unit once, a unit of the goal kind left out (the rules do not let it be taken out), a brought unit not also taken out
        if not e["removable"]:
            continue
        ed = e["edit"]
        src = ed[1]
        if src in used:
            continue
        used.add(src)
        set_edits.append(ed)
    return {"target": target, "source": source, "row": row, "facts": facts_out, "unmet": unmet, "fixes": fixes, "edits": uniq, "key": key, "stake": row.get("stake"), "line": line, "two_moves": two,
            "lead": (line[0] if (two and line) else None), "verdict": row.get("verdict"), "taker": taker, "owner": owner, "net": row.get("net"), "plies": plies,
            "stoppers": stoppers, "fix_set": fset, "set_edits": set_edits, "steps": steps, "n_unmet": len(unmet), "every": bool(row.get("every")),
            "own_moves": layer_one([(x["name"], x["status"], x["cause"]["type"] if x["cause"] else None) for x in facts_out], [(s["step"], s["moves"]) for s in steps]),     # layer one of reachable: our moves only, the other side passing
            "with_replies": row.get("verdict")}                                                                                                            # layer two: the read with the other side's replies (the line's verdict; None for the goal unit)


def goal_unit_goal(item: dict, side: int, c: dict, facts: Facts, editor, text: str, taker: str, owner: str, names: dict) -> "dict | None":
    """PLAN 78.74 A: THE GOAL UNIT as a target. It cannot be taken, so the claim is the forced end of the game the forcing read looks for (a row of `c["goal"]` where one is proved or suspected: then the goal works and is no goal). The facts: it is borne on
    (a unit of ours bears on its place: else what shuts the line or which unit could be brought), and for every place the rules let it move to: that place is covered (a unit of ours bears on it: else what shuts the line, or which unit could be brought,
    by which moves); and the forced end itself, unmet when the forcing read found none (`unread` when its fuse cut the read). `escapes` lists each place with who covers it; a place the goal unit's own side holds is not an escape (the rules do not let it move there)."""
    owner_ix = names["roles"].index(owner)
    if any(r.get("owner") == owner_ix for r in c.get("goal", ())):
        return None
    ft = facts.facts_for(text, taker)
    fo = facts.facts_for(text, owner)
    if ft is None or fo is None:
        return None
    gu = sorted(u for u, (s, k, _p) in fo.units.items() if s == owner and k == facts.goal_name)
    if not gu:
        return None
    uid = gu[0]
    target = fo.units[uid]
    t = target[2]
    tuid = ft.at.get(t)
    if tuid is None:
        return None
    vars_t = {"T": target}
    facts_out, fixes, steps, escapes = [], [], [], []
    bears = ft.bearers(t, taker)
    if bears:
        facts_out.append(_fact("borne_on", {"A": ft.units[bears[0]], "T": target}, "met"))
    else:
        shut = sorted(((len(b), u, b) for (u, p), b in ft.shut.items() if p == t and ft.units[u][0] == taker and b))
        if shut:
            _n, u, btw = shut[0]
            fix = {"edits": [("remove", ft.units[b][2]) for b in btw], "words": "the units between are moved off the line (the far end a variable)", "units": [ft.units[b] for b in btw]}
            facts_out.append(_fact("borne_on", {"A": ft.units[u], "T": target}, "unmet", _cause("blocked", [ft.units[b] for b in btw], None, "graph"), fix))
        else:
            fix = bring_fix(ft, facts, taker, t, editor)
            facts_out.append(_fact("borne_on", {"A": None, "T": target}, "unmet", _cause("no_bearer", [], None, "graph"), fix))
            if fix:
                steps.append({"step": "bring", "unit": fix["units"][0], "route": fix["route"], "moves": fix["moves"], "by": "the graph"})
        if fix:
            fixes.append(fix)
    for p in sorted(fo.may.get(uid, ())):
        cov = ft.bearers(p, taker)
        occupant = ft.units[ft.at[p]] if p in ft.at else None
        place_var = ("", "place", p)
        if cov:
            escapes.append({"place": p, "covered": True, "by": [ft.units[u] for u in cov], "occupant": occupant})
            facts_out.append(_fact("covered", {"T": target, "P": place_var, "A": ft.units[cov[0]]}, "met"))
            continue
        shut = sorted(((len(b), u, b) for (u, q), b in ft.shut.items() if q == p and ft.units[u][0] == taker and b))
        if shut:
            _n, u, btw = shut[0]
            fix = {"edits": [("remove", ft.units[b][2]) for b in btw], "words": "the units between are moved off the line (the far end a variable)", "units": [ft.units[b] for b in btw]}
            cause = _cause("blocked", [ft.units[b] for b in btw], None, "graph")
            facts_out.append(_fact("covered", {"T": target, "P": place_var, "A": ft.units[u]}, "unmet", cause, fix))
        else:
            fix = bring_fix(ft, facts, taker, p, editor)
            facts_out.append(_fact("covered", {"T": target, "P": place_var}, "unmet", _cause("no_bearer", [], None, "graph"), fix))
            if fix:
                steps.append({"step": "bring", "unit": fix["units"][0], "route": fix["route"], "moves": fix["moves"], "by": "the graph", "for": p})
        if fix:
            fixes.append(fix)
        escapes.append({"place": p, "covered": False, "by": [], "occupant": occupant, "fix": fix})
    gap = any("out of budget" in k for k in c.get("dropped", {}))
    facts_out.append(_fact("forced_end", vars_t, "unread" if gap else "unmet", _cause("out_of_budget" if gap else "no_forced_end", [], None, "line"), None))
    row = {"place": t, "stake": None, "verdict": None, "net": None, "two_moves": False}
    g = _finish(item, "goal_unit", row, facts, target, facts_out, fixes, steps, [], False, taker, owner, [])
    g["escapes"] = escapes
    return g


def _source_of(row: dict) -> "str | None":
    v = row.get("verdict")
    if row.get("why"):
        return row["why"]
    if v == "EXCHANGE":
        return "exchange"
    if v == "UNREAD":
        return "unread"
    if v == "CONFIRMED" and row.get("two_moves"):
        return "two_moves"
    return None


def rows_for(c: dict, side: int) -> "tuple[list, list]":
    """(the rows of the concrete list that are goals for `side`, the rows that already work): a row is a threat of `side` when the unit threatened is the other side's."""
    goals_rows, works_rows = [], []
    pool = list(c["mover"]) + list(c["other"]) + list(c["exchange"]) + list(c["dropped_rows"])
    for r in pool:
        if r.get("goal") or r.get("owner") == side:
            continue
        s = _source_of(r)
        if s is None:
            works_rows.append(r)
        else:
            goals_rows.append((r, s))
    return goals_rows, works_rows


def goals(item: dict, side: int, c: dict, facts: Facts, editor=None, record=None) -> list:
    """THE PROPOSER: the goals of `side` at the position of `item` (see the module doc). PLAN 78.74 A: one goal per unit of the other side whose taking does not work yet (the rows of `c` are the threat list's AND, when subgoal_units.every_unit
    has been run on `c`, one row for every other unit) and one for the goal unit. Ordered by how weak the target is (PLAN 78.74 C): the fewest unmet facts, then the largest stake (the chance of reaching is applied by the measure's table between the two)."""
    names = item["names"]
    if facts.goal_name is None:
        facts.goal_name = G2M.goal_kind_of(facts.variant)             # the name of the goal kind, as the engine words it
    roles = names["roles"]
    taker, owner = roles[side], roles[1 - side]
    text = CE.state(facts.variant, item.get("start"), list(item["moves"]))[0]
    out = []
    rs, _w = rows_for(c, side)
    for row, source in rs:
        g = build_goal(item, side, row, source, facts, editor, text, taker, owner, names)
        if g is not None:
            out.append(g)
    gu = goal_unit_goal(item, side, c, facts, editor, text, taker, owner, names)
    if gu is not None:
        out.append(gu)
    out.sort(key=lambda g: (len(g["unmet"]), -(g["stake"] if g["stake"] is not None else 9.0), g["target"][2]))
    return out


def works(item: dict, side: int, c: dict) -> list:
    return rows_for(c, side)[1]


# ------------------------------------------------------------------------------------------------------------------------------ the four claims

def subgoal_claims(goal: dict) -> list:
    """The four claims of spec 7 for one goal, each with the referee that can test it by our own reasoning (data: the measure runs the referees)."""
    ed = goal["edits"]
    return [
        {"name": "worth", "statement": "if the facts held, the taking of the target would end ahead for us by more than the record can tell apart, and the target's stake is above zero",
         "referee": "the concrete line read on the position with the fixes made true (a what-if edit; our own reasoning) and the stake of the kinds reading (our own estimate)", "needs": ed},
        {"name": "right fix", "statement": "with the fixes alone the goal works, and without them it does not",
         "referee": "the same line read twice, on the edited position and on the position re-spelt with no edit (and with a control edit: a unit of the same side taken out that is not the stopper)", "needs": ed},
        {"name": "reachable", "statement": "the unmet facts come true within the time, given the opponent, with a chance",
         "referee": "our own record of play: the share of goals with this key whose unmet facts later held, tabled on the games the table is not graded on (calibration printed)", "needs": goal["key"]},
        {"name": "best use of the moves", "statement": "no other goal of the position has a larger chance times stake",
         "referee": "our own estimate: the chance of the table times the stake of the kinds reading, compared among the goals of the position; the hindsight of plan 78.71 C asks whether the goal then won was among the first few", "needs": goal["key"]},
    ]
