"""plan 74.1 / 74.5 THE REASONING GRAPH (Kaveh 2026-10-27: "a chain or a line of reasoning ... lines of reasoning converge ... a graph of reasoning ... if I'm analyzing one line and
a question pops in my head, I queue it for later"; "Do it because it seems generic. As long as it's generic, I'm happy").

GENERIC: nothing here names a game object. A position is an opaque KEY, a read set is a frozenset of opaque locations, a line is a tuple of opaque step names. The game
reaches this module only through what the engine hands it (agenda.py), never through an import.

THE NODES (74.1):
  QUESTION     a goal asked at a position: its text, its key (what it asks about, position independent), its stake, the FRAME it binds on (a read set), its depth, and
               its PROVENANCE (the edges below). State: open | deferred (arose outside the frame of the line in hand: queued, not asked) | asked | served | cut.
  LINE         a route and the reply path to a position: the steps from the root, their trace names, the position key. The root line has no step. State: open |
               pending (its route waits on the questions enqueued at its position) | closed | cut.
  ANSWER       a claim, a kill or a GAP: the fact store's fact (its id is the node's key; the store keeps the read set, the justifications, the listener).
  OBSERVATION  a term evaluated at a position with its reads (74.4: the survey's lines first). Promoted to a QUESTION when a goal's frame touches its reads.
THE EDGES:
  AROSE_FROM   question -> the line (or the observation) it arose from: WHY it was asked.
  SERVES       answer -> the question it answers (a served question has the SERVES edge of an answer derived elsewhere: CONVERGENCE).
  READS        answer -> the read-set key it stands on (the fact store's key and the read set: two lines that reach positions agreeing on the read set meet at one answer).
  REFINES      a chase question -> the question it chases (the question of the parent position whose route made its line).

WHY A SIBLING MODULE, NOT THE FACT STORE ITSELF, AND NOT THE RUST DATALOG (74.1 leaves the choice to the code): the fact store's facts ARE the answers (an ANSWER node's key is the
fact id), so its keys, signatures and listener stay exactly as they are; but the store reads the board to sign a read set (`facts.board_cell`), so the questions, lines and
edges -- which read nothing -- live beside it, generic over position keys. The Rust kernel's Datalog (`ascent` in rules.rs) and the event-loop kernel (agenda.rs, plan 40) hold
string premise keys and conditions of their own; the agenda's questions are Python objects built and answered in Python, so a Rust graph would cross the boundary at every
node for no reading done there: it does not fit without moving the loop itself (72.7's remaining steps, not 74).

Related work: the given-clause worklist (Otter, E: passive / active sets), the justification-based TMS (Doyle 1979; de Kleer 1986), Soar's impasse -> subgoal, AND-OR search (Allis).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

QUESTION, LINE, ANSWER, OBSERVATION = "question", "line", "answer", "observation"
AROSE_FROM, SERVES, READS, REFINES = "arose_from", "serves", "reads", "refines"
NODE_KINDS = (QUESTION, LINE, ANSWER, OBSERVATION)
EDGE_KINDS = (AROSE_FROM, SERVES, READS, REFINES)
PREFIX = "why: "                                         # 74.5: every trace line of the graph carries this prefix (the guard masks it: scripts/trace_identity.py)


@dataclass
class Node:
    id: int
    kind: str
    text: str = ""
    key: object = None                                   # QUESTION: what it asks about; LINE: the position it reaches; ANSWER: the fact id; OBSERVATION: (term, position)
    pk: object = None                                    # the position the node is about
    depth: int = 0
    stake: int = 0
    frame: frozenset = frozenset()                       # QUESTION: the read set it binds on; ANSWER / OBSERVATION: what it read
    steps: tuple = ()                                    # LINE: the steps from the root (opaque names)
    names: tuple = ()                                    # LINE: the same steps as the trace names them
    state: str = "open"
    value: object = None
    label: str = ""                                      # QUESTION: the engine's local name for it ("Q3"), set once it is numbered
    data: dict = field(default_factory=dict)


class Graph:
    """The nodes and edges of one run (the fact store's lifetime: the answers are its facts). Edges are kept both ways; READS goes to a key, not a node."""

    def __init__(self):
        self.nodes: dict = {}
        self.out: dict = defaultdict(list)               # node id -> [(edge kind, target)]
        self.inn: dict = defaultdict(list)               # target -> [(edge kind, node id)]
        self.next_id = 1
        self.answer_of: dict = {}                        # fact id -> the ANSWER node
        self.by_reads: dict = defaultdict(list)          # READS key -> ANSWER node ids (convergence: two lines meet at one answer)
        self.root_line: "int | None" = None
        self.counts = dict(converged=0)                  # questions served an answer another question derived (the rest is read off the nodes: `tally`)

    # ------------------------------------------------------------------------------------------------------------ nodes and edges
    def add(self, kind: str, **kw) -> Node:
        assert kind in NODE_KINDS, kind
        n = Node(self.next_id, kind, **kw)
        self.next_id += 1
        self.nodes[n.id] = n
        return n

    def edge(self, kind: str, src: int, dst) -> None:
        assert kind in EDGE_KINDS, kind
        if (kind, dst) not in self.out[src]:
            self.out[src].append((kind, dst))
            self.inn[dst].append((kind, src))

    def targets(self, nid: int, kind: str) -> list:
        return [d for k, d in self.out.get(nid, ()) if k == kind]

    def sources(self, dst, kind: str) -> list:
        return [s for k, s in self.inn.get(dst, ()) if k == kind]

    # ------------------------------------------------------------------------------------------------------------ the four kinds
    def line(self, parent: "int | None" = None, steps: tuple = (), names: tuple = (), pk=None) -> Node:
        """A LINE: the parent line's steps and these. With no parent it is a root line (the position itself); the first root line is the run's root."""
        if parent is not None:
            p = self.nodes[parent]
            n = self.add(LINE, steps=p.steps + tuple(steps), names=p.names + tuple(names), pk=pk, key=pk, depth=p.depth + 1, data=dict(parent=parent))
        else:
            n = self.add(LINE, steps=tuple(steps), names=tuple(names), pk=pk, key=pk, depth=0)
            if self.root_line is None:
                self.root_line = n.id
        return n

    def question(self, text: str, key, stake: int, depth: int, pk=None, frame: frozenset = frozenset(), arose: "int | None" = None, refines: "int | None" = None,
                 state: str = "open") -> Node:
        """A QUESTION with its provenance: AROSE_FROM `arose` (a line or an observation), REFINES `refines` (the question its line's route chases)."""
        n = self.add(QUESTION, text=text, key=key, stake=stake, depth=depth, pk=pk, frame=frozenset(frame), state=state)
        if arose is not None:
            self.edge(AROSE_FROM, n.id, arose)
        if refines is not None:
            self.edge(REFINES, n.id, refines)
        return n

    def answer(self, fact_id: int, text: str, kind: str, value=None, reads_key=None, question: "int | None" = None, frame: frozenset = frozenset()) -> Node:
        """An ANSWER (the fact store's fact `fact_id`): SERVES the question that derived it, READS its read-set key."""
        n = self.nodes.get(self.answer_of.get(fact_id, -1))
        if n is None:
            n = self.add(ANSWER, text=text, key=fact_id, value=value, frame=frozenset(frame), data=dict(kind=kind))
            self.answer_of[fact_id] = n.id
            if reads_key is not None:
                self.edge(READS, n.id, reads_key)
                self.by_reads[reads_key].append(n.id)
        if question is not None:
            self.edge(SERVES, n.id, question)
        return n

    def observation(self, term, pk, reads: frozenset, value=None, text: str = "") -> Node:
        """74.4: an OBSERVATION -- a term evaluated at a position, with what it read."""
        return self.add(OBSERVATION, text=text, key=(term, pk), pk=pk, frame=frozenset(reads), value=value)

    # ------------------------------------------------------------------------------------------------------------ provenance and convergence
    def asked(self, q: int) -> None:
        n = self.nodes[q]
        n.state = "asked"

    def serve(self, q: int, fact_id: int, text: str = "", kind: str = "", value=None) -> "Node":
        """The question `q` is SERVED the answer of fact `fact_id` (derived before, under whatever line): a SERVES edge, and CONVERGENCE when the answer was derived for another
        question (two lines meet at one answer). -> the answer node."""
        a = self.answer(fact_id, text, kind, value)
        first = [s for s in self.targets(a.id, SERVES) if s != q]
        self.edge(SERVES, a.id, q)
        n = self.nodes[q]
        n.state = "served"
        n.data["served_by"] = a.id
        if first:
            self.counts["converged"] += 1
        return a

    def lookup(self, reads_key) -> "Node | None":
        """Serving by key: the latest ANSWER whose READS key is `reads_key`, or None."""
        ids = self.by_reads.get(reads_key)
        return self.nodes[ids[-1]] if ids else None

    def line_of(self, q: int) -> "Node | None":
        """The line a question arose from (following an observation to the line it was read on, when it arose from one)."""
        for d in self.targets(q, AROSE_FROM):
            n = self.nodes.get(d)
            if n is not None and n.kind == LINE:
                return n
        return None

    def origin(self, a: int) -> "Node | None":
        """The question that DERIVED an answer (the first it serves)."""
        qs = self.targets(a, SERVES)
        return self.nodes[qs[0]] if qs else None

    @staticmethod
    def line_name(ln: "Node | None") -> str:
        if ln is None:
            return "an unknown line"
        if not ln.names:
            return "the position"
        return " ".join(ln.names)

    def why(self, q: int) -> str:
        """74.5: why a question was asked -- "asked: arose from <line> at depth 1 (refines Q1 ...)" / "served: the answer #12 of <line>, its reads agreeing here" / "deferred: ..."."""
        n = self.nodes[q]
        obs = [self.nodes[d] for d in self.targets(q, AROSE_FROM) if self.nodes.get(d) is not None and self.nodes[d].kind == OBSERVATION]
        ln = self.line_of(q)
        src = (f"arose from the observation '{obs[0].text}'" if obs else
               "a root of the position" if ln is not None and not ln.names else f"arose from {self.line_name(ln)}")
        ref = [self.nodes[d] for d in self.targets(q, REFINES) if d in self.nodes]
        tail = f" (refines {ref[0].label or 'N' + str(ref[0].id)} {ref[0].text})" if ref else ""
        at = f" at depth {n.depth}"
        if n.state == "served":
            a = self.nodes[n.data["served_by"]]
            o = self.origin(a.id)
            ol = self.line_of(o.id) if o is not None and o.id != q else None
            whose = f"of {self.line_name(ol)}" if ol is not None else "derived before"
            return f"served: the answer #{a.key} {whose}, its reads agreeing here ({src}{at}{tail})"
        if n.state == "deferred":
            return f"deferred: {src}{at}{tail}; " + ("queued below every open question (not asked for this decision)" if obs else "outside the frame of the line in hand, queued (not asked)")
        if n.state == "asked":
            return f"asked: {src}{at}{tail}"
        if n.state == "open":
            return f"open: {src}{at}{tail}; not asked (its agenda stopped first)"
        how = n.data.get("how")
        return f"{n.state}: " + (f"{how} " if how else "") + f"({src}{at}{tail})"

    def activate(self, q: int, label: str = "") -> None:
        """A queued question is opened (its line's goal is dead: the line waits on it)."""
        n = self.nodes[q]
        n.state = "open"
        n.data["activated"] = True
        if label:
            n.label = label

    def mark_line(self, ln: int, state: str, on: "list | None" = None) -> None:
        """A LINE's state: 'pending' on the questions enqueued at its position (the route waits on them), then 'closed' (answered) or 'cut' (a GAP with its resume point)."""
        n = self.nodes[ln]
        if state == "pending" or n.data.get("pending_on"):
            n.data["was_pending"] = True
        n.state = state
        if on is not None:
            n.data.setdefault("pending_on", []).extend(on)

    # ------------------------------------------------------------------------------------------------------------ 74.4: the promotion rule (`elevate`: the generic module avoids the game word)
    def elevate(self, frame: frozenset, pk=None, refines: "int | None" = None, depth: int = 0) -> list:
        """THE PROMOTION RULE (74.4): every OBSERVATION (of position `pk`, when given) whose reads meet `frame` and that was not raised yet becomes a QUESTION -- AROSE_FROM the
        observation, REFINES the goal whose frame touched it, queued (it is asked when nothing above it is open). -> the new question nodes."""
        out = []
        for n in list(self.nodes.values()):
            if n.kind != OBSERVATION or n.state == "elevated" or (pk is not None and n.pk != pk) or not (n.frame & frame):
                continue
            n.state = "elevated"
            q = self.question(f"is it still so: {n.text}", ("observation", n.key), 0, depth, pk=n.pk, frame=n.frame, arose=n.id, refines=refines, state="deferred")
            n.data["elevated_to"] = q.id
            out.append(q)
        return out

    # ------------------------------------------------------------------------------------------------------------ outputs
    def tally(self) -> dict:
        """The counts the trace and the guard print, read off the nodes' states."""
        t = dict(questions=0, lines=0, answers=0, observations=0, asked=0, served=0, converged=self.counts["converged"], deferred=0, activated=0, other=0,
                 lines_served=0, lines_pending=0, lines_cut=0, elevated=0)
        for n in self.nodes.values():
            if n.kind == QUESTION:
                t["questions"] += 1
                t["asked" if n.state == "asked" else "served" if n.state == "served" else "deferred" if n.state == "deferred" else "other"] += 1
                t["activated"] += 1 if n.data.get("activated") else 0
            elif n.kind == LINE:
                t["lines"] += 1
                t["lines_served"] += n.state == "served"
                t["lines_pending"] += bool(n.data.get("was_pending"))
                t["lines_cut"] += n.state == "cut"
            elif n.kind == ANSWER:
                t["answers"] += 1
            else:
                t["observations"] += 1
                t["elevated"] += n.state == "elevated"
        return t

    def dump(self) -> list:
        """Every node in id order with its provenance (stable text: the guard's provenance list)."""
        out = []
        for n in self.nodes.values():
            e = " ".join(f"{k}>{d if not isinstance(d, tuple) else 'key'}" for k, d in self.out.get(n.id, ()))
            if n.kind == LINE:
                out.append(f"N{n.id} line d{n.depth} {self.line_name(n)} [{n.state}] {e}".rstrip())
            elif n.kind == QUESTION:
                out.append(f"N{n.id} question {n.label} d{n.depth} s{n.stake} {n.text} [{n.state}] {e}".rstrip())
            elif n.kind == ANSWER:
                out.append(f"N{n.id} answer #{n.key} {n.data.get('kind', '')} {n.text} [{n.value}] {e}".rstrip())
            else:
                out.append(f"N{n.id} observation {n.text} [{n.state}] {e}".rstrip())
        out.append(self.summary())
        return out

    def summary(self) -> str:
        t = self.tally()
        return (PREFIX + f"the reasoning graph: {t['questions']} question(s) (asked {t['asked']}, served {t['served']}, deferred {t['deferred']}, other {t['other']}; "
                f"{t['activated']} queued then opened), {t['lines']} line(s) ({t['lines_served']} met a known answer, {t['lines_pending']} waited on queued questions, "
                f"{t['lines_cut']} cut), {t['answers']} answer(s), {t['observations']} observation(s) ({t['elevated']} raised to questions); converged: {t['converged']} question(s) "
                f"served the answer another question derived")
