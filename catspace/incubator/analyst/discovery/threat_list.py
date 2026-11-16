"""PLAN 78.70 A, B, C: THE THREAT LIST OF A POSITION. SYSTEM_SPEC section 8 "The threat list" (the rule for choosing a move reads it later; it is NOT built here) and section 4 ("open tactics are carried
forward", "how a line ends"); the piece is the FRAME READ'S view for the wake: what is threatened, by what, what it is worth, and what answers it. A sandbox: nothing is wired into any player, nothing is written
to rules/learned or data/derived. Game-free: a kind, a role, a place is a word read from the engine at run time; no unit, place or kind is named here. No engine evaluation, no labeller, no external model, no
assumed worth.

INTERFACE (what this module takes, what it returns, who calls it).
  ThreatReader(threads)        one per process (the model is built once). `read(games, plies)` takes games as (start text or None, moves) pairs and, per game, the plies wanted; returns {"kinds", "roles",
                               "places", "rows": [...]}: per wanted row the frames read, each with its members and places (the READ SET of every claim made in it) and, per claim, the units that bear on
                               the unit, its class and basis, and the answers of its owner with their reason. The frame read itself is the Rust `BaseGraph.questions_threats` (rust/engine/py), which reuses
                               the 78.57 to 78.60 read (sheet/questions.rs) and adds `QReader::answers`; this module reads nothing else of the engine.
  threat_rows(raw, names, stake_of=None)
                               takes one raw row and the names; returns {"mover": [...], "other": [...]}: the rows of the list for each side, by the owner of the unit threatened (the list "against" that
                               side: the threats against the mover are theirs, those against the other side are ours). Each row: unit (kind, place), side, by (the units that bear on it), class (lost,
                               both, threat, conditional, contested: lost is kept apart from contested), frames (the smallest frame that wins and every wider frame that holds the unit, each named by its units
                               and with its class and answers), response (a list of (move, reason) or empty), reason of the first, widened (the response was found in a wider frame than the one that wins), ended
                               (how the line ended, the four reasons of spec 4: concrete / quiet / lost beyond doubt / out of budget; the read produces the first, second and fourth), read_set (the members and
                               places of the winning frame), stake (see below, or None).
  threat_list(moves, ply, reader, start=None, stake_of=None, spread=None)
                               the list of ONE position (the game's moves up to `ply`): `threat_rows` of its raw row, each row given its band (below).
  band_of(row, spread, rank_high)
                               the band 1 to 4 of a row: 1 high with no response, 2 high with a response, 3 low with no response, 4 low with a response. `high` is decided two ways, printed apart: by the SPREAD of the
                               two readings at the position (stake above it: spec 8, "PROPOSED") and by the stake RANKED within the position (`rank_high`: set by `rank_high_of`).
  Called by: threat_measure.py (the measure) and threat_print.py (the print) now; the wake's choice of move and the sleep's ladder later. No player calls it.

THE STAKE of a row = the change, in the kinds reading's chance of the result, for the unit's owner, if the unit is lost: its kind leaves the owner's kinds (the reading of 78.69's two_readings, the head on the
kinds of units alone); never a worth table and never a unit count as a value. `stake_of(kind_word, owner_is_mover)` is given by the caller (the measure builds it from the held-out rows' inputs). A unit of the
goal kind is not among the kinds head's inputs: its rows have stake None and are counted apart.

THE CLASS words are those of the 78.62 state record (tactical_state.CLASS_WORDS): lost (the other side forces its taking now), both, threat (it is taken if the side to move passes once), conditional (after one
arrival of the other side), contested (the other side bears on it and nothing is forced). THE ANSWERS are the candidate moves of the read (the Q1 set of 78.57) after which the other side can no longer force the
unit's taking for nothing; a candidate's class is its reason: a taking, the unit moves, a guard is added, a counter threat, a unit is put between. For a unit of the side NOT to move the answers are those of its
owner after the side to move has passed (the flipped root of a threat): what the owner would answer, not a move available now.

THE WIDENING (B). A unit that is lost (or both) in a frame and is not in a frame that contains the first one with answers is one row with no response. A unit lost in the smallest frame and with an answer in a
frame that contains the winning frame's units and more is ONE row "with a response", both frames named (`widened`).

HOW A LINE ENDED (spec 4), as the read reports it: the basis of the claim (forced, pruned by inclusion: CONCRETE, decided inside the frame; an exchange read by the estimate or a line ended by a bound or by
the quiet plies: QUIET, a settled end; lines the fuse left unread: OUT OF BUDGET). LOST BEYOND DOUBT is a verdict of the readings, not of the frame read, and is not produced here. This mapping is a choice of this
build (spec 4 does not map the read's bases to its four reasons), stated in the plan note.
"""
from __future__ import annotations

import time
from pathlib import Path

from catspace.incubator.analyst.discovery import frames_measure as OLD
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.io import paths

CLASS = TS.CLASS_WORDS                                   # lost, safe, both, unread, threat, conditional, contested
THREAT_CLASSES = (0, 2, 4, 5, 6)                         # the classes that make a row: lost, both, threat, conditional, contested
PRIORITY = (0, 2, 4, 5, 6)                               # the worst class first
HOW = ["a taking", "the unit moves", "a guard is added", "a counter threat", "a unit is put between"]
BASIS = TS.BASIS_WORDS                                   # forced, pruned, estimate, bound
ENDED = ("concrete", "quiet", "lost beyond doubt", "out of budget")
BAND_WORDS = ("high, no response", "high, with a response", "low, no response", "low, with a response")


class ThreatReader:
    """One per process: the frame read's model, built as 78.65 and 78.67 built it (the same seeds, fuses and order of kinds), and `read`."""

    def __init__(self, threads: int = 1, answer_fuse: int = TS.FUSE):
        rec = TS.human_kind_record()
        sample = TS.load_sample(Path(paths.experiment(*TS.HEADS_DIR)), TS.PER_SHARD, 0)
        self.model = TS.make_model(sample, rec, 0)
        picked = OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), TS.SEEDS)
        self.keys = [k for _dc, k, _d, _z in picked]
        self.goal = G2M.goal_kind_of(HD.VARIANT)
        self.threads, self.answer_fuse = threads, answer_fuse
        self.cost = {"wall": 0.0, "rows": 0}

    def read(self, games: list, plies: list) -> dict:
        t0 = time.perf_counter()
        res = self.model.questions_threats([(s, list(m)) for s, m in games], plies, self.keys, self.goal, TS.FUSE, TS.ROW_FUSE, OLD.MAX_SEED_SIZE, True, self.answer_fuse, self.threads)
        self.cost["wall"] += time.perf_counter() - t0
        self.cost["rows"] += len(res["rows"])
        return res


# ------------------------------------------------------------------------------------------------------------------------------ the rows

def cell_key(cell: tuple, mover: int) -> tuple:
    """A member as (the absolute side, kind, place) from (the mover's own flag, kind, place)."""
    return (mover if cell[0] == 1 else 1 - mover, cell[1], cell[2])


def read_set_of(frame: dict, mover: int) -> tuple:
    """THE READ SET of the claims made in a frame: its members (absolute side, kind, place), its places, and the side to move; what the read looked at, to the last unit."""
    return (mover, tuple(sorted(cell_key(tuple(m), mover) for m in frame["members"])), tuple(sorted(frame["places"])))


def claims_of(raw: dict) -> list:
    """The raw row's claims, flat: one dict per (frame, unit)."""
    mover, out = raw["mover"], []
    for fi, f in enumerate(raw["frames"]):
        rs = read_set_of(f, mover)
        for c in f["claims"]:
            owner_flag, kind, place = f["members"][c["member"]]
            out.append({"frame": fi, "frame_id": f["id"], "owner": mover if owner_flag == 1 else 1 - mover, "own": owner_flag, "kind": kind, "place": place, "class": c["class"], "basis": c["basis"],
                        "take_basis": c["take_basis"], "bearers": [tuple(b) for b in c["bearers"]], "answers": [tuple(a) for a in c["answers"]], "tried": c["tried"], "unread": c["unread"],
                        "n_members": len(f["members"]), "members": rs[1], "read_set": rs, "frame_unread": f["unread"]})
    return out


def ended_of(claim: dict) -> str:
    """How the line of the claim ended (spec 4), from what the read reports (the mapping is this build's choice, see the module doc)."""
    if claim["frame_unread"] > 0 or claim["unread"] > 0:
        return ENDED[3]
    b = claim["basis"] if claim["basis"] != 255 else claim["take_basis"]
    if b in (0, 1):
        return ENDED[0]
    return ENDED[1]


def basis_word(claim: dict) -> str:
    b = claim["basis"] if claim["basis"] != 255 else claim["take_basis"]
    return BASIS[b] if b < len(BASIS) else "nothing"


def threat_rows(raw: dict, names: dict, stake_of=None) -> dict:
    """The threat list of a raw row (see the module doc): {"mover": rows against the mover's units, "other": rows against the other side's}."""
    cl = claims_of(raw)
    kinds = names["kinds"]
    by_unit: dict = {}
    for c in cl:
        by_unit.setdefault((c["owner"], c["kind"], c["place"]), []).append(c)
    out = {"mover": [], "other": []}
    mover = raw["mover"]
    for (owner, kind, place), cs in by_unit.items():
        wins = [c for c in cs if c["class"] in THREAT_CLASSES]
        if not wins:
            continue
        worst = min(c["class"] for c in wins)
        win = sorted((c for c in wins if c["class"] == worst), key=lambda c: (c["n_members"], c["frame_id"]))[0]          # the smallest frame that explains wins
        wider = [c for c in cs if c is not win and set(c["members"]) > set(win["members"])]
        scope = [win] + sorted(wider, key=lambda c: (c["n_members"], c["frame_id"]))
        resp_frame, resp = None, []
        for c in scope:
            if c["answers"] and c["class"] != 0:
                resp_frame, resp = c, c["answers"]
                break
        # the class word of the row is the worst class in the smallest frame; a unit lost in the smallest frame and answered in a wider one is one row with a response
        names_of = names["places"]
        row = {"side": "mover" if owner == mover else "other", "owner": owner, "kind": kinds[kind], "place": names_of[place], "place_ix": place, "class": CLASS[worst],
               "by": [(kinds[k], names_of[p]) for k, p in win["bearers"]],
               "frames": [{"id": c["frame_id"], "n_members": c["n_members"], "class": CLASS[c["class"]], "answers": len(c["answers"]), "basis": basis_word(c)} for c in scope],
               "win_frame": {"id": win["frame_id"], "n_members": win["n_members"], "members": [(kinds[k], names_of[p], o) for o, k, p in win["members"]]},
               "response": [((names_of[a[0]], names_of[a[1]]), HOW[a[2]], bool(a[3])) for a in resp], "response_frame": (resp_frame["frame_id"] if resp_frame else None), "response_ix": [(a[0], a[1], a[2], a[3]) for a in resp], "resp_tried": (resp_frame["tried"] if resp_frame else 0),
               "widened": bool(resp_frame is not None and resp_frame is not win), "ended": ended_of(win), "basis": basis_word(win), "read_set": win["read_set"], "stake": None,
               "answers_tried": win["tried"], "answers_unread": win["unread"]}
        if stake_of is not None:
            row["stake"] = stake_of(kinds[kind], owner == mover)
        row["_key"] = (owner, kind, place)
        out[row["side"]].append(row)
    for side in out:
        out[side].sort(key=lambda r: (-(r["stake"] if r["stake"] is not None else -9.0), CLASS.index(r["class"]), r["place_ix"]))
    return out


def rank_high_of(rows_both: list) -> dict:
    """The stake ranked within the position (rows of both sides pooled): a row is high when its stake is in the top half of the position's rows that have a stake (the middle row of an odd count is high),
    and the stake is above nothing (a row whose loss would raise the owner's chance is never high). Returns {id(row): bool}."""
    have = sorted((r for r in rows_both if r["stake"] is not None), key=lambda r: -r["stake"])
    k = (len(have) + 1) // 2
    return {id(r): (i < k and r["stake"] > 0) for i, r in enumerate(have)}


def band_of(row: dict, spread: "float | None", rank_high: "bool | None", way: str) -> "int | None":
    """The band 1 to 4 by `way` ('spread': the stake above the spread of the two readings at the position; 'rank': high by the position's own ranking). None when the row has no stake."""
    if row["stake"] is None:
        return None
    if way == "spread":
        if spread is None:
            return None
        high = row["stake"] > spread
    else:
        if rank_high is None:
            return None
        high = rank_high
    resp = bool(row["response"])
    return (2 if resp else 1) if high else (4 if resp else 3)


def threat_list(moves: list, ply: int, reader: ThreatReader, start: "str | None" = None, stake_of=None, spread: "float | None" = None) -> dict:
    """The list of the position after the first `ply` moves of the game (start text or None): both sides, each row with its band by the spread (if given) and by the rank."""
    res = reader.read([(start, moves)], [[ply]])
    if not res["rows"]:
        return {"mover": [], "other": [], "raw": None}
    raw = res["rows"][0]
    lst = threat_rows(raw, res, stake_of)
    both = lst["mover"] + lst["other"]
    rk = rank_high_of(both)
    for r in both:
        r["band_spread"] = band_of(r, spread, None, "spread")
        r["band_rank"] = band_of(r, None, rk.get(id(r)), "rank")
    lst["raw"] = raw
    return lst
