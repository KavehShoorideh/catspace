"""THE REMEMBERED STATE (ANALYST_PLAN 50): every solve is stored as RELATIONS (pieces by kind/side/attacked/guarded, who did what by kind, the biggest
threat, the shape after), retrieved by relations (Jaccard over tokens, never by where the pieces stand), and the lemma that decided is ASKED FIRST next
time. The memory is a prior for the order of asking; the Lean proof of the schema still decides (never a value, never a proof).

JSON lines at paths.derived("reasoned_states.jsonl"); `store` appends one record per item (the per-item checkpoint) and skips ids already stored.

Deviation from 50.2 (the query has no roles yet: the decision is not made when we recall): similarity compares only the token kinds the QUERY carries
(pieces `p:`, threat `t:`; roles `r:` when the query has them), so a recall before the decision is not diluted by the roles of the stored record.
"""
from __future__ import annotations

import json
import os
import re

import chess

from catspace.io import paths

STORE = "reasoned_states.jsonl"
KINDS = "KQRBNP"
_ROW = re.compile(r"(our|their) (\S) on \w\w worth \d+: attacked (\d+)x, guarded (\d+)x -- (.*)")
BASE = {"mate_in_one": "mate", "mate_in_two": "mate2"}


def _path(path):
    return path or paths.derived(STORE)


def family_base(name: str) -> str:
    """'delayed_root_d1' -> 'delayed_root'; 'mate_in_two' stays (the stored schema name)."""
    return re.sub(r"_d\d$", "", name or "")


def ladder_name(family: str) -> str:
    """The stored schema name -> the selfmine family name the bind stage asks."""
    return BASE.get(family_base(family), family_base(family))


# ------------------------------------------------------------------------------------------------------------------------------ tokens

def piece_tokens(fen: str, us: str) -> list:
    """The level-one state's rows (puzzle_apply.state_lines) as tokens WITHOUT squares."""
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    incheck = chess.Board(fen).is_check()
    out = []
    for line in PA.state_lines(fen, us)[1:]:
        m = _ROW.search(line)
        if not m:
            continue
        side, sym, hit, guard, verdict = m.groups()
        t = f"{side} {sym.upper()} attacked{hit} guarded{guard}"
        if "win" in verdict:
            t += " hanging"
        if incheck and side == "our" and sym.upper() == "K":
            t += " check"
        out.append(t)
    return sorted(set(out))


def threat_token(fen: str, us: str, items=None, check: bool = False) -> str:
    """The threat stage's biggest threat to us: `threat=<kind of ours attacked> value=<w>` or `threat=none`. `items` = the dead-end verdict items when the caller has them."""
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    from catspace.incubator.analyst.discovery import selfmine as SM
    b = chess.Board(fen)
    if b.is_check() or check:
        return "threat=K value=100"
    if items is None:
        items = SM.verdicts(PA.passed(fen), us, log=lambda *a: None)["items"]
    costs = [it for it in items if it["kind"] == "costs"]
    costs.sort(key=lambda it: -SM.PIECE_WORTH[b.piece_at(chess.parse_square(it["t"])).symbol().lower()])
    if not costs:
        return "threat=none"
    pc = b.piece_at(chess.parse_square(costs[0]["t"]))
    return f"threat={pc.symbol().upper()} value={SM.PIECE_WORTH[pc.symbol().lower()]}"


def move_kind(fen: str, x: str, y: str) -> str:
    b = chess.Board(fen)
    m = chess.Move.from_uci(x + y)
    kinds = []
    if b.gives_check(m):
        kinds.append("check")
    if b.is_capture(m):
        kinds.append("capture")
    return "+".join(kinds) or "quiet"


def roles_of(fen: str, x: str, y: str) -> list:
    """Who did what, by kind not square: mover, victim (occupant of the target), move kind(s), recapturer (the biggest piece of theirs attacking the target)."""
    from catspace.incubator.analyst.discovery import selfmine as SM
    b = chess.Board(fen)
    mover = b.piece_at(chess.parse_square(x)).symbol().upper()
    occ = b.piece_at(chess.parse_square(y))
    victim = occ.symbol().upper() if occ else "none"
    ys = chess.parse_square(y)
    defs = [b.piece_at(s) for s in b.attackers(not b.turn, ys)]
    big = max(defs, key=lambda p: (SM.PIECE_WORTH[p.symbol().lower()], p.piece_type == chess.KING), default=None)
    rec = big.symbol().upper() if big else "none"
    return [f"mover={mover}", f"victim={victim}"] + [f"move={k}" for k in move_kind(fen, x, y).split("+")] + [f"recapturer={rec}"]


def signature(pieces=(), roles=(), threat=None) -> set:
    s = {"p:" + t for t in pieces} | {"r:" + t for t in roles}
    if threat:
        s.add("t:" + threat)
    return s


def sig_of_record(rec: dict) -> set:
    return signature(rec["pieces"], rec["roles"], rec["threat"])


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def similarity(query: set, stored: set) -> float:
    """Jaccard over the token kinds the query carries (see the module note)."""
    kinds = {t.split(":", 1)[0] for t in query}
    return jaccard(query, {t for t in stored if t.split(":", 1)[0] in kinds})


# ------------------------------------------------------------------------------------------------------------------------------ the record

def make_record(p: dict, set_name: str, fen: str, us: str, decision_uci: str, family: str, gain, correct: bool, threat: str,
                instance: str | None = None, source: str = "puzzle", goal: str | None = None, fixes=(), lemma: str | None = None) -> dict:
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    x, y = decision_uci[:2], decision_uci[2:4]
    b = chess.Board(fen)
    sh = PA.shape(fen, [(x, y)], us, threats=False)
    return dict(id=p["id"], set=set_name, source=source, fen=fen, us=us,
                decision=dict(san=b.san(chess.Move.from_uci(decision_uci)), uci=decision_uci),
                family=family, gain=gain, correct=bool(correct), instance=instance, goal=goal, fixes=list(fixes), lemma=lemma,
                pieces=piece_tokens(fen, us), roles=roles_of(fen, x, y), threat=threat,
                after=dict(hanging=len(sh["hanging"]), guarded=sh["guarded"], poking=sh["poking"], checks=len(sh["checks"])))


# ------------------------------------------------------------------------------------------------------------------------------ the store

def load(path=None) -> list:
    p = _path(path)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                       # a torn last line from an interrupted write: skipped, the item is redone
    return out


def stored_ids(path=None) -> set:
    return {r["id"] for r in load(path)}


def store(record: dict, path=None) -> bool:
    """Append one record (flushed and fsynced: the per-item checkpoint). False when the id is already stored."""
    if record["id"] in stored_ids(path):
        return False
    with open(_path(path), "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return True


def recall(sig: set, k: int = 3, exclude_id: str | None = None, path=None) -> list:
    """The k most similar records of OTHER puzzles: [(sim, record)], best first."""
    got = [(similarity(sig, sig_of_record(r)), r) for r in load(path) if r["id"] != exclude_id]
    got.sort(key=lambda t: -t[0])
    return [(round(s, 3), r) for s, r in got[:k]]


def propose(fen: str, us: str, k: int = 3, exclude_id: str | None = None, threat: str | None = None, path=None) -> list:
    """-> [(family, mover_kind, victim_kind, move_kind, sim, id)] from the k most similar records: the lemma to ask first and on which candidates."""
    if threat is None:
        threat = threat_token(fen, us)
    out = []
    for sim, r in recall(signature(piece_tokens(fen, us), (), threat), k, exclude_id, path):
        kv = {t.split("=")[0]: t.split("=")[1] for t in r["roles"] if not t.startswith("move=")}
        mk = "+".join(t.split("=")[1] for t in r["roles"] if t.startswith("move="))
        out.append((family_base(r["family"]), kv.get("mover"), kv.get("victim"), mk, sim, r["id"]))
    return out


def order_candidates(fen: str, candidates: list, mover_kind: str | None, mk: str | None) -> list:
    """The candidates matching the recalled mover kind and move kind first (stable), then the rest."""
    b = chess.Board(fen)

    def key(c):
        (x, y), = c
        return (0 if b.piece_at(chess.parse_square(x)).symbol().upper() == mover_kind else 1,
                0 if move_kind(fen, x, y) == mk else 1)
    return sorted(candidates, key=key)


def recall_line(sim: float, r: dict, fen: str) -> str:
    kv = dict(t.split("=", 1) for t in r["roles"])
    mk = [t.split("=")[1] for t in r["roles"] if t.startswith("move=")]
    what = " and ".join(("gives check" if m == "check" else "takes" if m == "capture" else "is quiet") for m in mk)
    names = dict(K="king", Q="queen", R="rook", B="bishop", N="knight", P="pawn")
    return (f"seen {r['id']} (sim {sim:.2f}): {kv['mover']} {what}, {family_base(r['family'])} decided "
            f"-- ask {ladder_name(r['family'])} on {names.get(kv['mover'], kv['mover'])} {'/'.join(mk)} moves first")
