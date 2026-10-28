"""plan 71.0 THE GUARD: a refactor stage is right only if the engine's TRACES are identical before and after it on a fixed case set at the COUNT budget
(deterministic: no clock). Usage:

  .venv/bin/python scripts/trace_identity.py snapshot <dir>          # write the traces (and the read sets) of the case set into <dir>
  .venv/bin/python scripts/trace_identity.py snapshot <dir> --sets wide   # plan 72.7 step 3: the WIDER case set instead (WIDE_SETS, every puzzle not in the ten)
  .venv/bin/python scripts/trace_identity.py snapshot <dir> --tier 3      # plan 72.7 step 3: the root at widening tier 3 (count budget): the wider tiers' words asked
  .venv/bin/python scripts/trace_identity.py compare <before> <after>  # diff two snapshots; prints OBS lines; exit 1 on any difference
  .venv/bin/python scripts/trace_identity.py crosscheck               # plan 72.7 step 0: SAN and legal-move order, python-chess against the Rust Pos

The case set covers the plans' acceptance positions (eSJqH, 2TUzC, 07Z0P, 4pC83, 0LRTF, W0VFO, qCXKp, j3ybl, RoR8t, 02XVK) and the two fixed positions
(the discovery position, pos2). Lines carrying wall-clock seconds are masked before the diff (the count budget decides, the seconds vary).

plan 72.7 step 0 (docs/RUST_BOUNDARY.md section 4, risk 2): beside each trace `<id>.txt` the snapshot writes `<id>.reads.txt`, THE READ SET OF EVERY CLAIM in a stable
sorted form -- the root agenda's questions (what each read, and the reads of the claim that verified it) and every fact of the run's store (the claims, kills, gaps and
root lists of every inner agenda and of their agendas, with their reads, their occupancy reads and the read set kept in the payload). The trace prints none of them,
so a claim that silently reads less (a word cache that does not replay its read set) moves no trace line; `compare` diffs these files too and counts them apart.
plan 72.7 step 2b: and `<id>.leaf.txt`, THE LEAF'S VERDICTS -- every static leaf read of the run as it is computed (route, schema, value, kept / refuted by what,
the lines it added): no trace line of the ten cases prints a leaf, so a wrong leaf moved nothing before (two wrong leaf builds proved it); `compare` counts them apart.
plan 72.7 step 3 (docs/RUST_BOUNDARY.md section 4: "the guard runs on a wider case set, 30 to 50 positions beyond the ten"): `snapshot <dir> --sets wide` writes the same three
files for EVERY puzzle of WIDE_SETS (dev_20 and rand14_10 .. rand21_10) that is not one of the ten -- 99 positions: the whole sets are taken rather than a hand-picked
part of them. `--tier K` runs the root at the widening tier K (count budget, no clock): at tier 0 the wider tiers' words (the mate in two, the delayed threat, no safe
move) are never asked, so a port of them is invisible there. Every snapshot runs at PYTHONHASHSEED=0 (the script re-executes itself): the traces of 2 of the 99 depend
on the string-hash order (agenda.py iterates a set of goal tuples, `ctx.route_goals`).
plan 73.2 (the lemma store): and `<id>.words.txt`, THE WORDS OF THE RUN -- every claim lemma, kill reason and GAP reason the root set and every lemma field of the run's
fact store (a claim's lemma, a kill's reason), distinct and sorted (`lemma_store.harvest`); `compare` counts them apart; `lemma_store validate <dir>` reads them beside the traces.
plan 74 (a) (the reasoning graph): and `<id>.why.txt`, THE PROVENANCE LIST -- the trace lines of the reasoning graph (plan 74.5: every line whose text, after the tier's
"[tier k] " label, begins with the graph's prefix "why: ") are MASKED out of `<id>.txt` and written here, followed by the run's whole graph (`reasoning.Graph.dump`: every
question with why it was asked, every line, answer and observation); `compare` counts them apart, so a restructure that keeps the traces may still be caught moving a
question's provenance.
`crosscheck` (risks 4 and 5): over the case set's positions, a sample of the puzzle sets, a seeded random walk from each and a list of hand-made corners (castling,
promotions, en passant, disambiguation, mates, evasions), `Pos.legal_ucis()` must equal `[m.uci() for m in board.legal_moves]` and `Pos.san(uci)` must equal
`board.san(move)` for every legal move; every corner that differs is printed as a FINDING.
"""
from __future__ import annotations

import difflib
import os
import random
import re
import sys

import chess

import analyst_core as AC
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import reasoning as RN
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State

CASES = [("rand4_10", "eSJqH"), ("rand8_10", "2TUzC"), ("dev_20", "07Z0P"), ("rand7_10", "4pC83"), ("rand9_10", "j3ybl"), ("rand3_10", "qCXKp"),
         ("rand8_10", "W0VFO"), ("rand23_5", "RoR8t")]
FENS = {"discovery": ("r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1", "w"),
        "pos2": ("r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1", "w")}
_SECS = re.compile(r"\b\d+\.\d+ s\b|\(\d+\.\d+ s\)|seconds \d+\.\d+|\d+\.\d+ s needed|\d+\.\d+ s for it|\d+\.\d+ s left|\d+\.\d+ms")
READS = ".reads.txt"
WHY = ".why.txt"                                       # plan 74 (a): the provenance list (the masked "why: " lines, then the graph's dump)
_WHY = re.compile(r"^\s*(\[[^\]]*\] )*" + re.escape(RN.PREFIX))

# the cross-check's sample: the puzzle sets of the plans (the guard's own sets, the rest of the rand sets, the dev set, an equality set and an endgame set) ...
SAMPLE_SETS = ("dev_20",) + tuple(f"rand{i}_10" for i in range(1, 22)) + ("rand22_5", "rand23_5", "rand61_40_equality", "rand62_30_rookEndgame_pawnEndgame_any")
WALK, SEED = 4, 727                                   # ... each followed by a seeded random walk of WALK plies (every position on it is checked)
# ... and the corners the SAN and the move order can differ in (each a legal position; checked as given and walked like the rest)
CORNERS = [
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",          # castling on both sides, rooks and kings with many moves (python-chess lists castling after the pieces)
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "5k2/8/8/8/8/8/8/4K2R w K - 0 1",                # O-O+ (the rook gives check from f1)
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",              # promotions with and without capture, Q R B N
    "3k4/1P6/8/8/8/8/8/4K3 w - - 0 1",               # b8=Q+ / b8=R+ against b8=B / b8=N
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",             # en passant exd6
    "8/8/8/K2pP2r/8/8/8/7k w - d6 0 2",              # en passant pinned along the rank: illegal
    "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 2",             # en passant cxd6
    "4k3/8/8/8/1N6/8/1N6/4K3 w - - 0 1",             # disambiguation by rank (N4d3 / N2d3)
    "4k3/8/8/8/8/8/1N3N2/4K3 w - - 0 1",             # disambiguation by file (Nbd3 / Nfd3)
    "5k2/8/8/8/Q6Q/8/8/Q3K3 w - - 0 1",              # by file and rank (Qa4d4)
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",             # Ra8#
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",                # stalemate: no legal move
    "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",               # in check: the king's moves first (Kxe2 among them)
    "4k3/8/8/8/8/8/3PP3/r3K2R w K - 0 1",            # in check on the back rank, castling rights but no castling
    "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",             # in check by the pawn that just moved: exd3 e.p. evades
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",       # Qxf7# (the scholar's mate)
]


def _sq(s) -> str:
    """A read set in a stable sorted form: the square names in square order ('None': not recorded; '{}': empty). A member that is not a square prints as itself."""
    if s is None:
        return "None"
    if not s:
        return "{}"
    return " ".join(chess.square_name(x) if isinstance(x, int) and 0 <= x < 64 else repr(x) for x in sorted(s, key=lambda x: (not isinstance(x, int), str(x) if not isinstance(x, int) else x)))


def _reads(ag) -> str:
    """THE READ SET OF EVERY CLAIM of one run (plan 72.7 step 0): the root agenda's questions in id order (their reads, the reads of the claim that verified them), then
    every fact of the run's store in id order (kind, side, position, reads, occupancy reads, the payload's read set)."""
    out = []
    for q in sorted(ag.all, key=lambda q: q.id):
        out.append(f"Q{q.id} {q.goal} {q.status} reads {_sq(q.reads)} | claimed {_sq(q.claimed[2]) if q.claimed is not None else '-'}")
    facts = ag.ctx.facts.facts if ag.ctx is not None else {}
    for f in sorted(facts.values(), key=lambda f: f.id):
        pr = f.payload.get("reads", "-") if isinstance(f.payload, dict) else "-"
        out.append(f"#{f.id} {f.kind} {'w' if f.turn else 'b'} {f.pk} reads {_sq(f.reads)} | occ {_sq(f.occ)} | payload {_sq(pr) if not isinstance(pr, str) else pr}")
    return "\n".join(out) + "\n"


LEAF = ".leaf.txt"


def _leaf_hook(out: list):
    """plan 72.7 step 2b: THE LEAF'S VERDICTS. Every static leaf read (`Agenda._leaf_read`, every agenda of the run, in call order) recorded as it is COMPUTED (a memo hit
    is not): the agenda's position and side, the route, the bound schema and value, the verdict (None or the refuting claim, its line, w, our gain there) and the
    lines and OBS lines the read added. The trace prints a leaf only when it downgrades or is not trusted (none of the ten cases does), so without this a wrong
    leaf moves nothing. -> the original method (to restore)."""
    orig = Agenda._leaf_read

    def wrapped(self, bind, r):
        fresh = r.move not in self._leaf
        n_lines, n_obs = len(self.lines), len(self.leaf_obs)
        got = orig(self, bind, r)
        if fresh:
            v = "kept" if got is None else f"refuted by {got['claim'].what} {got['claim'].value} ({got['claim'].lemma}) after {got['line']}, w {got['w']}, gain {got['gain']}"
            out.append(f"{self.pk} {self.us} {r.san} {bind.schema} {bind.value} -> {v} | lines {self.lines[n_lines:]} | obs {self.leaf_obs[n_obs:]}")
        return got
    Agenda._leaf_read = wrapped
    return orig


def _run(fen: str, us: str, leaf: "list | None" = None, tier: int = 0, words: "list | None" = None, why: "list | None" = None) -> tuple:
    """-> (the trace with the DECISION line, the read sets); `leaf` (a list) receives the leaf's verdicts (`_leaf_hook`). `tier` (plan 72.7 step 3): the root's widening
    tier at the count budget (`Agenda(tier=k)`: the context's tier -- 2 opens the mate-in-two threats, 1 the delayed threat and no safe move, 3 the chain items -- and a
    count fuse of 2^k times tier 0's; no clock, so deterministic like tier 0). At tier 0 (the default) the words of the wider tiers are never asked. `words` (a list;
    plan 73.2) receives the run's words, (slot, string), from the agenda and its fact store (`lemma_store.harvest`). `why` (a list; plan 74 (a)) receives the dump of the
    run's reasoning graph."""
    orig = _leaf_hook(leaf) if leaf is not None else None
    try:
        ag = Agenda(State(fen=fen, us=us, pieces=[]), tier=tier)
        ag.ask()
    finally:
        if orig is not None:
            Agenda._leaf_read = orig
    d = ag.decision()
    trace = ag.trace() + f"\nDECISION {d.what if d else None} {d.value if d else None} {d.lemma if d else None} proved={d.proved if d else None}\n"
    if words is not None:
        words += LS.harvest(ag)
    if why is not None and ag.ctx is not None:
        why += ag.ctx.graph.dump()
    return trace, _reads(ag)


def mask_why(trace: str) -> tuple:
    """plan 74 (a): -> (the trace without the reasoning graph's lines, those lines in order)."""
    lines = trace.split("\n")
    return "\n".join(x for x in lines if not _WHY.match(x)), [x for x in lines if _WHY.match(x)]


def _trace(fen: str, us: str) -> str:
    return _run(fen, us)[0]


WIDE_SETS = ("dev_20",) + tuple(f"rand{i}_10" for i in range(14, 22))      # plan 72.7 step 3: the wider case set's puzzle sets (every puzzle of them not in CASES)


def _wide_cases():
    """plan 72.7 step 3: every puzzle of WIDE_SETS that is not one of the ten, in set order then file order (a set missing on disk is skipped)."""
    ten = {pid for _s, pid in CASES}
    for set_name in WIDE_SETS:
        try:
            ps = PA.load_set(set_name)
        except FileNotFoundError:
            continue
        for p in ps:
            if p["id"] not in ten:
                yield p["id"], p["fen"], "w" if chess.Board(p["fen"]).turn else "b"


def _cases(sets: str = "ten"):
    if sets == "wide":
        yield from _wide_cases()
        return
    for set_name, pid in CASES:
        try:
            p = next(x for x in PA.load_set(set_name) if x["id"] == pid)
        except (StopIteration, FileNotFoundError):
            continue
        yield pid, p["fen"], "w" if chess.Board(p["fen"]).turn else "b"
    for name, (fen, us) in FENS.items():
        yield name, fen, us


def snapshot(out: str, sets: str = "ten", tier: int = 0) -> None:
    os.makedirs(out, exist_ok=True)
    n = 0
    for pid, fen, us in _cases(sets):
        leaf: list = []
        words: list = []
        graph: list = []
        trace, reads = _run(fen, us, leaf, tier, words, graph)
        trace, whys = mask_why(trace)
        open(os.path.join(out, pid + ".txt"), "w").write(_SECS.sub("<t>", trace))
        open(os.path.join(out, pid + WHY), "w").write(_SECS.sub("<t>", "\n".join(whys + graph)) + "\n")
        open(os.path.join(out, pid + READS), "w").write(reads)
        open(os.path.join(out, pid + LEAF), "w").write("\n".join(leaf) + "\n")
        open(os.path.join(out, pid + LS.SNAPSHOT_WORDS), "w").write("".join(f"{slot}\t{s}\n" for slot, s in sorted(set(words))))
        n += 1
    print(f"OBS trace identity: {n} traces, {n} read sets, {n} leaf verdict lists, {n} word lists and {n} provenance lists written to {out}")
    from catspace.incubator.analyst.discovery import graph as G
    from catspace.incubator.analyst.discovery import rboard as RB
    if G.BRIDGE_CHECK:                                   # plan 72.7 step 1: CATSPACE_BRIDGE_CHECK=1 -- every node's handle was compared with its board (an assertion stops the run on a split)
        print(f"OBS bridge: {G.BRIDGE_CHECKS[0]} handle / board pairs compared, none split; "           # plan 72.7 step 2a: and every RBoard read with its python-chess shadow
              f"{RB.BRIDGE_READS[0]} RBoard reads compared with python-chess, none split")


def compare(before: str, after: str) -> int:
    bad = {"trace": 0, "reads": 0, "leaf": 0, "words": 0, "why": 0}
    seen = {"trace": 0, "reads": 0, "leaf": 0, "words": 0, "why": 0}
    for f in sorted(os.listdir(before)):
        kind = "reads" if f.endswith(READS) else "leaf" if f.endswith(LEAF) else "words" if f.endswith(LS.SNAPSHOT_WORDS) else "why" if f.endswith(WHY) else "trace"
        seen[kind] += 1
        a = os.path.join(after, f)
        if not os.path.exists(a):
            print(f"OBS trace identity {f}: missing after"); bad[kind] += 1; continue
        x, y = open(os.path.join(before, f)).read().splitlines(), open(a).read().splitlines()
        if x != y:
            bad[kind] += 1
            diff = list(difflib.unified_diff(x, y, lineterm="", n=0))
            print(f"OBS trace identity {f}: DIFFERENT ({sum(1 for l in diff if l.startswith(('+', '-')) and not l.startswith(('+++', '---')))} lines)")
            for l in diff[2:14]:
                print("   " + l[:200])
    print(f"OBS trace identity: {bad['trace']} of {seen['trace']} traces differ")
    if seen["reads"]:
        print(f"OBS trace identity: {bad['reads']} of {seen['reads']} read sets differ")
    if seen["leaf"]:
        print(f"OBS trace identity: {bad['leaf']} of {seen['leaf']} leaf verdict lists differ")
    if seen["words"]:
        print(f"OBS trace identity: {bad['words']} of {seen['words']} word lists differ")
    if seen["why"]:
        print(f"OBS trace identity: {bad['why']} of {seen['why']} provenance lists differ")
    return 1 if bad["trace"] or bad["reads"] or bad["leaf"] or bad["words"] or bad["why"] else 0


# ---------------------------------------------------------------------------------------------------------------- plan 72.7 step 0: the cross-checks (risks 4, 5)

def crosscheck_positions(sets=SAMPLE_SETS, walk: int = WALK, seed: int = SEED, cases_only: bool = False) -> list:
    """The FENs the cross-check reads: the case set's positions, the corners, the puzzle sets' positions (unless `cases_only`), each followed by a seeded random walk of
    `walk` plies; every position on a walk is one more FEN. Duplicates dropped, the order kept."""
    roots = [fen for _pid, fen, _us in _cases()] + list(CORNERS)
    if not cases_only:
        for s in sets:
            try:
                roots += [p["fen"] for p in PA.load_set(s)]
            except FileNotFoundError:
                continue
    rng = random.Random(seed)
    out, seen = [], set()
    for f in roots:
        b = chess.Board(f)
        if not b.is_valid():
            continue
        for k in range(walk + 1):
            x = b.fen()
            if x not in seen:
                seen.add(x)
                out.append(x)
            ms = list(b.legal_moves)
            if k == walk or not ms:
                break
            b.push(rng.choice(ms))
    return out


def crosscheck(fens: list) -> dict:
    """-> {positions, moves, san (mismatches: [(fen, uci, python's, Rust's)]), order (mismatches: [(fen, python's, Rust's)]), unparsable}: python-chess against `AC.Pos`."""
    out = dict(positions=0, moves=0, san=[], order=[], unparsable=[])
    for fen in fens:
        b = chess.Board(fen)
        try:
            p = AC.Pos(b.fen())
        except ValueError:
            out["unparsable"].append(fen)
            continue
        out["positions"] += 1
        py = [m.uci() for m in b.legal_moves]
        rs = p.legal_ucis()
        if py != rs:
            out["order"].append((fen, py, rs))
        for m in b.legal_moves:
            out["moves"] += 1
            u = m.uci()
            s_py, s_rs = b.san(m), p.san(u)
            if s_py != s_rs:
                out["san"].append((fen, u, s_py, s_rs))
    return out


def crosscheck_main() -> int:
    fens = crosscheck_positions()
    r = crosscheck(fens)
    print(f"OBS crosscheck: positions checked {r['positions']} (sets {len(SAMPLE_SETS)}, corners {len(CORNERS)}, random walk {WALK} plies, seed {SEED}), "
          f"moves checked {r['moves']}, SAN mismatches {len(r['san'])}, order mismatches {len(r['order'])}, unparsable by shakmaty {len(r['unparsable'])}")
    for fen, u, a, b in r["san"][:40]:
        print(f"FINDING san {fen} {u}: python-chess {a}, Rust {b}")
    for fen, a, b in r["order"][:20]:
        i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"FINDING order {fen}: first difference at {i}: python-chess {a[i:i + 4]}, Rust {b[i:i + 4]} (lengths {len(a)}, {len(b)})")
    for fen in r["unparsable"][:10]:
        print(f"FINDING unparsable {fen}")
    return 1 if r["san"] or r["order"] else 0


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":           # plan 72.7 step 3: the traces of 2 of the 99 wide cases (aEC9x, fth9B) depend on Python's string-hash seed (two runs of the
        os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))       # same code differ unpinned, never pinned): every snapshot runs at seed 0
    if sys.argv[1] == "snapshot":
        snapshot(sys.argv[2], sys.argv[sys.argv.index("--sets") + 1] if "--sets" in sys.argv else "ten",
                 int(sys.argv[sys.argv.index("--tier") + 1]) if "--tier" in sys.argv else 0)
    elif sys.argv[1] == "crosscheck":
        sys.exit(crosscheck_main())
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
