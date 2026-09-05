"""motifs.py -- IS THE VOCABULARY COMPLETE? (Kaveh 2026-09-04: "have we exhausted all two-piece,
three-piece ideas, local sequences? how do we know?")
Every L3 relation is a pattern over the PRIMITIVE graph (pieces; edges attack / defend / through).
For a fixed arity the set of connected pattern TYPES is finite, so we can enumerate them, count
them in real positions, and check which have a name. Coverage = share of occurrences that a
named relation accounts for. Local SEQUENCES are enumerated the same way over the effect
vocabulary (what each ply captures / checks / creates / uses / resolves) along puzzle solutions,
and cross-tabulated with the Lichess themes we do and do not name.
    python -m catspace.incubator.analyst.motifs [--n 20000 --seq-n 8000 --k 3]"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import time
from collections import Counter, defaultdict

import chess
import zstandard

from catspace.io import paths
from .relations import relations
from .dynamics import line_effects

CLASS = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "N", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}   # minors share a class
NAMED_STATIC = ("pin", "skewer", "xray", "discovered_attack", "battery", "fork", "overloaded", "hanging_through_pin",
                "disabled_attack", "disabled_defense", "pinned_and_attacked", "hanging", "attacked_by_lower_value", "attacks")
NAMED_SEQ_EVENTS = ("fork", "pin", "skewer", "discovered_attack", "battery", "overloaded", "hanging", "attacked_by_lower_value",
                    "pinned_and_attacked", "hanging_through_pin", "mate")
THEMES_WE_NAME = {"fork", "pin", "skewer", "discoveredAttack", "hangingPiece", "xRayAttack", "zugzwang", "backRankMate",
                  "smotheredMate", "doubleCheck", "mateIn1", "mateIn2", "mateIn3"}
THEMES_UNNAMED = {"deflection", "attraction", "interference", "clearance", "sacrifice", "quietMove", "zwischenzug",
                  "trappedPiece", "capturingDefender", "exposedKing", "advancedPawn", "promotion", "enPassant",
                  "defensiveMove", "intermezzo", "kingsideAttack", "queensideAttack", "attackingF2F7", "desperado"}


# ---------------- primitive graph ----------------
def primitive_graph(board: chess.Board):
    """nodes: square -> label (side-relative colour is fixed at canonicalisation); edges: (a, b, type)."""
    nodes = {s: p for s, p in board.piece_map().items()}
    edges = []
    for s, p in nodes.items():
        for t in board.attacks(s):
            q = board.piece_at(t)
            if q is None:
                continue
            edges.append((s, t, "atk" if q.color != p.color else "def"))
        # through: what s would attack if the first piece on the line were removed
        for t in list(board.attacks(s)):
            if board.piece_at(t) is None:
                continue
            b2 = board.copy(stack=False); b2.remove_piece_at(t)
            for u in b2.attacks(s) - board.attacks(s):
                if board.piece_at(u) is not None:
                    edges.append((s, u, f"thru:{t}"))
    return nodes, edges


def canon(nodes_sub: dict, edges_sub: list, ref_color: bool) -> str:
    """Canonical string of a labelled subgraph. Labels: own/enemy relative to ref_color + piece class."""
    sqs = sorted(nodes_sub)
    best = None
    for perm in itertools.permutations(sqs):
        idx = {s: i for i, s in enumerate(perm)}
        lab = [f"{'o' if nodes_sub[s].color == ref_color else 'e'}{CLASS[nodes_sub[s].piece_type]}" for s in perm]
        es = sorted(f"{idx[a]}{t.split(':')[0]}{idx[b]}" for a, b, t in edges_sub if a in idx and b in idx)
        key = "|".join(lab) + "#" + ",".join(es)
        if best is None or key < best:
            best = key
    return best


def describe(key: str) -> str:
    """'eP|eQ|oP#1atk2,1def0' -> 'enemy Q attacks my P; enemy Q defends enemy P'"""
    labs, es = key.split("#")
    labs = labs.split("|")
    name = lambda i: ("my " if labs[int(i)][0] == "o" else "enemy ") + labs[int(i)][1]
    verbs = {"atk": "attacks", "def": "defends", "thru": "x-rays through a blocker to"}
    out = []
    for e in es.split(",") if es else []:
        m = __import__("re").match(r"(\d)(atk|def|thru)(\d)", e)
        if m:
            out.append(f"{name(m.group(1))} {verbs[m.group(2)]} {name(m.group(3))}")
    return "; ".join(out) or " + ".join(name(i) for i in range(len(labs)))


def connected_subsets(nodes, edges, k):
    adj = defaultdict(set)
    for a, b, t in edges:
        adj[a].add(b); adj[b].add(a)
    out = set()
    for s in nodes:
        frontier = [frozenset([s])]
        while frontier:
            cur = frontier.pop()
            if len(cur) == k:
                out.add(cur); continue
            for v in cur:
                for w in adj[v]:
                    if w not in cur:
                        nxt = cur | {w}
                        if len(nxt) <= k:
                            frontier.append(nxt)
    return out


def named_sets(board: chess.Board) -> dict[frozenset, str]:
    """piece-set -> a named relation whose pieces are exactly that set. 'hanging' (arity 1) is
    lifted to the (attacker, victim) pairs it implies so plain captures of loose pieces count as named."""
    out = {}
    for r in relations(board):
        if r.name not in NAMED_STATIC:
            continue
        if r.name == "hanging":
            for a in board.attackers(not r.owner, r.squares[0]):
                out.setdefault(frozenset((a, r.squares[0])), "hanging")
        else:
            out.setdefault(frozenset(r.squares), r.name)
    return out


def tension(edges_sub) -> bool:
    return any(t == "atk" or t.startswith("thru") for _, _, t in edges_sub)


# ---------------- puzzles ----------------
def sample(path, n, stride=100):
    out = []
    with open(path, "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % stride == 0:
                out.append(row)
                if len(out) >= n:
                    break
    return out


def solver_position(row):
    b = chess.Board(row["FEN"]); moves = row["Moves"].split(); b.push_uci(moves[0])
    return b, [chess.Move.from_uci(u) for u in moves[1:]]


def edge_set(board):
    nodes, edges = primitive_graph(board)
    return nodes, {(a, b, t.split(":")[0]) for a, b, t in edges}


def ply_events(board, move):
    """Edge-level events of ONE move, from the mover's side: what enemy defences ended and HOW
    (the defender moved away = 'lured', was captured = 'removed', the line was blocked = 'interposed'),
    what attacks on enemy pieces appeared, what own lines were cleared. The alphabet the classical
    sequence motifs (deflection, attraction, interference, clearance) are defined in."""
    mover = board.turn
    n0, e0 = edge_set(board)
    cap = board.piece_at(move.to_square)
    b1 = board.copy(stack=False); b1.push(move)
    n1, e1 = edge_set(b1)
    ev = []
    if cap: ev.append(f"x{CLASS[cap.piece_type]}")
    if b1.is_check(): ev.append("+")
    if b1.is_checkmate(): ev.append("#")
    gone, new = e0 - e1, e1 - e0
    for a, b, t in gone:
        pa, pb = n0.get(a), n0.get(b)
        if pa is None or pb is None:
            continue
        if t == "def" and pa.color != mover:                         # an enemy defence ended
            how = "removed" if a == move.to_square else "lured" if a == move.from_square else "interposed" if any(c == move.to_square for c in chess.SquareSet.between(a, b)) else "other"
            ev.append(f"enemy_def_ends:{how}")
        if t == "thru" and pa.color != mover:
            ev.append("enemy_line_ends")
        if t == "def" and pa.color == mover and a == move.from_square:
            ev.append("own_def_left")
    for a, b, t in new:
        pa, pb = n1.get(a), n1.get(b)
        if pa is None or pb is None:
            continue
        if t == "atk" and pa.color == mover:
            ev.append(f"atk:{CLASS[pb.piece_type]}" + ("_undef" if not b1.attackers(not mover, b) else ""))
            if a != move.to_square:
                ev.append(f"disc_atk:{CLASS[pb.piece_type]}")          # a piece that did not move now attacks: discovered
        if t == "thru" and pa.color == mover and pb.piece_type in (chess.KING, chess.QUEEN):
            ev.append(f"aim_thru:{CLASS[pb.piece_type]}")             # pin / skewer geometry created
        if t == "thru" and pa.color == mover:
            ev.append("own_line_opens")
    # the moved piece is now capturable: an offer (sacrifice if it is worth more than what takes it)
    if b1.attackers(not mover, move.to_square):
        ev.append("offered")
    return sorted(set(ev))


def seq_signature(board, solution, plies=3):
    sig, t = [], board.copy(stack=False)
    for i, m in enumerate(solution[:plies]):
        if m not in t.legal_moves:
            break
        who = "S" if i % 2 == 0 else "R"
        sig.append(who + "(" + " ".join(ply_events(t, m)) + ")")
        t.push(m)
    return " ".join(sig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seq-n", type=int, default=8000); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--db", default=str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    rows = sample(a.db, max(a.n, a.seq_n), stride=50)
    print(f"[motifs] {len(rows)} puzzles sampled in {time.perf_counter()-t0:.0f}s")
    # ---------- static types ----------
    types = {k: Counter() for k in range(2, a.k + 1)}; covered = {k: Counter() for k in range(2, a.k + 1)}
    example = {}
    t1 = time.perf_counter()
    for row in rows[:a.n]:
        try:
            b, sol = solver_position(row)
        except Exception:
            continue
        nodes, edges = primitive_graph(b)
        named = named_sets(b)
        ref = b.turn
        for k in range(2, a.k + 1):
            for sub in connected_subsets(nodes, edges, k):
                inner = [e for e in edges if e[0] in sub and e[1] in sub]
                if not tension(inner):
                    continue
                key = canon({s: nodes[s] for s in sub}, inner, ref)
                types[k][key] += 1
                if sub in named:
                    covered[k][key] += 1
                example.setdefault(key, (b.fen(), sorted(chess.square_name(s) for s in sub)))
    print(f"[motifs] static enumeration over {min(a.n, len(rows))} positions in {time.perf_counter()-t1:.0f}s")
    for k in range(2, a.k + 1):
        tot = sum(types[k].values()); cov = sum(covered[k].values())
        top = types[k].most_common()
        cum = 0; n90 = 0
        for key, c in top:
            cum += c; n90 += 1
            if cum >= 0.9 * tot: break
        print(f"OBS arity {k}: {len(types[k])} pattern types, {tot} occurrences; {n90} types cover 90% of occurrences; named-relation coverage {cov/max(tot,1):.2f}")
        shown = 0
        for key, c in top:
            if covered[k][key] / c < 0.5 and c >= 0.005 * tot:
                fen, sq = example[key]
                print(f"    UNNAMED {c/tot:5.1%}  {describe(key)}   e.g. {sq}")
                shown += 1
                if shown >= 8: break
    # ---------- sequence signatures ----------
    t2 = time.perf_counter()
    sig_count = Counter(); sig_themes = defaultdict(Counter); theme_named = Counter(); theme_total = Counter()
    for row in rows[:a.seq_n]:
        try:
            b, sol = solver_position(row)
            sig = seq_signature(b, sol)
        except Exception:
            continue
        themes = set(row["Themes"].split())
        sig_count[sig] += 1
        for t in themes:
            sig_themes[sig][t] += 1; theme_total[t] += 1
            if any(ev in sig for ev in ("c:", "u:", "r:", "#")):
                theme_named[t] += 1
    print(f"[motifs] sequence signatures over {min(a.seq_n, len(rows))} solutions in {time.perf_counter()-t2:.0f}s: {len(sig_count)} distinct")
    # for each theme: which single EVENT (any ply) best predicts it -- precision = P(theme | event), recall = P(event | theme)
    ev_total = Counter(); ev_theme = defaultdict(Counter)
    import re as _re
    for sig, c in sig_count.items():
        evs = set(); parts = _re.findall(r"([SR])\(([^)]*)\)", sig)
        per_ply = []
        for who, body in parts:
            es = [f"{who}:{e}" for e in body.split()]
            evs.update(es); per_ply.append(es)
        for i in range(len(per_ply) - 1):                    # two-ply conjunctions: what happened, then what it enabled
            for e1 in per_ply[i]:
                for e2 in per_ply[i + 1]:
                    evs.add(f"{e1} > {e2}")
        for e in evs:
            ev_total[e] += c
            for t, k in sig_themes[sig].items():
                ev_theme[e][t] += k
    print("OBS per-theme DEFINING EVENT (the single ply-event with the best precision x recall for that theme):")
    for t, n in sorted(theme_total.items(), key=lambda kv: -kv[1])[:30]:
        if t in ("short", "long", "veryLong", "oneMove", "master", "masterVsMaster", "superGM", "endgame", "middlegame", "opening", "crushing", "advantage", "equality"):
            continue
        best = max(ev_total, key=lambda e: (ev_theme[e][t] / max(ev_total[e], 1)) * (ev_theme[e][t] / n)) if ev_total else None
        if best is None:
            continue
        prec, rec = ev_theme[best][t] / ev_total[best], ev_theme[best][t] / n
        tag = "named" if t in THEMES_WE_NAME else "UNNAMED" if t in THEMES_UNNAMED else ""
        print(f"    {t:18s} n={n:4d}  {best:28s} precision {prec:.2f} recall {rec:.2f}  {tag}")
    print(f"[motifs] total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
