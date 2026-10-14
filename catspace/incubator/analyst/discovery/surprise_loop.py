"""LEARNING A NEW MATERIAL FROM SURPRISE (Kaveh 2026-10-12: "start playing games and then look at our own trace. And
whenever we are surprised ... look at all the cases that we were surprised and see what lemma can we create that will
help us not be surprised and win this game"; "you can use the KQ and the KR frames ... the KRR frame is the one that is
new"; surprise in the generic language: "when we expect it to shrink, but it doesn't shrink"; "when you were trying for
an outcome, but a different outcome ended up happening").

  PLAY     the planner plays the material from random tablebase wins (the earlier lemmas on: the king-and-queen and
           king-and-rook frames, the tactics); where it proves nothing, the GENERIC FRAME of the material on the board
           (vocab.rs: its measure as terms over the base; none known -- the empty frame); Stockfish with the Syzygy
           tablebases defends
  CLAIMS   every move of ours states one, in the frame language: a frame's -- its measure falls within k moves, or the
           goal (mate, or a capture we keep: the material changes, the next frame takes over); a proof's -- its goal (mate,
           material won) within its horizon (refuted: a prover bug -- the stalemate leaf, game 4 of the first smoke run);
           an arming move's -- the same goal, expected not proved; a defence's -- our lead does not fall below what it said
  CLASSES  Kaveh: "first, don't lose, and then try to win"; "when losing, drag the loss out as long as possible ... same
           thing with drawing" -- by the material lead: AHEAD, hold then win; LEVEL or BEHIND, hold (the hold frame keeps
           their frame's measure high, the mirror of an earlier lemma) -- a draw there is what was claimed
  SURPRISE the game refutes a claim -- RANK 1, ground lost: our lead after our move below the claim's, a loss, or (ahead) a
           draw; RANK 2 (ahead): the horizon passed and the measure did not shrink (nor the goal came), or the game ended
           in another outcome first
  MINE     ahead materials only: the rank-1 boards a hole whose term goes FIRST (schema prepend: don't lose); the rank-2
           boards a hole on the frame so far (append / sum / replace); the pipeline (proposers -> filter -> verifier, whose
           closed loop fails on ground lost) proposes; the best joins the material's frame; play again

  python -m catspace.incubator.analyst.discovery.surprise_loop --material KRRvKR --games 100 --rounds 3 [--tag T [--resume]]
      [--opponent stockfish:10|tablebase]   who defends in the games
      [--defender self|tablebase|stockfish:D]   who defends in the verifier's closed loop (Kaveh: a switch)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

from catspace.io import paths

FRAME_K = 2                    # vocab.rs FRAME_K: the generic frame's horizon
MATING_K = 3                   # mating.rs KQ_TEMPO_MOVES: the mating frames' horizon
ATTACK_K, MATE_K, DEFENCE_K = 10, 10, 2       # a proof's horizon: 10 (game 1064 collected "wins 4 by force" on move 6)
MATING_SPECS = {"KRvK": ["fn:wall", "fn:kr_generated"], "KQvK": ["fn:kq_offedge", "fn:kq_approach", "fn:kq_region"]}
if os.environ.get("CATSPACE_SCRATCH"):        # (from scratch the hand-derived mating frames are off: those materials are mined too)
    MATING_SPECS.clear()
LIMIT = 120                    # moves of ours per game


def lead_of(b) -> tuple:
    """(our material minus theirs, theirs) -- White is us."""
    import chess
    v = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    ours = sum(v[p.piece_type] for p in b.piece_map().values() if p.color and p.piece_type in v)
    theirs = sum(v[p.piece_type] for p in b.piece_map().values() if not p.color and p.piece_type in v)
    return ours - theirs, theirs


def claim_of(trace: list, why: str, material: str, frames: dict, lead: int) -> dict:
    """The claim of a decision, read from its own trace and reason."""
    if any(l.startswith("frame: ") for l in trace):
        # the frame that decided (a conditional frame, a check frame, the material's) and its horizon, from its own claim
        line = next(l for l in trace if l.startswith("frame: "))
        key = re.search(r"the (\S+) frame \[", line)
        k = re.search(r"falls within (\d+) moves", line)
        fkey = key.group(1) if key and key.group(1) in frames else material
        specs = [t for t in frames.get(fkey, []) if not t.startswith("horizon:")]
        return dict(kind="frame", specs=specs, k=int(k.group(1)) if k else FRAME_K, frame=fkey)
    g = re.search(r"the goal field: .* in (\d+) \(\+", why)
    if g:                                                 # a goal forced against every defence: a proof, its own horizon
        return dict(kind="proof", k=int(g.group(1)))
    if why.endswith(("(king and rook)", "(king and queen)")):
        return dict(kind="frame", specs=MATING_SPECS.get(material, []), k=MATING_K)
    if any(w in why for w in ("stops their", "answers their", "avoids the blunder", "cheapest")):
        net = re.search(r"nets (-?\d+)", why)
        return dict(kind="defence", k=DEFENCE_K, floor=lead + min(0, int(net.group(1)) if net else 0))
    if " arms " in why:                                   # a preparing move: its gain is expected, not proved
        return dict(kind="arming", k=ATTACK_K)
    if "mate" in why:                                     # a proof (every reply searched): refuted = a prover bug to find
        return dict(kind="proof", k=MATE_K)
    return dict(kind="proof", k=ATTACK_K)


def _game(args):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    seed, material, frames, opponent, *rest = args
    band = rest[0] if rest else None             # the curriculum's band of the starts' distance (None: any won start)
    import chess
    import chess.engine
    import chess.syzygy
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    from catspace.incubator.analyst.material_walk import _material, _start, castled_start, engine_start
    A.set_generic_frame(True)
    for m, terms in frames.items():
        A.set_material_frame(m, terms)
    tb = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    eng.configure({"SyzygyPath": str(paths.syzygy_dir())})
    men = len(material.replace("v", ""))
    b = (castled_start(random.Random(seed)) if material == "castled"
         else engine_start(random.Random(seed), material, eng) if men > 5          # beyond our tablebases
         else _start(random.Random(seed), material, tb, band))
    st = A.PlanState()
    rows, out, ended = [], None, None
    while out is None:
        if b.is_checkmate():
            out = "win" if b.turn == chess.BLACK else "loss"; break
        if b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100 or b.is_repetition(3):
            out = "draw"
            ended = ("stalemate" if b.is_stalemate() else "insufficient material" if b.is_insufficient_material()
                     else "fifty moves" if b.halfmove_clock >= 100 else "repetition")
            break
        if b.turn == chess.WHITE:
            if len(rows) >= LIMIT:
                out = "not finished"; break
            mat = _material(b)
            lead, theirs = lead_of(b)
            tr, dec, *_ = st.observe(b.fen())
            m = None
            if dec:
                for u in (dec[0], dec[1]):
                    try:
                        mv = chess.Move.from_uci(u)
                    except (ValueError, TypeError):
                        continue
                    if mv in b.legal_moves:
                        m = mv; break
            why = dec[3] if dec else "no move"
            claim = claim_of(tr, why, mat, frames, lead) if m else dict(kind="none", k=1)
            if m is None:                                     # no safe move at all: the first legal, said so
                m = next(iter(b.legal_moves))
            rows.append(dict(fen=b.fen(), material=mat, lead=lead, theirs=theirs, san=b.san(m), why=why, claim=claim, trace=tr))
            b.push(m)
            rows[-1]["lead_after"] = lead_of(b)[0]           # the lead our move leaves (a trade is not a loss)
        else:
            st.record(b.fen())
            if opponent.startswith("tablebase"):
                from . import defenders as D
                D._TB, D._ENG = tb, eng
                b.push(D.reply(b, "tablebase"))
            else:
                b.push(eng.play(b, chess.engine.Limit(depth=int(opponent.partition(":")[2] or 10))).move)
    eng.quit()
    if out in ("win", "loss") and ended is None:
        ended = "mate"
    return dict(seed=seed, out=out, ended=ended, moves=len(rows), rows=rows)


def judge(g: dict) -> None:
    """Each claim against the game. RANK 1 'lost ground' (our lead after our move below the claim's, a loss, or -- ahead --
    a draw within the horizon); then, ahead, RANK 2: 'kept', 'no shrink' (the horizon passed), 'other outcome' (the game
    ended first); not ahead, holding was the claim -- 'kept'."""
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    rows = g["rows"]
    cache = {}
    for i, r in enumerate(rows):
        c = r["claim"]
        if c["kind"] == "none":
            r["verdict"] = "no move"; continue
        win = rows[i + 1:i + 1 + c["k"]]
        ended = len(win) < c["k"]
        ahead = r["lead"] > 0
        clock = ended and g["out"] == "draw" and g.get("ended") == "fifty moves"
        if (min([r["lead_after"]] + [x["lead_after"] for x in win]) < r["lead"] or (ended and g["out"] == "loss")
                or (ahead and ended and g["out"] == "draw" and not clock)):
            r["verdict"] = "lost ground"; continue
        if ahead and clock:
            # (a draw by the fifty-move clock is fifty moves of no progress, not a move that lost ground: rank 2 --
            # 2026-10-12, the rank-1 holes were boards a median 2 moves from the clock's draw)
            r["verdict"] = "no shrink"; continue
        if not ahead:
            r["verdict"] = "kept"; continue
        reached = lambda x: x["theirs"] < r["theirs"] and x["lead"] >= r["lead"]      # the goal: a capture we kept
        verdict = None
        if c["kind"] == "defence":
            if any(x["lead"] < c["floor"] for x in win):
                verdict = "no shrink"                       # the lead fell below what the defence said
            elif len(win) < c["k"] and g["out"] not in ("win",):
                verdict = "other outcome"
            else:
                verdict = "kept"
        else:
            if any(reached(x) for x in win) or (len(win) < c["k"] and g["out"] == "win"):
                verdict = "kept"
            elif c["kind"] == "frame" and c["specs"]:
                key = tuple(c["specs"])
                if key not in cache:
                    vals = A.component_values([x["fen"] for x in rows], list(key))
                    cache[key] = vals
                vals = cache[key]
                if any(rows[i + 1 + j]["material"] == r["material"] and vals[i + 1 + j] < vals[i] for j in range(len(win))):
                    verdict = "kept"
            if verdict is None:
                verdict = "other outcome" if len(win) < c["k"] else "no shrink"
        r["verdict"] = verdict


def approach(rows: list, idx: list, end: int) -> list:
    """THE APPROACH TO A SURPRISE (Kaveh 2026-10-12: "if the surprise at one position wants to get us a lemma to solve that
    position, we have to look at all the positions leading up to that because that's where the lemma would apply"): our
    positions of the material's phase up to the surprise, ahead (where a frame of ours plays)."""
    return [rows[i]["fen"] for i in idx if i <= end and rows[i]["lead"] > 0]


def find_loops(g: dict, frames: dict) -> list:
    """THE LOOPS OF A GAME (Kaveh 2026-10-12: "once we go into a drawn tail, we recognize that we've gone into a drawn tail
    ... and give signal back to the proposer that we're looping"): in each material we are ahead in, with the frame it
    was played under, our positions as lemma-states (Rust lemma_state, with the material) and the frame's measure; the
    first return to a state without the measure lower than at its last visit is THE LOOP -> [{material, the entry (the
    position where that state was first reached), the revisit, the states of the cycle, whether the phase was won}]."""
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    rows, out = g["rows"], []
    phases = defaultdict(list)
    for i, r in enumerate(rows):
        phases[r["material"]].append(i)
    for mat, idx in phases.items():
        if mat in MATING_SPECS or rows[idx[0]]["lead"] <= 0:
            continue
        terms = frames.get(mat, [])
        fens = [rows[i]["fen"] for i in idx]
        states = A.lemma_state(fens)
        meas = A.component_values(fens, terms) if terms else [[] for _ in fens]
        seen = {}
        for j, i in enumerate(idx):
            key = (states[j], rows[i]["lead"], rows[i]["theirs"])
            if key in seen and not meas[j] < meas[seen[key][1]]:
                first = seen[key][0]
                last = idx[-1]
                won = (last + 1 < len(rows) and rows[last + 1]["theirs"] < rows[last]["theirs"]) or (last == len(rows) - 1 and g["out"] == "win")
                out.append(dict(material=mat, entry=first, revisit=i, entry_fen=rows[first]["fen"], won=won,
                                cycle=len({states[idx.index(x)] for x in idx if first <= x <= i})))
                break
            seen[key] = (i, j)
    return out


def play(material: str, games: int, frames: dict, seed0: int, workers: int = max(1, (os.cpu_count() or 8) - 2), opponent: str = "stockfish:10",
         band: tuple | None = None) -> list:
    from multiprocessing import Pool
    with Pool(workers) as p:
        res = sorted(p.imap_unordered(_game, [(seed0 + g, material, frames, opponent, band) for g in range(games)], chunksize=1), key=lambda r: r["seed"])
    for g in res:
        judge(g)
    return res


def _tb_play(fen: str) -> int | None:
    """The tablebase's own play from a start (the winner: the fewest plies to zeroing -- in these endings the mate; the
    loser: the longest) -> our moves to mate, None when not mated."""
    import chess
    import chess.syzygy
    with chess.syzygy.open_tablebase(str(paths.syzygy_dir())) as tb:
        b = chess.Board(fen)
        n = 0
        for _ in range(300):
            if b.is_game_over(claim_draw=True):
                break
            best = None
            for m in b.legal_moves:
                x = b.copy(stack=False); x.push(m)
                if x.is_checkmate() and b.turn == chess.WHITE:
                    best = (-1, m); break
                try:
                    d = tb.probe_dtz(x)
                except Exception:
                    return None
                key = (abs(d) if tb.probe_wdl(x) == -2 else 10 ** 6) if b.turn == chess.WHITE else -d
                if best is None or key < best[0]:
                    best = (key, m)
            if b.turn == chess.WHITE:
                n += 1
            b.push(best[1])
        return n if b.is_checkmate() else None


def benchmark(res: list) -> None:
    """THE BENCHMARK (Kaveh 2026-10-13: "how many moves does it take us to do it versus ..."): the tablebase's own play from
    the round's starts against ours, game by game, where both mated (5 men or fewer, no pawn)."""
    starts = [g["rows"][0]["fen"] for g in res if g["rows"]]
    if not starts or any("p" in f.split(" ")[0].lower() for f in starts) or len(starts[0].split(" ")[0].replace("/", "").strip("12345678")) > 5:
        return
    from multiprocessing import Pool
    with Pool(max(1, (os.cpu_count() or 8) - 2)) as p:
        tb = p.map(_tb_play, starts)
    pairs = [(g["moves"], t) for g, t in zip([g for g in res if g["rows"]], tb) if g["out"] == "win" and t]
    if pairs:
        print(f"OBS   THE BENCHMARK: where both mated ({len(pairs)}): our moves to mate {sum(a for a, _ in pairs) / len(pairs):.1f}, "
              f"the tablebase's {sum(b for _, b in pairs) / len(pairs):.1f}; the tablebase mated {sum(t is not None for t in tb)} of {len(tb)}")


def episode_report(res: list, frames: dict) -> None:
    """THE PROGRESS MEASURE (Kaveh: "are we measuring progress properly?"): the episodes of each phase we are ahead in --
    the share that ended well (material won, or a new best of the frame's measure: the ratchet)."""
    from . import episodes as E
    tot = defaultdict(Counter)
    for g in res:
        phases = defaultdict(list)
        for i, r in enumerate(g["rows"]):
            phases[r["material"]].append(i)
        for mat, idx in phases.items():
            if mat in MATING_SPECS or len(idx) < 2 or g["rows"][idx[0]]["lead"] <= 0:
                continue
            for e in E.episodes(g["rows"], idx, frames.get(mat, [])):
                tot[mat][e["outcome"]] += 1
    for mat, c in tot.items():
        n = sum(c.values())
        print(f"OBS   {mat} episodes: {n}, ended well {100 * sum(c[o] for o in E.DESIRED) / max(n, 1):.0f}% ({dict(c.most_common())})")


def report(res: list, label: str) -> None:
    rows = [r for g in res for r in g["rows"]]
    won = [g["moves"] for g in res if g["out"] == "win"]
    lost = [g["moves"] for g in res if g["out"] == "loss"]
    print(f"OBS {label}: {len(res)} games {dict(Counter(g['out'] for g in res))}"
          f"{f', {sum(won) / len(won):.1f} moves per win' if won else ''}{f', a loss lasted {sum(lost) / len(lost):.1f} moves' if lost else ''}; {len(rows)} moves of ours")
    # WHY GROUND WAS LOST (Kaveh: "when we lose ground we should stop and look at why"): how the drawn and lost games
    # ended, the material at the end, and how often we were in check in our last 20 positions (their rook harassing)
    import chess
    bad = [g for g in res if g["out"] in ("draw", "loss")]
    if bad:
        ends = Counter((g["out"], g.get("ended"), g["rows"][-1]["material"] if g["rows"] else "-") for g in bad)
        checked = sorted(sum(chess.Board(x["fen"]).is_check() for x in g["rows"][-20:]) for g in bad)
        print("OBS   ground lost, why: " + "; ".join(f"{o} by {e} in {m} {n}" for (o, e, m), n in ends.most_common(6)) +
              f" -- in check in our last 20 positions: median {checked[len(checked) // 2]}")
    for mat in sorted({r["material"] for r in rows}, key=lambda m: -sum(r["material"] == m for r in rows)):
        here = [r for r in rows if r["material"] == mat]
        by = Counter((r["claim"]["kind"], r["verdict"]) for r in here)
        sur = sum(v for (k, ver), v in by.items() if ver in ("no shrink", "other outcome", "lost ground"))
        print(f"OBS   {mat} ({len(here)} moves): surprised {sur} ({100 * sur / len(here):.0f}%) -- " +
              ", ".join(f"{k} {ver} {v}" for (k, ver), v in sorted(by.items())))


def compose(base: list, term: str, schema: str) -> list:
    if schema == "prepend":
        return [term] + base
    if schema == "append" or not base:
        return base + [term]
    if schema == "sum":
        return base[:-1] + [f"(sum {base[-1]} {term})"]
    return base[:-1] + [term]                               # replace


def guard_games(res: list, played: dict, key: str, terms: list, material: str, opponent: str = "stockfish:10", n: int = 40,
                band: tuple | None = None) -> dict:
    """FIRST, DON'T LOSE -- THE GUARD ON WHOLE GAMES (the night's autonomy, 2026-10-12: a K+Q v K+R frame promoted on
    failure boards -- where nothing holds, progress alone decided -- lost the queen to a skewer in 5 of round 2's games;
    replayed without it, 4 of those 5 kept the queen, one was won; a guard on positions of won phases saw nothing): a
    sample of the round's own starts (half from its won games, half from the rest) replayed with the candidate added to
    the frames the round was PLAYED with, each against the round's recorded game (played without it) on (the result,
    then the material lead kept); vetoed when worse in more games than better. A veto, never evidence for promotion."""
    from multiprocessing import Pool
    import random
    won = [g for g in res if g["out"] == "win"]
    rest = [g for g in res if g["out"] != "win"]
    rng = random.Random(f"{key}|{terms}")
    sample = rng.sample(won, min(n // 2, len(won))) + rng.sample(rest, min(n - min(n // 2, len(won)), len(rest)))
    frames_c = dict(played)
    frames_c[key] = terms
    with Pool(max(1, (os.cpu_count() or 8) - 2)) as p:
        again = p.map(_game, [(g["seed"], material, frames_c, opponent, band) for g in sample])
    val = lambda g: ({"win": 2, "draw": 1}.get(g["out"], 0), g["rows"][-1]["lead_after"] if g["rows"] else 0)
    w = sum(val(b) > val(a) for a, b in zip(sample, again))
    l = sum(val(b) < val(a) for a, b in zip(sample, again))
    return dict(w=w, l=l, veto=l > w, n=len(sample))


def to_test(best: dict | None, olds: list, cur: list) -> list:
    """The candidates a hole's confirmation boards test this round: the round's best (no evidence yet) and those PENDING
    from earlier rounds with their votes carried -- unless the frame they were measured against has changed since; the
    same candidate twice is tested once, its votes carried."""
    tested = [dict(term=best["term"], schema=best["schema"], w0=0, l0=0, rounds=0)] if best else []
    for old in olds:
        if old["base"] != cur:
            continue
        carried = dict(term=old["term"], schema=old["schema"], w0=old["w"], l0=old["l"], rounds=old["rounds"])
        same = [t for t in tested if (t["term"], t["schema"]) == (old["term"], old["schema"])]
        if same:
            same[0].update(carried)
        else:
            tested.append(carried)
    return tested


def futile(w: int, l: int, more: int, bar: float) -> bool:
    """THE REVERSE E-VALUE (Kaveh 2026-10-12: "not every lemma deserves to be going on forever"; the literature check --
    futility as an e-process): p1 is the least game win rate that would reach the bar over every game the candidate can
    have in the run (decided so far and `more` to come); e = (1/2 / p1)^w (1/2 / (1 - p1))^l is the evidence that its rate
    is below p1 (an e-value for "rate >= p1"); at the bar, the candidate is not worth the games. (Dropping never makes a
    promotion invalid -- it only gives up power.)"""
    from .defenders import sign_e
    n = w + l + more
    p1 = next((k / n for k in range(n // 2 + 1, n + 1) if sign_e(k, n - k) >= bar), None) if n else None
    if p1 is None:
        return True                                 # no rate reaches the bar in the games left
    if p1 >= 1:
        return l > 0                                # only winning every game would do, and one is lost
    return (0.5 / p1) ** w * (0.5 / (1 - p1)) ** l >= bar


def settle(tested: list, bar: float, more: int) -> tuple:
    """After the confirmation (each t with its pooled votes w, l and their e): -> (the promoted -- the most evidence at or
    over the bar -- or None, the ones kept pending): those still winning more games than they lose AND able to reach the
    bar with the votes still to come in the run (`more`: winning every one of them) -- the rest dropped: the opportunity
    cost (Kaveh 2026-10-12: "not every lemma deserves to be going on forever"; the deterministic futility rule loses no
    power: a candidate that cannot reach the bar can never be promoted)."""
    from .defenders import sign_e
    win = max((t for t in tested if t["e"] >= bar), key=lambda t: t["e"], default=None)
    if win:
        return win, []
    return None, sorted((t for t in tested if t["w"] > t["l"] and sign_e(t["w"] + more, t["l"]) >= bar and not futile(t["w"], t["l"], more, bar)),
                        key=lambda t: -t["e"])


def mine(res: list, frames: dict, rnd: int, budget: int, cap: int = 100, tag: str = "", defender: str = "self", n_boards: int = 30,
         register: bool = True, pending: dict | None = None, rounds_left: int = 0, opponent: str = "stockfish:10",
         band: tuple | None = None) -> dict:
    """Each material's refuted frame claims -> a hole -> the pipeline -> the best term joins the material's frame.
    `pending` ({"<frame key>|rank<n>": [candidates]}, updated in place): the candidates whose evidence is still open;
    `rounds_left`: the confirmations still to come in the run (a pending candidate's budget)."""
    pending = {} if pending is None else pending
    from . import base_vocabulary as V
    from . import holes as H
    from .pipeline import run
    new = dict(frames)
    by = defaultdict(dict)                     # (frame key, rank) -> {board: its game}
    contrast = defaultdict(lambda: {"win": [], "draw": []})
    diverged = Counter()
    trajs = defaultdict(list)
    from . import episodes as E
    for g in res:
        # THE GAMES' CONTRAST, per material (the contrast proposer's evidence): positions of ours shortly before a win of
        # that material's phase (a capture we kept, or mate), and the tails of the games drawn or lost in it
        phases = defaultdict(list)
        for i, r in enumerate(g["rows"]):
            phases[r["material"]].append(i)
        for mat, idx in phases.items():
            if mat in MATING_SPECS:
                continue
            last = idx[-1]
            left_won = last + 1 < len(g["rows"]) and g["rows"][last + 1]["theirs"] < g["rows"][last]["theirs"]
            if left_won or (last == len(g["rows"]) - 1 and g["out"] == "win"):
                contrast[mat]["win"] += [g["rows"][i]["fen"] for i in idx[-5:]]
            elif last == len(g["rows"]) - 1 and g["out"] in ("draw", "loss"):
                contrast[mat]["draw"] += [g["rows"][i]["fen"] for i in idx[-30:]]
        # THE EPISODES (Kaveh 2026-10-12: "a frame where we have a desired outcome and ... a non-desired outcome ...
        # episodic"; "once we figured out that we're not progressing, go back and identify the point at which we started
        # to diverge"): each phase we are ahead in, cut into episodes (episodes.py); the SURPRISES are the episodes that
        # ended badly -- in a phase that did not progress, from THE DIVERGENCE on (the first failed episode after the last
        # good one, and the two after it); in a won phase, its first failed episodes. Their KIND picks the frame to learn:
        # in check -> the material's CHECK frame; material lost -> rank 1 (don't lose); the rest -> progress (rank 2)
        for mat, idx in phases.items():
            if mat in MATING_SPECS or len(idx) < 2 or g["rows"][idx[0]]["lead"] <= 0:
                continue
            eps = E.episodes(g["rows"], idx, frames.get(mat, []))
            last = idx[-1]
            won = (last + 1 < len(g["rows"]) and g["rows"][last + 1]["theirs"] < g["rows"][last]["theirs"]) or (last == len(g["rows"]) - 1 and g["out"] == "win")
            if won:
                region = [e for e in eps if e["outcome"] not in E.DESIRED][:3]
            else:
                lg = max((k for k, e in enumerate(eps) if e["outcome"] in E.DESIRED), default=-1)
                region = [e for e in eps[lg + 1:lg + 4] if e["outcome"] not in E.DESIRED]
                if region:
                    diverged[mat] += 1
            for e in region:
                k0 = idx.index(e["start"])
                fens_e = [g["rows"][i]["fen"] for i in idx[k0:k0 + e["length"]] if g["rows"][i]["lead"] > 0]
                fk = f"{mat}:check" if e["kind"] == "check" else mat
                trajs[(fk, 1 if e["outcome"] == "material lost" else 2)].append((g["seed"], fens_e))
        for r in g["rows"]:
            # WHERE OUR OWN PROOF REACHES WHAT WE KNOW (the forced entry fired) is the "good" side of the contrast: the miner
            # learns to steer toward the positions our proofs can bank on -- labelled by our lemmas, no engine
            if ("forces entry into the known" in r["why"] or "prepares a forced entry" in r["why"]) and r["material"] not in MATING_SPECS:
                contrast[r["material"]]["win"].append(r["fen"])     # (regression levels 1 and 2: labelled by our proofs)
    if diverged:
        print("OBS   phases that did not progress, mined from their divergence: " + ", ".join(f"{m} {n}" for m, n in diverged.items()))
    # THE FAILED EPISODES INTO BOARDS: each gives its positions (evenly spaced when long), every failing game counted
    for key, ts in trajs.items():
        ts = [(s, t) for s, t in ts if t]
        per = max(3, cap // max(len(ts), 1))
        for s, t in ts:
            step = max(1, len(t) // per)
            for f in t[::step][-per:]:
                by[key].setdefault(f, s)
        print(f"OBS   {key[0]} rank {key[1]}: {len(ts)} failed episodes, {sum(len(t) for _, t in ts)} positions -> {len(by[key])} boards "
              f"from {len(set(by[key].values()))} games")
    for (fk, rank), fens in sorted(by.items(), key=lambda kv: (kv[0][1], -len(kv[1]))):
        mat = fk.split(":")[0]                     # the material; fk the frame learned (the material's, or its check frame)
        if len(fens) < 10:
            print(f"OBS   {fk} rank {rank}: {len(fens)} surprise boards -- too few"); continue
        # FRESH CLOCKS: the hole asks whether a frame makes progress FROM the position -- not whether it wins before a clock
        # the game had already run down (2026-10-12: the rank-1 boards were a median 2 moves from the fifty-move draw)
        def fresh(f):
            import chess
            b = chess.Board(f)
            b.halfmove_clock = 0
            return b.fen()
        picked = random.Random(rnd).sample(sorted(fens), min(cap, len(fens)))
        from .verifier import split, confirm
        # choosing and confirming on DIFFERENT GAMES (verifier.split); the hole's boards ordered choosing first -- the
        # pipeline chooses on the first len(choose)
        choose, check, check_games = split([fresh(f) for f in picked], n_boards, [fens[f] for f in picked], seed=f"{tag}|{fk}|{rank}|{rnd}")
        boards = choose + check
        gid = {fresh(f): fens[f] for f in picked}
        name = f"learn_{tag + '_' if tag else ''}{fk.replace(':', '_')}_rank{rank}_r{rnd}"   # the run's tag: no run reads another's hole
        cur = new.get(fk, [])
        rng = random.Random(rnd)
        pool = contrast[mat]
        if fk.endswith(":check"):                   # the check frame's evidence: the positions in check
            import chess
            pool = {k: [f for f in v if chess.Board(f).is_check()] for k, v in pool.items()}
        con = {k: rng.sample(v, min(150, len(v))) for k, v in pool.items()}
        hole = dict(name=name, material=mat, frame_key=fk, frames=dict(new), piece="*", base=cur, k=H.K, stuck=boards, sampled=len(fens),
                    games=[gid[b] for b in boards],
                    schemas=["prepend"] if rank == 1 and cur else None, contrast=con if con["win"] and con["draw"] else None)
        json.dump(hole, open(paths.derived(f"hole_{name}.json"), "w"))
        print(f"OBS   {fk} rank {rank} ({'material lost' if rank == 1 else 'failed episodes'}: from the divergence): {len(fens)} surprise boards -> hole {name} "
              f"(base {cur or 'none'}; {len(boards)} boards; the games' contrast: {len(con['win'])} before a win, {len(con['draw'])} in draws)")
        t0 = time.time()
        out = run(name, budget, hole=hole, defender=defender, n_boards=len(choose))
        from .pipeline import best_of
        base, best = out[0], best_of(out[1:])
        cf = None
        if "ranks" in base:                      # a real defender: confirmed board by board on boards not used to choose
            # THE EVIDENCE CARRIED ACROSS ROUNDS (Kaveh 2026-10-12, "Yes" -- an inconclusive e-value continues on the next
            # round's boards instead of the loop stopping): the round's best and the candidates PENDING for this hole,
            # each confirmed on this round's held-out GAMES (one vote a game); a pending candidate's votes are POOLED across
            # rounds and the one-sided sign e-value taken on the pool (the mixture martingale: anytime valid -- the next
            # round's games are new). Each candidate's bar is 20 FOR LIFE (Ville: its chance of ever being promoted while
            # no better is at most 1/20 -- re-charging it per round adds nothing: the literature check, 2026-10-12). Not
            # promoted, a candidate stays pending while it wins more games than it loses and can still reach the bar in
            # the run's remaining rounds; else dropped (or when the frame it was measured against changes)
            pkey = f"{fk}|rank{rank}"
            olds = pending.pop(pkey, [])
            olds = [olds] if isinstance(olds, dict) else olds          # (a run resumed from the one-pending format)
            for o in olds:
                if o["base"] != cur:
                    print(f"OBS   the pending {o['term']} dropped: the {fk} frame it was measured against changed")
            tested = to_test(best, olds, cur)
            from .defenders import E_PROMOTE, sign_e
            bar = E_PROMOTE
            more = len(set(check_games)) * rounds_left
            for t in tested:
                c = confirm(hole, t, check, defender, check_games)
                t.update(cf=c, w=t["w0"] + c["w"], l=t["l0"] + c["l"], rounds=t["rounds"] + 1)
                t["e"] = sign_e(t["w"], t["l"])
                who = "the round's best" if best and t is tested[0] else "pending"
                if t["rounds"] > 1:
                    who += f", tested since {t['rounds'] - 1} round(s) before"
                print(f"OBS   the confirmation ({c['n']} boards of {c['games']} games not used to choose): {t['term']} ({who}) "
                      f"beats the frame before it in {c['w']} games, loses in {c['l']} -- e-value {c['e']:.3g} this round, {t['e']:.3g} "
                      f"pooled over its rounds ({t['w']} to {t['l']}; the bar {bar:.0f})")
            win, keep = settle(tested, bar, more)
            if win:                                   # FIRST, DON'T LOSE: the guard on whole games (guard_games)
                guard = guard_games(res, frames, fk, compose(cur, win["term"], win["schema"]), mat, opponent, band=band)
                print(f"OBS   the guard ({guard['n']} of the round's starts replayed with {win['term']}): better in {guard['w']} games, worse in "
                      f"{guard['l']} on (the result, the material kept) -- {'VETOED: it breaks what the frames already do' if guard['veto'] else 'passed'}")
                if guard["veto"]:
                    win = None
            wins = bool(win)
            if win:
                cf = dict(win["cf"], e=win["e"], w=win["w"], l=win["l"], rounds=win["rounds"])
                best, base = dict(term=win["term"], schema=win["schema"], score=cf["cand"]["score"]), cf["base"]
            else:
                if keep:
                    pending[pkey] = [dict(term=k["term"], schema=k["schema"], base=cur, e=k["e"], w=k["w"], l=k["l"], rounds=k["rounds"]) for k in keep]
                for t in tested:
                    if t in keep:
                        print(f"OBS   PENDING for {fk} rank {rank}: {t['term']} -- {t['w']} to {t['l']}, e-value {t['e']:.3g}; continued on the next round's games")
                    else:
                        print(f"OBS   dropped for {fk} rank {rank}: {t['term']} -- {t['w']} to {t['l']}: "
                              + ("it loses as many games as it wins" if t["w"] <= t["l"] else
                                 f"winning all {more} games still to come in the run could not reach the bar" if sign_e(t["w"] + more, t["l"]) < bar else
                                 "the reverse e-value says its game win rate is below what the games left need"))
        else:
            wins = bool(best) and best["score"] > base["score"] + 0.05
        if wins:
            new[fk] = compose(cur, best["term"], best["schema"])
            # INTO THE GENERATED TRACK, by the loop itself (Kaveh: "couldn't the proposer do it?"): the term, its hole, its
            # verdict against the frame before it; the Lean definitions and Rust constants re-exported
            from . import registry as R
            what = "hold" if rank == 1 else "progress"
            ev = {"verifier": f"{best['score']:.4f} against the frame's own {base['score']:.4f} (defender {defender})",
                  "status": "a conjecture: its frame's claim is not proved"}
            if cf:
                c, b0 = cf["cand"], cf["base"]
                ev["verifier"] = (f"on {cf['n']} confirmation boards: held {c['n'] - c['lost']}/{c['n']}, won {c['mated']} -- the frame before it "
                                  f"held {b0['n'] - b0['lost']}/{b0['n']}, won {b0['mated']}; better on {cf['w']} boards, worse on {cf['l']} over "
                                  f"{cf['rounds']} round(s), e-value {cf['e']:.3g} (defender {defender})")
            if register:                          # (a --dry test run writes nothing to the generated track)
                R.add_generated(f"{fk.lower().replace(':', '_')}_{tag or 'run'}_r{rnd}_{what}", best["term"], "measure", name, ev,
                                used_by=f"the {fk} frame ({best['schema']}): {new[fk]}")
                R.export()
            print(f"OBS   MINED for {fk} ({time.time() - t0:.0f} s): [{best['schema']}] {best['term']} = "
                  f"{V.gloss(V.parse(best['term']))} -- {best['score']:.3f} against {base['score']:.3f}; the frame now {new[fk]}")
        else:
            print(f"OBS   nothing mined for {fk} rank {rank} ({time.time() - t0:.0f} s): the best {best and best['score']:.3f} against the frame's own {base['score']:.3f}")
    return new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="KRRvKR")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--frames", default=None, help="a JSON of {material: [terms]} to start from (default: the generated track's)")
    ap.add_argument("--fresh", action="store_true", help="start with no frames for the new materials (not the generated track's)")
    ap.add_argument("--dry", action="store_true", help="a test: write nothing to the generated track (its terms or its frames)")
    ap.add_argument("--tag", default=None, help="this run's name in its files (default: the start time); never overwritten")
    ap.add_argument("--resume", action="store_true", help="continue the run --tag names from its last round's mining")
    ap.add_argument("--dtz", default=None, help="LO:HI -- start only where the tablebase's distance to zeroing (plies) lies in the band")
    ap.add_argument("--ramp", action="store_true", help="THE CURRICULUM: after a round winning two thirds of its games, the band's top doubles")
    ap.add_argument("--opponent", default="stockfish:10", help="who defends in the games: stockfish:D (with the tablebases), tablebase")
    ap.add_argument("--defender", default="self", help="the verifier's closed-loop opponent: self, tablebase, stockfish:D")
    ap.add_argument("--boards", type=int, default=30, help="boards per hole the verifier plays against a real defender")
    ap.add_argument("--challenger", action="store_true", help="after each round's mining: the challenger's traps and conditional frames")
    ap.add_argument("--scratch", action="store_true",
                    help="FROM SCRATCH (Kaveh 2026-10-13): no frames, no registry lemmas, the hand-derived mating frames off, no "
                         "proposer trained on past runs, its own experience store, nothing written to the generated track")
    ap.add_argument("--loops", action="store_true", help="after each round's mining: loop frames from the round's own games")
    a = ap.parse_args(argv)
    tag = a.tag if a.tag is not None else time.strftime("%m%d_%H%M")
    if a.scratch:
        os.environ["CATSPACE_SCRATCH"] = tag                      # (every worker started from here on sees it)
        os.makedirs(paths.derived(f"scratch/{tag}"), exist_ok=True)
        MATING_SPECS.clear()
        import analyst_core as A
        A.set_kq_tactic(False)
        a.dry, a.fresh, a.frames = True, True, None
    stem = f"surprise_loop_{a.material}" + (f"_{tag}" if tag else "")
    state = paths.derived(f"{stem}.json")
    from . import registry as R
    frames = json.load(open(a.frames)) if a.frames else ({} if a.fresh else R.load_frames())
    history, start, res, pending, pending_frames = [], 0, None, {}, {}
    band = tuple(int(x) for x in a.dtz.split(":")) if a.dtz else None
    if a.resume:                         # the last round played: its games are saved, its mining is redone
        history = json.load(open(state))
        last = history[-1]
        start, frames = last["round"], last["frames"]
        pending = dict(history[-2].get("pending", {})) if len(history) > 1 else {}    # the evidence open before that mining
        pending_frames = dict(history[-2].get("pending_frames", history[-2].get("pending_stages", {}))) if len(history) > 1 else {}
        band = tuple(last["band"]) if last.get("band") else band
        res = json.load(open(paths.experiment(f"{stem}_r{start}.json")))
        print(f"OBS resuming {stem} at round {start}'s mining (frames {frames or 'none'})")
    elif os.path.exists(state):
        raise SystemExit(f"{state} exists -- a new --tag, or --resume")
    for rnd in range(start, a.rounds + 1):
        if res is None:
            t0 = time.time()
            res = play(a.material, a.games, frames, seed0=1000 * rnd, opponent=a.opponent, band=band)
            report(res, f"round {rnd} (frames {frames or 'none'}; {time.time() - t0:.0f} s)")
            benchmark(res)
            episode_report(res, frames)
            json.dump(res, open(paths.experiment(f"{stem}_r{rnd}.json"), "w"))
            history.append(dict(round=rnd, frames=frames, outcomes=dict(Counter(g["out"] for g in res)), band=band))
            json.dump(history, open(state, "w"), indent=1)
        if rnd == a.rounds:
            break
        # THE CURRICULUM (the night's autonomy, 2026-10-12: K+B+N v K drew 30 of 30 -- with no game won there is nothing to
        # learn from but progress): starts within a band of the tablebase's distance; a round winning two thirds of its
        # games widens it -- harder starts next round
        widened = False
        if a.ramp and band and sum(g["out"] == "win" for g in res) * 3 >= 2 * len(res):
            band, widened = (band[0], band[1] * 2), True
            print(f"OBS THE CURRICULUM: two thirds of the round won -- the starts' band widens to {band[0]}-{band[1]} plies")
        new = mine(res, frames, rnd, a.budget, tag=tag, defender=a.defender, n_boards=a.boards, register=not a.dry, pending=pending,
                   rounds_left=a.rounds - 1 - rnd, opponent=a.opponent, band=band)
        if a.challenger:
            # THE CHALLENGER (Kaveh: "reason both sides"): traps against the round's frames, answered by conditional frames
            from . import challenger as C
            new = C.round_(res, frames, new, tag or "run", rnd, register=not a.dry, pending=pending_frames, rounds_left=a.rounds - 1 - rnd)
        if a.loops:
            # LOOP FRAMES FROM THE ROUND'S OWN GAMES (Kaveh: "loops in the frames"): an arrangement seen again with their
            # king shifted, checked exactly, refined, held out -- added as conditional frames with their horizons
            from . import patterns as Pt
            got = Pt.derive_loops([paths.experiment(f"{stem}_r{rnd}.json")], {g["rows"][0]["material"] for g in res if g["rows"]})
            new = dict(new); new.update(got)
            print(f"OBS LOOP FRAMES from round {rnd}'s own games: {len(got)} kept")
        history[-1]["pending"], history[-1]["pending_frames"] = dict(pending), dict(pending_frames)
        json.dump(history, open(state, "w"), indent=1)
        if new == frames and not pending and not pending_frames and not widened:
            print("OBS nothing new mined and no evidence pending -- stopping"); break
        if new == frames:
            print(f"OBS nothing new mined; evidence pending for {sorted(pending)} -- the next round's boards continue it")
        if not a.dry:
            R.save_frames(new, f"{stem} round {rnd}", owned={k.split(":")[0] for k in new} | {a.material})  # the next run's start
        frames, res = new, None


if __name__ == "__main__":
    main()
    from . import proposers as _P
    _P.wait()                                   # no training left orphaned
