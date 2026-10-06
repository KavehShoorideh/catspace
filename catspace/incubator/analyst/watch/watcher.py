"""THE WATCHER'S LOOP. Games come from the Lichess records on disk (game_id % 97 == 7; any players), in file order. After
every move it forecasts, then the game reveals:
  short  the mover's material change once the exchanges settle (frames.observe), as the sum of one forecast per FRAME
         (a contested square) and one for ELSEWHERE (captures on squares that were not frames), convolved;
  long   the game's result for the mover (games lost on time are censored: no forecast is scored, nothing counted).
Surprise is -log2 of the probability given to what happened, in bits. Counts are updated only when an outcome is
revealed (a window's outcome when it closes, a result when the game ends), so no forecast sees its own answer.
    python -m catspace.incubator.analyst.watch.watcher batch [--games 20]     # the next batch of the stream
    python -m catspace.incubator.analyst.watch.watcher show [--batch N] [--top 20]
    python -m catspace.incubator.analyst.watch.watcher add --kind frame --parent F0 --cond '{...}' --text "..." --cornerstone ID
State (counts, the stream's position, batch summaries) in data/derived/watch/state.json; each batch's cases in
data/derived/watch/batch_NNN.json; the rules and the batch summaries in artifacts/watch (committed)."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import chess
import numpy as np

from catspace.io import paths

from .cases import COLS as CASE_COLS, N_PLIES as ATTACK_PLIES, counts_for, outcome as attack_outcome, sacrifice_attrs
from .frames import FrameMemory, material, observe

ATTACK_ATTRS = ["capturer", "victim", "recapturer", "investment", "gives_check", "king_dist", "king_zone_attackers",
                "king_shield", "material", "phase", "elo", "elo_diff"]
from .rules import KINDS, RuleBase, add_rule, bits, bucket, record_root

STRIDE, RESIDUE = 97, 7
SHARD = "records_00000.parquet"


def data_root() -> Path:
    return Path(os.environ.get("CATSPACE_WATCH_DATA") or paths.derived("watch"))


def games(start: int, n: int) -> list[dict]:
    """Games start..start+n of the watched stream."""
    import pyarrow.parquet as pq
    t = pq.read_table(paths.REPO_ROOT / "data" / "records" / "lichess_2019-01" / SHARD,
                      columns=["game_id", "white_elo", "black_elo", "result", "termination", "moves"])
    out, k = [], 0
    for row in t.to_pylist():
        if row["game_id"] % STRIDE != RESIDUE:
            continue
        if k >= start + n:
            break
        if k >= start:
            out.append(row)
        k += 1
    return out


def mate_in_one(b: chess.Board) -> bool:
    """The side to move has a mating move."""
    for m in b.legal_moves:
        b.push(m)
        mate = b.is_checkmate()
        b.pop()
        if mate:
            return True
    return False


def promotes_next(b: chess.Board, side: bool) -> bool:
    """`side` has a pawn one step from promotion with the square in front of it empty (a capture-promotion aside)."""
    rank, step = (6, 8) if side == chess.WHITE else (1, -8)
    return any(b.piece_at(sq + step) is None for sq in b.pieces(chess.PAWN, side) if chess.square_rank(sq) == rank)


def position_attrs(b: chess.Board, mover: bool, frames: list[int], elo: tuple[int, int]) -> dict:
    """Position attributes for 'elsewhere' and 'result' rules."""
    nonpawn = sum(v for p, v in ((p, {1: 0, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}[p.piece_type]) for p in b.piece_map().values()))
    me, them = (elo[0], elo[1]) if mover == chess.WHITE else (elo[1], elo[0])
    return dict(material=material(b, mover), n_frames=len(frames), in_check=b.is_check(), phase=nonpawn,
                elo_diff=(me or 0) - (them or 0), mover_elo=me or 0, ply=len(b.move_stack),
                mate1_tomove=mate_in_one(b),                                     # the side to move mates now
                promo_tomove=any(m.promotion for m in b.legal_moves),           # the side to move can promote now
                promo_mover=promotes_next(b, mover))                             # the mover can promote next move


def mover_result(result: int, mover: bool) -> str:
    """result: 1 White won, -1 Black won, 0 draw (the records' encoding)."""
    if result == 0:
        return "draw"
    return "win" if (result == 1) == (mover == chess.WHITE) else "loss"


def convolve(dists: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Sum of independent integer contributions, each over -10..10: -> (probabilities, offset of index 0)."""
    p, off = np.array([1.0]), 0
    for d in dists:
        p = np.convolve(p, d)
        off -= 10
    return p, off


def watch_game(g: dict, rb: RuleBase, policy: str = "evidence_blend") -> list[dict]:
    """Forecast, reveal, count, for every position of one game. -> one case per position."""
    moves = [chess.Move.from_uci(u) for u in g["moves"].split()]
    boards = [chess.Board()]
    for m in moves:
        nb = boards[-1].copy()
        nb.push(m)
        boards.append(nb)
    censored = g["termination"] == "Time forfeit"
    mem = FrameMemory()                                                      # the game, a move at a time (Kaveh 2026-10-05)
    pending: list[tuple] = []                                                # (reveal at, kind, matched rules, outcome, attributes)
    results: list[tuple[list, str]] = []
    cases = []
    for t in range(1, len(moves)):
        for item in [x for x in pending if x[0] <= t]:                       # outcomes revealed by now
            rb.update(item[2], item[3], item[4])
            pending.remove(item)
        b, mover = boards[t], not boards[t].turn
        frame_attrs = mem.step(boards[t - 1], moves[t - 1], b, t)
        fr = sorted(frame_attrs)
        obs = observe(boards, moves, t)
        dists, frames_out = [], []
        for sq in fr:
            a = frame_attrs[sq]
            d, rid, matched, how = rb.forecast("frame", a, policy)
            d = np.array(d)
            x = bucket("frame", obs["by_square"].get(sq, 0))
            dists.append(d)
            top = sorted(range(21), key=lambda i: -d[i])[:3]
            frames_out.append(dict(sq=chess.square_name(sq), attrs=a, rule=rid, matched=matched, outcome=x,
                                   p=float(d[x + 10]), p_all={m: rb.dist(m, a)[x + 10] for m in matched},
                                   top=[(i - 10, float(d[i])) for i in top], blend=how))   # the forecast as made
            pending.append((obs["end"], "frame", matched, x, a))
        pa = position_attrs(b, mover, fr, (g["white_elo"], g["black_elo"]))
        d_e, rid_e, matched_e, _ = rb.forecast("elsewhere", pa, policy)
        d_e = np.array(d_e)
        x_e = bucket("elsewhere", sum(v for s, v in obs["by_square"].items() if s not in fr))
        dists.append(d_e)
        pending.append((obs["end"], "elsewhere", matched_e, x_e, pa))
        total, off = convolve(dists)
        mean_delta = float(sum((i + off) * p for i, p in enumerate(total)))
        pa["settled"] = int(round(pa["material"] + mean_delta))             # the material once the forecast exchanges settle
        last = moves[t - 1]
        pa["exchange_on"] = boards[t - 1].is_capture(last) and any(m.to_square == last.to_square and b.is_capture(m) for m in b.legal_moves)
        x_total = sum(f["outcome"] for f in frames_out) + x_e
        short_bits = bits(float(total[x_total - off]))
        case = dict(game=g["game_id"], ply=t, fen=b.fen(), move=moves[t - 1].uci(), mover="white" if mover else "black",
                    frames=frames_out, elsewhere=dict(rule=rid_e, outcome=x_e, p=float(d_e[x_e + 10]),
                                                      p_all={m: rb.dist(m, pa)[x_e + 10] for m in matched_e}),
                    delta=x_total, p_delta=float(total[x_total - off]), short_bits=short_bits, end=obs["end"])
        if not censored:
            d_r, rid_r, matched_r, _ = rb.forecast("result", pa, policy)
            res = mover_result(g["result"], mover)
            p_r = d_r[KINDS["result"].index(res)]
            case.update(result=res, result_rule=rid_r, p_result=p_r, long_bits=bits(p_r), pattrs=pa,
                        result_dist=dict(zip(KINDS["result"], d_r)),
                        p_result_all={m: rb.dist(m, pa)[KINDS["result"].index(res)] for m in matched_r})
            results.append((matched_r, res))
        sac = sacrifice_attrs(boards, t - 1, moves[t - 1], g)                 # was the move just played a sacrifice?
        if sac is not None:
            sa = {k: sac[k] for k in ATTACK_ATTRS}
            d_a, rid_a, matched_a, how_a = rb.forecast("attack", sa, policy)
            res_a = "pays" if attack_outcome(boards, t - 1, mover)["success"] else "fails"   # for the side that sacrificed
            p_a = d_a[KINDS["attack"].index(res_a)]
            case["attack"] = dict(rule=rid_a, p_pays=d_a[1], outcome=res_a, p=p_a, bits=bits(p_a), attrs=sa, blend=how_a,
                                  p_all={m: rb.dist(m, sa)[KINDS["attack"].index(res_a)] for m in matched_a})
            pending.append((min(len(moves), t + ATTACK_PLIES + 1), "attack", matched_a, res_a, sa))
        case.setdefault("pattrs", pa)
        cases.append(case)
    for item in pending:                                                     # the game is over: every outcome is known
        rb.update(item[2], item[3], item[4])
    seen: dict = {}                                                          # a game's result is one observation: each rule
    for matched_r, _ in results:                                             # gets a total weight of 1 from the game, shared by
        for rid in matched_r:                                                # the positions it saw (each position used to count
            seen[rid] = seen.get(rid, 0) + 1                                 # as a whole observation: a long game outweighed all)
    for matched_r, res in results:
        rb.update(matched_r, res, None, {rid: 1.0 / seen[rid] for rid in matched_r})
    return cases


def per_game_mean(cases: list[dict], key: str) -> float:
    """The mean over GAMES of each game's mean (Kaveh 2026-10-05: a game's result counts once, in the score too)."""
    by: dict = {}
    for c in cases:
        by.setdefault(c["game"], []).append(c[key])
    return float(np.mean([np.mean(v) for v in by.values()]))


def backfill(rb: RuleBase, ids: set, upto_batch: int) -> dict:
    """Counts for rules that have none yet, from the stored past cases their conditions cover (Kaveh 2026-10-05: a new
    rule starts from the evidence that motivated it). Results count once per game. A case that lacks an attribute
    the condition reads (it predates the word) is skipped. -> {rule: [cases used, cases skipped]}."""
    from .rules import ROOTS
    ids = {i for i in ids if i not in ROOTS.values()}
    if not ids or upto_batch == 0:
        return {}
    for i in ids:
        rb.counts[i] = [0.0] * len(rb.counts[i])
    used = {i: [0, 0] for i in ids}
    for k in range(upto_batch):
        f = data_root() / f"batch_{k:03d}.json"
        if not f.exists():
            continue
        cases = json.loads(f.read_text())["cases"]
        per_game: dict = {}
        for c in cases:
            for fr in c["frames"]:
                _fill(rb, ids, "frame", fr["attrs"], fr["outcome"], 1.0, used)
            if "pattrs" in c:
                _fill(rb, ids, "elsewhere", c["pattrs"], c["elsewhere"]["outcome"], 1.0, used)
            if "result" in c:                                               # scored positions only (time forfeits are censored)
                per_game.setdefault(c["game"], []).append(c)
        for g, cs in per_game.items():                                     # results: each game once, shared by its positions
            hits: dict = {}
            for c in cs:
                for i in ids:
                    if rb.rules[i].kind == "result" and _applies(rb, i, c["pattrs"]):
                        hits[i] = hits.get(i, 0) + 1
            for i in hits:
                used[i][0] += 1                                             # one game
            for c in cs:
                for i, n in hits.items():
                    if _applies(rb, i, c["pattrs"]):
                        rb.counts[i][KINDS["result"].index(c["result"])] += 1.0 / n
    for i, (u, sk) in used.items():
        unit = "games" if rb.rules[i].kind == "result" else "cases"        # the result branch used to leave this at 0
        print(f"OBS backfill {i}: {u} past {unit} counted, {sk} skipped (they predate a word the rule reads)")
    return used


def backfill_attack(rb: RuleBase, ids: set) -> dict:
    """Attack rules (the root included) start from the indexed sacrifices their condition chain covers (watch/cases.py:
    the Lichess records without the watcher's stream), counted by SQL. -> {rule: [cases, 0]}."""
    out = {}
    for i in ids:
        conds, r = [], rb.rules[i]
        while r is not None and r.parent is not None:
            conds.append(r.cond)
            r = rb.rules[r.parent]
        f, s = counts_for(conds)
        rb.counts[i] = [float(f), float(s)]
        out[i] = [f + s, 0]
        print(f"OBS backfill {i}: {f + s} indexed sacrifices counted ({100 * s / max(f + s, 1):.0f}% paid)")
    return out


def _applies(rb: RuleBase, rid: str, a: dict) -> bool:
    """rid's condition and its ancestors' hold on a (False when a lacks a word they read)."""
    from .rules import holds
    r = rb.rules[rid]
    try:
        while r is not None:
            if r.parent is not None and not holds(r.cond, a):
                return False
            r = rb.rules.get(r.parent) if r.parent else None
        return True
    except KeyError:
        return False


def _fill(rb, ids, kind, a, outcome, w, used) -> None:
    from .rules import holds
    for i in ids:
        if rb.rules[i].kind != kind:
            continue
        try:
            ok = _applies_strict(rb, i, a)
        except KeyError:
            used[i][1] += 1
            continue
        if ok:
            rb.update([i], outcome, a, w)
            used[i][0] += 1


def _applies_strict(rb: RuleBase, rid: str, a: dict) -> bool:
    from .rules import holds
    r = rb.rules[rid]
    while r is not None and r.parent is not None:
        if not holds(r.cond, a):
            return False
        r = rb.rules[r.parent]
    return True


def load_state() -> dict:
    p = data_root() / "state.json"
    return json.loads(p.read_text()) if p.exists() else dict(offset=0, batches=[], counts={})


def save_state(st: dict) -> None:
    p = data_root() / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    tmp.replace(p)


def run_batch(n_games: int = 20, policy: str = "evidence_blend", upto: int | None = None, propose_after: bool = True) -> dict:
    """The next batch of the stream with the rules created before `upto` (default: every rule in the file)."""
    st = load_state()
    fresh = {rid for rid in RuleBase.load(upto=upto).rules if rid not in st["counts"]}
    rb = RuleBase.load(counts=st["counts"], upto=upto)
    upto = max(r.created for r in rb.rules.values()) + 1 if upto is None else upto
    filled = backfill(rb, {i for i in fresh if rb.rules[i].kind != "attack"}, len(st["batches"]))   # a new rule starts
    filled.update(backfill_attack(rb, {i for i in fresh if rb.rules[i].kind == "attack"}))       # from the past it covers
    t0 = time.perf_counter()
    gs = games(st["offset"], n_games)
    cases = []
    for g in gs:
        cases += watch_game(g, rb, policy)
    k = len(st["batches"])
    scored = [c for c in cases if "long_bits" in c]
    summary = dict(batch=k, games=len(gs), first_game=st["offset"], positions=len(cases), rules=len(rb.rules), rules_upto=upto,
                   backfilled=filled,
                   short_bits=float(np.mean([c["short_bits"] for c in cases])),
                   frame_bits=float(np.mean([-np.log2(max(f["p"], 1e-12)) for c in cases for f in c["frames"]])) if any(c["frames"] for c in cases) else 0.0,
                   frames=sum(len(c["frames"]) for c in cases),
                   sacrifices=sum(1 for c in cases if "attack" in c),
                   attack_bits=float(np.mean([c["attack"]["bits"] for c in cases if "attack" in c])) if any("attack" in c for c in cases) else None,
                   long_bits=per_game_mean(scored, "long_bits") if scored else None, long_positions=len(scored),
                   seconds=time.perf_counter() - t0, time=time.strftime("%Y-%m-%d %H:%M %Z"))
    st["offset"] += len(gs)
    st["counts"] = rb.counts
    st["batches"].append(summary)
    out = data_root() / f"batch_{k:03d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, cases=cases)))
    save_state(st)
    rec = record_root() / "batches.jsonl"
    rec.parent.mkdir(parents=True, exist_ok=True)
    with open(rec, "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    if propose_after and any("attack" in c for c in cases):                # the watcher proposes rules from its surprises
        from .cases import db_path as _cases_db
        if _cases_db().exists():
            from .propose import run as _propose
            _propose(k)
    print(f"OBS watch batch {k}: {summary['games']} games from #{summary['first_game']}, {summary['positions']} positions, "
          f"{summary['frames']} frames, {summary['rules']} rules; surprise per position: short {summary['short_bits']:.3f} bits, "
          f"per frame {summary['frame_bits']:.3f}, long {summary['long_bits'] if summary['long_bits'] is None else round(summary['long_bits'], 3)} bits "
          f"({summary['long_positions']} positions not censored); {summary['seconds']:.0f}s")
    return summary


def replay() -> list[dict]:
    """Rebuild the state from the start: re-run every recorded batch (its games, the rules that existed when it began)
    with today's code. Used after a fix to the counting; the batch files and summaries are rewritten."""
    old = [json.loads(l) for l in (record_root() / "batches.jsonl").read_text().splitlines() if l.strip()]
    st = dict(offset=0, batches=[], counts={})
    save_state(st)
    (record_root() / "batches.jsonl").write_text("")
    out = []
    for b in old:
        out.append(run_batch(b["games"], upto=b["rules_upto"], propose_after=False))
    return out


def health() -> list[dict]:
    """The loop's health line: per batch, the bits per position the rule base saved against the base rates (the roots)
    on the SAME positions, by horizon -- unlike the raw averages, which move with the games a batch happens to hold."""
    import math
    L = lambda p: -math.log2(max(p, 1e-12))
    out = []
    for k in range(len(load_state()["batches"])):
        d = json.loads((data_root() / f"batch_{k:03d}.json").read_text())
        cs = d["cases"]
        fr = sum(L(f["p_all"]["F0"]) - L(f["p"]) for c in cs for f in c["frames"])
        el = sum(L(c["elsewhere"]["p_all"]["E0"]) - L(c["elsewhere"]["p"]) for c in cs if "p_all" in c["elsewhere"])
        lg = [dict(game=c["game"], s=L(c["p_result_all"]["R0"]) - L(c["p_result"])) for c in cs if "p_result_all" in c]
        at = [L(c["attack"]["p_all"]["A0"]) - L(c["attack"]["p"]) for c in cs if "attack" in c]
        h = dict(batch=k, positions=len(cs), rules=d["summary"]["rules"], frames_saved=fr / len(cs), elsewhere_saved=el / len(cs),
                 long_saved=per_game_mean(lg, "s") if lg else 0.0,          # a game's result counts once (Kaveh 2026-10-05)
                 attack_saved=(sum(at) / len(at)) if at else 0.0, sacrifices=len(at))
        out.append(h)
        print(f"OBS health batch {k}: {h['rules']} rules; bits saved per position against the base rates: frames {h['frames_saved']:+.3f}, "
              f"elsewhere {h['elsewhere_saved']:+.3f}, result per game {h['long_saved']:+.3f} ({len({x['game'] for x in lg})} games), "
              f"attack per sacrifice {h['attack_saved']:+.3f} ({h['sacrifices']})")
    return out


def credit(batch: int | None = None) -> dict:
    """Each rule's credit on a batch: on the cases it decided, the bits it saved against its kind's root and against
    its parent (every applicable rule's probability of what happened is in the batch file)."""
    import math
    from collections import defaultdict
    st = load_state()
    k = len(st["batches"]) - 1 if batch is None else batch
    d = json.loads((data_root() / f"batch_{k:03d}.json").read_text())
    rb = RuleBase.load()
    L = lambda p: -math.log2(max(p, 1e-12))
    cred = defaultdict(lambda: [0, 0.0, 0.0])

    def add(r, pa, p, root):
        par = rb.rules[r].parent
        x = cred[r]
        x[0] += 1
        x[1] += L(pa[root]) - L(p)
        x[2] += (L(pa[par]) - L(p)) if par else 0.0
    for c in d["cases"]:
        for f in c["frames"]:
            add(f["rule"], f["p_all"], f["p"], "F0")
        if "p_all" in c["elsewhere"]:
            add(c["elsewhere"]["rule"], c["elsewhere"]["p_all"], c["elsewhere"]["p"], "E0")
        if "p_result_all" in c:
            add(c["result_rule"], c["p_result_all"], c["p_result"], "R0")
        if "attack" in c:
            add(c["attack"]["rule"], c["attack"]["p_all"], c["attack"]["p"], "A0")
    for r, (n, vr, vp) in sorted(cred.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]))):
        if rb.rules[r].parent is None:
            print(f"OBS batch {k} {r} (root) decided {n}")
        else:
            print(f"OBS batch {k} {r} decided {n}: bits saved vs the root {vr:+.1f}, vs its parent {vp:+.1f} | {rb.rules[r].text}")
    return {r: dict(cases=n, vs_root=vr, vs_parent=vp) for r, (n, vr, vp) in cred.items()}


def explain(case_id: str, batch: int | None = None) -> None:
    """One position as the watcher saw it: the board, the move just played, every frame with the rule that forecast it,
    its likeliest outcomes as forecast then, and what the game did; the off-frame term; the result forecast."""
    st = load_state()
    ks = range(len(st["batches"])) if batch is None else [batch]
    gid, ply = (int(x) for x in case_id.split(":")[:2])
    rb = RuleBase.load()
    for k in ks:
        for c in json.loads((data_root() / f"batch_{k:03d}.json").read_text())["cases"]:
            if c["game"] != gid or c["ply"] != ply:
                continue
            g = games_by_id(gid)
            moves = g["moves"].split()
            b = chess.Board()
            for u in moves[:ply - 1]:
                b.push_uci(u)
            san = b.san(chess.Move.from_uci(moves[ply - 1]))
            b.push_uci(moves[ply - 1])
            print(f"game {gid} (White {g['white_elo']}, Black {g['black_elo']}), after {ply} plies: {c['mover']} just played {san}\n")
            print(b.unicode(borders=False, empty_square="."), "\n")
            print("then:", b.variation_san([chess.Move.from_uci(u) for u in moves[ply:c["end"] + 1]]) or "(quiet)", "\n")
            for f in c["frames"]:
                r = rb.rules[f["rule"]]
                fc = ", ".join(f"{o:+d} {p:.0%}" for o, p in f.get("top", []))
                print(f"frame {f['sq']} ({'mover' if f['attrs']['occ_mover'] else 'opponent'}'s {f['attrs']['occ']}): rule {r.id} \"{r.text}\"")
                print(f"    forecast {fc}; happened {f['outcome']:+d} (probability given {f['p']:.0%}, {bits(f['p']):.2f} bits)")
            e = c["elsewhere"]
            print(f"off the frames: rule {e['rule']} \"{rb.rules[e['rule']].text}\"; happened {e['outcome']:+d} (probability given {e['p']:.0%})")
            print(f"material change for the mover: {c['delta']:+d}, probability given {c['p_delta']:.0%} ({c['short_bits']:.2f} bits)")
            if "result_dist" in c:
                rd = ", ".join(f"{k2} {v:.0%}" for k2, v in c["result_dist"].items())
                print(f"result for the mover: rule {c['result_rule']} \"{rb.rules[c['result_rule']].text}\": {rd}; it was a {c['result']}")
            return
    print(f"no case {case_id}")


def export(batches: list[int], out: str) -> dict:
    """The viewer's data (watch/viewer.html): for every game of the given batches, its header and moves, and for every
    position what the watcher saw, forecast and was surprised by -- read from the batch records, as they were made."""
    rb = RuleBase.load(counts=load_state()["counts"])                         # with their counts (an after-the-fact attack
    games_out = []                                                           # forecast read uniform without them, 2026-10-05)
    for k in batches:
        d = json.loads((data_root() / f"batch_{k:03d}.json").read_text())
        by_game: dict = {}
        for c in d["cases"]:
            by_game.setdefault(c["game"], []).append(c)
        for gid, cs in by_game.items():
            g = games_by_id(gid)
            ucis = g["moves"].split()
            b = chess.Board()
            sans, fens = [], [b.fen()]
            for u in ucis:
                m = chess.Move.from_uci(u)
                sans.append(b.san(m))
                b.push(m)
                fens.append(b.fen())
            plies = []
            bl = [chess.Board()]
            for u in ucis:
                nb = bl[-1].copy(stack=False); nb.push_uci(u); bl.append(nb)
            for c in sorted(cs, key=lambda c: c["ply"]):
                t0 = c["ply"]
                fr = [dict(sq=f["sq"], occ=f["attrs"]["occ"], own=f["attrs"]["occ_mover"], rule=f["rule"],
                           top=[[o, round(p, 3)] for o, p in f.get("top", [])], out=f["outcome"], p=round(f["p"], 3),
                           age=f["attrs"].get("age", 0), dec=f["attrs"].get("declined", 0),
                           w=round(f.get("blend", {}).get("weight", 1.0), 2), sup=f.get("blend", {}).get("superseded"))
                      for f in c["frames"]]
                e = c["elsewhere"]
                plies.append(dict(t=t0, mover=c["mover"][0], frames=fr, el=dict(rule=e["rule"], out=e["outcome"], p=round(e["p"], 3)),
                                  delta=c["delta"], pd=round(c["p_delta"], 3), sb=round(c["short_bits"], 2), end=c["end"],
                                  res=c.get("result"), rr=c.get("result_rule"), lb=round(c["long_bits"], 2) if "long_bits" in c else None,
                                  rd=[round(c["result_dist"][x], 3) for x in ("loss", "draw", "win")] if "result_dist" in c else None,
                                  attack=_attack_view(c, bl, ucis, g, rb)))
            games_out.append(dict(id=gid, batch=k, we=g["white_elo"], be=g["black_elo"], result=g["result"], term=g["termination"],
                                  san=sans, uci=ucis, fen=fens, plies=plies))
    rules = {rid: dict(kind=r.kind, parent=r.parent, text=r.text) for rid, r in rb.rules.items()}
    data = dict(made=time.strftime("%Y-%m-%d %H:%M %Z"), games=games_out, rules=rules)
    Path(out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"OBS export: {len(games_out)} games, {sum(len(g['plies']) for g in games_out)} positions, {Path(out).stat().st_size // 1024} KB -> {out}")
    return data


def _attack_view(c: dict, boards: list, ucis: list, g: dict, rb: RuleBase) -> dict | None:
    """The viewer's attack card for a position just after a sacrifice: the watcher's forecast (as made, or -- for a
    batch watched before the attack rules existed -- by today's rules, marked so) and why: the nearest indexed cases."""
    from .cases import db_path as _cdb, why
    t = c["ply"]
    sac = sacrifice_attrs(boards, t - 1, chess.Move.from_uci(ucis[t - 1]), g)
    if sac is None:
        return None
    if "attack" in c:
        a = c["attack"]
        view = dict(rule=a["rule"], p_pays=round(a["p_pays"], 3), outcome=a["outcome"], bits=round(a["bits"], 2), after=False)
    else:
        sa = {k: sac[k] for k in ATTACK_ATTRS}
        d, rid, _, _ = rb.forecast("attack", sa)
        res = "pays" if attack_outcome(boards, t - 1, boards[t - 1].turn)["success"] else "fails"
        p = d[KINDS["attack"].index(res)]
        view = dict(rule=rid, p_pays=round(d[1], 3), outcome=res, bits=round(bits(p), 2), after=True)
    view.update(capturer=sac["capturer"], victim=sac["victim"], recapturer=sac["recapturer"], investment=sac["investment"],
                king_dist=sac["king_dist"], zone=sac["king_zone_attackers"], elo=sac["elo"])
    if _cdb().exists():
        w = why(dict(sac, kind="sacrifice", game=-1, ply=-1))
        view["why"] = dict(n=w.get("n", 0), rate=round(w.get("rate", 0.0), 3),
                           leaves=[dict(l, rate=round(l["rate"], 3)) for l in w.get("leaves", [])])
    return view


def show(batch: int | None = None, top: int = 20) -> None:
    """The batch's largest surprises, with what Claude needs to backtrack: the position, the move, every frame's
    forecast and outcome, and the game's continuation until the window closed."""
    st = load_state()
    k = len(st["batches"]) - 1 if batch is None else batch
    d = json.loads((data_root() / f"batch_{k:03d}.json").read_text())
    cases = d["cases"]
    rb = RuleBase.load(counts=st["counts"])
    print(f"batch {k}: {d['summary']}")
    for c in sorted(cases, key=lambda c: -c["short_bits"])[:top]:
        g = games_by_id(c["game"])
        moves = g["moves"].split()
        b = chess.Board()
        for u in moves[:c["ply"]]:
            b.push_uci(u)
        cont = chess.Board(c["fen"]).variation_san([chess.Move.from_uci(u) for u in moves[c["ply"]:c["end"] + 2]])
        print(f"\n{c['short_bits']:.2f} bits  game {c['game']} ply {c['ply']} ({c['mover']} just played {c['move']}): delta {c['delta']:+d}, "
              f"P {c['p_delta']:.3f}\n  {c['fen']}\n  then: {cont}")
        for f in c["frames"]:
            a = f["attrs"]
            print(f"  frame {f['sq']}: {'our' if a['occ_mover'] else 'their'} {a['occ']} att {a['n_att']} (min {a['min_att']}) def {a['n_def']} "
                  f"(min {a['min_def']}) see {a['see_att']}{' to-move' if a['attacker_to_move'] else ''}{' pinned' if a['pinned'] else ''}"
                  f" | rule {f['rule']} P(outcome {f['outcome']:+d}) = {f['p']:.3f}")
        e = c["elsewhere"]
        print(f"  elsewhere: outcome {e['outcome']:+d}, rule {e['rule']}, P {e['p']:.3f}")


_GAMES: dict = {}


def games_by_id(gid: int) -> dict:
    if not _GAMES:
        for g in games(0, load_state()["offset"]):
            _GAMES[g["game_id"]] = g
    return _GAMES[gid]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["batch", "show", "add", "replay", "credit", "health", "explain", "export", "retire"])
    ap.add_argument("--rule", default=None, help="retire: the rule to retire")
    ap.add_argument("--note", default="", help="retire: why")
    ap.add_argument("--batches", type=int, nargs="*", default=None, help="export: the batches to include (default: the last two)")
    ap.add_argument("--out", default=None, help="export: the JSON file to write")
    ap.add_argument("--case", default=None, help="explain: game:ply")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--kind", default="frame")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--cond", default=None)
    ap.add_argument("--text", default="")
    ap.add_argument("--cornerstone", action="append", default=[])
    ap.add_argument("--ref", default=None, help='a reference: {"attr": "see_mover"} (the rule counts offsets from it)')
    a = ap.parse_args(argv)
    if a.cmd == "batch":
        run_batch(a.games)
    elif a.cmd == "show":
        show(a.batch, a.top)
    elif a.cmd == "replay":
        replay()
    elif a.cmd == "credit":
        credit(a.batch)
    elif a.cmd == "health":
        health()
    elif a.cmd == "explain":
        explain(a.case, a.batch)
    elif a.cmd == "retire":
        from .rules import retire
        retire(a.rule, a.note)
        print(f"retired {a.rule}: {a.note}")
    elif a.cmd == "export":
        n = len(load_state()["batches"])
        export(a.batches if a.batches else list(range(max(0, n - 2), n)), a.out or str(data_root() / "viewer_games.json"))
    else:
        r = add_rule(a.kind, a.parent, json.loads(a.cond), a.text, a.cornerstone, ref=json.loads(a.ref) if a.ref else None)
        print(f"added {r.id} ({r.kind}, parent {r.parent}): {r.text}")


if __name__ == "__main__":
    main()
